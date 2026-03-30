import gc
import logging
import os
import threading
import time
from concurrent import futures

import grpc
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from onnx_provider_utils import choose_execution_providers, provider_chain_to_string
import text_service_pb2
import text_service_pb2_grpc


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")

EMBEDDING_DIR_DEFAULT = os.path.join(MODELS_DIR, "qwen3-embedding-0.6b-onnx")
RERANKER_DIR_DEFAULT = os.path.join(MODELS_DIR, "qwen3-reranker-seq-cls-onnx")

EMBEDDING_DIR = os.environ.get("EMBEDDING_MODEL_DIR", EMBEDDING_DIR_DEFAULT)
RERANKER_DIR = os.environ.get("RERANKER_MODEL_DIR", RERANKER_DIR_DEFAULT)
EMBEDDING_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    os.path.join(EMBEDDING_DIR, "model.onnx"),
)
RERANKER_MODEL_PATH = os.environ.get(
    "RERANKER_MODEL_PATH",
    os.path.join(RERANKER_DIR, "model.onnx"),
)

EMBED_MAX_LENGTH = int(os.environ.get("EMBEDDING_MAX_LENGTH", "8192"))
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "8192"))
EMBED_BATCH_SIZE = max(1, int(os.environ.get("EMBEDDING_BATCH_SIZE", "8")))
RERANK_BATCH_SIZE = max(1, int(os.environ.get("RERANK_BATCH_SIZE", "4")))
_MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_INFERENCES", "2"))
_inference_semaphore = threading.Semaphore(_MAX_CONCURRENT)

DEFAULT_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
RERANK_PREFIX = (
    "<|im_start|>system\n"
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

LOGGER = logging.getLogger(__name__)


def _get_execution_providers():
    providers, reason, available = choose_execution_providers()
    LOGGER.info(
        "Selected ONNX providers: %s | reason=%s | available=%s",
        provider_chain_to_string(providers),
        reason,
        available,
    )
    return providers


def _create_session(model_path):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.log_severity_level = 3
    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=_get_execution_providers(),
    )


def _last_token_pool(last_hidden_states, attention_mask):
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be 2-dimensional")

    if np.all(attention_mask[:, -1] == 1):
        return last_hidden_states[:, -1, :]

    sequence_lengths = np.clip(attention_mask.sum(axis=1) - 1, a_min=0, a_max=None)
    batch_indices = np.arange(last_hidden_states.shape[0])
    return last_hidden_states[batch_indices, sequence_lengths, :]


def _normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float32)
    positive = values >= 0
    negative = ~positive
    result = np.empty_like(values, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[negative])
    result[negative] = exp_values / (1.0 + exp_values)
    return result


def _format_instruction(instruction, query, document):
    rerank_instruction = instruction or DEFAULT_RERANK_INSTRUCTION
    return (
        f"{RERANK_PREFIX}<Instruct>: {rerank_instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}{RERANK_SUFFIX}"
    )


def _resolve_model_name(model_dir, fallback_name):
    return os.path.basename(os.path.normpath(model_dir)) or fallback_name


def _resolve_embedding_batch_size(session, tokenizer, requested_batch_size):
    if requested_batch_size <= 1:
        return 1

    probe_texts = ["batch probe text 1", "batch probe text 2"]
    try:
        encoded = tokenizer(
            probe_texts,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=EMBED_MAX_LENGTH,
        )
        model_inputs = {item.name for item in session.get_inputs()}
        input_feed = {
            key: value.astype(np.int64)
            for key, value in encoded.items()
            if key in model_inputs
        }
        session.run(None, input_feed)
        return requested_batch_size
    except Exception as exc:
        LOGGER.warning(
            "Embedding ONNX model does not support true batching on current runtime; "
            "requested_batch_size=%s fallback_batch_size=1 error=%s",
            requested_batch_size,
            exc,
        )
        return 1


class TextGrpcServiceServicer(text_service_pb2_grpc.TextGrpcServiceServicer):
    def __init__(self):
        LOGGER.info("Loading ONNX models for gRPC service...")
        self.embedding_model_name = _resolve_model_name(EMBEDDING_DIR, "qwen3-embedding-0.6b-onnx")
        self.reranker_model_name = _resolve_model_name(RERANKER_DIR, "qwen3-reranker-seq-cls-onnx")

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_DIR,
            model_max_length=EMBED_MAX_LENGTH,
            padding_side="left",
            trust_remote_code=True,
        )
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_DIR,
            model_max_length=RERANK_MAX_LENGTH,
            padding_side="left",
            trust_remote_code=True,
        )

        self.embedding_session = _create_session(EMBEDDING_MODEL_PATH)
        self.reranker_session = _create_session(RERANKER_MODEL_PATH)
        self.embedding_batch_size = _resolve_embedding_batch_size(
            self.embedding_session,
            self.embedding_tokenizer,
            EMBED_BATCH_SIZE,
        )

        LOGGER.info("Embedding session providers: %s", self.embedding_session.get_providers())
        LOGGER.info("Reranker session providers: %s", self.reranker_session.get_providers())
        LOGGER.info("Embedding effective batch size: %s", self.embedding_batch_size)

    def _build_embedding_inputs(self, texts):
        encoded = self.embedding_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=EMBED_MAX_LENGTH,
        )
        model_inputs = {item.name for item in self.embedding_session.get_inputs()}
        input_feed = {
            key: value.astype(np.int64)
            for key, value in encoded.items()
            if key in model_inputs
        }
        return encoded, input_feed

    def _run_embedding_batch(self, texts):
        if not texts:
            return np.empty((0, 0), dtype=np.float32), [], 0

        embeddings = []
        text_lengths = []
        total_tokens = 0

        for start in range(0, len(texts), self.embedding_batch_size):
            chunk_texts = texts[start:start + self.embedding_batch_size]
            encoded, input_feed = self._build_embedding_inputs(chunk_texts)
            with _inference_semaphore:
                outputs = self.embedding_session.run(None, input_feed)
            last_hidden_state = np.asarray(outputs[0])
            del outputs
            pooled = _last_token_pool(last_hidden_state, encoded["attention_mask"])
            del last_hidden_state
            normalized = _normalize_embeddings(pooled).astype(np.float32)
            del pooled
            embeddings.extend(normalized)
            text_lengths.extend(
                len(self.embedding_tokenizer.encode(text, add_special_tokens=False))
                for text in chunk_texts
            )
            total_tokens += int(np.asarray(encoded["attention_mask"]).sum())
            del input_feed

        return np.stack(embeddings, axis=0), text_lengths, total_tokens

    def _run_rerank_batch(self, pairs):
        """对所有 pairs 分批推理，每批推理后立即释放中间张量，避免显存峰值过高。"""
        if not pairs:
            return np.array([], dtype=np.float32)

        all_logits = []
        for start in range(0, len(pairs), RERANK_BATCH_SIZE):
            chunk = pairs[start:start + RERANK_BATCH_SIZE]
            _, input_feed = self._build_rerank_inputs(chunk)
            with _inference_semaphore:
                outputs = self.reranker_session.run(None, input_feed)
            logits = np.asarray(outputs[0]).reshape(-1)
            del outputs, input_feed
            all_logits.append(logits)

        return np.concatenate(all_logits)

    def _build_rerank_inputs(self, pairs):
        encoded = self.reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=RERANK_MAX_LENGTH,
        )
        model_inputs = {item.name for item in self.reranker_session.get_inputs()}
        input_feed = {
            key: value.astype(np.int64)
            for key, value in encoded.items()
            if key in model_inputs
        }
        return encoded, input_feed

    def SimpleRerank(self, request, context):
        query = request.prompt
        documents = list(request.text_blocks)

        try:
            pairs = [_format_instruction(None, query, doc) for doc in documents]
            logits = self._run_rerank_batch(pairs)
            scores = _sigmoid(logits)

            results = []
            for index, (doc, score) in enumerate(zip(documents, scores.tolist())):
                results.append(
                    text_service_pb2.RerankResult(
                        score=float(score),
                        text_block=doc,
                        index=index,
                    )
                )

            results.sort(key=lambda item: item.score, reverse=True)
            return text_service_pb2.SimpleRerankResponse(
                ranked_results=results,
                total_blocks=len(documents),
                query=query,
            )
        except Exception as exc:
            LOGGER.exception("SimpleRerank failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return text_service_pb2.SimpleRerankResponse()

    def EmbedText(self, request, context):
        texts = list(request.input)

        try:
            embeddings, text_lengths, total_tokens = self._run_embedding_batch(texts)

            data = []
            for index, (embedding, text_length) in enumerate(zip(embeddings.tolist(), text_lengths)):
                data.append(
                    text_service_pb2.EmbeddingData(
                        object="embedding",
                        embedding=embedding,
                        index=index,
                        text_length=text_length,
                    )
                )

            usage = text_service_pb2.Usage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            )
            return text_service_pb2.EmbedTextResponse(
                data=data,
                model=self.embedding_model_name,
                object="list",
                usage=usage,
            )
        except Exception as exc:
            LOGGER.exception("EmbedText failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return text_service_pb2.EmbedTextResponse()

    def SplitTextIntoChunks(self, request, context):
        try:
            chunks = SplitText(
                request.input,
                request.chunk_size,
                request.overlap_size,
                self.embedding_tokenizer,
            )
            return text_service_pb2.SplitTextResponse(
                chunks=chunks,
                total_chunks=len(chunks),
                chunk_size=request.chunk_size,
                overlap_size=request.overlap_size,
                average_chunk_length=sum(len(chunk) for chunk in chunks) / max(len(chunks), 1),
            )
        except Exception as exc:
            LOGGER.exception("SplitTextIntoChunks failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return text_service_pb2.SplitTextResponse()

    def HealthCheck(self, request, context):
        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent(interval=0.1)
        return text_service_pb2.HealthCheckResponse(
            status="healthy",
            timestamp=time.time(),
            memory_usage_mb=str(memory_mb),
            cpu_percent=str(cpu_percent),
            pid=os.getpid(),
            models_loaded=text_service_pb2.ModelStatus(
                reranker=self.reranker_session is not None,
                embedding=self.embedding_session is not None,
            ),
        )

    def GetModelInfo(self, request, context):
        return text_service_pb2.ModelInfoResponse(
            reranker_model=text_service_pb2.ModelDetail(
                loaded=self.reranker_session is not None,
                type="reranker",
                model_name=self.reranker_model_name,
            ),
            embedding_model=text_service_pb2.ModelDetail(
                loaded=self.embedding_session is not None,
                type="embedding",
                model_name=self.embedding_model_name,
            ),
        )


def SplitText(text, chunk_size, overlap_size, tokenizer):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap_size < 0:
        raise ValueError("overlap_size cannot be negative")
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be smaller than chunk_size")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_tokens = tokenizer.convert_ids_to_tokens(chunk_ids)
        chunks.append(tokenizer.convert_tokens_to_string(chunk_tokens))
        if end >= len(token_ids):
            break
        start = end - overlap_size if overlap_size > 0 else end
    return chunks


def serve():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    port = os.environ.get("GRPC_PORT", "32688")
    servicer = TextGrpcServiceServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    text_service_pb2_grpc.add_TextGrpcServiceServicer_to_server(
        servicer,
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    LOGGER.info("gRPC server started on port %s", port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        LOGGER.info("Received interrupt, stopping gRPC server...")
        server.stop(grace=5)
    finally:
        servicer.embedding_session = None
        servicer.reranker_session = None
        servicer.embedding_tokenizer = None
        servicer.reranker_tokenizer = None
        gc.collect()
        LOGGER.info("gRPC server stopped, GPU resources released.")


if __name__ == "__main__":
    serve()
