from contextlib import asynccontextmanager
import gc
import logging
import os
import threading
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import onnxruntime as ort
from onnx_provider_utils import choose_execution_providers, provider_chain_to_string
from pydantic import BaseModel
from transformers import AutoTokenizer


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

embedding_tokenizer: Optional[AutoTokenizer] = None
reranker_tokenizer: Optional[AutoTokenizer] = None
embedding_session: Optional[ort.InferenceSession] = None
reranker_session: Optional[ort.InferenceSession] = None
runtime_provider: Optional[str] = None
embedding_batch_size = 1


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


def _ensure_ready():
    if embedding_tokenizer is None or reranker_tokenizer is None:
        raise RuntimeError("Tokenizers are not loaded")
    if embedding_session is None or reranker_session is None:
        raise RuntimeError("ONNX sessions are not loaded")


def _format_instruction(instruction, query, document):
    rerank_instruction = instruction or DEFAULT_RERANK_INSTRUCTION
    return (
        f"{RERANK_PREFIX}<Instruct>: {rerank_instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}{RERANK_SUFFIX}"
    )


def _build_embedding_inputs(texts):
    if embedding_tokenizer is None or embedding_session is None:
        raise RuntimeError("Embedding model not loaded")

    encoded = embedding_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="np",
        max_length=EMBED_MAX_LENGTH,
    )
    model_inputs = {item.name for item in embedding_session.get_inputs()}
    input_feed = {
        key: value.astype(np.int64)
        for key, value in encoded.items()
        if key in model_inputs
    }
    return encoded, input_feed


def _run_embedding_batch(texts):
    if embedding_tokenizer is None or embedding_session is None:
        raise RuntimeError("Embedding model not loaded")

    if not texts:
        return np.empty((0, 0), dtype=np.float32), [], 0

    embeddings = []
    text_lengths = []
    total_tokens = 0
    for start in range(0, len(texts), embedding_batch_size):
        chunk_texts = texts[start:start + embedding_batch_size]
        encoded, input_feed = _build_embedding_inputs(chunk_texts)
        with _inference_semaphore:
            outputs = embedding_session.run(None, input_feed)
        last_hidden_state = np.asarray(outputs[0])
        del outputs
        pooled = _last_token_pool(last_hidden_state, encoded["attention_mask"])
        del last_hidden_state
        normalized = _normalize_embeddings(pooled).astype(np.float32)
        del pooled
        embeddings.extend(normalized)
        text_lengths.extend(
            len(embedding_tokenizer.encode(text, add_special_tokens=False))
            for text in chunk_texts
        )
        total_tokens += int(np.asarray(encoded["attention_mask"]).sum())
        del input_feed

    return np.stack(embeddings, axis=0), text_lengths, total_tokens


def _run_rerank_batch(pairs):
    """对所有 pairs 分批推理，每批推理后立即释放中间张量，避免显存峰值过高。"""
    if reranker_session is None:
        raise RuntimeError("Reranker model not loaded")

    if not pairs:
        return np.array([], dtype=np.float32)

    all_logits = []
    for start in range(0, len(pairs), RERANK_BATCH_SIZE):
        chunk = pairs[start:start + RERANK_BATCH_SIZE]
        _, input_feed = _build_rerank_inputs(chunk)
        with _inference_semaphore:
            outputs = reranker_session.run(None, input_feed)
        logits = np.asarray(outputs[0]).reshape(-1)
        del outputs, input_feed
        all_logits.append(logits)

    return np.concatenate(all_logits)


def _build_rerank_inputs(pairs):
    if reranker_tokenizer is None or reranker_session is None:
        raise RuntimeError("Reranker model not loaded")

    encoded = reranker_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="np",
        max_length=RERANK_MAX_LENGTH,
    )
    model_inputs = {item.name for item in reranker_session.get_inputs()}
    input_feed = {
        key: value.astype(np.int64)
        for key, value in encoded.items()
        if key in model_inputs
    }
    return encoded, input_feed


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_tokenizer, reranker_tokenizer
    global embedding_session, reranker_session, runtime_provider, embedding_batch_size

    LOGGER.info("Starting up: loading ONNX models...")
    try:
        embedding_tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_DIR,
            model_max_length=EMBED_MAX_LENGTH,
            padding_side="left",
            trust_remote_code=True,
        )
        reranker_tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_DIR,
            model_max_length=RERANK_MAX_LENGTH,
            padding_side="left",
            trust_remote_code=True,
        )
        embedding_session = _create_session(EMBEDDING_MODEL_PATH)
        reranker_session = _create_session(RERANKER_MODEL_PATH)
        embedding_batch_size = _resolve_embedding_batch_size(
            embedding_session,
            embedding_tokenizer,
            EMBED_BATCH_SIZE,
        )
        runtime_provider = embedding_session.get_providers()[0]
        LOGGER.info("Embedding providers: %s", embedding_session.get_providers())
        LOGGER.info("Reranker providers: %s", reranker_session.get_providers())
        LOGGER.info("Embedding effective batch size: %s", embedding_batch_size)
    except Exception as exc:
        LOGGER.exception("Failed to load ONNX models")
        raise RuntimeError(f"Failed to load ONNX models: {exc}") from exc

    yield

    del embedding_session
    del reranker_session
    del embedding_tokenizer
    del reranker_tokenizer
    gc.collect()
    LOGGER.info("Shutdown complete, GPU resources released.")


app = FastAPI(
    title="Qwen3 Text Service",
    description="A FastAPI service for text embeddings and reranking using ONNX Qwen3 models",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

embedding_model_name = _resolve_model_name(EMBEDDING_DIR, "qwen3-embedding-0.6b-onnx")
reranker_model_name = _resolve_model_name(RERANKER_DIR, "qwen3-reranker-seq-cls-onnx")


class EmbedTextRequest(BaseModel):
    input: List[str]


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    instruction: Optional[str] = None


class SplitTextRequest(BaseModel):
    text: str
    chunksize: int
    overlap_size: int = 0


class SplitTextResponse(BaseModel):
    chunks: List[str]
    total_chunks: int
    model: str = embedding_model_name


class RerankResult(BaseModel):
    document: str
    score: float
    index: int


class RerankResponse(BaseModel):
    results: List[RerankResult]
    model: str = reranker_model_name


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int
    tokens_length: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbedTextResponse(BaseModel):
    data: List[EmbeddingData]
    model: str = embedding_model_name
    object: str = "list"
    usage: Usage


@app.post("/embed_text", response_model=EmbedTextResponse, tags=["Embedding"])
async def embed_text(request: EmbedTextRequest):
    try:
        _ensure_ready()
        embeddings, text_lengths, total_tokens = _run_embedding_batch(request.input)

        if embedding_tokenizer is None:
            raise RuntimeError("Embedding tokenizer not loaded")

        data = [
            EmbeddingData(
                embedding=embedding,
                index=index,
                tokens_length=text_length,
            )
            for index, (embedding, text_length) in enumerate(zip(embeddings.tolist(), text_lengths))
        ]
        usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)
        return EmbedTextResponse(data=data, usage=usage)
    except Exception as exc:
        LOGGER.exception("Error during embedding")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rerank", response_model=RerankResponse, tags=["Reranking"])
async def simple_rerank(request: RerankRequest):
    try:
        _ensure_ready()
        pairs = [
            _format_instruction(request.instruction, request.query, doc)
            for doc in request.documents
        ]
        logits = _run_rerank_batch(pairs)
        scores = _sigmoid(logits)

        results = [
            RerankResult(document=doc, score=float(score), index=index)
            for index, (doc, score) in enumerate(zip(request.documents, scores.tolist()))
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        return RerankResponse(results=results)
    except Exception as exc:
        LOGGER.exception("Error during reranking")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def SplitText(text: str, chunksize: int, overlap_size: int) -> List[str]:
    if embedding_tokenizer is None:
        raise RuntimeError("Embedding tokenizer not loaded")
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than 0")
    if overlap_size < 0:
        raise ValueError("overlap_size cannot be negative")
    if overlap_size >= chunksize:
        raise ValueError("overlap_size must be smaller than chunksize")

    tokens = embedding_tokenizer.encode(text, add_special_tokens=False)
    result = []
    start = 0
    while start < len(tokens):
        end = min(start + chunksize, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_token_strs = embedding_tokenizer.convert_ids_to_tokens(chunk_tokens)
        chunk_text = embedding_tokenizer.convert_tokens_to_string(chunk_token_strs)
        result.append(chunk_text)
        if end == len(tokens):
            break
        start = end - overlap_size if overlap_size > 0 else end
    return result


@app.post("/split_text", response_model=SplitTextResponse, tags=["Text Processing"])
async def split_text_api(request: SplitTextRequest):
    try:
        chunks = SplitText(request.text, request.chunksize, request.overlap_size)
        return SplitTextResponse(chunks=chunks, total_chunks=len(chunks))
    except Exception as exc:
        LOGGER.exception("Error during text splitting")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url or "/openapi.json",
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"},
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url or "/openapi.json",
        title=app.title + " - ReDoc",
        redoc_js_url="https://unpkg.com/redoc@2.1.0/bundles/redoc.standalone.js",
    )


@app.get("/oauth2-redirect", include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy" if embedding_session is not None and reranker_session is not None else "loading",
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "device": runtime_provider,
        "service": "Qwen3 Text Service",
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)