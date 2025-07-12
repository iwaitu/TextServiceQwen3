import grpc
from concurrent import futures
import time
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import os
import text_service_pb2
import text_service_pb2_grpc

# 模型路径
model_name = "Models/Qwen3-Embedding-0.6B"
reranker_model_name = "Models/Qwen3-Reranker-0.6B"

# 全局变量
print("Loading models for gRPC service...")
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=8192, padding_side='left')
model = AutoModel.from_pretrained(model_name)
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, padding_side='left')
reranker_model = AutoModelForCausalLM.from_pretrained(reranker_model_name).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
reranker_model.to(device)
model.eval()

token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
max_length = 8192
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc)
    return output

def process_inputs(pairs):
    inputs = reranker_tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs

@torch.no_grad()
def compute_logits(inputs):
    batch_scores = reranker_model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

def SplitText(text, chunk_size, overlap_size):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    result = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_token_strs = tokenizer.convert_ids_to_tokens(chunk_tokens)
        chunk_text = tokenizer.convert_tokens_to_string(chunk_token_strs)
        result.append(chunk_text)
        if end == len(tokens):
            break
        start = end - overlap_size if overlap_size > 0 else end
    return result

class TextGrpcServiceServicer(text_service_pb2_grpc.TextGrpcServiceServicer):
    def SimpleRerank(self, request, context):
        query = request.prompt
        documents = list(request.text_blocks)
        pairs = [format_instruction(None, query, doc) for doc in documents]
        inputs = process_inputs(pairs)
        scores = compute_logits(inputs)
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            result = text_service_pb2.RerankResult(
                score=score,
                text_block=doc,
                index=i
            )
            results.append(result)
        response = text_service_pb2.SimpleRerankResponse(
            ranked_results=results,
            total_blocks=len(documents),
            query=query
        )
        return response

    def EmbedText(self, request, context):
        inputs = list(request.input)
        tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=8192)
        tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, tokenized_inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_list = embeddings.tolist()
        data = []
        for i, (emb, text) in enumerate(zip(embeddings_list, inputs)):
            item = text_service_pb2.EmbeddingData(
                object="embedding",
                embedding=emb,
                index=i,
                text_length=len(tokenizer.encode(text))
            )
            data.append(item)
        total_tokens = sum(len(ids) for ids in tokenized_inputs['input_ids'])
        usage = text_service_pb2.Usage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens
        )
        response = text_service_pb2.EmbedTextResponse(
            data=data,
            model=model_name,
            object="list",
            usage=usage
        )
        return response

    def SplitTextIntoChunks(self, request, context):
        chunks = SplitText(request.input, request.chunk_size, request.overlap_size)
        response = text_service_pb2.SplitTextResponse(
            chunks=chunks,
            total_chunks=len(chunks),
            chunk_size=request.chunk_size,
            overlap_size=request.overlap_size,
            average_chunk_length=sum(len(c) for c in chunks) / max(len(chunks), 1)
        )
        return response

    def HealthCheck(self, request, context):
        import psutil
        pid = os.getpid()
        mem = psutil.Process(pid).memory_info().rss / 1024 / 1024
        cpu = psutil.Process(pid).cpu_percent(interval=0.1)
        status = "healthy"
        models_loaded = text_service_pb2.ModelStatus(
            reranker=True,
            embedding=True
        )
        response = text_service_pb2.HealthCheckResponse(
            status=status,
            timestamp=time.time(),
            memory_usage_mb=str(mem),
            cpu_percent=str(cpu),
            pid=pid,
            models_loaded=models_loaded
        )
        return response

    def GetModelInfo(self, request, context):
        reranker_model_detail = text_service_pb2.ModelDetail(
            loaded=True,
            type="reranker",
            model_name=reranker_model_name
        )
        embedding_model_detail = text_service_pb2.ModelDetail(
            loaded=True,
            type="embedding",
            model_name=model_name
        )
        response = text_service_pb2.ModelInfoResponse(
            reranker_model=reranker_model_detail,
            embedding_model=embedding_model_detail
        )
        return response

def serve():
    # 从环境变量获取端口，默认为32688
    port = os.environ.get('GRPC_PORT', '32688')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    text_service_pb2_grpc.add_TextGrpcServiceServicer_to_server(TextGrpcServiceServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"gRPC server started on port {port}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
