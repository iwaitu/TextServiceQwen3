from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import os

# Use Qwen3-Embedding model for embedding service
model_name = "Models/Qwen3-Embedding-0.6B"
reranker_model_name = "Models/Qwen3-Reranker-0.6B"

# Global variables for models and tokenizers
tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModel] = None
reranker_tokenizer: Optional[AutoTokenizer] = None
reranker_model: Optional[AutoModelForCausalLM] = None
device: Optional[str] = None
token_false_id: Optional[int] = None
token_true_id: Optional[int] = None
max_length = 8192
prefix_tokens: Optional[List[int]] = None
suffix_tokens: Optional[List[int]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model loading and cleanup"""
    global tokenizer, model, reranker_tokenizer, reranker_model, device
    global token_false_id, token_true_id, prefix_tokens, suffix_tokens
    
    # Startup: Load models
    print("Starting up: Loading models...")
    try:
        # Load embedding model
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=8192, padding_side='left')
        model = AutoModel.from_pretrained(model_name)
        
        # Load reranker model
        reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, padding_side='left')
        reranker_model = AutoModelForCausalLM.from_pretrained(reranker_model_name).eval()
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)  # type: ignore
        reranker_model.to(device)  # type: ignore
        model.eval()  # type: ignore
        
        # Reranker specific tokens and settings
        token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")  # type: ignore
        token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")  # type: ignore
        
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)  # type: ignore
        suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)  # type: ignore
        
        print(f"Models loaded successfully on device: {device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
    
    yield  # Application is running
    
    # Shutdown: Cleanup resources
    print("Shutting down: Cleaning up resources...")
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        print("Resources cleaned up successfully")
    except Exception as e:
        print(f"Error during cleanup: {e}")


app = FastAPI(
    title="Qwen3 Text Service",
    description="A FastAPI service for text embeddings and reranking using Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B models",
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    lifespan=lifespan
)


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
    model: str = model_name


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
    model: str = model_name
    object: str = "list"
    usage: Usage


@app.post("/embed_text", response_model=EmbedTextResponse, tags=["Embedding"])
async def embed_text(request: EmbedTextRequest):
    """
    Generate embeddings for documents without instruction formatting.
    
    This endpoint is optimized for document/passage embeddings where you don't 
    need instruction formatting. Use this for embedding knowledge base documents,
    passages, or any content that will be retrieved by queries.
    
    - **input**: List of document/passage strings to embed
    
    Example:
    ```json
    {
        "input": [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other."
        ]
    }
    ```
    
    Note: The task_description field is ignored for this endpoint.
    """
    if tokenizer is None or model is None or device is None:
        raise HTTPException(status_code=500, detail="Models not loaded. Please wait for startup to complete.")
    
    try:
        inputs = request.input
        
        # Tokenize inputs without instruction formatting
        tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=8192)  # type: ignore
        tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
        
        with torch.no_grad():
            outputs = model(**tokenized_inputs)  # type: ignore
            embeddings = last_token_pool(outputs.last_hidden_state, tokenized_inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        # 推理后清理显存
        if device == "cuda":
            torch.cuda.empty_cache()

        embeddings_list = embeddings.tolist()
        data = [
            EmbeddingData(embedding=emb, index=i, tokens_length=len(tokenizer.encode(text)))  # type: ignore
            for i, (emb, text) in enumerate(zip(embeddings_list, inputs))
        ]
        # 统计实际输入到模型的 token 数量（包含 padding 和 truncation 后的结果）
        total_tokens = sum(len(ids) for ids in tokenized_inputs['input_ids'])  # type: ignore
        usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)
        return EmbedTextResponse(data=data, usage=usage)

    except Exception as e:
        # 异常时也清理显存
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"Error during embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank", response_model=RerankResponse, tags=["Reranking"])
async def simple_rerank(request: RerankRequest):
    """
    Rerank documents based on their relevance to a query using Qwen3-Reranker-0.6B.
    
    This endpoint takes a query and a list of documents, then returns them 
    ranked by relevance score in descending order.
    
    - **query**: The search query string
    - **documents**: List of document strings to rank
    - **instruction**: Optional instruction for the reranking task
    
    Example:
    ```json
    {
        "query": "What is the capital of China?",
        "documents": [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other.",
            "Paris is the capital of France."
        ],
        "instruction": "Given a web search query, retrieve relevant passages that answer the query"
    }
    ```
    """
    if reranker_model is None or reranker_tokenizer is None or device is None:
        raise HTTPException(status_code=500, detail="Reranker models not loaded. Please wait for startup to complete.")
    
    try:
        query = request.query
        documents = request.documents
        instruction = request.instruction
        
        # Format input pairs
        pairs = [format_instruction(instruction, query, doc) for doc in documents]
        
        # Process inputs
        inputs = process_inputs(pairs)
        
        # Compute scores
        scores = compute_logits(inputs)
        
        # 推理后清理显存
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Create results with original indices and sort by score (descending)
        results = [
            RerankResult(document=doc, score=score, index=i)
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        
        # Sort by score in descending order
        results.sort(key=lambda x: x.score, reverse=True)
        
        return RerankResponse(results=results)
        
    except Exception as e:
        # 异常时也清理显存
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"Error during reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/split_text", response_model=SplitTextResponse, tags=["Text Processing"])
async def split_text_api(request: SplitTextRequest):
    """
    Split text into chunks based on token count with optional overlap.
    
    This endpoint splits a given text into smaller chunks based on the specified
    chunk size (in tokens) and overlap size (in tokens).
    
    - **text**: The input text to be split
    - **chunksize**: The maximum number of tokens per chunk (must be > 0)
    - **overlap_size**: The number of tokens to overlap between chunks (default: 0, must be >= 0)
    
    Example:
    ```json
    {
        "text": "This is a long text that needs to be split into smaller chunks for processing.",
        "chunksize": 10,
        "overlap_size": 2
    }
    ```
    """
    try:
        chunks = SplitText(request.text, request.chunksize, request.overlap_size)
        return SplitTextResponse(
            chunks=chunks,
            total_chunks=len(chunks)
        )
    except Exception as e:
        print(f"Error during text splitting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def format_instruction(instruction, query, doc):
    """Format instruction for reranker model"""
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc)
    return output


def process_inputs(pairs):
    """Process input pairs for reranker model"""
    if reranker_tokenizer is None or prefix_tokens is None or suffix_tokens is None:
        raise RuntimeError("Reranker tokenizer or tokens not initialized")
    
    inputs = reranker_tokenizer(  # type: ignore
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)  # type: ignore
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs


@torch.no_grad()
def compute_logits(inputs, **kwargs):
    """Compute reranker scores"""
    if reranker_model is None or token_true_id is None or token_false_id is None:
        raise RuntimeError("Reranker model or tokens not initialized")
    
    batch_scores = reranker_model(**inputs).logits[:, -1, :]  # type: ignore
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def SplitText(text: str, chunksize: int, overlap_size: int) -> List[str]:
    """
    按照 chunksize（token数量）和 overlap_size（token数量）将文本切分为多个段落，返回 List[str]
    """
    global tokenizer
    if tokenizer is None:
        raise RuntimeError("tokenizer 未初始化")
    if chunksize <= 0:
        raise ValueError("chunksize 必须大于 0")
    if overlap_size < 0:
        raise ValueError("overlap_size 不能为负数")
    # 分词
    tokens = tokenizer.encode(text, add_special_tokens=False)  # type: ignore
    result = []
    start = 0
    while start < len(tokens):
        end = min(start + chunksize, len(tokens))
        chunk_tokens = tokens[start:end]
        # 先将 token id 转为 token，再转为字符串，避免中文乱码
        chunk_token_strs = tokenizer.convert_ids_to_tokens(chunk_tokens)  # type: ignore
        chunk_text = tokenizer.convert_tokens_to_string(chunk_token_strs)  # type: ignore
        result.append(chunk_text)
        if end == len(tokens):
            break
        start = end - overlap_size if overlap_size > 0 else end
    return result


# Mount static files for offline Swagger UI
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
        swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"}
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
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embedding_model": model_name,
        "reranker_model": reranker_model_name,
        "device": device,
        "service": "Qwen3 Text Service"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)