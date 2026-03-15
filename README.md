# TextServiceQwen3

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![gRPC](https://img.shields.io/badge/gRPC-1.60%2B-4285F4.svg)](https://grpc.io/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](https://www.docker.com/)
[![GitHub stars](https://img.shields.io/github/stars/iwaitu/TextServiceQwen3.svg?style=social&label=Star)](https://github.com/iwaitu/TextServiceQwen3/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/iwaitu/TextServiceQwen3.svg?style=social&label=Fork)](https://github.com/iwaitu/TextServiceQwen3/network/members)
[![GitHub issues](https://img.shields.io/github/issues/iwaitu/TextServiceQwen3.svg)](https://github.com/iwaitu/TextServiceQwen3/issues)
[![GitHub release](https://img.shields.io/github/release/iwaitu/TextServiceQwen3.svg)](https://github.com/iwaitu/TextServiceQwen3/releases)

基于 Qwen3 ONNX 模型的文本处理服务，提供文本嵌入、重排序和分块功能。支持 FastAPI REST API 和 gRPC 两种接口方式。

## 功能特性

- 🔤 **文本嵌入**：使用 Qwen3-Embedding-0.6B ONNX 模型生成文本向量
- 🔄 **文本重排序**：使用 Qwen3-Reranker-0.6B ONNX 模型对文本块进行相关性排序
- ✂️ **文本分块**：智能分割长文本为可处理的块
- 🌐 **双接口支持**：同时提供 REST API 和 gRPC 接口
- 🐳 **Docker 支持**：完整的容器化部署方案
- 📊 **API 文档**：内置 Swagger UI 文档

## 快速开始

### 方式一：本地运行

#### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/iwaitu/TextServiceQwen3.git
cd TextServiceQwen3

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

#### 2. 模型下载

确保在 `Models/` 目录下有以下 ONNX 模型目录：
- `qwen3-embedding-0.6b-onnx/`
- `qwen3-reranker-seq-cls-onnx/`

#### 3. 启动服务

```bash
# 启动 FastAPI 服务（REST API）
python main.py

# 或者启动 gRPC 服务
python grpc_service.py
```

### 方式二：Docker 部署

当前 Docker 镜像默认启动的是 gRPC 服务，不会启动 FastAPI REST 服务。容器暴露端口为 `32688`，并依赖 NVIDIA GPU 运行时以及 `Models/` 目录挂载。

#### 1. 使用 Docker Compose（推荐）

```bash
# 构建并启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 查看容器健康状态
docker-compose ps
```

#### 2. 手动 Docker 构建

```bash
# 构建镜像
docker build -t qwen3-text-service .

# 运行容器
docker run -d --gpus all -p 32688:32688 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e EMBEDDING_MODEL_DIR=/app/Models/qwen3-embedding-0.6b-onnx \
  -e RERANKER_MODEL_DIR=/app/Models/qwen3-reranker-seq-cls-onnx \
  -e ONNX_EXECUTION_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
  -v ${PWD}/Models:/app/Models:ro \
  qwen3-text-service
```

如果你使用的是 Windows CMD，可以将卷挂载改为：

```bat
docker run -d --gpus all -p 32688:32688 ^
  -e CUDA_VISIBLE_DEVICES=0 ^
  -e EMBEDDING_MODEL_DIR=/app/Models/qwen3-embedding-0.6b-onnx ^
  -e RERANKER_MODEL_DIR=/app/Models/qwen3-reranker-seq-cls-onnx ^
  -e ONNX_EXECUTION_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider ^
  -v %cd%\Models:/app/Models:ro ^
  qwen3-text-service
```

## API 文档

### REST API

服务启动后，访问以下地址查看 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 主要接口

#### 1. 文本嵌入

```bash
POST /embed_text
Content-Type: application/json

{
  "input": ["文本1", "文本2"]
}
```

#### 2. 文本重排序

```bash
POST /rerank
Content-Type: application/json

{
  "query": "查询文本",
  "documents": ["候选文本1", "候选文本2"],
  "instruction": "可选的指令文本"
}
```

#### 3. 文本分块

```bash
POST /split_text
Content-Type: application/json

{
  "text": "要分块的长文本",
  "chunksize": 1000,
  "overlap_size": 100
}
```

### gRPC 接口

gRPC 服务运行在端口 32688，提供以下服务：

- `EmbedText`: 文本嵌入
- `SimpleRerank`: 文本重排序  
- `SplitTextIntoChunks`: 文本分块
- `HealthCheck`: 健康检查
- `GetModelInfo`: 模型信息

## 使用示例

### Python 客户端示例

```python
import requests

# REST API 示例
base_url = "http://localhost:8000"

# 文本嵌入
response = requests.post(f"{base_url}/embed_text", json={
    "input": ["Hello world", "你好世界"]
})
embeddings = response.json()

# 文本重排序
response = requests.post(f"{base_url}/rerank", json={
    "query": "人工智能技术",
  "documents": [
        "机器学习是人工智能的核心技术",
        "今天天气很好",
        "深度学习推动了AI的发展"
  ],
  "instruction": "Given a web search query, retrieve relevant passages that answer the query"
})
ranked_results = response.json()
```

### gRPC 客户端示例

```python
import grpc
import text_service_pb2
import text_service_pb2_grpc

# 连接 gRPC 服务
channel = grpc.insecure_channel('localhost:32688')
stub = text_service_pb2_grpc.TextGrpcServiceStub(channel)

# 健康检查
health = stub.HealthCheck(text_service_pb2.HealthCheckRequest())
print(health.status)

# 模型信息
model_info = stub.GetModelInfo(text_service_pb2.ModelInfoRequest())
print(model_info.embedding_model.model_name)
print(model_info.reranker_model.model_name)

# 文本嵌入
embed_request = text_service_pb2.EmbedTextRequest(input=["Hello world"])
embed_response = stub.EmbedText(embed_request)

# 文本重排序
rerank_request = text_service_pb2.SimpleRerankRequest(
    prompt="人工智能技术",
    text_blocks=[
        "机器学习是人工智能的核心技术",
        "今天天气很好",
        "深度学习推动了AI的发展"
    ]
)
rerank_response = stub.SimpleRerank(rerank_request)
```

## 配置说明

### 环境变量

- `CUDA_VISIBLE_DEVICES`: 指定使用的 GPU 设备
- `EMBEDDING_MODEL_DIR`: Embedding ONNX 模型目录
- `RERANKER_MODEL_DIR`: Reranker ONNX 模型目录
- `EMBEDDING_MAX_LENGTH`: Embedding 最大输入长度，默认 8192
- `RERANK_MAX_LENGTH`: Rerank 最大输入长度，默认 8192
- `EMBEDDING_BATCH_SIZE`: Embedding 推理批大小，默认 8
- `ONNX_EXECUTION_PROVIDERS`: 显式指定 ONNX Runtime Provider 链
- `ONNX_PREFERRED_PROVIDER`: 指定首选 Provider
- `ONNX_PROVIDER`: 兼容单 Provider 配置

### 模型配置

- **嵌入模型**: qwen3-embedding-0.6b-onnx
- **重排序模型**: qwen3-reranker-seq-cls-onnx
- **支持设备**: CPU / CUDA GPU

## 项目结构

```
TextServiceQwen3/
├── main.py                 # FastAPI 主服务
├── grpc_service.py         # gRPC 服务
├── requirements.txt        # Python 依赖
├── Dockerfile              # Docker 构建文件
├── docker-compose.yml      # Docker Compose 配置
├── protos/                 # gRPC Protocol Buffers 定义
│   └── text_service.proto
├── static/                 # 静态文件（Swagger UI）
├── Models/                 # 模型文件目录（被 .gitignore 忽略）
│   ├── qwen3-embedding-0.6b-onnx/
│   └── qwen3-reranker-seq-cls-onnx/
├── test_rerank.py          # 测试脚本
├── test_grpc_client.py     # gRPC smoke test
├── benchmark_service.py    # REST benchmark
└── benchmark_grpc.py       # gRPC benchmark
```

## 开发指南

### 生成 gRPC 代码

```bash
# 从 .proto 文件生成 Python 代码
python -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/text_service.proto
```

### 运行测试

```bash
# 运行重排序测试
python test_rerank.py
```

## 基准测试

### gRPC Benchmark

当前仓库新增了 `benchmark_grpc.py`，用于对 gRPC 服务执行健康检查、Embedding 吞吐测试、Rerank 吞吐测试和准确性评估。

在 Windows CMD 中，如果你的服务运行在 conda 的 `agent` 环境，推荐直接使用以下命令：

```bat
C:\Users\iwaitu\anaconda3\envs\agent\python.exe benchmark_grpc.py --target 127.0.0.1:32688
```

如果只想快速验证性能而跳过准确性样本，可以使用：

```bat
C:\Users\iwaitu\anaconda3\envs\agent\python.exe benchmark_grpc.py --target 127.0.0.1:32688 --skip-accuracy
```

### 最近一次 gRPC 实测结果

- 测试环境：Windows CMD + conda `agent` 环境
- 服务地址：`127.0.0.1:32688`
- 健康状态：`healthy`
- 模型加载状态：Embedding 与 Reranker 均已加载
- Embedding 压测：32 条文本，稳态平均 79.34 ms，P95 81.24 ms，吞吐量 403.35 条/秒，向量维度 1024
- Rerank 压测：50 条候选文本，稳态平均 147.46 ms，P95 156.39 ms，吞吐量 339.07 条/秒
- Rerank 准确性：5 个样本，Top1 Accuracy = 1.0，Top3 Accuracy = 1.0，MRR = 1.0

### REST Benchmark

仓库中的 `benchmark_service.py` 可用于 REST API 的同类测试。若 REST 服务已启动，可使用：

```bat
C:\Users\iwaitu\anaconda3\envs\agent\python.exe benchmark_service.py --base-url http://127.0.0.1:8000
```

## 性能优化

- 使用 GPU 加速（推荐 CUDA）
- 批量处理请求以提高吞吐量
- 根据硬件配置调整 `max_length` 参数
- 使用 Docker 容器限制资源使用

## 故障排除

### 常见问题

1. **模型加载失败**
   - 确保 `Models/` 目录下有正确的模型文件
   - 检查模型文件权限

2. **CUDA 内存不足**
   - 减少批处理大小
   - 使用 CPU 模式运行

3. **端口冲突**
   - 修改 `docker-compose.yml` 中的端口映射
   - 检查防火墙设置

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

版权所有 © 2025 青云微笙科技公司

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.0.0
- 初始版本发布
- 支持文本嵌入、重排序和分块功能
- 提供 REST API 和 gRPC 双接口
- Docker 容器化支持
