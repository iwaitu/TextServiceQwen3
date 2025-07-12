# TextServiceQwen3

基于 Qwen3 模型的文本处理服务，提供文本嵌入、重排序和分块功能。支持 FastAPI REST API 和 gRPC 两种接口方式。

## 功能特性

- 🔤 **文本嵌入**：使用 Qwen3-Embedding-0.6B 模型生成文本向量
- 🔄 **文本重排序**：使用 Qwen3-Reranker-0.6B 模型对文本块进行相关性排序
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

确保在 `Models/` 目录下有以下模型：
- `Qwen3-Embedding-0.6B/`
- `Qwen3-Reranker-0.6B/`

#### 3. 启动服务

```bash
# 启动 FastAPI 服务（REST API）
python main.py

# 或者启动 gRPC 服务
python grpc_service.py
```

### 方式二：Docker 部署

#### 1. 使用 Docker Compose（推荐）

```bash
# 构建并启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

#### 2. 手动 Docker 构建

```bash
# 构建镜像
docker build -t qwen3-text-service .

# 运行容器
docker run -d -p 32688:32688 qwen3-text-service
```

## API 文档

### REST API

服务启动后，访问以下地址查看 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 主要接口

#### 1. 文本嵌入

```bash
POST /embed
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
  "texts": ["候选文本1", "候选文本2"],
  "instruction": "可选的指令文本"
}
```

#### 3. 文本分块

```bash
POST /split
Content-Type: application/json

{
  "text": "要分块的长文本",
  "chunk_size": 1000,
  "overlap": 100
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
response = requests.post(f"{base_url}/embed", json={
    "input": ["Hello world", "你好世界"]
})
embeddings = response.json()

# 文本重排序
response = requests.post(f"{base_url}/rerank", json={
    "query": "人工智能技术",
    "texts": [
        "机器学习是人工智能的核心技术",
        "今天天气很好",
        "深度学习推动了AI的发展"
    ]
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

# 文本嵌入
request = text_service_pb2.EmbedTextRequest(input=["Hello world"])
response = stub.EmbedText(request)
```

## 配置说明

### 环境变量

- `CUDA_VISIBLE_DEVICES`: 指定使用的 GPU 设备
- `MODEL_MAX_LENGTH`: 模型最大输入长度（默认：8192）

### 模型配置

- **嵌入模型**: Qwen3-Embedding-0.6B
- **重排序模型**: Qwen3-Reranker-0.6B
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
│   ├── Qwen3-Embedding-0.6B/
│   └── Qwen3-Reranker-0.6B/
└── test_rerank.py          # 测试脚本
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

[添加您的许可证信息]

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.0.0
- 初始版本发布
- 支持文本嵌入、重排序和分块功能
- 提供 REST API 和 gRPC 双接口
- Docker 容器化支持
