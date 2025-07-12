# Qwen3 gRPC Service Docker 部署指南

本项目提供了基于 NVIDIA CUDA 的 Docker 容器化方案，支持 GPU 加速的 Qwen3 文本嵌入和重排序服务。

## 📋 前置要求

### 1. NVIDIA Docker 运行时
确保您的系统已安装：
- NVIDIA GPU 驱动程序
- Docker
- NVIDIA Container Toolkit

### 2. 安装 NVIDIA Container Toolkit
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3. 验证 GPU 支持
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 🚀 快速开始

### 方式1：使用构建脚本（推荐）
```bash
# Windows
build-docker.bat

# Linux/Mac
chmod +x build-docker.sh
./build-docker.sh
```

### 方式2：使用 docker-compose
```bash
docker-compose up -d
```

### 方式3：手动构建和运行
```bash
# 构建镜像
docker build -t qwen3-grpc-service:latest .

# 运行容器（GPU支持）
docker run -d \
  -p 32688:32688 \
  --gpus all \
  --name qwen3-grpc \
  -v $(pwd)/Models:/app/Models:ro \
  qwen3-grpc-service:latest
```

## 🔧 配置说明

### 环境变量
- `GRPC_PORT`: gRPC服务端口（默认：32688）
- `CUDA_VISIBLE_DEVICES`: 指定使用的GPU（默认：0）

### 端口映射
- 容器端口：32688
- 主机端口：32688

### 模型文件
模型文件应放在 `./Models/` 目录下：
```
Models/
├── Qwen3-Embedding-0.6B/
└── Qwen3-Reranker-0.6B/
```

## 🔍 健康检查

容器启动后，可以通过以下方式检查服务状态：

```bash
# 检查容器状态
docker ps

# 查看容器日志
docker logs qwen3-grpc

# 测试gRPC连接
python -c "import grpc; channel = grpc.insecure_channel('localhost:32688'); print('连接成功'); channel.close()"
```

## 🐛 故障排除

### 1. GPU 不可用
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查Docker GPU支持
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 2. 内存不足
- 调整 docker-compose.yml 中的内存限制
- 确保主机有足够的GPU内存

### 3. 模型加载失败
- 检查模型文件路径
- 确保模型文件完整且可读

## 📊 性能优化

### GPU 内存优化
- 设置适当的 `CUDA_VISIBLE_DEVICES`
- 调整批处理大小
- 启用模型量化（如适用）

### 容器资源限制
在 docker-compose.yml 中调整：
```yaml
deploy:
  resources:
    limits:
      memory: 16G  # 根据需要调整
    reservations:
      memory: 8G
```

## 📝 API 使用示例

服务启动后，可通过 gRPC 客户端调用以下服务：
- `EmbedText`: 文本嵌入
- `SimpleRerank`: 文档重排序
- `SplitTextIntoChunks`: 文本分块
- `HealthCheck`: 健康检查
- `GetModelInfo`: 模型信息

端口：`localhost:32688`
