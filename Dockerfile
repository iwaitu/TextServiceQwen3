# 使用NVIDIA CUDA基础镜像，支持GPU加速
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 安装Python 3.10和系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 设置环境变量
ENV GRPC_PORT=32688

# 暴露端口
EXPOSE 32688

# 设置健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:32688'); channel.close()" || exit 1

# 启动gRPC服务
CMD ["python", "grpc_service.py"]
