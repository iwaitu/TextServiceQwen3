@echo off
echo 构建基于 ONNX Runtime GPU 的 Docker 镜像...
docker build -t qwen3-grpc-service:latest .

if %ERRORLEVEL% EQU 0 (
    echo 镜像构建成功！
    echo.
    echo 使用以下命令启动 gRPC 服务（ONNX GPU 支持）：
    echo docker run -d -p 32688:32688 --gpus all --name qwen3-grpc qwen3-grpc-service:latest
    echo.
    echo 或者使用 docker-compose（推荐，自动注入 ONNX 模型目录和 GPU Provider）：
    echo docker-compose up -d
    echo.
    echo 服务将在端口32688上运行，默认使用 qwen3-embedding-0.6b-onnx 和 qwen3-reranker-seq-cls-onnx
    echo.
    echo 注意：确保已安装NVIDIA Docker运行时！
) else (
    echo 镜像构建失败！
    pause
    exit /b 1
)

pause
