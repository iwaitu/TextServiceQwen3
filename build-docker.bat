@echo off
echo 构建支持CUDA GPU的Docker镜像...
docker build -t qwen3-grpc-service:latest .

if %ERRORLEVEL% EQU 0 (
    echo 镜像构建成功！
    echo.
    echo 使用以下命令启动服务（GPU支持）：
    echo docker run -d -p 32688:32688 --gpus all --name qwen3-grpc qwen3-grpc-service:latest
    echo.
    echo 或者使用docker-compose（推荐，自动GPU支持）：
    echo docker-compose up -d
    echo.
    echo 服务将在端口32688上运行，支持CUDA GPU加速
    echo.
    echo 注意：确保已安装NVIDIA Docker运行时！
) else (
    echo 镜像构建失败！
    pause
    exit /b 1
)

pause
