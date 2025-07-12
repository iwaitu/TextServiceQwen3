@echo off
echo 启动 Qwen3 Embedding Service...
echo.
echo 服务将在以下地址启动:
echo   - 主页: http://localhost:8000/
echo   - Swagger UI (离线): http://localhost:8000/docs
echo   - ReDoc: http://localhost:8000/redoc
echo   - 健康检查: http://localhost:8000/health
echo.
echo 按 Ctrl+C 停止服务
echo.

cd /d "%~dp0"
python main.py
pause
