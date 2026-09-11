"""本机启动：python run.py → http://127.0.0.1:8000/docs（Swagger 直接试）。
首次调用 /embed 或 /search 会下载嵌入模型(all-MiniLM-L6-v2, ~90MB)。"""
import uvicorn

from app.core.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=s.debug)
