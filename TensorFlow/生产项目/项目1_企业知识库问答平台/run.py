"""本机启动:python run.py → http://127.0.0.1:8000/docs"""
import uvicorn
from app.core.config import get_settings
if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=s.debug)
