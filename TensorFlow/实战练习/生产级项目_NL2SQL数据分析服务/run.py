# 【构建 15/16 · 运维】本机入口（16=Dockerfile/compose/README/.env.example 最后写）
"""本机启动入口：python run.py  →  http://localhost:8000/docs (Swagger)"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
