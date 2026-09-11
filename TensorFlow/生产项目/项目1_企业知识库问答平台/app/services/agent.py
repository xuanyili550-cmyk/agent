"""Agent 决策:闲聊直接答 / 其余走 RAG。两条分支都带注入检测结果。"""
import re
from . import rag, security
_CHITCHAT = re.compile(r"你好|hi|hello|谢谢|再见|你是谁")


def handle(query: str) -> dict:
    inj = security.detect_injection(query)
    if _CHITCHAT.search((query or "").lower()):
        return {"route": "chitchat", "answer": "你好!我是企业知识库助手,可以问我文档里的内容。",
                "sources": [], "security": {"query_injection": bool(inj)}}
    r = rag.answer(query); r["route"] = "rag"
    return r
