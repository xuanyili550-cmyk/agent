"""
嘉宾信息检索工具（BM25）—— 三框架各一份。
每个框架用一个"加载函数"惰性构建：调用时才 load_dataset（联网），import 本模块很轻。
  smolagents  -> load_guest_dataset()                返回 GuestInfoRetrieverTool 实例
  llama-index -> load_guest_info_tool_llama()         返回 FunctionTool
  langgraph   -> load_guest_info_tool_langgraph()     返回 langchain Tool
"""

import datasets

_DATASET = "agents-course/unit3-invitees"


# =====================================================================
# #smolagents
# =====================================================================
from smolagents import Tool as _SmolTool
from langchain_community.retrievers import BM25Retriever as _LcBM25
from langchain_core.documents import Document as _LcDocument


def _load_docs_langchain():
    """加载数据集，转成 langchain 的 Document 列表。"""
    ds = datasets.load_dataset(_DATASET, split="train")
    return [
        _LcDocument(
            page_content="\n".join([
                f"Name: {g['name']}",
                f"Relation: {g['relation']}",
                f"Description: {g['description']}",
                f"Email: {g['email']}",
            ]),
            metadata={"name": g["name"]},
        )
        for g in ds
    ]


class GuestInfoRetrieverTool(_SmolTool):
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about.",
        }
    }
    output_type = "string"

    def __init__(self, docs):
        self.is_initialized = False
        self.retriever = _LcBM25.from_documents(docs)

    def forward(self, query: str):
        results = self.retriever.invoke(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        return "No matching guest information found."


def load_guest_dataset():
    """smolagents 版：加载数据并返回 guest_info 检索工具。"""
    return GuestInfoRetrieverTool(_load_docs_langchain())


# =====================================================================
# #llam-index
# =====================================================================
from llama_index.core.schema import Document as _LiDocument
from llama_index.core.tools import FunctionTool
from llama_index.retrievers.bm25 import BM25Retriever as _LiBM25


def load_guest_info_tool_llama():
    """llama-index 版：返回包装成 FunctionTool 的检索工具。"""
    ds = datasets.load_dataset(_DATASET, split="train")
    docs = [
        _LiDocument(
            text="\n".join([
                f"Name: {ds['name'][i]}",
                f"Relation: {ds['relation'][i]}",
                f"Description: {ds['description'][i]}",
                f"Email: {ds['email'][i]}",
            ]),
            metadata={"name": ds['name'][i]},
        )
        for i in range(len(ds))
    ]
    bm25 = _LiBM25.from_defaults(nodes=docs)

    def get_guest_info_retriever(query: str) -> str:
        """Retrieves detailed information about gala guests based on their name or relation."""
        results = bm25.retrieve(query)
        if results:
            return "\n\n".join([doc.text for doc in results[:3]])
        return "No matching guest information found."

    return FunctionTool.from_defaults(get_guest_info_retriever)


# =====================================================================
# #langgraph
# =====================================================================
from langchain_core.tools import Tool as _LcTool


def load_guest_info_tool_langgraph():
    """langgraph 版：返回 langchain Tool。"""
    bm25 = _LcBM25.from_documents(_load_docs_langchain())

    def extract_text(query: str) -> str:
        """Retrieves detailed information about gala guests based on their name or relation."""
        results = bm25.invoke(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        return "No matching guest information found."

    return _LcTool(
        name="guest_info_retriever",
        func=extract_text,
        description="Retrieves detailed information about gala guests based on their name or relation.",
    )
