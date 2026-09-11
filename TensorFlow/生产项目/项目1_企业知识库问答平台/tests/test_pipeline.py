"""离线端到端:stub 后端,不下模型。入库→检索→问答全跑通。"""
from app.services import rag
from app.services.index import get_index
from app.services import chunk as chunk_mod


def setup_function(_):
    get_index().clear()


def test_chunk_overlap():
    chunks = chunk_mod.chunk("句一。句二。句三。" * 40)
    assert len(chunks) >= 2


def test_ingest_and_answer():
    rag.ingest("faq", "退货政策是7天无理由退货。满99元包邮。客服工作时间9点到18点。")
    r = rag.answer("退货政策是什么")
    assert "退货" in r["answer"] and len(r["sources"]) >= 1    # 检索到含"退货"的块


def test_retrieval_relevance():
    get_index().clear()
    rag.ingest("kb", "苹果是一种水果。北京是中国的首都。Python 是编程语言。")
    r = rag.answer("首都在哪")
    assert "北京" in r["answer"]                              # 词面相关的块被检索到
