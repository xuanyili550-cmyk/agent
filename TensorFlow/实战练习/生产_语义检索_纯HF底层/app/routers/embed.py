"""嵌入路由：POST /api/embed —— 文本 → 向量(像一个自建的 Embeddings API)。
纯 HF 底层：AutoTokenizer 分词 → AutoModel 前向 → 池化 → 归一化(见 services/embedding)。"""
from fastapi import APIRouter

from ..core.config import get_settings
from ..core.exceptions import ValidationError
from ..schemas.search import EmbedRequest, EmbedResponse
from ..services import embedding

router = APIRouter(prefix="/api", tags=["embed"])


@router.post("/embed", response_model=EmbedResponse)
def api_embed(req: EmbedRequest):
    s = get_settings()
    if len(req.texts) > s.max_texts_per_request:
        raise ValidationError(f"单次文本过多(>{s.max_texts_per_request})。")
    if any(len(t) > s.max_text_chars for t in req.texts):
        raise ValidationError(f"存在超长文本(>{s.max_text_chars} 字符)。")
    vecs = embedding.embed_passages(req.texts)
    return EmbedResponse(dim=int(vecs.shape[1]) if vecs.size else 0, vectors=vecs.tolist())
