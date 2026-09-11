"""切块:中文按标点切句,再按字数聚块 + 重叠(中文无空格不能按词切)。"""
import re
from ..core.config import get_settings

_SENT = re.compile(r"[^。！？!?\n]+[。！？!?]?")


def chunk(text: str) -> list[str]:
    s = get_settings()
    sents = [x.strip() for x in _SENT.findall(text) if x.strip()]
    chunks, cur = [], ""
    for sent in sents:
        if len(cur) + len(sent) > s.chunk_size and cur:
            chunks.append(cur)
            cur = cur[-s.chunk_overlap:] + sent       # 重叠:保留上块尾部,防切断语义
        else:
            cur += sent
    if cur.strip():
        chunks.append(cur)
    return chunks or [text]
