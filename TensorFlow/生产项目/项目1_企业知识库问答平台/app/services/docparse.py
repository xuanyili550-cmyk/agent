"""文档解析:PyMuPDF 解析 PDF(懒导入),纯文本直接读。生产可扩 docx/html/OCR。"""
import os
from ..core.config import get_settings
from ..core.exceptions import ValidationError


def parse(path_or_text: str, is_text: bool = False) -> str:
    """返回纯文本。is_text=True 时直接当文本;否则按文件后缀解析。"""
    s = get_settings()
    if is_text:
        text = path_or_text
    elif path_or_text.lower().endswith(".pdf"):
        import fitz                                   # PyMuPDF,懒导入
        doc = fitz.open(path_or_text)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    else:                                             # .txt/.md 等
        text = open(path_or_text, encoding="utf-8", errors="ignore").read()
    if not text.strip():
        raise ValidationError("文档解析为空")
    if len(text) > s.max_doc_chars:
        text = text[:s.max_doc_chars]                 # 超长截断(防打爆)
    return text
