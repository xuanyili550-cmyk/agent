"""嵌入:可切后端。stub(哈希词袋向量,离线/测试可跑) | hf(AutoModel mask 加权 mean 池化)。"""
import hashlib
import re
import numpy as np
from ..core.config import get_settings

_TOK = re.compile(r"[a-z0-9]+|[一-鿿]")
_HF = {}


def _tokenize(t):
    return _TOK.findall(t.lower())


def _stub_embed(texts):
    """哈希词袋:每个 token 哈希到一维 +1,归一化。共享词多 → 余弦高(离线可检索)。"""
    s = get_settings(); dim = s.embed_dim
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for tok in _tokenize(t):
            out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    norm = np.linalg.norm(out, axis=1, keepdims=True); norm[norm == 0] = 1.0
    return out / norm


def _hf_embed(texts):
    import torch
    s = get_settings()
    if "m" not in _HF:
        from transformers import AutoModel, AutoTokenizer
        _HF["dev"] = "mps" if torch.backends.mps.is_available() else "cpu"
        _HF["tok"] = AutoTokenizer.from_pretrained(s.embed_model)
        _HF["m"] = AutoModel.from_pretrained(s.embed_model).to(_HF["dev"]).eval()
    enc = _HF["tok"](texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(_HF["dev"])
    with torch.inference_mode():
        h = _HF["m"](**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return v.cpu().numpy()


def embed(texts: list[str]) -> np.ndarray:
    return _hf_embed(texts) if get_settings().embed_backend == "hf" else _stub_embed(texts)
