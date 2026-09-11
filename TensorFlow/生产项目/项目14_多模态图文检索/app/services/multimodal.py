"""CLIP 式图文对齐:把图和文编到同一空间(stub 哈希词袋,离线),按余弦跨模态检索。"""
import hashlib, re
import numpy as np
from ..core.exceptions import ValidationError
DIM=64; _TOK=re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_STORE={"ids":[],"texts":[],"mat":None}
def _embed(items):
    out=np.zeros((len(items),DIM),dtype="float32")
    for i,t in enumerate(items):
        for tok in _TOK.findall(t.lower()):
            out[i,int(hashlib.md5(tok.encode()).hexdigest(),16)%DIM]+=1.0
    n=np.linalg.norm(out,axis=1,keepdims=True); n[n==0]=1; return out/n
def index(items):
    if not items: raise ValidationError("items 不能为空")
    vecs=_embed(items)
    _STORE["ids"]+=list(range(len(_STORE["ids"]),len(_STORE["ids"])+len(items)))
    _STORE["texts"]+=items
    _STORE["mat"]=vecs if _STORE["mat"] is None else np.vstack([_STORE["mat"],vecs])
    return {"count":len(_STORE["ids"])}
def search(query, k=3):
    if _STORE["mat"] is None: return []
    q=_embed([query])[0]; sims=_STORE["mat"]@q
    order=np.argsort(-sims)[:k]
    return [{"id":int(_STORE["ids"][i]),"item":_STORE["texts"][i],"score":round(float(sims[i]),3)} for i in order]
def clear(): _STORE.update(ids=[],texts=[],mat=None)
