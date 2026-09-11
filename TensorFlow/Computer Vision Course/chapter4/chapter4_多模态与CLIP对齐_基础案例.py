"""
 CV Course · Ch4 · 基础案例：CLIP 式图文余弦匹配(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np

def l2(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)

img = np.random.default_rng(3).standard_normal(8)
texts = np.random.default_rng(4).standard_normal((3, 8)); texts[0] = img + 0.1
sims = l2(texts) @ l2(img)
print("各描述与图的余弦:", sims.round(2), "→ 最匹配 idx", int(np.argmax(sims)))
print("✅ 对齐到同一空间后,余弦相似度即可跨模态比较")
