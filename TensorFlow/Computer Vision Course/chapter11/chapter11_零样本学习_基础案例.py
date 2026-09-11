"""
 CV Course · Ch11 · 基础案例：基于属性的零样本分类(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np

def l2(x): return x / (np.linalg.norm(x) + 1e-9)
class_attr = {"猫": [0,1,0], "虎": [0,1,1], "鹰": [1,0,0]}   # [会飞,有毛,条纹]
img_attr = np.array([0, 1, 1.0])                              # 预测:有毛+条纹
sims = {c: l2(img_attr) @ l2(np.array(a)) for c, a in class_attr.items()}
print("与各类属性相似度:", {k: round(v,2) for k,v in sims.items()})
print("✅ 预测类:", max(sims, key=sims.get), "(靠属性匹配,无需见过该类图像)")
