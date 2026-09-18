"""
 CV Course · Ch11 · 基础案例：基于属性的零样本分类(纯 numpy,可跑)
 零样本 = 训练时没见过某个类的任何图像,却靠"共享的语义属性"把它认出来(如 CLIP 靠文本嵌入)。
 机制:图像→属性向量,类别→属性向量,在共享属性空间里比余弦相似度,谁近就是谁——桥梁是"属性"而非"像素"。
 这里从零实现:训练类只提供 见过 的类,测试时混入一个 从未见过 的类(斑马),验证它照样能被正确认出。
 跑：python3 本文件
"""
import numpy as np


def l2(x):
    """L2 归一化 → 余弦相似度 = 归一化后点积(去掉向量长度只看方向)。"""
    x = np.asarray(x, float)
    return x / (np.linalg.norm(x) + 1e-9)


# 属性维度:[会飞, 有毛, 有条纹, 四条腿, 有蹄]  —— 这是类和图像共享的语义空间
CLASS_ATTR = {
    "猫":  [0, 1, 0, 1, 0],
    "虎":  [0, 1, 1, 1, 0],
    "鹰":  [1, 0, 0, 0, 0],
    "斑马": [0, 1, 1, 1, 1],   # ★ 测试时才出现的"未见类":训练从没给过斑马的图,只给了它的属性描述("有蹄"把它和虎区分开)
}
SEEN = {"猫", "虎", "鹰"}    # 训练阶段真正见过图像的类


def classify(img_attr, class_attr):
    """在属性空间里做最近邻:对每个候选类算余弦相似度,取最大。"""
    v = l2(img_attr)
    sims = {c: float(v @ l2(a)) for c, a in class_attr.items()}
    return max(sims, key=sims.get), sims


if __name__ == "__main__":
    # 一张斑马图被特征提取器估出的属性:有毛+有条纹+四条腿+有蹄(不会飞)。注意训练从没见过斑马图!
    zebra_img_attr = [0, 1, 1, 1, 1]
    pred, sims = classify(zebra_img_attr, CLASS_ATTR)

    assert pred == "斑马"                     # 未见类靠属性匹配被认出 → 这就是零样本
    assert pred not in SEEN                    # 证明:预测出的类根本不在训练见过的集合里
    # 若把候选限制在"见过的类"里,只能退而求其次认成属性最像的虎(说明未见类的加入确实有用)
    pred_seen_only, _ = classify(zebra_img_attr, {c: CLASS_ATTR[c] for c in SEEN})
    assert pred_seen_only == "虎"

    print("与各类属性余弦相似度:", {k: round(v, 2) for k, v in sims.items()})
    print(f"预测类: {pred}  (训练从未见过斑马图,仅凭属性描述认出 → 零样本)")
    print(f"若只在见过的类里选,会误判为: {pred_seen_only}")
    print("✅ 属性作桥梁:图像和类别投到同一语义空间比相似度,未见类也能识别")
    # 面试 Q&A：零样本为什么能识别没见过的类?——因为类别不再用"独热标签"表示,而用共享的语义向量
    #          (属性/文本嵌入)表示;只要图像能映射到同一空间,新类只需给出它的语义描述即可比对识别。
