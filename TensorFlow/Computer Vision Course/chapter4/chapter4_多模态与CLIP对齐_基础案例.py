"""
 CV Course · Ch4 · 基础案例：CLIP 对比学习对齐矩阵 + InfoNCE 损失(纯 numpy,机制真实,可跑)
 CLIP 把图、文各自编码成向量,L2 归一化后放进同一空间;一个 batch 内算 图×文 余弦相似度矩阵。
 训练目标(对比学习):对角线(配对的图文)相似度应最高,非对角(不配对)应低 → 用带温度的 InfoNCE。
 这里纯 numpy 真造相似度矩阵、真算双向 InfoNCE 损失,并验证"对齐后对角占优、检索命中"。
 跑：python3 本文件
"""
import numpy as np


def l2norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def similarity_matrix(img_emb, txt_emb, temp=0.07):
    """图文相似度矩阵 S[i,j] = 余弦(图i, 文j) / 温度。温度越小,softmax 越尖锐(拉大正负样本差距)。"""
    return (l2norm(img_emb) @ l2norm(txt_emb).T) / temp


def infonce(sim):
    """双向 InfoNCE:每张图要在所有文本里选中自己的配对(行 softmax),每段文本也要选中自己的图(列 softmax)。
    标签就是对角线 i↔i。损失 = 行方向交叉熵 + 列方向交叉熵 的平均。"""
    def ce_rows(s):
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return -np.log(np.diag(p) + 1e-9).mean()           # 取对角(正确配对)的负对数似然
    return 0.5 * (ce_rows(sim) + ce_rows(sim.T))


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    d, n = 8, 4
    # 造 n 对"语义配对":每对图文共享一个隐含主题向量 + 各自独立噪声(模拟已训练好的编码器输出)
    themes = rng.standard_normal((n, d))
    img_emb = themes + 0.10 * rng.standard_normal((n, d))
    txt_emb = themes + 0.10 * rng.standard_normal((n, d))

    sim = similarity_matrix(img_emb, txt_emb)
    loss = infonce(sim)
    # 对比:把文本打乱(破坏配对)后损失应显著变大
    loss_shuffled = infonce(similarity_matrix(img_emb, txt_emb[rng.permutation(n)]))

    print("相似度矩阵(行=图,列=文):\n", sim.round(1))
    print(f"图→文检索命中(每行 argmax 应等于行号): {[int(sim[i].argmax()) for i in range(n)]}")
    print(f"InfoNCE 损失 正确配对={loss:.3f}  打乱后={loss_shuffled:.3f}")
    assert all(sim[i].argmax() == i for i in range(n))     # 对角线相似度最大 → 检索正确
    assert loss < loss_shuffled                            # 正确配对损失更低,说明目标函数有效
    print("✅ 归一化到同空间 → 相似度矩阵对角占优 → InfoNCE 拉近配对/推开非配对,这就是 CLIP 的对齐机制")
    # 面试Q:CLIP 的温度系数(temperature)起什么作用?
    #      A:它缩放 logits——温度越小 softmax 越尖锐,更狠地惩罚难负样本、加速收敛但易过拟合;通常设为可学习参数并裁剪上限。
