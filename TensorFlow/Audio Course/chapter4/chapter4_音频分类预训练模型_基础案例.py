"""
 Audio Course · Ch4 · 基础案例：从零复现"预训练模型 + 微调头"做关键词识别 KWS（纯 numpy）
 用预训练音频模型的正确姿势：冻结骨干(pretrained backbone 当特征提取器)，只在上面训一个
 小小的线性分类头。这里把这套真跑出来：① 固定的随机投影当"冻结骨干" ② 提梅尔式频带特征
 ③ 用梯度下降真训一个 softmax 线性头去分辨几个关键词。跑：python3 本文件(无需联网/GPU)。
 文末 real_pipeline() 是需联网的真实写法(默认不跑)。
"""
import numpy as np

SR = 8000
KEYWORDS = {"yes": 400.0, "no": 800.0, "stop": 1600.0, "go": 3200.0}   # 关键词⇢主频(模拟)


def make_clip(freq, rng):
    t = np.linspace(0, 0.4, int(SR * 0.4), endpoint=False)
    return np.sin(2 * np.pi * freq * t) + 0.3 * rng.standard_normal(len(t))


def band_energy(clip, n_bands=24):
    """低层特征：功率谱分频带能量(取对数，仿梅尔谱的对数压缩)。"""
    power = np.abs(np.fft.rfft(clip * np.hanning(len(clip)))) ** 2
    e = np.array([b.sum() for b in np.array_split(power, n_bands)])
    return np.log(e + 1e-8)


class FrozenBackbone:
    """冻结的"预训练骨干"：一个固定的随机线性投影 + tanh 非线性，训练全程不更新它。"""
    def __init__(self, n_in, n_out, rng):
        self.W = rng.standard_normal((n_in, n_out)) / np.sqrt(n_in)   # 权重固定，模拟预训练权重
    def embed(self, x):
        return np.tanh(band_energy(x) @ self.W)                      # 波形→稳定的嵌入向量


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)


def train_head(X, Y, n_cls, epochs=300, lr=0.5):
    """微调头 = 一层 softmax 分类器，用交叉熵 + 梯度下降训练(骨干冻结，只动这一层)。"""
    W = np.zeros((X.shape[1], n_cls)); b = np.zeros(n_cls)
    for _ in range(epochs):
        P = softmax(X @ W + b)                          # 前向：预测概率
        G = (P - Y) / len(X)                             # softmax+交叉熵的梯度就是 (P - onehot)
        W -= lr * X.T @ G; b -= lr * G.sum(0)            # 只更新头的参数
    return W, b


def real_pipeline():   # 🟡 需联网首次下模型/数据，默认不执行；上面已把"冻结骨干+训头"手写了
    from transformers import pipeline                   # noqa
    from datasets import load_dataset                   # noqa
    ds = load_dataset("speech_commands", "v0.02", split="validation", streaming=True)
    kws = pipeline("audio-classification", model="MIT/ast-finetuned-speech-commands-v2")
    return kws(next(iter(ds))["audio"]["array"])[0]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    labels = list(KEYWORDS)
    backbone = FrozenBackbone(24, 32, rng)
    # 造训练集：每个关键词 30 条带噪样本 → 过冻结骨干拿嵌入
    Xtr, Ytr = [], []
    for i, kw in enumerate(labels):
        for _ in range(30):
            Xtr.append(backbone.embed(make_clip(KEYWORDS[kw], rng)))
            onehot = np.zeros(len(labels)); onehot[i] = 1; Ytr.append(onehot)
    Xtr, Ytr = np.array(Xtr), np.array(Ytr)
    W, b = train_head(Xtr, Ytr, len(labels))

    correct = 0                                          # 在新样本上测微调后的头
    for i, kw in enumerate(labels):
        for _ in range(20):
            p = softmax(backbone.embed(make_clip(KEYWORDS[kw], rng)) @ W + b)
            correct += (labels[p.argmax()] == kw)
    acc = correct / (len(labels) * 20)
    assert acc > 0.9                                     # 冻结骨干+小头就能高准确率
    print(f"✅ 冻结预训练骨干 + 微调线性头跑通：{len(labels)} 个关键词，测试准确率 {acc:.1%}")
    print(f"   骨干参数固定不训、只训头 {W.size + b.size} 个参数 —— 这正是迁移学习省数据/省算力的原因")

# 面试 Q&A：为什么微调常只训头、冻结骨干？预训练骨干已学到通用音频表征，下游数据少时全量微调
# 易过拟合且贵；冻结骨干只训一个小分类头，参数少、收敛快、泛化好，是低资源场景的首选。
