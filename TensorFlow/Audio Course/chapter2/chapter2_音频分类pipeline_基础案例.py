"""
 Audio Course · Ch2 · 基础案例：从零实现"音频分类 pipeline"的内核（纯 numpy，无需联网）
 一行 pipeline("audio-classification") 背后其实是三段：① 提特征(把波形变成向量)
 ② 打分(和各类原型比距离/算 logits) ③ softmax 归一 + 取 argmax。这里全手写，
 造几段不同频率的合成音当"意图类别"，训练最近原型分类器并像真 pipeline 那样输出 label+score。
 跑：python3 本文件。文末 real_pipeline() 是需联网/GPU 的真实写法(默认不跑)。
"""
import numpy as np

SR = 8000
CLASSES = {"低频·转账": 300.0, "中频·查询": 900.0, "高频·投诉": 2500.0}   # 频率⇢意图(模拟)


def make_clip(freq, noise=0.0, rng=None):
    """合成一段带噪音频：正弦主频 + 高斯噪声，模拟同类样本的自然差异。"""
    rng = rng or np.random.default_rng()
    t = np.linspace(0, 0.5, int(SR * 0.5), endpoint=False)
    return np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(len(t))


def features(clip, n_bands=16):
    """特征提取：功率谱按频带分桶求能量 → 归一化。这是"波形→定长向量"的最小可用做法。"""
    power = np.abs(np.fft.rfft(clip * np.hanning(len(clip)))) ** 2
    bands = np.array([b.sum() for b in np.array_split(power, n_bands)])
    return bands / (bands.sum() + 1e-12)                    # L1 归一，消除音量差异


def train_prototypes(rng):
    """"训练" = 每类多个样本的特征取均值当原型(nearest-centroid，真实可用的轻量分类器)。"""
    proto = {}
    for label, freq in CLASSES.items():
        feats = [features(make_clip(freq, noise=0.3, rng=rng)) for _ in range(20)]
        proto[label] = np.mean(feats, axis=0)
    return proto


def classify(clip, proto):
    """打分+softmax：用"负距离"当 logits，softmax 成概率，返回按分数排序的预测(仿 pipeline 输出)。"""
    labels = list(proto)
    x = features(clip)
    logits = np.array([-np.linalg.norm(x - proto[l]) for l in labels])   # 越近 logit 越大
    p = np.exp(logits - logits.max()); p /= p.sum()                      # 数值稳定 softmax
    order = np.argsort(p)[::-1]
    return [{"label": labels[i], "score": float(p[i])} for i in order]


def real_pipeline():   # 🟡 需联网首次下模型/数据，默认不执行；上面已把它的内核纯手写
    from transformers import pipeline                       # noqa
    from datasets import Audio, load_dataset                # noqa
    minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
    clf = pipeline("audio-classification", model="anton-l/xtreme_s_xlsr_300m_minds14")
    return clf(minds[0]["audio"]["array"])[0]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    proto = train_prototypes(rng)
    correct = 0
    for true_label, freq in CLASSES.items():                # 各类取新样本测准确率
        pred = classify(make_clip(freq, noise=0.3, rng=rng), proto)
        correct += (pred[0]["label"] == true_label)
        print(f"   真实={true_label:8s} → 预测={pred[0]['label']:8s} score={pred[0]['score']:.3f}")
    assert correct == len(CLASSES)                          # 3/3 全对才算跑通
    print(f"✅ 手写音频分类内核跑通：{correct}/{len(CLASSES)} 全对(提特征→比原型→softmax→argmax)")

# 面试 Q&A：pipeline 一行背后做了什么？前处理(重采样/提特征)→模型前向拿 logits→
# softmax 归一→按分数排序取 top-k label。这里用最近原型代替深度模型，但"特征→打分→归一"
# 这套骨架和真实 audio-classification 完全一致。
