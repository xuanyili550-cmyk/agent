"""
 Audio Course · Ch5 · 基础案例：从零手写 DTW 做语音识别（纯 numpy，无需联网/GPU）
 ASR 要解决的核心难题是"时间对齐"：同一个词说快说慢、拖长短不一，波形长度都不同。
 DTW(动态时间规整)用动态规划，允许时间轴弹性拉伸，找两条特征序列的最优对齐代价 —— 这是
 Whisper 之前经典 ASR/关键词检索的基石。这里造几个词模板，对一条测试语音做 DTW 最近模板识别。
 跑：python3 本文件。文末 real_pipeline() 是 Whisper 真实写法(需联网,默认不跑)。
"""
import numpy as np

SR = 8000
# 每个"词"= 一串随时间变化的主频序列(模拟不同发音的音高轨迹)
WORDS = {"hello": [300, 600, 900], "world": [1200, 800, 400], "ok": [500, 500]}


def synth(freq_track, rate=1.0, rng=None):
    """按主频轨迹合成语音，rate 控制语速(拉伸/压缩时长)—— 制造 DTW 要对齐的时间形变。"""
    segs = []
    for f in freq_track:
        n = int(SR * 0.15 * rate)                           # 每段时长随 rate 变化
        t = np.linspace(0, 0.15 * rate, n, endpoint=False)
        segs.append(np.sin(2 * np.pi * f * t))
    y = np.concatenate(segs)
    if rng is not None:
        y = y + 0.05 * rng.standard_normal(len(y))
    return y


def frames_feature(y, n_fft=256, hop=128):
    """把波形切帧，每帧取"主频"当特征序列 [帧数] —— DTW 就在这种时间序列上对齐。"""
    feats = []
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    for i in range(0, len(y) - n_fft + 1, hop):
        spec = np.abs(np.fft.rfft(y[i:i + n_fft] * np.hanning(n_fft)))
        feats.append(freqs[spec.argmax()])                  # 该帧能量最强的频率
    return np.array(feats)


def dtw_distance(a, b):
    """动态规划求 DTW 距离：D[i,j]=局部代价 + min(左, 上, 左上)，允许时间轴非线性对齐。"""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])                 # 两帧特征的局部距离
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)                                 # 归一化，消除序列长度差异


def recognize(y, templates):
    """识别 = 对每个词模板算 DTW 距离，取距离最小者(最近邻，经典模板匹配 ASR)。"""
    feat = frames_feature(y)
    dists = {w: dtw_distance(feat, t) for w, t in templates.items()}
    return min(dists, key=dists.get), dists


def real_pipeline():   # 🟡 Whisper 真实写法，需联网首次下模型；上面已把"时间对齐"内核手写
    from transformers import pipeline                       # noqa
    from datasets import load_dataset                       # noqa
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    return asr(ds[0]["audio"]["array"])["text"].strip()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    templates = {w: frames_feature(synth(track)) for w, track in WORDS.items()}   # 标准语速建模板

    correct = 0
    for w, track in WORDS.items():
        # 测试语音用不同语速 + 噪声，模板却是标准语速 → 逼 DTW 去弹性对齐
        test = synth(track, rate=1.6, rng=rng)
        pred, dists = recognize(test, templates)
        correct += (pred == w)
        print(f"   说「{w}」(1.6x语速) → 识别为「{pred}」  距离={ {k: round(float(v),1) for k,v in dists.items()} }")
    assert correct == len(WORDS)                            # 变速下仍全对，证明 DTW 真的在对齐时间
    print(f"✅ 手写 DTW 语音识别跑通：{correct}/{len(WORDS)} 全对，变速语音也能对上标准模板")

# 面试 Q&A：DTW 解决了 ASR 的什么问题？语速差异导致同词波形长度不同，欧氏距离无法直接比；
# DTW 用动态规划允许时间轴伸缩地一一对应，找到最优对齐路径与代价，从而做到变速鲁棒的匹配。
