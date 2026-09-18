"""
 Audio Course · Ch1 · 基础案例：从零手写梅尔频谱（音频模型最常吃的输入）
 为什么不直接调 librosa？因为梅尔谱的"真机制"就三步：① 分帧加窗 ② 每帧做 DFT 取功率(STFT)
 ③ 用一组三角"梅尔滤波器"把线性频率压成贴近人耳的梅尔刻度。这里全用 numpy 把这三步算出来，
 而不是喂给黑盒。跑：python3 本文件（纯 numpy，无需联网/GPU）
"""
import numpy as np


def stft_power(y, n_fft=256, hop=128):
    """短时傅里叶：把长信号切成一帧帧、各自加汉宁窗再做实数 FFT，取功率谱。"""
    win = np.hanning(n_fft)                                  # 汉宁窗：减少分帧边界的频谱泄漏
    frames = [y[i:i + n_fft] * win                          # 逐帧滑动，步长 hop（帧间重叠）
              for i in range(0, len(y) - n_fft + 1, hop)]
    spec = np.fft.rfft(np.stack(frames), axis=1)            # 每帧 DFT，只取非冗余的正频率一半
    return (np.abs(spec) ** 2).T                            # 功率 = |复数幅度|²，转成 [频点, 帧]


def hz_to_mel(f):  return 2595.0 * np.log10(1.0 + f / 700.0)     # 梅尔刻度：低频分得细、高频粗
def mel_to_hz(m):  return 700.0 * (10.0 ** (m / 2595.0) - 1.0)   # 逆变换，用来定滤波器中心频率


def mel_filterbank(sr, n_fft, n_mels):
    """造 n_mels 个三角滤波器：在梅尔轴上等距取中心，映回 Hz 后落到 FFT 频点上。"""
    n_bins = n_fft // 2 + 1
    mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)   # 首尾各多一个做三角边界
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)              # Hz→对应的 FFT 频点下标
    fb = np.zeros((n_mels, n_bins))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):  fb[m - 1, k] = (k - l) / max(c - l, 1)   # 上升沿
        for k in range(c, r):  fb[m - 1, k] = (r - k) / max(r - c, 1)   # 下降沿
    return fb


if __name__ == "__main__":
    sr, n_fft, n_mels = 16000, 256, 40
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.6 * np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t)  # 1k+3k 双音

    power = stft_power(y, n_fft)                             # STFT 功率谱 [频点, 帧]
    fb = mel_filterbank(sr, n_fft, n_mels)                   # 梅尔滤波器组 [梅尔带, 频点]
    mel = fb @ power                                         # 矩阵乘 = 每个梅尔带对功率加权求和
    mel_db = 10 * np.log10(mel + 1e-10)                      # 转 dB（贴近人耳的对数响度）

    assert mel.shape[0] == n_mels                            # 输出确实是 40 个梅尔带
    peak_hz = np.fft.rfftfreq(n_fft, 1 / sr)[power.mean(1).argmax()]   # 能量最强频点应≈1000Hz
    assert abs(peak_hz - 1000) < sr / n_fft                  # 误差在一个频点分辨率内
    print(f"✅ 手写梅尔谱 shape={mel_db.shape}(梅尔带, 帧)；主频检出 {peak_hz:.0f}Hz(真实 1000Hz)")
    print(f"   滤波器组每行和均>0：{np.all(fb.sum(1) > 0)}；dB 范围[{mel_db.min():.0f},{mel_db.max():.0f}]")

# 面试 Q&A：为何用梅尔刻度而非线性 Hz？人耳对低频分辨强、高频弱，梅尔在低频密采、高频疏采，
# 用更少的带就抓住感知上重要的信息，是所有音频模型(Whisper/wav2vec)前端特征的标准做法。
