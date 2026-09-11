"""
================================================================================
 Audio Course · Chapter 1 · 音频波形 / 频谱 / 频谱图（学习笔记 · 可运行真代码）
================================================================================
 一句话：搞懂"声音在计算机里长什么样"——时域波形、频域频谱、时频梅尔谱，是所有音频任务的地基。
 本章讲(每步都真算+存图到 /tmp/audio_ch1/)：
   ① 波形(waveform)：采样率 sr + 一串幅度值[-1,1]；len(y)/sr = 时长。
   ② 频谱(spectrum)：取一小段加窗 → DFT(rfft) → 各频率强度(dB)。看"有哪些频率成分"。
   ③ 梅尔频谱图(mel spectrogram)：x=时间 y=梅尔频率；模型最常吃的输入表示。
 要点：采样率决定能表示的最高频(奈奎斯特=sr/2)；梅尔刻度贴近人耳听感。
 跑：python3 chapter1_音频波形与频谱_学习笔记.py   （用 librosa 内置 trumpet，无需联网）
================================================================================
"""
import os
import matplotlib
matplotlib.use("Agg")                       # 无界面后端：只存图不弹窗(服务器/CI 友好)
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

OUT = "/tmp/audio_ch1"
os.makedirs(OUT, exist_ok=True)


def load_demo():
    y, sr = librosa.load(librosa.ex("trumpet"))   # 返回(幅度序列, 采样率)
    return y, sr


# ① 波形：幅度随时间变化
def show_waveform(y, sr):
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("waveform"); plt.tight_layout(); plt.savefig(f"{OUT}/1_waveform.png"); plt.close()
    return {"sr": sr, "samples": len(y), "dur_s": round(len(y) / sr, 2),
            "min": round(float(y.min()), 3), "max": round(float(y.max()), 3)}


# ② 频谱：一小段加汉宁窗 → DFT → dB 幅度谱
def show_spectrum(y, sr, n=4096):
    seg = y[:n] * np.hanning(n)                    # 加窗减少频谱泄漏
    amp = np.abs(np.fft.rfft(seg))                 # 实数 FFT → 幅度
    amp_db = librosa.amplitude_to_db(amp, ref=np.max)
    freq = librosa.fft_frequencies(sr=sr, n_fft=n)
    plt.figure(figsize=(12, 3)); plt.plot(freq, amp_db)
    plt.xscale("log"); plt.xlabel("Hz"); plt.ylabel("dB")
    plt.tight_layout(); plt.savefig(f"{OUT}/2_spectrum.png"); plt.close()
    return amp_db.shape


# ③ 梅尔频谱图：时频表示(模型输入)
def show_melspectrogram(y, sr, n_mels=64):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)      # 功率转 dB
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB"); plt.tight_layout()
    plt.savefig(f"{OUT}/3_melspectrogram.png"); plt.close()
    return S_db.shape                              # (n_mels, 帧数)


def main():
    y, sr = load_demo()
    w = show_waveform(y, sr)
    sp = show_spectrum(y, sr)
    m = show_melspectrogram(y, sr)
    assert sr > 0 and len(y) > 0 and m[0] == 64    # 梅尔谱第一维=n_mels
    print(f"✅ Ch1 跑通：波形{w} · 频谱{sp} · 梅尔谱{m}；图已存 {OUT}/")
    # 面试：Q 采样率决定什么? A 能表示的最高频(奈奎斯特 sr/2); Q 为什么用梅尔? A 贴近人耳非线性听感。


if __name__ == "__main__":
    main()
