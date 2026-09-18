"""
 Audio Course · Ch6 · 基础案例：从零构造 TTS 数据集样本并算特征（纯 numpy + stdlib wave）
 TTS 训练要的是成对样本 (文本/音素 → 目标梅尔谱 → 波形)。这里不装 soundfile，直接用标准库
 wave 把 16-bit PCM 波形写盘再读回(round-trip 验证无损)，并手算目标梅尔谱当训练标签，
 最后打印数据集统计(时长/采样率一致性)。跑：python3 本文件(无需联网，只用 numpy + 标准库)。
"""
import os
import wave
import numpy as np

SR = 16000
# 极简"发音词典"：把音素映射到共振峰式的主频组合，串起来当作会说话的目标波形
PHONEMES = {"a": [700, 1200], "i": [300, 2500], "u": [350, 800], "s": [4000]}
DATASET = [("sa", "sa"), ("si", "si"), ("us", "us"), ("ai", "ai")]     # (文本, 音素串)


def synth_speech(phoneme_str, dur_each=0.12):
    """按音素串把各共振峰主频叠加合成语音波形 —— 模拟 TTS 声学模型的目标输出。"""
    segs = []
    for ph in phoneme_str:
        t = np.linspace(0, dur_each, int(SR * dur_each), endpoint=False)
        wave_seg = sum(np.sin(2 * np.pi * f * t) for f in PHONEMES[ph]) / len(PHONEMES[ph])
        segs.append(0.3 * wave_seg)
    return np.concatenate(segs).astype(np.float32)


def write_wav(path, y, sr=SR):
    """用标准库 wave 写 16-bit PCM：float[-1,1] → int16。这就是 wav 文件真实的字节结构。"""
    pcm = np.clip(y, -1, 1)
    pcm = (pcm * 32767).astype("<i2")                       # 小端 16 位有符号整数
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    """读回 wav 并还原成 float 波形。"""
    with wave.open(path, "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype="<i2")
    return pcm.astype(np.float32) / 32767, sr


def mel_target(y, n_fft=512, hop=256, n_mels=40):
    """手算对数梅尔谱当 TTS 训练标签(声学模型预测的就是它，再由声码器转回波形)。"""
    win = np.hanning(n_fft)
    frames = [y[i:i + n_fft] * win for i in range(0, len(y) - n_fft + 1, hop)]
    power = (np.abs(np.fft.rfft(np.stack(frames), axis=1)) ** 2).T
    # 线性→梅尔的简化三角滤波器组
    hz = np.fft.rfftfreq(n_fft, 1 / SR)
    mel_edges = np.linspace(0, 2595 * np.log10(1 + SR / 2 / 700), n_mels + 2)
    centers = 700 * (10 ** (mel_edges / 2595) - 1)
    fb = np.maximum(0, 1 - np.abs(hz[None, :] - centers[1:-1, None]) / 300)
    return np.log(fb @ power + 1e-8)


if __name__ == "__main__":
    os.makedirs("/tmp/audio_ch6", exist_ok=True)
    durations = []
    for text, phon in DATASET:
        y = synth_speech(phon)
        path = f"/tmp/audio_ch6/{text}.wav"
        write_wav(path, y)
        back, sr2 = read_wav(path)
        assert sr2 == SR and len(back) == len(y)             # 采样率、长度一致
        assert np.max(np.abs(back - y)) < 1e-3               # 16-bit round-trip 近乎无损
        mel = mel_target(y)
        durations.append(len(y) / SR)
        print(f"   样本「{text}」音素{list(phon)} → {len(y)}样本/{len(y)/SR:.2f}s，梅尔标签 shape={mel.shape}")
    assert len(set(round(d, 2) for d in durations)) == 1     # 数据集时长一致(便于成 batch)
    print(f"✅ TTS 数据集跑通：{len(DATASET)} 条成对样本，wav 写读无损，梅尔标签已生成；均 {durations[0]:.2f}s")

# 面试 Q&A：TTS 声学模型的训练标签为何是梅尔谱而不是原始波形？波形采样点太多太长(16kHz)、
# 直接回归极难；梅尔谱更紧凑且贴近感知，声学模型先预测梅尔谱，再交给声码器(vocoder)还原波形，
# 这种"两段式"大幅降低了建模难度。
