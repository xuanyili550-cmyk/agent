"""
================================================================================
 Audio Course · Chapter 6 · 文本转语音(TTS)数据集（学习笔记 · 本地音频 I/O 可跑）
================================================================================
 一句话：做 TTS 先懂数据(单人/多人、单语/多语、采样率),再懂音频怎么读写(wav I/O)。
 本章讲：
   ① 常用 TTS 数据集特点：LJSpeech(单人英语基准)、VCTK(110 口音多说话人)、
      LibriTTS(24kHz 多人)、Multilingual LibriSpeech(多语种)。多人/多口音→语音更自然多样。
   ② 音频 I/O(本地真跑)：用 numpy 合成正弦波 → soundfile 写 wav → 读回校验。TTS 输出的就是这种波形。
   ③ 采样率的意义：越高越清晰但文件/算力越大(LJSpeech 22.05k, LibriTTS 24k)。
 要点：TTS 训练=文本→梅尔谱(声学模型)→波形(声码器);数据的说话人/采样率直接决定音质与多样性。
 跑：python3 chapter6_TTS数据集_学习笔记.py   （② 段本地真跑,无需联网)
================================================================================
"""
import os
import numpy as np
import soundfile as sf

OUT = "/tmp/audio_ch6"
os.makedirs(OUT, exist_ok=True)


# ② 音频 I/O：合成一个 440Hz 正弦(A4 音) → 写 wav → 读回
def synth_and_io(sr=22050, dur=1.0, freq=440.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)   # 幅度 0.3 防削顶
    path = os.path.join(OUT, "sine_a4.wav")
    sf.write(path, wave, sr)                                       # 写 wav
    back, sr2 = sf.read(path)                                      # 读回
    return path, sr2, len(back)


# ① 数据集加载(🟡 需联网/占空间,默认不执行,仅示范写法)
def load_ljspeech_example():
    from datasets import load_dataset
    ds = load_dataset("lj_speech", split="train", streaming=True)
    return next(iter(ds))          # {'audio', 'text', ...}


def main():
    path, sr, n = synth_and_io()
    assert sr == 22050 and n == 22050
    print(f"✅ Ch6 跑通：合成 440Hz 正弦 → 写读 wav({sr}Hz, {n} 样本, {n/sr:.1f}s) → {path}")
    print("   TTS 数据集选型：单人基准 LJSpeech / 多口音 VCTK / 24k 多人 LibriTTS / 多语种 MLS")
    # 面试：Q TTS 两阶段? A 文本→梅尔谱(声学模型)→波形(声码器如 HiFi-GAN); Q 多说话人数据的价值? A 音色/口音更丰富。


if __name__ == "__main__":
    main()
