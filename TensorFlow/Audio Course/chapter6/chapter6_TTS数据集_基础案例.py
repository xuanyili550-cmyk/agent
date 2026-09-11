"""
 Audio Course · Ch6 · 基础案例：合成正弦波并写成 wav（本地可跑,无需联网）
 目的：理解"音频=采样率+一串幅度",TTS 最终输出的就是这种波形。跑：python3 本文件
"""
import os
import numpy as np
import soundfile as sf

sr, dur, freq = 16000, 1.0, 440.0
t = np.linspace(0, dur, int(sr * dur), endpoint=False)
wave = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

os.makedirs("/tmp/audio_ch6", exist_ok=True)
sf.write("/tmp/audio_ch6/case_sine.wav", wave, sr)
back, sr2 = sf.read("/tmp/audio_ch6/case_sine.wav")
assert sr2 == sr and len(back) == sr
print(f"✅ 写读 wav 成功：{sr}Hz, {len(back)} 样本, {len(back)/sr:.1f}s → /tmp/audio_ch6/case_sine.wav")
