"""
 Audio Course · Ch1 · 基础案例：一段音频 → 梅尔频谱图（模型最常吃的输入）
 目标：最小代码走通"读音频 → 算梅尔谱 → 看形状/存图"。跑：python3 本文件（无需联网）
"""
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load(librosa.ex("trumpet"))                     # 读内置小号音频
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)     # 梅尔频谱(功率)
mel_db = librosa.power_to_db(mel, ref=np.max)                   # 转 dB 便于观察/喂模型

os.makedirs("/tmp/audio_ch1", exist_ok=True)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB"); plt.tight_layout()
plt.savefig("/tmp/audio_ch1/mel_case.png"); plt.close()

assert mel_db.shape[0] == 64
print(f"✅ 梅尔谱 shape = {mel_db.shape}  (n_mels=64, 时间帧数)；图存 /tmp/audio_ch1/mel_case.png")
