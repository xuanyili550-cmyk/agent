"""
 Audio Course · Ch7 · 基础案例：Whisper 把外语语音翻成英文（🟡 首次下模型/数据,需联网）
 跑：python3 本文件
"""
from transformers import pipeline
from datasets import load_dataset

asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
ds = load_dataset("facebook/voxpopuli", "it", split="validation", streaming=True)
sample = next(iter(ds))
en = asr(sample["audio"]["array"], generate_kwargs={"task": "translate"})["text"]
print("✅ 翻译(→英文):", en.strip())
