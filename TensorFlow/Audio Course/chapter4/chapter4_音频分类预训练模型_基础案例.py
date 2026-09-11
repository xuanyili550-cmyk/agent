"""
 Audio Course · Ch4 · 基础案例：关键词识别 KWS（🟡 首次下模型/数据,需联网）
 跑：python3 本文件
"""
from transformers import pipeline
from datasets import load_dataset

ds = load_dataset("speech_commands", "v0.02", split="validation", streaming=True)
sample = next(iter(ds))
kws = pipeline("audio-classification", model="MIT/ast-finetuned-speech-commands-v2")
print("✅ 识别到的关键词:", kws(sample["audio"]["array"])[0])   # {'score','label'}
