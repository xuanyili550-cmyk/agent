"""
 Audio Course · Ch5 · 基础案例：Whisper 语音转文字（🟡 首次下模型/数据,需联网）
 跑：python3 本文件
"""
from transformers import pipeline
from datasets import load_dataset

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
print("✅ 识别文本:", asr(ds[0]["audio"]["array"])["text"].strip())
