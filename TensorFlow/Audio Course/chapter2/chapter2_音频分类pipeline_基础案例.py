"""
 Audio Course · Ch2 · 基础案例：一行 pipeline 做音频分类（🟡 首次下模型/数据，需联网）
 跑：python3 本文件
"""
from transformers import pipeline
from datasets import Audio, load_dataset

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))   # ★对齐 16kHz

clf = pipeline("audio-classification", model="anton-l/xtreme_s_xlsr_300m_minds14")
pred = clf(minds[0]["audio"]["array"])
print("✅ 预测意图:", pred[0])   # {'score':..., 'label':...}
