"""
================================================================================
 Audio Course · Chapter 2 · 用 pipeline 做音频分类 / ASR（学习笔记 · 🟡首次下模型)
================================================================================
 一句话：一行 pipeline 跑通"音频→标签"(分类)和"音频→文本"(语音识别)。
 本章讲：
   ① 数据：MINDS-14(多语种银行客服意图)，cast 到 16kHz(必须对齐模型采样率)。
   ② 音频分类：pipeline("audio-classification") → 意图标签。
   ③ ASR：pipeline("automatic-speech-recognition") → 文本。
 要点：★采样率不匹配是音频最常见的坑——喂模型前先重采样到它训练时的采样率(这里 16kHz)。
 跑：python3 chapter2_音频分类pipeline_学习笔记.py   （🟡 首次会下数据集+模型，需联网）
================================================================================
"""
from transformers import pipeline


def run():
    from datasets import Audio, load_dataset
    # ① 数据：加载并统一到 16kHz
    minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
    sample = minds[0]

    # ② 音频分类
    clf = pipeline("audio-classification", model="anton-l/xtreme_s_xlsr_300m_minds14")
    pred = clf(sample["audio"]["array"])
    print("  [分类] top:", pred[0])

    # ③ ASR
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    text = asr(sample["audio"]["array"])["text"]
    print("  [ASR ]", text.strip())
    print("✅ Ch2 跑通：音频分类 + ASR(均通过 pipeline)")


if __name__ == "__main__":
    run()   # 🟡 首次下载 MINDS-14 + 两个模型，耗时/占空间；离线环境会失败
