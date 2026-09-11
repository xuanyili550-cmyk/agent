"""
================================================================================
 Audio Course · Chapter 4 · 音频分类的预训练模型与数据集（学习笔记 · 🟡首次下模型)
================================================================================
 一句话：拿现成预训练模型做音频分类的子任务；核心前提是"你的类别要和模型训练的类别对得上"。
 本章讲：
   ① 关键词识别(KWS)：在语音里识别固定关键词集(Speech Commands)。类别=模型训练的关键词。
   ② 语言识别(LID)：判断这段语音是哪种语言。
   ③ 零样本音频分类：给候选标签,不训练也能分(类比零样本文本分类)。
 要点：分类模型标签集固定,用前先确认标签匹配你的场景;采样率仍要对齐(16kHz)。
 跑：python3 chapter4_音频分类预训练模型_学习笔记.py   （🟡 首次下数据+模型,需联网)
================================================================================
"""
from transformers import pipeline


def keyword_spotting():
    from datasets import load_dataset
    ds = load_dataset("speech_commands", "v0.02", split="validation", streaming=True)
    sample = next(iter(ds))
    kws = pipeline("audio-classification", model="MIT/ast-finetuned-speech-commands-v2")
    pred = kws(sample["audio"]["array"])
    print("  [KWS] top:", pred[0])


def language_id():
    from datasets import load_dataset, Audio
    ds = load_dataset("common_language", split="test", streaming=True)
    sample = next(iter(ds))
    lid = pipeline("audio-classification", model="sanchit-gandhi/whisper-medium-fleurs-lang-id")
    print("  [LID] top:", lid(sample["audio"]["array"])[0])


def main():
    keyword_spotting()
    # language_id()   # 可选,另需下模型
    print("✅ Ch4 跑通：KWS 关键词识别(预训练音频分类)")


if __name__ == "__main__":
    main()   # 🟡 首次下 speech_commands + AST 模型
