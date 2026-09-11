"""
================================================================================
 Audio Course · Chapter 7 · 语音到语音翻译(STST)（学习笔记 · 🟡首次下模型)
================================================================================
 一句话：把一种语言的语音翻成另一种语言的语音——级联式：Whisper(语音→译文) + TTS(译文→语音)。
 本章讲：
   ① ASR 翻译：Whisper 的 generate(task="translate") 直接把外语语音转成英文文本。
   ② TTS 合成：把译文用 TTS 变回语音(声学模型 + 声码器)。
   ③ 级联 S2S 管道：语音 →(Whisper)→ 译文 →(TTS)→ 目标语语音。
 要点：级联简单可控但会累积误差(ASR 错→翻译错);端到端 S2S 更难但少一次转写损失。
 跑：python3 chapter7_语音到语音翻译_学习笔记.py   （🟡 首次下 Whisper+数据,需联网)
================================================================================
"""
import torch
from transformers import pipeline


def main():
    from datasets import load_dataset
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
    ds = load_dataset("facebook/voxpopuli", "it", split="validation", streaming=True)
    sample = next(iter(ds))
    # ① Whisper 翻译：意大利语语音 → 英文文本
    en = asr(sample["audio"]["array"], generate_kwargs={"task": "translate"})["text"]
    print("  [语音→译文]", en.strip())
    # ② 下一步：把 en 送进 TTS 合成英文语音(见本项目 音频制作_双语配音/ 的 TTS)
    print("✅ Ch7 跑通：Whisper 语音翻译(级联 S2S 的第一段)")


if __name__ == "__main__":
    main()   # 🟡 首次下模型/数据
