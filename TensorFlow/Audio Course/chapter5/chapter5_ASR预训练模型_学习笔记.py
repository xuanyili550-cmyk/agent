"""
================================================================================
 Audio Course · Chapter 5 · ASR 预训练模型：CTC vs Seq2Seq（学习笔记 · 🟡首次下模型)
================================================================================
 一句话：语音识别两大范式——CTC(仅编码器,逐帧对齐) vs Seq2Seq(编解码+交叉注意力,如 Whisper)。
 本章讲：
   ① CTC(Wav2Vec2)：仅编码器 + 线性 CTC 头,逐帧预测再折叠(见 ch3 案例)。快,但不建模输出语言。
   ② Seq2Seq(Whisper)：编码器听懂 + 解码器生成文本,自带标点/多语种/更稳。
   ③ 用 pipeline 各跑一个,对比输出。
 要点：要又快又糙用 CTC;要准/带标点/多语种用 Whisper。都要 16kHz 输入。
 跑：python3 chapter5_ASR预训练模型_学习笔记.py   （🟡 首次下 librispeech + 两模型,需联网)
================================================================================
"""
from transformers import pipeline


def main():
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = ds[0]["audio"]["array"]
    ref = ds[0]["text"]

    ctc = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    s2s = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    print("  参考:", ref)
    print("  [CTC  Wav2Vec2]", ctc(audio)["text"])
    print("  [Seq2Seq Whisper]", s2s(audio)["text"].strip())
    print("✅ Ch5 跑通：CTC vs Seq2Seq 两种 ASR 对比")


if __name__ == "__main__":
    main()   # 🟡 首次下模型/数据
