"""
 Audio Course · Ch7 · 基础案例：从零搭"语音→语音翻译"级联管线（纯 numpy + stdlib，无需联网）
 S2ST(speech-to-speech translation)最常见的实现是三段级联：① ASR 把源语音识别成源词
 ② MT 把源词翻成目标词 ③ TTS 把目标词合成成目标语音。这里三段都真算：识别用频率匹配、
 翻译用词典映射、合成用共振峰叠加，端到端跑通"意大利语音 → 英文语音"。
 跑：python3 本文件。文末 real_pipeline() 是 Whisper translate 真实写法(需联网,默认不跑)。
"""
import numpy as np

SR = 8000
# 源语言(意大利语)每个词 = 一个主频"声音指纹"
IT_WORDS = {"ciao": 400.0, "grazie": 900.0, "sole": 1500.0}
IT2EN = {"ciao": "hello", "grazie": "thanks", "sole": "sun"}            # ② 翻译词典
EN_WORDS = {"hello": [500, 1000], "thanks": [700, 1400], "sun": [600]}  # 目标语音的共振峰


def synth_source(word, rng):
    """把意大利语词合成成带噪语音(主频=该词指纹)。"""
    t = np.linspace(0, 0.5, int(SR * 0.5), endpoint=False)
    return np.sin(2 * np.pi * IT_WORDS[word] * t) + 0.3 * rng.standard_normal(len(t))


def asr_recognize(y):
    """① ASR：取整段主频，匹配最接近的源词指纹(最近邻声学识别)。"""
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    peak = np.fft.rfftfreq(len(y), 1 / SR)[spec.argmax()]
    return min(IT_WORDS, key=lambda w: abs(IT_WORDS[w] - peak))


def translate(src_word):
    """② MT：查词典把源词翻成目标词(真实里是 seq2seq，这里用确定映射)。"""
    return IT2EN[src_word]


def tts_synthesize(en_word):
    """③ TTS：按目标词的共振峰叠加合成目标语音波形。"""
    t = np.linspace(0, 0.5, int(SR * 0.5), endpoint=False)
    formants = EN_WORDS[en_word]
    return sum(np.sin(2 * np.pi * f * t) for f in formants) / len(formants)


def s2st(y):
    """级联管线：语音 → (ASR) 源词 → (MT) 目标词 → (TTS) 目标语音。"""
    src = asr_recognize(y)
    tgt = translate(src)
    out_wave = tts_synthesize(tgt)
    return src, tgt, out_wave


def real_pipeline():   # 🟡 Whisper 直接 speech→英文文本，需联网首次下模型；上面把三段级联手写了
    from transformers import pipeline                       # noqa
    from datasets import load_dataset                       # noqa
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    ds = load_dataset("facebook/voxpopuli", "it", split="validation", streaming=True)
    sample = next(iter(ds))
    return asr(sample["audio"]["array"], generate_kwargs={"task": "translate"})["text"].strip()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ok = 0
    for it_word, expect_en in IT2EN.items():
        y = synth_source(it_word, rng)
        src, tgt, out = s2st(y)
        # 验证末端语音真的携带了目标词的信息：反解合成语音主频，应落在目标共振峰内
        out_peak = np.fft.rfftfreq(len(out), 1 / SR)[np.abs(np.fft.rfft(out)).argmax()]
        carries = min(abs(out_peak - f) for f in EN_WORDS[tgt]) < SR / len(out)
        ok += (src == it_word and tgt == expect_en and carries)
        print(f"   听到「{src}」(it) → 翻译「{tgt}」(en) → 合成语音主频≈{out_peak:.0f}Hz  {'✓' if carries else '✗'}")
    assert ok == len(IT2EN)                                 # 三段级联端到端全对
    print(f"✅ 语音→语音翻译级联跑通：{ok}/{len(IT2EN)} 全对(ASR 识别→MT 翻译→TTS 合成，各段真算)")

# 面试 Q&A：级联式 S2ST 有什么优缺点？优点是每段可独立训练/替换、可解释、能复用现成 ASR/MT/TTS；
# 缺点是误差会逐级累积(ASR 错→翻译错)、丢失语音的韵律/情感、延迟叠加。端到端直翻模型正是为
# 缓解这些问题而生，但对数据和算力要求更高。
