"""
 Audio Course · Ch3 · 基础案例：手写 CTC 解码折叠（看懂"仅编码器 ASR"的输出怎么变成文本）
 CTC(连接主义时间分类)：模型对每一帧输出一个字符(或空白 '-')；解码时"先去重复、再删空白"。
 跑：python3 本文件（纯 python，无需联网）
"""


def ctc_collapse(frames: str, blank: str = "-") -> str:
    """把逐帧预测折叠成文本：① 合并连续相同 ② 删空白。这是 CTC 贪心解码的核心。"""
    out = []
    prev = None
    for ch in frames:
        if ch != prev:            # ① 只在"变化时"取字符 → 去掉连续重复
            if ch != blank:       # ② 空白不计入
                out.append(ch)
        prev = ch
    return "".join(out)


# 逐帧预测(如 CTC 头对每帧 argmax 的结果)："hh-e-ll-lloo" → "hello"
demo = "hh-e-ll-lloo"
result = ctc_collapse(demo)
assert result == "hello"
# 注意："ll"中间要有空白或字符隔开才会保留两个 l —— 这就是 CTC 用 blank 区分"真重复"的原因
assert ctc_collapse("aa--aa") == "aa"      # a-a 被 blank 隔开 → 两个 a
assert ctc_collapse("aaaa") == "a"          # 连续无 blank → 折叠成一个 a
print(f"✅ CTC 折叠：'{demo}' → '{result}'；'aa--aa'→'aa'，'aaaa'→'a'(blank 用来区分真重复)")
