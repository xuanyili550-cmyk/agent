"""LLM 系统评测:准确率/召回命中/幻觉率 + LLM-as-judge(stub 规则,离线)。对应生产问题手册第9节。
★ 拒答(说"未提及/不知道")是安全行为,不计入幻觉。"""
from ..core.exceptions import ValidationError
_REFUSAL = ("未提及", "不知道", "无法回答", "没有相关")


def _is_refusal(a):
    return any(r in (a or "") for r in _REFUSAL)


def _supported(answer, context):
    core = (answer or "").replace(" ", "")
    return any(core[i:i + 2] in (context or "") for i in range(max(1, len(core) - 1)))


def judge(q, answer, reference, context):
    score = 5
    if reference and reference not in answer and not _is_refusal(answer): score -= 2
    if context and not _is_refusal(answer) and not _supported(answer, context): score -= 3
    return max(0, score)


def evaluate(cases: list[dict]):
    if not cases: raise ValidationError("cases 不能为空")
    n = len(cases); acc = halluc = 0; scores = []
    for c in cases:
        a, ref, ctx = c.get("answer", ""), c.get("reference", ""), c.get("context", "")
        if ref and ref in a: acc += 1
        if ctx and not _is_refusal(a) and not _supported(a, ctx): halluc += 1   # 拒答不算幻觉
        scores.append(judge(c.get("query", ""), a, ref, ctx))
    return {"n": n, "accuracy": round(acc / n, 3), "hallucination_rate": round(halluc / n, 3),
            "avg_judge_score": round(sum(scores) / n, 2)}
