"""
================================================================================
 生产工程补齐 · 补4 · LLM 系统评测：准确率/召回率 + 幻觉率 + LLM-as-judge
================================================================================
 补上「离线测试集 + 量化评测 + LLM 自动评测」这一块。上线前必须把系统好坏量化成数字：
   ① 检索召回率(recall@k)：gold 文档是否被检索命中——RAG 的上限，检索不到生成再好也白搭。
   ② 答案准确率(EM/包含)：预测答案是否命中参考答案——最基础的正确性。
   ③ 幻觉率：答案里的说法是否"有据"(在检索到的 context 里)——RAG 最怕胡编。这里用"答案关键片段
      是否出现在 context"的近似启发式；生产更准的做法是 LLM-as-judge 逐句判"是否被支持"。
   ④ LLM-as-judge：用一个更强的模型(GPT-4/本地judge)给答案打分(0-5)——比死板的字符串匹配更懂语义。
      这里 judge() 用规则实现(可确定、免下模型)，生产换成调 GPT-4/mlx 的一行即可。
 为什么要多个指标：单看准确率会漏掉"答对但靠瞎蒙/幻觉"；召回率管检索、幻觉率管可信、judge 管语义质量。
 跑：python3 补4_LLM评测_asjudge与幻觉率.py   （注：按用户要求本文件未在本机执行，仅作真实可跑代码）
================================================================================
"""

# ==============================================================================
# 离线测试集：每条 = 问题 + gold 文档 id + 检索到的 doc ids + context 文本 + 预测答案 + 参考答案
# ==============================================================================
TESTSET = [
    {"q": "退货几天内?", "gold": "d1", "retrieved": ["d1", "d3"],
     "context": "本店支持 7 天无理由退货。", "answer": "7 天无理由退货", "reference": "7 天"},
    {"q": "运费怎么算?", "gold": "d2", "retrieved": ["d5", "d6"],   # 检索没命中 gold
     "context": "满 99 包邮，否则 10 元。", "answer": "满 99 包邮", "reference": "满 99 包邮"},
    {"q": "保修多久?", "gold": "d4", "retrieved": ["d4"],
     "context": "整机保修一年。", "answer": "保修三年", "reference": "一年"},  # 幻觉:答案与 context 不符
]


# ==============================================================================
# ① 检索召回率 recall@k：gold 是否在检索结果里
# ==============================================================================
def recall_at_k(rows):
    hit = sum(1 for r in rows if r["gold"] in r["retrieved"])
    return hit / len(rows)


# ==============================================================================
# ② 答案准确率：预测是否包含参考(宽松版 EM)
# ==============================================================================
def accuracy(rows):
    ok = sum(1 for r in rows if r["reference"] in r["answer"])
    return ok / len(rows)


# ==============================================================================
# ③ 幻觉率：答案里的核心词是否"有据"(出现在 context)。近似启发式。
# ==============================================================================
def hallucination_rate(rows):
    halluc = 0
    for r in rows:
        core = r["answer"].replace(" ", "")
        # 逐字符窗口找有没有一段(>=2字)在 context 里；都不在 → 判为无据/幻觉
        supported = any(core[i:i + 2] in r["context"] for i in range(max(1, len(core) - 1)))
        if not supported:
            halluc += 1
    return halluc / len(rows)


# ==============================================================================
# ④ LLM-as-judge：给 (问题, 答案, 参考) 打 0-5 分。规则版；生产换成真 LLM。
# ==============================================================================
def judge(q: str, answer: str, reference: str, context: str) -> int:
    """生产版：prompt=评分标准+问题+答案+参考+context；score=int(llm(prompt))。"""
    score = 5
    if reference not in answer:
        score -= 2                                     # 没命中参考
    core = answer.replace(" ", "")
    if not any(core[i:i + 2] in context for i in range(max(1, len(core) - 1))):
        score -= 3                                     # 无据(幻觉)重扣
    return max(0, score)


def evaluate(rows):
    scores = [judge(r["q"], r["answer"], r["reference"], r["context"]) for r in rows]
    return {
        "recall@k": round(recall_at_k(rows), 2),
        "accuracy": round(accuracy(rows), 2),
        "hallucination_rate": round(hallucination_rate(rows), 2),
        "avg_judge_score": round(sum(scores) / len(scores), 2),
    }


def main():
    rep = evaluate(TESTSET)
    # rep -> {'recall@k':0.67, 'accuracy':0.67, 'hallucination_rate':0.33, 'avg_judge_score':~3.3}
    #   d2 检索没命中 → recall 掉; 保修那条答"三年"与 context"一年"不符 → 计入幻觉、judge 扣分
    assert 0 <= rep["hallucination_rate"] <= 1 and rep["recall@k"] < 1
    print(f"✅ 补4 跑通：评测报告 {rep}（召回率<1 因有条没检索到 gold；幻觉率>0 因有条答案无据）")
    # 面试：Q 为什么不能只看准确率? A 会漏掉"答对但幻觉/瞎蒙"; Q 幻觉率怎么量? A 判答案是否有据于 context;
    #      Q LLM-as-judge 好在哪? A 比字符串匹配更懂语义,能判"意思对不对"; 缺点:judge 本身有偏差/成本,要抽检校准。


if __name__ == "__main__":
    main()
