"""评估:stub 用"输出非空率"当占位指标;真实可换困惑度/胜率/人工。"""
def evaluate(samples):
    ok = sum(1 for x in samples if x.get("output"))
    return {"eval_samples": len(samples), "valid_rate": round(ok / max(1, len(samples)), 3)}
