"""
================================================================================
 综合案例2 · 内容风控全链路（整合 Ch1情感 + Ch6/7 NER + 分级决策 + 生产结构）
================================================================================
 用户发的内容自动过审：风险打分 → 抽敏感实体(脱敏) → 按置信度分级(通过/人工/拦截)。
 整合：
   [Ch1/2 分类] uer/dianping 中文情感当风险信号(演示；生产用违规标注数据微调专用毒性模型)
   [Ch6/7 NER]  uer/cluener 抽 人名/公司/地址(PII 脱敏)
   [生产]        置信度分级(不是一刀切) + 脱敏 + 结构化输出，模型惰性单例
 全程确定性(可靠、无需 LLM)，秒级。
 跑：python3 综合2_内容风控全链路.py
================================================================================
"""
import re

_M = {}
BLOCK_TH, REVIEW_TH = 0.95, 0.60


def _dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load():
    if "ready" not in _M:
        import torch  # noqa
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        s = "uer/roberta-base-finetuned-dianping-chinese"
        _M["st"] = AutoTokenizer.from_pretrained(s)
        _M["sm"] = AutoModelForSequenceClassification.from_pretrained(s).to(_dev()).eval()
        _M["ner"] = pipeline("token-classification",
                            model="uer/roberta-base-finetuned-cluener2020-chinese",
                            aggregation_strategy="simple", device=-1)
        _M["ready"] = True


def risk_score(text):
    """负面概率当风险分(演示)。生产换中文毒性/违规分类模型。"""
    import torch
    enc = _M["st"](text[:512], return_tensors="pt", truncation=True).to(_dev())
    with torch.no_grad():
        p = torch.softmax(_M["sm"](**enc).logits, -1)[0]
    return float(p[0])                                    # 0=负面


def desensitize(text):
    """NER 抽 PII → 打码脱敏。"""
    ents = [(e["entity_group"], e["word"].replace(" ", "")) for e in _M["ner"](text)]
    masked = text
    for group, word in ents:
        if group in ("name", "company", "address", "organization", "position") and word:
            masked = masked.replace(word, f"[{group}]")
    return ents, masked


def moderate(text):
    _load()
    risk = risk_score(text)
    ents, masked = desensitize(text)
    if risk >= BLOCK_TH:
        action, level = "拦截", "高风险"
    elif risk >= REVIEW_TH:
        action, level = "转人工复核", "中风险"
    else:
        action, level = "放行", "低风险"
    return {"原文": text, "风险分": round(risk, 2), "分级": level, "决策": action,
            "实体": ents or "无", "脱敏后": masked}


if __name__ == "__main__":
    contents = [
        "这个产品太好用了，强烈推荐！",
        "我一定会找到你，让你付出代价，你这个废物！",
        "有问题联系张伟，他在北京的腾讯公司负责售后。",
    ]
    for c in contents:
        r = moderate(c)
        # print("─" * 64)  # 装饰分隔线（静音）
        print(f"内容: {r['原文']}")
        print(f"  风险分={r['风险分']} → {r['分级']}  决策={r['决策']}")
        print(f"  实体={r['实体']}\n  脱敏后: {r['脱敏后']}")
    assert moderate("张伟在腾讯公司")["脱敏后"] != "张伟在腾讯公司"    # 已脱敏
    assert 0 <= moderate("你好")["风险分"] <= 1
    # print("\n" + "─" * 64)  # 装饰分隔线（静音）
    # print("⚠ 情感≠毒性：这是点评情感模型充当风险信号的演示；生产必须用违规标注数据微调专用分类器。")
    print("✅ 综合2 跑通：风险分级(Ch1) + PII脱敏(Ch6/7 NER) + 置信度决策(生产)。")
