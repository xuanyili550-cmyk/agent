"""
================================================================================
 综合项目4 · 内容审核 / 风控系统（中文版，整合 Ch1 + Ch2 + Ch6 + Ch7）
================================================================================
 用户发的内容(评论/帖子)自动审核：判风险 → 抽敏感实体 → 按置信度分级(通过/人工/拦截)。
 全部用【中文模型】：
   [Ch1+Ch2 分类]  uer/roberta-base-finetuned-dianping-chinese 判负面(这里用情感当风险信号演示；
                   生产用你标注的“违规/正常”数据微调专用毒性分类器，见 分章项目/Ch2)。
   [Ch6+Ch7 NER]   uer/roberta-base-finetuned-cluener2020-chinese 抽人名/公司/地址等实体：脱敏、风控画像。
   [生产 置信度分级] 高置信违规→拦截；模糊→转人工复核；正常→放行。审核系统核心是“分级”不是“一刀切”。

 本地跑：python3 项目4_内容审核风控.py
 说明：情感≠毒性，这里仅演示“分级”骨架；真实审核用专门的中文毒性/违规分类模型 + 领域微调。
================================================================================
"""
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, pipeline)


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


CONTENTS = [
    "这个产品太好用了，强烈推荐给大家，物超所值！",
    "我一定会找到你，让你为此付出代价，你这个废物。",
    "有问题请联系张伟，他在北京的腾讯公司负责退款事宜。",
    "还行吧，凑合用，没什么特别的感觉。",
]

# 分级阈值：高置信负面→拦截；中间→人工；正常→放行
BLOCK_TH, REVIEW_TH = 0.95, 0.60


class Moderator:
    def __init__(self):
        self.dev = pick_device()
        # [Ch1/2] 中文风险分类器(演示用点评情感模型：负面≈风险信号；生产换中文毒性/违规分类)
        s = "uer/roberta-base-finetuned-dianping-chinese"
        self.tok = AutoTokenizer.from_pretrained(s)
        self.clf = AutoModelForSequenceClassification.from_pretrained(s).to(self.dev).eval()
        # [Ch6/7] 中文 NER 抽敏感实体(cluener 逐字，word 去空格)
        self.ner = pipeline("token-classification",
                            model="uer/roberta-base-finetuned-cluener2020-chinese",
                            aggregation_strategy="simple", device=0 if self.dev == "cuda" else -1)

    def risk(self, text):
        enc = self.tok(text, return_tensors="pt", truncation=True).to(self.dev)
        with torch.no_grad():
            p = torch.softmax(self.clf(**enc).logits, -1)[0]
        return float(p[0])                                     # 标签0=负面，负面概率当“风险分”

    def moderate(self, text):
        risk = self.risk(text)
        ents = [(e["entity_group"], e["word"].replace(" ", "")) for e in self.ner(text)]
        if risk >= BLOCK_TH:
            action, level = "拦截", "高风险"
        elif risk >= REVIEW_TH:
            action, level = "转人工复核", "中风险"
        else:
            action, level = "放行", "低风险"
        return {"text": text, "risk": risk, "level": level, "action": action, "entities": ents}


def main():
    m = Moderator()
    print(f">>> 设备={m.dev}  拦截阈值={BLOCK_TH}  复核阈值={REVIEW_TH}（全中文模型）\n")
    for c in CONTENTS:
        r = m.moderate(c)
        # print("─" * 62)
        print(f"内容: {r['text']}")
        print(f"  风险分={r['risk']:.2f} → {r['level']}  决策={r['action']}")
        if r["entities"]:
            print(f"  涉及实体(可脱敏): {r['entities']}")
    # print("─" * 62)
    print("✅ 中文审核流程跑通：风险分类(Ch1/2)+敏感实体(Ch6/7)+置信度分级(生产)。")
    # print("⚠ 注意看结果的“错排”：温和差评(如“还行吧凑合用”)被点评情感模型判成高负面而拦截，")
    # print("   真正的威胁言论风险分反而更低——因为这是【商品评论情感】模型，不是【毒性/违规】模型。")
    # print("   这正是本项目要教的点：情感≠毒性，生产内容审核【必须】用违规标注数据微调专用分类器。")


# ==============================================================================
# 生产要点 + 面试题
# ==============================================================================
# 生产：· 用标注的中文违规数据微调专用分类器(多标签：辱骂/色情/暴力/欺诈…)，别用情感模型凑。
#      · 多级：模型初筛 → 规则/黑名单(敏感词) → 人工复核 → 申诉。高召回优先(宁可多送人工，别漏)。
#      · 实体脱敏：中文 NER/UIE 抽出 PII(人名/电话/身份证)自动打码。
#      · 时效：审核要低延迟(发帖即审)，模型要轻量/蒸馏 + 批处理 + 缓存。
# 面试：Q 审核为什么要“分级”不是二分类？A 灰色内容多，硬判误伤高；分级把模糊的交人工，平衡准确率和体验。
#      Q 情感模型能直接当审核模型吗？A 不能，情感≠违规；只能当演示骨架，生产必须用毒性/违规标注数据微调。
#      Q 类别不均衡(违规样本少)怎么办？A 过采样/欠采样、focal loss、类别加权、宏 F1 评估、多标注。

if __name__ == "__main__":
    main()
