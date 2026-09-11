"""
================================================================================
 中文版 · 命名实体识别（用中文 NER 模型 CLUENER）
================================================================================
 英文 NER 模型不认中文实体；这里用在 CLUENER2020 中文数据上微调的模型，能识别中文的
 人名/地址/公司/机构/职位等。注意两个中文细节：
   ① 该模型是“逐字”的，输出的 word 里字之间有空格(如 '张 伟')，要 .replace(' ','') 去掉；
   ② 实体类型是 name/address/company/organization/position... 不是英文的 PER/ORG/LOC。
 (对应英文版 ../分章项目/Ch7_NER实体识别.py)
 跑：python3 中文_NER实体识别.py     (首次下模型 ~400MB)
================================================================================
"""
from transformers import pipeline

ner = pipeline("token-classification",
               model="uer/roberta-base-finetuned-cluener2020-chinese",
               aggregation_strategy="simple")

# 类型英文→中文，便于阅读
type_cn = {"name": "人名", "address": "地址", "company": "公司", "organization": "机构",
           "position": "职位", "government": "政府", "scene": "景点", "book": "书名",
           "movie": "影视", "game": "游戏"}

for text in ["我叫张伟，在北京的腾讯公司担任高级工程师。",
             "李娜昨天在上海参加了阿里巴巴的发布会。"]:
    print(f"\n文本: {text}")
    for e in ner(text):
        word = e["word"].replace(" ", "")               # ★去掉逐字模型的字间空格
        cn = type_cn.get(e["entity_group"], e["entity_group"])
        print(f"   {cn:4} {word!r:10} (score={e['score']:.3f})")

print("\n✅ 中文 NER 跑通(CLUENER)。实体类型比英文细(公司/机构/职位/景点...)。")
# print("面试：Q 中文 NER 和英文有什么不同? A 中文常逐字标注(要拼回词)、类型体系不同(CLUENER 有10类)；")
# print("      生产中文 NER 还常用 BiLSTM-CRF / bert4ner / UIE(通用信息抽取)等。")
