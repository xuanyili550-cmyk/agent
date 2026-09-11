"""
作品集项目4 · 多功能 NLP 工具箱（生产级，可部署 HF Spaces）
--------------------------------------------------------------------------------
 一句话：把多个 NLP 能力集成成一个多标签页的产品——情感 / 实体识别 / 翻译 / 摘要。
 生产要点：每个能力模型【惰性单例】(用到才加载、常驻)、输入校验(gr.Error)、Tab 布局、示例。
 语种：情感/NER 用【中文模型】(中文才准)，翻译=英译中，摘要=英文(t5)。
 运行：
   python3 app.py smoke   # 四个能力各调一次自检(不起服务)
   python3 app.py         # 起多标签页网页(HF Spaces 默认入口)
"""
import sys
import gradio as gr

_M = {}


def _dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---- ① 中文情感 ----
def sentiment(text):
    import torch
    if not text.strip():
        raise gr.Error("请输入中文文本")
    if "sent" not in _M:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        n = "uer/roberta-base-finetuned-dianping-chinese"
        _M["sent_tok"] = AutoTokenizer.from_pretrained(n)
        _M["sent"] = AutoModelForSequenceClassification.from_pretrained(n).to(_dev()).eval()
    enc = _M["sent_tok"](text[:512], return_tensors="pt", truncation=True).to(_dev())
    with torch.no_grad():
        p = torch.softmax(_M["sent"](**enc).logits, -1)[0]
    return {"正面": float(p[1]), "负面": float(p[0])}


# ---- ② 中文 NER（cluener 逐字模型：word 去空格；类型 name/company/address...）----
def ner(text):
    if not text.strip():
        raise gr.Error("请输入中文文本")
    if "ner" not in _M:
        from transformers import pipeline
        _M["ner"] = pipeline("token-classification",
                             model="uer/roberta-base-finetuned-cluener2020-chinese",
                             aggregation_strategy="simple", device=-1)
    cn = {"name": "人名", "company": "公司", "organization": "机构", "address": "地址",
          "position": "职位", "government": "政府", "scene": "景点"}
    return {"text": text,
            "entities": [{"entity": cn.get(e["entity_group"], e["entity_group"]),
                          "word": e["word"].replace(" ", ""),
                          "start": e["start"], "end": e["end"]} for e in _M["ner"](text)]}


# ---- ③ 翻译（英译中，opus-mt seq2seq）----
def translate(text):
    import torch
    if not text.strip():
        raise gr.Error("请输入英文文本")
    if "trans" not in _M:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        n = "Helsinki-NLP/opus-mt-en-zh"
        _M["trans_tok"] = AutoTokenizer.from_pretrained(n)
        _M["trans"] = AutoModelForSeq2SeqLM.from_pretrained(n).to(_dev()).eval()
    enc = _M["trans_tok"]([text], return_tensors="pt", padding=True, truncation=True).to(_dev())
    with torch.no_grad():
        out = _M["trans"].generate(**enc, max_new_tokens=128)
    return _M["trans_tok"].batch_decode(out, skip_special_tokens=True)[0]


# ---- ④ 摘要（英文，t5 加 "summarize:" 前缀）----
def summarize(text):
    import torch
    if len(text.split()) < 10:
        raise gr.Error("请输入较长的英文段落(至少10个词)")
    if "sum" not in _M:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _M["sum_tok"] = AutoTokenizer.from_pretrained("t5-small")
        _M["sum"] = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(_dev()).eval()
    enc = _M["sum_tok"](["summarize: " + text], return_tensors="pt", truncation=True).to(_dev())
    with torch.no_grad():
        out = _M["sum"].generate(**enc, max_new_tokens=80)
    return _M["sum_tok"].batch_decode(out, skip_special_tokens=True)[0]


def build_demo():
    with gr.Blocks(title="多功能 NLP 工具箱", analytics_enabled=False) as demo:
        gr.Markdown("# 多功能 NLP 工具箱\n情感分析 · 实体识别 · 翻译 · 摘要（一个界面集成多能力）")
        with gr.Tab("情感分析(中文)"):
            gr.Interface(sentiment, gr.Textbox(label="中文文本", lines=3), gr.Label(label="情感"),
                         examples=[["这家店太棒了！"], ["质量很差退货"]], flagging_mode="never")
        with gr.Tab("实体识别(中文)"):
            gr.Interface(ner, gr.Textbox(label="中文文本", lines=3), gr.JSON(label="实体"),
                         examples=[["我叫张伟，在北京的腾讯公司工作。"]], flagging_mode="never")
        with gr.Tab("翻译(英译中)"):
            gr.Interface(translate, gr.Textbox(label="English", lines=3), gr.Textbox(label="中文译文"),
                         examples=[["Machine learning is fun."]], flagging_mode="never")
        with gr.Tab("摘要(英文)"):
            gr.Interface(summarize, gr.Textbox(label="English paragraph", lines=5),
                         gr.Textbox(label="Summary"), flagging_mode="never")
    return demo


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        print("情感:", {k: round(v, 2) for k, v in sentiment("这家餐厅非常好吃！").items()})
        print("NER :", [e["word"] for e in ner("我在腾讯公司上班")["entities"]])
        print("翻译:", translate("Machine learning is fun."))
        print("摘要:", summarize("The transformer is a deep learning architecture based on attention. "
                                 "It powers most large language models and became the NLP standard.")[:60], "...")
        assert sentiment("这家餐厅非常好吃！")["正面"] > 0.5
        assert any("腾讯" in e["word"] for e in ner("我在腾讯公司上班")["entities"])
        build_demo()
        print("✅ 项目4 自检通过：情感/NER/翻译/摘要 四能力 + Tab 界面构建。")
    else:
        build_demo().queue().launch()
