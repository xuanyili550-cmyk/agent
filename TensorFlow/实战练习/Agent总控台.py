"""
================================================================================
 Agent 总控台（Gradio）· 把所有 Agent 案例串成一个可视化控制台
================================================================================
 一个界面用上前面所有 Agent 能力(复用各案例的真代码，importlib 加载，不重写)：
   · NL2SQL 问数        (Agent进阶案例/案例1)  自然语言→SQL→SQLite执行→回答
   · 深度研搜多智能体    (Agent进阶案例/案例2)  主管+检索/写作/审核子agent
   · 混合检索 RAG       (Agent进阶案例/案例3)  向量+BM25+RRF+重排
   · 联网搜索 Agent     (Agent工程化案例/案例1) DuckDuckGo 真联网+综合回答
   · 中文 NLP 工具      (Agent实战/案例2)      情感/NER/检索
 都是本机真跑(节点里调 mlx-lm，不需 Token)。
 运行：
   python3 Agent总控台.py smoke   # 加载各案例 + 构建界面 + 跑一个能力自检(不起服务)
   python3 Agent总控台.py         # 起 Gradio 总控台网页
================================================================================
"""
import os
import sys
import importlib.util
import gradio as gr

BASE = os.path.dirname(os.path.abspath(__file__))


def _load(rel, name):
    """按路径加载模块(案例文件名带中文，用 importlib；只加载不跑 __main__)。"""
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_MODS = {}


def _mod(key):
    if key not in _MODS:
        _MODS[key] = _load(*{
            "nl2sql": ("Agent进阶案例/案例1_NL2SQL数据分析Agent.py", "m_nl2sql"),
            "multi": ("Agent进阶案例/案例2_深度研搜多智能体.py", "m_multi"),
            "hybrid": ("Agent进阶案例/案例3_混合检索RAG_LangGraph.py", "m_hybrid"),
            "websearch": ("Agent工程化案例/案例1_联网搜索Agent.py", "m_websearch"),
            "nlp": ("Agent实战/案例2_NLP能力Agent_中文客服.py", "m_nlp"),
        }[key])
    return _MODS[key]


# —— 各能力的界面回调(复用案例真函数) ——
def do_nl2sql(q):
    if not q.strip():
        raise gr.Error("请输入问题")
    r = _mod("nl2sql").ask(q)
    return ("步骤：\n  " + "\n  ".join(r["trace"]) +
            f"\n\nSQL: {r['sql']}\n结果: {r['rows']}\n\n💬 {r['answer']}")


def do_multi(topic):
    if not topic.strip():
        raise gr.Error("请输入研究主题")
    r = _mod("multi").research(topic)
    return "多智能体协作：\n  " + "\n  ".join(r["trace"]) + "\n\n" + r["report"]


def do_hybrid(q):
    if not q.strip():
        raise gr.Error("请输入问题")
    r = _mod("hybrid").ask(q)
    return (f"向量召回 idx={r['vec']}  BM25召回 idx={r['bm25']}\n融合命中:\n  " +
            "\n  ".join(r["context"]) + f"\n\n💬 {r['answer']}")


def do_websearch(q):
    if not q.strip():
        raise gr.Error("请输入问题")
    r = _mod("websearch").research(q)
    return ("联网搜索：\n  " + "\n  ".join(r["trace"]) + f"\n\n💬 {r['answer']}\n\n来源:\n  " +
            "\n  ".join(r["sources"][:4]))


def do_nlp(text):
    if not text.strip():
        raise gr.Error("请输入文本")
    m = _mod("nlp")
    return (f"情感: {m.get_sentiment(text)}\n"
            f"实体: {m.extract_entities(text)}\n"
            f"FAQ检索: {m.search_faq(text)}")


def build_demo():
    with gr.Blocks(title="Agent 总控台", analytics_enabled=False) as demo:
        gr.Markdown("# 🎛️ Agent 总控台\n一个界面用上所有 Agent 能力（本机真跑，节点调 mlx-lm）")
        with gr.Tab("NL2SQL 问数"):
            gr.Interface(do_nl2sql, gr.Textbox(label="问数据", value="各城市销售额是多少？"),
                         gr.Textbox(label="SQL+结果+回答", lines=8), flagging_mode="never")
        with gr.Tab("深度研搜(多智能体)"):
            gr.Interface(do_multi, gr.Textbox(label="研究主题", value="RAG 检索增强生成"),
                         gr.Textbox(label="协作过程+报告", lines=10), flagging_mode="never")
        with gr.Tab("混合检索 RAG"):
            gr.Interface(do_hybrid, gr.Textbox(label="问题", value="专业版多少钱"),
                         gr.Textbox(label="双路召回+融合+回答", lines=8), flagging_mode="never")
        with gr.Tab("联网搜索"):
            gr.Interface(do_websearch, gr.Textbox(label="问题(联网)", value="什么是向量数据库"),
                         gr.Textbox(label="搜索+综合回答+来源", lines=10), flagging_mode="never")
        with gr.Tab("中文 NLP 工具"):
            gr.Interface(do_nlp, gr.Textbox(label="中文文本", value="你们的服务太差了，我要投诉"),
                         gr.Textbox(label="情感/实体/FAQ", lines=5), flagging_mode="never")
    return demo


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # 加载各案例模块 + 构建界面 + 跑一个能力自检(NL2SQL 不联网、较快)
        for k in ("nl2sql", "multi", "hybrid", "websearch", "nlp"):
            assert _mod(k), k
        build_demo()
        out = do_nl2sql("各城市销售额是多少？")
        print(out[:200])
        assert "SQL" in out and "北京" in out
        print("\n✅ 总控台自检通过：5 个案例模块已加载 + 界面构建 + NL2SQL 能力真跑。")
    else:
        build_demo().queue().launch()
