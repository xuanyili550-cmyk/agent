"""
作品集项目5 · LLM Agent / Function Calling（能调工具的小智能体，Mac 上真跑）
--------------------------------------------------------------------------------
 一句话：一个能【自己决定调用哪个工具】的小 Agent——本地 LLM(mlx-lm) 当大脑，
        把前面章节的中文 NLP 能力 + 计算器当工具，走“选工具 → 调用 → 综合回答”。
 前沿方向(function calling / agent)。工具本地真跑；大脑用 mlx-lm(小模型，配了关键词兜底保证稳)。
 生产升级：换 InferenceClientModel/vLLM 大模型 + smolagents(见 ../../实战练习/Agent实战)，能力更强。
 运行：
   python3 app.py smoke   # 三类问题各跑一遍(选工具→调用→回答)自检
   python3 app.py         # 起聊天 Web(看 Agent 的工具调用过程)
"""
import sys
import re
import gradio as gr

_M = {}
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
FAQ = ["重置密码：设置>安全>重置密码，按邮件链接操作。",
       "上传照片闪退：升级到 App v3.2 或更高版本。",
       "重复扣款：核实后 3-5 个工作日内原路退回。",
       "存储空间：免费 5GB，Pro 版 1TB。"]


def _dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ==============================================================================
# 工具集（本地真跑）
# ==============================================================================
def tool_sentiment(text):
    """判断中文文本情感。"""
    import torch
    if "sent" not in _M:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        n = "uer/roberta-base-finetuned-dianping-chinese"
        _M["st"] = AutoTokenizer.from_pretrained(n)
        _M["sm"] = AutoModelForSequenceClassification.from_pretrained(n).to(_dev()).eval()
    enc = _M["st"](text[:512], return_tensors="pt", truncation=True).to(_dev())
    with torch.no_grad():
        p = torch.softmax(_M["sm"](**enc).logits, -1)[0]
    return f"情感={'正面' if int(p.argmax()) == 1 else '负面'}(置信度{float(p.max()):.2f})"


def tool_search_faq(query):
    """在客服 FAQ 里语义检索。"""
    import torch
    import torch.nn.functional as F
    if "emb" not in _M:
        from transformers import AutoTokenizer, AutoModel
        n = "BAAI/bge-small-zh-v1.5"
        _M["et"] = AutoTokenizer.from_pretrained(n)
        _M["em"] = AutoModel.from_pretrained(n).to(_dev()).eval()

    def emb(ts, q=False):
        if q:
            ts = [QUERY_PREFIX + t for t in ts]
        enc = _M["et"](ts, padding=True, truncation=True, return_tensors="pt").to(_dev())
        with torch.no_grad():
            v = _M["em"](**enc).last_hidden_state[:, 0]
        return F.normalize(v, p=2, dim=1)
    if "fv" not in _M:
        _M["fv"] = emb(FAQ)
    sims = (emb([query], q=True) @ _M["fv"].T)[0]
    return FAQ[int(sims.argmax())]


def tool_calculator(expr):
    """计算一个算术表达式(只允许数字和 + - * / ( ) )。"""
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr or ""):
        return "非法表达式"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))     # 已限制字符，安全
    except Exception:
        return "计算失败"


TOOLS = {"tool_sentiment": tool_sentiment, "tool_search_faq": tool_search_faq,
         "tool_calculator": tool_calculator}
TOOL_DESC = ("tool_sentiment(判断中文情感) / tool_search_faq(查客服FAQ) / "
             "tool_calculator(算数学表达式)")


# ==============================================================================
# Agent 大脑：LLM 选工具 → 调用 → 综合回答（带关键词兜底，保证小模型也稳）
# ==============================================================================
def _llm(prompt, max_tokens=80):
    if "llm" not in _M:
        from mlx_lm import load
        _M["llm"], _M["lt"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import generate
    text = _M["lt"].apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    return generate(_M["llm"], _M["lt"], prompt=text, max_tokens=max_tokens, verbose=False).strip()


def _keyword_route(q):
    """兜底：小模型选错时按关键词兜底选工具，保证 Demo 稳定。"""
    if re.search(r"[+\-*/]|\d+\s*[加减乘除]|算|等于", q):
        return "tool_calculator", re.sub(r"[^\d+\-*/().]", "", q) or q
    if re.search(r"情感|正面|负面|评价|满意|生气|投诉", q):
        return "tool_sentiment", q
    return "tool_search_faq", q


def run_agent(question):
    """function calling 循环：① LLM 选工具+参数 ② 执行 ③ LLM 综合回答。"""
    steps = []
    # ① 让 LLM 选工具(严格格式，便于解析)
    pick = _llm(f"可用工具：{TOOL_DESC}。\n用户问题：{question}\n"
                f"只输出一行：工具名|参数。例如 tool_search_faq|怎么退款", max_tokens=40)
    m = re.search(r"(tool_\w+)\s*[|｜]\s*(.+)", pick)
    tool = m.group(1) if (m and m.group(1) in TOOLS) else None
    arg = m.group(2).strip() if m else ""
    # 规则校验 LLM 的提议(小模型可靠性有限)：工具不存在/参数为空/参数是工具名/算术却无数字 → 兜底
    bad = (tool is None or not arg or arg.startswith("tool_")
           or (tool == "tool_calculator" and not re.search(r"\d", arg)))
    if bad:
        tool, arg = _keyword_route(question)                 # 规则兜底，保证 Demo 稳定
        steps.append(f"🧠 LLM 提议不可靠 → 规则校验兜底：{tool}（参数：{arg[:30]}）")
    else:
        steps.append(f"🧠 LLM 选择：{tool}（参数：{arg[:30]}）")
    # ② 执行工具
    obs = TOOLS[tool](arg)
    steps.append(f"🔧 调用 {tool} → 观察：{obs}")
    # ③ LLM 综合回答
    final = _llm(f"用户问：{question}\n工具返回：{obs}\n请用一句话自然地回答用户：", max_tokens=80)
    steps.append(f"💬 回答：{final}")
    return "\n".join(steps), tool, obs


def chat_fn(message, history):
    trace, _, _ = run_agent(message)
    return trace


def build_demo():
    return gr.ChatInterface(
        fn=chat_fn, title="能调工具的小 Agent(function calling)",
        description="本地 mlx-lm 当大脑，工具=中文情感/FAQ检索/计算器。会显示“选工具→调用→回答”全过程。",
        examples=["帮我算一下 128 乘以 39", "我要退款怎么弄", "分析这句话情感：你们服务太差了"],
        analytics_enabled=False,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        for q in ["帮我算 128*39", "我要退款怎么办", "分析情感：你们服务太差了"]:
            trace, tool, obs = run_agent(q)
            print(f"\n【{q}】\n{trace}")
        # 兜底保证工具选对：算术→计算器，退款→FAQ，情感→情感
        assert run_agent("帮我算 12+8")[1] == "tool_calculator"
        assert "退" in run_agent("怎么退款")[2] or run_agent("怎么退款")[1] == "tool_search_faq"
        build_demo()
        print("\n✅ 项目5 自检通过：LLM 选工具 → 调用(情感/FAQ/计算) → 综合回答，带关键词兜底。")
    else:
        build_demo().queue().launch()
