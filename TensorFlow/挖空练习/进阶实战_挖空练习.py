"""
================================================================================
 进阶实战 · 挖空练习（对应 实战练习/ 里新增的 8 个进阶文件）
================================================================================
 玩法：
   1) 先别看笔记，把每个练习函数里的每个 ______ 凭记忆填上（对照 ../实战练习/ 里的进阶文件）
   2) 单独跑某一节：python3 进阶实战_挖空练习.py 生成         （节名见 SECTIONS）
      或跑全部：   python3 进阶实战_挖空练习.py
   3) 没填的报 NameError，告诉你漏哪行；卡住 → 文件底部「答案区」
 覆盖：① 生成采样(Ch1) ② 数据工程(Ch5) ③ 手写分词(Ch6) ④ 四大NLP任务(Ch7)
      ⑤ Gradio生产化(Ch9) ⑥ 训练工程(Ch11) ⑦ MCP协议进阶
================================================================================
"""
import sys


# ==============================================================================
# ① 生成采样(Ch1)：解码策略 / 采样参数 / 抑制重复 / 流式
# ==============================================================================
def 生成():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    tok = AutoTokenizer.from_pretrained("distilgpt2"); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    enc = tok("In the future, AI will", return_tensors="pt")

    # 练习1：贪婪解码——关掉采样（do_sample 填什么？）
    greedy = model.generate(**enc, max_new_tokens=20, do_sample=______, pad_token_id=tok.eos_token_id)
    # 练习2：核采样——开采样 + top_p=0.9（两个参数名+值）
    nucleus = model.generate(**enc, max_new_tokens=20, do_sample=True, ______=0.9,
                             pad_token_id=tok.eos_token_id)
    # 练习3：治复读机——惩罚重复词（参数名？常用值 1.3）
    norep = model.generate(**enc, max_new_tokens=30, ______=1.3, pad_token_id=tok.eos_token_id)
    # 练习4：把生成参数打包成对象（类名？）
    cfg = ______(max_new_tokens=20, do_sample=True, temperature=0.8, pad_token_id=tok.eos_token_id)
    _ = model.generate(**enc, generation_config=cfg)
    print("① 生成采样：greedy/nucleus/no-repeat/GenerationConfig 都跑通了")
    assert greedy.shape[1] > enc["input_ids"].shape[1]


# ==============================================================================
# ② 数据工程(Ch5)：加载 / 清洗 / 长文切块
# ==============================================================================
def 数据():
    import pandas as pd
    from transformers import AutoTokenizer
    df = pd.read_parquet("hf://datasets/fancyzhx/ag_news/data/train-00000-of-00001.parquet")
    df = df.sample(n=100, random_state=42)
    # 练习5：过滤——只留 text 长度≥50 的行（布尔索引，条件填空）
    df = df[df["text"].str.len() >= ______].copy()
    # 练习6：改列名 label→category（DataFrame 的哪个方法？参数 columns={...}）
    df = df.______(columns={"label": "category"})
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    long_text = " ".join(f"word{i}" for i in range(200))
    # 练习7：长文切块——超长不丢，切成多块（参数名？值 True）
    enc = tok(long_text, max_length=32, truncation=True, ______=True, stride=8)
    print(f"② 数据工程：过滤后 {len(df)} 行，长文切成 {len(enc['input_ids'])} 块")
    assert "category" in df.columns and len(enc["input_ids"]) > 1


# ==============================================================================
# ③ 手写分词(Ch6)：BPE 合并 + Unigram 维特比核心步
# ==============================================================================
def 分词():
    from collections import defaultdict
    splits = {"tokenize": list("tokenize"), "token": list("token")}
    freqs = {"tokenize": 3, "token": 5}
    # 练习8：统计相邻对频率（BPE 的核心——遍历每个词的相邻字符对累加词频）
    pf = defaultdict(int)
    for w, f in freqs.items():
        s = splits[w]
        for i in range(len(s) - 1):
            pf[(s[i], s[i + 1])] += ______           # 累加什么？(该词的频次)
    # 练习9：选最高频的相邻对合并（max 的 key 填什么？）
    best = max(pf, key=______)
    print(f"③ 手写分词：最高频相邻对 = {best}（应为 ('t','o') 或高频对）")
    assert best in pf


# ==============================================================================
# ④ 四大NLP任务(Ch7)：抽取式QA 取 span + seq2seq 前缀
# ==============================================================================
def 任务():
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    m = "distilbert-base-cased-distilled-squad"
    tok = AutoTokenizer.from_pretrained(m)
    # 练习10：抽取式 QA 用哪个 AutoModel 类？（已给出，注意任务头）
    model = AutoModelForQuestionAnswering.from_pretrained(m).eval()
    enc = tok("Where do I work?", "I work at Hugging Face.", return_tensors="pt")
    with torch.no_grad():
        o = model(**enc)
    # 练习11：抽取式 QA 预测答案的“起始 token 下标”（对哪个 logits 取 argmax？）
    s = int(o.______.argmax())
    # 练习12：结束 token 下标
    e = int(o.______.argmax())
    ans = tok.decode(enc["input_ids"][0][s:e + 1])
    print(f"④ 四大任务(QA)：预测答案 = {ans!r}（应含 Hugging Face）")
    assert "Hugging" in ans


# ==============================================================================
# ⑤ Gradio生产化(Ch9)：State / 流式 yield / 挂 FastAPI
# ==============================================================================
def gradio():
    import gradio as gr
    # 练习13：会话状态组件（每个用户独立），初值 0（Gradio 的哪个组件？）
    with gr.Blocks() as demo:
        state = gr.______(0)
        gr.Textbox()

    # 练习14：流式输出——逐字返回。yield 出“累计到目前的文本”（填变量名）
    def stream(msg):
        acc = ""
        for ch in msg:
            acc += ch
            yield ______                             # 边算边吐，yield 出什么？
    # 练习15：把 Gradio 挂到 FastAPI（gradio 的哪个函数？）
    from fastapi import FastAPI
    app = FastAPI()
    app = gr.______(app, demo, path="/gradio")
    routes = [r.path for r in app.routes]
    print(f"⑤ Gradio生产化：State/流式/挂载 OK，路由含 /gradio = {'/gradio' in [r for r in routes]}")
    assert list(stream("hi"))[-1] == "hi"


# ==============================================================================
# ⑥ 训练工程(Ch11)：Accelerate + 适配器分离加载
# ==============================================================================
def 训练():
    from accelerate import Accelerator
    from peft import PeftModel  # noqa: F401
    acc = Accelerator()
    # 练习16：Accelerate 把 模型/优化器/数据 一次“准备”好（方法名字符串，如 "xxx"）
    #   用法：prepared = acc.prepare(model, opt, loader)
    assert hasattr(acc, ______)                      # 填 acc 上“准备”那个方法名(字符串)
    # 练习17：多卡/混合精度下用它代替 loss.backward()（方法名？）
    assert hasattr(acc, ______)                      # 填 backward
    # 练习18：分离部署——基座 + 适配器重新拼，用 PeftModel 的哪个类方法？
    assert hasattr(PeftModel, ______)                # 填 from_pretrained
    print("⑥ 训练工程：Accelerator.prepare/backward + PeftModel.from_pretrained 都对上了")


# ==============================================================================
# ⑦ MCP协议进阶：能力协商取交集 / 通知 / Sampling
# ==============================================================================
def mcp():
    client_caps = {"roots": {}, "sampling": {}}
    server_caps = {"tools": {}, "sampling": {}}
    # 练习19：能力协商——只有双方都支持的能力才可用（集合取“交集”的方法名？）
    usable = set(client_caps).______(set(server_caps))
    # 练习20：判断是不是“通知”——有 method 但【没有】哪个字段？
    def is_notification(msg):
        return "method" in msg and ______ not in msg
    # 练习21：服务器反请 Host 的 LLM，用哪个 method 名？
    sampling_method = "______"
    print(f"⑦ MCP协议：可用能力交集 = {usable}（应为 {{'sampling'}}）")
    assert usable == {"sampling"}
    assert is_notification({"method": "notifications/initialized"})
    assert sampling_method == "sampling/createMessage"


SECTIONS = {"生成": 生成, "数据": 数据, "分词": 分词, "任务": 任务,
            "gradio": gradio, "训练": 训练, "mcp": mcp}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in SECTIONS:
        SECTIONS[sys.argv[1]]()
    else:
        for name, fn in SECTIONS.items():
            try:
                fn()
            except Exception as e:
                print(f"[{name}] 还没填完 → {type(e).__name__}: {str(e)[:60]}")
        print("\n填完所有 ______ 后应全部通过；卡住看文件底部答案区。")


# ==============================================================================
#  答案区（卡住再看！对照 ../实战练习/chapter实战/分章项目 与 mcp实战 里的进阶文件）
# ------------------------------------------------------------------------------
#  1: do_sample=False                      （贪婪=不采样）
#  2: top_p=0.9                            （核采样参数名 top_p）
#  3: repetition_penalty=1.3
#  4: GenerationConfig(...)
#  5: >= 50                                （text 长度≥50）
#  6: df.rename(columns={"label":"category"})
#  7: return_overflowing_tokens=True        （配合 stride 做重叠切块）
#  8: pf[(s[i], s[i+1])] += f               （累加该词频次 f）
#  9: max(pf, key=pf.get)                   （按频次取最大）
# 10: AutoModelForQuestionAnswering         （抽取式 QA 的头）
# 11: o.start_logits.argmax()               （起始 token）
# 12: o.end_logits.argmax()                 （结束 token）
# 13: gr.State(0)
# 14: yield acc                             （流式用 yield）
# 15: gr.mount_gradio_app(app, demo, path=...)
# 16: acc.prepare(...)  → hasattr(acc, "prepare")
# 17: acc.backward(loss) → hasattr(acc, "backward")
# 18: PeftModel.from_pretrained → hasattr(PeftModel, "from_pretrained")
# 19: set(client_caps) & set(server_caps)   （交集用 &）
# 20: "id" not in msg                       （通知=有 method 无 id）
# 21: "sampling/createMessage"
# ==============================================================================
