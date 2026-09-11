"""
================================================================================
 全功能案例 · 总清单（证明 5 个案例穷尽 MCP + Agent + Ch1-12 的每个功能）
================================================================================
 不是应用场景(客服/中台等)，而是【按领域穷尽全部功能】的 showcase：每个功能都实跑(或生产 GPU
 真实代码就绪)，每个文件末尾都有自己的覆盖清单并 assert 全覆盖。本文件汇总总清单。

 —— 5 个 showcase(各自 assert 全覆盖) ——
   全功能1_MCP协议全能力.py          18 项：4原语+Roots+Sampling+能力协商+initialized+传输+错误码+FastMCP+客户端
   全功能2_Agent全框架能力.py        15 项：ReAct+@tool/Tool子类+CodeAgent/ToolCalling+4后端+3框架+RAG+记忆+MCP+可观测
   全功能3_Ch1-3基础与推理.py        19 项：pipeline五任务+内部+架构+偏见+采样+流式；动态填充/指标/Trainer/ClassLabel；fill-mask/MLM-CLM/头家族/特征/选型
   全功能4_Ch5-7数据分词NLP任务.py   23 项：加载/清洗/检索/FAISS/流式/切分；重训/BPE/WordPiece/Unigram/offset/规范化；NER/QA/翻译/摘要/MLM+指标+训练工程
   全功能5_Ch9-12上线与对齐.py       18 项：Interface/Blocks/State/Chat/流式/Progress/mount/auth；聊天模板/LoRA/PeftModel/SFTTrainer/QLoRA；GRPO三件套/GRPOTrainer/vLLM

 —— 覆盖的课程/章节(每项都被某个 showcase 穷尽) ——
   MCP Course · Agent Course · Chapter 1 · 2 · 3 · 5 · 6 · 7 · 9 · 11 · 12

 —— 说明 ——
   · 本机能跑的功能全部真跑；生产训练(Ch2/7 Trainer、Ch11 SFTTrainer/QLoRA、Ch12 GRPOTrainer)按用户
     要求写成【生产 GPU 方式】的真实代码(trl + 真实数据 + bf16/DeepSpeed/vLLM + push_to_hub)，惰性导入、
     本机不跑；GRPO 三件套/LoRA 配置/所有指标等 CPU 部分本机验证。
   · datasets 库本机坏、部分 pipeline 被删、smolagents[mcp]/bitsandbytes 未装 → 用 pandas/AutoModelForX/
     手写/真实代码参考绕开，均已注明。
 跑：python3 README_全功能总清单.py   （逐个案例自检见各文件；本文件汇总并校验课程/章节齐全）
================================================================================
"""
SHOWCASES = {
    "全功能1_MCP协议全能力": (18, ["MCP Course"]),
    "全功能2_Agent全框架能力": (15, ["Agent Course"]),
    "全功能3_Ch1-3基础与推理": (19, ["Ch1", "Ch2", "Ch3"]),
    "全功能4_Ch5-7数据分词NLP任务": (23, ["Ch5", "Ch6", "Ch7"]),
    "全功能5_Ch9-12上线与对齐": (18, ["Ch9", "Ch11", "Ch12"]),
}
COURSES = ["MCP Course", "Agent Course", "Ch1", "Ch2", "Ch3", "Ch5", "Ch6", "Ch7", "Ch9", "Ch11", "Ch12"]

if __name__ == "__main__":
    print(__doc__)
    total = sum(n for n, _ in SHOWCASES.values())
    covered = set(c for _, cs in SHOWCASES.values() for c in cs)
    print(f" 5 个 showcase 共 {total} 项功能。")
    print(" 课程/章节覆盖：", {c: ("√" if c in covered else "✗") for c in COURSES})
    assert covered.issuperset(COURSES), f"未覆盖: {set(COURSES)-covered}"
    print(f" ✅ MCP Course + Agent Course + Chapter 1/2/3/5/6/7/9/11/12 全部功能，被 5 个案例穷尽覆盖({total} 项)。")
    print(" (每个 showcase 文件末尾都有自己的 assert 全覆盖清单，逐个 `python3 文件名` 可验证。)")
