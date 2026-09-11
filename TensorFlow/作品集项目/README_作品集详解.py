"""
================================================================================
 作品集项目 · 逐文件详解 + 阅读顺序（求职作品集，6 个可部署项目）
================================================================================
 每个项目 3 个文件，固定读法：先 README.md(懂目标/部署/简历bullet) → 再 app.py(核心代码)
 → 最后 requirements.txt(依赖)。运行：`python3 app.py smoke`(自检) / `python3 app.py`(起Web)。
 项目由易到难，建议顺序：项目1 → 4 → 6 → 2 → 3 → 5。
================================================================================

顶层
  作品集项目/README.md          6 个项目总览 + 技术栈表 + 每个的简历 bullet(先扫这个)

━━━━━ 项目1 · 中文情感分析 API（最简，先做这个建立信心）━━━━━
  项目1_中文情感分析API/
    README.md          目标/功能/部署到HF Spaces步骤/简历bullet；顶部 YAML 是 Spaces 配置
    app.py             ★核心：dianping中文情感模型(单例) + Gradio界面 + FastAPI /api/predict
                       读代码顺序：_load(单例加载) → predict(核心推理:softmax) → build_demo(Gradio)
                       → make_api(FastAPI挂载) → __main__(smoke/serve/api 三模式)
    requirements.txt   transformers/torch/gradio/fastapi/uvicorn/pydantic/sentencepiece

━━━━━ 项目4 · 多功能 NLP 工具箱（多能力集成）━━━━━
  项目4_多功能NLP工具箱/
    README.md          4 能力(情感/NER/翻译/摘要)说明 + 部署
    app.py             ★4 个能力函数各自惰性单例(sentiment/ner/translate/summarize) + Tab 界面
                       读法：先看 4 个能力函数(每个:校验→加载→推理) → build_demo 的 Tabs
    requirements.txt   + sacremoses(opus-mt 翻译建议装)

━━━━━ 项目6 · 智能客服助手（集大成：情感+NER+RAG）━━━━━
  项目6_智能客服助手/
    README.md          工单处理流程 + 阈值分流说明
    app.py             ★handle(text) 一条龙：情感分流 → NER路由/脱敏 → bge检索FAQ → 决策
                       读法：_load(三模型单例) → _embed(bge CLS池化) → handle(核心决策逻辑)
    requirements.txt   transformers/torch/gradio

━━━━━ 项目2 · 文档问答 RAG（最热门方向）━━━━━
  项目2_文档问答RAG/
    README.md          RAG 流程 + ChromaDB + 部署(mlx仅Mac,上云换vLLM)
    app.py             ★上传文档→切块→bge嵌入→ChromaDB→检索→mlx生成带引用回答
                       读法：chunk(中文切块) → _embed(bge) → build_index(入ChromaDB) →
                       retrieve → answer(检索+mlx生成，含降级兜底)
    requirements.txt   + chromadb + mlx-lm(Mac)

━━━━━ 项目3 · 本地 LLM 推理服务（OpenAI 兼容）━━━━━
  项目3_本地LLM推理服务/
    README.md          mlx-lm + OpenAI兼容API + 流式；上云换vLLM(契约一致)
    app.py             ★mlx-lm 本地LLM + FastAPI /v1/chat/completions(非流式+流式SSE) + ChatInterface
                       读法：_load(mlx单例) → complete/stream → chat_fn(ChatInterface流式) →
                       make_api(OpenAI兼容端点)
    requirements.txt   gradio/mlx-lm/fastapi/uvicorn/pydantic

━━━━━ 项目5 · LLM Agent（前沿：function calling）━━━━━
  项目5_LLM_Agent/
    README.md          function calling + 规则兜底说明 + 生产升级路径
    app.py             ★工具(情感/FAQ检索/计算器) + mlx选工具 + 规则校验兜底 + ChatInterface
                       读法：3 个 tool_* 函数 → _llm(mlx) → _keyword_route(兜底) →
                       run_agent(选工具→调用→综合回答)
    requirements.txt   transformers/torch/gradio/mlx-lm

================================================================================
 —— 每个项目怎么跑 ——
   cd 项目X_.../ ; pip install -r requirements.txt
   python3 app.py smoke     # 自检(不起服务，看逻辑通不通)
   python3 app.py           # 起 Gradio Web
   项目1/3 还有 `python3 app.py api` 起 FastAPI
 —— 部署 —— 项目1/4/6 可直接传 HF Spaces(README 顶部有 YAML)；项目2/3/5 含 mlx 仅 Mac，上云换 vLLM。
================================================================================
"""
PROJECTS = [
    ("项目1 中文情感分析API", "分类+服务化", "Transformers+Gradio+FastAPI", "★先做,最简"),
    ("项目4 多功能NLP工具箱", "多任务集成", "情感/NER/翻译/摘要+Tabs", "多能力"),
    ("项目6 智能客服助手", "集大成", "情感+NER+RAG+决策", "综合"),
    ("项目2 文档问答RAG", "RAG最热", "bge+ChromaDB+mlx", "热门方向"),
    ("项目3 本地LLM推理服务", "LLM服务", "mlx-lm+OpenAI兼容API", "服务化"),
    ("项目5 LLM Agent", "前沿", "function calling+规则兜底", "进阶"),
]
if __name__ == "__main__":
    print(__doc__)
    print(f" 建议顺序  {'项目':<24}{'方向':<12}{'技术':<28}{'备注'}")
    for i, (n, d, t, r) in enumerate(PROJECTS, 1):
        print(f"   {i}.      {n:<24}{d:<12}{t:<28}{r}")
