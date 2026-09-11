"""
================================================================================
 实战练习 · 逐文件详解 + 文件级阅读顺序（每个文件是什么、先读哪个后读哪个）
================================================================================
 配合 README_学习路线.py(讲阶段)看。这里【每个文件】都有说明，folder 内按数字顺序读。
 约定：几乎每个文件顶部 docstring 都有"跑法/覆盖点/面试题"，本表是精简索引。
================================================================================

████ 一、chapter实战/分章项目/ (单章聚焦，★整个实战练习从这里开始，按 Ch 号读) ████
 1  Ch1_情感分析服务.py       pipeline全景(情感/零样本/生成/填空/NER)+内部三步+三架构家族+偏见demo
 2  Ch1_进阶_生成与采样.py    解码策略(贪婪/采样/束搜索)+temperature/top_p/top_k/重复惩罚+GenerationConfig+流式
 3  Ch2_文本分类微调.py       Trainer高层 vs 手写循环对照 + 动态填充(DataCollatorWithPadding)
 4  Ch3_预训练模型应用.py     fill-mask手写 + MLM/CLM对比 + AutoModelForX头家族 + 特征抽取
 5  Ch5_语义搜索引擎.py       嵌入+mean池化+余弦检索(RAG地基)；换说法也命中
 6  Ch5_进阶_数据工程.py      加载/清洗算子/长文overflow切块/落盘(datasets坏→pandas等价)
 7  Ch6_领域分词器.py         train_new_from_iterator 领域重训 + 压缩率(序列短41%)
 8  Ch6_进阶_手写分词算法.py  手写 BPE/WordPiece/Unigram + 规范化/预分词API
 9  Ch7_NER实体识别.py        NER: pipeline + 手写token级BIO + 子词对齐/-100
 10 Ch7_进阶_四大NLP任务.py   MLM训练+困惑度 / 翻译+BLEU / 摘要+ROUGE / 抽取式QA+SQuAD(指标手写)
 11 Ch9_Gradio应用.py         Gradio Interface 最简上线(smoke 自检)
 12 Ch9_进阶_生产化.py        Blocks/gr.State/gr.Progress/流式yield/mount_gradio_app/auth
 13 Ch11_LoRA指令微调.py      聊天模板apply_chat_template + LoRA(只训<1%) + SFT真训练
 14 Ch11_进阶_训练工程.py     Accelerate(prepare/backward) + PeftModel.from_pretrained分离加载 + QLoRA(参考)
 15 Ch12_GRPO强化学习.py      GRPO三件套: 组内优势 + 可验证奖励 + 裁剪损失/KL + 多奖励塑形
   读法：每章先读 ChN_主题.py(基础)，再读 ChN_进阶_*.py(深挖)。

████ 二、chapter实战/案例/ (跨章端到端，照着手写练熟) ████
 1  README_总导航.py / README_实战练习导航.py   本目录/整体导航
 2  案例1_文本分类端到端.py   Ch1+2+6：分词→数据管道→训练→评估(手写P/R/F1+混淆矩阵)+类别不均衡
 3  案例2_RAG语义检索问答.py  Ch5：切块→嵌入→检索→带引用回答；含生产要点+面试题
 4  案例3_LoRA指令微调.py     Ch11：LoRA SFT 完整闭环(真训练小步)
 5  案例4_GRPO强化学习.py     Ch12：★手写 GRPO 真训练循环(答对率上升，证明懂底层)
 6  综合大demo_智能客服全流程.py  情感+NER+检索一条龙(英文模型；中文版见 ../中文版)

████ 三、chapter实战/综合项目/ (6个整合项目，全中文模型) ████
 1 项目1_智能客服助手.py    情感分流+NER路由+bge检索FAQ+决策(Assistant类,run_cli/run_ui)
 2 项目2_RAG企业知识库.py   中文文档→按句切块→bge检索→带引用回答(Gradio ui)
 3 项目3_LLM对齐全流程.py   Ch3→Ch11→Ch12：聊天模板→SFT+LoRA真训练→评估→GRPO
 4 项目4_内容审核风控.py    情感当风险信号+NER脱敏+置信度分级(注:情感≠毒性,有教学说明)
 5 项目5_电商语义搜索.py    商品语义召回+重排两段式
 6 项目6_推理服务与部署.py  逐条 vs 批处理吞吐对比 + vLLM/部署要点

████ 四、chapter实战/中文版/ (中文模型版,英文模型喂中文会失准) ████
 1 README_中文版说明.py     中文模型选型说明
 2 中文_情感分析.py         uer/dianping(0负1正)
 3 中文_NER实体识别.py      uer/cluener(逐字,word去空格,类型name/company/address..)
 4 中文_语义检索RAG.py      bge-small-zh(★CLS池化+查询前缀)
 5 中文_智能客服全流程.py   三个中文模型协作

████ 五、chapter实战/生产架构/ (真上线怎么搭,真实代码需GPU/服务,本机不跑) ████
 1 README_生产架构导航.py   中美技术栈对照 + 生产三铁律
 2 生产01_LLM服务_vLLM_FastAPI.py   vLLM(OpenAI兼容)+FastAPI网关(鉴权/限流)
 3 生产02_RAG_Qdrant向量库.py       Qdrant向量库+嵌入+检索+生成(新版API)
 4 生产03_微调_多卡_Accelerate.py   trl.SFTTrainer+DeepSpeed+推模型仓库
 5 生产04_GRPO_trl_vLLM.py          trl.GRPOTrainer+vLLM采样+可验证奖励
 6 生产05_调用LLM服务_客户端.py     OpenAI客户端/InferenceClient/流式/结构化JSON
 7 部署_Docker_Compose_K8s.py       Dockerfile/compose/K8s清单/监控(配置文本)

████ 六、chapter实战/面试高频题库.py  各章高频面试题+答案(求职准备) ████

████ 七、mcp实战/ (MCP协议,按案例号读) ████
 0 README_mcp实战导航.py    8案例导航 + MCP原语×章节映射 + 面试概念
 1 案例1_MCP工具服务器_NLP能力.py  ★手写stdio+JSON-RPC: 情感/NER暴露成tools(握手→发现→调用)
 2 案例2_MCP服务器_语义检索.py     search_knowledge工具
 3 案例3_MCP服务器_LLM生成.py      本地小LLM generate工具
 4 案例4_MCP_Agent多工具编排.py    连多个MCP服务器,规则选工具
 5 案例5_MCP生产部署.py            Tools+Resources+Prompts三原语 + 部署全景
 6 案例6_FastMCP官方SDK生产版.py   @mcp.tool 官方SDK写法
 7 案例7_MCP协议进阶_Roots_Sampling_能力协商.py  能力协商/initialized/Roots/Sampling/资源模板(消息形态)
 8 案例8_GradioMCP与官方SDK客户端.py  Gradio mcp_server + smolagents/hf Agent客户端

████ 八、Agent实战/ (Agent框架,smolagents为主) ████
 0 README_Agent实战导航.py  Agent概念 + 能力×章节映射 + 本机现实(小模型弱)
 1 案例1_smolagents基础_工具与Agent.py  @tool/Tool子类/CodeAgent vs ToolCallingAgent/模型后端
 2 案例2_NLP能力Agent_中文客服.py  ★把中文情感/NER/检索包成smolagents工具+规则编排
 3 案例3_Agentic_RAG_中文知识库.py  检索工具+Agentic检索决策(bge)
 4 案例4_Agent接MCP工具.py         手写MCP客户端连案例1服务器(真跑)
 5 案例5_多框架对比_smolagents_LangGraph_LlamaIndex.py  三框架组装对比
 6 生产_Agent部署_模型后端与观测.py  vLLM模型+GradioUI+Langfuse观测(真实代码)

████ 九、综合案例/ (6个全技术综合,确定性编排) ████
 0 README_综合案例导航.py  6案例 + 指向两个子目录(生产综合案例/全功能案例)
 1 综合1_企业智能助手.py   意图路由→RAG/情感NER/计算→mlx综合回答
 2 综合2_内容风控全链路.py 风险分级+NER脱敏+置信度决策(确定性,快)
 3 综合3_文档智能处理.py   文档→实体+情感→抽取摘要→ChromaDB→检索问答
 4 综合4_MCP能力Agent.py   Agent用远程MCP工具+本地检索
 5 综合5_微调到服务闭环.py 聊天模板→LoRA真训练→保存→PeftModel重载→推理
 6 综合6_全能力客服Agent.py 情感分流+NER+RAG+mlx生成回复+决策(集大成)

████ 十、综合案例/全功能案例/ (★按领域穷尽全部功能+总清单93项) ████
 0 README_全功能总清单.py  汇总校验:5案例覆盖MCP+Agent+Ch全部功能
 1 全功能1_MCP协议全能力.py   18项:四原语+Roots+Sampling+能力协商+传输+错误码+FastMCP+客户端
 2 全功能2_Agent全框架能力.py 15项:ReAct+@tool+CodeAgent/ToolCalling+4后端+3框架+RAG+记忆+观测
 3 全功能3_Ch1-3基础与推理.py 19项:pipeline五任务+内部+架构+偏见+采样+流式;微调;头家族/特征
 4 全功能4_Ch5-7数据分词NLP任务.py 23项:数据工程;手写BPE/WP/Unigram;NER/QA/翻译/摘要/MLM+指标
 5 全功能5_Ch9-12上线与对齐.py 18项:Gradio全形态;LoRA/QLoRA;GRPO全套

████ 十一、综合案例/生产综合案例/ (5个真实生产应用,Gradio Web+API) ████
 0 README_生产综合案例.py   5案例覆盖矩阵(自校验全课程功能)
 1 生产案例1_内容智能中台.py       情感/实体/翻译/摘要/填空/分词六能力+Web+API
 2 生产案例2_RAG_Agent知识平台.py  ChromaDB-RAG+MCP工具+Agent路由+mlx
 3 生产案例3_微调与对齐平台.py     ★三条【生产GPU训练】(trl Trainer/SFT/GRPO,需GPU)+GRPO原理自检
 4 生产案例4_LLM推理服务与Agent网关.py  mlx流式+OpenAI兼容API+MCP+Agent
 5 生产案例5_智能客服全栈.py       情感+NER脱敏+RAG+决策+mlx回复

████ 十二、Agent进阶案例/ (LangGraph真跑,对标 didilili 三仓库) ████
 0 README_Agent进阶案例.py   补的短板 + 本项目仍强在哪
 1 案例1_NL2SQL数据分析Agent.py  ★LangGraph状态机: 元数据检索→生成SQL→校验→SQLite执行→总结(对标shopkeeper)
 2 案例2_深度研搜多智能体.py     LangGraph多智能体: 主管supervisor+检索/写作/审核子agent+循环(对标deepsearch)
 3 案例3_混合检索RAG_LangGraph.py 向量bge+BM25+RRF融合+重排+LangGraph(对标ai-agents-from-zero)

████ 十三、Agent工程化案例/ (上线工程化) ████
 0 README_Agent工程化案例.py  补的短板清单
 1 案例1_联网搜索Agent.py     ★DuckDuckGo真联网+LangGraph+mlx综合带来源(Tavily真函数备选)
 2 案例2_FastAPI_WebSocket全栈服务/  FastAPI+WebSocket流式+前端HTML+Dockerfile+compose(TestClient真测)
     app.py / static/index.html / Dockerfile / docker-compose.yml / requirements.txt
 3 案例3_Ollama跨平台与低代码对接.py  Ollama客户端+Dify/Coze对接(真实代码,需外部服务)

████ 十四、Agent总控台.py  ★把上面Agent案例串成一个Gradio总控台(importlib复用真函数,一界面全用上) ████

████ 十五、生产级项目_NL2SQL数据分析服务/ (★★真·生产级多文件项目,对标shopkeeper) ████
 读法(自上而下)：
  README.md               先读:项目结构/流程/运行/配置/部署/简历bullet
  run.py                  本机入口(python run.py → /docs)
  app/main.py             应用工厂:中间件+异常+路由+lifespan启动
  app/core/config.py      ★pydantic配置(LLM后端stub/mlx/openai、DB sqlite/mysql 可切)
  app/core/logging.py     结构化JSON日志+request-id
  app/core/exceptions.py  自定义异常+全局处理
  app/db/models.py        SQLAlchemy ORM(Product/Order)+表元数据+种子
  app/db/database.py      引擎/会话/依赖注入/初始化
  app/schemas/query.py    请求/响应pydantic契约
  app/services/llm.py     ★LLM后端抽象(StubLLM/MlxLLM/OpenAILLM)+单例+重试
  app/services/nl2sql.py  ★★核心:LangGraph状态机[检索→生成SQL→安全校验→执行→总结]
  app/routers/query.py    POST /api/query   ·  health.py: GET /health
  tests/conftest.py       测试用stub后端+测试库
  tests/test_api.py       接口层测试   ·  test_nl2sql.py: 服务层+SQL安全校验测试
  跑: APP_LLM_BACKEND=stub pytest -q   (8 passed)  ·  python run.py (真起服务)

================================================================================
 —— 顶层导航文件 ——
   README_学习路线.py      阶段级路线(先看它)
   README_逐文件详解.py    本文件(逐文件+文件级顺序)
================================================================================
"""
if __name__ == "__main__":
    print(__doc__)
