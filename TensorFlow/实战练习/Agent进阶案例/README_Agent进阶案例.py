"""
================================================================================
 Agent 进阶案例 · 对标 ai-agents-from-zero / shopkeeper / deepsearch，补"应用工程"短板
================================================================================
 本项目原本偏 HF 底层原理 + 中文 NLP + 手写协议/算法(底层懂)；这三个案例补齐"应用工程落地"：
 LangGraph 真状态机/多智能体、NL2SQL、企业级混合检索。全部【本机真跑】(LangGraph 节点里调 mlx-lm，
 不需要 Token)，编排用确定性状态机(小模型选路由不稳)，生产可换大模型主导。

 —— 三个案例 ——
   案例1_NL2SQL数据分析Agent.py       对标 shopkeeper-agent(电商问数)：LangGraph 状态机
                                     [元数据检索→生成SQL→SQL校验→SQLite执行→自然语言总结]，SQL 校验+兜底
   案例2_深度研搜多智能体.py           对标 deepsearch-agents：LangGraph 多智能体
                                     [主管 supervisor + 检索/写作/审核 子agent + 循环反馈]，联网搜索(Tavily)给参考
   案例3_混合检索RAG_LangGraph.py     对标 ai-agents-from-zero 核心：向量(bge)+BM25(关键词)
                                     +RRF 融合+重排 + LangGraph 编排 + mlx 生成，cross-encoder 重排给生产参考

 —— 相比 didilili 三仓库，本项目补齐了 ——
   ✅ LangGraph 真状态机(节点+条件边+循环)   ✅ 多智能体协作(supervisor-worker)
   ✅ NL2SQL 数据分析                        ✅ 企业级混合检索(向量+BM25+RRF+重排)
   仍可继续补：联网搜索(Tavily 需 key)、Ollama 跨平台本地部署、Docker Compose 全栈一键、低代码 Coze/Dify。

 —— 本项目仍比 didilili 强的 ——
   HF 底层原理(手写 pipeline/BPE/WordPiece/Unigram/GRPO 三件套/MCP 协议 JSON-RPC)、中文 NLP 全模型、
   NLP 任务全覆盖(QA/翻译/摘要/MLM/困惑度/BLEU/ROUGE)。两者互补：这里懂底层，didilili 熟框架落地。
 跑：python3 案例1_NL2SQL数据分析Agent.py  (等三个都带自检)
================================================================================
"""
CASES = [
    ("案例1_NL2SQL数据分析Agent", "自然语言问数→SQL→执行→回答", "LangGraph+SQLite+mlx", "shopkeeper-agent"),
    ("案例2_深度研搜多智能体", "主管+检索/写作/审核子agent", "LangGraph 多智能体+mlx", "deepsearch-agents"),
    ("案例3_混合检索RAG_LangGraph", "向量+BM25+RRF+重排", "混合检索+LangGraph+mlx", "ai-agents-from-zero"),
]
if __name__ == "__main__":
    print(__doc__)
    print(f" {'案例':<28}{'做什么':<26}{'技术':<24}{'对标'}")
    for n, w, t, r in CASES:
        print(f" {n:<28}{w:<26}{t:<24}{r}")
