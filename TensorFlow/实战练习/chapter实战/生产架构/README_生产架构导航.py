"""
================================================================================
 实战练习 · 生产架构版（真实生产环境的架构参考代码，需 GPU/云端，本机跑不了）
================================================================================
 说明：`实战练习/` 里那些是“本地能跑的小版本”(学原理用)。这个 `生产架构/` 是
 “真上线怎么搭”的架构参考——用真实推理服务器、向量数据库、API 网关、容器/K8s。
 代码/配置都是真实写法，但依赖 GPU/服务，本机跑不通，重在“照着搭生产系统”。

 —— 文件 ——
   生产01_LLM服务_vLLM_FastAPI.py   用 vLLM 起 OpenAI 兼容的 LLM 服务 + FastAPI 网关(鉴权/限流/流式)
   生产02_RAG_Qdrant向量库.py       生产级 RAG：向量数据库(Qdrant/Milvus) + 嵌入服务 + 检索 + 生成
   生产03_微调_多卡_Accelerate.py    云端多卡微调：Accelerate/DeepSpeed + 配置 + 推模型仓库
   生产04_GRPO_trl_vLLM.py          生产级 GRPO：trl.GRPOTrainer + vLLM 加速采样 + 多卡
   生产05_调用LLM服务_客户端.py       客户端侧：OpenAI 兼容客户端 / HF InferenceClient / 流式 / 结构化 JSON
   部署_Docker_Compose_K8s.py       Dockerfile / docker-compose / K8s 清单 / 监控

 —— 美国 vs 中国 常用技术栈对照（面试常问）——
   环节            美国常用                          中国常用
   LLM 推理服务    vLLM / TGI(HuggingFace) / TensorRT-LLM   vLLM / LMDeploy(商汤) / TensorRT-LLM
   向量数据库      Pinecone / Weaviate / pgvector / Milvus  Milvus(Zilliz,中国团队) / 腾讯/阿里向量库
   模型托管/推理   AWS SageMaker / Bedrock / Together / Fireworks  阿里 PAI-EAS / 腾讯 TI / 火山方舟 / 硅基流动
   训练平台        AWS/GCP + DeepSpeed / Ray / SkyPilot     阿里 PAI / 华为 ModelArts / 火山 / 自建 K8s
   编排/部署       Kubernetes + KServe / Ray Serve          Kubernetes + 自建 / KServe
   网关/服务       FastAPI + Nginx / Envoy                  FastAPI + Nginx / APISIX
   监控            Prometheus + Grafana / Langfuse          Prometheus + Grafana / 自建
   实验/追踪       Weights & Biases / MLflow                MLflow / SwanLab(国产) / W&B

 —— 生产三条铁律 ——
   1) 推理和训练分离：训练产出模型 → 存模型仓库 → 推理服务加载。别在一个进程里又训又服务。
   2) 无状态 + 可横向扩展：服务本身无状态，状态(向量/缓存)放外部(向量库/Redis)，副本随流量伸缩。
   3) 可观测：延迟 p50/p95/p99、吞吐 QPS、GPU 利用率、错误率都要有监控和告警。
================================================================================
"""
print(__doc__)
