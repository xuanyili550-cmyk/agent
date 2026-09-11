"""
================================================================================
 生产部署 · Docker / docker-compose / Kubernetes / 监控
================================================================================
 ⚠ 部署配置参考(需 Docker/K8s 环境)。把 生产01(网关) + 生产02(向量库) + vLLM 串成一套上线系统。

 为什么这么部署：
   · 为什么容器化：环境一致(本地=测试=生产)、可复现、秒级伸缩、隔离依赖。
   · 为什么 docker-compose(单机) vs K8s(集群)：小规模/开发用 compose 一键拉起多个服务;
     生产规模用 K8s——自动扩缩容(HPA)、滚动更新、健康探针、故障自愈、负载均衡。
   · 为什么服务拆分：网关/向量库/推理引擎各自一个容器,分别伸缩(GPU 贵,只给 vLLM 加副本)。
================================================================================
"""

DOCKERFILE = '''
# Dockerfile —— 打包 FastAPI 网关(生产01)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# 生产用 gunicorn+uvicorn worker,多进程扛并发
CMD ["gunicorn", "生产01_LLM服务_vLLM_FastAPI:app", "-w", "4", \\
     "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
'''

COMPOSE = '''
# docker-compose.yml —— 单机把 网关 + Qdrant 拉起来(vLLM 通常单独在 GPU 机)
services:
  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]
    volumes: ["./qdrant_storage:/qdrant/storage"]   # 数据持久化(向量库是有状态的)
  gateway:
    build: .
    ports: ["8000:8000"]
    environment:
      - VLLM_URL=http://gpu-host:8001/v1/chat/completions
      - API_KEYS=key1,key2
    depends_on: [qdrant]
  # vLLM 一般在带 GPU 的机器/节点单独跑(compose 里可用 deploy.resources 指定 GPU)
'''

K8S = '''
# vllm-deployment.yaml —— K8s 里跑 vLLM(GPU 节点),带存活/就绪探针 + HPA 自动扩缩
apiVersion: apps/v1
kind: Deployment
metadata: {name: vllm}
spec:
  replicas: 2
  selector: {matchLabels: {app: vllm}}
  template:
    metadata: {labels: {app: vllm}}
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args: ["--model","Qwen/Qwen2.5-7B-Instruct","--port","8001"]
        resources:
          limits: {nvidia.com/gpu: 1}        # 每副本 1 张卡
        readinessProbe:                       # 就绪探针:没 ready 不接流量
          httpGet: {path: /health, port: 8001}
        livenessProbe:                        # 存活探针:挂了自动重启
          httpGet: {path: /health, port: 8001}
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: {name: vllm-hpa}
spec:
  scaleTargetRef: {apiVersion: apps/v1, kind: Deployment, name: vllm}
  minReplicas: 2
  maxReplicas: 10
  metrics:                                    # GPU 利用率高就自动加副本
  - type: Resource
    resource: {name: cpu, target: {type: Utilization, averageUtilization: 70}}
'''

MONITORING = """
 监控/可观测(生产必须)：
  · 指标：延迟 p50/p95/p99、吞吐 QPS、GPU 利用率/显存、错误率、每请求 token 数/成本。
    → FastAPI 挂 prometheus 中间件,Prometheus 抓,Grafana 看板 + 告警。
  · LLM 专用：Langfuse / Phoenix 追踪每次调用的 prompt/输出/延迟/成本,排查幻觉和坏 case。
  · 日志：结构化日志(JSON) → ELK / Loki;敏感信息脱敏。
  · 灰度/回滚：新模型先小流量灰度,指标 OK 再放量;出问题一键回滚到上个版本。
"""

if __name__ == "__main__":
    print(__doc__)
    print("=== Dockerfile ===", DOCKERFILE)
    print("=== docker-compose.yml ===", COMPOSE)
    print("=== K8s(vLLM Deployment + HPA) ===", K8S)
    print("=== 监控 ===", MONITORING)
