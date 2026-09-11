# 语义检索服务 · 纯 Hugging Face 底层（生产级）

一个可上线的**语义检索 / Embedding 微服务**，刻意**只用 HF 底层原理**手工实现，不用 `pipeline()` 快捷方式、
不用 sentence-transformers、不接外部服务——把"pipeline 内部三步"落成生产代码。

## 用到的底层原理（这也是本案例的价值）
| 底层原理 | 在哪 |
|---------|------|
| 分词底层：padding / truncation / **attention_mask** | `services/embedding.py` |
| `AutoModel` 前向拿 **last_hidden_state**（不用 pipeline） | `services/embedding.py` |
| **手写 mean 池化(mask 加权)** / CLS 池化 —— 为什么要 mask | `services/pooling.py` |
| **L2 归一化 → 点积=余弦** | `services/pooling.py` / `index.py` |
| **手写数值稳定 softmax** | `services/pooling.py` |
| 动态批处理（一批一起前向） | `services/embedding.py` |
| 向量检索（余弦 top-k，可平替 FAISS） | `services/index.py` |
| **cross-encoder 重排**（`AutoModelForSequenceClassification`→logit） | `services/rerank.py` |

## 快速开始
```bash
cd 生产_语义检索_纯HF底层
pip install -r requirements.txt
python run.py                       # → http://127.0.0.1:8000/docs
```
首次调用 /embed 或 /search 会下载嵌入模型(all-MiniLM-L6-v2, ~90MB)。

## 接口
- `POST /api/embed`      文本 → 向量（自建 Embeddings API）
- `POST /api/documents`  文档入库（编码存向量 + 原文）；`GET /documents/stats`；`DELETE /documents`
- `POST /api/search`     查询 → 向量粗召回 →(可选)重排 → 返回原文+分数
- `GET  /health`         模型/池化/库规模

示例：
```bash
curl -X POST localhost:8000/api/documents -H 'content-type: application/json' \
  -d '{"documents":[{"id":"1","text":"退货政策是7天无理由"},{"id":"2","text":"满99包邮"}]}'
curl -X POST localhost:8000/api/search -H 'content-type: application/json' \
  -d '{"query":"怎么退货","top_k":2}'
```

## 生产细节（已处理）
- **模型单例 + 懒加载 + 双检锁**：进程内只加载一次，首个请求才载入(启动快)。
- **inference_mode / eval**：关梯度关 dropout，省显存更快。
- **池化配对**：mean↔MiniLM/E5、cls↔bge，配错向量质量下降(可配 `EMB_POOLING`)。
- **两阶段检索**：向量粗召回 + cross-encoder 精排（`EMB_RERANK_ENABLED=true` 开启）。
- **持久化 + 线程安全**：向量库/原文落盘，加锁防并发竞态。
- **输入限制/统一异常/健康检查**齐全。

## 配置（环境变量 EMB_*）
`EMB_MODEL_NAME`(all-MiniLM-L6-v2) `EMB_POOLING`(mean|cls) `EMB_NORMALIZE`(true)
`EMB_QUERY_PREFIX` `EMB_PASSAGE_PREFIX`（E5/bge 用）`EMB_RERANK_ENABLED` `EMB_RERANK_MODEL`
`EMB_DEVICE`(auto) `EMB_INDEX_DIR` `EMB_BATCH_SIZE` `EMB_MAX_SEQ_LEN`

换中文：`EMB_MODEL_NAME=BAAI/bge-small-zh-v1.5 EMB_POOLING=cls`。

## 测试
```bash
pytest        # 池化/归一化/softmax/向量检索的底层数学，全部用合成张量验证，不下模型
```
