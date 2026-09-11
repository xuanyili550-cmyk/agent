---
title: 语义检索服务
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# 语义检索服务（纯 HF 底层 · 生产级）

把一批文档建成向量库,按"意思"检索——**刻意只用 `AutoTokenizer+AutoModel` 手写**(不用 pipeline/sentence-transformers),
展示 pipeline 内部三步的生产手写版。

## 功能特性（生产级）
- **纯底层**：分词(attention_mask)→AutoModel 前向→**手写 mean 池化(mask 加权)**→L2 归一化→余弦检索。
- **模型单例**：冷启动加载一次常驻。
- **输入校验**：空文档/空查询用 `gr.Error` 提示。
- **Web + API**：Gradio 界面;FastAPI `POST /api/search`、`GET /api/health`。

## 运行
```
python3 app.py smoke   # 离线自检(校验+构建UI,不下模型)
python3 app.py         # Gradio 网页(首次下 all-MiniLM-L6-v2 ~90MB)
python3 app.py api     # FastAPI(网页挂 /,API 在 /api/search)
```
