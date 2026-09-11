---
title: 文档问答 RAG
emoji: 📚
colorFrom: green
colorTo: teal
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# 文档问答 RAG 系统

上传/粘贴一篇文档 → 切块 → 向量化入库(ChromaDB) → 提问时语义检索 → 本地 LLM 生成带引用的回答。
当前最热门方向：让 LLM 基于你的私有资料回答，**准确、可溯源、免重训**。

## 功能特性（生产级）
- **中文嵌入**：`BAAI/bge-small-zh-v1.5`（★CLS 池化 + 查询加前缀，中文召回更准）。
- **真向量库**：ChromaDB（支持增量/持久化；本项目用内存实例，Spaces 够用）。
- **检索 + 生成**：语义检索 top-k → 拼进提示 → 本地 LLM(mlx-lm)生成，**要求只根据资料回答**防幻觉。
- **可溯源**：回答旁展示检索到的资料块（引用）。
- **降级兜底**：LLM 生成失败自动降级为抽取式返回最相关块。

## 运行
```bash
pip install -r requirements.txt
python3 app.py smoke   # 建库+检索+生成 自检
python3 app.py         # 起 Web：粘贴文档 → 建索引 → 提问
```

## 部署
- **本地/Mac**：直接跑（mlx-lm 用 Apple Silicon）。
- **HF Spaces / 云(Linux)**：mlx-lm 不支持非 Apple 芯片 → 把 `_llm_answer` 换成 vLLM/HF
  InferenceClient(见 `实战练习/chapter实战/生产05`)，检索部分不变。

## 技术栈
Transformers(bge) · ChromaDB · mlx-lm · Gradio 6

## 简历 bullet
> 构建文档问答 RAG 系统：中文向量嵌入(bge)+ ChromaDB 向量数据库检索 + 本地 LLM 生成带引用的回答，
> 实现切块/语义检索/防幻觉提示/降级兜底，回答可溯源。
