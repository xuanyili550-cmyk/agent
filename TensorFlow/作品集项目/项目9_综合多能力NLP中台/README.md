---
title: 多能力NLP中台
emoji: 🧰
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# 多能力 NLP 中台（综合 · 生产级）

把多项 NLP 能力集成到一个带 Tab 的 Web 应用 + 统一 REST API：**情感分析 / 零样本分类 / 命名实体识别 / 完形填空**。
（避开 transformers v5 已移除的 summarization/translation/QA pipeline,全用仍在的任务。）

## 功能特性（生产级）
- **多能力单例**：各能力模型懒加载、分别单例,用到才载。
- **Tab 界面 + 统一 API**：`/api/sentiment` `/api/ner` `/api/zeroshot` `/api/fillmask` `/api/health`。
- **输入校验 + 中文模型**(情感用中文点评模型)。

## 运行
```
python3 app.py smoke   # 离线自检(不下模型)
python3 app.py / api   # Gradio / FastAPI
```
