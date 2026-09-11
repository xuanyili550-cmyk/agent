---
title: 智能客服助手
emoji: 🎧
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# 智能客服助手（集大成）

一条客服工单进来，系统自动：**判情绪(定优先级) → 抽实体(路由/脱敏) → 语义检索 FAQ → 决策(自动回复/转人工)**。
把「情感分类 + 实体识别 + RAG 检索 + 阈值分流」串成一个**完整可上线的客服产品**。

## 功能特性（生产级）
- **全中文模型**：情感(`uer/dianping`) + NER(`uer/cluener`) + 检索(`bge-small-zh`，CLS 池化+查询前缀)。
- **阈值分流**：情感置信度低 / 无对口 FAQ → 转人工；负面工单标记优先处理。
- **结构化输出**：情感/实体/命中 FAQ/决策/建议回复。
- 三模型惰性单例、输入校验(gr.Error)、队列并发、可部署。

## 运行
```bash
pip install -r requirements.txt
python3 app.py smoke   # 跑几条工单自检
python3 app.py         # 起客服后台 Web
```

## 部署到 Hugging Face Spaces
上传 `app.py` + `requirements.txt` + 本 `README.md`，自动构建出公开链接。

## 技术栈
Transformers(中文情感/NER) · bge-small-zh · Gradio 6

## 简历 bullet
> 独立开发并部署智能客服助手：整合中文情感分类、命名实体识别与语义检索(RAG)，实现工单情绪分流、
> 实体路由与自动回复/转人工决策，结构化输出 + 一键部署至 Hugging Face Spaces。
