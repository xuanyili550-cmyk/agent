---
title: 零样本文本分类
emoji: 🏷️
colorFrom: green
colorTo: teal
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# 零样本文本分类（生产级）

**不用训练**,给一段文本 + 你自己的候选标签,模型(NLI)直接判它属于哪一类。适合标签经常变的场景。

## 功能特性（生产级）
- **零样本**：`pipeline("zero-shot-classification")`,标签运行时传,改标签不用重训。
- **多语言**：默认多语言 NLI 模型,中英文都能分。
- **模型单例 + 输入校验 + Web + FastAPI API**。

## 运行
```
python3 app.py smoke   # 离线自检(不下模型)
python3 app.py         # Gradio 网页(首次下 NLI 模型)
python3 app.py api     # FastAPI
```
