---
title: 多功能 NLP 工具箱
emoji: 🧰
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# 多功能 NLP 工具箱

把多个 NLP 能力集成成一个多标签页产品：**情感分析 / 命名实体识别 / 翻译 / 摘要**。

## 功能特性（生产级）
- **情感分析(中文)**：`uer/roberta 点评微调`（判正面/负面）。
- **实体识别(中文)**：`uer/cluener`（人名/公司/机构/地址…；逐字模型已处理字间空格）。
- **翻译(英译中)**：`Helsinki-NLP/opus-mt-en-zh`。
- **摘要(英文)**：`t5-small`（"summarize:" 前缀）。
- 每个能力模型**惰性单例**加载；输入校验(gr.Error)；Tab 布局；示例。

## 运行
```bash
pip install -r requirements.txt
python3 app.py smoke   # 四能力各调一次自检
python3 app.py         # 起多标签页网页
```

## 部署到 Hugging Face Spaces
上传 `app.py` + `requirements.txt` + 本 `README.md`（顶部 YAML 是 Spaces 配置），自动构建出公开链接。

## 技术栈
Transformers(中文情感/NER + t5 + opus-mt) · Gradio 6

## 简历 bullet
> 开发多任务中文 NLP 工具箱（情感/实体识别/翻译/摘要），多标签页产品化界面，
> 模型惰性单例加载 + 输入校验，一键部署至 Hugging Face Spaces。
