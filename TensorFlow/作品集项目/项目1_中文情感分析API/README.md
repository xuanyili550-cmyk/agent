---
title: 中文情感分析
emoji: 😊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# 中文情感分析 API + Web 界面

把一个中文情感分类模型包成【带 Web 界面 + REST API】的在线服务。输入中文评论/文本，
返回正面/负面的概率。

## 功能特性（生产级）
- **中文模型**：`uer/roberta-base-finetuned-dianping-chinese`（中文点评微调；英文模型喂中文会失准）。
- **模型单例加载**：冷启动只加载一次、常驻复用，避免每次请求重载。
- **输入校验**：空输入用 `gr.Error` 弹友好提示。
- **Web 界面**：Gradio，带示例、概率可视化。
- **REST API**：FastAPI 挂载，`POST /api/predict` 供其他服务调用，`GET /api/health` 探活。
- **并发**：Gradio 队列。

## 运行
```bash
pip install -r requirements.txt
python3 app.py smoke   # 自检(不起服务)
python3 app.py         # 起 Web 界面 http://localhost:7860
python3 app.py api     # 起 FastAPI：网页 / + API /api/predict
```
调用 API：
```bash
curl -X POST http://localhost:7860/api/predict -H "Content-Type: application/json" \
     -d '{"text":"这家店服务很好，下次还来！"}'
```

## 部署到 Hugging Face Spaces
1. 新建 Space（SDK 选 Gradio）。
2. 上传 `app.py` + `requirements.txt` + 本 `README.md`（顶部 YAML 是 Spaces 配置）。
3. Space 自动构建并给出公开链接。

## 技术栈
Transformers · PyTorch(mps) · Gradio 6 · FastAPI

## 简历 bullet
> 独立开发并部署中文情感分析在线服务（Transformers 中文模型 + Gradio + FastAPI），
> 实现模型单例加载、输入校验与 REST API，一键部署至 Hugging Face Spaces 获得公开访问链接。
