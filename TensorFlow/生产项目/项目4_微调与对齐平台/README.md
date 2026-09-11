# 微调与对齐平台（生产级 · 训练流水线）

数据校验 → SFT 训练 → 评估 → 模型版本注册,异步 job + 轮询进度。
默认 `stub` 训练后端 → **离线验证整条流程**(递减 loss 曲线);生产切 `FT_TRAIN_BACKEND=trl`(🔴 GPU)接真实 trl SFT+LoRA。

## 技术栈
数据校验 · 训练后端抽象(stub/trl) · 评估 · 模型注册表(版本+指标) · 异步训练 job · FastAPI · 测试 · Docker。

## 跑
```
pip install -r requirements.txt && python run.py    # /docs
pytest                                              # 离线验证流程(stub 训练)
```
训练=慢操作 → 异步 job + 轮询;真实训练在有 GPU 的机器上 `FT_TRAIN_BACKEND=trl`。见 `构建顺序.md`。
