# 内容审核中台（生产级 · 综合多技术栈）

文本 → 敏感词/PII 检测 → 风险打分 → 分级(通过/人工复审/拦截) → PII 脱敏。单条 + 异步批量。
默认规则后端 → 离线可跑可测;生产切 `MOD_RISK_BACKEND=model` 接毒性分类模型。

## 技术栈
规则检测(敏感词+正则PII) · 风险打分(可切 rule/model) · 分级决策 · PII 脱敏 · 异步批量 job · FastAPI · 测试 · Docker。

## 跑
```
pip install -r requirements.txt && python run.py   # /docs
pytest                                             # 离线测试
```
架构见 `构建顺序.md` 与 `../README_如何写生产项目与架构规范.md`。
