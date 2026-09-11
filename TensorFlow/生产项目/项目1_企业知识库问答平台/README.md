# 企业知识库问答平台（生产级 · 综合多技术栈）

文档 → 解析 → 切块 → 嵌入 → 向量检索 → 重排 → LLM 生成带来源;Agent 决定"查库 or 直答"。
**默认全 stub 后端 → 零下载、离线端到端可跑可测**;生产切 `hf` 嵌入 + `openai/mlx` LLM 即用真模型。

## 综合的技术栈
文档解析(PyMuPDF) · 切块(中文标点+重叠) · 嵌入(可切 stub/HF 底层) · 向量检索(余弦,可平替FAISS) ·
重排(词面/cross-encoder) · LLM(可切 stub/mlx/openai) · Agent 路由 · 缓存 · FastAPI · 前端 · Docker · 测试。

## 跑
```
pip install -r requirements.txt
python run.py            # → http://127.0.0.1:8000  (Swagger /docs)
pytest                   # 离线端到端测试(stub 后端,不下模型)
```
切生产后端:`KB_EMBED_BACKEND=hf KB_LLM_BACKEND=openai KB_OPENAI_BASE=http://vllm:8001/v1 python run.py`

## 架构
见 `构建顺序.md`(按依赖 bottom-up 写) 和 `../README_如何写生产项目与架构规范.md`(总纲)。
分层:core(地基) → schemas(契约) → services(业务) → routers(接口) → main(组装)。

## 生产细节
后端可切换(测试离线) · 输入校验 · 单例懒加载 · 内容哈希缓存 · 统一异常 · 健康检查 · 临时/上限保护 · Docker 化。
