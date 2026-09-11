# LLM 推理网关（生产级 · 综合多技术栈）

统一的 LLM 网关:多后端路由(stub/openai/mlx) + 提示缓存 + 限流 + **失败降级兜底** + OpenAI 风格接口。
默认 stub → 离线可跑可测;生产切 `GW_BACKEND=openai GW_OPENAI_BASE=http://vllm:8001/v1`。

## 技术栈
多后端抽象+路由 · 降级兜底(真实后端挂→回退 stub) · 内容哈希缓存 · 滑动窗口限流 · FastAPI · 测试 · Docker。

## 跑
```
pip install -r requirements.txt && python run.py   # /docs
pytest                                             # 离线测试
curl -X POST localhost:8000/api/chat -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"你好"}]}'
```
架构见 `构建顺序.md` 与 `../README_如何写生产项目与架构规范.md`。
