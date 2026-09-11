# NL2SQL 数据分析服务（生产级）

自然语言问数据的 AI Agent 服务。对标 shopkeeper-agent，是一个**完整工程结构的生产级项目**
（不是单文件 demo）：分层架构 + 配置管理 + ORM + 结构化日志 + 中间件 + 统一异常 + 重试 +
测试 + 多阶段 Docker。LLM/数据库后端**配置化**：本机 SQLite+mlx，改配置即切 MySQL+vLLM。

## 项目结构
```
app/
  main.py              # 应用工厂：中间件(CORS/请求日志/request-id) + 全局异常 + 路由 + 启动
  core/
    config.py          # pydantic-settings 配置(多环境、后端可切)
    logging.py         # 结构化 JSON 日志 + request-id 上下文
    exceptions.py      # 自定义异常 + 全局异常处理
  db/
    models.py          # SQLAlchemy ORM(Product/Order) + 表元数据 + 种子
    database.py        # 引擎/会话/依赖注入/初始化
  schemas/query.py     # 请求/响应模型(pydantic 契约)
  services/
    llm.py             # LLM 后端抽象(stub/mlx/openai) + 单例 + 重试
    nl2sql.py          # 核心：LangGraph 状态机[检索元数据→生成SQL→安全校验→执行→总结]
  routers/
    query.py           # POST /api/query
    health.py          # GET /health
tests/                 # pytest：接口层 + 服务层 + SQL 安全校验
Dockerfile             # 多阶段构建 + 非 root + healthcheck
docker-compose.yml     # api + mysql + ollama 一键起
```

## 核心流程（NL2SQL Agent，LangGraph 状态机）
`检索相关表(元数据) → LLM 生成 SQL → 安全校验(只放行 SELECT、禁写、可解析) → [不通过→规则兜底] → 执行 → LLM 自然语言总结`

## 运行
```bash
pip install -r requirements.txt
# 本机开发(mlx 生成 SQL)
python run.py                 # http://localhost:8000/docs
# 或
uvicorn app.main:app --reload

# 调接口
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" \
     -d '{"question":"各城市销售额是多少？"}'
```

## 测试
```bash
APP_LLM_BACKEND=stub pytest -q     # 用 stub 后端(确定性、不下模型、秒过)
```

## 配置（.env，前缀 APP_）
- `APP_LLM_BACKEND`：`stub`(测试) / `mlx`(本机 Mac) / `openai`(生产，连 vLLM/Ollama/云端)
- `APP_DATABASE_URL`：`sqlite:///...`(本机) / `mysql+pymysql://...`(生产)
- 见 `.env.example`

## 部署
```bash
docker compose up --build      # api + MySQL + Ollama；Ollama 起后 pull 模型
```
生产大规模把 Ollama 换 vLLM(GPU)；数据库用托管 MySQL；加 Alembic 迁移、Prometheus 指标、CI/CD。

## 安全
- SQL 只放行 `SELECT`，正则拦截 `INSERT/UPDATE/DELETE/DROP/...`；执行前 `EXPLAIN` 校验可解析。
- 统一异常处理不泄露栈；非 root 容器运行；输入 pydantic 校验。

## 简历 bullet
> 独立设计并实现生产级 NL2SQL 数据分析 Agent 服务（FastAPI + LangGraph + SQLAlchemy）：
> 分层架构、LLM/DB 后端配置化(mlx/vLLM、SQLite/MySQL)、SQL 安全校验与规则兜底、结构化日志与
> request-id 追踪、pytest 测试、多阶段 Docker + docker-compose 一键部署。
```
