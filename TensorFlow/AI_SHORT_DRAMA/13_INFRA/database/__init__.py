"""13_INFRA.database 包：平台的持久层（SQLAlchemy 表定义、会话管理、仓储函数、alembic 迁移）。

这个包本身不导出任何名字，调用方按需 import 子模块：
- ``models``：表定义（业务对象整体存 JSON ``data`` 列，主键带项目前缀）；
- ``session``：engine / SessionLocal / ``session_scope()``；
- ``repository``：队列任务和 API 读写数据库的唯一入口。
故意留空是为了避免 import 副作用：``session`` 一旦被 import 就会读 settings 并建 engine，
而 alembic 的 env.py 只想拿 ``models.Base.metadata``，不希望顺带连库。
"""
