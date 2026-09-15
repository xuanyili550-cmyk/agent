"""FastAPI 依赖项集中出口。

路由模块统一从这里拿 ``get_db``（按请求开关的 SQLAlchemy Session），而不是直接依赖 ``database.session``，
这样以后要换 Session 生命周期策略或在测试里替换依赖时只需改这一处。
"""

from ..database.session import get_db

__all__ = ["get_db"]
