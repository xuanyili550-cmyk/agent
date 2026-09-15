"""列表接口的统一分页：``?limit=&offset=`` 查询参数 + ``X-Total-Count`` 响应头。

所有 ``GET /xxx`` 列表路由都复用这里的 ``page_params`` 依赖和 ``paginate`` 帮助函数，
这样分页参数的校验规则（上限 200）和总数返回方式只在一处定义。
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Query, Response

# 单页上限：防止调用方一次拉全表把数据库和 API 拖垮
MAX_LIMIT = 200


@dataclass
class Page:
    """一页的位置：``limit`` 每页条数，``offset`` 跳过条数。"""

    limit: int
    offset: int


def page_params(
    limit: int = Query(50, ge=1, le=MAX_LIMIT, description="每页条数"),
    offset: int = Query(0, ge=0, description="跳过条数"),
) -> Page:
    """FastAPI 依赖：从查询参数解析分页信息，越界值由 Query 的 ge/le 直接拦成 422。"""
    return Page(limit=limit, offset=offset)


def paginate(query, page: Page, response: Response):
    """对 SQLAlchemy query 做 limit/offset，并把总数放进 X-Total-Count 响应头。
    响应体保持数组形式，兼容已有调用方。"""
    # order_by(None) 先去掉排序再 count：排序对计数无意义，而且部分数据库在带 ORDER BY 的子查询上 count 会报错/变慢
    total = query.order_by(None).count()
    response.headers["X-Total-Count"] = str(total)
    return query.offset(page.offset).limit(page.limit).all()
