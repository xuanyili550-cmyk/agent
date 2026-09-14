from __future__ import annotations

from dataclasses import dataclass

from fastapi import Query, Response

MAX_LIMIT = 200


@dataclass
class Page:
    limit: int
    offset: int


def page_params(
    limit: int = Query(50, ge=1, le=MAX_LIMIT, description="每页条数"),
    offset: int = Query(0, ge=0, description="跳过条数"),
) -> Page:
    return Page(limit=limit, offset=offset)


def paginate(query, page: Page, response: Response):
    """对 SQLAlchemy query 做 limit/offset，并把总数放进 X-Total-Count 响应头。
    响应体保持数组形式，兼容已有调用方。"""
    total = query.order_by(None).count()
    response.headers["X-Total-Count"] = str(total)
    return query.offset(page.offset).limit(page.limit).all()
