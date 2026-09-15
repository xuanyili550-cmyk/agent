"""/assets 路由：生成素材（图片/视频/音频文件）登记的 CRUD。

Asset 记录的是"某个文件是用什么模型、什么 prompt、什么 seed 生成的"这类溯源信息，
供 QC 回溯、重新生成和版权/许可审计使用；文件本体在存储层，这里只存路径。
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from ...database.models import Asset
from ..deps import get_db
from ..pagination import Page, page_params, paginate
from ..schemas import AssetCreate, AssetRead
from ..security import require_api_key

router = APIRouter(prefix="/assets", tags=["assets"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=AssetRead, status_code=201)
def create_asset(payload: AssetCreate, db: Session = Depends(get_db)) -> Asset:
    """登记一条素材。字段和 ORM 模型一一对应，直接 ``**model_dump()`` 展开；关联 id 不做存在性校验（素材可能先于关联对象登记）。"""
    asset = Asset(**payload.model_dump())
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


@router.get("", response_model=List[AssetRead])
def list_assets(response: Response, db: Session = Depends(get_db), page: Page = Depends(page_params)) -> List[Asset]:
    """分页列出素材，按创建时间倒序。"""
    return paginate(db.query(Asset).order_by(Asset.created_at.desc()), page, response)


@router.get("/{asset_id}", response_model=AssetRead)
def get_asset(asset_id: str, db: Session = Depends(get_db)) -> Asset:
    """按 id 取单条素材，不存在返回 404。"""
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


@router.delete("/{asset_id}", status_code=204)
def delete_asset(asset_id: str, db: Session = Depends(get_db)) -> None:
    """删除素材登记（只删数据库记录，不删存储里的文件），成功返回 204。"""
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    db.delete(asset)
    db.commit()
