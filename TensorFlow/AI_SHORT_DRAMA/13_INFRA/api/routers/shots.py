"""/shots 路由：镜头的 CRUD。

镜头挂在场景（Scene）下；创建时校验 scene 存在。流水线生成的镜头由 story_task 落库，
这里主要供手工补录/调试和列表查看。
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from ...database.models import Scene, Shot
from ..deps import get_db
from ..pagination import Page, page_params, paginate
from ..schemas import ShotCreate, ShotRead
from ..security import require_api_key

router = APIRouter(prefix="/shots", tags=["shots"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=ShotRead, status_code=201)
def create_shot(payload: ShotCreate, db: Session = Depends(get_db)) -> Shot:
    """创建镜头；scene_id 不存在返回 404。"""
    if db.get(Scene, payload.scene_id) is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    shot = Shot(
        scene_id=payload.scene_id,
        shot_number=payload.shot_number,
        description=payload.description,
        duration_sec=payload.duration_sec,
        video_path=payload.video_path,
    )
    db.add(shot)
    db.commit()
    db.refresh(shot)
    return shot


@router.get("", response_model=List[ShotRead])
def list_shots(response: Response, db: Session = Depends(get_db), page: Page = Depends(page_params)) -> List[Shot]:
    """分页列出镜头，按创建时间倒序。"""
    return paginate(db.query(Shot).order_by(Shot.created_at.desc()), page, response)


@router.get("/{shot_id}", response_model=ShotRead)
def get_shot(shot_id: str, db: Session = Depends(get_db)) -> Shot:
    """按 id 取单个镜头，不存在返回 404。"""
    shot = db.get(Shot, shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="Shot not found")
    return shot


@router.delete("/{shot_id}", status_code=204)
def delete_shot(shot_id: str, db: Session = Depends(get_db)) -> None:
    """删除镜头，成功返回 204。"""
    shot = db.get(Shot, shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="Shot not found")
    db.delete(shot)
    db.commit()
