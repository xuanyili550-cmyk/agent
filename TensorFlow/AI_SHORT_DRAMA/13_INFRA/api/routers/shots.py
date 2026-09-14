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
    return paginate(db.query(Shot).order_by(Shot.created_at.desc()), page, response)


@router.get("/{shot_id}", response_model=ShotRead)
def get_shot(shot_id: str, db: Session = Depends(get_db)) -> Shot:
    shot = db.get(Shot, shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="Shot not found")
    return shot


@router.delete("/{shot_id}", status_code=204)
def delete_shot(shot_id: str, db: Session = Depends(get_db)) -> None:
    shot = db.get(Shot, shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="Shot not found")
    db.delete(shot)
    db.commit()
