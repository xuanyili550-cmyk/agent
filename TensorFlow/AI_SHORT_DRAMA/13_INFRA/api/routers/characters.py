"""/characters 路由：角色的 CRUD。

角色带 reference_image_path（参考图），生图阶段靠它做人物一致性；创建时校验 project 存在，
避免出现挂在不存在项目下的孤儿角色。
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from ...database.models import Character, Project
from ..deps import get_db
from ..pagination import Page, page_params, paginate
from ..schemas import CharacterCreate, CharacterRead
from ..security import require_api_key

router = APIRouter(prefix="/characters", tags=["characters"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=CharacterRead, status_code=201)
def create_character(payload: CharacterCreate, db: Session = Depends(get_db)) -> Character:
    """创建角色；project_id 不存在返回 404（用 404 而不是 422：是引用的资源不存在，不是请求格式问题）。"""
    if db.get(Project, payload.project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    character = Character(
        project_id=payload.project_id,
        name=payload.name,
        description=payload.description,
        reference_image_path=payload.reference_image_path,
    )
    db.add(character)
    db.commit()
    db.refresh(character)
    return character


@router.get("", response_model=List[CharacterRead])
def list_characters(response: Response, db: Session = Depends(get_db), page: Page = Depends(page_params)) -> List[Character]:
    """分页列出角色，按创建时间倒序。"""
    return paginate(db.query(Character).order_by(Character.created_at.desc()), page, response)


@router.get("/{character_id}", response_model=CharacterRead)
def get_character(character_id: str, db: Session = Depends(get_db)) -> Character:
    """按 id 取单个角色，不存在返回 404。"""
    character = db.get(Character, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return character


@router.delete("/{character_id}", status_code=204)
def delete_character(character_id: str, db: Session = Depends(get_db)) -> None:
    """删除角色，成功返回 204。"""
    character = db.get(Character, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    db.delete(character)
    db.commit()
