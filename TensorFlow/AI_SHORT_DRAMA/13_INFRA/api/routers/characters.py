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
    return paginate(db.query(Character).order_by(Character.created_at.desc()), page, response)


@router.get("/{character_id}", response_model=CharacterRead)
def get_character(character_id: str, db: Session = Depends(get_db)) -> Character:
    character = db.get(Character, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return character


@router.delete("/{character_id}", status_code=204)
def delete_character(character_id: str, db: Session = Depends(get_db)) -> None:
    character = db.get(Character, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    db.delete(character)
    db.commit()
