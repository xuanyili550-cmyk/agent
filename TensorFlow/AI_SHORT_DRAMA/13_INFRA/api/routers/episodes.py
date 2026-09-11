from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...database.models import Episode, Project
from ..deps import get_db
from ..schemas import EpisodeCreate, EpisodeRead

router = APIRouter(prefix="/episodes", tags=["episodes"])


@router.post("", response_model=EpisodeRead, status_code=201)
def create_episode(payload: EpisodeCreate, db: Session = Depends(get_db)) -> Episode:
    if db.get(Project, payload.project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    episode = Episode(
        project_id=payload.project_id,
        episode_number=payload.episode_number,
        title=payload.title,
        status=payload.status,
    )
    db.add(episode)
    db.commit()
    db.refresh(episode)
    return episode


@router.get("", response_model=List[EpisodeRead])
def list_episodes(db: Session = Depends(get_db)) -> List[Episode]:
    return db.query(Episode).order_by(Episode.created_at.desc()).all()


@router.get("/{episode_id}", response_model=EpisodeRead)
def get_episode(episode_id: str, db: Session = Depends(get_db)) -> Episode:
    episode = db.get(Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode


@router.delete("/{episode_id}", status_code=204)
def delete_episode(episode_id: str, db: Session = Depends(get_db)) -> None:
    episode = db.get(Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    db.delete(episode)
    db.commit()
