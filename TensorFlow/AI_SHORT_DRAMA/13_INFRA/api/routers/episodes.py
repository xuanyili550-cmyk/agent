from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from ...database.models import Episode, Project
from ..deps import get_db
from ..pagination import Page, page_params, paginate
from ..schemas import EpisodeCreate, EpisodeRead, EpisodeReviewRequest
from ..security import require_api_key

router = APIRouter(prefix="/episodes", tags=["episodes"], dependencies=[Depends(require_api_key)])


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
def list_episodes(response: Response, db: Session = Depends(get_db), page: Page = Depends(page_params)) -> List[Episode]:
    return paginate(db.query(Episode).order_by(Episode.created_at.desc()), page, response)


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


@router.get("/{episode_id}/script")
def get_episode_script(episode_id: str, db: Session = Depends(get_db)) -> dict:
    """剧本正文 + 结构化 Episode + 质检官/总编审判定，供审核页面展示。"""
    episode = db.get(Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return {
        "episode": episode.data,
        "script": episode.script,
        "editorial": episode.editorial,
        "review_status": episode.review_status,
        "review_notes": episode.review_notes,
    }


@router.post("/{episode_id}/review", response_model=EpisodeRead)
def review_episode(episode_id: str, payload: EpisodeReviewRequest, db: Session = Depends(get_db)) -> Episode:
    """人工审校单集：approved 后该集才能进入生产（也可通过 /pipelines/{run_id}/approve 批量放行）。"""
    episode = db.get(Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    episode.review_status = payload.decision
    episode.review_notes = payload.notes
    db.commit()
    db.refresh(episode)
    return episode
