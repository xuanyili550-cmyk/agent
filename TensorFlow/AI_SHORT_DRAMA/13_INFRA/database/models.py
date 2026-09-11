from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"

    project_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    characters: Mapped[List["Character"]] = relationship(back_populates="project")
    episodes: Mapped[List["Episode"]] = relationship(back_populates="project")


class Character(Base):
    __tablename__ = "characters"

    character_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_image_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    project: Mapped["Project"] = relationship(back_populates="characters")
    assets: Mapped[List["Asset"]] = relationship(back_populates="character")


class Episode(Base):
    __tablename__ = "episodes"

    episode_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), nullable=False)
    episode_number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="draft")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    project: Mapped["Project"] = relationship(back_populates="episodes")
    scenes: Mapped[List["Scene"]] = relationship(back_populates="episode")
    assets: Mapped[List["Asset"]] = relationship(back_populates="episode")


class Scene(Base):
    __tablename__ = "scenes"

    scene_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.episode_id"), nullable=False)
    scene_number: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    episode: Mapped["Episode"] = relationship(back_populates="scenes")
    shots: Mapped[List["Shot"]] = relationship(back_populates="scene")


class Shot(Base):
    __tablename__ = "shots"

    shot_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    scene_id: Mapped[str] = mapped_column(ForeignKey("scenes.scene_id"), nullable=False)
    shot_number: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    video_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    scene: Mapped["Scene"] = relationship(back_populates="shots")
    assets: Mapped[List["Asset"]] = relationship(back_populates="shot")


class Asset(Base):
    __tablename__ = "assets"

    asset_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    character_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("characters.character_id"), nullable=True
    )
    episode_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("episodes.episode_id"), nullable=True
    )
    shot_id: Mapped[Optional[str]] = mapped_column(ForeignKey("shots.shot_id"), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    license: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    character: Mapped[Optional["Character"]] = relationship(back_populates="assets")
    episode: Mapped[Optional["Episode"]] = relationship(back_populates="assets")
    shot: Mapped[Optional["Shot"]] = relationship(back_populates="assets")
