from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

__all__ = [
    "QCDecision",
    "QCItemScore",
    "QCReport",
    "QCThresholds",
    "build_qc_report",
    "run_full_qc",
]

PathLike = Union[str, Path]


class QCDecision(str, Enum):
    APPROVED = "approved"
    RETRY = "retry"


class QCItemScore(BaseModel):
    name: str
    score: Optional[float] = None
    threshold: Optional[float] = None
    passed: bool
    detail: Optional[str] = None


class QCThresholds(BaseModel):
    character_min_similarity: float = 0.75
    scene_min_similarity: float = 0.50
    audio_required: bool = True
    video_required: bool = True


class QCReport(BaseModel):
    asset_id: Optional[str] = None
    shot_id: Optional[str] = None
    episode_id: Optional[str] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    character: Optional[QCItemScore] = None
    scene: Optional[QCItemScore] = None
    video: Optional[QCItemScore] = None
    audio: Optional[QCItemScore] = None
    decision: QCDecision
    retry_reasons: List[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.decision == QCDecision.APPROVED

    def to_json(self, path: Optional[PathLike] = None) -> str:
        payload = self.model_dump(mode="json")
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


def _character_item(result, threshold: float) -> QCItemScore:
    passed = result.similarity >= threshold
    return QCItemScore(
        name="character_consistency",
        score=result.similarity,
        threshold=threshold,
        passed=passed,
        detail=None if passed else (
            f"character similarity {result.similarity:.3f} below threshold {threshold:.3f}"
        ),
    )


def _scene_item(result, threshold: float) -> QCItemScore:
    passed = result.similarity >= threshold
    return QCItemScore(
        name="scene_alignment",
        score=result.similarity,
        threshold=threshold,
        passed=passed,
        detail=None if passed else (
            f"scene alignment {result.similarity:.3f} below threshold {threshold:.3f}"
        ),
    )


def _video_item(result) -> QCItemScore:
    passed = result.passed
    reasons = []
    if not result.duration_ok:
        reasons.append(f"duration {result.probe.duration_sec:.2f}s out of range")
    if not result.resolution_ok:
        reasons.append(f"resolution {result.probe.width}x{result.probe.height} too small")
    if not result.black_frames_ok:
        reasons.append(f"black frame ratio {result.black_frame_ratio:.2%} too high")
    if not result.still_frames_ok:
        reasons.append(f"still frame ratio {result.still_frame_ratio:.2%} too high")
    return QCItemScore(
        name="video_quality",
        score=1.0 - max(result.black_frame_ratio, result.still_frame_ratio * 0.5),
        threshold=None,
        passed=passed,
        detail="; ".join(reasons) or None,
    )


def _audio_item(result) -> QCItemScore:
    passed = result.passed
    reasons = []
    if not result.loudness_ok:
        reasons.append(f"loudness {result.loudness_dbfs:.1f} dBFS out of range")
    if not result.silence_ok:
        reasons.append(f"excess silence: {result.total_silence_sec:.1f}s total")
    return QCItemScore(
        name="audio_quality",
        score=result.loudness_dbfs,
        threshold=None,
        passed=passed,
        detail="; ".join(reasons) or None,
    )


def build_qc_report(
    *,
    asset_id: Optional[str] = None,
    shot_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    character_result=None,
    scene_result=None,
    video_result=None,
    audio_result=None,
    thresholds: Optional[QCThresholds] = None,
) -> QCReport:
    """Pure aggregation step: Generate -> QC -> PASS? -> Retry/Approved.

    Takes already-computed results from the four QC checkers (any subset may be
    ``None`` if that dimension isn't applicable to this asset) and produces a single
    QCReport with a pass/fail decision and, on failure, a list of retry reasons.
    """
    thresholds = thresholds or QCThresholds()
    items: Dict[str, QCItemScore] = {}
    retry_reasons: List[str] = []

    if character_result is not None:
        item = _character_item(character_result, thresholds.character_min_similarity)
        items["character"] = item
        if not item.passed:
            retry_reasons.append(item.detail or "character_consistency failed")

    if scene_result is not None:
        item = _scene_item(scene_result, thresholds.scene_min_similarity)
        items["scene"] = item
        if not item.passed:
            retry_reasons.append(item.detail or "scene_alignment failed")

    if video_result is not None:
        item = _video_item(video_result)
        items["video"] = item
        if not item.passed:
            retry_reasons.append(item.detail or "video_quality failed")
    elif thresholds.video_required:
        retry_reasons.append("video QC result missing but required")

    if audio_result is not None:
        item = _audio_item(audio_result)
        items["audio"] = item
        if not item.passed:
            retry_reasons.append(item.detail or "audio_quality failed")
    elif thresholds.audio_required:
        retry_reasons.append("audio QC result missing but required")

    decision = QCDecision.APPROVED if not retry_reasons else QCDecision.RETRY

    return QCReport(
        asset_id=asset_id,
        shot_id=shot_id,
        episode_id=episode_id,
        character=items.get("character"),
        scene=items.get("scene"),
        video=items.get("video"),
        audio=items.get("audio"),
        decision=decision,
        retry_reasons=retry_reasons,
    )


def run_full_qc(
    *,
    asset_id: Optional[str] = None,
    shot_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    reference_character_image: Optional[PathLike] = None,
    generated_image: Optional[PathLike] = None,
    character_id: Optional[str] = None,
    scene_description: Optional[str] = None,
    scene_media_image: Optional[PathLike] = None,
    video_path: Optional[PathLike] = None,
    audio_path: Optional[PathLike] = None,
    thresholds: Optional[QCThresholds] = None,
) -> QCReport:
    """End-to-end orchestration: runs whichever of the four checkers have inputs
    provided, then aggregates into a QCReport via build_qc_report.

    Heavy dependencies (transformers/torch for CLIP, opencv, pydub) are imported
    lazily here so importing this module alone stays lightweight.
    """
    character_result = None
    scene_result = None
    video_result = None
    audio_result = None

    if reference_character_image is not None and generated_image is not None:
        from ..character.consistency_checker import CharacterConsistencyChecker

        checker = CharacterConsistencyChecker()
        character_result = checker.score(
            reference_character_image, generated_image, character_id=character_id
        )

    if scene_description is not None and scene_media_image is not None:
        from ..scene.scene_qc import SceneQC

        scene_checker = SceneQC()
        scene_result = scene_checker.score_image(scene_description, scene_media_image)

    if video_path is not None:
        from ..video.video_qc import VideoQC

        video_checker = VideoQC()
        video_result = video_checker.check(video_path)

    if audio_path is not None:
        from ..audio.audio_qc import AudioQC

        audio_checker = AudioQC()
        audio_result = audio_checker.check(audio_path)

    return build_qc_report(
        asset_id=asset_id,
        shot_id=shot_id,
        episode_id=episode_id,
        character_result=character_result,
        scene_result=scene_result,
        video_result=video_result,
        audio_result=audio_result,
        thresholds=thresholds,
    )
