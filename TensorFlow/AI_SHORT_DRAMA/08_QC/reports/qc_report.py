"""QC 报告聚合：把四个检查器（角色一致性 / 场景对齐 / 视频 / 音频）的结果汇总成一份 QCReport，
并给出 APPROVED / RETRY 决定。

流水线位置：08_QC 的出口——"Generate -> QC -> PASS? -> Retry/Approved" 里的 PASS? 判断就在这里。
13_INFRA/queue/shot_task.py 读 ``decision`` 决定是把素材推进到 09_POST 还是喂给三级重试阶梯，
``retry_reasons`` 则作为改写提示词时的上下文。
为什么拆成 build_qc_report（纯聚合）和 run_full_qc（端到端）：前者不依赖任何重库，
可以用手工构造的结果对象单测阈值逻辑；后者才真正 import torch / opencv / pydub 跑检查。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

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
    """QC 最终决定：通过进入后期，或退回重试。只有两个值，没有"人工复核"——那由编审 Agent 另行处理。"""

    APPROVED = "approved"
    RETRY = "retry"


class QCItemScore(BaseModel):
    """单项检查的打分：分数 / 阈值可为 None（视频、音频是多条件布尔判断，没有单一阈值），``detail`` 只在失败时填写。"""

    name: str
    score: Optional[float] = None
    threshold: Optional[float] = None
    passed: bool
    detail: Optional[str] = None


class QCThresholds(BaseModel):
    """可配置的通过阈值。

    - ``character_min_similarity=0.75``：图像-图像余弦映射到 [0,1] 后，同一角色不同镜头通常在 0.8 以上，
      无关人物约 0.6-0.7，0.75 是经验分界；
    - ``scene_min_similarity=0.50``：对应 SceneQC 缩放前原始 CLIP 分 0.2，是"文本与画面基本相关"的常见下界；
    - ``audio_required`` / ``video_required``：缺少对应结果时是否直接判 RETRY——默认必须有，
      避免因为检查器没跑就"默认通过"。
    """

    character_min_similarity: float = 0.75
    scene_min_similarity: float = 0.50
    audio_required: bool = True
    video_required: bool = True


class QCReport(BaseModel):
    """一份完整的 QC 报告：素材 / 镜头 / 剧集 ID、四项检查（缺省为 None）、决定和重试原因。"""

    asset_id: Optional[str] = None
    shot_id: Optional[str] = None
    episode_id: Optional[str] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    character: Optional[QCItemScore] = None
    scene: Optional[QCItemScore] = None
    video: Optional[QCItemScore] = None
    audio: Optional[QCItemScore] = None
    decision: QCDecision
    retry_reasons: List[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """decision 是否为 APPROVED。"""
        return self.decision == QCDecision.APPROVED

    def to_json(self, path: Optional[PathLike] = None) -> str:
        """序列化为缩进 JSON 字符串；给了 path 就同时写文件。``ensure_ascii=False`` 保证中文原因可读。"""
        payload = self.model_dump(mode="json")
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


def _character_item(result, threshold: float) -> QCItemScore:
    """把 CharacterConsistencyResult 按阈值转成 QCItemScore。"""
    passed = result.similarity >= threshold
    return QCItemScore(
        name="character_consistency",
        score=result.similarity,
        threshold=threshold,
        passed=passed,
        detail=None if passed else (f"character similarity {result.similarity:.3f} below threshold {threshold:.3f}"),
    )


def _scene_item(result, threshold: float) -> QCItemScore:
    """把 SceneAlignmentResult 按阈值转成 QCItemScore。"""
    passed = result.similarity >= threshold
    return QCItemScore(
        name="scene_alignment",
        score=result.similarity,
        threshold=threshold,
        passed=passed,
        detail=None if passed else (f"scene alignment {result.similarity:.3f} below threshold {threshold:.3f}"),
    )


def _video_item(result) -> QCItemScore:
    """把 VideoQCResult 转成 QCItemScore：通过与否沿用 result.passed，detail 列出所有未通过的子项。

    ``score`` 只是给报表看的综合分：1 - max(黑帧比例, 静帧比例 * 0.5)——静帧比黑帧"轻"，所以打五折。
    """
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
    """把 AudioQCResult 转成 QCItemScore；``score`` 直接放响度 dBFS 方便报表排查。"""
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
    """纯聚合步骤：Generate -> QC -> PASS? -> Retry/Approved。

    接收四个 QC 检查器已经算好的结果（不适用于该素材的维度可以传 ``None``），
    产出一份带通过 / 失败决定的 QCReport；失败时附带重试原因列表。
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
        # 视频结果缺失且被要求：不能"没查就算过"，记一条原因让它 RETRY
        retry_reasons.append("video QC result missing but required")

    if audio_result is not None:
        item = _audio_item(audio_result)
        items["audio"] = item
        if not item.passed:
            retry_reasons.append(item.detail or "audio_quality failed")
    elif thresholds.audio_required:
        retry_reasons.append("audio QC result missing but required")

    # 任何一条原因都足以退回：QC 是"一票否决"制
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
    """端到端编排：只运行那些给了输入的检查器，然后通过 build_qc_report 聚合成 QCReport。

    重依赖（CLIP 用的 transformers/torch、opencv、pydub）在这里按需延迟 import，
    这样单独 import 本模块仍然很轻量。
    """
    character_result = None
    scene_result = None
    video_result = None
    audio_result = None

    if reference_character_image is not None and generated_image is not None:
        from ..character.consistency_checker import CharacterConsistencyChecker

        checker = CharacterConsistencyChecker()
        character_result = checker.score(reference_character_image, generated_image, character_id=character_id)

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
