from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Genre(str, Enum):
    REVENGE = "revenge"
    REBIRTH = "rebirth"
    ROMANCE = "romance"
    CEO_DRAMA = "ceo_drama"
    FAMILY_DRAMA = "family_drama"
    FANTASY = "fantasy"
    SUSPENSE = "suspense"


class CharacterRole(str, Enum):
    PROTAGONIST = "protagonist"
    DEUTERAGONIST = "deuteragonist"
    ANTAGONIST = "antagonist"
    SUPPORTING = "supporting"
    MINOR = "minor"


class Gender(str, Enum):
    FEMALE = "female"
    MALE = "male"
    OTHER = "other"


class RelationType(str, Enum):
    FAMILY = "family"
    ROMANTIC = "romantic"
    RIVAL = "rival"
    ALLY = "ally"
    ENEMY = "enemy"
    EMPLOYER = "employer"
    EMPLOYEE = "employee"
    FRIEND = "friend"


class Emotion(str, Enum):
    ANGER = "anger"
    SADNESS = "sadness"
    JOY = "joy"
    FEAR = "fear"
    SURPRISE = "surprise"
    CONTEMPT = "contempt"
    DESPAIR = "despair"
    DETERMINATION = "determination"
    CALM = "calm"
    ANXIETY = "anxiety"
    VENGEFUL = "vengeful"
    TRIUMPH = "triumph"
    LOVE = "love"
    SHOCK = "shock"


class TimeOfDay(str, Enum):
    DAWN = "dawn"
    MORNING = "morning"
    NOON = "noon"
    AFTERNOON = "afternoon"
    DUSK = "dusk"
    NIGHT = "night"
    LATE_NIGHT = "late_night"


class IntExt(str, Enum):
    INT = "int"
    EXT = "ext"
    INT_EXT = "int_ext"


class ShotSize(str, Enum):
    EXTREME_CLOSE_UP = "extreme_close_up"
    CLOSE_UP = "close_up"
    MEDIUM_CLOSE_UP = "medium_close_up"
    MEDIUM_SHOT = "medium_shot"
    MEDIUM_LONG_SHOT = "medium_long_shot"
    LONG_SHOT = "long_shot"
    EXTREME_LONG_SHOT = "extreme_long_shot"
    TWO_SHOT = "two_shot"
    OVER_THE_SHOULDER = "over_the_shoulder"
    POV = "pov"
    INSERT = "insert"


class CameraAngle(str, Enum):
    EYE_LEVEL = "eye_level"
    LOW_ANGLE = "low_angle"
    HIGH_ANGLE = "high_angle"
    DUTCH_ANGLE = "dutch_angle"
    BIRDS_EYE = "birds_eye"
    OVERHEAD = "overhead"


class CameraMovement(str, Enum):
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    TRACKING = "tracking"
    HANDHELD = "handheld"
    CRANE = "crane"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    WHIP_PAN = "whip_pan"


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    mood: str = ""


class WorldSetting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    time_period: str
    locations: list[Location] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    social_rules: list[str] = Field(default_factory=list)
    tone: str = ""


class Relationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_id: str
    relation_type: RelationType
    description: str


class Character(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    role: CharacterRole
    gender: Gender
    age: int
    occupation: str
    appearance: str
    personality_traits: list[str] = Field(default_factory=list)
    backstory: str
    motivation: str
    relationships: list[Relationship] = Field(default_factory=list)
    arc_summary: str = ""
    voice_style: str = ""


class StoryBeat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    act: str
    description: str
    episode_ids: list[str] = Field(default_factory=list)


class SeasonArc(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    season_number: int
    title: str
    synopsis: str
    central_question: str
    resolution: str
    beats: list[StoryBeat] = Field(default_factory=list)
    episode_ids: list[str] = Field(default_factory=list)


class StoryBible(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    logline: str
    genre: list[Genre]
    themes: list[str]
    tone: str
    target_audience: str
    main_conflict: str
    unique_selling_point: str
    episode_count_planned: int
    world: WorldSetting
    character_ids: list[str] = Field(default_factory=list)


class Episode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    episode_number: int
    season_id: str
    title: str
    synopsis: str
    # 短剧强约束：开场必须在极短时间内建立强钩子，结尾必须留悬念以驱动下一集点击
    hook: str = Field(..., min_length=1)
    cliffhanger: str = Field(..., min_length=1)
    duration_seconds: int
    scene_ids: list[str] = Field(default_factory=list)
    characters_featured: list[str] = Field(default_factory=list)
    emotional_arc: str = ""


class DialogueLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    shot_id: str
    character_id: str
    character_name: str
    order: int
    line_zh: str
    line_en: str = ""
    emotion: Emotion
    voice_direction: str = ""
    subtitle_style: str = ""


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    episode_id: str
    scene_number: int
    location_id: str
    time_of_day: TimeOfDay
    int_ext: IntExt
    description: str
    characters_present: list[str] = Field(default_factory=list)
    emotional_tone: str = ""
    shot_ids: list[str] = Field(default_factory=list)
    dialogue: list[DialogueLine] = Field(default_factory=list)


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_size: ShotSize
    angle: CameraAngle = CameraAngle.EYE_LEVEL
    movement: CameraMovement = CameraMovement.STATIC
    lens_notes: str = ""


class Shot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    episode: int
    scene: int
    shot: int
    scene_id: str
    character: list[str]
    location: str
    action: str
    emotion: Emotion
    camera: CameraSpec
    duration: float
    dialogue_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class Script(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    episode_id: str
    title: str
    language: str = "zh"
    content: str
    scene_ids: list[str] = Field(default_factory=list)
    version: int = 1


class ImagePrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    shot_id: str
    prompt_text: str
    negative_prompt: str = ""
    style_tags: list[str] = Field(default_factory=list)
    aspect_ratio: str = "9:16"
    reference_character_ids: list[str] = Field(default_factory=list)
    seed: Optional[int] = None


class VideoPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    shot_id: str
    image_prompt_id: Optional[str] = None
    prompt_text: str
    motion_description: str
    camera_movement: CameraMovement
    duration: float
    fps: int = 24
    model_target: str = "generic"
    negative_prompt: str = ""


# ---- 编审链路：质检官（QC Officer）与总编审（Chief Editor）的判定 ----------------------


class IssueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKER = "blocker"


class QCIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str  # 例如 hook_too_slow / character_inconsistent / enum_mismatch / dialogue_off_character
    severity: IssueSeverity
    location: str = ""  # 出问题的 episode/scene/shot/dialogue id
    description: str
    suggestion: str = ""


class QCVerdict(BaseModel):
    """质检官对一集剧本+分镜的检查结果。passed=False 时流水线把 issues 回灌给编剧/分镜重写。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    episode_id: str
    passed: bool
    score: int = Field(ge=0, le=100)
    issues: list[QCIssue] = Field(default_factory=list)
    summary: str = ""


class EditorialDecisionType(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


class EditorialDecision(BaseModel):
    """总编审的最终决定：approve 进入生产；revise 带修改意见打回；reject 终止并等待人工。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    episode_id: str
    decision: EditorialDecisionType
    notes: str
    required_changes: list[str] = Field(default_factory=list)
    commercial_score: int = Field(default=0, ge=0, le=100)  # 付费转化/完播潜力的主观评分
    audience_fit: str = ""


class StoryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    story_bible: StoryBible
    season_arcs: list[SeasonArc] = Field(default_factory=list)


class CharactersFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    characters: list[Character]


class ScenesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenes: list[Scene]


class EpisodesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episodes: list[Episode]


class ShotsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shots: list[Shot]


class PromptsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_prompts: list[ImagePrompt]
    video_prompts: list[VideoPrompt]


# 同一个模块会以两种名字被 import：02/05 内部走 sys.path 的 `import schemas`，13_INFRA 走
# importlib 的 "03_STRUCTURED_DATA.schemas"。不做别名就会出现两份 Shot 类，
# ShotsFile(shots=[另一份的 Shot]) 会被 pydantic 判为类型不匹配。
import sys as _sys

for _name in ("schemas", "03_STRUCTURED_DATA.schemas"):
    _sys.modules.setdefault(_name, _sys.modules[__name__])
