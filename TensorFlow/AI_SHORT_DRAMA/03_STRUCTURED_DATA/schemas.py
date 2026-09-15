"""结构化数据 schema 定义（pydantic 模型 + 枚举）。

这个模块是整个 AI 竖屏短剧生产流水线的"数据契约层"：02_STORY_ENGINE（故事/剧本生成）、
03 自身（结构化数据校验）、05_TRAINING（训练数据构建）、07_GENERATION（图像/视频/音频生成）、
08_QC（质检）、13_INFRA 等模块都通过这里定义的模型来读写 JSON 文件、校验 LLM 输出、
在各阶段之间传递数据。

设计上有两个关键点：

1. 几乎所有模型都用 ``model_config = ConfigDict(extra="forbid")``：LLM 生成的结构化输出
   经常会"自由发挥"多塞几个字段，如果不禁止多余字段，脏数据会静默混进流水线，等到下游
   （比如生成视频提示词时）才报错，定位成本很高。所以这里选择"宁可校验失败在最早的
   阶段暴露问题"，而不是宽松接受。
2. ``Shot``（分镜镜头）是全系统唯一的关联键：episode/scene/shot 三级编号 + scene_id
   把剧本、分镜、图像 prompt、视频 prompt、QC 报告全部串联起来，下游各个生成/质检
   模块都以 shot_id 做关联查找，所以 Shot 的字段设计尽量保持稳定，不轻易变动。

文件末尾还有一段 sys.modules 别名处理，用来解决同一个模块在不同代码路径下被
以不同名字 import 导致出现"两份类定义"从而让 pydantic 类型校验失败的问题，
详见该处注释。
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Genre(str, Enum):
    """短剧题材/类型标签，用于故事圣经（StoryBible）标注一部剧属于哪些题材。"""

    REVENGE = "revenge"
    REBIRTH = "rebirth"
    ROMANCE = "romance"
    CEO_DRAMA = "ceo_drama"
    FAMILY_DRAMA = "family_drama"
    FANTASY = "fantasy"
    SUSPENSE = "suspense"


class CharacterRole(str, Enum):
    """角色在故事中的功能定位（主角/二番/反派/配角/龙套）。"""

    PROTAGONIST = "protagonist"
    DEUTERAGONIST = "deuteragonist"
    ANTAGONIST = "antagonist"
    SUPPORTING = "supporting"
    MINOR = "minor"


class Gender(str, Enum):
    """角色性别标签。"""

    FEMALE = "female"
    MALE = "male"
    OTHER = "other"


class RelationType(str, Enum):
    """两个角色之间的关系类型（家人/恋人/对手/盟友/敌人/雇佣关系/朋友）。"""

    FAMILY = "family"
    ROMANTIC = "romantic"
    RIVAL = "rival"
    ALLY = "ally"
    ENEMY = "enemy"
    EMPLOYER = "employer"
    EMPLOYEE = "employee"
    FRIEND = "friend"


class Emotion(str, Enum):
    """情绪标签，贯穿台词（DialogueLine）、分镜（Shot）等需要标注表演/情绪的地方。"""

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
    """场景发生的时间段，用于场景（Scene）的时间标注，也影响后续图像/视频的光线风格。"""

    DAWN = "dawn"
    MORNING = "morning"
    NOON = "noon"
    AFTERNOON = "afternoon"
    DUSK = "dusk"
    NIGHT = "night"
    LATE_NIGHT = "late_night"


class IntExt(str, Enum):
    """内景/外景/内外景标记（剧本行业惯用的 INT./EXT. 标注）。"""

    INT = "int"
    EXT = "ext"
    INT_EXT = "int_ext"


class ShotSize(str, Enum):
    """景别（镜头取景范围），决定分镜画面里人物/环境的相对比例。"""

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
    """机位角度（平视/仰拍/俯拍/荷兰角/鸟瞰/正上方）。"""

    EYE_LEVEL = "eye_level"
    LOW_ANGLE = "low_angle"
    HIGH_ANGLE = "high_angle"
    DUTCH_ANGLE = "dutch_angle"
    BIRDS_EYE = "birds_eye"
    OVERHEAD = "overhead"


class CameraMovement(str, Enum):
    """运镜方式，供 CameraSpec 和 VideoPrompt 描述镜头如何运动。"""

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
    """剧本世界观中的一个具体场地（比如"总裁办公室""老宅祖屋"）。"""

    # extra="forbid"：LLM 生成结构化 JSON 时偶尔会多塞字段（比如多写一个"address"），
    # 禁止多余字段能让这类脏数据在校验阶段就报错，而不是被悄悄吞掉、下游才出问题。
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    mood: str = ""


class WorldSetting(BaseModel):
    """故事的世界观设定：时代背景、场地列表、组织机构、社会规则等，供全剧共用。"""

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
    """某个角色与另一个角色之间的单向关系描述，挂在 Character.relationships 下。"""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    relation_type: RelationType
    description: str


class Character(BaseModel):
    """完整的角色卡：外貌、性格、背景、动机、人物关系等，供编剧和图像生成环节共同参考。"""

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
    """剧情节拍：一幕（act）内的一个关键情节点，可关联到具体集数。"""

    model_config = ConfigDict(extra="forbid")

    act: str
    description: str
    episode_ids: list[str] = Field(default_factory=list)


class SeasonArc(BaseModel):
    """一季的整体故事弧：核心问题、结局走向，以及由若干 StoryBeat 组成的节拍序列。"""

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
    """全剧的"故事圣经"：题材、主题、基调、核心冲突、卖点等顶层创作设定，是整条流水线的起点。"""

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
    """单集剧本的元信息。"""

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
    """一句台词，归属于某个分镜（shot_id）和角色，包含中/英文文本与情绪标注。"""

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
    """一场戏：发生在某个地点、某个时间段，包含在场角色和该场戏内的台词列表。"""

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
    """一个镜头的摄影参数：景别、机位角度、运镜方式，供分镜和视频生成环节使用。"""

    model_config = ConfigDict(extra="forbid")

    shot_size: ShotSize
    angle: CameraAngle = CameraAngle.EYE_LEVEL
    movement: CameraMovement = CameraMovement.STATIC
    lens_notes: str = ""


class Shot(BaseModel):
    """分镜镜头：全系统唯一的关联键。

    episode/scene/shot 三级编号加上 scene_id，把剧本、图像 prompt、视频 prompt、
    QC 报告等所有下游产物串联在一起——各生成/质检模块都是以 shot 的 id 做关联查找，
    因此这个模型的字段一旦确定就应尽量保持稳定，避免牵连整条流水线。
    """

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
    """剧本文本文件：某一集的完整台本内容（区别于结构化的 Scene/Shot 数据）。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    episode_id: str
    title: str
    language: str = "zh"
    content: str
    scene_ids: list[str] = Field(default_factory=list)
    version: int = 1


class ImagePrompt(BaseModel):
    """给某个分镜生成关键帧图像所用的提示词（正向/反向 prompt、风格标签、参考角色等）。"""

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
    """给某个分镜生成视频所用的提示词，可关联对应的 ImagePrompt 作为首帧参考。"""

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
    """QC 问题的严重程度，决定是否需要打回重写（BLOCKER 级别必然导致 QCVerdict.passed=False）。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKER = "blocker"


class QCIssue(BaseModel):
    """质检官发现的单条问题：问题代码、严重程度、出问题的位置、描述与修改建议。"""

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
    """总编审的三种决定类型：approve 通过、revise 需修改后重审、reject 直接拒绝。"""

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
    """对应磁盘上"story"文件的顶层结构：故事圣经 + 各季故事弧，用于整体读写与校验。"""

    model_config = ConfigDict(extra="forbid")

    story_bible: StoryBible
    season_arcs: list[SeasonArc] = Field(default_factory=list)


class CharactersFile(BaseModel):
    """对应磁盘上"characters"文件的顶层结构：全部角色列表。"""

    model_config = ConfigDict(extra="forbid")

    characters: list[Character]


class ScenesFile(BaseModel):
    """对应磁盘上"scenes"文件的顶层结构：全部场景列表。"""

    model_config = ConfigDict(extra="forbid")

    scenes: list[Scene]


class EpisodesFile(BaseModel):
    """对应磁盘上"episodes"文件的顶层结构：全部分集列表。"""

    model_config = ConfigDict(extra="forbid")

    episodes: list[Episode]


class ShotsFile(BaseModel):
    """对应磁盘上"shots"文件的顶层结构：全部分镜列表。"""

    model_config = ConfigDict(extra="forbid")

    shots: list[Shot]


class PromptsFile(BaseModel):
    """对应磁盘上"prompts"文件的顶层结构：全部图像 prompt 与视频 prompt 列表。"""

    model_config = ConfigDict(extra="forbid")

    image_prompts: list[ImagePrompt]
    video_prompts: list[VideoPrompt]


# 同一个模块会以两种名字被 import：02/05 内部走 sys.path 的 `import schemas`，13_INFRA 走
# importlib 的 "03_STRUCTURED_DATA.schemas"。不做别名就会出现两份 Shot 类，
# ShotsFile(shots=[另一份的 Shot]) 会被 pydantic 判为类型不匹配。
import sys as _sys

for _name in ("schemas", "03_STRUCTURED_DATA.schemas"):
    _sys.modules.setdefault(_name, _sys.modules[__name__])
