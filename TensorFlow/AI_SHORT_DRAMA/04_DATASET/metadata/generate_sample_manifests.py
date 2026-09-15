"""
示例清单生成器：向 04_DATASET/metadata 下的各个 JSONL 清单写入合成样本行。

在流水线中的位置：数据准备阶段的一键初始化脚本。运行后会：
1. 调用 ``generate_placeholder_images.generate_set`` 生成角色 / 环境 / 风格占位图；
2. 用 ``schemas.py`` 里的 Pydantic 模型构造样本行并序列化成 JSONL。

设计说明：
- 角色 / 场景 / 风格行指向真实存在的占位 PNG，所以 ``dataset_loaders`` 能真的解码图片。
- 语音 / 视频行的文件路径是纯虚构、并不存在的：本项目不生成真实音频 / 视频，这些行只用来跑通加载器的路径与字段逻辑。
- 所有行的 ``source`` / ``license`` 都标记为 synthetic-placeholder，明确它们不是可商用素材。
"""

from __future__ import annotations

import sys
from pathlib import Path

METADATA_DIR = Path(__file__).resolve().parent
DATASET_ROOT = METADATA_DIR.parent

# 目录名带数字前缀（04_DATASET）不是合法包名，无法用常规相对导入，只能把兄弟目录塞进 sys.path
sys.path.insert(0, str(DATASET_ROOT / "images"))
sys.path.insert(0, str(METADATA_DIR))

from generate_placeholder_images import generate_set  # noqa: E402

from schemas import (  # noqa: E402
    CharacterManifestEntry,
    SceneManifestEntry,
    StyleManifestEntry,
    VideoManifestEntry,
    VoiceManifestEntry,
)

# 三组素材 id，必须与 generate_placeholder_images.main 里的列表一致
CHARACTERS = ["char_lin_wei", "char_su_yan", "char_zhao_ming"]
ENVIRONMENTS = ["env_office_day", "env_alley_night"]
STYLES = ["style_cinematic_moody", "style_bright_commercial"]


def write_jsonl(path: Path, rows) -> None:
    """把一组 Pydantic 行对象逐行序列化为 JSON 写入 ``path``（每行一个对象，即 JSONL 格式）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")


def build_character_rows() -> list[CharacterManifestEntry]:
    """构造三位示例角色的清单行；``image_path`` 是相对数据集根目录的路径，与占位图目录结构对应。"""
    captions = {
        "char_lin_wei": "a young woman with short black hair wearing a grey office blazer, neutral studio lighting",
        "char_su_yan": "a man in his thirties with glasses and a dark green sweater, three-quarter view",
        "char_zhao_ming": "an elderly man with grey hair and a traditional tang suit, warm lighting",
    }
    return [
        CharacterManifestEntry(
            character_id=cid,
            image_path=f"characters/{cid}/ref_001.png",
            caption=captions[cid],
            pose="front-facing portrait",
            expression="neutral",
            source="synthetic-placeholder",
            license="CC0-synthetic-placeholder",
        )
        for cid in CHARACTERS
    ]


def build_scene_rows() -> list[SceneManifestEntry]:
    """构造两个示例环境（白天办公室 / 雨夜小巷）的清单行，附带时段与天气标签。"""
    captions = {
        "env_office_day": "a modern open-plan office with floor-to-ceiling windows, daytime, soft natural light",
        "env_alley_night": "a narrow rain-soaked alley at night, neon signage reflections, moody lighting",
    }
    # (time_of_day, weather)
    meta = {
        "env_office_day": ("day", "clear"),
        "env_alley_night": ("night", "rain"),
    }
    return [
        SceneManifestEntry(
            environment_id=eid,
            image_path=f"environments/{eid}/ref_001.png",
            caption=captions[eid],
            time_of_day=meta[eid][0],
            weather=meta[eid][1],
            source="synthetic-placeholder",
            license="CC0-synthetic-placeholder",
        )
        for eid in ENVIRONMENTS
    ]


def build_style_rows() -> list[StyleManifestEntry]:
    """构造两种示例视觉风格（电影感暗调 / 明亮商业感）的清单行。"""
    captions = {
        "style_cinematic_moody": "cinematic moody grade, high contrast, teal and orange color palette",
        "style_bright_commercial": "bright clean commercial look, high-key lighting, saturated colors",
    }
    tags = {
        "style_cinematic_moody": ["cinematic", "moody", "high-contrast"],
        "style_bright_commercial": ["commercial", "bright", "high-key"],
    }
    return [
        StyleManifestEntry(
            style_id=sid,
            image_path=f"styles/{sid}/ref_001.png",
            caption=captions[sid],
            style_tags=tags[sid],
            source="synthetic-placeholder",
            license="CC0-synthetic-placeholder",
        )
        for sid in STYLES
    ]


def build_voice_rows() -> list[VoiceManifestEntry]:
    """构造两条示例语音行。``audio_path`` 指向并不存在的文件——只为验证清单字段与加载器，不做音频解码。"""
    return [
        VoiceManifestEntry(
            voice_id="voice_lin_wei_001",
            character_id="char_lin_wei",
            audio_path="voices/char_lin_wei/line_001.wav",
            transcript="今晚这场雨,好像一直下不完。",
            speaker="char_lin_wei",
            duration_sec=3.2,
            sample_rate=24000,
            source="synthetic-placeholder",
            license="CC0-synthetic-placeholder",
        ),
        VoiceManifestEntry(
            voice_id="voice_su_yan_001",
            character_id="char_su_yan",
            audio_path="voices/char_su_yan/line_001.wav",
            transcript="这件事,从头到尾都不是巧合。",
            speaker="char_su_yan",
            duration_sec=2.8,
            sample_rate=24000,
            source="synthetic-placeholder",
            license="CC0-synthetic-placeholder",
        ),
    ]


def build_video_rows() -> list[VideoManifestEntry]:
    """构造两条示例视频片段行。``video_path`` 同样是虚构路径，仅用于跑通加载器。"""
    return [
        VideoManifestEntry(
            clip_id="clip_alley_chase_001",
            video_path="videos/env_alley_night/clip_001.mp4",
            caption="a man running through a rain-soaked alley at night, tracking shot",
            character_id="char_zhao_ming",
            duration_sec=4.5,
            fps=24,
            source="synthetic-placeholder",
            license="CC0-synthetic-placeholder",
        ),
        VideoManifestEntry(
            clip_id="clip_office_dialogue_001",
            video_path="videos/env_office_day/clip_001.mp4",
            caption="two people talking across a desk in a bright office, static medium shot",
            character_id="char_lin_wei",
            duration_sec=6.0,
            fps=24,
            source="synthetic-placeholder",
            license="CC0-synthetic-placeholder",
        ),
    ]


def main() -> None:
    """脚本入口：先生成占位图（保证图像行指向的文件存在），再写出五个清单文件。"""
    generate_set(DATASET_ROOT, CHARACTERS, "characters", "CHARACTER")
    generate_set(DATASET_ROOT, ENVIRONMENTS, "environments", "ENVIRONMENT")
    generate_set(DATASET_ROOT, STYLES, "styles", "STYLE")

    write_jsonl(METADATA_DIR / "character_manifest.jsonl", build_character_rows())
    write_jsonl(METADATA_DIR / "scene_manifest.jsonl", build_scene_rows())
    write_jsonl(METADATA_DIR / "style_manifest.jsonl", build_style_rows())
    write_jsonl(METADATA_DIR / "voice_manifest.jsonl", build_voice_rows())
    write_jsonl(METADATA_DIR / "video_manifest.jsonl", build_video_rows())
    print("sample manifests written to", METADATA_DIR)


if __name__ == "__main__":
    main()
