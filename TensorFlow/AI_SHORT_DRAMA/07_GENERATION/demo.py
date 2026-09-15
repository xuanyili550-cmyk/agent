"""端到端离线冒烟测试：读 Shot JSON -> 拼图像 prompt -> 生成占位图素材 -> 写素材记录。

不使用任何真实模型权重或 API key——DummyImageGenerator（见 07_GENERATION/image/image_generator.py）
画一张带文字标注的占位 PNG，替代真正的扩散模型推理。直接运行：

  python demo.py

流水线位置：模拟"03_STRUCTURED_DATA 的结构化镜头 -> 07_GENERATION 生成 -> asset_registry 登记"
这一小段，用来在没有 GPU / 网络的机器上验证数据结构、文件布局和素材台账能否跑通。
"""

from __future__ import annotations

import sys
from pathlib import Path

GENERATION_ROOT = Path(__file__).resolve().parent
STRUCTURED_DATA_ROOT = GENERATION_ROOT.parent / "03_STRUCTURED_DATA"

# 目录名以数字开头不是合法包名，只能通过 sys.path 让下面的裸模块名 import 生效
sys.path.insert(0, str(GENERATION_ROOT))
sys.path.insert(0, str(GENERATION_ROOT / "image"))
sys.path.insert(0, str(STRUCTURED_DATA_ROOT))

from asset_registry import AssetRecord, new_asset_id, now_iso, write_asset_record  # noqa: E402
from image_generator import DummyImageGenerator  # noqa: E402

from schemas import (  # noqa: E402
    CameraAngle,
    CameraMovement,
    CameraSpec,
    Emotion,
    ImagePrompt,
    Shot,
    ShotSize,
)

OUTPUT_DIR = GENERATION_ROOT / "demo_outputs"
IMAGES_DIR = OUTPUT_DIR / "images"
ASSET_LOG_PATH = OUTPUT_DIR / "asset_records.jsonl"

# 顶替故事引擎（02_STORY_ENGINE）将来写入 03_STRUCTURED_DATA/shots.json + prompts.json 的内容。
# 纯合成数据，没有真实剧情。使用 03_STRUCTURED_DATA/schemas.py 里的标准 Shot/ImagePrompt 模型——
# Shot 本身不带 prompt/风格文本，那些放在按 shot_id 关联的独立 ImagePrompt 记录上。
SAMPLE_SHOTS: list[tuple[Shot, ImagePrompt]] = [
    (
        Shot(
            id="demo_shot_ep001_sc01_sh001",
            episode=1,
            scene=1,
            shot=1,
            scene_id="demo_scene_ep001_sc01",
            character=["char_lin_wei"],
            location="env_office_day",
            action="she stands by a window in a bright office and exhales slowly",
            emotion=Emotion.CALM,
            camera=CameraSpec(
                shot_size=ShotSize.MEDIUM_SHOT,
                angle=CameraAngle.EYE_LEVEL,
                movement=CameraMovement.STATIC,
            ),
            duration=4.0,
        ),
        ImagePrompt(
            id="demo_img_ep001_sc01_sh001",
            shot_id="demo_shot_ep001_sc01_sh001",
            prompt_text="medium shot of a young woman in a grey blazer standing by a window in a bright office, cinematic moody grade",
            style_tags=["style_cinematic_moody"],
            reference_character_ids=["char_lin_wei"],
        ),
    ),
    (
        Shot(
            id="demo_shot_ep001_sc02_sh001",
            episode=1,
            scene=2,
            shot=1,
            scene_id="demo_scene_ep001_sc02",
            character=["char_zhao_ming"],
            location="env_alley_night",
            action="he stops and looks back over his shoulder in a rain-soaked alley",
            emotion=Emotion.FEAR,
            camera=CameraSpec(
                shot_size=ShotSize.LONG_SHOT,
                angle=CameraAngle.LOW_ANGLE,
                movement=CameraMovement.STATIC,
            ),
            duration=5.0,
        ),
        ImagePrompt(
            id="demo_img_ep001_sc02_sh001",
            shot_id="demo_shot_ep001_sc02_sh001",
            prompt_text="wide shot of an elderly man in a tang suit walking through a rain-soaked neon-lit alley at night",
            style_tags=["style_cinematic_moody"],
            reference_character_ids=["char_zhao_ming"],
        ),
    ),
]


def main() -> None:
    """遍历合成镜头：为每个镜头生成一张占位图、落盘并写一条素材记录。"""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    generator = DummyImageGenerator()

    print(f"loaded {len(SAMPLE_SHOTS)} synthetic shots")

    for shot, image_prompt in SAMPLE_SHOTS:
        prompt = image_prompt.prompt_text
        # 由镜头 ID 派生 seed：同一镜头每次 demo 得到的占位图颜色稳定，便于肉眼对比
        seed = abs(hash(shot.id)) % (2**31)

        image = generator.generate(prompt=prompt, seed=seed, width=512, height=512)
        file_path = IMAGES_DIR / f"{shot.id}.png"
        image.save(file_path)

        record = AssetRecord(
            asset_id=new_asset_id("img"),
            file_path=str(file_path),
            character_id=shot.character[0] if shot.character else None,
            episode_id=f"EP{shot.episode:03d}",
            shot_id=shot.id,
            model="dummy-placeholder-generator",
            prompt=prompt,
            seed=seed,
            license="CC0-synthetic-placeholder",
            created_at=now_iso(),
        )
        write_asset_record(record, ASSET_LOG_PATH)

        print(f"shot {shot.id}: wrote {file_path} and asset record {record.asset_id}")

    print(f"\ndone. asset log at {ASSET_LOG_PATH}")


if __name__ == "__main__":
    main()
