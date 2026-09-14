"""
End-to-end offline smoke test: read Shot JSON -> build an image prompt ->
produce a placeholder image asset -> write an asset record.

No real model weights or API keys are used -- DummyImageGenerator (see
07_GENERATION/image/image_generator.py) draws a labelled placeholder PNG
instead of running a diffusion pipeline. Run directly:

  python demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

GENERATION_ROOT = Path(__file__).resolve().parent
STRUCTURED_DATA_ROOT = GENERATION_ROOT.parent / "03_STRUCTURED_DATA"

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

# Stand-in for what the Story Engine (02_STORY_ENGINE) would eventually write
# to 03_STRUCTURED_DATA/shots.json + prompts.json. Purely synthetic, no real
# story content. Uses the canonical Shot/ImagePrompt schemas from
# 03_STRUCTURED_DATA/schemas.py -- Shot itself carries no prompt/style text,
# that lives on the separate ImagePrompt record keyed by shot_id.
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
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    generator = DummyImageGenerator()

    print(f"loaded {len(SAMPLE_SHOTS)} synthetic shots")

    for shot, image_prompt in SAMPLE_SHOTS:
        prompt = image_prompt.prompt_text
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
