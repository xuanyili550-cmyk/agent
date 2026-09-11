"""
Generates solid-color placeholder PNGs with a text label using PIL only.
No real image assets, no network access. Used so downstream training/generation
code has something to load during offline smoke tests.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = [
    (211, 84, 0),
    (41, 128, 185),
    (39, 174, 96),
    (142, 68, 173),
    (192, 57, 43),
    (22, 160, 133),
]


def make_placeholder(path: Path, label: str, size: tuple[int, int] = (512, 512), color_idx: int = 0) -> Path:
    color = COLORS[color_idx % len(COLORS)]
    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    draw.multiline_text((16, 16), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def generate_set(dataset_root: Path, ids: list[str], subdir_name: str, label_prefix: str) -> list[Path]:
    paths = []
    for i, item_id in enumerate(ids):
        out = dataset_root / subdir_name / item_id / "ref_001.png"
        make_placeholder(out, f"{label_prefix}\n{item_id}", color_idx=i)
        paths.append(out)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate placeholder PNGs for dataset smoke tests")
    parser.add_argument("--dataset-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.dataset_root)

    characters = ["char_lin_wei", "char_su_yan", "char_zhao_ming"]
    environments = ["env_office_day", "env_alley_night"]
    styles = ["style_cinematic_moody", "style_bright_commercial"]

    written = []
    written += generate_set(root, characters, "characters", "CHARACTER")
    written += generate_set(root, environments, "environments", "ENVIRONMENT")
    written += generate_set(root, styles, "styles", "STYLE")

    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
