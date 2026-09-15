"""
占位图生成器：只用 PIL 生成带文字标签的纯色 PNG。

在流水线中的位置：属于 04_DATASET 的数据准备工具，在 ``generate_sample_manifests.py`` 写清单之前先运行，
为角色 / 环境 / 风格三类参考图各生成一张 ``ref_001.png``。

为什么需要它：仓库不附带任何真实图片素材，也不允许联网下载；但下游的 LoRA 训练脚本和数据集加载器在离线冒烟测试时
必须能真的打开一张图片。纯色 + 文字的占位图体积极小、生成可复现，足够跑通整条加载链路。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

# 一组区分度高的背景色，按索引轮换，方便肉眼区分不同素材
COLORS = [
    (211, 84, 0),
    (41, 128, 185),
    (39, 174, 96),
    (142, 68, 173),
    (192, 57, 43),
    (22, 160, 133),
]


def make_placeholder(path: Path, label: str, size: tuple[int, int] = (512, 512), color_idx: int = 0) -> Path:
    """生成一张纯色背景、左上角写有 ``label`` 的 PNG 并保存到 ``path``，返回该路径。

    ``color_idx`` 对 ``COLORS`` 长度取模，所以传任意整数都安全；默认 512x512 与 SDXL 训练脚本的分辨率参数一致。
    """
    color = COLORS[color_idx % len(COLORS)]
    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    draw.multiline_text((16, 16), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def generate_set(dataset_root: Path, ids: list[str], subdir_name: str, label_prefix: str) -> list[Path]:
    """为一组素材 id 各生成一张占位图，目录结构为 ``<dataset_root>/<subdir_name>/<id>/ref_001.png``。

    这个目录约定与清单里的 ``image_path`` 相对路径一一对应，改动时两边要同步。
    """
    paths = []
    for i, item_id in enumerate(ids):
        out = dataset_root / subdir_name / item_id / "ref_001.png"
        make_placeholder(out, f"{label_prefix}\n{item_id}", color_idx=i)
        paths.append(out)
    return paths


def main() -> None:
    """命令行入口：在数据集根目录下生成角色 / 环境 / 风格三组占位图并打印写入路径。"""
    parser = argparse.ArgumentParser(description="Generate placeholder PNGs for dataset smoke tests")
    # 默认根目录 = 04_DATASET（本文件的上两级）
    parser.add_argument("--dataset-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.dataset_root)

    # 这三组 id 必须与 generate_sample_manifests.py 里的常量保持一致
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
