"""
数据集加载器：把 04_DATASET/metadata/*.jsonl 清单文件加载成 HuggingFace ``datasets.Dataset`` 对象。

在流水线中的位置：位于"数据准备"层，是 05_TRAINING 下各训练脚本（角色 LoRA / 风格 LoRA / SFT）读取训练样本的统一入口。
训练脚本不直接解析 JSONL，而是调用这里的 ``build_*_dataset`` 函数，这样清单格式一旦调整只需改这一处。

设计要点：
- 图像类清单（character / scene / style）会把 ``image_path`` 相对数据集根目录解析成绝对路径，并转成 ``datasets.Image`` 特征，
  这样训练脚本拿到的就是可直接解码的 PIL 图片，而不是字符串路径。
- 语音 / 视频清单的路径列保持为普通字符串：本仓库不附带真实音频 / 视频文件，强行转成 Audio/Video 特征会在解码时报错。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset
from datasets import Image as HFImage

# 数据集根目录 = 本文件所在目录（04_DATASET），清单里的相对路径都以它为基准
DATASET_ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA_DIR = DATASET_ROOT / "metadata"


def _load_jsonl(manifest_path: str | Path) -> Dataset:
    """用 ``datasets`` 的 json 加载器读取一个 JSONL 清单，固定取 ``train`` 切分（json 加载器只产出这一个切分）。"""
    return load_dataset("json", data_files=str(manifest_path), split="train")


def _resolve_paths(ds: Dataset, column: str, dataset_root: Path) -> Dataset:
    """把数据集中 ``column`` 列的相对路径拼上 ``dataset_root`` 变成绝对路径。

    为什么要这样做：清单里存相对路径便于仓库迁移；但 ``datasets.Image`` 解码时需要能直接打开的完整路径。
    """

    def _resolve(example):
        """对单条样本做路径拼接（供 ``Dataset.map`` 逐行调用）。"""
        example[column] = str(dataset_root / example[column])
        return example

    return ds.map(_resolve)


def _build_image_manifest_dataset(
    manifest_path: str | Path,
    dataset_root: Optional[Path] = None,
    cast_image: bool = True,
) -> Dataset:
    """图像类清单的通用构建流程：读 JSONL -> 解析路径 -> （可选）把 ``image_path`` 列改名为 ``image`` 并转成 Image 特征。

    ``cast_image=False`` 用于只想拿路径字符串、不想触发图片解码的场景（例如只做清单校验）。
    """
    dataset_root = dataset_root or DATASET_ROOT
    ds = _load_jsonl(manifest_path)
    ds = _resolve_paths(ds, "image_path", dataset_root)
    if cast_image:
        ds = ds.rename_column("image_path", "image").cast_column("image", HFImage())
    return ds


def build_character_dataset(
    manifest_path: str | Path = DEFAULT_METADATA_DIR / "character_manifest.jsonl",
    dataset_root: Optional[Path] = None,
    cast_image: bool = True,
) -> Dataset:
    """加载角色参考图清单（character_manifest.jsonl），供角色 LoRA 训练使用。"""
    return _build_image_manifest_dataset(manifest_path, dataset_root, cast_image)


def build_scene_dataset(
    manifest_path: str | Path = DEFAULT_METADATA_DIR / "scene_manifest.jsonl",
    dataset_root: Optional[Path] = None,
    cast_image: bool = True,
) -> Dataset:
    """加载场景 / 环境参考图清单（scene_manifest.jsonl）。"""
    return _build_image_manifest_dataset(manifest_path, dataset_root, cast_image)


def build_style_dataset(
    manifest_path: str | Path = DEFAULT_METADATA_DIR / "style_manifest.jsonl",
    dataset_root: Optional[Path] = None,
    cast_image: bool = True,
) -> Dataset:
    """加载视觉风格参考图清单（style_manifest.jsonl），供风格 LoRA 训练使用。"""
    return _build_image_manifest_dataset(manifest_path, dataset_root, cast_image)


def build_image_caption_dataset(
    manifest_path: str | Path,
    dataset_root: Optional[Path] = None,
    cast_image: bool = True,
) -> Dataset:
    """任意"image_path + caption"清单的通用加载器。

    character_manifest.jsonl、scene_manifest.jsonl、style_manifest.jsonl 都满足这个约定。
    SDXL / FLUX 的 LoRA 训练脚本只关心图片和描述文本，不关心各领域特有的额外列，用这个函数即可。
    """
    return _build_image_manifest_dataset(manifest_path, dataset_root, cast_image)


def build_voice_dataset(
    manifest_path: str | Path = DEFAULT_METADATA_DIR / "voice_manifest.jsonl",
) -> Dataset:
    """加载语音清单（voice_manifest.jsonl）；``audio_path`` 保持字符串，不做音频解码。"""
    return _load_jsonl(manifest_path)


def build_video_dataset(
    manifest_path: str | Path = DEFAULT_METADATA_DIR / "video_manifest.jsonl",
) -> Dataset:
    """加载视频片段清单（video_manifest.jsonl）；``video_path`` 保持字符串，不做视频解码。"""
    return _load_jsonl(manifest_path)


if __name__ == "__main__":
    # 冒烟测试：依次构建五类数据集并打印结构，确认清单格式与加载器兼容
    for name, builder in [
        ("character", build_character_dataset),
        ("scene", build_scene_dataset),
        ("style", build_style_dataset),
        ("voice", build_voice_dataset),
        ("video", build_video_dataset),
    ]:
        ds = builder()
        print(name, ds)
