"""
Loads 04_DATASET/metadata/*.jsonl manifests into datasets.Dataset objects.
Image-backed manifests (character/scene/style) resolve image_path relative to
the dataset root and cast it to a datasets Image feature. Voice/video
manifests keep their path columns as plain strings since no real audio/video
files are shipped with this repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset
from datasets import Image as HFImage

DATASET_ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA_DIR = DATASET_ROOT / "metadata"


def _load_jsonl(manifest_path: str | Path) -> Dataset:
    return load_dataset("json", data_files=str(manifest_path), split="train")


def _resolve_paths(ds: Dataset, column: str, dataset_root: Path) -> Dataset:
    def _resolve(example):
        example[column] = str(dataset_root / example[column])
        return example

    return ds.map(_resolve)


def _build_image_manifest_dataset(
    manifest_path: str | Path,
    dataset_root: Optional[Path] = None,
    cast_image: bool = True,
) -> Dataset:
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
    return _build_image_manifest_dataset(manifest_path, dataset_root, cast_image)


def build_scene_dataset(
    manifest_path: str | Path = DEFAULT_METADATA_DIR / "scene_manifest.jsonl",
    dataset_root: Optional[Path] = None,
    cast_image: bool = True,
) -> Dataset:
    return _build_image_manifest_dataset(manifest_path, dataset_root, cast_image)


def build_style_dataset(
    manifest_path: str | Path = DEFAULT_METADATA_DIR / "style_manifest.jsonl",
    dataset_root: Optional[Path] = None,
    cast_image: bool = True,
) -> Dataset:
    return _build_image_manifest_dataset(manifest_path, dataset_root, cast_image)


def build_image_caption_dataset(
    manifest_path: str | Path,
    dataset_root: Optional[Path] = None,
    cast_image: bool = True,
) -> Dataset:
    """Generic loader for any manifest that has image_path + caption columns
    (character_manifest.jsonl, scene_manifest.jsonl, style_manifest.jsonl all qualify).
    Useful for SDXL/FLUX LoRA training scripts that don't care about the extra
    domain-specific columns."""
    return _build_image_manifest_dataset(manifest_path, dataset_root, cast_image)


def build_voice_dataset(
    manifest_path: str | Path = DEFAULT_METADATA_DIR / "voice_manifest.jsonl",
) -> Dataset:
    return _load_jsonl(manifest_path)


def build_video_dataset(
    manifest_path: str | Path = DEFAULT_METADATA_DIR / "video_manifest.jsonl",
) -> Dataset:
    return _load_jsonl(manifest_path)


if __name__ == "__main__":
    for name, builder in [
        ("character", build_character_dataset),
        ("scene", build_scene_dataset),
        ("style", build_style_dataset),
        ("voice", build_voice_dataset),
        ("video", build_video_dataset),
    ]:
        ds = builder()
        print(name, ds)
