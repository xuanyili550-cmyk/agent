"""Pydantic model + loader/validator for the per-modality registry.json files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

ModelType = Literal["llm", "image", "video", "vlm", "tts", "asr", "lipsync"]

MODELS_ROOT = Path(__file__).resolve().parent


class ModelRegistryEntry(BaseModel):
    name: str
    type: ModelType
    source: str
    license: str
    commercial_use_allowed: Optional[bool] = None
    hf_id_or_local_path: str
    version: str
    notes: str


def load_registry(registry_path: str | Path) -> list[ModelRegistryEntry]:
    with open(registry_path, encoding="utf-8") as f:
        raw = json.load(f)
    return [ModelRegistryEntry(**entry) for entry in raw]


def load_all_registries(models_root: str | Path = MODELS_ROOT) -> dict[str, list[ModelRegistryEntry]]:
    models_root = Path(models_root)
    registries: dict[str, list[ModelRegistryEntry]] = {}
    for registry_file in sorted(models_root.glob("*/registry.json")):
        modality = registry_file.parent.name
        registries[modality] = load_registry(registry_file)
    return registries


if __name__ == "__main__":
    for modality, entries in load_all_registries().items():
        print(f"{modality}: {len(entries)} entries")
        for entry in entries:
            print(f"  - {entry.name} ({entry.hf_id_or_local_path}) commercial_use_allowed={entry.commercial_use_allowed}")
