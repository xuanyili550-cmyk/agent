"""
模型注册表：定义各模态 ``registry.json`` 的条目结构，并提供加载 / 校验函数。

在流水线中的位置：06_MODELS 下按模态分目录（llm / image / video / tts ...），每个目录一个 ``registry.json``，
登记该模态可用的模型（HF id 或本地路径、版本、许可证、是否允许商用）。生成与训练脚本通过这里统一查询模型来源，
而不是把模型 id 硬编码在各处。

为什么要有 ``commercial_use_allowed`` 与 ``license`` 字段：短剧平台要商业发行，模型许可证是合规红线；
把它记进注册表并在加载时做结构校验，能在选型阶段就发现问题，而不是上线后才追溯。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

# 模态枚举：与 06_MODELS 下的子目录名对应
ModelType = Literal["llm", "image", "video", "vlm", "tts", "asr", "lipsync"]

MODELS_ROOT = Path(__file__).resolve().parent


class ModelRegistryEntry(BaseModel):
    """registry.json 中的一个模型条目。``commercial_use_allowed`` 允许为 None，表示许可证尚未人工确认。"""

    name: str
    type: ModelType
    source: str
    license: str
    commercial_use_allowed: Optional[bool] = None
    hf_id_or_local_path: str
    version: str
    notes: str


def load_registry(registry_path: str | Path) -> list[ModelRegistryEntry]:
    """读取单个 registry.json（顶层是条目数组）并逐条做 Pydantic 校验，字段缺失或类型错误会直接抛异常。"""
    with open(registry_path, encoding="utf-8") as f:
        raw = json.load(f)
    return [ModelRegistryEntry(**entry) for entry in raw]


def load_all_registries(models_root: str | Path = MODELS_ROOT) -> dict[str, list[ModelRegistryEntry]]:
    """扫描 ``models_root`` 下所有 ``*/registry.json``，返回 {模态目录名: 条目列表}。

    用 ``sorted`` 保证遍历顺序稳定，便于输出可复现、测试可断言。
    """
    models_root = Path(models_root)
    registries: dict[str, list[ModelRegistryEntry]] = {}
    for registry_file in sorted(models_root.glob("*/registry.json")):
        modality = registry_file.parent.name
        registries[modality] = load_registry(registry_file)
    return registries


if __name__ == "__main__":
    # 命令行自检：打印每个模态的条目数与商用许可状态
    for modality, entries in load_all_registries().items():
        print(f"{modality}: {len(entries)} entries")
        for entry in entries:
            print(f"  - {entry.name} ({entry.hf_id_or_local_path}) commercial_use_allowed={entry.commercial_use_allowed}")
