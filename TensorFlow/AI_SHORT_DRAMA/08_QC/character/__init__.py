"""角色一致性质检子包：导出 CharacterConsistencyChecker（检查器）、CharacterConsistencyResult（结果）
和 CLIPModelUnavailableError（CLIP 权重不可用时的异常，供调用方决定跳过还是失败）。"""

from .consistency_checker import (
    CharacterConsistencyChecker,
    CharacterConsistencyResult,
    CLIPModelUnavailableError,
)

__all__ = [
    "CharacterConsistencyChecker",
    "CharacterConsistencyResult",
    "CLIPModelUnavailableError",
]
