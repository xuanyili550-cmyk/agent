"""单集规划器：不调用 LLM，纯规则化地为某一集确定时长约束、所属幕（act）等元数据。

竖屏短剧的核心生产约束是「单集必须极短、开场必须极快建立钩子」，这些约束是
业务规则而非创意判断，所以不交给 LLM 生成，而是用固定常量和简单查表逻辑
（根据 SeasonArcPlanner 产出的 beats 找到当前集所属的幕）来保证确定性和可复现性。
"""

from __future__ import annotations

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        # 向上遍历父目录找到项目根，把结构化数据模块目录和本模块目录插入
        # sys.path，这样无论从哪个工作目录启动脚本都能正常 import schemas。
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

import schemas as sch

# 单集允许的最短时长（秒）：低于此值难以承载一个完整的钩子+剧情节拍。
MIN_EPISODE_SECONDS = 60
# 单集允许的最长时长（秒）：竖屏短剧要保持强节奏，过长会拖慢刷剧体验。
MAX_EPISODE_SECONDS = 600
HOOK_WINDOW_SECONDS = 3  # 短剧核心约束：开场必须在极短时间内建立强钩子，否则用户会划走


class EpisodePlanner:
    """为单集生成时长、所属幕次等规划信息的规划器（纯规则，不调用 LLM）。"""

    def plan_episode(
        self,
        season_arc: sch.SeasonArc,
        episode_number: int,
        target_duration_seconds: int = 300,
    ) -> dict:
        """校验目标时长并生成该集的规划信息字典（所属幕、开场钩子窗口等）。"""
        if not (MIN_EPISODE_SECONDS <= target_duration_seconds <= MAX_EPISODE_SECONDS):
            raise ValueError(f"短剧单集时长必须在 {MIN_EPISODE_SECONDS}-{MAX_EPISODE_SECONDS} 秒之间，收到 {target_duration_seconds}")
        episode_id = f"ep_{episode_number:03d}"
        # 在季度大纲的 beats 中查找包含当前集 id 的那一幕；找不到则兜底用第一幕，
        # 避免因为数据不完整直接抛异常中断整条流水线。
        beat = next((b for b in season_arc.beats if episode_id in b.episode_ids), season_arc.beats[0])
        return {
            "episode_id": episode_id,
            "episode_number": episode_number,
            "act": beat.act,
            "act_description": beat.description,
            "target_duration_seconds": target_duration_seconds,
            "hook_window_seconds": HOOK_WINDOW_SECONDS,
            "requires_cliffhanger": True,
        }

    def beat_for_episode(self, season_arc: sch.SeasonArc, episode_number: int) -> sch.StoryBeat:
        """根据集数查出其所属的剧情节拍（StoryBeat），找不到时兜底返回第一幕。"""
        episode_id = f"ep_{episode_number:03d}"
        return next((b for b in season_arc.beats if episode_id in b.episode_ids), season_arc.beats[0])
