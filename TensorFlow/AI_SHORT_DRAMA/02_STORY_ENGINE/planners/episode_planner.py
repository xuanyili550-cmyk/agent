from __future__ import annotations

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

import schemas as sch

MIN_EPISODE_SECONDS = 60
MAX_EPISODE_SECONDS = 600
HOOK_WINDOW_SECONDS = 3  # 短剧核心约束：开场必须在极短时间内建立强钩子，否则用户会划走


class EpisodePlanner:
    def plan_episode(
        self,
        season_arc: sch.SeasonArc,
        episode_number: int,
        target_duration_seconds: int = 300,
    ) -> dict:
        if not (MIN_EPISODE_SECONDS <= target_duration_seconds <= MAX_EPISODE_SECONDS):
            raise ValueError(f"短剧单集时长必须在 {MIN_EPISODE_SECONDS}-{MAX_EPISODE_SECONDS} 秒之间，收到 {target_duration_seconds}")
        episode_id = f"ep_{episode_number:03d}"
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
        episode_id = f"ep_{episode_number:03d}"
        return next((b for b in season_arc.beats if episode_id in b.episode_ids), season_arc.beats[0])
