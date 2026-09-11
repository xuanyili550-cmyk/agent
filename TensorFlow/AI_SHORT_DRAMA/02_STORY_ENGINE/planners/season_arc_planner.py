from __future__ import annotations

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

import schemas as sch

ACT_NAMES = ["第一幕：重生/开局与建立强钩子", "第二幕：冲突升级与正面交锋", "第三幕：真相揭晓与最终清算"]


class SeasonArcPlanner:
    def __init__(self, num_episodes: int = 12, num_acts: int = 3):
        if num_episodes < num_acts:
            raise ValueError("num_episodes 不能小于 num_acts")
        self.num_episodes = num_episodes
        self.num_acts = num_acts

    def plan(self, story_bible: sch.StoryBible, season_number: int = 1) -> sch.SeasonArc:
        episode_ids = [f"ep_{i:03d}" for i in range(1, self.num_episodes + 1)]
        per_act = self.num_episodes // self.num_acts
        beats: list[sch.StoryBeat] = []
        cursor = 0
        for i in range(self.num_acts):
            is_last = i == self.num_acts - 1
            end = self.num_episodes if is_last else cursor + per_act
            act_name = ACT_NAMES[i] if i < len(ACT_NAMES) else f"第{i + 1}幕"
            beats.append(
                sch.StoryBeat(
                    act=act_name,
                    description=f"{act_name}：推进「{story_bible.main_conflict}」这一核心矛盾",
                    episode_ids=episode_ids[cursor:end],
                )
            )
            cursor = end
        return sch.SeasonArc(
            id=f"season_{season_number:03d}",
            season_number=season_number,
            title=f"{story_bible.title}·第{season_number}季",
            synopsis=story_bible.logline,
            central_question=f"主角能否在「{story_bible.main_conflict}」中逆转局势？",
            resolution="随各集剧情推进逐步揭示，最终在最后一幕完成清算与反转",
            beats=beats,
            episode_ids=episode_ids,
        )
