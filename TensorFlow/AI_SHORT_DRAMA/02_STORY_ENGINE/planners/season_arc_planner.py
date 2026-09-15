"""季度大纲规划器：把 Story Bible 的核心冲突按三幕结构拆解为季度剧情弧线（SeasonArc）。

同样不调用 LLM，而是用固定的三幕模板和均分算法把总集数分配到各幕，目的是让
整季的剧情节奏（开局建立钩子 -> 冲突升级 -> 真相揭晓与清算）在生成之前就
被结构化地确定下来，后续 EpisodePlanner/StoryboardAgent 只需按图索骥填充细节，
而不用每次都重新决定「这一集属于哪个阶段」。
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

# 固定的三幕结构名称，对应竖屏短剧的通用叙事节奏。
ACT_NAMES = ["第一幕：重生/开局与建立强钩子", "第二幕：冲突升级与正面交锋", "第三幕：真相揭晓与最终清算"]


class SeasonArcPlanner:
    """把 Story Bible 拆解为若干幕、每幕包含若干集的季度剧情弧线规划器。"""

    def __init__(self, num_episodes: int = 12, num_acts: int = 3):
        """初始化规划器，校验总集数不能少于幕数（否则无法保证每幕至少一集）。"""
        if num_episodes < num_acts:
            raise ValueError("num_episodes 不能小于 num_acts")
        self.num_episodes = num_episodes
        self.num_acts = num_acts

    def plan(self, story_bible: sch.StoryBible, season_number: int = 1) -> sch.SeasonArc:
        """根据 Story Bible 生成完整的季度剧情弧线（SeasonArc），包含各幕的剧情节拍。"""
        episode_ids = [f"ep_{i:03d}" for i in range(1, self.num_episodes + 1)]
        # 按幕数把集数尽量均分，最后一幕吸收整除后剩余的集数，保证集数不丢失。
        per_act = self.num_episodes // self.num_acts
        beats: list[sch.StoryBeat] = []
        cursor = 0
        for i in range(self.num_acts):
            is_last = i == self.num_acts - 1
            end = self.num_episodes if is_last else cursor + per_act
            # 幕数超过预设名称数量时，用「第N幕」兜底命名，避免下标越界。
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
