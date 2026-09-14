"""流水线中间产物的交叉校验。

LLM 每个 agent 各自输出合法 JSON 不等于整体一致：StoryAgent 写的 character_ids 和
CharacterAgent 实际生成的角色 id 可能对不上；对白引用的 shot_id 可能不存在。这些错误
Pydantic 单对象校验查不出来，只能在流水线层面做交叉检查——而且要尽早查，错到分镜阶段
再发现，前面的 LLM 调用就全白花了。
"""

from __future__ import annotations

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

import schemas as sch


class DramaConsistencyError(ValueError):
    pass


def reconcile_story_bible(story_bible: sch.StoryBible, characters: list[sch.Character]) -> sch.StoryBible:
    """以 CharacterAgent 实际生成的角色 id 为准，回写 StoryBible.character_ids。

    StoryBible 是先生成的，那时角色还不存在，它写的 id 只是"计划"；角色真正生成后，
    真源应该是角色表。角色的 relationships 里引用的 id 也要在角色表内。
    """
    ids = [c.id for c in characters]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise DramaConsistencyError(f"角色 id 重复：{dupes}")
    id_set = set(ids)
    for c in characters:
        for rel in c.relationships:
            if rel.character_id not in id_set:
                raise DramaConsistencyError(f"角色 {c.id} 的关系引用了不存在的角色 {rel.character_id}")
    return story_bible.model_copy(update={"character_ids": ids})


def validate_episode_assets(
    episode: sch.Episode,
    scenes: list[sch.Scene],
    shots: list[sch.Shot],
    characters: list[sch.Character],
) -> None:
    """分镜阶段结束后的硬校验：所有引用必须闭合。任何一条不满足就抛错终止本集，
    不让坏数据流进生产队列。"""
    char_ids = {c.id for c in characters}
    scene_ids = {s.id for s in scenes}
    shot_ids = {s.id for s in shots}
    problems: list[str] = []

    for scene in scenes:
        if scene.episode_id != episode.id:
            problems.append(f"{scene.id}.episode_id={scene.episode_id} != {episode.id}")
        problems += [f"{scene.id} 出场角色 {cid} 不存在" for cid in scene.characters_present if cid not in char_ids]
        for sid in scene.shot_ids:
            if sid not in shot_ids:
                problems.append(f"{scene.id}.shot_ids 引用不存在的镜头 {sid}")
        for line in scene.dialogue:
            if line.shot_id not in shot_ids:
                problems.append(f"对白 {line.id} 指向不存在的镜头 {line.shot_id}")
            if line.character_id not in char_ids:
                problems.append(f"对白 {line.id} 的角色 {line.character_id} 不存在")
    for shot in shots:
        if shot.scene_id not in scene_ids:
            problems.append(f"{shot.id}.scene_id={shot.scene_id} 不存在")
        problems += [f"{shot.id} 角色 {cid} 不存在" for cid in shot.character if cid not in char_ids]
        if shot.episode != episode.episode_number:
            problems.append(f"{shot.id}.episode={shot.episode} != {episode.episode_number}")
    if len(shot_ids) != len(shots):
        problems.append("镜头 id 重复")

    if problems:
        raise DramaConsistencyError(f"{episode.id} 交叉校验失败：" + "；".join(problems[:20]))
