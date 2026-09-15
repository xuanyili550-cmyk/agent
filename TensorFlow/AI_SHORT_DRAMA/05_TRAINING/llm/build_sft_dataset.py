"""从故事引擎的结构化产出构造 SFT 数据集（instruction/response jsonl）。

在流水线中的位置：05_TRAINING/llm 的第一步（本脚本 -> sft_train.py 训练 -> evaluate_structured_output.py 评估）。
它把 02_STORY_ENGINE 各 agent 已经生成并通过校验（或人工审校通过）的结构化对象"反向"整理成训练样本，
让微调后的模型学会在同样的 prompt 下直接给出合法 JSON。

数据来源两种：
- ``--from-demo-output 02_STORY_ENGINE/demo_output``：demo 写出的 story/characters/episodes/scenes/shots/prompts.json；
- ``--from-db``：13_INFRA 数据库里所有 review_status=approved 的集（生产上真正积累的、人工审过的数据）。

每个 agent 的任务各构造一类样本，instruction 就是该 agent 在流水线里实际收到的 user prompt
（含 [TARGET_SCHEMA=...] 标记和枚举清单），response 是通过校验的 JSON——训练出来的模型可以
直接替换 LocalTransformersProvider 里的底座，prompt 格式零改动。

为什么 instruction 要与线上 prompt 逐字一致：SFT 学的是"这个 prompt -> 这个输出"的映射，训练与推理的 prompt 分布
一旦不同（哪怕只是少了枚举清单），微调收益就会大打折扣。

用法：
  python 05_TRAINING/llm/build_sft_dataset.py --from-demo-output 02_STORY_ENGINE/demo_output --out data/sft.jsonl
  python 05_TRAINING/llm/build_sft_dataset.py --from-db --out data/sft.jsonl
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

# 顶层目录名带数字前缀，无法用常规包导入；把仓库根、故事引擎、结构化数据目录塞进 sys.path
_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "02_STORY_ENGINE", _ROOT / "03_STRUCTURED_DATA"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import schemas as sch  # noqa: E402
from agents.base import _enum_cheatsheet  # noqa: E402


def _marker(schema: type) -> str:
    """生成 BaseAgent 在线上 prompt 末尾追加的 [TARGET_SCHEMA=...] 标记 + 枚举清单，保证训练样本与线上 prompt 一致。"""
    return (
        f"\n\n[TARGET_SCHEMA={schema.__name__}]\n"
        "只输出符合该 schema 字段结构的单个 JSON 对象，不要包含任何解释文字或 markdown 代码块标记。"
        f"{_enum_cheatsheet(schema)}"
    )


def _sample(task: str, instruction: str, obj: Any, schema: type, system_prompt_file: str) -> dict[str, Any]:
    """组装一条训练样本。``response`` 用 ``exclude_none=True`` 序列化：让模型学会省略空字段，而不是输出一堆 null。"""
    return {
        "task": task,
        "system_prompt_file": system_prompt_file,
        "instruction": instruction + _marker(schema),
        "response": obj.model_dump_json(exclude_none=True),
    }


def samples_from_state(
    story_bible: sch.StoryBible,
    characters: list[sch.Character],
    episodes: list[sch.Episode],
    scenes: list[sch.Scene],
    shots: list[sch.Shot],
    image_prompts: list[sch.ImagePrompt],
    video_prompts: list[sch.VideoPrompt],
    scripts: dict[str, sch.Script] | None = None,
    idea: str = "",
) -> Iterator[dict[str, Any]]:
    """把一个项目的完整结构化状态展开成各 agent 任务的训练样本（生成器）。

    覆盖的任务：story_bible / character / episode / script / scene / shots / image_prompt / video_prompt。
    每种任务的 instruction 文案都照抄对应 agent 的 user prompt 模板，只把变量替换为该项目的实际值。
    """
    # demo 产出没有保存原始创意，退而用 logline 充当"故事创意"输入
    idea = idea or story_bible.logline
    yield _sample(
        "story_bible",
        f"故事创意：{idea}\n\n请基于以上创意生成一部竖屏短剧的完整 Story Bible，包含 world（世界观设定，含至少 2 个 location）、"
        "genre、themes、tone、target_audience、main_conflict、unique_selling_point、episode_count_planned、character_ids。",
        story_bible,
        sch.StoryBible,
        "story_agent_system.txt",
    )
    for c in characters:
        yield _sample(
            "character",
            f"Story Bible 标题：{story_bible.title}\nlogline：{story_bible.logline}\n世界观：{story_bible.world.name}\n"
            f"请生成一个角色：{c.role.value}，与故事核心冲突「{story_bible.main_conflict}」直接相关。id 使用 '{c.id}'。",
            c,
            sch.Character,
            "character_agent_system.txt",
        )
    for ep in episodes:
        yield _sample(
            "episode",
            f"季线标题：{story_bible.title}\n这是第 {ep.episode_number} 集，season_id 为 '{ep.season_id}'，id 请使用 '{ep.id}'。\n"
            f"时长约束：duration_seconds 必须在 {ep.duration_seconds} 秒左右。强制约束：hook 与 cliffhanger 都不能为空。",
            ep,
            sch.Episode,
            "episode_agent_system.txt",
        )
        # 剧本是可选的（demo 产出没有，DB 里可能有），有才构造 script 样本
        script = (scripts or {}).get(ep.id)
        if script:
            yield _sample(
                "script",
                f"集标题：{ep.title}\n本集梗概：{ep.synopsis}\n开场钩子：{ep.hook}\n结尾悬念：{ep.cliffhanger}\n"
                f"请为 episode_id='{ep.id}' 生成完整剧本文本，id 使用 'script_' 加集号。",
                script,
                sch.Script,
                "screenplay_agent_system.txt",
            )
    shots_by_scene: dict[str, list[sch.Shot]] = {}
    for s in shots:
        shots_by_scene.setdefault(s.scene_id, []).append(s)
    for scene in scenes:
        ep = next((e for e in episodes if e.id == scene.episode_id), None)
        # 分场 agent 在线上是先产出不含 shot_ids / dialogue 的"骨架"，分镜再另行填充，训练目标要与之对齐
        bare = scene.model_copy(update={"shot_ids": [], "dialogue": []})
        yield _sample(
            "scene",
            f"请为 episode_id='{scene.episode_id}' 生成第 {scene.scene_number} 场（scene_number={scene.scene_number}）的分场信息，"
            f"id 使用 '{scene.id}'。shot_ids 和 dialogue 先留空列表。",
            bare,
            sch.Scene,
            "storyboard_agent_system.txt",
        )
        scene_shots = shots_by_scene.get(scene.id, [])
        if scene_shots:
            yield _sample(
                "shots",
                f"场次描述：{scene.description}\n出场角色：{'、'.join(scene.characters_present)}\n"
                f"请为 scene_id='{scene.id}'（episode={ep.episode_number if ep else scene.scene_number}, scene={scene.scene_number}）生成分镜镜头，"
                '输出必须是 {"shots": [...]} 结构。',
                sch.ShotsFile(shots=scene_shots),
                sch.ShotsFile,
                "storyboard_agent_system.txt",
            )
    prompts_by_shot = {p.shot_id: p for p in image_prompts}
    for shot in shots:
        ip = prompts_by_shot.get(shot.id)
        if ip:
            yield _sample(
                "image_prompt",
                f"镜头信息：地点={shot.location}，角色={'、'.join(shot.character)}，动作={shot.action}，情绪={shot.emotion.value}，"
                f"景别={shot.camera.shot_size.value}，机位角度={shot.camera.angle.value}。\n请为 shot_id='{shot.id}' 生成图像生成 prompt，id 使用 '{ip.id}'。",
                ip,
                sch.ImagePrompt,
                "prompt_agent_system.txt",
            )
    for vp in video_prompts:
        shot = next((s for s in shots if s.id == vp.shot_id), None)
        if shot:
            yield _sample(
                "video_prompt",
                f"镜头运动：{shot.camera.movement.value}，时长：{shot.duration} 秒。请为 shot_id='{shot.id}' 生成视频生成 prompt，id 使用 '{vp.id}'。",
                vp,
                sch.VideoPrompt,
                "prompt_agent_system.txt",
            )


def load_demo_output(directory: str | Path) -> dict[str, Any]:
    """读取 02_STORY_ENGINE demo 写出的一组 JSON 文件，返回可直接 ``**`` 展开传给 ``samples_from_state`` 的字典。"""
    d = Path(directory)
    story = sch.StoryFile.model_validate_json((d / "story.json").read_text(encoding="utf-8"))
    return {
        "story_bible": story.story_bible,
        "characters": sch.CharactersFile.model_validate_json((d / "characters.json").read_text(encoding="utf-8")).characters,
        "episodes": sch.EpisodesFile.model_validate_json((d / "episodes.json").read_text(encoding="utf-8")).episodes,
        "scenes": sch.ScenesFile.model_validate_json((d / "scenes.json").read_text(encoding="utf-8")).scenes,
        "shots": sch.ShotsFile.model_validate_json((d / "shots.json").read_text(encoding="utf-8")).shots,
        "image_prompts": sch.PromptsFile.model_validate_json((d / "prompts.json").read_text(encoding="utf-8")).image_prompts,
        "video_prompts": sch.PromptsFile.model_validate_json((d / "prompts.json").read_text(encoding="utf-8")).video_prompts,
    }


def load_from_db(only_approved: bool = True) -> Iterator[dict[str, Any]]:
    """每个项目一份 state：只取审校通过的集（这才是值得学的数据）。

    为什么默认只取 approved：数据库里也存着被编审打回的集，把它们喂给模型等于教它犯同样的错；
    人工审校通过才是"正确答案"的信号。13_INFRA 的模块按需动态导入，避免不用 DB 时也要装数据库依赖。
    """
    models = importlib.import_module("13_INFRA.database.models")
    session_mod = importlib.import_module("13_INFRA.database.session")
    repo = importlib.import_module("13_INFRA.database.repository")
    with session_mod.session_scope() as db:
        for bible_row in db.query(models.StoryBible).all():
            project_id = bible_row.project_id
            chars = [sch.Character.model_validate(r.data) for r in db.query(models.Character).filter_by(project_id=project_id).all() if r.data]
            ep_rows = db.query(models.Episode).filter_by(project_id=project_id).all()
            if only_approved:
                ep_rows = [r for r in ep_rows if r.review_status == "approved"]
            episodes, scripts, scenes, shots = [], {}, [], []
            for r in ep_rows:
                if not r.data:
                    continue
                ep = sch.Episode.model_validate(r.data)
                episodes.append(ep)
                if r.script:
                    scripts[ep.id] = sch.Script.model_validate(r.script)
                scenes.extend(repo.load_scenes_for_episode(db, r.episode_id))
                shots.extend(s for _, s in repo.load_shots_for_episode(db, r.episode_id))
            # prompt_id 以 "<project_id><SCOPE_SEP>" 开头，用 LIKE 前缀匹配拿到该项目的全部 prompt
            prompt_rows = db.query(models.Prompt).filter(models.Prompt.prompt_id.like(f"{project_id}{repo.SCOPE_SEP}%")).all()
            yield {
                "story_bible": sch.StoryBible.model_validate(bible_row.data),
                "characters": chars,
                "episodes": episodes,
                "scripts": scripts,
                "scenes": scenes,
                "shots": shots,
                "image_prompts": [sch.ImagePrompt.model_validate(r.data) for r in prompt_rows if r.kind == "image"],
                "video_prompts": [sch.VideoPrompt.model_validate(r.data) for r in prompt_rows if r.kind == "video"],
            }


def write_jsonl(samples: Iterator[dict[str, Any]], out: str | Path) -> int:
    """把样本流逐行写成 JSONL（``ensure_ascii=False`` 保留中文可读性），返回写入条数。"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    """命令行入口：至少指定一种数据来源（可同时指定，样本会拼接），写出 JSONL 并打印条数。"""
    parser = argparse.ArgumentParser(description="构造 SFT 数据集")
    parser.add_argument("--from-demo-output", help="02_STORY_ENGINE/demo_output 这样的目录")
    parser.add_argument("--from-db", action="store_true", help="从 13_INFRA 数据库读审校通过的集")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not args.from_demo_output and not args.from_db:
        parser.error("需要 --from-demo-output 或 --from-db")

    def gen():
        """按参数依次产出 demo 与 DB 两路样本；用生成器避免把全部样本先堆在内存里。"""
        if args.from_demo_output:
            yield from samples_from_state(**load_demo_output(args.from_demo_output))
        if args.from_db:
            for state in load_from_db():
                yield from samples_from_state(**state)

    n = write_jsonl(gen(), args.out)
    print(f"写入 {n} 条样本 -> {args.out}")


if __name__ == "__main__":
    main()
