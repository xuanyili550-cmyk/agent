"""故事引擎流水线（LangGraph）。

节点顺序：
    story_bible -> season_arc(章节规划师) -> characters -> [每集] episode -> script -> editorial(质检官+总编审)
    -> storyboard -> validate -> prompts -> END

多集：``episode_numbers`` 给几集就连续生成几集，每集的 EpisodeAgent 都拿到前几集的"前情提要"。
编审闭环：质检官不通过或总编审 revise 时，把 required_changes 回灌给编剧重写剧本，最多
``max_revision_rounds`` 轮；总编审 reject 或轮次用尽仍不过 -> 记录判定并把该集标为 needs_human_review，
不再继续分镜（由 13_INFRA 的流水线停在 awaiting_review 等人）。
提示词：``prompt_mode="llm"`` 用 PromptAgent（质量高、要花 token），``"template"`` 用 jinja 模板
（零成本、确定性，适合开发和批量草稿）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

# 向上找到项目根目录（含 03_STRUCTURED_DATA 的那一层），把 03_STRUCTURED_DATA 和 02_STORY_ENGINE
# 都加进 sys.path，这样本文件才能用顶层包名 import schemas / agents / generators / planners
for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

from langgraph.graph import END, StateGraph

import schemas as sch
from agents.character_agent import CharacterAgent
from agents.chief_editor_agent import ChiefEditorAgent
from agents.episode_agent import EpisodeAgent
from agents.prompt_agent import PromptAgent
from agents.qc_officer_agent import QCOfficerAgent
from agents.screenplay_agent import ScreenplayAgent
from agents.story_agent import StoryAgent
from agents.storyboard_agent import StoryboardAgent
from generators import build_image_prompt, build_video_prompt
from planners.episode_planner import EpisodePlanner
from planners.season_arc_planner import SeasonArcPlanner
from workflows.validation import reconcile_story_bible, validate_episode_assets

ROLE_HINTS = [
    "女主角，重生复仇者",
    "男主角，冷峻总裁，后期成为女主盟友",
    "反派，女主的继妹，表面无辜实则心机深重",
    "反派，女主前未婚夫，懦弱贪婪",
    "反派，女主继母，幕后主谋",
    "配角，男主的得力秘书",
]

PromptMode = Literal["llm", "template"]


class DramaState(TypedDict, total=False):
    """LangGraph 在各节点间流转的共享状态。

    ``total=False``：所有字段都是可选的，因为流水线是逐节点增量填充状态的，
    早期节点执行时，后面才会出现的字段（如 scenes/shots）还不存在。
    """

    idea: str
    story_bible: sch.StoryBible
    season_arc: sch.SeasonArc
    characters: list[sch.Character]
    target_duration_seconds: int
    # 多集：列表按集号排序；episode/script 保留为"最后一集"，兼容只处理单集的旧调用方
    episodes: list[sch.Episode]
    scripts: list[sch.Script]
    episode: sch.Episode
    script: sch.Script
    episode_number: int
    scenes: list[sch.Scene]
    shots: list[sch.Shot]
    image_prompts: list[sch.ImagePrompt]
    video_prompts: list[sch.VideoPrompt]
    # 编审结果：episode_id -> {"qc": QCVerdict, "editorial": EditorialDecision, "revision_rounds": int, "status": ...}
    editorial_by_episode: dict[str, dict]
    # 编审没通过的集：这些集不做分镜，等人工处理
    episodes_needing_review: list[str]


@dataclass
class AgentBundle:
    """流水线要用到的所有 agent / planner 实例的容器。

    集中管理便于测试时用假 agent 替换真实实现；qc_officer/chief_editor 允许省略
    （省略即跳过编审闭环，见 ``_editorial_loop``）。
    """

    story_agent: StoryAgent
    character_agent: CharacterAgent
    episode_agent: EpisodeAgent
    screenplay_agent: ScreenplayAgent
    storyboard_agent: StoryboardAgent
    prompt_agent: PromptAgent | None
    season_planner: SeasonArcPlanner | object  # SeasonArcPlanner 或 ChapterPlannerAgent，都有 .plan(story_bible)
    episode_planner: EpisodePlanner
    qc_officer: QCOfficerAgent | None = None
    chief_editor: ChiefEditorAgent | None = None
    max_revision_rounds: int = 2
    extra: dict = field(default_factory=dict)


def _editorial_loop(
    bundle: AgentBundle,
    state: DramaState,
    episode: sch.Episode,
    script: sch.Script,
    scenes: list[sch.Scene],
    shots: list[sch.Shot],
) -> tuple[sch.Episode, sch.Script, dict]:
    """质检官 -> 总编审 -> （打回则编剧重写）循环。返回最终的 episode/script 和判定记录。"""
    record: dict = {"revision_rounds": 0, "status": "approved"}
    if bundle.qc_officer is None or bundle.chief_editor is None:
        record["status"] = "skipped"
        return episode, script, record

    for round_no in range(bundle.max_revision_rounds + 1):
        verdict = bundle.qc_officer.review(episode, script, scenes, shots, state["characters"])
        decision = bundle.chief_editor.decide(state["story_bible"], state["season_arc"], episode, script, verdict, revision_round=round_no)
        record.update(
            {
                "qc": verdict.model_dump(mode="json"),
                "editorial": decision.model_dump(mode="json"),
                "revision_rounds": round_no,
            }
        )
        if decision.decision == sch.EditorialDecisionType.APPROVE and verdict.passed:
            record["status"] = "approved"
            return episode, script, record
        if decision.decision == sch.EditorialDecisionType.REJECT:
            record["status"] = "rejected"
            return episode, script, record
        if round_no == bundle.max_revision_rounds:
            record["status"] = "needs_human_review"
            return episode, script, record
        # revise：把意见回灌给编剧，在上一版基础上改
        notes = list(decision.required_changes) or [i.suggestion or i.description for i in verdict.issues]
        script = bundle.screenplay_agent.generate_script(episode, state["characters"], previous_script=script, revision_notes=notes)
    return episode, script, record  # pragma: no cover


def build_graph(
    bundle: AgentBundle,
    num_characters: int = 6,
    num_scenes: int = 4,
    episode_number: int = 1,
    episode_numbers: list[int] | None = None,
    prompt_mode: PromptMode = "llm",
):
    """组装并编译完整的 LangGraph 流水线图。

    各节点函数在这里以闭包形式定义（而不是模块级函数），因为它们都要用到
    bundle / role_hints / episode_numbers / prompt_mode 这几个构图参数：闭包比每个
    节点都显式传参更省事，也符合 LangGraph 节点签名只接受 state 一个参数的约定。
    """
    role_hints = ROLE_HINTS[:num_characters]
    episode_numbers = sorted(episode_numbers or [episode_number])
    if prompt_mode == "llm" and bundle.prompt_agent is None:
        raise ValueError("prompt_mode='llm' 需要 bundle.prompt_agent")

    def node_story_bible(state: DramaState) -> dict:
        """流水线第一个节点：根据创意生成故事圣经（世界观、主线设定）。"""
        return {"story_bible": bundle.story_agent.generate_story_bible(state["idea"])}

    def node_season_arc(state: DramaState) -> dict:
        """基于故事圣经规划季度弧线（章节/节拍规划）。"""
        return {"season_arc": bundle.season_planner.plan(state["story_bible"])}

    def node_characters(state: DramaState) -> dict:
        """按角色提示词逐个生成角色，再以实际生成结果回写 story_bible 的角色 id，保证前后一致。"""
        characters = [bundle.character_agent.generate_character(state["story_bible"], hint) for hint in role_hints]
        # 交叉校验：以实际生成的角色为准回写 bible.character_ids
        story_bible = reconcile_story_bible(state["story_bible"], characters)
        return {"characters": characters, "story_bible": story_bible}

    def node_episodes(state: DramaState) -> dict:
        """逐集：剧集 -> 剧本 -> （编审预检，只看剧本层面）。分镜在下一节点做，编审在有分镜后再终审。"""
        target_duration = state.get("target_duration_seconds", 300)
        episodes: list[sch.Episode] = []
        scripts: list[sch.Script] = []
        for number in episode_numbers:
            constraints = bundle.episode_planner.plan_episode(state["season_arc"], number, target_duration)
            beat = bundle.episode_planner.beat_for_episode(state["season_arc"], number)
            episode = bundle.episode_agent.generate_episode(state["season_arc"], number, beat, constraints, previous_episodes=episodes)
            script = bundle.screenplay_agent.generate_script(episode, state["characters"])
            episodes.append(episode)
            scripts.append(script)
        return {"episodes": episodes, "scripts": scripts, "episode": episodes[-1], "script": scripts[-1], "episode_number": episode_numbers[-1]}

    def node_storyboard(state: DramaState) -> dict:
        """对每集出分镜并跑编审闭环：剧本被改写过的集要按新剧本重新出分镜，编审不通过的集不进后续硬校验与统计。"""
        scenes: list[sch.Scene] = []
        shots: list[sch.Shot] = []
        editorial: dict[str, dict] = {}
        needing_review: list[str] = []
        episodes = list(state["episodes"])
        scripts = list(state["scripts"])
        for idx, (episode, script) in enumerate(zip(episodes, scripts)):
            ep_scenes, ep_shots = _storyboard_episode(bundle, episode, script, num_scenes)
            episode, script, record = _editorial_loop(bundle, state, episode, script, ep_scenes, ep_shots)
            if record["revision_rounds"] > 0 and record["status"] == "approved":
                # 剧本改过了，分镜要按新剧本重出
                ep_scenes, ep_shots = _storyboard_episode(bundle, episode, script, num_scenes)
            editorial[episode.id] = record
            episodes[idx], scripts[idx] = episode, script
            if record["status"] in ("rejected", "needs_human_review"):
                needing_review.append(episode.id)
                continue
            validate_episode_assets(episode, ep_scenes, ep_shots, state["characters"])
            scenes.extend(ep_scenes)
            shots.extend(ep_shots)
        return {
            "scenes": scenes,
            "shots": shots,
            "episodes": episodes,
            "scripts": scripts,
            "episode": episodes[-1],
            "script": scripts[-1],
            "editorial_by_episode": editorial,
            "episodes_needing_review": needing_review,
        }

    def node_prompts(state: DramaState) -> dict:
        """给每个镜头生成图像/视频提示词：template 模式走零成本 jinja 模板，llm 模式走 PromptAgent（质量高、费 token）。"""
        image_prompts: list[sch.ImagePrompt] = []
        video_prompts: list[sch.VideoPrompt] = []
        for shot in state["shots"]:
            if prompt_mode == "template":
                image_prompt = build_image_prompt(shot)
                video_prompt = build_video_prompt(shot, image_prompt)
            else:
                image_prompt = bundle.prompt_agent.generate_image_prompt(shot, state["characters"])
                video_prompt = bundle.prompt_agent.generate_video_prompt(shot, image_prompt)
            image_prompts.append(image_prompt)
            video_prompts.append(video_prompt)
        return {"image_prompts": image_prompts, "video_prompts": video_prompts}

    graph = StateGraph(DramaState)
    graph.add_node("story_bible", node_story_bible)
    graph.add_node("season_arc", node_season_arc)
    graph.add_node("characters", node_characters)
    graph.add_node("episodes", node_episodes)
    graph.add_node("storyboard", node_storyboard)
    graph.add_node("prompts", node_prompts)

    graph.set_entry_point("story_bible")
    graph.add_edge("story_bible", "season_arc")
    graph.add_edge("season_arc", "characters")
    graph.add_edge("characters", "episodes")
    graph.add_edge("episodes", "storyboard")
    graph.add_edge("storyboard", "prompts")
    graph.add_edge("prompts", END)

    return graph.compile()


def _storyboard_episode(bundle: AgentBundle, episode: sch.Episode, script: sch.Script, num_scenes: int) -> tuple[list[sch.Scene], list[sch.Shot]]:
    """给单集生成场景与镜头：逐场景生成场景本身、镜头列表、对白，再把对白 id 回填进对应镜头的 dialogue_ids。"""
    scenes: list[sch.Scene] = []
    shots: list[sch.Shot] = []
    for scene_number in range(1, num_scenes + 1):
        scene = bundle.storyboard_agent.generate_scene(episode, scene_number, script)
        scene_shots = bundle.storyboard_agent.generate_shots(scene, episode.episode_number)
        dialogue = bundle.storyboard_agent.generate_dialogue(scene, scene_shots)
        dialogue_by_shot: dict[str, list[str]] = {}
        for line in dialogue:
            dialogue_by_shot.setdefault(line.shot_id, []).append(line.id)
        # 按 shot_id 把对白 id 分组，再回填到对应镜头；model_copy 避免直接改 agent 返回的对象
        scene_shots = [sh.model_copy(update={"dialogue_ids": dialogue_by_shot.get(sh.id, [])}) for sh in scene_shots]
        scene = scene.model_copy(update={"shot_ids": [sh.id for sh in scene_shots], "dialogue": dialogue, "episode_id": episode.id})
        scenes.append(scene)
        shots.extend(scene_shots)
    return scenes, shots
