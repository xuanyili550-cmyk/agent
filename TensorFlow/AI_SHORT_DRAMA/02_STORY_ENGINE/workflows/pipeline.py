from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

from langgraph.graph import END, StateGraph

import schemas as sch
from agents.character_agent import CharacterAgent
from agents.episode_agent import EpisodeAgent
from agents.prompt_agent import PromptAgent
from agents.screenplay_agent import ScreenplayAgent
from agents.story_agent import StoryAgent
from agents.storyboard_agent import StoryboardAgent
from planners.episode_planner import EpisodePlanner
from planners.season_arc_planner import SeasonArcPlanner

ROLE_HINTS = [
    "女主角，重生复仇者",
    "男主角，冷峻总裁，后期成为女主盟友",
    "反派，女主的继妹，表面无辜实则心机深重",
    "反派，女主前未婚夫，懦弱贪婪",
    "反派，女主继母，幕后主谋",
    "配角，男主的得力秘书",
]


class DramaState(TypedDict, total=False):
    idea: str
    story_bible: sch.StoryBible
    season_arc: sch.SeasonArc
    characters: list[sch.Character]
    episode_number: int
    target_duration_seconds: int
    episode: sch.Episode
    script: sch.Script
    scenes: list[sch.Scene]
    shots: list[sch.Shot]
    image_prompts: list[sch.ImagePrompt]
    video_prompts: list[sch.VideoPrompt]


@dataclass
class AgentBundle:
    story_agent: StoryAgent
    character_agent: CharacterAgent
    episode_agent: EpisodeAgent
    screenplay_agent: ScreenplayAgent
    storyboard_agent: StoryboardAgent
    prompt_agent: PromptAgent
    season_planner: SeasonArcPlanner
    episode_planner: EpisodePlanner


def build_graph(
    bundle: AgentBundle,
    num_characters: int = 6,
    num_scenes: int = 4,
    episode_number: int = 1,
):
    role_hints = ROLE_HINTS[:num_characters]

    def node_story_bible(state: DramaState) -> dict:
        return {"story_bible": bundle.story_agent.generate_story_bible(state["idea"])}

    def node_season_arc(state: DramaState) -> dict:
        return {"season_arc": bundle.season_planner.plan(state["story_bible"])}

    def node_characters(state: DramaState) -> dict:
        characters = [bundle.character_agent.generate_character(state["story_bible"], hint) for hint in role_hints]
        return {"characters": characters}

    def node_episode(state: DramaState) -> dict:
        target_duration = state.get("target_duration_seconds", 300)
        constraints = bundle.episode_planner.plan_episode(state["season_arc"], episode_number, target_duration)
        beat = bundle.episode_planner.beat_for_episode(state["season_arc"], episode_number)
        episode = bundle.episode_agent.generate_episode(state["season_arc"], episode_number, beat, constraints)
        return {"episode": episode, "episode_number": episode_number}

    def node_script(state: DramaState) -> dict:
        return {"script": bundle.screenplay_agent.generate_script(state["episode"], state["characters"])}

    def node_storyboard(state: DramaState) -> dict:
        scenes: list[sch.Scene] = []
        shots: list[sch.Shot] = []
        for scene_number in range(1, num_scenes + 1):
            scene = bundle.storyboard_agent.generate_scene(state["episode"], scene_number, state["script"])
            scene_shots = bundle.storyboard_agent.generate_shots(scene, state["episode"].episode_number)
            dialogue = bundle.storyboard_agent.generate_dialogue(scene, scene_shots)
            dialogue_by_shot: dict[str, list[str]] = {}
            for line in dialogue:
                dialogue_by_shot.setdefault(line.shot_id, []).append(line.id)
            scene_shots = [
                sh.model_copy(update={"dialogue_ids": dialogue_by_shot.get(sh.id, [])}) for sh in scene_shots
            ]
            scene = scene.model_copy(update={"shot_ids": [sh.id for sh in scene_shots], "dialogue": dialogue})
            scenes.append(scene)
            shots.extend(scene_shots)
        return {"scenes": scenes, "shots": shots}

    def node_prompts(state: DramaState) -> dict:
        image_prompts: list[sch.ImagePrompt] = []
        video_prompts: list[sch.VideoPrompt] = []
        for shot in state["shots"]:
            image_prompt = bundle.prompt_agent.generate_image_prompt(shot)
            video_prompt = bundle.prompt_agent.generate_video_prompt(shot, image_prompt)
            image_prompts.append(image_prompt)
            video_prompts.append(video_prompt)
        return {"image_prompts": image_prompts, "video_prompts": video_prompts}

    graph = StateGraph(DramaState)
    graph.add_node("story_bible", node_story_bible)
    graph.add_node("season_arc", node_season_arc)
    graph.add_node("characters", node_characters)
    graph.add_node("episode", node_episode)
    graph.add_node("script", node_script)
    graph.add_node("storyboard", node_storyboard)
    graph.add_node("prompts", node_prompts)

    graph.set_entry_point("story_bible")
    graph.add_edge("story_bible", "season_arc")
    graph.add_edge("season_arc", "characters")
    graph.add_edge("characters", "episode")
    graph.add_edge("episode", "script")
    graph.add_edge("script", "storyboard")
    graph.add_edge("storyboard", "prompts")
    graph.add_edge("prompts", END)

    return graph.compile()
