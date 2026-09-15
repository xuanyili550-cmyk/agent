"""短剧制片助理（DramaAssistantAgent）：拿着一套工具在一次流水线运行的产出上回答问题、做检查、
标记需要人工介入的集——对应《Agent 搭建指南》"实战搭建：一个智能数据处理 Agent"的落地版。

指南里的示例工具是"搜索飞书文档 / 执行 SQL / 生成图表"，这里换成短剧生产真正需要的：

    list_characters / get_character        查角色表（相当于"搜索文档"）
    list_episodes / get_script / count_shots 查剧集、剧本正文、分镜统计（相当于"查数据库"）
    run_rule_check                         对一集跑质检官的代码硬校验（"验证"这一步，零 token）
    flag_episode_for_human_review          把一集标记为需要人工复核 —— 有副作用，requires_confirmation=True

用户说"帮我检查第 1 集有没有引用错误，有问题就标记人工复核"时，Agent 会自己规划：
    [思考] 先看有哪些集 -> [工具] list_episodes -> [工具] run_rule_check(ep_001)
    -> [思考] 发现 blocker -> [工具] flag_episode_for_human_review（过确认门）-> [回复] 汇总

``DramaContext`` 是工具们共享的只读数据快照：02 的 demo 直接从 LangGraph 的 DramaState 构造，
13_INFRA 从数据库构造（见 13_INFRA/queue/assistant_task.py），工具代码只写一份。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

import schemas as sch

from .base import PROMPTS_DIR, LLMProvider
from .memory import ConversationMemory
from .qc_officer_agent import rule_check
from .tool_agent import ToolAgent
from .tools import ConfirmationGate, ToolRegistry

FlagHandler = Callable[[str, str], None]  # (episode_id, reason) -> None


@dataclass
class DramaContext:
    """一次运行的产出快照，工具只读它；``flag_handler`` 是唯一的"写"出口。"""

    story_bible: sch.StoryBible | None = None
    characters: list[sch.Character] = field(default_factory=list)
    episodes: list[sch.Episode] = field(default_factory=list)
    scripts: dict[str, sch.Script] = field(default_factory=dict)  # episode_id -> Script
    scenes: dict[str, list[sch.Scene]] = field(default_factory=dict)  # episode_id -> scenes
    shots: dict[str, list[sch.Shot]] = field(default_factory=dict)  # episode_id -> shots
    editorial: dict[str, dict[str, Any]] = field(default_factory=dict)  # episode_id -> 编审记录
    flag_handler: FlagHandler | None = None  # 标记人工复核时的落库回调；None 时只记在 flagged 里
    flagged: list[dict[str, str]] = field(default_factory=list)  # 本次会话里被标记的集

    @classmethod
    def from_state(cls, state: dict[str, Any], flag_handler: FlagHandler | None = None) -> DramaContext:
        """从 LangGraph 的 DramaState（build_graph().invoke 的返回值）构造。"""
        scenes_by_ep: dict[str, list[sch.Scene]] = {}
        for scene in state.get("scenes", []):
            scenes_by_ep.setdefault(scene.episode_id, []).append(scene)
        scene_to_ep = {s.id: s.episode_id for s in state.get("scenes", [])}
        shots_by_ep: dict[str, list[sch.Shot]] = {}
        for shot in state.get("shots", []):
            shots_by_ep.setdefault(scene_to_ep.get(shot.scene_id, f"ep_{shot.episode:03d}"), []).append(shot)
        return cls(
            story_bible=state.get("story_bible"),
            characters=list(state.get("characters", [])),
            episodes=list(state.get("episodes", [])),
            scripts={s.episode_id: s for s in state.get("scripts", [])},
            scenes=scenes_by_ep,
            shots=shots_by_ep,
            editorial=dict(state.get("editorial_by_episode", {})),
            flag_handler=flag_handler,
        )

    def episode(self, episode_id: str) -> sch.Episode:
        """按 id 找集；找不到抛 LookupError（工具层会把它变成给模型看的错误信息）。"""
        for ep in self.episodes:
            if ep.id == episode_id:
                return ep
        raise LookupError(f"没有 id 为 {episode_id} 的集，可用：{[e.id for e in self.episodes]}")


def build_drama_toolbox(ctx: DramaContext) -> ToolRegistry:
    """把 DramaContext 包成一组工具。每个工具都是普通函数，签名即参数 schema，docstring 即工具说明。"""
    registry = ToolRegistry()

    @registry.tool
    def list_characters() -> list[dict[str, str]]:
        """列出全部角色的 id、姓名、定位（主角/反派等）。"""
        return [{"id": c.id, "name": c.name, "role": c.role.value} for c in ctx.characters]

    @registry.tool
    def get_character(character_id: str) -> dict[str, Any]:
        """按 character_id 取一个角色的完整设定（外貌、性格、动机、人物关系）。"""
        for c in ctx.characters:
            if c.id == character_id:
                return c.model_dump(mode="json")
        raise LookupError(f"没有 id 为 {character_id} 的角色，可用：{[c.id for c in ctx.characters]}")

    @registry.tool
    def list_episodes() -> list[dict[str, Any]]:
        """列出全部集：id、集号、标题、开场钩子、结尾悬念、编审状态。"""
        return [
            {
                "id": ep.id,
                "episode_number": ep.episode_number,
                "title": ep.title,
                "hook": ep.hook,
                "cliffhanger": ep.cliffhanger,
                "editorial_status": (ctx.editorial.get(ep.id) or {}).get("status", "unknown"),
            }
            for ep in ctx.episodes
        ]

    @registry.tool
    def get_script(episode_id: str, max_chars: int = 2000) -> dict[str, Any]:
        """取一集的剧本正文（默认最多 2000 字，避免撑爆上下文）。"""
        ctx.episode(episode_id)
        script = ctx.scripts.get(episode_id)
        if script is None:
            return {"episode_id": episode_id, "script": None, "note": "这一集还没有剧本"}
        return {"episode_id": episode_id, "version": script.version, "content": script.content[:max_chars], "truncated": len(script.content) > max_chars}

    @registry.tool
    def count_shots(episode_id: str) -> dict[str, Any]:
        """统计一集的场次数、镜头数、分镜总时长与目标时长。"""
        ep = ctx.episode(episode_id)
        shots = ctx.shots.get(episode_id, [])
        return {
            "episode_id": episode_id,
            "scenes": len(ctx.scenes.get(episode_id, [])),
            "shots": len(shots),
            "storyboard_seconds": round(sum(s.duration for s in shots), 1),
            "target_seconds": ep.duration_seconds,
        }

    @registry.tool
    def run_rule_check(episode_id: str) -> dict[str, Any]:
        """对一集跑质检官的代码硬校验（引用一致性、时长范围、钩子/悬念非空），返回问题清单；不消耗 LLM。"""
        ep = ctx.episode(episode_id)
        issues = rule_check(ep, ctx.scenes.get(episode_id, []), ctx.shots.get(episode_id, []), ctx.characters)
        return {
            "episode_id": episode_id,
            "issue_count": len(issues),
            "has_blocker": any(i.severity == sch.IssueSeverity.BLOCKER for i in issues),
            "issues": [i.model_dump(mode="json") for i in issues],
        }

    @registry.tool(requires_confirmation=True)
    def flag_episode_for_human_review(episode_id: str, reason: str) -> dict[str, str]:
        """把一集标记为需要人工复核并附上原因。这是有副作用的操作，需要人工确认。"""
        ctx.episode(episode_id)
        if ctx.flag_handler is not None:
            ctx.flag_handler(episode_id, reason)
        ctx.flagged.append({"episode_id": episode_id, "reason": reason})
        return {"episode_id": episode_id, "status": "flagged", "reason": reason}

    return registry


class DramaAssistantAgent(ToolAgent):
    """制片助理：系统提示词在 prompts/drama_assistant_system.txt，工具由 build_drama_toolbox 提供。"""

    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "drama_assistant_system.txt"

    def __init__(
        self,
        provider: LLMProvider,
        ctx: DramaContext,
        *,
        max_tool_rounds: int = 5,
        gate: ConfirmationGate | None = None,
        memory: ConversationMemory | None = None,
        max_retries: int = 3,
    ) -> None:
        """``ctx`` 决定工具能看到什么数据；``gate`` 决定 flag 工具能不能自动执行。"""
        self.ctx = ctx
        super().__init__(
            provider,
            self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"),
            build_drama_toolbox(ctx),
            max_tool_rounds=max_tool_rounds,
            gate=gate,
            memory=memory,
            max_retries=max_retries,
        )

    def ask(self, question: str):
        """自然语言提问；返回 AgentRunResult（answer 是文本）。"""
        return self.run(question)
