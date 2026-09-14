"""故事引擎：编审回灌循环、交叉校验、质检官硬校验、章节规划师兜底、provider 工厂、会话锁。"""

from __future__ import annotations

import importlib
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "02_STORY_ENGINE", ROOT / "03_STRUCTURED_DATA"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import schemas as sch  # noqa: E402
from agents import (  # noqa: E402
    ChapterPlannerAgent,
    CharacterAgent,
    ChiefEditorAgent,
    ConversationLockTimeout,
    ConversationMemory,
    EpisodeAgent,
    FileConversationStore,
    InMemoryConversationStore,
    PromptAgent,
    QCOfficerAgent,
    ScreenplayAgent,
    StoryAgent,
    StoryboardAgent,
    build_memory,
    build_provider,
    resolve_provider_kind,
    rule_check,
)
from planners import EpisodePlanner, SeasonArcPlanner  # noqa: E402
from workflows.pipeline import AgentBundle, build_graph  # noqa: E402
from workflows.validation import DramaConsistencyError, reconcile_story_bible, validate_episode_assets  # noqa: E402

demo = importlib.import_module("02_STORY_ENGINE.demo")


def _bundle(provider, memory=None, editorial=True, planner="rule"):
    return AgentBundle(
        story_agent=StoryAgent(provider, memory=memory),
        character_agent=CharacterAgent(provider, memory=memory),
        episode_agent=EpisodeAgent(provider, memory=memory),
        screenplay_agent=ScreenplayAgent(provider, memory=memory),
        storyboard_agent=StoryboardAgent(provider, memory=memory),
        prompt_agent=PromptAgent(provider, memory=memory),
        season_planner=ChapterPlannerAgent(provider, memory=memory, num_episodes=12) if planner == "llm" else SeasonArcPlanner(num_episodes=12),
        episode_planner=EpisodePlanner(),
        qc_officer=QCOfficerAgent(provider, memory=memory) if editorial else None,
        chief_editor=ChiefEditorAgent(provider, memory=memory) if editorial else None,
        max_revision_rounds=2,
    )


def test_editorial_revise_loop_feeds_notes_back_to_screenplay():
    """总编审第一次 revise，第二次 approve：剧本应被重写为 version 2，且重写 prompt 里带修改意见。"""
    provider = demo.build_mock_provider()
    decisions = iter(["revise", "approve"])

    def editorial(_s, user):
        m = __import__("re").search(r"id 使用 '([^']+)'，episode_id 为 '([^']+)'", user)
        kind = next(decisions, "approve")
        d = sch.EditorialDecision(
            id=m.group(1),
            episode_id=m.group(2),
            decision=sch.EditorialDecisionType(kind),
            notes="开场钩子太慢" if kind == "revise" else "可以投产",
            required_changes=["把死亡画面提前到第 1 秒"] if kind == "revise" else [],
            commercial_score=60 if kind == "revise" else 85,
        )
        return d.model_dump_json()

    provider.fixtures["EditorialDecision"] = editorial
    graph = build_graph(_bundle(provider), num_characters=3, num_scenes=2, episode_numbers=[1])
    state = graph.invoke({"idea": "重生复仇", "target_duration_seconds": 300})
    rec = state["editorial_by_episode"]["ep_001"]
    assert rec["status"] == "approved" and rec["revision_rounds"] == 1
    assert state["scripts"][0].version == 2
    rewrite_calls = [c for c in provider.calls if "总编审/质检官要求修改" in c["user_prompt"]]
    assert rewrite_calls and "把死亡画面提前到第 1 秒" in rewrite_calls[0]["user_prompt"]
    assert "上一版剧本（version 1）" in rewrite_calls[0]["user_prompt"]


def test_editorial_reject_marks_episode_for_human_review_and_skips_storyboard():
    provider = demo.build_mock_provider()
    provider.fixtures["EditorialDecision"] = lambda _s, user: sch.EditorialDecision(
        id="ed_ep_001", episode_id="ep_001", decision=sch.EditorialDecisionType.REJECT, notes="影射真实企业", commercial_score=10
    ).model_dump_json()
    graph = build_graph(_bundle(provider), num_characters=3, num_scenes=2, episode_numbers=[1])
    state = graph.invoke({"idea": "重生复仇", "target_duration_seconds": 300})
    assert state["episodes_needing_review"] == ["ep_001"]
    assert state["editorial_by_episode"]["ep_001"]["status"] == "rejected"
    assert state["shots"] == [] and state["image_prompts"] == []


def test_blocker_from_qc_forces_revise_even_if_editor_approves():
    provider = demo.build_mock_provider()
    provider.fixtures["QCVerdict"] = lambda _s, user: sch.QCVerdict(
        id="qc_ep_001",
        episode_id="ep_001",
        passed=False,
        score=30,
        issues=[sch.QCIssue(code="compliance", severity=sch.IssueSeverity.BLOCKER, description="出现真实品牌名")],
    ).model_dump_json()
    editor = ChiefEditorAgent(provider)
    bible = sch.StoryBible.model_validate_json(demo._story_bible_fixture())
    arc = SeasonArcPlanner().plan(bible)
    ep = sch.Episode.model_validate_json(demo._episode_fixture("", "这是第 1 集"))
    script = sch.Script.model_validate_json(demo._script_fixture("", "episode_id='ep_001'"))
    verdict = sch.QCVerdict.model_validate_json(provider.fixtures["QCVerdict"]("", ""))
    decision = editor.decide(bible, arc, ep, script, verdict)
    assert decision.decision == sch.EditorialDecisionType.REVISE
    assert "出现真实品牌名" in decision.required_changes


def test_multi_episode_recap_is_passed_to_episode_agent():
    provider = demo.build_mock_provider()
    graph = build_graph(_bundle(provider, editorial=False), num_characters=3, num_scenes=1, episode_numbers=[1, 2], prompt_mode="template")
    state = graph.invoke({"idea": "重生复仇", "target_duration_seconds": 300})
    assert [e.episode_number for e in state["episodes"]] == [1, 2]
    ep2_prompt = next(c["user_prompt"] for c in provider.calls if "这是第 2 集" in c["user_prompt"])
    assert "前情提要" in ep2_prompt and "重生夜" in ep2_prompt
    assert len(state["image_prompts"]) == 4 and state["image_prompts"][0].reference_character_ids == ["char_su_wanwan"]


def test_rule_check_catches_dangling_references():
    ep = sch.Episode.model_validate_json(demo._episode_fixture("", "这是第 1 集"))
    chars = [sch.Character.model_validate_json(demo._character_fixture("", h)) for h in ("女主角", "男主角", "反派")]
    scene = sch.Scene.model_validate_json(demo._scene_fixture("", "episode_id='ep_001' scene_number=1"))
    shots = sch.ShotsFile.model_validate_json(demo._shots_fixture("", "scene_id='scene_001_01'")).shots
    bad_shot = shots[0].model_copy(update={"character": ["char_ghost"], "duration": 20.0})
    issues = rule_check(ep, [scene], [bad_shot, shots[1]], chars)
    codes = {i.code for i in issues}
    assert "reference_mismatch" in codes and "pacing" in codes
    with pytest.raises(DramaConsistencyError):
        validate_episode_assets(ep, [scene], [bad_shot], chars)


def test_reconcile_story_bible_uses_generated_character_ids():
    bible = sch.StoryBible.model_validate_json(demo._story_bible_fixture()).model_copy(update={"character_ids": ["planned_only"]})
    chars = [sch.Character.model_validate_json(demo._character_fixture("", h)) for h in ("女主角", "男主角", "反派")]
    fixed = reconcile_story_bible(bible, chars)
    assert fixed.character_ids == [c.id for c in chars]
    dupe = chars + [chars[0]]
    with pytest.raises(DramaConsistencyError):
        reconcile_story_bible(bible, dupe)


def test_chapter_planner_falls_back_when_llm_misassigns_episode_ids():
    provider = demo.build_mock_provider()
    bible = sch.StoryBible.model_validate_json(demo._story_bible_fixture())
    good = SeasonArcPlanner(num_episodes=12).plan(bible)
    broken = good.model_copy(update={"beats": [good.beats[0].model_copy(update={"episode_ids": ["ep_001", "ep_001"]})], "episode_ids": ["ep_001"]})
    provider.fixtures["SeasonArc"] = broken.model_dump_json()
    arc = ChapterPlannerAgent(provider, num_episodes=12, num_acts=3).plan(bible)
    assert arc.episode_ids == [f"ep_{i:03d}" for i in range(1, 13)]
    assert [e for b in arc.beats for e in b.episode_ids] == arc.episode_ids and len(arc.beats) == 3


def test_provider_factory_resolution_and_cache():
    assert resolve_provider_kind("claude-sonnet-4-5") == "anthropic"
    assert resolve_provider_kind("gpt-4o-mini") == "openai"
    assert resolve_provider_kind("Qwen/Qwen2.5-7B-Instruct") == "local"
    a = build_provider("local", "HuggingFaceTB/SmolLM2-360M-Instruct")
    b = build_provider("local", "HuggingFaceTB/SmolLM2-360M-Instruct")
    assert a is b and a.context_window == 8_192  # 本地模型不加载权重也能构造


def test_usage_sink_receives_every_call():
    provider = demo.build_mock_provider()
    agent = StoryAgent(provider)
    seen = []
    agent.usage_sink = lambda name, usage: seen.append((name, usage.input_tokens > 0))
    agent.generate_story_bible("重生复仇")
    assert seen == [("StoryAgent", True)] and agent.usage_log[0].provider == "mock"


# ---- 会话锁 ------------------------------------------------------------------------------


def test_file_store_lock_serializes_concurrent_writers(tmp_path):
    store = FileConversationStore(tmp_path)
    errors = []

    def writer(idx):
        try:
            mem = ConversationMemory("shared", store, max_context_tokens=100_000, reserve_tokens=0)
            with mem.transaction():
                mem.add_turn(f"q{idx}", f"a{idx}")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    final = ConversationMemory("shared", store)
    assert final.turn_count == 8  # 没有锁的话会互相覆盖，轮次少于 8


def test_lock_timeout_raises(tmp_path):
    store = InMemoryConversationStore()
    a = ConversationMemory("ctx", store, lock_timeout=0.2)
    b = ConversationMemory("ctx", store, lock_timeout=0.2)
    with a.transaction():
        with pytest.raises(ConversationLockTimeout):
            with b.transaction():
                pass


def test_agent_generate_holds_lock_and_reloads_history(tmp_path):
    store = FileConversationStore(tmp_path)
    provider = demo.build_mock_provider()
    mem_a = build_memory(provider, "ctx", store)
    mem_b = build_memory(provider, "ctx", store)
    StoryAgent(provider, memory=mem_a).generate_story_bible("A")
    # 第二个 memory 实例在事务里会重新加载，看到第一个实例写的轮次
    StoryAgent(provider, memory=mem_b).generate_story_bible("B")
    assert provider.calls[-1]["history_len"] == 2
    assert json.loads((tmp_path / "ctx.json").read_text())["messages"].__len__() == 4


@pytest.mark.skipif(not __import__("shutil").which("redis-server"), reason="需要本机 redis-server")
def test_redis_store_lock_roundtrip(tmp_path):
    import subprocess
    import time

    from agents import RedisConversationStore

    proc = subprocess.Popen(["redis-server", "--port", "6398", "--save", "", "--appendonly", "no"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.5)
        store = RedisConversationStore("redis://127.0.0.1:6398/2", ttl_seconds=60)
        mem = ConversationMemory("r", store)
        mem.clear()
        with mem.transaction():
            mem.add_turn("你好", "你好")
        assert ConversationMemory("r", store).turn_count == 1
        other = ConversationMemory("r", store, lock_timeout=0.2)
        with mem.transaction():
            with pytest.raises(ConversationLockTimeout):
                with other.transaction():
                    pass
    finally:
        proc.terminate()
        proc.wait(timeout=5)
