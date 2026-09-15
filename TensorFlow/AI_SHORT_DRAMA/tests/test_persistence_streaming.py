"""LangGraph 持久化（checkpointer + thread_id）与流式输出的测试。

覆盖：
- ``StoreCheckpointSaver``：检查点写进 ConversationStore（SQLite/文件/内存）、跨实例可读、thread 隔离、锁；
- 续跑语义：跑完的 thread 再调直接跳过；中途崩溃的 thread 只重跑未完成的节点；
- 反序列化白名单：state 里的 pydantic 对象必须原样还原（不能退化成 dict）；
- provider 级流式（mock 分块、假流式默认实现、降级 provider 的流式切换规则）；
- ``ToolAgent.run_stream``：token/step/tool/notice/result 事件序列，以及和 ``run()`` 的一致性；
- 13_INFRA：故事任务写进度、``GET /pipelines/{id}/checkpoint``、两个 SSE 接口。
全程 mock，不需要 API key、模型权重或外部服务。
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "02_STORY_ENGINE", ROOT / "03_STRUCTURED_DATA"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import schemas as sch  # noqa: E402
from agents import (  # noqa: E402
    FallbackProvider,
    FileConversationStore,
    InMemoryConversationStore,
    LLMProvider,
    SQLiteConversationStore,
    ToolAgent,
    ToolRegistry,
    TransientLLMError,
)
from planners import EpisodePlanner, SeasonArcPlanner  # noqa: E402
from workflows import StoreCheckpointSaver, build_checkpointer, build_graph, graph_progress, run_pipeline, stream_pipeline, thread_config  # noqa: E402
from workflows.pipeline import AgentBundle  # noqa: E402

demo = importlib.import_module("02_STORY_ENGINE.demo")


def _bundle(provider) -> AgentBundle:
    """一套最小的 Agent 组合：模板提示词 + 规则规划 + 不开编审，跑得快又足够覆盖六个节点。"""
    return AgentBundle(
        story_agent=demo.StoryAgent(provider),
        character_agent=demo.CharacterAgent(provider),
        episode_agent=demo.EpisodeAgent(provider),
        screenplay_agent=demo.ScreenplayAgent(provider),
        storyboard_agent=demo.StoryboardAgent(provider),
        prompt_agent=None,
        season_planner=SeasonArcPlanner(num_episodes=12),
        episode_planner=EpisodePlanner(),
    )


def _graph(provider, checkpointer=None):
    """编译一张带（或不带）checkpointer 的故事图。"""
    return build_graph(_bundle(provider), num_characters=3, num_scenes=1, episode_numbers=[1], prompt_mode="template", checkpointer=checkpointer)


# ---- checkpointer 存储 ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_store", [lambda p: InMemoryConversationStore(), lambda p: FileConversationStore(p), lambda p: SQLiteConversationStore(p / "c.db")]
)
def test_checkpoint_roundtrip_across_store_backends(make_store, tmp_path):
    """三种存储后端都能存取检查点，且新建一个 saver 指向同一存储还能读到（跨进程等价）。"""
    store = make_store(tmp_path)
    provider = demo.build_mock_provider()
    graph = _graph(provider, build_checkpointer(store))
    state = run_pipeline(graph, {"idea": "重生复仇"}, "run:x")
    assert len(state["shots"]) == 2

    fresh = _graph(demo.build_mock_provider(), build_checkpointer(store))
    progress = graph_progress(fresh, "run:x")
    assert progress["finished"] and progress["completed"] == ["story_bible", "season_arc", "characters", "episodes", "storyboard", "prompts"]
    assert progress["pending"] == []


def test_checkpoint_restores_pydantic_objects_not_dicts(tmp_path):
    """state 里的 03 schema 对象必须原样还原。

    锁住一个真实踩过的坑：不给 LangGraph 的反序列化白名单登记 schemas 里的类时，
    pydantic 对象会静默降级成 dict，续跑时 ``script.content`` 直接 AttributeError。
    """
    store = SQLiteConversationStore(tmp_path / "c.db")
    run_pipeline(_graph(demo.build_mock_provider(), build_checkpointer(store)), {"idea": "重生复仇"}, "run:t")
    fresh = _graph(demo.build_mock_provider(), build_checkpointer(store))
    values = fresh.get_state(thread_config("run:t")).values
    assert isinstance(values["script"], sch.Script) and values["script"].content
    assert isinstance(values["story_bible"], sch.StoryBible)
    assert all(isinstance(s, sch.Shot) for s in values["shots"])
    assert isinstance(values["characters"][0].role, sch.CharacterRole)


def test_threads_are_isolated(tmp_path):
    """不同 thread_id 之间完全隔离：一个会话的 state 不会漏到另一个。"""
    store = SQLiteConversationStore(tmp_path / "c.db")
    graph = _graph(demo.build_mock_provider(), build_checkpointer(store))
    run_pipeline(graph, {"idea": "甲故事"}, "run:1")
    run_pipeline(graph, {"idea": "乙故事"}, "run:2")
    assert graph.get_state(thread_config("run:1")).values["idea"] == "甲故事"
    assert graph.get_state(thread_config("run:2")).values["idea"] == "乙故事"
    assert graph_progress(graph, "run:3") == {"completed": [], "pending": [], "finished": False, "steps": 0}


def test_saver_requires_thread_id_and_supports_delete(tmp_path):
    """没传 thread_id 直接报错（避免多个会话静默共用一份状态）；delete_thread 能清干净。"""
    saver = StoreCheckpointSaver(SQLiteConversationStore(tmp_path / "c.db"))
    with pytest.raises(ValueError):
        saver.get_tuple({"configurable": {}})
    graph = _graph(demo.build_mock_provider(), saver)
    run_pipeline(graph, {"idea": "重生复仇"}, "run:d")
    assert saver.get_tuple(thread_config("run:d")) is not None
    assert len(list(saver.list(thread_config("run:d")))) > 1  # 每个节点一个检查点
    saver.delete_thread("run:d")
    assert saver.get_tuple(thread_config("run:d")) is None


def test_checkpoint_list_respects_limit_and_before(tmp_path):
    """list() 倒序返回，limit 截断、before 排除较新的检查点（回溯调试用）。"""
    saver = StoreCheckpointSaver(SQLiteConversationStore(tmp_path / "c.db"))
    run_pipeline(_graph(demo.build_mock_provider(), saver), {"idea": "重生复仇"}, "run:l")
    everything = list(saver.list(thread_config("run:l")))
    assert len(saver_limited := list(saver.list(thread_config("run:l"), limit=2))) == 2
    assert saver_limited == everything[:2]
    middle = everything[2].config
    assert all(
        item.config["configurable"]["checkpoint_id"] < middle["configurable"]["checkpoint_id"] for item in saver.list(thread_config("run:l"), before=middle)
    )


def test_max_checkpoints_trims_oldest(tmp_path):
    """超过 max_checkpoints 就丢最旧的，记录不会无限增长。"""
    saver = StoreCheckpointSaver(SQLiteConversationStore(tmp_path / "c.db"), max_checkpoints=3)
    run_pipeline(_graph(demo.build_mock_provider(), saver), {"idea": "重生复仇"}, "run:trim")
    assert len(list(saver.list(thread_config("run:trim")))) == 3


# ---- 续跑语义 -------------------------------------------------------------------------------


def test_finished_thread_is_skipped_without_new_llm_calls(tmp_path):
    """已经跑完的 thread 再被调用（Celery 重试）时直接返回存下来的 state，不再花一次 LLM 调用。"""
    provider = demo.build_mock_provider()
    graph = _graph(provider, build_checkpointer(SQLiteConversationStore(tmp_path / "c.db")))
    run_pipeline(graph, {"idea": "重生复仇"}, "run:s")
    calls_before = len(provider.calls)
    events: list[dict] = []
    state = run_pipeline(graph, {"idea": "重生复仇"}, "run:s", on_progress=events.append)
    assert len(provider.calls) == calls_before  # 零新增调用
    assert len(state["shots"]) == 2
    assert [e["event"] for e in events] == ["done"] and events[0]["skipped"] is True


def test_crashed_thread_resumes_only_remaining_nodes(tmp_path):
    """节点中途抛错后，续跑只重算未完成的节点——这是 checkpointer 真正省钱的地方。"""
    provider = demo.build_mock_provider()
    calls = {"n": 0}

    def flaky_scene(system, user):
        """第一次调用抛错，模拟 storyboard 节点崩在半路。"""
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("模拟 storyboard 崩溃")
        return demo._scene_fixture(system, user)

    provider.fixtures["Scene"] = flaky_scene
    graph = _graph(provider, build_checkpointer(SQLiteConversationStore(tmp_path / "c.db")))
    with pytest.raises(RuntimeError, match="模拟 storyboard 崩溃"):
        run_pipeline(graph, {"idea": "重生复仇"}, "run:c")

    crashed = graph_progress(graph, "run:c")
    assert crashed["completed"] == ["story_bible", "season_arc", "characters", "episodes"]
    assert crashed["pending"] == ["storyboard"] and not crashed["finished"]

    before = len(provider.calls)
    events: list[dict] = []
    state = run_pipeline(graph, {"idea": "重生复仇"}, "run:c", on_progress=events.append)
    resumed_calls = len(provider.calls) - before
    # 从头跑需要 9 次（bible 1 + arc 0 + 角色 3 + 集 1 + 剧本 1 + 场次 1 + 镜头 1 + 对白 1）；
    # 续跑只剩 storyboard + prompts，调用次数必须明显更少
    assert 0 < resumed_calls < 9
    assert [e["event"] for e in events][:2] == ["resumed", "node"]
    assert events[0]["pending"] == ["storyboard"]
    assert [e["node"] for e in events if e["event"] == "node"] == ["storyboard", "prompts"]
    assert len(state["shots"]) == 2 and graph_progress(graph, "run:c")["finished"]


def test_stream_pipeline_reports_每个节点(tmp_path):
    """stream_pipeline 按节点吐进度事件，字段名和列表长度都带上，最后一条是 done。"""
    graph = _graph(demo.build_mock_provider(), build_checkpointer(InMemoryConversationStore()))
    events = list(stream_pipeline(graph, {"idea": "重生复仇"}, "run:p"))
    nodes = [e["node"] for e in events if e["event"] == "node"]
    assert nodes == ["story_bible", "season_arc", "characters", "episodes", "storyboard", "prompts"]
    assert events[-1]["event"] == "done" and events[-1]["state"]["idea"] == "重生复仇"
    characters = next(e for e in events if e["node"] == "characters")
    assert characters["counts"]["characters"] == 3 and "story_bible" in characters["keys"]


def test_pipeline_without_checkpointer_still_works():
    """不传 checkpointer 时退回一次性 invoke（默认行为不变）。"""
    graph = _graph(demo.build_mock_provider())
    assert graph.checkpointer is None
    state = run_pipeline(graph, {"idea": "重生复仇"}, "run:none")
    assert len(state["shots"]) == 2
    assert graph_progress(graph, "run:none")["completed"] == []


# ---- provider 流式 ---------------------------------------------------------------------------


class _Echo(LLMProvider):
    """只实现 complete() 的最小 provider，用来验证默认"假流式"。"""

    context_window = 10_000
    max_tokens = 100
    provider_name = "echo"
    model = "echo"

    def complete(self, system_prompt, user_prompt, history=None):
        """固定回复。"""
        self.last_usage = self._estimate_usage(system_prompt, user_prompt, history, "回复")
        return "回复"


def test_default_stream_is_single_chunk_fallback():
    """没覆盖 stream() 的 provider 也能被流式调用，只是一次吐完。"""
    assert list(_Echo().stream("s", "u")) == ["回复"]


def test_mock_provider_streams_in_chunks_and_records_usage():
    """mock 按 chunk_size 切块，拼起来等于 complete()，并且记了用量。"""
    provider = demo.build_mock_provider()
    chunks = list(provider.stream("s", "随便问"))
    assert len(chunks) > 1
    assert "".join(chunks) == provider.complete("s", "随便问")
    assert provider.last_usage.total_tokens > 0


def test_fallback_streams_and_only_switches_before_first_token():
    """降级 provider 的流式：没吐内容时可以切备用；已经吐了内容再出错必须原样抛，不能拼接两个模型的输出。"""

    class Dead(LLMProvider):
        """一开口就报错。"""

        context_window = 8_000
        max_tokens = 50
        model = "dead"
        provider_name = "dead"

        def complete(self, system_prompt, user_prompt, history=None):
            """不会被调到。"""
            raise TransientLLMError("429")

        def stream(self, system_prompt, user_prompt, history=None):
            """直接抛，不吐任何增量。"""
            raise TransientLLMError("429")
            yield  # pragma: no cover - 让它成为生成器

    class HalfWay(LLMProvider):
        """吐一块之后才报错。"""

        context_window = 8_000
        max_tokens = 50
        model = "half"
        provider_name = "half"

        def complete(self, system_prompt, user_prompt, history=None):
            """不会被调到。"""
            raise TransientLLMError("boom")

        def stream(self, system_prompt, user_prompt, history=None):
            """先给一块再炸。"""
            yield "前半句"
            raise TransientLLMError("中途断了")

    good = demo.build_mock_provider()
    events: list[tuple] = []
    fb = FallbackProvider([Dead(), good], on_fallback=lambda a, b, e: events.append((a, b)))
    assert "".join(fb.stream("s", "随便问")) == good.complete("s", "随便问")
    assert events == [("dead", "mock")] and fb.model == "mock"

    fb2 = FallbackProvider([HalfWay(), demo.build_mock_provider()])
    with pytest.raises(TransientLLMError, match="中途断了"):
        list(fb2.stream("s", "随便问"))


# ---- ToolAgent.run_stream --------------------------------------------------------------------


class _Scripted(LLMProvider):
    """按脚本逐块吐字，用来精确断言事件序列。"""

    context_window = 50_000
    max_tokens = 200
    provider_name = "scripted"
    model = "scripted"

    def __init__(self, replies: list[str]) -> None:
        """``replies``：每次调用返回的完整文本。"""
        self.replies = list(replies)
        self.calls: list[str] = []

    def complete(self, system_prompt, user_prompt, history=None):
        """弹出下一条。"""
        self.calls.append(user_prompt)
        text = self.replies.pop(0)
        self.last_usage = self._estimate_usage(system_prompt, user_prompt, history, text)
        return text

    def stream(self, system_prompt, user_prompt, history=None):
        """把它切成三块，制造多条 token 事件。"""
        text = self.complete(system_prompt, user_prompt, history)
        size = max(1, len(text) // 3)
        for i in range(0, len(text), size):
            yield text[i : i + size]


@pytest.fixture
def toolbox() -> ToolRegistry:
    """一个无副作用工具 + 一个需确认工具。"""
    reg = ToolRegistry()

    @reg.tool
    def add(a: int, b: int = 1) -> int:
        """两数相加。"""
        return a + b

    @reg.tool(requires_confirmation=True)
    def wipe(target: str) -> str:
        """删除目标。"""
        return f"wiped {target}"

    return reg


def _tool_msg(name: str, **args) -> str:
    """一条"调用工具"的协议消息。"""
    return json.dumps({"thought": "t", "action": "tool", "tool": name, "args": args}, ensure_ascii=False)


def _final_msg(answer) -> str:
    """一条"最终回答"的协议消息。"""
    return json.dumps({"thought": "t", "action": "final", "answer": answer}, ensure_ascii=False)


def test_run_stream_event_sequence(toolbox):
    """事件顺序：token… -> step -> tool -> token… -> result；result 带答案、轮数和用量。"""
    provider = _Scripted([_tool_msg("add", a=2, b=3), _final_msg("等于 5")])
    events = list(ToolAgent(provider, "s", toolbox).run_stream("2+3=?"))
    kinds = [e["event"] for e in events]
    assert kinds.count("token") >= 4 and kinds.count("step") == 1 and kinds.count("tool") == 1
    assert kinds[-1] == "result" and kinds.index("step") < kinds.index("tool")
    step = next(e for e in events if e["event"] == "step")
    tool = next(e for e in events if e["event"] == "tool")
    assert step["tool"] == "add" and step["args"] == {"a": 2, "b": 3}
    assert tool["status"] == "success" and tool["output"] == "5" and tool["duration_seconds"] >= 0
    result = events[-1]
    assert result["answer"] == "等于 5" and result["tool_rounds"] == 1 and result["stopped_reason"] == "final"
    assert result["usage"]["llm_calls"] == 2 and "_result" not in result


def test_run_stream_reports_denied_tool_and_notices(toolbox):
    """确认门拒绝会作为 tool 事件的 denied 状态出现；轮次上限产生 notice 事件。"""
    provider = _Scripted([_tool_msg("wipe", target="db"), _final_msg("未获批准")])
    events = list(ToolAgent(provider, "s", toolbox).run_stream("清库"))
    assert next(e for e in events if e["event"] == "tool")["status"] == "denied"

    provider = _Scripted([_tool_msg("add", a=1), _tool_msg("add", a=2), _final_msg("到顶了")])
    events = list(ToolAgent(provider, "s", toolbox, max_tool_rounds=1).run_stream("q"))
    notice = next(e for e in events if e["event"] == "notice")
    assert notice["kind"] == "limit_reached"
    assert events[-1]["stopped_reason"] == "limit_reached" and events[-1]["answer"] == "到顶了"


def test_run_stream_matches_run_and_writes_memory_and_sinks(toolbox, tmp_path):
    """流式和非流式给出同样的答案与轮数；流式同样只把 (问题, 最终答案) 记进记忆，并上报两个 sink。"""
    replies = [_tool_msg("add", a=1, b=1), _final_msg("等于 2")]
    plain = ToolAgent(_Scripted(list(replies)), "s", toolbox).run("1+1=?")

    from agents import build_memory

    provider = _Scripted(list(replies))
    memory = build_memory(provider, "ctx", SQLiteConversationStore(tmp_path / "m.db"))
    agent = ToolAgent(provider, "s", toolbox, memory=memory)
    tool_calls, runs = [], []
    agent.tool_call_sink = tool_calls.append
    agent.run_sink = lambda name, result: runs.append((name, result.stopped_reason, result.tool_rounds))
    streamed = list(agent.run_stream("1+1=?"))[-1]

    assert streamed["answer"] == plain.answer == "等于 2"
    assert streamed["tool_rounds"] == plain.tool_rounds == 1
    assert [c.tool for c in tool_calls] == ["add"] and runs == [("ToolAgent", "final", 1)]
    assert memory.turn_count == 1 and memory.messages[0]["content"] == "1+1=?"


def test_drama_assistant_run_stream(tmp_path):
    """制片助理的流式：真实走一遍 list_episodes -> run_rule_check -> 汇报。"""
    from agents import DramaAssistantAgent, DramaContext

    provider = demo.build_mock_provider()
    state = _graph(provider).invoke({"idea": "重生复仇"})
    ctx = DramaContext.from_state(state)
    events = list(DramaAssistantAgent(demo.build_mock_provider(), ctx).run_stream("检查第 1 集"))
    assert [e["tool"] for e in events if e["event"] == "tool"] == ["list_episodes", "run_rule_check"]
    assert "无 blocker" in events[-1]["answer"] and events[-1]["tool_rounds"] == 2


# ---- 13_INFRA：进度落库 + checkpoint 接口 + 两个 SSE ------------------------------------------


@pytest.fixture(scope="module")
def client(api_headers):
    """带 API key 的 TestClient。"""
    main_mod = importlib.import_module("13_INFRA.api.main")
    with TestClient(main_mod.app, headers=api_headers) as c:
        yield c


@pytest.fixture(scope="module")
def seeded_run(client) -> str:
    """跑一条真实流水线（mock LLM），返回 run_id，供后面几个用例复用。"""
    project_id = client.post("/projects", json={"name": "persist", "description": "t"}).json()["project_id"]
    resp = client.post(
        "/pipelines/episodes",
        json={"project_id": project_id, "idea": "重生复仇", "num_characters": 3, "num_scenes": 1, "prompt_mode": "template", "planner": "rule"},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["run_id"]


def test_story_task_writes_node_progress(client, seeded_run):
    """故事任务按节点把进度写进 run.result，前端进度条和 SSE 都依赖它。"""
    result = client.get(f"/pipelines/{seeded_run}").json()["result"]
    story_task = importlib.import_module("13_INFRA.queue.story_task")
    assert result["progress_nodes"] == story_task.STORY_NODES
    assert result["progress_total"] == len(story_task.STORY_NODES)
    assert [p["node"] for p in result["progress"] if p["event"] == "node"] == story_task.STORY_NODES


def test_checkpoint_endpoint_shows_finished_thread(client, seeded_run):
    """/checkpoint 报告这次运行的检查点：六个节点全部完成、无待执行。"""
    body = client.get(f"/pipelines/{seeded_run}/checkpoint").json()
    assert body["enabled"] and body["finished"] and body["pending"] == []
    assert body["completed"] == body["nodes"] and body["steps"] == len(body["nodes"])
    assert client.get("/pipelines/nope/checkpoint").status_code == 404


def _sse_events(response) -> list[tuple[str, dict]]:
    """把 SSE 响应体切成 (事件名, 数据) 列表。"""
    out: list[tuple[str, dict]] = []
    for frame in response.text.split("\n\n"):
        name, data = None, ""
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[7:].strip()
            elif line.startswith("data: "):
                data += line[6:]
        if name and data:
            out.append((name, json.loads(data)))
    return out


def test_progress_sse_streams_snapshot_then_end(client, seeded_run):
    """进度 SSE：先推一条当前快照，run 已是终态（awaiting_review）就紧接着 end。"""
    with client.stream("GET", f"/pipelines/{seeded_run}/stream") as resp:
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
        resp.read()
        events = _sse_events(resp)
    names = [n for n, _ in events]
    assert names == ["progress", "end"]
    snapshot = events[0][1]
    assert snapshot["status"] == "awaiting_review" and snapshot["progress_nodes"]


def test_assistant_sse_streams_tokens_tools_and_result(client, seeded_run):
    """助理 SSE：start -> token/step/tool -> result -> end，token 事件拼起来是模型原始输出。"""
    with client.stream("POST", f"/pipelines/{seeded_run}/assistant/stream", json={"question": "检查第 1 集"}) as resp:
        assert resp.status_code == 200
        resp.read()
        events = _sse_events(resp)
    names = [n for n, _ in events]
    assert names[0] == "start" and names[-1] == "end"
    assert names.count("token") > 1 and [d["tool"] for n, d in events if n == "tool"] == ["list_episodes", "run_rule_check"]
    result = next(d for n, d in events if n == "result")
    assert "无 blocker" in result["answer"] and result["tool_rounds"] == 2
    assert result["usage"]["llm_calls"] == 3

    # 没有剧集产出的 run 拒绝流式提问
    project_id = client.post("/projects", json={"name": "empty"}).json()["project_id"]
    orchestration = importlib.import_module("13_INFRA.orchestration")
    empty_run = orchestration.create_run(project_id, {"idea": "还没跑"})
    assert client.post(f"/pipelines/{empty_run}/assistant/stream", json={"question": "在吗"}).status_code == 409


def test_console_page_uses_streaming_endpoints(client):
    """控制台页面确实接了这两个流式接口和检查点接口（防止前后端漂移）。"""
    html = client.get("/").text
    routes = set(client.get("/openapi.json").json()["paths"])
    for tpl, marker in (
        ("/pipelines/{run_id}/stream", "/stream"),
        ("/pipelines/{run_id}/assistant/stream", "/assistant/stream"),
        ("/pipelines/{run_id}/checkpoint", "/checkpoint"),
    ):
        assert tpl in routes and marker in html, tpl
    assert "text/event-stream" in html and "getReader" in html  # 用 fetch+ReadableStream 解析 SSE
