"""带工具调用循环的 Agent（规划 -> 执行 -> 验证 -> 修正 -> 再执行）。

``BaseAgent.generate()`` 是"一问一答出一个结构化对象"；``ToolAgent.run()`` 则是《Agent 搭建指南》
里"步骤 5：多步骤工作流"的实现：模型自己决定先查什么、再算什么、最后怎么答，每一步要么
调用一个工具（``action="tool"``），要么给出最终答案（``action="final"``）。

协议（对所有 LLMProvider 通用，靠 prompt + JSON 解析而不是各家 SDK 的 function calling）：

    模型每轮只输出一个 JSON 对象，二选一：
      {"thought": "...", "action": "tool",  "tool": "<工具名>", "args": {...}}
      {"thought": "...", "action": "final", "answer": "<文本或 JSON 对象>"}
    工具执行结果会作为下一条 user 消息回给模型（"观察"），模型据此决定下一步。

生产上最容易踩的坑在这里都有对应处理：
- 坑 1 无限循环：``max_tool_rounds``（默认 5，而不是很多框架的 10）硬上限；另外连续 ``repeat_limit``
  次调用同一工具 + 同样参数会被识别为"原地打转"，直接要求模型给最终答案。
- 坑 2 上下文爆炸：一次 run 内部的中间轮次不进长期记忆，只有 (用户问题, 最终答案) 这一对才
  ``memory.add_turn()``；带 memory 时的历史窗口仍由 ConversationMemory 按上下文上限裁剪/压缩。
- 坑 3 权限过大：需要确认的工具由 ``gate``（ConfirmationGate）把关，默认拒绝。
- 监控：每次 LLM 调用走 ``usage_sink``（token），每次工具调用走 ``tool_call_sink``（成功率），
  每次 run 走 ``run_sink``（轮数、停止原因）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .base import AgentGenerationError, BaseAgent, LLMProvider, LLMUsage, _enum_cheatsheet, _extract_json
from .memory import ConversationMemory, Message
from .tools import ConfirmationGate, ToolCallSink, ToolRegistry, ToolResult

T = TypeVar("T", bound=BaseModel)


class AgentLoopError(AgentGenerationError):
    """工具循环没能在限制内收敛：达到轮次上限或反复重复同一调用后模型仍不给最终答案。"""


@dataclass
class AgentRunResult:
    """一次 ``ToolAgent.run()`` 的完整结果：答案 + 每一步工具调用的记录 + 用量。"""

    answer: Any  # 传了 schema 就是校验后的 pydantic 对象，否则是模型给的文本/JSON
    steps: list[ToolResult] = field(default_factory=list)
    tool_rounds: int = 0
    stopped_reason: str = "final"  # final | limit_reached | repeat_detected
    usage: list[LLMUsage] = field(default_factory=list)
    transcript: list[Message] = field(default_factory=list)  # 本次 run 内部的完整消息序列，便于调试/回放

    @property
    def total_tokens(self) -> int:
        """本次 run 所有 LLM 调用的 token 总和。"""
        return sum(u.total_tokens for u in self.usage)

    @property
    def tool_success_rate(self) -> float | None:
        """本次 run 的工具调用成功率；没调过工具返回 None。"""
        if not self.steps:
            return None
        return sum(1 for s in self.steps if s.ok) / len(self.steps)


RunSink = Callable[[str, AgentRunResult], None]

_PROTOCOL_TEMPLATE = (
    "\n\n## 可用工具\n"
    "你可以调用下面这些工具来获取信息或执行操作（JSON 列表，parameters 是每个工具的参数 JSON Schema）：\n"
    "{tools}\n\n"
    "## 输出协议\n"
    "每一轮你只能输出一个 JSON 对象，不要输出任何解释文字或 markdown 代码块标记。二选一：\n"
    '1. 调用工具：{{"thought": "为什么要调这个工具", "action": "tool", "tool": "<工具名>", "args": {{...}}}}\n'
    '2. 最终回答：{{"thought": "推理过程简述", "action": "final", "answer": {answer_hint}}}\n'
    "规则：\n"
    "- 最多调用 {max_rounds} 次工具，工具结果会以下一条消息回给你；先规划再调用，能直接回答就不要调工具。\n"
    "- 不要用同样的参数重复调用同一个工具；工具报错时先读错误信息，修正参数或换工具。\n"
    "- requires_confirmation 为 true 的工具需要人工确认，可能被拒绝；被拒绝就换别的方式完成任务或如实说明。"
)


class ToolAgent(BaseAgent):
    """会调用工具的 Agent。子类通常只需要给系统提示词和工具注册表。"""

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str,
        registry: ToolRegistry,
        *,
        max_tool_rounds: int = 5,
        repeat_limit: int = 2,
        gate: ConfirmationGate | None = None,
        max_retries: int = 3,
        memory: ConversationMemory | None = None,
    ) -> None:
        """
        ``max_tool_rounds``：一次 run 最多调用几次工具（坑 1 的硬上限）。
        ``repeat_limit``：连续几次相同 (工具, 参数) 视为原地打转。
        ``gate``：需要确认的工具走哪个确认门；None 表示默认拒绝。
        ``max_retries``：模型输出不是合法 JSON 时最多纠正几次（沿用 BaseAgent 语义）。
        """
        super().__init__(provider, system_prompt, max_retries=max_retries, memory=memory)
        self.registry = registry
        self.max_tool_rounds = max_tool_rounds
        self.repeat_limit = max(1, repeat_limit)
        self.gate = gate
        self.tool_call_sink: ToolCallSink | None = None
        self.run_sink: RunSink | None = None

    # ---- prompt 组装 ---------------------------------------------------------------------

    def _full_system_prompt(self, schema: type[BaseModel] | None) -> str:
        """业务系统提示词 + 工具清单 + 输出协议（+ 结构化答案的字段/枚举要求）。"""
        if schema is None:
            answer_hint = '"给用户的最终回答文本"'
        else:
            answer_hint = f"符合 {schema.__name__} 字段结构的 JSON 对象{_enum_cheatsheet(schema)}"
        return self.system_prompt + _PROTOCOL_TEMPLATE.format(
            tools=self.registry.describe_for_prompt(), answer_hint=answer_hint, max_rounds=self.max_tool_rounds
        )

    # ---- 解析模型输出 ---------------------------------------------------------------------

    @staticmethod
    def _parse_step(raw: str) -> dict[str, Any]:
        """把模型输出解析成 {"action": ..., ...}；格式不对抛 ValueError，由循环回灌给模型纠正。"""
        try:
            step = json.loads(_extract_json(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"不是合法 JSON：{exc}") from exc
        if not isinstance(step, dict):
            raise ValueError("顶层必须是 JSON 对象")
        action = step.get("action")
        if action == "tool":
            if not isinstance(step.get("tool"), str) or not step["tool"]:
                raise ValueError('action="tool" 时必须给出字符串 tool 字段')
            if step.get("args") is not None and not isinstance(step["args"], dict):
                raise ValueError("args 必须是 JSON 对象")
            return step
        if action == "final":
            if "answer" not in step:
                raise ValueError('action="final" 时必须给出 answer 字段')
            return step
        raise ValueError('action 只能是 "tool" 或 "final"')

    @staticmethod
    def _validate_answer(answer: Any, schema: type[T] | None) -> Any:
        """有 schema 时把 answer 校验成 pydantic 对象；失败抛 ValueError 让模型改。"""
        if schema is None:
            return answer
        try:
            if isinstance(answer, str):
                return schema.model_validate_json(_extract_json(answer))
            return schema.model_validate(answer)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise ValueError(f"answer 不符合 {schema.__name__}：{exc}") from exc

    # ---- 主循环 ---------------------------------------------------------------------------

    def run(self, user_prompt: str, schema: type[T] | None = None) -> AgentRunResult:
        """执行一次多步任务。带 memory 时整个 run 持有会话锁，只把 (问题, 最终答案) 记进历史。"""
        if self.memory is not None:
            with self.memory.transaction():
                history = self.memory.window(self._full_system_prompt(schema), user_prompt)
                result, final_raw = self._loop(user_prompt, schema, history)
                self.memory.add_turn(user_prompt, final_raw)
        else:
            result, final_raw = self._loop(user_prompt, schema, [])
        if self.run_sink is not None:
            try:
                self.run_sink(self.__class__.__name__, result)
            except Exception:  # 指标失败不影响业务
                pass
        return result

    def _loop(self, user_prompt: str, schema: type[T] | None, history: list[Message]) -> tuple[AgentRunResult, str]:
        """真正的 规划 -> 调工具 -> 观察 -> 再规划 循环；返回 (结果, 最终答案的原始文本)。"""
        system = self._full_system_prompt(schema)
        result = AgentRunResult(answer=None)
        transcript: list[Message] = [{"role": "user", "content": user_prompt}]
        pending_user = user_prompt  # 本轮要发给模型的 user 消息（首轮是问题，之后是工具观察结果）
        parse_failures = 0
        recent_calls: list[str] = []  # 最近几次 (工具, 参数) 的指纹，用来识别原地打转
        forced_final = False  # 已经要求模型"别再调工具、直接作答"

        while True:
            # 传给 provider 的 history = 长期记忆窗口 + 本次 run 已经发生的轮次（不含 pending_user）
            raw = self.provider.complete(system, pending_user, history + transcript[:-1])
            self._record_usage()
            if self.provider.last_usage is not None:
                result.usage.append(self.provider.last_usage)
            transcript.append({"role": "assistant", "content": raw})

            try:
                step = self._parse_step(raw)
                if step["action"] == "final":
                    answer = self._validate_answer(step["answer"], schema)
            except ValueError as exc:
                parse_failures += 1
                if parse_failures > self.max_retries:
                    raise AgentGenerationError(f"{self.__class__.__name__} 连续 {parse_failures} 次输出不符合协议：{exc}") from exc
                pending_user = f"你上一条输出不符合输出协议：{exc}\n请只输出一个合法的 JSON 对象（action 为 tool 或 final）。"
                transcript.append({"role": "user", "content": pending_user})
                continue

            if step["action"] == "final":
                result.answer = answer
                result.transcript = transcript
                return result, raw

            # ---- action == "tool" ----
            if forced_final:
                # 已经明确要求作答，模型仍坚持调工具：不再纵容，抛错交给上层（任务层可重试/降级）
                raise AgentLoopError(f"{self.__class__.__name__} 在被要求直接作答后仍请求调用工具 {step['tool']}（{result.stopped_reason}）")

            fingerprint = json.dumps({"tool": step["tool"], "args": step.get("args") or {}}, ensure_ascii=False, sort_keys=True)
            recent_calls.append(fingerprint)
            repeating = len(recent_calls) >= self.repeat_limit and len(set(recent_calls[-self.repeat_limit :])) == 1

            if result.tool_rounds >= self.max_tool_rounds:
                result.stopped_reason = "limit_reached"
                forced_final = True
                pending_user = f"已达到本次任务的工具调用上限（{self.max_tool_rounds} 次），不能再调用工具。请基于已有信息直接给出最终答案（action=final）。"
                transcript.append({"role": "user", "content": pending_user})
                continue
            if repeating:
                result.stopped_reason = "repeat_detected"
                forced_final = True
                pending_user = f"你已经连续 {self.repeat_limit} 次用同样的参数调用 {step['tool']}，这不会得到新信息。请直接给出最终答案（action=final）。"
                transcript.append({"role": "user", "content": pending_user})
                continue

            tool_result = self.registry.call(step["tool"], step.get("args"), gate=self.gate, sink=self.tool_call_sink, attempt=result.tool_rounds + 1)
            result.steps.append(tool_result)
            result.tool_rounds += 1
            pending_user = f"工具 {step['tool']} 的结果：\n{tool_result.as_observation()}"
            transcript.append({"role": "user", "content": pending_user})
