"""工具层：Agent 能做什么，取决于你给了它什么工具。

本模块对应《Agent 搭建指南》里的"步骤 3：注册工具能力"和"坑 3：工具调用权限过大"：

1. ``ToolRegistry.tool`` 装饰器 —— 把一个普通 Python 函数注册成 Agent 可调用的工具。
   参数 JSON Schema 直接从函数签名 + 类型标注生成（用 pydantic 动态建模），docstring 就是
   给模型看的工具说明，不需要手写一份和代码不同步的 schema。
2. ``ConfirmationGate`` —— 工具级确认门。标了 ``requires_confirmation=True`` 的工具（写库、
   发布、标记人工审核这类有副作用的操作）在执行前必须过门：默认 ``AutoDenyGate`` 一律拒绝，
   宁可让 Agent 少做一步，也不能让它在没人看着的时候乱改数据；本地 demo 用 ``ConsoleGate``
   在终端里问 y/n，生产用 ``AllowlistGate`` 把明确放行的工具名写进配置。
3. ``ToolResult`` + ``ToolCallSink`` —— 每次工具调用的成功/失败/被拒/参数错误和耗时都记下来，
   13_INFRA 把 sink 接到 Prometheus，就是"四大监控指标"里的"工具调用成功率"。

为什么不用各家 SDK 自带的 function calling：本项目的 ``LLMProvider`` 抽象只有一个
``complete(system, user, history) -> str``，Claude / OpenAI / 本地 transformers / mock 四种
后端共用；工具调用协议做在 prompt + JSON 解析这一层（见 ``tool_agent.py``），任何后端都能跑，
测试也不需要 API key。要接原生 tool_use 时只需换 ``ToolAgent`` 里"怎么把工具清单给模型"这一步。
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError, create_model


class ToolError(RuntimeError):
    """工具层的基类异常。"""


class ToolNotFoundError(ToolError):
    """模型点名调用了一个没注册的工具（通常是模型幻觉出来的名字）。"""


class ToolArgumentError(ToolError):
    """模型给的参数不符合工具签名（缺参数、类型不对、多了不认识的字段）。"""


class ToolDeniedError(ToolError):
    """需要确认的工具没有通过确认门。"""


class ToolExecutionError(ToolError):
    """工具函数自身抛了异常；原始异常放在 ``__cause__`` 里。"""


# --------------------------------------------------------------------------------------
# 工具定义
# --------------------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """一个已注册工具的完整描述：名字、说明、参数 schema、实现函数、是否需要确认。"""

    name: str
    description: str
    fn: Callable[..., Any]
    args_model: type[BaseModel]  # 由函数签名生成的 pydantic 模型，负责参数校验与类型转换
    requires_confirmation: bool = False

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """给模型看的 JSON Schema（去掉 pydantic 自带的 title 噪音，省 token）。"""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return schema

    def to_prompt_dict(self) -> dict[str, Any]:
        """塞进系统提示词的紧凑形式。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass
class ToolResult:
    """一次工具调用的结果记录；不管成功失败都会产生一条，供 Agent 回灌给模型和上报指标。"""

    tool: str
    args: dict[str, Any]
    status: str  # success | error | denied | invalid_args | not_found
    output: str = ""  # 成功时是工具返回值的文本形式（dict/list 会 JSON 化），失败时为空
    error: str | None = None
    duration_seconds: float = 0.0
    attempt: int = 1

    @property
    def ok(self) -> bool:
        """只有真正执行成功才算 ok；被拒绝/参数错误都不算。"""
        return self.status == "success"

    def as_observation(self) -> str:
        """喂回给模型的"观察结果"文本：成功给输出，失败给错误原因，让模型能自己纠正。"""
        if self.ok:
            return self.output
        return f"[工具 {self.tool} 调用失败：{self.status}] {self.error or ''}".strip()


ToolCallSink = Callable[[ToolResult], None]


def _args_model_from_signature(fn: Callable[..., Any], name: str) -> type[BaseModel]:
    """把函数签名翻译成 pydantic 模型：有默认值的参数可选，没有的必填，没标注类型的当 Any。

    ``extra="forbid"``：模型多传一个不存在的参数直接判 invalid_args，而不是静默忽略——
    多出来的参数往往意味着模型误解了工具用途，早暴露比晚暴露好。
    """
    hints: dict[str, Any] = {}
    try:
        hints = inspect.get_annotations(fn, eval_str=True)
    except Exception:  # 前向引用解析失败时退回 Any，不因为类型标注问题让注册失败
        hints = getattr(fn, "__annotations__", {}) or {}
    fields: dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ToolError(f"工具 {name} 的签名不能有 *args/**kwargs：模型没法知道该传什么")
        annotation = hints.get(param.name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (annotation, default)
    return create_model(f"{name}_Args", __config__=ConfigDict(extra="forbid"), **fields)


def _stringify(value: Any) -> str:
    """工具返回值统一转成文本给模型看：pydantic 对象/dict/list 走 JSON，其余 str()。"""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json(exclude_none=True)
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


# --------------------------------------------------------------------------------------
# 确认门（人机协作：关键决策点引入人类确认）
# --------------------------------------------------------------------------------------


class ConfirmationGate(Protocol):
    """确认门协议：返回 True 放行，False 拒绝。实现方决定"问谁、怎么问"。"""

    def confirm(self, spec: ToolSpec, args: dict[str, Any]) -> bool:
        """返回是否允许执行该工具。"""
        ...


class AutoDenyGate:
    """默认门：一律拒绝。没人配置确认机制时，有副作用的工具就不该被自动执行。"""

    def confirm(self, spec: ToolSpec, args: dict[str, Any]) -> bool:
        """总是拒绝。"""
        return False


class AutoApproveGate:
    """全部放行。只给单元测试和你完全信任工具集的离线批处理用。"""

    def confirm(self, spec: ToolSpec, args: dict[str, Any]) -> bool:
        """总是放行。"""
        return True


class AllowlistGate:
    """按工具名白名单放行：生产环境把评审过的工具名写进配置，其余需确认的工具一律拒绝。"""

    def __init__(self, allowed: Iterable[str]) -> None:
        """``allowed``：允许自动执行的工具名集合。"""
        self.allowed = set(allowed)

    def confirm(self, spec: ToolSpec, args: dict[str, Any]) -> bool:
        """工具名在白名单里才放行。"""
        return spec.name in self.allowed


class CallbackGate:
    """把决定权交给一个回调（例如接到审批系统 / 聊天机器人按钮）。"""

    def __init__(self, fn: Callable[[ToolSpec, dict[str, Any]], bool]) -> None:
        """``fn(spec, args) -> bool``：返回 True 放行。"""
        self.fn = fn

    def confirm(self, spec: ToolSpec, args: dict[str, Any]) -> bool:
        """调用回调拿结果。"""
        return bool(self.fn(spec, args))


class ConsoleGate:
    """本地 demo 用：在终端里打印工具名和参数，问一句 y/n。"""

    def __init__(self, input_fn: Callable[[str], str] = input) -> None:
        """``input_fn`` 可替换成别的读入函数，方便测试。"""
        self.input_fn = input_fn

    def confirm(self, spec: ToolSpec, args: dict[str, Any]) -> bool:
        """回答以 y 开头视为同意。"""
        answer = self.input_fn(f"[确认] Agent 想调用工具 {spec.name}，参数 {json.dumps(args, ensure_ascii=False)}。允许吗？[y/N] ")
        return answer.strip().lower().startswith("y")


# --------------------------------------------------------------------------------------
# 注册表
# --------------------------------------------------------------------------------------


@dataclass
class ToolRegistry:
    """工具注册表：装饰器注册、按名字查找、带确认门和指标上报地执行。"""

    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        requires_confirmation: bool = False,
    ):
        """装饰器：``@registry.tool`` 或 ``@registry.tool(name=..., requires_confirmation=True)``。

        说明文字优先用 ``description``，没有就取函数 docstring 的第一段；两者都没有则拒绝注册——
        没有说明的工具模型不知道什么时候该用，等于没注册。
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            """真正的装饰器：生成 ToolSpec 并注册，原函数原样返回。"""
            tool_name = name or func.__name__
            doc = description or inspect.getdoc(func) or ""
            doc = doc.strip().split("\n\n", 1)[0].strip()
            if not doc:
                raise ToolError(f"工具 {tool_name} 缺少说明：请写 docstring 或传 description")
            self.register(ToolSpec(tool_name, doc, func, _args_model_from_signature(func, tool_name), requires_confirmation))
            return func

        return decorator(fn) if fn is not None else decorator

    def register(self, spec: ToolSpec) -> None:
        """登记一个 ToolSpec；重名视为配置错误直接报错，避免后注册的悄悄覆盖先注册的。"""
        if spec.name in self._tools:
            raise ToolError(f"工具 {spec.name} 已经注册过")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        """按名字取工具，不存在抛 ToolNotFoundError。"""
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"没有名为 {name} 的工具，可用工具：{sorted(self._tools)}") from None

    def names(self) -> list[str]:
        """已注册的工具名列表（按名字排序，方便稳定地出现在 prompt 里）。"""
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        """已注册的全部 ToolSpec。"""
        return [self._tools[n] for n in self.names()]

    def __len__(self) -> int:
        """工具数量。"""
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        """支持 ``name in registry`` 判断是否已注册。"""
        return name in self._tools

    def describe_for_prompt(self) -> str:
        """把工具清单渲染成 JSON 文本嵌进系统提示词。"""
        return json.dumps([s.to_prompt_dict() for s in self.specs()], ensure_ascii=False, indent=2)

    def call(
        self,
        name: str,
        args: dict[str, Any] | None,
        *,
        gate: ConfirmationGate | None = None,
        sink: ToolCallSink | None = None,
        attempt: int = 1,
    ) -> ToolResult:
        """执行一次工具调用：查找 -> 参数校验 -> 确认门 -> 执行 -> 计时/上报。

        永远返回 ``ToolResult`` 而不抛异常（工具函数自己的异常也被包成 status="error"）：
        Agent 循环需要把失败原因回灌给模型让它换个做法，而不是整个任务崩掉。
        """
        args = dict(args or {})
        started = time.perf_counter()

        def finish(status: str, output: str = "", error: str | None = None) -> ToolResult:
            """统一收口：构造 ToolResult、计时、上报 sink。"""
            result = ToolResult(name, args, status, output, error, time.perf_counter() - started, attempt)
            if sink is not None:
                try:
                    sink(result)
                except Exception:  # 指标上报失败不能影响业务
                    pass
            return result

        try:
            spec = self.get(name)
        except ToolNotFoundError as exc:
            return finish("not_found", error=str(exc))
        try:
            clean = spec.args_model.model_validate(args)
        except ValidationError as exc:
            return finish("invalid_args", error=f"参数不合法：{exc.errors(include_url=False)}")
        if spec.requires_confirmation:
            approved = (gate or AutoDenyGate()).confirm(spec, args)
            if not approved:
                return finish("denied", error=f"工具 {name} 需要人工确认，本次未获批准")
        try:
            output = spec.fn(**clean.model_dump())
        except Exception as exc:
            return finish("error", error=f"{type(exc).__name__}: {exc}")
        return finish("success", output=_stringify(output))
