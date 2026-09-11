"""
================================================================================
 Context Course · Chapter 5 · 钩子(Hooks)（学习笔记描述）
================================================================================
 一句话：Hook 是 agent 生命周期确定性节点上的用户处理器——观测/拦截/注入，不靠模型自觉。
 本章讲：
   ① 事件点：UserPromptSubmit(注入上下文)、PreToolUse(允许/拒绝/改参)、PostToolUse(后处理)。
   ② 用 hook 做安全拦截、日志、上下文注入、事件流。
   ③ 为什么用 hook 而不是让模型自己做(确定性 + 不可绕过)。
 要点：Hook = 在 agent 循环里插确定性护栏(对标 update-config 里的 settings hooks)。
 说明：讲解 + 示例，参考为主。
================================================================================
"""

# # Unit 5: Hooks
#
# ## What Are Hooks?
#
# Hooks are user-defined handlers that run at deterministic points in an agent's lifecycle. When an agent is about to call a tool, finish a turn, or start a new session, the runtime pauses, invokes any hooks registered for that event, and then continues. Hooks let you observe what the agent is doing, block unsafe actions, inject extra context, or stream events to an external system. Crucially, they let you do this without asking the model to do it itself.
#
# ```text
# User prompt
#     │
#     ├─[UserPromptSubmit hook]─► log / inject context
#     ▼
# Model reasoning
#     │
#     ├─[PreToolUse hook]─► allow, deny, rewrite args
#     ▼
# Tool executes
#     │
#     ├─[PostToolUse hook]─► log, analyze, post-process
#     ▼
# Model continues
#     │
#     └─[Stop / SessionEnd hooks]─► persist, notify, tear down
# ```
#
# Skills, MCP, plugins, and subagents all shape *what the agent can do*. Hooks shape *what happens around every step the agent takes*. That makes them the right surface for observability, guardrails, and automation glue.
#
# ## Why Hooks Matter
#
# A model asked to "always run the linter after editing" will eventually forget or may not be consistent in its implementation. A hook on `PostToolUse` that runs the linter is consistent every single time and always runs. Hooks turn conventions into code.
#
# > [!NOTE]
# > A linter is a tool that checks code for common mistakes, style issues, and project rule violations. It does not usually run the whole program; it scans the code and reports things that look wrong or inconsistent.
#
# Three common patterns benefit the most:
#
# **Observability.** Every tool call, prompt, and stop event can be captured in a log or dashboard so you can see what the agent actually did — not what it said it did.
#
# **Guardrails.** A hook can inspect a `Bash` command before it runs and deny anything that touches `~/.ssh/` or `rm -rf /`. Guardrails expressed as code are more reliable than guardrails expressed as prompt instructions.
#
# **Automation.** Hooks can run formatters, linters, type-checkers, or test suites automatically after a file edit. The agent does not need to remember to do it — the runtime does.
#
# ## Platform Landscape
#
# The four platforms in this course all support hooks, but with different shapes:
#
# - **Claude Code** — JSON configuration in `.claude/settings.json` (or inside a plugin). Events use `PascalCase` names like `PreToolUse` and `Stop`. Hooks can be shell commands, HTTP endpoints, prompts, or subagents.
# - **Codex** — JSON configuration in `.codex/hooks.json`, gated behind a feature flag in `config.toml`. A smaller set of events (also `PascalCase`), and shell-command handlers that receive a JSON payload on stdin.
# - **OpenCode** — TypeScript/JavaScript plugin modules in `.opencode/plugins/`. Hooks are object keys on the exported plugin, like `"tool.execute.before"` or a generic `event` callback. There is no JSON event config — everything is code.
# - **Pi** — TypeScript/JavaScript extensions in `.pi/extensions/` or `~/.pi/agent/extensions/`. Events use `lower_snake_case` names like `before_agent_start`, `tool_call`, and `tool_result`, and handlers are registered in code with `pi.on(...)`.
#
# The mental model is consistent across all four: events fire, handlers run, and handlers can influence the agent's next step. Only the syntax and the event names change.
#
# ## What You'll Build
#
# This unit walks through the hook lifecycle, shows the exact config shape for each platform, and then builds a working **Agent Activity Dashboard**: a Gradio app that receives hook events over HTTP and visualizes tool calls, prompts, and sessions in real time. By the end you will have one dashboard that works against all four agents.
#
# ## What You'll Learn
#
# This unit covers the common hook lifecycle shared by Claude Code, Codex, OpenCode, and Pi; where hook configuration lives on each platform and what JSON or code shape it takes; how hook handlers influence the agent through exit codes, JSON output, thrown errors, or returned values; and how to build a Gradio dashboard that turns hook events into a live view of agent activity.
#
# Next, a tour of the hook events themselves.
#
# # Hook Events and the Agent Lifecycle
#
# Before wiring anything up, it helps to know which events actually fire and when. The vocabulary is slightly different on each platform, but the lifecycle they describe is shared.
#
# <iframe
#     src="https://context-course-hook-lifecycle.static.hf.space"
#     frameborder="0"
#     width="850"
#     height="450"
# >
#
# ## The Shared Lifecycle
#
# At the abstract level, every agent session moves through the same phases:
#
# ```
# ┌─────────────────────────────────────────────┐
# │ Session starts                              │ ← SessionStart
# ├─────────────────────────────────────────────┤
# │ Repeat for each turn:                       │
# │   User submits a prompt                     │ ← UserPromptSubmit
# │   Model reasons and may call tools          │
# │     Before each tool call                   │ ← PreToolUse
# │     After each tool call                    │ ← PostToolUse
# │   Turn ends                                 │ ← Stop
# ├─────────────────────────────────────────────┤
# │ Session ends                                │ ← SessionEnd
# └─────────────────────────────────────────────┘
# ```
#
# This is the shared mental model. The platforms in this course map onto these moments with different names, different configuration surfaces, and a few gaps or extra events.
#
# ## Events by Platform
#
# Claude Code has the richest event set of the four platforms. Every event uses `PascalCase` and is configured in `.claude/settings.json` (or inside a plugin's `hooks/hooks.json`).
#
# **Core lifecycle events:**
# - `SessionStart` — New session begins (fresh, resume, or compaction continuation).
# - `InstructionsLoaded` — `CLAUDE.md` files have been loaded.
# - `UserPromptSubmit` — User submitted a prompt, before the model sees it.
# - `PreToolUse` — Before a tool call runs.
# - `PermissionRequest` — Permission prompt about to be shown.
# - `PermissionDenied` — A tool call was denied by the auto-mode classifier.
# - `PostToolUse` — After a tool call returns.
# - `PostToolUseFailure` — Tool call failed.
# - `Stop` — Turn finished.
# - `SessionEnd` — Session closed.
#
# **Subagent and task events:**
# - `SubagentStart`, `SubagentStop` — A delegated subagent starts or finishes.
# - `TaskCreated`, `TaskCompleted` — The TodoWrite task list changes.
#
# **Environment events:**
# - `CwdChanged`, `FileChanged` — Working directory or a tracked file changed.
# - `WorktreeCreate`, `WorktreeRemove` — Git worktree created or removed.
# - `PreCompact`, `PostCompact` — Context compaction about to happen / just finished.
# - `ConfigChange`, `Notification` — Settings changed; Claude raised a notification.
#
# Matcher groups filter on event-specific fields. Tool events match tool names (for example, `"matcher": "Bash"` or `"matcher": "Edit|Write"`), while handler-level `if` conditions can use permission-rule syntax to inspect tool arguments (for example, `"if": "Bash(rm *)"`).
#
# Codex hooks are experimental and must be enabled explicitly. In `~/.codex/config.toml`:
#
# ```toml
# [features]
# codex_hooks = true
# ```
#
# Codex ships a smaller core set of events, all `PascalCase`, configured in `~/.codex/hooks.json` or `<repo>/.codex/hooks.json`:
#
# - `SessionStart` — Session begins. Matcher filters the `source` field: `startup`, `resume`, or `clear`.
# - `UserPromptSubmit` — User prompt submitted.
# - `PreToolUse` — Before a supported tool call, including `Bash`, `apply_patch` (`Edit|Write` aliases), and MCP tools.
# - `PermissionRequest` — Permission prompt about to be shown.
# - `PostToolUse` — After a supported tool call, including `Bash`, `apply_patch` (`Edit|Write` aliases), and MCP tools.
# - `Stop` — Turn finished.
#
# The Codex event set is intentionally close to the classic Claude Code lifecycle, which makes cross-agent hook scripts easier to share — but note that the tool-event surface is still narrower than Claude Code's and does not intercept every tool path.
#
# Matchers are regexes evaluated against `tool_name` for tool events or `source` for `SessionStart`. Codex hooks are changing quickly, so small details may differ between Codex versions. The examples below show the event set used in this course.
#
# OpenCode does not have a JSON event config. Plugins are TypeScript or JavaScript modules in `.opencode/plugins/`, and each exported plugin returns an object whose **keys are event names**. The runtime calls the matching key when the event fires.
#
# **Typed hook keys (first-class):**
# - `"tool.execute.before"` — About to execute a tool.
# - `"tool.execute.after"` — Tool execution finished.
# - `"shell.env"` — About to launch a shell command; mutate the environment.
# - `"experimental.session.compacting"` — Session is being compacted.
#
# **Generic `event` callback** — receives `{ event }` with an `event.type` field. Types include lifecycle and UI events like:
# - `session.created`, `session.updated`, `session.idle`, `session.compacted`, `session.deleted`, `session.error`, `session.diff`, `session.status`
# - `message.updated`, `message.removed`, `message.part.updated`, `message.part.removed`
# - `command.executed`, `file.edited`, `file.watcher.updated`
# - `permission.asked`, `permission.replied`
# - `lsp.client.diagnostics`, `lsp.updated`
# - `todo.updated`, `server.connected`, `installation.updated`
# - `tui.prompt.append`, `tui.command.execute`, `tui.toast.show`
#
# OpenCode has no `UserPromptSubmit` event. The nearest equivalent is reading the new user message through the generic `event` callback when `message.updated` fires.
#
# Pi hooks live inside TS/JS extensions in `.pi/extensions/` or `~/.pi/agent/extensions/`. The built-in lifecycle is lower_snake_case:
#
# **Core lifecycle events:**
# - `session_start` — Session begins (`startup`, `reload`, `new`, `resume`, or `fork`)
# - `before_agent_start` — User prompt received; can inject messages or edit the system prompt
# - `agent_start` — Agent loop begins
# - `turn_start`, `turn_end` — One model turn starts or finishes
# - `tool_call` — Before a tool executes; can mutate args or block
# - `tool_result` — After a tool executes; can rewrite the result
# - `agent_end` — Request finished
# - `session_shutdown` — Extension runtime is being torn down
#
# **Session/control events:**
# - `session_before_switch`, `session_before_fork` — Intercept `/new`, `/resume`, `/fork`, or `/clone`
# - `session_before_compact`, `session_compact`, `session_before_tree`, `session_tree` — Context management and tree navigation
# - `model_select`, `user_bash` — Model changes and user-initiated shell commands
#
# Pi doesn't use a JSON hook file. You register handlers in code with `pi.on("event_name", handler)` and ship them in an extension or Pi package.
#
# ## Event Input: What Your Hook Receives
#
# Command hooks receive a JSON payload on **stdin**. HTTP hooks receive the same payload as the POST body. Common fields across events:
#
# ```json
# {
#   "session_id": "...",
#   "transcript_path": "/path/to/transcript.jsonl",
#   "cwd": "/path/to/project",
#   "permission_mode": "default",
#   "hook_event_name": "PreToolUse"
# }
# ```
#
# Tool-related events add `tool_name`, `tool_input`, and (on `PostToolUse`) `tool_response`. Subagent events add `agent_id` and `agent_type`. The project root is also available as an environment variable `CLAUDE_PROJECT_DIR`.
#
# Command hooks receive JSON on stdin. Common fields:
#
# ```json
# {
#   "session_id": "...",
#   "transcript_path": "/path/to/transcript.jsonl",
#   "cwd": "/path/to/project",
#   "hook_event_name": "PreToolUse",
#   "model": "gpt-5-codex"
# }
# ```
#
# Turn-scoped events add `turn_id`. Tool events add `tool_name`, `tool_use_id`, and `tool_input`; for `Bash` and `apply_patch`, the command is in `tool_input.command`. `PostToolUse` also includes `tool_response`. `UserPromptSubmit` adds `prompt`. `Stop` adds `stop_hook_active` and `last_assistant_message`.
#
# OpenCode hooks are JS/TS functions, not stdin scripts. Each event type has a typed `(input, output)` signature:
#
# ```ts
# "tool.execute.before": async (input, output) => {
#   // input.tool       — tool name (e.g. "read", "bash")
#   // input.sessionID  — session identifier
#   // output.args      — tool arguments (mutable)
# }
# ```
#
# `"tool.execute.after"` adds the result on `output`. `"shell.env"` gives you `output.env` to mutate. The generic `event` callback receives `{ event }` where `event.type` identifies the event.
#
# The plugin also receives a context object at construction time: `{ project, directory, worktree, client, $ }`. Use `client.app.log({...})` for structured logging and `$` (Bun shell) to run commands.
#
# Pi hooks are TS/JS functions registered in an extension. Common event shapes look like this:
#
# ```ts
# pi.on("before_agent_start", async (event, ctx) => {
#   // event.prompt        — user prompt text
#   // event.systemPrompt  — current system prompt for this turn
#   // event.images        — attached images (if any)
# });
#
# pi.on("tool_call", async (event, ctx) => {
#   // event.toolName      — "bash", "read", "write", etc.
#   // event.toolCallId    — unique tool call id
#   // event.input         — mutable tool arguments
# });
#
# pi.on("tool_result", async (event, ctx) => {
#   // event.toolName      — tool name
#   // event.input         — final tool arguments
#   // event.content       — tool result content blocks
#   // event.details       — structured tool details
#   // event.isError       — whether the tool failed
# });
# ```
#
# For nested async work inside a handler, use `ctx.signal` so Esc can cancel the extension's own `fetch()` calls or other abort-aware work.
#
# ## How Hooks Influence the Agent
#
# Hooks are not purely observational — they can change what happens next.
#
# **Exit codes (command hooks):**
# - Exit `0` — allow, no change.
# - Exit `2` with a message on stderr — block or continue per event semantics (e.g. block the tool call on `PreToolUse`, erase the prompt on `UserPromptSubmit`).
#
# **JSON on stdout** for finer control:
#
# ```json
# {
#   "hookSpecificOutput": {
#     "hookEventName": "PreToolUse",
#     "permissionDecision": "deny",
#     "permissionDecisionReason": "No network access in this project"
#   }
# }
# ```
#
# Also supported: top-level `continue`, `stopReason`, `suppressOutput`, `systemMessage`, and the legacy `{ "decision": "block", "reason": "..." }` form.
#
# **HTTP hooks** return the same JSON as a 2xx response body. Non-2xx responses or timeouts are treated as non-blocking errors.
#
# **Exit codes:**
# - `0` — success.
# - `2` with stderr message — block or continue per event.
#
# **JSON on stdout** uses a shape similar to Claude Code's:
#
# ```json
# {
#   "systemMessage": "Injected context for the model",
#   "hookSpecificOutput": {
#     "hookEventName": "PreToolUse",
#     "permissionDecision": "deny",
#     "permissionDecisionReason": "No network access in this project"
#   }
# }
# ```
#
# For `SessionStart`, `UserPromptSubmit`, and `PostToolUse`, `hookSpecificOutput` can include `additionalContext` to inject text into the conversation.
#
# OpenCode hooks influence the agent by mutating the `output` object or by throwing. To rewrite a tool call, mutate `output.args`. To block it, throw an error:
#
# ```ts
# "tool.execute.before": async (input, output) => {
#   if (input.tool === "bash" && /rm -rf/.test(output.args.command ?? "")) {
#     throw new Error("dangerous command blocked by policy")
#   }
# }
# ```
#
# There are no exit codes or JSON-over-stdin. All influence happens through code.
#
# Pi hooks influence the agent by returning structured values or mutating event state in code.
#
# - Return `{ block: true, reason: "..." }` from `tool_call` to cancel a tool.
# - Mutate `event.input` inside `tool_call` to rewrite arguments before execution.
# - Return `{ content, details, isError }` from `tool_result` to rewrite a tool result.
# - Return `message` or `systemPrompt` from `before_agent_start` to inject context into the next turn.
#
# ```ts
# pi.on("tool_call", async (event) => {
#   if (event.toolName === "bash" && /rm -rf/.test(event.input.command as string)) {
#     return { block: true, reason: "dangerous command blocked by policy" };
#   }
# });
#
# pi.on("before_agent_start", async (event) => ({
#   systemPrompt: event.systemPrompt + "\n\nAlways explain risky shell commands before running them.",
# }));
# ```
#
# ## Choosing the Right Event
#
# A quick reference for common goals:
#
# - **Log every tool call** → `PreToolUse` / `tool.execute.before` / `tool_call`.
# - **Run a linter after an edit** → `PostToolUse` (match `Edit|Write`) / `tool.execute.after` / `tool_result`.
# - **Block dangerous commands** → `PreToolUse` with exit code `2`, a thrown error, or `tool_call` with `{ block: true }`.
# - **Inject repo context on each turn** → `UserPromptSubmit` with `additionalContext` (Claude Code / Codex), the `event` callback on `message.updated` (OpenCode), or `before_agent_start` (Pi).
# - **Persist conversation state** → `Stop` / `SessionEnd`, `session.idle` on OpenCode, or `agent_end` / `session_shutdown` on Pi.
# - **Add environment variables to shells** → `shell.env` (OpenCode), or mutate `tool_call` / wrap user bash handling in an extension on Pi.
#
# ## Key Takeaways
#
# The platforms expose the same underlying lifecycle with different vocabulary and different levels of granularity. Claude Code has the richest event set and supports four handler types. Codex has a smaller, tightly focused set behind a feature flag. OpenCode folds hooks into its plugin system as typed function keys rather than a JSON config. Pi uses extension events such as `before_agent_start`, `tool_call`, and `tool_result`. Once you pick an event for the goal at hand, the rest of the work is the same across them.
#
# Next, a quick quiz before we wire these events into a live Gradio dashboard.
#
# # Hands-On: Build an Agent Activity Dashboard with Gradio
#
# This project wires the hooks you configured in the previous lesson into a live dashboard. The dashboard is a Gradio app that accepts hook events over HTTP, shows the most recent calls in a table, and plots tool usage over time. It works against Claude Code, Codex, OpenCode, and Pi — with the same agent doing the work, you see what actually happens under the hood.
#
# ## What You'll Build
#
# One Python process that does two things at once:
# 1. A **FastAPI** endpoint at `POST /event` that any hook can post to.
# 2. A **Gradio** app mounted on the same server that renders the events live.
#
# The Gradio app polls the in-memory event buffer and re-renders every second, so you can watch the agent work in real time.
#
# ## Project Setup
#
# Create a fresh project directory:
#
# ```bash
# mkdir agent-activity-dashboard
# cd agent-activity-dashboard
# ```
#
# Install dependencies:
#
# ```bash
# pip install "gradio>=4.41" "fastapi" "uvicorn[standard]" "pandas"
# ```
#
# Create `requirements.txt` for later deployment:
#
# ```text
# gradio>=4.41
# fastapi
# uvicorn[standard]
# pandas
# ```
#
# ## Step 1: Build the Receiver and Dashboard
#
# Create `app.py`:
#
# ```python
# import datetime as dt
# from collections import Counter, deque
# from typing import Any
#
# import gradio as gr
# import pandas as pd
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
#
# # ---------- Shared state ----------
# MAX_EVENTS = 500
# events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENTS)
#
# def _truncate(value: Any, n: int) -> str:
#     text = "" if value is None else str(value)
#     return text if len(text) <= n else text[: n - 1] + "…"
#
# def _normalize(body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
#     """Map Claude Code / Codex / OpenCode / Pi payloads to one shape."""
#     platform = (
#         body.get("platform")
#         or headers.get("x-platform")
#         or "unknown"
#     )
#     event_name = body.get("event") or body.get("hook_event_name") or "Unknown"
#     tool = body.get("tool") or body.get("tool_name") or ""
#     args = body.get("args") or body.get("tool_input") or body.get("prompt") or ""
#     return {
#         "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
#         "platform": str(platform),
#         "event": str(event_name),
#         "tool": str(tool),
#         "args": _truncate(args, 200),
#     }
#
# # ---------- FastAPI receiver ----------
# api = FastAPI(title="Agent Activity Dashboard")
#
# @api.post("/event")
# async def event(req: Request):
#     try:
#         body = await req.json()
#     except Exception:
#         body = {}
#     record = _normalize(body, {k.lower(): v for k, v in req.headers.items()})
#     events.appendleft(record)
#     # Return an empty body so hook callers never block on the dashboard receiver.
#     return JSONResponse({})
#
# @api.get("/health")
# def health():
#     return {"ok": True, "events": len(events)}
#
# # ---------- Gradio views ----------
# COLUMNS = ["timestamp", "platform", "event", "tool", "args"]
#
# def events_df() -> pd.DataFrame:
#     if not events:
#         return pd.DataFrame(columns=COLUMNS)
#     return pd.DataFrame(list(events), columns=COLUMNS)
#
# def tool_counts_df() -> pd.DataFrame:
#     counter = Counter(e["tool"] for e in events if e["tool"])
#     rows = [{"tool": tool, "count": n} for tool, n in counter.most_common(15)]
#     return pd.DataFrame(rows, columns=["tool", "count"])
#
# def summary_md() -> str:
#     total = len(events)
#     platforms = sorted({e["platform"] for e in events}) or ["(none)"]
#     tools = sorted({e["tool"] for e in events if e["tool"]})
#     tools_display = ", ".join(tools) if tools else "(none)"
#     return (
#         f"**Events:** {total} (buffer holds up to {MAX_EVENTS})  \n"
#         f"**Platforms seen:** {', '.join(platforms)}  \n"
#         f"**Tools seen:** {tools_display}"
#     )
#
# def refresh():
#     return events_df(), tool_counts_df(), summary_md()
#
# def clear_events():
#     events.clear()
#     return refresh()
#
# with gr.Blocks(title="Agent Activity Dashboard") as ui:
#     gr.Markdown("# Agent Activity Dashboard")
#     gr.Markdown(
#         "Point your Claude Code, Codex, OpenCode, or Pi hooks/extensions at "
#         "`POST http://localhost:8000/event` to see live activity here."
#     )
#
#     header = gr.Markdown(value=summary_md())
#
#     with gr.Row():
#         clear_btn = gr.Button("Clear events", variant="secondary")
#
#     chart = gr.BarPlot(
#         value=tool_counts_df(),
#         x="tool",
#         y="count",
#         title="Tool usage",
#         tooltip=["tool", "count"],
#         height=280,
#     )
#
#     table = gr.Dataframe(
#         value=events_df(),
#         headers=COLUMNS,
#         label="Recent events (newest first)",
#         wrap=True,
#         interactive=False,
#     )
#
#     # Poll the shared buffer once a second.
#     gr.Timer(1.0).tick(refresh, outputs=[table, chart, header])
#     clear_btn.click(clear_events, outputs=[table, chart, header])
#
# # Mount Gradio on the FastAPI app so `/event` and the UI share one process.
# app = gr.mount_gradio_app(api, ui, path="/")
#
# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# ```
#
# Run it:
#
# ```bash
# python app.py
# ```
#
# Open `http://localhost:8000` in a browser. The dashboard is empty — that's expected until a hook sends an event.
#
# > [!TIP]
# > The FastAPI route for `/event` is defined before Gradio is mounted at `/`, so POSTs hit the receiver and everything else lands on the Gradio UI.
#
# ## Step 2: Connect Your Agent
#
# Use the hook config from the previous lesson. A quick reminder of the shape for each platform, adjusted so the normalizer in `app.py` can make sense of the payload.
#
# `.claude/settings.json`:
#
# ```json
# {
#   "hooks": {
#     "PreToolUse":  [{ "matcher": "*", "hooks": [{ "type": "http", "url": "http://localhost:8000/event", "headers": { "X-Platform": "claude-code" } }] }],
#     "PostToolUse": [{ "matcher": "*", "hooks": [{ "type": "http", "url": "http://localhost:8000/event", "headers": { "X-Platform": "claude-code" } }] }],
#     "UserPromptSubmit": [{ "hooks": [{ "type": "http", "url": "http://localhost:8000/event", "headers": { "X-Platform": "claude-code" } }] }],
#     "Stop":            [{ "hooks": [{ "type": "http", "url": "http://localhost:8000/event", "headers": { "X-Platform": "claude-code" } }] }],
#     "SessionStart":    [{ "hooks": [{ "type": "http", "url": "http://localhost:8000/event", "headers": { "X-Platform": "claude-code" } }] }]
#   }
# }
# ```
#
# Claude Code's payload already includes `hook_event_name`, `tool_name`, and `tool_input`, so the normalizer picks them up without extra work. The `X-Platform` header tags them as `claude-code`.
#
# Start a new Claude Code session in this project directory and ask it to do something concrete:
#
# ```
# List the files in this directory, then read README.md and summarize it.
# ```
#
# Make sure `~/.codex/config.toml` has:
#
# ```toml
# [features]
# codex_hooks = true
# ```
#
# These examples assume `jq` and `curl` are installed and available on your `PATH`.
#
# `.codex/hooks.json`:
#
# ```json
# {
#   "hooks": {
#     "PreToolUse":  [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": "jq -c '{platform:\"codex\", event:\"PreToolUse\", tool:.tool_name, args:.tool_input}' | curl -s --max-time 2 -X POST -H 'Content-Type: application/json' --data-binary @- http://localhost:8000/event || true" }] }],
#     "PostToolUse": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": "jq -c '{platform:\"codex\", event:\"PostToolUse\", tool:.tool_name, args:.tool_response}' | curl -s --max-time 2 -X POST -H 'Content-Type: application/json' --data-binary @- http://localhost:8000/event || true" }] }],
#     "UserPromptSubmit": [{ "hooks": [{ "type": "command", "command": "jq -c '{platform:\"codex\", event:\"UserPromptSubmit\", args:.prompt}' | curl -s --max-time 2 -X POST -H 'Content-Type: application/json' --data-binary @- http://localhost:8000/event || true" }] }],
#     "Stop":            [{ "hooks": [{ "type": "command", "command": "jq -c '{platform:\"codex\", event:\"Stop\", args:.last_assistant_message}' | curl -s --max-time 2 -X POST -H 'Content-Type: application/json' --data-binary @- http://localhost:8000/event || true" }] }],
#     "SessionStart":    [{ "hooks": [{ "type": "command", "command": "jq -c '{platform:\"codex\", event:\"SessionStart\"}' | curl -s --max-time 2 -X POST -H 'Content-Type: application/json' --data-binary @- http://localhost:8000/event || true" }] }]
#   }
# }
# ```
#
# Each hook reshapes Codex's payload into the normalized `{platform, event, tool, args}` shape, then POSTs it. The `--max-time 2` on `curl` protects the agent from hanging if the dashboard is slow, and `|| true` keeps logging best-effort if the dashboard is offline.
#
# Restart Codex to pick up the hooks, then try:
#
# ```
# Run `ls` in this directory and then show me the first 20 lines of app.py.
# ```
#
# `.opencode/plugins/dashboard.ts`:
#
# ```ts
# import type { Plugin } from "@opencode-ai/plugin"
#
# const URL = process.env.DASHBOARD_URL ?? "http://localhost:8000/event"
#
# async function send(event: string, payload: Record<string, unknown>) {
#   try {
#     await fetch(URL, {
#       method: "POST",
#       headers: { "Content-Type": "application/json" },
#       body: JSON.stringify({ platform: "opencode", event, ...payload }),
#       signal: AbortSignal.timeout(2000),
#     })
#   } catch {
#     // dashboard may be offline; never block a tool
#   }
# }
#
# export const DashboardPlugin: Plugin = async () => ({
#   "tool.execute.before": async (input, output) =>
#     send("PreToolUse", { tool: input.tool, args: output.args }),
#
#   "tool.execute.after": async (input, output) =>
#     send("PostToolUse", {
#       tool: input.tool,
#       args:
#         typeof (output as { output?: unknown }).output === "string"
#           ? ((output as { output: string }).output.slice(0, 200))
#           : "",
#     }),
#
#   event: async ({ event }) => {
#     if (event.type === "session.created") await send("SessionStart", {})
#     if (event.type === "session.idle") await send("Stop", {})
#   },
# })
# ```
#
# If this is the first OpenCode plugin in the project, initialize a local package.json too:
#
# ```bash
# cd .opencode
# bun init -y
# bun add -d @opencode-ai/plugin
# ```
#
# Restart OpenCode so the plugin loads at startup, then run:
#
# ```
# Read README.md and list its sections.
# ```
#
# `.pi/extensions/dashboard.ts`:
#
# ```ts
# import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
#
# const URL = process.env.DASHBOARD_URL ?? "http://localhost:8000/event";
#
# async function send(event: string, payload: Record<string, unknown>) {
#   try {
#     await fetch(URL, {
#       method: "POST",
#       headers: { "Content-Type": "application/json" },
#       body: JSON.stringify({ platform: "pi", event, ...payload }),
#       signal: AbortSignal.timeout(2000),
#     });
#   } catch {
#     // dashboard may be offline; never block the agent
#   }
# }
#
# export default function (pi: ExtensionAPI) {
#   pi.on("session_start", async () => {
#     await send("SessionStart", {});
#   });
#
#   pi.on("before_agent_start", async (event) => {
#     await send("UserPromptSubmit", { args: event.prompt });
#   });
#
#   pi.on("tool_call", async (event) => {
#     await send("PreToolUse", { tool: event.toolName, args: event.input });
#   });
#
#   pi.on("tool_result", async (event) => {
#     await send("PostToolUse", {
#       tool: event.toolName,
#       args: event.details ?? event.content,
#     });
#   });
#
#   pi.on("agent_end", async () => {
#     await send("Stop", {});
#   });
# }
# ```
#
# Create `.pi/extensions/` if needed, then start Pi (or run `/reload` if Pi is already open) and try:
#
# ```
# Read README.md and list its sections.
# ```
#
# ## Step 3: Watch It Live
#
# Leave `python app.py` running in one terminal and your agent running in another. As the agent works, events stream into the dashboard:
#
# ```
# timestamp              platform     event            tool    args
# 2026-04-20T10:15:02Z   claude-code  UserPromptSubmit          List the files…
# 2026-04-20T10:15:03Z   claude-code  PreToolUse       Bash    {"command":"ls"}
# 2026-04-20T10:15:03Z   claude-code  PostToolUse      Bash    {"exit_code":0,…}
# 2026-04-20T10:15:04Z   claude-code  PreToolUse       Read    {"path":"README…
# 2026-04-20T10:15:04Z   claude-code  PostToolUse      Read    {"content":"# …
# 2026-04-20T10:15:06Z   claude-code  Stop                     …
# ```
#
# The bar chart updates live as tools accumulate, which makes it obvious when the agent is stuck in a loop on the same tool.
#
# ## Step 4: Add a Guardrail
#
# A log-only dashboard is already useful, but the real power of hooks is that they can intervene. Extend the Claude Code hook to block dangerous commands.
#
# Add a second hook handler for `PreToolUse` that runs before the dashboard logger:
#
# ```json
# {
#   "hooks": {
#     "PreToolUse": [
#       {
#         "matcher": "Bash",
#         "hooks": [
#           {
#             "type": "command",
#             "command": "jq -r '.tool_input.command // \"\"' | grep -Eq 'rm -rf|:\\(\\)\\{.*\\|.*&.*\\}:' && { echo 'blocked: dangerous shell pattern' >&2; exit 2; } || exit 0"
#           }
#         ]
#       },
#       {
#         "matcher": "*",
#         "hooks": [
#           { "type": "http", "url": "http://localhost:8000/event",
#             "headers": { "X-Platform": "claude-code" } }
#         ]
#       }
#     ]
#   }
# }
# ```
#
# Now the same `PreToolUse` event does two things: denies obviously destructive shell patterns (exit code `2` with a stderr message) and logs everything else to the dashboard. Create a disposable test folder:
#
# ```bash
# mkdir -p /tmp/hook-guardrail-demo-delete-me
# ```
#
# Then ask the agent to run:
#
# ```text
# rm -rf /tmp/hook-guardrail-demo-delete-me
# ```
#
# You'll see it refuse before the command executes, and you'll see the guardrail event in the dashboard.
#
# > [!TIP]
# > The equivalent OpenCode guard is a `throw new Error("blocked: dangerous shell pattern")` inside `"tool.execute.before"`. For Codex, the hook can either exit `2` with a stderr reason or print `jq -c '{hookSpecificOutput:{hookEventName:"PreToolUse", permissionDecision:"deny", permissionDecisionReason:"…"}}'` on stdout.
#
# ## Step 5: Deploy (Optional)
#
# For a team-facing dashboard, host the Gradio app on [Hugging Face Spaces](https://huggingface.co/spaces). Push `app.py` and `requirements.txt`, and update the hook URLs from `http://localhost:8000/event` to `https://YOUR-USERNAME-agent-dashboard.hf.space/event`.
#
# A few cautions for a shared dashboard:
#
# - Payloads can contain secrets (prompts, file contents, command lines). Redact sensitive fields in `_normalize` before appending to the buffer.
# - A public Space is, well, public. Put authentication in front of `/event`, or keep the Space private and use a personal access token in the hook `headers`.
# - The in-memory buffer resets on restart. For durable history, swap `deque` for a SQLite file or a Spaces persistent volume.
#
# ## Full Directory Layout
#
# ```
# agent-activity-dashboard/
# ├── app.py                              # Gradio + FastAPI server
# ├── requirements.txt
# ├── .claude/
# │   └── settings.json                   # Claude Code hooks
# ├── .codex/
# │   └── hooks.json                      # Codex hooks
# ├── .opencode/
# │   └── plugins/
# │       └── dashboard.ts                # OpenCode plugin
# └── .pi/
#     └── extensions/
#         └── dashboard.ts                # Pi extension
# ```
#
# ## Best Practices Demonstrated
#
# This project shows how to pick the right event for observability (`PreToolUse` and `PostToolUse`), how to collapse four different payload shapes into one normalized record, and how to use exit codes / `hookSpecificOutput` / thrown errors / returned values to enforce policy. The Gradio + FastAPI pattern — one process, two surfaces — keeps the whole thing short and makes it easy to host on Spaces.
#
# ## Key Takeaways
#
# A hook is only useful if its output goes somewhere. A Gradio dashboard is a fast way to turn raw hook events into something you can actually learn from: you can watch an agent work, see where it's stuck, and prove that guardrails fire. The same dashboard works across Claude Code, Codex, OpenCode, and Pi because the event shape is narrow and the normalizer is small.
#
# Next, one more quiz to wrap up the unit.
#
import datetime as dt
from collections import Counter, deque
from typing import Any

import gradio as gr
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---------- Shared state ----------
MAX_EVENTS = 500
events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENTS)


def _truncate(value: Any, n: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= n else text[: n - 1] + "…"


def _normalize(body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    """Map Claude Code / Codex / OpenCode / Pi payloads to one shape."""
    platform = (
        body.get("platform")
        or headers.get("x-platform")
        or "unknown"
    )
    event_name = body.get("event") or body.get("hook_event_name") or "Unknown"
    tool = body.get("tool") or body.get("tool_name") or ""
    args = body.get("args") or body.get("tool_input") or body.get("prompt") or ""
    return {
        "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "platform": str(platform),
        "event": str(event_name),
        "tool": str(tool),
        "args": _truncate(args, 200),
    }


# ---------- FastAPI receiver ----------
api = FastAPI(title="Agent Activity Dashboard")


@api.post("/event")
async def event(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}
    record = _normalize(body, {k.lower(): v for k, v in req.headers.items()})
    events.appendleft(record)
    # Return an empty body so hook callers never block on the dashboard receiver.
    return JSONResponse({})


@api.get("/health")
def health():
    return {"ok": True, "events": len(events)}


# ---------- Gradio views ----------
COLUMNS = ["timestamp", "platform", "event", "tool", "args"]


def events_df() -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(list(events), columns=COLUMNS)


def tool_counts_df() -> pd.DataFrame:
    counter = Counter(e["tool"] for e in events if e["tool"])
    rows = [{"tool": tool, "count": n} for tool, n in counter.most_common(15)]
    return pd.DataFrame(rows, columns=["tool", "count"])


def summary_md() -> str:
    total = len(events)
    platforms = sorted({e["platform"] for e in events}) or ["(none)"]
    tools = sorted({e["tool"] for e in events if e["tool"]})
    tools_display = ", ".join(tools) if tools else "(none)"
    return (
        f"**Events:** {total} (buffer holds up to {MAX_EVENTS})  \n"
        f"**Platforms seen:** {', '.join(platforms)}  \n"
        f"**Tools seen:** {tools_display}"
    )


def refresh():
    return events_df(), tool_counts_df(), summary_md()


def clear_events():
    events.clear()
    return refresh()


with gr.Blocks(title="Agent Activity Dashboard") as ui:
    gr.Markdown("# Agent Activity Dashboard")
    gr.Markdown(
        "Point your Claude Code, Codex, OpenCode, or Pi hooks/extensions at "
        "`POST http://localhost:8000/event` to see live activity here."
    )

    header = gr.Markdown(value=summary_md())

    with gr.Row():
        clear_btn = gr.Button("Clear events", variant="secondary")

    chart = gr.BarPlot(
        value=tool_counts_df(),
        x="tool",
        y="count",
        title="Tool usage",
        tooltip=["tool", "count"],
        height=280,
    )

    table = gr.Dataframe(
        value=events_df(),
        headers=COLUMNS,
        label="Recent events (newest first)",
        wrap=True,
        interactive=False,
    )

    # Poll the shared buffer once a second.
    gr.Timer(1.0).tick(refresh, outputs=[table, chart, header])
    clear_btn.click(clear_events, outputs=[table, chart, header])


# Mount Gradio on the FastAPI app so `/event` and the UI share one process.
app = gr.mount_gradio_app(api, ui, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)