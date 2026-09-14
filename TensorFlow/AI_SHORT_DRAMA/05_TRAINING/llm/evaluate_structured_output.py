"""评估一个 LLM（或微调后的 checkpoint）做结构化输出的可靠性。

指标（对 build_sft_dataset.py 产出的 jsonl 逐条跑）：
- json_valid_rate     ：输出能被解析成 JSON 的比例
- schema_valid_rate   ：能通过对应 Pydantic schema 校验的比例（枚举值、必填字段、extra=forbid）
- first_try_rate      ：不靠 BaseAgent 重试、第一次就合法的比例
- per_task            ：按 task（story_bible / shots / image_prompt ...）分组的同样指标
这就是"换底座模型 / 微调前后"要看的那几个数字。跑真实模型需要 GPU；``--provider mock`` 用
02 的 fixture 走一遍验证脚本本身。

用法：
  python 05_TRAINING/llm/evaluate_structured_output.py --dataset data/sft.jsonl --provider local --model ./out/qwen-sft
  python 05_TRAINING/llm/evaluate_structured_output.py --dataset data/sft.jsonl --provider mock --limit 20
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "02_STORY_ENGINE", _ROOT / "03_STRUCTURED_DATA"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import schemas as sch  # noqa: E402
from agents.base import PROMPTS_DIR, BaseAgent, LLMProvider, _extract_json  # noqa: E402

TASK_SCHEMAS: dict[str, type] = {
    "story_bible": sch.StoryBible,
    "character": sch.Character,
    "episode": sch.Episode,
    "script": sch.Script,
    "scene": sch.Scene,
    "shots": sch.ShotsFile,
    "image_prompt": sch.ImagePrompt,
    "video_prompt": sch.VideoPrompt,
    "qc_verdict": sch.QCVerdict,
    "editorial": sch.EditorialDecision,
}


class _EvalAgent(BaseAgent):
    pass


def evaluate(provider: LLMProvider, samples: list[dict[str, Any]], max_retries: int = 3) -> dict[str, Any]:
    totals = defaultdict(lambda: {"n": 0, "json_valid": 0, "schema_valid": 0, "first_try": 0, "retry_success": 0})
    system_cache: dict[str, str] = {}
    for sample in samples:
        task = sample["task"]
        schema = TASK_SCHEMAS[task]
        system_file = sample.get("system_prompt_file", "story_agent_system.txt")
        if system_file not in system_cache:
            system_cache[system_file] = (PROMPTS_DIR / system_file).read_text(encoding="utf-8")
        bucket = totals[task]
        bucket["n"] += 1
        raw = provider.complete(system_cache[system_file], sample["instruction"])
        try:
            json.loads(_extract_json(raw))
            bucket["json_valid"] += 1
        except json.JSONDecodeError:
            pass
        parsed, _ = BaseAgent._try_parse(raw, schema)
        if parsed is not None:
            bucket["schema_valid"] += 1
            bucket["first_try"] += 1
            continue
        # 第一次不合法：走 BaseAgent 的带报错重试，看重试能不能救回来
        agent = _EvalAgent(provider, system_cache[system_file], max_retries=max_retries)
        instruction = sample["instruction"].split("\n\n[TARGET_SCHEMA=")[0]
        try:
            agent.generate(instruction, schema)
            bucket["retry_success"] += 1
            bucket["schema_valid"] += 1
        except Exception:
            pass

    def rates(b: dict[str, int]) -> dict[str, float]:
        n = max(b["n"], 1)
        return {
            "n": b["n"],
            "json_valid_rate": round(b["json_valid"] / n, 4),
            "schema_valid_rate": round(b["schema_valid"] / n, 4),
            "first_try_rate": round(b["first_try"] / n, 4),
            "retry_rescued": b["retry_success"],
        }

    overall = {"n": 0, "json_valid": 0, "schema_valid": 0, "first_try": 0, "retry_success": 0}
    for b in totals.values():
        for k in overall:
            overall[k] += b[k]
    return {
        "overall": rates(overall),
        "per_task": {task: rates(b) for task, b in sorted(totals.items())},
        "provider": provider.provider_name,
        "model": provider.model,
    }


def load_samples(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def build_provider_from_args(kind: str, model: str | None) -> LLMProvider:
    if kind == "mock":
        return importlib.import_module("02_STORY_ENGINE.demo").build_mock_provider()
    factory = importlib.import_module("02_STORY_ENGINE.agents.provider_factory")
    return factory.build_provider(kind, model, use_cache=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估结构化输出可靠性")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--provider", default="local", choices=["mock", "local", "anthropic", "openai"])
    parser.add_argument("--model", default=None, help="模型 id 或微调 checkpoint 路径")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="把结果 JSON 写到这个文件")
    args = parser.parse_args()
    result = evaluate(build_provider_from_args(args.provider, args.model), load_samples(args.dataset, args.limit))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
