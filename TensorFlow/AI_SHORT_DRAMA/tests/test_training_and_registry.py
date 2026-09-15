"""05_TRAINING 数据集构造/评估脚本 + 06_MODELS registry 与 provider 默认值的一致性。"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_sft_dataset_from_demo_output(tmp_path):
    """验证能从 02_STORY_ENGINE 的 demo 产物构造出覆盖全部任务类型的 SFT 样本，
    且每条样本的 instruction 带 schema 标记/枚举清单、response 是能回读校验的合法 JSON——
    这是训练数据可用性的底线，坏样本会直接污染微调。"""
    builder = importlib.import_module("05_TRAINING.llm.build_sft_dataset")
    state = builder.load_demo_output(ROOT / "02_STORY_ENGINE" / "demo_output")
    out = tmp_path / "sft.jsonl"
    n = builder.write_jsonl(builder.samples_from_state(**state), out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert n == len(rows) >= 20
    tasks = {r["task"] for r in rows}
    assert {"story_bible", "character", "episode", "scene", "shots", "image_prompt", "video_prompt"} <= tasks
    # instruction 带 schema 标记和枚举清单，response 是能回读的合法 JSON
    shot_sample = next(r for r in rows if r["task"] == "shots")
    assert "[TARGET_SCHEMA=ShotsFile]" in shot_sample["instruction"] and "ShotSize" in shot_sample["instruction"]
    sch = importlib.import_module("03_STRUCTURED_DATA.schemas")
    assert sch.ShotsFile.model_validate_json(shot_sample["response"]).shots


def test_build_sft_dataset_from_db_after_pipeline(tmp_path):
    """依赖前面 e2e 用例在库里留下的 approved 集；单独跑时可能为 0 条，只验证不报错。"""
    builder = importlib.import_module("05_TRAINING.llm.build_sft_dataset")
    states = list(builder.load_from_db(only_approved=True))
    for state in states:
        assert all(ep.id.startswith("ep_") for ep in state["episodes"])  # 读回的是业务 id
    total = sum(1 for state in states for _ in builder.samples_from_state(**state))
    assert total >= 0


def test_evaluate_with_mock_provider(tmp_path):
    """验证评估脚本在 mock provider（总是产出合法 JSON）下能跑通，且各项通过率都是 100%，
    确认评估流程本身没有 bug（区别于评估真实模型时通过率不达标）。"""
    builder = importlib.import_module("05_TRAINING.llm.build_sft_dataset")
    evaluator = importlib.import_module("05_TRAINING.llm.evaluate_structured_output")
    state = builder.load_demo_output(ROOT / "02_STORY_ENGINE" / "demo_output")
    samples = list(builder.samples_from_state(**state))[:12]
    provider = importlib.import_module("02_STORY_ENGINE.demo").build_mock_provider()
    result = evaluator.evaluate(provider, samples)
    assert result["overall"]["n"] == 12
    assert result["overall"]["schema_valid_rate"] == 1.0 and result["overall"]["json_valid_rate"] == 1.0
    assert "story_bible" in result["per_task"]


def test_evaluate_counts_invalid_outputs():
    """验证评估脚本能正确识别"完全不是 JSON"的模型输出，把各项通过率算成 0 而不是抛异常
    或误判为通过——评估工具本身必须能处理最差情况。"""
    evaluator = importlib.import_module("05_TRAINING.llm.evaluate_structured_output")
    base = importlib.import_module("02_STORY_ENGINE.agents.base")

    class Bad(base.LLMProvider):
        """总是返回非 JSON 文本的假 provider，用来验证评估脚本对无效输出的统计。"""

        provider_name, model = "bad", "bad"

        def complete(self, system_prompt, user_prompt, history=None):
            """无视输入，固定返回一段不是 JSON 的文本。"""
            return "not json at all"

    result = evaluator.evaluate(Bad(), [{"task": "episode", "instruction": "x", "system_prompt_file": "episode_agent_system.txt"}], max_retries=1)
    assert result["overall"] == {"n": 1, "json_valid_rate": 0.0, "schema_valid_rate": 0.0, "first_try_rate": 0.0, "retry_rescued": 0}


def test_cli_scripts_run(tmp_path):
    """验证 build_sft_dataset.py / evaluate_structured_output.py 两个脚本能以命令行方式独立跑通
    （不只是被当作库 import），这是训练流程实际使用的入口，必须单独覆盖到。"""
    out = tmp_path / "sft.jsonl"
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "05_TRAINING/llm/build_sft_dataset.py"),
            "--from-demo-output",
            str(ROOT / "02_STORY_ENGINE/demo_output"),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    r = subprocess.run(
        [sys.executable, str(ROOT / "05_TRAINING/llm/evaluate_structured_output.py"), "--dataset", str(out), "--provider", "mock", "--limit", "5"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["overall"]["n"] == 5


def test_registry_lists_provider_defaults_and_character_loras_valid():
    """验证 provider 工厂默认使用的本地模型确实登记在 06_MODELS 的模型注册表里（许可证可查，
    不能用一个未登记来源不明的模型作为默认值），并核对角色 LoRA 登记表的字段完整性和命名约定。"""
    registry = importlib.import_module("06_MODELS.model_registry")
    factory = importlib.import_module("02_STORY_ENGINE.agents.provider_factory")
    all_regs = registry.load_all_registries()
    llm_ids = {e.hf_id_or_local_path for e in all_regs["llm"]}
    assert factory.DEFAULT_LOCAL_MODEL in llm_ids  # 默认本地模型必须在登记表里（许可证可查）
    loras = json.loads((ROOT / "06_MODELS" / "character_loras.json").read_text(encoding="utf-8"))
    for cid, entry in loras["characters"].items():
        assert cid.startswith("char_") and set(entry) >= {"lora_path", "reference_images"}
