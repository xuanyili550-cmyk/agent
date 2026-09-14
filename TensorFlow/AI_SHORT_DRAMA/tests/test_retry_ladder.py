"""三级重试阶梯：纯逻辑单测 + 接进 shot_task 的集成测试（dummy 生成器 + 假 QC）。"""

from __future__ import annotations

import importlib

import pytest

ladder_mod = importlib.import_module("07_GENERATION.retry_ladder")
errors = importlib.import_module("07_GENERATION.errors")
image_mod = importlib.import_module("07_GENERATION.image.image_generator")
http_retry = importlib.import_module("07_GENERATION.http_retry")
RetryLadder, RetryLevel, FailureKind = ladder_mod.RetryLadder, ladder_mod.RetryLevel, ladder_mod.FailureKind


def test_transient_failure_retries_same_params_first():
    ladder = RetryLadder(max_attempts=4)
    ladder.first_attempt()
    plan = ladder.next_attempt(FailureKind.TRANSIENT, "网络超时")
    assert plan.level == RetryLevel.SAME_PARAMS and (plan.width, plan.height) == (1080, 1920) and not plan.rewrite_prompt


def test_qc_rejection_goes_straight_to_prompt_rewrite_then_downscale():
    ladder = RetryLadder(max_attempts=4)
    ladder.first_attempt()
    p2 = ladder.next_attempt(FailureKind.QC_REJECTED, "角色相似度 0.4")
    assert p2.level == RetryLevel.REWRITE_PROMPT and p2.rewrite_prompt
    p3 = ladder.next_attempt(FailureKind.QC_REJECTED)
    assert p3.level == RetryLevel.DOWNSCALE and (p3.width, p3.height) == (720, 1280)
    p4 = ladder.next_attempt(FailureKind.QC_REJECTED)
    assert p4.level == RetryLevel.DOWNSCALE and (p4.width, p4.height) == (540, 960)
    assert ladder.next_attempt(FailureKind.QC_REJECTED) is None  # max_attempts=4 用尽
    assert [p["level"] for p in ladder.summary()] == ["initial", "rewrite_prompt", "downscale", "downscale"]


def test_resource_failure_downscales_immediately_and_stops_at_ladder_end():
    ladder = RetryLadder(resolution_ladder=[(1080, 1920), (720, 1280)], max_attempts=6)
    ladder.first_attempt()
    p = ladder.next_attempt(FailureKind.RESOURCE, "OOM")
    assert p.level == RetryLevel.DOWNSCALE and (p.width, p.height) == (720, 1280)
    assert ladder.next_attempt(FailureKind.RESOURCE) is None


def test_full_ladder_transient_then_rewrite_then_downscale():
    ladder = RetryLadder(max_attempts=4)
    ladder.first_attempt()
    levels = [
        ladder.next_attempt(FailureKind.TRANSIENT).level,
        ladder.next_attempt(FailureKind.TRANSIENT).level,  # 同参数只试一次，第二次瞬时错误升级到改写
        ladder.next_attempt(FailureKind.QC_REJECTED).level,
    ]
    assert levels == [RetryLevel.SAME_PARAMS, RetryLevel.REWRITE_PROMPT, RetryLevel.DOWNSCALE]


def test_http_retry_backoff_and_error_mapping(monkeypatch):
    calls = []

    class Resp:
        def __init__(self, status, text="", headers=None):
            self.status_code, self.text, self.headers, self.content = status, text, headers or {}, b""

        def raise_for_status(self):
            raise http_retry.requests.HTTPError(str(self.status_code))

    def fake_request(method, url, timeout, **kw):
        calls.append(url)
        return responses.pop(0)

    monkeypatch.setattr(http_retry.requests, "request", fake_request)
    sleeps = []
    responses = [Resp(503), Resp(429, headers={"Retry-After": "2"}), Resp(200)]
    resp = http_retry.request_with_retry("GET", "http://x", retries=3, sleep=sleeps.append)
    assert resp.status_code == 200 and len(calls) == 3 and sleeps[1] == 2.0

    responses = [Resp(400, "bad prompt")]
    with pytest.raises(errors.GenerationRejectedError):
        http_retry.request_with_retry("POST", "http://x", sleep=lambda s: None)
    responses = [Resp(401)]
    with pytest.raises(errors.NotConfiguredError):
        http_retry.request_with_retry("POST", "http://x", sleep=lambda s: None)
    responses = [Resp(500), Resp(500), Resp(500), Resp(500)]
    with pytest.raises(errors.TransientProviderError):
        http_retry.request_with_retry("POST", "http://x", retries=3, sleep=lambda s: None)


# ---- shot_task 集成 --------------------------------------------------------------------


@pytest.fixture
def shot_env(tmp_path):
    shot_task = importlib.import_module("13_INFRA.queue.shot_task")
    qc = importlib.import_module("08_QC.reports.qc_report")
    config = importlib.import_module("13_INFRA.config")
    settings = config.get_settings()

    def make_scorer(decisions):
        seq = list(decisions)

        def scorer(*, generated_image, reference_images, shot_id, asset_id, character_id):
            decision = seq.pop(0) if seq else "approved"
            if decision == "approved":
                return qc.build_qc_report(asset_id=asset_id, shot_id=shot_id, thresholds=qc.QCThresholds(video_required=False, audio_required=False))
            return qc.QCReport(
                asset_id=asset_id, shot_id=shot_id, decision=qc.QCDecision.RETRY, retry_reasons=["character similarity 0.41 below threshold 0.75"]
            )

        return scorer

    def run(payload, generator, decisions=(), rewriter=None):
        base = {"shot_id": "shot_test", "prompt": "苏晚晚在宴会厅对峙，血迹", "output_dir": str(tmp_path / "out"), "reference_character_ids": []}
        return shot_task.produce_shot({**base, **payload}, settings=settings, generator=generator, scorer=make_scorer(decisions), rewriter=rewriter)

    return run


def test_shot_task_qc_reject_triggers_prompt_rewrite(shot_env):
    gen = image_mod.DummyImageGenerator()
    rewrites = []

    def rewriter(prompt, reasons, attempt):
        rewrites.append((prompt, reasons, attempt))
        return prompt.replace("血迹", "") + "，高清"

    result = shot_env({}, gen, decisions=["retry", "approved"], rewriter=rewriter)
    assert result["status"] == "approved"
    assert [a["level"] for a in result["attempts"]] == ["initial", "rewrite_prompt"]
    assert rewrites and "0.41" in rewrites[0][1][0]
    assert "血迹" not in gen.calls[1]["prompt"] and gen.calls[1]["prompt"].endswith("高清")
    assert result["attempts"][0]["qc"] == "retry" and result["attempts"][1]["qc"] == "approved"


def test_shot_task_transient_then_success(shot_env):
    gen = image_mod.DummyImageGenerator(fail_on={"transient": 1})
    result = shot_env({"seed": 7}, gen)
    assert result["status"] == "approved"
    assert [a["level"] for a in result["attempts"]] == ["initial", "same_params"]
    assert gen.calls[0]["seed"] == 7 and gen.calls[1]["seed"] != 7  # 同参数重试换 seed
    assert (gen.calls[1]["width"], gen.calls[1]["height"]) == (256, 448)


def test_shot_task_resource_error_downscales(shot_env):
    gen = image_mod.DummyImageGenerator(fail_on={"resource": 1})
    result = shot_env({}, gen)
    assert result["status"] == "approved"
    assert [a["level"] for a in result["attempts"]] == ["initial", "downscale"]
    assert (gen.calls[1]["width"], gen.calls[1]["height"]) == (192, 336)


def test_shot_task_ladder_exhausted_returns_failed_without_raising(shot_env):
    gen = image_mod.DummyImageGenerator()
    result = shot_env({}, gen, decisions=["retry", "retry", "retry", "retry", "retry"], rewriter=lambda p, r, a: p + "!")
    assert result["status"] == "failed" and result["asset"] is None
    assert [a["level"] for a in result["attempts"]] == ["initial", "rewrite_prompt", "downscale", "downscale"]
    assert result["final_reasons"]


def test_rule_based_rewrite_strips_risky_words():
    shot_task = importlib.import_module("13_INFRA.queue.shot_task")
    out = shot_task.rule_based_rewrite("血腥的暴力场面，nude", ["safety"])
    assert "血" not in out and "nude" not in out and "高清" in out
