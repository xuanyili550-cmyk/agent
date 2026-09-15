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
    """验证阶梯设计的第一条约束：瞬时错误（网络超时等）先原分辨率重试，不立刻降级/改写。"""
    ladder = RetryLadder(max_attempts=4)
    ladder.first_attempt()
    plan = ladder.next_attempt(FailureKind.TRANSIENT, "网络超时")
    assert plan.level == RetryLevel.SAME_PARAMS and (plan.width, plan.height) == (1080, 1920) and not plan.rewrite_prompt


def test_qc_rejection_goes_straight_to_prompt_rewrite_then_downscale():
    """验证 QC 不通过时跳过"同参数重试"直接改写提示词（同参数重试对 QC 不通过没意义），
    改写仍不过才降分辨率，且阶梯用尽后 next_attempt 返回 None 而不是继续生成。"""
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
    """验证资源不足（如显存 OOM）直接降分辨率，不浪费一次尝试在"同参数重试"上；
    分辨率阶梯用完后停止而不是抛异常或无限重试。"""
    ladder = RetryLadder(resolution_ladder=[(1080, 1920), (720, 1280)], max_attempts=6)
    ladder.first_attempt()
    p = ladder.next_attempt(FailureKind.RESOURCE, "OOM")
    assert p.level == RetryLevel.DOWNSCALE and (p.width, p.height) == (720, 1280)
    assert ladder.next_attempt(FailureKind.RESOURCE) is None


def test_full_ladder_transient_then_rewrite_then_downscale():
    """验证跨失败类型时阶梯状态是累加的：先瞬时错误重试一次，第二次瞬时错误不再原地重试
    （避免死循环），升级为改写提示词；符合"同参数只给一次机会"的设计。"""
    ladder = RetryLadder(max_attempts=4)
    ladder.first_attempt()
    levels = [
        ladder.next_attempt(FailureKind.TRANSIENT).level,
        ladder.next_attempt(FailureKind.TRANSIENT).level,  # 同参数只试一次，第二次瞬时错误升级到改写
        ladder.next_attempt(FailureKind.QC_REJECTED).level,
    ]
    assert levels == [RetryLevel.SAME_PARAMS, RetryLevel.REWRITE_PROMPT, RetryLevel.DOWNSCALE]


def test_http_retry_backoff_and_error_mapping(monkeypatch):
    """验证 http_retry.request_with_retry 的两件事：
    1) 503/429 会退避重试，429 的 Retry-After 头要被读出来作为等待秒数；
    2) HTTP 状态码正确映射到三种业务异常（400 拒绝/401 未配置/持续 5xx 视为瞬时错误耗尽重试）。
    """
    calls = []

    class Resp:
        """假的 requests.Response：只实现被测代码用到的 status_code/text/headers/content/raise_for_status。"""

        def __init__(self, status, text="", headers=None):
            """记录状态码、响应体、响应头；content 固定为空字节，被测代码不关心它的内容。"""
            self.status_code, self.text, self.headers, self.content = status, text, headers or {}, b""

        def raise_for_status(self):
            """模拟 requests 在非 2xx 时抛出的 HTTPError，消息里带上状态码方便断言。"""
            raise http_retry.requests.HTTPError(str(self.status_code))

    def fake_request(method, url, timeout, **kw):
        """替换 requests.request：按调用顺序依次弹出预先排好的响应列表，不发真实网络请求。"""
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
    """搭一套"调用 produce_shot"的最小环境，返回一个 run(payload, generator, decisions, rewriter)
    辅助函数，测试只需要关心生成器行为和预设的 QC 判定序列，不用每次都拼 settings/scorer。"""
    shot_task = importlib.import_module("13_INFRA.queue.shot_task")
    qc = importlib.import_module("08_QC.reports.qc_report")
    config = importlib.import_module("13_INFRA.config")
    settings = config.get_settings()

    def make_scorer(decisions):
        """按预设的 decisions 序列依次返回 approved/retry 的假 QC 结果；序列用完后默认 approved。"""
        seq = list(decisions)

        def scorer(*, generated_image, reference_images, shot_id, asset_id, character_id):
            """一次假打分：从预设序列里弹出这次该给的判定，构造对应的 QCReport。"""
            decision = seq.pop(0) if seq else "approved"
            if decision == "approved":
                return qc.build_qc_report(asset_id=asset_id, shot_id=shot_id, thresholds=qc.QCThresholds(video_required=False, audio_required=False))
            return qc.QCReport(
                asset_id=asset_id, shot_id=shot_id, decision=qc.QCDecision.RETRY, retry_reasons=["character similarity 0.41 below threshold 0.75"]
            )

        return scorer

    def run(payload, generator, decisions=(), rewriter=None):
        """套上默认 payload 后调用 produce_shot；decisions 控制每次尝试 QC 是否通过。"""
        base = {"shot_id": "shot_test", "prompt": "苏晚晚在宴会厅对峙，血迹", "output_dir": str(tmp_path / "out"), "reference_character_ids": []}
        return shot_task.produce_shot({**base, **payload}, settings=settings, generator=generator, scorer=make_scorer(decisions), rewriter=rewriter)

    return run


def test_shot_task_qc_reject_triggers_prompt_rewrite(shot_env):
    """验证 QC 拒绝后 produce_shot 真的调用了改写器、把拒绝原因传给它，且改写后的 prompt
    确实被用在下一次生成调用里（而不是原地重试原 prompt）。"""
    gen = image_mod.DummyImageGenerator()
    rewrites = []

    def rewriter(prompt, reasons, attempt):
        """记录每次改写调用的入参，便于断言；改写逻辑很简单：去掉敏感词、加清晰度后缀。"""
        rewrites.append((prompt, reasons, attempt))
        return prompt.replace("血迹", "") + "，高清"

    result = shot_env({}, gen, decisions=["retry", "approved"], rewriter=rewriter)
    assert result["status"] == "approved"
    assert [a["level"] for a in result["attempts"]] == ["initial", "rewrite_prompt"]
    assert rewrites and "0.41" in rewrites[0][1][0]
    assert "血迹" not in gen.calls[1]["prompt"] and gen.calls[1]["prompt"].endswith("高清")
    assert result["attempts"][0]["qc"] == "retry" and result["attempts"][1]["qc"] == "approved"


def test_shot_task_transient_then_success(shot_env):
    """验证瞬时错误重试时种子会换（同参数指 prompt/分辨率不变，不含 seed），分辨率保持不变。"""
    gen = image_mod.DummyImageGenerator(fail_on={"transient": 1})
    result = shot_env({"seed": 7}, gen)
    assert result["status"] == "approved"
    assert [a["level"] for a in result["attempts"]] == ["initial", "same_params"]
    assert gen.calls[0]["seed"] == 7 and gen.calls[1]["seed"] != 7  # 同参数重试换 seed
    assert (gen.calls[1]["width"], gen.calls[1]["height"]) == (256, 448)


def test_shot_task_resource_error_downscales(shot_env):
    """验证资源不足错误经 produce_shot 一路传到阶梯并触发降分辨率，且用的是 settings 里配置的阶梯。"""
    gen = image_mod.DummyImageGenerator(fail_on={"resource": 1})
    result = shot_env({}, gen)
    assert result["status"] == "approved"
    assert [a["level"] for a in result["attempts"]] == ["initial", "downscale"]
    assert (gen.calls[1]["width"], gen.calls[1]["height"]) == (192, 336)


def test_shot_task_ladder_exhausted_returns_failed_without_raising(shot_env):
    """验证阶梯用尽时 produce_shot 返回 status=failed 而不抛异常——见 shot_task 模块 docstring：
    chord 里一个镜头失败不该让整集其它镜头白做。"""
    gen = image_mod.DummyImageGenerator()
    result = shot_env({}, gen, decisions=["retry", "retry", "retry", "retry", "retry"], rewriter=lambda p, r, a: p + "!")
    assert result["status"] == "failed" and result["asset"] is None
    assert [a["level"] for a in result["attempts"]] == ["initial", "rewrite_prompt", "downscale", "downscale"]
    assert result["final_reasons"]


def test_rule_based_rewrite_strips_risky_words():
    """验证没有 LLM 时的兜底改写：能去掉常见风险词，并补上画质描述后缀。"""
    shot_task = importlib.import_module("13_INFRA.queue.shot_task")
    out = shot_task.rule_based_rewrite("血腥的暴力场面，nude", ["safety"])
    assert "血" not in out and "nude" not in out and "高清" in out
