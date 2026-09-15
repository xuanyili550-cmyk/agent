"""
视频生成 provider 抽象。RunwayVideoProvider 按 Runway 开发者 API（image_to_video +
tasks 轮询 + 下载产物）完整实现；PikaVideoProvider 没有公开稳定的 API 契约，保留为骨架。
两者都在构造时检查 API key，缺失直接抛 NotConfiguredError；HTTP 调用统一走
http_retry.request_with_retry（429/5xx 指数退避，401/403 直接报凭证错误）。

流水线位置：07_GENERATION 的"图生视频"环节——image_generator.py 产出的关键帧图 + 镜头 prompt
在这里变成竖屏视频片段，随后交给 lipsync_generator.py 对口型、08_QC 质检。
为什么做 Provider 抽象：云端 API（Runway / Pika）和本地开源模型（Mochi）的调用方式完全不同，
但对编排层来说都是 ``generate(prompt, image, duration, seed) -> mp4 路径``；VideoGenerator
门面只负责调用 provider 和登记素材，换 provider 不需要动编排层，也便于用 Dummy 做离线测试。
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asset_registry import AssetRecord, new_asset_id, now_iso, write_asset_record  # noqa: E402
from errors import GenerationRejectedError, NotConfiguredError, TransientProviderError  # noqa: E402
from http_retry import request_with_retry  # noqa: E402


class BaseVideoProvider(ABC):
    """视频生成 provider 的抽象接口。"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        duration_sec: float = 4.0,
        seed: Optional[int] = None,
        output_dir: str = "./outputs",
    ) -> str:
        """返回生成视频的本地文件路径。"""


class RunwayVideoProvider(BaseVideoProvider):
    """Runway 开发者 API：POST image_to_video 建任务 -> GET tasks/{id} 轮询 -> 下载 output[0]。

    promptImage 必须是模型能访问的 URL 或 data URI；本地文件会自动转成 data URI 上传。
    Runway 只支持 5/10 秒两档时长，duration_sec 会向上取整到这两档。
    """

    API_BASE = "https://api.dev.runwayml.com/v1"
    API_VERSION = "2024-11-06"

    def __init__(self, model: str = "gen3a_turbo", poll_interval_sec: float = 5.0, max_wait_sec: float = 600.0):
        """从 RUNWAY_API_KEY 读取凭证；``max_wait_sec`` 是轮询任务的总超时，超时按瞬时错误处理。"""
        self.api_key = os.environ.get("RUNWAY_API_KEY")
        if not self.api_key:
            raise NotConfiguredError("RUNWAY_API_KEY is not set")
        self.model = model
        self.poll_interval_sec = poll_interval_sec
        self.max_wait_sec = max_wait_sec

    def _headers(self) -> dict:
        """Runway 要求显式带 X-Runway-Version 头固定 API 版本，避免服务端升级导致字段变化。"""
        return {"Authorization": f"Bearer {self.api_key}", "X-Runway-Version": self.API_VERSION, "Content-Type": "application/json"}

    @staticmethod
    def _image_ref(image_path: str) -> str:
        """把关键帧图转成 Runway 能读取的引用：URL / data URI 原样返回，本地文件 base64 内联。"""
        if image_path.startswith(("http://", "https://", "data:")):
            return image_path
        import base64
        import mimetypes

        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        data = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        duration_sec: float = 4.0,
        seed: Optional[int] = None,
        output_dir: str = "./outputs",
    ) -> str:
        """建任务 -> 轮询直到 SUCCEEDED -> 下载第一个产物到 output_dir/<task_id>.mp4。

        失败码分流：SAFETY / INPUT_PREPROCESSING 属于 prompt 或输入被拒（进第 2 级改写），
        其他失败和超时都当瞬时错误（进第 1 级重试）。
        """
        if not image_path:
            raise GenerationRejectedError("Runway image_to_video 需要关键帧图片（image_path）")
        payload = {
            "model": self.model,
            "promptText": prompt[:1000],  # Runway 对 promptText 有长度上限
            "promptImage": self._image_ref(image_path),
            "duration": 5 if duration_sec <= 5 else 10,  # 只有 5s / 10s 两档
            "ratio": "768:1280",  # 竖屏 9:16
        }
        if seed is not None:
            payload["seed"] = int(seed) % (2**32)  # Runway 要求 32 位无符号 seed

        created = request_with_retry("POST", f"{self.API_BASE}/image_to_video", json=payload, headers=self._headers()).json()
        task_id = created["id"]

        deadline = time.monotonic() + self.max_wait_sec
        while True:
            task = request_with_retry("GET", f"{self.API_BASE}/tasks/{task_id}", headers=self._headers()).json()
            status = task.get("status")
            if status == "SUCCEEDED":
                break
            if status in ("FAILED", "CANCELLED"):
                failure = task.get("failure") or task.get("failureCode") or "unknown"
                # Runway 的内容审核失败码以 SAFETY 开头：改写提示词再试；其余当瞬时错误
                if "SAFETY" in str(failure).upper() or "INPUT_PREPROCESSING" in str(failure).upper():
                    raise GenerationRejectedError(f"Runway 拒绝了生成：{failure}")
                raise TransientProviderError(f"Runway 任务失败：{failure}")
            if time.monotonic() > deadline:
                raise TransientProviderError(f"Runway 任务 {task_id} 超过 {self.max_wait_sec}s 未完成")
            time.sleep(self.poll_interval_sec)

        outputs = task.get("output") or []
        if not outputs:
            raise TransientProviderError(f"Runway 任务 {task_id} 成功但没有产物 URL")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{task_id}.mp4"
        video = request_with_retry("GET", outputs[0], timeout=300)  # 视频文件较大，下载超时放宽
        out_path.write_bytes(video.content)
        return str(out_path)


class PikaVideoProvider(BaseVideoProvider):
    """Pika API 客户端骨架：只保证请求带退避重试，响应解析和产物下载尚未实现。"""

    API_URL = "https://api.pika.art/v1/generate"

    def __init__(self):
        """从 PIKA_API_KEY 读取凭证；缺失立即报错。"""
        self.api_key = os.environ.get("PIKA_API_KEY")
        if not self.api_key:
            raise NotConfiguredError("PIKA_API_KEY is not set")

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        duration_sec: float = 4.0,
        seed: Optional[int] = None,
        output_dir: str = "./outputs",
    ) -> str:
        """提交生成请求，返回按任务 id 约定的输出路径（骨架实现，不下载产物）。"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"prompt": prompt, "duration": duration_sec, "seed": seed}
        if image_path:
            payload["image"] = image_path

        # Pika 没有公开稳定的 API 契约：这里只保证请求带退避重试，响应解析仍是骨架
        response = request_with_retry("POST", self.API_URL, json=payload, headers=headers)
        task = response.json()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{task.get('id', uuid.uuid4().hex)}.mp4"
        return str(out_path)


class MochiLocalVideoProvider(BaseVideoProvider):
    """通过 diffusers 在本地运行 Genmo Mochi 1（Apache-2.0，免费且可商用）——
    无需 API key，但需要大显存 GPU 和已缓存到本地的权重
    （见 06_MODELS/video/registry.json: genmo/mochi-1-preview）。
    image_path 参数只是为了与 API provider 接口一致而保留，实际未使用：
    这里的 Mochi 是文生视频，不是图生视频。"""

    def __init__(self, model_id: str = "genmo/mochi-1-preview", device: str = "cuda", fps: int = 15):
        """只记录配置；pipeline 延迟到第一次 generate 时再加载。"""
        self.model_id = model_id
        self.device = device
        self.fps = fps
        self._pipe = None

    def _load(self):
        """懒加载并缓存 MochiPipeline；bfloat16 + CPU offload 是为了把显存占用压到消费级显卡能跑的范围。"""
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import MochiPipeline

        pipe = MochiPipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        self._pipe = pipe
        return pipe

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        duration_sec: float = 4.0,
        seed: Optional[int] = None,
        output_dir: str = "./outputs",
    ) -> str:
        """按 duration_sec * fps 算帧数做文生视频，导出 mp4；seed 固定时结果可复现。"""
        import torch
        from diffusers.utils import export_to_video

        pipe = self._load()
        # generator 放在 CPU 上：与 device 无关的确定性采样，且 CPU offload 模式下更稳
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        num_frames = max(1, round(duration_sec * self.fps))
        result = pipe(prompt=prompt, num_frames=num_frames, generator=generator)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{uuid.uuid4().hex}.mp4"
        export_to_video(result.frames[0], str(out_path), fps=self.fps)
        return str(out_path)


class VideoGenerator:
    """视频生成门面：调用 provider 并把产物登记到素材台账，编排层只依赖这一层。"""

    def __init__(self, provider: BaseVideoProvider, asset_log_path: str = "./outputs/asset_records.jsonl"):
        """注入 provider 与台账路径。"""
        self.provider = provider
        self.asset_log_path = asset_log_path

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        duration_sec: float = 4.0,
        seed: Optional[int] = None,
        output_dir: str = "./outputs",
        character_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        shot_id: Optional[str] = None,
        license: Optional[str] = None,
    ) -> str:
        """生成视频并写 AssetRecord（model 字段记 provider 类名以便追溯）；返回产物路径。"""
        file_path = self.provider.generate(prompt=prompt, image_path=image_path, duration_sec=duration_sec, seed=seed, output_dir=output_dir)
        record = AssetRecord(
            asset_id=new_asset_id("vid"),
            file_path=file_path,
            character_id=character_id,
            episode_id=episode_id,
            shot_id=shot_id,
            model=type(self.provider).__name__,
            prompt=prompt,
            seed=seed,
            license=license,
            created_at=now_iso(),
        )
        write_asset_record(record, self.asset_log_path)
        return file_path
