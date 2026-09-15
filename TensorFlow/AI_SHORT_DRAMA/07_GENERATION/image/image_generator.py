"""图像生成。

DiffusersImageGenerator 是真实的本地 diffusers 封装（需要 GPU + 已下载权重），支持两种
跨镜头角色一致性手段：
- ``lora_path``：角色 LoRA（05_TRAINING/character_lora 训练出来的权重）；
- ``reference_images``：IP-Adapter 参考图——把角色定妆照作为图像条件注入，不需要训练。
两者可以叠加。DummyImageGenerator 画一张带文字标注的占位 PNG，让流水线在没有权重时也能
离线冒烟测试。

流水线位置：07_GENERATION 的第一步——由 03_STRUCTURED_DATA 的 ImagePrompt 生成每个镜头的
关键帧图，随后作为 video_generator.py 图生视频的输入，并交给 08_QC 做角色一致性 / 场景对齐检查。
为什么做 Provider 抽象：真实生成需要 GPU 和几 GB 权重，CI 与本地开发机跑不了；
统一 ``generate(...) -> PIL.Image`` 接口后，Dummy 实现可以在任何机器上把下游全部跑通，
还能通过 ``fail_on`` 模拟各类失败来测试三级重试阶梯。
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04_DATASET" / "images"))
from errors import GenerationRejectedError, ResourceExhaustedError, TransientProviderError  # noqa: E402
from generate_placeholder_images import COLORS  # noqa: E402

PathLike = str | Path


class BaseImageGenerator(ABC):
    """图像生成器的抽象接口；真实实现和 Dummy 实现签名完全一致，编排层可互换。"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        lora_path: Optional[str] = None,
        reference_images: Optional[Sequence[PathLike]] = None,
        ip_adapter_scale: float = 0.6,
    ) -> Image.Image:
        """按 prompt 生成一张图；``lora_path`` / ``reference_images`` 用于跨镜头角色一致性。"""
        ...


class DiffusersImageGenerator(BaseImageGenerator):
    """封装 diffusers.AutoPipelineForText2Image。需要 torch + GPU（或很慢的 CPU 推理）
    和已下载/缓存的模型权重——离线 demo 不会走到这里。

    IP-Adapter 权重默认用 h94/IP-Adapter 的 SDXL 版；换基础模型时把 ``ip_adapter_repo`` /
    ``ip_adapter_subfolder`` / ``ip_adapter_weight`` 一起换掉。
    """

    def __init__(
        self,
        model_id_or_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        ip_adapter_repo: str = "h94/IP-Adapter",
        ip_adapter_subfolder: str = "sdxl_models",
        ip_adapter_weight: str = "ip-adapter_sdxl.bin",
    ):
        """只记录配置；pipeline、IP-Adapter、LoRA 都延迟到需要时再加载。

        ``_loaded_lora`` 记录当前挂着的 LoRA 路径，用来在切换角色时正确卸载。
        """
        self.model_id_or_path = model_id_or_path
        self.device = device
        self.dtype = dtype
        self.ip_adapter_repo = ip_adapter_repo
        self.ip_adapter_subfolder = ip_adapter_subfolder
        self.ip_adapter_weight = ip_adapter_weight
        self._pipe = None
        self._ip_adapter_loaded = False
        self._loaded_lora: str | None = None

    def _load(self):
        """懒加载并缓存 text2image pipeline；dtype 用字符串配置是为了能从 YAML / 环境变量直接读。"""
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import AutoPipelineForText2Image

        torch_dtype = getattr(torch, self.dtype)
        pipe = AutoPipelineForText2Image.from_pretrained(self.model_id_or_path, torch_dtype=torch_dtype)
        pipe = pipe.to(self.device)
        self._pipe = pipe
        return pipe

    def _ensure_ip_adapter(self, pipe) -> None:
        """第一次需要参考图时才加载 IP-Adapter 权重，之后复用（加载一次约几百 MB）。"""
        if not self._ip_adapter_loaded:
            pipe.load_ip_adapter(self.ip_adapter_repo, subfolder=self.ip_adapter_subfolder, weight_name=self.ip_adapter_weight)
            self._ip_adapter_loaded = True

    def _ensure_lora(self, pipe, lora_path: Optional[str]) -> None:
        """让 pipeline 上挂载的 LoRA 与本次请求一致：同一个则跳过，不同则先卸再载。"""
        # 换角色时卸掉上一个 LoRA，否则两个角色的权重叠加在一起长相会串
        if lora_path == self._loaded_lora:
            return
        if self._loaded_lora is not None:
            pipe.unload_lora_weights()
            self._loaded_lora = None
        if lora_path:
            pipe.load_lora_weights(lora_path)
            self._loaded_lora = lora_path

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        lora_path: Optional[str] = None,
        reference_images: Optional[Sequence[PathLike]] = None,
        ip_adapter_scale: float = 0.6,
    ) -> Image.Image:
        """真实推理：挂 LoRA / IP-Adapter 后跑 pipeline，并把两类底层失败翻译成项目异常。

        - CUDA OOM -> ResourceExhaustedError（三级重试第 3 级：降分辨率）；
        - 安全检查器拦截 -> GenerationRejectedError（第 2 级：改写提示词）。
        """
        import torch

        pipe = self._load()
        self._ensure_lora(pipe, lora_path)
        kwargs = {}
        if reference_images:
            self._ensure_ip_adapter(pipe)
            pipe.set_ip_adapter_scale(ip_adapter_scale)
            refs = [Image.open(p).convert("RGB") for p in reference_images]
            # diffusers 单张参考图接受 Image，多张接受 list；不要把单张包成 list 以免被当成 batch
            kwargs["ip_adapter_image"] = refs if len(refs) > 1 else refs[0]
        elif self._ip_adapter_loaded:
            pipe.set_ip_adapter_scale(0.0)  # 已加载但本镜头没有参考图：把权重关掉

        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()  # 释放碎片，否则降分辨率后的重试也可能立刻再 OOM
            # 交给三级重试的第三级：降分辨率
            raise ResourceExhaustedError(f"CUDA 显存不足（{width}x{height}）：{exc}") from exc
        image = result.images[0]
        # diffusers 安全检查器判 NSFW 时返回全黑图 + nsfw_content_detected=True：当作 prompt 被拒
        flagged = getattr(result, "nsfw_content_detected", None)
        if flagged and any(flagged):
            raise GenerationRejectedError("安全检查器拦截了生成结果，请改写提示词")
        return image


class DummyImageGenerator(BaseImageGenerator):
    """不需要权重、不需要 GPU。画一张纯色 + 文字的占位图，让流水线其余部分
    （素材记录、文件布局、QC、后期）可以离线跑通。

    ``fail_on`` 用于测试三级重试：{"transient": n} 表示前 n 次调用抛 TransientProviderError。
    """

    def __init__(self, fail_on: dict[str, int] | None = None):
        """``fail_on`` 复制一份以免调用方的 dict 被就地递减；``calls`` 记录每次调用参数供测试断言。"""
        self.fail_on = dict(fail_on or {})
        self.calls: list[dict] = []

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 1,
        guidance_scale: float = 0.0,
        lora_path: Optional[str] = None,
        reference_images: Optional[Sequence[PathLike]] = None,
        ip_adapter_scale: float = 0.6,
    ) -> Image.Image:
        """记录调用参数，按 ``fail_on`` 配额模拟失败，否则返回带 prompt / 参考图 / LoRA 标注的占位图。"""
        self.calls.append({"prompt": prompt, "seed": seed, "width": width, "height": height, "refs": len(reference_images or []), "lora": lora_path})
        # 按 transient -> rejected -> resource 的顺序消耗失败配额，让测试能精确控制第几次抛哪种异常
        for kind, cls in (("transient", TransientProviderError), ("rejected", GenerationRejectedError), ("resource", ResourceExhaustedError)):
            if self.fail_on.get(kind, 0) > 0:
                self.fail_on[kind] -= 1
                raise cls(f"DummyImageGenerator 模拟 {kind} 失败")
        color = COLORS[(seed or 0) % len(COLORS)]  # seed 决定底色：同 seed 同图，方便肉眼核对可复现性
        img = Image.new("RGB", (width, height), color=color)
        draw = ImageDraw.Draw(img)
        label = prompt[:200]
        if reference_images:
            label += f"\n[ref x{len(reference_images)}]"
        if lora_path:
            label += f"\n[lora {Path(lora_path).name}]"
        draw.multiline_text((16, 16), label, fill=(255, 255, 255))
        return img
