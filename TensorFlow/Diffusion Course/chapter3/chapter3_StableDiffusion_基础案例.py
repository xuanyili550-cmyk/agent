"""
 Diffusion Course · Ch3 · 基础案例：Stable Diffusion 文生图（🔴 需 GPU/下 SD 模型,参考）
 本机(尤其无 CUDA)不建议直接跑;这是真实可用写法。跑(有卡时)：python3 本文件
"""
import torch
from diffusers import StableDiffusionPipeline

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
image = pipe("a cozy cabin in the snow, digital art", guidance_scale=7.5, num_inference_steps=25).images[0]
image.save("/tmp/sd_case.png")
print("✅ 已生成 /tmp/sd_case.png (guidance_scale=7.5)")
