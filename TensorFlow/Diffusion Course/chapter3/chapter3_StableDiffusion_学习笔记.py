"""
================================================================================
 Diffusion Course · Chapter 3 · Stable Diffusion（学习笔记 · 潜空间概念 numpy 可跑）
================================================================================
 一句话：SD = 在"潜空间"做扩散(VAE 压缩 + 文本条件 UNet + CLIP 文本编码),又快又强。
 本章讲：
   ① 潜空间(latent)：VAE 把 512×512×3 图压成 64×64×4 潜表示 → 扩散在小图上做,省算力(本文件 numpy 看维度)。
   ② 文本条件：CLIP 编码提示 → 交叉注意力注入 UNet,引导生成内容。
   ③ 各 pipeline：文生图 / 图生图 / inpaint(局部重绘) / depth2img。
 要点：SD 在低维潜空间扩散 → 消费级卡也能跑;guidance_scale/步数/提示 决定出图质量。
 跑：python3 chapter3_StableDiffusion_学习笔记.py   （①维度演示纯 numpy 真跑;真实生成 🔴需GPU/下SD)
================================================================================
"""
import numpy as np


def latent_shrink(h=512, w=512, factor=8, latent_ch=4):
    """VAE 下采样倍数(SD 为 8)：像素 HxWx3 → 潜空间 (H/8)x(W/8)x4。算下省了多少。"""
    pixel = h * w * 3
    latent = (h // factor) * (w // factor) * latent_ch
    return pixel, latent, round(pixel / latent, 1)


def txt2img_real():   # 🔴 需 GPU/下 SD 模型,默认不调用
    import torch
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                   torch_dtype=torch.float16)
    return pipe("a cat astronaut, digital art", guidance_scale=7.5).images[0]


def main():
    pixel, latent, ratio = latent_shrink()
    assert ratio > 40   # 潜空间元素数远小于像素数
    print(f"✅ Ch3 跑通：潜空间压缩 像素{pixel} → 潜{latent}(约省 {ratio}×,所以 SD 快)")
    print("   pipeline：txt2img / img2img / inpaint / depth2img;文本靠 CLIP+交叉注意力注入。")
    # 面试：Q SD 为何比像素扩散快? A 在 64×64 潜空间做而非 512×512 像素; Q 文本怎么进? A CLIP 编码+交叉注意力。


if __name__ == "__main__":
    main()
