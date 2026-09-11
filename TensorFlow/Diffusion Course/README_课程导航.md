# Diffusion Course · 课程导航

> 每个文件顶部有完整「学习笔记描述」。多为 HF 官方教程的 notebook 摘录/转写，
> 以**读懂原理**为主，不少需特定环境(GPU/gym/Unity/Blender 等)，不保证本机直接跑。

| 章 | 文件 | 一句话 |
|:--:|------|--------|
| 1 | `IntroductiontoDiffusers.py` | 扩散模型=学"从噪声一步步去噪成图"——本章用 diffusers 跑通 DDPM 全流程。 |
| 2 | `Fine-Tuning and Guidance.py` | 把预训练扩散模型微调到新画风 + 用"引导"控制生成方向。 |
| 3 | `Stable Diffusion.py` | SD = 在潜空间做扩散(VAE 压缩 + 文本条件 UNet + CLIP 文本编码)，又快又强。 |
| 4 | `DDIM Inversion.py` | DDIM 反演 = 把一张真实图"反推回噪声",再带新提示去噪→实现精准图像编辑。 |
