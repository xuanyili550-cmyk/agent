# ML for 3D · 课程导航

> 每个文件顶部有完整「学习笔记描述」。多为 HF 官方教程的 notebook 摘录/转写，
> 以**读懂原理**为主，不少需特定环境(GPU/gym/Unity/Blender 等)，不保证本机直接跑。

| 章 | 文件 | 一句话 |
|:--:|------|--------|
| 2 | `Generative 3D pipelines.py` | 从"一张图/一句话"生成 3D——先用多视图扩散生成多个视角,再重建成 3D。 |
| 3 | `Differentiable Rasterization.py` | 把"3D→2D 投影渲染"做成可微的,梯度就能回传→用图像监督优化 3D 表示。 |
| 5 | `LGM-tiny.py` | LGM 把"多视图扩散 + 高斯泼溅"打包成 image→3D 的端到端 pipeline,配 Gradio demo。 |
