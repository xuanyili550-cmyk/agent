# Computer Vision Course · 课程导航

> 每个文件顶部有完整「学习笔记描述」。多为 HF 官方教程的 notebook 摘录/转写，
> 以**读懂原理**为主，不少需特定环境(GPU/gym/Unity/Blender 等)，不保证本机直接跑。

| 章 | 文件 | 一句话 |
|:--:|------|--------|
| 1 | `Feature Matching.py` | 在两张图里找"同一个点/区域"的经典视觉技术——检测关键点、算描述子、再匹配。 |
| 2 | `Introduction to Convolutional Neural.py` | CV 的地基——卷积提局部特征、池化降维、堆叠成深网(LeNet→VGG→ResNet)。 |
| 3 | `Swin Transformer.py` | 把 Transformer 用到视觉的高效做法——移动窗口(Shifted Window)注意力，线性复杂度。 |
| 4 | `A Multimodal World.py` | 真实信息是多模态的(视觉+文本+音频)，本章概览模态组合、数据集与任务。 |
| 5 | `Variational Autoencoders.py` | 图像生成三条经典路线——GAN(对抗)、自编码器(重构)、VAE(带概率的编解码)。 |
| 6 | `Object Detection.py` | 不仅"是什么"(分类)还要"在哪"(定位)——目标检测 = 分类 + 定位；分割到像素级。 |
| 7 | `Multimodal Based Video Models.py` | 视频 = 图像序列 + 声音/文本/动作，多模态视频模型要同时建模"时间 + 多模态"。 |
| 8 | `Applications of 3D Vision.py` | 让机器"看懂三维世界"——应用遍及机器人/自动驾驶/医疗/AR/VR，底层是三维线性代数。 |
| 9 | `Model optimization tools and frameworks.py` | 把训好的视觉模型压小压快好部署——量化、剪枝、蒸馏 + 各家工具。 |
| 10 | `Synthetic Datasets.py` | 真实标注贵又少，用 3D 渲染器(Blender/Unity)造"越真越好"的合成训练数据。 |
| 11 | `Zero-shot Learning.py` | 识别"训练时没见过的类别"——靠语义嵌入把"没见过的类"和"见过的知识"连起来。 |
| 12 | `Exploring Ethical Foundations in CV Models.py` | 视觉模型会从数据里学到社会偏见，负责任的 AI 要能发现、评估、缓解偏见。 |
| 13 | `Retention In Vision.py` | RetNet 用"多尺度保持(MSR)"替代注意力，兼顾并行训练 + O(1) 推理 + 长序列。 |
