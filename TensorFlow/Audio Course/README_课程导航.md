# Audio Course · 课程导航

> 每个文件顶部有完整「学习笔记描述」。多为 HF 官方教程的 notebook 摘录/转写，
> 以**读懂原理**为主，不少需特定环境(GPU/gym/Unity/Blender 等)，不保证本机直接跑。

| 章 | 文件 | 一句话 |
|:--:|------|--------|
| 1 | `Audio as a waveform.py` | 搞懂"声音在计算机里长什么样"——时域波形、频域频谱、时频频谱图，是所有音频任务的地基。 |
| 2 | `Audio classification with a pipeline.py` | 一行 pipeline 跑通"音频→标签"(分类)和"音频→文本"(语音识别)，先感受能干什么。 |
| 3 | `Transformer architectures for audio.py` | Transformer 怎么用到音频——编码器/解码器/注意力，以及音频任务的输入输出形态。 |
| 4 | `Pre-trained models and datasets for audio classification.py` | 拿现成预训练模型做音频分类的几个子任务，关键是"你的类别要和模型训练的类别对得上"。 |
| 5 | `Pre-trained models for automatic speech recognition.py` | 语音识别两大范式——CTC(仅编码器)和 Seq2Seq(编解码+交叉注意力，如 Whisper)。 |
| 6 | `Text-to-speech datasets.py` | 做 TTS 先懂数据——单说话人 vs 多说话人、单语种 vs 多语种、采样率/时长差异。 |
| 7 | `Speech-to-speech translation.py` | 把一种语言的语音翻成另一种语言的语音——级联式：ASR翻译 + TTS 合成。 |
