"""
================================================================================
 Audio Course · Chapter 2 · 用 pipeline 做音频分类 / ASR（学习笔记描述）
================================================================================
 一句话：一行 pipeline 跑通"音频→标签"(分类)和"音频→文本"(语音识别)，先感受能干什么。
 本章讲：
   ① 音频分类：MINDS-14 数据集(多语种银行客服意图)，pipeline("audio-classification")。
   ② 数据准备：cast_column 到 16kHz(模型要求的采样率必须对齐)。
   ③ ASR：pipeline("automatic-speech-recognition") 把语音转文字。
 要点：采样率不匹配是音频最常见的坑——喂模型前先重采样到它训练时的采样率。
 说明：需 datasets/transformers + 联网下模型和数据；参考为主。
================================================================================
"""

#使用流水线进行音频分类
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)
example = minds[0]
classifier(example["audio"]["array"])
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])

#使用流水线进行自动语音识别

from transformers import pipeline

asr = pipeline("automatic-speech-recognition")
example = minds[0]
asr(example["audio"]["array"])
example["english_transcription"]
#对 MINDS-14 的德语部分进行操作。加载“de-DE”子集：
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="de-DE", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
example = minds[0]
example["transcription"]
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="maxidl/wav2vec2-large-xlsr-german")
asr(example["audio"]["array"])


#使用管道进行音频生成
#pip install --upgrade transformers
from transformers import pipeline

pipe = pipeline("text-to-speech", model="suno/bark-small")
text = "Ladybugs have had important roles in culture and religion, being associated with luck, love, fertility and prophecy. "
output = pipe(text)
from IPython.display import Audio

Audio(output["audio"].squeeze(), rate=output["sampling_rate"])

fr_text = "Contrairement à une idée répandue, le nombre de points sur les élytres d'une coccinelle ne correspond pas à son âge, ni en nombre d'années, ni en nombre de mois. "
output = pipe(fr_text)
Audio(output["audio"].squeeze(), rate=output["sampling_rate"])

song = "♪ In the jungle, the mighty jungle, the ladybug was seen. ♪ "
output = pipe(song)
Audio(output["audio"].squeeze(), rate=output["sampling_rate"])

#生成音乐
music_pipe = pipeline("text-to-audio", model="facebook/musicgen-small")
text = "90s rock song with electric guitar and heavy drums"
forward_params = {"max_new_tokens": 512}

output = music_pipe(text, forward_params=forward_params)
Audio(output["audio"][0], rate=output["sampling_rate"])

# 使用 🤗 数据集以流式模式加载您选择的语言的facebook/voxpopuli数据集的训练集。
# 从数据集中提取第三个示例train并进行探索。根据该示例的特征，你可以将此数据集用于哪些类型的音频任务？
# 绘制此示例的波形图和频谱图。
# 前往🤗 Hub，浏览预训练模型，找到一个可以用于自动语音识别的模型，该模型适用于您之前选择的语言。使用找到的模型实例化相应的管道，并转录示例。
# 将您从流程中获得的转录结果与示例中提供的转录结果进行比较。
