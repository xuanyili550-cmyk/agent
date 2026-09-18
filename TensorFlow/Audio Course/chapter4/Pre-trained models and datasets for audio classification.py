"""
================================================================================
 Audio Course · Chapter 4 · 音频分类的预训练模型与数据集（学习笔记描述）
================================================================================
 一句话：拿现成预训练模型做音频分类的几个子任务，关键是"你的类别要和模型训练的类别对得上"。
 本章讲：
   ① 关键词识别(KWS)：在语音里识别固定关键词集(Speech Commands 等)。
   ② 语言识别(LID)、零样本音频分类等其它分类任务与对应模型/数据集。
   ③ 继续用 MINDS-14 做意图分类的 pipeline 实践。
 要点：预训练分类模型的标签集是固定的，用前先确认标签匹配你的场景。
 说明：需 transformers/datasets + 联网；参考为主。
================================================================================
"""

#用于音频分类的预训练模型和数据集
#pip install git+https://github.com/huggingface/transformers
# 关键词发现
# 关键词识别（KWS）是指在语音中识别关键词的任务。所有可能的关键词构成预测的类别标签集。
# 因此，要使用预训练的关键词识别模型，您应该确保您的关键词与模型预训练所用的关键词相匹配。
# 下面，我们将介绍两个用于关键词识别的数据集和模型。
#
# 心灵-14
# 我们继续使用上一单元中探索过的MINDS-14intent_class数据集。如果您还记得，MINDS-14
# 包含人们用多种语言和方言向网上银行系统提问的录音，并且每条录音都有其对应的信息。我们可以根据通话意图对这些录音进行分类。

from datasets import load_dataset
from gradio import Audio

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")

from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)

classifier(minds[0]["audio"])

#语音指令
# Speech Commands 是一个语音词汇数据集，旨在评估音频分类模型在简单命令词上的性能。
# 该数据集包含 15 个关键词类别、一个静音类别和一个未知类别（用于包含误报）。
# 这 15 个关键词均为单个词语，通常用于设备设置中，以控制基本任务或启动其他进程。
#
# 类似的模型也在你的手机上持续运行。这里，我们不使用单个命令词，而是使用设备特有的“唤醒词”，
# 例如“嘿 Google”或“嘿 Siri”。当音频分类模型检测到这些唤醒词时，它会触发手机开始监听麦克风，
# 并使用语音识别模型将你的语音转录成音频。
speech_commands = load_dataset(
    "speech_commands", "v0.02", split="validation", streaming=True
)
sample = next(iter(speech_commands))
classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2"
)
classifier(sample["audio"].copy())

# 语言识别
# 语言识别（LID）的任务是从候选语言列表中识别音频样本中所使用的语言。
# LID 在许多语音处理流程中扮演着重要角色。例如，给定一个未知语言的音频样本，
# 可以使用 LID 模型对音频样本中所使用的语言进行分类，然后选择一个针对该语言训练的合适语音识别模型来转录音频。
fleurs = load_dataset("google/fleurs", "all", split="validation", streaming=True)
sample = next(iter(fleurs))
classifier = pipeline(
    "audio-classification", model="sanchit-gandhi/whisper-medium-fleurs-lang-id"
)
classifier(sample["audio"])

# 零样本音频分类
# 在传统的音频分类范式中，模型会根据预定义的类别集合预测类别标签。
# 这限制了预训练模型在音频分类中的应用，因为预训练模型的标签集必须与下游任务的标签集相匹配。
# 以之前的 LID 为例，模型必须预测其训练所用的 102 个语言类别中的一个。如果下游任务实际需要 110 种语言，
# 那么模型将无法预测其中 8 种语言，因此需要重新训练才能实现完全覆盖。这限制了迁移学习在音频分类任务中的有效性。
# Transformer 仅支持一种零样本音频分类模型：CLAP 模型。
# CLAP 是一种基于 Transformer 的模型，它同时接受音频和文本作为输入，
# 并计算两者之间的相似度。如果输入的文本与音频高度相关，则会得到较高的相似度得分；
# 反之，如果输入的文本与音频完全无关，则会返回较低的相似度得分。
dataset = load_dataset("ashraq/esc50", split="train", streaming=True)
audio_sample = next(iter(dataset))["audio"]["array"]
candidate_labels = ["Sound of a dog", "Sound of vacuum cleaner"]
classifier = pipeline(
    task="zero-shot-audio-classification", model="laion/clap-htsat-unfused"
)
classifier(audio_sample, candidate_labels=candidate_labels)
Audio(audio_sample, rate=16000)


#音乐分类模型微调
from datasets import load_dataset

gtzan = load_dataset("marsyas/gtzan", "all")
gtzan
gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)
gtzan
gtzan["train"][0]
id2label_fn = gtzan["train"].features["genre"].int2str
id2label_fn(gtzan["train"][0]["genre"])

import gradio as gr


def generate_audio():
    example = gtzan["train"].shuffle()[0]
    audio = example["audio"]
    return (
        audio["sampling_rate"],
        audio["array"],
    ), id2label_fn(example["genre"])


with gr.Blocks() as demo:
    with gr.Column():
        for _ in range(4):
            audio, label = generate_audio()
            output = gr.Audio(audio, label=label)

demo.launch(debug=True)

#使用 Gradio 构建演示
from transformers import pipeline

model_id = "sanchit-gandhi/distilhubert-finetuned-gtzan"
pipe = pipeline("audio-classification", model=model_id)
def classify_audio(filepath):
    preds = pipe(filepath)
    outputs = {}
    for p in preds:
        outputs[p["label"]] = p["score"]
    return outputs

import gradio as gr

demo = gr.Interface(
    fn=classify_audio, inputs=gr.Audio(type="filepath"), outputs=gr.Label()  # 修复:gr.outputs 在 Gradio 4+ 已删,直接用 gr.Label
)
demo.launch(debug=True)

#选择🤗 Hub上任何您认为适合音频分类的模型，并使用完全相同的数据marsyas/gtzan集来构建您自己的分类器。

# 你的目标是使用分类器在该数据集上达到 87% 的准确率。
