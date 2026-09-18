"""
================================================================================
 Audio Course · Chapter 5 · ASR 预训练模型：CTC vs Seq2Seq（学习笔记描述）
================================================================================
 一句话：语音识别两大范式——CTC(仅编码器)和 Seq2Seq(编解码+交叉注意力，如 Whisper)。
 本章讲：
   ① CTC：仅编码器 + 线性 CTC 头，逐帧对齐(Wav2Vec2 类)。
   ② Seq2Seq：编码器理解语音 + 解码器生成文本(Whisper)，交叉注意力对齐。
   ③ 数据集探测、推理与(WER 等)评估思路。
 要点：CTC 快但不建模输出语言、Whisper 更强更稳且多语种/带标点。
 说明：需 datasets/transformers + 联网下模型；含长代码，参考为主。
================================================================================
"""

#用于自动语音识别的预训练模型
# 语音识别模型大致可分为两类：
#
# 连接主义时间分类（CTC）：顶部带有线性分类（CTC）头的仅编码器模型
# 序列到序列（Seq2Seq）：编码器-解码器模型，编码器和解码器之间具有交叉注意力机制。

#探测CTC模型
from datasets import load_dataset

dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
dataset

from IPython.display import Audio

sample = dataset[2]

print(sample["text"])
Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])


from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-100h")


pipe(sample["audio"].copy())

# Target:      HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
# Prediction:  HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH **CHRISTMAUS** AND **ROSE** BEEF LOOMING BEFORE US **SIMALYIS** DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND

# 毕业到Seq2Seq
# 提示：Seq2Seq 模型！如第三单元所述，Seq2Seq 模型由编码器和解码器组成，
# 二者通过交叉注意力机制连接。编码器的作用与之前相同，负责计算音频输入的隐藏状态表示，
# 而解码器则扮演语言模型的角色。解码器处理来自编码器的整个隐藏状态表示序列，
# 并生成相应的文本转录。凭借音频输入的全局上下文，解码器能够在进行预测时利用语言模型上下文，
# 实时纠正拼写错误，从而避免语音预测的问题。
#
# Seq2Seq模型有两个缺点：
#
# 它们的解码速度天生就比较慢，因为解码过程是一步一步进行的，而不是一次性完成的。
# 它们对数据的需求更高，需要更多的训练数据才能达到收敛。


import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base", device=device
)
pipe(sample["audio"], max_new_tokens=256)

dataset = load_dataset(
    "facebook/multilingual_librispeech", "spanish", split="validation", streaming=True
)
sample = next(iter(dataset))

print(sample["text"])
Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])


pipe(sample["audio"].copy(), max_new_tokens=256, generate_kwargs={"task": "transcribe"})

pipe(sample["audio"], max_new_tokens=256, generate_kwargs={"task": "translate"})

#完整版转录和时间戳
import numpy as np

target_length_in_m = 5

# 将分钟转换为秒（* 60）再转换为样本数（* 采样率）
sampling_rate = pipe.feature_extractor.sampling_rate
target_length_in_samples = target_length_in_m * 60 * sampling_rate

# 遍历我们的流式数据集，连接样本，直到达到目标值
long_audio = []
for sample in dataset:
    long_audio.extend(sample["audio"]["array"])
    if len(long_audio) > target_length_in_samples:
        break

long_audio = np.asarray(long_audio)

# 我们做得怎么样？
seconds = len(long_audio) / 16000
minutes, seconds = divmod(seconds, 60)
print(f"Length of audio sample is {minutes} minutes {seconds:.2f} seconds")

# 直接将这段较长的音频样本转发给模型存在两个问题：
#
# Whisper 的设计初衷是处理 30 秒的音频样本：任何短于 30 秒的音频都会用静音填充到 30 秒，
# 任何长于 30 秒的音频都会被截断到 30 秒，因此如果我们直接传递音频，就只能得到前 30 秒的转录结果。
# Transformer 网络中的内存需求与序列长度的平方成正比：输入长度翻倍，内存需求就会增加四倍，
# 因此传输超长音频文件必然会导致内存不足 (OOM) 错误。

pipe(
    long_audio,
    max_new_tokens=256,
    generate_kwargs={"task": "transcribe"},
    chunk_length_s=30,
    batch_size=8,
)

pipe(
    long_audio,
    max_new_tokens=256,
    generate_kwargs={"task": "transcribe"},
    chunk_length_s=30,
    batch_size=8,
    return_timestamps=True,
)["chunks"]


# 语音数据集的特征
# 1. 小时数
# 简而言之，训练时长表明了数据集的大小。这类似于自然语言处理（NLP）数据集中的训练样本数量。
# 然而，更大的数据集并不一定更好。如果我们想要一个泛化能力强的模型，我们需要一个包含各种不同说话人、
# 领域和说话风格的多样化数据集。
#
# 2. 领域
# 数据来源领域指的是数据的来源，例如有声读物、播客、YouTube 或金融会议。每个领域的数据分布都不同。
# 例如，有声读物是在高质量的录音棚环境下录制的（没有背景噪音），文本也取自文学作品。
# 而 YouTube 上的音频则可能包含更多背景噪音，并且语调也更随意。
#
# 我们需要使模型的训练领域与推理时预期的条件相匹配。例如，如果我们用有声读物训练模型，
# 就不能指望它在嘈杂的环境中表现良好。
#
# 3. 说话风格
# 说话风格可分为以下两类：
#
# 旁白：根据剧本朗读
# 自发性：未经脚本的、对话式的言语
# 4. 转录风格
# 转录风格指的是目标文本是否包含标点符号、大小写或两者兼有。
# 如果我们想要一个系统生成可用于出版物或会议记录的完整格式文本，
# 则需要包含标点符号和大小写的训练数据。如果我们只需要未经格式化的口语文本，
# 则标点符号和大小写都不是必需的。在这种情况下，我们可以选择一个不包含标点符号和大小写的数据集，
# 或者选择一个包含标点符号和大小写的数据集，然后通过预处理将其从目标文本中移除。
#
# ASR 的评估指标
# 如果您熟悉自然语言处理中的莱文斯坦距离，那么用于评估语音识别系统的指标对您来说应该并不陌生！
# 如果您不熟悉也不用担心，我们会从头到尾详细解释，确保您了解不同的指标及其含义。
#
# 在评估语音识别系统时，我们将系统的预测结果与目标文本转录进行比较，并标注其中存在的任何错误。
# 我们将这些错误分为以下三类：
#
# 替换（S）：我们在预测中转录了错误的单词（例如，“sit”而不是“sat”）。
# 插入（I）：我们在预测中添加一个额外的词
# 删除（D）：指我们在预测中移除一个词。

#pip install --upgrade evaluate jiwer
from evaluate import load

wer_metric = load("wer")

# 课程里的示例句：1 个替换 + 1 个删除
reference = "the cat sat on the mat"
prediction = "the cat sit on the"

wer = wer_metric.compute(references=[reference], predictions=[prediction])

print(wer)
#
# 词错误率(WER)
# 用于衡量转录的准确性，而实时性倒数(RTFx)
# 用于衡量自动语音识别(ASR)
# 系统的速度。RTFx
# 是处理时间与音频时长的倒数比值：
# RTFx=音频时长/处理时间
# 例如，如果转录
# 100
# 秒的音频需要
# 10
# 秒，则
# RTFx
# 为
# 100 / 10 = 10。RTFx
# 大于
# 1.0
# 表示系统可以比实时速度更快地转录音频，这对于视频会议或实时字幕等实时转录应用至关重要。RTFx
# 为
# 1.0
# 表示系统以实时速度处理，而小于
# 1.0
# 的值则表示处理速度慢于实时速度。
#
# RTFx
# 的要点：
#
# 数值越高越好：RTFx
# 值越高，处理速度越快。
# RTFx > 1.0：比实时速度更快（适用于流媒体应用）
# RTFx = 1.0：以实时速度处理
# RTFx < 1.0：速度比实时慢（可能适用于批量处理）
# RTFx
# 依赖于硬件，并会因以下因素而异：
#
# 模型尺寸（较大的模型通常具有较低的RTFx）
# 硬件加速（GPU
# vs
# CPU）
# 批量大小
# 音频特性（采样率、声道数）
# 在评估自动语音识别（ASR）系统时，必须同时考虑词错误率（WER）和实时性指标（RTFx）。WER
# 极低但
# RTFx
# 很低的模型可能不适用于实时应用，而
# WER
# 稍高但
# RTFx
# 高的模型可能更适合对延迟敏感的应用场景。
#
# 词语准确率
# 我们可以反过来衡量词错误率（WER），得到一个数值越高越好的指标。与其衡量词错误率，不如衡量系统的
# 词准确率（WAcc） ：
#WAcc=1−WER
# WAcc
# 也是在词级层面上衡量的，它只是将
# WER
# 重新表述为准确率指标而非错误率指标。WAcc
# 在语音文献中很少被引用——我们通常以词错误来衡量系统的预测，因此更倾向于使用与这类错误标注更相关的错误率指标。

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

prediction = " He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly is drawn from eating and its results occur most readily to the mind."
normalized_prediction = normalizer(prediction)

normalized_prediction

reference = "HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND"
normalized_referece = normalizer(reference)

wer = wer_metric.compute(
    references=[normalized_referece], predictions=[normalized_prediction]
)
wer

#把所有东西整合起来
from transformers import pipeline
import torch

if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    torch_dtype=torch_dtype,
    device=device,
)

from datasets import load_dataset

common_voice_test = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="test"
)

from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

all_predictions = []

# run streamed inference
for prediction in tqdm(
    pipe(
        KeyDataset(common_voice_test, "audio"),
        max_new_tokens=128,
        generate_kwargs={"task": "transcribe"},
        batch_size=32,
    ),
    total=len(common_voice_test),
):
    all_predictions.append(prediction["text"])



from evaluate import load

wer_metric = load("wer")

wer_ortho = 100 * wer_metric.compute(
    references=common_voice_test["sentence"], predictions=all_predictions
)
wer_ortho




from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

# 计算归一化词错误率
all_predictions_norm = [normalizer(pred) for pred in all_predictions]
all_references_norm = [normalizer(label) for label in common_voice_test["sentence"]]

# 过滤步骤，仅评估与非零引用对应的样本
all_predictions_norm = [
    all_predictions_norm[i]
    for i in range(len(all_predictions_norm))
    if len(all_references_norm[i]) > 0
]
all_references_norm = [
    all_references_norm[i]
    for i in range(len(all_references_norm))
    if len(all_references_norm[i]) > 0
]

wer = 100 * wer_metric.compute(
    references=all_references_norm, predictions=all_predictions_norm
)

wer



#对 ASR 模型进行微调
from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="train+validation"
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="test"
)

print(common_voice)

common_voice = common_voice.select_columns(["audio", "sentence"])

# 特征提取器、分词器和处理器
# ASR流程可以分解为三个阶段：
#
# 特征提取器对原始音频输入进行预处理，生成对数梅尔频谱图。
# 执行序列到序列映射的模型
# 分词器对预测的词元进行后处理，生成文本。
# 在 Transformer 模型中，Whisper 模型关联着一个特征提取器和一个分词器，
# 分别称为WhisperFeatureExtractor和WhisperTokenizer 。为了简化操作，
# 我们将这两个对象封装在一个名为WhisperProcessor 的类中。我们可以调用 WhisperProcessor
# 来执行音频预处理和文本分词后处理。这样一来，在训练过程中我们只需要维护两个对象：处理器和模型。
# 在进行多语言微调时，我们需要在实例化处理器时设置 `<output_language>`"language"和
# `<task_language>`。` <output_language>` 应设置为源音频语言，`<task_language>`
# 应设置为语音识别或 语音翻译。这些参数会修改分词器的行为，必须正确设置以确保目标标签被正确编码。
# "task""language""transcribe""translate"

from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

TO_LANGUAGE_CODE

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="sinhalese", task="transcribe"
)
#数据预处理

common_voice["train"].features

from datasets import Audio

sampling_rate = processor.feature_extractor.sampling_rate
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sampling_rate))

def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["sentence"],
    )

    # 计算音频样本的输入长度（以秒为单位）
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example

common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1
)

max_input_length = 30.0


def is_audio_in_length_range(length):
    return length < max_input_length

common_voice["train"] = common_voice["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)

common_voice["train"]

#WhisperProcessor之前定义的方法来执行特征提取和分词操作
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # 由于输入和标签长度不同，需要不同的填充方法，因此需要将它们分开
        # 首先，通过直接返回 torch 张量来处理音频输入
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 获取分词后的标签序列
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # 将标签填充到最大长度
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 将填充替换为 -100 以正确忽略损失
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 如果在之前的分词步骤中附加了 bos 标记，
        # 则在此处截断 bos 标记，因为它无论如何稍后都会附加
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
#评估指标
import evaluate
metric = evaluate.load("wer")
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
normalizer = BasicTextNormalizer()
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # 将 -100 替换为 pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    # 我们不希望在计算指标时对标记进行分组
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    # 计算正交权重
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)
    # 计算归一化 WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # 过滤步骤，仅评估与非零参考值对应的样本：
    pred_str_norm = [
         pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
        ]
    label_str_norm = [
       label_str_norm[i]
           for i in range(len(label_str_norm))
             if len(label_str_norm[i]) > 0
         ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}

#加载预训练检查点
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

from functools import partial

# 由于缓存与梯度检查点模型不兼容，因此在训练期间禁用缓存
model.config.use_cache = False

# 设置生成语言和任务，并重新启用缓存
model.generate = partial(
    model.generate, language="sinhalese", task="transcribe", use_cache=True
)

#定义训练配置

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-dv",  # name on the HF Hub
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # 当批次大小每减少 2 倍时，梯度增加 2 倍
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=500,  # 如果您有自己的 GPU 或 Colab 付费计划，请增加到 4000
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    eval_strategy="steps",  # 修复:transformers v5 把 evaluation_strategy 改名为 eval_strategy
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,  # 修复:transformers v5 用 processing_class 取代了 tokenizer 参数
)

trainer.train()

kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_13_0",
    "dataset": "Common Voice 13",  # 训练数据集的“美观”名称
    "language": "dv",
    "model_name": "Whisper Small Dv - Sanchit Gandhi",  # 模型的“美观”名称
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
}

from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="sanchit-gandhi/whisper-small-dv")


#使用 Gradio 构建演示
from transformers import pipeline

model_id = "sanchit-gandhi/whisper-small-dv"  # 更新为您的模型 ID
pipe = pipeline("automatic-speech-recognition", model=model_id)

def transcribe_speech(filepath):
    output = pipe(
        filepath,
        max_new_tokens=256,
        generate_kwargs={
            "task": "transcribe",
            "language": "sinhalese",
        }, # 使用您已微调的语言进行更新
        chunk_length_s=30,
        batch_size=8,
    )
    return output["text"]

import gradio as gr

demo = gr.Blocks()

mic_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=gr.components.Textbox(),
)

file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=gr.components.Textbox(),
)


with demo:
    gr.TabbedInterface(
        [mic_transcribe, file_transcribe],
        ["Transcribe Microphone", "Transcribe Audio File"],
    )

demo.launch(debug=True)
#
# 下是操作说明：
#
# ”openai/whisper-tiny”使用数据集中的美式英语（“en-US”）子集对模型进行微调”PolyAI/minds14”。
# 使用前450 个样本进行训练，其余样本用于评估。确保num_proc=1在使用该方法预处理数据集时进行设置.map
# （这将确保您的模型能够正确提交以进行评估）。
# 要评估模型，请使用本单元中描述的指标。但是，wer不要将指标乘以 100 转换为百分比
# （例如，如果 WER 为 42%，我们预计在本练习中看到的值为 0.42）。wer_ortho

