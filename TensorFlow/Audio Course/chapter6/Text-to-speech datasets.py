"""
================================================================================
 Audio Course · Chapter 6 · 文本转语音(TTS)数据集（学习笔记描述）
================================================================================
 一句话：做 TTS 先懂数据——单说话人 vs 多说话人、单语种 vs 多语种、采样率/时长差异。
 本章讲：
   ① LJSpeech(单人英语基准)、Multilingual LibriSpeech(多语种)。
   ② VCTK(110 位不同口音说话人)、LibriTTS/LibriTTS-R(24kHz 多说话人)。
   ③ 各数据集特点与"训什么样的 TTS 该选哪个"。
 要点：多说话人/多口音数据 → 训出的语音更自然多样；采样率决定音质与算力。
 说明：以数据集说明为主，配套本项目 音频制作_双语配音/ 的 TTS 实战。
================================================================================
"""

#文本转语音数据集
# LJSpeech
# LJSpeech数据集包含 13,100 个英语音频片段及其对应的转录文本。
# 该数据集包含一位说话者朗读 7 本英语非虚构类书籍中的句子的录音。
# 由于其高音频质量和丰富的语言内容，LJSpeech 常被用作评估文本转语音 (TTS) 模型的基准数据集。
#
# 多语言 LibriSpeech
# 多语言 LibriSpeech是 LibriSpeech 数据集的多语言扩展版本，
# 后者是一个大规模的英语有声读物集合。多语言 LibriSpeech 在此基础上增加了德语、荷兰语、西班牙语、法语、
# 意大利语、葡萄牙语和波兰语等其他语言。它提供每种语言的音频录音及其对应的转录文本。
# 该数据集为开发多语言文本转语音 (TTS) 系统和探索跨语言语音合成技术提供了宝贵的资源。
#
# VCTK（语音克隆工具包）
# VCTK是一个专为文本转语音 (TTS) 研究和开发而设计的数据集。它包含 110 位英语母语者不同口音的录音。
# 每位母语者朗读了约 400 个句子，这些句子选自一份报纸、一段彩虹段落以及一段用于构建语音口音库的诱导段落。
# VCTK 为训练具有不同语音和口音的 TTS 模型提供了宝贵的资源，从而能够实现更自然、更多样化的语音合成。
#
# Libri-TTS/ LibriTTS-R
# Libri-TTS/LibriTTS-R是一个包含约 585 小时英语语音的多说话人语料库，采样率为 24kHz，
# 由 Heiga Zen 在 Google Speech 和 Google Brain 团队成员的协助下创建。LibriTTS 语料库专为 TTS 研究而设计。
# 它源自 LibriSpeech 语料库的原始材料（来自 LibriVox 的 mp3 音频文件和来自 Project Gutenberg 的文本文件）。
# 与 LibriSpeech 语料库的主要区别如下：
#
# 音频文件的采样率为 24kHz。
# 演讲内容按句子分段。
# 包含原始文本和规范化文本。
# 可以提取上下文信息（例如，相邻句子）。
# 排除背景噪音较大的语音。
# 构建一个好的TTS数据集并非易事，因为这样的数据集必须具备几个关键特征：
#
# 高质量且多样化的录音，涵盖广泛的语音模式、口音、语言和情感。录音应清晰、无背景噪音，并展现自然的语音特征。
# 文字稿：每段音频录音都应附有相应的文字稿。
# 语言内容的多样性：数据集应包含多样化的语言内容，包括不同类型的句子、短语和词语。
# 它应涵盖各种主题、体裁和领域，以确保模型能够处理不同的语言环境。

#用于文本转语音的预训练模型
# SpeechT5
# SpeechT5是由微软的 Junyi Ao 等人发布的一款模型，能够处理多种语音任务。
# 虽然本单元主要关注文本转语音功能，但该模型也可用于语音转文本任务（例如自动语音识别或说话人识别），
# 以及语音转语音任务（例如语音增强或不同语音之间的转换）。这得益于该模型的设计和预训练方式。
#
# SpeechT5 的核心是一个标准的 Transformer 编码器-解码器模型。与其他 Transformer 模型一样，
# 该编码器-解码器网络使用隐藏表示来模拟序列到序列的转换。SpeechT5 支持的所有任务都使用相同的 Transformer 骨干网络。
#
# 该Transformer模型配备了六个特定模态（语音/文本）的预处理网络和后处理网络。
# 输入的语音或文本（取决于任务）会先经过相应的预处理网络进行预处理，以获得Transformer模型可以使用的隐藏表示。
# Transformer模型的输出随后会传递给后处理网络，后处理网络会利用这些隐藏表示生成目标模态的输出。
# SpeechT5 专门用于 TTS 任务的前置网络和后置网络分别是什么：
#
# 文本编码器预网络：一个文本嵌入层，它将文本标记映射到编码器期望的隐藏表示。
# 这类似于 BERT 等自然语言处理模型中的工作原理。
# 语音解码器预网络：它以对数梅尔频谱图作为输入，并使用一系列线性层将频谱图压缩成隐藏表示。
# 语音解码器后网络：该网络预测一个残差，并将其添加到输出频谱图中，用于改进结果。

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
inputs = processor(text="Don't count the days, make the days count.", return_tensors="pt")

from datasets import load_dataset

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

import torch

speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

from transformers import SpeechT5HifiGan

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

from IPython.display import Audio

Audio(speech, rate=16000)
# Bark 是 Suno AI 在suno-ai/bark中提出的基于 Transformer 的文本到语音模型
# Bark 由 4 个主要模型组成：
#
# BarkSemanticModel（也称为“文本”模型）：一个因果自回归转换器模型，以分词文本作为输入，
# 并预测能够捕捉文本含义的语义文本标记。
# BarkCoarseModel（也称为“粗略声学”模型）：一个因果自回归变换器，
# 以模型结果作为输入BarkSemanticModel。它的目标是预测 EnCodec 所需的前两个音频码本。
# BarkFineModel（‘精细声学’模型），这次是一个非因果自编码器转换器，
# 它根据先前码本嵌入的总和迭代地预测最后一个码本。
# Bark预测了所有码本通道EncodecModel，并利用这些通道解码输出音频数组。

from transformers import BarkModel, BarkProcessor

model = BarkModel.from_pretrained("suno/bark-small")
processor = BarkProcessor.from_pretrained("suno/bark-small")
# 添加说话人嵌入

inputs = processor("This is a test!", voice_preset="v2/en_speaker_3")

speech_output = model.generate(**inputs).cpu().numpy()
# 尝试用法语，我们也添加一个法语说话人嵌入

inputs = processor("C'est un test!", voice_preset="v2/fr_speaker_1")

inputs = processor(
    "[clears throat] This is a test ... and I just took a long pause.",
    voice_preset="v2/fr_speaker_1",
)

speech_output = model.generate(**inputs).cpu().numpy()

inputs = processor(
    "♪ In the mighty jungle, I'm trying to generate barks.",
)

speech_output = model.generate(**inputs).cpu().numpy()

input_list = [
    "[clears throat] Hello uh ..., my dog is cute [laughter]",
    "Let's try generating speech, with Bark, a text-to-speech model",
    "♪ In the jungle, the mighty jungle, the lion barks tonight ♪",
]

# 同时添加说话人嵌入
inputs = processor(input_list, voice_preset="v2/en_speaker_3")

speech_output = model.generate(**inputs).cpu().numpy()


from IPython.display import Audio

sampling_rate = model.generation_config.sample_rate
Audio(speech_output[0], rate=sampling_rate)
Audio(speech_output[1], rate=sampling_rate)
Audio(speech_output[2], rate=sampling_rate)

#大规模多语言语音（MMS）
#VITS 是一种语音生成网络，它将文本转换为原始语音波形。
# 它的工作原理类似于条件变分自编码器，从输入文本中估计音频特征
# 。首先，生成以频谱图形式表示的声学特征。然后，使用源自 HiFi-GAN
# 的转置卷积层对波形进行解码。在推理过程中，文本编码经过上采样，并使用流程模块和 HiFi-GAN 解码器转换为波形。
# 与 Bark 类似，VITS 无需声码器，因为它直接生成波形。
from transformers import VitsModel, VitsTokenizer

model = VitsModel.from_pretrained("facebook/mms-tts-deu")
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-deu")
text_example = (
    "Ich bin Schnappi das kleine Krokodil, komm aus Ägypten das liegt direkt am Nil."
)
import torch

inputs = tokenizer(text_example, return_tensors="pt")
input_ids = inputs["input_ids"]


with torch.no_grad():
    outputs = model(input_ids)

speech = outputs["waveform"]
from IPython.display import Audio

Audio(speech, rate=16000)

#微调 SpeechT5

#pip install transformers datasets soundfile speechbrain accelerate
from datasets import load_dataset, Audio

dataset = load_dataset("qmeeus/voxpopuli", "nl", split="train")
len(dataset)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

from transformers import SpeechT5Processor

checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)
#SpeechT5 分词的文本清理
tokenizer = processor.tokenizer

dataset[0]
def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset.column_names,
)

dataset_vocab = set(vocabs["vocab"][0])
tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
dataset_vocab - tokenizer_vocab
replacements = [
    ("à", "a"),
    ("ç", "c"),
    ("è", "e"),
    ("ë", "e"),
    ("í", "i"),
    ("ï", "i"),
    ("ö", "o"),
    ("ü", "u"),
]


def cleanup_text(inputs):
    for src, dst in replacements:
        inputs["text"] = inputs["text"].replace(src, dst)
    return inputs


dataset = dataset.map(cleanup_text)


from collections import defaultdict

speaker_counts = defaultdict(int)

for speaker_id in dataset["speaker_id"]:
    speaker_counts[speaker_id] += 1



import matplotlib.pyplot as plt

plt.figure()
plt.hist(speaker_counts.values(), bins=20)
plt.ylabel("Speakers")
plt.xlabel("Examples")
plt.show()


def select_speaker(speaker_id):
    return 100 <= speaker_counts[speaker_id] <= 400


dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])


len(set(dataset["speaker_id"]))

len(dataset)

#说话人嵌入
#为了让TTS模型能够区分多个说话人，您需要为每个示例创建说话人嵌入。
# 说话人嵌入是模型的一个额外输入，用于捕捉特定说话人的语音特征。要生成这些说话人嵌入，
# 请使用SpeechBrain提供的预训练模型spkrec-xvect-voxceleb 。
import os
import torch
from speechbrain.pretrained import EncoderClassifier

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)


def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings


def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        text=example["text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # 去除批次维度
    example["labels"] = example["labels"][0]

    # 使用 SpeechBrain 获取 x 向量
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example

processed_example = prepare_dataset(dataset[0])
list(processed_example.keys())


import matplotlib.pyplot as plt

plt.figure()
plt.imshow(processed_example["labels"].T)
plt.show()


dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)


def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length < 200


dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
len(dataset)



from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # 将输入和目标整理成一批
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # 将填充替换为 -100 以正确忽略损失
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # 微调期间未使用
        del batch["decoder_attention_mask"]

        # 则将目标长度向下取整到缩减因子的倍数：
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # 同时添加说话人嵌入
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch

data_collator = TTSDataCollatorWithPadding(processor=processor)
#训练模型
from transformers import SpeechT5ForTextToSpeech

model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)

from functools import partial

# 由于缓存与梯度检查点模型不兼容，因此在训练期间禁用缓存
model.config.use_cache = False

# 设置生成模型的语言和任务，并重新启用缓存
model.generate = partial(model.generate, use_cache=True)

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="speecht5_finetuned_voxpopuli_nl",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=2,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)

trainer.train()

#推理
model = SpeechT5ForTextToSpeech.from_pretrained(
    "YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl"
)

example = dataset["test"][304]
speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)

text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
inputs = processor(text=text, return_tensors="pt")
from transformers import SpeechT5HifiGan

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
from IPython.display import Audio

Audio(speech.numpy(), rate=16000)

#评估文本转语音模型

