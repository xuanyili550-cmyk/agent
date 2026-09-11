"""
================================================================================
 Audio Course · Chapter 7 · 语音到语音翻译(STST)（学习笔记描述）
================================================================================
 一句话：把一种语言的语音翻成另一种语言的语音——级联式：ASR翻译 + TTS 合成。
 本章讲：
   ① 用 Whisper 做"语音→目标语文本"(generate task=translate)。
   ② 再用 TTS 把译文合成成目标语语音，串成 S2S 管道。
   ③ 流式数据集(voxpopuli)取样与试跑。
 要点：级联简单可控但会累积误差；端到端 S2S 更难但少一次转写损失。
 说明：需 transformers/datasets + 联网下 Whisper；参考为主。
================================================================================
"""

#语音到语音翻译
#语音到语音翻译（STST 或 S2ST）是一种相对较新的口语处理任务。它涉及将一种语言的语音翻译成另一种语言的语音：
#语音翻译
import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base", device=device
)
from datasets import load_dataset

dataset = load_dataset("qmeeus/voxpopuli", "it", split="validation", streaming=True)
sample = next(iter(dataset))
from IPython.display import Audio

Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])
def translate(audio):
    outputs = pipe(audio, max_new_tokens=256, generate_kwargs={"task": "translate"})
    return outputs["text"]
translate(sample["audio"].copy())
sample["text"]


#文本转语音
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
model.to(device)
vocoder.to(device)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder
    )
    return speech.cpu()

speech = synthesise("Hey there! This is a test!")

Audio(speech, rate=16000)

#建 STST 演示
import numpy as np

target_dtype = np.int16
max_range = np.iinfo(target_dtype).max


def speech_to_speech_translation(audio):
    translated_text = translate(audio)
    synthesised_speech = synthesise(translated_text)
    synthesised_speech = (synthesised_speech.numpy() * max_range).astype(np.int16)
    return 16000, synthesised_speech

sampling_rate, synthesised_speech = speech_to_speech_translation(sample["audio"])

Audio(synthesised_speech, rate=sampling_rate)

import gradio as gr

demo = gr.Blocks()

mic_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
)

file_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
)

with demo:
    gr.TabbedInterface([mic_translate, file_translate], ["Microphone", "Audio File"])

demo.launch(debug=True)


#创建语音助手
#
# 1. 唤醒词检测
# 语音助手会持续监听通过设备麦克风传入的音频输入，但只有当说出特定的“唤醒词”或“触发词”时，它们才会启动。
#
# 唤醒词检测任务由一个小型的机载音频分类模型处理，该模型比语音识别模型小得多、轻量得多，
# 通常只有几百万个参数，而语音识别模型则有数亿个参数。因此，它可以持续在设备上运行而不会耗尽电池电量。
# 只有当检测到唤醒词时，才会启动较大的语音识别模型，之后该模型会自动关闭。
#
# 2. 语音转录
# 流程的下一步是将语音查询转录为文本。实际上，由于音频文件通常较大，将本地设备中的音频文件传输到云端速度较慢，
# 因此使用设备上的自动语音识别 (ASR) 模型直接转录比使用云端模型效率更高。设备端模型可能体积较小，
# 因此准确率可能低于云端模型，但更快的推理速度使其更具优势，因为我们可以近乎实时地进行语音识别，
# 在我们说话的同时即可将其转录为文本。
#
# 我们现在已经非常熟悉语音识别过程了，所以这应该轻而易举！
#
# 3. 语言模型查询
# 既然我们知道了用户的问题，接下来就需要生成回复！完成这项任务的最佳候选模型是 大型语言模型（LLM），
# 因为它们能够有效地理解文本查询的语义并生成合适的回复。
#
# 由于我们的文本查询很小（只有几个文本标记），而语言模型很大（数十亿个参数），
# 因此运行 LLM 推理最有效的方法是将我们的文本查询从我们的设备发送到在云端运行的 LLM，生成文本响应，
# 然后将响应返回到设备。
#
# 4. 合成语音
# 最后，我们将使用文本转语音（TTS）模型将文本回复合成为语音。这可以在设备端完成，
# 但您也可以在云端运行 TTS 模型，生成音频输出并将其传输回设备。

from transformers import pipeline
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
)
classifier.model.config.id2label
classifier.model.config.id2label[27]
'marvin'

from transformers.pipelines.audio_utils import ffmpeg_microphone_live


def launch_fn(
    wake_word="marvin",
    prob_threshold=0.5,
    chunk_length_s=2.0,
    stream_chunk_s=0.25,
    debug=False,
):
    if wake_word not in classifier.model.config.label2id.keys():
        raise ValueError(
            f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {classifier.model.config.label2id.keys()}."
        )

    sampling_rate = classifier.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Listening for wake word...")
    for prediction in classifier(mic):
        prediction = prediction[0]
        if debug:
            print(prediction)
        if prediction["label"] == wake_word:
            if prediction["score"] > prob_threshold:
                return True


launch_fn(debug=True)

#语音转录
transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base.en", device=device
)

import sys


def transcribe(chunk_length_s=5.0, stream_chunk_s=1.0):
    sampling_rate = transcriber.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Start speaking...")
    for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
        sys.stdout.write("\033[K")
        print(item["text"], end="\r")
        if not item["partial"][0]:
            break

        return item["text"]

transcribe()
#语言模型查询
from huggingface_hub import HfFolder
import requests


def query(text, model_id="tiiuae/falcon-7b-instruct"):
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HfFolder().get_token()}"}
    payload = {"inputs": text}

    print(f"Querying...: {text}")
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()[0]["generated_text"][len(text) + 1 :]

query("What does Hugging Face do?")

#合成语音
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

from datasets import load_dataset

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder
    )
    return speech.cpu()

from IPython.display import Audio

audio = synthesise(
    "Hugging Face is a company that provides natural language processing and machine learning tools for developers."
)

Audio(audio, rate=16000)


launch_fn()
transcription = transcribe()
response = query(transcription)
audio = synthesise(response)

Audio(audio, rate=16000, autoplay=True)



from transformers import HfAgent

agent = HfAgent(
    url_endpoint="https://api-inference.huggingface.co/models/bigcode/starcoder"
)

agent.run("Generate an image of a cat")

launch_fn()
transcription = transcribe()
agent.run(transcription)

#会议记录

# 说话人分割
# 说话人分割（或称说话人识别）是指对未标注的音频输入进行预测，确定“谁在何时说话”。
# 通过这种方法，我们可以预测每个说话人回合的开始/结束时间戳，即每个说话人开始和结束说话的时间。

#pip install --upgrade pyannote.audio
from pyannote.audio import Pipeline

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1", use_auth_token=True
)
from datasets import load_dataset

concatenated_librispeech = load_dataset(
    "sanchit-gandhi/concatenated_librispeech", split="train", streaming=True
)
sample = next(iter(concatenated_librispeech))

from IPython.display import Audio

Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])

import torch

input_tensor = torch.from_numpy(sample["audio"]["array"][None, :]).float()
outputs = diarization_pipeline(
    {"waveform": input_tensor, "sample_rate": sample["audio"]["sampling_rate"]}
)

outputs.for_json()["content"]

#语音转录
from transformers import pipeline

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
)

asr_pipeline(
    sample["audio"].copy(),
    generate_kwargs={"max_new_tokens": 256},
    return_timestamps=True,
)

#语音盒
from speechbox import ASRDiarizationPipeline

pipeline = ASRDiarizationPipeline(
    asr_pipeline=asr_pipeline, diarization_pipeline=diarization_pipeline
)

pipeline(sample["audio"].copy())

def tuple_to_string(start_end_tuple, ndigits=1):
    return str((round(start_end_tuple[0], ndigits), round(start_end_tuple[1], ndigits)))


def format_as_transcription(raw_segments):
    return "\n\n".join(
        [
            chunk["speaker"] + " " + tuple_to_string(chunk["timestamp"]) + chunk["text"]
            for chunk in raw_segments
        ]
    )

outputs = pipeline(sample["audio"].copy())

format_as_transcription(outputs)