"""
================================================================================
 Chapter 9 · Gradio 构建与分享演示（原始课程文件·Gradio 3.x API，仅作对照）
================================================================================
 ⚠ 重要：这份是课程原文，用的是 Gradio 3.x 的旧 API，而你本机是 Gradio 6.26。
   很多写法在 v6 已被移除/改名，直接跑会报错，例如：
     gr.Audio(source="microphone")   → v6 改成 gr.Audio(sources=["microphone"])
     gr.Interface.load("huggingface/...")  → v6 移除，改用 gr.load(...) 或直接调模型
     gr.Textbox.update(...)           → v6 移除，改成直接 return gr.Textbox(...)
     interpretation="default"         → v6 移除
     allow_flagging=/allow_screenshot= → v6 改成 flagging_mode=
     from transformers import pipelines→ 应是 pipeline(单数，本文件第 11 行是笔误)
   所以本文件只当“课程知识点清单”看；能在你机器上真跑的“生产级”版本见同目录：
     · Chapter9_Gradio生产实战_学习笔记.py   （Gradio 6 生产写法 + 为什么，可运行自检）
     · Chapter9_案例闯关_Gradio生产实战.py   （4 个完整可上线的生产 App，菜单启动）

 本文件覆盖的知识点(按出现顺序)：
   Interface 基础 / 组件(Textbox) / 接模型预测 / Audio / 多输入多输出 / 语音转文字 /
   分享链接与 title/description/examples / 自定义模型 sketchpad / 会话状态 state /
   预测可解释性 / Blocks 布局 / Tabs / 加载在线模型 / 多步骤 Blocks / 动态更新组件
================================================================================
"""
# ------------------------------------------------------------------------------
# 1) Interface 基础：fn + inputs + outputs 三件套就是一个 App
# ------------------------------------------------------------------------------
import  gradio as gr
def greet(name):
    return "hello"+name
demo=gr.Interface(fn=greet,inputs="text",outputs="text")
print(demo.launch())
textbox = gr.Textbox(label="Type your name here:", placeholder="John Doe", lines=2)
gr.Interface(fn=greet, inputs=textbox, outputs="text").launch()


#包括模型预测
from transformers import pipelines
model=pipelines("text-generation" )
def predict(prompt):
    completion=model(prompt)[0]['generated_text']
    return completion
predict("My favorite programming language is")
gr.Interface(fn=predict,inputs="text",outputs="text").launch()

#Audio输出组件
import numpy as np
def reverse_audio(audio):
    sr, data = audio
    reversed_audio = (sr, np.flipud(data))
    return reversed_audio

mic = gr.Audio(source="microphone", type="numpy", label="Speak here...")
gr.Interface(reverse_audio, mic, "audio").launch()
#处理多个输入和输出
import numpy as np
import gradio as gr

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def generate_tone(note, octave, duration):
    sr = 48000
    a4_freq, tones_from_a4 = 440, 12 * (octave - 4) + (note - 9)
    frequency = a4_freq * 2 ** (tones_from_a4 / 12)
    duration = int(duration)
    audio = np.linspace(0, duration, duration * sr)
    audio = (20000 * np.sin(audio * (2 * np.pi * frequency))).astype(np.int16)
    return (sr, audio)


gr.Interface(
    generate_tone,
    [
        gr.Dropdown(notes, type="index"),
        gr.Slider(minimum=4, maximum=6, step=1),
        gr.Number(value=1, label="Duration in seconds"),
    ],
    "audio",
).launch()

#语音转文字
from transformers import pipeline
import gradio as gr

model = pipeline("automatic-speech-recognition")


def transcribe_audio(audio):
    transcription = model(audio)["text"]
    return transcription


gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
).launch()
#分享链接其他人访问

title = "Ask Rick a Question"
description = """
The bot was trained to answer questions based on Rick and Morty dialogues. Ask Rick anything!
<img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png" width=200px>
"""

article = "Check out [the original Rick and Morty Bot](https://huggingface.co/spaces/kingabzpro/Rick_and_Morty_Bot) that this demo is based off of."

gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text",
    title=title,
    description=description,
    article=article,
    examples=[["What are you doing?"], ["Where should we time travel to?"]],
).launch()
#gr.Interface(classify_image, "image", "label").launch(share=True)

#文件加载模型并创建一个predict()函数
from pathlib import Path
import torch
from torch import nn

LABELS = Path("class_names.txt").read_text().splitlines()

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1152, 256),
    nn.ReLU(),
    nn.Linear(256, len(LABELS)),
)
state_dict = torch.load("pytorch_model.bin", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()


def predict(im):
    x = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(x)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    values, indices = torch.topk(probabilities, 5)
    return {LABELS[i]: v.item() for i, v in zip(indices, values)}
interface = gr.Interface(
    predict,
    inputs="sketchpad",
    outputs="label",
    theme="huggingface",
    title="Sketch Recognition",
    description="Who wants to play Pictionary? Draw a common object like a shovel or a laptop, and the algorithm will guess in real time!",
    article="<p style='text-align: center'>Sketch Recognition | Demo Model</p>",
    live=True,
)
interface.launch(share=True)
#使用状态来持久化数据
#Gradio 支持会话状态，即数据在页面加载期间多次提交后仍能保持有效。
# 会话状态对于构建演示模型（例如聊天机器人）非常有用
import random

import gradio as gr


def chat(message, history):
    history = history or []
    if message.startswith("How many"):
        response = random.randint(1, 10)
    elif message.startswith("How"):
        response = random.choice(["Great", "Good", "Okay", "Bad"])
    elif message.startswith("Where"):
        response = random.choice(["Here", "There", "Somewhere"])
    else:
        response = "I don't know"
    history.append((message, response))
    return history, history


iface = gr.Interface(
    chat,
    ["text", "state"],
    ["chatbot", "state"],
    allow_screenshot=False,
    allow_flagging="never",
)
iface.launch()
#运用解释来理解预测
import requests
import tensorflow as tf

import gradio as gr
# 加载模型
inception_net = tf.keras.applications.MobileNetV2()

# 下载 ImageNet 的人类可读标签。response
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def classify_image(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}


image = gr.Image(shape=(224, 224))
label = gr.Label(num_top_classes=3)

title = "Gradio Image Classifiction + Interpretation Example"
gr.Interface(
    fn=classify_image, inputs=image, outputs=label, interpretation="default", title=title
).launch()

# ------------------------------------------------------------------------------
# 2) Blocks：比 Interface 更灵活的“积木式”布局(生产 App 基本都用 Blocks)
# ------------------------------------------------------------------------------
import gradio as gr

def flip_text(x):
    return x[::-1]

demo = gr.Blocks()

with demo:
    gr.Markdown(
        """
    # Flip Text!
    Start typing below to see the output.
    """
    )
    input = gr.Textbox(placeholder="Flip this text")
    output = gr.Textbox()
    input.change(fn=flip_text, inputs=input, outputs=output)
demo.launch()

import numpy as np
import gradio as gr

demo = gr.Blocks()

#flip_image()为演示添加一个功能，并添加一个用于翻转图像的新标签页
def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)

with demo:
    gr.Markdown("Flip text or image files using this demo.")
    with gr.Tabs():
        with gr.TabItem("Flip Text"):
            with gr.Row():
                text_input = gr.Textbox()
                text_output = gr.Textbox()
            text_button = gr.Button("Flip")
        with gr.TabItem("Flip Image"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")
    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)
demo.launch()

import gradio as gr

api = gr.Interface.load("huggingface/EleutherAI/gpt-j-6B")


def complete_with_gpt(text):
    # 使用文本的最后 50 个字符作为上下文
    return text[:-50] + api(text[-50:])


with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Type here and press enter...", lines=4)
    btn = gr.Button("Generate")

    btn.click(complete_with_gpt, textbox, textbox)

demo.launch()
#创建多步骤演示
from transformers import pipeline

import gradio as gr

asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
classifier = pipeline("text-classification")


def speech_to_text(speech):
    text = asr(speech)["text"]
    return text


def text_to_sentiment(text):
    return classifier(text)[0]["label"]


demo = gr.Blocks()

with demo:
    audio_file = gr.Audio(type="filepath")
    text = gr.Textbox()
    label = gr.Label()

    b1 = gr.Button("Recognize Speech")
    b2 = gr.Button("Classify Sentiment")

    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    b2.click(text_to_sentiment, inputs=text, outputs=label)

demo.launch()

#更新组件属性
import gradio as gr
def change_textbox(choice):
    if choice == "short":
        return gr.Textbox.update(lines=2, visible=True)
    elif choice == "long":
        return gr.Textbox.update(lines=8, visible=True)
    else:
        return gr.Textbox.update(visible=False)


with gr.Blocks() as block:
    radio = gr.Radio(
        ["short", "long", "none"], label="What kind of essay would you like to write?"
    )
    text = gr.Textbox(lines=2, interactive=True)

    radio.change(fn=change_textbox, inputs=radio, outputs=text)
    block.launch()

