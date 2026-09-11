"""
================================================================================
 ML for 3D · Chapter 5 · LGM(大高斯模型)image→3D（学习笔记描述）
================================================================================
 一句话：LGM 把"多视图扩散 + 高斯泼溅"打包成 image→3D 的端到端 pipeline,配 Gradio demo。
 本章讲：
   ① 用 diffusers 加载 dylanebert/LGM-full 自定义 pipeline。
   ② 图片 → 3D 高斯/网格,导出可看的 3D。
   ③ 用 Gradio 包成交互 demo。
 要点：LGM 是当下 image→3D 的代表方案之一。
 说明：需 CUDA + 特殊轮子(diff_gaussian_rasterization);参考为主。
================================================================================
"""

import shlex
import subprocess

import gradio as gr
import numpy as np
import torch
from diffusers import DiffusionPipeline

subprocess.run(
    shlex.split(
        "pip install https://huggingface.co/spaces/dylanebert/LGM-mini/resolve/main/wheel/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl"
    )
)

pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/LGM-full",
    custom_pipeline="dylanebert/LGM-full",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")


def run(image):
    input_image = np.array(image, dtype=np.float32) / 255.0
    splat = pipeline(
        "", input_image, guidance_scale=5, num_inference_steps=30, elevation=0
    )
    splat_file = "/tmp/output.ply"
    pipeline.save_ply(splat, splat_file)
    return splat_file


demo = gr.Interface(
    fn=run,
    title="LGM Tiny",
    description="An extremely simplified version of [LGM](https://huggingface.co/ashawkey/LGM). Intended as resource for the [ML for 3D Course](https://huggingface.co/learn/ml-for-3d-course/unit0/introduction).",
    inputs="image",
    outputs=gr.Model3D(),
    examples=[
        "https://huggingface.co/datasets/dylanebert/iso3d/resolve/main/jpg@512/a_cat_statue.jpg"
    ],
    cache_examples=True,
    allow_duplication=True,
)
demo.queue().launch()


import shlex
import subprocess

import gradio as gr
import numpy as np
import spaces
import torch
from diffusers import DiffusionPipeline

subprocess.run(
    shlex.split(
        "pip install https://huggingface.co/spaces/dylanebert/LGM-mini/resolve/main/wheel/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl"
    )
)


pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/LGM-full",
    custom_pipeline="dylanebert/LGM-full",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")



@spaces.GPU
def run(image):
    input_image = np.array(image, dtype=np.float32) / 255.0
    splat = pipeline(
        "", input_image, guidance_scale=5, num_inference_steps=30, elevation=0
    )
    splat_file = "/tmp/output.ply"
    pipeline.save_ply(splat, splat_file)
    return splat_file



demo = gr.Interface(
    fn=run,
    title="LGM Tiny",
    description="An extremely simplified version of [LGM](https://huggingface.co/ashawkey/LGM). Intended as resource for the [ML for 3D Course](https://huggingface.co/learn/ml-for-3d-course/unit0/introduction).",
    inputs="image",
    outputs=gr.Model3D(),
    examples=[
        "https://huggingface.co/datasets/dylanebert/iso3d/resolve/main/jpg@512/a_cat_statue.jpg"
    ],
    cache_examples=True,
    allow_duplication=True,
)
demo.queue().launch()


#要通过 API 运行

import gradio as gr
from gradio_client import Client, file


def run(image_url):
    client = Client("dylanebert/LGM-tiny")
    image = file(image_url)
    result = client.predict(image, api_name="/predict")
    return result


demo = gr.Interface(
    fn=run,
    title="LGM Tiny API",
    description="An API wrapper for [LGM Tiny](https://huggingface.co/spaces/dylanebert/LGM-tiny). Intended as a resource for the [ML for 3D Course](https://huggingface.co/learn/ml-for-3d-course).",
    inputs=gr.Textbox(label="Image URL", placeholder="Enter image URL, e.g. https://huggingface.co/datasets/dylanebert/iso3d/resolve/main/jpg@512/a_cat_statue.jpg"),
    outputs=gr.Model3D(),
    examples=[
        "https://huggingface.co/datasets/dylanebert/iso3d/resolve/main/jpg@512/a_cat_statue.jpg"
    ],
    allow_duplication=True,
)
demo.queue().launch()