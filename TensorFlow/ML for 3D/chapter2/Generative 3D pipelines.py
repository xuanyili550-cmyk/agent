"""
================================================================================
 ML for 3D · Chapter 2 · 生成式 3D 流程(多视图扩散)（学习笔记描述）
================================================================================
 一句话：从"一张图/一句话"生成 3D——先用多视图扩散生成多个视角,再重建成 3D。
 本章讲：
   ① 用 diffusers 加载 multi-view-diffusion 自定义 pipeline。
   ② 单图 → 多视图一致的图像集。
   ③ 为后续 3D 重建(高斯泼溅/LGM)提供多视角输入。
 要点：多视图一致性是 image→3D 的关键中间步。
 说明：需 diffusers + CUDA(代码写死 .to("cuda"));参考为主。
================================================================================
"""

import torch
from diffusers import DiffusionPipeline

multi_view_diffusion_pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/multi-view-diffusion",
    custom_pipeline="dylanebert/multi-view-diffusion",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")

#运行多视图扩散流程。
import requests
from PIL import Image
from io import BytesIO


image_url = "https://huggingface.co/datasets/dylanebert/3d-arena/resolve/main/inputs/images/a_cat_statue.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image


import numpy as np

def create_image_grid(images):
    images = [Image.fromarray((img * 255).astype("uint8")) for img in images]

    width, height = images[0].size
    grid_img = Image.new("RGB", (2 * width, 2 * height))

    grid_img.paste(images[0], (0, 0))
    grid_img.paste(images[1], (width, 0))
    grid_img.paste(images[2], (0, height))
    grid_img.paste(images[3], (width, height))

    return grid_img

#将图像转换为归一化的 numpy 数组
image = np.array(image, dtype=np.float32) / 255.0
#将其传递给管道
images = multi_view_diffusion_pipeline("", image, guidance_scale=5, num_inference_steps=30, elevation=0)

create_image_grid(images)

import torch
from diffusers import DiffusionPipeline

multi_view_diffusion_pipeline = DiffusionPipeline.from_pretrained(
    "<your-user-name>/<your-model-name>",
    custom_pipeline="dylanebert/multi-view-diffusion",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")


import gradio as gr

def run(image):
  image = np.array(image, dtype=np.float32) / 255.0
  images = multi_view_diffusion_pipeline("", image, guidance_scale=5, num_inference_steps=30, elevation=0)

  images = [Image.fromarray((img * 255).astype("uint8")) for img in images]

  width, height = images[0].size
  grid_img = Image.new("RGB", (2 * width, 2 * height))

  grid_img.paste(images[0], (0, 0))
  grid_img.paste(images[1], (width, 0))
  grid_img.paste(images[2], (0, height))
  grid_img.paste(images[3], (width, height))

  return grid_img

demo = gr.Interface(fn=run, inputs="image", outputs="image")
demo.launch()


import gradio as gr
import numpy as np
import torch
from diffusers import DiffusionPipeline
from PIL import Image

multi_view_diffusion_pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/multi-view-diffusion",
    custom_pipeline="dylanebert/multi-view-diffusion",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")


def run(image):
    image = np.array(image, dtype=np.float32) / 255.0
    images = multi_view_diffusion_pipeline(
        "", image, guidance_scale=5, num_inference_steps=30, elevation=0
    )

    images = [Image.fromarray((img * 255).astype("uint8")) for img in images]

    width, height = images[0].size
    grid_img = Image.new("RGB", (2 * width, 2 * height))

    grid_img.paste(images[0], (0, 0))
    grid_img.paste(images[1], (width, 0))
    grid_img.paste(images[2], (0, height))
    grid_img.paste(images[3], (width, height))

    return grid_img


demo = gr.Interface(fn=run, inputs="image", outputs="image")
demo.launch()
