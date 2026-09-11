"""
================================================================================
 ML for 3D · Chapter 3 · 可微栅格化与高斯泼溅（学习笔记描述）
================================================================================
 一句话：把"3D→2D 投影渲染"做成可微的,梯度就能回传→用图像监督优化 3D 表示。
 本章讲：
   ① 高斯泼溅(3D Gaussian Splatting)：用一堆 3D 高斯点表示场景,投影+排序+混合成像。
   ② 可微渲染让重建/生成能端到端优化。
   ③ 与多视图扩散 + LGM 串起 image→3D。
 要点：可微渲染 = 用 2D 图像的 loss 反向优化 3D 参数。
 说明：需 diffusers/CUDA + 特殊 rasterizer 轮子;参考为主。
================================================================================
"""

#高斯飞溅是一种可微的栅格化技术。（下面是课程的伪代码：投影→排序→逐像素累加贡献，不是可运行代码）
# splat2d = splat.project_and_sort()
# for point in splat2d:
#     for pixel in image:
#         pixel += compute_contribution(point, pixel)

import torch
from diffusers import DiffusionPipeline

image_pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/multi-view-diffusion",
    custom_pipeline="dylanebert/multi-view-diffusion",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")


splat_pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/LGM",
    custom_pipeline="dylanebert/LGM",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")



import requests
from PIL import Image
from io import BytesIO

image_url = "https://huggingface.co/datasets/dylanebert/3d-arena/resolve/main/inputs/images/a_cat_statue.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image


import numpy as np
from google.colab import files

input_image = np.array(image, dtype=np.float32) / 255.
multi_view_images = image_pipeline("", input_image, guidance_scale=5, num_inference_steps=30, elevation=0)




splat = splat_pipeline(multi_view_images)

output_path = "/tmp/output.ply"
splat_pipeline.save_ply(splat, output_path)
files.download(output_path)




import gradio as gr

def run(image):
    input_image = image.astype("float32") / 255.0
    images = image_pipeline("", input_image, guidance_scale=5, num_inference_steps=30, elevation=0)
    splat = splat_pipeline(images)
    output_path = "/tmp/output.ply"
    splat_pipeline.save_ply(splat, output_path)
    return output_path

demo = gr.Interface(fn=run, inputs="image", outputs=gr.Model3D())
demo.launch()


