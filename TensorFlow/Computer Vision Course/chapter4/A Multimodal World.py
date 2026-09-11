"""
================================================================================
 CV Course · Chapter 4 · 多模态世界（学习笔记描述）
================================================================================
 一句话：真实信息是多模态的(视觉+文本+音频)，本章概览模态组合、数据集与任务。
 本章讲：
   ① 常见模态组合(视觉+文本/音频)与真实场景(信息图/字幕视频/语音笔记)。
   ② 多模态数据集(LAION-5B、VGG-Sound、IEMOCAP 等)。
   ③ 多模态任务：VQA、图像描述、OCR、文生图、文生视频等。
 要点：多模态 = 对齐不同模态的表示到同一空间(如 CLIP)。
 说明：概念综述为主(注释)，无需运行。
================================================================================
"""

#多模态世界
# 常见模式组合及实际案例：
#
# 视觉+文字：信息图表、表情包、文章、博客。
# 视觉+音频：与朋友进行 Skype 通话，双向对话。
# 视觉+听觉+文字：观看带有字幕的 YouTube 视频或电影，以及一般的社交媒体内容都是多模态的。
# 音频+文本：语音笔记、带歌词的音乐文件。
# 多模态数据集
# 由多种模态组成的数据集称为多模态数据集。以下是一些常见的模态组合示例：
#
# 视觉 + 文本：视觉故事数据集、视觉问答数据集、LAION-5B 数据集。
# 视觉 + 音频：VGG-Sound 数据集、RAVDESS 数据集、视听身份数据库 (AVID)。
# 视觉 + 音频 + 文本：RECOLA 数据库，IEMOCAP 数据集。
# 变体支持的一些多模态任务：
#
# 视觉+文字：
# 视觉问答或 VQA：帮助视障人士，高效图像检索，视频搜索，视频问答，文档 VQA。
# 图像转文本：图像描述、光学字符识别 (OCR)、Pix2Struct。
# 文本转图像：图像生成。
# 文本转视频：文本转视频编辑、文本转视频搜索、视频翻译、文本驱动的视频预测。
# 音频+文本：
# 自动语音识别（或语音转文本）：虚拟语音助手、字幕生成。
# 文本转语音：语音助手、广播系统。

#Multimodal Tasks and Models
# 视觉问答（VQA）
#
# 输入：图像-问题对（图像和关于该图像的问题）。
# 输出：在多项选择题场景中：从预定义的选项中选择正确答案，并生成相应的标签。在开放式问题场景中：根据图像和问题生成自由形式的自然语言答案。
# 任务：回答有关图像的问题。（大多数视觉问答模型将图像问题视为具有预定义答案的分类问题）。请参考以上示例。
# 视觉推理
#
# 输入：因具体的视觉推理任务而异：
# VQA 式任务：图像-问题对。
# 匹配任务：图片与文字描述。
# 蕴含任务：图像和文本对（可能包含多个语句）。
# 子问题任务：图像和主要问题，以及与感知相关的附加子问题。
# 输出：因任务而异：
# VQA：关于图像的问题的答案。
# 匹配题：判断文本内容与图像描述是否相符（真/假）。
# 蕴含关系：预测图像是否在语义上蕴含文本。
# 子问题：与感知相关的子问题的答案。
# 任务：对图像执行各种推理任务。请参考以上示例。

from PIL import Image
from transformers import pipeline

vqa_pipeline = pipeline(
    "visual-question-answering", model="Salesforce/blip-vqa-capfilt-large"
)

image = Image.open("elephant.jpeg")
question = "Is there an elephant?"

vqa_pipeline(image, question, top_k=1)


from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

processor = Pix2StructProcessor.from_pretrained("google/deplot")
model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")

url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    images=image,
    text="Generate underlying data table of the figure below:",
    return_tensors="pt",
)
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))


from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# prepare image + question
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cats are there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])


#文档可视化问答（DocVQA）
# 输入：
#
# 文档图像：包含文本、布局和视觉元素的文档的扫描或数字图像。
# 关于文档的问题：以文本格式提出的自然语言问题。
# 任务：
#
# 分析和理解：DocVQA 模型必须处理文档中的视觉和文本信息，才能充分理解其内容。
# 推理和推断：该模型需要建立视觉元素、文本和问题之间的关系，以得出相关的结论。
# 生成自然语言答案：模型必须以自然语言文本格式生成清晰、简洁、准确的问题答案。请参考以上示例。
# 输出：问题的答案：直接回答查询并准确反映文档中信息的文本回复。

from transformers import pipeline
from PIL import Image

pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

question = "What is the purchase amount?"
image = Image.open("your-document.png")

pipe(image=image, question=question)

## [{'answer': '20,000$'}]


from transformers import pipeline
from PIL import Image

pipe = pipeline(
    "document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa"
)

question = "What is the purchase amount?"
image = Image.open("your-document.png")

pipe(image=image, question=question)

## [{'answer': '20,000$'}]

from huggingface_hub import hf_hub_download
import re
from PIL import Image

from transformers import NougatProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import torch

processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# prepare PDF image for the model
filepath = hf_hub_download(
    repo_id="hf-internal-testing/fixtures_docvqa",
    filename="nougat_paper.png",
    repo_type="dataset",
)
image = Image.open(filepath)
pixel_values = processor(image, return_tensors="pt").pixel_values

# generate transcription (here we only generate 30 tokens)
outputs = model.generate(
    pixel_values.to(device),
    min_length=1,
    max_new_tokens=30,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)
# note: we're using repr here such for the sake of printing the \n characters, feel free to just print the sequence
print(repr(sequence))

from transformers import pipeline

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")

# [{'generated_text': 'a soccer game with a player jumping to catch the ball '}]

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image

processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)
#CLIP（对比语言-图像预训练

from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt",
    padding=True,
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(
    dim=1
)  # we can take the softmax to get the label probabilities
#OWL-ViT： OWL-ViT（用于开放世界定位的视觉转换器）是一个强大的目标检测模型，它基于标准的视觉转换器架构，并使用大规模图像-文本对进行训练。

import requests
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process_object_detection(
    outputs=outputs, threshold=0.1, target_sizes=target_sizes
)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Print detected objects and rescaled box coordinates
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}"
    )


#文本转图像生成
# pip install diffusers --upgrade
# pip install invisible_watermark transformers accelerate safetensors
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to("cuda")

prompt = "An astronaut riding a unicorn"

images = pipe(prompt=prompt).images[0]
#CLIP 和亲属
# 预剪辑
# 本部分将探讨CLIP之前多模态人工智能领域的创新尝试。重点关注那些利用深度学习在该领域取得重大进展的具有影响力的论文：
#
# Ngiam等人（2011）的论文《多模态深度学习》：该论文展示了深度学习在多模态输入中的应用，强调了神经网络在整合不同数据类型方面的潜力。它为未来多模态人工智能的创新奠定了基础。
#
# 多模态深度学习
# Karpathy 和 Fei-Fei (2015) 的论文“Deep Visual-Semantic Alignments for Generating Image Descriptions”提出了一种将文本数据与特定图像区域对齐的方法，增强了多模态系统的可解释性，并加深了对复杂视觉文本关系的理解。
#
# 用于生成图像描述的深度视觉语义对齐
# Vinyals 等人 (2015) 的论文“Show and Tell: A Neural Image Caption Generator”：该论文通过展示如何将 CNN 和 RNN 结合起来，将视觉信息转换为描述性语言，标志着实用多模态人工智能迈出了重要一步。
#
# 展示与讲解：一种神经图像描述生成器
# 剪辑后
# CLIP 的出现为多模态模型带来了新的维度，如下发展所示：
#
# CLIP： OpenAI 的 CLIP 改变了游戏规则，它从大量的互联网文本图像对中学习，并实现了零样本学习，这与早期的模型形成了鲜明对比。
#
# 夹子
# GroupViT： GroupViT 在分割和语义理解方面进行了创新，并将这些方面与语言相结合，展现了语言和视觉的先进融合。
#
# GroupViT
# BLIP： BLIP 引入了视觉和语言之间的双向学习，突破了从视觉输入生成文本的界限。
#
# BLIP
# OWL-VIT： OWL-VIT 专注于以对象为中心的表示，增进了对图像中对象在文本上下文中的理解。
#
# OWL-VIT


#损失
#
# 培训目标
# 收缩性损失
# 对比损失是对比学习中最先使用的训练目标之一。它以一对相似或不相似的样本作为输入，目标是将相似的样本在嵌入空间中映射得更接近，并将不相似的样本推开。
#
# 从技术角度来说，假设我们有一个输入样本列表。
# x
# n
# x
# n
# ​
# 来自多个类别。我们想要一个函数，其中来自同一类别的样本在嵌入空间中的嵌入位置相近，而来自不同类别的样本则相距甚远。将其转化为数学方程式，我们得到：
# L
# =
# 1
# [
#     是
#     我
# =
# 是
# j
# ]
# ∣
# ∣
# x
# 我
# −
# x
# j
# ∣
# ∣
# 2
# +
# 1
# [
#     是
#     我
# ≠
# 是
# j
# ]
# 米
# 一个
# x
# （
# 0
# ，
# ϵ
# −
# ∣
# ∣
# x
# 我
# −
# x
# j
# ∣
# ∣
# 2
# ）
# L = 1[y
# 我
# ​
# =是
# j
# ​
# ] ∣∣ x
# 我
# ​
# −x
# j
# ​
# ∣ ∣
# 2
# +1[y
# 我
# ​
#
# 
# =是
# j
# ​
# ] max(0,​ϵ−∣∣ x
# 我
# ​
# −x
# j
# ​
# ∣ ∣
# 2
# ）
#
# 简单来说：
#
# 如果样本相似
# 是
# 我
# =
# 是
# j
# 是
# 我
# ​
# =是
# j
# ​
# 然后我们最小化该项
# ∣
# ∣
# x
# 我
# −
# x
# j
# ∣
# ∣
# 2
# ∣∣ x
# 我
# ​
# −x
# j
# ​
# ∣ ∣
# 2
# 这对应于它们的欧氏距离，也就是说，我们希望使它们更接近；
# 如果样本不相似
# （
# 是
# 我
# ≠
# 是
# j
# ）
# （y
# 我
# ​
#
# 
# =是
# j
# ​
# ）然后我们最小化该项
# 米
# 一个
# x
# （
# 0
# ，
# ϵ
# −
# ∣
# ∣
# x
# 我
# −
# x
# j
# ∣
# ∣
# 2
# ）
# max （0 ，​ϵ−∣∣ x
# 我
# ​
# −x
# j
# ​
# ∣ ∣
# 2
# ）这等价于最大化它们的欧氏距离，直到达到某个极限。
# ϵ
# ϵ也就是说，我们希望让他们彼此保持距离。

#对比语言-图像预训练（CLIP）
# 用例
# CLIP 可用于多种应用场景。以下是一些值得注意的应用案例：
#
# 零样本图像分类；
# 相似性搜索；
# 扩散模型条件。
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt",
    padding=True,
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

#多模态文本生成（BLIP）
# CapFilt：字幕和过滤
# 由于多模态模型需要庞大的数据集，因此通常需要从互联网上抓取图像和替代文本（alt-text）对。然而，替代文本往往无法准确描述图像的视觉内容，使其成为噪声信号，不利于学习视觉语言对齐。因此，BLIP论文引入了一种图像描述过滤机制（CapFilt）。该机制由一个深度学习模型和一个用于生成图像描述的图像描述生成模型组成。这两个模型首先使用人工标注的数据集进行微调。研究发现，使用CapFilt清理数据集比直接使用网络数据集性能更优。有关此过程的更多细节，请参阅BLIP论文。
# BLIP架构和培训
# BLIP架构结合了视觉编码器和多模态混合编码器-解码器（MED），能够灵活处理视觉和文本数据。其结构如下图所示，图中特征部分（颜色相同的模块共享参数）：
#
# Vision Transformer (ViT)：这是一个简单的视觉转换器，具有自注意力、前馈模块和用于嵌入表示的 [CLS] 标记。
# 单模态文本编码器：类似于 BERT 的架构，它使用 [CLS] 标记进行嵌入，并采用 CLIP 等对比损失来对齐图像和文本表示。
# 基于图像的文本编码器：它将 [CLS] 标记替换为 [Encode] 标记。交叉注意力层能够整合图像和文本嵌入，从而创建多模态表示。它采用线性层来评估图像-文本对的一致性。
# 基于图像的文本解码器：该解码器用因果自注意力代替双向自注意力，通过交叉熵损失以自回归的方式进行训练，用于生成图像描述或回答视觉问题等任务。
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "Question: How many remotes are there? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(
    device, torch.float16
)
outputs = model.generate(**inputs)
text = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text)
#
# 多模态目标检测（OWL-ViT）
# 介绍
# 目标检测是计算机视觉领域的一项关键任务，近年来随着 YOLO 等模型的出现（原始论文，最新代码版本）取得了显著进展。然而，像 YOLO 这样的传统模型在检测训练数据集之外的目标时存在局限性。为了解决这个问题，人工智能领域开始致力于开发能够识别更广泛目标的模型，由此催生了类似于 CLIP 的目标检测模型。
#
# OWL-ViT：增强功能和特性
# OWL-ViT 代表了开放词汇目标检测领域的一次飞跃。它首先进行类似于 CLIP 的训练阶段，重点在于使用对比损失函数构建视觉和语言编码器。这一基础阶段使模型能够学习视觉和文本数据的共享表征空间。
#
# 目标检测的微调
# OWL-ViT 的创新之处在于其目标检测的微调阶段。与 CLIP 中使用的标记池化和最终投影层不同，OWL-ViT 采用对每个输出标记进行线性投影来获得每个目标的图像嵌入。这些嵌入随后用于分类，而边界框坐标则通过一个小型多层感知器 (MLP) 从标记表示中导出。这种方法使得 OWL-ViT 能够检测图像中的目标及其空间位置，相比传统的目标检测模型而言，这是一个显著的进步。


import requests
from PIL import Image, ImageDraw
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = [["a photo of a cat", "a photo of a dog", "remote control", "cat tail"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.Tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=0.1
)
i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Create a draw object
draw = ImageDraw.Draw(image)

# Draw each bounding box
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}"
    )
    # Draw the bounding box on the image
    draw.rectangle(box, outline="red")

# Display the image
image