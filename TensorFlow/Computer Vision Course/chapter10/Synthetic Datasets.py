"""
================================================================================
 CV Course · Chapter 10 · 合成数据集(3D 渲染造数据)（学习笔记描述）
================================================================================
 一句话：真实标注贵又少，用 3D 渲染器(Blender/Unity)造"越真越好"的合成训练数据。
 本章讲：
   ① 基于物理的渲染(PBR)：模拟光照/材质让合成图逼真。
   ② 用 Blender Cycles / Unity 生成带自动标注的图像。
   ③ 合成数据的价值与 sim-to-real 差距。
 要点：合成数据自带完美标注(分割/深度/位姿)，但要缩小与真实的域差。
 说明：需 Blender/Unity 等外部工具；参考为主。
================================================================================
"""

# # Using a 3D Renderer to Generate Synthetic Data
#
# When creating computer-generated images to use as synthetic training data, ideally we want the images to look as realistic as possible.
# Physically Based Renderers (PBR) such as [Blender Cycles](https://www.blender.org)
# or [Unity](https://unity.com) help to create images that are super realistic and look and feel just like they do in the real world.
#
# Imagine you're creating an image of a shiny apple. Now, when you color that apple, you want it to look realistic, right?
# That's where something called PBR comes in.
#
# ![apple](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/apple.jpg)
#
# Okay, let's break it down:
#
# _Colors and Light:_
#
# - When light shines on objects, it interacts with them in different ways. PBR tries to figure out and simulate this interaction.
# - Think about how light hits the apple. Some parts may be brighter where the light directly hits, and some parts might be darker
#   where the light is blocked or doesn't reach as much.
#
# _Materials:_
#
# - Different materials react to light differently. For example, a shiny metallic surface reflects light more than a soft, matte fabric.
# - PBR takes into account the material of an object, so if you're rendering a metal vase, it will reflect light differently than a fluffy teddy bear.
#
# _Textures:_
#
# - PBR uses textures to add details like bumps, scratches, or tiny grooves on the surface of objects. This makes things look more real because, in the real world, very few things are perfectly smooth.
#
# _Realism:_
#
# - PBR aims to make things look as close to real life as possible. It does this by considering how light behaves in reality, how different materials interact with light, and how surfaces have small imperfections.
#
# _Layers of Light:_
#
# - Imagine you're looking at a glass of water. PBR will try to simulate the way light passes through the water and how it might distort what you see.
# - It considers how multiple layers of how light interact with different parts of an object, making the rendered image more realistic.
#
# PBR also simplifies the workflow. Instead of manually tweaking many parameters to get the right look,
# you can use a set of standardized materials and lighting models.
# This makes the process more intuitive and user-friendly.
#
# Now, think about training AI models like those used in computer vision.
# If you're teaching a computer to recognize objects in images, it's beneficial to have a diverse set of
# images that closely mimic real-world scenarios. PBR helps in generating synthetic data that looks so real that it can be used
# to train computer vision models effectively.
#
# There are several 3D rendering engines that you can use for PBR, including [Blender Cycles](https://www.blender.org)
# or [Unity](https://unity.com). We are going to focus on Blender because it is open source and there are a lot of resources about Blender.
#
# ## Blender
#
# Blender is a powerful, open-source 3D computer graphics software used for creating animated films, visual effects, art, 3D games, and more.
# It encompasses a wide range of features, making it a versatile tool for artists, animators, and developers.
# Let's start off by walking through a basic example of rendering a synthetic image of an elephant.
#
# Here are the essential steps:
#
# - Create the elephant model. The one shown below was created with the [Metascan](https://metascan.ai) app using Photogrammetry.
#   Photogrammetry is a way of turning regular photos into a 3D model. It's like taking a bunch of pictures of your toy from different angles
#   and then using those pictures to make a computer version of it.
# - Create the background - this was a multi-step process. See [here](https://github.com/kfahn22/Synthetic-Data-Creation-in-Blender/tree/main/BACKGROUND)
#   for a more detailed explanation.
# - Adjust the lighting and camera positions.
# - Fix the location and rotation of the elephant so that it fits within the frame (or camera view).
#
# Here is the elephant image generated in Blender:
#
# ![elephant image](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-PBR/rendered_elephant.png)
#
# It is not completely photorealistic, but probably close enough to train a model to monitor elephant populations. Of course, to do
# that we need to create a large dataset of synthetic elephant images! You can use the Blender python environment
# [bpy](https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html) to render
# a large number of images with the location and rotation of the elephant randomized.
# You can also use a script to help with segmentation, depth, normal, and pose estimation.
#
# Great! How do we get started?
#
# Unfortunately, there is a pretty steep learning curve associated with Blender. None of the steps are too complicated,
# but wouldn't it be nice if we could render the dataset without trying to figure all of this out?
# Luckily for us, there is a library called BlenderProc that has all the scripts we need to render realistic synthetic data
# and annotations and it is built on top of Blender.
#
# ## BlenderProc
#
# The BlenderProc pipeline was introduced in [BlenderProc](https://arxiv.org/abs/1911.01911), Denninger, et. al. and is a modular pipeline built on top of [Blender](https://www.blender.org/).
# It can be used to generate images in a variety of use cases, including segmentation, depth, normal and pose estimation.
#
# It is specifically created to help in the generation of realistic looking images for the training of convolutional neural networks.
#  It has the following properties which make it a great choice for synthetic data generation:
#
# - Procedural Generation: Enables the automated creation of complex 3D scenes with variations using procedural techniques.
# - Simulation: Supports the integration of simulations, including physics simulations, to enhance realism.
# - Large-Scale Generation: Designed to handle large-scale scene generation efficiently, making it suitable for diverse applications.
# - Automation and Scalability:
#   - Scripting: Allows users to automate the generation process by employing python scripts to tailor BlenderProc to their specific needs and configure parameters.
#   - Parallel Processing: Supports parallel processing for scalability, making it efficient for generating a large number of scenes.
#
# You can install BlenderProc via pip:
#
# ```bash
# pip install blenderProc
# ```
#
# Alternately, you can clone the official [BlenderProc repository](https://github.com/DLR-RM/BlenderProc) from GitHub using Git:
#
# ```bash
# git clone https://github.com/DLR-RM/BlenderProc
# ```
#
# BlenderProc must be run inside the blender python environment (bpy), as this is the only way to access the Blender API.
#
# ```bash
# blenderproc run
# ```
#
# You can check out this notebook to try BlenderProc in Google Colab, demos the basic examples provided [here](https://github.com/DLR-RM/BlenderProc/tree/main/examples/basics).
# Here are some images rendered with the basic example:
#
# ![colors](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-PBR/colors.png)
# ![normals](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-PBR/normals.png)
# ![depth](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-PBR/depth.png)
#
# ## Blender Resources
#
# - [User Manual](https://docs.blender.org/manual/en/latest/0)
# - [Awesome-blender -- Extensive list of resources](https://awesome-blender.netlify.app)
# - [Blender Youtube Channel](https://www.youtube.com/@BlenderOfficial)
#
# ### The following video explains how to render a 3D syntehtic dataset in Blender:
#
# ### The following video explains how to create a 3D object using Photogrammetry:
#
# ## Papers / Blogs
#
# - [Developing digital twins of multi-camera metrology systems in Blender](https://iopscience.iop.org/article/10.1088/1361-6501/acc59e/pdf_)
# - [Generate Depth and Normal Maps with Blender](https://www.saifkhichi.com/blog/blender-depth-map-surface-normals)
# - [Object detection with synthetic training data](https://medium.com/rowden/object-detection-with-synthetic-training-data-f6735a5a34bc)
#
# ## BlenderProc Resources
#
# - [BlenderProc Github Repo](https://github.com/DLR-RM/BlenderProc)
# - [BlenderProc: Reducing the Reality Gap with Photorealistic Rendering](https://elib.dlr.de/139317/1/denninger.pdf)
# - [Documentation](https://dlr-rm.github.io/BlenderProc/)
#
# ### The following video provides an overview of the BlenderProc pipeline:
#
# ## Papers
#
# - [3D Menagerie: Modeling the 3D Shape and Pose of Animals]()
# - [Fake It Till You Make It: Face analysis in the wild using synthetic data alone]()
# - [Object Detection and Autoencoder-Based 6D Pose Estimation for Highly Cluttered Bin Picking](https://arxiv.org/pdf/2106.08045.pdf)
# - [Learning from Synthetic Animals](https://arxiv.org/abs/1912.08265)
#
# # Synthetic Data Generation Using DCGAN
#
# We learned in Unit 5 that a GAN is a framework in machine learning where two neural networks, a Generator and a Discriminator, are in a constant duel. The Generator creates synthetic images, and the Discriminator tries to distinguish between real and fake images. They keep improving through this adversarial process, with the Generator getting better at creating realistic images, and the Discriminator getting better at distinguishing between fake and real images.
#
# We now will look at how we can use a GAN to generate medical images, a domain that is challenged with small datasets, privacy concerns, and a limited amount of annotated samples. Researchers have used GANs to generate synthetic images such as lung X-ray images, retina images, brain scans, and liver images. In [GAN-based synthetic brain PET image generation](https://braininformatics.springeropen.com/counter/pdf/10.1186/s40708-020-00104-2.pdf), the authors created brain PET images for three different stages of Alzheimer’s disease. [GAN-based Synthetic Medical Image Augmentation for increased CNN Performance in Liver Lesion Classification](https://arxiv.org/abs/1803.01229) generated synthetic liver images. [BrainGAN: Brain MRI Image Generation and Classification Framework Using GAN Architectures and CNN Models](https://www.mdpi.com/1424-8220/22/11/4297) developed a framework for generating brain MRI images using multiple GAN architectures, and [A Novel COVID-19 Detection Model Based on DCGAN and Deep Transfer Learning](https://www.sciencedirect.com/science/article/pii/S1877050922007463) used DCGAN to generate synthetic lung X-ray images to aid in COVID-19 detection.
#
# ## DCGAN (Deep Convolutional Generative Adversarial Network)
#
# DCGAN was proposed in [Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Radford, et al. and is the model many researchers have used to generate synthetic medical images. We are going to use it to generate synthetic lung images. Before we use DCGAN to train the model, we will briefly review its architecture. The generator network takes random noise as input and generates synthetic lung images, while the discriminator network tries to distinguish between real and synthetic images. It uses convolutional layers in both the generator and discriminator to capture spatial features effectively. DCGAN also replaces max-pooling with strided convolutions to downsample the spatial dimensions.
#
# The generator has the following model architecture:
#
# - The input is a vector a 100 random numbers and the output is a image of size 128*128*3.
# - The model has 4 convolutional layers:
#   - Conv2D layer
#   - Batch Normalization layer
#   - ReLU activation
# - Conv2D layer with Tanh activation.
#
# The discriminator has the following model architecture:
#
# - The input is an image and the output is a probability indicating whether the image is fake or real.
# - The model has one convolutional layer:
#   - Conv2D layer
#   - Leaky ReLU activation
# - Three convolutional layers with:
#   - Conv2D layer
#   - Batch Normalization layer
#   - Leaky ReLU activation
# - Conv2D layer with Sigmoid.
#
# **Data Collection**
#
# First, we need to obtain a [dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2) of real lung images. We will be downloading the [Chest X-Ray Images (Pneumonia)](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia) dataset from the Hugging Face Hub.
#
# Here is some information about the dataset:
#
#     * From [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867418301545%3Fshowall%3Dtrue):
#
#     * The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
#
#     * Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.
#
#     * For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.
#
# We will start by logging into the Hugging Face hub.
#
# ```python
# from huggingface_hub import notebook_login
#
# notebook_login()
# ```
#
# Next, we will load the dataset.
#
# ```python
# from datasets import load_dataset
#
# dataset = load_dataset("hf-vision/chest-xray-pneumonia")
# ```
#
# We will preprocess the lung images by resizing and normalizing the pixel values.
#
# ```python
# import torchvision.transforms as transforms
# from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
#
# transform = Compose(
#     [
#         transforms.Resize(image_size),
#         transforms.CenterCrop(image_size),
#         transforms.ToTensor(),
#     ]
# )
# ```
#
# During training, the generator aims to produce synthetic lung images that are indistinguishable from real images, while the discriminator learns to correctly classify the images as real or synthetic. We start by initializing the generator with random noise and will train for 100 epochs.
#
# Let's visualize the progress:
#
# ![trainig-gif](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/dcgan_training_animation.gif)
#
# ## How did we do?
#
# Here are 64 "good" synthetic images, defined as receiving a "real" label from the discriminator with a 70% probability.
#
# ![lung-images](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/good_images.png)
#
# We can see that some of the synthetic lung images look good, but others are hazy. There are some important things to mention. First, researchers generating synthetic medical images typically employ a "human in the middle"--in this case, a expert radiologist--to evaluate the synthetic images. Then, only the images that fool the expert are included with the real data to train the model. Second, the generated images that appear to look OK (at least to an amateur) look very similar. This is another known issue with GANs - they can suffer from "mode collapse." Essentially, this happens when the generator starts producing the same type of output. Think of it as someone who has gotten a lot of praise for making chocolate chip cookies, and therefore, gets _really, really_ good at making those cookies but can't make any other type of cookies.
#
# Given the known challenges associated with employing GANs to train high-quality medical images, some researchers have explored the use of diffusion models to generate lung images. Medfusion, a conditional latent DDPM for medical images, was proposed in [Diffusion Probabilistic Models Beat GAN On Medical 2D Images](https://arxiv.org/pdf/2212.07501.pdf). [Synthetically Enhanced: Unveiling Synthetic Data's Potential In Medical Imaging Research](https://arxiv.org/pdf/2311.09402.pdf), Khosravi et. al find that using a mix of real and synthetically generated lung images using a diffusion process improved model performance.
#
# ## Resources and Further Reading
#
# - [A Novel COVID-19 Detection Model Based on DCGAN and Deep Transfer Learning](https://www.sciencedirect.com/science/article/pii/S1877050922007463)
# - [Augmentation_Gan](https://github.com/rossettisimone/AUGMENTATION_GAN)
# - [BrainGAN: Brain MRI Image Generation and Classification Framework Using GAN Architectures and CNN Models](https://www.mdpi.com/1424-8220/22/11/4297)
# - [dentifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.cell.com/action/showPdf?pii=S0092-8674%2818%2930154-5)
# - [Diffusion Probabilistic Models beat GANs on Medical Images](https://arxiv.org/abs/2212.07501)
# - [DR-DCGAN: A Deep Convolutional Generative Adversarial Network (DC-GAN) for Diabetic Retinopathy Image Synthesis]()
# - [Deepfake Image Generation for Improved Brain Tumor Segmentation](https://aps.arxiv.org/abs/2307.14273)
# - [GAN Lab](https://poloclub.github.io/ganlab/)
# - [GANs for Medical Image Synthesis: An Empirical Study](https://arxiv.org/abs/2105.05318)
# - [Medfusion Github repo](https://github.com/mueller-franzes/medfusion)
# - [Medical image editing in the latent space of Generative Adversarial Networks](https://www.sciencedirect.com/science/article/pii/S2666521221000168?ref=pdf_download&fr=RR-2&rr=833e48fa5e777142)
# - [Medical Image Synthesis with Context-Aware Generative Adversarial Networks](https://arxiv.org/abs/1612.05362)
# - [MedSynAnalyzer](https://github.com/ayanglab/MedSynAnalyzer)[dcgan_faces_tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
# - [pytorch-fid](https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py)
# - [StudioGAN: A Taxonomy and Benchmark of GANs for Image Synthesis](https://arxiv.org/abs/2206.09479)
# - [PyTorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN)
# - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
#
# # Synthetic Data Generation with Diffusion Models
#
# Imagine trying to train a model for tumor segmentation. As it's hard to gather data for medical imaging, it'd be really difficult for the model to converge.
# Ideally, we expect to have at least enough data to build a simple baseline, but what if you have just a few samples? Synthetic data generation methods try to solve this dilemma, and now we have many more options with the boom of generative models!
#
# As you've seen in the previous sections, it is possible to use generative models such as DCGAN to generate synthetic images. In this section, we will focus on diffusion models using [diffusers](https://huggingface.co/docs/diffusers/index)!
#
# ## Recap of the Diffusion Models
#
# Diffusion models are generative models that gained significant popularity in recent years, thanks to their capabilities to produce high-quality images. Nowadays, they're widely used in image, video, and text synthesis.
#
# A diffusion model works by learning to denoise random Gaussian noise step-by-step. The training process requires adding Gaussian noise to the input samples and letting the model learn denoising.
#
# Diffusion models are generally conditioned to a kind of input besides the data distribution, such as text prompts, images or even audio. Additionally, it's also possible to [build an unconditional generator](https://huggingface.co/docs/diffusers/training/unconditional_training).
#
# There are many underlying concepts behind the model's inner workings but the simplified version is similar to this:
#
# Firstly, the model adds noise to the input and processes it.
#
# ![noising](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/noising.jpg?download=true)
#
# Then the model learns to denoise the given data distribution.
#
# ![denoising](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/denoising.jpg?download=true)
#
# We won't dive deep into the theory, but understanding how a diffusion model works will come really handy when we need to pick a technique to generate synthetic data for our use case.
#
# ## Text-To-Image Diffusion Model: Stable Diffusion
#
# Essentially, the way Stable Diffusion (SD) works is the same as we mentioned above. It uses three main components that help it to produce high-quality images.
#
# 1. **The diffusion process:** The input is processed multiple times to generate useful information about the image. The "usefulness" is learned while training the model.
#
# 2. **Image encoder and decoder model:** Lets the model compress images from pixel space to a smaller dimensional space, abstracting meaningless information while increasing performance.
#
# 3. **Optional conditional encoder:** This component is used to condition to generation process on an input. This extra input can be text prompts, images, audio, and other representations. Originally, it was a text encoder.
#
# So while the general workflow looks like this:
# ![general stable diffusion workflow](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/general-workflow.png?download=true)
#
# Originally we used a text encoder to condition the model on our prompts:
# ![original stable diffusion workflow](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/stable-diffusion-workflow.png?download=true)
#
# This short overview is just the tip of the iceberg! If you wish to dive deep into the theory behind stable diffusion (or diffusion models), you can out the [further reading section](#further-reading-about-stable-diffusion)!
#
# [diffusers](https://huggingface.co/docs/diffusers/index) provide us ready to use pipelines for different tasks, such as:
#
# | Task | Description | Pipeline |
# |------|-------------|----------|
# | Unconditional Image Generation | generate an image from Gaussian noise | [unconditional_image_generation](https://huggingface.co/docs/diffusers/using-diffusers/unconditional_image_generation) |
# | Text-Guided Image Generation | generate an image given a text prompt |[conditional_image_generation](https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation) |
# | Text-Guided Image-to-Image Translation | adapt an image guided by a text prompt | [img2img](https://huggingface.co/docs/diffusers/using-diffusers/img2img) |
# | Text-Guided Image-Inpainting | fill the masked part of an image given the image, the mask and a text prompt | [inpaint](https://huggingface.co/docs/diffusers/using-diffusers/inpaint) |
# | Text-Guided Depth-to-Image Translation | adapt parts of an image guided by a text prompt while preserving structure via depth estimation | [depth2img](https://huggingface.co/docs/diffusers/using-diffusers/depth2img) |
#
# There's also a complete list of supported tasks that you can find from the [Diffusers Summary](https://huggingface.co/docs/diffusers/api/pipelines/overview#diffusers-summary) table.
#
# This means we have many tools under our belt to generate synthetic data!
#
# ## Approaches to Synthetic Data Generation
#
# There are generally three cases for needing synthetic data:
#
# **Extending an existing dataset:**
#
# - **There are not enough samples:** A nice example is a medical imaging dataset such as [DDSM](https://www.mammoimage.org/databases/) (Digital Database for Screening Mammography, ~2500 samples), a small amount of samples makes it harder to build a model for further analysis. It's also rather expensive to build such medical imaging datasets.
#
# **Creating a Dataset from Scratch**:
#
# - **There aren't any samples at all:** Let's assume that you want to build a weapon detection system on top of CCTV video streams. But there aren't any samples for the specific weapon you want to detect, you can use similar observations in different settings to apply style transfer to make them look like CCTV streams!
#
# **Preserving Privacy**:
#
# - Hospitals collect huge amounts of data on patients, surveillance cameras capture raw information about individuals' faces and activities, and all of these introduce a potential infringe on privacy. We can use diffusion models to generate privacy-preserving datasets to develop our solutions, without giving up on anyone's privacy rights.
#
# There are different methods for us to utilize a text-to-image diffusion model to generate customized outputs. For example, by simply utilizing the pre-trained diffusion model (such as [Stable Diffusion XL](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)), you can try to construct a nice prompt to generate images. But the quality of the generated images may not be consistent and it might be really difficult to construct such a prompt for your specific use case.
#
# You will usually be required to change some parts of the model to generate the personalized output you want, here are a few techniques that you can use for that:
#
# **Training with [Textual Inversion](https://huggingface.co/docs/diffusers/main/en/training/text_inversion):**
#
# ![textual-inversion](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/textual_inversion.png?download=true)
#
# Textual inversion is a technique that works by interfering with the text embeddings in the model's architecture. You can add a new token to the vocabulary and then fine-tune the embeddings using a few examples.
#
# By providing samples corresponding to the new token, we try to optimize the embeddings to capture the characteristics of the object.
#
# **Training a [LoRA (Low-Rank Adaptation)](https://huggingface.co/docs/diffusers/main/en/training/lora?installation=PyTorch) model:**
#
# LoRA has targeted the problem of fine-tuning LLMs. It represents the weight updates with two smaller update matrices through low-rank decomposition, which has substantially less amount of parameters. Then these matrices can be trained to adapt the new data!
#
# The base model's weights remain frozen in the whole process, so we just train the new update matrices. Then lastly, the adapted weights are combined with the original weights.
#
# This means training LoRA is essentially much faster than full model fine-tuning! As an important note, it's also possible to combine LoRA with other techniques since it can be added on top of the model itself.
#
# **Training with [DreamBooth](https://huggingface.co/docs/diffusers/main/en/training/dreambooth):**
#
# ![dreambooth](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/dreambooth.png?download=true)
#
# DreamBooth is a technique to fine-tune the model to personalize the outputs. Given a few images of a subject, it lets you fine-tune a pre-trained text-to-image model. The main idea is to associate a unique identifier with that specific subject.
#
# For training, we use the tokens in the vocabulary and build a dataset using preferably a rare-token identifier. Because if you choose a rather common identifier, the model would also have to learn to disentangle from their original meaning.
#
# In the original paper, authors find the rare tokens in the vocabulary and then choose identifiers from those. This reduces the risk of an identifier having a strong prior. It is also stated that the best results were achieved by fine-tuning all the layers of the model.
#
# **Training with [Custom Diffusion](https://huggingface.co/docs/diffusers/main/en/training/custom_diffusion):**
#
# ![custom diffusion](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/custom_diffusion.png?download=true)
#
# Custom Diffusion is a really powerful technique to personalize the model. It requires just a few samples like the previously mentioned methods, but its power lies in being able to learn multiple concepts at the same time!
#
# It works by training just a part of the diffusion process and the text encoder we mentioned above, meaning that there are fewer parameters to optimize. Thus, this method also enables fast fine-tuning!
#
# ## Real-world Examples of Using Diffusion Models for Dataset Generation
#
# There are many unique cases where diffusion models are used to generate synthetic datasets!
#
# Find them below.
#
# - [Apple Images for Apple Detection in Orchards](https://arxiv.org/abs/2306.09762)
# - [Automatically Labeled Polyp Images for Medical Image Segmentation](https://arxiv.org/abs/2310.16794)
# - [3D Medical Image Generation with DDPMs](https://www.nature.com/articles/s41598-023-34341-2.pdf)
# - [Generating Transmission Line Images with DDPM](https://ieeexplore.ieee.org/document/10281144)
# - [Synthetic Aerial Dataset for Unmanned Aerial Vehicle Detection](https://ieeexplore.ieee.org/document/10195076)
# - [Differentially Private Diffusion Models for Privacy-preserving Synthetic Image Generation](https://arxiv.org/pdf/2302.13861.pdf)
#
# ## Further reading about Stable Diffusion
#
# - [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
# - [Diffusion Explainer: Stable Diffusion Explained with Visualization](https://poloclub.github.io/diffusion-explainer/)
# - [Introduction to Diffusion Models](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)
# - [FastAI, Practical Deep Learning for Coders - Lesson 9: Stable Diffusion](https://course.fast.ai/Lessons/lesson9.html)
# - [Original paper](https://arxiv.org/abs/2112.10752)
#
# # Challenges and Opportunities Associated With Using Synthetic Data
#
# Training machine learning models requires vast amounts of data. Synthetic data can help by addressing privacy issues, augmenting limited data, and correcting imbalances in the real data. We have learned how to generate synthetic data using several different methods. Before using synthetic data to train a model, however, there are several important things that need to be considered.
#
# ## Overfitting the Model
#
# Overfitting occurs when a machine learning model learns the training data so well that it doesn't perform well on new, unseen data.
# It's akin to learning a specific way to solve a problem but then encountering a new situation where the strategy doesn't work. If the process of generating synthetic data is too simple or there are are overly-consistent patterns, your model might overfit to the limited variations present in the synthetic data. As a very simple example, suppose you trained a model using a synthetic dataset of 25 red circles and 25 blue squares. The model will probably learn to associate circles with the color red and squares with the color blue. This model would likely fail if presented with a red square.
#
#   Be sure to double check that your dataset doesn't have the following types of
#   patterns!
#
# _Overly Consistent Color_
# ![consistent-color](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-overfit/overfit-color.jpg)
#
# _Overly Consistent Size_
# ![consistent-size](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-overfit/overfit-size.jpg)
#
# _Overly Consistent Background_
# ![consistent-background](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-overfit/overfit-background.jpg)
#
# _Overly Consistent Location_
# ![consistent-location](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-overfit/overfit-location.jpg)
#
# ## Are there biases in the synthetic data?
#
# If the process of generating synthetic data has biases or inaccuracies, your model may unintentionally learn and perpetuate those biases. Beware of the following pitfalls:
#
# **Limited Diversity**
#
# One challenge is that synthetic data may fail to adequately represent the complexity and diversity of the real data. The shape example might seem trivial, but there are lots of situations where failing to account for the wide variety of people, places, animals, or objects will result in a model that doesn't perform well. For example, suppose you wanted to train a model to monitor the population of an endangered species, such as aye-aye lemurs. If your dataset only contains images of ring-tailed lemurs, the model might struggle to accurately identify aye-aye lemurs in the wild. This limitation could lead to errors in population assessments. The great thing is that if you are mindful of any imbalances in the underlying dataset, you can potentially use synthetic data to de-bias the real data by augmenting with synthetic data from the under-represented class.
#
#   Try to make sure your dataset reflects the variety found in the real world!
#
# **Nice Variety**
# ![nice-variety](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-overfit/good-variety.jpg)
#
# **Copying Existing Biases:**
#
# If the data you used to create the synthetic images already had biases, your model might unintentionally learn and replicate those biases. It's like copying a friend's notes without realizing they made a mistake – your computer might end up with the same errors.
#
# ## Does the benefit of using synthetic data outweigh the computational cost?
#
# Generating high-quality synthetic data can be computationally expensive. This may pose challenges in terms of both time and resources, especially for complex models or large datasets. As a general rule, generating and using a synthetic dataset only makes sense if it ultimately saves resources (money, time, etc.).
#
# ### What is the perceived quality of the synthetic images?
#
# Let's consider the lung images we generated using DCGAN. While some of the images looked pretty realistic, others were not so good. A model trained with the low-quality images might fail to detect pneumonia because they contained noise that isn't present in the real images. It is also possible that your model might get really good at recognizing patterns in the synthetic data, but those patterns might not exist or may be different in the real world.
#
# A good practice is to evaluate your dataset using a metric such as Frechet Inception Distance (FID), Inception Score (IS), or the Classification Accuracy Score (CAS).
#
# _FID:_
#
# FID uses a pre-trained neural network model, often [Inception](https://huggingface.co/docs/timm/models/inception-v4), which is good at recognizing objects in images. The model is used to extract features from both the real and generated images.FID is a measure of how "far" one distribution is from another, taking into account both the mean and covariance of the distributions.
#
# A low FID suggests that the feature distributions of real and generated images are similar and the generated images are more likely
# to be realistic.
#
# _IS:_
#
# IS uses a pre-trained Inception model to evaluate the quality of generated images produced by generative models, particularly GANs.
# For each generated image, the Inception model assigns a score based on its confidence in recognizing objects within that image. High scores are better and indicate that the Inception model is confident about the content of the image.
#
# _CAS:_
#
# Classification accuracy is another measure of how well your model is performing on the synthetic data. A higher accuracy indicates that the model is effectively capturing the features and patterns of the real images. Low accuracy scores for certain classes may indicate issues with the generation process, such as unrealistic backgrounds, incorrect textures, or inconsistent lighting conditions. You can use CIS to help you identify and address these problems to improve the overall quality of the synthetic dataset.
#
# ## Conclusion
#
# Even after training your model, it is crucial to continuously monitor its performance in real-world scenarios. If your model encounters new situations or trends that weren't present in the synthetic data, it might struggle to adapt. Addressing these challenges involves mindful design of the synthetic data generation process and evaluation of the model's performance on real data. Applying these principles will help to unlock the potential of synthetic data!
#
# ## Resources and Further Reading
#
# - [Analyzing Effects of Fake Training Data on the Performance of Deep Learning Systems](https://arxiv.org/pdf/2303.01268.pdf)
# - [Bridging the Gap: Enhancing the Utility of Synthetic Data Via Post-Processing Techniques](https://arxiv.org/pdf/2305.10118.pdf)
# - [CIFAKE: Image Classification and Explanable Identification of AI-Generated Synthetic Images](https://arxiv.org/pdf/2303.14126.pdf)
# - [Classification Accuracy Score for Conditional Generative Models](https://arxiv.org/abs/1905.10887)
# - [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
# - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
# - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567v3.pdf)
# - [Metrics](https://github.com/huggingface/community-events/tree/main/huggan/pytorch/metrics)
# - [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
#
# # Introduction
#
# ## What is a point cloud?
#
# A point cloud is a collection of individual points, each representing a sample of a surface within a three-dimensional space denoted by [x, y, z] coordinates. Beyond their spatial coordinates, these points often carry additional attributes like normals, RGB color, albedo, and Bidirectional Reflectance Distribution Function (BRDF).
#
# Here, albedo is the measure of how much light a surface reflects. It's essentially the ratio of reflected light to the incident light that strikes the surface. In simpler terms, it describes how much of the incoming light is bounced back. A high albedo indicates a surface that reflects a lot of light, such as snow, while a low albedo suggests a surface that absorbs more light, like asphalt.
#
# The BRDF is a function that describes how light is scattered or reflected at an opaque surface. It details the way light is reflected at an intersection point on a surface, considering the incoming light direction and the outgoing direction. It provides a mathematical description of the surface's reflective properties, including factors like glossiness, roughness, and the distribution of reflected light over different angles.
# These attributes serve crucial roles in various applications such as modeling, rendering, and scene comprehension.
#
# While the concept of point cloud data isn't new and has been integral in fields like graphics and physics simulation for many years, its significance has notably surged due to two key trends.
# Firstly, the widespread availability of cost-effective and user-friendly point cloud acquisition devices has significantly increased accessibility.
#
# Augmented Reality and autonomous vehicles have further underscored their relevance in today's technological landscape.
#
# ![An Example of Point Clouds in Action](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/point_cloud_example.jpeg)
#
# **Now that we know what a Point Cloud is, what can we do with them?**
#
# The 3D Point Data is mainly used in self-driving capabilities, but now other AI models using computer vision like drones and robots are also using LiDAR for better visual perception. LiDAR is a remote sensing process that collects measurements used to create 3D models and maps of objects and environments. Using ultraviolet, visible, or near-infrared light, LiDAR gauges spatial relationships and shapes by measuring the time it takes for signals to bounce off objects and return to the scanner.
#
# ## Generation and Data Representation
#
# We will be using the python library [point-cloud-utils](https://github.com/fwilliams/point-cloud-utils), and [open-3d](https://github.com/isl-org/Open3D), which can be installed by:
#
# ```bash
# pip install point-cloud-utils
# ```
#
# We will be also using the python library open-3d, which can be installed by:
#
# ```bash
# pip install open3d
# ```
#
# OR a Smaller CPU only version:
#
# ```bash
# pip install open3d-cpu
# ```
#
# Now, first we need to understand the formats in which these point clouds are stored in, and for that, we need to look at mesh cloud.
#
# **Why?**
#
# - `point-cloud-utils` supports reading common mesh formats (PLY, STL, OFF, OBJ, 3DS, VRML 2.0, X3D, COLLADA).
# - If it can be imported into [MeshLab](https://github.com/cnr-isti-vclab/meshlab), we can read it! (from their readme)
#
# The type of file is inferred from its file extension. Some of the extensions supported are:
#
# ** PLY (Polygon File Format) **
#
# - A simple PLY object consists of a collection of elements for representation of the object. It consists of a list of (x,y,z) triplets of a vertex and a list of faces that are actually indices into the list of vertices.
# - Vertices and faces are two examples of elements and the majority of the PLY file consists of these two elements.
# - New properties can also be created and attached to the elements of an object, but these should be added in such a way that old programs do not break when these new properties are encountered.
#
# ** STL (Standard Tessellation Language) **
#
# - This format approximates the surfaces of a solid model with triangles.
# - These triangles are also known as facets, where each facet is described by a perpendicular direction and three points representing the vertices of the triangle.
# - However, these files have no description of Color and Texture.
#
# ** OFF (Object File Format) **
#
# - Object File Format (.OFF) files are used to represent the geometry of a model by specifying the polygons of the model's surface. The polygons can have any number of vertices.
# - It supports ASCII text versions of objects for the purpose of interchange, and binary versions for efficiency of reading and writing
# - It is also refered to as the .obj format.
#
# ** 3DS (3D Studio) **
#
# - A file with .3ds extension represents the 3D Studio mesh file format used by Autodesk 3D Studio.
# - The 3DS format utilizes a binary file structure, enabling faster and smaller file sizes compared to text-based formats, with data organized into chunks within the file.
# - These Chunks store the shapes, lighting, and viewing information that together represent the three-dimensional scene.
#
# ** X3D (Extensible 3D Graphics) **
#
# - X3D is an XML based 3D graphics file format for presentation of 3D information. It is a modular standard and is defined through several ISO specifications.
# - The format supports vector and raster graphics, transparency, lighting effects, and animation settings including rotations, fades, and swings.
# - X3D has the advantage of encoding color information (unlike STL) that is used during printing the model on a color 3D printer.
#
# ** DAE (Digital Asset Exchange) **
#
# - This is an XML schema which is an open standard XML schema, from which DAE files are built.
# - This file format is based on the COLLADA (COLLAborative Design Activity) XML schema which is an open standard XML schema for the exchange of digital assets among graphics software applications.
# - The format's biggest selling point is its compatibility across multiple platforms.
# - COLLADA files aren't restricted to one program or manufacturer. Instead, they offer a standard way to store 3D assets.
#
