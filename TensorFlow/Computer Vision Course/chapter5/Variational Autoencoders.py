"""
================================================================================
 CV Course · Chapter 5 · 生成模型：GAN / AE / VAE（学习笔记描述）
================================================================================
 一句话：图像生成三条经典路线——GAN(对抗)、自编码器(重构)、VAE(带概率的编解码)。
 本章讲：
   ① GAN：生成器 vs 判别器对抗训练。
   ② 自编码器：编码到低维潜表示再解码重构(降维/去噪)。
   ③ VAE：潜空间加概率约束，可采样生成新图；GAN vs VAE 优缺点对比。
 要点：VAE 生成稳但偏糊、GAN 清晰但难训；扩散模型后来居上(见 Diffusion Course)。
 说明：超长笔记(概念为主)，需 torch；参考为主。
================================================================================
"""

# # #变分自编码器
# # 生成对抗网络
# # 介绍
# # 生成对抗网络（GAN）是一类深度学习模型，由Ian Goodfellow及其同事于2014年提出。GAN的核心思想是训练一个生成器网络来生成与真实数据无法区分的数据，同时训练一个判别器网络来区分真实数据和生成数据。
# #
# # 架构概述： GAN 由两个主要组件构成：the generator和the discriminator。
# # 生成器：生成器接收随机噪声
# # z
# # z它以合成数据样本作为输入，并生成合成数据样本。其目标是创建足够逼真的数据，以欺骗判别器。
# # 判别器：判别器类似于侦探，负责评估给定样本是真实的（来自实际数据集）还是伪造的（由生成器生成）。其目标是不断提高区分真实样本和生成样本的准确率。
# # 自编码器简介
# # 自编码器是一类主要用于无监督学习和降维的神经网络。自编码器的基本思想是将输入数据编码成低维表示，然后再解码回原始数据，旨在最小化重构误差。自编码器的基本架构由两个主要组件构成——the encoder和the decoder。
# #
# # 编码器：编码器负责将输入数据转换为压缩的或潜在的表示。它通常由一层或多层神经元组成，这些神经元逐步降低输入数据的维度。
# # 解码器：解码器则接收编码器生成的压缩表示，并尝试重构原始输入数据。与编码器类似，它通常由一层或多层组成，但顺序相反，维度逐渐增加。
# # 生成对抗网络（GAN）与变分自编码器（VAE）
# # 生成对抗网络（GAN）和变分自编码器（VAE）都是机器学习中常用的生成模型，但它们各有优缺点。哪个“更好”取决于具体的任务和需求。以下是它们优缺点的详细分析。
# #
# # 图像生成：
# # 生成对抗网络（GAN）：
# # 优势：生成更高质量的图像，尤其适用于具有清晰细节和逼真纹理的复杂数据。
# # 缺点：训练难度较大，且容易出现不稳定现象。
# # 例如： GAN 生成的卧室图像可能与真实的卧室图像无法区分，而 VAE 生成的卧室图像可能会显得模糊或光照不真实。
# # VAE：
# # 优势：比 GAN 更容易训练，也更稳定。
# # 缺点：可能会生成模糊、细节较少且特征不真实的图像。
# # 其他任务：
# # 生成对抗网络（GAN）：
# # 优势：可用于超分辨率和图像到图像转换等任务。
# # 缺点：对于需要在数据点之间平滑过渡的任务，可能不是最佳选择。
# # VAE：
# # 优势：广泛用于图像去噪和异常检测等任务。
# # 缺点：对于需要高质量图像生成的任务，可能不如 GAN 有效。
# # 以下表格总结了主要区别：
# #
# # 特征	生成对抗网络	VAE
# # 图像质量	更高	降低
# # 培训的便捷性	更难	更轻松
# # 稳定	稳定性较差	更稳定
# # 应用程序	图像生成、超分辨率、图像到图像的转换	图像去噪、异常检测、信号分析
# # 训练
# # GAN
# # 训练
# # GAN
# # 涉及一个独特的对抗过程，其中生成器和判别器进行猫捉老鼠的游戏。
# #
# # 对抗训练过程：生成器和判别器同时进行训练。生成器的目标是生成与真实数据无法区分的数据，而判别器则致力于提高其区分真实样本和伪造样本的能力。
# # 目标函数：训练过程采用最小 - 最大博弈类型的目标函数，用于优化生成器和判别器。生成器的目标是最小化判别器将生成的样本正确分类为假样本的概率，而判别器的目标是最大化该概率。该目标函数表示为：
# # 最小
# # ⁡
# # G
# # 最大限度
# # ⁡
# # D
# # L
# # （
# # D
# # ，
# # G
# # ）
# # =
# # E
# # x
# # ∼
# # p
# # r
# # （
# # x
# # ）
# # [
# #     日志
# # ⁡
# # D
# # （
# # x
# # ）
# # ]
# # +
# # E
# # x
# # ∼
# # p
# # 克
# # （
# # x
# # ）
# # [
# #     日志
# # ⁡
# # （
# # 1
# # −
# # D
# # （
# # x
# # ）
# # ）
# # ]
# # G
# # 最小
# # ​
# #
# # D
# # 最大限度
# # ​
# # L （D ，G ）=E
# # x ∼ p
# # r
# # ​
# # （x ）
# # ​
# # [日志​D(x)]+E
# # x ∼ p
# # 克
# # ​
# # （x ）
# # ​
# # [log(1​−D(x))]
# # 在这里，判别器试图最大化这个损失函数，而生成器试图最小化它，因此具有对抗性。
# # 迭代改进：随着训练的进行，生成器越来越擅长生成逼真的样本，判别器也变得越来越具有鉴别力。这种对抗循环持续进行，直到生成器生成的数据与真实数据几乎无法区分为止。
# # Introduction to Diffusion Models
#
# What you will learn from this chapter:
#
# - What are diffusion models and how do they differ from GANs
# - Major sub categories of diffusion models
# - Use cases of diffusion models
# - Drawback in diffusion models
#
# ## Diffusion Models and their Difference from GANs
#
# Diffusion models are a new and exciting area in computer vision that has shown impressive results in creating images. These generative models work on two stages, a forward diffusion stage and a reverse diffusion stage: first, they slightly change the input data by adding some noise, and then they try to undo these changes to get back to the original data. This process of making changes and then undoing them helps generate realistic images.
#
# These generative models raised the bar to a new level in the area of generative modeling, particularly referring to models such as [Imagen](https://imagen.research.google/) and [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)(LDMs). For instance consider the below images generated via such models.
#
# ![Example images generated using diffusion models](https://huggingface.co/datasets/hwaseem04/Documentation-files/resolve/main/CV-Course/diffusion-eg.png)
#
# GANs were considered by many as state-of-the-art generative models in terms of the quality of the generated samples, before the recent rise of diffusion models. GANs are also known as being difficult to train due to their adversarial objective, and often suffer from mode collapse. Think of different modes as different categories. Consider cat and dog as two seperate mode. If the task of the generator is to produce cat and dog images, if mode collapse occurs it means the generator just produces plausible images of either cat or dog alone. One reason for this to happen is the failure of the discriminator to move from local minima ending up repeatatively classifying only one of the mode(either cat or dog) as fake. In contrast, diffusion models have a stable training process and provide more diversity because they are likelihood-based.
#
# However, diffusion models tend to be computationally intensive and require longer inference times compared to GANs due to the step-by-step reverse process.
#
# In Science, Diffusion is a process where solute particles move from higher concentrated region of solute to lower concentrated region of solute in a solvent. Consider the below diffusion analogy for high-level intuition:
#
# ![Diffusion analogy-drop of ink in water](https://huggingface.co/datasets/hwaseem04/Documentation-files/resolve/main/CV-Course/diffusion-intuition.jpeg)
#
# Above is the traditional diffusion process where the drop of ink completely merges after some time when dropped in a clean water glass. Practically reversing this is impossible, i.e., getting the drop from the mixture. But this is what is done in diffusion models, i.e., removing noise and hence producing a clean image.
#
# In diffusion models, Gaussian noise is added step-by-step to the training images to turn them completely into junk noisy images. Through this process, the model learns to remove the noise step-by-step, hence it is capable of turning any Gaussian noisy image into a new diverse image (can also be conditioned based on text prompts).
#
# ![Reverse-process](https://huggingface.co/datasets/hwaseem04/Documentation-files/resolve/main/CV-Course/diffusion-process.jpg)
#
# ## Major Variants of Diffusion models
#
# There are 3 major diffusion modelling frameworks:
# - Denoising diffusion probabilistic models (DDPMs):
# 	- DDPMs are models that employ latent variables to estimate the probability distribution. From this point of view, DDPMs can be viewed as a special kind of variational auto-encoders (VAEs), where the forward diffusion stage corresponds to the encoding process inside VAE, while the reverse diffusion stage corresponds to the decoding process.
# - Noise conditioned score networks (NCSNs):
# 	- It is based on training a shared neural network via score matching to estimate the score function (defined as the gradient of the log density) of the perturbed data distribution at different noise levels.
# - Stochastic differential equations (SDEs):
# 	- It represents an alternative way to model diffusion, forming the third subcategory of diffusion models. Modeling diffusion via forward and reverse SDEs leads to efficient generation strategies as well as strong theoretical results. This can be viewed as a generalization over DDPMs and NCSNs.
#
# ![Sub categories of diffusion](https://huggingface.co/datasets/hwaseem04/Documentation-files/resolve/main/CV-Course/diffusion-sub-categories.png)
#
# ## Use Cases of Diffusion Models
#
# Diffusion is used in a variety of tasks including, but not limited to:
# - Image generation - Generating images based on prompts.
# - Image super-resolution - Increasing resolution of images.
# - Image inpainting - Filling up a degraded portion of an image based on prompts.
# - Image editing - Editing specific/entire part of the image without losing its visual identity.
# - Image-to-image translation - This includes changing background, attributes of the location etc.
# - Learned Latent representation from diffusion models can also be used for.
#     - Image segmentation
#     - Classification
#     - Anomaly detection
#
# Want to play with diffusion models? No worries, Hugging Face's [Diffusers](https://huggingface.co/docs/diffusers/index) library comes to rescue. You can use almost all recent diffusion SOTA models for almost any task.
#
# ## Drawbacks in Diffusion Models
#
# The most significant disadvantage of diffusion models remains the need to perform multiple steps at inference time to generate only one sample. [Latent consistency models](https://latent-consistency-models.github.io/)(LCMs) is one direction of research proposed to overcome the slow iterative sampling process of Latent Diffusion models (LDMs), enabling fast inference with minimal steps on any pre-trained LDMs (e.g Stable Diffusion). Despite the important amount of research conducted in this direction, GANs are still faster at producing images.
#
# Other issues of diffusion models can be linked to the commonly used strategy to employ CLIP embeddings for text-to-image generation. Few literature studies highlight that their model struggles to render readable text in an image and motivates the behavior by stating that CLIP embeddings do not contain information about spelling.
#
# # CycleGAN Introduction
#
# This
# section
# introduces
# CycleGAN, short
# for *Cycle - Consistent Generative Adversarial Network *, which is a framework designed for image-to-image translation tasks where paired examples are not available.Introduced by Zhu et al.in a[2017](https://
#     arxiv.org / abs / 1703.10593) paper, it
# represents
# a
# significant
# advancement in the
# field
# of
# computer
# vision and machine
# learning.
#
# In
# many
# image - to - image
# translation
# tasks, the
# goal is to
# learn
# a
# mapping
# between
# an
# input
# image and an
# output
# image.Traditional
# approaches
# often
# rely
# on
# large
# datasets
# of
# paired
# examples(e.g., photos and corresponding
# sketches).However, obtaining
# such
# paired
# datasets
# can
# be
# extremely
# challenging or even
# infeasible in many
# scenarios.This is where
# CycleGAN
# comes
# into
# play, as it is designed
# to
# work
# with unpaired datasets.
#
# ## What Is Unpaired Image-to-Image Translation
# ![paired and unpaired_images](
#     https: // huggingface.co / datasets / hf - vision / course - assets / resolve / main / unpaired_images.png)
#
# In
# many
# image
# translation
# scenarios, we
# encounter
# datasets
# lacking
# direct, one - to - one
# correspondence
# between
# image
# pairs.This
# scenario is where
# unpaired
# image - to - image
# translation
# comes
# into
# play.Here, instead
# of
# having
# matching
# pairs
# of
# images, you
# work
# with two distinct sets or "piles" of images, each representing a different style or domain, such as X and Y.For instance, one pile might consist of realistic photographs, while the other could contain artworks by Monet, Cezanne, or other artists.Alternatively, the piles could represent different seasons, with one showcasing winter landscapes and the other summer scenes.Another example could be a collection of horse images in one pile and zebra images in the other, without any direct pairing between the two.The objective in unpaired image-to-image translation is for the model to learn and extract the general stylistic elements from each pile and apply these learned styles to transform images from one domain to another.This transformation often works both ways, allowing images from domain X to be translated into the style of domain Y, and vice versa.This approach is particularly valuable when exact image pairs are unavailable or difficult to obtain.
#
# ## Main Components of CycleGAN
#
# ** Dual
# GAN
# Structure **:
#
# CycleGAN
# employs
# two
# GANs(Generative
# Adversarial
# Networks), one
# for translating from the first set to the second (e.g., zebra to horse) and another for the reverse process (horse to zebra).This dual structure ensures realism (via the adversarial process) and content preservation (via cycle consistency).
# - PatchGAN
# Discriminators: The
# discriminators
# used in CycleGAN
# are
# based
# on[PatchGAN](https: // arxiv.org / pdf / 1611.07004.pdf) architecture, which
# assesses
# patches
# of
# an
# image
# rather
# than
# the
# whole, contributing
# to
# more
# detailed and localized
# realism.
# - Generator
# Architecture: The
# generators in CycleGAN
# draw
# from
#
# [U - Net](https: // arxiv.org / abs / 1505.04597) and [DCGAN](https: // arxiv.org / abs / 1511.06434
# architectures, involving
# downsampling(encoding), upsampling(decoding), and convolutional
# layers
# with batch normalization and ReLU.
# These generators are enhanced with additional convolutional layers and skip connections, known as residual connections, which aid in learning identity functions and supporting deeper transformations.
#
# ** Cycle Consistency Loss **:
#
#     Ensures
# that
# the
# style
# of
# an
# image
# can
# be
# changed(e.g., sad
# face
# to
# hugging
# face) and then
# reverted
# back
# to
# its
# original
# form(hugging
# face
# back
# to
# sad
# face) with minimal loss of detail or content.
#
# Implementation Involves two stages of transformation.
#
# ** First Stage **: A
# sad
# face is transformed
# into
# a
# hugging
# face. ** Second
# Stage **: This
# hugging
# face is then
# converted
# back
# into
# a
# sad
# face.The
# model
# aims
# for the final image (reverted sad face) to closely resemble the original sad face.This is achieved by minimizing the pixel difference between these two images, which is added to the model's loss function. The cycle consistency is applied in both directions: sad face to hugging face to sad face, and hugging face to sad face to hugging face. The loss for each cycle is calculated by summing the pixel differences and is then incorporated into the overall generator loss.
# Integration with Adversarial Loss: Cycle
# consistency
# loss is combined
# with adversarial loss, commonly used in GANs, to form a comprehensive loss function.This combined loss function is optimized simultaneously for both generators in CycleGAN.
#
# ** Least-Square Loss **:
#
#     It
# 's a method that minimizes the sum of squared residuals. What that means is that it tries to find the best-fit line that has the smallest sum of squared distances between that line and all the points. In GANs, the **line** represents the label (real or fake), and the points represent the discriminator'
# s
# predictions.The
# discriminator
# 's loss function in Least Squares adversarial loss is formulated by calculating the squared distance between its predictions and the actual labels (real or fake) across multiple images. For the generator, the loss function is designed to make its fake outputs appear as real as possible, measured by how far these outputs deviate from the label of **real**. Having Least-Square Loss addresses vanishing gradients and mode collapse issues common in BCE loss.
#
# ** Indentity
# Loss **:
#
# Introduced in CycleGAN, identity
# loss is an
# optional
# loss
# term
# aimed
# at
# enhancing
# color
# preservation in generated
# images.It
# works
# by
# ensuring
# that
# an
# image
# input
# into
# a
# generator(e.g., a
# horse
# image
# into
# a
# zebra - to - horse
# generator) should
# ideally
# output
# the
# same
# image, as it is already in the
# target
# style.The
# loss is calculated as the
# pixel
# distance
# between
# the
# real
# input and the
# output
# of
# the
# generator.A
# zero
# pixel
# distance(no
# change in the
# image) results in zero
# identity
# loss, which is the
# desired
# outcome.
# For
# CycleGANs
# generators, this
# loss is added
# alongside
# adversarial and cycle
# consistency
# losses.It
# 's adjusted using a lambda term (weighting factor).
#
# ![CycleGAN](https: // huggingface.co / datasets / hf - vision / course - assets / resolve / main / cycleGAN1.jpg)
# This
# figure
# shows
# the
# combined
# GAN
# architecture
# functionality
# for both ** GANs **.These GANs are linked by cycle consistency, forming a cycle.
# For real images, the classification matrix contains ones.For fake images, it contains zeros.In summary, CycleGAN intricately combines two GANs with various loss functions, including adversarial, cycle consistency, and optional identity loss,
# to effectively transfer styles between two domains
# while preserving the essential characteristics of the input images.
#
# # CycleGAN Introduction
#
# This
# section
# introduces
# CycleGAN, short
# for *Cycle - Consistent Generative Adversarial Network *, which is a framework designed for image-to-image translation tasks where paired examples are not available.Introduced by Zhu et al.in a[2017](https://
#     arxiv.org / abs / 1703.10593) paper, it
# represents
# a
# significant
# advancement in the
# field
# of
# computer
# vision and machine
# learning.
#
# In
# many
# image - to - image
# translation
# tasks, the
# goal is to
# learn
# a
# mapping
# between
# an
# input
# image and an
# output
# image.Traditional
# approaches
# often
# rely
# on
# large
# datasets
# of
# paired
# examples(e.g., photos and corresponding
# sketches).However, obtaining
# such
# paired
# datasets
# can
# be
# extremely
# challenging or even
# infeasible in many
# scenarios.This is where
# CycleGAN
# comes
# into
# play, as it is designed
# to
# work
# with unpaired datasets.
#
# ## What Is Unpaired Image-to-Image Translation
# ![paired and unpaired_images](
#     https: // huggingface.co / datasets / hf - vision / course - assets / resolve / main / unpaired_images.png)
#
# In
# many
# image
# translation
# scenarios, we
# encounter
# datasets
# lacking
# direct, one - to - one
# correspondence
# between
# image
# pairs.This
# scenario is where
# unpaired
# image - to - image
# translation
# comes
# into
# play.Here, instead
# of
# having
# matching
# pairs
# of
# images, you
# work
# with two distinct sets or "piles" of images, each representing a different style or domain, such as X and Y.For instance, one pile might consist of realistic photographs, while the other could contain artworks by Monet, Cezanne, or other artists.Alternatively, the piles could represent different seasons, with one showcasing winter landscapes and the other summer scenes.Another example could be a collection of horse images in one pile and zebra images in the other, without any direct pairing between the two.The objective in unpaired image-to-image translation is for the model to learn and extract the general stylistic elements from each pile and apply these learned styles to transform images from one domain to another.This transformation often works both ways, allowing images from domain X to be translated into the style of domain Y, and vice versa.This approach is particularly valuable when exact image pairs are unavailable or difficult to obtain.
#
# ## Main Components of CycleGAN
#
# ** Dual
# GAN
# Structure **:
#
# CycleGAN
# employs
# two
# GANs(Generative
# Adversarial
# Networks), one
# for translating from the first set to the second (e.g., zebra to horse) and another for the reverse process (horse to zebra).This dual structure ensures realism (via the adversarial process) and content preservation (via cycle consistency).
# - PatchGAN
# Discriminators: The
# discriminators
# used in CycleGAN
# are
# based
# on[PatchGAN](https: // arxiv.org / pdf / 1611.07004.pdf) architecture, which
# assesses
# patches
# of
# an
# image
# rather
# than
# the
# whole, contributing
# to
# more
# detailed and localized
# realism.
# - Generator
# Architecture: The
# generators in CycleGAN
# draw
# from
#
# [U - Net](https: // arxiv.org / abs / 1505.04597) and [DCGAN](https: // arxiv.org / abs / 1511.06434
# architectures, involving
# downsampling(encoding), upsampling(decoding), and convolutional
# layers
# with batch normalization and ReLU.
# These generators are enhanced with additional convolutional layers and skip connections, known as residual connections, which aid in learning identity functions and supporting deeper transformations.
#
# ** Cycle Consistency Loss **:
#
#     Ensures
# that
# the
# style
# of
# an
# image
# can
# be
# changed(e.g., sad
# face
# to
# hugging
# face) and then
# reverted
# back
# to
# its
# original
# form(hugging
# face
# back
# to
# sad
# face) with minimal loss of detail or content.
#
# Implementation Involves two stages of transformation.
#
# ** First Stage **: A
# sad
# face is transformed
# into
# a
# hugging
# face. ** Second
# Stage **: This
# hugging
# face is then
# converted
# back
# into
# a
# sad
# face.The
# model
# aims
# for the final image (reverted sad face) to closely resemble the original sad face.This is achieved by minimizing the pixel difference between these two images, which is added to the model's loss function. The cycle consistency is applied in both directions: sad face to hugging face to sad face, and hugging face to sad face to hugging face. The loss for each cycle is calculated by summing the pixel differences and is then incorporated into the overall generator loss.
# Integration with Adversarial Loss: Cycle
# consistency
# loss is combined
# with adversarial loss, commonly used in GANs, to form a comprehensive loss function.This combined loss function is optimized simultaneously for both generators in CycleGAN.
#
# ** Least-Square Loss **:
#
#     It
# 's a method that minimizes the sum of squared residuals. What that means is that it tries to find the best-fit line that has the smallest sum of squared distances between that line and all the points. In GANs, the **line** represents the label (real or fake), and the points represent the discriminator'
# s
# predictions.The
# discriminator
# 's loss function in Least Squares adversarial loss is formulated by calculating the squared distance between its predictions and the actual labels (real or fake) across multiple images. For the generator, the loss function is designed to make its fake outputs appear as real as possible, measured by how far these outputs deviate from the label of **real**. Having Least-Square Loss addresses vanishing gradients and mode collapse issues common in BCE loss.
#
# ** Indentity
# Loss **:
#
# Introduced in CycleGAN, identity
# loss is an
# optional
# loss
# term
# aimed
# at
# enhancing
# color
# preservation in generated
# images.It
# works
# by
# ensuring
# that
# an
# image
# input
# into
# a
# generator(e.g., a
# horse
# image
# into
# a
# zebra - to - horse
# generator) should
# ideally
# output
# the
# same
# image, as it is already in the
# target
# style.The
# loss is calculated as the
# pixel
# distance
# between
# the
# real
# input and the
# output
# of
# the
# generator.A
# zero
# pixel
# distance(no
# change in the
# image) results in zero
# identity
# loss, which is the
# desired
# outcome.
# For
# CycleGANs
# generators, this
# loss is added
# alongside
# adversarial and cycle
# consistency
# losses.It
# 's adjusted using a lambda term (weighting factor).
#
# ![CycleGAN](https: // huggingface.co / datasets / hf - vision / course - assets / resolve / main / cycleGAN1.jpg)
# This
# figure
# shows
# the
# combined
# GAN
# architecture
# functionality
# for both ** GANs **.These GANs are linked by cycle consistency, forming a cycle.
# For real images, the classification matrix contains ones.For fake images, it contains zeros.In summary, CycleGAN intricately combines two GANs with various loss functions, including adversarial, cycle consistency, and optional identity loss,
# to effectively transfer styles between two domains
# while preserving the essential characteristics of the input images.
#
# # Introduction to Stable Diffusion
# This
# chapter
# introduces
# the
# building
# blocks
# of
# Stable
# Diffusion
# which is a
# generative
# artificial
# intelligence(generative
# AI) model
# that
# produces
# unique
# photorealistic
# images
# from text and image
# prompts.It
# originally
# launched in 2022 and was
# made
# possible
# thanks
# to
# a
# collaboration
# with
#     [Stability AI](https: // stability.ai /), [RunwayML](https: // runwayml.com /) and CompVis
#     Group
#     at
#     LMU
#     Munich
#     following
#     the[paper](https: // arxiv.org / pdf / 2112.10752.pdf).
#
#     What
#     will
#     you
#     learn
#     from this chapter?
#     - Fundamental
#     components
#     of
#     Stable
#     Diffusion
#     - How
#     to
#     use
#     `text - to - image`, `image2image`, inpainting
#     pipelines
#
#     ## What Do We Need for Stable Diffusion to Work?
#     To
#     make
#     this
#     section
#     interesting
#     we
#     will
#     try to answer some questions to understand the basic components of the Stable Diffusion process.
#     We
#     will
#     briefly
#     discuss
#     each
#     component as they
#     are
#     already
#     covered in our
#     Diffusers
#     course.Also, you
#     can
#     visit
#     our
#     previous
#     section, which
#     talks
#     about
#     GANS and Diffusion
#     models in details.
#
#     - What
#     strategies
#     does
#     Stable
#     Diffusion
#     employ
#     to
#     learn
#     new
#     information?
#     - It
#     uses
#     forward and reverse
#     processes
#     of
#     diffusion
#     models.In
#     the
#     forward
#     process, we
#     add
#     Gaussian
#     noise
#     to
#     an
#     image
#     until
#     all
#     that
#     remains is the
#     random
#     noise.Usually
#     we
#     cannot
#     identify
#     the
#     final
#     noisy
#     version
#     of
#     the
#     image.
#     - In
#     the
#     reserve
#     process, we
#     have
#     a
#     learned
#     neural
#     network
#     trained
#     to
#     gradually
#     denoise
#     an
#     image
#     starting
#     from pure noise, until
#
#     you
#     end
#     up
#     with an actual image.
#
#     Both
#     of
#     these
#     processes
#     happens
#     for a finite number of steps `T`( as per DDPM paper T=1000).You begin the process at time \\(t_0\\) by sampling a real image from your data distribution, and the forward process samples some noise from a Gaussian distribution at each time step t, which is added to the image of the previous time step.To get more mathematical intuition, please read[Hugging Face Blog](https://
#         huggingface.co / blog / annotated - diffusion) on
#     Diffusion
#     Models.
#
#     - Since
#     our
#     images
#     can
#     be
#     huge
#     how
#     can
#     we
#     compress
#     it?
#
#     When
#     you
#     have
#     large
#     images, they
#     require
#     more
#     computing
#     power
#     to
#     process.This
#     becomes
#     very
#     noticeable in a
#     specific
#     operation
#     known as self - attention.The
#     bigger
#     the
#     image, the
#     more
#     calculations
#     are
#     needed, and these
#     calculations
#     increase
#     very
#     quickly( in a
#     way
#     mathematicians
#     call
#     "quadratically") with the size of the image.
#     For
#     example,
#     if you have an image that's 128 pixels wide and tall, it has four times more pixels than an image that's only 64 pixels wide and tall.Because of how self-attention works, dealing with this larger image doesn't just need four times more memory and computing power, it actually needs sixteen times more (since 4 times 4 equals 16). This makes it challenging to work with very high-resolution images, as they require a lot of resources to process.
#     Latent
#     diffusion
#     models
#     address
#     the
#     high
#     computational
#     demands
#     of
#     processing
#     large
#     images
#     by
#     using
#     a
#     Variational
#     Auto - Encoder(VAE)
#     to
#     shrink
#     the
#     images
#     into
#     a
#     more
#     manageable
#     size.The
#     idea is that
#     many
#     images
#     have
#     repetitive or unnecessary
#     information.A
#     VAE, after
#     being
#     trained
#     on
#     a
#     lot
#     of
#     data, can
#     compress
#     an
#     image
#     into
#     a
#     much
#     smaller, condensed
#     form.This
#     smaller
#     version
#     still
#     retains
#     the
#     essential
#     features
#     of
#     the
#     original
#     image.
#
#     - How
#     are
#     we
#     fusing
#     texts
#     with images since we are using prompts?
#
#     We
#     know
#     that
#     during
#     inference
#     time, we
#     can
#     feed in the
#     description
#     of
#     an
#     image
#     we
#     'd like to see and some pure noise as a starting point, and the model does its best to '
#     denoise
#     ' the random input into something that matches the caption.
#     SD
#     leverages
#     a
#     pre - trained
#     transformer
#     model
#     based
#     on
#     something
#     called[CLIP](
#         https: // huggingface.co / learn / computer - vision - course / unit4 / multimodal - models / clip - and -relatives / clip).CLIP
#     's text encoder was designed to process image captions into a form that could be used to compare images and text, so it is well suited to the task of creating useful representations from image descriptions. An input prompt is first tokenized (based on a large vocabulary where each word or sub-word is assigned a specific token) and then fed through the CLIP text encoder, producing a 768-dimensional (in the case of SD 1.X) or 1024-dimensional (SD 2.X) vector for each token. To keep things consistent prompts are always padded/truncated to be 77 tokens long, and so the final representation which we use as conditioning is a tensor of shape 77x1024 per prompt.
#
#     - How
#     can
#     we
#     add - in good
#     inductive
#     biases?
#
#     Since, we
#     are
#     trying
#     to
#     generate
#     something
#     new(e.g., a
#     realistic
#     Pokemon), we
#     need
#     a
#     way
#     to
#     go
#     beyond
#     the
#     images
#     we
#     have
#     seen
#     before(e.g., an
#     anime
#     Pokemon).That
#     's where U-Net and self-attention come into the picture. Given a noisy version of an image, the model is tasked with predicting the denoised version based on additional clues such as a text description of the image. Ok, how do we actually feed this conditioning information into the U-Net for it to use as it makes predictions? The answer is something called cross-attention. Scattered throughout the U-Net are cross-attention layers.
#     Each
#     spatial
#     location in the
#     U - Net
#     can
#     'attend'
#     to
#     different
#     tokens in the
#     text
#     conditioning, bringing in relevant
#     information
#     from the prompt.
#
#     ## How to use `text-to-image`, `image-to-image`, Inpainting Models in Diffusers
#     This
#     section
#     introduces
#     helpful
#     usecases and how
#     we
#     can
#     perform
#     these
#     tasks
#     using
#     the[Diffusers](https: // github.com / huggingface / diffusers) library.
#     - Steps
#     for `text - to - image` inference
#         The
#         idea is to
#         pass in the
#         text
#         prompt, which is converted
#         to
#         the
#         output
#         image.
#
#     Using
#     the
#     `diffusers`
#     library
#     you
#     can
#     get
#     `text - to - image`
#     working in 2
#     steps.
#
#     Let
#     's install the `diffusers` library first.
#     ```bash
#     pip
#     install
#     diffusers
#     ```
#
#     We
#     will
#     now
#     initialize
#     the
#     pipeline and
#     pass
#     our
#     prompt
#     inside and infer.
#     ```python
#     from diffusers import AutoPipelineForText2Image
#     import torch
#
#     pipeline = AutoPipelineForText2Image.from_pretrained(
#         "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
#     ).to("cuda")
#     generator = torch.Generator(device="cuda").manual_seed(31)
#     image = pipeline(
#         "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
#         generator=generator,
#     ).images[0]
#     ```
#
#     - Steps
#     for image - to - image inference
#         In
#         similar
#         fashion, we
#         can
#         initialize
#         the
#         pipeline, but
#         pass
#         an
#         image and a
#         text
#         prompt
#         instead.
#     ```python
#     import torch
#     from diffusers import AutoPipelineForImage2Image
#     from diffusers.utils import load_image, make_image_grid
#
#     pipeline = AutoPipelineForImage2Image.from_pretrained(
#         "kandinsky-community/kandinsky-2-2-decoder",
#         torch_dtype=torch.float16,
#         use_safetensors=True,
#     )
#     pipeline.enable_model_cpu_offload()
#     # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
#     pipeline.enable_xformers_memory_efficient_attention()
#
#     # Load an image to pass to the pipeline:
#     init_image = load_image(
#         "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
#     )
#
#     # Pass a prompt and image to the pipeline to generate an image:
#     prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
#     image = pipeline(prompt, image=init_image).images[0]
#     make_image_grid([init_image, image], rows=1, cols=2)
#     ```
#
#     - Steps
#     for Inpainting
#         For
#         inpainting
#         pipeline, we
#         need
#         to
#         pass
#         an
#         image, a
#         text
#         prompt, and a
#         mask
#         based
#         on
#         an
#         object in that
#         image, which
#         indicates
#         what
#         to
#         inpaint in the
#         image.
#     In
#     this
#     example
#     we
#     also
#     pass
#     a
#     negative
#     prompt
#     to
#     further
#     influence
#     the
#     inference
#     on
#     what
#     we
#     want
#     to
#     avoid.
#     ```python
#     # Load the pipeline
#     import torch
#     from diffusers import AutoPipelineForInpainting
#     from diffusers.utils import load_image, make_image_grid
#
#     pipeline = AutoPipelineForInpainting.from_pretrained(
#         "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
#     )
#     pipeline.enable_model_cpu_offload()
#     # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
#     pipeline.enable_xformers_memory_efficient_attention()
#
#     # Load the base and mask images:
#     init_image = load_image(
#         "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
#     )
#     mask_image = load_image(
#         "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
#     )
#
#     # Create a prompt to inpaint the image with and pass it to the pipeline with the base and mask images:
#     prompt = (
#         "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
#     )
#     negative_prompt = "bad anatomy, deformed, ugly, disfigured"
#     image = pipeline(
#         prompt=prompt,
#         negative_prompt=negative_prompt,
#         image=init_image,
#         mask_image=mask_image,
#     ).images[0]
#     make_image_grid([init_image, mask_image, image], rows=1, cols=3)
#     ```
#
#     ### Further Reading
#     - [Diffusers documentation](https: // huggingface.co / docs / diffusers / using - diffusers / pipeline_overview)
#     - [Diffusers installation](https: // huggingface.co / docs / diffusers / installation)
#
#     # Control over Diffusion Models
#
#     ## Dreambooth
#
#     Although
#     diffusion
#     models and GANs
#     can
#     generate
#     many
#     unique
#     images, they
#     can
#     't always generate what you need exactly. Hence, you have to fine-tune a model, which usually requires a lot of data and computation. However, some techniques can be used to personalize a model with just a few examples.
#
#     One
#     example is Dreambooth
#     by
#     Google
#     Research, a
#     training
#     technique
#     that
#     updates
#     the
#     entire
#     diffusion
#     model
#     by
#     training
#     on
#     just
#     a
#     few
#     images
#     of
#     a
#     subject or style.It
#     works
#     by
#     associating
#     a
#     special
#     word in the
#     prompt
#     with the example images.The details on Dreambooth can be found in the[paper](https://
#         dreambooth.github.io /) and the[Hugging
#     Face
#     Dreambooth
#     training
#     documentation](https: // huggingface.co / docs / diffusers / training / dreambooth).
#
#     Below, you
#     can
#     see
#     the
#     results
#     of
#     Dreambooth
#     being
#     used
#     to
#     train
#     on
#     4
#     images
#     of
#     a
#     dog and some
#     inference
#     examples.
#     ![Dreambooth Dog Example](
#         https: // huggingface.co / datasets / hf - vision / course - assets / resolve / main / teaser_static.jpg)
#     You
#     can
#     recreate
#     the
#     results
#     following
#     the
#     Hugging
#     Face
#     documentation
#     given
#     above.
#
#     From
#     this
#     example, it
#     's clear that the model has learned about that specific dog and can generate new images of that dog in diverse poses and backgrounds. Although computing, data, and time are improvements, others have found more efficient ways to customize a model.
#
#     This is where
#     the
#     current
#     most
#     popular
#     method
#     for this comes in, which is Low Rank Adaptation (LoRA).This method was initially developed to efficiently fine-tune Large Language Models by Microsoft in this[paper](https://
#         arxiv.org / abs / 2106.09685).The
#     main
#     idea is to
#     factorise
#     the
#     weight
#     update
#     matrix
#     into
#     2
#     low - rank
#     matrices, which
#     are
#     optimized
#     during
#     training
#     whilst
#     the
#     rest
#     of
#     the
#     model is frozen.[Hugging
#     face
#     documentation](https: // huggingface.co / docs / peft / conceptual_guides / lora)
#     has
#     a
#     good
#     conceptual
#     guide
#     on
#     how
#     LoRA
#     works.
#
#     Now,
#     if we put those ideas together we can use LoRA to efficiently fine-tune a diffusion model on a few examples using Dreambooth.A tutorial Google Colab notebook on how to do this can be found[here](https://
#         colab.research.google.com / github / huggingface / notebooks / blob / main / diffusers / SDXL_DreamBooth_LoRA_.ipynb).
#
#     Due
#     to
#     the
#     quality and efficiency
#     of
#     this
#     method, many
#     people
#     have
#     created
#     their
#     own
#     LoRA
#     parameters
#     which
#     many
#     can
#     be
#     found
#     on
#     a
#     website
#     called[Civitai](https: // civitai.com / models) and [Hugging Face](
#         https: // huggingface.co / collections / multimodalart / awesome - sdxl - loras - 64
#     f9af6d5cce4f4e8f351466).
#     For
#     Civitai
#     you
#     can
#     download
#     the
#     LoRA
#     weights
#     which
#     usually
#     are in the
#     range
#     of
#     50 - 500
#     MB and in the
#     case
#     of
#     Hugging
#     Face
#     version
#     you
#     can
#     just
#     load
#     the
#     model
#     directly
#     from the model
#
#     hub.
#     Below is an
#     example
#     of
#     how
#     to
#     load
#     the
#     LoRA
#     weights in both
#     cases and then
#     fuse
#     them
#     with the model.
#
#     We
#     can
#     start
#     with installing diffusers library.
#     ```bash
#     pip
#     install
#     diffusers
#     ````
#     We
#     will
#     initialize
#     the
#     `StableDiffusionXLPipeline` and load
#     LoRA
#     adapter
#     weights.
#     ```python
#     from diffusers import StableDiffusionXLPipeline
#     import torch
#
#     model = "stabilityai/stable-diffusion-xl-base-1.0"
#     pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=torch.float16)
#     pipe.load_lora_weights(
#         "lora_weights.safetensors"
#     )  # if you want to install from a weight file
#     pipe.load_lora_weights(
#         "ostris/crayon_style_lora_sdxl"
#     )  # if you wish to install a lora from a repository directly
#     pipe.fuse_lora(lora_scale=0.8)
#     ```
#
#     This
#     makes
#     it
#     quick
#     to
#     load
#     a
#     customised
#     diffusion
#     model and use
#     it
#     for inference, especially since there are a lot of models to choose from .Then, if we want to remove the LoRA weights, we can call `pipe.unfuse_lora()` which will return the model to its original state.As for the `lora_scale` parameter, this is a hyperparameter that controls how much the LoRA weights are used during inference.A value of 1.0 means the LoRA weights are fully used and a value of 0.0 means the LoRA weights are not used at all.The best value is often between 0.7 and 1.0 but it's worth experimenting with different values to see what works best for your use case.
#
#     You
#     can
#     try some of the Hugging Face LoRA models in this Gradio demo:
#
#     ## Guided Diffusion via ControlNet
#
#     Diffusion
#     models
#     have
#     many
#     ways in which
#     they
#     can
#     be
#     guided
#     to
#     create
#     a
#     desired
#     output, such as prompts, negative
#     prompts, guidance
#     scale, inpainting and many
#     others.Here, we
#     will
#     focus
#     on
#     a
#     method
#     that
#     has
#     many
#     variants and can
#     be
#     combined
#     with all the other methods, called ControlNet.It was introduced in this[paper](https://
#         arxiv.org / abs / 2302.05543) by
#     Stanford
#     University.This
#     method
#     allows
#     us
#     to
#     guide
#     the
#     diffusion
#     model
#     with an image that usually holds very specific information such as depth, pose, edges, and many others.This allows for more consistency in the generated images, which is often a problem with diffusion models.
#
#     ControlNet
#     can
#     be
#     used in both
#     text - to - image and image - to - image.Below is a
#     text
#     2
#     image
#     example
#     using
#     a
#     ControlNet
#     which
#     was
#     trained
#     on
#     edge
#     detection
#     conditioning,
#     with the top left image being used as input.
#     Here
#     we
#     can
#     see
#     how
#     all
#     of
#     the
#     generated
#     images
#     have
#     a
#     very
#     similar
#     shape
#     but
#     with different colours.This is because the ControlNet is guiding the diffusion model to create images with the same shape as the input image.
#     ![bird](https: // github.com / lllyasviel / ControlNet / raw / main / github_page / p1.png)
#
#     For
#     code
#     to
#     run
#     ControlNet
#     with Stable Diffusion XL refer to the official documentation[here](https://
#         huggingface.co / docs / diffusers / api / pipelines / controlnet_sdxl  # diffusers.StableDiffusionXLControlNetPipeline) but if you just want to test out some examples take a look at this Gradio demo that lets you try different types of ControlNet:
#
#     # Control over Diffusion Models
#
#     ## Dreambooth
#
#     Although
#     diffusion
#     models and GANs
#     can
#     generate
#     many
#     unique
#     images, they
#     can
#     't always generate what you need exactly. Hence, you have to fine-tune a model, which usually requires a lot of data and computation. However, some techniques can be used to personalize a model with just a few examples.
#
#     One
#     example is Dreambooth
#     by
#     Google
#     Research, a
#     training
#     technique
#     that
#     updates
#     the
#     entire
#     diffusion
#     model
#     by
#     training
#     on
#     just
#     a
#     few
#     images
#     of
#     a
#     subject or style.It
#     works
#     by
#     associating
#     a
#     special
#     word in the
#     prompt
#     with the example images.The details on Dreambooth can be found in the[paper](https://
#         dreambooth.github.io /) and the[Hugging
#     Face
#     Dreambooth
#     training
#     documentation](https: // huggingface.co / docs / diffusers / training / dreambooth).
#
#     Below, you
#     can
#     see
#     the
#     results
#     of
#     Dreambooth
#     being
#     used
#     to
#     train
#     on
#     4
#     images
#     of
#     a
#     dog and some
#     inference
#     examples.
#     ![Dreambooth Dog Example](
#         https: // huggingface.co / datasets / hf - vision / course - assets / resolve / main / teaser_static.jpg)
#     You
#     can
#     recreate
#     the
#     results
#     following
#     the
#     Hugging
#     Face
#     documentation
#     given
#     above.
#
#     From
#     this
#     example, it
#     's clear that the model has learned about that specific dog and can generate new images of that dog in diverse poses and backgrounds. Although computing, data, and time are improvements, others have found more efficient ways to customize a model.
#
#     This is where
#     the
#     current
#     most
#     popular
#     method
#     for this comes in, which is Low Rank Adaptation (LoRA).This method was initially developed to efficiently fine-tune Large Language Models by Microsoft in this[paper](https://
#         arxiv.org / abs / 2106.09685).The
#     main
#     idea is to
#     factorise
#     the
#     weight
#     update
#     matrix
#     into
#     2
#     low - rank
#     matrices, which
#     are
#     optimized
#     during
#     training
#     whilst
#     the
#     rest
#     of
#     the
#     model is frozen.[Hugging
#     face
#     documentation](https: // huggingface.co / docs / peft / conceptual_guides / lora)
#     has
#     a
#     good
#     conceptual
#     guide
#     on
#     how
#     LoRA
#     works.
#
#     Now,
#     if we put those ideas together we can use LoRA to efficiently fine-tune a diffusion model on a few examples using Dreambooth.A tutorial Google Colab notebook on how to do this can be found[here](https://
#         colab.research.google.com / github / huggingface / notebooks / blob / main / diffusers / SDXL_DreamBooth_LoRA_.ipynb).
#
#     Due
#     to
#     the
#     quality and efficiency
#     of
#     this
#     method, many
#     people
#     have
#     created
#     their
#     own
#     LoRA
#     parameters
#     which
#     many
#     can
#     be
#     found
#     on
#     a
#     website
#     called[Civitai](https: // civitai.com / models) and [Hugging Face](
#         https: // huggingface.co / collections / multimodalart / awesome - sdxl - loras - 64
#     f9af6d5cce4f4e8f351466).
#     For
#     Civitai
#     you
#     can
#     download
#     the
#     LoRA
#     weights
#     which
#     usually
#     are in the
#     range
#     of
#     50 - 500
#     MB and in the
#     case
#     of
#     Hugging
#     Face
#     version
#     you
#     can
#     just
#     load
#     the
#     model
#     directly
#     from the model
#
#     hub.
#     Below is an
#     example
#     of
#     how
#     to
#     load
#     the
#     LoRA
#     weights in both
#     cases and then
#     fuse
#     them
#     with the model.
#
#     We can start with installing diffusers library.
#     ```bash
#     pip install diffusers
#     ````
#     We will initialize the `StableDiffusionXLPipeline` and load LoRA adapter weights.
#     ```python
#     from diffusers import StableDiffusionXLPipeline
#     import torch
#
#     model = "stabilityai/stable-diffusion-xl-base-1.0"
#     pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=torch.float16)
#     pipe.load_lora_weights(
#     "lora_weights.safetensors"
#     )  # if you want to install from a weight file
#     pipe.load_lora_weights(
#     "ostris/crayon_style_lora_sdxl"
#     )  # if you wish to install a lora from a repository directly
#     pipe.fuse_lora(lora_scale=0.8)
#     ```
#
#     This makes it quick to load a customised diffusion model and use it for inference, especially since there are a lot of models to choose from .Then, if we want to remove the LoRA weights, we can call `pipe.unfuse_lora()` which will
#     return the
#     model
#     to
#     its
#     original
#     state.As
#     for the `lora_scale` parameter, this is a hyperparameter that controls how much the LoRA weights are used during inference.A value of 1.0 means the LoRA weights are fully used and a value of 0.0 means the LoRA weights are not used at all.The best value is often between 0.7 and 1.0 but it's worth experimenting with different values to see what works best for your use case.
#
#     You
#     can
#     try some of the Hugging Face LoRA models in this Gradio demo:
#
#     ## Guided Diffusion via ControlNet
#
#     Diffusion
#     models
#     have
#     many
#     ways in which
#     they
#     can
#     be
#     guided
#     to
#     create
#     a
#     desired
#     output, such as prompts, negative
#     prompts, guidance
#     scale, inpainting and many
#     others.Here, we
#     will
#     focus
#     on
#     a
#     method
#     that
#     has
#     many
#     variants and can
#     be
#     combined
#     with all the other methods, called ControlNet.It was introduced in this[paper](https://
#         arxiv.org / abs / 2302.05543) by
#     Stanford
#     University.This
#     method
#     allows
#     us
#     to
#     guide
#     the
#     diffusion
#     model
#     with an image that usually holds very specific information such as depth, pose, edges, and many others.This allows for more consistency in the generated images, which is often a problem with diffusion models.
#
#     ControlNet
#     can
#     be
#     used in both
#     text - to - image and image - to - image.Below is a
#     text
#     2
#     image
#     example
#     using
#     a
#     ControlNet
#     which
#     was
#     trained
#     on
#     edge
#     detection
#     conditioning,
#     with the top left image being used as input.
#     Here
#     we
#     can
#     see
#     how
#     all
#     of
#     the
#     generated
#     images
#     have
#     a
#     very
#     similar
#     shape
#     but
#     with different colours.This is because the ControlNet is guiding the diffusion model to create images with the same shape as the input image.
#     ![bird](https: // github.com / lllyasviel / ControlNet / raw / main / github_page / p1.png)
#
#     For
#     code
#     to
#     run
#     ControlNet
#     with Stable Diffusion XL refer to the official documentation[here](https://
#         huggingface.co / docs / diffusers / api / pipelines / controlnet_sdxl  # diffusers.StableDiffusionXLControlNetPipeline) but if you just want to test out some examples take a look at this Gradio demo that lets you try different types of ControlNet:
#
#
#
