"""
================================================================================
 CV Course · Chapter 13 · 视觉中的保持网络(RetNet)（学习笔记描述）
================================================================================
 一句话：RetNet 用"多尺度保持(MSR)"替代注意力，兼顾并行训练 + O(1) 推理 + 长序列。
 本章讲：
   ① RetNet 三种计算范式：并行(训练快)/循环(推理省，O(1))/分块循环(长序列)。
   ② MSR 相对多头注意力的优势(免 KV Cache)。
   ③ 从语言迁移到图像的思路。
 要点：并行训练 + 循环推理"双形态"是 RetNet 的核心卖点。
 说明：超长架构综述(概念为主)，参考为主。
================================================================================
"""

# # Retention In Vision
#
# ## What are Retention Networks
# Retentive Network (RetNet) is a foundational architecture proposed for large language models in the paper [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621). This architecture is designed to address key challenges in the realm of large-scale language modeling: training parallelism, low-cost inference, and good performance.
#
# ![LLM Challenges](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/LLM%20Challenges.png)
# RetNet is able to tackle these challenges by introducing the Multi-Scale Retention (MSR) mechanism, which is an alternative to the multi-head attention mechanism commonly used in Transformer models.
# MSR has a dual form of recurrence and parallelism, so it is possible to train the models in a parallel way while recurrently conducting inference. We will explore RetNet in detail in the later chapter.
#
# The Multi-Scale Retention mechanism operates under three computation paradigms:
# - **Parallel Representation:** This aspect of RetNet is designed similar to self-attention in Transformer, where it enables us train the models with GPUs efficiently.
#
# - **Recurrent Representation:** This representation facilitates efficient inference with O(1) complexity in terms of memory and computational requirements. It significantly reduces deployment costs and latency, and simplifies implementation by eliminating the need for key-value cache strategies often used in traditional models.
#
# - **Chunkwise Recurrent Representation:** This third paradigm addresses the challenge of long-sequence modeling. It achieves this by encoding each local block in parallel for computational speed while simultaneously and recurrently encoding global blocks to optimize GPU memory usage.
#
# During the training phase, the approach incorporates both parallel and chunkwise recurrent representations, optimizing GPU usage for fast computation and being particularly effective for long sequences in terms of computational efficiency and memory use.
# For the inference phase, the recurrent representation is used, favoring autoregressive decoding. This method efficiently reduces memory usage and latency, maintaining equivalent performance outcomes.
#
# ## From Language to Image
# ### RMT
# The paper [RMT: Retentive Networks Meet Vision Transformers](https://arxiv.org/abs/2309.11523) proposes a new vision backbone inspired by the RetNet architecture. The authors propose RMT to enhance the Vision Transformer (ViT) by introducing explicit spatial priors and reducing computational complexity, drawing inspiration from the RetNet's parallel representation.
# This includes adapting the RetNet’s temporal decay to spatial domains and using a [Manhattan distance-based](https://en.wikipedia.org/wiki/Taxicab_geometry) spatial decay matrix, along with an attention decomposition form, to improve efficiency and scalability in vision tasks.
#
# - **Manhattan Self-Attention (MaSA)**
# ![Attention Comparison](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Attention%20Comparison.png)
# MaSA incorporates Self-Attention mechanism with a two-dimensional bidirectional spatial decay matrix based on the Manhattan distance among the tokens. This matrix decreases attention scores for tokens further away from a target token, allowing it to perceive global information while varying attention based on distance.
#
# - **Decomposed Manhattan Self-Attention (MaSAD)**
# ![MaSAD](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MaSAD.png)
# This mechanism decomposes Self-Attention in images along horizontal and vertical axes of the image, maintaining the spatial decay matrix without losing prior information. This decomposition allows the Manhattan Self-Attention (MaSA) to model global information efficiently with linear complexity, while preserving the original MaSA's receptive field shape.
#
# However, unlike the original RetNet, which performs training with parallel representation and inference with recurrent representation, RMT does both processes with the MaSA mechanism. The authors have made comparisons between MaSA and other RetNet's representations, and they show that MaSA has the best throughput with the highest accuracy.
# ![MaSA vs Retention](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MaSA%20vs%20Retention.png)
#
# ### ViR
# ![ViR](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/ViR.png)
#
# Another work inspired by the RetNet architecture is the ViR, as discussed in the paper [ViR: Vision Retention Networks](http://arxiv.org/abs/2310.19731). In this architecture, the authors propose a general vision backbone with a redesigned retention mechanism. They demonstrate that ViR can scale favorably to larger image resolutions in terms of image throughput and memory consumption by leveraging the dual parallel and recurrent properties of the retentive network.
#
# The overall architecture of ViR is quite similar to that of ViT, except that it replaces the Multi-Head Attention (MHA) with Multi-Head Retention (MHR). This MHR mechanism is free of any gating function and can be switched between parallel, recurrent, or chunkwise (a hybrid between parallel and recurrent) modes. Another difference in ViR is that the positional embedding is first added to the patch embedding, and then the [class] token is appended.
#
# ## Further Reading
#
# - [RetNet's official repo](https://github.com/microsoft/torchscale/blob/main/torchscale/architecture/retnet.py)
# - [RetNet's Multi-Scale Retention official repo](https://github.com/microsoft/torchscale/blob/main/torchscale/component/multiscale_retention.py)
# - [Retentive Networks (RetNet) Explained: The much-awaited Transformers-killer is here](https://medium.com/ai-fusion-labs/retentive-networks-retnet-explained-the-much-awaited-transformers-killer-is-here-6c17e3e8add8)
# - [Retentive Network: A Successor to Transformer for Large Language Models (Paper Explained)](https://www.youtube.com/watch?v=ec56a8wmfRk)
# - [RMT's official repo](https://github.com/qhfan/RMT)
# - [ViR's official repo](https://github.com/NVlabs/ViR)
#
# Then, these feature maps are concatenated together and flattened. Later on, we use something called dropout to drop a portion of parameters to avoid overfitting. Finally, the final form of weights will go through a dense layer to get classified and backpropagation will take place. : If you're looking to understand the detailed training process of MERLOT, make sure to refer to the MERLOT paper as well as earlier works like [HERO](https://aclanthology.org/2020.emnlp-main.161.pdf), [CBT](https://arxiv.org/pdf/1906.05743) and [HAMMER](https://aclanthology.org/2020.emnlp-main.161.pdf). Convolutional Neural Networks (CNNs) are excellent at analyzing spatial features in images. However, they aren't designed to handle sequences where temporal relationships matter. This is where Recurrent Neural Networks (RNNs) come in. RNNs are specialized for processing sequential data because they have a "memory" that captures information from previous steps. This makes them well-suited for understanding how video frames relate to each other over time.
# Overview
# What is Hiera?
# Hiera (Hierarchical Vision Transformer) is an architecture that achieves high accuracy without the need for specialized components found in other vision models. The authors propose pretraining Hiera with a strong visual pretext task to remove unnecessary complexity and create a faster and more accurate model.
#
# Hiera Architecture
#
# From CNNs to ViTs
# CNNs and hierarchical models are well-suited for computer vision tasks because they can effectively capture the hierarchical and spatial structure of visual data. These models use fewer channels but higher spatial resolution in the early stages to extract simpler features and more channels but lower spatial resolution in the later stages to extract more complex features.
#
# CNNs
#
# On the other hand, Vision Transformers (ViTs) are more accurate, scalable, and architecturally simple models that took computer vision by storm when they were introduced. However, this simplicity comes at a cost: they lack this “vision inductive bias” (their architecture is not designed to work specifically with visual data).
#
# Many efforts have been made to adapt ViTs, generally by adding hierarchical components to compensate for this lack of inductive bias in their architecture. Unfortunately, all of the resulting models turned out to be slower, bigger and more difficult to scale.
#
# Hiera’s Approach: Pretraining Task is All You Need
# Authors of the Hiera paper argue that a ViT model can learn spatial reasoning and perform well on computer vision tasks by using a strong visual pretext task called MAE and thus, they can remove unnecessary components and complexity from state-of-the-art multi-stage vision transformers to achieve greater accuracy and speed.
#
# What components are the paper authors actually removing? To understand this we first have to introduce MViTv2 which is the base hierarchical architecture from which Hiera is derived. MViTv2 learns multi-scale representations over its four stages: it starts by modeling low level features with a small channel capacity but high spatial resolution, and then in each stage trades channel capacity for spatial resolution to model more complex high-level features in deeper layers.
#
# MViTv2
#
# Instead of digging deep into MViTv2’s key features (since it’s not our main topic), we will breifly explain them in the next section to illustrate how researchers create Hiera by simplyfing this base architecture.
#
# Simplifying MViTv2
# Simplifying MViTv2
#
# This table lists all the changes that authors made to MViTv2 in order to create Hiera and how each change affects accuracy and speed on images and video.
#
# Replacing relative with absolute position embeddings: MViTv2 swaps the absolute position embeddings from the original ViT paper with relative ones added to attention in each block. Authors undo this change because it added more complexity to the model and, as it can be seen in the table, these relative position embeddings are not necessary when training with MAE (both accuracy and speed improve with this change).
# Removing convolutions: Since the key idea of the paper is that a model can learn spatial biases by pretraining with a strong visual pretext task, removing convolutions, which are vision specific modules and add potentially unnecessary overhead seems to be an important change. Authors first replace every conv with a max pooling layer which decreases accuracy at first because of the huge impact it has on the image features. However, they realize that they can remove some of these extra max pooling layers, specifically the ones with a stride of 1 since they are basically applying a ReLU on every feature map. By doing so, authors nearly returned to the accuracy they had before, while speeding up the model by 22% for images and 27% for video.
# Masked Autoencoder
# Masked Autoencoder (MAE) is an unsupervised training paradigm. As with any other autoencoder, it consists of encoding high dimensional data (images) into a lower dimension representation (embeddings) in such a way that this data can be decoded into the original high dimensional data again. However, the visual MAE technique consists of dropping a certain amount of patches (around 75%), encoding the rest of the patches, and then trying to predict the missing ones. This idea has been used extensively in recent years as a pre-training task for image encoders.
#
# MAE
#
# # Hyena
#
# ## Overview
#
# ### What is Hyena
# While
# Transformer is a
# well
# established and very
# capable
# architecture, the
# quadratic
# computational
# cost is an
# expensive
# price
# to
# pay, especially in inference.
#
# Hyena is a
# new
# type of operator
# that
# serves as a
# substitute
# for the attention mechanism.
# Developed
# by
# Hazy
# Research, it
# features
# a
# subquadratic
# computational
# efficiency, constructed
# by
# interleaving
# implicitly
# parametrized
# long
# convolutions and data - controlled
# gating.
#
# Long
# convolutions
# are
# similar
# to
# standard
# convolutions except the
# kernel is the
# size
# of
# the
# input.
# It is equivalent
# to
# having
# a
# global receptive
# field
# instead
# of
# a
# local
# one.
# Having
# an
# implicitly
# parametrized
# convolution
# means
# that
# the
# convolution
# filters
# values
# are
# not directly
# learned.Instead, learning
# a
# function
# that
# can
# recover
# thoses
# values is preferred.
#
# Gating
# mechanisms
# control
# the
# path
# through
# which
# information
# flows in the
# network.They
# help
# to
# define
# how
# long
# an
# information
# should
# be
# remembered.Usally
# they
# consist in elementwise
# multiplications.
# An
# interresting
# blog
# article
# about
# gating
# can
# be
# found
# here.
#
# ![transformer2hyena.png](
#     https: // huggingface.co / datasets / hf - vision / course - assets / resolve / main / outlook_hyena_images / transformer2hyena.png)
# The
# Hyena
# operator
# consists
# of
# recursively
# computing
# convolutions and multiplicative
# element - wise
# gating
# operations
# with one projection at a time, until all projections are exhausted.This approach builds on top of the[Hungry Hungry Hippo (H3)](https://
#     arxiv.org / abs / 2212.14052) mechanism, also
# developed
# by
# the
# same
# researchers.The
# H3
# mechanism is characterized
# by
# its
# data - controlled, parametric
# decomposition, acting as a
# surrogate
# attention
# mechanism.
#
# Another
# way
# of
# understanding
# Hyena is to
# consider
# it as a
# generalization
# of
# the
# H3
# layer
# for an arbitrary number of projections, where the Hyena layer extends recursively H3 with a different choice of parametrization for the long convolution.
# ![hyena_recurence.png](
#     https: // huggingface.co / datasets / hf - vision / course - assets / resolve / main / outlook_hyena_images / hyena_recurence.png)
# ### From Attention to Hyena operator
#
# The
# attention
# mechanism is characterized
# by
# two
# fundamental
# properties:
# 1.
# It
# possesses
# a
# global contextual
# awareness, enabling
# it
# to
# assess
# interactions
# between
# pairs
# of
# visual
# tokens
# within
# a
# sequence.
# 2.
# It is data - dependent, meaning
# the
# operation
# of
# the
# attention
# equation
# varies
# based
# on
# the
# input
# data
# itself, specifically
# the
# input
# projections  \\(q\\), \\(k\\), \\(v\\).
#
# ![Alt text](
#     https: // huggingface.co / datasets / hf - vision / course - assets / resolve / main / outlook_hyena_images / self - attention - schema.png)
#
# The
# attention
# mechanism is defined
# by
# three
# projections: query \\(q\\), key \\(k\\), value \\(v\\), that
# are
# generated
# by
# mutiliplying
# the
# input
# visual
# token
# by
# three
# matrices \\(W_q\\), \\(W_k\\) and \\(W_v\\)
# that
# are
# learned
# during
# training.
#
# For
# a
# given
# visual
# token, we
# can
# compute
# an
# attention
# score
# using
# thoses
# projections.The
# attention
# score
# determines
# how
# much
# focus
# to
# give
# on
# other
# parts
# of
# the
# input
# image.
# For
# a
# nice
# detailled
# explainer
# of
# Attention
# you
# can
# refer
# on
# this[illustrated
# blog
# article](https: // jalammar.github.io / illustrated-transformer /).
#
# In
# an
# attempt
# to
# replicate
# these
# characteristics, the
# Hyena
# operator
# incorporates
# two
# key
# elements:
# 1.
# It
# employs
# long
# convolution
# to
# provide
# a
# sense
# of
# global context, akin
# to
# the
# first
# property
# of
# the
# attention
# mechanism.
# 2.
# For
# data
# dependency, Hyena
# uses
# element - wise
# gating.This is essentially
# an
# element - wise
# multiplication
# of
# input
# projections, mirroring
# the
# data - dependent
# nature
# of
# traditional
# attention.
#
# In
# the
# realm
# of
# computational
# efficiency, the
# Hyena
# operator
# attains
# an
# evaluation
# time
# complexity
# of \\(O(L \times \log_2 L\\)), indicating
# a
# noteworthy
# enhancement in processing
# speed.
#
# ### Hyena operator
#
# Let
# 's delve into the second-order recursion of the Hyena operator, which simplifies its representation for illustrative purposes.
# ![hyena_mechanism.png](
#     https: // huggingface.co / datasets / hf - vision / course - assets / resolve / main / outlook_hyena_images / hyena - order2 - schema.png)
#
# In
# this
# order, we
# compute
# 3
# projections
# analogous
# to \\(q\\), \\(k\\) and \\(v\\)
# attention
# vectors
# from the Attention
#
# mechanism.
#
# However, unlike
# the
# attention
# mechanism, which
# typically
# uses
# a
# single
# dense
# layer
# for projecting the input sequence into representations, Hyena incorporates both a dense layer and standard convolutions that are performed on each channels (refered as  \\(T_q\\),  \\(T_k\\) and  \\(T_v\\) on the schema, but it is an explicit convolution in practice).The softmax function is also discared.
#
# The
# core
# idea is to
# repeatedly
# apply
# linear
# operators
# that
# are
# fast
# to
# evaluate
# to
# an
# input
# sequence \\(u \ in  \mathbb{R} ^ {L}\\)
# with \\(L\\) the length of the sequence.
# Because
# global convolutions
# have
# a
# large
# number
# of
# parameters, they
# are
# expensive
# to
# train.A
# notable
# design
# choice is the
# use
# of ** implicit
# convolutions **.
# Unlike
# standard
# convolutional
# layers, the
# convolution
# filter \\(h\\) is learned
# implicitly
# with a small neural network \\(\gamma_{\theta}\\) (also called the Hyena Filter).
# This
# network
# takes
# the
# positional
# index and potentially
# positional
# encodings as inputs.From
# the
# outputs
# of \\(\gamma_{\theta}\\)
# one
# can
# construct
# a
# Toeplitz
# matrix \\(T_h\\).
#
# This
# implies
# that
# instead
# of
# learning
# the
# values
# of
# the
# convolution
# filter
# directly, we
# learn
# a
# mapping
# from a temporal
#
# positional
# encoding
# to
# the
# values, which is more
# computationally
# efficient, especially
# for long sequences.
#
# It
# 's important to note that the mapping function can be conceptualized within various abstract models, such as Neural Field or State Space Models (S4) as discussed in H3 Paper.
#
# ### Implicit convolutions
#
# A
# linear
# convolution
# can
# be
# formulated as a
# matrix
# multiplication in which
# one
# of
# the
# inputs is reshaped
# into
# a[Toeplitz
# matrix](https: // en.wikipedia.org / wiki / Toeplitz_matrix).
#
# This
# transformation
# leads
# to
# greater
# parameter
# efficiency.
# Instead
# of
# directly
# learning
# fixed
# kernel
# weight
# values, a
# parametrized
# function is employed.
# This
# function
# intelligently
# deduces
# the
# values
# of
# the
# kernel
# weights and their
# dimensions
# during
# the
# network
# 's forward pass, optimizing resource use.
#
# One
# way
# to
# have
# an
# intuition
# about
# implicit
# parametrization is to
# think
# about
# an
# afine
# function \\(y=f(x) = a \times
# x + b\\) we
# want
# to
# learn.Instead
# of
# learning
# every
# single
# point
# positions
# it is more
# efficient
# to
# learn
# a and b and compute
# the
# points
# when
# needed.
#
# In
# practice, convolutions
# are
# accelerated
# to
# a
# subquadratic
# time
# complexity
# by
# the
# Cooley - Tukey
# fast
# Fourier
# transform(FFT)
# algorithm.
# Some
# work
# has
# been
# conducted
# to
# speed
# up
# this
# computation
# like
# FastFFTConv
# based
# on
# Monarch
# decomposition.
#
# ### Wrapping Up Everything
#
# ![nd_hyena.png](
#     https: // huggingface.co / datasets / hf - vision / course - assets / resolve / main / outlook_hyena_images / nd_hyena.png)
# In
# essence, Hyena
# can
# be
# performed in two
# steps:
# 1.
# Compute
# a
# set
# of
# N + 1
# linear
# projections
# similarly
# of
# attention(it
# can
# be
# more
# than
# 3
# projections).
# 2.
# Mixing
# up
# the
# projections: The
# matrix \\(H(u)\\) is defined
# by
# a
# combination
# of
# matrix
# multiplications.
#
# ## Why Hyena Matters
#
# The
# H3
# mechanism
# proposition
# went
# close
# to
# the
# perplexity
# of
# multi - headed
# attention
# mechanisms, but
# there
# was
# still
# a
# narrow
# gap in terms
# of
# perplexity
# that
# had
# to
# be
# bridged.
#
# A
# variety
# of
# attention
# replacements
# have
# been
# proposed
# over
# the
# last
# few
# years, and evaluating
# the
# quality
# of
# a
# new
# architecture
# during
# the
# exploratory
# phase
# remains
# challenging.
# Creating
# a
# versatile
# layer
# that
# can
# effectively
# process
# N - Dimensional
# data
# within
# deep
# neural
# networks
# while maintaining good expressiveness is a significant area of ongoing research.
#
# Empirically, Hyena
# operators
# are
# able
# to
# significantly
# shrink
# the
# quality
# gap
# with attention at scale, reaching similar perplexity and downstream performance with a smaller computational budget and without hybridization of attention.
# It
# has
# already
# achieved
# a
# state - of - the - art
# status
# for [DNA sequence modeling](https: //
#     arxiv.org / abs / 2306.15794) and shows
# great
# promise in the
# field
# of
# large
# language
# models
# with Stripped - Hyena - 7B.
#
# Similarly
# to
# Attention, Hyena
# can
# be
# used in computer
# vision
# tasks.In
# image
# classification, Hyena is able
# to
# match
# attention in accuracy
# when
# training
# on
# ImageNet - 1
# k
# from scratch.
#
# ![hyena_vision_benchmarks.png](
#     https: // huggingface.co / datasets / hf - vision / course - assets / resolve / main / outlook_hyena_images / hyena_vision_benchmarks.png)
# Hyena
# has
# been
# applied
# to
# N - Dimensional
# data
# with the Hyena N-D layer and can be used as direct drop- in replacement within the ViT, Swin, DeiT backbones.
#
# ![vit_vs_hyenavit.png](
#     https: // huggingface.co / datasets / hf - vision / course - assets / resolve / main / outlook_hyena_images / vit_vs_hyenavit.png)
# here is a
# noticeable
# enhancement in GPU
# memory
# efficiency
# with the increase in the number of image patches.
#
# Hyena
# Hierarchy
# facilitates
# the
# development
# of
# larger, more
# efficient
# convolution
# models
# for long sequences.
#     The
#     potential
#     for Hyena type models for computer vision would be a more efficient GPU memory consumption of patches, that would allow:
#         - The
#     processing
#     of
#     larger, higher - resolution
#     images
# - The
# use
# of
# smaller
# patches, allowing
# a
# fine - graine
# feature
# representation
#
# These
# qualities
# would
# be
# particularly
# beneficial in areas
# such as Medical
# Imaging and Remote
# Sensing.
#
# ## Towards Transformers Alternatives
# Building
# new
# layers
# from simple design
#
# principles is an
# emerging
# research
# field
# that is progressing
# very
# quickly.
#
# The
# H3
# mechanism
# serves as the
# foundation
# for many State Space Model based (SSM) architectures, typically featuring a structure that alternates between a block inspired by linear attention and a multi-layer perceptron (MLP) block.
# Hyena, as an
# enhancement
# of
# this
# approach, has
# paved
# the
# way
# for even more efficient architectures such as Mamba and its derivatives for vision (Vision Mamba, VMamba etc...).
#
# ## Further Reading
# - Hyena
# offical
# repo: [Convolutions for Sequence Modeling](https: // github.com / HazyResearch / safari)
# - On
# the
# landscape
# of
# subquadratic
# models: [The Safari of Deep Signal Processing: Hyena and Beyond · Hazy
# Research(stanford.edu)](https: // hazyresearch.stanford.edu / blog / 2023-06-08-hyena-safari)
# - On
# speeding
# up
# the
# FFT
# algorithm: [FlashFFTConv: Efficient
# Convolutions
# for Long Sequences with Tensor Cores · Hazy Research (stanford.edu)](https://
#     hazyresearch.stanford.edu / blog / 2023 - 11 - 13 - flashfftconv)
# - On
# the
# subquadratic
# model
# landscape: [Zoology(Blogpost 1): Measuring and Improving
# Recall in Efficient
# Language
# Models · Hazy
# Research(stanford.edu)](https: // hazyresearch.stanford.edu / blog / 2023-12-11-zoology1-analysis)
# - Hyena
# applied
# to
# computer
# vision: [[2309.13600] Multi - Dimensional Hyena
# for Spatial Inductive Bias (arxiv.org)](https://
#     arxiv.org / abs / 2309.13600)
# - An
# improved
# approach: [[2401.09417] Vision Mamba: Efficient
# Visual
# Representation
# Learning
# with Bidirectional State Space Model (arxiv.org)](https://
#     arxiv.org / abs / 2401.09417)
#
# # Image-based Joint-Embedding Predictive Architecture (I-JEPA)
#
# ## Overview
#
# The Image-based Joint-Embedding Predictive Architecture (I-JEPA) is a groundbreaking self-supervised learning model [introduced by Meta AI in 2023](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/). It tackles the challenge of understanding images without relying on traditional labels or hand-crafted data augmentations.
# To get to know I-JEPA better, let’s first discuss a few concepts.
#
# ### Invariance-based vs. Generative Pretraining Methods
#
# We can say that there are broadly two main approaches for self-supervised learning from images: invariance-based methods and generative methods. Both approaches have their strengths and weaknesses.
#
# - **Invariance-based methods**: In these methods, the model tries to reproduce similar embeddings for different views of the same image. And, of course, these different views are hand-crafted, the image augmentations we’re all familiar with. For example, rotating, scaling, and cropping. These methods are good at producing representations at high semantic levels, but the problem is that they introduce strong biases that may be detrimental to certain downstream tasks. For example, image classification and instance segmentation do not require data augmentations.
#
# - **Generative methods**: The model tries to reconstruct the input image using these methods. That’s why these methods are sometimes called reconstruction-based self-supervised learning. Masks hide patches of the input image, and the model tries to reconstruct these corrupted patches at the pixel or token level (let’s keep this point in mind). This masked approach can easily generalize beyond image modality but doesn’t produce representations at the quality level of invariance-based methods. Also, these methods are computationally expensive and require large datasets for robust training.
#
# Now let’s talk about Joint-Embedding Architectures.
#
# ### Joint-Embedding Architectures
#
# This is a recent and popular approach for self-supervised learning from images in which two networks are trained to produce similar embeddings for different views of the same image. Basically, they train two networks to "speak the same language" about different views of the same picture. A common choice is the Siamese network architecture where the two networks share the same weights. But like everything else, it has its own problems:
#
# - **Representation collapse**: A case in which the model produces the same representation regardless of the input.
#
# - **Inputs compatibility criteria**: Finding good and appropriate compatibility measures can be challenging sometimes.
#
# An example of a Joint-Embedding Architecture is [VICReg](https://arxiv.org/abs/2105.04906)
#
# Different training methods can be employed to train Joint-Embedding Architectures, for example:
#
# - Contrastive methods
# - Non-Contrastive methods
# - Clustering methods
#
# So far so good, now to I-JEPA. As a start, the picture below from the I-JEPA paper shows the difference between Joint-Embedding methods, generative methods, and I-JEPA.
#
# ![I-JEPA Comparisons](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/i-jepa-1.png)
#
# ### Image-based Joint-Embedding Predictive Architecture (I-JEPA)
#
# I-JEPA tries to improve on both generative and joint-embedding methods. Conceptually, it is similar to generative methods but with the following key differences:
#
# 1. **Abstract prediction**: This is the most fascinating aspect of I-JEPA in my opinion. Remember when we mentioned generative methods and how they try to reconstruct the corrupted input at the pixel level? Now, unlike generative methods, I-JEPA tries to predict it in representation space using its introduced predictor, that’s why they call it abstract prediction. This leads to the model learning more powerful semantic features.
#
# 2. **Multi-block masking**: Another design choice that improves the semantic features produced by I-JEPA is masking sufficiently large blocks of the input image.
#
# ### I-JEPA Components
#
# The previous diagrams show and compare the I-JEPA architecture, below is a brief description of its main components:
#
# 1. **Target Encoder (y-encoder)**: Encodes target images and the target blocks are produced by masking its output.
#
# 2. **Context Encoder (x-encoder)**: Encodes a randomly sampled context block from the image to obtain a corresponding patch-level representation.
#
# 3. **Predictor**: Takes as input the output of the context encoder and a mask token for each patch we wish to predict and tries to predict the masked target blocks.
#
# The target-encoder, context-encoder, and predictor all use a Vision Transformer (ViT) architecture. You have a refresher about them in unit 3 of this course.
#
# The image below from the paper illustrates how I-JEPA works.
#
# ![I-JEPA method](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/i-jepa-2.png)
#
# ## Why It Matters
#
# So, why I-JEPA? I-JEPA introduced many new design features while still being a simple and efficient method for learning semantic image representations without relying on hand-crafted data augmentations. Briefly,
#
# 1. I-JEPA outperforms pixel-reconstruction methods such as Masked autoencoders (MAE) on ImageNet-1K linear probing, semi-supervised 1% ImageNet-1K, and semantic transfer tasks.
#
# 2. I-JEPA is competitive with view-invariant pretraining approaches on semantic tasks and achieves better performance on low-level vision tasks such as object counting and depth prediction.
#
# 3. By using a simpler model with less rigid inductive bias, I-JEPA is applicable to a wider set of tasks.
#
# 4. I-JEPA is also scalable and efficient. Pretraining on ImageNet requires *less than 1200 GPU hours*.
#
# ## References
#
# - [I-JEPA paper](https://arxiv.org/abs/2301.08243)
#
# - [Meta's blog post about I-JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
#
# - [I-JEPA official GitHub repository](https://github.com/facebookresearch/ijepa)
#
