"""
================================================================================
 CV Course · Chapter 12 · CV 模型的伦理与偏见（学习笔记描述）
================================================================================
 一句话：视觉模型会从数据里学到社会偏见，负责任的 AI 要能发现、评估、缓解偏见。
 本章讲：
   ① 真实案例(ImageNet Roulette、显著性算法偏见等)。
   ② 偏见如何渗入文本/视觉/语音各模态、有哪些类型。
   ③ 发现偏见的指标与缓解策略。
 要点：上线前必做偏见评估——这是伦理也是合规要求。
 说明：伦理综述(概念为主)，无需运行。
================================================================================
"""

# # Exploring Ethical Foundations in CV Models
#
# Welcome to the Ethics and Bias unit of our Computer Vision Course! 📸✨ This segment is designed to explore the critical elements of ethics and bias within the domain of computer vision.
#
# ## What you’ll learn 🖼️🤖
#
# In this unit we'll understand the ethical dimensions and potential biases within AI models and how it is crucial for responsible development and deployment of computer vision systems.
# A brief outline of the unit is given below:
#
# - We will start off this unit with Chapter 1, where we discuss the implications of the popular ImageNet Roulette Case Study.
# - In Chapter 2, we'll discuss about the ethical considerations connected with AI and computer vision technologies and why it is important to be fair, while developing CV systems.
# - In Chapter 3, we'll learn how bias can infiltrate AI models across various modalities such as text, vision, and speech.
# - In Chapter 4, we'll discuss various types of biases and their implications on computer vision models.
# - In Chapter 5, we'll discuss different ways to spot bias, and metrics for evaluating biases in CV models with the help of practical case studies.
# - In Chapter 6, we'll learn about the strategies and methods to mitigate biases specifically within computer vision models.
# - Finally, we close with Chapter 7 and discuss HuggingFace's mission and initiatives toward fostering ethical AI for society.
#
# ## Journey through the unit 🏃🏻‍♂️🏃🏻‍♀️
#
# Let's begin our journey that merges theoretical foundations, practical case studies, and ethical concerns inherent in the landscape of computer vision. From exploring real-world examples like the ImageNet Roulette case study to evaluating biases in AI models recognizing "Gay Face", Twitter's Saliency Algorithm and similar case studies, this unit dives deep into understanding, assessing, and mitigating biases in computer vision systems.
#
# By the conclusion of this unit, you'll have gained insights into recognizing biases, evaluating them in CV models, and effectively employing strategies to mitigate these biases. Additionally, you'll explore Hugging Face's efforts to promote ethical practices within AI, providing a roadmap towards responsible and transparent AI development.
#
# Join us as we navigate the domain of ethics and bias in computer vision, equipping ourselves to contribute ethically and responsibly to the future of AI and society.
# Let's shape the AI that is not only smart, but fair and responsible.
#
# Let’s dive in! 🚀🤗🌎
#
# # Ethics and Bias in AI 🧑‍🤝‍🧑
#
# We hope that you found the ImageNet Roulette case study interesting and learned what can go wrong with AI models in general. In this chapter, we will go through yet another example of a powerful technology that has cool applications but can also raise ethical concerns if kept unchecked. Let's first quickly summarize the ImageNet Roulette case study and its after-effects.
# - ImageNet Roulette is a great example of an AI model gone wrong, due to the inherent biases and overlooking the labelling and data pre-processing stages.
# - The experiment was hand-crafted just to demonstrate how things can go wrong if kept unchecked.
# - This project led to a great deal of corrections done by the ImageNet team in the dataset, as well as implementation of appropriate measures to mitigate problems like face obfuscation, removal of harmful and triggering synsets, removal of corresponding images, etc.
# - Finally, it opened up an ongoing discussion and fuelled the research work on mitigating the risks.
#
# Before we take a look at another example of a powerful technology, let's step back and reflect on some questions. In general, is technology good or bad? Is electricity good or bad? Is the internet generally safe or harmful? Etc. Keep these questions in mind as we begin our journey.
#
# ## Deepfakes 🎥
#
# Imagine you are a recent graduate who wants to learn about Deep Learning. You enroll in a course called "Introduction to Deep Learning (MIT 6.S191)" by MIT. To make things more interesting, the course team released a really cool video on things that can be done using deep learning. Check the video here:
#
# An introduction to the course MIT 6.S191, where deepfakes were used to give an impression of a welcome by Barack Obama.
#
# Yup, the introduction session was made in such a way that it puts an impression that the students are being welcomed by none other than Barack Obama himself. Really cool application, a course on deep learning with an introduction curated to showcase one of the use cases of deep generative models. For the first-timers, this would be really engaging, making them interested in the technology and everyone would like to try it out. After all you can actually make such videos and images easily within a few minutes using a decent GPU and start posting memes, posts etc surrounding this.
# Let's see another example of this technology but with different after-effects. Imagine if we could come up with the same deepfake of an influential political leader or actor during elections or wars. The same fake video can be used to spread hatred and misinformation, leading to marginalizing different people. Even though the person did not spread misinformation, the video itself can cause massive outrage. This can be horrifying. However, the main problem lies in the fact that once the misinformation is spread, the harm is already done, and people are divided, even if it later becomes clear that the video was manipulated. So, the harm can only be avoided if the manipulated video is not made public in the first place. This makes this technology dangerous, but is the technology itself safe or harmful? Technology itself is never good or bad, but its usage (who uses it and for what purpose) can have good or bad effects.
#
# Deepfakes are synthetic media that are created using the help of deep generative CV models. You can actually manipulate images with a different person's image and also generate videos through it. Audio deepfake is another technology that can complement the CV counterpart by mimicking the exact voice of the subject under consideration. This was just one of the examples of how deepfakes can cause havoc, but in reality, the implications are far more dangerous as they can have a lifelong impact on the lives of the victims.
#
# ## What is Ethics and Bias in AI?
#
# From the previous example, a few aspects of this technology to keep in mind would be:
# - Consent of the subject before using the images/videos to manipulate and form new media.
# - Algorithms that facilitate the creation of synthetic media which can be used for manipulation.
# - Algorithms that can be used to detect such synthetic media.
# - Awareness about these algorithms and their after-effects.
#
# 💡Check out The Consentful Tech Project [here](https://www.consentfultech.io/). This project raises awareness, develops strategies, and shares skills to help people build and use technology consentfully.
#
# Let us now formalize some definitions based on these examples. So what is Ethics and Bias? Ethics can be defined simply as a set of moral principles that help us distinguish between wrong and right. Now, AI Ethics can be defined as the set of values, principles, and techniques that employ widely accepted standards of right and wrong to guide moral conduct in the development and use of AI. AI Ethics is a multidisciplinary field that studies how to optimize AI's beneficial impact while reducing risks and adverse outcomes. This field involves a variety of stakeholders:
# - **Researchers, Engineers, Developers, and AI Ethicists:** people in charge of models, algorithms, datasets development, and curation.
# - **Government bodies, legal authorities (like lawyers):** bodies and people in charge of regulatory aspects of Ethical AI development.
# - **Companies and organizations:** stakeholders that are at the forefront of delivering AI products and services.
# - **Citizens:** people who use AI services and products in their daily lives and are largely affected by the technology.
#
# Bias in AI refers to the biases in the output of the algorithms, which might happen due to assumptions during model development or training data. These assumptions stem from the inherent biases that are inside humans who were responsible for the development. As a result, AI models and algorithms start reflecting on these biases. These biases can disrupt ethical development or principles and, therefore, need attention and ways to mitigate them. We will cover more about biases, how they creep into different AI models, their types, evaluation, and mitigation (with a focus on CV models) in detail in the upcoming chapters of the unit. To understand more about Ethics in AI, let us look closely into the principles for Ethical AI.
#
# ## Ethical AI Principles 🤗 🌎
#
# ### Asimov's Three Laws of Robotics 🤖
#
# There have been many historic works that reflect on development of ethics for technology. The earliest work can be traced back to the famous science fiction writer Isaac Asimov. He came up with the three laws of robotics keeping in mind the potential risks of autonomous AI agents. The laws are:
# - A robot may not injure a human being or, through inaction, allow a human being to come to harm.
# - A robot must obey orders given to it by human beings except where such orders would conflict with the first law.
# - A robot must protect its own existence as long as such protection does not conflict with the first or second law.
#
# ### Asilomar AI Principles 🧑🏻‍⚖️🧑🏻‍🎓🧑🏻‍💻
#
# Asimov's laws of robotics were one of the earliest works in ethics for technology. In 2017, a conference was organized at Asilomar Conference Grounds, California. This conference was held to discuss the impacts of AI on society. The outcome of this conference was the development of guidelines for the responsible development of AI. The guideline has 23 principles, which were signed by around 5,000 individuals, including 844 AI and robotics researchers.
#
# ![Asilomar AI Principles](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/asilomar-ai.png)
# 23 Asilomar AI Principles for Responsible AI Development
#
# 💡You can check the full list of the 23 Asilomar AI Principles and the signatories [here](https://futureoflife.org/open-letter/ai-principles/).
#
# These principles are a guide for ethical development and implementation of AI models in general. Let's now look into a recent work on ethical AI guidelines by UNESCO.
#
# ### UNESCO's report: Recommendations on the Ethics of Artificial Intelligence 🧑🏼‍🤝‍🧑🏼🌐
#
# UNESCO came up with a global standard on AI ethics in the form of a report named **"Recommendation on the Ethics of Artificial Intelligence"**, which was adopted by 193 member countries in November 2021. Previous guidelines on Ethical AI were lacking in terms of actionable policy. Still, the recent report by UNESCO allows policymakers to translate the core principles into action concerning different domains like data governance, environment, gender, health, etc. The four core values of the recommendation which lay the foundation for AI systems are:
# - **Human rights and human dignity** Respect, protection, and promotion of human rights and fundamental freedoms and human dignity.
# - **Living in Peaceful** just and interconnected societies.
# - **Ensuring diversity and inclusiveness.**
# - **Environment and ecosystem flourishing.**
#
# ![AI Policy Areas](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/ai_policy.png)
# 11 key policy areas for responsible developments in AI.
#
# The ten core principles that lay out a human-rights centered approach to Ethics of AI by UNESCO are give below:
# - **Proportionality and Do No Harm:** The use of AI systems must not go beyond what is necessary to achieve a legitimate aim. Risk assessment should be used to prevent harm that may result from such uses.
# - **Safety and Security:** Unwanted harms (safety risks), as well as vulnerabilities to attack (security risks), should be avoided and addressed by AI actors.
# - **Right to Privacy and Data Protection:** Privacy must be protected and promoted throughout the AI lifecycle. Adequate data protection frameworks should also be established.
# - **Multi-stakeholder and Adaptive Governance & Collaboration:** International law & national sovereignty must be respected in the use of data. Additionally, the participation of diverse stakeholders is necessary for inclusive approaches to AI governance.
# - **Responsibility and Accountability:** AI systems should be auditable and traceable. There should be oversight, impact assessment, audit, and due diligence mechanisms in place to avoid conflicts with human rights norms and threats to environmental well-being.
# - **Transparency and Explainability:** The ethical deployment of AI systems depends on their transparency & explainability (T&E). The level of T&E should be appropriate to the context, as there may be tensions between T&E and other principles such as privacy, safety, and security.
# - **Human Oversight and Determination:** Member States should ensure that AI systems do not displace ultimate human responsibility and accountability.
# - **Sustainability:** AI technologies should be assessed against their impacts on "sustainability", understood as a set of constantly evolving goals, including those set out in the UN's Sustainable Development Goals.
# - **Awareness & Literacy:** Public understanding of AI and data should be promoted through open & accessible education, civic engagement, digital skills & AI ethics training, and media & information literacy.
# - **Fairness and Non-Discrimination:** AI actors should promote social justice, fairness, and non-discrimination while taking an inclusive approach to ensure AI's benefits are accessible to all.
#
# 💡To read the complete report by UNESCO on "Recommendations on the Ethics of Artificial Intelligence", you can visit [here](https://unesdoc.unesco.org/ark:/48223/pf0000381137).
#
# As we close the unit we will also look into Hugging Face's efforts to ensure Ethical AI practises. In the next chapter, we will learn more about biases, types and how they creep in different AI models.
#
# # Hugging Face's efforts: Ethics and Society 🤗🌎
#
# We hope you liked exploring the unit on ethics and bias in computer vision. As we conclude this unit, let us take a look into the efforts by Hugging Face to improve ethics in society. This chapter will encourage you to explore the world of ethics and bias in AI in general, which is constantly evolving. Hugging Face's core mission is to *democratize good machine learning*. So what is Good ML? There are some principles for Good ML.
#
# ## Democratizing Good ML 🤗
#
# **1. Collaboration:** Providing tools for easier collaboration with the open-source community. Some examples of these tools are:
#     a. [*Model Cards*](https://huggingface.co/docs/hub/model-cards) are files that accompany the models and provide information about the model, its intended uses and potential limitations (including ethical considerations), training parameters and experimental information, datasets used for training and evaluation results. This ensures that the models uploaded to the hub are transparent and open to the community.
#     b. [*Evaluation*](https://huggingface.co/blog/eval-on-the-hub) lets users to evaluate any model on any dataset that is openly available on the Hub without writing a single line of code. ,
#     c. [*Community discussion*](https://huggingface.co/blog/community-update) is important, whether you upload a model, dataset or a space or you just want to know more about them from the authors. Everyone can give feedback, flag a given Space, and improve or contribute directly to the repository via PRs.
#     d. [*Discord*](https://discord.com/invite/JfAtkvEtRb) group for the 🤗 community, where a wide range of channels discuss about different domains like reinforcement-learning, NLP, game development, audio, computer vision and so on.
# **2. Transparency:** Being transparent about the intent, sources of data, model training and performance. Efforts in this direction include:
#     a. [*Ethical charter for multimodal project*](https://huggingface.co/blog/ethical-charter-multimodal) discusses the values of the multimodal learning group at 🤗. This is a project-specific charter.
#     b. [*Work on AI policy @* 🤗](https://huggingface.co/blog/us-national-ai-research-resource) mentions the response of Hugging Face on U.S. National AI Research Resource Interim Report.
# **3. Responsibility:** Assessing the impacts of ML models and tools, and making them more auditable and understandable even for people with less expertise on ML. Some efforts in this directions are:
#     a. [🤗 *for Education Project*](https://huggingface.co/blog/education) aims at educating people from all backgrounds, beginners and instructors. Various experts and team members organize meetups, conferences and workshops.
#     b. [*Data Measurement Tool*](https://huggingface.co/spaces/huggingface/data-measurements-tool) is an interactive interface and open-source library that lets dataset creators and users automatically calculate metrics that are meaningful and useful for responsible data development.
#
# ## Categories of Hugging Face Spaces
#
# As we move forward, let us now take a look into how spaces are categorized by Hugging Face. Hugging Face categorizes the spaces in 6 high levels based on the ethical aspects in Machine Learning. Hugging Face Spaces are categorized as:
#
# ### ✍️ Rigorous
#
# Rigorous projects pay special attention to examining failure cases, protecting privacy through security measures, and ensuring that potential users (technical and non-technical) are informed of the project's limitations. Some examples:
# - Projects built with models that are well-documented with Model Cards.
# - Tools that provide transparency into how a model was trained and how it behaves.
# - Evaluations against cutting-edge benchmarks, with results reported against disaggregated sets.
# - Demonstrations of models failing across gender, skin type, ethnicity, age or other attributes.
# - Techniques for mitigating issues like over-fitting and training data memorization.
# - Techniques for detoxifying language models.
#
# An example space is the [**Diffusion Bias Explorer**](https://huggingface.co/spaces/society-ethics/DiffusionBiasExplorer) which lets the users compare three text-to-image models SD 1.4, SD 2.0 and Dall-E 2 for different prompts and how they represent different professions and adjectives.
#
# ### 🤝 Consentful
#
# Consentful technology supports the self-determination of people who use and are affected by these technologies. Some examples:
# - Demonstrating a commitment to acquiring data from willing, informed, and appropriately compensated sources.
# - Designing systems that respect end-user autonomy, e.g. with privacy-preserving techniques.
# - Avoiding extractive, chauvinist, "dark", and otherwise "unethical" patterns of engagement.
#
# Some example spaces for this category are:
# 1. [**Does CLIP Know My Face:**](https://huggingface.co/spaces/AIML-TUDA/does-clip-know-my-face) this space lets you choose a model, enter your name and upload some pictures. Depending on this information, the model tries to predict your name from the images, if it predicts name correctly for multiple images there are high chances that you were part of the training data.
# 2. [**Photoguard:**](https://huggingface.co/spaces/RamAnanth1/photoguard) this space demonstrates an approach to safeguarding images against manipulation by ML-powered photo-editing models such as SD, through immunization of images.
#
# ### 👁️‍🗨️ Socially Conscious
#
# Socially Conscious work shows us how machine learning can support efforts toward a stronger society. Some examples:
# - Using machine learning as part of an effort to tackle climate change.
# - Building tools to assist with medical research and practice.
# - Models for text-to-speech, image captioning, and other tasks aimed at increasing accessibility.
# - Creating systems for the digital humanities, such as for Indigenous language revitalization.
#
# Some example spaces:
# 1. [**Socratic Models Image Captioning**](https://huggingface.co/spaces/Geonmo/socratic-models-image-captioning-with-BLOOM)
# 2. [**Comparing Image Captioning Models**](https://huggingface.co/spaces/nielsr/comparing-captioning-models)
#
# ### 🌎 Sustainable
#
# This is work that highlights and explores techniques for making machine learning ecologically sustainable. Some examples:
# - Tracking emissions from training and running inferences on large language models.
# - Quantization and distillation methods to reduce carbon footprints without sacrificing model quality.
#
# 1. [**EfficientFormer**](https://huggingface.co/spaces/adirik/efficientformer)
# 2. [**EfficientNetV2 Deepfakes Video Detector**](https://huggingface.co/spaces/Ron0420/EfficientNetV2_Deepfakes_Video_Detector)
#
# ### 🧑‍🤝‍🧑 Inclusive
#
# These are projects which broaden the scope of who builds and benefits in the machine learning world. Some examples:
# - Curating diverse datasets that increase the representation of underserved groups.
# - Training language models on languages that aren't yet available on the Hugging Face Hub.
# - Creating no-code and low-code frameworks that allow non-technical folk to engage with AI.
#
# An example space is [**Promptist Demo**](https://huggingface.co/spaces/microsoft/Promptist).
#
# ### 🤔 Inquisitive
#
# Some projects take a radical new approach to concepts which may have become commonplace. These projects, often rooted in critical theory, shine a light on inequities and power structures which challenge the community to rethink its relationship to technology. Some examples:
# - Reframing AI and machine learning from Indigenous perspectives.
# - Highlighting LGBTQIA2S+ marginalization in AI.
# - Critiquing the harms perpetuated by AI systems.
# - Discussing the role of "openness" in AI research.
#
# An example space is [**PAIR: Datasets Have Worldviews**](https://huggingface.co/spaces/merve/dataset-worldviews).
#
# Finally, if you would like to explore more about Hugging Face's efforts, do check out the [**Society and Ethics**](https://huggingface.co/society-ethics) organzation on Hugging Face. Also check out the dedicated channel on [**#ethics-and-society**](https://discord.gg/hugging-face-879548962464493619).
#
# # Supplementary reading and resources 🤗🌎
#
# We
# hope
# that
# you
# had
# an
# excited
# learning
# journey
# throughout
# the
# unit
# of
# Ethics and Bias in CV
# models.To
# explore
# more
# about
# the
# field in general, you
# can
# go
# through
# these
# learning
# resources:
#
# - [ ** Ethics and Society
# Newsletter **](https: // huggingface.co / blog?tag=ethics)
# by
# Hugging
# Face.This
# newsletter
# discusses
# the
# efforts
# of
# Hugging
# Face in the
# domain
# of
# Ethical
# AI.Hugging
# Face
# also
# has
# a
# separate
# space
# dedicated
# to
# collections, spaces, datasets and models
# involving
# ethical
# AI, [here](https: // huggingface.co / society - ethics).
# - [ ** Data
# Ethics **](https: // ethics.fast.ai /)
# course
# by
# fast.ai.
# - [ ** Intro
# to
# AI
# Ethics **](https: // www.kaggle.com / learn / intro-to-ai-ethics)
# course
# on
# Kaggle
# Learn.This is a
# short
# course
# with exercises for beginners in the field.
# - [ ** Towards
# Fairer
# Datasets: Filtering and Balancing
# the
# Distribution
# of
# the
# People
# Subtree in the
# ImageNet
# Hierarchy **](https: // dl.acm.org / doi / abs / 10.1145 / 3351095.3375709)
# This
# paper
# discusses
# about
# the
# ImageNet
# dataset and how
# the
# bias in dataset
# was
# removed
# by
# filtering
# most
# of
# the
# synsets and balancing
# according
# to
# age, gender and color.
# - [ ** The
# AI
# Ethics
# Brief **](https: // brief.montrealethics.ai /)
# newsletter
# by
# Montreal
# AI
# Ethics
# Institute,
# with a section on[Computer Vision](https://
#     montrealethics.ai / category / columns / the - ethics - of - computer - vision /).The
# newsletter is a
# must
# read
# for learners and practitioners in the field.
# - [ ** Ethics
# of
# AI **](https: // ethics-of-ai.mooc.fi /)
# MOOC
# by
# University
# of
# Helsinki.
# - [ ** CS
# 281 **](https: // stanfordaiethics.github.io /)
# Ethics
# of
# AI
# Course
# by
# Stanford
# University.
# - [ ** Ethics
# of
# AI
# Bias **](https: // ocw.mit.edu / courses / res-10-002-ethics-of-ai-bias-spring-2023 /)
# course
# by
# MIT
# OCW.
#
