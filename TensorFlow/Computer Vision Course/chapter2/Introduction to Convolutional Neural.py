"""
================================================================================
 CV Course · Chapter 2 · 卷积神经网络(CNN)入门（学习笔记描述）
================================================================================
 一句话：CV 的地基——卷积提局部特征、池化降维、堆叠成深网(LeNet→VGG→ResNet)。
 本章讲：
   ① 卷积层(Conv2D)/池化(MaxPool)/全连接的作用与堆叠。
   ② 经典结构：VGG19 的 PyTorch 实现；Keras/PyTorch 两种写法对照。
   ③ 为什么卷积适合图像(局部性 + 参数共享 + 平移不变)。
 要点：CNN 靠"局部感受野 + 权重共享"高效抓图像特征，是 ViT 之前的主力。
 说明：代码为结构示例(部分注释态)，需 torch/keras；参考为主。
================================================================================
"""

# model = keras.Sequential(
#     [
#         keras.Input(shape=input_shape),
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )
# model.summary()
#
#
# import torch.nn as nn
#
#
# class VGG19(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(VGG19, self).__init__()
#
#         # 特征提取层：卷积层和池化层
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(
#                 3, 64, kernel_size=3, padding=1
#             ),  # 3 个输入通道，64 个输出通道，3x3 内核，1 填充
#             nn.ReLU()，
#         nn.Conv2d(64, 64, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.MaxPool2d(
#             kernel_size=2, stride=2
#         ),  # 使用 2x2 内核和步长 2 进行最大池化
#         nn.Conv2d(64, 128, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.Conv2d(128, 128, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.MaxPool2d(kernel_size=2, stride=2),
#         nn.Conv2d(128, 256, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.Conv2d(256, 256, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.Conv2d(256, 256, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.Conv2d(256, 256, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.MaxPool2d(kernel_size=2, stride=2),
#         nn.Conv2d(256, 512, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.MaxPool2d(kernel_size=2, stride=2),
#         nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         nn.ReLU()，
#         nn.MaxPool2d(kernel_size=2, stride=2),
#         ）
#
#         # 池化层
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
#
#         # 用于分类的全连接层
#         self.classifier = nn.Sequential(
#             nn.Linear(
#                 512 * 7 * 7，4096
#         ），  # 512 个通道，最大池化后的空间维度为7x7
#         nn.ReLU()，
#         nn.Dropout(0.5),  # Dropout 概率为 0.5 的 Dropout 层
#         nn.Linear(4096, 4096),
#         nn.ReLU()，
#         nn.Dropout(0.5),
#         nn.Linear(4096, num_classes),  # 输出层，输出单元数为 'num_classes'
#         ）
#
#         def forward(self, x):
#             x = self.feature_extractor(x)  # 将输入传递给特征提取层
#             x = self.avgpool(x)  # 将数据传递给池化层
#             x = x.view(x.size(0), -1)  # 将输出展平，以便传递给全连接层
#             x = self.classifier(x)  # 将展平后的输出传递给分类器层
#             return x
#
# #GoogleNet 的卷积架构。
# import torch
# import torch.nn as nn
#
#
# class BaseConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(BaseConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         return x
#
#
# class InceptionModule(nn.Module):
#     def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_proj):
#         super(InceptionModule, self).__init__()
#
#         self.b1 = nn.Sequential(
#             nn.Conv2d(in_channels, n1x1, kernel_size=1),
#             nn.ReLU(True),
#         )
#
#         self.b2 = nn.Sequential(
#             BaseConv2d(in_channels, n3x3red, kernel_size=1),
#             BaseConv2d(n3x3red, n3x3, kernel_size=3, padding=1),
#         )
#
#         self.b3 = nn.Sequential(
#             BaseConv2d(in_channels, n5x5red, kernel_size=1),
#             BaseConv2d(n5x5red, n5x5, kernel_size=5, padding=2),
#         )
#
#         self.b4 = nn.Sequential(
#             nn.MaxPool2d(3, stride=1, padding=1),
#             BaseConv2d(in_channels, pool_proj, kernel_size=1),
#         )
#
#     def forward(self, x):
#         y1 = self.b1(x)
#         y2 = self.b2(x)
#         y3 = self.b3(x)
#         y4 = self.b4(x)
#         return torch.cat([y1, y2, y3, y4], 1)
#
#
# class AuxiliaryClassifier(nn.Module):
#     def __init__(self, in_channels, num_classes, dropout=0.7):
#         super(AuxiliaryClassifier, self).__init__()
#         self.pool = nn.AvgPool2d(5, stride=3)
#         self.conv = BaseConv2d(in_channels, 128, kernel_size=1)
#         self.relu = nn.ReLU(True)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(2048, 1024)
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(1024, num_classes)
#
#     def forward(self, x):
#         x = self.pool(x)
#         x = self.conv(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
#
#
# class GoogLeNet(nn.Module):
#     def __init__(self, use_aux=True):
#         super(GoogLeNet, self).__init__()
#
#         self.use_aux = use_aux
#         ## block 1
#         self.conv1 = BaseConv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.lrn1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
#         self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
#
#         ## block 2
#         self.conv2 = BaseConv2d(64, 64, kernel_size=1)
#         self.conv3 = BaseConv2d(64, 192, kernel_size=3, padding=1)
#         self.lrn2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
#         self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
#
#         ## block 3
#         self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
#         self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
#         self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
#
#         ## block 4
#         self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
#         self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
#         self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
#         self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
#         self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
#         self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
#
#         ## block 5
#         self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
#         self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
#
#         ## auxiliary classifier
#         if self.use_aux:
#             self.aux1 = AuxiliaryClassifier(512, 1000)
#             self.aux2 = AuxiliaryClassifier(528, 1000)
#
#         ## block 6
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.dropout = nn.Dropout(0.4)
#         self.fc = nn.Linear(1024, 1000)
#
#     def forward(self, x):
#         ## block 1
#         x = self.conv1(x)
#         x = self.maxpool1(x)
#         x = self.lrn1(x)
#
#         ## block 2
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.lrn2(x)
#         x = self.maxpool2(x)
#
#         ## block 3
#         x = self.inception3a(x)
#         x = self.inception3b(x)
#         x = self.maxpool3(x)
#
#         ## block 4
#         x = self.inception4a(x)
#         if self.use_aux:
#             aux1 = self.aux1(x)
#         x = self.inception4b(x)
#         x = self.inception4c(x)
#         x = self.inception4d(x)
#         if self.use_aux:
#             aux2 = self.aux2(x)
#         x = self.inception4e(x)
#         x = self.maxpool4(x)
#
#         ## block 5
#         x = self.inception5a(x)
#         x = self.inception5b(x)
#
#         ## block 6
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         x = self.fc(x)
#
#         if self.use_aux:
#             return x, aux1, aux2
#         else:
#             return x
#
# #移动网络
# # MobileNet 是一种专为移动设备设计的神经网络架构。它由谷歌的研究团队开发，并于 2017 年首次推出。MobileNet 的主要目标是在智能手机、平板电脑和其他资源受限的设备上提供高性能、低延迟的图像分类和目标检测。
# #
# # MobileNet 通过使用深度可分离卷积来实现这一点，深度可分离卷积是一种比标准卷积更高效的替代方案。深度可分离卷积将计算分解为两个独立的步骤：深度卷积和逐点卷积。这显著减少了参数数量和计算复杂度，使 MobileNet 能够在移动设备上高效运行。
# #
# # MobileNet上的卷积类型
# # 通过用深度可分离卷积和逐点卷积替换常规卷积层，MobileNet 在最大限度地降低计算开销的同时实现了高精度，使其非常适合移动设备和其他资源受限的平台。MobileNet 中使用了两种类型的卷积：
# #
# # 深度可分离卷积
# # 在传统的卷积层中，每个滤波器同时对所有输入通道应用其权重。深度可分离卷积将其分解为两个步骤：深度可分离卷积和逐点可分离卷积。
# #
# # 此步骤使用较小的滤波器（通常为 3x3）对输入图像中的每个通道（单个颜色或特征）分别执行卷积。此步骤的输出与输入图像大小相同，但通道数更少。
# #
# # 逐点可分离卷积
# # 这种卷积方式在输入层和输出层的所有通道上都应用同一个滤波器（通常为 1x1）。它的参数比常规卷积少，可以看作是全连接层的替代方案，因此适用于计算资源有限的移动设备。
# #
# # 在深度可分离卷积之后，此步骤使用另一个 1x1 卷积层将前几个步骤的滤波输出进行组合。此操作有效地将深度可分离卷积学习到的特征聚合为一个更小的特征集，从而在保留重要信息的同时降低整体复杂度。
#
#
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from PIL import Image
# import requests
#
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# # 初始化处理器和模型
# preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
# model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
#
# # 预处理输入 inputs
# inputs = preprocessor(images=image, return_tensors="pt")
#
# # 获取输出和类别标签
# outputs = model(**inputs)
# logits = outputs.logits
#
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])
#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):
#         super().__init__()
#         self.depthwise = nn.Conv2d(
#             in_channels,
#             in_channels,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             groups=in_channels,
#         )
#         self.pointwise = nn.Conv2d(
#             in_channels, out_channels, kernel_size=1, stride=1, padding=0
#         )
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x
#
#
# class MobileNet(nn.Module):
#     def __init__(self, num_classes=1000):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
#
#         # MobileNet body
#         self.dw_conv2 = DepthwiseSeparableConv(32, 64, 1)
#         self.dw_conv3 = DepthwiseSeparableConv(64, 128, 2)
#         self.dw_conv4 = DepthwiseSeparableConv(128, 128, 1)
#         self.dw_conv5 = DepthwiseSeparableConv(128, 256, 2)
#         self.dw_conv6 = DepthwiseSeparableConv(256, 256, 1)
#         self.dw_conv7 = DepthwiseSeparableConv(256, 512, 2)
#
#         # 5 depthwise separable convolutions with stride 1
#         self.dw_conv8 = DepthwiseSeparableConv(512, 512, 1)
#         self.dw_conv9 = DepthwiseSeparableConv(512, 512, 1)
#         self.dw_conv10 = DepthwiseSeparableConv(512, 512, 1)
#         self.dw_conv11 = DepthwiseSeparableConv(512, 512, 1)
#         self.dw_conv12 = DepthwiseSeparableConv(512, 512, 1)
#
#         self.dw_conv13 = DepthwiseSeparableConv(512, 1024, 2)
#         self.dw_conv14 = DepthwiseSeparableConv(1024, 1024, 1)
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(1024, num_classes)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#
#         x = self.dw_conv2(x)
#         x = F.relu(x)
#         x = self.dw_conv3(x)
#         x = F.relu(x)
#         x = self.dw_conv4(x)
#         x = F.relu(x)
#         x = self.dw_conv5(x)
#         x = F.relu(x)
#         x = self.dw_conv6(x)
#         x = F.relu(x)
#         x = self.dw_conv7(x)
#         x = F.relu(x)
#
#         x = self.dw_conv8(x)
#         x = F.relu(x)
#         x = self.dw_conv9(x)
#         x = F.relu(x)
#         x = self.dw_conv10(x)
#         x = F.relu(x)
#         x = self.dw_conv11(x)
#         x = F.relu(x)
#         x = self.dw_conv12(x)
#         x = F.relu(x)
#
#         x = self.dw_conv13(x)
#         x = F.relu(x)
#         x = self.dw_conv14(x)
#         x = F.relu(x)
#
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x
#
#
# # Create the model
# mobilenet = MobileNet(num_classes=1000)
# print(mobilenet)
#
#
# #ConvNext——面向2020年代的卷积神经网络（2022年）
#
# # 微型设计
# # 除了上述修改之外，作者还对模型进行了一些微观设计上的调整。微观设计指的是底层结构决策，例如激活函数的选择和层细节的设定。一些值得注意的微观调整包括：
# #
# # 激活：将 ReLU 激活替换为 GELU（高斯误差线性单元），并从残差块中消除所有 GELU 层，只保留两个 1×1 层之间的一个 GELU 层。
# # 归一化：通过移除两个 BatchNorm 层并将 BatchNorm 替换为 LayerNorm 来减少归一化层数，在 conv 1 × 1 层之前只保留一个 LayerNorm 层。
# # 下采样层：在 ResNet 各阶段之间添加一个单独的下采样层。这些最终的修改将 ConvNext 的准确率从 80.6% 提高到 82.0%。最终的 ConvNext 模型超过了 Swin Transformer 的 81.3% 的准确率。
#
# # MobileNet
# # Timm是什么？
# # timm（或PyTorch Image Models ）是一个 Python库，它提供了一系列预训练的深度学习模型，主要专注于计算机视觉任务，以及用于训练、微调和推理的实用程序。
#
# import timm
# import torch
#
# # 加载预训练的 MobileNet 模型
# model_name = "mobilenetv3_large_100"
#
# model = timm.create_model(model_name, pretrained=True)
#
# # 如果你想将该模型用于推理
# model.eval()
#
# # 使用虚拟输入进行前向传播
# # 批大小为 1，3 个颜色通道，224x224 图像
# input_tensor = torch.rand(1, 3, 224, 224)
#
# output = model(input_tensor)
# print(output)
#
# #ResNet（残差网络）
#
# from transformers import ResNetForImageClassification
#
# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
#
# model.eval()
#
# from transformers import AutoFeatureExtractor, ResNetForImageClassification
# import torch
# from datasets import load_dataset
#
# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]
#
# feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
#
# inputs = feature_extractor(image, return_tensors="pt")
#
# with torch.no_grad():
#     logits = model(**inputs).logits
#
# # 模型预测 ImageNet 中 1000 个类别之一
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])
#
# # R-CNN、Fast R-CNN、Faster R-CNN
# # R-CNN（基于区域的卷积神经网络）
# # RCNN是利用卷积神经网络进行目标检测的最简单方法之一。简单来说，其基本思想是先检测一个“区域”，然后使用CNN对该区域进行分类。因此，这是一个多步骤的过程。基于这一思想，RCNN论文于2012年发表[1]。
# #
# # RCNN 使用以下步骤：
# #
# # 使用选择性搜索算法选择一个区域。
# # 使用基于卷积神经网络的分类器对该区域中的对象进行分类。
# # 为了进行培训，该论文提出了以下步骤。
# #
# # 创建一个包含从目标检测数据集中检测到的区域的数据集。
# # 在 regions 数据集上对 Alexnet 模型进行微调。
# # 然后使用微调后的模型进行目标检测数据集测试。
# # 以下是 R-CNN 的基本流程。
# # Fast RCNN
# # Fast RCNN 专注于对原始 RCNN 的改进。他们增加了以下四项改进。
# #
# # 与 R-CNN 的多阶段训练不同，本模型采用单阶段训练。使用多任务损失函数。
# # 无需磁盘存储。
# # 引入 ROI 池化层，仅获取感兴趣区域中的特征。
# # 与使用多任务损失的多步骤 RCNN / SPPnet 模型不同，该模型训练的是端到端模型。
# # Faster RCNN
# # Faster R-CNN 完全摒弃了选择性搜索算法！这些特性使得推理时间比 Fast R-CNN 缩短了 90%！
# #
# # 它引入了RPN，即区域提议网络。RPN是一种基于注意力机制的模型，它训练模型对图像中包含目标对象的区域给予“关注”。
# # 它将 RPN 与 Fast RCNN 相结合，使其成为端到端的目标检测模型。
# # 特征金字塔网络（FPN）
# # 特征金字塔网络是一种用于目标检测的 Inception 模型。
# # 它首先将图像降维为低维嵌入。
# # 然后它再次升级它们。
# # 它尝试根据每张放大后的图像预测输出结果（在本例中为类别）。
# # 但是，相似维度特征之间也存在跳跃连接！
