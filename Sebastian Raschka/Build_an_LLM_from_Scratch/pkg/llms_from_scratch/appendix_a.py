# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块对应《从零构建大语言模型》一书附录A(PyTorch 入门)的可复用代码。

附录A是全书唯一不涉及"大语言模型"本身的一章,目的是让读者熟悉 PyTorch
的两个最基础、也是后续所有章节都会反复用到的构建模块:

1. ``NeuralNetwork``:一个用 ``torch.nn.Module`` 搭建的最简单的多层感知机
   (MLP,全连接前馈神经网络),用来演示如何定义网络层、如何写
   ``forward`` 前向传播方法。
2. ``ToyDataset``:一个用 ``torch.utils.data.Dataset`` 搭建的最简单的自定义
   数据集类,用来演示 PyTorch 数据加载(Dataset + DataLoader)的标准接口。

这两个类本身与"语言模型"无关,只是教学用的玩具(toy)示例,但它们展示的
写法(继承 nn.Module 并实现 __init__/forward;继承 Dataset 并实现
__getitem__/__len__)在后续章节构建 GPT 等真实模型时会被反复复用,因此
非常值得仔细理解。
"""

import torch
from torch.utils.data import Dataset


class NeuralNetwork(torch.nn.Module):
    """一个最简单的多层感知机(MLP),演示 PyTorch 中定义模型的标准写法。

    该网络结构为:
        输入(num_inputs 维) -> 全连接(30) -> ReLU
                             -> 全连接(20) -> ReLU
                             -> 全连接(num_outputs) -> 输出 logits

    这是一个"隐藏层维度写死"的教学示例(30、20 为固定值,不是可配置的
    超参数),目的是让读者直观看到 nn.Sequential 如何串联多层。

    继承自 ``torch.nn.Module``:这是 PyTorch 中所有神经网络模块的基类,
    继承它之后 PyTorch 会自动帮我们管理网络内部的可学习参数(权重、偏置)、
    支持 .to(device)、.parameters()、反向传播等机制。

    参数:
        num_inputs (int): 输入特征的维度(即输入张量最后一维的大小)。
        num_outputs (int): 输出的维度(例如分类任务中的类别数)。
    """

    def __init__(self, num_inputs, num_outputs):
        # 必须先调用父类 nn.Module 的构造函数,PyTorch 才能正确注册
        # 后续在 self 上定义的子模块(如下面的 self.layers),
        # 否则这些子模块不会被识别为模型参数,无法被优化器更新、
        # 也无法随 .to(device) 一起搬到 GPU。
        super().__init__()

        # nn.Sequential:按顺序堆叠多个层,前一层的输出自动作为
        # 后一层的输入,等价于手写多次 self.fcX = ... 再在 forward
        # 中依次调用,但代码更简洁。
        self.layers = torch.nn.Sequential(

            # 1st hidden layer
            # 全连接层:将 (batch, num_inputs) 的输入线性映射为
            # (batch, 30) 的隐藏表示;30 是本示例中写死的隐藏层宽度。
            torch.nn.Linear(num_inputs, 30),
            # ReLU 激活函数:引入非线性,否则多层线性层叠加等价于
            # 单层线性变换,网络将失去表达非线性关系的能力。
            torch.nn.ReLU(),

            # 2nd hidden layer
            # 第二个全连接层:(batch, 30) -> (batch, 20)
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            # 输出层:(batch, 20) -> (batch, num_outputs),
            # 注意这里不加激活函数,输出的是未归一化的 logits,
            # 后续通常配合 CrossEntropyLoss(内部自带 softmax)使用。
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        """前向传播:将输入张量依次通过 self.layers 中定义的各层。

        参数:
            x (torch.Tensor): 形状为 (batch_size, num_inputs) 的输入张量。

        返回:
            torch.Tensor: 形状为 (batch_size, num_outputs) 的未归一化
            logits(尚未经过 softmax/sigmoid 等归一化函数)。
        """
        # 依次经过:Linear(num_inputs,30) -> ReLU -> Linear(30,20) -> ReLU
        # -> Linear(20,num_outputs),最终得到 logits。
        logits = self.layers(x)
        return logits


class ToyDataset(Dataset):
    """一个最简单的自定义数据集类,演示 PyTorch Dataset 接口的标准写法。

    继承自 ``torch.utils.data.Dataset`` 并实现 ``__getitem__`` 与
    ``__len__`` 两个魔术方法后,该对象即可直接传给
    ``torch.utils.data.DataLoader`` 使用,由 DataLoader 负责自动分批
    (batching)、打乱顺序(shuffle)、多进程加载等。

    参数:
        X (array-like / torch.Tensor): 特征数据,形状为
            (num_samples, num_features),即样本数 × 每个样本的特征维度。
        y (array-like / torch.Tensor): 标签数据,形状为 (num_samples,)。
    """

    def __init__(self, X, y):
        # 直接持有引用,不做拷贝;X、y 应当是样本数对齐的
        # (即 X.shape[0] == y.shape[0])。
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        """按索引取出单个样本,DataLoader 会在内部对每个 batch 内的多个
        索引重复调用本方法,再将结果自动堆叠(stack)成一个 batch。

        参数:
            index (int): 样本索引,取值范围 [0, len(self))。

        返回:
            tuple: (one_x, one_y),分别为该样本的特征和标签
            (未经堆叠的单条数据,不带 batch 维度)。
        """
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        """返回数据集中的样本总数。

        DataLoader 依据该值确定一共有多少个样本可供迭代/分批,
        以及配合 shuffle 生成索引排列时的取值范围。

        返回:
            int: 样本数量,取 labels 张量第 0 维(样本维)的大小。
        """
        # labels 的第 0 维即样本数量(标签是一维张量,形状为
        # (num_samples,)),因此 shape[0] 就是数据集长度。
        return self.labels.shape[0]
