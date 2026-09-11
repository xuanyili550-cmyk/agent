# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块对应《从零构建大语言模型》一书附录 E 的可复用代码。

附录 E 主题：参数高效微调（Parameter-Efficient Fine-Tuning）方法之一 —— LoRA
（Low-Rank Adaptation，低秩自适应）。

LoRA 的核心思想：
    在微调大模型时，不直接更新原始权重矩阵 W（形状通常很大，如 [in_dim, out_dim]），
    而是冻结 W，另外引入一对低秩矩阵 A（[in_dim, rank]）和 B（[rank, out_dim]），
    用 ΔW ≈ A @ B 来近似权重的增量更新。由于 rank 远小于 in_dim/out_dim，
    需要训练和存储的参数量大幅减少，同时前向计算时把 LoRA 分支的输出
    与原始线性层的输出相加，等价于对原权重做了一次低秩修正：
        y = x @ W + (alpha / rank) * (x @ A @ B)
    这样既能保留预训练权重中的知识，又能以极小的额外参数量适配下游任务。

本文件包含三部分：
    1. LoRALayer：实现上述低秩分支 A、B 及其前向传播。
    2. LinearWithLoRA：把一个普通的 nn.Linear 层包装成"原始线性层 + LoRA 分支"的组合层。
    3. replace_linear_with_lora：递归遍历模型，把模型中所有的 nn.Linear 层
       替换为 LinearWithLoRA，从而给整个模型加上 LoRA 适配能力。
"""

import torch
import math


class LoRALayer(torch.nn.Module):
    """LoRA 低秩适配层，只学习权重的"低秩增量"，不直接学习完整的权重矩阵。

    数学形式：ΔW = A @ B，其中 A 形状为 [in_dim, rank]，B 形状为 [rank, out_dim]，
    rank 远小于 in_dim 和 out_dim，因此 A、B 的参数总量（rank*(in_dim+out_dim)）
    远小于直接学习一个 [in_dim, out_dim] 的满秩矩阵所需的参数量（in_dim*out_dim）。

    前向传播时用 alpha/rank 做缩放，alpha 是可调超参数，
    作用类似学习率的缩放因子，用于控制 LoRA 分支对最终输出的影响幅度。

    Args:
        in_dim (int): 输入特征维度，对应原始线性层的 in_features。
        out_dim (int): 输出特征维度，对应原始线性层的 out_features。
        rank (int): 低秩分解的秩（rank），即中间维度大小，决定了 A、B 的瓶颈宽度。
        alpha (float): 缩放系数，用于控制 LoRA 增量对输出的贡献大小。
    """

    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # A 矩阵：形状 [in_dim, rank]，将输入从 in_dim 维"压缩"到 rank 维（低秩瓶颈）
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
        # B 矩阵：形状 [rank, out_dim]，将 rank 维的中间表示"展开"到 out_dim 维
        # 初始化为全零，保证训练刚开始时 A@B = 0，即 LoRA 分支不改变模型的初始输出
        # （这是 LoRA 论文中的关键设计：训练起点等价于未加 LoRA 的原始模型）
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha  # 缩放系数，控制低秩增量对输出的影响强度
        self.rank = rank  # 低秩分解的秩，与 alpha 一起构成缩放因子 alpha/rank

    def forward(self, x):
        # x 的形状: [..., in_dim]
        # x @ self.A -> 形状 [..., rank]        （降维到低秩瓶颈空间）
        # (x @ self.A) @ self.B -> 形状 [..., out_dim]  （从低秩空间映射回输出维度）
        # 最终乘以 alpha/rank 做缩放，得到与原始线性层输出同维度的增量 ΔW·x
        x = (self.alpha / self.rank) * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    """将一个已有的 nn.Linear 层与一个 LoRALayer 组合起来的包装模块。

    前向传播时，输出 = 原始线性层输出 + LoRA 分支输出，
    即 y = x @ W + (alpha/rank) * (x @ A @ B)，
    这正是"冻结原始权重 W，只训练低秩增量 A、B"这一 LoRA 微调策略的实现方式。

    Args:
        linear (torch.nn.Linear): 待包装的原始线性层，其权重通常会被冻结（不参与梯度更新）。
        rank (int): 传给内部 LoRALayer 的低秩秩数。
        alpha (float): 传给内部 LoRALayer 的缩放系数。
    """

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear  # 保留原始线性层（通常会在训练脚本中被冻结参数）
        # 依据原始线性层的输入/输出维度构造对应的 LoRA 分支
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        # 原始线性层的输出与 LoRA 分支输出相加，即用低秩增量修正原始权重的效果
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank, alpha):
    """递归遍历模型的所有子模块，把其中的 nn.Linear 层原地替换为 LinearWithLoRA。

    这样无需手动修改模型定义代码，就能给一个已有模型（例如预训练好的 GPT 模型）
    的每一个线性层都注入 LoRA 低秩适配分支，从而实现参数高效微调：
    冻结原始的 nn.Linear 权重，只训练新增的 LoRA 参数 A、B。

    Args:
        model (torch.nn.Module): 待改造的模型（或子模块），会被就地（in-place）修改。
        rank (int): 传给每个新建 LoRALayer 的低秩秩数。
        alpha (float): 传给每个新建 LoRALayer 的缩放系数。

    Returns:
        None: 该函数直接在原模型对象上通过 setattr 替换子模块，没有返回值。
    """
    # named_children() 只遍历直接子模块（不含孙子模块），因此需要递归处理更深层结构
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            # 命中线性层：用 LinearWithLoRA 包装后，通过 setattr 原地替换到父模块的对应属性上
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            # 不是线性层（可能是容器模块，如 nn.Sequential、自定义 Block 等）：
            # 递归深入其子模块继续查找并替换其中的线性层
            replace_linear_with_lora(module, rank, alpha)
