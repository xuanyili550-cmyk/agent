"""Qwen3.5 helper blocks copied from Hugging Face Transformers

Source file:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py

License: Apache License Version 2.0
License URL: https://github.com/huggingface/transformers/blob/main/LICENSE

---------------------------------------------------------------------------
中文说明(模块级 docstring,补充于原英文说明之后):

本文件是从 HuggingFace `transformers` 库中 Qwen3.5(即 Qwen3-Next 系列架构)
模型实现里摘取出来的"门控 DeltaNet"(Gated DeltaNet)相关代码,用于教学笔记本
中单独演示这一层的实现原理,而不必安装完整的 `transformers` 库。

背景知识:
- Qwen3.5 / Qwen3-Next 是一种混合架构:大部分层使用标准的因果自注意力
  (full attention),但穿插了若干"线性注意力"层,这些线性注意力层就是本文件
  实现的 `Qwen3_5GatedDeltaNet`。它的计算/显存复杂度相对序列长度是线性的
  (而不是标准注意力的平方复杂度),原理接近 Mamba2(状态空间模型,SSM)与
  DeltaNet(增量法则线性注意力)的结合体:
    1) 用一个深度可分离的短因果卷积(short causal conv)先在局部窗口内混合信息;
    2) 再用"门控增量法则"(gated delta rule)以类似 RNN 的方式,把信息压缩进
       一个大小固定、不随序列长度增长的"循环状态"(recurrent state)中,
       并支持分块(chunk)并行计算以提升训练效率,同时支持逐 token 递归计算
       以支持自回归解码时的增量式 KV 缓存。

本文件相对于 HuggingFace 仓库做了如下"notebook 专用"改写(均已在下方对应
位置以注释标出),纯属工程简化,不影响数值结果:
  1) 将可选的高性能 CUDA 融合算子(causal_conv1d、fla 的
     chunk_gated_delta_rule/fused_recurrent_gated_delta_rule 等)全部禁用为
     `None`/`False`,统一走纯 PyTorch 实现,避免笔记本环境必须安装这些
     可选依赖包;
  2) 用极简的占位类 `Qwen3_5Config`、`Qwen3_5DynamicCache` 替代真实的
     HuggingFace 配置类和缓存类,只是为了保持与原始代码相同的书写结构,
     本文件内部并不真正依赖它们的具体字段/接口;
  3) 在 `Qwen3_5GatedDeltaNet.__init__` 末尾追加了 `self.to(dtype=config.dtype)`,
     用于强制统一子模块的 dtype,避免混合精度(如 bf16 与 fp32 混用)下的
     矩阵乘法报错(这是本文件相对于上游做的一处"最小改动",已在下方对应
     位置注明)。

经过与 `transformers` 官方仓库(main 分支的 `modeling_qwen3_5.py`,以及
历史版本 v4.57.1 的 `modeling_qwen3_next.py`)逐行比对确认:本文件中
门控增量法则的核心数学实现(`torch_chunk_gated_delta_rule` /
`torch_recurrent_gated_delta_rule` / `Qwen3_5RMSNormGated` /
`torch_causal_conv1d_update`)与上游代码逐字节一致,`Qwen3_5GatedDeltaNet`
的整体结构(拆分的 `in_proj_qkv/in_proj_z/in_proj_b/in_proj_a` 投影 +
更简单的列表式 KV 缓存接口)也与上游某一版本完全吻合,并未发现本文件
自身引入的确定性(deterministic)bug。比对中发现两处**继承自上游、且并非
本文件引入**的风险点,按要求只标注、不修改,见下方 `A_log` 初始化与
`self.act` 属性处的行内注释。
---------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Notebook shims for optional fast kernels in transformers
# 中文:以下这些变量在真实的 transformers 库中,分别对应到
# causal-conv1d 和 flash-linear-attention (fla) 两个可选加速库提供的
# CUDA 融合算子。本笔记本没有安装这些可选依赖,因此统一把它们置为 None/False,
# 使后面 `Qwen3_5GatedDeltaNet` 里 "有加速核就用加速核,没有就退回 PyTorch
# 实现" 的逻辑总是落到纯 PyTorch 分支(即本文件中的 torch_xxx 系列函数)。
# ---------------------------------------------------------------------------
causal_conv1d_fn = None  # 因果卷积的"整段序列"融合实现(未安装,禁用)
causal_conv1d_update = None  # 因果卷积的"单步/增量解码"融合实现(未安装,禁用)
chunk_gated_delta_rule = None  # 门控增量法则的"分块并行"融合实现(未安装,禁用)
fused_recurrent_gated_delta_rule = None  # 门控增量法则的"逐 token 递归"融合实现(未安装,禁用)
FusedRMSNormGated = None  # 融合版 Gated RMSNorm(未安装,禁用,退回下方纯 PyTorch 的 Qwen3_5RMSNormGated)
ACT2FN = {"silu": F.silu}  # 简化版激活函数注册表,本模型只会用到 "silu"
is_fast_path_available = False  # 快速路径(CUDA 融合算子)整体不可用的标记,恒为 False


class _NotebookLogger:
    """极简日志器,用于替代 transformers 内部的 `logging.get_logger`。

    只提供 `warning_once` 方法:同一条警告信息只打印一次,避免在训练/推理
    循环中反复刷屏(真实 transformers 的 logger 也有类似的"只警告一次"语义)。
    """

    def __init__(self):
        self._seen = set()  # 记录已经打印过的警告文本,用于去重

    def warning_once(self, msg):
        """打印一条警告信息,但同样内容的消息只打印一次。

        参数:
            msg (str): 警告文本。
        返回:
            None。
        """
        if msg in self._seen:
            return
        self._seen.add(msg)
        print(msg)


logger = _NotebookLogger()  # 模块级单例,供下方代码调用 logger.warning_once(...)


# ---------------------------------------------------------------------------
# Placeholder types for copied annotations
# 中文:这两个类只是"占位类型"。在真实 transformers 源码中,
# Qwen3_5Config 是完整的模型配置类,Qwen3_5DynamicCache 是维护
# KV 缓存/卷积状态/循环状态的缓存类;本文件把它们简化为空类,仅用于保持
# 与原始代码相同的书写结构(比如函数签名里引用这些类型名),
# 本文件内部实际并不依赖它们的任何具体字段或方法实现——调用方在使用
# Qwen3_5GatedDeltaNet 时,只需传入"鸭子类型"(duck-typing)满足下方
# forward() 里用到的 .has_previous_state / .conv_states[...] /
# .recurrent_states[...] 接口的对象即可。
# ---------------------------------------------------------------------------
class Qwen3_5Config:
    """占位配置类,不在本文件中实现任何字段,仅用于保持与上游代码结构一致。"""

    pass


class Qwen3_5DynamicCache:
    """占位缓存类,不在本文件中实现任何字段,仅用于保持与上游代码结构一致。"""

    pass


class Qwen3_5RMSNormGated(nn.Module):
    """门控 RMSNorm(Gated RMSNorm)。

    与标准 RMSNorm 的区别在于:归一化之后不是直接输出,而是再乘以一个
    经过 SiLU 激活的门控信号 `gate`,实现"先归一化、再按门控缩放"的组合,
    是 Mamba2 / GatedDeltaNet 等线性注意力结构中常见的输出归一化方式。

    参数:
        hidden_size (int): 归一化作用的最后一维大小(这里等于每个 value head
            的维度 head_v_dim)。
        eps (float): 数值稳定项,加在方差上防止除零。
    """

    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习缩放参数,形状 [hidden_size]
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        """前向计算。

        参数:
            hidden_states (Tensor): 形状 [..., hidden_size],通常是
                reshape 成 2D 的 [batch*seq*num_heads, head_v_dim]。
            gate (Tensor): 门控张量,形状与 hidden_states 相同,来自
                Qwen3_5GatedDeltaNet 中的 `z` 投影分支。
        返回:
            Tensor: 与 hidden_states 相同形状,dtype 还原为输入的 dtype。
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)  # 归一化过程使用 fp32 提升数值稳定性
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # 均方值(不减均值的"均方根"归一化)
        # Norm before gate
        # 中文:先做 RMSNorm(除以均方根),再套用门控,这是函数名 "NormGated" 的由来
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)  # 应用可学习权重,并转回原 dtype
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))  # 乘以 SiLU(gate) 门控(gate 先转 fp32 计算再参与乘法)

        return hidden_states.to(input_dtype)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66

    中文:把 padding token 位置的隐藏状态置零。

    动机:线性注意力/SSM 类模型(包括本文件的 GatedDeltaNet)会把序列信息
    压缩进一个跨时间步累积的"循环状态"里;如果 padding token 的隐藏状态
    不清零,它们的数值会污染卷积窗口和循环状态,导致有效 token 的计算结果
    被 padding 干扰(这一点与标准注意力不同——标准注意力可以直接靠
    attention mask 屏蔽掉 padding 的注意力权重,但循环状态类模型必须先把
    padding 位置的输入清零)。

    参数:
        hidden_states (Tensor): 形状 [batch_size, seq_len, hidden_size]。
        attention_mask (Tensor | None): 形状 [batch_size, seq_len] 的 2D
            布尔/0-1 张量,1 表示真实 token,0 表示 padding。
    返回:
        Tensor: 与 hidden_states 同形状;仅当满足下方条件时才真正做掩码乘法,
        否则原样返回。
    """
    # NOTE: attention mask is a 2D boolean tensor
    # 中文:只有当序列长度 > 1(处于一次性处理多个 token 的 prefill/训练阶段,
    # 而不是自回归解码时的单 token 输入)且 batch_size > 1(存在跨样本对齐
    # 产生的 padding,单个样本本身不需要 padding)时,才需要真正执行掩码乘法;
    # 这是历史版本 transformers(Mamba2/Qwen3-Next 系列)中的一处性能优化写法,
    # 在其他情况下（单 token 解码或 batch=1）跳过该乘法不会影响正确性。
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)  # 广播到 hidden_size 维,padding 位置乘 0

    return hidden_states


def torch_causal_conv1d_update(
        hidden_states,
        conv_state,
        weight,
        bias=None,
        activation=None,
):
    """纯 PyTorch 实现的"增量式因果卷积更新",用于自回归解码阶段。

    是 causal-conv1d 库中融合 CUDA kernel（`causal_conv1d_update`）的功能等价
    回退实现：每次只处理新到来的 `seq_len`（解码时通常为 1）个 token，同时
    利用/更新保存在缓存里的历史卷积窗口 `conv_state`，避免每步都重新对整段
    历史序列做卷积。

    参数:
        hidden_states (Tensor): 新的输入,形状 [batch, hidden_size, seq_len]
            (hidden_size 这里就是 QKV 混合后的 conv_dim 通道数)。
        conv_state (Tensor): 缓存中的历史卷积窗口状态,形状
            [batch, hidden_size, state_len]；函数内部会原地(in-place)更新它。
        weight (Tensor): 深度可分离卷积核权重,形状 [hidden_size, kernel_size]
            (从 nn.Conv1d.weight 的 [hidden_size, 1, kernel_size] squeeze 而来)。
        bias (Tensor | None): 卷积偏置,形状 [hidden_size] 或 None。
        activation (str | None): 未在本函数体中直接使用该字符串来查表——
            这里直接写死调用 F.silu（因为本笔记本场景激活函数固定是 silu，
            与顶部 `ACT2FN = {"silu": F.silu}` 的化简保持一致），仅保留参数
            位是为了和调用方签名一致。
    返回:
        Tensor: 卷积 + 激活后的输出,形状 [batch, hidden_size, seq_len],
        dtype 还原为 hidden_states 的原始 dtype。
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]  # 历史窗口长度(与写入时保持一致即可,不要求严格等于 kernel_size-1)

    # 拼接"历史窗口 + 新输入"作为这一次卷积的完整输入
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    # 用拼接后的末尾 state_len 个位置更新缓存里的卷积状态(滑动窗口前移),为下一步解码做准备
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    # padding=0 是因为历史上下文已经通过 conv_state 拼接在左侧提供了,不需要框架再自动补零
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    # 只截取对应"新输入"这 seq_len 个位置的卷积结果(前面属于历史部分的输出丢弃)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x, dim=-1, eps=1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library.

    中文:对张量在 `dim` 维上做 L2 归一化(除以其 L2 范数)。

    在 GatedDeltaNet 中用于对 query/key 做归一化,使其数值尺度稳定,
    并与 flash-linear-attention (FLA) 库里对应算子的数值行为对齐,便于将来
    切换到该库的融合 CUDA 实现时结果一致。

    参数:
        x (Tensor): 任意形状的输入张量。
        dim (int): 计算 L2 范数的维度,默认为最后一维。
        eps (float): 防止除零的数值稳定项。
    返回:
        Tensor: 与 x 形状相同,该维度上的 L2 范数近似为 1。
    """
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)  # 1 / sqrt(sum(x^2) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
):
    """门控增量法则(Gated Delta Rule)的"分块并行"纯 PyTorch 实现。

    这是 DeltaNet / Gated DeltaNet 线性注意力的核心算法:把序列切成若干个
    大小为 chunk_size 的块,块内用矩阵运算并行求解(通过对一个下三角矩阵做
    前向替换求逆的"UT 变换"技巧,一次性把块内本应顺序执行的增量法则更新
    合并成矩阵乘法),块间仍然按时间顺序遍历(因为跨块的循环状态存在依赖)。
    数学上,其结果与逐 token 顺序执行 `torch_recurrent_gated_delta_rule`
    完全等价,但由于块内做了并行化,训练时效率更高。

    参数:
        query (Tensor): 形状 [batch, seq_len, num_heads, k_head_dim]。
        key (Tensor): 形状 [batch, seq_len, num_heads, k_head_dim]。
        value (Tensor): 形状 [batch, seq_len, num_heads, v_head_dim]。
        g (Tensor): 对数空间的衰减门控,形状 [batch, seq_len, num_heads]；
            数值应 <= 0(exp(g) 是每步施加在循环状态上的"遗忘系数",越负遗忘越快)。
        beta (Tensor): "写入强度"/更新门控,形状 [batch, seq_len, num_heads],
            取值范围 (0, 1)(来自 sigmoid),类似增量法则里的学习率。
        chunk_size (int): 分块大小,默认为 64。
        initial_state (Tensor | None): 可选的初始循环状态,形状
            [batch, num_heads, k_head_dim, v_head_dim];为 None 时从全零状态开始。
        output_final_state (bool): 是否返回处理完整段序列后的最终循环状态
            (用于保存进 KV 缓存,供后续增量解码使用)。
        use_qk_l2norm_in_kernel (bool): 是否在核内对 q/k 做 L2 归一化。
    返回:
        tuple[Tensor, Tensor | None]:
            - core_attn_out: 形状 [batch, seq_len, num_heads, v_head_dim]。
            - last_recurrent_state: 形状同 initial_state,若
              output_final_state 为 False 则为 None。
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # 统一转成 [batch, num_heads, seq_len, dim] 布局,并转 fp32 提升数值稳定性
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    # 把序列长度 pad 到 chunk_size 的整数倍,方便后面 reshape 成 [..., n_chunks, chunk_size, dim]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)  # 标准注意力式的 1/sqrt(d) 缩放
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)  # 按 beta 加权后的 value,即"待写入状态的量"
    k_beta = key * beta.unsqueeze(-1)  # 按 beta 加权后的 key
    # reshape to chunks
    # 中文:把序列维拆分成 [n_chunks, chunk_size] 两维,方便块内并行计算
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    # 块内(chunk_size x chunk_size)上三角掩码(含对角线),用于后面把"非因果"部分置零
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    # 中文:在对数空间对衰减门控做块内累加(cumsum),得到"块起点到位置 t 的累积衰减量"
    g = g.cumsum(dim=-1)
    # decay_mask[i, j] = exp(cumsum_i - cumsum_j),表示位置 j 到位置 i 之间累积的衰减系数;
    # 先 tril 保留因果(j<=i)部分再 exp,避免非因果位置的差值在 exp 后爆炸,最后再 tril 一次做兜底
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    # 构造 UT 变换所需的"待求逆"矩阵:先算块内 k_beta·key^T 并乘上衰减,再取严格下三角(masked_fill 上三角+对角线为 0),
    # 取负号是因为下面的前向替换是在求解 (I - L)^{-1} 中的 L(严格下三角部分)
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    # 前向替换(forward substitution),逐行把已经"展开"的部分累加进来,
    # 等价于计算严格下三角矩阵 L 的 (I - L)^{-1} 展开式,一次性把块内本该逐 token
    # 顺序执行的增量更新合并为矩阵形式(即"UT 变换"技巧的核心循环)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)  # 加上单位矩阵,得到完整的 (I-L)^{-1}
    value = attn @ v_beta  # 块内"修正后"的新值(对应 DeltaNet 论文中的 u,已经把块内顺序依赖消解掉)
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))  # 块内衰减调整后的 key,后面用于从上一块的循环状态里做"预测"
    # 初始化循环状态:没有 initial_state 则从全零开始,否则沿用调用方传入的状态(用于增量解码/续写)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    # 块内严格上三角掩码(不含对角线),用于屏蔽块内"未来看到未来"的非因果注意力
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    # 中文:块间必须按时间顺序遍历,因为每一块都依赖前一块结束时的循环状态
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        # 块内因果注意力(带衰减权重),并屏蔽掉块内"未来"位置
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state  # 用上一块结束时的状态"预测"出的值
        v_new = v_i - v_prime  # 真实值与预测值的残差(delta rule 的"新息"),即真正需要写入状态的信息
        # 读取上一块状态对当前块 query 的贡献(类似线性注意力里的"记忆读取"项)
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        # 当前块输出 = 跨块记忆读取 + 块内(因果、含衰减)注意力对残差 v_new 的加权求和
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        # 更新循环状态:先按本块末尾的累积衰减系数对旧状态做衰减(遗忘),
        # 再叠加本块 key(经过跨块衰减调整)与残差 v_new 的外积,作为新的状态写入(delta rule 更新)
        last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None  # 调用方不需要缓存状态时(比如没有 KV cache)直接丢弃,节省显存
    # 把 [batch, num_heads, n_chunks, chunk_size, v_head_dim] 拉平回 [batch, num_heads, total_seq_len, v_head_dim]
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]  # 去掉之前为了对齐 chunk_size 而补的 padding
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)  # 转回 [batch, seq_len, num_heads, v_head_dim] 及原始 dtype
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
        query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    """门控增量法则(Gated Delta Rule)的"逐 token 递归"纯 PyTorch 实现。

    与 `torch_chunk_gated_delta_rule` 数学上完全等价,区别在于本函数严格
    按时间步一步步顺序计算(不做分块并行),因此更适合自回归解码阶段
    "每步只新增 1 个 token"的场景——此时分块并行本就没有优势,顺序实现
    反而更直接、开销更低。

    参数:
        query (Tensor): 形状 [batch, seq_len, num_heads, k_head_dim]
            (解码时 seq_len 通常为 1)。
        key (Tensor): 形状 [batch, seq_len, num_heads, k_head_dim]。
        value (Tensor): 形状 [batch, seq_len, num_heads, v_head_dim]。
        g (Tensor): 对数空间的衰减门控,形状 [batch, seq_len, num_heads]。
        beta (Tensor): 写入强度门控,形状 [batch, seq_len, num_heads]。
        initial_state (Tensor | None): 上一步的循环状态,形状
            [batch, num_heads, k_head_dim, v_head_dim];为 None 时从全零状态开始。
        output_final_state (bool): 是否返回处理完这段序列后的最终循环状态。
        use_qk_l2norm_in_kernel (bool): 是否对 q/k 做 L2 归一化。
    返回:
        tuple[Tensor, Tensor | None]:
            - core_attn_out: 形状 [batch, seq_len, num_heads, v_head_dim]。
            - last_recurrent_state: 形状同 initial_state,若
              output_final_state 为 False 则为 None。
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # 统一转成 [batch, num_heads, seq_len, dim] 布局,并转 fp32 提升数值稳定性
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)  # 1/sqrt(d) 缩放,与分块版本一致
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    # 初始化循环状态(全零或沿用传入的 initial_state,例如上一次解码步保存的状态)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    # 逐 token 顺序扫描(RNN 式递归),sequence_length 在解码时通常为 1
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # 当前步的衰减系数(exp 回到线性空间),形状可广播到状态矩阵
        beta_t = beta[:, :, i].unsqueeze(-1)  # 当前步的写入强度

        last_recurrent_state = last_recurrent_state * g_t  # 状态先按衰减系数"遗忘"
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)  # 用当前 key 从(已衰减)状态中读出"预测值"
        delta = (v_t - kv_mem) * beta_t  # 真实值与预测值的残差,乘以 beta 得到"实际写入量"(delta rule 核心)
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # 用外积把残差写回状态
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)  # 用当前 query 从最新状态中读出输出

    if not output_final_state:
        last_recurrent_state = None  # 不需要缓存状态时直接丢弃
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)  # 转回 [batch, seq_len, num_heads, v_head_dim]
    return core_attn_out, last_recurrent_state


# Minimal change: enforce config dtype at the end to avoid bf16/fp32 matmul mismatch
# in a mixed notebook implementation
# 中文:上面这条英文注释是原作者标注的"本文件相对上游的最小改动"——
# 在 __init__ 末尾强制把所有子模块转换到 config.dtype,避免笔记本环境里
# 因为某些张量是 bf16、某些是 fp32 而导致矩阵乘法报 dtype 不匹配的错误。
# 该改动本身不改变模型数值语义,只是统一了权重的存储精度。
class Qwen3_5GatedDeltaNet(nn.Module):
    """Qwen3.5(Qwen3-Next 系列)中的"门控 DeltaNet"线性注意力层。

    整体结构:
        输入 hidden_states
          -> 4 个独立线性投影,分别得到 QKV 混合向量、门控 z、beta(b)、
             衰减相关输入 a
          -> QKV 混合向量经过深度可分离的短因果卷积(short causal conv)
             做局部混合,并（在解码阶段）与 KV 缓存中的卷积历史状态交互
          -> 拆分出 query / key / value,并按需做 GQA 风格的头维度复制
          -> 送入门控增量法则(分块并行版或逐 token 递归版,取决于是否处于
             KV 缓存增量解码阶段),压缩进/读取一个固定大小的循环状态
          -> 用 Gated RMSNorm 结合门控 z 做输出归一化
          -> 最终线性投影回 hidden_size

    参数:
        config: 模型配置对象,需要提供 hidden_size、
            linear_num_value_heads、linear_num_key_heads、
            linear_key_head_dim、linear_value_head_dim、
            linear_conv_kernel_dim、hidden_act、rms_norm_eps、dtype 等字段
            (本文件中用占位类 Qwen3_5Config 表示,实际使用时需传入真实配置对象)。
        layer_idx (int): 当前层在整个模型中的索引,用于在 KV 缓存里定位
            属于本层的卷积状态/循环状态。
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads  # value 头数
        self.num_k_heads = config.linear_num_key_heads  # key/query 头数(可以少于 value 头数,类似 GQA)
        self.head_k_dim = config.linear_key_head_dim  # 每个 key/query 头的维度
        self.head_v_dim = config.linear_value_head_dim  # 每个 value 头的维度
        self.key_dim = self.head_k_dim * self.num_k_heads  # key/query 总维度
        self.value_dim = self.head_v_dim * self.num_v_heads  # value 总维度

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        # 风险/说明(不修改代码):self.act 在本类的 forward 中从未被实际调用
        # (卷积激活是在 torch_causal_conv1d_update 里直接写死用 F.silu 完成的)。
        # 经与上游 transformers 源码比对,这是上游原始实现里就存在的"死代码"，
        # 并非本文件在拷贝/notebook 改写过程中引入的问题，故按规则只标注、不清理。
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        # 中文:QKV 混合投影的总通道数 = 2 份 key 维度(q 和 k 各占一份)+ 1 份 value 维度
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,  # groups == in_channels,即深度可分离(depthwise)卷积,每个通道独立卷积不混合
            padding=self.conv_kernel_size - 1,  # 左侧因果 padding,保证卷积核只看到当前及历史位置(输出后需裁剪右侧多出的部分)
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        # 中文:dt_bias 对应 Mamba/S4 风格状态空间模型里"时间步长离散化"的偏置项,
        # 会在真实的 PretrainedModel.init_weights 中被专门初始化(这里先占位为全 1)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        # 风险提示(不修改代码):这里从 Uniform(0, 16) 采样再取 log 得到 A_log。
        # HuggingFace 官方仓库在更晚的版本中，已将下界从 0 改为 0.01（并附注释
        # "Lower bound kept away from 0 so log(A) never becomes -inf"），
        # 用来避免极小概率采样到 0 导致 log(0) = -inf。
        # 但这是一次性的随机权重初始化，而非可稳定复现的确定性 bug（命中概率
        # 极低），且此写法与本文件所忠实复制的那一版上游代码完全一致，因此按
        # "风险项不改只标注"的原则保留，不在本文件中擅自修改。
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))  # log(A),配合 forward 中的 -exp(A_log) 得到恒为负的衰减系数

        self.norm = (
            Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
            if FusedRMSNormGated is None
            else FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
            )
        )
        # 中文:由于本文件顶部把 FusedRMSNormGated 固定设为 None,上面的分支
        # 恒定会走 Qwen3_5RMSNormGated(纯 PyTorch 实现),else 分支实际不可达,
        # 保留是为了与真实 transformers 代码结构一致(该库安装了融合算子时会用到)。

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # 中文:下面几行把"融合 CUDA 实现优先、否则退回纯 PyTorch 实现"的选择
        # 保存成实例属性,方便 forward 中统一调用；由于模块顶部相关全局变量
        # 均为 None,这里实际总是选中 torch_xxx 系列纯 PyTorch 函数。
        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of the required library is not installed. Falling back to "
                "torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

        # 中文:与部分上游版本(把 QKV/门控合并成一到两个大投影再 split)不同,
        # 本文件把 4 个投影拆成 4 个独立的 nn.Linear,分别得到 QKV 混合向量、
        # 门控 z、beta 的输入 b、衰减相关输入 a,逻辑等价但代码更直观。
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)  # 一次性投影出 q、k、v 拼接向量
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)  # 门控分支 z,供 Gated RMSNorm 使用
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)  # beta(写入强度)的输入,每个 value head 一个标量
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)  # 衰减系数 g 的输入 a,每个 value head 一个标量

        # Notebook adaptation for dtype consistency.
        # 中文:笔记本专用改写——如果配置里指定了 dtype(例如 bf16),
        # 就把本模块的所有参数/子模块统一转换成该 dtype,避免和外部混合精度
        # 环境下的其他张量做矩阵乘法时出现 dtype 不一致的报错(见文件顶部说明)。
        if config.dtype is not None:
            self.to(dtype=config.dtype)

    def forward(
            self,
            hidden_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
    ):
        """前向计算。

        支持两种模式:
            1) 训练 / prefill(一次性处理多个 token,或没有 KV 缓存):
               走分块并行的 `chunk_gated_delta_rule`。
            2) 自回归解码且已有历史缓存(seq_len == 1 且 cache_params 有
               上一步保存的状态):走增量式的 `causal_conv1d_update` +
               逐 token 的 `recurrent_gated_delta_rule`,只处理新来的 1 个
               token,并原地更新缓存中的卷积状态/循环状态。

        参数:
            hidden_states (Tensor): 形状 [batch_size, seq_len, hidden_size]。
            cache_params (Qwen3_5DynamicCache | None): 缓存对象,需提供
                `has_previous_state`(bool)、`conv_states`(按 layer_idx 索引
                的卷积状态列表)、`recurrent_states`(按 layer_idx 索引的
                循环状态列表)等属性;为 None 表示不使用/不更新缓存。
            cache_position (Tensor | None): 当前处理的 token 在整段序列中的
                位置索引,仅用于判断是否处于"已有历史、且只新增 1 个 token"
                的增量解码场景。
            attention_mask (Tensor | None): 形状 [batch_size, seq_len] 的
                2D padding mask。
        返回:
            Tensor: 形状 [batch_size, seq_len, hidden_size],与输入
            hidden_states 形状一致。
        """
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)  # 先清零 padding 位置,避免污染卷积/状态

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        # 判断是否处于"已有历史缓存 + 本次只新增 1 个 token"的增量解码模式
        use_precomputed_states = (
                cache_params is not None
                and cache_params.has_previous_state
                and seq_len == 1
                and cache_position is not None
        )

        # getting projected states from cache if it exists
        # 中文:若存在缓存,先取出属于本层(layer_idx)的历史卷积状态/循环状态,
        # 供下面按需使用(增量解码分支会用到,prefill 分支会在算完之后写入新的状态)
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states)  # [batch, seq_len, key_dim*2+value_dim]
        mixed_qkv = mixed_qkv.transpose(1, 2)  # 转成 [batch, conv_dim, seq_len],适配 Conv1d 的通道维在前的约定

        z = self.in_proj_z(hidden_states)  # [batch, seq_len, value_dim]
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)  # [batch, seq_len, num_v_heads, head_v_dim]

        b = self.in_proj_b(hidden_states)  # [batch, seq_len, num_v_heads],beta 的原始输入
        a = self.in_proj_a(hidden_states)  # [batch, seq_len, num_v_heads],衰减系数 g 的原始输入

        if use_precomputed_states:
            # 2. Convolution sequence transformation
            # NOTE: the conv state is updated in `causal_conv1d_update`
            # 中文:增量解码分支——只处理新来的 1 个 token,卷积状态在函数内部原地更新
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),  # [conv_dim, 1, kernel_size] -> [conv_dim, kernel_size]
                self.conv1d.bias,
                self.activation,
            )
        else:
            # 中文:prefill / 训练分支——一次性对整段序列做因果卷积
            if cache_params is not None:
                # 用当前这段序列的末尾部分构造出"下一次增量解码"所需的初始卷积状态,
                # 若序列长度不足 conv_kernel_size 则左侧补零；写入缓存供下一步解码使用
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                # 走可选的融合 CUDA 实现(本文件中恒为 None,不会执行到这里)
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                # 纯 PyTorch 回退:nn.Conv1d 自带 padding=kernel_size-1 做因果左侧补零,
                # 卷积后再截取前 seq_len 个位置(丢弃右侧因 padding 多出的部分),最后接 SiLU 激活
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)  # 转回 [batch, seq_len, conv_dim]
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )  # 按通道维拆分出 query、key、value

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)  # [batch, seq_len, num_k_heads, head_k_dim]
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)  # [batch, seq_len, num_k_heads, head_k_dim]
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)  # [batch, seq_len, num_v_heads, head_v_dim]

        beta = b.sigmoid()  # 写入强度门控,映射到 (0, 1) 区间
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        # 中文:g = -exp(A_log) * softplus(a + dt_bias),恒为 <=0 的对数空间衰减系数;
        # 显式 .float() 是为了避免 fp16 精度下 exp(A_log) 数值上溢/下溢导致 -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            # 中文:类似 GQA(分组查询注意力),当 value 头数多于 key/query 头数时,
            # 把 query/key 沿头维重复,使其头数与 value 对齐,才能逐头做后续矩阵运算
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            # prefill / 训练:走分块并行版门控增量法则,initial_state 为 None(从头开始累积状态)
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,  # 只有存在缓存时才需要拿到并保存最终状态
                use_qk_l2norm_in_kernel=True,
            )

        else:
            # 增量解码:走逐 token 递归版门控增量法则,initial_state 为上一步缓存中的循环状态
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        # Update cache
        # 中文:把这一步算出的最新循环状态写回缓存,供下一次解码步使用
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        # reshape input data into 2D tensor
        # 中文:把 core_attn_out 和门控 z 都拉平成 2D [batch*seq_len*num_v_heads, head_v_dim],
        # 以匹配 Qwen3_5RMSNormGated.forward 期望的输入形状
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)  # 门控 RMSNorm:归一化后按 SiLU(z) 缩放
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)  # 还原成 [batch, seq_len, value_dim]

        output = self.out_proj(core_attn_out)  # 线性投影回 hidden_size
        return output
#%%
