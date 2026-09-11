"""
本文件中文说明：
本测试文件属于《从零构建大语言模型》（Build a Large Language Model From Scratch）一书
第 3 章附赠内容（02_bonus_efficient-multihead-attention）的一部分。

该 bonus 章节展示了多种"多头注意力（Multi-Head Attention, MHA）"的等价实现方式，
包括：
    - Ch03_MHA：第 3 章正文中最基础、最直观的多头注意力实现（使用 nn.Linear）。
    - MHAEinsum：使用 einsum（爱因斯坦求和约定）改写的等价实现，通常用于教学演示
      矩阵乘法背后的下标运算，也可能带来一定的性能差异。
    - MHAPyTorchSDPAWithoutFlash：基于 PyTorch 官方
      `torch.nn.functional.scaled_dot_product_attention`（SDPA）实现的版本，
      但显式关闭了 Flash Attention 加速核（"WithoutFlash"），以便对比/调试。

这些不同实现都定义在同目录下的 Jupyter Notebook 文件 `mha-implementations.ipynb` 中，
本测试文件通过工具函数 `import_definitions_from_notebook` 把 notebook 里的类定义
动态导入为一个 Python 模块对象，然后针对这些实现做「一致性测试」：
    1) 验证不同写法（Linear 版 vs einsum 版）在权重相同的情况下，输出是否数值一致；
    2) 验证因果注意力（causal attention，即只能看到当前及之前的 token，看不到未来的 token）
       的掩码是否生效——修改未来位置的输入，不应该影响当前及之前位置的输出。

这类测试在学习 Transformer/注意力机制时非常有价值：它们用可运行的代码证明了
"数学上等价的不同实现，数值结果应当相同"，以及"因果掩码保证了自回归语言模型
不会'偷看'未来信息"这一核心性质。
"""

from pathlib import Path
import torch
import pytest


from llms_from_scratch.utils import import_definitions_from_notebook


@pytest.fixture
def import_notebook_defs():
    """
    Pytest fixture（测试夹具）：动态导入 notebook 中定义的多头注意力实现类。

    作用：
        - 定位到当前测试文件所在目录的上一级目录（即 02_bonus_efficient-multihead-attention），
          因为 `mha-implementations.ipynb` 就存放在那里；
        - 调用 `import_definitions_from_notebook` 工具函数，把该 notebook 中定义的
          所有类（如 Ch03_MHA、MHAEinsum、MHAPyTorchSDPAWithoutFlash 等）
          当作一个普通 Python 模块加载进来，这样测试函数就可以像
          `import_notebook_defs.Ch03_MHA` 这样直接使用 notebook 里定义的类。

    返回值：
        mod: 一个动态构造的模块对象，包含 notebook 中定义的所有类/函数，
             供后续测试函数调用。
    """
    # 定位到 notebook 所在目录：parents[1] 表示从当前测试文件往上两级目录
    # （tests/ 的上一级），即 02_bonus_efficient-multihead-attention/
    nb_dir = Path(__file__).resolve().parents[1]
    # 将 notebook 文件中的代码定义（类、函数）提取出来，构造成一个可导入使用的模块
    mod = import_definitions_from_notebook(nb_dir, "mha-implementations.ipynb")
    return mod


def copy_weights(from_mha, to_mha):
    """
    在两种不同实现的多头注意力模块之间，把权重从一个模型拷贝到另一个模型。

    目的：
        Ch03_MHA（基础 Linear 实现）和 MHAEinsum（einsum 实现）在数学上应当完全等价，
        但它们各自初始化时使用的是随机权重，直接比较输出没有意义。
        因此需要先把 `from_mha`（源模型，Ch03_MHA）的权重复制到 `to_mha`（目标模型，MHAEinsum），
        确保两者使用完全相同的参数，这样才能验证"实现方式不同、但计算结果是否一致"。

    参数：
        from_mha: 权重来源模型（此处为 Ch03_MHA 实例，内部使用 nn.Linear 存储 W_query/W_key/W_value）。
        to_mha:   权重写入目标模型（此处为 MHAEinsum 实例，内部使用普通 nn.Parameter 张量存储权重）。

    返回值：
        无返回值（原地修改 to_mha 的参数）。
    """
    # 使用 torch.no_grad() 上下文，避免在权重拷贝过程中被自动求导追踪，
    # 因为这只是参数初始化操作，不需要计算梯度。
    with torch.no_grad():
        # nn.Linear 的权重矩阵形状为 (out_features, in_features)，
        # 而 einsum 实现里习惯把权重存成 (in_features, out_features)，
        # 所以这里需要对权重做转置 .T 后再拷贝，以保证数学上等价。
        to_mha.W_query.copy_(from_mha.W_query.weight.T)
        to_mha.W_key.copy_(from_mha.W_key.weight.T)
        to_mha.W_value.copy_(from_mha.W_value.weight.T)

        # 输出投影层（out_proj）两边都是 nn.Linear，形状一致，直接拷贝权重和偏置即可
        to_mha.out_proj.weight.copy_(from_mha.out_proj.weight)
        to_mha.out_proj.bias.copy_(from_mha.out_proj.bias)


@pytest.mark.parametrize(
    "d_in,d_out,batch,seq_len,num_heads,seed",
    [
        (768, 768, 2, 4, 12, 123),  # d_in == d_out
        (768, 1536, 2, 4, 12, 456),  # d_in != d_out
        (1024, 512, 2, 4, 8, 789),   # d_in > d_out
    ],
)
def test_mha_einsum_matches_ch03(d_in, d_out, batch, seq_len, num_heads, seed, import_notebook_defs):
    """
    一致性测试：验证「基础 Linear 实现（Ch03_MHA）」与「einsum 实现（MHAEinsum）」
    在权重完全相同的前提下，前向传播输出是否数值一致。

    该测试使用 pytest.mark.parametrize 参数化，覆盖三种典型场景：
        - d_in == d_out：输入输出维度相同（最常见的 Transformer block 内部情形）；
        - d_in != d_out（d_out 更大）：验证升维场景；
        - d_in > d_out：验证降维场景。
    这样可以确保两种实现在不同维度配置下都能保持等价，而不仅仅是在某一种特殊情况下"恰好"相等。

    参数：
        d_in (int): 输入嵌入维度（embedding dimension）。
        d_out (int): 输出维度，即多头注意力总的输出通道数（等于 num_heads * head_dim）。
        batch (int): 批大小 batch size。
        seq_len (int): 序列长度（token 数量），本测试中同时也用作 context_length。
        num_heads (int): 注意力头的数量。
        seed (int): 随机种子，保证每组参数化用例的输入/权重初始化可复现。
        import_notebook_defs: 上面定义的 fixture，提供从 notebook 中导入的各实现类。

    核心断言：
        - 两个模型输出的形状均为 (batch, seq_len, d_out)；
        - 两个模型输出的数值在容差 atol=1e-5 内一致（torch.allclose）。
    """
    # 固定随机种子，确保输入张量 x 以及后续模型初始化的可复现性
    torch.manual_seed(seed)

    # 构造随机输入张量，形状为 (batch, seq_len, d_in)，模拟一批文本序列的词嵌入
    x = torch.randn(batch, seq_len, d_in)

    # 实例化第 3 章正文中的基础多头注意力实现（使用 nn.Linear 做 Q/K/V 投影）
    mha_linear = import_notebook_defs.Ch03_MHA(
        d_in=d_in,
        d_out=d_out,
        context_length=seq_len,
        dropout=0.0,  # 测试阶段关闭 dropout，保证结果确定性、可比较
        num_heads=num_heads,
        qkv_bias=False,  # 不使用偏置项，简化对比
    ).eval()  # 切换到 eval 模式，避免 dropout/batchnorm 等训练态行为影响结果

    # 实例化使用 einsum 改写的等价多头注意力实现，超参数保持一致
    mha_einsum = import_notebook_defs.MHAEinsum(
        d_in=d_in,
        d_out=d_out,
        context_length=seq_len,
        dropout=0.0,
        num_heads=num_heads,
        qkv_bias=False,
    ).eval()

    # 关键步骤：把 mha_linear 的权重复制给 mha_einsum，
    # 确保两个模型在计算时使用完全相同的参数，这样才能公平比较"实现方式"本身的差异
    copy_weights(mha_linear, mha_einsum)

    # 分别对同一输入 x 做前向传播
    out_linear = mha_linear(x)
    out_einsum = mha_einsum(x)

    # 断言1：两种实现输出的张量形状必须一致，且等于预期的 (batch, seq_len, d_out)
    assert out_linear.shape == out_einsum.shape == torch.Size([batch, seq_len, d_out])
    # 断言2：两种实现在数值上应当（近似）完全相等，证明 einsum 写法与 Linear 写法在数学上等价
    assert torch.allclose(out_linear, out_einsum, atol=1e-5)


def test_sdpa_without_flash_does_not_attend_to_future_tokens(import_notebook_defs):
    """
    因果掩码（causal mask）有效性测试：验证基于 PyTorch SDPA（scaled_dot_product_attention）
    但关闭 Flash Attention 加速核的多头注意力实现（MHAPyTorchSDPAWithoutFlash），
    是否严格遵守"只能看到当前及之前位置，看不到未来位置"这一自回归语言模型的核心约束。

    测试思路：
        构造两份输入序列，二者仅在"未来"的 token（序列中靠后的位置，索引从 2 开始）上有差异
        （其中一份把未来位置的值加上了一个很大的偏移量 +10）。
        如果因果掩码工作正常，那么模型在处理"当前及更早"位置（索引 0、1）时，
        其输出不应当受到"未来"位置（索引 2、3）取值变化的影响——
        因为在自回归/因果注意力中，位置 i 的输出只能依赖于位置 <= i 的信息，
        不能"偷看"位置 > i 的信息，否则会造成训练/推理时的信息泄漏（information leakage）。

    参数：
        import_notebook_defs: fixture，提供从 notebook 导入的多头注意力实现类。

    核心断言：
        out[:, :2] 与 out_with_changed_future[:, :2] 应当在容差范围内完全相同，
        即前两个位置（未来变化之前的部分）的输出不受"未来"扰动的影响。
    """
    # 固定随机种子，保证模型权重初始化和输入的可复现性
    torch.manual_seed(123)

    # 实例化基于 SDPA、但显式关闭 Flash Attention 内核的多头注意力实现。
    # "WithoutFlash" 通常意味着使用的是标准的数学实现路径（如 math 或 mem-efficient 后端），
    # 便于在没有 GPU / 不支持 Flash Attention 硬件上做对照测试，同时也用于验证因果掩码逻辑本身的正确性。
    model = import_notebook_defs.MHAPyTorchSDPAWithoutFlash(
        d_in=12,
        d_out=12,
        context_length=4,  # 序列长度为 4，对应下面构造的 seq_len=4 的输入
        dropout=0.0,
        num_heads=3,
        qkv_bias=False,
    ).eval()

    # 构造形状为 (batch=1, seq_len=4, d_in=12) 的原始输入
    x = torch.randn(1, 4, 12)
    # 克隆一份输入，用于制造"未来 token 发生变化"的对照样本，避免直接修改原始 x
    x_with_changed_future = x.clone()
    # 只修改序列中索引 2 及之后（即位置 2、3，代表"未来"的 token）的数值，
    # 加上一个很大的偏移量 +10，制造明显的扰动，便于检测因果掩码是否真正生效
    x_with_changed_future[:, 2:] += 10

    # 对两份输入分别做前向传播，得到注意力输出
    out = model(x)
    out_with_changed_future = model(x_with_changed_future)

    # 核心断言：如果因果掩码正确实现，位置 0 和位置 1（即 [:, :2]）的输出
    # 不应当受到位置 2、3（"未来"）数值变化的影响，因为在因果注意力中，
    # 早期位置的 query 在计算注意力权重时，其对应的 key/value 只能来自不晚于自身的位置，
    # 未来位置的改动会被掩码（mask）屏蔽掉，不会传播到过去位置的输出中。
    assert torch.allclose(out[:, :2], out_with_changed_future[:, :2], atol=1e-5)
