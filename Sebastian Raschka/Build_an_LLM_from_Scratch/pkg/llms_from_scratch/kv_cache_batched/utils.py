# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文说明：
本文件定义了支持“批量（batched）”推理场景下使用的 KV cache 容器类 `KVCache`。

与非批量版本（每条样本单独维护一个全局 KV cache，通常直接把各层的
(key, value) 张量存成一个列表）不同，这里的 `KVCache` 内部用一个二维列表
按 [层数 n_layers, 批次大小 batch_size] 组织缓存：
    self.cache[layer_idx][batch_idx]
即“第 layer_idx 层、第 batch_idx 个样本”各自独立保存自己的 (key, value)
缓存张量（具体张量类型/形状由调用方决定，通常是一个二元组
`(k, v)`，其中 k、v 形状为 `(1, num_kv_groups, seq_len_so_far, head_dim)`）。

这种“按样本拆分存储”的设计使得同一批次中不同样本可以拥有长度不同、
增长速度不同的历史缓存（例如批次内每条 prompt 长度不一致，或者不同样本
提前结束生成），从而配合 `generate.py` 中按样本维护的 `current_pos`
（形状 `(batch_size,)`）一起，实现“变长批量生成”场景下的 KV cache 复用。

以下代码在原始英文注释基础上，新增了详细的中文注释，未改动任何可执行逻辑。
"""

class KVCache:
    """
    批量 KV 缓存容器。

    内部维护一个形状概念上为 [n_layers, batch_size] 的二维列表
    `self.cache`，其中每个元素 `self.cache[layer_idx][batch_idx]`
    对应“第 layer_idx 个 Transformer 层、批次中第 batch_idx 个样本”
    的历史 (key, value) 缓存。初始化时所有位置均为 `None`，表示
    该层、该样本尚未写入任何历史 key/value（即还没有做过前向传播）。

    该类本身不关心缓存里存的具体张量形状/类型，只负责按
    (layer_idx, batch_idx) 二维索引做“存、取、按层批量取、整体清空”
    这几种操作，具体的张量拼接逻辑（例如沿序列维度 cat 新旧 key/value）
    由调用方（如 qwen3.py 中的 `GroupedQueryAttention.forward`）完成。
    """

    def __init__(self, n_layers, batch_size):
        """
        初始化一个空的批量 KV 缓存。

        参数：
            n_layers (int):
                Transformer 模型的层数，决定 `self.cache` 外层列表的长度。
            batch_size (int):
                批次大小，决定 `self.cache` 内层列表的长度，即每一层
                需要为批次中多少个样本各自维护一份独立的 KV 缓存。

        效果：
            构造出 `self.cache`，形状（概念上）为 `[n_layers][batch_size]`，
            所有位置初始值均为 `None`，表示尚未缓存任何 key/value。
        """
        self.cache = [
            # 对每一层（共 n_layers 层），创建一个长度为 batch_size 的列表，
            # 列表中每个位置先占位为 None，等待后续 update() 写入真实的
            # (key, value) 缓存张量。
            [None for _ in range(batch_size)] for _ in range(n_layers)
        ]

    def get(self, layer_idx, batch_idx):
        """
        读取指定层、指定批次样本当前已缓存的 (key, value)。

        参数：
            layer_idx (int): 目标 Transformer 层的索引（从 0 开始）。
            batch_idx (int): 目标批次样本在批次内的索引（从 0 开始）。

        返回：
            该位置当前缓存的内容；若尚未写入过，则为 `None`；
            否则通常是一个 `(k, v)` 二元组，k、v 的形状一般为
            `(1, num_kv_groups, seq_len_so_far, head_dim)`，
            表示该样本截至目前为止在该层累积的历史 key/value。
        """
        return self.cache[layer_idx][batch_idx]

    def update(self, layer_idx, batch_idx, value):
        """
        写入（覆盖）指定层、指定批次样本的缓存内容。

        参数：
            layer_idx (int): 目标 Transformer 层的索引。
            batch_idx (int): 目标批次样本在批次内的索引。
            value:
                新的缓存内容，通常是调用方已经把“历史缓存”与
                “本步新计算出的 key/value”沿序列维度拼接（cat）之后
                得到的最新 `(k, v)` 二元组，用来整体替换旧值，
                从而实现“缓存随生成步数不断增长”的效果。

        返回：
            无返回值；直接原地修改 `self.cache[layer_idx][batch_idx]`。
        """
        self.cache[layer_idx][batch_idx] = value

    def get_layer(self, layer_idx):
        """
        取出某一层在整个批次上的缓存列表。

        参数：
            layer_idx (int): 目标 Transformer 层的索引。

        返回：
            长度为 batch_size 的列表，其中第 i 个元素即批次内第 i 个
            样本在该层的缓存内容（可能是 `None` 或 `(k, v)` 二元组）。
            该方法常用于需要“一次性拿到某一层所有样本缓存”的场景，
            例如按层遍历、批量拼接等。
        """
        return self.cache[layer_idx]

    def reset(self):
        """
        清空整个 KV 缓存。

        效果：
            遍历 `self.cache` 中的每一层（layer），再遍历该层内每个
            批次样本的位置，将其重置为 `None`，即把所有已缓存的
            历史 key/value 全部丢弃。通常在开始处理新的一批
            （新的 batch）序列、或需要重新从头生成时调用，
            以避免复用上一次生成残留的历史缓存。
        """
        for layer in self.cache:
            for i in range(len(layer)):
                # 将该层中每个批次样本的缓存位置重新置为 None，
                # 相当于“忘记”该样本在这一层此前累积的所有历史 key/value。
                layer[i] = None
