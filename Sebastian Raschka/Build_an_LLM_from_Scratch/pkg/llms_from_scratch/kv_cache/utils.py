# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块提供 KV Cache（键值缓存）相关的通用工具类。

在自回归（autoregressive）文本生成过程中，每生成一个新 token，Transformer
的每一层注意力都需要用到「历史所有 token」的 Key（K）和 Value（V）张量。
如果不做缓存，每生成一个新 token 就要把之前所有 token 重新过一遍模型来
计算 K、V，计算量会随生成长度呈平方级增长，非常浪费。

KV Cache 的思路是：把每一层已经算好的 K、V 张量缓存下来，生成下一个 token
时只需要计算「新 token」自己的 K、V，然后与缓存中的历史 K、V 拼接（concat）
在一起，从而把每步的计算复杂度从 O(seq_len^2) 降低到 O(seq_len)（均摊）。

本文件中的 `KVCache` 类就是一个非常简单的容器，用一个 Python list 按层
（layer）存放每一层的缓存内容（通常是形如 (K, V) 的元组，K/V 的典型形状为
[batch_size, num_heads, seq_len_so_far, head_dim]），并提供获取、更新、
批量获取以及重置（清空，用于开始处理新的一批序列时）等操作。
"""


class KVCache:
    """KV 缓存容器类。

    按 Transformer 的层数（n_layers）维护一个列表，列表的每个位置对应
    一层的 KV 缓存。缓存内容本身的具体结构（例如是单个张量还是 (K, V)
    元组）由调用方（各层的注意力实现）自行决定，本类只负责按层索引
    存取，不关心其内部结构或形状。

    典型用法：
        cache = KVCache(n_layers=12)
        for layer_idx, layer in enumerate(model.layers):
            k, v = layer.attn.compute_new_kv(x)          # 计算新 token 的 K、V
            prev = cache.get(layer_idx)                   # 取出该层历史缓存
            if prev is not None:
                k = torch.cat([prev[0], k], dim=2)         # 沿 seq_len 维拼接
                v = torch.cat([prev[1], v], dim=2)
            cache.update(layer_idx, (k, v))                # 写回更新后的缓存
        # 处理下一个新序列前，重置缓存：
        cache.reset()
    """

    def __init__(self, n_layers):
        """初始化 KV 缓存容器。

        参数：
            n_layers (int): Transformer 的层数，决定缓存列表的长度。
                每一层拥有独立的一份 KV 缓存槽位。

        属性：
            self.cache (list): 长度为 n_layers 的列表，初始时每个元素
                都是 None，表示该层还没有任何缓存（即尚未处理过任何
                token，或刚被 reset 过）。
        """
        # 用 None 占位，表示每一层的缓存尚未写入；
        # 后续通过 update() 写入实际的 (K, V) 等缓存内容。
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        """获取指定层当前的 KV 缓存内容。

        参数：
            layer_idx (int): 层索引（从 0 开始）。

        返回：
            该层当前缓存的内容；如果该层尚未写入过缓存（或刚被
            reset），则返回 None。
        """
        # 直接按下标读取，不做拼接或形状变换，纯粹是「取出」操作。
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        """更新（覆盖写入）指定层的 KV 缓存内容。

        参数：
            layer_idx (int): 层索引（从 0 开始）。
            value: 新的缓存内容，通常是拼接了历史缓存与本次新算出的
                K、V 之后的结果（例如 (K_cat, V_cat)，其中 K_cat/V_cat
                的 seq_len 维度已经比上一次多出本次新增的 token 数）。

        说明：
            这里是「整体覆盖」而非「追加」——即调用方需要在外部先把
            旧缓存与新 K、V 拼接好，再整体传进来覆盖掉旧值。
        """
        # 注意：这是覆盖赋值，不是原地追加；
        # 拼接（如 torch.cat）的逻辑应由调用方在调用 update 之前完成。
        self.cache[layer_idx] = value

    def get_all(self):
        """获取所有层的 KV 缓存列表。

        返回：
            list: 长度为 n_layers 的列表，每个元素对应一层的缓存内容
                （可能为 None，表示该层尚未写入）。常用于需要一次性
                查看/序列化整个模型缓存状态的场景。
        """
        # 直接返回内部列表的引用，调用方拿到的是同一个 list 对象，
        # 若在外部直接修改其元素，也会影响到本缓存实例。
        return self.cache

    def reset(self):
        """重置（清空）所有层的 KV 缓存。

        通常在开始处理一个全新的输入序列（与之前的序列无关）之前调用，
        避免把上一次生成过程残留的历史 K、V 错误地带入这一次的计算中。

        参数：无

        返回：无（原地修改 self.cache）
        """
        # 逐层将缓存重新置为 None，效果等价于重新执行
        # self.cache = [None] * len(self.cache)，
        # 但这里选择原地逐个赋值，避免重新分配一个新的 list 对象。
        for i in range(len(self.cache)):
            self.cache[i] = None
