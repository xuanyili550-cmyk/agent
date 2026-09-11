"""
================================================================================
 服务层 · 池化与数值底层（★本服务的"底层原理"核心，纯张量、可离线验证）
================================================================================
 【为什么单独一个文件】把 Transformer 输出变成"一句话一个向量"，靠的就是【池化】；而检索靠
   【L2 归一化 + 余弦】；打分靠【softmax】。这些是最容易被面试问、也最容易写错的底层，单独放、写清楚、单测覆盖。

 【三个底层原理】
   ① mean pooling 为什么要用 mask 加权：
      模型对每个 token 输出一个向量(last_hidden_state: [B, L, H])。要压成一句一个向量就得平均。
      但一个 batch 里短句会被 [PAD] 补齐，直接 mean 会把 PAD 位的"废向量"也平均进去、稀释语义。
      所以用 attention_mask(1=真实 token,0=PAD)加权：只对真实 token 求和 ÷ 真实 token 数。
   ② CLS pooling：BERT 类模型第 0 个位置是 [CLS]，bge 等模型就取它当句向量(取 [:,0])。
      —— 用哪种池化【由模型决定】，配错了向量质量会明显下降(常见坑)。
   ③ L2 归一化后，点积 = 余弦相似度：
      cos(a,b)=a·b/(|a||b|)；把每个向量除以自己的模长(变成单位向量)后 |a|=|b|=1，于是 a·b 直接就是余弦。
      好处：整库检索只需一次矩阵乘(query·库^T)，且只看"方向(语义)"不受"长度(篇幅)"影响。

 【softmax 为什么要减最大值】exp 容易上溢(大数变 inf)。减去每行最大值后再 exp，结果不变但数值稳定。
================================================================================
"""
import torch


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """mask 加权平均池化。last_hidden_state:[B,L,H], attention_mask:[B,L] → [B,H]。
    只平均真实 token(排除 [PAD]),否则短句被 PAD 稀释。"""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)   # [B,L,1] 便于逐元素乘
    summed = (last_hidden_state * mask).sum(dim=1)                    # 真实 token 向量求和 [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                          # 真实 token 个数(防 0 除)
    return summed / counts                                           # 求平均


def cls_pooling(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """取 [CLS](第 0 个 token)的向量当句向量。bge 等模型用这种。"""
    return last_hidden_state[:, 0]


def pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, method: str) -> torch.Tensor:
    """按配置选池化方式——池化方式必须和模型匹配(mean↔MiniLM/E5，cls↔bge)。"""
    if method == "cls":
        return cls_pooling(last_hidden_state)
    return mean_pooling(last_hidden_state, attention_mask)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2 归一化成单位向量。归一化后：两个向量点积 = 余弦相似度(检索地基)。"""
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """数值稳定 softmax：先减每行最大值再 exp(防上溢)，结果与朴素 softmax 一致。"""
    x = x - x.max(dim=dim, keepdim=True).values
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)
