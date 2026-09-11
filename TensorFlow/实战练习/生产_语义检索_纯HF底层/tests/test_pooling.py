"""底层数学单测：不加载任何模型，用合成张量验证池化/归一化/softmax 正确。"""
import torch

from app.services import pooling


def test_mean_pooling_excludes_pad():
    # 3 个 token 的向量；attention_mask=[1,1,0] 表示第 3 个是 PAD，应被排除
    h = torch.tensor([[[2.0, 4.0], [4.0, 8.0], [100.0, 100.0]]])   # [1,3,2]
    mask = torch.tensor([[1, 1, 0]])
    out = pooling.mean_pooling(h, mask)                            # 只平均前两行 → (3,6)
    assert torch.allclose(out, torch.tensor([[3.0, 6.0]]))


def test_cls_pooling_takes_first():
    h = torch.tensor([[[1.0, 2.0], [9.0, 9.0]]])
    assert torch.allclose(pooling.cls_pooling(h), torch.tensor([[1.0, 2.0]]))


def test_l2_normalize_unit_norm():
    v = torch.tensor([[3.0, 4.0]])
    n = pooling.l2_normalize(v)
    assert torch.allclose(n.norm(dim=-1), torch.tensor([1.0]))    # 模长变 1
    # 归一化后点积 = 余弦：自己和自己 = 1
    assert torch.allclose((n * n).sum(-1), torch.tensor([1.0]))


def test_stable_softmax_matches_and_sums_to_one():
    x = torch.tensor([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])  # 第二行是大数(测数值稳定)
    p = pooling.stable_softmax(x)
    assert torch.allclose(p.sum(-1), torch.ones(2))
    assert torch.allclose(p, torch.softmax(x, dim=-1), atol=1e-6)
