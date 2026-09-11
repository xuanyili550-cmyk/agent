"""
================================================================================
 CV Course · Chapter 5 · 生成模型 VAE（学习笔记 · 重参数+KL numpy 可跑）
================================================================================
 一句话：VAE = 编码器把图压成"带概率的潜变量(μ,σ)" + 解码器重构;靠重参数技巧让采样可训练。
 本章讲(纯 numpy 演示 重参数 + KL 散度,无需 torch)：
   ① 编码器输出 μ,σ(潜分布);② 重参数:z=μ+σ·ε(ε~N(0,1)) → 让"采样"这一步可反向传播。
   ③ 损失 ELBO = 重构误差 + KL(N(μ,σ²)‖N(0,1))(把潜分布拉向标准正态,便于采样生成)。
   ④ GAN vs VAE:GAN 清晰但难训;VAE 稳但偏糊;扩散后来居上(见 Diffusion Course)。
 要点：重参数把随机性移到 ε,梯度能过 μ,σ;KL 正则让潜空间规整可采样。
 跑：python3 chapter5_VAE生成模型_学习笔记.py   （重参数+KL 纯 numpy 真跑)
================================================================================
"""
import numpy as np


def reparameterize(mu, logvar, rng):
    """z = μ + σ·ε,σ=exp(0.5·logvar)。随机性在 ε,μ/σ 可求导。"""
    std = np.exp(0.5 * logvar)
    eps = rng.standard_normal(mu.shape)
    return mu + std * eps


def kl_divergence(mu, logvar):
    """KL(N(μ,σ²)‖N(0,1)) = -0.5·Σ(1+logσ²-μ²-σ²)。拉潜分布向标准正态。"""
    return -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar))


def main():
    rng = np.random.default_rng(0)
    mu = np.array([0.0, 0.0]); logvar = np.array([0.0, 0.0])   # 恰为 N(0,1)
    z = reparameterize(mu, logvar, rng)
    kl0 = kl_divergence(mu, logvar)
    kl1 = kl_divergence(np.array([2.0, -2.0]), np.array([1.0, 1.0]))
    assert abs(kl0) < 1e-9 and kl1 > kl0        # 标准正态 KL=0;偏离越远 KL 越大
    print(f"✅ Ch5 跑通：重参数 z={z.round(2)};KL(N(0,1))={kl0:.1f}, KL(偏离)={kl1:.2f}(越偏越大)")
    # 面试：Q 重参数解决什么? A 让采样可反向传播(随机性移到 ε); Q KL 项作用? A 正则潜空间向 N(0,1),便于生成。


if __name__ == "__main__":
    main()
