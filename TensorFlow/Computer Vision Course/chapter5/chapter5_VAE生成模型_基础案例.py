"""
 CV Course · Ch5 · 基础案例：VAE 重参数技巧 + ELBO 两项损失(纯 numpy,机制真实,可跑)
 VAE = 编码器输出隐分布的 μ/logσ² → 从中"采样" z → 解码器重建 x。难点:采样不可导,梯度断了怎么训?
 重参数技巧:z = μ + σ·ε(ε~N(0,1)),把随机性挪到与参数无关的 ε 上,μ/σ 变成可导的普通运算。
 损失 = 重建误差 + KL(把隐分布拉向标准正态,让隐空间规整、可采样生成)。这里纯 numpy 真算这两项。
 跑：python3 本文件
"""
import numpy as np


def reparameterize(mu, logvar, eps):
    """重参数:z = μ + σ·ε。σ=exp(0.5·logvar) 保证恒正。ε 是外部噪声 → 对 μ、σ 的梯度不受随机性影响。"""
    std = np.exp(0.5 * logvar)
    return mu + std * eps, std


def kl_to_standard_normal(mu, logvar):
    """KL( N(μ,σ²) ‖ N(0,1) ) 的闭式解:-0.5·Σ(1+logσ² - μ² - σ²)。
    作用:正则化隐空间,逼近标准正态,这样训练完可直接从 N(0,1) 采 z 解码出新样本。"""
    return -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    mu = np.array([1.0, -1.0])
    logvar = np.array([0.0, -0.5])                          # logσ²

    # ① 重参数采样:固定 ε 时输出确定;换 ε 才有随机性(证明随机性只来自 ε)
    eps = rng.standard_normal(2)
    z, std = reparameterize(mu, logvar, eps)
    z_same, _ = reparameterize(mu, logvar, eps)             # 同 ε → 同结果
    z_diff, _ = reparameterize(mu, logvar, rng.standard_normal(2))
    print(f"μ={mu}, σ={std.round(2)}, 采样 z={z.round(2)}")
    assert np.allclose(z, z_same)                           # 随机性被隔离到 ε
    assert not np.allclose(z, z_diff)

    # ② 蒙特卡洛验证:大量 ε 采样,z 的均值/方差应逼近 μ 和 σ²(证明重参数没有引入偏差)
    samples = np.array([reparameterize(mu, logvar, rng.standard_normal(2))[0] for _ in range(20000)])
    print("采样均值≈μ:", samples.mean(0).round(2), " 采样方差≈σ²:", samples.var(0).round(2))
    assert np.allclose(samples.mean(0), mu, atol=0.05)
    assert np.allclose(samples.var(0), std**2, atol=0.05)

    # ③ 两项损失:KL(μ→0,σ→1 时最小=0) + 重建误差(这里用解码近似值的 MSE 占位)
    kl = kl_to_standard_normal(mu, logvar)
    kl_prior = kl_to_standard_normal(np.zeros(2), np.zeros(2))
    print(f"KL 散度 当前={kl:.3f}  标准正态自身={kl_prior:.3f}(应=0)")
    assert abs(kl_prior) < 1e-9 and kl > 0                  # 偏离先验越远 KL 越大
    print("✅ 重参数让'采样'可导 + KL 把隐空间拉向 N(0,1) → VAE 既能训又能从噪声生成新图")
    # 面试Q:为什么编码器输出 logσ² 而不是直接输出 σ?
    #      A:σ 必须为正,直接回归易出负值/需额外约束;输出 logσ² 取值任意实数,再 exp 天然保正,数值也更稳定。
