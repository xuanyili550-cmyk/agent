"""
 Diffusion Course · Ch2 · 基础案例：无分类器引导 CFG 的真实机制(纯 numpy,可跑)
 CFG:同一个网络跑两次,得到"有条件预测 ε_cond"与"无条件预测 ε_uncond",
 再外推:ε_guided = ε_uncond + s·(ε_cond − ε_uncond)。s 是引导强度。
 本文件用一个"理想去噪器"(它能从 x_t 反出对应的 x₀)搭真实场景:
   · 有条件去噪器知道目标 x₀_cond;无条件去噪器只会拉向数据均值 x₀_uncond。
 观察引导后重建的 x̂₀ 如何随 s 从"均值"→"精确条件"→"过冲(过饱和)"。跑：python3 本文件
"""
import numpy as np

# 前向系数(取某个中间 t 的 ᾱ)
abar = 0.3
sa, s1a = np.sqrt(abar), np.sqrt(1 - abar)
rng = np.random.default_rng(1)

x0_cond = np.array([1.0, -0.5, 0.8, 0.2, -1.0])     # 条件想要生成的"干净目标"
x0_uncond = np.zeros(5)                              # 无条件去噪器只会拉向数据均值(这里=0)
eps_true = rng.standard_normal(5)                    # 前向用到的真噪声
x_t = sa * x0_cond + s1a * eps_true                  # 前向加噪得到含噪样本


def denoise_eps(x0_pred):
    """理想去噪器:给定它心里认为的干净图 x0_pred,反推它会预测的 ε。
    这正是 ε 与 x₀ 的一一对应:ε = (x_t − √ᾱ·x₀)/√(1-ᾱ)。"""
    return (x_t - sa * x0_pred) / s1a


def cfg(scale):
    eps_uncond = denoise_eps(x0_uncond)              # 无条件分支:拉向均值
    eps_cond = denoise_eps(x0_cond)                  # 有条件分支:命中目标
    eps_g = eps_uncond + scale * (eps_cond - eps_uncond)   # ← CFG 外推公式
    x0_hat = (x_t - s1a * eps_g) / sa                # 由引导后的 ε 重建 x̂₀
    return eps_g, x0_hat


if __name__ == "__main__":
    print("scale   ‖ε_g−ε_uncond‖   ‖x̂₀−目标‖   ‖x̂₀−均值‖")
    dist_to_cond = {}
    for s in [0.0, 1.0, 3.0, 7.5]:
        eps_g, x0_hat = cfg(s)
        d_cond = np.linalg.norm(x0_hat - x0_cond)
        d_mean = np.linalg.norm(x0_hat - x0_uncond)
        dist_to_cond[s] = d_cond
        print(f"{s:5}   {np.linalg.norm(eps_g - denoise_eps(x0_uncond)):8.3f}      {d_cond:7.3f}    {d_mean:7.3f}")

    # 机制核对:s=0 → 落在均值; s=1 → 精确命中条件目标; s 越大 → 偏离越远(过冲)
    _, x0_at0 = cfg(0.0)
    _, x0_at1 = cfg(1.0)
    assert np.allclose(x0_at0, x0_uncond, atol=1e-9)          # 无引导=无条件均值
    assert np.allclose(x0_at1, x0_cond, atol=1e-9)            # s=1 恰好=条件目标
    assert dist_to_cond[1.0] < 1e-9 < dist_to_cond[7.5]      # s=7.5 越过目标 → 过冲
    assert dist_to_cond[7.5] > dist_to_cond[3.0]             # 越大越过冲

    print("\n✅ CFG 机制成立:s=0 退回均值、s=1 命中条件、s>1 沿(cond−uncond)方向外推(增强但可能过饱和)")
    print("面试 Q：CFG 为什么能不训练分类器就实现条件增强？"
          " A：把条件信号按概率随机置空一起训,推理时同一网络出 cond/uncond 两支,"
          "沿二者差值外推等价于放大 ∇log p(c|x),故名'无分类器'引导。")
