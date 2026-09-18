"""
 Diffusion Course · Ch4 · 基础案例：DDIM 确定性采样与反演(纯 numpy,可跑)
 DDIM(η=0)把采样变成确定性映射,因此"可逆":能把真实图反演成噪声、改条件后再生成 → 图像编辑之本。
 每步两式:x̂₀ = (x_t − √(1-ᾱ_t)·ε)/√ᾱ_t;  x_s = √ᾱ_s·x̂₀ + √(1-ᾱ_s)·ε (s 是相邻步)。
 本文件:① 已知 ε 时一步精确反解 x₀;② 用一个确定性"ε 模型"跑多步 DDIM 采样;
 ③ DDIM 反演:x₀→噪声→再采样回 x₀,验证往返闭合(误差随步数增多而下降=离散化误差)。
 跑：python3 本文件
"""
import numpy as np

rng = np.random.default_rng(2)
D = 4

# 噪声调度:ᾱ 从接近 1(干净) 递减到接近 0(纯噪声)
def make_schedule(n):
    return np.linspace(0.9999, 0.02, n + 1)      # abar[0]=最干净, abar[-1]=最噪

# 一个"训练好的"确定性 ε 模型:这里用固定小线性映射代替 U-Net(关键是"确定性可复现")
A = 0.15 * rng.standard_normal((D, D))
def eps_model(x):
    return np.tanh(x @ A)                          # 有界、确定性:同一 x 永远给同一 ε


def ddim_reverse(x_t, abar):
    """采样:从最噪 abar[-1] 一路确定性去噪到 abar[0]。"""
    x = x_t
    for i in range(len(abar) - 1, 0, -1):
        at, as_ = abar[i], abar[i - 1]            # 当前步→更干净的相邻步
        eps = eps_model(x)
        x0_hat = (x - np.sqrt(1 - at) * eps) / np.sqrt(at)
        x = np.sqrt(as_) * x0_hat + np.sqrt(1 - as_) * eps
    return x


def ddim_invert(x_0, abar):
    """反演:从干净 abar[0] 沿相反方向确定性推到最噪 abar[-1](DDIM 采样的逆映射)。"""
    x = x_0
    for i in range(0, len(abar) - 1):
        as_, at = abar[i], abar[i + 1]            # 当前步→更噪的相邻步
        eps = eps_model(x)                        # 反演在"当前点"估 ε(故有离散化误差)
        x0_hat = (x - np.sqrt(1 - as_) * eps) / np.sqrt(as_)
        x = np.sqrt(at) * x0_hat + np.sqrt(1 - at) * eps
    return x


if __name__ == "__main__":
    # ① 已知真 ε 时,一步精确反解 x₀(逆过程的理想目标)
    x0 = rng.standard_normal(D)
    abar_t, eps = 0.25, rng.standard_normal(D)
    x_t = np.sqrt(abar_t) * x0 + np.sqrt(1 - abar_t) * eps
    x0_hat = (x_t - np.sqrt(1 - abar_t) * eps) / np.sqrt(abar_t)
    assert np.allclose(x0_hat, x0, atol=1e-6)
    print("① 已知 ε → 一步精确反解 x₀:‖误差‖ =", f"{np.linalg.norm(x0_hat - x0):.2e}")

    # ②③ DDIM 反演往返:x₀ →(反演)→ x_T →(采样)→ x₀_rec,验证闭合;步数越多误差越小
    x0 = rng.standard_normal(D)
    print("\n步数   往返误差 ‖x₀_rec − x₀‖")
    errs = []
    for n in [10, 50, 200]:
        abar = make_schedule(n)
        x_T = ddim_invert(x0, abar)               # 真实图 → 噪声
        x0_rec = ddim_reverse(x_T, abar)          # 噪声 → 重建图
        e = np.linalg.norm(x0_rec - x0)
        errs.append(e)
        print(f"{n:4d}   {e:.3e}")
    assert errs[0] > errs[1] > errs[2]            # 步数越多 → 离散化误差单调下降
    assert errs[-1] < 0.1                         # 200 步已相当闭合(残差为一阶离散化误差,非零但小)
    assert errs[0] / errs[-1] > 5                 # 加密步数带来数量级的误差改善

    print("\n✅ DDIM 确定性+可逆成立:同一 ε 模型下 反演↔采样 往返闭合,误差随步数增多而收敛")
    print("面试 Q：DDIM 反演为什么能做真实图像编辑,而 DDPM 不行？"
          " A：DDPM 每步注入随机噪声不可逆;DDIM 令 η=0 变确定性双射,可把真实图反演到其对应噪声,"
          "再换文本条件重新采样,从而在保留原图结构的前提下编辑内容。")
