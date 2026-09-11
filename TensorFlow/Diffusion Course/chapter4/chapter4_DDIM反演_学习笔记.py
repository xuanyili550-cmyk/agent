"""
================================================================================
 Diffusion Course · Chapter 4 · DDIM 反演做图像编辑（学习笔记 · DDIM 公式 numpy 可跑）
================================================================================
 一句话：DDIM 采样是确定性的、可逆——把真实图"反推回噪声",再带新提示去噪 → 精准编辑真实照片。
 本章讲：
   ① DDIM 确定性采样：由 x_t 和预测噪声 ε 算 x₀̂,再走到 x_{t-1}(无随机项,可逆)。
   ② 反演(inversion)：反向跑得到对应潜噪声。
   ③ 换提示重新去噪 → 编辑(改风格/物体),保留原图结构。
 要点：DDIM 无随机 → 图↔噪声可来回;这让"编辑真实图"而非"从零生成"成为可能。
 跑：python3 chapter4_DDIM反演_学习笔记.py   （DDIM 单步公式纯 numpy 真跑;真实反演 🔴需GPU/下SD)
================================================================================
"""
import numpy as np


def ddim_predict_x0(x_t, eps, abar_t):
    """由当前 x_t 和预测噪声 ε,反解"干净图估计" x₀̂ = (x_t - √(1-ᾱ_t)·ε)/√ᾱ_t。"""
    return (x_t - np.sqrt(1 - abar_t) * eps) / np.sqrt(abar_t)


def ddim_step(x_t, eps, abar_t, abar_prev):
    """DDIM 确定性一步：x_{t-1} = √ᾱ_{t-1}·x₀̂ + √(1-ᾱ_{t-1})·ε(无随机项 → 可逆)。"""
    x0 = ddim_predict_x0(x_t, eps, abar_t)
    return np.sqrt(abar_prev) * x0 + np.sqrt(1 - abar_prev) * eps


def invert_real():   # 🔴 需 GPU/下 SD,默认不调用
    from diffusers import DDIMScheduler, StableDiffusionPipeline  # noqa: F401
    return "见 diffusers DDIM 反演流程(先反推 latent,再换 prompt 去噪)"


def main():
    rng = np.random.default_rng(0)
    x0_true = rng.standard_normal(8)
    abar_t, abar_prev = 0.3, 0.5
    eps = rng.standard_normal(8)
    x_t = np.sqrt(abar_t) * x0_true + np.sqrt(1 - abar_t) * eps   # 构造一个 x_t
    x0_hat = ddim_predict_x0(x_t, eps, abar_t)
    assert np.allclose(x0_hat, x0_true, atol=1e-6)               # 已知 ε 时能精确反解 x₀
    x_prev = ddim_step(x_t, eps, abar_t, abar_prev)
    assert x_prev.shape == x0_true.shape
    print("✅ Ch4 跑通：DDIM 由(x_t,ε)精确反解 x₀̂;确定性一步 x_t→x_{t-1}(无随机→可逆,支持反演编辑)")
    # 面试：Q DDIM vs DDPM? A DDIM 确定性/少步/可逆; Q 反演做什么? A 把真实图反推成噪声再重生成→编辑。


if __name__ == "__main__":
    main()
