"""
================================================================================
 Diffusion Course · Chapter 1 · 扩散模型入门（学习笔记 · 原理段 numpy 可跑）
================================================================================
 一句话：扩散模型 = 学"从纯噪声一步步去噪成图"。训练时学"每步加了多少噪声",采样时反过来减。
 本章讲：
   ① 前向扩散(加噪,有闭式)：x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε —— 本文件用 numpy 真算,看噪声怎么涨。
   ② 反向去噪：UNet 预测每步噪声 ε_θ(x_t,t),按调度器一步步减回 x₀。
   ③ 噪声调度器(betas/alphas/ᾱ) + 采样步数;diffusers 的 DDPMPipeline 封装好这一切。
 要点：核心不是"直接生成图",而是"学会预测噪声";采样=从 N(0,1) 反复去噪。
 跑：python3 chapter1_扩散模型入门_学习笔记.py   （①原理段纯 numpy 真跑;真实生成见 sample_real 🔴需GPU/下模型)
================================================================================
"""
import numpy as np


def make_schedule(T=200, beta1=1e-4, beta2=0.02):
    betas = np.linspace(beta1, beta2, T)      # 每步加噪强度(线性调度)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)                 # ᾱ_t = ∏α：累计"保留原信号"的比例
    return betas, alphas, abar


def q_sample(x0, t, abar, rng):
    """前向加噪闭式：一步到位算出 x_t(不用真的循环加 t 次)。"""
    eps = rng.standard_normal(x0.shape)
    return np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps, eps


def sample_real():   # 🔴 需 GPU/下模型,默认不调用
    from diffusers import DDPMPipeline
    pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
    return pipe(num_inference_steps=50).images[0]


def main():
    rng = np.random.default_rng(0)
    x0 = np.linspace(-1, 1, 64)               # 一个"干净信号"当 x₀
    _, _, abar = make_schedule()
    stds = [round(float(q_sample(x0, t, abar, rng)[0].std()), 2) for t in [0, 50, 100, 199]]
    assert stds[0] < stds[-1] and stds[-1] > 0.8   # t 越大越接近纯噪声(std→1)
    print(f"✅ Ch1 跑通：前向扩散 t=0/50/100/199 的 std ≈ {stds}(越后越像纯噪声)")
    print("   反向：UNet 预测每步噪声 → 按调度器减回 x₀;真实生成用 diffusers DDPMPipeline。")
    # 面试：Q 扩散模型学的是什么? A 预测每步加的噪声 ε; Q 采样起点? A 纯高斯噪声,反复去噪。


if __name__ == "__main__":
    main()
