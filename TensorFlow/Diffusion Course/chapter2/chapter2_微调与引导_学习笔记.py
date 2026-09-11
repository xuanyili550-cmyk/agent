"""
================================================================================
 Diffusion Course · Chapter 2 · 扩散模型微调与引导（学习笔记 · 引导公式 numpy 可跑）
================================================================================
 一句话：把预训练扩散模型微调到新画风 + 用"引导"在采样时把生成往目标方向推。
 本章讲：
   ① 微调：在自定义图集上继续训 UNet(DDIMScheduler 加速采样)。
   ② 无分类器引导(CFG)：ε = ε_uncond + scale·(ε_cond − ε_uncond) —— 本文件 numpy 真算。
   ③ 引导强度 scale：越大越贴合条件但越可能失真/过饱和。
 要点：CFG 让"无条件生成"变"可控生成",是 SD 出好图的关键;一次前向同时算有/无条件两支。
 跑：python3 chapter2_微调与引导_学习笔记.py   （②CFG 原理纯 numpy 真跑;真实微调 🔴需GPU)
================================================================================
"""
import numpy as np


def cfg(eps_uncond, eps_cond, scale):
    """无分类器引导：在"无条件预测"基础上,朝"条件-无条件"方向放大 scale 倍。"""
    return eps_uncond + scale * (eps_cond - eps_uncond)


def finetune_real():   # 🔴 需 GPU,默认不调用
    from diffusers import DDIMScheduler, DDPMPipeline  # noqa: F401
    # image_pipe = DDPMPipeline.from_pretrained(...); 在自定义数据上继续训 UNet ...
    return "见 diffusers 官方微调流程"


def main():
    rng = np.random.default_rng(0)
    eps_u = rng.standard_normal(8)            # 无条件噪声预测
    eps_c = rng.standard_normal(8)            # 条件(带提示)噪声预测
    g1 = cfg(eps_u, eps_c, 1.0)               # scale=1 → 等于纯条件预测
    g7 = cfg(eps_u, eps_c, 7.5)               # scale=7.5 → SD 常用,更贴合提示
    assert np.allclose(g1, eps_c)             # scale=1 时正好还原条件预测
    assert np.linalg.norm(g7 - eps_u) > np.linalg.norm(g1 - eps_u)  # scale 越大偏离无条件越远
    print(f"✅ Ch2 跑通：CFG scale=1 还原条件预测;scale=7.5 更强引导(偏离无条件更远)")
    # 面试：Q CFG 公式? A ε_u+scale·(ε_c-ε_u); Q scale 太大? A 过饱和/失真; Q 为何叫"无分类器"? A 不需单独训分类器,靠有/无条件之差。


if __name__ == "__main__":
    main()
