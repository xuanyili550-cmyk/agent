"""
 Diffusion Course · Ch3 · 基础案例：Stable Diffusion 原理版(纯 numpy,离线可跑)
 SD = 潜空间扩散(latent diffusion):不在像素上加噪,先用 VAE 编码到低维潜空间,
 在潜空间"前向加噪 → U-Net 去噪",再解码回像素。本机无 GPU/无模型,这里用:
   · 可逆正交变换当"VAE"(编码 z=Q·x,解码 x=Qᵀ·z,无损,演示"潜空间"这一核心结构);
   · 可解析的 MMSE(维纳)去噪器代替 U-Net —— 它从"数据先验"一步预测干净潜向量,
     真实、诚实地不完美(不像已知 ε 那样作弊反解)。
 真实可用的调库写法见文末 🔴 函数(默认不执行)。跑：python3 本文件
"""
import numpy as np

rng = np.random.default_rng(3)
D = 64                                        # 8×8 "小图" 展平成 64 维


def make_gradient_images(m):
    """造一批"小图":随机斜率的线性渐变(当作训练数据分布,用来估计先验)。"""
    ramp = np.linspace(-1, 1, D)
    slopes = rng.uniform(0.3, 1.5, (m, 1))
    offs = rng.uniform(-0.3, 0.3, (m, 1))
    return slopes * ramp + offs               # (m, D)


# —— "VAE":一个固定正交矩阵 Q,编码 z=Q·x、解码 x=Qᵀ·z(正交 → 无损可逆) ——
Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
encode = lambda x: x @ Q.T                    # 像素 → 潜空间
decode = lambda z: z @ Q                      # 潜空间 → 像素

# —— 在潜空间估计数据先验(每维均值/方差),这是 MMSE 去噪器的"知识" ——
Z_train = encode(make_gradient_images(2000))
m_lat, v_lat = Z_train.mean(0), Z_train.var(0)

abar = 0.2                                     # 取某个较嘈杂的 t 的 ᾱ
sa, s1a = np.sqrt(abar), np.sqrt(1 - abar)


def mmse_denoise(z_t):
    """维纳(MMSE)一步去噪:给定 z_t=√ᾱ·z₀+√(1-ᾱ)·ε 且先验 z₀~N(m,v),
    后验均值 E[z₀|z_t]=m + (√ᾱ·v)/(ᾱ·v+(1-ᾱ))·(z_t−√ᾱ·m)。这就是"学到的去噪器"的解析对应。"""
    gain = (sa * v_lat) / (abar * v_lat + (1 - abar))
    return m_lat + gain * (z_t - sa * m_lat)


if __name__ == "__main__":
    # ① VAE 无损:decode(encode(x)) == x
    x_probe = make_gradient_images(5)
    assert np.allclose(decode(encode(x_probe)), x_probe, atol=1e-10)

    # ② 前向加噪 + 一步去噪:在一批新图上比较"朴素反除" vs "MMSE 去噪"的误差
    X0 = make_gradient_images(500)                     # held-out 测试图
    Z0 = encode(X0)
    naive_err, mmse_err = [], []
    for z0, x0 in zip(Z0, X0):
        eps = rng.standard_normal(D)
        z_t = sa * z0 + s1a * eps                      # 潜空间前向加噪
        z_naive = z_t / sa                             # 朴素:直接除 √ᾱ(不去噪,含放大的噪声)
        z_hat = mmse_denoise(z_t)                      # MMSE 一步去噪
        naive_err.append(np.linalg.norm(decode(z_naive) - x0))
        mmse_err.append(np.linalg.norm(z_hat - z0))
    naive_err, mmse_err = np.mean(naive_err), np.mean(mmse_err)
    print(f"前向加噪 ᾱ={abar}(噪声占比√(1-ᾱ)={s1a:.2f})")
    print(f"朴素反除平均误差 = {naive_err:.3f}")
    print(f"MMSE 去噪平均误差 = {mmse_err:.3f}  → 去噪器利用先验,误差显著更小")
    assert mmse_err < naive_err                        # 去噪器真的有用(且诚实地非零)
    assert mmse_err > 1e-3                              # 诚实:一步去噪不完美,残差存在

    print("✅ SD 原理跑通:潜空间(可逆VAE)里前向加噪,再用 MMSE 去噪器一步预测干净潜向量并解码")
    print("面试 Q：Stable Diffusion 为何在'潜空间'而非像素空间做扩散？"
          " A：VAE 把高维像素压到低维潜空间(如 512²×3→64²×4),扩散与 U-Net 只在小得多的张量上运算,"
          "算力/显存大降且几乎不损画质;文本经 CLIP 编码作为交叉注意力条件、配 CFG 实现文生图。")


# 🔴 真实可用写法(需 GPU/下 SD 模型;默认不执行,仅参考) ————————————————
def real_stable_diffusion():  # pragma: no cover
    import torch
    from diffusers import StableDiffusionPipeline
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    image = pipe("a cozy cabin in the snow, digital art", guidance_scale=7.5, num_inference_steps=25).images[0]
    image.save("/tmp/sd_case.png")
    print("✅ 已生成 /tmp/sd_case.png (guidance_scale=7.5)")
