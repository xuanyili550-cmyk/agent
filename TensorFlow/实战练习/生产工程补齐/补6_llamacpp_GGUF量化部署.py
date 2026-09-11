"""
================================================================================
 生产工程补齐 · 补6 · llama.cpp / GGUF 量化部署（CPU/边缘也能跑大模型）
================================================================================
 补上「llama.cpp 推理引擎 + GGUF 量化」这条线(之前只讲了 vLLM，且没讲 GGUF/AWQ/GPTQ 动手)。
   ① GGUF：llama.cpp 的模型格式，把权重按不同精度打包成单文件，CPU/Mac(Metal)/边缘设备都能跑。
   ② 量化等级(常见)：Q4_K_M(4bit,最常用,省一半以上体积、质量损失小) / Q5_K_M(略大更准) /
      Q8_0(8bit,接近原版) / Q2_K(极限压缩,质量掉明显)。体积↓延迟↓但质量↓,按设备权衡。
   ③ 三种量化方案优劣(面试高频)：
        · GGUF(llama.cpp)：面向 CPU/Mac/边缘，格式统一、部署最简，社区模型最多。
        · GPTQ：面向 GPU 的后训练量化(4bit)，配 vLLM/AutoGPTQ，推理快。
        · AWQ：激活感知量化，同 4bit 下通常比 GPTQ 精度更好，vLLM 支持。
      一句话：本地/CPU/Mac 用 GGUF；GPU 上线用 AWQ/GPTQ(配 vLLM)。
   ④ 起服务：llama.cpp 自带 OpenAI 兼容 server，一行 `python -m llama_cpp.server --model xxx.gguf`，
      业务代码用 OpenAI SDK 指到它即可(和 vLLM 的调用方式一致,只是后端换了)。
 依赖：llama-cpp-python(本机已装)；但需要一个 .gguf 模型文件。无模型时本文件只打印部署说明，不下载。
 跑：python3 补6_llamacpp_GGUF量化部署.py [模型.gguf]   （注：按用户要求本文件未在本机执行，仅作真实可跑代码）
================================================================================
"""
import os
import sys


def run_gguf(model_path: str, prompt: str = "用一句话解释什么是量化。"):
    """用 llama.cpp 加载 GGUF 模型做一次推理（真实可用代码；需本地 .gguf 文件）。"""
    from llama_cpp import Llama
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,          # 上下文长度
        n_gpu_layers=-1,     # -1=尽量放 GPU/Metal，0=纯 CPU
        verbose=False,
    )
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64, temperature=0.7,
    )
    return out["choices"][0]["message"]["content"]


# —— 起 OpenAI 兼容服务(生产写法，注释供参考) ——
# 命令行：python -m llama_cpp.server --model ./qwen2.5-0.5b-instruct-q4_k_m.gguf --n_gpu_layers -1
# 客户端：from openai import OpenAI; OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
#         —— 和调 vLLM 完全一样，业务代码零改动，只是后端从 GPU vLLM 换成本地 llama.cpp。

QUANT_TABLE = {
    "Q4_K_M": "4bit 最常用；体积/质量最佳平衡",
    "Q5_K_M": "5bit；比 Q4 略大更准",
    "Q8_0":   "8bit；接近原版，体积大",
    "Q2_K":   "2bit；极限压缩，质量掉明显",
}


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else ""
    if model and os.path.exists(model):
        ans = run_gguf(model)          # 有模型 → 真跑
        print(f"✅ 补6 跑通：llama.cpp 加载 {os.path.basename(model)} 推理 → {ans[:40]}…")
    else:
        # 无 .gguf 模型 → 打印部署说明，不下载(遵循本机不跑大模型的约定)
        print("ℹ️ 补6（未提供 .gguf 模型，仅打印说明；量化方案/命令见文件顶部 docstring）")
        # 量化等级参考：QUANT_TABLE
        assert "Q4_K_M" in QUANT_TABLE
    # 面试：Q GGUF/AWQ/GPTQ 怎么选? A CPU/Mac/边缘用 GGUF; GPU 上线用 AWQ/GPTQ 配 vLLM;
    #      Q Q4_K_M 是什么? A 4bit 量化,体积省一半以上、质量损失小,最常用; Q llama.cpp 怎么起服务? A llama_cpp.server,OpenAI 兼容。


if __name__ == "__main__":
    main()
