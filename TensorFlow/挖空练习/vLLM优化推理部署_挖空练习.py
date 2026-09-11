"""
================================================================================
 vLLM 优化推理部署 挖空练习（vLLM 在 Mac 装不了，用 mlx-lm 实践等价概念）
================================================================================
 玩法：
   1) 把每个 ______ 换成正确代码（凭记忆）
   2) 运行：python3 vLLM优化推理部署_挖空练习.py
   3) 没填的报 NameError；卡住 → 文件底部「答案区」
 概念对应 vLLM：load≈LLM()，make_sampler≈SamplingParams，generate≈llm.generate。
================================================================================
"""
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# 练习1：加载模型（函数名？返回 model, tokenizer）
model, tok = load(MODEL)

# 练习2：造采样器——温度 0.7、top_p 0.95（对应 vLLM 的 SamplingParams）
sampler = make_sampler(temp=0.7, top_p=0.95)

prompt = "Explain what an LLM is in one sentence."
mx.random.seed(0)

# 练习3：生成——传 model, tok, prompt；最多 40 个 token；用上面的 sampler
out = generate(model, tok, prompt, max_tokens=40, sampler=sampler)

print("生成：", out.strip())
print("👉 对应 vLLM：llm = LLM(model); llm.generate(prompt, SamplingParams(temperature=.., top_p=..))")


# ==============================================================================
#  答案区（卡住再看！）
# ------------------------------------------------------------------------------
#  1: load(MODEL)
#  2: temp=0.7, top_p=0.95
#  3: max_tokens=40, sampler=sampler
# ==============================================================================
