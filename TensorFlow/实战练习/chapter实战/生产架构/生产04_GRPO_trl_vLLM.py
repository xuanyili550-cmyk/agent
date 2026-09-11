"""
================================================================================
 生产04 · 生产级 GRPO 强化学习：trl.GRPOTrainer + vLLM 加速采样 + 多卡
================================================================================
 ⚠ 需要 GPU(采样密集，CPU 上跑不动)，本机跑不了；这是“真上线怎么搭”的架构参考。
   本文件的奖励函数 + train() 是真实可用的 Python(不是字符串)：trl/datasets 惰性导入，
   `python3 生产04_GRPO_trl_vLLM.py` 只打印说明，不会真去下模型或开训。
   本地能跑的手写小版本见 ../案例4_GRPO强化学习.py(证明你懂算法底层)。

 为什么生产 GRPO 这么搭：
   · 为什么用 trl.GRPOTrainer：官方实现,处理好了采样/优势/裁剪/KL/分布式,不用手写。
   · 为什么必须 vLLM(use_vllm=True)：GRPO 每步要对每个提示采“一组(8-16个)”补全,
     采样是最大瓶颈;vLLM 的连续批处理让采样快几倍到几十倍,不然训练慢到不可用。
   · 为什么可验证奖励：数学/代码/事实类任务答案能自动验证(对/错),不用训奖励模型,
     这就是 DeepSeek-R1 训推理能力的路子(见 ../../Chapter 12)。
   · 为什么配 LoRA：RL 采样密集本就贵,LoRA 只训 <1% 参数,省显存、能挂多适配器。

 —— 依赖 ——
   pip install "trl[vllm]" peft math_verify
================================================================================
"""
import os
import re

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")


def extract_answer(text):
    return text.split("####")[-1].strip() if "####" in text else text


# 多个奖励函数加权组合(生产常这么塑造行为)——都是纯 Python，可直接单测：
def correctness_reward(completions, answer, **kw):
    """可验证：抽出模型答案和标准答案比,对=2 错=0。"""
    preds = [extract_answer(c) for c in completions]
    return [2.0 if p == a else 0.0 for p, a in zip(preds, answer)]


def format_reward(completions, **kw):
    """格式：必须按 <reasoning>...</reasoning><answer>...</answer> 输出。"""
    pat = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    return [0.5 if re.search(pat, c, re.DOTALL) else 0.0 for c in completions]


def load_gsm8k():
    """GSM8K 数学题(带标准答案，可验证)。生产 datasets 正常;本机坏了可用 pandas 直读。"""
    from datasets import load_dataset          # 惰性导入
    return load_dataset("openai/gsm8k", "main", split="train")


def train():
    from peft import LoraConfig                # 惰性导入
    from trl import GRPOConfig, GRPOTrainer

    dataset = load_gsm8k()

    args = GRPOConfig(
        output_dir="grpo-qwen",
        learning_rate=5e-6,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=8,               # ★每个提示采一组 8 个(GRPO 的“组”)
        max_prompt_length=256,
        max_completion_length=1024,      # 推理链要长
        use_vllm=True,                   # ★用 vLLM 加速采样(生产必开，否则太慢)
        bf16=True,
        max_steps=1000,
        logging_steps=1,
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=[correctness_reward, format_reward],   # 多奖励加权
        args=args,
        train_dataset=dataset,
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                               task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model("grpo-qwen-final")     # 推到模型仓库供推理服务加载
    return trainer


NOTES = """
 生产要点：
  · 采样是瓶颈：num_generations 越大信号越稳但越慢;务必 use_vllm=True。生产常把“采样”和
    “训练”分到不同 GPU(vLLM 专门吃采样),或用异步/离线采样。
  · 奖励设计是灵魂：选“组内有对有错(有方差)”的题才有信号;可验证奖励(答案/单测/编译)最可靠;
    多个奖励(正确性+格式+简洁)加权组合塑造完整行为。防“奖励黑客”(模型钻奖励空子)。
  · 监控：reward 均值(该升)、reward_std(组内方差,别塌成0)、KL(与参考模型的距离,别跑太远崩掉)。
  · 显存：GRPO 要同时放 策略模型 + 参考模型 + 采样,很吃显存;LoRA + vLLM + 多卡是标配。
  · 平台/工具：美国常配 Ray + vLLM;也有 veRL / OpenRLHF 等 RLHF 框架。中国类似,自建居多。
"""

if __name__ == "__main__":
    import sys
    # 奖励函数可以在本机直接自检(纯 Python,不碰 GPU/模型)：
    comps = ["<reasoning>2+3=5</reasoning><answer>5</answer>", "answer is 5", "#### 5"]
    assert format_reward(comps) == [0.5, 0.0, 0.0]
    assert correctness_reward(["#### 5", "#### 4"], ["5", "5"]) == [2.0, 0.0]
    print(">>> 奖励函数本机自检通过(format_reward / correctness_reward)。")
    if "--train" in sys.argv:
        train()                       # 需 GPU + trl[vllm]
    else:
        print(__doc__)
        print("=== 生产要点 ===", NOTES)
        print(">>> 架构参考。真跑：GPU 机 `python3 生产04_GRPO_trl_vLLM.py --train`。"
              "本地手写小版本见 ../案例4_GRPO强化学习.py")
