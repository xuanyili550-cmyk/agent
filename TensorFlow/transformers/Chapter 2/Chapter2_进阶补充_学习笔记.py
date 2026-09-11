"""
================================================================================
 Chapter 2 · 进阶补充（补齐审计发现的缺口）—— 可直接运行
================================================================================
 这是对 Chapter2_微调_学习笔记.py 的补充。之前那份把「数据→Trainer→自定义循环」主干
 讲透了，本文件补上被略过的工程细节与方法论：
   1) ClassLabel 标签映射：0/1 ↔ not_equivalent/equivalent 从哪来、怎么查
   2) convert_ids_to_tokens：看 [CLS] s1 [SEP] s2 [SEP] 的编码内部
   3) ★手写循环的“现代优化”三件套：梯度裁剪 / 混合精度 autocast / 检查点保存恢复
   4) 学习率 warmup：为什么开头要慢慢升
   5) 优化器进阶：8-bit Adam、大模型用更低 lr
   6) 过拟合的症状与四类解法
   7) 损失曲线怎么读(训练监控方法论)
   8) 生产/进阶主题：多指标、超参调优、LoRA、量化、梯度检查点、wandb 实验追踪
   9) 一次性整体分词为什么占内存(map 的动机)

 直接运行：python3 Chapter2_进阶补充_学习笔记.py
 只用小分词器(bert-base-uncased，已缓存) + 自造张量，几秒跑完，不下数据/不训练大模型。
================================================================================
"""

import torch


def banner(t):
    print("\n" + "=" * 72 + f"\n {t}\n" + "=" * 72)


# ==============================================================================
# 1) ClassLabel 标签映射：0/1 到语义标签的来源
# ==============================================================================
banner("1) ClassLabel：别硬编码 {0:'不同义'}，它其实存在数据集的 features 里")
from datasets import ClassLabel

# 真实数据集里 raw["train"].features["label"] 就是这样一个 ClassLabel 对象
label_feature = ClassLabel(names=["not_equivalent", "equivalent"])
print("  names        =", label_feature.names)
print("  int2str(1)   =", label_feature.int2str(1))     # 1 -> 'equivalent'
print("  str2int('not_equivalent') =", label_feature.str2int("not_equivalent"))
print("  用法：raw['train'].features['label'] 就是它；别自己猜 0/1 含义，查这个最可靠。")


# ==============================================================================
# 2) convert_ids_to_tokens：看句子对被编码成什么样
# ==============================================================================
banner("2) convert_ids_to_tokens：看编码内部的 [CLS]/[SEP] 结构")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
enc = tok("How are you?", "I am fine")           # 句子对：传两个字符串
print("  input_ids     :", enc["input_ids"])
print("  对应 tokens   :", tok.convert_ids_to_tokens(enc["input_ids"]))
print("  token_type_ids:", enc["token_type_ids"], " ← 0=句A, 1=句B，模型靠它区分前后句")
print("  结构 = [CLS] 句A [SEP] 句B [SEP]；convert_ids_to_tokens 让你把 id 反查回 token 看内部。")


# ==============================================================================
# 3) ★手写训练循环的“现代优化”三件套
# ==============================================================================
banner("3) 手写循环的现代优化：梯度裁剪 / 混合精度 / 检查点")

# ---- 3.1 梯度裁剪 clip_grad_norm_：把梯度范数限制住，防“梯度爆炸”----
lin = torch.nn.Linear(4, 1)
x = torch.randn(8, 4)
loss = ((lin(x) - 100.0) ** 2).mean()     # 造一个大 loss → 大梯度
loss.backward()
before = torch.nn.utils.clip_grad_norm_(lin.parameters(), max_norm=1.0)  # 返回裁剪前的总范数
after = torch.sqrt(sum((p.grad ** 2).sum() for p in lin.parameters()))
print(f"  3.1 梯度裁剪：裁剪前范数={before:.2f} → 裁剪后范数={after:.2f}(≤1.0)")
print("      放在 optimizer.step() 之前；防止某步梯度过大把参数带飞。")

# ---- 3.2 混合精度 autocast：前向用 fp16/bf16 算，更快更省显存 ----
print("  3.2 混合精度：训练时把部分计算用半精度(fp16/bf16)，速度/显存都省。")
print("      GPU 上标准写法：")
print("        scaler = torch.cuda.amp.GradScaler()")
print("        with torch.autocast('cuda', dtype=torch.float16):")
print("            loss = model(**batch).loss")
print("        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()")
print("      GradScaler 负责把 fp16 下过小的梯度放大回来，防下溢(underflow)。")

# ---- 3.3 检查点保存/恢复：训练中断能续上 ----
import tempfile, os
ckpt_path = os.path.join(tempfile.mkdtemp(), "ckpt.pt")
optimizer = torch.optim.AdamW(lin.parameters(), lr=1e-3)
# 保存：模型 + 优化器 + 训练进度 一起存(只存 model 不够，优化器动量也要存)
torch.save({"model": lin.state_dict(), "optim": optimizer.state_dict(), "epoch": 3}, ckpt_path)
# 恢复：新建同结构对象再 load_state_dict，从第 3 轮接着练
lin2 = torch.nn.Linear(4, 1)
opt2 = torch.optim.AdamW(lin2.parameters(), lr=1e-3)
state = torch.load(ckpt_path)
lin2.load_state_dict(state["model"]); opt2.load_state_dict(state["optim"])
print(f"  3.3 检查点：已存/读回，恢复到 epoch={state['epoch']}(模型+优化器状态都要存，才能真续训)")


# ==============================================================================
# 4) 学习率 warmup：为什么开头要慢慢升
# ==============================================================================
banner("4) warmup：前若干步把学习率从 0 慢慢升到目标值")
from transformers import get_scheduler

opt = torch.optim.AdamW(lin.parameters(), lr=1.0)   # 目标 lr=1.0，便于看比例
sched = get_scheduler("linear", optimizer=opt, num_warmup_steps=3, num_training_steps=10)
lrs = []
for _ in range(10):
    lrs.append(round(opt.param_groups[0]["lr"], 3))
    opt.step(); sched.step()
print("  10 步的 lr 轨迹:", lrs)
print("  前 3 步(warmup)从 0 升到峰值，再线性衰减到 0。")
print("  为什么 warmup：训练初期参数还很乱，一上来就大 lr 容易把模型带崩；先小步热身更稳。")


# ==============================================================================
# 5) 优化器进阶
# ==============================================================================
banner("5) 优化器进阶")
print("""
  · AdamW 是微调标配(比 Adam 的权重衰减更正确)。
  · 显存紧张 → 8-bit Adam(pip install bitsandbytes)：把优化器状态量化到 8bit，省一大半显存。
      import bitsandbytes as bnb;  optimizer = bnb.optim.Adam8bit(model.parameters(), lr=...)
  · 学习率经验：微调大模型用较低 lr(1e-5 ~ 3e-5)；小模型/从零可高些。lr 太大→震荡/发散。
  · weight_decay(权重衰减，如 0.01)：给大权重加惩罚，防过拟合；bias/LayerNorm 一般不衰减。
""")


# ==============================================================================
# 6) 过拟合：症状与四类解法
# ==============================================================================
banner("6) 过拟合的症状与解法")
print("""
  症状(怎么看出来)：
    · 训练 loss 一直降，但验证 loss 开始回升(两条曲线分叉)。
    · 训练准确率远高于验证准确率(差距越拉越大)。
  四类解法：
    1 正则化：weight_decay、dropout。
    2 提前停止 Early Stopping：验证指标不再变好就停(见下方 EarlyStoppingCallback)。
    3 数据增强 / 加更多数据。
    4 降低模型复杂度 / 减少训练轮数。
  EarlyStoppingCallback 用法(配 Trainer)：
    args = TrainingArguments(..., load_best_model_at_end=True,
                             metric_for_best_model="f1", eval_strategy="epoch")
    trainer = Trainer(..., callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
""")


# ==============================================================================
# 7) 损失曲线怎么读(训练监控方法论)
# ==============================================================================
banner("7) 训练过程该盯什么(读图能力)")
print("""
  训练 loss 的正常形态：初期高 → 快速下降 → 逐渐平缓收敛。
  训练中盯 4 件事：
    · 损失是否在收敛(还在降 or 已平)      · 有没有过拟合迹象(验证 loss 回升)
    · 学习率是否合理(loss 剧烈震荡=lr 太大)  · 训练是否稳定(有没有 NaN/突刺)
  训练后分析：最终性能达标没、训练效率(时间/显存)、泛化(验证≈测试?)、整体趋势健康否。
  工具：logging_steps/eval_steps/save_steps 控制“多少步记一次日志/评估/存档”。
""")


# ==============================================================================
# 8) 生产 / 进阶主题(名词扫盲，知道有这些方向)
# ==============================================================================
banner("8) 生产与进阶主题")
print("""
  · 多指标评估：不只看 accuracy，配 f1/precision/recall(类别不均衡时尤其重要)。
  · 超参数调优：Optuna / Ray Tune 自动搜 lr、batch、epochs(Trainer 有 hyperparameter_search)。
  · 参数高效微调 PEFT：LoRA / AdaLoRA —— 只训极少量新增参数，省显存、能在小卡上微调大模型。
  · 量化：8bit/4bit 加载大模型(bitsandbytes)，进一步省显存。
  · 梯度检查点 gradient_checkpointing：用时间换显存(重算激活值而非全存)，能训更大模型/更长序列。
  · 分享到 Hub：trainer.push_to_hub() / model.push_to_hub("name")。
  · 实验追踪 wandb：TrainingArguments(report_to="wandb")，自动把 loss/lr/指标画成曲线、
    对比多次实验。(之前笔记里 report_to='none' 只是为了关掉它，它本来的用途是实验管理。)
""")


# ==============================================================================
# 9) 一次性整体分词为什么占内存(map 的动机) + tqdm
# ==============================================================================
banner("9) 为什么用 map 而不是一次性分词 + tqdm 进度条")
print("  · 一次性 tokenizer(整列文本) 会把所有结果一次性堆在内存里 → 大数据集直接爆内存，")
print("    而且不能配合动态填充。用 Dataset.map(batched=True) 分批处理 + 存成 Arrow(磁盘映射)，")
print("    内存友好，还能缓存(第二次秒开)。")

from tqdm.auto import tqdm
s = 0
for i in tqdm(range(2000), desc="  演示进度条", leave=False):
    s += i
print("  · tqdm 给循环加进度条：手写训练循环里 for batch in tqdm(loader) 就能看训练进度。")

print("\n✅ 全部跑完。这些是 Chapter 2 主干之外的工程细节与方法论。")
