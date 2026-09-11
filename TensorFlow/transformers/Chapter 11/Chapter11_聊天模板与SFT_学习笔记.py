"""
================================================================================
 Chapter 11 · 聊天模板 + 监督微调(SFT) —— 学习笔记（可直接运行）
================================================================================
 配套「Fine-tuning LLMs」章前半：聊天模板 与 SFT 概念。与同目录的
 Chapter11_LoRA微调实战.py(真训练闭环)配套 —— 这份讲“喂给 LLM 的数据长什么样、
 SFT 在干嘛”，那份讲“怎么用 LoRA 高效训”。

 直接运行：python3 Chapter11_聊天模板与SFT_学习笔记.py
 只下小的“分词器文件”(SmolLM2 / Qwen 的 tokenizer，很小)，不下大模型。

 目录：
   1  messages 格式：LLM 对话的标准数据结构(system/user/assistant/tool)
   2  apply_chat_template：把 messages 变成模型认识的带特殊标记的字符串
   3  不同模型模板不一样(ChatML / Llama / Mistral…)，别手写、用分词器自带的
   4  add_generation_prompt：推理时留个“该助手说话了”的开头
   5  SFT 是什么、什么时候要做
   6  SFT 训练配置的关键参数
   7  TRL 的 SFTTrainer 骨架 + packing + formatting_func
================================================================================
"""

from transformers import AutoTokenizer


def banner(t):
    print("\n" + "=" * 72 + f"\n {t}\n" + "=" * 72)


# ==============================================================================
# 1) messages 格式：对话的标准数据结构
# ==============================================================================
banner("1) messages 格式（现代 LLM 对话的标准输入）")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "What's the weather?"},
]
print("  一段对话 = 一个 messages 列表，每条含 role + content：")
for m in messages:
    print(f"    {m['role']:10} : {m['content']}")
print("  role 有 system(设定人设/规则) / user(用户) / assistant(模型回复) / tool(工具返回)。")


# ==============================================================================
# 2) apply_chat_template：messages → 模型认识的字符串
# ==============================================================================
banner("2) apply_chat_template：别手拼特殊标记，让分词器按模型模板转")
# 每个 instruct 模型有自己的“对话模板”(哪里加 <|im_start|>、system 怎么放…)，
# 分词器 tokenizer.chat_template 里存着，apply_chat_template 会照它拼。
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
text = tok.apply_chat_template(messages, tokenize=False)
print("  SmolLM2 模板格式化结果(tokenize=False 只看字符串)：\n")
print(text)
# tokenize=True 直接出编码(本版返回 BatchEncoding 字典，取 input_ids)
enc = tok.apply_chat_template(messages, tokenize=True)
print(f"  tokenize=True → 直接得到 {len(enc['input_ids'])} 个 input_ids(可喂模型)")


# ==============================================================================
# 3) 不同模型模板不一样：同一段对话，两个模型格式化出来不同
# ==============================================================================
banner("3) 不同模型的模板不同（所以千万别写死，用各自分词器的）")
simple = [{"role": "system", "content": "You are helpful."},
          {"role": "user", "content": "Hi"}]
for name in ["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]:
    try:
        t = AutoTokenizer.from_pretrained(name)
        formatted = t.apply_chat_template(simple, tokenize=False)
        print(f"  【{name.split('/')[-1]}】")
        print("   ", repr(formatted[:130]))
    except Exception as e:
        print(f"  (跳过 {name}: {type(e).__name__})")
print("\n  各家用不同特殊标记：ChatML 用 <|im_start|>/<|im_end|>；Llama 用 <|start_header_id|>；")
print("  Mistral 用 [INST][/INST]。手拼极易错 → 永远用 apply_chat_template。")


# ==============================================================================
# 4) add_generation_prompt：推理时留个“轮到助手”的开头
# ==============================================================================
banner("4) add_generation_prompt：推理时必加")
infer_msgs = [{"role": "system", "content": "You are helpful."},
              {"role": "user", "content": "Name a color."}]
without = tok.apply_chat_template(infer_msgs, tokenize=False)
withgen = tok.apply_chat_template(infer_msgs, tokenize=False, add_generation_prompt=True)
print("  不加(训练用，对话已完整)：", repr(without[-40:]))
print("  加了(推理用，末尾留出助手开头让模型接着生成)：", repr(withgen[-40:]))
print("  规则：训练时对话是“完整的一问一答”不用加；推理时要加，告诉模型“该你(助手)说了”。")


# ==============================================================================
# 5) SFT 是什么、什么时候要做（文字）
# ==============================================================================
banner("5) 监督微调 SFT 是什么、何时需要")
print("""
  SFT(Supervised Fine-Tuning)：用“指令→理想回复”的成对数据，教基座模型按你要的方式回答。
  预训练模型只会“续写”，SFT 把它变成“会听指令、按格式答”的助手。
  什么时候需要 SFT：
    · 模板/格式控制：要求固定输出结构(JSON/特定聊天格式/风格统一)。
    · 领域自适应：教它领域术语、专业规范、行业话术(医疗/法律/客服)。
    · 行为对齐：让它遵循特定准则、拒答某些内容。
  不需要 SFT 的情况：通用任务用现成 instruct 模型 + 好的提示词(prompt)往往就够了。
  SFT 之后常接 LoRA(见 Chapter11_LoRA微调实战.py)—— 用 LoRA 做 SFT = 省显存的高效微调。
""")


# ==============================================================================
# 6) SFT 训练配置的关键参数（文字）
# ==============================================================================
banner("6) SFT 关键训练参数")
print("""
  训练时长：num_train_epochs(训几轮) 或 max_steps(训多少步，二选一)。多→学得好但易过拟合。
  批大小  ：per_device_train_batch_size(单卡batch) + gradient_accumulation_steps(累积模拟大batch)。
            大batch梯度更稳但吃显存；显存不够就靠累积。
  学习率  ：learning_rate(更新幅度) + warmup(前若干步慢慢升)。高→不稳，低→学得慢。
            LoRA 通常用比全量微调更大的 lr(如 2e-4)。
  监测    ：logging_steps(多久记日志) / eval_steps(多久评估) / save_steps(多久存档)。
""")


# ==============================================================================
# 7) TRL 的 SFTTrainer 骨架（真训练在 LoRA 实战文件里，这里给最小骨架）
# ==============================================================================
banner("7) TRL SFTTrainer 骨架")
print('''
  from trl import SFTConfig, SFTTrainer
  from peft import LoraConfig
  args = SFTConfig(output_dir="out", max_steps=100, per_device_train_batch_size=4,
                   learning_rate=2e-4, dataset_text_field="text", max_length=512)
  trainer = SFTTrainer(
      model="HuggingFaceTB/SmolLM2-135M",
      args=args,
      train_dataset=ds,                 # 有 "text" 列，或 "messages" 列(自动套 chat 模板)
      peft_config=LoraConfig(r=8, lora_alpha=16, task_type="CAUSAL_LM"),  # 传了就是 LoRA 微调
  )
  trainer.train()

  两个提效开关：
   · packing=True     把多条短样本拼进一个定长序列，减少 padding 浪费、吞吐更高。
   · formatting_func  自定义把结构化字段拼成训练文本：
       def fmt(ex): return f"### Question: {ex['question']}\\n ### Answer: {ex['answer']}"
''')

print("✅ 全部跑完。聊天模板部分是真跑(apply_chat_template)，SFT 训练闭环见 LoRA 实战文件。")
