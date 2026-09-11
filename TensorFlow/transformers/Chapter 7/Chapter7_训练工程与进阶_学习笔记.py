"""
================================================================================
 Chapter 7 · 训练工程 + 任务进阶（补齐审计发现的缺口）—— 可直接运行
================================================================================
 这是对 Chapter7_主要NLP任务_学习笔记.py 的补充。之前那份讲清了「每个任务独有的
 预处理/标签/评估逻辑」，但漏了两大块，本文件专门补：
   B 类·训练工程(跨任务通用)：Trainer vs 自定义循环、Accelerate、优化器、学习率
        调度+warmup、梯度累积、混合精度fp16、梯度裁剪、分布式评估、push_to_hub
   A 类·任务进阶(每个任务的高级变体)：全词掩码、从零建模、keytoken加权损失、
        QA 验证集预处理、NER aggregation_strategy 四策略

 直接运行：
     python3 Chapter7_训练工程与进阶_学习笔记.py
 ★ 亮点：第 B1 节是一个**真能在 Mac 上跑起来的迷你训练循环**(bert-tiny + 合成数据，
   几秒训完)，把 Accelerate/调度器/梯度累积/评估 全串起来，让你看它 loss 真的在降。
 需要下的都是超小模型(bert-tiny ~17MB / distilgpt2 tokenizer / bert-cased tokenizer)。

 记忆主线：
   微调 = 数据(DataLoader+collator) → 前向(loss) → 反向(backward) → 更新(optimizer.step)
          → 调学习率(scheduler.step) → 清梯度(zero_grad)，循环。
   Trainer 把这套封装成一行 trainer.train()；自定义循环把它摊开，方便你插自定义逻辑。
================================================================================
"""

import numpy as np
import torch


def banner(t):
    print("\n" + "=" * 72 + f"\n {t}\n" + "=" * 72)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ##############################################################################
# B 类 · 训练工程（跨任务通用）
# ##############################################################################
banner("B 类 · 训练工程：Trainer vs 自定义循环 + 真·迷你训练循环")

# ------------------------------------------------------------------------------
# B0 两条路线：Trainer(封装) vs 自定义训练循环(摊开)
# ------------------------------------------------------------------------------
# · Trainer + TrainingArguments：一行 trainer.train()，自动管日志/评估/保存/多卡/fp16。
#   适合标准微调，代码最少。Chapter 7 每个任务都先给了 Trainer 版。
# · 自定义训练循环 + 🤗 Accelerate：把每一步摊开，方便插自定义损失/日志/梯度技巧。
#   适合非标准需求(如 keytoken 加权损失、边训边推特殊逻辑)。
# 下面 B1 用「自定义循环」跑一遍真训练，因为它能让你看清每一步在干嘛。

# ------------------------------------------------------------------------------
# B1 ★真·迷你训练循环（bert-tiny 做二分类，几秒训完，loss 真的会降）
# ------------------------------------------------------------------------------
def mini_training_loop():
    from transformers import (AutoTokenizer, BertConfig,
                              BertForSequenceClassification,
                              DataCollatorWithPadding, get_scheduler)
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    try:
        from accelerate import Accelerator
        use_accel = True
    except ImportError:
        use_accel = False

    # 合成一个“情感二分类”小数据集(正面含 good/love，负面含 bad/hate)，无需下载数据
    pos = ["i love this", "so good", "great and wonderful", "a fantastic movie",
           "really love it", "good good good", "best ever", "wonderful experience",
           "i am happy", "amazing and good", "love love love", "very nice and good"]
    neg = ["i hate this", "so bad", "terrible and awful", "a horrible movie",
           "really hate it", "bad bad bad", "worst ever", "awful experience",
           "i am sad", "boring and bad", "hate hate hate", "very poor and bad"]
    texts = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)

    # 只用 bert-base-uncased 的“快速分词器”(已缓存)，模型则用 BertConfig 从零造一个
    # 迷你 BERT(2层/128维)——完全不下模型权重，任务是“关键词情感”很简单，几秒就学会。
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig(vocab_size=tok.vocab_size, hidden_size=128, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=512,
                        max_position_embeddings=64, num_labels=2)
    model = BertForSequenceClassification(config)     # 权重随机，下面真训练

    # 手动组一个 dataset(list[dict])，再用 DataCollatorWithPadding 动态填充
    enc = tok(texts, truncation=True)
    dataset = [{"input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "labels": labels[i]} for i in range(len(texts))]
    collator = DataCollatorWithPadding(tokenizer=tok)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=5e-3)     # 迷你模型+简单任务，lr 可大点
    num_epochs = 20                                     # 从零练，多给几轮才够更新步数
    grad_accum = 2                                       # ← 梯度累积：每 2 个 batch 才更新一次
    num_update_steps = (len(train_loader) // grad_accum) * num_epochs
    # 学习率调度：linear 线性衰减 + warmup(前 10% 步从 0 慢慢升到 lr，防一开始就爆)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                 num_warmup_steps=max(1, num_update_steps // 10),
                                 num_training_steps=num_update_steps)

    if use_accel:
        accelerator = Accelerator()   # 自动选设备(含 mps)、管分布式/混合精度
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    else:
        model.to(pick_device())

    def backward(loss):
        accelerator.backward(loss) if use_accel else loss.backward()

    print(f"  设备={'accelerate:'+str(accelerator.device) if use_accel else pick_device()}"
          f"  样本={len(texts)}  梯度累积={grad_accum}  更新步数={num_update_steps}")
    model.train()
    step = 0
    for epoch in range(num_epochs):
        running = 0.0
        for i, batch in enumerate(train_loader):
            if not use_accel:
                batch = {k: v.to(pick_device()) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum      # 累积时先除，等价于把多个小 batch 当一个大 batch
            backward(loss)
            running += loss.item() * grad_accum
            # 每累积够 grad_accum 个 batch，才真正更新一次参数
            if (i + 1) % grad_accum == 0:
                # 梯度裁剪：把梯度范数限制在 1.0 内，防梯度爆炸(尤其大模型/长训练)
                if use_accel:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()          # 用梯度更新参数
                lr_scheduler.step()       # 调整学习率(走 warmup/衰减曲线)
                optimizer.zero_grad()     # 清空梯度，准备下一轮
                step += 1
        if epoch % 4 == 0 or epoch == num_epochs - 1:   # 每4轮打印一次，别刷屏
            print(f"  epoch {epoch:2}: 平均loss={running/len(train_loader):.4f}  "
                  f"当前lr={lr_scheduler.get_last_lr()[0]:.2e}")

    # 简单评估：拿训练里没强调的新句子看模型学没学会
    model.eval()
    dev = accelerator.device if use_accel else pick_device()
    tests = ["this is good", "this is bad", "i love it", "i hate it"]
    t = tok(tests, padding=True, truncation=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        pred = (accelerator.unwrap_model(model) if use_accel else model)(**t).logits.argmax(-1)
    print("  评估:", [(s, "正面" if p == 1 else "负面") for s, p in zip(tests, pred.tolist())])
    print("  要点：loss 整体下降(中间偶有波动很正常) + 4 句判断正确 = 这套“循环骨架”真的在学。")


mini_training_loop()

# ------------------------------------------------------------------------------
# B2 训练工程其余关键点（代码模式 + 为什么）
# ------------------------------------------------------------------------------
print("""
[B2] 其余训练工程要点（Chapter 7 各任务反复用到）：
  · TrainingArguments 关键项：
      eval_strategy/save_strategy="epoch"(每轮评估/保存)、weight_decay(权重衰减防过拟合)、
      fp16=True(混合精度，现代GPU上更快更省显存)、gradient_accumulation_steps(显存不够时
      用“小batch多步累积”模拟大batch)、lr_scheduler_type="cosine"、warmup_steps。
  · Seq2SeqTrainingArguments 多一个 predict_with_generate=True：
      翻译/摘要评估时要真正 generate() 出文本再算 BLEU/ROUGE，而不是只看 loss。
  · push_to_hub / Repository / get_full_repo_name：边训边把模型推到 Hub(需登录+联网)：
      trainer.push_to_hub()  或  自定义循环里 repo.push_to_hub(commit_message=..., blocking=False)
  · 分布式评估要先对齐再收集(多卡时)：
      accelerator.pad_across_processes(preds, dim=1, pad_index=-100)  # 各卡形状补一致
      accelerator.gather(preds)                                       # 再把各卡结果汇总
    不这么做，多卡评估会因形状不一致报错或卡死。
  · AdamW 是标配优化器；显存紧张可用 8-bit Adam(bitsandbytes)；大模型用更低 lr(1e-5~3e-5)。
""")


# ##############################################################################
# A 类 · 任务进阶（每个任务的高级变体）
# ##############################################################################
banner("A 类 · 任务进阶：全词掩码 / 从零建模 / keytoken损失 / QA验证映射 / 聚合策略")

# ------------------------------------------------------------------------------
# A1 MLM 全词掩码(whole word masking)：整个词一起遮，而不是遮半个子词
# ------------------------------------------------------------------------------
# 普通 MLM 随机遮 token，可能只遮 'token'→['tok','##en'] 里的 '##en'，太好猜。
# 全词掩码：用 word_ids 把子词按“单词”分组，随机选中某个词就把它的所有子词一起遮。
def whole_word_masking_demo():
    import collections
    from transformers import AutoTokenizer, default_data_collator
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text = "the tokenization process is fantastic"
    enc = tok(text)
    word_ids = enc.word_ids()
    # 建“单词序号 → 该词的 token 下标列表”映射
    mapping = collections.defaultdict(list)
    cur_word, cur_idx = None, -1
    for idx, wid in enumerate(word_ids):
        if wid is not None:
            if wid != cur_word:
                cur_idx += 1
                cur_word = wid
            mapping[cur_idx].append(idx)
    # 随机选 20% 的“单词”整词遮住
    rng = np.random.RandomState(0)
    mask = rng.binomial(1, 0.3, (len(mapping),))
    input_ids = enc["input_ids"][:]
    for word_id in np.where(mask)[0]:
        for idx in mapping[word_id]:            # 该词的每个子词都换成 [MASK]
            input_ids[idx] = tok.mask_token_id
    print("[A1] 全词掩码：", tok.decode(input_ids))
    print("     对比：普通掩码可能只遮 '##en' 半个词；全词掩码把整个词的子词一起遮，更难猜、学得更好。")


whole_word_masking_demo()

# ------------------------------------------------------------------------------
# A2 从零建模：AutoConfig + GPT2LMHeadModel(config)（“从零训练”的立身之本）
# ------------------------------------------------------------------------------
# 微调 = from_pretrained(加载别人训好的权重)；从零训练 = 只借架构、权重全随机初始化，自己练。
def from_scratch_model_demo():
    from transformers import AutoConfig, GPT2LMHeadModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    # 借 gpt2 的“架构配置”，但改小上下文长度、对齐词表大小；注意是 (config) 不是 from_pretrained
    config = AutoConfig.from_pretrained("gpt2", vocab_size=len(tok), n_ctx=128,
                                        n_positions=128,
                                        bos_token_id=tok.bos_token_id or 0,
                                        eos_token_id=tok.eos_token_id or 0)
    model = GPT2LMHeadModel(config)          # ← 权重全随机！这是一个“待训练的新模型”
    size = sum(t.numel() for t in model.parameters())
    print(f"[A2] 从零造的 GPT-2：{size/1e6:.1f}M 参数，权重随机初始化(不是预训练)。")
    print("     from_pretrained=加载权重(微调)；GPT2LMHeadModel(config)=只借架构从零练。")


from_scratch_model_demo()

# ------------------------------------------------------------------------------
# A3 keytoken 加权损失的真实现（不是权重列表示意，是逐 token 的交叉熵 + 移位对齐）
# ------------------------------------------------------------------------------
# 因果 LM 用“第 t 个 token 预测第 t+1 个”，所以要把 logits 和 labels 各错一位对齐。
# 想让模型更重视某些关键 token(如 plt/pd)，就给含这些 token 的样本的 loss 加权。
def keytoken_loss_demo():
    from torch.nn import CrossEntropyLoss
    torch.manual_seed(0)
    keytoken_ids = [7]                         # 假装 id=7 是“关键 token”
    # 故意构造：第0条含 2 个关键token(7)，第1条一个都没有 → 演示权重差异
    inputs = torch.tensor([[7, 3, 7, 5, 2, 9], [1, 3, 4, 5, 2, 9]])
    logits = torch.randn(2, 6, 50)            # 模型输出(词表 50)(假数据)
    # ★移位对齐：用前 n-1 个预测后 n-1 个
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    loss_fct = CrossEntropyLoss(reduction="none")            # 要逐 token 的 loss
    per_token = loss_fct(shift_logits.view(-1, 50), shift_labels.view(-1))
    per_sample = per_token.view(shift_logits.size(0), -1).mean(axis=1)   # 每条样本平均 loss
    # 数每条样本里出现了多少个关键 token → 出现越多，权重越大
    weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(axis=[0, 2])
    weights = 1.0 + weights                                   # alpha=1
    weighted = (per_sample * weights).mean()
    print(f"[A3] keytoken 加权损失：每样本loss={per_sample.tolist()} 权重={weights.tolist()} "
          f"→ 加权后={weighted.item():.3f}")
    print("     要点：CrossEntropyLoss(reduction='none') 拿逐token损失 + 移位对齐 + 按关键token计数加权。")


keytoken_loss_demo()

# ------------------------------------------------------------------------------
# A4 QA 验证集专用预处理：offset 非 context 位置置 None + example_id 映射回原样本
# ------------------------------------------------------------------------------
# 训练时标 start/end 下标；但“评估时”一条长文被 stride 切成多个 feature，算完分要能
# 归组回原始问题。所以验证预处理要：①记住每个 feature 属于哪个 example ②把问题/特殊
# 标记位置的 offset 置 None(后面选答案时只在 context 里找)。
def qa_validation_preprocess_demo():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    questions = ["When completed?", "Where?"]
    contexts = ["The tower was completed in 1889 in Paris.", "It stands in Paris, France."]
    ids = ["q0", "q1"]
    enc = tok(questions, contexts, max_length=32, truncation="only_second", stride=8,
              return_overflowing_tokens=True, return_offsets_mapping=True,
              padding="max_length")
    sample_map = enc.pop("overflow_to_sample_mapping")   # 每个 feature 来自第几条样本
    example_ids = []
    for i in range(len(enc["input_ids"])):
        example_ids.append(ids[sample_map[i]])           # ← 把 feature 映射回原 example id
        seq_ids = enc.sequence_ids(i)
        # 把“非 context(seq_id!=1)”位置的 offset 置 None → 选答案时天然跳过问题/[SEP]/[PAD]
        enc["offset_mapping"][i] = [
            o if seq_ids[k] == 1 else None for k, o in enumerate(enc["offset_mapping"][i])]
    print(f"[A4] QA验证预处理：{len(questions)} 条问题切成 {len(enc['input_ids'])} 个 feature，"
          f"example_id={example_ids}")
    print("     每个 feature 的 offset 里，非 context 位置已置 None(选答案时只在原文 context 里找)。")


qa_validation_preprocess_demo()

# ------------------------------------------------------------------------------
# A5 NER aggregation_strategy 四策略（把子词得分合成实体得分的 4 种算法）
# ------------------------------------------------------------------------------
# pipeline("token-classification", aggregation_strategy=?) 决定“实体的分数”怎么从子词算：
def aggregation_strategy_demo():
    # 假设 'Sylvain' 切成 4 个子词，各自的“是人名”的分数如下：
    subword_scores = {"S": 0.99, "##yl": 0.85, "##va": 0.90, "##in": 0.95}
    scores = list(subword_scores.values())
    print("[A5] 子词分数:", subword_scores)
    print(f"     simple/average : 各子词平均 = {np.mean(scores):.3f}")
    print(f"     first          : 取第一个子词 = {scores[0]:.3f}")
    print(f"     max            : 取最高 = {max(scores):.3f}")
    print("     (simple 与 average 在单实体上一样；多词实体如 'Hugging Face' 时 average 按单词平均)")
    print("     用法: pipeline('token-classification', aggregation_strategy='simple')")


aggregation_strategy_demo()

print("\n✅ 全部跑完。B1 是真训练(loss 会降)；其余是各任务进阶变体的可跑演示。")
