# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块对应《从零构建大语言模型》一书附录 D（Appendix D）的内容。

附录 D 在第 5 章基础训练循环之上，增加了三项常见的训练稳定性/效率增强手段：
    1. 学习率预热（Linear Warmup）：训练初期学习率从一个很小的初始值
       线性增大到设定的峰值学习率，避免训练刚开始时（参数随机初始化、
       梯度估计不稳定）使用过大的学习率导致训练发散。
    2. 余弦退火（Cosine Annealing）：预热结束后，学习率按余弦函数的
       形状从峰值逐渐衰减到一个很小的最小学习率，相比阶梯式衰减更平滑，
       有助于模型在训练后期更精细地收敛。
    3. 梯度裁剪（Gradient Clipping）：对参数梯度的整体范数设置上限，
       防止出现梯度爆炸（尤其是在学习率较大或序列较长时），从而提升
       训练稳定性。

本文件提供的 train_model 函数是对第 5 章 ch05.py 中同名训练循环的增强版，
额外记录并返回每一步的学习率变化（track_lrs），便于绘图观察预热+余弦退火
的学习率调度曲线。
"""

from .ch05 import calc_loss_batch, evaluate_model, generate_and_print_sample

import math
import torch


def find_highest_gradient(model):
    """
    遍历模型所有参数，找出当前梯度张量中数值最大的那个元素。

    该函数常用于训练过程中的调试/监控：在反向传播（loss.backward()）之后、
    优化器更新参数（optimizer.step()）之前调用，可以观察梯度的最大幅值，
    从而判断是否存在梯度爆炸的迹象（例如数值异常增大，远超正常范围）。

    参数:
        model: 一个 torch.nn.Module 实例，其参数应已经通过反向传播
               计算出了 .grad（梯度）。

    返回:
        max_grad: 一个标量张量（0维 Tensor），表示所有参数梯度中的
                  最大值；如果模型所有参数都没有梯度（例如尚未反向传播），
                  则返回 None。
    """
    # 用于记录当前遍历到的最大梯度值，初始为 None 表示尚未找到任何梯度
    max_grad = None
    for param in model.parameters():
        # 并非所有参数都一定有梯度（例如被冻结/不需要梯度的参数），需要判空
        if param.grad is not None:
            # 将梯度张量展平为一维向量，形状从任意维度 (*,) 变为 (N,)，
            # 其中 N 为该参数张量中元素的总个数，便于直接取最大值
            grad_values = param.grad.data.flatten()
            # 取该参数梯度中的最大值（标量张量）
            max_grad_param = grad_values.max()
            # 与目前记录的全局最大值比较，更新全局最大梯度
            if max_grad is None or max_grad_param > max_grad:
                max_grad = max_grad_param
    return max_grad


def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6, orig_book_version=False):
    """
    带学习率预热 + 余弦退火 + 梯度裁剪的增强版训练循环（对应附录 D）。

    相比第 5 章基础版 train_model_simple，本函数在每一个训练步（global_step）
    都会动态计算并设置当前的学习率：
        - 在预热阶段（global_step < warmup_steps）：学习率从 initial_lr
          线性增长到 optimizer 中设置的峰值学习率 peak_lr。
        - 预热结束后：学习率按余弦函数从 peak_lr 平滑衰减到 min_lr。
    同时，在预热阶段结束之后对梯度进行范数裁剪（max_norm=1.0），
    防止较大学习率下出现梯度爆炸。

    参数:
        model: 待训练的语言模型（如 GPTModel），需实现前向传播接口，
               其输出的 logits 形状通常为 (batch_size, seq_len, vocab_size)。
        train_loader: 训练集 DataLoader，每次迭代产出 (input_batch, target_batch)，
               二者形状均为 (batch_size, seq_len)。
        val_loader: 验证集 DataLoader，用于周期性评估模型效果。
        optimizer: 优化器（如 AdamW），其 param_groups[0]["lr"] 被视为
               本次训练的“峰值学习率”（即预热阶段结束后达到的学习率）。
        device: 训练所用的设备（"cpu"、"cuda" 或 "mps" 等）。
        n_epochs: 训练的总轮数（epoch 数）。
        eval_freq: 每隔多少个训练步（global_step）评估一次训练/验证损失。
        eval_iter: 评估时使用的批次数量（用于估计平均损失，减少评估开销）。
        start_context: 用于生成样例文本的起始提示字符串，便于人工观察
               训练过程中模型生成质量的变化。
        tokenizer: 分词器，用于将 start_context 编码为 token id，
               以及将模型生成的 token id 解码回文本。
        warmup_steps: 学习率线性预热所持续的步数。
        initial_lr: 预热阶段开始时的初始学习率（默认 3e-5），
               通常远小于峰值学习率。
        min_lr: 余弦退火阶段结束时的最小学习率（默认 1e-6）。
        orig_book_version: 是否使用书中最初版本的梯度裁剪触发条件
               （global_step > warmup_steps，会导致预热结束后紧接的
               那一步跳过裁剪）。默认为 False，使用修正后的条件
               （global_step >= warmup_steps）。

    返回:
        train_losses: 每次评估时记录的训练集损失列表。
        val_losses: 每次评估时记录的验证集损失列表。
        track_tokens_seen: 每次评估时累计已处理的 token 总数列表，
               可用作绘图时的横轴（训练进度）。
        track_lrs: 每个训练步（global_step）对应的学习率列表，
               用于绘制学习率随训练步变化的调度曲线（预热+余弦退火形状）。
    """

    # 分别用于记录：训练损失、验证损失、累计处理的 token 数、每步学习率
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    # tokens_seen：累计处理过的 token 总数；global_step：全局训练步计数器，
    # 初始为 -1 是因为下面循环体一开始就会 +1，这样第一步的 global_step 恰好为 0
    tokens_seen, global_step = 0, -1

    # Retrieve the maximum learning rate from the optimizer
    # 从优化器当前设置中取出学习率，作为预热阶段结束后要达到的“峰值学习率”
    peak_lr = optimizer.param_groups[0]["lr"]

    # Calculate the total number of iterations in the training process
    # 训练总步数 = 每个 epoch 的 batch 数 × epoch 数，
    # 用于后续计算余弦退火阶段的进度比例（progress）
    total_training_steps = len(train_loader) * n_epochs

    # Calculate the learning rate increment during the warmup phase
    # 预热阶段每一步学习率的线性增量：从 initial_lr 到 peak_lr 均匀分成
    # warmup_steps 份，每步增加 lr_increment
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(n_epochs):
        # 切换为训练模式（启用 Dropout 等训练专用行为）
        model.train()
        for input_batch, target_batch in train_loader:
            # 每个训练步开始前清空上一步残留的梯度，避免梯度累积
            optimizer.zero_grad()
            global_step += 1

            # Adjust the learning rate based on the current phase (warmup or cosine annealing)
            # 根据当前所处的训练阶段（预热 or 余弦退火）动态计算本步应使用的学习率
            if global_step < warmup_steps:
                # Linear warmup
                # 预热阶段：学习率随步数线性增长，起点 initial_lr，
                # 每步增加 lr_increment，warmup_steps 步后恰好到达 peak_lr
                lr = initial_lr + global_step * lr_increment
            else:
                # Cosine annealing after warmup
                # progress 表示预热结束后，当前步在“剩余训练步数”中的相对进度，
                # 取值范围约为 [0, 1]（0 表示刚结束预热，1 表示训练即将结束）
                progress = ((global_step - warmup_steps) /
                            (total_training_steps - warmup_steps))
                # 余弦退火公式：学习率在 [min_lr, peak_lr] 区间内按余弦曲线平滑下降。
                # 当 progress=0 时 cos(0)=1，lr=min_lr+(peak_lr-min_lr)*1=peak_lr（即预热结束时的峰值）；
                # 当 progress=1 时 cos(pi)=-1，lr=min_lr+(peak_lr-min_lr)*0=min_lr（训练结束时的最小值）
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # Apply the calculated learning rate to the optimizer
            # 将计算出的学习率写回优化器的每一个参数组（大多数情况下只有一个参数组）
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # Store the current learning rate  # 记录当前步的学习率，供后续绘图分析调度曲线

            # Calculate and backpropagate the loss
            # 前向传播计算损失：input_batch/target_batch 形状均为 (batch_size, seq_len)，
            # calc_loss_batch 内部会计算模型输出 logits (batch_size, seq_len, vocab_size)
            # 与 target_batch 之间的交叉熵损失（标量）
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 反向传播，计算模型所有可训练参数的梯度（.grad）
            loss.backward()

            # Apply gradient clipping after the warmup phase to avoid exploding gradients
            # 预热阶段结束之后才启用梯度裁剪：因为预热阶段学习率本身很小，
            # 梯度更新幅度有限，不易发生梯度爆炸；而进入余弦退火阶段后学习率
            # 处于较高水平（尤其是刚结束预热时接近峰值学习率），此时更需要
            # 通过裁剪梯度整体的 L2 范数（max_norm=1.0）来防止参数更新过猛
            if orig_book_version:
                if global_step > warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                if global_step >= warmup_steps:  # the book originally used global_step > warmup_steps, which led to a skipped clipping step after warmup
                    # 修正版：使用 >= 而非 >，避免预热刚结束的第一步（global_step == warmup_steps）
                    # 被意外跳过梯度裁剪（书中原始写法的一个小疏漏，这里已修正）
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 使用（可能已裁剪的）梯度和当前学习率更新模型参数
            optimizer.step()
            # 累加本批次处理的 token 总数：input_batch.numel() = batch_size * seq_len
            tokens_seen += input_batch.numel()

            # Periodically evaluate the model on the training and validation sets
            # 每隔 eval_freq 步评估一次模型，便于监控训练/验证损失的变化趋势
            if global_step % eval_freq == 0:
                # evaluate_model 内部会临时切换为 eval 模式，
                # 分别在训练集和验证集上各采样 eval_iter 个批次计算平均损失，
                # 之后再切回 train 模式（具体实现见 ch05.py）
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # Print the current losses
                # 打印当前 epoch、训练步数、训练/验证损失，便于实时观察训练进度
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        # Generate and print a sample from the model to monitor progress
        # 每个 epoch 结束后，用当前模型基于 start_context 生成一段文本并打印，
        # 通过定性观察生成质量来直观感受模型训练效果的演进
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    # 返回训练/验证损失曲线、已处理 token 数曲线、学习率调度曲线，
    # 供调用方（如笔记本/脚本）绘图分析
    return train_losses, val_losses, track_tokens_seen, track_lrs
