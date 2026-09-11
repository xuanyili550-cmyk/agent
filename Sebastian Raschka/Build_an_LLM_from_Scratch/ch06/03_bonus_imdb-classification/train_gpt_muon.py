# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块用途（中文说明，原英文头部注释见上方，予以保留）：
使用 Muon 优化器（配合 AdamW 作为兜底优化器）微调自实现的 GPT-2 模型，
在 IMDb 数据集上做二分类（正面/负面情感分类）任务。

核心知识点——Muon 与 AdamW 的参数拆分（parameter split）：
- Muon（MomentUm Orthogonalized by Newton-schulz）优化器只适合用来更新
  "二维矩阵形状"的参数（即 `ndim == 2`），例如 Transformer 中各个线性层
  （注意力的 Q/K/V/输出投影、前馈网络的权重矩阵等）的权重矩阵。因为 Muon
  的核心思想是对参数的梯度更新量做正交化（orthogonalization），这只有在
  参数本身可以被视为一个线性变换矩阵时才有意义。
- 而词嵌入（token embedding）、位置嵌入（position embedding）等
  `nn.Embedding` 的权重，虽然形状也是二维的，但其每一行对应的是一个
  "查表"意义下的向量，并不是一个真正参与矩阵乘法的线性变换，所以不适合用
  Muon 更新，仍然要用 AdamW。
- 一维参数（如各种 bias、LayerNorm 的 weight/bias）同样不适合用 Muon，
  也归入 AdamW 分组。
因此本文件的关键函数 `create_muon_optimizers` 会遍历模型的所有可训练参数，
把"二维且非 embedding"的参数交给 Muon 优化器，其余参数交给 AdamW 优化器，
最终返回一个优化器列表，训练循环中需要对列表中的每个优化器分别调用
`zero_grad()` 和 `step()`。
"""

import argparse
from pathlib import Path
import time

import pandas as pd
import tiktoken
import torch
# Import Dynamo before TensorFlow is loaded by gpt_download to avoid
# Triton/TensorFlow initialization crash on Linux aarch64
import torch._dynamo  # noqa: F401
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt


class IMDbDataset(Dataset):
    """IMDb 影评数据集的 PyTorch `Dataset` 封装。

    从 CSV 文件（需包含 "text" 和 "label" 两列）读取影评文本与标签，
    使用传入的 `tokenizer` 将文本预先分词（tokenize）成 token id 序列，
    并统一填充（pad）到相同长度，方便后续按批次（batch）加载训练。
    """

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        参数:
            csv_file: CSV 文件路径，需包含 "text"（文本）和 "label"（标签）列。
            tokenizer: 分词器对象（如 tiktoken 的 GPT-2 编码器），需实现 `encode` 方法。
            max_length: 统一填充/截断到的最大序列长度；为 None 时自动取数据集中
                最长样本的编码长度（见 `_longest_encoded_length`）。
            pad_token_id: 用于填充（padding）的 token id，默认使用 GPT-2 的
                文本结束符（<|endoftext|>）id 50256 作为填充符。
        """
        self.data = pd.read_csv(csv_file)  # 读取 CSV 到 DataFrame
        # 若未显式指定 max_length，则动态计算数据集中最长文本的编码长度
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        # Pre-tokenize texts
        # 预先对所有文本分词，并截断到 max_length（超长部分直接丢弃）
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]
            for text in self.data["text"]
        ]
        # Pad sequences to the longest sequence
        # 对长度不足 max_length 的序列，在末尾填充 pad_token_id，使所有样本等长，
        # 这样才能被 DataLoader 直接堆叠（stack）成一个 batch 张量。
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))
            for et in self.encoded_texts
        ]

    def __getitem__(self, index):
        """按索引取出一条样本，返回 (输入 token id 张量, 标签张量)。"""
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        """返回数据集样本总数（DataLoader 需要用它来划分批次）。"""
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        """遍历所有文本，计算分词后最长的 token 序列长度，用作统一的 max_length。"""
        max_length = 0
        for text in self.data["text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def instantiate_model(choose_model, load_weights):
    """构建 GPT-2 模型实例，可选择性地加载 OpenAI 官方发布的预训练权重。

    参数:
        choose_model: 字符串，指定模型规模，取值为
            "gpt2-small (124M)" / "gpt2-medium (355M)" /
            "gpt2-large (774M)" / "gpt2-xl (1558M)" 之一。
        load_weights: 布尔值，True 表示下载并加载对应规模的 GPT-2 预训练权重，
            False 表示使用随机初始化权重（可复现地固定随机种子）。

    返回:
        实例化好的 `GPTModel`（自实现的 GPT-2 架构）。
    """

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    # 不同规模 GPT-2 模型对应的结构超参数（嵌入维度、层数、注意力头数）
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[choose_model])  # 合并出该规模模型的完整配置

    if not load_weights:
        # 若不加载预训练权重，则固定随机种子，保证随机初始化结果可复现
        torch.manual_seed(123)
    model = GPTModel(BASE_CONFIG)

    if load_weights:
        # 从模型名（如 "gpt2-small (124M)"）中提取参数量标记，如 "124M"
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        # 下载/加载 OpenAI 官方 TensorFlow 版 GPT-2 权重
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        # 将下载到的权重拷贝进自实现的 GPTModel 对应参数中
        load_weights_into_gpt(model, params)

    model.eval()  # 默认设为评估模式（后续训练时会再切回 train 模式）
    return model


def calc_loss_batch(input_batch, target_batch, model, device,
                    trainable_token_pos=-1, average_embeddings=False):
    """计算单个 batch 的分类交叉熵损失。

    参数:
        input_batch: 形状 [batch_size, seq_len] 的输入 token id。
        target_batch: 形状 [batch_size] 的分类标签（0/1）。
        model: GPT 模型，前向输出形状为 [batch_size, seq_len, num_classes]
            （因为 out_head 已被替换为二分类的 Linear 层）。
        device: 计算设备（cpu/cuda）。
        trainable_token_pos: 使用序列中哪个位置的输出向量作为分类 logits，
            默认 -1 表示取最后一个 token 位置。
        average_embeddings: 若为 True，则对序列维度取平均，得到"平均池化"后的
            logits，而不是只取某个固定位置的 token。

    返回:
        标量交叉熵损失（loss）。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    model_output = model(input_batch)
    if average_embeddings:
        # Average over the sequence dimension (dim=1)
        # 对所有 token 位置的输出做平均池化，作为分类用的 logits
        logits = model_output.mean(dim=1)
    else:
        # Select embeddings at the specified token position
        # 只取指定位置（默认是最后一个 token）的输出作为分类用的 logits
        logits = model_output[:, trainable_token_pos, :]

    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device,
                     num_batches=None, trainable_token_pos=-1,
                     average_embeddings=False):
    """在给定的数据加载器（DataLoader）上计算平均损失，用于训练过程中的监控。

    参数:
        data_loader: 待评估的 DataLoader（如训练集或验证集）。
        model: GPT 模型。
        device: 计算设备。
        num_batches: 最多评估多少个 batch；为 None 时评估整个 data_loader。
        trainable_token_pos / average_embeddings: 同 `calc_loss_batch`。

    返回:
        这些 batch 上的平均损失（float）。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")  # 空数据集时返回 NaN，避免除零错误
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        # 若指定的 num_batches 超过了 data_loader 实际的批次数，则取二者较小值
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
            )
            total_loss += loss.item()  # 累加各 batch 的损失（标量）
        else:
            break  # 达到指定批次数后提前终止，避免遍历整个 data_loader
    return total_loss / num_batches


@torch.no_grad()  # Disable gradient tracking for efficiency
# 用装饰器关闭梯度跟踪，评估阶段不需要反向传播，可节省显存/加速计算
def calc_accuracy_loader(data_loader, model, device,
                         num_batches=None, trainable_token_pos=-1,
                         average_embeddings=False):
    """在给定数据加载器上计算分类准确率（accuracy）。

    参数含义同 `calc_loss_loader`；返回值为预测正确的样本数占比（0~1 之间的浮点数）。
    """
    model.eval()  # 切换到评估模式（关闭 Dropout 等训练专用行为）
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            model_output = model(input_batch)
            if average_embeddings:
                # Average over the sequence dimension (dim=1)
                logits = model_output.mean(dim=1)
            else:
                # Select embeddings at the specified token position
                logits = model_output[:, trainable_token_pos, :]

            predicted_labels = torch.argmax(logits, dim=-1)  # 取概率最大的类别作为预测标签

            num_examples += predicted_labels.shape[0]  # 累计已评估样本数
            correct_predictions += (predicted_labels == target_batch).sum().item()  # 累计预测正确数
        else:
            break
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device, eval_iter,
                   trainable_token_pos=-1, average_embeddings=False):
    """训练过程中周期性调用：分别在训练集和验证集上各取 `eval_iter` 个 batch，
    计算平均损失，用于打印训练日志、观察是否过拟合。

    调用前后会临时把模型切到 eval 模式（关闭 Dropout），计算完再切回 train 模式，
    因为该函数是从训练循环内部被调用的，调用结束后训练要继续进行。
    """
    model.eval()
    with torch.no_grad():  # 评估时不需要梯度，关闭以节省显存、加速计算
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
    model.train()  # 恢复训练模式，继续后续训练
    return train_loss, val_loss


def create_muon_optimizers(model, adamw_learning_rate, muon_learning_rate, weight_decay=0.1):
    """核心函数：把模型的可训练参数拆分为两组，分别交给 Muon 与 AdamW 优化器。

    拆分规则（也是本文件最重要的知识点）：
        1. 只有 `requires_grad=True` 的参数才会被纳入任一优化器（被冻结的参数
           不会出现在任何优化器的参数组里，自然也就不会被更新）。
        2. 在可训练参数中，凡是形状为二维（`param.ndim == 2`）且不属于
           `nn.Embedding` 权重的参数（也就是 Transformer 内部各种线性层的权重
           矩阵，包括本文件里新建的分类头 `out_head`），交给 Muon 优化器处理；
           Muon 对这类"矩阵状"参数做正交化更新，收敛速度和稳定性通常优于 AdamW。
        3. 其余参数——包括 `nn.Embedding` 的权重（tok_emb / pos_emb，这些虽然
           也是二维张量，但语义上是"查表"而非线性变换，不适合正交化）、
           以及所有一维参数（各种 bias、LayerNorm 的 weight/bias）——交给
           AdamW 优化器处理。
        4. 若某一组参数为空（例如所有可训练参数都是一维的），则不会为空的那一组
           创建优化器，避免 PyTorch 优化器因为传入空参数列表而报错。

    参数:
        model: 待微调的模型（GPTModel）。
        adamw_learning_rate: AdamW 优化器的学习率。
        muon_learning_rate: Muon 优化器的学习率（通常可以比 AdamW 设置得更大）。
        weight_decay: 两个优化器共用的权重衰减系数。

    返回:
        一个优化器列表（最多包含一个 Muon 实例和一个 AdamW 实例）；训练循环中
        需要遍历这个列表，对每个优化器分别调用 `zero_grad()` / `step()`。

    风险提示（不修改，仅标注上报）：
        - `torch.optim.Muon` 截至本次审阅时并非所有 PyTorch 版本都自带（属于
          较新/实验性的优化器实现），代码已用 `hasattr` 做了可用性检查并在
          不可用时抛出明确的 RuntimeError，属于合理的防御性写法，但如果你的
          PyTorch 版本没有该类，训练会直接在此处失败，需要升级 PyTorch。
        - `adjust_lr_fn="match_rms_adamw"` 是传给 `torch.optim.Muon` 构造函数
          的一个具体实现细节参数，其名称/取值属于该优化器的公开 API 一部分，
          不同 PyTorch 版本之间可能发生变化，如遇到 `TypeError: unexpected
          keyword argument` 之类的报错，需要查阅对应版本的官方文档确认参数名。
    """
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon is not available. Please update to a more recent PyTorch version.")

    # 第一步：收集所有 nn.Embedding 模块（tok_emb、pos_emb 等）里参数的"完整名字"
    # （形如 "tok_emb.weight"），后面按名字排除这些参数，不让它们进入 Muon 分组。
    embedding_param_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            for param_name, _ in module.named_parameters(recurse=False):
                # 顶层模块（module_name 为空字符串）时不加前缀点号，否则拼成 "父模块名.参数名"
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                embedding_param_names.add(full_name)

    # 第二步：遍历模型所有具名参数，按照上面的规则做分组
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳过被冻结的参数，它们不需要出现在任何优化器里
        if param.ndim == 2 and name not in embedding_param_names:
            # 二维矩阵、且不是 embedding 权重 -> 交给 Muon
            muon_params.append(param)
        else:
            # 一维参数（bias/LayerNorm）或 embedding 权重 -> 交给 AdamW
            adamw_params.append(param)

    # 第三步：分别为两组参数创建对应的优化器实例，只有非空的分组才创建优化器
    optimizers = []
    if muon_params:
        optimizers.append(
            torch.optim.Muon(
                muon_params, lr=muon_learning_rate, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw"
            )
        )
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=adamw_learning_rate, weight_decay=weight_decay))
    if not optimizers:
        # 两组都为空意味着模型没有任何可训练参数，属于配置错误，提前报错更清晰
        raise ValueError("No trainable parameters found.")
    return optimizers


def train_classifier_simple(model, train_loader, val_loader, optimizers, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None, trainable_token_pos=-1,
                            average_embeddings=False):
    """标准的分类微调训练循环，支持同时使用多个优化器（此处是 Muon + AdamW）。

    与常见单优化器版本的关键区别：`optimizers` 是一个优化器列表，每个训练
    step 都需要对列表中的每一个优化器分别调用 `zero_grad()`（清空上一步的梯度）
    和 `step()`（用当前梯度更新对应参数分组）；因为一次 `loss.backward()`
    会把梯度同时填充到 Muon 和 AdamW 各自负责的参数上，两个优化器各自只更新
    自己名下的那部分参数，互不干扰。

    参数:
        model: 待训练模型。
        train_loader / val_loader: 训练集/验证集的 DataLoader。
        optimizers: 优化器列表（如 [Muon优化器, AdamW优化器]）。
        device: 计算设备。
        num_epochs: 训练轮数。
        eval_freq: 每隔多少个训练 step 打印一次训练/验证损失。
        eval_iter: 每次评估时最多使用多少个 batch（加速评估，不用跑完整个数据集）。
        max_steps: 可选的最大训练步数上限，达到后提前结束训练。
        trainable_token_pos / average_embeddings: 同 `calc_loss_batch`。

    返回:
        (train_losses, val_losses, train_accs, val_accs, examples_seen)
        分别是各次评估记录下来的训练/验证损失列表、每个 epoch 结束后的训练/
        验证准确率列表，以及总共见过的样本数。
    """
    # Initialize lists to track losses and tokens seen
    # 用于记录训练过程中的损失和准确率，便于训练结束后绘图/分析
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1  # global_step 从 -1 开始，第一次自增后变为 0

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode  # 切换到训练模式（启用 Dropout 等）

        for input_batch, target_batch in train_loader:
            for optimizer in optimizers:
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
                # 对 Muon 和 AdamW 两个优化器分别清空各自负责参数的梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device,
                                   trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings)
            loss.backward()  # Calculate loss gradients
            # 反向传播只需要调用一次；梯度会自动填充到所有 requires_grad=True 的参数上，
            # 不管这些参数最终归属于 Muon 还是 AdamW 分组
            for optimizer in optimizers:
                optimizer.step()  # Update model weights using loss gradients
                # 每个优化器只会更新自己参数组里的参数（Muon 更新二维权重矩阵，
                # AdamW 更新 embedding/bias/LayerNorm 等参数）
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                    trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # 【风险标注，未修改】：此处判断条件为 `global_step > max_steps`，
            # 而 global_step 是从 0 开始计数的训练步序号，因此当 max_steps=N 时，
            # 实际会多执行一步才会跳出循环（是一个"差一"/off-by-one 的写法）。
            # 由于本文件调用处（见文末 `__main__`）始终传入 max_steps=None，
            # 这条分支实际不会被触发，不影响当前脚本的运行结果；且该写法与同目录
            # 下 train_gpt.py 保持一致，属于历史遗留的边界写法而非本次改动引入的
            # 问题，按“风险项不改只标注”的原则，这里不做修改，如需精确控制步数，
            # 使用时请注意该边界行为。
            if max_steps is not None and global_step > max_steps:
                break

        # New: Calculate accuracy after each epoch
        # 每个 epoch 结束后，额外计算一次训练集/验证集准确率（用 eval_iter 个 batch 近似）
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if max_steps is not None and global_step > max_steps:
            break

    return train_losses, val_losses, train_accs, val_accs, examples_seen


if __name__ == "__main__":

    # 以下为命令行参数定义（argparse），用于控制模型规模、权重来源、
    # 冻结/训练哪些层、序列长度、学习率等训练配置。
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model_size",
        type=str,
        default="gpt2-small (124M)",
        help=(
            "Which GPT model to use. Options: 'gpt2-small (124M)', 'gpt2-medium (355M)',"
            " 'gpt2-large (774M)', 'gpt2-xl (1558M)'."
        )
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrained",
        help=(
            "Whether to use 'pretrained' or 'random' weights."
        )
    )
    parser.add_argument(
        "--trainable_layers",
        type=str,
        default="last_block",
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_layer'."
        )
    )
    parser.add_argument(
        "--trainable_token_pos",
        type=str,
        default="last",
        help=(
            "Which token to train. Options: 'first', 'last'."
        )
    )
    parser.add_argument(
        "--average_embeddings",
        action="store_true",
        default=False,
        help=(
            "Average the output embeddings from all tokens instead of using"
            " only the embedding at the token position specified by `--trainable_token_pos`."
        )
    )
    parser.add_argument(
        "--context_length",
        type=str,
        default="256",
        help=(
            "The context length of the data inputs."
            "Options: 'longest_training_example', 'model_context_length' or integer value."
        )
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help=(
            "Number of epochs."
        )
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=(
            "Learning rate for AdamW parameters."
        )
    )
    parser.add_argument(
        "--muon_learning_rate",
        type=float,
        default=1e-4,
        help=(
            "Learning rate for Muon parameters."
        )
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="If set, model compilation will be enabled."
    )
    args = parser.parse_args()

    # 把可读的字符串参数 "first"/"last" 转换成实际用于张量索引的位置下标
    if args.trainable_token_pos == "first":
        args.trainable_token_pos = 0
    elif args.trainable_token_pos == "last":
        args.trainable_token_pos = -1
    else:
        raise ValueError("Invalid --trainable_token_pos argument")

    ###############################
    # Load model
    ###############################

    if args.weights == "pretrained":
        load_weights = True
    elif args.weights == "random":
        load_weights = False
    else:
        raise ValueError("Invalid --weights argument.")

    # 加载/构建 GPT-2 模型（原本用于语言建模，vocab_size=50257 的输出头）
    model = instantiate_model(args.model_size, load_weights)
    # 先冻结所有参数，后面再按 --trainable_layers 的设置有选择地解冻
    for param in model.parameters():
        param.requires_grad = False

    # 根据模型规模确定嵌入维度，用于构建新的分类头 out_head
    if args.model_size == "gpt2-small (124M)":
        in_features = 768
    elif args.model_size == "gpt2-medium (355M)":
        in_features = 1024
    elif args.model_size == "gpt2-large (774M)":
        in_features = 1280
    elif args.model_size == "gpt2-xl (1558M)":
        in_features = 1600
    else:
        raise ValueError("Invalid --model_size argument")

    torch.manual_seed(123)
    # 把原本用于语言建模（预测下一个 token，输出维度 = 词表大小）的 out_head，
    # 替换为一个二分类的线性层（新建层默认 requires_grad=True，天然是可训练的）
    model.out_head = torch.nn.Linear(in_features=in_features, out_features=2)

    # 根据 --trainable_layers 决定解冻哪些层参与微调
    if args.trainable_layers == "last_layer":
        pass  # 只训练新建的分类头 out_head（前面已默认可训练），其余层保持冻结
    elif args.trainable_layers == "last_block":
        # 额外解冻最后一个 Transformer block 以及最终的 LayerNorm
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
    elif args.trainable_layers == "all":
        # 解冻全部参数，做全量微调（此时 embedding 也会被训练，
        # create_muon_optimizers 中排除 embedding 的逻辑就会真正起作用）
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Invalid --trainable_layers argument.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.compile:
        # 启用 torch.compile 加速；先设置矩阵乘法精度为 "high" 以获得更好的性能/精度折中
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)

    ###############################
    # Instantiate dataloaders
    ###############################

    base_path = Path(".")  # 假定 train/validation/test 的 csv 文件位于当前工作目录

    tokenizer = tiktoken.get_encoding("gpt2")  # GPT-2 使用的 BPE 分词器

    train_dataset = None
    if args.context_length == "model_context_length":
        # 使用模型支持的最大上下文长度（位置编码的行数）作为序列长度
        max_length = model.pos_emb.weight.shape[0]
    elif args.context_length == "longest_training_example":
        # 先构建训练集以求得训练集中最长样本的长度，再用它作为统一长度
        train_dataset = IMDbDataset(base_path / "train.csv", max_length=None, tokenizer=tokenizer)
        max_length = train_dataset.max_length
    else:
        try:
            max_length = int(args.context_length)  # 直接使用用户指定的整数长度
        except ValueError:
            raise ValueError("Invalid --context_length argument")

    # 若上面还没有构建训练集（即不是 "longest_training_example" 分支），这里补建
    if train_dataset is None:
        train_dataset = IMDbDataset(base_path / "train.csv", max_length=max_length, tokenizer=tokenizer)
    val_dataset = IMDbDataset(base_path / "validation.csv", max_length=max_length, tokenizer=tokenizer)
    test_dataset = IMDbDataset(base_path / "test.csv", max_length=max_length, tokenizer=tokenizer)

    num_workers = 0
    batch_size = 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,       # 训练集需要打乱顺序
        num_workers=num_workers,
        drop_last=True,     # 丢弃最后不足一个 batch 的样本，保证每个 batch 大小一致
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,    # 验证/测试无需打乱、也无需丢弃末尾不完整 batch
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    ###############################
    # Train model
    ###############################

    start_time = time.time()
    torch.manual_seed(123)  # 固定随机种子，保证 DataLoader 的 shuffle 顺序等可复现
    # 关键调用：按 Muon/AdamW 参数拆分规则，构建两个优化器
    optimizers = create_muon_optimizers(
        model, adamw_learning_rate=args.learning_rate, muon_learning_rate=args.muon_learning_rate, weight_decay=0.1
    )

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizers, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=20,
        max_steps=None, trainable_token_pos=args.trainable_token_pos,
        average_embeddings=args.average_embeddings
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    ###############################
    # Evaluate model
    ###############################

    print("\nEvaluating on the full datasets ...\n")

    train_accuracy = calc_accuracy_loader(
        train_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    val_accuracy = calc_accuracy_loader(
        val_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    test_accuracy = calc_accuracy_loader(
        test_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")