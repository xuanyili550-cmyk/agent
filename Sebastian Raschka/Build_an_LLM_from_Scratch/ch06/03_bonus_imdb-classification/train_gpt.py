# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块：使用自实现的 GPT-2（来自前几章的 GPTModel）对 IMDB 影评数据集做
情感二分类（positive/negative）微调训练脚本。

核心思路（分类微调 vs. 预训练/指令微调的关键区别）：
1. 加载预训练好的 GPT-2 权重（也可选择随机初始化权重做对照实验）；
2. 把原来的语言模型输出头（预测下一个 token，维度 = 词表大小 50257）
   替换成一个新的线性分类头（输出维度 = 类别数 2）；
3. 默认冻结几乎全部参数，只让新分类头 + 最后一个 Transformer block（可配置）
   参与梯度更新，这样微调成本低、收敛快、且不容易在小数据集上过拟合；
4. 训练/评估时不使用整个序列的输出，而是只取「某个位置」（默认序列最后一个
   token，因为是因果注意力，最后一个位置能看到前面所有真实 token 的信息）
   或者对所有位置做平均池化，得到一个 (batch, emb_dim) 的向量喂给分类头。

以下沿用原文件的英文注释，并在此基础上补充详细中文注释。
"""

import argparse
from pathlib import Path
import time

import pandas as pd
import tiktoken
import torch
# Import Dynamo before TensorFlow is loaded by gpt_download to avoid
# Triton/TensorFlow initialization crash on Linux aarch64
# （中文：必须在 gpt_download 触发 TensorFlow 加载之前先 import torch._dynamo，
#   否则在 Linux aarch64 平台上 Triton/TensorFlow 的初始化顺序冲突会导致崩溃。
#   这里只是为了提前触发导入，本身在代码里不会被直接使用，所以加了 noqa。）
import torch._dynamo  # noqa: F401
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt


class IMDbDataset(Dataset):
    """IMDB 影评情感分类数据集封装。

    职责：
    - 从 csv 文件中读取文本列 "text" 和标签列 "label"；
    - 用给定的 tokenizer 把每条文本预先编码为 token id 序列；
    - 统一裁剪/填充（pad）到相同长度 max_length，方便批量组 batch。

    参数：
        csv_file (str | Path): 数据集 csv 文件路径，需包含 "text" 和 "label" 两列。
        tokenizer: 具备 .encode(str) -> List[int] 接口的分词器（此处用 tiktoken 的 gpt2 编码）。
        max_length (int | None): 序列统一长度。为 None 时，会自动扫描整个数据集，
            取「最长样本的编码长度」作为 max_length（见 _longest_encoded_length）。
        pad_token_id (int): 用于填充的 token id，默认用 GPT-2 的 <|endoftext|> (50256)。
            注意：由于模型使用因果注意力（每个位置只能看到自己及之前的 token），
            序列末尾填充的 pad token 不会污染前面真实 token 的表示；
            但如果分类时取的是「最后一个位置」的输出，这个位置本身可能就是 pad token，
            之所以仍然有效，是因为因果注意力让它能聚合到前面所有真实 token 的信息。
    """

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        # 若未显式指定 max_length，则动态计算数据集中最长样本的编码长度
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        # Pre-tokenize texts
        # 中文：预先把所有文本编码成 token id 列表，并截断到 max_length，
        #   避免在训练循环里重复分词、提升效率。
        #   风险提示（不改动）：若某条文本编码长度超过模型的 context_length（GPT-2 为 1024），
        #   且 max_length 设置得比 1024 还大，后续模型的位置编码 (pos_emb) 查表会越界报错，
        #   这是跨配置的使用风险，非本文件的确定性 bug，故只标注不修改。
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]
            for text in self.data["text"]
        ]
        # Pad sequences to the longest sequence
        # 中文：把每条编码后的序列在末尾补齐 pad_token_id，使所有样本长度都等于 max_length，
        #   这样 DataLoader 才能把它们堆叠（stack）成形状为 (batch, max_length) 的张量。
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))
            for et in self.encoded_texts
        ]

    def __getitem__(self, index):
        """按索引取出一条样本。

        参数：
            index (int): 样本下标。
        返回：
            (encoded, label):
                encoded: torch.LongTensor，形状 (max_length,)，token id 序列。
                label:   torch.LongTensor，标量（0 维），情感标签（0=负面, 1=正面，具体取决于数据集编码）。
        """
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        """返回数据集样本总数。"""
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        """遍历数据集中所有文本，找出编码后 token 数量最多的长度。

        参数：
            tokenizer: 分词器，需实现 .encode(str) -> List[int]。
        返回：
            int：数据集中最长样本的编码 token 数。
        """
        max_length = 0
        for text in self.data["text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def instantiate_model(choose_model, load_weights):
    """构建 GPT-2 模型实例，可选择加载 OpenAI 官方预训练权重。

    参数：
        choose_model (str): 模型规格名称，取值范围：
            "gpt2-small (124M)" / "gpt2-medium (355M)" / "gpt2-large (774M)" / "gpt2-xl (1558M)"。
        load_weights (bool): True 时下载并加载对应规格的官方预训练权重；
            False 时使用随机初始化权重（常用于做对照实验，验证预训练权重的价值）。
    返回：
        GPTModel：已构建好（并按需加载权重）的 GPT-2 模型，处于 eval() 模式。
    """

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # 中文：把具体规格（层数/头数/embedding 维度）合并进基础配置字典
    BASE_CONFIG.update(model_configs[choose_model])

    if not load_weights:
        # 中文：不加载预训练权重时，固定随机种子保证「随机初始化」这一分支的结果可复现，
        #   便于和「加载预训练权重」的分支做公平对比实验。
        torch.manual_seed(123)
    model = GPTModel(BASE_CONFIG)

    if load_weights:
        # 中文：从形如 "gpt2-small (124M)" 中提取出 "124M" 作为 gpt_download 需要的 model_size 参数
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        # 中文：把下载到的 OpenAI 官方 TensorFlow 权重按名称映射，拷贝进我们自实现的 GPTModel 中
        load_weights_into_gpt(model, params)

    model.eval()  # 中文：默认设为推理模式（外层代码会在训练时再切回 model.train()）
    return model


def calc_loss_batch(input_batch, target_batch, model, device,
                    trainable_token_pos=-1, average_embeddings=False):
    """计算单个 batch 的分类交叉熵损失。

    分类微调的关键点：模型本身仍然是逐 token 输出 (batch, seq_len, vocab_size) 的语言模型 logits，
    但这里的 model 已经把输出头替换成了二分类线性层，所以 model_output 的形状实际是
    (batch, seq_len, num_classes)。我们只需要从中取出「用于分类的那个位置」的向量。

    参数：
        input_batch (Tensor): 形状 (batch, seq_len)，token id 序列。
        target_batch (Tensor): 形状 (batch,)，真实类别标签。
        model (nn.Module): 分类头已替换的 GPT 模型。
        device (torch.device): 计算设备。
        trainable_token_pos (int): 取哪个序列位置的输出做分类，默认 -1（最后一个 token）。
        average_embeddings (bool): 若为 True，则不取单一位置，而是对所有位置的输出做平均池化。
    返回：
        Tensor：标量，交叉熵损失。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    model_output = model(input_batch)  # 形状 (batch, seq_len, num_classes)
    if average_embeddings:
        # Average over the sequence dimension (dim=1)
        # 中文：对序列维度取平均，得到 (batch, num_classes)。
        #   注意：这里会把 padding token 对应位置的输出也一并平均进去（见 IMDbDataset 的说明），
        #   属于该消融实验（"average_embeddings"）本身的设计取舍，不在此处修改。
        logits = model_output.mean(dim=1)
    else:
        # Select embeddings at the specified token position
        # 中文：只取指定位置（默认最后一个 token）的输出向量，形状 (batch, num_classes)。
        #   之所以最后一个位置可行，是因为因果注意力使其能看到前面所有真实 token 的上下文。
        logits = model_output[:, trainable_token_pos, :]

    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device,
                     num_batches=None, trainable_token_pos=-1,
                     average_embeddings=False):
    """在整个（或部分）DataLoader 上计算平均分类损失，用于训练过程中的监控评估。

    参数：
        data_loader (DataLoader): 待评估的数据加载器。
        model (nn.Module): 分类模型。
        device (torch.device): 计算设备。
        num_batches (int | None): 最多评估多少个 batch；None 表示评估全部 batch。
        trainable_token_pos (int): 同 calc_loss_batch。
        average_embeddings (bool): 同 calc_loss_batch。
    返回：
        float：平均损失（各 batch 损失的算术平均）；若 data_loader 为空则返回 float("nan")。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        # 中文：如果调用方要求评估的 batch 数超过了 data_loader 实际拥有的 batch 数，
        #   就把 num_batches 截断到实际可用的数量，避免越界。
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad()  # Disable gradient tracking for efficiency
# 中文：评估阶段不需要反向传播，用 @torch.no_grad() 装饰器关闭梯度追踪，节省显存和计算量。
def calc_accuracy_loader(data_loader, model, device,
                         num_batches=None, trainable_token_pos=-1,
                         average_embeddings=False):
    """在（部分）DataLoader 上计算分类准确率。

    参数：
        data_loader (DataLoader): 待评估的数据加载器。
        model (nn.Module): 分类模型（函数内部会强制切到 eval 模式）。
        device (torch.device): 计算设备。
        num_batches (int | None): 最多评估多少个 batch；None 表示评估全部。
        trainable_token_pos (int): 同 calc_loss_batch。
        average_embeddings (bool): 同 calc_loss_batch。
    返回：
        float：预测正确样本数 / 总评估样本数。
        风险提示（不改动）：若 data_loader 为空（num_examples 始终为 0），
        这里会发生除以 0，calc_loss_loader 对此做了 nan 保护，但本函数没有，
        属于跨场景使用风险，非本文件当前调用路径下会触发的确定性 bug，故只标注。
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            model_output = model(input_batch)  # 形状 (batch, seq_len, num_classes)
            if average_embeddings:
                # Average over the sequence dimension (dim=1)
                logits = model_output.mean(dim=1)
            else:
                # Select embeddings at the specified token position
                logits = model_output[:, trainable_token_pos, :]

            # 中文：取每个样本 logits 中最大值对应的类别下标作为预测标签，形状 (batch,)
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device, eval_iter,
                   trainable_token_pos=-1, average_embeddings=False):
    """训练过程中周期性调用的轻量评估函数：只用少量 batch（eval_iter）估计训练/验证损失。

    参数：
        model (nn.Module): 分类模型。
        train_loader / val_loader (DataLoader): 训练集/验证集加载器。
        device (torch.device): 计算设备。
        eval_iter (int): 用于快速评估的 batch 数量上限（避免每次评估都跑全量数据，拖慢训练）。
        trainable_token_pos (int): 同 calc_loss_batch。
        average_embeddings (bool): 同 calc_loss_batch。
    返回：
        (train_loss, val_loss): 两个 float，分别是训练集、验证集上的平均损失估计。
    """
    model.eval()  # 中文：评估前切到 eval 模式（关闭 dropout 等，这里 drop_rate=0 影响不大，但仍是规范做法）
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
    model.train()  # 中文：评估结束后切回训练模式，确保外层训练循环不受影响
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None, trainable_token_pos=-1,
                            average_embeddings=False):
    """分类微调的主训练循环（标准的前向 -> 反向传播 -> 参数更新流程）。

    参数：
        model (nn.Module): 待微调的分类模型（部分参数 requires_grad=False，已被冻结）。
        train_loader / val_loader (DataLoader): 训练集/验证集加载器。
        optimizer (torch.optim.Optimizer): 优化器（外部只会更新 requires_grad=True 的参数）。
        device (torch.device): 计算设备。
        num_epochs (int): 训练轮数。
        eval_freq (int): 每隔多少个 global_step 做一次快速评估（打印训练/验证损失）。
        eval_iter (int): 快速评估时使用的 batch 数上限。
        max_steps (int | None): 若设置，则训练达到该 step 数后提前终止（用于快速调试/演示）。
        trainable_token_pos (int): 分类时取哪个 token 位置的输出，同 calc_loss_batch。
        average_embeddings (bool): 是否用平均池化代替取单一位置，同 calc_loss_batch。
    返回：
        (train_losses, val_losses, train_accs, val_accs, examples_seen):
            前四项为 list，记录训练过程中各次评估/各 epoch 的指标；
            examples_seen (int) 为训练过程中累计见过的样本总数。
    """
    # Initialize lists to track losses and tokens seen
    # 中文：用于记录训练曲线的容器，方便后续画图/分析
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        # 中文：切到训练模式（启用 dropout 等；本配置 drop_rate=0，但保持规范写法）

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            # 中文：清空上一步遗留的梯度，避免梯度累加导致更新错误
            loss = calc_loss_batch(input_batch, target_batch, model, device,
                                   trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings)
            loss.backward()  # Calculate loss gradients
            # 中文：反向传播——由于大部分参数 requires_grad=False，
            #   梯度只会流向新分类头以及被显式解冻的那部分层（如最后一个 Transformer block）
            optimizer.step()  # Update model weights using loss gradients
            # 中文：optimizer 在构造时传入的是 model.parameters()，
            #   但 requires_grad=False 的参数不会被更新，实际生效的只有可训练子集
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            # 中文：分类任务按「样本数」而非「token 数」统计训练进度，与预训练脚本的统计口径不同
            global_step += 1

            # Optional evaluation step
            # 中文：global_step 从 -1 开始，第一次训练迭代后变为 0，
            #   0 % eval_freq == 0 恒成立，所以训练一开始就会先打印一次基线损失
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                    trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:
                # 中文：达到最大 step 数上限，跳出当前 epoch 的 batch 循环（提前结束本轮训练）
                break

        # New: Calculate accuracy after each epoch
        # 中文：每个 epoch 结束后（或因 max_steps 被提前打断后）用 eval_iter 个 batch 估算训练/验证准确率
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
            # 中文：同样检查一次，确保达到 max_steps 后能跳出最外层的 epoch 循环
            break

    return train_losses, val_losses, train_accs, val_accs, examples_seen


if __name__ == "__main__":
    # 中文：命令行入口。整体流程：解析参数 -> 构建/改造模型 -> 构建数据集与 DataLoader
    #   -> 训练 -> 在训练/验证/测试集上做最终评估并打印准确率。

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
            "Learning rate."
        )
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="If set, model compilation will be enabled."
    )
    args = parser.parse_args()

    # 中文：把字符串形式的 "first"/"last" 转换成实际用于索引的整数位置 0 / -1
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

    model = instantiate_model(args.model_size, load_weights)
    for param in model.parameters():
        # 中文：第一步，先把全部参数冻结（requires_grad=False），
        #   后面再根据 --trainable_layers 有选择地解冻部分层，实现「只微调一小部分参数」
        param.requires_grad = False

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
    # 中文：分类微调的核心一步——把原本输出词表分布的语言模型头，替换成一个新的线性分类头。
    #   新创建的 nn.Linear 层参数默认 requires_grad=True，所以即使前面统一冻结过一遍，
    #   这个新的 out_head 依然是可训练的（这也是 "last_layer" 模式下唯一被训练的部分）。
    model.out_head = torch.nn.Linear(in_features=in_features, out_features=2)

    if args.trainable_layers == "last_layer":
        # 中文：什么都不用做——out_head 本身就是可训练的（见上方注释），
        #   其余参数在前面已经全部冻结，因此只训练最后的分类头
        pass
    elif args.trainable_layers == "last_block":
        # 中文：额外解冻最后一个 Transformer block 以及最终的 LayerNorm（final_norm），
        #   让模型有更多容量适配下游分类任务，同时训练成本仍远小于全量微调
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
    elif args.trainable_layers == "all":
        # 中文：解冻全部参数，等价于全量微调（显存/计算开销最大，但上限也最高）
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Invalid --trainable_layers argument.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.compile:
        # 中文：启用 torch.compile 做图编译加速；先把矩阵乘法精度设为 "high"
        #   （在支持的 GPU 上用 TF32，兼顾速度与精度）
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)

    ###############################
    # Instantiate dataloaders
    ###############################

    base_path = Path(".")

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = None
    if args.context_length == "model_context_length":
        # 中文：直接用模型支持的最大上下文长度（GPT-2 为 1024）作为统一序列长度
        max_length = model.pos_emb.weight.shape[0]
    elif args.context_length == "longest_training_example":
        # 中文：先构建一次训练集（不指定 max_length），触发内部自动扫描最长样本长度，
        #   再复用这个 train_dataset，避免重复分词
        train_dataset = IMDbDataset(base_path / "train.csv", max_length=None, tokenizer=tokenizer)
        max_length = train_dataset.max_length
    else:
        try:
            # 中文：否则把 --context_length 当作用户显式指定的整数长度
            max_length = int(args.context_length)
        except ValueError:
            raise ValueError("Invalid --context_length argument")

    if train_dataset is None:
        train_dataset = IMDbDataset(base_path / "train.csv", max_length=max_length, tokenizer=tokenizer)
    # 中文：验证集/测试集必须使用和训练集相同的 max_length，保证输入张量形状一致、评估口径一致
    val_dataset = IMDbDataset(base_path / "validation.csv", max_length=max_length, tokenizer=tokenizer)
    test_dataset = IMDbDataset(base_path / "test.csv", max_length=max_length, tokenizer=tokenizer)

    num_workers = 0
    batch_size = 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        # 中文：训练集打乱顺序 + 丢弃最后不满一个 batch 的数据，
        #   避免小尾巴 batch（batch_size 过小）造成的梯度估计不稳定/BatchNorm 类问题
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        # 中文：验证/测试不需要打乱，也不丢弃最后一个 batch，保证评估覆盖全部样本
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
    torch.manual_seed(123)
    # 中文：AdamW 优化器只会更新 requires_grad=True 的参数子集（即分类头 + 按需解冻的层）；
    #   weight_decay=0.1 对权重做 L2 正则，缓解小数据集上的过拟合
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
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

    # 中文：训练结束后，在训练/验证/测试集全量数据上做一次完整评估（num_batches=None 表示不限制）
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
