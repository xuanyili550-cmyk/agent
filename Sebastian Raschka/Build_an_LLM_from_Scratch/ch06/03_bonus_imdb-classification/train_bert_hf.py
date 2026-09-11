# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是第 6 章的“对照实验”脚本：不使用书中从零实现的 GPT 模型，
而是直接调用 HuggingFace `transformers` 库中现成的 BERT 系列预训练模型
(DistilBERT / BERT / RoBERTa / ModernBERT / DeBERTa-v3)，
在其之上做 IMDB 影评情感分类（二分类：正面/负面）的微调，
以便与书中自研 GPT 分类微调的效果做对比。

核心流程：
1. 用 `AutoModelForSequenceClassification` 加载带分类头的预训练模型；
2. 按 `--trainable_layers` 参数冻结/解冻不同范围的参数（只训练分类头 /
   分类头+最后一个 Transformer block / 全部参数），这是“参数高效微调”
   的常见实践，能在显存和效果之间做权衡；
3. 用 `AutoTokenizer` 对 IMDB 文本做定长编码（padding + 可选 attention mask）；
4. 用标准的交叉熵损失做监督训练，并周期性评估训练/验证集上的
   loss 与准确率；
5. 训练结束后在训练/验证/测试集上分别报告最终准确率。

Source for "Build a Large Language Model From Scratch"
  - https://www.manning.com/books/build-a-large-language-model-from-scratch
Code: https://github.com/rasbt/LLMs-from-scratch
"""

import argparse
from pathlib import Path
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IMDbDataset(Dataset):
    """IMDB 影评数据集的 PyTorch `Dataset` 封装。

    负责读取 CSV 文件（需包含 "text" 和 "label" 两列），用给定的
    HuggingFace 分词器把每条文本编码为定长的 token id 序列，
    并按需生成对应的 attention mask（用于告诉模型哪些位置是真实
    token、哪些是 padding，从而在自注意力中屏蔽 padding 位置）。

    参数:
        csv_file (str | Path): 数据集 CSV 文件路径，含 "text"、"label" 两列。
        tokenizer: HuggingFace 分词器实例（如 AutoTokenizer 返回的对象）。
        max_length (int | None): 序列的目标长度（token 数）。若为 None，
            则自动扫描整个数据集，取最长编码长度作为 max_length。
        pad_token_id (int): 用于填充的 token id。默认值 50256 是 GPT-2
            分词器的 pad id，但在本脚本的实际调用中，会显式传入
            `tokenizer.pad_token_id`（BERT 系分词器通常是 0），所以
            这个默认值仅在未显式传参时才会生效，不影响实际运行结果。
        use_attention_mask (bool): 是否为每个样本生成/返回 attention mask。
            若为 False，则 `__getitem__` 会返回一个全 1 的“伪”mask
            （相当于告诉模型所有位置都参与注意力计算，包括 padding），
            用来做“不使用 attention mask”的对照实验。
    """

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, use_attention_mask=False):
        self.data = pd.read_csv(csv_file)
        # 若未显式指定 max_length，则先扫描一遍数据集算出最长编码长度，
        # 保证所有样本 padding 后长度一致，可以拼成规整的 batch 张量。
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)
        self.pad_token_id = pad_token_id
        self.use_attention_mask = use_attention_mask

        # Pre-tokenize texts and create attention masks if required
        # 预先把所有文本分词编码好（而不是在 __getitem__ 里现算），
        # 用空间换时间，加快 DataLoader 迭代速度。
        # truncation=True + max_length 保证过长文本会被截断到统一长度。
        self.encoded_texts = [
            tokenizer.encode(text, truncation=True, max_length=self.max_length)
            for text in self.data["text"]
        ]
        # 对编码结果做右侧 padding，补齐到 self.max_length，
        # 这样每个样本的 token 序列长度一致，方便 DataLoader 直接堆叠成 batch。
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))
            for et in self.encoded_texts
        ]

        if self.use_attention_mask:
            # 真实使用 attention mask：真实 token 位置标 1，padding 位置标 0，
            # 这样自注意力计算时模型不会“看到”并被 padding 干扰。
            self.attention_masks = [
                self._create_attention_mask(et)
                for et in self.encoded_texts
            ]
        else:
            self.attention_masks = None

    def _create_attention_mask(self, encoded_text):
        """根据编码后的 token 序列生成 attention mask。

        参数:
            encoded_text (list[int]): 长度为 max_length 的 token id 列表。
        返回:
            list[int]: 与输入等长的 0/1 列表，1 表示真实 token，0 表示 padding。
        """
        return [1 if token_id != self.pad_token_id else 0 for token_id in encoded_text]

    def __getitem__(self, index):
        """按索引取出一条训练样本。

        参数:
            index (int): 样本下标。
        返回:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - input_ids: 形状 (max_length,) 的 long 张量，token id 序列。
                - attention_mask: 形状 (max_length,) 的 long 张量，0/1 掩码。
                - label: 0 维 long 张量，情感标签（0=负面，1=正面）。
        """
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["label"]

        if self.use_attention_mask:
            attention_mask = self.attention_masks[index]
        else:
            # 对照实验分支：不区分 padding，全部位置 mask=1，
            # 相当于让模型把 padding token 也当作“有效输入”参与注意力计算。
            attention_mask = torch.ones(self.max_length, dtype=torch.long)

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        """返回数据集样本总数。"""
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        """遍历整个数据集，计算未截断时最长的编码 token 数。

        用于在未显式指定 max_length 时，自动确定统一的 padding 长度。

        参数:
            tokenizer: HuggingFace 分词器实例。
        返回:
            int: 数据集中最长文本的编码长度（token 数）。
        """
        max_length = 0
        for text in self.data["text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device):
    """计算单个 batch 的分类交叉熵损失。

    参数:
        input_batch (torch.Tensor): 形状 (batch_size, seq_len)，token id 序列。
        attention_mask_batch (torch.Tensor): 形状 (batch_size, seq_len)，0/1 掩码。
        target_batch (torch.Tensor): 形状 (batch_size,)，分类标签（0/1）。
        model: HuggingFace `AutoModelForSequenceClassification` 模型。
        device (torch.device): 计算设备（"cuda" 或 "cpu"）。
    返回:
        torch.Tensor: 0 维标量张量，该 batch 的平均交叉熵损失（带梯度）。
    """
    attention_mask_batch = attention_mask_batch.to(device)
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # logits = model(input_batch)[:, -1, :]  # Logits of last output token
    # 与书中自研 GPT 分类模型不同：这里不需要手动取“最后一个 token”的输出，
    # HuggingFace 的 *ForSequenceClassification 模型内部已经用池化/CLS
    # 表示等方式聚合了序列信息，直接从 .logits 拿到形状为
    # (batch_size, num_labels) 的分类得分。
    logits = model(input_batch, attention_mask=attention_mask_batch).logits
    # 二分类交叉熵：logits 形状 (batch_size, 2)，target 形状 (batch_size,)。
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


# Same as in chapter 5
def calc_loss_loader(data_loader, model, device, num_batches=None):
    """在整个（或部分）DataLoader 上计算平均损失。

    参数:
        data_loader (DataLoader): 产出 (input_ids, attention_mask, label) 三元组的加载器。
        model: 分类模型。
        device (torch.device): 计算设备。
        num_batches (int | None): 最多评估的 batch 数；None 表示遍历全部 batch。
    返回:
        float: 所评估 batch 的平均损失。
    """
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        # 如果传入的 num_batches 超过了 DataLoader 实际的 batch 数量，
        # 就取两者较小值，避免越界/多算。
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad()  # Disable gradient tracking for efficiency
# 用装饰器在函数级别关闭梯度跟踪：评估阶段不需要反向传播，
# 这样可以省显存、加快前向推理速度。
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """在（部分）DataLoader 上计算分类准确率。

    参数:
        data_loader (DataLoader): 产出 (input_ids, attention_mask, label) 三元组的加载器。
        model: 分类模型。
        device (torch.device): 计算设备。
        num_batches (int | None): 最多评估的 batch 数；None 表示遍历全部 batch。
    返回:
        float: 准确率，取值范围 [0, 1]。
    """
    model.eval()  # 切换到评估模式：关闭 dropout 等训练专用行为
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            attention_mask_batch = attention_mask_batch.to(device)
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            # logits = model(input_batch)[:, -1, :]  # Logits of last output token
            logits = model(input_batch, attention_mask=attention_mask_batch).logits
            # 形状 (batch_size, 2) -> 沿类别维取 argmax，得到每个样本的预测类别 (batch_size,)
            predicted_labels = torch.argmax(logits, dim=1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """训练过程中定期调用的“快速评估”函数：分别在训练/验证集上抽样计算损失。

    参数:
        model: 分类模型。
        train_loader (DataLoader): 训练集加载器。
        val_loader (DataLoader): 验证集加载器。
        device (torch.device): 计算设备。
        eval_iter (int): 每次评估时最多使用的 batch 数（抽样评估，节省时间）。
    返回:
        tuple[float, float]: (train_loss, val_loss)。
    """
    model.eval()  # 评估期间关闭 dropout，保证结果确定、可比较
    with torch.no_grad():  # 评估不需要梯度，进一步节省显存和计算
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 评估结束后切回训练模式，恢复 dropout 等行为
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None):
    """分类微调的主训练循环。

    参数:
        model: 待微调的分类模型（部分参数可能已被冻结，见 requires_grad 设置）。
        train_loader (DataLoader): 训练集加载器。
        val_loader (DataLoader): 验证集加载器。
        optimizer (torch.optim.Optimizer): 优化器（脚本中使用 AdamW）。
        device (torch.device): 计算设备。
        num_epochs (int): 训练轮数。
        eval_freq (int): 每隔多少个 global step 做一次快速评估并打印日志。
        eval_iter (int): 每次快速评估使用的 batch 数上限。
        max_steps (int | None): 若设置，则训练达到该 step 数后提前终止
            （便于调试时跑少量 step 验证脚本可用性）。
    返回:
        tuple: (train_losses, val_losses, train_accs, val_accs, examples_seen)
            - train_losses / val_losses (list[float]): 各次快速评估记录的损失。
            - train_accs / val_accs (list[float]): 每个 epoch 结束时的准确率。
            - examples_seen (int): 训练过程中累计见过的样本数。
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode  # 开启 dropout 等训练行为

        for input_batch, attention_mask_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration  # 清空上一步残留梯度
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients  # 反向传播计算梯度
            # 注意：只有 requires_grad=True 的参数才会被这里的反向传播
            # 计算出梯度，进而被下面的 optimizer.step() 更新——这正是
            # "冻结/解冻部分参数实现参数高效微调" 的关键机制。
            optimizer.step()  # Update model weights using loss gradients  # 用梯度更新可训练参数
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:
                break

        # New: Calculate accuracy after each epoch
        # 每个 epoch 结束后，用较多的 batch 数（eval_iter）估算训练/验证准确率，
        # 用于观察模型是否在学习、是否过拟合。
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if max_steps is not None and global_step > max_steps:
            break

    return train_losses, val_losses, train_accs, val_accs, examples_seen


if __name__ == "__main__":
    # 命令行参数解析：控制用哪个预训练模型、训练哪些层、是否用 attention mask 等。
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--trainable_layers",
        type=str,
        default="all",
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_layer'."
        )
    )
    parser.add_argument(
        "--use_attention_mask",
        type=str,
        default="true",
        help=(
            "Whether to use a attention mask for padding tokens. Options: 'true', 'false'."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert",
        help=(
            "Which model to train. Options: 'distilbert', 'bert', 'roberta', 'modernbert-base/-large', 'deberta-v3-base'."
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
        default=5e-6,
        help=(
            "Learning rate."
        )
    )
    args = parser.parse_args()

    ###############################
    # Load model
    # 加载预训练模型，并根据 --trainable_layers 冻结/解冻不同范围的参数。
    # 这是本脚本的核心：对比“只训练分类头”“训练分类头+最后一个 block”
    # “全参数微调”三种策略在下游任务上的效果差异。
    ###############################

    torch.manual_seed(123)  # 固定随机种子，保证新建分类头等随机初始化过程可复现
    if args.model == "distilbert":

        # num_labels=2 会让 HuggingFace 自动创建一个输出维度为 2 的分类头
        # （对 DistilBERT 来说就是 model.classifier），并随机初始化其权重
        # （因为 distilbert-base-uncased 本身是无分类头的预训练权重）。
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        # --- 确定性 bug 修复 ---
        # 原代码: model.out_head = torch.nn.Linear(in_features=768, out_features=2)
        # 为什么是 bug：DistilBertForSequenceClassification 的分类头属性名是
        # `classifier`（见 transformers 源码 self.classifier = nn.Linear(...)），
        # 前向传播 forward() 中用的也是 self.classifier，模型内部根本没有
        # `out_head` 这个属性/模块。原写法会把一个全新的 Linear 层挂到
        # `model.out_head` 上，但这个层从未参与前向计算，属于“死代码”；
        # 更严重的是，若 --trainable_layers=last_layer，会导致真正参与
        # 前向/反向传播的参数全部被冻结，只有这个不参与计算图的 out_head
        # 被设为可训练，最终 loss 不带梯度，训练时 loss.backward() 会报错。
        # 这里改为 model.classifier，与下方 bert/roberta/modernbert/deberta
        # 分支的写法保持一致，也让"重置分类头"的意图真正生效。
        model.classifier = torch.nn.Linear(in_features=768, out_features=2)
        for param in model.parameters():
            param.requires_grad = False  # 先冻结全部参数，再按需解冻
        if args.trainable_layers == "last_layer":
            # 同步修复：原代码引用 model.out_head.parameters()，
            # 现改为 model.classifier.parameters()，与上面的修复保持一致。
            for param in model.classifier.parameters():
                param.requires_grad = True  # 只训练最后的分类头
        elif args.trainable_layers == "last_block":
            for param in model.pre_classifier.parameters():
                param.requires_grad = True  # 训练分类头前的池化投影层
            for param in model.distilbert.transformer.layer[-1].parameters():
                param.requires_grad = True  # 再加上最后一个 Transformer block
        elif args.trainable_layers == "all":
            for param in model.parameters():
                param.requires_grad = True  # 全参数微调
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    elif args.model == "bert":

        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        # BertForSequenceClassification 的分类头属性名确实是 `classifier`，
        # 这里重新赋值等价于对分类头做一次随机重新初始化。
        model.classifier = torch.nn.Linear(in_features=768, out_features=2)
        for param in model.parameters():
            param.requires_grad = False
        if args.trainable_layers == "last_layer":
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.bert.pooler.dense.parameters():
                param.requires_grad = True  # BERT 特有的 pooler（[CLS] 表征的线性+tanh变换）
            for param in model.bert.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif args.model == "roberta":

        model = AutoModelForSequenceClassification.from_pretrained(
            "FacebookAI/roberta-large", num_labels=2
        )
        # RobertaForSequenceClassification 的分类头是一个小模块
        # RobertaClassificationHead，真正产出 logits 的是其中的 out_proj
        # 子层（dense -> tanh -> dropout -> out_proj），因此这里替换的是
        # model.classifier.out_proj，而不是整个 model.classifier。
        # roberta-large 隐藏维度是 1024，故 in_features=1024。
        model.classifier.out_proj = torch.nn.Linear(in_features=1024, out_features=2)
        for param in model.parameters():
            param.requires_grad = False
        if args.trainable_layers == "last_layer":
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.roberta.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")

    elif args.model in ("modernbert-base", "modernbert-large"):

        if args.model == "modernbert-base":
            model = AutoModelForSequenceClassification.from_pretrained(
                "answerdotai/ModernBERT-base", num_labels=2
            )
            model.classifier = torch.nn.Linear(in_features=768, out_features=2)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                "answerdotai/ModernBERT-large", num_labels=2
            )
            model.classifier = torch.nn.Linear(in_features=1024, out_features=2)
        for param in model.parameters():
            param.requires_grad = False
        if args.trainable_layers == "last_layer":
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.model.layers[-1].parameters():
                param.requires_grad = True  # ModernBERT 主干最后一个编码层
            for param in model.head.parameters():
                param.requires_grad = True  # 分类前的预测头（pre-classifier 部分）
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    elif args.model == "deberta-v3-base":
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base", num_labels=2
        )
        model.classifier = torch.nn.Linear(in_features=768, out_features=2)
        for param in model.parameters():
            param.requires_grad = False
        if args.trainable_layers == "last_layer":
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.pooler.parameters():
                param.requires_grad = True  # DeBERTa 的 ContextPooler
            for param in model.deberta.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    else:
        # --- 确定性 bug 修复 ---
        # 原代码: raise ValueError("Selected --model {args.model} not supported.")
        # 为什么是 bug：字符串前缺少 f 前缀，`{args.model}` 不会被求值插值，
        # 报错信息会原样打印花括号里的文本，而不是用户实际传入的模型名，
        # 不利于排查问题。这里补上 f 前缀使其成为真正的 f-string。
        raise ValueError(f"Selected --model {args.model} not supported.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 先设为评估模式；真正训练时 train_classifier_simple 内部会切回 model.train()

    ###############################
    # Instantiate dataloaders
    # 构建训练/验证/测试集的 DataLoader。
    ###############################

    base_path = Path(".")

    if args.use_attention_mask.lower() == "true":
        use_attention_mask = True
    elif args.use_attention_mask.lower() == "false":
        use_attention_mask = False
    else:
        raise ValueError("Invalid argument for `use_attention_mask`.")

    # max_length=256：固定序列长度上限，超过的文本会被截断，不足的会被 padding，
    # 保证同一 batch 内张量形状一致，便于批量计算。
    train_dataset = IMDbDataset(
        base_path / "train.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,  # 使用当前分词器真实的 pad id（而非默认的 GPT-2 pad id）
        use_attention_mask=use_attention_mask
    )
    val_dataset = IMDbDataset(
        base_path / "validation.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )
    test_dataset = IMDbDataset(
        base_path / "test.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )

    num_workers = 0
    batch_size = 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集需要打乱，避免样本顺序引入偏差
        num_workers=num_workers,
        drop_last=True,  # 丢弃最后不满一个 batch 的样本，保证每个 batch 大小一致
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,  # 评估阶段无需丢弃末尾样本，保证评估覆盖全部数据
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    ###############################
    # Train model
    # 用 AdamW 优化器对（部分或全部）参数做微调训练。
    ###############################

    start_time = time.time()
    torch.manual_seed(123)  # 再次固定种子，保证 DataLoader 的 shuffle 顺序可复现
    # weight_decay=0.1：AdamW 的解耦权重衰减，起到正则化、缓解过拟合的作用。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=20,
        max_steps=None
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    ###############################
    # Evaluate model
    # 训练结束后，在完整的训练/验证/测试集上做一次全量评估，
    # 得到最终、无抽样误差的准确率指标。
    ###############################

    print("\nEvaluating on the full datasets ...\n")

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
