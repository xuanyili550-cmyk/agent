# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块中文说明:
使用 HuggingFace `transformers` 库中的预训练 BERT 系列模型
(distilbert / bert / roberta)微调做垃圾短信(spam)二分类任务,
作为与本书第 6 章「从零手写 GPT 做分类微调」方案的对照实验(baseline/对比基准)。

整体流程:
1. 根据命令行参数选择底座模型(distilbert-base-uncased / bert-base-uncased /
   FacebookAI/roberta-large),并将其分类头(classifier)替换成一个新的随机初始化
   的线性层,再按 `--trainable_layers` 的取值冻结/解冻不同范围的参数
   (只训练最后一层 / 只训练最后一个 Transformer block / 全部参数训练)。
2. 下载并解压 UCI SMS Spam Collection 数据集,构造成 ham(0)/spam(1) 的
   平衡二分类数据集,并切分成 train/validation/test 三个 csv 文件。
3. 用 HuggingFace tokenizer 对文本做定长 padding/截断编码,构造
   PyTorch `Dataset` / `DataLoader`。
4. 用标准的训练循环(前向传播 -> 交叉熵损失 -> 反向传播 -> 参数更新)微调模型,
   并周期性地在训练/验证集上评估 loss 和准确率。
5. 训练结束后在完整的 train/validation/test 数据集上评估最终准确率。

Source for "Build a Large Language Model From Scratch"
  - https://www.manning.com/books/build-a-large-language-model-from-scratch
Code: https://github.com/rasbt/LLMs-from-scratch
"""

import argparse
import os
from pathlib import Path
import time
import requests
import zipfile

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SpamDataset(Dataset):
    """
    中文说明:
    这是第 5/6 章「从零手写 GPT」版本中使用的垃圾短信数据集类,保留在本文件中
    仅作对照参考,本脚本的训练/评估流程实际使用的是下面的 `SPAMDataset`
    (注意类名大小写不同),因此这个类在本文件中并未被实例化/调用,属于
    保留下来的历史/对照代码,不影响本脚本的实际运行结果。

    与 `SPAMDataset` 的主要区别:
    - 使用 GPT 风格的 `pad_token_id`(默认 50256,即 GPT-2 的 <|endoftext|>)。
    - 不生成 attention_mask(因为原始 GPT-from-scratch 实现里没有用到)。
    - `no_padding=True` 时可以关闭定长 padding,只做截断编码。
    """

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, no_padding=False):
        """
        中文说明:读取 csv 文件并对文本列做预分词(tokenize),可选是否做定长 padding。

        参数:
            csv_file: 数据集 csv 文件路径,需包含 "Text" 和 "Label" 两列。
            tokenizer: 用于编码文本的分词器对象(需提供 .encode 方法)。
            max_length: 统一的序列长度上限;为 None 时自动取数据集中最长编码序列的长度。
            pad_token_id: 用于填充(padding)的 token id。
            no_padding: 若为 True,则不对编码结果做 padding,仅做长度截断。
        """
        self.data = pd.read_csv(csv_file)
        # 若未显式指定 max_length,则自动扫描整个数据集,取编码后最长的序列长度
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        # Pre-tokenize texts
        # 中文:预先对所有文本做分词编码,并按 max_length 截断(超长部分丢弃)
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]
            for text in self.data["Text"]
        ]

        if not no_padding:
            # Pad sequences to the longest sequence
            # 中文:将所有编码序列用 pad_token_id 填充到统一长度 max_length,
            # 这样才能拼成规整的张量批次(batch)。
            self.encoded_texts = [
                et + [pad_token_id] * (self.max_length - len(et))
                for et in self.encoded_texts
            ]

    def __getitem__(self, index):
        """中文说明:按索引取出一条样本,返回 (编码后的输入张量, 标签张量)。"""
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        """中文说明:返回数据集样本总数,供 DataLoader 使用。"""
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        """中文说明:遍历所有文本,计算分词编码后最长的序列长度,用作默认的 max_length。"""
        max_length = 0
        for text in self.data["Text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
        # Note: A more pythonic version to implement this method
        # is the following, which is also used in the next chapter:
        # return max(len(encoded_text) for encoded_text in self.encoded_texts)
        # 中文:上面英文注释是原作者留下的提示——更 pythonic 的写法是直接对
        # 已经算好的 self.encoded_texts 取长度最大值,下一章也是这么实现的。


def download_and_unzip(url, zip_path, extract_to, new_file_path):
    """
    中文说明:从给定 url 下载 zip 压缩包,解压后把里面的 SMSSpamCollection 文件
    重命名为带 .tsv 后缀的新文件名,便于后续用 pandas 按制表符读取。
    若目标文件已存在,则直接跳过下载和解压(避免重复下载)。

    参数:
        url: 数据集 zip 包的下载地址。
        zip_path: 下载下来的 zip 文件保存路径。
        extract_to: 解压目标目录。
        new_file_path: 重命名后的最终数据文件路径。
    """
    if new_file_path.exists():
        print(f"{new_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    # 中文:以流式(stream=True)方式下载文件,避免大文件一次性占用过多内存;
    # timeout=60 防止请求无限期挂起。
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Unzipping the file
    # 中文:解压 zip 包到指定目录
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # Renaming the file to indicate its format
    # 中文:解压出来的原始文件名为 "SMSSpamCollection"(无后缀,内容是 tab 分隔),
    # 这里重命名成带 .tsv 后缀的新文件名,方便识别文件格式。
    original_file = Path(extract_to) / "SMSSpamCollection"
    os.rename(original_file, new_file_path)
    print(f"File downloaded and saved as {new_file_path}")


def random_split(df, train_frac, val_frac):
    """
    中文说明:将 DataFrame 随机打乱后,按给定比例切分成 train/validation/test
    三部分(test 部分为剩余比例 1 - train_frac - val_frac)。

    参数:
        df: 待切分的 DataFrame。
        train_frac: 训练集比例。
        val_frac: 验证集比例。
    返回:
        (train_df, val_df, test_df) 三个切分后的 DataFrame。
    """
    # Shuffle the entire DataFrame
    # 中文:固定 random_state=123 以保证结果可复现
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    # 中文:根据比例计算切分下标
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)

    # Split the DataFrame
    # 中文:按下标切片得到三个子集
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df


def create_dataset_csvs(new_file_path):
    """
    中文说明:读取原始的 tab 分隔短信数据(label + text),构造类别平衡的数据集
    (ham 样本数下采样到与 spam 样本数相同),将标签映射为数值(ham=0, spam=1),
    随机切分成 train/validation/test 后分别保存为 csv 文件到当前工作目录。

    参数:
        new_file_path: 原始 tsv 数据文件路径。
    """
    df = pd.read_csv(new_file_path, sep="\t", header=None, names=["Label", "Text"])

    # Create balanced dataset
    # 中文:spam 样本数量远少于 ham,这里对 ham 做下采样(不放回抽样),
    # 使得两个类别的样本数量相等,构成类别平衡的数据集,避免分类器偏向多数类。
    n_spam = df[df["Label"] == "spam"].shape[0]
    ham_sampled = df[df["Label"] == "ham"].sample(n_spam, random_state=123)
    balanced_df = pd.concat([ham_sampled, df[df["Label"] == "spam"]])
    balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    # Sample and save csv files
    # 中文:按 70%/10%/20% 的比例切分为 train/validation/test,并落盘为 csv
    train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    val_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)


class SPAMDataset(Dataset):
    """
    中文说明:
    本脚本实际使用的垃圾短信数据集类(注意与上面的 `SpamDataset` 类名大小写不同,
    是两个独立的类)。与 GPT 版本的实现相比,这里额外支持生成 attention_mask,
    并使用 HuggingFace tokenizer 自带的 truncation/max_length 参数做截断,
    以适配 HuggingFace 预训练模型(distilbert/bert/roberta)的输入格式。
    """

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, use_attention_mask=False):
        """
        中文说明:读取 csv 文件,对文本做定长编码(截断+padding),
        并可选地生成对应的 attention_mask(标记哪些位置是真实 token,哪些是 padding)。

        参数:
            csv_file: 数据集 csv 文件路径,需包含 "Text" 和 "Label" 两列。
            tokenizer: HuggingFace 分词器,需支持 tokenizer.encode(text, truncation=True, max_length=...)。
            max_length: 统一序列长度;为 None 时自动取数据集中最长编码序列的长度。
            pad_token_id: 用于填充(padding)的 token id,应使用 tokenizer.pad_token_id。
            use_attention_mask: 是否生成 attention_mask 并在前向传播中使用。
        """
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)
        self.pad_token_id = pad_token_id
        self.use_attention_mask = use_attention_mask

        # Pre-tokenize texts and create attention masks if required
        # 中文:预先对所有文本做分词编码,truncation=True 配合 max_length
        # 让 tokenizer 自己完成截断(比手动切片更安全,能正确处理特殊 token)。
        self.encoded_texts = [
            tokenizer.encode(text, truncation=True, max_length=self.max_length)
            for text in self.data["Text"]
        ]
        # 中文:将编码结果统一 padding 到 max_length,便于组成规整的 batch 张量
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))
            for et in self.encoded_texts
        ]

        if self.use_attention_mask:
            # 中文:若启用 attention_mask,则预先为每条样本生成对应的 mask
            self.attention_masks = [
                self._create_attention_mask(et)
                for et in self.encoded_texts
            ]
        else:
            self.attention_masks = None

    def _create_attention_mask(self, encoded_text):
        """中文说明:根据编码序列生成 attention_mask,真实 token 位置为 1,padding 位置为 0。"""
        return [1 if token_id != self.pad_token_id else 0 for token_id in encoded_text]

    def __getitem__(self, index):
        """
        中文说明:按索引返回一条样本 (输入张量, attention_mask 张量, 标签张量)。
        若未启用 use_attention_mask,则返回全 1 的 mask(等价于告诉模型所有位置都是有效 token,
        包括 padding 部分——这在语义上并不完全正确,但当 use_attention_mask=False 时
        属于用户主动选择的行为,不属于 bug)。
        """
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]

        if self.use_attention_mask:
            attention_mask = self.attention_masks[index]
        else:
            attention_mask = torch.ones(self.max_length, dtype=torch.long)

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        """中文说明:返回数据集样本总数,供 DataLoader 使用。"""
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        """中文说明:遍历所有文本,计算分词编码后最长的序列长度,用作默认的 max_length。"""
        max_length = 0
        for text in self.data["Text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device):
    """
    中文说明:计算单个 batch 的交叉熵损失。
    将输入、attention_mask、标签都搬到目标设备(CPU/GPU)上,
    调用 HuggingFace 模型前向传播得到 logits,再与真实标签计算交叉熵损失。
    """
    attention_mask_batch = attention_mask_batch.to(device)
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # logits = model(input_batch)[:, -1, :]  # Logits of last output token
    # 中文:上面这行注释是从「从零手写 GPT」版本遗留下来的写法(取最后一个 token 的
    # logits 作为分类结果);而 HuggingFace 的序列分类模型内部已经通过 pooling
    # (如取 [CLS] token 或池化层)得到了句子级别的 logits,不需要手动取最后一个 token。
    logits = model(input_batch, attention_mask=attention_mask_batch).logits
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


# Same as in chapter 5
# 中文:与第 5 章中的实现相同,用于在若干个 batch 上计算平均损失(用于评估阶段)
def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    中文说明:在给定的 DataLoader 上累积计算平均损失。
    num_batches 用于限制只评估前若干个 batch(加速验证过程),
    若为 None 则遍历整个 data_loader。
    """
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        # 中文:若传入的 num_batches 超过了 data_loader 实际的 batch 总数,
        # 则取两者较小值,避免除零或索引越界之类的问题。
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad()  # Disable gradient tracking for efficiency
# 中文:装饰器禁用梯度追踪,评估阶段不需要反向传播,这样能节省显存/内存并加速计算
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    中文说明:在给定的 DataLoader 上计算分类准确率(正确预测数 / 总样本数)。
    num_batches 用于限制只评估前若干个 batch,None 表示遍历整个 data_loader。
    """
    model.eval()  # 中文:切换到评估模式(关闭 dropout 等训练专用行为)
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
            # 中文:取每个样本 logits 中最大值对应的类别下标,作为预测标签
            predicted_labels = torch.argmax(logits, dim=1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    中文说明:在训练过程中周期性调用,分别在训练集和验证集上计算(部分 batch 的)
    平均损失,用于监控训练进度和是否过拟合。计算完成后会把模型切回训练模式。

    参数:
        eval_iter: 每次评估时最多使用多少个 batch(而不是遍历整个数据集,节省时间)。
    返回:
        (train_loss, val_loss)
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None):
    """
    中文说明:标准的监督分类训练循环。

    对每个 epoch:
        对每个训练 batch: 前向传播计算损失 -> 反向传播 -> 优化器更新参数;
        每隔 eval_freq 步在训练/验证集上做一次快速评估(loss);
        每个 epoch 结束后额外计算一次训练/验证集准确率。

    参数:
        model: 待训练的 HuggingFace 分类模型。
        train_loader / val_loader: 训练/验证集的 DataLoader。
        optimizer: 优化器(本脚本中为 AdamW)。
        device: 训练设备(cpu/cuda)。
        num_epochs: 训练轮数。
        eval_freq: 每隔多少个训练 step 做一次 loss 评估。
        eval_iter: 每次评估最多使用多少个 batch。
        max_steps: 可选的最大训练步数上限,用于快速调试(达到后提前终止训练)。
    返回:
        (train_losses, val_losses, train_accs, val_accs, examples_seen)
    """
    # Initialize lists to track losses and tokens seen
    # 中文:初始化用于记录训练过程中损失和已训练样本数的列表/计数器
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    # 中文:主训练循环,按 epoch 遍历
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        # 中文:切换到训练模式(启用 dropout 等)

        for input_batch, attention_mask_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            # 中文:清空上一个 batch 遗留的梯度,避免梯度累积错误
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            # 中文:反向传播计算梯度
            optimizer.step()  # Update model weights using loss gradients
            # 中文:根据梯度更新模型参数(只有 requires_grad=True 的参数才会被更新)
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            # 中文:累加已训练过的样本数量(而不是像语言模型那样统计 token 数)
            global_step += 1

            # Optional evaluation step
            # 中文:可选的周期性评估步骤,每 eval_freq 个 step 打印一次训练/验证 loss
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:
                # 中文:若达到设置的最大训练步数,则提前跳出内层 batch 循环(用于快速调试)
                break

        # New: Calculate accuracy after each epoch
        # 中文:每个 epoch 结束后,额外计算一次训练/验证准确率(基于 eval_iter 个 batch 的采样估计)
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if max_steps is not None and global_step > max_steps:
            # 中文:同样地,达到最大步数上限时提前跳出外层 epoch 循环
            break

    return train_losses, val_losses, train_accs, val_accs, examples_seen


if __name__ == "__main__":
    # 中文:脚本入口。整体分为四个阶段:
    # 1) 解析命令行参数; 2) 加载/配置预训练模型; 3) 构造数据集与 DataLoader;
    # 4) 训练模型并在完整数据集上评估最终效果。

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
            "Which model to train. Options: 'distilbert', 'bert', 'roberta'."
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
    # 中文:加载并配置预训练模型
    ###############################

    torch.manual_seed(123)
    if args.model == "distilbert":

        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        # 【bug 修复】原代码: model.out_head = torch.nn.Linear(in_features=768, out_features=2)
        # 为什么是 bug: HuggingFace 的 DistilBertForSequenceClassification 内部并没有
        # `out_head` 这个属性,其 forward() 方法实际调用的是 `self.classifier`
        # (经 `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)`
        # 构造时已自动创建为 nn.Linear(768, 2))。原代码里的 `out_head` 属性名是从
        # 本书「从零手写 GPT」版本的模型代码(该模型确实有名为 out_head 的输出层)
        # 直接照搬过来的,在这里只是新建了一个从未被前向传播使用的悬空子模块。
        # 由于下面 "last_layer" 分支只解冻了 model.out_head 的参数,而真正参与推理、
        # 需要学习分类边界的 model.classifier 却一直被冻结在随机初始化状态,
        # 这会导致 --trainable_layers last_layer 模式下分类头永远学不到东西、
        # 训练/验证准确率停留在接近随机猜测的水平。修复方式:
        # 将新建的线性层直接赋值给真正被使用的 `model.classifier` 属性,
        # 与下方 bert / roberta 分支的写法保持一致。
        model.classifier = torch.nn.Linear(in_features=768, out_features=2)
        for param in model.parameters():
            param.requires_grad = False
        if args.trainable_layers == "last_layer":
            # 中文:仅解冻(训练)新替换的分类头(此处已修复为 model.classifier)
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":
            # 中文:解冻 pre_classifier(分类头前的全连接层)以及
            # distilbert 编码器最后一个 Transformer block 的参数。
            # 注意:此分支未显式解冻 model.classifier 本身,这一点与下方
            # bert/roberta 分支的 "last_block" 逻辑不完全一致(那两个分支
            # 会同时解冻各自的 classifier)。这属于模型间实现不对称的风险点,
            # 是否需要统一取决于作者原始设计意图,这里按"只修复确定性 bug、
            # 不擅自改动其他逻辑"的原则不做修改,仅作标注,供后续review确认。
            for param in model.pre_classifier.parameters():
                param.requires_grad = True
            for param in model.distilbert.transformer.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            # 中文:解冻全部参数,即全量微调(full fine-tuning)
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    elif args.model == "bert":

        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        # 中文:用新的线性层替换分类头(等价于重新随机初始化最后一层)
        model.classifier = torch.nn.Linear(in_features=768, out_features=2)
        for param in model.parameters():
            param.requires_grad = False
        if args.trainable_layers == "last_layer":
            # 中文:只训练分类头
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":
            # 中文:训练分类头 + pooler 全连接层 + 编码器最后一个 Transformer 层
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.bert.pooler.dense.parameters():
                param.requires_grad = True
            for param in model.bert.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            # 中文:全量微调
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif args.model == "roberta":

        model = AutoModelForSequenceClassification.from_pretrained(
            "FacebookAI/roberta-large", num_labels=2
        )
        # 中文:roberta 的分类头是一个包含 out_proj 子层的结构,这里替换其中的
        # out_proj 为新的线性层(输入维度 1024 对应 roberta-large 的隐藏层维度)
        model.classifier.out_proj = torch.nn.Linear(in_features=1024, out_features=2)
        for param in model.parameters():
            param.requires_grad = False
        if args.trainable_layers == "last_layer":
            # 中文:只训练分类头(包含 dense + out_proj 等子层)
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":
            # 中文:训练分类头 + 编码器最后一个 Transformer 层
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.roberta.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            # 中文:全量微调
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
    else:
        raise ValueError("Selected --model {args.model} not supported.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ###############################
    # Instantiate dataloaders
    # 中文:构造数据集和 DataLoader
    ###############################

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extract_to = "sms_spam_collection"
    new_file_path = Path(extract_to) / "SMSSpamCollection.tsv"

    base_path = Path(".")
    file_names = ["train.csv", "validation.csv", "test.csv"]
    all_exist = all((base_path / file_name).exists() for file_name in file_names)

    if not all_exist:
        # 中文:若 train/validation/test 三个 csv 尚不存在,则下载原始数据并重新切分生成
        try:
            download_and_unzip(url, zip_path, extract_to, new_file_path)
        except (requests.exceptions.RequestException, TimeoutError) as e:
            # 中文:主下载源失败时(网络错误/超时),自动尝试备用镜像地址
            print(f"Primary URL failed: {e}. Trying backup URL...")
            backup_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
            download_and_unzip(backup_url, zip_path, extract_to, new_file_path)
        create_dataset_csvs(new_file_path)

    if args.use_attention_mask.lower() == "true":
        use_attention_mask = True
    elif args.use_attention_mask.lower() == "false":
        use_attention_mask = False
    else:
        raise ValueError("Invalid argument for `use_attention_mask`.")

    # 中文:分别为训练/验证/测试集构造 Dataset,max_length 固定为 256,
    # 使用所选模型自带的 tokenizer 及其 pad_token_id 做填充
    train_dataset = SPAMDataset(
        base_path / "train.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )
    val_dataset = SPAMDataset(
        base_path / "validation.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )
    test_dataset = SPAMDataset(
        base_path / "test.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )

    num_workers = 0
    batch_size = 8

    # 中文:训练集 DataLoader,shuffle=True 打乱样本顺序,
    # drop_last=True 丢弃最后不足一个 batch 的样本(保证每个 batch 大小一致)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # 中文:验证/测试集不需要打乱顺序,也不丢弃末尾不足一个 batch 的样本
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    ###############################
    # Train model
    # 中文:训练模型
    ###############################

    start_time = time.time()
    torch.manual_seed(123)
    # 中文:使用 AdamW 优化器,weight_decay=0.1 做权重衰减正则化;
    # 注意 model.parameters() 传入了全部参数,但只有 requires_grad=True 的
    # 参数才会被实际更新(其余参数梯度恒为 0/None,不受优化器影响)。
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
    # 中文:在完整数据集上评估最终效果
    ###############################

    print("\nEvaluating on the full datasets ...\n")

    # 中文:训练结束后,分别在完整的 train/validation/test 数据集上
    # (不再限制 num_batches)计算最终准确率,作为本次实验的最终报告结果
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
