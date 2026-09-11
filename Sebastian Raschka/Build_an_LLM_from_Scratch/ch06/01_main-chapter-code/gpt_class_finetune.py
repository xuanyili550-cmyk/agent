# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This is a summary file containing the main takeaways from chapter 6.

"""
中文模块说明
============
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 第 6 章
的“汇总脚本”，展示了如何把一个预训练好的 GPT-2 模型改造成一个**二分类器**，
用来判断一条短信是垃圾短信（spam）还是正常短信（ham）——即所谓的“分类微调”
（classification finetuning），区别于第 7 章的“指令微调”。

脚本主要包含以下几个部分：
1. 下载并解压 SMS Spam Collection 数据集（下载失败时会自动尝试备用 URL）；
2. 数据预处理：类别平衡采样、训练/验证/测试集划分、Tokenizer 编码与填充；
3. 定义 `SpamDataset`（PyTorch Dataset），并构建对应的 `DataLoader`；
4. 加载预训练的 GPT-2 权重，把原本的语言模型输出头替换成只有 2 个类别的线性层，
   并且只解冻最后一层 Transformer block、最终归一化层和新输出头，其余参数冻结；
5. 微调训练循环（`train_classifier_simple`），期间周期性地评估训练/验证损失，
   并在每个 epoch 结束后计算分类准确率；
6. 绘制损失曲线和准确率曲线并保存为 PDF。

这是一个可直接运行的脚本（`if __name__ == "__main__"`），也可以作为模块被
`load_finetuned_model.py` 等其他脚本导入其中的函数/类。
"""

import requests
import zipfile
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
import tiktoken
import torch
# Import Dynamo before TensorFlow is loaded by gpt_download to avoid a native
# Triton/TensorFlow initialization crash with recent PyTorch nightly builds.
# 中文：必须在 gpt_download 触发 TensorFlow 加载之前先导入 torch._dynamo，
# 否则在较新的 PyTorch nightly 版本下，Triton/TensorFlow 的原生初始化可能会崩溃。
import torch._dynamo  # noqa: F401
from torch.utils.data import Dataset, DataLoader

from previous_chapters import GPTModel, load_weights_into_gpt


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """
    下载并解压 SMS Spam Collection 数据集（若目标文件已存在则跳过）。

    参数:
        url (str): 数据集压缩包的下载地址。
        zip_path (str): 下载后压缩包在本地保存的路径。
        extracted_path (str): 解压目标目录。
        data_file_path (Path): 期望得到的最终数据文件路径（.tsv 文件）。

    返回:
        None。副作用是在磁盘上生成 `data_file_path` 指向的文件。
    """
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    # 中文：以流式方式下载压缩包，避免一次性把整个文件读入内存；
    # timeout=60 防止网络异常时脚本无限期挂起。
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Unzipping the file
    # 中文：解压 zip 压缩包到指定目录。
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    # 中文：原始解压出来的文件没有扩展名（SMSSpamCollection），
    # 这里重命名为带 .tsv 后缀的文件，方便后续用 pandas 按制表符读取。
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def create_balanced_dataset(df):
    """
    构造类别平衡的数据集：让 spam 和 ham 样本数量相等。

    原始数据集中 ham（正常短信）远多于 spam（垃圾短信），如果直接训练，
    模型可能会倾向于把所有样本都预测成多数类。这里对 ham 做欠采样，
    使其数量与 spam 一致，从而得到类别均衡的数据集。

    参数:
        df (pandas.DataFrame): 包含 "Label" 列（取值 "ham"/"spam"）的原始数据。

    返回:
        pandas.DataFrame: 行数为 `2 * num_spam` 的平衡后数据集。
    """
    # Count the instances of "spam"
    # 中文：统计 spam 样本的数量，作为下采样 ham 的目标数量。
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    # 中文：从 ham 样本中随机抽取与 spam 数量相等的子集（固定随机种子保证可复现）。
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    # 中文：拼接欠采样后的 ham 子集与全部 spam 样本，得到平衡数据集。
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


def random_split(df, train_frac, validation_frac):
    """
    将数据集随机打乱后按比例划分为训练集、验证集、测试集。

    参数:
        df (pandas.DataFrame): 待划分的数据集。
        train_frac (float): 训练集所占比例。
        validation_frac (float): 验证集所占比例（剩余部分自动作为测试集）。

    返回:
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
            (train_df, validation_df, test_df)
    """
    # Shuffle the entire DataFrame
    # 中文：frac=1 表示对全部行做随机重排（打乱顺序），固定随机种子保证可复现，
    # reset_index(drop=True) 重新生成从 0 开始的连续索引。
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    # 中文：根据比例计算训练集结束位置和验证集结束位置的索引下标。
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    # 中文：按切片方式划分三个子集，剩余部分（validation_end 之后）作为测试集。
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


class SpamDataset(Dataset):
    """
    垃圾短信分类数据集（PyTorch Dataset 的具体实现）。

    功能：
        1. 从 CSV 文件读取文本和标签；
        2. 使用给定的 tokenizer 对文本预先编码（tokenize）；
        3. 根据 max_length 对编码后的序列做截断或统计出最长长度；
        4. 使用 pad_token_id 将所有序列填充（pad）到统一长度 max_length，
           以便能够组成规整的 batch 张量。

    属性:
        max_length (int): 数据集中样本编码后的统一长度（用于对齐 padding）。
    """

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        参数:
            csv_file (str): 包含 "Text" 和 "Label" 两列的 CSV 文件路径。
            tokenizer: 具备 `.encode(text) -> List[int]` 接口的分词器（如 tiktoken 的 GPT-2 编码器）。
            max_length (int | None): 统一序列长度。为 None 时自动取数据集中
                最长编码序列的长度（常用于训练集，以避免额外截断信息）。
            pad_token_id (int): 用于填充（padding）的 token id，默认使用 GPT-2 的
                `<|endoftext|>` token（id 为 50256）。
        """
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        # 中文：提前把所有文本编码成 token id 列表，避免训练时重复分词，加快 __getitem__ 速度。
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            # 中文：未指定 max_length 时，用数据集中最长的编码序列长度作为统一长度。
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            # 中文：对超过 max_length 的序列做截断，保证长度不超过限制
            # （常用于验证/测试集，沿用训练集算出的 max_length）。
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        # 中文：对所有序列右侧补齐 pad_token_id，使得每条样本长度都等于 self.max_length，
        # 这样才能在 DataLoader 中被 stack 成形状为 [batch_size, max_length] 的张量。
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        """
        按索引取出一条样本。

        返回:
            tuple[torch.LongTensor, torch.LongTensor]:
                - 编码后的输入序列，形状 [max_length]；
                - 对应的分类标签（0=ham, 1=spam），标量张量。
        """
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        """返回数据集样本总数。"""
        return len(self.data)

    def _longest_encoded_length(self):
        """
        遍历所有已编码文本，返回其中最长序列的长度。

        返回:
            int: 最长编码序列的 token 数量。
        """
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
        # Note: A more pythonic version to implement this method
        # is the following, which is also used in the next chapter:
        # return max(len(encoded_text) for encoded_text in self.encoded_texts)
        # 中文：上面注释给出了更 Pythonic 的写法（用 max() + 生成器表达式），
        # 效果与当前实现相同，第 7 章代码中采用了这种写法。


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    在给定的数据加载器上计算分类准确率。

    参数:
        data_loader (DataLoader): 提供 (input_batch, target_batch) 的数据加载器。
        model (torch.nn.Module): 待评估的（分类头已替换的）GPT 模型。
        device (torch.device | str): 计算设备（"cpu" 或 "cuda"）。
        num_batches (int | None): 最多评估多少个 batch；None 表示评估全部 batch。

    返回:
        float: 正确预测样本数 / 总样本数，即分类准确率。
    """
    model.eval()  # 中文：切换到评估模式，关闭 dropout 等训练专用行为
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():  # 中文：评估阶段不需要梯度，节省显存并加速
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
                # 中文：model(input_batch) 的输出形状为 [batch_size, seq_len, num_classes]；
                # 分类微调只取序列最后一个位置的 logits（形状变为 [batch_size, num_classes]），
                # 因为该位置能看到完整的输入上下文（自回归的因果注意力机制保证了这一点）。
            predicted_labels = torch.argmax(logits, dim=-1)
            # 中文：对每个样本在类别维度上取 argmax，得到预测的类别标签，形状 [batch_size]。

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
            # 中文：逐元素比较预测标签与真实标签，统计预测正确的样本数量。
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算单个 batch 的分类交叉熵损失。

    参数:
        input_batch (torch.LongTensor): 输入 token id，形状 [batch_size, seq_len]。
        target_batch (torch.LongTensor): 真实类别标签，形状 [batch_size]。
        model (torch.nn.Module): 分类模型。
        device: 计算设备。

    返回:
        torch.Tensor: 标量损失值（可反向传播）。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    # 中文：同样只取最后一个 token 位置的 logits 作为分类依据，形状 [batch_size, num_classes]。
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    # 中文：交叉熵损失内部会自动对 logits 做 softmax 再计算负对数似然，
    # target_batch 是类别索引（不是 one-hot），符合 F.cross_entropy 的输入要求。
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    在给定的数据加载器上计算平均损失（用于训练/验证损失监控）。

    参数:
        data_loader (DataLoader): 数据加载器。
        model (torch.nn.Module): 分类模型。
        device: 计算设备。
        num_batches (int | None): 最多使用多少个 batch 估算平均损失；
            None 表示使用全部 batch。

    返回:
        float: 平均每个 batch 的损失值；若 data_loader 为空则返回 NaN。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    在训练过程中周期性地评估模型在训练集和验证集上的损失。

    为了不拖慢训练速度，通常只用少量 batch（由 eval_iter 控制）做近似评估，
    而不是遍历整个数据集。

    参数:
        model (torch.nn.Module): 分类模型。
        train_loader (DataLoader): 训练集数据加载器。
        val_loader (DataLoader): 验证集数据加载器。
        device: 计算设备。
        eval_iter (int): 评估时使用的 batch 数量上限。

    返回:
        tuple[float, float]: (train_loss, val_loss)
    """
    model.eval()  # 中文：切换到评估模式，避免 dropout 影响损失计算
    with torch.no_grad():  # 中文：评估阶段禁用梯度计算，节省显存和时间
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 中文：评估结束后切回训练模式，恢复 dropout 等行为
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    """
    分类微调的主训练循环。

    与预训练/生成任务的训练循环相比，这里的关键区别是：
        - 损失函数是分类交叉熵（只对序列最后一个 token 的 logits 计算），而非语言建模的
          逐 token 交叉熵；
        - 每个 epoch 结束后额外计算并打印训练/验证准确率。

    参数:
        model (torch.nn.Module): 待微调的分类模型（GPT 主干 + 新分类头）。
        train_loader (DataLoader): 训练集数据加载器。
        val_loader (DataLoader): 验证集数据加载器。
        optimizer (torch.optim.Optimizer): 优化器（如 AdamW）。
        device: 计算设备。
        num_epochs (int): 训练轮数。
        eval_freq (int): 每隔多少个训练 step 做一次损失评估。
        eval_iter (int): 每次评估使用的 batch 数量上限。

    返回:
        tuple: (train_losses, val_losses, train_accs, val_accs, examples_seen)
            - train_losses / val_losses (list[float]): 各次评估记录的训练/验证损失；
            - train_accs / val_accs (list[float]): 每个 epoch 结束后记录的训练/验证准确率；
            - examples_seen (int): 训练过程中累计处理过的样本（不是 token）数量。
    """
    # Initialize lists to track losses and tokens seen
    # 中文：用于记录训练过程中的损失曲线和准确率曲线数据，供后续绘图使用。
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        # 中文：显式设置为训练模式，确保 dropout 等层按训练方式工作
        # （尤其是上一轮 evaluate_model 结束后已切回 train，这里是保险起见）。

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            # 中文：每个 batch 开始前清空梯度，避免梯度在多个 batch 间累积。
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            # 中文：反向传播计算损失相对于（可训练）参数的梯度。
            optimizer.step()  # Update model weights using loss gradients
            # 中文：根据梯度更新模型参数（只有 requires_grad=True 的参数会被更新，
            # 前面冻结的大部分 GPT 主干参数梯度虽然会被跳过更新）。
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            # 中文：这里统计的是“样本数”而非第 5 章预训练中统计的“token 数”，
            # 因为分类任务关心的是处理了多少条短信样本。
            global_step += 1

            # Optional evaluation step
            # 中文：每 eval_freq 个 step 做一次轻量级损失评估并打印进度。
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        # 中文：每个 epoch 结束后，用 eval_iter 个 batch 估算训练/验证准确率并打印。
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    """
    绘制训练/验证指标（损失或准确率）随 epoch 和样本数变化的曲线图，并保存为 PDF。

    图中使用双 x 轴：
        - 下方 x 轴（ax1）：以 epoch 为单位；
        - 上方 x 轴（ax2）：以累计处理样本数为单位（与下方轴对齐但不可见，仅用于刻度映射）。

    参数:
        epochs_seen (Sequence[float]): 与记录点对应的 epoch 数（可为小数，用于插值对齐）。
        examples_seen (Sequence[float]): 与记录点对应的累计样本数。
        train_values (Sequence[float]): 训练集上的指标值（损失或准确率）。
        val_values (Sequence[float]): 验证集上的指标值。
        label (str): 指标名称，用于坐标轴标签、图例文字和输出文件名（如 "loss"、"accuracy"）。

    返回:
        None。副作用是生成并保存名为 "{label}-plot.pdf" 的图像文件。
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    # 中文：主坐标轴（ax1）以 epoch 为横轴，绘制训练/验证曲线。
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for tokens seen
    # 中文：创建共享同一 y 轴的第二条 x 轴（ax2），用来标注“已处理样本数”这一维度，
    # 让读者既能看到训练进行到第几个 epoch，也能看到消耗了多少数据样本。
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    # 中文：这条曲线设置 alpha=0（完全透明），只是为了让 matplotlib 根据
    # examples_seen 的数值范围自动生成上方 x 轴的刻度，本身不会显示在图上。
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    # plt.show()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Finetune a GPT model for classification"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("This flag runs the model in test mode for internal testing purposes. "
              "Otherwise, it runs the model as it is used in the chapter (recommended).")
    )
    args = parser.parse_args()
    # 中文：命令行参数 --test_mode 用于内部测试（使用一个极小的随机初始化模型、
    # 在 CPU 上跑通整套流程），正式运行本章内容时不要加这个参数。

    ########################################
    # Download and prepare dataset
    ########################################
    # 中文：下载并准备 SMS 垃圾短信分类数据集。

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    try:
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    except (requests.exceptions.RequestException, TimeoutError) as e:
        # 中文：主下载地址（UCI 官方源）失败时（网络问题/超时等），
        # 自动切换到作者提供的 Backblaze 备用镜像地址重试一次。
        print(f"Primary URL failed: {e}. Trying backup URL...")
        url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    # 中文：原始 tsv 文件没有表头，用制表符分隔，手动指定列名为 "Label" 和 "Text"。
    balanced_df = create_balanced_dataset(df)
    # 中文：对 ham/spam 样本数量做平衡处理，避免类别不均衡影响训练效果。
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    # 中文：把字符串标签映射为整数类别 id，0 表示正常短信，1 表示垃圾短信，
    # 以便后续用于交叉熵损失计算。

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    # 中文：按 70% / 10% / 20% 的比例划分训练/验证/测试集。
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)
    # 中文：将划分好的三个子集分别落盘为 CSV 文件，供 SpamDataset 读取，
    # 也方便复用/调试而无需重复下载和划分。

    ########################################
    # Create data loaders
    ########################################
    tokenizer = tiktoken.get_encoding("gpt2")
    # 中文：使用与 GPT-2 预训练一致的 BPE 分词器，保证 token id 与预训练权重的词表对应。

    train_dataset = SpamDataset(
        csv_file="train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    # 中文：训练集不指定 max_length，让其自动取训练集中最长样本的编码长度，
    # 作为整个数据集统一的序列长度基准。

    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    test_dataset = SpamDataset(
        csv_file="test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    # 中文：验证集和测试集复用训练集算出的 max_length，保证三者的序列长度一致，
    # 避免因为验证/测试集出现比训练集更长的样本而导致模型看到训练时没见过的长度。

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)
    # 中文：固定随机种子，保证 DataLoader 的 shuffle 行为可复现。

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        # 中文：训练集需要打乱顺序（shuffle=True），并丢弃最后不满一个 batch 的样本
        # （drop_last=True），避免不完整 batch 影响梯度统计或引发 batch 维度不一致问题。
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        # 中文：验证集不需要打乱顺序，也不丢弃末尾不完整的 batch，
        # 因为评估阶段希望尽量利用全部样本、结果也不涉及梯度更新。
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    ########################################
    # Load pretrained model
    ########################################
    # 中文：加载预训练 GPT-2 模型权重（或测试模式下的迷你随机模型）。

    # Small GPT model for testing purposes
    if args.test_mode:
        # 中文：测试模式下使用极小的模型配置（1 层、2 个注意力头、极小的嵌入维度），
        # 目的是快速跑通整个训练/评估流程做功能性验证，而不追求实际分类效果。
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 120,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        device = "cpu"

    # Code as it is used in the main chapter
    else:
        # 中文：正式流程——选择一个具体规格的 GPT-2 模型（默认 124M 参数的 small 版本），
        # 并从 OpenAI 官方发布的检查点下载对应权重。
        CHOOSE_MODEL = "gpt2-small (124M)"
        INPUT_PROMPT = "Every effort moves"

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
        # 中文：不同规格 GPT-2 模型对应的结构超参数（嵌入维度、层数、注意力头数）。

        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
        # 中文：把所选模型规格的专属超参数合并进基础配置字典中。

        assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
            f"Dataset length {train_dataset.max_length} exceeds model's context "
            f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
            f"`max_length={BASE_CONFIG['context_length']}`"
        )
        # 中文：安全检查——数据集编码后的最大长度不能超过模型支持的上下文长度，
        # 否则位置编码会越界，训练时会报错。

        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        # 中文：从字符串 "gpt2-small (124M)" 中提取出 "124M" 这样的模型规格标识，
        # 用于定位下载对应的权重文件目录。
        from gpt_download import download_and_load_gpt2

        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        # 中文：下载（若已存在则跳过）并加载 OpenAI 官方发布的 GPT-2 TensorFlow 检查点，
        # 返回模型配置 settings 和以 numpy 数组形式存储的参数字典 params。

        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)
        # 中文：构建与配置匹配的 GPTModel，并把下载到的预训练权重逐层拷贝进去。
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################################
    # Modify and pretrained model
    ########################################
    # 中文：改造预训练模型以适配分类任务——冻结大部分参数，替换输出头，
    # 只解冻少量层用于微调（这是一种参数高效微调策略，能显著降低训练成本）。

    for param in model.parameters():
        param.requires_grad = False
    # 中文：先冻结模型的全部参数（requires_grad=False），
    # 即反向传播时不会为这些参数计算/更新梯度。

    torch.manual_seed(123)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    # 中文：把原本用于预测词表分布（out_features=vocab_size）的语言模型输出头，
    # 替换为一个新的全连接层，输出维度变为 2（对应 ham/spam 两个类别）。
    # 新创建的 nn.Linear 层默认 requires_grad=True，因此它是可训练的。
    model.to(device)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    # 中文：额外解冻最后一个 Transformer block 的参数，让模型能针对分类任务
    # 微调靠近输出端的高层特征表示，同时仍保留浅层已学到的通用语言知识。

    for param in model.final_norm.parameters():
        param.requires_grad = True
    # 中文：同时解冻最终的 LayerNorm（final_norm）参数，
    # 因为它的输出直接送入新的分类头，需要随之微调以匹配新任务分布。

    ########################################
    # Finetune modified model
    ########################################
    # 中文：正式开始分类微调训练。

    start_time = time.time()
    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    # 中文：使用 AdamW 优化器（Adam 的解耦权重衰减版本），
    # 传入 model.parameters() 时 PyTorch 会自动只更新 requires_grad=True 的那部分参数。
    # 学习率 5e-5 是微调场景下常用的较小学习率，避免破坏预训练权重。

    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )
    # 中文：执行 5 个 epoch 的微调训练；每 50 个 step 评估一次损失（用 5 个 batch 近似）。

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    ########################################
    # Plot results
    ########################################
    # 中文：绘制并保存训练过程中的损失曲线和准确率曲线。

    # loss plot
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    # 中文：由于损失是按 eval_freq 间隔记录的，记录点数量与 epoch 数不是一一对应，
    # 这里用 torch.linspace 在 [0, num_epochs]（以及 [0, examples_seen]）区间内
    # 均匀生成与记录点数量相同的坐标值，用于绘图时的横轴对齐。
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    # accuracy plot
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    # 中文：准确率是每个 epoch 结束后记录一次，因此记录点数量等于 num_epochs，
    # 同样用 linspace 生成对应的横轴坐标。
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")
