# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
第 6 章：分类微调（Classification Fine-Tuning）可复用代码模块。

本模块对应《从零构建大语言模型》第 6 章的内容，主题是把一个预训练好的
GPT 模型（只会做“预测下一个词”）微调成一个二分类器，用来判断一条短信
是垃圾短信（spam）还是正常短信（ham）。

模块内容大致分为三部分：
1. 数据准备：下载/解压 SMS Spam Collection 数据集、构造类别平衡的数据集、
   按比例划分训练/验证/测试集、以及把文本编码 + 填充（padding）成定长张量
   的 `SpamDataset`。
2. 训练与评估：计算分类损失（交叉熵）、计算分类准确率、按 batch/epoch
   跑训练循环、以及绘制损失或准确率随训练进度变化的曲线。
3. 推理：给定一段文本，用微调后的分类模型判断它是否为垃圾短信。

需要特别说明的是：本文件本身并不包含“替换模型输出头（out_head）”和
“冻结模型参数（requires_grad = False）”这两步操作的代码——这两步通常
是在调用本模块的训练脚本/Notebook 中完成的（例如：先冻结骨干网络所有
参数，再把最后的语言模型输出头 `out_head` 替换成一个输出维度为类别数
（这里是 2：spam / not spam）的新 `nn.Linear` 层，并只解冻这个新层以及
最后一个 Transformer block 和最终 LayerNorm 参与训练）。本文件中的
`calc_accuracy_loader`、`calc_loss_batch` 等函数都是在“假设模型输出头
已经被替换为分类头”这一前提下工作的：它们统一只取序列最后一个 token
位置的 logits（形状从语言建模的 `[batch, seq_len, vocab_size]`
变为分类任务的 `[batch, seq_len, num_classes]`，再切片得到
`[batch, num_classes]`），并把它当作分类得分来计算损失/准确率。
"""


import zipfile
import os
from pathlib import Path

import requests
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import pandas as pd


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """下载并解压 SMS Spam Collection 数据集，最终得到一个 .tsv 格式的数据文件。

    如果目标数据文件已经存在，则直接跳过下载和解压，避免重复联网。

    参数：
        url (str): 数据集压缩包的下载地址。
        zip_path (str | Path): 下载得到的 zip 压缩包在本地的保存路径。
        extracted_path (str | Path): 解压后文件存放的目录。
        data_file_path (Path): 最终重命名后的数据文件路径（带 .tsv 后缀），
            函数会先检查该路径是否已存在。

    返回：
        None。该函数只做磁盘 I/O（下载、解压、重命名），不返回数据本身。
    """
    if data_file_path.exists():
        # 数据文件已存在，直接跳过后续下载与解压步骤，节省时间和带宽
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    # 以流式（stream=True）方式发起 GET 请求，避免一次性把整个文件读入内存
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()  # 若 HTTP 状态码表示错误，则抛出异常
    with open(zip_path, "wb") as out_file:
        # 按 8KB 大小的数据块逐块写入本地文件，适合下载较大的文件
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Unzipping the file
    # 将下载好的 zip 包解压到指定目录
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    # 原始解压出来的文件名为 "SMSSpamCollection"（无扩展名），
    # 这里将其重命名为调用方期望的带 .tsv 扩展名的路径，方便后续用 pandas 读取
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def create_balanced_dataset(df):
    """从原始（类别不平衡的）DataFrame 中构造一个正负样本数量相等的平衡数据集。

    原始 SMS 数据集中 "ham"（正常短信）数量远多于 "spam"（垃圾短信），
    直接用来训练容易让分类器偏向多数类。这里的做法是：以 spam 的数量为基准，
    从 ham 中随机抽取等量的样本，再与全部 spam 样本拼接，得到 1:1 的平衡数据集。

    参数：
        df (pandas.DataFrame): 包含 "Label"（取值 "spam"/"ham"）和 "Text" 两列
            的原始数据集。

    返回：
        pandas.DataFrame: 类别数量相等（spam 数 == ham 数）的平衡数据集，
            行顺序为“抽样后的 ham 在前，spam 在后”（尚未打乱）。
    """

    # Count the instances of "spam"
    # 统计垃圾短信（spam）的样本数量，作为下采样 ham 类别的目标数量
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    # 从正常短信（ham）中随机抽取与 spam 数量相同的样本，固定随机种子保证可复现
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    # 将下采样后的 ham 子集与全部 spam 样本拼接，得到类别平衡的数据集
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


def random_split(df, train_frac, validation_frac):
    """将 DataFrame 随机打乱后，按给定比例切分为训练集、验证集和测试集。

    参数：
        df (pandas.DataFrame): 待切分的数据集（通常是已经类别平衡过的数据集）。
        train_frac (float): 训练集所占比例，取值范围 (0, 1)。
        validation_frac (float): 验证集所占比例，取值范围 (0, 1)。
            剩余部分（1 - train_frac - validation_frac）自动作为测试集比例。

    返回：
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
            依次为 (train_df, validation_df, test_df)。
    """
    # Shuffle the entire DataFrame
    # frac=1 表示对全体数据进行随机重排（相当于打乱顺序），固定随机种子保证可复现，
    # reset_index(drop=True) 重置索引并丢弃旧索引，避免后续按位置切片时索引错乱
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    # 根据比例计算训练集结束位置、验证集结束位置（测试集为剩余部分）
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    # 按照上面计算出的切分点，依次切出训练集、验证集、测试集
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


class SpamDataset(Dataset):
    """垃圾短信分类任务的 PyTorch Dataset。

    负责把 CSV 文件中的原始文本（"Text" 列）用给定的分词器编码成 token id
    序列，并统一填充（pad）到相同长度，使得同一个 batch 内的所有样本可以
    直接堆叠成一个矩形张量，供 DataLoader 批量加载使用。

    属性：
        data (pandas.DataFrame): 从 csv_file 读取的原始数据，包含 "Text" 和
            "Label" 两列（"Label" 需要是已经转换好的整数，如 0/1）。
        encoded_texts (list[list[int]]): 每条文本经过分词器编码、
            截断/填充后的 token id 列表。
        max_length (int): 数据集中所有样本统一使用的序列长度（即填充后的长度）。
    """
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """初始化数据集：读取 CSV、分词编码、确定长度、截断并填充。

        参数：
            csv_file (str | Path): 包含 "Text" 和 "Label" 两列的 CSV 文件路径。
            tokenizer: 具备 `.encode(text) -> list[int]` 方法的分词器（如 tiktoken 的 BPE 分词器）。
            max_length (int | None): 序列的统一长度。若为 None，则自动取数据集中
                最长编码序列的长度作为 max_length；否则超过该长度的序列会被截断。
            pad_token_id (int): 用于填充（padding）的 token id，默认使用 GPT-2
                分词器中的 `<|endoftext|>` 对应的 id（50256）。
        """
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        # 预先把所有文本一次性分词编码为 token id 列表，避免在 __getitem__ 中重复分词，提升效率
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            # 未指定 max_length 时，取数据集中最长的编码序列长度作为统一长度
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            # 若某些样本编码后长度超过 max_length，则截断到 max_length，避免序列过长
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        # 对所有样本用 pad_token_id 在末尾填充，使其长度都等于 self.max_length，
        # 这样同一 batch 内的样本可以直接堆叠成 [batch, max_length] 的张量
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        """按索引取出一条样本，返回 (input_ids 张量, label 张量) 二元组。

        参数：
            index (int): 样本下标。

        返回：
            tuple[torch.LongTensor, torch.LongTensor]:
                - 第一个张量形状为 [max_length]，是填充/截断后的 token id 序列；
                - 第二个张量为标量（0 维张量），是该样本的分类标签（如 0=ham, 1=spam）。
        """
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        """返回数据集中样本的总数，供 DataLoader 计算 batch 数量等使用。"""
        return len(self.data)

    def _longest_encoded_length(self):
        """遍历所有已编码文本，找出最长序列的长度。

        返回：
            int: 数据集中编码后最长序列所包含的 token 数量。
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


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """在给定的数据加载器上计算分类准确率（correct / total）。

    该函数假设传入的 `model` 已经是“分类头版本”的模型，即模型最后一层
    （原本用于预测词表上每个词的 out_head）已经被替换为输出维度等于类别数
    的线性层，因此 `model(input_batch)` 输出的最后一维大小是类别数而不是词表大小。

    参数：
        data_loader (torch.utils.data.DataLoader): 提供 (input_batch, target_batch)
            批次数据的加载器。
        model (torch.nn.Module): 待评估的（分类头版本）模型。
        device (torch.device | str): 计算设备，如 "cuda" 或 "cpu"。
        num_batches (int | None): 最多评估的 batch 数量；为 None 时评估整个 data_loader。

    返回：
        float: 准确率，取值范围 [0, 1]，等于预测正确的样本数除以总样本数。
    """
    model.eval()  # 切换到评估模式，关闭 dropout 等训练专用行为
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        # 未指定评估的 batch 数量时，默认遍历整个 data_loader
        num_batches = len(data_loader)
    else:
        # 若指定的 num_batches 超过了 data_loader 实际的 batch 数，则取二者较小值
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():  # 评估阶段不需要计算梯度，节省显存和计算量
                # 模型输出形状为 [batch, seq_len, num_classes]；
                # 取序列最后一个位置（即最后一个 token）的 logits，
                # 因为经过因果自注意力后，最后一个 token 的隐藏状态已经“看到”了整段输入，
                # 切片后得到 [batch, num_classes]
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            # 在类别维度上取最大值对应的下标，作为预测的类别标签，形状为 [batch]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]  # 累加已评估的样本总数
            # 逐元素比较预测标签与真实标签是否相等，求和得到本批次预测正确的样本数
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个 batch 的分类交叉熵损失。

    参数：
        input_batch (torch.LongTensor): 形状为 [batch, seq_len] 的输入 token id。
        target_batch (torch.LongTensor): 形状为 [batch] 的真实类别标签。
        model (torch.nn.Module): 分类头版本的模型。
        device (torch.device | str): 计算设备。

    返回：
        torch.Tensor: 标量张量，表示该 batch 的平均交叉熵损失（可直接调用 .backward()）。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 同样只取序列最后一个 token 位置的 logits 作为分类得分，形状 [batch, num_classes]
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    # 交叉熵损失内部会自动对 logits 做 log_softmax，再与整数标签 target_batch 计算负对数似然，
    # 因此这里不需要手动对 logits 做 softmax
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """在给定数据加载器上计算多个 batch 的平均分类损失。

    参数：
        data_loader (torch.utils.data.DataLoader): 提供 (input_batch, target_batch) 的数据加载器。
        model (torch.nn.Module): 分类头版本的模型。
        device (torch.device | str): 计算设备。
        num_batches (int | None): 用于计算平均损失的 batch 数量；为 None 时使用全部 batch。

    返回：
        float: 所评估 batch 的平均损失；若 data_loader 为空，返回 float("nan")。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        # 空数据加载器无法计算损失，直接返回 NaN 作为哨兵值，避免除零错误
        return float("nan")
    elif num_batches is None:
        # 未指定 batch 数量时，使用 data_loader 中全部的 batch
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        # 若请求的 num_batches 超过了 data_loader 实际拥有的 batch 数，取较小值，防止越界
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # 累加每个 batch 的（标量）损失值
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    # 对累计的损失求平均，得到这 num_batches 个 batch 的平均损失
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """在训练/验证集上各评估固定数量的 batch，得到训练损失和验证损失，用于监控训练过程。

    参数：
        model (torch.nn.Module): 分类头版本的模型。
        train_loader (torch.utils.data.DataLoader): 训练集数据加载器。
        val_loader (torch.utils.data.DataLoader): 验证集数据加载器。
        device (torch.device | str): 计算设备。
        eval_iter (int): 评估时各自使用的 batch 数量（而非整个数据集），
            以加快训练过程中的中间评估速度。

    返回：
        tuple[float, float]: (train_loss, val_loss)，分别是训练集和验证集上的平均损失。
    """
    model.eval()  # 评估期间关闭 dropout 等，保证损失计算的确定性
    with torch.no_grad():  # 评估阶段不需要反向传播，禁用梯度计算以节省显存
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 评估结束后切回训练模式，恢复 dropout 等训练时行为，避免影响后续训练
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    """分类微调的主训练循环：按 batch 更新参数，并定期评估损失与准确率。

    参数：
        model (torch.nn.Module): 待微调的分类头版本模型（通常骨干参数已冻结，
            只有新替换的分类头 + 部分顶层参数是可训练的）。
        train_loader (torch.utils.data.DataLoader): 训练集数据加载器。
        val_loader (torch.utils.data.DataLoader): 验证集数据加载器。
        optimizer (torch.optim.Optimizer): 优化器，只会更新 `requires_grad=True` 的参数。
        device (torch.device | str): 计算设备。
        num_epochs (int): 训练的总轮数（epoch 数）。
        eval_freq (int): 每训练多少个全局 step 做一次中间评估（打印训练/验证损失）。
        eval_iter (int): 中间评估时训练/验证集各自使用的 batch 数量。

    返回：
        tuple[list[float], list[float], list[float], list[float], int]:
            依次为 (train_losses, val_losses, train_accs, val_accs, examples_seen)：
            - train_losses / val_losses：每次中间评估记录下的训练/验证损失列表；
            - train_accs / val_accs：每个 epoch 结束后记录的训练/验证准确率列表；
            - examples_seen：训练过程中累计处理过的样本总数。
    """
    # Initialize lists to track losses and examples seen
    # 用于记录训练过程中损失、准确率随时间的变化，便于后续绘图分析
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        # 切换到训练模式，启用 dropout 等只在训练时生效的层

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            # 清空上一步遗留的梯度，避免梯度在不同 batch 间累积
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            # 反向传播，计算损失相对于所有可训练参数（分类头等）的梯度
            optimizer.step()  # Update model weights using loss gradients
            # 根据梯度更新一次可训练参数（骨干中被冻结的参数因 requires_grad=False 不会被更新）
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            # 累计已经处理过的样本数量（而不是像语言模型预训练那样统计 token 数）
            global_step += 1

            # Optional evaluation step
            # 每隔 eval_freq 个全局 step，做一次基于少量 batch 的快速评估，用于监控训练进度
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        # 每个 epoch 结束后，用 calc_accuracy_loader 计算一次训练集/验证集上的分类准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    """绘制训练/验证指标（损失或准确率）随 epoch 与样本数变化的曲线图，并保存为 PDF。

    图中使用两条 x 轴：底部 x 轴表示 epoch 数，顶部 x 轴表示累计样本数（examples seen），
    二者共享同一个 y 轴，便于同时从“训练轮数”和“样本吞吐量”两个角度观察训练进度。

    参数：
        epochs_seen (Sequence[float]): 与 train_values/val_values 对应的 epoch 坐标序列。
        examples_seen (Sequence[int]): 与 train_values/val_values 对应的累计样本数坐标序列。
        train_values (Sequence[float]): 训练集上的指标值（如损失或准确率）序列。
        val_values (Sequence[float]): 验证集上的指标值序列。
        label (str): 指标名称，用于坐标轴标签、图例文字以及保存文件名，默认 "loss"。

    返回：
        None。函数会调用 `plt.show()` 显示图像，并将图像保存为 "{label}-plot.pdf"。
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    # 在主坐标轴（底部 x 轴为 epoch）上分别绘制训练曲线和验证曲线（验证曲线用虚点线区分）
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    # 创建共享同一 y 轴的第二个 x 轴（顶部），用于展示“累计样本数”这一维度
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    # 这里绘制一条透明度为 0（完全不可见）的曲线，其唯一目的是让 matplotlib
    # 根据 examples_seen 的取值范围自动对齐并生成顶部 x 轴的刻度
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    # 自动调整子图间距和边距，避免坐标轴标签被裁剪
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """对单条文本（如一条短信）做垃圾短信/正常短信的分类推理。

    参数：
        text (str): 待分类的原始文本。
        model (torch.nn.Module): 已训练好的分类头版本模型。
        tokenizer: 具备 `.encode(text) -> list[int]` 方法的分词器。
        device (torch.device | str): 计算设备。
        max_length (int): 输入序列填充/截断到的目标长度（通常应与训练时使用的
            SpamDataset.max_length 保持一致）。
        pad_token_id (int): 填充所用的 token id，默认沿用 GPT-2 的 `<|endoftext|>`（50256）。

    返回：
        str: 分类结果，取值为 "spam"（垃圾短信）或 "not spam"（正常短信）。
    """
    model.eval()  # 推理前切换到评估模式，关闭 dropout 等

    # Prepare inputs to the model
    # 用分词器将原始文本编码为 token id 列表
    input_ids = tokenizer.encode(text)
    # 读取模型位置编码矩阵的行数，即模型实际支持的最大上下文长度
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    # 取 max_length 与模型支持的最大上下文长度二者中较小的一个作为实际截断长度，
    # 避免输入长度超出模型位置编码的支持范围
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    # 若截断后长度仍不足 max_length，则用 pad_token_id 在末尾补齐到 max_length
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    # 转换为张量并在最前面增加一个 batch 维度，得到形状 [1, max_length]，
    # 因为模型的前向计算默认接收带 batch 维度的输入
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():  # 推理阶段无需梯度，禁用梯度计算以节省显存、加快速度
        # 同训练/评估逻辑一致：只取序列最后一个 token 位置的 logits 作为分类得分，
        # 形状从 [1, seq_len, num_classes] 切片为 [1, num_classes]
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    # 取 logits 中数值最大的类别下标作为预测标签（0 或 1），.item() 将单元素张量转为 Python 标量
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    # 按约定：标签 1 表示垃圾短信（spam），其余（0）表示正常短信（not spam）
    return "spam" if predicted_label == 1 else "not spam"
