# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是第 6 章(文本分类微调)的"附加对照实验"脚本。

它在同一套 GPT-2 分类微调流程上,暴露了一组命令行开关,用来做消融/对照实验,
包括但不限于:
    - 只微调最后一层 / 最后一个 Transformer block / 最后两个 block / 全部参数 / LoRA;
    - 用哪个 token 位置的输出做分类(第一个 token、最后一个 token、"flexible"
      即每条序列最后一个非 padding token);
    - 是否对所有 token 的输出做平均池化(average pooling)代替取单一 token;
    - 是否禁用因果注意力掩码(causal mask),即允许模型看到未来 token(类似双向注意力);
    - 是否使用 LoRA(低秩适应)做参数高效微调,以及两种数学等价但实现方式不同的 LoRA
      写法(分离的 LoRA 分支 vs. 将 LoRA 权重与原始权重合并);
    - 是否对样本做 padding、上下文长度如何选择、梯度累积步数等训练细节。

脚本本身可以直接以命令行方式运行,会自动下载 SMS 垃圾短信数据集、加载预训练
GPT-2 权重、构造分类头、按所选实验配置微调,并在训练集/验证集/测试集上报告准确率。

Source for "Build a Large Language Model From Scratch"
    - https://www.manning.com/books/build-a-large-language-model-from-scratch
Code: https://github.com/rasbt/LLMs-from-scratch
"""

import argparse
import math
import os
from pathlib import Path
import time
import zipfile

import pandas as pd
import requests
import tiktoken
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt


# If the `previous_chapters.py` file is not available locally,
# you can import it from the `llms-from-scratch` PyPI package.
# For details, see: https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# E.g.,
# from llms_from_scratch.ch04 import GPTModel
# from llms_from_scratch.ch05 import download_and_load_gpt2, load_weights_into_gpt


class LoRALayer(torch.nn.Module):
    """LoRA(Low-Rank Adaptation,低秩适应)的核心可训练分支。

    用两个低秩矩阵 A、B 近似表示对原始权重矩阵的增量更新 ΔW = alpha * A @ B,
    从而只需训练远小于原权重矩阵参数量的 A、B,即可实现参数高效微调。

    Args:
        in_dim (int): 输入特征维度(对应原始 Linear 层的 in_features)。
        out_dim (int): 输出特征维度(对应原始 Linear 层的 out_features)。
        rank (int): 低秩分解的秩 r,即 A 的列数 / B 的行数,决定了新增参数量与
            拟合能力之间的折中,r 越大越接近全量微调,但参数量也越多。
        alpha (float): 缩放系数,用于控制 LoRA 分支对输出的影响幅度,
            通常与 rank 搭配使用(例如 alpha == rank 时相当于不做额外缩放)。
    """

    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # A: [in_dim, rank],使用 kaiming_uniform_ 初始化,与 nn.Linear 默认初始化方式一致
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B: [rank, out_dim],初始化为全零,保证训练刚开始时 LoRA 分支不改变原始输出
        # (即 A @ B == 0,与预训练权重完全一致,训练是从"无扰动"状态开始的)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        # x: [..., in_dim] -> x @ A: [..., rank] -> (x @ A) @ B: [..., out_dim]
        # 整体等价于对输入施加一个秩不超过 rank 的线性变换,再乘以缩放系数 alpha
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    """把原始 nn.Linear 与一个并联的 LoRA 分支组合起来的包装层。

    前向传播时,原始 Linear 的输出与 LoRA 分支的输出相加,原始 Linear 的权重
    保持冻结(不参与梯度更新,由外部训练脚本控制 requires_grad),只有 LoRA
    分支的 A、B 参与训练。

    Args:
        linear (torch.nn.Linear): 待适配的原始线性层,权重会被保留但通常冻结。
        rank (int): 传给内部 LoRALayer 的低秩秩数。
        alpha (float): 传给内部 LoRALayer 的缩放系数。
    """

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        # x: [..., in_features] -> 输出: [..., out_features]
        # 原始线性变换 + LoRA 低秩增量,两路相加得到最终输出
        return self.linear(x) + self.lora(x)


# This LoRA code is equivalent to LinearWithLoRA
class LinearWithLoRAMerged(torch.nn.Module):
    """与 LinearWithLoRA 数学上等价的另一种实现:将 LoRA 增量直接合并进权重矩阵。

    不同于 LinearWithLoRA 中"原始 Linear 输出 + LoRA 分支输出"两次矩阵乘法的写法,
    这里先把 LoRA 的低秩增量 alpha * A @ B 与原始权重矩阵相加,合并成一个新的权重
    矩阵,再做一次 F.linear。两种写法在数学上完全等价,但计算路径不同(适合用来
    验证/对比实现是否等价,或者在推理阶段把 LoRA "烧录"进权重以减少一次矩阵乘法)。

    Args:
        linear (torch.nn.Linear): 待适配的原始线性层。
        rank (int): 传给内部 LoRALayer 的低秩秩数。
        alpha (float): 传给内部 LoRALayer 的缩放系数。
    """

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        # lora: A @ B -> [in_features, out_features]
        lora = self.lora.A @ self.lora.B
        # self.linear.weight 的形状是 [out_features, in_features](nn.Linear 的约定),
        # 所以要对 lora 做转置(得到 [out_features, in_features])才能与其相加
        combined_weight = self.linear.weight + self.lora.alpha*lora.T
        # 用合并后的权重矩阵一次性完成线性变换,效果等价于 LinearWithLoRA
        return torch.nn.functional.linear(x, combined_weight, self.linear.bias)


class SpamDataset(Dataset):
    """垃圾短信分类数据集,读取 CSV(列名为 Text / Label),做分词与定长 padding。

    Args:
        csv_file (str or Path): 数据集 CSV 文件路径,需包含 "Text" 和 "Label" 两列。
        tokenizer: 具备 `.encode(str) -> List[int]` 接口的分词器(此脚本中为 tiktoken 的 gpt2 编码器)。
        max_length (int, optional): 截断/padding 到的最大 token 长度。若为 None,
            则自动取数据集中最长样本的编码长度(见 `_longest_encoded_length`)。
        pad_token_id (int): 用于填充的 token id,默认 50256 即 GPT-2 的 `<|endoftext|>`。
        no_padding (bool): 若为 True,则不做 padding,每条样本保留原始(截断后)的长度,
            此时同一个 batch 内各样本长度可能不同,需配合 `batch_size=1` 使用,
            否则默认的 DataLoader collate 逻辑无法把不等长的张量拼成一个 batch。
    """

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, no_padding=False):
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        # Pre-tokenize texts
        # 预先把所有文本编码成 token id 序列,并截断到 self.max_length
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]
            for text in self.data["Text"]
        ]

        if not no_padding:
            # Pad sequences to the longest sequence
            # 用 pad_token_id 把每条序列右侧补齐到 self.max_length,方便按 batch 堆叠成张量
            self.encoded_texts = [
                et + [pad_token_id] * (self.max_length - len(et))
                for et in self.encoded_texts
            ]

    def __getitem__(self, index):
        """返回单条样本。

        Returns:
            tuple: (encoded, label)
                encoded: torch.LongTensor,形状 [seq_len](padding 模式下 seq_len == self.max_length)
                label: torch.LongTensor,标量,0 表示正常短信(ham),1 表示垃圾短信(spam)
        """
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        """遍历数据集,统计所有样本编码后 token 序列的最大长度,用作默认的 max_length。

        Returns:
            int: 数据集中最长样本的编码 token 数。
        """
        max_length = 0
        for text in self.data["Text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
        # Note: A more pythonic version to implement this method
        # is the following, which is also used in the next chapter:
        # return max(len(encoded_text) for encoded_text in self.encoded_texts)


def download_and_unzip(url, zip_path, extract_to, new_file_path):
    """下载压缩包并解压,若目标文件已存在则跳过。

    Args:
        url (str): 数据集压缩包下载地址。
        zip_path (str or Path): 压缩包在本地的保存路径。
        extract_to (str or Path): 解压目标目录。
        new_file_path (Path): 解压并重命名后期望得到的最终文件路径;若该文件已存在,
            直接跳过下载与解压(避免重复联网)。
    """
    if new_file_path.exists():
        print(f"{new_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    # 以流式方式下载,避免一次性把大文件读入内存
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # Renaming the file to indicate its format
    # 原始数据集里的文件名没有后缀,这里重命名为 .tsv 以表明其制表符分隔的格式
    original_file = Path(extract_to) / "SMSSpamCollection"
    os.rename(original_file, new_file_path)
    print(f"File downloaded and saved as {new_file_path}")


def random_split(df, train_frac, val_frac):
    """把 DataFrame 随机打乱后按比例切分为训练/验证/测试三部分。

    Args:
        df (pandas.DataFrame): 待切分的数据集。
        train_frac (float): 训练集占比。
        val_frac (float): 验证集占比;剩余部分(1 - train_frac - val_frac)作为测试集。

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]: (train_df, val_df, test_df)
    """
    # Shuffle the entire DataFrame
    # 固定 random_state=123 以保证多次运行的切分结果可复现
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df


def create_dataset_csvs(new_file_path):
    """从原始 tsv 文件构造均衡数据集,并落盘为 train/validation/test 三个 CSV。

    因为原始短信数据集中 ham(正常短信)数量远多于 spam(垃圾短信),这里对 ham
    做下采样(sample 数量与 spam 相同),得到类别均衡的数据集,再随机切分保存。

    Args:
        new_file_path (Path): 原始 tsv 文件路径,列为 [Label, Text],Label 取值 "ham"/"spam"。
    """
    df = pd.read_csv(new_file_path, sep="\t", header=None, names=["Label", "Text"])

    # Create balanced dataset
    # 统计 spam 样本数,并从 ham 样本中随机采样等量的数据,构造类别均衡的数据集
    n_spam = df[df["Label"] == "spam"].shape[0]
    ham_sampled = df[df["Label"] == "ham"].sample(n_spam, random_state=123)
    balanced_df = pd.concat([ham_sampled, df[df["Label"] == "spam"]])
    balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)
    # 把字符串标签映射为整数标签,ham -> 0,spam -> 1,供交叉熵损失使用
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    # Sample and save csv files
    # 按 70% / 10% / 20% 的比例切分为训练 / 验证 / 测试集并保存到当前目录
    train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    val_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)


def instantiate_model(choose_model, load_weights):
    """根据模型规格构造 GPTModel,并可选加载官方 GPT-2 预训练权重。

    Args:
        choose_model (str): 模型规格名称,取值为
            "gpt2-small (124M)" / "gpt2-medium (355M)" / "gpt2-large (774M)" / "gpt2-xl (1558M)" 之一。
        load_weights (bool): 是否从 OpenAI 官方权重加载(即 "pretrained" 模式);
            若为 False,则使用随机初始化权重(即 "random" 模式,消融实验用,验证预训练是否真的有帮助)。

    Returns:
        GPTModel: 已切换到 eval() 模式的 GPT 模型实例。

    注意:此函数内部读取了模块级全局变量 `args.disable_causal_mask`,而不是把它作为
    参数传入。这依赖于调用方在 `if __name__ == "__main__":` 块中先解析好 `args`
    再调用本函数的隐含约定;当前脚本满足这一调用顺序,所以不会报错,但这是一种
    脆弱的隐式依赖(若把本函数抽出到其他脚本 import 使用,或改变调用顺序,会直接
    因 NameError 而崩溃)。这是风险点,不属于本次要修的确定性 bug,故不改动,仅在此标注。
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

    BASE_CONFIG.update(model_configs[choose_model])

    if not load_weights:
        # 随机初始化权重时固定随机种子,保证不同实验配置下的"随机基线"是可复现的
        torch.manual_seed(123)
    # disable_causal_mask: 是否关闭因果注意力掩码。关闭后模型在每个位置都能看到
    # 序列中所有 token(包括"未来"的 token),更接近双向编码器(如 BERT)的行为,
    # 用于对比"自回归/因果注意力"与"双向注意力"在分类任务上的效果差异
    model = GPTModel(BASE_CONFIG, disable_causal_mask=args.disable_causal_mask)

    if load_weights:
        # 从形如 "gpt2-small (124M)" 中提取出 "124M" 作为 gpt_download 模块所需的 model_size
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        load_weights_into_gpt(model, params)

    model.eval()
    return model


def calc_loss_batch(input_batch, target_batch, model, device,
                    trainable_token_pos=-1, ignore_index=-100, average_embeddings=False):
    """计算单个 batch 的分类交叉熵损失,支持多种"从序列输出中取分类 logits"的策略。

    Args:
        input_batch (torch.LongTensor): 输入 token id,形状 [batch_size, seq_len]。
        target_batch (torch.LongTensor): 分类标签,形状 [batch_size](0=ham, 1=spam)。
        model (torch.nn.Module): GPT 分类模型,前向输出形状 [batch_size, seq_len, num_classes]。
        device (torch.device): 计算设备。
        trainable_token_pos (int or str): 取哪个位置的输出作为分类 logits。
            - int(如 -1 表示最后一个位置,0 表示第一个位置):直接按固定下标切片;
            - "flexible":每条序列各自的最后一个非 padding token 位置(因为不同样本
              真实长度不同,padding 位置的输出不代表有效信息,需要动态定位)。
            当 `average_embeddings=True` 时此参数被忽略(见下)。
        ignore_index (int): 传给交叉熵损失的 ignore_index,标签等于该值的样本不参与损失计算。
        average_embeddings (bool): 若为 True,则对序列维度(所有 token 位置)的输出做
            平均池化得到分类 logits,而不是只取某个单一 token 位置的输出;此时会
            覆盖 `trainable_token_pos` 的取值逻辑。

    Returns:
        torch.Tensor: 标量损失值。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    if trainable_token_pos == "flexible":  # Selects the last tokens before the padding tokens
        # From https://github.com/rasbt/LLMs-from-scratch/discussions/434
        # Find the last non-padding token for each sequence in the batch
        # "flexible" 模式:因为训练数据右侧被 padding 补齐,真正有效的最后一个 token
        # 在每条样本中的位置可能不同,这里通过统计非 pad_token_id 的数量来定位
        pad_token_id = 50256  # <|endoftext|> token used for padding
        mask = input_batch != pad_token_id
        last_token_pos = mask.sum(dim=1) - 1  # Get position of last real token

        # Get model outputs
        logits = model(input_batch)  # shape: [batch_size, seq_len, num_classes]

        # Select the logits corresponding to the last real token of each sequence
        # 用 (batch 索引, 每条序列各自的最后一个有效位置) 做花式索引,取出对应 logits
        batch_size = logits.size(0)
        selected_logits = logits[torch.arange(batch_size), last_token_pos]

        # 注:此处未传入 ignore_index 参数(与下方 else 分支不一致)。由于本脚本中
        # target_batch 始终是 0/1 的分类标签,不会出现等于 ignore_index(默认 -100)
        # 的取值,所以当前不会造成实际行为差异;但这是与 else 分支不一致的地方,
        # 若未来扩展到会产生 -100 标签的场景需要注意。属于风险点,不在本次范围内修改。
        loss = torch.nn.functional.cross_entropy(selected_logits, target_batch)
        return loss

    else:
        model_output = model(input_batch)
        if average_embeddings:
            # Average over the sequence dimension (dim=1)
            # 对所有 token 位置的输出取平均,得到 [batch_size, num_classes] 的池化 logits
            logits = model_output.mean(dim=1)
        else:
            # Select embeddings at the specified token position
            # 只取指定位置(如最后一个 token)的输出作为分类 logits
            logits = model_output[:, trainable_token_pos, :]

        loss = torch.nn.functional.cross_entropy(logits, target_batch, ignore_index=ignore_index)
        return loss


def calc_loss_loader(data_loader, model, device,
                     num_batches=None, trainable_token_pos=-1,
                     ignore_index=-100, average_embeddings=False):
    """遍历 DataLoader 的若干个 batch,计算平均损失。

    Args:
        data_loader (DataLoader): 待评估的数据加载器。
        model (torch.nn.Module): 分类模型。
        device (torch.device): 计算设备。
        num_batches (int, optional): 最多评估多少个 batch;为 None 时评估全部 batch。
        trainable_token_pos, ignore_index, average_embeddings: 含义同 `calc_loss_batch`,
            会原样透传给每个 batch 的损失计算。

    Returns:
        float: 所评估 batch 的平均损失;若 data_loader 为空则返回 NaN。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        # 若指定的 num_batches 超过了实际可用的 batch 数,取二者较小值,避免越界
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                average_embeddings=average_embeddings
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad()  # Disable gradient tracking for efficiency
def calc_accuracy_loader(data_loader, model, device, num_batches=None,
                         trainable_token_pos=-1, average_embeddings=False):
    """遍历 DataLoader 的若干个 batch,计算分类准确率。

    逻辑与 `calc_loss_loader` 类似,但统计的是预测标签(argmax)与真实标签的匹配率,
    同样支持 "flexible" 取最后一个非 padding token,以及 average_embeddings 平均池化
    两种与 `calc_loss_batch` 一致的策略分支。

    Args:
        data_loader (DataLoader): 待评估的数据加载器。
        model (torch.nn.Module): 分类模型,会被临时切换到 eval() 模式。
        device (torch.device): 计算设备。
        num_batches (int, optional): 最多评估多少个 batch;为 None 时评估全部 batch。
        trainable_token_pos (int or str): 同 `calc_loss_batch`。
        average_embeddings (bool): 同 `calc_loss_batch`。

    Returns:
        float: 准确率,取值范围 [0, 1]。
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    if trainable_token_pos == "flexible":
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                # Find the last non-padding token for each sequence in the batch
                pad_token_id = 50256  # <|endoftext|> token used for padding
                mask = input_batch != pad_token_id
                last_token_pos = mask.sum(dim=1) - 1  # Get position of last real token

                logits = model(input_batch)  # Logits of last output token
                # Select the logits corresponding to the last real token of each sequence
                batch_size = logits.size(0)
                selected_logits = logits[torch.arange(batch_size), last_token_pos]
                predicted_labels = torch.argmax(selected_logits, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break

    else:
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

                predicted_labels = torch.argmax(logits, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device,
                   eval_iter, trainable_token_pos=-1,
                   ignore_index=-100, average_embeddings=False):
    """在训练过程中周期性调用:临时切到 eval 模式,分别计算训练集/验证集的平均损失。

    Args:
        model (torch.nn.Module): 分类模型。
        train_loader, val_loader (DataLoader): 训练集 / 验证集加载器。
        device (torch.device): 计算设备。
        eval_iter (int): 每次评估最多使用多少个 batch(用少量 batch 快速估计损失,
            避免每次评估都跑完整个数据集,节省时间)。
        trainable_token_pos, ignore_index, average_embeddings: 同 `calc_loss_batch`。

    Returns:
        tuple[float, float]: (train_loss, val_loss)
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
    model.train()  # 评估结束后切回训练模式(恢复 dropout 等行为),不影响外层训练循环
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None, trainable_token_pos=-1,
                            accumulation_steps=1, ignore_index=-100, average_embeddings=False):
    """分类微调的主训练循环,支持梯度累积、周期性评估和按步数提前停止。

    Args:
        model (torch.nn.Module): 分类模型,只有 requires_grad=True 的参数会被更新
            (由调用方根据 `--trainable_layers` 提前设置好哪些参数可训练)。
        train_loader, val_loader (DataLoader): 训练集 / 验证集加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        device (torch.device): 计算设备。
        num_epochs (int): 训练轮数。
        eval_freq (int): 每隔多少个 global_step 做一次训练/验证损失评估并打印。
        eval_iter (int): 每次评估使用的 batch 数上限(传给 `evaluate_model`)。
        max_steps (int, optional): 若设置,则达到该全局步数后提前终止训练
            (用于快速调试或限制实验运行时间)。
        trainable_token_pos, ignore_index, average_embeddings: 同 `calc_loss_batch`,
            决定分类 logits 取自序列的哪个位置。
        accumulation_steps (int): 梯度累积步数。当显存不足以支撑较大 batch_size 时,
            可以用较小的 batch_size 搭配多步累积来模拟大 batch 训练效果
            (参见 https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html)。

    Returns:
        tuple: (train_losses, val_losses, train_accs, val_accs, examples_seen)
            train_losses/val_losses (list[float]): 每次评估时刻记录的损失。
            train_accs/val_accs (list[float]): 每个 epoch 结束时记录的准确率。
            examples_seen (int): 训练过程中累计处理过的样本总数。
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                average_embeddings=average_embeddings
            )

            # Use gradient accumulation if accumulation_steps > 1
            # See https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html
            # for an explanation
            # 梯度累积:先把损失按累积步数缩小,多个 batch 的梯度会自然累加,
            # 等效于用累积步数倍的 batch_size 训练,但每步显存占用不变
            loss /= accumulation_steps

            loss.backward()  # Calculate loss gradients

            # Use gradient accumulation if accumulation_steps > 1
            # 只有累积够 accumulation_steps 个 batch,或者到了本 epoch 最后一个 batch,
            # 才真正执行一次参数更新
            is_update_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if is_update_step:
                optimizer.step()  # Update model weights using loss gradients
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                    trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                    average_embeddings=average_embeddings
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:
                # 达到最大步数限制,跳出内层 batch 循环(外层 epoch 循环见下方同样的判断)
                break

        # New: Calculate accuracy after each epoch
        # 每个 epoch 结束后,用同样的 eval_iter 上限估计训练/验证准确率(而非损失)
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


def replace_linear_with_lora(model, rank, alpha, alternative=False):
    """递归遍历模型的所有子模块,把每一个 `torch.nn.Linear` 替换成 LoRA 包装版本。

    Args:
        model (torch.nn.Module): 待改造的模型(或子模块),原地(in-place)修改其子模块。
        rank (int): LoRA 低秩秩数,透传给 `LoRALayer`。
        alpha (float): LoRA 缩放系数,透传给 `LoRALayer`。
        alternative (bool): 选择两种数学等价但实现不同的 LoRA 包装方式之一——
            True 时用 `LinearWithLoRAMerged`(合并权重矩阵后一次性做线性变换),
            False 时用 `LinearWithLoRA`(原始线性变换 + LoRA 分支相加)。
            对应命令行的 `--trainable_layers lora_alternative` 与 `lora` 两个选项。
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            if alternative:
                setattr(model, name, LinearWithLoRAMerged(module, rank, alpha))
            else:
                setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            # [Bug 修复] 原代码为 `replace_linear_with_lora(module, rank, alpha)`,
            # 递归调用时遗漏了 `alternative` 参数,导致其静默使用默认值 False。
            # 由于 GPTModel 顶层只有 `out_head` 是直接的 Linear 子模块,其余 Linear
            # (各 Transformer block 内部的 W_query/W_key/W_value/out_proj/前馈层等)
            # 都位于更深层级、需要经过本函数递归才能访问到。原代码这一遗漏会使得
            # 无论用户传入 `--trainable_layers lora` 还是 `lora_alternative`,
            # 除 out_head 外的所有 Linear 层永远被替换为 `LinearWithLoRA`(alternative=False),
            # `lora_alternative` 选项对绝大多数参数实际上不起作用,是一个确定性 bug。
            # 修复方式:递归调用时把 alternative 原样透传下去。
            replace_linear_with_lora(module, rank, alpha, alternative=alternative)


if __name__ == "__main__":

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
            # 决定"微调哪些参数"这一实验变量:
            # last_layer:只训练新加的分类头(out_head);
            # last_block/last_two_blocks:额外解冻最后一个/两个 Transformer block 及 final_norm;
            # all:全参数微调;
            # lora/lora_alternative:用 LoRA 低秩适配所有 Linear 层(两种等价实现)
            "Which layers to train. Options: 'all', 'last_block', 'last_two_blocks', 'last_layer', 'lora', 'lora_alternative'."
        )
    )
    parser.add_argument(
        "--trainable_token_pos",
        type=str,
        default="last",
        help=(
            # 决定"用序列中哪个位置的输出做分类"这一实验变量,详见 calc_loss_batch 中的说明
            "Which token position to train. Options: 'first', 'last', 'flexible'."
        )
    )
    parser.add_argument(
        "--average_embeddings",
        action="store_true",
        default=False,
        help=(
            # 若开启,则忽略 --trainable_token_pos,改为对所有 token 输出做平均池化
            "Average the output embeddings from all tokens instead of using"
            " only the embedding at the token position specified by `--trainable_token_pos`."
        )
    )
    parser.add_argument(
        "--context_length",
        type=str,
        default="longest_training_example",
        help=(
            "The context length of the data inputs."
            " Options: 'longest_training_example', 'model_context_length' or integer value."
        )
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help=(
            "The LoRA rank when choosing `--trainable_layers lora`"
        )
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help=(
            "The LoRA alpha value when choosing `--trainable_layers lora`"
        )
    )
    parser.add_argument(
        "--no_padding",
        action="store_true",
        default=False,
        help=(
            # 注意:此处仅在帮助文本中说明"需要配合 batch_size=1 使用",但代码里
            # 并未对此做强制校验(既没有 assert,也没有在 batch_size>1 时报错)。
            # 若用户开启 --no_padding 却仍使用默认的 batch_size=8,default collate_fn
            # 会因样本长度不一致而在拼 batch 时报错。这是一个使用风险点(而非本次
            # 修复范围内的"确定性 bug",因为按文档说明正确使用不会触发),故只标注不改动。
            "Disable padding, which means each example may have a different length."
            " This requires setting `--batch_size 1`."
        )
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help=(
            "Number of training epochs."
        )
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=(
            "The batch size used for training."
        )
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help=(
            "Accumulation steps to allow for gradient accumulation."
            " See https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html for explanation."
            " For example, setting `batch_size=8` and `accumulation_steps=1` compute the exact same"
            " loss and weight updates as setting `batch_size=1` and `accumulation_steps=8`, however,"
            " the latter setting uses more iterations."
        )
    )
    parser.add_argument(
        "--disable_causal_mask",
        action="store_true",
        default=False,
        help=(
            # 关闭后模型注意力变为"双向"(可看到未来 token),用于对比因果/双向注意力
            # 对分类任务效果的影响
            "Disables the causal attention mask."
        )
    )
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=-100,
        help=(
            "Sets the `ignore_index` in the cross-entropy loss."
        )
    )

    args = parser.parse_args()

    if args.trainable_token_pos == "first":
        args.trainable_token_pos = 0
    elif args.trainable_token_pos == "last":
        args.trainable_token_pos = -1
    # The "flexible" setting selects the last tokens before the padding tokens
    # See https://github.com/rasbt/LLMs-from-scratch/discussions/434
    elif args.trainable_token_pos == "flexible":
        args.trainable_token_pos = "flexible"
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

    # 构造模型(内部会读取全局 args.disable_causal_mask,见 instantiate_model 的说明)
    model = instantiate_model(args.model_size, load_weights)
    # 先冻结全部参数,再按 --trainable_layers 的选择逐步解冻需要训练的部分
    for param in model.parameters():
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
    # 把语言模型头替换为二分类头(spam/ham 两类),新层默认 requires_grad=True
    model.out_head = torch.nn.Linear(in_features=in_features, out_features=2)

    if args.trainable_layers == "last_layer":
        # 只训练新加的分类头,其余保持冻结
        pass
    elif args.trainable_layers == "last_block" or args.trainable_layers == "last_two_blocks":
        # 额外解冻最后一个 Transformer block 以及最终的 LayerNorm
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
        if args.trainable_layers == "last_two_blocks":
            # 若选择 last_two_blocks,再额外解冻倒数第二个 block
            for param in model.trf_blocks[-2].parameters():
                param.requires_grad = True
    elif args.trainable_layers == "all":
        # 全参数微调
        for param in model.parameters():
            param.requires_grad = True
    elif args.trainable_layers in ("lora", "lora_alternative"):
        # LoRA 微调:先冻结的所有 Linear 层的原始权重保持 requires_grad=False,
        # 替换后新增的 LoRA 参数(A、B)默认是 requires_grad=True 的新 Parameter,
        # 因此天然只有 LoRA 分支参与训练
        if args.trainable_layers == "lora_alternative":
            alternative = True
        else:
            alternative = False
        replace_linear_with_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, alternative=alternative)
    else:
        raise ValueError("Invalid --trainable_layers argument.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ###############################
    # Instantiate dataloaders
    ###############################

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extract_to = "sms_spam_collection"
    new_file_path = Path(extract_to) / "SMSSpamCollection.tsv"

    base_path = Path(".")
    file_names = ["train.csv", "validation.csv", "test.csv"]
    all_exist = all((base_path / file_name).exists() for file_name in file_names)

    if not all_exist:
        try:
            download_and_unzip(url, zip_path, extract_to, new_file_path)
        except (requests.exceptions.RequestException, TimeoutError) as e:
            # 主下载地址失败时(例如网络问题或 archive.ics.uci.edu 不可访问),
            # 自动切换到项目自建的备用下载地址重试一次
            print(f"Primary URL failed: {e}. Trying backup URL...")
            backup_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
            download_and_unzip(backup_url, zip_path, extract_to, new_file_path)
        create_dataset_csvs(new_file_path)

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = None

    if args.no_padding:
        # 不做 padding 时无需统一的 max_length,由 SpamDataset 内部按各自实际长度处理
        max_length = None

    else:
        if args.context_length == "model_context_length":
            # 使用模型支持的最大上下文长度(即位置编码表的行数)作为统一长度
            max_length = model.pos_emb.weight.shape[0]
        elif args.context_length == "longest_training_example":
            # 先构造一次训练集(不指定 max_length,触发自动统计最长样本长度),
            # 再用统计出的长度作为 val/test 数据集的统一 max_length
            train_dataset = SpamDataset(base_path / "train.csv", max_length=None, tokenizer=tokenizer, no_padding=args.no_padding)
            max_length = train_dataset.max_length
        else:
            try:
                # 允许直接传入一个整数,自定义固定的上下文长度
                max_length = int(args.context_length)
            except ValueError:
                raise ValueError("Invalid --context_length argument")

    if train_dataset is None:
        train_dataset = SpamDataset(base_path / "train.csv", max_length=max_length, tokenizer=tokenizer, no_padding=args.no_padding)
    val_dataset = SpamDataset(base_path / "validation.csv", max_length=max_length, tokenizer=tokenizer, no_padding=args.no_padding)
    test_dataset = SpamDataset(base_path / "test.csv", max_length=max_length, tokenizer=tokenizer, no_padding=args.no_padding)

    num_workers = 0

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    # 校验数据集的最大长度没有超出模型支持的上下文长度上限,否则位置编码会越界
    assert train_dataset.max_length <= model.pos_emb.weight.shape[0], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {model.pos_emb.weight.shape[0]}. Reinitialize data sets with "
        f"`max_length={model.pos_emb.weight.shape[0]}`"
    )

    ###############################
    # Train model
    ###############################

    start_time = time.time()
    torch.manual_seed(123)
    # 只有 requires_grad=True 的参数会被优化器更新,具体取决于之前 --trainable_layers 的设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=5,
        max_steps=None, trainable_token_pos=args.trainable_token_pos,
        accumulation_steps=args.accumulation_steps, average_embeddings=args.average_embeddings,
        # [Bug 修复] 原代码此处未传入 ignore_index 参数,导致 train_classifier_simple 内部
        # 始终使用其默认值 -100,用户通过命令行 `--ignore_index` 传入的值会被完全忽略、
        # 从未生效——这是一个确定性 bug(参数被解析却从未被使用/传递)。
        # 修复方式:显式传入 args.ignore_index,使命令行开关真正生效。
        ignore_index=args.ignore_index
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    ###############################
    # Evaluate model
    ###############################

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
