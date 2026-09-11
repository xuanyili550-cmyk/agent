# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块与 ch05/01_main-chapter-code/gpt_train.py 几乎完全一致（同样的训练循环、
同样的数据加载、同样的评估与采样逻辑），唯一的核心区别在于：**优化器换成了
Muon + AdamW 的混合方案**，而不是单纯的 AdamW。

中文模块说明——Muon 与 AdamW 的关键差异：

1. AdamW 是什么：
   - 对每个参数逐元素维护一阶矩（动量）和二阶矩（梯度平方的滑动平均），
     并用二阶矩做自适应学习率缩放，再加上解耦的权重衰减（decoupled weight decay）。
   - 它把每个标量参数当作独立个体处理，完全不关心参数矩阵的"形状/结构"信息。
   - 适用于几乎任何形状的参数：1D 的偏置(bias)、LayerNorm 的 scale/shift、
     以及 Embedding 表这种"按行查表"的参数（每一行是独立的词向量，不应该被
     "正交化"这种整体几何变换搅在一起）。

2. Muon 是什么（本文件的重点）：
   - Muon 全称 "MomentUm Orthogonalized by Newton-schulz"，是专门为**二维权重矩阵**
     （比如 nn.Linear 的 weight）设计的优化器，核心思想：
       a) 先用类似 SGD-momentum 的方式得到一个"原始更新方向"矩阵 G（动量项）；
       b) 再用 **Newton-Schulz 迭代**（一种不需要做 SVD/特征分解、只用矩阵乘法
          反复逼近的数值方法）把 G 正交化，得到一个新的矩阵 O，使得 O 的所有
          奇异值都接近 1（即 O 近似是一个正交/半正交矩阵，只保留"方向"信息，
          丢掉了各奇异值之间原本悬殊的大小差异）；
       c) 用正交化后的 O（而不是原始梯度/动量 G）去更新权重矩阵。
   - 直觉：对一个二维权重矩阵而言，梯度更新中不同奇异值方向上的"步长"往往
     很不均衡（某些方向更新过猛、某些方向几乎不动），Newton-Schulz 正交化相当于
     把更新方向"拉平"成近似各向同性（各个方向步长趋于一致），从而让训练更稳定、
     收敛更快，尤其在大模型的隐藏层权重矩阵上效果显著。
   - 因为这套"正交化"操作只有对**二维矩阵**才有意义（需要能计算 W^T W 或
     W W^T 这样的矩阵乘法去做 Newton-Schulz 迭代），所以 Muon **不适用于**：
       - 1D 参数（偏置 bias、LayerNorm 的 scale/shift）——没有"矩阵正交化"的概念；
       - Embedding 查表参数——虽然形状是二维 (vocab_size, emb_dim)，但语义上
         每一行是独立的词向量，将其正交化会破坏词向量之间原本的相对几何关系，
         因此惯例上把 Embedding 交给 AdamW 处理。
   - 因此本文件采用"混合优化器"策略：模型中所有满足
       ndim == 2 且不是 Embedding 参数
     的权重矩阵（也就是各层 nn.Linear 的 weight：注意力的 Q/K/V/输出投影、
     前馈网络的两个线性层、输出头 out_head 等）交给 Muon；其余参数
     （Embedding 权重、所有 1D 的 bias/LayerNorm scale/shift）交给 AdamW。
   - 由于 Muon 和 AdamW 是两个完全独立的优化器对象、各自持有互不重叠的参数
     子集，训练循环里需要对"一组优化器"（一个 list）分别调用
     zero_grad()/step()，而不是像单优化器场景那样只调用一次。

3. 学习率的可比性：
   - Muon 更新后的矩阵在数值尺度上和 AdamW 的更新方式不同（因为一个是正交化后
     的方向，一个是逐元素自适应缩放的方向），所以两者通常需要**不同量级的学习率**
     （本文件用 muon_learning_rate 和 adamw_learning_rate 两个独立超参数）。
   - `adjust_lr_fn="match_rms_adamw"` 这个开关的作用是让 PyTorch 内部按照参数矩阵的
     形状自动缩放 Muon 的有效学习率，使其更新幅度的均方根（RMS）与 AdamW 的更新
     量级对齐，方便调参时把两者放在同一个数量级上比较/迁移超参数。
"""

import argparse
import matplotlib.pyplot as plt
import os
import requests
import torch
import tiktoken


# Import from local files
from previous_chapters import GPTModel, create_dataloader_v1, generate_text_simple


def text_to_token_ids(text, tokenizer):
    """将原始文本编码为 token id 张量，并添加 batch 维度。

    参数：
        text (str): 原始文本。
        tokenizer: 拥有 .encode(str) -> List[int] 方法的分词器（此处为 tiktoken 的 gpt2 编码）。
    返回：
        torch.Tensor，形状 (1, seq_len)，为后续模型输入所需的 batch 化 token id。
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension  中文：增加 batch 维度，形状变为 (1, seq_len)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将模型输出/生成的 token id 张量解码回文本字符串。

    参数：
        token_ids (torch.Tensor): 形状 (1, seq_len) 的 token id。
        tokenizer: 拥有 .decode(List[int]) -> str 方法的分词器。
    返回：
        str，解码后的文本。
    """
    flat = token_ids.squeeze(0)  # remove batch dimension  中文：去掉 batch 维度，变回一维 token 序列
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个 batch 的交叉熵损失（下一个 token 预测任务）。

    参数：
        input_batch (torch.Tensor): 形状 (batch_size, seq_len) 的输入 token id。
        target_batch (torch.Tensor): 形状 (batch_size, seq_len) 的目标 token id（即 input 整体右移一位）。
        model (nn.Module): GPT 模型，前向输出 logits 形状 (batch_size, seq_len, vocab_size)。
        device (torch.device): 计算设备。
    返回：
        标量 torch.Tensor，该 batch 的平均交叉熵损失。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # 中文：把 (batch, seq_len, vocab) 展平成 (batch*seq_len, vocab)，
    # 把 target 展平成 (batch*seq_len,)，交叉熵在展平后的“样本”维度上逐位置计算再取平均。
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """在整个（或部分）DataLoader 上计算平均损失，用于训练/验证集评估。

    参数：
        data_loader (DataLoader): 训练或验证集的 DataLoader。
        model (nn.Module): GPT 模型。
        device (torch.device): 计算设备。
        num_batches (int | None): 最多评估多少个 batch；None 表示遍历整个 loader。
    返回：
        float，平均损失；若 data_loader 为空则返回 float("nan")。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))  # 中文：num_batches 不能超过 loader 实际长度
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """在训练/验证集上各评估 eval_iter 个 batch，返回 (train_loss, val_loss)。

    评估过程中会临时切到 eval 模式（关闭 Dropout）并禁用梯度计算，
    评估结束后再切回 train 模式，避免影响后续继续训练。
    """
    model.eval()  # 中文：切换到评估模式，关闭 Dropout 等训练专属行为
    with torch.no_grad():  # 中文：评估不需要反向传播，禁用梯度计算以节省显存/加速
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 中文：评估完毕，切回训练模式
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """用当前模型对 start_context 做一次贪心解码生成，并打印结果，用于直观观察训练效果。"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]  # 中文：位置嵌入表的行数即模型支持的最大上下文长度
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def create_muon_optimizers(model, adamw_learning_rate, muon_learning_rate, weight_decay=0.1):
    """按参数形状把模型参数拆分给 Muon 和 AdamW 两个优化器，构造混合优化器方案。

    中文说明——这是本文件与普通 AdamW 训练脚本相比最核心的差异所在：

    拆分规则：
        - 参数是二维（ndim == 2）且不属于任何 nn.Embedding 模块
          （即排除 tok_emb.weight / pos_emb.weight 这类查表参数）
          -> 交给 Muon（典型对象：各层 nn.Linear 的 weight，包括注意力的
             Q/K/V/输出投影、前馈网络两层线性变换、输出头 out_head.weight）。
        - 其余所有参数（1D 的 bias、LayerNorm 的 scale/shift、以及 Embedding
          权重本身）-> 交给 AdamW。
      原因见模块顶部 docstring：Muon 的 Newton-Schulz 正交化只对“二维矩阵”
      有意义，且不适合用在按行取值的 Embedding 表上。

    参数：
        model (nn.Module): 待训练的 GPT 模型。
        adamw_learning_rate (float): AdamW 部分参数使用的学习率。
        muon_learning_rate (float): Muon 部分参数使用的学习率
            （Muon 更新经过正交化，量级与 AdamW 不同，通常需要单独调参）。
        weight_decay (float): 两个优化器共用的权重衰减系数。
    返回：
        List[torch.optim.Optimizer]：包含 1~2 个优化器的列表
        （若模型参数全部落在某一类，则列表中只会有对应的那一个优化器）。
    异常：
        RuntimeError: 当前安装的 PyTorch 版本没有 torch.optim.Muon 时抛出
            （Muon 是较新加入 PyTorch 的优化器，旧版本 PyTorch 不含此类）。
            —— 跨版本风险点：脚本不会自动降级为纯 AdamW，而是直接报错，
            使用者需要升级 PyTorch 或改用 ch05/18_muon/gpt_train.py（纯 AdamW 版本）。
        ValueError: 模型没有任何可训练参数时抛出（理论上不会发生，防御性检查）。
    """
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon is not available. Please update to a more recent PyTorch version.")

    # 中文：先收集所有 nn.Embedding 子模块的参数全名（如 "tok_emb.weight"、"pos_emb.weight"），
    # 后面用这个集合把 Embedding 参数从"二维矩阵一律走 Muon"的规则中排除出去。
    embedding_param_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                embedding_param_names.add(full_name)

    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and name not in embedding_param_names:
            # 中文：二维且非 Embedding -> 典型的 nn.Linear.weight，交给 Muon 做正交化动量更新
            muon_params.append(param)
        else:
            # 中文：其余情况（1D bias/LayerNorm 参数，或 Embedding 权重）交给 AdamW
            adamw_params.append(param)

    optimizers = []
    if muon_params:
        optimizers.append(
            torch.optim.Muon(
                # 中文：adjust_lr_fn="match_rms_adamw" —— 让 PyTorch 按参数矩阵形状自动缩放
                # Muon 的有效学习率，使更新幅度的 RMS 与 AdamW 对齐，便于两个优化器的学习率互相参照。
                muon_params, lr=muon_learning_rate, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw"
            )
        )
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=adamw_learning_rate, weight_decay=weight_decay))
    if not optimizers:
        raise ValueError("No trainable parameters found.")
    return optimizers


def train_model_simple(model, train_loader, val_loader, optimizers, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, max_steps=None):
    """最简训练循环：对 train_loader 做多轮（num_epochs）遍历，每一步用（可能是多个的）
    优化器更新参数，定期在训练/验证集上评估并打印 loss，每个 epoch 结束后打印一次采样文本。

    与纯 AdamW 版本（ch05/18_muon/gpt_train.py）相比，这里的关键差异是：
        - optimizers 参数是一个**优化器列表**而不是单个优化器对象
          （因为 Muon 和 AdamW 分别持有不重叠的参数子集，必须都参与
          zero_grad() 和 step()，缺一个都会导致对应那部分参数不更新）；
        - 新增 max_steps，用于（例如在 CI/演示中）限制总的优化器步数，
          不必等一个完整 epoch 跑完。

    参数：
        model (nn.Module): 待训练模型。
        train_loader / val_loader (DataLoader): 训练 / 验证数据加载器。
        optimizers (List[torch.optim.Optimizer]): 一个或多个优化器
            （本文件里通常是 [Muon优化器, AdamW优化器]）。
        device (torch.device): 计算设备。
        num_epochs (int): 训练轮数。
        eval_freq (int): 每隔多少个 global_step 做一次训练/验证集评估。
        eval_iter (int): 每次评估时各在 train/val 上取多少个 batch。
        start_context (str): 每个 epoch 结束后用于生成示例文本的起始上下文。
        tokenizer: 分词器。
        max_steps (int | None): 若不为 None，达到该优化器步数后立即停止训练
            （可能在 epoch 中途、也可能在 epoch 边界处停止）。
    返回：
        (train_losses, val_losses, track_tokens_seen)：三个等长列表，记录每次评估时的
        训练损失、验证损失，以及截至该次评估已经"看过"的 token 总数，供后续画图使用。
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            for optimizer in optimizers:
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
                # 中文：Muon 和 AdamW 各自管理的参数互不重叠，因此必须对"每一个"优化器都清零梯度，
                # 只清零其中一个会导致另一部分参数的梯度在多个 batch 间错误累积。
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients  中文：一次反向传播即可为所有参数（无论后面走 Muon 还是 AdamW）填好 .grad
            for optimizer in optimizers:
                optimizer.step()  # Update model weights using loss gradients
                # 中文：依次调用每个优化器的 step()——Muon 优化器内部会对它负责的那部分
                # 二维权重矩阵的动量做 Newton-Schulz 正交化后再更新；AdamW 优化器则按常规的
                # 一阶/二阶矩自适应方式更新它负责的那部分参数（bias、LayerNorm、Embedding 等）。
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step + 1 >= max_steps:
                # 中文：global_step 从 -1 起步，完成第 N 个优化步后 global_step == N-1，
                # 因此这里用 global_step + 1 表示"已经完成的步数"，达到 max_steps 就提前退出内层循环。
                break

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

        if max_steps is not None and global_step + 1 >= max_steps:
            break  # 中文：内层循环因 max_steps 提前退出后，外层 epoch 循环也要一并退出

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """绘制训练/验证损失曲线，横轴同时标注 epoch 数和已见 token 数（双 x 轴）。"""
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks  中文：alpha=0 使这条曲线不可见，只用来对齐上方 token 数刻度
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def main(gpt_config, settings):
    """端到端训练入口：下载/加载数据 -> 构建模型与 Muon+AdamW 混合优化器 -> 构建
    训练/验证 DataLoader -> 执行训练循环，最终返回训练过程中的损失记录和训练好的模型。
    """

    torch.manual_seed(123)  # 中文：固定随机种子，保证模型初始化/数据打乱等过程可复现
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # Download data if necessary
    ##############################

    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizers = create_muon_optimizers(
        model,
        adamw_learning_rate=settings["learning_rate"],
        muon_learning_rate=settings["muon_learning_rate"],
        weight_decay=settings["weight_decay"]
    )
    # 中文：注意这里得到的是"优化器列表"（通常是 [Muon, AdamW] 两个），
    # 后面 train_model_simple 会对列表里的每一个优化器分别做 zero_grad()/step()。

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizers, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer,
        max_steps=settings.get("train_steps")
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--train_steps",
        type=int,
        default=None,
        help="Maximum number of optimizer steps to run. If omitted, trains for all epochs."
    )
    args = parser.parse_args()

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-5,       # 中文：AdamW 部分（bias/LayerNorm/Embedding）的学习率
        "muon_learning_rate": 1e-4,  # 中文：Muon 部分（各 Linear 层二维权重矩阵）的学习率，与 AdamW 学习率量级不同、需分开设置
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1,
        "train_steps": args.train_steps,
    }

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

    ###########################
    # After training
    ###########################

    # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(GPT_CONFIG_124M)
    # 风险点（未修改，原书代码同款写法）：torch.load 未传 map_location，若保存时模型在 GPU 上、
    # 而重新加载时所在环境没有可用 CUDA 设备，这里会抛出反序列化错误；
    # 如需在不同设备间迁移，应改为 torch.load("model.pth", map_location=device, weights_only=True)。
    model.load_state_dict(torch.load("model.pth", weights_only=True))
