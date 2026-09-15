"""
SDXL 风格 LoRA 训练脚本，改编自 diffusers 的 dreambooth_lora_sdxl 示例。

在流水线中的位置：05_TRAINING/style_lora。读取 04_DATASET 的 style_manifest.jsonl，按 ``style_id`` 筛出某一种视觉风格
的全部参考图，训练一个只挂在 UNet 注意力层上的 LoRA；产物供 07/08 的图像生成阶段加载，让整部剧的画面保持统一调色与质感。

与角色 LoRA 的区别：风格 LoRA 学的是"整体的色调 / 光影 / 质感"，而不是某个具体主体的身份，所以：
- 不做 prior-preservation（无需 class images）——没有"防止把某个概念坍缩到新 token"的需求；
- 训练 prompt = 原始 caption + 风格触发词，让模型把风格绑定到触发词上，推理时加触发词即可唤起该风格；
- LoRA rank 默认 8（角色 LoRA 是 16）：风格属于低频、全局的特征，需要的容量更小，rank 太大反而容易把内容也学进去。

示例：
  python train_style_lora_sdxl.py \\
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \\
    --manifest_path ../../04_DATASET/metadata/style_manifest.jsonl \\
    --style_id style_cinematic_moody \\
    --style_trigger_token "in sks_cinematic_moody style" \\
    --output_dir ./out/style_cinematic_moody_lora
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, CLIPTextModelWithProjection, PretrainedConfig

# 04_DATASET 目录名带数字前缀，不能常规导入，只能加进 sys.path
DATASET_DIR = Path(__file__).resolve().parents[2] / "04_DATASET"
sys.path.insert(0, str(DATASET_DIR))
from dataset_loaders import build_style_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    默认值取舍：``resolution=1024`` 是 SDXL 的原生分辨率；``lora_rank=8, lora_alpha=8``（alpha/r=1）对风格这类全局特征
    容量足够；``batch=1 x grad_accum=4`` 让 1024 分辨率能在单张 24G 显卡上跑；``mixed_precision=fp16`` 减半显存。
    """
    parser = argparse.ArgumentParser(description="SDXL style LoRA training")
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--manifest_path", required=True, help="path to style_manifest.jsonl")
    parser.add_argument("--style_id", required=True, help="style_id to filter the manifest on")
    parser.add_argument("--style_trigger_token", required=True, help="appended to each caption during training")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class StyleLoraDataset(Dataset):
    """把筛好的 HF 风格数据集包装成 PyTorch Dataset：输出归一化到 [-1, 1] 的图像张量 + 带触发词的 prompt。"""

    def __init__(self, hf_dataset, style_trigger_token: str, resolution: int):
        """把 HF 数据集一次性物化成列表（样本量小，换取随机索引访问的速度），并记录触发词与目标分辨率。"""
        self.examples = [ex for ex in hf_dataset]
        self.style_trigger_token = style_trigger_token
        self.resolution = resolution

    def __len__(self) -> int:
        """样本数 = 该 style_id 的参考图数量。"""
        return len(self.examples)

    def __getitem__(self, index):
        """读取第 ``index`` 张图：转 RGB、缩放到正方形、变成 CHW 张量并做 (x/127.5 - 1) 归一化；prompt 末尾拼上风格触发词。"""
        example = self.examples[index]
        pil_image = example["image"].convert("RGB").resize((self.resolution, self.resolution))
        # 直接从 PIL 的原始字节构造张量，绕开 numpy 中间拷贝；HWC -> CHW 后归一化到 [-1, 1]（VAE 的输入范围）
        tensor = (
            torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(pil_image.tobytes()))).reshape(self.resolution, self.resolution, 3).numpy())
            .permute(2, 0, 1)
            .float()
            / 127.5
            - 1.0
        )
        prompt = f"{example['caption']}, {self.style_trigger_token}"
        return {"image": tensor, "prompt": prompt}


def load_text_encoder_class(pretrained_model_name_or_path: str, subfolder: str):
    """根据模型 config 里声明的 architectures 决定用哪个文本编码器类。

    SDXL 有两个文本编码器：text_encoder 是 CLIPTextModel，text_encoder_2 是 CLIPTextModelWithProjection，
    不能写死同一个类，所以先读 config 再选。
    """
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
    architecture = text_encoder_config.architectures[0]
    if architecture == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    if architecture == "CLIPTextModelWithProjection":
        return CLIPTextModelWithProjection
    raise ValueError(f"unsupported text encoder architecture: {architecture}")


def encode_prompt(tokenizers, text_encoders, prompt: str, device):
    """用 SDXL 的双文本编码器编码 prompt，返回 (拼接后的序列嵌入, pooled 嵌入)。

    - 序列嵌入取倒数第二层 hidden_states（SDXL 训练时就是这么用的，最后一层效果更差）并在特征维拼接两个编码器的输出；
    - pooled 嵌入只保留第二个编码器的（循环结束后变量里就是它），作为 UNet 的 ``text_embeds`` 条件。
    """
    prompt_embeds_list = []
    pooled_prompt_embeds = None
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        output = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = output[0]
        prompt_embeds_list.append(output.hidden_states[-2])
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def main() -> None:
    """训练主流程：加载 SDXL 各组件 -> 冻结全部权重、只在 UNet 注意力上挂 LoRA -> 标准噪声预测 MSE 训练 -> 保存 LoRA。"""
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]

    # SDXL 有两套 tokenizer + 文本编码器；use_fast=False 与 diffusers 官方示例保持一致，避免分词差异
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)
    text_encoder_cls_one = load_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder")
    text_encoder_cls_two = load_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder_2")
    text_encoder_one = text_encoder_cls_one.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_two = text_encoder_cls_two.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # 冻结所有底座权重，可训练参数只来自后面挂上的 LoRA
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # 只在注意力的 q/k/v/out 投影上挂 LoRA：这是 diffusers 官方推荐的最小集合，足以学到风格且产物很小
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)

    hf_dataset = build_style_dataset(manifest_path=args.manifest_path)
    hf_dataset = hf_dataset.filter(lambda ex: ex["style_id"] == args.style_id)
    if len(hf_dataset) == 0:
        raise ValueError(f"no rows found for style_id={args.style_id!r} in {args.manifest_path}")

    train_dataset = StyleLoraDataset(hf_dataset, args.style_trigger_token, args.resolution)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # 只把 requires_grad 的参数（即 LoRA 权重）交给优化器
    optimizer = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=args.learning_rate)

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
    # 冻结的模块不经过 accelerator.prepare，手动搬到目标设备并转成半精度以省显存
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    global_step = 0
    # 以"优化步数"而非 epoch 计数：样本很少时一个 epoch 只有几步，按步数控制更直观
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # 图像 -> VAE 潜空间，并乘 scaling_factor 使潜变量方差接近 1（SDXL 训练约定）
                pixel_values = batch["image"].to(accelerator.device, dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                # 标准 DDPM 训练目标：随机选时间步加噪，让 UNet 预测所加的噪声
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # batch_size 固定为 1，所以直接取 batch["prompt"][0]
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    [tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two], batch["prompt"][0], accelerator.device
                )
                # SDXL 的微条件：(原图高, 原图宽, 裁剪 top, 裁剪 left, 目标高, 目标宽)；这里图已缩放成正方形、无裁剪
                add_time_ids = torch.tensor(
                    [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

                model_pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample
                # 转 float 再算 loss，避免 fp16 下 MSE 溢出 / 精度损失
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # 梯度累积期间 sync_gradients 为 False，只有真正更新参数时才算一步
            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % 10 == 0:
                    print(f"step {global_step}/{args.max_train_steps} loss={loss.item():.4f}")
            if global_step >= args.max_train_steps:
                break

    # 多卡时只由主进程保存；unwrap 去掉 accelerate 的分布式包装后再调 peft 的 save_pretrained（只存 LoRA 权重）
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(args.output_dir)
        print(f"saved style LoRA to {args.output_dir}")


if __name__ == "__main__":
    main()
