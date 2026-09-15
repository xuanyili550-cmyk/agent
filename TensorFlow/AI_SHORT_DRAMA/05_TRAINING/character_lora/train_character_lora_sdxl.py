"""
SDXL 角色 LoRA 训练脚本（DreamBooth 风格），改编自 diffusers 的 dreambooth_lora_sdxl 示例。

在流水线中的位置：05_TRAINING/character_lora。读取 04_DATASET 的 character_manifest.jsonl（而不是普通图片文件夹），
按 ``character_id`` 筛出某一个角色的参考图，为该角色单独训练一个挂在 UNet 上的 LoRA。产物供图像生成阶段加载，
保证同一角色在整部剧的每个镜头里长相一致——这是竖屏短剧"角色一致性"的核心手段。

为什么用 DreamBooth 配方：角色 LoRA 要学的是"一个具体主体的身份"，训练 prompt 统一用带罕见触发词的
``instance_prompt``（如 "a photo of sks_lin_wei person"），把身份绑定到触发词上。

可选的 prior-preservation（class images）：只用几张同一人物的图训练，模型容易把整个 "person" 概念都坍缩成这个人；
加入一批同类别的普通图（class images）并对其计算 prior loss，可以保住底座对 "person" 的泛化能力。
class images 必须由用户事先准备好，本脚本不会生成或下载。

示例：
  python train_character_lora_sdxl.py \\
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \\
    --manifest_path ../../04_DATASET/metadata/character_manifest.jsonl \\
    --character_id char_lin_wei \\
    --instance_prompt "a photo of sks_lin_wei person" \\
    --output_dir ./out/char_lin_wei_lora
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
from dataset_loaders import build_character_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    默认值取舍：``lora_rank=16, lora_alpha=16``（比风格 LoRA 的 8 大一倍）——人脸 / 体态这类身份特征细节多，
    需要更大的低秩容量才能稳定复现；``resolution=1024`` 是 SDXL 原生分辨率；``batch=1 x grad_accum=4`` 适配单卡显存；
    ``prior_loss_weight=1.0`` 沿用 DreamBooth 论文设定，让实例损失与先验损失同权。
    """
    parser = argparse.ArgumentParser(description="SDXL character LoRA training (DreamBooth-style)")
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--manifest_path", required=True, help="path to character_manifest.jsonl")
    parser.add_argument("--character_id", required=True, help="character_id to filter the manifest on")
    parser.add_argument("--instance_prompt", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)

    # prior-preservation 相关参数：三者需一起提供才会生效
    parser.add_argument("--with_prior_preservation", action="store_true")
    parser.add_argument("--class_data_dir", default=None, help="folder of pre-existing class images (user-supplied)")
    parser.add_argument("--class_prompt", default=None)
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)

    return parser.parse_args()


class CharacterLoraDataset(Dataset):
    """包装按 character_id 筛好的 HF 角色数据集，可选地与 prior-preservation 的 class images 交错配对。

    每个样本固定返回 instance 图 + instance prompt；开启 class images 时再附带一张 class 图 + class prompt，
    训练循环据此同时计算实例损失与先验损失。
    """

    def __init__(self, hf_dataset, instance_prompt: str, resolution: int, class_data_dir=None, class_prompt=None):
        """物化实例样本列表，并在给定 ``class_data_dir`` 时收集（排序后的）class 图片路径，保证遍历顺序可复现。"""
        self.instance_examples = [ex for ex in hf_dataset]
        self.instance_prompt = instance_prompt
        self.resolution = resolution
        self.class_image_paths = sorted(Path(class_data_dir).glob("*")) if class_data_dir else []
        self.class_prompt = class_prompt

    def __len__(self) -> int:
        """有 class 图时长度取两者较大值，让数量较少的一方通过取模循环复用，保证每张图都被看到。"""
        return max(len(self.instance_examples), len(self.class_image_paths)) if self.class_image_paths else len(self.instance_examples)

    def _prepare_image(self, pil_image):
        """PIL 图 -> 转 RGB、缩放到正方形 -> CHW 张量并做 (x/127.5 - 1) 归一化到 [-1, 1]（VAE 的输入范围）。"""
        pil_image = pil_image.convert("RGB").resize((self.resolution, self.resolution))
        # 直接从 PIL 原始字节构造张量，绕开 numpy 中间拷贝
        tensor = (
            torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(pil_image.tobytes()))).reshape(self.resolution, self.resolution, 3).numpy())
            .permute(2, 0, 1)
            .float()
            / 127.5
            - 1.0
        )
        return tensor

    def __getitem__(self, index):
        """返回第 ``index`` 个样本；实例图与 class 图各自按自身长度取模，所以两边数量不等也能配对。"""
        instance_example = self.instance_examples[index % len(self.instance_examples)]
        item = {
            "instance_image": self._prepare_image(instance_example["image"]),
            "instance_prompt": self.instance_prompt,
        }
        if self.class_image_paths:
            # 局部导入：只有启用 class images 时才需要 PIL 直接开文件（实例图由 HF datasets 解码）
            from PIL import Image as PILImage

            class_path = self.class_image_paths[index % len(self.class_image_paths)]
            item["class_image"] = self._prepare_image(PILImage.open(class_path))
            item["class_prompt"] = self.class_prompt
        return item


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

    - 序列嵌入取倒数第二层 hidden_states（SDXL 训练时就是这么用的）并在特征维拼接两个编码器的输出；
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
    """训练主流程：加载 SDXL 各组件 -> 冻结底座、只在 UNet 注意力上挂 LoRA -> 噪声预测 MSE（可选 + 先验损失）训练 -> 保存 LoRA。"""
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

    # 只在注意力的 q/k/v/out 投影上挂 LoRA：diffusers 官方推荐的最小集合，身份特征主要通过交叉注意力注入
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)

    hf_dataset = build_character_dataset(manifest_path=args.manifest_path)
    hf_dataset = hf_dataset.filter(lambda ex: ex["character_id"] == args.character_id)
    if len(hf_dataset) == 0:
        raise ValueError(f"no rows found for character_id={args.character_id!r} in {args.manifest_path}")

    train_dataset = CharacterLoraDataset(
        hf_dataset,
        instance_prompt=args.instance_prompt,
        resolution=args.resolution,
        class_data_dir=args.class_data_dir,
        class_prompt=args.class_prompt,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # 只把 requires_grad 的参数（即 LoRA 权重）交给优化器
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
    # 冻结的模块不经过 accelerator.prepare，手动搬到目标设备并转成半精度以省显存
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    global_step = 0
    # 以"优化步数"而非 epoch 计数：角色参考图通常只有几张，一个 epoch 只有几步，按步数控制更直观
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # 图像 -> VAE 潜空间，并乘 scaling_factor 使潜变量方差接近 1（SDXL 训练约定）
                pixel_values = batch["instance_image"].to(accelerator.device, dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                # 标准 DDPM 训练目标：随机选时间步加噪，让 UNet 预测所加的噪声
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # batch_size 固定为 1，所以直接取 batch["instance_prompt"][0]
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    [tokenizer_one, tokenizer_two],
                    [text_encoder_one, text_encoder_two],
                    batch["instance_prompt"][0],
                    accelerator.device,
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

                # prior-preservation：对 class 图用同一批 timesteps 做同样的噪声预测，其损失按权重加到总损失上，
                # 约束模型在学新角色的同时不破坏对通用 "person" 概念的生成能力
                if args.with_prior_preservation and "class_image" in batch:
                    class_pixel_values = batch["class_image"].to(accelerator.device, dtype=weight_dtype)
                    class_latents = vae.encode(class_pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    class_noise = torch.randn_like(class_latents)
                    class_noisy_latents = noise_scheduler.add_noise(class_latents, class_noise, timesteps)
                    class_prompt_embeds, class_pooled_embeds = encode_prompt(
                        [tokenizer_one, tokenizer_two],
                        [text_encoder_one, text_encoder_two],
                        batch["class_prompt"][0],
                        accelerator.device,
                    )
                    class_added_cond_kwargs = {"text_embeds": class_pooled_embeds, "time_ids": add_time_ids}
                    class_pred = unet(class_noisy_latents, timesteps, class_prompt_embeds, added_cond_kwargs=class_added_cond_kwargs).sample
                    prior_loss = F.mse_loss(class_pred.float(), class_noise.float(), reduction="mean")
                    loss = loss + args.prior_loss_weight * prior_loss

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
        print(f"saved character LoRA to {args.output_dir}")


if __name__ == "__main__":
    main()
