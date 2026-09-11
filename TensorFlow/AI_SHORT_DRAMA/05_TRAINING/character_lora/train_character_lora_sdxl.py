"""
Character LoRA training for SDXL, adapted from diffusers' dreambooth_lora_sdxl
example. Reads a character_manifest.jsonl row set (see 04_DATASET/metadata)
instead of a plain image folder, and trains a LoRA on the UNet (+ optional
text encoders) for one character_id at a time.

Optional prior-preservation (class images) mirrors the DreamBooth recipe: it
avoids the model collapsing the whole "person" concept onto the new token.
Class images must be supplied by the user beforehand (this script never
generates or downloads them).

Example:
  python train_character_lora_sdxl.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --manifest_path ../../04_DATASET/metadata/character_manifest.jsonl \
    --character_id char_lin_wei \
    --instance_prompt "a photo of sks_lin_wei person" \
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
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, CLIPTextModelWithProjection, PretrainedConfig

DATASET_DIR = Path(__file__).resolve().parents[2] / "04_DATASET"
sys.path.insert(0, str(DATASET_DIR))
from dataset_loaders import build_character_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
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

    parser.add_argument("--with_prior_preservation", action="store_true")
    parser.add_argument("--class_data_dir", default=None, help="folder of pre-existing class images (user-supplied)")
    parser.add_argument("--class_prompt", default=None)
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)

    return parser.parse_args()


class CharacterLoraDataset(Dataset):
    """Wraps the character_manifest HF dataset, filtered to one character_id,
    optionally interleaved with prior-preservation class images."""

    def __init__(self, hf_dataset, instance_prompt: str, resolution: int, class_data_dir=None, class_prompt=None):
        self.instance_examples = [ex for ex in hf_dataset]
        self.instance_prompt = instance_prompt
        self.resolution = resolution
        self.class_image_paths = sorted(Path(class_data_dir).glob("*")) if class_data_dir else []
        self.class_prompt = class_prompt

    def __len__(self) -> int:
        return max(len(self.instance_examples), len(self.class_image_paths)) if self.class_image_paths else len(
            self.instance_examples
        )

    def _prepare_image(self, pil_image):
        pil_image = pil_image.convert("RGB").resize((self.resolution, self.resolution))
        tensor = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(pil_image.tobytes())))
            .reshape(self.resolution, self.resolution, 3)
            .numpy()
        ).permute(2, 0, 1).float() / 127.5 - 1.0
        return tensor

    def __getitem__(self, index):
        instance_example = self.instance_examples[index % len(self.instance_examples)]
        item = {
            "instance_image": self._prepare_image(instance_example["image"]),
            "instance_prompt": self.instance_prompt,
        }
        if self.class_image_paths:
            from PIL import Image as PILImage

            class_path = self.class_image_paths[index % len(self.class_image_paths)]
            item["class_image"] = self._prepare_image(PILImage.open(class_path))
            item["class_prompt"] = self.class_prompt
        return item


def load_text_encoder_class(pretrained_model_name_or_path: str, subfolder: str):
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
    architecture = text_encoder_config.architectures[0]
    if architecture == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    if architecture == "CLIPTextModelWithProjection":
        return CLIPTextModelWithProjection
    raise ValueError(f"unsupported text encoder architecture: {architecture}")


def encode_prompt(tokenizers, text_encoders, prompt: str, device):
    prompt_embeds_list = []
    pooled_prompt_embeds = None
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        output = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = output[0]
        prompt_embeds_list.append(output.hidden_states[-2])
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False
    )

    text_encoder_cls_one = load_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder")
    text_encoder_cls_two = load_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder_2")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

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

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    global_step = 0
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["instance_image"].to(accelerator.device, dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    [tokenizer_one, tokenizer_two],
                    [text_encoder_one, text_encoder_two],
                    batch["instance_prompt"][0],
                    accelerator.device,
                )

                add_time_ids = torch.tensor(
                    [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

                model_pred = unet(
                    noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs
                ).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

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
                    class_pred = unet(
                        class_noisy_latents, timesteps, class_prompt_embeds, added_cond_kwargs=class_added_cond_kwargs
                    ).sample
                    prior_loss = F.mse_loss(class_pred.float(), class_noise.float(), reduction="mean")
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % 10 == 0:
                    print(f"step {global_step}/{args.max_train_steps} loss={loss.item():.4f}")
            if global_step >= args.max_train_steps:
                break

    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(args.output_dir)
        print(f"saved character LoRA to {args.output_dir}")


if __name__ == "__main__":
    main()
