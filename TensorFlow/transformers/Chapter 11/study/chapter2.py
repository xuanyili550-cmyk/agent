import argparse
import os
from dataclasses import dataclass, field
@dataclass
class Config:
    model_name: str = "HuggingFaceTB/SmolLM2-135M"        # 真实小 LLM(1.35 亿参数)
    dataset_name: str = "databricks/databricks-dolly-15k"  # 真实指令微调数据集
    n_train: int = 300                 # 训练子集(快速验证；调大更好)
    max_steps: int = 40                # 训练步数(Mac 上先小步跑通)
    batch_size: int = 2
    grad_accum: int = 4                # 有效 batch = 2*4 = 8
    lr: float = 2e-4                   # LoRA 通常比全量微调用更大的 lr
    max_length: int = 256
    # --- LoRA 超参 ---
    lora_r: int = 8                    # 低秩矩阵的秩：越大容量越大、可训参数越多(常用 8/16/32)
    lora_alpha: int = 16               # 缩放系数：一般设成 2*r
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])  # 给哪些层加 LoRA
    output_dir: str = "smollm2-dolly-lora"


PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
def build_dataset(cfg):
    from datasets import load_dataset
    ds = load_dataset(cfg.dataset_name, split="train").shuffle(seed=42)
    ds=ds.select(range(min(cfg.n_train,len(ds))))
    def to_text(ex):
        return {"text": PROMPT.format(instruction=ex["instruction"],
                                      response=ex["response"])}
    return ds.map(to_text,remove_columns=ds.column_names)
def train(cfg: Config):
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    train_ds=build_dataset(cfg)
    lora_config = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules, task_type="CAUSAL_LM", bias="none")
    args = SFTConfig(
        output_dir=cfg.output_dir, max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr, warmup_steps=max(1, cfg.max_steps // 10),
        logging_steps=10, max_length=cfg.max_length,
        dataset_text_field="text", report_to=[],
    )
    trainer=SFTTrainer(model=cfg.model_name,args=args,train_dataset=train_ds,peft_config=lora_config)
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(cfg.output_dir)
    adapter_mb = sum(os.path.getsize(os.path.join(cfg.output_dir, f))
                     for f in os.listdir(cfg.output_dir)
                     if f.endswith((".safetensors", ".bin"))) / 1e6

def inference_and_merge(cfg: Config):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    if not os.path.isdir(cfg.output_dir):
        print("(还没训练出适配器，跳过)"); return
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    lora_model = PeftModel.from_pretrained(base, cfg.output_dir).eval()
    prompt = PROMPT.format(instruction="Name three colors.", response="")
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = lora_model.generate(**inputs, max_new_tokens=30, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)

    merged = lora_model.merge_and_unload()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--n", type=int, default=300)
    args = p.parse_args()
    cfg = Config(max_steps=args.steps, n_train=args.n)
    train(cfg)
    inference_and_merge(cfg)
