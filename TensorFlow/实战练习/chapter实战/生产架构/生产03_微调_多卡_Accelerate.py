"""
================================================================================
 生产03 · 云端多卡微调：Accelerate / DeepSpeed + LoRA + 推模型仓库
================================================================================
 ⚠ 需要多卡 GPU 云环境，本机跑不了；这是“真上线怎么搭”的架构参考。
   本文件的 train() 是真实可用的 Python(不是字符串)：数据/模型/import 都惰性化，所以
   `python3 生产03_微调_多卡_Accelerate.py` 只打印架构说明，不会真去下 7B 模型或连 S3。
   把 train() 放到多卡机上用 `accelerate launch` 拉起就能真训。
   本地能跑的小版本见 ../案例3_LoRA指令微调.py。

 为什么生产微调这么搭：
   · 为什么多卡 + DeepSpeed：单卡放不下大模型的权重+梯度+优化器状态;DeepSpeed ZeRO
     把这三者切到多卡(ZeRO-1/2/3 逐级切),让 7B/70B 也能训;还能 offload 到 CPU/NVMe。
   · 为什么 LoRA/QLoRA：只训 <1% 参数,显存和存储都省;QLoRA 再把基座 4-bit 量化,
     单卡也能微调很大的模型(见 ../../Chapter 11)。
   · 为什么 Accelerate：一套代码,单卡/多卡/多机/混合精度透明切换,不用改训练逻辑。
   · 为什么推模型仓库：训练和推理分离(生产铁律)——训完把权重/适配器推到 HF Hub 或
     内部模型仓库(MLflow/S3/OSS),推理服务(生产01)从仓库拉取加载。

 —— 一、Accelerate 多卡配置(accelerate config 生成，或写 yaml) ——
   # accelerate_config.yaml
   compute_environment: LOCAL_MACHINE
   distributed_type: DEEPSPEED
   num_processes: 8              # 8 张卡
   mixed_precision: bf16
   deepspeed_config:
     zero_stage: 2               # ZeRO-2：切分优化器状态+梯度(7B 常用);大模型用 zero_stage:3
     offload_optimizer_device: none   # 显存紧就 cpu
     gradient_accumulation_steps: 4

 —— 二、启动(多卡/多机) ——
   accelerate launch --config_file accelerate_config.yaml 生产03_微调_多卡_Accelerate.py
   # 多机：每台机器设 --machine_rank / --main_process_ip；或用 SkyPilot/Ray 拉起集群。
================================================================================
"""
import os

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
DATA_URI = os.getenv("DATA_URI", "s3://your-bucket/sft_data.parquet")  # 或 OSS/HDFS
HUB_MODEL_ID = os.getenv("HUB_MODEL_ID", "your-org/qwen-sft-lora")


def load_dataset_from_lake(uri):
    """生产从数据湖/标注平台来；这里示意从 parquet 读，格式化成 messages(chat 格式)。"""
    import pandas as pd                          # 惰性导入
    from datasets import Dataset
    df = pd.read_parquet(uri)                    # 需有 "messages" 列(chat 格式)或 "text" 列
    return Dataset.from_pandas(df)


def train():
    """用 trl SFTTrainer + LoRA + DeepSpeed(由 accelerate launch 启动，透明多卡)。"""
    from peft import LoraConfig                  # 惰性导入(装了也放函数里，避免 import 即触发)
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset_from_lake(DATA_URI)

    lora = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                      lora_dropout=0.05, task_type="CAUSAL_LM")

    args = SFTConfig(
        output_dir="/mnt/ckpt/qwen-sft",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,        # 有效 batch = 4*4*8卡 = 128
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,                            # 现代卡用 bf16(比 fp16 稳)
        gradient_checkpointing=True,          # 用时间换显存,能训更长序列
        logging_steps=10, save_steps=500,
        report_to="wandb",                    # 实验追踪(国产可用 SwanLab)
        push_to_hub=True, hub_model_id=HUB_MODEL_ID,   # 训完推模型仓库
    )
    # SFTTrainer 可直接传模型名字符串，内部会加载；多卡由 accelerate/deepspeed 接管
    trainer = SFTTrainer(model=MODEL_ID, args=args, train_dataset=ds, peft_config=lora)
    trainer.train()
    trainer.push_to_hub()                     # → 推理服务(生产01)从这里拉取
    return trainer


NOTES = """
 生产要点：
  · 数据：从数据湖(S3/OSS/HDFS)读,别塞本地;大数据用流式/分片。
  · 检查点：定期 save + 断点续训(spot/抢占实例会被回收);存到共享存储(NFS/S3)。
  · 显存不够的阶梯：梯度累积 → 梯度检查点 → LoRA → QLoRA(4bit) → DeepSpeed ZeRO-3 + offload。
  · 混合精度：A100/H100 用 bf16;老卡(V100)用 fp16 + GradScaler。
  · 评估：训练中定期在验证集/基准上评,盯过拟合;别只看 train loss。
  · 产出物：LoRA 适配器(几十 MB) 或 合并后的全权重,推到 HF Hub / MLflow / 内部仓库,打好版本 tag。
  · 平台：美国 SageMaker/Ray/SkyPilot;中国 阿里 PAI / 华为 ModelArts / 火山方舟 / 自建 K8s+Volcano。
"""

if __name__ == "__main__":
    # 由 accelerate launch 拉起时才真训；直接 python3 运行只打印说明(避免误下 7B/连 S3)
    import sys
    if "--train" in sys.argv:
        train()
    else:
        print(__doc__)
        print("=== 生产要点 ===", NOTES)
        print(">>> 架构参考。真跑：多卡机上 `accelerate launch --config_file accelerate_config.yaml "
              "生产03_微调_多卡_Accelerate.py --train`。本地小版本见 ../案例3_LoRA指令微调.py")
