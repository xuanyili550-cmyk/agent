"""训练后端:stub(离线模拟,返回递减 loss 曲线验证流程) | trl(🔴 真实 SFT+LoRA,需 GPU)。"""
from ..core.config import get_settings
def train(samples, progress_cb=None):
    s = get_settings()
    if s.train_backend == "trl":
        return _train_trl(samples, progress_cb)      # 🔴 真实训练
    # stub:模拟每个 epoch loss 递减(验证"数据→训练→指标"流程,不需 GPU)
    losses = []
    for e in range(s.epochs):
        loss = round(2.0 / (e + 1), 3)
        losses.append(loss)
        if progress_cb:
            progress_cb(e + 1, s.epochs)
    return {"epochs": s.epochs, "loss_curve": losses, "final_loss": losses[-1], "samples": len(samples)}
def _train_trl(samples, progress_cb):   # 🔴 需 trl/peft/transformers + GPU
    # from trl import SFTTrainer ...  apply_chat_template → LoRA → SFTTrainer.train()
    raise RuntimeError("trl 后端需 GPU 环境;本机用 stub 验证流程")
