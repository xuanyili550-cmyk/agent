"""图像生成(Diffusion):stub(确定性占位,离线) | real(🔴 StableDiffusion,需 GPU)。"""
import hashlib
from ..core.config import get_settings
from ..core.exceptions import ValidationError
def generate(prompt: str, steps: int = 25, guidance: float = 7.5):
    if not (prompt or "").strip(): raise ValidationError("prompt 不能为空")
    if get_settings().backend == "real":
        raise RuntimeError("real 后端需 diffusers + GPU;本机用 stub")
    img_id = hashlib.md5(f"{prompt}{steps}{guidance}".encode()).hexdigest()[:12]
    return {"image_id": img_id, "width": 512, "height": 512, "steps": steps,
            "guidance": guidance, "note": "stub 占位;real 后端出真实图(需 GPU)"}
