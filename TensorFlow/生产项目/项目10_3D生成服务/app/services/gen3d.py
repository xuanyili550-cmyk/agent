"""3D 生成(ML for 3D):stub(返回高斯点云参数规模,离线) | real(🔴 LGM,需 GPU)。"""
from ..core.config import get_settings
from ..core.exceptions import ValidationError
def generate3d(prompt: str):
    if not (prompt or "").strip(): raise ValidationError("prompt 不能为空")
    if get_settings().backend == "real":
        raise RuntimeError("real 后端需 LGM + CUDA;本机用 stub")
    n = 2048
    return {"num_gaussians": n, "params_per_gaussian": 14, "format": "ply",
            "pipeline": "多视图扩散 → 高斯泼溅(LGM)", "note": "stub;real 出真实 3D(需 GPU)"}
