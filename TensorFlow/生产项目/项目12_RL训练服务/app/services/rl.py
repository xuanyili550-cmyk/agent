"""RL 训练/评估(Deep RL):stub(模拟递增奖励曲线,离线) | real(🔴 gym+SB3)。"""
from ..core.config import get_settings
from ..core.exceptions import ValidationError
def train(env: str, steps: int = 5):
    if not (env or "").strip(): raise ValidationError("env 不能为空")
    if get_settings().backend == "real":
        raise RuntimeError("real 后端需 gym + stable-baselines3;本机用 stub")
    curve = [round(1.0 - 1.0 / (i + 1), 3) for i in range(steps)]   # 奖励随训练递增
    return {"env": env, "steps": steps, "reward_curve": curve, "final_reward": curve[-1]}
def evaluate(env: str):
    return {"env": env, "avg_reward": 0.85, "episodes": 100}
