"""机器人策略(Robotics):stub(确定性策略,离线) | real(🔴 LeRobot 策略模型)。"""
from ..core.config import get_settings
from ..core.exceptions import ValidationError
def act(observation):
    if not observation: raise ValidationError("observation 不能为空")
    if get_settings().backend == "real":
        raise RuntimeError("real 后端需 lerobot 策略权重;本机用 stub")
    # stub 策略:按观测和的符号决定方向,均值决定力度(确定性,可测)
    s = sum(observation); mean = s / len(observation)
    action = [1.0 if s > 0 else -1.0, round(mean, 3)]
    return {"action": action, "dim": len(action), "policy": "stub(规则)"}
