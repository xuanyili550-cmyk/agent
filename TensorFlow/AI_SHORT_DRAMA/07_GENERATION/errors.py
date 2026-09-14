class NotConfiguredError(RuntimeError):
    """生成 provider 缺少必需的环境变量凭证。"""


class TransientProviderError(RuntimeError):
    """网络抖动 / 429 / 5xx / 超时：同样的参数再试一次大概率能成功。

    这是三级重试阶梯的第一级触发条件；由 Celery 任务层按指数退避重新投递。
    """


class GenerationRejectedError(RuntimeError):
    """provider 明确拒绝了这条请求（安全过滤、prompt 违规、参数非法）：同参数重试没意义，
    应该进入第二级（改写提示词）。"""


class ResourceExhaustedError(RuntimeError):
    """显存不足 / 分辨率超过 provider 上限：应该进入第三级（降分辨率）。"""


# 同一个模块可能以两种名字被 import（07 内部 `from errors import ...` 走 sys.path，13_INFRA 走
# importlib "07_GENERATION.errors"）。不做别名就会出现两份类对象，except 捕获不到对方抛的异常。
import sys as _sys

for _name in ("errors", "07_GENERATION.errors"):
    _sys.modules.setdefault(_name, _sys.modules[__name__])
