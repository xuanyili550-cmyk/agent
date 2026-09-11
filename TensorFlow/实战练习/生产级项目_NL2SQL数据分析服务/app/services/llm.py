# 【构建 7/16 · 服务】依赖 config(1)+exceptions(3)+logging(2)
"""LLM 后端抽象：stub(测试) / mlx(本机 Mac) / openai(vLLM/Ollama/云端)，配置切换 + 重试。
生产切换只改 APP_LLM_BACKEND，业务代码不动。"""
import time
from abc import ABC, abstractmethod

from app.core.config import get_settings
from app.core.exceptions import LLMBackendError
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 120) -> str: ...


class StubLLM(LLM):
    """确定性桩(测试/CI 用)：根据问题关键词返回固定 SQL，无需下模型、秒回。"""
    def generate(self, prompt: str, max_tokens: int = 120) -> str:
        q = prompt
        if "城市" in q and ("金额" in q or "销售" in q):
            return "SELECT city, SUM(amount) AS total FROM orders GROUP BY city ORDER BY total DESC;"
        if "类目" in q:
            return "SELECT category, COUNT(*) AS n FROM products GROUP BY category;"
        if "最贵" in q or "最高" in q:
            return "SELECT name, price FROM products ORDER BY price DESC LIMIT 1;"
        if "回答" in q or "结果" in q:               # 总结阶段
            return "查询已完成，结果见数据。"
        return "SELECT SUM(amount) AS total FROM orders;"


class MlxLLM(LLM):
    """本机 Mac(Apple 芯片)：mlx-lm。"""
    def __init__(self, model: str):
        from mlx_lm import load
        self._model, self._tok = load(model)

    def generate(self, prompt: str, max_tokens: int = 120) -> str:
        from mlx_lm import generate
        text = self._tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
        return generate(self._model, self._tok, prompt=text, max_tokens=max_tokens, verbose=False).strip()


class OpenAILLM(LLM):
    """生产：OpenAI 兼容端点(vLLM / Ollama / 云端)，换后端只改 base_url。"""
    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=60, max_retries=2)
        self._model = model

    def generate(self, prompt: str, max_tokens: int = 120) -> str:
        r = self._client.chat.completions.create(
            model=self._model, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=max_tokens)
        return r.choices[0].message.content.strip()


_INSTANCE: LLM | None = None


def get_llm() -> LLM:
    """单例工厂：按配置构造后端(懒加载，只在第一次用时加载模型)。"""
    global _INSTANCE
    if _INSTANCE is None:
        s = get_settings()
        try:
            if s.llm_backend == "stub":
                _INSTANCE = StubLLM()
            elif s.llm_backend == "mlx":
                _INSTANCE = MlxLLM(s.llm_model)
            elif s.llm_backend == "openai":
                _INSTANCE = OpenAILLM(s.openai_base_url, s.openai_api_key, s.openai_model)
            else:
                raise LLMBackendError(f"未知 LLM 后端: {s.llm_backend}")
            logger.info("LLM 后端已初始化: %s", s.llm_backend)
        except LLMBackendError:
            raise
        except Exception as e:
            raise LLMBackendError(f"LLM 后端初始化失败: {e}") from e
    return _INSTANCE


def generate_with_retry(prompt: str, retries: int = 2, max_tokens: int = 120) -> str:
    """带重试的生成(上游抖动/超时时重试，指数退避)。"""
    last = None
    for i in range(retries + 1):
        try:
            return get_llm().generate(prompt, max_tokens=max_tokens)
        except Exception as e:                       # noqa: BLE001
            last = e
            logger.warning("LLM 生成失败(第%d次): %s", i + 1, e)
            time.sleep(0.2 * (2 ** i))
    raise LLMBackendError(f"LLM 生成重试 {retries} 次仍失败: {last}")
