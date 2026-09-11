"""对齐流水线编排:数据校验 → 训练(SFT) → 评估 → 注册模型版本。"""
from . import dataset, evaluator, registry, trainer
def run(samples, progress_cb=None) -> dict:
    n = dataset.validate(samples)                    # ① 校验
    train_metrics = trainer.train(samples, progress_cb)   # ② 训练(stub/trl)
    eval_metrics = evaluator.evaluate(samples)       # ③ 评估
    entry = registry.register({**train_metrics, **eval_metrics})   # ④ 注册
    return {"trained_on": n, **entry}
