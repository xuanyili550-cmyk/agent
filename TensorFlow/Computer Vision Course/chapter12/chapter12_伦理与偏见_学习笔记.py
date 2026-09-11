"""
================================================================================
 CV Course · Chapter 12 · CV 伦理与偏见（学习笔记 · 偏见指标 numpy 可跑）
================================================================================
 一句话：视觉模型会从数据学到社会偏见;负责任的 AI 要能"量化发现"再缓解,不能只靠总体准确率。
 本章讲(纯 numpy 算"分组公平性指标",看总体准确率如何掩盖子群差异)：
   ① 总体准确率会掩盖子群差异:对 A 组准、对 B 组差,平均看还行。
   ② 分组准确率差(accuracy gap)、人口平等差(demographic parity)等公平指标。
   ③ 真实案例(ImageNet Roulette 等) + 缓解:平衡数据/重加权/后处理。
 要点：必须按子群拆开看指标;总体高 ≠ 公平。上线前做偏见评估是伦理也是合规。
 跑：python3 chapter12_伦理与偏见_学习笔记.py   （偏见指标纯 numpy 真跑)
================================================================================
"""
import numpy as np


def group_accuracy(y_true, y_pred, group):
    """按子群算准确率,返回 {组: acc} 和 总体 acc。"""
    accs = {}
    for g in np.unique(group):
        m = group == g
        accs[str(g)] = float((y_true[m] == y_pred[m]).mean())
    return accs, float((y_true == y_pred).mean())


def main():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    group = np.array([0] * 100 + [1] * 100)      # 两个子群
    pred = y.copy()
    # 模拟:对 group1 故意错一部分(偏见)
    flip = (group == 1) & (rng.random(200) < 0.4)
    pred[flip] = 1 - pred[flip]
    accs, overall = group_accuracy(y, pred, group)
    gap = abs(accs["0"] - accs["1"])
    assert accs["0"] > accs["1"] and gap > 0.2   # 子群差距明显,但总体还不算太低
    print(f"✅ Ch12 跑通：总体acc={overall:.2f} 掩盖了子群差异 组0={accs['0']:.2f}/组1={accs['1']:.2f}(gap={gap:.2f})")
    # 面试：Q 为什么不能只看总体准确率? A 会掩盖子群偏见; Q 怎么缓解? A 平衡数据/重加权/阈值后处理/公平约束。


if __name__ == "__main__":
    main()
