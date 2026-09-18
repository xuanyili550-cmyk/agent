"""
 CV Course · Ch12 · 基础案例：分组公平性指标(总体准确率如何掩盖偏见,纯 numpy,可跑)
 偏见评估的关键:别只看总体准确率,要按敏感子群(如性别/肤色组)拆开算指标。
 这里从真标签+预测出发,真算三个业界常用公平指标:
   ① 准确率差(各组 acc 之差) ② 人口均等差(各组正类预测率之差,demographic parity)
   ③ 机会均等差(各组真正例率 TPR 之差,equal opportunity)。
 用一个"总体看着不错、B 组其实被坑"的数据验证它们能揪出偏见。跑：python3 本文件
"""
import numpy as np


def group_metrics(y_true, y_pred):
    """给单组算:准确率、正类预测率(PPR)、真正例率(TPR=召回)。全用混淆矩阵真算。"""
    acc = float((y_true == y_pred).mean())
    ppr = float(y_pred.mean())                                   # 被预测为正类的比例
    pos = y_true == 1
    tpr = float((y_pred[pos] == 1).mean()) if pos.any() else 0.0  # 真正例中被抓到的比例
    return acc, ppr, tpr


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    # 两组真实正类率相同(基础率一致),但模型对 B 组明显更差 → 制造真实的算法偏见
    yA = rng.integers(0, 2, n); yB = rng.integers(0, 2, n)
    # A 组:95% 预测正确
    pA = np.where(rng.random(n) < 0.95, yA, 1 - yA)
    # B 组:只有 65% 正确,且错误更多把正类漏判成负类(TPR 被压低)
    pB = np.where(rng.random(n) < 0.65, yB, 1 - yB)

    accA, pprA, tprA = group_metrics(yA, pA)
    accB, pprB, tprB = group_metrics(yB, pB)
    overall = float((np.concatenate([yA, yB]) == np.concatenate([pA, pB])).mean())

    acc_gap = abs(accA - accB)          # ① 准确率差
    dp_gap = abs(pprA - pprB)           # ② 人口均等差
    eo_gap = abs(tprA - tprB)           # ③ 机会均等差
    # 总体准确率被 A 组拉高,看着还行;但拆开后 B 组明显吃亏 → 指标必须揪出这个差距
    assert overall > 0.75
    assert acc_gap > 0.2 and eo_gap > 0.1

    print(f"A 组: acc={accA:.2f} 正类预测率={pprA:.2f} TPR={tprA:.2f}")
    print(f"B 组: acc={accB:.2f} 正类预测率={pprB:.2f} TPR={tprB:.2f}")
    print(f"总体 acc={overall:.2f}  ← 看着不差,却掩盖了子群差距")
    print(f"公平差距: 准确率差={acc_gap:.2f}  人口均等差={dp_gap:.2f}  机会均等差(TPR)={eo_gap:.2f}")
    print("✅ 必须按子群拆开评估:总体指标会掩盖偏见,分组公平指标才能揭示它")
    # 面试 Q&A：人口均等 vs 机会均等有何区别?——人口均等要求各组"正类预测率"相同(不看真值),
    #          机会均等只要求"真正例中的召回率(TPR)"相同;二者常无法同时满足(公平性不可能定理),需按场景取舍。
