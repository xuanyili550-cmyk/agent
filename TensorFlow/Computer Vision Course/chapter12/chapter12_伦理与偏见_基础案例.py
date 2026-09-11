"""
 CV Course · Ch12 · 基础案例：分组准确率差(偏见指标,纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np

# 两组各 50 个样本:组A 全对、组B 只对 60%
acc_A, acc_B = 1.0, 0.6
overall = (acc_A * 50 + acc_B * 50) / 100
print(f"组A acc={acc_A}, 组B acc={acc_B} → 总体={overall}")
print(f"准确率差 gap={abs(acc_A-acc_B):.2f}  ← 总体 {overall} 看着还行,却掩盖了对 B 组的偏见")
print("✅ 公平评估要按子群拆开看,不能只看总体")
