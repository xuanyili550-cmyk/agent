"""
 CV Course · Ch8 · 基础案例：透视投影"近大远小"(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np

pts = np.array([[1, 1, 1], [1, 1, 5]], float)     # 同 x,y,深度 z 不同
proj = pts[:, :2] / pts[:, 2:3]                    # 除以 z
print("z=1 投影:", proj[0], " z=5 投影:", proj[1])
print("✅ 除以深度 z → 远处物体投影更靠中心、更小(透视=近大远小)")
