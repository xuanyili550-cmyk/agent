"""
 ML for 3D · Ch2 · 基础案例：3D 点绕轴旋转+投影得多视图(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

pts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
for deg in (0, 45, 90):
    v = (pts @ rot_y(np.radians(deg)).T)[:, :2]
    print(f"视角 {deg:3d}° 投影 xy:\n{v.round(2)}")
print("✅ 同一物体不同视角 → 不同 2D 投影;多视图一致是重建 3D 的基础")
