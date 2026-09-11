"""
 Robotics Course · Ch2 · 基础案例：二连杆机械臂正运动学(纯 numpy,可跑)
 关节角 → 末端位置。跑：python3 本文件
"""
import numpy as np

def fk(t1, t2, l1=1.0, l2=1.0):
    return (l1*np.cos(t1)+l2*np.cos(t1+t2), l1*np.sin(t1)+l2*np.sin(t1+t2))

for deg1, deg2 in [(0, 0), (90, 0), (45, 45)]:
    x, y = fk(np.radians(deg1), np.radians(deg2))
    print(f"关节角({deg1}°,{deg2}°) → 末端位置 ({x:.2f}, {y:.2f})")
print("✅ 正运动学:给定关节角,几何唯一确定末端位置")
