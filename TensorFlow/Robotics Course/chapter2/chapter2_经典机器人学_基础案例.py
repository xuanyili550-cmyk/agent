"""
 Robotics Course · Ch2 · 基础案例：二连杆机械臂正/逆运动学并互验(纯 numpy,机制真实,可跑)
 正运动学(FK):关节角 → 末端位置,几何唯一;逆运动学(IK):末端位置 → 关节角,可能多解/无解。
 这里用余弦定理从零解析 IK(不迭代不调库),再把解回代 FK 验证能回到原点——正逆互为逆运算。
 跑：python3 本文件
"""
import numpy as np

L1, L2 = 1.0, 1.0                                        # 两段连杆长度


def fk(t1, t2, l1=L1, l2=L2):
    """正运动学:末端 = 第一段 + 第二段的矢量和(第二段角度是 t1+t2,因为关节角是相对的)。"""
    x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
    y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
    return np.array([x, y])


def ik(x, y, l1=L1, l2=L2, elbow_up=True):
    """逆运动学(解析解):余弦定理先求肘关节 t2,再用几何关系求肩关节 t1。"""
    r2 = x * x + y * y                                   # 末端到原点距离的平方
    reach = (r2 > (l1 + l2) ** 2 + 1e-9) or (r2 < (l1 - l2) ** 2 - 1e-9)
    if reach:
        return None                                     # 超出工作空间(太远/太近)→ 无解
    cos_t2 = (r2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)    # 余弦定理:c²=a²+b²-2ab·cos
    cos_t2 = np.clip(cos_t2, -1.0, 1.0)                 # 防浮点越界
    t2 = np.arccos(cos_t2)                              # 肘朝一侧
    if not elbow_up:
        t2 = -t2                                        # 另一组解:肘朝另一侧(同一位置两种姿态)
    # 肩角 = 指向末端的方位角 - 第二段在肩处张开的夹角
    t1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(t2), l1 + l2 * np.cos(t2))
    return np.array([t1, t2])


if __name__ == "__main__":
    print("正运动学 FK:关节角 → 末端位置")
    for deg1, deg2 in [(0, 0), (90, 0), (45, 45)]:
        x, y = fk(np.radians(deg1), np.radians(deg2))
        print(f"  关节角({deg1:>3d}°,{deg2:>3d}°) → 末端 ({x:.3f}, {y:.3f})")

    print("逆运动学 IK:末端位置 → 关节角,并回代 FK 互验")
    for target in [(1.5, 0.5), (0.0, 1.0), (-0.8, 0.6)]:
        sol = ik(*target)
        back = fk(*sol)                                 # 把 IK 解回代 FK,应回到目标点
        print(f"  目标 {target} → 关节角 ({np.degrees(sol[0]):.1f}°,{np.degrees(sol[1]):.1f}°) → FK 回代 {back.round(3)}")
        assert np.allclose(back, target, atol=1e-9)     # 正逆互为逆运算 → 严格闭环

    assert ik(5.0, 0.0) is None                         # 目标超出臂长和 → 老实返回无解
    up, down = ik(1.0, 1.0, elbow_up=True), ik(1.0, 1.0, elbow_up=False)
    assert np.allclose(fk(*up), fk(*down))              # 肘上/肘下两解 → 同一末端位置(多解性)
    print("✅ FK 唯一、IK 可多解/无解;解析 IK 回代 FK 严格闭环验证,并复现'同一点两种肘姿'")
    # 面试 Q&A：为什么 IK 通常比 FK 难?
    #   A：FK 是关节角的确定函数(唯一解);IK 是反解,存在多解(肘上/肘下)、无解(超工作空间)、
    #      甚至奇异位形(雅可比降秩),所以工业上常用解析解+择优或数值迭代(牛顿/雅可比伪逆)。
