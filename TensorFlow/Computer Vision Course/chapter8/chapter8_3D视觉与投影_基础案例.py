"""
 CV Course · Ch8 · 基础案例：小孔相机完整投影管线(内参 K + 外参[R|t],纯 numpy,可跑)
 3D 视觉核心 = 把世界坐标点投到像素:先用外参[R|t]把「世界系」转到「相机系」,
 再用内参 K(焦距 fx,fy + 主点 cx,cy)把相机系投到像素平面,最后除以深度 z(透视除法)。
 这里不写死结果:构造真旋转矩阵当外参、真 K 当内参,验证「逐步算」== 「投影矩阵 P 一步算」,
 并验证"近大远小"确实来自那个除以 z。跑：python3 本文件
"""
import numpy as np


def rot_y(theta):
    """绕 y 轴的真旋转矩阵(相机朝向变化);行列式=1、正交 → 是合法旋转。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def make_camera():
    """构造一台真相机:内参 K(fx=fy=800,主点在图像中心 320,240) + 外参[R|t]。"""
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    R = rot_y(np.deg2rad(15))            # 相机绕 y 轴转 15°(外参的旋转部分)
    C = np.array([0.5, 0.0, -2.0])       # 相机在世界系的位置
    t = -R @ C                           # 平移 t = -R·C(把世界原点搬到相机系)
    return K, R, t


def project_stepwise(K, R, t, Xw):
    """逐步投影:① 世界系→相机系 x_c=R·X+t;② 内参投影 K·x_c;③ 除以深度 z。"""
    x_cam = (R @ Xw.T).T + t             # (N,3) 相机坐标系下的点
    homog = (K @ x_cam.T).T             # (N,3) 未归一化的像素齐次坐标
    return homog[:, :2] / homog[:, 2:3], x_cam[:, 2]   # 返回像素 + 深度 z


def project_matrix(K, R, t, Xw):
    """一步投影:整条管线其实是一个 3x4 投影矩阵 P=K·[R|t] 作用在齐次世界坐标上。"""
    P = K @ np.hstack([R, t[:, None]])   # (3,4) 完整投影矩阵
    Xh = np.hstack([Xw, np.ones((len(Xw), 1))])        # 齐次世界坐标 (N,4)
    homog = (P @ Xh.T).T
    return homog[:, :2] / homog[:, 2:3]


if __name__ == "__main__":
    # ① 用一台带真旋转+平移的相机,验证「逐步算」严格等于「投影矩阵 P 一步算」
    K, R, t = make_camera()
    Xw = np.array([[0.0, 0.0, 3.0], [1.0, -0.5, 5.0], [-0.8, 0.4, 6.0]])
    px_step, _ = project_stepwise(K, R, t, Xw)
    px_mat = project_matrix(K, R, t, Xw)
    assert np.allclose(px_step, px_mat)                  # 逐步 == 投影矩阵一步,机制自洽

    # ② 用一台正对场景的简单相机(R=I,原点),两点在相机系同 x,y 只是深度不同 → 验证近大远小
    R0, t0 = np.eye(3), np.zeros(3)
    on_ray = np.array([[1.0, 0.5, 3.0], [1.0, 0.5, 8.0]])  # 同 x,y,深度 z 不同
    px_ray, depth = project_stepwise(K, R0, t0, on_ray)
    principal = np.array([320.0, 240.0])
    d_near = np.linalg.norm(px_ray[0] - principal)
    d_far = np.linalg.norm(px_ray[1] - principal)
    assert depth[1] > depth[0] and d_far < d_near        # 深度大 → 像素更靠主点(远物变小/居中)

    print("带旋转相机投影像素(逐步):", np.round(px_step, 1).tolist())
    print(f"近点 z={depth[0]} 距主点 {d_near:.1f}px  远点 z={depth[1]} 距主点 {d_far:.1f}px")
    print("✅ 完整管线 P=K·[R|t]+透视除法跑通;逐步==矩阵一步,且除以 z 天然产生近大远小")
    # 面试 Q&A：为什么必须除以 z(透视除法)?——因为小孔成像是非线性投影,3D→2D 丢了深度,
    #          唯有按各点自身深度 z 归一化,才能让远处物体在像平面上收缩变小(而正交投影不除 z)。
