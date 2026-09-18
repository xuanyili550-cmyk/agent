"""
 ML for 3D · Ch2 · 基础案例：绕轴旋转 + 透视投影渲染多视图(纯 numpy,机制真实,可跑)
 生成式 3D(如 Zero-1-to-3 / SDS)的地基是"可从任意相机看同一个 3D 物体":
 同一组 3D 点 → 绕 Y 轴转到不同视角 → 透视投影成 2D → 得到一组多视图图像。
 这里从零实现旋转矩阵 + 相机平移 + 针孔透视投影(除以深度),不写死坐标。
 跑：python3 本文件
"""
import numpy as np


def rot_y(a):
    """绕 Y 轴旋转矩阵(标准右手系):绕竖直轴转,x/z 混合,y 不变。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def project(pts, angle, cam_dist=3.0, focal=1.0):
    """把一组 3D 点从某视角透视投影到 2D:① 绕Y转 ② 相机后退 ③ 除以深度(近大远小)。"""
    rotated = pts @ rot_y(angle).T                       # ① 世界点旋转到该相机视角
    cam = rotated + np.array([0, 0, cam_dist])           # ② 相机沿 +z 后退,物体全落在正深度前方
    z = cam[:, 2:3]                                       # 深度(每个点到相机的距离分量)
    uv = focal * cam[:, :2] / z                           # ③ 透视投影:x/z, y/z —— 这一步带来近大远小
    return uv, z.ravel()


if __name__ == "__main__":
    # 一个不对称的小立方体顶点(不对称才能看出旋转真的改变了投影,而非巧合对称)
    pts = np.array([[-0.5, -0.5, -0.5], [0.6, -0.5, -0.5], [0.6, 0.7, -0.5],
                    [-0.5, 0.7, 0.8], [0.6, -0.5, 0.8], [-0.5, 0.7, -0.5]], float)

    views = {}
    for deg in (0, 45, 90, 180):
        uv, depth = project(pts, np.radians(deg))
        views[deg] = uv
        print(f"视角 {deg:3d}° 投影(前2个顶点 uv):{uv[:2].round(3).tolist()}  平均深度 {depth.mean():.3f}")

    # 机制自检 1:0° 与 360° 是同一视角 → 投影必须逐点相同
    uv0, _ = project(pts, np.radians(0))
    uv360, _ = project(pts, np.radians(360))
    assert np.allclose(uv0, uv360, atol=1e-9)

    # 机制自检 2:不同视角必须给出不同投影(否则说明旋转没生效)
    assert not np.allclose(views[0], views[90])

    # 机制自检 3:透视是"近大远小"——把整个物体推远,投影范围应收缩
    near, _ = project(pts, 0.0, cam_dist=3.0)
    far, _ = project(pts, 0.0, cam_dist=8.0)
    assert np.ptp(far, axis=0).max() < np.ptp(near, axis=0).max()
    print("✅ 旋转矩阵+透视投影生成多视图:同视角一致、异视角不同、越远投影越小——多视图一致性正是重建/生成 3D 的监督信号")
    # 面试 Q&A：为什么生成式 3D 常绕"多视图一致性"做文章?
    #   A：单张图缺深度、有歧义;逼模型在多个相机视角下都自洽,才能约束出真正的 3D 结构,
    #      这也是 NeRF/高斯泼溅/SDS 蒸馏的共同思路——用 2D 多视图监督反推 3D 表示。
