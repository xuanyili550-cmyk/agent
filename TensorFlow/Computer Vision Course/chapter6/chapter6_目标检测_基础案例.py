"""
 CV Course · Ch6 · 基础案例：手写 IoU 交并比(纯 numpy,可跑)
 跑：python3 本文件
"""
def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua

print("完全重合:", iou([0,0,10,10], [0,0,10,10]))     # 1.0
print("部分重叠:", round(iou([0,0,10,10], [5,5,15,15]), 3))
print("不相交  :", iou([0,0,10,10], [20,20,30,30]))    # 0.0
print("✅ IoU=交集/并集,是目标检测评估与 NMS 的基础")
