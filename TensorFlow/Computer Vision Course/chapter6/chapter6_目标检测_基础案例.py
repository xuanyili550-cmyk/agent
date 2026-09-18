"""
 CV Course · Ch6 · 基础案例：IoU + 非极大值抑制 NMS(纯 numpy,机制真实,可跑)
 检测器会对同一物体吐出一堆高度重叠的框;NMS 去重:按置信度排序,留最高分,再删掉与它 IoU 超阈值的框,循环。
 IoU=交集/并集,是"两个框有多重合"的度量,既用于评估也用于 NMS。这里纯 numpy 真算 IoU 与完整 NMS。
 跑：python3 本文件
"""
import numpy as np


def iou(a, b):
    """交并比:交集面积 / 并集面积。框格式 [x1,y1,x2,y2]。不相交时交集边长取 0(max(0,·))。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, thresh=0.5):
    """非极大值抑制:① 按分数降序 → ② 取当前最高分框保留 → ③ 删掉与它 IoU>阈值的框 → ④ 对剩下的重复。"""
    order = list(np.argsort(scores)[::-1])                 # 分数从高到低的下标
    keep = []
    while order:
        i = order.pop(0)                                    # 当前最高分,一定保留
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) <= thresh]  # 抑制与它太重叠的框
    return keep


if __name__ == "__main__":
    # IoU 三种情形
    print("完全重合 IoU:", iou([0,0,10,10], [0,0,10,10]))
    print("部分重叠 IoU:", round(iou([0,0,10,10], [5,5,15,15]), 3))
    print("不相交   IoU:", iou([0,0,10,10], [20,20,30,30]))
    assert iou([0,0,10,10], [0,0,10,10]) == 1.0
    assert iou([0,0,10,10], [20,20,30,30]) == 0.0

    # NMS:前 3 个框几乎重叠(同一物体的重复预测),第 4 个是另一物体
    boxes = np.array([[10,10,50,50], [12,11,52,51], [11,12,49,52], [100,100,140,140]])
    scores = np.array([0.9, 0.85, 0.8, 0.7])
    keep = nms(boxes, scores, thresh=0.5)
    print("NMS 保留框下标:", keep, "(应只留 0 和 3:每个物体一个框)")
    assert set(keep) == {0, 3}                              # 重复框被抑制,两个物体各留最高分
    assert keep[0] == 0                                     # 最高分框总被保留
    print("✅ IoU 量化重叠 + NMS 按分数去重 → 每个物体只留一个最佳框(检测后处理标配)")
    # 面试Q:NMS 有什么缺陷,Soft-NMS 怎么改进?
    #      A:硬 NMS 直接删重叠框,两个真物体挨太近时会误删一个;Soft-NMS 不删而是按 IoU 衰减其分数,保留召回同时压制冗余。
