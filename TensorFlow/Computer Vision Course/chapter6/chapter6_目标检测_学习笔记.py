"""
================================================================================
 CV Course · Chapter 6 · 目标检测与分割（学习笔记 · IoU/NMS numpy 可跑）
================================================================================
 一句话：检测 = 分类 + 定位(框);评估靠 IoU/mAP;去重叠框靠 NMS。分割则到像素级。
 本章讲(纯 numpy 手写 IoU 和 NMS,无需模型)：
   ① IoU(交并比)：两框重叠面积/并集面积,衡量预测框和真值框有多贴。
   ② NMS(非极大值抑制)：同一物体多个框 → 按分数排序,抑制与高分框 IoU 过大的重复框。
   ③ mAP：不同 IoU 阈值下综合精确率/召回率;分割=逐像素分类。
 要点：IoU 是检测评估地基,NMS 是后处理去重;真实检测见 detect_real(🔴 DETR,需 transformers)。
 跑：python3 chapter6_目标检测_学习笔记.py   （IoU/NMS 纯 numpy 真跑)
================================================================================
"""
import numpy as np


def iou(a, b):
    """框 [x1,y1,x2,y2] 的交并比。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes, scores, thr=0.5):
    """非极大值抑制:留高分框,压掉与它 IoU>thr 的重复框。"""
    idx = list(np.argsort(scores)[::-1])
    keep = []
    while idx:
        i = idx.pop(0); keep.append(i)
        idx = [j for j in idx if iou(boxes[i], boxes[j]) <= thr]
    return keep


def detect_real(image_path):   # 🔴 需 transformers,默认不调用
    from transformers import pipeline
    return pipeline("object-detection", model="facebook/detr-resnet-50")


def main():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]])   # 前两个重叠
    scores = np.array([0.9, 0.8, 0.7])
    assert 0.6 < iou(boxes[0], boxes[1]) < 0.9         # 前两框高度重叠
    keep = nms(boxes, scores, thr=0.5)
    assert keep == [0, 2]                              # 压掉重复的 box1,留 0 和 2
    print(f"✅ Ch6 跑通：IoU(box0,box1)={iou(boxes[0],boxes[1]):.2f};NMS 保留 {keep}(压掉重复框)")
    # 面试：Q IoU 是什么? A 交/并,衡量框贴合度; Q NMS 干嘛? A 同物体多框去重,留最高分。


if __name__ == "__main__":
    main()
