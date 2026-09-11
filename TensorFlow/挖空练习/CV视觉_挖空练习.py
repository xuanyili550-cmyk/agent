"""
================================================================================
 CV 视觉 挖空练习 · 卷积 / IoU / 量化（对应 Computer Vision Course）
================================================================================
 玩法：遮住每个「练习N」下面那行,凭记忆自己默写,再运行对照。
      运行：python3 CV视觉_挖空练习.py  (纯 numpy,无需联网)
      卡住→看文件底部「答案要点」。
================================================================================
"""
import numpy as np

# —— 练习1：手写卷积(卷积核滑窗,逐元素乘求和) ——
def conv2d(img, k):
    H, W = img.shape; kh, kw = k.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = (img[i:i+kh, j:j+kw] * k).sum()   # ← 练习1:逐元素乘再求和
    return out

# —— 练习2：IoU = 交集面积 / 并集面积 ——
def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)        # ← 练习2:交集(负数截0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua

# —— 练习3：int8 量化(线性映射到 0~255) ——
def quantize(x):
    lo, hi = x.min(), x.max(); scale = (hi - lo) / 255
    q = np.round((x - lo) / scale).astype(int)           # ← 练习3:(x-lo)/scale 再四舍五入
    return q, scale, lo


def main():
    img = np.zeros((5, 5)); img[:, 2:] = 1
    edge = conv2d(img, np.array([[-1, 0, 1]] * 3))
    assert edge.max() > 0                                # 检测到竖边
    assert abs(iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9   # 完全重合=1
    q, s, lo = quantize(np.array([-1.0, 0.0, 1.0]))
    assert q.min() == 0 and q.max() == 255
    print("✅ CV 挖空全部通过:卷积边缘检测 + IoU + int8 量化")


if __name__ == "__main__":
    main()

# ================================ 答案要点 ====================================
# 练习1: out[i,j] = (img[i:i+kh, j:j+kw] * k).sum()
# 练习2: inter = max(0, ix2-ix1) * max(0, iy2-iy1)
# 练习3: q = np.round((x - lo) / scale)   ; scale=(max-min)/255
# ============================================================================
