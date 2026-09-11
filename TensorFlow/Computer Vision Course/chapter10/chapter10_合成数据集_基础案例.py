"""
 CV Course · Ch10 · 基础案例：程序化生成带标签的形状图(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np

size = 16
yy, xx = np.mgrid[0:size, 0:size]
circle = ((yy-8)**2 + (xx-8)**2 <= 25).astype(int)     # 圆,标签0
square = ((np.abs(yy-8)<=4) & (np.abs(xx-8)<=4)).astype(int)  # 方,标签1
print("圆图前景像素:", circle.sum(), "标签0")
print("方图前景像素:", square.sum(), "标签1")
print("✅ 程序化造数据:图像+标签(+掩码)全自动,合成数据的核心优势")
