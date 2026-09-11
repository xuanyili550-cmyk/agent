import  numpy as np
from sympy import true

#numpy创建
list=[1,2,3]
arr=np.array(list,dtype=np.float64)
print(arr)
print("数组的纬度：",arr.ndim)

arr1=np.copy(arr)
print(arr1)
arr1[0]=8
print(arr1)
print("数组的纬度：",arr1.ndim)

#预定义形状
arr=np.zeros((2,3),dtype=int)
print(arr)
print(arr.dtype)

arr=np.zeros((2,),dtype=int)
print(arr)
print(arr.dtype)

#全1的
arr=np.ones((2,3),dtype=int)
print(arr)
print(arr.dtype)

#未初始化
arr=np.empty((2,3))
print(arr)
print(arr.dtype)

#全满
arr=np.full((2,5),2026)
print(arr)
print(arr.dtype)

#像
arr1=np.zeros_like(arr)
print(arr1)

arr1=np.empty_like(arr)
print(arr1)

arr1=np.ones_like(arr)
print(arr1)

arr1=np.full_like(arr,2024)
print(arr1)

#np.fill_diagonal() 的意思不是根据第一个角标，而是根据主对角线的位置进行替换
arr2 = np.full((5, 5), 111111)
np.fill_diagonal(arr2, 3, wrap=True)
print(arr2)
print(arr2.dtype)


#等差数列
arr=np.arange(1,10,1)
print(arr)

arr=np.arange(2,10,2)
print(arr)

#等间隔数列均分
app=np.linspace(0,10,5,dtype=int)
print(app)

app=np.arange(0,101,25)
print(app)

#对数间隔数列 第一个参数为第几个开始，第二个为base的几倍，
# num为展示几个并且同时返回以对数尺度均匀分布的数字。在线性空间中，
# 序列从 ``base ** start`` 开始`base` 的 `start` 次方），到 ``base ** stop`` 结束
arr=np.logspace(0,4,3,base=3)
print(arr)

#特殊矩阵
#单位矩阵：为对角线上的数字为1，其他为0
arr=np.eye(3,4,dtype=int)
print(arr)

#对角矩阵：主对角线非0的数字。把一维数组放到矩阵的主对角线上
arr=np.diag([1,2,3])
print(arr)

#实际上等价于：

arr = np.zeros((4,4))

arr[0][0] = 10
arr[1][1] = 20
arr[2][2] = 30
arr[3][3] = 40

ar=np.diag([10,20,30,40])
print(ar)


#随机数组生成
#生成0-1之间的浮点数，（均匀分布）
arr=np.random.rand(2,2)
print(arr)

#生成范围区间的随机浮点数   3-6之间的数据
arr=np.random.uniform(3,6,(2,3))
print(arr)

#生成范围区间的随机浮点数整数
arr=np.random.randint(3,6,(2,3))
print(arr)

#生成范围区间的随机浮点数正态分布   两边概率小，中间概率大。-3到3之间的数
arr=np.random.randn(2,3)
print(arr)


#设置随机种子 设置一样的种子
np.random.seed(20)
arr=np.random.randint(1,5,(2,5))
print(arr)

#从 NumPy 1.17 开始，更推荐使用 default_rng()：
# 每次初始化权重都一样。每次训练结果更容易复现。方便和别人比较实验结果。
rng = np.random.default_rng(42)

print(rng.integers(1, 100, 5))
