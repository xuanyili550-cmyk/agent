import  numpy as np
#索引和切片

#一维    随机数
arr=np.random.randint(1,100,20)
print(arr)
print(arr[2:7])
print(arr[arr>=50])

#numpy运算
arr = np.array([1,2,3])
ar=np.array([4,5,6])
print(arr-ar)
print(arr+ar)
print(arr*ar)
print(arr/ar)

a=[1,2,3]
b=[4,5,6]
print(a+b)

#基本数学函数
#计算平方根
print(np.sqrt(9))
print(np.sqrt([1,4,9]))
arr=np.array([1,25,81])
print(np.sqrt(arr))

#计算指数 计算 e^x  =y
print(np.exp(0))
print(np.exp(1))


#计算自然对数 计算lnx lny=x
print(np.log(2))
print(np.log(3))

#正炫值，余炫值
print(np.sin(np.pi))
print(np.cos(np.pi))

#计算绝对值
print(np.abs(np.array([-1,2,-3,4])))

#计算a的b次幂
print(np.power(3,3))

#四舍五入
print(np.round([3.2,4,5.6,6.1,8.9]))

#向上取整，向下取整
arr=np.array([3.2,4,5.6,6.1,8.9])
print(np.ceil(arr))
print(np.floor(arr))

#检测损失值NAN
print(np.isnan([1,2,3]))
print(np.isnan([1,np.nan,3]))

#统计函数

#求和，平均值，计算中位数，

#. 求和 np.sum()
arr = np.array([1, 2, 3, 4, 5])
print(np.sum(arr))

#   平均值 np.mean()
print(np.mean(arr))

#中位数 np.median()中间位置的数
print(np.median(arr))
#最大值 np.max()
print(np.max(arr))
#最小值 np.min()
print(np.min(arr))
#标准差 np.std()数据的离散程度。
print(np.std(arr))
#方差 np.var()方差=标准差2。np.var(arr) == np.std(arr)**2
print(np.var(arr))
#累计求和 np.cumsum()
print(np.cumsum(arr))
#百分位数 np.percentile()
print(np.percentile(arr,50))
print(np.percentile(arr,25) ) # 第一四分位
print(np.percentile(arr,50) ) # 中位数
print(np.percentile(arr,75) ) # 第三四分位


#二维数组统计
arr = np.array([
    [1,2,3],
    [4,5,6]
])

#全部求和：
print(np.sum(arr))
#按行统计
print(np.sum(arr,axis=1))
#按列统计
print(np.sum(arr,axis=0))

#AI 数据处理中最常用组合
scores = np.array([0.8,0.9,0.75,0.95])
print(np.mean(scores))    # 平均准确率
print(np.max(scores))     # 最好结果
print(np.min(scores))     # 最差结果
print(np.std(scores))     # 稳定性

#比较函数
#大于
print(np.greater([2,4.,6,8,3],4))
#小于
print(np.less([2,4,6,8,3],4))
#是否等于
print(np.equal([2,4,6,8,3],4))
print(np.equal([2,4,6,8,3],[3,4,5,6,7]))


#或等
print(np.logical_or([0,0],[1,1]))
print(np.logical_and([1,0],[1,1]))
print(np.logical_not([1,0]))

#检查元素
#至少有一个为true
print(np.any([0,1,0,0,1]))
#检查是否全部都是true
print(np.all([0,1,0,0,1]))
#自定义条件
arr=np.array([1,2,3,4,5])
#第一个值条件，第二个值为显示数据或者自定义显示其他，第三个值不符合的设置为其他
print(np.where(arr>3,arr,0))
print(np.where(arr>3,1,0))

arr=np.array([40,60,30,80,100])
print(np.where(arr>50,"及格","不及格"))
print(np.where(arr<50,"不及格",np.where(arr<80,"良好","优秀")))
print(np.select([arr>80,(arr>=80)&(arr<=50),arr<50,arr==60],["优秀","良好","不及格","及格"],default="未知"))

#排序
np.random.seed(0)
arr=np.random.randint(1,100,20)
print(arr)
arr.sort()
print(arr)
print(np.sort(arr))
print(np.argsort(arr))

#去重
print(np.unique(arr))

#数组的拼接
arr=np.array([1,2,3,4])
arr1=np.array([5,6,7,8])
print(np.concatenate((arr,arr1)))

#数组的分割
print(np.split(arr,2))
print(np.split(arr,[2,4,1]))



temps=np.array([28,30,29,31,32,30,29])
print(np.max(temps))
print(np.min(temps))
print('%.3f'%np.mean(temps))
print(len(temps[temps>30]))
print(np.cumsum(np.where(temps > 30, 1, 0))[-1])
print(np.count_nonzero(temps > 30))

score=np.array([85,90,78,92,88])
print(np.mean(score))
print(np.median(score))
print(score/10)
print(np.std(scores))

a = np.array( [[1, 2], [3, 4]])
b= np.array( [[5, 6], [7, 8]])
print(a+b)
print(a*b)
#矩阵点击乘法C11 = A[1:]*B[:1] = (1 2)*(5 7) = 1*5 + 2*7 = 19。 [[19 22]
# [43 50]]
print(a@b)


np.random.seed(0)
arr=np.random.randint(0,10,(3,4))
print(arr)
print(np.max(arr,axis=0))
print(np.min(arr,axis=1))
print(np.where(arr%2==1,-1,arr))
arr[arr%2==1]=-1
print(arr)

arr=np.arange(1,13)
print(arr)
res=np.reshape(arr,(3,4))
print(res)
print(np.sum(res,axis=0))
print(np.sum(res,axis=1))
print(np.mean(res,axis=0))

print(np.reshape(res,(res.size)))


np.random.seed(0)
arr=np.random.randint(0,20,(5,5))
print(arr)
print(arr[arr>10])

money=np.array([120,135,110,125,130,140])
print(np.sum(money))
print(np.mean(money))
print(np.var(money))
print(np.argmax(money)+1)
print(np.argmin(money)+1)

a=np.array([1,2,3])
b=np.array([4,5,6])
print(np.concatenate([a,b]))
print(np.reshape(np.concatenate([a,b]),(2,3)))

arr = np.array([2, 1, 2, 3, 1, 4, 3])
print(np.unique(arr))
print(np.unique(arr, return_counts=True))