import pandas as pd

s=pd.Series([1,2,3,4,5])
#s=pd.Series([1,2,3,4,5],index=["A","B","C","D","E"],name="月份")
print(s)
# 通过字典来创建
s = pd.Series({"a":1, "b":2, "c":3, "d":4, "e":5})
print(s)
s1 = pd. Series(s, index=["a", "c"])
print(s1)

#属性
print(s.index)#获取属性索引
print(s.values)
print(s.shape,s.ndim,s.size)
s.name='data'
print(s.dtype,s.name)
print(s.loc['b'])#显示索引按照标签
print(s.iloc[0:2])#隐士索引按照位置
print(s.at['a'])#显示索引
print(s.iat[0])#隐士索引
print(s.head(3))#默认取前五行
print(s.tail(2))#默认取前五行
print(s.describe())#描述性信息
print(s.count())#获取元素的个数
print(s.keys())#方法索引
print(s.isna())#检查series里的每一个元素是否为缺失值
print(s.isin([4]))#检查series里的每一个元素是否在这个区间里
print(s.mean())
print(s.std())
print(s.var())
print(s.sum())
print(s.median())
print(s.sort_values())
print(s.quantile(0.25))