import pandas as pd
import numpy as np

#创建
s1=pd.Series([1,2,3,4,5])
s2=pd.Series([6,7,8,9,10])
df=pd.DataFrame({"第一列":s1,"第二列":s2})
print(df)
print(type(df["第一列"]))

df=pd.DataFrame(
    {
        "id":[1,2,3,4,5],
        "name":["hhs","hee","ssds","weqw","fasdsa"],
        "age":[6,7,8,9,10],
        "score":[50,40,60,80,70]
    },index=[1,2,3,4,5],columns=["name","age","score"]
)
print(df)

#属性
print("行索引")
print(df.index)
print("列标签")
print(df.columns)
print("值")
print(df.values)
print("纬度",df.ndim)
print("数据类型",df.dtypes)
print("形状",df.shape)
print("个数",df.size)
print("行列转置",df.T)
print("行列转置",df.T.index)
print("行列转置",df.T.columns)
print("获取元素某行",df.loc[4])
print("显获取元素某列",df.loc[:,"name"])
print("隐获取元素某列",df.iloc[:,0])
#获取单个元素
print(df.at[3,"score"])
print(df.iat[2,1])
print(df.loc[3,"score"])
print(df.iloc[2,1])
print(df.name)
print(type(df.name))
print(type(df[["name"]]))
print(type(df[["name","score"]]))

#访问dataframe的数据
print(df.head())#查看前n行数据，默认前5行
print(df.tail(3))#查看后n行数据，默认后5行
print(df[df.score>50])
#随机抽取
print(df.sample())
print(df.isin([9,'ssds']))#是否在某个集合里面
print(df.isna())#是否是缺失值
print(df["score"].sum())
print(df.value_counts())#出现次数
print(df.drop_duplicates())#去重
print(df.replace(15,50))#替换某一个值
df.cumsum()#累加
df1 = df[['score', 'age']]
print(df1)
print(df1.cummax(axis=0))
print(df.sort_values(by=["score","age"],ascending=[True,False]))
print(df.nlargest(2,columns=["score","age"]).sort_values(by="age",ascending=False))