import pandas as pd
import numpy as np

#df=pd.read_csv('data')
#print(type(df))
#print(df.tail())
#print(df.salary.mean())
#df.to_csv('data/')
#df=pd.read_json('data/')

s=pd.Series([1,2,np.nan,None,pd.NA])
print(s)
print(s.isna())
print(s.isnull)
df=pd.DataFrame([[1,pd.NA,2],[2,3,4],[None,6,7]],columns=['第一列','第二列','第三列'])
print(df)
print(df.isna())
print(df.isnull())
print(df.isna().sum())
print(s.isna().sum())
print(s.dropna())#剔除缺失值
print(df.dropna())
print(df.dropna(how='all'))#如果所有的值为缺失值，删除这一行
print(df.dropna(thresh=1))#如果至少又n个值不是缺失值，就保留
print(df.dropna(axis=1))#删除这一列数据
print(df.dropna(subset=['第一列']))#如果有某列有缺失值，则删除这一行
#df=pd.read_csv('data')
#df.tail()
#df.isna().sum(axis=0)
#df.head()
#print(df.fillna({'temp_max':20,'wind':2.5}).tail())#填充固定住
#print(df.fillna(df[['wind']].mean()).tail())#使用统计值填充
#print(df.ffill())#前面的值填充
#print(df.bfill().tail())#后面的值填充


data = {
"name": ['alice', 'alice', 'bob', 'alice', 'jack', 'bob'],
"age":[26,25,30, 25,35, 30],
'city' :['NY', 'NY', 'LA', 'NY', 'SF', 'LA']
}
df=pd.DataFrame(data)
print(df)
print(df.duplicated())
print(df.drop_duplicates())#去掉重复的
print(df.drop_duplicates(subset=['name']))#根据指定的列去掉重复的
print(df.drop_duplicates(subset=['name'],keep='last'))#根据指定的列去掉重复的保留最后一次或者最新的

df=pd.read_csv('data')
df['age']=df['age'].astype('int16')

#數据安形
import pandas as pd
data = {
'ID': [1, 2],
'name': ['alice','bob'],
'Math': [90, 85],
'English': [88, 92],
'Science': [95, 89]}
df = pd. DataFrame(data)
df.T
df2=pd.melt(df,id_vars=['ID','name'],var_name='科目',value_name='分数')#宽表转为长表
df2.sort_values('name')
#长表转宽表
pd.pivot(df2,index=["ID","name"],columns='科目',values="分数")


data = {
'ID': [1, 2],
'name':['alice smith','bob smith'],
'Math': [90, 85],
'English': [88, 92],
'Science': [95, 89]
}
df = pd.DataFrame(data)
#分列
df[[ 'first', 'last' ]] = df[' name']. str.split(" ", expand=True)

#数据分箱
df=pd.read_csv("date/")
df.head(10)
df1=df.head(10)[["em","salary"]]
pd.cut(df1['salary'], bins=2)#bins分段几个

pd.cut(df1[' salary'],bins=3).value_counts()
pd.cut(df1[' salary'], bins= [0,10000,20800,30000]) #bins=list, SinER区问
pd.cut(df1[' salary'], bins= [0,10000,20000,30000]).vatye_counts()
df1['收入范團'] =pd.cut(df1['salary'],bins=[910000,20000,30000],Labels=['低','中','高'])#bins=List，分成n段区间
pd.qcut(df1["salary"],3).vatye_counts()


# df.rename（）df.set_index() df.reset_index()
df = pd. DataFrame ({
'name':['jack', 'alice', 'tom','bob'],
'age':[20,30,40, 50],
'gender':['female', 'male', 'female', 'male ']
})
print(df)
df.set_index("name",inplace=True)
df.reset_index(inplace=True)
df.rename(columns={'age':"年龄"})
df.columns=['姓名','年龄','性别']


#时间数据的处理
d=pd.Timestamp('2025-05-02 10:22')

#分组聚合
df=pd.read_csv("date/")
df['deframent'].isna().sum()
df.dropna(subset=["deframent"])
df['deframent']=df['deframent'].astype('int64')
df.groupby('deframent').groups#查看分组
df.groupby('deframent').get_group(20)#查看分组
df2=df.groupby('deframent')[['salary']].mean()
df2['salary']=df2['salary'].round(2)


df = pd.DataFrame({
    "name":["张三","李四","王五","赵六","小明","小红"],
    "department":[10,20,10,20,30,10],
    "salary":[8000,12000,9000,15000,10000,8500]
})

print(df)

df.groupby("department")
#按照 department（部门） 分组。
df.groupby("department").groups
#查看有哪些组

#查看某一个组
df.groupby("department").get_group(20)

#求平均工资
df2 = df.groupby("department")[["salary"]].mean()

#保留两位小数
df2["salary"] = df2["salary"].round(2)

#一次统计所有
df.groupby("department")["salary"].agg(
    ["count","mean","max","min","sum"]
)