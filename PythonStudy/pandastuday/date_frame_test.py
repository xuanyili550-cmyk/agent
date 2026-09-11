import pandas as pd
import numpy as np
#案例1：学生成绩分析
#场景：某班级的学生成绩数据如下，请完成以下任务：
#：1. 计算每位学生的总分和平均分。
#2．找出数学成绩高于9日分或英语成绩高于85分的学生。
#3.按总分从高到低排序，并输出前3名学生。

data = {
'姓名':['张三','李四','王五','赵六','钱七'],
'数学':[85,92,78,88,95],
'英语':[90,88,85,92,80],
'物理':[75,80,88,85,90]
}
scores=pd.DataFrame(data)
print(scores)

scores["总分"]=scores[["数学","英语","物理"]].sum(axis=1)
print(scores)
print(scores.index.size-2)
scores["平均分"]=scores["总分"]/(scores.index.size-2)
print(scores)
scores1=scores[(scores['数学']>90)|(scores['英语']>85)]
print(scores1)
scores2=scores.nlargest(3,columns='总分')
print(scores2)
scores=scores.sort_values('总分',ascending=False).head(3)
print(scores)


#案例2：销售数据分析
#场景：某公司销售数据如下，请完成以下任务：
#1.计算每种产品的总销售额（销售额：单价×销量）。
#2. 找出销售额最高的产品。
#3．按销售额从高到低排序，并输出所有产品信息。
data = {
'产品名称':['A','B','C','D'],
'单价':[100,150,200,120],
'销量':[50,30,20,49]
}
df = pd.DataFrame(data)
print(df)
df['总销售额']=df['单价']*df['销量']
print(df)
print(df.sort_values('销量',ascending=False).head(1))
print(df.nlargest(1,columns='销量'))
print(df.sort_values('销量',ascending=False))

#1. 计算每位用户的总消费金额（消费金额= 商品单价 ×购买数量）
#2. 找出消费金额最高的用户，并输出其所有信息
#3. 计算所有用户的平均消费金額（保留2位小数）
#4. 统计电子产品的总购买数量
data ={
'用户1D':[101,102,103,104,105],
'用户名':['ALice','Bob','Charlie','David', 'Eve'],
'商品类别':['电子产品','服饰','电子产品','家居','服饬'],
'商品单价':[120,308,800,150,200],
'购买数量':[2,3,2,5,4]
}
df = pd. DataFrame(data)
print(df)
df['总消费金额']=df['商品单价']*df['购买数量']
print(df['总消费金额'])
print(df.nlargest(1,'总消费金额'))
print(df.sort_values('总消费金额',ascending=False).head(1))

df['平均消费金額']=df['总消费金额'].mean()
print(df['平均消费金額'])
su=df[df['商品类别']=='电子产品']['购买数量'].sum()
print(su)

