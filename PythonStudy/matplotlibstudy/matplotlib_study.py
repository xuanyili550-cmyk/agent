import random
from cProfile import label

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import lineStyles
from sympy import rotations
from sympy.abc import alpha

'''
折线图。plat
条形图   bar
饼图   pie
散点图  scatter
箱线图  boxplot

多图表
组合图
'''

#折线图
#创建图表，设置大小
plt.figure(figsize=(10,5))
#rcParams['font.family']='STHeiti'  #字体
#rcParams['font.sans-serif']=10
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB"]
#rcParams['font.sans-serif']='STHeiti'

#绘制数据
moth=['1月','2月','3月',"4月"]
sales=[100,50,80,130]
#绘制折线图
plt.plot(moth,sales,label='产品',color='orange',linestyle='--',linewidth=2,marker='o')
#标题
plt.title('销售',color='red',fontsize=20)

#添加坐标轴的标签
plt.xlabel('月',fontsize=10)
plt.ylabel('销售额（万）',fontsize=10)
#添加图列
plt.legend(loc='upper left')
#添加网格线  两个方向一起
plt.grid(True,alpha=0.1,color='blue',linestyle='--')
#plt.grid(axis='y')
#plt.grid(axis='x')
#设置刻度字体的大小
plt.xticks(rotation=0,fontsize=30)
plt.yticks(rotation=0,fontsize=30)
#设置轴的范围
plt.ylim(0,160)
#plt.xlim('1月','12月')
#每个数据点上显示的值
for x,y in zip(moth,sales):
    plt.text(x,y+1,str(y),ha='center',va='bottom',fontsize=10)
#显示图标
plt.show()

#柱状图
#创建图表，设置大小
plt.figure(figsize=(10,5))
#rcParams['font.family']='STHeiti'  #字体
#rcParams['font.sans-serif']=10
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB"]
#数据
subjects=['语文','数学','英语','科学']
scores=[84,32,54,65,67]
#绘制柱状图
plt.bar(subjects,scores,label("小红"),color='orange',linestyle='--',linewidth=2,marker='o')
#标题
plt.title('成绩',color='red',fontsize=20)
#添加图列
plt.legend(loc='upper left')
#添加网格线  两个方向一起
plt.grid(axis='y',alpha=0.1,color='blue',linestyle='--')
#添加坐标轴的标签
plt.xlabel('科目',fontsize=10)
plt.ylabel('成绩',fontsize=10)
#设置刻度字体的大小
plt.xticks(rotation=0,fontsize=30)
plt.yticks(rotation=0,fontsize=30)
#设置轴的范围
plt.ylim(0,100)
#每个数据点上显示的值
for x,y in zip(subjects,scores):
    plt.text(x,y+1,str(y),ha='center',va='bottom',fontsize=10)
#自动优化排版
plt.tight_layout()
#显示图标
plt.show()

#条形图

#创建图表，设置大小
plt.figure(figsize=(10,5))
#rcParams['font.family']='STHeiti'  #字体
#rcParams['font.sans-serif']=10
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB"]
#数据
subjects=['united states','china','japan','germany','india']
scores=[84,92,4,19,40]
#绘制条形图
plt.barh(subjects,scores,label("gdp"),color='orange',linestyle='--',linewidth=2,marker='o')
#标题
plt.title('gdp',color='red',fontsize=20)
#添加图列
plt.legend(loc='upper left')
#添加网格线  两个方向一起
plt.grid(axis='y',alpha=0.1,color='blue',linestyle='--')
#添加坐标轴的标签
plt.xlabel('gdp',fontsize=10)
plt.ylabel('国家',fontsize=10)
#设置刻度字体的大小
plt.xticks(rotation=0,fontsize=30)
plt.yticks(rotation=0,fontsize=30)
#设置轴的范围
plt.ylim(0,100)
#每个数据点上显示的值
for x,y in zip(subjects,scores):
    plt.text(x,y+1,str(y),ha='center',va='bottom',fontsize=10)
#自动优化排版
plt.tight_layout()
#显示图标
plt.show()


#饼图

#创建图表，设置大小
plt.figure(figsize=(10,5))
#rcParams['font.family']='STHeiti'  #字体
#rcParams['font.sans-serif']=10
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB"]
#数据
things=['学习','娱乐','睡觉','运动','其他']
times=[6,4,1,8,5]
#配色
colors = ['#66b3ff', '#99ff99', '#ffcc99', '#Ff9999', '#Ff4499']
#设置突出的位置
explode=[0.1,0,0,0,0]
 # #绘制条形图
plt.pie(times,labels=things,autopct='%.4f%%' #小数点百分比
        ,startangle=90 #调整初始画图的角度
        ,colors=colors,
        wedgeprops={'width':0.5}#圆环宽度
        ,pctdistance=0.6 #设置百分比数字距离
        ,explode=explode
        )

#标题
plt.title('一天的分布',color='red',fontsize=20)
plt.text(0,0,'总计：/100%',ha='center',va='bottom',fontsize=10)
#自动优化排版
plt.tight_layout()
#显示图标
plt.show()



#散点图

#创建图表，设置大小
plt.figure(figsize=(20,15))
#rcParams['font.family']='STHeiti'  #字体
#rcParams['font.sans-serif']=10
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB"]
#rcParams['font.sans-serif']='STHeiti'

#绘制数据
scores= []
hours=[]
for i in range(500):
    tmp = random.uniform(0,10)
    scores.append(tmp)
    hours.append(2*tmp + random.gauss(0,2))#高斯噪声
#绘制折线图
plt.scatter(hours,scores,label='成绩',color='blu',
            alpha=0.4,#透明度
            s=20,#圆点大小
            )
#标题
plt.title('成绩',color='red',fontsize=20)

#添加坐标轴的标签
plt.xlabel('小时',fontsize=10)
plt.ylabel('成绩',fontsize=10)
#添加图列
plt.legend(loc='upper left')
#添加网格线  两个方向一起
plt.grid(True,alpha=0.1,color='blue',linestyle='--')
#plt.grid(axis='y')
#plt.grid(axis='x')
#设置刻度字体的大小
plt.xticks(rotation=0,fontsize=30)
plt.yticks(rotation=0,fontsize=30)
#设置轴的范围
plt.ylim(0,40)
#plt.xlim('1月','12月')
plt.plot([0,10],[0,20],colors='red',linewidth=2,linestyle='--')
#显示图标
plt.show()


#箱线图
# 模拟 3 门课的成绩
data = {
'语文':[82,85,88,70,90,76,84,83,95],
'数学':[175.80, 79,93, 88,82,87,39,92],
'英语':[70,72,68,65,78,80,85,90,95]
}
plt.figure(figsize=(8, 6))
plt.boxplot(data.values(), tick_labels=data.keys())
plt.title("各科成績分布（箱线图）" )
plt.ylabel("分数")
plt.grid(True, axis='y', linestyle='--',alpha=0.5)
plt.show()

#对表绘制
#绘制数据
moth=['1','2','3',"4"]
sales=[100,50,80,130]
f1=plt.subplot(2,2,1)#生成子图，行，列，索引
#f1=plt.subplot(221)
f1.plot(moth,sales)
f2=plt.subplot(2,2,2)
f2.bar(moth,sales)
#............
#显示图标
plt.show()
