import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB"]
penguins=pd.read_csv("data/da.csv")
penguins.dropna(inplace=True)#去重
penguins.info()

#直方图
sns.histplot(data=penguins,x='species')

sns.countplot(data=penguins,x='island')

#散点图
sns.scatterplot(data=penguins,x='body_mass_g',y='flipper_length_mm',hue='sex')
#蜂窝图
sns.jointplot(data=penguins,x='body_mass_g',y='flipper_length_mm',hue='hex')

#二维核密度估计图
sns.kdeplot(data=penguins,x='body_mass_g',y='flipper_length_mm')

#通过fill=true填充 char=true为颜色
sns.kdeplot(data=penguins,x='body_mass_g',y='flipper_length_mm',fill=True,char=True)

#条形图
sns.barplot(data=penguins,x='body_mass_g',y='flipper_length_mm',estimator='mean',errorbar=None)

#箱线图
sns.boxplot(data=penguins,x='body_mass_g',y='flipper_length_mm')

#小提琴图
sns.violinplot(data=penguins,x='body_mass_g',y='flipper_length_mm')

#成对关系图
sns.pairplot(data=penguins,hue='species')
