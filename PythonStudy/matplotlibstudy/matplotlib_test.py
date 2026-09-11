import matplotlib.pyplot as  plt
import pandas as pd
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Hiragino Sans GB"]

#倒入数据
df=pd.read_csv("data/da.csv")
#绘制气温变化图
df['date']=pd.to_datetime(df['date'])
#设置图形大小
plt.figure(figsize=(15,10))
plt.plot(df['date'],df['temp_max'],lable='最高气温')
plt.plot(df['date'],df['temp_min'],lable='最低气温')
plt.title('气温变化图')
plt.xlabel('日期')
plt.ylabel('气温')
plt.legend()

df['temp_mean']=(df['temp_max']+df['temp_min'])/2

plt.figure(figsize=(15,10))
plt.plot(df['date'],df['temp_mean'],lable='平均气温')
plt.title('气温变化图')
plt.xlabel('日期')
plt.ylabel('气温')
plt.legend()
#绘制降水量的直方图
plt.hist(df['precipitation'],bins=5)


plt.show()