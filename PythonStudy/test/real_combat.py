import matplotlib.pyplot as  plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams


rcParams["font.sans-serif"] = ["Hiragino Sans GB"]
df=pd.read_csv("data/da.csv")
df.drop(columns='origin_url',inplace=True)
df.isna().sum()
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)#删除重复数据
df['area']=df['area'].str.replace('m','').astype(float)
df['price']=df['price'].str.replace('万','').astype(float)
df['toward']=df['toward'].astype('category')
df['unit']=df['unit'].str.replace('元/m2','').astype(float)
df['year']=df['year'].str.replace('年建','').astype(int)
#异常处理
q1=df['price'].quantile(0.25)
q3=df['price'].quantile(0.75)
iqr=q3-q1
low_price=q1-1.5*iqr
high_price=q3+1.5*iqr
df=df[(df['price']<high_price)&(df['price']>low_price)]

#新数据特征的构造
df['district']=df['address'].str.split('-').str[0]
df['floor_type']=df['floor'].str.split('（').str[0].astype('category')

def fun1(str1):
    if pd.isna(str1):
        return '未知'
    elif '低' in str1:
        return '低楼层'
    elif '中' in str1:
        return '中楼层'
    elif '高' in str1:
        return '高楼层'
    else:
        return '未知'
df['floor_type2']=df['floor'].apply(fun1).astype('category')

df['zxs']=df['city'].apply(lambda x:1 if x in["北京","上海","天津","重庆"] else 0)


# 卧室的敛量bedrooms
df[' bedrooms'] = df['fooms']. str. split ('室'). str(0). astype (int)



#迭擇数値型特征
a = df[['price', 'area', 'unit', 'building_age']].corr()#**
# 对房价的影响最大的几个因素的排序
a['price'].sort_values(ascending=False) [1:]
# 相关性的热力图
plt.figure(figsize = (10,10))
sns.heatmap(a,cmap='coolwarm')
plt.title("热力图")
plt.tight_layout()
plt.show()