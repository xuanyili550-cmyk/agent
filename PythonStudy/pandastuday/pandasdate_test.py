import numpy as np
import pandas as pd

df=pd.read_csv('')
df.info()

print(df.isna())
df.dropna(inplace=True)
df.isna().sum()

df['sex']=df['sex'].astype('category')#替换类型
df.info()
df['bill_rato']=df['bill_length_mm']/df["bill_depth_mm"]#数据构造

df['mass_level']=pd.cut(df['boby_mass_g'],bins=3,labels=['低','中','高'])
print(df['mass_level'].value_counts())

df.groupby(["sex"]).agg({
    'boby_mass_g':["count","mean","max","min","sum"],
})

df.groupby(["island"]).agg({
    'boby_mass_g':["count","mean","max","min","sum"],
})

df.groupby(["island","sex"]).agg({
    'boby_mass_g':["count","mean","max","min","sum"],
})