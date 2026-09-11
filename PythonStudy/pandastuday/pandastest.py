import numpy as np
import pandas as pd

np.random.seed(42)
values=np.random.randint(50, 101, 10)

indexes=[]
for i in range(1,len(values)+1):
    indexes.append("学生"+str(i))
scores=pd.Series(values, indexes,name='学生')
print(scores)
print(scores.mean())
print(scores.max())
print(scores.min())
print(len(scores[scores>scores.mean()]))
print(scores[scores>scores.mean()])
print(scores[scores>scores.mean()].count())
temperatures = pd.Series([28, 31, 29, 32, 30, 27, 33],
                         index=['周一','周二','周三','周四','周五','周六','周日'])
print(temperatures)
# 温度大于30℃
n = temperatures[temperatures > 30].count()
print(n)

# 平均温度
print(temperatures.mean())

# 按温度降序
tem = temperatures.sort_values(ascending=False)
print(tem)

# 相邻两天温差
difftem = temperatures.diff().abs()
print(difftem)

# 温差最大的两天
sortdiff = difftem.sort_values(ascending=False).index
print(*sortdiff[:2].tolist())

# 取出对应温度并按温度降序排列
result = temperatures.loc[sortdiff[:2]].sort_values(ascending=False)
print(result)

#给定某股票连续10个交易日的收盘价Series：
#计算每日收益率（当日收盘价/前日收盘价 - 1）pct_change这个函数
#找出收益率最高和最低的日期
# 计算波动率（收益率的标准差)
prices = pd.Series([102.3, 103.5, 105.1, 104.8, 106.2, 107.0, 106.5, 108.1,
109.3, 110.2], index=pd.date_range('2023-01-01', periods=10))

print(prices)
data=pd.date_range('2026-01-01', periods=6)
print(list(data))
pct=prices.pct_change()
print(pct)
print(pct.idxmax())
print(pct.idxmin())
print(pct.std())
print(pct.median())

#某产品过去12个月的销售量Serjes：
#计算季度平均销量（每3个月为一个季度）
#找出销量最高的月份计算月环比增长率
#找出连续增长超过2个月的月份
sales = pd.Series([120, 135, 145, 160, 155, 170, 180, 175, 190, 200, 210,
220],index=pd.date_range('2022-01-01', periods=12, freq='ME'))
print(sales)
#'ME', 'YE', 'QE', 'BME','BA', 'BQE'
salesroom=sales.resample('QS').mean()
print(salesroom)
print(sales.resample('YS').mean())
print(sales.max())
print(sales.idxmax())
#月环比增长率
print(sales.pct_change())

a=sales.pct_change()
b=a>0
var = b[b.rolling(3).sum()==3].keys().tolist()
print(var)


#某商店每小时销售额Series：
#按天重采样计算每日总销售额
#计算每天营业时间（8:00-22:00）和非营业时间的销售额比例
# 找出销售额最高的3个小时
np.random.seed(42)
hourly_sales = pd.Series(np.random.randint(0, 100, 24),
index=pd.date_range('2025-01-01', periods=24, freq='h'))
print(hourly_sales)
day_sales=hourly_sales.resample('D').sum()
print(day_sales)
#计算每天营业时间（8:00-22:00）和非营业时间的销售额比例
between_hourly=hourly_sales.between_time('8:00', '22:00')
print("between_hourly",between_hourly)
sales_index=hourly_sales[(hourly_sales.index.hour>=8)&(hourly_sales.index.hour<=22)]
not_sales_index=hourly_sales.drop(sales_index.index)
not_between_hourly=hourly_sales.drop(between_hourly.index)
print("计算每天营业时间",sales_index.sum())
print("计算每天营业时间",sales_index.sum()/(not_sales_index.sum()))
print("计算每天营业时间",sales_index.sum()/(day_sales-sales_index.sum()))
print("计算每天营业时间",between_hourly.sum()/(not_between_hourly.sum()))
# 找出销售额最高的3个小时
print(hourly_sales.nlargest(3))

