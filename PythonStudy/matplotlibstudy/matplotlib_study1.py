import matplotlib.pyplot as plt


#对表绘制
#绘制数据
moth=['1','2','3',"4"]
sales=[100,50,80,130]
f1=plt.subplot(2,2,1)
f1.plot(moth,sales)
f2=plt.subplot(2,2,2)
f2.bar(moth,sales)
#............
#显示图标
plt.show()