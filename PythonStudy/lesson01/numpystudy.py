import  numpy as np

arr=np.array(5)
print(arr)
print("arr的维度：",arr.ndim)

#不同纬度的数据类型转换为相同的类型
arr=np.array([1,"hello"])
print(arr)
print("arr的维度：",arr.ndim)

#不同纬度的数据类型转换为相同的类型
arr=np.array([1,3,5,2.5])
print(arr)
print("arr的维度：",arr.ndim)

#不同纬度的数据类型转换为相同的类型
arr=np.array([[1,3,5],[2,3,5]])
print(arr)
print("arr的维度：",arr.ndim)

#属性

arr=np.array(1)
print(arr)
print("数组的形状",arr.shape)
print("数组的纬度：",arr.ndim)
print("数组的元素的个数：",arr.size)
print("数组的数据类型：",arr.dtype)
print("数组的元素转置：",arr.T)


arr=np.array([1,2,3])
print(arr)
print("数组的形状",arr.shape)
print("数组的纬度：",arr.ndim)
print("数组的元素的个数：",arr.size)
print("数组的数据类型：",arr.dtype)
print("数组的元素转置：",arr.T)


arr=np.array([[1,2,3],[4,5,6]])
print(arr)
print("数组的形状",arr.shape)
print("数组的纬度：",arr.ndim)
print("数组的元素的个数：",arr.size)
print("数组的数据类型：",arr.dtype)
print("数组的元素转置：",arr.T)


arr=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr)
print("数组的形状",arr.shape)
print("数组的纬度：",arr.ndim)
print("数组的元素的个数：",arr.size)
print("数组的数据类型：",arr.dtype)
print("数组的元素转置：",arr.T)


