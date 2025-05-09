from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error

# 载入数据
data = np.loadtxt('house_price.txt',dtype=int,delimiter=',')     
print(data)
x_data=data[:,0]
y_data=data[:,1]


lr=0.0001
b=0
k=0
epoch=50

def loss(k,b,x_data,y_data):
    res=0
    for i,j in zip(x_data,y_data):
        res=res+(j-(k*i+b))**2
    res=res/(2*len(x_data))
    return res


def gd(b,k):
    num=0
    for i in range(epoch):
        for j in range(len(x_data)):
            b_cur=b-lr*(k*x_data[j]+b-y_data[j])/len(x_data)
            k_cur=k-lr*(k*x_data[j]+b-y_data[j])/len(x_data)*x_data[j]
            b=b_cur
            k=k_cur
    return b,k
my_b,my_k=gd(b,k)
print(f"斜率为：{my_k} 截距为:{my_b}")
print('损失函数：'+str(loss(my_k,my_b,x_data,y_data)))
ans=my_k*130+my_b
print('结果为：'+str(ans))