import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('house_price.txt',dtype=int,delimiter=',')     

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
            num=num+1

    return b,k
my_b,my_k=gd(b,k)
print(my_b,my_k)