import numpy as np
import matplotlib.pyplot as plt  
from sklearn import datasets

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data  # 特征数据
y = iris.target  # 真实标签
feature_names = iris.feature_names
target_names = iris.target_names
data=X[:,0:2]


class MyKmean():
    def __init__(self,k,data):
        self.k=k
        self.data=data

# 计算距离 
    def euclDistance(self,vector1, vector2):  
        return np.sqrt(sum((vector2 - vector1)**2))
  
    def initCentroid(self,data,k):
        num_Samples,dim=data.shape
        centroids=np.zeros((k,dim))
        index=np.random.randint(0,num_Samples,size=k)
        for i in range(k):
            centroids[i,:]=self.data[index[i],:]
        return centroids


    def kmeans_1(self,data,k):

        num_Samples=data.shape[0]
        clusterData=np.zeros((num_Samples,2))
        clusterChange=True
        centroids=self.initCentroid(data,k)
        while clusterChange:
            for i in range(num_Samples):
                cur_clu=clusterData[i,0]
                min_dis=10**5
                for j in range(k):
                    dis=self.euclDistance(data[i,:],centroids[j,:])
                    if min_dis>dis:
                        min_dis=dis
                        clusterData[i,0]=j
                clusterData[i,1]=min_dis
                if clusterData[i,0]!=cur_clu:
                    clusterChange=False

            for i in range(k):
                data_1=data[np.nonzero(clusterData[:,0]==i)]
                # print((clusterData[:,0]==i).shape)
                # print(np.nonzero(clusterData[:,0]==i))
                # print(data_1)
                centroids[i,0],centroids[i,1]=np.mean(data_1,axis=0)
        return centroids, clusterData  

    def showcase_1(self,data,k,centroids,clusterData):
        num_Samples,dim=data.shape
        print(num_Samples,dim)
        if dim != 2:
            raise ValueError
 
        if k > 5:
            print('k is too large')
    #绘制各个点
    #对应聚类结果和真实标签
        mapping = {}
        for i in range(k):
        # 找出预测为i的样本的真实标签
            indices = clusterData[:,0] == i
            # print(indices)
            true_labels = y[indices]

        # 统计真实标签中最多的类别
            if len(true_labels) > 0:
                most_common = np.bincount(true_labels).argmax()
                mapping[i] = most_common

    # 调整预测标签
        y_label = np.array([mapping[label] for label in clusterData[:,0]])
        x0 = []
        x1 = []
        x2 = []
        y0 = []
        y1 = []
        y2 = []
        # 切分不同类别的数据
        for i in range(num_Samples):
            if (y_label[i])==0.:
                x0.append(data[i,0])
                y0.append(data[i,1])
            elif int(y_label[i])==1.:
                x1.append(data[i,0])
                y1.append(data[i,1])
            elif int(y_label[i]) == 2.:
                x2.append(data[i,0])
                y2.append(data[i,1])
        # 画图
        fig, axes = plt.subplots(1,1)
        scatter0 = axes.scatter(x0, y0, c='b', marker='o')
        scatter1 = axes.scatter(x1, y1, c='c', marker='o')
        scatter2 = axes.scatter(x2, y2, c='g', marker='o')
        #画图例
        axes.legend(handles=[scatter0,scatter1,scatter2],labels=['setosa','versicolor','virginica'],loc='best')
        axes.set_xlabel('sepal width')
        axes.set_ylabel('petal length')
        axes.set_title('Prediction Result')
        num=0
        for i in range(len(y_label)):
            if y_label[i]==y[i]:
                num+=1
    #     for i in range(0,num_Samples):
    #         plt.plot(data[i,0],data[i,1],marks1[round(clusterData[i,0])])

    # #绘制质心
    #     for i in range(k):
    #         plt.plot(centroids[i,0],centroids[i,1],marks2[i])
        marks2=['+r', 'sr', 'dr', '<r','+r']
        for i in range(k):
            plt.plot(centroids[i,0],centroids[i,1],marks2[i])

        plt.show()
        return float(num)/num_Samples,fig
    
    def showcase_2(self,data,k,centroids,clusterData):
        num_Samples,dim=data.shape
        if dim != 2:
            raise ValueError
        marks1=['sc', 'ob', 'og', 'ok','oc']
        marks2=['+r', 'sr', 'dr', '<r','+r']
        fig, axes = plt.subplots(1,1)
        if k > len(marks1):
            print('k is too large')
        for i in range(0,num_Samples):
            axes.plot(data[i,0],data[i,1],marks1[round(clusterData[i,0])])

    #绘制质心
        for i in range(k):
            axes.plot(centroids[i,0],centroids[i,1],marks2[i])
        axes.set_xlabel('sepal width')
        axes.set_ylabel('petal length')
        axes.set_title('Prediction Result')
        plt.show()
        return fig

    
    def showResult(self):
        # 设置k值
        # k = 3
        min_loss = 10000
        min_loss_centroids = np.array([])
        min_loss_clusterData = np.array([])

        for i in range(100):
            # centroids 簇的中心点 
            # cluster Data样本的属性，第一列保存该样本属于哪个簇，第二列保存该样本跟它所属簇的误差
            centroids, clusterData = self.kmeans_1(self.data, self.k)  
            loss = sum(clusterData[:,1])/self.data.shape[0]
            if loss < min_loss:
                min_loss = loss
                min_loss_centroids = centroids
                min_loss_clusterData = clusterData
                
        #     print('loss',min_loss)
        print('cluster complete!')      
        centroids = min_loss_centroids
        clusterData = min_loss_clusterData

        # 显示结果          
        if self.k!=3:
            fig=self.showcase_2(self.data, self.k, centroids, clusterData)
            return fig
        else:
            accuracy,fig=self.showcase_1(self.data, self.k, centroids, clusterData)
        # print(accuracy)
        # accuracy=accuracy/150
            print(f"精准度为：{accuracy}")
            return accuracy,fig

    #肘部法则
    def elbow(self):
        list_lost = []
        for k in range(2,6):
            min_loss = 10000
            # min_loss_centroids = np.array([])
            # min_loss_clusterData = np.array([])
            for i in range(50):
                # centroids 簇的中心点 
                # cluster Data样本的属性，第一列保存该样本属于哪个簇，第二列保存该样本跟它所属簇的误差
                centroids, clusterData = self.kmeans_1(self.data, k)  
                loss = sum(clusterData[:,1])/self.data.shape[0]
                if loss < min_loss:
                    min_loss = loss
                    # min_loss_centroids = centroids
                    # min_loss_clusterData = clusterData
            list_lost.append(min_loss)
        return list_lost


my_k=MyKmean(5,data)
my_k.showResult()

