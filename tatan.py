import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


# df_train=pd.read_csv('data_processed.csv')
# # print(df_train.head())
# first_column = df_train.columns[0]  # 获取第一列的列名
# df_train = df_train.drop(first_column, axis=1)  # 删除第一列
# # print(df_train.head())

# label_train = df_train.loc[:, 'Survived']
# data_train = df_train.drop(columns=['Survived'])
# x_train,x_test,y_train,y_test=train_test_split(data_train,label_train,random_state=1,train_size=0.7)


# RF = RandomForestClassifier(n_estimators=200)
# RF.fit(x_train, y_train)
# print(RF.score(x_test,y_test))
# # RF.score(x_test, y_test)
# my_data2=[[1,0,0.34,0.15,1,0,0,2]]
# y_train_hat=RF.predict(my_data2)
# print(y_train_hat)

# # 定义参数网格
# param_grid = {
#     'n_estimators': [50, 100, 150,200],
#     'max_depth': [None, 10, 20,30],
#     'min_samples_split': [2, 5,10,20]
# }

# # 创建GridSearchCV对象
# grid_search = GridSearchCV(
#     estimator=RF,
#     param_grid=param_grid,
#     cv=5,                 # 5折交叉验证
#     scoring='accuracy',   # 评估指标为准确率
#     n_jobs=-1             # 使用所有CPU核
# )

# # 执行网格搜索
# better_RF=grid_search.fit(x_train, y_train)
# print(better_RF.best_params_)
# print(better_RF.score(x_test,y_test))

# my_data2=[[1,0,0.34,0.15,1,0,0,2]]
# y_train_hat=RF.predict(my_data2)
# print(y_train_hat)

# pred_RFC = better_RF.predict(x_train)

# # 混淆矩阵
# print(confusion_matrix(y_train, pred_RFC))

# print(classification_report(y_train, pred_RFC))

# keys = ['Pclass', 'Sex', 'Age','Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S','Size']
# values = RF.feature_importances_
# my_dic=dict(zip(keys,values))
# print(my_dic)
# for i,j in my_dic.items():
#     print(f'{i}的重要性为：{j*100}%')


def load_tan():
    df_train=pd.read_csv('data_processed.csv')
    # print(df_train.head())
    first_column = df_train.columns[0]  # 获取第一列的列名
    df_train = df_train.drop(first_column, axis=1)  # 删除第一列
    label_train = df_train.loc[:, 'Survived']
    data_train = df_train.drop(columns=['Survived'])
    x_train,x_test,y_train,y_test=train_test_split(data_train,label_train,random_state=1,train_size=0.7)
    return x_train,x_test,y_train,y_test


def load_RF():
    x_train,x_test,y_train,y_test=load_tan()
    RF = RandomForestClassifier(max_depth=None,min_samples_split=10,n_estimators=200)
    RF.fit(x_train, y_train)
    return RF

my_data2=[[1,0,0.34,0.15,1,0,0,2]]
my_RF=load_RF()
print(my_RF.predict(my_data2))