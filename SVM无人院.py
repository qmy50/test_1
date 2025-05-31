from sklearn import svm               # svm函数需要的
import numpy as np                    # numpy科学计算库
from sklearn import model_selection   
import matplotlib.pyplot as plt       # 画图的库
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

def iris_type(s):
    it = {b'setosa': 0, b'versicolor': 1, b'virginica': 2}
    return it[s]

  # 之前保存的文件路径
data = np.loadtxt(r'D:\vscode\machine_learning\机器学习\机器学习\支持向量机SVM\iris.txt',dtype=float,delimiter=',',converters={4: iris_type})     
print(data)
np.savetxt('iris1.txt', data,fmt='%.2f')
X, y = np.split(data, [4],axis=1)
print(X)
x = X[:, 0:2]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3)
print(x_train)


model = svm.SVC(kernel='rbf',                      # 核函数
               gamma=0.1,
             decision_function_shape='ovo',     
              C=1)
model.fit(x_train, y_train)      

print(model.score(x_train, y_train)) 
y_train_hat=model.predict(x_train)
y_train_1d=y_train.reshape((-1))


print(model.score(x_test,y_test))

plt.figure()
plt.subplot(121)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.reshape((-1)), edgecolors='k',s=50)
plt.subplot(122)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train_hat.reshape((-1)), edgecolors='k',s=50)
plt.show()

# 3. 在测试集上进行预测
y_pred = model.predict(x_test)
# y_pred_proba = model.predict_proba(x_test)

# 4. 计算基本评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")


# 5. 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels='a', yticklabels='b')
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("SVM分类器的混淆矩阵")
plt.show()

