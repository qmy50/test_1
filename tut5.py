import streamlit as st
from PIL import Image
import house as h
import pandas as pd
import python_kmean as k_Mean
from sklearn import datasets
from sklearn import model_selection   
import matplotlib.pyplot as plt       # 画图的库
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn import svm   
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

@st.cache_resource
def load_tan():
    df_train=pd.read_csv('data_processed.csv')
    # print(df_train.head())
    first_column = df_train.columns[0]  # 获取第一列的列名
    df_train = df_train.drop(first_column, axis=1)  # 删除第一列
    label_train = df_train.loc[:, 'Survived']
    data_train = df_train.drop(columns=['Survived'])
    x_train,x_test,y_train,y_test=train_test_split(data_train,label_train,random_state=1,train_size=0.7)
    return x_train,x_test,y_train,y_test

@st.cache_resource
def load_RF():
    x_train,x_test,y_train,y_test=load_tan()
    RF = RandomForestClassifier(class_weight='balanced',
    min_samples_leaf=5,
    max_depth=8,
    random_state=42,
    n_estimators=200)
    RF.fit(x_train, y_train)
    return RF


@st.cache_resource
def load_model():
    iris = datasets.load_iris()
    X = iris.data  # 特征数据
    y = iris.target  # 真实标签
    feature_names = iris.feature_names
    target_names = iris.target_names
    data=X[:,0:2]
    return data

def iris_type(s):
    it = {b'setosa': 0, b'versicolor': 1, b'virginica': 2}
    return it[s]


@st.cache_resource
def k_Mean_model(slider_val,data):
    my_k=k_Mean.MyKmean(slider_val,data)
    return my_k


@st.cache_resource
def load_iris():
    data = np.loadtxt('iris.txt',dtype=float,delimiter=',',converters={4: iris_type})     
    # print(data)
    X, y = np.split(data, [4],axis=1)
    # print(X)
    x = X[:, 0:2]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3)
    # print(x_train)
    return x_train,x_test,y_train,y_test

@st.cache_resource
def SVM_model(option_model,val):
    model = svm.SVC(kernel=option_model,                      # 核函数
               gamma=0.1,
             decision_function_shape='ovo',     
              C=val)
    return model

@st.cache_resource
def confusion(test_data,predict_data):
    cm = confusion_matrix(test_data, predict_data)
    fig=plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['setosa','versicolor','virginica'], yticklabels=['setosa','versicolor','virginica'])
    plt.xlabel("predict")
    plt.ylabel("true")
    plt.title("SVM result")
    return fig

st.title('机器学习作业展示：')

if "step" not in st.session_state:
    st.session_state['step']=1

def goto_step(step_num:int):
    st.session_state['step']=step_num

if st.session_state.step==1:
    st.subheader('算法选择')
    st.button(label='梯度下降——房价问题',on_click=goto_step,args=((2,)))
    st.button(label='K聚类——鸢尾花分类',on_click=goto_step,args=((3,)))
    st.button(label='SVM——鸢尾花分类',on_click=goto_step,args=((4,)))
    st.button(label='随机森林——泰坦尼克存活预测',on_click=goto_step,args=((5,)))


if st.session_state.step==2:
    st.title('梯度下降')
    st.button(label='Go prev',on_click=goto_step,args=((1,)))
    example="""核心代码:b_cur=b-lr*(k*x_data[j]+b-y_data[j])/len(x_data)
                k_cur=k-lr*(k*x_data[j]+b-y_data[j])/len(x_data)*x_data[j] """

    st.code(example,language='python')

    st.divider()

    slider_val=st.slider(label='请输入房屋面积：',
          min_value=50,
          max_value=150,
          value=0,
          step=1)
    ans=h.my_k*slider_val+h.my_b

    st.write("预测房价结果为：",'%.2f' % ans)


if st.session_state.step==3:
    st.title('K聚类')
    data=load_model()
    st.button(label='Go prev',on_click=goto_step,args=((1,)))
    slider_val=st.slider(label='请选择k值:',
          min_value=2,
          max_value=5,
          value=2,
          step=1)
    my_k=k_Mean_model(slider_val,data)
    
    # if slider_val==3:
    #     # st.balloons()
    #     # acc,my_fig=my_k.showResult()
    #     acc,_=my_k.showResult()
    #     acc=round(acc,2)
    #     st.write(f'预测准确度为：{acc}')
    # else:
    #     imge_path=['2.png','4.png','5,png']
    #     images=[]

    #     img = Image.open(imge_path[slider_val])
    #     images.append(img)
        # my_fig=my_k.showResult()
    st.write('分类可视化如下:')
    imge_path=['2.png','3.png','4.png','5.png']
    images=[]
    img=Image.open(imge_path[slider_val-2])
    if slider_val==3:
        acc,my_fig=my_k.showResult()
        acc,_=my_k.showResult()
        acc=round(acc,2)
        st.write(f'预测准确率为：{acc}')
        st.image(img)
    else:
        st.image(img)

    # st.pyplot(my_fig)
#肘部法则
    image_1=Image.open('image1.png')
    st.write('k取不同值时损失函数结果如下,可见取3时为肘部,符合肘部法则')
    st.image(image_1)

if st.session_state.step==4:
    st.title('SVM-支持向量机')
    st.button(label='Go prev',on_click=goto_step,args=((1,)))
    option = st.radio(
    "请选择你的核函数",
    ("rbf", "sigmoid", "linear", "poly")
)
    slider_val=st.slider(label='请选择c值:',
          min_value=0.02,
          max_value=1.5,
          value=1.,
          step=0.05)
    x_train,x_test,y_train,y_test=load_iris()
    svm_model=SVM_model(option,slider_val)
    svm_model.fit(x_train, y_train) 
    st.subheader('模型评估')
    st.write(f'测试集准确率为:{svm_model.score(x_test, y_test):.3f}') 
    st.write('混淆矩阵如下：')
    y_pred=svm_model.predict(x_test)
    st.pyplot(confusion(y_test,y_pred))

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.write(f"精确率: {precision:.3f}")
    st.write(f"召回率: {recall:.3f}")
    st.write(f"F1分数: {f1:.3f}")

if st.session_state.step==5:
    st.title('随机森林预测')
    st.button(label='Go prev',on_click=goto_step,args=((1,)))
    my_data=[]
    Pclass = st.selectbox(
        "请选择船舱等级",
        [1, 2, 3],
    )
    my_data.append(Pclass)
    Sex = st.selectbox(
    "请选择性别",
    ['男','女']
    )
    dict_sex={'男':1,'女':0}
    my_data.append(dict_sex[Sex])
    Age = st.number_input("请输入年龄", min_value=1, max_value=80, value=25)
    Age = (Age-1)/(80-1)
    my_data.append(Age)
    Fare = st.slider(
    "请选择船票价格",
    min_value=5,
    max_value=500,
    value=50,
    step=5
    )
    Fare=(Fare)/(512)
    my_data.append(Fare)
    City = st.selectbox(
    "请选择登船城市",
    ['南安普敦','瑟堡 - 奥克特维尔','昆士敦']
    )
    dict_city={'南安普敦':[1,0,0],'瑟堡 - 奥克特维尔':[0,1,0],'昆士敦':[0,0,1]}
    my_data.extend(dict_city[City])
    Size = st.slider(
    "请选择一同登船的家庭成员人数",
    min_value=1,
    max_value=10,
    value=2,
    step=1
    )
    my_data.append(Size+1)
    my_data=np.array(my_data)
    my_data=my_data[np.newaxis,:]
    my_data=my_data.tolist()
    # print(my_data)
    my_RF=load_RF()
    if my_RF.predict(my_data)[0]==1:
        st.write('大概率存活')
    else:
        st.write('大概率遇难')
    st.subheader('模型评估')
    st.write('测试集总体精确率为:0.79')
    st.write('预测遇难的样本中：精确率0.76 召回率0.92 F1率0.83')
    st.write('预测幸存的样本中：精确率0.85 召回率0.62 F1率0.71')
st.divider()

image_2=Image.open('image.png')

st.image(image_2)

