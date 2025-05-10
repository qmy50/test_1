import streamlit as st
from PIL import Image
import house as h
import python_kmean as k_Mean
from sklearn import datasets

@st.cache_resource
def load_model():
    iris = datasets.load_iris()
    X = iris.data  # 特征数据
    y = iris.target  # 真实标签
    feature_names = iris.feature_names
    target_names = iris.target_names
    data=X[:,0:2]
    return data

st.title('机器学习作业展示：')

if "step" not in st.session_state:
    st.session_state['step']=1

def goto_step(step_num:int):
    st.session_state['step']=step_num

if st.session_state.step==1:
    st.subheader('算法选择')
    st.button(label='梯度下降——房价问题',on_click=goto_step,args=((2,)))
    st.button(label='K聚类——鸢尾花分类',on_click=goto_step,args=((3,)))


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
          max_value=4,
          value=2,
          step=1)
    my_k=k_Mean.MyKmean(slider_val,data)
    if slider_val==3:
        st.balloons()
        acc,my_fig=my_k.showResult()
        acc=round(acc,2)
        st.write(f'预测准确度为：{acc}')
    else:
        my_fig=my_k.showResult()
    st.write('分类可视化如下:')
    st.pyplot(my_fig)
#肘部法则
    image_1=Image.open('image1.png')
    st.write('k取不同值时损失函数结果如下,可见取3时为肘部,符合肘部法则')
    st.image(image_1)


st.divider()

image_2=Image.open('image.png')

st.image(image_2)

