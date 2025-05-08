import streamlit as st
from PIL import Image

st.write('机器学习作业展示：')

if "step" not in st.session_state:
    st.session_state['step']=1

def goto_step(step_num:int):
    st.session_state['step']=step_num

if st.session_state.step==1:
    st.title('算法选择')
    st.button(label='梯度下降——房价问题',on_click=goto_step,args=((2,)))
    st.button(label='SVM——鸢尾花分类',on_click=goto_step,args=((3,)))


if st.session_state.step==2:
    st.title('梯度下降')
    st.button(label='Go prev',on_click=goto_step,args=((1,)))
    example="""iris = datasets.load_iris()
x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target) """


    st.code(example,language='python')

    st.divider()

    slider_val=st.slider(label='请输入房屋面积：',
          min_value=50,
          max_value=150,
          value=0,
          step=1)
    ans=2*slider_val+3

    st.write("预测房价结果为：",'%.2f' % ans)


if st.session_state.step==3:
    st.title('支持向量机')
    st.button(label='Go prev',on_click=goto_step,args=((1,)))


# example="""iris = datasets.load_iris()
# x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target) """


# st.code(example,language='python')

# st.divider()

# slider_val=st.slider(label='请输入房屋面积：',
#           min_value=50,
#           max_value=150,
#           value=0,
#           step=1)
# ans=h.my_k*slider_val+h.my_b

# st.write("预测房价结果为：",'%.2f' % ans)



