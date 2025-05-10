import streamlit as st
from PIL import Image


st.title('祝mom母亲节快乐')

if st.button("点击庆祝"):
    st.balloons()
    st.success("嘿嘿")


if "step" not in st.session_state:
    st.session_state['step']=1

def goto_step(step_num:int):
    st.session_state['step']=step_num

if st.session_state.step==1:
    st.subheader('戳这里↓')
    st.button(label='睁大',on_click=goto_step,args=((2,)))
    # image_2=Image.open('6.jpg')
    # image_2 = image_2.rotate(270, expand=True)
    # st.image(image_2,width=300)



if st.session_state.step==2:
    st.title('小礼物')
    image_1=Image.open('7.jpg')
    st.image(image_1)
    st.button(label='上一页',on_click=goto_step,args=((1,)))
    st.divider()

# image_2=Image.open('6.jpg')
# image_2 = image_2.rotate(270, expand=True)
# st.image(image_2,width=300)


