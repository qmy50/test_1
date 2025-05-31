import pandas as pd
import streamlit as st

df = pd.DataFrame({
    "姓名": ["张三", "李四", "王五"],
    "年龄": [25, 30, 35],
    "分数": [85, 92, 78]
})

column = st.selectbox(
    "选择要显示的列",
    df.columns
)
st.write(df[column])