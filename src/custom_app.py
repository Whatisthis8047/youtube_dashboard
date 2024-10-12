import streamlit as st
# from streamlit_extras.switch_page_button import switch_page
import pandas as pd
import numpy as np
import random
import datetime
from PIL import Image


st.set_page_config(page_title="TrendPop", initial_sidebar_state="collapsed")
st.title('K-Pop YouTube Comment Analysis')

# session state 선언
if 'video_url' not in st.session_state:
    st.session_state['video_url'] = None

idol_group_container = st.container(border=True)
idols = ["aespa", "itzy", "lesserafim", "ive", "nmixx"]

img = st.image(Image.open("static/aespa.jpg"), use_column_width=True)
st.button(Image.open("static/itzy.jpg"))
st.subheader("아티스트")

for idx, (text, col) in enumerate(zip(idols, st.columns(len(idols)))):
    if col.button(text, use_container_width=True):
        st.write("hi")


#
st.subheader("트렌드 분석")

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["좋아요", "긍정", "부정"])

st.line_chart(chart_data)


st.subheader("이슈 분석")
# prompt = st.chat_input("Say something")
# if prompt:
#     st.write(f"User has sent the following prompt: {prompt}")


st.subheader("뮤직비디오")
mv_button = {'Black Mamba': "https://www.youtube.com/watch?v=ZeerrnuLi5E",
             'Next Level' : "https://www.youtube.com/watch?v=4TWR90KJl84",
             'Supernova': "https://www.youtube.com/watch?v=phuiiNCxRMg"}
for idx, (text, col) in enumerate(zip(mv_button.keys(), st.columns(len(mv_button)))):
    if col.button(text, use_container_width=True):
        st.session_state['video_url'] = mv_button[text]
        #st.write(st.session_state['select_idol'])
st.video(st.session_state['video_url'])

st.subheader("감성분석 예측")
chart_data2 = pd.DataFrame(np.random.randn(20, 3), columns=["좋아요", "긍정", "부정"])

st.line_chart(chart_data2)


st.subheader("멤버 별 분석")
