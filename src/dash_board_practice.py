import streamlit as st
import pandas as pd
import numpy as np
import random
import datetime
st.set_page_config(page_title="K-pop YouTube Analysis")
st.title('K-Pop YouTube Comment Analysis')

button_html = """
<button type='button' class='btn btn-primary'>
  <img src = img/AESPA.jpg>
</button>
"""
# Display the button with a custom icon
st.markdown(button_html, unsafe_allow_html=True)

# session state 선언
if 'select_idol' not in st.session_state:
    st.session_state['select_idol'] = 0 # 그룹 0에서 9까지 (축소하면 4까지)
if 'video_url' not in st.session_state:
    st.session_state['video_url'] = None

idol_group_container = st.container(border=True)
idol_button = ["에스파(aespa)", "ITZY", "LE SSERAFIM", "IVE", "NMIXX"]

st.subheader("아티스트")
for idx, (text, col) in enumerate(zip(idol_button, st.columns(len(idol_button)))):
    if col.button(text, use_container_width=True):
        st.session_state['select_idol'] = idx
        #st.write(st.session_state['select_idol'])

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
