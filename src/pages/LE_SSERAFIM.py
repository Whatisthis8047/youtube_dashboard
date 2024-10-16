import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TrendPop", initial_sidebar_state="collapsed")
st.title('K-Pop YouTube Comment Analysis')

# session state 선언
app_path = "http://localhost:8501"
page_path = "pages/"
if 'idol_group' not in st.session_state:
    st.session_state['idol_group'] = "newjeans"

idol_group_container = st.container(border=True)
idols = ["aespa", "itzy", "lesserafim", "ive", "nmixx"]
st.subheader("아티스트")
## 노가다
# https://yt3.googleusercontent.com/6CGtw27YRSZesaZSxp2fpT_KhH_Px_OOTWSq4igylqyibV1xpIZeazWBtepjhw1qeUkXDOVF=s176-c-k-c0x00ffffff-no-rj-mo
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"[![aespa](https://yt3.googleusercontent.com/6CGtw27YRSZesaZSxp2fpT_KhH_Px_OOTWSq4igylqyibV1xpIZeazWBtepjhw1qeUkXDOVF=s176-c-k-c0x00ffffff-no-rj-mo)]({app_path})")
with col2:
    st.markdown(f"[![itzy](https://yt3.googleusercontent.com/6jkoL_f5fcFSsIJwDBF7OWVXXWZ6NJp-n9UxeS3QY6vPClfLE_ZDpns20rhbZJ3spC8cUVoX=s160-c-k-c0x00ffffff-no-rj)]({app_path}/ITZY)")
with col3:
    st.markdown("![lesserafim](https://yt3.googleusercontent.com/4qrLd3kEQp0HKI5kKfewXXHxR_WxBAb7r2-Dp_V4ZHp9XVvLfDQ9OpBa3Nhr1lgkOQuodZzymyU=s160-c-k-c0x00ffffff-no-rj)")
with col4:
    st.markdown("![ive](https://yt3.googleusercontent.com/Fg5o4LNedtb4kLRjRZ2waWSG_xnAU-IvdO8_HyNGoxC7a1OPYwDFkxFLjpDmb35dPgdhkaYGoVE=s160-c-k-c0x00ffffff-no-rj)")
with col5:
    st.markdown("![nmixx](https://yt3.googleusercontent.com/jxMyaM3RVSxyma7n8wxBfi_IiMR0RmXXiqBHZyz08ELfyz5jh8Txd1Q-3ma_Zob9QI8v3fwo8g=s160-c-k-c0x00ffffff-no-rj)")
#st.markdown("<a href='ITZY' target='_self'>itzy</a>", unsafe_allow_html=True)

st.subheader("트렌드 분석")

chart_data = pd.read_csv("static/sample_label.csv")

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
