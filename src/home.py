import streamlit as st
# from streamlit_extras.switch_page_button import switch_page
from utils.plot import *
from utils.gemini import create_prompt, get_chatbot_response
from utils.widgets import *
import pandas as pd
import numpy as np
import random
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="TrendPop", initial_sidebar_state="collapsed") #layout="wide"
st.title('K-Pop YouTube Comment Analysis')

# session state 선언
if 'MV_url' not in st.session_state:
    st.session_state['MV_url'] = ""

if 'idol_group' not in st.session_state:
    st.session_state['idol_group'] = "ITZY"

app_path = "http://localhost:8501"
selected_group = st.session_state['idol_group']
idols = ["itzy", "newjeans", "aespa", "lesserafim", "ive", "nmixx"]

conn = st.connection('trendpop_db', type='sql', url="mysql+pymysql://keonmo:mysql@localhost:3306/trendpop_db")

def main():
    st.subheader("아티스트")
    artist_widget(0)
    st.divider()
    st.header(st.session_state['idol_group'])

    st.subheader("트렌드 분석")
    group_df = conn.query(f"SELECT * FROM sample_label WHERE `Group` = '{st.session_state['idol_group']}';", ttl=600)
    group_df['date'] = pd.to_datetime(group_df['date'])

    fig = create_sentiment_chart(group_df, selected_group)
    selected_points = sentiment_plot(fig, key="trend_analysis", click_event=True)

    st.subheader("이슈 분석")
    st.chat_message("assistant").write(f"Gemini-1.5-Flash: Select date time points to ask about {selected_group}!")
    st.divider()

    if selected_points:
        point = selected_points[0]
        date = point['x']
        value = point['y']
        sentiment = point['curveNumber']  # 0: Negative, 1: Neutral, 2: Positive
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

        chatbot_response = get_chatbot_response(selected_group, date, sentiment_map[sentiment], value)
        st.write(f"Showing 3 {sentiment_map[sentiment]} news links related to K-pop group '{selected_group}' during {date}:")
        st.markdown(chatbot_response)

    #st.plotly_chart(fig, use_container_width=True)

    # prompt = st.chat_input("Say something")
    # if prompt:
    #     st.write(f"User has sent the following prompt: {prompt}")

    st.subheader("뮤직비디오")
    mv_df = conn.query(f"SELECT * FROM thumbnail WHERE `group` = '{st.session_state['idol_group']}';", ttl=600)
    #set column name
    mv_title = mv_widget(mv_df, st.session_state['idol_group'])

    # st.subheader("감성분석 예측")
    # fig = create_sentiment_chart(df, selected_group, mv_title)
    # sentiment_plot(fig, key="mv_analysis", click_event=False)


    st.subheader("멤버 별 분석")

if __name__ == '__main__':
    main()