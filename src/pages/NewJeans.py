import streamlit as st
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from utils.widgets import artist_widget
from utils.plot import *
from utils.gemini import create_prompt, get_chatbot_response
from utils.widgets import artist_widget

st.set_page_config(page_title="TrendPop", initial_sidebar_state="collapsed",layout="wide")
st.title('K-Pop YouTube Comment Analysis')

# session state 선언
app_path = "http://localhost:8501"
page_path = "pages/"
st.session_state['idol_group'] = "NewJeans"

idol_group_container = st.container(border=True)

def main():
    st.subheader("아티스트")
    artist_widget(1)

    st.subheader("트렌드 분석")

    df = load_data()
    group_list = sorted(df['Group'].unique())

    fig = create_sentiment_chart(df, st.session_state['idol_group'])
    selected_points = sentiment_plot(fig, click_event=True)

    st.subheader("이슈 분석")
    st.chat_message("assistant").write("Select date time points to ask Gemini-Pro!")

    if selected_points:
        point = selected_points[0]
        date = point['x']
        value = point['y']
        sentiment = point['curveNumber']  # 0: Negative, 1: Neutral, 2: Positive
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

        with st.spinner("Retrieveing News articles..."):
            chatbot_response = get_chatbot_response(selected_group, date, sentiment_map[sentiment], value)
        st.write(f"Showing 3 {sentiment_map[sentiment]} news links related to K-pop group '{selected_group}' during {date}:")
        st.markdown(chatbot_response)

    #st.plotly_chart(fig, use_container_width=True)


    # prompt = st.chat_input("Say something")
    # if prompt:
    #     st.write(f"User has sent the following prompt: {prompt}")


    st.subheader("뮤직비디오")
    mv_button = {'Black Mamba': "https://www.youtube.com/watch?v=ZeerrnuLi5E",
                 'Next Level' : "https://www.youtube.com/watch?v=4TWR90KJl84",
                 'Supernova': "https://www.youtube.com/watch?v=phuiiNCxRMg"}
    for idx, (text, col) in enumerate(zip(mv_button.keys(), st.columns(len(mv_button)))):
        if col.button(text, use_container_width=True):
            st.session_state['mv_url'] = mv_button[text]

    st.subheader("감성분석 예측")
    chart_data2 = pd.DataFrame(np.random.randn(20, 3), columns=["좋아요", "긍정", "부정"])

    st.line_chart(chart_data2)


    st.subheader("멤버 별 분석")

if __name__ == '__main__':
    main()