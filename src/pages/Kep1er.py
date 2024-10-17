import streamlit as st
import sys
sys.path.append('..')
from utils.widgets import artist_widget
from utils.plot import *
from utils.gemini import create_prompt, get_chatbot_response
from utils.widgets import *

st.set_page_config(page_title="TrendPop", initial_sidebar_state="collapsed")
st.title('K-Pop YouTube Comment Analysis')

# session state 선언
app_path = "http://localhost:8501"
page_path = "pages/"
st.session_state['idol_group'] = "Kep1er"
selected_group = st.session_state['idol_group']
conn = st.connection('trendpop_db', type='sql', url="mysql+pymysql://keonmo:mysql@localhost:3306/trendpop_db")

def main():
    st.subheader("아티스트")
    artist_widget(6)
    st.divider()
    st.header(st.session_state['idol_group'])

    st.subheader("아티스트 감정 지표")
    group_df = conn.query(f"SELECT * FROM sample_label WHERE `Group` = '{st.session_state['idol_group']}';", ttl=600)

    fig = create_sentiment_chart(group_df)
    selected_points = sentiment_plot(fig, key="trend_analysis", click_event=True)

    st.subheader("Ask Gemini! 이슈 분석")
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
    mv_thumbnail = conn.query(f"SELECT * FROM thumbnail WHERE `group` = '{st.session_state['idol_group']}';", ttl=600)
    #set column name
    mv_title = mv_widget(mv_thumbnail, st.session_state['idol_group'])

    st.subheader("뮤직비디오 감정 지표")
    mv_df = conn.query(f"SELECT * FROM sample_label WHERE `Group` = '{st.session_state['idol_group']}' AND `Title` = '{mv_title}';", ttl=600)
    fig = create_sentiment_chart(mv_df)
    sentiment_plot(fig, key="mv_analysis", click_event=False)

    st.header("멤버 별 분석")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**멤버/그룹 선택**")

    with col2:
        st.markdown("**키워드**")

        st.markdown("**필터**")

        st.radio(
            "필터",
            ["None", "Positive", "Neutral", "Negative"],
            key="sentiment",
            label_visibility="collapsed"
        )


if __name__ == '__main__':
    main()