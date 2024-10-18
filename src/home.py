from functools import lru_cache
from utils import *
from carousel_component import *

st.set_page_config(page_title="TrendPop", initial_sidebar_state="collapsed",layout="wide") #layout="wide"

# session state 선언
if 'MV_url' not in st.session_state:
    st.session_state['MV_url'] = ""

if 'idol_group' not in st.session_state:
    st.session_state['idol_group'] = "ITZY"

if 'sentiment' not in st.session_state:
    st.session_state['sentiment'] = "None"

app_path = "http://localhost:8501"
selected_group = st.session_state['idol_group']
idols = ["itzy", "newjeans", "aespa", "lesserafim", "ive", "nmixx"]

conn = st.connection('trendpop_db', type='sql', url="mysql+pymysql://keonmo:mysql@localhost:3306/trendpop_db")

@lru_cache(maxsize=None)
def cached_query(query, ttl=600):
    return conn.query(query, ttl=ttl)

# Fetch all necessary data at once
def fetch_all_data(idol_group):
    queries = {
        'group_comments': f"SELECT * FROM comments WHERE `Group` = '{idol_group}';",
        'mv_thumbnail': f"SELECT * FROM mv_thumbnail WHERE `group` = '{idol_group}';",
        'member_thumbnail': f"SELECT * FROM member_thumbnail WHERE `Group` = '{idol_group}';"
    }
    return {key: cached_query(query) for key, query in queries.items()}

def main():
    #set page margin
    main_css = """
        <style>
            html, body, [class*="css"] {
                font-size: 20px;
            }
            .main > div {
                padding-left: 12%;
                padding-right: 12%;
            }
        </style>
    """
    st.markdown(main_css, unsafe_allow_html=True)

    st.title('🎩Trend  Pop')
    st.subheader("아티스트")
    artist_widget(0)
    st.divider()
    st.header(st.session_state['idol_group'])

    st.subheader("아티스트 감정 지표")
    group_df = conn.query(f"SELECT * FROM comments WHERE `Group` = '{st.session_state['idol_group']}';", ttl=600)

    fig = create_sentiment_chart(group_df)
    selected_points = sentiment_plot(fig, key="trend_analysis", click_event=True)

    st.subheader("Ask Gemini! 이슈 분석")
    st.chat_message("assistant").write(f"Gemini-1.5-Flash: Select date time points to ask about {selected_group}!")
    st.divider()

    if selected_points:
        point = selected_points[0]
        date, value, sentiment = point['x'], point['y'], point['curveNumber']
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        chatbot_response = get_chatbot_response(st.session_state['idol_group'], date, sentiment_map[sentiment], value)
        st.write(
            f"Showing 3 {sentiment_map[sentiment]} news links related to K-pop group '{st.session_state['idol_group']}' during {date}:")
        st.markdown(chatbot_response)

    st.subheader("뮤직비디오")
    mv_thumbnail = conn.query(f"SELECT * FROM mv_thumbnail WHERE `group` = '{st.session_state['idol_group']}';", ttl=600)
    mv_info = carousel_component(data=mv_thumbnail, layout='default', key='mv_carousel')

    if mv_info is None:
        st.write("확인할 뮤직비디오를 선택해 주세요")
    else:
        _, col2, _ = st.columns([1, 4, 1])
        col2.video(mv_info['link'])
        mv_df = conn.query(
            f"SELECT * FROM comments WHERE `Group` = '{st.session_state['idol_group']}' AND `Title` = '{mv_info['title']}';",
            ttl=600)
        if mv_df.empty:
            st.write(f"No data available for the selected music video: {mv_info['title']}")
        else:
            fig = create_sentiment_chart(mv_df)
            sentiment_plot(fig, key="mv_analysis", click_event=False)


    st.header("그룹/멤버 별 분석")
    # member_thumbnail = pd.DataFrame(
    #     {'group':[], 'member':[], 'thumbnail':[], 'keyword':[]}
    # )
    if mv_info is None:
        member_thumbnail = conn.query(
            f"SELECT * FROM member_thumbnail WHERE `Group` = '{st.session_state['idol_group']}';",
            ttl=600)
    else:
        member_thumbnail = conn.query(f"SELECT * FROM member_thumbnail WHERE `Group` = '{st.session_state['idol_group']}' AND `Title` = '{mv_info['title']}';", ttl=600)

    selected_member = carousel_component(data=member_thumbnail, layout='alternate', key="member_carousel")
    st.write(selected_member)

    st.markdown("**키워드**")
    st.markdown("**필터**")
    filter_col1, filter_col2 = st.columns([1,5])

    with filter_col1:
        st.radio(
            "필터",
            ["None", "Positive", "Neutral", "Negative"],
            key="sentiment",
            label_visibility="collapsed"
        )


if __name__ == '__main__':
    main()