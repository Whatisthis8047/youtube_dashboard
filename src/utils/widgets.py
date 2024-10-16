import streamlit as st
import pandas as pd
import sys, os

app_path = "http://localhost:8501"


def artist_widget(ord:int):
    """
    아이돌 그룹 로고 위젯
    :param ord: 0부터 9까지의 탑 10 아이돌 순서(정수).
    :return:
    """
    col0, col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(10)
    columns = [col0, col1, col2, col3, col4, col5, col6, col7, col8, col9]

    for i, col in enumerate(columns):
        bg_color = "#808080" if i == ord else "transparent"
        with col:
            if i == 0:
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                        <a target="_self" href="{app_path}">
                            <img src="https://yt3.googleusercontent.com/6jkoL_f5fcFSsIJwDBF7OWVXXWZ6NJp-n9UxeS3QY6vPClfLE_ZDpns20rhbZJ3spC8cUVoX=s160-c-k-c0x00ffffff-no-rj" alt="ITZY" width="100%">
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif i == 1:
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                        <a target="_self" href="{app_path}/NewJeans">
                            <img src="https://yt3.googleusercontent.com/LeP3KhMTfh_dq8a41PhcRbln9s_Aa1MUKAQRLt3XBXNo2M0m1nPBnU9ZLeIjiDfLAF5m-hfKuQ=s160-c-k-c0x00ffffff-no-rj" alt="NewJeans" width="100%">
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif i == 2:
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                        <a target="_self" href="{app_path}/aespa">
                            <img src="https://yt3.googleusercontent.com/6CGtw27YRSZesaZSxp2fpT_KhH_Px_OOTWSq4igylqyibV1xpIZeazWBtepjhw1qeUkXDOVF=s176-c-k-c0x00ffffff-no-rj-mo" alt="aespa" width="100%">
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif i == 3:
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                        <a target="_self" href="{app_path}">
                            <img src="https://yt3.googleusercontent.com/4qrLd3kEQp0HKI5kKfewXXHxR_WxBAb7r2-Dp_V4ZHp9XVvLfDQ9OpBa3Nhr1lgkOQuodZzymyU=s160-c-k-c0x00ffffff-no-rj" alt="LE SSERAFIM" width="100%">
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif i == 4:
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                        <a target="_self" href="{app_path}">
                            <img src="https://yt3.googleusercontent.com/Fg5o4LNedtb4kLRjRZ2waWSG_xnAU-IvdO8_HyNGoxC7a1OPYwDFkxFLjpDmb35dPgdhkaYGoVE=s160-c-k-c0x00ffffff-no-rj" alt="IVE" width="100%">
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif i == 5:
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                        <a target="_self" href="{app_path}">
                            <img src="https://yt3.googleusercontent.com/jxMyaM3RVSxyma7n8wxBfi_IiMR0RmXXiqBHZyz08ELfyz5jh8Txd1Q-3ma_Zob9QI8v3fwo8g=s160-c-k-c0x00ffffff-no-rj" alt="NMIXX" width="100%">
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif i == 6:
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                        <a target="_self" href="{app_path}">
                            <img src="https://yt3.googleusercontent.com/e5JEZjIHLs669zNv6sVQtF84QGppJ_2NMoBldOr6OfOBN-vJt5eS9zGvIe5oUGcoeE_JZS8sLZ4=s160-c-k-c0x00ffffff-no-rj" alt="tripleS" width="100%">
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif i == 7:
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                        <a target="_self" href="{app_path}">
                            <img src="https://yt3.googleusercontent.com/_nhXya-TB7QYEVPZ8CWxCuR75lcKGp6cihYIzwsrqAcbjpNHwlabKc4okTKTpDVh-GcA7IWN=s160-c-k-c0x00ffffff-no-rj" alt="Kep1er" width="100%">
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif i == 8:
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                        <a target="_self" href="{app_path}">
                            <img src="https://yt3.googleusercontent.com/3WVx4yHwddjqDmVHZtGl1IxCEFDs5O74KA9cCmOTPqG-y2l53zTavQzVkQEdYw-tqWYF_PgB=s160-c-k-c0x00ffffff-no-rj" alt="FIFTY FIFTY" width="100%">
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif i == 9:
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                        <a target="_self" href="{app_path}">
                            <img src="https://yt3.googleusercontent.com/CaeNdw8_MAi96qW5L3bMi-24tSldENUypIoJhjoF1hNFRRDX1U5uqpIbl14qYSQUUxkVJSxWOg=s160-c-k-c0x00ffffff-no-rj" alt="STAYC" width="100%">
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def mv_widget(df, group):
    #df = pd.read_csv('src/static/thumbnail.csv')
    df = df.rename(columns={'C1': 'group', 'C2': 'title', 'C3': 'link', 'C4': 'thumbnail'})

    # Filter the dataframe by group if specified
    if df.empty:
        st.error(f"No found for group")
        return 'None'

    # Set default video to the first one in the filtered dataframe
    default_video = df.iloc[0]

    if 'selected_video_title' not in st.session_state:
        st.session_state.selected_video_title = default_video['title']
        st.session_state.selected_video_url = default_video['link']

    # CSS to make thumbnails clickable and adjust text size
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: rgb(255, 255, 255, 0);
        border: none;
        color: rgb(255, 255, 255, 0);
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        z-index: 1;
    }
    div.stButton > button:hover {
        border: 3px solid rgb(255, 75, 75);
        color: rgb(255, 255, 255, 0);
    }
    .video-caption {
        font-size: 0.8em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
        text-align: center;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create a grid layout
    cols = st.columns(4)

    # Variable to store the selected video URL
    selected_video_url = None
    selected_video_title = None

    # Display thumbnails in a grid
    for index, row in df.iterrows():
        col = cols[index % 4]
        with col:
            # Container for thumbnail and button
            with st.container():
                st.image(row['thumbnail'], use_column_width=True)
                if st.button("", key=f"btn_{index}"):
                    st.session_state.selected_video_title = row['title']
                    st.session_state.selected_video_url = row['link']
                st.markdown(f"<div class='video-caption'>{row['title']}</div>", unsafe_allow_html=True)

    # Display the selected video
    st.subheader(f"Now Playing: {st.session_state.selected_video_title}")
    st.video(st.session_state.selected_video_url)
    # else:
    #     st.subheader(f"Now Playing: {}")
    #     st.video(selected_video_url)

    return st.session_state.selected_video_title