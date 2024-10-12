import streamlit as st

st.set_page_config(page_title="TrendPop", initial_sidebar_state="collapsed")
st.title('K-Pop YouTube Comment Analysis')

# session state 선언
# if 'select_idol' not in st.session_state:
#     st.session_state['select_idol'] = 0 # 그룹 0에서 9까지 (축소하면 4까지)
if 'video_url' not in st.session_state:
    st.session_state['video_url'] = None

idol_group_container = st.container(border=True)
idol_button = ["에스파(aespa)", "ITZY", "LE SSERAFIM", "IVE", "NMIXX"]

st.subheader("아티스트")
for idx, (text, col) in enumerate(zip(idol_button, st.columns(len(idol_button)))):
    if col.button(text, use_container_width=True):
        if text == "에스파(aespa)":
            st.switch_page("custom_app.py")
        else:
            st.switch_page(f"pages/{text}.py")
        # st.session_state['select_idol'] = idx
        # if st.session_state['selected_idol'] == idx:
        #     switch_page("ITZY")
        #st.write(st.session_state['select_idol'])