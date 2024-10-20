import pandas as pd
import streamlit as st
from functools import lru_cache

@lru_cache(maxsize=None)
def cached_query(conn, query, ttl=600):
    return conn.query(query, ttl=ttl)

def fetch_all_data(conn, idol_group):
    queries = {
        'group_comments': f"SELECT * FROM comments WHERE `Group` = '{idol_group}';",
        'mv_thumbnail': f"SELECT * FROM mv_thumbnail WHERE `group` = '{idol_group}';",
        'member_thumbnail': f"SELECT * FROM member_thumbnail WHERE `Group` = '{idol_group}';"
    }
    return {key: cached_query(conn, query) for key, query in queries.items()}


def filter_kpop_comments(df, member_filter="All", keyword_filter="없음", sentiment_filter="없음"):
    filtered_df = df.copy()

    if member_filter and member_filter != "All":
        filtered_df = filtered_df[filtered_df['members'].str.contains(member_filter, case=False, na=False)]

    if keyword_filter and keyword_filter != "#없음":
        filtered_df = filtered_df[filtered_df['word_list'].str.contains(keyword_filter, case=False, na=False)]

    sentiment_map = {"긍정": 2, "중립": 1, "부정": 0}
    if sentiment_filter and sentiment_filter != "없음" and sentiment_filter in sentiment_map:
        filtered_df = filtered_df[filtered_df['predicted_label'] == sentiment_map[sentiment_filter]]

    return filtered_df[['comment', 'likes', 'author', 'date']]