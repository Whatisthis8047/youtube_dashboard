import streamlit as st
import google.generativeai as genai
import utils.config as config
from datetime import datetime, timedelta

import os
import numpy as np
import pandas as pd
import json
import requests
import faiss
from datetime import datetime, timedelta

from langchain import LLMChain, PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS

import google.generativeai as genai
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

genai.configure(api_key=config.GOOGLE_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

os.environ["GOOGLE_API_KEY"] = config.GOOGLE_API_KEY
os.environ["OPENAI_API_KEY"] = config.OPEN_API_KEY

embeddings = OpenAIEmbeddings()
def similarity_search(news_db, query, k=3):

    # 데이터셋 크기보다 k가 클 경우, k를 데이터셋 크기로 설정
    if len(news_db) < k:
        k = len(news_db)

    # query를 임베딩
    #query_embedding = model.encode([query])
    query_embedding = np.array(embeddings.embed_query(query)).reshape(1, -1)  # 2D로 변환

    # news_db의 title을 임베딩
    #title_embeddings = model.encode(news_db['title'].tolist())
    title_embeddings = np.array(embeddings.embed_documents(news_db['title'].tolist()))

    # 코사인 유사도를 계산
    similarities = cosine_similarity(query_embedding, title_embeddings).flatten()

    # 유사도가 높은 k개의 인덱스를 선택
    top_k_indices = similarities.argsort()[-k:][::-1]

    # 해당 인덱스에 해당하는 행을 반환
    return news_db.reset_index().iloc[top_k_indices]

## 단순히 가장 가까운 사건을 리턴하는 버전
# 날짜 차이를 계산하는 함수
def calculate_date_diff(row, query_date):
    query_date = datetime.strptime(query_date, "%Y-%m-%d")
    event_date = datetime.strptime(row['date'], "%Y.%m.%d")
    return abs((query_date - event_date).days)

# 가장 가까운 사건 k개를 리턴하는 함수
def find_closest_events(news_db, group, date, k):
    # 입력된 날짜 변환
    query_date = datetime.strptime(date, "%Y-%m-%d")

    # 특정 그룹의 사건에 대해 날짜 차이를 계산하여 새로운 열에 추가
    df_filtered = news_db[news_db['group'] == group].copy()
    df_filtered['date_diff'] = df_filtered.apply(lambda row: calculate_date_diff(row, date), axis=1)

    # 날짜 차이가 적은 순으로 정렬 후 상위 k개 선택
    df_sorted = df_filtered.sort_values('date_diff').head(k)

    # 필요한 정보만 반환
    return df_sorted[['date', 'title', 'link']]
# 특정 그룹과 날짜 전후 15일 이내의 데이터프레임을 반환하는 함수
def filter_by_group_and_date(news_db, group, target_date):

    # 날짜 형식을 datetime으로 변환
    news_db['date_datetime'] = pd.to_datetime(news_db['date'], format="%Y.%m.%d")

    # 입력 받은 날짜를 datetime으로 변환
    try:
        # "Y.m.d" 형식 먼저 시도
        target_date_obj = datetime.strptime(target_date, "%Y.%m.%d")
    except ValueError:
        try:
            # "Y-m-d" 형식 시도
            target_date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            # 두 형식 모두 실패할 경우 오류 발생
            raise ValueError("Invalid date format. Please use 'YYYY.MM.DD' or 'YYYY-MM-DD'.")

    # 날짜 범위 계산 (target_date_obj 기준 ±15일)
    start_date = target_date_obj - timedelta(days=15)
    end_date = target_date_obj + timedelta(days=15)

    # 조건에 맞는 데이터 필터링
    filtered_df = news_db[
        (news_db['group'] == group) &
        (news_db['date_datetime'] >= start_date) &
        (news_db['date_datetime'] <= end_date)
    ]

    return filtered_df[['date', 'group', 'title', 'link']]

def create_query(group, date):

    # 문자열을 datetime 객체로 변환
    date_obj = datetime.strptime(date, '%Y-%m-%d')

    # 연도와 월 추출
    year = date_obj.year
    month = date_obj.month

    return f'{year}년 {month}월 {group} 관련 사건사고'

def generate_summary_for_article(llm, prompt_template, row, query):
    title = row['title']
    date = row['date']
    url = row['link']
    loader = UnstructuredURLLoader([url])
    data = loader.load()
    page_content = data[0].page_content

    summarize_chain = prompt_template | llm
    response = summarize_chain.invoke({"docs": page_content, "query": query})
    summary = response.content.strip()

    # 불필요한 "SUMMARY :" 부분 제거
    if "SUMMARY :" in summary:
        summary = summary.replace("SUMMARY :", "").strip()

    return f"{date}\n[{title}]\n{url}\n{summary}"


def summarizer(query, articles):

    # LLM 준비
    generation_config = {"온도": 0.7, "topP": 0.8, "topK": 40}
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", **generation_config)

    template = """
    {docs}

    너는 음악산업에 오래 종사해온 프로듀서야.
    KPOP 그룹에 대한 긍부정 감정에 영향을 미치는 여론 동향을 분석하기 위한 흥미롭고, 유용한 아티클을 찾고 있어.
    첨부한 기사를 바탕으로 보고서에 들어갈 요약문을 작성해줘.

    요약문은 아래의 가이드라인에 맞춰서 적어줘.

    1. 콘텐츠로부터 정보를 얻을 수 있고 유용해야 해.
    2. 콘텐츠는 읽기 쉽게 쓰여야 하고, 간결해야 해.
    3. 콘텐츠가 너무 짧거나 길지 않고 144자 정도를 유지해야 해.
    4. "{query}"에 관련된 내용을 잘 나타내고 있는 내용이어야 해.
    5. 긍정적이거나 부정적인 영향을 미칠 수 있을만한 내용이어야 해.

    읽기 편하게 '입니다'체의 한국어로 작성해 줘. 답변에 가이드라인의 내용이 포함되지 않도록 할 것.

    SUMMARY :
    """

    prompt_template = PromptTemplate(input_variables=["docs", "query"], template=template)

    # 각 기사별로 요약 생성
    summaries = [generate_summary_for_article(llm, prompt_template, row, query) for _, row in articles.iterrows()]

    # 각 요약을 합쳐 최종 결과로 반환
    return "\n\n\n".join(summaries)


@st.cache_data
def get_chatbot_response(group, date):
    """
    Generate a response from the Gemini chatbot based on the selected data point.
    """
    news_db = pd.read_csv("src/static/news_db.csv")
    try:
        query = create_query(group, date)

        filtered_df = filter_by_group_and_date(news_db, group, date)

        # 가장 가까운 사건과 기사 찾기
        # articles = find_closest_events(filtered_df, group, question_date)
        articles = similarity_search(filtered_df, query, k=3)

        # 요약답변 생성
        return summarizer(query, articles)


    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"
