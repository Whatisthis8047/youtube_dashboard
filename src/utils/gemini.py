import streamlit as st
import google.generativeai as genai
import utils.config as config
from datetime import datetime, timedelta

# Configure the Gemini API
genai.configure(api_key=config.GOOGLE_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')


def create_prompt(group, date, sentiment, value):
    """
    Create a concise prompt for the Gemini chatbot based on the selected data point.
    """
    sentiment_word = "positive" if sentiment == "Positive" else "negative" if sentiment == "Negative" else "neutral"

    # Convert date string to datetime object
    date_obj = datetime.strptime(date, '%Y-%m-%d')

    # Create a date range (3 days before and after the selected date)
    start_date = (date_obj - timedelta(days=15)).strftime('%Y-%m-%d')
    end_date = (date_obj + timedelta(days=15)).strftime('%Y-%m-%d')

    prompt = f"""
    {start_date} 와 {end_date} 사이에 한국 아이돌 '{group}' 에 대해 {sentiment_word}한 뉴스 기사 세 개를 뽑아서 다음 마크다운 형식처럼 보여줘.:
    기사 링크의 기사의 제목과 내용은 일치 하도록 해. 정보가 없을 때는 정보가 없다고 이야기 해.
    1. 기사 링크]: [한 줄 요약]
    2.
    3.
    """

    return prompt

@st.cache_data
def get_chatbot_response(group, date, sentiment, value):
    """
    Generate a response from the Gemini chatbot based on the selected data point.
    """
    prompt = create_prompt(group, date, sentiment, value)

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"

