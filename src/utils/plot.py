from streamlit_plotly_events import plotly_events
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def sentiment_plot(fig, key, click_event=False):
    return plotly_events(fig, click_event=click_event, key=key)


def create_sentiment_chart(df):
    df['date'] = pd.to_datetime(df['date'])
    sentiment_counts = df.groupby([df['date'].dt.date, 'sentiment_label']).size().unstack(fill_value=0)

    sentiment_counts.columns = ['Negative', 'Neutral', 'Positive']
    sentiment_counts = sentiment_counts.reset_index()
    sentiment_counts['date'] = pd.to_datetime(sentiment_counts['date'])

    color_map = {
        'Negative': '#c0392b',  # Red
        'Neutral': '#5d6d7e',  # Grey
        'Positive': '#2980b9'  # Blue
    }

    fig = go.Figure()

    for sentiment in ['Negative', 'Neutral', 'Positive']:
        fig.add_trace(go.Scatter(
            x=sentiment_counts['date'],
            y=sentiment_counts[sentiment],
            mode='lines',
            line=dict(width=0.5, color=color_map[sentiment]),
            stackgroup='one',
            groupnorm='percent',  # sets the normalization for the sum of the stackgroup
            name=sentiment
        ))

    fig.update_layout(
        title="Normalized Stacked Area Sentiment Chart",
        xaxis_title="Date",
        yaxis_title="Percentage",
        legend_title="Sentiment",
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent surrounding paper
    )

    fig.update_yaxes(
        range=[0, 100],
        ticksuffix='%'
    )

    return fig