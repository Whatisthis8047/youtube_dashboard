from streamlit_plotly_events import plotly_events
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def sentiment_plot(fig, key, click_event=False):
    return plotly_events(fig, click_event=click_event, key=key)

def create_sentiment_chart(df):
    df['date'] = pd.to_datetime(df['date'])
    sentiment_counts = df.groupby([df['date'].dt.date, 'sentiment_label']).size().unstack(fill_value=0)

    sentiment_counts.columns = ['Negative', 'Neutral', 'Positive']

    color_map = {
        'Negative': '#c0392b',  # Red
        'Neutral': '#5d6d7e',  # Grey
        'Positive': '#2980b9'  # Blue
    }

    # Create the line plot using Plotly
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for sentiment in ['Negative', 'Neutral', 'Positive']:
        fig.add_trace(
            go.Scatter(
                x=sentiment_counts.index,
                y=sentiment_counts[sentiment],
                mode='lines+markers',
                name=sentiment,
                line=dict(color=color_map[sentiment], width=2),
                marker=dict(color=color_map[sentiment], size=8),
                customdata=sentiment_counts[sentiment],
                hovertemplate="<b>Date</b>: %{x}<br><b>" + sentiment + "</b>: %{y}<extra></extra>"
            )
        )

    fig.update_layout(
        title= "Sentiment Chart",
        xaxis_title="Date",
        yaxis_title="Number of Comments",
        legend_title="Sentiment",
        hovermode="closest",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent surrounding paper
    )

    return fig