from streamlit_plotly_events import plotly_events
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from plotly.subplots import make_subplots

def get_date_range(df):
    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    return min_date, max_date

def sentiment_plot(fig, key, click_event=True):
    return plotly_events(fig, click_event=click_event, key=key)

def create_frequency_chart(df, window_length=7, polyorder=3, date_range=None):
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Group by date for total comments
    total_comments = df.groupby(df['date'].dt.date).size()

    # Apply Savitzky-Golay filter for smoothing
    # smoothed_comments = savgol_filter(total_comments.values, window_length, polyorder)
    smoothed_comments = total_comments.ewm(span=14).mean()

    # Create the line chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=total_comments.index,
        y=smoothed_comments,
        mode='lines',
        name='Total Comments',
        opacity=0.9,
        line=dict(color='black'),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Comments: %{y:.0f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=total_comments.index,
        y=total_comments,
        mode='lines',
        name='Total Comments',
        opacity=0.5,
        line=dict(color='#1f77b4', width=0.3),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Comments: %{y:.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='댓글 빈도 지표',
        title_x=0.5,
        title_y=0.98, title_yanchor='top',
        yaxis_title='빈도 (Log)',
        hovermode='x unified',
        margin=dict(t=20, l=50, r=50, b=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.5)"
        )
    )

    fig.update_xaxes(hoverformat="%Y-%m-%d")
    fig.update_yaxes(type='log')
    if date_range:
        fig.update_xaxes(range=date_range)

    return fig

def create_sentiment_chart(df, window_length=60, polyorder=3, date_range=None):
    df['date'] = pd.to_datetime(df['date'])
    grouped = df.groupby([df['date'].dt.date, 'predicted_label']).size().unstack(fill_value=0)

    for label in [0, 1, 2]:
        if label not in grouped.columns:
            grouped[label] = 0

    grouped = grouped.sort_index(axis=1)

    grouped_percentage = grouped.div(grouped.sum(axis=1), axis=0) * 100

    # Create the stacked area chart
    fig = go.Figure()

    smoothed_data = pd.DataFrame(index=grouped_percentage.index, columns=grouped_percentage.columns)
    for col in grouped_percentage.columns:
        y = grouped_percentage[col].values
        y_smooth = savgol_filter(y, window_length, polyorder)
        smoothed_data[col] = y_smooth

    colors = ['#1f77b4', '#c0c0c0', '#ff7f7f']  # Blue, Gray, Red for Positive, Neutral, Negative
    labels = ['Positive', 'Neutral', 'Negative']
    column_order = [2, 1, 0]  # Corresponding to Positive, Neutral, Negative

    for i, col in enumerate(column_order):
        fig.add_trace(go.Scatter(
            x=smoothed_data.index,
            y=smoothed_data[col],
            mode='lines',
            stackgroup='one',
            line=dict(width=0.5, color=colors[i]),
            name=f"{labels[i]}",
            hoverinfo='y',
            hovertemplate=f'{labels[i]}: ' + '%{y:.2f}%<extra></extra>'
        ))

    fig.update_layout(
        yaxis_title='비율',
        xaxis_title='날짜',
        hovermode='x unified',
        margin=dict(t=0, l=50, r=50, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.5)"
        )
    )

    fig.update_yaxes(
        range=[0, 100],
        ticksuffix='%'
    )
    fig.update_xaxes(
        hoverformat="%Y-%m-%d"
    )
    if date_range:
        fig.update_xaxes(range=date_range)

    # Add click event to return date and sentiment
    fig.update_layout(clickmode='event')

    return fig


