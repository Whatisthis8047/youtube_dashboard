from streamlit_plotly_events import plotly_events
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data():
    df = pd.read_csv('static/sample_label.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

def sentiment_plot(fig, key, click_event=False):
    return plotly_events(fig, click_event=click_event, key=key)

def create_sentiment_chart(df, group, mv_title='None'):
    # Filter the dataframe for the selected group and title
    if mv_title == 'None':
        group_df = df[df['Group'] == group]
        title = f"Sentiment Timeline for {group}"
    else:
        st.write(mv_title)
        group_df = df[(df['Group'] == group) & (df['Title'] == mv_title)]
        title = f"Sentiment Timeline for {group} - {mv_title}"

    if group_df.empty:
        st.warning(f"No data found for group '{group}' and title '{mv_title}'")
        return None

    # Group by date and sentiment_label, and count occurrences
    sentiment_counts = group_df.groupby([group_df['date'].dt.date, 'sentiment_label']).size().unstack(fill_value=0)

    # Rename columns for clarity
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
        title=title,
        xaxis_title="Date",
        yaxis_title="Number of Comments",
        legend_title="Sentiment",
        hovermode="closest",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent surrounding paper
    )

    return fig