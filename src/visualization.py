"""
src/visualization.py
Employee Feedback Intelligence Platform — Plotly chart helpers.
"""

from collections import Counter
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = {"Positive": "#22c55e", "Neutral": "#f59e0b", "Negative": "#ef4444"}


def sentiment_distribution_chart(
    df: pd.DataFrame, label_col: str = "sentiment_label"
) -> go.Figure:
    counts = df[label_col].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.bar(
        counts, x="sentiment", y="count",
        color="sentiment",
        color_discrete_map=PALETTE,
        title="Sentiment Distribution",
        labels={"sentiment": "", "count": "Count"},
    )
    fig.update_layout(showlegend=False)
    return fig


def word_frequency_chart(
    df: pd.DataFrame, text_col: str = "cleaned_text", top_n: int = 20
) -> go.Figure:
    all_words = [
        w for text in df[text_col].dropna()
        for w in str(text).split()
    ]
    top = Counter(all_words).most_common(top_n)
    words, freqs = zip(*top) if top else ([], [])
    fig = go.Figure(go.Bar(
        x=list(freqs)[::-1], y=list(words)[::-1],
        orientation="h", marker_color="#6366f1",
    ))
    fig.update_layout(title=f"Top {top_n} Words", xaxis_title="Frequency")
    return fig


def sentiment_trend_chart(
    df: pd.DataFrame,
    date_col: str = "date",
    label_col: str = "sentiment_label",
) -> go.Figure:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year_month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    trend = df.groupby(["year_month", label_col]).size().reset_index(name="count")
    fig = px.line(
        trend, x="year_month", y="count", color=label_col,
        color_discrete_map=PALETTE,
        title="Sentiment Trend Over Time",
        markers=True,
        labels={"year_month": "Month", "count": "Reviews", label_col: ""},
    )
    return fig


def department_sentiment_chart(
    df: pd.DataFrame,
    label_col: str = "sentiment_label",
    dept_col: str = "department",
) -> Optional[go.Figure]:
    if dept_col not in df.columns:
        return None
    pivot = (
        df.groupby([dept_col, label_col])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    fig = px.bar(
        pivot.melt(id_vars=dept_col, var_name="sentiment", value_name="count"),
        x=dept_col, y="count", color="sentiment",
        color_discrete_map=PALETTE,
        barmode="group",
        title="Sentiment by Department",
    )
    return fig


def metrics_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    metrics = ["accuracy", "precision", "recall", "f1"]
    available = [m for m in metrics if m in comparison_df.columns]
    fig = go.Figure()
    colors = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444"]
    for i, model in enumerate(comparison_df["model"]):
        row = comparison_df[comparison_df["model"] == model].iloc[0]
        fig.add_trace(go.Bar(
            name=model,
            x=available,
            y=[row[m] for m in available],
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        title="Model Performance Comparison",
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1.05]),
    )
    return fig


def confusion_matrix_heatmap(
    cm: List[List[int]],
    labels: List[str],
    title: str = "Confusion Matrix",
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig
