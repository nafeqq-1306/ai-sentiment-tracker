import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from database import get_session, Article
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="AI Sentiment Tracker", layout="wide")

st.title("🤖 AI News Sentiment Tracker")
st.markdown("Real-time sentiment analysis of AI industry news")

# Sidebar for controls
st.sidebar.header("Controls")
days_back = st.sidebar.slider("Show data from last N days:", 7, 90, 30)

# Fetch data
session = get_session()
cutoff_date = datetime.utcnow() - timedelta(days=days_back)
articles = session.query(Article).filter(Article.published_at >= cutoff_date).all()
session.close()

if not articles:
    st.warning("No articles found. Run the scraper first!")
    st.stop()

# Convert to DataFrame for analysis
df = pd.DataFrame([
    {
        "title": a.title,
        "source": a.source,
        "published_at": a.published_at,
        "sentiment_score": a.sentiment_score or 0,
        "sentiment_label": a.sentiment_label or "NEUTRAL",
        "url": a.url
    }
    for a in articles
])

# Dashboard metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Articles", len(df))

with col2:
    positive = len(df[df["sentiment_label"] == "POSITIVE"])
    st.metric("Positive", positive)

with col3:
    negative = len(df[df["sentiment_label"] == "NEGATIVE"])
    st.metric("Negative", negative)

with col4:
    avg_sentiment = df["sentiment_score"].mean()
    st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")

st.divider()

# Sentiment over time
st.subheader("Sentiment Trend Over Time")

daily_sentiment = df.groupby(df["published_at"].dt.date).agg({
    "sentiment_score": "mean",
    "title": "count"
}).rename(columns={"title": "article_count"})

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=daily_sentiment.index,
    y=daily_sentiment["sentiment_score"],
    mode="lines+markers",
    name="Avg Sentiment",
    line=dict(color="blue", width=2)
))
fig_trend.update_layout(
    xaxis_title="Date",
    yaxis_title="Average Sentiment Score",
    hovermode="x unified",
    height=400
)
st.plotly_chart(fig_trend, use_container_width=True)

# Distribution
st.subheader("Sentiment Distribution")

col1, col2 = st.columns(2)

with col1:
    sentiment_counts = df["sentiment_label"].value_counts()
    fig_dist = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color_discrete_map={"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"}
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    source_counts = df["source"].value_counts().head(10)
    fig_sources = px.bar(
        x=source_counts.values,
        y=source_counts.index,
        orientation="h",
        title="Top 10 Sources"
    )
    st.plotly_chart(fig_sources, use_container_width=True)

# Recent articles
st.subheader("Recent Articles")

df_sorted = df.sort_values("published_at", ascending=False)

for _, row in df_sorted.head(10).iterrows():
    sentiment_emoji = "🟢" if row["sentiment_label"] == "POSITIVE" else "🔴" if row["sentiment_label"] == "NEGATIVE" else "⚪"
    
    with st.container():
        st.markdown(f"{sentiment_emoji} **{row['title']}**")
        st.markdown(f"*{row['source']}* | {row['published_at'].strftime('%Y-%m-%d %H:%M')}")
        st.markdown(f"Score: {row['sentiment_score']:.2f}")
        st.markdown(f"[Read →]({row['url']})")
        st.divider()


# Detailed article browser
st.divider()
st.subheader("📚 Article Explorer")

# Filter options
col1, col2, col3 = st.columns(3)

with col1:
    sentiment_filter = st.multiselect(
        "Filter by sentiment:",
        ["POSITIVE", "NEGATIVE", "NEUTRAL"],
        default=["POSITIVE", "NEGATIVE", "NEUTRAL"]
    )

with col2:
    source_filter = st.multiselect(
        "Filter by source:",
        df["source"].unique(),
        default=df["source"].unique()[:5]  # Show top 5 by default
    )

with col3:
    score_range = st.slider(
        "Sentiment score range:",
        -1.0, 1.0, (-1.0, 1.0)
    )

# Apply filters
filtered_df = df[
    (df["sentiment_label"].isin(sentiment_filter)) &
    (df["source"].isin(source_filter)) &
    (df["sentiment_score"] >= score_range[0]) &
    (df["sentiment_score"] <= score_range[1])
]

# Show filtered count
st.write(f"**Showing {len(filtered_df)} of {len(df)} articles**")

# Display all filtered articles in a table
st.dataframe(
    filtered_df[[
        "title", "source", "published_at", 
        "sentiment_score", "sentiment_label"
    ]].sort_values("published_at", ascending=False),
    use_container_width=True,
    height=500
)

# Option to export
if st.button("📥 Show all details (expandable)"):
    for idx, row in filtered_df.iterrows():
        with st.expander(f"📄 {row['title'][:60]}... | {row['source']}"):
            st.write(f"**Sentiment:** {row['sentiment_label']} ({row['sentiment_score']:.2f})")
            st.write(f"**Published:** {row['published_at']}")
            st.write(f"**Source:** {row['source']}")
            st.markdown(f"[Read full article →]({row['url']})")