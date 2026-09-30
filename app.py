import streamlit as st
import pandas as pd
import numpy as np  # ADD THIS
from datetime import datetime, timedelta
from database import get_session, Article
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Sentiment Tracker", layout="wide")

st.title("🤖 AI News Sentiment Tracker")
st.markdown("Real-time sentiment analysis of AI industry news")

# Sidebar for controls
st.sidebar.header("Controls")
days_back = st.sidebar.slider("Show data from last N days:", 7, 90, 30)

# Fetch data
try:
    session = get_session()
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
    articles = session.query(Article).filter(Article.published_at >= cutoff_date).all()
    session.close()
    
    if not articles:
        st.error("No articles found. This app requires a local database.")
        st.info("To use this app:")
        st.write("1. Clone the repo: `git clone https://github.com/nafeqq-1306/ai-sentiment-tracker`")
        st.write("2. Run locally: `python scraper.py` then `streamlit run app.py`")
        st.stop()
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.info("This app is designed to run locally with a local database. See GitHub for setup instructions.")
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


# Company & Topic Analysis Section
st.divider()
st.subheader("🏢 Company Sentiment Analysis")

# Company analysis
companies = {
    "OpenAI": ["openai", "chatgpt", "gpt"],
    "Anthropic": ["anthropic", "claude"],
    "Google": ["google ai", "gemini", "bard"],
    "Meta": ["meta ai", "llama"],
    "DeepMind": ["deepmind"]
}

company_results = {}

for company, keywords in companies.items():
    company_articles = []
    
    for _, row in df.iterrows():
        text = (row["title"] + " " + (row.get("description") or "")).lower()
        if any(kw in text for kw in keywords):
            company_articles.append(row["sentiment_score"])
    
    if company_articles:
        avg_sentiment = sum(company_articles) / len(company_articles)
        positive = sum(1 for s in company_articles if s > 0.3)
        negative = sum(1 for s in company_articles if s < -0.3)
        
        company_results[company] = {
            "avg_sentiment": avg_sentiment,
            "articles": len(company_articles),
            "positive": positive,
            "negative": negative
        }

# Sort by sentiment
sorted_companies = sorted(company_results.items(), key=lambda x: x[1]["avg_sentiment"], reverse=True)

# Display as table
company_data = []
for company, data in sorted_companies:
    company_data.append({
        "Company": company,
        "Avg Sentiment": f"{data['avg_sentiment']:.2f}",
        "Total Articles": data["articles"],
        "Positive": data["positive"],
        "Negative": data["negative"]
    })

st.dataframe(company_data, use_container_width=True)

# Company sentiment chart
if company_results:
    company_names = [c[0] for c in sorted_companies]
    company_sentiments = [c[1]["avg_sentiment"] for c in sorted_companies]
    
    fig_company = px.bar(
        x=company_names,
        y=company_sentiments,
        title="Average Sentiment by Company",
        labels={"x": "Company", "y": "Average Sentiment Score"},
        color=company_sentiments,
        color_continuous_scale="RdYlGn"
    )
    st.plotly_chart(fig_company, use_container_width=True)

# Topic Analysis
st.subheader("📊 Topic Analysis")

topics = {
    "funding": ["funding", "investment", "raise", "series", "billion"],
    "security": ["security", "breach", "leak", "hack", "vulnerability"],
    "jobs": ["jobs", "employment", "workforce", "replaced", "displaced"],
    "regulation": ["regulation", "regulator", "government", "law", "policy"],
    "capability": ["breakthrough", "capability", "model", "performance", "beats"],
    "safety": ["safety", "concern", "risk", "dangerous", "alignment"],
    "competition": ["competitor", "competition", "vs", "battle", "race"]
}

# Count topics in positive vs negative articles
positive_topics = {}
negative_topics = {}

for topic, keywords in topics.items():
    pos_count = 0
    neg_count = 0
    
    for _, row in df.iterrows():
        text = (row["title"] + " " + (row.get("description") or "")).lower()
        if any(kw in text for kw in keywords):
            if row["sentiment_score"] > 0.3:
                pos_count += 1
            elif row["sentiment_score"] < -0.3:
                neg_count += 1
    
    positive_topics[topic] = pos_count
    negative_topics[topic] = neg_count

# Create comparison chart
topic_comparison = []
for topic in topics.keys():
    topic_comparison.append({
        "Topic": topic,
        "Positive": positive_topics[topic],
        "Negative": negative_topics[topic]
    })

topic_df = pd.DataFrame(topic_comparison)

fig_topics = px.bar(
    topic_df,
    x="Topic",
    y=["Positive", "Negative"],
    barmode="group",
    title="Topics in Positive vs Negative Articles",
    labels={"value": "Number of Articles"},
    color_discrete_map={"Positive": "green", "Negative": "red"}
)

st.plotly_chart(fig_topics, use_container_width=True)

# Show insights
st.subheader("💡 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("**Most Positive Topics:**")
    top_pos_topics = sorted(positive_topics.items(), key=lambda x: x[1], reverse=True)[:3]
    for topic, count in top_pos_topics:
        st.write(f"• {topic.capitalize()}: {count} articles")

with col2:
    st.write("**Most Negative Topics:**")
    top_neg_topics = sorted(negative_topics.items(), key=lambda x: x[1], reverse=True)[:3]
    for topic, count in top_neg_topics:
        st.write(f"• {topic.capitalize()}: {count} articles")

# Sentiment Forecasting Section
st.divider()
st.subheader("🔮 Sentiment Forecast (Next 7 Days)")

# Import forecast function
import sys
sys.path.append('.')

try:
    from forecast_sentiment import prepare_data_for_prophet, build_and_forecast
    
    # Prepare data and build model
    with st.spinner("Building forecast model..."):
        data = prepare_data_for_prophet()
        model, forecast = build_and_forecast(data, forecast_days=7)
    
    # Get forecast results
    last_date = data['ds'].max()
    future_forecast = forecast[forecast['ds'] > last_date].copy()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_avg = data['y'].iloc[-7:].mean()
        st.metric("Current Sentiment (7-day avg)", f"{current_avg:.3f}")
    
    with col2:
        forecasted_avg = future_forecast['yhat'].mean()
        st.metric("Forecasted Sentiment (7-day avg)", f"{forecasted_avg:.3f}")
    
    with col3:
        change = forecasted_avg - current_avg
        st.metric("Expected Change", f"{change:+.3f}", delta=f"{change:+.1%}")
    
    st.divider()
    
    # Display daily forecast
    st.subheader("Daily Forecast")
    
    forecast_display = []
    for idx, row in future_forecast.iterrows():
        date = row['ds'].strftime('%Y-%m-%d')
        forecast_val = row['yhat']
        lower = row['yhat_lower']
        upper = row['yhat_upper']
        
        if forecast_val > 0.2:
            sentiment = "POSITIVE 🟢"
        elif forecast_val < -0.2:
            sentiment = "NEGATIVE 🔴"
        else:
            sentiment = "NEUTRAL ⚪"
        
        forecast_display.append({
            "Date": date,
            "Sentiment": sentiment,
            "Forecast": f"{forecast_val:.3f}",
            "Lower Bound": f"{lower:.3f}",
            "Upper Bound": f"{upper:.3f}"
        })
    
    st.dataframe(pd.DataFrame(forecast_display), use_container_width=True)
    
    st.divider()
    
    # Plot forecast
    st.subheader("Forecast Visualization")
    
    fig = model.plot(forecast)
    plt.title('AI News Sentiment Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    st.pyplot(fig)
    
    # Interpretation
    st.subheader("📊 Forecast Interpretation")
    
    if forecasted_avg > current_avg:
        st.success(f"✓ Sentiment is expected to **IMPROVE** by {change:.3f}")
        st.write("This suggests positive AI news (breakthroughs, funding) will outweigh negative news (safety concerns, regulation).")
    elif forecasted_avg < current_avg:
        st.error(f"✗ Sentiment is expected to **DECLINE** by {abs(change):.3f}")
        st.write("This suggests negative AI news (security issues, regulation) will outweigh positive news.")
    else:
        st.info(f"↔ Sentiment expected to **STABILIZE**")
        st.write("No significant change expected in AI sentiment over the next week.")
    
except Exception as e:
    st.error(f"Could not generate forecast: {e}")
    st.write("Make sure you have run `python forecast_sentiment.py` first and have at least 10 days of data.")

   # Topic Modeling Section
st.divider()
st.subheader("🔍 Topic Analysis (LDA)")

try:
    from topic_modeling import prepare_documents, build_lda_model, get_document_topics, display_topics
    
    with st.spinner("Building topic model..."):
        documents, article_ids = prepare_documents()
        lda_model, vectorizer, doc_term_matrix = build_lda_model(documents, n_topics=5)
        doc_topic_dist = get_document_topics(lda_model, doc_term_matrix)
    
    # Display discovered topics
    st.subheader("📌 Discovered Topics")
    
    feature_names = vectorizer.get_feature_names_out()
    
    for topic_id, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-7:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        
        st.write(f"**Topic {topic_id}:** {', '.join(top_words)}")
    
    st.divider()
    
    # Company analysis by topic
    st.subheader("🏢 Companies by Topic")
    
    companies = {
        "OpenAI": ["openai", "chatgpt", "gpt"],
        "Anthropic": ["anthropic", "claude"],
        "Google": ["google", "gemini", "bard"],
        "Meta": ["meta", "llama"],
        "DeepMind": ["deepmind"],
        "Microsoft": ["microsoft", "copilot"],
    }
    
    n_topics = doc_topic_dist.shape[1]
    topic_company_data = []
    
    for topic_id in range(n_topics):
        company_counts = {company: 0 for company in companies}
        
        for idx, doc in enumerate(documents):
            doc_lower = doc.lower()
            dominant_topic = np.argmax(doc_topic_dist[idx])
            
            if dominant_topic == topic_id:
                for company, keywords in companies.items():
                    if any(kw in doc_lower for kw in keywords):
                        company_counts[company] += 1
        
        total = sum(company_counts.values())
        if total > 0:
            for company, count in company_counts.items():
                if count > 0:
                    topic_company_data.append({
                        "Topic": f"Topic {topic_id}",
                        "Company": company,
                        "Mentions": count,
                        "Percentage": f"{(count/total)*100:.1f}%"
                    })
    
    if topic_company_data:
        company_df = pd.DataFrame(topic_company_data)
        st.dataframe(company_df, use_container_width=True)
        
        # Visualization
        company_pivot = company_df.pivot_table(
            values='Mentions',
            index='Company',
            columns='Topic',
            fill_value=0
        )
        
        fig = go.Figure(data=[
            go.Bar(name=col, x=company_pivot.index, y=company_pivot[col])
            for col in company_pivot.columns
        ])
        fig.update_layout(
            title="Company Mentions by Topic",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Company sentiment by topic
    st.subheader("💬 Company Sentiment by Topic")
    
    sentiment_by_topic = []
    
    for topic_id in range(n_topics):
        for company, keywords in companies.items():
            sentiments = []
            
            for idx, doc in enumerate(documents):
                doc_lower = doc.lower()
                dominant_topic = np.argmax(doc_topic_dist[idx])
                
                if dominant_topic == topic_id and any(kw in doc_lower for kw in keywords):
                    article_id = article_ids[idx]
                    session = get_session()
                    article = session.query(Article).filter(Article.id == article_id).first()
                    session.close()
                    
                    if article and article.sentiment_score is not None:
                        sentiments.append(article.sentiment_score)
            
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                sentiment_by_topic.append({
                    "Topic": f"Topic {topic_id}",
                    "Company": company,
                    "Avg Sentiment": f"{avg_sentiment:.3f}",
                    "Count": len(sentiments)
                })
    
    if sentiment_by_topic:
        sentiment_df = pd.DataFrame(sentiment_by_topic)
        st.dataframe(sentiment_df, use_container_width=True)

except Exception as e:
    st.error(f"Could not generate topic analysis: {e}")
    st.write("Make sure topic_modeling.py has been run successfully.") 