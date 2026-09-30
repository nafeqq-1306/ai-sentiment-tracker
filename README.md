# AI News Sentiment Tracker

## Overview
End-to-end NLP pipeline analyzing sentiment of AI industry news using multi-layer analysis:
sentiment scoring, topic modeling, and forecasting.

## Key Findings

**Dataset:** 481 articles (Sep 3-26, 2026)

**Sentiment Distribution:**
- 51.7% Positive
- 36.0% Negative  
- 12.3% Neutral
- Average sentiment: 0.11

**Company Analysis:**
- OpenAI: -0.03 avg sentiment (123 articles, most coverage but negative)
- Anthropic: +0.04 avg sentiment (83 articles)
- Google: +0.15 avg sentiment (26 articles)
- DeepMind: +0.24 avg sentiment (7 articles, most positive)

**Topic Analysis (LDA):**
5 latent topics discovered:
- Topic 0: OpenAI/ChatGPT specific incidents
- Topic 1: AI agents, government, regulation
- Topic 2: General AI industry news
- Topic 3: Machine learning/technical
- Topic 4: Google Gemini, control debates

**Sentiment by Topic:**
- Topic 4 (Google/Control): +0.0717 more positive
- Topic 1 (Regulation): -0.0514 more negative
- Topic 3 (ML/Technical): -0.0338 more negative

**Forecasting (Prophet):**
- 7-day forecast: Sentiment improving by +0.046
- Model MAE: 0.1963, RMSE: 0.2198

## Architecture


## Sentiment Analysis: VADER vs Custom Model

**VADER Performance:** 78.3% accuracy on test set
**Custom Model Performance:** 65.0% accuracy on test set

**Finding:** VADER works well on generic sentiment but systematically misclassifies AI security/safety incidents as positive.

**Example failures:**
- "OpenAI agents hacked government sites and leaked data" → VADER: POSITIVE | Actual: NEGATIVE
- "AI regulation passes, restricts training" → VADER: NEUTRAL | Actual: NEGATIVE

**Conclusion:** Generic sentiment models fail on domain-specific context. For AI news, VADER serves as baseline but requires manual review for accuracy.

## Technical Stack

- **Data Collection:** NewsAPI, requests
- **Sentiment:** VADER (nltk), Logistic Regression (scikit-learn)
- **Topic Modeling:** LDA (scikit-learn)
- **Forecasting:** Prophet (Facebook)
- **Database:** SQLite, SQLAlchemy
- **Dashboard:** Streamlit, Plotly
- **Analysis:** pandas, numpy

## How to Use

### Local Setup
```bash
git clone https://github.com/nafeqq-1306/ai-sentiment-tracker
cd ai-sentiment-tracker
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Run Pipeline
```bash
# 1. Scrape new articles
python scraper.py

# 2. Analyze sentiment
python sentiment.py

# 3. Topic modeling
python topic_modeling.py

# 4. Forecasting
python forecast_sentiment.py

# 5. View dashboard
streamlit run app.py
```

Visit: http://localhost:8501

## Key Insights

1. **AI news sentiment is polarized by topic, not company**
   - Regulation topics: -13% sentiment
   - Google control measures: +7% sentiment
   - Capability breakthroughs: +17% sentiment

2. **OpenAI gets most coverage but negative sentiment**
   - 123 articles (26% of dataset)
   - Driven by security incidents and regulation concerns
   - When discussing capabilities: neutral sentiment
   - When discussing incidents: -0.31 sentiment

3. **Safety concerns are double-edged**
   - 24 negative articles mention safety
   - 19 positive articles mention safety
   - Difference: How it's framed (problem vs solution)

4. **Sentiment is improving**
   - Current (last 7 days): 0.186
   - Forecast (next 7 days): 0.232
   - Positive news expected to outweigh negative

## Files

- `scraper.py` - NewsAPI data collection + filtering
- `sentiment.py` - VADER sentiment analysis
- `topic_modeling.py` - LDA topic discovery
- `forecast_sentiment.py` - Prophet time series forecasting
- `train_sentiment_model.py` - Custom model training
- `app.py` - Streamlit dashboard
- `database.py` - SQLAlchemy models
- `config.py` - Configuration

## What I Learned

- Quality data > quantity (filtering noise is 80% of the work)
- Domain-specific models outperform generic ones (VADER limitation)
- Topic modeling reveals context sentiment analysis misses
- Statistical rigor matters (labeled data, model evaluation)
- End-to-end projects are harder but more valuable than individual components