# 🤖 AI Sentiment Tracker

Real-time sentiment analysis of AI industry news using web scraping, NLP, and data visualization.

## Overview

This project scrapes AI-related news articles from multiple sources, analyzes their sentiment using transformer models, and visualizes trends through an interactive dashboard.

**Live Demo:** (You'll add this after deploying to Streamlit Cloud)

## Features

- **Web Scraping**: Fetches articles from NewsAPI (100+ articles/day)
- **NLP Sentiment Analysis**: Uses DistilBERT for sentiment classification
- **Data Pipeline**: SQLite database for persistent storage
- **Interactive Dashboard**: Streamlit-based visualization with:
  - Sentiment trends over time
  - Sentiment distribution (positive/negative/neutral)
  - Top news sources
  - Recent articles with sentiment scores

## Tech Stack

- **Backend**: Python, SQLAlchemy, SQLite
- **NLP**: Hugging Face Transformers (DistilBERT)
- **Frontend**: Streamlit
- **Data**: Pandas, Plotly
- **API**: NewsAPI

## Setup

### Prerequisites
- Python 3.8+
- NewsAPI key (free at [newsapi.org](https://newsapi.org))

### Installation

1. Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/ai-sentiment-tracker.git
cd ai-sentiment-tracker
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add your NewsAPI key to `.env`: