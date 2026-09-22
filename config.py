import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Database Configuration
DATABASE_URL = "sqlite:///data/articles.db"

# Scraping Configuration
SEARCH_QUERY = "AI OR artificial intelligence OR machine learning"
ARTICLES_PER_REQUEST = 100