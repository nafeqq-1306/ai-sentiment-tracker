import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Database Configuration
DATABASE_URL = "sqlite:///data/articles.db"

# Scraping Configuration - Multiple search queries for better coverage
SEARCH_QUERIES = [
    # Original 12
    "OpenAI", "Anthropic Claude", "Google AI Gemini", "ChatGPT",
    "machine learning", "artificial intelligence", "deep learning",
    "neural networks", "LLM large language model", "AI safety",
    "AI regulation", "transformer model",
    
    # New additions
    "Claude AI", "Gemini AI", "AI jobs", "AI ethics",
    "AI startup", "AI investment", "AI chip", "GPU TPU",
    "generative AI", "foundation model", "prompt engineering",
    "fine tuning", "AI model training", "large language model debate",
    "AI copyright", "AI bias", "AI transparency", "responsible AI",
    "AGI artificial general intelligence", "AI risk", "AI safety research",
]

ARTICLES_PER_REQUEST = 100
DAYS_BACK = 30  # Scrape last 30 days