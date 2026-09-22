import requests
from datetime import datetime, timedelta
from config import NEWS_API_KEY, NEWS_API_URL, SEARCH_QUERY, ARTICLES_PER_REQUEST
from database import get_session, Article
import hashlib

def fetch_articles():
    """Fetch articles from NewsAPI for the last 7 days"""
    
    # Calculate date range (last 7 days)
    today = datetime.utcnow()
    week_ago = today - timedelta(days=7)
    
    params = {
        "q": SEARCH_QUERY,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": ARTICLES_PER_REQUEST,
        "apiKey": NEWS_API_KEY,
        "from": week_ago.isoformat(),
        "to": today.isoformat()
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] != "ok":
            print(f"API Error: {data.get('message')}")
            return []
        
        articles = data.get("articles", [])
        print(f"Fetched {len(articles)} articles")
        return articles
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching articles: {e}")
        return []

def save_articles(articles):
    """Save articles to database, avoiding duplicates"""
    session = get_session()
    saved_count = 0
    duplicate_count = 0
    skipped_count = 0
    
    for article in articles:
        # Skip articles with missing titles
        if not article.get("title"):
            skipped_count += 1
            continue
        
        # Create unique ID from URL
        article_id = hashlib.md5(article["url"].encode()).hexdigest()
        
        # Check if already exists
        existing = session.query(Article).filter_by(id=article_id).first()
        if existing:
            duplicate_count += 1
            continue
        
        # Parse date
        try:
            published_at = datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00"))
        except:
            published_at = datetime.utcnow()
        
        # Create new article
        new_article = Article(
            id=article_id,
            title=article.get("title", ""),
            description=article.get("description", ""),
            url=article.get("url", ""),
            source=article.get("source", {}).get("name", "Unknown"),
            published_at=published_at,
            content=article.get("content", "")
        )
        
        session.add(new_article)
        saved_count += 1
    
    session.commit()
    session.close()
    
    print(f"Saved {saved_count} new articles, {duplicate_count} duplicates skipped, {skipped_count} articles skipped (missing data)")


def run_scraper():
    """Main scraping function"""
    print("Starting scraper...")
    articles = fetch_articles()
    if articles:
        save_articles(articles)
    print("Scraper complete!")

if __name__ == "__main__":
    run_scraper()