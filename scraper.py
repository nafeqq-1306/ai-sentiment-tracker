import requests
from datetime import datetime, timedelta
from config import NEWS_API_KEY, NEWS_API_URL, SEARCH_QUERIES, ARTICLES_PER_REQUEST, DAYS_BACK
from database import get_session, Article
import hashlib

def fetch_articles():
    """Fetch articles from NewsAPI for multiple search queries"""
    
    from config import DAYS_BACK
    
    today = datetime.utcnow()
    start_date = today - timedelta(days=DAYS_BACK)
    
    all_articles = []
    
    for query in SEARCH_QUERIES:
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": ARTICLES_PER_REQUEST,
            "apiKey": NEWS_API_KEY,
            "from": start_date.isoformat(),
            "to": today.isoformat(),
            "excludeDomains": "pypi.org,github.com,npmjs.com,reddit.com"  # Exclude noise sources
        }
        
        try:
            response = requests.get(NEWS_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] != "ok":
                print(f"API Error for '{query}': {data.get('message')}")
                continue
            
            articles = data.get("articles", [])
            all_articles.extend(articles)
            print(f"Fetched {len(articles)} articles for '{query}'")
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching articles for '{query}': {e}")
            continue
    
    print(f"\nTotal articles fetched: {len(all_articles)}")
    return all_articles

def is_ai_related(article):
    """Check if article is actually ABOUT AI, not just mentions it"""
    title = article.get("title", "") or ""
    description = article.get("description", "") or ""
    source = article.get("source", {}).get("name", "") or ""
    
    title = str(title).lower()
    description = str(description).lower()
    source = str(source).lower()
    
    # AI keywords that indicate article is ABOUT AI
    ai_keywords = [
        "ai", "artificial intelligence", "machine learning", "ml",
        "neural", "deep learning", "gpt", "chatgpt", "claude", "gemini",
        "llm", "large language model", "transformer", "bert",
        "openai", "anthropic", "google ai", "deepmind",
        "algorithm", "model training", "ai model", "ai safety",
        "ai regulation", "generative ai", "foundation model"
    ]
    
    # NOISE patterns to exclude (not real news)
    noise_patterns = [
        r"\d+\.\d+\.\d+",  # Version numbers (0.4.0, 1.2.3)
        r"added to pypi",
        r"released on pypi",
        r"pypi\.org",
        r"github\.com/.*release",
        r"npm package",
        r"library release",
        r"version bump",
        r"dependency update",
    ]
    
    # Blacklisted sources (mostly just version announcements)
    blacklist_sources = [
        "pypi.org",
        "npm",
        "nuget",
        "crates.io",
        "rubygems"
    ]
    
    # Skip if from blacklisted sources
    for blacklist in blacklist_sources:
        if blacklist in source:
            return False
    
    # Skip if matches noise patterns
    import re
    full_text = title + " " + description
    for pattern in noise_patterns:
        if re.search(pattern, full_text):
            return False
    
    # Count AI keywords
    text = title + " " + description
    ai_count = sum(1 for keyword in ai_keywords if keyword in text)
    
    # Require at least 2 AI keywords AND article must be substantial
    if ai_count < 2:
        return False
    
    # Exclude very short titles (likely just announcements)
    if len(title) < 30:
        return False
    
    # Article must be about AI, not just mention it in passing
    # Check that AI keywords are meaningful portion of text
    if ai_count >= 2:
        return True
    
    return False
def save_articles(articles):
    """Save articles to database, avoiding duplicates"""
    session = get_session()
    saved_count = 0
    duplicate_count = 0
    skipped_count = 0
    not_ai_count = 0
    
    for article in articles:
        # Skip articles with missing titles
        if not article.get("title"):
            skipped_count += 1
            continue
        
        # Skip if not actually about AI
        if not is_ai_related(article):
            not_ai_count += 1
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
    
    print(f"\nScraping Summary:")
    print(f"Saved {saved_count} new articles")
    print(f"  - {duplicate_count} duplicates skipped")
    print(f"  - {skipped_count} articles skipped (missing data)")
    print(f"  - {not_ai_count} articles skipped (not AI-related)")

def run_scraper():
    """Main scraping function"""
    print("Starting scraper...")
    print(f"Searching for articles from last {DAYS_BACK} days...\n")
    articles = fetch_articles()
    if articles:
        save_articles(articles)
    print("Scraper complete!")

if __name__ == "__main__":
    run_scraper()