from scraper import fetch_articles, save_articles
from sentiment import update_sentiment_scores
from alerting import detect_changes_and_alert

def run_full_pipeline():
    """Run scraper → sentiment → alerting"""
    print("Running full pipeline...")
    
    # 1. Scrape
    print("Step 1: Scraping...")
    articles = fetch_articles()
    save_articles(articles)
    
    # 2. Analyze sentiment
    print("Step 2: Analyzing sentiment...")
    update_sentiment_scores()
    
    # 3. Check for alerts
    print("Step 3: Checking for alerts...")
    detect_changes_and_alert()
    
    print("✓ Pipeline complete")

if __name__ == "__main__":
    run_full_pipeline()