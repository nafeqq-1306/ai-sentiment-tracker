from database import get_session, Article
from collections import Counter
import re

def extract_topics():
    """Extract key topics from articles by sentiment"""
    session = get_session()
    
    positive_articles = session.query(Article).filter(
        Article.sentiment_score > 0.3
    ).all()
    
    negative_articles = session.query(Article).filter(
        Article.sentiment_score < -0.3
    ).all()
    
    # Keywords to track
    topics = {
        "funding": ["funding", "investment", "raise", "series", "billion"],
        "security": ["security", "breach", "leak", "hack", "vulnerability"],
        "jobs": ["jobs", "employment", "workforce", "replaced", "displaced"],
        "regulation": ["regulation", "regulator", "government", "law", "policy"],
        "capability": ["breakthrough", "capability", "model", "performance", "beats"],
        "safety": ["safety", "concern", "risk", "dangerous", "alignment"],
        "competition": ["competitor", "competition", "vs", "battle", "race"]
    }
    
    positive_topics = Counter()
    negative_topics = Counter()
    
    # Count topic mentions in positive articles
    for article in positive_articles:
        text = (article.title + " " + (article.description or "")).lower()
        for topic, keywords in topics.items():
            if any(kw in text for kw in keywords):
                positive_topics[topic] += 1
    
    # Count topic mentions in negative articles
    for article in negative_articles:
        text = (article.title + " " + (article.description or "")).lower()
        for topic, keywords in topics.items():
            if any(kw in text for kw in keywords):
                negative_topics[topic] += 1
    
    session.close()
    
    print("\n=== POSITIVE SENTIMENT TOPICS ===")
    for topic, count in positive_topics.most_common():
        print(f"{topic}: {count} articles")
    
    print("\n=== NEGATIVE SENTIMENT TOPICS ===")
    for topic, count in negative_topics.most_common():
        print(f"{topic}: {count} articles")

if __name__ == "__main__":
    extract_topics()