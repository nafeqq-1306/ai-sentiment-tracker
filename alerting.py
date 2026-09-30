from database import get_session, Article
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

def get_sentiment_summary(hours_back=1):
    """Get sentiment summary for last N hours"""
    session = get_session()
    cutoff = datetime.utcnow() - timedelta(hours=hours_back)
    
    articles = session.query(Article).filter(
        Article.scraped_at >= cutoff
    ).all()
    session.close()
    
    if not articles:
        return None
    
    scores = [a.sentiment_score for a in articles if a.sentiment_score is not None]
    if not scores:
        return None
    
    return {
        'count': len(articles),
        'avg_sentiment': sum(scores) / len(scores),
        'positive': sum(1 for s in scores if s > 0.05),
        'negative': sum(1 for s in scores if s < -0.05),
        'neutral': sum(1 for s in scores if -0.05 <= s <= 0.05)
    }

def get_company_sentiment(company_keywords, hours_back=1):
    """Get sentiment for specific company"""
    session = get_session()
    cutoff = datetime.utcnow() - timedelta(hours=hours_back)
    
    articles = session.query(Article).filter(
        Article.scraped_at >= cutoff
    ).all()
    session.close()
    
    company_articles = []
    for article in articles:
        text = (article.title + " " + (article.description or "")).lower()
        if any(kw in text for kw in company_keywords):
            company_articles.append(article)
    
    if not company_articles:
        return None
    
    scores = [a.sentiment_score for a in company_articles if a.sentiment_score is not None]
    if not scores:
        return None
    
    return {
        'count': len(company_articles),
        'avg_sentiment': sum(scores) / len(scores),
        'articles': [a.title for a in company_articles[:3]]
    }

def detect_changes_and_alert():
    """Main alerting logic - compare last hour to previous hour"""
    
    print(f"[{datetime.utcnow()}] Running sentiment check...")
    
    # Get current hour sentiment
    current = get_sentiment_summary(hours_back=1)
    
    # Get previous hour sentiment
    session = get_session()
    cutoff_current = datetime.utcnow() - timedelta(hours=1)
    cutoff_previous = datetime.utcnow() - timedelta(hours=2)
    
    articles_previous = session.query(Article).filter(
        Article.scraped_at.between(cutoff_previous, cutoff_current)
    ).all()
    session.close()
    
    if not articles_previous:
        print("No previous data, skipping comparison")
        return
    
    scores_previous = [a.sentiment_score for a in articles_previous if a.sentiment_score is not None]
    if not scores_previous:
        return
    
    previous = {
        'avg_sentiment': sum(scores_previous) / len(scores_previous),
        'count': len(articles_previous)
    }
    
    if not current:
        print("No current data")
        return
    
    # Calculate change
    change = current['avg_sentiment'] - previous['avg_sentiment']
    pct_change = (change / abs(previous['avg_sentiment'])) * 100 if previous['avg_sentiment'] != 0 else 0
    
    print(f"Current sentiment: {current['avg_sentiment']:.3f}")
    print(f"Previous sentiment: {previous['avg_sentiment']:.3f}")
    print(f"Change: {change:+.3f} ({pct_change:+.1f}%)")
    
    alerts = []
    
    # Alert 1: Major sentiment shift
    if abs(change) > 0.15:
        if change < 0:
            alerts.append({
                'emoji': '🔴',
                'title': 'Negative Sentiment Surge',
                'message': f"Sentiment dropped {abs(change):.3f} points ({pct_change:.1f}%)\nCurrent: {current['avg_sentiment']:.3f} | Previous: {previous['avg_sentiment']:.3f}",
                'details': f"Negative: {current['negative']} | Neutral: {current['neutral']} | Positive: {current['positive']}"
            })
        else:
            alerts.append({
                'emoji': '🟢',
                'title': 'Positive Sentiment Surge',
                'message': f"Sentiment improved {change:.3f} points ({pct_change:+.1f}%)\nCurrent: {current['avg_sentiment']:.3f} | Previous: {previous['avg_sentiment']:.3f}",
                'details': f"Positive: {current['positive']} | Neutral: {current['neutral']} | Negative: {current['negative']}"
            })
    
    # Alert 2: Company-specific alerts
    companies = {
        "OpenAI": ["openai", "chatgpt", "gpt"],
        "Anthropic": ["anthropic", "claude"],
        "Google": ["google", "gemini"],
    }
    
    for company_name, keywords in companies.items():
        company_data = get_company_sentiment(keywords, hours_back=1)
        if company_data and company_data['count'] > 0:
            if company_data['avg_sentiment'] < -0.3:
                alerts.append({
                    'emoji': '⚠️',
                    'title': f'{company_name} Negative Coverage',
                    'message': f"Sentiment: {company_data['avg_sentiment']:.3f} | {company_data['count']} articles",
                    'details': f"Latest: {company_data['articles'][0] if company_data['articles'] else 'N/A'}"
                })
    
    # Send all alerts
    if alerts:
        for alert in alerts:
            send_slack_alert(alert)
    else:
        print("No significant changes, no alerts sent")

def send_slack_alert(alert):
    """Send formatted alert to Slack"""
    
    message = {
        "text": f"{alert['emoji']} {alert['title']}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{alert['emoji']} *{alert['title']}*\n{alert['message']}"
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": alert['details']
                    }
                ]
            },
            {
                "type": "divider"
            }
        ]
    }
    
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=message)
        if response.status_code == 200:
            print(f"✓ Alert sent: {alert['title']}")
        else:
            print(f"✗ Failed to send alert: {response.status_code}")
    except Exception as e:
        print(f"✗ Error sending alert: {e}")

if __name__ == "__main__":
    detect_changes_and_alert()