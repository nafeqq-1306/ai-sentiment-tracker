from transformers import pipeline
from database import get_session, Article

# Load sentiment model (downloads on first run, ~500MB)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    """Analyze sentiment of text and return score (-1 to 1) and label"""
    if not text or len(text.strip()) == 0:
        return 0.0, "NEUTRAL"
    
    # Limit text length (model works better on shorter inputs)
    text = text[:512]
    
    try:
        result = sentiment_pipeline(text)[0]
        label = result["label"]
        score = result["score"]
        
        # Convert to -1 to 1 scale
        if label == "POSITIVE":
            return score, "POSITIVE"
        else:
            return -score, "NEGATIVE"
    
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 0.0, "NEUTRAL"

def update_sentiment_scores():
    """Calculate sentiment for all articles without scores"""
    session = get_session()
    
    # Get articles without sentiment scores
    articles = session.query(Article).filter(Article.sentiment_score == None).all()
    
    print(f"Analyzing sentiment for {len(articles)} articles...")
    
    for i, article in enumerate(articles):
        # Use title + description for sentiment (full content might be too long)
        text = f"{article.title} {article.description or ''}"
        score, label = analyze_sentiment(text)
        
        article.sentiment_score = score
        article.sentiment_label = label
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(articles)}")
    
    session.commit()
    session.close()
    print("Sentiment analysis complete!")

if __name__ == "__main__":
    update_sentiment_scores()