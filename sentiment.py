from nltk.sentiment import SentimentIntensityAnalyzer
from database import get_session, Article
import nltk

# Download VADER lexicon
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyze sentiment using VADER"""
    if not text or len(text.strip()) == 0:
        return 0.0, "NEUTRAL"
    
    try:
        scores = sia.polarity_scores(text)
        compound = scores['compound']  # -1 to 1 score
        
        # Classify based on compound score
        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        return compound, label
    
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 0.0, "NEUTRAL"

def update_sentiment_scores():
    """Calculate sentiment for all articles without scores"""
    session = get_session()
    
    articles = session.query(Article).filter(Article.sentiment_score == None).all()
    
    print(f"Analyzing sentiment for {len(articles)} articles with VADER...")
    
    for i, article in enumerate(articles):
        text = f"{article.title} {article.description or ''}"
        score, label = analyze_sentiment(text)
        
        article.sentiment_score = score
        article.sentiment_label = label
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(articles)}")
    
    session.commit()
    session.close()
    print("Sentiment analysis complete!")

if __name__ == "__main__":
    update_sentiment_scores()