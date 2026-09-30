import json
import pickle
import numpy as np
from database import get_session, Article
from nltk.sentiment import SentimentIntensityAnalyzer

# Load custom model
with open('custom_sentiment_model.pkl', 'rb') as f:
    custom_model, vectorizer = pickle.load(f)

# Load VADER
sia = SentimentIntensityAnalyzer()

# Get all articles
session = get_session()
articles = session.query(Article).all()
session.close()

print("Comparing VADER vs Custom Model on all articles...\n")

differences = []

for article in articles[:100]:  # Test on first 100
    text = f"{article.title} {article.description}"
    
    # VADER score
    vader_score = article.sentiment_score
    
    # Custom model score
    text_vec = vectorizer.transform([text])
    custom_pred = custom_model.predict(text_vec)[0]
    custom_proba = custom_model.predict_proba(text_vec)[0]
    
    # Compare
    vader_label = "POSITIVE" if vader_score > 0.05 else "NEGATIVE" if vader_score < -0.05 else "NEUTRAL"
    custom_label = "POSITIVE" if custom_pred == 1 else "NEGATIVE" if custom_pred == -1 else "NEUTRAL"
    
    match = "✓" if vader_label == custom_label else "✗"
    
    if vader_label != custom_label:
        differences.append({
            'title': article.title[:60],
            'vader': vader_label,
            'custom': custom_label,
            'match': match
        })

print(f"Agreements: {100 - len(differences)}")
print(f"Disagreements: {len(differences)}\n")

print("Examples of disagreement:")
for diff in differences[:5]:
    print(f"{diff['match']} {diff['title']}")
    print(f"   VADER: {diff['vader']} | Custom: {diff['custom']}\n")