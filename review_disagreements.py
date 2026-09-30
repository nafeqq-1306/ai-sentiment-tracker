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

# Load your labeled data
with open('labeled_articles.json', 'r') as f:
    labeled_data = json.load(f)

# Get all articles
session = get_session()
articles = session.query(Article).all()
session.close()

# Find disagreements
disagreements = []

for article in articles[:100]:
    text = f"{article.title} {article.description if article.description else ''}"
    
    # VADER score
    vader_score = article.sentiment_score
    vader_label = "POSITIVE" if vader_score > 0.05 else "NEGATIVE" if vader_score < -0.05 else "NEUTRAL"
    
    # Custom model score
    text_vec = vectorizer.transform([text])
    custom_pred = custom_model.predict(text_vec)[0]
    custom_label = "POSITIVE" if custom_pred == 1 else "NEGATIVE" if custom_pred == -1 else "NEUTRAL"
    
    # Find your human label
    human_label = None
    for item in labeled_data:
        if item['title'] == article.title:
            human_label_int = item['human_label']
            human_label = "POSITIVE" if human_label_int == 1 else "NEGATIVE" if human_label_int == -1 else "NEUTRAL"
            break
    
    # Check disagreement
    if vader_label != custom_label:
        disagreements.append({
            'title': article.title,
            'description': (article.description if article.description else "")[:150],
            'vader_label': vader_label,
            'custom_label': custom_label,
            'human_label': human_label,
            'vader_score': vader_score
        })

print(f"Found {len(disagreements)} disagreements\n")

# Show each one and let user correct
corrected = 0

for idx, item in enumerate(disagreements):
    print(f"\n{'='*70}")
    print(f"[{idx+1}/{len(disagreements)}]")
    print(f"{'='*70}")
    print(f"\nTitle: {item['title']}")
    print(f"Description: {item['description']}...")
    print(f"\nVADER says: {item['vader_label']} (score: {item['vader_score']:.3f})")
    print(f"Custom says: {item['custom_label']}")
    print(f"You labeled: {item['human_label']}")
    
    print(f"\nWhich is correct?")
    print(f"  1 = POSITIVE")
    print(f"  0 = NEUTRAL")
    print(f" -1 = NEGATIVE")
    print(f"  s = SKIP (keep current label)")
    
    while True:
        choice = input("\nYour decision: ").strip()
        if choice in ['1', '0', '-1', 's']:
            break
    
    if choice != 's':
        # Find and update the label
        for label_item in labeled_data:
            if label_item['title'] == item['title']:
                label_item['human_label'] = int(choice)
                corrected += 1
                print(f"✓ Updated label to {choice}")
                break

# Save corrected data
with open('labeled_articles.json', 'w') as f:
    json.dump(labeled_data, f, indent=2)

print(f"\n{'='*70}")
print(f"✓ Corrected {corrected} labels")
print(f"✓ Saved to labeled_articles.json")
print(f"\nNow run: python train_sentiment_model.py")