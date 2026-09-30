from database import get_session, Article
import random

def label_articles():
    """Manually label articles for training"""
    
    session = get_session()
    articles = session.query(Article).limit(300).all()
    session.close()
    
    labeled = []
    
    for idx, article in enumerate(articles):
        print(f"\n[{idx+1}/{len(articles)}]")
        print(f"Title: {article.title}")
        
        # Handle None description
        description = article.description if article.description else "No description available"
        print(f"Description: {description[:200]}...")
        print(f"Current VADER score: {article.sentiment_score}")
        
        # Get human label
        while True:
            label = input("Label (1=POSITIVE, 0=NEUTRAL, -1=NEGATIVE): ").strip()
            if label in ['-1', '0', '1']:
                break
        
        labeled.append({
            'id': article.id,
            'title': article.title,
            'description': description,
            'vader_score': article.sentiment_score,
            'human_label': int(label)
        })
    
    # Save labeled data
    import json
    with open('labeled_articles.json', 'w') as f:
        json.dump(labeled, f, indent=2)
    
    print(f"\n✓ Labeled {len(labeled)} articles")

if __name__ == "__main__":
    label_articles()