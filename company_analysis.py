from database import get_session, Article
import statistics

def analyze_by_company():
    """Compare sentiment across AI companies"""
    session = get_session()
    
    companies = {
        "OpenAI": ["openai", "chatgpt", "gpt"],
        "Anthropic": ["anthropic", "claude"],
        "Google": ["google ai", "gemini", "bard"],
        "Meta": ["meta ai", "llama"],
        "DeepMind": ["deepmind"]
    }
    
    results = {}
    
    for company, keywords in companies.items():
        # Find articles mentioning this company
        articles = session.query(Article).all()
        company_articles = []
        
        for article in articles:
            text = (article.title + " " + (article.description or "")).lower()
            if any(kw in text for kw in keywords):
                if article.sentiment_score is not None:
                    company_articles.append(article.sentiment_score)
        
        if company_articles:
            avg_sentiment = statistics.mean(company_articles)
            positive = sum(1 for s in company_articles if s > 0.3)
            negative = sum(1 for s in company_articles if s < -0.3)
            
            results[company] = {
                "avg_sentiment": avg_sentiment,
                "articles": len(company_articles),
                "positive": positive,
                "negative": negative
            }
    
    session.close()
    
    print("\n=== SENTIMENT BY COMPANY ===")
    for company, data in sorted(results.items(), key=lambda x: x[1]["avg_sentiment"], reverse=True):
        print(f"\n{company}:")
        print(f"  Average Sentiment: {data['avg_sentiment']:.2f}")
        print(f"  Total Articles: {data['articles']}")
        print(f"  Positive: {data['positive']} | Negative: {data['negative']}")

if __name__ == "__main__":
    analyze_by_company()