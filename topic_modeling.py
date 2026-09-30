from database import get_session, Article
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np

def prepare_documents():
    """Get all article titles and descriptions"""
    session = get_session()
    articles = session.query(Article).all()
    session.close()
    
    # Combine title + description for each article
    documents = []
    article_ids = []
    
    for article in articles:
        if article.title and article.sentiment_score is not None:
            text = (article.title + " " + (article.description or "")).lower()
            documents.append(text)
            article_ids.append(article.id)
    
    print(f"Prepared {len(documents)} documents for LDA")
    return documents, article_ids

def build_lda_model(documents, n_topics=5):
    """Build LDA model to discover topics"""
    
    print(f"\nBuilding LDA model with {n_topics} topics...")
    
    # Create vocabulary - convert text to word counts
    # Stop words: common words like "the", "a", "is" that don't add meaning
    vectorizer = CountVectorizer(
        max_df=0.95,  # Ignore words that appear in >95% of docs
        min_df=2,      # Ignore words that appear in <2 docs
        stop_words='english',
        max_features=1000
    )
    
    # Transform documents to word count matrix
    doc_term_matrix = vectorizer.fit_transform(documents)
    
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"Document-term matrix shape: {doc_term_matrix.shape}")
    
    # Build LDA model
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method='online'
    )
    
    lda_model.fit(doc_term_matrix)
    print("✓ LDA model fitted successfully")
    
    return lda_model, vectorizer, doc_term_matrix

def display_topics(lda_model, vectorizer, n_words=10):
    """Display the discovered topics"""
    
    print("\n" + "="*70)
    print("DISCOVERED TOPICS (Top words per topic)")
    print("="*70)
    
    feature_names = vectorizer.get_feature_names_out()
    
    topic_descriptions = {}
    
    for topic_id, topic in enumerate(lda_model.components_):
        # Get top words for this topic
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [topic[i] for i in top_indices]
        
        print(f"\n📌 Topic {topic_id}:")
        for word, weight in zip(top_words, top_weights):
            print(f"   {word}: {weight:.4f}")
        
        # Store for later use
        topic_descriptions[topic_id] = top_words
    
    print("\n" + "="*70)
    return topic_descriptions

def get_document_topics(lda_model, doc_term_matrix):
    """Get topic distribution for each document"""
    
    # Transform documents to topic distribution
    doc_topic_dist = lda_model.transform(doc_term_matrix)
    
    return doc_topic_dist

def analyze_topics_by_sentiment(documents, doc_topic_dist, article_ids, sentiment_scores):
    """Analyze which topics appear in positive vs negative articles"""
    
    print("\n" + "="*70)
    print("TOPICS BY SENTIMENT")
    print("="*70)
    
    n_topics = doc_topic_dist.shape[1]
    
    # Separate positive and negative articles
    positive_topic_dist = []
    negative_topic_dist = []
    
    for idx, (doc_id, sentiment) in enumerate(zip(article_ids, sentiment_scores)):
        if sentiment > 0.3:
            positive_topic_dist.append(doc_topic_dist[idx])
        elif sentiment < -0.3:
            negative_topic_dist.append(doc_topic_dist[idx])
    
    positive_topic_dist = np.array(positive_topic_dist)
    negative_topic_dist = np.array(negative_topic_dist)
    
    print(f"\nPositive articles: {len(positive_topic_dist)}")
    print(f"Negative articles: {len(negative_topic_dist)}")
    
    print("\n" + "-"*70)
    print("Average topic prevalence in POSITIVE articles:")
    print("-"*70)
    
    positive_avg = positive_topic_dist.mean(axis=0)
    for topic_id, avg_weight in enumerate(positive_avg):
        print(f"Topic {topic_id}: {avg_weight:.4f}")
    
    print("\n" + "-"*70)
    print("Average topic prevalence in NEGATIVE articles:")
    print("-"*70)
    
    negative_avg = negative_topic_dist.mean(axis=0)
    for topic_id, avg_weight in enumerate(negative_avg):
        print(f"Topic {topic_id}: {avg_weight:.4f}")
    
    print("\n" + "-"*70)
    print("DIFFERENCE (What makes articles positive vs negative?):")
    print("-"*70)
    
    difference = positive_avg - negative_avg
    sorted_topics = np.argsort(difference)[::-1]
    
    print("\nTopics more common in POSITIVE articles:")
    for topic_id in sorted_topics[:3]:
        if difference[topic_id] > 0:
            print(f"  Topic {topic_id}: +{difference[topic_id]:.4f}")
    
    print("\nTopics more common in NEGATIVE articles:")
    for topic_id in sorted_topics[-3:]:
        if difference[topic_id] < 0:
            print(f"  Topic {topic_id}: {difference[topic_id]:.4f}")

def analyze_companies_by_topic(documents, doc_topic_dist, article_ids):
    """Analyze which companies appear in which topics"""
    
    companies = {
        "OpenAI": ["openai", "chatgpt", "gpt"],
        "Anthropic": ["anthropic", "claude"],
        "Google": ["google", "gemini", "bard"],
        "Meta": ["meta", "llama"],
        "DeepMind": ["deepmind"],
        "Microsoft": ["microsoft", "copilot"],
    }
    
    print("\n" + "="*70)
    print("COMPANIES BY TOPIC")
    print("="*70)
    
    n_topics = doc_topic_dist.shape[1]
    
    # For each topic, track which companies appear
    topic_company_matrix = {topic_id: {company: [] for company in companies} for topic_id in range(n_topics)}
    
    # Parse documents and map to companies
    for idx, doc in enumerate(documents):
        doc_lower = doc.lower()
        if len(doc_topic_dist[idx]) > 0:
            dominant_topic = np.argmax(doc_topic_dist[idx])
            
            for company, keywords in companies.items():
                if any(kw in doc_lower for kw in keywords):
                    if dominant_topic < n_topics:
                        topic_company_matrix[dominant_topic][company].append(idx)
    
    # Display results
    for topic_id in range(n_topics):
        print(f"\n📍 Topic {topic_id}:")
        
        company_counts = {
            company: len(topic_company_matrix[topic_id][company])
            for company in companies
        }
        
        sorted_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)
        
        for company, count in sorted_companies:
            if count > 0:
                percentage = (count / sum(company_counts.values())) * 100
                print(f"  {company}: {count} articles ({percentage:.1f}%)")

def analyze_company_sentiment_by_topic(documents, doc_topic_dist, article_ids, sentiment_scores):
    """Analyze company sentiment within each topic"""
    
    companies = {
        "OpenAI": ["openai", "chatgpt", "gpt"],
        "Anthropic": ["anthropic", "claude"],
        "Google": ["google", "gemini", "bard"],
        "Meta": ["meta", "llama"],
        "DeepMind": ["deepmind"],
        "Microsoft": ["microsoft", "copilot"],
    }
    
    print("\n" + "="*70)
    print("COMPANY SENTIMENT WITHIN TOPICS")
    print("="*70)
    
    n_topics = doc_topic_dist.shape[1]
    
    for topic_id in range(n_topics):
        print(f"\n📌 Topic {topic_id}:")
        
        company_sentiments = {}
        
        for idx, doc in enumerate(documents):
            doc_lower = doc.lower()
            doc_topic_strength = doc_topic_dist[idx][topic_id]
            sentiment = sentiment_scores[idx]
            
            for company, keywords in companies.items():
                if any(kw in doc_lower for kw in keywords):
                    if company not in company_sentiments:
                        company_sentiments[company] = []
                    company_sentiments[company].append({
                        'sentiment': sentiment,
                        'topic_strength': doc_topic_strength
                    })
        
        # Calculate average sentiment per company in this topic
        for company in sorted(company_sentiments.keys()):
            sentiments = [s['sentiment'] for s in company_sentiments[company]]
            avg_sentiment = np.mean(sentiments)
            count = len(sentiments)
            
            if count > 0:
                sentiment_label = "POSITIVE 🟢" if avg_sentiment > 0.2 else "NEGATIVE 🔴" if avg_sentiment < -0.2 else "NEUTRAL ⚪"
                print(f"  {company}: {avg_sentiment:.3f} ({sentiment_label}) - {count} mentions")

def analyze_topic_evolution_by_company(documents, doc_topic_dist, article_ids, published_dates):
    """Show how companies are discussed across different topics"""
    
    companies = {
        "OpenAI": ["openai", "chatgpt", "gpt"],
        "Anthropic": ["anthropic", "claude"],
        "Google": ["google", "gemini", "bard"],
    }
    
    print("\n" + "="*70)
    print("TOPIC DISTRIBUTION PER COMPANY (What topics mention each company?)")
    print("="*70)
    
    n_topics = doc_topic_dist.shape[1]
    
    for company, keywords in companies.items():
        print(f"\n🏢 {company}:")
        
        company_topic_weights = np.zeros(n_topics)
        company_doc_count = 0
        
        for idx, doc in enumerate(documents):
            doc_lower = doc.lower()
            if any(kw in doc_lower for kw in keywords):
                # Add this document's topic distribution weighted
                company_topic_weights += doc_topic_dist[idx]
                company_doc_count += 1
        
        if company_doc_count > 0:
            # Normalize
            company_topic_weights /= company_doc_count
            
            # Sort by weight
            sorted_topics = np.argsort(company_topic_weights)[::-1]
            
            print(f"  (Based on {company_doc_count} articles)")
            for topic_id in sorted_topics[:3]:
                weight = company_topic_weights[topic_id]
                if weight > 0.05:
                    print(f"    Topic {topic_id}: {weight:.3f}")

def main():
    """Main execution"""
    print("Step 1: Preparing documents...")
    documents, article_ids = prepare_documents()
    
    print("\nStep 2: Building LDA model...")
    lda_model, vectorizer, doc_term_matrix = build_lda_model(documents, n_topics=5)
    
    print("\nStep 3: Displaying discovered topics...")
    topic_descriptions = display_topics(lda_model, vectorizer, n_words=10)
    
    print("\nStep 4: Getting document-topic distribution...")
    doc_topic_dist = get_document_topics(lda_model, doc_term_matrix)
    
    print("\nStep 5: Analyzing topics by sentiment...")
    # Get sentiment scores from database
    session = get_session()
    articles = session.query(Article).filter(Article.id.in_(article_ids)).all()
    sentiment_scores = [a.sentiment_score for a in articles]
    session.close()
    
    analyze_topics_by_sentiment(documents, doc_topic_dist, article_ids, sentiment_scores)
    
    print("\n✓ Topic modeling complete!")
    
    return lda_model, vectorizer, doc_topic_dist, topic_descriptions

if __name__ == "__main__":
    main()