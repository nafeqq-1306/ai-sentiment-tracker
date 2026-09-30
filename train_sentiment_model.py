import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle

# Load labeled data
with open('labeled_articles.json', 'r') as f:
    labeled_data = json.load(f)

# Prepare training data
texts = [f"{item['title']} {item['description']}" for item in labeled_data]
labels = np.array([item['human_label'] for item in labeled_data])

print(f"Training data: {len(texts)} articles")
print(f"Positive: {sum(labels == 1)}, Neutral: {sum(labels == 0)}, Negative: {sum(labels == -1)}")

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nTrain set: {len(X_train)}, Test set: {len(X_test)}")

# Convert text to numerical features using TF-IDF
print("\nVectorizing text...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train logistic regression model
print("Training model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate custom model
print("\n" + "="*70)
print("CUSTOM MODEL PERFORMANCE")
print("="*70)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")

# Compare with VADER
print("\n" + "="*70)
print("VADER PERFORMANCE")
print("="*70)

sia = SentimentIntensityAnalyzer()
vader_predictions = []

for text in X_test:
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    if compound > 0.05:
        vader_predictions.append(1)  # POSITIVE
    elif compound < -0.05:
        vader_predictions.append(-1)  # NEGATIVE
    else:
        vader_predictions.append(0)  # NEUTRAL

vader_predictions = np.array(vader_predictions)

vader_accuracy = accuracy_score(y_test, vader_predictions)
vader_precision = precision_score(y_test, vader_predictions, average='weighted')
vader_recall = recall_score(y_test, vader_predictions, average='weighted')
vader_f1 = f1_score(y_test, vader_predictions, average='weighted')

print(f"Accuracy:  {vader_accuracy:.3f}")
print(f"Precision: {vader_precision:.3f}")
print(f"Recall:    {vader_recall:.3f}")
print(f"F1-Score:  {vader_f1:.3f}")

# Comparison
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

print(f"\nCustom Model vs VADER:")
print(f"  Accuracy:  {accuracy:.3f} vs {vader_accuracy:.3f} ({(accuracy-vader_accuracy):+.3f})")
print(f"  Precision: {precision:.3f} vs {vader_precision:.3f} ({(precision-vader_precision):+.3f})")
print(f"  Recall:    {recall:.3f} vs {vader_recall:.3f} ({(recall-vader_recall):+.3f})")
print(f"  F1-Score:  {f1:.3f} vs {vader_f1:.3f} ({(f1-vader_f1):+.3f})")

if accuracy > vader_accuracy:
    print(f"\n✓ Custom model is {(accuracy-vader_accuracy)*100:.1f}% more accurate")
else:
    print(f"\n✗ VADER is {(vader_accuracy-accuracy)*100:.1f}% more accurate")

# Save model
print("\nSaving model...")
with open('custom_sentiment_model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)

print("✓ Model saved as custom_sentiment_model.pkl")