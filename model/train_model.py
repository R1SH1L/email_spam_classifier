import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np

# Same preprocessing as Streamlit app
def preprocess_text(text):
    if not text:
        return ""
    
    text = str(text).lower()
    
    # Remove URLs and emails but keep other patterns
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Keep important punctuation and numbers for spam detection
    text = re.sub(r'[^\w\s!$%]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Load data
try:
    try:
        df = pd.read_csv('../data/spam.csv', encoding='latin-1')
    except:
        try:
            df = pd.read_csv('data/spam.csv', encoding='latin-1')
        except:
            df = pd.read_csv('spam.csv', encoding='latin-1')
    
    df = df.iloc[:, :2]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df.dropna()
    
    print(f"Loaded {len(df)} emails")
    print(f"Spam: {sum(df['label'])}, Ham: {len(df) - sum(df['label'])}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Preprocess
df['text'] = df['text'].apply(preprocess_text)
df = df[df['text'].str.len() > 0]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

# Improved model with better parameters
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=15000,
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams
        min_df=1,
        max_df=0.8,
        sublinear_tf=True,  # Better for spam detection
        token_pattern=r'\b\w+\b'
    )),
    ('classifier', LogisticRegression(
        random_state=42,
        max_iter=2000,
        C=1.0,  # Try different regularization
        class_weight='balanced'  # Handle class imbalance
    ))
])

print("Training improved model...")
model.fit(X_train, y_train)

# Test
accuracy = model.score(X_test, y_test)
y_pred = model.predict(X_test)

print(f"\n✅ Improved Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Test the specific problematic text
test_texts = [
    "URGENT! You've won $1000000! Click here NOW! Limited time offer!",
    "FREE money! Call now! Act fast!",
    "Congratulations! You have won $1000! Claim now!",
    "Hi, can we schedule a meeting for tomorrow at 3pm?",
    "Thanks for the document, I'll review it today."
]

print("\n🧪 Testing specific cases:")
for text in test_texts:
    cleaned = preprocess_text(text)
    pred = model.predict([cleaned])[0]
    prob = model.predict_proba([cleaned])[0]
    label = "SPAM" if pred == 1 else "HAM"
    confidence = prob[pred]
    print(f"Text: '{text}'")
    print(f"Cleaned: '{cleaned}'")
    print(f"Result: {label} ({confidence:.1%})")
    print("-" * 60)

# Save improved model
joblib.dump(model, 'model.pkl')
print(f"\n💾 Improved model saved!")
