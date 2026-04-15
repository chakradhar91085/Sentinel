import pandas as pd
import re
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def clean_text(text):
    """Lowercases, removes punctuation, keeps only alphabets and spaces."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

def main():
    print("Loading dataset...")
    # Load dataset
    dataset = load_dataset("tasksource/jigsaw_toxicity", split="train")
    
    # We will use exactly 50,000 samples
    dataset = dataset.select(range(50000))
    df = pd.DataFrame(dataset)

    print("Preprocessing data...")
    # 3. Apply text preprocessing
    df['comment_text'] = df['comment_text'].apply(clean_text)
    
    # 1. Combine Toxic Labels 
    # Determine the presence of any toxic identity attribute
    toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    available_cols = [col for col in toxic_columns if col in df.columns]
    
    df['is_toxic'] = df[available_cols].max(axis=1).astype(int)

    print(f"Total samples before balancing: {len(df)} (Toxic: {df['is_toxic'].sum()})")

    # 2. Handle Dataset Imbalance
    toxic_df = df[df['is_toxic'] == 1]
    clean_df = df[df['is_toxic'] == 0]
    
    # Under-sample majority class exactly to match toxic
    n_toxic = len(toxic_df)
    clean_downsampled = clean_df.sample(n=n_toxic, random_state=42)
    
    # 5. Create balanced dataset
    balanced_df = pd.concat([toxic_df, clean_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total samples after balancing: {len(balanced_df)} (Toxic: {balanced_df['is_toxic'].sum()})")

    X = balanced_df['comment_text']
    y = balanced_df['is_toxic']

    # Train Test Split (80/20) for 6. Evaluation metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    # 4. Improve TF-IDF Vectorizer
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english'),
        LogisticRegression(max_iter=1000)
    )

    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    
    # 6. Print evaluation metrics
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['non-toxic', 'toxic']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("--------------------------\n")

    # 7. Save model pipeline
    model_path = "toxicity_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}.")

if __name__ == "__main__":
    main()
