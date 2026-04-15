import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

def main():
    print("Loading dataset...")
    # Load the training split of the jigsaw dataset
    dataset = load_dataset("tasksource/jigsaw_toxicity", split="train")
    
    # We will use exactly 50,000 samples for the MVP to ensure better representation
    dataset = dataset.select(range(50000))
    df = pd.DataFrame(dataset)

    print("Preprocessing data...")
    # For MVP, we'll just consider 'toxic' column as the main label
    X = df['comment_text'].fillna('')
    y = df['toxic']

    print("Training model...")
    # Create an ML pipeline
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=10000, stop_words='english'),
        LogisticRegression(max_iter=1000)
    )

    pipeline.fit(X, y)
    print("Training completed.")

    # Save the model
    model_path = "toxicity_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}.")

if __name__ == "__main__":
    main()
