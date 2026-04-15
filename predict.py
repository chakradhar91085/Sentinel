import joblib
import json
import argparse
import sys

def predict_toxicity(text):
    try:
        model = joblib.load("toxicity_model.pkl")
    except FileNotFoundError:
        print(json.dumps({"error": "Model not found. Please run train.py first."}))
        sys.exit(1)

    # Predict probability of class 1 (toxic)
    # predict_proba returns [[P(class 0), P(class 1)]]
    prob = model.predict_proba([text])[0][1]
    
    result = {
        "toxicity": round(float(prob), 4),
        "label": "toxic" if prob >= 0.5 else "non-toxic"
    }
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict toxicity of a comment.")
    parser.add_argument("text", type=str, help="The comment text to analyze")
    args = parser.parse_args()

    predict_toxicity(args.text)
