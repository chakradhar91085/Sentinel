import json
import argparse
import sys
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODEL_DIR = "sentinel_model"

def load_model():
    """Load the fine-tuned DistilBERT model and tokenizer."""
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        return tokenizer, model
    except Exception:
        print(json.dumps({"error": f"Model not found in '{MODEL_DIR}/'. Please run train_transformer.py first."}))
        sys.exit(1)

def predict_toxicity(raw_text, tokenizer, model):
    """Run inference on a single text input using the transformer model."""
    # DistilBERT tokenizer handles all preprocessing (no manual clean_text needed)
    encoding = tokenizer(
        raw_text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )

    # Softmax to convert logits → probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    toxic_prob = probs[0][1].item()  # Class 1 = toxic

    result = {
        "toxicity": round(toxic_prob, 4),
        "label": "toxic" if toxic_prob >= 0.5 else "non-toxic"
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict toxicity of a comment.")
    parser.add_argument("text", type=str, help="The comment text to analyze")
    args = parser.parse_args()

    tokenizer, model = load_model()
    predict_toxicity(args.text, tokenizer, model)
