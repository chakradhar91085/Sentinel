from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

# Define Request Structure
class PredictRequest(BaseModel):
    text: str

app = FastAPI(
    title="Sentinel API",
    description="AI-Powered Toxicity Detection & Content Moderation System",
    version="2.0.0"
)

# Add CORS Middleware to allow requests from our UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the DistilBERT model + tokenizer into memory on startup
MODEL_DIR = "sentinel_model"

if os.path.exists(MODEL_DIR):
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    print(f"[OK] DistilBERT model loaded from '{MODEL_DIR}/'")
else:
    tokenizer = None
    model = None
    print(f"[!] Model not found at '{MODEL_DIR}/'. Run train_transformer.py first.")


@app.post("/predict")
async def predict_toxicity(request: PredictRequest):
    """Analyze text for toxicity using the fine-tuned DistilBERT model."""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please run train_transformer.py to generate it."
        )

    # Tokenize the raw input (DistilBERT handles text preprocessing natively)
    encoding = tokenizer(
        request.text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Inference with no gradient computation for speed
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )

    # Softmax logits → probability
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    toxic_prob = probs[0][1].item()  # Class 1 = toxic

    return {
        "toxicity": round(float(toxic_prob), 4),
        "label": "toxic" if toxic_prob >= 0.5 else "non-toxic"
    }


@app.get("/health")
async def health_check():
    """Check API status and model availability."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_dir": MODEL_DIR,
    }


# Serve frontend static files (must be mounted AFTER API routes)
FRONTEND_DIR = "frontend"
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    print(f"[OK] Frontend served from '{FRONTEND_DIR}/'")
else:
    @app.get("/")
    async def root():
        return {"message": "Sentinel API v2.0 — Visit /docs for interactive API documentation."}
