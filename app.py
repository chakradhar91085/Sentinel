from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Define Request Structure
class PredictRequest(BaseModel):
    text: str

class PredictBatchRequest(BaseModel):
    texts: List[str]

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Sentinel API",
    description="AI-Powered Toxicity Detection & Content Moderation System",
    version="2.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("100/minute")
async def predict_toxicity(request: Request, payload: PredictRequest):
    """Analyze text for toxicity using the fine-tuned DistilBERT model."""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please run train_transformer.py to generate it."
        )

    # Tokenize the raw input (DistilBERT handles text preprocessing natively)
    encoding = tokenizer(
        payload.text,
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

    # Apply Temperature Scaling to soften the extreme probabilities
    # Neural networks are often overconfident (outputting 99% or 1%). 
    # Dividing the logits by a temperature > 1.0 smooths out the distribution.
    temperature = 10.0 
    scaled_logits = outputs.logits / temperature

    # Softmax logits → probability
    probs = torch.nn.functional.softmax(scaled_logits, dim=1)
    toxic_prob = probs[0][1].item()  # Class 1 = toxic

    return {
        "toxicity": round(float(toxic_prob), 4),
        "label": "toxic" if toxic_prob >= 0.5 else "non-toxic"
    }


@app.post("/predict_batch")
@limiter.limit("100/minute")
async def predict_toxicity_batch(request: Request, payload: PredictBatchRequest):
    """Analyze a batch of texts for toxicity."""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    encoding = tokenizer(
        payload.texts,
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

    temperature = 10.0 
    scaled_logits = outputs.logits / temperature
    probs = torch.nn.functional.softmax(scaled_logits, dim=1)

    results = []
    for i in range(len(payload.texts)):
        toxic_prob = probs[i][1].item()
        results.append({
            "toxicity": round(float(toxic_prob), 4),
            "label": "toxic" if toxic_prob >= 0.5 else "non-toxic"
        })

    return {"results": results}


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
