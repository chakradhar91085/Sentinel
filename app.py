from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import os

# Define Request Structure
class PredictRequest(BaseModel):
    text: str

app = FastAPI(
    title="Sentinel API",
    description="Backend API for Toxicity Detection",
    version="1.0.0"
)

# Add CORS Middleware to allow requests from our UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model directly into memory on startup
MODEL_PATH = "toxicity_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    model = None
    print(f"Warning: Model not found at {MODEL_PATH}")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

@app.post("/predict")
async def predict_toxicity(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Please run train.py to generate it.")
    
    clean = clean_text(request.text)
    
    # Model generates probabilities
    prob = model.predict_proba([clean])[0][1]
    
    return {
        "toxicity": round(float(prob), 4),
        "label": "toxic" if prob >= 0.5 else "non-toxic"
    }

@app.get("/")
async def root():
    return {"message": "Welcome to Sentinel API. Navigate to /docs to test the model."}
