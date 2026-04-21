# Sentinel Project Handoff

Welcome to the **Sentinel – AI-Powered Toxicity Detection & Moderation System**! 

This document serves as a contextual handoff to help you (the AI) understand the project's state, architecture, and recent changes so that you can immediately assist the user.

## Project Overview
Sentinel is an end-to-end content moderation tool analyzing text toxicity in real-time. It recently evolved from a simple TF-IDF & Logistic Regression baseline (`train.py`) to a state-of-the-art **Transformer-based implementation utilizing DistilBERT** (`train_transformer.py`).

## Tech Stack
*   **Machine Learning:** PyTorch, Hugging Face Transformers (`DistilBERT`), Datasets (Jigsaw Toxic Comment Dataset).
*   **Backend / API:** Python, FastAPI, Uvicorn.
*   **Frontend:** Vanilla HTML, CSS, JavaScript featuring a warm editorial design with Instrument Serif + DM Sans typography, terracotta accents, and a clean arc gauge.

## Environment & Hardware Context
*   **Operating System:** Windows, utilizing **WSL2**.
*   **GPU:** The user possesses an **NVIDIA RTX 5060**.
*   **Note:** PyTorch natively struggled with Blackwell architecture binaries on Windows, so the user set up CUDA via WSL2 to ensure hardware-accelerated model training and inference. Make sure any script running PyTorch seamlessly checks for and uses `cuda` where applicable.

## Project Structure
*   `app.py`: The FastAPI application serving the REST endpoints (`/predict`, `/health`) and mounting the frontend static files.
*   `train_transformer.py`: The script to fetch the Jigsaw dataset, perform NLP preprocessing, and fine-tune `distilbert-base-uncased`. The resulting model is saved in the `sentinel_model/` directory.
*   `train.py` & `predict.py`: Legacy and CLI scripts (respectively). 
*   `frontend/`: Contains `index.html`, `style.css`, and `script.js` for the interactive web UI.
*   `Dockerfile.train`: Docker setup for training tasks.

## Current Status & recent changes
1. The `train_transformer.py` script was implemented and successfully utilizes GPU acceleration for training.
2. The `app.py` script was updated to load the Hugging Face model from `sentinel_model/` instead of the legacy `.pkl` file.
3. The codebase has been synchronized with the user's GitHub repository.

## How to proceed
When taking over, read the latest user request. If they need to train the model further, ensure it is run in the WSL environment for GPU access. If modifying the UI, remember the aesthetic is strictly premium "**glassmorphism**" as per the user's prior specifications.
