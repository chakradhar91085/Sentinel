"""
Sentinel Phase 4 — Fine-tune DistilBERT for Toxicity Detection
Optimized for CPU training:
  - Pre-tokenizes ALL data upfront (no lazy tokenization)
  - Freezes DistilBERT base layers, only trains classifier head (massively faster)
  - Auto-detects GPU (CUDA) and falls back to CPU
"""

import os
import time
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128          # Max token length per comment
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
SAMPLE_SIZE = 50_000   # Raw samples to pull from Jigsaw
OUTPUT_DIR = "sentinel_model"


# ──────────────────────────────────────────────
# Pre-tokenize all data upfront (critical for CPU speed)
# ──────────────────────────────────────────────
class PreTokenizedDataset(Dataset):
    """Stores pre-tokenized tensors for zero-overhead DataLoader iteration."""

    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": self.labels[idx],
        }


def tokenize_all(texts, tokenizer, max_len):
    """Batch-tokenize all texts at once. Much faster than one-by-one."""
    print(f"  Tokenizing {len(texts)} samples (batch mode)...")
    start = time.time()

    encoding = tokenizer(
        texts.tolist(),
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    elapsed = time.time() - start
    print(f"  Tokenization complete in {elapsed:.1f}s")

    return encoding["input_ids"], encoding["attention_mask"]


# ──────────────────────────────────────────────
# Training loop for one epoch
# ──────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch_num):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Progress every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - start_time
            batches_done = batch_idx + 1
            batches_left = num_batches - batches_done
            time_per_batch = elapsed / batches_done
            eta = batches_left * time_per_batch

            print(f"    [{batches_done}/{num_batches}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {correct/total:.4f} | "
                  f"ETA: {eta:.0f}s")

    avg_loss = total_loss / num_batches
    accuracy = correct / total
    return avg_loss, accuracy


# ──────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def main():
    total_start = time.time()

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  (Tip: GPU training is ~10x faster. Use WSL2 with CUDA for best results.)")
    print()

    # 2. Load & balance dataset (identical logic to Phase 1)
    print("Loading Jigsaw dataset...")
    dataset = load_dataset("tasksource/jigsaw_toxicity", split="train")
    dataset = dataset.select(range(SAMPLE_SIZE))
    df = pd.DataFrame(dataset)

    toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    available_cols = [col for col in toxic_columns if col in df.columns]
    df['is_toxic'] = df[available_cols].max(axis=1).astype(int)

    # Under-sample majority class to balance
    toxic_df = df[df['is_toxic'] == 1]
    clean_df = df[df['is_toxic'] == 0]
    n_toxic = len(toxic_df)
    clean_downsampled = clean_df.sample(n=n_toxic, random_state=42)
    balanced_df = pd.concat([toxic_df, clean_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced dataset: {len(balanced_df)} samples ({balanced_df['is_toxic'].sum()} toxic / {(balanced_df['is_toxic'] == 0).sum()} clean)")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_df['comment_text'], balanced_df['is_toxic'],
        test_size=0.2, random_state=42, stratify=balanced_df['is_toxic']
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print()

    # 4. Tokenizer — load once
    print("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    # 5. PRE-TOKENIZE everything upfront (critical optimization)
    print("Pre-tokenizing all data...")
    train_ids, train_masks = tokenize_all(X_train, tokenizer, MAX_LEN)
    test_ids, test_masks = tokenize_all(X_test, tokenizer, MAX_LEN)

    train_labels = torch.tensor(y_train.values, dtype=torch.long)
    test_labels = torch.tensor(y_test.values, dtype=torch.long)

    train_dataset = PreTokenizedDataset(train_ids, train_masks, train_labels)
    test_dataset = PreTokenizedDataset(test_ids, test_masks, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 6. Model
    print("\nLoading DistilBERT model for fine-tuning...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # FREEZE the DistilBERT base transformer layers.
    # Only the classifier head (pre_classifier + classifier) will train.
    # This prevents overfitting on small datasets and produces well-calibrated
    # probability scores instead of overconfident 0%/100% predictions.
    for param in model.distilbert.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable params: {trainable:,} | Frozen params: {frozen:,}")

    model.to(device)

    # 7. Optimizer & scheduler (only for trainable params)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 8. Training loop
    print(f"\n{'='*55}")
    print(f"  TRAINING — {EPOCHS} epochs, {len(train_loader)} batches/epoch")
    print(f"{'='*55}\n")

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"  Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        epoch_elapsed = time.time() - epoch_start
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Time: {epoch_elapsed:.0f}s")

        val_loss, val_preds, val_labels = evaluate(model, test_loader, device)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"    Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print()

    # 9. Final evaluation
    print(f"{'='*55}")
    print(f"  FINAL EVALUATION")
    print(f"{'='*55}\n")

    _, final_preds, final_labels = evaluate(model, test_loader, device)
    print(f"Accuracy: {accuracy_score(final_labels, final_preds):.4f}\n")
    print("Classification Report:")
    print(classification_report(final_labels, final_preds, target_names=["non-toxic", "toxic"]))
    print("Confusion Matrix:")
    print(confusion_matrix(final_labels, final_preds))

    # 10. Save model + tokenizer
    print(f"\nSaving model to '{OUTPUT_DIR}/'...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    total_elapsed = time.time() - total_start
    print(f"\nTotal training time: {total_elapsed/60:.1f} minutes")
    print(f"Model + tokenizer saved to '{OUTPUT_DIR}/' successfully!")


if __name__ == "__main__":
    main()
