"""
Text-Only BioClinicalBERT Classifier

Paper Method:
  - Uses only BioClinicalBERT for sequence classification (no separate demographics MLP).
  - Input text includes demographics (sex, age, PRS) + diagnosis sequences with temporal info.
  - Fine-tunes AutoModelForSequenceClassification on the text sequence alone.

Reference: "LLM: BioClinical BERT" in the paper.
Table 1: BioClinical BERT pure F1=0.707, Sensitivity=0.726, Specificity=0.670

Train Config:
  - Learning rate: 2e-6 (from notebook observations; originally 2e-5 but we adapt).
  - Epochs: up to 20 (with early stopping; 15-20 in original notebooks).
  - Batch size: 16
  - WandB logging: yes (optional with --no-wandb flag).
  - Train/Val/Test split: 60/13.5/10 via test_size=0.1 then 0.15.
  - Gradient clipping: max_norm=1.0
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import pandas as pd

# Try to import wandb, but allow --no-wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from sequence_builder import apply_icd10_sequences


class TextOnlyDementiaDataset(Dataset):
    """Dataset for text-only BERT classification."""
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }


def prepare_data(df):
    """Prepare tokenized BERT inputs (text-only, no structured features)."""
    # Build ICD-10 sequences (including demographics as part of text)
    df = apply_icd10_sequences(df, include_demographics=True)

    # Label mapping
    label_map = {"Dementia": 0, "Control": 1}
    df["label"] = df["Class"].map(label_map)

    # Train/val/test split: 60/13.5/10
    df_train_val, df_test = train_test_split(
        df, test_size=0.1, stratify=df["label"], random_state=42
    )
    df_train, df_val = train_test_split(
        df_train_val, test_size=0.15, stratify=df_train_val["label"], random_state=42
    )

    # Tokenize sequences
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tokenized_train = tokenizer(
        df_train["icd10_sequence"].tolist(),
        padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    tokenized_val = tokenizer(
        df_val["icd10_sequence"].tolist(),
        padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    tokenized_test = tokenizer(
        df_test["icd10_sequence"].tolist(),
        padding=True, truncation=True, max_length=512, return_tensors="pt"
    )

    labels_train = torch.tensor(df_train["label"].tolist(), dtype=torch.long)
    labels_val = torch.tensor(df_val["label"].tolist(), dtype=torch.long)
    labels_test = torch.tensor(df_test["label"].tolist(), dtype=torch.long)

    return (
        tokenized_train, tokenized_val, tokenized_test,
        labels_train, labels_val, labels_test,
    )


def train_and_evaluate(
    data_path: str = None,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 2e-6,
    use_wandb: bool = True,
    checkpoint_dir: str = "./checkpoints",
    max_grad_norm: float = 1.0,
):
    """Train and evaluate the text-only BERT classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    if data_path is None:
        print("No data path provided; generating synthetic data...")
        from generate_synthetic_data import generate_synthetic_data
        data_path = "/tmp/synthetic_dementia.csv"
        generate_synthetic_data(data_path, n_patients=60)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    # Prepare data
    (tokenized_train, tokenized_val, tokenized_test,
     labels_train, labels_val, labels_test) = prepare_data(df)

    # Create datasets and loaders
    train_dataset = TextOnlyDementiaDataset(
        tokenized_train['input_ids'], tokenized_train['attention_mask'],
        labels_train
    )
    val_dataset = TextOnlyDementiaDataset(
        tokenized_val['input_ids'], tokenized_val['attention_mask'],
        labels_val
    )
    test_dataset = TextOnlyDementiaDataset(
        tokenized_test['input_ids'], tokenized_test['attention_mask'],
        labels_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Initialize wandb if enabled
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="alzheimers-multimodal-neurips",
            name="BioClinicalBERT_pure_v1",
            config={
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "model": "Bio_ClinicalBERT_pure",
                "max_grad_norm": max_grad_norm,
            },
        )
    else:
        use_wandb = False

    # Model, optimizer, scheduler, loss
    model = AutoModelForSequenceClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT", num_labels=2
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint = os.path.join(checkpoint_dir, "best_text_only_classifier.pt")
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        # Validation
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        print(f"Epoch {epoch+1}/{epochs}: Train Loss {avg_train_loss:.4f}, "
              f"Val Loss {avg_val_loss:.4f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "learning_rate": scheduler.get_last_lr()[0],
            })

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_checkpoint)
            print(f"  -> Saved best model to {best_checkpoint}")

    # Test evaluation
    model.load_state_dict(torch.load(best_checkpoint))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())  # prob of class 1 (control)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds, pos_label=0)  # recall class 0 (dementia)
    specificity = recall_score(all_labels, all_preds, pos_label=1)  # recall class 1 (control)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print("\n" + "="*60)
    print("TEST RESULTS (Text-Only BioClinicalBERT)")
    print("="*60)
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Sensitivity (Dementia recall): {sensitivity:.4f}")
    print(f"Specificity (Control recall):  {specificity:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"AUCROC:     {auc:.4f}")
    print("="*60)

    if use_wandb:
        wandb.log({
            "test/accuracy": accuracy,
            "test/sensitivity": sensitivity,
            "test/specificity": specificity,
            "test/f1": f1,
            "test/auc": auc,
        })
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train text-only BioClinicalBERT classifier for dementia prediction."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV with dementia data. If not provided, generates synthetic data.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-6, help="Learning rate for optimizer."
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging."
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints",
        help="Directory to save model checkpoints."
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="Maximum gradient norm for clipping."
    )

    args = parser.parse_args()

    train_and_evaluate(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb,
        checkpoint_dir=args.checkpoint_dir,
        max_grad_norm=args.max_grad_norm,
    )
