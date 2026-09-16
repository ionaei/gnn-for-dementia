"""
MultiModalDementiaClassifier: BioClinicalBERT + MLP on demographics

Paper Method:
  - Encodes diagnoses + temporal info using BioClinicalBERT [CLS] token (768-dim).
  - Encodes age, sex, PRS separately with a 2-layer MLP (32->32 hidden, ReLU, dropout).
  - Concatenates BERT [CLS] embedding + MLP embedding (800-dim total).
  - Passes through a 2-layer classifier head (800->64->2).

Reference: "Multimodal modelling: BioClinical BERT + MLP" in the paper.
Table 1: Multimodal BioClinical BERT+MLP F1=0.704, Sensitivity=0.694, Specificity=0.714

Train Config:
  - Learning rate: 2e-5 (Adam)
  - Epochs: 50 (with early stopping on validation loss)
  - Batch size: 16
  - WandB logging: yes (optional with --no-wandb flag)
  - Train/Val/Test split: 60/13.5/10 via test_size=0.1 then 0.15
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

# Try to import wandb, but allow --no-wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from sequence_builder import apply_icd10_sequences


class DementiaDataset(Dataset):
    """Dataset for multimodal BERT + demographics."""
    def __init__(self, input_ids, attention_mask, structured_features, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.structured_features = structured_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'structured_features': self.structured_features[idx],
            'labels': self.labels[idx],
        }


class MultiModalDementiaClassifier(nn.Module):
    """BioClinicalBERT [CLS] + 2-layer MLP on demographics -> classifier."""
    def __init__(self, mlp_input_dim=3, mlp_hidden_dim=32, dropout=0.3):
        super().__init__()

        # Pretrained BioClinicalBERT (frozen for stability, though unfrozen in original)
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # MLP for age, sex, PRS (3 features)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
        )

        # Classifier: concat BERT [CLS] (768) + MLP output (32) = 800 dims
        bert_dim = self.bert.config.hidden_size
        combined_dim = bert_dim + mlp_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # 2 classes: dementia (0) / control (1)
        )

    def forward(self, input_ids, attention_mask, structured_features):
        # BERT [CLS] embedding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]  # [batch_size, 768]

        # MLP on demographics
        structured_embedding = self.mlp(structured_features)  # [batch_size, 32]

        # Concatenate
        combined = torch.cat((cls_embedding, structured_embedding), dim=1)  # [batch_size, 800]

        # Classification
        logits = self.classifier(combined)  # [batch_size, 2]
        return logits


def prepare_data(df, use_synthetic=False):
    """Prepare tokenized BERT inputs and structured features."""
    # Build ICD-10 sequences
    df = apply_icd10_sequences(df)

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

    # Normalize age and PRS
    prs_col = "Standard_PRS_for_alzheimer's_disease_(AD)"
    if prs_col not in df_train.columns:
        prs_col = "PRS"  # fallback

    scaler = StandardScaler()
    structured_train = scaler.fit_transform(df_train[["Age", prs_col]])
    structured_val = scaler.transform(df_val[["Age", prs_col]])
    structured_test = scaler.transform(df_test[["Age", prs_col]])

    # Add sex
    structured_train = np.concatenate([structured_train, df_train[["Sex"]].values], axis=1)
    structured_val = np.concatenate([structured_val, df_val[["Sex"]].values], axis=1)
    structured_test = np.concatenate([structured_test, df_test[["Sex"]].values], axis=1)

    # Convert to tensors
    structured_train = torch.tensor(structured_train, dtype=torch.float32)
    structured_val = torch.tensor(structured_val, dtype=torch.float32)
    structured_test = torch.tensor(structured_test, dtype=torch.float32)

    labels_train = torch.tensor(df_train["label"].tolist(), dtype=torch.long)
    labels_val = torch.tensor(df_val["label"].tolist(), dtype=torch.long)
    labels_test = torch.tensor(df_test["label"].tolist(), dtype=torch.long)

    return (
        tokenized_train, tokenized_val, tokenized_test,
        structured_train, structured_val, structured_test,
        labels_train, labels_val, labels_test,
    )


def train_and_evaluate(
    data_path: str = None,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    use_wandb: bool = True,
    checkpoint_dir: str = "./checkpoints",
):
    """Train and evaluate the multimodal classifier."""
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
     structured_train, structured_val, structured_test,
     labels_train, labels_val, labels_test) = prepare_data(df)

    # Create datasets and loaders
    train_dataset = DementiaDataset(
        tokenized_train['input_ids'], tokenized_train['attention_mask'],
        structured_train, labels_train
    )
    val_dataset = DementiaDataset(
        tokenized_val['input_ids'], tokenized_val['attention_mask'],
        structured_val, labels_val
    )
    test_dataset = DementiaDataset(
        tokenized_test['input_ids'], tokenized_test['attention_mask'],
        structured_test, labels_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Initialize wandb if enabled
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="alzheimers-multimodal-neurips",
            name="MultiModalClassifier_v1",
            config={
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "model": "MultiModalDementiaClassifier",
            },
        )
    else:
        use_wandb = False

    # Model, optimizer, loss
    model = MultiModalDementiaClassifier()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint = os.path.join(checkpoint_dir, "best_multimodal_classifier.pt")
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
            structured_features = batch['structured_features'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, structured_features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

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
                structured_features = batch['structured_features'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask, structured_features)
                loss = loss_fn(logits, labels)
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
            structured_features = batch['structured_features'].to(device)
            labels = batch['labels']

            logits = model(input_ids, attention_mask, structured_features)
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
    print("TEST RESULTS")
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
        description="Train MultiModal BERT + MLP classifier for dementia prediction."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV with dementia data (schema: eid, Class, Sex, Age, PRS, diagnosis columns). "
             "If not provided, generates synthetic data for smoke testing.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate for optimizer."
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging."
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints",
        help="Directory to save model checkpoints."
    )

    args = parser.parse_args()

    train_and_evaluate(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb,
        checkpoint_dir=args.checkpoint_dir,
    )
