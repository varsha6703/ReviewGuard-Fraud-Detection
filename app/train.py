
import argparse
import os
import json
import sys

# Work around the blocked optional pyarrow DLL on Windows.
# This script uses a custom PyTorch training loop.
sys.modules["pyarrow"] = None

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

from app.modeling import DistilBertWithVelocity


class ReviewDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length=128):
        self.velocity = np.clip(
            frame["velocity"].astype(float).to_numpy() / 10.0,
            0,
            1
        )

        self.labels = frame["is_fraud"].astype(int).to_numpy()

        self.enc = tokenizer(
            frame["text"].astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(
                self.enc["input_ids"][i], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                self.enc["attention_mask"][i], dtype=torch.long
            ),
            "velocity": torch.tensor(
                [self.velocity[i]], dtype=torch.float32
            ),
            "labels": torch.tensor(
                self.labels[i], dtype=torch.long
            )
        }


def evaluate(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)

            inputs = {
                key: batch[key].to(device)
                for key in ("input_ids", "attention_mask", "velocity")
            }

            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs.logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    if not all_labels:
        raise ValueError("Evaluation dataset is empty.")

    predictions = (np.array(all_probs) >= 0.5).astype(int)

    # ROC-AUC requires both classes in the evaluation dataset.
    auc = (
        float(roc_auc_score(all_labels, all_probs))
        if len(set(all_labels)) == 2
        else None
    )

    return {
        "loss": total_loss / max(len(loader), 1),
        "auc_roc": auc,
        "fraud_f1": float(
            f1_score(all_labels, predictions, zero_division=0)
        ),
        "probabilities": all_probs,
        "labels": all_labels,
        "predictions": predictions
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train the ReviewGuard fraud review detector"
    )

    parser.add_argument(
        "--data",
        default="data/reviews_dataset.csv"
    )
    parser.add_argument(
        "--output",
        default="artifacts"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of rows for a small test run"
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if args.limit != 0 and args.limit < 20:
        raise ValueError(
            "--limit must be 0 or at least 20 for stratified splitting."
        )

    os.makedirs(args.output, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Training device:", device)

    # Load dataset.
    df = pd.read_csv(args.data)

    required = {"text", "velocity", "is_fraud"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    df = df.dropna(subset=["text", "velocity", "is_fraud"]).copy()
    df["is_fraud"] = df["is_fraud"].astype(int)

    if not set(df["is_fraud"].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 and 1.")

    if df["is_fraud"].nunique() != 2:
        raise ValueError(
            "Dataset must contain both genuine (0) and fraud (1) reviews."
        )

    # Optional stratified sampling for a quick test.
    if args.limit > 0:
        fraud_df = df[df["is_fraud"] == 1]
        genuine_df = df[df["is_fraud"] == 0]

        fraud_count = max(
            10,
            round(args.limit * len(fraud_df) / len(df))
        )
        genuine_count = args.limit - fraud_count

        if fraud_count < 2 or genuine_count < 2:
            raise ValueError(
                "The selected limit must leave at least 2 samples "
                "from each class."
            )

        if (
            fraud_count > len(fraud_df)
            or genuine_count > len(genuine_df)
        ):
            raise ValueError(
                "Not enough samples in the source dataset."
            )

        df = pd.concat(
            [
                genuine_df.sample(
                    n=genuine_count, random_state=42
                ),
                fraud_df.sample(
                    n=fraud_count, random_state=42
                )
            ],
            ignore_index=True
        ).sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into 80% training, 10% validation, 10% test.
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["is_fraud"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["is_fraud"]
    )

    print(
        f"Rows: train={len(train_df)}, "
        f"validation={len(val_df)}, test={len(test_df)}"
    )

    print(
        "Training class counts:",
        train_df["is_fraud"].value_counts().to_dict()
    )

    model_name = "distilbert/distilbert-base-uncased"

    print("Loading tokenizer and pretrained model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DistilBertWithVelocity.from_pretrained(model_name)
    model.to(device)

    train_ds = ReviewDataset(train_df, tokenizer)
    val_ds = ReviewDataset(val_df, tokenizer)
    test_ds = ReviewDataset(test_df, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["is_fraud"].to_numpy()
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5
    )

    # Training loop.
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            labels = batch["labels"].to(device)

            inputs = {
                key: batch[key].to(device)
                for key in ("input_ids", "attention_mask", "velocity")
            }

            optimizer.zero_grad()

            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()
            running_loss += loss.item()

            if step % 10 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{args.epochs}, "
                    f"step {step}/{len(train_loader)}, "
                    f"loss={running_loss / min(10, step % 10 or 10):.4f}"
                )
                running_loss = 0.0

        val_result = evaluate(
            model, val_loader, device, criterion
        )

        print(
            f"Validation: loss={val_result['loss']:.4f}, "
            f"AUC={val_result['auc_roc']}, "
            f"fraud F1={val_result['fraud_f1']:.4f}"
        )

    # Evaluate final model on held-out test data.
    test_result = evaluate(
        model, test_loader, device, criterion
    )

    metrics = {
        "validation_auc_roc": val_result["auc_roc"],
        "validation_fraud_f1": val_result["fraud_f1"],
        "test_auc_roc": test_result["auc_roc"],
        "test_fraud_f1": test_result["fraud_f1"],
        "test_classification_report": classification_report(
            test_result["labels"],
            test_result["predictions"],
            output_dict=True,
            zero_division=0
        )
    }

    # Save trained model and tokenizer.
    output_model = os.path.join(
        args.output, "best_model"
    )
    os.makedirs(output_model, exist_ok=True)

    model.save_pretrained(output_model)
    tokenizer.save_pretrained(output_model)

    with open(
        os.path.join(args.output, "metrics.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    with open(
        os.path.join(args.output, "reference_scores.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            [float(x) for x in test_result["probabilities"]],
            file,
            indent=2
        )

    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
