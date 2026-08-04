import os
import json
import random
import numpy as np
import torch

from src.classification.dataset import  Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

from dataset import load_cuad_dataset, encode_labels
from config import *


# ============================================================
# Reproducibility
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("Loading Official CUAD Dataset...")
print("=" * 60)

dataset = load_cuad_dataset(DATASET_PATH)
dataset, label2id, id2label = encode_labels(dataset)

print(f"Total Samples : {len(dataset)}")
print(f"Total Labels  : {len(label2id)}")


# ============================================================
# Use Training Subset (same as train.py)
# ============================================================

if USE_SUBSET:
    dataset = random.sample(
        dataset,
        min(MAX_TRAIN_SAMPLES, len(dataset))
    )

    print(f"\nUsing Subset : {len(dataset)} samples")


# ============================================================
# HuggingFace Dataset
# ============================================================

hf_dataset = Dataset.from_list(dataset)

hf_dataset = hf_dataset.rename_column(
    "label_id",
    "labels"
)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)


def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


tokenized_dataset = hf_dataset.map(tokenize)

tokenized_dataset = tokenized_dataset.remove_columns(
    ["text", "label"]
)


# ============================================================
# Dataset Split
# ============================================================

train_test = tokenized_dataset.train_test_split(
    test_size=(1 - TRAIN_SPLIT),
    seed=RANDOM_SEED
)

temp = train_test["test"].train_test_split(
    test_size=TEST_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT),
    seed=RANDOM_SEED
)

test_dataset = temp["test"]

print(f"\nTest Samples : {len(test_dataset)}")


# ============================================================
# Load Model
# ============================================================

print("\nLoading Trained Model...\n")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_SAVE_PATH
)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics
)


# ============================================================
# Evaluation
# ============================================================

print("=" * 60)
print("Evaluating Model...")
print("=" * 60)

results = trainer.predict(test_dataset)

predictions = np.argmax(
    results.predictions,
    axis=1
)

labels = results.label_ids

metrics = compute_metrics(
    (results.predictions, labels)
)


# ============================================================
# Print Results
# ============================================================

print("\nEvaluation Results")
print("=" * 60)

print(f"Accuracy  : {metrics['accuracy']:.4f}")
print(f"Precision : {metrics['precision']:.4f}")
print(f"Recall    : {metrics['recall']:.4f}")
print(f"F1 Score  : {metrics['f1']:.4f}")


print("\nClassification Report")
print("=" * 60)

print(
    classification_report(
        labels,
        predictions,
        zero_division=0
    )
)


# ============================================================
# Save Results
# ============================================================

os.makedirs("results", exist_ok=True)

with open(
    "results/evaluation_results.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


print("\nResults saved to:")
print("results/evaluation_results.json")