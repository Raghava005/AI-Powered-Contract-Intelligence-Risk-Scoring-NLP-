from src.classification.dataset import  Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

import random
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
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
# Convert to HuggingFace Dataset
# ============================================================
dataset = load_cuad_dataset(DATASET_PATH)

dataset, label2id, id2label = encode_labels(dataset)

print(f"Total Samples : {len(dataset)}")
print(f"Total Labels  : {len(label2id)}")
# ============================================================
# Use Subset for Training
# ============================================================

if USE_SUBSET:
    dataset = dataset[:MAX_TRAIN_SAMPLES]

    print("\nUsing Training Subset")
    print(f"Training Samples : {len(dataset)}")
hf_dataset = Dataset.from_list(dataset)

hf_dataset = hf_dataset.rename_column("label_id", "labels")


# ============================================================
# Tokenizer
# ============================================================

print("\nLoading Tokenizer...\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


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

train_dataset = train_test["train"]

eval_dataset = temp["train"]

test_dataset = temp["test"]


print("=" * 60)
print("Dataset Split")
print("=" * 60)

print(f"Training Samples   : {len(train_dataset)}")
print(f"Validation Samples : {len(eval_dataset)}")
print(f"Test Samples       : {len(test_dataset)}")


# ============================================================
# Model
# ============================================================

print("\nLoading RoBERTa Model...\n")

model = AutoModelForSequenceClassification.from_pretrained(

    MODEL_NAME,

    num_labels=len(label2id),

    id2label=id2label,

    label2id=label2id
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

    accuracy = accuracy_score(labels, predictions)

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1
    }

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing Device : {device.upper()}")
# ============================================================
# Training Arguments
# ============================================================

training_args = TrainingArguments(

    output_dir="./results",

    

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    save_total_limit=2,

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    logging_dir="./logs",

    logging_strategy="steps",

    logging_steps=25,

    report_to="none",

    seed=RANDOM_SEED
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    compute_metrics=compute_metrics
)


# ============================================================
# Train
# ============================================================

print("\n" + "=" * 60)
print("Training Started")
print("=" * 60)

trainer.train()


# ============================================================
# Final Evaluation
# ============================================================

print("\nEvaluating Best Model...\n")

results = trainer.evaluate()

print("\nEvaluation Results")

for key, value in results.items():

    print(f"{key} : {value}")


# ============================================================
# Save Model
# ============================================================

trainer.save_model(MODEL_SAVE_PATH)

tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("\n" + "=" * 60)
print("Training Completed Successfully")
print("=" * 60)

print(f"\nModel saved at:\n{MODEL_SAVE_PATH}")