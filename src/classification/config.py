"""
Configuration file for Legal Clause Classification
"""

# -----------------------------
# Model Configuration
# -----------------------------

MODEL_NAME = "distilroberta-base"

# -----------------------------
# Dataset Paths
# -----------------------------

# Official CUAD dataset
DATASET_PATH = "data/raw/CUAD_v1.json"

# Directory to store processed dataset
PROCESSED_DATASET_PATH = "data/processed"

# -----------------------------
# Model Save Path
# -----------------------------

MODEL_SAVE_PATH = "models/legal_clause_classifier"

# -----------------------------
# Training Parameters
# -----------------------------

MAX_LENGTH = 256

BATCH_SIZE = 4

LEARNING_RATE = 2e-5

EPOCHS = 3

RANDOM_SEED = 42

# -----------------------------
# Dataset Split
# -----------------------------

TRAIN_SPLIT = 0.8

VALIDATION_SPLIT = 0.1

TEST_SPLIT = 0.1
# ============================================================
# Training Dataset Configuration
# ============================================================

USE_SUBSET = True

MAX_TRAIN_SAMPLES = 1000