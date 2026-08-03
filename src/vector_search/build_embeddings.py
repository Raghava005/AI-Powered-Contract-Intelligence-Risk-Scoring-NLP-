import os
import pickle
import numpy as np

import sys
import os

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "classification"))

from dataset import load_cuad_dataset
from config import DATASET_PATH
from embedder import ContractEmbedder


print("=" * 60)
print("Generating Contract Embeddings")
print("=" * 60)


# Load dataset
dataset = load_cuad_dataset(DATASET_PATH)

print(f"Loaded {len(dataset)} clauses")


# Collect clause texts
texts = [sample["text"] for sample in dataset]


# Load embedding model
embedder = ContractEmbedder()


# Generate embeddings
embeddings = embedder.generate_embeddings(texts)


print(f"\nEmbedding Shape : {embeddings.shape}")


# Create output folder
os.makedirs("vector_db", exist_ok=True)


# Save embeddings
np.save(
    "vector_db/cuad_embeddings.npy",
    embeddings
)


# Save metadata
with open(
    "vector_db/cuad_metadata.pkl",
    "wb"
) as f:

    pickle.dump(dataset, f)


print("\nEmbeddings Saved Successfully!")

print("Location : vector_db/")