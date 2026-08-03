import os
import numpy as np
import faiss

print("=" * 60)
print("Building FAISS Vector Database")
print("=" * 60)

# Load embeddings
embeddings = np.load("vector_db/cuad_embeddings.npy")
print(f"Loaded Embeddings: {embeddings.shape}")

# Convert to float32 (required by FAISS)
embeddings = embeddings.astype("float32")

# Create index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add vectors
index.add(embeddings)

print(f"Indexed Vectors: {index.ntotal}")

# Save index
os.makedirs("vector_db", exist_ok=True)
faiss.write_index(index, "vector_db/faiss_index.bin")

print("\nFAISS Index Saved Successfully!")
print("Saved to: vector_db/faiss_index.bin")