import os
import sys
import pickle
import numpy as np
import faiss

# Allow importing embedder.py
sys.path.append(os.path.dirname(__file__))

from embedder import ContractEmbedder

print("=" * 60)
print("Semantic Contract Search")
print("=" * 60)

# -------------------------------------------------------
# Load FAISS Index
# -------------------------------------------------------

index = faiss.read_index("vector_db/faiss_index.bin")

# -------------------------------------------------------
# Load Metadata
# -------------------------------------------------------

with open("vector_db/cuad_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"Indexed Clauses : {len(metadata)}")

# -------------------------------------------------------
# Load Embedding Model
# -------------------------------------------------------

embedder = ContractEmbedder()

print("\nSystem Ready!")

# -------------------------------------------------------
# Search Loop
# -------------------------------------------------------

while True:

    print("\nEnter search query (or type 'exit')")

    query = input(">> ").strip()

    if query.lower() == "exit":
        print("\nExiting Semantic Search...")
        break

    if not query:
        continue

    query_embedding = embedder.generate_embedding(query)

    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    distances, indices = index.search(query_embedding, 5)

    print("\nTop 5 Results")
    print("=" * 60)

    for rank, idx in enumerate(indices[0], start=1):

        clause = metadata[idx]

        similarity = 1 / (1 + distances[0][rank - 1])

        print(f"\nResult {rank}")
        print("-" * 60)
        print(f"Label      : {clause['label']}")
        print(f"Similarity : {similarity:.4f}")
        print(f"Clause:\n{clause['text']}")