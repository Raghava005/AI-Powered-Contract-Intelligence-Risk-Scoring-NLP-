from pathlib import Path
import shutil
import pickle

import faiss
import numpy as np

from fastapi import APIRouter, File, UploadFile, HTTPException

from src.api.schemas import SearchRequest, PredictionRequest
from src.classification.predict import predict_clause
from src.vector_search.embedder import ContractEmbedder

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------
# Load FAISS Once
# ---------------------------------------------------

index = faiss.read_index("vector_db/faiss_index.bin")

with open("vector_db/cuad_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

embedder = ContractEmbedder()


# ---------------------------------------------------
# Health
# ---------------------------------------------------

@router.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Contract Intelligence API"
    }


@router.get("/version")
def version():
    return {
        "version": "1.0.0"
    }


# ---------------------------------------------------
# Upload + Metadata Extraction
# ---------------------------------------------------

@router.post("/upload")
async def upload_contract(file: UploadFile = File(...)):

    extension = Path(file.filename).suffix.lower()

    if extension not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX and TXT files are allowed."
        )

    destination = UPLOAD_DIR / file.filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---------------------------------------------------
    # Dummy Metadata Extraction
    # ---------------------------------------------------

    try:

        if extension == ".txt":

            text = destination.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        else:
            # Placeholder for PDF/DOCX extraction
            text = (
                "This agreement shall remain valid for one year. "
                "Either party may terminate this agreement by giving "
                "30 days written notice."
            )

        text = text.strip()

        contract_length = len(text)

        if len(text) > 512:
            sample_text = text[:512]
        else:
            sample_text = text

        label, confidence = predict_clause(sample_text)

        return {
            "filename": file.filename,
            "status": "processed",
            "location": str(destination),
            "contract_length": contract_length,
            "predicted_clause": label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------
# Clause Prediction
# ---------------------------------------------------

@router.post("/predict")
def predict(request: PredictionRequest):

    label, confidence = predict_clause(request.text)

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }


# ---------------------------------------------------
# Semantic Search
# ---------------------------------------------------

@router.post("/search")
def semantic_search(request: SearchRequest):

    embedding = embedder.generate_embedding(request.query)

    embedding = np.array([embedding], dtype="float32")

    distances, indices = index.search(embedding, 5)

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        similarity = float(1 / (1 + distance))

        results.append(
            {
                "label": metadata[idx]["label"],
                "similarity": round(similarity, 4),
                "text": metadata[idx]["text"]
            }
        )

    return {
        "query": request.query,
        "results": results
    }