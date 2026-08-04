from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="AI-Powered Contract Intelligence API",
    description="""
REST API for Legal Contract Intelligence.

Features:
- Upload Contract
- Clause Classification
- Semantic Search
- Metadata Extraction
- Health Monitoring
""",
    version="1.1.0"
)

app.include_router(router)


@app.get("/")
def home():
    return {
        "message": "AI-Powered Contract Intelligence API is running"
    }