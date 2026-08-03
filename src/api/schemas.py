from pydantic import BaseModel


class SearchRequest(BaseModel):

    query: str


class PredictionResponse(BaseModel):

    label: str

    confidence: float