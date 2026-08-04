from pydantic import BaseModel


class PredictionRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str