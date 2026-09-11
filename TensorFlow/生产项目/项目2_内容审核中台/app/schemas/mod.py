from pydantic import BaseModel, Field
class ModerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
class ModerateResponse(BaseModel):
    grade: str; risk_score: float; sensitive: list[str]; pii: dict; masked: str
class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
