from pydantic import BaseModel, Field
class GenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    steps: int = Field(25, ge=1, le=100)
    guidance: float = Field(7.5, ge=0, le=20)
