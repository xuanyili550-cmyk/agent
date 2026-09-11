from pydantic import BaseModel, Field
class TrainRequest(BaseModel):
    env: str = Field(..., min_length=1)
    steps: int = Field(5, ge=1, le=100)
class EvalRequest(BaseModel):
    env: str = Field(..., min_length=1)
