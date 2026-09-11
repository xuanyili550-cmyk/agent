from pydantic import BaseModel, Field
class ActRequest(BaseModel):
    observation: list[float] = Field(..., min_length=1, description="观测向量(关节/目标位置等)")
