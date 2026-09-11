from pydantic import BaseModel, Field
class Sample(BaseModel):
    instruction: str; output: str
class TrainRequest(BaseModel):
    samples: list[Sample] = Field(..., min_length=1)
