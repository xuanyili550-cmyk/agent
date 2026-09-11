from pydantic import BaseModel, Field
class Gen3DRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
