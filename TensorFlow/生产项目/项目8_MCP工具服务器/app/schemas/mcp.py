from pydantic import BaseModel, Field
class CallRequest(BaseModel):
    name: str = Field(..., min_length=1)
    args: dict = {}
