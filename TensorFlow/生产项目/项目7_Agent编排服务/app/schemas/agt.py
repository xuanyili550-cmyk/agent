from pydantic import BaseModel, Field
class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
