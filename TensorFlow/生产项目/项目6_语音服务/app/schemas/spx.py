from pydantic import BaseModel, Field
class ASRRequest(BaseModel):
    audio_b64: str = Field(..., min_length=1)
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
