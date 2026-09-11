from fastapi import APIRouter
from ..schemas.spx import ASRRequest, TTSRequest
from ..services import speech
router = APIRouter(prefix="/api", tags=["speech"])
@router.post("/asr")
def asr(req: ASRRequest): return speech.asr(req.audio_b64)
@router.post("/tts")
def tts(req: TTSRequest): return speech.tts(req.text)
