"""
音频/分析路由。
 · POST /api/analyze：文本 → 句子+token+英文 IPA。前端渲染音标、逐句高亮全靠它(纯计算，快)。
 · POST /api/tts    ：整段合成 mp3 返回(播放用)。相同文本命中缓存秒回；带 Cache-Control 让浏览器也缓存。
入参用 pydantic(TextRequest)自动校验；业务上限(长度/句数)在 service 层校验并抛 422。
"""
import os

from fastapi import APIRouter
from fastapi.responses import FileResponse

from ..schemas.audio import AnalyzeResponse, TextRequest
from ..services import tts
from ..services.tokenizer import analyze

router = APIRouter(prefix="/api", tags=["audio"])


@router.post("/analyze", response_model=AnalyzeResponse)
def api_analyze(req: TextRequest):
    """文本 → 句子 + token + 英文 IPA(供前端渲染音标/逐句高亮)。"""
    return analyze(req.text)


@router.post("/tts")
def api_tts(req: TextRequest):
    """整段合成 mp3 返回(播放用)。相同文本命中缓存秒回。"""
    mp3 = tts.synth_mp3(req.text)
    return FileResponse(mp3, media_type="audio/mpeg",
                        headers={"Cache-Control": "public, max-age=3600"},
                        filename=os.path.basename(mp3))
