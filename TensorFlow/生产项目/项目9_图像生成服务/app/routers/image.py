from fastapi import APIRouter
from ..schemas.img import GenRequest
from ..services import image
router = APIRouter(prefix="/api", tags=["image"])
@router.post("/generate")
def generate(req: GenRequest): return image.generate(req.prompt, req.steps, req.guidance)
