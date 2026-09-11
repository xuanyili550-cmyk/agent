from fastapi import APIRouter
from ..schemas.vis import ImageRequest
from ..services import vision
router = APIRouter(prefix="/api", tags=["vision"])
@router.post("/classify")
def classify(req: ImageRequest): return vision.classify(req.image_b64)
@router.post("/detect")
def detect(req: ImageRequest): return vision.detect(req.image_b64)
