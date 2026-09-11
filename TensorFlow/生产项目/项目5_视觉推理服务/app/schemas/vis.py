from pydantic import BaseModel, Field
class ImageRequest(BaseModel):
    image_b64: str = Field(..., min_length=1, description="base64 图像(stub 后端任意字符串即可)")
