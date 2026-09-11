"""
API 请求/响应模型（pydantic）：既是【入参校验】(类型/长度不对自动 422)，也是【接口契约】
(自动生成 /docs 文档)。前后端按这里的字段对齐，不会对不上。
"""
from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    # min/max_length 是"粗校验"(挡空、挡超大)；更细的业务上限在 service 层(能给中文报错信息)
    text: str = Field(..., min_length=1, max_length=20000, description="中英混排文本")


class Token(BaseModel):
    text: str
    kind: str                     # en/num/cjk/space/punct
    ipa: str | None = None        # 仅英文词有
    ipa_source: str | None = None # eng_to_ipa/builtin/approx —— approx 前端会标"近似"


class Sentence(BaseModel):
    index: int
    text: str
    tokens: list[Token]


class AnalyzeResponse(BaseModel):
    sentences: list[Sentence]
    stats: dict


class JobCreated(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    status: str                   # queued/running/done/error
    progress: float               # 0~1
    error: str | None = None
    download_url: str | None = None
