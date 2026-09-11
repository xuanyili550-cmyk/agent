"""
视频导出路由（异步 job 三段式：提交 → 轮询 → 下载）。
 为什么分三个接口：导出慢，不能一个请求干等到底(会超时、没进度)。所以：
   · POST /api/export       ：提交任务，立刻返回 job_id(真正合成在后台线程池里跑)。
   · GET  /api/jobs/{id}    ：轮询状态/进度(前端进度条据此更新)；done 时带上 download_url。
   · GET  /api/download/{id}：任务完成后下载 mp4。
"""
import os

from fastapi import APIRouter
from fastapi.responses import FileResponse

from ..core.exceptions import NotFoundError
from ..schemas.audio import JobCreated, JobStatus, TextRequest
from ..services import jobs, video

router = APIRouter(prefix="/api", tags=["video"])


@router.post("/export", response_model=JobCreated)
def api_export(req: TextRequest):
    """提交视频导出任务(异步)，返回 job_id；用 /api/jobs/{id} 轮询进度。"""
    def _task(progress_cb, text):
        # 在后台线程里跑：build 内部逐句渲染并通过 progress_cb 上报进度
        out = os.path.join(video.get_settings().cache_dir, "export.mp4")
        return video.build(text, out, progress_cb=progress_cb)

    job_id = jobs.submit(_task, req.text)
    return JobCreated(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def api_job(job_id: str):
    """查任务状态；完成时附下载链接。任务不存在/已过期 → 404。"""
    j = jobs.get(job_id)
    if not j:
        raise NotFoundError("任务不存在或已过期。")
    dl = f"/api/download/{job_id}" if j["status"] == "done" else None
    return JobStatus(status=j["status"], progress=j["progress"], error=j["error"], download_url=dl)


@router.get("/download/{job_id}")
def api_download(job_id: str):
    """下载导出的 mp4(仅在任务 done 且产物存在时)。"""
    j = jobs.get(job_id)
    if not j or j["status"] != "done" or not j["result"]:
        raise NotFoundError("视频未就绪。")
    return FileResponse(j["result"], media_type="video/mp4", filename="双语配音.mp4")
