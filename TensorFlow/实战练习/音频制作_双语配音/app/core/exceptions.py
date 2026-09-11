"""自定义异常 + 全局处理器：统一错误响应，不泄露栈；区分用户错(4xx)与系统错(5xx)。"""
from fastapi import Request
from fastapi.responses import JSONResponse

from .logging import get_logger

log = get_logger("exceptions")


class AppError(Exception):
    status_code = 400

    def __init__(self, message: str, code: str = "bad_request", status_code: int | None = None):
        self.message = message
        self.code = code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(message)


class ValidationError(AppError):
    def __init__(self, message):
        super().__init__(message, code="validation_error", status_code=422)


class DependencyError(AppError):
    """系统依赖缺失(如 ffmpeg/say/字体)。"""
    def __init__(self, message):
        super().__init__(message, code="dependency_error", status_code=503)


class NotFoundError(AppError):
    def __init__(self, message):
        super().__init__(message, code="not_found", status_code=404)


def register_exception_handlers(app):
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500,
                            content={"error": {"code": "internal_error", "message": "服务器内部错误"}})
