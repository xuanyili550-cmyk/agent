"""自定义异常 + 全局处理:统一错误响应,不泄露栈。"""
from fastapi import Request
from fastapi.responses import JSONResponse
from .logging import get_logger
log = get_logger("exceptions")


class AppError(Exception):
    status_code = 400
    def __init__(self, message, code="bad_request", status_code=None):
        self.message, self.code = message, code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(message)


class ValidationError(AppError):
    def __init__(self, m): super().__init__(m, "validation_error", 422)


class NotFoundError(AppError):
    def __init__(self, m): super().__init__(m, "not_found", 404)


def register_exception_handlers(app):
    @app.exception_handler(AppError)
    async def _a(req: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})
    @app.exception_handler(Exception)
    async def _u(req: Request, exc: Exception):
        log.exception("unhandled %s", req.url.path)
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "服务器内部错误"}})
