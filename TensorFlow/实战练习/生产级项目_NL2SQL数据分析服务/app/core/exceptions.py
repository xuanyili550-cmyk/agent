# 【构建 3/16 · 地基】依赖 logging(2)；供 services/main 用
"""自定义异常 + 全局异常处理(统一错误响应，不泄露栈)。"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


class AppError(Exception):
    """业务异常基类。"""
    status_code = 400
    code = "app_error"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class InvalidSQLError(AppError):
    status_code = 422
    code = "invalid_sql"


class SQLExecutionError(AppError):
    status_code = 500
    code = "sql_execution_error"


class LLMBackendError(AppError):
    status_code = 502
    code = "llm_backend_error"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        logger.warning("业务异常: %s", exc.message)
        return JSONResponse(status_code=exc.status_code,
                            content={"ok": False, "error": {"code": exc.code, "message": exc.message},
                                     "request_id": request_id_ctx.get()})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("未处理异常")               # 记录栈，但不返回给客户端
        return JSONResponse(status_code=500,
                            content={"ok": False, "error": {"code": "internal_error", "message": "内部错误"},
                                     "request_id": request_id_ctx.get()})
