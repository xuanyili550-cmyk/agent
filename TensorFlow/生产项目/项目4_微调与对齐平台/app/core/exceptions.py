from fastapi import Request
from fastapi.responses import JSONResponse
from .logging import get_logger
log=get_logger("exc")
class AppError(Exception):
    status_code=400
    def __init__(self,m,code="bad_request",status_code=None):
        self.message,self.code=m,code
        if status_code is not None:self.status_code=status_code
        super().__init__(m)
class ValidationError(AppError):
    def __init__(self,m): super().__init__(m,"validation_error",422)
class NotFoundError(AppError):
    def __init__(self,m): super().__init__(m,"not_found",404)
def register_exception_handlers(app):
    @app.exception_handler(AppError)
    async def _a(r:Request,e:AppError): return JSONResponse(status_code=e.status_code,content={"error":{"code":e.code,"message":e.message}})
    @app.exception_handler(Exception)
    async def _u(r:Request,e:Exception):
        log.exception("unhandled %s",r.url.path)
        return JSONResponse(status_code=500,content={"error":{"code":"internal_error","message":"服务器内部错误"}})
