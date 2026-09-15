"""13_INFRA：AI 竖屏短剧生产平台的基础设施层（API、数据库、队列、存储、编排、可观测性）。

包名以数字开头，不能用普通 ``import`` 语句导入，其他顶层包要通过 ``importlib.import_module("13_INFRA.xxx")``
访问；包内部统一用相对 import。本文件不导出任何符号：API 进程和 worker 进程各自只导入自己需要的子包，
避免在包初始化时把两边的依赖（FastAPI / Celery / boto3 等）互相拖进来。
"""
