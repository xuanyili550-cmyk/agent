"""demo.py 里那套 CRUD 冒烟测试的 pytest 入口（``pytest 13_INFRA/api/test_demo.py``）。

demo.py 既能直接 ``python`` 运行也能被 pytest 收集：这里只是把 ``run_demo`` 包成一个测试函数，
让 CI 和本地开发共用同一份端到端检查逻辑，不用维护两套。
"""

from .demo import run_demo


def test_crud_demo() -> None:
    """跑完整的 API CRUD + 任务入队/查状态冒烟流程；任何断言失败即测试失败。"""
    run_demo()
