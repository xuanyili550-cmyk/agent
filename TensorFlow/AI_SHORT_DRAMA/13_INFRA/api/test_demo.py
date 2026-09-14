"""pytest entrypoint for the same CRUD smoke test as demo.py (`pytest 13_INFRA/api/test_demo.py`)."""

from .demo import run_demo


def test_crud_demo() -> None:
    run_demo()
