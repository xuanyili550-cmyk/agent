# 常用命令。生产部署见 13_INFRA/docker/docker-compose.yml；本地开发不需要 Postgres/Redis。
PY ?= python3

.PHONY: install lint test test-fast demo api worker migrate docker-build docker-up ci

install:
	$(PY) -m pip install -r requirements.txt -r 13_INFRA/requirements-infra.txt ruff

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest tests/test_retry_ladder.py tests/test_story_engine.py tests/test_tool_agent.py 02_STORY_ENGINE/tests

demo:
	$(PY) 02_STORY_ENGINE/demo.py
	$(PY) 13_INFRA/api/demo.py

# 本地起 API（SQLite + eager 队列，不需要 Redis）。生产用 docker-compose。
# 起来后浏览器打开 http://localhost:8000/ 就是 Web 控制台（API key 默认 dev-key）。
api:
	@echo "Web 控制台: http://localhost:8000/   API 文档: http://localhost:8000/docs   API key: $${API_KEYS:-dev-key}"
	APP_ENV=dev CELERY_TASK_ALWAYS_EAGER=true CELERY_BROKER_URL=memory:// CELERY_RESULT_BACKEND=cache+memory:// \
	LLM_PROVIDER=$${LLM_PROVIDER:-mock} QC_BACKEND=$${QC_BACKEND:-none} API_KEYS=$${API_KEYS:-dev-key} \
	$(PY) -m uvicorn 13_INFRA.api.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	$(PY) -m celery -A 13_INFRA.workers.celery_app worker -Q llm,image,video,tts,lipsync,qc --loglevel=INFO

migrate:
	cd 13_INFRA/database && $(PY) -m alembic upgrade head

docker-build:
	docker compose -f 13_INFRA/docker/docker-compose.yml build

docker-up:
	docker compose -f 13_INFRA/docker/docker-compose.yml up -d

ci: lint test
