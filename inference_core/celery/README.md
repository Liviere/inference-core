## Environment variables

- CELERY_BROKER_URL (e.g. redis://localhost:6379/0)
- CELERY_RESULT_BACKEND (e.g. redis://localhost:6379/1)
- DEBUG_CELERY=1 enables debugpy in the worker on port 5678

Sample values are provided in `.env.example`.

## Local development (without Docker)

1. Start Redis (locally or via Docker)
   - Locally installed Redis, or
   - Using Docker: `docker run -p 6379:6379 redis:7-alpine`
2. Start the API server
   - `poetry run fastapi dev`
3. Start a Celery worker in another terminal
   - `poetry run celery -A inference_core.celery.celery_main:celery_app worker --pool=gevent --autoscale=200,10 --loglevel=info --queues=default`
4. Optional: start Flower (Celery monitoring UI)
   - `poetry run celery -A inference_core.celery.celery_main:celery_app flower --port=5555`

Notes

- If you set `DEBUG_CELERY=1`, the worker will wait for a debugger to attach on port 5678.
- Queues and routing are configurable in `inference_core/celery/config.py`.

## Using Docker

The base compose file already wires up: API, Redis, Celery worker, and Flower.

- Bring everything up (example with SQLite):
  - `docker compose -f docker-compose.base.yml -f docker/docker-compose.sqlite.yml up -d --build`
- Flower UI: http://localhost:5555
- Redis: `localhost:6379` (mapped from the container by default)

You can swap the database layer by using the MySQL or PostgreSQL compose overlays in `docker/`.

## Task management API

The API exposes endpoints for inspecting and controlling Celery tasks:

- GET `/api/v1/tasks/health` – quick health of the task system and active workers
- GET `/api/v1/tasks/{task_id}/status` – current state and metadata
- GET `/api/v1/tasks/{task_id}/result` – fetches task result (waits up to an optional timeout)
- DELETE `/api/v1/tasks/{task_id}` – attempts to cancel a task
- GET `/api/v1/tasks/active` – lists active/scheduled/reserved tasks per worker
- GET `/api/v1/tasks/workers/stats` – worker stats, ping, and registered tasks

Example checks

- Health: `curl http://localhost:${PORT:-8000}/api/v1/tasks/health`
- Worker stats: `curl http://localhost:${PORT:-8000}/api/v1/tasks/workers/stats`

## Adding your own tasks

Create your tasks under `inference_core/celery/tasks/` and register them with Celery. For example:

```
# inference_core/celery/tasks/example.py
from inference_core.celery.celery_main import celery_app


@celery_app.task(bind=True, name="tasks.add")
def add(self, a: int, b: int) -> int:
     return a + b
```

Trigger the task from anywhere in your code (API handler, service, etc.):

```
from inference_core.celery.celery_main import celery_app

task_id = celery_app.send_task("tasks.add", args=[2, 3])
# or if you import the function:
# from inference_core.celery.tasks.example import add
# task_id = add.delay(2, 3).id
```

Task discovery

- The project currently uses explicit includes/autodiscovery placeholders. Update `inference_core/celery/celery_main.py` to add your task modules to `autodiscover_tasks([...])` and/or `celery_app.conf.update(include=[...])` as you add files under `inference_core/celery/tasks/`.
