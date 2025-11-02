# Batch Lifecycle Tasks - Manual Testing Guide

This guide provides steps for manually testing the batch lifecycle Celery tasks.

## Prerequisites

1. **Redis Running**: Start Redis using Docker

   ```bash
   docker run -d --name redis-test -p 6379:6379 redis:7-alpine
   ```

2. **Database Migrations**: Run database migrations to create batch tables

   ```bash
   poetry run alembic upgrade head
   ```

3. **LLM Config**: Copy and configure LLM config file
   ```bash
   cp llm_config.example.yaml llm_config.yaml
   # Edit llm_config.yaml as needed
   ```

## Starting Celery Components

### 1. Start Celery Worker

```bash
# In terminal 1 - Start worker for batch tasks
poetry run celery -A inference_core.celery.celery_main:celery_app worker \
  --loglevel=info \
  --queues=batch_tasks
```

### 2. Start Celery Beat (Optional - for automatic polling)

```bash
# In terminal 2 - Start beat scheduler
poetry run celery -A inference_core.celery.celery_main:celery_app beat \
  --loglevel=info
```

### 3. Start Flower Monitoring (Optional)

```bash
# In terminal 3 - Start Flower for task monitoring
poetry run celery -A inference_core.celery.celery_main:celery_app flower \
  --port=5555
```

## Manual Task Testing

### Test 1: Redis Lock Functionality

```python
# Test Redis connectivity and lock mechanism
from inference_core.core.redis_client import get_sync_redis
from inference_core.celery.tasks.batch_tasks import BATCH_POLL_LOCK_KEY

redis_client = get_sync_redis()
print("Redis ping:", redis_client.ping())

# Test lock acquisition
lock1 = redis_client.set(BATCH_POLL_LOCK_KEY, '1', nx=True, ex=300)
lock2 = redis_client.set(BATCH_POLL_LOCK_KEY, '1', nx=True, ex=300)
print(f"First lock: {lock1}, Second lock: {lock2}")

# Cleanup
redis_client.delete(BATCH_POLL_LOCK_KEY)
```

### Test 2: Task Registration

```python
# Verify all batch tasks are registered
from inference_core.celery.celery_main import celery_app
import inference_core.celery.tasks.batch_tasks

print("Batch tasks registered:")
for task in sorted(celery_app.tasks.keys()):
    if task.startswith('batch.'):
        print(f"  {task}")

print("\nBeat schedule:")
for name, config in celery_app.conf.beat_schedule.items():
    print(f"  {name}: {config}")
```

### Test 3: Direct Task Execution

```python
# Test task execution directly (without Celery worker)
from inference_core.celery.tasks.batch_tasks import batch_poll

# This will use Redis lock and try to query database
result = batch_poll()
print("Poll result:", result)
```

### Test 4: Async Task Execution (requires running worker)

```python
# Send tasks to Celery worker
from inference_core.celery.tasks.batch_tasks import batch_poll, batch_submit

# Test polling task
poll_task = batch_poll.delay()
print(f"Poll task ID: {poll_task.id}")
print(f"Poll result: {poll_task.get(timeout=10)}")

# Test submit task (will fail without valid job_id)
try:
    submit_task = batch_submit.delay("invalid-job-id")
    print(f"Submit task ID: {submit_task.id}")
    print(f"Submit result: {submit_task.get(timeout=10)}")
except Exception as e:
    print(f"Expected error: {e}")
```

## Integration Testing

### Test with Real Batch Job (requires database setup)

1. **Create a batch job** using the API or database directly
2. **Submit the job** using `batch_submit.delay(job_id)`
3. **Monitor polling** - `batch_poll` should detect and update the job
4. **Fetch results** using `batch_fetch.delay(job_id)` when completed
5. **Test retry** using `batch_retry_failed.delay(job_id)` if items failed

### Monitor Task Execution

- **Celery logs**: Check worker terminal for task execution logs
- **Flower UI**: Visit http://localhost:5555 to monitor tasks
- **Redis inspection**: Check Redis for locks and task state

## Expected Behaviors

### Successful Execution

- Tasks should log start/completion with durations
- Redis locks should prevent duplicate polling
- Tasks should handle database errors gracefully
- Beat scheduler should automatically trigger polling every 30 seconds

### Error Handling

- Transient errors should trigger retries with exponential backoff
- Permanent errors should fail fast without excessive retries
- Database errors should be logged but not crash the worker
- Missing jobs/items should return appropriate error messages

### Idempotency

- Running the same task multiple times should not corrupt state
- Polling should safely handle jobs in any status
- Fetching should detect already-fetched results
- Retrying should only create new jobs for actually failed items

## Troubleshooting

### Common Issues

1. **Tasks not registering**: Ensure imports are working and modules are included
2. **Redis connection errors**: Check if Redis container is running on port 6379
3. **Database errors**: Run migrations and ensure database is accessible
4. **Worker not receiving tasks**: Check queue names and routing configuration
5. **Beat not scheduling**: Verify beat_schedule configuration is loaded

### Debug Commands

```bash
# Check Celery configuration
poetry run celery -A inference_core.celery.celery_main:celery_app inspect conf

# List active tasks
poetry run celery -A inference_core.celery.celery_main:celery_app inspect active

# Check worker stats
poetry run celery -A inference_core.celery.celery_main:celery_app inspect stats

# Monitor task events
poetry run celery -A inference_core.celery.celery_main:celery_app events
```
