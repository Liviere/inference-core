# Database Migrations (Alembic)

This project uses [Alembic](https://alembic.sqlalchemy.org/) for database schema migrations. It allows us to track changes to the database structure and apply them consistently across different environments.

## Architecture

The migration system is configured to work seamlessly with the project's async database setup.

- **Location**: All migrations are stored in the `migrations/` directory at the root of the repository.
- **Async Handling**: Although the application uses async drivers (like `asyncpg`), Alembic runs in a synchronous context. The `migrations/env.py` file automatically maps async driver names to their synchronous counterparts (e.g., `postgresql+asyncpg` -> `postgresql+psycopg`) to ensure compatibility.
- **Configuration**: Connection strings and settings are pulled directly from the application's `Settings` class, ensuring consistency between the app and the migration tool.

## Common Workflows

### 1. Applying Migrations

To bring your database up to the latest schema version:

```bash
poetry run alembic upgrade head
```

### 2. Creating a New Migration

When you modify SQLAlchemy models in `inference_core/database/sql/models/`, you need to generate a new migration file. Alembic can detect these changes automatically:

```bash
poetry run alembic revision --autogenerate -m "describe your changes here"
```

**Warning**: Always review the generated file in `migrations/versions/` before applying it. Autogenerate is powerful but not perfect (it might miss some index changes or custom constraints).

### 3. Checking Migration Status

To see which migration is currently applied:

```bash
poetry run alembic current
```

To see the history of all migrations:

```bash
poetry run alembic history --verbose
```

### 4. Rolling Back

To undo the last applied migration:

```bash
poetry run alembic downgrade -1
```

## Applications with a Migration Tree of Their Own

An application that keeps its own Alembic tree over these tables does not run
the revisions in `migrations/versions/`; it repeats each schema change in a
revision of its own. Revisions that change a table an application is likely to
have:

| Revision | Change |
| --- | --- |
| `e70bd305a912` | `user_agent_instances.skills` (JSON, nullable) |
| `widen_agent_instance_text_cols` | `user_agent_instances.instance_name` and `display_name` widened to `Text` |
| `add_reported_model_name` | `llm_request_logs.reported_model_name` (`String(255)`, nullable) |

Apply the schema change before the code that writes the column: a usage log
that cannot be inserted is dropped (`usage_logging.fail_open`), and the call it
described is then missing from the accounts.

## Best Practices

- **One Revision per PR**: Combine related schema changes into a single migration file.
- **Descriptive Names**: Use clear messages for migrations (e.g., `add_user_last_login_column` instead of `fix_db`).
- **Dependencies**: Ensure any new models are imported in `inference_core/database/sql/base.py` or directly in `migrations/env.py` so Alembic can "see" them via `Base.metadata`.

## Troubleshooting

### "Driver not found"

If you see an error about missing `asyncpg` or `psycopg2` during migration:

1. Ensure you have installed all dependencies: `poetry install`.
2. Check your `DATABASE_SERVICE` in `.env`.
3. The mapping logic in `migrations/env.py` handles standard async drivers, but custom drivers might need manual entry in the `driver_mapping` dictionary.
