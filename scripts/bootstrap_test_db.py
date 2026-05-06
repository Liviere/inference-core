"""Bootstrap the current schema into a fresh database and stamp Alembic head.

WHY:
    This repository's historical Alembic chain starts from schema-altering
    revisions instead of a full base-schema migration. A brand-new database can
    therefore fail during ``alembic upgrade head`` before the core tables exist.
    This helper gives local test environments one repeatable initialization
    command: create the current SQLAlchemy metadata and then stamp the database
    to the latest Alembic revision so future incremental upgrades keep working.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from inference_core.core.env import load_project_dotenv

DOTENV_PATH = load_project_dotenv()

from inference_core.database.sql.connection import Base, get_engine

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_INI_PATH = PROJECT_ROOT / "alembic.ini"


def _read_table_names(sync_connection) -> list[str]:
    """Return the current table names after schema bootstrap.

    WHY: Operators need a compact confirmation that the bootstrap created the
    expected core schema before the database is stamped to head.
    """
    inspector = inspect(sync_connection)
    return sorted(inspector.get_table_names())


async def bootstrap_schema() -> list[str]:
    """Create the current SQLAlchemy metadata in the configured database.

    WHY: Fresh local test databases need the full table set before Alembic's
    version table can be aligned with the latest application schema.
    """
    engine = get_engine()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            return await connection.run_sync(_read_table_names)
    finally:
        await engine.dispose()


def stamp_head() -> None:
    """Mark the bootstrapped database as being on the latest Alembic revision.

    WHY: After the first-time bootstrap, operators should be able to use normal
    ``alembic upgrade head`` flows for future schema deltas.
    """
    alembic_config = Config(str(ALEMBIC_INI_PATH))
    command.stamp(alembic_config, "head")


async def main() -> None:
    """Initialize a fresh local database for auth-dependent test scenarios.

    WHY: Test environments need one explicit bootstrap command that is safe to
    rerun and does not depend on operators knowing the repository's migration
    history quirks.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Loaded dotenv file: %s", DOTENV_PATH)
    table_names = await bootstrap_schema()
    logger.info("Verified %s tables", len(table_names))
    logger.info("Tables: %s", ", ".join(table_names))
    stamp_head()
    logger.info("Stamped database to Alembic head")


if __name__ == "__main__":
    asyncio.run(main())
