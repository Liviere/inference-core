"""
LLM Configuration Tables Migration Script

Creates the database tables for dynamic LLM configuration:
- llm_config_overrides: Admin-level runtime configuration overrides
- user_llm_preferences: User-specific LLM preferences
- allowed_user_overrides: Defines which config keys users can modify

Usage:
    poetry run python -m inference_core.database.sql.migrations.create_llm_config_tables

Or run directly:
    python scripts/create_llm_config_tables.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from sqlalchemy import inspect, text

from inference_core.database.sql.connection import get_async_session, get_engine
from inference_core.database.sql.models.llm_config import (
    AllowedUserOverride,
    LLMConfigOverride,
    UserLLMPreference,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_tables():
    """Create LLM configuration tables if they don't exist."""
    engine = get_engine()

    tables_to_create = [
        LLMConfigOverride.__table__,
        UserLLMPreference.__table__,
        AllowedUserOverride.__table__,
    ]

    async with engine.begin() as conn:
        for table in tables_to_create:
            # Check if table exists
            def check_table(connection, table_name):
                inspector = inspect(connection)
                return table_name in inspector.get_table_names()

            exists = await conn.run_sync(check_table, table.name)

            if exists:
                logger.info(f"Table '{table.name}' already exists, skipping...")
            else:
                logger.info(f"Creating table '{table.name}'...")
                await conn.run_sync(table.create, checkfirst=True)
                logger.info(f"Table '{table.name}' created successfully")

    logger.info("LLM configuration tables migration completed")


async def seed_default_allowed_overrides():
    """Seed default allowed user overrides for common configuration keys."""

    default_overrides = [
        {
            "config_key": "temperature",
            "display_name": "Temperature",
            "description": "Controls randomness in responses. Lower = more deterministic, higher = more creative.",
            "constraints": {
                "type": "number",
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
            },
        },
        {
            "config_key": "max_tokens",
            "display_name": "Max Tokens",
            "description": "Maximum number of tokens in the response.",
            "constraints": {
                "type": "number",
                "min": 1,
                "max": 8192,
                "step": 1,
            },
        },
        {
            "config_key": "top_p",
            "display_name": "Top P (Nucleus Sampling)",
            "description": "Controls diversity via nucleus sampling. Lower = less random.",
            "constraints": {
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
        },
        {
            "config_key": "default_model",
            "display_name": "Default Model",
            "description": "Your preferred default model for LLM operations.",
            "constraints": {
                "type": "select",
                # allowed_values will be populated dynamically from available models
            },
        },
    ]

    async with get_async_session() as db:
        from sqlalchemy import select

        for override_data in default_overrides:
            # Check if already exists
            stmt = select(AllowedUserOverride).where(
                AllowedUserOverride.config_key == override_data["config_key"]
            )
            result = await db.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                logger.info(
                    f"Allowed override '{override_data['config_key']}' already exists, skipping..."
                )
                continue

            allowed = AllowedUserOverride(
                config_key=override_data["config_key"],
                display_name=override_data.get("display_name"),
                description=override_data.get("description"),
                constraints=override_data.get("constraints"),
                is_active=True,
            )
            db.add(allowed)
            logger.info(f"Created allowed override: {override_data['config_key']}")

        await db.commit()

    logger.info("Default allowed overrides seeded successfully")


async def main():
    """Run the migration."""
    logger.info("Starting LLM configuration tables migration...")

    await create_tables()
    await seed_default_allowed_overrides()

    logger.info("Migration completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
