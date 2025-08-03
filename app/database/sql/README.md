# Database Module

This directory contains all the necessary components for interacting with the application's database using SQLAlchemy in an asynchronous manner. It provides a structured and reusable way to define models, manage connections, and handle data serialization.

## Core Components

### `connection.py`

This file is the heart of the database connection management.

- **Engine and Session Management**: It creates and manages a single, asynchronous SQLAlchemy engine (`_engine`) and an async session maker (`_async_session_maker`) for the entire application.
- **Async Session Provider**: The `get_async_session()` context manager is the standard way to obtain a database session. It ensures that sessions are properly handled, transactions are rolled back on error, and connections are always closed.
- **Database-Specific Optimizations**: It uses SQLAlchemy's event system to apply specific `PRAGMA` settings for SQLite (e.g., enabling foreign keys) and session settings for MySQL, making the setup robust across different database systems.
- **`DatabaseManager`**: A utility class for high-level database operations like creating tables (`create_tables`), checking health (`health_check`), and resetting the database.

### `base.py`

This file defines the base classes and mixins for all database models, promoting code reuse and consistency.

- **`SmartJSON`**: A custom SQLAlchemy type that automatically adapts to different databases:
  - **PostgreSQL**: Uses native `JSON` type for optimal performance and validation
  - **MySQL 5.7+**: Uses native `JSON` type with proper indexing support
  - **SQLite/older databases**: Falls back to `TEXT` with application-level JSON handling
- **`BaseModel`**: The primary base class for all models. It automatically provides:
  - A `UUID` primary key (`id`).
  - `TimestampMixin`: `created_at` and `updated_at` timestamp fields.
  - `SoftDeleteMixin`: `is_deleted` flag and `deleted_at` timestamp for soft-deleting records instead of permanently removing them.
  - Helper methods like `to_dict()` and `update_from_dict()`.
- **`AuditMixin`**: Adds `created_by` and `updated_by` fields to track which user created or last updated a record.
- **`MetadataMixin`**: Provides flexible `metadata_json` (JSON) and `tags` (string) fields for storing unstructured data:
  - `metadata_json`: Uses `SmartJSON` type for cross-database compatibility
  - `tags`: Uses `String(500)` to ensure MySQL compatibility
- **`FullAuditModel`**: A comprehensive base model that includes all the features from `BaseModel`, `AuditMixin`, and `MetadataMixin`. New models should typically inherit from this.

### `serializers.py`

This module handles the serialization (Python object -> dictionary/JSON) and deserialization (dictionary/JSON -> Python object) of SQLAlchemy models. This is crucial for communicating with the API.

- **`ModelSerializer`**: A class containing static methods to perform serialization.
  - `to_dict()`: Converts a model instance into a dictionary. It can handle relationships and exclude specific fields.
  - `from_dict()`: Creates a model instance from a dictionary, automatically converting data types like ISO date strings to `datetime` objects and string UUIDs to `UUID` objects.
  - `bulk_to_dict()`: Converts a list of model instances to a list of dictionaries.

### `models/` Directory

This subdirectory is the designated location for all application-specific database models (e.g., `user.py`, `product.py`). Each model should be defined in its own file and inherit from one of the base classes provided in `app/database/base.py` (e.g., `class User(FullAuditModel): ...`).

## Database Compatibility

This module is designed to work seamlessly across multiple database systems:

### Supported Databases

- **PostgreSQL** ✅
  - Native JSON type support for optimal performance
  - Full UUID support
  - Advanced indexing capabilities
- **MySQL 5.7+** ✅
  - Native JSON type support (MySQL 5.7+)
  - Proper VARCHAR length specifications
  - Timezone-aware datetime handling
- **SQLite** ✅
  - TEXT-based JSON storage with application validation
  - Pragma optimizations for better performance
  - Development-friendly setup

### Cross-Database Features

- **Automatic Type Adaptation**: The `SmartJSON` type automatically selects the best storage format for each database
- **Consistent Field Lengths**: All string fields have explicit lengths for MySQL compatibility
- **Universal UUID Support**: Primary keys use database-agnostic UUID implementation
- **Timezone Handling**: All datetime fields are timezone-aware across all databases

### Database-Specific Optimizations

- **SQLite**: Automatic pragma settings for foreign keys, WAL mode, and performance tuning
- **MySQL**: Session-level timezone and SQL mode configuration
- **PostgreSQL**: Native JSON indexing and query optimization support
