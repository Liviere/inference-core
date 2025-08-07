## Docker Deployment

This application includes Docker support for containerized deployment with SQLite, MySQL, and PostgreSQL database options. The Docker configuration is organized in the `docker/` directory with separate compose files for each database type.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Quick Start with Docker

1. **Copy environment configuration:**

   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** with your specific settings (database credentials, etc.)

3. **Choose your database and start the application:**

   **SQLite (for development):**

   ```bash
   docker-compose -f docker-compose.base.yml -f docker/docker-compose.sqlite.yml up -d
   ```

   **MySQL:**

   ```bash
   docker-compose -f docker-compose.base.yml -f docker/docker-compose.mysql.yml up -d
   ```

   **PostgreSQL (recommended for production):**

   ```bash
   docker-compose -f docker-compose.base.yml -f docker/docker-compose.postgres.yml up -d
   ```

4. **Access the application** at `http://localhost:{PORT}` (default is 8000).

### Docker Compose Structure

The Docker configuration is split into multiple files for better modularity:

- `docker-compose.base.yml` - Base application configuration
- `docker/docker-compose.sqlite.yml` - SQLite database configuration
- `docker/docker-compose.mysql.yml` - MySQL database configuration
- `docker/docker-compose.postgres.yml` - PostgreSQL database configuration

This modular approach allows you to easily switch between different database backends without modifying the core application configuration.

### Useful Docker Commands

**Stop the application:**

```bash
docker-compose -f docker-compose.base.yml -f docker/docker-compose.{database}.yml down
```

**View logs:**

```bash
docker-compose -f docker-compose.base.yml -f docker/docker-compose.{database}.yml logs -f
```

**Rebuild and restart:**

```bash
docker-compose -f docker-compose.base.yml -f docker/docker-compose.{database}.yml up -d --build
```

Replace `{database}` with `sqlite`, `mysql`, or `postgres` as needed.
