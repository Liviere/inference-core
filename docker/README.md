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

## Playwright MCP – headful mode (visible browser window)

The repository includes a helper compose file to run the Playwright MCP server in "headed" mode (browser window visible on a Linux host with X11/XWayland):

- File: `docker/docker-compose.playwright-mcp.headful.yml`
- Browser: Chromium in headed mode (no `--headless`), `--no-sandbox` for running inside the container
- Stability: `tmpfs: /dev/shm` mounted and a separate volume for the browser cache

Host requirements (Linux):

- Running X11 or XWayland and the `DISPLAY` environment variable set
- Container access to the X11 socket: mount `/tmp/.X11-unix`
- (Optional) Allow local connections: `xhost +local:` (revoke after the session: `xhost -local:`)

Quick start – optional commands (not required by the application, only for MCP debugging):

````bash
# Allow containers to access X11 (optional, depending on X policy)
xhost +local:

# Start the MCP server in headful mode
docker compose -f docker/docker-compose.playwright-mcp.headful.yml --profile headful up

# MCP will listen on http://localhost:8931

Notes:

- If you use Wayland without XWayland, consider enabling XWayland or using a VNC/noVNC alternative.
- For tests, stick to headless mode and normalized port settings suitable for CI.
- The `playwright-ms` volume stores cache/browsers under `/ms-playwright` (resolves ENOENT errors on read-only filesystems).

#### MCP server configuration (keep context between sessions)

- Config file: `docker/playwright-mcp.config.json`
 - Config file: `docker/playwright-mcp.config.example.json` (copy to `docker/playwright-mcp.config.json` and edit; the non-example file is gitignored)
   - Sets `isolated: false` and a `userDataDir` to persist the browser profile.
   - Enables `saveSession: true` to keep artifacts/state.
   - Headful compose variants mount this file and pass `--config=/app/playwright-mcp.config.json`.

Timeouts configuration via environment variables

You can override the MCP CLI timeouts using environment variables when starting the containers. These are injected into the CLI command by the compose files and have sensible defaults:

- `PLAYWRIGHT_TIMEOUT_ACTION` — action timeout in milliseconds (CLI flag `--timeout-action`), default: `5000`
- `PLAYWRIGHT_TIMEOUT_NAVIGATION` — navigation timeout in milliseconds (CLI flag `--timeout-navigation`), default: `60000`

Examples (temporary variables in shell):

```bash
PLAYWRIGHT_TIMEOUT_ACTION=8000 \
PLAYWRIGHT_TIMEOUT_NAVIGATION=120000 \
docker compose -f docker/docker-compose.playwright-mcp.headful.yml --profile headful up
```

Or add them to your `.env` file in the repository root (recommended):

```
# .env
PLAYWRIGHT_TIMEOUT_ACTION=8000
PLAYWRIGHT_TIMEOUT_NAVIGATION=120000
```

The compose files provide default fallbacks, so these variables are optional.
- MCP artifacts are written to `docker/outputs` (mapped to `/outputs`).

Shared context across clients:

- Headful compose variants additionally pass `--shared-browser-context`, which causes all HTTP connections to MCP to use the same browser context.
- Effect: state/login/cookies are shared between MCP client sessions and preserved across connections (together with the persistent `userDataDir` profile).
- Separation note: in shared environments or when handling concurrent requests from different users, consider disabling this option or running separate MCP instances.

Optional: to ensure the browser window never closes when the MCP session ends, attach to an existing browser:

- Chrome DevTools (CDP): run Chrome with `--remote-debugging-port=9222` and set `browser.cdpEndpoint` in the config.
- Extension: start the server with `--extension` to connect via the Chrome/Edge extension.

### Windows / WSL2 variants

There are two ready compose files for Windows/WSL2:

1. WSLg (Windows 11, GUI via WSL)

- File: `docker/docker-compose.playwright-mcp.headful-wslg.yml`
- Requires WSLg; run from a WSL shell
- Mounts `/mnt/wslg/.X11-unix` and `runtime-dir`, sets `DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`

Run:

```bash
docker compose -f docker/docker-compose.playwright-mcp.headful-wslg.yml --profile headful-wslg up
````

2. X-server on Windows (VcXsrv/Xming) over TCP

- File: `docker/docker-compose.playwright-mcp.headful-windows-xserver.yml`
- Requires a running X‑server on Windows (e.g., VcXsrv) and port 6000 open
- Sets `DISPLAY=host.docker.internal:0.0` (Docker Desktop)

Run:

```bash
docker compose -f docker/docker-compose.playwright-mcp.headful-windows-xserver.yml --profile headful-win up
```

Security note: temporarily disabling access control in the X‑server simplifies startup, but consider using restrictions (xauth) in shared environments.
