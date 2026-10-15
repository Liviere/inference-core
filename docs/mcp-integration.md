# Model Context Protocol (MCP) Integration

## Overview

Inference Core supports the **Model Context Protocol (MCP)** to enable tool-augmented reasoning in LLM tasks. MCP provides a standardized way for LLMs to interact with external tools and capabilities such as:

- **Web browsing** via Playwright MCP servers
- **File system operations**
- **Database queries**
- **API integrations**
- **Custom tools** via your own MCP servers

This integration uses:

- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters) for LangChain compatibility
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) for MCP client/server communication

## Security & Isolation

⚠️ **IMPORTANT**: MCP tool access is **disabled by default** (`mcp.enabled: false`).

### What agents enforce

An agent gets MCP tools through `AgentMCPToolManager` (`inference_core/agents/agent_mcp_tools.py`). On that path:

1. **Default OFF**: nothing loads while `mcp.enabled` is false (`MCP_ENABLED` overrides it).
2. **A profile belongs to agent types**: an agent type uses the profile its configuration names (`mcp_profile`). An agent instance may select another profile only when that profile lists the type under `agents`. The check runs when an instance is created or changed (`UserAgentInstanceService`, also when only its base agent changes) and again when an agent starts.
3. **Superusers select profiles**: selecting a profile for an instance takes a superuser where the profile's `require_superuser`, or else the global `mcp.require_superuser`, says so. The tools of the profile an agent type names load for every user of that type.
4. **Tool allowlists**: a profile's `include_tools` limits its tools, and an agent's `allowed_tools`, when set, applies to MCP tools as to any other tool.
5. **No shared clients**: every agent start lists the tools anew, and every tool call opens a session of its own.
6. **Load timeout**: each server's tools are listed within the profile's `load_timeout_seconds` (default 15). A server that fails or does not answer is left out of that run; the other servers' tools still load.
7. **One tool per name**: when two tools share a name, the first one stays and the repeat is dropped with an error in the log. Middleware tools keep their names.

`allowlist_hosts`, `rate_limits`, `max_run_seconds`, `max_steps` and `tool_retry_attempts` are read only by the older `MCPToolManager` (`inference_core/llm/mcp_tools.py`), not by agents. Restrict where MCP servers may connect in the deployment (isolated containers, an egress proxy).

### Security Best Practices

1. **Keep it off** unless an agent type needs it, and name profiles only on the types that should have them.
2. **Isolated Execution**: Run risky MCP servers (like Playwright) in isolated containers.
3. **Limit tools** with `include_tools` to what the agents need.
4. **Treat tool output as untrusted**: an MCP server's answers reach the model as they are.

### Recommended Isolation for Risky Servers

For servers that access the network or file system (e.g., Playwright):

```yaml
# Docker Compose example
services:
  playwright-mcp:
    image: playwright-mcp-server:latest
    networks:
      - isolated
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    tmpfs:
      - /tmp
    mem_limit: 512m
    cpus: 0.5
```

## Configuration

MCP is configured in `llm_config.yaml` under the `mcp` section.

### Basic Example

```yaml
# llm_config.yaml

mcp:
  # Global feature flag - default OFF for security
  enabled: false

  # Require superuser permissions for MCP tool access
  require_superuser: true

  # Default profile to use when none specified
  default_profile: null

  # Profiles group servers and safety limits per use case
  profiles:
    local-tools:
      description: 'Safe local tools for computation'
      servers: ['math']
      max_steps: 10 # Maximum agent reasoning steps
      max_run_seconds: 30 # Hard timeout per request
      rate_limits:
        requests_per_minute: 60
        tokens_per_minute: 90000

  # Individual MCP server configurations
  servers:
    math:
      transport: 'stdio'
      command: 'python'
      args: ['examples/mcp_math_server.py']
      env:
        LOG_LEVEL: 'info'
```

### Advanced Example (Web Browsing)

```yaml
mcp:
  enabled: true # ENABLE MCP (disabled by default)
  require_superuser: true
  default_profile: 'web-browsing'

  profiles:
    web-browsing:
      description: 'Web browsing and interaction via Playwright'
      servers: ['playwright']
      max_steps: 5 # Limit agent steps
      max_run_seconds: 60 # Hard timeout
      allowlist_hosts: # Restrict to specific domains
        - 'example.com'
        - '*.docs.example.com'
      rate_limits:
        requests_per_minute: 30
        tokens_per_minute: 60000

  servers:
    playwright:
      transport: 'streamable_http'
      url: 'http://localhost:3000/mcp'
      headers:
        Authorization: '${MCP_PLAYWRIGHT_TOKEN:-}' # Env var expansion
      timeouts:
        connect_seconds: 10
        read_seconds: 300
      terminate_on_close: true
```

## Transport Types

MCP supports multiple transport mechanisms:

### stdio (Standard Input/Output)

For local processes:

```yaml
servers:
  local-tool:
    transport: 'stdio'
    command: 'python'
    args: ['/path/to/server.py']
    env:
      LOG_LEVEL: 'info'
    cwd: '/working/directory' # Optional
```

### streamable_http (HTTP with Streaming)

For remote HTTP servers:

```yaml
servers:
  remote-tool:
    transport: 'streamable_http'
    url: 'http://localhost:8000/mcp'
    headers:
      Authorization: 'Bearer ${MCP_TOKEN:-}'
      X-Custom-Header: 'value'
    timeouts:
      connect_seconds: 10
      read_seconds: 300
```

### sse (Server-Sent Events)

For streaming event connections:

```yaml
servers:
  sse-tool:
    transport: 'sse'
    url: 'https://sse.example.com/events'
    headers:
      Accept: 'text/event-stream'
    timeouts:
      connect_seconds: 10
      read_seconds: 300
```

### websocket

For bidirectional WebSocket connections:

```yaml
servers:
  ws-tool:
    transport: 'websocket'
    url: 'wss://realtime.example.com/mcp'
```

## Agent Configuration

Link MCP profiles to named agents:

```yaml
# agents section in llm_config.yaml

agents:
  default_agent:
    primary: 'gpt-5-mini'
    fallback: ['gemini-2.5-flash', 'claude-3-5-haiku-latest']
    testing: ['gpt-5-nano']
    description: 'Tool-augmented reasoning with MCP'
    mcp_profile: 'local-tools' # Link to MCP profile
```

Who else may use a profile, and how long listing its tools may take, is set on the profile:

```yaml
mcp:
  profiles:
    local-tools:
      servers: ['math']
      agents: ['assistant_agent'] # instances of these types may select it too
      require_superuser: true # only a superuser selects it for an instance
      load_timeout_seconds: 15 # per server, when an agent starts
```

An agent instance selects a profile with `config_overrides.mcp_profile`. A profile the instance's type may not use is refused when the instance is saved, and loads nothing at run time.

## Environment Variables

Use environment variables for secrets:

```yaml
servers:
  secure-server:
    headers:
      Authorization: '${MCP_AUTH_TOKEN:-default_fallback}'
      API-Key: '${MCP_API_KEY:-}'
```

Set variables in `.env`:

```bash
# .env
MCP_ENABLED=true
MCP_AUTH_TOKEN=your-secret-token
MCP_API_KEY=your-api-key
```

## Usage Examples

### Programmatic Usage

```python
from inference_core.llm.mcp_tools import get_mcp_tool_manager

# Get the MCP tool manager
manager = get_mcp_tool_manager()

# Check if MCP is enabled
if manager.is_enabled():
    # Get tools for a profile
    user = {"is_superuser": True}  # Superuser required by default
    tools = await manager.get_tools(profile_name="local-tools", user=user)

    print(f"Loaded {len(tools)} tools")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # Execute a tool
    result = await tools[0].ainvoke({"a": 5, "b": 3})
    print(f"Result: {result}")

    # Clean up
    await manager.close()
```

### API Usage (Future)

Once API integration is complete:

```bash
# Enable MCP
export MCP_ENABLED=true

# Start API server
poetry run fastapi dev

# Call agent endpoint with tools
curl -X POST http://localhost:8000/api/v1/llm/agent \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "prompt": "Calculate (5 + 3) * 12",
    "mcp_profile": "local-tools"
  }'
```

## Creating Your Own MCP Server

### Simple Math Server Example

```python
# my_math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Configure in `llm_config.yaml`:

```yaml
servers:
  my-math:
    transport: 'stdio'
    command: 'python'
    args: ['my_math_server.py']
```

## Testing

### Unit Tests

```bash
# Test MCP configuration parsing
poetry run pytest tests/unit/llm/test_mcp_config.py -v

# Test MCP tool manager
poetry run pytest tests/unit/llm/test_mcp_tools.py -v
```

### Integration Tests

```bash
# Test with real MCP server (math example)
poetry run pytest tests/integration/test_mcp_integration.py -v
```

## Observability

MCP operations are logged and traced:

```python
# Logs appear as:
logger.info(f"Loading MCP tools for profile: {profile_name}")
logger.info(f"Loaded {len(tools)} tools from profile '{profile_name}'")
logger.warning("MCP tools require superuser permissions")
logger.error(f"Error loading MCP tools: {e}")
```

## Troubleshooting

### MCP not loading tools

1. Check `mcp.enabled: true` in config
2. Check that the agent's type names the profile, or that the profile lists the type under `agents`
3. Check that the agent's `allowed_tools`, when set, names the MCP tools
4. Check server configuration (command, url, etc.)
5. Review logs for error messages ("Not loading MCP tools for agent type …", "Error getting tools from MCP server …")

### Permission Denied

Saving an agent instance answers `MCP profile '…' can only be selected by a superuser` or `… is not available to agent type '…'` when the instance may not select that profile (see "What agents enforce").

The older `MCPToolManager` raises:

```python
PermissionError: User does not have permission to use MCP tools
```

**Solutions:**

- Set `require_superuser: false` in config for testing
- Ensure user has `is_superuser: true` flag
- Or enable MCP globally: `export MCP_ENABLED=true`

### Server Connection Failed

```
McpError: Connection closed
```

**Solutions:**

- Verify server command/path is correct
- Check server is executable: `python server.py`
- Review server logs
- Check timeouts are sufficient

### Dependencies Not Installed

```
ValueError: MCP dependencies not installed
```

**Solution:**

```bash
poetry add mcp langchain-mcp-adapters
```

## Limitations & Future Work

Current implementation:

- ✅ Configuration and schema
- ✅ MCP client management
- ✅ Tool loading and execution
- ✅ Security controls (RBAC, timeouts)
- ✅ Integration tests

Future enhancements:

- ⏳ Streaming tool events from agent runs
- ⏳ Celery worker integration
- ⏳ Additional Agent Instances API coverage
- ⏳ Rate limiting implementation
- ⏳ Detailed audit logging

## References

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [LangChain Tools Documentation](https://python.langchain.com/docs/concepts/tools/)

---

**Note**: This feature is experimental and under active development. API may change in future releases.
