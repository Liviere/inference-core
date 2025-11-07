# Pluggable Tool Providers for LLM Service

This document describes how to use custom LangChain tools with the LLM service without requiring MCP (Model Context Protocol) servers.

## Overview

The pluggable tool provider system allows applications embedding `inference-core` to attach custom LangChain tools to chat/completion tasks. This provides a lightweight alternative to MCP for application-local tools.

### Key Features

- **Config-driven**: Tool providers are configured per task in `llm_config.yaml`
- **Backward compatible**: No breaking changes to existing code
- **Security controls**: Optional allowlists, limits, and permission checks
- **MCP compatible**: Works alongside MCP tools; both can be used together
- **Extensible**: Simple protocol-based design for future enhancements

## Quick Start

### 1. Create a Tool Provider

A tool provider is a simple Python class that implements the `ToolProvider` protocol:

```python
# app/tools/assistant_tools.py
from langchain_core.tools import BaseTool
from pydantic import Field

class CreateTodoTaskTool(BaseTool):
    """Tool for creating TODO tasks"""
    name = "create_todo_task"
    description = "Create a new TODO task for the user"
    user_id: str = Field(..., description="User ID for task creation")
    
    def _run(self, title: str, description: str = "") -> str:
        """Synchronous implementation"""
        # Your task creation logic here
        return f"Created task: {title}"
    
    async def _arun(self, title: str, description: str = "") -> str:
        """Async implementation"""
        # Your async task creation logic here
        return f"Created task: {title}"


class AssistantToolsProvider:
    """Tool provider for assistant tasks"""
    name = "assistant_tools"
    
    async def get_tools(self, task_type: str, user_context=None):
        """Return tools based on context"""
        user_id = (user_context or {}).get("user_id")
        if not user_id:
            return []
        
        return [
            CreateTodoTaskTool(user_id=user_id),
            # Add more tools as needed
        ]
```

### 2. Register the Provider at Application Startup

```python
# app/startup.py
from inference_core.llm.tools import register_tool_provider
from app.tools.assistant_tools import AssistantToolsProvider

def initialize_tool_providers():
    """Register all tool providers at startup"""
    register_tool_provider(AssistantToolsProvider())
    # Register other providers as needed
```

### 3. Configure in `llm_config.yaml`

```yaml
tasks:
  assistant_converse:
    primary: gpt-5-mini
    local_tool_providers: ['assistant_tools']  # References the provider name
    tool_limits:
      max_steps: 4              # Maximum agent reasoning steps
      max_run_seconds: 30       # Hard timeout per request
      tool_retry_attempts: 2    # Retry attempts on tool failures
    allowed_tools:              # Optional allowlist
      - create_todo_task
```

### 4. Use in Your Application

No changes needed to the call sites! The tools are automatically attached when you use the configured task:

```python
from inference_core.services.llm_service import get_llm_service

llm_service = get_llm_service()

response = await llm_service.chat(
    session_id="user-123-session",
    user_input="Create a task to review the quarterly report",
    task_type="assistant_converse",  # Uses the configured task
)

# The agent will automatically use the create_todo_task tool
print(response.result["reply"])
# Tools used are logged in response.result["tools_used"] if tools were invoked
```

## Configuration Reference

### Task Configuration Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `local_tool_providers` | `List[str]` | No | List of registered tool provider names |
| `tool_limits` | `ToolLimits` | No | Execution limits for tool usage |
| `allowed_tools` | `List[str]` | No | Allowlist of tool names (omit to allow all) |

### Tool Limits Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_steps` | `int` | 10 | Maximum agent reasoning steps (1-50) |
| `max_run_seconds` | `int` | 60 | Hard timeout per request in seconds (1-600) |
| `tool_retry_attempts` | `int` | 2 | Retry attempts on tool failures (0-10) |

## Advanced Usage

### Combining MCP and Local Tools

You can use both MCP tools and local providers together. They will be merged with deduplication:

```yaml
tasks:
  hybrid_assistant:
    primary: gpt-5-mini
    mcp_profile: base_tools           # MCP tools from profile
    local_tool_providers: ['app_tools']  # Plus local tools
    tool_limits:
      max_steps: 6
      max_run_seconds: 45
```

**Deduplication**: If both MCP and local providers return a tool with the same name, the MCP tool takes precedence.

### Context-Aware Tools

Tool providers can return different tools based on the task type and user context:

```python
class ContextAwareProvider:
    name = "context_aware_tools"
    
    async def get_tools(self, task_type: str, user_context=None):
        tools = []
        
        # Task-specific tools
        if task_type == "chat":
            tools.append(ConversationalTool())
        elif task_type == "assistant_converse":
            tools.append(TaskManagementTool())
        
        # Permission-based tools
        if user_context and user_context.get("is_superuser"):
            tools.append(AdminTool())
        
        # User-specific tools
        user_id = (user_context or {}).get("user_id")
        if user_id:
            tools.append(PersonalTool(user_id=user_id))
        
        return tools
```

### Security with Allowlists

Use `allowed_tools` to restrict which tools the agent can use, even if providers return more:

```yaml
tasks:
  restricted_assistant:
    primary: gpt-5-mini
    local_tool_providers: ['all_tools']  # Provider may return many tools
    allowed_tools:                        # But only these are allowed
      - search_knowledge_base
      - create_note
```

### Per-Request User Context

While tool providers are configured globally, you can pass user context per request to enable per-user or per-session tool customization:

```python
# In your application code
response = await llm_service.chat(
    session_id=session_id,
    user_input=user_input,
    task_type="assistant_converse",
    # Note: user_context parameter would need to be added to LLMService.chat()
    # For now, use session-based context or configure tools at startup
)
```

## Tool Provider Protocol

The `ToolProvider` protocol is simple and flexible:

```python
from typing import Protocol, Any, Dict, List, Optional

class ToolProvider(Protocol):
    """Protocol for tool providers"""
    
    name: str  # Unique identifier for this provider
    
    async def get_tools(
        self,
        task_type: str,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Return list of LangChain BaseTool instances
        
        Args:
            task_type: Task type (e.g., "chat", "assistant_converse")
            user_context: Optional dict with user metadata
                - user_id: User identifier
                - is_superuser: Admin flag
                - session_id: Session identifier
                - Any other app-specific context
        
        Returns:
            List of LangChain BaseTool instances
        """
        ...
```

## Registry API

### `register_tool_provider(provider: ToolProvider) -> None`

Register a tool provider globally. Must be called at application startup.

```python
from inference_core.llm.tools import register_tool_provider

register_tool_provider(MyToolProvider())
```

**Raises**: `ValueError` if provider lacks required attributes.

### `get_registered_providers() -> Dict[str, ToolProvider]`

Get all registered tool providers (useful for debugging).

```python
from inference_core.llm.tools import get_registered_providers

providers = get_registered_providers()
print(f"Registered providers: {list(providers.keys())}")
```

### `unregister_tool_provider(name: str) -> None`

Unregister a provider by name (useful for testing).

```python
from inference_core.llm.tools import unregister_tool_provider

unregister_tool_provider("test_provider")
```

### `clear_tool_providers() -> None`

Clear all registered providers (primarily for testing).

```python
from inference_core.llm.tools import clear_tool_providers

clear_tool_providers()  # Resets the registry
```

## Best Practices

### 1. Register Providers at Startup

Always register tool providers during application initialization, not on-demand:

```python
# Good: Register at startup
def create_app():
    app = FastAPI()
    initialize_tool_providers()  # Register once
    return app

# Bad: Registering per-request
@app.post("/chat")
async def chat_endpoint(...):
    register_tool_provider(...)  # Don't do this!
```

### 2. Use Specific Task Types

Create specific task types for different tool configurations rather than overloading generic tasks:

```yaml
tasks:
  chat:                    # Generic chat without tools
    primary: gpt-5-mini
  
  assistant_converse:      # Chat with assistant tools
    primary: gpt-5-mini
    local_tool_providers: ['assistant_tools']
  
  research_assistant:      # Chat with research tools
    primary: gpt-5-mini
    local_tool_providers: ['research_tools']
```

### 3. Set Appropriate Limits

Tune limits based on your tool complexity and latency requirements:

```yaml
tool_limits:
  max_steps: 4              # Fewer steps for simple tools
  max_run_seconds: 30       # Tight timeout for user-facing tasks
  tool_retry_attempts: 2    # Retry for reliability
```

### 4. Log Tool Usage

Tool usage is automatically logged to the response. Monitor this for debugging:

```python
response = await llm_service.chat(...)
if "tools_used" in response.result:
    logger.info(f"Tools used: {response.result['tools_used']}")
```

### 5. Handle Tool Errors Gracefully

Tools should catch and handle errors internally, returning user-friendly messages:

```python
class SafeTool(BaseTool):
    def _run(self, query: str) -> str:
        try:
            return self._do_work(query)
        except Exception as e:
            logger.error(f"Tool error: {e}")
            return f"I encountered an error: {str(e)}"
```

## Troubleshooting

### Tools Not Loading

**Problem**: Tools aren't being used even though configured.

**Solutions**:
1. Check provider is registered: `get_registered_providers()`
2. Verify provider name matches config: `local_tool_providers: ['exact_name']`
3. Check logs for provider errors
4. Ensure task type matches your call: `task_type="assistant_converse"`

### Tools Filtered Out

**Problem**: Some tools don't appear even though provider returns them.

**Solutions**:
1. Check `allowed_tools` allowlist in config
2. Verify tools have unique `name` attributes
3. Check for tool name collisions with MCP tools

### Timeout Errors

**Problem**: Requests timeout when using tools.

**Solutions**:
1. Increase `max_run_seconds` in `tool_limits`
2. Reduce `max_steps` to limit agent iterations
3. Optimize slow tools (add caching, timeouts)

## Migration from MCP

If you currently use MCP servers for application-local tools, you can migrate to local providers:

**Before (MCP)**:
```yaml
# Requires running MCP server process
mcp:
  enabled: true
  servers:
    app_tools:
      transport: stdio
      command: python
      args: ['-m', 'app.mcp_server']

tasks:
  assistant:
    primary: gpt-5-mini
    mcp_profile: app_tools_profile
```

**After (Local Providers)**:
```yaml
# No server process needed
tasks:
  assistant:
    primary: gpt-5-mini
    local_tool_providers: ['app_tools']
```

```python
# Register provider at startup
register_tool_provider(AppToolsProvider())
```

**Benefits**:
- Simpler deployment (no separate MCP server process)
- Lower latency (direct Python calls instead of IPC)
- Easier debugging (tools run in same process)
- Better integration with application state

**When to keep MCP**:
- Sharing tools across multiple applications
- Tools implemented in different languages
- Need for sandboxing/isolation
- Using third-party MCP servers

## Examples

### Example 1: Simple Calculator Tool

```python
from langchain_core.tools import BaseTool

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform basic calculations"
    
    def _run(self, expression: str) -> str:
        """Safe calculation without using eval()
        
        In production, use a library like sympy or implement proper parsing.
        This is a simplified example for demonstration.
        """
        try:
            import operator
            
            # Basic operator map for safe calculation
            ops = {
                '+': operator.add,
                '-': operator.sub,
                '*': operator.mul,
                '/': operator.truediv,
            }
            
            # Very basic parser for "num op num" expressions
            for op_str, op_func in ops.items():
                if op_str in expression:
                    parts = expression.split(op_str)
                    if len(parts) == 2:
                        left = float(parts[0].strip())
                        right = float(parts[1].strip())
                        result = op_func(left, right)
                        return f"Result: {result}"
            
            return f"Could not parse '{expression}'. Use format like '2 + 2'"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        return self._run(expression)


class CalculatorProvider:
    name = "calculator_tools"
    
    async def get_tools(self, task_type, user_context=None):
        return [CalculatorTool()]


# Register at startup
register_tool_provider(CalculatorProvider())
```

### Example 2: Database Query Tool

```python
from langchain_core.tools import BaseTool
from app.database import get_db

class DatabaseQueryTool(BaseTool):
    name = "query_database"
    description = "Query the application database"
    user_id: str
    
    async def _arun(self, query: str) -> str:
        db = get_db()
        try:
            results = await db.execute_query(query, user_id=self.user_id)
            return f"Found {len(results)} results"
        except Exception as e:
            return f"Query failed: {str(e)}"


class DatabaseToolProvider:
    name = "database_tools"
    
    async def get_tools(self, task_type, user_context=None):
        user_id = (user_context or {}).get("user_id")
        if not user_id:
            return []
        return [DatabaseQueryTool(user_id=user_id)]


register_tool_provider(DatabaseToolProvider())
```

### Example 3: Multi-Tool Provider with Permissions

```python
class EnterpriseToolProvider:
    name = "enterprise_tools"
    
    async def get_tools(self, task_type, user_context=None):
        tools = []
        
        # Basic tools for all users
        tools.append(SearchDocumentsTool())
        tools.append(CreateNoteTool())
        
        # Admin-only tools
        if user_context and user_context.get("is_superuser"):
            tools.append(DeleteUserTool())
            tools.append(ViewAuditLogTool())
        
        # Department-specific tools
        department = (user_context or {}).get("department")
        if department == "engineering":
            tools.append(DeploymentTool())
        elif department == "sales":
            tools.append(CRMTool())
        
        return tools


register_tool_provider(EnterpriseToolProvider())
```

## Testing

### Unit Testing Tool Providers

```python
import pytest
from app.tools.my_tools import MyToolProvider

@pytest.mark.asyncio
async def test_provider_returns_tools():
    provider = MyToolProvider()
    tools = await provider.get_tools("chat", user_context={"user_id": "123"})
    assert len(tools) > 0
    assert all(hasattr(t, "name") for t in tools)

@pytest.mark.asyncio
async def test_provider_respects_permissions():
    provider = MyToolProvider()
    
    # Regular user
    tools = await provider.get_tools("chat", {"is_superuser": False})
    tool_names = {t.name for t in tools}
    assert "admin_tool" not in tool_names
    
    # Admin user
    admin_tools = await provider.get_tools("chat", {"is_superuser": True})
    admin_tool_names = {t.name for t in admin_tools}
    assert "admin_tool" in admin_tool_names
```

### Integration Testing

```python
import pytest
from inference_core.llm.tools import register_tool_provider, clear_tool_providers
from inference_core.services.llm_service import LLMService

@pytest.fixture(autouse=True)
def reset_providers():
    clear_tool_providers()
    yield
    clear_tool_providers()

@pytest.mark.asyncio
async def test_chat_with_tools(reset_providers):
    # Register test provider
    register_tool_provider(TestToolProvider())
    
    # Configure and test
    llm_service = LLMService()
    # Mock config to include local_tool_providers
    # ... test chat with tools ...
```

## See Also

- [MCP Integration](./mcp-integration.md) - Using MCP servers
- [Custom Prompts](./custom-prompts.md) - Customizing system prompts
- [Configuration](./configuration.md) - Full configuration reference
