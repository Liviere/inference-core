# MCP Tool Integration Refactoring

## Problem Statement

The original MCP tool integration used a separate `_chat_with_tools` method that created its own model and agent directly, completely bypassing the `_build_chat_chain` factory hook. This meant:

- Custom subclasses that override `_build_chat_chain` would lose their custom configurations when MCP tools were enabled
- Custom system prompts, model parameters, and prompt templates were ignored
- The factory hook pattern was broken for tool-enabled chats

**Example of the problem:**
```python
class CustomLLMService(LLMService):
    def __init__(self):
        super().__init__(
            default_chat_system_prompt="You are a specialized assistant with domain expertise."
        )
    
    def _build_chat_chain(self, **kwargs):
        # Custom chain building logic
        # This was BYPASSED when MCP tools were enabled!
        return super()._build_chat_chain(**kwargs)
```

When MCP tools were configured for a chat task, the custom system prompt and any chain customization would be lost.

## Solution

The refactored implementation uses a hybrid approach that preserves the factory hook pattern while enabling MCP tool functionality:

### 1. New Method: `_chat_with_tools_via_chain`

This method replaces the old `_chat_with_tools` and:
- **Calls `_build_chat_chain` factory hook** to build the base chain
- Augments the system prompt with MCP tool instructions **before** passing to the factory hook
- Uses the chain's resolved model name and parameters
- Wraps the chain with agent tooling for tool execution

```python
async def _chat_with_tools_via_chain(
    self,
    *,
    session_id: str,
    user_input: str,
    model_name: Optional[str],
    model_params: Dict[str, Any],
    prompt_name: Optional[str],
    system_prompt: Optional[str],
    tooling: "LLMService._ToolingContext",
    callbacks,
) -> LLMResponse:
    """Run tool-enabled chat using the chain factory hook.
    
    This method builds a chat chain using the factory hook (_build_chat_chain),
    ensuring custom configurations are preserved, and then wraps it with tool
    execution capability.
    """
    # Augment system prompt with tool instructions
    augmented_system_prompt = system_prompt
    if tooling.instructions:
        if augmented_system_prompt:
            augmented_system_prompt = f"{augmented_system_prompt}\n\n{tooling.instructions}"
        else:
            augmented_system_prompt = tooling.instructions
    
    # Build the base chat chain using the factory hook
    # This ensures all custom configurations are preserved
    chain = self._build_chat_chain(
        model_name=model_name,
        model_params=model_params,
        prompt_name=prompt_name,
        system_prompt=augmented_system_prompt,
    )
    
    # Use the chain's resolved model for agent creation
    # ... rest of agent setup and execution ...
```

### 2. Updated `chat` Method

The `chat` method now:
- Detects if MCP tools are configured
- Calls `_chat_with_tools_via_chain` (which uses the factory hook)
- Falls back gracefully to standard chain if tool execution fails

```python
# Check for MCP tooling context
tooling_ctx = await self._get_tooling_context(effective_task, user_context=None)

# If tools are available, use agent-based chat with tool execution
if tooling_ctx is not None:
    try:
        response = await self._chat_with_tools_via_chain(
            session_id=session_id,
            user_input=user_input,
            model_name=model_name,
            model_params=model_params,
            prompt_name=prompt_name or self._default_prompt_names.get(effective_task),
            system_prompt=system_prompt or self._default_chat_system_prompt,
            tooling=tooling_ctx,
            callbacks=callbacks,
        )
        # ... handle success ...
    except Exception as tool_err:
        logger.warning("Tool-enabled chat failed, falling back to standard chain: %s", tool_err)
        # Fall through to standard chat chain

# Use factory hook for standard chat (no tools)
with task_override(effective_task):
    chain = self._build_chat_chain(
        model_name=model_name,
        model_params=model_params,
        prompt_name=prompt_name or self._default_prompt_names.get(effective_task),
        system_prompt=system_prompt or self._default_chat_system_prompt,
    )
```

## Benefits

### 1. Preserves Factory Hook Pattern
Custom subclasses can now override `_build_chat_chain` and their customizations will work even when MCP tools are enabled:

```python
class DomainExpertLLMService(LLMService):
    def __init__(self):
        super().__init__(
            default_chat_system_prompt="You are a medical domain expert. Provide evidence-based answers.",
            default_model_params={"chat": {"temperature": 0.2}},
        )
    
    def _build_chat_chain(self, **kwargs):
        # Custom logic here
        # This NOW WORKS with MCP tools!
        return super()._build_chat_chain(**kwargs)
```

### 2. System Prompt Augmentation
Custom system prompts are preserved and augmented with MCP tool instructions:

```python
# Original custom system prompt
"You are a medical domain expert. Provide evidence-based answers."

# Becomes (when MCP tools enabled):
"You are a medical domain expert. Provide evidence-based answers.

You have access to external tools via the Model Context Protocol (MCP).
Use them when needed to complete the task. Call a tool only if it helps.
Active profile: medical-tools.
Maximum tool iterations: 10.
Available tools:
- search_pubmed: Search medical literature in PubMed
- calculate_dosage: Calculate medication dosage based on patient parameters
..."
```

### 3. Model Parameter Preservation
Default and runtime model parameters flow through correctly:

```python
# Service with custom defaults
service = LLMService(
    default_model_params={"chat": {"temperature": 0.3, "max_tokens": 500}}
)

# Runtime override
await service.chat(
    session_id="test",
    user_input="Hello",
    temperature=0.7,  # Overrides default
    task_type="chat",  # Has MCP tools configured
)

# Result: temperature=0.7, max_tokens=500, plus MCP tool binding
```

### 4. Graceful Degradation
If tool execution fails, the system falls back to standard chat:

```python
# MCP tool execution fails
logger.warning("Tool-enabled chat failed, falling back to standard chain: %s", tool_err)
# Continues with standard chat chain - no data loss
```

## Migration Guide

### For Custom Subclasses

No changes needed! If you were overriding `_build_chat_chain`, your customizations will now work with MCP tools:

```python
# Before: This didn't work with MCP tools
# After: This works perfectly!
class MyCustomService(LLMService):
    def _build_chat_chain(self, **kwargs):
        # Your custom logic
        return super()._build_chat_chain(**kwargs)
```

### For MCP Configuration

No changes to MCP configuration needed. Continue using `llm_config.yaml`:

```yaml
tasks:
  medical_chat:
    primary: 'gpt-4o-mini'
    description: 'Medical consultation with tool access'
    mcp_profile: "medical-tools"  # Links to MCP profile

mcp:
  enabled: true
  profiles:
    medical-tools:
      description: "Medical research tools"
      servers: ["pubmed_search"]
      max_steps: 10
      max_run_seconds: 60
```

## Testing

Comprehensive tests verify the refactoring:

### 1. Custom System Prompt Preserved
```python
def test_custom_system_prompt_preserved_with_mcp_tools():
    """Custom system prompt is augmented with tool instructions."""
    service = LLMService(
        default_chat_system_prompt="Custom instructions"
    )
    # Verify: "Custom instructions\n\nTool instructions"
```

### 2. Custom Model Parameters Preserved
```python
def test_custom_model_params_preserved_with_mcp_tools():
    """Custom model parameters flow through to chain."""
    service = LLMService(
        default_model_params={"chat": {"temperature": 0.3}}
    )
    # Verify: parameters are merged correctly
```

### 3. Subclass Override Preserved
```python
def test_custom_subclass_override_preserved():
    """Subclass _build_chat_chain override is called with MCP tools."""
    class CustomService(LLMService):
        def _build_chat_chain(self, **kwargs):
            # Custom logic here
            return super()._build_chat_chain(**kwargs)
    # Verify: override is called even with MCP tools
```

### 4. Standard Chain Fallback
```python
def test_no_mcp_tools_uses_standard_chain():
    """Standard chain is used when MCP not configured."""
    service = LLMService(...)
    # Verify: no MCP tools -> standard chain path
```

## Implementation Details

### Key Components

1. **`_chat_with_tools_via_chain`**: New method that uses factory hook
2. **System prompt augmentation**: Done before calling factory hook
3. **Model resolution**: Uses chain's resolved model name
4. **Agent creation**: Standard LangChain AgentExecutor with proper callbacks
5. **Error handling**: Graceful fallback to standard chain

### Control Flow

```
chat()
  ├─> _get_tooling_context()
  │     └─> Returns None if MCP not enabled/configured
  │
  ├─> If tooling_ctx exists:
  │     └─> _chat_with_tools_via_chain()
  │           ├─> Augment system_prompt with tool instructions
  │           ├─> _build_chat_chain() [FACTORY HOOK]
  │           ├─> Create agent with chain's model
  │           └─> Execute agent with tools
  │
  └─> Else (or on error):
        └─> _build_chat_chain() [FACTORY HOOK]
              └─> Standard chat chain
```

## Conclusion

The refactored MCP tool integration preserves the factory hook pattern, ensuring that custom configurations in subclasses work correctly even when MCP tools are enabled. This maintains the extensibility and customization capabilities of the LLMService while adding powerful tool-augmented reasoning via MCP.

**Key Takeaway**: Custom tasks can now use MCP tools without losing their custom configurations!
