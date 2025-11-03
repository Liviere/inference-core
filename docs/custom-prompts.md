# Custom Prompts (Jinja2)

You can provide your own prompt templates without modifying the core `prompts.py` by placing files in `inference_core/custom_prompts/`.

## Layout

- Completion templates: `inference_core/custom_prompts/completion/<name>.j2`
  - Should reference `{prompt}` as input variable
- Chat system prompts: `inference_core/custom_prompts/chat/<name>.j2` or `<name>.system.j2`
  - Used as the system message; the chat prompt is composed as: system → history → human(`{user_input}`)

## Usage

- Programmatic (per call):

  ```python
  await llm.completion(prompt="...", prompt_name="short_answer")
  await llm.chat(session_id="s1", user_input="hi", prompt_name="tutor")
  ```

- Programmatic with multiple variables (input_vars):

  ```python
  # Completion template can reference {prompt}, {topic}, {tone}, etc.
  await llm.completion(
    input_vars={
      "prompt": "Describe photosynthesis",
      "topic": "biology",
      "tone": "simplified",
    },
    prompt_name="simple_explainer",
  )

  # Chat template can use {user_input} plus any additional fields
  await llm.chat(
    session_id="s1",
    user_input="How does JWT work?",  # main field for the conversation history
    input_vars={
      "context": "Backend FastAPI",
      "audience": "junior dev",
    },
    prompt_name="tutor",
  )
  ```

- Service defaults (subclass `LLMService`):

  ```python
  class MyLLM(LLMService):
      def __init__(self):
          super().__init__(
              default_prompt_names={"completion": "simple_explainer", "chat": "tutor"}
          )
  ```

- Streaming:
  ```python
  await llm.stream_completion(prompt="...", prompt_name="short_answer")
  await llm.stream_chat(session_id="s1", user_input="hi", prompt_name="sales_assistant")
  ```

## Custom task types (task_type) and prompts

You can route requests via a custom `task_type` (e.g., `summarization`) declared in `llm_config.yaml` (under `tasks:`). This changes how models are chosen (primary/fallback/testing), and you can also attach default prompt names per task in your service instance.

- Config example (`llm_config.yaml`):

  ```yaml
  tasks:
    summarization:
      primary: 'claude-3-5-haiku-latest'
      fallback: ['gpt-5-nano']
      testing: ['gpt-5-nano']
  ```

- Programmatic usage with default prompts per task:

  ```python
  llm = LLMService().copy_with(
      default_prompt_names={
          "summarization": "summary_short",  # completion template name
          "chat": "tutor",
      }
  )

  # Completion using the custom task type and template
  await llm.completion(task_type="summarization", input_vars={"prompt": "..."})
  ```

Notes:

- `task_type` is optional; without it, the core uses built-in tasks: `completion` and `chat`.
- File placement still follows the two folders (`custom_prompts/completion` and `custom_prompts/chat`). Mapping from task type → template name comes from `default_prompt_names` in your `LLMService` instance.
- For chat, you can also override `system_prompt` per-call, which takes precedence over the file.

## Notes

- Built-in prompts remain unchanged; custom names fall back to the file-based loader.
- For chat, if you also provide `system_prompt=...` (string) per-call, it overrides the file for that request only.
- Template engine is Jinja-compatible; plain text content works fine.
- For backward compatibility, `completion(...)` still accepts the `question` parameter as an alias for `prompt`.

## MCP tool instruction templates (Jinja2)

You can inject additional, configurable instructions specifically for MCP tool-enabled chats using Jinja2 templates placed in `inference_core/custom_prompts/mcp/`.

- Location: `inference_core/custom_prompts/mcp/`
- File naming (first match wins):
  - `<profile>.j2`, `<profile>.jinja2`, `<profile>.tmpl`

Template context variables available:

- `profile_name` – active MCP profile name
- `tools` – list of tool dicts with keys: `name`, `description`, `args_schema`
- `limits` – dict with keys such as `max_steps`, `max_run_seconds`, `tool_retry_attempts`, `allowlist_hosts`, `rate_limits`

Example `inference_core/custom_prompts/mcp/profile_name.j2`:

```
{# Extra guidance for MCP sessions #}
Custom MCP Instructions for profile "{{ profile_name }}":
- Prefer robust selectors and re-locate elements after navigation.
- If an action triggers navigation, wait for the page to settle (use wait tool) before interacting again.
```

The rendered text is appended to the auto-generated tool instructions.
