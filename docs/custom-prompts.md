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

## Notes

- Built-in prompts remain unchanged; custom names fall back to the file-based loader.
- For chat, if you also provide `system_prompt=...` (string) per-call, it overrides the file for that request only.
- Template engine is Jinja-compatible; plain text content works fine.
- Dla zgodności wstecznej `completion(...)` wciąż akceptuje parametr `question` jako alias `prompt`.
