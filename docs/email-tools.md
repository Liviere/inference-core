# Email Tools for AI Agents

This module provides LangChain tools for AI agents to interact with email systems - both reading (IMAP) and sending (SMTP) emails.

## Overview

The email tools integrate with the existing email configuration system and provide:

- **ReadUnseenEmailsTool** - Fetch unread emails from IMAP mailbox
- **SearchEmailsTool** - Search emails by IMAP criteria
- **SendEmailTool** - Compose and send emails via SMTP
- **SummarizeEmailTool** - Summarize and classify emails using LLM
- **ListEmailAccountsTool** - List available email accounts

## Configuration

### Update `email_config.yaml`

Add IMAP configuration alongside existing SMTP settings:

```yaml
email:
  default_host: primary
  default_poll_interval_seconds: 60 # Default polling interval

  hosts:
    primary:
      # SMTP settings (existing)
      host: smtp.gmail.com
      port: 465
      use_ssl: true
      use_starttls: false
      username: ${SMTP_USERNAME}
      password_env: SMTP_PASSWORD
      from_email: support@example.com
      from_name: Support Team

      # Account metadata for agents (new)
      context: 'Customer support inbox'
      signature: |
        Best regards,
        Support Team

      # IMAP settings (new)
      imap:
        host: imap.gmail.com
        port: 993
        use_ssl: true
        username: ${IMAP_USERNAME}
        password_env: IMAP_PASSWORD
        default_folder: INBOX
        timeout: 30
        poll_interval_seconds: 45 # Override default
```

### Environment Variables

Set credentials in `.env`:

```env
SMTP_USERNAME=support@example.com
SMTP_PASSWORD=your-smtp-password
IMAP_USERNAME=support@example.com
IMAP_PASSWORD=your-imap-password
```

## Usage

### Direct Tool Usage

```python
from inference_core.agents.tools import get_email_tools
from inference_core.services.email_service import get_email_service
from inference_core.services.imap_service import get_imap_service

# Get services
email_service = get_email_service()
imap_service = get_imap_service()

# Create tools with account restrictions
tools = get_email_tools(
    email_service=email_service,
    imap_service=imap_service,
    allowed_accounts=['primary', 'support'],  # Restrict access
    default_account='primary',
    include_summarize=True,
    summarize_model='gpt-5-nano',
)
```

### Using with AgentService

```python
from inference_core.services.agents_service import AgentService
from inference_core.agents.tools import get_email_tools

# Create email tools
email_tools = get_email_tools(
    email_service=get_email_service(),
    imap_service=get_imap_service(),
    allowed_accounts=['primary'],
)

# Add to agent
agent_service = AgentService(
    agent_name="email_agent",
    tools=email_tools,
    use_checkpoints=True,
    checkpoint_config={"thread_id": session_id},
)

await agent_service.create_agent(
    system_prompt="You are an email assistant..."
)
```

### Using Tool Provider (Recommended)

Register the provider at application startup:

```python
# In main.py or startup
from inference_core.agents.tools import register_email_tools_provider

register_email_tools_provider(
    allowed_accounts=['primary', 'support'],
    default_account='primary',
)
```

Then configure in `llm_config.yaml`:

```yaml
agents:
  email_agent:
    primary: 'gpt-5-mini'
    local_tool_providers: ['email_tools']
    description: 'Email management agent'
```

## Celery Background Tasks

### Poll Single Account

```python
from inference_core.celery.tasks.email_tasks import poll_imap_task

# Poll default account
result = poll_imap_task.delay()

# Poll specific account
result = poll_imap_task.delay(
    host_alias='support',
    folder='INBOX',
    limit=50,
    callback_task='myapp.tasks.process_email',  # Optional callback
)
```

### Poll All Accounts

```python
from inference_core.celery.tasks.email_tasks import poll_all_imap_accounts_task

result = poll_all_imap_accounts_task.delay(
    callback_task='myapp.tasks.handle_new_email',
)
```

### Scheduled Polling (Celery Beat)

Add to your Celery beat schedule:

```python
# celery_main.py
celery_app.conf.beat_schedule = {
    'poll-emails-every-minute': {
        'task': 'email.poll_all_imap',
        'schedule': 60.0,  # Every minute
        'kwargs': {
            'callback_task': 'myapp.tasks.process_email',
        },
    },
}
```

## Tool Reference

### read_unseen_emails

Fetch unread emails from a mailbox.

**Arguments:**

- `account_name` (optional): Email account alias
- `folder` (optional): Mailbox folder. Default: INBOX
- `limit` (optional): Max emails to fetch. Default: 10
- `mark_as_read` (optional): Mark as read. Default: false

### search_emails

Search emails by IMAP criteria.

**Arguments:**

- `query` (required): IMAP search query. Examples:
  - `FROM sender@example.com`
  - `SUBJECT meeting`
  - `SINCE 01-Jan-2026`
  - `UNSEEN FROM boss@company.com`
- `account_name` (optional): Account alias
- `folder` (optional): Folder to search
- `limit` (optional): Max results

### send_email

Send an email.

**Arguments:**

- `to` (required): Recipient(s), comma-separated
- `subject` (required): Subject line
- `body` (required): Email body
- `account_name` (optional): Account to send from
- `cc` (optional): CC recipients
- `reply_to_message_id` (optional): For reply threading

### summarize_email

Analyze and classify email content.

**Arguments:**

- `subject` (required): Email subject
- `body` (required): Email body
- `from_address` (optional): Sender for context

**Returns:**

- `summary`: Concise summary
- `email_type`: Classification (spam, newsletter, important, transactional, personal)
- `requires_action`: Whether response needed
- `key_points`: Extracted key points

### list_email_accounts

List available email accounts and their capabilities.

**No arguments.**

## Security Considerations

1. **Account Restrictions**: Always use `allowed_accounts` to limit which accounts agents can access.

2. **Password Storage**: Passwords are stored in environment variables, never in config files.

3. **Read-Only Default**: Emails are not marked as read by default - explicit opt-in required.

4. **Signature Isolation**: Each account can have its own signature, automatically appended.

## Example: Email Processing Agent

```python
import uuid
from inference_core.services.agents_service import AgentService
from inference_core.agents.tools import (
    get_email_tools,
    generate_email_tools_system_instructions,
)

async def create_email_agent(user_id: str):
    email_tools = get_email_tools(
        email_service=get_email_service(),
        imap_service=get_imap_service(),
        allowed_accounts=['support'],
    )

    system_prompt = """You are an email assistant for customer support.
Your job is to:
1. Check for new emails
2. Summarize important ones
3. Draft responses when requested
4. Always confirm before sending

""" + generate_email_tools_system_instructions()

    agent_service = AgentService(
        agent_name="email_agent",
        tools=email_tools,
        use_checkpoints=True,
        checkpoint_config={"thread_id": str(uuid.uuid4())},
        user_id=uuid.UUID(user_id),
    )

    await agent_service.create_agent(system_prompt=system_prompt)
    return agent_service
```
