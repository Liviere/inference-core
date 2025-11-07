"""
Example: Using Pluggable Tool Providers with LLM Service

This example demonstrates how to:
1. Create custom LangChain tools
2. Wrap them in a tool provider
3. Register the provider
4. Configure tasks to use the tools
5. Use the tools in chat interactions

Run this example:
    poetry run python examples/tool_provider_example.py
"""

import asyncio
import logging
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import Field

from inference_core.llm.tools import register_tool_provider
from inference_core.services.llm_service import LLMService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Step 1: Define Custom Tools
# ============================================================================


class CalculatorTool(BaseTool):
    """Simple calculator tool for basic arithmetic"""

    name: str = "calculator"
    description: str = "Perform basic arithmetic calculations. Input should be a mathematical expression like '2 + 2' or '10 * 5'"

    def _run(self, expression: str) -> str:
        """Execute calculation synchronously"""
        try:
            # Use a simple calculation parser instead of eval for security
            # In production, use a library like sympy or implement proper parsing
            # This is a simplified example for demonstration only
            import operator
            
            # Basic operator map for safe calculation
            ops = {
                '+': operator.add,
                '-': operator.sub,
                '*': operator.mul,
                '/': operator.truediv,
            }
            
            # Very basic parser - in production use a proper math library
            # This handles simple "num op num" expressions
            for op_str, op_func in ops.items():
                if op_str in expression:
                    parts = expression.split(op_str)
                    if len(parts) == 2:
                        try:
                            left = float(parts[0].strip())
                            right = float(parts[1].strip())
                            result = op_func(left, right)
                            return f"The result is: {result}"
                        except ValueError:
                            pass
            
            return f"Could not parse expression '{expression}'. Use format like '2 + 2' or '10 * 5'"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Execute calculation asynchronously"""
        return self._run(expression)


class WeatherTool(BaseTool):
    """Mock weather lookup tool"""

    name: str = "get_weather"
    description: str = "Get current weather for a city. Input should be a city name."

    def _run(self, city: str) -> str:
        """Get weather synchronously"""
        # In a real application, this would call a weather API
        mock_weather = {
            "london": "Rainy, 15°C",
            "new york": "Sunny, 22°C",
            "tokyo": "Cloudy, 18°C",
            "paris": "Sunny, 20°C",
        }
        weather = mock_weather.get(city.lower(), "Weather data not available")
        return f"Weather in {city}: {weather}"

    async def _arun(self, city: str) -> str:
        """Get weather asynchronously"""
        return self._run(city)


class TodoTool(BaseTool):
    """Tool for managing TODO tasks"""

    name: str = "create_todo"
    description: str = "Create a new TODO task. Input should be the task description."
    user_id: str = Field(..., description="User ID for the task")

    def _run(self, task: str) -> str:
        """Create task synchronously"""
        # In a real application, this would store the task in a database
        return f"Created TODO task for user {self.user_id}: {task}"

    async def _arun(self, task: str) -> str:
        """Create task asynchronously"""
        return self._run(task)


# ============================================================================
# Step 2: Create Tool Providers
# ============================================================================


class UtilityToolsProvider:
    """Provider for general utility tools"""

    name = "utility_tools"

    async def get_tools(self, task_type: str, user_context=None):
        """Return utility tools (calculator, weather)"""
        logger.info(f"Loading utility tools for task: {task_type}")
        return [
            CalculatorTool(),
            WeatherTool(),
        ]


class UserToolsProvider:
    """Provider for user-specific tools"""

    name = "user_tools"

    async def get_tools(self, task_type: str, user_context=None):
        """Return user-specific tools"""
        # Only provide tools if we have a user ID
        user_id = (user_context or {}).get("user_id")
        if not user_id:
            logger.warning("No user_id in context, returning no user tools")
            return []

        logger.info(f"Loading user tools for user: {user_id}")
        return [
            TodoTool(user_id=user_id),
        ]


# ============================================================================
# Step 3: Register Providers and Configure
# ============================================================================


def setup_tool_providers():
    """Register all tool providers at startup"""
    logger.info("Registering tool providers...")

    register_tool_provider(UtilityToolsProvider())
    register_tool_provider(UserToolsProvider())

    logger.info("Tool providers registered successfully")


# ============================================================================
# Step 4: Example Usage
# ============================================================================


async def example_basic_chat():
    """Example 1: Basic chat without tools"""
    print("\n" + "=" * 80)
    print("Example 1: Basic Chat (No Tools)")
    print("=" * 80)

    llm_service = LLMService()

    try:
        response = await llm_service.chat(
            session_id="example-session-1",
            user_input="Hello! How are you?",
            task_type="chat",  # This task has no tool providers configured
        )

        print(f"\nUser: Hello! How are you?")
        print(f"Assistant: {response.result['reply']}")
        print(f"\nTools used: {response.result.get('tools_used', 'None')}")

    except Exception as e:
        logger.error(f"Error in basic chat: {e}")


async def example_with_mock_config():
    """Example 2: Chat with tools (using mocked config)"""
    print("\n" + "=" * 80)
    print("Example 2: Chat with Tools (Mocked Config)")
    print("=" * 80)

    # Note: In a real application, you would configure this in llm_config.yaml
    # This example shows the concept with mocked configuration
    print("\nNOTE: This example requires proper configuration in llm_config.yaml")
    print("See the example configuration in llm_config.example.yaml")
    print("\nTo use local tool providers, add to your llm_config.yaml:")
    print("""
tasks:
  assistant_chat:
    primary: gpt-5-mini
    local_tool_providers: ['utility_tools', 'user_tools']
    tool_limits:
      max_steps: 5
      max_run_seconds: 30
    """)


async def demo_tool_capabilities():
    """Example 3: Demonstrate what the tools can do"""
    print("\n" + "=" * 80)
    print("Example 3: Tool Capabilities Demo")
    print("=" * 80)

    # Demonstrate tools directly
    calc = CalculatorTool()
    weather = WeatherTool()
    todo = TodoTool(user_id="demo-user")

    print("\n--- Calculator Tool ---")
    result = calc.run("15 * 3")
    print(f"Input: '15 * 3'")
    print(f"Output: {result}")

    print("\n--- Weather Tool ---")
    result = weather.run("London")
    print(f"Input: 'London'")
    print(f"Output: {result}")

    print("\n--- Todo Tool ---")
    result = todo.run("Review quarterly report")
    print(f"Input: 'Review quarterly report'")
    print(f"Output: {result}")


async def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("Pluggable Tool Providers Example")
    print("=" * 80)

    # Setup
    setup_tool_providers()

    # Run examples
    await demo_tool_capabilities()
    await example_basic_chat()
    await example_with_mock_config()

    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Configure tasks in llm_config.yaml with local_tool_providers")
    print("2. Register your tool providers at application startup")
    print("3. Use the configured tasks in your chat/completion calls")
    print("4. See docs/pluggable-tool-providers.md for full documentation")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
