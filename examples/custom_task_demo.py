#!/usr/bin/env python3
"""
Example script demonstrating custom task usage logging with inference-core.

This script shows how to use the generic usage/cost logging abstraction
for custom LLM tasks like entity extraction, summarization, and classification.

Run with:
    poetry run python examples/custom_task_demo.py
"""

import asyncio
import uuid

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Import the generic usage logging helpers
from inference_core.llm.custom_task import run_with_usage, stream_with_usage
from inference_core.llm.models import get_model_factory


async def example_entity_extraction():
    """Example: Entity extraction with usage logging"""
    print("\n" + "=" * 60)
    print("Example 1: Entity Extraction with Usage Logging")
    print("=" * 60)

    # Create extraction chain
    factory = get_model_factory()
    model = factory.create_model("gpt-5-nano")  # Using test model

    extraction_prompt = ChatPromptTemplate.from_template("""
Extract named entities from the following text and list them:

Text: {text}

Entities:
""")

    chain = extraction_prompt | model | StrOutputParser()

    # Execute with automatic usage logging
    result = await run_with_usage(
        task_type="entity_extraction",
        runnable=chain,
        input={"text": "Apple Inc. announced new products in California yesterday."},
        model_name="gpt-5-nano",
        session_id=f"demo-extraction-{uuid.uuid4()}",
        request_id=f"req-{uuid.uuid4()}",
    )

    print(f"\nExtracted entities:\n{result}")
    print("\n✓ Usage logged to database with task_type='entity_extraction'")


async def example_summarization_streaming():
    """Example: Streaming summarization with usage logging"""
    print("\n" + "=" * 60)
    print("Example 2: Streaming Summarization with Usage Logging")
    print("=" * 60)

    # Create streaming summarization chain
    factory = get_model_factory()
    streaming_model = factory.create_model("gpt-5-nano", streaming=True)

    summarization_prompt = ChatPromptTemplate.from_template("""
Provide a brief summary of the following text:

{document}

Summary:
""")

    chain = summarization_prompt | streaming_model | StrOutputParser()

    # Stream with automatic usage logging
    print("\nStreaming summary:")
    full_summary = ""
    async for chunk in stream_with_usage(
        task_type="summarization",
        runnable=chain,
        input={
            "document": "LLMs have revolutionized natural language processing. "
            "They can perform various tasks like translation, summarization, "
            "and question answering with high accuracy."
        },
        model_name="gpt-5-nano",
        session_id=f"demo-summary-{uuid.uuid4()}",
    ):
        print(chunk, end="", flush=True)
        full_summary += chunk

    print("\n\n✓ Usage logged with task_type='summarization' and streamed=True")


async def example_sentiment_analysis():
    """Example: Sentiment analysis with usage logging"""
    print("\n" + "=" * 60)
    print("Example 3: Sentiment Analysis with Usage Logging")
    print("=" * 60)

    # Create sentiment analysis chain
    factory = get_model_factory()
    model = factory.create_model("gpt-5-nano")

    sentiment_prompt = ChatPromptTemplate.from_template("""
Analyze the sentiment of the following text.
Respond with only: positive, negative, or neutral

Text: {text}

Sentiment:
""")

    chain = sentiment_prompt | model | StrOutputParser()

    # Analyze multiple texts with usage logging
    texts = [
        "This product is amazing! I love it!",
        "Terrible experience, would not recommend.",
        "It's okay, nothing special.",
    ]

    print("\nAnalyzing sentiments:")
    for i, text in enumerate(texts, 1):
        result = await run_with_usage(
            task_type="sentiment_analysis",
            runnable=chain,
            input={"text": text},
            model_name="gpt-5-nano",
            request_id=f"sentiment-{i}-{uuid.uuid4()}",
        )
        print(f"\n{i}. Text: {text[:40]}...")
        print(f"   Sentiment: {result.strip()}")

    print("\n✓ All 3 requests logged with task_type='sentiment_analysis'")


async def example_error_handling():
    """Example: Error handling with usage logging"""
    print("\n" + "=" * 60)
    print("Example 4: Error Handling")
    print("=" * 60)

    # Create a chain that might fail
    factory = get_model_factory()
    model = factory.create_model("gpt-5-nano")

    chain = ChatPromptTemplate.from_template("{invalid}") | model | StrOutputParser()

    print("\nAttempting task with invalid input...")
    try:
        await run_with_usage(
            task_type="test_error",
            runnable=chain,
            input={"text": "This won't match the template"},
            model_name="gpt-5-nano",
        )
    except Exception as e:
        print(f"\n✓ Error caught: {type(e).__name__}")
        print("✓ Usage logged with success=False and error details")


async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Custom Task Usage Logging Examples")
    print("=" * 60)
    print("\nThis demo shows how to use the generic usage/cost logging")
    print("abstraction for custom LLM tasks.")
    print("\nNote: These examples use mock models for demonstration.")
    print("In production, use real models with API keys configured.")

    try:
        # Run examples
        await example_entity_extraction()
        await example_summarization_streaming()
        await example_sentiment_analysis()
        await example_error_handling()

        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print("\nAll usage data has been logged to the database.")
        print("Check the llm_request_logs table to see the logged data.")
        print("\nFor more examples and documentation, see:")
        print("  docs/custom-task-usage-logging.md")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
