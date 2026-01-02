"""
Simple Math MCP Server for Testing

A minimal MCP server that provides basic math operations (add, multiply).
Used for integration testing without external dependencies.

Source: MCP Python SDK – FastMCP quickstart examples
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool(structured_output=True)
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool(structured_output=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    # Run server with stdio transport
    import sys

    mcp.run(transport="stdio")
