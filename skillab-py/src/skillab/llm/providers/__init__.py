"""
LLM Providers - Native SDK implementations.

Each provider handles:
- API authentication via environment variables
- System message formatting (provider-specific)
- Streaming and non-streaming responses
"""
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .google import GoogleProvider
from .local import LocalProvider
from .langchain import (
    LangChainAdapter,
    wrap_langchain,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    SystemMessage,
)

__all__ = [
    # Native providers
    "AnthropicProvider",
    "OpenAIProvider",
    "GoogleProvider",
    "LocalProvider",
    # LangChain adapter
    "LangChainAdapter",
    "wrap_langchain",
    # Message types
    "HumanMessage",
    "AIMessage",
    "AIMessageChunk",
    "SystemMessage",
]
