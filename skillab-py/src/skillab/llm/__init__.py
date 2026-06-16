"""
SkillLab LLM Module - Provider-agnostic LLM abstraction.

Usage:
    from skillab.llm import get_llm

    # Auto-detect provider from environment
    llm = get_llm()

    # Explicit provider
    llm = get_llm("anthropic")
    llm = get_llm("openai", model="gpt-4-turbo")
    llm = get_llm("local", model="gemma")

    # Async streaming
    async for chunk in llm.stream(messages):
        print(chunk, end="")

    # Async complete
    response = await llm.generate(messages)

    # Sync (runs async internally)
    response = llm.generate_sync(messages)

Messages format:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
"""
from typing import Optional

from .base import LLMProvider, Message
from .registry import LLMRegistry, get_registry
from .providers.langchain import (
    LangChainAdapter,
    wrap_langchain,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

__all__ = [
    # Classes
    "LLMProvider",
    "LLMRegistry",
    "LangChainAdapter",
    "Message",
    # Message types (LangChain-compatible)
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    # Functions
    "get_llm",
    "get_registry",
    "init_llm",
    "wrap_langchain",
    "list_configured_providers",
    "list_registered_providers",
]


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    cached: bool = True,
) -> LLMProvider:
    """
    Get an LLM provider instance.

    Args:
        provider: Provider name ("anthropic", "openai", "google", "local")
                 If None, auto-detects from environment
        model: Model identifier (uses provider default if None)
        temperature: Temperature setting (default: 0.0)
        cached: Return cached instance for same config (default: True)

    Returns:
        LLMProvider instance with generate() and stream() methods

    Raises:
        KeyError: If provider not registered
        ValueError: If provider not configured

    Examples:
        llm = get_llm()  # Auto-detect
        llm = get_llm("anthropic")
        llm = get_llm("openai", model="gpt-4-turbo")
        llm = get_llm("local", model="llama")
    """
    registry = get_registry()

    if provider:
        return registry.get(provider, model=model, temperature=temperature, cached=cached)
    else:
        return registry.get_default(model=model, temperature=temperature, cached=cached)


# Singleton instance for app-wide use
_initialized_llm: Optional[LLMProvider] = None


def init_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Initialize global LLM instance for app startup.

    Args:
        provider: Provider name (auto-detect if None)
        model: Model identifier
        **kwargs: Additional provider arguments

    Returns:
        LLMProvider instance
    """
    global _initialized_llm
    _initialized_llm = get_llm(provider=provider, model=model, **kwargs)
    return _initialized_llm


def get_initialized_llm() -> LLMProvider:
    """Get the global initialized LLM (initializes if needed)."""
    global _initialized_llm
    if _initialized_llm is None:
        _initialized_llm = get_llm()
    return _initialized_llm


def list_registered_providers() -> list[str]:
    """List all registered provider names."""
    return get_registry().list_registered()


def list_configured_providers() -> list[str]:
    """List providers that have required configuration."""
    return get_registry().list_configured()
