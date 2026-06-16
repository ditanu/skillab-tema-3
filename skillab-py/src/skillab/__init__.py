"""
SkillLab Python Utilities
Shared components for AI Agent Development Course webapps.

Modules:
    - llm: Provider-agnostic LLM abstraction with native SDKs
    - prompts: YAML-based prompt templates with Jinja2 rendering
    - db: Database utilities (PostgreSQL)
    - tools: Tool registration and wrapper utilities
"""
# LLM Module
from .llm import (
    get_llm,
    init_llm,
    get_registry,
    get_initialized_llm,
    list_configured_providers,
    list_registered_providers,
    LLMProvider,
    LLMRegistry,
    LangChainAdapter,
    wrap_langchain,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# Prompts Module
from .prompts import (
    PromptTemplate,
    PromptRegistry,
    get_prompt_registry,
    render_prompt,
)

# Database Module
from .db import get_db_url, get_engine, get_session, init_db

# Tools Module
from .tools import ToolWrapper, register_tool

__all__ = [
    # LLM - Core
    "get_llm",
    "init_llm",
    "get_registry",
    "get_initialized_llm",
    "list_configured_providers",
    "list_registered_providers",
    # LLM - Classes
    "LLMProvider",
    "LLMRegistry",
    "LangChainAdapter",
    "wrap_langchain",
    # LLM - Message types
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    # Prompts
    "PromptTemplate",
    "PromptRegistry",
    "get_prompt_registry",
    "render_prompt",
    # Database
    "get_db_url",
    "get_engine",
    "get_session",
    "init_db",
    # Tools
    "ToolWrapper",
    "register_tool",
]

# Backward compatibility alias
get_provider = get_llm
