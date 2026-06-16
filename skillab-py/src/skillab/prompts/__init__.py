"""
SkillLab Prompts Module - YAML-based prompt management.

Usage:
    from skillab.prompts import PromptRegistry, PromptTemplate

    # Load from folder
    registry = PromptRegistry("/path/to/prompts")

    # Render with variables
    prompt = registry.render("qa_system", role="expert", domain="Python")

    # Or use inline
    registry.register_inline(
        name="greeting",
        prompt="Hello {{ name }}, welcome to {{ place }}!",
        version="1.0.0"
    )

YAML Format:
    name: qa_system
    version: 1.0.0
    description: System prompt for Q&A assistant
    prompt: |
      You are {{ role }}, specialized in {{ domain }}.
      Answer questions step by step.
"""
from .template import PromptTemplate
from .registry import (
    PromptRegistry,
    get_prompt_registry,
    render_prompt,
)

__all__ = [
    "PromptTemplate",
    "PromptRegistry",
    "get_prompt_registry",
    "render_prompt",
]
