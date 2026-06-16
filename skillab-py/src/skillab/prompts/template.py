"""
Prompt Template - Dataclass for YAML-based prompts.

Supports:
- Versioned prompts for tracking changes
- Jinja2 templating with variables
- Metadata (description, author, etc.)
"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class PromptTemplate:
    """
    Immutable prompt template loaded from YAML.

    Attributes:
        name: Unique identifier for the prompt
        version: Semantic version string (e.g., "1.0.0")
        prompt: The prompt text with Jinja2 variables (e.g., "{{ variable }}")
        description: Human-readable description of the prompt's purpose
        metadata: Additional key-value pairs (author, category, etc.)

    Example YAML:
        name: qa_system
        version: 1.0.0
        description: System prompt for Q&A assistant
        prompt: |
          You are {{ role }}, specialized in {{ domain }}.
          Answer questions step by step.
          Limit response to {{ max_words }} words.
    """

    name: str
    version: str
    prompt: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate required fields."""
        if not self.name:
            raise ValueError("PromptTemplate requires a name")
        if not self.prompt:
            raise ValueError("PromptTemplate requires a prompt")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptTemplate":
        """
        Create PromptTemplate from dictionary (e.g., parsed YAML).

        Args:
            data: Dictionary with name, version, prompt, and optional fields

        Returns:
            PromptTemplate instance
        """
        # Extract known fields
        known_fields = {"name", "version", "prompt", "description"}
        metadata = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            prompt=data.get("prompt", ""),
            description=data.get("description", ""),
            metadata=metadata,
        )

    def render(self, **variables) -> str:
        """
        Render the prompt with provided variables.

        Uses Jinja2 templating.

        Args:
            **variables: Key-value pairs to substitute in template

        Returns:
            Rendered prompt string

        Example:
            template.render(role="expert", domain="Python", max_words=100)
        """
        try:
            from jinja2 import Template
        except ImportError:
            raise ImportError("jinja2 package required. Install: pip install jinja2")

        jinja_template = Template(self.prompt)
        return jinja_template.render(**variables)

    def get_variables(self) -> list[str]:
        """
        Extract variable names from the prompt template.

        Returns:
            List of variable names used in the template
        """
        import re
        # Match {{ variable }} patterns
        pattern = r"\{\{\s*(\w+)\s*\}\}"
        return list(set(re.findall(pattern, self.prompt)))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        result = {
            "name": self.name,
            "version": self.version,
            "prompt": self.prompt,
        }
        if self.description:
            result["description"] = self.description
        if self.metadata:
            result.update(self.metadata)
        return result
