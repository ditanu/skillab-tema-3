"""
Prompt Registry - Load and manage YAML prompt templates.

Features:
- Load all .yaml files from a directory
- Render prompts with Jinja2 variables
- Version tracking for prompts
- Thread-safe singleton pattern
"""
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

from .template import PromptTemplate

logger = logging.getLogger(__name__)


class PromptRegistry:
    """
    Registry for loading and rendering prompt templates from YAML files.

    Usage:
        # Load from directory
        registry = PromptRegistry("/path/to/prompts")

        # Get rendered prompt
        prompt = registry.render("qa_system", role="expert", domain="Python")

        # Get template object
        template = registry.get("qa_system")
        print(template.version)
    """

    def __init__(self, folder: Optional[str] = None):
        """
        Initialize registry and optionally load from folder.

        Args:
            folder: Path to directory containing .yaml files
        """
        self._templates: Dict[str, PromptTemplate] = {}
        self._lock = threading.Lock()

        if folder:
            self.load_folder(folder)

    def load_folder(self, folder: str) -> int:
        """
        Load all .yaml files from a directory.

        Args:
            folder: Path to directory

        Returns:
            Number of templates loaded

        Raises:
            FileNotFoundError: If folder doesn't exist
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml package required. Install: pip install pyyaml")

        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Prompt folder not found: {folder}")

        count = 0
        for path in folder_path.rglob("*.yaml"):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if data and isinstance(data, dict):
                    template = PromptTemplate.from_dict(data)
                    self.register(template)
                    count += 1
                    logger.debug(f"Loaded prompt: {template.name} v{template.version}")
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        logger.info(f"Loaded {count} prompt templates from {folder}")
        return count

    def register(self, template: PromptTemplate) -> None:
        """
        Register a prompt template.

        Args:
            template: PromptTemplate instance
        """
        with self._lock:
            if template.name in self._templates:
                existing = self._templates[template.name]
                logger.debug(
                    f"Replacing prompt '{template.name}' "
                    f"v{existing.version} -> v{template.version}"
                )
            self._templates[template.name] = template

    def register_inline(
        self,
        name: str,
        prompt: str,
        version: str = "1.0.0",
        description: str = "",
    ) -> PromptTemplate:
        """
        Register a prompt from inline definition.

        Args:
            name: Unique identifier
            prompt: Prompt text with Jinja2 variables
            version: Version string
            description: Optional description

        Returns:
            Created PromptTemplate
        """
        template = PromptTemplate(
            name=name,
            version=version,
            prompt=prompt,
            description=description,
        )
        self.register(template)
        return template

    def get(self, name: str) -> PromptTemplate:
        """
        Get a prompt template by name.

        Args:
            name: Template identifier

        Returns:
            PromptTemplate instance

        Raises:
            KeyError: If template not found
        """
        if name not in self._templates:
            available = ", ".join(self._templates.keys()) or "(none)"
            raise KeyError(f"Prompt '{name}' not found. Available: {available}")
        return self._templates[name]

    def render(self, name: str, **variables) -> str:
        """
        Get and render a prompt with variables.

        Args:
            name: Template identifier
            **variables: Variables to substitute

        Returns:
            Rendered prompt string

        Example:
            prompt = registry.render(
                "qa_system",
                role="expert",
                domain="Python",
                max_words=100
            )
        """
        template = self.get(name)
        return template.render(**variables)

    def list_templates(self) -> list[str]:
        """List all registered template names."""
        return list(self._templates.keys())

    def list_with_versions(self) -> dict[str, str]:
        """List all templates with their versions."""
        return {name: tpl.version for name, tpl in self._templates.items()}

    def unregister(self, name: str) -> bool:
        """
        Remove a template from registry.

        Args:
            name: Template identifier

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._templates:
                del self._templates[name]
                return True
            return False

    def clear(self) -> None:
        """Remove all templates."""
        with self._lock:
            self._templates.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._templates

    def __len__(self) -> int:
        return len(self._templates)

    def __iter__(self):
        return iter(self._templates.values())


# === Global registry instance ===

_global_registry: Optional[PromptRegistry] = None
_global_lock = threading.Lock()


def get_prompt_registry(folder: Optional[str] = None) -> PromptRegistry:
    """
    Get or create the global PromptRegistry.

    Args:
        folder: Path to load prompts from (only used on first call)

    Returns:
        Global PromptRegistry instance
    """
    global _global_registry

    if _global_registry is None:
        with _global_lock:
            if _global_registry is None:
                _global_registry = PromptRegistry(folder)

    return _global_registry


def render_prompt(name: str, **variables) -> str:
    """
    Convenience function to render from global registry.

    Args:
        name: Template identifier
        **variables: Variables to substitute

    Returns:
        Rendered prompt string
    """
    return get_prompt_registry().render(name, **variables)
