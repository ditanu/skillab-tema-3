"""
LLM Provider Registry - Singleton pattern for managing LLM providers.

Thread-safe registry that:
- Registers provider classes
- Caches provider instances (per model configuration)
- Auto-detects available providers from environment
"""
import logging
import threading
from typing import Dict, Optional, Type

from .base import LLMProvider

logger = logging.getLogger(__name__)


class LLMRegistry:
    """
    Singleton registry for LLM providers.

    Usage:
        registry = LLMRegistry()
        registry.register("anthropic", AnthropicProvider)

        # Get provider (cached)
        llm = registry.get("anthropic")
        llm = registry.get("anthropic", model="claude-opus-4-20250514")

        # Auto-detect from environment
        llm = registry.get_default()
    """

    _instance: Optional["LLMRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "LLMRegistry":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._providers: Dict[str, Type[LLMProvider]] = {}
                    instance._instances: Dict[str, LLMProvider] = {}
                    instance._default: Optional[str] = None
                    cls._instance = instance
                    logger.debug("LLMRegistry singleton created")
        return cls._instance

    def register(self, name: str, provider_class: Type[LLMProvider]) -> None:
        """
        Register a provider class.

        Args:
            name: Provider identifier (e.g., "anthropic")
            provider_class: LLMProvider subclass
        """
        if not issubclass(provider_class, LLMProvider):
            raise TypeError(f"{provider_class} must be a subclass of LLMProvider")

        with self._lock:
            self._providers[name] = provider_class
            logger.debug(f"Registered provider: {name}")

    def unregister(self, name: str) -> None:
        """Remove a provider from registry."""
        with self._lock:
            self._providers.pop(name, None)
            # Clear cached instances for this provider
            keys_to_remove = [k for k in self._instances if k.startswith(f"{name}:")]
            for key in keys_to_remove:
                del self._instances[key]

    def get(
        self,
        name: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        cached: bool = True,
    ) -> LLMProvider:
        """
        Get a provider instance.

        Args:
            name: Provider identifier
            model: Optional model override
            temperature: Temperature setting
            cached: If True, return cached instance for same config

        Returns:
            LLMProvider instance

        Raises:
            KeyError: If provider not registered
            ValueError: If provider not configured (missing API key)
        """
        if name not in self._providers:
            available = ", ".join(self._providers.keys())
            raise KeyError(f"Provider '{name}' not registered. Available: {available}")

        # Create cache key
        cache_key = f"{name}:{model or 'default'}:{temperature}"

        if cached and cache_key in self._instances:
            return self._instances[cache_key]

        # Create new instance
        provider_class = self._providers[name]
        instance = provider_class(model=model, temperature=temperature)

        if not instance.is_configured():
            raise ValueError(
                f"Provider '{name}' is not configured. "
                f"Set the required environment variables."
            )

        if cached:
            with self._lock:
                self._instances[cache_key] = instance

        logger.info(f"Created provider: {name} (model={instance.model})")
        return instance

    def get_default(self, **kwargs) -> LLMProvider:
        """
        Get the default provider (auto-detect or explicit).

        Detection order:
        1. Explicitly set default via set_default()
        2. LLM_PROVIDER environment variable
        3. First configured provider in registration order

        Returns:
            LLMProvider instance

        Raises:
            ValueError: If no provider is configured
        """
        import os

        # 1. Explicit default
        if self._default:
            return self.get(self._default, **kwargs)

        # 2. Environment variable
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider and env_provider in self._providers:
            return self.get(env_provider, **kwargs)

        # 3. First configured provider
        for name in self._providers:
            try:
                return self.get(name, **kwargs)
            except ValueError:
                continue

        raise ValueError(
            "No LLM provider configured. Set one of: "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, "
            "or configure local models."
        )

    def set_default(self, name: str) -> None:
        """Set the default provider."""
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered")
        self._default = name

    def list_registered(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    def list_configured(self) -> list[str]:
        """List providers that have required configuration."""
        configured = []
        for name, provider_class in self._providers.items():
            try:
                instance = provider_class()
                if instance.is_configured():
                    configured.append(name)
            except Exception:
                continue
        return configured

    def clear_cache(self) -> None:
        """Clear all cached instances."""
        with self._lock:
            self._instances.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._providers

    def __len__(self) -> int:
        return len(self._providers)


# === Module-level singleton and convenience functions ===

_registry: Optional[LLMRegistry] = None


def get_registry() -> LLMRegistry:
    """Get the global LLMRegistry instance."""
    global _registry
    if _registry is None:
        _registry = LLMRegistry()
        _register_default_providers(_registry)
    return _registry


def _register_default_providers(registry: LLMRegistry) -> None:
    """Register built-in providers."""
    from .providers import (
        AnthropicProvider,
        OpenAIProvider,
        GoogleProvider,
        LocalProvider,
    )

    registry.register("anthropic", AnthropicProvider)
    registry.register("openai", OpenAIProvider)
    registry.register("google", GoogleProvider)
    registry.register("local", LocalProvider)
