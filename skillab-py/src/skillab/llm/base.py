"""
LLM Provider Base - Abstract interface for all LLM providers.

Each provider implements native SDK calls with consistent interface.
"""
import logging
import threading
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

# Simple message type - no complex classes
Message = dict[str, str]  # {"role": "system|user|assistant", "content": "..."}

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Each provider:
    - Uses native SDK (not LangChain wrappers)
    - Handles system messages in provider-specific way
    - Supports both sync and async operations
    - Implements streaming
    """

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        self.model = model or self.default_model
        self.temperature = temperature
        self._client = None
        self._client_lock = threading.Lock()

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'anthropic', 'openai')."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model for this provider."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if required environment variables are set."""
        pass

    @abstractmethod
    async def generate(self, messages: list[Message]) -> str:
        """Generate a complete response (async)."""
        pass

    @abstractmethod
    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Stream response chunks (async generator)."""
        pass

    _sync_loop = None
    _sync_loop_lock = threading.Lock()

    @classmethod
    def _get_sync_loop(cls):
        """Get or create a reusable event loop for sync calls."""
        import asyncio
        with cls._sync_loop_lock:
            if cls._sync_loop is None or cls._sync_loop.is_closed():
                cls._sync_loop = asyncio.new_event_loop()
            return cls._sync_loop

    def generate_sync(self, messages: list[Message]) -> str:
        """Synchronous generate - runs async in reusable event loop."""
        loop = self._get_sync_loop()
        return loop.run_until_complete(self.generate(messages))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
