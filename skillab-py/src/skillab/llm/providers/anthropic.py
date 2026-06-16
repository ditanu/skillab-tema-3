"""
Anthropic Claude Provider - Native SDK implementation.

System message handling:
- Extracted from messages list and passed as separate `system` parameter
- This is Anthropic's recommended approach
"""
import logging
import os
from typing import AsyncIterator, Optional

from ..base import LLMProvider, Message

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider using native SDK.

    Environment variables:
        ANTHROPIC_API_KEY: Required API key
        ANTHROPIC_MODEL: Optional model override (default: claude-sonnet-4-20250514)
    """

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        model = model or os.getenv("ANTHROPIC_MODEL")
        super().__init__(model=model, temperature=temperature)

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-20250514"

    def is_configured(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def _get_client(self):
        """Lazy initialization of Anthropic client (thread-safe)."""
        with self._client_lock:
            if self._client is None:
                try:
                    import anthropic
                    self._client = anthropic.AsyncAnthropic()
                    logger.debug(f"Initialized Anthropic client for model {self.model}")
                except ImportError:
                    raise ImportError("anthropic package required. Install: pip install anthropic")
        return self._client

    def _prepare_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        """
        Extract system message and format remaining messages.

        Anthropic API expects:
        - system: str (separate parameter)
        - messages: list[{role, content}] (user/assistant only)

        Returns:
            (system_text, formatted_messages)
        """
        system = ""
        formatted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system = content
            else:
                formatted.append({"role": role, "content": content})

        return system, formatted

    async def generate(self, messages: list[Message]) -> str:
        """Generate complete response."""
        client = self._get_client()
        system, msgs = self._prepare_messages(messages)

        logger.debug(f"Anthropic generate: {len(msgs)} messages, system={bool(system)}")

        response = await client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system if system else [],
            messages=msgs,
            temperature=self.temperature,
        )

        return response.content[0].text

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Stream response chunks."""
        client = self._get_client()
        system, msgs = self._prepare_messages(messages)

        logger.debug(f"Anthropic stream: {len(msgs)} messages, system={bool(system)}")

        async with client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system if system else [],
            messages=msgs,
            temperature=self.temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text
