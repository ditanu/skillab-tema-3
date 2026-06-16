"""
OpenAI GPT Provider - Native SDK implementation.

System message handling:
- Included directly in messages array with role="system"
- OpenAI natively supports system role in messages
"""
import logging
import os
from typing import AsyncIterator, Optional

from ..base import LLMProvider, Message

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI GPT provider using native SDK.

    Environment variables:
        OPENAI_API_KEY: Required API key
        OPENAI_MODEL: Optional model override (default: gpt-4o)
    """

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        model = model or os.getenv("OPENAI_MODEL")
        super().__init__(model=model, temperature=temperature)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return "gpt-4-turbo"

    def is_configured(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def _get_client(self):
        """Lazy initialization of OpenAI client (thread-safe)."""
        with self._client_lock:
            if self._client is None:
                try:
                    import openai
                    self._client = openai.AsyncOpenAI()
                    logger.debug(f"Initialized OpenAI client for model {self.model}")
                except ImportError:
                    raise ImportError("openai package required. Install: pip install openai")
        return self._client

    def _prepare_messages(self, messages: list[Message]) -> list[dict]:
        """
        Format messages for OpenAI API.

        OpenAI API expects:
        - messages: list[{role, content}] (system/user/assistant all supported)

        Returns:
            formatted_messages
        """
        return [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in messages
        ]

    async def generate(self, messages: list[Message]) -> str:
        """Generate complete response."""
        client = self._get_client()
        msgs = self._prepare_messages(messages)

        logger.debug(f"OpenAI generate: {len(msgs)} messages")

        response = await client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.temperature,
        )

        return response.choices[0].message.content

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Stream response chunks."""
        client = self._get_client()
        msgs = self._prepare_messages(messages)

        logger.debug(f"OpenAI stream: {len(msgs)} messages")

        stream = await client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
