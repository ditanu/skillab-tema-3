"""
Local LLM Provider - Via LiteLLM proxy to Ollama models.

System message handling:
- Uses OpenAI-compatible format (system role supported)
- LiteLLM routes to Ollama which handles system messages natively

Available model aliases:
- qwen (qwen2.5:3b)
- gemma (gemma2:2b)
- llama (llama3.2:3b)
"""
import logging
import os
from typing import AsyncIterator, Optional

from ..base import LLMProvider, Message

logger = logging.getLogger(__name__)


class LocalProvider(LLMProvider):
    """
    Local LLM provider via LiteLLM proxy server.

    Connects to LiteLLM Docker service which routes to Ollama models.
    Uses OpenAI-compatible API format.

    Environment variables:
        LITELLM_BASE_URL: API endpoint (default: http://localhost:4000)
        LOCAL_MODEL: Model alias (default: qwen)
    """

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        model = model or os.getenv("LOCAL_MODEL")
        super().__init__(model=model, temperature=temperature)

    @property
    def name(self) -> str:
        return "local"

    @property
    def default_model(self) -> str:
        return "qwen"

    @property
    def base_url(self) -> str:
        return os.getenv("LITELLM_BASE_URL", "http://localhost:4000")

    def is_configured(self) -> bool:
        """Check if LiteLLM server is reachable."""
        try:
            import httpx
            response = httpx.get(
                f"{self.base_url}/health/liveliness",
                timeout=2.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Local provider not configured: {e}")
            return False

    def _get_client(self):
        """Lazy initialization of OpenAI-compatible client for LiteLLM (thread-safe)."""
        with self._client_lock:
            if self._client is None:
                try:
                    import openai
                    self._client = openai.AsyncOpenAI(
                        base_url=f"{self.base_url}/v1",
                        api_key="local",  # LiteLLM doesn't require real key
                    )
                    logger.debug(f"Initialized Local client for model {self.model}")
                except ImportError:
                    raise ImportError("openai package required. Install: pip install openai")
        return self._client

    def _prepare_messages(self, messages: list[Message]) -> list[dict]:
        """
        Format messages for OpenAI-compatible API.

        LiteLLM/Ollama supports system role natively.

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

        logger.debug(f"Local generate: {len(msgs)} messages, model={self.model}")

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

        logger.debug(f"Local stream: {len(msgs)} messages, model={self.model}")

        stream = await client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
