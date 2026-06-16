"""
Google Gemini Provider - Native SDK implementation.

System message handling:
- Prepended to the first user message content
- Google's API doesn't have a separate system role
- Role mapping: assistant -> model
"""
import logging
import os
from typing import AsyncIterator, Optional

from ..base import LLMProvider, Message

logger = logging.getLogger(__name__)


class GoogleProvider(LLMProvider):
    """
    Google Gemini provider using native google-genai SDK.

    Environment variables:
        GOOGLE_API_KEY: Required API key
        GOOGLE_MODEL: Optional model override (default: gemini-2.0-flash)
    """

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        model = model or os.getenv("GOOGLE_MODEL")
        super().__init__(model=model, temperature=temperature)

    @property
    def name(self) -> str:
        return "google"

    @property
    def default_model(self) -> str:
        return "gemini-2.0-flash"

    def is_configured(self) -> bool:
        return bool(os.getenv("GOOGLE_API_KEY"))

    def _get_client(self):
        """Lazy initialization of Google client (thread-safe)."""
        with self._client_lock:
            if self._client is None:
                try:
                    from google import genai
                    self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
                    logger.debug(f"Initialized Google client for model {self.model}")
                except ImportError:
                    raise ImportError("google-genai package required. Install: pip install google-genai")
        return self._client

    def _prepare_messages(self, messages: list[Message]) -> list[dict]:
        """
        Format messages for Google API.

        Google API expects:
        - contents: list[{role, parts}]
        - role: "user" or "model" (not "assistant")
        - System message must be prepended to first user message

        Returns:
            formatted_contents
        """
        contents = []
        system_text = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_text = content
            elif role == "user":
                # Prepend system to first user message
                text = content
                if system_text:
                    text = f"{system_text}\n\n{text}"
                    system_text = ""
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                # Google uses "model" instead of "assistant"
                contents.append({"role": "model", "parts": [{"text": content}]})

        return contents

    async def generate(self, messages: list[Message]) -> str:
        """Generate complete response."""
        client = self._get_client()
        contents = self._prepare_messages(messages)
        logger.debug(f"Google generate: {len(contents)} messages")

        response = await client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config={"temperature": self.temperature},
        )

        return response.text

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Stream response chunks."""
        client = self._get_client()
        contents = self._prepare_messages(messages)

        logger.debug(f"Google stream: {len(contents)} messages")

        async for chunk in await client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config={"temperature": self.temperature},
        ):
            if chunk.text:
                yield chunk.text
