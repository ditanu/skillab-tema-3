"""
LangChain Adapter - Provides LangChain-compatible interface over native providers.

This adapter allows using skillab providers with LangChain's interface:
- .invoke(messages) -> AIMessage
- .stream(messages) -> Iterator[AIMessageChunk]

Handles sync/async bridging correctly:
- Detects if running in existing event loop (FastAPI, Jupyter)
- Uses thread pool for sync calls in async context
- True streaming with queue-based sync/async bridge
"""
import asyncio
import logging
import queue
import threading
from typing import Any, Iterator, Optional

from ..base import LLMProvider, Message

logger = logging.getLogger(__name__)


def _run_async(coro):
    """
    Run async coroutine from sync context, handling existing event loops.

    Works correctly in:
    - Plain Python scripts
    - FastAPI endpoints
    - Jupyter notebooks
    - Any context with or without running event loop
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)

    # Already in async context - run in thread pool
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


def _stream_async_to_sync(async_gen_func, *args, **kwargs) -> Iterator[str]:
    """
    Bridge async generator to sync iterator with true streaming.

    Uses a queue to pass chunks from async thread to sync iterator.
    Chunks are yielded as soon as they arrive - no buffering.
    """
    result_queue: queue.Queue = queue.Queue()
    exception_holder: list = []

    def run_async_gen():
        async def consume():
            try:
                async for chunk in async_gen_func(*args, **kwargs):
                    result_queue.put(("chunk", chunk))
                result_queue.put(("done", None))
            except Exception as e:
                exception_holder.append(e)
                result_queue.put(("error", e))

        asyncio.run(consume())

    # Start async generator in separate thread
    thread = threading.Thread(target=run_async_gen, daemon=True)
    thread.start()

    # Yield chunks as they arrive
    while True:
        try:
            msg_type, value = result_queue.get(timeout=60.0)

            if msg_type == "chunk":
                yield value
            elif msg_type == "done":
                break
            elif msg_type == "error":
                raise value

        except queue.Empty:
            raise TimeoutError("Stream timeout - no data received for 60 seconds")

    thread.join(timeout=1.0)


class LangChainAdapter:
    """
    Adapter that wraps any LLMProvider with LangChain-compatible interface.

    Provides the same method signatures as LangChain's BaseChatModel:
    - invoke() / ainvoke() - complete response
    - stream() / astream() - streaming response

    Usage:
        from skillab.llm import get_llm, wrap_langchain

        llm = wrap_langchain(get_llm("anthropic"))

        # Sync (works in FastAPI, Jupyter, scripts)
        response = llm.invoke(messages)

        # True streaming
        for chunk in llm.stream(messages):
            print(chunk.content, end="", flush=True)

        # Async
        response = await llm.ainvoke(messages)
        async for chunk in llm.astream(messages):
            print(chunk.content, end="", flush=True)
    """

    def __init__(self, provider: LLMProvider):
        """
        Create adapter wrapping an LLMProvider.

        Args:
            provider: Any LLMProvider instance
        """
        self.provider = provider

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def name(self) -> str:
        return self.provider.name

    def _convert_to_dicts(self, messages: list[Any]) -> list[Message]:
        """
        Convert various message formats to simple dicts.

        Accepts:
        - list[dict] - already in correct format
        - list[BaseMessage] - LangChain message objects
        - list[tuple] - (role, content) pairs
        """
        result = []

        for msg in messages:
            if isinstance(msg, dict):
                result.append(msg)
            elif isinstance(msg, tuple) and len(msg) == 2:
                result.append({"role": msg[0], "content": msg[1]})
            elif hasattr(msg, "type") and hasattr(msg, "content"):
                # LangChain BaseMessage-like object
                role = getattr(msg, "type", "user")
                role_map = {
                    "human": "user",
                    "ai": "assistant",
                    "system": "system",
                }
                result.append({
                    "role": role_map.get(role, role),
                    "content": msg.content,
                })
            else:
                raise ValueError(f"Unsupported message format: {type(msg)}")

        return result

    def invoke(self, messages: list[Any], **kwargs) -> "AIMessage":
        """
        LangChain-compatible invoke method (sync).

        Works correctly in any context (FastAPI, Jupyter, scripts).

        Args:
            messages: List of messages (dicts, BaseMessage, or tuples)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            AIMessage with response content
        """
        msgs = self._convert_to_dicts(messages)
        content = _run_async(self.provider.generate(msgs))
        return AIMessage(content=content)

    async def ainvoke(self, messages: list[Any], **kwargs) -> "AIMessage":
        """Async version of invoke."""
        msgs = self._convert_to_dicts(messages)
        content = await self.provider.generate(msgs)
        return AIMessage(content=content)

    def stream(self, messages: list[Any], **kwargs) -> Iterator["AIMessageChunk"]:
        """
        LangChain-compatible stream method (sync).

        True streaming - yields chunks as soon as they arrive from the API.
        Works correctly in any context.

        Args:
            messages: List of messages
            **kwargs: Additional arguments (ignored)

        Yields:
            AIMessageChunk for each streamed piece
        """
        msgs = self._convert_to_dicts(messages)

        async def async_stream(messages):
            async for chunk in self.provider.stream(messages):
                yield chunk

        for chunk in _stream_async_to_sync(async_stream, msgs):
            yield AIMessageChunk(content=chunk)

    async def astream(self, messages: list[Any], **kwargs):
        """Async version of stream."""
        msgs = self._convert_to_dicts(messages)
        async for chunk in self.provider.stream(msgs):
            yield AIMessageChunk(content=chunk)


# === Simple message classes (LangChain-compatible interface) ===


class BaseMessage:
    """Base class for message types."""

    def __init__(self, content: str, type: str = "base"):
        self.content = content
        self.type = type

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"{self.__class__.__name__}(content={preview!r})"

    def __str__(self) -> str:
        return self.content


class HumanMessage(BaseMessage):
    """User/human message."""

    def __init__(self, content: str):
        super().__init__(content=content, type="human")


class AIMessage(BaseMessage):
    """AI/assistant message."""

    def __init__(self, content: str):
        super().__init__(content=content, type="ai")


class AIMessageChunk(BaseMessage):
    """Streaming chunk from AI."""

    def __init__(self, content: str):
        super().__init__(content=content, type="ai")


class SystemMessage(BaseMessage):
    """System message."""

    def __init__(self, content: str):
        super().__init__(content=content, type="system")


# === Factory function ===


def wrap_langchain(provider: LLMProvider) -> LangChainAdapter:
    """
    Wrap any LLMProvider with LangChain-compatible interface.

    Args:
        provider: LLMProvider instance

    Returns:
        LangChainAdapter with .invoke() and .stream() methods

    Example:
        from skillab import get_llm, wrap_langchain

        llm = wrap_langchain(get_llm())

        # Like LangChain
        response = llm.invoke(messages)
        for chunk in llm.stream(messages):
            print(chunk.content, end="")
    """
    return LangChainAdapter(provider)
