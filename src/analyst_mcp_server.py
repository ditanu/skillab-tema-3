"""
MCP server pentru agentii L6.

Expune doua tool-uri MCP:
  - data_analyst: Analyst + NL2SQL pentru intrebari analitice peste tabele
  - orchestrator_rag: Orchestrator Supervisor + RAG pentru intrebari peste documente

Pentru fiecare tool:
  - tools/list: publica schema de input si contractul de output
  - tools/call: valideaza argumentele, apeleaza agentul si returneaza JSON text

Rulare locala (stdio, recomandat pentru clienti MCP):
    python src/analyst_mcp_server.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

import mcp.types as types
from dotenv import load_dotenv
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from pydantic import BaseModel, ConfigDict, Field

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
SKILLAB_SRC = ROOT_DIR / "skillab-py" / "src"

sys.path.insert(0, str(SKILLAB_SRC))
sys.path.insert(0, str(Path(__file__).parent))

from analyst_agent import AnalystAgent  # noqa: E402
from mcp_guardrails import validate_arguments_shape, validate_safe_text  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from skillab import get_llm  # noqa: E402
from state import OrchestratorState  # noqa: E402

load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


SERVER_NAME = "l6-agents"
SERVER_VERSION = "1.0.0"
DATA_ANALYST_TOOL_NAME = "data_analyst"
ORCHESTRATOR_TOOL_NAME = "orchestrator_rag"
DEFAULT_DB_URL = "postgresql://demo:demo123@localhost:5433/rag_demo"
DATA_ANALYST_ALLOWED_FIELDS = {"question", "include_plan", "include_preview", "max_preview_rows"}
ORCHESTRATOR_ALLOWED_FIELDS = {"query", "include_rag_context"}
MAX_USER_TEXT_CHARS = 2_000

TABLES_CONFIG = {
    "achizitii_directe": {
        "schema_path": str(DATA_DIR / "nl2sql_agent" / "schema_achizitii_directe.json"),
        "business_path": str(DATA_DIR / "nl2sql_agent" / "business_achizitii_directe.json"),
    },
    "anunturi_initiere": {
        "schema_path": str(DATA_DIR / "nl2sql_agent" / "schema_anunturi_initiere.json"),
        "business_path": str(DATA_DIR / "nl2sql_agent" / "business_anunturi_initiere.json"),
    },
}


class DataAnalystInput(BaseModel):
    """Schema input pentru tool-ul MCP."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=MAX_USER_TEXT_CHARS,
        description="Intrebarea de analiza in limbaj natural.",
    )
    include_plan: bool = Field(
        default=True,
        description="Include planul si rezultatele pasilor executati.",
    )
    include_preview: bool = Field(
        default=True,
        description="Include un preview tabelar pentru slice-ul final.",
    )
    max_preview_rows: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Numarul maxim de randuri returnate in preview.",
    )


class FinalPreview(BaseModel):
    """Preview pentru ultimul DataFrame produs de agent."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    row_count: int
    columns: list[str]
    rows: list[dict[str, Any]]


class DataAnalystOutput(BaseModel):
    """Schema output pentru rezultatul serializat in TextContent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "failed", "no_plan"] = Field(
        description="Statusul final al agentului."
    )
    answer: str = Field(description="Raspunsul sintetizat de Data Analyst Agent.")
    reasoning: str = Field(default="", description="Rationamentul folosit la planificare.")
    plan: list[dict[str, Any]] = Field(default_factory=list)
    step_results: list[dict[str, Any]] = Field(default_factory=list)
    final_preview: FinalPreview | None = None


DATA_ANALYST_INPUT_SCHEMA = DataAnalystInput.model_json_schema()
DATA_ANALYST_OUTPUT_SCHEMA = DataAnalystOutput.model_json_schema()


class OrchestratorInput(BaseModel):
    """Schema input pentru tool-ul MCP Orchestrator."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=MAX_USER_TEXT_CHARS,
        description="Intrebarea pentru Orchestrator Supervisor + RAG.",
    )
    include_rag_context: bool = Field(
        default=True,
        description="Include metadate despre rezultatul RAG folosit de Orchestrator.",
    )


class RAGContextSummary(BaseModel):
    """Rezumat serializabil pentru rezultatul RAG final."""

    model_config = ConfigDict(extra="forbid")

    query_used: str = ""
    result_count: int = 0
    max_score: float = 0.0
    avg_score: float = 0.0
    sources: list[dict[str, Any]] = Field(default_factory=list)


class OrchestratorOutput(BaseModel):
    """Schema output pentru rezultatul Orchestrator serializat in TextContent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["pending", "success", "partial", "failed"] = Field(
        description="Statusul final al Orchestrator Agent."
    )
    answer: str = Field(description="Raspunsul final sintetizat de Orchestrator.")
    iteration: int = Field(default=0, description="Numarul de iteratii RAG executate.")
    feedback: dict[str, Any] | None = None
    rag_context: RAGContextSummary | None = None


ORCHESTRATOR_INPUT_SCHEMA = OrchestratorInput.model_json_schema()
ORCHESTRATOR_OUTPUT_SCHEMA = OrchestratorOutput.model_json_schema()

server = Server(SERVER_NAME)
_analyst: AnalystAgent | None = None
_orchestrator_app: Any | None = None


def _resolve_provider(provider: str | None) -> str | None:
    aliases = {"gemini": "google", "ollama": "local"}
    return aliases.get(provider.lower(), provider.lower()) if provider else None


def _get_model_from_env(provider: str | None) -> str | None:
    resolved = _resolve_provider(provider)
    if not resolved:
        return None

    model_env_vars = {
        "google": "GOOGLE_MODEL",
        "anthropic": "ANTHROPIC_MODEL",
        "openai": "OPENAI_MODEL",
        "local": "OLLAMA_MODEL",
    }
    return os.getenv("LLM_MODEL") or os.getenv(model_env_vars.get(resolved, f"{resolved.upper()}_MODEL"))


def get_analyst() -> AnalystAgent:
    """Initializeaza agentul o singura data, la primul tool call."""
    global _analyst

    if _analyst is None:
        provider = _resolve_provider(os.getenv("LLM_PROVIDER"))
        model = _get_model_from_env(os.getenv("LLM_PROVIDER"))
        db_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

        logger.info("Initializing Data Analyst Agent with provider=%s model=%s", provider, model)
        llm = get_llm(provider=provider, model=model)
        _analyst = AnalystAgent(
            tables_config=TABLES_CONFIG,
            db_url=db_url,
            llm=llm,
        )

    return _analyst


def get_orchestrator_app() -> Any:
    """Initializeaza graful Orchestrator o singura data, la primul tool call."""
    global _orchestrator_app

    if _orchestrator_app is None:
        provider = _resolve_provider(os.getenv("LLM_PROVIDER"))
        model = _get_model_from_env(os.getenv("LLM_PROVIDER"))

        logger.info("Initializing Orchestrator Agent with provider=%s model=%s", provider, model)
        llm = get_llm(provider=provider, model=model)
        orchestrator = Orchestrator(llm=llm)
        _orchestrator_app = orchestrator.build_graph()

    return _orchestrator_app


def _model_to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _final_preview(state: dict[str, Any], max_rows: int) -> dict[str, Any] | None:
    slices = state.get("slices") or {}
    plan = state.get("plan") or []

    final_step_id = plan[-1].id if plan and hasattr(plan[-1], "id") else None
    if final_step_id is None and slices:
        final_step_id = next(reversed(slices))

    if not final_step_id or final_step_id not in slices:
        return None

    df = slices[final_step_id]
    return {
        "step_id": final_step_id,
        "row_count": len(df),
        "columns": [str(column) for column in df.columns],
        "rows": df.head(max_rows).to_dict(orient="records"),
    }


def _serialize_result(
    state: dict[str, Any],
    request: DataAnalystInput,
) -> DataAnalystOutput:
    result: dict[str, Any] = {
        "status": state.get("status", "failed"),
        "answer": state.get("answer", ""),
        "reasoning": state.get("reasoning", ""),
    }

    if request.include_plan:
        result["plan"] = [_model_to_dict(step) for step in state.get("plan", [])]
        result["step_results"] = [
            _model_to_dict(step_result)
            for step_result in state.get("step_results", [])
        ]

    if request.include_preview:
        result["final_preview"] = _final_preview(state, request.max_preview_rows)

    return DataAnalystOutput.model_validate(result)


def run_data_analyst_tool(arguments: dict[str, Any]) -> DataAnalystOutput:
    """Handlerul tool-ului MCP: valideaza inputul si apeleaza agentul."""
    validate_arguments_shape(arguments, allowed_fields=DATA_ANALYST_ALLOWED_FIELDS)
    if "question" in arguments:
        arguments = {
            **arguments,
            "question": validate_safe_text(
                arguments["question"],
                field_name="question",
                max_chars=MAX_USER_TEXT_CHARS,
            ),
        }

    request = DataAnalystInput.model_validate(arguments)

    logger.info("Running %s for question=%r", DATA_ANALYST_TOOL_NAME, request.question)
    analyst = get_analyst()
    state = analyst.chat(request.question)

    return _serialize_result(state=state, request=request)


def _rag_context_summary(state: dict[str, Any]) -> RAGContextSummary | None:
    rag_result = state.get("rag_result")
    if not rag_result:
        return None

    results = rag_result.results or []
    sources = [
        {
            "file_name": result.file_name,
            "summary": result.summary,
            "score": result.score,
        }
        for result in results
    ]

    return RAGContextSummary(
        query_used=rag_result.query_used,
        result_count=len(results),
        max_score=rag_result.max_score,
        avg_score=rag_result.avg_score,
        sources=sources,
    )


def _serialize_orchestrator_result(
    state: dict[str, Any],
    request: OrchestratorInput,
) -> OrchestratorOutput:
    return OrchestratorOutput(
        status=state.get("status", "failed"),
        answer=state.get("answer", ""),
        iteration=state.get("iteration", 0),
        feedback=_model_to_dict(state.get("feedback")) if state.get("feedback") else None,
        rag_context=_rag_context_summary(state) if request.include_rag_context else None,
    )


def run_orchestrator_tool(arguments: dict[str, Any]) -> OrchestratorOutput:
    """Handlerul tool-ului MCP: valideaza inputul si apeleaza Orchestrator."""
    validate_arguments_shape(arguments, allowed_fields=ORCHESTRATOR_ALLOWED_FIELDS)
    if "query" in arguments:
        arguments = {
            **arguments,
            "query": validate_safe_text(
                arguments["query"],
                field_name="query",
                max_chars=MAX_USER_TEXT_CHARS,
            ),
        }

    request = OrchestratorInput.model_validate(arguments)

    logger.info("Running %s for query=%r", ORCHESTRATOR_TOOL_NAME, request.query)
    app = get_orchestrator_app()
    state = app.invoke(OrchestratorState(query=request.query))

    return _serialize_orchestrator_result(state=state, request=request)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    data_analyst_description = (
        "Ruleaza Data Analyst Agent pentru intrebari analitice peste tabelele "
        "achizitii_directe si anunturi_initiere. Agentul creeaza un plan, "
        "apeleaza NL2SQL pentru query-uri si foloseste tool-uri locale pentru "
        "join/filter. Output-ul este JSON text conform DATA_ANALYST_OUTPUT_SCHEMA."
    )
    orchestrator_description = (
        "Ruleaza Orchestrator Agent, supervisor peste RAG Agent. Foloseste cautare "
        "RAG in documente, evalueaza daca informatia gasita este suficienta si "
        "genereaza raspunsul final. Output-ul este JSON text conform "
        "ORCHESTRATOR_OUTPUT_SCHEMA."
    )

    return [
        types.Tool(
            name=DATA_ANALYST_TOOL_NAME,
            description=f"{data_analyst_description}\n\nOutput schema: {json.dumps(DATA_ANALYST_OUTPUT_SCHEMA)}",
            inputSchema=DATA_ANALYST_INPUT_SCHEMA,
        ),
        types.Tool(
            name=ORCHESTRATOR_TOOL_NAME,
            description=f"{orchestrator_description}\n\nOutput schema: {json.dumps(ORCHESTRATOR_OUTPUT_SCHEMA)}",
            inputSchema=ORCHESTRATOR_INPUT_SCHEMA,
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    if name == DATA_ANALYST_TOOL_NAME:
        payload = run_data_analyst_tool(arguments)
    elif name == ORCHESTRATOR_TOOL_NAME:
        payload = run_orchestrator_tool(arguments)
    else:
        available = [DATA_ANALYST_TOOL_NAME, ORCHESTRATOR_TOOL_NAME]
        raise ValueError(f"Tool necunoscut: {name}. Tools disponibile: {available}")

    return [
        types.TextContent(
            type="text",
            text=json.dumps(payload.model_dump(), ensure_ascii=False, indent=2, default=str),
        )
    ]


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(
            read,
            write,
            InitializationOptions(
                server_name=SERVER_NAME,
                server_version=SERVER_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
