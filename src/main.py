"""
Main - Test agenții
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "skillab-py" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from skillab import get_llm
from state import OrchestratorState
from orchestrator import Orchestrator

# Citește config LLM din .env
LLM_PROVIDER = os.getenv("LLM_PROVIDER")

# Alias-uri pentru provideri (gemini -> google, ollama -> local)
_PROVIDER_ALIASES = {
    "gemini": "google",
    "ollama": "local",
}

# Mapping provider -> env var pentru model
_MODEL_ENV_VARS = {
    "google": "GOOGLE_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "openai": "OPENAI_MODEL",
    "local": "OLLAMA_MODEL",
}

def _resolve_provider(provider: str | None) -> str | None:
    """Rezolvă alias-uri (gemini -> google)."""
    if not provider:
        return None
    return _PROVIDER_ALIASES.get(provider.lower(), provider.lower())

def _get_model_from_env(provider: str | None) -> str | None:
    """Citește model din env var specific provider-ului."""
    resolved = _resolve_provider(provider)
    if not resolved:
        return None
    env_var = _MODEL_ENV_VARS.get(resolved, f"{resolved.upper()}_MODEL")
    return os.getenv("LLM_MODEL") or os.getenv(env_var)

# Rezolvă provider și model
LLM_PROVIDER = _resolve_provider(os.getenv("LLM_PROVIDER"))
LLM_MODEL = _get_model_from_env(os.getenv("LLM_PROVIDER"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)


def test_orchestrator():
    """Test Orchestrator + RAG."""
    print("\n" + "=" * 50)
    print("TEST: Orchestrator + RAG")
    print("=" * 50)

    # Creează LLM explicit din config
    llm = get_llm(provider=LLM_PROVIDER, model=LLM_MODEL)
    print(f"Using LLM: {LLM_PROVIDER} / {llm.model}")

    orch = Orchestrator(llm=llm)
    app = orch.build_graph()

    queries = [
        "Care e totalul facturilor TechSoft?",
        "Ce contact are DataPro?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = app.invoke(OrchestratorState(query=query))
        print(f"Status: {result['status']}")
        print(f"Answer: {result['answer'][:200]}...")


def test_analyst():
    """Test Analyst + NL2SQL."""
    print("\n" + "=" * 50)
    print("TEST: Analyst + NL2SQL")
    print("=" * 50)

    from analyst_agent import AnalystAgent

    # Creează LLM explicit din config
    llm = get_llm(provider=LLM_PROVIDER, model=LLM_MODEL)
    print(f"Using LLM: {LLM_PROVIDER} / {llm.model}")

    DATA_DIR = Path(__file__).parent.parent / "data"

    analyst = AnalystAgent(
        tables_config={
            "achizitii_directe": {
                "schema_path": str(DATA_DIR / "nl2sql_agent" / "schema_achizitii_directe.json"),
                "business_path": str(DATA_DIR / "nl2sql_agent" / "business_achizitii_directe.json"),
            },
            "anunturi_initiere": {
                "schema_path": str(DATA_DIR / "nl2sql_agent" / "schema_anunturi_initiere.json"),
                "business_path": str(DATA_DIR / "nl2sql_agent" / "business_anunturi_initiere.json"),
            },
        },
        llm=llm,
    )

    result = analyst.chat("Care sunt top 5 furnizori după valoare?")
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer'][:200]}...")


if __name__ == "__main__":
    test_orchestrator()
    # test_analyst()  # uncomment după ce ai schema JSON
