"""
RAG Agent - Graf pentru căutare în pgvector

Flow:
    START → refine → search → END

- refine: dacă are feedback, rafinează query-ul (TODO pentru studenți)
- search: caută în pgvector (COMPLET)
"""
import json
import logging
import re
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry

from state import (
    RAGAgentState,
    RAGSearchResult,
    SearchResultItem,
    RefinedQuery,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class RAGAgentConfig:
    top_k: int = 5
    default_threshold: float = 0.25


class RAGAgent:
    """
    Agent pentru căutare în pgvector.

    Flow:
        refine → search

    Dacă primește feedback de la Orchestrator, rafinează query-ul
    înainte de a căuta.
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: RAGAgentConfig | None = None,
    ):
        self.llm = llm
        self.config = config or RAGAgentConfig()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))
        self.graph = self._build_graph()

    # === NODES ===

    def node_refine(self, state: RAGAgentState) -> dict:
        """
        TODO: Rafinează query-ul dacă avem feedback.

        Dacă state.feedback este None (prima căutare):
            - Returnează refined = RefinedQuery(query=state.query)

        Dacă state.feedback există (orchestratorul a zis că nu poate răspunde):
            1. Construiește found_summary din state.result.results (dacă există)
            2. Renderează prompt "rag_refine" cu:
               - original_query, current_query, found_summary
               - max_score, avg_score, current_threshold
               - feedback (can_answer, missing_info, suggestion)
            3. Apelează LLM
            4. Parsează JSON în RefinedQuery cu model_validate_json()
            5. Return {"refined": RefinedQuery(...)}
        """
        logger.info(f"[REFINE] feedback={state.feedback is not None}")

        if state.feedback is None:
            return {"refined": RefinedQuery(query=state.query)}

        found_summary = ""
        max_score = 0.0
        avg_score = 0.0

        if state.result and state.result.results:
            found_summary = "\n".join(
                f"- [{result.file_name}] {result.summary or result.content[:200]}"
                for result in state.result.results
            )
            max_score = state.result.max_score
            avg_score = state.result.avg_score

        prompt = self.prompts.render(
            "rag_refine",
            original_query=state.query,
            current_query=state.current_query,
            found_summary=found_summary or "Nu există rezultate anterioare disponibile.",
            max_score=max_score,
            avg_score=avg_score,
            current_threshold=state.current_threshold or self.config.default_threshold,
            can_answer=state.feedback.can_answer,
            missing_info=state.feedback.missing_info,
            suggestion=state.feedback.suggestion,
        )

        response = self.llm.generate_sync([{"role": "user", "content": prompt}])
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        json_str = match.group(1) if match else response.strip()
        refined = RefinedQuery.model_validate_json(json_str)

        return {"refined": refined}

    def node_search(self, state: RAGAgentState) -> dict:
        """Caută chunks similare în pgvector. COMPLET - nu modifica."""
        from database import transaction
        from rag_service import RAGService

        query = state.current_query
        threshold = state.current_threshold or self.config.default_threshold

        logger.info(f"[SEARCH] '{query}' (top_k={self.config.top_k}, threshold={threshold})")

        with transaction() as db:
            rag = RAGService(db)
            results = rag.search(query, top_k=self.config.top_k, threshold=threshold)

            # Transformă în SearchResultItem cât timp sesiunea DB este încă activă.
            items = [
                SearchResultItem(
                    content=chunk.content,
                    summary=chunk.summary or "",
                    file_name=chunk.file_name,
                    score=score,
                )
                for chunk, score in results
            ]

        # Calculează statistici
        scores = [item.score for item in items]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "result": RAGSearchResult(
                query_used=query,
                results=items,
                max_score=max_score,
                avg_score=avg_score,
            )
        }

    # === GRAPH ===

    def _build_graph(self):
        """Construiește graful RAG Agent."""
        graph = StateGraph(RAGAgentState)

        graph.add_node("refine", self.node_refine)
        graph.add_node("search", self.node_search)

        graph.add_edge(START, "refine")
        graph.add_edge("refine", "search")
        graph.add_edge("search", END)

        return graph.compile()

    def run(self, query: str, feedback=None) -> RAGAgentState:
        """Execută agentul."""
        initial = RAGAgentState(query=query, feedback=feedback)
        result = self.graph.invoke(initial)
        return RAGAgentState.model_validate(result)
