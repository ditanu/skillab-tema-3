"""
Orchestrator - Supervizor care coordonează RAG Agent

Flow:
    call_rag → evaluate ──┬──→ answer → END
         ↑                │
         │   can_answer   │
         │   = false      │
         └────────────────┘

TODO pentru studenți: node_evaluate, node_answer
"""
import logging
import json
import re
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, START, END
from skillab import get_llm
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry

from state import OrchestratorState, OrchestratorFeedback
from rag_agent import RAGAgent, RAGAgentConfig

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class OrchestratorConfig:
    max_iterations: int = 3
    min_score: float = 0.25


class Orchestrator:
    """
    Supervizor care coordonează RAG Agent.

    Flow:
        1. Apelează RAG Agent pentru căutare
        2. Evaluează: pot răspunde cu aceste chunks?
        3. Dacă DA → generează răspuns
        4. Dacă NU → trimite feedback la RAG Agent, repeat
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        llm: LLMProvider | None = None,
    ):
        self.config = config or OrchestratorConfig()
        self.llm = llm or get_llm()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))

        # RAG Agent - graf separat
        rag_config = RAGAgentConfig()
        rag_config.default_threshold = self.config.min_score
        self.rag = RAGAgent(self.llm, rag_config)

    # === NODES ===

    def node_call_rag(self, state: OrchestratorState) -> dict:
        """Apelează RAG Agent pentru căutare. COMPLET - nu modifica."""
        logger.info(f"[CALL_RAG] iter {state.iteration + 1}")

        # Apelează RAG Agent cu query și feedback (dacă există)
        rag_result = self.rag.run(
            query=state.query,
            feedback=state.feedback,  # None prima dată
        )

        return {
            "rag_result": rag_result.result,
            "iteration": state.iteration + 1,
        }

    def node_evaluate(self, state: OrchestratorState) -> dict:
        """
        TODO: Evaluează dacă contextul RAG e suficient.

        1. Construiește context din state.rag_result.results:
           context = "\\n\\n".join(f"[{r.file_name}]\\n{r.content}" for r in results)

        2. Renderează prompt "rag_evaluate" cu:
           - query=state.query
           - context=context
           - max_score=state.rag_result.max_score
           - avg_score=state.rag_result.avg_score

        3. Apelează LLM:
           response = self.llm.generate_sync([{"role": "user", "content": prompt}])

        4. Parsează JSON în Pydantic:
           - Extrage JSON din ```json ... ```
           - feedback = OrchestratorFeedback.model_validate_json(json_str)

        5. Return {"feedback": feedback}
        """
        logger.info(f"[EVALUATE] iter {state.iteration}")

        if not state.rag_result or not state.rag_result.results:
            return {
                "feedback": OrchestratorFeedback(
                    can_answer=False,
                    missing_info="Nu au fost găsite informații relevante.",
                    suggestion="Reformulează căutarea folosind termeni mai specifici.",
                )
            }

        context = "\n\n".join(
            f"[{result.file_name}]\n{result.content}"
            for result in state.rag_result.results
        )

        prompt = self.prompts.render(
            "rag_evaluate",
            query=state.query,
            context=context,
            max_score=state.rag_result.max_score,
            avg_score=state.rag_result.avg_score,
        )

        response = self.llm.generate_sync([{"role": "user", "content": prompt}])
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        json_str = match.group(1) if match else response.strip()
        feedback_data = json.loads(json_str)
        for field in ("missing_info", "wrong_context", "suggestion"):
            if feedback_data.get(field) is None:
                feedback_data[field] = ""
        feedback = OrchestratorFeedback.model_validate(feedback_data)

        return {"feedback": feedback}

    def node_answer(self, state: OrchestratorState) -> dict:
        """
        TODO: Generează răspunsul final.

        1. Construiește context din state.rag_result.results:
           context = "\\n\\n".join(f"[{r.file_name}]\\n{r.content}" for r in results)

        2. Renderează prompt "rag_answer" cu:
           - query=state.query
           - context=context

        3. Apelează LLM:
           answer = self.llm.generate_sync([{"role": "user", "content": prompt}])

        4. Determină status:
           - "success" dacă feedback.can_answer == True
           - "partial" dacă am răspuns dar fără can_answer
           - "failed" dacă nu avem rezultate

        5. Return {"answer": answer, "status": status}
        """
        logger.info("[ANSWER]")

        if not state.rag_result or not state.rag_result.results:
            return {
                "answer": "Nu am găsit informații relevante pentru a răspunde la întrebare.",
                "status": "failed",
            }

        context = "\n\n".join(
            f"[{result.file_name}]\n{result.content}"
            for result in state.rag_result.results
        )

        prompt = self.prompts.render(
            "rag_answer",
            query=state.query,
            context=context,
        )

        answer = self.llm.generate_sync([{"role": "user", "content": prompt}])
        status = "success" if state.feedback and state.feedback.can_answer else "partial"

        return {"answer": answer, "status": status}

    # === ROUTING ===

    def _should_continue(self, state: OrchestratorState) -> Literal["call_rag", "answer"]:
        """Decide dacă continuăm căutarea sau răspundem."""
        if state.feedback and state.feedback.can_answer:
            return "answer"
        if state.iteration >= self.config.max_iterations:
            logger.info(f"[ROUTING] Max iterations ({self.config.max_iterations}) reached")
            return "answer"
        return "call_rag"

    # === GRAPH ===

    def build_graph(self):
        """Construiește graful Orchestrator."""
        graph = StateGraph(OrchestratorState)

        graph.add_node("call_rag", self.node_call_rag)
        graph.add_node("evaluate", self.node_evaluate)
        graph.add_node("answer", self.node_answer)

        graph.add_edge(START, "call_rag")
        graph.add_edge("call_rag", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {"call_rag": "call_rag", "answer": "answer"}
        )
        graph.add_edge("answer", END)

        return graph.compile()
