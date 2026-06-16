"""
Pydantic State Models pentru agenți
"""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# RAG AGENT
# ============================================================

class SearchResultItem(BaseModel):
    """Un chunk găsit."""
    content: str
    summary: str
    file_name: str
    score: float


class RAGSearchResult(BaseModel):
    """Rezultatul căutării RAG."""
    query_used: str
    results: list[SearchResultItem] = Field(default_factory=list)
    max_score: float = 0.0
    avg_score: float = 0.0


class RefinedQuery(BaseModel):
    """Rezultatul rafinării query-ului."""
    query: str
    threshold: float | None = None  # None = păstrează curent


class OrchestratorFeedback(BaseModel):
    """Feedback de la Orchestrator către RAG."""
    can_answer: bool
    missing_info: str = ""
    wrong_context: str = ""
    suggestion: str = ""


class OrchestratorState(BaseModel):
    """State pentru Orchestrator."""
    query: str                                      # întrebarea originală
    rag_result: RAGSearchResult | None = None       # chunks de la RAG Agent
    feedback: OrchestratorFeedback | None = None    # evaluarea orchestratorului
    iteration: int = 0
    answer: str = ""
    status: Literal["pending", "success", "partial", "failed"] = "pending"

    class Config:
        extra = "allow"


class RAGAgentState(BaseModel):
    """State pentru RAG Agent (graf separat)."""
    query: str                                        # întrebarea originală
    feedback: OrchestratorFeedback | None = None      # feedback de la orchestrator
    refined: RefinedQuery | None = None               # query rafinat (după node_refine)
    result: RAGSearchResult | None = None             # chunks găsite (după node_search)

    @property
    def current_query(self) -> str:
        """Query-ul curent (rafinat sau original)."""
        return self.refined.query if self.refined else self.query

    @property
    def current_threshold(self) -> float | None:
        """Threshold-ul curent (din refined sau None)."""
        return self.refined.threshold if self.refined else None


# ============================================================
# NL2SQL AGENT
# ============================================================

class NL2SQLState(BaseModel):
    """State pentru NL2SQL agent."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    table_name: str = ""
    schema_context: dict = Field(default_factory=dict)
    sql_query: str = ""
    is_valid: bool = False
    validation_error: str = ""
    result: pd.DataFrame = Field(default_factory=pd.DataFrame)
    execution_error: str = ""
    retry_count: int = 0
    max_retries: int = 2
    status: Literal["pending", "success", "failed"] = "pending"


# ============================================================
# ANALYST AGENT
# ============================================================

class QueryStep(BaseModel):
    """Query pe un tabel via NL2SQL."""
    id: str  # ex: "q1", "achizitii_top10"
    action: Literal["query"] = "query"
    table: str
    sub_question: str


class ToolStep(BaseModel):
    """Apel tool din registry (join_data, filter_data, etc.)."""
    id: str  # ex: "joined", "filtered_result"
    action: Literal["tool"] = "tool"
    tool_name: str
    input_steps: list[str] = Field(default_factory=list)  # referințe la ID-uri
    params: dict = Field(default_factory=dict)


# Union type pentru plan steps
PlanStep = QueryStep | ToolStep


class StepResult(BaseModel):
    """Rezultatul unui pas (pentru logging)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_id: str
    action: str
    description: str
    status: Literal["success", "failed"]
    row_count: int = 0
    error: str = ""


class AnalystState(BaseModel):
    """State pentru Analyst agent."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    reasoning: str = ""
    plan: list[QueryStep | ToolStep] = Field(default_factory=list)
    current_step: int = 0
    slices: dict[str, pd.DataFrame] = Field(default_factory=dict)  # step_id -> DataFrame
    step_results: list[StepResult] = Field(default_factory=list)   # pentru logging
    answer: str = ""
    status: Literal["pending", "success", "failed", "no_plan"] = "pending"
