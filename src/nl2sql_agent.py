"""
NL2SQL Agent - TODO: Implementează nodurile
"""
import json
import logging
import re
from pathlib import Path
from typing import Literal

import pandas as pd
import sqlparse
from langgraph.graph import StateGraph, END
from sqlalchemy import text
from skillab import get_llm
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry

from state import NL2SQLState

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class NL2SQLAgent:
    """
    Flow:
        get_context → generate_sql → validate → execute
                                        ↓
                                    handle_error
    """

    def __init__(
        self,
        table_name: str,
        schema_path: str,
        db_url: str = "postgresql://demo:demo123@localhost:5433/rag_demo",
        max_retries: int = 2,
        llm: LLMProvider | None = None,
    ):
        self.table_name = table_name
        self.db_url = db_url
        self.max_retries = max_retries
        self.llm = llm or get_llm()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))

        # Load schema
        self.schema = json.loads(Path(schema_path).read_text())

        self.graph = self._build_graph()

    # === NODES ===

    def node_get_context(self, state: NL2SQLState) -> dict:
        """COMPLET."""
        return {
            "schema_context": self.schema,
            "table_name": self.table_name,
        }

    def node_generate_sql(self, state: NL2SQLState) -> dict:
        """
        TODO: Generează SQL.

        1. Renderează prompt "nl2sql_generate"
        2. Apelează LLM
        3. Curăță răspunsul (remove ```sql blocks)
        4. Return {"sql_query": ...}
        """
        logger.info(f"[GENERATE] {state.question}")

        schema = state.schema_context or self.schema
        prompt = self.prompts.render(
            "nl2sql_generate",
            table_name=state.table_name or self.table_name,
            table_description=schema.get("description", ""),
            columns=schema.get("columns", {}),
            business_rules=schema.get("business_rules") or schema.get("rules", {}),
            question=state.question,
        )

        response = self.llm.generate_sync([{"role": "user", "content": prompt}])
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        sql_query = match.group(1) if match else response
        sql_query = sql_query.strip()

        return {"sql_query": sql_query}

    def node_validate_sql(self, state: NL2SQLState) -> dict:
        """
        TODO: Validează SQL.

        1. Check pentru SQL injection patterns
        2. Parsează cu sqlparse
        3. Verifică că e SELECT
        4. Return {"is_valid": bool, "validation_error": str}
        """
        sql = state.sql_query
        logger.info(f"[VALIDATE] {sql[:50]}...")

        if not sql or not sql.strip():
            return {"is_valid": False, "validation_error": "SQL query is empty"}

        sql_clean = sql.strip()
        dangerous_patterns = [
            r"\bINSERT\b",
            r"\bUPDATE\b",
            r"\bDELETE\b",
            r"\bDROP\b",
            r"\bALTER\b",
            r"\bTRUNCATE\b",
            r"\bCREATE\b",
            r"\bGRANT\b",
            r"\bREVOKE\b",
            r"\bCOPY\b",
            r"\bEXECUTE\b",
            r"\bCALL\b",
            r"--",
            r"/\*",
            r"\*/",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, sql_clean, re.IGNORECASE):
                return {
                    "is_valid": False,
                    "validation_error": f"SQL contains forbidden pattern: {pattern}",
                }

        statements = [stmt for stmt in sqlparse.parse(sql_clean) if str(stmt).strip()]
        if len(statements) != 1:
            return {
                "is_valid": False,
                "validation_error": "SQL must contain exactly one statement",
            }

        statement = statements[0]
        statement_type = statement.get_type().upper()
        if statement_type not in {"SELECT", "UNKNOWN"}:
            return {
                "is_valid": False,
                "validation_error": f"Only SELECT statements are allowed, got {statement_type}",
            }

        first_token = next(
            (
                token
                for token in statement.tokens
                if not token.is_whitespace and str(token).strip()
            ),
            None,
        )
        if not first_token:
            return {"is_valid": False, "validation_error": "SQL statement is empty"}

        first_value = first_token.value.upper()
        if first_value not in {"SELECT", "WITH"}:
            return {
                "is_valid": False,
                "validation_error": "SQL must start with SELECT or WITH",
            }

        return {"is_valid": True, "validation_error": ""}

    def node_execute_sql(self, state: NL2SQLState) -> dict:
        """
        TODO: Execută SQL.

        1. Folosește transaction() + session.execute(text(sql))
        2. Convertește la DataFrame: df = pd.DataFrame(result.mappings().all())
        3. Return {"result": df, "status": "success"} sau {"execution_error": ...}
        """
        logger.info("[EXECUTE]")

        try:
            from database import transaction

            with transaction() as db:
                result = db.execute(text(state.sql_query))
                df = pd.DataFrame(result.mappings().all())

            return {
                "result": df,
                "execution_error": "",
                "status": "success",
            }
        except Exception as e:
            logger.exception("[EXECUTE] SQL failed")
            return {
                "result": pd.DataFrame(),
                "execution_error": str(e),
                "status": "failed",
            }

    def node_handle_error(self, state: NL2SQLState) -> dict:
        """
        TODO: Handle error + retry.

        1. Increment retry_count
        2. Dacă >= max_retries: return {"status": "failed"}
        3. Renderează prompt "nl2sql_error"
        4. Apelează LLM pentru SQL corectat
        5. Return {"sql_query": new_sql, "retry_count": ...}
        """
        logger.info("[ERROR]")

        retry_count = state.retry_count + 1
        if retry_count >= state.max_retries:
            return {"retry_count": retry_count, "status": "failed"}

        error_message = state.execution_error or state.validation_error
        prompt = self.prompts.render(
            "nl2sql_error",
            table_name=state.table_name or self.table_name,
            question=state.question,
            failed_sql=state.sql_query,
            error_message=error_message,
            columns=state.schema_context.get("columns", {}),
        )

        response = self.llm.generate_sync([{"role": "user", "content": prompt}])
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        sql_query = match.group(1) if match else response
        sql_query = sql_query.strip()

        return {
            "sql_query": sql_query,
            "retry_count": retry_count,
            "validation_error": "",
            "execution_error": "",
            "status": "pending",
        }

    # === ROUTING ===

    def _route_after_validate(self, state: NL2SQLState) -> str:
        return "execute_sql" if state.is_valid else "handle_error"

    def _route_after_execute(self, state: NL2SQLState) -> str:
        return END if not state.execution_error else "handle_error"

    def _route_after_error(self, state: NL2SQLState) -> str:
        return "validate_sql" if state.retry_count < self.max_retries else END

    # === GRAPH ===

    def _build_graph(self):
        graph = StateGraph(NL2SQLState)

        graph.add_node("get_context", self.node_get_context)
        graph.add_node("generate_sql", self.node_generate_sql)
        graph.add_node("validate_sql", self.node_validate_sql)
        graph.add_node("execute_sql", self.node_execute_sql)
        graph.add_node("handle_error", self.node_handle_error)

        graph.set_entry_point("get_context")
        graph.add_edge("get_context", "generate_sql")
        graph.add_edge("generate_sql", "validate_sql")
        graph.add_conditional_edges("validate_sql", self._route_after_validate, ["execute_sql", "handle_error"])
        graph.add_conditional_edges("execute_sql", self._route_after_execute, [END, "handle_error"])
        graph.add_conditional_edges("handle_error", self._route_after_error, ["validate_sql", END])

        return graph.compile()

    def run(self, question: str) -> NL2SQLState:
        """Execută agentul."""
        initial = NL2SQLState(
            question=question,
            table_name=self.table_name,
            max_retries=self.max_retries,
        )
        return self.graph.invoke(initial)
