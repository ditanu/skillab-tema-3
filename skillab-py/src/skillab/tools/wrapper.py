"""
ToolWrapper — pattern din slide-ul 59.

Oferă metode statice pentru:
- Execuție tools
- Generare catalog
- Conversie la format LangChain
- Enable/disable tools
"""
from typing import TYPE_CHECKING

from .registry import TOOL_REGISTRY
from .schemas import ToolInfo, ToolParameter, ToolWithStatus

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool


class ToolWrapper:
    """Wrapper pentru execuție și generare catalog."""

    @staticmethod
    def call(name: str, args: dict):
        """
        Execută un tool din registry.

        Returns:
            Rezultatul tool-ului (tipul depinde de tool).
            Raise exception în caz de eroare.
        """
        # 1. Lookup în registry
        if name not in TOOL_REGISTRY:
            raise KeyError(f"Tool '{name}' nu există.")

        tool = TOOL_REGISTRY[name]

        # 1.5 Check dacă e enabled
        if not tool["enabled"]:
            raise PermissionError(f"Tool '{name}' este dezactivat.")

        # 2. Validate — Pydantic verifică tipuri și constrângeri
        params = tool["params_model"](**args)

        # 3. Execute + 4. Return
        return tool["func"](params)

    @staticmethod
    def catalog() -> list[ToolInfo]:
        """Generează catalogul de tools ENABLED pentru frontend."""
        return [
            ToolWrapper._tool_to_info(name, tool)
            for name, tool in TOOL_REGISTRY.items()
            if tool["enabled"]
        ]

    @staticmethod
    def catalog_with_status() -> list[ToolWithStatus]:
        """Generează catalogul COMPLET cu status enabled/disabled."""
        return [
            ToolWithStatus(
                **ToolWrapper._tool_to_info(name, tool).model_dump(),
                enabled=tool["enabled"]
            )
            for name, tool in TOOL_REGISTRY.items()
        ]

    @staticmethod
    def _tool_to_info(name: str, tool: dict) -> ToolInfo:
        """
        Convertește un tool din registry în ToolInfo.

        Notă: Sare peste `input_dfs` deoarece este pasat programatic,
        nu de LLM. Doar parametrii "user-facing" sunt incluși.
        """
        params_model = tool["params_model"]
        parameters = []

        # Iterăm prin câmpurile modelului Pydantic
        for field_name, field_info in params_model.model_fields.items():
            # Skip input_dfs - e pasat programatic de Analyst
            if field_name == "input_dfs":
                continue

            # Determină tipul
            annotation = field_info.annotation
            if hasattr(annotation, "__origin__"):
                # Generic type like Literal["inner", "left"]
                type_str = str(annotation)
            else:
                type_str = getattr(annotation, "__name__", "string")

            parameters.append(
                ToolParameter(
                    name=field_name,
                    type=type_str,
                    description=field_info.description or "",
                    required=field_info.is_required(),
                )
            )

        return ToolInfo(
            name=name,
            description=tool["description"],
            parameters=parameters,
        )

    @staticmethod
    def to_langchain_tools() -> list["StructuredTool"]:
        """
        Convertește tools ENABLED la format LangChain pentru bind_tools().

        Requires: pip install skillab[tools] (langchain-core)
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise ImportError(
                "langchain-core nu este instalat. "
                "Rulează: pip install skillab[tools]"
            )

        lc_tools = []
        for name, tool in TOOL_REGISTRY.items():
            # Skip disabled tools
            if not tool["enabled"]:
                continue

            params_model = tool["params_model"]
            func = tool["func"]

            def make_wrapper(f, pm):
                def wrapper(**kwargs) -> str:
                    params = pm(**kwargs)
                    return str(f(params))
                return wrapper

            lc_tool = StructuredTool.from_function(
                func=make_wrapper(func, params_model),
                name=name,
                description=tool["description"],
                args_schema=params_model,
            )
            lc_tools.append(lc_tool)
        return lc_tools

    @staticmethod
    def set_enabled(name: str, enabled: bool) -> bool:
        """Activează/dezactivează un tool. Returnează True dacă a reușit."""
        if name not in TOOL_REGISTRY:
            return False
        TOOL_REGISTRY[name]["enabled"] = enabled
        return True

    @staticmethod
    def is_enabled(name: str) -> bool:
        """Verifică dacă un tool este activ."""
        if name not in TOOL_REGISTRY:
            return False
        return TOOL_REGISTRY[name]["enabled"]

    @staticmethod
    def list_all() -> list[str]:
        """Returnează lista cu numele tuturor tools."""
        return list(TOOL_REGISTRY.keys())

    @staticmethod
    def list_enabled() -> list[str]:
        """Returnează lista cu numele tools-urilor active."""
        return [
            name for name, tool in TOOL_REGISTRY.items()
            if tool["enabled"]
        ]

    @staticmethod
    def to_prompt_string(header: str = "Tools disponibile:") -> str:
        """
        Generează string cu tools active pentru inserare în system prompt.

        Args:
            header: Text la început (default: "Tools disponibile:")

        Returns:
            String formatat cu lista de tools și descrieri.
            Returnează mesaj dacă nu sunt tools active.

        Example output:
            Tools disponibile:
            - calculator: Evaluează expresii matematice
            - weather: Returnează vremea pentru un oraș
        """
        tools = ToolWrapper.catalog()
        if not tools:
            return "Nu ai tools disponibile momentan."

        lines = [header]
        for tool in tools:
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)
