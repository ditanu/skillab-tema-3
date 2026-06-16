"""
Tools package — reusable tool implementations for agents.

Convenție: toate tools primesc `input_dfs` (lista de DataFrames) + parametri specifici.

Usage:
    from skillab.tools import ToolWrapper

    # Get tool catalog for prompts
    catalog = ToolWrapper.to_prompt_string()

    # Execute a tool
    result_df = ToolWrapper.call("join_data", {
        "input_dfs": [df1, df2],
        "left_key": "id",
        "right_key": "user_id",
        "how": "inner"
    })
"""
from .registry import register_tool, TOOL_REGISTRY
from .wrapper import ToolWrapper
from .schemas import ToolInfo, ToolParameter
from .params import JoinDataParams, FilterDataParams

# Import implementations to auto-register tools
from . import implementations  # noqa: F401

__all__ = [
    # Core
    "register_tool",
    "TOOL_REGISTRY",
    "ToolWrapper",
    # Schemas
    "ToolInfo",
    "ToolParameter",
    # Params
    "JoinDataParams",
    "FilterDataParams",
]
