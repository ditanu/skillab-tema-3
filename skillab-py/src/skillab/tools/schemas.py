"""
Schemas for tool catalog and API responses.
"""
from pydantic import BaseModel


class ToolParameter(BaseModel):
    """Tool parameter schema."""
    name: str
    type: str
    description: str
    required: bool = True


class ToolInfo(BaseModel):
    """Tool information for catalog."""
    name: str
    description: str
    parameters: list[ToolParameter]


class ToolWithStatus(ToolInfo):
    """Tool information with enabled status."""
    enabled: bool = True
