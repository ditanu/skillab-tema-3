"""
Pydantic models pentru parametrii tool-urilor.

Convenție: toate tools primesc `input_dfs` (lista de DataFrames) + parametri specifici.
"""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class JoinDataParams(BaseModel):
    """Parametri pentru tool-ul join_data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_dfs: list[pd.DataFrame] = Field(
        description="Lista de DataFrames: [left_df, right_df]"
    )
    left_key: str = Field(
        description="Coloana cheie din primul DataFrame"
    )
    right_key: str = Field(
        description="Coloana cheie din al doilea DataFrame"
    )
    how: Literal["inner", "left", "right", "outer"] = Field(
        default="inner",
        description="Tipul de join: 'inner', 'left', 'right', 'outer'"
    )


class FilterDataParams(BaseModel):
    """Parametri pentru tool-ul filter_data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_dfs: list[pd.DataFrame] = Field(
        description="Lista de DataFrames: [df]"
    )
    column: str = Field(
        description="Coloana pe care se aplică filtrul"
    )
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "contains"] = Field(
        description="Operatorul de comparație"
    )
    value: str = Field(
        description="Valoarea pentru comparație"
    )
