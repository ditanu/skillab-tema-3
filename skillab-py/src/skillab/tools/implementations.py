"""
Tool implementations — funcțiile efective ale tool-urilor.

Convenție: toate tools primesc params cu `input_dfs` (lista de DataFrames) + parametri specifici.
"""
import pandas as pd

from .registry import register_tool
from .params import JoinDataParams, FilterDataParams


@register_tool
def join_data(params: JoinDataParams) -> pd.DataFrame:
    """
    Combină două DataFrames pe baza unei chei comune (join).
    Suportă inner, left, right, outer join.

    Args:
        params.input_dfs: [left_df, right_df]
        params.left_key: coloana cheie din primul DataFrame
        params.right_key: coloana cheie din al doilea DataFrame
        params.how: tipul de join

    Returns:
        DataFrame rezultat după join
    """
    if len(params.input_dfs) != 2:
        raise ValueError("join_data requires exactly two input DataFrames")

    left_df, right_df = params.input_dfs
    if params.left_key not in left_df.columns:
        raise KeyError(f"Column '{params.left_key}' not found in left DataFrame")
    if params.right_key not in right_df.columns:
        raise KeyError(f"Column '{params.right_key}' not found in right DataFrame")

    return pd.merge(
        left_df,
        right_df,
        left_on=params.left_key,
        right_on=params.right_key,
        how=params.how,
    )


@register_tool
def filter_data(params: FilterDataParams) -> pd.DataFrame:
    """
    Filtrează un DataFrame pe baza unei condiții.
    Suportă operatori: ==, !=, >, <, >=, <=, contains.

    Args:
        params.input_dfs: [df]
        params.column: coloana pe care se aplică filtrul
        params.operator: operatorul de comparație
        params.value: valoarea pentru comparație

    Returns:
        DataFrame filtrat
    """
    if len(params.input_dfs) != 1:
        raise ValueError("filter_data requires exactly one input DataFrame")

    df = params.input_dfs[0]
    if params.column not in df.columns:
        raise KeyError(f"Column '{params.column}' not found in DataFrame")

    col = df[params.column]
    value = params.value

    if params.operator == "contains":
        mask = col.astype(str).str.contains(str(value), case=False, na=False)
    elif params.operator in {">", "<", ">=", "<="}:
        left = pd.to_numeric(col, errors="coerce")
        right = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(right):
            raise ValueError(f"Value '{value}' cannot be converted to a number")

        if params.operator == ">":
            mask = left > right
        elif params.operator == "<":
            mask = left < right
        elif params.operator == ">=":
            mask = left >= right
        else:
            mask = left <= right
    elif params.operator == "==":
        mask = col.astype(str) == str(value)
    elif params.operator == "!=":
        mask = col.astype(str) != str(value)
    else:
        raise ValueError(f"Unsupported operator: {params.operator}")

    return df[mask].copy()
