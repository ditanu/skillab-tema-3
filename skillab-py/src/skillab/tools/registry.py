"""
Tool Registry — storage și decorator pentru înregistrare tools.

Pattern din slide-ul 57: decorator cu validări stricte.
"""
import inspect
from typing import Callable
from pydantic import BaseModel


# Storage global pentru tools
TOOL_REGISTRY: dict[str, dict] = {}


def register_tool(func: Callable) -> Callable:
    """
    Decorator care înregistrează un tool în registry.

    Validări (slide 57):
    1. Un singur parametru de tip BaseModel
    2. Docstring obligatoriu (min 15 caractere) — devine description pentru LLM

    Adaugă automat: enabled=True

    Usage:
        @register_tool
        def my_tool(params: MyParams) -> str:
            '''Tool description for LLM (min 15 chars).'''
            return "result"
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Validare 1: un singur param de tip BaseModel
    if len(params) != 1:
        raise TypeError(
            f"{func.__name__}: trebuie să aibă exact un parametru de tip BaseModel"
        )

    param_type = params[0].annotation
    if not (isinstance(param_type, type) and issubclass(param_type, BaseModel)):
        raise TypeError(
            f"{func.__name__}: parametrul trebuie să fie de tip BaseModel, "
            f"nu {param_type}"
        )

    # Validare 2: docstring obligatoriu (devine description pentru LLM)
    docstring = (func.__doc__ or "").strip()
    if not docstring:
        raise ValueError(
            f"{func.__name__}: docstring obligatoriu — devine "
            f"description vizibil pentru LLM."
        )
    if len(docstring) < 15:
        raise ValueError(
            f"{func.__name__}: docstring prea scurt ({len(docstring)} "
            f"caractere). LLM-ul are nevoie de min 15 ca să decidă."
        )

    # Înregistrare în registry — cu enabled=True default
    TOOL_REGISTRY[func.__name__] = {
        "func": func,
        "params_model": param_type,
        "description": docstring,
        "enabled": True,
    }
    return func
