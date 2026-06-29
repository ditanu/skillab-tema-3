"""
Guardrails pentru serverul MCP.

Layer 1, in stilul lectiei 9:
  - validare forma payload: dict, dimensiune maxima, campuri permise
  - normalizare Unicode si eliminare zero-width chars
  - detectie prompt injection prin regex/keywords, fail-fast

Acest modul nu apeleaza LLM-ul. Scopul este sa blocheze inputul periculos
inainte ca agentii sa fie initializati sau sa primeasca textul in prompt.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


MAX_ARGUMENTS_BYTES = 16_384
MAX_TEXT_CHARS = 2_000

ZERO_WIDTH_CHARS = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # byte-order mark / zero-width no-break
}

PROMPT_INJECTION_PATTERNS = [
    # Override instructions - English
    r"(?i)\bignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?\b",
    r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above|earlier)\b",
    r"(?i)\bforget\s+(everything|all|what)\b",
    r"(?i)\boverride\s+(the\s+)?(system|developer|previous)\s+(prompt|instructions?)\b",
    r"(?i)\bdo\s+not\s+follow\s+(the\s+)?(system|developer|previous)\s+(prompt|instructions?)\b",
    # Override instructions - Romanian
    r"(?i)\bignora\s+(toate\s+)?(instructiunile|instructiuni|regulile)\b",
    r"(?i)\bignor[aă]\s+(toate\s+)?(instruc[tț]iunile|regulile)\b",
    r"(?i)\bnu\s+(mai\s+)?respecta\s+(instructiunile|instruc[tț]iunile|regulile)\b",
    r"(?i)\buit[aă]\s+(tot|toate|ce)\b",
    # Jailbreak keywords
    r"(?i)\byou\s+are\s+now\s+(DAN|evil|unrestricted|developer)\b",
    r"(?i)\bdeveloper\s+mode\b",
    r"(?i)\bjailbreak\b",
    r"(?i)\bDAN\s*[-:]?\s*do\s+anything\s+now\b",
    # System prompt / secret extraction
    r"(?i)\bwhat\s+(is|are)\s+your\s+(system\s+)?prompt\b",
    r"(?i)\brepeat\s+(your\s+)?(system\s+)?prompt\b",
    r"(?i)\bshow\s+(me\s+)?(your\s+)?(system|developer)\s+(prompt|instructions?)\b",
    r"(?i)\b(leak|exfiltrate|dump)\s+(the\s+)?(system\s+)?prompt\b",
    r"(?i)\b(arata|arat[aă]|afi[sș]eaz[aă]|spune)\s+.*\b(system prompt|promptul de sistem)\b",
    # Role / message boundary injection
    r"(?i)\b(system|developer|assistant)\s*:\s*(ignore|override|leak|dump|reveal|exfiltrate)\b",
    r"(?i)\b(SYSTEM|DEVELOPER|ASSISTANT)\s*:\b",
    # Tool / data exfiltration instructions
    r"(?i)\b(send|email|post|upload)\s+.*\b(secret|token|api[_ -]?key|password|system prompt)\b",
    r"(?i)\btrimite\s+.*\b(secret|token|parola|cheie api|promptul de sistem)\b",
]


@dataclass
class GuardrailResult:
    passed: bool
    method: str = "none"
    details: dict[str, Any] = field(default_factory=dict)


class GuardrailViolation(ValueError):
    """Eroare ridicata cand inputul MCP pica un guardrail."""

    def __init__(self, message: str, result: GuardrailResult):
        super().__init__(message)
        self.result = result


def normalize_text(text: str) -> str:
    """Normalizeaza Unicode si elimina caractere invizibile folosite in bypass."""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in normalized if ch not in ZERO_WIDTH_CHARS)


def validate_arguments_shape(
    arguments: Any,
    *,
    allowed_fields: set[str],
    max_bytes: int = MAX_ARGUMENTS_BYTES,
) -> None:
    """Valideaza payload-ul brut inainte de Pydantic si de agent."""
    if not isinstance(arguments, dict):
        raise GuardrailViolation(
            "Argumentele tool-ului trebuie sa fie un obiect JSON.",
            GuardrailResult(False, "type", {"expected": "object"}),
        )

    try:
        size = len(json.dumps(arguments, ensure_ascii=False, default=str).encode("utf-8"))
    except TypeError as exc:
        raise GuardrailViolation(
            "Argumentele tool-ului trebuie sa fie serializabile JSON.",
            GuardrailResult(False, "serialization", {"error": str(exc)}),
        ) from exc

    if size > max_bytes:
        raise GuardrailViolation(
            f"Payload prea mare: {size} bytes. Limita este {max_bytes} bytes.",
            GuardrailResult(False, "size", {"size": size, "max_bytes": max_bytes}),
        )

    unknown_fields = sorted(set(arguments) - allowed_fields)
    if unknown_fields:
        raise GuardrailViolation(
            f"Campuri nepermise in input: {unknown_fields}.",
            GuardrailResult(False, "allowed_fields", {"unknown_fields": unknown_fields}),
        )


def validate_safe_text(
    value: Any,
    *,
    field_name: str,
    max_chars: int = MAX_TEXT_CHARS,
) -> str:
    """
    Valideaza textul userului si returneaza varianta normalizata.

    Fail-fast:
      1. tip + gol + dimensiune
      2. caractere de control / zero-width normalizate
      3. regex prompt injection
    """
    if not isinstance(value, str):
        raise GuardrailViolation(
            f"Campul '{field_name}' trebuie sa fie string.",
            GuardrailResult(False, "type", {"field": field_name, "expected": "string"}),
        )

    normalized = normalize_text(value).strip()
    if not normalized:
        raise GuardrailViolation(
            f"Campul '{field_name}' este obligatoriu si nu poate fi gol.",
            GuardrailResult(False, "empty", {"field": field_name}),
        )

    if len(normalized) > max_chars:
        raise GuardrailViolation(
            f"Campul '{field_name}' este prea lung: {len(normalized)} caractere. Limita este {max_chars}.",
            GuardrailResult(
                False,
                "size",
                {"field": field_name, "length": len(normalized), "max_chars": max_chars},
            ),
        )

    control_chars = [
        {"index": index, "codepoint": f"U+{ord(ch):04X}"}
        for index, ch in enumerate(normalized)
        if unicodedata.category(ch) in {"Cc", "Cf"} and ch not in {"\n", "\r", "\t"}
    ]
    if control_chars:
        raise GuardrailViolation(
            f"Campul '{field_name}' contine caractere de control nepermise.",
            GuardrailResult(False, "control_chars", {"field": field_name, "chars": control_chars[:5]}),
        )

    matched_patterns = [
        pattern
        for pattern in PROMPT_INJECTION_PATTERNS
        if re.search(pattern, normalized)
    ]
    if matched_patterns:
        raise GuardrailViolation(
            f"Prompt injection detectat in campul '{field_name}'.",
            GuardrailResult(
                False,
                "prompt_injection_regex",
                {"field": field_name, "patterns": matched_patterns},
            ),
        )

    return normalized
