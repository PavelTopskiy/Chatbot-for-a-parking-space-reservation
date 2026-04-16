"""Guardrails: lightweight PII / secret detection for chatbot input and output.

Goals:
- Prevent user-supplied secrets from being persisted or echoed back unchanged.
- Prevent the chatbot from leaking sensitive patterns that may have crept into
  the vector DB (e.g. an internal email, an API token, a credit card).
- Catch obvious prompt-injection attempts on the input side.

This is a regex-first implementation chosen for transparency and zero extra
dependencies. It is *not* a substitute for a hardened PII service like
Microsoft Presidio in production, but is suitable for Stage 1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Pattern


# ---------- patterns ----------

# Each pattern is conservative; we prefer false positives (over-redaction)
# to false negatives (leaks).
PATTERNS: dict[str, Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    "API_KEY": re.compile(
        r"\b(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,})\b"
    ),
    "JWT": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
}

# Words that strongly suggest a secret is being shared even without a
# recognisable format.
SECRET_KEYWORDS = re.compile(
    r"\b(password|passwd|api[\s_-]?key|secret[\s_-]?key|access[\s_-]?token|private[\s_-]?key)\b",
    re.IGNORECASE,
)

# Obvious prompt-injection markers on the input side.
INJECTION_MARKERS = [
    re.compile(r"ignore (the )?(previous|above|prior) (instructions|prompt)", re.I),
    re.compile(r"disregard (all|the) (previous|prior) (instructions|rules)", re.I),
    re.compile(r"you are now (a|an) [a-z ]+", re.I),
    re.compile(r"reveal (the )?(system )?prompt", re.I),
    re.compile(r"print your (system )?(prompt|instructions)", re.I),
]


@dataclass
class GuardrailResult:
    text: str
    findings: list[str] = field(default_factory=list)
    blocked: bool = False
    reason: str | None = None

    @property
    def safe(self) -> bool:
        return not self.blocked


def _redact(text: str) -> tuple[str, list[str]]:
    findings: list[str] = []
    redacted = text
    for label, pat in PATTERNS.items():
        if pat.search(redacted):
            findings.append(label)
            redacted = pat.sub(f"[REDACTED:{label}]", redacted)
    if SECRET_KEYWORDS.search(redacted):
        findings.append("SECRET_KEYWORD")
    return redacted, findings


def sanitize_input(text: str) -> GuardrailResult:
    """Run on raw user input *before* sending to the model.

    - Block obvious prompt injection.
    - Redact PII/secrets so they never reach the LLM or get stored.
    """
    if not text or not text.strip():
        return GuardrailResult(text=text, blocked=True, reason="empty input")

    for pat in INJECTION_MARKERS:
        if pat.search(text):
            return GuardrailResult(
                text=text,
                findings=["PROMPT_INJECTION"],
                blocked=True,
                reason="possible prompt injection detected",
            )

    redacted, findings = _redact(text)
    return GuardrailResult(text=redacted, findings=findings)


def sanitize_output(text: str) -> GuardrailResult:
    """Run on model output *before* showing it to the user.

    Always redacts; never blocks (we want the user to get *something* back).
    """
    redacted, findings = _redact(text)
    return GuardrailResult(text=redacted, findings=findings)


__all__ = ["sanitize_input", "sanitize_output", "GuardrailResult"]
