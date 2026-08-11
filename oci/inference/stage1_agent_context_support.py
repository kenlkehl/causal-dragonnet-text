"""Compatibility boundary for mature Stage 1 agent-context formatting.

Only the optional proposal-context path needs these formatting functions.
Keeping their imports inside the calls prevents targeted non-agentic runs from
initializing the retired all-in-one runner.
"""

from __future__ import annotations

from typing import Any


def _build_evidence_digest_agent_context(**kwargs: Any) -> dict[str, Any]:
    from .multi_model_agentic_forest import _build_evidence_digest_agent_context as implementation

    return implementation(**kwargs)


def _compact_multi_model_agent_context(context: dict[str, Any]) -> dict[str, Any]:
    from .multi_model_agentic_forest import _compact_multi_model_agent_context as implementation

    return implementation(context)


__all__ = [
    "_build_evidence_digest_agent_context",
    "_compact_multi_model_agent_context",
]
