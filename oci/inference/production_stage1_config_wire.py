"""Canonical wire representation for the effective production Stage 1 config."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

from ..config import AppliedInferenceConfig, MultiModelAgenticForestConfig


_LEGACY_AGENTIC_FIELDS = frozenset(
    field.name for field in fields(MultiModelAgenticForestConfig)
)


def production_stage1_effective_config_payload(
    config: AppliedInferenceConfig,
) -> dict[str, Any]:
    """Serialize one effective config without leaking an integrated subclass.

    The production runner aliases ``MultiModelForestConfig`` into the legacy
    ``multi_model_agentic_forest`` slot so shared embedding code observes the
    same scientific settings. Dataclass serialization follows that runtime
    subclass, but the wire parser constructs the legacy slot as the narrower
    ``MultiModelAgenticForestConfig``. Projecting only that duplicate slot onto
    its declared base schema keeps shared values (including
    ``fold_parallelism``) while leaving integrated-only controls such as
    ``bow_fold_parallelism`` and ``htr_fold_parallelism`` exclusively under
    ``multi_model_forest``.
    """

    if not isinstance(config, AppliedInferenceConfig):
        raise TypeError("production Stage 1 config must be AppliedInferenceConfig")
    payload = asdict(config)
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("production Stage 1 config lacks its architecture")
    legacy_agentic = architecture.get("multi_model_agentic_forest")
    if not isinstance(legacy_agentic, dict):
        raise ValueError(
            "production Stage 1 config lacks multi_model_agentic_forest"
        )
    architecture["multi_model_agentic_forest"] = {
        key: value
        for key, value in legacy_agentic.items()
        if key in _LEGACY_AGENTIC_FIELDS
    }
    return payload


__all__ = ["production_stage1_effective_config_payload"]
