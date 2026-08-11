"""Oncology Causal Inference (OCI).

Public compatibility names are loaded on first access so lightweight workflow
commands do not initialize the standalone model stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "ExperimentConfig": ("oci.config", "ExperimentConfig"),
    "AppliedInferenceConfig": ("oci.config", "AppliedInferenceConfig"),
    "ModelArchitectureConfig": ("oci.config", "ModelArchitectureConfig"),
    "TrainingConfig": ("oci.config", "TrainingConfig"),
    "create_default_config": ("oci.config", "create_default_config"),
    "ExperimentRunner": ("oci.experiments", "ExperimentRunner"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
