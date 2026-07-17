"""Cross-fit-stable schemas for context-fitted upstream predictions.

Several upstream discovery models intentionally learn a different collection
of topics, clusters, queries, or orphan n-grams in every fit.  Their raw column
identities therefore cannot be aligned across the meta-inner fits consumed by
``FinalContextFitUpstreamProducer``.  This module provides a narrow backend
wrapper that makes the alignment rule explicit and precommitted.

Calibrated treatment-effect sources are never inferred from raw features.  An
exact, configured ``(name, kind)`` pair is the only calibrated source that may
pass through.  Raw features are grouped by their configured
``(source_kind, consumer_role)`` and retain that consumer role after being
reduced to permutation-invariant row summaries.  Discovered feature names are
never used in the output schema.

The wrapper's backend API accepts observable labels only for the context fit.
It has no gate-treatment or gate-outcome argument, and it forwards only gate
row IDs and text to its child backend.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .context_fit_upstream_gate_provider import (
    ContextFitUpstreamBackend,
    ContextFitUpstreamPrediction,
)

STABLE_CONTEXT_FIT_UPSTREAM_BACKEND_ID = "stable_context_fit_upstream_backend_v2"

_FORBIDDEN = ("true", "oracle", "ground_truth")
_ROLES = frozenset(
    {
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    }
)
_MAX_SIGNED_ORDER_WIDTH = 256
_NAMESPACE_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")


def _safe_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result or any(token in result.lower() for token in _FORBIDDEN):
        raise ValueError(f"{name} is empty or contains forbidden benchmark metadata")
    return result


def _closed_json(value: Any, *, path: str) -> Any:
    """Return closed identity metadata with benchmark truth channels rejected."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _safe_string(str(raw_key), name=f"{path} key")
            if key in result:
                raise ValueError(f"{path} contains colliding identity fields")
            result[key] = _closed_json(raw_value, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_closed_json(item, path=f"{path}[]") for item in value]
    if isinstance(value, np.generic):
        return _closed_json(value.item(), path=path)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _safe_string(value, name=path)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains non-finite identity metadata")
        return value
    raise TypeError(f"{path} must contain closed JSON-compatible metadata")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _positive_width(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("signed_order_width must be an integer")
    result = int(value)
    if result < 1 or result > _MAX_SIGNED_ORDER_WIDTH:
        raise ValueError(f"signed_order_width must be between 1 and {_MAX_SIGNED_ORDER_WIDTH}")
    return result


@dataclass(frozen=True)
class PrecommittedCalibratedSource:
    """One exact child source that is permitted to remain calibrated."""

    child_name: str
    source_kind: str
    output_name: str | None = None

    def __post_init__(self) -> None:
        child_name = _safe_string(self.child_name, name="calibrated source child_name")
        source_kind = _safe_string(self.source_kind, name="calibrated source source_kind")
        output_name = (
            child_name
            if self.output_name is None
            else _safe_string(self.output_name, name="calibrated source output_name")
        )
        object.__setattr__(self, "child_name", child_name)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "output_name", output_name)

    def identity(self) -> Mapping[str, Any]:
        return {
            "child_name": self.child_name,
            "source_kind": self.source_kind,
            "output_name": self.output_name,
            "exact_name_and_kind_required": True,
        }


@dataclass(frozen=True)
class PrecommittedRawFeatureFamily:
    """One raw family and its fixed-width permutation-invariant reduction."""

    source_kind: str
    consumer_role: str
    signed_order_width: int
    required: bool = True
    exact_passthrough_feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        source_kind = _safe_string(self.source_kind, name="raw family source_kind")
        consumer_role = _safe_string(self.consumer_role, name="raw family consumer_role")
        if consumer_role not in _ROLES:
            raise ValueError("raw family consumer_role is unsupported")
        if not isinstance(self.required, bool):
            raise TypeError("raw family required must be boolean")
        raw_passthrough_names = self.exact_passthrough_feature_names
        if isinstance(raw_passthrough_names, (str, bytes, Mapping)):
            raise TypeError("exact_passthrough_feature_names must be a sequence of names")
        passthrough_names = tuple(
            _safe_string(value, name="exact passthrough feature name")
            for value in raw_passthrough_names
        )
        width = _positive_width(self.signed_order_width)
        if passthrough_names:
            if not self.required:
                raise ValueError("exact preaggregated passthrough families must be required")
            if len(passthrough_names) != width + 2:
                raise ValueError(
                    "exact passthrough feature names must contain signed mean, absolute "
                    "max, and exactly signed_order_width order statistics"
                )
            if len(passthrough_names) != len(set(passthrough_names)):
                raise ValueError("exact passthrough feature names must be unique")
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "consumer_role", consumer_role)
        object.__setattr__(self, "signed_order_width", width)
        object.__setattr__(self, "exact_passthrough_feature_names", passthrough_names)

    @property
    def key(self) -> tuple[str, str]:
        return self.source_kind, self.consumer_role

    def identity(self) -> Mapping[str, Any]:
        if self.exact_passthrough_feature_names:
            return {
                "source_kind": self.source_kind,
                "consumer_role": self.consumer_role,
                "signed_order_width": self.signed_order_width,
                "required": self.required,
                "reduction": "exact_preaggregated_passthrough",
                "exact_passthrough_feature_names": list(self.exact_passthrough_feature_names),
            }
        return {
            "source_kind": self.source_kind,
            "consumer_role": self.consumer_role,
            "signed_order_width": self.signed_order_width,
            "required": self.required,
            "summaries": [
                *([] if self.required else ["presence"]),
                "signed_mean",
                "absolute_max",
                "signed_descending_order",
            ],
        }


@dataclass(frozen=True)
class CrossFitStableUpstreamSchemaConfig:
    """Complete precommitment for one stable upstream output namespace."""

    namespace: str
    calibrated_sources: tuple[PrecommittedCalibratedSource, ...] = ()
    raw_families: tuple[PrecommittedRawFeatureFamily, ...] = ()
    reject_unconfigured_calibrated_sources: bool = True
    reject_unconfigured_raw_families: bool = True
    source_config_sha256: str = ""

    def __post_init__(self) -> None:
        namespace = _safe_string(self.namespace, name="schema namespace").lower()
        if _NAMESPACE_PATTERN.fullmatch(namespace) is None:
            raise ValueError("schema namespace must match [a-z][a-z0-9_]*")
        sources = tuple(self.calibrated_sources)
        families = tuple(self.raw_families)
        if not sources and not families:
            raise ValueError("stable schema must precommit at least one output")
        if not all(isinstance(item, PrecommittedCalibratedSource) for item in sources):
            raise TypeError("calibrated_sources must contain PrecommittedCalibratedSource")
        if not all(isinstance(item, PrecommittedRawFeatureFamily) for item in families):
            raise TypeError("raw_families must contain PrecommittedRawFeatureFamily")
        source_keys = tuple((item.child_name, item.source_kind) for item in sources)
        output_names = tuple(str(item.output_name) for item in sources)
        family_keys = tuple(item.key for item in families)
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("calibrated source precommitments must be unique")
        if len(output_names) != len(set(output_names)):
            raise ValueError("calibrated output names must be unique")
        if len(family_keys) != len(set(family_keys)):
            raise ValueError("raw family (source_kind, consumer_role) keys must be unique")
        for name, value in (
            (
                "reject_unconfigured_calibrated_sources",
                self.reject_unconfigured_calibrated_sources,
            ),
            ("reject_unconfigured_raw_families", self.reject_unconfigured_raw_families),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
        source_config_sha256 = str(self.source_config_sha256).strip().lower()
        if source_config_sha256 and re.fullmatch(r"[0-9a-f]{64}", source_config_sha256) is None:
            raise ValueError("source_config_sha256 must be empty or one lowercase SHA-256")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "calibrated_sources", sources)
        object.__setattr__(self, "raw_families", families)
        object.__setattr__(self, "source_config_sha256", source_config_sha256)

    def identity(self) -> Mapping[str, Any]:
        identity = {
            "namespace": self.namespace,
            "calibrated_sources": [item.identity() for item in self.calibrated_sources],
            "raw_families": [item.identity() for item in self.raw_families],
            "reject_unconfigured_calibrated_sources": self.reject_unconfigured_calibrated_sources,
            "reject_unconfigured_raw_families": self.reject_unconfigured_raw_families,
        }
        if self.source_config_sha256:
            identity["source_config_sha256"] = self.source_config_sha256
        return identity

    def raw_output_schema(self) -> tuple[tuple[str, str, str], ...]:
        """Return fixed ``(name, kind, role)`` metadata in output-column order."""

        output: list[tuple[str, str, str]] = []
        for index, family in enumerate(self.raw_families, start=1):
            if family.exact_passthrough_feature_names:
                output.extend(
                    (name, family.source_kind, family.consumer_role)
                    for name in family.exact_passthrough_feature_names
                )
                continue
            prefix = f"{self.namespace}__family_{index:03d}"
            metrics = [
                *([] if family.required else ["presence"]),
                "signed_mean",
                "absolute_max",
                *(f"signed_order_{rank:03d}" for rank in range(1, family.signed_order_width + 1)),
            ]
            output.extend(
                (f"{prefix}__{metric}", family.source_kind, family.consumer_role)
                for metric in metrics
            )
        return tuple(output)


class CrossFitStableUpstreamBackend:
    """Wrap a context-fit backend with an identity-bound rectangular schema."""

    def __init__(
        self,
        backend: ContextFitUpstreamBackend,
        *,
        config: CrossFitStableUpstreamSchemaConfig,
    ) -> None:
        if not callable(getattr(backend, "identity", None)) or not callable(
            getattr(backend, "fit_predict", None)
        ):
            raise TypeError("backend must implement identity() and fit_predict()")
        if not isinstance(config, CrossFitStableUpstreamSchemaConfig):
            raise TypeError("config must be CrossFitStableUpstreamSchemaConfig")
        self.backend = backend
        # FinalContextFitUpstreamProducer recursively authenticates objects in
        # ``backends``.  Exposing the child here binds its runtime code as well
        # as the semantic identity bound below.
        self.backends = (backend,)
        self.config = config
        self._child_identity = _closed_json(backend.identity(), path="child.identity")
        self._config_identity = _closed_json(config.identity(), path="wrapper.config")

    def _assert_stable(self) -> None:
        current = _closed_json(self.backend.identity(), path="child.identity")
        if _canonical_json(current) != _canonical_json(self._child_identity):
            raise ValueError("stable-schema child backend identity changed")
        if _closed_json(self.config.identity(), path="wrapper.config") != self._config_identity:
            raise ValueError("stable-schema wrapper config changed")

    def identity(self) -> Mapping[str, Any]:
        self._assert_stable()
        return {
            "backend": STABLE_CONTEXT_FIT_UPSTREAM_BACKEND_ID,
            "child": copy.deepcopy(self._child_identity),
            "config": copy.deepcopy(self._config_identity),
            "gate_labels_exposed_to_child": False,
            "raw_features_relabelled_as_calibrated_sources": False,
            "discovered_feature_names_used_for_alignment": False,
            "raw_family_reduction": (
                "configured_permutation_invariant_summary_or_exact_preaggregated_passthrough"
            ),
            "exact_preaggregated_features_reduced_again": False,
            "same_rectangular_schema_safe_for_gate_and_final_consumers": True,
        }

    def _stable_sources(
        self, prediction: ContextFitUpstreamPrediction
    ) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
        actual = {
            (name, kind): index
            for index, (name, kind) in enumerate(
                zip(
                    prediction.calibrated_source_names,
                    prediction.calibrated_source_kinds,
                )
            )
        }
        configured = {
            (source.child_name, source.source_kind) for source in self.config.calibrated_sources
        }
        missing = configured - set(actual)
        if missing:
            raise RuntimeError(
                "child prediction is missing precommitted calibrated sources: "
                + ", ".join(f"{name} ({kind})" for name, kind in sorted(missing))
            )
        unexpected = set(actual) - configured
        if unexpected and self.config.reject_unconfigured_calibrated_sources:
            raise RuntimeError(
                "child prediction exposed unconfigured calibrated sources: "
                + ", ".join(f"{name} ({kind})" for name, kind in sorted(unexpected))
            )
        columns = [
            prediction.calibrated_source_values[:, actual[(source.child_name, source.source_kind)]]
            for source in self.config.calibrated_sources
        ]
        values = (
            np.column_stack(columns)
            if columns
            else np.empty((len(prediction.gate_row_ids), 0), dtype=float)
        )
        return (
            tuple(str(source.output_name) for source in self.config.calibrated_sources),
            tuple(source.source_kind for source in self.config.calibrated_sources),
            values,
        )

    @staticmethod
    def _family_summary(
        values: np.ndarray | None,
        *,
        rows: int,
        family: PrecommittedRawFeatureFamily,
    ) -> np.ndarray:
        width = family.signed_order_width
        if values is None:
            # Optional structural absence is explicit and cannot be confused
            # with an observed all-zero family because the presence bit is 0.
            return np.zeros((rows, width + 3), dtype=float)
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] < 1:
            raise ValueError("raw feature family has an invalid matrix shape")
        if not np.isfinite(matrix).all():
            raise ValueError("raw feature family contains non-finite values")
        signed_descending = np.sort(matrix, axis=1, kind="stable")[:, ::-1]
        ordered = np.zeros((rows, width), dtype=float)
        copied = min(width, signed_descending.shape[1])
        ordered[:, :copied] = signed_descending[:, :copied]
        columns = [
            np.mean(matrix, axis=1),
            np.max(np.abs(matrix), axis=1),
            ordered,
        ]
        if not family.required:
            columns.insert(0, np.ones(rows, dtype=float))
        return np.column_stack(columns)

    def _stable_raw_features(
        self, prediction: ContextFitUpstreamPrediction
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], np.ndarray]:
        groups: dict[tuple[str, str], list[int]] = {}
        for index, key in enumerate(zip(prediction.feature_kinds, prediction.feature_roles)):
            groups.setdefault(key, []).append(index)
        configured = {family.key for family in self.config.raw_families}
        unexpected = set(groups) - configured
        if unexpected and self.config.reject_unconfigured_raw_families:
            raise RuntimeError(
                "child prediction exposed unconfigured raw feature families: "
                + ", ".join(f"{kind} ({role})" for kind, role in sorted(unexpected))
            )
        summaries: list[np.ndarray] = []
        for family in self.config.raw_families:
            indices = groups.get(family.key)
            if not indices and family.required:
                raise RuntimeError(
                    "child prediction is missing required raw feature family "
                    f"{family.source_kind} ({family.consumer_role})"
                )
            values = None if not indices else prediction.feature_values[:, indices]
            if family.exact_passthrough_feature_names:
                assert indices is not None
                actual_names = tuple(prediction.feature_names[index] for index in indices)
                if actual_names != family.exact_passthrough_feature_names:
                    raise RuntimeError(
                        "preaggregated raw feature family does not match its exact "
                        f"passthrough schema: {family.source_kind} ({family.consumer_role})"
                    )
                assert values is not None
                summaries.append(np.asarray(values, dtype=float).copy())
            else:
                summaries.append(
                    self._family_summary(
                        values,
                        rows=len(prediction.gate_row_ids),
                        family=family,
                    )
                )
        schema = self.config.raw_output_schema()
        output = (
            np.column_stack(summaries)
            if summaries
            else np.empty((len(prediction.gate_row_ids), 0), dtype=float)
        )
        return (
            tuple(item[0] for item in schema),
            tuple(item[1] for item in schema),
            tuple(item[2] for item in schema),
            output,
        )

    def fit_predict(
        self,
        *,
        outer_fold: int,
        context_row_ids: tuple[int, ...],
        context_texts: tuple[str, ...],
        context_treatment: np.ndarray,
        context_outcome: np.ndarray,
        gate_row_ids: tuple[int, ...],
        gate_texts: tuple[str, ...],
        work_dir: Path,
    ) -> ContextFitUpstreamPrediction:
        self._assert_stable()
        if set(context_row_ids) & set(gate_row_ids):
            raise ValueError("stable-schema context and gate rows must be disjoint")
        treatment = np.asarray(context_treatment, dtype=float)
        outcome = np.asarray(context_outcome, dtype=float)
        if treatment.ndim != 1 or outcome.ndim != 1:
            raise ValueError("context treatment and outcome must be one-dimensional")
        if len(treatment) != len(context_row_ids) or len(outcome) != len(context_row_ids):
            raise ValueError("context labels must align with context rows")
        if not np.isfinite(treatment).all() or not np.isfinite(outcome).all():
            raise ValueError("context labels must be finite")
        treatment_copy = treatment.copy()
        outcome_copy = outcome.copy()
        treatment_copy.setflags(write=False)
        outcome_copy.setflags(write=False)
        prediction = self.backend.fit_predict(
            outer_fold=outer_fold,
            context_row_ids=tuple(context_row_ids),
            context_texts=tuple(context_texts),
            context_treatment=treatment_copy,
            context_outcome=outcome_copy,
            gate_row_ids=tuple(gate_row_ids),
            gate_texts=tuple(gate_texts),
            work_dir=Path(work_dir) / "stable_schema_child",
        )
        self._assert_stable()
        if type(prediction) is not ContextFitUpstreamPrediction:
            raise TypeError("stable-schema child returned the wrong prediction type")
        if prediction.gate_row_ids != tuple(gate_row_ids):
            raise ValueError("stable-schema child changed gate row identity/order")
        source_names, source_kinds, source_values = self._stable_sources(prediction)
        feature_names, feature_kinds, feature_roles, feature_values = self._stable_raw_features(
            prediction
        )
        return ContextFitUpstreamPrediction(
            gate_row_ids=prediction.gate_row_ids,
            calibrated_source_names=source_names,
            calibrated_source_kinds=source_kinds,
            calibrated_source_values=source_values,
            feature_names=feature_names,
            feature_kinds=feature_kinds,
            feature_roles=feature_roles,
            feature_values=feature_values,
        )


__all__ = [
    "STABLE_CONTEXT_FIT_UPSTREAM_BACKEND_ID",
    "CrossFitStableUpstreamBackend",
    "CrossFitStableUpstreamSchemaConfig",
    "PrecommittedCalibratedSource",
    "PrecommittedRawFeatureFamily",
]
