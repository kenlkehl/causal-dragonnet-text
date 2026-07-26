"""Coordinate-preserving schemas for context-fitted upstream predictions.

Some context-fitted upstream columns have a stable, precommitted semantic
identity while others are fit-local topics, clusters, or queries.  Treating
both classes as fit-local discards useful coordinate alignment; treating both
classes as stable makes meta-inner matrices silently misaligned.  This module
provides an additive wrapper that partitions the two classes explicitly:

* calibrated sources are selected by an exact configured ``(name, kind)``;
* stable raw coordinates are selected by an exact configured
  ``(name, kind, consumer_role)`` and retain their values; and
* every remaining member of a configured volatile ``(kind, consumer_role)``
  family is reduced to a permutation-invariant row summary.

Every child column must be consumed exactly once.  Stable coordinates are
claimed first, so a volatile family with the same kind and role summarizes
only its remaining fit-local members.  Required inputs fail when absent.
Optional raw inputs are zero-filled behind an explicit presence column.

The backend signature exposes observable labels only for context fitting.  It
forwards no gate labels to its child and exposes the child through ``backends``
so :class:`FinalContextFitUpstreamProducer` can recursively authenticate the
runtime implementation.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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

COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID = (
    "coordinate_preserving_context_fit_upstream_backend_v3"
)

_FORBIDDEN = ("true", "oracle", "ground_truth")
_ROLES = frozenset(
    {
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    }
)
_NAMESPACE_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class VolatileRawFeatureFamilyCapacityOverflowError(RuntimeError):
    """A fixed-width family cannot represent every supplied child coordinate."""


def _safe_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result or any(token in result.lower() for token in _FORBIDDEN):
        raise ValueError(f"{name} is empty or contains forbidden benchmark metadata")
    return result


def _closed_json(value: Any, *, path: str) -> Any:
    """Return closed identity metadata with benchmark-answer channels rejected."""

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
    if result < 1:
        raise ValueError("signed_order_width must be positive")
    return result


def _required_flag(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _integer_rows(values: Sequence[Any], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of integer row IDs")
    rows: list[int] = []
    for value in tuple(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must contain canonical integer row IDs")
        row_id = int(value)
        if row_id < 0:
            raise ValueError(f"{name} cannot contain negative row IDs")
        rows.append(row_id)
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"{name} must be non-empty and unique")
    return tuple(rows)


def _exact_texts(values: Sequence[Any], *, name: str, length: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of exact strings")
    result = tuple(values)
    if len(result) != int(length) or not all(isinstance(value, str) for value in result):
        raise ValueError(f"{name} must contain exactly {length} strings")
    return result


@dataclass(frozen=True)
class PrecommittedExactCalibratedSource:
    """One calibrated child source selected by its exact name and kind."""

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

    @property
    def key(self) -> tuple[str, str]:
        return self.child_name, self.source_kind

    def identity(self) -> Mapping[str, Any]:
        return {
            "child_name": self.child_name,
            "source_kind": self.source_kind,
            "output_name": self.output_name,
            "matching": "exact_child_name_and_source_kind",
            "required": True,
        }


@dataclass(frozen=True)
class PrecommittedNamedRawCoordinate:
    """One stable raw coordinate selected by its complete child metadata."""

    child_name: str
    source_kind: str
    consumer_role: str
    output_name: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        child_name = _safe_string(self.child_name, name="named coordinate child_name")
        source_kind = _safe_string(self.source_kind, name="named coordinate source_kind")
        consumer_role = _safe_string(self.consumer_role, name="named coordinate consumer_role")
        if consumer_role not in _ROLES:
            raise ValueError("named coordinate consumer_role is unsupported")
        output_name = (
            child_name
            if self.output_name is None
            else _safe_string(self.output_name, name="named coordinate output_name")
        )
        required = _required_flag(self.required, name="named coordinate required")
        object.__setattr__(self, "child_name", child_name)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "consumer_role", consumer_role)
        object.__setattr__(self, "output_name", output_name)
        object.__setattr__(self, "required", required)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.child_name, self.source_kind, self.consumer_role

    def identity(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {
            "child_name": self.child_name,
            "source_kind": self.source_kind,
            "consumer_role": self.consumer_role,
            "output_name": self.output_name,
            "matching": "exact_child_name_source_kind_and_consumer_role",
            "required": self.required,
        }
        if not self.required:
            result["absence_encoding"] = "presence_zero_then_zero_filled_coordinate"
        return result


@dataclass(frozen=True)
class PrecommittedVolatileRawFeatureFamily:
    """One fit-local raw family reduced after named coordinates are claimed."""

    source_kind: str
    consumer_role: str
    signed_order_width: int
    required: bool = True
    child_name_pattern: str = ""

    def __post_init__(self) -> None:
        source_kind = _safe_string(self.source_kind, name="volatile family source_kind")
        consumer_role = _safe_string(self.consumer_role, name="volatile family consumer_role")
        if consumer_role not in _ROLES:
            raise ValueError("volatile family consumer_role is unsupported")
        required = _required_flag(self.required, name="volatile family required")
        width = _positive_width(self.signed_order_width)
        child_name_pattern = str(self.child_name_pattern).strip()
        if child_name_pattern:
            _safe_string(
                child_name_pattern,
                name="volatile family child_name_pattern",
            )
            try:
                re.compile(child_name_pattern)
            except re.error as exc:
                raise ValueError(
                    "volatile family child_name_pattern must be a valid regular expression"
                ) from exc
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "consumer_role", consumer_role)
        object.__setattr__(self, "signed_order_width", width)
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "child_name_pattern", child_name_pattern)

    @property
    def key(self) -> tuple[str, str]:
        return self.source_kind, self.consumer_role

    def identity(self) -> Mapping[str, Any]:
        result = {
            "source_kind": self.source_kind,
            "consumer_role": self.consumer_role,
            "signed_order_width": self.signed_order_width,
            "maximum_member_count": self.signed_order_width,
            "required": self.required,
            "membership": "remaining_columns_after_named_coordinate_claims",
            "summaries": [
                *([] if self.required else ["presence"]),
                "signed_mean",
                "absolute_max",
                "signed_descending_order",
            ],
        }
        if self.child_name_pattern:
            result["child_name_pattern"] = self.child_name_pattern
            result["child_name_matching"] = "full_regular_expression_match"
        return result


@dataclass(frozen=True)
class CoordinatePreservingUpstreamSchemaConfig:
    """Complete fixed-order precommitment for one hybrid stable schema."""

    namespace: str
    calibrated_sources: tuple[PrecommittedExactCalibratedSource, ...] = ()
    named_raw_coordinates: tuple[PrecommittedNamedRawCoordinate, ...] = ()
    volatile_raw_families: tuple[PrecommittedVolatileRawFeatureFamily, ...] = ()
    source_config_sha256: str = ""

    def __post_init__(self) -> None:
        namespace = _safe_string(self.namespace, name="schema namespace").lower()
        if _NAMESPACE_PATTERN.fullmatch(namespace) is None:
            raise ValueError("schema namespace must match [a-z][a-z0-9_]*")
        sources = tuple(self.calibrated_sources)
        coordinates = tuple(self.named_raw_coordinates)
        families = tuple(self.volatile_raw_families)
        if not sources and not coordinates and not families:
            raise ValueError("coordinate-preserving schema must precommit at least one output")
        if not all(isinstance(item, PrecommittedExactCalibratedSource) for item in sources):
            raise TypeError("calibrated_sources must contain PrecommittedExactCalibratedSource")
        if not all(isinstance(item, PrecommittedNamedRawCoordinate) for item in coordinates):
            raise TypeError("named_raw_coordinates must contain PrecommittedNamedRawCoordinate")
        if not all(isinstance(item, PrecommittedVolatileRawFeatureFamily) for item in families):
            raise TypeError(
                "volatile_raw_families must contain PrecommittedVolatileRawFeatureFamily"
            )
        source_keys = tuple(item.key for item in sources)
        source_output_names = tuple(str(item.output_name) for item in sources)
        coordinate_keys = tuple(item.key for item in coordinates)
        family_keys = tuple(item.key for item in families)
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("calibrated source precommitments must be unique")
        if len(source_output_names) != len(set(source_output_names)):
            raise ValueError("calibrated source output names must be unique")
        if len(coordinate_keys) != len(set(coordinate_keys)):
            raise ValueError("named raw coordinate precommitments must be unique")
        if len(family_keys) != len(set(family_keys)):
            raise ValueError("volatile raw family kind/role keys must be unique")
        source_config_sha256 = str(self.source_config_sha256).strip().lower()
        if source_config_sha256 and _SHA256_PATTERN.fullmatch(source_config_sha256) is None:
            raise ValueError("source_config_sha256 must be empty or one lowercase SHA-256")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "calibrated_sources", sources)
        object.__setattr__(self, "named_raw_coordinates", coordinates)
        object.__setattr__(self, "volatile_raw_families", families)
        object.__setattr__(self, "source_config_sha256", source_config_sha256)
        raw_names = tuple(item[0] for item in self.raw_output_schema())
        if len(raw_names) != len(set(raw_names)):
            raise ValueError("fixed raw output names must be globally unique")

    def identity(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {
            "namespace": self.namespace,
            "calibrated_sources": [item.identity() for item in self.calibrated_sources],
            "named_raw_coordinates": [item.identity() for item in self.named_raw_coordinates],
            "volatile_raw_families": [item.identity() for item in self.volatile_raw_families],
            "child_column_partition_order": (
                "exact_calibrated_then_named_raw_then_remaining_volatile_raw"
            ),
            "unconfigured_child_columns": "reject",
        }
        if self.source_config_sha256:
            result["source_config_sha256"] = self.source_config_sha256
        return result

    def raw_output_schema(self) -> tuple[tuple[str, str, str], ...]:
        """Return fixed ``(name, kind, role)`` metadata in output order."""

        output: list[tuple[str, str, str]] = []
        for index, coordinate in enumerate(self.named_raw_coordinates, start=1):
            if not coordinate.required:
                output.append(
                    (
                        f"{self.namespace}__named_coordinate_{index:03d}__presence",
                        coordinate.source_kind,
                        coordinate.consumer_role,
                    )
                )
            output.append(
                (
                    str(coordinate.output_name),
                    coordinate.source_kind,
                    coordinate.consumer_role,
                )
            )
        for index, family in enumerate(self.volatile_raw_families, start=1):
            prefix = f"{self.namespace}__volatile_family_{index:03d}"
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


class CoordinatePreservingContextFitUpstreamBackend:
    """Wrap a context-fit backend with a hybrid coordinate-stable schema."""

    def __init__(
        self,
        backend: ContextFitUpstreamBackend,
        *,
        config: CoordinatePreservingUpstreamSchemaConfig,
    ) -> None:
        if not callable(getattr(backend, "identity", None)) or not callable(
            getattr(backend, "fit_predict", None)
        ):
            raise TypeError("backend must implement identity() and fit_predict()")
        if not isinstance(config, CoordinatePreservingUpstreamSchemaConfig):
            raise TypeError("config must be CoordinatePreservingUpstreamSchemaConfig")
        self.backend = backend
        # FinalContextFitUpstreamProducer recursively authenticates every object
        # exposed here, including the child module and method bytecode.
        self.backends = (backend,)
        self.config = config
        self._child_identity = _closed_json(backend.identity(), path="child.identity")
        self._config_identity = _closed_json(config.identity(), path="wrapper.config")

    def _assert_stable(self) -> None:
        current_child = _closed_json(self.backend.identity(), path="child.identity")
        if _canonical_json(current_child) != _canonical_json(self._child_identity):
            raise ValueError("coordinate-preserving child backend identity changed")
        current_config = _closed_json(self.config.identity(), path="wrapper.config")
        if _canonical_json(current_config) != _canonical_json(self._config_identity):
            raise ValueError("coordinate-preserving wrapper config changed")

    def identity(self) -> Mapping[str, Any]:
        self._assert_stable()
        return {
            "backend": COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID,
            "child": copy.deepcopy(self._child_identity),
            "config": copy.deepcopy(self._config_identity),
            "gate_labels_exposed_to_child": False,
            "raw_features_relabelled_as_calibrated_sources": False,
            "named_raw_coordinate_alignment": "exact_child_name_kind_and_role",
            "volatile_raw_reduction": "permutation_invariant_after_named_claims",
            "child_column_consumption": "exactly_once",
            "fixed_output_order": True,
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
        configured = {source.key for source in self.config.calibrated_sources}
        missing = configured - set(actual)
        if missing:
            raise RuntimeError(
                "child prediction is missing exact calibrated sources: "
                + ", ".join(f"{name} ({kind})" for name, kind in sorted(missing))
            )
        unexpected = set(actual) - configured
        if unexpected:
            raise RuntimeError(
                "child prediction exposed unconfigured calibrated sources: "
                + ", ".join(f"{name} ({kind})" for name, kind in sorted(unexpected))
            )
        columns = [
            prediction.calibrated_source_values[:, actual[source.key]]
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
        family: PrecommittedVolatileRawFeatureFamily,
    ) -> np.ndarray:
        width = family.signed_order_width
        if values is None:
            # Only optional families reach this branch.  Presence 0 makes
            # absence distinct from an observed all-zero family.
            if family.required:
                raise RuntimeError("required volatile raw feature family cannot be absent")
            return np.zeros((rows, width + 3), dtype=float)
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] < 1:
            raise ValueError("volatile raw feature family has an invalid matrix shape")
        if matrix.shape[1] > width:
            raise VolatileRawFeatureFamilyCapacityOverflowError(
                "volatile raw feature family exceeds its precommitted member capacity: "
                f"{matrix.shape[1]} child columns exceed its explicit "
                f"signed_order_width={width}; refusing silent child-column omission"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("volatile raw feature family contains non-finite values")
        signed_descending = np.sort(matrix, axis=1, kind="stable")[:, ::-1]
        ordered = np.zeros((rows, width), dtype=float)
        copied = signed_descending.shape[1]
        ordered[:, :copied] = signed_descending[:, :copied]
        columns: list[np.ndarray] = [
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
        rows = len(prediction.gate_row_ids)
        keys = tuple(
            zip(
                prediction.feature_names,
                prediction.feature_kinds,
                prediction.feature_roles,
            )
        )
        actual = {key: index for index, key in enumerate(keys)}
        actual_by_name = {name: (kind, role) for name, kind, role in keys}
        consumed: set[int] = set()
        blocks: list[np.ndarray] = []

        for coordinate in self.config.named_raw_coordinates:
            index = actual.get(coordinate.key)
            if index is None:
                observed_metadata = actual_by_name.get(coordinate.child_name)
                if observed_metadata is not None:
                    raise RuntimeError(
                        "named raw coordinate metadata changed for "
                        f"{coordinate.child_name}: expected "
                        f"({coordinate.source_kind}, {coordinate.consumer_role}), observed "
                        f"({observed_metadata[0]}, {observed_metadata[1]})"
                    )
                if coordinate.required:
                    raise RuntimeError(
                        "child prediction is missing required named raw coordinate "
                        f"{coordinate.child_name} ({coordinate.source_kind}, "
                        f"{coordinate.consumer_role})"
                    )
                blocks.append(np.zeros((rows, 2), dtype=float))
                continue
            if index in consumed:
                raise RuntimeError("one child raw feature column was claimed more than once")
            consumed.add(index)
            values = np.asarray(prediction.feature_values[:, index], dtype=float)
            if coordinate.required:
                blocks.append(values[:, None].copy())
            else:
                blocks.append(np.column_stack([np.ones(rows, dtype=float), values]))

        for family in self.config.volatile_raw_families:
            indices = [
                index
                for index, (_, kind, role) in enumerate(keys)
                if index not in consumed and (kind, role) == family.key
            ]
            if family.child_name_pattern:
                invalid_names = sorted(
                    name
                    for index, (name, _kind, _role) in enumerate(keys)
                    if index in indices and re.fullmatch(family.child_name_pattern, name) is None
                )
                if invalid_names:
                    raise RuntimeError(
                        "volatile raw feature family contains names outside its "
                        "precommitted membership pattern: " + ", ".join(invalid_names)
                    )
            if not indices:
                if family.required:
                    raise RuntimeError(
                        "child prediction is missing required volatile raw feature family "
                        f"{family.source_kind} ({family.consumer_role})"
                    )
                blocks.append(self._family_summary(None, rows=rows, family=family))
                continue
            overlap = consumed.intersection(indices)
            if overlap:
                raise RuntimeError("one child raw feature column was claimed more than once")
            consumed.update(indices)
            blocks.append(
                self._family_summary(
                    prediction.feature_values[:, indices],
                    rows=rows,
                    family=family,
                )
            )

        unclaimed = sorted(set(range(len(keys))) - consumed)
        if unclaimed:
            descriptions = ", ".join(
                f"{keys[index][0]} ({keys[index][1]}, {keys[index][2]})" for index in unclaimed
            )
            raise RuntimeError(
                "child prediction exposed unconfigured raw feature columns: " + descriptions
            )
        if len(consumed) != len(keys):
            raise RuntimeError("child raw feature consumption accounting is incomplete")

        schema = self.config.raw_output_schema()
        output = np.column_stack(blocks) if blocks else np.empty((rows, 0), dtype=float)
        if output.shape != (rows, len(schema)):
            raise RuntimeError("coordinate-preserving raw output violated its fixed schema")
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
        context_rows = _integer_rows(context_row_ids, name="context_row_ids")
        gate_rows = _integer_rows(gate_row_ids, name="gate_row_ids")
        if set(context_rows) & set(gate_rows):
            raise ValueError("coordinate-preserving context and gate rows must be disjoint")
        exact_context_texts = _exact_texts(
            context_texts, name="context_texts", length=len(context_rows)
        )
        exact_gate_texts = _exact_texts(gate_texts, name="gate_texts", length=len(gate_rows))
        treatment = np.asarray(context_treatment, dtype=float)
        outcome = np.asarray(context_outcome, dtype=float)
        if treatment.ndim != 1 or outcome.ndim != 1:
            raise ValueError("context treatment and outcome must be one-dimensional")
        if len(treatment) != len(context_rows) or len(outcome) != len(context_rows):
            raise ValueError("context labels must align with context rows")
        if not np.isfinite(treatment).all() or not np.isfinite(outcome).all():
            raise ValueError("context labels must be finite")
        treatment_copy = treatment.copy()
        outcome_copy = outcome.copy()
        treatment_copy.setflags(write=False)
        outcome_copy.setflags(write=False)
        prediction = self.backend.fit_predict(
            outer_fold=outer_fold,
            context_row_ids=context_rows,
            context_texts=exact_context_texts,
            context_treatment=treatment_copy,
            context_outcome=outcome_copy,
            gate_row_ids=gate_rows,
            gate_texts=exact_gate_texts,
            work_dir=Path(work_dir) / "coordinate_preserving_child",
        )
        self._assert_stable()
        if type(prediction) is not ContextFitUpstreamPrediction:
            raise TypeError("coordinate-preserving child returned the wrong prediction type")
        if prediction.gate_row_ids != gate_rows:
            raise ValueError("coordinate-preserving child changed gate row identity/order")
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
    "COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID",
    "CoordinatePreservingContextFitUpstreamBackend",
    "CoordinatePreservingUpstreamSchemaConfig",
    "PrecommittedExactCalibratedSource",
    "PrecommittedNamedRawCoordinate",
    "PrecommittedVolatileRawFeatureFamily",
    "VolatileRawFeatureFamilyCapacityOverflowError",
]
