"""Context-fitted upstream evidence for untouched post-extraction gates.

The post-extraction reviewer may compare a proposed explicit-feature registry
with numerical signals, but only when those signals were fitted without any
row in the review gate.  This module provides the narrow authenticated bridge
for such backends.  A backend receives observed context rows plus label-free
gate rows and returns two deliberately separate products:

* probability-scale treatment-effect predictions, which may be evaluated by
  the calibrated source/R-loss guards; and
* uncalibrated nuisance or modifier bases, which may only be evaluated through
  role-matched preservation correlations.

The provider seals the exact input projection, backend identity, output bytes,
and recursive fit-row lineage.  It never passes gate treatment or outcome to a
backend and can implement both runner-facing provider protocols from one fit.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable, Mapping, Protocol, Sequence

import numpy as np

from .all_evidence_post_extraction_review import (
    GateFeatureBankView,
    GateSourceSignalView,
    ObservableCausalRows,
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .fold_honest_r_stack import FitRowProvenance

CONTEXT_FIT_UPSTREAM_PREDICTION_SCHEMA_VERSION = "context_fit_upstream_prediction_v1"
CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION = "context_fit_upstream_gate_cache_v6"
CONTEXT_FIT_UPSTREAM_PROVIDER_ID = "context_fit_upstream_gate_provider_v6"
CONTEXT_FIT_UPSTREAM_CALL_CHECKPOINT_SCHEMA_VERSION = "context_fit_upstream_call_checkpoint_v1"

_ROLES = frozenset(
    {
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    }
)
_FORBIDDEN = ("true", "oracle", "ground_truth")
_CACHE_FIELDS = frozenset(
    {
        "schema_version",
        "cache_key",
        "binding",
        "context_row_ids",
        "context_inner_fold_ids",
        "gate_row_ids",
        "source_names",
        "source_kinds",
        "source_values_file",
        "source_values_sha256",
        "source_context_values_file",
        "source_context_values_sha256",
        "feature_names",
        "feature_kinds",
        "feature_roles",
        "feature_values_file",
        "feature_values_sha256",
        "feature_context_values_file",
        "feature_context_values_sha256",
        "content_sha256",
    }
)
_CALL_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "cache_key",
        "binding",
        "gate_row_ids",
        "source_names",
        "source_kinds",
        "source_values_file",
        "source_values_sha256",
        "feature_names",
        "feature_kinds",
        "feature_roles",
        "feature_values_file",
        "feature_values_sha256",
        "content_sha256",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _exclusive_cache_lock(path: Path):
    """Serialize one exact cache key across processes and release on crashes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _load_authenticated_npy(path: Path, *, expected_sha256: str) -> np.ndarray:
    """Authenticate and parse one immutable byte snapshot of a NumPy matrix."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError("context-fit upstream cache matrix is unreadable") from exc
    if hashlib.sha256(payload).hexdigest() != str(expected_sha256):
        raise ValueError("context-fit upstream cache matrix SHA-256 mismatch")
    try:
        return np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError("context-fit upstream cache matrix is unreadable") from exc


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _closed_json(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            if not key or any(token in key.lower() for token in _FORBIDDEN):
                raise ValueError(f"{path} contains a forbidden or empty identity field")
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
        if any(token in value.lower() for token in _FORBIDDEN):
            raise ValueError(f"{path} contains a forbidden identity value")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite identity value")
        return value
    raise TypeError(f"{path} must contain closed JSON-compatible metadata")


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _integer_rows(values: Sequence[Any], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of integer row IDs")
    result: list[int] = []
    for value in tuple(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must contain canonical integer row IDs")
        normalized = int(value)
        if normalized < 0:
            raise ValueError(f"{name} cannot contain negative row IDs")
        result.append(normalized)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    return tuple(result)


def _context_folds(values: Sequence[Any], *, length: int) -> tuple[Hashable, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError("context.inner_fold_ids must be a sequence")
    result: list[Hashable] = []
    for value in tuple(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer, str)):
            raise TypeError("context.inner_fold_ids must contain integer or string IDs")
        normalized: Hashable = int(value) if isinstance(value, (int, np.integer)) else value.strip()
        if isinstance(normalized, str) and not normalized:
            raise ValueError("context.inner_fold_ids cannot contain empty strings")
        result.append(normalized)
    if len(result) != int(length) or len(set(result)) < 2:
        raise ValueError("context.inner_fold_ids must define at least two aligned folds")
    return tuple(result)


def _exact_texts(values: Sequence[Any], *, name: str, length: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of exact strings")
    raw = tuple(values)
    if len(raw) != int(length) or not all(isinstance(value, str) for value in raw):
        raise ValueError(f"{name} must contain exactly {length} strings")
    return raw


def _finite_matrix(
    values: Any,
    *,
    name: str,
    rows: int,
    columns: int,
    allow_empty: bool,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (int(rows), int(columns)):
        raise ValueError(f"{name} must have shape {(int(rows), int(columns))}")
    if not allow_empty and int(columns) < 1:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    result = matrix.copy()
    result.setflags(write=False)
    return result


def _names(values: Sequence[Any], *, name: str, allow_empty: bool = True) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if (not allow_empty and not result) or any(not value for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    if any(any(token in value.lower() for token in _FORBIDDEN) for value in result):
        raise ValueError(f"{name} contains forbidden benchmark metadata")
    return result


def _float_hex_sha256(values: Sequence[float]) -> str:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("observable vectors must be finite and one-dimensional")
    return _sha256_json([float(value).hex() for value in vector])


@dataclass(frozen=True)
class ContextFitUpstreamPrediction:
    """Closed output of one context-only upstream backend fit."""

    gate_row_ids: tuple[int, ...]
    calibrated_source_names: tuple[str, ...] = ()
    calibrated_source_kinds: tuple[str, ...] = ()
    calibrated_source_values: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float), repr=False
    )
    feature_names: tuple[str, ...] = ()
    feature_kinds: tuple[str, ...] = ()
    feature_roles: tuple[str, ...] = ()
    feature_values: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float), repr=False
    )

    def __post_init__(self) -> None:
        rows = _integer_rows(self.gate_row_ids, name="prediction.gate_row_ids")
        source_names = _names(
            self.calibrated_source_names,
            name="prediction.calibrated_source_names",
        )
        source_kinds = tuple(str(value).strip() for value in self.calibrated_source_kinds)
        if len(source_kinds) != len(source_names) or any(not value for value in source_kinds):
            raise ValueError("calibrated_source_kinds must align with calibrated sources")
        if any(any(token in value.lower() for token in _FORBIDDEN) for value in source_kinds):
            raise ValueError("calibrated source kinds contain forbidden benchmark metadata")
        source_values = _finite_matrix(
            self.calibrated_source_values,
            name="prediction.calibrated_source_values",
            rows=len(rows),
            columns=len(source_names),
            allow_empty=True,
        )
        feature_names = _names(self.feature_names, name="prediction.feature_names")
        feature_kinds = tuple(str(value).strip() for value in self.feature_kinds)
        feature_roles = tuple(str(value).strip() for value in self.feature_roles)
        if len(feature_kinds) != len(feature_names) or any(not value for value in feature_kinds):
            raise ValueError("feature_kinds must align with feature_names")
        if len(feature_roles) != len(feature_names) or set(feature_roles) - _ROLES:
            raise ValueError("feature_roles contain an unsupported consumer role")
        if any(
            any(token in value.lower() for token in _FORBIDDEN)
            for value in (*feature_kinds, *feature_roles)
        ):
            raise ValueError("feature metadata contains forbidden benchmark metadata")
        feature_values = _finite_matrix(
            self.feature_values,
            name="prediction.feature_values",
            rows=len(rows),
            columns=len(feature_names),
            allow_empty=True,
        )
        if not source_names and not feature_names:
            raise ValueError("an upstream prediction must expose at least one safe output")
        object.__setattr__(self, "gate_row_ids", rows)
        object.__setattr__(self, "calibrated_source_names", source_names)
        object.__setattr__(self, "calibrated_source_kinds", source_kinds)
        object.__setattr__(self, "calibrated_source_values", source_values)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "feature_kinds", feature_kinds)
        object.__setattr__(self, "feature_roles", feature_roles)
        object.__setattr__(self, "feature_values", feature_values)


class ContextFitUpstreamBackend(Protocol):
    """Backend boundary that makes gate labels structurally unavailable."""

    def identity(self) -> Mapping[str, Any]: ...

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
    ) -> ContextFitUpstreamPrediction: ...


class CompositeContextFitUpstreamBackend:
    """Combine independently fitted context-only backends without re-roling data."""

    def __init__(self, backends: Sequence[ContextFitUpstreamBackend]) -> None:
        self.backends = tuple(backends)
        if not self.backends:
            raise ValueError("composite upstream backend requires at least one backend")
        identities: list[Any] = []
        for index, backend in enumerate(self.backends):
            if not callable(getattr(backend, "identity", None)) or not callable(
                getattr(backend, "fit_predict", None)
            ):
                raise TypeError("composite member does not implement the backend protocol")
            identities.append(_closed_json(backend.identity(), path=f"backends[{index}].identity"))
        if len({_sha256_json(value) for value in identities}) != len(identities):
            raise ValueError("composite upstream backend identities must be unique")
        self._identities = tuple(identities)

    def _assert_members_stable(self) -> None:
        current = tuple(
            _closed_json(backend.identity(), path=f"backends[{index}].identity")
            for index, backend in enumerate(self.backends)
        )
        if current != self._identities:
            raise ValueError("composite upstream member identity changed")

    def identity(self) -> Mapping[str, Any]:
        self._assert_members_stable()
        return {
            "backend": "composite_context_fit_upstream_backend_v1",
            "members": list(self._identities),
            "gate_labels_exposed_to_members": False,
        }

    def fit_predict(self, **kwargs: Any) -> ContextFitUpstreamPrediction:
        self._assert_members_stable()
        expected_gate = tuple(kwargs["gate_row_ids"])
        source_names: list[str] = []
        source_kinds: list[str] = []
        source_columns: list[np.ndarray] = []
        feature_names: list[str] = []
        feature_kinds: list[str] = []
        feature_roles: list[str] = []
        feature_columns: list[np.ndarray] = []
        base_work_dir = Path(kwargs["work_dir"])
        for index, backend in enumerate(self.backends, start=1):
            member_kwargs = dict(kwargs)
            member_kwargs["work_dir"] = base_work_dir / f"member_{index:03d}"
            prediction = backend.fit_predict(**member_kwargs)
            self._assert_members_stable()
            if not isinstance(prediction, ContextFitUpstreamPrediction):
                raise TypeError("composite member returned the wrong prediction type")
            if prediction.gate_row_ids != expected_gate:
                raise ValueError("composite member changed gate row identity/order")
            source_names.extend(prediction.calibrated_source_names)
            source_kinds.extend(prediction.calibrated_source_kinds)
            source_columns.extend(
                prediction.calibrated_source_values[:, column]
                for column in range(len(prediction.calibrated_source_names))
            )
            feature_names.extend(prediction.feature_names)
            feature_kinds.extend(prediction.feature_kinds)
            feature_roles.extend(prediction.feature_roles)
            feature_columns.extend(
                prediction.feature_values[:, column]
                for column in range(len(prediction.feature_names))
            )
        if len(source_names) != len(set(source_names)):
            raise ValueError("composite calibrated source names collide")
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("composite feature names collide")
        n_rows = len(expected_gate)
        return ContextFitUpstreamPrediction(
            gate_row_ids=expected_gate,
            calibrated_source_names=tuple(source_names),
            calibrated_source_kinds=tuple(source_kinds),
            calibrated_source_values=(
                np.column_stack(source_columns)
                if source_columns
                else np.empty((n_rows, 0), dtype=float)
            ),
            feature_names=tuple(feature_names),
            feature_kinds=tuple(feature_kinds),
            feature_roles=tuple(feature_roles),
            feature_values=(
                np.column_stack(feature_columns)
                if feature_columns
                else np.empty((n_rows, 0), dtype=float)
            ),
        )


@dataclass(frozen=True)
class _PreparedViews:
    source: GateSourceSignalView | None
    features: GateFeatureBankView | None


class ContextFitUpstreamGateProvider:
    """Fit, seal, and serve one all-upstream bundle per untouched gate."""

    def __init__(self, cache_dir: Path | str, *, backend: ContextFitUpstreamBackend) -> None:
        self.cache_dir = Path(cache_dir)
        if not callable(getattr(backend, "identity", None)) or not callable(
            getattr(backend, "fit_predict", None)
        ):
            raise TypeError("backend must implement identity() and fit_predict()")
        self.backend = backend
        self._backend_identity = _closed_json(backend.identity(), path="backend.identity")
        self._prepared: dict[tuple[int, tuple[int, ...]], _PreparedViews] = {}

    def _assert_identity_stable(self) -> None:
        current = _closed_json(self.backend.identity(), path="backend.identity")
        if current != self._backend_identity:
            raise ValueError("upstream backend identity changed after provider construction")

    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": CONTEXT_FIT_UPSTREAM_PROVIDER_ID,
            "provider_code_sha256": _module_sha256(),
            "backend": self._backend_identity,
            "gate_bind_api": "exact_row_ids_and_text_only_v1",
            "gate_labels_exposed_to_backend": False,
            "raw_features_are_calibrated_effects": False,
            "cache_matrix_authentication": "single_read_sha256_bytesio_numpy_v1",
            "fit_call_checkpoint_schema": (CONTEXT_FIT_UPSTREAM_CALL_CHECKPOINT_SCHEMA_VERSION),
            "fit_call_checkpoint_binding": "exact_observable_inputs_no_gate_labels_v1",
            "fit_call_checkpoint_publication": "matrices_then_manifest_atomic_replace_v1",
            "cache_key_concurrency": "exclusive_advisory_file_lock_across_compute_publish_v1",
            "context_values": "exact_inner_fold_oof_v1",
            "gate_values": "complete_spent_context_fit_v1",
            "adaptive_acceptance_conditional_context_supported": True,
        }

    def _binding(
        self,
        *,
        outer_fold: int,
        context: ObservableCausalRows,
        gate_row_ids: tuple[int, ...],
        context_texts: tuple[str, ...],
        gate_texts: tuple[str, ...],
        context_inner_fold_ids: tuple[Hashable, ...],
    ) -> dict[str, Any]:
        return {
            "provider_identity": _closed_json(self.identity(), path="provider.identity"),
            "outer_fold": int(outer_fold),
            "context_row_ids_sha256": _sha256_json(list(map(int, context.row_ids))),
            "context_text_sha256": _sha256_json(list(context_texts)),
            "context_treatment_sha256": _float_hex_sha256(context.treatment),
            "context_outcome_sha256": _float_hex_sha256(context.outcome),
            "context_inner_fold_assignment_sha256": _sha256_json(
                {
                    "row_ids": list(map(int, context.row_ids)),
                    "inner_fold_ids": list(context_inner_fold_ids),
                }
            ),
            "gate_row_ids_sha256": _sha256_json(list(gate_row_ids)),
            "gate_text_sha256": _sha256_json(list(gate_texts)),
            "context_row_count": len(context.row_ids),
            "gate_row_count": len(gate_row_ids),
            "gate_labels_in_binding": False,
            "gate_labels_exposed_to_backend": False,
            "context_values_cross_fitted_by_exact_inner_fold": True,
        }

    def _paths(self, cache_key: str) -> tuple[Path, Path, Path, Path, Path]:
        root = self.cache_dir / cache_key
        return (
            root / "manifest.json",
            root / "calibrated_sources.npy",
            root / "features.npy",
            root / "calibrated_sources_context_oof.npy",
            root / "features_context_oof.npy",
        )

    @staticmethod
    def _views(
        prediction: ContextFitUpstreamPrediction,
        *,
        context_row_ids: tuple[int, ...],
        context_inner_fold_ids: tuple[Hashable, ...],
        source_context_values: np.ndarray,
        feature_context_values: np.ndarray,
    ) -> _PreparedViews:
        lineage = FitRowProvenance(fit_row_ids=frozenset(context_row_ids))
        rows_by_fold = {
            fold_id: frozenset(
                row_id
                for row_id, candidate_fold in zip(context_row_ids, context_inner_fold_ids)
                if candidate_fold == fold_id
            )
            for fold_id in dict.fromkeys(context_inner_fold_ids)
        }
        context_lineage_by_row = tuple(
            FitRowProvenance(fit_row_ids=frozenset(context_row_ids) - rows_by_fold[fold_id])
            for fold_id in context_inner_fold_ids
        )
        source = None
        if prediction.calibrated_source_names:
            source = GateSourceSignalView(
                row_ids=prediction.gate_row_ids,
                source_names=prediction.calibrated_source_names,
                source_kinds=prediction.calibrated_source_kinds,
                values=prediction.calibrated_source_values,
                fit_row_provenance=tuple(lineage for _ in prediction.calibrated_source_names),
                context_row_ids=context_row_ids,
                context_inner_fold_ids=context_inner_fold_ids,
                context_values=source_context_values,
                context_fit_row_provenance=tuple(
                    context_lineage_by_row for _ in prediction.calibrated_source_names
                ),
            )
        features = None
        if prediction.feature_names:
            features = GateFeatureBankView(
                row_ids=prediction.gate_row_ids,
                feature_names=prediction.feature_names,
                source_kinds=prediction.feature_kinds,
                consumer_roles=prediction.feature_roles,
                values=prediction.feature_values,
                fit_row_provenance=tuple(lineage for _ in prediction.feature_names),
                context_row_ids=context_row_ids,
                context_inner_fold_ids=context_inner_fold_ids,
                context_values=feature_context_values,
                context_fit_row_provenance=tuple(
                    context_lineage_by_row for _ in prediction.feature_names
                ),
            )
        return _PreparedViews(source=source, features=features)

    def _load_cache(
        self,
        *,
        manifest_path: Path,
        source_path: Path,
        feature_path: Path,
        source_context_path: Path,
        feature_context_path: Path,
        cache_key: str,
        binding: Mapping[str, Any],
        context_row_ids: tuple[int, ...],
        context_inner_fold_ids: tuple[Hashable, ...],
        gate_row_ids: tuple[int, ...],
    ) -> _PreparedViews:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("context-fit upstream cache manifest is unreadable") from exc
        if not isinstance(payload, Mapping) or set(payload) != _CACHE_FIELDS:
            raise ValueError("context-fit upstream cache does not match its closed schema")
        if payload["schema_version"] != CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported context-fit upstream cache schema")
        if payload["cache_key"] != cache_key or payload["binding"] != binding:
            raise ValueError("context-fit upstream cache binding mismatch")
        content = {key: value for key, value in payload.items() if key != "content_sha256"}
        if payload["content_sha256"] != _sha256_json(content):
            raise ValueError("context-fit upstream cache manifest SHA-256 mismatch")
        if (
            payload["source_values_file"] != source_path.name
            or payload["feature_values_file"] != feature_path.name
            or payload["source_context_values_file"] != source_context_path.name
            or payload["feature_context_values_file"] != feature_context_path.name
        ):
            raise ValueError("context-fit upstream cache filenames were changed")
        source_values = _load_authenticated_npy(
            source_path,
            expected_sha256=payload["source_values_sha256"],
        )
        feature_values = _load_authenticated_npy(
            feature_path,
            expected_sha256=payload["feature_values_sha256"],
        )
        source_context_values = _load_authenticated_npy(
            source_context_path,
            expected_sha256=payload["source_context_values_sha256"],
        )
        feature_context_values = _load_authenticated_npy(
            feature_context_path,
            expected_sha256=payload["feature_context_values_sha256"],
        )
        prediction = ContextFitUpstreamPrediction(
            gate_row_ids=tuple(payload["gate_row_ids"]),
            calibrated_source_names=tuple(payload["source_names"]),
            calibrated_source_kinds=tuple(payload["source_kinds"]),
            calibrated_source_values=source_values,
            feature_names=tuple(payload["feature_names"]),
            feature_kinds=tuple(payload["feature_kinds"]),
            feature_roles=tuple(payload["feature_roles"]),
            feature_values=feature_values,
        )
        if prediction.gate_row_ids != gate_row_ids:
            raise ValueError("context-fit upstream cache changed gate identity/order")
        if (
            tuple(payload["context_row_ids"]) != context_row_ids
            or tuple(payload["context_inner_fold_ids"]) != context_inner_fold_ids
        ):
            raise ValueError("context-fit upstream cache changed context identity/folds")
        return self._views(
            prediction,
            context_row_ids=context_row_ids,
            context_inner_fold_ids=context_inner_fold_ids,
            source_context_values=source_context_values,
            feature_context_values=feature_context_values,
        )

    def _write_cache(
        self,
        *,
        manifest_path: Path,
        source_path: Path,
        feature_path: Path,
        source_context_path: Path,
        feature_context_path: Path,
        cache_key: str,
        binding: Mapping[str, Any],
        prediction: ContextFitUpstreamPrediction,
        context_row_ids: tuple[int, ...],
        context_inner_fold_ids: tuple[Hashable, ...],
        source_context_values: np.ndarray,
        feature_context_values: np.ndarray,
    ) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_paths: list[Path] = []
        try:
            for destination, values in (
                (source_path, prediction.calibrated_source_values),
                (feature_path, prediction.feature_values),
                (source_context_path, source_context_values),
                (feature_context_path, feature_context_values),
            ):
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=manifest_path.parent,
                    prefix=f".{destination.name}.",
                    delete=False,
                ) as handle:
                    np.save(handle, np.asarray(values, dtype=np.float64), allow_pickle=False)
                    temporary = Path(handle.name)
                temporary_paths.append(temporary)
                temporary.replace(destination)
                temporary_paths.remove(temporary)
            content = {
                "schema_version": CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION,
                "cache_key": cache_key,
                "binding": binding,
                "context_row_ids": list(context_row_ids),
                "context_inner_fold_ids": list(context_inner_fold_ids),
                "gate_row_ids": list(prediction.gate_row_ids),
                "source_names": list(prediction.calibrated_source_names),
                "source_kinds": list(prediction.calibrated_source_kinds),
                "source_values_file": source_path.name,
                "source_values_sha256": _sha256_file(source_path),
                "source_context_values_file": source_context_path.name,
                "source_context_values_sha256": _sha256_file(source_context_path),
                "feature_names": list(prediction.feature_names),
                "feature_kinds": list(prediction.feature_kinds),
                "feature_roles": list(prediction.feature_roles),
                "feature_values_file": feature_path.name,
                "feature_values_sha256": _sha256_file(feature_path),
                "feature_context_values_file": feature_context_path.name,
                "feature_context_values_sha256": _sha256_file(feature_context_path),
            }
            payload = {**content, "content_sha256": _sha256_json(content)}
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=manifest_path.parent,
                prefix=f".{manifest_path.name}.",
                delete=False,
            ) as handle:
                handle.write(_canonical_json(payload) + "\n")
                temporary_manifest = Path(handle.name)
            temporary_paths.append(temporary_manifest)
            temporary_manifest.replace(manifest_path)
            temporary_paths.remove(temporary_manifest)
        finally:
            for path in temporary_paths:
                path.unlink(missing_ok=True)

    def _call_checkpoint_binding(
        self,
        *,
        outer_fold: int,
        context_row_ids: tuple[int, ...],
        context_texts: tuple[str, ...],
        context_treatment: np.ndarray,
        context_outcome: np.ndarray,
        gate_row_ids: tuple[int, ...],
        gate_texts: tuple[str, ...],
    ) -> dict[str, Any]:
        """Seal one backend call without ever admitting gate labels."""

        return {
            "checkpoint_schema_version": (CONTEXT_FIT_UPSTREAM_CALL_CHECKPOINT_SCHEMA_VERSION),
            "provider_code_sha256": _module_sha256(),
            "backend_identity": self._backend_identity,
            "outer_fold": int(outer_fold),
            "context_row_ids_sha256": _sha256_json(list(context_row_ids)),
            "context_text_sha256": _sha256_json(list(context_texts)),
            "context_treatment_sha256": _float_hex_sha256(context_treatment),
            "context_outcome_sha256": _float_hex_sha256(context_outcome),
            "gate_row_ids_sha256": _sha256_json(list(gate_row_ids)),
            "gate_text_sha256": _sha256_json(list(gate_texts)),
            "context_row_count": len(context_row_ids),
            "gate_row_count": len(gate_row_ids),
            "gate_labels_in_binding": False,
            "gate_labels_exposed_to_backend": False,
        }

    def _call_checkpoint_paths(self, cache_key: str) -> tuple[Path, Path, Path]:
        root = self.cache_dir / "_fit_call_checkpoints" / cache_key
        return root / "manifest.json", root / "calibrated_sources.npy", root / "features.npy"

    def _load_call_checkpoint(
        self,
        *,
        manifest_path: Path,
        source_path: Path,
        feature_path: Path,
        cache_key: str,
        binding: Mapping[str, Any],
        gate_row_ids: tuple[int, ...],
    ) -> ContextFitUpstreamPrediction:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("context-fit call checkpoint manifest is unreadable") from exc
        if not isinstance(payload, Mapping) or set(payload) != _CALL_CHECKPOINT_FIELDS:
            raise ValueError("context-fit call checkpoint does not match its closed schema")
        if payload["schema_version"] != CONTEXT_FIT_UPSTREAM_CALL_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported context-fit call checkpoint schema")
        if payload["cache_key"] != cache_key or payload["binding"] != binding:
            raise ValueError("context-fit call checkpoint binding mismatch")
        content = {key: value for key, value in payload.items() if key != "content_sha256"}
        if payload["content_sha256"] != _sha256_json(content):
            raise ValueError("context-fit call checkpoint manifest SHA-256 mismatch")
        if (
            payload["source_values_file"] != source_path.name
            or payload["feature_values_file"] != feature_path.name
        ):
            raise ValueError("context-fit call checkpoint filenames were changed")
        prediction = ContextFitUpstreamPrediction(
            gate_row_ids=tuple(payload["gate_row_ids"]),
            calibrated_source_names=tuple(payload["source_names"]),
            calibrated_source_kinds=tuple(payload["source_kinds"]),
            calibrated_source_values=_load_authenticated_npy(
                source_path,
                expected_sha256=payload["source_values_sha256"],
            ),
            feature_names=tuple(payload["feature_names"]),
            feature_kinds=tuple(payload["feature_kinds"]),
            feature_roles=tuple(payload["feature_roles"]),
            feature_values=_load_authenticated_npy(
                feature_path,
                expected_sha256=payload["feature_values_sha256"],
            ),
        )
        if prediction.gate_row_ids != gate_row_ids:
            raise ValueError("context-fit call checkpoint changed gate identity/order")
        return prediction

    def _write_call_checkpoint(
        self,
        *,
        manifest_path: Path,
        source_path: Path,
        feature_path: Path,
        cache_key: str,
        binding: Mapping[str, Any],
        prediction: ContextFitUpstreamPrediction,
    ) -> None:
        """Publish matrices first and an authenticated completion marker last."""

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_paths: list[Path] = []
        try:
            for destination, values in (
                (source_path, prediction.calibrated_source_values),
                (feature_path, prediction.feature_values),
            ):
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=manifest_path.parent,
                    prefix=f".{destination.name}.",
                    delete=False,
                ) as handle:
                    np.save(handle, np.asarray(values, dtype=np.float64), allow_pickle=False)
                    temporary = Path(handle.name)
                temporary_paths.append(temporary)
                temporary.replace(destination)
                temporary_paths.remove(temporary)
            content = {
                "schema_version": CONTEXT_FIT_UPSTREAM_CALL_CHECKPOINT_SCHEMA_VERSION,
                "cache_key": cache_key,
                "binding": binding,
                "gate_row_ids": list(prediction.gate_row_ids),
                "source_names": list(prediction.calibrated_source_names),
                "source_kinds": list(prediction.calibrated_source_kinds),
                "source_values_file": source_path.name,
                "source_values_sha256": _sha256_file(source_path),
                "feature_names": list(prediction.feature_names),
                "feature_kinds": list(prediction.feature_kinds),
                "feature_roles": list(prediction.feature_roles),
                "feature_values_file": feature_path.name,
                "feature_values_sha256": _sha256_file(feature_path),
            }
            payload = {**content, "content_sha256": _sha256_json(content)}
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=manifest_path.parent,
                prefix=f".{manifest_path.name}.",
                delete=False,
            ) as handle:
                handle.write(_canonical_json(payload) + "\n")
                temporary_manifest = Path(handle.name)
            temporary_paths.append(temporary_manifest)
            temporary_manifest.replace(manifest_path)
            temporary_paths.remove(temporary_manifest)
        finally:
            for path in temporary_paths:
                path.unlink(missing_ok=True)

    def _fit_predict_checkpointed(
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
        """Resume exactly completed backend calls and reject corrupt completions."""

        binding = self._call_checkpoint_binding(
            outer_fold=outer_fold,
            context_row_ids=context_row_ids,
            context_texts=context_texts,
            context_treatment=context_treatment,
            context_outcome=context_outcome,
            gate_row_ids=gate_row_ids,
            gate_texts=gate_texts,
        )
        cache_key = _sha256_json(binding)
        manifest_path, source_path, feature_path = self._call_checkpoint_paths(cache_key)
        with _exclusive_cache_lock(manifest_path.parent / ".fit_call.lock"):
            if manifest_path.exists():
                prediction = self._load_call_checkpoint(
                    manifest_path=manifest_path,
                    source_path=source_path,
                    feature_path=feature_path,
                    cache_key=cache_key,
                    binding=binding,
                    gate_row_ids=gate_row_ids,
                )
                self._assert_identity_stable()
                return prediction

            prediction = self.backend.fit_predict(
                outer_fold=int(outer_fold),
                context_row_ids=context_row_ids,
                context_texts=context_texts,
                context_treatment=np.asarray(context_treatment, dtype=float).copy(),
                context_outcome=np.asarray(context_outcome, dtype=float).copy(),
                gate_row_ids=gate_row_ids,
                gate_texts=gate_texts,
                work_dir=work_dir,
            )
            self._assert_identity_stable()
            if not isinstance(prediction, ContextFitUpstreamPrediction):
                raise TypeError("upstream backend returned the wrong prediction type")
            if prediction.gate_row_ids != gate_row_ids:
                raise ValueError("upstream backend changed gate row identity/order")
            self._write_call_checkpoint(
                manifest_path=manifest_path,
                source_path=source_path,
                feature_path=feature_path,
                cache_key=cache_key,
                binding=binding,
                prediction=prediction,
            )
            return prediction

    @staticmethod
    def _prediction_schema(prediction: ContextFitUpstreamPrediction) -> tuple[Any, ...]:
        if not isinstance(prediction, ContextFitUpstreamPrediction):
            raise TypeError("upstream backend returned the wrong prediction type")
        return (
            prediction.calibrated_source_names,
            prediction.calibrated_source_kinds,
            prediction.feature_names,
            prediction.feature_kinds,
            prediction.feature_roles,
        )

    def _cross_fit_context_and_predict_gate(
        self,
        *,
        outer_fold: int,
        context: ObservableCausalRows,
        context_row_ids: tuple[int, ...],
        context_inner_fold_ids: tuple[Hashable, ...],
        context_texts: tuple[str, ...],
        gate_row_ids: tuple[int, ...],
        gate_texts: tuple[str, ...],
        work_dir: Path,
    ) -> tuple[ContextFitUpstreamPrediction, np.ndarray, np.ndarray]:
        """Produce exact inner-fold OOF context banks plus one full-context gate bank."""

        positions = np.arange(len(context_row_ids), dtype=int)
        folds = np.asarray(context_inner_fold_ids, dtype=object)
        source_context: np.ndarray | None = None
        feature_context: np.ndarray | None = None
        expected_schema: tuple[Any, ...] | None = None
        ordered_folds = sorted(set(context_inner_fold_ids), key=_canonical_json)
        for display_index, fold_id in enumerate(ordered_folds, start=1):
            heldout = folds == fold_id
            fit = ~heldout
            if not np.any(heldout) or not np.any(fit):
                raise ValueError("every context inner fold must have fit and heldout rows")
            fit_positions = positions[fit]
            heldout_positions = positions[heldout]
            prediction = self._fit_predict_checkpointed(
                outer_fold=int(outer_fold),
                context_row_ids=tuple(context_row_ids[index] for index in fit_positions),
                context_texts=tuple(context_texts[index] for index in fit_positions),
                context_treatment=np.asarray(context.treatment[fit_positions], dtype=float),
                context_outcome=np.asarray(context.outcome[fit_positions], dtype=float),
                gate_row_ids=tuple(context_row_ids[index] for index in heldout_positions),
                gate_texts=tuple(context_texts[index] for index in heldout_positions),
                work_dir=work_dir / "context_oof" / f"fold_{display_index:04d}",
            )
            self._assert_identity_stable()
            schema = self._prediction_schema(prediction)
            if prediction.gate_row_ids != tuple(
                context_row_ids[index] for index in heldout_positions
            ):
                raise ValueError("upstream backend changed an OOF context row identity/order")
            if expected_schema is None:
                expected_schema = schema
                source_context = np.full(
                    (len(context_row_ids), len(prediction.calibrated_source_names)),
                    np.nan,
                    dtype=float,
                )
                feature_context = np.full(
                    (len(context_row_ids), len(prediction.feature_names)),
                    np.nan,
                    dtype=float,
                )
            elif schema != expected_schema:
                raise ValueError("upstream backend changed schema across context OOF fits")
            assert source_context is not None and feature_context is not None
            source_context[heldout_positions] = prediction.calibrated_source_values
            feature_context[heldout_positions] = prediction.feature_values

        gate_prediction = self._fit_predict_checkpointed(
            outer_fold=int(outer_fold),
            context_row_ids=context_row_ids,
            context_texts=context_texts,
            context_treatment=np.asarray(context.treatment, dtype=float),
            context_outcome=np.asarray(context.outcome, dtype=float),
            gate_row_ids=gate_row_ids,
            gate_texts=gate_texts,
            work_dir=work_dir / "untouched_gate",
        )
        self._assert_identity_stable()
        gate_schema = self._prediction_schema(gate_prediction)
        if gate_prediction.gate_row_ids != gate_row_ids:
            raise ValueError("upstream backend changed gate row identity/order")
        if expected_schema is None or gate_schema != expected_schema:
            raise ValueError("upstream backend changed schema between context and gate fits")
        assert source_context is not None and feature_context is not None
        if not np.isfinite(source_context).all() or not np.isfinite(feature_context).all():
            raise RuntimeError("context OOF upstream fits did not cover every row")
        return gate_prediction, source_context, feature_context

    def bind_fold(
        self,
        *,
        outer_fold: int,
        context: ObservableCausalRows,
        context_texts: Sequence[str],
        gate_texts: Sequence[str],
        exact_gate_row_ids: Sequence[int],
    ) -> "BoundContextFitUpstreamGateProvider":
        fold = _positive_int(outer_fold, name="outer_fold")
        self._assert_identity_stable()
        if not isinstance(context, ObservableCausalRows):
            raise TypeError("context must be ObservableCausalRows")
        context_ids = _integer_rows(context.row_ids, name="context.row_ids")
        if context.inner_fold_ids is None:
            raise ValueError("context.inner_fold_ids are required for cross-fitted upstream banks")
        context_folds = _context_folds(
            context.inner_fold_ids,
            length=len(context_ids),
        )
        gate_ids = _integer_rows(exact_gate_row_ids, name="exact_gate_row_ids")
        if set(context_ids) & set(gate_ids):
            raise ValueError("review context and gate rows must be disjoint")
        exact_context_texts = _exact_texts(
            context_texts, name="context_texts", length=len(context_ids)
        )
        exact_gate_texts = _exact_texts(gate_texts, name="gate_texts", length=len(gate_ids))
        binding = self._binding(
            outer_fold=fold,
            context=context,
            gate_row_ids=gate_ids,
            context_texts=exact_context_texts,
            gate_texts=exact_gate_texts,
            context_inner_fold_ids=context_folds,
        )
        cache_key = _sha256_json(binding)
        (
            manifest_path,
            source_path,
            feature_path,
            source_context_path,
            feature_context_path,
        ) = self._paths(cache_key)
        with _exclusive_cache_lock(manifest_path.parent / ".complete_bind.lock"):
            if manifest_path.exists():
                views = self._load_cache(
                    manifest_path=manifest_path,
                    source_path=source_path,
                    feature_path=feature_path,
                    source_context_path=source_context_path,
                    feature_context_path=feature_context_path,
                    cache_key=cache_key,
                    binding=binding,
                    context_row_ids=context_ids,
                    context_inner_fold_ids=context_folds,
                    gate_row_ids=gate_ids,
                )
            else:
                prediction, source_context_values, feature_context_values = (
                    self._cross_fit_context_and_predict_gate(
                        outer_fold=fold,
                        context=context,
                        context_row_ids=context_ids,
                        context_inner_fold_ids=context_folds,
                        context_texts=exact_context_texts,
                        gate_row_ids=gate_ids,
                        gate_texts=exact_gate_texts,
                        work_dir=manifest_path.parent / "backend_work",
                    )
                )
                self._write_cache(
                    manifest_path=manifest_path,
                    source_path=source_path,
                    feature_path=feature_path,
                    source_context_path=source_context_path,
                    feature_context_path=feature_context_path,
                    cache_key=cache_key,
                    binding=binding,
                    prediction=prediction,
                    context_row_ids=context_ids,
                    context_inner_fold_ids=context_folds,
                    source_context_values=source_context_values,
                    feature_context_values=feature_context_values,
                )
                views = self._views(
                    prediction,
                    context_row_ids=context_ids,
                    context_inner_fold_ids=context_folds,
                    source_context_values=source_context_values,
                    feature_context_values=feature_context_values,
                )
        prepared_key = (fold, gate_ids)
        self._prepared[prepared_key] = views
        return BoundContextFitUpstreamGateProvider(
            outer_fold=fold,
            exact_gate_row_ids=gate_ids,
            views=views,
            parent_identity=self.identity(),
            cache_manifest_path=manifest_path,
            cache_manifest_sha256=_sha256_file(manifest_path),
        )

    def get_gate_source_view(
        self, *, outer_fold: int, exact_gate_row_ids: Sequence[int]
    ) -> GateSourceSignalView:
        views = self._lookup(outer_fold=outer_fold, exact_gate_row_ids=exact_gate_row_ids)
        if views.source is None:
            raise RuntimeError("prepared upstream backend has no calibrated sources")
        return views.source

    def get_gate_feature_bank_view(
        self, *, outer_fold: int, exact_gate_row_ids: Sequence[int]
    ) -> GateFeatureBankView:
        views = self._lookup(outer_fold=outer_fold, exact_gate_row_ids=exact_gate_row_ids)
        if views.features is None:
            raise RuntimeError("prepared upstream backend has no role-aware feature banks")
        return views.features

    def _lookup(self, *, outer_fold: int, exact_gate_row_ids: Sequence[int]) -> _PreparedViews:
        fold = _positive_int(outer_fold, name="outer_fold")
        rows = _integer_rows(exact_gate_row_ids, name="exact_gate_row_ids")
        try:
            return self._prepared[(fold, rows)]
        except KeyError as exc:
            raise RuntimeError("upstream gate has not been prepared with bind_fold()") from exc


class BoundContextFitUpstreamGateProvider:
    """One immutable, label-free lookup returned after a context-only fit."""

    def __init__(
        self,
        *,
        outer_fold: int,
        exact_gate_row_ids: Sequence[int],
        views: _PreparedViews,
        parent_identity: Mapping[str, Any],
        cache_manifest_path: Path | str,
        cache_manifest_sha256: str,
    ) -> None:
        self.outer_fold = _positive_int(outer_fold, name="outer_fold")
        self.exact_gate_row_ids = _integer_rows(exact_gate_row_ids, name="exact_gate_row_ids")
        self._views = views
        self._parent_identity = _closed_json(parent_identity, path="parent_identity")
        manifest_path = Path(cache_manifest_path).resolve(strict=True)
        if manifest_path.name != "manifest.json" or not manifest_path.is_file():
            raise ValueError("bound upstream cache manifest path is not canonical")
        manifest_sha256 = str(cache_manifest_sha256).strip().lower()
        if len(manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in manifest_sha256
        ):
            raise ValueError("bound upstream cache manifest SHA-256 is malformed")
        if _sha256_file(manifest_path) != manifest_sha256:
            raise ValueError("bound upstream cache manifest failed authentication")
        self._cache_manifest_path = manifest_path
        self._cache_manifest_sha256 = manifest_sha256
        for view in (views.source, views.features):
            if view is not None and tuple(view.row_ids) != self.exact_gate_row_ids:
                raise ValueError("bound upstream view changed gate identity/order")

    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": "bound_context_fit_upstream_gate_provider_v6",
            "outer_fold": self.outer_fold,
            "gate_row_ids_sha256": _sha256_json(list(self.exact_gate_row_ids)),
            "parent_identity_sha256": _sha256_json(self._parent_identity),
            "cache_manifest_sha256": self._cache_manifest_sha256,
        }

    @property
    def authenticated_cache_manifest_path(self) -> Path:
        """Return the exact cache manifest after reauthenticating its bytes."""

        if (
            not self._cache_manifest_path.is_file()
            or _sha256_file(self._cache_manifest_path) != self._cache_manifest_sha256
        ):
            raise ValueError("bound upstream cache manifest changed after binding")
        return self._cache_manifest_path

    def _validate(self, outer_fold: int, exact_gate_row_ids: Sequence[int]) -> None:
        fold = _positive_int(outer_fold, name="outer_fold")
        rows = _integer_rows(exact_gate_row_ids, name="exact_gate_row_ids")
        if fold != self.outer_fold or rows != self.exact_gate_row_ids:
            raise ValueError("bound upstream provider requested for a different fold or gate")

    def get_gate_source_view(
        self, *, outer_fold: int, exact_gate_row_ids: Sequence[int]
    ) -> GateSourceSignalView:
        self._validate(outer_fold, exact_gate_row_ids)
        if self._views.source is None:
            raise RuntimeError("bound upstream provider has no calibrated sources")
        return self._views.source

    def get_gate_feature_bank_view(
        self, *, outer_fold: int, exact_gate_row_ids: Sequence[int]
    ) -> GateFeatureBankView:
        self._validate(outer_fold, exact_gate_row_ids)
        if self._views.features is None:
            raise RuntimeError("bound upstream provider has no role-aware feature banks")
        return self._views.features


__all__ = [
    "CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION",
    "CONTEXT_FIT_UPSTREAM_CALL_CHECKPOINT_SCHEMA_VERSION",
    "CONTEXT_FIT_UPSTREAM_PREDICTION_SCHEMA_VERSION",
    "CONTEXT_FIT_UPSTREAM_PROVIDER_ID",
    "BoundContextFitUpstreamGateProvider",
    "CompositeContextFitUpstreamBackend",
    "ContextFitUpstreamBackend",
    "ContextFitUpstreamGateProvider",
    "ContextFitUpstreamPrediction",
]
