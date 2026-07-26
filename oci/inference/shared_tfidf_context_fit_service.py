"""In-process reuse of exact-spent TF-IDF fits for gate/final transforms.

The adaptive reviewer needs two views of the same observable context:

* concept-only TF-IDF discovery before a proposal is frozen; and
* numerical TF-IDF transforms on a label-free gate or post-freeze final target.

``TfidfTopicOrphanSpentDiscoveryBackend`` and
``TfidfTopicOrphanContextBackend`` historically fit that complete context
independently.  The fitted TF-IDF implementation already persists everything
needed for the second operation: complete-context nuisance stacks, the common
vectorizer, and fitted topic banks.  This module snapshots the complete
authenticated JSON/NPY inventory before the spent provider removes its private
work directory and reuses it only when a later context call is an exact match
for the most recently registered spent context in the same process.

The service deliberately has no cache path or import API.  It cannot accept an
artifact from a prior process.  Non-matching contexts, including every ordinary
context-OOF subset, are delegated to the existing backend unchanged.
Gate treatment and outcome are absent from both the context-backend protocol
and this transform path.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import io
import json
import marshal
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .all_evidence_fusion import TFIDF_TOPIC_SOURCE
from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .context_fit_upstream_gate_provider import ContextFitUpstreamPrediction
from .review_spent_evidence_provider import SpentDiscoveryEvidence
from .tfidf_topic_discovery import FittedTopicContext
from .tfidf_safe_artifacts import (
    INDEX_FILENAME,
    load_fitted_topic_context,
)
from .tfidf_upstream_gate_backend import (
    TFIDF_CONTEXT_BACKEND_ID,
    TfidfOrphanFeatureCapacityOverflowError,
)

SHARED_TFIDF_FIT_SERVICE_ID = "in_memory_shared_tfidf_context_fit_service_v2"
SHARED_TFIDF_SPENT_BACKEND_ID = "shared_tfidf_spent_discovery_backend_v2"
SHARED_TFIDF_CONTEXT_BACKEND_ID = "shared_tfidf_context_gate_backend_v2"
SHARED_TFIDF_FIT_SNAPSHOT_SCHEMA = "shared_tfidf_fit_snapshot_v2"
SHARED_TFIDF_RUNTIME_GRAPH_ID = "shared_tfidf_context_fit_graph_v2"
UNWRAPPED_TFIDF_RUNTIME_GRAPH_ID = "unwrapped_tfidf_context_fit_graph_v1"

_REQUIRED_EFFECT_SCORE_COLUMNS = frozenset(
    {
        "feature",
        "eligible",
        "combined_importance",
        "support_control",
        "support_treated",
    }
)
_FORBIDDEN_IDENTITY_TOKENS = ("true", "oracle", "ground_truth")


def _optional_positive_capacity(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer or None")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer or None")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


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


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _module_for_object_sha256(value: Any) -> str:
    module = inspect.getmodule(type(value))
    path = None if module is None else getattr(module, "__file__", None)
    if path is None:
        raise TypeError("shared TF-IDF delegates must come from a file-backed module")
    return hashlib.sha256(Path(path).resolve().read_bytes()).hexdigest()


def _instance_dict(value: Any) -> Mapping[str, Any]:
    try:
        result = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        return {}
    return result if isinstance(result, Mapping) else {}


def _unwrap_runtime_method(value: Any) -> Any:
    if isinstance(value, (staticmethod, classmethod)):
        return value.__func__
    return value


@dataclass(frozen=True)
class _RuntimeMethodAttestation:
    owner: type[Any] = field(repr=False)
    method_names: tuple[str, ...]
    method_descriptors: tuple[Any, ...] = field(repr=False)
    module_sha256: str
    payload: Mapping[str, Any]

    @classmethod
    def capture(
        cls,
        value: Any,
        *,
        method_names: Sequence[str],
        path: str,
    ) -> "_RuntimeMethodAttestation":
        owner = type(value)
        names = tuple(str(name) for name in method_names)
        if not names or len(names) != len(set(names)) or any(not name for name in names):
            raise ValueError(f"{path} runtime method names must be unique and non-empty")
        descriptors: list[Any] = []
        method_hashes: dict[str, str] = {}
        for name in names:
            try:
                descriptor = inspect.getattr_static(owner, name)
            except AttributeError as exc:
                raise TypeError(f"{path} class must define {name}()") from exc
            implementation = _unwrap_runtime_method(descriptor)
            code = getattr(implementation, "__code__", None)
            if code is None:
                raise TypeError(f"{path}.{name}() must have an authenticated Python body")
            descriptors.append(descriptor)
            method_hashes[name] = hashlib.sha256(marshal.dumps(code)).hexdigest()
        module_sha = _module_for_object_sha256(value)
        payload = {
            "class_module": owner.__module__,
            "class_qualname": owner.__qualname__,
            "module_file_sha256": module_sha,
            "method_code_sha256": method_hashes,
            "per_instance_method_overrides_allowed": False,
        }
        result = cls(
            owner=owner,
            method_names=names,
            method_descriptors=tuple(descriptors),
            module_sha256=module_sha,
            payload=payload,
        )
        result.assert_stable(value, path=path)
        return result

    def assert_stable(self, value: Any, *, path: str) -> None:
        if type(value) is not self.owner:
            raise RuntimeError(f"{path} runtime class changed")
        overridden = sorted(set(self.method_names) & set(_instance_dict(value)))
        if overridden:
            raise RuntimeError(
                f"{path} has unauthenticated per-instance method overrides: {overridden}"
            )
        if _module_for_object_sha256(value) != self.module_sha256:
            raise RuntimeError(f"{path} runtime module changed")
        for name, expected in zip(self.method_names, self.method_descriptors):
            if inspect.getattr_static(self.owner, name) is not expected:
                raise RuntimeError(f"{path}.{name} runtime implementation changed")

    def call(self, value: Any, name: str, /, *args: Any, path: str, **kwargs: Any) -> Any:
        self.assert_stable(value, path=path)
        try:
            index = self.method_names.index(name)
        except ValueError as exc:
            raise RuntimeError(f"{path}.{name} is not an authenticated runtime method") from exc
        descriptor = self.method_descriptors[index]
        if isinstance(descriptor, staticmethod):
            return descriptor.__func__(*args, **kwargs)
        if isinstance(descriptor, classmethod):
            return descriptor.__func__(self.owner, *args, **kwargs)
        return descriptor(value, *args, **kwargs)

    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(dict(self.payload))


class _SealedRuntimeMethodInstances:
    _sealed_runtime_method_names: frozenset[str] = frozenset()

    def __setattr__(self, name: str, value: Any) -> None:
        if name in type(self)._sealed_runtime_method_names:
            raise AttributeError(f"{type(self).__qualname__}.{name} cannot be overridden")
        object.__setattr__(self, name, value)

    def __getattribute__(self, name: str) -> Any:
        sealed = object.__getattribute__(self, "_sealed_runtime_method_names")
        if name in sealed and name in _instance_dict(self):
            raise RuntimeError(
                f"{type(self).__qualname__}.{name} has an unauthenticated instance override"
            )
        return object.__getattribute__(self, name)


def _closed_identity(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_child in value.items():
            key = str(raw_key).strip()
            if not key or any(token in key.lower() for token in _FORBIDDEN_IDENTITY_TOKENS):
                raise ValueError(f"{path} contains a forbidden or empty identity field")
            if key in result:
                raise ValueError(f"{path} contains colliding identity fields")
            result[key] = _closed_identity(raw_child, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_closed_identity(child, path=f"{path}[]") for child in value]
    if isinstance(value, np.generic):
        return _closed_identity(value.item(), path=path)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if any(token in value.lower() for token in _FORBIDDEN_IDENTITY_TOKENS):
            raise ValueError(f"{path} contains a forbidden identity value")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite identity value")
        return value
    raise TypeError(f"{path} must contain closed JSON-compatible metadata")


def _delegate_identity(value: Any, *, path: str) -> dict[str, Any]:
    runtime = _RuntimeMethodAttestation.capture(
        value,
        method_names=("identity",),
        path=path,
    )
    closed = _closed_identity(
        runtime.call(value, "identity", path=path),
        path=f"{path}.identity",
    )
    if not isinstance(closed, dict):
        raise TypeError(f"{path}.identity() must return a mapping")
    return closed


def classify_tfidf_context_member_identity(value: Mapping[str, Any]) -> str:
    """Classify one authenticated composite TF-IDF member identity."""

    if not isinstance(value, Mapping):
        raise TypeError("TF-IDF context member identity must be a mapping")
    backend_id = value.get("backend")
    if backend_id == TFIDF_CONTEXT_BACKEND_ID:
        return UNWRAPPED_TFIDF_RUNTIME_GRAPH_ID
    if backend_id != SHARED_TFIDF_CONTEXT_BACKEND_ID:
        raise ValueError("context-fit runtime has no recognized TF-IDF backend member")
    delegate = value.get("delegate")
    service = value.get("service")
    if not isinstance(delegate, Mapping) or delegate.get("backend") != TFIDF_CONTEXT_BACKEND_ID:
        raise ValueError("shared TF-IDF member lacks its exact context-backend delegate")
    if not isinstance(service, Mapping) or service.get("service") != SHARED_TFIDF_FIT_SERVICE_ID:
        raise ValueError("shared TF-IDF member lacks its exact in-process fit service")
    return SHARED_TFIDF_RUNTIME_GRAPH_ID


def _fit_source_identity(value: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    """Project a context-backend identity onto fields that affect fitting.

    The two orphan limits are transform-time selection bounds in
    ``TfidfTopicOrphanContextBackend._orphan_values``.  They do not enter
    ``fit_tfidf_topic_context``.  Every other identity field remains exact.
    """

    closed = _closed_identity(value, path=path)
    if not isinstance(closed, dict):
        raise TypeError(f"{path} must be a mapping")
    result = copy.deepcopy(closed)
    for field_name in (
        "max_orphan_features",
        "minimum_orphan_arm_support",
        "minimum_orphan_arm_support_source",
    ):
        if field_name not in result:
            raise ValueError(f"{path} lacks required transform field {field_name}")
        result.pop(field_name)
    return result


def _positive_outer_fold(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("outer_fold must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError("outer_fold must be positive")
    return result


def _row_ids(values: Sequence[Any], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of row IDs")
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


def _texts(values: Sequence[Any], *, name: str, length: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if len(result) != int(length) or not all(isinstance(value, str) for value in result):
        raise ValueError(f"{name} must contain exactly {int(length)} strings")
    return result


def _finite_vector(values: Any, *, name: str, length: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) != int(length) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {int(length)}")
    return result.copy()


def _vector_sha256(values: np.ndarray) -> str:
    return _sha256_json([float(value).hex() for value in np.asarray(values, dtype=float)])


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _snapshot_bytes(path: Path, *, root: Path, label: str) -> bytes:
    root = root.resolve()
    path = (path if path.is_absolute() else root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escaped the private spent-fit work directory") from exc
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    before = _stat_signature(path)
    payload = path.read_bytes()
    after = _stat_signature(path)
    if before != after:
        raise RuntimeError(f"{label} changed while it was snapshotted")
    return payload


def _parse_json_bytes(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _safe_bundle_sha256(payload: Sequence[tuple[str, bytes]]) -> str:
    return _sha256_json(
        [
            {
                "relative_path": relative,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for relative, raw in payload
        ]
    )


def _snapshot_fitted_context(
    index_path: Path,
    *,
    root: Path,
) -> tuple[tuple[str, bytes], ...]:
    """Authenticate and snapshot every byte in a safe fitted-context artifact."""

    fitted = load_fitted_topic_context(index_path)
    if not isinstance(fitted, FittedTopicContext):
        raise TypeError("spent TF-IDF artifact did not contain FittedTopicContext")
    artifact_root = index_path.parent
    try:
        artifact_root.relative_to(root)
    except ValueError as exc:
        raise ValueError("spent fitted TF-IDF context escaped the private work directory") from exc
    paths = sorted(artifact_root.iterdir(), key=lambda path: path.name)
    if not paths or INDEX_FILENAME not in {path.name for path in paths}:
        raise ValueError("spent fitted TF-IDF context has no safe artifact index")
    return tuple(
        (
            path.name,
            _snapshot_bytes(
                path,
                root=artifact_root,
                label=f"spent fitted TF-IDF payload {path.name}",
            ),
        )
        for path in paths
    )


def _load_fitted_context(payload: Sequence[tuple[str, bytes]]) -> FittedTopicContext:
    if not payload or [name for name, _raw in payload] != sorted(name for name, _raw in payload):
        raise ValueError("current-process fitted TF-IDF inventory is empty or reordered")
    with tempfile.TemporaryDirectory(prefix="oci-shared-tfidf-safe-") as temporary:
        root = Path(temporary) / "fitted_context"
        root.mkdir()
        for relative, raw in payload:
            if relative != Path(relative).name or not isinstance(raw, bytes):
                raise ValueError("current-process fitted TF-IDF inventory is invalid")
            (root / relative).write_bytes(raw)
        try:
            fitted = load_fitted_topic_context(root / INDEX_FILENAME)
        except Exception as exc:
            raise ValueError("current-process fitted TF-IDF snapshot is unreadable") from exc
    if not isinstance(fitted, FittedTopicContext):
        raise TypeError("spent TF-IDF artifact did not contain FittedTopicContext")
    if not callable(getattr(fitted, "transform_topics", None)):
        raise TypeError("spent fitted context lacks transform_topics()")
    if not callable(getattr(fitted.treatment_stack, "predict", None)) or not callable(
        getattr(fitted.outcome_stack, "predict", None)
    ):
        raise TypeError("spent fitted context lacks complete nuisance-stack predictors")
    vectorizer = getattr(fitted, "common_vectorizer", None)
    if vectorizer is None or not callable(getattr(vectorizer, "transform", None)):
        raise TypeError("spent fitted context lacks its common vectorizer")
    if not isinstance(getattr(vectorizer, "vocabulary_", None), Mapping):
        raise TypeError("spent fitted context lacks a fitted common vocabulary")
    if set(fitted.topic_banks) != {"treatment", "outcome", "effect"}:
        raise ValueError("spent fitted context must contain all three topic banks")
    return fitted


def _effect_scores(payload: bytes) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        raise ValueError("current-process effect n-gram snapshot is unreadable") from exc
    if not _REQUIRED_EFFECT_SCORE_COLUMNS <= set(frame.columns):
        missing = sorted(_REQUIRED_EFFECT_SCORE_COLUMNS - set(frame.columns))
        raise ValueError(f"effect n-gram snapshot is missing required fields: {missing}")
    if len(frame) < 1:
        raise ValueError("effect n-gram snapshot cannot be empty")
    return frame


def _represented_topic_terms(metadata: Mapping[str, Any], bank: str) -> frozenset[str]:
    terms: set[str] = set()
    bank_metadata = (metadata.get("topic_banks") or {}).get(bank) or {}
    for topic in bank_metadata.get("topics") or ():
        if not isinstance(topic, Mapping):
            continue
        for raw in topic.get("terms") or ():
            value = raw.get("term") or raw.get("feature") if isinstance(raw, Mapping) else raw
            text = str(value or "").strip()
            if text:
                terms.add(text)
    return frozenset(terms)


@dataclass(frozen=True)
class _CurrentProcessFitSnapshot:
    schema_version: str
    fit_key: str
    outer_fold: int
    context_row_ids: tuple[int, ...]
    context_text_sha256: str
    treatment_sha256: str
    outcome_sha256: str
    metadata_json: bytes = field(repr=False)
    metadata_sha256: str
    fitted_context_files: tuple[tuple[str, bytes], ...] = field(repr=False)
    fitted_context_sha256: str
    effect_scores_parquet: bytes = field(repr=False)
    effect_scores_sha256: str


class InMemorySharedTfidfContextFitService(_SealedRuntimeMethodInstances):
    """Register current-process spent fits and transform exact active matches."""

    _sealed_runtime_method_names = frozenset(
        {
            "_assert_runtime_stable",
            "_fit_binding",
            "_normalized_fit_inputs",
            "_orphan_values",
            "_verify_snapshot",
            "assert_source_identity",
            "identity",
            "register_spent_fit",
            "transform_active_exact",
        }
    )

    def __init__(self, *, source_backend_identity: Mapping[str, Any]) -> None:
        self._runtime_methods = _RuntimeMethodAttestation.capture(
            self,
            method_names=tuple(sorted(self._sealed_runtime_method_names)),
            path="shared_tfidf_service",
        )
        closed = _fit_source_identity(
            source_backend_identity,
            path="source_backend_identity",
        )
        self._source_backend_identity = closed
        self._service_code_sha256 = _module_sha256()
        self._identity = {
            "service": SHARED_TFIDF_FIT_SERVICE_ID,
            "snapshot_schema": SHARED_TFIDF_FIT_SNAPSHOT_SCHEMA,
            "service_code_sha256": self._service_code_sha256,
            "service_runtime_attestation": self._runtime_methods.identity(),
            "source_backend_identity": copy.deepcopy(closed),
            "fit_identity_excludes_transform_only_fields": [
                "max_orphan_features",
                "minimum_orphan_arm_support",
                "minimum_orphan_arm_support_source",
            ],
            "storage": "private_current_process_memory_only",
            "cross_run_artifact_or_joblib_acceptance": False,
            "fitted_context_serialization": "closed_json_and_per_array_npy_v1",
            "cache_path_or_import_api_exposed": False,
            "fit_binding": "exact_ordered_rows_text_treatment_outcome_and_outer_fold_v1",
            "reuse_scope": "most_recent_exact_spent_context_per_outer_fold_v1",
            "gate_labels_accepted": False,
        }
        self._snapshots: dict[str, _CurrentProcessFitSnapshot] = {}
        self._active_key_by_outer_fold: dict[int, str] = {}
        self._reuse_count = 0

    def _assert_runtime_stable(self) -> None:
        self._runtime_methods.assert_stable(self, path="shared_tfidf_service")
        if _module_sha256() != self._service_code_sha256:
            raise RuntimeError("shared TF-IDF service code changed after construction")

    def identity(self) -> Mapping[str, Any]:
        self._assert_runtime_stable()
        return copy.deepcopy(self._identity)

    @property
    def registered_fit_count(self) -> int:
        return len(self._snapshots)

    @property
    def reuse_count(self) -> int:
        return int(self._reuse_count)

    def assert_source_identity(self, value: Mapping[str, Any]) -> None:
        self._assert_runtime_stable()
        current = _fit_source_identity(value, path="source_backend.identity")
        if current != self._source_backend_identity:
            raise ValueError("shared TF-IDF source backend identity is incompatible")

    def _fit_binding(
        self,
        *,
        outer_fold: int,
        context_row_ids: tuple[int, ...],
        context_texts: tuple[str, ...],
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "schema_version": SHARED_TFIDF_FIT_SNAPSHOT_SCHEMA,
            "service_identity_sha256": _sha256_json(self._identity),
            "source_backend_identity_sha256": _sha256_json(self._source_backend_identity),
            "outer_fold": int(outer_fold),
            "ordered_context_row_ids": list(context_row_ids),
            "ordered_context_text_sha256": _sha256_json(list(context_texts)),
            "context_treatment_sha256": _vector_sha256(treatment),
            "context_outcome_sha256": _vector_sha256(outcome),
            "fit_uses_gate_text_or_labels": False,
        }

    def _normalized_fit_inputs(
        self,
        *,
        outer_fold: Any,
        context_row_ids: Sequence[Any],
        context_texts: Sequence[Any],
        context_treatment: Any,
        context_outcome: Any,
    ) -> tuple[int, tuple[int, ...], tuple[str, ...], np.ndarray, np.ndarray, str]:
        fold = _positive_outer_fold(outer_fold)
        rows = _row_ids(context_row_ids, name="context_row_ids")
        texts = _texts(context_texts, name="context_texts", length=len(rows))
        treatment = _finite_vector(
            context_treatment,
            name="context_treatment",
            length=len(rows),
        )
        outcome = _finite_vector(context_outcome, name="context_outcome", length=len(rows))
        if not set(np.unique(treatment)).issubset({0.0, 1.0}):
            raise ValueError("context_treatment must be binary")
        binding = self._fit_binding(
            outer_fold=fold,
            context_row_ids=rows,
            context_texts=texts,
            treatment=treatment,
            outcome=outcome,
        )
        return fold, rows, texts, treatment, outcome, _sha256_json(binding)

    def register_spent_fit(
        self,
        *,
        outer_fold: int,
        context_row_ids: Sequence[int],
        context_texts: Sequence[str],
        context_treatment: Any,
        context_outcome: Any,
        artifact_dir: Path | str,
    ) -> str:
        """Snapshot one fit produced moments earlier by the wrapped spent backend."""

        self._assert_runtime_stable()
        fold, rows, texts, treatment, outcome, fit_key = self._normalized_fit_inputs(
            outer_fold=outer_fold,
            context_row_ids=context_row_ids,
            context_texts=context_texts,
            context_treatment=context_treatment,
            context_outcome=context_outcome,
        )
        root = Path(artifact_dir).resolve()
        metadata_bytes = _snapshot_bytes(
            root / "context_metadata.json",
            root=root,
            label="spent TF-IDF context metadata",
        )
        metadata = _parse_json_bytes(metadata_bytes, label="spent TF-IDF context metadata")
        if tuple(metadata.get("fit_row_ids") or ()) != rows:
            raise ValueError("spent TF-IDF metadata changed the exact fit row identity/order")
        heldout_rows = tuple(metadata.get("heldout_row_ids") or ())
        if not heldout_rows or not set(map(int, heldout_rows)) <= set(rows):
            raise ValueError("spent TF-IDF registration contains a non-spent heldout row")
        artifacts = metadata.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("spent TF-IDF metadata has no artifact registry")
        raw_scores = artifacts.get("ngram_scores")
        if not isinstance(raw_scores, Mapping):
            raise ValueError("spent TF-IDF metadata has no n-gram score registry")
        fitted_path = Path(str(artifacts.get("fitted_context") or ""))
        effect_path = Path(str(raw_scores.get("effect") or ""))
        fitted_files = _snapshot_fitted_context(
            fitted_path,
            root=root,
        )
        effect_bytes = _snapshot_bytes(
            effect_path,
            root=root,
            label="spent effect n-gram scores",
        )
        fitted = _load_fitted_context(fitted_files)
        _effect_scores(effect_bytes)
        if str(metadata.get("config_hash") or "") != str(fitted.config_hash):
            raise ValueError("spent fitted TF-IDF config hash disagrees with metadata")
        if set((metadata.get("topic_banks") or {})) != {"treatment", "outcome", "effect"}:
            raise ValueError("spent TF-IDF metadata must describe all three topic banks")

        snapshot = _CurrentProcessFitSnapshot(
            schema_version=SHARED_TFIDF_FIT_SNAPSHOT_SCHEMA,
            fit_key=fit_key,
            outer_fold=fold,
            context_row_ids=rows,
            context_text_sha256=_sha256_json(list(texts)),
            treatment_sha256=_vector_sha256(treatment),
            outcome_sha256=_vector_sha256(outcome),
            metadata_json=metadata_bytes,
            metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
            fitted_context_files=fitted_files,
            fitted_context_sha256=_safe_bundle_sha256(fitted_files),
            effect_scores_parquet=effect_bytes,
            effect_scores_sha256=hashlib.sha256(effect_bytes).hexdigest(),
        )
        prior = self._snapshots.get(fit_key)
        if prior is not None and prior != snapshot:
            raise RuntimeError("the same exact spent TF-IDF fit key produced different bytes")
        self._snapshots[fit_key] = snapshot
        self._active_key_by_outer_fold[fold] = fit_key
        return fit_key

    @staticmethod
    def _verify_snapshot(snapshot: _CurrentProcessFitSnapshot) -> None:
        if snapshot.schema_version != SHARED_TFIDF_FIT_SNAPSHOT_SCHEMA:
            raise RuntimeError("shared TF-IDF snapshot schema changed in memory")
        for payload, expected, label in (
            (snapshot.metadata_json, snapshot.metadata_sha256, "metadata"),
            (
                snapshot.fitted_context_files,
                snapshot.fitted_context_sha256,
                "fitted context",
            ),
            (
                snapshot.effect_scores_parquet,
                snapshot.effect_scores_sha256,
                "effect scores",
            ),
        ):
            observed = (
                _safe_bundle_sha256(payload)
                if label == "fitted context"
                else hashlib.sha256(payload).hexdigest()
            )
            if observed != expected:
                raise RuntimeError(f"shared TF-IDF {label} bytes changed in memory")

    @staticmethod
    def _orphan_values(
        *,
        fitted: FittedTopicContext,
        metadata: Mapping[str, Any],
        scores: pd.DataFrame,
        gate_texts: tuple[str, ...],
        max_orphan_features: int | None,
        minimum_orphan_arm_support: int,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        represented = _represented_topic_terms(metadata, "effect")
        candidates = scores.loc[
            scores["eligible"].fillna(False).astype(bool)
            & (scores["support_control"] >= int(minimum_orphan_arm_support))
            & (scores["support_treated"] >= int(minimum_orphan_arm_support))
            & ~scores["feature"].astype(str).isin(represented)
        ].copy()
        candidates["_absolute_importance"] = pd.to_numeric(
            candidates["combined_importance"], errors="coerce"
        ).abs()
        candidates = candidates.sort_values(
            ["_absolute_importance", "feature"],
            ascending=[False, True],
        )
        vectorizer = fitted.common_vectorizer
        selected_terms: list[str] = []
        selected_term_set: set[str] = set()
        selected_columns: list[int] = []
        for term in candidates["feature"].astype(str):
            column = vectorizer.vocabulary_.get(term)
            if column is None or term in selected_term_set:
                continue
            selected_terms.append(term)
            selected_term_set.add(term)
            selected_columns.append(int(column))
        if not selected_terms:
            raise RuntimeError("context-fitted TF-IDF model produced no eligible orphan n-grams")
        if max_orphan_features is not None and len(selected_terms) > max_orphan_features:
            raise TfidfOrphanFeatureCapacityOverflowError(
                "context-fitted TF-IDF model produced "
                f"{len(selected_terms)} eligible orphan n-grams, exceeding "
                f"max_orphan_features={max_orphan_features}; refusing silent "
                "orphan-feature omission"
            )
        matrix = vectorizer.transform(list(gate_texts))[:, selected_columns]
        values = np.asarray(matrix.toarray(), dtype=float)
        names = tuple(
            f"tfidf_orphan_{index:03d}_{hashlib.sha256(term.encode('utf-8')).hexdigest()[:12]}"
            for index, term in enumerate(selected_terms, start=1)
        )
        return names, values

    def transform_active_exact(
        self,
        *,
        outer_fold: int,
        context_row_ids: Sequence[int],
        context_texts: Sequence[str],
        context_treatment: Any,
        context_outcome: Any,
        gate_row_ids: Sequence[int],
        gate_texts: Sequence[str],
        max_orphan_features: int | None,
        minimum_orphan_arm_support: int,
    ) -> ContextFitUpstreamPrediction | None:
        """Return a transform-only hit, or ``None`` for mandatory delegation."""

        self._assert_runtime_stable()
        fold, rows, _texts_value, _treatment, _outcome, fit_key = self._normalized_fit_inputs(
            outer_fold=outer_fold,
            context_row_ids=context_row_ids,
            context_texts=context_texts,
            context_treatment=context_treatment,
            context_outcome=context_outcome,
        )
        if self._active_key_by_outer_fold.get(fold) != fit_key:
            return None
        snapshot = self._snapshots.get(fit_key)
        if snapshot is None or snapshot.context_row_ids != rows:
            raise RuntimeError("active shared TF-IDF fit has no exact in-memory snapshot")
        gate_rows = _row_ids(gate_row_ids, name="gate_row_ids")
        if set(rows) & set(gate_rows):
            raise ValueError("shared TF-IDF context and gate rows must be disjoint")
        exact_gate_texts = _texts(gate_texts, name="gate_texts", length=len(gate_rows))
        orphan_capacity = _optional_positive_capacity(
            max_orphan_features,
            name="max_orphan_features",
        )
        orphan_arm_support = _positive_integer(
            minimum_orphan_arm_support,
            name="minimum_orphan_arm_support",
        )
        self._verify_snapshot(snapshot)
        metadata = _parse_json_bytes(
            snapshot.metadata_json,
            label="in-memory spent TF-IDF context metadata",
        )
        fitted = _load_fitted_context(snapshot.fitted_context_files)
        scores = _effect_scores(snapshot.effect_scores_parquet)

        topics = fitted.transform_topics(exact_gate_texts)
        if set(topics) != {"treatment", "outcome", "effect"}:
            raise RuntimeError("shared TF-IDF transform did not produce all three topic banks")
        treatment_nuisance, _ = fitted.treatment_stack.predict(exact_gate_texts)
        outcome_nuisance, _ = fitted.outcome_stack.predict(exact_gate_texts)
        names: list[str] = ["tfidf_nuisance_treatment", "tfidf_nuisance_outcome"]
        kinds: list[str] = ["tfidf_topics", "tfidf_topics"]
        roles: list[str] = [
            PROPENSITY_NUISANCE_FEATURE_ROLE,
            OUTCOME_NUISANCE_FEATURE_ROLE,
        ]
        columns: list[np.ndarray] = [
            np.asarray(treatment_nuisance, dtype=float),
            np.asarray(outcome_nuisance, dtype=float),
        ]
        role_by_bank = {
            "treatment": PROPENSITY_NUISANCE_FEATURE_ROLE,
            "outcome": OUTCOME_NUISANCE_FEATURE_ROLE,
            "effect": UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        }
        kind_by_bank = {
            "treatment": "tfidf_topics",
            "outcome": "tfidf_topics",
            "effect": "tfidf_topic_contrast",
        }
        for bank in ("treatment", "outcome", "effect"):
            values = np.asarray(topics[bank], dtype=float)
            if values.ndim != 2 or values.shape[0] != len(gate_rows) or values.shape[1] < 1:
                raise ValueError(f"shared TF-IDF {bank} topic bank has an invalid shape")
            for column in range(values.shape[1]):
                names.append(f"tfidf_{bank}_topic_{column + 1:03d}")
                kinds.append(kind_by_bank[bank])
                roles.append(role_by_bank[bank])
                columns.append(values[:, column])

        orphan_names, orphan_values = self._orphan_values(
            fitted=fitted,
            metadata=metadata,
            scores=scores,
            gate_texts=exact_gate_texts,
            max_orphan_features=orphan_capacity,
            minimum_orphan_arm_support=orphan_arm_support,
        )
        for column, name in enumerate(orphan_names):
            names.append(name)
            kinds.append("tfidf_orphan_ngrams")
            roles.append(UNCALIBRATED_EFFECT_MODIFIER_ROLE)
            columns.append(orphan_values[:, column])
        prediction = ContextFitUpstreamPrediction(
            gate_row_ids=gate_rows,
            calibrated_source_names=(),
            calibrated_source_kinds=(),
            calibrated_source_values=np.empty((len(gate_rows), 0), dtype=float),
            feature_names=tuple(names),
            feature_kinds=tuple(kinds),
            feature_roles=tuple(roles),
            feature_values=np.column_stack(columns),
        )
        self._reuse_count += 1
        return prediction


class SharedTfidfSpentDiscoveryBackend(_SealedRuntimeMethodInstances):
    """Register artifacts produced by an otherwise unchanged spent backend."""

    _sealed_runtime_method_names = frozenset({"_assert_stable", "fit_discovery", "identity"})

    def __init__(self, *, backend: Any, service: InMemorySharedTfidfContextFitService) -> None:
        if type(service) is not InMemorySharedTfidfContextFitService:
            raise TypeError("service must be the exact InMemorySharedTfidfContextFitService")
        source = getattr(backend, "source", None)
        if source is None:
            raise TypeError("spent TF-IDF backend must expose its context-fit source")
        self.backend = backend
        self.service = service
        self._self_runtime = _RuntimeMethodAttestation.capture(
            self,
            method_names=tuple(sorted(self._sealed_runtime_method_names)),
            path="shared_spent_tfidf_wrapper",
        )
        self._delegate_runtime = _RuntimeMethodAttestation.capture(
            backend,
            method_names=("identity", "fit_discovery"),
            path="spent_backend",
        )
        self._source_runtime = _RuntimeMethodAttestation.capture(
            source,
            method_names=("identity", "fit_predict"),
            path="spent_backend.source",
        )
        self._service_runtime = _RuntimeMethodAttestation.capture(
            service,
            method_names=(
                "identity",
                "assert_source_identity",
                "register_spent_fit",
                "transform_active_exact",
            ),
            path="shared_tfidf_service",
        )
        self._wrapper_code_sha256 = _module_sha256()
        self._delegate_module_sha256 = _module_for_object_sha256(backend)
        self._source_module_sha256 = _module_for_object_sha256(source)
        self._delegate_identity = _closed_identity(
            self._delegate_runtime.call(backend, "identity", path="spent_backend"),
            path="spent_backend.identity",
        )
        self._source_identity = _closed_identity(
            self._source_runtime.call(source, "identity", path="spent_backend.source"),
            path="spent_backend.source.identity",
        )
        if not isinstance(self._delegate_identity, dict) or not isinstance(
            self._source_identity, dict
        ):
            raise TypeError("spent TF-IDF delegate identities must be mappings")
        self._service_runtime.call(
            service,
            "assert_source_identity",
            self._source_identity,
            path="shared_tfidf_service",
        )
        self._identity = {
            "backend": SHARED_TFIDF_SPENT_BACKEND_ID,
            "wrapper_code_sha256": self._wrapper_code_sha256,
            "delegate_module_sha256": self._delegate_module_sha256,
            "fit_source_module_sha256": self._source_module_sha256,
            "wrapper_runtime_attestation": self._self_runtime.identity(),
            "delegate_runtime_attestation": self._delegate_runtime.identity(),
            "fit_source_runtime_attestation": self._source_runtime.identity(),
            "service_runtime_attestation": self._service_runtime.identity(),
            "delegate": copy.deepcopy(self._delegate_identity),
            "fit_source": copy.deepcopy(self._source_identity),
            "service": self._service_runtime.call(
                service,
                "identity",
                path="shared_tfidf_service",
            ),
            "registration_timing": "after_fit_before_spent_evidence_return_v1",
            "future_gate_text_or_labels_accepted": False,
            "cross_run_artifact_acceptance": False,
        }

    def _assert_stable(self) -> None:
        self._self_runtime.assert_stable(self, path="shared_spent_tfidf_wrapper")
        self._delegate_runtime.assert_stable(self.backend, path="spent_backend")
        self._source_runtime.assert_stable(
            self.backend.source,
            path="spent_backend.source",
        )
        self._service_runtime.assert_stable(
            self.service,
            path="shared_tfidf_service",
        )
        if _module_sha256() != self._wrapper_code_sha256:
            raise RuntimeError("shared spent TF-IDF wrapper code changed")
        if _module_for_object_sha256(self.backend) != self._delegate_module_sha256:
            raise RuntimeError("wrapped spent TF-IDF backend module changed")
        if _module_for_object_sha256(self.backend.source) != self._source_module_sha256:
            raise RuntimeError("wrapped spent TF-IDF fit source module changed")
        current_delegate = _closed_identity(
            self._delegate_runtime.call(
                self.backend,
                "identity",
                path="spent_backend",
            ),
            path="spent_backend.identity",
        )
        if current_delegate != self._delegate_identity:
            raise RuntimeError("wrapped spent TF-IDF backend identity changed")
        current_source = _closed_identity(
            self._source_runtime.call(
                self.backend.source,
                "identity",
                path="spent_backend.source",
            ),
            path="spent_backend.source.identity",
        )
        if current_source != self._source_identity:
            raise RuntimeError("wrapped spent TF-IDF fit source identity changed")
        self._service_runtime.call(
            self.service,
            "assert_source_identity",
            current_source,
            path="shared_tfidf_service",
        )

    def identity(self) -> Mapping[str, Any]:
        self._assert_stable()
        return copy.deepcopy(self._identity)

    def fit_discovery(self, **kwargs: Any) -> SpentDiscoveryEvidence:
        self._assert_stable()
        required = {
            "outer_fold",
            "review_round",
            "exact_spent_row_ids",
            "spent_texts",
            "spent_treatment",
            "spent_outcome",
            "work_dir",
        }
        if set(kwargs) != required:
            raise TypeError("shared spent TF-IDF backend received an unsupported call schema")
        result = self._delegate_runtime.call(
            self.backend,
            "fit_discovery",
            path="spent_backend",
            **kwargs,
        )
        self._assert_stable()
        if not isinstance(result, SpentDiscoveryEvidence):
            raise TypeError("wrapped spent TF-IDF backend returned the wrong result type")
        if result.source_kind != TFIDF_TOPIC_SOURCE:
            raise ValueError("wrapped spent TF-IDF backend changed its source family")
        self._service_runtime.call(
            self.service,
            "register_spent_fit",
            path="shared_tfidf_service",
            outer_fold=kwargs["outer_fold"],
            context_row_ids=kwargs["exact_spent_row_ids"],
            context_texts=kwargs["spent_texts"],
            context_treatment=kwargs["spent_treatment"],
            context_outcome=kwargs["spent_outcome"],
            artifact_dir=kwargs["work_dir"],
        )
        return result


class SharedTfidfContextBackend(_SealedRuntimeMethodInstances):
    """Transform exact active spent fits; delegate every other context call."""

    _sealed_runtime_method_names = frozenset({"_assert_stable", "fit_predict", "identity"})

    def __init__(self, *, backend: Any, service: InMemorySharedTfidfContextFitService) -> None:
        if type(service) is not InMemorySharedTfidfContextFitService:
            raise TypeError("service must be the exact InMemorySharedTfidfContextFitService")
        for name in ("max_orphan_features", "minimum_orphan_arm_support"):
            if not hasattr(backend, name):
                raise TypeError(f"context backend must expose {name}")
        self.backend = backend
        # FinalContextFitUpstreamProducer recursively attests objects exposed
        # through ``backends``.  Keep the delegate visible instead of relying
        # only on the wrapper's semantic identity.
        self.backends = (backend,)
        self.service = service
        self._self_runtime = _RuntimeMethodAttestation.capture(
            self,
            method_names=tuple(sorted(self._sealed_runtime_method_names)),
            path="shared_context_tfidf_wrapper",
        )
        self._delegate_runtime = _RuntimeMethodAttestation.capture(
            backend,
            method_names=("identity", "fit_predict"),
            path="context_backend",
        )
        self._service_runtime = _RuntimeMethodAttestation.capture(
            service,
            method_names=(
                "identity",
                "assert_source_identity",
                "register_spent_fit",
                "transform_active_exact",
            ),
            path="shared_tfidf_service",
        )
        self._wrapper_code_sha256 = _module_sha256()
        self._delegate_module_sha256 = _module_for_object_sha256(backend)
        self._delegate_identity = _closed_identity(
            self._delegate_runtime.call(backend, "identity", path="context_backend"),
            path="context_backend.identity",
        )
        if not isinstance(self._delegate_identity, dict):
            raise TypeError("context TF-IDF delegate identity must be a mapping")
        self._service_runtime.call(
            service,
            "assert_source_identity",
            self._delegate_identity,
            path="shared_tfidf_service",
        )
        self._identity = {
            "backend": SHARED_TFIDF_CONTEXT_BACKEND_ID,
            "wrapper_code_sha256": self._wrapper_code_sha256,
            "delegate_module_sha256": self._delegate_module_sha256,
            "wrapper_runtime_attestation": self._self_runtime.identity(),
            "delegate_runtime_attestation": self._delegate_runtime.identity(),
            "service_runtime_attestation": self._service_runtime.identity(),
            "delegate": copy.deepcopy(self._delegate_identity),
            "service": self._service_runtime.call(
                service,
                "identity",
                path="shared_tfidf_service",
            ),
            "reuse_condition": "most_recent_exact_spent_fit_key_match_only_v1",
            "non_exact_and_subset_calls": "delegate_unchanged_v1",
            "gate_transform": "label_free_fitted_context_transform_only_v1",
            "gate_labels_accepted": False,
            "cross_run_artifact_acceptance": False,
        }

    def _assert_stable(self) -> None:
        self._self_runtime.assert_stable(self, path="shared_context_tfidf_wrapper")
        self._delegate_runtime.assert_stable(self.backend, path="context_backend")
        self._service_runtime.assert_stable(
            self.service,
            path="shared_tfidf_service",
        )
        if self.backends != (self.backend,):
            raise RuntimeError("shared context TF-IDF recursive delegate changed")
        if _module_sha256() != self._wrapper_code_sha256:
            raise RuntimeError("shared context TF-IDF wrapper code changed")
        if _module_for_object_sha256(self.backend) != self._delegate_module_sha256:
            raise RuntimeError("wrapped context TF-IDF backend module changed")
        current = _closed_identity(
            self._delegate_runtime.call(
                self.backend,
                "identity",
                path="context_backend",
            ),
            path="context_backend.identity",
        )
        if current != self._delegate_identity:
            raise RuntimeError("wrapped context TF-IDF backend identity changed")
        self._service_runtime.call(
            self.service,
            "assert_source_identity",
            current,
            path="shared_tfidf_service",
        )

    def identity(self) -> Mapping[str, Any]:
        self._assert_stable()
        return copy.deepcopy(self._identity)

    def fit_predict(self, **kwargs: Any) -> ContextFitUpstreamPrediction:
        self._assert_stable()
        required = {
            "outer_fold",
            "context_row_ids",
            "context_texts",
            "context_treatment",
            "context_outcome",
            "gate_row_ids",
            "gate_texts",
            "work_dir",
        }
        if set(kwargs) != required:
            raise TypeError("shared context TF-IDF backend received an unsupported call schema")
        reused = self._service_runtime.call(
            self.service,
            "transform_active_exact",
            path="shared_tfidf_service",
            outer_fold=kwargs["outer_fold"],
            context_row_ids=kwargs["context_row_ids"],
            context_texts=kwargs["context_texts"],
            context_treatment=kwargs["context_treatment"],
            context_outcome=kwargs["context_outcome"],
            gate_row_ids=kwargs["gate_row_ids"],
            gate_texts=kwargs["gate_texts"],
            max_orphan_features=self.backend.max_orphan_features,
            minimum_orphan_arm_support=int(self.backend.minimum_orphan_arm_support),
        )
        if reused is not None:
            self._assert_stable()
            return reused
        prediction = self._delegate_runtime.call(
            self.backend,
            "fit_predict",
            path="context_backend",
            **kwargs,
        )
        self._assert_stable()
        if not isinstance(prediction, ContextFitUpstreamPrediction):
            raise TypeError("wrapped context TF-IDF backend returned the wrong result type")
        return prediction


@dataclass(frozen=True)
class SharedTfidfContextFitBackends:
    """One service and the two wrappers that share its exact fitted state."""

    service: InMemorySharedTfidfContextFitService
    spent_discovery_backend: SharedTfidfSpentDiscoveryBackend
    context_backend: SharedTfidfContextBackend

    def __post_init__(self) -> None:
        if not isinstance(self.service, InMemorySharedTfidfContextFitService):
            raise TypeError("service must be InMemorySharedTfidfContextFitService")
        if not isinstance(self.spent_discovery_backend, SharedTfidfSpentDiscoveryBackend):
            raise TypeError("spent_discovery_backend must be the authenticated shared wrapper")
        if not isinstance(self.context_backend, SharedTfidfContextBackend):
            raise TypeError("context_backend must be the authenticated shared wrapper")
        if (
            self.spent_discovery_backend.service is not self.service
            or self.context_backend.service is not self.service
        ):
            raise ValueError("both TF-IDF wrappers must use the exact same service instance")


def build_shared_tfidf_context_fit_backends(
    *,
    spent_discovery_backend: Any,
    context_backend: Any,
) -> SharedTfidfContextFitBackends:
    """Bind spent discovery and all later transforms to one private service.

    Construction is deliberately centralized so production callers cannot
    accidentally give the spent and gate/final paths different services.  The
    wrapper constructors authenticate the delegate code and identities, while
    the service rejects any fit-affecting mismatch between the spent backend's
    source and the context backend.
    """

    source_identity = _delegate_identity(context_backend, path="context_backend")
    service = InMemorySharedTfidfContextFitService(
        source_backend_identity=source_identity,
    )
    spent = SharedTfidfSpentDiscoveryBackend(
        backend=spent_discovery_backend,
        service=service,
    )
    context = SharedTfidfContextBackend(
        backend=context_backend,
        service=service,
    )
    return SharedTfidfContextFitBackends(
        service=service,
        spent_discovery_backend=spent,
        context_backend=context,
    )


__all__ = [
    "InMemorySharedTfidfContextFitService",
    "SHARED_TFIDF_CONTEXT_BACKEND_ID",
    "SHARED_TFIDF_FIT_SERVICE_ID",
    "SHARED_TFIDF_FIT_SNAPSHOT_SCHEMA",
    "SHARED_TFIDF_RUNTIME_GRAPH_ID",
    "SHARED_TFIDF_SPENT_BACKEND_ID",
    "SharedTfidfContextBackend",
    "SharedTfidfContextFitBackends",
    "SharedTfidfSpentDiscoveryBackend",
    "UNWRAPPED_TFIDF_RUNTIME_GRAPH_ID",
    "build_shared_tfidf_context_fit_backends",
    "classify_tfidf_context_member_identity",
]
