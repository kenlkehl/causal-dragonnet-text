"""Spent-only neural-query discovery and untouched-gate feature banks.

The ordinary outer-fold neural-query artifact is honest for a frozen final
prediction, but it is not recursively honest for a multi-round adaptive review:
an early OOF query can have been fitted on a later review gate.  This module
fits the same ungated three-bank query discovery on the rows already spent by
the reviewer, caches that fit by exact observable inputs, and then provides:

* concept-bearing neural-query evidence derived only from spent chunks; and
* label-free query activations for the current untouched gate.

The frozen embedding cache is the only semantic encoder used here.  No language
model is loaded or called, and no treatment/outcome value for a gate is accepted
by any public method.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from .all_evidence_fusion import (
    FoldEvidenceInput,
    FoldEvidenceProvenance,
    NEURAL_QUERY_MOMENTS,
    NEURAL_QUERY_SOURCE,
)
from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .context_fit_upstream_gate_provider import ContextFitUpstreamPrediction
from .fold_honest_r_stack import FitRowProvenance
from .neural_cohort_witness import soft_retrieval_activations
from .neural_query_agentic_forest import (
    NeuralQueryAgenticForestConfig,
    build_query_evidence,
)
from .neural_query_discovery_runtime import (
    NEURAL_QUERY_DISCOVERY_RUNTIME_ID,
    fit_in_memory_query_discovery,
)
from .stage1_upstream_gate_backend import (
    HistoricalStage1ConfigSnapshot,
    _historical_stage1_config_snapshot,
)
from .review_spent_evidence_provider import (
    BoundSpentFrozenChunkEmbeddingProvider,
    SpentDiscoveryEvidence,
    SpentOnlyFrozenChunkEmbeddingCache,
    _safe_concept_phrase,
)
from .tfidf_topic_discovery import (
    _strata,
    fit_joint_cross_fitted_nuisance_stacks,
)

NEURAL_QUERY_CONTEXT_SERVICE_ID = "neural_query_context_service_v7"
NEURAL_QUERY_CONTEXT_BACKEND_ID = "neural_query_context_gate_backend_v3"
NEURAL_QUERY_SPENT_EVIDENCE_PROVIDER_ID = "neural_query_spent_evidence_provider_v2"
NEURAL_QUERY_SPENT_DISCOVERY_BACKEND_ID = "neural_query_spent_discovery_backend_v2"
NEURAL_QUERY_SPENT_EVIDENCE_SCHEMA = "context_fit_neural_query_evidence_v2"
NEURAL_QUERY_OWNED_SNAPSHOT_SCHEMA = "context_fit_neural_query_owned_snapshot_v1"
NEURAL_QUERY_NUISANCE_OUTPUT_BINDING_SCHEMA = "context_fit_neural_query_nuisance_output_binding_v1"

_BANKS = ("treatment", "outcome", "effect")
_ROLE_BY_BANK = {
    "treatment": PROPENSITY_NUISANCE_FEATURE_ROLE,
    "outcome": OUTCOME_NUISANCE_FEATURE_ROLE,
    "effect": UNCALIBRATED_EFFECT_MODIFIER_ROLE,
}
_DISCOVERY_MANIFEST_SCHEMA = "context_fit_neural_query_cache_v6"
_DISCOVERY_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "cache_key",
        "binding",
        "checkpoint_file",
        "checkpoint_sha256",
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


def _array_content_sha256(value: np.ndarray) -> str:
    """Hash one closed numerical array including its dtype and exact shape."""

    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("neural-query snapshot arrays cannot contain Python objects")
    header = _canonical_json(
        {
            "dtype": array.dtype.str,
            "shape": [int(dimension) for dimension in array.shape],
        }
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _stable_json_file(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("neural-query snapshot metadata must be one regular file")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if before_identity != after_identity:
        raise RuntimeError("neural-query snapshot metadata changed while reading")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("neural-query snapshot metadata is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("neural-query snapshot metadata must be one JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _atomic_write_new_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace neural-query snapshot file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_new_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace neural-query snapshot file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_fresh_executable_cache_root(path: Path | str) -> Path:
    """Accept only a new/empty real directory as an executable cache root."""

    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ValueError("neural-query executable cache root cannot be a symlink")
    resolved = raw.resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError("neural-query executable cache root must be a directory")
        if any(resolved.iterdir()):
            raise ValueError(
                "neural-query executable cache root must be nonexistent or empty; "
                "pre-existing checkpoints are forbidden"
            )
    return resolved


def _json_state(value: Any) -> Any:
    """Detach mutable configuration state into deterministic closed JSON."""

    if is_dataclass(value) and not isinstance(value, type):
        return _json_state(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_state(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_state(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_state(value.tolist())
    if isinstance(value, np.generic):
        return _json_state(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("configuration state contains a non-finite float")
        return value
    if hasattr(value, "__dict__"):
        return _json_state(vars(value))
    raise TypeError(f"unsupported live configuration state: {type(value).__name__}")


def _float_hex_sha256(values: Sequence[float]) -> str:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("observable vectors must be finite and one-dimensional")
    return _sha256_json([float(value).hex() for value in vector])


def _integer_rows(
    values: Sequence[Any], *, name: str, allow_empty: bool = False
) -> tuple[int, ...]:
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
    if (not allow_empty and not rows) or len(rows) != len(set(rows)):
        raise ValueError(f"{name} must be {'unique' if allow_empty else 'non-empty and unique'}")
    return tuple(rows)


def _exact_texts(values: Sequence[Any], *, name: str, length: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if len(result) != int(length) or not all(isinstance(value, str) for value in result):
        raise ValueError(f"{name} must contain exactly {length} strings")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} cannot be negative")
    return result


def _validate_binary(values: np.ndarray, *, name: str, require_both: bool) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector")
    unique = set(np.unique(vector).tolist())
    allowed = {0.0, 1.0}
    if not unique <= allowed or (require_both and unique != allowed):
        raise ValueError(f"{name} must contain binary 0/1 values")
    return vector


def _query_discovery_runtime_code_sha256() -> str:
    from . import neural_query_discovery_runtime as runtime_module

    return _sha256_file(Path(runtime_module.__file__).resolve())


def _dependency_code_sha256s() -> dict[str, str]:
    from . import neural_cohort_witness as witness_module
    from . import neural_query_agentic_forest as evidence_module
    from . import review_spent_evidence_provider as spent_evidence_module
    from . import stage1_upstream_gate_backend as cache_adapter_module
    from . import tfidf_topic_discovery as nuisance_module

    return {
        "neural_cohort_witness": _sha256_file(Path(witness_module.__file__).resolve()),
        "neural_query_evidence": _sha256_file(Path(evidence_module.__file__).resolve()),
        "stage1_cache_adapter": _sha256_file(Path(cache_adapter_module.__file__).resolve()),
        "spent_evidence_policy": _sha256_file(Path(spent_evidence_module.__file__).resolve()),
        "tfidf_nuisance": _sha256_file(Path(nuisance_module.__file__).resolve()),
    }


def _safe_query_ngram_rows(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    output: list[dict[str, Any]] = []
    for raw in values:
        if not isinstance(raw, Mapping):
            continue
        term = _safe_concept_phrase(raw.get("term") or raw.get("feature") or raw.get("ngram"))
        if not term:
            continue
        row: dict[str, Any] = {"term": term}
        for key in (
            "tfidf_contrast",
            "loading",
            "signed_score",
            "fit_signed_score",
            "standardized_score",
            "rank",
            "fit_rank",
        ):
            value = raw.get(key)
            if isinstance(value, (bool, np.bool_)):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
                float(value)
            ):
                row[key] = float(value)
        output.append(row)
    return output


def _require_sha256(value: Any, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _owned_discovery_memory_sha256(discovery: Mapping[str, Any]) -> str:
    if not isinstance(discovery, Mapping):
        raise TypeError("owned neural-query discovery must be a mapping")
    return _sha256_json(_json_state(discovery))


def _owned_discovery_snapshot_parts(
    discovery: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Project trusted fitted state into numerical arrays and closed JSON metadata."""

    if not isinstance(discovery, Mapping):
        raise TypeError("owned neural-query discovery must be a mapping")
    banks = discovery.get("banks")
    if not isinstance(banks, Mapping) or set(banks) != set(_BANKS):
        raise ValueError("owned neural-query discovery must contain exactly three banks")
    if discovery.get("runtime") != NEURAL_QUERY_DISCOVERY_RUNTIME_ID:
        raise ValueError("owned neural-query discovery has the wrong fit runtime")
    if (
        discovery.get("all_queries_retained") is not True
        or discovery.get("validation_audits_used_for_selection") is not False
        or discovery.get("executable_checkpoint_io") is not False
    ):
        raise ValueError("owned neural-query discovery lacks its ungated in-memory attestations")
    _require_sha256(
        discovery.get("fit_input_binding_sha256"),
        name="neural-query fit input binding",
    )
    nuisance = discovery.get("fit_nuisance_output_binding")
    if not isinstance(nuisance, Mapping) or (
        nuisance.get("schema_version") != NEURAL_QUERY_NUISANCE_OUTPUT_BINDING_SCHEMA
        or nuisance.get("heldout_labels_accessed") is not False
    ):
        raise ValueError("owned neural-query discovery has no fitted nuisance-output binding")
    nuisance_rows = _integer_rows(
        nuisance.get("fit_row_ids") or (),
        name="fit_nuisance_output_binding.fit_row_ids",
    )
    _require_sha256(nuisance.get("fit_e_sha256"), name="fit_e_sha256")
    _require_sha256(nuisance.get("fit_m_sha256"), name="fit_m_sha256")

    arrays: dict[str, np.ndarray] = {}
    bank_metadata: dict[str, Any] = {}
    inventory: dict[str, Any] = {}
    query_count_by_bank: dict[str, int] = {}
    fit_row_count: int | None = None
    for bank in _BANKS:
        raw = banks[bank]
        if not isinstance(raw, Mapping):
            raise ValueError(f"owned neural-query {bank} bank is malformed")
        queries = np.asarray(raw.get("queries"), dtype=np.float32)
        train_activations = np.asarray(raw.get("train_activations"), dtype=np.float32)
        records = raw.get("records")
        if (
            queries.ndim != 2
            or queries.shape[0] < 1
            or queries.shape[1] < 1
            or not np.isfinite(queries).all()
            or train_activations.ndim != 2
            or train_activations.shape[0] < 1
            or train_activations.shape[1] != queries.shape[0]
            or not np.isfinite(train_activations).all()
            or not isinstance(records, list)
            or len(records) != queries.shape[0]
            or not isinstance(raw.get("consensus"), Mapping)
            or not isinstance(raw.get("objective"), str)
            or not str(raw.get("objective")).strip()
            or raw.get("all_queries_retained") is not True
            or raw.get("statistical_gate_applied") is not False
        ):
            raise ValueError(f"owned neural-query {bank} fit state is incomplete")
        if fit_row_count is None:
            fit_row_count = int(train_activations.shape[0])
        elif fit_row_count != int(train_activations.shape[0]):
            raise ValueError("owned neural-query bank activations have different fit row counts")
        query_count_by_bank[bank] = int(queries.shape[0])
        for suffix, array in (
            ("queries", queries),
            ("train_activations", train_activations),
        ):
            key = f"{bank}_{suffix}"
            arrays[key] = np.ascontiguousarray(array)
            inventory[key] = {
                "dtype": arrays[key].dtype.str,
                "shape": [int(dimension) for dimension in arrays[key].shape],
                "content_sha256": _array_content_sha256(arrays[key]),
            }
        bank_metadata[bank] = {
            str(key): _json_state(value)
            for key, value in raw.items()
            if str(key) not in {"queries", "train_activations"}
        }
    if fit_row_count != len(nuisance_rows):
        raise ValueError("fitted nuisance rows do not align with neural-query train activations")

    discovery_metadata = {
        str(key): _json_state(value) for key, value in discovery.items() if str(key) != "banks"
    }
    discovery_metadata["banks"] = bank_metadata
    details = {
        "array_inventory": inventory,
        "query_count_by_bank": query_count_by_bank,
        "fit_row_count": int(fit_row_count),
        "owned_discovery_content_sha256": _sha256_json(
            {
                "array_inventory": inventory,
                "discovery_metadata": discovery_metadata,
            }
        ),
    }
    return arrays, discovery_metadata, details


def validate_owned_discovery_snapshot(
    snapshot_dir: Path | str,
    *,
    expected_cache_key: str | None = None,
    expected_binding: Mapping[str, Any] | None = None,
    expected_service_identity_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Read and validate the non-executable NPZ/JSON snapshot only.

    This validator never reads the service's joblib audit checkpoint and calls
    ``numpy.load`` with object deserialization disabled.
    """

    root = Path(snapshot_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("neural-query owned snapshot must be one real directory")
    candidates = sorted(root.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() for path in candidates) or {path.name for path in candidates} != {
        "arrays.npz",
        "metadata.json",
    }:
        raise ValueError(
            "neural-query owned snapshot must contain only NPZ arrays and JSON metadata"
        )
    arrays_path = root / "arrays.npz"
    metadata_path = root / "metadata.json"
    metadata, _metadata_sha256 = _stable_json_file(metadata_path)
    expected_fields = {
        "schema_version",
        "cache_key",
        "binding",
        "service_identity_sha256",
        "arrays_file",
        "arrays_sha256",
        "array_inventory",
        "query_count_by_bank",
        "fit_row_count",
        "owned_discovery_content_sha256",
        "discovery_metadata",
        "snapshot_source",
        "executable_serialization_present",
        "joblib_checkpoint_loaded",
        "content_sha256",
    }
    if set(metadata) != expected_fields:
        raise ValueError("neural-query owned snapshot metadata has an open or incomplete schema")
    body = {key: value for key, value in metadata.items() if key != "content_sha256"}
    if (
        metadata.get("schema_version") != NEURAL_QUERY_OWNED_SNAPSHOT_SCHEMA
        or metadata.get("content_sha256") != _sha256_json(body)
        or metadata.get("arrays_file") != "arrays.npz"
        or metadata.get("snapshot_source") != "trusted_current_service_memory"
        or metadata.get("executable_serialization_present") is not False
        or metadata.get("joblib_checkpoint_loaded") is not False
    ):
        raise ValueError("neural-query owned snapshot metadata is not self-authenticating")
    cache_key = _require_sha256(metadata.get("cache_key"), name="snapshot cache_key")
    service_sha256 = _require_sha256(
        metadata.get("service_identity_sha256"),
        name="snapshot service identity",
    )
    if expected_cache_key is not None and cache_key != _require_sha256(
        expected_cache_key,
        name="expected snapshot cache_key",
    ):
        raise ValueError("neural-query owned snapshot has another cache key")
    if expected_service_identity_sha256 is not None and service_sha256 != _require_sha256(
        expected_service_identity_sha256,
        name="expected service identity",
    ):
        raise ValueError("neural-query owned snapshot has another service identity")
    binding = metadata.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("neural-query owned snapshot has no fit binding")
    if expected_binding is not None and dict(binding) != copy.deepcopy(dict(expected_binding)):
        raise ValueError("neural-query owned snapshot has another exact fit binding")
    if _sha256_json(binding) != cache_key:
        raise ValueError("neural-query owned snapshot cache key does not bind its fit inputs")
    binding_rows = _integer_rows(
        binding.get("row_ids") or (),
        name="snapshot binding.row_ids",
    )
    if int(binding.get("row_count", 0)) != len(binding_rows):
        raise ValueError("neural-query owned snapshot fit row count is invalid")
    for field in (
        "service_identity_sha256",
        "text_sha256",
        "treatment_sha256",
        "outcome_sha256",
        "embedding_row_binding_sha256",
    ):
        _require_sha256(binding.get(field), name=f"snapshot binding.{field}")
    if binding.get("service_identity_sha256") != service_sha256:
        raise ValueError("neural-query snapshot service binding is inconsistent")

    arrays_sha256 = _sha256_file(arrays_path)
    if arrays_sha256 != _require_sha256(
        metadata.get("arrays_sha256"),
        name="snapshot arrays_sha256",
    ):
        raise RuntimeError("neural-query owned snapshot NPZ changed after emission")
    inventory = metadata.get("array_inventory")
    counts = metadata.get("query_count_by_bank")
    discovery_metadata = metadata.get("discovery_metadata")
    if (
        not isinstance(inventory, Mapping)
        or not isinstance(counts, Mapping)
        or set(counts) != set(_BANKS)
        or not isinstance(discovery_metadata, Mapping)
    ):
        raise ValueError("neural-query owned snapshot has malformed array metadata")
    expected_array_keys = {
        f"{bank}_{suffix}" for bank in _BANKS for suffix in ("queries", "train_activations")
    }
    if set(inventory) != expected_array_keys:
        raise ValueError("neural-query owned snapshot array inventory is incomplete")
    observed_inventory: dict[str, Any] = {}
    before_npz_sha256 = arrays_sha256
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_array_keys:
            raise ValueError("neural-query owned snapshot NPZ has unexpected arrays")
        for key in sorted(expected_array_keys):
            array = np.asarray(archive[key])
            if array.dtype.hasobject or array.ndim != 2 or not np.isfinite(array).all():
                raise ValueError("neural-query owned snapshot contains an invalid numerical array")
            observed_inventory[key] = {
                "dtype": array.dtype.str,
                "shape": [int(dimension) for dimension in array.shape],
                "content_sha256": _array_content_sha256(array),
            }
    if _sha256_file(arrays_path) != before_npz_sha256:
        raise RuntimeError("neural-query owned snapshot NPZ changed while validating")
    if observed_inventory != dict(inventory):
        raise RuntimeError("neural-query owned snapshot array inventory does not match its NPZ")
    fit_row_count = int(metadata.get("fit_row_count", 0))
    if fit_row_count != len(binding_rows):
        raise ValueError("neural-query owned snapshot activations are bound to another row count")
    for bank in _BANKS:
        query_count = int(counts[bank])
        query_shape = observed_inventory[f"{bank}_queries"]["shape"]
        activation_shape = observed_inventory[f"{bank}_train_activations"]["shape"]
        if (
            query_count < 1
            or query_shape[0] != query_count
            or activation_shape != [fit_row_count, query_count]
        ):
            raise ValueError(f"neural-query owned snapshot {bank} shapes are not scope-bound")
    nuisance = discovery_metadata.get("fit_nuisance_output_binding")
    if (
        discovery_metadata.get("runtime") != NEURAL_QUERY_DISCOVERY_RUNTIME_ID
        or discovery_metadata.get("executable_checkpoint_io") is not False
        or discovery_metadata.get("all_queries_retained") is not True
        or discovery_metadata.get("validation_audits_used_for_selection") is not False
        or not isinstance(nuisance, Mapping)
        or nuisance.get("schema_version") != NEURAL_QUERY_NUISANCE_OUTPUT_BINDING_SCHEMA
        or tuple(map(int, nuisance.get("fit_row_ids") or ())) != binding_rows
        or nuisance.get("heldout_labels_accessed") is not False
    ):
        raise ValueError("neural-query owned snapshot discovery metadata is not fit-scope closed")
    for field in ("fit_input_binding_sha256",):
        _require_sha256(discovery_metadata.get(field), name=f"snapshot discovery.{field}")
    for field in ("fit_e_sha256", "fit_m_sha256"):
        _require_sha256(nuisance.get(field), name=f"snapshot nuisance.{field}")
    observed_owned_sha256 = _sha256_json(
        {
            "array_inventory": observed_inventory,
            "discovery_metadata": copy.deepcopy(dict(discovery_metadata)),
        }
    )
    if observed_owned_sha256 != _require_sha256(
        metadata.get("owned_discovery_content_sha256"),
        name="snapshot owned discovery content",
    ):
        raise RuntimeError("neural-query owned snapshot discovery content is inconsistent")
    return copy.deepcopy(metadata)


def _fit_context_query_discovery(
    *,
    row_ids: tuple[int, ...],
    chunks: Sequence[np.ndarray],
    texts: tuple[str, ...],
    treatment: np.ndarray,
    outcome: np.ndarray,
    outcome_binary: bool,
    nuisance_views: Sequence[Any],
    query_config: NeuralQueryAgenticForestConfig,
    nuisance_folds: int,
    devices: tuple[str, ...],
    seed: int,
) -> Mapping[str, Any]:
    """Run production in-memory neural-query discovery on one spent context."""

    nuisance = fit_joint_cross_fitted_nuisance_stacks(
        texts=list(texts),
        treatment=np.asarray(treatment, dtype=float),
        outcome=np.asarray(outcome, dtype=float),
        outcome_binary=bool(outcome_binary),
        strata=_strata(treatment, outcome, outcome_binary=bool(outcome_binary)),
        views=list(nuisance_views),
        folds=int(nuisance_folds),
        random_state=int(seed + 10_000),
    )
    fit_e = np.asarray(nuisance["treatment"]["stacked_oof"], dtype=float)
    fit_m = np.asarray(nuisance["outcome"]["stacked_oof"], dtype=float)
    del nuisance
    return fit_in_memory_query_discovery(
        fit_ids=row_ids,
        fit_chunks=chunks,
        fit_texts=texts,
        treatment=np.asarray(treatment, dtype=float),
        outcome=np.asarray(outcome, dtype=float),
        outcome_binary=bool(outcome_binary),
        fit_e=fit_e,
        fit_m=fit_m,
        nuisance_views=list(nuisance_views),
        config=query_config,
        nuisance_folds=int(nuisance_folds),
        devices=devices,
        seed=int(seed),
    )


class ContextFitNeuralQueryService:
    """Cache exact spent-context query definitions for evidence and gate use."""

    def __init__(
        self,
        *,
        cache_dir: Path | str,
        dataset_path: Path | str,
        text_column: str,
        embedding_cache_dir: Path | str | None = None,
        stage1_config_path: Path | str | None = None,
        embedding_cache: SpentOnlyFrozenChunkEmbeddingCache | None = None,
        stage1_config_snapshot: HistoricalStage1ConfigSnapshot | None = None,
        query_config: NeuralQueryAgenticForestConfig = NeuralQueryAgenticForestConfig(),
        nuisance_folds: int = 3,
        devices: Sequence[str] = ("cuda:0",),
        seed: int = 42,
        outcome_type: str = "binary",
    ) -> None:
        self.cache_dir = _validated_fresh_executable_cache_root(cache_dir)
        # A checkpoint is executable Python serialization. Persisted copies are
        # audit artifacts only: a live service keeps its trusted discoveries in
        # memory and never executes mutable bytes from the cache directory.
        self._owned_discoveries: dict[str, Mapping[str, Any]] = {}
        self._owned_discovery_bindings: dict[str, Mapping[str, Any]] = {}
        self._owned_discovery_content_sha256s: dict[str, str] = {}
        self.dataset_path = Path(dataset_path).resolve()
        if not self.dataset_path.is_file():
            raise FileNotFoundError("neural-query context dataset must exist")
        self._stage1_config_snapshot = _historical_stage1_config_snapshot(
            stage1_config_path,
            stage1_config_snapshot,
        )
        self.stage1_config_path = self._stage1_config_snapshot.source_path
        self.text_column = str(text_column).strip()
        if not self.text_column:
            raise ValueError("text_column must be non-empty")
        # Do not materialize the dataset text projection here.  The global
        # cache authenticates its bytes and row count without decoding JSON;
        # semantic text is decoded and checked only for rows explicitly bound
        # by a spent-context or post-proposal gate call.
        if embedding_cache is not None:
            if not isinstance(embedding_cache, SpentOnlyFrozenChunkEmbeddingCache):
                raise TypeError("embedding_cache must be SpentOnlyFrozenChunkEmbeddingCache")
            if (
                embedding_cache_dir is not None
                and embedding_cache.cache_dir != Path(embedding_cache_dir).resolve()
            ):
                raise ValueError("embedding_cache_dir does not match supplied embedding_cache")
            self.embedding_cache = embedding_cache
        else:
            if embedding_cache_dir is None:
                raise ValueError("embedding_cache_dir or embedding_cache is required")
            self.embedding_cache = SpentOnlyFrozenChunkEmbeddingCache(embedding_cache_dir)
        self._dataset_row_count = self.embedding_cache.row_count
        applied = self._stage1_config_snapshot.applied_config()
        self._nuisance_views = copy.deepcopy(
            tuple(applied.architecture.multi_model_forest.bow_views)
        )
        if not self._nuisance_views:
            raise ValueError("Stage-1 config contains no nuisance views")
        self.query_config = copy.deepcopy(query_config)
        self.query_config.validate()
        self.nuisance_folds = _positive_int(nuisance_folds, name="nuisance_folds")
        if self.nuisance_folds < 2:
            raise ValueError("nuisance_folds must be at least two")
        self.devices = tuple(str(value).strip() for value in devices)
        if not self.devices or any(
            not (value == "cpu" or value.startswith("cuda:")) for value in self.devices
        ):
            raise ValueError("devices must contain explicit CPU/CUDA device names")
        self.seed = int(seed)
        self.outcome_type = str(outcome_type).strip().lower()
        if self.outcome_type not in {"binary", "continuous"}:
            raise ValueError("outcome_type must be binary or continuous")
        self._identity = self._identity_payload()

    def _identity_payload(self) -> dict[str, Any]:
        self._stage1_config_snapshot.verify_source()
        return {
            "service": NEURAL_QUERY_CONTEXT_SERVICE_ID,
            "service_code_sha256": _sha256_file(Path(__file__).resolve()),
            "query_discovery_runtime": {
                "runtime": NEURAL_QUERY_DISCOVERY_RUNTIME_ID,
                "code_sha256": _query_discovery_runtime_code_sha256(),
                "executable_checkpoint_io": False,
            },
            "dependency_code_sha256s": _dependency_code_sha256s(),
            "library_versions": {
                "joblib": joblib.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
            },
            "dataset_row_count": self._dataset_row_count,
            "text_column": self.text_column,
            "stage1_config_sha256": self._stage1_config_snapshot.sha256,
            "embedding_cache": self.embedding_cache.identity(),
            "nuisance_views_sha256": _sha256_json(
                [_json_state(view) for view in self._nuisance_views]
            ),
            "query_config": asdict(self.query_config),
            "nuisance_folds": self.nuisance_folds,
            "devices": list(self.devices),
            "seed": self.seed,
            "outcome_type": self.outcome_type,
            "gate_labels_accepted": False,
            "novel_semantic_encoding_allowed": False,
            "preexisting_executable_cache_entries_accepted": False,
            "executable_cache_reuse_scope": "current_service_instance_only",
        }

    def identity(self) -> Mapping[str, Any]:
        current = self._identity_payload()
        if current != self._identity:
            raise RuntimeError("neural-query context service state changed after binding")
        return copy.deepcopy(self._identity)

    def _normalize_rows_and_texts(
        self,
        row_ids: Sequence[Any],
        texts: Sequence[Any],
        *,
        row_name: str,
        text_name: str,
    ) -> tuple[tuple[int, ...], tuple[str, ...]]:
        rows = _integer_rows(row_ids, name=row_name)
        exact = _exact_texts(texts, name=text_name, length=len(rows))
        if any(row_id >= self._dataset_row_count for row_id in rows):
            raise IndexError(f"{row_name} contains a row outside the frozen embedding cache")
        return rows, exact

    def _bind_rows_and_texts(
        self,
        row_ids: Sequence[Any],
        texts: Sequence[Any],
        *,
        row_name: str,
        text_name: str,
    ) -> tuple[
        tuple[int, ...],
        tuple[str, ...],
        BoundSpentFrozenChunkEmbeddingProvider,
    ]:
        rows, exact = self._normalize_rows_and_texts(
            row_ids,
            texts,
            row_name=row_name,
            text_name=text_name,
        )
        bound = self.embedding_cache.bind_spent(rows, exact)
        self.identity()
        return rows, exact, bound

    def _binding(
        self,
        *,
        outer_fold: int,
        row_ids: tuple[int, ...],
        texts: tuple[str, ...],
        treatment: np.ndarray,
        outcome: np.ndarray,
        embedding_provider: BoundSpentFrozenChunkEmbeddingProvider,
    ) -> dict[str, Any]:
        return {
            "service_identity_sha256": _sha256_json(self._identity),
            "outer_fold": int(outer_fold),
            "row_ids": list(row_ids),
            "text_sha256": _sha256_json(list(texts)),
            "treatment_sha256": _float_hex_sha256(treatment),
            "outcome_sha256": _float_hex_sha256(outcome),
            "row_count": len(row_ids),
            "embedding_row_binding_sha256": _sha256_json(embedding_provider.identity()),
        }

    def _load_or_fit_discovery(
        self,
        *,
        outer_fold: int,
        row_ids: tuple[int, ...],
        texts: tuple[str, ...],
        treatment: np.ndarray,
        outcome: np.ndarray,
        embedding_provider: BoundSpentFrozenChunkEmbeddingProvider,
    ) -> tuple[Mapping[str, Any], str]:
        self.identity()
        binding = self._binding(
            outer_fold=outer_fold,
            row_ids=row_ids,
            texts=texts,
            treatment=treatment,
            outcome=outcome,
            embedding_provider=embedding_provider,
        )
        cache_key = _sha256_json(binding)
        root = self.cache_dir / cache_key
        manifest_path = root / "manifest.json"
        checkpoint_path = root / "query_discovery.joblib"
        trusted = self._owned_discoveries.get(cache_key)
        if trusted is not None:
            owned_binding = self._owned_discovery_bindings.get(cache_key)
            owned_sha256 = self._owned_discovery_content_sha256s.get(cache_key)
            if owned_binding != binding or owned_sha256 != _owned_discovery_memory_sha256(trusted):
                raise RuntimeError("trusted neural-query discovery changed after ownership binding")
            discovery = copy.deepcopy(trusted)
            self._validate_discovery(discovery)
            self.identity()
            return discovery, cache_key
        if manifest_path.exists():
            raise ValueError(
                "refusing a neural-query executable cache entry not held in trusted "
                "service memory"
            )

        if root.exists() and any(root.iterdir()):
            raise ValueError("refusing a pre-populated neural-query executable cache entry")
        root.mkdir(parents=True, exist_ok=True)
        chunks = embedding_provider.chunk_matrices(row_ids)
        discovery = _fit_context_query_discovery(
            row_ids=row_ids,
            chunks=chunks,
            texts=texts,
            treatment=treatment,
            outcome=outcome,
            outcome_binary=self.outcome_type == "binary",
            nuisance_views=self._nuisance_views,
            query_config=copy.deepcopy(self.query_config),
            nuisance_folds=self.nuisance_folds,
            devices=self.devices,
            seed=int(self.seed + 100_000 * int(outer_fold)),
        )
        self._validate_discovery(discovery)
        self.identity()
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=root, prefix=".query_discovery.", delete=False
        ) as handle:
            temporary_checkpoint = Path(handle.name)
        try:
            joblib.dump(discovery, temporary_checkpoint)
            temporary_checkpoint.replace(checkpoint_path)
        finally:
            temporary_checkpoint.unlink(missing_ok=True)
        content = {
            "schema_version": _DISCOVERY_MANIFEST_SCHEMA,
            "cache_key": cache_key,
            "binding": binding,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
        }
        payload = {**content, "content_sha256": _sha256_json(content)}
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root, prefix=".manifest.", delete=False
        ) as handle:
            handle.write(_canonical_json(payload) + "\n")
            temporary_manifest = Path(handle.name)
        try:
            temporary_manifest.replace(manifest_path)
        finally:
            temporary_manifest.unlink(missing_ok=True)
        owned = copy.deepcopy(discovery)
        self._owned_discoveries[cache_key] = owned
        self._owned_discovery_bindings[cache_key] = copy.deepcopy(binding)
        self._owned_discovery_content_sha256s[cache_key] = _owned_discovery_memory_sha256(owned)
        return copy.deepcopy(discovery), cache_key

    def _validate_discovery(self, discovery: Any) -> None:
        if not isinstance(discovery, Mapping):
            raise TypeError("neural-query context discovery must be a mapping")
        banks = discovery.get("banks")
        if not isinstance(banks, Mapping) or set(banks) != set(_BANKS):
            raise ValueError("neural-query context discovery must contain three banks")
        for bank in _BANKS:
            result = banks[bank]
            if not isinstance(result, Mapping):
                raise ValueError(f"neural-query {bank} bank is malformed")
            queries = np.asarray(result.get("queries"), dtype=np.float32)
            records = result.get("records")
            expected = self.query_config.query_count(bank)
            if (
                queries.ndim != 2
                or queries.shape[0] != expected
                or not np.isfinite(queries).all()
                or not isinstance(records, list)
                or len(records) != expected
            ):
                raise ValueError(f"neural-query {bank} bank has an invalid shape")
            for record in records:
                if not isinstance(record, Mapping):
                    raise ValueError(f"neural-query {bank} fit record is malformed")
                raw_score = record.get("fit_standardized_score")
                if isinstance(raw_score, (bool, np.bool_)):
                    raise ValueError(f"neural-query {bank} fit standardized score must be finite")
                try:
                    fit_score = float(raw_score)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"neural-query {bank} fit standardized score must be finite"
                    ) from exc
                if not math.isfinite(fit_score):
                    raise ValueError(f"neural-query {bank} fit standardized score must be finite")

    def discovery_for_context(
        self,
        *,
        outer_fold: int,
        context_row_ids: Sequence[Any],
        context_texts: Sequence[Any],
        context_treatment: Sequence[float],
        context_outcome: Sequence[float],
    ) -> tuple[Mapping[str, Any], str]:
        fold = _positive_int(outer_fold, name="outer_fold")
        rows, texts, embedding_provider = self._bind_rows_and_texts(
            context_row_ids,
            context_texts,
            row_name="context_row_ids",
            text_name="context_texts",
        )
        treatment = _validate_binary(
            np.asarray(context_treatment, dtype=float),
            name="context_treatment",
            require_both=True,
        )
        outcome = np.asarray(context_outcome, dtype=float)
        if outcome.ndim != 1 or len(outcome) != len(rows) or not np.isfinite(outcome).all():
            raise ValueError("context_outcome must be a finite vector aligned to context rows")
        if len(treatment) != len(rows):
            raise ValueError("context_treatment must align to context rows")
        if self.outcome_type == "binary":
            outcome = _validate_binary(outcome, name="context_outcome", require_both=True)
        return self._load_or_fit_discovery(
            outer_fold=fold,
            row_ids=rows,
            texts=texts,
            treatment=treatment,
            outcome=outcome,
            embedding_provider=embedding_provider,
        )

    def write_owned_discovery_snapshot(
        self,
        *,
        cache_key: str,
        output_dir: Path | str,
    ) -> Mapping[str, Any]:
        """Persist one trusted fit as non-executable NPZ arrays plus closed JSON.

        The executable audit checkpoint is neither read nor copied.  Only state
        retained in this service instance after the genuine fit is eligible.
        """

        self.identity()
        key = _require_sha256(cache_key, name="owned neural-query cache_key")
        try:
            discovery = self._owned_discoveries[key]
            binding = self._owned_discovery_bindings[key]
            owned_sha256 = self._owned_discovery_content_sha256s[key]
        except (AttributeError, KeyError) as exc:
            raise ValueError("neural-query snapshot key is not owned by this service") from exc
        if _sha256_json(binding) != key:
            raise RuntimeError("owned neural-query binding no longer matches its cache key")
        if _owned_discovery_memory_sha256(discovery) != owned_sha256:
            raise RuntimeError("owned neural-query discovery changed after fit")
        arrays, discovery_metadata, details = _owned_discovery_snapshot_parts(discovery)
        root = Path(output_dir)
        if root.exists() or root.is_symlink():
            raise FileExistsError("neural-query owned snapshot target must not already exist")
        root.parent.mkdir(parents=True, exist_ok=True)
        if root.parent.is_symlink():
            raise ValueError("neural-query owned snapshot parent cannot be a symlink")
        root.mkdir(exist_ok=False)
        arrays_path = root / "arrays.npz"
        _atomic_write_new_npz(arrays_path, arrays)
        service_identity_sha256 = _sha256_json(self._identity)
        body = {
            "schema_version": NEURAL_QUERY_OWNED_SNAPSHOT_SCHEMA,
            "cache_key": key,
            "binding": copy.deepcopy(dict(binding)),
            "service_identity_sha256": service_identity_sha256,
            "arrays_file": arrays_path.name,
            "arrays_sha256": _sha256_file(arrays_path),
            "array_inventory": details["array_inventory"],
            "query_count_by_bank": details["query_count_by_bank"],
            "fit_row_count": details["fit_row_count"],
            "owned_discovery_content_sha256": details["owned_discovery_content_sha256"],
            "discovery_metadata": discovery_metadata,
            "snapshot_source": "trusted_current_service_memory",
            "executable_serialization_present": False,
            "joblib_checkpoint_loaded": False,
        }
        metadata = {**body, "content_sha256": _sha256_json(body)}
        _atomic_write_new_bytes(
            root / "metadata.json",
            (_canonical_json(metadata) + "\n").encode("utf-8"),
        )
        validated = validate_owned_discovery_snapshot(
            root,
            expected_cache_key=key,
            expected_binding=binding,
            expected_service_identity_sha256=service_identity_sha256,
        )
        self.identity()
        return validated

    def safe_evidence(
        self,
        *,
        discovery: Mapping[str, Any],
        context_row_ids: tuple[int, ...],
        context_texts: tuple[str, ...],
        device_offset: int = 0,
    ) -> list[dict[str, Any]]:
        self.identity()
        bound_row_ids, _texts, embedding_provider = self._bind_rows_and_texts(
            context_row_ids,
            context_texts,
            row_name="context_row_ids",
            text_name="context_texts",
        )
        self._validate_discovery(discovery)
        chunks = embedding_provider.chunk_matrices(bound_row_ids)
        # The evidence helper accepts a corpus-indexed sequence. Populate only
        # spent rows; sealed chunk text is never materialized for this call.
        corpus: list[Sequence[str]] = [()] * self._dataset_row_count
        for row_id, row_chunks in zip(
            bound_row_ids,
            embedding_provider.chunk_texts(bound_row_ids),
        ):
            corpus[int(row_id)] = tuple(row_chunks)
        output: list[dict[str, Any]] = []
        for bank_index, bank in enumerate(_BANKS):
            result = discovery["banks"][bank]
            evidence_rows = build_query_evidence(
                bank=bank,
                queries=np.asarray(result["queries"], dtype=np.float32),
                query_records=result["records"],
                row_ids=bound_row_ids,
                chunk_matrices=chunks,
                all_chunk_texts=corpus,
                config=copy.deepcopy(self.query_config),
                device=self.devices[(bank_index + int(device_offset)) % len(self.devices)],
                seed=int(self.seed + 3_000 + bank_index),
            )
            for row in evidence_rows:
                # Contrastive n-grams and aggregate diagnostics carry the
                # ontology signal. Raw note excerpts, row IDs, and chunk IDs do
                # not cross the spent-evidence provider boundary.
                output.append(
                    {
                        "query_id": str(row["query_id"]),
                        "bank": str(row["bank"]),
                        "mechanical_role": str(row["mechanical_role"]),
                        "statistical_gate_applied": False,
                        "member_count": int(row.get("member_count", 0)),
                        "fit_standardized_score": (
                            None
                            if row.get("fit_standardized_score") is None
                            else float(row["fit_standardized_score"])
                        ),
                        "top_chunks": [],
                        "top_contrastive_ngrams": _safe_query_ngram_rows(
                            row.get("top_contrastive_ngrams")
                        ),
                    }
                )
        if len(output) != sum(self.query_config.query_count(bank) for bank in _BANKS):
            raise RuntimeError("every context-fitted query must produce safe evidence")
        self.identity()
        return output


class NeuralQueryContextBackend:
    """Expose exact fixed per-bank v3-style neural-query moments.

    Query indices are intentionally *not* treated as semantically aligned
    across context fits.  Each bank is reduced here to a signed mean, the
    absolute maximum of the original unsigned activations, and the descending
    order statistics of activations oriented by the finite fit-score sign.
    """

    def __init__(self, service: ContextFitNeuralQueryService) -> None:
        if not isinstance(service, ContextFitNeuralQueryService):
            raise TypeError("service must be a ContextFitNeuralQueryService")
        self.service = service
        self._service_identity = service.identity()

    def identity(self) -> Mapping[str, Any]:
        if self.service.identity() != self._service_identity:
            raise RuntimeError("neural-query context service identity changed")
        return {
            "backend": NEURAL_QUERY_CONTEXT_BACKEND_ID,
            "service": copy.deepcopy(self._service_identity),
            "gate_labels_exposed": False,
            "query_features_calibrated_effects": False,
            "query_feature_semantics": (
                "v3_signed_mean_unsigned_absolute_max_signed_descending_order"
            ),
            "fit_score_sign_source": "finite_fit_standardized_score",
            "fit_local_query_indices_semantically_aligned_across_context_fits": False,
            "fit_local_query_indices_exposed": False,
            "permutation_invariant_reduction_applied_before_cross_fit_alignment": True,
            "exact_preaggregated_stable_wrapper_passthrough_required": True,
        }

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
        del work_dir
        self.identity()
        context_rows, context_exact_texts = self.service._normalize_rows_and_texts(
            context_row_ids,
            context_texts,
            row_name="context_row_ids",
            text_name="context_texts",
        )
        gate_rows, gate_exact_texts = self.service._normalize_rows_and_texts(
            gate_row_ids,
            gate_texts,
            row_name="gate_row_ids",
            text_name="gate_texts",
        )
        if set(context_rows) & set(gate_rows):
            raise ValueError("neural-query context and gate rows must be disjoint")
        discovery, _cache_key = self.service.discovery_for_context(
            outer_fold=outer_fold,
            context_row_ids=context_rows,
            context_texts=context_exact_texts,
            context_treatment=context_treatment,
            context_outcome=context_outcome,
        )
        self.service._validate_discovery(discovery)
        # Gate text is bound only after the spent-context proposal has been
        # fitted/loaded.  No gate treatment or outcome is accepted here.
        _gate_rows, _gate_texts, gate_embedding_provider = self.service._bind_rows_and_texts(
            gate_rows,
            gate_exact_texts,
            row_name="gate_row_ids",
            text_name="gate_texts",
        )
        gate_chunks = gate_embedding_provider.chunk_matrices(gate_rows)
        names: list[str] = []
        kinds: list[str] = []
        roles: list[str] = []
        columns: list[np.ndarray] = []
        for bank_index, bank in enumerate(_BANKS):
            result = discovery["banks"][bank]
            activations = soft_retrieval_activations(
                gate_chunks,
                np.asarray(result["queries"], dtype=np.float32),
                temperature=float(self.service.query_config.temperature),
                device=self.service.devices[bank_index % len(self.service.devices)],
            )
            expected = self.service.query_config.query_count(bank)
            if (
                activations.shape != (len(gate_rows), expected)
                or not np.isfinite(activations).all()
            ):
                raise ValueError(f"neural-query {bank} gate activations are invalid")
            fit_scores = np.asarray(
                [float(record["fit_standardized_score"]) for record in result["records"]],
                dtype=float,
            )
            if fit_scores.shape != (expected,) or not np.isfinite(fit_scores).all():
                raise ValueError(f"neural-query {bank} fit standardized scores are invalid")
            signed_activations = np.asarray(activations, dtype=float) * np.sign(fit_scores)[None, :]
            signed_descending = np.sort(signed_activations, axis=1, kind="stable")[:, ::-1]
            bank_names = (
                f"neural_query_{bank}_signed_mean",
                f"neural_query_{bank}_absolute_max",
                *(
                    f"neural_query_{bank}_signed_order_{rank:02d}"
                    for rank in range(1, expected + 1)
                ),
            )
            bank_values = np.column_stack(
                (
                    np.mean(signed_descending, axis=1),
                    np.max(np.abs(activations), axis=1),
                    signed_descending,
                )
            )
            if bank_values.shape != (len(gate_rows), expected + 2):
                raise RuntimeError(f"neural-query {bank} moment reduction is not rectangular")
            for column, name in enumerate(bank_names):
                names.append(name)
                kinds.append(f"neural_query_{bank}_moments")
                roles.append(_ROLE_BY_BANK[bank])
                columns.append(np.asarray(bank_values[:, column], dtype=float))
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
        self.identity()
        return prediction


class NeuralQuerySpentEvidenceProvider:
    """Return concept evidence from exact spent rows and no sealed text."""

    def __init__(self, service: ContextFitNeuralQueryService) -> None:
        if not isinstance(service, ContextFitNeuralQueryService):
            raise TypeError("service must be a ContextFitNeuralQueryService")
        self.service = service
        self._service_identity = service.identity()

    def identity(self) -> Mapping[str, Any]:
        if self.service.identity() != self._service_identity:
            raise RuntimeError("neural-query context service identity changed")
        return {
            "provider": NEURAL_QUERY_SPENT_EVIDENCE_PROVIDER_ID,
            "service_identity_sha256": _sha256_json(self._service_identity),
            "sealed_text_or_labels_accepted": False,
            "row_level_excerpts_emitted": False,
        }

    def get_spent_evidence_inputs(
        self,
        *,
        outer_fold: int,
        review_round: int,
        exact_spent_row_ids: tuple[int, ...],
        exact_sealed_row_ids: tuple[int, ...],
        spent_texts: tuple[str, ...],
        spent_treatment: np.ndarray,
        spent_outcome: np.ndarray,
    ) -> Sequence[FoldEvidenceInput]:
        self.identity()
        fold = _positive_int(outer_fold, name="outer_fold")
        round_id = _nonnegative_int(review_round, name="review_round")
        spent_rows, exact_texts = self.service._normalize_rows_and_texts(
            exact_spent_row_ids,
            spent_texts,
            row_name="exact_spent_row_ids",
            text_name="spent_texts",
        )
        sealed_rows = _integer_rows(exact_sealed_row_ids, name="exact_sealed_row_ids")
        if set(spent_rows) & set(sealed_rows):
            raise ValueError("spent and sealed neural-query rows overlap")
        if any(row_id >= self.service._dataset_row_count for row_id in sealed_rows):
            raise IndexError("a sealed neural-query row is outside the dataset projection")
        discovery, cache_key = self.service.discovery_for_context(
            outer_fold=fold,
            context_row_ids=spent_rows,
            context_texts=exact_texts,
            context_treatment=spent_treatment,
            context_outcome=spent_outcome,
        )
        provenance = FoldEvidenceProvenance(
            outer_fold=fold,
            train_row_ids=spent_rows,
            heldout_row_ids=sealed_rows,
            scope="inner_train",
            inner_fold=round_id + 1,
            artifact_id=f"neural-query-spent-{fold}-{round_id}-{cache_key[:16]}",
        )
        payload = {
            "schema_version": NEURAL_QUERY_SPENT_EVIDENCE_SCHEMA,
            "source_kind": NEURAL_QUERY_SOURCE,
            "source_family": NEURAL_QUERY_MOMENTS,
            "outer_fold": fold,
            "scope": "inner_train",
            "inner_fold": round_id + 1,
            "adapter_mode": "context_fit_neural_query_moments",
            "query_evidence": self.service.safe_evidence(
                discovery=discovery,
                context_row_ids=spent_rows,
                context_texts=exact_texts,
                device_offset=max(0, round_id - 1),
            ),
        }
        result = (
            FoldEvidenceInput(
                source_kind=NEURAL_QUERY_SOURCE,
                payload=payload,
                provenance=provenance,
            ),
        )
        self.identity()
        return result


class NeuralQuerySpentDiscoveryBackend:
    """Composable neural-query backend for ContextFitReviewSpentEvidenceProvider."""

    def __init__(self, service: ContextFitNeuralQueryService) -> None:
        if not isinstance(service, ContextFitNeuralQueryService):
            raise TypeError("service must be a ContextFitNeuralQueryService")
        self.service = service
        self._service_identity = service.identity()

    def identity(self) -> Mapping[str, Any]:
        if self.service.identity() != self._service_identity:
            raise RuntimeError("neural-query context service identity changed")
        return {
            "backend": NEURAL_QUERY_SPENT_DISCOVERY_BACKEND_ID,
            "service_identity_sha256": _sha256_json(self._service_identity),
            "sealed_text_or_labels_accepted": False,
            "row_level_excerpts_emitted": False,
        }

    def fit_discovery(
        self,
        *,
        outer_fold: int,
        review_round: int,
        exact_spent_row_ids: tuple[int, ...],
        spent_texts: tuple[str, ...],
        spent_treatment: np.ndarray,
        spent_outcome: np.ndarray,
        work_dir: Path,
    ) -> SpentDiscoveryEvidence:
        del work_dir
        self.identity()
        fold = _positive_int(outer_fold, name="outer_fold")
        round_id = _nonnegative_int(review_round, name="review_round")
        spent_rows, exact_texts = self.service._normalize_rows_and_texts(
            exact_spent_row_ids,
            spent_texts,
            row_name="exact_spent_row_ids",
            text_name="spent_texts",
        )
        discovery, _cache_key = self.service.discovery_for_context(
            outer_fold=fold,
            context_row_ids=spent_rows,
            context_texts=exact_texts,
            context_treatment=spent_treatment,
            context_outcome=spent_outcome,
        )
        payload = {
            "schema_version": NEURAL_QUERY_SPENT_EVIDENCE_SCHEMA,
            "source_kind": NEURAL_QUERY_SOURCE,
            "source_family": NEURAL_QUERY_MOMENTS,
            "outer_fold": fold,
            "scope": "inner_train",
            "inner_fold": round_id + 1,
            "adapter_mode": "context_fit_neural_query_moments",
            "query_evidence": self.service.safe_evidence(
                discovery=discovery,
                context_row_ids=spent_rows,
                context_texts=exact_texts,
                device_offset=max(0, round_id - 1),
            ),
        }
        result = SpentDiscoveryEvidence.create(
            source_kind=NEURAL_QUERY_SOURCE,
            payload=payload,
            fit_row_provenance=FitRowProvenance(fit_row_ids=frozenset(spent_rows)),
        )
        self.identity()
        return result


__all__ = [
    "NEURAL_QUERY_CONTEXT_BACKEND_ID",
    "NEURAL_QUERY_CONTEXT_SERVICE_ID",
    "NEURAL_QUERY_NUISANCE_OUTPUT_BINDING_SCHEMA",
    "NEURAL_QUERY_OWNED_SNAPSHOT_SCHEMA",
    "NEURAL_QUERY_SPENT_EVIDENCE_PROVIDER_ID",
    "NEURAL_QUERY_SPENT_DISCOVERY_BACKEND_ID",
    "ContextFitNeuralQueryService",
    "NeuralQueryContextBackend",
    "NeuralQuerySpentEvidenceProvider",
    "NeuralQuerySpentDiscoveryBackend",
    "validate_owned_discovery_snapshot",
]
