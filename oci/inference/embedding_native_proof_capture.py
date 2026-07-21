"""Authenticated capture and replay for exact-scope embedding evidence.

The frozen embedding cache is a permissible unsupervised input, but the
supervised directions, cluster-local contrasts, retrieved witnesses, and the
semantic TF-IDF projection must be rebuilt inside the registered fit scope.
This module records the exact numerical inputs and actual KMeans/SVD states,
then replays the native generator against the authenticated fit-only cache.

Semantic retrieval has no label-selected hyperparameter.  Its model and
calibration partitions are therefore honest, label-free replay canaries only;
the authoritative vocabulary/projection remains exhaustive over every frozen
retrieval tail from the complete exact-fit scope.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from oci.config import EmbeddingContrastDiscoveryConfig

from .embedding_contrast_discovery import _named_pseudo_targets
from .review_spent_evidence_provider import (
    BoundSpentFrozenChunkEmbeddingProvider,
    _FrozenCacheEmbeddingEvidenceGenerator,
    _embedding_concepts_only,
)

EMBEDDING_NATIVE_CAPTURE_SCHEMA = "production_embedding_native_capture_v1"
EMBEDDING_NATIVE_ARRAY_SCHEMA = "production_embedding_native_capture_array_v1"
SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA = (
    "semantic_retrieval_training_only_exhaustive_no_selection_v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_KEY = re.compile(r"^[a-z0-9_]+$")
_FORBIDDEN_SUFFIXES = (
    ".joblib",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
)
_ARTIFACT_FILENAMES = (
    "arrays.npz",
    "metadata.json",
    "raw_embedding_evidence.json",
    "semantic_calibration_replay_canary.json",
    "semantic_full_scope_evidence.json",
    "semantic_model_replay_canary.json",
)
_RETRIEVAL_KEYS = (
    "positive_aligned_chunks",
    "negative_aligned_chunks",
    "positive_external_chunks",
    "negative_external_chunks",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate embedding JSON object key: {key}")
        result[key] = value
    return result


def _read_json_object_reject_duplicates(path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be one JSON object")
    return value


def _clone(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "schema_version": EMBEDDING_NATIVE_ARRAY_SCHEMA,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        ).encode("utf-8")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _row_fingerprint(row_ids: Sequence[int]) -> str:
    rows = tuple(map(int, row_ids))
    if not rows or len(rows) != len(set(rows)) or any(row < 0 for row in rows):
        raise ValueError("row IDs must be unique non-negative integers")
    return _sha256_json({"ordered_row_ids": list(rows)})


def _text_sha256(row_ids: Sequence[int], texts: Sequence[str]) -> str:
    rows = tuple(map(int, row_ids))
    values = tuple(str(text) for text in texts)
    if len(rows) != len(values):
        raise ValueError("text binding requires one text per row ID")
    digest = hashlib.sha256(b"production-embedding-fit-text-binding-v1\0")
    for row_id, text in zip(rows, values):
        encoded = text.encode("utf-8")
        digest.update(int(row_id).to_bytes(8, byteorder="little", signed=False))
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"refusing to replace immutable embedding artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"refusing to replace immutable embedding artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
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


class _ArrayStore:
    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.inventory: dict[str, dict[str, Any]] = {}

    def add(self, key: str, value: Any) -> str:
        key = str(key)
        if _SAFE_KEY.fullmatch(key) is None or key in self.arrays:
            raise ValueError(f"invalid or duplicate embedding capture array key: {key}")
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise ValueError("embedding capture arrays cannot use object dtype")
        self.arrays[key] = array
        self.inventory[key] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "content_sha256": _array_sha256(array),
        }
        return key


def _finite_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or len(array) != int(length) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite vector with the exact fit length")
    return array


def _embedding_config_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw = dict(value)
    elif is_dataclass(value):
        raw = asdict(value)
    else:
        raw = vars(value)
    config = _clone(raw)
    # The frozen-cache generator deliberately disables every external or
    # language-model concept source.  Its native proof may not weaken that.
    if (
        config.get("enabled") is not True
        or config.get("include_bow_phrases_as_concepts") is not False
        or config.get("concept_phrases") != []
        or config.get("external_corpus_cache_dirs") != []
    ):
        raise ValueError("embedding native capture requires the exact frozen-cache configuration")
    return config


def _cache_row_inventory(
    provider: BoundSpentFrozenChunkEmbeddingProvider,
    row_ids: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_id in map(int, row_ids):
        chunks = tuple(map(str, provider.chunk_texts((row_id,))[0]))
        matrix = np.asarray(provider.chunk_matrices((row_id,))[0], dtype=np.float32)
        if matrix.ndim != 2 or len(chunks) != matrix.shape[0]:
            raise ValueError("frozen cache chunk text and embedding rows differ")
        rows.append(
            {
                "row_id": row_id,
                "chunk_count": len(chunks),
                "chunk_texts_sha256": _sha256_json(list(chunks)),
                "chunk_matrix_dtype": matrix.dtype.str,
                "chunk_matrix_shape": list(matrix.shape),
                "chunk_matrix_sha256": _array_sha256(matrix),
            }
        )
    return rows


def build_semantic_retrieval_training_only_policy(
    *,
    fit_row_ids: Sequence[int],
    outer_fold: int,
    inner_fold: int,
    configured_fold_count: int,
    seed: int,
) -> dict[str, Any]:
    """Build a label-free nested replay partition with no selection authority."""

    rows = tuple(map(int, fit_row_ids))
    _row_fingerprint(rows)
    configured = int(configured_fold_count)
    if configured < 2 or len(rows) < 4:
        raise ValueError("semantic replay canaries require >=4 rows and >=2 configured folds")
    fold_count = min(configured, max(2, len(rows) // 2))
    nested_seed = int(seed) + 73_000 + 1_009 * int(outer_fold) + int(inner_fold)
    positions = np.arange(len(rows), dtype=int)
    np.random.RandomState(nested_seed).shuffle(positions)
    folds = tuple(np.asarray(part, dtype=int) for part in np.array_split(positions, fold_count))
    selected_index = (int(outer_fold) + int(inner_fold) - 1) % fold_count
    calibration_positions = set(map(int, folds[selected_index]))
    model_rows = [row for index, row in enumerate(rows) if index not in calibration_positions]
    calibration_rows = [row for index, row in enumerate(rows) if index in calibration_positions]
    if (
        not model_rows
        or not calibration_rows
        or set(model_rows) & set(calibration_rows)
        or set(model_rows) | set(calibration_rows) != set(rows)
    ):
        raise RuntimeError("semantic replay canaries did not partition the exact fit scope")
    return {
        "schema_version": SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA,
        "policy": "training_only_exhaustive_no_selection",
        "selection_kind": "none_deterministic_exhaustive",
        "nested_calibration_applicability": "no_label_or_hyperparameter_selection",
        "seed": nested_seed,
        "fold_parameter": "tfidf_nested_calibration_folds",
        "configured_fold_count": configured,
        "fold_count": fold_count,
        "split_method": "ordered_row_positions_seeded_label_free_partition",
        "model_fit_row_ids": model_rows,
        "calibration_row_ids": calibration_rows,
        "model_fit_row_order_fingerprint": _row_fingerprint(model_rows),
        "calibration_row_order_fingerprint": _row_fingerprint(calibration_rows),
        "partitions_are_replay_canaries_only": True,
        "partition_canaries_select_or_drop_terms": False,
        "authoritative_projection_scope": "all_exact_fit_frozen_retrieval_tails",
        "projection_vocabulary_max_features": None,
        "projection_output_limit": None,
        "all_nonzero_sanitized_terms_preserved": True,
        "upstream_embedding_directions_and_retrieval_use_exact_fit_labels_only": True,
        "nested_calibration_labels_accessed": False,
        "registered_heldout_labels_accessed": False,
        "registered_heldout_text_accessed": False,
        "registered_heldout_transform_performed": False,
        "selection_frozen_before_registered_heldout_use": True,
        "projection_frozen_before_registered_heldout_use": True,
        "canonical_hierarchy_partition_count_used_as_calibration_folds": False,
        "interaction_inner_folds_used_as_calibration_folds": False,
    }


def _filter_retrieval_rows(
    evidence: Mapping[str, Any],
    *,
    allowed_row_ids: set[int],
) -> dict[str, Any]:
    filtered = copy.deepcopy(dict(evidence))
    contrasts = filtered.get("contrasts")
    if not isinstance(contrasts, list):
        raise ValueError("embedding evidence has no contrast list")
    for contrast in contrasts:
        if not isinstance(contrast, dict):
            raise ValueError("embedding evidence contrast is not one object")
        for key in ("positive_external_chunks", "negative_external_chunks"):
            if contrast.get(key):
                raise ValueError("frozen native semantic projection cannot use external chunks")
        for key in ("positive_aligned_chunks", "negative_aligned_chunks"):
            rows = contrast.get(key) or []
            if not isinstance(rows, list):
                raise ValueError("embedding retrieval tail is not a list")
            output = []
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("embedding retrieval row is not one object")
                row_id = int(row.get("row_id"))
                if row_id in allowed_row_ids:
                    output.append(copy.deepcopy(dict(row)))
            contrast[key] = output
    return filtered


def semantic_retrieval_projection_bundle(
    raw_evidence: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Replay full exhaustive projection plus non-selecting partition canaries."""

    if policy.get("schema_version") != SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA:
        raise ValueError("semantic retrieval policy has another schema")
    if (
        policy.get("selection_kind") != "none_deterministic_exhaustive"
        or policy.get("nested_calibration_labels_accessed") is not False
        or policy.get("partition_canaries_select_or_drop_terms") is not False
        or policy.get("projection_vocabulary_max_features") is not None
        or policy.get("projection_output_limit") is not None
    ):
        raise ValueError("semantic retrieval policy permits selection or truncation")
    model_rows = set(map(int, policy.get("model_fit_row_ids") or ()))
    calibration_rows = set(map(int, policy.get("calibration_row_ids") or ()))
    if not model_rows or not calibration_rows or model_rows & calibration_rows:
        raise ValueError("semantic retrieval replay partitions are invalid")
    full = _embedding_concepts_only(raw_evidence, contrastive_term_limit=None)
    model = _embedding_concepts_only(
        _filter_retrieval_rows(raw_evidence, allowed_row_ids=model_rows),
        contrastive_term_limit=None,
    )
    calibration = _embedding_concepts_only(
        _filter_retrieval_rows(raw_evidence, allowed_row_ids=calibration_rows),
        contrastive_term_limit=None,
    )
    return {"full": full, "model_canary": model, "calibration_canary": calibration}


class NativeEmbeddingProofCaptureSink:
    """Observe one actual frozen-cache embedding fit and persist its replay proof."""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        scope_id: str,
        outer_fold: int,
        inner_fold: int,
        fit_row_ids: Sequence[int],
        heldout_row_ids: Sequence[int],
        fit_texts: Sequence[str],
        expected_fit_treatment: Sequence[float],
        expected_fit_outcome: Sequence[float],
        text_column: str,
        outcome_type: str,
        embedding_provider: BoundSpentFrozenChunkEmbeddingProvider,
        embedding_config: EmbeddingContrastDiscoveryConfig | Mapping[str, Any],
        tfidf_nested_calibration_folds: int,
        seed: int,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        if self.artifact_dir.exists():
            raise RuntimeError("embedding native proof artifact directory must be new")
        self.scope_id = str(scope_id)
        self.outer_fold = int(outer_fold)
        self.inner_fold = int(inner_fold)
        if self.outer_fold < 1 or self.inner_fold < 1:
            raise ValueError("embedding proof capture requires a positive exact-inner scope")
        self.fit_row_ids = tuple(map(int, fit_row_ids))
        self.heldout_row_ids = tuple(map(int, heldout_row_ids))
        _row_fingerprint(self.fit_row_ids)
        _row_fingerprint(self.heldout_row_ids)
        if set(self.fit_row_ids) & set(self.heldout_row_ids):
            raise ValueError("embedding proof fit and held-out rows overlap")
        self.fit_texts = tuple(str(value) for value in fit_texts)
        if len(self.fit_texts) != len(self.fit_row_ids):
            raise ValueError("embedding proof fit text binding changed length")
        self.expected_fit_treatment = _finite_vector(
            expected_fit_treatment,
            name="canonical fit treatment",
            length=len(self.fit_row_ids),
        )
        self.expected_fit_outcome = _finite_vector(
            expected_fit_outcome,
            name="canonical fit outcome",
            length=len(self.fit_row_ids),
        )
        self.text_column = str(text_column)
        self.outcome_type = str(outcome_type).lower()
        if type(embedding_provider) is not BoundSpentFrozenChunkEmbeddingProvider:
            raise TypeError("embedding proof requires the exact bound frozen-cache provider")
        if tuple(map(int, embedding_provider.row_ids)) != self.fit_row_ids:
            raise ValueError("embedding proof provider must be bound to fit rows only")
        self.embedding_provider = embedding_provider
        self.embedding_config = _embedding_config_mapping(embedding_config)
        if isinstance(seed, bool):
            raise TypeError("embedding native capture seed must be an integer")
        self.seed = int(seed)
        self.semantic_policy = build_semantic_retrieval_training_only_policy(
            fit_row_ids=self.fit_row_ids,
            outer_fold=self.outer_fold,
            inner_fold=self.inner_fold,
            configured_fold_count=int(tfidf_nested_calibration_folds),
            seed=self.seed,
        )
        self._store = _ArrayStore()
        self._kmeans_state: dict[str, Any] | None = None
        self._svd_states: list[dict[str, Any]] = []
        self._build_state: dict[str, Any] | None = None
        self._registered_fit_outputs: dict[str, Any] | None = None
        self._raw_evidence: dict[str, Any] | None = None
        self._semantic_bundle: dict[str, dict[str, Any]] | None = None
        self._finalized = False

    def record_registered_fit_outputs(
        self,
        *,
        fit_row_ids: Sequence[int],
        treatment: Any,
        outcome: Any,
        pseudo_target: Any,
        t_resid: Any,
        pseudo_target_names: Sequence[str] | None,
    ) -> None:
        """Bind the native runner's fit outputs before generator observation."""

        if self._finalized or self._registered_fit_outputs is not None:
            raise RuntimeError(
                "embedding registered fit outputs were emitted twice or after finalization"
            )
        if tuple(map(int, fit_row_ids)) != self.fit_row_ids:
            raise ValueError("embedding registered fit outputs changed exact fit row order")
        row_count = len(self.fit_row_ids)
        treatment_array = _finite_vector(
            treatment,
            name="registered fit treatment",
            length=row_count,
        )
        outcome_array = _finite_vector(
            outcome,
            name="registered fit outcome",
            length=row_count,
        )
        if not np.array_equal(treatment_array, self.expected_fit_treatment):
            raise ValueError("embedding registered treatment differs from canonical fit labels")
        if not np.array_equal(outcome_array, self.expected_fit_outcome):
            raise ValueError("embedding registered outcome differs from canonical fit labels")
        named_targets = _named_pseudo_targets(
            pseudo_target,
            t_resid,
            pseudo_target_names,
        )
        if not named_targets:
            raise ValueError("embedding registered fit outputs have no pseudo-target view")
        names: list[str] = []
        target_keys: list[str] = []
        residual_keys: list[str] = []
        for index, (name, target, residual) in enumerate(named_targets):
            name = str(name)
            if not name or name in names:
                raise ValueError(
                    "embedding registered pseudo-target names must be nonempty and unique"
                )
            names.append(name)
            target_keys.append(
                self._store.add(
                    f"registered_pseudo_target_{index:04d}",
                    _finite_vector(
                        target,
                        name=f"registered pseudo_target[{name}]",
                        length=row_count,
                    ),
                )
            )
            residual_keys.append(
                self._store.add(
                    f"registered_t_resid_{index:04d}",
                    _finite_vector(
                        residual,
                        name=f"registered t_resid[{name}]",
                        length=row_count,
                    ),
                )
            )
        self._registered_fit_outputs = {
            "fit_row_ids": list(self.fit_row_ids),
            "fit_row_order_fingerprint": _row_fingerprint(self.fit_row_ids),
            "treatment": self._store.add("registered_treatment", treatment_array),
            "outcome": self._store.add("registered_outcome", outcome_array),
            "pseudo_target_names": names,
            "pseudo_target_arrays": target_keys,
            "t_resid_arrays": residual_keys,
            "emitted_by": (
                "MultiModelForestStage1Runner._build_primary_embedding_contrast_evidence"
            ),
        }

    def record_cluster_kmeans(
        self,
        *,
        fit_row_ids: Sequence[int],
        usable_mask: Sequence[bool],
        cluster_labels: Sequence[int],
        cluster_centers: Any,
        cluster_counts: Sequence[int],
        n_iter: int,
        inertia: float,
        parameters: Mapping[str, Any],
    ) -> None:
        if self._finalized or self._kmeans_state is not None:
            raise RuntimeError("embedding KMeans state was emitted twice or after finalization")
        labels = np.asarray(cluster_labels, dtype=np.int64)
        usable = np.asarray(usable_mask, dtype=np.bool_)
        centers = np.asarray(cluster_centers, dtype=np.float64)
        counts = np.asarray(cluster_counts, dtype=np.int64)
        if (
            tuple(map(int, fit_row_ids)) != self.fit_row_ids
            or labels.shape != usable.shape
            or labels.shape != (len(self.fit_row_ids),)
        ):
            raise ValueError("embedding KMeans state changed exact fit length")
        self._kmeans_state = {
            "fit_row_ids": list(map(int, fit_row_ids)),
            "parameters": _clone(parameters),
            "usable_mask": self._store.add("cluster_kmeans_usable_mask", usable),
            "cluster_labels": self._store.add("cluster_kmeans_labels", labels),
            "cluster_centers": self._store.add("cluster_kmeans_centers", centers),
            "cluster_counts": self._store.add("cluster_kmeans_counts", counts),
            "n_iter": int(n_iter),
            "inertia": float(inertia),
        }

    def record_cluster_svd(
        self,
        *,
        family_key: str,
        item_cluster_ids: Sequence[int],
        weighted_matrix: Any,
        singular_values: Any,
        components: Any,
    ) -> None:
        if self._finalized:
            raise RuntimeError("embedding SVD state was emitted after finalization")
        family_key = str(family_key)
        if family_key not in {"treatment", "residualized_interaction"} or any(
            row["family_key"] == family_key for row in self._svd_states
        ):
            raise ValueError("embedding SVD family is unknown or duplicated")
        index = len(self._svd_states)
        matrix = np.asarray(weighted_matrix, dtype=np.float64)
        values = np.asarray(singular_values, dtype=np.float64)
        vectors = np.asarray(components, dtype=np.float64)
        if matrix.ndim != 2 or values.ndim != 1 or vectors.ndim != 2:
            raise ValueError("embedding SVD state has invalid dimensions")
        self._svd_states.append(
            {
                "family_key": family_key,
                "item_cluster_ids": list(map(int, item_cluster_ids)),
                "weighted_matrix": self._store.add(f"cluster_svd_{index}_matrix", matrix),
                "singular_values": self._store.add(f"cluster_svd_{index}_values", values),
                "components": self._store.add(f"cluster_svd_{index}_components", vectors),
            }
        )

    def record_build(
        self,
        *,
        generator: _FrozenCacheEmbeddingEvidenceGenerator,
        discovery_df: pd.DataFrame,
        y: Any,
        t: Any,
        pseudo_target: Any,
        t_resid: Any,
        pseudo_target_names: Sequence[str] | None,
        importance: Mapping[str, Any] | None,
        evidence: Mapping[str, Any],
    ) -> None:
        if self._finalized or self._build_state is not None:
            raise RuntimeError("embedding native build was emitted twice or after finalization")
        if type(generator) is not _FrozenCacheEmbeddingEvidenceGenerator:
            raise TypeError("embedding native capture requires the exact frozen-cache generator")
        if getattr(generator, "_spent_provider", None) is not self.embedding_provider:
            raise ValueError("embedding native generator changed its bound cache provider")
        if _embedding_config_mapping(generator.embedding_config) != self.embedding_config:
            raise ValueError("embedding native generator changed its effective configuration")
        if "_oci_row_id" not in discovery_df:
            raise ValueError("embedding native discovery frame has no canonical row IDs")
        discovery_rows = tuple(discovery_df["_oci_row_id"].astype(int).tolist())
        if discovery_rows != self.fit_row_ids:
            raise ValueError("embedding native build changed exact fit row order")
        row_count = len(self.fit_row_ids)
        y_array = _finite_vector(y, name="embedding outcome", length=row_count)
        t_array = _finite_vector(t, name="embedding treatment", length=row_count)
        named_targets = _named_pseudo_targets(pseudo_target, t_resid, pseudo_target_names)
        if not named_targets:
            raise ValueError("embedding native build has no pseudo-target view")
        registered = self._registered_fit_outputs
        if not isinstance(registered, Mapping):
            raise RuntimeError(
                "embedding native runner did not register fit outputs before generator build"
            )
        if (
            registered.get("fit_row_ids") != list(discovery_rows)
            or not np.array_equal(
                self._store.arrays[str(registered["treatment"])],
                t_array,
            )
            or not np.array_equal(
                self._store.arrays[str(registered["outcome"])],
                y_array,
            )
            or registered.get("pseudo_target_names")
            != [str(name) for name, _target, _residual in named_targets]
        ):
            raise ValueError("embedding generator inputs differ from registered native fit outputs")
        for index, (_name, target, residual) in enumerate(named_targets):
            if not np.array_equal(
                self._store.arrays[str(registered["pseudo_target_arrays"][index])],
                _finite_vector(
                    target,
                    name=f"generator pseudo_target[{index}]",
                    length=row_count,
                ),
            ):
                raise ValueError(
                    "embedding pseudo-target differs from registered native fit output"
                )
            if not np.array_equal(
                self._store.arrays[str(registered["t_resid_arrays"][index])],
                _finite_vector(
                    residual,
                    name=f"generator t_resid[{index}]",
                    length=row_count,
                ),
            ):
                raise ValueError(
                    "embedding treatment residual differs from registered native fit output"
                )
        names: list[str] = []
        target_keys: list[str] = []
        residual_keys: list[str] = []
        for index, (name, target, residual) in enumerate(named_targets):
            name = str(name)
            if not name or name in names:
                raise ValueError("embedding pseudo-target names must be nonempty and unique")
            names.append(name)
            target_keys.append(
                self._store.add(
                    f"pseudo_target_{index:04d}",
                    _finite_vector(target, name=f"pseudo_target[{name}]", length=row_count),
                )
            )
            residual_keys.append(
                self._store.add(
                    f"t_resid_{index:04d}",
                    _finite_vector(residual, name=f"t_resid[{name}]", length=row_count),
                )
            )
        residual_columns = [
            str(column)
            for column in self.embedding_config.get("residualize_columns") or ()
            if str(column) in discovery_df.columns
        ]
        projection = {"_oci_row_id": list(discovery_rows)}
        for column in residual_columns:
            projection[column] = _clone(discovery_df[column].tolist())
        raw = _clone(dict(evidence))
        if not raw.get("contrasts"):
            raise ValueError("embedding native build emitted no contrasts")
        for contrast in raw["contrasts"]:
            if not isinstance(contrast, Mapping):
                raise ValueError("embedding native build emitted a malformed contrast")
            for key in ("positive_external_chunks", "negative_external_chunks"):
                if contrast.get(key):
                    raise ValueError("embedding native build emitted forbidden external evidence")
            for key in ("positive_aligned_chunks", "negative_aligned_chunks"):
                for row in contrast.get(key) or ():
                    if int(row.get("row_id")) not in set(self.fit_row_ids):
                        raise ValueError("embedding retrieval escaped the exact fit scope")
        semantic_bundle = semantic_retrieval_projection_bundle(
            raw,
            policy=self.semantic_policy,
        )
        if not semantic_bundle["full"].get("contrasts"):
            raise ValueError("semantic retrieval emitted no exhaustive contrast evidence")
        self._build_state = {
            "discovery_projection": projection,
            "residualize_columns_present": residual_columns,
            "outcome": self._store.add("outcome", y_array),
            "treatment": self._store.add("treatment", t_array),
            "pseudo_target_names": names,
            "pseudo_target_arrays": target_keys,
            "t_resid_arrays": residual_keys,
            "importance_sha256": _sha256_json(dict(importance or {})),
        }
        self._raw_evidence = raw
        self._semantic_bundle = semantic_bundle

    def finalize(self) -> Mapping[str, Any]:
        if self._finalized:
            raise RuntimeError("embedding proof capture is already finalized")
        self._finalized = True
        if (
            self._registered_fit_outputs is None
            or self._build_state is None
            or self._raw_evidence is None
            or self._semantic_bundle is None
        ):
            raise RuntimeError("embedding proof capture observed no native build")
        if self.embedding_config.get("include_cluster_contrast_vectors") is True and (
            self._kmeans_state is None or not self._svd_states
        ):
            raise RuntimeError("clustered embedding proof has no actual KMeans/SVD state")
        self.artifact_dir.mkdir(parents=True, exist_ok=False)
        arrays_path = self.artifact_dir / "arrays.npz"
        _atomic_write_npz(arrays_path, self._store.arrays)
        evidence_files = {
            "raw_embedding_evidence": "raw_embedding_evidence.json",
            "semantic_full_scope_evidence": "semantic_full_scope_evidence.json",
            "semantic_model_replay_canary": "semantic_model_replay_canary.json",
            "semantic_calibration_replay_canary": "semantic_calibration_replay_canary.json",
        }
        values = {
            "raw_embedding_evidence": self._raw_evidence,
            "semantic_full_scope_evidence": self._semantic_bundle["full"],
            "semantic_model_replay_canary": self._semantic_bundle["model_canary"],
            "semantic_calibration_replay_canary": self._semantic_bundle["calibration_canary"],
        }
        evidence_inventory: dict[str, dict[str, Any]] = {}
        for key, filename in evidence_files.items():
            path = self.artifact_dir / filename
            _atomic_write_json(path, values[key])
            evidence_inventory[key] = {
                "filename": filename,
                "sha256": _sha256_file(path),
                "content_sha256": _sha256_json(values[key]),
            }
        body = {
            "schema_version": EMBEDDING_NATIVE_CAPTURE_SCHEMA,
            "scope_id": self.scope_id,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "fit_row_ids": list(self.fit_row_ids),
            "heldout_row_ids": list(self.heldout_row_ids),
            "fit_row_fingerprint": _row_fingerprint(self.fit_row_ids),
            "heldout_row_fingerprint": _row_fingerprint(self.heldout_row_ids),
            "fit_text_sha256": _text_sha256(self.fit_row_ids, self.fit_texts),
            "text_column": self.text_column,
            "outcome_type": self.outcome_type,
            "seed": self.seed,
            "embedding_config": self.embedding_config,
            "embedding_provider_identity": _clone(self.embedding_provider.identity()),
            "fit_cache_row_inventory": _cache_row_inventory(
                self.embedding_provider,
                self.fit_row_ids,
            ),
            "tfidf_training_scope_policy": self.semantic_policy,
            "registered_fit_outputs": self._registered_fit_outputs,
            "build": self._build_state,
            "cluster_kmeans": self._kmeans_state,
            "cluster_svds": self._svd_states,
            "array_inventory": self._store.inventory,
            "array_file": arrays_path.name,
            "array_file_sha256": _sha256_file(arrays_path),
            "evidence_inventory": evidence_inventory,
            "heldout_columns_read": ["_oci_row_id"],
            "heldout_text_accessed": False,
            "heldout_labels_accessed": False,
            "heldout_transform_performed": False,
            "oracle_fields_accessed": False,
            "secrets_accessed": False,
            "external_retrieval_used": False,
            "executable_serialization_used": False,
            "joblib_or_pickle_used": False,
        }
        metadata = {**body, "content_sha256": _sha256_json(body)}
        _atomic_write_json(self.artifact_dir / "metadata.json", metadata)
        return validate_embedding_native_capture(
            self.artifact_dir,
            embedding_provider=self.embedding_provider,
            fit_texts=self.fit_texts,
            expected_fit_treatment=self.expected_fit_treatment,
            expected_fit_outcome=self.expected_fit_outcome,
            expected_scope_id=self.scope_id,
            expected_fit_row_ids=self.fit_row_ids,
            expected_heldout_row_ids=self.heldout_row_ids,
        )


class _ReplayObserver:
    def __init__(self) -> None:
        self.kmeans: dict[str, Any] | None = None
        self.svds: list[dict[str, Any]] = []
        self.evidence: dict[str, Any] | None = None

    def record_cluster_kmeans(self, **kwargs: Any) -> None:
        if self.kmeans is not None:
            raise RuntimeError("replay emitted KMeans state twice")
        self.kmeans = {
            "fit_row_ids": list(map(int, kwargs["fit_row_ids"])),
            "parameters": _clone(kwargs["parameters"]),
            "usable_mask": np.asarray(kwargs["usable_mask"], dtype=np.bool_),
            "cluster_labels": np.asarray(kwargs["cluster_labels"], dtype=np.int64),
            "cluster_centers": np.asarray(kwargs["cluster_centers"], dtype=np.float64),
            "cluster_counts": np.asarray(kwargs["cluster_counts"], dtype=np.int64),
            "n_iter": int(kwargs["n_iter"]),
            "inertia": float(kwargs["inertia"]),
        }

    def record_cluster_svd(self, **kwargs: Any) -> None:
        self.svds.append(
            {
                "family_key": str(kwargs["family_key"]),
                "item_cluster_ids": list(map(int, kwargs["item_cluster_ids"])),
                "weighted_matrix": np.asarray(kwargs["weighted_matrix"], dtype=np.float64),
                "singular_values": np.asarray(kwargs["singular_values"], dtype=np.float64),
                "components": np.asarray(kwargs["components"], dtype=np.float64),
            }
        )

    def record_build(self, **kwargs: Any) -> None:
        if self.evidence is not None:
            raise RuntimeError("replay emitted native build twice")
        self.evidence = _clone(dict(kwargs["evidence"]))


def _load_capture(
    artifact_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    root = Path(artifact_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("embedding native capture must be one real directory")
    children = tuple(sorted(item.name for item in root.iterdir()))
    if children != _ARTIFACT_FILENAMES or any(item.is_symlink() for item in root.iterdir()):
        raise ValueError("embedding native capture has an open or symlinked layout")
    if any(item.name.lower().endswith(_FORBIDDEN_SUFFIXES) for item in root.rglob("*")):
        raise ValueError("embedding native capture contains executable serialization")
    metadata = _read_json_object_reject_duplicates(
        root / "metadata.json",
        field_name="embedding native capture metadata",
    )
    metadata_fields = {
        "schema_version",
        "scope_id",
        "outer_fold",
        "inner_fold",
        "fit_row_ids",
        "heldout_row_ids",
        "fit_row_fingerprint",
        "heldout_row_fingerprint",
        "fit_text_sha256",
        "text_column",
        "outcome_type",
        "seed",
        "embedding_config",
        "embedding_provider_identity",
        "fit_cache_row_inventory",
        "tfidf_training_scope_policy",
        "registered_fit_outputs",
        "build",
        "cluster_kmeans",
        "cluster_svds",
        "array_inventory",
        "array_file",
        "array_file_sha256",
        "evidence_inventory",
        "heldout_columns_read",
        "heldout_text_accessed",
        "heldout_labels_accessed",
        "heldout_transform_performed",
        "oracle_fields_accessed",
        "secrets_accessed",
        "external_retrieval_used",
        "executable_serialization_used",
        "joblib_or_pickle_used",
        "content_sha256",
    }
    body = {key: value for key, value in metadata.items() if key != "content_sha256"}
    if (
        set(metadata) != metadata_fields
        or metadata.get("schema_version") != EMBEDDING_NATIVE_CAPTURE_SCHEMA
        or metadata.get("content_sha256") != _sha256_json(body)
        or metadata.get("heldout_columns_read") != ["_oci_row_id"]
        or metadata.get("heldout_text_accessed") is not False
        or metadata.get("heldout_labels_accessed") is not False
        or metadata.get("heldout_transform_performed") is not False
        or metadata.get("oracle_fields_accessed") is not False
        or metadata.get("secrets_accessed") is not False
        or metadata.get("external_retrieval_used") is not False
        or metadata.get("executable_serialization_used") is not False
        or metadata.get("joblib_or_pickle_used") is not False
    ):
        raise ValueError("embedding native capture has an invalid closed envelope")
    arrays_path = root / str(metadata.get("array_file") or "")
    inventory = metadata.get("array_inventory")
    if (
        arrays_path.name != "arrays.npz"
        or metadata.get("array_file_sha256") != _sha256_file(arrays_path)
        or not isinstance(inventory, Mapping)
        or not inventory
    ):
        raise ValueError("embedding native capture has an invalid array binding")
    try:
        with np.load(arrays_path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(inventory):
                raise ValueError("embedding native capture array inventory is incomplete")
            arrays = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("embedding"):
            raise
        raise ValueError("embedding native capture NPZ is unsafe or malformed") from exc
    for key, record in inventory.items():
        array = arrays[str(key)]
        if (
            not isinstance(record, Mapping)
            or record.get("dtype") != array.dtype.str
            or record.get("shape") != list(array.shape)
            or record.get("content_sha256") != _array_sha256(array)
        ):
            raise RuntimeError(f"embedding native capture array changed: {key}")
    evidence_inventory = metadata.get("evidence_inventory")
    if not isinstance(evidence_inventory, Mapping) or set(evidence_inventory) != {
        "raw_embedding_evidence",
        "semantic_full_scope_evidence",
        "semantic_model_replay_canary",
        "semantic_calibration_replay_canary",
    }:
        raise ValueError("embedding native capture evidence inventory is incomplete")
    evidence: dict[str, dict[str, Any]] = {}
    for key, record in evidence_inventory.items():
        if not isinstance(record, Mapping):
            raise ValueError("embedding native evidence inventory row is malformed")
        path = root / str(record.get("filename") or "")
        if path.parent != root or not path.is_file() or record.get("sha256") != _sha256_file(path):
            raise ValueError("embedding native evidence file binding changed")
        value = _read_json_object_reject_duplicates(
            path,
            field_name="embedding native evidence file",
        )
        if not isinstance(value, dict) or record.get("content_sha256") != _sha256_json(value):
            raise ValueError("embedding native evidence content binding changed")
        evidence[str(key)] = value
    return metadata, arrays, evidence


def _assert_array_close(observed: Any, expected: Any, *, name: str) -> None:
    left = np.asarray(observed)
    right = np.asarray(expected)
    if left.shape != right.shape:
        raise RuntimeError(f"embedding native replay shape differs for {name}")
    if left.dtype.kind in "biu" and right.dtype.kind in "biu":
        equal = np.array_equal(left, right)
    else:
        equal = np.allclose(
            left.astype(np.float64),
            right.astype(np.float64),
            rtol=2e-7,
            atol=2e-8,
        )
    if not equal:
        raise RuntimeError(f"embedding native replay differs for {name}")


def _replay_generator(
    *,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    provider: BoundSpentFrozenChunkEmbeddingProvider,
) -> tuple[dict[str, Any], _ReplayObserver]:
    fit_rows = tuple(map(int, metadata["fit_row_ids"]))
    chunks_by_row = provider.chunk_texts(fit_rows)
    matrices = provider.chunk_matrices(fit_rows)
    offsets = np.zeros(len(fit_rows) + 1, dtype=np.int64)
    cursor = 0
    flat: list[np.ndarray] = []
    for index, matrix in enumerate(matrices):
        matrix = np.asarray(matrix, dtype=np.float32)
        cursor += int(matrix.shape[0])
        offsets[index + 1] = cursor
        flat.append(matrix)
    embedding_dimension = int(flat[0].shape[1]) if flat else 0
    flat_embeddings = (
        np.vstack(flat).astype(np.float32, copy=False)
        if flat
        else np.empty((0, embedding_dimension), dtype=np.float32)
    )
    generator = object.__new__(_FrozenCacheEmbeddingEvidenceGenerator)
    generator.config = SimpleNamespace(
        outcome_type=str(metadata["outcome_type"]),
        text_column=str(metadata["text_column"]),
    )
    generator.embedding_config = EmbeddingContrastDiscoveryConfig(
        **copy.deepcopy(dict(metadata["embedding_config"]))
    )
    generator.output_dir = Path(".")
    generator.embedding_provider = None
    generator.precompute_devices = []
    generator._prepared = True
    generator._row_ids = list(fit_rows)
    generator._row_id_to_position = {row_id: index for index, row_id in enumerate(fit_rows)}
    generator._chunks_by_position = [list(map(str, chunks)) for chunks in chunks_by_row]
    generator._flat_embeddings = flat_embeddings
    generator._offsets = offsets
    generator._cache = None
    generator._cache_dir = None
    generator._chunk_cache_reused = True
    generator._concept_probe_skip_reason = None
    generator._external_corpora = []
    generator._spent_provider = provider
    observer = _ReplayObserver()
    generator._native_embedding_proof_observer = observer
    build = metadata["build"]
    discovery_df = pd.DataFrame(copy.deepcopy(dict(build["discovery_projection"])))
    pseudo_targets = [arrays[str(key)] for key in build["pseudo_target_arrays"]]
    residuals = [arrays[str(key)] for key in build["t_resid_arrays"]]
    evidence = generator.build_evidence(
        discovery_df=discovery_df,
        y=arrays[str(build["outcome"])],
        t=arrays[str(build["treatment"])],
        pseudo_target=pseudo_targets,
        t_resid=residuals,
        pseudo_target_names=build["pseudo_target_names"],
        importance={},
    )
    return _clone(evidence), observer


def validate_embedding_native_capture(
    artifact_dir: Path,
    *,
    embedding_provider: BoundSpentFrozenChunkEmbeddingProvider,
    fit_texts: Sequence[str],
    expected_fit_treatment: Sequence[float] | None = None,
    expected_fit_outcome: Sequence[float] | None = None,
    expected_discovery_projection: Mapping[str, Sequence[Any]] | None = None,
    expected_scope_id: str | None = None,
    expected_fit_row_ids: Sequence[int] | None = None,
    expected_heldout_row_ids: Sequence[int] | None = None,
) -> Mapping[str, Any]:
    """Authenticate and numerically replay one closed embedding proof capture."""

    metadata, arrays, evidence = _load_capture(Path(artifact_dir))
    fit_rows = tuple(map(int, metadata.get("fit_row_ids") or ()))
    heldout_rows = tuple(map(int, metadata.get("heldout_row_ids") or ()))
    if (
        not fit_rows
        or not heldout_rows
        or set(fit_rows) & set(heldout_rows)
        or metadata.get("fit_row_fingerprint") != _row_fingerprint(fit_rows)
        or metadata.get("heldout_row_fingerprint") != _row_fingerprint(heldout_rows)
    ):
        raise ValueError("embedding native capture has invalid exact row bindings")
    if expected_scope_id is not None and metadata.get("scope_id") != str(expected_scope_id):
        raise ValueError("embedding native capture belongs to another scope")
    if expected_fit_row_ids is not None and fit_rows != tuple(map(int, expected_fit_row_ids)):
        raise ValueError("embedding native capture changed exact fit row order")
    if expected_heldout_row_ids is not None and heldout_rows != tuple(
        map(int, expected_heldout_row_ids)
    ):
        raise ValueError("embedding native capture changed exact held-out row order")
    if type(embedding_provider) is not BoundSpentFrozenChunkEmbeddingProvider:
        raise TypeError("embedding native replay requires the exact frozen-cache provider")
    if tuple(map(int, embedding_provider.row_ids)) != fit_rows:
        raise ValueError("embedding native replay provider is not fit-only or changed order")
    fit_texts = tuple(map(str, fit_texts))
    if metadata.get("fit_text_sha256") != _text_sha256(fit_rows, fit_texts):
        raise ValueError("embedding native capture fit text binding changed")
    if metadata.get("embedding_provider_identity") != _clone(embedding_provider.identity()):
        raise ValueError("embedding native capture provider identity changed")
    if metadata.get("fit_cache_row_inventory") != _cache_row_inventory(
        embedding_provider,
        fit_rows,
    ):
        raise RuntimeError("embedding native capture frozen-cache rows changed")
    registered = metadata.get("registered_fit_outputs")
    build = metadata.get("build")
    if not isinstance(registered, Mapping) or not isinstance(build, Mapping):
        raise ValueError("embedding native capture has no registered fit-output lineage")
    registered_fields = {
        "fit_row_ids",
        "fit_row_order_fingerprint",
        "treatment",
        "outcome",
        "pseudo_target_names",
        "pseudo_target_arrays",
        "t_resid_arrays",
        "emitted_by",
    }
    build_fields = {
        "discovery_projection",
        "residualize_columns_present",
        "outcome",
        "treatment",
        "pseudo_target_names",
        "pseudo_target_arrays",
        "t_resid_arrays",
        "importance_sha256",
    }
    if (
        set(registered) != registered_fields
        or set(build) != build_fields
        or not isinstance(registered.get("pseudo_target_names"), list)
        or not registered.get("pseudo_target_names")
        or not isinstance(registered.get("pseudo_target_arrays"), list)
        or not isinstance(registered.get("t_resid_arrays"), list)
        or not isinstance(build.get("discovery_projection"), Mapping)
        or registered.get("fit_row_ids") != list(fit_rows)
        or registered.get("fit_row_order_fingerprint") != _row_fingerprint(fit_rows)
        or registered.get("emitted_by")
        != "MultiModelForestStage1Runner._build_primary_embedding_contrast_evidence"
        or registered.get("pseudo_target_names") != build.get("pseudo_target_names")
        or len(registered.get("pseudo_target_arrays") or ())
        != len(build.get("pseudo_target_arrays") or ())
        or len(registered.get("t_resid_arrays") or ()) != len(build.get("t_resid_arrays") or ())
    ):
        raise ValueError("embedding native registered fit-output lineage changed")
    _assert_array_close(
        arrays[str(registered.get("treatment"))],
        arrays[str(build.get("treatment"))],
        name="registered_fit_outputs.treatment",
    )
    _assert_array_close(
        arrays[str(registered.get("outcome"))],
        arrays[str(build.get("outcome"))],
        name="registered_fit_outputs.outcome",
    )
    for index, (registered_key, build_key) in enumerate(
        zip(
            registered.get("pseudo_target_arrays") or (),
            build.get("pseudo_target_arrays") or (),
        )
    ):
        _assert_array_close(
            arrays[str(registered_key)],
            arrays[str(build_key)],
            name=f"registered_fit_outputs.pseudo_target[{index}]",
        )
    for index, (registered_key, build_key) in enumerate(
        zip(
            registered.get("t_resid_arrays") or (),
            build.get("t_resid_arrays") or (),
        )
    ):
        _assert_array_close(
            arrays[str(registered_key)],
            arrays[str(build_key)],
            name=f"registered_fit_outputs.t_resid[{index}]",
        )
    if (expected_fit_treatment is None) != (expected_fit_outcome is None):
        raise ValueError("canonical fit treatment and outcome must be supplied together")
    if expected_fit_treatment is not None and expected_fit_outcome is not None:
        _assert_array_close(
            arrays[str(registered["treatment"])],
            _finite_vector(
                expected_fit_treatment,
                name="expected canonical fit treatment",
                length=len(fit_rows),
            ),
            name="canonical_fit.treatment",
        )
        _assert_array_close(
            arrays[str(registered["outcome"])],
            _finite_vector(
                expected_fit_outcome,
                name="expected canonical fit outcome",
                length=len(fit_rows),
            ),
            name="canonical_fit.outcome",
        )
    if expected_discovery_projection is not None and build.get("discovery_projection") != _clone(
        dict(expected_discovery_projection)
    ):
        raise RuntimeError("embedding native discovery projection differs from canonical data")
    policy = metadata.get("tfidf_training_scope_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("embedding native capture has no semantic training-only policy")
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("embedding native capture seed is invalid")
    expected_partition = build_semantic_retrieval_training_only_policy(
        fit_row_ids=fit_rows,
        outer_fold=int(metadata["outer_fold"]),
        inner_fold=int(metadata["inner_fold"]),
        configured_fold_count=int(policy.get("configured_fold_count", 0)),
        seed=seed,
    )
    if policy != expected_partition:
        raise ValueError("semantic retrieval training-only partition or policy changed")
    replayed, observer = _replay_generator(
        metadata=metadata,
        arrays=arrays,
        provider=embedding_provider,
    )
    if replayed != evidence["raw_embedding_evidence"] or observer.evidence != replayed:
        raise RuntimeError("embedding native generator output differs on replay")
    semantic = semantic_retrieval_projection_bundle(replayed, policy=policy)
    if (
        semantic["full"] != evidence["semantic_full_scope_evidence"]
        or semantic["model_canary"] != evidence["semantic_model_replay_canary"]
        or semantic["calibration_canary"] != evidence["semantic_calibration_replay_canary"]
    ):
        raise RuntimeError("semantic retrieval exhaustive projection differs on replay")
    captured_kmeans = metadata.get("cluster_kmeans")
    if not isinstance(captured_kmeans, Mapping) or observer.kmeans is None:
        raise ValueError("embedding clustered proof has no replayable KMeans state")
    if (
        captured_kmeans.get("fit_row_ids") != observer.kmeans["fit_row_ids"]
        or captured_kmeans.get("parameters") != observer.kmeans["parameters"]
        or int(captured_kmeans.get("n_iter", -1)) != observer.kmeans["n_iter"]
        or not np.isclose(
            float(captured_kmeans.get("inertia")),
            observer.kmeans["inertia"],
            rtol=2e-7,
            atol=2e-8,
        )
    ):
        raise RuntimeError("embedding native KMeans metadata differs on replay")
    for field in ("usable_mask", "cluster_labels", "cluster_centers", "cluster_counts"):
        _assert_array_close(
            arrays[str(captured_kmeans[field])],
            observer.kmeans[field],
            name=f"cluster_kmeans.{field}",
        )
    captured_svds = metadata.get("cluster_svds")
    if not isinstance(captured_svds, list) or len(captured_svds) != len(observer.svds):
        raise ValueError("embedding native SVD state inventory differs on replay")
    for index, (captured, replay_svd) in enumerate(zip(captured_svds, observer.svds)):
        if (
            captured.get("family_key") != replay_svd["family_key"]
            or captured.get("item_cluster_ids") != replay_svd["item_cluster_ids"]
        ):
            raise RuntimeError("embedding native SVD metadata differs on replay")
        for field in ("weighted_matrix", "singular_values", "components"):
            _assert_array_close(
                arrays[str(captured[field])],
                replay_svd[field],
                name=f"cluster_svd[{index}].{field}",
            )
    return _clone(metadata)


__all__ = [
    "EMBEDDING_NATIVE_CAPTURE_SCHEMA",
    "SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA",
    "NativeEmbeddingProofCaptureSink",
    "build_semantic_retrieval_training_only_policy",
    "semantic_retrieval_projection_bundle",
    "validate_embedding_native_capture",
]
