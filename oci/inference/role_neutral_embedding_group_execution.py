"""Two-phase role-neutral execution for the three embedding evidence families.

The module deliberately keeps the clustered fit in the preflight phase.  A
physical Stage 1 worker receives an authenticated, sealed copy of the
preflight KMeans/SVD state, proves that every array is the array named by the
canonical preflight audit, and reuses it.  It never refits clustering merely
to compare a second fit with the preflight.

The fit phase can read only the canonical owner's ordered fit rows, complete
fit text, fit labels/targets, and its row-scoped frozen embedding provider.
It publishes three fit-only family seals and every deduplicated logical
alias before the primary owner's held-out loader can be invoked.  The loader
has no treatment/outcome field.  Every physical owner receives its own
complete held-out numerical transform; deduplicated aliases remain fit-only
references.

All numerical payloads are individual, non-object ``.npy`` files loaded with
``allow_pickle=False``.  Strings and indexes use closed canonical JSON.  The
configured cache chunk cap and tokenizer limit must be proved nonbinding, and
configured retrieval/term limits abort on overflow instead of selecting or
dropping evidence.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ..models.concept_embedding_utils import chunk_text_words
from .all_evidence_discovery_interfaces import (
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HETEROGENEITY_AXIS,
    OUTCOME_AXIS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TREATMENT_AXIS,
)
from .lossless_stage1_evidence_catalog import (
    NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
    SEMANTIC_RETRIEVAL_DERIVATION,
)
from .production_stage1_cluster_preflight_artifact import (
    ProductionStage1ClusterPreflightArtifact,
)
from .production_stage1_cluster_preflight_artifact_v2 import (
    PortableProductionStage1ClusterPreflightArtifact,
)
from .production_stage1_legacy_scope_fragments import (
    LEGACY_STAGE1_FIT_ONLY_FAMILY_SEAL_SCHEMA,
)
from .production_stage1_scope_scheduler import Stage1ScopePlan, Stage1ScopeSpec
from .review_spent_evidence_provider import (
    BoundSpentFrozenChunkEmbeddingProvider,
)


ROLE_NEUTRAL_EMBEDDING_REQUEST_SCHEMA = (
    "production_role_neutral_embedding_physical_group_request_v1"
)
ROLE_NEUTRAL_EMBEDDING_CONFIG_SCHEMA = (
    "production_role_neutral_embedding_scientific_config_v5"
)
ROLE_NEUTRAL_EMBEDDING_CLUSTER_STATE_SCHEMA = (
    "production_canonical_clustered_preflight_scope_state_v2"
)
ROLE_NEUTRAL_EMBEDDING_CLUSTER_STATE_BUNDLE_SCHEMA = (
    "production_canonical_clustered_preflight_state_bundle_v2"
)
ROLE_NEUTRAL_EMBEDDING_FIT_STATE_SCHEMA = (
    "production_role_neutral_embedding_fit_state_v2"
)
ROLE_NEUTRAL_EMBEDDING_LOGICAL_VIEW_SCHEMA = (
    "production_role_neutral_embedding_logical_view_v1"
)
ROLE_NEUTRAL_EMBEDDING_EXECUTION_SCHEMA = (
    "production_role_neutral_embedding_group_execution_v1"
)

_CLUSTER_STATE_MANIFEST = "cluster_state_manifest.json"
_CLUSTER_STATE_BUNDLE_MANIFEST = "cluster_state_bundle_manifest.json"
_FIT_STATE_DIRECTORY = "fit_state"
_FIT_METADATA = "metadata.json"
_FIT_CHUNKS = "fit_source_chunks.json"
_FIT_VOCABULARY = "semantic_vocabulary.json"
_LOGICAL_VIEW_DIRECTORY = "logical_views"
_EXACT_DIRECTORY = "exact_transforms"
_TERMINAL_FILE = "execution_manifest.json"
_HEX = frozenset("0123456789abcdef")
_FAMILIES = (
    EMBEDDING_WHOLE_COHORT,
    EMBEDDING_CLUSTERED,
    TFIDF_SEMANTIC_RETRIEVAL,
)
_SEAL_FILENAMES = {
    EMBEDDING_WHOLE_COHORT: "fit_only_embedding_whole_cohort_seal.json",
    EMBEDDING_CLUSTERED: "fit_only_embedding_clustered_seal.json",
    TFIDF_SEMANTIC_RETRIEVAL: "fit_only_tfidf_semantic_retrieval_seal.json",
}
_WHOLE_CONTRAST_FAMILIES = frozenset(
    {
        "marginal",
        "marginal_confounder_average",
        "within_treatment_arm_outcome",
        "treatment_outcome_cell_interaction",
        "r_pseudo_target",
        "orthogonal_r_score",
        "residualized_treatment_outcome_cell_interaction",
    }
)
_CLUSTER_CONTRAST_AXES = {
    "cluster_local_treatment_contrast_basis": (TREATMENT_AXIS,),
    "cluster_local_residualized_interaction_contrast_basis": (
        HETEROGENEITY_AXIS,
    ),
}
_WORD_RE = re.compile(r"\S+")
_ClusterPreflightArtifact = (
    ProductionStage1ClusterPreflightArtifact
    | PortableProductionStage1ClusterPreflightArtifact
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


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"JSON contains duplicate key: {key}")
        output[key] = value
    return output


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    target = Path(path)
    digest_before, _ = _sha256_file(target)
    try:
        value = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    digest_after, _ = _sha256_file(target)
    if digest_before != digest_after:
        raise RuntimeError(f"{label} changed while it was decoded")
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> tuple[str, int]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"artifact is not one regular file: {target}")
    before = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise ValueError(f"artifact file is linked or nonregular: {target}")
    digest = hashlib.sha256()
    size = 0
    descriptor = os.open(
        target,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            size += len(block)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = target.stat(follow_symlinks=False)
    fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    expected = tuple(int(getattr(before, field)) for field in fields)
    if (
        tuple(int(getattr(opened, field)) for field in fields) != expected
        or tuple(int(getattr(closed, field)) for field in fields) != expected
        or tuple(int(getattr(after, field)) for field in fields) != expected
        or size != int(after.st_size)
    ):
        raise RuntimeError(f"artifact changed while hashing: {target}")
    return digest.hexdigest(), size


def _write_new_bytes(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace immutable artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, target)
        descriptor = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_bytes(
        path,
        (
            json.dumps(
                dict(value),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_new_npy(path: Path, value: np.ndarray) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace immutable array: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("embedding artifacts cannot use object arrays")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError("embedding artifacts cannot contain non-finite arrays")
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        suffix=".npy",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, target)
        descriptor = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _array_identity(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("array identity cannot represent object dtype")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def _array_registration(path: Path, *, relative_to: Path, value: np.ndarray) -> dict[str, Any]:
    digest, size = _sha256_file(path)
    identity = _array_identity(value)
    return {
        "relative_path": path.relative_to(relative_to).as_posix(),
        "file_sha256": digest,
        "size_bytes": size,
        **identity,
    }


def _read_registered_array(
    *,
    root: Path,
    registration: Mapping[str, Any],
    label: str,
) -> np.ndarray:
    required = {
        "relative_path",
        "file_sha256",
        "size_bytes",
        "dtype",
        "shape",
        "sha256",
    }
    if not isinstance(registration, Mapping) or set(registration) != required:
        raise ValueError(f"{label} has an invalid array registration")
    relative = Path(str(registration["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path escapes its artifact")
    path = root / relative
    digest, size = _sha256_file(path)
    if digest != registration["file_sha256"] or size != registration["size_bytes"]:
        raise ValueError(f"{label} file registration changed")
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"{label} is not a safe NPY array") from exc
    if value.dtype.hasobject or _array_identity(value) != {
        "dtype": registration["dtype"],
        "shape": registration["shape"],
        "sha256": registration["sha256"],
    }:
        raise ValueError(f"{label} array content changed")
    return value


def _row_order_fingerprint(row_ids: Sequence[int]) -> str:
    return _sha256_json([int(row_id) for row_id in row_ids])


def _ordered_text_sha256(row_ids: Sequence[int], texts: Sequence[str]) -> str:
    return _sha256_json(
        [
            {"row_id": int(row_id), "text": str(text)}
            for row_id, text in zip(row_ids, texts, strict=True)
        ]
    )


def _float_hex_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return _sha256_json([float(item).hex() for item in array])


def _axes_for_whole_contrast(contrast_family: str, name: str) -> tuple[str, ...]:
    family = str(contrast_family)
    if family == "marginal":
        if name == "treatment":
            return (TREATMENT_AXIS,)
        if name == "outcome":
            return (OUTCOME_AXIS,)
        raise ValueError("marginal embedding contrast must be treatment or outcome")
    if family == "marginal_confounder_average":
        return (TREATMENT_AXIS, OUTCOME_AXIS)
    if family == "within_treatment_arm_outcome":
        return (OUTCOME_AXIS,)
    if family in {
        "treatment_outcome_cell_interaction",
        "r_pseudo_target",
        "orthogonal_r_score",
        "residualized_treatment_outcome_cell_interaction",
    }:
        return (HETEROGENEITY_AXIS,)
    raise ValueError(f"unknown whole-cohort embedding contrast family: {family}")


@dataclass(frozen=True)
class EmbeddingContrastSpec:
    """One explicitly configured fit-side direction."""

    name: str
    contrast_family: str
    target_name: str
    sample_weight_target_name: str | None
    split_rule: str

    def __post_init__(self) -> None:
        for field_name in ("name", "contrast_family", "target_name", "split_rule"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"embedding contrast {field_name} must be nonempty")
            object.__setattr__(self, field_name, value)
        if self.sample_weight_target_name is not None:
            sample_weight_target_name = str(
                self.sample_weight_target_name
            ).strip()
            if not sample_weight_target_name:
                raise ValueError(
                    "embedding contrast sample_weight_target_name must be "
                    "null or nonempty"
                )
            object.__setattr__(
                self,
                "sample_weight_target_name",
                sample_weight_target_name,
            )
        if self.contrast_family not in _WHOLE_CONTRAST_FAMILIES:
            raise ValueError("embedding contrast family is not a whole-cohort family")
        if self.split_rule not in {
            "binary_zero_one",
            "stable_ordered_halves",
            "configured_quantile_tails",
            "treated_arm_outcome_cell_difference",
            "untreated_arm_outcome_cell_difference",
            "treatment_outcome_cell_difference_in_differences",
            "average_normalized_treatment_outcome_marginals",
            "cell_difference_in_differences_residualized_from_marginals",
        }:
            raise ValueError("embedding contrast split_rule is unsupported")
        if (
            self.sample_weight_target_name is not None
            and self.split_rule != "configured_quantile_tails"
        ):
            raise ValueError(
                "embedding sample weights are supported only for configured "
                "quantile-tail contrasts"
            )
        _axes_for_whole_contrast(self.contrast_family, self.name)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoleNeutralEmbeddingScientificConfig:
    """All scientific and capacity settings used by the embedding fit.

    There are intentionally no defaults.  Capacity fields are optional only
    when the caller explicitly supplies ``None``.  A finite capacity is a
    fail-on-overflow assertion; it is never a selection instruction.
    """

    contrasts: tuple[EmbeddingContrastSpec, ...]
    normalize_patient_embeddings: bool
    patient_embedding_pooling: str
    numeric_compute_dtype: str
    vector_norm_order: str
    direction_norm_epsilon: float
    pseudo_target_quantile: float
    pseudo_target_weighted: bool
    quantile_method: str
    minimum_contrast_side_rows: int
    lstsq_rcond: float | None
    lstsq_solution_policy: str
    semantic_input: str
    semantic_encoding: str
    semantic_decode_error: str
    semantic_preprocessor: None
    semantic_tokenizer: None
    semantic_analyzer: str
    semantic_ngram_min: int
    semantic_ngram_max: int
    semantic_token_pattern: str
    semantic_lowercase: bool
    semantic_strip_accents: str | None
    semantic_min_df: int
    semantic_max_df: float
    semantic_sublinear_tf: bool
    semantic_norm: str | None
    semantic_use_idf: bool
    semantic_smooth_idf: bool
    semantic_binary: bool
    semantic_dtype: str
    semantic_stop_words: str | tuple[str, ...] | None
    semantic_vocabulary: None
    semantic_max_features: None
    semantic_member_batch_size: int
    maximum_source_chunks_per_row: int | None
    maximum_retrieval_chunks_per_side: int | None
    maximum_semantic_terms: int | None
    overflow_policy: str

    def __post_init__(self) -> None:
        contrasts = tuple(self.contrasts)
        if not contrasts or any(not isinstance(value, EmbeddingContrastSpec) for value in contrasts):
            raise ValueError("embedding configuration requires explicit contrast specs")
        if len({value.name for value in contrasts}) != len(contrasts):
            raise ValueError("embedding contrast names must be unique")
        axes = {
            axis
            for contrast in contrasts
            for axis in _axes_for_whole_contrast(
                contrast.contrast_family,
                contrast.name,
            )
        }
        if not {TREATMENT_AXIS, OUTCOME_AXIS, HETEROGENEITY_AXIS}.issubset(axes):
            raise ValueError(
                "whole-cohort embedding contrasts must cover treatment, outcome, "
                "and heterogeneity axes"
            )
        object.__setattr__(self, "contrasts", contrasts)
        boolean_fields = (
            "normalize_patient_embeddings",
            "pseudo_target_weighted",
            "semantic_lowercase",
            "semantic_sublinear_tf",
            "semantic_use_idf",
            "semantic_smooth_idf",
            "semantic_binary",
        )
        for field_name in boolean_fields:
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be Boolean")
        if (
            not isinstance(self.patient_embedding_pooling, str)
            or self.patient_embedding_pooling != "arithmetic_mean"
        ):
            raise ValueError(
                "patient_embedding_pooling must be the explicitly supported "
                "'arithmetic_mean' policy"
            )
        if (
            not isinstance(self.numeric_compute_dtype, str)
            or self.numeric_compute_dtype != "float64"
        ):
            raise ValueError(
                "role-neutral embedding v5 supports only explicitly configured "
                "numeric_compute_dtype='float64'"
            )
        if not isinstance(self.vector_norm_order, str) or self.vector_norm_order != "l2":
            raise ValueError("vector_norm_order must be the configured 'l2' norm")
        if (
            isinstance(self.direction_norm_epsilon, bool)
            or not isinstance(self.direction_norm_epsilon, (int, float))
        ):
            raise TypeError("direction_norm_epsilon must be numeric")
        epsilon = float(self.direction_norm_epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("direction_norm_epsilon must be finite and positive")
        object.__setattr__(self, "direction_norm_epsilon", epsilon)
        if (
            isinstance(self.pseudo_target_quantile, bool)
            or not isinstance(self.pseudo_target_quantile, (int, float))
        ):
            raise TypeError("pseudo_target_quantile must be numeric")
        quantile = float(self.pseudo_target_quantile)
        if not math.isfinite(quantile) or not 0.0 < quantile < 0.5:
            raise ValueError("pseudo_target_quantile must be in (0, 0.5)")
        object.__setattr__(self, "pseudo_target_quantile", quantile)
        quantile_methods = {
            "inverted_cdf",
            "averaged_inverted_cdf",
            "closest_observation",
            "interpolated_inverted_cdf",
            "hazen",
            "weibull",
            "linear",
            "median_unbiased",
            "normal_unbiased",
            "lower",
            "higher",
            "midpoint",
            "nearest",
        }
        if (
            not isinstance(self.quantile_method, str)
            or self.quantile_method not in quantile_methods
        ):
            raise ValueError("quantile_method is not a supported NumPy quantile method")
        if (
            isinstance(self.minimum_contrast_side_rows, bool)
            or not isinstance(self.minimum_contrast_side_rows, int)
            or self.minimum_contrast_side_rows < 1
        ):
            raise ValueError("minimum_contrast_side_rows must be a positive integer")
        if self.lstsq_rcond is not None:
            if isinstance(self.lstsq_rcond, bool) or not isinstance(
                self.lstsq_rcond,
                (int, float),
            ):
                raise TypeError("lstsq_rcond must be null or numeric")
            rcond = float(self.lstsq_rcond)
            if not math.isfinite(rcond) or rcond <= 0.0:
                raise ValueError("lstsq_rcond must be null or finite and positive")
            object.__setattr__(self, "lstsq_rcond", rcond)
        if (
            not isinstance(self.lstsq_solution_policy, str)
            or self.lstsq_solution_policy != "numpy_minimum_norm_v1"
        ):
            raise ValueError(
                "lstsq_solution_policy must be 'numpy_minimum_norm_v1'"
            )
        pseudo_contrasts = tuple(
            value
            for value in contrasts
            if value.contrast_family == "r_pseudo_target"
        )
        if not pseudo_contrasts:
            raise ValueError(
                "embedding configuration requires an R-pseudo-target contrast"
            )
        if self.pseudo_target_weighted:
            if any(
                value.sample_weight_target_name is None
                for value in pseudo_contrasts
            ):
                raise ValueError(
                    "weighted R-pseudo-target contrasts require an explicit "
                    "sample-weight target"
                )
        elif any(
            value.sample_weight_target_name is not None
            for value in pseudo_contrasts
        ):
            raise ValueError(
                "unweighted R-pseudo-target configuration cannot name a "
                "sample-weight target"
            )
        if not isinstance(self.semantic_input, str) or self.semantic_input != "content":
            raise ValueError("semantic_input must be explicitly configured as 'content'")
        if (
            not isinstance(self.semantic_encoding, str)
            or self.semantic_encoding.lower().replace("_", "-") != "utf-8"
        ):
            raise ValueError("semantic_encoding must be UTF-8")
        if (
            not isinstance(self.semantic_decode_error, str)
            or self.semantic_decode_error != "strict"
        ):
            raise ValueError("semantic_decode_error must be 'strict'")
        if self.semantic_preprocessor is not None:
            raise ValueError("semantic_preprocessor must be null")
        if self.semantic_tokenizer is not None:
            raise ValueError("semantic_tokenizer must be null")
        if (
            not isinstance(self.semantic_analyzer, str)
            or self.semantic_analyzer not in {"word", "char", "char_wb"}
        ):
            raise ValueError(
                "semantic_analyzer must be word, char, or char_wb"
            )
        if (
            isinstance(self.semantic_ngram_min, bool)
            or isinstance(self.semantic_ngram_max, bool)
            or not isinstance(self.semantic_ngram_min, int)
            or not isinstance(self.semantic_ngram_max, int)
            or self.semantic_ngram_min < 1
            or self.semantic_ngram_max < self.semantic_ngram_min
        ):
            raise ValueError("semantic n-gram bounds are invalid")
        if not isinstance(self.semantic_token_pattern, str) or not self.semantic_token_pattern:
            raise ValueError("semantic_token_pattern must be explicitly configured")
        if (
            self.semantic_strip_accents is not None
            and (
                not isinstance(self.semantic_strip_accents, str)
                or self.semantic_strip_accents not in {"ascii", "unicode"}
            )
        ):
            raise ValueError("semantic_strip_accents must be null, ascii, or unicode")
        if (
            isinstance(self.semantic_min_df, bool)
            or not isinstance(self.semantic_min_df, int)
            or self.semantic_min_df < 1
        ):
            raise ValueError("semantic_min_df must be a positive integer")
        if (
            isinstance(self.semantic_max_df, bool)
            or not isinstance(self.semantic_max_df, (int, float))
        ):
            raise TypeError("semantic_max_df must be numeric")
        maximum_df = float(self.semantic_max_df)
        if not math.isfinite(maximum_df) or not 0.0 < maximum_df <= 1.0:
            raise ValueError("semantic_max_df must be in (0, 1]")
        object.__setattr__(self, "semantic_max_df", maximum_df)
        if (
            self.semantic_norm is not None
            and (
                not isinstance(self.semantic_norm, str)
                or self.semantic_norm not in {"l1", "l2"}
            )
        ):
            raise ValueError("semantic_norm must be null, l1, or l2")
        if (
            not isinstance(self.semantic_dtype, str)
            or self.semantic_dtype not in {"float32", "float64"}
        ):
            raise ValueError("semantic_dtype must be float32 or float64")
        if isinstance(self.semantic_stop_words, str):
            if self.semantic_stop_words != "english":
                raise ValueError(
                    "semantic_stop_words string must be the configured "
                    "scikit-learn 'english' vocabulary"
                )
        elif self.semantic_stop_words is not None:
            words = tuple(str(value) for value in self.semantic_stop_words)
            if any(not value for value in words) or len(words) != len(set(words)):
                raise ValueError("semantic_stop_words must be nonempty and unique")
            object.__setattr__(self, "semantic_stop_words", words)
        if self.semantic_vocabulary is not None:
            raise ValueError(
                "semantic_vocabulary must be null so the complete fit vocabulary "
                "is learned and authenticated"
            )
        if self.semantic_max_features is not None:
            raise ValueError(
                "semantic_max_features must be null; feature selection is forbidden"
            )
        if (
            isinstance(self.semantic_member_batch_size, bool)
            or not isinstance(self.semantic_member_batch_size, int)
            or self.semantic_member_batch_size < 1
        ):
            raise ValueError(
                "semantic_member_batch_size must be an explicitly configured "
                "positive integer"
            )
        for field_name in (
            "maximum_source_chunks_per_row",
            "maximum_retrieval_chunks_per_side",
            "maximum_semantic_terms",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{field_name} must be null or a positive integer")
        if self.overflow_policy != "fail_closed_no_selection":
            raise ValueError("embedding capacity overflow must fail closed")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROLE_NEUTRAL_EMBEDDING_CONFIG_SCHEMA,
            "contrasts": [value.as_dict() for value in self.contrasts],
            "normalize_patient_embeddings": bool(self.normalize_patient_embeddings),
            "patient_embedding_pooling": self.patient_embedding_pooling,
            "numeric_compute_dtype": self.numeric_compute_dtype,
            "vector_norm_order": self.vector_norm_order,
            "direction_norm_epsilon": float(self.direction_norm_epsilon),
            "pseudo_target_quantile": float(self.pseudo_target_quantile),
            "pseudo_target_weighted": bool(self.pseudo_target_weighted),
            "quantile_method": self.quantile_method,
            "minimum_contrast_side_rows": int(self.minimum_contrast_side_rows),
            "lstsq_rcond": self.lstsq_rcond,
            "lstsq_solution_policy": self.lstsq_solution_policy,
            "semantic_input": self.semantic_input,
            "semantic_encoding": self.semantic_encoding,
            "semantic_decode_error": self.semantic_decode_error,
            "semantic_preprocessor": self.semantic_preprocessor,
            "semantic_tokenizer": self.semantic_tokenizer,
            "semantic_analyzer": self.semantic_analyzer,
            "semantic_ngram_min": int(self.semantic_ngram_min),
            "semantic_ngram_max": int(self.semantic_ngram_max),
            "semantic_token_pattern": self.semantic_token_pattern,
            "semantic_lowercase": bool(self.semantic_lowercase),
            "semantic_strip_accents": self.semantic_strip_accents,
            "semantic_min_df": int(self.semantic_min_df),
            "semantic_max_df": float(self.semantic_max_df),
            "semantic_sublinear_tf": bool(self.semantic_sublinear_tf),
            "semantic_norm": self.semantic_norm,
            "semantic_use_idf": bool(self.semantic_use_idf),
            "semantic_smooth_idf": bool(self.semantic_smooth_idf),
            "semantic_binary": bool(self.semantic_binary),
            "semantic_dtype": self.semantic_dtype,
            "semantic_stop_words": (
                None
                if self.semantic_stop_words is None
                else (
                    self.semantic_stop_words
                    if isinstance(self.semantic_stop_words, str)
                    else list(self.semantic_stop_words)
                )
            ),
            "semantic_vocabulary": self.semantic_vocabulary,
            "semantic_max_features": self.semantic_max_features,
            "semantic_member_batch_size": int(
                self.semantic_member_batch_size
            ),
            "maximum_source_chunks_per_row": self.maximum_source_chunks_per_row,
            "maximum_retrieval_chunks_per_side": (
                self.maximum_retrieval_chunks_per_side
            ),
            "maximum_semantic_terms": self.maximum_semantic_terms,
            "overflow_policy": self.overflow_policy,
            "source_chunk_policy": "exact_uncapped_cache_projection_v1",
            "semantic_term_policy": "complete_configured_vocabulary_fail_on_cap_v1",
            "text_truncation_allowed": False,
        }

    @property
    def content_sha256(self) -> str:
        return _sha256_json(self.as_dict())


@dataclass(frozen=True)
class RoleNeutralEmbeddingPhysicalGroupRequest:
    """Closed owner/group authority, independent of device assignment."""

    plan_scientific_content_sha256: str
    physical_owner: Stage1ScopeSpec
    logical_members: tuple[Stage1ScopeSpec, ...]
    content_sha256: str

    @classmethod
    def from_plan(
        cls,
        *,
        plan: Stage1ScopePlan,
        physical_owner_scope_id: str,
    ) -> "RoleNeutralEmbeddingPhysicalGroupRequest":
        if not isinstance(plan, Stage1ScopePlan):
            raise TypeError("embedding request requires a Stage1ScopePlan")
        owner = plan.scope(str(physical_owner_scope_id))
        if plan.physical_owner(owner.scope_id).scope_id != owner.scope_id:
            raise ValueError("embedding request must name a physical owner")
        matches = [
            members
            for candidate, members in plan.physical_scope_groups
            if candidate.scope_id == owner.scope_id
        ]
        if len(matches) != 1:
            raise RuntimeError("embedding physical owner has no unique group")
        members = tuple(matches[0])
        if members[0].scope_id != owner.scope_id or any(
            tuple(member.fit_row_ids) != tuple(owner.fit_row_ids)
            or member.scope_seed != owner.scope_seed
            for member in members
        ):
            raise ValueError("embedding group does not share an exact fit set and seed")
        aliases = members[1:]
        if aliases and (
            owner.scope_kind != "exact_inner"
            or any(member.scope_kind != "cumulative_spent" for member in aliases)
        ):
            raise ValueError("embedding fit reuse supports exact/cumulative aliases only")
        body = cls._body(
            plan_scientific_content_sha256=plan.scientific_content_sha256,
            owner=owner,
            members=members,
        )
        return cls(
            plan_scientific_content_sha256=plan.scientific_content_sha256,
            physical_owner=owner,
            logical_members=members,
            content_sha256=_sha256_json(body),
        )

    @staticmethod
    def _body(
        *,
        plan_scientific_content_sha256: str,
        owner: Stage1ScopeSpec,
        members: Sequence[Stage1ScopeSpec],
    ) -> dict[str, Any]:
        return {
            "schema_version": ROLE_NEUTRAL_EMBEDDING_REQUEST_SCHEMA,
            "plan_scientific_content_sha256": plan_scientific_content_sha256,
            "physical_owner": owner.as_dict(),
            "logical_members": [member.as_dict() for member in members],
            "logical_scope_count": len(members),
            "fit_row_ids": list(owner.fit_row_ids),
            "fit_row_order_fingerprint": _row_order_fingerprint(owner.fit_row_ids),
            "canonical_group_seed": int(owner.scope_seed),
            "heldout_labels_supplied": False,
            "peer_group_definitions_supplied": False,
        }

    def as_dict(self) -> dict[str, Any]:
        _require_sha256(
            self.plan_scientific_content_sha256,
            label="embedding scientific plan identity",
        )
        body = self._body(
            plan_scientific_content_sha256=self.plan_scientific_content_sha256,
            owner=self.physical_owner,
            members=self.logical_members,
        )
        if _sha256_json(body) != self.content_sha256:
            raise RuntimeError("role-neutral embedding request changed")
        return {**body, "content_sha256": self.content_sha256}


@dataclass(frozen=True)
class ExactHeldoutEmbeddingBatch:
    """The only held-out capability accepted by the executor.

    There is intentionally no treatment/outcome/label member.
    """

    row_ids: tuple[int, ...]
    texts: tuple[str, ...]
    embedding_provider: BoundSpentFrozenChunkEmbeddingProvider

    def __post_init__(self) -> None:
        rows = tuple(map(int, self.row_ids))
        texts = tuple(self.texts)
        if not rows or len(rows) != len(set(rows)):
            raise ValueError("exact held-out row IDs must be nonempty and unique")
        if len(texts) != len(rows) or not all(isinstance(value, str) for value in texts):
            raise ValueError("exact held-out text must align one-for-one with row IDs")
        if (
            type(self.embedding_provider) is not BoundSpentFrozenChunkEmbeddingProvider
            or tuple(map(int, self.embedding_provider.row_ids)) != rows
        ):
            raise ValueError("exact held-out provider must expose exactly the requested rows")
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "texts", texts)


def _content_identity(value: Mapping[str, Any], *, label: str) -> str:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one mapping")
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    identity = _require_sha256(
        value.get("content_sha256"),
        label=f"{label}.content_sha256",
    )
    if identity != _sha256_json(body):
        raise ValueError(f"{label} content identity changed")
    return identity


def _canonical_preflight_scope_binding(
    *,
    preflight: _ClusterPreflightArtifact,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
    provider_cache_identity: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not isinstance(
        preflight,
        (
            ProductionStage1ClusterPreflightArtifact,
            PortableProductionStage1ClusterPreflightArtifact,
        ),
    ):
        raise TypeError(
            "cluster state requires an authenticated "
            "clustered-preflight artifact"
        )
    identity = preflight.identity()
    owner = request.physical_owner
    if isinstance(
        preflight,
        PortableProductionStage1ClusterPreflightArtifact,
    ):
        audit_identity = _require_sha256(
            identity.get("cluster_audit_content_sha256"),
            label="portable clustered preflight source audit",
        )
        scope = preflight.logical_scope_record(
            owner.scope_id,
            include_concepts=True,
        )
        audit_header = preflight.source_audit_header()
    else:
        audit = copy.deepcopy(dict(preflight.audit))
        audit_identity = _content_identity(
            audit,
            label="clustered preflight audit",
        )
        if identity.get("cluster_audit_content_sha256") != audit_identity:
            raise ValueError(
                "clustered preflight result changed its audit binding"
            )
        matches = [
            copy.deepcopy(dict(row))
            for row in audit.get("scopes") or ()
            if isinstance(row, Mapping)
            and row.get("scope_id") == owner.scope_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "clustered preflight has no unique canonical owner scope"
            )
        scope = matches[0]
        audit_header = audit
    fit_identity = scope.get("cluster_fit_identity")
    if not isinstance(fit_identity, Mapping):
        raise ValueError("clustered preflight scope has no fitted state identity")
    fit_identity_sha256 = _content_identity(
        fit_identity,
        label="clustered preflight fit identity",
    )
    owner_dict = owner.as_dict()
    final_concepts = fit_identity.get("final_catalog_concepts")
    if (
        scope.get("fit_row_count") != owner.fit_row_count
        or scope.get("fit_row_order_fingerprint")
        != owner_dict["fit_row_order_fingerprint"]
        or fit_identity.get("fit_row_ids") != list(owner.fit_row_ids)
        or fit_identity.get("fit_row_order_fingerprint")
        != owner_dict["fit_row_order_fingerprint"]
        or scope.get("canonical_group_seed") != owner.scope_seed
        or fit_identity.get("canonical_group_seed") != owner.scope_seed
        or fit_identity.get("ordered_fit_row_seed_policy")
        != "canonical_ordered_fit_rows_group_seed_v1"
        or scope.get("token_bounded_row_count") != 0
        or scope.get("uncapped_semantic_projection") is not True
        or not isinstance(final_concepts, Mapping)
        or not final_concepts.get(EMBEDDING_CLUSTERED)
        or not final_concepts.get(TFIDF_SEMANTIC_RETRIEVAL)
    ):
        raise ValueError(
            "clustered preflight scope is incomplete, capped, or belongs to "
            "another ordered fit scope"
        )
    expected_cache_identity = audit_header.get(
        "embedding_cache_identity_sha256"
    )
    _require_sha256(
        expected_cache_identity,
        label="clustered preflight embedding-cache identity",
    )
    if provider_cache_identity is not None and _sha256_json(
        dict(provider_cache_identity)
    ) != expected_cache_identity:
        raise ValueError("fit embedding provider differs from clustered preflight cache")
    binding = {
        "cluster_audit_content_sha256": audit_identity,
        "cluster_scope_id": owner.scope_id,
        "cluster_scope_record_sha256": _sha256_json(scope),
        "cluster_fit_identity_sha256": fit_identity_sha256,
        "embedding_cache_identity_sha256": expected_cache_identity,
        # The typed preflight handle has freshly authenticated its request and
        # manifest. Their legacy byte identities include physical locators, so
        # they belong to execution attestation rather than this scientific
        # binding.
        "source_preflight_freshly_authenticated": True,
        "preflight_location_bound_in_scientific_identity": False,
    }
    return binding, scope


def _preflight_array_expectations(
    scope: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], list[Mapping[str, Any]]]:
    fit_identity = scope["cluster_fit_identity"]
    kmeans = fit_identity.get("kmeans")
    svds = fit_identity.get("svd_families")
    expected_kmeans = {
        "usable_mask": "cluster_kmeans_usable_mask",
        "cluster_labels": "cluster_kmeans_labels",
        "cluster_centers": "cluster_kmeans_centers",
        "cluster_counts": "cluster_kmeans_counts",
    }
    if (
        not isinstance(kmeans, Mapping)
        or not isinstance(svds, list)
        or len(svds) != 2
        or [row.get("family_key") for row in svds if isinstance(row, Mapping)]
        != ["treatment", "residualized_interaction"]
    ):
        raise ValueError("clustered preflight fit identity has incomplete fitted arrays")
    kmeans_expectations: dict[str, Mapping[str, Any]] = {}
    for identity_key, artifact_key in expected_kmeans.items():
        value = kmeans.get(identity_key)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"dtype", "shape", "sha256"}
        ):
            raise ValueError("clustered preflight KMeans array identity is malformed")
        kmeans_expectations[artifact_key] = value
    for row in svds:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("weighted_matrix"), Mapping)
            or not isinstance(row.get("singular_values"), Mapping)
            or not isinstance(row.get("components"), Mapping)
        ):
            raise ValueError("clustered preflight SVD array identity is malformed")
    return kmeans_expectations, svds


def _normalize_cluster_state_arrays(
    *,
    scope: Mapping[str, Any],
    kmeans_state: Mapping[str, Any],
    svd_states: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    expected_kmeans, expected_svds = _preflight_array_expectations(scope)
    required_kmeans = {
        "fit_row_ids",
        "parameters",
        "scientific_configuration",
        "canonical_group_seed",
        "ordered_fit_row_seed_policy",
        "usable_mask",
        "cluster_labels",
        "cluster_centers",
        "cluster_counts",
        "n_iter",
        "inertia",
    }
    if not isinstance(kmeans_state, Mapping) or set(kmeans_state) != required_kmeans:
        raise ValueError("canonical clustered preflight KMeans state is incomplete")
    if kmeans_state.get("fit_row_ids") != scope["cluster_fit_identity"]["fit_row_ids"]:
        raise ValueError("canonical clustered preflight KMeans row order changed")
    arrays = {
        "cluster_kmeans_usable_mask": np.ascontiguousarray(
            np.asarray(kmeans_state["usable_mask"], dtype=np.bool_)
        ),
        "cluster_kmeans_labels": np.ascontiguousarray(
            np.asarray(kmeans_state["cluster_labels"], dtype=np.int64)
        ),
        "cluster_kmeans_centers": np.ascontiguousarray(
            np.asarray(kmeans_state["cluster_centers"], dtype=np.float64)
        ),
        "cluster_kmeans_counts": np.ascontiguousarray(
            np.asarray(kmeans_state["cluster_counts"], dtype=np.int64)
        ),
    }
    # The preflight identity records exact dtypes as well as values. Do not
    # silently coerce a producer's scientific state into a different identity.
    supplied = {
        "cluster_kmeans_usable_mask": np.ascontiguousarray(
            np.asarray(kmeans_state["usable_mask"])
        ),
        "cluster_kmeans_labels": np.ascontiguousarray(
            np.asarray(kmeans_state["cluster_labels"])
        ),
        "cluster_kmeans_centers": np.ascontiguousarray(
            np.asarray(kmeans_state["cluster_centers"])
        ),
        "cluster_kmeans_counts": np.ascontiguousarray(
            np.asarray(kmeans_state["cluster_counts"])
        ),
    }
    for key, expected in expected_kmeans.items():
        if _array_identity(supplied[key]) != dict(expected):
            raise ValueError(f"canonical clustered preflight array differs: {key}")
        arrays[key] = supplied[key]
    svd_records: list[dict[str, Any]] = []
    if len(tuple(svd_states)) != len(expected_svds):
        raise ValueError("canonical clustered preflight SVD family count changed")
    for index, (state, expected) in enumerate(
        zip(svd_states, expected_svds, strict=True)
    ):
        required = {
            "family_key",
            "item_cluster_ids",
            "weighted_matrix",
            "singular_values",
            "components",
            "parameters",
            "sign_canonicalization_policy",
            "rank_tolerance_policy",
            "rank_tolerance_dtype",
            "rank_tolerance_multiplier",
            "rank_tolerance",
            "numerical_rank",
            "replay_comparison_policy",
            "replay_relative_tolerance",
            "replay_absolute_tolerance",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != required
            or state.get("family_key") != expected.get("family_key")
            or list(state.get("item_cluster_ids") or ())
            != expected.get("item_cluster_ids")
        ):
            raise ValueError("canonical clustered preflight SVD metadata changed")
        record = {
            "family_key": str(state["family_key"]),
            "item_cluster_ids": list(map(int, state["item_cluster_ids"])),
            "parameters": copy.deepcopy(dict(state["parameters"])),
            "sign_canonicalization_policy": str(
                state["sign_canonicalization_policy"]
            ),
            "rank_tolerance_policy": str(state["rank_tolerance_policy"]),
            "rank_tolerance_dtype": str(state["rank_tolerance_dtype"]),
            "rank_tolerance_multiplier_hex": float(
                state["rank_tolerance_multiplier"]
            ).hex(),
            "rank_tolerance_hex": float(state["rank_tolerance"]).hex(),
            "numerical_rank": int(state["numerical_rank"]),
            "replay_comparison_policy": str(
                state["replay_comparison_policy"]
            ),
            "replay_relative_tolerance_hex": float(
                state["replay_relative_tolerance"]
            ).hex(),
            "replay_absolute_tolerance_hex": float(
                state["replay_absolute_tolerance"]
            ).hex(),
        }
        for field_name in ("weighted_matrix", "singular_values", "components"):
            key = f"cluster_svd_{index}_{field_name}"
            value = np.ascontiguousarray(np.asarray(state[field_name]))
            expected_identity = expected.get(field_name)
            if (
                not isinstance(expected_identity, Mapping)
                or _array_identity(value) != dict(expected_identity)
            ):
                raise ValueError(
                    "canonical clustered preflight SVD array differs: "
                    f"{state['family_key']}/{field_name}"
                )
            arrays[key] = value
            record[field_name] = key
        svd_records.append(record)
    metadata = {
        "kmeans_parameters": copy.deepcopy(dict(kmeans_state["parameters"])),
        "cluster_scientific_configuration": copy.deepcopy(
            dict(kmeans_state["scientific_configuration"])
        ),
        "canonical_group_seed": int(kmeans_state["canonical_group_seed"]),
        "ordered_fit_row_seed_policy": str(
            kmeans_state["ordered_fit_row_seed_policy"]
        ),
        "kmeans_n_iter": int(kmeans_state["n_iter"]),
        "kmeans_inertia_hex": float(kmeans_state["inertia"]).hex(),
        "svd_states": svd_records,
    }
    return arrays, metadata


@dataclass(frozen=True)
class AuthenticatedClusteredPreflightScopeState:
    root: Path
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    scope_record: Mapping[str, Any]

    @property
    def content_sha256(self) -> str:
        return str(self.manifest["content_sha256"])


@dataclass(frozen=True)
class _LazyPortableClusterScopeRecord(Mapping[str, Any]):
    """One-owner view that never retains a cohort-wide concept aggregate."""

    preflight: PortableProductionStage1ClusterPreflightArtifact
    scope_id: str

    def _snapshot(self) -> dict[str, Any]:
        return self.preflight.logical_scope_record(
            self.scope_id,
            include_concepts=True,
        )

    def __getitem__(self, key: str) -> Any:
        return self._snapshot()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(tuple(self._snapshot()))

    def __len__(self) -> int:
        return len(self._snapshot())

    def __deepcopy__(self, _memo: dict[int, Any]) -> "_LazyPortableClusterScopeRecord":
        # Copying this capability must not materialize its lossless payload.
        return self


def _materialize_cluster_scope_for_one_operation(
    scope: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Resolve a portable scope once for one bounded consumer operation."""

    if isinstance(scope, _LazyPortableClusterScopeRecord):
        return scope._snapshot()
    return scope


@dataclass(frozen=True)
class AuthenticatedClusteredPreflightStateBundle:
    """Freshly authenticated state for every deduplicated physical owner."""

    root: Path
    manifest: Mapping[str, Any]
    states: Mapping[str, AuthenticatedClusteredPreflightScopeState]

    @property
    def content_sha256(self) -> str:
        return str(self.manifest["content_sha256"])

    def manifest_path_for_owner(self, scope_id: str) -> Path:
        selected = self.states.get(str(scope_id))
        if selected is None:
            raise ValueError(
                "clustered preflight has no sealed state for that physical owner"
            )
        return selected.root / _CLUSTER_STATE_MANIFEST


def seal_canonical_clustered_preflight_scope_state(
    *,
    output_root: Path | str,
    preflight: _ClusterPreflightArtifact,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
    kmeans_state: Mapping[str, Any],
    svd_states: Sequence[Mapping[str, Any]],
) -> AuthenticatedClusteredPreflightScopeState:
    """Publish the actual canonical preflight arrays once, without refitting."""

    if not isinstance(request, RoleNeutralEmbeddingPhysicalGroupRequest):
        raise TypeError("cluster state publication requires an embedding request")
    request.as_dict()
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("cluster state output root must be absolute")
    if root.exists() or root.is_symlink():
        raise FileExistsError("cluster state output root must be fresh")
    root.mkdir(parents=True, exist_ok=False)
    binding, scope = _canonical_preflight_scope_binding(
        preflight=preflight,
        request=request,
        provider_cache_identity=None,
    )
    arrays, state_metadata = _normalize_cluster_state_arrays(
        scope=scope,
        kmeans_state=kmeans_state,
        svd_states=tuple(svd_states),
    )
    registrations: dict[str, dict[str, Any]] = {}
    arrays_root = root / "arrays"
    arrays_root.mkdir(parents=True, exist_ok=False)
    for key in sorted(arrays):
        path = arrays_root / f"{key}.npy"
        _write_new_npy(path, arrays[key])
        registrations[key] = _array_registration(
            path,
            relative_to=root,
            value=arrays[key],
        )
    body = {
        "schema_version": ROLE_NEUTRAL_EMBEDDING_CLUSTER_STATE_SCHEMA,
        "status": "complete",
        "group_request_content_sha256": request.content_sha256,
        "plan_scientific_content_sha256": request.plan_scientific_content_sha256,
        "physical_owner_scope_id": request.physical_owner.scope_id,
        "fit_row_ids": list(request.physical_owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(
            request.physical_owner.fit_row_ids
        ),
        "canonical_group_seed": int(request.physical_owner.scope_seed),
        "cluster_scientific_configuration_sha256": _sha256_json(
            state_metadata["cluster_scientific_configuration"]
        ),
        "preflight_binding": binding,
        "state_metadata": state_metadata,
        "array_order": sorted(registrations),
        "arrays": registrations,
        "state_origin": "canonical_clustered_preflight_no_refit_v1",
        "executable_serialization_used": False,
        "pickle_joblib_or_npz_used": False,
    }
    manifest = {**body, "content_sha256": _sha256_json(body)}
    _write_new_json(root / _CLUSTER_STATE_MANIFEST, manifest)
    return load_canonical_clustered_preflight_scope_state(
        manifest_path=root / _CLUSTER_STATE_MANIFEST,
        preflight=preflight,
        request=request,
    )


def load_canonical_clustered_preflight_scope_state(
    *,
    manifest_path: Path | str,
    preflight: _ClusterPreflightArtifact,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
) -> AuthenticatedClusteredPreflightScopeState:
    """Freshly authenticate one safe-array preflight state handoff."""

    supplied = Path(manifest_path)
    if not supplied.is_absolute() or supplied.name != _CLUSTER_STATE_MANIFEST:
        raise ValueError("cluster state manifest path must be absolute and canonical")
    root = supplied.parent
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("cluster state root must be one real canonical directory")
    binding, scope = _canonical_preflight_scope_binding(
        preflight=preflight,
        request=request,
        provider_cache_identity=None,
    )
    manifest = _read_json(supplied, label="cluster state manifest")
    body = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "content_sha256"
    }
    required = {
        "schema_version",
        "status",
        "group_request_content_sha256",
        "plan_scientific_content_sha256",
        "physical_owner_scope_id",
        "fit_row_ids",
        "fit_row_order_fingerprint",
        "canonical_group_seed",
        "cluster_scientific_configuration_sha256",
        "preflight_binding",
        "state_metadata",
        "array_order",
        "arrays",
        "state_origin",
        "executable_serialization_used",
        "pickle_joblib_or_npz_used",
        "content_sha256",
    }
    arrays_raw = manifest.get("arrays")
    array_order = manifest.get("array_order")
    if (
        set(manifest) != required
        or manifest.get("schema_version")
        != ROLE_NEUTRAL_EMBEDDING_CLUSTER_STATE_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("group_request_content_sha256") != request.content_sha256
        or manifest.get("plan_scientific_content_sha256")
        != request.plan_scientific_content_sha256
        or manifest.get("physical_owner_scope_id")
        != request.physical_owner.scope_id
        or manifest.get("fit_row_ids") != list(request.physical_owner.fit_row_ids)
        or manifest.get("fit_row_order_fingerprint")
        != _row_order_fingerprint(request.physical_owner.fit_row_ids)
        or manifest.get("canonical_group_seed")
        != request.physical_owner.scope_seed
        or manifest.get("cluster_scientific_configuration_sha256")
        != _sha256_json(
            (manifest.get("state_metadata") or {}).get(
                "cluster_scientific_configuration"
            )
        )
        or manifest.get("preflight_binding") != binding
        or manifest.get("state_origin")
        != "canonical_clustered_preflight_no_refit_v1"
        or manifest.get("executable_serialization_used") is not False
        or manifest.get("pickle_joblib_or_npz_used") is not False
        or manifest.get("content_sha256") != _sha256_json(body)
        or not isinstance(arrays_raw, Mapping)
        or not isinstance(array_order, list)
        or array_order != sorted(arrays_raw)
    ):
        raise ValueError("cluster state manifest is invalid or reordered")
    expected_names = {
        _CLUSTER_STATE_MANIFEST,
        "arrays",
    }
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("cluster state artifact tree contains unregistered entries")
    arrays_root = root / "arrays"
    if (
        arrays_root.is_symlink()
        or not arrays_root.is_dir()
        or {path.name for path in arrays_root.iterdir()}
        != {f"{key}.npy" for key in array_order}
    ):
        raise ValueError("cluster state array inventory is incomplete or reordered")
    arrays = {
        key: _read_registered_array(
            root=root,
            registration=arrays_raw[key],
            label=f"cluster state array {key}",
        )
        for key in array_order
    }
    state_metadata = manifest.get("state_metadata")
    if not isinstance(state_metadata, Mapping):
        raise ValueError("cluster state metadata is malformed")
    kmeans_state = {
        "fit_row_ids": list(request.physical_owner.fit_row_ids),
        "parameters": state_metadata.get("kmeans_parameters"),
        "scientific_configuration": state_metadata.get(
            "cluster_scientific_configuration"
        ),
        "canonical_group_seed": state_metadata.get("canonical_group_seed"),
        "ordered_fit_row_seed_policy": state_metadata.get(
            "ordered_fit_row_seed_policy"
        ),
        "usable_mask": arrays.get("cluster_kmeans_usable_mask"),
        "cluster_labels": arrays.get("cluster_kmeans_labels"),
        "cluster_centers": arrays.get("cluster_kmeans_centers"),
        "cluster_counts": arrays.get("cluster_kmeans_counts"),
        "n_iter": state_metadata.get("kmeans_n_iter"),
        "inertia": float.fromhex(str(state_metadata.get("kmeans_inertia_hex"))),
    }
    svd_states = []
    for row in state_metadata.get("svd_states") or ():
        if not isinstance(row, Mapping):
            raise ValueError("cluster state SVD metadata is malformed")
        svd_states.append(
            {
                "family_key": row.get("family_key"),
                "item_cluster_ids": row.get("item_cluster_ids"),
                "weighted_matrix": arrays.get(str(row.get("weighted_matrix"))),
                "singular_values": arrays.get(str(row.get("singular_values"))),
                "components": arrays.get(str(row.get("components"))),
                "parameters": row.get("parameters"),
                "sign_canonicalization_policy": row.get(
                    "sign_canonicalization_policy"
                ),
                "rank_tolerance_policy": row.get("rank_tolerance_policy"),
                "rank_tolerance_dtype": row.get("rank_tolerance_dtype"),
                "rank_tolerance_multiplier": float.fromhex(
                    str(row.get("rank_tolerance_multiplier_hex"))
                ),
                "rank_tolerance": float.fromhex(
                    str(row.get("rank_tolerance_hex"))
                ),
                "numerical_rank": row.get("numerical_rank"),
                "replay_comparison_policy": row.get(
                    "replay_comparison_policy"
                ),
                "replay_relative_tolerance": float.fromhex(
                    str(row.get("replay_relative_tolerance_hex"))
                ),
                "replay_absolute_tolerance": float.fromhex(
                    str(row.get("replay_absolute_tolerance_hex"))
                ),
            }
        )
    normalized, normalized_metadata = _normalize_cluster_state_arrays(
        scope=scope,
        kmeans_state=kmeans_state,
        svd_states=svd_states,
    )
    if (
        normalized_metadata != state_metadata
        or set(normalized) != set(arrays)
        or any(
            _array_identity(normalized[key]) != _array_identity(arrays[key])
            for key in normalized
        )
    ):
        raise ValueError("cluster state replay differs from its canonical preflight")
    retained_scope: Mapping[str, Any]
    if isinstance(
        preflight,
        PortableProductionStage1ClusterPreflightArtifact,
    ):
        retained_scope = _LazyPortableClusterScopeRecord(
            preflight=preflight,
            scope_id=request.physical_owner.scope_id,
        )
    else:
        retained_scope = copy.deepcopy(scope)
    return AuthenticatedClusteredPreflightScopeState(
        root=root,
        manifest=copy.deepcopy(manifest),
        arrays=arrays,
        scope_record=retained_scope,
    )


def _cluster_state_bundle_preflight_binding(
    preflight: _ClusterPreflightArtifact,
) -> dict[str, Any]:
    if not isinstance(
        preflight,
        (
            ProductionStage1ClusterPreflightArtifact,
            PortableProductionStage1ClusterPreflightArtifact,
        ),
    ):
        raise TypeError(
            "cluster state bundle requires an authenticated preflight artifact"
        )
    identity = preflight.identity()
    return {
        "cluster_audit_content_sha256": _require_sha256(
            identity.get("cluster_audit_content_sha256"),
            label="preflight cluster audit identity",
        ),
        "stage1_request_sha256": _require_sha256(
            identity.get("stage1_request_sha256"),
            label="preflight Stage 1 request identity",
        ),
        "scope_fit_identity_sha256": _require_sha256(
            identity.get("scope_fit_identity_sha256"),
            label="preflight fitted-state inventory identity",
        ),
        "source_preflight_freshly_authenticated": True,
        "preflight_physical_locator_included": False,
    }


def seal_canonical_clustered_preflight_state_bundle(
    *,
    output_root: Path | str,
    preflight: _ClusterPreflightArtifact,
    plan: Stage1ScopePlan,
    captured_scope_states: Mapping[str, Mapping[str, Any]],
) -> AuthenticatedClusteredPreflightStateBundle:
    """Publish exactly one safe KMeans/SVD state per physical fit owner."""

    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError("cluster state bundle publication requires a scope plan")
    plan.as_dict()
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("cluster state bundle output root must be absolute")
    if root.exists() or root.is_symlink():
        raise FileExistsError("cluster state bundle output root must be fresh")
    owner_ids = tuple(scope.scope_id for scope in plan.physical_scopes)
    if (
        not isinstance(captured_scope_states, Mapping)
        or set(captured_scope_states) != set(owner_ids)
    ):
        raise ValueError(
            "cluster state bundle requires exactly the canonical physical owners"
        )
    root.mkdir(parents=True, exist_ok=False)
    owners_root = root / "owners"
    owners_root.mkdir(parents=True, exist_ok=False)
    registrations: list[dict[str, Any]] = []
    for index, owner_id in enumerate(owner_ids):
        captured = captured_scope_states[owner_id]
        if (
            not isinstance(captured, Mapping)
            or set(captured)
            != {
                "schema_version",
                "scope_id",
                "cluster_fit_identity_content_sha256",
                "kmeans_state",
                "svd_states",
                "captured_from_canonical_preflight_fit",
                "refit_performed_for_state_capture",
            }
            or captured.get("schema_version")
            != "production_stage1_cluster_preflight_scope_state_capture_v2"
            or captured.get("scope_id") != owner_id
            or captured.get("captured_from_canonical_preflight_fit") is not True
            or captured.get("refit_performed_for_state_capture") is not False
        ):
            raise ValueError(
                f"canonical preflight fitted state is malformed: {owner_id}"
            )
        request = RoleNeutralEmbeddingPhysicalGroupRequest.from_plan(
            plan=plan,
            physical_owner_scope_id=owner_id,
        )
        state_root = owners_root / f"{index:03d}"
        state = seal_canonical_clustered_preflight_scope_state(
            output_root=state_root,
            preflight=preflight,
            request=request,
            kmeans_state=captured.get("kmeans_state"),
            svd_states=captured.get("svd_states"),
        )
        scope_fit_identity = state.scope_record.get("cluster_fit_identity")
        if (
            not isinstance(scope_fit_identity, Mapping)
            or captured.get("cluster_fit_identity_content_sha256")
            != scope_fit_identity.get("content_sha256")
        ):
            raise ValueError(
                f"captured state names another preflight fit: {owner_id}"
            )
        registrations.append(
            {
                "canonical_index": index,
                "physical_owner_scope_id": owner_id,
                "group_request_content_sha256": request.content_sha256,
                "relative_manifest_path": (
                    state.root / _CLUSTER_STATE_MANIFEST
                ).relative_to(root).as_posix(),
                "state_content_sha256": state.content_sha256,
                "cluster_fit_identity_content_sha256": (
                    scope_fit_identity["content_sha256"]
                ),
                "state_origin": "canonical_clustered_preflight_no_refit_v1",
            }
        )
    body = {
        "schema_version": ROLE_NEUTRAL_EMBEDDING_CLUSTER_STATE_BUNDLE_SCHEMA,
        "status": "complete",
        "plan_scientific_content_sha256": plan.scientific_content_sha256,
        "preflight_binding": _cluster_state_bundle_preflight_binding(preflight),
        "physical_owner_count": len(owner_ids),
        "physical_owner_scope_order": list(owner_ids),
        "logical_scope_count": len(plan.scopes),
        "deduplicated_logical_scope_count": len(plan.scopes) - len(owner_ids),
        "states": registrations,
        "all_physical_owners_have_one_state": True,
        "logical_alias_state_copies_published": False,
        "cluster_refit_performed": False,
        "serialization_policy": "canonical_json_and_individual_npy_only_v1",
    }
    manifest = {**body, "content_sha256": _sha256_json(body)}
    _write_new_json(root / _CLUSTER_STATE_BUNDLE_MANIFEST, manifest)
    return load_canonical_clustered_preflight_state_bundle(
        manifest_path=root / _CLUSTER_STATE_BUNDLE_MANIFEST,
        preflight=preflight,
        plan=plan,
    )


def load_canonical_clustered_preflight_state_bundle(
    *,
    manifest_path: Path | str,
    preflight: _ClusterPreflightArtifact,
    plan: Stage1ScopePlan,
) -> AuthenticatedClusteredPreflightStateBundle:
    """Authenticate all physical-owner arrays and reject audit-only legacy input."""

    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError("cluster state bundle loading requires a scope plan")
    plan.as_dict()
    supplied = Path(manifest_path)
    if (
        not supplied.is_absolute()
        or supplied.name != _CLUSTER_STATE_BUNDLE_MANIFEST
    ):
        raise ValueError(
            "cluster state bundle manifest path must be absolute and canonical; "
            "legacy audit-only preflight artifacts have no reusable fitted state"
        )
    root = supplied.parent
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("cluster state bundle root must be one real directory")
    manifest = _read_json(supplied, label="cluster state bundle manifest")
    body = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "content_sha256"
    }
    owner_ids = tuple(scope.scope_id for scope in plan.physical_scopes)
    rows = manifest.get("states")
    required = {
        "schema_version",
        "status",
        "plan_scientific_content_sha256",
        "preflight_binding",
        "physical_owner_count",
        "physical_owner_scope_order",
        "logical_scope_count",
        "deduplicated_logical_scope_count",
        "states",
        "all_physical_owners_have_one_state",
        "logical_alias_state_copies_published",
        "cluster_refit_performed",
        "serialization_policy",
        "content_sha256",
    }
    if (
        set(manifest) != required
        or manifest.get("schema_version")
        != ROLE_NEUTRAL_EMBEDDING_CLUSTER_STATE_BUNDLE_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("plan_scientific_content_sha256")
        != plan.scientific_content_sha256
        or manifest.get("preflight_binding")
        != _cluster_state_bundle_preflight_binding(preflight)
        or manifest.get("physical_owner_count") != len(owner_ids)
        or manifest.get("physical_owner_scope_order") != list(owner_ids)
        or manifest.get("logical_scope_count") != len(plan.scopes)
        or manifest.get("deduplicated_logical_scope_count")
        != len(plan.scopes) - len(owner_ids)
        or manifest.get("all_physical_owners_have_one_state") is not True
        or manifest.get("logical_alias_state_copies_published") is not False
        or manifest.get("cluster_refit_performed") is not False
        or manifest.get("serialization_policy")
        != "canonical_json_and_individual_npy_only_v1"
        or manifest.get("content_sha256") != _sha256_json(body)
        or not isinstance(rows, list)
        or len(rows) != len(owner_ids)
    ):
        raise ValueError("cluster state bundle manifest is invalid or incomplete")
    if {path.name for path in root.iterdir()} != {
        _CLUSTER_STATE_BUNDLE_MANIFEST,
        "owners",
    }:
        raise ValueError("cluster state bundle contains unregistered entries")
    owners_root = root / "owners"
    expected_directories = {f"{index:03d}" for index in range(len(owner_ids))}
    if (
        owners_root.is_symlink()
        or not owners_root.is_dir()
        or {path.name for path in owners_root.iterdir()} != expected_directories
    ):
        raise ValueError("cluster state bundle owner inventory is incomplete")
    states: dict[str, AuthenticatedClusteredPreflightScopeState] = {}
    row_fields = {
        "canonical_index",
        "physical_owner_scope_id",
        "group_request_content_sha256",
        "relative_manifest_path",
        "state_content_sha256",
        "cluster_fit_identity_content_sha256",
        "state_origin",
    }
    for index, (owner_id, row) in enumerate(zip(owner_ids, rows, strict=True)):
        request = RoleNeutralEmbeddingPhysicalGroupRequest.from_plan(
            plan=plan,
            physical_owner_scope_id=owner_id,
        )
        expected_relative = (
            Path("owners") / f"{index:03d}" / _CLUSTER_STATE_MANIFEST
        ).as_posix()
        if (
            not isinstance(row, Mapping)
            or set(row) != row_fields
            or row.get("canonical_index") != index
            or row.get("physical_owner_scope_id") != owner_id
            or row.get("group_request_content_sha256") != request.content_sha256
            or row.get("relative_manifest_path") != expected_relative
            or row.get("state_origin")
            != "canonical_clustered_preflight_no_refit_v1"
        ):
            raise ValueError("cluster state bundle owner binding changed")
        state = load_canonical_clustered_preflight_scope_state(
            manifest_path=root / expected_relative,
            preflight=preflight,
            request=request,
        )
        fit_identity = state.scope_record.get("cluster_fit_identity")
        if (
            row.get("state_content_sha256") != state.content_sha256
            or not isinstance(fit_identity, Mapping)
            or row.get("cluster_fit_identity_content_sha256")
            != fit_identity.get("content_sha256")
        ):
            raise ValueError("cluster state bundle substituted fitted arrays")
        states[owner_id] = state
    if set(states) != set(owner_ids):
        raise RuntimeError("cluster state bundle omitted a physical owner")
    return AuthenticatedClusteredPreflightStateBundle(
        root=root,
        manifest=copy.deepcopy(manifest),
        states=states,
    )


def _provider_projection(
    *,
    provider: BoundSpentFrozenChunkEmbeddingProvider,
    row_ids: Sequence[int],
    texts: Sequence[str],
    config: RoleNeutralEmbeddingScientificConfig,
    label: str,
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[np.ndarray, ...],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows = tuple(map(int, row_ids))
    exact_texts = tuple(texts)
    if (
        type(provider) is not BoundSpentFrozenChunkEmbeddingProvider
        or tuple(map(int, provider.row_ids)) != rows
        or len(exact_texts) != len(rows)
        or not all(isinstance(value, str) for value in exact_texts)
    ):
        raise ValueError(f"{label} provider/text projection changed row order")
    if tuple(provider.token_bounded_row_ids):
        raise ValueError(f"{label} embedding provider used token-bounded reconciliation")
    metadata = provider.metadata
    required_attestations = {
        "chunk_cap_nonbinding": True,
        "semantic_truncation_allowed": False,
        "tokenizer_truncation_allowed": False,
    }
    if any(metadata.get(key) is not expected for key, expected in required_attestations.items()):
        raise ValueError(
            f"{label} cache lacks a complete nontruncation attestation"
        )
    try:
        chunk_size = int(metadata["chunk_size_words"])
        overlap = int(metadata["chunk_overlap_words"])
        configured_max_chunks = int(metadata["max_chunks"])
        selection = str(metadata["chunk_selection"])
        global_chunk_counts = list(map(int, metadata["chunk_counts"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} cache has incomplete chunk configuration") from exc
    if (
        chunk_size < 1
        or overlap < 0
        or overlap >= chunk_size
        or configured_max_chunks < 1
        or selection not in {"first", "last"}
        or not global_chunk_counts
        or metadata.get("uncapped_chunk_counts_sha256")
        != _sha256_json(global_chunk_counts)
    ):
        raise ValueError(f"{label} cache chunk proof is invalid")
    cached_chunks = provider.chunk_texts(rows)
    matrices = provider.chunk_matrices(rows)
    uncapped_counts: list[int] = []
    for row_id, text, cached, matrix in zip(
        rows,
        exact_texts,
        cached_chunks,
        matrices,
        strict=True,
    ):
        word_count = len(_WORD_RE.findall(text or ""))
        stride = chunk_size - overlap
        uncapped_count = (
            1
            if word_count == 0
            else max(1, int(math.ceil(word_count / stride)))
        )
        # ``chunk_text_words`` produces starts 0,stride,... below word_count.
        # Supplying that exact count makes its cap provably nonbinding.
        uncapped = tuple(
            chunk_text_words(
                text,
                chunk_size,
                overlap,
                uncapped_count,
                selection,
            )
        )
        uncapped_counts.append(len(uncapped))
        if (
            cached != uncapped
            or row_id < 0
            or row_id >= len(global_chunk_counts)
            or global_chunk_counts[row_id] != len(uncapped)
        ):
            raise ValueError(
                f"{label} cache cap would truncate or drop source chunks"
            )
        if config.maximum_source_chunks_per_row is not None and len(cached) > int(
            config.maximum_source_chunks_per_row
        ):
            raise ValueError(
                f"{label} source-chunk capacity would truncate row {row_id}"
            )
        array = np.asarray(matrix)
        if (
            array.ndim != 2
            or array.shape[0] != len(cached)
            or array.shape[1] < 1
            or array.dtype.hasobject
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"{label} chunk matrix is malformed")
    if sum(uncapped_counts) < 1:
        raise ValueError(f"{label} has no source chunks")
    dimensions = {int(np.asarray(value).shape[1]) for value in matrices}
    if len(dimensions) != 1:
        raise ValueError(f"{label} chunk embedding dimensions differ")
    flat_texts = tuple(chunk for chunks in cached_chunks for chunk in chunks)
    flat_matrices = np.concatenate(
        [np.asarray(value, dtype=np.float64) for value in matrices],
        axis=0,
    )
    row_positions = np.concatenate(
        [
            np.full(len(chunks), index, dtype=np.int64)
            for index, chunks in enumerate(cached_chunks)
        ]
    )
    chunk_positions = np.concatenate(
        [
            np.arange(len(chunks), dtype=np.int64)
            for chunks in cached_chunks
        ]
    )
    if (
        len(flat_texts) != flat_matrices.shape[0]
        or row_positions.shape != (len(flat_texts),)
        or chunk_positions.shape != (len(flat_texts),)
    ):
        raise RuntimeError(f"{label} chunk flattening lost alignment")
    return (
        cached_chunks,
        tuple(np.asarray(value, dtype=np.float64) for value in matrices),
        flat_texts,
        flat_matrices,
        row_positions,
        chunk_positions,
    )


def _patient_means(
    matrices: Sequence[np.ndarray],
    *,
    normalize: bool,
    epsilon: float,
    pooling: str,
    norm_order: str,
) -> np.ndarray:
    if pooling != "arithmetic_mean":
        raise ValueError("unsupported patient embedding pooling policy")
    numpy_norm_order = _numpy_norm_order(norm_order)
    means = np.stack(
        [np.asarray(value, dtype=np.float64).mean(axis=0) for value in matrices],
        axis=0,
    )
    if normalize:
        norms = np.linalg.norm(means, ord=numpy_norm_order, axis=1)
        if np.any(norms <= float(epsilon)):
            raise ValueError("patient embedding normalization encountered a zero vector")
        means = means / norms[:, None]
    if not np.isfinite(means).all():
        raise ValueError("patient embedding means are not finite")
    return means


def _numpy_norm_order(configured_order: str) -> int:
    """Map the closed scientific norm policy to NumPy without hidden defaults."""

    if configured_order != "l2":
        raise ValueError("unsupported configured vector norm order")
    return 2


def _target_matrix(
    *,
    targets: Mapping[str, Sequence[float]],
    config: RoleNeutralEmbeddingScientificConfig,
    row_count: int,
) -> np.ndarray:
    if not isinstance(targets, Mapping):
        raise TypeError("fit_targets must be one explicit mapping")
    required = list(_target_order(config))
    if set(targets) != set(required):
        raise ValueError(
            "fit_targets must contain exactly the configured target names"
        )
    columns: list[np.ndarray] = []
    for name in required:
        value = np.asarray(targets[name], dtype=np.float64)
        if value.shape != (row_count,) or not np.isfinite(value).all():
            raise ValueError(f"fit target {name} must align to exact fit rows")
        columns.append(value)
    return np.stack(columns, axis=1)


def _target_order(
    config: RoleNeutralEmbeddingScientificConfig,
) -> tuple[str, ...]:
    names: list[str] = []
    for contrast in config.contrasts:
        for name in (
            contrast.target_name,
            contrast.sample_weight_target_name,
        ):
            if name is not None and name not in names:
                names.append(name)
    return tuple(names)


def _mean_difference_coefficients(
    *,
    positive: np.ndarray,
    negative: np.ndarray,
    sample_weights: np.ndarray | None = None,
    label: str,
    minimum_side_rows: int,
) -> np.ndarray:
    positive = np.asarray(positive, dtype=np.bool_)
    negative = np.asarray(negative, dtype=np.bool_)
    if (
        positive.shape != negative.shape
        or np.any(positive & negative)
        or int(np.sum(positive)) < int(minimum_side_rows)
        or int(np.sum(negative)) < int(minimum_side_rows)
    ):
        raise ValueError(
            f"embedding contrast {label} requires at least "
            f"{int(minimum_side_rows)} disjoint patients on each side"
        )
    if sample_weights is None:
        positive_weights = positive.astype(np.float64)
        negative_weights = negative.astype(np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if (
            weights.shape != positive.shape
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
        ):
            raise ValueError(
                f"embedding contrast {label} sample weights are invalid"
            )
        positive_weights = np.where(positive, weights, 0.0)
        negative_weights = np.where(negative, weights, 0.0)
        if (
            float(np.sum(positive_weights)) <= 0.0
            or float(np.sum(negative_weights)) <= 0.0
        ):
            raise ValueError(
                f"embedding contrast {label} has a zero-weight side"
            )
    positive_weights /= float(np.sum(positive_weights))
    negative_weights /= float(np.sum(negative_weights))
    return positive_weights - negative_weights


def _cell_values(target: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(target, dtype=np.float64)
    rounded = np.rint(values)
    if not np.array_equal(values, rounded) or not set(
        rounded.astype(int).tolist()
    ).issubset({0, 1, 2, 3}):
        raise ValueError(
            f"embedding cell target {label} must use exact 2*T+Y codes"
        )
    return rounded.astype(np.int8)


def _contrast_geometry(
    *,
    target_matrix: np.ndarray,
    patient_embeddings: np.ndarray,
    config: RoleNeutralEmbeddingScientificConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_count = target_matrix.shape[0]
    if row_count < 4:
        raise ValueError("embedding contrasts require at least four fit rows")
    if patient_embeddings.ndim != 2 or patient_embeddings.shape[0] != row_count:
        raise ValueError("embedding contrast geometry lost patient alignment")
    target_names = _target_order(config)
    if target_matrix.shape != (row_count, len(target_names)):
        raise ValueError("embedding target matrix differs from configured order")
    columns = {
        name: np.asarray(target_matrix[:, index], dtype=np.float64)
        for index, name in enumerate(target_names)
    }
    groups = np.full(
        (row_count, len(config.contrasts)),
        -1,
        dtype=np.int8,
    )
    coefficient_columns: list[np.ndarray] = []
    direction_rows: list[np.ndarray] = []
    row_positions = np.arange(row_count, dtype=np.int64)
    for index, contrast in enumerate(config.contrasts):
        target = columns[contrast.target_name]
        rule = contrast.split_rule
        if contrast.split_rule == "binary_zero_one":
            if set(np.unique(target)) != {0.0, 1.0}:
                raise ValueError(
                    f"binary embedding target {contrast.target_name} must contain both 0 and 1"
                )
            coefficients = _mean_difference_coefficients(
                positive=target == 1.0,
                negative=target == 0.0,
                label=contrast.name,
                minimum_side_rows=config.minimum_contrast_side_rows,
            )
        elif rule == "stable_ordered_halves":
            order = np.lexsort((row_positions, target))
            negative_count = row_count // 2
            if negative_count < 1 or negative_count == row_count:
                raise ValueError("ordered-half embedding split is infeasible")
            positive = np.zeros(row_count, dtype=np.bool_)
            negative = np.zeros(row_count, dtype=np.bool_)
            negative[order[:negative_count]] = True
            positive[order[negative_count:]] = True
            coefficients = _mean_difference_coefficients(
                positive=positive,
                negative=negative,
                label=contrast.name,
                minimum_side_rows=config.minimum_contrast_side_rows,
            )
        elif rule == "configured_quantile_tails":
            low, high = np.quantile(
                target,
                [
                    float(config.pseudo_target_quantile),
                    1.0 - float(config.pseudo_target_quantile),
                ],
                method=config.quantile_method,
            )
            negative = target <= low
            positive = target >= high
            overlap = negative & positive
            negative[overlap] = False
            positive[overlap] = False
            weights = (
                None
                if contrast.sample_weight_target_name is None
                else columns[contrast.sample_weight_target_name]
            )
            coefficients = _mean_difference_coefficients(
                positive=positive,
                negative=negative,
                sample_weights=weights,
                label=contrast.name,
                minimum_side_rows=config.minimum_contrast_side_rows,
            )
        elif rule in {
            "treated_arm_outcome_cell_difference",
            "untreated_arm_outcome_cell_difference",
            "treatment_outcome_cell_difference_in_differences",
            "average_normalized_treatment_outcome_marginals",
            "cell_difference_in_differences_residualized_from_marginals",
        }:
            cell = _cell_values(target, label=contrast.name)
            treatment_coefficients = _mean_difference_coefficients(
                positive=cell >= 2,
                negative=cell < 2,
                label=f"{contrast.name}:treatment_marginal",
                minimum_side_rows=config.minimum_contrast_side_rows,
            )
            outcome_coefficients = _mean_difference_coefficients(
                positive=(cell % 2) == 1,
                negative=(cell % 2) == 0,
                label=f"{contrast.name}:outcome_marginal",
                minimum_side_rows=config.minimum_contrast_side_rows,
            )
            if rule == "treated_arm_outcome_cell_difference":
                coefficients = _mean_difference_coefficients(
                    positive=cell == 3,
                    negative=cell == 2,
                    label=contrast.name,
                    minimum_side_rows=config.minimum_contrast_side_rows,
                )
            elif rule == "untreated_arm_outcome_cell_difference":
                coefficients = _mean_difference_coefficients(
                    positive=cell == 1,
                    negative=cell == 0,
                    label=contrast.name,
                    minimum_side_rows=config.minimum_contrast_side_rows,
                )
            else:
                raw_coefficients = (
                    _mean_difference_coefficients(
                        positive=cell == 3,
                        negative=cell == 2,
                        label=f"{contrast.name}:treated",
                        minimum_side_rows=config.minimum_contrast_side_rows,
                    )
                    - _mean_difference_coefficients(
                        positive=cell == 1,
                        negative=cell == 0,
                        label=f"{contrast.name}:untreated",
                        minimum_side_rows=config.minimum_contrast_side_rows,
                    )
                )
                if rule == "treatment_outcome_cell_difference_in_differences":
                    coefficients = raw_coefficients
                else:
                    treatment_direction = (
                        treatment_coefficients @ patient_embeddings
                    )
                    outcome_direction = (
                        outcome_coefficients @ patient_embeddings
                    )
                    treatment_norm = float(
                        np.linalg.norm(
                            treatment_direction,
                            ord=_numpy_norm_order(config.vector_norm_order),
                        )
                    )
                    outcome_norm = float(
                        np.linalg.norm(
                            outcome_direction,
                            ord=_numpy_norm_order(config.vector_norm_order),
                        )
                    )
                    if (
                        treatment_norm <= config.direction_norm_epsilon
                        or outcome_norm <= config.direction_norm_epsilon
                    ):
                        raise ValueError(
                            f"embedding marginal basis is degenerate: "
                            f"{contrast.name}"
                        )
                    normalized_treatment_coefficients = (
                        treatment_coefficients / treatment_norm
                    )
                    normalized_outcome_coefficients = (
                        outcome_coefficients / outcome_norm
                    )
                    if (
                        rule
                        == "average_normalized_treatment_outcome_marginals"
                    ):
                        coefficients = 0.5 * (
                            normalized_treatment_coefficients
                            + normalized_outcome_coefficients
                        )
                    else:
                        raw_direction = (
                            raw_coefficients @ patient_embeddings
                        )
                        basis = np.stack(
                            [
                                treatment_direction / treatment_norm,
                                outcome_direction / outcome_norm,
                            ],
                            axis=1,
                        )
                        try:
                            projection, *_ = np.linalg.lstsq(
                                basis,
                                raw_direction,
                                rcond=config.lstsq_rcond,
                            )
                        except np.linalg.LinAlgError as exc:
                            raise ValueError(
                                "embedding residualized interaction basis "
                                "solve failed"
                            ) from exc
                        coefficients = (
                            raw_coefficients
                            - projection[0]
                            * normalized_treatment_coefficients
                            - projection[1]
                            * normalized_outcome_coefficients
                        )
        else:  # pragma: no cover - constructor closes this branch.
            raise RuntimeError("embedding contrast split rule escaped validation")
        direction = coefficients @ patient_embeddings
        norm = float(
            np.linalg.norm(
                direction,
                ord=_numpy_norm_order(config.vector_norm_order),
            )
        )
        if (
            not np.isfinite(coefficients).all()
            or not math.isfinite(norm)
            or norm <= config.direction_norm_epsilon
        ):
            raise ValueError(
                f"whole-cohort embedding direction is degenerate: "
                f"{contrast.name}"
            )
        groups[coefficients < 0.0, index] = 0
        groups[coefficients > 0.0, index] = 1
        if not {0, 1}.issubset(set(groups[:, index].tolist())):
            raise RuntimeError("embedding target split produced an empty side")
        coefficient_columns.append(coefficients)
        direction_rows.append(direction / norm)
    return (
        groups,
        np.stack(coefficient_columns, axis=1),
        np.stack(direction_rows, axis=0),
    )


def _semantic_vectorizer(
    config: RoleNeutralEmbeddingScientificConfig,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        input=config.semantic_input,
        encoding=config.semantic_encoding,
        decode_error=config.semantic_decode_error,
        preprocessor=config.semantic_preprocessor,
        tokenizer=config.semantic_tokenizer,
        analyzer=config.semantic_analyzer,
        lowercase=bool(config.semantic_lowercase),
        strip_accents=config.semantic_strip_accents,
        token_pattern=config.semantic_token_pattern,
        ngram_range=(
            int(config.semantic_ngram_min),
            int(config.semantic_ngram_max),
        ),
        min_df=int(config.semantic_min_df),
        max_df=float(config.semantic_max_df),
        binary=bool(config.semantic_binary),
        sublinear_tf=bool(config.semantic_sublinear_tf),
        norm=config.semantic_norm,
        use_idf=bool(config.semantic_use_idf),
        smooth_idf=bool(config.semantic_smooth_idf),
        stop_words=(
            None
            if config.semantic_stop_words is None
            else (
                config.semantic_stop_words
                if isinstance(config.semantic_stop_words, str)
                else list(config.semantic_stop_words)
            )
        ),
        max_features=config.semantic_max_features,
        vocabulary=config.semantic_vocabulary,
        dtype=(
            np.float32
            if config.semantic_dtype == "float32"
            else np.float64
        ),
    )


@dataclass
class _FitComputation:
    chunk_texts_by_row: tuple[tuple[str, ...], ...]
    flat_chunk_texts: tuple[str, ...]
    chunk_embeddings: np.ndarray
    chunk_row_positions: np.ndarray
    chunk_positions: np.ndarray
    patient_embeddings: np.ndarray
    target_matrix: np.ndarray
    group_memberships: np.ndarray
    direction_matrix: np.ndarray
    patient_scores: np.ndarray
    chunk_scores: np.ndarray
    vocabulary: tuple[str, ...]
    semantic_idf: np.ndarray
    semantic_signed_scores: np.ndarray
    vectorizer: TfidfVectorizer
    evidence_payloads: Mapping[str, Mapping[str, Any]]


def _cluster_architecture_evidence(
    scope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = scope["cluster_fit_identity"]["final_catalog_concepts"][
        EMBEDDING_CLUSTERED
    ]
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("content"), Mapping):
            raise ValueError("clustered preflight concept is malformed")
        content = copy.deepcopy(dict(row["content"]))
        contrast = content.get("contrast")
        family = (
            str(contrast.get("contrast_family"))
            if isinstance(contrast, Mapping)
            else ""
        )
        if family not in _CLUSTER_CONTRAST_AXES:
            raise ValueError("clustered preflight concept has another family")
        serialized = _canonical_json(content).casefold()
        if any(
            forbidden in serialized
            for forbidden in (
                "positive_aligned_chunks",
                "negative_aligned_chunks",
                "raw_note",
                "full_note",
            )
        ):
            raise ValueError("clustered preflight reviewer evidence retains source excerpts")
        output.append(
            {
                "atom_kind": str(row.get("atom_kind") or "embedding_contrast"),
                "source_kind": "legacy_all_source",
                "observable_axes": list(_CLUSTER_CONTRAST_AXES[family]),
                "content": content,
                "canonical_preflight_scope_reused": True,
                "canonical_preflight_atom_index": index,
            }
        )
    if not output:
        raise ValueError("clustered preflight produced no native evidence")
    return output


def _preflight_semantic_architecture_evidence(
    scope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = scope["cluster_fit_identity"]["final_catalog_concepts"][
        TFIDF_SEMANTIC_RETRIEVAL
    ]
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("content"), Mapping):
            raise ValueError("clustered semantic preflight concept is malformed")
        content = copy.deepcopy(dict(row["content"]))
        if (
            content.get("architecture_view") != SEMANTIC_RETRIEVAL_DERIVATION
            or content.get("source_passages_removed") is not True
        ):
            raise ValueError("clustered semantic preflight view retains source passages")
        contrast = content.get("contrast")
        family = (
            str(contrast.get("contrast_family"))
            if isinstance(contrast, Mapping)
            else ""
        )
        if family not in _CLUSTER_CONTRAST_AXES:
            raise ValueError("clustered semantic preflight view has another family")
        output.append(
            {
                "atom_kind": "tfidf_semantic_retrieval_contrast",
                "source_kind": "legacy_all_source",
                "observable_axes": list(_CLUSTER_CONTRAST_AXES[family]),
                "content": content,
                "canonical_preflight_scope_reused": True,
                "canonical_preflight_atom_index": index,
            }
        )
    if not output:
        raise ValueError("clustered preflight produced no semantic retrieval evidence")
    return output


def _embedding_evidence_payloads(
    *,
    config: RoleNeutralEmbeddingScientificConfig,
    vocabulary: Sequence[str],
    semantic_signed_scores: np.ndarray,
    source_chunk_count: int,
    cluster_scope: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    terms = tuple(map(str, vocabulary))
    scores = np.asarray(semantic_signed_scores, dtype=np.float64)
    if scores.shape != (len(config.contrasts), len(terms)):
        raise ValueError("semantic evidence scores differ from configured vocabulary")
    whole_evidence: list[dict[str, Any]] = []
    lexical_evidence: list[dict[str, Any]] = []
    for index, contrast in enumerate(config.contrasts):
        axes = _axes_for_whole_contrast(
            contrast.contrast_family,
            contrast.name,
        )
        witnesses = [
            {
                "concept": term,
                "score": float(score),
                "semantic_feature_index": term_index,
            }
            for term_index, (term, score) in enumerate(
                zip(terms, scores[index], strict=True)
            )
        ]
        batch_size = int(config.semantic_member_batch_size)
        batches = tuple(
            witnesses[start : start + batch_size]
            for start in range(0, len(witnesses), batch_size)
        )
        if not batches:
            raise RuntimeError(
                "configured embedding vocabulary produced no semantic-member "
                "batch"
            )
        for batch_index, batch in enumerate(batches, start=1):
            common = {
                "contrast": {
                    "name": contrast.name,
                    "contrast_family": contrast.contrast_family,
                    "direction_source": (
                        f"fit_target:{contrast.target_name}:{contrast.split_rule}"
                    ),
                },
                "concept_witnesses": copy.deepcopy(batch),
                "full_member_count": len(witnesses),
                "member_batch_index": batch_index,
                "member_batch_count": len(batches),
                "source_chunk_count": int(source_chunk_count),
                "all_source_chunks_accounted_once": True,
                "all_configured_semantic_terms_accounted_once": True,
            }
            whole_evidence.append(
                {
                    "atom_kind": "embedding_contrast",
                    "source_kind": "legacy_all_source",
                    "observable_axes": list(axes),
                    "content": {
                        "architecture_view": "embedding_contrast",
                        **copy.deepcopy(common),
                    },
                }
            )
            lexical_evidence.append(
                {
                    "atom_kind": "tfidf_semantic_retrieval_contrast",
                    "source_kind": "legacy_all_source",
                    "observable_axes": list(axes),
                    "content": {
                        "architecture_view": SEMANTIC_RETRIEVAL_DERIVATION,
                        "source_passages_removed": True,
                        **copy.deepcopy(common),
                    },
                }
            )
    cluster_evidence = _cluster_architecture_evidence(cluster_scope)
    lexical_evidence.extend(
        _preflight_semantic_architecture_evidence(cluster_scope)
    )
    payloads = {
        EMBEDDING_WHOLE_COHORT: {
            "schema_version": NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
            "family": EMBEDDING_WHOLE_COHORT,
            "architecture_evidence": whole_evidence,
        },
        EMBEDDING_CLUSTERED: {
            "schema_version": NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
            "family": EMBEDDING_CLUSTERED,
            "architecture_evidence": cluster_evidence,
        },
        TFIDF_SEMANTIC_RETRIEVAL: {
            "schema_version": NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
            "family": TFIDF_SEMANTIC_RETRIEVAL,
            "architecture_evidence": lexical_evidence,
        },
    }
    if any(not payload["architecture_evidence"] for payload in payloads.values()):
        raise RuntimeError("one embedding evidence family is empty")
    return payloads


def _fit_embedding_families(
    *,
    config: RoleNeutralEmbeddingScientificConfig,
    row_ids: tuple[int, ...],
    texts: tuple[str, ...],
    provider: BoundSpentFrozenChunkEmbeddingProvider,
    targets: Mapping[str, Sequence[float]],
    cluster_scope: Mapping[str, Any],
) -> _FitComputation:
    # Both native clustered and lexical-semantic evidence consume this scope.
    # Materialize the lazy portable capability once so the two projections do
    # not independently reconstruct and copy the same lossless owner payload.
    cluster_scope = _materialize_cluster_scope_for_one_operation(
        cluster_scope
    )
    (
        chunks_by_row,
        matrices,
        flat_texts,
        flat_embeddings,
        row_positions,
        chunk_positions,
    ) = _provider_projection(
        provider=provider,
        row_ids=row_ids,
        texts=texts,
        config=config,
        label="fit",
    )
    patient_embeddings = _patient_means(
        matrices,
        normalize=bool(config.normalize_patient_embeddings),
        epsilon=float(config.direction_norm_epsilon),
        pooling=config.patient_embedding_pooling,
        norm_order=config.vector_norm_order,
    )
    target_matrix = _target_matrix(
        targets=targets,
        config=config,
        row_count=len(row_ids),
    )
    groups, coefficient_matrix, direction_matrix = _contrast_geometry(
        target_matrix=target_matrix,
        patient_embeddings=patient_embeddings,
        config=config,
    )
    for index, contrast in enumerate(config.contrasts):
        positive = groups[:, index] == 1
        negative = groups[:, index] == 0
        positive_chunk_count = int(
            sum(len(chunks_by_row[position]) for position in np.flatnonzero(positive))
        )
        negative_chunk_count = int(
            sum(len(chunks_by_row[position]) for position in np.flatnonzero(negative))
        )
        limit = config.maximum_retrieval_chunks_per_side
        if limit is not None and max(positive_chunk_count, negative_chunk_count) > int(limit):
            raise ValueError(
                "configured retrieval capacity would truncate source chunks in "
                f"contrast {contrast.name}"
            )
    patient_scores = patient_embeddings @ direction_matrix.T
    chunk_scores = flat_embeddings @ direction_matrix.T
    vectorizer = _semantic_vectorizer(config)
    try:
        lexical_matrix = vectorizer.fit_transform(flat_texts)
    except ValueError as exc:
        raise ValueError("configured semantic vocabulary is empty") from exc
    vocabulary = tuple(map(str, vectorizer.get_feature_names_out()))
    if not vocabulary or len(vocabulary) != len(set(vocabulary)):
        raise RuntimeError("semantic vocabulary is empty or duplicated")
    if (
        config.maximum_semantic_terms is not None
        and len(vocabulary) > int(config.maximum_semantic_terms)
    ):
        raise ValueError(
            "configured semantic-term capacity would truncate fitted terms"
        )
    semantic_idf = (
        np.asarray(vectorizer.idf_, dtype=np.float64)
        if bool(config.semantic_use_idf)
        else np.ones(len(vocabulary), dtype=np.float64)
    )
    chunks_per_row = np.asarray(
        [len(value) for value in chunks_by_row],
        dtype=np.float64,
    )
    signed_rows: list[np.ndarray] = []
    for index, _contrast in enumerate(config.contrasts):
        chunk_coefficients = (
            coefficient_matrix[row_positions, index]
            / chunks_per_row[row_positions]
        )
        signed = np.asarray(
            chunk_coefficients @ lexical_matrix
        ).reshape(-1)
        if signed.shape != (len(vocabulary),) or not np.isfinite(signed).all():
            raise RuntimeError("semantic signed scores lost vocabulary alignment")
        signed_rows.append(signed)
    semantic_signed_scores = np.stack(signed_rows, axis=0)
    payloads = _embedding_evidence_payloads(
        config=config,
        vocabulary=vocabulary,
        semantic_signed_scores=semantic_signed_scores,
        source_chunk_count=len(flat_texts),
        cluster_scope=cluster_scope,
    )
    return _FitComputation(
        chunk_texts_by_row=chunks_by_row,
        flat_chunk_texts=flat_texts,
        chunk_embeddings=np.ascontiguousarray(flat_embeddings),
        chunk_row_positions=np.ascontiguousarray(row_positions),
        chunk_positions=np.ascontiguousarray(chunk_positions),
        patient_embeddings=np.ascontiguousarray(patient_embeddings),
        target_matrix=np.ascontiguousarray(target_matrix),
        group_memberships=np.ascontiguousarray(groups),
        direction_matrix=np.ascontiguousarray(direction_matrix),
        patient_scores=np.ascontiguousarray(patient_scores),
        chunk_scores=np.ascontiguousarray(chunk_scores),
        vocabulary=vocabulary,
        semantic_idf=np.ascontiguousarray(semantic_idf),
        semantic_signed_scores=np.ascontiguousarray(semantic_signed_scores),
        vectorizer=vectorizer,
        evidence_payloads=payloads,
    )


def _tree_sha256(root: Path) -> str:
    tree = Path(root)
    if tree.is_symlink() or not tree.is_dir():
        raise ValueError("artifact tree must be one real directory")
    rows: list[dict[str, Any]] = []
    for path in sorted(
        tree.rglob("*"),
        key=lambda value: value.relative_to(tree).as_posix(),
    ):
        relative = path.relative_to(tree).as_posix()
        if path.is_symlink():
            raise ValueError("artifact tree cannot contain symbolic links")
        if path.is_dir():
            rows.append({"path": relative, "kind": "directory"})
        else:
            digest, size = _sha256_file(path)
            rows.append(
                {
                    "path": relative,
                    "kind": "file",
                    "sha256": digest,
                    "size_bytes": size,
                }
            )
    if not rows:
        raise ValueError("artifact tree is empty")
    return _sha256_json(
        {
            "schema_version": "production_role_neutral_embedding_tree_v1",
            "inventory": rows,
        }
    )


def _producer_identity() -> str:
    path = Path(__file__).resolve(strict=True)
    module_sha256, _ = _sha256_file(path)
    chunker_sha256 = hashlib.sha256(
        inspect.getsource(chunk_text_words).encode("utf-8")
    ).hexdigest()
    body = {
        "schema_version": "production_role_neutral_embedding_producer_identity_v1",
        "module_sha256": module_sha256,
        "chunker_source_sha256": chunker_sha256,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "serialization": "canonical_json_and_individual_npy_v1",
        "executable_serialization": False,
    }
    return _sha256_json(body)


def _file_registration(path: Path, *, relative_to: Path) -> dict[str, Any]:
    digest, size = _sha256_file(path)
    return {
        "relative_path": path.relative_to(relative_to).as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def _persist_fit_state(
    *,
    root: Path,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
    config: RoleNeutralEmbeddingScientificConfig,
    fit_texts: tuple[str, ...],
    fit_targets: Mapping[str, Sequence[float]],
    provider_identity: Mapping[str, Any],
    cluster_state: AuthenticatedClusteredPreflightScopeState,
    computation: _FitComputation,
) -> tuple[dict[str, Any], str]:
    fit_root = root / _FIT_STATE_DIRECTORY
    fit_root.mkdir(parents=True, exist_ok=False)
    arrays_root = fit_root / "arrays"
    arrays_root.mkdir(parents=True, exist_ok=False)
    arrays = {
        "fit_chunk_embeddings": computation.chunk_embeddings,
        "fit_chunk_row_positions": computation.chunk_row_positions,
        "fit_chunk_positions": computation.chunk_positions,
        "fit_patient_embeddings": computation.patient_embeddings,
        "fit_target_matrix": computation.target_matrix,
        "fit_group_memberships": computation.group_memberships,
        "whole_direction_matrix": computation.direction_matrix,
        "fit_patient_direction_scores": computation.patient_scores,
        "fit_chunk_direction_scores": computation.chunk_scores,
        "semantic_idf": computation.semantic_idf,
        "semantic_signed_scores": computation.semantic_signed_scores,
    }
    array_registrations: dict[str, dict[str, Any]] = {}
    for key in sorted(arrays):
        path = arrays_root / f"{key}.npy"
        _write_new_npy(path, arrays[key])
        array_registrations[key] = _array_registration(
            path,
            relative_to=fit_root,
            value=arrays[key],
        )
    chunk_body = {
        "schema_version": "production_role_neutral_embedding_source_chunks_v1",
        "row_order": list(request.physical_owner.fit_row_ids),
        "rows": [
            {
                "row_id": int(row_id),
                "chunks": list(chunks),
                "chunk_count": len(chunks),
            }
            for row_id, chunks in zip(
                request.physical_owner.fit_row_ids,
                computation.chunk_texts_by_row,
                strict=True,
            )
        ],
        "flat_chunk_count": len(computation.flat_chunk_texts),
        "all_uncapped_source_chunks_accounted_once": True,
        "text_truncation_applied": False,
    }
    chunks = {**chunk_body, "content_sha256": _sha256_json(chunk_body)}
    chunks_path = fit_root / _FIT_CHUNKS
    _write_new_json(chunks_path, chunks)
    vocabulary_body = {
        "schema_version": "production_role_neutral_embedding_semantic_vocabulary_v1",
        "terms": list(computation.vocabulary),
        "term_count": len(computation.vocabulary),
        "feature_indices": list(range(len(computation.vocabulary))),
        "all_configured_terms_accounted_once": True,
        "semantic_term_truncation_applied": False,
    }
    vocabulary = {
        **vocabulary_body,
        "content_sha256": _sha256_json(vocabulary_body),
    }
    vocabulary_path = fit_root / _FIT_VOCABULARY
    _write_new_json(vocabulary_path, vocabulary)
    target_hashes = {
        name: _float_hex_sha256(np.asarray(fit_targets[name], dtype=np.float64))
        for name in sorted(fit_targets)
    }
    provider_closed = copy.deepcopy(dict(provider_identity))
    metadata_body = {
        "schema_version": ROLE_NEUTRAL_EMBEDDING_FIT_STATE_SCHEMA,
        "group_request_content_sha256": request.content_sha256,
        "plan_scientific_content_sha256": request.plan_scientific_content_sha256,
        "physical_owner_scope_id": request.physical_owner.scope_id,
        "physical_owner_scope_sha256": request.physical_owner.as_dict()["scope_sha256"],
        "fit_row_ids": list(request.physical_owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(
            request.physical_owner.fit_row_ids
        ),
        "canonical_group_seed": int(request.physical_owner.scope_seed),
        "fit_text_sha256": _ordered_text_sha256(
            request.physical_owner.fit_row_ids,
            fit_texts,
        ),
        "fit_target_order": list(_target_order(config)),
        "fit_target_sha256": target_hashes,
        "scientific_configuration": config.as_dict(),
        "configuration_identity_sha256": config.content_sha256,
        "producer_identity_sha256": _producer_identity(),
        "runtime_compatibility_class": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "machine_or_device_identifiers_included": False,
        },
        "embedding_provider_identity": provider_closed,
        "embedding_provider_identity_sha256": _sha256_json(provider_closed),
        "canonical_clustered_preflight_state_content_sha256": (
            cluster_state.content_sha256
        ),
        "canonical_clustered_preflight_binding": copy.deepcopy(
            dict(cluster_state.manifest["preflight_binding"])
        ),
        "cluster_refit_performed": False,
        "source_chunks": _file_registration(chunks_path, relative_to=fit_root),
        "source_chunks_content_sha256": chunks["content_sha256"],
        "semantic_vocabulary": _file_registration(
            vocabulary_path,
            relative_to=fit_root,
        ),
        "semantic_vocabulary_content_sha256": vocabulary["content_sha256"],
        "array_order": sorted(array_registrations),
        "array_inventory": array_registrations,
        "array_layout": "one_npy_per_array_mmap_safe_v1",
        "evidence_payload_sha256": {
            family: _sha256_json(computation.evidence_payloads[family])
            for family in _FAMILIES
        },
        "family_independence": {
            "whole_cohort_native_nonempty": bool(
                computation.evidence_payloads[EMBEDDING_WHOLE_COHORT][
                    "architecture_evidence"
                ]
            ),
            "cluster_local_native_nonempty": bool(
                computation.evidence_payloads[EMBEDDING_CLUSTERED][
                    "architecture_evidence"
                ]
            ),
            "lexical_semantic_retrieval_native_nonempty": bool(
                computation.evidence_payloads[TFIDF_SEMANTIC_RETRIEVAL][
                    "architecture_evidence"
                ]
            ),
            "whole_and_cluster_are_distinct_payloads": (
                _sha256_json(
                    computation.evidence_payloads[EMBEDDING_WHOLE_COHORT]
                )
                != _sha256_json(
                    computation.evidence_payloads[EMBEDDING_CLUSTERED]
                )
            ),
        },
        "fit_source_chunk_count": len(computation.flat_chunk_texts),
        "fit_semantic_term_count": len(computation.vocabulary),
        "all_source_chunks_accounted_once": True,
        "all_configured_semantic_terms_accounted_once": True,
        "registered_heldout_text_accessed": False,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
        "semantic_term_truncation_applied": False,
        "pickle_joblib_npz_or_compression_used": False,
    }
    metadata = {**metadata_body, "content_sha256": _sha256_json(metadata_body)}
    _write_new_json(fit_root / _FIT_METADATA, metadata)
    return metadata, _tree_sha256(fit_root)


def _fit_seal(
    *,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
    config: RoleNeutralEmbeddingScientificConfig,
    family: str,
    evidence_payload: Mapping[str, Any],
    fit_state_sha256: str,
) -> dict[str, Any]:
    if family not in _FAMILIES:
        raise ValueError("embedding fit seal names another family")
    payload = copy.deepcopy(dict(evidence_payload))
    if (
        payload.get("schema_version")
        != NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION
        or payload.get("family") != family
        or not isinstance(payload.get("architecture_evidence"), list)
        or not payload["architecture_evidence"]
    ):
        raise ValueError("embedding fit seal requires nonempty native evidence")
    payload_sha256 = _sha256_json(payload)
    owner = request.physical_owner
    events = [
        {
            "sequence": 1,
            "event": "fit_completed",
            "fit_state_artifact_sha256": fit_state_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
        },
        {
            "sequence": 2,
            "event": "fit_family_artifact_sealed",
            "fit_state_artifact_sha256": fit_state_sha256,
            "evidence_payload_sha256": payload_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
        },
    ]
    body = {
        "schema_version": LEGACY_STAGE1_FIT_ONLY_FAMILY_SEAL_SCHEMA,
        "plan_scientific_content_sha256": request.plan_scientific_content_sha256,
        "physical_owner_scope_id": owner.scope_id,
        "physical_owner_scope_sha256": owner.as_dict()["scope_sha256"],
        "family": family,
        "fit_row_ids": list(owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(owner.fit_row_ids),
        "canonical_group_seed": int(owner.scope_seed),
        "producer_identity_sha256": _producer_identity(),
        "configuration_identity_sha256": config.content_sha256,
        "fit_state_artifact_sha256": fit_state_sha256,
        "evidence_payload_sha256": payload_sha256,
        "evidence_payload": payload,
        "event_order": events,
        "logical_view_transform_started": False,
        "registered_heldout_text_accessed": False,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _safe_scope_filename(scope_id: str) -> str:
    value = str(scope_id)
    if not value or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise ValueError("logical scope ID is not safe as one artifact filename")
    return value


def _logical_view_filename(scope_id: str, family: str) -> str:
    family_slug = {
        EMBEDDING_WHOLE_COHORT: "embedding_whole_cohort",
        EMBEDDING_CLUSTERED: "embedding_clustered",
        TFIDF_SEMANTIC_RETRIEVAL: "tfidf_semantic_retrieval",
    }[family]
    return f"{_safe_scope_filename(scope_id)}.{family_slug}.json"


def _reference_only_view(
    *,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
    member: Stage1ScopeSpec,
    family: str,
    seal: Mapping[str, Any],
    seal_registration: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": ROLE_NEUTRAL_EMBEDDING_LOGICAL_VIEW_SCHEMA,
        "group_request_content_sha256": request.content_sha256,
        "plan_scientific_content_sha256": request.plan_scientific_content_sha256,
        "logical_scope_id": member.scope_id,
        "logical_scope_sha256": member.as_dict()["scope_sha256"],
        "logical_purpose": member.scope_kind,
        "physical_owner_scope_id": request.physical_owner.scope_id,
        "family": family,
        "fit_only_family_seal_sha256": seal_registration["sha256"],
        "fit_only_family_seal_content_sha256": seal["content_sha256"],
        "view_input_policy": "fit_only_reference_no_heldout_open_v1",
        "logical_transform_performed": False,
        "prediction_artifacts": [],
        "registered_heldout_text_accessed": False,
        "registered_heldout_labels_accessed": False,
        "reuses_canonical_physical_fit": True,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _payload_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        if path == root / _TERMINAL_FILE:
            continue
        if path.is_symlink():
            raise ValueError("embedding artifact tree contains a symbolic link")
        if path.is_dir():
            continue
        digest, size = _sha256_file(path)
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    if not rows or [row["relative_path"] for row in rows] != sorted(
        row["relative_path"] for row in rows
    ):
        raise RuntimeError("embedding payload inventory is empty or reordered")
    return rows


def _exact_transform_values(
    *,
    batch: ExactHeldoutEmbeddingBatch,
    config: RoleNeutralEmbeddingScientificConfig,
    computation: _FitComputation,
    cluster_state: AuthenticatedClusteredPreflightScopeState,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
]:
    (
        chunks_by_row,
        matrices,
        flat_texts,
        flat_embeddings,
        row_positions,
        chunk_positions,
    ) = _provider_projection(
        provider=batch.embedding_provider,
        row_ids=batch.row_ids,
        texts=batch.texts,
        config=config,
        label="exact held-out",
    )
    patient_embeddings = _patient_means(
        matrices,
        normalize=bool(config.normalize_patient_embeddings),
        epsilon=float(config.direction_norm_epsilon),
        pooling=config.patient_embedding_pooling,
        norm_order=config.vector_norm_order,
    )
    whole_patient_scores = patient_embeddings @ computation.direction_matrix.T
    whole_chunk_scores = flat_embeddings @ computation.direction_matrix.T
    centers = np.asarray(
        cluster_state.arrays["cluster_kmeans_centers"],
        dtype=np.float64,
    )
    if centers.ndim != 2 or centers.shape[1] != patient_embeddings.shape[1]:
        raise ValueError(
            "canonical clustered preflight centers do not match held-out embeddings"
        )
    cluster_distances = np.linalg.norm(
        patient_embeddings[:, None, :] - centers[None, :, :],
        ord=_numpy_norm_order(config.vector_norm_order),
        axis=2,
    )
    cluster_assignments = np.argmin(cluster_distances, axis=1).astype(np.int64)
    arrays: dict[str, np.ndarray] = {
        "heldout_patient_embeddings": np.ascontiguousarray(patient_embeddings),
        "heldout_chunk_embeddings": np.ascontiguousarray(flat_embeddings),
        "heldout_chunk_row_positions": np.ascontiguousarray(row_positions),
        "heldout_chunk_positions": np.ascontiguousarray(chunk_positions),
        "heldout_whole_patient_scores": np.ascontiguousarray(whole_patient_scores),
        "heldout_whole_chunk_scores": np.ascontiguousarray(whole_chunk_scores),
        "heldout_cluster_distances": np.ascontiguousarray(cluster_distances),
        "heldout_cluster_assignments": np.ascontiguousarray(cluster_assignments),
    }
    svd_projection_keys: list[dict[str, Any]] = []
    for index, state in enumerate(
        cluster_state.manifest["state_metadata"]["svd_states"]
    ):
        components = np.asarray(
            cluster_state.arrays[str(state["components"])],
            dtype=np.float64,
        )
        if components.ndim != 2 or components.shape[1] != patient_embeddings.shape[1]:
            raise ValueError(
                "canonical clustered preflight component basis does not match "
                "held-out embeddings"
            )
        key = f"heldout_cluster_svd_{index}_projections"
        arrays[key] = np.ascontiguousarray(patient_embeddings @ components.T)
        svd_projection_keys.append(
            {
                "family_key": str(state["family_key"]),
                "array_key": key,
                "component_count": int(components.shape[0]),
            }
        )
    lexical_chunk_matrix = computation.vectorizer.transform(flat_texts).tocsr()
    if lexical_chunk_matrix.shape != (
        len(flat_texts),
        len(computation.vocabulary),
    ):
        raise RuntimeError("held-out lexical transform changed fitted vocabulary")
    counts = np.bincount(
        row_positions,
        minlength=len(batch.row_ids),
    ).astype(np.float64)
    if np.any(counts < 1):
        raise RuntimeError("held-out lexical transform omitted a patient")
    aggregator = sparse.csr_matrix(
        (
            1.0 / counts[row_positions],
            (row_positions, np.arange(len(row_positions), dtype=np.int64)),
        ),
        shape=(len(batch.row_ids), len(row_positions)),
        dtype=np.float64,
    )
    lexical_patient_matrix = (aggregator @ lexical_chunk_matrix).tocsr()
    lexical_patient_matrix.sort_indices()
    arrays.update(
        {
            "heldout_lexical_csr_data": np.ascontiguousarray(
                lexical_patient_matrix.data
            ),
            "heldout_lexical_csr_indices": np.ascontiguousarray(
                lexical_patient_matrix.indices.astype(np.int64)
            ),
            "heldout_lexical_csr_indptr": np.ascontiguousarray(
                lexical_patient_matrix.indptr.astype(np.int64)
            ),
        }
    )
    metadata = {
        "heldout_source_chunk_counts": [len(value) for value in chunks_by_row],
        "heldout_flat_chunk_count": len(flat_texts),
        "whole_contrast_names": [value.name for value in config.contrasts],
        "cluster_svd_projections": svd_projection_keys,
        "lexical_csr_shape": list(lexical_patient_matrix.shape),
        "lexical_csr_nnz": int(lexical_patient_matrix.nnz),
        "semantic_vocabulary_sha256": _sha256_json(
            list(computation.vocabulary)
        ),
        "all_heldout_source_chunks_transformed_once": True,
        "heldout_labels_accessed": False,
    }
    return arrays, metadata


def _persist_exact_transform(
    *,
    root: Path,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
    batch: ExactHeldoutEmbeddingBatch,
    arrays: Mapping[str, np.ndarray],
    transform_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    exact_root = root / _EXACT_DIRECTORY
    exact_root.mkdir(parents=True, exist_ok=False)
    arrays_root = exact_root / "arrays"
    arrays_root.mkdir(parents=True, exist_ok=False)
    registrations: dict[str, dict[str, Any]] = {}
    for key in sorted(arrays):
        path = arrays_root / f"{key}.npy"
        _write_new_npy(path, arrays[key])
        registrations[key] = _array_registration(
            path,
            relative_to=root,
            value=arrays[key],
        )
    body = {
        "schema_version": "production_role_neutral_embedding_exact_transform_v1",
        "group_request_content_sha256": request.content_sha256,
        "plan_scientific_content_sha256": request.plan_scientific_content_sha256,
        "logical_scope_id": request.physical_owner.scope_id,
        "logical_scope_sha256": request.physical_owner.as_dict()["scope_sha256"],
        "heldout_row_ids": list(batch.row_ids),
        "heldout_row_order_fingerprint": _row_order_fingerprint(batch.row_ids),
        "heldout_text_sha256": _ordered_text_sha256(batch.row_ids, batch.texts),
        "transform_metadata": copy.deepcopy(dict(transform_metadata)),
        "array_order": sorted(registrations),
        "array_inventory": registrations,
        "array_layout": "one_npy_per_array_mmap_safe_v1",
        "fit_seals_preexisted": True,
        "registered_heldout_text_accessed": True,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
        "semantic_term_truncation_applied": False,
        "pickle_joblib_npz_or_compression_used": False,
    }
    metadata = {**body, "content_sha256": _sha256_json(body)}
    _write_new_json(exact_root / "metadata.json", metadata)
    return metadata, registrations


def _exact_view(
    *,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
    family: str,
    seal: Mapping[str, Any],
    seal_registration: Mapping[str, Any],
    exact_metadata: Mapping[str, Any],
    array_registrations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    family_keys = {
        EMBEDDING_WHOLE_COHORT: [
            "heldout_chunk_row_positions",
            "heldout_chunk_positions",
            "heldout_whole_patient_scores",
            "heldout_whole_chunk_scores",
        ],
        EMBEDDING_CLUSTERED: [
            "heldout_cluster_distances",
            "heldout_cluster_assignments",
            *[
                row["array_key"]
                for row in exact_metadata["transform_metadata"][
                    "cluster_svd_projections"
                ]
            ],
        ],
        TFIDF_SEMANTIC_RETRIEVAL: [
            "heldout_lexical_csr_data",
            "heldout_lexical_csr_indices",
            "heldout_lexical_csr_indptr",
        ],
    }
    selected = [
        copy.deepcopy(dict(array_registrations[key]))
        for key in family_keys[family]
    ]
    body = {
        "schema_version": ROLE_NEUTRAL_EMBEDDING_LOGICAL_VIEW_SCHEMA,
        "group_request_content_sha256": request.content_sha256,
        "plan_scientific_content_sha256": request.plan_scientific_content_sha256,
        "logical_scope_id": request.physical_owner.scope_id,
        "logical_scope_sha256": request.physical_owner.as_dict()["scope_sha256"],
        "logical_purpose": request.physical_owner.scope_kind,
        "physical_owner_scope_id": request.physical_owner.scope_id,
        "family": family,
        "fit_only_family_seal_sha256": seal_registration["sha256"],
        "fit_only_family_seal_content_sha256": seal["content_sha256"],
        "view_input_policy": "heldout_row_text_embedding_no_labels_v1",
        "logical_transform_performed": True,
        "prediction_artifacts": selected,
        "exact_transform_content_sha256": exact_metadata["content_sha256"],
        "registered_heldout_text_accessed": True,
        "registered_heldout_labels_accessed": False,
        "reuses_canonical_physical_fit": True,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def execute_role_neutral_embedding_physical_group(
    *,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
    output_root: Path | str,
    fit_texts: Sequence[str],
    fit_targets: Mapping[str, Sequence[float]],
    fit_embedding_provider: BoundSpentFrozenChunkEmbeddingProvider,
    scientific_config: RoleNeutralEmbeddingScientificConfig,
    clustered_preflight: _ClusterPreflightArtifact,
    clustered_preflight_state_manifest: Path | str,
    exact_heldout_loader: (
        Callable[[tuple[int, ...]], ExactHeldoutEmbeddingBatch] | None
    ),
) -> dict[str, Any]:
    """Fit/seal all three families, then open the primary held-out rows.

    ``exact_heldout_loader`` retains its original public name for callers
    compiled against the exact-inner-only implementation.  "Exact" now means
    the canonical primary owner's exact held-out row order; it is required for
    full-outer, exact-inner, and independent cumulative physical owners.
    """

    if not isinstance(request, RoleNeutralEmbeddingPhysicalGroupRequest):
        raise TypeError("embedding execution requires its typed group request")
    request.as_dict()
    if not isinstance(scientific_config, RoleNeutralEmbeddingScientificConfig):
        raise TypeError("embedding execution requires explicit scientific configuration")
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("embedding output root must be absolute")
    if root.exists() or root.is_symlink():
        raise FileExistsError("embedding output root must be fresh")
    texts = tuple(fit_texts)
    if (
        len(texts) != request.physical_owner.fit_row_count
        or not all(isinstance(value, str) for value in texts)
    ):
        raise ValueError("fit text must align to the physical owner's ordered rows")
    provider_identity = fit_embedding_provider.identity()
    provider_cache_identity = provider_identity.get("cache")
    if not isinstance(provider_cache_identity, Mapping):
        raise ValueError("fit embedding provider lacks its cache identity")
    _canonical_preflight_scope_binding(
        preflight=clustered_preflight,
        request=request,
        provider_cache_identity=provider_cache_identity,
    )
    cluster_state = load_canonical_clustered_preflight_scope_state(
        manifest_path=clustered_preflight_state_manifest,
        preflight=clustered_preflight,
        request=request,
    )
    root.mkdir(parents=True, exist_ok=False)
    computation = _fit_embedding_families(
        config=scientific_config,
        row_ids=request.physical_owner.fit_row_ids,
        texts=texts,
        provider=fit_embedding_provider,
        targets=fit_targets,
        cluster_scope=cluster_state.scope_record,
    )
    metadata, fit_state_sha256 = _persist_fit_state(
        root=root,
        request=request,
        config=scientific_config,
        fit_texts=texts,
        fit_targets=fit_targets,
        provider_identity=provider_identity,
        cluster_state=cluster_state,
        computation=computation,
    )
    events: list[dict[str, Any]] = [
        {
            "sequence": 1,
            "event": "fit_completed",
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        }
    ]
    seals: dict[str, dict[str, Any]] = {}
    seal_registrations: dict[str, dict[str, Any]] = {}
    for family in _FAMILIES:
        seal = _fit_seal(
            request=request,
            config=scientific_config,
            family=family,
            evidence_payload=computation.evidence_payloads[family],
            fit_state_sha256=fit_state_sha256,
        )
        path = root / _SEAL_FILENAMES[family]
        _write_new_json(path, seal)
        registration = _file_registration(path, relative_to=root)
        registration["content_sha256"] = seal["content_sha256"]
        seals[family] = seal
        seal_registrations[family] = registration
        events.append(
            {
                "sequence": len(events) + 1,
                "event": "fit_family_artifact_sealed",
                "family": family,
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
            }
        )
    logical_root = root / _LOGICAL_VIEW_DIRECTORY
    logical_root.mkdir(parents=True, exist_ok=False)
    logical_registrations: list[dict[str, Any]] = []
    reference_members = request.logical_members[1:]
    for member in reference_members:
        for family in _FAMILIES:
            view = _reference_only_view(
                request=request,
                member=member,
                family=family,
                seal=seals[family],
                seal_registration=seal_registrations[family],
            )
            path = logical_root / _logical_view_filename(member.scope_id, family)
            _write_new_json(path, view)
            logical_registrations.append(
                {
                    "logical_scope_id": member.scope_id,
                    "family": family,
                    **_file_registration(path, relative_to=root),
                    "content_sha256": view["content_sha256"],
                }
            )
            events.append(
                {
                    "sequence": len(events) + 1,
                    "event": "fit_only_logical_view_published",
                    "logical_scope_id": member.scope_id,
                    "logical_purpose": member.scope_kind,
                    "family": family,
                    "registered_heldout_text_accessed": False,
                    "registered_heldout_labels_accessed": False,
                }
            )
    exact_metadata: dict[str, Any] | None = None
    exact_batch: ExactHeldoutEmbeddingBatch | None = None
    if exact_heldout_loader is None:
        # Fail only after the independently useful fit state and all fit-only
        # alias references have been durably sealed.
        raise RuntimeError(
            "primary held-out embedding transport is unavailable after fit seals"
        )
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "exact_heldout_text_and_embeddings_opened",
            "logical_scope_id": request.physical_owner.scope_id,
            "logical_purpose": request.physical_owner.scope_kind,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
        }
    )
    loaded = exact_heldout_loader(request.physical_owner.heldout_row_ids)
    if not isinstance(loaded, ExactHeldoutEmbeddingBatch):
        raise TypeError(
            "exact held-out loader must return ExactHeldoutEmbeddingBatch"
        )
    if loaded.row_ids != request.physical_owner.heldout_row_ids:
        raise ValueError("exact held-out loader changed canonical row order")
    exact_batch = loaded
    exact_arrays, transform_metadata = _exact_transform_values(
        batch=loaded,
        config=scientific_config,
        computation=computation,
        cluster_state=cluster_state,
    )
    exact_metadata, exact_registrations = _persist_exact_transform(
        root=root,
        request=request,
        batch=loaded,
        arrays=exact_arrays,
        transform_metadata=transform_metadata,
    )
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "exact_heldout_transform_completed",
            "logical_scope_id": request.physical_owner.scope_id,
            "logical_purpose": request.physical_owner.scope_kind,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
        }
    )
    for family in _FAMILIES:
        view = _exact_view(
            request=request,
            family=family,
            seal=seals[family],
            seal_registration=seal_registrations[family],
            exact_metadata=exact_metadata,
            array_registrations=exact_registrations,
        )
        path = logical_root / _logical_view_filename(
            request.physical_owner.scope_id,
            family,
        )
        _write_new_json(path, view)
        logical_registrations.append(
            {
                "logical_scope_id": request.physical_owner.scope_id,
                "family": family,
                **_file_registration(path, relative_to=root),
                "content_sha256": view["content_sha256"],
            }
        )
        events.append(
            {
                "sequence": len(events) + 1,
                "event": "exact_logical_view_published",
                "logical_scope_id": request.physical_owner.scope_id,
                "logical_purpose": request.physical_owner.scope_kind,
                "family": family,
                "registered_heldout_text_accessed": True,
                "registered_heldout_labels_accessed": False,
            }
        )
    member_order = {
        member.scope_id: index
        for index, member in enumerate(request.logical_members)
    }
    family_order = {family: index for index, family in enumerate(_FAMILIES)}
    logical_registrations.sort(
        key=lambda row: (
            member_order[row["logical_scope_id"]],
            family_order[row["family"]],
        )
    )
    terminal_body = {
        "schema_version": ROLE_NEUTRAL_EMBEDDING_EXECUTION_SCHEMA,
        "status": "complete",
        "group_request": request.as_dict(),
        "families": list(_FAMILIES),
        "scientific_configuration_identity_sha256": (
            scientific_config.content_sha256
        ),
        "producer_identity_sha256": _producer_identity(),
        "fit_state_content_sha256": metadata["content_sha256"],
        "fit_state_artifact_sha256": fit_state_sha256,
        "canonical_clustered_preflight_state_content_sha256": (
            cluster_state.content_sha256
        ),
        "cluster_refit_performed": False,
        "fit_only_family_seals": seal_registrations,
        "logical_views": logical_registrations,
        "exact_transform_content_sha256": (
            None if exact_metadata is None else exact_metadata["content_sha256"]
        ),
        "event_order": events,
        "fit_completed_before_registered_heldout_text_access": True,
        "all_three_families_sealed_before_registered_heldout_text_access": True,
        "alias_views_published_before_registered_heldout_text_access": True,
        "only_physical_owner_transformed_heldout": True,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
        "semantic_term_truncation_applied": False,
        "pickle_joblib_npz_or_compression_used": False,
        "payload_inventory": _payload_inventory(root),
    }
    terminal = {**terminal_body, "content_sha256": _sha256_json(terminal_body)}
    _write_new_json(root / _TERMINAL_FILE, terminal)
    return validate_role_neutral_embedding_group_execution(
        root=root,
        request=request,
        clustered_preflight=clustered_preflight,
        clustered_preflight_state_manifest=clustered_preflight_state_manifest,
        expected_scientific_config=scientific_config,
        expected_fit_texts=texts,
        expected_fit_targets=fit_targets,
        expected_exact_batch=exact_batch,
    )


def _config_from_dict(value: Mapping[str, Any]) -> RoleNeutralEmbeddingScientificConfig:
    if not isinstance(value, Mapping):
        raise TypeError("persisted embedding scientific configuration must be a mapping")
    raw = dict(value)
    required = {
        "schema_version",
        "contrasts",
        "normalize_patient_embeddings",
        "patient_embedding_pooling",
        "numeric_compute_dtype",
        "vector_norm_order",
        "direction_norm_epsilon",
        "pseudo_target_quantile",
        "pseudo_target_weighted",
        "quantile_method",
        "minimum_contrast_side_rows",
        "lstsq_rcond",
        "lstsq_solution_policy",
        "semantic_input",
        "semantic_encoding",
        "semantic_decode_error",
        "semantic_preprocessor",
        "semantic_tokenizer",
        "semantic_analyzer",
        "semantic_ngram_min",
        "semantic_ngram_max",
        "semantic_token_pattern",
        "semantic_lowercase",
        "semantic_strip_accents",
        "semantic_min_df",
        "semantic_max_df",
        "semantic_sublinear_tf",
        "semantic_norm",
        "semantic_use_idf",
        "semantic_smooth_idf",
        "semantic_binary",
        "semantic_dtype",
        "semantic_stop_words",
        "semantic_vocabulary",
        "semantic_max_features",
        "semantic_member_batch_size",
        "maximum_source_chunks_per_row",
        "maximum_retrieval_chunks_per_side",
        "maximum_semantic_terms",
        "overflow_policy",
        "source_chunk_policy",
        "semantic_term_policy",
        "text_truncation_allowed",
    }
    if set(raw) != required or raw.get("schema_version") != ROLE_NEUTRAL_EMBEDDING_CONFIG_SCHEMA:
        raise ValueError("persisted embedding scientific configuration has another schema")
    contrasts_raw = raw["contrasts"]
    if not isinstance(contrasts_raw, list):
        raise ValueError("persisted embedding contrast configuration is malformed")
    config = RoleNeutralEmbeddingScientificConfig(
        contrasts=tuple(
            EmbeddingContrastSpec(
                name=row["name"],
                contrast_family=row["contrast_family"],
                target_name=row["target_name"],
                sample_weight_target_name=row[
                    "sample_weight_target_name"
                ],
                split_rule=row["split_rule"],
            )
            for row in contrasts_raw
            if isinstance(row, Mapping)
        ),
        normalize_patient_embeddings=raw["normalize_patient_embeddings"],
        patient_embedding_pooling=raw["patient_embedding_pooling"],
        numeric_compute_dtype=raw["numeric_compute_dtype"],
        vector_norm_order=raw["vector_norm_order"],
        direction_norm_epsilon=raw["direction_norm_epsilon"],
        pseudo_target_quantile=raw["pseudo_target_quantile"],
        pseudo_target_weighted=raw["pseudo_target_weighted"],
        quantile_method=raw["quantile_method"],
        minimum_contrast_side_rows=raw["minimum_contrast_side_rows"],
        lstsq_rcond=raw["lstsq_rcond"],
        lstsq_solution_policy=raw["lstsq_solution_policy"],
        semantic_input=raw["semantic_input"],
        semantic_encoding=raw["semantic_encoding"],
        semantic_decode_error=raw["semantic_decode_error"],
        semantic_preprocessor=raw["semantic_preprocessor"],
        semantic_tokenizer=raw["semantic_tokenizer"],
        semantic_analyzer=raw["semantic_analyzer"],
        semantic_ngram_min=raw["semantic_ngram_min"],
        semantic_ngram_max=raw["semantic_ngram_max"],
        semantic_token_pattern=raw["semantic_token_pattern"],
        semantic_lowercase=raw["semantic_lowercase"],
        semantic_strip_accents=raw["semantic_strip_accents"],
        semantic_min_df=raw["semantic_min_df"],
        semantic_max_df=raw["semantic_max_df"],
        semantic_sublinear_tf=raw["semantic_sublinear_tf"],
        semantic_norm=raw["semantic_norm"],
        semantic_use_idf=raw["semantic_use_idf"],
        semantic_smooth_idf=raw["semantic_smooth_idf"],
        semantic_binary=raw["semantic_binary"],
        semantic_dtype=raw["semantic_dtype"],
        semantic_stop_words=(
            None
            if raw["semantic_stop_words"] is None
            else (
                raw["semantic_stop_words"]
                if isinstance(raw["semantic_stop_words"], str)
                else tuple(raw["semantic_stop_words"])
            )
        ),
        semantic_vocabulary=raw["semantic_vocabulary"],
        semantic_max_features=raw["semantic_max_features"],
        semantic_member_batch_size=raw["semantic_member_batch_size"],
        maximum_source_chunks_per_row=raw["maximum_source_chunks_per_row"],
        maximum_retrieval_chunks_per_side=raw[
            "maximum_retrieval_chunks_per_side"
        ],
        maximum_semantic_terms=raw["maximum_semantic_terms"],
        overflow_policy=raw["overflow_policy"],
    )
    if config.as_dict() != raw:
        raise ValueError("persisted embedding scientific configuration changed")
    return config


def _validate_file_registration(
    *,
    root: Path,
    registration: Mapping[str, Any],
    expected_relative_path: str,
    label: str,
) -> Path:
    fields = set(registration) if isinstance(registration, Mapping) else set()
    if (
        not isinstance(registration, Mapping)
        or fields
        not in (
            {"relative_path", "sha256", "size_bytes"},
            {"relative_path", "sha256", "size_bytes", "content_sha256"},
        )
        or registration.get("relative_path") != expected_relative_path
    ):
        raise ValueError(f"{label} registration is invalid")
    path = root / expected_relative_path
    digest, size = _sha256_file(path)
    if digest != registration["sha256"] or size != registration["size_bytes"]:
        raise ValueError(f"{label} registration differs from its bytes")
    return path


def _reopen_fit_state(
    *,
    root: Path,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
    cluster_state: AuthenticatedClusteredPreflightScopeState,
    expected_config: RoleNeutralEmbeddingScientificConfig | None,
    expected_fit_texts: Sequence[str] | None,
    expected_fit_targets: Mapping[str, Sequence[float]] | None,
) -> tuple[
    dict[str, Any],
    RoleNeutralEmbeddingScientificConfig,
    _FitComputation,
    str,
]:
    fit_root = root / _FIT_STATE_DIRECTORY
    metadata = _read_json(
        fit_root / _FIT_METADATA,
        label="embedding fit metadata",
    )
    body = {
        key: copy.deepcopy(value)
        for key, value in metadata.items()
        if key != "content_sha256"
    }
    if (
        metadata.get("schema_version") != ROLE_NEUTRAL_EMBEDDING_FIT_STATE_SCHEMA
        or metadata.get("content_sha256") != _sha256_json(body)
        or metadata.get("group_request_content_sha256") != request.content_sha256
        or metadata.get("plan_scientific_content_sha256")
        != request.plan_scientific_content_sha256
        or metadata.get("physical_owner_scope_id")
        != request.physical_owner.scope_id
        or metadata.get("physical_owner_scope_sha256")
        != request.physical_owner.as_dict()["scope_sha256"]
        or metadata.get("fit_row_ids") != list(request.physical_owner.fit_row_ids)
        or metadata.get("fit_row_order_fingerprint")
        != _row_order_fingerprint(request.physical_owner.fit_row_ids)
        or metadata.get("canonical_group_seed")
        != request.physical_owner.scope_seed
        or metadata.get("producer_identity_sha256") != _producer_identity()
        or metadata.get("canonical_clustered_preflight_state_content_sha256")
        != cluster_state.content_sha256
        or metadata.get("canonical_clustered_preflight_binding")
        != cluster_state.manifest["preflight_binding"]
        or metadata.get("cluster_refit_performed") is not False
        or metadata.get("array_layout")
        != "one_npy_per_array_mmap_safe_v1"
        or metadata.get("all_source_chunks_accounted_once") is not True
        or metadata.get("all_configured_semantic_terms_accounted_once") is not True
        or metadata.get("registered_heldout_text_accessed") is not False
        or metadata.get("registered_heldout_labels_accessed") is not False
        or metadata.get("oracle_fields_accessed") is not False
        or metadata.get("text_truncation_applied") is not False
        or metadata.get("semantic_term_truncation_applied") is not False
        or metadata.get("pickle_joblib_npz_or_compression_used") is not False
    ):
        raise ValueError("embedding fit metadata is invalid")
    config = _config_from_dict(metadata["scientific_configuration"])
    if (
        metadata.get("configuration_identity_sha256") != config.content_sha256
        or (
            expected_config is not None
            and expected_config.as_dict() != config.as_dict()
        )
    ):
        raise ValueError("embedding scientific configuration identity changed")
    if expected_fit_texts is not None and metadata.get("fit_text_sha256") != (
        _ordered_text_sha256(
            request.physical_owner.fit_row_ids,
            tuple(expected_fit_texts),
        )
    ):
        raise ValueError("embedding fit text changed")
    chunks_path = _validate_file_registration(
        root=fit_root,
        registration=metadata["source_chunks"],
        expected_relative_path=_FIT_CHUNKS,
        label="embedding source chunks",
    )
    chunks = _read_json(chunks_path, label="embedding source chunks")
    chunks_body = {
        key: copy.deepcopy(value)
        for key, value in chunks.items()
        if key != "content_sha256"
    }
    rows = chunks.get("rows")
    if (
        chunks.get("schema_version")
        != "production_role_neutral_embedding_source_chunks_v1"
        or chunks.get("content_sha256") != _sha256_json(chunks_body)
        or chunks.get("content_sha256")
        != metadata["source_chunks_content_sha256"]
        or chunks.get("row_order") != list(request.physical_owner.fit_row_ids)
        or not isinstance(rows, list)
        or len(rows) != request.physical_owner.fit_row_count
        or chunks.get("all_uncapped_source_chunks_accounted_once") is not True
        or chunks.get("text_truncation_applied") is not False
    ):
        raise ValueError("embedding source-chunk coverage proof is invalid")
    chunks_by_row: list[tuple[str, ...]] = []
    for expected_row, row in zip(
        request.physical_owner.fit_row_ids,
        rows,
        strict=True,
    ):
        if (
            not isinstance(row, Mapping)
            or row.get("row_id") != expected_row
            or not isinstance(row.get("chunks"), list)
            or not all(isinstance(value, str) for value in row["chunks"])
            or row.get("chunk_count") != len(row["chunks"])
            or not row["chunks"]
        ):
            raise ValueError("embedding source chunks are missing or reordered")
        chunks_by_row.append(tuple(row["chunks"]))
    flat_texts = tuple(value for row in chunks_by_row for value in row)
    if (
        chunks.get("flat_chunk_count") != len(flat_texts)
        or metadata.get("fit_source_chunk_count") != len(flat_texts)
    ):
        raise ValueError("embedding flat source-chunk count changed")
    vocabulary_path = _validate_file_registration(
        root=fit_root,
        registration=metadata["semantic_vocabulary"],
        expected_relative_path=_FIT_VOCABULARY,
        label="embedding semantic vocabulary",
    )
    vocabulary_raw = _read_json(
        vocabulary_path,
        label="embedding semantic vocabulary",
    )
    vocabulary_body = {
        key: copy.deepcopy(value)
        for key, value in vocabulary_raw.items()
        if key != "content_sha256"
    }
    vocabulary = tuple(map(str, vocabulary_raw.get("terms") or ()))
    if (
        vocabulary_raw.get("schema_version")
        != "production_role_neutral_embedding_semantic_vocabulary_v1"
        or vocabulary_raw.get("content_sha256") != _sha256_json(vocabulary_body)
        or vocabulary_raw.get("content_sha256")
        != metadata["semantic_vocabulary_content_sha256"]
        or vocabulary_raw.get("term_count") != len(vocabulary)
        or vocabulary_raw.get("feature_indices") != list(range(len(vocabulary)))
        or vocabulary_raw.get("all_configured_terms_accounted_once") is not True
        or vocabulary_raw.get("semantic_term_truncation_applied") is not False
        or not vocabulary
        or len(vocabulary) != len(set(vocabulary))
        or metadata.get("fit_semantic_term_count") != len(vocabulary)
    ):
        raise ValueError("embedding semantic vocabulary is incomplete or reordered")
    inventory = metadata.get("array_inventory")
    order = metadata.get("array_order")
    if (
        not isinstance(inventory, Mapping)
        or not isinstance(order, list)
        or order != sorted(inventory)
    ):
        raise ValueError("embedding fit array inventory is invalid or reordered")
    expected_array_names = {
        "fit_chunk_embeddings",
        "fit_chunk_row_positions",
        "fit_chunk_positions",
        "fit_patient_embeddings",
        "fit_target_matrix",
        "fit_group_memberships",
        "whole_direction_matrix",
        "fit_patient_direction_scores",
        "fit_chunk_direction_scores",
        "semantic_idf",
        "semantic_signed_scores",
    }
    if set(inventory) != expected_array_names:
        raise ValueError("embedding fit array inventory is incomplete")
    arrays = {
        key: np.asarray(
            _read_registered_array(
                root=fit_root,
                registration=inventory[key],
                label=f"embedding fit array {key}",
            )
        )
        for key in order
    }
    chunk_embeddings = np.asarray(arrays["fit_chunk_embeddings"], dtype=np.float64)
    row_positions = np.asarray(arrays["fit_chunk_row_positions"], dtype=np.int64)
    chunk_positions = np.asarray(arrays["fit_chunk_positions"], dtype=np.int64)
    if (
        chunk_embeddings.ndim != 2
        or chunk_embeddings.shape[0] != len(flat_texts)
        or row_positions.shape != (len(flat_texts),)
        or chunk_positions.shape != (len(flat_texts),)
    ):
        raise ValueError("embedding fit chunks differ from their arrays")
    matrices: list[np.ndarray] = []
    cursor = 0
    expected_rows: list[int] = []
    expected_chunks: list[int] = []
    for row_index, row_chunks in enumerate(chunks_by_row):
        stop = cursor + len(row_chunks)
        matrices.append(chunk_embeddings[cursor:stop])
        expected_rows.extend([row_index] * len(row_chunks))
        expected_chunks.extend(range(len(row_chunks)))
        cursor = stop
    if (
        row_positions.tolist() != expected_rows
        or chunk_positions.tolist() != expected_chunks
    ):
        raise ValueError("embedding chunk coordinates are missing or reordered")
    patient_embeddings = _patient_means(
        matrices,
        normalize=bool(config.normalize_patient_embeddings),
        epsilon=float(config.direction_norm_epsilon),
        pooling=config.patient_embedding_pooling,
        norm_order=config.vector_norm_order,
    )
    target_matrix = np.asarray(arrays["fit_target_matrix"], dtype=np.float64)
    if expected_fit_targets is not None:
        expected_target_matrix = _target_matrix(
            targets=expected_fit_targets,
            config=config,
            row_count=request.physical_owner.fit_row_count,
        )
        expected_target_hashes = {
            name: _float_hex_sha256(
                np.asarray(expected_fit_targets[name], dtype=np.float64)
            )
            for name in sorted(expected_fit_targets)
        }
        if (
            not np.array_equal(target_matrix, expected_target_matrix)
            or metadata.get("fit_target_sha256") != expected_target_hashes
        ):
            raise ValueError("embedding fit targets changed")
    if target_matrix.shape != (
        request.physical_owner.fit_row_count,
        len(_target_order(config)),
    ):
        raise ValueError("embedding fit target matrix shape changed")
    groups, coefficient_matrix, direction_matrix = _contrast_geometry(
        target_matrix=target_matrix,
        patient_embeddings=patient_embeddings,
        config=config,
    )
    patient_scores = patient_embeddings @ direction_matrix.T
    chunk_scores = chunk_embeddings @ direction_matrix.T
    vectorizer = _semantic_vectorizer(config)
    lexical_matrix = vectorizer.fit_transform(flat_texts)
    replay_vocabulary = tuple(map(str, vectorizer.get_feature_names_out()))
    replay_idf = (
        np.asarray(vectorizer.idf_, dtype=np.float64)
        if config.semantic_use_idf
        else np.ones(len(replay_vocabulary), dtype=np.float64)
    )
    chunks_per_row = np.asarray(
        [len(value) for value in chunks_by_row],
        dtype=np.float64,
    )
    signed_rows: list[np.ndarray] = []
    for index in range(len(config.contrasts)):
        chunk_coefficients = (
            coefficient_matrix[row_positions, index]
            / chunks_per_row[row_positions]
        )
        signed_rows.append(
            np.asarray(
                chunk_coefficients @ lexical_matrix
            ).reshape(-1)
        )
    signed = np.stack(signed_rows, axis=0)
    expected_arrays = {
        "fit_patient_embeddings": patient_embeddings,
        "fit_group_memberships": groups,
        "whole_direction_matrix": direction_matrix,
        "fit_patient_direction_scores": patient_scores,
        "fit_chunk_direction_scores": chunk_scores,
        "semantic_idf": replay_idf,
        "semantic_signed_scores": signed,
    }
    if replay_vocabulary != vocabulary or any(
        not np.allclose(
            np.asarray(arrays[key]),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )
        for key, expected in expected_arrays.items()
    ):
        raise ValueError("embedding fit replay differs from persisted scientific state")
    payloads = _embedding_evidence_payloads(
        config=config,
        vocabulary=vocabulary,
        semantic_signed_scores=signed,
        source_chunk_count=len(flat_texts),
        cluster_scope=cluster_state.scope_record,
    )
    if metadata.get("evidence_payload_sha256") != {
        family: _sha256_json(payloads[family]) for family in _FAMILIES
    }:
        raise ValueError("embedding evidence payload identities changed")
    computation = _FitComputation(
        chunk_texts_by_row=tuple(chunks_by_row),
        flat_chunk_texts=flat_texts,
        chunk_embeddings=chunk_embeddings,
        chunk_row_positions=row_positions,
        chunk_positions=chunk_positions,
        patient_embeddings=patient_embeddings,
        target_matrix=target_matrix,
        group_memberships=groups,
        direction_matrix=direction_matrix,
        patient_scores=patient_scores,
        chunk_scores=chunk_scores,
        vocabulary=vocabulary,
        semantic_idf=replay_idf,
        semantic_signed_scores=signed,
        vectorizer=vectorizer,
        evidence_payloads=payloads,
    )
    return metadata, config, computation, _tree_sha256(fit_root)


def _reopen_exact_state(
    *,
    root: Path,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    exact_root = root / _EXACT_DIRECTORY
    metadata = _read_json(
        exact_root / "metadata.json",
        label="embedding exact-transform metadata",
    )
    body = {
        key: copy.deepcopy(value)
        for key, value in metadata.items()
        if key != "content_sha256"
    }
    inventory = metadata.get("array_inventory")
    order = metadata.get("array_order")
    transform = metadata.get("transform_metadata")
    if (
        metadata.get("schema_version")
        != "production_role_neutral_embedding_exact_transform_v1"
        or metadata.get("content_sha256") != _sha256_json(body)
        or metadata.get("group_request_content_sha256") != request.content_sha256
        or metadata.get("plan_scientific_content_sha256")
        != request.plan_scientific_content_sha256
        or metadata.get("logical_scope_id") != request.physical_owner.scope_id
        or metadata.get("logical_scope_sha256")
        != request.physical_owner.as_dict()["scope_sha256"]
        or metadata.get("heldout_row_ids")
        != list(request.physical_owner.heldout_row_ids)
        or metadata.get("heldout_row_order_fingerprint")
        != _row_order_fingerprint(request.physical_owner.heldout_row_ids)
        or not isinstance(transform, Mapping)
        or not isinstance(inventory, Mapping)
        or not isinstance(order, list)
        or order != sorted(inventory)
        or metadata.get("array_layout")
        != "one_npy_per_array_mmap_safe_v1"
        or metadata.get("fit_seals_preexisted") is not True
        or metadata.get("registered_heldout_text_accessed") is not True
        or metadata.get("registered_heldout_labels_accessed") is not False
        or metadata.get("oracle_fields_accessed") is not False
        or metadata.get("text_truncation_applied") is not False
        or metadata.get("semantic_term_truncation_applied") is not False
        or metadata.get("pickle_joblib_npz_or_compression_used") is not False
    ):
        raise ValueError("embedding exact-transform metadata is invalid")
    base = {
        "heldout_patient_embeddings",
        "heldout_chunk_embeddings",
        "heldout_chunk_row_positions",
        "heldout_chunk_positions",
        "heldout_whole_patient_scores",
        "heldout_whole_chunk_scores",
        "heldout_cluster_distances",
        "heldout_cluster_assignments",
        "heldout_lexical_csr_data",
        "heldout_lexical_csr_indices",
        "heldout_lexical_csr_indptr",
    }
    projections = transform.get("cluster_svd_projections")
    if not isinstance(projections, list) or not projections:
        raise ValueError("embedding exact transform lacks cluster SVD projections")
    projection_keys = {
        str(row.get("array_key"))
        for row in projections
        if isinstance(row, Mapping)
    }
    if len(projection_keys) != len(projections) or set(inventory) != base | projection_keys:
        raise ValueError("embedding exact array inventory is incomplete")
    arrays = {
        key: np.asarray(
            _read_registered_array(
                root=root,
                registration=inventory[key],
                label=f"embedding exact array {key}",
            )
        )
        for key in order
    }
    row_count = request.physical_owner.heldout_row_count
    chunk_count = int(transform.get("heldout_flat_chunk_count", -1))
    vocabulary_shape = transform.get("lexical_csr_shape")
    if (
        row_count < 1
        or chunk_count < 1
        or arrays["heldout_patient_embeddings"].shape[0] != row_count
        or arrays["heldout_chunk_embeddings"].shape[0] != chunk_count
        or arrays["heldout_chunk_row_positions"].shape != (chunk_count,)
        or arrays["heldout_chunk_positions"].shape != (chunk_count,)
        or not isinstance(vocabulary_shape, list)
        or len(vocabulary_shape) != 2
        or vocabulary_shape[0] != row_count
    ):
        raise ValueError("embedding exact arrays changed row/chunk alignment")
    data = arrays["heldout_lexical_csr_data"]
    indices = arrays["heldout_lexical_csr_indices"]
    indptr = arrays["heldout_lexical_csr_indptr"]
    if (
        data.ndim != 1
        or indices.shape != data.shape
        or indptr.shape != (row_count + 1,)
        or int(indptr[0]) != 0
        or int(indptr[-1]) != len(data)
        or np.any(np.diff(indptr) < 0)
        or np.any(indices < 0)
        or np.any(indices >= int(vocabulary_shape[1]))
        or transform.get("lexical_csr_nnz") != len(data)
        or transform.get("all_heldout_source_chunks_transformed_once") is not True
        or transform.get("heldout_labels_accessed") is not False
    ):
        raise ValueError("embedding exact lexical CSR state is invalid")
    return metadata, arrays


def _compare_exact_replay(
    *,
    root: Path,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
    batch: ExactHeldoutEmbeddingBatch,
    config: RoleNeutralEmbeddingScientificConfig,
    computation: _FitComputation,
    cluster_state: AuthenticatedClusteredPreflightScopeState,
) -> dict[str, np.ndarray]:
    metadata, stored = _reopen_exact_state(root=root, request=request)
    if (
        batch.row_ids != request.physical_owner.heldout_row_ids
        or metadata.get("heldout_text_sha256")
        != _ordered_text_sha256(batch.row_ids, batch.texts)
    ):
        raise ValueError("embedding exact replay received another held-out projection")
    replay, replay_metadata = _exact_transform_values(
        batch=batch,
        config=config,
        computation=computation,
        cluster_state=cluster_state,
    )
    if replay_metadata != metadata["transform_metadata"] or set(replay) != set(stored):
        raise ValueError("embedding exact replay metadata changed")
    for key in replay:
        expected = np.asarray(stored[key])
        actual = np.asarray(replay[key])
        if (
            expected.shape != actual.shape
            or expected.dtype.kind != actual.dtype.kind
            or (
                np.issubdtype(expected.dtype, np.floating)
                and not np.allclose(expected, actual, rtol=1e-12, atol=1e-12)
            )
            or (
                not np.issubdtype(expected.dtype, np.floating)
                and not np.array_equal(expected, actual)
            )
        ):
            raise ValueError(f"embedding exact replay differs for array {key}")
    return replay


def _validate_registered_json(
    *,
    root: Path,
    registration: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    required = {
        "logical_scope_id",
        "family",
        "relative_path",
        "sha256",
        "size_bytes",
        "content_sha256",
    }
    if not isinstance(registration, Mapping) or set(registration) != required:
        raise ValueError(f"{label} registration is invalid")
    relative = Path(str(registration["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} registration escapes its artifact")
    path = root / relative
    digest, size = _sha256_file(path)
    value = _read_json(path, label=label)
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    if (
        digest != registration["sha256"]
        or size != registration["size_bytes"]
        or value.get("content_sha256") != _sha256_json(body)
        or value.get("content_sha256") != registration["content_sha256"]
    ):
        raise ValueError(f"{label} differs from its registration")
    return value


def validate_role_neutral_embedding_group_execution(
    *,
    root: Path | str,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
    clustered_preflight: _ClusterPreflightArtifact,
    clustered_preflight_state_manifest: Path | str,
    expected_scientific_config: RoleNeutralEmbeddingScientificConfig | None = None,
    expected_fit_texts: Sequence[str] | None = None,
    expected_fit_targets: Mapping[str, Sequence[float]] | None = None,
    expected_exact_batch: ExactHeldoutEmbeddingBatch | None = None,
) -> dict[str, Any]:
    """Fresh path-only validator for one complete physical-group artifact."""

    if not isinstance(request, RoleNeutralEmbeddingPhysicalGroupRequest):
        raise TypeError("embedding validator requires its typed request")
    request.as_dict()
    artifact_root = Path(root)
    if (
        not artifact_root.is_absolute()
        or artifact_root.is_symlink()
        or not artifact_root.is_dir()
        or artifact_root.resolve(strict=True) != artifact_root
    ):
        raise ValueError("embedding artifact root must be one canonical real directory")
    cluster_state = load_canonical_clustered_preflight_scope_state(
        manifest_path=clustered_preflight_state_manifest,
        preflight=clustered_preflight,
        request=request,
    )
    metadata, config, computation, fit_state_sha256 = _reopen_fit_state(
        root=artifact_root,
        request=request,
        cluster_state=cluster_state,
        expected_config=expected_scientific_config,
        expected_fit_texts=expected_fit_texts,
        expected_fit_targets=expected_fit_targets,
    )
    terminal = _read_json(
        artifact_root / _TERMINAL_FILE,
        label="embedding execution manifest",
    )
    terminal_body = {
        key: copy.deepcopy(value)
        for key, value in terminal.items()
        if key != "content_sha256"
    }
    seals_raw = terminal.get("fit_only_family_seals")
    views_raw = terminal.get("logical_views")
    if (
        terminal.get("schema_version") != ROLE_NEUTRAL_EMBEDDING_EXECUTION_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("content_sha256") != _sha256_json(terminal_body)
        or terminal.get("group_request") != request.as_dict()
        or terminal.get("families") != list(_FAMILIES)
        or terminal.get("scientific_configuration_identity_sha256")
        != config.content_sha256
        or terminal.get("producer_identity_sha256") != _producer_identity()
        or terminal.get("fit_state_content_sha256") != metadata["content_sha256"]
        or terminal.get("fit_state_artifact_sha256") != fit_state_sha256
        or terminal.get("canonical_clustered_preflight_state_content_sha256")
        != cluster_state.content_sha256
        or terminal.get("cluster_refit_performed") is not False
        or not isinstance(seals_raw, Mapping)
        or set(seals_raw) != set(_FAMILIES)
        or not isinstance(views_raw, list)
        or terminal.get("fit_completed_before_registered_heldout_text_access")
        is not True
        or terminal.get(
            "all_three_families_sealed_before_registered_heldout_text_access"
        )
        is not True
        or terminal.get(
            "alias_views_published_before_registered_heldout_text_access"
        )
        is not True
        or terminal.get("only_physical_owner_transformed_heldout") is not True
        or terminal.get("registered_heldout_labels_accessed") is not False
        or terminal.get("oracle_fields_accessed") is not False
        or terminal.get("text_truncation_applied") is not False
        or terminal.get("semantic_term_truncation_applied") is not False
        or terminal.get("pickle_joblib_npz_or_compression_used") is not False
        or terminal.get("payload_inventory") != _payload_inventory(artifact_root)
    ):
        raise ValueError("embedding execution manifest is invalid")
    expected_files = {
        _TERMINAL_FILE,
        *(_SEAL_FILENAMES[family] for family in _FAMILIES),
        f"{_FIT_STATE_DIRECTORY}/{_FIT_METADATA}",
        f"{_FIT_STATE_DIRECTORY}/{_FIT_CHUNKS}",
        f"{_FIT_STATE_DIRECTORY}/{_FIT_VOCABULARY}",
        *(
            f"{_FIT_STATE_DIRECTORY}/{registration['relative_path']}"
            for registration in metadata["array_inventory"].values()
        ),
        *(
            str(row["relative_path"])
            for row in views_raw
            if isinstance(row, Mapping)
        ),
    }
    expected_directories = {
        _FIT_STATE_DIRECTORY,
        f"{_FIT_STATE_DIRECTORY}/arrays",
        _LOGICAL_VIEW_DIRECTORY,
    }
    exact_preview = _read_json(
        artifact_root / _EXACT_DIRECTORY / "metadata.json",
        label="embedding exact-transform metadata inventory",
    )
    expected_files.add(f"{_EXACT_DIRECTORY}/metadata.json")
    expected_files.update(
        str(registration["relative_path"])
        for registration in (
            exact_preview.get("array_inventory") or {}
        ).values()
        if isinstance(registration, Mapping)
    )
    expected_directories.update(
        {
            _EXACT_DIRECTORY,
            f"{_EXACT_DIRECTORY}/arrays",
        }
    )
    actual_files = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    actual_directories = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError("embedding artifact tree contains missing or extra entries")
    seals: dict[str, dict[str, Any]] = {}
    for family in _FAMILIES:
        registration = seals_raw[family]
        expected_path = _SEAL_FILENAMES[family]
        path = _validate_file_registration(
            root=artifact_root,
            registration=registration,
            expected_relative_path=expected_path,
            label=f"{family} fit-only seal",
        )
        seal = _read_json(path, label=f"{family} fit-only seal")
        expected_seal = _fit_seal(
            request=request,
            config=config,
            family=family,
            evidence_payload=computation.evidence_payloads[family],
            fit_state_sha256=fit_state_sha256,
        )
        if seal != expected_seal:
            raise ValueError(f"{family} fit-only seal is invalid")
        if registration.get("content_sha256") != seal["content_sha256"]:
            raise ValueError(f"{family} fit-only seal content registration changed")
        seals[family] = seal
    expected_view_count = len(request.logical_members) * len(_FAMILIES)
    if len(views_raw) != expected_view_count:
        raise ValueError("embedding logical view coverage is incomplete")
    member_by_id = {member.scope_id: member for member in request.logical_members}
    expected_order = [
        (member.scope_id, family)
        for member in request.logical_members
        for family in _FAMILIES
    ]
    if [
        (row.get("logical_scope_id"), row.get("family"))
        for row in views_raw
        if isinstance(row, Mapping)
    ] != expected_order:
        raise ValueError("embedding logical views are missing or reordered")
    exact_metadata, exact_arrays = _reopen_exact_state(
        root=artifact_root,
        request=request,
    )
    if (
        terminal.get("exact_transform_content_sha256")
        != exact_metadata["content_sha256"]
    ):
        raise ValueError("embedding exact-transform identity changed")
    for registration in views_raw:
        view = _validate_registered_json(
            root=artifact_root,
            registration=registration,
            label="embedding logical view",
        )
        scope_id = str(registration["logical_scope_id"])
        family = str(registration["family"])
        member = member_by_id[scope_id]
        seal = seals[family]
        seal_registration = seals_raw[family]
        if member.scope_id == request.physical_owner.scope_id:
            expected_view = _exact_view(
                request=request,
                family=family,
                seal=seal,
                seal_registration=seal_registration,
                exact_metadata=exact_metadata,
                array_registrations=exact_metadata["array_inventory"],
            )
        else:
            expected_view = _reference_only_view(
                request=request,
                member=member,
                family=family,
                seal=seal,
                seal_registration=seal_registration,
            )
        if view != expected_view:
            raise ValueError("embedding logical view has an invalid fit/view binding")
    events = terminal.get("event_order")
    if (
        not isinstance(events, list)
        or [row.get("sequence") for row in events if isinstance(row, Mapping)]
        != list(range(1, len(events) + 1))
        or not events
        or events[0].get("event") != "fit_completed"
        or [row.get("family") for row in events[1:4]]
        != list(_FAMILIES)
        or any(
            row.get("registered_heldout_labels_accessed") is not False
            for row in events
            if isinstance(row, Mapping)
        )
    ):
        raise ValueError("embedding execution event order is invalid")
    open_indices = [
        index
        for index, row in enumerate(events)
        if row.get("event") == "exact_heldout_text_and_embeddings_opened"
    ]
    cumulative_indices = [
        index
        for index, row in enumerate(events)
        if row.get("event") == "fit_only_logical_view_published"
        and row.get("logical_purpose") == "cumulative_spent"
    ]
    if (
        len(open_indices) != 1
        or open_indices[0] < 4
        or any(index > open_indices[0] for index in cumulative_indices)
    ):
        raise ValueError(
            "embedding held-out text access preceded fit seals or alias views"
        )
    if expected_exact_batch is not None:
        _compare_exact_replay(
            root=artifact_root,
            request=request,
            batch=expected_exact_batch,
            config=config,
            computation=computation,
            cluster_state=cluster_state,
        )
    if tuple(artifact_root.rglob("*.pkl")) or tuple(
        artifact_root.rglob("*.pickle")
    ) or tuple(artifact_root.rglob("*.joblib")) or tuple(
        artifact_root.rglob("*.npz")
    ):
        raise ValueError("embedding artifact contains executable or compressed serialization")
    return terminal


def replay_role_neutral_embedding_exact_transform(
    *,
    root: Path | str,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
    clustered_preflight: _ClusterPreflightArtifact,
    clustered_preflight_state_manifest: Path | str,
    exact_heldout_batch: ExactHeldoutEmbeddingBatch,
) -> dict[str, Any]:
    """Reopen only authenticated JSON/NPY state and replay one exact transform."""

    artifact_root = Path(root)
    validate_role_neutral_embedding_group_execution(
        root=artifact_root,
        request=request,
        clustered_preflight=clustered_preflight,
        clustered_preflight_state_manifest=clustered_preflight_state_manifest,
    )
    cluster_state = load_canonical_clustered_preflight_scope_state(
        manifest_path=clustered_preflight_state_manifest,
        preflight=clustered_preflight,
        request=request,
    )
    _metadata, config, computation, _fit_sha = _reopen_fit_state(
        root=artifact_root,
        request=request,
        cluster_state=cluster_state,
        expected_config=None,
        expected_fit_texts=None,
        expected_fit_targets=None,
    )
    arrays = _compare_exact_replay(
        root=artifact_root,
        request=request,
        batch=exact_heldout_batch,
        config=config,
        computation=computation,
        cluster_state=cluster_state,
    )
    return {
        "schema_version": "production_role_neutral_embedding_exact_replay_v1",
        "logical_scope_id": request.physical_owner.scope_id,
        "array_order": sorted(arrays),
        "arrays": arrays,
        "state_source": "authenticated_canonical_json_and_individual_npy_only",
        "live_model_objects_available": False,
        "cluster_refit_performed": False,
        "pickle_joblib_or_npz_loaded": False,
        "registered_heldout_labels_accessed": False,
    }


__all__ = [
    "AuthenticatedClusteredPreflightStateBundle",
    "AuthenticatedClusteredPreflightScopeState",
    "EmbeddingContrastSpec",
    "ExactHeldoutEmbeddingBatch",
    "ROLE_NEUTRAL_EMBEDDING_CLUSTER_STATE_SCHEMA",
    "ROLE_NEUTRAL_EMBEDDING_CLUSTER_STATE_BUNDLE_SCHEMA",
    "ROLE_NEUTRAL_EMBEDDING_CONFIG_SCHEMA",
    "ROLE_NEUTRAL_EMBEDDING_EXECUTION_SCHEMA",
    "ROLE_NEUTRAL_EMBEDDING_FIT_STATE_SCHEMA",
    "ROLE_NEUTRAL_EMBEDDING_LOGICAL_VIEW_SCHEMA",
    "ROLE_NEUTRAL_EMBEDDING_REQUEST_SCHEMA",
    "RoleNeutralEmbeddingPhysicalGroupRequest",
    "RoleNeutralEmbeddingScientificConfig",
    "execute_role_neutral_embedding_physical_group",
    "load_canonical_clustered_preflight_state_bundle",
    "load_canonical_clustered_preflight_scope_state",
    "replay_role_neutral_embedding_exact_transform",
    "seal_canonical_clustered_preflight_state_bundle",
    "seal_canonical_clustered_preflight_scope_state",
    "validate_role_neutral_embedding_group_execution",
]
