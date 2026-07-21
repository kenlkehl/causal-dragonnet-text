"""Authenticated, non-grounding inventory of direct Stage-1 numerical signals.

The feature-discovery model may learn patient-feature names only from lexical or
semantic evidence.  Row-aligned upstream values travel through a separate
channel.  This module records every column in that channel, its exact matrix
position, architecture, observable axis, fit lineage, and—critically—whether
the upstream coordinate survived or was reduced to a row-wise permutation-
invariant statistic.

The manifest deliberately links semantic evidence only at architecture level.
There is no coordinate-to-atom linkage and ``concept_grounding_allowed`` is
always false.  A selector-facing projection exposes counts and stability modes,
never row values, filenames, column names, or lexical pairings.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
    BOW_NUISANCE,
    BOW_R_LOSS,
    DIRECT_UPSTREAM_NUMERICAL_CHANNEL,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HETEROGENEITY_AXIS,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    OUTCOME_AXIS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPICS,
    TREATMENT_AXIS,
)
from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .context_fit_upstream_gate_provider import (
    CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION,
)
from .coordinate_preserving_context_fit_upstream_backend import (
    COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID,
)
from .final_context_fit_upstream_bank import FINAL_CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA
from .stable_context_fit_upstream_backend import STABLE_CONTEXT_FIT_UPSTREAM_BACKEND_ID

DIRECT_NUMERICAL_MANIFEST_SCHEMA_VERSION = "direct_upstream_numerical_manifest_v1"
DIRECT_NUMERICAL_SELECTOR_VIEW_VERSION = "direct_upstream_numerical_selector_view_v1"

CALIBRATED_SOURCES_BLOCK = "calibrated_sources"
RAW_FEATURES_BLOCK = "raw_features"
MATRIX_BLOCKS = (CALIBRATED_SOURCES_BLOCK, RAW_FEATURES_BLOCK)

CONTEXT_OOF_SCOPE = "outer_train_context_oof"
PREDICTION_SCOPE = "label_free_prediction_rows"
ROW_SCOPES = (CONTEXT_OOF_SCOPE, PREDICTION_SCOPE)

EXACT_PRECOMMITTED_ALIGNMENT = "exact_precommitted_source_coordinate"
EXACT_NAMED_RAW_ALIGNMENT = "exact_precommitted_named_raw_coordinate"
CONDITIONAL_PRESENCE_ALIGNMENT = "precommitted_conditional_coordinate_presence_indicator"
PERMUTATION_SUMMARY_ALIGNMENT = "permutation_invariant_row_summary"
PREAGGREGATED_PERMUTATION_SUMMARY_ALIGNMENT = "permutation_invariant_preaggregated_row_summary"

NESTED_CALIBRATED_STATUS = "nested_calibrated_tau_prediction"
UNCALIBRATED_BASIS_STATUS = "uncalibrated_role_aware_basis"
EFFECT_REGRESSION_COVARIATE_ROLE = "effect_regression_covariate"

SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON = (
    "concept_only_semantic_retrieval_shares_parent_embedding_projection_"
    "and_has_no_independent_row_signal"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[a-z][a-z0-9_.:-]*\Z")
_FORBIDDEN = re.compile(r"(?:oracle|ground_truth|true_ite|true_cate|true_effect)", re.I)
_SIGNED_ORDER = re.compile(r"signed_order_([0-9]+)\Z")

_RAW_STATISTIC_KINDS = frozenset(
    {
        "exact_named_coordinate",
        "presence",
        "signed_mean",
        "absolute_max",
        "signed_descending_order",
    }
)
_RAW_ALIGNMENT_MODES = frozenset(
    {
        EXACT_NAMED_RAW_ALIGNMENT,
        CONDITIONAL_PRESENCE_ALIGNMENT,
        PERMUTATION_SUMMARY_ALIGNMENT,
        PREAGGREGATED_PERMUTATION_SUMMARY_ALIGNMENT,
    }
)

_ROLE_AXIS = {
    PROPENSITY_NUISANCE_FEATURE_ROLE: TREATMENT_AXIS,
    OUTCOME_NUISANCE_FEATURE_ROLE: OUTCOME_AXIS,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE: HETEROGENEITY_AXIS,
}

# A direct source kind can be more granular than the concept-bearing Stage-1
# architecture.  The mapping is closed so an unexpected producer fails rather
# than being silently assigned to a convenient family.
_RAW_KIND_FAMILY = {
    "bow_nuisance": BOW_NUISANCE,
    "bow_r_loss": BOW_R_LOSS,
    "htr_nuisance": HTR_NEURAL,
    "htr_neural": HTR_NEURAL,
    "matched_pair_uplift": MATCHED_PAIR_UPLIFT,
    "embedding_whole_cohort": EMBEDDING_WHOLE_COHORT,
    "embedding_clustered": EMBEDDING_CLUSTERED,
    "tfidf_topics": TFIDF_TOPICS,
    "tfidf_topic_contrast": TFIDF_TOPICS,
    "tfidf_orphan_ngrams": TFIDF_ORPHAN_NGRAMS,
    "neural_query_treatment_moments": NEURAL_QUERY_MOMENTS,
    "neural_query_outcome_moments": NEURAL_QUERY_MOMENTS,
    "neural_query_effect_moments": NEURAL_QUERY_MOMENTS,
}
_CALIBRATED_KIND_FAMILY = {
    "nested_calibrated_bow_weighted_r": BOW_R_LOSS,
    "nested_calibrated_htr_weighted_r": HTR_NEURAL,
}
_RAW_KIND_PRODUCER = {
    "bow_nuisance": "pooled_configured_bow_nuisance_views",
    "bow_r_loss": "pooled_configured_bow_r_loss_views",
    "htr_nuisance": "htr_nuisance_predictions",
    "htr_neural": "htr_residual_effect_predictions",
    # Historical Stage 1 pools the BoW and HTR pair paths before the stable
    # reduction; representation origin cannot be recovered from that cache.
    "matched_pair_uplift": "pooled_bow_and_htr_matched_pair_uplift",
    "embedding_whole_cohort": "whole_cohort_embedding_contrasts",
    "embedding_clustered": "clustered_embedding_contrasts",
    "tfidf_topics": "tfidf_treatment_and_outcome_topics",
    "tfidf_topic_contrast": "tfidf_residual_effect_topic_contrasts",
    "tfidf_orphan_ngrams": "tfidf_residual_orphan_ngrams",
    "neural_query_treatment_moments": "neural_query_treatment_moments",
    "neural_query_outcome_moments": "neural_query_outcome_moments",
    "neural_query_effect_moments": "neural_query_effect_moments",
}

_GATE_MANIFEST_FIELDS = frozenset(
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
_FINAL_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "cache_key",
        "binding",
        "calibrated_sources",
        "raw_features",
        "matrix_files",
        "content_sha256",
    }
)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("direct numerical metadata must be finite JSON") from exc


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _safe_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if _FORBIDDEN.search(value):
        raise ValueError(f"{label} contains forbidden benchmark metadata")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    result = _safe_text(value, label=label)
    if _SAFE_ID.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase opaque identifier")
    return result


def _positive_or_zero(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _string_tuple(
    value: Sequence[Any], *, label: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(_safe_text(item, label=f"{label}[]") for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{label} cannot be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} cannot contain duplicates")
    return result


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_closed_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable JSON manifest: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError("manifest must be one JSON object")
    return value


def _matrix_shape(path: Path, *, expected_sha256: str) -> tuple[int, int]:
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise ValueError(f"matrix failed SHA-256 authentication: {path.name}")
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"matrix is not a safe NumPy array: {path.name}") from exc
    if values.ndim != 2 or values.dtype.kind not in "fiu":
        raise ValueError(f"matrix must be a two-dimensional numeric array: {path.name}")
    return int(values.shape[0]), int(values.shape[1])


@dataclass(frozen=True)
class AuthenticatedMatrixBinding:
    matrix_block: str
    row_scope: str
    filename: str
    sha256: str
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.matrix_block not in MATRIX_BLOCKS:
            raise ValueError("matrix_block is unsupported")
        if self.row_scope not in ROW_SCOPES:
            raise ValueError("row_scope is unsupported")
        filename = _safe_text(self.filename, label="matrix filename")
        if Path(filename).name != filename or not filename.endswith(".npy"):
            raise ValueError("matrix filename must be one safe NumPy basename")
        _require_sha256(self.sha256, label="matrix sha256")
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(_positive_or_zero(value, label="matrix shape") < 1 for value in self.shape)
        ):
            raise ValueError("matrix shape must contain two positive dimensions")

    @property
    def key(self) -> tuple[str, str]:
        return self.matrix_block, self.row_scope

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix_block": self.matrix_block,
            "row_scope": self.row_scope,
            "filename": self.filename,
            "sha256": self.sha256,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class AuthenticatedRawCoordinateSemantics:
    """Authenticated fixed-schema meaning for one emitted raw column."""

    coordinate_name: str
    source_kind: str
    consumer_role: str
    producer_subarchitecture: str
    statistic_kind: str
    statistic_rank: int | None
    statistic_width: int
    alignment_mode: str
    source_coordinate_identity_preserved: bool

    def __post_init__(self) -> None:
        _safe_text(self.coordinate_name, label="raw semantic coordinate_name")
        if self.source_kind not in _RAW_KIND_FAMILY:
            raise ValueError("raw semantic source_kind is unsupported")
        if self.consumer_role not in _ROLE_AXIS:
            raise ValueError("raw semantic consumer_role is unsupported")
        _safe_text(
            self.producer_subarchitecture,
            label="raw semantic producer_subarchitecture",
        )
        if self.statistic_kind not in _RAW_STATISTIC_KINDS:
            raise ValueError("raw semantic statistic_kind is unsupported")
        width = _positive_or_zero(self.statistic_width, label="raw semantic statistic_width")
        if width < 1:
            raise ValueError("raw semantic statistic_width must be positive")
        if self.statistic_kind == "signed_descending_order":
            if self.statistic_rank is None:
                raise ValueError("raw signed order semantic requires a rank")
            rank = _positive_or_zero(self.statistic_rank, label="raw semantic statistic_rank")
            if rank < 1 or rank > width:
                raise ValueError("raw semantic statistic_rank is outside its fixed width")
        elif self.statistic_rank is not None:
            raise ValueError("only raw signed order semantics may carry a rank")
        if self.alignment_mode not in _RAW_ALIGNMENT_MODES:
            raise ValueError("raw semantic alignment_mode is unsupported")
        if not isinstance(self.source_coordinate_identity_preserved, bool):
            raise TypeError("raw semantic identity-preserved flag must be boolean")
        if (
            self.alignment_mode == EXACT_NAMED_RAW_ALIGNMENT
        ) != self.source_coordinate_identity_preserved:
            raise ValueError(
                "only exact named raw coordinates may preserve source-coordinate identity"
            )
        if self.alignment_mode == CONDITIONAL_PRESENCE_ALIGNMENT and (
            self.statistic_kind != "presence"
        ):
            raise ValueError("conditional-presence alignment requires a presence statistic")


@dataclass(frozen=True)
class AuthenticatedNumericalBankSnapshot:
    """Authenticated metadata extracted from one gate or final bank."""

    source_manifest_path: Path
    source_manifest_sha256: str
    source_cache_schema: str
    source_cache_key: str
    producer_identity_sha256: str
    stable_output_schema_sha256: str
    shared_lineage_sha256: str
    lineage_scope: str
    matrices: tuple[AuthenticatedMatrixBinding, ...]
    calibrated_source_names: tuple[str, ...]
    calibrated_source_kinds: tuple[str, ...]
    raw_feature_names: tuple[str, ...]
    raw_feature_kinds: tuple[str, ...]
    raw_feature_roles: tuple[str, ...]
    raw_coordinate_semantics: tuple[AuthenticatedRawCoordinateSemantics, ...]

    def __post_init__(self) -> None:
        path = Path(self.source_manifest_path).resolve(strict=True)
        if not path.is_file() or path.name != "manifest.json":
            raise ValueError("source_manifest_path must name an existing manifest.json")
        if _sha256_file(path) != _require_sha256(
            self.source_manifest_sha256, label="source_manifest_sha256"
        ):
            raise ValueError("source manifest bytes changed")
        _safe_text(self.source_cache_schema, label="source_cache_schema")
        _require_sha256(self.source_cache_key, label="source_cache_key")
        _require_sha256(self.producer_identity_sha256, label="producer_identity_sha256")
        _require_sha256(self.stable_output_schema_sha256, label="stable_output_schema_sha256")
        _require_sha256(self.shared_lineage_sha256, label="shared_lineage_sha256")
        _safe_text(self.lineage_scope, label="lineage_scope")
        matrix_keys = tuple(item.key for item in self.matrices)
        expected_keys = tuple((block, scope) for block in MATRIX_BLOCKS for scope in ROW_SCOPES)
        if set(matrix_keys) != set(expected_keys) or len(matrix_keys) != len(set(matrix_keys)):
            raise ValueError("snapshot must authenticate both row scopes for both matrix blocks")
        if len(self.calibrated_source_names) != len(self.calibrated_source_kinds):
            raise ValueError("calibrated source metadata is not aligned")
        if not (
            len(self.raw_feature_names)
            == len(self.raw_feature_kinds)
            == len(self.raw_feature_roles)
            == len(self.raw_coordinate_semantics)
        ):
            raise ValueError("raw feature metadata is not aligned")
        for label, values in (
            ("calibrated_source_names", self.calibrated_source_names),
            ("calibrated_source_kinds", self.calibrated_source_kinds),
            ("raw_feature_names", self.raw_feature_names),
            ("raw_feature_kinds", self.raw_feature_kinds),
            ("raw_feature_roles", self.raw_feature_roles),
        ):
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{label} contains an invalid string")
        for block, count in (
            (CALIBRATED_SOURCES_BLOCK, len(self.calibrated_source_names)),
            (RAW_FEATURES_BLOCK, len(self.raw_feature_names)),
        ):
            widths = {item.shape[1] for item in self.matrices if item.matrix_block == block}
            if widths != {count}:
                raise ValueError(f"{block} matrices do not match their metadata width")
        for index, semantic in enumerate(self.raw_coordinate_semantics):
            if not isinstance(semantic, AuthenticatedRawCoordinateSemantics):
                raise TypeError(
                    "raw_coordinate_semantics must contain " "AuthenticatedRawCoordinateSemantics"
                )
            actual = (
                self.raw_feature_names[index],
                self.raw_feature_kinds[index],
                self.raw_feature_roles[index],
            )
            expected = (
                semantic.coordinate_name,
                semantic.source_kind,
                semantic.consumer_role,
            )
            if actual != expected:
                raise ValueError("raw coordinate semantics are not metadata-aligned")
        object.__setattr__(self, "source_manifest_path", path)


@dataclass(frozen=True)
class DirectNumericalCoordinate:
    coordinate_id: str
    matrix_block: str
    column_index: int
    coordinate_name: str
    source_family: str
    source_kind: str
    producer_subarchitecture: str
    consumer_role: str
    observable_axes: tuple[str, ...]
    calibration_status: str
    statistic_kind: str
    statistic_rank: int | None
    statistic_width: int
    alignment_mode: str
    output_coordinate_identity_stable: bool
    source_coordinate_identity_preserved: bool
    source_cache_key: str
    matrix_binding_sha256: str
    column_values_sha256: str
    context_nonzero_count: int
    prediction_nonzero_count: int
    combined_standard_deviation: float
    observed_nonzero: bool
    observed_varying: bool
    shared_lineage_sha256: str
    lineage_scope: str
    concept_grounding_allowed: bool
    coordinate_identity_sha256: str
    signal_instance_sha256: str

    def coordinate_identity_fields(self) -> dict[str, Any]:
        """Fields that remain identical across outer folds and row scopes."""

        return {
            "matrix_block": self.matrix_block,
            "column_index": self.column_index,
            "coordinate_name": self.coordinate_name,
            "source_family": self.source_family,
            "source_kind": self.source_kind,
            "producer_subarchitecture": self.producer_subarchitecture,
            "consumer_role": self.consumer_role,
            "observable_axes": list(self.observable_axes),
            "calibration_status": self.calibration_status,
            "statistic_kind": self.statistic_kind,
            "statistic_rank": self.statistic_rank,
            "statistic_width": self.statistic_width,
            "alignment_mode": self.alignment_mode,
            "output_coordinate_identity_stable": self.output_coordinate_identity_stable,
            "source_coordinate_identity_preserved": self.source_coordinate_identity_preserved,
            "concept_grounding_allowed": self.concept_grounding_allowed,
        }

    def signal_instance_fields(self) -> dict[str, Any]:
        """Fold/package-specific binding for the stable coordinate."""

        return {
            "coordinate_identity_sha256": self.coordinate_identity_sha256,
            "source_cache_key": self.source_cache_key,
            "matrix_binding_sha256": self.matrix_binding_sha256,
            "column_values_sha256": self.column_values_sha256,
            "shared_lineage_sha256": self.shared_lineage_sha256,
            "lineage_scope": self.lineage_scope,
        }

    def __post_init__(self) -> None:
        _safe_id(self.coordinate_id, label="coordinate_id")
        if self.matrix_block not in MATRIX_BLOCKS:
            raise ValueError("coordinate matrix_block is unsupported")
        _positive_or_zero(self.column_index, label="column_index")
        _safe_text(self.coordinate_name, label="coordinate_name")
        if self.source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("coordinate source_family is inactive or unknown")
        _safe_text(self.source_kind, label="source_kind")
        _safe_text(self.producer_subarchitecture, label="producer_subarchitecture")
        _safe_text(self.consumer_role, label="consumer_role")
        if not self.observable_axes or len(set(self.observable_axes)) != len(self.observable_axes):
            raise ValueError("observable_axes must be non-empty and unique")
        if set(self.observable_axes) - {TREATMENT_AXIS, OUTCOME_AXIS, HETEROGENEITY_AXIS}:
            raise ValueError("direct numerical coordinate has an unsupported observable axis")
        if self.calibration_status not in {NESTED_CALIBRATED_STATUS, UNCALIBRATED_BASIS_STATUS}:
            raise ValueError("calibration_status is unsupported")
        if self.statistic_kind not in {"direct_prediction", *_RAW_STATISTIC_KINDS}:
            raise ValueError("statistic_kind is unsupported")
        width = _positive_or_zero(self.statistic_width, label="statistic_width")
        if width < 1:
            raise ValueError("statistic_width must be positive")
        if self.statistic_kind == "signed_descending_order":
            if self.statistic_rank is None:
                raise ValueError("signed order statistic requires a rank")
            rank = _positive_or_zero(self.statistic_rank, label="statistic_rank")
            if rank < 1 or rank > width:
                raise ValueError("statistic_rank must fall within statistic_width")
        elif self.statistic_rank is not None:
            raise ValueError("only signed order statistics may have a rank")
        if self.alignment_mode not in {
            EXACT_PRECOMMITTED_ALIGNMENT,
            *_RAW_ALIGNMENT_MODES,
        }:
            raise ValueError("alignment_mode is unsupported")
        if self.output_coordinate_identity_stable is not True:
            raise ValueError("every emitted output coordinate must have stable identity")
        exact_alignments = {
            EXACT_PRECOMMITTED_ALIGNMENT,
            EXACT_NAMED_RAW_ALIGNMENT,
        }
        if (self.alignment_mode in exact_alignments) != (self.source_coordinate_identity_preserved):
            raise ValueError("only exact precommitted coordinates may preserve source identity")
        if self.concept_grounding_allowed is not False:
            raise ValueError("direct numerical coordinates can never ground concepts")
        _require_sha256(self.source_cache_key, label="source_cache_key")
        _require_sha256(self.matrix_binding_sha256, label="matrix_binding_sha256")
        _require_sha256(self.column_values_sha256, label="column_values_sha256")
        context_nonzero = _positive_or_zero(
            self.context_nonzero_count, label="context_nonzero_count"
        )
        prediction_nonzero = _positive_or_zero(
            self.prediction_nonzero_count, label="prediction_nonzero_count"
        )
        if (
            isinstance(self.combined_standard_deviation, (bool, np.bool_))
            or not isinstance(self.combined_standard_deviation, (float, int, np.number))
            or not np.isfinite(float(self.combined_standard_deviation))
            or float(self.combined_standard_deviation) < 0.0
        ):
            raise ValueError("combined_standard_deviation must be finite and non-negative")
        if not isinstance(self.observed_nonzero, bool) or not isinstance(
            self.observed_varying, bool
        ):
            raise TypeError("observed activity flags must be boolean")
        if self.observed_nonzero != bool(context_nonzero + prediction_nonzero):
            raise ValueError("observed_nonzero conflicts with exact nonzero counts")
        if self.observed_varying != bool(float(self.combined_standard_deviation) > 0.0):
            raise ValueError("observed_varying conflicts with combined standard deviation")
        _require_sha256(self.shared_lineage_sha256, label="shared_lineage_sha256")
        _safe_text(self.lineage_scope, label="lineage_scope")
        expected = content_sha256(self.coordinate_identity_fields())
        if (
            _require_sha256(self.coordinate_identity_sha256, label="coordinate_identity_sha256")
            != expected
        ):
            raise ValueError("coordinate identity SHA-256 mismatch")
        if _require_sha256(
            self.signal_instance_sha256, label="signal_instance_sha256"
        ) != content_sha256(self.signal_instance_fields()):
            raise ValueError("signal instance SHA-256 mismatch")

    def as_dict(self) -> dict[str, Any]:
        return {
            "coordinate_id": self.coordinate_id,
            **self.coordinate_identity_fields(),
            "coordinate_identity_sha256": self.coordinate_identity_sha256,
            "source_cache_key": self.source_cache_key,
            "matrix_binding_sha256": self.matrix_binding_sha256,
            "column_values_sha256": self.column_values_sha256,
            "activity": {
                "context_nonzero_count": self.context_nonzero_count,
                "prediction_nonzero_count": self.prediction_nonzero_count,
                "combined_standard_deviation": self.combined_standard_deviation,
                "observed_nonzero": self.observed_nonzero,
                "observed_varying": self.observed_varying,
            },
            "shared_lineage_sha256": self.shared_lineage_sha256,
            "lineage_scope": self.lineage_scope,
            "signal_instance_sha256": self.signal_instance_sha256,
        }


@dataclass(frozen=True)
class DirectNumericalFamilyCoverage:
    source_family: str
    coordinate_ids: tuple[str, ...]
    source_kinds: tuple[str, ...]
    semantic_atom_ids: tuple[str, ...]
    semantic_atom_ids_sha256: str
    numerical_zero_reason: str = ""

    def __post_init__(self) -> None:
        if self.source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("family coverage has an inactive or unknown source_family")
        for label, values in (
            ("coordinate_ids", self.coordinate_ids),
            ("source_kinds", self.source_kinds),
            ("semantic_atom_ids", self.semantic_atom_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} cannot contain duplicates")
            for value in values:
                (
                    _safe_id(value, label=label)
                    if label != "source_kinds"
                    else _safe_text(value, label=label)
                )
        if not self.semantic_atom_ids:
            raise ValueError("every active architecture must bind its semantic atom IDs")
        if _require_sha256(
            self.semantic_atom_ids_sha256, label="semantic_atom_ids_sha256"
        ) != content_sha256(list(self.semantic_atom_ids)):
            raise ValueError("semantic atom ID SHA-256 mismatch")
        (
            _safe_text(
                self.numerical_zero_reason,
                label="numerical_zero_reason",
            )
            if self.numerical_zero_reason
            else None
        )
        if self.coordinate_ids and self.numerical_zero_reason:
            raise ValueError("a non-empty numerical family cannot have a zero reason")
        if not self.coordinate_ids and not self.numerical_zero_reason:
            raise ValueError("an empty numerical family requires an explicit zero reason")

    def as_dict(
        self,
        *,
        coordinates: Sequence[DirectNumericalCoordinate],
    ) -> dict[str, Any]:
        coordinate_id_set = set(self.coordinate_ids)
        family_coordinates = tuple(
            item for item in coordinates if item.coordinate_id in coordinate_id_set
        )
        return {
            "source_family": self.source_family,
            "coordinate_ids": list(self.coordinate_ids),
            "signal_count": len(self.coordinate_ids),
            "source_kinds": list(self.source_kinds),
            # These IDs bind complete architecture participation but are never
            # paired with an individual numerical coordinate.
            "semantic_atom_ids": list(self.semantic_atom_ids),
            "semantic_atom_ids_sha256": self.semantic_atom_ids_sha256,
            "coordinate_to_semantic_atom_linkage": False,
            "numerical_zero_reason": self.numerical_zero_reason,
            "observed_nonzero_signal_count": sum(
                item.observed_nonzero for item in family_coordinates
            ),
            "observed_varying_signal_count": sum(
                item.observed_varying for item in family_coordinates
            ),
            "distinct_nonzero_vector_count": len(
                {item.column_values_sha256 for item in family_coordinates if item.observed_nonzero}
            ),
        }


@dataclass(frozen=True)
class DirectUpstreamNumericalManifest:
    source_cache_schema: str
    source_cache_key: str
    source_manifest_sha256: str
    producer_identity_sha256: str
    stable_output_schema_sha256: str
    semantic_catalog_sha256: str
    shared_lineage_sha256: str
    lineage_scope: str
    matrices: tuple[AuthenticatedMatrixBinding, ...]
    coordinates: tuple[DirectNumericalCoordinate, ...]
    family_coverage: tuple[DirectNumericalFamilyCoverage, ...]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _safe_text(self.source_cache_schema, label="source_cache_schema")
        _require_sha256(self.source_cache_key, label="source_cache_key")
        _require_sha256(self.source_manifest_sha256, label="source_manifest_sha256")
        _require_sha256(self.producer_identity_sha256, label="producer_identity_sha256")
        _require_sha256(self.stable_output_schema_sha256, label="stable_output_schema_sha256")
        _require_sha256(self.semantic_catalog_sha256, label="semantic_catalog_sha256")
        _require_sha256(self.shared_lineage_sha256, label="shared_lineage_sha256")
        _safe_text(self.lineage_scope, label="lineage_scope")
        matrix_keys = [item.key for item in self.matrices]
        expected_matrix_keys = {(block, scope) for block in MATRIX_BLOCKS for scope in ROW_SCOPES}
        if set(matrix_keys) != expected_matrix_keys or len(matrix_keys) != len(set(matrix_keys)):
            raise ValueError("direct manifest matrix coverage is incomplete or duplicated")
        coordinate_ids = [item.coordinate_id for item in self.coordinates]
        if len(coordinate_ids) != len(set(coordinate_ids)):
            raise ValueError("direct numerical coordinate IDs must be unique")
        identities = [item.coordinate_identity_sha256 for item in self.coordinates]
        if len(identities) != len(set(identities)):
            raise ValueError("direct numerical coordinate identities must be unique")
        instances = [item.signal_instance_sha256 for item in self.coordinates]
        if len(instances) != len(set(instances)):
            raise ValueError("direct numerical signal instances must be unique")
        for block in MATRIX_BLOCKS:
            positions = [
                item.column_index for item in self.coordinates if item.matrix_block == block
            ]
            if positions != list(range(len(positions))):
                raise ValueError(f"{block} coordinate indices must be contiguous and ordered")
            widths = {item.shape[1] for item in self.matrices if item.matrix_block == block}
            if widths != {len(positions)}:
                raise ValueError(f"{block} coordinate count differs from matrix width")
        families = [item.source_family for item in self.family_coverage]
        if tuple(families) != ACTIVE_STAGE1_CONCEPT_FAMILIES:
            raise ValueError("family coverage must contain every active architecture in order")
        coordinates_by_family = {
            family: tuple(
                item.coordinate_id for item in self.coordinates if item.source_family == family
            )
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        }
        all_atom_ids: list[str] = []
        for coverage in self.family_coverage:
            if coverage.coordinate_ids != coordinates_by_family[coverage.source_family]:
                raise ValueError("family coverage does not exactly match numerical coordinates")
            expected_kinds = tuple(
                dict.fromkeys(
                    item.source_kind
                    for item in self.coordinates
                    if item.source_family == coverage.source_family
                )
            )
            if coverage.source_kinds != expected_kinds:
                raise ValueError("family source kinds do not match numerical coordinates")
            all_atom_ids.extend(coverage.semantic_atom_ids)
        if len(all_atom_ids) != len(set(all_atom_ids)):
            raise ValueError("semantic atom IDs must be globally unique across architectures")
        object.__setattr__(self, "content_sha256", content_sha256(self.content_dict()))

    @property
    def signal_count(self) -> int:
        return len(self.coordinates)

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DIRECT_NUMERICAL_MANIFEST_SCHEMA_VERSION,
            "channel": DIRECT_UPSTREAM_NUMERICAL_CHANNEL,
            "source_cache_schema": self.source_cache_schema,
            "source_cache_key": self.source_cache_key,
            "source_manifest_sha256": self.source_manifest_sha256,
            "producer_identity_sha256": self.producer_identity_sha256,
            "stable_output_schema_sha256": self.stable_output_schema_sha256,
            "semantic_catalog_sha256": self.semantic_catalog_sha256,
            "shared_lineage_sha256": self.shared_lineage_sha256,
            "lineage_scope": self.lineage_scope,
            "matrix_authentication": [item.as_dict() for item in self.matrices],
            "coordinates": [item.as_dict() for item in self.coordinates],
            "family_coverage": [
                item.as_dict(coordinates=self.coordinates) for item in self.family_coverage
            ],
            "signal_count": len(self.coordinates),
            "observed_nonzero_signal_count": sum(
                item.observed_nonzero for item in self.coordinates
            ),
            "observed_varying_signal_count": sum(
                item.observed_varying for item in self.coordinates
            ),
            "distinct_nonzero_vector_count": len(
                {item.column_values_sha256 for item in self.coordinates if item.observed_nonzero}
            ),
            "all_active_stage1_architectures_covered": True,
            "coordinate_to_semantic_atom_linkage": False,
            "concept_grounding_allowed": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.content_dict(), "content_sha256": self.content_sha256}

    def family(self, source_family: str) -> DirectNumericalFamilyCoverage:
        for item in self.family_coverage:
            if item.source_family == source_family:
                return item
        raise KeyError(source_family)


@dataclass(frozen=True)
class PersistedDirectNumericalManifest:
    path: Path
    file_sha256: str
    manifest: DirectUpstreamNumericalManifest

    def verify(self) -> None:
        path = Path(self.path).resolve(strict=True)
        if not path.is_file() or _sha256_file(path) != self.file_sha256:
            raise ValueError("persisted direct numerical manifest bytes changed")
        raw = _read_closed_json(path)
        if raw != self.manifest.as_dict():
            raise ValueError("persisted direct numerical manifest content changed")


def _matrix_binding(
    root: Path,
    *,
    block: str,
    scope: str,
    filename: Any,
    sha256: Any,
) -> AuthenticatedMatrixBinding:
    name = _safe_text(filename, label="matrix filename")
    digest = _require_sha256(sha256, label="matrix sha256")
    path = root / name
    return AuthenticatedMatrixBinding(
        matrix_block=block,
        row_scope=scope,
        filename=name,
        sha256=digest,
        shape=_matrix_shape(path, expected_sha256=digest),
    )


def _stable_output_schema_sha256(
    *,
    source_names: Sequence[str],
    source_kinds: Sequence[str],
    feature_names: Sequence[str],
    feature_kinds: Sequence[str],
    feature_roles: Sequence[str],
) -> str:
    return content_sha256(
        {
            "calibrated_sources": [
                {"name": name, "kind": kind} for name, kind in zip(source_names, source_kinds)
            ],
            "raw_features": [
                {"name": name, "kind": kind, "consumer_role": role}
                for name, kind, role in zip(feature_names, feature_kinds, feature_roles)
            ],
        }
    )


def _raw_semantic(
    *,
    name: str,
    kind: str,
    role: str,
    producer: str,
    statistic_kind: str,
    statistic_rank: int | None,
    statistic_width: int,
    alignment_mode: str,
    source_coordinate_identity_preserved: bool,
) -> AuthenticatedRawCoordinateSemantics:
    return AuthenticatedRawCoordinateSemantics(
        coordinate_name=name,
        source_kind=kind,
        consumer_role=role,
        producer_subarchitecture=producer,
        statistic_kind=statistic_kind,
        statistic_rank=statistic_rank,
        statistic_width=statistic_width,
        alignment_mode=alignment_mode,
        source_coordinate_identity_preserved=source_coordinate_identity_preserved,
    )


def _named_raw_producer(name: str, kind: str, role: str) -> str:
    """Recover the stable producing subarchitecture from an exact child name."""

    if kind == "bow_nuisance":
        match = re.fullmatch(
            r"stage1_raw__bow__([A-Za-z0-9_]+)__(?:treatment|outcome)_pred__as_"
            r"(?:propensity|outcome)",
            name,
        )
        if match is not None:
            return f"bow_nuisance_view:{match.group(1)}"
    elif kind == "bow_r_loss":
        match = re.fullmatch(
            r"stage1_raw__bow__([A-Za-z0-9_]+)__effect_pseudo_target_pred",
            name,
        )
        if match is not None:
            return f"bow_r_loss_view:{match.group(1)}"
    elif kind == "htr_nuisance":
        if re.fullmatch(
            r"stage1_raw__htr__nuisance__(?:treatment|outcome)_pred__as_" r"(?:propensity|outcome)",
            name,
        ):
            return "htr_nuisance_predictions"
    elif kind == "htr_neural":
        if name == "stage1_raw__htr__effect_pseudo_target_pred":
            return "htr_residual_effect_predictions"
    elif kind == "matched_pair_uplift":
        match = re.fullmatch(
            r"stage1_raw__bow__([A-Za-z0-9_]+)__matched_pair_"
            r"(?:uplift_delta_logit|treated_outcome_prob)",
            name,
        )
        if match is not None:
            return f"matched_pair_bow_view:{match.group(1)}"
        if re.fullmatch(
            r"stage1_raw__htr__matched_pair_" r"(?:uplift_delta_logit|treated_outcome_prob)",
            name,
        ):
            return "matched_pair_htr"
    elif kind == "embedding_whole_cohort":
        if re.fullmatch(
            r"stage1_raw__embedding__global_[A-Za-z0-9_]+__"
            r"(?:mean_cosine|max_cosine)(?:__as_(?:propensity|outcome))?",
            name,
        ):
            return "whole_cohort_embedding_contrasts"
    elif kind == "tfidf_topics":
        if name in {"tfidf_nuisance_treatment", "tfidf_nuisance_outcome"}:
            return "tfidf_nuisance_predictions"
    elif kind.startswith("neural_query_"):
        bank = {
            "neural_query_treatment_moments": "treatment",
            "neural_query_outcome_moments": "outcome",
            "neural_query_effect_moments": "effect",
        }.get(kind)
        if bank is not None and re.fullmatch(
            rf"neural_query_{bank}_(?:signed_mean|absolute_max|signed_order_[0-9]+)",
            name,
        ):
            return _RAW_KIND_PRODUCER[kind]
    raise ValueError(
        "exact named raw coordinate does not match its authenticated "
        f"producer convention: {(name, kind, role)!r}"
    )


def _volatile_raw_producer(kind: str, role: str) -> str:
    producer = _RAW_KIND_PRODUCER.get(kind)
    if producer is None:
        raise ValueError(f"unsupported volatile raw numerical source kind: {kind!r}")
    return f"{producer}:{role}"


def _validate_v2_gate_stable_schema_config(
    backend: Mapping[str, Any],
    *,
    source_names: Sequence[str],
    source_kinds: Sequence[str],
    feature_names: Sequence[str],
    feature_kinds: Sequence[str],
    feature_roles: Sequence[str],
) -> tuple[AuthenticatedRawCoordinateSemantics, ...]:
    """Authenticate the legacy pooled-family v2 fixed schema."""

    config = backend.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("gate cache stable backend has no structured config")
    if (
        config.get("reject_unconfigured_calibrated_sources") is not True
        or config.get("reject_unconfigured_raw_families") is not True
    ):
        raise ValueError("gate cache stable schema is not fail-closed")
    namespace = _safe_text(config.get("namespace"), label="stable schema namespace")
    configured_sources = config.get("calibrated_sources")
    configured_families = config.get("raw_families")
    if not isinstance(configured_sources, list) or not isinstance(configured_families, list):
        raise ValueError("gate cache stable schema lists are malformed")

    expected_sources: list[tuple[str, str]] = []
    for index, record in enumerate(configured_sources):
        if (
            not isinstance(record, Mapping)
            or record.get("exact_name_and_kind_required") is not True
        ):
            raise ValueError(f"calibrated source config {index} is not exact")
        name = record.get("output_name")
        kind = record.get("source_kind")
        if not isinstance(name, str) or not isinstance(kind, str):
            raise ValueError("calibrated source config metadata is malformed")
        if kind not in _CALIBRATED_KIND_FAMILY:
            raise ValueError(f"unrecognized calibrated source kind in stable config: {kind!r}")
        expected_sources.append((name, kind))
    if tuple(expected_sources) != tuple(zip(source_names, source_kinds)):
        raise ValueError("calibrated output metadata differs from the stable config")

    semantics: list[AuthenticatedRawCoordinateSemantics] = []
    for ordinal, record in enumerate(configured_families, start=1):
        if not isinstance(record, Mapping):
            raise ValueError("raw family config must contain objects")
        kind = record.get("source_kind")
        role = record.get("consumer_role")
        width = record.get("signed_order_width")
        if kind not in _RAW_KIND_FAMILY or role not in _ROLE_AXIS:
            raise ValueError("stable raw family kind or role is unsupported")
        width = _positive_or_zero(width, label="configured signed_order_width")
        if width < 1:
            raise ValueError("configured signed_order_width must be positive")
        if record.get("required") is not True:
            raise ValueError("benchmark direct numerical families must be required")
        exact_names = record.get("exact_passthrough_feature_names")
        if exact_names is not None:
            if record.get("reduction") != "exact_preaggregated_passthrough":
                raise ValueError("exact passthrough family changed its reduction")
            if not isinstance(exact_names, list) or len(exact_names) != width + 2:
                raise ValueError("exact passthrough names do not match configured width")
            names = tuple(str(name) for name in exact_names)
            alignment = PREAGGREGATED_PERMUTATION_SUMMARY_ALIGNMENT
        else:
            if record.get("summaries") != [
                "signed_mean",
                "absolute_max",
                "signed_descending_order",
            ]:
                raise ValueError("raw family summary semantics changed")
            prefix = f"{namespace}__family_{ordinal:03d}"
            names = (
                f"{prefix}__signed_mean",
                f"{prefix}__absolute_max",
                *(f"{prefix}__signed_order_{rank:03d}" for rank in range(1, width + 1)),
            )
            alignment = PERMUTATION_SUMMARY_ALIGNMENT
        producer = _RAW_KIND_PRODUCER[str(kind)]
        for name in names:
            statistic_kind, statistic_rank, statistic_width = _statistic_metadata(
                name,
                group_names=names,
            )
            semantics.append(
                _raw_semantic(
                    name=name,
                    kind=str(kind),
                    role=str(role),
                    producer=producer,
                    statistic_kind=statistic_kind,
                    statistic_rank=statistic_rank,
                    statistic_width=statistic_width,
                    alignment_mode=alignment,
                    source_coordinate_identity_preserved=False,
                )
            )
    expected_features = tuple(
        (item.coordinate_name, item.source_kind, item.consumer_role) for item in semantics
    )
    if expected_features != tuple(zip(feature_names, feature_kinds, feature_roles)):
        raise ValueError("raw output metadata differs from the structured stable config")
    return tuple(semantics)


def _validate_v3_gate_stable_schema_config(
    backend: Mapping[str, Any],
    *,
    source_names: Sequence[str],
    source_kinds: Sequence[str],
    feature_names: Sequence[str],
    feature_kinds: Sequence[str],
    feature_roles: Sequence[str],
) -> tuple[AuthenticatedRawCoordinateSemantics, ...]:
    """Authenticate v3 exact named coordinates and bounded volatile summaries."""

    expected_backend_fields = {
        "backend",
        "child",
        "config",
        "gate_labels_exposed_to_child",
        "raw_features_relabelled_as_calibrated_sources",
        "named_raw_coordinate_alignment",
        "volatile_raw_reduction",
        "child_column_consumption",
        "fixed_output_order",
        "same_rectangular_schema_safe_for_gate_and_final_consumers",
    }
    if set(backend) != expected_backend_fields:
        raise ValueError("coordinate-preserving backend identity has a wrong closed schema")
    expected_controls = {
        "gate_labels_exposed_to_child": False,
        "raw_features_relabelled_as_calibrated_sources": False,
        "named_raw_coordinate_alignment": "exact_child_name_kind_and_role",
        "volatile_raw_reduction": "permutation_invariant_after_named_claims",
        "child_column_consumption": "exactly_once",
        "fixed_output_order": True,
        "same_rectangular_schema_safe_for_gate_and_final_consumers": True,
    }
    if any(backend.get(key) != value for key, value in expected_controls.items()):
        raise ValueError("coordinate-preserving backend safety controls changed")
    config = backend.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("coordinate-preserving backend has no structured config")
    allowed_config_fields = {
        "namespace",
        "calibrated_sources",
        "named_raw_coordinates",
        "volatile_raw_families",
        "child_column_partition_order",
        "unconfigured_child_columns",
        "source_config_sha256",
    }
    if not set(config).issubset(allowed_config_fields) or not (
        allowed_config_fields - {"source_config_sha256"}
    ).issubset(config):
        raise ValueError("coordinate-preserving config has a wrong closed schema")
    if (
        config.get("child_column_partition_order")
        != ("exact_calibrated_then_named_raw_then_remaining_volatile_raw")
        or config.get("unconfigured_child_columns") != "reject"
    ):
        raise ValueError("coordinate-preserving child partition is not fail-closed")
    if "source_config_sha256" in config:
        _require_sha256(config["source_config_sha256"], label="source_config_sha256")
    namespace = _safe_text(config.get("namespace"), label="coordinate schema namespace")
    configured_sources = config.get("calibrated_sources")
    configured_coordinates = config.get("named_raw_coordinates")
    configured_families = config.get("volatile_raw_families")
    if not all(
        isinstance(value, list)
        for value in (configured_sources, configured_coordinates, configured_families)
    ):
        raise ValueError("coordinate-preserving config lists are malformed")

    expected_sources: list[tuple[str, str]] = []
    source_fields = {
        "child_name",
        "source_kind",
        "output_name",
        "matching",
        "required",
    }
    for index, record in enumerate(configured_sources):
        if not isinstance(record, Mapping) or set(record) != source_fields:
            raise ValueError(f"coordinate-preserving calibrated source {index} is malformed")
        child_name = _safe_text(record["child_name"], label="calibrated child_name")
        output_name = _safe_text(record["output_name"], label="calibrated output_name")
        kind = _safe_text(record["source_kind"], label="calibrated source_kind")
        if (
            record["matching"] != "exact_child_name_and_source_kind"
            or record["required"] is not True
            or kind not in _CALIBRATED_KIND_FAMILY
        ):
            raise ValueError("coordinate-preserving calibrated source is not exact")
        _calibrated_producer(child_name, kind)
        expected_sources.append((output_name, kind))
    if tuple(expected_sources) != tuple(zip(source_names, source_kinds)):
        raise ValueError("calibrated output metadata differs from the v3 config")

    semantics: list[AuthenticatedRawCoordinateSemantics] = []
    named_base_fields = {
        "child_name",
        "source_kind",
        "consumer_role",
        "output_name",
        "matching",
        "required",
    }
    for ordinal, record in enumerate(configured_coordinates, start=1):
        if not isinstance(record, Mapping):
            raise ValueError("named raw coordinate config must contain objects")
        required = record.get("required")
        expected_fields = (
            named_base_fields if required is True else named_base_fields | {"absence_encoding"}
        )
        if not isinstance(required, bool) or set(record) != expected_fields:
            raise ValueError("named raw coordinate has a wrong closed schema")
        if record.get("matching") != "exact_child_name_source_kind_and_consumer_role":
            raise ValueError("named raw coordinate matching contract changed")
        if required is False and record.get("absence_encoding") != (
            "presence_zero_then_zero_filled_coordinate"
        ):
            raise ValueError("optional named coordinate absence encoding changed")
        child_name = _safe_text(record["child_name"], label="named child_name")
        output_name = _safe_text(record["output_name"], label="named output_name")
        kind = _safe_text(record["source_kind"], label="named source_kind")
        role = _safe_text(record["consumer_role"], label="named consumer_role")
        if kind not in _RAW_KIND_FAMILY or role not in _ROLE_AXIS:
            raise ValueError("named raw coordinate kind or role is unsupported")
        producer = _named_raw_producer(child_name, kind, role)
        if required is False:
            semantics.append(
                _raw_semantic(
                    name=f"{namespace}__named_coordinate_{ordinal:03d}__presence",
                    kind=kind,
                    role=role,
                    producer=f"conditional_presence_for:{producer}",
                    statistic_kind="presence",
                    statistic_rank=None,
                    statistic_width=1,
                    alignment_mode=CONDITIONAL_PRESENCE_ALIGNMENT,
                    source_coordinate_identity_preserved=False,
                )
            )
        semantics.append(
            _raw_semantic(
                name=output_name,
                kind=kind,
                role=role,
                producer=producer,
                statistic_kind="exact_named_coordinate",
                statistic_rank=None,
                statistic_width=1,
                alignment_mode=EXACT_NAMED_RAW_ALIGNMENT,
                source_coordinate_identity_preserved=True,
            )
        )

    volatile_base_fields = {
        "source_kind",
        "consumer_role",
        "signed_order_width",
        "maximum_member_count",
        "required",
        "membership",
        "summaries",
    }
    for ordinal, record in enumerate(configured_families, start=1):
        if not isinstance(record, Mapping):
            raise ValueError("volatile raw family config must contain objects")
        fields = set(record)
        has_pattern = "child_name_pattern" in fields or "child_name_matching" in fields
        expected_fields = (
            volatile_base_fields | {"child_name_pattern", "child_name_matching"}
            if has_pattern
            else volatile_base_fields
        )
        if fields != expected_fields:
            raise ValueError("volatile raw family has a wrong closed schema")
        kind = _safe_text(record["source_kind"], label="volatile source_kind")
        role = _safe_text(record["consumer_role"], label="volatile consumer_role")
        if kind not in _RAW_KIND_FAMILY or role not in _ROLE_AXIS:
            raise ValueError("volatile raw family kind or role is unsupported")
        width = _positive_or_zero(record["signed_order_width"], label="volatile signed_order_width")
        maximum = _positive_or_zero(
            record["maximum_member_count"], label="volatile maximum_member_count"
        )
        if width < 1 or maximum != width:
            raise ValueError("volatile raw family width/capacity contract changed")
        required = record.get("required")
        if not isinstance(required, bool):
            raise ValueError("volatile raw family required flag is malformed")
        expected_summaries = [
            *([] if required is True else ["presence"]),
            "signed_mean",
            "absolute_max",
            "signed_descending_order",
        ]
        if (
            record.get("membership") != "remaining_columns_after_named_coordinate_claims"
            or record.get("summaries") != expected_summaries
        ):
            raise ValueError("volatile raw family reduction contract changed")
        if has_pattern:
            pattern = _safe_text(record["child_name_pattern"], label="volatile child_name_pattern")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError("volatile child_name_pattern is invalid") from exc
            if record["child_name_matching"] != "full_regular_expression_match":
                raise ValueError("volatile child-name matching contract changed")
        prefix = f"{namespace}__volatile_family_{ordinal:03d}"
        producer = _volatile_raw_producer(kind, role)
        metrics: list[tuple[str, str, int | None]] = []
        if required is False:
            metrics.append(("presence", "presence", None))
        metrics.extend(
            (
                ("signed_mean", "signed_mean", None),
                ("absolute_max", "absolute_max", None),
                *(
                    (
                        f"signed_order_{rank:03d}",
                        "signed_descending_order",
                        rank,
                    )
                    for rank in range(1, width + 1)
                ),
            )
        )
        for metric, statistic_kind, statistic_rank in metrics:
            semantics.append(
                _raw_semantic(
                    name=f"{prefix}__{metric}",
                    kind=kind,
                    role=role,
                    producer=producer,
                    statistic_kind=statistic_kind,
                    statistic_rank=statistic_rank,
                    statistic_width=width,
                    alignment_mode=(
                        CONDITIONAL_PRESENCE_ALIGNMENT
                        if statistic_kind == "presence"
                        else PERMUTATION_SUMMARY_ALIGNMENT
                    ),
                    source_coordinate_identity_preserved=False,
                )
            )
    expected_features = tuple(
        (item.coordinate_name, item.source_kind, item.consumer_role) for item in semantics
    )
    if expected_features != tuple(zip(feature_names, feature_kinds, feature_roles)):
        raise ValueError("raw output metadata differs from the structured v3 config")
    return tuple(semantics)


def _validate_gate_stable_schema_config(
    provider_identity: Mapping[str, Any],
    *,
    source_names: Sequence[str],
    source_kinds: Sequence[str],
    feature_names: Sequence[str],
    feature_kinds: Sequence[str],
    feature_roles: Sequence[str],
) -> tuple[AuthenticatedRawCoordinateSemantics, ...]:
    """Prove per-output semantics from the authenticated structured config."""

    backend = provider_identity.get("backend")
    if not isinstance(backend, Mapping):
        raise ValueError("gate cache does not expose the exact stable-schema backend")
    backend_id = backend.get("backend")
    arguments = {
        "source_names": source_names,
        "source_kinds": source_kinds,
        "feature_names": feature_names,
        "feature_kinds": feature_kinds,
        "feature_roles": feature_roles,
    }
    if backend_id == STABLE_CONTEXT_FIT_UPSTREAM_BACKEND_ID:
        return _validate_v2_gate_stable_schema_config(backend, **arguments)
    if backend_id == COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID:
        return _validate_v3_gate_stable_schema_config(backend, **arguments)
    raise ValueError("gate cache does not expose a supported stable-schema backend")


def _legacy_final_raw_coordinate_semantics(
    feature_names: Sequence[str],
    feature_kinds: Sequence[str],
    feature_roles: Sequence[str],
) -> tuple[AuthenticatedRawCoordinateSemantics, ...]:
    """Interpret only the old v2 self-describing final-cache output names.

    A v1 final-bank manifest authenticates the producer hash but does not embed
    its structured identity.  Therefore exact v3 named coordinates must fail
    here rather than be guessed from a filename-like output label.
    """

    groups: dict[tuple[str, str], list[str]] = {}
    for name, kind, role in zip(feature_names, feature_kinds, feature_roles):
        groups.setdefault((kind, role), []).append(name)
    semantics: list[AuthenticatedRawCoordinateSemantics] = []
    for name, kind, role in zip(feature_names, feature_kinds, feature_roles):
        if kind not in _RAW_KIND_FAMILY or role not in _ROLE_AXIS:
            raise ValueError("final raw numerical kind or role is unsupported")
        try:
            statistic_kind, statistic_rank, statistic_width = _statistic_metadata(
                name,
                group_names=groups[(kind, role)],
            )
        except ValueError as exc:
            raise ValueError(
                "final cache lacks structured producer identity needed to authenticate "
                "coordinate-preserving v3 raw semantics"
            ) from exc
        semantics.append(
            _raw_semantic(
                name=name,
                kind=kind,
                role=role,
                producer=_RAW_KIND_PRODUCER[kind],
                statistic_kind=statistic_kind,
                statistic_rank=statistic_rank,
                statistic_width=statistic_width,
                alignment_mode=(
                    PREAGGREGATED_PERMUTATION_SUMMARY_ALIGNMENT
                    if kind.startswith("neural_query_")
                    else PERMUTATION_SUMMARY_ALIGNMENT
                ),
                source_coordinate_identity_preserved=False,
            )
        )
    return tuple(semantics)


def load_authenticated_numerical_bank_snapshot(
    source_manifest_path: Path | str,
) -> AuthenticatedNumericalBankSnapshot:
    """Authenticate a v6 gate cache or v1 final cache without reading labels."""

    path = Path(source_manifest_path).resolve(strict=True)
    raw = _read_closed_json(path)
    schema = raw.get("schema_version")
    expected_fields = (
        _GATE_MANIFEST_FIELDS
        if schema == CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION
        else _FINAL_MANIFEST_FIELDS if schema == FINAL_CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA else None
    )
    if expected_fields is None:
        raise ValueError("unsupported upstream numerical cache schema")
    if set(raw) != expected_fields:
        raise ValueError("upstream numerical source manifest has a wrong closed schema")
    content = {key: value for key, value in raw.items() if key != "content_sha256"}
    if _require_sha256(raw["content_sha256"], label="source content_sha256") != content_sha256(
        content
    ):
        raise ValueError("upstream numerical source manifest content hash mismatch")
    cache_key = _require_sha256(raw["cache_key"], label="source cache_key")
    binding = raw["binding"]
    if not isinstance(binding, Mapping) or content_sha256(binding) != cache_key:
        raise ValueError("upstream numerical source binding does not match its cache key")
    if path.name != "manifest.json" or path.parent.name != cache_key:
        raise ValueError("upstream numerical source manifest path is not canonical")

    root = path.parent
    if schema == CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION:
        source_names = tuple(raw["source_names"])
        source_kinds = tuple(raw["source_kinds"])
        feature_names = tuple(raw["feature_names"])
        feature_kinds = tuple(raw["feature_kinds"])
        feature_roles = tuple(raw["feature_roles"])
        matrices = (
            _matrix_binding(
                root,
                block=CALIBRATED_SOURCES_BLOCK,
                scope=CONTEXT_OOF_SCOPE,
                filename=raw["source_context_values_file"],
                sha256=raw["source_context_values_sha256"],
            ),
            _matrix_binding(
                root,
                block=CALIBRATED_SOURCES_BLOCK,
                scope=PREDICTION_SCOPE,
                filename=raw["source_values_file"],
                sha256=raw["source_values_sha256"],
            ),
            _matrix_binding(
                root,
                block=RAW_FEATURES_BLOCK,
                scope=CONTEXT_OOF_SCOPE,
                filename=raw["feature_context_values_file"],
                sha256=raw["feature_context_values_sha256"],
            ),
            _matrix_binding(
                root,
                block=RAW_FEATURES_BLOCK,
                scope=PREDICTION_SCOPE,
                filename=raw["feature_values_file"],
                sha256=raw["feature_values_sha256"],
            ),
        )
        if matrices[0].shape[0] != len(raw["context_row_ids"]):
            raise ValueError("source context matrix row count changed")
        if matrices[2].shape[0] != len(raw["context_row_ids"]):
            raise ValueError("feature context matrix row count changed")
        if matrices[1].shape[0] != len(raw["gate_row_ids"]):
            raise ValueError("source prediction matrix row count changed")
        if matrices[3].shape[0] != len(raw["gate_row_ids"]):
            raise ValueError("feature prediction matrix row count changed")
        provider_identity = binding.get("provider_identity")
        if not isinstance(provider_identity, Mapping):
            raise ValueError("gate cache binding has no authenticated provider identity")
        raw_coordinate_semantics = _validate_gate_stable_schema_config(
            provider_identity,
            source_names=source_names,
            source_kinds=source_kinds,
            feature_names=feature_names,
            feature_kinds=feature_kinds,
            feature_roles=feature_roles,
        )
        producer_sha = content_sha256(provider_identity)
        lineage = {
            "source_cache_key": cache_key,
            "context_row_ids_sha256": binding.get("context_row_ids_sha256"),
            "context_inner_fold_assignment_sha256": binding.get(
                "context_inner_fold_assignment_sha256"
            ),
            "gate_row_ids_sha256": binding.get("gate_row_ids_sha256"),
            "context_values_cross_fitted_by_exact_inner_fold": binding.get(
                "context_values_cross_fitted_by_exact_inner_fold"
            ),
            "gate_labels_exposed_to_backend": binding.get("gate_labels_exposed_to_backend"),
        }
        lineage_scope = (
            "exact_inner_fold_oof_context_rows_and_complete_spent_context_fit_"
            "for_label_free_gate_rows"
        )
    else:
        source_record = raw["calibrated_sources"]
        feature_record = raw["raw_features"]
        if not isinstance(source_record, Mapping) or set(source_record) != {
            "names",
            "kinds",
            "content_sha256",
        }:
            raise ValueError("final calibrated-source metadata is malformed")
        if not isinstance(feature_record, Mapping) or set(feature_record) != {
            "names",
            "kinds",
            "roles",
            "content_sha256",
        }:
            raise ValueError("final raw-feature metadata is malformed")
        source_names = tuple(source_record["names"])
        source_kinds = tuple(source_record["kinds"])
        feature_names = tuple(feature_record["names"])
        feature_kinds = tuple(feature_record["kinds"])
        feature_roles = tuple(feature_record["roles"])
        records = raw["matrix_files"]
        if not isinstance(records, Mapping) or set(records) != {
            "source_train_oof",
            "source_outer_heldout",
            "feature_train_oof",
            "feature_outer_heldout",
        }:
            raise ValueError("final matrix records are malformed")

        def final_matrix(key: str, block: str, scope: str) -> AuthenticatedMatrixBinding:
            record = records[key]
            if not isinstance(record, Mapping) or set(record) != {"filename", "sha256"}:
                raise ValueError(f"final matrix record {key!r} is malformed")
            return _matrix_binding(
                root,
                block=block,
                scope=scope,
                filename=record["filename"],
                sha256=record["sha256"],
            )

        matrices = (
            final_matrix("source_train_oof", CALIBRATED_SOURCES_BLOCK, CONTEXT_OOF_SCOPE),
            final_matrix("source_outer_heldout", CALIBRATED_SOURCES_BLOCK, PREDICTION_SCOPE),
            final_matrix("feature_train_oof", RAW_FEATURES_BLOCK, CONTEXT_OOF_SCOPE),
            final_matrix("feature_outer_heldout", RAW_FEATURES_BLOCK, PREDICTION_SCOPE),
        )
        train_rows = binding.get("outer_train_row_ids")
        heldout_rows = binding.get("outer_heldout_row_ids")
        if not isinstance(train_rows, list) or not isinstance(heldout_rows, list):
            raise ValueError("final binding has malformed row identities")
        if matrices[0].shape[0] != len(train_rows) or matrices[2].shape[0] != len(train_rows):
            raise ValueError("final OOF matrix row count changed")
        if matrices[1].shape[0] != len(heldout_rows) or matrices[3].shape[0] != len(heldout_rows):
            raise ValueError("final heldout matrix row count changed")
        producer_sha = _require_sha256(
            binding.get("producer_identity_sha256"), label="producer_identity_sha256"
        )
        lineage = {
            "source_cache_key": cache_key,
            "outer_train_row_ids_sha256": content_sha256(train_rows),
            "outer_heldout_row_ids_sha256": content_sha256(heldout_rows),
            "meta_inner_fold_ids_sha256": content_sha256(binding.get("meta_inner_fold_ids")),
            "outer_heldout_labels_accepted": binding.get("outer_heldout_labels_accepted"),
        }
        lineage_scope = (
            "meta_inner_oof_outer_train_and_full_outer_train_fit_for_label_free_"
            "outer_heldout_rows"
        )
        raw_coordinate_semantics = _legacy_final_raw_coordinate_semantics(
            feature_names,
            feature_kinds,
            feature_roles,
        )

    return AuthenticatedNumericalBankSnapshot(
        source_manifest_path=path,
        source_manifest_sha256=_sha256_file(path),
        source_cache_schema=str(schema),
        source_cache_key=cache_key,
        producer_identity_sha256=producer_sha,
        stable_output_schema_sha256=_stable_output_schema_sha256(
            source_names=source_names,
            source_kinds=source_kinds,
            feature_names=feature_names,
            feature_kinds=feature_kinds,
            feature_roles=feature_roles,
        ),
        shared_lineage_sha256=content_sha256(lineage),
        lineage_scope=lineage_scope,
        matrices=matrices,
        calibrated_source_names=source_names,
        calibrated_source_kinds=source_kinds,
        raw_feature_names=feature_names,
        raw_feature_kinds=feature_kinds,
        raw_feature_roles=feature_roles,
        raw_coordinate_semantics=raw_coordinate_semantics,
    )


def _statistic_metadata(
    name: str,
    *,
    group_names: Sequence[str],
) -> tuple[str, int | None, int]:
    ranks = []
    for candidate in group_names:
        match = _SIGNED_ORDER.search(candidate)
        if match:
            ranks.append(int(match.group(1)))
    width = max(ranks, default=1)
    if name.endswith("signed_mean"):
        return "signed_mean", None, width
    if name.endswith("absolute_max"):
        return "absolute_max", None, width
    if name.endswith("presence"):
        return "presence", None, width
    match = _SIGNED_ORDER.search(name)
    if match:
        return "signed_descending_order", int(match.group(1)), width
    raise ValueError(f"raw feature name does not encode its stable statistic: {name!r}")


def _calibrated_producer(name: str, kind: str) -> str:
    if kind == "nested_calibrated_bow_weighted_r":
        match = re.fullmatch(
            r"stage1_calibrated__bow__([A-Za-z0-9_]+)__effect_weighted_r_tau_pred",
            name,
        )
        if match is None:
            raise ValueError("calibrated BoW source name does not expose its fixed view")
        return f"bow_weighted_r_view:{match.group(1)}"
    if kind == "nested_calibrated_htr_weighted_r":
        if name != "stage1_calibrated__htr__effect_weighted_r_tau_pred":
            raise ValueError("calibrated HTR source name changed")
        return "htr_weighted_r"
    raise ValueError(f"unsupported calibrated source kind: {kind!r}")


def _column_activity_map(
    snapshot: AuthenticatedNumericalBankSnapshot,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Reauthenticate arrays and summarize activity without exposing row values."""

    arrays: dict[tuple[str, str], np.ndarray] = {}
    for binding in snapshot.matrices:
        path = snapshot.source_manifest_path.parent / binding.filename
        if _sha256_file(path) != binding.sha256:
            raise ValueError("numerical matrix changed after snapshot authentication")
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if values.shape != binding.shape or not np.isfinite(values).all():
            raise ValueError("numerical matrix shape or finite-value contract changed")
        arrays[binding.key] = values
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for block in MATRIX_BLOCKS:
        context = arrays[(block, CONTEXT_OOF_SCOPE)]
        prediction = arrays[(block, PREDICTION_SCOPE)]
        for index in range(context.shape[1]):
            context_column = np.asarray(context[:, index], dtype="<f8")
            prediction_column = np.asarray(prediction[:, index], dtype="<f8")
            combined = np.concatenate((context_column, prediction_column))
            digest = hashlib.sha256()
            for scope, column in (
                (CONTEXT_OOF_SCOPE, context_column),
                (PREDICTION_SCOPE, prediction_column),
            ):
                digest.update(scope.encode("utf-8"))
                digest.update(len(column).to_bytes(8, "big"))
                digest.update(column.tobytes(order="C"))
            context_nonzero = int(np.count_nonzero(context_column))
            prediction_nonzero = int(np.count_nonzero(prediction_column))
            standard_deviation = float(np.std(combined, dtype=np.float64))
            result[(block, index)] = {
                "column_values_sha256": digest.hexdigest(),
                "context_nonzero_count": context_nonzero,
                "prediction_nonzero_count": prediction_nonzero,
                "combined_standard_deviation": standard_deviation,
                "observed_nonzero": bool(context_nonzero + prediction_nonzero),
                "observed_varying": bool(standard_deviation > 0.0),
            }
    return result


def _make_coordinate(
    *,
    matrix_block: str,
    column_index: int,
    coordinate_name: str,
    source_family: str,
    source_kind: str,
    producer_subarchitecture: str,
    consumer_role: str,
    observable_axis: str,
    calibration_status: str,
    statistic_kind: str,
    statistic_rank: int | None,
    statistic_width: int,
    alignment_mode: str,
    source_coordinate_identity_preserved: bool,
    activity: Mapping[str, Any],
    snapshot: AuthenticatedNumericalBankSnapshot,
) -> DirectNumericalCoordinate:
    coordinate_identity = {
        "matrix_block": matrix_block,
        "column_index": column_index,
        "coordinate_name": coordinate_name,
        "source_family": source_family,
        "source_kind": source_kind,
        "producer_subarchitecture": producer_subarchitecture,
        "consumer_role": consumer_role,
        "observable_axes": [observable_axis],
        "calibration_status": calibration_status,
        "statistic_kind": statistic_kind,
        "statistic_rank": statistic_rank,
        "statistic_width": statistic_width,
        "alignment_mode": alignment_mode,
        "output_coordinate_identity_stable": True,
        "source_coordinate_identity_preserved": source_coordinate_identity_preserved,
        "concept_grounding_allowed": False,
    }
    identity_sha = content_sha256(coordinate_identity)
    coordinate_id = f"num.{matrix_block}.{column_index:04d}.{identity_sha[:12]}"
    matrix_binding_sha = content_sha256(
        [item.as_dict() for item in snapshot.matrices if item.matrix_block == matrix_block]
    )
    instance = {
        "coordinate_identity_sha256": identity_sha,
        "source_cache_key": snapshot.source_cache_key,
        "matrix_binding_sha256": matrix_binding_sha,
        "column_values_sha256": activity["column_values_sha256"],
        "shared_lineage_sha256": snapshot.shared_lineage_sha256,
        "lineage_scope": snapshot.lineage_scope,
    }
    return DirectNumericalCoordinate(
        coordinate_id=coordinate_id,
        matrix_block=matrix_block,
        column_index=column_index,
        coordinate_name=coordinate_name,
        source_family=source_family,
        source_kind=source_kind,
        producer_subarchitecture=producer_subarchitecture,
        consumer_role=consumer_role,
        observable_axes=(observable_axis,),
        calibration_status=calibration_status,
        statistic_kind=statistic_kind,
        statistic_rank=statistic_rank,
        statistic_width=statistic_width,
        alignment_mode=alignment_mode,
        output_coordinate_identity_stable=True,
        source_coordinate_identity_preserved=source_coordinate_identity_preserved,
        source_cache_key=snapshot.source_cache_key,
        matrix_binding_sha256=matrix_binding_sha,
        column_values_sha256=str(activity["column_values_sha256"]),
        context_nonzero_count=int(activity["context_nonzero_count"]),
        prediction_nonzero_count=int(activity["prediction_nonzero_count"]),
        combined_standard_deviation=float(activity["combined_standard_deviation"]),
        observed_nonzero=bool(activity["observed_nonzero"]),
        observed_varying=bool(activity["observed_varying"]),
        shared_lineage_sha256=snapshot.shared_lineage_sha256,
        lineage_scope=snapshot.lineage_scope,
        concept_grounding_allowed=False,
        coordinate_identity_sha256=identity_sha,
        signal_instance_sha256=content_sha256(instance),
    )


def build_direct_upstream_numerical_manifest(
    snapshot: AuthenticatedNumericalBankSnapshot,
    *,
    semantic_catalog_sha256: str,
    semantic_atom_ids_by_family: Mapping[str, Sequence[str]],
    numerical_zero_reasons: Mapping[str, str] | None = None,
) -> DirectUpstreamNumericalManifest:
    """Build a complete all-architecture manifest from authenticated metadata."""

    if not isinstance(snapshot, AuthenticatedNumericalBankSnapshot):
        raise TypeError("snapshot must be AuthenticatedNumericalBankSnapshot")
    _require_sha256(semantic_catalog_sha256, label="semantic_catalog_sha256")
    if set(semantic_atom_ids_by_family) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError("semantic atom bindings must cover every active Stage-1 architecture")
    zero_reasons = dict(numerical_zero_reasons or {})
    if set(zero_reasons) - ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError("numerical zero reasons contain an unknown architecture")

    activity = _column_activity_map(snapshot)
    coordinates: list[DirectNumericalCoordinate] = []
    for index, (name, kind) in enumerate(
        zip(snapshot.calibrated_source_names, snapshot.calibrated_source_kinds)
    ):
        family = _CALIBRATED_KIND_FAMILY.get(kind)
        if family is None:
            raise ValueError(f"unsupported calibrated source kind: {kind!r}")
        coordinates.append(
            _make_coordinate(
                matrix_block=CALIBRATED_SOURCES_BLOCK,
                column_index=index,
                coordinate_name=name,
                source_family=family,
                source_kind=kind,
                producer_subarchitecture=_calibrated_producer(name, kind),
                consumer_role=EFFECT_REGRESSION_COVARIATE_ROLE,
                observable_axis=HETEROGENEITY_AXIS,
                calibration_status=NESTED_CALIBRATED_STATUS,
                statistic_kind="direct_prediction",
                statistic_rank=None,
                statistic_width=1,
                alignment_mode=EXACT_PRECOMMITTED_ALIGNMENT,
                source_coordinate_identity_preserved=True,
                activity=activity[(CALIBRATED_SOURCES_BLOCK, index)],
                snapshot=snapshot,
            )
        )

    for index, semantic in enumerate(snapshot.raw_coordinate_semantics):
        family = _RAW_KIND_FAMILY.get(semantic.source_kind)
        axis = _ROLE_AXIS.get(semantic.consumer_role)
        if family is None:
            raise ValueError(f"unsupported raw numerical source kind: {semantic.source_kind!r}")
        if axis is None:
            raise ValueError(
                "unsupported raw numerical consumer role: " f"{semantic.consumer_role!r}"
            )
        coordinates.append(
            _make_coordinate(
                matrix_block=RAW_FEATURES_BLOCK,
                column_index=index,
                coordinate_name=semantic.coordinate_name,
                source_family=family,
                source_kind=semantic.source_kind,
                producer_subarchitecture=semantic.producer_subarchitecture,
                consumer_role=semantic.consumer_role,
                observable_axis=axis,
                calibration_status=UNCALIBRATED_BASIS_STATUS,
                statistic_kind=semantic.statistic_kind,
                statistic_rank=semantic.statistic_rank,
                statistic_width=semantic.statistic_width,
                alignment_mode=semantic.alignment_mode,
                source_coordinate_identity_preserved=(
                    semantic.source_coordinate_identity_preserved
                ),
                activity=activity[(RAW_FEATURES_BLOCK, index)],
                snapshot=snapshot,
            )
        )

    family_coverage: list[DirectNumericalFamilyCoverage] = []
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        family_coordinates = tuple(
            item.coordinate_id for item in coordinates if item.source_family == family
        )
        kinds = tuple(
            dict.fromkeys(item.source_kind for item in coordinates if item.source_family == family)
        )
        atom_ids = _string_tuple(
            semantic_atom_ids_by_family[family],
            label=f"semantic_atom_ids_by_family[{family}]",
        )
        reason = zero_reasons.get(family, "")
        family_coverage.append(
            DirectNumericalFamilyCoverage(
                source_family=family,
                coordinate_ids=family_coordinates,
                source_kinds=kinds,
                semantic_atom_ids=atom_ids,
                semantic_atom_ids_sha256=content_sha256(list(atom_ids)),
                numerical_zero_reason=reason,
            )
        )

    return DirectUpstreamNumericalManifest(
        source_cache_schema=snapshot.source_cache_schema,
        source_cache_key=snapshot.source_cache_key,
        source_manifest_sha256=snapshot.source_manifest_sha256,
        producer_identity_sha256=snapshot.producer_identity_sha256,
        stable_output_schema_sha256=snapshot.stable_output_schema_sha256,
        semantic_catalog_sha256=semantic_catalog_sha256,
        shared_lineage_sha256=snapshot.shared_lineage_sha256,
        lineage_scope=snapshot.lineage_scope,
        matrices=snapshot.matrices,
        coordinates=tuple(coordinates),
        family_coverage=tuple(family_coverage),
    )


def selector_facing_numerical_summary(
    manifest: DirectUpstreamNumericalManifest,
    *,
    strength_by_family: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a compact non-grounding view with no coordinate/atom pairing."""

    if not isinstance(manifest, DirectUpstreamNumericalManifest):
        raise TypeError("manifest must be DirectUpstreamNumericalManifest")
    strengths = dict(strength_by_family or {})
    if set(strengths) - ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError("strength summary contains an unknown architecture")
    rows = []
    for coverage in manifest.family_coverage:
        family_coordinates = [
            item for item in manifest.coordinates if item.source_family == coverage.source_family
        ]
        strength = strengths.get(coverage.source_family)
        if strength is not None:
            if not isinstance(strength, Mapping):
                raise TypeError("family strength must be one JSON object")
            # Round-trip through closed JSON and reject any lexical or row-level
            # attachment fields.  Numeric aggregation semantics are supplied by
            # the caller's independently authenticated summary.
            for key in strength:
                lowered = str(key).lower()
                if any(token in lowered for token in ("atom", "feature_name", "row", "patient")):
                    raise ValueError("strength summary cannot link numerical and lexical evidence")
            strength_view: Any = json.loads(canonical_json(strength))
        else:
            strength_view = {
                "available": False,
                "reason": "not_computed_in_metadata_only_manifest",
            }
        rows.append(
            {
                "source_family": coverage.source_family,
                "signal_count": len(family_coordinates),
                "observed_nonzero_signal_count": sum(
                    item.observed_nonzero for item in family_coordinates
                ),
                "observed_varying_signal_count": sum(
                    item.observed_varying for item in family_coordinates
                ),
                "distinct_nonzero_vector_count": len(
                    {
                        item.column_values_sha256
                        for item in family_coordinates
                        if item.observed_nonzero
                    }
                ),
                "calibrated_signal_count": sum(
                    item.calibration_status == NESTED_CALIBRATED_STATUS
                    for item in family_coordinates
                ),
                "alignment_modes": list(
                    dict.fromkeys(item.alignment_mode for item in family_coordinates)
                ),
                "source_coordinate_identity_preserved_count": sum(
                    item.source_coordinate_identity_preserved for item in family_coordinates
                ),
                "strength": strength_view,
                "scope": manifest.lineage_scope,
                "numerical_zero_reason": coverage.numerical_zero_reason,
            }
        )
    return {
        "schema_version": DIRECT_NUMERICAL_SELECTOR_VIEW_VERSION,
        "channel": DIRECT_UPSTREAM_NUMERICAL_CHANNEL,
        "manifest_sha256": manifest.content_sha256,
        "signal_count": manifest.signal_count,
        "observed_nonzero_signal_count": sum(
            item.observed_nonzero for item in manifest.coordinates
        ),
        "observed_varying_signal_count": sum(
            item.observed_varying for item in manifest.coordinates
        ),
        "distinct_nonzero_vector_count": len(
            {item.column_values_sha256 for item in manifest.coordinates if item.observed_nonzero}
        ),
        "families": rows,
        "row_values_included": False,
        "coordinate_names_included": False,
        "semantic_atom_ids_included": False,
        "coordinate_to_semantic_atom_linkage": False,
        "concept_grounding_allowed": False,
    }


def validate_architecture_dossier_numerical_binding(
    dossier: Any,
    manifest: DirectUpstreamNumericalManifest,
) -> None:
    """Prove that one compact dossier names its exact direct channel slice."""

    # Local import avoids making the role-independent interface depend on this
    # storage/authentication module.
    from .all_evidence_discovery_interfaces import ArchitectureDossier

    if not isinstance(dossier, ArchitectureDossier):
        raise TypeError("dossier must be ArchitectureDossier")
    if not isinstance(manifest, DirectUpstreamNumericalManifest):
        raise TypeError("manifest must be DirectUpstreamNumericalManifest")
    coverage = manifest.family(dossier.source_family)
    if dossier.catalog_sha256 != manifest.semantic_catalog_sha256:
        raise ValueError("dossier catalog SHA-256 differs from the numerical manifest")
    if dossier.direct_numerical_manifest_sha256 != manifest.content_sha256:
        raise ValueError("dossier direct numerical manifest SHA-256 changed")
    if dossier.catalog_evidence_ids != coverage.semantic_atom_ids:
        raise ValueError("dossier evidence IDs differ from the manifest family binding")
    if dossier.direct_numerical_signal_count != len(coverage.coordinate_ids):
        raise ValueError("dossier direct numerical signal count changed")
    if dossier.direct_numerical_zero_reason != coverage.numerical_zero_reason:
        raise ValueError("dossier direct numerical zero reason changed")


def write_direct_upstream_numerical_manifest(
    manifest: DirectUpstreamNumericalManifest,
    destination: Path | str,
) -> PersistedDirectNumericalManifest:
    """Atomically persist one immutable sidecar; identical replay is allowed."""

    if not isinstance(manifest, DirectUpstreamNumericalManifest):
        raise TypeError("manifest must be DirectUpstreamNumericalManifest")
    path = Path(destination).resolve()
    if path.name != "direct_upstream_numerical_manifest.json":
        raise ValueError("destination must use the canonical direct manifest filename")
    payload = canonical_json(manifest.as_dict()) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError("refusing to overwrite a different direct numerical manifest")
    else:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".direct_upstream_numerical_manifest.",
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    persisted = PersistedDirectNumericalManifest(
        path=path,
        file_sha256=_sha256_file(path),
        manifest=manifest,
    )
    persisted.verify()
    return persisted


__all__ = [
    "CALIBRATED_SOURCES_BLOCK",
    "CONDITIONAL_PRESENCE_ALIGNMENT",
    "CONTEXT_OOF_SCOPE",
    "DIRECT_NUMERICAL_MANIFEST_SCHEMA_VERSION",
    "DIRECT_NUMERICAL_SELECTOR_VIEW_VERSION",
    "EFFECT_REGRESSION_COVARIATE_ROLE",
    "EXACT_PRECOMMITTED_ALIGNMENT",
    "EXACT_NAMED_RAW_ALIGNMENT",
    "NESTED_CALIBRATED_STATUS",
    "PERMUTATION_SUMMARY_ALIGNMENT",
    "PREAGGREGATED_PERMUTATION_SUMMARY_ALIGNMENT",
    "PREDICTION_SCOPE",
    "RAW_FEATURES_BLOCK",
    "SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON",
    "UNCALIBRATED_BASIS_STATUS",
    "AuthenticatedMatrixBinding",
    "AuthenticatedNumericalBankSnapshot",
    "AuthenticatedRawCoordinateSemantics",
    "DirectNumericalCoordinate",
    "DirectNumericalFamilyCoverage",
    "DirectUpstreamNumericalManifest",
    "PersistedDirectNumericalManifest",
    "build_direct_upstream_numerical_manifest",
    "load_authenticated_numerical_bank_snapshot",
    "selector_facing_numerical_summary",
    "validate_architecture_dossier_numerical_binding",
    "write_direct_upstream_numerical_manifest",
]
