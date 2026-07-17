"""Authenticated fold-local fusion of numerical treatment-effect signals.

This module is intentionally separate from the historical Stage-1 artifact
readers.  Those artifacts contain useful numerical predictions, but their
parquet and NPZ schemas do not record the exact row-level fit lineage needed by
``FoldHonestRStack``.  Treating a prose ``split_role`` as proof of cross-fitting
would silently weaken the honesty contract.

The schema below therefore requires an authenticated sidecar with:

* exact outer-train and outer-heldout row identities;
* one exact inner-OOF lineage and generation fold per training prediction;
* outer-heldout predictions whose recursive fit lineage is confined to the
  outer-train rows; and
* explicit producer attestations that post-hoc targets, outer-heldout labels,
  and dataset-specific truth metadata were not consumed.

No oracle treatment-effect target is accepted by any estimator interface.
Observed treatment and outcome are supplied only for the outer-train R-loss
fit after the artifact has passed structural and provenance validation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from .fold_honest_r_stack import (
    INNER_OOF_SCOPE,
    OUTER_HELDOUT_SCOPE,
    FitRowProvenance,
    FoldHonestRStack,
    SignalBundle,
)

FOLD_NUMERICAL_SIGNAL_SCHEMA_VERSION = "fold_honest_numerical_signals_v2"
FOLD_NUMERICAL_SIGNAL_MANIFEST_SCHEMA_VERSION = (
    "fold_honest_numerical_signals_manifest_v1"
)
NUMERICAL_SIGNAL_FUSION_AUDIT_SCHEMA_VERSION = "fold_honest_numerical_signal_fusion_audit_v2"

CALIBRATED_TAU_ROLE = "calibrated_tau"
RAW_FEATURE_ROLE = "raw_feature"
SUPPORTED_SIGNAL_ROLES = frozenset({CALIBRATED_TAU_ROLE, RAW_FEATURE_ROLE})

PRODUCER_CODE_MATERIAL = "producer_code"
PRODUCER_CONFIG_MATERIAL = "producer_config"
INPUT_MATERIAL = "input"
BACKEND_CODE_MATERIAL = "backend_code"
BACKEND_CONFIG_MATERIAL = "backend_config"
MODEL_PROJECTION_MATERIAL = "model_projection"
REQUIRED_MATERIAL_CATEGORIES = frozenset(
    {
        PRODUCER_CODE_MATERIAL,
        PRODUCER_CONFIG_MATERIAL,
        INPUT_MATERIAL,
        BACKEND_CODE_MATERIAL,
        BACKEND_CONFIG_MATERIAL,
        MODEL_PROJECTION_MATERIAL,
    }
)

BOW_R_LOSS = "bow_r_loss"
HTR_NEURAL = "htr_neural"
MATCHED_PAIR_UPLIFT = "matched_pair_uplift"
WHOLE_EMBEDDING_CONTRAST = "whole_embedding_contrast"
CLUSTER_EMBEDDING_CONTRAST = "cluster_embedding_contrast"
TFIDF_TOPIC_CONTRAST = "tfidf_topic_contrast"
NEURAL_QUERY_MOMENTS = "neural_query_moments"
# Backwards-compatible symbol for callers that treat all numerical inputs as
# generic signals.  The value intentionally matches the established
# all-evidence source-family name.
NEURAL_QUERY_SIGNAL = NEURAL_QUERY_MOMENTS

SUPPORTED_SIGNAL_KINDS = frozenset(
    {
        BOW_R_LOSS,
        HTR_NEURAL,
        MATCHED_PAIR_UPLIFT,
        WHOLE_EMBEDDING_CONTRAST,
        CLUSTER_EMBEDDING_CONTRAST,
        TFIDF_TOPIC_CONTRAST,
        NEURAL_QUERY_MOMENTS,
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_NAME = re.compile(r"(?:^|_)(?:true|oracle|ground_truth)(?:_|$)", flags=re.IGNORECASE)
_AUTHENTICATED_LOADER_CAPABILITY = object()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_closed_json(raw: bytes, *, name: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc


def _validate_sha256(value: Any, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _canonical_row_id(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must contain canonical integer row IDs")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} cannot contain negative row IDs")
    return normalized


def _row_id_tuple(
    values: Sequence[Any], *, name: str, require_nonempty: bool = True
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc
    normalized = tuple(_canonical_row_id(value, name=name) for value in raw_values)
    if require_nonempty and not normalized:
        raise ValueError(f"{name} must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must contain unique row IDs")
    return normalized


def _fold_id_tuple(values: Sequence[Any], *, name: str, length: int) -> tuple[Hashable, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc
    normalized: list[Hashable] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer, str)):
            raise TypeError(f"{name} entries must be integer or string fold IDs")
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError(f"{name} entries must be non-empty")
        else:
            value = int(value)
            if value < 1:
                raise ValueError(f"{name} integer entries must be positive")
        normalized.append(value)
    if len(normalized) != int(length):
        raise ValueError(f"{name} must have length {length}")
    if len(set(normalized)) < 2:
        raise ValueError(f"{name} must contain at least two inner folds")
    return tuple(normalized)


def _numeric_vector(values: Sequence[Any], *, name: str, length: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) != int(length):
        raise ValueError(f"{name} must be one-dimensional with length {length}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    vector = vector.copy()
    vector.setflags(write=False)
    return vector


def row_set_fingerprint(row_ids: Sequence[Any]) -> str:
    """Return the repository-compatible order-insensitive row fingerprint."""

    normalized = _row_id_tuple(row_ids, name="row_ids")
    payload = json.dumps(
        sorted(str(value) for value in normalized),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _closed_object(
    value: Any,
    *,
    required: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    keys = set(value)
    missing = sorted(required - keys)
    unexpected = sorted(keys - required)
    if missing or unexpected:
        raise ValueError(
            f"{name} does not match its closed schema; "
            f"missing={missing} unexpected={unexpected}"
        )
    return value


@dataclass(frozen=True)
class AuthenticatedMaterialFile:
    """One exact on-disk producer material authenticated by the manifest.

    Categories are closed and semantic.  In particular, a model projection is
    not interchangeable with a backend configuration or an input artifact.
    The writer hashes the file itself; callers never supply its digest.
    """

    category: str
    name: str
    path: Path | str

    def __post_init__(self) -> None:
        category = str(self.category).strip().lower()
        if category not in REQUIRED_MATERIAL_CATEGORIES:
            raise ValueError(
                f"material category must be one of {sorted(REQUIRED_MATERIAL_CATEGORIES)}"
            )
        name = str(self.name).strip()
        if not name or _FORBIDDEN_NAME.search(name):
            raise ValueError("material name must be non-empty and truth-agnostic")
        path = Path(self.path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"authenticated material is not a regular file: {path}")
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class _AuthenticatedMaterialRecord:
    category: str
    name: str
    path: Path
    sha256: str
    size_bytes: int


def _array_authentication_record(values: Any) -> Mapping[str, Any]:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    header = {
        "dtype": "<f8",
        "shape": list(array.shape),
        "order": "C",
    }
    encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        **header,
        "sha256": _sha256_bytes(encoded_header + b"\0" + array.tobytes(order="C")),
    }


@dataclass(frozen=True)
class NumericalSignalProducerAudit:
    """Fail-closed producer identity and non-oracle attestations."""

    producer_id: str
    producer_code_sha256: str
    producer_config_sha256: str
    input_artifact_sha256s: Mapping[str, str]
    posthoc_targets_consumed: bool
    outer_heldout_labels_consumed: bool
    dataset_specific_truth_consumed: bool

    def __post_init__(self) -> None:
        producer_id = str(self.producer_id).strip()
        if not producer_id or _FORBIDDEN_NAME.search(producer_id):
            raise ValueError("producer_id must be non-empty and truth-agnostic")
        inputs = dict(self.input_artifact_sha256s)
        if not inputs:
            raise ValueError("input_artifact_sha256s must authenticate at least one input")
        normalized_inputs: dict[str, str] = {}
        for name, digest in inputs.items():
            normalized_name = str(name).strip()
            if not normalized_name or _FORBIDDEN_NAME.search(normalized_name):
                raise ValueError("input artifact names must be non-empty and truth-agnostic")
            if normalized_name in normalized_inputs:
                raise ValueError(f"duplicate input artifact name {normalized_name!r}")
            normalized_inputs[normalized_name] = _validate_sha256(
                digest, name=f"input_artifact_sha256s[{normalized_name!r}]"
            )
        for flag_name in (
            "posthoc_targets_consumed",
            "outer_heldout_labels_consumed",
            "dataset_specific_truth_consumed",
        ):
            value = getattr(self, flag_name)
            if not isinstance(value, bool):
                raise TypeError(f"{flag_name} must be a boolean")
            if value:
                raise ValueError(f"{flag_name} must be false")
        object.__setattr__(self, "producer_id", producer_id)
        object.__setattr__(
            self,
            "producer_code_sha256",
            _validate_sha256(self.producer_code_sha256, name="producer_code_sha256"),
        )
        object.__setattr__(
            self,
            "producer_config_sha256",
            _validate_sha256(self.producer_config_sha256, name="producer_config_sha256"),
        )
        object.__setattr__(
            self,
            "input_artifact_sha256s",
            MappingProxyType(normalized_inputs),
        )


@dataclass(frozen=True)
class CrossFittedVector:
    """One cross-fitted nuisance vector and its exact generation lineage."""

    name: str
    row_ids: tuple[int, ...]
    values: np.ndarray = field(repr=False)
    inner_fold_ids: tuple[Hashable, ...]
    fit_row_provenance: tuple[FitRowProvenance, ...] = field(repr=False)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name or _FORBIDDEN_NAME.search(name):
            raise ValueError("CrossFittedVector.name must be non-empty and truth-agnostic")
        rows = _row_id_tuple(self.row_ids, name=f"{name}.row_ids")
        values = _numeric_vector(self.values, name=f"{name}.values", length=len(rows))
        folds = _fold_id_tuple(
            self.inner_fold_ids,
            name=f"{name}.inner_fold_ids",
            length=len(rows),
        )
        provenance = tuple(self.fit_row_provenance)
        if len(provenance) != len(rows) or not all(
            isinstance(item, FitRowProvenance) for item in provenance
        ):
            raise TypeError(f"{name}.fit_row_provenance must contain one FitRowProvenance per row")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "inner_fold_ids", folds)
        object.__setattr__(self, "fit_row_provenance", provenance)

    def _positions(self) -> dict[int, int]:
        return {row_id: index for index, row_id in enumerate(self.row_ids)}

    def aligned_values(self, row_ids: Sequence[Any]) -> np.ndarray:
        requested = _row_id_tuple(row_ids, name=f"{self.name}.requested_row_ids")
        positions = self._positions()
        missing = [row_id for row_id in requested if row_id not in positions]
        if missing:
            raise ValueError(f"{self.name} is missing requested rows {missing[:3]}")
        return np.asarray([self.values[positions[row_id]] for row_id in requested], dtype=float)

    def aligned_fold_ids(self, row_ids: Sequence[Any]) -> tuple[Hashable, ...]:
        requested = _row_id_tuple(row_ids, name=f"{self.name}.requested_row_ids")
        positions = self._positions()
        missing = [row_id for row_id in requested if row_id not in positions]
        if missing:
            raise ValueError(f"{self.name} is missing requested rows {missing[:3]}")
        return tuple(self.inner_fold_ids[positions[row_id]] for row_id in requested)

    def aligned_provenance(self, row_ids: Sequence[Any]) -> tuple[FitRowProvenance, ...]:
        requested = _row_id_tuple(row_ids, name=f"{self.name}.requested_row_ids")
        positions = self._positions()
        missing = [row_id for row_id in requested if row_id not in positions]
        if missing:
            raise ValueError(f"{self.name} is missing requested rows {missing[:3]}")
        return tuple(self.fit_row_provenance[positions[row_id]] for row_id in requested)


@dataclass(frozen=True)
class CrossFittedNuisance:
    propensity: CrossFittedVector
    outcome_prediction: CrossFittedVector

    def __post_init__(self) -> None:
        if self.propensity.name != "propensity":
            raise ValueError("propensity vector must be named 'propensity'")
        if self.outcome_prediction.name != "outcome_prediction":
            raise ValueError("outcome vector must be named 'outcome_prediction'")
        if set(self.propensity.row_ids) != set(self.outcome_prediction.row_ids):
            raise ValueError("nuisance vectors must cover the same rows")
        if np.any(self.propensity.values <= 0.0) or np.any(
            self.propensity.values >= 1.0
        ):
            raise ValueError("cross-fitted propensity must be strictly inside (0, 1)")
        if np.any(self.outcome_prediction.values < 0.0) or np.any(
            self.outcome_prediction.values > 1.0
        ):
            raise ValueError("binary outcome_prediction must be inside [0, 1]")


@dataclass(frozen=True)
class FoldLocalSignal:
    """Matched train-inner-OOF and outer-heldout versions of one signal."""

    signal_name: str
    source_kind: str
    signal_role: str
    inner_oof: SignalBundle
    inner_fold_ids: tuple[Hashable, ...]
    outer_heldout: SignalBundle

    def __post_init__(self) -> None:
        name = str(self.signal_name).strip()
        if not name or _FORBIDDEN_NAME.search(name):
            raise ValueError("signal_name must be non-empty and truth-agnostic")
        kind = str(self.source_kind).strip().lower()
        if kind not in SUPPORTED_SIGNAL_KINDS:
            raise ValueError(f"source_kind must be one of {sorted(SUPPORTED_SIGNAL_KINDS)}")
        role = str(self.signal_role).strip().lower()
        if role not in SUPPORTED_SIGNAL_ROLES:
            raise ValueError(f"signal_role must be one of {sorted(SUPPORTED_SIGNAL_ROLES)}")
        if self.inner_oof.source_family != name or self.outer_heldout.source_family != name:
            raise ValueError("signal bundles must use signal_name as source_family")
        if self.inner_oof.prediction_scope != INNER_OOF_SCOPE:
            raise ValueError("inner signal bundle must have inner_oof scope")
        if self.outer_heldout.prediction_scope != OUTER_HELDOUT_SCOPE:
            raise ValueError("outer signal bundle must have outer_heldout scope")
        folds = _fold_id_tuple(
            self.inner_fold_ids,
            name=f"{name}.inner_fold_ids",
            length=len(self.inner_oof.row_ids),
        )
        object.__setattr__(self, "signal_name", name)
        object.__setattr__(self, "source_kind", kind)
        object.__setattr__(self, "signal_role", role)
        object.__setattr__(self, "inner_fold_ids", folds)

    def aligned_fold_ids(self, row_ids: Sequence[Any]) -> tuple[Hashable, ...]:
        requested = _row_id_tuple(row_ids, name=f"{self.signal_name}.requested_row_ids")
        positions = {row_id: index for index, row_id in enumerate(self.inner_oof.row_ids)}
        missing = [row_id for row_id in requested if row_id not in positions]
        if missing:
            raise ValueError(f"{self.signal_name} is missing requested rows {missing[:3]}")
        return tuple(self.inner_fold_ids[positions[row_id]] for row_id in requested)


@dataclass(frozen=True)
class OuterTrainNumericalDiagnosticView:
    """Train-only numerical inputs for diagnostics, ablations, and revision.

    The view owns copied, read-only arrays and has no field that can expose an
    outer-heldout value or heldout label.  Recursive upstream lineage remains
    attached so downstream logic can re-check source honesty.

    This full OOF view is suitable for descriptive diagnostics and
    precommitted outer-train fits.  It is *not* safe for adaptive untouched-gate
    acceptance: source values on the meta-fit rows can recursively depend on
    rows in the chosen gate.  Such acceptance requires separately generated
    per-gate nested meta-fit OOF matrices, which schema v2 does not contain.
    """

    outer_fold: int
    split_fingerprint: str
    artifact_sha256: str
    row_ids: tuple[int, ...]
    signal_names: tuple[str, ...]
    source_kinds: tuple[str, ...]
    signal_roles: tuple[str, ...]
    signal_matrix: np.ndarray = field(repr=False)
    propensity: np.ndarray = field(repr=False)
    outcome_prediction: np.ndarray = field(repr=False)
    joint_inner_fold_ids: tuple[int, ...]
    signal_fit_row_provenance: tuple[tuple[FitRowProvenance, ...], ...] = field(repr=False)
    propensity_fit_row_provenance: tuple[FitRowProvenance, ...] = field(repr=False)
    outcome_fit_row_provenance: tuple[FitRowProvenance, ...] = field(repr=False)

    def __post_init__(self) -> None:
        rows = _row_id_tuple(self.row_ids, name="diagnostic.row_ids")
        names = tuple(str(value).strip() for value in self.signal_names)
        kinds = tuple(str(value).strip().lower() for value in self.source_kinds)
        roles = tuple(str(value).strip().lower() for value in self.signal_roles)
        if not names or any(not value for value in names) or len(names) != len(set(names)):
            raise ValueError("diagnostic signal_names must be non-empty and unique")
        if len(kinds) != len(names) or set(kinds) - SUPPORTED_SIGNAL_KINDS:
            raise ValueError("diagnostic source_kinds do not match supported signals")
        if len(roles) != len(names) or set(roles) - SUPPORTED_SIGNAL_ROLES:
            raise ValueError("diagnostic signal_roles do not match supported roles")
        matrix = np.asarray(self.signal_matrix, dtype=float)
        if matrix.shape != (len(rows), len(names)) or not np.isfinite(matrix).all():
            raise ValueError("diagnostic signal_matrix has the wrong shape or non-finite values")
        matrix = matrix.copy()
        matrix.setflags(write=False)
        propensity = _numeric_vector(
            self.propensity, name="diagnostic.propensity", length=len(rows)
        )
        outcome_prediction = _numeric_vector(
            self.outcome_prediction,
            name="diagnostic.outcome_prediction",
            length=len(rows),
        )
        joint_folds = tuple(int(value) for value in self.joint_inner_fold_ids)
        if len(joint_folds) != len(rows) or len(set(joint_folds)) < 2:
            raise ValueError("diagnostic joint_inner_fold_ids are invalid")
        signal_provenance = tuple(
            tuple(source_lineage) for source_lineage in self.signal_fit_row_provenance
        )
        if len(signal_provenance) != len(names) or any(
            len(source_lineage) != len(rows) for source_lineage in signal_provenance
        ):
            raise ValueError("diagnostic signal provenance has the wrong shape")
        propensity_provenance = tuple(self.propensity_fit_row_provenance)
        outcome_provenance = tuple(self.outcome_fit_row_provenance)
        if len(propensity_provenance) != len(rows) or len(outcome_provenance) != len(rows):
            raise ValueError("diagnostic nuisance provenance has the wrong length")
        for lineage in (
            *propensity_provenance,
            *outcome_provenance,
            *(item for source in signal_provenance for item in source),
        ):
            if not isinstance(lineage, FitRowProvenance):
                raise TypeError("diagnostic provenance entries must be FitRowProvenance")
        rows_by_fold: dict[int, set[int]] = {}
        for row_id, fold_id in zip(rows, joint_folds):
            rows_by_fold.setdefault(fold_id, set()).add(row_id)
        row_set = set(rows)
        components = (
            ("propensity", propensity_provenance),
            ("outcome_prediction", outcome_provenance),
            *((signal_name, lineage) for signal_name, lineage in zip(names, signal_provenance)),
        )
        for component_name, component_lineage in components:
            for row_id, fold_id, lineage in zip(rows, joint_folds, component_lineage):
                recursive = set(lineage.recursive_fit_row_ids())
                if not recursive or not recursive <= row_set:
                    raise ValueError(
                        f"diagnostic {component_name} row {row_id} provenance is not "
                        "confined to outer train"
                    )
                overlap = recursive & rows_by_fold[fold_id]
                if overlap:
                    raise ValueError(
                        f"diagnostic {component_name} row {row_id} provenance overlaps "
                        f"its joint heldout fold: {sorted(overlap)[:3]}"
                    )
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "signal_names", names)
        object.__setattr__(self, "source_kinds", kinds)
        object.__setattr__(self, "signal_roles", roles)
        object.__setattr__(self, "signal_matrix", matrix)
        object.__setattr__(self, "propensity", propensity)
        object.__setattr__(self, "outcome_prediction", outcome_prediction)
        object.__setattr__(self, "joint_inner_fold_ids", joint_folds)
        object.__setattr__(self, "signal_fit_row_provenance", signal_provenance)
        object.__setattr__(self, "propensity_fit_row_provenance", propensity_provenance)
        object.__setattr__(self, "outcome_fit_row_provenance", outcome_provenance)

    def signal_column(self, signal_name: str) -> np.ndarray:
        """Return a read-only train-OOF column by authenticated source name."""

        normalized = str(signal_name).strip()
        if normalized not in self.signal_names:
            raise KeyError(normalized)
        return self.signal_matrix[:, self.signal_names.index(normalized)]

    @property
    def adaptive_untouched_gate_safe(self) -> bool:
        """Always false for a full OOF view without per-gate nested meta-fit data."""

        return False

    @property
    def usage_scope(self) -> str:
        return "descriptive_or_precommitted_outer_train_only"

    def require_adaptive_untouched_gate_safety(self) -> None:
        raise RuntimeError(
            "Full OOF numerical diagnostics are not safe for adaptive untouched-gate "
            "scoring: meta-fit source values may recursively depend on gate rows. "
            "Generate per-gate nested meta-fit OOF signals and nuisances first."
        )


@dataclass(frozen=True)
class FoldLocalNumericalSignals:
    """A fully authenticated, provenance-checked outer-fold signal package."""

    outer_fold: int
    split_fingerprint: str
    outer_train_row_ids: tuple[int, ...]
    outer_heldout_row_ids: tuple[int, ...]
    artifact_sha256: str
    producer_audit: NumericalSignalProducerAudit
    nuisance: CrossFittedNuisance
    signals: tuple[FoldLocalSignal, ...]
    manifest_sha256: str = "0" * 64
    _artifact_path: Path | None = field(default=None, init=False, repr=False, compare=False)
    _manifest_path: Path | None = field(default=None, init=False, repr=False, compare=False)
    _material_records: tuple[_AuthenticatedMaterialRecord, ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    _authentication_capability: Any = field(
        default=None, init=False, repr=False, compare=False
    )
    _authenticated_content_sha256: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.outer_fold, (bool, np.bool_)) or not isinstance(
            self.outer_fold, (int, np.integer)
        ):
            raise TypeError("outer_fold must be a positive integer")
        outer_fold = int(self.outer_fold)
        if outer_fold < 1:
            raise ValueError("outer_fold must be a positive integer")
        train_rows = _row_id_tuple(self.outer_train_row_ids, name="outer_train_row_ids")
        heldout_rows = _row_id_tuple(self.outer_heldout_row_ids, name="outer_heldout_row_ids")
        if set(train_rows) & set(heldout_rows):
            raise ValueError("outer train and heldout row IDs must be disjoint")
        signals = tuple(self.signals)
        if not signals or not all(isinstance(item, FoldLocalSignal) for item in signals):
            raise TypeError("signals must contain at least one FoldLocalSignal")
        names = [signal.signal_name for signal in signals]
        if len(names) != len(set(names)):
            raise ValueError("signal_name values must be unique within an outer fold")
        train_set = set(train_rows)
        heldout_set = set(heldout_rows)

        for vector in (self.nuisance.propensity, self.nuisance.outcome_prediction):
            if set(vector.row_ids) != train_set:
                raise ValueError(f"{vector.name} rows do not exactly match outer train rows")
            _validate_inner_oof_lineage(
                component_name=vector.name,
                row_ids=vector.row_ids,
                fold_ids=vector.inner_fold_ids,
                provenance=vector.fit_row_provenance,
                outer_train_rows=train_set,
            )

        for signal in signals:
            if set(signal.inner_oof.row_ids) != train_set:
                raise ValueError(
                    f"Signal {signal.signal_name!r} inner rows do not exactly match outer train"
                )
            if set(signal.outer_heldout.row_ids) != heldout_set:
                raise ValueError(
                    f"Signal {signal.signal_name!r} outer rows do not exactly match heldout"
                )
            _validate_inner_oof_lineage(
                component_name=signal.signal_name,
                row_ids=signal.inner_oof.row_ids,
                fold_ids=signal.inner_fold_ids,
                provenance=signal.inner_oof.fit_row_provenance,
                outer_train_rows=train_set,
            )
            _validate_outer_lineage(
                component_name=signal.signal_name,
                provenance=signal.outer_heldout.fit_row_provenance,
                outer_train_rows=train_set,
                outer_heldout_rows=heldout_set,
            )

        object.__setattr__(self, "outer_fold", outer_fold)
        object.__setattr__(
            self,
            "split_fingerprint",
            _validate_sha256(self.split_fingerprint, name="split_fingerprint"),
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _validate_sha256(self.artifact_sha256, name="artifact_sha256"),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            _validate_sha256(self.manifest_sha256, name="manifest_sha256"),
        )
        object.__setattr__(self, "outer_train_row_ids", train_rows)
        object.__setattr__(self, "outer_heldout_row_ids", heldout_rows)
        object.__setattr__(self, "signals", signals)
        # This is the partition passed to FoldHonestRStack.  It is the
        # intersection of every producer's own honest generation partition.
        # Consequently every component excludes every row in its joint
        # heldout cell, even when individual producers used different folds.
        _validate_joint_meta_partition(self)

    def verify_authenticated_content(self) -> None:
        """Re-authenticate values, identities, manifest, and material bytes.

        This check intentionally happens at every estimator boundary.  Frozen
        dataclasses and read-only arrays are convenience guards, not a security
        boundary: ``object.__setattr__`` and reconstructed instances can still
        retain plausible self-attested identity strings.
        """

        if self._authentication_capability is not _AUTHENTICATED_LOADER_CAPABILITY:
            raise ValueError(
                "numerical signal package is unauthenticated; use the manifest loader"
            )
        if self._artifact_path is None or self._manifest_path is None:
            raise ValueError("authenticated package is missing loader-owned file bindings")
        artifact_raw = self._artifact_path.read_bytes()
        if _sha256_bytes(artifact_raw) != self.artifact_sha256:
            raise ValueError("authenticated numerical signal artifact changed on disk")
        current_content_sha256 = _sha256_bytes(
            _canonical_json_bytes(_fold_signal_payload(self))
        )
        if (
            self._authenticated_content_sha256 is None
            or current_content_sha256 != self._authenticated_content_sha256
        ):
            raise ValueError("in-memory numerical signal content differs from its artifact")
        manifest_raw = self._manifest_path.read_bytes()
        if _sha256_bytes(manifest_raw) != self.manifest_sha256:
            raise ValueError("authenticated numerical signal manifest changed on disk")
        manifest = _load_closed_json(manifest_raw, name="numerical signal manifest")
        if not isinstance(manifest, Mapping):
            raise ValueError("authenticated numerical signal manifest is not an object")
        if dict(manifest.get("arrays", {})) != dict(
            _array_authentication_records(self)
        ):
            raise ValueError("in-memory array bytes differ from authenticated manifest")
        identity = manifest.get("identity")
        expected_identity = {
            "outer_fold": self.outer_fold,
            "split_fingerprint": self.split_fingerprint,
            "outer_train_row_ids": list(self.outer_train_row_ids),
            "outer_heldout_row_ids": list(self.outer_heldout_row_ids),
            "nuisance_inner_fold_ids": {
                "propensity": list(self.nuisance.propensity.inner_fold_ids),
                "outcome_prediction": list(
                    self.nuisance.outcome_prediction.inner_fold_ids
                ),
            },
            "ordered_signals": [
                {
                    "signal_name": signal.signal_name,
                    "source_kind": signal.source_kind,
                    "signal_role": signal.signal_role,
                    "inner_fold_ids": list(signal.inner_fold_ids),
                }
                for signal in self.signals
            ],
        }
        if identity != expected_identity:
            raise ValueError("in-memory identity differs from authenticated manifest")
        manifest_materials = manifest.get("materials") if isinstance(manifest, Mapping) else None
        if not isinstance(manifest_materials, list):
            raise ValueError("authenticated manifest no longer has material bindings")
        bound_keys = {
            (record.category, record.name, str(record.path), record.sha256, record.size_bytes)
            for record in self._material_records
        }
        manifest_keys: set[tuple[str, str, str, str, int]] = set()
        for raw_material in manifest_materials:
            if not isinstance(raw_material, Mapping):
                raise ValueError("authenticated manifest has an invalid material binding")
            material = _AuthenticatedMaterialRecord(
                category=str(raw_material.get("category", "")),
                name=str(raw_material.get("name", "")),
                path=Path(str(raw_material.get("path", ""))).resolve(strict=True),
                sha256=_validate_sha256(
                    raw_material.get("sha256"), name="manifest material sha256"
                ),
                size_bytes=int(raw_material.get("size_bytes", -1)),
            )
            manifest_keys.add(
                (
                    material.category,
                    material.name,
                    str(material.path),
                    material.sha256,
                    material.size_bytes,
                )
            )
            try:
                material_raw = material.path.read_bytes()
            except OSError as exc:
                raise ValueError(
                    f"authenticated material is unavailable: {material.path}"
                ) from exc
            if len(material_raw) != material.size_bytes or _sha256_bytes(
                material_raw
            ) != material.sha256:
                raise ValueError(
                    "authenticated producer material changed on disk: "
                    f"{material.category}:{material.name}"
                )
        if manifest_keys != bound_keys:
            raise ValueError("in-memory material bindings differ from authenticated manifest")

    def joint_inner_fold_ids(self) -> tuple[int, ...]:
        train_rows = self.outer_train_row_ids
        component_folds = [
            self.nuisance.propensity.aligned_fold_ids(train_rows),
            self.nuisance.outcome_prediction.aligned_fold_ids(train_rows),
            *(signal.aligned_fold_ids(train_rows) for signal in self.signals),
        ]
        signatures = [
            tuple(component[index] for component in component_folds)
            for index in range(len(train_rows))
        ]
        ids_by_signature: dict[tuple[Hashable, ...], int] = {}
        result: list[int] = []
        for signature in signatures:
            if signature not in ids_by_signature:
                ids_by_signature[signature] = len(ids_by_signature) + 1
            result.append(ids_by_signature[signature])
        if len(ids_by_signature) < 2:
            raise ValueError("joint inner provenance must yield at least two safe folds")
        return tuple(result)

    def outer_train_diagnostic_view(self) -> OuterTrainNumericalDiagnosticView:
        """Copy only authenticated train-OOF values into a diagnostic view."""

        self.verify_authenticated_content()
        rows = self.outer_train_row_ids
        return OuterTrainNumericalDiagnosticView(
            outer_fold=self.outer_fold,
            split_fingerprint=self.split_fingerprint,
            artifact_sha256=self.artifact_sha256,
            row_ids=rows,
            signal_names=tuple(signal.signal_name for signal in self.signals),
            source_kinds=tuple(signal.source_kind for signal in self.signals),
            signal_roles=tuple(signal.signal_role for signal in self.signals),
            signal_matrix=np.column_stack(
                [signal.inner_oof.aligned_predictions(rows) for signal in self.signals]
            ),
            propensity=self.nuisance.propensity.aligned_values(rows),
            outcome_prediction=self.nuisance.outcome_prediction.aligned_values(rows),
            joint_inner_fold_ids=self.joint_inner_fold_ids(),
            signal_fit_row_provenance=tuple(
                signal.inner_oof.aligned_provenance(rows) for signal in self.signals
            ),
            propensity_fit_row_provenance=self.nuisance.propensity.aligned_provenance(rows),
            outcome_fit_row_provenance=(self.nuisance.outcome_prediction.aligned_provenance(rows)),
        )

    def adaptive_gate_diagnostic_views(self) -> None:
        """Fail closed until the schema gains per-gate nested meta-fit matrices."""

        raise NotImplementedError(
            "fold_honest_numerical_signals_v2 has no per-meta-fold nested diagnostic "
            "views. Its full OOF matrix must not be used for adaptive untouched-gate "
            "acceptance."
        )


def _validate_inner_oof_lineage(
    *,
    component_name: str,
    row_ids: Sequence[int],
    fold_ids: Sequence[Hashable],
    provenance: Sequence[FitRowProvenance],
    outer_train_rows: set[int],
) -> None:
    rows_by_fold: dict[Hashable, set[int]] = {}
    for row_id, fold_id in zip(row_ids, fold_ids):
        rows_by_fold.setdefault(fold_id, set()).add(int(row_id))
    for row_id, fold_id, lineage in zip(row_ids, fold_ids, provenance):
        recursive = set(lineage.recursive_fit_row_ids())
        if not recursive:
            raise ValueError(f"{component_name} row {row_id} has empty fit provenance")
        outside = recursive - outer_train_rows
        if outside:
            raise ValueError(
                f"{component_name} row {row_id} provenance leaves outer train: "
                f"{sorted(outside)[:3]}"
            )
        overlap = recursive & rows_by_fold[fold_id]
        if overlap:
            raise ValueError(
                f"{component_name} row {row_id} provenance overlaps its exact inner "
                f"heldout fold: {sorted(overlap)[:3]}"
            )


def _validate_outer_lineage(
    *,
    component_name: str,
    provenance: Sequence[FitRowProvenance],
    outer_train_rows: set[int],
    outer_heldout_rows: set[int],
) -> None:
    for lineage in provenance:
        recursive = set(lineage.recursive_fit_row_ids())
        if not recursive:
            raise ValueError(f"{component_name} outer prediction has empty fit provenance")
        outside = recursive - outer_train_rows
        if outside:
            heldout_overlap = outside & outer_heldout_rows
            if heldout_overlap:
                raise ValueError(
                    f"{component_name} outer provenance consumes heldout rows: "
                    f"{sorted(heldout_overlap)[:3]}"
                )
            raise ValueError(
                f"{component_name} outer provenance leaves outer train: " f"{sorted(outside)[:3]}"
            )


def _validate_joint_meta_partition(package: FoldLocalNumericalSignals) -> None:
    train_rows = package.outer_train_row_ids
    joint_folds = package.joint_inner_fold_ids()
    rows_by_fold: dict[int, set[int]] = {}
    for row_id, fold_id in zip(train_rows, joint_folds):
        rows_by_fold.setdefault(fold_id, set()).add(row_id)

    components: list[tuple[str, tuple[FitRowProvenance, ...]]] = [
        (
            package.nuisance.propensity.name,
            package.nuisance.propensity.aligned_provenance(train_rows),
        ),
        (
            package.nuisance.outcome_prediction.name,
            package.nuisance.outcome_prediction.aligned_provenance(train_rows),
        ),
        *(
            (
                signal.signal_name,
                signal.inner_oof.aligned_provenance(train_rows),
            )
            for signal in package.signals
        ),
    ]
    for component_name, provenance in components:
        for row_id, fold_id, lineage in zip(train_rows, joint_folds, provenance):
            overlap = set(lineage.recursive_fit_row_ids()) & rows_by_fold[fold_id]
            if overlap:
                raise ValueError(
                    f"{component_name} row {row_id} provenance overlaps safe joint "
                    f"inner fold: {sorted(overlap)[:3]}"
                )


class FoldHonestNumericalSignalFusion:
    """Fold-local R-loss fusion over authenticated numerical signals."""

    def __init__(
        self,
        *,
        ridge_alphas: Sequence[float] = (1.0,),
        nonnegative: bool = False,
    ) -> None:
        self.ridge_alphas = tuple(ridge_alphas)
        if not isinstance(nonnegative, bool):
            raise TypeError("nonnegative must be a boolean")
        self.nonnegative = nonnegative
        self._stack = FoldHonestRStack(
            ridge_alphas=self.ridge_alphas,
            nonnegative=self.nonnegative,
        )

    def fit(
        self,
        package: FoldLocalNumericalSignals,
        *,
        row_ids: Sequence[Any],
        treatment: Sequence[float],
        outcome: Sequence[float],
    ) -> "FoldHonestNumericalSignalFusion":
        if not isinstance(package, FoldLocalNumericalSignals):
            raise TypeError("package must be a FoldLocalNumericalSignals instance")
        package.verify_authenticated_content()
        invalid_roles = [
            f"{signal.signal_name}:{signal.signal_role}"
            for signal in package.signals
            if signal.signal_role != CALIBRATED_TAU_ROLE
        ]
        if invalid_roles:
            raise ValueError(
                "R-stack accepts calibrated tau signals only; raw features must stay "
                f"in the feature-bank path: {invalid_roles[:3]}"
            )
        requested_rows = _row_id_tuple(row_ids, name="row_ids")
        if requested_rows != package.outer_train_row_ids:
            raise ValueError(
                "R-stack fit rows/order do not exactly match authenticated outer train"
            )
        propensity = package.nuisance.propensity.aligned_values(requested_rows)
        outcome_prediction = package.nuisance.outcome_prediction.aligned_values(requested_rows)
        joint_folds = package.joint_inner_fold_ids()
        self._stack.fit(
            row_ids=requested_rows,
            treatment=treatment,
            outcome=outcome,
            propensity=propensity,
            outcome_prediction=outcome_prediction,
            inner_fold_ids=joint_folds,
            signals=[signal.inner_oof for signal in package.signals],
        )
        self.outer_fold_ = package.outer_fold
        self.split_fingerprint_ = package.split_fingerprint
        self.input_artifact_sha256_ = package.artifact_sha256
        self.input_manifest_sha256_ = package.manifest_sha256
        self.outer_train_row_ids_ = package.outer_train_row_ids
        self.outer_heldout_row_ids_ = package.outer_heldout_row_ids
        self.source_kinds_ = tuple(signal.source_kind for signal in package.signals)
        self.signal_roles_ = tuple(signal.signal_role for signal in package.signals)
        self.joint_inner_fold_ids_ = joint_folds
        self.joint_inner_fold_count_ = len(set(joint_folds))
        return self

    def predict(
        self,
        package: FoldLocalNumericalSignals,
    ) -> np.ndarray:
        self._validate_prediction_package(package)
        return self._stack.predict(
            row_ids=package.outer_heldout_row_ids,
            signals=[signal.outer_heldout for signal in package.signals],
        )

    def predict_bundle(
        self,
        package: FoldLocalNumericalSignals,
        *,
        source_family: str = "provenance_checked_numerical_r_stack",
    ) -> SignalBundle:
        self._validate_prediction_package(package)
        return self._stack.predict_bundle(
            row_ids=package.outer_heldout_row_ids,
            signals=[signal.outer_heldout for signal in package.signals],
            source_family=source_family,
        )

    def audit_record(self) -> Mapping[str, Any]:
        self._require_fitted()
        return {
            "schema_version": NUMERICAL_SIGNAL_FUSION_AUDIT_SCHEMA_VERSION,
            "outer_fold": self.outer_fold_,
            "split_fingerprint": self.split_fingerprint_,
            "input_artifact_sha256": self.input_artifact_sha256_,
            "input_manifest_sha256": self.input_manifest_sha256_,
            "outer_train_row_fingerprint": row_set_fingerprint(self.outer_train_row_ids_),
            "outer_heldout_row_fingerprint": row_set_fingerprint(self.outer_heldout_row_ids_),
            "source_names": list(self._stack.source_families_),
            "source_kinds": list(self.source_kinds_),
            "signal_roles": list(self.signal_roles_),
            "source_weights": dict(self._stack.source_weights_),
            "standardized_source_weights": {
                family: float(weight)
                for family, weight in zip(
                    self._stack.source_families_, self._stack.standardized_weights_
                )
            },
            "source_train_means": {
                family: float(value)
                for family, value in zip(
                    self._stack.source_families_, self._stack.source_means_
                )
            },
            "source_train_scales": {
                family: float(value)
                for family, value in zip(
                    self._stack.source_families_, self._stack.source_scales_
                )
            },
            "constant_effect": self._stack.constant_effect_,
            "regularization_strategy": self._stack.regularization_strategy_,
            "precommitted_ridge_alpha": self._stack.precommitted_alpha,
            "outer_train_r_loss": self._stack.training_r_loss_,
            "safe_joint_inner_fold_count": self.joint_inner_fold_count_,
            "posthoc_targets_consumed": False,
            "outer_heldout_labels_consumed": False,
        }

    def _validate_prediction_package(self, package: FoldLocalNumericalSignals) -> None:
        self._require_fitted()
        if not isinstance(package, FoldLocalNumericalSignals):
            raise TypeError("package must be a FoldLocalNumericalSignals instance")
        package.verify_authenticated_content()
        identity = (
            package.outer_fold,
            package.split_fingerprint,
            package.artifact_sha256,
            package.manifest_sha256,
            package.outer_train_row_ids,
            package.outer_heldout_row_ids,
            tuple(signal.signal_name for signal in package.signals),
            tuple(signal.source_kind for signal in package.signals),
            tuple(signal.signal_role for signal in package.signals),
        )
        expected = (
            self.outer_fold_,
            self.split_fingerprint_,
            self.input_artifact_sha256_,
            self.input_manifest_sha256_,
            self.outer_train_row_ids_,
            self.outer_heldout_row_ids_,
            self._stack.source_families_,
            self.source_kinds_,
            self.signal_roles_,
        )
        if identity != expected:
            raise ValueError("prediction package does not match the authenticated fit package")

    def _require_fitted(self) -> None:
        if not hasattr(self, "input_artifact_sha256_"):
            raise RuntimeError("FoldHonestNumericalSignalFusion must be fit before use")


def make_inner_oof_provenance(
    *,
    row_ids: Sequence[Any],
    inner_fold_ids: Sequence[Any],
    fit_row_ids_by_fold: Mapping[Hashable, Sequence[Any]],
    upstream_by_fold: Mapping[Hashable, Sequence[FitRowProvenance]] | None = None,
) -> tuple[FitRowProvenance, ...]:
    """Build exact per-row lineage for a producer's inner-OOF vector.

    This helper is the intended Stage-1 producer hook.  The producer must pass
    the row IDs it actually used for every fitted fold model; the helper does
    not infer them from a role label or from a deterministic splitter seed.
    """

    rows = _row_id_tuple(row_ids, name="row_ids")
    folds = _fold_id_tuple(inner_fold_ids, name="inner_fold_ids", length=len(rows))
    expected_folds = set(folds)
    if set(fit_row_ids_by_fold) != expected_folds:
        raise ValueError("fit_row_ids_by_fold keys must exactly match inner_fold_ids")
    upstream_mapping = {} if upstream_by_fold is None else dict(upstream_by_fold)
    if set(upstream_mapping) - expected_folds:
        raise ValueError("upstream_by_fold contains an unknown inner fold")
    rows_by_fold: dict[Hashable, set[int]] = {}
    for row_id, fold_id in zip(rows, folds):
        rows_by_fold.setdefault(fold_id, set()).add(row_id)
    outer_train = set(rows)
    lineage_by_fold: dict[Hashable, FitRowProvenance] = {}
    for fold_id in dict.fromkeys(folds):
        fit_rows = _row_id_tuple(
            fit_row_ids_by_fold[fold_id],
            name=f"fit_row_ids_by_fold[{fold_id!r}]",
        )
        outside = set(fit_rows) - outer_train
        if outside:
            raise ValueError(
                f"inner fold {fold_id!r} fit rows leave outer train: {sorted(outside)[:3]}"
            )
        overlap = set(fit_rows) & rows_by_fold[fold_id]
        if overlap:
            raise ValueError(
                f"inner fold {fold_id!r} fit rows overlap heldout rows: " f"{sorted(overlap)[:3]}"
            )
        upstream = tuple(upstream_mapping.get(fold_id, ()))
        if not all(isinstance(item, FitRowProvenance) for item in upstream):
            raise TypeError("upstream_by_fold values must contain FitRowProvenance instances")
        lineage_by_fold[fold_id] = FitRowProvenance(
            fit_row_ids=frozenset(fit_rows),
            upstream=upstream,
        )
    return tuple(lineage_by_fold[fold_id] for fold_id in folds)


def make_outer_train_provenance(
    *,
    heldout_row_ids: Sequence[Any],
    outer_train_fit_row_ids: Sequence[Any],
    upstream: Sequence[FitRowProvenance] = (),
) -> tuple[FitRowProvenance, ...]:
    """Build shared outer-heldout lineage from exact outer-train fit rows."""

    heldout_rows = _row_id_tuple(heldout_row_ids, name="heldout_row_ids")
    fit_rows = _row_id_tuple(outer_train_fit_row_ids, name="outer_train_fit_row_ids")
    if set(heldout_rows) & set(fit_rows):
        raise ValueError("outer-heldout rows overlap outer-train fit rows")
    upstream_nodes = tuple(upstream)
    if not all(isinstance(item, FitRowProvenance) for item in upstream_nodes):
        raise TypeError("upstream must contain FitRowProvenance instances")
    lineage = FitRowProvenance(
        fit_row_ids=frozenset(fit_rows),
        upstream=upstream_nodes,
    )
    return tuple(lineage for _row_id in heldout_rows)


@dataclass(frozen=True)
class WrittenFoldNumericalSignalArtifact:
    path: Path
    sha256: str
    manifest_path: Path
    manifest_sha256: str
    package: FoldLocalNumericalSignals


def write_fold_numerical_signal_artifact(
    path: Path | str,
    *,
    outer_fold: int,
    split_fingerprint: str,
    outer_train_row_ids: Sequence[Any],
    outer_heldout_row_ids: Sequence[Any],
    producer_audit: NumericalSignalProducerAudit,
    nuisance: CrossFittedNuisance,
    signals: Sequence[FoldLocalSignal],
    authenticated_materials: Sequence[AuthenticatedMaterialFile],
    random_seed: int,
    library_versions: Mapping[str, str],
    manifest_path: Path | str | None = None,
) -> WrittenFoldNumericalSignalArtifact:
    """Validate, serialize, and re-authenticate a strict producer sidecar.

    Existing identical bytes are accepted for resumability.  An existing file
    with different bytes is rejected rather than overwritten.
    """

    train_rows = _row_id_tuple(outer_train_row_ids, name="outer_train_row_ids")
    heldout_rows = _row_id_tuple(outer_heldout_row_ids, name="outer_heldout_row_ids")
    validated = FoldLocalNumericalSignals(
        outer_fold=outer_fold,
        split_fingerprint=split_fingerprint,
        outer_train_row_ids=train_rows,
        outer_heldout_row_ids=heldout_rows,
        artifact_sha256="0" * 64,
        producer_audit=producer_audit,
        nuisance=nuisance,
        signals=tuple(signals),
    )
    materials = _validate_authenticated_materials(
        authenticated_materials,
        producer_audit=producer_audit,
    )
    if isinstance(random_seed, (bool, np.bool_)) or not isinstance(
        random_seed, (int, np.integer)
    ):
        raise TypeError("random_seed must be an integer")
    versions = _normalize_library_versions(library_versions)
    payload = _fold_signal_payload(validated)
    encoded = _canonical_json_bytes(payload)
    requested = Path(path).resolve()
    if requested.exists():
        if requested.read_bytes() != encoded:
            raise FileExistsError(
                f"Refusing to overwrite different numerical signal artifact: {requested}"
            )
    else:
        if not requested.parent.exists():
            raise FileNotFoundError(
                f"Numerical signal artifact parent does not exist: {requested.parent}"
            )
        with requested.open("xb") as handle:
            handle.write(encoded)
    digest = _sha256_bytes(encoded)
    requested_manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else requested.with_name(requested.name + ".manifest.json")
    )
    manifest_payload = _fold_signal_manifest_payload(
        package=validated,
        artifact_path=requested,
        artifact_sha256=digest,
        material_records=materials,
        random_seed=int(random_seed),
        library_versions=versions,
    )
    manifest_encoded = _canonical_json_bytes(manifest_payload)
    if requested_manifest.exists():
        if requested_manifest.read_bytes() != manifest_encoded:
            raise FileExistsError(
                "Refusing to overwrite different numerical signal manifest: "
                f"{requested_manifest}"
            )
    else:
        if not requested_manifest.parent.exists():
            raise FileNotFoundError(
                "Numerical signal manifest parent does not exist: "
                f"{requested_manifest.parent}"
            )
        with requested_manifest.open("xb") as handle:
            handle.write(manifest_encoded)
    manifest_digest = _sha256_bytes(manifest_encoded)
    loaded = load_fold_numerical_signal_artifact(
        requested,
        manifest_path=requested_manifest,
        expected_manifest_sha256=manifest_digest,
        expected_sha256=digest,
        expected_outer_fold=outer_fold,
        expected_split_fingerprint=split_fingerprint,
        expected_outer_train_row_ids=train_rows,
        expected_outer_heldout_row_ids=heldout_rows,
        required_source_kinds=[signal.source_kind for signal in signals],
    )
    return WrittenFoldNumericalSignalArtifact(
        path=requested,
        sha256=digest,
        manifest_path=requested_manifest,
        manifest_sha256=manifest_digest,
        package=loaded,
    )


def _normalize_library_versions(values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("library_versions must be a non-empty mapping")
    result: dict[str, str] = {}
    for raw_name, raw_version in values.items():
        name = str(raw_name).strip()
        version = str(raw_version).strip()
        if not name or not version or name in result:
            raise ValueError("library_versions must have unique non-empty strings")
        result[name] = version
    return MappingProxyType(result)


def _validate_authenticated_materials(
    values: Sequence[AuthenticatedMaterialFile],
    *,
    producer_audit: NumericalSignalProducerAudit,
) -> tuple[_AuthenticatedMaterialRecord, ...]:
    materials = tuple(values)
    if not materials or not all(
        isinstance(value, AuthenticatedMaterialFile) for value in materials
    ):
        raise TypeError(
            "authenticated_materials must contain AuthenticatedMaterialFile instances"
        )
    keys = [(value.category, value.name) for value in materials]
    if len(keys) != len(set(keys)):
        raise ValueError("authenticated material category/name pairs must be unique")
    categories = {value.category for value in materials}
    missing_categories = sorted(REQUIRED_MATERIAL_CATEGORIES - categories)
    if missing_categories:
        raise ValueError(
            f"authenticated materials are missing categories: {missing_categories}"
        )
    records: list[_AuthenticatedMaterialRecord] = []
    for material in materials:
        raw = material.path.read_bytes()
        records.append(
            _AuthenticatedMaterialRecord(
                category=material.category,
                name=material.name,
                path=material.path,
                sha256=_sha256_bytes(raw),
                size_bytes=len(raw),
            )
        )
    by_category: dict[str, list[_AuthenticatedMaterialRecord]] = {}
    for record in records:
        by_category.setdefault(record.category, []).append(record)
    producer_code = by_category[PRODUCER_CODE_MATERIAL]
    producer_config = by_category[PRODUCER_CONFIG_MATERIAL]
    if len(producer_code) != 1 or len(producer_config) != 1:
        raise ValueError("exactly one producer_code and producer_config material are required")
    if producer_code[0].sha256 != producer_audit.producer_code_sha256:
        raise ValueError("producer code audit hash does not match authenticated bytes")
    if producer_config[0].sha256 != producer_audit.producer_config_sha256:
        raise ValueError("producer config audit hash does not match authenticated bytes")
    input_records = {record.name: record.sha256 for record in by_category[INPUT_MATERIAL]}
    if input_records != dict(producer_audit.input_artifact_sha256s):
        raise ValueError("input audit hashes do not match authenticated input bytes")
    return tuple(records)


def _array_authentication_records(
    package: FoldLocalNumericalSignals,
) -> Mapping[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {
        "nuisance/propensity": _array_authentication_record(
            package.nuisance.propensity.values
        ),
        "nuisance/outcome_prediction": _array_authentication_record(
            package.nuisance.outcome_prediction.values
        ),
    }
    for index, signal in enumerate(package.signals):
        prefix = f"signals/{index}/{signal.signal_name}"
        records[f"{prefix}/inner_oof"] = _array_authentication_record(
            signal.inner_oof.tau_predictions
        )
        records[f"{prefix}/outer_heldout"] = _array_authentication_record(
            signal.outer_heldout.tau_predictions
        )
    return records


def _fold_signal_manifest_payload(
    *,
    package: FoldLocalNumericalSignals,
    artifact_path: Path,
    artifact_sha256: str,
    material_records: Sequence[_AuthenticatedMaterialRecord],
    random_seed: int,
    library_versions: Mapping[str, str],
) -> Mapping[str, Any]:
    return {
        "schema_version": FOLD_NUMERICAL_SIGNAL_MANIFEST_SCHEMA_VERSION,
        "signal_artifact": {
            "path": str(artifact_path),
            "sha256": artifact_sha256,
            "size_bytes": artifact_path.stat().st_size,
        },
        "identity": {
            "outer_fold": package.outer_fold,
            "split_fingerprint": package.split_fingerprint,
            "outer_train_row_ids": list(package.outer_train_row_ids),
            "outer_heldout_row_ids": list(package.outer_heldout_row_ids),
            "nuisance_inner_fold_ids": {
                "propensity": list(package.nuisance.propensity.inner_fold_ids),
                "outcome_prediction": list(
                    package.nuisance.outcome_prediction.inner_fold_ids
                ),
            },
            "ordered_signals": [
                {
                    "signal_name": signal.signal_name,
                    "source_kind": signal.source_kind,
                    "signal_role": signal.signal_role,
                    "inner_fold_ids": list(signal.inner_fold_ids),
                }
                for signal in package.signals
            ],
        },
        "arrays": _array_authentication_records(package),
        "materials": [
            {
                "category": record.category,
                "name": record.name,
                "path": str(record.path),
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
            }
            for record in material_records
        ],
        "runtime": {
            "random_seed": random_seed,
            "library_versions": dict(library_versions),
        },
        "honesty": {
            "posthoc_targets_consumed": False,
            "outer_heldout_labels_consumed": False,
            "dataset_specific_truth_consumed": False,
            "nested_fit_row_lineage_required": True,
        },
    }


def _fold_signal_payload(package: FoldLocalNumericalSignals) -> Mapping[str, Any]:
    lineage_id_by_key: dict[tuple[Any, ...], str] = {}
    lineage_payloads: dict[str, Mapping[str, Any]] = {}
    active: set[int] = set()

    def register_lineage(lineage: FitRowProvenance) -> str:
        identity = id(lineage)
        if identity in active:
            raise ValueError("Fit-row provenance contains a cycle")
        active.add(identity)
        upstream_ids = tuple(register_lineage(item) for item in lineage.upstream)
        active.remove(identity)
        fit_rows = tuple(
            sorted(_canonical_row_id(value, name="fit_row_ids") for value in lineage.fit_row_ids)
        )
        key: tuple[Any, ...] = (fit_rows, upstream_ids)
        existing = lineage_id_by_key.get(key)
        if existing is not None:
            return existing
        key_payload = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)
        lineage_id = f"lineage_{hashlib.sha256(key_payload.encode('utf-8')).hexdigest()}"
        if lineage_id in lineage_payloads:
            raise RuntimeError("SHA-256 collision while serializing provenance")
        lineage_id_by_key[key] = lineage_id
        lineage_payloads[lineage_id] = {
            "fit_row_ids": list(fit_rows),
            "upstream_lineage_ids": list(upstream_ids),
        }
        return lineage_id

    def lineage_ids(values: Sequence[FitRowProvenance]) -> list[str]:
        return [register_lineage(value) for value in values]

    def inner_vector_payload(vector: CrossFittedVector) -> Mapping[str, Any]:
        return {
            "row_ids": list(vector.row_ids),
            "values": vector.values.tolist(),
            "inner_fold_ids": list(vector.inner_fold_ids),
            "lineage_ids": lineage_ids(vector.fit_row_provenance),
        }

    signal_payloads = []
    for signal in package.signals:
        signal_payloads.append(
            {
                "signal_name": signal.signal_name,
                "source_kind": signal.source_kind,
                "signal_role": signal.signal_role,
                "inner_oof": {
                    "row_ids": list(signal.inner_oof.row_ids),
                    "values": signal.inner_oof.tau_predictions.tolist(),
                    "inner_fold_ids": list(signal.inner_fold_ids),
                    "lineage_ids": lineage_ids(signal.inner_oof.fit_row_provenance),
                },
                "outer_heldout": {
                    "row_ids": list(signal.outer_heldout.row_ids),
                    "values": signal.outer_heldout.tau_predictions.tolist(),
                    "lineage_ids": lineage_ids(signal.outer_heldout.fit_row_provenance),
                },
            }
        )
    audit = package.producer_audit
    return {
        "schema_version": FOLD_NUMERICAL_SIGNAL_SCHEMA_VERSION,
        "outer_fold": package.outer_fold,
        "split_fingerprint": package.split_fingerprint,
        "outer_train_row_ids": list(package.outer_train_row_ids),
        "outer_heldout_row_ids": list(package.outer_heldout_row_ids),
        "outer_train_row_fingerprint": row_set_fingerprint(package.outer_train_row_ids),
        "outer_heldout_row_fingerprint": row_set_fingerprint(package.outer_heldout_row_ids),
        "producer_audit": {
            "producer_id": audit.producer_id,
            "producer_code_sha256": audit.producer_code_sha256,
            "producer_config_sha256": audit.producer_config_sha256,
            "input_artifact_sha256s": dict(audit.input_artifact_sha256s),
            "posthoc_targets_consumed": audit.posthoc_targets_consumed,
            "outer_heldout_labels_consumed": audit.outer_heldout_labels_consumed,
            "dataset_specific_truth_consumed": audit.dataset_specific_truth_consumed,
        },
        "nuisance": {
            "propensity": inner_vector_payload(package.nuisance.propensity),
            "outcome_prediction": inner_vector_payload(package.nuisance.outcome_prediction),
        },
        "signals": signal_payloads,
        "lineages": lineage_payloads,
    }


_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "outer_fold",
        "split_fingerprint",
        "outer_train_row_ids",
        "outer_heldout_row_ids",
        "outer_train_row_fingerprint",
        "outer_heldout_row_fingerprint",
        "producer_audit",
        "nuisance",
        "signals",
        "lineages",
    }
)
_PRODUCER_AUDIT_FIELDS = frozenset(
    {
        "producer_id",
        "producer_code_sha256",
        "producer_config_sha256",
        "input_artifact_sha256s",
        "posthoc_targets_consumed",
        "outer_heldout_labels_consumed",
        "dataset_specific_truth_consumed",
    }
)
_NUISANCE_FIELDS = frozenset({"propensity", "outcome_prediction"})
_INNER_VECTOR_FIELDS = frozenset({"row_ids", "values", "inner_fold_ids", "lineage_ids"})
_OUTER_VECTOR_FIELDS = frozenset({"row_ids", "values", "lineage_ids"})
_SIGNAL_FIELDS = frozenset(
    {"signal_name", "source_kind", "signal_role", "inner_oof", "outer_heldout"}
)
_LINEAGE_FIELDS = frozenset({"fit_row_ids", "upstream_lineage_ids"})
_MANIFEST_FIELDS = frozenset(
    {"schema_version", "signal_artifact", "identity", "arrays", "materials", "runtime", "honesty"}
)
_MANIFEST_ARTIFACT_FIELDS = frozenset({"path", "sha256", "size_bytes"})
_MANIFEST_IDENTITY_FIELDS = frozenset(
    {
        "outer_fold",
        "split_fingerprint",
        "outer_train_row_ids",
        "outer_heldout_row_ids",
        "nuisance_inner_fold_ids",
        "ordered_signals",
    }
)
_MANIFEST_SIGNAL_FIELDS = frozenset(
    {"signal_name", "source_kind", "signal_role", "inner_fold_ids"}
)
_MANIFEST_MATERIAL_FIELDS = frozenset(
    {"category", "name", "path", "sha256", "size_bytes"}
)
_MANIFEST_RUNTIME_FIELDS = frozenset({"random_seed", "library_versions"})
_MANIFEST_HONESTY_FIELDS = frozenset(
    {
        "posthoc_targets_consumed",
        "outer_heldout_labels_consumed",
        "dataset_specific_truth_consumed",
        "nested_fit_row_lineage_required",
    }
)


def load_fold_numerical_signal_artifact(
    path: Path | str,
    *,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    expected_outer_fold: int,
    expected_split_fingerprint: str,
    expected_outer_train_row_ids: Sequence[Any],
    expected_outer_heldout_row_ids: Sequence[Any],
    required_source_kinds: Sequence[str] = (),
    expected_sha256: str | None = None,
) -> FoldLocalNumericalSignals:
    """Authenticate and load a closed-schema numerical signal sidecar.

    Every expected identity is mandatory.  A legacy Stage-1 prediction file or
    feature matrix cannot be supplied directly because it has neither this
    schema nor exact recursive fit-row lineage.
    """

    requested = Path(path).resolve(strict=True)
    requested_manifest = Path(manifest_path).resolve(strict=True)
    manifest_raw = requested_manifest.read_bytes()
    actual_manifest_sha256 = _sha256_bytes(manifest_raw)
    expected_manifest_digest = _validate_sha256(
        expected_manifest_sha256, name="expected_manifest_sha256"
    )
    if actual_manifest_sha256 != expected_manifest_digest:
        raise ValueError("numerical signal manifest SHA-256 mismatch")
    manifest = _closed_object(
        _load_closed_json(manifest_raw, name="numerical signal manifest"),
        required=_MANIFEST_FIELDS,
        name="manifest",
    )
    if manifest["schema_version"] != FOLD_NUMERICAL_SIGNAL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported numerical signal manifest schema_version")
    manifest_artifact = _closed_object(
        manifest["signal_artifact"],
        required=_MANIFEST_ARTIFACT_FIELDS,
        name="manifest.signal_artifact",
    )
    if Path(str(manifest_artifact["path"])).resolve(strict=True) != requested:
        raise ValueError("manifest signal artifact path mismatch")
    raw = requested.read_bytes()
    actual_sha256 = _sha256_bytes(raw)
    manifest_artifact_sha256 = _validate_sha256(
        manifest_artifact["sha256"], name="manifest.signal_artifact.sha256"
    )
    if actual_sha256 != manifest_artifact_sha256:
        raise ValueError("numerical signal artifact SHA-256 mismatch")
    if expected_sha256 is not None and actual_sha256 != _validate_sha256(
        expected_sha256, name="expected_sha256"
    ):
        raise ValueError("numerical signal artifact SHA-256 mismatch")
    if isinstance(manifest_artifact["size_bytes"], (bool, np.bool_)) or not isinstance(
        manifest_artifact["size_bytes"], (int, np.integer)
    ):
        raise TypeError("manifest.signal_artifact.size_bytes must be an integer")
    if int(manifest_artifact["size_bytes"]) != len(raw):
        raise ValueError("manifest signal artifact size mismatch")
    payload = _load_closed_json(raw, name="numerical signal artifact")
    top = _closed_object(payload, required=_TOP_LEVEL_FIELDS, name="artifact")
    if top["schema_version"] != FOLD_NUMERICAL_SIGNAL_SCHEMA_VERSION:
        raise ValueError("unsupported numerical signal artifact schema_version")
    if isinstance(expected_outer_fold, (bool, np.bool_)) or not isinstance(
        expected_outer_fold, (int, np.integer)
    ):
        raise TypeError("expected_outer_fold must be a positive integer")
    expected_fold = int(expected_outer_fold)
    if expected_fold < 1 or top["outer_fold"] != expected_fold:
        raise ValueError("numerical signal artifact outer_fold mismatch")
    expected_split = _validate_sha256(expected_split_fingerprint, name="expected_split_fingerprint")
    if top["split_fingerprint"] != expected_split:
        raise ValueError("numerical signal artifact split_fingerprint mismatch")
    expected_train = _row_id_tuple(
        expected_outer_train_row_ids, name="expected_outer_train_row_ids"
    )
    expected_heldout = _row_id_tuple(
        expected_outer_heldout_row_ids, name="expected_outer_heldout_row_ids"
    )
    artifact_train = _row_id_tuple(top["outer_train_row_ids"], name="outer_train_row_ids")
    artifact_heldout = _row_id_tuple(top["outer_heldout_row_ids"], name="outer_heldout_row_ids")
    if artifact_train != expected_train or artifact_heldout != expected_heldout:
        raise ValueError("numerical signal artifact row identity/order mismatch")
    if top["outer_train_row_fingerprint"] != row_set_fingerprint(artifact_train):
        raise ValueError("outer train row fingerprint mismatch")
    if top["outer_heldout_row_fingerprint"] != row_set_fingerprint(artifact_heldout):
        raise ValueError("outer heldout row fingerprint mismatch")

    manifest_identity = _closed_object(
        manifest["identity"],
        required=_MANIFEST_IDENTITY_FIELDS,
        name="manifest.identity",
    )
    if (
        manifest_identity["outer_fold"] != top["outer_fold"]
        or manifest_identity["split_fingerprint"] != top["split_fingerprint"]
        or _row_id_tuple(
            manifest_identity["outer_train_row_ids"],
            name="manifest.identity.outer_train_row_ids",
        )
        != artifact_train
        or _row_id_tuple(
            manifest_identity["outer_heldout_row_ids"],
            name="manifest.identity.outer_heldout_row_ids",
        )
        != artifact_heldout
    ):
        raise ValueError("manifest identity does not match numerical signal artifact")

    runtime = _closed_object(
        manifest["runtime"],
        required=_MANIFEST_RUNTIME_FIELDS,
        name="manifest.runtime",
    )
    if isinstance(runtime["random_seed"], (bool, np.bool_)) or not isinstance(
        runtime["random_seed"], (int, np.integer)
    ):
        raise TypeError("manifest.runtime.random_seed must be an integer")
    _normalize_library_versions(runtime["library_versions"])
    honesty = _closed_object(
        manifest["honesty"],
        required=_MANIFEST_HONESTY_FIELDS,
        name="manifest.honesty",
    )
    required_honesty = {
        "posthoc_targets_consumed": False,
        "outer_heldout_labels_consumed": False,
        "dataset_specific_truth_consumed": False,
        "nested_fit_row_lineage_required": True,
    }
    if dict(honesty) != required_honesty:
        raise ValueError("manifest honesty flags fail the closed fail-safe contract")

    manifest_materials = manifest["materials"]
    if isinstance(manifest_materials, (str, bytes)) or not isinstance(
        manifest_materials, Sequence
    ):
        raise TypeError("manifest.materials must be a sequence")
    material_records: list[_AuthenticatedMaterialRecord] = []
    for index, raw_material in enumerate(manifest_materials):
        material = _closed_object(
            raw_material,
            required=_MANIFEST_MATERIAL_FIELDS,
            name=f"manifest.materials[{index}]",
        )
        category = str(material["category"]).strip().lower()
        name = str(material["name"]).strip()
        if category not in REQUIRED_MATERIAL_CATEGORIES:
            raise ValueError("manifest contains an unsupported material category")
        if not name or _FORBIDDEN_NAME.search(name):
            raise ValueError("manifest contains an invalid material name")
        material_path = Path(str(material["path"])).resolve(strict=True)
        if not material_path.is_file():
            raise ValueError("manifest material path is not a regular file")
        digest = _validate_sha256(
            material["sha256"], name=f"manifest.materials[{index}].sha256"
        )
        size = material["size_bytes"]
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)):
            raise TypeError("manifest material size_bytes must be an integer")
        material_raw = material_path.read_bytes()
        if int(size) != len(material_raw) or digest != _sha256_bytes(material_raw):
            raise ValueError(
                f"authenticated producer material mismatch: {category}:{name}"
            )
        material_records.append(
            _AuthenticatedMaterialRecord(
                category=category,
                name=name,
                path=material_path,
                sha256=digest,
                size_bytes=int(size),
            )
        )
    material_keys = [(record.category, record.name) for record in material_records]
    if len(material_keys) != len(set(material_keys)):
        raise ValueError("manifest material category/name pairs must be unique")
    missing_categories = sorted(
        REQUIRED_MATERIAL_CATEGORIES - {record.category for record in material_records}
    )
    if missing_categories:
        raise ValueError(f"manifest is missing material categories: {missing_categories}")

    producer_raw = _closed_object(
        top["producer_audit"],
        required=_PRODUCER_AUDIT_FIELDS,
        name="producer_audit",
    )
    input_hashes = producer_raw["input_artifact_sha256s"]
    if not isinstance(input_hashes, Mapping):
        raise TypeError("producer_audit.input_artifact_sha256s must be an object")
    producer = NumericalSignalProducerAudit(
        producer_id=producer_raw["producer_id"],
        producer_code_sha256=producer_raw["producer_code_sha256"],
        producer_config_sha256=producer_raw["producer_config_sha256"],
        input_artifact_sha256s=input_hashes,
        posthoc_targets_consumed=producer_raw["posthoc_targets_consumed"],
        outer_heldout_labels_consumed=producer_raw["outer_heldout_labels_consumed"],
        dataset_specific_truth_consumed=producer_raw["dataset_specific_truth_consumed"],
    )

    lineage_table_raw = top["lineages"]
    if not isinstance(lineage_table_raw, Mapping) or not lineage_table_raw:
        raise ValueError("lineages must be a non-empty object")
    lineage_specs: dict[str, Mapping[str, Any]] = {}
    for raw_id, raw_spec in lineage_table_raw.items():
        lineage_id = str(raw_id).strip()
        if not lineage_id or lineage_id != raw_id:
            raise ValueError("lineage IDs must be non-empty canonical strings")
        lineage_specs[lineage_id] = _closed_object(
            raw_spec,
            required=_LINEAGE_FIELDS,
            name=f"lineages[{lineage_id!r}]",
        )
    lineage_cache: dict[str, FitRowProvenance] = {}
    active: set[str] = set()
    used_lineages: set[str] = set()

    def build_lineage(lineage_id: Any) -> FitRowProvenance:
        normalized_id = str(lineage_id).strip()
        if normalized_id not in lineage_specs:
            raise ValueError(f"unknown lineage ID {normalized_id!r}")
        used_lineages.add(normalized_id)
        if normalized_id in lineage_cache:
            return lineage_cache[normalized_id]
        if normalized_id in active:
            raise ValueError("lineage table contains a cycle")
        active.add(normalized_id)
        spec = lineage_specs[normalized_id]
        fit_rows = _row_id_tuple(
            spec["fit_row_ids"],
            name=f"lineages[{normalized_id!r}].fit_row_ids",
            require_nonempty=False,
        )
        upstream_ids = spec["upstream_lineage_ids"]
        if isinstance(upstream_ids, (str, bytes)) or not isinstance(upstream_ids, Sequence):
            raise TypeError("upstream_lineage_ids must be a sequence")
        canonical_upstream_ids = tuple(str(value).strip() for value in upstream_ids)
        if any(not value for value in canonical_upstream_ids):
            raise ValueError("upstream lineage IDs must be non-empty")
        if len(canonical_upstream_ids) != len(set(canonical_upstream_ids)):
            raise ValueError("upstream lineage IDs must be unique")
        upstream = tuple(build_lineage(value) for value in canonical_upstream_ids)
        active.remove(normalized_id)
        lineage = FitRowProvenance(
            fit_row_ids=frozenset(fit_rows),
            upstream=upstream,
        )
        if not lineage.recursive_fit_row_ids():
            raise ValueError(f"lineage {normalized_id!r} has no recursive fit rows")
        lineage_cache[normalized_id] = lineage
        return lineage

    def lineage_sequence(values: Any, *, name: str, length: int) -> tuple[FitRowProvenance, ...]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{name} must be a sequence")
        if len(values) != int(length):
            raise ValueError(f"{name} must have length {length}")
        return tuple(build_lineage(value) for value in values)

    nuisance_raw = _closed_object(top["nuisance"], required=_NUISANCE_FIELDS, name="nuisance")

    def parse_inner_vector(raw_value: Any, *, name: str) -> CrossFittedVector:
        value = _closed_object(
            raw_value,
            required=_INNER_VECTOR_FIELDS,
            name=f"nuisance.{name}",
        )
        rows = _row_id_tuple(value["row_ids"], name=f"nuisance.{name}.row_ids")
        return CrossFittedVector(
            name=name,
            row_ids=rows,
            values=_numeric_vector(
                value["values"], name=f"nuisance.{name}.values", length=len(rows)
            ),
            inner_fold_ids=_fold_id_tuple(
                value["inner_fold_ids"],
                name=f"nuisance.{name}.inner_fold_ids",
                length=len(rows),
            ),
            fit_row_provenance=lineage_sequence(
                value["lineage_ids"],
                name=f"nuisance.{name}.lineage_ids",
                length=len(rows),
            ),
        )

    nuisance = CrossFittedNuisance(
        propensity=parse_inner_vector(nuisance_raw["propensity"], name="propensity"),
        outcome_prediction=parse_inner_vector(
            nuisance_raw["outcome_prediction"], name="outcome_prediction"
        ),
    )

    signal_rows = top["signals"]
    if isinstance(signal_rows, (str, bytes)) or not isinstance(signal_rows, Sequence):
        raise TypeError("signals must be a sequence")
    signals: list[FoldLocalSignal] = []
    for index, raw_signal in enumerate(signal_rows):
        signal = _closed_object(
            raw_signal,
            required=_SIGNAL_FIELDS,
            name=f"signals[{index}]",
        )
        signal_name = str(signal["signal_name"]).strip()
        inner = _closed_object(
            signal["inner_oof"],
            required=_INNER_VECTOR_FIELDS,
            name=f"signals[{index}].inner_oof",
        )
        outer = _closed_object(
            signal["outer_heldout"],
            required=_OUTER_VECTOR_FIELDS,
            name=f"signals[{index}].outer_heldout",
        )
        inner_rows = _row_id_tuple(inner["row_ids"], name=f"signals[{index}].inner_oof.row_ids")
        outer_rows = _row_id_tuple(outer["row_ids"], name=f"signals[{index}].outer_heldout.row_ids")
        signals.append(
            FoldLocalSignal(
                signal_name=signal_name,
                source_kind=signal["source_kind"],
                signal_role=signal["signal_role"],
                inner_oof=SignalBundle(
                    row_ids=inner_rows,
                    source_family=signal_name,
                    tau_predictions=_numeric_vector(
                        inner["values"],
                        name=f"signals[{index}].inner_oof.values",
                        length=len(inner_rows),
                    ),
                    prediction_scope=INNER_OOF_SCOPE,
                    fit_row_provenance=lineage_sequence(
                        inner["lineage_ids"],
                        name=f"signals[{index}].inner_oof.lineage_ids",
                        length=len(inner_rows),
                    ),
                ),
                inner_fold_ids=_fold_id_tuple(
                    inner["inner_fold_ids"],
                    name=f"signals[{index}].inner_oof.inner_fold_ids",
                    length=len(inner_rows),
                ),
                outer_heldout=SignalBundle(
                    row_ids=outer_rows,
                    source_family=signal_name,
                    tau_predictions=_numeric_vector(
                        outer["values"],
                        name=f"signals[{index}].outer_heldout.values",
                        length=len(outer_rows),
                    ),
                    prediction_scope=OUTER_HELDOUT_SCOPE,
                    fit_row_provenance=lineage_sequence(
                        outer["lineage_ids"],
                        name=f"signals[{index}].outer_heldout.lineage_ids",
                        length=len(outer_rows),
                    ),
                ),
            )
        )
    if set(lineage_specs) != used_lineages:
        unused = sorted(set(lineage_specs) - used_lineages)
        raise ValueError(f"lineage table contains unused entries: {unused[:3]}")

    package = FoldLocalNumericalSignals(
        outer_fold=top["outer_fold"],
        split_fingerprint=top["split_fingerprint"],
        outer_train_row_ids=artifact_train,
        outer_heldout_row_ids=artifact_heldout,
        artifact_sha256=actual_sha256,
        producer_audit=producer,
        nuisance=nuisance,
        signals=tuple(signals),
        manifest_sha256=actual_manifest_sha256,
    )
    ordered_signals = manifest_identity["ordered_signals"]
    if isinstance(ordered_signals, (str, bytes)) or not isinstance(
        ordered_signals, Sequence
    ):
        raise TypeError("manifest.identity.ordered_signals must be a sequence")
    normalized_ordered_signals: list[Mapping[str, Any]] = []
    for index, raw_signal in enumerate(ordered_signals):
        normalized_ordered_signals.append(
            _closed_object(
                raw_signal,
                required=_MANIFEST_SIGNAL_FIELDS,
                name=f"manifest.identity.ordered_signals[{index}]",
            )
        )
    expected_ordered_signals = [
        {
            "signal_name": signal.signal_name,
            "source_kind": signal.source_kind,
            "signal_role": signal.signal_role,
            "inner_fold_ids": list(signal.inner_fold_ids),
        }
        for signal in package.signals
    ]
    if [dict(value) for value in normalized_ordered_signals] != expected_ordered_signals:
        raise ValueError("manifest ordered signal identity does not match artifact")
    expected_nuisance_folds = {
        "propensity": list(package.nuisance.propensity.inner_fold_ids),
        "outcome_prediction": list(package.nuisance.outcome_prediction.inner_fold_ids),
    }
    if manifest_identity["nuisance_inner_fold_ids"] != expected_nuisance_folds:
        raise ValueError("manifest nuisance fold identity does not match artifact")
    manifest_arrays = manifest["arrays"]
    if not isinstance(manifest_arrays, Mapping):
        raise TypeError("manifest.arrays must be an object")
    if dict(manifest_arrays) != dict(_array_authentication_records(package)):
        raise ValueError("manifest array byte records do not match artifact values")

    material_objects = tuple(
        AuthenticatedMaterialFile(
            category=record.category,
            name=record.name,
            path=record.path,
        )
        for record in material_records
    )
    checked_records = _validate_authenticated_materials(
        material_objects,
        producer_audit=producer,
    )
    if checked_records != tuple(material_records):
        raise ValueError("manifest material records changed during authentication")
    requested_kinds = tuple(str(value).strip().lower() for value in required_source_kinds)
    invalid_kinds = sorted(set(requested_kinds) - SUPPORTED_SIGNAL_KINDS)
    if invalid_kinds:
        raise ValueError(f"required_source_kinds contains unsupported kinds: {invalid_kinds}")
    missing_kinds = sorted(set(requested_kinds) - {signal.source_kind for signal in signals})
    if missing_kinds:
        raise ValueError(f"numerical signal artifact is missing required kinds: {missing_kinds}")
    object.__setattr__(package, "_artifact_path", requested)
    object.__setattr__(package, "_manifest_path", requested_manifest)
    object.__setattr__(package, "_material_records", tuple(material_records))
    object.__setattr__(
        package,
        "_authenticated_content_sha256",
        _sha256_bytes(_canonical_json_bytes(_fold_signal_payload(package))),
    )
    object.__setattr__(
        package,
        "_authentication_capability",
        _AUTHENTICATED_LOADER_CAPABILITY,
    )
    package.verify_authenticated_content()
    return package


__all__ = [
    "AuthenticatedMaterialFile",
    "BOW_R_LOSS",
    "BACKEND_CODE_MATERIAL",
    "BACKEND_CONFIG_MATERIAL",
    "CALIBRATED_TAU_ROLE",
    "CLUSTER_EMBEDDING_CONTRAST",
    "CrossFittedNuisance",
    "CrossFittedVector",
    "FOLD_NUMERICAL_SIGNAL_SCHEMA_VERSION",
    "FOLD_NUMERICAL_SIGNAL_MANIFEST_SCHEMA_VERSION",
    "FoldHonestNumericalSignalFusion",
    "FoldLocalNumericalSignals",
    "FoldLocalSignal",
    "HTR_NEURAL",
    "INPUT_MATERIAL",
    "MATCHED_PAIR_UPLIFT",
    "MODEL_PROJECTION_MATERIAL",
    "NEURAL_QUERY_MOMENTS",
    "NEURAL_QUERY_SIGNAL",
    "NUMERICAL_SIGNAL_FUSION_AUDIT_SCHEMA_VERSION",
    "NumericalSignalProducerAudit",
    "OuterTrainNumericalDiagnosticView",
    "PRODUCER_CODE_MATERIAL",
    "PRODUCER_CONFIG_MATERIAL",
    "RAW_FEATURE_ROLE",
    "REQUIRED_MATERIAL_CATEGORIES",
    "SUPPORTED_SIGNAL_KINDS",
    "SUPPORTED_SIGNAL_ROLES",
    "TFIDF_TOPIC_CONTRAST",
    "WHOLE_EMBEDDING_CONTRAST",
    "WrittenFoldNumericalSignalArtifact",
    "load_fold_numerical_signal_artifact",
    "make_inner_oof_provenance",
    "make_outer_train_provenance",
    "row_set_fingerprint",
    "write_fold_numerical_signal_artifact",
]
