"""Authenticated, role-aware neural-query feature-bank ingestion.

Neural-query activations are representations, not treatment-effect predictions.
Treatment-bank features are reserved for propensity nuisance consumers,
outcome-bank features are reserved for outcome nuisance consumers, and
effect-bank features are uncalibrated modifier bases.  This module therefore
does not construct ``SignalBundle`` or ``FoldLocalSignal`` objects.

Consumers must load a manifest by path and supply its SHA-256 through a trusted
channel.  The manifest in turn binds the signal parquet and every registered
query checkpoint by SHA-256.  A mutable in-memory ``FoldHonestQuerySignals``
instance is deliberately not an accepted consumer input.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .all_evidence_fusion import FoldEvidenceProvenance
from .fold_honest_r_stack import FitRowProvenance
from .fold_honest_signal_fusion import row_set_fingerprint
from .neural_query_signal_artifact import (
    QUERY_BANKS,
    QUERY_SIGNAL_MANIFEST_SCHEMA_VERSION,
    QUERY_SIGNAL_SCHEMA_VERSION,
    query_signal_columns,
)

TREATMENT_NUISANCE_ROLE = "propensity_nuisance_features"
OUTCOME_NUISANCE_ROLE = "outcome_nuisance_features"
EFFECT_MODIFIER_ROLE = "uncalibrated_effect_modifier_basis"
QUERY_BANK_CONSUMER_ROLES = {
    "treatment": TREATMENT_NUISANCE_ROLE,
    "outcome": OUTCOME_NUISANCE_ROLE,
    "effect": EFFECT_MODIFIER_ROLE,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BASE_SIGNAL_COLUMNS = (
    "_oci_row_id",
    "outer_fold",
    "row_scope",
    "inner_fold",
)
_TRAIN_SCOPE = "outer_train_inner_oof"
_HELDOUT_SCOPE = "outer_heldout_final_refit"
_MANIFEST_FIELDS = frozenset({"schema_version", "signal_parquet", "audit"})
_PARQUET_FIELDS = frozenset({"path", "sha256"})
_CHECKPOINT_FIELDS = frozenset({"path", "sha256"})
_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "outer_fold",
        "split_fingerprint",
        "final_refit_fit_row_ids",
        "outer_heldout_row_ids",
        "fit_row_fingerprint",
        "heldout_row_fingerprint",
        "fit_row_count",
        "heldout_row_count",
        "query_count_by_bank",
        "parent_input_binding_sha256",
        "query_discovery_identity",
        "final_refit_checkpoint",
        "subfold_checkpoints",
        "outer_train_activation_scope",
        "outer_heldout_activation_scope",
        "full_refit_train_activations_used",
        "validation_audit_scores_used_as_signal",
        "outer_heldout_labels_accessed",
        "posthoc_targets_consumed",
        "dataset_specific_truth_consumed",
        "rectangular_signal_alignment",
        "fold_local_query_ids_semantically_aligned_across_inner_folds",
    }
)
_SUBFOLD_FIELDS = frozenset(
    {
        "inner_fold",
        "path",
        "sha256",
        "fit_row_ids",
        "validation_row_ids",
        "fit_row_fingerprint",
        "validation_row_fingerprint",
        "validation_row_count",
        "identity",
        "parent_input_binding_sha256",
        "split_fingerprint",
    }
)


def _closed_object(value: Any, *, fields: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    missing = sorted(fields - set(value))
    unexpected = sorted(set(value) - fields)
    if missing or unexpected:
        raise ValueError(
            f"{name} does not match its closed schema; "
            f"missing={missing} unexpected={unexpected}"
        )
    return value


def _sha256(value: Any, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_ids(values: Sequence[Any], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc
    if not raw:
        raise ValueError(f"{name} must be non-empty")
    result: list[int] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must contain canonical integer row IDs")
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} cannot contain negative row IDs")
        result.append(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique row IDs")
    return tuple(result)


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _numeric_matrix(values: Any, *, name: str, rows: int, columns: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (int(rows), int(columns)) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} has the wrong shape or contains non-finite values")
    matrix = matrix.copy()
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True)
class NeuralQueryFeatureBank:
    """One fold-local activation bank with an explicit permitted consumer role."""

    bank: str
    consumer_role: str
    feature_names: tuple[str, ...]
    outer_train_row_ids: tuple[int, ...]
    outer_heldout_row_ids: tuple[int, ...]
    outer_train_inner_oof: np.ndarray = field(repr=False)
    outer_heldout_final_refit: np.ndarray = field(repr=False)
    inner_fold_ids: tuple[int, ...]
    inner_fit_row_provenance: tuple[FitRowProvenance, ...] = field(repr=False)
    outer_fit_row_provenance: tuple[FitRowProvenance, ...] = field(repr=False)

    def __post_init__(self) -> None:
        bank = str(self.bank).strip().lower()
        if bank not in QUERY_BANKS:
            raise ValueError(f"bank must be one of {list(QUERY_BANKS)}")
        expected_role = QUERY_BANK_CONSUMER_ROLES[bank]
        if self.consumer_role != expected_role:
            raise ValueError(f"{bank} query bank must use consumer role {expected_role!r}")
        names = tuple(str(value).strip() for value in self.feature_names)
        if not names or any(not value for value in names) or len(names) != len(set(names)):
            raise ValueError("feature_names must be non-empty and unique")
        prefix = f"neural_query_{bank}_"
        if any(not name.startswith(prefix) for name in names):
            raise ValueError(f"{bank} feature names violate their role prefix")
        train_ids = _row_ids(self.outer_train_row_ids, name=f"{bank}.outer_train_row_ids")
        heldout_ids = _row_ids(self.outer_heldout_row_ids, name=f"{bank}.outer_heldout_row_ids")
        if set(train_ids) & set(heldout_ids):
            raise ValueError("query feature train and heldout rows overlap")
        train_values = _numeric_matrix(
            self.outer_train_inner_oof,
            name=f"{bank}.outer_train_inner_oof",
            rows=len(train_ids),
            columns=len(names),
        )
        heldout_values = _numeric_matrix(
            self.outer_heldout_final_refit,
            name=f"{bank}.outer_heldout_final_refit",
            rows=len(heldout_ids),
            columns=len(names),
        )
        folds = tuple(
            _positive_int(value, name=f"{bank}.inner_fold_ids") for value in self.inner_fold_ids
        )
        if len(folds) != len(train_ids) or len(set(folds)) < 2:
            raise ValueError(f"{bank}.inner_fold_ids do not define an inner partition")
        inner_lineage = tuple(self.inner_fit_row_provenance)
        outer_lineage = tuple(self.outer_fit_row_provenance)
        if len(inner_lineage) != len(train_ids) or len(outer_lineage) != len(heldout_ids):
            raise ValueError(f"{bank} fit-row provenance has the wrong length")
        if not all(isinstance(item, FitRowProvenance) for item in (*inner_lineage, *outer_lineage)):
            raise TypeError("query feature provenance must contain FitRowProvenance")
        train_set = set(train_ids)
        heldout_set = set(heldout_ids)
        rows_by_fold: dict[int, set[int]] = {}
        for row_id, fold_id in zip(train_ids, folds):
            rows_by_fold.setdefault(fold_id, set()).add(row_id)
        for row_id, fold_id, lineage in zip(train_ids, folds, inner_lineage):
            recursive = set(lineage.recursive_fit_row_ids())
            if not recursive or not recursive <= train_set or recursive & rows_by_fold[fold_id]:
                raise ValueError(f"{bank} row {row_id} has non-honest inner provenance")
        for row_id, lineage in zip(heldout_ids, outer_lineage):
            recursive = set(lineage.recursive_fit_row_ids())
            if not recursive or not recursive <= train_set or recursive & heldout_set:
                raise ValueError(f"{bank} row {row_id} has non-honest outer provenance")
        object.__setattr__(self, "bank", bank)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "outer_train_row_ids", train_ids)
        object.__setattr__(self, "outer_heldout_row_ids", heldout_ids)
        object.__setattr__(self, "outer_train_inner_oof", train_values)
        object.__setattr__(self, "outer_heldout_final_refit", heldout_values)
        object.__setattr__(self, "inner_fold_ids", folds)
        object.__setattr__(self, "inner_fit_row_provenance", inner_lineage)
        object.__setattr__(self, "outer_fit_row_provenance", outer_lineage)

    @property
    def calibrated_tau(self) -> bool:
        """Raw query activations are never calibrated treatment effects."""

        return False

    def require_calibrated_tau(self) -> None:
        raise RuntimeError(
            "Neural-query activations are role-specific feature bases, not tau "
            "predictions. Fit a nested calibrated R backend before constructing "
            "a FoldLocalSignal."
        )


@dataclass(frozen=True)
class NeuralQueryFeatureBanks:
    """Authenticated role-separated banks for explicit downstream consumers."""

    outer_fold: int
    split_fingerprint: str
    manifest_sha256: str
    signal_parquet_sha256: str
    outer_train_row_ids: tuple[int, ...]
    outer_heldout_row_ids: tuple[int, ...]
    treatment: NeuralQueryFeatureBank
    outcome: NeuralQueryFeatureBank
    effect: NeuralQueryFeatureBank

    def __post_init__(self) -> None:
        fold = _positive_int(self.outer_fold, name="outer_fold")
        split = _sha256(self.split_fingerprint, name="split_fingerprint")
        manifest_sha = _sha256(self.manifest_sha256, name="manifest_sha256")
        signal_sha = _sha256(self.signal_parquet_sha256, name="signal_parquet_sha256")
        train_ids = _row_ids(self.outer_train_row_ids, name="outer_train_row_ids")
        heldout_ids = _row_ids(self.outer_heldout_row_ids, name="outer_heldout_row_ids")
        for expected_bank, bank in (
            ("treatment", self.treatment),
            ("outcome", self.outcome),
            ("effect", self.effect),
        ):
            if not isinstance(bank, NeuralQueryFeatureBank) or bank.bank != expected_bank:
                raise TypeError(f"{expected_bank} must be its matching NeuralQueryFeatureBank")
            if bank.outer_train_row_ids != train_ids or bank.outer_heldout_row_ids != heldout_ids:
                raise ValueError("query feature banks do not share exact row identity/order")
        object.__setattr__(self, "outer_fold", fold)
        object.__setattr__(self, "split_fingerprint", split)
        object.__setattr__(self, "manifest_sha256", manifest_sha)
        object.__setattr__(self, "signal_parquet_sha256", signal_sha)
        object.__setattr__(self, "outer_train_row_ids", train_ids)
        object.__setattr__(self, "outer_heldout_row_ids", heldout_ids)

    def for_propensity_nuisance(self) -> NeuralQueryFeatureBank:
        return self.treatment

    def for_outcome_nuisance(self) -> NeuralQueryFeatureBank:
        return self.outcome

    def for_effect_modifier_basis(self) -> NeuralQueryFeatureBank:
        return self.effect


def load_authenticated_neural_query_feature_banks(
    manifest_path: Path | str,
    *,
    expected_manifest_sha256: str,
    expected_outer_fold: int,
    expected_split_fingerprint: str,
    expected_outer_train_row_ids: Sequence[Any],
    expected_outer_heldout_row_ids: Sequence[Any],
    expected_parent_input_binding_sha256: str,
    expected_query_discovery_identity: str,
) -> NeuralQueryFeatureBanks:
    """Authenticate a manifest and return role-separated activation features."""

    requested_manifest = Path(manifest_path).resolve(strict=True)
    if not requested_manifest.is_file():
        raise FileNotFoundError(f"query signal manifest is not a file: {requested_manifest}")
    manifest_bytes = requested_manifest.read_bytes()
    actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    expected_manifest_sha256 = _sha256(expected_manifest_sha256, name="expected_manifest_sha256")
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("neural query manifest SHA-256 mismatch")
    expected_parent_input_binding_sha256 = _sha256(
        expected_parent_input_binding_sha256,
        name="expected_parent_input_binding_sha256",
    )
    expected_query_discovery_identity = _sha256(
        expected_query_discovery_identity,
        name="expected_query_discovery_identity",
    )
    try:
        manifest = json.loads(
            manifest_bytes.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("neural query manifest is not valid UTF-8 JSON") from exc
    manifest = _closed_object(manifest, fields=_MANIFEST_FIELDS, name="query manifest")
    if manifest["schema_version"] != QUERY_SIGNAL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported neural query manifest schema_version")
    parquet_spec = _closed_object(
        manifest["signal_parquet"], fields=_PARQUET_FIELDS, name="signal_parquet"
    )
    parquet_sha256 = _sha256(parquet_spec["sha256"], name="signal_parquet.sha256")
    parquet_path = _resolve_manifest_local_file(
        requested_manifest.parent,
        parquet_spec["path"],
        name="signal_parquet.path",
    )
    parquet_bytes = parquet_path.read_bytes()
    if hashlib.sha256(parquet_bytes).hexdigest() != parquet_sha256:
        raise ValueError("neural query signal parquet SHA-256 mismatch")

    audit = _closed_object(manifest["audit"], fields=_AUDIT_FIELDS, name="query signal audit")
    if audit["schema_version"] != QUERY_SIGNAL_SCHEMA_VERSION:
        raise ValueError("unsupported neural query signal schema_version")
    outer_fold = _positive_int(expected_outer_fold, name="expected_outer_fold")
    if _positive_int(audit["outer_fold"], name="audit.outer_fold") != outer_fold:
        raise ValueError("neural query signal outer_fold mismatch")
    split_fingerprint = _sha256(expected_split_fingerprint, name="expected_split_fingerprint")
    if audit["split_fingerprint"] != split_fingerprint:
        raise ValueError("neural query signal outer split fingerprint mismatch")
    train_ids = _row_ids(expected_outer_train_row_ids, name="expected_outer_train_row_ids")
    heldout_ids = _row_ids(expected_outer_heldout_row_ids, name="expected_outer_heldout_row_ids")
    if set(train_ids) & set(heldout_ids):
        raise ValueError("expected query feature train and heldout rows overlap")
    audit_train = _row_ids(audit["final_refit_fit_row_ids"], name="audit.final_refit_fit_row_ids")
    audit_heldout = _row_ids(audit["outer_heldout_row_ids"], name="audit.outer_heldout_row_ids")
    if audit_train != train_ids or audit_heldout != heldout_ids:
        raise ValueError("neural query signal audit row identity/order mismatch")
    if _positive_int(audit["fit_row_count"], name="audit.fit_row_count") != len(train_ids):
        raise ValueError("neural query signal fit row count mismatch")
    if _positive_int(audit["heldout_row_count"], name="audit.heldout_row_count") != len(
        heldout_ids
    ):
        raise ValueError("neural query signal heldout row count mismatch")
    if audit["fit_row_fingerprint"] != row_set_fingerprint(train_ids):
        raise ValueError("neural query signal fit row fingerprint mismatch")
    if audit["heldout_row_fingerprint"] != row_set_fingerprint(heldout_ids):
        raise ValueError("neural query signal heldout row fingerprint mismatch")
    parent_input_binding_sha256 = _sha256(
        audit["parent_input_binding_sha256"],
        name="audit.parent_input_binding_sha256",
    )
    if parent_input_binding_sha256 != expected_parent_input_binding_sha256:
        raise ValueError("neural query signal parent input binding mismatch")
    query_discovery_identity = _sha256(
        audit["query_discovery_identity"],
        name="audit.query_discovery_identity",
    )
    if query_discovery_identity != expected_query_discovery_identity:
        raise ValueError("neural query signal discovery identity mismatch")
    if audit["outer_train_activation_scope"] != "strict_inner_oof_only":
        raise ValueError("neural query train activations are not strict inner OOF")
    if audit["outer_heldout_activation_scope"] != "full_outer_train_refit_queries_text_only":
        raise ValueError("neural query heldout activations are not final outer-train refit")
    for field_name in (
        "full_refit_train_activations_used",
        "validation_audit_scores_used_as_signal",
        "outer_heldout_labels_accessed",
        "posthoc_targets_consumed",
        "dataset_specific_truth_consumed",
        "fold_local_query_ids_semantically_aligned_across_inner_folds",
    ):
        if audit[field_name] is not False:
            raise ValueError(f"neural query audit requires {field_name}=false")
    if (
        audit["rectangular_signal_alignment"]
        != "permutation_invariant_signed_activation_order_statistics_by_bank"
    ):
        raise ValueError("neural query signal alignment contract mismatch")
    counts = audit["query_count_by_bank"]
    expected_columns = query_signal_columns(counts)

    _verify_checkpoint_registration(
        audit["final_refit_checkpoint"],
        fields=_CHECKPOINT_FIELDS,
        manifest_directory=requested_manifest.parent,
        name="final_refit_checkpoint",
    )
    lineage_by_fold, fold_by_train_row = _validate_subfolds(
        audit["subfold_checkpoints"],
        outer_fold=outer_fold,
        train_ids=train_ids,
        heldout_ids=heldout_ids,
        manifest_directory=requested_manifest.parent,
        parent_input_binding_sha256=parent_input_binding_sha256,
    )
    try:
        raw_frame = pd.read_parquet(io.BytesIO(parquet_bytes))
    except Exception as exc:
        raise ValueError("neural query signal parquet cannot be decoded") from exc
    frame = _validate_signal_frame(
        raw_frame,
        outer_fold=outer_fold,
        train_ids=train_ids,
        heldout_ids=heldout_ids,
        fold_by_train_row=fold_by_train_row,
        expected_signal_columns=expected_columns,
    )
    train_frame = frame.loc[frame["row_scope"] == _TRAIN_SCOPE].set_index("_oci_row_id", drop=False)
    heldout_frame = frame.loc[frame["row_scope"] == _HELDOUT_SCOPE].set_index(
        "_oci_row_id", drop=False
    )
    train_fold_ids = tuple(fold_by_train_row[row_id] for row_id in train_ids)
    train_lineage = tuple(lineage_by_fold[fold_id] for fold_id in train_fold_ids)
    outer_lineage = FitRowProvenance(fit_row_ids=frozenset(train_ids))
    banks: dict[str, NeuralQueryFeatureBank] = {}
    for bank in QUERY_BANKS:
        prefix = f"neural_query_{bank}_"
        feature_names = tuple(name for name in expected_columns if name.startswith(prefix))
        banks[bank] = NeuralQueryFeatureBank(
            bank=bank,
            consumer_role=QUERY_BANK_CONSUMER_ROLES[bank],
            feature_names=feature_names,
            outer_train_row_ids=train_ids,
            outer_heldout_row_ids=heldout_ids,
            outer_train_inner_oof=train_frame.loc[list(train_ids), list(feature_names)].to_numpy(
                dtype=float
            ),
            outer_heldout_final_refit=heldout_frame.loc[
                list(heldout_ids), list(feature_names)
            ].to_numpy(dtype=float),
            inner_fold_ids=train_fold_ids,
            inner_fit_row_provenance=train_lineage,
            outer_fit_row_provenance=tuple(outer_lineage for _row_id in heldout_ids),
        )
    return NeuralQueryFeatureBanks(
        outer_fold=outer_fold,
        split_fingerprint=split_fingerprint,
        manifest_sha256=actual_manifest_sha256,
        signal_parquet_sha256=parquet_sha256,
        outer_train_row_ids=train_ids,
        outer_heldout_row_ids=heldout_ids,
        treatment=banks["treatment"],
        outcome=banks["outcome"],
        effect=banks["effect"],
    )


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"neural query manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _resolve_manifest_local_file(directory: Path, raw_path: Any, *, name: str) -> Path:
    value = str(raw_path or "").strip()
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name in {".", ".."}
    ):
        raise ValueError(f"{name} must be a single manifest-local filename")
    resolved = (directory / relative).resolve(strict=True)
    if resolved.parent != directory.resolve() or not resolved.is_file():
        raise ValueError(f"{name} does not resolve to a manifest-local file")
    return resolved


def _verify_checkpoint_registration(
    raw_registration: Any,
    *,
    fields: frozenset[str],
    manifest_directory: Path,
    name: str,
) -> Path:
    registration = _closed_object(raw_registration, fields=fields, name=name)
    expected_sha256 = _sha256(registration["sha256"], name=f"{name}.sha256")
    raw_path = str(registration["path"] or "").strip()
    if not raw_path:
        raise ValueError(f"{name}.path must be non-empty")
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_directory / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"registered query checkpoint is not a file: {path}")
    if _sha256_path(path) != expected_sha256:
        raise ValueError(f"registered query checkpoint SHA-256 mismatch: {name}")
    return path


def _validate_subfolds(
    raw_subfolds: Any,
    *,
    outer_fold: int,
    train_ids: tuple[int, ...],
    heldout_ids: tuple[int, ...],
    manifest_directory: Path,
    parent_input_binding_sha256: str,
) -> tuple[dict[int, FitRowProvenance], dict[int, int]]:
    if isinstance(raw_subfolds, (str, bytes)) or not isinstance(raw_subfolds, Sequence):
        raise TypeError("query subfold_checkpoints must be a sequence")
    if len(raw_subfolds) < 2:
        raise ValueError("query signals require at least two subfold checkpoints")
    train_set = set(train_ids)
    heldout_set = set(heldout_ids)
    lineage_by_fold: dict[int, FitRowProvenance] = {}
    fold_by_train_row: dict[int, int] = {}
    for position, raw in enumerate(raw_subfolds):
        name = f"query subfold_checkpoints[{position}]"
        subfold = _closed_object(raw, fields=_SUBFOLD_FIELDS, name=name)
        inner_fold = _positive_int(subfold["inner_fold"], name=f"{name}.inner_fold")
        if inner_fold in lineage_by_fold:
            raise ValueError("query subfold inner_fold values are duplicate")
        _sha256(subfold["identity"], name=f"{name}.identity")
        subfold_parent_input_binding_sha256 = _sha256(
            subfold["parent_input_binding_sha256"],
            name=f"{name}.parent_input_binding_sha256",
        )
        if subfold_parent_input_binding_sha256 != parent_input_binding_sha256:
            raise ValueError(
                f"query subfold {inner_fold} parent input binding does not match "
                "the query signal audit"
            )
        _verify_checkpoint_registration(
            subfold,
            fields=_SUBFOLD_FIELDS,
            manifest_directory=manifest_directory,
            name=name,
        )
        fit_ids = _row_ids(subfold["fit_row_ids"], name=f"query subfold {inner_fold} fit_row_ids")
        validation_ids = _row_ids(
            subfold["validation_row_ids"],
            name=f"query subfold {inner_fold} validation_row_ids",
        )
        if set(fit_ids) & set(validation_ids) or set(fit_ids) | set(validation_ids) != train_set:
            raise ValueError(f"query subfold {inner_fold} does not partition outer train")
        if (set(fit_ids) | set(validation_ids)) & heldout_set:
            raise ValueError(f"query subfold {inner_fold} contains outer-heldout rows")
        if _positive_int(
            subfold["validation_row_count"],
            name=f"query subfold {inner_fold} validation_row_count",
        ) != len(validation_ids):
            raise ValueError(f"query subfold {inner_fold} validation row count mismatch")
        if subfold["fit_row_fingerprint"] != row_set_fingerprint(fit_ids):
            raise ValueError(f"query subfold {inner_fold} fit row fingerprint mismatch")
        if subfold["validation_row_fingerprint"] != row_set_fingerprint(validation_ids):
            raise ValueError(f"query subfold {inner_fold} validation row fingerprint mismatch")
        expected_split = FoldEvidenceProvenance(
            outer_fold=outer_fold,
            train_row_ids=fit_ids,
            heldout_row_ids=validation_ids,
            scope="inner_train",
            inner_fold=inner_fold,
            artifact_id="query-feature-bank-validation",
        ).split_fingerprint
        if subfold["split_fingerprint"] != expected_split:
            raise ValueError(f"query subfold {inner_fold} split fingerprint mismatch")
        lineage_by_fold[inner_fold] = FitRowProvenance(fit_row_ids=frozenset(fit_ids))
        for row_id in validation_ids:
            if row_id in fold_by_train_row:
                raise ValueError("query subfold validations overlap")
            fold_by_train_row[row_id] = inner_fold
    if set(fold_by_train_row) != train_set:
        raise ValueError("query subfold validations do not cover outer train exactly once")
    return lineage_by_fold, fold_by_train_row


def _validate_signal_frame(
    raw_frame: Any,
    *,
    outer_fold: int,
    train_ids: tuple[int, ...],
    heldout_ids: tuple[int, ...],
    fold_by_train_row: Mapping[int, int],
    expected_signal_columns: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(raw_frame, pd.DataFrame):
        raise TypeError("query signals must be a pandas DataFrame")
    frame = raw_frame.copy()
    expected_columns = {*_BASE_SIGNAL_COLUMNS, *expected_signal_columns}
    actual_columns = set(frame.columns)
    if frame.columns.duplicated().any() or actual_columns != expected_columns:
        raise ValueError(
            "query signal frame does not match the exact generated statistic schema; "
            f"missing={sorted(expected_columns - actual_columns, key=str)} "
            f"unexpected={sorted(actual_columns - expected_columns, key=str)}"
        )
    if len(frame) != len(train_ids) + len(heldout_ids):
        raise ValueError("query signal frame has the wrong row count")
    if frame["_oci_row_id"].isna().any() or frame["_oci_row_id"].duplicated().any():
        raise ValueError("query signal frame row IDs are missing or duplicated")
    frame_row_ids = _row_ids(frame["_oci_row_id"].tolist(), name="query signal row IDs")
    frame.loc[:, "_oci_row_id"] = frame_row_ids
    for value in frame["outer_fold"].tolist():
        if _positive_int(value, name="query signal outer_fold") != outer_fold:
            raise ValueError("query signal frame outer_fold mismatch")
    if set(frame["row_scope"]) != {_TRAIN_SCOPE, _HELDOUT_SCOPE}:
        raise ValueError("query signal frame has invalid row_scope values")
    train = frame.loc[frame["row_scope"] == _TRAIN_SCOPE]
    heldout = frame.loc[frame["row_scope"] == _HELDOUT_SCOPE]
    if set(train["_oci_row_id"]) != set(train_ids) or set(heldout["_oci_row_id"]) != set(
        heldout_ids
    ):
        raise ValueError("query signal frame scopes do not match manifest row partitions")
    if train["inner_fold"].isna().any() or heldout["inner_fold"].notna().any():
        raise ValueError("query signal frame inner_fold nullability violates row scope")
    for row_id, raw_inner_fold in zip(train["_oci_row_id"], train["inner_fold"]):
        inner_fold = _positive_int(raw_inner_fold, name="query signal inner_fold")
        if inner_fold != fold_by_train_row[int(row_id)]:
            raise ValueError("query signal frame inner_fold disagrees with manifest lineage")
    numeric = (
        frame.loc[:, list(expected_signal_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    if not np.isfinite(numeric).all():
        raise ValueError("query signal frame contains non-finite activation statistics")
    frame.loc[:, list(expected_signal_columns)] = numeric
    return frame


__all__ = [
    "EFFECT_MODIFIER_ROLE",
    "NeuralQueryFeatureBank",
    "NeuralQueryFeatureBanks",
    "OUTCOME_NUISANCE_ROLE",
    "QUERY_BANK_CONSUMER_ROLES",
    "TREATMENT_NUISANCE_ROLE",
    "load_authenticated_neural_query_feature_banks",
]
