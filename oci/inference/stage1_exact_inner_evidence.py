"""Authenticated, canonical-split production contract for exact-inner Stage 1 evidence.

The historical multi-model runner assigned the same integer ``inner_fold`` to
architecture-local cross-fits that used different random seeds.  It then copied
full-outer discovery evidence into rows labelled as inner-training evidence.
Counts such as ``train_rows`` cannot prove a fit scope, so those rows are not a
safe production input.

This module defines the replacement boundary.  One canonical registry owns every
row split.  Each of the ten active Stage 1 architecture producers is invoked on
the same exact-inner request, which contains fit labels but only held-out text.
Producer identities, fit audits, payloads, and the resulting bundle are content
addressed.  The validator fails closed on missing families, split drift, identity
drift, forbidden data, or byte-semantic reuse of a registered full-outer payload.

The module deliberately does not implement any modeling architecture.  Production
architecture adapters must implement :class:`ExactInnerStage1FamilyProducer` and
are therefore independently testable and independently authenticated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
)

CANONICAL_STAGE1_SPLIT_REGISTRY_VERSION = "canonical_stage1_split_registry_v1"
EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION = "exact_inner_stage1_family_producer_identity_v1"
EXACT_INNER_FIT_AUDIT_VERSION = "exact_inner_stage1_family_fit_audit_v1"
EXACT_INNER_FAMILY_ARTIFACT_VERSION = "exact_inner_stage1_family_artifact_v1"
EXACT_INNER_EVIDENCE_BUNDLE_VERSION = "exact_inner_stage1_evidence_bundle_v1"
EXACT_INNER_REQUEST_VERSION = "exact_inner_stage1_family_request_v1"

EXACT_INNER_REFIT = "exact_inner_refit"
EXACT_SCOPE_CACHE_REPLAY = "exact_scope_authenticated_cache_replay"
_VALID_FIT_SEMANTICS = frozenset({EXACT_INNER_REFIT, EXACT_SCOPE_CACHE_REPLAY})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_EVIDENCE_KEY = re.compile(
    r"(?:^|_)(?:"
    r"oracle|ground_truth|true_ite|true_cate|true_effect|"
    r"row_id|row_ids|patient_id|patient_ids|mrn|medical_record_number|"
    r"api_key|access_token|refresh_token|password|passwd|secret|credential|"
    r"authorization"
    r")(?:_|$)",
    flags=re.IGNORECASE,
)
_FORBIDDEN_REUSE_KEY = re.compile(
    r"(?:^|_)(?:evidence_reused_from_fold_key|reused_from_fold_key|"
    r"full_outer_source_artifact)(?:_|$)",
    flags=re.IGNORECASE,
)
_FORBIDDEN_REUSE_VALUE = re.compile(
    r"reused[_\s-]*full[_\s-]*outer|full[_\s-]*outer[_\s-]*reuse",
    flags=re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:bearer\s+[a-z0-9._~+/=-]{12,}|\bsk-[a-z0-9_-]{12,})",
    flags=re.IGNORECASE,
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


def _require_sha256(value: Any, *, field_name: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return text


def _canonical_row_ids(values: Sequence[Any], *, field_name: str) -> tuple[int, ...]:
    row_ids: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{field_name} cannot contain boolean row IDs")
        try:
            row_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain integer row IDs") from exc
        if row_id < 0 or row_id != value:
            raise ValueError(f"{field_name} must contain non-negative integer row IDs")
        row_ids.append(row_id)
    if not row_ids:
        raise ValueError(f"{field_name} cannot be empty")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"{field_name} must contain unique row IDs")
    return tuple(row_ids)


def row_order_fingerprint(row_ids: Sequence[int]) -> str:
    """Hash an ordered row registry without accepting ambiguous mixed ID types."""

    rows = _canonical_row_ids(row_ids, field_name="row_ids")
    return _sha256_json({"ordered_row_ids": list(rows)})


@dataclass(frozen=True)
class CanonicalInnerSplit:
    outer_fold: int
    inner_fold: int
    fit_row_ids: tuple[int, ...]
    heldout_row_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if int(self.outer_fold) < 1 or int(self.inner_fold) < 1:
            raise ValueError("outer_fold and inner_fold must be positive")
        fit = _canonical_row_ids(self.fit_row_ids, field_name="fit_row_ids")
        heldout = _canonical_row_ids(self.heldout_row_ids, field_name="heldout_row_ids")
        if set(fit) & set(heldout):
            raise ValueError("exact-inner fit and held-out rows overlap")
        object.__setattr__(self, "outer_fold", int(self.outer_fold))
        object.__setattr__(self, "inner_fold", int(self.inner_fold))
        object.__setattr__(self, "fit_row_ids", fit)
        object.__setattr__(self, "heldout_row_ids", heldout)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "fit_row_ids": list(self.fit_row_ids),
            "heldout_row_ids": list(self.heldout_row_ids),
            "fit_row_fingerprint": row_order_fingerprint(self.fit_row_ids),
            "heldout_row_fingerprint": row_order_fingerprint(self.heldout_row_ids),
        }

    @property
    def scope_fingerprint(self) -> str:
        return _sha256_json(self.as_dict())


@dataclass(frozen=True)
class CanonicalOuterSplit:
    outer_fold: int
    train_row_ids: tuple[int, ...]
    heldout_row_ids: tuple[int, ...]
    inner_splits: tuple[CanonicalInnerSplit, ...]

    def __post_init__(self) -> None:
        if int(self.outer_fold) < 1:
            raise ValueError("outer_fold must be positive")
        train = _canonical_row_ids(self.train_row_ids, field_name="outer train_row_ids")
        heldout = _canonical_row_ids(
            self.heldout_row_ids,
            field_name="outer heldout_row_ids",
        )
        if set(train) & set(heldout):
            raise ValueError("outer fit and held-out rows overlap")
        splits = tuple(self.inner_splits)
        if len(splits) < 2:
            raise ValueError("each outer fold requires at least two exact-inner splits")
        if tuple(split.inner_fold for split in splits) != tuple(range(1, len(splits) + 1)):
            raise ValueError("inner folds must be complete, ordered, and one-based")
        heldout_counts = {row_id: 0 for row_id in train}
        for split in splits:
            if split.outer_fold != int(self.outer_fold):
                raise ValueError("inner split changed its outer fold")
            if set(split.fit_row_ids) | set(split.heldout_row_ids) != set(train):
                raise ValueError("inner split does not partition the canonical outer train rows")
            if tuple(row_id for row_id in train if row_id in set(split.fit_row_ids)) != (
                split.fit_row_ids
            ):
                raise ValueError("inner fit rows changed canonical row order")
            if tuple(row_id for row_id in train if row_id in set(split.heldout_row_ids)) != (
                split.heldout_row_ids
            ):
                raise ValueError("inner held-out rows changed canonical row order")
            for row_id in split.heldout_row_ids:
                heldout_counts[row_id] += 1
        if set(heldout_counts.values()) != {1}:
            raise ValueError("inner held-out folds must partition outer training exactly once")
        object.__setattr__(self, "outer_fold", int(self.outer_fold))
        object.__setattr__(self, "train_row_ids", train)
        object.__setattr__(self, "heldout_row_ids", heldout)
        object.__setattr__(self, "inner_splits", splits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_fold": self.outer_fold,
            "train_row_ids": list(self.train_row_ids),
            "heldout_row_ids": list(self.heldout_row_ids),
            "train_row_fingerprint": row_order_fingerprint(self.train_row_ids),
            "heldout_row_fingerprint": row_order_fingerprint(self.heldout_row_ids),
            "inner_splits": [split.as_dict() for split in self.inner_splits],
        }


@dataclass(frozen=True)
class CanonicalStage1SplitRegistry:
    dataset_row_ids: tuple[int, ...]
    outer_splits: tuple[CanonicalOuterSplit, ...]
    inner_fold_count: int
    inner_seed_base: int

    def __post_init__(self) -> None:
        dataset_rows = _canonical_row_ids(
            self.dataset_row_ids,
            field_name="dataset_row_ids",
        )
        outer_splits = tuple(self.outer_splits)
        if not outer_splits:
            raise ValueError("canonical split registry has no outer folds")
        if tuple(split.outer_fold for split in outer_splits) != tuple(
            range(1, len(outer_splits) + 1)
        ):
            raise ValueError("outer folds must be complete, ordered, and one-based")
        heldout_counts = {row_id: 0 for row_id in dataset_rows}
        for split in outer_splits:
            if set(split.train_row_ids) | set(split.heldout_row_ids) != set(dataset_rows):
                raise ValueError("outer split does not partition the dataset registry")
            if len(split.inner_splits) != int(self.inner_fold_count):
                raise ValueError("outer folds disagree with the configured inner fold count")
            for row_id in split.heldout_row_ids:
                heldout_counts[row_id] += 1
        if set(heldout_counts.values()) != {1}:
            raise ValueError("outer held-out folds must partition the dataset exactly once")
        if int(self.inner_fold_count) < 2:
            raise ValueError("inner_fold_count must be at least two")
        object.__setattr__(self, "dataset_row_ids", dataset_rows)
        object.__setattr__(self, "outer_splits", outer_splits)
        object.__setattr__(self, "inner_fold_count", int(self.inner_fold_count))
        object.__setattr__(self, "inner_seed_base", int(self.inner_seed_base))

    @classmethod
    def build(
        cls,
        *,
        dataset_row_ids: Sequence[int],
        outer_heldout_row_ids: Mapping[int, Sequence[int]],
        inner_fold_count: int,
        inner_seed_base: int = 51_000,
    ) -> "CanonicalStage1SplitRegistry":
        rows = _canonical_row_ids(dataset_row_ids, field_name="dataset_row_ids")
        fold_ids = tuple(sorted(int(value) for value in outer_heldout_row_ids))
        if fold_ids != tuple(range(1, len(fold_ids) + 1)):
            raise ValueError("outer_heldout_row_ids must use complete one-based folds")
        row_set = set(rows)
        outer_splits: list[CanonicalOuterSplit] = []
        for outer_fold in fold_ids:
            supplied = _canonical_row_ids(
                outer_heldout_row_ids[outer_fold],
                field_name=f"outer fold {outer_fold} heldout rows",
            )
            if not set(supplied) <= row_set:
                raise ValueError("outer held-out registry contains unknown rows")
            supplied_set = set(supplied)
            heldout = tuple(row_id for row_id in rows if row_id in supplied_set)
            train = tuple(row_id for row_id in rows if row_id not in supplied_set)
            if len(train) < int(inner_fold_count):
                raise ValueError("outer training rows are too few for requested inner folds")
            splitter = KFold(
                n_splits=int(inner_fold_count),
                shuffle=True,
                random_state=int(inner_seed_base) + int(outer_fold),
            )
            inner_splits: list[CanonicalInnerSplit] = []
            for inner_fold, (fit_pos, heldout_pos) in enumerate(
                splitter.split(np.arange(len(train))),
                start=1,
            ):
                fit_set = {int(value) for value in fit_pos}
                heldout_set = {int(value) for value in heldout_pos}
                inner_splits.append(
                    CanonicalInnerSplit(
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                        fit_row_ids=tuple(
                            row_id for position, row_id in enumerate(train) if position in fit_set
                        ),
                        heldout_row_ids=tuple(
                            row_id
                            for position, row_id in enumerate(train)
                            if position in heldout_set
                        ),
                    )
                )
            outer_splits.append(
                CanonicalOuterSplit(
                    outer_fold=outer_fold,
                    train_row_ids=train,
                    heldout_row_ids=heldout,
                    inner_splits=tuple(inner_splits),
                )
            )
        return cls(
            dataset_row_ids=rows,
            outer_splits=tuple(outer_splits),
            inner_fold_count=int(inner_fold_count),
            inner_seed_base=int(inner_seed_base),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CANONICAL_STAGE1_SPLIT_REGISTRY_VERSION,
            "dataset_row_ids": list(self.dataset_row_ids),
            "dataset_row_fingerprint": row_order_fingerprint(self.dataset_row_ids),
            "inner_fold_count": self.inner_fold_count,
            "inner_seed_base": self.inner_seed_base,
            "outer_splits": [split.as_dict() for split in self.outer_splits],
        }

    @property
    def content_sha256(self) -> str:
        return _sha256_json(self.as_dict())

    def inner_split(self, outer_fold: int, inner_fold: int) -> CanonicalInnerSplit:
        try:
            outer = self.outer_splits[int(outer_fold) - 1]
            split = outer.inner_splits[int(inner_fold) - 1]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("requested exact-inner split is absent from the registry") from exc
        if outer.outer_fold != int(outer_fold) or split.inner_fold != int(inner_fold):
            raise ValueError("requested exact-inner split is absent from the registry")
        return split


@dataclass(frozen=True)
class Stage1FitRow:
    row_id: int
    text: str
    treatment: float
    outcome: float


@dataclass(frozen=True)
class Stage1HeldoutRow:
    """A transform-only row; held-out treatment and outcome are intentionally absent."""

    row_id: int
    text: str


@dataclass(frozen=True)
class ExactInnerStage1FamilyRequest:
    family: str
    outer_fold: int
    inner_fold: int
    split_registry_sha256: str
    split_scope_fingerprint: str
    data_projection_sha256: str
    fit_rows: tuple[Stage1FitRow, ...]
    heldout_rows: tuple[Stage1HeldoutRow, ...]

    @property
    def binding(self) -> dict[str, Any]:
        return {
            "schema_version": EXACT_INNER_REQUEST_VERSION,
            "family": self.family,
            "scope": "inner_train",
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "split_registry_sha256": self.split_registry_sha256,
            "split_scope_fingerprint": self.split_scope_fingerprint,
            "data_projection_sha256": self.data_projection_sha256,
            "fit_row_fingerprint": row_order_fingerprint(
                tuple(row.row_id for row in self.fit_rows)
            ),
            "heldout_row_fingerprint": row_order_fingerprint(
                tuple(row.row_id for row in self.heldout_rows)
            ),
            "fit_row_count": len(self.fit_rows),
            "heldout_row_count": len(self.heldout_rows),
            "heldout_columns": ["_oci_row_id", "text"],
            "heldout_labels_available": False,
        }

    @property
    def binding_sha256(self) -> str:
        return _sha256_json(self.binding)


def exact_inner_data_projection_sha256(
    *,
    fit_rows: Sequence[Stage1FitRow],
    heldout_rows: Sequence[Stage1HeldoutRow],
) -> str:
    """Hash the only data projection an exact-inner producer may receive.

    Production adapters often authenticate a native scope artifact before the
    protocol invokes ``produce``.  Exposing the canonical projection hash keeps
    that pre-binding byte-identical to the request constructed below instead of
    requiring a second implementation of this security boundary.
    """

    fit = tuple(fit_rows)
    heldout = tuple(heldout_rows)
    if not all(isinstance(row, Stage1FitRow) for row in fit):
        raise TypeError("fit_rows must contain Stage1FitRow values")
    if not all(isinstance(row, Stage1HeldoutRow) for row in heldout):
        raise TypeError("heldout_rows must contain Stage1HeldoutRow values")
    projection = {
        "fit_rows": [
            {
                "row_id": row.row_id,
                "text": row.text,
                "treatment": row.treatment,
                "outcome": row.outcome,
            }
            for row in fit
        ],
        "heldout_transform_rows": [{"row_id": row.row_id, "text": row.text} for row in heldout],
        "heldout_labels_included": False,
    }
    return _sha256_json(projection)


@dataclass(frozen=True)
class ExactInnerFamilyEvidenceDraft:
    evidence_payload: Mapping[str, Any]
    evidence_item_count: int
    input_binding_sha256: str
    fit_semantics: str
    fit_audit: Mapping[str, Any]


@runtime_checkable
class ExactInnerStage1FamilyProducer(Protocol):
    """One architecture adapter invoked once per canonical exact-inner scope."""

    def identity(self) -> Mapping[str, Any]: ...

    def produce(
        self,
        request: ExactInnerStage1FamilyRequest,
    ) -> ExactInnerFamilyEvidenceDraft: ...


def _reject_forbidden_evidence(value: Any, *, path: str = "evidence_payload") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if _FORBIDDEN_EVIDENCE_KEY.search(key):
                raise ValueError(f"forbidden oracle/identifier/secret field at {path}.{key}")
            if _FORBIDDEN_REUSE_KEY.search(key):
                raise ValueError(f"forbidden full-outer reuse field at {path}.{key}")
            _reject_forbidden_evidence(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_evidence(child, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if _FORBIDDEN_REUSE_VALUE.search(value):
            raise ValueError(f"forbidden full-outer reuse claim at {path}")
        if _SECRET_VALUE.search(value):
            raise ValueError(f"secret-like value entered {path}")


def _validate_producer_identity(
    raw_identity: Mapping[str, Any],
    *,
    family: str,
) -> dict[str, Any]:
    if not isinstance(raw_identity, Mapping):
        raise TypeError(f"{family} producer identity must be a mapping")
    identity = copy.deepcopy(dict(raw_identity))
    if identity.get("schema_version") != EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION:
        raise ValueError(f"{family} producer has an unsupported identity schema")
    if identity.get("family") != family:
        raise ValueError(f"{family} producer identity changed its architecture family")
    for key in ("producer_name", "producer_version"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            raise ValueError(f"{family} producer identity requires {key}")
    _require_sha256(identity.get("code_sha256"), field_name=f"{family} code_sha256")
    _require_sha256(
        identity.get("configuration_sha256"),
        field_name=f"{family} configuration_sha256",
    )
    _reject_forbidden_evidence(identity, path=f"producer_identity.{family}")
    _canonical_json(identity)
    return identity


def _validate_full_outer_hashes(
    values: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("full_outer_payload_sha256_by_family must be a mapping")
    if set(values) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        missing = sorted(ACTIVE_STAGE1_CONCEPT_FAMILY_SET - set(values))
        extra = sorted(set(values) - ACTIVE_STAGE1_CONCEPT_FAMILY_SET)
        raise ValueError(
            "full-outer payload hash registry must cover exactly all ten families; "
            f"missing={missing} extra={extra}"
        )
    return {
        family: _require_sha256(
            values[family],
            field_name=f"full-outer payload hash for {family}",
        )
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }


def _validate_expected_producer_identity_hashes(
    values: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("expected producer identity hashes must be a mapping")
    if set(values) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        missing = sorted(ACTIVE_STAGE1_CONCEPT_FAMILY_SET - set(values))
        extra = sorted(set(values) - ACTIVE_STAGE1_CONCEPT_FAMILY_SET)
        raise ValueError(
            "expected producer identities must cover exactly all ten families; "
            f"missing={missing} extra={extra}"
        )
    return {
        family: _require_sha256(
            values[family],
            field_name=f"expected producer identity hash for {family}",
        )
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }


def _validate_fit_audit(
    audit: Mapping[str, Any],
    *,
    family: str,
    request_sha256: str,
    split_scope_fingerprint: str,
    fit_semantics: str,
) -> dict[str, Any]:
    if not isinstance(audit, Mapping):
        raise TypeError(f"{family} fit_audit must be a mapping")
    value = copy.deepcopy(dict(audit))
    if value.get("schema_version") != EXACT_INNER_FIT_AUDIT_VERSION:
        raise ValueError(f"{family} fit audit has an unsupported schema")
    if value.get("family") != family:
        raise ValueError(f"{family} fit audit changed its architecture family")
    if value.get("scope") != "inner_train":
        raise ValueError(f"{family} fit audit is not exact-inner evidence")
    if value.get("input_binding_sha256") != request_sha256:
        raise ValueError(f"{family} fit audit is bound to a different request")
    if value.get("split_scope_fingerprint") != split_scope_fingerprint:
        raise ValueError(f"{family} fit audit is bound to a different row scope")
    if value.get("fit_semantics") != fit_semantics:
        raise ValueError(f"{family} fit audit changed its fit semantics")
    for flag in (
        "heldout_labels_accessed",
        "oracle_fields_accessed",
        "secrets_accessed",
    ):
        if value.get(flag) is not False:
            raise ValueError(f"{family} fit audit must attest {flag}=false")
    _require_sha256(
        value.get("fit_execution_sha256"),
        field_name=f"{family} fit_execution_sha256",
    )
    _require_sha256(
        value.get("model_artifact_sha256"),
        field_name=f"{family} model_artifact_sha256",
    )
    if fit_semantics == EXACT_SCOPE_CACHE_REPLAY:
        if value.get("cache_source_scope_fingerprint") != split_scope_fingerprint:
            raise ValueError(f"{family} cache replay came from a different row scope")
        _require_sha256(
            value.get("cache_source_artifact_sha256"),
            field_name=f"{family} cache_source_artifact_sha256",
        )
    elif value.get("cache_source_scope_fingerprint") not in (None, ""):
        raise ValueError(f"{family} fresh refit cannot claim a cache source scope")
    # The three explicitly validated ``*_accessed`` attestation keys contain
    # words that are forbidden in scientific evidence by design.  The audit is
    # a machine envelope, not model-facing evidence, so do not run the evidence
    # key scanner over it.
    _canonical_json(value)
    return value


def _finite_float(value: Any, *, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _build_projected_request_rows(
    *,
    dataset: pd.DataFrame,
    split: CanonicalInnerSplit,
    row_id_column: str,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
) -> tuple[tuple[Stage1FitRow, ...], tuple[Stage1HeldoutRow, ...], str]:
    required = (row_id_column, text_column, treatment_column, outcome_column)
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise ValueError(f"dataset is missing exact-inner input columns: {missing}")
    frame = dataset.reset_index(drop=True)
    row_ids = _canonical_row_ids(frame[row_id_column].tolist(), field_name=row_id_column)
    if len(row_ids) != len(dataset):
        raise ValueError("dataset row registry length mismatch")
    positions = {row_id: index for index, row_id in enumerate(row_ids)}
    requested = set(split.fit_row_ids) | set(split.heldout_row_ids)
    if not requested <= set(positions):
        raise ValueError("canonical exact-inner split contains rows absent from the dataset")

    # Select columns explicitly.  In particular, no oracle or secret-bearing
    # column is projected into either producer request.
    fit_projection = frame.loc[
        [positions[row_id] for row_id in split.fit_row_ids],
        [row_id_column, text_column, treatment_column, outcome_column],
    ]
    heldout_projection = frame.loc[
        [positions[row_id] for row_id in split.heldout_row_ids],
        [row_id_column, text_column],
    ]
    fit_rows = tuple(
        Stage1FitRow(
            row_id=int(row[row_id_column]),
            text="" if pd.isna(row[text_column]) else str(row[text_column]),
            treatment=_finite_float(
                row[treatment_column],
                field_name=f"fit row {int(row[row_id_column])} treatment",
            ),
            outcome=_finite_float(
                row[outcome_column],
                field_name=f"fit row {int(row[row_id_column])} outcome",
            ),
        )
        for _, row in fit_projection.iterrows()
    )
    heldout_rows = tuple(
        Stage1HeldoutRow(
            row_id=int(row[row_id_column]),
            text="" if pd.isna(row[text_column]) else str(row[text_column]),
        )
        for _, row in heldout_projection.iterrows()
    )
    projection_sha256 = exact_inner_data_projection_sha256(
        fit_rows=fit_rows,
        heldout_rows=heldout_rows,
    )
    return fit_rows, heldout_rows, projection_sha256


def produce_exact_inner_stage1_evidence_bundle(
    *,
    dataset: pd.DataFrame,
    registry: CanonicalStage1SplitRegistry,
    outer_fold: int,
    inner_fold: int,
    producers: Mapping[str, ExactInnerStage1FamilyProducer],
    full_outer_payload_sha256_by_family: Mapping[str, str],
    row_id_column: str = "_oci_row_id",
    text_column: str = "clinical_text",
    treatment_column: str = "treatment_indicator",
    outcome_column: str = "outcome_indicator",
) -> dict[str, Any]:
    """Invoke every architecture on one exact scope and return a sealed bundle."""

    if not isinstance(registry, CanonicalStage1SplitRegistry):
        raise TypeError("registry must be CanonicalStage1SplitRegistry")
    if set(producers) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        missing = sorted(ACTIVE_STAGE1_CONCEPT_FAMILY_SET - set(producers))
        extra = sorted(set(producers) - ACTIVE_STAGE1_CONCEPT_FAMILY_SET)
        raise ValueError(
            "exact-inner production requires all ten architecture producers; "
            f"missing={missing} extra={extra}"
        )
    full_outer_hashes = _validate_full_outer_hashes(full_outer_payload_sha256_by_family)
    split = registry.inner_split(outer_fold, inner_fold)
    fit_rows, heldout_rows, projection_sha256 = _build_projected_request_rows(
        dataset=dataset,
        split=split,
        row_id_column=row_id_column,
        text_column=text_column,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
    )

    family_artifacts: list[dict[str, Any]] = []
    producer_identity_hashes: dict[str, str] = {}
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        producer = producers[family]
        if not isinstance(producer, ExactInnerStage1FamilyProducer):
            raise TypeError(f"{family} producer does not implement the exact-inner protocol")
        request = ExactInnerStage1FamilyRequest(
            family=family,
            outer_fold=int(outer_fold),
            inner_fold=int(inner_fold),
            split_registry_sha256=registry.content_sha256,
            split_scope_fingerprint=split.scope_fingerprint,
            data_projection_sha256=projection_sha256,
            fit_rows=fit_rows,
            heldout_rows=heldout_rows,
        )
        identity_before = _validate_producer_identity(producer.identity(), family=family)
        draft = producer.produce(request)
        identity_after = _validate_producer_identity(producer.identity(), family=family)
        if identity_after != identity_before:
            raise RuntimeError(f"{family} producer identity changed during exact-inner fitting")
        producer_identity_hashes[family] = _sha256_json(identity_before)
        if not isinstance(draft, ExactInnerFamilyEvidenceDraft):
            raise TypeError(f"{family} producer returned an unsupported evidence draft")
        if draft.input_binding_sha256 != request.binding_sha256:
            raise ValueError(f"{family} producer returned evidence for a different input scope")
        fit_semantics = str(draft.fit_semantics)
        if fit_semantics not in _VALID_FIT_SEMANTICS:
            raise ValueError(f"{family} producer did not perform an exact-scope fit or replay")
        try:
            evidence_item_count = int(draft.evidence_item_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{family} evidence_item_count must be an integer") from exc
        if evidence_item_count < 1:
            raise ValueError(f"{family} produced no concept-bearing evidence")
        if not isinstance(draft.evidence_payload, Mapping) or not draft.evidence_payload:
            raise ValueError(f"{family} evidence payload must be a non-empty mapping")
        payload = copy.deepcopy(dict(draft.evidence_payload))
        _reject_forbidden_evidence(payload)
        payload_sha256 = _sha256_json(payload)
        if payload_sha256 == full_outer_hashes[family]:
            raise ValueError(
                f"{family} exact-inner payload is byte-semantically identical to its "
                "registered full-outer payload"
            )
        fit_audit = _validate_fit_audit(
            draft.fit_audit,
            family=family,
            request_sha256=request.binding_sha256,
            split_scope_fingerprint=split.scope_fingerprint,
            fit_semantics=fit_semantics,
        )
        artifact = {
            "schema_version": EXACT_INNER_FAMILY_ARTIFACT_VERSION,
            "family": family,
            "scope": "inner_train",
            "outer_fold": int(outer_fold),
            "inner_fold": int(inner_fold),
            "split_registry_sha256": registry.content_sha256,
            "split_scope_fingerprint": split.scope_fingerprint,
            "request_binding_sha256": request.binding_sha256,
            "data_projection_sha256": projection_sha256,
            "fit_row_fingerprint": row_order_fingerprint(split.fit_row_ids),
            "heldout_row_fingerprint": row_order_fingerprint(split.heldout_row_ids),
            "producer_identity": identity_before,
            "producer_identity_sha256": _sha256_json(identity_before),
            "fit_semantics": fit_semantics,
            "fit_audit": fit_audit,
            "fit_audit_sha256": _sha256_json(fit_audit),
            "evidence_item_count": evidence_item_count,
            "evidence_payload": payload,
            "evidence_payload_sha256": payload_sha256,
        }
        artifact["artifact_sha256"] = _sha256_json(artifact)
        family_artifacts.append(artifact)

    bundle = {
        "schema_version": EXACT_INNER_EVIDENCE_BUNDLE_VERSION,
        "scope": "inner_train",
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "split_registry_sha256": registry.content_sha256,
        "split_scope_fingerprint": split.scope_fingerprint,
        "fit_row_ids": list(split.fit_row_ids),
        "heldout_row_ids": list(split.heldout_row_ids),
        "fit_row_fingerprint": row_order_fingerprint(split.fit_row_ids),
        "heldout_row_fingerprint": row_order_fingerprint(split.heldout_row_ids),
        "data_projection_sha256": projection_sha256,
        "heldout_labels_available_to_producers": False,
        "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
        "producer_identity_sha256_by_family": producer_identity_hashes,
        "full_outer_payload_sha256_by_family": full_outer_hashes,
        "family_artifacts": family_artifacts,
    }
    bundle["bundle_sha256"] = _sha256_json(bundle)
    validate_exact_inner_stage1_evidence_bundle(
        bundle,
        registry=registry,
        expected_data_projection_sha256=projection_sha256,
        expected_producer_identity_sha256_by_family=producer_identity_hashes,
        full_outer_payload_sha256_by_family=full_outer_hashes,
    )
    return bundle


def validate_exact_inner_stage1_evidence_bundle(
    bundle: Mapping[str, Any],
    *,
    registry: CanonicalStage1SplitRegistry,
    expected_data_projection_sha256: str,
    expected_producer_identity_sha256_by_family: Mapping[str, str],
    full_outer_payload_sha256_by_family: Mapping[str, str],
) -> None:
    """Fail closed unless an exact-inner bundle is complete and authenticated."""

    if not isinstance(bundle, Mapping):
        raise TypeError("exact-inner evidence bundle must be a mapping")
    value = copy.deepcopy(dict(bundle))
    supplied_bundle_sha256 = _require_sha256(
        value.pop("bundle_sha256", None),
        field_name="bundle_sha256",
    )
    if _sha256_json(value) != supplied_bundle_sha256:
        raise ValueError("exact-inner evidence bundle SHA-256 mismatch")
    if value.get("schema_version") != EXACT_INNER_EVIDENCE_BUNDLE_VERSION:
        raise ValueError("unsupported exact-inner evidence bundle schema")
    if value.get("scope") != "inner_train":
        raise ValueError("exact-inner evidence bundle changed scope")
    outer_fold = int(value.get("outer_fold", 0))
    inner_fold = int(value.get("inner_fold", 0))
    split = registry.inner_split(outer_fold, inner_fold)
    if value.get("split_registry_sha256") != registry.content_sha256:
        raise ValueError("exact-inner bundle is bound to a different split registry")
    if value.get("split_scope_fingerprint") != split.scope_fingerprint:
        raise ValueError("exact-inner bundle is bound to a different row scope")
    if tuple(value.get("fit_row_ids") or ()) != split.fit_row_ids:
        raise ValueError("exact-inner bundle changed canonical fit rows or order")
    if tuple(value.get("heldout_row_ids") or ()) != split.heldout_row_ids:
        raise ValueError("exact-inner bundle changed canonical held-out rows or order")
    if value.get("fit_row_fingerprint") != row_order_fingerprint(split.fit_row_ids):
        raise ValueError("exact-inner bundle fit-row fingerprint mismatch")
    if value.get("heldout_row_fingerprint") != row_order_fingerprint(split.heldout_row_ids):
        raise ValueError("exact-inner bundle held-out-row fingerprint mismatch")
    projection_sha256 = _require_sha256(
        expected_data_projection_sha256,
        field_name="expected_data_projection_sha256",
    )
    if value.get("data_projection_sha256") != projection_sha256:
        raise ValueError("exact-inner bundle changed its projected data bytes")
    if value.get("heldout_labels_available_to_producers") is not False:
        raise ValueError("exact-inner producers must not receive held-out labels")
    if tuple(value.get("architecture_order") or ()) != ACTIVE_STAGE1_CONCEPT_FAMILIES:
        raise ValueError("exact-inner bundle changed the ten-family architecture order")
    expected_identity_hashes = _validate_expected_producer_identity_hashes(
        expected_producer_identity_sha256_by_family
    )
    if value.get("producer_identity_sha256_by_family") != expected_identity_hashes:
        raise ValueError("exact-inner bundle changed its authenticated producer identities")
    full_outer_hashes = _validate_full_outer_hashes(full_outer_payload_sha256_by_family)
    if value.get("full_outer_payload_sha256_by_family") != full_outer_hashes:
        raise ValueError("exact-inner bundle changed its registered full-outer references")
    artifacts = value.get("family_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(ACTIVE_STAGE1_CONCEPT_FAMILIES):
        raise ValueError("exact-inner bundle must contain exactly ten family artifacts")
    if tuple(item.get("family") for item in artifacts if isinstance(item, Mapping)) != (
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    ):
        raise ValueError("exact-inner family artifacts are missing, duplicated, or reordered")

    for family, raw_artifact in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, artifacts):
        if not isinstance(raw_artifact, Mapping):
            raise TypeError(f"{family} artifact must be a mapping")
        artifact = copy.deepcopy(dict(raw_artifact))
        supplied_artifact_sha256 = _require_sha256(
            artifact.pop("artifact_sha256", None),
            field_name=f"{family} artifact_sha256",
        )
        if _sha256_json(artifact) != supplied_artifact_sha256:
            raise ValueError(f"{family} artifact SHA-256 mismatch")
        if artifact.get("schema_version") != EXACT_INNER_FAMILY_ARTIFACT_VERSION:
            raise ValueError(f"{family} artifact has an unsupported schema")
        if artifact.get("family") != family or artifact.get("scope") != "inner_train":
            raise ValueError(f"{family} artifact changed family or scope")
        if (
            int(artifact.get("outer_fold", 0)) != outer_fold
            or int(artifact.get("inner_fold", 0)) != inner_fold
        ):
            raise ValueError(f"{family} artifact changed its fold identity")
        if artifact.get("split_registry_sha256") != registry.content_sha256:
            raise ValueError(f"{family} artifact changed its split registry")
        if artifact.get("split_scope_fingerprint") != split.scope_fingerprint:
            raise ValueError(f"{family} artifact changed its exact row scope")
        if artifact.get("data_projection_sha256") != projection_sha256:
            raise ValueError(f"{family} artifact changed its projected input bytes")
        if artifact.get("fit_row_fingerprint") != row_order_fingerprint(split.fit_row_ids):
            raise ValueError(f"{family} artifact changed exact fit rows")
        if artifact.get("heldout_row_fingerprint") != row_order_fingerprint(split.heldout_row_ids):
            raise ValueError(f"{family} artifact changed exact held-out rows")
        identity = _validate_producer_identity(
            artifact.get("producer_identity") or {},
            family=family,
        )
        if artifact.get("producer_identity_sha256") != _sha256_json(identity):
            raise ValueError(f"{family} producer identity SHA-256 mismatch")
        if artifact.get("producer_identity_sha256") != expected_identity_hashes[family]:
            raise ValueError(f"{family} producer identity is not the authenticated identity")
        fit_semantics = str(artifact.get("fit_semantics") or "")
        if fit_semantics not in _VALID_FIT_SEMANTICS:
            raise ValueError(f"{family} artifact lacks exact-scope fit semantics")
        request_binding = {
            "schema_version": EXACT_INNER_REQUEST_VERSION,
            "family": family,
            "scope": "inner_train",
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "split_registry_sha256": registry.content_sha256,
            "split_scope_fingerprint": split.scope_fingerprint,
            "data_projection_sha256": projection_sha256,
            "fit_row_fingerprint": row_order_fingerprint(split.fit_row_ids),
            "heldout_row_fingerprint": row_order_fingerprint(split.heldout_row_ids),
            "fit_row_count": len(split.fit_row_ids),
            "heldout_row_count": len(split.heldout_row_ids),
            "heldout_columns": ["_oci_row_id", "text"],
            "heldout_labels_available": False,
        }
        request_sha256 = _sha256_json(request_binding)
        if artifact.get("request_binding_sha256") != request_sha256:
            raise ValueError(f"{family} artifact changed its request binding")
        fit_audit = _validate_fit_audit(
            artifact.get("fit_audit") or {},
            family=family,
            request_sha256=request_sha256,
            split_scope_fingerprint=split.scope_fingerprint,
            fit_semantics=fit_semantics,
        )
        if artifact.get("fit_audit_sha256") != _sha256_json(fit_audit):
            raise ValueError(f"{family} fit audit SHA-256 mismatch")
        if int(artifact.get("evidence_item_count", 0)) < 1:
            raise ValueError(f"{family} artifact has no concept-bearing evidence")
        payload = artifact.get("evidence_payload")
        if not isinstance(payload, Mapping) or not payload:
            raise ValueError(f"{family} artifact has an empty evidence payload")
        _reject_forbidden_evidence(payload)
        payload_sha256 = _sha256_json(payload)
        if artifact.get("evidence_payload_sha256") != payload_sha256:
            raise ValueError(f"{family} evidence payload SHA-256 mismatch")
        if payload_sha256 == full_outer_hashes[family]:
            raise ValueError(
                f"{family} exact-inner payload is byte-semantically identical to its "
                "registered full-outer payload"
            )


__all__ = [
    "CANONICAL_STAGE1_SPLIT_REGISTRY_VERSION",
    "EXACT_INNER_EVIDENCE_BUNDLE_VERSION",
    "EXACT_INNER_FAMILY_ARTIFACT_VERSION",
    "EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION",
    "EXACT_INNER_FIT_AUDIT_VERSION",
    "EXACT_INNER_REFIT",
    "EXACT_SCOPE_CACHE_REPLAY",
    "CanonicalInnerSplit",
    "CanonicalOuterSplit",
    "CanonicalStage1SplitRegistry",
    "ExactInnerFamilyEvidenceDraft",
    "ExactInnerStage1FamilyProducer",
    "ExactInnerStage1FamilyRequest",
    "Stage1FitRow",
    "Stage1HeldoutRow",
    "exact_inner_data_projection_sha256",
    "produce_exact_inner_stage1_evidence_bundle",
    "row_order_fingerprint",
    "validate_exact_inner_stage1_evidence_bundle",
]
