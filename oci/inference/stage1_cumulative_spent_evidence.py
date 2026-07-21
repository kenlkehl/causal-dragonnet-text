"""Typed native-producer boundary for canonical cumulative-spent Stage 1 scopes.

Exact-inner Stage 1 fits may transform registered held-out text (without held-out
labels).  Hierarchical review has a stricter boundary: a producer receives the
text, treatment, and outcome only for rows already spent by the canonical
schedule, plus the integer IDs of rows that remain sealed.  Sealed text and
labels never enter the producer request.

This module makes that distinction structural.  It invokes all ten architecture
producers on one exact cumulative-spent scope, authenticates their identities,
fit audits, and evidence payloads, and emits a closed bundle.  The production
wrapper can then convert the native artifacts into the registered role-neutral
catalog and hierarchy proof/index graph without reconstructing a different
runtime schedule.
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

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)
from .stage1_exact_inner_evidence import (
    EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
    Stage1FitRow,
    row_order_fingerprint,
)

CUMULATIVE_SPENT_REQUEST_SCHEMA = "production_stage1_hierarchy_spent_family_request_v1"
CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA = "cumulative_spent_stage1_family_fit_audit_v1"
CUMULATIVE_SPENT_FAMILY_ARTIFACT_SCHEMA = "cumulative_spent_stage1_family_artifact_v1"
CUMULATIVE_SPENT_EVIDENCE_BUNDLE_SCHEMA = "cumulative_spent_stage1_evidence_bundle_v1"

CUMULATIVE_SPENT_REFIT = "exact_cumulative_spent_refit"
CUMULATIVE_SPENT_CACHE_REPLAY = "exact_cumulative_spent_authenticated_cache_replay"
_VALID_FIT_SEMANTICS = frozenset({CUMULATIVE_SPENT_REFIT, CUMULATIVE_SPENT_CACHE_REPLAY})
_TFIDF_FAMILIES = frozenset({TFIDF_SEMANTIC_RETRIEVAL, TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCOPE_ID = re.compile(r"^outer_[0-9]{3}_hierarchy_epoch_[0-9]{3}$")
_FORBIDDEN_EVIDENCE_KEY = re.compile(
    r"(?:^|_)(?:"
    r"oracle|ground_truth|true_ite|true_cate|true_effect|"
    r"row_id|row_ids|patient_id|patient_ids|mrn|medical_record_number|"
    r"api_key|access_token|refresh_token|password|passwd|secret|credential|"
    r"authorization"
    r")(?:_|$)",
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


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, field_name: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return text


def _ordered_unique_row_ids(values: Sequence[Any], *, field_name: str) -> tuple[int, ...]:
    rows: list[int] = []
    for raw in values:
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"{field_name} cannot contain boolean row IDs")
        try:
            row_id = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain integer row IDs") from exc
        if row_id < 0 or row_id != raw:
            raise ValueError(f"{field_name} must contain non-negative integer row IDs")
        rows.append(row_id)
    if not rows:
        raise ValueError(f"{field_name} cannot be empty")
    if len(rows) != len(set(rows)):
        raise ValueError(f"{field_name} must contain unique row IDs")
    return tuple(rows)


def _reject_forbidden_evidence(value: Any, *, path: str = "evidence_payload") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if _FORBIDDEN_EVIDENCE_KEY.search(key):
                raise ValueError(f"forbidden identifier/oracle/secret field at {path}.{key}")
            _reject_forbidden_evidence(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_evidence(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and _SECRET_VALUE.search(value):
        raise ValueError(f"secret-like value entered {path}")


def cumulative_spent_data_projection_sha256(
    *,
    outer_fold: int,
    context_epoch: int,
    spent_rows: Sequence[Stage1FitRow],
    sealed_row_ids: Sequence[int],
) -> str:
    """Hash the only row-level data projection a cumulative producer may see."""

    fit = tuple(spent_rows)
    if not fit or not all(isinstance(row, Stage1FitRow) for row in fit):
        raise TypeError("spent_rows must contain Stage1FitRow values")
    spent_ids = _ordered_unique_row_ids(
        tuple(row.row_id for row in fit),
        field_name="spent row IDs",
    )
    sealed = _ordered_unique_row_ids(sealed_row_ids, field_name="sealed_row_ids")
    if set(spent_ids) & set(sealed):
        raise ValueError("spent and sealed hierarchy rows overlap")
    if any(not isinstance(row.text, str) or not row.text.strip() for row in fit):
        raise ValueError("spent hierarchy text must be explicit and non-empty")
    if any(
        not math.isfinite(float(value)) for row in fit for value in (row.treatment, row.outcome)
    ):
        raise ValueError("spent hierarchy labels must be finite")
    body = {
        "schema_version": CUMULATIVE_SPENT_REQUEST_SCHEMA,
        "outer_fold": int(outer_fold),
        "context_epoch": int(context_epoch),
        "spent_rows": [
            {
                "row_id": row.row_id,
                "text": row.text,
                "treatment": float(row.treatment),
                "outcome": float(row.outcome),
            }
            for row in fit
        ],
        "sealed_row_ids_only": list(sealed),
        "sealed_text_available": False,
        "sealed_labels_available": False,
    }
    return _sha(body)


@dataclass(frozen=True)
class CumulativeSpentStage1FamilyRequest:
    family: str
    request_sha256: str
    schedule_sha256: str
    scope_id: str
    outer_fold: int
    context_epoch: int
    provider_inner_fold: int
    split_scope_fingerprint: str
    data_projection_sha256: str
    spent_rows: tuple[Stage1FitRow, ...]
    sealed_row_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("family is not an active Stage 1 architecture")
        _require_sha256(self.request_sha256, field_name="request_sha256")
        _require_sha256(self.schedule_sha256, field_name="schedule_sha256")
        _require_sha256(
            self.split_scope_fingerprint,
            field_name="split_scope_fingerprint",
        )
        _require_sha256(
            self.data_projection_sha256,
            field_name="data_projection_sha256",
        )
        outer = int(self.outer_fold)
        epoch = int(self.context_epoch)
        provider_fold = int(self.provider_inner_fold)
        if outer < 1 or epoch < 0 or provider_fold != epoch + 1:
            raise ValueError("cumulative-spent fold identity is invalid")
        expected_scope_id = f"outer_{outer:03d}_hierarchy_epoch_{epoch:03d}"
        if self.scope_id != expected_scope_id or _SCOPE_ID.fullmatch(self.scope_id) is None:
            raise ValueError("cumulative-spent scope_id is not canonical")
        spent = tuple(self.spent_rows)
        sealed = _ordered_unique_row_ids(self.sealed_row_ids, field_name="sealed_row_ids")
        expected_projection = cumulative_spent_data_projection_sha256(
            outer_fold=outer,
            context_epoch=epoch,
            spent_rows=spent,
            sealed_row_ids=sealed,
        )
        if expected_projection != self.data_projection_sha256:
            raise ValueError("cumulative-spent request changed its projected data")
        object.__setattr__(self, "outer_fold", outer)
        object.__setattr__(self, "context_epoch", epoch)
        object.__setattr__(self, "provider_inner_fold", provider_fold)
        object.__setattr__(self, "spent_rows", spent)
        object.__setattr__(self, "sealed_row_ids", sealed)

    @property
    def spent_row_ids(self) -> tuple[int, ...]:
        return tuple(row.row_id for row in self.spent_rows)

    @property
    def binding(self) -> dict[str, Any]:
        return {
            "schema_version": CUMULATIVE_SPENT_REQUEST_SCHEMA,
            "request_sha256": self.request_sha256,
            "schedule_sha256": self.schedule_sha256,
            "scope_id": self.scope_id,
            "outer_fold": self.outer_fold,
            "context_epoch": self.context_epoch,
            "provider_inner_fold": self.provider_inner_fold,
            "split_fingerprint": self.split_scope_fingerprint,
            "spent_row_order_fingerprint": row_order_fingerprint(self.spent_row_ids),
            "sealed_row_order_fingerprint": row_order_fingerprint(self.sealed_row_ids),
            "data_projection_sha256": self.data_projection_sha256,
            "sealed_text_available": False,
            "sealed_labels_available": False,
        }

    @property
    def binding_sha256(self) -> str:
        return _sha(self.binding)


@dataclass(frozen=True)
class CumulativeSpentFamilyEvidenceDraft:
    evidence_payload: Mapping[str, Any]
    evidence_item_count: int
    input_binding_sha256: str
    fit_semantics: str
    fit_audit: Mapping[str, Any]


@runtime_checkable
class CumulativeSpentStage1FamilyProducer(Protocol):
    """One genuine architecture producer for one canonical spent scope."""

    def identity(self) -> Mapping[str, Any]: ...

    def produce_cumulative_spent(
        self,
        request: CumulativeSpentStage1FamilyRequest,
    ) -> CumulativeSpentFamilyEvidenceDraft: ...


def _validate_producer_identity(value: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{family} producer identity must be a mapping")
    identity = copy.deepcopy(dict(value))
    expected_keys = {
        "schema_version",
        "family",
        "producer_name",
        "producer_version",
        "code_sha256",
        "configuration_sha256",
    }
    if set(identity) != expected_keys:
        raise ValueError(f"{family} producer identity is not a closed schema")
    if (
        identity.get("schema_version") != EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION
        or identity.get("family") != family
    ):
        raise ValueError(f"{family} producer identity changed its family or schema")
    for key in ("producer_name", "producer_version"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            raise ValueError(f"{family} producer identity requires {key}")
    _require_sha256(identity.get("code_sha256"), field_name=f"{family} code_sha256")
    _require_sha256(
        identity.get("configuration_sha256"),
        field_name=f"{family} configuration_sha256",
    )
    _canonical_json(identity)
    return identity


def _validate_fit_audit(
    value: Mapping[str, Any],
    *,
    family: str,
    input_binding_sha256: str,
    scope_id: str,
    split_scope_fingerprint: str,
    fit_semantics: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{family} cumulative-spent fit audit must be a mapping")
    audit = copy.deepcopy(dict(value))
    expected_keys = {
        "schema_version",
        "family",
        "scope",
        "scope_id",
        "input_binding_sha256",
        "split_scope_fingerprint",
        "fit_semantics",
        "fit_execution_sha256",
        "model_artifact_sha256",
        "source_artifact_sha256",
        "sealed_text_accessed",
        "sealed_labels_accessed",
        "oracle_fields_accessed",
        "secrets_accessed",
        "cache_source_scope_fingerprint",
        "cache_source_artifact_sha256",
        "tfidf_training_scope_policy",
    }
    if set(audit) != expected_keys:
        raise ValueError(f"{family} cumulative-spent fit audit is not a closed schema")
    if (
        audit.get("schema_version") != CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA
        or audit.get("family") != family
        or audit.get("scope") != "cumulative_spent_train"
        or audit.get("scope_id") != scope_id
        or audit.get("input_binding_sha256") != input_binding_sha256
        or audit.get("split_scope_fingerprint") != split_scope_fingerprint
        or audit.get("fit_semantics") != fit_semantics
    ):
        raise ValueError(f"{family} cumulative-spent fit audit changed its request binding")
    for flag in (
        "sealed_text_accessed",
        "sealed_labels_accessed",
        "oracle_fields_accessed",
        "secrets_accessed",
    ):
        if audit.get(flag) is not False:
            raise ValueError(f"{family} cumulative-spent fit audit must attest {flag}=false")
    for key in (
        "fit_execution_sha256",
        "model_artifact_sha256",
        "source_artifact_sha256",
    ):
        _require_sha256(audit.get(key), field_name=f"{family} {key}")
    if fit_semantics == CUMULATIVE_SPENT_CACHE_REPLAY:
        if audit.get("cache_source_scope_fingerprint") != split_scope_fingerprint:
            raise ValueError(f"{family} cache replay came from another cumulative scope")
        _require_sha256(
            audit.get("cache_source_artifact_sha256"),
            field_name=f"{family} cache_source_artifact_sha256",
        )
    elif (
        audit.get("cache_source_scope_fingerprint") is not None
        or audit.get("cache_source_artifact_sha256") is not None
    ):
        raise ValueError(f"{family} fresh cumulative refit cannot claim a cache source")
    policy = audit.get("tfidf_training_scope_policy")
    if family in _TFIDF_FAMILIES:
        if not isinstance(policy, Mapping) or not policy:
            raise TypeError(f"{family} must component-emit a TF-IDF training-scope policy")
    elif policy is not None:
        raise ValueError(f"{family} cannot claim a TF-IDF training-scope policy")
    _canonical_json(audit)
    return audit


def _project_request_rows(
    *,
    dataset: pd.DataFrame,
    spent_row_ids: Sequence[int],
    sealed_row_ids: Sequence[int],
    row_id_column: str,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
) -> tuple[tuple[Stage1FitRow, ...], tuple[int, ...]]:
    required = (row_id_column, text_column, treatment_column, outcome_column)
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise ValueError(f"dataset is missing cumulative-spent input columns: {missing}")
    frame = dataset.reset_index(drop=True)
    dataset_ids = _ordered_unique_row_ids(
        frame[row_id_column].tolist(),
        field_name=row_id_column,
    )
    positions = {row_id: index for index, row_id in enumerate(dataset_ids)}
    spent = _ordered_unique_row_ids(spent_row_ids, field_name="spent_row_ids")
    sealed = _ordered_unique_row_ids(sealed_row_ids, field_name="sealed_row_ids")
    if set(spent) & set(sealed):
        raise ValueError("spent and sealed hierarchy rows overlap")
    if not (set(spent) | set(sealed)) <= set(dataset_ids):
        raise ValueError("spent or sealed hierarchy rows are absent from the supplied cohort")
    projection = frame.loc[
        [positions[row_id] for row_id in spent],
        [row_id_column, text_column, treatment_column, outcome_column],
    ]
    fit_rows: list[Stage1FitRow] = []
    for _, row in projection.iterrows():
        row_id = int(row[row_id_column])
        text = "" if pd.isna(row[text_column]) else str(row[text_column])
        try:
            treatment = float(row[treatment_column])
            outcome = float(row[outcome_column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"spent row {row_id} labels must be numeric") from exc
        if not text.strip() or not math.isfinite(treatment) or not math.isfinite(outcome):
            raise ValueError(f"spent row {row_id} must have explicit text and finite labels")
        fit_rows.append(
            Stage1FitRow(
                row_id=row_id,
                text=text,
                treatment=treatment,
                outcome=outcome,
            )
        )
    return tuple(fit_rows), sealed


def produce_cumulative_spent_stage1_evidence_bundle(
    *,
    dataset: pd.DataFrame,
    request_sha256: str,
    schedule_sha256: str,
    scope_id: str,
    outer_fold: int,
    context_epoch: int,
    provider_inner_fold: int,
    split_scope_fingerprint: str,
    spent_row_ids: Sequence[int],
    sealed_row_ids: Sequence[int],
    producers: Mapping[str, CumulativeSpentStage1FamilyProducer],
    row_id_column: str = "_oci_row_id",
    text_column: str = "clinical_text",
    treatment_column: str = "treatment_indicator",
    outcome_column: str = "outcome_indicator",
) -> dict[str, Any]:
    """Run and authenticate all ten producers on one canonical spent scope."""

    if set(producers) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        missing = sorted(ACTIVE_STAGE1_CONCEPT_FAMILY_SET - set(producers))
        extra = sorted(set(producers) - ACTIVE_STAGE1_CONCEPT_FAMILY_SET)
        raise ValueError(
            "cumulative-spent production requires all ten architecture producers; "
            f"missing={missing} extra={extra}"
        )
    fit_rows, sealed = _project_request_rows(
        dataset=dataset,
        spent_row_ids=spent_row_ids,
        sealed_row_ids=sealed_row_ids,
        row_id_column=row_id_column,
        text_column=text_column,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
    )
    projection_sha256 = cumulative_spent_data_projection_sha256(
        outer_fold=outer_fold,
        context_epoch=context_epoch,
        spent_rows=fit_rows,
        sealed_row_ids=sealed,
    )
    family_artifacts: list[dict[str, Any]] = []
    producer_hashes: dict[str, str] = {}
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        producer = producers[family]
        if not isinstance(producer, CumulativeSpentStage1FamilyProducer):
            raise TypeError(f"{family} producer does not implement the cumulative-spent protocol")
        request = CumulativeSpentStage1FamilyRequest(
            family=family,
            request_sha256=request_sha256,
            schedule_sha256=schedule_sha256,
            scope_id=scope_id,
            outer_fold=outer_fold,
            context_epoch=context_epoch,
            provider_inner_fold=provider_inner_fold,
            split_scope_fingerprint=split_scope_fingerprint,
            data_projection_sha256=projection_sha256,
            spent_rows=fit_rows,
            sealed_row_ids=sealed,
        )
        identity_before = _validate_producer_identity(producer.identity(), family=family)
        draft = producer.produce_cumulative_spent(request)
        identity_after = _validate_producer_identity(producer.identity(), family=family)
        if identity_after != identity_before:
            raise RuntimeError(f"{family} producer identity changed during cumulative fitting")
        if not isinstance(draft, CumulativeSpentFamilyEvidenceDraft):
            raise TypeError(f"{family} producer returned an unsupported cumulative draft")
        if draft.input_binding_sha256 != request.binding_sha256:
            raise ValueError(f"{family} producer returned evidence for another spent scope")
        fit_semantics = str(draft.fit_semantics)
        if fit_semantics not in _VALID_FIT_SEMANTICS:
            raise ValueError(f"{family} did not perform an exact cumulative fit or replay")
        try:
            item_count = int(draft.evidence_item_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{family} evidence_item_count must be an integer") from exc
        if (
            isinstance(draft.evidence_item_count, (bool, np.bool_))
            or item_count != draft.evidence_item_count
            or item_count < 1
        ):
            raise ValueError(f"{family} produced no concept-bearing cumulative evidence")
        if not isinstance(draft.evidence_payload, Mapping) or not draft.evidence_payload:
            raise ValueError(f"{family} cumulative evidence payload must be non-empty")
        payload = copy.deepcopy(dict(draft.evidence_payload))
        _reject_forbidden_evidence(payload)
        payload_sha256 = _sha(payload)
        audit = _validate_fit_audit(
            draft.fit_audit,
            family=family,
            input_binding_sha256=request.binding_sha256,
            scope_id=request.scope_id,
            split_scope_fingerprint=request.split_scope_fingerprint,
            fit_semantics=fit_semantics,
        )
        identity_sha256 = _sha(identity_before)
        producer_hashes[family] = identity_sha256
        artifact = {
            "schema_version": CUMULATIVE_SPENT_FAMILY_ARTIFACT_SCHEMA,
            "family": family,
            "scope": "cumulative_spent_train",
            "scope_id": scope_id,
            "outer_fold": int(outer_fold),
            "context_epoch": int(context_epoch),
            "provider_inner_fold": int(provider_inner_fold),
            "schedule_sha256": schedule_sha256,
            "split_scope_fingerprint": split_scope_fingerprint,
            "request_binding_sha256": request.binding_sha256,
            "data_projection_sha256": projection_sha256,
            "spent_row_order_fingerprint": row_order_fingerprint(request.spent_row_ids),
            "sealed_row_order_fingerprint": row_order_fingerprint(sealed),
            "producer_identity": identity_before,
            "producer_identity_sha256": identity_sha256,
            "fit_semantics": fit_semantics,
            "fit_audit": audit,
            "fit_audit_sha256": _sha(audit),
            "evidence_item_count": item_count,
            "evidence_payload": payload,
            "evidence_payload_sha256": payload_sha256,
        }
        artifact["artifact_sha256"] = _sha(artifact)
        family_artifacts.append(artifact)

    spent_ids = tuple(row.row_id for row in fit_rows)
    bundle = {
        "schema_version": CUMULATIVE_SPENT_EVIDENCE_BUNDLE_SCHEMA,
        "scope": "cumulative_spent_train",
        "scope_id": scope_id,
        "outer_fold": int(outer_fold),
        "context_epoch": int(context_epoch),
        "provider_inner_fold": int(provider_inner_fold),
        "request_sha256": request_sha256,
        "schedule_sha256": schedule_sha256,
        "split_scope_fingerprint": split_scope_fingerprint,
        "spent_row_ids": list(spent_ids),
        "sealed_row_ids": list(sealed),
        "spent_row_order_fingerprint": row_order_fingerprint(spent_ids),
        "sealed_row_order_fingerprint": row_order_fingerprint(sealed),
        "data_projection_sha256": projection_sha256,
        "sealed_text_available_to_producers": False,
        "sealed_labels_available_to_producers": False,
        "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
        "producer_identity_sha256_by_family": producer_hashes,
        "family_artifacts": family_artifacts,
    }
    bundle["bundle_sha256"] = _sha(bundle)
    validate_cumulative_spent_stage1_evidence_bundle(
        bundle,
        expected_request_sha256=request_sha256,
        expected_schedule_sha256=schedule_sha256,
        expected_scope_id=scope_id,
        expected_split_scope_fingerprint=split_scope_fingerprint,
        expected_spent_row_ids=spent_ids,
        expected_sealed_row_ids=sealed,
        expected_data_projection_sha256=projection_sha256,
        expected_producer_identity_sha256_by_family=producer_hashes,
    )
    return bundle


def validate_cumulative_spent_stage1_evidence_bundle(
    bundle: Mapping[str, Any],
    *,
    expected_request_sha256: str,
    expected_schedule_sha256: str,
    expected_scope_id: str,
    expected_split_scope_fingerprint: str,
    expected_spent_row_ids: Sequence[int],
    expected_sealed_row_ids: Sequence[int],
    expected_data_projection_sha256: str,
    expected_producer_identity_sha256_by_family: Mapping[str, str],
) -> None:
    """Fail closed unless a cumulative-spent bundle is complete and immutable."""

    if not isinstance(bundle, Mapping):
        raise TypeError("cumulative-spent evidence bundle must be a mapping")
    value = copy.deepcopy(dict(bundle))
    supplied_hash = _require_sha256(
        value.pop("bundle_sha256", None),
        field_name="bundle_sha256",
    )
    if _sha(value) != supplied_hash:
        raise ValueError("cumulative-spent evidence bundle SHA-256 mismatch")
    expected_bundle_keys = {
        "schema_version",
        "scope",
        "scope_id",
        "outer_fold",
        "context_epoch",
        "provider_inner_fold",
        "request_sha256",
        "schedule_sha256",
        "split_scope_fingerprint",
        "spent_row_ids",
        "sealed_row_ids",
        "spent_row_order_fingerprint",
        "sealed_row_order_fingerprint",
        "data_projection_sha256",
        "sealed_text_available_to_producers",
        "sealed_labels_available_to_producers",
        "architecture_order",
        "producer_identity_sha256_by_family",
        "family_artifacts",
    }
    if set(value) != expected_bundle_keys:
        raise ValueError("cumulative-spent evidence bundle is not a closed schema")
    expected_request = _require_sha256(
        expected_request_sha256,
        field_name="expected_request_sha256",
    )
    expected_schedule = _require_sha256(
        expected_schedule_sha256,
        field_name="expected_schedule_sha256",
    )
    expected_split = _require_sha256(
        expected_split_scope_fingerprint,
        field_name="expected_split_scope_fingerprint",
    )
    expected_projection = _require_sha256(
        expected_data_projection_sha256,
        field_name="expected_data_projection_sha256",
    )
    spent = _ordered_unique_row_ids(expected_spent_row_ids, field_name="expected spent rows")
    sealed = _ordered_unique_row_ids(expected_sealed_row_ids, field_name="expected sealed rows")
    if set(spent) & set(sealed):
        raise ValueError("expected spent and sealed rows overlap")
    outer_fold = int(value.get("outer_fold", 0))
    context_epoch = int(value.get("context_epoch", -1))
    provider_inner_fold = int(value.get("provider_inner_fold", 0))
    if (
        value.get("schema_version") != CUMULATIVE_SPENT_EVIDENCE_BUNDLE_SCHEMA
        or value.get("scope") != "cumulative_spent_train"
        or value.get("scope_id") != expected_scope_id
        or expected_scope_id != f"outer_{outer_fold:03d}_hierarchy_epoch_{context_epoch:03d}"
        or outer_fold < 1
        or context_epoch < 0
        or provider_inner_fold != context_epoch + 1
        or value.get("request_sha256") != expected_request
        or value.get("schedule_sha256") != expected_schedule
        or value.get("split_scope_fingerprint") != expected_split
        or tuple(value.get("spent_row_ids") or ()) != spent
        or tuple(value.get("sealed_row_ids") or ()) != sealed
        or value.get("spent_row_order_fingerprint") != row_order_fingerprint(spent)
        or value.get("sealed_row_order_fingerprint") != row_order_fingerprint(sealed)
        or value.get("data_projection_sha256") != expected_projection
        or value.get("sealed_text_available_to_producers") is not False
        or value.get("sealed_labels_available_to_producers") is not False
        or tuple(value.get("architecture_order") or ()) != ACTIVE_STAGE1_CONCEPT_FAMILIES
    ):
        raise ValueError("cumulative-spent bundle changed its canonical scope or security binding")
    expected_identity_hashes = {
        family: _require_sha256(
            expected_producer_identity_sha256_by_family.get(family),
            field_name=f"expected producer identity for {family}",
        )
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    if set(expected_producer_identity_sha256_by_family) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError("expected producer identities must cover exactly all ten families")
    if value.get("producer_identity_sha256_by_family") != expected_identity_hashes:
        raise ValueError("cumulative-spent bundle changed producer identities")
    artifacts = value.get("family_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(ACTIVE_STAGE1_CONCEPT_FAMILIES):
        raise ValueError("cumulative-spent bundle must contain exactly ten family artifacts")
    if (
        tuple(artifact.get("family") for artifact in artifacts if isinstance(artifact, Mapping))
        != ACTIVE_STAGE1_CONCEPT_FAMILIES
    ):
        raise ValueError("cumulative-spent family artifacts are missing, duplicated, or reordered")

    common_binding = {
        "schema_version": CUMULATIVE_SPENT_REQUEST_SCHEMA,
        "request_sha256": expected_request,
        "schedule_sha256": expected_schedule,
        "scope_id": expected_scope_id,
        "outer_fold": outer_fold,
        "context_epoch": context_epoch,
        "provider_inner_fold": provider_inner_fold,
        "split_fingerprint": expected_split,
        "spent_row_order_fingerprint": row_order_fingerprint(spent),
        "sealed_row_order_fingerprint": row_order_fingerprint(sealed),
        "data_projection_sha256": expected_projection,
        "sealed_text_available": False,
        "sealed_labels_available": False,
    }
    binding_sha256 = _sha(common_binding)
    for family, raw_artifact in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, artifacts):
        if not isinstance(raw_artifact, Mapping):
            raise TypeError(f"{family} cumulative artifact must be a mapping")
        artifact = copy.deepcopy(dict(raw_artifact))
        artifact_sha256 = _require_sha256(
            artifact.pop("artifact_sha256", None),
            field_name=f"{family} artifact_sha256",
        )
        if _sha(artifact) != artifact_sha256:
            raise ValueError(f"{family} cumulative artifact SHA-256 mismatch")
        expected_artifact_keys = {
            "schema_version",
            "family",
            "scope",
            "scope_id",
            "outer_fold",
            "context_epoch",
            "provider_inner_fold",
            "schedule_sha256",
            "split_scope_fingerprint",
            "request_binding_sha256",
            "data_projection_sha256",
            "spent_row_order_fingerprint",
            "sealed_row_order_fingerprint",
            "producer_identity",
            "producer_identity_sha256",
            "fit_semantics",
            "fit_audit",
            "fit_audit_sha256",
            "evidence_item_count",
            "evidence_payload",
            "evidence_payload_sha256",
        }
        if set(artifact) != expected_artifact_keys:
            raise ValueError(f"{family} cumulative artifact is not a closed schema")
        if (
            artifact.get("schema_version") != CUMULATIVE_SPENT_FAMILY_ARTIFACT_SCHEMA
            or artifact.get("family") != family
            or artifact.get("scope") != "cumulative_spent_train"
            or artifact.get("scope_id") != expected_scope_id
            or int(artifact.get("outer_fold", 0)) != outer_fold
            or int(artifact.get("context_epoch", -1)) != context_epoch
            or int(artifact.get("provider_inner_fold", 0)) != provider_inner_fold
            or artifact.get("schedule_sha256") != expected_schedule
            or artifact.get("split_scope_fingerprint") != expected_split
            or artifact.get("request_binding_sha256") != binding_sha256
            or artifact.get("data_projection_sha256") != expected_projection
            or artifact.get("spent_row_order_fingerprint") != row_order_fingerprint(spent)
            or artifact.get("sealed_row_order_fingerprint") != row_order_fingerprint(sealed)
        ):
            raise ValueError(f"{family} cumulative artifact changed its scope binding")
        identity = _validate_producer_identity(
            artifact.get("producer_identity") or {},
            family=family,
        )
        if (
            artifact.get("producer_identity_sha256") != _sha(identity)
            or artifact.get("producer_identity_sha256") != expected_identity_hashes[family]
        ):
            raise ValueError(f"{family} cumulative producer identity hash mismatch")
        fit_semantics = str(artifact.get("fit_semantics") or "")
        if fit_semantics not in _VALID_FIT_SEMANTICS:
            raise ValueError(f"{family} cumulative artifact lacks exact fit semantics")
        audit = _validate_fit_audit(
            artifact.get("fit_audit") or {},
            family=family,
            input_binding_sha256=binding_sha256,
            scope_id=expected_scope_id,
            split_scope_fingerprint=expected_split,
            fit_semantics=fit_semantics,
        )
        if artifact.get("fit_audit_sha256") != _sha(audit):
            raise ValueError(f"{family} cumulative fit audit SHA-256 mismatch")
        raw_item_count = artifact.get("evidence_item_count")
        try:
            item_count = int(raw_item_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{family} cumulative evidence count is invalid") from exc
        if (
            isinstance(raw_item_count, (bool, np.bool_))
            or item_count != raw_item_count
            or item_count < 1
        ):
            raise ValueError(f"{family} cumulative artifact has no evidence")
        payload = artifact.get("evidence_payload")
        if not isinstance(payload, Mapping) or not payload:
            raise ValueError(f"{family} cumulative artifact has an empty payload")
        _reject_forbidden_evidence(payload)
        if artifact.get("evidence_payload_sha256") != _sha(payload):
            raise ValueError(f"{family} cumulative evidence payload SHA-256 mismatch")


__all__ = [
    "CUMULATIVE_SPENT_CACHE_REPLAY",
    "CUMULATIVE_SPENT_EVIDENCE_BUNDLE_SCHEMA",
    "CUMULATIVE_SPENT_FAMILY_ARTIFACT_SCHEMA",
    "CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA",
    "CUMULATIVE_SPENT_REFIT",
    "CUMULATIVE_SPENT_REQUEST_SCHEMA",
    "CumulativeSpentFamilyEvidenceDraft",
    "CumulativeSpentStage1FamilyProducer",
    "CumulativeSpentStage1FamilyRequest",
    "cumulative_spent_data_projection_sha256",
    "produce_cumulative_spent_stage1_evidence_bundle",
    "validate_cumulative_spent_stage1_evidence_bundle",
]
