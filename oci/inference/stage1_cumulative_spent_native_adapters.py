"""Native legacy-family adapters for cumulative-spent Stage 1 scopes.

The exact-inner proof captures replay models on registered held-out text.  A
cumulative-spent hierarchy request deliberately cannot provide that text.  The
legacy Stage 1 runner nevertheless needs a non-empty transform frame, so this
module defines a narrowly scoped replacement: one deterministic *alias* of an
already-spent text is used only as a replay canary.  The alias is not a cohort
row, contributes no labels, and cannot contribute concept evidence.

The BoW, HTR, and matched-pair capture sinks can therefore retain and replay
their genuine fitted state without receiving any sealed text.  This module
binds those captures to the cumulative request, a component-emitted execution
record, and a sanitized family payload.  It intentionally does not register a
hierarchy index or change either production-readiness gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from .all_evidence_discovery_interfaces import (
    BOW_NUISANCE,
    BOW_R_LOSS,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
)
from .bow_native_proof_capture import (
    BOW_NATIVE_CAPTURE_SCHEMA,
    validate_bow_native_capture,
)
from .htr_native_proof_capture import (
    HTR_NATIVE_CAPTURE_SCHEMA,
    validate_htr_native_capture,
)
from .matched_pair_native_proof_capture import (
    MATCHED_PAIR_NATIVE_CAPTURE_SCHEMA,
    validate_matched_pair_native_capture,
)
from .multi_model_agentic_forest import _normalize_texts
from .stage1_cumulative_spent_evidence import (
    CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA,
    CUMULATIVE_SPENT_REFIT,
    CumulativeSpentFamilyEvidenceDraft,
    CumulativeSpentStage1FamilyRequest,
)
from .stage1_exact_inner_evidence import (
    EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
    row_order_fingerprint,
)
from .stage1_exact_inner_family_adapters import (
    native_artifact_sha256,
    native_family_code_identity,
)

CUMULATIVE_SPENT_NATIVE_ADAPTER_VERSION = "native_cumulative_spent_stage1_adapter_v1"
CUMULATIVE_SPENT_NATIVE_EXECUTION_RECORD_SCHEMA = (
    "native_cumulative_spent_stage1_execution_record_v1"
)
CUMULATIVE_SPENT_REPLAY_CANARY_SCHEMA = "cumulative_spent_replay_canary_v1"
CUMULATIVE_SPENT_REPLAY_CANARY_SEMANTICS = "spent_text_transform_alias_not_a_cohort_row"

LEGACY_CUMULATIVE_CAPTURE_FAMILIES = frozenset(
    {BOW_NUISANCE, BOW_R_LOSS, HTR_NEURAL, MATCHED_PAIR_UPLIFT}
)

_BOW_FAMILIES = frozenset({BOW_NUISANCE, BOW_R_LOSS})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ALIAS_ROW_ID = (1 << 63) - 1


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, field_name: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return text


def _stable_module_sha256() -> str:
    path = Path(__file__)
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    ) != (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    ):
        raise RuntimeError("cumulative native adapter changed while hashing")
    return _sha256_bytes(payload)


def _read_stable_json(path: Path | str) -> tuple[dict[str, Any], str]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("native cumulative execution record must be one regular file")
    before = source.stat()
    payload = source.read_bytes()
    after = source.stat()
    before_key = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    after_key = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if before_key != after_key:
        raise RuntimeError("native cumulative execution record changed while reading")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in native execution record: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("native cumulative execution record is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("native cumulative execution record must be one JSON object")
    return value, _sha256_bytes(payload)


def _validate_identity(value: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("native cumulative producer identity must be a mapping")
    identity = copy.deepcopy(dict(value))
    expected = {
        "schema_version",
        "family",
        "producer_name",
        "producer_version",
        "code_sha256",
        "configuration_sha256",
    }
    if set(identity) != expected:
        raise ValueError("native cumulative producer identity is not a closed schema")
    if (
        identity.get("schema_version") != EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION
        or identity.get("family") != family
        or identity.get("producer_name") != f"native_cumulative_spent_{family}"
        or identity.get("producer_version") != CUMULATIVE_SPENT_NATIVE_ADAPTER_VERSION
    ):
        raise ValueError("native cumulative producer identity changed its native implementation")
    for key in ("producer_name", "producer_version"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            raise ValueError(f"native cumulative producer identity requires {key}")
    _require_sha256(identity.get("code_sha256"), field_name="producer code_sha256")
    _require_sha256(
        identity.get("configuration_sha256"),
        field_name="producer configuration_sha256",
    )
    _canonical_json(identity)
    return identity


def _cumulative_spent_native_code_sha256(family: str) -> str:
    native_code = native_family_code_identity(family)
    return _sha256_json(
        {
            "schema_version": CUMULATIVE_SPENT_NATIVE_ADAPTER_VERSION,
            "adapter_module_sha256": _stable_module_sha256(),
            "native_family_code_identity": native_code,
        }
    )


def cumulative_spent_native_family_identity(
    *,
    family: str,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Content-address one legacy cumulative adapter and its native fit code."""

    if family not in LEGACY_CUMULATIVE_CAPTURE_FAMILIES:
        raise ValueError("family has no legacy cumulative capture adapter")
    if not isinstance(configuration, Mapping):
        raise TypeError("cumulative native configuration must be a mapping")
    code_sha256 = _cumulative_spent_native_code_sha256(family)
    configuration_sha256 = _sha256_json(
        {
            "schema_version": CUMULATIVE_SPENT_NATIVE_ADAPTER_VERSION,
            "family": family,
            "configuration": copy.deepcopy(dict(configuration)),
        }
    )
    return {
        "schema_version": EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
        "family": family,
        "producer_name": f"native_cumulative_spent_{family}",
        "producer_version": CUMULATIVE_SPENT_NATIVE_ADAPTER_VERSION,
        "code_sha256": code_sha256,
        "configuration_sha256": configuration_sha256,
    }


@dataclass(frozen=True)
class CumulativeSpentReplayCanary:
    """One text-only alias derived from the already-spent side of a request."""

    request_binding_sha256: str
    alias_row_id: int
    source_spent_position: int
    source_spent_row_fingerprint: str
    source_text_sha256: str
    _text: str = field(repr=False, compare=False)
    schema_version: str = CUMULATIVE_SPENT_REPLAY_CANARY_SCHEMA
    semantics: str = CUMULATIVE_SPENT_REPLAY_CANARY_SEMANTICS

    def __post_init__(self) -> None:
        _require_sha256(
            self.request_binding_sha256,
            field_name="canary request_binding_sha256",
        )
        _require_sha256(
            self.source_spent_row_fingerprint,
            field_name="canary source_spent_row_fingerprint",
        )
        _require_sha256(self.source_text_sha256, field_name="canary source_text_sha256")
        if self.schema_version != CUMULATIVE_SPENT_REPLAY_CANARY_SCHEMA:
            raise ValueError("unsupported cumulative replay-canary schema")
        if self.semantics != CUMULATIVE_SPENT_REPLAY_CANARY_SEMANTICS:
            raise ValueError("unsupported cumulative replay-canary semantics")
        if not 0 <= int(self.alias_row_id) <= _MAX_ALIAS_ROW_ID:
            raise ValueError("cumulative replay-canary alias is outside signed int64")
        if int(self.source_spent_position) < 0:
            raise ValueError("cumulative replay-canary spent position is invalid")
        if not isinstance(self._text, str) or not self._text.strip():
            raise ValueError("cumulative replay-canary text must be non-empty")
        if _sha256_bytes(self._text.encode("utf-8")) != self.source_text_sha256:
            raise ValueError("cumulative replay-canary text changed after binding")
        object.__setattr__(self, "alias_row_id", int(self.alias_row_id))
        object.__setattr__(self, "source_spent_position", int(self.source_spent_position))

    @classmethod
    def from_request(
        cls,
        request: CumulativeSpentStage1FamilyRequest,
        *,
        source_spent_position: int = 0,
    ) -> "CumulativeSpentReplayCanary":
        if not isinstance(request, CumulativeSpentStage1FamilyRequest):
            raise TypeError("cumulative replay canary requires a typed request")
        position = int(source_spent_position)
        if not 0 <= position < len(request.spent_rows):
            raise ValueError("cumulative replay-canary position escapes spent rows")
        used = set(request.spent_row_ids) | set(request.sealed_row_ids)
        alias = _MAX_ALIAS_ROW_ID
        while alias in used:
            alias -= 1
            if alias < 0:
                raise RuntimeError("no signed-int64 cumulative replay alias is available")
        source = request.spent_rows[position]
        return cls(
            request_binding_sha256=request.binding_sha256,
            alias_row_id=alias,
            source_spent_position=position,
            source_spent_row_fingerprint=row_order_fingerprint((source.row_id,)),
            source_text_sha256=_sha256_bytes(source.text.encode("utf-8")),
            _text=source.text,
        )

    def assert_matches(self, request: CumulativeSpentStage1FamilyRequest) -> None:
        if request.binding_sha256 != self.request_binding_sha256:
            raise ValueError("cumulative replay canary belongs to another request")
        if self.alias_row_id in set(request.spent_row_ids) | set(request.sealed_row_ids):
            raise ValueError("cumulative replay alias collides with a cohort row")
        try:
            source = request.spent_rows[self.source_spent_position]
        except IndexError as exc:
            raise ValueError("cumulative replay-canary source row disappeared") from exc
        if (
            row_order_fingerprint((source.row_id,)) != self.source_spent_row_fingerprint
            or _sha256_bytes(source.text.encode("utf-8")) != self.source_text_sha256
            or source.text != self._text
        ):
            raise ValueError("cumulative replay-canary spent source changed")

    @property
    def binding(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantics": self.semantics,
            "request_binding_sha256": self.request_binding_sha256,
            "alias_row_id": self.alias_row_id,
            "alias_is_cohort_row": False,
            "source_spent_position": self.source_spent_position,
            "source_spent_row_fingerprint": self.source_spent_row_fingerprint,
            "source_text_sha256": self.source_text_sha256,
            "source_labels_copied_to_transform": False,
            "contributes_to_concept_evidence": False,
        }

    def transform_frame(self, *, text_column: str) -> pd.DataFrame:
        column = str(text_column)
        if not column or column == "_oci_row_id":
            raise ValueError("cumulative replay canary requires a distinct text column")
        return pd.DataFrame(
            {"_oci_row_id": [self.alias_row_id], column: [self._text]},
            columns=["_oci_row_id", column],
        )

    @property
    def text(self) -> str:
        return self._text


def _capture_kind(family: str) -> tuple[str, str]:
    if family in _BOW_FAMILIES:
        return "bow", BOW_NATIVE_CAPTURE_SCHEMA
    if family == HTR_NEURAL:
        return "htr", HTR_NATIVE_CAPTURE_SCHEMA
    if family == MATCHED_PAIR_UPLIFT:
        return "matched_pair", MATCHED_PAIR_NATIVE_CAPTURE_SCHEMA
    raise ValueError("family has no legacy cumulative capture adapter")


def _validate_native_capture(
    *,
    family: str,
    request: CumulativeSpentStage1FamilyRequest,
    canary: CumulativeSpentReplayCanary,
    capture_artifact_path: Path | str,
    htr_model_path: Path | str | None,
    expected_htr_model_tree_sha256: str | None,
    device: torch.device | str,
) -> Mapping[str, Any]:
    canary.assert_matches(request)
    raw_fit_texts = tuple(row.text for row in request.spent_rows)
    raw_heldout_texts = (canary.text,)
    common = {
        "expected_scope_id": request.scope_id,
        "expected_fit_row_ids": request.spent_row_ids,
        "expected_heldout_row_ids": (canary.alias_row_id,),
        "expected_fit_treatment": tuple(
            float(row.treatment) for row in request.spent_rows
        ),
        "expected_fit_outcome": tuple(float(row.outcome) for row in request.spent_rows),
    }
    if family in _BOW_FAMILIES:
        metadata = validate_bow_native_capture(
            Path(capture_artifact_path),
            **common,
            fit_texts=tuple(_normalize_texts(raw_fit_texts)),
            heldout_texts=tuple(_normalize_texts(raw_heldout_texts)),
        )
    elif family == HTR_NEURAL:
        metadata = validate_htr_native_capture(
            capture_artifact_path,
            **common,
            fit_texts=raw_fit_texts,
            heldout_texts=raw_heldout_texts,
            htr_model_path=htr_model_path,
            expected_model_tree_sha256=expected_htr_model_tree_sha256,
            device=device,
        )
    elif family == MATCHED_PAIR_UPLIFT:
        metadata = validate_matched_pair_native_capture(
            capture_artifact_path,
            **common,
            fit_texts=tuple(_normalize_texts(raw_fit_texts)),
            heldout_texts=tuple(_normalize_texts(raw_heldout_texts)),
            htr_model_path=htr_model_path,
            expected_htr_model_tree_sha256=expected_htr_model_tree_sha256,
            device=device,
        )
    else:
        raise ValueError("family has no legacy cumulative capture adapter")
    _kind, expected_schema = _capture_kind(family)
    if (
        metadata.get("schema_version") != expected_schema
        or int(metadata.get("outer_fold", 0)) != request.outer_fold
        or int(metadata.get("inner_fold", 0)) != request.provider_inner_fold
        or metadata.get("heldout_labels_accessed") is not False
        or metadata.get("oracle_fields_accessed") is not False
        or metadata.get("secrets_accessed") is not False
    ):
        raise ValueError("native cumulative capture changed its scope or security envelope")
    return metadata


def cumulative_spent_native_execution_record(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    producer_identity: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    evidence_item_count: int,
    replay_canary: CumulativeSpentReplayCanary,
    capture_artifact_path: Path | str,
    source_artifact_path: Path | str,
    htr_model_path: Path | str | None = None,
    expected_htr_model_tree_sha256: str | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Build the exact record the native component must persist after fitting."""

    family = request.family
    if family not in LEGACY_CUMULATIVE_CAPTURE_FAMILIES:
        raise ValueError("family has no legacy cumulative capture adapter")
    identity = _validate_identity(producer_identity, family=family)
    if identity["code_sha256"] != _cumulative_spent_native_code_sha256(family):
        raise RuntimeError("native cumulative producer code identity is stale")
    if not isinstance(evidence_payload, Mapping) or not evidence_payload:
        raise ValueError("cumulative native evidence payload must be non-empty")
    if isinstance(evidence_item_count, bool) or int(evidence_item_count) != evidence_item_count:
        raise ValueError("cumulative native evidence count must be an integer")
    if int(evidence_item_count) < 1:
        raise ValueError("cumulative native evidence count must be positive")
    metadata = _validate_native_capture(
        family=family,
        request=request,
        canary=replay_canary,
        capture_artifact_path=capture_artifact_path,
        htr_model_path=htr_model_path,
        expected_htr_model_tree_sha256=expected_htr_model_tree_sha256,
        device=device,
    )
    capture_kind, capture_schema = _capture_kind(family)
    return {
        "schema_version": CUMULATIVE_SPENT_NATIVE_EXECUTION_RECORD_SCHEMA,
        "status": "completed",
        "family": family,
        "scope": "cumulative_spent_train",
        "scope_id": request.scope_id,
        "outer_fold": request.outer_fold,
        "context_epoch": request.context_epoch,
        "provider_inner_fold": request.provider_inner_fold,
        "request_binding_sha256": request.binding_sha256,
        "split_scope_fingerprint": request.split_scope_fingerprint,
        "data_projection_sha256": request.data_projection_sha256,
        "spent_row_order_fingerprint": row_order_fingerprint(request.spent_row_ids),
        "sealed_row_order_fingerprint": row_order_fingerprint(request.sealed_row_ids),
        "fit_semantics": CUMULATIVE_SPENT_REFIT,
        "producer_identity_sha256": _sha256_json(identity),
        "producer_code_sha256": identity["code_sha256"],
        "configuration_sha256": identity["configuration_sha256"],
        "capture_kind": capture_kind,
        "capture_schema_version": capture_schema,
        "capture_metadata_sha256": _sha256_json(metadata),
        "model_artifact_sha256": native_artifact_sha256(capture_artifact_path),
        "source_artifact_sha256": native_artifact_sha256(source_artifact_path),
        "evidence_payload_sha256": _sha256_json(copy.deepcopy(dict(evidence_payload))),
        "evidence_item_count": int(evidence_item_count),
        "replay_canary": replay_canary.binding,
        "sealed_text_accessed": False,
        "sealed_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "replay_canary_contributes_to_concept_evidence": False,
        "executable_serialization_used": False,
    }


@dataclass(frozen=True)
class NativeCumulativeSpentFamilyProducer:
    """Request-bound producer reconstructed from genuine native artifacts."""

    family: str
    _request_binding_sha256: str = field(repr=False)
    _identity: Mapping[str, Any] = field(repr=False)
    _evidence_payload: Mapping[str, Any] = field(repr=False)
    _evidence_item_count: int
    _replay_canary: CumulativeSpentReplayCanary = field(repr=False)
    _capture_artifact_path: str = field(repr=False)
    _source_artifact_path: str = field(repr=False)
    _execution_record_path: str = field(repr=False)
    _execution_record_file_sha256: str = field(repr=False)
    _execution_record: Mapping[str, Any] = field(repr=False)
    _htr_model_path: str | None = field(default=None, repr=False)
    _expected_htr_model_tree_sha256: str | None = field(default=None, repr=False)
    _device: str = field(default="cpu", repr=False)

    def identity(self) -> Mapping[str, Any]:
        if self._identity.get("code_sha256") != _cumulative_spent_native_code_sha256(self.family):
            raise RuntimeError("native cumulative producer code changed after binding")
        return copy.deepcopy(dict(self._identity))

    def _revalidate(self, request: CumulativeSpentStage1FamilyRequest) -> dict[str, Any]:
        if request.family != self.family or request.binding_sha256 != self._request_binding_sha256:
            raise ValueError("native cumulative producer was invoked for another request")
        record = cumulative_spent_native_execution_record(
            request=request,
            producer_identity=self._identity,
            evidence_payload=self._evidence_payload,
            evidence_item_count=self._evidence_item_count,
            replay_canary=self._replay_canary,
            capture_artifact_path=self._capture_artifact_path,
            source_artifact_path=self._source_artifact_path,
            htr_model_path=self._htr_model_path,
            expected_htr_model_tree_sha256=self._expected_htr_model_tree_sha256,
            device=self._device,
        )
        persisted, file_sha256 = _read_stable_json(self._execution_record_path)
        if persisted != record or persisted != dict(self._execution_record):
            raise RuntimeError("component-emitted cumulative execution record changed")
        if file_sha256 != self._execution_record_file_sha256:
            raise RuntimeError("component-emitted cumulative execution bytes changed")
        return record

    def produce_cumulative_spent(
        self,
        request: CumulativeSpentStage1FamilyRequest,
    ) -> CumulativeSpentFamilyEvidenceDraft:
        record = self._revalidate(request)
        audit = {
            "schema_version": CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA,
            "family": self.family,
            "scope": "cumulative_spent_train",
            "scope_id": request.scope_id,
            "input_binding_sha256": request.binding_sha256,
            "split_scope_fingerprint": request.split_scope_fingerprint,
            "fit_semantics": CUMULATIVE_SPENT_REFIT,
            "fit_execution_sha256": _sha256_json(record),
            "model_artifact_sha256": record["model_artifact_sha256"],
            "source_artifact_sha256": record["source_artifact_sha256"],
            "sealed_text_accessed": False,
            "sealed_labels_accessed": False,
            "oracle_fields_accessed": False,
            "secrets_accessed": False,
            "cache_source_scope_fingerprint": None,
            "cache_source_artifact_sha256": None,
            "tfidf_training_scope_policy": None,
        }
        return CumulativeSpentFamilyEvidenceDraft(
            evidence_payload=copy.deepcopy(dict(self._evidence_payload)),
            evidence_item_count=self._evidence_item_count,
            input_binding_sha256=request.binding_sha256,
            fit_semantics=CUMULATIVE_SPENT_REFIT,
            fit_audit=audit,
        )


def bind_cumulative_spent_native_family_producer(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    producer_identity: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    evidence_item_count: int,
    replay_canary: CumulativeSpentReplayCanary,
    capture_artifact_path: Path | str,
    source_artifact_path: Path | str,
    execution_record_path: Path | str,
    htr_model_path: Path | str | None = None,
    expected_htr_model_tree_sha256: str | None = None,
    device: torch.device | str = "cpu",
) -> NativeCumulativeSpentFamilyProducer:
    """Reconstruct one producer only from its persisted component record."""

    expected = cumulative_spent_native_execution_record(
        request=request,
        producer_identity=producer_identity,
        evidence_payload=evidence_payload,
        evidence_item_count=evidence_item_count,
        replay_canary=replay_canary,
        capture_artifact_path=capture_artifact_path,
        source_artifact_path=source_artifact_path,
        htr_model_path=htr_model_path,
        expected_htr_model_tree_sha256=expected_htr_model_tree_sha256,
        device=device,
    )
    persisted, file_sha256 = _read_stable_json(execution_record_path)
    if persisted != expected:
        raise ValueError("persisted cumulative execution record is not component-authentic")
    capture_path = str(Path(capture_artifact_path).resolve(strict=True))
    source_path = str(Path(source_artifact_path).resolve(strict=True))
    execution_path = str(Path(execution_record_path).resolve(strict=True))
    model_path = None if htr_model_path is None else str(Path(htr_model_path).resolve(strict=True))
    return NativeCumulativeSpentFamilyProducer(
        family=request.family,
        _request_binding_sha256=request.binding_sha256,
        _identity=_validate_identity(producer_identity, family=request.family),
        _evidence_payload=copy.deepcopy(dict(evidence_payload)),
        _evidence_item_count=int(evidence_item_count),
        _replay_canary=replay_canary,
        _capture_artifact_path=capture_path,
        _source_artifact_path=source_path,
        _execution_record_path=execution_path,
        _execution_record_file_sha256=file_sha256,
        _execution_record=copy.deepcopy(expected),
        _htr_model_path=model_path,
        _expected_htr_model_tree_sha256=(
            None
            if expected_htr_model_tree_sha256 is None
            else _require_sha256(
                expected_htr_model_tree_sha256,
                field_name="expected HTR model-tree SHA-256",
            )
        ),
        _device=str(torch.device(device)),
    )


__all__ = [
    "CUMULATIVE_SPENT_NATIVE_ADAPTER_VERSION",
    "CUMULATIVE_SPENT_NATIVE_EXECUTION_RECORD_SCHEMA",
    "CUMULATIVE_SPENT_REPLAY_CANARY_SCHEMA",
    "CUMULATIVE_SPENT_REPLAY_CANARY_SEMANTICS",
    "CumulativeSpentReplayCanary",
    "LEGACY_CUMULATIVE_CAPTURE_FAMILIES",
    "NativeCumulativeSpentFamilyProducer",
    "bind_cumulative_spent_native_family_producer",
    "cumulative_spent_native_execution_record",
    "cumulative_spent_native_family_identity",
]
