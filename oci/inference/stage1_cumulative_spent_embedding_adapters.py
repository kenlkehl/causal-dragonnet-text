"""Native cumulative-spent adapters for the three shared embedding outputs.

The whole-cohort embedding, clustered embedding, and semantic-retrieval
architectures are three lossless views of one native embedding fit.  This
module deliberately accepts a live, unfinalized :class:`NativeEmbeddingProofCaptureSink`
instead of an artifact path.  It finalizes that exact sink itself and issues
process-local emissions, preventing an exact-inner capture from being copied
and relabeled as a cumulative-spent refit.

The module is intentionally independent of ``production_stage1_bundle``.  A
production worker can later construct the sink, run the already-existing
embedding generator with it, and bind the returned producers into the common
cumulative-spent bundle.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .all_evidence_discovery_interfaces import (
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    TFIDF_SEMANTIC_RETRIEVAL,
)
from .all_evidence_fusion import (
    LEGACY_ALL_SOURCE,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
)
from .embedding_native_proof_capture import (
    EMBEDDING_NATIVE_CAPTURE_SCHEMA,
    SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA,
    NativeEmbeddingProofCaptureSink,
    validate_embedding_native_capture,
)
from .review_spent_evidence_provider import (
    BoundSpentFrozenChunkEmbeddingProvider,
    SemanticWitnessScientificConfig,
)
from .lossless_stage1_evidence_catalog import build_role_neutral_evidence_catalog
from .stage1_cumulative_spent_evidence import (
    CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA,
    CUMULATIVE_SPENT_REFIT,
    CumulativeSpentFamilyEvidenceDraft,
    CumulativeSpentStage1FamilyRequest,
)
from .stage1_cumulative_spent_native_adapters import CumulativeSpentReplayCanary
from .stage1_exact_inner_evidence import (
    EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
    row_order_fingerprint,
)
from .stage1_exact_inner_family_adapters import (
    NATIVE_FAMILY_PAYLOAD_VERSION,
    family_payload_from_catalog,
    native_artifact_sha256,
    native_family_code_identity,
)

CUMULATIVE_SPENT_EMBEDDING_ADAPTER_VERSION = "native_cumulative_spent_shared_embedding_adapter_v1"
CUMULATIVE_SPENT_EMBEDDING_EXECUTION_RECORD_SCHEMA = (
    "native_cumulative_spent_shared_embedding_execution_record_v1"
)
CUMULATIVE_SPENT_EMBEDDING_PAYLOAD_SCHEMA = NATIVE_FAMILY_PAYLOAD_VERSION
CUMULATIVE_SPENT_EMBEDDING_EMISSION_SCHEMA = "same_process_cumulative_spent_embedding_emission_v1"
CUMULATIVE_SPENT_EMBEDDING_RELOAD_VALIDATION_SCHEMA = (
    "cumulative_spent_embedding_persisted_reload_validation_v1"
)

CUMULATIVE_SPENT_EMBEDDING_FAMILIES = frozenset(
    {
        EMBEDDING_WHOLE_COHORT,
        EMBEDDING_CLUSTERED,
        TFIDF_SEMANTIC_RETRIEVAL,
    }
)

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
_CLUSTER_CONTRAST_FAMILIES = frozenset(
    {
        "cluster_local_treatment_contrast_basis",
        "cluster_local_residualized_interaction_contrast_basis",
    }
)
_SEMANTIC_SOURCE_FIELDS = frozenset(
    {"enabled", "concept_derivation", "raw_retrieved_excerpts_retained", "contrasts"}
)
_SEMANTIC_CONTRAST_FIELDS = frozenset(
    {
        "name",
        "role_hint",
        "contrast_family",
        "direction_source",
        "cluster_component_index",
        "concept_probe_scores",
    }
)
_SEMANTIC_SCORE_FIELDS = frozenset({"concept", "score"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMISSION_AUTHORITY = object()


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
        raise RuntimeError("cumulative embedding adapter changed while hashing")
    return _sha256_bytes(payload)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key in cumulative embedding artifact: {key}")
        output[key] = value
    return output


def _read_stable_json(path: Path | str, *, field_name: str) -> tuple[dict[str, Any], str]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{field_name} must be one regular file")
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
        raise RuntimeError(f"{field_name} changed while reading")
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be one JSON object")
    return value, _sha256_bytes(payload)


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> str:
    target = Path(path)
    if target.exists():
        raise RuntimeError(f"refusing to replace cumulative embedding artifact: {target}")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if target.exists():
            raise RuntimeError(f"refusing to replace cumulative embedding artifact: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_bytes(payload)


def _adapter_code_sha256(family: str) -> str:
    if family not in CUMULATIVE_SPENT_EMBEDDING_FAMILIES:
        raise ValueError("family has no cumulative shared-embedding adapter")
    return _sha256_json(
        {
            "schema_version": CUMULATIVE_SPENT_EMBEDDING_ADAPTER_VERSION,
            "adapter_module_sha256": _stable_module_sha256(),
            "native_family_code_identity": native_family_code_identity(family),
        }
    )


def cumulative_spent_embedding_family_identity(
    *,
    family: str,
    capture_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Content-address one of the three views and the native shared fit."""

    if family not in CUMULATIVE_SPENT_EMBEDDING_FAMILIES:
        raise ValueError("family has no cumulative shared-embedding adapter")
    if not isinstance(capture_configuration, Mapping):
        raise TypeError("capture_configuration must be a mapping")
    configuration = copy.deepcopy(dict(capture_configuration))
    _canonical_json(configuration)
    return {
        "schema_version": EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
        "family": family,
        "producer_name": f"native_cumulative_spent_{family}",
        "producer_version": CUMULATIVE_SPENT_EMBEDDING_ADAPTER_VERSION,
        "code_sha256": _adapter_code_sha256(family),
        "configuration_sha256": _sha256_json(
            {
                "schema_version": CUMULATIVE_SPENT_EMBEDDING_ADAPTER_VERSION,
                "family": family,
                "capture_configuration": configuration,
            }
        ),
    }


def _validate_identity(value: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("cumulative embedding identity must be a mapping")
    identity = copy.deepcopy(dict(value))
    fields = {
        "schema_version",
        "family",
        "producer_name",
        "producer_version",
        "code_sha256",
        "configuration_sha256",
    }
    if set(identity) != fields:
        raise ValueError("cumulative embedding identity is not a closed schema")
    if (
        identity.get("schema_version") != EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION
        or identity.get("family") != family
        or identity.get("producer_name") != f"native_cumulative_spent_{family}"
        or identity.get("producer_version") != CUMULATIVE_SPENT_EMBEDDING_ADAPTER_VERSION
        or identity.get("code_sha256") != _adapter_code_sha256(family)
    ):
        raise ValueError("cumulative embedding identity changed its native implementation")
    _require_sha256(identity.get("configuration_sha256"), field_name="configuration_sha256")
    return identity


def _request_map(
    requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    replay_canary: CumulativeSpentReplayCanary,
) -> dict[str, CumulativeSpentStage1FamilyRequest]:
    if not isinstance(requests, Mapping) or set(requests) != set(
        CUMULATIVE_SPENT_EMBEDDING_FAMILIES
    ):
        raise ValueError("shared embedding emission requires exactly its three family requests")
    result: dict[str, CumulativeSpentStage1FamilyRequest] = {}
    reference: CumulativeSpentStage1FamilyRequest | None = None
    for family in sorted(CUMULATIVE_SPENT_EMBEDDING_FAMILIES):
        request = requests[family]
        if not isinstance(request, CumulativeSpentStage1FamilyRequest):
            raise TypeError("shared embedding requests must use the typed cumulative boundary")
        if request.family != family:
            raise ValueError("shared embedding request mapping changed a family")
        replay_canary.assert_matches(request)
        if reference is None:
            reference = request
        elif request.binding != reference.binding:
            raise ValueError("shared embedding family requests do not describe one spent scope")
        result[family] = request
    return result


def _canonical_labels(
    request: CumulativeSpentStage1FamilyRequest,
) -> tuple[np.ndarray, np.ndarray]:
    treatment = np.asarray([row.treatment for row in request.spent_rows], dtype=np.float64)
    outcome = np.asarray([row.outcome for row in request.spent_rows], dtype=np.float64)
    if (
        treatment.shape != outcome.shape
        or not np.isfinite(treatment).all()
        or not np.isfinite(outcome).all()
    ):
        raise ValueError("cumulative embedding canonical labels are malformed")
    return treatment, outcome


def _assert_fresh_spent_sink(
    *,
    sink: NativeEmbeddingProofCaptureSink,
    request: CumulativeSpentStage1FamilyRequest,
    replay_canary: CumulativeSpentReplayCanary,
) -> None:
    if type(sink) is not NativeEmbeddingProofCaptureSink:
        raise TypeError("cumulative embedding emission requires the exact native capture sink")
    if sink._finalized or sink.artifact_dir.exists():
        raise ValueError(
            "cumulative embedding capture must be a genuinely new, unfinalized sink artifact"
        )
    treatment, outcome = _canonical_labels(request)
    if (
        sink.scope_id != request.scope_id
        or sink.outer_fold != request.outer_fold
        or sink.inner_fold != request.provider_inner_fold
        or sink.fit_row_ids != request.spent_row_ids
        or sink.heldout_row_ids != (replay_canary.alias_row_id,)
        or sink.fit_texts != tuple(row.text for row in request.spent_rows)
        or type(sink.embedding_provider) is not BoundSpentFrozenChunkEmbeddingProvider
        or tuple(map(int, sink.embedding_provider.row_ids)) != request.spent_row_ids
    ):
        raise ValueError("cumulative embedding sink changed its exact spent-only scope")
    if set(sink.fit_row_ids) & set(request.sealed_row_ids):
        raise ValueError("cumulative embedding sink provider entered sealed rows")
    if not np.array_equal(sink.expected_fit_treatment, treatment) or not np.array_equal(
        sink.expected_fit_outcome, outcome
    ):
        raise ValueError("cumulative embedding sink changed canonical treatment/outcome labels")


def _validate_policy(policy: Any, *, request: CumulativeSpentStage1FamilyRequest) -> dict[str, Any]:
    if not isinstance(policy, Mapping):
        raise ValueError("semantic retrieval capture has no training-only policy")
    value = copy.deepcopy(dict(policy))
    model_rows = tuple(map(int, value.get("model_fit_row_ids") or ()))
    calibration_rows = tuple(map(int, value.get("calibration_row_ids") or ()))
    if (
        value.get("schema_version") != SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA
        or value.get("policy") != "training_only_exhaustive_no_selection"
        or value.get("selection_kind") != "none_deterministic_exhaustive"
        or value.get("nested_calibration_applicability") != "no_label_or_hyperparameter_selection"
        or value.get("partitions_are_replay_canaries_only") is not True
        or value.get("partition_canaries_select_or_drop_terms") is not False
        or value.get("authoritative_projection_scope") != "all_exact_fit_frozen_retrieval_tails"
        or value.get("projection_vocabulary_max_features") is not None
        or value.get("projection_output_limit") is not None
        or value.get("all_nonzero_sanitized_terms_preserved") is not True
        or value.get("nested_calibration_labels_accessed") is not False
        or value.get("registered_heldout_labels_accessed") is not False
        or value.get("registered_heldout_text_accessed") is not False
        or value.get("registered_heldout_transform_performed") is not False
        or value.get("canonical_hierarchy_partition_count_used_as_calibration_folds") is not False
        or value.get("interaction_inner_folds_used_as_calibration_folds") is not False
        or not model_rows
        or not calibration_rows
        or set(model_rows) & set(calibration_rows)
        or set(model_rows) | set(calibration_rows) != set(request.spent_row_ids)
    ):
        raise ValueError(
            "semantic retrieval policy must be exhaustive, uncapped, label-free, and nonselecting"
        )
    return value


def _capture_configuration(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "capture_schema_version": EMBEDDING_NATIVE_CAPTURE_SCHEMA,
        "text_column": metadata["text_column"],
        "outcome_type": metadata["outcome_type"],
        "seed": metadata["seed"],
        "embedding_config": copy.deepcopy(metadata["embedding_config"]),
        "semantic_witness_scientific_config": copy.deepcopy(
            metadata["semantic_witness_scientific_config"]
        ),
        "semantic_witness_scientific_config_sha256": str(
            metadata["semantic_witness_scientific_config_sha256"]
        ),
        "tfidf_nested_calibration_folds": int(
            metadata["tfidf_training_scope_policy"]["configured_fold_count"]
        ),
    }


def _semantic_score_eligibility_policy(
    metadata: Mapping[str, Any],
) -> str:
    raw = metadata.get("semantic_witness_scientific_config")
    config = SemanticWitnessScientificConfig.from_mapping(
        raw,
        label="native capture semantic-witness scientific config",
    )
    if (
        metadata.get("semantic_witness_scientific_config_sha256")
        != config.identity_sha256
    ):
        raise ValueError(
            "native capture semantic-witness configuration hash changed"
        )
    return config.retrieval_score_eligibility_policy


def _validate_semantic_source(
    value: Mapping[str, Any],
    *,
    score_eligibility_policy: str,
) -> tuple[list[dict[str, Any]], int]:
    if score_eligibility_policy != "all_finite_including_zero_v1":
        raise ValueError(
            "cumulative semantic adapter does not implement the configured "
            "retrieval score-eligibility policy"
        )
    if set(value) != set(_SEMANTIC_SOURCE_FIELDS):
        raise ValueError("semantic full-scope evidence is not a closed schema")
    if (
        value.get("enabled") is not True
        or value.get("concept_derivation")
        != "tfidf_ngrams_contrasting_frozen_embedding_retrieval_tails"
        or value.get("raw_retrieved_excerpts_retained") is not False
        or not isinstance(value.get("contrasts"), list)
    ):
        raise ValueError("semantic full-scope evidence changed its safe projection")
    contrasts: list[dict[str, Any]] = []
    member_count = 0
    for raw in value["contrasts"]:
        if not isinstance(raw, Mapping) or not set(raw) <= set(_SEMANTIC_CONTRAST_FIELDS):
            raise ValueError("semantic full-scope contrast is not a closed schema")
        required = {"name", "contrast_family", "direction_source", "concept_probe_scores"}
        if not required <= set(raw) or not isinstance(raw["concept_probe_scores"], list):
            raise ValueError("semantic full-scope contrast is incomplete")
        contrast = copy.deepcopy(dict(raw))
        if not all(
            isinstance(contrast[key], str) and contrast[key]
            for key in required - {"concept_probe_scores"}
        ):
            raise ValueError("semantic full-scope contrast identity is invalid")
        scores = contrast["concept_probe_scores"]
        if not scores:
            raise ValueError("semantic full-scope contrast has no concept evidence")
        for score in scores:
            if not isinstance(score, Mapping) or set(score) != set(_SEMANTIC_SCORE_FIELDS):
                raise ValueError("semantic contrastive term is not a closed schema")
            numeric = score.get("score")
            if (
                not isinstance(score.get("concept"), str)
                or not score["concept"].strip()
                or isinstance(numeric, bool)
                or not math.isfinite(float(numeric))
            ):
                raise ValueError(
                    "semantic contrastive term is empty or non-finite"
                )
        member_count += len(scores)
        contrasts.append(contrast)
    if not contrasts or member_count < 1:
        raise ValueError("semantic full-scope evidence is empty")
    return contrasts, member_count


def _payloads_from_capture(
    capture_dir: Path,
    *,
    request: CumulativeSpentStage1FamilyRequest,
    metadata: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    source, _file_sha = _read_stable_json(
        capture_dir / "semantic_full_scope_evidence.json",
        field_name="semantic full-scope evidence",
    )
    all_contrasts, _all_member_count = _validate_semantic_source(
        source,
        score_eligibility_policy=_semantic_score_eligibility_policy(
            metadata
        ),
    )
    expected_families = _WHOLE_CONTRAST_FAMILIES | _CLUSTER_CONTRAST_FAMILIES
    observed_families = {str(row["contrast_family"]) for row in all_contrasts}
    if not observed_families <= expected_families:
        raise ValueError("semantic projection contains an unknown embedding contrast family")
    grouped: dict[str, list[dict[str, Any]]] = {
        "confounders": [],
        "effect_modifiers": [],
    }
    for raw in all_contrasts:
        role = str(raw.get("role_hint") or "").strip()
        section = {
            "confounder": "confounders",
            "effect_modifier": "effect_modifiers",
        }.get(role)
        if section is None:
            raise ValueError("semantic embedding contrast has an unknown role hint")
        grouped[section].append(
            {
                **copy.deepcopy(raw),
                "concept_derivation": source["concept_derivation"],
                "raw_retrieved_excerpts_retained": source["raw_retrieved_excerpts_retained"],
            }
        )
    digest = {
        section: {"embedding_chunks": contrasts}
        for section, contrasts in grouped.items()
        if contrasts
    }
    provenance = FoldEvidenceProvenance(
        outer_fold=request.outer_fold,
        train_row_ids=request.spent_row_ids,
        heldout_row_ids=request.sealed_row_ids,
        scope="inner_train",
        inner_fold=request.provider_inner_fold,
        artifact_id=f"cumulative-embedding-{request.scope_id}",
    )
    if provenance.split_fingerprint != request.split_scope_fingerprint:
        raise ValueError("cumulative embedding catalog differs from the canonical split")
    catalog = build_role_neutral_evidence_catalog(
        (
            FoldEvidenceInput(
                LEGACY_ALL_SOURCE,
                {"context": {"evidence_digest": digest}},
                provenance,
            ),
        ),
        require_all_source_kinds=False,
        require_all_architecture_families=False,
        require_upstream_completeness=True,
    )
    payloads: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for family in sorted(CUMULATIVE_SPENT_EMBEDDING_FAMILIES):
        payload, count = family_payload_from_catalog(catalog, family=family)
        if int(count) < 1 or not payload.get("architecture_evidence"):
            raise ValueError(f"cumulative shared embedding payload is empty for {family}")
        payloads[family] = payload
        counts[family] = int(count)
    return payloads, counts


_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "family",
        "scope",
        "scope_id",
        "outer_fold",
        "context_epoch",
        "provider_inner_fold",
        "request_sha256",
        "schedule_sha256",
        "request_binding_sha256",
        "split_scope_fingerprint",
        "data_projection_sha256",
        "spent_row_order_fingerprint",
        "sealed_row_order_fingerprint",
        "fit_semantics",
        "producer_identity_sha256",
        "producer_code_sha256",
        "configuration_sha256",
        "capture_origin",
        "capture_schema_version",
        "capture_metadata_sha256",
        "semantic_policy_sha256",
        "semantic_model_replay_canary_sha256",
        "semantic_calibration_replay_canary_sha256",
        "model_artifact_sha256",
        "source_artifact_sha256",
        "evidence_payload_sha256",
        "evidence_item_count",
        "replay_canary",
        "sealed_text_accessed",
        "sealed_labels_accessed",
        "oracle_fields_accessed",
        "secrets_accessed",
        "replay_canaries_are_label_free_nonselecting",
        "replay_canary_contributes_to_concept_evidence",
        "executable_serialization_used",
    }
)


def _execution_record(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    replay_canary: CumulativeSpentReplayCanary,
    identity: Mapping[str, Any],
    metadata: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    evidence_item_count: int,
    capture_dir: Path,
) -> dict[str, Any]:
    family = request.family
    validated_identity = _validate_identity(identity, family=family)
    policy = _validate_policy(metadata.get("tfidf_training_scope_policy"), request=request)
    source_path = capture_dir / "semantic_full_scope_evidence.json"
    model_canary_path = capture_dir / "semantic_model_replay_canary.json"
    calibration_canary_path = capture_dir / "semantic_calibration_replay_canary.json"
    record = {
        "schema_version": CUMULATIVE_SPENT_EMBEDDING_EXECUTION_RECORD_SCHEMA,
        "status": "completed",
        "family": family,
        "scope": "cumulative_spent_train",
        "scope_id": request.scope_id,
        "outer_fold": request.outer_fold,
        "context_epoch": request.context_epoch,
        "provider_inner_fold": request.provider_inner_fold,
        "request_sha256": request.request_sha256,
        "schedule_sha256": request.schedule_sha256,
        "request_binding_sha256": request.binding_sha256,
        "split_scope_fingerprint": request.split_scope_fingerprint,
        "data_projection_sha256": request.data_projection_sha256,
        "spent_row_order_fingerprint": row_order_fingerprint(request.spent_row_ids),
        "sealed_row_order_fingerprint": row_order_fingerprint(request.sealed_row_ids),
        "fit_semantics": CUMULATIVE_SPENT_REFIT,
        "producer_identity_sha256": _sha256_json(validated_identity),
        "producer_code_sha256": validated_identity["code_sha256"],
        "configuration_sha256": validated_identity["configuration_sha256"],
        "capture_origin": "live_unfinalized_spent_bound_native_embedding_sink",
        "capture_schema_version": EMBEDDING_NATIVE_CAPTURE_SCHEMA,
        "capture_metadata_sha256": _sha256_json(metadata),
        "semantic_policy_sha256": _sha256_json(policy),
        "semantic_model_replay_canary_sha256": native_artifact_sha256(model_canary_path),
        "semantic_calibration_replay_canary_sha256": native_artifact_sha256(
            calibration_canary_path
        ),
        "model_artifact_sha256": native_artifact_sha256(capture_dir),
        "source_artifact_sha256": native_artifact_sha256(source_path),
        "evidence_payload_sha256": _sha256_json(copy.deepcopy(dict(evidence_payload))),
        "evidence_item_count": int(evidence_item_count),
        "replay_canary": replay_canary.binding,
        "sealed_text_accessed": False,
        "sealed_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "replay_canaries_are_label_free_nonselecting": True,
        "replay_canary_contributes_to_concept_evidence": False,
        "executable_serialization_used": False,
    }
    if set(record) != set(_RECORD_FIELDS):
        raise RuntimeError("cumulative embedding execution record is not closed")
    return record


def _validate_record_envelope(record: Mapping[str, Any], *, family: str) -> None:
    if set(record) != set(_RECORD_FIELDS):
        raise ValueError("cumulative embedding execution record is not a closed schema")
    if (
        record.get("schema_version") != CUMULATIVE_SPENT_EMBEDDING_EXECUTION_RECORD_SCHEMA
        or record.get("status") != "completed"
        or record.get("family") != family
        or record.get("scope") != "cumulative_spent_train"
        or record.get("fit_semantics") != CUMULATIVE_SPENT_REFIT
        or record.get("capture_origin") != "live_unfinalized_spent_bound_native_embedding_sink"
        or record.get("capture_schema_version") != EMBEDDING_NATIVE_CAPTURE_SCHEMA
        or record.get("sealed_text_accessed") is not False
        or record.get("sealed_labels_accessed") is not False
        or record.get("oracle_fields_accessed") is not False
        or record.get("secrets_accessed") is not False
        or record.get("replay_canaries_are_label_free_nonselecting") is not True
        or record.get("replay_canary_contributes_to_concept_evidence") is not False
        or record.get("executable_serialization_used") is not False
    ):
        raise ValueError("cumulative embedding execution record changed its security envelope")
    for key in (
        "producer_identity_sha256",
        "producer_code_sha256",
        "configuration_sha256",
        "capture_metadata_sha256",
        "semantic_policy_sha256",
        "semantic_model_replay_canary_sha256",
        "semantic_calibration_replay_canary_sha256",
        "model_artifact_sha256",
        "source_artifact_sha256",
        "evidence_payload_sha256",
    ):
        _require_sha256(record.get(key), field_name=key)
    if (
        isinstance(record.get("evidence_item_count"), bool)
        or int(record.get("evidence_item_count", 0)) < 1
    ):
        raise ValueError("cumulative embedding execution record has no evidence")


@dataclass(frozen=True)
class CumulativeSpentEmbeddingFamilyEmission:
    """Opaque same-process authority issued only while finalizing a live sink."""

    family: str
    request_binding_sha256: str
    capture_artifact_path: str
    source_artifact_path: str
    execution_record_path: str
    execution_artifact_sha256: str
    _identity: Mapping[str, Any] = field(repr=False)
    _evidence_payload: Mapping[str, Any] = field(repr=False)
    _evidence_item_count: int = field(repr=False)
    _metadata: Mapping[str, Any] = field(repr=False)
    _embedding_provider: BoundSpentFrozenChunkEmbeddingProvider = field(repr=False)
    _authority: object = field(repr=False, compare=False)
    schema_version: str = CUMULATIVE_SPENT_EMBEDDING_EMISSION_SCHEMA

    def __post_init__(self) -> None:
        if self._authority is not _EMISSION_AUTHORITY:
            raise TypeError("cumulative embedding emissions must be issued by a live sink")
        if (
            self.schema_version != CUMULATIVE_SPENT_EMBEDDING_EMISSION_SCHEMA
            or self.family not in CUMULATIVE_SPENT_EMBEDDING_FAMILIES
        ):
            raise ValueError("invalid cumulative embedding emission")
        _require_sha256(self.request_binding_sha256, field_name="request_binding_sha256")
        _require_sha256(
            self.execution_artifact_sha256,
            field_name="execution_artifact_sha256",
        )


def emit_cumulative_spent_embedding_capture(
    *,
    requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    replay_canary: CumulativeSpentReplayCanary,
    capture_sink: NativeEmbeddingProofCaptureSink,
    execution_record_dir: Path | str,
) -> Mapping[str, CumulativeSpentEmbeddingFamilyEmission]:
    """Finalize one new spent-only sink and emit all three family records."""

    if not isinstance(replay_canary, CumulativeSpentReplayCanary):
        raise TypeError("shared embedding emission requires a cumulative replay canary")
    typed = _request_map(requests, replay_canary)
    request = typed[EMBEDDING_WHOLE_COHORT]
    _assert_fresh_spent_sink(sink=capture_sink, request=request, replay_canary=replay_canary)
    record_root = Path(execution_record_dir)
    if record_root.exists():
        raise RuntimeError("cumulative embedding execution-record directory must be new")

    # ``finalize`` is the point of no relabeling: this call consumes the live
    # in-memory observer state and creates a previously nonexistent artifact.
    capture_sink.finalize()
    treatment, outcome = _canonical_labels(request)
    metadata = validate_embedding_native_capture(
        capture_sink.artifact_dir,
        embedding_provider=capture_sink.embedding_provider,
        fit_texts=tuple(row.text for row in request.spent_rows),
        expected_fit_treatment=treatment,
        expected_fit_outcome=outcome,
        expected_discovery_projection={"_oci_row_id": list(request.spent_row_ids)},
        expected_scope_id=request.scope_id,
        expected_fit_row_ids=request.spent_row_ids,
        expected_heldout_row_ids=(replay_canary.alias_row_id,),
    )
    if (
        metadata.get("schema_version") != EMBEDDING_NATIVE_CAPTURE_SCHEMA
        or metadata.get("heldout_text_accessed") is not False
        or metadata.get("heldout_labels_accessed") is not False
        or metadata.get("oracle_fields_accessed") is not False
        or metadata.get("secrets_accessed") is not False
    ):
        raise ValueError("cumulative embedding capture changed its security envelope")
    _validate_policy(metadata.get("tfidf_training_scope_policy"), request=request)
    payloads, counts = _payloads_from_capture(
        capture_sink.artifact_dir,
        request=request,
        metadata=metadata,
    )
    configuration = _capture_configuration(metadata)
    record_root.mkdir(parents=True, exist_ok=False)
    emissions: dict[str, CumulativeSpentEmbeddingFamilyEmission] = {}
    for family in sorted(CUMULATIVE_SPENT_EMBEDDING_FAMILIES):
        family_request = typed[family]
        identity = cumulative_spent_embedding_family_identity(
            family=family,
            capture_configuration=configuration,
        )
        record = _execution_record(
            request=family_request,
            replay_canary=replay_canary,
            identity=identity,
            metadata=metadata,
            evidence_payload=payloads[family],
            evidence_item_count=counts[family],
            capture_dir=capture_sink.artifact_dir,
        )
        path = record_root / f"{family}.json"
        execution_sha256 = _write_immutable_json(path, record)
        emissions[family] = CumulativeSpentEmbeddingFamilyEmission(
            family=family,
            request_binding_sha256=family_request.binding_sha256,
            capture_artifact_path=str(capture_sink.artifact_dir.resolve(strict=True)),
            source_artifact_path=str(
                (capture_sink.artifact_dir / "semantic_full_scope_evidence.json").resolve(
                    strict=True
                )
            ),
            execution_record_path=str(path.resolve(strict=True)),
            execution_artifact_sha256=execution_sha256,
            _identity=copy.deepcopy(identity),
            _evidence_payload=copy.deepcopy(payloads[family]),
            _evidence_item_count=counts[family],
            _metadata=copy.deepcopy(metadata),
            _embedding_provider=capture_sink.embedding_provider,
            _authority=_EMISSION_AUTHORITY,
        )
    return emissions


def _fit_audit_from_record(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    record: Mapping[str, Any],
    execution_artifact_sha256: str,
    semantic_policy: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA,
        "family": request.family,
        "scope": "cumulative_spent_train",
        "scope_id": request.scope_id,
        "input_binding_sha256": request.binding_sha256,
        "split_scope_fingerprint": request.split_scope_fingerprint,
        "fit_semantics": CUMULATIVE_SPENT_REFIT,
        "fit_execution_sha256": _require_sha256(
            execution_artifact_sha256,
            field_name="execution_artifact_sha256",
        ),
        "model_artifact_sha256": record["model_artifact_sha256"],
        "source_artifact_sha256": record["source_artifact_sha256"],
        "sealed_text_accessed": False,
        "sealed_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "cache_source_scope_fingerprint": None,
        "cache_source_artifact_sha256": None,
        "tfidf_training_scope_policy": (
            copy.deepcopy(dict(semantic_policy))
            if request.family == TFIDF_SEMANTIC_RETRIEVAL
            else None
        ),
    }


def validate_cumulative_spent_embedding_family_artifact(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    replay_canary: CumulativeSpentReplayCanary,
    embedding_provider: BoundSpentFrozenChunkEmbeddingProvider,
    producer_identity: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    evidence_item_count: int,
    capture_artifact_path: Path | str,
    execution_record_path: Path | str,
    expected_fit_audit: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Independently reload one persisted shared-embedding family."""

    if not isinstance(request, CumulativeSpentStage1FamilyRequest) or (
        request.family not in CUMULATIVE_SPENT_EMBEDDING_FAMILIES
    ):
        raise TypeError("embedding reload requires a typed shared-family request")
    if not isinstance(replay_canary, CumulativeSpentReplayCanary):
        raise TypeError("embedding reload requires a cumulative replay canary")
    replay_canary.assert_matches(request)
    if type(embedding_provider) is not BoundSpentFrozenChunkEmbeddingProvider:
        raise TypeError("embedding reload requires the exact spent-bound provider")
    if not isinstance(evidence_payload, Mapping):
        raise TypeError("embedding reload evidence_payload must be a mapping")
    if (
        isinstance(evidence_item_count, bool)
        or int(evidence_item_count) != evidence_item_count
        or int(evidence_item_count) < 1
    ):
        raise ValueError("embedding reload evidence_item_count must be positive")
    capture_root = Path(capture_artifact_path)
    if capture_root.is_symlink() or not capture_root.is_dir():
        raise ValueError("embedding reload capture root must be one real directory")
    capture_dir = capture_root.resolve(strict=True)
    treatment, outcome = _canonical_labels(request)
    metadata = validate_embedding_native_capture(
        capture_dir,
        embedding_provider=embedding_provider,
        fit_texts=tuple(row.text for row in request.spent_rows),
        expected_fit_treatment=treatment,
        expected_fit_outcome=outcome,
        expected_discovery_projection={"_oci_row_id": list(request.spent_row_ids)},
        expected_scope_id=request.scope_id,
        expected_fit_row_ids=request.spent_row_ids,
        expected_heldout_row_ids=(replay_canary.alias_row_id,),
    )
    policy = _validate_policy(metadata.get("tfidf_training_scope_policy"), request=request)
    payloads, counts = _payloads_from_capture(
        capture_dir,
        request=request,
        metadata=metadata,
    )
    regenerated_payload = payloads[request.family]
    regenerated_count = counts[request.family]
    if (
        copy.deepcopy(dict(evidence_payload)) != regenerated_payload
        or int(evidence_item_count) != regenerated_count
    ):
        raise ValueError("persisted embedding payload/count differs from native capture source")
    expected_identity = cumulative_spent_embedding_family_identity(
        family=request.family,
        capture_configuration=_capture_configuration(metadata),
    )
    supplied_identity = _validate_identity(producer_identity, family=request.family)
    if supplied_identity != expected_identity:
        raise ValueError("persisted embedding identity differs from authenticated native config")
    expected_record = _execution_record(
        request=request,
        replay_canary=replay_canary,
        identity=supplied_identity,
        metadata=metadata,
        evidence_payload=regenerated_payload,
        evidence_item_count=regenerated_count,
        capture_dir=capture_dir,
    )
    persisted, execution_sha256 = _read_stable_json(
        execution_record_path,
        field_name="cumulative embedding execution record",
    )
    _validate_record_envelope(persisted, family=request.family)
    if persisted != expected_record:
        raise ValueError("persisted embedding execution record differs from canonical replay")
    fit_audit = _fit_audit_from_record(
        request=request,
        record=persisted,
        execution_artifact_sha256=execution_sha256,
        semantic_policy=policy,
    )
    if expected_fit_audit is not None and (
        not isinstance(expected_fit_audit, Mapping)
        or copy.deepcopy(dict(expected_fit_audit)) != fit_audit
    ):
        raise ValueError("persisted embedding fit audit differs from canonical replay")
    return {
        "schema_version": CUMULATIVE_SPENT_EMBEDDING_RELOAD_VALIDATION_SCHEMA,
        "family": request.family,
        "request_binding_sha256": request.binding_sha256,
        "capture_metadata_sha256": _sha256_json(metadata),
        "producer_identity": supplied_identity,
        "evidence_payload_sha256": _sha256_json(regenerated_payload),
        "evidence_item_count": regenerated_count,
        "execution_artifact_sha256": execution_sha256,
        "fit_audit": fit_audit,
    }


def validate_cumulative_spent_embedding_artifacts(
    *,
    requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    replay_canary: CumulativeSpentReplayCanary,
    embedding_provider: BoundSpentFrozenChunkEmbeddingProvider,
    producer_identity_by_family: Mapping[str, Mapping[str, Any]],
    evidence_payload_by_family: Mapping[str, Mapping[str, Any]],
    evidence_item_count_by_family: Mapping[str, int],
    capture_artifact_path: Path | str,
    execution_record_path_by_family: Mapping[str, Path | str],
    expected_fit_audit_by_family: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Validate the complete persisted three-view embedding component."""

    typed = _request_map(requests, replay_canary)
    required = set(CUMULATIVE_SPENT_EMBEDDING_FAMILIES)
    for name, value in (
        ("producer identities", producer_identity_by_family),
        ("evidence payloads", evidence_payload_by_family),
        ("evidence counts", evidence_item_count_by_family),
        ("execution records", execution_record_path_by_family),
    ):
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"persisted embedding {name} must cover exactly three families")
    if expected_fit_audit_by_family is not None and (
        not isinstance(expected_fit_audit_by_family, Mapping)
        or set(expected_fit_audit_by_family) != required
    ):
        raise ValueError("persisted embedding fit audits must cover exactly three families")
    output: dict[str, Mapping[str, Any]] = {}
    for family in sorted(required):
        output[family] = validate_cumulative_spent_embedding_family_artifact(
            request=typed[family],
            replay_canary=replay_canary,
            embedding_provider=embedding_provider,
            producer_identity=producer_identity_by_family[family],
            evidence_payload=evidence_payload_by_family[family],
            evidence_item_count=evidence_item_count_by_family[family],
            capture_artifact_path=capture_artifact_path,
            execution_record_path=execution_record_path_by_family[family],
            expected_fit_audit=(
                None
                if expected_fit_audit_by_family is None
                else expected_fit_audit_by_family[family]
            ),
        )
    return output


@dataclass(frozen=True)
class NativeCumulativeSpentEmbeddingFamilyProducer:
    family: str
    _request_binding_sha256: str = field(repr=False)
    _identity: Mapping[str, Any] = field(repr=False)
    _evidence_payload: Mapping[str, Any] = field(repr=False)
    _evidence_item_count: int = field(repr=False)
    _replay_canary: CumulativeSpentReplayCanary = field(repr=False)
    _capture_artifact_path: str = field(repr=False)
    _execution_record_path: str = field(repr=False)
    _execution_artifact_sha256: str = field(repr=False)
    _embedding_provider: BoundSpentFrozenChunkEmbeddingProvider = field(repr=False)

    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(_validate_identity(self._identity, family=self.family))

    def _revalidate(
        self,
        request: CumulativeSpentStage1FamilyRequest,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if request.family != self.family or request.binding_sha256 != self._request_binding_sha256:
            raise ValueError("cumulative embedding producer was invoked for another request")
        self._replay_canary.assert_matches(request)
        capture_dir = Path(self._capture_artifact_path)
        treatment, outcome = _canonical_labels(request)
        metadata = validate_embedding_native_capture(
            capture_dir,
            embedding_provider=self._embedding_provider,
            fit_texts=tuple(row.text for row in request.spent_rows),
            expected_fit_treatment=treatment,
            expected_fit_outcome=outcome,
            expected_discovery_projection={"_oci_row_id": list(request.spent_row_ids)},
            expected_scope_id=request.scope_id,
            expected_fit_row_ids=request.spent_row_ids,
            expected_heldout_row_ids=(self._replay_canary.alias_row_id,),
        )
        expected_identity = cumulative_spent_embedding_family_identity(
            family=self.family,
            capture_configuration=_capture_configuration(metadata),
        )
        if _validate_identity(self._identity, family=self.family) != expected_identity:
            raise RuntimeError("cumulative embedding identity differs from native config")
        policy = _validate_policy(metadata.get("tfidf_training_scope_policy"), request=request)
        payloads, counts = _payloads_from_capture(
            capture_dir,
            request=request,
            metadata=metadata,
        )
        if (
            payloads[self.family] != dict(self._evidence_payload)
            or counts[self.family] != self._evidence_item_count
        ):
            raise RuntimeError("cumulative embedding payload changed after native capture")
        expected = _execution_record(
            request=request,
            replay_canary=self._replay_canary,
            identity=self._identity,
            metadata=metadata,
            evidence_payload=self._evidence_payload,
            evidence_item_count=self._evidence_item_count,
            capture_dir=capture_dir,
        )
        persisted, file_sha256 = _read_stable_json(
            self._execution_record_path,
            field_name="cumulative embedding execution record",
        )
        _validate_record_envelope(persisted, family=self.family)
        if persisted != expected:
            raise RuntimeError("component-emitted cumulative embedding execution record changed")
        if file_sha256 != self._execution_artifact_sha256:
            raise RuntimeError("cumulative embedding execution artifact bytes changed")
        return persisted, policy

    def produce_cumulative_spent(
        self,
        request: CumulativeSpentStage1FamilyRequest,
    ) -> CumulativeSpentFamilyEvidenceDraft:
        record, validated_policy = self._revalidate(request)
        audit = _fit_audit_from_record(
            request=request,
            record=record,
            execution_artifact_sha256=self._execution_artifact_sha256,
            semantic_policy=validated_policy,
        )
        return CumulativeSpentFamilyEvidenceDraft(
            evidence_payload=copy.deepcopy(dict(self._evidence_payload)),
            evidence_item_count=self._evidence_item_count,
            input_binding_sha256=request.binding_sha256,
            fit_semantics=CUMULATIVE_SPENT_REFIT,
            fit_audit=audit,
        )


def bind_persisted_cumulative_spent_embedding_producers(
    *,
    requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    replay_canary: CumulativeSpentReplayCanary,
    embedding_provider: BoundSpentFrozenChunkEmbeddingProvider,
    producer_identity_by_family: Mapping[str, Mapping[str, Any]],
    evidence_payload_by_family: Mapping[str, Mapping[str, Any]],
    evidence_item_count_by_family: Mapping[str, int],
    capture_artifact_path: Path | str,
    execution_record_path_by_family: Mapping[str, Path | str],
    expected_fit_audit_by_family: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, NativeCumulativeSpentEmbeddingFamilyProducer]:
    """Bind persisted shared-embedding artifacts into revalidating producers."""

    typed = _request_map(requests, replay_canary)
    validated = validate_cumulative_spent_embedding_artifacts(
        requests=typed,
        replay_canary=replay_canary,
        embedding_provider=embedding_provider,
        producer_identity_by_family=producer_identity_by_family,
        evidence_payload_by_family=evidence_payload_by_family,
        evidence_item_count_by_family=evidence_item_count_by_family,
        capture_artifact_path=capture_artifact_path,
        execution_record_path_by_family=execution_record_path_by_family,
        expected_fit_audit_by_family=expected_fit_audit_by_family,
    )
    capture_root = Path(capture_artifact_path)
    if capture_root.is_symlink() or not capture_root.is_dir():
        raise ValueError("embedding reload capture root must be one real directory")
    capture_dir = capture_root.resolve(strict=True)
    output: dict[str, NativeCumulativeSpentEmbeddingFamilyProducer] = {}
    for family in sorted(CUMULATIVE_SPENT_EMBEDDING_FAMILIES):
        producer = NativeCumulativeSpentEmbeddingFamilyProducer(
            family=family,
            _request_binding_sha256=typed[family].binding_sha256,
            _identity=copy.deepcopy(dict(producer_identity_by_family[family])),
            _evidence_payload=copy.deepcopy(dict(evidence_payload_by_family[family])),
            _evidence_item_count=int(evidence_item_count_by_family[family]),
            _replay_canary=replay_canary,
            _capture_artifact_path=str(capture_dir),
            _execution_record_path=str(
                Path(execution_record_path_by_family[family]).resolve(strict=True)
            ),
            _execution_artifact_sha256=str(validated[family]["execution_artifact_sha256"]),
            _embedding_provider=embedding_provider,
        )
        producer._revalidate(typed[family])
        output[family] = producer
    return output


def bind_cumulative_spent_embedding_family_producer(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    replay_canary: CumulativeSpentReplayCanary,
    emission: CumulativeSpentEmbeddingFamilyEmission,
) -> NativeCumulativeSpentEmbeddingFamilyProducer:
    """Bind one family only from an issuer-authenticated same-process emission."""

    if type(emission) is not CumulativeSpentEmbeddingFamilyEmission:
        raise TypeError("binding requires a same-process cumulative embedding emission")
    if emission._authority is not _EMISSION_AUTHORITY:
        raise TypeError("cumulative embedding emission authority is invalid")
    if not isinstance(request, CumulativeSpentStage1FamilyRequest):
        raise TypeError("binding requires a typed cumulative spent request")
    if not isinstance(replay_canary, CumulativeSpentReplayCanary):
        raise TypeError("binding requires a cumulative replay canary")
    replay_canary.assert_matches(request)
    if (
        request.family != emission.family
        or request.binding_sha256 != emission.request_binding_sha256
    ):
        raise ValueError("cumulative embedding emission belongs to another request")
    producer = NativeCumulativeSpentEmbeddingFamilyProducer(
        family=request.family,
        _request_binding_sha256=request.binding_sha256,
        _identity=copy.deepcopy(emission._identity),
        _evidence_payload=copy.deepcopy(emission._evidence_payload),
        _evidence_item_count=emission._evidence_item_count,
        _replay_canary=replay_canary,
        _capture_artifact_path=emission.capture_artifact_path,
        _execution_record_path=emission.execution_record_path,
        _execution_artifact_sha256=emission.execution_artifact_sha256,
        _embedding_provider=emission._embedding_provider,
    )
    producer._revalidate(request)
    return producer


__all__ = [
    "CUMULATIVE_SPENT_EMBEDDING_ADAPTER_VERSION",
    "CUMULATIVE_SPENT_EMBEDDING_EMISSION_SCHEMA",
    "CUMULATIVE_SPENT_EMBEDDING_EXECUTION_RECORD_SCHEMA",
    "CUMULATIVE_SPENT_EMBEDDING_FAMILIES",
    "CUMULATIVE_SPENT_EMBEDDING_PAYLOAD_SCHEMA",
    "CUMULATIVE_SPENT_EMBEDDING_RELOAD_VALIDATION_SCHEMA",
    "CumulativeSpentEmbeddingFamilyEmission",
    "NativeCumulativeSpentEmbeddingFamilyProducer",
    "bind_cumulative_spent_embedding_family_producer",
    "bind_persisted_cumulative_spent_embedding_producers",
    "cumulative_spent_embedding_family_identity",
    "emit_cumulative_spent_embedding_capture",
    "validate_cumulative_spent_embedding_artifacts",
    "validate_cumulative_spent_embedding_family_artifact",
]
