"""Fresh cumulative-spent adapters for topic/orphan TF-IDF and neural queries.

These families already have truthful native spent-scope APIs, but their
exact-inner proof records cannot be relabeled for a hierarchy epoch.  The two
emitters below therefore own component execution:

* the paired TF-IDF emitter performs a new nested-calibration fit from the
  canonical spent rows and uses a text-only alias of one spent row only after
  selection is frozen; and
* the neural-query emitter performs a new context fit in the supplied live
  service and immediately snapshots state owned by that same service instance.

Only opaque, process-local emissions can be bound into cumulative producers.
This module deliberately does not register a production bundle or mutate the
existing native components.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from oci.config import AppliedInferenceConfig

from .all_evidence_discovery_interfaces import (
    NEURAL_QUERY_MOMENTS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPICS,
)
from .all_evidence_fusion import (
    NEURAL_QUERY_SOURCE,
    TFIDF_TOPIC_SOURCE,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
)
from .lossless_stage1_evidence_catalog import build_role_neutral_evidence_catalog
from .neural_query_context_backend import (
    NEURAL_QUERY_CONTEXT_SERVICE_ID,
    NEURAL_QUERY_DISCOVERY_RUNTIME_ID,
    NEURAL_QUERY_NUISANCE_OUTPUT_BINDING_SCHEMA,
    NEURAL_QUERY_OWNED_SNAPSHOT_SCHEMA,
    ContextFitNeuralQueryService,
    validate_owned_discovery_snapshot,
)
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
    family_payload_from_catalog,
    native_artifact_sha256,
    native_family_code_identity,
)
from .tfidf_topic_discovery import (
    DISCOVERY_SCHEMA_VERSION,
    row_set_fingerprint,
    stable_hash,
)
from .tfidf_topic_stage1 import (
    TFIDF_NESTED_CALIBRATION_SCHEMA_VERSION,
    _fit_tfidf_topic_context_nested_calibration,
    _float_hex_sha256 as _tfidf_float_hex_sha256,
)

CUMULATIVE_SPENT_REMAINING_ADAPTER_VERSION = "native_cumulative_spent_remaining_families_adapter_v1"
CUMULATIVE_SPENT_REMAINING_EXECUTION_RECORD_SCHEMA = (
    "native_cumulative_spent_remaining_execution_record_v1"
)
CUMULATIVE_SPENT_REMAINING_EMISSION_SCHEMA = "same_process_cumulative_spent_remaining_emission_v1"
CUMULATIVE_SPENT_REMAINING_RELOAD_VALIDATION_SCHEMA = (
    "cumulative_spent_remaining_reload_validation_v1"
)
CUMULATIVE_SPENT_NEURAL_QUERY_POLICY_SCHEMA = "cumulative_spent_neural_query_truthful_policy_v1"

TFIDF_CUMULATIVE_FAMILIES = frozenset({TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS})
REMAINING_CUMULATIVE_FAMILIES = frozenset({TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS, NEURAL_QUERY_MOMENTS})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMISSION_AUTHORITY = object()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
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
        raise RuntimeError("remaining-family adapter changed while hashing")
    return _sha256_bytes(payload)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key in cumulative remaining artifact: {key}")
        output[key] = value
    return output


def _read_stable_json(path: Path | str, *, field_name: str) -> tuple[dict[str, Any], str]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{field_name} must be one regular file")
    before = source.stat()
    payload = source.read_bytes()
    after = source.stat()
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
        raise RuntimeError(f"refusing to replace cumulative remaining artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if target.exists():
            raise RuntimeError(f"refusing to replace cumulative remaining artifact: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_bytes(payload)


def _canonical_labels(
    request: CumulativeSpentStage1FamilyRequest,
) -> tuple[np.ndarray, np.ndarray]:
    treatment = np.asarray([row.treatment for row in request.spent_rows], dtype=float)
    outcome = np.asarray([row.outcome for row in request.spent_rows], dtype=float)
    if (
        treatment.shape != (len(request.spent_rows),)
        or outcome.shape != treatment.shape
        or not np.isfinite(treatment).all()
        or not np.isfinite(outcome).all()
    ):
        raise ValueError("canonical cumulative labels are malformed")
    return treatment, outcome


def _adapter_code_sha256(family: str) -> str:
    if family not in REMAINING_CUMULATIVE_FAMILIES:
        raise ValueError("family has no remaining cumulative adapter")
    return _sha256_json(
        {
            "schema_version": CUMULATIVE_SPENT_REMAINING_ADAPTER_VERSION,
            "adapter_module_sha256": _stable_module_sha256(),
            "native_family_code_identity": native_family_code_identity(family),
        }
    )


def cumulative_spent_remaining_family_identity(
    *,
    family: str,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    if family not in REMAINING_CUMULATIVE_FAMILIES:
        raise ValueError("family has no remaining cumulative adapter")
    if not isinstance(configuration, Mapping):
        raise TypeError("remaining-family configuration must be a mapping")
    config = copy.deepcopy(dict(configuration))
    _canonical_json(config)
    return {
        "schema_version": EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
        "family": family,
        "producer_name": f"native_cumulative_spent_{family}",
        "producer_version": CUMULATIVE_SPENT_REMAINING_ADAPTER_VERSION,
        "code_sha256": _adapter_code_sha256(family),
        "configuration_sha256": _sha256_json(
            {
                "schema_version": CUMULATIVE_SPENT_REMAINING_ADAPTER_VERSION,
                "family": family,
                "configuration": config,
            }
        ),
    }


def _validate_identity(value: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("remaining-family identity must be a mapping")
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
        raise ValueError("remaining-family identity is not a closed schema")
    if (
        identity.get("schema_version") != EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION
        or identity.get("family") != family
        or identity.get("producer_name") != f"native_cumulative_spent_{family}"
        or identity.get("producer_version") != CUMULATIVE_SPENT_REMAINING_ADAPTER_VERSION
        or identity.get("code_sha256") != _adapter_code_sha256(family)
    ):
        raise ValueError("remaining-family identity changed its native implementation")
    _require_sha256(identity.get("configuration_sha256"), field_name="configuration_sha256")
    return identity


def _paired_tfidf_requests(
    requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    replay_canary: CumulativeSpentReplayCanary,
) -> dict[str, CumulativeSpentStage1FamilyRequest]:
    if not isinstance(requests, Mapping) or set(requests) != set(TFIDF_CUMULATIVE_FAMILIES):
        raise ValueError("TF-IDF emission requires exactly topic and orphan requests")
    result: dict[str, CumulativeSpentStage1FamilyRequest] = {}
    reference: CumulativeSpentStage1FamilyRequest | None = None
    for family in sorted(TFIDF_CUMULATIVE_FAMILIES):
        request = requests[family]
        if not isinstance(request, CumulativeSpentStage1FamilyRequest):
            raise TypeError("TF-IDF emission requires typed cumulative requests")
        if request.family != family:
            raise ValueError("TF-IDF request mapping changed a family")
        replay_canary.assert_matches(request)
        if reference is None:
            reference = request
        elif request.binding != reference.binding:
            raise ValueError("TF-IDF requests do not describe one cumulative scope")
        result[family] = request
    return result


def _component_file(root: Path, value: Any, *, field_name: str) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field_name} must be one regular component file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{field_name} escapes its fresh component") from exc
    return resolved


_TFIDF_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "scope_id",
        "fit_row_fingerprint",
        "heldout_row_fingerprint",
        "fit_row_ids",
        "heldout_row_ids",
        "config_hash",
        "common_vocabulary_size",
        "common_vocabulary",
        "nuisance",
        "topic_banks",
        "heldout_score_tests_enabled",
        "topic_score_tests",
        "artifacts",
        "model_fit_row_ids",
        "model_fit_row_fingerprint",
        "registered_fit_treatment_sha256",
        "registered_fit_outcome_sha256",
        "nested_model_fit_treatment_sha256",
        "nested_model_fit_outcome_sha256",
        "nested_calibration_treatment_sha256",
        "nested_calibration_outcome_sha256",
        "score_selection_label_policy",
        "selection_nesting",
        "selection_frozen_sha256",
        "registered_heldout_columns_read",
        "registered_heldout_labels_accessed",
    }
)


def _tfidf_policy(
    metadata: Mapping[str, Any],
    *,
    request: CumulativeSpentStage1FamilyRequest,
) -> dict[str, Any]:
    nesting = metadata.get("selection_nesting")
    if not isinstance(nesting, Mapping):
        raise ValueError("TF-IDF context has no nested training-only policy")
    policy = copy.deepcopy(dict(nesting))
    model_rows = tuple(map(int, policy.get("model_fit_row_ids") or ()))
    calibration_rows = tuple(map(int, policy.get("calibration_row_ids") or ()))
    if (
        policy.get("schema_version") != TFIDF_NESTED_CALIBRATION_SCHEMA_VERSION
        or policy.get("policy") != "nested_fit_calibration"
        or policy.get("fold_parameter") != "tfidf_nested_calibration_folds"
        or policy.get("canonical_hierarchy_partition_count_used") is not False
        or policy.get("interaction_inner_folds_used") is not False
        or policy.get("registered_heldout_labels_accessed") is not False
        or policy.get("nested_calibration_labels_accessed") is not True
        or policy.get("selection_frozen_before_registered_heldout_transform") is not True
        or not model_rows
        or not calibration_rows
        or set(model_rows) & set(calibration_rows)
        or set(model_rows) | set(calibration_rows) != set(request.spent_row_ids)
    ):
        raise ValueError("TF-IDF selection is not nested inside exact spent training rows")
    return {
        **policy,
        "selection_frozen_sha256": _require_sha256(
            metadata.get("selection_frozen_sha256"),
            field_name="selection_frozen_sha256",
        ),
        "registered_fit_treatment_sha256": metadata.get("registered_fit_treatment_sha256"),
        "registered_fit_outcome_sha256": metadata.get("registered_fit_outcome_sha256"),
        "nested_model_fit_treatment_sha256": metadata.get("nested_model_fit_treatment_sha256"),
        "nested_model_fit_outcome_sha256": metadata.get("nested_model_fit_outcome_sha256"),
        "nested_calibration_treatment_sha256": metadata.get("nested_calibration_treatment_sha256"),
        "nested_calibration_outcome_sha256": metadata.get("nested_calibration_outcome_sha256"),
    }


def _validate_tfidf_context(
    *,
    artifact_dir: Path,
    request: CumulativeSpentStage1FamilyRequest,
    replay_canary: CumulativeSpentReplayCanary,
    expected_text_column: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    metadata_path = artifact_dir / "context_metadata.json"
    metadata, _metadata_file_sha = _read_stable_json(
        metadata_path,
        field_name="cumulative TF-IDF context metadata",
    )
    if set(metadata) != set(_TFIDF_METADATA_FIELDS):
        raise ValueError("cumulative TF-IDF context metadata is not a closed schema")
    treatment, outcome = _canonical_labels(request)
    if (
        metadata.get("schema_version") != DISCOVERY_SCHEMA_VERSION
        or metadata.get("scope_id") != request.scope_id
        or tuple(map(int, metadata.get("fit_row_ids") or ())) != request.spent_row_ids
        or tuple(map(int, metadata.get("heldout_row_ids") or ())) != (replay_canary.alias_row_id,)
        or metadata.get("fit_row_fingerprint") != row_set_fingerprint(request.spent_row_ids)
        or metadata.get("heldout_row_fingerprint")
        != row_set_fingerprint((replay_canary.alias_row_id,))
        or metadata.get("registered_fit_treatment_sha256") != _tfidf_float_hex_sha256(treatment)
        or metadata.get("registered_fit_outcome_sha256") != _tfidf_float_hex_sha256(outcome)
        or metadata.get("score_selection_label_policy") != "nested_fit_calibration"
        or metadata.get("registered_heldout_columns_read")
        != ["_oci_row_id", str(expected_text_column)]
        or metadata.get("registered_heldout_labels_accessed") is not False
        or metadata.get("heldout_score_tests_enabled") is not True
    ):
        raise ValueError("cumulative TF-IDF context changed its spent scope or labels")
    policy = _tfidf_policy(metadata, request=request)
    model_rows = tuple(map(int, policy["model_fit_row_ids"]))
    calibration_rows = tuple(map(int, policy["calibration_row_ids"]))
    positions = {row_id: index for index, row_id in enumerate(request.spent_row_ids)}
    expected_hashes = {
        "nested_model_fit_treatment_sha256": _tfidf_float_hex_sha256(
            treatment[[positions[row_id] for row_id in model_rows]]
        ),
        "nested_model_fit_outcome_sha256": _tfidf_float_hex_sha256(
            outcome[[positions[row_id] for row_id in model_rows]]
        ),
        "nested_calibration_treatment_sha256": _tfidf_float_hex_sha256(
            treatment[[positions[row_id] for row_id in calibration_rows]]
        ),
        "nested_calibration_outcome_sha256": _tfidf_float_hex_sha256(
            outcome[[positions[row_id] for row_id in calibration_rows]]
        ),
    }
    if any(metadata.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("TF-IDF nested partition labels differ from canonical spent labels")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("cumulative TF-IDF context has no native artifacts")
    model_path = _component_file(
        artifact_dir,
        artifacts.get("fitted_context"),
        field_name="cumulative TF-IDF fitted context",
    )
    score_path = _component_file(
        artifact_dir,
        artifacts.get("topic_score_tests"),
        field_name="cumulative TF-IDF score-selection artifact",
    )
    score, _score_file_sha = _read_stable_json(
        score_path,
        field_name="cumulative TF-IDF score-selection artifact",
    )
    orphan = score.get("effect_orphan_ngram_branch")
    if (
        score.get("status") != "completed"
        or score.get("score_selection_label_policy") != "nested_fit_calibration"
        or score.get("uses_heldout_treatment_and_outcome") is not False
        or score.get("uses_registered_heldout_treatment_and_outcome") is not False
        or score.get("uses_nested_fit_calibration_treatment_and_outcome") is not True
        or score.get("selection_frozen_sha256") != policy["selection_frozen_sha256"]
        or not isinstance(orphan, Mapping)
        or orphan.get("uses_registered_heldout_treatment_and_outcome") is not False
        or orphan.get("uses_nested_fit_calibration_treatment_and_outcome") is not True
        or orphan.get("cluster_construction_uses_heldout_rows_or_labels") is not False
        or orphan.get("topic_term_exclusion_is_fit_side") is not True
    ):
        raise ValueError("TF-IDF source violates nested training-only selection")
    return metadata, policy, model_path, score_path


def _concept_term(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    term = str(raw.get("term") or raw.get("feature") or raw.get("ngram") or "").strip()
    if not term:
        return None
    row: dict[str, Any] = {"term": term}
    for key in (
        "combined_importance",
        "fit_rank",
        "fit_signed_score",
        "signed_score",
        "support_control",
        "support_treated",
    ):
        if raw.get(key) is not None:
            row[key] = raw[key]
    similarity = raw.get("lexical_similarity_to_seed")
    if similarity is None:
        similarity = raw.get("cluster_seed_similarity")
    if similarity is not None:
        row["lexical_similarity_to_seed"] = similarity
    return row


def _concept_cluster(raw: Mapping[str, Any]) -> dict[str, Any]:
    cluster_id = str(raw.get("cluster_id") or raw.get("topic_id") or "").strip()
    values = raw.get("terms")
    if values is None:
        values = raw.get("member_terms", raw.get("supporting_terms"))
    if values is None:
        values = raw.get("term_scores")
    if not cluster_id or not isinstance(values, (list, tuple)):
        raise ValueError("native orphan cluster has no identity or term collection")
    terms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        term_row = {"term": value} if isinstance(value, str) else value
        if not isinstance(term_row, Mapping):
            raise ValueError("native orphan cluster contains a malformed term")
        candidates = [term_row]
        aliases = term_row.get("nested_aliases")
        if aliases is not None:
            if not isinstance(aliases, (list, tuple)):
                raise ValueError("native orphan nested aliases are malformed")
            candidates.extend(aliases)
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("native orphan alias is malformed")
            compact = _concept_term(candidate)
            if compact is not None and compact["term"] not in seen:
                seen.add(compact["term"])
                terms.append(compact)
    if not terms:
        raise ValueError("native orphan cluster has no lossless concept terms")
    ranks = [int(row["fit_rank"]) for row in terms if row.get("fit_rank") is not None]
    scores = [
        abs(float(row.get("fit_signed_score", row.get("signed_score"))))
        for row in terms
        if row.get("fit_signed_score", row.get("signed_score")) is not None
    ]
    return {
        "cluster_id": cluster_id,
        "evidence_kind": str(raw.get("evidence_kind") or "orphan_raw_ngram_cluster"),
        "terms": terms,
        "seed_term": str(raw.get("seed_term") or terms[0]["term"]),
        "fit_rank": min(ranks) if ranks else None,
        "maximum_abs_fit_signed_score": max(scores) if scores else None,
        "grouping_method": str(
            raw.get("grouping_method") or "native_fit_side_orphan_ngram_cluster"
        ),
    }


def _tfidf_catalog_projection(
    metadata: Mapping[str, Any],
    score: Mapping[str, Any],
) -> dict[str, Any]:
    raw_orphan = score.get("effect_orphan_ngram_branch")
    if not isinstance(raw_orphan, Mapping):
        raise ValueError("native nested TF-IDF score has no orphan branch")

    def clusters(key: str) -> list[dict[str, Any]]:
        values = raw_orphan.get(key) or []
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"native orphan {key} must be a sequence")
        if not all(isinstance(value, Mapping) for value in values):
            raise ValueError(f"native orphan {key} contains a malformed cluster")
        return [_concept_cluster(value) for value in values]

    selected = clusters("selected_clusters")
    all_clusters = clusters("clusters")
    selected_ids = [str(value) for value in raw_orphan.get("selected_cluster_ids") or ()]
    if set(selected_ids) != {row["cluster_id"] for row in selected}:
        raise ValueError("native orphan selected IDs differ from selected clusters")
    return {
        "topic_banks": copy.deepcopy(metadata.get("topic_banks")),
        "effect_orphan_ngram_branch": {
            "schema_version": "tfidf_nested_fit_orphan_concept_projection_v1",
            "status": raw_orphan.get("status"),
            "candidate_definition": raw_orphan.get("candidate_definition"),
            "uses_outer_heldout_labels": False,
            "uses_heldout_treatment_and_outcome": False,
            "fits_patient_level_cate_model": False,
            "topic_term_exclusion_is_fit_side": raw_orphan.get("topic_term_exclusion_is_fit_side"),
            "cluster_construction_uses_heldout_rows_or_labels": raw_orphan.get(
                "cluster_construction_uses_heldout_rows_or_labels"
            ),
            "candidate_count_before_topic_exclusion": raw_orphan.get(
                "candidate_count_before_topic_exclusion"
            ),
            "represented_topic_term_exclusion_count": raw_orphan.get(
                "represented_topic_term_exclusion_count"
            ),
            "candidate_count_before_nested_deduplication": raw_orphan.get(
                "candidate_count_before_nested_deduplication"
            ),
            "deduplicated_alias_count": raw_orphan.get("deduplicated_alias_count"),
            "representative_count": raw_orphan.get("representative_count"),
            "cluster_count": len(all_clusters),
            "selected_cluster_ids": selected_ids,
            "selected_clusters": selected,
            "clusters": all_clusters,
            "selection_count": len(selected),
            "selection_rule": raw_orphan.get("selection_rule"),
            "minimum_selected_clusters": raw_orphan.get("minimum_selected_clusters"),
            "maximum_selected_clusters": raw_orphan.get("maximum_selected_clusters"),
        },
    }


def _catalog_payloads(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    source_kind: str,
    payload: Mapping[str, Any],
    families: Sequence[str],
    heldout_row_ids: Sequence[int],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    provenance = FoldEvidenceProvenance(
        outer_fold=request.outer_fold,
        train_row_ids=request.spent_row_ids,
        heldout_row_ids=tuple(map(int, heldout_row_ids)),
        scope="inner_train",
        inner_fold=request.provider_inner_fold,
        artifact_id=f"cumulative-{source_kind}-{request.scope_id}",
    )
    if provenance.split_fingerprint != request.split_scope_fingerprint:
        raise ValueError(
            "cumulative remaining evidence provenance differs from the canonical split"
        )
    catalog = build_role_neutral_evidence_catalog(
        (FoldEvidenceInput(source_kind, copy.deepcopy(dict(payload)), provenance),),
        require_all_source_kinds=False,
        require_all_architecture_families=False,
        require_upstream_completeness=True,
    )
    projected: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for family in families:
        value, count = family_payload_from_catalog(catalog, family=family)
        if int(count) < 1 or not value.get("architecture_evidence"):
            raise ValueError(f"native cumulative scope has no lossless evidence for {family}")
        projected[family] = value
        counts[family] = int(count)
    return projected, counts


def _tfidf_configuration(
    config: AppliedInferenceConfig, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    nn_config = config.architecture.multi_model_forest
    return {
        "text_column": config.text_column,
        "treatment_column": config.treatment_column,
        "outcome_column": config.outcome_column,
        "outcome_type": config.outcome_type,
        "seed": int(getattr(config, "seed", 42)),
        "nuisance_folds": int(nn_config.nuisance_folds),
        "bow_views": [asdict(value) for value in nn_config.bow_views],
        "tfidf_nested_calibration_folds": int(nn_config.tfidf_nested_calibration_folds),
        "topic_configuration": asdict(nn_config.tfidf_topic),
        "topic_configuration_hash": metadata.get("config_hash"),
        "score_selection_label_policy": "nested_fit_calibration",
        "heldout_alias_policy": "id_and_spent_text_only_after_selection_freeze",
    }


def _validated_tfidf_configuration(
    config: AppliedInferenceConfig,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(config, AppliedInferenceConfig):
        raise TypeError("TF-IDF reload requires AppliedInferenceConfig")
    topic_config = config.architecture.multi_model_forest.tfidf_topic
    if (
        config.text_column in {config.treatment_column, config.outcome_column, "_oci_row_id"}
        or str(topic_config.score_selection_label_policy) != "nested_fit_calibration"
        or not bool(topic_config.score_test_enabled)
        or not bool(topic_config.orphan_ngram_enabled)
        or int(config.architecture.multi_model_forest.tfidf_nested_calibration_folds) < 2
        or metadata.get("config_hash") != stable_hash(asdict(topic_config))
    ):
        raise ValueError("persisted TF-IDF artifacts differ from the expected native config")
    return _tfidf_configuration(config, metadata)


def _query_label_sha256(values: Sequence[float]) -> str:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("neural-query label vector must be finite and one-dimensional")
    return _sha256_json([float(value).hex() for value in vector])


_QUERY_SOURCE_SCHEMA = "cumulative_spent_neural_query_safe_evidence_v1"
_QUERY_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "source_family",
        "scope_id",
        "outer_fold",
        "context_epoch",
        "provider_inner_fold",
        "request_binding_sha256",
        "spent_row_order_fingerprint",
        "sealed_row_order_fingerprint",
        "query_cache_key",
        "owned_snapshot_content_sha256",
        "query_evidence",
        "all_queries_retained",
        "validation_audits_used_for_selection",
        "statistical_gate_applied",
        "sealed_text_accessed",
        "sealed_labels_accessed",
        "row_level_excerpts_emitted",
    }
)


def _query_policy(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    discovery = snapshot.get("discovery_metadata")
    nuisance = (
        discovery.get("fit_nuisance_output_binding") if isinstance(discovery, Mapping) else None
    )
    if (
        not isinstance(discovery, Mapping)
        or discovery.get("runtime") != NEURAL_QUERY_DISCOVERY_RUNTIME_ID
        or discovery.get("all_queries_retained") is not True
        or discovery.get("validation_audits_used_for_selection") is not False
        or discovery.get("executable_checkpoint_io") is not False
        or not isinstance(nuisance, Mapping)
        or nuisance.get("schema_version") != NEURAL_QUERY_NUISANCE_OUTPUT_BINDING_SCHEMA
        or nuisance.get("heldout_labels_accessed") is not False
    ):
        raise ValueError("neural-query owned snapshot changed its truthful ungated policy")
    return {
        "schema_version": CUMULATIVE_SPENT_NEURAL_QUERY_POLICY_SCHEMA,
        "policy": "exact_spent_fit_ungated_all_queries_retained",
        "fit_treatment_and_outcome_used": True,
        "sealed_treatment_and_outcome_used": False,
        "all_queries_retained": True,
        "validation_audits_used_for_selection": False,
        "statistical_gate_applied": False,
        "fit_nuisance_heldout_labels_accessed": False,
        "snapshot_source": snapshot.get("snapshot_source"),
        "joblib_checkpoint_loaded_for_snapshot": snapshot.get("joblib_checkpoint_loaded"),
    }


def _validated_service_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("neural-query reload requires an expected service identity mapping")
    identity = copy.deepcopy(dict(value))
    _canonical_json(identity)
    text_column = identity.get("text_column")
    row_count = identity.get("dataset_row_count")
    if (
        identity.get("service") != NEURAL_QUERY_CONTEXT_SERVICE_ID
        or not isinstance(text_column, str)
        or not text_column.strip()
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 1
        or identity.get("gate_labels_accepted") is not False
        or identity.get("novel_semantic_encoding_allowed") is not False
        or identity.get("preexisting_executable_cache_entries_accepted") is not False
        or identity.get("executable_cache_reuse_scope") != "current_service_instance_only"
    ):
        raise ValueError("expected neural-query service identity changed its security envelope")
    return identity


def _query_configuration(
    *,
    service_identity: Mapping[str, Any],
    query_cache_key: str,
) -> dict[str, Any]:
    identity = _validated_service_identity(service_identity)
    return {
        "service_identity": identity,
        "query_cache_key": _require_sha256(query_cache_key, field_name="query_cache_key"),
        "text_column": identity["text_column"],
        "policy_schema_version": CUMULATIVE_SPENT_NEURAL_QUERY_POLICY_SCHEMA,
        "sealed_input_policy": "sealed_row_ids_only",
    }


def _validate_query_artifacts(
    *,
    model_path: Path,
    source_path: Path,
    request: CumulativeSpentStage1FamilyRequest,
    expected_service_identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    source, _source_file_sha = _read_stable_json(
        source_path,
        field_name="cumulative neural-query safe evidence artifact",
    )
    if set(source) != set(_QUERY_SOURCE_FIELDS):
        raise ValueError("cumulative neural-query source is not a closed schema")
    if (
        source.get("schema_version") != _QUERY_SOURCE_SCHEMA
        or source.get("source_family") != NEURAL_QUERY_MOMENTS
        or source.get("scope_id") != request.scope_id
        or int(source.get("outer_fold", 0)) != request.outer_fold
        or int(source.get("context_epoch", -1)) != request.context_epoch
        or int(source.get("provider_inner_fold", 0)) != request.provider_inner_fold
        or source.get("request_binding_sha256") != request.binding_sha256
        or source.get("spent_row_order_fingerprint") != row_order_fingerprint(request.spent_row_ids)
        or source.get("sealed_row_order_fingerprint")
        != row_order_fingerprint(request.sealed_row_ids)
        or source.get("all_queries_retained") is not True
        or source.get("validation_audits_used_for_selection") is not False
        or source.get("statistical_gate_applied") is not False
        or source.get("sealed_text_accessed") is not False
        or source.get("sealed_labels_accessed") is not False
        or source.get("row_level_excerpts_emitted") is not False
        or not isinstance(source.get("query_evidence"), list)
        or not source["query_evidence"]
    ):
        raise ValueError("cumulative neural-query source changed scope or security policy")
    cache_key = _require_sha256(source.get("query_cache_key"), field_name="query_cache_key")
    expected_service_sha256 = None
    if expected_service_identity is not None:
        expected_service_sha256 = _sha256_json(
            _validated_service_identity(expected_service_identity)
        )
    snapshot = validate_owned_discovery_snapshot(
        model_path,
        expected_cache_key=cache_key,
        expected_service_identity_sha256=expected_service_sha256,
    )
    binding = snapshot.get("binding")
    treatment, outcome = _canonical_labels(request)
    if (
        not isinstance(binding, Mapping)
        or tuple(map(int, binding.get("row_ids") or ())) != request.spent_row_ids
        or int(binding.get("outer_fold", 0)) != request.outer_fold
        or int(binding.get("row_count", 0)) != len(request.spent_row_ids)
        or binding.get("text_sha256") != _sha256_json([row.text for row in request.spent_rows])
        or binding.get("treatment_sha256") != _query_label_sha256(treatment)
        or binding.get("outcome_sha256") != _query_label_sha256(outcome)
        or source.get("owned_snapshot_content_sha256") != snapshot.get("content_sha256")
    ):
        raise ValueError("neural-query owned snapshot differs from canonical spent inputs")
    policy = _query_policy(snapshot)
    payloads, counts = _catalog_payloads(
        request=request,
        source_kind=NEURAL_QUERY_SOURCE,
        payload={"query_evidence": copy.deepcopy(source["query_evidence"])},
        families=(NEURAL_QUERY_MOMENTS,),
        heldout_row_ids=request.sealed_row_ids,
    )
    return snapshot, policy, payloads[NEURAL_QUERY_MOMENTS], counts[NEURAL_QUERY_MOMENTS]


_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "family",
        "native_kind",
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
        "native_metadata_sha256",
        "native_training_policy",
        "native_training_policy_sha256",
        "canonical_treatment_sha256",
        "canonical_outcome_sha256",
        "model_artifact_sha256",
        "source_artifact_sha256",
        "evidence_payload_sha256",
        "evidence_item_count",
        "replay_canary",
        "fit_input_columns",
        "sealed_text_accessed",
        "sealed_labels_accessed",
        "oracle_fields_accessed",
        "secrets_accessed",
        "replay_canary_contributes_to_concept_evidence",
    }
)


def _execution_record(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    replay_canary: CumulativeSpentReplayCanary,
    native_kind: str,
    identity: Mapping[str, Any],
    native_metadata_sha256: str,
    native_policy: Mapping[str, Any],
    model_path: Path,
    source_path: Path,
    evidence_payload: Mapping[str, Any],
    evidence_item_count: int,
) -> dict[str, Any]:
    validated_identity = _validate_identity(identity, family=request.family)
    treatment, outcome = _canonical_labels(request)
    record = {
        "schema_version": CUMULATIVE_SPENT_REMAINING_EXECUTION_RECORD_SCHEMA,
        "status": "completed",
        "family": request.family,
        "native_kind": native_kind,
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
        "native_metadata_sha256": _require_sha256(
            native_metadata_sha256,
            field_name="native_metadata_sha256",
        ),
        "native_training_policy": copy.deepcopy(dict(native_policy)),
        "native_training_policy_sha256": _sha256_json(native_policy),
        "canonical_treatment_sha256": _query_label_sha256(treatment),
        "canonical_outcome_sha256": _query_label_sha256(outcome),
        "model_artifact_sha256": native_artifact_sha256(model_path),
        "source_artifact_sha256": native_artifact_sha256(source_path),
        "evidence_payload_sha256": _sha256_json(copy.deepcopy(dict(evidence_payload))),
        "evidence_item_count": int(evidence_item_count),
        "replay_canary": replay_canary.binding,
        "fit_input_columns": ["_oci_row_id", "text", "treatment", "outcome"],
        "sealed_text_accessed": False,
        "sealed_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "replay_canary_contributes_to_concept_evidence": False,
    }
    if set(record) != set(_RECORD_FIELDS):
        raise RuntimeError("remaining-family execution record is not closed")
    return record


def _validate_record(record: Mapping[str, Any], *, family: str, native_kind: str) -> None:
    if set(record) != set(_RECORD_FIELDS):
        raise ValueError("remaining-family execution record is not a closed schema")
    if (
        record.get("schema_version") != CUMULATIVE_SPENT_REMAINING_EXECUTION_RECORD_SCHEMA
        or record.get("status") != "completed"
        or record.get("family") != family
        or record.get("native_kind") != native_kind
        or record.get("scope") != "cumulative_spent_train"
        or record.get("fit_semantics") != CUMULATIVE_SPENT_REFIT
        or record.get("sealed_text_accessed") is not False
        or record.get("sealed_labels_accessed") is not False
        or record.get("oracle_fields_accessed") is not False
        or record.get("secrets_accessed") is not False
        or record.get("replay_canary_contributes_to_concept_evidence") is not False
    ):
        raise ValueError("remaining-family execution record changed its security envelope")
    for key in (
        "producer_identity_sha256",
        "producer_code_sha256",
        "configuration_sha256",
        "native_metadata_sha256",
        "native_training_policy_sha256",
        "canonical_treatment_sha256",
        "canonical_outcome_sha256",
        "model_artifact_sha256",
        "source_artifact_sha256",
        "evidence_payload_sha256",
    ):
        _require_sha256(record.get(key), field_name=key)
    if (
        isinstance(record.get("evidence_item_count"), bool)
        or int(record.get("evidence_item_count", 0)) < 1
    ):
        raise ValueError("remaining-family execution record has no evidence")


def _fit_audit_from_record(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    record: Mapping[str, Any],
    execution_artifact_sha256: str,
    native_policy: Mapping[str, Any],
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
            copy.deepcopy(dict(native_policy))
            if request.family in TFIDF_CUMULATIVE_FAMILIES
            else None
        ),
    }


@dataclass(frozen=True)
class CumulativeSpentRemainingFamilyEmission:
    family: str
    native_kind: str
    request_binding_sha256: str
    native_metadata_path: str
    model_artifact_path: str
    source_artifact_path: str
    execution_record_path: str
    execution_artifact_sha256: str
    expected_text_column: str
    _configuration: Mapping[str, Any] = field(repr=False)
    _identity: Mapping[str, Any] = field(repr=False)
    _evidence_payload: Mapping[str, Any] = field(repr=False)
    _evidence_item_count: int = field(repr=False)
    _native_policy: Mapping[str, Any] = field(repr=False)
    _authority: object = field(repr=False, compare=False)
    schema_version: str = CUMULATIVE_SPENT_REMAINING_EMISSION_SCHEMA

    def __post_init__(self) -> None:
        if self._authority is not _EMISSION_AUTHORITY:
            raise TypeError("remaining-family emissions must be issued by a live component")
        if (
            self.schema_version != CUMULATIVE_SPENT_REMAINING_EMISSION_SCHEMA
            or self.family not in REMAINING_CUMULATIVE_FAMILIES
            or self.native_kind not in {"nested_tfidf", "owned_neural_query"}
        ):
            raise ValueError("invalid remaining-family emission")
        _require_sha256(self.request_binding_sha256, field_name="request_binding_sha256")
        _require_sha256(
            self.execution_artifact_sha256,
            field_name="execution_artifact_sha256",
        )
        if not isinstance(self.expected_text_column, str) or not self.expected_text_column:
            raise ValueError("remaining-family emission requires its canonical text column")


def _issue_emission(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    replay_canary: CumulativeSpentReplayCanary,
    native_kind: str,
    configuration: Mapping[str, Any],
    identity: Mapping[str, Any],
    native_metadata_path: Path,
    model_path: Path,
    source_path: Path,
    policy: Mapping[str, Any],
    payload: Mapping[str, Any],
    count: int,
    record_path: Path,
    expected_text_column: str,
) -> CumulativeSpentRemainingFamilyEmission:
    expected_identity = cumulative_spent_remaining_family_identity(
        family=request.family,
        configuration=configuration,
    )
    if _validate_identity(identity, family=request.family) != expected_identity:
        raise ValueError("remaining-family emission identity differs from its native config")
    record = _execution_record(
        request=request,
        replay_canary=replay_canary,
        native_kind=native_kind,
        identity=identity,
        native_metadata_sha256=native_artifact_sha256(native_metadata_path),
        native_policy=policy,
        model_path=model_path,
        source_path=source_path,
        evidence_payload=payload,
        evidence_item_count=count,
    )
    execution_sha256 = _write_immutable_json(record_path, record)
    return CumulativeSpentRemainingFamilyEmission(
        family=request.family,
        native_kind=native_kind,
        request_binding_sha256=request.binding_sha256,
        native_metadata_path=str(native_metadata_path.resolve(strict=True)),
        model_artifact_path=str(model_path.resolve(strict=True)),
        source_artifact_path=str(source_path.resolve(strict=True)),
        execution_record_path=str(record_path.resolve(strict=True)),
        execution_artifact_sha256=execution_sha256,
        expected_text_column=str(expected_text_column),
        _configuration=copy.deepcopy(dict(configuration)),
        _identity=copy.deepcopy(dict(identity)),
        _evidence_payload=copy.deepcopy(dict(payload)),
        _evidence_item_count=int(count),
        _native_policy=copy.deepcopy(dict(policy)),
        _authority=_EMISSION_AUTHORITY,
    )


def emit_cumulative_spent_tfidf_capture(
    *,
    requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    replay_canary: CumulativeSpentReplayCanary,
    config: AppliedInferenceConfig,
    artifact_dir: Path | str,
    execution_record_dir: Path | str,
) -> Mapping[str, CumulativeSpentRemainingFamilyEmission]:
    """Perform one new nested TF-IDF fit and issue its two family views."""

    if not isinstance(replay_canary, CumulativeSpentReplayCanary):
        raise TypeError("TF-IDF emission requires a cumulative replay canary")
    if not isinstance(config, AppliedInferenceConfig):
        raise TypeError("TF-IDF emission requires AppliedInferenceConfig")
    typed = _paired_tfidf_requests(requests, replay_canary)
    request = typed[TFIDF_TOPICS]
    root = Path(artifact_dir)
    record_root = Path(execution_record_dir)
    if root.exists() or root.is_symlink() or record_root.exists() or record_root.is_symlink():
        raise RuntimeError("cumulative TF-IDF artifact and execution directories must be new")
    topic_config = config.architecture.multi_model_forest.tfidf_topic
    if (
        config.text_column in {config.treatment_column, config.outcome_column, "_oci_row_id"}
        or str(topic_config.score_selection_label_policy) != "nested_fit_calibration"
        or not bool(topic_config.score_test_enabled)
        or not bool(topic_config.orphan_ngram_enabled)
        or int(config.architecture.multi_model_forest.tfidf_nested_calibration_folds) < 2
    ):
        raise ValueError("cumulative TF-IDF requires nested training-only topic/orphan selection")
    treatment, outcome = _canonical_labels(request)
    fit = pd.DataFrame(
        {
            "_oci_row_id": list(request.spent_row_ids),
            config.text_column: [row.text for row in request.spent_rows],
            config.treatment_column: treatment,
            config.outcome_column: outcome,
        },
        columns=[
            "_oci_row_id",
            config.text_column,
            config.treatment_column,
            config.outcome_column,
        ],
    )
    heldout = replay_canary.transform_frame(text_column=config.text_column)
    root.mkdir(parents=True, exist_ok=False)
    _fit_tfidf_topic_context_nested_calibration(
        spec={
            "outer_fold": request.outer_fold,
            "inner_fold": request.provider_inner_fold,
            "scope_id": request.scope_id,
            "fit_df": fit,
            "heldout_df": heldout,
        },
        config=config,
        artifact_dir=root,
    )
    metadata, policy, model_path, source_path = _validate_tfidf_context(
        artifact_dir=root,
        request=request,
        replay_canary=replay_canary,
        expected_text_column=config.text_column,
    )
    if metadata.get("registered_heldout_columns_read") != ["_oci_row_id", config.text_column]:
        raise ValueError("TF-IDF cumulative replay read more than alias ID/text")
    score, _score_sha = _read_stable_json(
        source_path,
        field_name="cumulative TF-IDF score-selection artifact",
    )
    projection = _tfidf_catalog_projection(metadata, score)
    payloads, counts = _catalog_payloads(
        request=request,
        source_kind=TFIDF_TOPIC_SOURCE,
        payload={"discovery": projection},
        families=tuple(sorted(TFIDF_CUMULATIVE_FAMILIES)),
        heldout_row_ids=request.sealed_row_ids,
    )
    configuration = _tfidf_configuration(config, metadata)
    record_root.mkdir(parents=True, exist_ok=False)
    emissions: dict[str, CumulativeSpentRemainingFamilyEmission] = {}
    for family in sorted(TFIDF_CUMULATIVE_FAMILIES):
        identity = cumulative_spent_remaining_family_identity(
            family=family,
            configuration=configuration,
        )
        emissions[family] = _issue_emission(
            request=typed[family],
            replay_canary=replay_canary,
            native_kind="nested_tfidf",
            configuration=configuration,
            identity=identity,
            native_metadata_path=root / "context_metadata.json",
            model_path=model_path,
            source_path=source_path,
            policy=policy,
            payload=payloads[family],
            count=counts[family],
            record_path=record_root / f"{family}.json",
            expected_text_column=config.text_column,
        )
    return emissions


def emit_cumulative_spent_neural_query_capture(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    replay_canary: CumulativeSpentReplayCanary,
    service: ContextFitNeuralQueryService,
    artifact_dir: Path | str,
    execution_record_dir: Path | str,
) -> CumulativeSpentRemainingFamilyEmission:
    """Fit one new live neural-query context and snapshot its owned state."""

    if not isinstance(request, CumulativeSpentStage1FamilyRequest) or (
        request.family != NEURAL_QUERY_MOMENTS
    ):
        raise TypeError("neural-query emission requires its typed cumulative request")
    if not isinstance(replay_canary, CumulativeSpentReplayCanary):
        raise TypeError("neural-query emission requires a cumulative replay canary")
    replay_canary.assert_matches(request)
    if type(service) is not ContextFitNeuralQueryService:
        raise TypeError("neural-query emission requires the exact live context service")
    service_identity = service.identity()
    if any(
        row_id >= int(service_identity["dataset_row_count"]) for row_id in request.sealed_row_ids
    ):
        raise ValueError("sealed neural-query row ID escapes the cohort service")
    root = Path(artifact_dir)
    record_root = Path(execution_record_dir)
    if root.exists() or root.is_symlink() or record_root.exists() or record_root.is_symlink():
        raise RuntimeError("neural-query artifact and execution directories must be new")
    rows = request.spent_row_ids
    texts = tuple(row.text for row in request.spent_rows)
    treatment, outcome = _canonical_labels(request)
    bound_rows, bound_texts, provider = service._bind_rows_and_texts(
        rows,
        texts,
        row_name="cumulative_spent_row_ids",
        text_name="cumulative_spent_texts",
    )
    binding = service._binding(
        outer_fold=request.outer_fold,
        row_ids=bound_rows,
        texts=bound_texts,
        treatment=treatment,
        outcome=outcome,
        embedding_provider=provider,
    )
    cache_key = _sha256_json(binding)
    cache_root = Path(service.cache_dir) / cache_key
    if (
        cache_key in getattr(service, "_owned_discoveries", {})
        or cache_key in getattr(service, "_owned_discovery_bindings", {})
        or cache_root.exists()
    ):
        raise ValueError("neural-query cumulative emission requires a genuinely new live fit")
    discovery, observed_key = service.discovery_for_context(
        outer_fold=request.outer_fold,
        context_row_ids=rows,
        context_texts=texts,
        context_treatment=treatment,
        context_outcome=outcome,
    )
    if observed_key != cache_key:
        raise RuntimeError("neural-query service changed its canonical fit binding")
    safe_evidence = service.safe_evidence(
        discovery=discovery,
        context_row_ids=rows,
        context_texts=texts,
        device_offset=request.context_epoch,
    )
    root.mkdir(parents=True, exist_ok=False)
    snapshot = service.write_owned_discovery_snapshot(
        cache_key=cache_key,
        output_dir=root / "owned_snapshot",
    )
    snapshot = validate_owned_discovery_snapshot(
        root / "owned_snapshot",
        expected_cache_key=cache_key,
        expected_binding=binding,
        expected_service_identity_sha256=_sha256_json(service_identity),
    )
    policy = _query_policy(snapshot)
    source = {
        "schema_version": _QUERY_SOURCE_SCHEMA,
        "source_family": NEURAL_QUERY_MOMENTS,
        "scope_id": request.scope_id,
        "outer_fold": request.outer_fold,
        "context_epoch": request.context_epoch,
        "provider_inner_fold": request.provider_inner_fold,
        "request_binding_sha256": request.binding_sha256,
        "spent_row_order_fingerprint": row_order_fingerprint(request.spent_row_ids),
        "sealed_row_order_fingerprint": row_order_fingerprint(request.sealed_row_ids),
        "query_cache_key": cache_key,
        "owned_snapshot_content_sha256": snapshot["content_sha256"],
        "query_evidence": copy.deepcopy(safe_evidence),
        "all_queries_retained": True,
        "validation_audits_used_for_selection": False,
        "statistical_gate_applied": False,
        "sealed_text_accessed": False,
        "sealed_labels_accessed": False,
        "row_level_excerpts_emitted": False,
    }
    if set(source) != set(_QUERY_SOURCE_FIELDS):
        raise RuntimeError("neural-query source is not closed")
    source_path = root / "safe_evidence.json"
    _write_immutable_json(source_path, source)
    _snapshot, validated_policy, payload, count = _validate_query_artifacts(
        model_path=root / "owned_snapshot",
        source_path=source_path,
        request=request,
    )
    if validated_policy != policy:
        raise RuntimeError("neural-query policy changed after snapshot")
    configuration = _query_configuration(
        service_identity=service_identity,
        query_cache_key=cache_key,
    )
    identity = cumulative_spent_remaining_family_identity(
        family=NEURAL_QUERY_MOMENTS,
        configuration=configuration,
    )
    record_root.mkdir(parents=True, exist_ok=False)
    return _issue_emission(
        request=request,
        replay_canary=replay_canary,
        native_kind="owned_neural_query",
        configuration=configuration,
        identity=identity,
        native_metadata_path=root / "owned_snapshot" / "metadata.json",
        model_path=root / "owned_snapshot",
        source_path=source_path,
        policy=policy,
        payload=payload,
        count=count,
        record_path=record_root / f"{NEURAL_QUERY_MOMENTS}.json",
        expected_text_column=str(service_identity["text_column"]),
    )


def _exact_family_mapping(
    value: Mapping[str, Any],
    *,
    families: set[str],
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != families:
        raise ValueError(f"persisted {field_name} must cover exactly {sorted(families)}")
    return value


def _reload_summary(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    native_kind: str,
    native_metadata_sha256: str,
    identity: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    evidence_item_count: int,
    execution_artifact_sha256: str,
    fit_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CUMULATIVE_SPENT_REMAINING_RELOAD_VALIDATION_SCHEMA,
        "family": request.family,
        "native_kind": native_kind,
        "request_binding_sha256": request.binding_sha256,
        "native_metadata_sha256": _require_sha256(
            native_metadata_sha256,
            field_name="native_metadata_sha256",
        ),
        "producer_identity": copy.deepcopy(dict(identity)),
        "evidence_payload_sha256": _sha256_json(copy.deepcopy(dict(evidence_payload))),
        "evidence_item_count": int(evidence_item_count),
        "execution_artifact_sha256": _require_sha256(
            execution_artifact_sha256,
            field_name="execution_artifact_sha256",
        ),
        "fit_audit": copy.deepcopy(dict(fit_audit)),
    }


def _validated_tfidf_reload_state(
    *,
    requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    replay_canary: CumulativeSpentReplayCanary,
    config: AppliedInferenceConfig,
    producer_identity_by_family: Mapping[str, Mapping[str, Any]],
    evidence_payload_by_family: Mapping[str, Mapping[str, Any]],
    evidence_item_count_by_family: Mapping[str, int],
    artifact_dir: Path | str,
    execution_record_path_by_family: Mapping[str, Path | str],
    expected_fit_audit_by_family: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(replay_canary, CumulativeSpentReplayCanary):
        raise TypeError("TF-IDF reload requires a cumulative replay canary")
    typed = _paired_tfidf_requests(requests, replay_canary)
    required = set(TFIDF_CUMULATIVE_FAMILIES)
    for field_name, value in (
        ("TF-IDF producer identities", producer_identity_by_family),
        ("TF-IDF evidence payloads", evidence_payload_by_family),
        ("TF-IDF evidence counts", evidence_item_count_by_family),
        ("TF-IDF execution records", execution_record_path_by_family),
    ):
        _exact_family_mapping(value, families=required, field_name=field_name)
    if expected_fit_audit_by_family is not None:
        _exact_family_mapping(
            expected_fit_audit_by_family,
            families=required,
            field_name="TF-IDF fit audits",
        )
    root_path = Path(artifact_dir)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ValueError("persisted TF-IDF artifact root must be one real directory")
    root = root_path.resolve(strict=True)
    reference = typed[TFIDF_TOPICS]
    metadata, policy, model_path, source_path = _validate_tfidf_context(
        artifact_dir=root,
        request=reference,
        replay_canary=replay_canary,
        expected_text_column=config.text_column,
    )
    configuration = _validated_tfidf_configuration(config, metadata)
    score, _score_sha256 = _read_stable_json(
        source_path,
        field_name="cumulative TF-IDF score-selection artifact",
    )
    projection = _tfidf_catalog_projection(metadata, score)
    payloads, counts = _catalog_payloads(
        request=reference,
        source_kind=TFIDF_TOPIC_SOURCE,
        payload={"discovery": projection},
        families=tuple(sorted(TFIDF_CUMULATIVE_FAMILIES)),
        heldout_row_ids=reference.sealed_row_ids,
    )
    metadata_path = root / "context_metadata.json"
    metadata_sha256 = native_artifact_sha256(metadata_path)
    output: dict[str, dict[str, Any]] = {}
    for family in sorted(required):
        request = typed[family]
        supplied_payload = evidence_payload_by_family[family]
        supplied_count = evidence_item_count_by_family[family]
        if not isinstance(supplied_payload, Mapping):
            raise TypeError(f"persisted TF-IDF payload for {family} must be a mapping")
        if (
            isinstance(supplied_count, bool)
            or not isinstance(supplied_count, int)
            or supplied_count < 1
            or copy.deepcopy(dict(supplied_payload)) != payloads[family]
            or int(supplied_count) != counts[family]
        ):
            raise ValueError(f"persisted TF-IDF payload/count differs for {family}")
        expected_identity = cumulative_spent_remaining_family_identity(
            family=family,
            configuration=configuration,
        )
        supplied_identity = _validate_identity(
            producer_identity_by_family[family],
            family=family,
        )
        if supplied_identity != expected_identity:
            raise ValueError(f"persisted TF-IDF identity differs from expected config for {family}")
        expected_record = _execution_record(
            request=request,
            replay_canary=replay_canary,
            native_kind="nested_tfidf",
            identity=supplied_identity,
            native_metadata_sha256=metadata_sha256,
            native_policy=policy,
            model_path=model_path,
            source_path=source_path,
            evidence_payload=payloads[family],
            evidence_item_count=counts[family],
        )
        persisted, execution_sha256 = _read_stable_json(
            execution_record_path_by_family[family],
            field_name=f"cumulative TF-IDF {family} execution record",
        )
        _validate_record(persisted, family=family, native_kind="nested_tfidf")
        if persisted != expected_record:
            raise ValueError(
                f"persisted TF-IDF execution record differs from canonical replay for {family}"
            )
        fit_audit = _fit_audit_from_record(
            request=request,
            record=persisted,
            execution_artifact_sha256=execution_sha256,
            native_policy=policy,
        )
        if expected_fit_audit_by_family is not None:
            expected_audit = expected_fit_audit_by_family[family]
            if (
                not isinstance(expected_audit, Mapping)
                or copy.deepcopy(dict(expected_audit)) != fit_audit
            ):
                raise ValueError(f"persisted TF-IDF fit audit differs for {family}")
        output[family] = {
            "request": request,
            "configuration": copy.deepcopy(configuration),
            "identity": supplied_identity,
            "payload": payloads[family],
            "count": counts[family],
            "policy": copy.deepcopy(policy),
            "metadata_path": metadata_path,
            "model_path": model_path,
            "source_path": source_path,
            "execution_record_path": Path(execution_record_path_by_family[family]).resolve(
                strict=True
            ),
            "execution_sha256": execution_sha256,
            "fit_audit": fit_audit,
            "summary": _reload_summary(
                request=request,
                native_kind="nested_tfidf",
                native_metadata_sha256=metadata_sha256,
                identity=supplied_identity,
                evidence_payload=payloads[family],
                evidence_item_count=counts[family],
                execution_artifact_sha256=execution_sha256,
                fit_audit=fit_audit,
            ),
        }
    return output


def validate_cumulative_spent_tfidf_artifacts(
    *,
    requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    replay_canary: CumulativeSpentReplayCanary,
    config: AppliedInferenceConfig,
    producer_identity_by_family: Mapping[str, Mapping[str, Any]],
    evidence_payload_by_family: Mapping[str, Mapping[str, Any]],
    evidence_item_count_by_family: Mapping[str, int],
    artifact_dir: Path | str,
    execution_record_path_by_family: Mapping[str, Path | str],
    expected_fit_audit_by_family: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Independently reload the paired persisted topic/orphan component."""

    state = _validated_tfidf_reload_state(
        requests=requests,
        replay_canary=replay_canary,
        config=config,
        producer_identity_by_family=producer_identity_by_family,
        evidence_payload_by_family=evidence_payload_by_family,
        evidence_item_count_by_family=evidence_item_count_by_family,
        artifact_dir=artifact_dir,
        execution_record_path_by_family=execution_record_path_by_family,
        expected_fit_audit_by_family=expected_fit_audit_by_family,
    )
    return {family: copy.deepcopy(value["summary"]) for family, value in state.items()}


def _validated_neural_query_reload_state(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    replay_canary: CumulativeSpentReplayCanary,
    expected_service_identity: Mapping[str, Any],
    producer_identity: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    evidence_item_count: int,
    model_artifact_path: Path | str,
    source_artifact_path: Path | str,
    execution_record_path: Path | str,
    expected_fit_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(request, CumulativeSpentStage1FamilyRequest) or (
        request.family != NEURAL_QUERY_MOMENTS
    ):
        raise TypeError("neural-query reload requires its typed cumulative request")
    if not isinstance(replay_canary, CumulativeSpentReplayCanary):
        raise TypeError("neural-query reload requires a cumulative replay canary")
    replay_canary.assert_matches(request)
    service_identity = _validated_service_identity(expected_service_identity)
    row_count = int(service_identity["dataset_row_count"])
    if any(row_id >= row_count for row_id in (*request.spent_row_ids, *request.sealed_row_ids)):
        raise ValueError("neural-query request escapes the expected cohort service")
    model_path = Path(model_artifact_path)
    source_path = Path(source_artifact_path)
    if model_path.is_symlink() or not model_path.is_dir():
        raise ValueError("persisted neural-query model artifact must be one real directory")
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("persisted neural-query source artifact must be one regular file")
    model_path = model_path.resolve(strict=True)
    source_path = source_path.resolve(strict=True)
    if source_path.parent != model_path.parent:
        raise ValueError("persisted neural-query model and source do not share one component root")
    snapshot, policy, regenerated_payload, regenerated_count = _validate_query_artifacts(
        model_path=model_path,
        source_path=source_path,
        request=request,
        expected_service_identity=service_identity,
    )
    if not isinstance(evidence_payload, Mapping):
        raise TypeError("persisted neural-query payload must be a mapping")
    if (
        isinstance(evidence_item_count, bool)
        or not isinstance(evidence_item_count, int)
        or evidence_item_count < 1
        or copy.deepcopy(dict(evidence_payload)) != regenerated_payload
        or int(evidence_item_count) != regenerated_count
    ):
        raise ValueError("persisted neural-query payload/count differs from native source")
    configuration = _query_configuration(
        service_identity=service_identity,
        query_cache_key=str(snapshot.get("cache_key") or ""),
    )
    expected_identity = cumulative_spent_remaining_family_identity(
        family=NEURAL_QUERY_MOMENTS,
        configuration=configuration,
    )
    supplied_identity = _validate_identity(
        producer_identity,
        family=NEURAL_QUERY_MOMENTS,
    )
    if supplied_identity != expected_identity:
        raise ValueError("persisted neural-query identity differs from expected service config")
    metadata_path = model_path / "metadata.json"
    metadata_sha256 = native_artifact_sha256(metadata_path)
    expected_record = _execution_record(
        request=request,
        replay_canary=replay_canary,
        native_kind="owned_neural_query",
        identity=supplied_identity,
        native_metadata_sha256=metadata_sha256,
        native_policy=policy,
        model_path=model_path,
        source_path=source_path,
        evidence_payload=regenerated_payload,
        evidence_item_count=regenerated_count,
    )
    persisted, execution_sha256 = _read_stable_json(
        execution_record_path,
        field_name="cumulative neural-query execution record",
    )
    _validate_record(
        persisted,
        family=NEURAL_QUERY_MOMENTS,
        native_kind="owned_neural_query",
    )
    if persisted != expected_record:
        raise ValueError("persisted neural-query execution record differs from canonical replay")
    fit_audit = _fit_audit_from_record(
        request=request,
        record=persisted,
        execution_artifact_sha256=execution_sha256,
        native_policy=policy,
    )
    if expected_fit_audit is not None and (
        not isinstance(expected_fit_audit, Mapping)
        or copy.deepcopy(dict(expected_fit_audit)) != fit_audit
    ):
        raise ValueError("persisted neural-query fit audit differs from canonical replay")
    return {
        "request": request,
        "configuration": configuration,
        "identity": supplied_identity,
        "payload": regenerated_payload,
        "count": regenerated_count,
        "policy": policy,
        "metadata_path": metadata_path,
        "model_path": model_path,
        "source_path": source_path,
        "execution_record_path": Path(execution_record_path).resolve(strict=True),
        "execution_sha256": execution_sha256,
        "fit_audit": fit_audit,
        "summary": _reload_summary(
            request=request,
            native_kind="owned_neural_query",
            native_metadata_sha256=metadata_sha256,
            identity=supplied_identity,
            evidence_payload=regenerated_payload,
            evidence_item_count=regenerated_count,
            execution_artifact_sha256=execution_sha256,
            fit_audit=fit_audit,
        ),
    }


def validate_cumulative_spent_neural_query_artifact(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    replay_canary: CumulativeSpentReplayCanary,
    expected_service_identity: Mapping[str, Any],
    producer_identity: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    evidence_item_count: int,
    model_artifact_path: Path | str,
    source_artifact_path: Path | str,
    execution_record_path: Path | str,
    expected_fit_audit: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Independently reload one persisted, non-executable neural-query snapshot."""

    state = _validated_neural_query_reload_state(
        request=request,
        replay_canary=replay_canary,
        expected_service_identity=expected_service_identity,
        producer_identity=producer_identity,
        evidence_payload=evidence_payload,
        evidence_item_count=evidence_item_count,
        model_artifact_path=model_artifact_path,
        source_artifact_path=source_artifact_path,
        execution_record_path=execution_record_path,
        expected_fit_audit=expected_fit_audit,
    )
    return copy.deepcopy(state["summary"])


@dataclass(frozen=True)
class NativeCumulativeSpentRemainingFamilyProducer:
    family: str
    native_kind: str
    _request_binding_sha256: str = field(repr=False)
    _expected_configuration: Mapping[str, Any] = field(repr=False)
    _identity: Mapping[str, Any] = field(repr=False)
    _evidence_payload: Mapping[str, Any] = field(repr=False)
    _evidence_item_count: int = field(repr=False)
    _native_policy: Mapping[str, Any] = field(repr=False)
    _replay_canary: CumulativeSpentReplayCanary = field(repr=False)
    _native_metadata_path: str = field(repr=False)
    _model_artifact_path: str = field(repr=False)
    _source_artifact_path: str = field(repr=False)
    _execution_record_path: str = field(repr=False)
    _execution_artifact_sha256: str = field(repr=False)
    _expected_text_column: str = field(repr=False)

    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(_validate_identity(self._identity, family=self.family))

    def _revalidate(
        self,
        request: CumulativeSpentStage1FamilyRequest,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if request.family != self.family or request.binding_sha256 != self._request_binding_sha256:
            raise ValueError("remaining-family producer was invoked for another request")
        self._replay_canary.assert_matches(request)
        expected_identity = cumulative_spent_remaining_family_identity(
            family=self.family,
            configuration=self._expected_configuration,
        )
        if _validate_identity(self._identity, family=self.family) != expected_identity:
            raise RuntimeError("remaining-family producer identity differs from expected config")
        model_path = Path(self._model_artifact_path)
        source_path = Path(self._source_artifact_path)
        metadata_path = Path(self._native_metadata_path)
        if self.native_kind == "nested_tfidf":
            metadata, policy, observed_model, observed_source = _validate_tfidf_context(
                artifact_dir=metadata_path.parent,
                request=request,
                replay_canary=self._replay_canary,
                expected_text_column=self._expected_text_column,
            )
            if observed_model != model_path or observed_source != source_path:
                raise RuntimeError("TF-IDF emission artifact addresses changed")
            if self._expected_configuration.get(
                "text_column"
            ) != self._expected_text_column or metadata.get(
                "config_hash"
            ) != self._expected_configuration.get(
                "topic_configuration_hash"
            ):
                raise RuntimeError("TF-IDF native artifact differs from expected config")
            score, _score_sha = _read_stable_json(
                source_path,
                field_name="cumulative TF-IDF score-selection artifact",
            )
            projection = _tfidf_catalog_projection(metadata, score)
            payloads, counts = _catalog_payloads(
                request=request,
                source_kind=TFIDF_TOPIC_SOURCE,
                payload={"discovery": projection},
                families=(self.family,),
                heldout_row_ids=request.sealed_row_ids,
            )
            payload = payloads[self.family]
            count = counts[self.family]
            native_metadata_sha256 = native_artifact_sha256(metadata_path)
        elif self.native_kind == "owned_neural_query":
            service_identity = self._expected_configuration.get("service_identity")
            if not isinstance(service_identity, Mapping):
                raise RuntimeError("neural-query producer lost its expected service identity")
            snapshot, policy, payload, count = _validate_query_artifacts(
                model_path=model_path,
                source_path=source_path,
                request=request,
                expected_service_identity=service_identity,
            )
            observed_configuration = _query_configuration(
                service_identity=service_identity,
                query_cache_key=str(snapshot.get("cache_key") or ""),
            )
            if observed_configuration != dict(self._expected_configuration):
                raise RuntimeError("neural-query native artifact differs from expected config")
            native_metadata_sha256 = native_artifact_sha256(metadata_path)
            if snapshot.get("content_sha256") is None:
                raise RuntimeError("neural-query snapshot lost its content binding")
        else:  # pragma: no cover - constructor prevents this
            raise RuntimeError("unknown remaining-family native kind")
        if (
            policy != dict(self._native_policy)
            or payload != dict(self._evidence_payload)
            or count != self._evidence_item_count
        ):
            raise RuntimeError("remaining-family native payload or policy changed")
        expected = _execution_record(
            request=request,
            replay_canary=self._replay_canary,
            native_kind=self.native_kind,
            identity=self._identity,
            native_metadata_sha256=native_metadata_sha256,
            native_policy=policy,
            model_path=model_path,
            source_path=source_path,
            evidence_payload=payload,
            evidence_item_count=count,
        )
        persisted, file_sha256 = _read_stable_json(
            self._execution_record_path,
            field_name="cumulative remaining-family execution record",
        )
        _validate_record(persisted, family=self.family, native_kind=self.native_kind)
        if persisted != expected:
            raise RuntimeError("component-emitted remaining-family execution record changed")
        if file_sha256 != self._execution_artifact_sha256:
            raise RuntimeError("remaining-family execution artifact bytes changed")
        return persisted, policy

    def produce_cumulative_spent(
        self,
        request: CumulativeSpentStage1FamilyRequest,
    ) -> CumulativeSpentFamilyEvidenceDraft:
        record, policy = self._revalidate(request)
        audit = _fit_audit_from_record(
            request=request,
            record=record,
            execution_artifact_sha256=self._execution_artifact_sha256,
            native_policy=policy,
        )
        return CumulativeSpentFamilyEvidenceDraft(
            evidence_payload=copy.deepcopy(dict(self._evidence_payload)),
            evidence_item_count=self._evidence_item_count,
            input_binding_sha256=request.binding_sha256,
            fit_semantics=CUMULATIVE_SPENT_REFIT,
            fit_audit=audit,
        )


def bind_persisted_cumulative_spent_tfidf_producers(
    *,
    requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    replay_canary: CumulativeSpentReplayCanary,
    config: AppliedInferenceConfig,
    producer_identity_by_family: Mapping[str, Mapping[str, Any]],
    evidence_payload_by_family: Mapping[str, Mapping[str, Any]],
    evidence_item_count_by_family: Mapping[str, int],
    artifact_dir: Path | str,
    execution_record_path_by_family: Mapping[str, Path | str],
    expected_fit_audit_by_family: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, NativeCumulativeSpentRemainingFamilyProducer]:
    """Bind paired persisted TF-IDF artifacts into revalidating producers."""

    state = _validated_tfidf_reload_state(
        requests=requests,
        replay_canary=replay_canary,
        config=config,
        producer_identity_by_family=producer_identity_by_family,
        evidence_payload_by_family=evidence_payload_by_family,
        evidence_item_count_by_family=evidence_item_count_by_family,
        artifact_dir=artifact_dir,
        execution_record_path_by_family=execution_record_path_by_family,
        expected_fit_audit_by_family=expected_fit_audit_by_family,
    )
    output: dict[str, NativeCumulativeSpentRemainingFamilyProducer] = {}
    for family, value in state.items():
        producer = NativeCumulativeSpentRemainingFamilyProducer(
            family=family,
            native_kind="nested_tfidf",
            _request_binding_sha256=value["request"].binding_sha256,
            _expected_configuration=copy.deepcopy(value["configuration"]),
            _identity=copy.deepcopy(value["identity"]),
            _evidence_payload=copy.deepcopy(value["payload"]),
            _evidence_item_count=int(value["count"]),
            _native_policy=copy.deepcopy(value["policy"]),
            _replay_canary=replay_canary,
            _native_metadata_path=str(value["metadata_path"]),
            _model_artifact_path=str(value["model_path"]),
            _source_artifact_path=str(value["source_path"]),
            _execution_record_path=str(value["execution_record_path"]),
            _execution_artifact_sha256=str(value["execution_sha256"]),
            _expected_text_column=str(config.text_column),
        )
        producer._revalidate(value["request"])
        output[family] = producer
    return output


def bind_persisted_cumulative_spent_neural_query_producer(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    replay_canary: CumulativeSpentReplayCanary,
    expected_service_identity: Mapping[str, Any],
    producer_identity: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    evidence_item_count: int,
    model_artifact_path: Path | str,
    source_artifact_path: Path | str,
    execution_record_path: Path | str,
    expected_fit_audit: Mapping[str, Any] | None = None,
) -> NativeCumulativeSpentRemainingFamilyProducer:
    """Bind a persisted safe neural-query snapshot into a revalidating producer."""

    state = _validated_neural_query_reload_state(
        request=request,
        replay_canary=replay_canary,
        expected_service_identity=expected_service_identity,
        producer_identity=producer_identity,
        evidence_payload=evidence_payload,
        evidence_item_count=evidence_item_count,
        model_artifact_path=model_artifact_path,
        source_artifact_path=source_artifact_path,
        execution_record_path=execution_record_path,
        expected_fit_audit=expected_fit_audit,
    )
    producer = NativeCumulativeSpentRemainingFamilyProducer(
        family=NEURAL_QUERY_MOMENTS,
        native_kind="owned_neural_query",
        _request_binding_sha256=request.binding_sha256,
        _expected_configuration=copy.deepcopy(state["configuration"]),
        _identity=copy.deepcopy(state["identity"]),
        _evidence_payload=copy.deepcopy(state["payload"]),
        _evidence_item_count=int(state["count"]),
        _native_policy=copy.deepcopy(state["policy"]),
        _replay_canary=replay_canary,
        _native_metadata_path=str(state["metadata_path"]),
        _model_artifact_path=str(state["model_path"]),
        _source_artifact_path=str(state["source_path"]),
        _execution_record_path=str(state["execution_record_path"]),
        _execution_artifact_sha256=str(state["execution_sha256"]),
        _expected_text_column=str(state["configuration"]["text_column"]),
    )
    producer._revalidate(request)
    return producer


def bind_cumulative_spent_remaining_family_producer(
    *,
    request: CumulativeSpentStage1FamilyRequest,
    replay_canary: CumulativeSpentReplayCanary,
    emission: CumulativeSpentRemainingFamilyEmission,
) -> NativeCumulativeSpentRemainingFamilyProducer:
    if type(emission) is not CumulativeSpentRemainingFamilyEmission or (
        emission._authority is not _EMISSION_AUTHORITY
    ):
        raise TypeError("binding requires an issuer-authenticated same-process emission")
    if not isinstance(request, CumulativeSpentStage1FamilyRequest):
        raise TypeError("binding requires a typed cumulative request")
    if not isinstance(replay_canary, CumulativeSpentReplayCanary):
        raise TypeError("binding requires a cumulative replay canary")
    replay_canary.assert_matches(request)
    if (
        request.family != emission.family
        or request.binding_sha256 != emission.request_binding_sha256
    ):
        raise ValueError("remaining-family emission belongs to another request")
    producer = NativeCumulativeSpentRemainingFamilyProducer(
        family=request.family,
        native_kind=emission.native_kind,
        _request_binding_sha256=request.binding_sha256,
        _expected_configuration=copy.deepcopy(dict(emission._configuration)),
        _identity=copy.deepcopy(dict(emission._identity)),
        _evidence_payload=copy.deepcopy(dict(emission._evidence_payload)),
        _evidence_item_count=emission._evidence_item_count,
        _native_policy=copy.deepcopy(dict(emission._native_policy)),
        _replay_canary=replay_canary,
        _native_metadata_path=emission.native_metadata_path,
        _model_artifact_path=emission.model_artifact_path,
        _source_artifact_path=emission.source_artifact_path,
        _execution_record_path=emission.execution_record_path,
        _execution_artifact_sha256=emission.execution_artifact_sha256,
        _expected_text_column=emission.expected_text_column,
    )
    producer._revalidate(request)
    return producer


__all__ = [
    "CUMULATIVE_SPENT_NEURAL_QUERY_POLICY_SCHEMA",
    "CUMULATIVE_SPENT_REMAINING_ADAPTER_VERSION",
    "CUMULATIVE_SPENT_REMAINING_EMISSION_SCHEMA",
    "CUMULATIVE_SPENT_REMAINING_EXECUTION_RECORD_SCHEMA",
    "CUMULATIVE_SPENT_REMAINING_RELOAD_VALIDATION_SCHEMA",
    "CumulativeSpentRemainingFamilyEmission",
    "NativeCumulativeSpentRemainingFamilyProducer",
    "REMAINING_CUMULATIVE_FAMILIES",
    "TFIDF_CUMULATIVE_FAMILIES",
    "bind_cumulative_spent_remaining_family_producer",
    "bind_persisted_cumulative_spent_neural_query_producer",
    "bind_persisted_cumulative_spent_tfidf_producers",
    "cumulative_spent_remaining_family_identity",
    "emit_cumulative_spent_neural_query_capture",
    "emit_cumulative_spent_tfidf_capture",
    "validate_cumulative_spent_neural_query_artifact",
    "validate_cumulative_spent_tfidf_artifacts",
]
