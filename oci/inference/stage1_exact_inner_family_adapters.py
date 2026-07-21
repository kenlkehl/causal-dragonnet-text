"""Authenticated adapters from native Stage 1 fits to the exact-inner contract.

The ten discovery families are already implemented by four native modeling
surfaces rather than ten independent trainer classes:

* ``MultiModelForestStage1Runner`` emits the BoW, HTR, matched-pair, and
  embedding families;
* frozen-retrieval TF-IDF contrasts are a semantic projection of the embedding
  fit, not a second embedding fit;
* ``fit_tfidf_topic_context`` emits the topic and orphan-n-gram families; and
* ``ContextFitNeuralQueryService`` emits learned query witnesses and moments.

This module does not relabel synthetic evidence as a fit.  It adapts only a
closed, authenticated native scope result whose exact ordered rows, data
projection, model/output artifact, and execution proof have already been
verified.  A native component loaded from a sealed artifact is reported as an
exact-scope cache replay; an adapter around a just-completed fit is reported as
an exact-inner refit.  Missing evidence or proof for any family fails closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)
from .all_evidence_fusion import FoldEvidenceProvenance
from .embedding_native_proof_capture import (
    EMBEDDING_NATIVE_CAPTURE_SCHEMA,
    SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA,
)
from .lossless_stage1_evidence_catalog import (
    SEMANTIC_RETRIEVAL_DERIVATION,
    RoleNeutralEvidenceCatalog,
    validate_role_neutral_catalog,
)
from .stage1_exact_inner_evidence import (
    EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
    EXACT_INNER_FIT_AUDIT_VERSION,
    EXACT_INNER_REFIT,
    EXACT_SCOPE_CACHE_REPLAY,
    ExactInnerFamilyEvidenceDraft,
    ExactInnerStage1FamilyRequest,
    row_order_fingerprint,
)
from .tfidf_topic_discovery import TOPIC_SCORE_TEST_SCHEMA_VERSION, row_set_fingerprint
from .tfidf_topic_stage1 import TFIDF_NESTED_CALIBRATION_SCHEMA_VERSION

NATIVE_EXACT_INNER_ADAPTER_VERSION = "native_exact_inner_stage1_family_adapter_v2"
NATIVE_SCOPE_RESULT_VERSION = "authenticated_native_exact_inner_scope_result_v1"
NATIVE_FAMILY_FIT_PROOF_VERSION = "authenticated_native_stage1_family_fit_proof_v1"
NATIVE_FAMILY_PAYLOAD_VERSION = "native_stage1_family_concept_evidence_v1"
NATIVE_FAMILY_EXECUTION_RECORD_VERSION = "native_stage1_family_execution_record_v1"
NATIVE_FULL_OUTER_PAYLOAD_REGISTRY_VERSION = "authenticated_native_full_outer_payload_registry_v1"

_NATIVE_PROOF_CONSTRUCTION_AUTHORITY = object()
_NATIVE_SCOPE_CONSTRUCTION_AUTHORITY = object()
_NATIVE_FULL_OUTER_REGISTRY_AUTHORITY = object()

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VALID_FIT_SEMANTICS = frozenset({EXACT_INNER_REFIT, EXACT_SCOPE_CACHE_REPLAY})
_SEMANTIC_RETRIEVAL_TRAINING_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "selection_kind",
        "nested_calibration_applicability",
        "seed",
        "fold_parameter",
        "configured_fold_count",
        "fold_count",
        "split_method",
        "model_fit_row_ids",
        "calibration_row_ids",
        "model_fit_row_order_fingerprint",
        "calibration_row_order_fingerprint",
        "partitions_are_replay_canaries_only",
        "partition_canaries_select_or_drop_terms",
        "authoritative_projection_scope",
        "projection_vocabulary_max_features",
        "projection_output_limit",
        "all_nonzero_sanitized_terms_preserved",
        "upstream_embedding_directions_and_retrieval_use_exact_fit_labels_only",
        "nested_calibration_labels_accessed",
        "registered_heldout_labels_accessed",
        "registered_heldout_text_accessed",
        "registered_heldout_transform_performed",
        "selection_frozen_before_registered_heldout_use",
        "projection_frozen_before_registered_heldout_use",
        "canonical_hierarchy_partition_count_used_as_calibration_folds",
        "interaction_inner_folds_used_as_calibration_folds",
    }
)

FAMILY_NATIVE_BACKEND: Mapping[str, str] = {
    BOW_NUISANCE: "multi_model_forest_bow_nuisance",
    BOW_R_LOSS: "multi_model_forest_bow_r_loss",
    HTR_NEURAL: "multi_model_forest_htr_nuisance_and_effect",
    MATCHED_PAIR_UPLIFT: "multi_model_forest_bow_and_htr_matched_pair",
    EMBEDDING_WHOLE_COHORT: "frozen_cache_embedding_whole_cohort_contrasts",
    EMBEDDING_CLUSTERED: "frozen_cache_embedding_cluster_local_contrasts",
    TFIDF_SEMANTIC_RETRIEVAL: "frozen_embedding_retrieval_tail_tfidf_contrasts",
    TFIDF_TOPICS: "tfidf_consensus_nmf_topic_context",
    TFIDF_ORPHAN_NGRAMS: "tfidf_fit_side_orphan_ngram_context",
    NEURAL_QUERY_MOMENTS: "context_fit_neural_query_service",
}

# These are the real, already-existing fit/derivation APIs used by the production
# scope runner.  The adapter identity names them explicitly so a catalog
# projection cannot be mistaken for an eleventh synthetic modeling backend.
FAMILY_NATIVE_APIS: Mapping[str, tuple[str, ...]] = {
    BOW_NUISANCE: (
        "oci.inference.multi_model_forest_stage1."
        "MultiModelForestStage1Runner._fit_bow_binary_train_test",
        "oci.inference.multi_model_forest_stage1."
        "MultiModelForestStage1Runner._fit_primary_feature_importance_models",
        "oci.inference.bow_native_proof_capture.NativeBoWProofCaptureSink",
        "oci.inference.bow_native_proof_capture.validate_bow_native_capture",
    ),
    BOW_R_LOSS: (
        "oci.inference.multi_model_forest_stage1."
        "MultiModelForestStage1Runner._fit_bow_regression_train_test",
        "oci.inference.multi_model_forest_stage1."
        "MultiModelForestStage1Runner._fit_primary_feature_importance_models",
        "oci.inference.bow_native_proof_capture.NativeBoWProofCaptureSink",
        "oci.inference.bow_native_proof_capture.validate_bow_native_capture",
    ),
    HTR_NEURAL: (
        "oci.inference.multi_model_forest_stage1."
        "MultiModelForestStage1HTRProvider.fit_nuisance_inner_ensemble_predict",
        "oci.inference.multi_model_forest_stage1."
        "MultiModelForestStage1HTRProvider.fit_effect_variant_inner_ensemble_predict",
        "oci.inference.htr_native_proof_capture.NativeHTRProofCaptureSink",
        "oci.inference.htr_native_proof_capture.validate_htr_native_capture",
    ),
    MATCHED_PAIR_UPLIFT: (
        "oci.inference.multi_model_pair_uplift.fit_bow_pair_uplift_train_test",
        "oci.inference.multi_model_pair_uplift.fit_htr_pair_uplift_train_test",
        "oci.inference.matched_pair_native_proof_capture." "NativeMatchedPairProofCaptureSink",
        "oci.inference.matched_pair_native_proof_capture." "validate_matched_pair_native_capture",
    ),
    EMBEDDING_WHOLE_COHORT: (
        "oci.inference.embedding_contrast_discovery."
        "EmbeddingContrastEvidenceGenerator.build_evidence",
        "oci.inference.embedding_native_proof_capture.NativeEmbeddingProofCaptureSink",
        "oci.inference.embedding_native_proof_capture.validate_embedding_native_capture",
    ),
    EMBEDDING_CLUSTERED: (
        "oci.inference.embedding_contrast_discovery."
        "EmbeddingContrastEvidenceGenerator._build_cluster_contrast_vectors",
        "oci.inference.embedding_native_proof_capture.NativeEmbeddingProofCaptureSink",
        "oci.inference.embedding_native_proof_capture.validate_embedding_native_capture",
    ),
    TFIDF_SEMANTIC_RETRIEVAL: (
        "oci.inference.embedding_contrast_discovery."
        "EmbeddingContrastEvidenceGenerator.build_evidence",
        "oci.inference.review_spent_evidence_provider._embedding_concepts_only",
        "oci.inference.embedding_native_proof_capture.NativeEmbeddingProofCaptureSink",
        "oci.inference.embedding_native_proof_capture.validate_embedding_native_capture",
        "oci.inference.lossless_stage1_evidence_catalog." "_CatalogBuilder._embedding",
    ),
    TFIDF_TOPICS: (
        "oci.inference.tfidf_topic_discovery.fit_tfidf_topic_context",
        "oci.inference.tfidf_topic_stage1." "_fit_tfidf_topic_context_nested_calibration",
        "oci.inference.tfidf_topic_stage1.run_tfidf_topic_stage1",
    ),
    TFIDF_ORPHAN_NGRAMS: (
        "oci.inference.tfidf_topic_discovery.fit_tfidf_topic_context",
        "oci.inference.tfidf_topic_stage1." "_fit_tfidf_topic_context_nested_calibration",
        "oci.inference.tfidf_topic_stage1.run_tfidf_topic_stage1",
    ),
    NEURAL_QUERY_MOMENTS: (
        "oci.inference.neural_query_context_backend."
        "ContextFitNeuralQueryService.discovery_for_context",
        "oci.inference.neural_query_context_backend."
        "ContextFitNeuralQueryService.write_owned_discovery_snapshot",
        "oci.inference.neural_query_context_backend." "ContextFitNeuralQueryService.safe_evidence",
        "oci.inference.neural_query_context_backend.NeuralQueryContextBackend.fit_predict",
    ),
}

_COMMON_CODE_PATHS = (
    "oci/config.py",
    "oci/inference/all_evidence_discovery_interfaces.py",
    "oci/inference/all_evidence_fusion.py",
    "oci/inference/stage1_exact_inner_evidence.py",
    "oci/inference/stage1_exact_inner_family_adapters.py",
    "oci/inference/production_stage1_bundle.py",
    "oci/inference/lossless_stage1_evidence_catalog.py",
)
_FAMILY_CODE_PATHS: Mapping[str, tuple[str, ...]] = {
    BOW_NUISANCE: (
        "oci/inference/bow_native_proof_capture.py",
        "oci/inference/multi_model_forest_stage1.py",
        "oci/inference/multi_model_agentic_forest.py",
    ),
    BOW_R_LOSS: (
        "oci/inference/bow_native_proof_capture.py",
        "oci/inference/multi_model_forest_stage1.py",
        "oci/inference/multi_model_agentic_forest.py",
    ),
    HTR_NEURAL: (
        "oci/inference/htr_native_proof_capture.py",
        "oci/inference/multi_model_forest_stage1.py",
        "oci/inference/agentic_attention_variable_forest.py",
    ),
    MATCHED_PAIR_UPLIFT: (
        "oci/inference/matched_pair_native_proof_capture.py",
        "oci/inference/multi_model_forest_stage1.py",
        "oci/inference/multi_model_pair_uplift.py",
    ),
    EMBEDDING_WHOLE_COHORT: (
        "oci/inference/embedding_native_proof_capture.py",
        "oci/inference/multi_model_forest_stage1.py",
        "oci/inference/embedding_contrast_discovery.py",
        "oci/inference/review_spent_evidence_provider.py",
    ),
    EMBEDDING_CLUSTERED: (
        "oci/inference/embedding_native_proof_capture.py",
        "oci/inference/multi_model_forest_stage1.py",
        "oci/inference/embedding_contrast_discovery.py",
        "oci/inference/review_spent_evidence_provider.py",
    ),
    TFIDF_SEMANTIC_RETRIEVAL: (
        "oci/inference/embedding_native_proof_capture.py",
        "oci/inference/embedding_contrast_discovery.py",
        "oci/inference/review_spent_evidence_provider.py",
    ),
    TFIDF_TOPICS: (
        "oci/inference/tfidf_topic_discovery.py",
        "oci/inference/tfidf_topic_stage1.py",
        "oci/inference/tfidf_topic_agentic_forest.py",
        "oci/inference/tfidf_topic_score_selection.py",
    ),
    TFIDF_ORPHAN_NGRAMS: (
        "oci/inference/tfidf_topic_discovery.py",
        "oci/inference/tfidf_topic_stage1.py",
        "oci/inference/tfidf_topic_agentic_forest.py",
        "oci/inference/tfidf_topic_score_selection.py",
        "oci/inference/tfidf_orphan_evidence_adapter.py",
    ),
    NEURAL_QUERY_MOMENTS: (
        "oci/inference/neural_query_context_backend.py",
        "oci/inference/neural_query_discovery_runtime.py",
        "oci/inference/neural_query_agentic_forest.py",
    ),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _stable_file_inventory_row(path: Path, *, relative_path: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"native identity path is not one regular file: {relative_path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if before_identity != after_identity:
        raise RuntimeError(f"native identity file changed while hashing: {relative_path}")
    return {
        "relative_path": relative_path,
        "size": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def native_family_code_identity(family: str) -> dict[str, Any]:
    """Content-address the adapter and the real native APIs for one family."""

    if family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError("family is not an active Stage 1 architecture")
    repository_root = Path(__file__).resolve().parents[2]
    relative_paths = tuple(dict.fromkeys((*_COMMON_CODE_PATHS, *_FAMILY_CODE_PATHS[family])))
    files = [
        _stable_file_inventory_row(repository_root / relative_path, relative_path=relative_path)
        for relative_path in relative_paths
    ]
    identity = {
        "schema_version": NATIVE_EXACT_INNER_ADAPTER_VERSION,
        "family": family,
        "native_backend": FAMILY_NATIVE_BACKEND[family],
        "native_fit_apis": list(FAMILY_NATIVE_APIS[family]),
        "files": files,
    }
    return {**identity, "content_sha256": _sha256_json(identity)}


def native_family_configuration_sha256(
    family: str,
    configuration: Mapping[str, Any],
) -> str:
    """Bind one JSON configuration projection to its architecture and APIs."""

    if family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError("family is not an active Stage 1 architecture")
    if not isinstance(configuration, Mapping):
        raise TypeError("native family configuration must be a mapping")
    return _sha256_json(
        {
            "schema_version": NATIVE_EXACT_INNER_ADAPTER_VERSION,
            "family": family,
            "native_backend": FAMILY_NATIVE_BACKEND[family],
            "native_fit_apis": list(FAMILY_NATIVE_APIS[family]),
            "configuration": copy.deepcopy(dict(configuration)),
        }
    )


def native_artifact_sha256(path: Path | str) -> str:
    """Hash one immutable native file or directory tree without path aliases."""

    root = Path(path)
    if root.is_symlink():
        raise ValueError("native artifact cannot be a symlink")
    if root.is_file():
        return _stable_file_inventory_row(root, relative_path=root.name)["sha256"]
    if not root.is_dir():
        raise FileNotFoundError(f"native artifact is absent: {root}")
    candidates = sorted(item for item in root.rglob("*") if item.is_file() or item.is_symlink())
    if not candidates:
        raise ValueError("native artifact directory cannot be empty")
    inventory: list[dict[str, Any]] = []
    for candidate in candidates:
        relative = candidate.relative_to(root).as_posix()
        inventory.append(_stable_file_inventory_row(candidate, relative_path=relative))
    return _sha256_json({"artifact_tree": inventory})


def _resolved_artifact_path(path: Path | str) -> str:
    artifact = Path(path)
    if artifact.is_symlink():
        raise ValueError("native artifact cannot be a symlink")
    try:
        return str(artifact.resolve(strict=True))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"native artifact is absent: {artifact}") from exc


def _read_stable_json_artifact(path: Path | str) -> tuple[dict[str, Any], str]:
    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError("native JSON artifact must be one regular file")
    before = artifact.stat()
    payload = artifact.read_bytes()
    after = artifact.stat()
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if before_identity != after_identity:
        raise RuntimeError("native JSON artifact changed while reading")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("native artifact is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("native artifact must be one JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _native_family_execution_record_body(
    *,
    family: str,
    fit_semantics: str,
    outer_fold: int,
    inner_fold: int,
    split_scope_fingerprint: str,
    data_projection_sha256: str,
    fit_row_fingerprint: str,
    heldout_row_fingerprint: str,
    evidence_payload_sha256: str,
    producer_code_sha256: str,
    configuration_sha256: str,
    native_fit_metadata_sha256: str,
    model_artifact_sha256: str,
    source_artifact_sha256: str,
    model_artifact_semantics: str,
) -> dict[str, Any]:
    return {
        "schema_version": NATIVE_FAMILY_EXECUTION_RECORD_VERSION,
        "status": "completed",
        "family": family,
        "native_backend": FAMILY_NATIVE_BACKEND[family],
        "native_fit_apis": list(FAMILY_NATIVE_APIS[family]),
        "fit_semantics": fit_semantics,
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "split_scope_fingerprint": split_scope_fingerprint,
        "data_projection_sha256": data_projection_sha256,
        "fit_row_fingerprint": fit_row_fingerprint,
        "heldout_row_fingerprint": heldout_row_fingerprint,
        "evidence_payload_sha256": evidence_payload_sha256,
        "producer_code_sha256": producer_code_sha256,
        "configuration_sha256": configuration_sha256,
        "native_fit_metadata_sha256": native_fit_metadata_sha256,
        "model_artifact_sha256": model_artifact_sha256,
        "source_artifact_sha256": source_artifact_sha256,
        "model_artifact_semantics": model_artifact_semantics,
        "heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
    }


def native_family_execution_record(
    *,
    family: str,
    fit_semantics: str,
    outer_fold: int,
    inner_fold: int,
    split_scope_fingerprint: str,
    data_projection_sha256: str,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    evidence_payload: Mapping[str, Any],
    configuration: Mapping[str, Any],
    native_fit_metadata_path: Path | str,
    model_artifact_path: Path | str,
    source_artifact_path: Path | str,
    model_artifact_semantics: str,
) -> dict[str, Any]:
    """Build the closed record a genuine native scope runner must persist."""

    if family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError("family is not an active Stage 1 architecture")
    if fit_semantics not in _VALID_FIT_SEMANTICS:
        raise ValueError("native execution has unsupported fit semantics")
    if int(outer_fold) < 1 or int(inner_fold) < 1:
        raise ValueError("native execution fold identifiers must be positive")
    if not isinstance(model_artifact_semantics, str) or not model_artifact_semantics.strip():
        raise ValueError("native model-artifact semantics cannot be empty")
    fit_rows = _ordered_integer_rows(fit_row_ids, name="fit_row_ids")
    heldout_rows = _ordered_integer_rows(heldout_row_ids, name="heldout_row_ids")
    if set(fit_rows) & set(heldout_rows):
        raise ValueError("native execution fit and held-out rows overlap")
    if not isinstance(evidence_payload, Mapping) or not evidence_payload:
        raise ValueError("native evidence payload must be one nonempty mapping")
    code_sha256 = str(native_family_code_identity(family)["content_sha256"])
    configuration_sha256 = native_family_configuration_sha256(family, configuration)
    (
        native_fit_metadata_sha256,
        tfidf_score_artifact_path,
        tfidf_model_artifact_path,
    ) = _verify_tfidf_nested_fit_metadata(
        native_fit_metadata_path,
        family=family,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        configuration=configuration,
    )
    if tfidf_score_artifact_path is not None and _resolved_artifact_path(
        source_artifact_path
    ) != _resolved_artifact_path(tfidf_score_artifact_path):
        raise ValueError("TF-IDF source artifact must be its verified score-selection JSON")
    if tfidf_model_artifact_path is not None and _resolved_artifact_path(
        model_artifact_path
    ) != _resolved_artifact_path(tfidf_model_artifact_path):
        raise ValueError("TF-IDF model artifact must be its fitted-context artifact")
    return _native_family_execution_record_body(
        family=family,
        fit_semantics=fit_semantics,
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        split_scope_fingerprint=_require_sha256(
            split_scope_fingerprint,
            name="split_scope_fingerprint",
        ),
        data_projection_sha256=_require_sha256(
            data_projection_sha256,
            name="data_projection_sha256",
        ),
        fit_row_fingerprint=row_order_fingerprint(fit_rows),
        heldout_row_fingerprint=row_order_fingerprint(heldout_rows),
        evidence_payload_sha256=_sha256_json(copy.deepcopy(dict(evidence_payload))),
        producer_code_sha256=code_sha256,
        configuration_sha256=configuration_sha256,
        native_fit_metadata_sha256=native_fit_metadata_sha256,
        model_artifact_sha256=native_artifact_sha256(model_artifact_path),
        source_artifact_sha256=native_artifact_sha256(source_artifact_path),
        model_artifact_semantics=model_artifact_semantics,
    )


def _verify_semantic_retrieval_training_scope_metadata(
    metadata: Mapping[str, Any],
    *,
    fit_rows: tuple[int, ...],
    heldout_rows: tuple[int, ...],
    configured_calibration_folds: int,
    configuration: Mapping[str, Any],
) -> None:
    """Validate exhaustive semantic TF-IDF replay without false selection claims."""

    policy = metadata.get("tfidf_training_scope_policy")
    if not isinstance(policy, Mapping) or set(policy) != set(
        _SEMANTIC_RETRIEVAL_TRAINING_POLICY_FIELDS
    ):
        raise ValueError("semantic retrieval training-scope policy is not a closed schema")
    model_rows = _ordered_integer_rows(
        policy.get("model_fit_row_ids") or (),
        name="semantic model replay rows",
    )
    calibration_rows = _ordered_integer_rows(
        policy.get("calibration_row_ids") or (),
        name="semantic calibration replay rows",
    )
    try:
        seed = int(policy.get("seed"))
        fold_count = int(policy.get("fold_count"))
        base_seed = int(configuration.get("seed"))
        outer_fold = int(metadata.get("outer_fold"))
        inner_fold = int(metadata.get("inner_fold"))
    except (TypeError, ValueError) as exc:
        raise ValueError("semantic retrieval replay seed/fold count is invalid") from exc
    if (
        metadata.get("capture_schema_version") != EMBEDDING_NATIVE_CAPTURE_SCHEMA
        or isinstance(configuration.get("seed"), bool)
        or base_seed != configuration.get("seed")
        or isinstance(metadata.get("seed"), bool)
        or metadata.get("seed") != base_seed
        or outer_fold < 1
        or inner_fold < 1
        or configuration.get("capture_schema_version") != EMBEDDING_NATIVE_CAPTURE_SCHEMA
        or configuration.get("semantic_policy_schema_version")
        != SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA
        or configuration.get("heldout_label_policy") != "id_only_no_transform"
        or tuple(map(int, metadata.get("fit_row_ids") or ())) != fit_rows
        or tuple(map(int, metadata.get("heldout_row_ids") or ())) != heldout_rows
        or metadata.get("fit_row_order_fingerprint") != row_order_fingerprint(fit_rows)
        or metadata.get("heldout_row_order_fingerprint") != row_order_fingerprint(heldout_rows)
        or metadata.get("registered_heldout_columns_read") != ["_oci_row_id"]
        or metadata.get("registered_heldout_labels_accessed") is not False
        or metadata.get("registered_heldout_text_accessed") is not False
        or metadata.get("registered_heldout_transform_performed") is not False
        or metadata.get("tfidf_training_scope_policy") != dict(policy)
        or set(model_rows) & set(calibration_rows)
        or set(model_rows) | set(calibration_rows) != set(fit_rows)
        or policy.get("model_fit_row_order_fingerprint") != row_order_fingerprint(model_rows)
        or policy.get("calibration_row_order_fingerprint")
        != row_order_fingerprint(calibration_rows)
        or isinstance(policy.get("seed"), bool)
        or seed != policy.get("seed")
        or seed != base_seed + 73_000 + 1_009 * outer_fold + inner_fold
        or isinstance(policy.get("fold_count"), bool)
        or fold_count != policy.get("fold_count")
        or policy.get("schema_version") != SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA
        or policy.get("policy") != "training_only_exhaustive_no_selection"
        or policy.get("selection_kind") != "none_deterministic_exhaustive"
        or policy.get("nested_calibration_applicability") != "no_label_or_hyperparameter_selection"
        or policy.get("fold_parameter") != "tfidf_nested_calibration_folds"
        or int(policy.get("configured_fold_count", 0)) != configured_calibration_folds
        or not 2 <= fold_count <= configured_calibration_folds
        or policy.get("split_method") != "ordered_row_positions_seeded_label_free_partition"
        or policy.get("partitions_are_replay_canaries_only") is not True
        or policy.get("partition_canaries_select_or_drop_terms") is not False
        or policy.get("authoritative_projection_scope") != "all_exact_fit_frozen_retrieval_tails"
        or policy.get("projection_vocabulary_max_features") is not None
        or policy.get("projection_output_limit") is not None
        or policy.get("all_nonzero_sanitized_terms_preserved") is not True
        or policy.get("upstream_embedding_directions_and_retrieval_use_exact_fit_labels_only")
        is not True
        or policy.get("nested_calibration_labels_accessed") is not False
        or policy.get("registered_heldout_labels_accessed") is not False
        or policy.get("registered_heldout_text_accessed") is not False
        or policy.get("registered_heldout_transform_performed") is not False
        or policy.get("selection_frozen_before_registered_heldout_use") is not True
        or policy.get("projection_frozen_before_registered_heldout_use") is not True
        or policy.get("canonical_hierarchy_partition_count_used_as_calibration_folds") is not False
        or policy.get("interaction_inner_folds_used_as_calibration_folds") is not False
    ):
        raise ValueError(
            "semantic retrieval policy must be exhaustive, label-free, uncapped, "
            "and independent of registered heldout rows"
        )


def _verify_tfidf_nested_fit_metadata(
    path: Path | str,
    *,
    family: str,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    configuration: Mapping[str, Any],
) -> tuple[str, Path | None, Path | None]:
    metadata, metadata_sha256 = _read_stable_json_artifact(path)
    if family not in {
        TFIDF_SEMANTIC_RETRIEVAL,
        TFIDF_TOPICS,
        TFIDF_ORPHAN_NGRAMS,
    }:
        return metadata_sha256, None, None
    expected_text_column = configuration.get("text_column")
    if not isinstance(expected_text_column, str) or not expected_text_column.strip():
        raise ValueError("TF-IDF native configuration must bind its text_column")
    if any(
        marker in expected_text_column.casefold()
        for marker in ("treatment", "outcome", "oracle", "true_")
    ):
        raise ValueError("TF-IDF native text_column cannot name a label or oracle field")
    fit_rows = tuple(map(int, fit_row_ids))
    heldout_rows = tuple(map(int, heldout_row_ids))
    configured_calibration_folds = int(configuration.get("tfidf_nested_calibration_folds", 0))
    if configured_calibration_folds < 2:
        raise ValueError(
            "TF-IDF native configuration must bind tfidf_nested_calibration_folds >= 2"
        )
    if family == TFIDF_SEMANTIC_RETRIEVAL:
        _verify_semantic_retrieval_training_scope_metadata(
            metadata,
            fit_rows=fit_rows,
            heldout_rows=heldout_rows,
            configured_calibration_folds=configured_calibration_folds,
            configuration=configuration,
        )
        return metadata_sha256, None, None
    nesting = metadata.get("selection_nesting")
    if not isinstance(nesting, Mapping):
        raise ValueError("TF-IDF native metadata has no nested selection plan")
    model_rows = tuple(map(int, nesting.get("model_fit_row_ids") or ()))
    calibration_rows = tuple(map(int, nesting.get("calibration_row_ids") or ()))
    registered_columns = metadata.get("registered_heldout_columns_read")
    if (
        metadata.get("score_selection_label_policy") != "nested_fit_calibration"
        or metadata.get("registered_heldout_labels_accessed") is not False
        or _SHA256.fullmatch(str(metadata.get("selection_frozen_sha256") or "")) is None
        or tuple(map(int, metadata.get("fit_row_ids") or ())) != fit_rows
        or tuple(map(int, metadata.get("heldout_row_ids") or ())) != heldout_rows
        or metadata.get("fit_row_fingerprint") != row_set_fingerprint(fit_rows)
        or metadata.get("heldout_row_fingerprint") != row_set_fingerprint(heldout_rows)
        or nesting.get("schema_version") != TFIDF_NESTED_CALIBRATION_SCHEMA_VERSION
        or nesting.get("policy") != "nested_fit_calibration"
        or nesting.get("fold_parameter") != "tfidf_nested_calibration_folds"
        or int(nesting.get("configured_fold_count", 0)) != configured_calibration_folds
        or not 2 <= int(nesting.get("fold_count", 0)) <= configured_calibration_folds
        or not 1 <= int(nesting.get("selected_fold", 0)) <= int(nesting.get("fold_count", 0))
        or nesting.get("canonical_hierarchy_partition_count_used") is not False
        or nesting.get("interaction_inner_folds_used") is not False
        or nesting.get("registered_heldout_labels_accessed") is not False
        or nesting.get("nested_calibration_labels_accessed") is not True
        or nesting.get("selection_frozen_before_registered_heldout_transform") is not True
        or not model_rows
        or not calibration_rows
        or set(model_rows) & set(calibration_rows)
        or set(model_rows) | set(calibration_rows) != set(fit_rows)
        or not set(model_rows) < set(fit_rows)
        or nesting.get("model_fit_row_fingerprint") != row_set_fingerprint(model_rows)
        or nesting.get("calibration_row_fingerprint") != row_set_fingerprint(calibration_rows)
        or not isinstance(registered_columns, list)
        or len(registered_columns) != 2
        or registered_columns[0] != "_oci_row_id"
        or registered_columns[1] != expected_text_column
    ):
        raise ValueError("TF-IDF native metadata violates nested heldout-label isolation")
    compact = metadata.get("topic_score_tests")
    if (
        not isinstance(compact, Mapping)
        or compact.get("schema_version") != TOPIC_SCORE_TEST_SCHEMA_VERSION
        or compact.get("status") != "completed"
        or compact.get("uses_heldout_treatment_and_outcome") is not False
        or compact.get("uses_registered_heldout_treatment_and_outcome") is not False
        or compact.get("uses_nested_fit_calibration_treatment_and_outcome") is not True
        or compact.get("score_selection_label_policy") != "nested_fit_calibration"
        or compact.get("selection_frozen_sha256") != metadata.get("selection_frozen_sha256")
    ):
        raise ValueError("TF-IDF compact score metadata is not nested-label-safe")
    score_path = Path(str((metadata.get("artifacts") or {}).get("topic_score_tests") or ""))
    if not score_path.is_absolute():
        score_path = Path(path).resolve().parent / score_path
    model_path = Path(str((metadata.get("artifacts") or {}).get("fitted_context") or ""))
    if not model_path.is_absolute():
        model_path = Path(path).resolve().parent / model_path
    if not model_path.is_file():
        raise ValueError("TF-IDF native metadata has no fitted-context artifact")
    score, _score_sha256 = _read_stable_json_artifact(score_path)
    if (
        score.get("schema_version") != TOPIC_SCORE_TEST_SCHEMA_VERSION
        or score.get("status") != "completed"
        or score.get("score_selection_label_policy") != "nested_fit_calibration"
        or score.get("uses_heldout_treatment_and_outcome") is not False
        or score.get("uses_registered_heldout_treatment_and_outcome") is not False
        or score.get("uses_nested_fit_calibration_treatment_and_outcome") is not True
        or score.get("selection_frozen_sha256") != metadata.get("selection_frozen_sha256")
        or score.get("nested_calibration_schema_version") != TFIDF_NESTED_CALIBRATION_SCHEMA_VERSION
        or score.get("nested_model_fit_row_fingerprint") != row_set_fingerprint(model_rows)
        or score.get("nested_calibration_row_fingerprint") != row_set_fingerprint(calibration_rows)
    ):
        raise ValueError("TF-IDF score artifact is not nested-label-safe")
    if family == TFIDF_ORPHAN_NGRAMS:
        orphan = score.get("effect_orphan_ngram_branch")
        if (
            not isinstance(orphan, Mapping)
            or orphan.get("uses_heldout_treatment_and_outcome") is not False
            or orphan.get("uses_registered_heldout_treatment_and_outcome") is not False
            or orphan.get("uses_nested_fit_calibration_treatment_and_outcome") is not True
            or orphan.get("cluster_construction_uses_heldout_rows_or_labels") is not False
            or orphan.get("topic_term_exclusion_is_fit_side") is not True
        ):
            raise ValueError("TF-IDF orphan metadata is not nested-label-safe")
    return metadata_sha256, score_path, model_path


def _ordered_integer_rows(values: Sequence[Any], *, name: str) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError(f"{name} cannot contain boolean row values")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain integers") from exc
        if parsed < 0 or parsed != value:
            raise ValueError(f"{name} must contain non-negative integers")
        result.append(parsed)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    return tuple(result)


def _without_member_ids(value: Any) -> Any:
    """Remove catalog-local IDs while preserving every semantic member.

    The full-outer clone canary must compare scientific payload semantics, not
    split-specific catalog IDs.  Otherwise copied evidence could evade the
    canary merely by being re-addressed under a different split fingerprint.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _without_member_ids(child)
            for key, child in value.items()
            if str(key) != "member_id"
        }
    if isinstance(value, (list, tuple)):
        return [_without_member_ids(child) for child in value]
    return copy.deepcopy(value)


def family_payload_from_catalog(
    catalog: RoleNeutralEvidenceCatalog,
    *,
    family: str,
) -> tuple[dict[str, Any], int]:
    """Project one architecture only from a validated lossless catalog.

    Evidence IDs, origins, split hashes, and member IDs stay in the machine
    envelope.  The returned payload contains every architecture-local semantic
    atom and is therefore suitable both for the clone canary and for later
    architecture-at-a-time interpretation.
    """

    if not isinstance(catalog, RoleNeutralEvidenceCatalog):
        raise TypeError("catalog must be RoleNeutralEvidenceCatalog")
    validate_role_neutral_catalog(catalog)
    if family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError("family is not an active Stage 1 concept architecture")
    items = [
        {
            "atom_kind": atom.atom_kind,
            "source_kind": atom.source_kind,
            "observable_axes": list(atom.observable_axes),
            "content": _without_member_ids(atom.content),
        }
        for atom in catalog.family_atoms(family)
    ]
    items.sort(key=_canonical_json)
    if not items:
        raise ValueError(f"native scope has no concept-bearing evidence for {family}")
    if family == TFIDF_SEMANTIC_RETRIEVAL and any(
        item.get("atom_kind") != "tfidf_semantic_retrieval_contrast"
        or (item.get("content") or {}).get("architecture_view") != SEMANTIC_RETRIEVAL_DERIVATION
        or (item.get("content") or {}).get("source_passages_removed") is not True
        for item in items
    ):
        raise ValueError("semantic-retrieval TF-IDF payload is not the label-free projection")
    return (
        {
            "schema_version": NATIVE_FAMILY_PAYLOAD_VERSION,
            "family": family,
            "architecture_evidence": items,
        },
        len(items),
    )


def _verified_catalog_artifact_sha256(
    catalog: RoleNeutralEvidenceCatalog,
    path: Path | str,
) -> str:
    persisted, artifact_sha256 = _read_stable_json_artifact(path)
    if persisted != catalog.as_dict():
        raise ValueError("persisted native catalog differs from the validated catalog object")
    return artifact_sha256


@dataclass(frozen=True)
class AuthenticatedNativeFullOuterPayloadRegistry:
    """Catalog-derived clone canaries for one genuine full-outer Stage 1 run."""

    outer_fold: int
    split_scope_fingerprint: str
    fit_row_ids: tuple[int, ...]
    heldout_row_ids: tuple[int, ...]
    catalog_sha256: str
    catalog_artifact_sha256: str
    payload_sha256_by_family: Mapping[str, str]
    _catalog_artifact_path: str = field(repr=False, compare=False)
    _construction_authority: object = field(repr=False, compare=False)
    schema_version: str = NATIVE_FULL_OUTER_PAYLOAD_REGISTRY_VERSION
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self._construction_authority is not _NATIVE_FULL_OUTER_REGISTRY_AUTHORITY:
            raise TypeError("full-outer payload registries must be built from a catalog artifact")
        if self.schema_version != NATIVE_FULL_OUTER_PAYLOAD_REGISTRY_VERSION:
            raise ValueError("unsupported full-outer payload-registry schema")
        if int(self.outer_fold) < 1:
            raise ValueError("full-outer fold identifier must be positive")
        object.__setattr__(self, "outer_fold", int(self.outer_fold))
        for name in (
            "split_scope_fingerprint",
            "catalog_sha256",
            "catalog_artifact_sha256",
        ):
            _require_sha256(getattr(self, name), name=f"full-outer {name}")
        fit_rows = _ordered_integer_rows(self.fit_row_ids, name="full-outer fit_row_ids")
        heldout_rows = _ordered_integer_rows(
            self.heldout_row_ids,
            name="full-outer heldout_row_ids",
        )
        if set(fit_rows) & set(heldout_rows):
            raise ValueError("full-outer fit and held-out rows overlap")
        if set(self.payload_sha256_by_family) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("full-outer payload registry must cover exactly all ten families")
        payload_hashes = {
            family: _require_sha256(
                self.payload_sha256_by_family[family],
                name=f"{family} full-outer payload_sha256",
            )
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        }
        body = {
            "schema_version": self.schema_version,
            "outer_fold": int(self.outer_fold),
            "scope": "outer_train",
            "split_scope_fingerprint": self.split_scope_fingerprint,
            "fit_row_ids": list(fit_rows),
            "heldout_row_ids": list(heldout_rows),
            "catalog_sha256": self.catalog_sha256,
            "catalog_artifact_sha256": self.catalog_artifact_sha256,
            "payload_sha256_by_family": payload_hashes,
        }
        object.__setattr__(self, "fit_row_ids", fit_rows)
        object.__setattr__(self, "heldout_row_ids", heldout_rows)
        object.__setattr__(
            self,
            "payload_sha256_by_family",
            MappingProxyType(payload_hashes),
        )
        object.__setattr__(
            self,
            "_catalog_artifact_path",
            _resolved_artifact_path(self._catalog_artifact_path),
        )
        object.__setattr__(self, "content_sha256", _sha256_json(body))
        self.verify_artifact_bytes()

    def verify_artifact_bytes(self) -> None:
        if native_artifact_sha256(self._catalog_artifact_path) != self.catalog_artifact_sha256:
            raise RuntimeError("authenticated full-outer catalog artifact changed after binding")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "outer_fold": self.outer_fold,
            "scope": "outer_train",
            "split_scope_fingerprint": self.split_scope_fingerprint,
            "fit_row_ids": list(self.fit_row_ids),
            "heldout_row_ids": list(self.heldout_row_ids),
            "catalog_sha256": self.catalog_sha256,
            "catalog_artifact_sha256": self.catalog_artifact_sha256,
            "payload_sha256_by_family": dict(self.payload_sha256_by_family),
            "content_sha256": self.content_sha256,
        }


def native_full_outer_payload_registry_from_catalog(
    *,
    catalog: RoleNeutralEvidenceCatalog,
    outer_fold: int,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    catalog_artifact_path: Path | str,
) -> AuthenticatedNativeFullOuterPayloadRegistry:
    """Authenticate an outer catalog and derive all ten clone-canary hashes."""

    if not isinstance(catalog, RoleNeutralEvidenceCatalog):
        raise TypeError("catalog must be RoleNeutralEvidenceCatalog")
    validate_role_neutral_catalog(catalog)
    fit_rows = _ordered_integer_rows(fit_row_ids, name="full-outer fit_row_ids")
    heldout_rows = _ordered_integer_rows(
        heldout_row_ids,
        name="full-outer heldout_row_ids",
    )
    if set(fit_rows) & set(heldout_rows):
        raise ValueError("full-outer fit and held-out rows overlap")
    if (
        catalog.scope != "outer_train"
        or catalog.outer_fold != int(outer_fold)
        or catalog.inner_fold is not None
    ):
        raise ValueError("catalog fold provenance differs from the full-outer scope")
    expected_split = FoldEvidenceProvenance(
        outer_fold=int(outer_fold),
        train_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        scope="outer_train",
        artifact_id="native-full-outer-payload-registry",
    ).split_fingerprint
    if catalog.split_fingerprint != expected_split:
        raise ValueError("catalog row provenance differs from the full-outer scope")
    catalog_artifact_sha256 = _verified_catalog_artifact_sha256(
        catalog,
        catalog_artifact_path,
    )
    payload_hashes: dict[str, str] = {}
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        payload, _count = family_payload_from_catalog(catalog, family=family)
        payload_hashes[family] = _sha256_json(payload)
    return AuthenticatedNativeFullOuterPayloadRegistry(
        outer_fold=int(outer_fold),
        split_scope_fingerprint=expected_split,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        catalog_sha256=catalog.catalog_sha256,
        catalog_artifact_sha256=catalog_artifact_sha256,
        payload_sha256_by_family=payload_hashes,
        _catalog_artifact_path=_resolved_artifact_path(catalog_artifact_path),
        _construction_authority=_NATIVE_FULL_OUTER_REGISTRY_AUTHORITY,
    )


@dataclass(frozen=True)
class NativeFamilyFitProof:
    """Machine-envelope proof emitted by one genuine native family fit."""

    family: str
    native_backend: str
    native_fit_apis: tuple[str, ...]
    fit_semantics: str
    outer_fold: int
    inner_fold: int
    split_scope_fingerprint: str
    data_projection_sha256: str
    fit_row_fingerprint: str
    heldout_row_fingerprint: str
    evidence_payload_sha256: str
    producer_code_sha256: str
    configuration_sha256: str
    native_fit_metadata_sha256: str
    native_execution_record_sha256: str
    fit_execution_sha256: str
    model_artifact_sha256: str
    source_artifact_sha256: str
    model_artifact_semantics: str
    _native_fit_metadata_path: str = field(repr=False, compare=False)
    _native_execution_record_path: str = field(repr=False, compare=False)
    _model_artifact_path: str = field(repr=False, compare=False)
    _source_artifact_path: str = field(repr=False, compare=False)
    _construction_authority: object = field(repr=False, compare=False)
    heldout_labels_accessed: bool = False
    oracle_fields_accessed: bool = False
    secrets_accessed: bool = False
    schema_version: str = NATIVE_FAMILY_FIT_PROOF_VERSION

    def __post_init__(self) -> None:
        if self._construction_authority is not _NATIVE_PROOF_CONSTRUCTION_AUTHORITY:
            raise TypeError("native family proofs must be built from verified artifacts")
        if self.schema_version != NATIVE_FAMILY_FIT_PROOF_VERSION:
            raise ValueError("unsupported native family fit-proof schema")
        if self.family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("native fit proof has an inactive architecture family")
        if self.native_backend != FAMILY_NATIVE_BACKEND[self.family]:
            raise ValueError(f"{self.family} fit proof changed its native modeling backend")
        native_fit_apis = tuple(self.native_fit_apis)
        if native_fit_apis != FAMILY_NATIVE_APIS[self.family]:
            raise ValueError(f"{self.family} fit proof changed its real native fit APIs")
        object.__setattr__(self, "native_fit_apis", native_fit_apis)
        if self.fit_semantics not in _VALID_FIT_SEMANTICS:
            raise ValueError(f"{self.family} fit proof has unsupported fit semantics")
        if int(self.outer_fold) < 1 or int(self.inner_fold) < 1:
            raise ValueError(f"{self.family} native fit proof has invalid fold coordinates")
        object.__setattr__(self, "outer_fold", int(self.outer_fold))
        object.__setattr__(self, "inner_fold", int(self.inner_fold))
        for name in (
            "split_scope_fingerprint",
            "data_projection_sha256",
            "fit_row_fingerprint",
            "heldout_row_fingerprint",
            "evidence_payload_sha256",
            "producer_code_sha256",
            "configuration_sha256",
            "native_fit_metadata_sha256",
            "native_execution_record_sha256",
            "fit_execution_sha256",
            "model_artifact_sha256",
            "source_artifact_sha256",
        ):
            _require_sha256(getattr(self, name), name=f"{self.family} {name}")
        if (
            not isinstance(self.model_artifact_semantics, str)
            or not self.model_artifact_semantics.strip()
        ):
            raise ValueError(f"{self.family} model-artifact semantics cannot be empty")
        for name in (
            "heldout_labels_accessed",
            "oracle_fields_accessed",
            "secrets_accessed",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"{self.family} native fit proof must attest {name}=false")
        expected_execution = _native_fit_execution_sha256(
            family=self.family,
            fit_semantics=self.fit_semantics,
            outer_fold=self.outer_fold,
            inner_fold=self.inner_fold,
            split_scope_fingerprint=self.split_scope_fingerprint,
            data_projection_sha256=self.data_projection_sha256,
            fit_row_fingerprint=self.fit_row_fingerprint,
            heldout_row_fingerprint=self.heldout_row_fingerprint,
            evidence_payload_sha256=self.evidence_payload_sha256,
            producer_code_sha256=self.producer_code_sha256,
            configuration_sha256=self.configuration_sha256,
            native_fit_metadata_sha256=self.native_fit_metadata_sha256,
            native_execution_record_sha256=self.native_execution_record_sha256,
            model_artifact_sha256=self.model_artifact_sha256,
            source_artifact_sha256=self.source_artifact_sha256,
            model_artifact_semantics=self.model_artifact_semantics,
        )
        if self.fit_execution_sha256 != expected_execution:
            raise ValueError(f"{self.family} fit execution identity is not scope-bound")
        for name in (
            "_native_fit_metadata_path",
            "_native_execution_record_path",
            "_model_artifact_path",
            "_source_artifact_path",
        ):
            object.__setattr__(self, name, _resolved_artifact_path(getattr(self, name)))
        self.verify_artifact_bytes()

    def verify_artifact_bytes(self) -> None:
        observed = {
            "native fit metadata": native_artifact_sha256(self._native_fit_metadata_path),
            "native execution record": native_artifact_sha256(self._native_execution_record_path),
            "model artifact": native_artifact_sha256(self._model_artifact_path),
            "source artifact": native_artifact_sha256(self._source_artifact_path),
        }
        expected = {
            "native fit metadata": self.native_fit_metadata_sha256,
            "native execution record": self.native_execution_record_sha256,
            "model artifact": self.model_artifact_sha256,
            "source artifact": self.source_artifact_sha256,
        }
        changed = [name for name in expected if observed[name] != expected[name]]
        if changed:
            raise RuntimeError(
                f"{self.family} authenticated native artifact changed after binding: {changed}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "native_backend": self.native_backend,
            "native_fit_apis": list(self.native_fit_apis),
            "fit_semantics": self.fit_semantics,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "split_scope_fingerprint": self.split_scope_fingerprint,
            "data_projection_sha256": self.data_projection_sha256,
            "fit_row_fingerprint": self.fit_row_fingerprint,
            "heldout_row_fingerprint": self.heldout_row_fingerprint,
            "evidence_payload_sha256": self.evidence_payload_sha256,
            "producer_code_sha256": self.producer_code_sha256,
            "configuration_sha256": self.configuration_sha256,
            "native_fit_metadata_sha256": self.native_fit_metadata_sha256,
            "native_execution_record_sha256": self.native_execution_record_sha256,
            "fit_execution_sha256": self.fit_execution_sha256,
            "model_artifact_sha256": self.model_artifact_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "model_artifact_semantics": self.model_artifact_semantics,
            "heldout_labels_accessed": self.heldout_labels_accessed,
            "oracle_fields_accessed": self.oracle_fields_accessed,
            "secrets_accessed": self.secrets_accessed,
        }


def _native_fit_execution_sha256(
    *,
    family: str,
    fit_semantics: str,
    outer_fold: int,
    inner_fold: int,
    split_scope_fingerprint: str,
    data_projection_sha256: str,
    fit_row_fingerprint: str,
    heldout_row_fingerprint: str,
    evidence_payload_sha256: str,
    producer_code_sha256: str,
    configuration_sha256: str,
    native_fit_metadata_sha256: str,
    native_execution_record_sha256: str,
    model_artifact_sha256: str,
    source_artifact_sha256: str,
    model_artifact_semantics: str,
) -> str:
    return _sha256_json(
        {
            "schema_version": NATIVE_FAMILY_FIT_PROOF_VERSION,
            "family": family,
            "native_backend": FAMILY_NATIVE_BACKEND[family],
            "native_fit_apis": list(FAMILY_NATIVE_APIS[family]),
            "fit_semantics": fit_semantics,
            "outer_fold": int(outer_fold),
            "inner_fold": int(inner_fold),
            "split_scope_fingerprint": split_scope_fingerprint,
            "data_projection_sha256": data_projection_sha256,
            "fit_row_fingerprint": fit_row_fingerprint,
            "heldout_row_fingerprint": heldout_row_fingerprint,
            "evidence_payload_sha256": evidence_payload_sha256,
            "producer_code_sha256": producer_code_sha256,
            "configuration_sha256": configuration_sha256,
            "native_fit_metadata_sha256": native_fit_metadata_sha256,
            "native_execution_record_sha256": native_execution_record_sha256,
            "model_artifact_sha256": model_artifact_sha256,
            "source_artifact_sha256": source_artifact_sha256,
            "model_artifact_semantics": model_artifact_semantics,
            "heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
            "secrets_accessed": False,
        }
    )


def bind_native_family_fit_proof(
    *,
    family: str,
    fit_semantics: str,
    outer_fold: int,
    inner_fold: int,
    split_scope_fingerprint: str,
    data_projection_sha256: str,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    evidence_payload: Mapping[str, Any],
    configuration: Mapping[str, Any],
    native_fit_metadata_path: Path | str,
    native_execution_record_path: Path | str,
    model_artifact_path: Path | str,
    source_artifact_path: Path | str,
    model_artifact_semantics: str,
) -> NativeFamilyFitProof:
    """Verify real artifacts and seal them to one exact scope and payload."""

    if family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError("family is not an active Stage 1 architecture")
    if not isinstance(evidence_payload, Mapping) or not evidence_payload:
        raise ValueError("native evidence payload must be one nonempty mapping")
    fit_rows = _ordered_integer_rows(fit_row_ids, name="fit_row_ids")
    heldout_rows = _ordered_integer_rows(heldout_row_ids, name="heldout_row_ids")
    if set(fit_rows) & set(heldout_rows):
        raise ValueError("native proof fit and held-out rows overlap")
    fit_row_sha256 = row_order_fingerprint(fit_rows)
    heldout_row_sha256 = row_order_fingerprint(heldout_rows)
    payload_sha256 = hashlib.sha256(
        _canonical_json(copy.deepcopy(dict(evidence_payload))).encode("utf-8")
    ).hexdigest()
    code_sha256 = str(native_family_code_identity(family)["content_sha256"])
    configuration_sha256 = native_family_configuration_sha256(family, configuration)
    (
        native_fit_metadata_sha256,
        tfidf_score_artifact_path,
        tfidf_model_artifact_path,
    ) = _verify_tfidf_nested_fit_metadata(
        native_fit_metadata_path,
        family=family,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        configuration=configuration,
    )
    if tfidf_score_artifact_path is not None and _resolved_artifact_path(
        source_artifact_path
    ) != _resolved_artifact_path(tfidf_score_artifact_path):
        raise ValueError("TF-IDF source artifact must be its verified score-selection JSON")
    if tfidf_model_artifact_path is not None and _resolved_artifact_path(
        model_artifact_path
    ) != _resolved_artifact_path(tfidf_model_artifact_path):
        raise ValueError("TF-IDF model artifact must be its fitted-context artifact")
    model_artifact_sha256 = native_artifact_sha256(model_artifact_path)
    source_artifact_sha256 = native_artifact_sha256(source_artifact_path)
    expected_record = _native_family_execution_record_body(
        family=family,
        fit_semantics=fit_semantics,
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        split_scope_fingerprint=split_scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        fit_row_fingerprint=fit_row_sha256,
        heldout_row_fingerprint=heldout_row_sha256,
        evidence_payload_sha256=payload_sha256,
        producer_code_sha256=code_sha256,
        configuration_sha256=configuration_sha256,
        native_fit_metadata_sha256=native_fit_metadata_sha256,
        model_artifact_sha256=model_artifact_sha256,
        source_artifact_sha256=source_artifact_sha256,
        model_artifact_semantics=model_artifact_semantics,
    )
    execution_record, execution_record_sha256 = _read_stable_json_artifact(
        native_execution_record_path
    )
    if execution_record != expected_record:
        raise ValueError("native execution record differs from verified scope artifacts")
    execution_sha256 = _native_fit_execution_sha256(
        family=family,
        fit_semantics=fit_semantics,
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        split_scope_fingerprint=split_scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        fit_row_fingerprint=fit_row_sha256,
        heldout_row_fingerprint=heldout_row_sha256,
        evidence_payload_sha256=payload_sha256,
        producer_code_sha256=code_sha256,
        configuration_sha256=configuration_sha256,
        native_fit_metadata_sha256=native_fit_metadata_sha256,
        native_execution_record_sha256=execution_record_sha256,
        model_artifact_sha256=model_artifact_sha256,
        source_artifact_sha256=source_artifact_sha256,
        model_artifact_semantics=model_artifact_semantics,
    )
    return NativeFamilyFitProof(
        family=family,
        native_backend=FAMILY_NATIVE_BACKEND[family],
        native_fit_apis=FAMILY_NATIVE_APIS[family],
        fit_semantics=fit_semantics,
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        split_scope_fingerprint=_require_sha256(
            split_scope_fingerprint,
            name="split_scope_fingerprint",
        ),
        data_projection_sha256=_require_sha256(
            data_projection_sha256,
            name="data_projection_sha256",
        ),
        fit_row_fingerprint=fit_row_sha256,
        heldout_row_fingerprint=heldout_row_sha256,
        evidence_payload_sha256=payload_sha256,
        producer_code_sha256=code_sha256,
        configuration_sha256=configuration_sha256,
        native_fit_metadata_sha256=native_fit_metadata_sha256,
        native_execution_record_sha256=execution_record_sha256,
        fit_execution_sha256=execution_sha256,
        model_artifact_sha256=model_artifact_sha256,
        source_artifact_sha256=source_artifact_sha256,
        model_artifact_semantics=model_artifact_semantics,
        _native_fit_metadata_path=_resolved_artifact_path(native_fit_metadata_path),
        _native_execution_record_path=_resolved_artifact_path(native_execution_record_path),
        _model_artifact_path=_resolved_artifact_path(model_artifact_path),
        _source_artifact_path=_resolved_artifact_path(source_artifact_path),
        _construction_authority=_NATIVE_PROOF_CONSTRUCTION_AUTHORITY,
    )


@dataclass(frozen=True)
class AuthenticatedNativeExactInnerScope:
    """All-ten native evidence and proofs for one canonical exact-inner scope."""

    outer_fold: int
    inner_fold: int
    split_scope_fingerprint: str
    data_projection_sha256: str
    fit_row_ids: tuple[int, ...]
    heldout_row_ids: tuple[int, ...]
    catalog_sha256: str
    catalog_artifact_sha256: str
    full_outer_registry: AuthenticatedNativeFullOuterPayloadRegistry = field(repr=False)
    evidence_payload_by_family: Mapping[str, Mapping[str, Any]] = field(repr=False)
    evidence_item_count_by_family: Mapping[str, int]
    fit_proof_by_family: Mapping[str, NativeFamilyFitProof] = field(repr=False)
    _catalog_artifact_path: str = field(repr=False, compare=False)
    _construction_authority: object = field(repr=False, compare=False)
    _payload_json_by_family: Mapping[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self._construction_authority is not _NATIVE_SCOPE_CONSTRUCTION_AUTHORITY:
            raise TypeError("native exact-inner scopes must be built from a catalog artifact")
        if int(self.outer_fold) < 1 or int(self.inner_fold) < 1:
            raise ValueError("native exact-inner fold identifiers must be positive")
        object.__setattr__(self, "outer_fold", int(self.outer_fold))
        object.__setattr__(self, "inner_fold", int(self.inner_fold))
        _require_sha256(
            self.split_scope_fingerprint,
            name="native split_scope_fingerprint",
        )
        _require_sha256(
            self.data_projection_sha256,
            name="native data_projection_sha256",
        )
        _require_sha256(self.catalog_sha256, name="native catalog_sha256")
        _require_sha256(
            self.catalog_artifact_sha256,
            name="native catalog_artifact_sha256",
        )
        object.__setattr__(
            self,
            "_catalog_artifact_path",
            _resolved_artifact_path(self._catalog_artifact_path),
        )
        if not isinstance(
            self.full_outer_registry,
            AuthenticatedNativeFullOuterPayloadRegistry,
        ):
            raise TypeError("native exact-inner scope requires a full-outer catalog registry")
        if self.full_outer_registry.outer_fold != int(self.outer_fold):
            raise ValueError("full-outer registry changed the exact-inner outer fold")
        fit_rows = _ordered_integer_rows(self.fit_row_ids, name="native fit_row_ids")
        heldout_rows = _ordered_integer_rows(
            self.heldout_row_ids,
            name="native heldout_row_ids",
        )
        if set(fit_rows) & set(heldout_rows):
            raise ValueError("native exact-inner fit and held-out rows overlap")
        if set(fit_rows) | set(heldout_rows) != set(self.full_outer_registry.fit_row_ids):
            raise ValueError("exact-inner rows do not partition the registered outer fit rows")
        outer_order = self.full_outer_registry.fit_row_ids
        if fit_rows != tuple(row_id for row_id in outer_order if row_id in set(fit_rows)):
            raise ValueError("exact-inner fit rows changed registered outer row order")
        if heldout_rows != tuple(row_id for row_id in outer_order if row_id in set(heldout_rows)):
            raise ValueError("exact-inner held-out rows changed registered outer row order")
        object.__setattr__(self, "fit_row_ids", fit_rows)
        object.__setattr__(self, "heldout_row_ids", heldout_rows)

        supplied_sets = (
            set(self.evidence_payload_by_family),
            set(self.evidence_item_count_by_family),
            set(self.fit_proof_by_family),
        )
        if any(values != ACTIVE_STAGE1_CONCEPT_FAMILY_SET for values in supplied_sets):
            raise ValueError("native exact-inner scope must cover exactly all ten families")
        payload_json: dict[str, str] = {}
        proofs: dict[str, NativeFamilyFitProof] = {}
        counts: dict[str, int] = {}
        fit_row_sha256 = row_order_fingerprint(fit_rows)
        heldout_row_sha256 = row_order_fingerprint(heldout_rows)
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
            payload = self.evidence_payload_by_family[family]
            if not isinstance(payload, Mapping) or not payload:
                raise ValueError(f"{family} native evidence payload is empty")
            if (
                payload.get("schema_version") != NATIVE_FAMILY_PAYLOAD_VERSION
                or payload.get("family") != family
                or not isinstance(payload.get("architecture_evidence"), list)
            ):
                raise ValueError(f"{family} native evidence payload has an invalid envelope")
            payload_json[family] = _canonical_json(copy.deepcopy(dict(payload)))
            count = int(self.evidence_item_count_by_family[family])
            if count < 1 or count != len(payload["architecture_evidence"]):
                raise ValueError(f"{family} native evidence item count is invalid")
            counts[family] = count
            proof = self.fit_proof_by_family[family]
            if not isinstance(proof, NativeFamilyFitProof) or proof.family != family:
                raise TypeError(f"{family} native fit proof is absent or mismatched")
            expected_binding = {
                "outer_fold": int(self.outer_fold),
                "inner_fold": int(self.inner_fold),
                "split_scope_fingerprint": self.split_scope_fingerprint,
                "data_projection_sha256": self.data_projection_sha256,
                "fit_row_fingerprint": fit_row_sha256,
                "heldout_row_fingerprint": heldout_row_sha256,
                "evidence_payload_sha256": hashlib.sha256(
                    payload_json[family].encode("utf-8")
                ).hexdigest(),
            }
            observed_binding = {key: getattr(proof, key) for key in expected_binding}
            if observed_binding != expected_binding:
                raise ValueError(f"{family} native fit proof is bound to another scope or payload")
            if (
                expected_binding["evidence_payload_sha256"]
                == self.full_outer_registry.payload_sha256_by_family[family]
            ):
                raise ValueError(
                    f"{family} exact-inner payload is identical to the authenticated "
                    "full-outer payload"
                )
            proofs[family] = proof
        object.__setattr__(
            self,
            "_payload_json_by_family",
            MappingProxyType(payload_json),
        )
        object.__setattr__(
            self,
            "evidence_payload_by_family",
            MappingProxyType({}),
        )
        object.__setattr__(
            self,
            "evidence_item_count_by_family",
            MappingProxyType(counts),
        )
        object.__setattr__(self, "fit_proof_by_family", MappingProxyType(proofs))
        self.verify_artifact_bytes()

    def payload(self, family: str) -> dict[str, Any]:
        try:
            encoded = self._payload_json_by_family[family]
        except KeyError as exc:
            raise ValueError("requested inactive or absent native family") from exc
        return json.loads(encoded)

    @property
    def full_outer_payload_sha256_by_family(self) -> Mapping[str, str]:
        return self.full_outer_registry.payload_sha256_by_family

    def verify_artifact_bytes(self) -> None:
        self.verify_catalog_artifact_bytes()
        for proof in self.fit_proof_by_family.values():
            proof.verify_artifact_bytes()

    def verify_catalog_artifact_bytes(self) -> None:
        if native_artifact_sha256(self._catalog_artifact_path) != self.catalog_artifact_sha256:
            raise RuntimeError("authenticated exact-inner catalog artifact changed after binding")
        self.full_outer_registry.verify_artifact_bytes()

    def validate_request(self, request: ExactInnerStage1FamilyRequest) -> None:
        if not isinstance(request, ExactInnerStage1FamilyRequest):
            raise TypeError("native family adapter requires ExactInnerStage1FamilyRequest")
        if (
            request.outer_fold != self.outer_fold
            or request.inner_fold != self.inner_fold
            or request.split_scope_fingerprint != self.split_scope_fingerprint
            or request.data_projection_sha256 != self.data_projection_sha256
            or tuple(row.row_id for row in request.fit_rows) != self.fit_row_ids
            or tuple(row.row_id for row in request.heldout_rows) != self.heldout_row_ids
        ):
            raise ValueError("native family evidence is bound to a different exact-inner scope")


class NativeExactInnerStage1FamilyProducer:
    """Protocol adapter for one family in one authenticated native scope."""

    def __init__(
        self,
        *,
        family: str,
        scope: AuthenticatedNativeExactInnerScope,
        expected_configuration_sha256: str | None = None,
        expected_code_sha256: str | None = None,
        producer_version: str = NATIVE_EXACT_INNER_ADAPTER_VERSION,
    ) -> None:
        if family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("family is not an active Stage 1 architecture")
        self._family = family
        self._scope = scope
        proof = scope.fit_proof_by_family[family]
        scope.verify_catalog_artifact_bytes()
        proof.verify_artifact_bytes()
        code_identity = native_family_code_identity(family)
        code_sha256 = str(code_identity["content_sha256"])
        if code_sha256 != proof.producer_code_sha256:
            raise RuntimeError(f"{family} proof was emitted by different native code")
        if (
            expected_code_sha256 is not None
            and _require_sha256(
                expected_code_sha256,
                name="expected adapter code_sha256",
            )
            != code_sha256
        ):
            raise RuntimeError(f"{family} native adapter code identity changed")
        if (
            expected_configuration_sha256 is not None
            and _require_sha256(
                expected_configuration_sha256,
                name=f"expected {family} configuration_sha256",
            )
            != proof.configuration_sha256
        ):
            raise RuntimeError(f"{family} native fit configuration differs from deployment")
        identity = {
            "schema_version": EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
            "family": family,
            "producer_name": f"native_exact_inner_adapter__{FAMILY_NATIVE_BACKEND[family]}",
            "producer_version": str(producer_version),
            "code_sha256": code_sha256,
            "configuration_sha256": proof.configuration_sha256,
            "native_backend": FAMILY_NATIVE_BACKEND[family],
            "native_fit_apis": list(FAMILY_NATIVE_APIS[family]),
        }
        self._code_sha256 = code_sha256
        self._configuration_sha256 = proof.configuration_sha256
        self._identity_json = _canonical_json(identity)

    def identity(self) -> Mapping[str, Any]:
        current = native_family_code_identity(self._family)["content_sha256"]
        if current != self._code_sha256:
            raise RuntimeError(f"{self._family} native adapter code changed after binding")
        if (
            self._scope.fit_proof_by_family[self._family].configuration_sha256
            != self._configuration_sha256
        ):
            raise RuntimeError(f"{self._family} native configuration proof changed after binding")
        return json.loads(self._identity_json)

    def produce(
        self,
        request: ExactInnerStage1FamilyRequest,
    ) -> ExactInnerFamilyEvidenceDraft:
        if request.family != self._family:
            raise ValueError("native family adapter received another architecture request")
        self._scope.verify_catalog_artifact_bytes()
        self._scope.fit_proof_by_family[self._family].verify_artifact_bytes()
        self._scope.validate_request(request)
        proof = self._scope.fit_proof_by_family[self._family]
        audit: dict[str, Any] = {
            "schema_version": EXACT_INNER_FIT_AUDIT_VERSION,
            "family": self._family,
            "scope": "inner_train",
            "input_binding_sha256": request.binding_sha256,
            "split_scope_fingerprint": request.split_scope_fingerprint,
            "fit_semantics": proof.fit_semantics,
            "heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
            "secrets_accessed": False,
            "fit_execution_sha256": proof.fit_execution_sha256,
            "producer_code_sha256": proof.producer_code_sha256,
            "configuration_sha256": proof.configuration_sha256,
            "native_fit_metadata_sha256": proof.native_fit_metadata_sha256,
            "native_execution_record_sha256": proof.native_execution_record_sha256,
            "model_artifact_sha256": proof.model_artifact_sha256,
            "native_backend": proof.native_backend,
            "native_fit_apis": list(proof.native_fit_apis),
            "native_source_artifact_sha256": proof.source_artifact_sha256,
            "model_artifact_semantics": proof.model_artifact_semantics,
            "data_projection_sha256": proof.data_projection_sha256,
            "fit_row_fingerprint": proof.fit_row_fingerprint,
            "heldout_row_fingerprint": proof.heldout_row_fingerprint,
            "native_evidence_payload_sha256": proof.evidence_payload_sha256,
            "native_catalog_sha256": self._scope.catalog_sha256,
            "native_catalog_artifact_sha256": self._scope.catalog_artifact_sha256,
            "native_full_outer_payload_registry_sha256": (
                self._scope.full_outer_registry.content_sha256
            ),
        }
        if proof.fit_semantics == EXACT_SCOPE_CACHE_REPLAY:
            audit.update(
                {
                    "cache_source_scope_fingerprint": request.split_scope_fingerprint,
                    "cache_source_artifact_sha256": proof.source_artifact_sha256,
                }
            )
        return ExactInnerFamilyEvidenceDraft(
            evidence_payload=self._scope.payload(self._family),
            evidence_item_count=self._scope.evidence_item_count_by_family[self._family],
            input_binding_sha256=request.binding_sha256,
            fit_semantics=proof.fit_semantics,
            fit_audit=audit,
        )


def family_producers_for_native_scope(
    scope: AuthenticatedNativeExactInnerScope,
    *,
    expected_configuration_sha256_by_family: Mapping[str, str] | None = None,
    expected_code_sha256_by_family: Mapping[str, str] | None = None,
) -> dict[str, NativeExactInnerStage1FamilyProducer]:
    """Construct the complete, canonically ordered exact-inner producer map."""

    if not isinstance(scope, AuthenticatedNativeExactInnerScope):
        raise TypeError("scope must be AuthenticatedNativeExactInnerScope")
    if expected_configuration_sha256_by_family is not None and (
        set(expected_configuration_sha256_by_family) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET
    ):
        raise ValueError("expected configuration hashes must cover exactly all ten families")
    if expected_code_sha256_by_family is not None and (
        set(expected_code_sha256_by_family) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET
    ):
        raise ValueError("expected producer code hashes must cover exactly all ten families")
    return {
        family: NativeExactInnerStage1FamilyProducer(
            family=family,
            scope=scope,
            expected_configuration_sha256=(
                None
                if expected_configuration_sha256_by_family is None
                else expected_configuration_sha256_by_family[family]
            ),
            expected_code_sha256=(
                None
                if expected_code_sha256_by_family is None
                else expected_code_sha256_by_family[family]
            ),
        )
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }


def native_scope_from_catalog(
    *,
    catalog: RoleNeutralEvidenceCatalog,
    catalog_artifact_path: Path | str,
    full_outer_registry: AuthenticatedNativeFullOuterPayloadRegistry,
    outer_fold: int,
    inner_fold: int,
    split_scope_fingerprint: str,
    data_projection_sha256: str,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    fit_proof_by_family: Mapping[str, NativeFamilyFitProof],
) -> AuthenticatedNativeExactInnerScope:
    """Build an all-ten adapter scope from genuine native catalog evidence."""

    if not isinstance(catalog, RoleNeutralEvidenceCatalog):
        raise TypeError("catalog must be RoleNeutralEvidenceCatalog")
    validate_role_neutral_catalog(catalog)
    if not isinstance(
        full_outer_registry,
        AuthenticatedNativeFullOuterPayloadRegistry,
    ):
        raise TypeError("full_outer_registry must be catalog-authenticated")
    fit_rows = _ordered_integer_rows(fit_row_ids, name="native fit_row_ids")
    heldout_rows = _ordered_integer_rows(
        heldout_row_ids,
        name="native heldout_row_ids",
    )
    if set(fit_rows) & set(heldout_rows):
        raise ValueError("native exact-inner fit and held-out rows overlap")
    if (
        catalog.scope != "inner_train"
        or catalog.outer_fold != int(outer_fold)
        or catalog.inner_fold != int(inner_fold)
    ):
        raise ValueError("catalog fold provenance differs from the exact-inner scope")
    expected_catalog_split = FoldEvidenceProvenance(
        outer_fold=int(outer_fold),
        train_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        scope="inner_train",
        inner_fold=int(inner_fold),
        artifact_id="native-exact-inner-scope-binding",
    ).split_fingerprint
    if catalog.split_fingerprint != expected_catalog_split:
        raise ValueError("catalog row provenance differs from the exact-inner scope")
    catalog_artifact_sha256 = _verified_catalog_artifact_sha256(
        catalog,
        catalog_artifact_path,
    )
    payloads: dict[str, Mapping[str, Any]] = {}
    counts: dict[str, int] = {}
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        payload, count = family_payload_from_catalog(catalog, family=family)
        payloads[family] = payload
        counts[family] = count
    return AuthenticatedNativeExactInnerScope(
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        split_scope_fingerprint=split_scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        catalog_sha256=catalog.catalog_sha256,
        catalog_artifact_sha256=catalog_artifact_sha256,
        full_outer_registry=full_outer_registry,
        evidence_payload_by_family=payloads,
        evidence_item_count_by_family=counts,
        fit_proof_by_family=fit_proof_by_family,
        _catalog_artifact_path=_resolved_artifact_path(catalog_artifact_path),
        _construction_authority=_NATIVE_SCOPE_CONSTRUCTION_AUTHORITY,
    )


__all__ = [
    "FAMILY_NATIVE_BACKEND",
    "FAMILY_NATIVE_APIS",
    "NATIVE_EXACT_INNER_ADAPTER_VERSION",
    "NATIVE_FAMILY_FIT_PROOF_VERSION",
    "NATIVE_FAMILY_EXECUTION_RECORD_VERSION",
    "NATIVE_FAMILY_PAYLOAD_VERSION",
    "NATIVE_FULL_OUTER_PAYLOAD_REGISTRY_VERSION",
    "NATIVE_SCOPE_RESULT_VERSION",
    "AuthenticatedNativeExactInnerScope",
    "AuthenticatedNativeFullOuterPayloadRegistry",
    "NativeExactInnerStage1FamilyProducer",
    "NativeFamilyFitProof",
    "bind_native_family_fit_proof",
    "family_payload_from_catalog",
    "family_producers_for_native_scope",
    "native_artifact_sha256",
    "native_family_code_identity",
    "native_family_configuration_sha256",
    "native_family_execution_record",
    "native_full_outer_payload_registry_from_catalog",
    "native_scope_from_catalog",
]
