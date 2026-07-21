"""Fail-closed production builder for all-evidence Stage 1 bundles.

The historical experiment scripts do not form a safe ingestion interface for a
new cohort.  In particular, their public integrated route currently selects the
TF-IDF-only implementation, and the legacy helper's nominal inner handoffs are
derived by copying full-outer evidence.  This module deliberately bypasses both
behaviours:

* one canonical outer/inner split registry is created before any model fit;
* only unit ID, text, treatment, and observed outcome columns are projected;
* every legacy full-outer and exact-inner evidence scope is independently fit;
* every neural-query scope is independently fit from the frozen cache;
* TF-IDF consumes the same registry;
* the exact-inner producer contract exposes held-out text but never held-out
  treatment or outcome labels;
* all ten concept-bearing families must be non-empty in every scope; and
* a content-addressed component is reusable only after its terminal manifest
  and every registered file hash validate.

Hashes are internal integrity and resume controls.  There is intentionally no
human digest-approval argument or two-phase approval ceremony for this local
Stage 1 build command.

Native
``ExactInnerStage1FamilyProducer`` adapters now describe all ten existing
architectures; the native BoW component registers genuine nuisance and R-loss
proofs, the native HTR component registers genuine nested neural nuisance and
effect-model proofs, the paired component registers both genuine BoW and HTR
matched-uplift fits, the TF-IDF component registers genuine topic and
orphan-n-gram proofs, and the neural-query component registers fitted query
arrays plus exact heldout moments.  The embedding component now also registers
whole-cohort directions, actual clustered KMeans/SVD state, and the exhaustive
semantic-retrieval TF-IDF projection.  All ten exact-inner family registrations
are present.  The wrapper now also emits the typed cumulative-spent all-ten
root graph and can construct or revalidate an arbitrary cohort's embedding
cache.  Candidate bundle construction is enabled; final production hierarchy
certification remains false until a genuine full-cohort one-shot run succeeds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from ..config import AppliedInferenceConfig, ExperimentConfig
from ..models.hierarchical_transformer_extractor import split_text_into_word_chunks
from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)
from .all_evidence_fusion import (
    FoldEvidenceInput,
    FoldEvidenceProvenance,
    LEGACY_ALL_SOURCE,
    NEURAL_QUERY_MOMENTS,
    NEURAL_QUERY_SOURCE,
    TFIDF_TOPIC_SOURCE,
)
from .all_evidence_fusion_runner import (
    _sanitize_retained_legacy_digest,
    load_legacy_full_outer_evidence,
    load_outer_splits_from_primary_predictions,
    load_resealed_tfidf_handoff,
)
from .bow_native_proof_capture import (
    BOW_NATIVE_CAPTURE_SCHEMA,
    NativeBoWProofCaptureSink,
    validate_bow_native_capture,
)
from .embedding_native_proof_capture import (
    EMBEDDING_NATIVE_CAPTURE_SCHEMA,
    SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA,
    NativeEmbeddingProofCaptureSink,
    validate_embedding_native_capture,
)
from .htr_native_proof_capture import (
    HTR_NATIVE_CAPTURE_SCHEMA,
    NativeHTRProofCaptureSink,
    validate_htr_native_capture,
)
from .matched_pair_native_proof_capture import (
    MATCHED_PAIR_NATIVE_CAPTURE_SCHEMA,
    NativeMatchedPairProofCaptureSink,
    validate_matched_pair_native_capture,
)
from .multi_model_agentic_forest import (
    _agentic_discovery_handoff_row,
    _normalize_text,
    _normalize_texts,
)
from .multi_model_forest_stage1 import MultiModelForestStage1Runner
from .neural_query_agentic_forest import NeuralQueryAgenticForestConfig
from .neural_query_context_backend import (
    NEURAL_QUERY_OWNED_SNAPSHOT_SCHEMA,
    ContextFitNeuralQueryService,
    NeuralQueryContextBackend,
    validate_owned_discovery_snapshot,
)
from .lossless_stage1_evidence_catalog import (
    SEMANTIC_RETRIEVAL_DERIVATION,
    RoleNeutralEvidenceCatalog,
    assemble_cumulative_spent_role_neutral_catalog,
    build_role_neutral_evidence_catalog,
)
from .review_spent_evidence_provider import (
    BoundSpentFrozenChunkEmbeddingProvider,
    SpentOnlyFrozenChunkEmbeddingCache,
    _FrozenCacheEmbeddingEvidenceGenerator,
    _embedding_concepts_only,
    _htr_concepts_only,
    _sanitize_digest_terms,
)
from .stage1_exact_inner_evidence import (
    EXACT_INNER_REFIT,
    CanonicalInnerSplit,
    CanonicalStage1SplitRegistry,
    Stage1FitRow,
    Stage1HeldoutRow,
    exact_inner_data_projection_sha256,
    produce_exact_inner_stage1_evidence_bundle,
    row_order_fingerprint,
)
from .stage1_exact_inner_family_adapters import (
    FAMILY_NATIVE_APIS,
    FAMILY_NATIVE_BACKEND,
    NativeFamilyFitProof,
    bind_native_family_fit_proof,
    family_producers_for_native_scope,
    family_payload_from_catalog,
    native_artifact_sha256,
    native_family_execution_record,
    native_full_outer_payload_registry_from_catalog,
    native_scope_from_catalog,
)
from .stage1_cumulative_spent_evidence import (
    CUMULATIVE_SPENT_REFIT,
    CumulativeSpentStage1FamilyRequest,
    cumulative_spent_data_projection_sha256,
    produce_cumulative_spent_stage1_evidence_bundle,
)
from .stage1_cumulative_spent_native_adapters import (
    CUMULATIVE_SPENT_NATIVE_EXECUTION_RECORD_SCHEMA,
    CUMULATIVE_SPENT_REPLAY_CANARY_SEMANTICS,
    CumulativeSpentReplayCanary,
    bind_cumulative_spent_native_family_producer,
    cumulative_spent_native_execution_record,
    cumulative_spent_native_family_identity,
)
from .stage1_cumulative_spent_embedding_adapters import (
    CUMULATIVE_SPENT_EMBEDDING_FAMILIES,
    bind_cumulative_spent_embedding_family_producer,
    bind_persisted_cumulative_spent_embedding_producers,
    cumulative_spent_embedding_family_identity,
    emit_cumulative_spent_embedding_capture,
    validate_cumulative_spent_embedding_artifacts,
)
from .stage1_cumulative_spent_remaining_adapters import (
    REMAINING_CUMULATIVE_FAMILIES,
    TFIDF_CUMULATIVE_FAMILIES,
    bind_cumulative_spent_remaining_family_producer,
    bind_persisted_cumulative_spent_neural_query_producer,
    bind_persisted_cumulative_spent_tfidf_producers,
    emit_cumulative_spent_neural_query_capture,
    emit_cumulative_spent_tfidf_capture,
)
from .production_stage1_hierarchy_contract import (
    current_production_stage1_hierarchy_contract_identity,
    production_stage1_hierarchy_architecture_bindings,
    validate_production_stage1_hierarchy_request_bindings,
)
from .production_embedding_cache_builder import (
    build_production_embedding_cache,
    validate_published_production_embedding_cache,
)
from .stage1_upstream_gate_backend import (
    PrivateHTRModelTreeSnapshot,
    _directory_tree_sha256,
)
from .tfidf_topic_agentic_forest import validate_tfidf_topic_stage2_handoff
from .tfidf_topic_discovery import row_set_fingerprint
from .tfidf_topic_split_registry import (
    TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
    load_tfidf_topic_split_registry,
)
from .tfidf_topic_stage1 import (
    _nested_calibration_plan,
    make_joint_treatment_outcome_splits,
    run_tfidf_topic_stage1,
    tfidf_topic_stage1_cache_is_valid,
)

STAGE1_BUNDLE_REQUEST_SCHEMA = "production_all_evidence_stage1_request_v4"
STAGE1_COMPONENT_MANIFEST_SCHEMA = "production_all_evidence_stage1_component_v2"
STAGE1_BUNDLE_MANIFEST_SCHEMA = "production_all_evidence_stage1_bundle_v3"
STAGE1_SCOPE_INDEX_SCHEMA = "production_all_evidence_stage1_scope_index_v2"
STAGE1_QUERY_ARTIFACT_SCHEMA = "production_neural_query_evidence_scope_v1"
STAGE1_QUERY_MOMENT_ARTIFACT_SCHEMA = "production_neural_query_heldout_moments_v1"
STAGE1_QUERY_NATIVE_FIT_METADATA_SCHEMA = "production_neural_query_native_fit_metadata_v1"
STAGE1_BOW_NATIVE_FIT_METADATA_SCHEMA = "production_bow_native_fit_metadata_v1"
STAGE1_EMBEDDING_NATIVE_FIT_METADATA_SCHEMA = "production_embedding_native_fit_metadata_v1"
STAGE1_HTR_NATIVE_FIT_METADATA_SCHEMA = "production_htr_native_fit_metadata_v1"
STAGE1_MATCHED_PAIR_NATIVE_FIT_METADATA_SCHEMA = "production_matched_pair_native_fit_metadata_v1"
STAGE1_EXACT_INNER_ADAPTER_GATE_SCHEMA = "production_stage1_exact_inner_adapter_gate_v2"
STAGE1_RAW_EVIDENCE_SIDECAR_SCHEMA = "production_stage1_raw_evidence_sidecar_v1"
STAGE1_MATCHED_PAIR_PROOF_SCHEMA = "production_stage1_matched_pair_subproducer_proof_v1"
STAGE1_BEHAVIOR_IDENTITY_SCHEMA = "production_stage1_behavior_identity_v1"
STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA = (
    "production_stage1_native_family_proof_registration_v1"
)
STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA = "production_stage1_native_family_proof_index_v1"
STAGE1_CUMULATIVE_LEGACY_NATIVE_SCOPE_SCHEMA = "production_stage1_cumulative_legacy_native_scope_v1"
STAGE1_CUMULATIVE_LEGACY_NATIVE_INDEX_SCHEMA = "production_stage1_cumulative_legacy_native_index_v1"
STAGE1_CUMULATIVE_EMBEDDING_NATIVE_SCOPE_SCHEMA = (
    "production_stage1_cumulative_embedding_native_scope_v1"
)
STAGE1_CUMULATIVE_EMBEDDING_NATIVE_INDEX_SCHEMA = (
    "production_stage1_cumulative_embedding_native_index_v1"
)
STAGE1_CUMULATIVE_REMAINING_NATIVE_SCOPE_SCHEMA = (
    "production_stage1_cumulative_remaining_native_scope_v1"
)
STAGE1_CUMULATIVE_TFIDF_NATIVE_INDEX_SCHEMA = "production_stage1_cumulative_tfidf_native_index_v1"
STAGE1_CUMULATIVE_QUERY_NATIVE_INDEX_SCHEMA = (
    "production_stage1_cumulative_neural_query_native_index_v1"
)
STAGE1_CUMULATIVE_ALL_TEN_ROOT_INDEX_SCHEMA = "production_stage1_cumulative_all_ten_root_index_v1"
STAGE1_EXACT_INNER_ROOT_INDEX_SCHEMA = "production_stage1_exact_inner_evidence_index_v2"
STAGE1_TFIDF_RESUME_POLICY = "sealed_complete_component_only_no_partial_checkpoint_reuse_v1"
STAGE1_HTR_INPUT_NONTRUNCATION_AUDIT_SCHEMA = "production_stage1_htr_input_nontruncation_audit_v1"

# These component-local tuples enumerate all ten families whose native producer
# persists every artifact required by ``bind_native_family_fit_proof``.  Keeping
# the grouping explicit prevents a catalog projection or output-only diagnostic
# from being advertised as a registered native producer.
PRODUCTION_TFIDF_REGISTERED_NATIVE_FAMILY_ADAPTERS = (
    TFIDF_TOPICS,
    TFIDF_ORPHAN_NGRAMS,
)
PRODUCTION_QUERY_REGISTERED_NATIVE_FAMILY_ADAPTERS = (NEURAL_QUERY_MOMENTS,)
PRODUCTION_CUMULATIVE_TFIDF_NATIVE_FAMILY_ADAPTERS = (
    TFIDF_TOPICS,
    TFIDF_ORPHAN_NGRAMS,
)
PRODUCTION_CUMULATIVE_QUERY_NATIVE_FAMILY_ADAPTERS = (NEURAL_QUERY_MOMENTS,)
PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS = (BOW_NUISANCE, BOW_R_LOSS)
PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS = (HTR_NEURAL,)
PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS = (MATCHED_PAIR_UPLIFT,)
PRODUCTION_CUMULATIVE_LEGACY_NATIVE_FAMILY_ADAPTERS = (
    BOW_NUISANCE,
    BOW_R_LOSS,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
)
PRODUCTION_CUMULATIVE_EMBEDDING_NATIVE_FAMILY_ADAPTERS = (
    EMBEDDING_WHOLE_COHORT,
    EMBEDDING_CLUSTERED,
    TFIDF_SEMANTIC_RETRIEVAL,
)
PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS = (
    EMBEDDING_WHOLE_COHORT,
    EMBEDDING_CLUSTERED,
    TFIDF_SEMANTIC_RETRIEVAL,
)
PRODUCTION_REGISTERED_NATIVE_FAMILY_ADAPTERS = (
    *PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    *PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    *PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    *PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    *PRODUCTION_TFIDF_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    *PRODUCTION_QUERY_REGISTERED_NATIVE_FAMILY_ADAPTERS,
)

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DEVICE = re.compile(r"^(?:cpu|cuda:[0-9]+)$")
_SECRET_FIELD = re.compile(
    r"(?:api[_-]?key|access[_-]?token|secret|password|credential)", re.IGNORECASE
)
_ORACLE_COLUMN = re.compile(
    r"(?:^|_)(?:oracle|true|ground_truth)(?:_|$)|(?:oracle|ground_truth)",
    re.IGNORECASE,
)
_HTR_INPUT_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "row_count",
        "normalized_text_projection_sha256",
        "chunk_size_words",
        "chunk_overlap_words",
        "max_chunks",
        "configured_max_chunk_length",
        "model_max_sequence_length",
        "effective_max_chunk_length",
        "total_chunks",
        "uncapped_total_chunks",
        "ordered_chunk_counts_sha256",
        "ordered_token_counts_sha256",
        "max_observed_token_count",
        "chunk_cap_nonbinding",
        "all_chunks_within_effective_max_length",
        "semantic_truncation_allowed",
        "tokenizer_truncation_allowed",
        "tokenizer_class",
        "tokenizer_vocab_size",
        "htr_model_tree_sha256",
        "applies_to_families",
        "content_sha256",
    }
)
_HTR_TOKENIZATION_AUDIT_BATCH_SIZE = 256


def _canonical_json(value: Any) -> str:
    result = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        result.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("production Stage 1 values must be valid UTF-8") from exc
    return result


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_json_object_reject_duplicates(path: Path, *, field_name: str) -> dict[str, Any]:
    """Read one non-aliased JSON object without last-key-wins ambiguity."""

    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError(f"{field_name} must be one regular JSON file")
    try:
        value = json.loads(
            artifact.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be one JSON object")
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _float_hex_sha256(values: Sequence[float]) -> str:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("label vectors must be finite and one-dimensional")
    return _sha256_json([float(value).hex() for value in vector])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_stable_sha256(path: Path) -> tuple[str, tuple[int, int, int, int, int]]:
    before_stat = path.stat()
    before = (
        int(before_stat.st_dev),
        int(before_stat.st_ino),
        int(before_stat.st_size),
        int(before_stat.st_mtime_ns),
        int(before_stat.st_ctime_ns),
    )
    digest = _sha256_file(path)
    after_stat = path.stat()
    after = (
        int(after_stat.st_dev),
        int(after_stat.st_ino),
        int(after_stat.st_size),
        int(after_stat.st_mtime_ns),
        int(after_stat.st_ctime_ns),
    )
    if before != after:
        raise RuntimeError(f"input changed while it was being authenticated: {path}")
    return digest, after


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as h:
        temporary = Path(h.name)
        h.write(payload)
        h.flush()
        os.fsync(h.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )


def _atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_json(path: Path, payload: Any) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to mutate immutable file: {path}")
        return
    _atomic_write_bytes(path, encoded)


def _sanitize_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            result[key] = (
                "<redacted-not-used-by-stage1>"
                if _SECRET_FIELD.search(key)
                else _sanitize_secrets(child)
            )
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize_secrets(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_sanitize_secrets(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _catalog_ready_legacy_digest(
    *,
    importance: Mapping[str, Any],
    embedding_evidence: Mapping[str, Any],
    htr_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an untruncated concept projection without a prompt compactor.

    The architecture producers decide which concepts they emit.  This boundary
    preserves every emitted BoW row, embedding witness, and HTR phrase.  Raw
    retrieval/attention records are authenticated separately as non-prompt
    sidecars by :meth:`_run_legacy_component`; they are never substituted for
    this privacy-safe lexical projection.
    """

    def bow_groups(
        source: Mapping[str, Any], *, role: str, source_prefix: str = ""
    ) -> list[dict[str, Any]]:
        if not isinstance(source, Mapping):
            return []
        role_keys = (
            (
                "confounder_overlap",
                "treatment_positive",
                "treatment_negative",
                "outcome_positive",
                "outcome_negative",
            )
            if role == "confounder"
            else (
                "pseudo_target_positive",
                "pseudo_target_negative",
                "uplift_pair_features",
                "uplift_delta_logit_positive",
                "uplift_delta_logit_negative",
                "ridge_delta_probability_positive",
                "ridge_delta_probability_negative",
            )
        )
        descriptions = {
            "confounder_overlap": "Terms predictive of both treatment and outcome nuisance models.",
            "treatment_positive": "Terms positively associated with treatment assignment.",
            "treatment_negative": "Terms negatively associated with treatment assignment.",
            "outcome_positive": "Terms positively associated with outcome risk.",
            "outcome_negative": "Terms negatively associated with outcome risk.",
            "pseudo_target_positive": "Terms positively associated with the R-stage pseudo-target.",
            "pseudo_target_negative": "Terms negatively associated with the R-stage pseudo-target.",
            "uplift_pair_features": "Matched-pair uplift terms from paired treated/control patients.",
            "uplift_delta_logit_positive": "Terms increasing matched-pair treated outcome delta logit.",
            "uplift_delta_logit_negative": "Terms decreasing matched-pair treated outcome delta logit.",
            "ridge_delta_probability_positive": "Terms increasing matched-pair treated outcome delta probability.",
            "ridge_delta_probability_negative": "Terms decreasing matched-pair treated outcome delta probability.",
        }
        output: list[dict[str, Any]] = []
        for raw_view in source.get("views") or ():
            if not isinstance(raw_view, Mapping):
                continue
            view_name = str(raw_view.get("view_name") or raw_view.get("view_index") or "view")
            view_config = raw_view.get("view_config")
            bow_model = (
                (view_config.get("bow_model") if isinstance(view_config, Mapping) else None)
                or raw_view.get("bow_model")
                or "linear"
            )
            for evidence_type in role_keys:
                raw_rows = raw_view.get(evidence_type) or ()
                if not isinstance(raw_rows, (list, tuple)):
                    raise ValueError(f"legacy {view_name}.{evidence_type} evidence must be a list")
                rows = [copy.deepcopy(dict(row)) for row in raw_rows if isinstance(row, Mapping)]
                if not rows:
                    continue
                output.append(
                    {
                        "source": f"{source_prefix}{view_name}.{evidence_type}",
                        "view_name": view_name,
                        "bow_model": str(bow_model),
                        "evidence_type": evidence_type,
                        "meaning": descriptions[evidence_type],
                        "rows": rows,
                    }
                )
        if role == "effect_modifier":
            for nested_key in ("ensemble_r", "matched_pair_uplift"):
                nested = source.get(nested_key)
                if isinstance(nested, Mapping):
                    output.extend(
                        bow_groups(
                            nested,
                            role=role,
                            source_prefix=f"{source_prefix}{nested_key}.",
                        )
                    )
        return output

    def embedding_groups(role: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for raw in embedding_evidence.get("contrasts") or ():
            if not isinstance(raw, Mapping):
                continue
            role_text = " ".join(
                str(raw.get(key) or "").casefold()
                for key in (
                    "name",
                    "role_hint",
                    "contrast_family",
                    "direction_source",
                    "score_formula",
                )
            )
            is_effect = any(
                marker in role_text
                for marker in (
                    "effect",
                    "modifier",
                    "interaction",
                    "pseudo",
                    "r-score",
                    "r_score",
                    "orthogonal",
                    "residual",
                    "uplift",
                )
            )
            if (role == "effect_modifier") != is_effect:
                continue
            item = {
                key: copy.deepcopy(raw[key])
                for key in (
                    "name",
                    "role_hint",
                    "contrast_family",
                    "direction_source",
                    "cluster_component_index",
                    "concept_probe_scores",
                )
                if raw.get(key) is not None
            }
            if item.get("concept_probe_scores"):
                item["concept_derivation"] = SEMANTIC_RETRIEVAL_DERIVATION
                item["raw_retrieved_excerpts_retained"] = False
                output.append(item)
        return output

    def htr_groups(role: str) -> list[dict[str, Any]]:
        stages = ("nuisance",) if role == "confounder" else ("effect", "pair_uplift")
        descriptions = {
            "nuisance": "HTR nuisance-model attention for treatment assignment and baseline outcome risk.",
            "effect": "HTR R-stage/effect-model attention for treatment-effect heterogeneity.",
            "pair_uplift": "HTR matched-pair uplift attention for paired treated/control outcome delta prediction.",
        }
        output: list[dict[str, Any]] = []
        for stage_name in stages:
            stage = htr_evidence.get(stage_name)
            if not isinstance(stage, Mapping):
                continue
            rows = stage.get("attention") or ()
            if not isinstance(rows, (list, tuple)):
                raise ValueError(f"legacy HTR {stage_name} attention must be a list")
            copied = [copy.deepcopy(dict(row)) for row in rows if isinstance(row, Mapping)]
            if copied:
                output.append(
                    {
                        "stage": stage_name,
                        "meaning": descriptions[stage_name],
                        "metrics": {},
                        "rows": copied,
                    }
                )
        return output

    digest = {
        "confounders": {
            "role": "confounder",
            "role_definition": (
                "Variables predictive of treatment assignment and baseline outcome risk."
            ),
            "bow_blurbs": bow_groups(importance, role="confounder"),
            "embedding_chunks": embedding_groups("confounder"),
            "htr_blurbs": htr_groups("confounder"),
        },
        "effect_modifiers": {
            "role": "effect_modifier",
            "role_definition": (
                "Variables predictive of treatment-effect heterogeneity or matched-pair uplift."
            ),
            "bow_blurbs": bow_groups(importance, role="effect_modifier"),
            "embedding_chunks": embedding_groups("effect_modifier"),
            "htr_blurbs": htr_groups("effect_modifier"),
        },
    }
    digest = _sanitize_digest_terms(digest)
    digest, _dropped = _sanitize_retained_legacy_digest(digest)
    if not isinstance(digest, dict):  # pragma: no cover - helper always returns a mapping
        raise RuntimeError("legacy concept projection did not return an evidence digest")
    for role in ("confounders", "effect_modifiers"):
        section = digest.get(role)
        if not isinstance(section, dict):
            continue
        for contrast in section.get("embedding_chunks") or ():
            if not isinstance(contrast, dict):
                raise ValueError("projected embedding contrast must be an object")
            probes = contrast.get("concept_probe_scores")
            if not isinstance(probes, list) or not probes:
                raise ValueError("projected embedding contrast has no lexical witnesses")
            contrast["concept_derivation"] = SEMANTIC_RETRIEVAL_DERIVATION
            contrast["raw_retrieved_excerpts_retained"] = False
            forbidden = {
                "positive_aligned_chunks",
                "negative_aligned_chunks",
                "positive_external_chunks",
                "negative_external_chunks",
            }
            if any(contrast.get(key) for key in forbidden):
                raise RuntimeError(
                    "row-level embedding retrieval text crossed the catalog boundary"
                )
    if "prompt_compaction" in _canonical_json(digest):
        raise RuntimeError("prompt-compaction metadata entered the production concept catalog")
    return digest


def _catalog_ready_tfidf_discovery(discovery: Mapping[str, Any]) -> dict[str, Any]:
    """Project native orphan clusters to lossless fit-side concept evidence.

    Native score rows carry calibration statistics that are useful for the
    machine audit but are not concept-bearing prompt evidence.  The catalog
    contract instead accepts the complete cluster vocabulary and its fit-side
    support/rank fields.  This mechanical projection keeps every main term and
    nested alias while leaving the unmodified score artifact available for
    authenticated drill-back and proof binding.
    """

    if not isinstance(discovery, Mapping):
        raise TypeError("TF-IDF discovery must be a mapping")
    projected = copy.deepcopy(dict(discovery))
    orphan = projected.get("effect_orphan_ngram_branch")
    if not isinstance(orphan, Mapping):
        for key in ("topic_score_tests", "topic_score_selection", "score_tests"):
            nested = projected.get(key)
            if isinstance(nested, Mapping) and isinstance(
                nested.get("effect_orphan_ngram_branch"), Mapping
            ):
                orphan = nested["effect_orphan_ngram_branch"]
                break
    if not isinstance(orphan, Mapping):
        return projected

    def concept_term(raw: Mapping[str, Any]) -> dict[str, Any] | None:
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

    def concept_cluster(raw: Mapping[str, Any]) -> dict[str, Any]:
        cluster_id = str(raw.get("cluster_id") or raw.get("topic_id") or "").strip()
        if not cluster_id:
            raise ValueError("native orphan cluster has no cluster_id")
        raw_terms = raw.get("terms")
        if raw_terms is None:
            raw_terms = raw.get("member_terms", raw.get("supporting_terms"))
        if raw_terms is None:
            raw_terms = raw.get("term_scores")
        if not isinstance(raw_terms, (list, tuple)):
            raise ValueError(f"native orphan cluster {cluster_id} has no term collection")
        terms: list[dict[str, Any]] = []
        seen_terms: set[str] = set()
        for value in raw_terms:
            term_row = {"term": value} if isinstance(value, str) else value
            if not isinstance(term_row, Mapping):
                raise ValueError(f"native orphan cluster {cluster_id} has a malformed term")
            candidates = [term_row]
            aliases = term_row.get("nested_aliases")
            if aliases is not None:
                if not isinstance(aliases, (list, tuple)):
                    raise ValueError("native orphan nested_aliases must be a sequence")
                candidates.extend(aliases)
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    raise ValueError("native orphan alias must be a mapping")
                compact = concept_term(candidate)
                if compact is None or compact["term"] in seen_terms:
                    continue
                seen_terms.add(compact["term"])
                terms.append(compact)
        if not terms:
            raise ValueError(f"native orphan cluster {cluster_id} has no concept terms")
        ranks = [int(row["fit_rank"]) for row in terms if row.get("fit_rank") is not None]
        signed = [
            abs(float(row.get("fit_signed_score", row.get("signed_score"))))
            for row in terms
            if row.get("fit_signed_score", row.get("signed_score")) is not None
        ]
        return {
            "cluster_id": cluster_id,
            "evidence_kind": str(raw.get("evidence_kind") or "orphan_raw_ngram_cluster"),
            "terms": terms,
            "seed_term": str(raw.get("seed_term") or terms[0]["term"]),
            "fit_rank": (min(ranks) if ranks else None),
            "maximum_abs_fit_signed_score": (max(signed) if signed else None),
            "grouping_method": str(
                raw.get("grouping_method") or "native_fit_side_orphan_ngram_cluster"
            ),
        }

    def clusters(key: str) -> list[dict[str, Any]]:
        values = orphan.get(key) or []
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"native orphan {key} must be a sequence")
        return [concept_cluster(value) for value in values if isinstance(value, Mapping)]

    selected_clusters = clusters("selected_clusters")
    all_clusters = clusters("clusters")
    selected_ids = [str(value) for value in (orphan.get("selected_cluster_ids") or [])]
    if set(selected_ids) != {row["cluster_id"] for row in selected_clusters}:
        raise ValueError("native orphan selected IDs differ from its selected clusters")
    projected["effect_orphan_ngram_branch"] = {
        "schema_version": "tfidf_nested_fit_orphan_concept_projection_v1",
        "status": orphan.get("status"),
        "candidate_definition": orphan.get("candidate_definition"),
        "uses_outer_heldout_labels": False,
        "uses_heldout_treatment_and_outcome": False,
        "fits_patient_level_cate_model": False,
        "topic_term_exclusion_is_fit_side": orphan.get("topic_term_exclusion_is_fit_side"),
        "cluster_construction_uses_heldout_rows_or_labels": orphan.get(
            "cluster_construction_uses_heldout_rows_or_labels"
        ),
        "candidate_count_before_topic_exclusion": orphan.get(
            "candidate_count_before_topic_exclusion"
        ),
        "represented_topic_term_exclusion_count": orphan.get(
            "represented_topic_term_exclusion_count"
        ),
        "candidate_count_before_nested_deduplication": orphan.get(
            "candidate_count_before_nested_deduplication"
        ),
        "deduplicated_alias_count": orphan.get("deduplicated_alias_count"),
        "representative_count": orphan.get("representative_count"),
        "cluster_count": len(all_clusters),
        "selected_cluster_ids": selected_ids,
        "selected_clusters": selected_clusters,
        "clusters": all_clusters,
        "selection_count": len(selected_clusters),
        "selection_rule": orphan.get("selection_rule"),
        "minimum_selected_clusters": orphan.get("minimum_selected_clusters"),
        "maximum_selected_clusters": orphan.get("maximum_selected_clusters"),
    }
    return projected


def _numeric_array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(
        _canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)}).encode("utf-8")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _matched_pair_subproducer_proofs(
    *,
    bundle: Any,
    expected_bow_views: Sequence[str],
    scope_id: str,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
) -> Mapping[str, Any]:
    """Require independent BoW and HTR matched-pair outputs as a gate diagnostic.

    The legacy runner historically swallowed subproducer failures and could
    leave one surviving matched-pair branch looking like success for the
    combined family.  These proofs bind each branch to its own output columns,
    evidence/model-output artifact, and exact scope.  They are not genuine
    producer-emitted model hashes or fit audits, so the production adapter gate
    must remain closed until both branches emit those stronger proofs.
    """

    x_names = tuple(map(str, bundle.x_names))
    x_train = np.asarray(bundle.x_train)
    x_heldout = np.asarray(bundle.x_test)
    if (
        x_train.ndim != 2
        or x_heldout.ndim != 2
        or (x_train.shape[1] != len(x_names) or x_heldout.shape[1] != len(x_names))
    ):
        raise RuntimeError("matched-pair proof received malformed Stage 1 feature matrices")
    failures = [
        row
        for row in (bundle.inner_model_rows or ())
        if isinstance(row, Mapping)
        and row.get("source_family") in {"bow_pair_uplift", "htr_pair_uplift"}
        and row.get("skipped") is not None
    ]
    if failures:
        failed_families = sorted({str(row.get("source_family")) for row in failures})
        raise RuntimeError(
            "matched-pair subproducer failed inside the required scope: "
            + ", ".join(failed_families)
        )

    importance = (bundle.handoff_evidence or {}).get("importance") or {}
    pair_importance = importance.get("matched_pair_uplift")
    if not isinstance(pair_importance, Mapping):
        raise RuntimeError("BoW matched-pair subproducer produced no independent evidence")
    expected_views = tuple(map(str, expected_bow_views))
    observed_views = tuple(
        str(row.get("view_name") or "").removeprefix("pair_uplift__")
        for row in pair_importance.get("views") or ()
        if isinstance(row, Mapping)
    )
    if set(observed_views) != set(expected_views) or len(observed_views) != len(expected_views):
        raise RuntimeError("BoW matched-pair subproducer did not complete every configured view")
    bow_columns = tuple(
        name
        for view in expected_views
        for name in (
            f"bow__{view}__matched_pair_uplift_delta_logit",
            f"bow__{view}__matched_pair_treated_outcome_prob",
        )
    )
    htr_columns = (
        "htr__matched_pair_uplift_delta_logit",
        "htr__matched_pair_treated_outcome_prob",
    )
    missing_columns = [name for name in (*bow_columns, *htr_columns) if name not in x_names]
    if missing_columns:
        raise RuntimeError(
            "matched-pair subproducer output columns are missing: " + ", ".join(missing_columns)
        )
    htr_stage = (bundle.handoff_evidence or {}).get("htr_evidence") or {}
    htr_stage = htr_stage.get("pair_uplift") if isinstance(htr_stage, Mapping) else None
    if not isinstance(htr_stage, Mapping):
        raise RuntimeError("HTR matched-pair subproducer produced no independent evidence")

    def proof(kind: str, columns: Sequence[str], evidence: Mapping[str, Any]) -> dict[str, Any]:
        indices = [x_names.index(name) for name in columns]
        model_output = {
            "columns": list(columns),
            "train_values_sha256": _numeric_array_sha256(x_train[:, indices]),
            "heldout_values_sha256": _numeric_array_sha256(x_heldout[:, indices]),
            "evidence_sha256": _sha256_json(evidence),
        }
        model_artifact_sha256 = _sha256_json(model_output)
        execution = {
            "scope_id": scope_id,
            "subproducer": kind,
            "fit_row_fingerprint": row_set_fingerprint(fit_row_ids),
            "heldout_row_fingerprint": row_set_fingerprint(heldout_row_ids),
            "model_artifact_sha256": model_artifact_sha256,
        }
        return {
            "schema_version": STAGE1_MATCHED_PAIR_PROOF_SCHEMA,
            "subproducer": kind,
            "success": True,
            "output_columns": list(columns),
            "model_artifact_sha256": model_artifact_sha256,
            "fit_execution_sha256": _sha256_json(execution),
            "artifact_semantics": "sealed_model_outputs_and_concept_evidence",
        }

    proofs = {
        "bow": proof("bow", bow_columns, pair_importance),
        "htr": proof("htr", htr_columns, htr_stage),
    }
    return {
        "schema_version": STAGE1_MATCHED_PAIR_PROOF_SCHEMA,
        "scope_id": scope_id,
        "all_required_subproducers_succeeded": True,
        "subproducers": proofs,
        "content_sha256": _sha256_json(proofs),
    }


def _write_raw_evidence_sidecar(
    path: Path,
    *,
    component_root: Path,
    scope: Mapping[str, Any],
    split_registry_content_sha256: str,
    raw_evidence: Mapping[str, Any],
    matched_pair_proofs: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Persist exact raw model evidence behind a non-prompt authenticated boundary."""

    body = {
        "schema_version": STAGE1_RAW_EVIDENCE_SIDECAR_SCHEMA,
        "scope_id": str(scope["scope_id"]),
        "outer_fold": int(scope["outer_fold"]),
        "inner_fold": (None if scope.get("inner_fold") is None else int(scope["inner_fold"])),
        "fit_row_fingerprint": row_set_fingerprint(scope["fit_row_ids"]),
        "heldout_row_fingerprint": row_set_fingerprint(scope["heldout_row_ids"]),
        "split_registry_content_sha256": split_registry_content_sha256,
        "prompt_grounding_allowed": False,
        "raw_drillback_requires_authenticated_id": True,
        "model_evidence": _sanitize_secrets(raw_evidence),
        "matched_pair_subproducer_proofs": copy.deepcopy(dict(matched_pair_proofs)),
    }
    payload = {**body, "content_sha256": _sha256_json(body)}
    _write_immutable_json(path, payload)
    return {
        "relative_path": path.relative_to(component_root).as_posix(),
        "size": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "content_sha256": payload["content_sha256"],
        "prompt_grounding_allowed": False,
    }


def _validate_raw_evidence_sidecar(
    path: Path,
    *,
    registration: Mapping[str, Any],
    scope: Mapping[str, Any],
    split_registry_content_sha256: str,
) -> Mapping[str, Any]:
    if (
        not path.is_file()
        or path.stat().st_size != int(registration.get("size", -1))
        or _sha256_file(path) != registration.get("sha256")
    ):
        raise RuntimeError(f"raw evidence sidecar changed: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    declared = body.pop("content_sha256", None)
    if not _HEX_SHA256.fullmatch(str(declared or "")) or _sha256_json(body) != declared:
        raise RuntimeError(f"raw evidence sidecar content hash is invalid: {path}")
    expected = {
        "schema_version": STAGE1_RAW_EVIDENCE_SIDECAR_SCHEMA,
        "scope_id": str(scope["scope_id"]),
        "outer_fold": int(scope["outer_fold"]),
        "inner_fold": (None if scope.get("inner_fold") is None else int(scope["inner_fold"])),
        "fit_row_fingerprint": row_set_fingerprint(scope["fit_row_ids"]),
        "heldout_row_fingerprint": row_set_fingerprint(scope["heldout_row_ids"]),
        "split_registry_content_sha256": split_registry_content_sha256,
        "prompt_grounding_allowed": False,
        "raw_drillback_requires_authenticated_id": True,
    }
    mismatched = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatched or registration.get("content_sha256") != declared:
        raise RuntimeError(
            f"raw evidence sidecar scope binding is invalid: {path}; fields={mismatched}"
        )
    proofs = payload.get("matched_pair_subproducer_proofs")
    subproducers = proofs.get("subproducers") if isinstance(proofs, Mapping) else None
    if (
        not isinstance(proofs, Mapping)
        or proofs.get("schema_version") != STAGE1_MATCHED_PAIR_PROOF_SCHEMA
        or proofs.get("scope_id") != str(scope["scope_id"])
        or proofs.get("all_required_subproducers_succeeded") is not True
        or not isinstance(subproducers, Mapping)
        or set(subproducers) != {"bow", "htr"}
        or proofs.get("content_sha256") != _sha256_json(subproducers or {})
        or any(
            not isinstance(row, Mapping)
            or row.get("schema_version") != STAGE1_MATCHED_PAIR_PROOF_SCHEMA
            or row.get("subproducer") != name
            or row.get("success") is not True
            or not isinstance(row.get("output_columns"), list)
            or not row.get("output_columns")
            or not _HEX_SHA256.fullmatch(str(row.get("model_artifact_sha256") or ""))
            or not _HEX_SHA256.fullmatch(str(row.get("fit_execution_sha256") or ""))
            for name, row in (subproducers or {}).items()
        )
    ):
        raise RuntimeError(f"raw evidence sidecar lacks separate matched-pair proofs: {path}")
    return payload


def _component_regular_file(
    component_root: Path,
    raw_path: Path | str,
    *,
    field_name: str,
) -> Path:
    """Resolve one declared native artifact without accepting path aliases."""

    root = Path(component_root).resolve(strict=True)
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{field_name} must be one regular component file")
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field_name} escapes its sealed component") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{field_name} traverses a symlink")
    return resolved


def _component_file_registration(path: Path, *, component_root: Path) -> dict[str, Any]:
    root = Path(component_root).resolve(strict=True)
    artifact = _component_regular_file(root, path, field_name="registered native artifact")
    return {
        "relative_path": artifact.relative_to(root).as_posix(),
        "size": int(artifact.stat().st_size),
        "sha256": _sha256_file(artifact),
    }


def _component_native_artifact_registration(
    path: Path,
    *,
    component_root: Path,
) -> dict[str, Any]:
    """Register one native file or directory without accepting path aliases."""

    root = Path(component_root).resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_symlink() or not candidate.exists():
        raise ValueError("registered native artifact must exist and cannot be a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("registered native artifact escapes its sealed component") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("registered native artifact traverses a symlink")
    if resolved.is_dir():
        descendants = sorted(resolved.rglob("*"))
        if not descendants or any(item.is_symlink() for item in descendants):
            raise ValueError("registered native artifact directory is empty or contains a symlink")
        files = [item for item in descendants if item.is_file()]
        if not files:
            raise ValueError("registered native artifact directory has no files")
        return {
            "relative_path": relative.as_posix(),
            "kind": "directory",
            "file_count": len(files),
            "size": sum(int(item.stat().st_size) for item in files),
            "sha256": native_artifact_sha256(resolved),
        }
    if not resolved.is_file():
        raise ValueError("registered native artifact is not a regular file or directory")
    return {
        "relative_path": relative.as_posix(),
        "kind": "file",
        "file_count": 1,
        "size": int(resolved.stat().st_size),
        "sha256": native_artifact_sha256(resolved),
    }


def _numerical_array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("production neural-query numerical artifacts cannot contain objects")
    header = _canonical_json(
        {
            "dtype": array.dtype.str,
            "shape": [int(dimension) for dimension in array.shape],
        }
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _validate_neural_query_moment_artifact(
    metadata_path: Path,
    *,
    expected_scope_id: str | None = None,
    expected_fit_row_ids: Sequence[int] | None = None,
    expected_heldout_row_ids: Sequence[int] | None = None,
    expected_query_cache_key: str | None = None,
    expected_snapshot_content_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Validate exact heldout moments without accepting executable array types."""

    metadata_path = Path(metadata_path)
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError("neural-query moment metadata must be one regular file")
    before = metadata_path.stat()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("neural-query moment metadata is not valid JSON") from exc
    after = metadata_path.stat()
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
        raise RuntimeError("neural-query moment metadata changed while reading")
    if not isinstance(metadata, dict):
        raise ValueError("neural-query moment metadata must be one JSON object")
    expected_fields = {
        "schema_version",
        "scope_id",
        "outer_fold",
        "inner_fold",
        "fit_row_ids",
        "heldout_row_ids",
        "fit_row_fingerprint",
        "heldout_row_fingerprint",
        "split_scope_fingerprint",
        "data_projection_sha256",
        "query_cache_key",
        "owned_snapshot_schema_version",
        "owned_snapshot_content_sha256",
        "arrays_file",
        "arrays_sha256",
        "array_inventory",
        "feature_count",
        "heldout_row_count",
        "heldout_columns_read",
        "heldout_labels_supplied",
        "gate_treatment_or_outcome_accepted_by_backend",
        "content_sha256",
    }
    body = {key: value for key, value in metadata.items() if key != "content_sha256"}
    if (
        set(metadata) != expected_fields
        or metadata.get("schema_version") != STAGE1_QUERY_MOMENT_ARTIFACT_SCHEMA
        or metadata.get("content_sha256") != _sha256_json(body)
        or metadata.get("owned_snapshot_schema_version") != NEURAL_QUERY_OWNED_SNAPSHOT_SCHEMA
        or metadata.get("heldout_labels_supplied") is not False
        or metadata.get("gate_treatment_or_outcome_accepted_by_backend") is not False
    ):
        raise ValueError("neural-query moment metadata has an invalid closed envelope")
    scope_id = str(metadata.get("scope_id") or "")
    fit_rows = tuple(map(int, metadata.get("fit_row_ids") or ()))
    heldout_rows = tuple(map(int, metadata.get("heldout_row_ids") or ()))
    if (
        not fit_rows
        or not heldout_rows
        or len(fit_rows) != len(set(fit_rows))
        or len(heldout_rows) != len(set(heldout_rows))
        or set(fit_rows) & set(heldout_rows)
        or metadata.get("fit_row_fingerprint") != row_set_fingerprint(fit_rows)
        or metadata.get("heldout_row_fingerprint") != row_set_fingerprint(heldout_rows)
        or int(metadata.get("heldout_row_count", 0)) != len(heldout_rows)
    ):
        raise ValueError("neural-query moment metadata changed its exact row scope")
    if expected_scope_id is not None and scope_id != str(expected_scope_id):
        raise ValueError("neural-query moments are registered under another scope")
    if expected_fit_row_ids is not None and fit_rows != tuple(map(int, expected_fit_row_ids)):
        raise ValueError("neural-query moments have another fit-row order")
    if expected_heldout_row_ids is not None and heldout_rows != tuple(
        map(int, expected_heldout_row_ids)
    ):
        raise ValueError("neural-query moments have another heldout-row order")
    query_cache_key = str(metadata.get("query_cache_key") or "")
    snapshot_sha256 = str(metadata.get("owned_snapshot_content_sha256") or "")
    if (
        _HEX_SHA256.fullmatch(query_cache_key) is None
        or _HEX_SHA256.fullmatch(snapshot_sha256) is None
    ):
        raise ValueError("neural-query moment metadata lacks its fitted-query binding")
    if expected_query_cache_key is not None and query_cache_key != str(expected_query_cache_key):
        raise ValueError("neural-query moments use another fitted query cache key")
    if expected_snapshot_content_sha256 is not None and snapshot_sha256 != str(
        expected_snapshot_content_sha256
    ):
        raise ValueError("neural-query moments use another owned query snapshot")
    arrays_path = metadata_path.parent / str(metadata.get("arrays_file") or "")
    if (
        arrays_path.parent != metadata_path.parent
        or arrays_path.is_symlink()
        or not arrays_path.is_file()
    ):
        raise ValueError("neural-query moment NPZ path is invalid")
    if _sha256_file(arrays_path) != str(metadata.get("arrays_sha256") or ""):
        raise RuntimeError("neural-query heldout moment NPZ changed after emission")
    expected_keys = {
        "heldout_row_ids",
        "feature_values",
        "feature_names",
        "feature_kinds",
        "feature_roles",
    }
    inventory = metadata.get("array_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != expected_keys:
        raise ValueError("neural-query moment array inventory is incomplete")
    observed_inventory: dict[str, Any] = {}
    before_npz = _sha256_file(arrays_path)
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError("neural-query moment NPZ has unexpected arrays")
        arrays = {key: np.asarray(archive[key]) for key in expected_keys}
        for key, array in arrays.items():
            if array.dtype.hasobject:
                raise ValueError("neural-query moment NPZ contains executable object arrays")
            observed_inventory[key] = {
                "dtype": array.dtype.str,
                "shape": [int(dimension) for dimension in array.shape],
                "content_sha256": _numerical_array_sha256(array),
            }
    if _sha256_file(arrays_path) != before_npz:
        raise RuntimeError("neural-query moment NPZ changed while validating")
    if observed_inventory != dict(inventory):
        raise RuntimeError("neural-query moment array inventory differs from its NPZ")
    row_ids = np.asarray(arrays["heldout_row_ids"], dtype=np.int64)
    values = np.asarray(arrays["feature_values"], dtype=float)
    names = tuple(map(str, arrays["feature_names"].tolist()))
    kinds = tuple(map(str, arrays["feature_kinds"].tolist()))
    roles = tuple(map(str, arrays["feature_roles"].tolist()))
    feature_count = int(metadata.get("feature_count", 0))
    if (
        row_ids.ndim != 1
        or tuple(map(int, row_ids.tolist())) != heldout_rows
        or values.shape != (len(heldout_rows), feature_count)
        or not np.isfinite(values).all()
        or feature_count < 1
        or len(names) != feature_count
        or len(kinds) != feature_count
        or len(roles) != feature_count
        or len(names) != len(set(names))
        or any(not name.startswith("neural_query_") for name in names)
    ):
        raise ValueError("neural-query heldout moment arrays are not rectangular and scope-bound")
    columns = metadata.get("heldout_columns_read")
    if (
        not isinstance(columns, list)
        or len(columns) != 2
        or columns[0] != "_oci_row_id"
        or not isinstance(columns[1], str)
        or not columns[1].strip()
    ):
        raise ValueError("neural-query heldout transform did not attest ID/text-only access")
    return copy.deepcopy(metadata)


def _write_neural_query_moment_artifact(
    model_artifact_dir: Path,
    *,
    scope_id: str,
    outer_fold: int,
    inner_fold: int | None,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    split_scope_fingerprint: str,
    data_projection_sha256: str,
    query_cache_key: str,
    owned_snapshot_metadata: Mapping[str, Any],
    text_column: str,
    prediction: Any,
) -> Mapping[str, Any]:
    fit_rows = tuple(map(int, fit_row_ids))
    heldout_rows = tuple(map(int, heldout_row_ids))
    if tuple(map(int, prediction.gate_row_ids)) != heldout_rows:
        raise ValueError("neural-query prediction changed exact heldout row order")
    values = np.asarray(prediction.feature_values, dtype=np.float32)
    names = tuple(map(str, prediction.feature_names))
    kinds = tuple(map(str, prediction.feature_kinds))
    roles = tuple(map(str, prediction.feature_roles))
    if (
        values.shape != (len(heldout_rows), len(names))
        or not np.isfinite(values).all()
        or not names
        or len(kinds) != len(names)
        or len(roles) != len(names)
    ):
        raise ValueError("neural-query backend returned invalid heldout moments")
    arrays = {
        "heldout_row_ids": np.asarray(heldout_rows, dtype=np.int64),
        "feature_values": values,
        "feature_names": np.asarray(names, dtype=str),
        "feature_kinds": np.asarray(kinds, dtype=str),
        "feature_roles": np.asarray(roles, dtype=str),
    }
    model_root = Path(model_artifact_dir)
    if model_root.is_symlink() or not model_root.is_dir():
        raise ValueError("neural-query native model artifact root must be a real directory")
    arrays_path = model_root / "heldout_moments.npz"
    metadata_path = model_root / "heldout_moments.metadata.json"
    if arrays_path.exists() or metadata_path.exists():
        raise RuntimeError("refusing to replace immutable neural-query heldout moments")
    _atomic_write_npz(arrays_path, **arrays)
    inventory = {
        key: {
            "dtype": array.dtype.str,
            "shape": [int(dimension) for dimension in array.shape],
            "content_sha256": _numerical_array_sha256(array),
        }
        for key, array in arrays.items()
    }
    body = {
        "schema_version": STAGE1_QUERY_MOMENT_ARTIFACT_SCHEMA,
        "scope_id": str(scope_id),
        "outer_fold": int(outer_fold),
        "inner_fold": None if inner_fold is None else int(inner_fold),
        "fit_row_ids": list(fit_rows),
        "heldout_row_ids": list(heldout_rows),
        "fit_row_fingerprint": row_set_fingerprint(fit_rows),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_rows),
        "split_scope_fingerprint": str(split_scope_fingerprint),
        "data_projection_sha256": str(data_projection_sha256),
        "query_cache_key": str(query_cache_key),
        "owned_snapshot_schema_version": NEURAL_QUERY_OWNED_SNAPSHOT_SCHEMA,
        "owned_snapshot_content_sha256": str(owned_snapshot_metadata["content_sha256"]),
        "arrays_file": arrays_path.name,
        "arrays_sha256": _sha256_file(arrays_path),
        "array_inventory": inventory,
        "feature_count": len(names),
        "heldout_row_count": len(heldout_rows),
        "heldout_columns_read": ["_oci_row_id", str(text_column)],
        "heldout_labels_supplied": False,
        "gate_treatment_or_outcome_accepted_by_backend": False,
    }
    metadata = {**body, "content_sha256": _sha256_json(body)}
    _write_immutable_json(metadata_path, metadata)
    return _validate_neural_query_moment_artifact(
        metadata_path,
        expected_scope_id=scope_id,
        expected_fit_row_ids=fit_rows,
        expected_heldout_row_ids=heldout_rows,
        expected_query_cache_key=query_cache_key,
        expected_snapshot_content_sha256=str(owned_snapshot_metadata["content_sha256"]),
    )


def _register_neural_query_native_family_proof(
    *,
    component_root: Path,
    proof_directory: Path,
    scope_id: str,
    catalog: RoleNeutralEvidenceCatalog,
    query_artifact_path: Path,
    model_artifact_path: Path,
    outer_fold: int,
    inner_fold: int,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    fit_treatment: Sequence[float],
    fit_outcome: Sequence[float],
    split_scope_fingerprint: str,
    data_projection_sha256: str,
    configuration: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Register a genuine fitted-query snapshot and exact heldout transform."""

    if int(outer_fold) < 1 or int(inner_fold) < 1:
        raise ValueError("neural-query native proof registration requires an exact-inner scope")
    root = Path(component_root).resolve(strict=True)
    source_path = _component_regular_file(
        root,
        query_artifact_path,
        field_name="neural-query safe evidence artifact",
    )
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("neural-query safe evidence artifact is not valid JSON") from exc
    fit_rows = tuple(map(int, fit_row_ids))
    heldout_rows = tuple(map(int, heldout_row_ids))
    if (
        not isinstance(source, Mapping)
        or source.get("schema_version") != STAGE1_QUERY_ARTIFACT_SCHEMA
        or source.get("source_family") != NEURAL_QUERY_MOMENTS
        or source.get("scope_id") != str(scope_id)
        or int(source.get("outer_fold", 0)) != int(outer_fold)
        or int(source.get("inner_fold", 0)) != int(inner_fold)
        or tuple(map(int, source.get("fit_row_ids") or ())) != fit_rows
        or tuple(map(int, source.get("heldout_row_ids") or ())) != heldout_rows
        or source.get("fit_row_fingerprint") != row_set_fingerprint(fit_rows)
        or source.get("heldout_row_fingerprint") != row_set_fingerprint(heldout_rows)
        or source.get("heldout_labels_supplied") is not False
    ):
        raise ValueError("neural-query safe evidence artifact changed its exact-inner scope")
    cache_key = str(source.get("query_cache_key") or "")
    if _HEX_SHA256.fullmatch(cache_key) is None:
        raise ValueError("neural-query safe evidence artifact lacks its owned cache key")
    model_registration = _component_native_artifact_registration(
        Path(model_artifact_path),
        component_root=root,
    )
    if model_registration["kind"] != "directory":
        raise ValueError("neural-query native model artifact must be a directory tree")
    model_root = root / str(model_registration["relative_path"])
    snapshot_metadata = validate_owned_discovery_snapshot(
        model_root / "owned_snapshot",
        expected_cache_key=cache_key,
    )
    binding = snapshot_metadata["binding"]
    if tuple(map(int, binding.get("row_ids") or ())) != fit_rows or int(
        binding.get("outer_fold", 0)
    ) != int(outer_fold):
        raise ValueError("neural-query owned snapshot changed its exact fit rows")
    if binding.get("treatment_sha256") != _float_hex_sha256(fit_treatment) or binding.get(
        "outcome_sha256"
    ) != _float_hex_sha256(fit_outcome):
        raise ValueError("neural-query owned snapshot differs from canonical fit labels")
    moment_metadata_path = _component_regular_file(
        root,
        model_root / "heldout_moments.metadata.json",
        field_name="neural-query heldout moment metadata",
    )
    moment_metadata = _validate_neural_query_moment_artifact(
        moment_metadata_path,
        expected_scope_id=scope_id,
        expected_fit_row_ids=fit_rows,
        expected_heldout_row_ids=heldout_rows,
        expected_query_cache_key=cache_key,
        expected_snapshot_content_sha256=str(snapshot_metadata["content_sha256"]),
    )
    declared_model = source.get("native_model_artifact")
    declared_moments = source.get("heldout_moment_artifact")
    if (
        not isinstance(declared_model, Mapping)
        or declared_model.get("relative_path") != model_registration["relative_path"]
        or declared_model.get("sha256") != model_registration["sha256"]
        or not isinstance(declared_moments, Mapping)
        or declared_moments.get("relative_path")
        != (model_root / "heldout_moments.npz").relative_to(root).as_posix()
        or declared_moments.get("sha256") != moment_metadata["arrays_sha256"]
    ):
        raise ValueError("neural-query safe evidence does not bind its native numerical artifacts")
    if configuration.get("scope_id") != str(scope_id) or (
        configuration.get("heldout_label_policy") != "id_and_text_only"
    ):
        raise ValueError("neural-query native configuration is not exact-scope label-safe")

    evidence_payload, evidence_item_count = family_payload_from_catalog(
        catalog,
        family=NEURAL_QUERY_MOMENTS,
    )
    proof_root = Path(proof_directory)
    if not proof_root.is_absolute():
        proof_root = root / proof_root
    if proof_root.exists() and proof_root.is_symlink():
        raise ValueError("neural-query proof directory cannot be a symlink")
    proof_root.mkdir(parents=True, exist_ok=True)
    try:
        proof_root.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError("neural-query proof directory escapes its component") from exc
    payload_path = proof_root / f"{NEURAL_QUERY_MOMENTS}.evidence_payload.json"
    metadata_path = proof_root / f"{NEURAL_QUERY_MOMENTS}.fit_metadata.json"
    execution_path = proof_root / f"{NEURAL_QUERY_MOMENTS}.execution.json"
    _write_immutable_json(payload_path, evidence_payload)
    nuisance_binding = snapshot_metadata["discovery_metadata"]["fit_nuisance_output_binding"]
    fit_metadata_body = {
        "schema_version": STAGE1_QUERY_NATIVE_FIT_METADATA_SCHEMA,
        "scope_id": str(scope_id),
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "fit_semantics": EXACT_INNER_REFIT,
        "fit_row_ids": list(fit_rows),
        "heldout_row_ids": list(heldout_rows),
        "fit_row_fingerprint": row_set_fingerprint(fit_rows),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_rows),
        "split_scope_fingerprint": str(split_scope_fingerprint),
        "data_projection_sha256": str(data_projection_sha256),
        "query_cache_key": cache_key,
        "service_identity_sha256": snapshot_metadata["service_identity_sha256"],
        "owned_snapshot_content_sha256": snapshot_metadata["content_sha256"],
        "owned_discovery_content_sha256": snapshot_metadata["owned_discovery_content_sha256"],
        "fit_input_binding_sha256": snapshot_metadata["discovery_metadata"][
            "fit_input_binding_sha256"
        ],
        "fit_e_sha256": nuisance_binding["fit_e_sha256"],
        "fit_m_sha256": nuisance_binding["fit_m_sha256"],
        "heldout_moment_content_sha256": moment_metadata["content_sha256"],
        "heldout_moment_arrays_sha256": moment_metadata["arrays_sha256"],
        "model_artifact_sha256": model_registration["sha256"],
        "source_artifact_sha256": _sha256_file(source_path),
        "heldout_columns_read": moment_metadata["heldout_columns_read"],
        "heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "executable_checkpoint_retained": False,
        "joblib_checkpoint_loaded_for_snapshot": False,
    }
    fit_metadata = {
        **fit_metadata_body,
        "content_sha256": _sha256_json(fit_metadata_body),
    }
    _write_immutable_json(metadata_path, fit_metadata)
    semantics = (
        "non-executable owned fitted query arrays, fit activations, and exact "
        "ID/text-only heldout moment transforms"
    )
    execution_record = native_family_execution_record(
        family=NEURAL_QUERY_MOMENTS,
        fit_semantics=EXACT_INNER_REFIT,
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        split_scope_fingerprint=split_scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        evidence_payload=evidence_payload,
        configuration=configuration,
        native_fit_metadata_path=metadata_path,
        model_artifact_path=model_root,
        source_artifact_path=source_path,
        model_artifact_semantics=semantics,
    )
    _write_immutable_json(execution_path, execution_record)
    proof = bind_native_family_fit_proof(
        family=NEURAL_QUERY_MOMENTS,
        fit_semantics=EXACT_INNER_REFIT,
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        split_scope_fingerprint=split_scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        evidence_payload=evidence_payload,
        configuration=configuration,
        native_fit_metadata_path=metadata_path,
        native_execution_record_path=execution_path,
        model_artifact_path=model_root,
        source_artifact_path=source_path,
        model_artifact_semantics=semantics,
    )
    proof.verify_artifact_bytes()
    family_row = {
        "family": NEURAL_QUERY_MOMENTS,
        "evidence_item_count": int(evidence_item_count),
        "proof": proof.as_dict(),
        "evidence_payload": _component_file_registration(payload_path, component_root=root),
        "native_execution_record": _component_file_registration(
            execution_path,
            component_root=root,
        ),
        "native_fit_metadata": _component_file_registration(
            metadata_path,
            component_root=root,
        ),
        "model_artifact": model_registration,
        "source_artifact": _component_file_registration(source_path, component_root=root),
        "heldout_moment_metadata": _component_file_registration(
            moment_metadata_path,
            component_root=root,
        ),
    }
    registration_body = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
        "scope_id": str(scope_id),
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "fit_row_ids": list(fit_rows),
        "heldout_row_ids": list(heldout_rows),
        "split_scope_fingerprint": str(split_scope_fingerprint),
        "data_projection_sha256": str(data_projection_sha256),
        "fit_semantics": EXACT_INNER_REFIT,
        "registered_families": list(PRODUCTION_QUERY_REGISTERED_NATIVE_FAMILY_ADAPTERS),
        "heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "family_proofs": [family_row],
    }
    registration = {
        **registration_body,
        "content_sha256": _sha256_json(registration_body),
    }
    registration_path = proof_root / "registration.json"
    _write_immutable_json(registration_path, registration)
    return {
        **registration,
        "registration": _component_file_registration(
            registration_path,
            component_root=root,
        ),
    }


def _validate_component_native_registration(
    component_root: Path,
    registration: Mapping[str, Any],
) -> Path:
    if not isinstance(registration, Mapping):
        raise ValueError("native component artifact registration must be a mapping")
    root = Path(component_root).resolve(strict=True)
    raw_relative = str(registration.get("relative_path") or "")
    if not raw_relative or Path(raw_relative).is_absolute():
        raise ValueError("native component artifact registration path is invalid")
    candidate = root / raw_relative
    if "kind" in registration or "file_count" in registration:
        observed = _component_native_artifact_registration(candidate, component_root=root)
        keys = ("relative_path", "kind", "file_count", "size", "sha256")
    else:
        observed = _component_file_registration(candidate, component_root=root)
        keys = ("relative_path", "size", "sha256")
    for key in keys:
        if observed.get(key) != registration.get(key):
            raise RuntimeError("registered native component artifact changed")
    return candidate.resolve(strict=True)


def _record_reloaded_exact_inner_family(
    collector: dict[str, dict[str, Mapping[str, Any]]] | None,
    *,
    scope_id: str,
    family: str,
    proof: NativeFamilyFitProof,
    evidence_payload: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
) -> None:
    if collector is None:
        return
    by_family = collector.setdefault(scope_id, {})
    if family in by_family:
        raise RuntimeError(f"duplicate reloaded exact-inner family: {scope_id}/{family}")
    by_family[family] = {
        "proof": proof,
        "evidence_payload": copy.deepcopy(dict(evidence_payload)),
        "artifact_paths": dict(artifact_paths),
    }


def _validate_neural_query_native_family_proof_index(
    *,
    component_root: Path,
    index_registration: Mapping[str, Any],
    expected_inner_scopes: Mapping[str, Mapping[str, Any]],
    split_registry_content_sha256: str,
    modeling_data: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    expected_configuration_by_scope: Mapping[str, Mapping[str, Any]] | None = None,
    reloaded_native_by_scope: dict[str, dict[str, Mapping[str, Any]]] | None = None,
) -> Mapping[str, Any]:
    root = Path(component_root).resolve(strict=True)
    index_path = _validate_component_native_registration(root, index_registration)
    if not index_path.is_file():
        raise ValueError("neural-query native proof index must be one regular file")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("neural-query native proof index is not valid JSON") from exc
    if not isinstance(index, dict):
        raise ValueError("neural-query native proof index must be one JSON object")
    index_body = {key: value for key, value in index.items() if key != "content_sha256"}
    scopes = index.get("scopes")
    if (
        index.get("schema_version") != STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA
        or index.get("split_registry_content_sha256") != split_registry_content_sha256
        or index.get("registered_families")
        != list(PRODUCTION_QUERY_REGISTERED_NATIVE_FAMILY_ADAPTERS)
        or index.get("executable_checkpoint_files_retained") is not False
        or index.get("content_sha256") != _sha256_json(index_body)
        or not isinstance(scopes, list)
        or int(index.get("exact_inner_scope_count", -1)) != len(scopes)
    ):
        raise ValueError("neural-query native proof index has an invalid closed envelope")
    indexed_scopes = {str(row.get("scope_id")): row for row in scopes if isinstance(row, Mapping)}
    if len(indexed_scopes) != len(scopes) or set(indexed_scopes) != set(expected_inner_scopes):
        raise ValueError("neural-query native proof index has incomplete exact-inner coverage")
    for scope_id, expected in expected_inner_scopes.items():
        row = indexed_scopes[scope_id]
        if (
            int(row.get("outer_fold", 0)) != int(expected["outer_fold"])
            or int(row.get("inner_fold", 0)) != int(expected["inner_fold"])
            or row.get("registered_families")
            != list(PRODUCTION_QUERY_REGISTERED_NATIVE_FAMILY_ADAPTERS)
        ):
            raise ValueError(f"neural-query native proof index scope mismatch: {scope_id}")
        registration_path = _validate_component_native_registration(
            root,
            row.get("registration") or {},
        )
        if not registration_path.is_file():
            raise ValueError("neural-query native proof registration must be one JSON file")
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        registration_body = {
            key: value for key, value in registration.items() if key != "content_sha256"
        }
        family_rows = registration.get("family_proofs")
        if (
            registration.get("schema_version") != STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA
            or registration.get("scope_id") != scope_id
            or registration.get("registered_families")
            != list(PRODUCTION_QUERY_REGISTERED_NATIVE_FAMILY_ADAPTERS)
            or registration.get("heldout_labels_accessed") is not False
            or registration.get("oracle_fields_accessed") is not False
            or registration.get("secrets_accessed") is not False
            or registration.get("content_sha256") != _sha256_json(registration_body)
            or registration.get("content_sha256") != row.get("content_sha256")
            or not isinstance(family_rows, list)
            or len(family_rows) != 1
        ):
            raise ValueError(f"neural-query native proof registration is invalid: {scope_id}")
        [family_row] = family_rows
        proof = family_row.get("proof") if isinstance(family_row, Mapping) else None
        if (
            not isinstance(proof, Mapping)
            or family_row.get("family") != NEURAL_QUERY_MOMENTS
            or proof.get("family") != NEURAL_QUERY_MOMENTS
            or proof.get("fit_semantics") != EXACT_INNER_REFIT
            or proof.get("heldout_labels_accessed") is not False
            or proof.get("oracle_fields_accessed") is not False
            or proof.get("secrets_accessed") is not False
            or tuple(map(int, registration.get("fit_row_ids") or ()))
            != tuple(map(int, expected["fit_row_ids"]))
            or tuple(map(int, registration.get("heldout_row_ids") or ()))
            != tuple(map(int, expected["heldout_row_ids"]))
        ):
            raise ValueError(f"neural-query native family proof is not scope-bound: {scope_id}")
        artifact_paths = {
            key: _validate_component_native_registration(root, family_row.get(key) or {})
            for key in (
                "evidence_payload",
                "native_execution_record",
                "native_fit_metadata",
                "model_artifact",
                "source_artifact",
                "heldout_moment_metadata",
            )
        }
        model_sha256 = native_artifact_sha256(artifact_paths["model_artifact"])
        source_sha256 = _sha256_file(artifact_paths["source_artifact"])
        metadata_sha256 = _sha256_file(artifact_paths["native_fit_metadata"])
        execution_sha256 = _sha256_file(artifact_paths["native_execution_record"])
        if (
            proof.get("model_artifact_sha256") != model_sha256
            or proof.get("source_artifact_sha256") != source_sha256
            or proof.get("native_fit_metadata_sha256") != metadata_sha256
            or proof.get("native_execution_record_sha256") != execution_sha256
        ):
            raise RuntimeError(f"neural-query native proof artifact changed: {scope_id}")
        source = json.loads(artifact_paths["source_artifact"].read_text(encoding="utf-8"))
        snapshot = validate_owned_discovery_snapshot(
            artifact_paths["model_artifact"] / "owned_snapshot",
            expected_cache_key=str(source.get("query_cache_key") or ""),
        )
        fit_rows = tuple(map(int, expected["fit_row_ids"]))
        binding = snapshot.get("binding") or {}
        if binding.get("treatment_sha256") != _float_hex_sha256(
            modeling_data.iloc[list(fit_rows)][treatment_column].to_numpy(dtype=float)
        ) or binding.get("outcome_sha256") != _float_hex_sha256(
            modeling_data.iloc[list(fit_rows)][outcome_column].to_numpy(dtype=float)
        ):
            raise ValueError(
                f"neural-query owned snapshot differs from canonical fit labels: {scope_id}"
            )
        _validate_neural_query_moment_artifact(
            artifact_paths["heldout_moment_metadata"],
            expected_scope_id=scope_id,
            expected_fit_row_ids=expected["fit_row_ids"],
            expected_heldout_row_ids=expected["heldout_row_ids"],
            expected_query_cache_key=str(source.get("query_cache_key") or ""),
            expected_snapshot_content_sha256=str(snapshot["content_sha256"]),
        )
        if reloaded_native_by_scope is not None:
            if expected_configuration_by_scope is None or scope_id not in (
                expected_configuration_by_scope
            ):
                raise ValueError("neural-query proof reload lacks expected configuration")
            evidence_payload = _read_json_object_reject_duplicates(
                artifact_paths["evidence_payload"],
                field_name=f"{scope_id} neural-query evidence payload",
            )
            rebound = bind_native_family_fit_proof(
                family=NEURAL_QUERY_MOMENTS,
                fit_semantics=EXACT_INNER_REFIT,
                outer_fold=int(expected["outer_fold"]),
                inner_fold=int(expected["inner_fold"]),
                split_scope_fingerprint=str(registration["split_scope_fingerprint"]),
                data_projection_sha256=str(registration["data_projection_sha256"]),
                fit_row_ids=fit_rows,
                heldout_row_ids=tuple(map(int, expected["heldout_row_ids"])),
                evidence_payload=evidence_payload,
                configuration=expected_configuration_by_scope[scope_id],
                native_fit_metadata_path=artifact_paths["native_fit_metadata"],
                native_execution_record_path=artifact_paths["native_execution_record"],
                model_artifact_path=artifact_paths["model_artifact"],
                source_artifact_path=artifact_paths["source_artifact"],
                model_artifact_semantics=(
                    "non-executable owned fitted query arrays, fit activations, and exact "
                    "ID/text-only heldout moment transforms"
                ),
            )
            if rebound.as_dict() != family_row.get("proof"):
                raise RuntimeError(f"neural-query native proof identity changed: {scope_id}")
            _record_reloaded_exact_inner_family(
                reloaded_native_by_scope,
                scope_id=scope_id,
                family=NEURAL_QUERY_MOMENTS,
                proof=rebound,
                evidence_payload=evidence_payload,
                artifact_paths=artifact_paths,
            )
    return copy.deepcopy(index)


def _bow_capture_family_bindings(
    capture: Mapping[str, Any],
    *,
    family: str,
) -> Mapping[str, Any]:
    if family not in PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS:
        raise ValueError("BoW capture binding requested for another family")
    inventory = capture.get("array_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("BoW capture has no numerical inventory")
    fold_rows = []
    for row in capture.get("folds") or ():
        if not isinstance(row, Mapping) or row.get("family") != family:
            continue
        keys = (
            "fit_target",
            "validation_target",
            "fit_sample_weight",
            "validation_prediction",
            "heldout_prediction",
        )
        numerical = {
            key: (None if row.get(key) is None else copy.deepcopy(inventory[str(row[key])]))
            for key in keys
        }
        learner = row.get("learner") or {}
        vectorizer = row.get("vectorizer")
        fold_rows.append(
            {
                "view_name": row.get("view_name"),
                "objective": row.get("objective"),
                "fold": row.get("fold"),
                "seed": row.get("seed"),
                "fit_row_ids": copy.deepcopy(row.get("fit_row_ids")),
                "validation_row_ids": copy.deepcopy(row.get("validation_row_ids")),
                "fit_row_fingerprint": row.get("fit_row_fingerprint"),
                "validation_row_fingerprint": row.get("validation_row_fingerprint"),
                "classification": row.get("classification"),
                "learner_kind": learner.get("kind"),
                "learner_class_name": learner.get("class_name"),
                "learner_parameters": copy.deepcopy(learner.get("parameters")),
                "classes": copy.deepcopy(learner.get("classes")),
                "vectorizer_kind": (None if vectorizer is None else vectorizer.get("kind")),
                "vectorizer_feature_count": (
                    0 if vectorizer is None else int(vectorizer.get("feature_count", 0))
                ),
                "numerical": numerical,
                "heldout_labels_accessed": False,
            }
        )
    full_objectives = (
        {"treatment_importance", "outcome_importance"}
        if family == BOW_NUISANCE
        else {"effect_weighted_r_importance"}
    )
    full_rows = []
    for row in capture.get("full_fit_models") or ():
        if not isinstance(row, Mapping) or row.get("objective") not in full_objectives:
            continue
        learner = row.get("learner") or {}
        full_rows.append(
            {
                "view_name": row.get("view_name"),
                "objective": row.get("objective"),
                "seed": row.get("seed"),
                "fit_row_fingerprint": row.get("fit_row_fingerprint"),
                "classification": row.get("classification"),
                "learner_kind": learner.get("kind"),
                "learner_class_name": learner.get("class_name"),
                "learner_parameters": copy.deepcopy(learner.get("parameters")),
                "classes": copy.deepcopy(learner.get("classes")),
                "target": copy.deepcopy(inventory[str(row["target"])]),
                "sample_weight": (
                    None
                    if row.get("sample_weight") is None
                    else copy.deepcopy(inventory[str(row["sample_weight"])])
                ),
                "fit_prediction": copy.deepcopy(inventory[str(row["fit_prediction"])]),
                "heldout_labels_accessed": False,
            }
        )
    scope_outputs = capture.get("scope_outputs")
    if not isinstance(scope_outputs, Mapping):
        raise ValueError("BoW capture has no scope outputs")
    common_names = {
        "treatment",
        "outcome",
        "ensemble_e_fit",
        "ensemble_m_fit",
        "ensemble_e_heldout",
        "ensemble_m_heldout",
    }
    if family == BOW_NUISANCE:
        selected_names = common_names | {
            name
            for name in scope_outputs
            if str(name).startswith(("view_", "nuisance_source_"))
            and not any(marker in str(name) for marker in ("pseudo", "weighted"))
        }
    else:
        selected_names = (
            common_names
            | {
                "y_residual",
                "t_residual",
                "pseudo_target",
                "r_weight",
            }
            | {
                name
                for name in scope_outputs
                if str(name).startswith("view_")
                and any(marker in str(name) for marker in ("pseudo", "weighted"))
            }
        )
    numerical_outputs = {}
    for name in sorted(selected_names):
        row = scope_outputs.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"BoW capture is missing family numerical output: {name}")
        numerical_outputs[name] = {
            "role": row.get("role"),
            **copy.deepcopy(inventory[str(row.get("array"))]),
        }
    return {
        "schema_version": BOW_NATIVE_CAPTURE_SCHEMA,
        "family": family,
        "e_clip": capture.get("e_clip"),
        "nuisance_folds": capture.get("nuisance_folds"),
        "effect_folds": capture.get("effect_folds"),
        "view_configs": copy.deepcopy(capture.get("view_configs")),
        "nuisance_source_names": copy.deepcopy(capture.get("nuisance_source_names")),
        "fold_states": fold_rows,
        "full_fit_states": full_rows,
        "scope_numerical_outputs": numerical_outputs,
        "heldout_labels_accessed": False,
    }


def _register_bow_native_family_proofs(
    *,
    component_root: Path,
    proof_directory: Path,
    scope_id: str,
    catalog: RoleNeutralEvidenceCatalog,
    capture_artifact_path: Path,
    source_artifact_path: Path,
    outer_fold: int,
    inner_fold: int,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    fit_texts: Sequence[str],
    heldout_texts: Sequence[str],
    fit_treatment: Sequence[float],
    fit_outcome: Sequence[float],
    split_scope_fingerprint: str,
    data_projection_sha256: str,
    configuration: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Register actual paired BoW nuisance/R-loss fits after safe replay."""

    if int(outer_fold) < 1 or int(inner_fold) < 1:
        raise ValueError("BoW native proof registration requires an exact-inner scope")
    root = Path(component_root).resolve(strict=True)
    fit_rows = tuple(map(int, fit_row_ids))
    heldout_rows = tuple(map(int, heldout_row_ids))
    source_path = _component_regular_file(
        root,
        source_artifact_path,
        field_name="BoW raw evidence sidecar",
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("schema_version") != STAGE1_RAW_EVIDENCE_SIDECAR_SCHEMA
        or source.get("scope_id") != str(scope_id)
        or int(source.get("outer_fold", 0)) != int(outer_fold)
        or int(source.get("inner_fold", 0)) != int(inner_fold)
        or source.get("fit_row_fingerprint") != row_set_fingerprint(fit_rows)
        or source.get("heldout_row_fingerprint") != row_set_fingerprint(heldout_rows)
        or source.get("prompt_grounding_allowed") is not False
    ):
        raise ValueError("BoW source artifact changed its exact-inner scope")
    model_registration = _component_native_artifact_registration(
        Path(capture_artifact_path),
        component_root=root,
    )
    if model_registration["kind"] != "directory":
        raise ValueError("BoW native capture must be one directory artifact")
    model_path = root / str(model_registration["relative_path"])
    capture = validate_bow_native_capture(
        model_path,
        expected_scope_id=scope_id,
        expected_fit_row_ids=fit_rows,
        expected_heldout_row_ids=heldout_rows,
        fit_texts=fit_texts,
        heldout_texts=heldout_texts,
        expected_fit_treatment=fit_treatment,
        expected_fit_outcome=fit_outcome,
    )
    if (
        capture.get("outer_fold") != int(outer_fold)
        or capture.get("inner_fold") != int(inner_fold)
        or configuration.get("scope_id") != str(scope_id)
        or configuration.get("heldout_label_policy") != "id_and_text_only"
        or configuration.get("capture_schema_version") != BOW_NATIVE_CAPTURE_SCHEMA
    ):
        raise ValueError("BoW native configuration or capture changed scope identity")
    proof_root = Path(proof_directory)
    if not proof_root.is_absolute():
        proof_root = root / proof_root
    if proof_root.exists() and proof_root.is_symlink():
        raise ValueError("BoW proof directory cannot be a symlink")
    proof_root.mkdir(parents=True, exist_ok=True)
    try:
        proof_root.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError("BoW proof directory escapes its component") from exc
    family_rows = []
    semantics = (
        "non-executable JSON/NPZ TfidfVectorizer and learner state with exact "
        "per-fold/full-fit replay and ID/text-only heldout transforms"
    )
    for family in PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS:
        evidence_payload, evidence_item_count = family_payload_from_catalog(
            catalog,
            family=family,
        )
        if int(evidence_item_count) < 1:
            raise RuntimeError(f"BoW native scope has no catalog evidence for {family}")
        payload_path = proof_root / f"{family}.evidence_payload.json"
        metadata_path = proof_root / f"{family}.fit_metadata.json"
        execution_path = proof_root / f"{family}.execution.json"
        _write_immutable_json(payload_path, evidence_payload)
        bindings = _bow_capture_family_bindings(capture, family=family)
        fit_metadata_body = {
            "schema_version": STAGE1_BOW_NATIVE_FIT_METADATA_SCHEMA,
            "family": family,
            "scope_id": str(scope_id),
            "outer_fold": int(outer_fold),
            "inner_fold": int(inner_fold),
            "fit_semantics": EXACT_INNER_REFIT,
            "fit_row_ids": list(fit_rows),
            "heldout_row_ids": list(heldout_rows),
            "fit_row_fingerprint": row_set_fingerprint(fit_rows),
            "heldout_row_fingerprint": row_set_fingerprint(heldout_rows),
            "split_scope_fingerprint": str(split_scope_fingerprint),
            "data_projection_sha256": str(data_projection_sha256),
            "capture_schema_version": BOW_NATIVE_CAPTURE_SCHEMA,
            "capture_content_sha256": capture["content_sha256"],
            "capture_artifact_sha256": model_registration["sha256"],
            "source_artifact_sha256": _sha256_file(source_path),
            "configuration": copy.deepcopy(dict(configuration)),
            "family_state_bindings": bindings,
            "heldout_columns_read": ["_oci_row_id", capture["text_column"]],
            "heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
            "secrets_accessed": False,
            "executable_checkpoint_retained": False,
            "joblib_or_pickle_loaded": False,
        }
        fit_metadata = {
            **fit_metadata_body,
            "content_sha256": _sha256_json(fit_metadata_body),
        }
        _write_immutable_json(metadata_path, fit_metadata)
        execution_record = native_family_execution_record(
            family=family,
            fit_semantics=EXACT_INNER_REFIT,
            outer_fold=int(outer_fold),
            inner_fold=int(inner_fold),
            split_scope_fingerprint=split_scope_fingerprint,
            data_projection_sha256=data_projection_sha256,
            fit_row_ids=fit_rows,
            heldout_row_ids=heldout_rows,
            evidence_payload=evidence_payload,
            configuration=configuration,
            native_fit_metadata_path=metadata_path,
            model_artifact_path=model_path,
            source_artifact_path=source_path,
            model_artifact_semantics=semantics,
        )
        _write_immutable_json(execution_path, execution_record)
        proof = bind_native_family_fit_proof(
            family=family,
            fit_semantics=EXACT_INNER_REFIT,
            outer_fold=int(outer_fold),
            inner_fold=int(inner_fold),
            split_scope_fingerprint=split_scope_fingerprint,
            data_projection_sha256=data_projection_sha256,
            fit_row_ids=fit_rows,
            heldout_row_ids=heldout_rows,
            evidence_payload=evidence_payload,
            configuration=configuration,
            native_fit_metadata_path=metadata_path,
            native_execution_record_path=execution_path,
            model_artifact_path=model_path,
            source_artifact_path=source_path,
            model_artifact_semantics=semantics,
        )
        proof.verify_artifact_bytes()
        family_rows.append(
            {
                "family": family,
                "evidence_item_count": int(evidence_item_count),
                "proof": proof.as_dict(),
                "evidence_payload": _component_file_registration(
                    payload_path,
                    component_root=root,
                ),
                "native_execution_record": _component_file_registration(
                    execution_path,
                    component_root=root,
                ),
                "native_fit_metadata": _component_file_registration(
                    metadata_path,
                    component_root=root,
                ),
                "model_artifact": copy.deepcopy(model_registration),
                "source_artifact": _component_file_registration(
                    source_path,
                    component_root=root,
                ),
            }
        )
    registration_body = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
        "scope_id": str(scope_id),
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "fit_row_ids": list(fit_rows),
        "heldout_row_ids": list(heldout_rows),
        "split_scope_fingerprint": str(split_scope_fingerprint),
        "data_projection_sha256": str(data_projection_sha256),
        "fit_semantics": EXACT_INNER_REFIT,
        "registered_families": list(PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS),
        "heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "family_proofs": family_rows,
    }
    registration = {
        **registration_body,
        "content_sha256": _sha256_json(registration_body),
    }
    registration_path = proof_root / "registration.json"
    _write_immutable_json(registration_path, registration)
    return {
        **registration,
        "registration": _component_file_registration(
            registration_path,
            component_root=root,
        ),
    }


def _validate_bow_native_family_proof_index(
    *,
    component_root: Path,
    index_registration: Mapping[str, Any],
    expected_inner_scopes: Mapping[str, Mapping[str, Any]],
    split_registry_content_sha256: str,
    modeling_data: pd.DataFrame,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    reloaded_native_by_scope: dict[str, dict[str, Mapping[str, Any]]] | None = None,
) -> Mapping[str, Any]:
    root = Path(component_root).resolve(strict=True)
    index_path = _validate_component_native_registration(root, index_registration)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in index.items() if key != "content_sha256"}
    scopes = index.get("scopes")
    if (
        index.get("schema_version") != STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA
        or index.get("split_registry_content_sha256") != split_registry_content_sha256
        or index.get("registered_families")
        != list(PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS)
        or index.get("executable_checkpoint_files_retained") is not False
        or index.get("content_sha256") != _sha256_json(body)
        or not isinstance(scopes, list)
        or int(index.get("exact_inner_scope_count", -1)) != len(scopes)
    ):
        raise ValueError("BoW native proof index has an invalid closed envelope")
    indexed = {str(row.get("scope_id")): row for row in scopes if isinstance(row, Mapping)}
    if len(indexed) != len(scopes) or set(indexed) != set(expected_inner_scopes):
        raise ValueError("BoW native proof index has incomplete exact-inner coverage")
    for scope_id, expected in expected_inner_scopes.items():
        row = indexed[scope_id]
        registration_path = _validate_component_native_registration(
            root,
            row.get("registration") or {},
        )
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        registration_body = {
            key: value for key, value in registration.items() if key != "content_sha256"
        }
        family_rows = registration.get("family_proofs")
        if (
            int(row.get("outer_fold", 0)) != int(expected["outer_fold"])
            or int(row.get("inner_fold", 0)) != int(expected["inner_fold"])
            or row.get("registered_families")
            != list(PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS)
            or registration.get("scope_id") != scope_id
            or int(registration.get("outer_fold", 0)) != int(expected["outer_fold"])
            or int(registration.get("inner_fold", 0)) != int(expected["inner_fold"])
            or registration.get("registered_families")
            != list(PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS)
            or registration.get("content_sha256") != _sha256_json(registration_body)
            or registration.get("content_sha256") != row.get("content_sha256")
            or tuple(map(int, registration.get("fit_row_ids") or ()))
            != tuple(map(int, expected["fit_row_ids"]))
            or tuple(map(int, registration.get("heldout_row_ids") or ()))
            != tuple(map(int, expected["heldout_row_ids"]))
            or registration.get("heldout_labels_accessed") is not False
            or not isinstance(family_rows, list)
            or [item.get("family") for item in family_rows]
            != list(PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS)
        ):
            raise ValueError(f"BoW native proof registration is invalid: {scope_id}")
        fit_rows = tuple(map(int, expected["fit_row_ids"]))
        heldout_rows = tuple(map(int, expected["heldout_row_ids"]))
        fit_texts = tuple(_normalize_texts(modeling_data.iloc[list(fit_rows)][text_column]))
        heldout_texts = tuple(_normalize_texts(modeling_data.iloc[list(heldout_rows)][text_column]))
        fit_treatment = tuple(
            modeling_data.iloc[list(fit_rows)][treatment_column].to_numpy(dtype=float)
        )
        fit_outcome = tuple(
            modeling_data.iloc[list(fit_rows)][outcome_column].to_numpy(dtype=float)
        )
        seen_model_path = None
        for family_row in family_rows:
            family = str(family_row["family"])
            paths = {
                key: _validate_component_native_registration(
                    root,
                    family_row.get(key) or {},
                )
                for key in (
                    "evidence_payload",
                    "native_execution_record",
                    "native_fit_metadata",
                    "model_artifact",
                    "source_artifact",
                )
            }
            if seen_model_path is not None and paths["model_artifact"] != seen_model_path:
                raise ValueError("paired BoW family proofs use different native captures")
            seen_model_path = paths["model_artifact"]
            capture = validate_bow_native_capture(
                paths["model_artifact"],
                expected_scope_id=scope_id,
                expected_fit_row_ids=fit_rows,
                expected_heldout_row_ids=heldout_rows,
                fit_texts=fit_texts,
                heldout_texts=heldout_texts,
                expected_fit_treatment=fit_treatment,
                expected_fit_outcome=fit_outcome,
            )
            metadata = json.loads(paths["native_fit_metadata"].read_text(encoding="utf-8"))
            metadata_body = {
                key: value for key, value in metadata.items() if key != "content_sha256"
            }
            if (
                metadata.get("schema_version") != STAGE1_BOW_NATIVE_FIT_METADATA_SCHEMA
                or metadata.get("family") != family
                or metadata.get("capture_content_sha256") != capture["content_sha256"]
                or metadata.get("content_sha256") != _sha256_json(metadata_body)
                or metadata.get("heldout_labels_accessed") is not False
                or metadata.get("family_state_bindings")
                != _bow_capture_family_bindings(capture, family=family)
            ):
                raise ValueError(f"BoW native fit metadata is invalid: {scope_id}/{family}")
            evidence_payload = json.loads(paths["evidence_payload"].read_text(encoding="utf-8"))
            rebound = bind_native_family_fit_proof(
                family=family,
                fit_semantics=EXACT_INNER_REFIT,
                outer_fold=int(expected["outer_fold"]),
                inner_fold=int(expected["inner_fold"]),
                split_scope_fingerprint=str(registration["split_scope_fingerprint"]),
                data_projection_sha256=str(registration["data_projection_sha256"]),
                fit_row_ids=fit_rows,
                heldout_row_ids=heldout_rows,
                evidence_payload=evidence_payload,
                configuration=metadata["configuration"],
                native_fit_metadata_path=paths["native_fit_metadata"],
                native_execution_record_path=paths["native_execution_record"],
                model_artifact_path=paths["model_artifact"],
                source_artifact_path=paths["source_artifact"],
                model_artifact_semantics=(
                    "non-executable JSON/NPZ TfidfVectorizer and learner state with "
                    "exact per-fold/full-fit replay and ID/text-only heldout transforms"
                ),
            )
            if rebound.as_dict() != family_row.get("proof"):
                raise RuntimeError(f"BoW native proof identity changed: {scope_id}/{family}")
            _record_reloaded_exact_inner_family(
                reloaded_native_by_scope,
                scope_id=scope_id,
                family=family,
                proof=rebound,
                evidence_payload=evidence_payload,
                artifact_paths=paths,
            )
    return copy.deepcopy(index)


def _embedding_capture_family_bindings(
    capture: Mapping[str, Any],
    *,
    family: str,
) -> Mapping[str, Any]:
    """Project the replayed embedding capture into one architecture binding."""

    if family not in PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS:
        raise ValueError("embedding capture binding requested for another family")
    inventory = capture.get("array_inventory")
    build = capture.get("build")
    evidence_inventory = capture.get("evidence_inventory")
    if (
        not isinstance(inventory, Mapping)
        or not isinstance(build, Mapping)
        or not isinstance(evidence_inventory, Mapping)
    ):
        raise ValueError("embedding capture lacks its closed numerical/evidence inventory")

    def numerical(key: Any) -> Mapping[str, Any]:
        name = str(key or "")
        row = inventory.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"embedding capture lacks numerical state: {name}")
        return copy.deepcopy(dict(row))

    common = {
        "schema_version": EMBEDDING_NATIVE_CAPTURE_SCHEMA,
        "family": family,
        "embedding_config": copy.deepcopy(capture.get("embedding_config")),
        "embedding_provider_identity": copy.deepcopy(capture.get("embedding_provider_identity")),
        "fit_cache_row_inventory": copy.deepcopy(capture.get("fit_cache_row_inventory")),
        "discovery_projection": copy.deepcopy(build.get("discovery_projection")),
        "residualize_columns_present": copy.deepcopy(build.get("residualize_columns_present")),
        "outcome": numerical(build.get("outcome")),
        "treatment": numerical(build.get("treatment")),
        "pseudo_target_names": copy.deepcopy(build.get("pseudo_target_names")),
        "pseudo_targets": [numerical(key) for key in (build.get("pseudo_target_arrays") or ())],
        "treatment_residuals": [numerical(key) for key in (build.get("t_resid_arrays") or ())],
        "importance_sha256": build.get("importance_sha256"),
        "heldout_columns_read": ["_oci_row_id"],
        "heldout_text_accessed": False,
        "heldout_labels_accessed": False,
    }
    if family == EMBEDDING_WHOLE_COHORT:
        return {
            **common,
            "native_evidence": copy.deepcopy(evidence_inventory.get("raw_embedding_evidence")),
            "scope": "all_exact_fit_rows",
        }
    if family == EMBEDDING_CLUSTERED:
        kmeans = capture.get("cluster_kmeans")
        svds = capture.get("cluster_svds")
        if not isinstance(kmeans, Mapping) or not isinstance(svds, list) or not svds:
            raise ValueError("clustered embedding capture lacks actual KMeans/SVD state")
        return {
            **common,
            "native_evidence": copy.deepcopy(evidence_inventory.get("raw_embedding_evidence")),
            "kmeans": {
                "fit_row_ids": copy.deepcopy(kmeans.get("fit_row_ids")),
                "parameters": copy.deepcopy(kmeans.get("parameters")),
                "usable_mask": numerical(kmeans.get("usable_mask")),
                "cluster_labels": numerical(kmeans.get("cluster_labels")),
                "cluster_centers": numerical(kmeans.get("cluster_centers")),
                "cluster_counts": numerical(kmeans.get("cluster_counts")),
                "n_iter": kmeans.get("n_iter"),
                "inertia": kmeans.get("inertia"),
            },
            "svds": [
                {
                    "family_key": row.get("family_key"),
                    "item_cluster_ids": copy.deepcopy(row.get("item_cluster_ids")),
                    "weighted_matrix": numerical(row.get("weighted_matrix")),
                    "singular_values": numerical(row.get("singular_values")),
                    "components": numerical(row.get("components")),
                }
                for row in svds
            ],
        }
    policy = capture.get("tfidf_training_scope_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("semantic-retrieval capture lacks its training-only policy")
    return {
        **common,
        "tfidf_training_scope_policy": copy.deepcopy(dict(policy)),
        "authoritative_full_scope_evidence": copy.deepcopy(
            evidence_inventory.get("semantic_full_scope_evidence")
        ),
        "model_partition_replay_canary": copy.deepcopy(
            evidence_inventory.get("semantic_model_replay_canary")
        ),
        "calibration_partition_replay_canary": copy.deepcopy(
            evidence_inventory.get("semantic_calibration_replay_canary")
        ),
        "partitions_select_or_drop_terms": False,
        "projection_vocabulary_max_features": None,
        "projection_output_limit": None,
    }


def _native_scope_text_projections(
    values: Sequence[Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return raw neural-model text and normalized sparse-model text."""

    raw = tuple(str(value) for value in values)
    return raw, tuple(_normalize_texts(raw))


def _canonical_embedding_scope_lineage(
    *,
    modeling_data: pd.DataFrame,
    outer_fold: int,
    inner_fold: int,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    embedding_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Recompute the canonical split, data projection, labels, and fit frame."""

    required_columns = {
        str(text_column),
        str(treatment_column),
        str(outcome_column),
    }
    missing = sorted(required_columns - set(modeling_data.columns))
    if missing:
        raise ValueError("embedding canonical lineage lacks columns: " + ", ".join(missing))
    fit_rows = tuple(map(int, fit_row_ids))
    heldout_rows = tuple(map(int, heldout_row_ids))
    if any(row < 0 or row >= len(modeling_data) for row in (*fit_rows, *heldout_rows)):
        raise ValueError("embedding canonical row IDs escape the modeling cohort")
    split = CanonicalInnerSplit(
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
    )
    fit_frame = modeling_data.iloc[list(fit_rows)]
    heldout_frame = modeling_data.iloc[list(heldout_rows)]
    if "_oci_row_id" in modeling_data.columns:
        if (
            tuple(map(int, fit_frame["_oci_row_id"].tolist())) != fit_rows
            or tuple(map(int, heldout_frame["_oci_row_id"].tolist())) != heldout_rows
        ):
            raise ValueError("embedding canonical row IDs differ from modeling-data positions")
    raw_fit_texts = tuple(str(value) for value in fit_frame[text_column].tolist())
    raw_heldout_texts = tuple(str(value) for value in heldout_frame[text_column].tolist())
    treatment = fit_frame[treatment_column].to_numpy(dtype=float)
    outcome = fit_frame[outcome_column].to_numpy(dtype=float)
    if not np.isfinite(treatment).all() or not np.isfinite(outcome).all():
        raise ValueError("embedding canonical fit labels must be finite")
    projection_sha256 = exact_inner_data_projection_sha256(
        fit_rows=tuple(
            Stage1FitRow(
                row_id=row_id,
                text=text,
                treatment=float(treatment[index]),
                outcome=float(outcome[index]),
            )
            for index, (row_id, text) in enumerate(zip(fit_rows, raw_fit_texts))
        ),
        heldout_rows=tuple(
            Stage1HeldoutRow(row_id=row_id, text=text)
            for row_id, text in zip(heldout_rows, raw_heldout_texts)
        ),
    )
    residual_columns = [
        str(column)
        for column in embedding_config.get("residualize_columns") or ()
        if str(column) in fit_frame.columns
    ]
    discovery_projection: dict[str, Any] = {"_oci_row_id": list(fit_rows)}
    for column in residual_columns:
        discovery_projection[column] = fit_frame[column].tolist()
    # Force the same strict JSON normalization used by immutable artifacts.
    discovery_projection = json.loads(_canonical_json(discovery_projection))
    return {
        "split_scope_fingerprint": split.scope_fingerprint,
        "data_projection_sha256": projection_sha256,
        # Embedding chunking consumes the exact source string.  BoW, HTR, and
        # matched-pair normalization is a separate architecture projection.
        "fit_texts": raw_fit_texts,
        "fit_treatment": treatment,
        "fit_outcome": outcome,
        "discovery_projection": discovery_projection,
    }


def _register_embedding_native_family_proofs(
    *,
    component_root: Path,
    proof_directory: Path,
    scope_id: str,
    catalog: RoleNeutralEvidenceCatalog,
    capture_artifact_path: Path,
    outer_fold: int,
    inner_fold: int,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    modeling_data: pd.DataFrame,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    embedding_provider: BoundSpentFrozenChunkEmbeddingProvider,
    split_scope_fingerprint: str,
    data_projection_sha256: str,
    configuration: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Register whole, clustered, and semantic evidence from one native fit."""

    if int(outer_fold) < 1 or int(inner_fold) < 1:
        raise ValueError("embedding native proof registration requires an exact-inner scope")
    root = Path(component_root).resolve(strict=True)
    fit_rows = tuple(map(int, fit_row_ids))
    heldout_rows = tuple(map(int, heldout_row_ids))
    configuration_fields = {
        "schema_version",
        "scope_id",
        "text_column",
        "treatment_column",
        "outcome_column",
        "outcome_type",
        "embedding_config",
        "capture_schema_version",
        "semantic_policy_schema_version",
        "tfidf_nested_calibration_folds",
        "heldout_label_policy",
        "seed",
        "split_registry_content_sha256",
    }
    if not isinstance(configuration, Mapping) or set(configuration) != configuration_fields:
        raise ValueError("embedding native configuration is not one closed envelope")
    embedding_configuration = configuration.get("embedding_config")
    if not isinstance(embedding_configuration, Mapping):
        raise ValueError("embedding native configuration has no embedding_config")
    canonical = _canonical_embedding_scope_lineage(
        modeling_data=modeling_data,
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        text_column=text_column,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        embedding_config=embedding_configuration,
    )
    if (
        str(split_scope_fingerprint) != canonical["split_scope_fingerprint"]
        or str(data_projection_sha256) != canonical["data_projection_sha256"]
    ):
        raise ValueError("embedding registration differs from canonical split/data lineage")
    model_registration = _component_native_artifact_registration(
        Path(capture_artifact_path),
        component_root=root,
    )
    if model_registration.get("kind") != "directory":
        raise ValueError("embedding native capture must be one directory artifact")
    model_path = root / str(model_registration["relative_path"])
    capture = validate_embedding_native_capture(
        model_path,
        embedding_provider=embedding_provider,
        fit_texts=canonical["fit_texts"],
        expected_fit_treatment=canonical["fit_treatment"],
        expected_fit_outcome=canonical["fit_outcome"],
        expected_discovery_projection=canonical["discovery_projection"],
        expected_scope_id=scope_id,
        expected_fit_row_ids=fit_rows,
        expected_heldout_row_ids=heldout_rows,
    )
    if (
        int(capture.get("outer_fold", 0)) != int(outer_fold)
        or int(capture.get("inner_fold", 0)) != int(inner_fold)
        or configuration.get("schema_version") != STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA
        or configuration.get("scope_id") != str(scope_id)
        or configuration.get("text_column") != str(text_column)
        or configuration.get("treatment_column") != str(treatment_column)
        or configuration.get("outcome_column") != str(outcome_column)
        or configuration.get("capture_schema_version") != EMBEDDING_NATIVE_CAPTURE_SCHEMA
        or configuration.get("semantic_policy_schema_version")
        != SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA
        or configuration.get("heldout_label_policy") != "id_only_no_transform"
        or _HEX_SHA256.fullmatch(str(configuration.get("split_registry_content_sha256") or ""))
        is None
        or configuration.get("outcome_type") != capture.get("outcome_type")
        or configuration.get("embedding_config") != capture.get("embedding_config")
        or isinstance(configuration.get("seed"), bool)
        or configuration.get("seed") != capture.get("seed")
        or int(configuration.get("tfidf_nested_calibration_folds", 0))
        != int(
            (capture.get("tfidf_training_scope_policy") or {}).get(
                "configured_fold_count",
                0,
            )
        )
    ):
        raise ValueError("embedding native configuration or capture changed scope identity")
    proof_root = Path(proof_directory)
    if not proof_root.is_absolute():
        proof_root = root / proof_root
    if proof_root.exists():
        raise RuntimeError("embedding native proof directory already exists")
    proof_root.mkdir(parents=True, exist_ok=False)
    try:
        proof_root.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError("embedding proof directory escapes its component") from exc

    source_names = {
        EMBEDDING_WHOLE_COHORT: "raw_embedding_evidence.json",
        EMBEDDING_CLUSTERED: "raw_embedding_evidence.json",
        TFIDF_SEMANTIC_RETRIEVAL: "semantic_full_scope_evidence.json",
    }
    semantics = {
        EMBEDDING_WHOLE_COHORT: (
            "fit-only frozen-cache embedding directions, retrieval witnesses, and exact "
            "generator replay with no heldout transform"
        ),
        EMBEDDING_CLUSTERED: (
            "fit-only frozen-cache clustered contrasts with exact KMeans/SVD numerical "
            "state and generator replay"
        ),
        TFIDF_SEMANTIC_RETRIEVAL: (
            "exhaustive uncapped TF-IDF projection over all exact-fit frozen retrieval "
            "tails; label-free partitions are replay canaries only"
        ),
    }
    family_rows: list[dict[str, Any]] = []
    for family in PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS:
        evidence_payload, evidence_item_count = family_payload_from_catalog(
            catalog,
            family=family,
        )
        if int(evidence_item_count) < 1:
            raise RuntimeError(f"embedding native scope has no catalog evidence for {family}")
        source_path = _component_regular_file(
            root,
            model_path / source_names[family],
            field_name=f"{family} native evidence source",
        )
        payload_path = proof_root / f"{family}.evidence_payload.json"
        metadata_path = proof_root / f"{family}.fit_metadata.json"
        execution_path = proof_root / f"{family}.execution.json"
        _write_immutable_json(payload_path, evidence_payload)
        fit_metadata_body: dict[str, Any] = {
            "schema_version": STAGE1_EMBEDDING_NATIVE_FIT_METADATA_SCHEMA,
            "family": family,
            "scope_id": str(scope_id),
            "outer_fold": int(outer_fold),
            "inner_fold": int(inner_fold),
            "fit_semantics": EXACT_INNER_REFIT,
            "fit_row_ids": list(fit_rows),
            "heldout_row_ids": list(heldout_rows),
            "fit_row_order_fingerprint": row_order_fingerprint(fit_rows),
            "heldout_row_order_fingerprint": row_order_fingerprint(heldout_rows),
            "split_scope_fingerprint": str(split_scope_fingerprint),
            "data_projection_sha256": str(data_projection_sha256),
            "capture_schema_version": EMBEDDING_NATIVE_CAPTURE_SCHEMA,
            "seed": capture["seed"],
            "capture_content_sha256": capture["content_sha256"],
            "capture_artifact_sha256": model_registration["sha256"],
            "source_artifact_sha256": _sha256_file(source_path),
            "configuration": copy.deepcopy(dict(configuration)),
            "family_state_bindings": _embedding_capture_family_bindings(
                capture,
                family=family,
            ),
            "registered_heldout_columns_read": ["_oci_row_id"],
            "registered_heldout_labels_accessed": False,
            "registered_heldout_text_accessed": False,
            "registered_heldout_transform_performed": False,
            "oracle_fields_accessed": False,
            "secrets_accessed": False,
            "executable_checkpoint_retained": False,
            "joblib_or_pickle_loaded": False,
        }
        if family == TFIDF_SEMANTIC_RETRIEVAL:
            fit_metadata_body["tfidf_training_scope_policy"] = copy.deepcopy(
                capture["tfidf_training_scope_policy"]
            )
        fit_metadata = {
            **fit_metadata_body,
            "content_sha256": _sha256_json(fit_metadata_body),
        }
        _write_immutable_json(metadata_path, fit_metadata)
        execution_record = native_family_execution_record(
            family=family,
            fit_semantics=EXACT_INNER_REFIT,
            outer_fold=int(outer_fold),
            inner_fold=int(inner_fold),
            split_scope_fingerprint=split_scope_fingerprint,
            data_projection_sha256=data_projection_sha256,
            fit_row_ids=fit_rows,
            heldout_row_ids=heldout_rows,
            evidence_payload=evidence_payload,
            configuration=configuration,
            native_fit_metadata_path=metadata_path,
            model_artifact_path=model_path,
            source_artifact_path=source_path,
            model_artifact_semantics=semantics[family],
        )
        _write_immutable_json(execution_path, execution_record)
        proof = bind_native_family_fit_proof(
            family=family,
            fit_semantics=EXACT_INNER_REFIT,
            outer_fold=int(outer_fold),
            inner_fold=int(inner_fold),
            split_scope_fingerprint=split_scope_fingerprint,
            data_projection_sha256=data_projection_sha256,
            fit_row_ids=fit_rows,
            heldout_row_ids=heldout_rows,
            evidence_payload=evidence_payload,
            configuration=configuration,
            native_fit_metadata_path=metadata_path,
            native_execution_record_path=execution_path,
            model_artifact_path=model_path,
            source_artifact_path=source_path,
            model_artifact_semantics=semantics[family],
        )
        proof.verify_artifact_bytes()
        family_rows.append(
            {
                "family": family,
                "evidence_item_count": int(evidence_item_count),
                "proof": proof.as_dict(),
                "evidence_payload": _component_file_registration(
                    payload_path,
                    component_root=root,
                ),
                "native_execution_record": _component_file_registration(
                    execution_path,
                    component_root=root,
                ),
                "native_fit_metadata": _component_file_registration(
                    metadata_path,
                    component_root=root,
                ),
                "model_artifact": copy.deepcopy(model_registration),
                "source_artifact": _component_file_registration(
                    source_path,
                    component_root=root,
                ),
            }
        )
    registration_body = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
        "scope_id": str(scope_id),
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "fit_row_ids": list(fit_rows),
        "heldout_row_ids": list(heldout_rows),
        "split_scope_fingerprint": str(split_scope_fingerprint),
        "data_projection_sha256": str(data_projection_sha256),
        "fit_semantics": EXACT_INNER_REFIT,
        "registered_families": list(PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS),
        "heldout_labels_accessed": False,
        "heldout_text_accessed": False,
        "heldout_transform_performed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "family_proofs": family_rows,
    }
    registration = {
        **registration_body,
        "content_sha256": _sha256_json(registration_body),
    }
    registration_path = proof_root / "registration.json"
    _write_immutable_json(registration_path, registration)
    return {
        **registration,
        "registration": _component_file_registration(
            registration_path,
            component_root=root,
        ),
    }


def _validate_embedding_native_family_proof_index(
    *,
    component_root: Path,
    index_registration: Mapping[str, Any],
    expected_inner_scopes: Mapping[str, Mapping[str, Any]],
    split_registry_content_sha256: str,
    modeling_data: pd.DataFrame,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    embedding_cache: SpentOnlyFrozenChunkEmbeddingCache,
    reloaded_native_by_scope: dict[str, dict[str, Mapping[str, Any]]] | None = None,
) -> Mapping[str, Any]:
    """Replay every registered embedding fit before accepting its proof index."""

    root = Path(component_root).resolve(strict=True)
    file_registration_fields = {"relative_path", "size", "sha256"}
    directory_registration_fields = {
        "relative_path",
        "kind",
        "file_count",
        "size",
        "sha256",
    }
    if not isinstance(index_registration, Mapping) or set(index_registration) != (
        file_registration_fields
    ):
        raise ValueError("embedding native proof index registration is not closed")
    index_path = _validate_component_native_registration(root, index_registration)
    index = _read_json_object_reject_duplicates(
        index_path,
        field_name="embedding native proof index",
    )
    body = {key: value for key, value in index.items() if key != "content_sha256"}
    scopes = index.get("scopes")
    registered = list(PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS)
    index_fields = {
        "schema_version",
        "split_registry_content_sha256",
        "registered_families",
        "exact_inner_scope_count",
        "executable_checkpoint_files_retained",
        "scopes",
        "content_sha256",
    }
    if (
        set(index) != index_fields
        or index.get("schema_version") != STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA
        or index.get("split_registry_content_sha256") != split_registry_content_sha256
        or index.get("registered_families") != registered
        or index.get("executable_checkpoint_files_retained") is not False
        or index.get("content_sha256") != _sha256_json(body)
        or not isinstance(scopes, list)
        or int(index.get("exact_inner_scope_count", -1)) != len(scopes)
    ):
        raise ValueError("embedding native proof index has an invalid closed envelope")
    scope_row_fields = {
        "scope_id",
        "outer_fold",
        "inner_fold",
        "registered_families",
        "content_sha256",
        "registration",
    }
    indexed = {
        str(row.get("scope_id")): row
        for row in scopes
        if isinstance(row, Mapping) and set(row) == scope_row_fields
    }
    if len(indexed) != len(scopes) or set(indexed) != set(expected_inner_scopes):
        raise ValueError("embedding native proof index has incomplete exact-inner coverage")
    semantics = {
        EMBEDDING_WHOLE_COHORT: (
            "fit-only frozen-cache embedding directions, retrieval witnesses, and exact "
            "generator replay with no heldout transform"
        ),
        EMBEDDING_CLUSTERED: (
            "fit-only frozen-cache clustered contrasts with exact KMeans/SVD numerical "
            "state and generator replay"
        ),
        TFIDF_SEMANTIC_RETRIEVAL: (
            "exhaustive uncapped TF-IDF projection over all exact-fit frozen retrieval "
            "tails; label-free partitions are replay canaries only"
        ),
    }
    registration_fields = {
        "schema_version",
        "scope_id",
        "outer_fold",
        "inner_fold",
        "fit_row_ids",
        "heldout_row_ids",
        "split_scope_fingerprint",
        "data_projection_sha256",
        "fit_semantics",
        "registered_families",
        "heldout_labels_accessed",
        "heldout_text_accessed",
        "heldout_transform_performed",
        "oracle_fields_accessed",
        "secrets_accessed",
        "family_proofs",
        "content_sha256",
    }
    family_row_fields = {
        "family",
        "evidence_item_count",
        "proof",
        "evidence_payload",
        "native_execution_record",
        "native_fit_metadata",
        "model_artifact",
        "source_artifact",
    }
    configuration_fields = {
        "schema_version",
        "scope_id",
        "text_column",
        "treatment_column",
        "outcome_column",
        "outcome_type",
        "embedding_config",
        "capture_schema_version",
        "semantic_policy_schema_version",
        "tfidf_nested_calibration_folds",
        "heldout_label_policy",
        "seed",
        "split_registry_content_sha256",
    }
    metadata_fields = {
        "schema_version",
        "family",
        "scope_id",
        "outer_fold",
        "inner_fold",
        "fit_semantics",
        "fit_row_ids",
        "heldout_row_ids",
        "fit_row_order_fingerprint",
        "heldout_row_order_fingerprint",
        "split_scope_fingerprint",
        "data_projection_sha256",
        "capture_schema_version",
        "seed",
        "capture_content_sha256",
        "capture_artifact_sha256",
        "source_artifact_sha256",
        "configuration",
        "family_state_bindings",
        "registered_heldout_columns_read",
        "registered_heldout_labels_accessed",
        "registered_heldout_text_accessed",
        "registered_heldout_transform_performed",
        "oracle_fields_accessed",
        "secrets_accessed",
        "executable_checkpoint_retained",
        "joblib_or_pickle_loaded",
        "content_sha256",
    }
    for scope_id, expected in expected_inner_scopes.items():
        row = indexed[scope_id]
        registration_descriptor = row.get("registration")
        if (
            not isinstance(registration_descriptor, Mapping)
            or set(registration_descriptor) != file_registration_fields
        ):
            raise ValueError(f"embedding native proof registration descriptor is open: {scope_id}")
        registration_path = _validate_component_native_registration(
            root,
            registration_descriptor,
        )
        registration = _read_json_object_reject_duplicates(
            registration_path,
            field_name=f"embedding native proof registration {scope_id}",
        )
        registration_body = {
            key: value for key, value in registration.items() if key != "content_sha256"
        }
        family_rows = registration.get("family_proofs")
        fit_rows = tuple(map(int, expected["fit_row_ids"]))
        heldout_rows = tuple(map(int, expected["heldout_row_ids"]))
        if (
            set(registration) != registration_fields
            or int(row.get("outer_fold", 0)) != int(expected["outer_fold"])
            or int(row.get("inner_fold", 0)) != int(expected["inner_fold"])
            or row.get("registered_families") != registered
            or row.get("content_sha256") != registration.get("content_sha256")
            or registration.get("schema_version") != STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA
            or registration.get("scope_id") != scope_id
            or int(registration.get("outer_fold", 0)) != int(expected["outer_fold"])
            or int(registration.get("inner_fold", 0)) != int(expected["inner_fold"])
            or registration.get("fit_semantics") != EXACT_INNER_REFIT
            or registration.get("registered_families") != registered
            or registration.get("content_sha256") != _sha256_json(registration_body)
            or tuple(map(int, registration.get("fit_row_ids") or ())) != fit_rows
            or tuple(map(int, registration.get("heldout_row_ids") or ())) != heldout_rows
            or registration.get("heldout_labels_accessed") is not False
            or registration.get("heldout_text_accessed") is not False
            or registration.get("heldout_transform_performed") is not False
            or registration.get("oracle_fields_accessed") is not False
            or registration.get("secrets_accessed") is not False
            or not isinstance(family_rows, list)
            or any(not isinstance(item, Mapping) for item in (family_rows or ()))
            or [item.get("family") for item in family_rows] != registered
        ):
            raise ValueError(f"embedding native proof registration is invalid: {scope_id}")
        canonical: Mapping[str, Any] | None = None
        canonical_configuration: Mapping[str, Any] | None = None
        provider: BoundSpentFrozenChunkEmbeddingProvider | None = None
        seen_model_path: Path | None = None
        seen_capture: Mapping[str, Any] | None = None
        for family_row in family_rows:
            if not isinstance(family_row, Mapping) or set(family_row) != family_row_fields:
                raise ValueError(f"embedding family proof row is not closed: {scope_id}")
            family = str(family_row["family"])
            if (
                family not in registered
                or isinstance(family_row.get("evidence_item_count"), bool)
                or int(family_row.get("evidence_item_count", 0)) < 1
            ):
                raise ValueError(f"embedding family proof row is invalid: {scope_id}/{family}")
            artifact_descriptors = {
                key: family_row.get(key)
                for key in (
                    "evidence_payload",
                    "native_execution_record",
                    "native_fit_metadata",
                    "model_artifact",
                    "source_artifact",
                )
            }
            for key, descriptor in artifact_descriptors.items():
                expected_fields = (
                    directory_registration_fields
                    if key == "model_artifact"
                    else file_registration_fields
                )
                if not isinstance(descriptor, Mapping) or set(descriptor) != expected_fields:
                    raise ValueError(
                        f"embedding {key} registration is not closed: {scope_id}/{family}"
                    )
            paths = {
                key: _validate_component_native_registration(
                    root,
                    descriptor,
                )
                for key, descriptor in artifact_descriptors.items()
            }
            if seen_model_path is not None and paths["model_artifact"] != seen_model_path:
                raise ValueError("embedding family proofs use different native captures")
            seen_model_path = paths["model_artifact"]
            metadata = _read_json_object_reject_duplicates(
                paths["native_fit_metadata"],
                field_name=f"embedding native fit metadata {scope_id}/{family}",
            )
            required_metadata_fields = set(metadata_fields)
            if family == TFIDF_SEMANTIC_RETRIEVAL:
                required_metadata_fields.add("tfidf_training_scope_policy")
            configuration = metadata.get("configuration")
            embedding_configuration = (
                configuration.get("embedding_config")
                if isinstance(configuration, Mapping)
                else None
            )
            if (
                set(metadata) != required_metadata_fields
                or not isinstance(configuration, Mapping)
                or set(configuration) != configuration_fields
                or not isinstance(embedding_configuration, Mapping)
                or configuration.get("schema_version")
                != STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA
                or configuration.get("scope_id") != scope_id
                or configuration.get("text_column") != str(text_column)
                or configuration.get("treatment_column") != str(treatment_column)
                or configuration.get("outcome_column") != str(outcome_column)
                or configuration.get("capture_schema_version") != EMBEDDING_NATIVE_CAPTURE_SCHEMA
                or configuration.get("semantic_policy_schema_version")
                != SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA
                or configuration.get("heldout_label_policy") != "id_only_no_transform"
                or configuration.get("split_registry_content_sha256")
                != split_registry_content_sha256
            ):
                raise ValueError(
                    f"embedding native fit configuration is invalid: {scope_id}/{family}"
                )
            if canonical is None:
                canonical = _canonical_embedding_scope_lineage(
                    modeling_data=modeling_data,
                    outer_fold=int(expected["outer_fold"]),
                    inner_fold=int(expected["inner_fold"]),
                    fit_row_ids=fit_rows,
                    heldout_row_ids=heldout_rows,
                    text_column=text_column,
                    treatment_column=treatment_column,
                    outcome_column=outcome_column,
                    embedding_config=embedding_configuration,
                )
                canonical_configuration = copy.deepcopy(dict(configuration))
                if (
                    registration.get("split_scope_fingerprint")
                    != canonical["split_scope_fingerprint"]
                    or registration.get("data_projection_sha256")
                    != canonical["data_projection_sha256"]
                ):
                    raise ValueError(
                        f"embedding registration differs from canonical data: {scope_id}"
                    )
                provider = embedding_cache.bind_spent(
                    fit_rows,
                    canonical["fit_texts"],
                )
            elif configuration != canonical_configuration:
                raise ValueError("embedding family proofs changed their shared configuration")
            assert canonical is not None and provider is not None
            capture = validate_embedding_native_capture(
                paths["model_artifact"],
                embedding_provider=provider,
                fit_texts=canonical["fit_texts"],
                expected_fit_treatment=canonical["fit_treatment"],
                expected_fit_outcome=canonical["fit_outcome"],
                expected_discovery_projection=canonical["discovery_projection"],
                expected_scope_id=scope_id,
                expected_fit_row_ids=fit_rows,
                expected_heldout_row_ids=heldout_rows,
            )
            if seen_capture is not None and capture != seen_capture:
                raise RuntimeError("embedding family replay identities differ")
            seen_capture = capture
            expected_source_name = (
                "semantic_full_scope_evidence.json"
                if family == TFIDF_SEMANTIC_RETRIEVAL
                else "raw_embedding_evidence.json"
            )
            if paths["source_artifact"] != paths["model_artifact"] / expected_source_name:
                raise ValueError(f"embedding source artifact is wrong for {scope_id}/{family}")
            _read_json_object_reject_duplicates(
                paths["source_artifact"],
                field_name=f"embedding source evidence {scope_id}/{family}",
            )
            _read_json_object_reject_duplicates(
                paths["native_execution_record"],
                field_name=f"embedding native execution record {scope_id}/{family}",
            )
            metadata_body = {
                key: value for key, value in metadata.items() if key != "content_sha256"
            }
            if (
                metadata.get("schema_version") != STAGE1_EMBEDDING_NATIVE_FIT_METADATA_SCHEMA
                or metadata.get("family") != family
                or metadata.get("scope_id") != scope_id
                or int(metadata.get("outer_fold", 0)) != int(expected["outer_fold"])
                or int(metadata.get("inner_fold", 0)) != int(expected["inner_fold"])
                or metadata.get("fit_semantics") != EXACT_INNER_REFIT
                or tuple(map(int, metadata.get("fit_row_ids") or ())) != fit_rows
                or tuple(map(int, metadata.get("heldout_row_ids") or ())) != heldout_rows
                or metadata.get("fit_row_order_fingerprint") != row_order_fingerprint(fit_rows)
                or metadata.get("heldout_row_order_fingerprint")
                != row_order_fingerprint(heldout_rows)
                or metadata.get("split_scope_fingerprint") != canonical["split_scope_fingerprint"]
                or metadata.get("data_projection_sha256") != canonical["data_projection_sha256"]
                or metadata.get("capture_schema_version") != EMBEDDING_NATIVE_CAPTURE_SCHEMA
                or metadata.get("capture_content_sha256") != capture["content_sha256"]
                or metadata.get("seed") != capture.get("seed")
                or configuration.get("seed") != capture.get("seed")
                or configuration.get("outcome_type") != capture.get("outcome_type")
                or configuration.get("embedding_config") != capture.get("embedding_config")
                or int(configuration.get("tfidf_nested_calibration_folds", 0))
                != int(
                    (capture.get("tfidf_training_scope_policy") or {}).get(
                        "configured_fold_count",
                        0,
                    )
                )
                or metadata.get("capture_artifact_sha256")
                != native_artifact_sha256(paths["model_artifact"])
                or metadata.get("source_artifact_sha256") != _sha256_file(paths["source_artifact"])
                or metadata.get("family_state_bindings")
                != _embedding_capture_family_bindings(capture, family=family)
                or metadata.get("registered_heldout_columns_read") != ["_oci_row_id"]
                or metadata.get("registered_heldout_labels_accessed") is not False
                or metadata.get("registered_heldout_text_accessed") is not False
                or metadata.get("registered_heldout_transform_performed") is not False
                or metadata.get("oracle_fields_accessed") is not False
                or metadata.get("secrets_accessed") is not False
                or metadata.get("executable_checkpoint_retained") is not False
                or metadata.get("joblib_or_pickle_loaded") is not False
                or metadata.get("content_sha256") != _sha256_json(metadata_body)
            ):
                raise ValueError(f"embedding native fit metadata is invalid: {scope_id}/{family}")
            if family == TFIDF_SEMANTIC_RETRIEVAL and metadata.get(
                "tfidf_training_scope_policy"
            ) != capture.get("tfidf_training_scope_policy"):
                raise ValueError("semantic retrieval policy changed after native capture")
            evidence_payload = _read_json_object_reject_duplicates(
                paths["evidence_payload"],
                field_name=f"embedding evidence payload {scope_id}/{family}",
            )
            rebound = bind_native_family_fit_proof(
                family=family,
                fit_semantics=EXACT_INNER_REFIT,
                outer_fold=int(expected["outer_fold"]),
                inner_fold=int(expected["inner_fold"]),
                split_scope_fingerprint=str(canonical["split_scope_fingerprint"]),
                data_projection_sha256=str(canonical["data_projection_sha256"]),
                fit_row_ids=fit_rows,
                heldout_row_ids=heldout_rows,
                evidence_payload=evidence_payload,
                configuration=configuration,
                native_fit_metadata_path=paths["native_fit_metadata"],
                native_execution_record_path=paths["native_execution_record"],
                model_artifact_path=paths["model_artifact"],
                source_artifact_path=paths["source_artifact"],
                model_artifact_semantics=semantics[family],
            )
            if rebound.as_dict() != family_row.get("proof"):
                raise RuntimeError(f"embedding native proof identity changed: {scope_id}/{family}")
            _record_reloaded_exact_inner_family(
                reloaded_native_by_scope,
                scope_id=scope_id,
                family=family,
                proof=rebound,
                evidence_payload=evidence_payload,
                artifact_paths=paths,
            )
    return copy.deepcopy(index)


def _cumulative_spent_request_from_modeling_data(
    *,
    family: str,
    modeling_data: pd.DataFrame,
    request_sha256: str,
    schedule_sha256: str,
    scope_id: str,
    outer_fold: int,
    context_epoch: int,
    provider_inner_fold: int,
    split_scope_fingerprint: str,
    spent_row_ids: Sequence[int],
    sealed_row_ids: Sequence[int],
    text_column: str,
    treatment_column: str,
    outcome_column: str,
) -> CumulativeSpentStage1FamilyRequest:
    """Project the only labeled rows a cumulative native producer may receive."""

    required = {str(text_column), str(treatment_column), str(outcome_column)}
    missing = sorted(required - set(modeling_data.columns))
    if missing:
        raise ValueError("cumulative native modeling data lacks columns: " + ", ".join(missing))
    spent = tuple(map(int, spent_row_ids))
    sealed = tuple(map(int, sealed_row_ids))
    if not spent or not sealed or set(spent) & set(sealed):
        raise ValueError("cumulative native spent/sealed rows must be nonempty and disjoint")
    if any(row < 0 or row >= len(modeling_data) for row in (*spent, *sealed)):
        raise ValueError("cumulative native row IDs escape the modeling cohort")
    if "_oci_row_id" in modeling_data.columns:
        observed = tuple(
            map(
                int,
                modeling_data.iloc[list((*spent, *sealed))]["_oci_row_id"].tolist(),
            )
        )
        if observed != (*spent, *sealed):
            raise ValueError("cumulative native row IDs differ from modeling-data positions")
    fit_rows = tuple(
        Stage1FitRow(
            row_id=row_id,
            text=str(modeling_data.iloc[row_id][text_column]),
            treatment=float(modeling_data.iloc[row_id][treatment_column]),
            outcome=float(modeling_data.iloc[row_id][outcome_column]),
        )
        for row_id in spent
    )
    projection_sha256 = cumulative_spent_data_projection_sha256(
        outer_fold=int(outer_fold),
        context_epoch=int(context_epoch),
        spent_rows=fit_rows,
        sealed_row_ids=sealed,
    )
    return CumulativeSpentStage1FamilyRequest(
        family=str(family),
        request_sha256=str(request_sha256),
        schedule_sha256=str(schedule_sha256),
        scope_id=str(scope_id),
        outer_fold=int(outer_fold),
        context_epoch=int(context_epoch),
        provider_inner_fold=int(provider_inner_fold),
        split_scope_fingerprint=str(split_scope_fingerprint),
        data_projection_sha256=projection_sha256,
        spent_rows=fit_rows,
        sealed_row_ids=sealed,
    )


def _cumulative_request_for_family(
    request: CumulativeSpentStage1FamilyRequest,
    *,
    family: str,
) -> CumulativeSpentStage1FamilyRequest:
    return CumulativeSpentStage1FamilyRequest(
        family=family,
        request_sha256=request.request_sha256,
        schedule_sha256=request.schedule_sha256,
        scope_id=request.scope_id,
        outer_fold=request.outer_fold,
        context_epoch=request.context_epoch,
        provider_inner_fold=request.provider_inner_fold,
        split_scope_fingerprint=request.split_scope_fingerprint,
        data_projection_sha256=request.data_projection_sha256,
        spent_rows=request.spent_rows,
        sealed_row_ids=request.sealed_row_ids,
    )


def _canonical_cumulative_spent_schedule(
    registry: Mapping[str, Any],
) -> Any:
    """Rebuild the one wrapper-authoritative hierarchy spent schedule."""

    exact_registry = _canonical_exact_registry_from_wrapper(registry)
    review_rounds = int(exact_registry.inner_fold_count) - 3
    if review_rounds < 1:
        raise ValueError(
            "cumulative hierarchy emission requires at least four canonical inner "
            "partitions: three initially spent plus one review gate"
        )
    # Imported lazily because the handoff module imports this module's stable
    # hashing helpers.  No wrapper-local partitioner is permitted here.
    from .production_stage1_hierarchy_handoff import CanonicalHierarchySpentSchedule

    return CanonicalHierarchySpentSchedule.build(
        registry=exact_registry,
        review_rounds=review_rounds,
    )


def _hierarchy_spent_evidence_contract(
    *,
    registry: Mapping[str, Any],
    config: AppliedInferenceConfig,
    hierarchical_discovery_contract_identity_sha256: str,
) -> Mapping[str, Any]:
    """Seal the one canonical hierarchy schedule into the immutable request."""

    from .production_stage1_hierarchy_handoff import (
        STAGE1_HIERARCHY_SPENT_CONTRACT_SCHEMA,
    )

    schedule = _canonical_cumulative_spent_schedule(registry)
    review_rounds = int(schedule.review_rounds)
    interaction_inner_folds = int(
        config.architecture.explicit_feature_forest.interaction_inner_folds
    )
    tfidf_nested_calibration_folds = int(
        config.architecture.multi_model_forest.tfidf_nested_calibration_folds
    )
    if interaction_inner_folds < 2 or tfidf_nested_calibration_folds < 2:
        raise ValueError(
            "hierarchy interaction and TF-IDF nested-calibration folds must each be at least two"
        )
    return {
        "schema_version": STAGE1_HIERARCHY_SPENT_CONTRACT_SCHEMA,
        "review_rounds": review_rounds,
        "partition_authority": ("canonical_stage1_inner_heldout_partitions_in_registry_order"),
        "initial_spent_partition_count": 3,
        "canonical_hierarchy_partition_count": review_rounds + 3,
        "interaction_inner_folds": interaction_inner_folds,
        "tfidf_nested_calibration_folds": tfidf_nested_calibration_folds,
        "fold_domains_are_distinct": True,
        "required_families": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
        "hierarchical_discovery_contract_identity_sha256": str(
            hierarchical_discovery_contract_identity_sha256
        ),
        "schedule_sha256": schedule.schedule_sha256,
        "component_emitted_catalogs_and_proofs_required": True,
        "independent_runtime_stage1_refit_allowed": False,
        "manual_digest_approval_required": False,
    }


def _register_cumulative_spent_embedding_scope(
    *,
    component_root: Path,
    proof_directory: Path,
    requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    replay_canary: CumulativeSpentReplayCanary,
    emissions: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind the live shared-embedding fit and persist its three family views."""

    families = tuple(PRODUCTION_CUMULATIVE_EMBEDDING_NATIVE_FAMILY_ADAPTERS)
    if (
        set(families) != set(CUMULATIVE_SPENT_EMBEDDING_FAMILIES)
        or set(requests) != set(families)
        or set(emissions) != set(families)
    ):
        raise ValueError("cumulative embedding registration requires exactly three families")
    root = Path(component_root).resolve(strict=True)
    proof_root = Path(proof_directory)
    if not proof_root.is_absolute():
        proof_root = root / proof_root
    if proof_root.exists() or proof_root.is_symlink():
        raise RuntimeError("cumulative embedding proof directory must be new")
    proof_root.mkdir(parents=True, exist_ok=False)
    try:
        proof_root.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError("cumulative embedding proof directory escapes its component") from exc

    reference = requests[families[0]]
    replay_canary.assert_matches(reference)
    family_rows: list[dict[str, Any]] = []
    for family in families:
        request = requests[family]
        replay_canary.assert_matches(request)
        producer = bind_cumulative_spent_embedding_family_producer(
            request=request,
            replay_canary=replay_canary,
            emission=emissions[family],
        )
        draft = producer.produce_cumulative_spent(request)
        if draft.input_binding_sha256 != request.binding_sha256:
            raise RuntimeError("cumulative embedding draft changed its request binding")
        payload_path = proof_root / f"{family}.evidence_payload.json"
        _write_immutable_json(payload_path, draft.evidence_payload)
        model_registration = _component_native_artifact_registration(
            Path(emissions[family].capture_artifact_path),
            component_root=root,
        )
        source_registration = _component_file_registration(
            Path(emissions[family].source_artifact_path),
            component_root=root,
        )
        execution_registration = _component_file_registration(
            Path(emissions[family].execution_record_path),
            component_root=root,
        )
        if (
            model_registration.get("kind") != "directory"
            or draft.fit_audit.get("model_artifact_sha256") != model_registration["sha256"]
            or draft.fit_audit.get("source_artifact_sha256") != source_registration["sha256"]
            or draft.fit_audit.get("fit_execution_sha256") != execution_registration["sha256"]
        ):
            raise RuntimeError("cumulative embedding draft differs from emitted artifacts")
        identity = producer.identity()
        body = {
            "family": family,
            "request_binding_sha256": request.binding_sha256,
            "fit_semantics": CUMULATIVE_SPENT_REFIT,
            "evidence_item_count": int(draft.evidence_item_count),
            "producer_identity": identity,
            "fit_audit": copy.deepcopy(dict(draft.fit_audit)),
            "replay_canary": replay_canary.binding,
            "evidence_payload": _component_file_registration(
                payload_path,
                component_root=root,
            ),
            "execution_record": execution_registration,
            "model_artifact": model_registration,
            "source_artifact": source_registration,
        }
        family_rows.append({**body, "content_sha256": _sha256_json(body)})
    registration_body = {
        "schema_version": STAGE1_CUMULATIVE_EMBEDDING_NATIVE_SCOPE_SCHEMA,
        "request_sha256": reference.request_sha256,
        "schedule_sha256": reference.schedule_sha256,
        "scope_id": reference.scope_id,
        "outer_fold": reference.outer_fold,
        "context_epoch": reference.context_epoch,
        "provider_inner_fold": reference.provider_inner_fold,
        "spent_row_ids": list(reference.spent_row_ids),
        "sealed_row_ids": list(reference.sealed_row_ids),
        "split_scope_fingerprint": reference.split_scope_fingerprint,
        "data_projection_sha256": reference.data_projection_sha256,
        "registered_families": list(families),
        "replay_canary": replay_canary.binding,
        "sealed_text_accessed": False,
        "sealed_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "family_proofs": family_rows,
    }
    registration = {
        **registration_body,
        "content_sha256": _sha256_json(registration_body),
    }
    registration_path = proof_root / "registration.json"
    _write_immutable_json(registration_path, registration)
    return {
        **registration,
        "registration": _component_file_registration(
            registration_path,
            component_root=root,
        ),
    }


def _write_cumulative_spent_embedding_index(
    *,
    component_root: Path,
    index_path: Path,
    request_sha256: str,
    schedule_sha256: str,
    split_registry_content_sha256: str,
    scope_registrations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    root = Path(component_root).resolve(strict=True)
    rows = [
        {
            "scope_id": str(value["scope_id"]),
            "outer_fold": int(value["outer_fold"]),
            "context_epoch": int(value["context_epoch"]),
            "provider_inner_fold": int(value["provider_inner_fold"]),
            "spent_row_ids": list(map(int, value["spent_row_ids"])),
            "sealed_row_ids": list(map(int, value["sealed_row_ids"])),
            "split_scope_fingerprint": str(value["split_scope_fingerprint"]),
            "data_projection_sha256": str(value["data_projection_sha256"]),
            "content_sha256": str(value["content_sha256"]),
            "registration": copy.deepcopy(dict(value["registration"])),
        }
        for value in scope_registrations
    ]
    rows.sort(key=lambda value: (value["outer_fold"], value["context_epoch"]))
    if len({row["scope_id"] for row in rows}) != len(rows):
        raise ValueError("cumulative embedding index contains duplicate scopes")
    body = {
        "schema_version": STAGE1_CUMULATIVE_EMBEDDING_NATIVE_INDEX_SCHEMA,
        "request_sha256": str(request_sha256),
        "schedule_sha256": str(schedule_sha256),
        "split_registry_content_sha256": str(split_registry_content_sha256),
        "registered_families": list(PRODUCTION_CUMULATIVE_EMBEDDING_NATIVE_FAMILY_ADAPTERS),
        "cumulative_scope_count": len(rows),
        "sealed_text_available_to_producers": False,
        "sealed_labels_available_to_producers": False,
        "scopes": rows,
    }
    target = Path(index_path)
    if not target.is_absolute():
        target = root / target
    _write_immutable_json(target, {**body, "content_sha256": _sha256_json(body)})
    return _component_file_registration(target, component_root=root)


def _validate_cumulative_spent_embedding_index(
    *,
    component_root: Path,
    index_registration: Mapping[str, Any],
    expected_requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    request_sha256: str,
    schedule_sha256: str,
    split_registry_content_sha256: str,
    embedding_cache: SpentOnlyFrozenChunkEmbeddingCache,
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    """Revalidate the closed three-family index against canonical spent inputs."""

    root = Path(component_root).resolve(strict=True)
    if not isinstance(index_registration, Mapping) or set(index_registration) != {
        "relative_path",
        "size",
        "sha256",
    }:
        raise ValueError("cumulative embedding index registration is not closed")
    index_path = _validate_component_native_registration(root, index_registration)
    index = _read_json_object_reject_duplicates(
        index_path,
        field_name="cumulative embedding native index",
    )
    body = {key: value for key, value in index.items() if key != "content_sha256"}
    families = list(PRODUCTION_CUMULATIVE_EMBEDDING_NATIVE_FAMILY_ADAPTERS)
    scopes = index.get("scopes")
    if (
        set(index)
        != {
            "schema_version",
            "request_sha256",
            "schedule_sha256",
            "split_registry_content_sha256",
            "registered_families",
            "cumulative_scope_count",
            "sealed_text_available_to_producers",
            "sealed_labels_available_to_producers",
            "scopes",
            "content_sha256",
        }
        or index.get("schema_version") != STAGE1_CUMULATIVE_EMBEDDING_NATIVE_INDEX_SCHEMA
        or index.get("request_sha256") != request_sha256
        or index.get("schedule_sha256") != schedule_sha256
        or index.get("split_registry_content_sha256") != split_registry_content_sha256
        or index.get("registered_families") != families
        or index.get("sealed_text_available_to_producers") is not False
        or index.get("sealed_labels_available_to_producers") is not False
        or not isinstance(scopes, list)
        or int(index.get("cumulative_scope_count", -1)) != len(scopes)
        or index.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("cumulative embedding index has an invalid closed envelope")
    indexed = {str(row.get("scope_id")): row for row in scopes if isinstance(row, Mapping)}
    if (
        len(indexed) != len(scopes)
        or set(indexed) != set(expected_requests)
        or tuple(str(row.get("scope_id")) for row in scopes) != tuple(expected_requests)
    ):
        raise ValueError("cumulative embedding index has incomplete scope coverage")

    scope_fields = {
        "scope_id",
        "outer_fold",
        "context_epoch",
        "provider_inner_fold",
        "spent_row_ids",
        "sealed_row_ids",
        "split_scope_fingerprint",
        "data_projection_sha256",
        "content_sha256",
        "registration",
    }
    registration_fields = {
        "schema_version",
        "request_sha256",
        "schedule_sha256",
        "scope_id",
        "outer_fold",
        "context_epoch",
        "provider_inner_fold",
        "spent_row_ids",
        "sealed_row_ids",
        "split_scope_fingerprint",
        "data_projection_sha256",
        "registered_families",
        "replay_canary",
        "sealed_text_accessed",
        "sealed_labels_accessed",
        "oracle_fields_accessed",
        "secrets_accessed",
        "family_proofs",
        "content_sha256",
    }
    family_fields = {
        "family",
        "request_binding_sha256",
        "fit_semantics",
        "evidence_item_count",
        "producer_identity",
        "fit_audit",
        "replay_canary",
        "evidence_payload",
        "execution_record",
        "model_artifact",
        "source_artifact",
        "content_sha256",
    }
    producers_by_scope: dict[str, dict[str, Any]] = {}
    for scope_id, request in expected_requests.items():
        row = indexed[scope_id]
        if (
            set(row) != scope_fields
            or not isinstance(row.get("registration"), Mapping)
            or set(row["registration"]) != {"relative_path", "size", "sha256"}
            or int(row.get("outer_fold", 0)) != request.outer_fold
            or int(row.get("context_epoch", -1)) != request.context_epoch
            or int(row.get("provider_inner_fold", 0)) != request.provider_inner_fold
            or tuple(map(int, row.get("spent_row_ids") or ())) != request.spent_row_ids
            or tuple(map(int, row.get("sealed_row_ids") or ())) != request.sealed_row_ids
            or row.get("split_scope_fingerprint") != request.split_scope_fingerprint
            or row.get("data_projection_sha256") != request.data_projection_sha256
        ):
            raise ValueError(f"cumulative embedding index scope mismatch: {scope_id}")
        registration_path = _validate_component_native_registration(
            root,
            row.get("registration") or {},
        )
        registration = _read_json_object_reject_duplicates(
            registration_path,
            field_name=f"cumulative embedding registration {scope_id}",
        )
        registration_body = {
            key: value for key, value in registration.items() if key != "content_sha256"
        }
        family_rows = registration.get("family_proofs")
        if (
            set(registration) != registration_fields
            or registration.get("schema_version") != STAGE1_CUMULATIVE_EMBEDDING_NATIVE_SCOPE_SCHEMA
            or registration.get("request_sha256") != request.request_sha256
            or registration.get("schedule_sha256") != request.schedule_sha256
            or registration.get("scope_id") != scope_id
            or int(registration.get("outer_fold", 0)) != request.outer_fold
            or int(registration.get("context_epoch", -1)) != request.context_epoch
            or int(registration.get("provider_inner_fold", 0)) != request.provider_inner_fold
            or tuple(map(int, registration.get("spent_row_ids") or ())) != request.spent_row_ids
            or tuple(map(int, registration.get("sealed_row_ids") or ())) != request.sealed_row_ids
            or registration.get("split_scope_fingerprint") != request.split_scope_fingerprint
            or registration.get("data_projection_sha256") != request.data_projection_sha256
            or registration.get("registered_families") != families
            or registration.get("sealed_text_accessed") is not False
            or registration.get("sealed_labels_accessed") is not False
            or registration.get("oracle_fields_accessed") is not False
            or registration.get("secrets_accessed") is not False
            or registration.get("content_sha256") != _sha256_json(registration_body)
            or registration.get("content_sha256") != row.get("content_sha256")
            or not isinstance(family_rows, list)
            or len(family_rows) != len(families)
        ):
            raise ValueError(f"cumulative embedding registration is invalid: {scope_id}")
        source_position = int(
            (registration.get("replay_canary") or {}).get("source_spent_position", -1)
        )
        canary = CumulativeSpentReplayCanary.from_request(
            request,
            source_spent_position=source_position,
        )
        if canary.binding != registration.get("replay_canary"):
            raise ValueError("cumulative embedding replay canary changed")
        bound = embedding_cache.bind_spent(
            request.spent_row_ids,
            tuple(row.text for row in request.spent_rows),
        )
        seen_model_path: Path | None = None
        capture: Mapping[str, Any] | None = None
        producer_identities: dict[str, Mapping[str, Any]] = {}
        evidence_payloads: dict[str, Mapping[str, Any]] = {}
        evidence_counts: dict[str, int] = {}
        execution_paths: dict[str, Path] = {}
        fit_audits: dict[str, Mapping[str, Any]] = {}
        for family, family_row in zip(families, family_rows):
            if not isinstance(family_row, Mapping) or set(family_row) != family_fields:
                raise ValueError("cumulative embedding family proof is not closed")
            for key in ("evidence_payload", "execution_record", "source_artifact"):
                descriptor = family_row.get(key)
                if not isinstance(descriptor, Mapping) or set(descriptor) != {
                    "relative_path",
                    "size",
                    "sha256",
                }:
                    raise ValueError("cumulative embedding file registration is not closed")
            model_descriptor = family_row.get("model_artifact")
            if not isinstance(model_descriptor, Mapping) or set(model_descriptor) != {
                "relative_path",
                "kind",
                "file_count",
                "size",
                "sha256",
            }:
                raise ValueError("cumulative embedding model registration is not closed")
            family_body = {
                key: value for key, value in family_row.items() if key != "content_sha256"
            }
            family_request = _cumulative_request_for_family(request, family=family)
            if (
                family_row.get("family") != family
                or family_row.get("request_binding_sha256") != family_request.binding_sha256
                or family_row.get("fit_semantics") != CUMULATIVE_SPENT_REFIT
                or family_row.get("replay_canary") != canary.binding
                or int(family_row.get("evidence_item_count", 0)) < 1
                or family_row.get("content_sha256") != _sha256_json(family_body)
            ):
                raise ValueError(f"cumulative embedding family proof is invalid: {family}")
            paths = {
                key: _validate_component_native_registration(root, family_row.get(key) or {})
                for key in (
                    "evidence_payload",
                    "execution_record",
                    "model_artifact",
                    "source_artifact",
                )
            }
            if seen_model_path is None:
                seen_model_path = paths["model_artifact"]
                capture = validate_embedding_native_capture(
                    seen_model_path,
                    embedding_provider=bound,
                    fit_texts=tuple(row.text for row in request.spent_rows),
                    expected_fit_treatment=tuple(row.treatment for row in request.spent_rows),
                    expected_fit_outcome=tuple(row.outcome for row in request.spent_rows),
                    expected_discovery_projection={"_oci_row_id": list(request.spent_row_ids)},
                    expected_scope_id=scope_id,
                    expected_fit_row_ids=request.spent_row_ids,
                    expected_heldout_row_ids=(canary.alias_row_id,),
                )
            elif paths["model_artifact"] != seen_model_path:
                raise ValueError("cumulative embedding families use different native fits")
            if capture is None:
                raise RuntimeError("cumulative embedding capture was not validated")
            capture_configuration = {
                "capture_schema_version": EMBEDDING_NATIVE_CAPTURE_SCHEMA,
                "text_column": capture["text_column"],
                "outcome_type": capture["outcome_type"],
                "seed": capture["seed"],
                "embedding_config": copy.deepcopy(capture["embedding_config"]),
                "tfidf_nested_calibration_folds": int(
                    capture["tfidf_training_scope_policy"]["configured_fold_count"]
                ),
            }
            expected_identity = cumulative_spent_embedding_family_identity(
                family=family,
                capture_configuration=capture_configuration,
            )
            if family_row.get("producer_identity") != expected_identity:
                raise ValueError("cumulative embedding producer identity changed")
            payload = _read_json_object_reject_duplicates(
                paths["evidence_payload"],
                field_name=f"cumulative embedding payload {scope_id}/{family}",
            )
            execution = _read_json_object_reject_duplicates(
                paths["execution_record"],
                field_name=f"cumulative embedding execution {scope_id}/{family}",
            )
            audit = family_row.get("fit_audit")
            policy = (
                capture.get("tfidf_training_scope_policy")
                if family == TFIDF_SEMANTIC_RETRIEVAL
                else None
            )
            if (
                not isinstance(audit, Mapping)
                or audit.get("input_binding_sha256") != family_request.binding_sha256
                or audit.get("fit_execution_sha256") != family_row["execution_record"]["sha256"]
                or audit.get("model_artifact_sha256") != family_row["model_artifact"]["sha256"]
                or audit.get("source_artifact_sha256") != family_row["source_artifact"]["sha256"]
                or audit.get("tfidf_training_scope_policy") != policy
                or any(
                    audit.get(flag) is not False
                    for flag in (
                        "sealed_text_accessed",
                        "sealed_labels_accessed",
                        "oracle_fields_accessed",
                        "secrets_accessed",
                    )
                )
                or execution.get("family") != family
                or execution.get("request_binding_sha256") != family_request.binding_sha256
                or execution.get("model_artifact_sha256") != family_row["model_artifact"]["sha256"]
                or execution.get("source_artifact_sha256")
                != family_row["source_artifact"]["sha256"]
                or execution.get("evidence_payload_sha256") != _sha256_json(payload)
                or int(execution.get("evidence_item_count", 0))
                != int(family_row["evidence_item_count"])
            ):
                raise ValueError(f"cumulative embedding artifacts are inconsistent: {family}")
            producer_identities[family] = copy.deepcopy(expected_identity)
            evidence_payloads[family] = copy.deepcopy(payload)
            evidence_counts[family] = int(family_row["evidence_item_count"])
            execution_paths[family] = paths["execution_record"]
            fit_audits[family] = copy.deepcopy(dict(audit))
        if seen_model_path is None:
            raise RuntimeError("cumulative embedding registration has no native capture")
        family_requests = {
            family: _cumulative_request_for_family(request, family=family) for family in families
        }
        bound_producers = bind_persisted_cumulative_spent_embedding_producers(
            requests=family_requests,
            replay_canary=canary,
            embedding_provider=bound,
            producer_identity_by_family=producer_identities,
            evidence_payload_by_family=evidence_payloads,
            evidence_item_count_by_family=evidence_counts,
            capture_artifact_path=seen_model_path,
            execution_record_path_by_family=execution_paths,
            expected_fit_audit_by_family=fit_audits,
        )
        scope_producers = dict(bound_producers)
        for family, producer in scope_producers.items():
            draft = producer.produce_cumulative_spent(family_requests[family])
            if (
                draft.evidence_payload != evidence_payloads[family]
                or int(draft.evidence_item_count) != evidence_counts[family]
                or draft.fit_audit != fit_audits[family]
            ):
                raise RuntimeError(f"cumulative embedding replay changed: {scope_id}/{family}")
        producers_by_scope[scope_id] = scope_producers
    return copy.deepcopy(index), producers_by_scope


def _register_cumulative_spent_remaining_scope(
    *,
    component_root: Path,
    proof_directory: Path,
    requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    replay_canary: CumulativeSpentReplayCanary,
    emissions: Mapping[str, Any],
    families: Sequence[str],
) -> Mapping[str, Any]:
    """Persist component-emitted topic/orphan or neural-query family views."""

    ordered_families = tuple(map(str, families))
    if ordered_families not in {
        tuple(PRODUCTION_CUMULATIVE_TFIDF_NATIVE_FAMILY_ADAPTERS),
        tuple(PRODUCTION_CUMULATIVE_QUERY_NATIVE_FAMILY_ADAPTERS),
    }:
        raise ValueError("unsupported cumulative remaining-family group")
    if (
        set(requests) != set(ordered_families)
        or set(emissions) != set(ordered_families)
        or not ordered_families
    ):
        raise ValueError("cumulative remaining-family registration is incomplete")
    root = Path(component_root).resolve(strict=True)
    proof_root = Path(proof_directory)
    if not proof_root.is_absolute():
        proof_root = root / proof_root
    if proof_root.exists() or proof_root.is_symlink():
        raise RuntimeError("cumulative remaining-family proof directory must be new")
    proof_root.mkdir(parents=True, exist_ok=False)
    try:
        proof_root.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError("cumulative remaining-family proof directory escapes component") from exc

    reference = requests[ordered_families[0]]
    replay_canary.assert_matches(reference)
    family_rows: list[dict[str, Any]] = []
    for family in ordered_families:
        request = requests[family]
        replay_canary.assert_matches(request)
        producer = bind_cumulative_spent_remaining_family_producer(
            request=request,
            replay_canary=replay_canary,
            emission=emissions[family],
        )
        draft = producer.produce_cumulative_spent(request)
        if (
            draft.input_binding_sha256 != request.binding_sha256
            or int(draft.evidence_item_count) < 1
        ):
            raise RuntimeError("cumulative remaining-family draft changed its request")
        payload_path = proof_root / f"{family}.evidence_payload.json"
        _write_immutable_json(payload_path, draft.evidence_payload)
        model_registration = _component_native_artifact_registration(
            Path(emissions[family].model_artifact_path),
            component_root=root,
        )
        metadata_registration = _component_file_registration(
            Path(emissions[family].native_metadata_path),
            component_root=root,
        )
        source_registration = _component_file_registration(
            Path(emissions[family].source_artifact_path),
            component_root=root,
        )
        execution_registration = _component_file_registration(
            Path(emissions[family].execution_record_path),
            component_root=root,
        )
        if (
            draft.fit_audit.get("model_artifact_sha256") != model_registration["sha256"]
            or draft.fit_audit.get("source_artifact_sha256") != source_registration["sha256"]
            or draft.fit_audit.get("fit_execution_sha256") != execution_registration["sha256"]
        ):
            raise RuntimeError("cumulative remaining-family draft differs from native artifacts")
        body = {
            "family": family,
            "native_kind": str(emissions[family].native_kind),
            "request_binding_sha256": request.binding_sha256,
            "fit_semantics": CUMULATIVE_SPENT_REFIT,
            "evidence_item_count": int(draft.evidence_item_count),
            "producer_identity": copy.deepcopy(dict(producer.identity())),
            "fit_audit": copy.deepcopy(dict(draft.fit_audit)),
            "replay_canary": replay_canary.binding,
            "evidence_payload": _component_file_registration(
                payload_path,
                component_root=root,
            ),
            "execution_record": execution_registration,
            "native_metadata": metadata_registration,
            "model_artifact": model_registration,
            "source_artifact": source_registration,
        }
        family_rows.append({**body, "content_sha256": _sha256_json(body)})

    registration_body = {
        "schema_version": STAGE1_CUMULATIVE_REMAINING_NATIVE_SCOPE_SCHEMA,
        "request_sha256": reference.request_sha256,
        "schedule_sha256": reference.schedule_sha256,
        "scope_id": reference.scope_id,
        "outer_fold": reference.outer_fold,
        "context_epoch": reference.context_epoch,
        "provider_inner_fold": reference.provider_inner_fold,
        "spent_row_ids": list(reference.spent_row_ids),
        "sealed_row_ids": list(reference.sealed_row_ids),
        "split_scope_fingerprint": reference.split_scope_fingerprint,
        "data_projection_sha256": reference.data_projection_sha256,
        "registered_families": list(ordered_families),
        "replay_canary": replay_canary.binding,
        "sealed_text_accessed": False,
        "sealed_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "family_proofs": family_rows,
    }
    registration = {
        **registration_body,
        "content_sha256": _sha256_json(registration_body),
    }
    registration_path = proof_root / "registration.json"
    _write_immutable_json(registration_path, registration)
    return {
        **registration,
        "registration": _component_file_registration(
            registration_path,
            component_root=root,
        ),
    }


def _write_cumulative_spent_remaining_index(
    *,
    component_root: Path,
    index_path: Path,
    index_schema: str,
    families: Sequence[str],
    request_sha256: str,
    schedule_sha256: str,
    split_registry_content_sha256: str,
    scope_registrations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    root = Path(component_root).resolve(strict=True)
    ordered_families = tuple(map(str, families))
    expected_schema = {
        tuple(PRODUCTION_CUMULATIVE_TFIDF_NATIVE_FAMILY_ADAPTERS): (
            STAGE1_CUMULATIVE_TFIDF_NATIVE_INDEX_SCHEMA
        ),
        tuple(PRODUCTION_CUMULATIVE_QUERY_NATIVE_FAMILY_ADAPTERS): (
            STAGE1_CUMULATIVE_QUERY_NATIVE_INDEX_SCHEMA
        ),
    }.get(ordered_families)
    if expected_schema is None or index_schema != expected_schema:
        raise ValueError("cumulative remaining-family index schema/group mismatch")
    rows = [
        {
            "scope_id": str(value["scope_id"]),
            "outer_fold": int(value["outer_fold"]),
            "context_epoch": int(value["context_epoch"]),
            "provider_inner_fold": int(value["provider_inner_fold"]),
            "spent_row_ids": list(map(int, value["spent_row_ids"])),
            "sealed_row_ids": list(map(int, value["sealed_row_ids"])),
            "split_scope_fingerprint": str(value["split_scope_fingerprint"]),
            "data_projection_sha256": str(value["data_projection_sha256"]),
            "content_sha256": str(value["content_sha256"]),
            "registration": copy.deepcopy(dict(value["registration"])),
        }
        for value in scope_registrations
    ]
    rows.sort(key=lambda value: (value["outer_fold"], value["context_epoch"]))
    if len({row["scope_id"] for row in rows}) != len(rows):
        raise ValueError("cumulative remaining-family index contains duplicate scopes")
    body = {
        "schema_version": index_schema,
        "request_sha256": str(request_sha256),
        "schedule_sha256": str(schedule_sha256),
        "split_registry_content_sha256": str(split_registry_content_sha256),
        "registered_families": list(ordered_families),
        "cumulative_scope_count": len(rows),
        "sealed_text_available_to_producers": False,
        "sealed_labels_available_to_producers": False,
        "scopes": rows,
    }
    target = Path(index_path)
    if not target.is_absolute():
        target = root / target
    _write_immutable_json(target, {**body, "content_sha256": _sha256_json(body)})
    return _component_file_registration(target, component_root=root)


def _validate_cumulative_spent_remaining_index(
    *,
    component_root: Path,
    index_registration: Mapping[str, Any],
    index_schema: str,
    families: Sequence[str],
    expected_requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    request_sha256: str,
    schedule_sha256: str,
    split_registry_content_sha256: str,
    tfidf_config: AppliedInferenceConfig | None = None,
    query_service_identity: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    """Reload component bytes into producers that revalidate on every call."""

    root = Path(component_root).resolve(strict=True)
    file_fields = {"relative_path", "size", "sha256"}
    native_fields = {"relative_path", "kind", "file_count", "size", "sha256"}
    ordered_families = tuple(map(str, families))
    if ordered_families == tuple(PRODUCTION_CUMULATIVE_TFIDF_NATIVE_FAMILY_ADAPTERS):
        if index_schema != STAGE1_CUMULATIVE_TFIDF_NATIVE_INDEX_SCHEMA or not isinstance(
            tfidf_config,
            AppliedInferenceConfig,
        ):
            raise TypeError("TF-IDF cumulative reload requires its exact config and schema")
        if query_service_identity is not None:
            raise ValueError("TF-IDF cumulative reload cannot receive a query service identity")
    elif ordered_families == tuple(PRODUCTION_CUMULATIVE_QUERY_NATIVE_FAMILY_ADAPTERS):
        if index_schema != STAGE1_CUMULATIVE_QUERY_NATIVE_INDEX_SCHEMA or not isinstance(
            query_service_identity,
            Mapping,
        ):
            raise TypeError("query cumulative reload requires its service identity and schema")
        if tfidf_config is not None:
            raise ValueError("query cumulative reload cannot receive a TF-IDF config")
    else:
        raise ValueError("unsupported cumulative remaining-family reload group")
    if not isinstance(index_registration, Mapping) or set(index_registration) != file_fields:
        raise ValueError("cumulative remaining-family index registration is not closed")
    index_path = _validate_component_native_registration(root, index_registration)
    index = _read_json_object_reject_duplicates(
        index_path,
        field_name="cumulative remaining-family native index",
    )
    index_body = {key: value for key, value in index.items() if key != "content_sha256"}
    scopes = index.get("scopes")
    if (
        set(index)
        != {
            "schema_version",
            "request_sha256",
            "schedule_sha256",
            "split_registry_content_sha256",
            "registered_families",
            "cumulative_scope_count",
            "sealed_text_available_to_producers",
            "sealed_labels_available_to_producers",
            "scopes",
            "content_sha256",
        }
        or index.get("schema_version") != index_schema
        or index.get("request_sha256") != request_sha256
        or index.get("schedule_sha256") != schedule_sha256
        or index.get("split_registry_content_sha256") != split_registry_content_sha256
        or index.get("registered_families") != list(ordered_families)
        or index.get("sealed_text_available_to_producers") is not False
        or index.get("sealed_labels_available_to_producers") is not False
        or not isinstance(scopes, list)
        or int(index.get("cumulative_scope_count", -1)) != len(scopes)
        or index.get("content_sha256") != _sha256_json(index_body)
    ):
        raise ValueError("cumulative remaining-family index has an invalid envelope")
    indexed = {str(row.get("scope_id")): row for row in scopes if isinstance(row, Mapping)}
    if (
        len(indexed) != len(scopes)
        or set(indexed) != set(expected_requests)
        or tuple(str(row.get("scope_id")) for row in scopes) != tuple(expected_requests)
    ):
        raise ValueError("cumulative remaining-family index has incomplete scope coverage")

    scope_fields = {
        "scope_id",
        "outer_fold",
        "context_epoch",
        "provider_inner_fold",
        "spent_row_ids",
        "sealed_row_ids",
        "split_scope_fingerprint",
        "data_projection_sha256",
        "content_sha256",
        "registration",
    }
    registration_fields = {
        "schema_version",
        "request_sha256",
        "schedule_sha256",
        "scope_id",
        "outer_fold",
        "context_epoch",
        "provider_inner_fold",
        "spent_row_ids",
        "sealed_row_ids",
        "split_scope_fingerprint",
        "data_projection_sha256",
        "registered_families",
        "replay_canary",
        "sealed_text_accessed",
        "sealed_labels_accessed",
        "oracle_fields_accessed",
        "secrets_accessed",
        "family_proofs",
        "content_sha256",
    }
    family_fields = {
        "family",
        "native_kind",
        "request_binding_sha256",
        "fit_semantics",
        "evidence_item_count",
        "producer_identity",
        "fit_audit",
        "replay_canary",
        "evidence_payload",
        "execution_record",
        "native_metadata",
        "model_artifact",
        "source_artifact",
        "content_sha256",
    }
    producers_by_scope: dict[str, dict[str, Any]] = {}
    for scope_id, reference in expected_requests.items():
        row = indexed[scope_id]
        if (
            set(row) != scope_fields
            or not isinstance(row.get("registration"), Mapping)
            or set(row["registration"]) != file_fields
            or int(row.get("outer_fold", 0)) != reference.outer_fold
            or int(row.get("context_epoch", -1)) != reference.context_epoch
            or int(row.get("provider_inner_fold", 0)) != reference.provider_inner_fold
            or tuple(map(int, row.get("spent_row_ids") or ())) != reference.spent_row_ids
            or tuple(map(int, row.get("sealed_row_ids") or ())) != reference.sealed_row_ids
            or row.get("split_scope_fingerprint") != reference.split_scope_fingerprint
            or row.get("data_projection_sha256") != reference.data_projection_sha256
        ):
            raise ValueError(f"cumulative remaining-family scope mismatch: {scope_id}")
        registration_path = _validate_component_native_registration(
            root,
            row["registration"],
        )
        registration = _read_json_object_reject_duplicates(
            registration_path,
            field_name=f"cumulative remaining-family registration {scope_id}",
        )
        registration_body = {
            key: value for key, value in registration.items() if key != "content_sha256"
        }
        proofs = registration.get("family_proofs")
        if (
            set(registration) != registration_fields
            or registration.get("schema_version") != STAGE1_CUMULATIVE_REMAINING_NATIVE_SCOPE_SCHEMA
            or registration.get("request_sha256") != reference.request_sha256
            or registration.get("schedule_sha256") != reference.schedule_sha256
            or registration.get("scope_id") != scope_id
            or int(registration.get("outer_fold", 0)) != reference.outer_fold
            or int(registration.get("context_epoch", -1)) != reference.context_epoch
            or int(registration.get("provider_inner_fold", 0)) != reference.provider_inner_fold
            or tuple(map(int, registration.get("spent_row_ids") or ())) != reference.spent_row_ids
            or tuple(map(int, registration.get("sealed_row_ids") or ())) != reference.sealed_row_ids
            or registration.get("split_scope_fingerprint") != reference.split_scope_fingerprint
            or registration.get("data_projection_sha256") != reference.data_projection_sha256
            or registration.get("registered_families") != list(ordered_families)
            or registration.get("sealed_text_accessed") is not False
            or registration.get("sealed_labels_accessed") is not False
            or registration.get("oracle_fields_accessed") is not False
            or registration.get("secrets_accessed") is not False
            or registration.get("content_sha256") != _sha256_json(registration_body)
            or registration.get("content_sha256") != row.get("content_sha256")
            or not isinstance(proofs, list)
            or len(proofs) != len(ordered_families)
        ):
            raise ValueError(f"cumulative remaining-family registration is invalid: {scope_id}")
        canary = CumulativeSpentReplayCanary.from_request(reference)
        if registration.get("replay_canary") != canary.binding:
            raise ValueError("cumulative remaining-family replay canary changed")

        identities: dict[str, Mapping[str, Any]] = {}
        payloads: dict[str, Mapping[str, Any]] = {}
        counts: dict[str, int] = {}
        audits: dict[str, Mapping[str, Any]] = {}
        execution_paths: dict[str, Path] = {}
        model_paths: dict[str, Path] = {}
        source_paths: dict[str, Path] = {}
        metadata_paths: dict[str, Path] = {}
        for family, raw_proof in zip(ordered_families, proofs):
            if not isinstance(raw_proof, Mapping) or set(raw_proof) != family_fields:
                raise ValueError("cumulative remaining-family proof is not closed")
            proof_body = {key: value for key, value in raw_proof.items() if key != "content_sha256"}
            family_request = _cumulative_request_for_family(reference, family=family)
            expected_kind = (
                "nested_tfidf" if family in TFIDF_CUMULATIVE_FAMILIES else "owned_neural_query"
            )
            if (
                raw_proof.get("family") != family
                or raw_proof.get("native_kind") != expected_kind
                or raw_proof.get("request_binding_sha256") != family_request.binding_sha256
                or raw_proof.get("fit_semantics") != CUMULATIVE_SPENT_REFIT
                or raw_proof.get("replay_canary") != canary.binding
                or int(raw_proof.get("evidence_item_count", 0)) < 1
                or raw_proof.get("content_sha256") != _sha256_json(proof_body)
            ):
                raise ValueError(f"cumulative remaining-family proof is invalid: {family}")
            for key in (
                "evidence_payload",
                "execution_record",
                "native_metadata",
                "source_artifact",
            ):
                descriptor = raw_proof.get(key)
                if not isinstance(descriptor, Mapping) or set(descriptor) != file_fields:
                    raise ValueError(f"cumulative remaining-family {key} descriptor is open")
            model_descriptor = raw_proof.get("model_artifact")
            if not isinstance(model_descriptor, Mapping) or set(model_descriptor) != native_fields:
                raise ValueError("cumulative remaining-family model descriptor is open")
            paths = {
                key: _validate_component_native_registration(root, raw_proof[key])
                for key in (
                    "evidence_payload",
                    "execution_record",
                    "native_metadata",
                    "model_artifact",
                    "source_artifact",
                )
            }
            payload = _read_json_object_reject_duplicates(
                paths["evidence_payload"],
                field_name=f"cumulative remaining-family payload {scope_id}/{family}",
            )
            audit = raw_proof.get("fit_audit")
            if (
                not isinstance(audit, Mapping)
                or audit.get("input_binding_sha256") != family_request.binding_sha256
                or audit.get("fit_execution_sha256") != raw_proof["execution_record"]["sha256"]
                or audit.get("model_artifact_sha256") != raw_proof["model_artifact"]["sha256"]
                or audit.get("source_artifact_sha256") != raw_proof["source_artifact"]["sha256"]
                or any(
                    audit.get(flag) is not False
                    for flag in (
                        "sealed_text_accessed",
                        "sealed_labels_accessed",
                        "oracle_fields_accessed",
                        "secrets_accessed",
                    )
                )
            ):
                raise ValueError(f"cumulative remaining-family audit is invalid: {family}")
            identities[family] = copy.deepcopy(dict(raw_proof["producer_identity"]))
            payloads[family] = payload
            counts[family] = int(raw_proof["evidence_item_count"])
            audits[family] = copy.deepcopy(dict(audit))
            execution_paths[family] = paths["execution_record"]
            model_paths[family] = paths["model_artifact"]
            source_paths[family] = paths["source_artifact"]
            metadata_paths[family] = paths["native_metadata"]

        requests = {
            family: _cumulative_request_for_family(reference, family=family)
            for family in ordered_families
        }
        if ordered_families == tuple(PRODUCTION_CUMULATIVE_TFIDF_NATIVE_FAMILY_ADAPTERS):
            metadata_path = metadata_paths[TFIDF_TOPICS]
            if (
                len(set(metadata_paths.values())) != 1
                or len(set(model_paths.values())) != 1
                or len(set(source_paths.values())) != 1
                or metadata_path.parent != model_paths[TFIDF_TOPICS].parent
                or metadata_path != metadata_path.parent / "context_metadata.json"
            ):
                raise ValueError("cumulative TF-IDF families do not share one native fit")
            bound = bind_persisted_cumulative_spent_tfidf_producers(
                requests=requests,
                replay_canary=canary,
                config=tfidf_config,
                producer_identity_by_family=identities,
                evidence_payload_by_family=payloads,
                evidence_item_count_by_family=counts,
                artifact_dir=metadata_path.parent,
                execution_record_path_by_family=execution_paths,
                expected_fit_audit_by_family=audits,
            )
            scope_producers = dict(bound)
            for family, producer in scope_producers.items():
                if (
                    Path(producer._native_metadata_path).resolve(strict=True)
                    != metadata_paths[family]
                    or Path(producer._model_artifact_path).resolve(strict=True)
                    != model_paths[family]
                    or Path(producer._source_artifact_path).resolve(strict=True)
                    != source_paths[family]
                ):
                    raise ValueError(
                        "cumulative TF-IDF descriptor aliases its canonical metadata paths"
                    )
        else:
            family = NEURAL_QUERY_MOMENTS
            if (
                metadata_paths[family] != model_paths[family] / "metadata.json"
                or source_paths[family] != model_paths[family].parent / "safe_evidence.json"
            ):
                raise ValueError("cumulative neural-query artifact layout is noncanonical")
            scope_producers = {
                family: bind_persisted_cumulative_spent_neural_query_producer(
                    request=requests[family],
                    replay_canary=canary,
                    expected_service_identity=query_service_identity,
                    producer_identity=identities[family],
                    evidence_payload=payloads[family],
                    evidence_item_count=counts[family],
                    model_artifact_path=model_paths[family],
                    source_artifact_path=source_paths[family],
                    execution_record_path=execution_paths[family],
                    expected_fit_audit=audits[family],
                )
            }
            producer = scope_producers[family]
            if (
                Path(producer._native_metadata_path).resolve(strict=True) != metadata_paths[family]
                or Path(producer._model_artifact_path).resolve(strict=True) != model_paths[family]
                or Path(producer._source_artifact_path).resolve(strict=True) != source_paths[family]
            ):
                raise ValueError(
                    "cumulative neural-query descriptor aliases canonical snapshot paths"
                )
        for family, producer in scope_producers.items():
            draft = producer.produce_cumulative_spent(requests[family])
            if (
                draft.evidence_payload != payloads[family]
                or int(draft.evidence_item_count) != counts[family]
                or draft.fit_audit != audits[family]
            ):
                raise RuntimeError(f"cumulative remaining-family replay changed: {family}")
        producers_by_scope[scope_id] = scope_producers
    return copy.deepcopy(index), producers_by_scope


def _validate_cumulative_spent_tfidf_index(
    *,
    component_root: Path,
    index_registration: Mapping[str, Any],
    expected_requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    request_sha256: str,
    schedule_sha256: str,
    split_registry_content_sha256: str,
    config: AppliedInferenceConfig,
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    return _validate_cumulative_spent_remaining_index(
        component_root=component_root,
        index_registration=index_registration,
        index_schema=STAGE1_CUMULATIVE_TFIDF_NATIVE_INDEX_SCHEMA,
        families=PRODUCTION_CUMULATIVE_TFIDF_NATIVE_FAMILY_ADAPTERS,
        expected_requests=expected_requests,
        request_sha256=request_sha256,
        schedule_sha256=schedule_sha256,
        split_registry_content_sha256=split_registry_content_sha256,
        tfidf_config=config,
    )


def _validate_cumulative_spent_query_index(
    *,
    component_root: Path,
    index_registration: Mapping[str, Any],
    expected_requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    request_sha256: str,
    schedule_sha256: str,
    split_registry_content_sha256: str,
    service_identity: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    return _validate_cumulative_spent_remaining_index(
        component_root=component_root,
        index_registration=index_registration,
        index_schema=STAGE1_CUMULATIVE_QUERY_NATIVE_INDEX_SCHEMA,
        families=PRODUCTION_CUMULATIVE_QUERY_NATIVE_FAMILY_ADAPTERS,
        expected_requests=expected_requests,
        request_sha256=request_sha256,
        schedule_sha256=schedule_sha256,
        split_registry_content_sha256=split_registry_content_sha256,
        query_service_identity=service_identity,
    )


def _cumulative_legacy_configuration_by_family(
    *,
    config: AppliedInferenceConfig,
    scope_id: str,
    split_registry_content_sha256: str,
    htr_model_tree_sha256: str,
    seed: int,
) -> Mapping[str, Mapping[str, Any]]:
    pair_config = config.architecture.multi_model_forest
    htr_config = config.architecture.agentic_attention_variable_forest
    shared = {
        "schema_version": STAGE1_CUMULATIVE_LEGACY_NATIVE_SCOPE_SCHEMA,
        "scope_id": str(scope_id),
        "text_column": config.text_column,
        "treatment_column": config.treatment_column,
        "outcome_column": config.outcome_column,
        "outcome_type": config.outcome_type,
        "transform_policy": CUMULATIVE_SPENT_REPLAY_CANARY_SEMANTICS,
        "split_registry_content_sha256": str(split_registry_content_sha256),
        "seed": int(seed),
    }
    return {
        BOW_NUISANCE: {
            **shared,
            "family": BOW_NUISANCE,
            "capture_schema_version": BOW_NATIVE_CAPTURE_SCHEMA,
            "nuisance_folds": int(pair_config.nuisance_folds),
            "effect_folds": int(pair_config.effect_folds),
            "bow_views": [asdict(view) for view in pair_config.bow_views],
        },
        BOW_R_LOSS: {
            **shared,
            "family": BOW_R_LOSS,
            "capture_schema_version": BOW_NATIVE_CAPTURE_SCHEMA,
            "nuisance_folds": int(pair_config.nuisance_folds),
            "effect_folds": int(pair_config.effect_folds),
            "r_loss_nuisance_source": "ensemble_mean_nuisance",
            "bow_views": [asdict(view) for view in pair_config.bow_views],
        },
        HTR_NEURAL: {
            **shared,
            "family": HTR_NEURAL,
            "capture_schema_version": HTR_NATIVE_CAPTURE_SCHEMA,
            "nuisance_folds": int(htr_config.nuisance_folds),
            "effect_folds": int(htr_config.effect_folds),
            "htr_model_tree_sha256": str(htr_model_tree_sha256),
        },
        MATCHED_PAIR_UPLIFT: {
            **shared,
            "family": MATCHED_PAIR_UPLIFT,
            "capture_schema_version": MATCHED_PAIR_NATIVE_CAPTURE_SCHEMA,
            "effect_folds": int(pair_config.effect_folds),
            "required_subproducers": ["bow", "htr"],
            "htr_model_tree_sha256": str(htr_model_tree_sha256),
        },
    }


def _register_legacy_cumulative_spent_native_scope(
    *,
    component_root: Path,
    proof_directory: Path,
    request: CumulativeSpentStage1FamilyRequest,
    catalog: RoleNeutralEvidenceCatalog,
    replay_canary: CumulativeSpentReplayCanary,
    capture_artifact_by_family: Mapping[str, Path],
    configuration_by_family: Mapping[str, Mapping[str, Any]],
    htr_model_path: Path | None = None,
    htr_model_sha256: str | None = None,
    device: torch.device | str = "cpu",
) -> Mapping[str, Any]:
    """Persist and immediately replay four genuine cumulative legacy fits."""

    families = tuple(PRODUCTION_CUMULATIVE_LEGACY_NATIVE_FAMILY_ADAPTERS)
    if set(capture_artifact_by_family) != set(families) or set(configuration_by_family) != set(
        families
    ):
        raise ValueError("cumulative legacy registration requires exactly four families")
    if (
        catalog.outer_fold != request.outer_fold
        or catalog.scope != "inner_train"
        or catalog.inner_fold != request.provider_inner_fold
        or catalog.split_fingerprint != request.split_scope_fingerprint
    ):
        raise ValueError("cumulative legacy catalog belongs to another canonical scope")
    root = Path(component_root).resolve(strict=True)
    replay_canary.assert_matches(request)
    proof_root = Path(proof_directory)
    if not proof_root.is_absolute():
        proof_root = root / proof_root
    if proof_root.exists():
        raise RuntimeError("cumulative legacy proof directory already exists")
    proof_root.mkdir(parents=True, exist_ok=False)
    try:
        proof_root.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError("cumulative legacy proof directory escapes its component") from exc

    family_rows: list[dict[str, Any]] = []
    for family in families:
        family_request = _cumulative_request_for_family(request, family=family)
        replay_canary.assert_matches(family_request)
        evidence_payload, evidence_item_count = family_payload_from_catalog(
            catalog,
            family=family,
        )
        if int(evidence_item_count) < 1:
            raise RuntimeError(f"cumulative legacy scope has no evidence for {family}")
        configuration = copy.deepcopy(dict(configuration_by_family[family]))
        identity = cumulative_spent_native_family_identity(
            family=family,
            configuration=configuration,
        )
        capture_registration = _component_native_artifact_registration(
            Path(capture_artifact_by_family[family]),
            component_root=root,
        )
        if capture_registration.get("kind") != "directory":
            raise ValueError("cumulative legacy capture must be a directory artifact")
        capture_path = root / str(capture_registration["relative_path"])
        payload_path = proof_root / f"{family}.evidence_payload.json"
        execution_path = proof_root / f"{family}.execution.json"
        _write_immutable_json(payload_path, evidence_payload)
        record = cumulative_spent_native_execution_record(
            request=family_request,
            producer_identity=identity,
            evidence_payload=evidence_payload,
            evidence_item_count=evidence_item_count,
            replay_canary=replay_canary,
            capture_artifact_path=capture_path,
            source_artifact_path=payload_path,
            htr_model_path=(
                htr_model_path if family in {HTR_NEURAL, MATCHED_PAIR_UPLIFT} else None
            ),
            expected_htr_model_tree_sha256=(
                htr_model_sha256 if family in {HTR_NEURAL, MATCHED_PAIR_UPLIFT} else None
            ),
            device=device,
        )
        _write_immutable_json(execution_path, record)
        producer = bind_cumulative_spent_native_family_producer(
            request=family_request,
            producer_identity=identity,
            evidence_payload=evidence_payload,
            evidence_item_count=evidence_item_count,
            replay_canary=replay_canary,
            capture_artifact_path=capture_path,
            source_artifact_path=payload_path,
            execution_record_path=execution_path,
            htr_model_path=(
                htr_model_path if family in {HTR_NEURAL, MATCHED_PAIR_UPLIFT} else None
            ),
            expected_htr_model_tree_sha256=(
                htr_model_sha256 if family in {HTR_NEURAL, MATCHED_PAIR_UPLIFT} else None
            ),
            device=device,
        )
        draft = producer.produce_cumulative_spent(family_request)
        family_body = {
            "family": family,
            "request_binding_sha256": family_request.binding_sha256,
            "fit_semantics": CUMULATIVE_SPENT_REFIT,
            "evidence_item_count": int(evidence_item_count),
            "producer_configuration": configuration,
            "producer_identity": identity,
            "fit_audit": copy.deepcopy(dict(draft.fit_audit)),
            "replay_canary": replay_canary.binding,
            "evidence_payload": _component_file_registration(
                payload_path,
                component_root=root,
            ),
            "execution_record": _component_file_registration(
                execution_path,
                component_root=root,
            ),
            "model_artifact": capture_registration,
            "source_artifact": _component_file_registration(
                payload_path,
                component_root=root,
            ),
        }
        family_rows.append({**family_body, "content_sha256": _sha256_json(family_body)})
    registration_body = {
        "schema_version": STAGE1_CUMULATIVE_LEGACY_NATIVE_SCOPE_SCHEMA,
        "request_sha256": request.request_sha256,
        "schedule_sha256": request.schedule_sha256,
        "scope_id": request.scope_id,
        "outer_fold": request.outer_fold,
        "context_epoch": request.context_epoch,
        "provider_inner_fold": request.provider_inner_fold,
        "spent_row_ids": list(request.spent_row_ids),
        "sealed_row_ids": list(request.sealed_row_ids),
        "split_scope_fingerprint": request.split_scope_fingerprint,
        "data_projection_sha256": request.data_projection_sha256,
        "catalog_sha256": catalog.catalog_sha256,
        "registered_families": list(families),
        "replay_canary": replay_canary.binding,
        "sealed_text_accessed": False,
        "sealed_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "family_proofs": family_rows,
    }
    registration = {
        **registration_body,
        "content_sha256": _sha256_json(registration_body),
    }
    registration_path = proof_root / "registration.json"
    _write_immutable_json(registration_path, registration)
    return {
        **registration,
        "registration": _component_file_registration(
            registration_path,
            component_root=root,
        ),
    }


def _write_legacy_cumulative_spent_native_index(
    *,
    component_root: Path,
    index_path: Path,
    request_sha256: str,
    schedule_sha256: str,
    split_registry_content_sha256: str,
    scope_registrations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    root = Path(component_root).resolve(strict=True)
    rows = [
        {
            "scope_id": str(registration["scope_id"]),
            "outer_fold": int(registration["outer_fold"]),
            "context_epoch": int(registration["context_epoch"]),
            "provider_inner_fold": int(registration["provider_inner_fold"]),
            "spent_row_ids": list(map(int, registration["spent_row_ids"])),
            "sealed_row_ids": list(map(int, registration["sealed_row_ids"])),
            "split_scope_fingerprint": str(registration["split_scope_fingerprint"]),
            "data_projection_sha256": str(registration["data_projection_sha256"]),
            "catalog_sha256": str(registration["catalog_sha256"]),
            "content_sha256": str(registration["content_sha256"]),
            "registration": copy.deepcopy(dict(registration["registration"])),
        }
        for registration in scope_registrations
    ]
    if len({row["scope_id"] for row in rows}) != len(rows):
        raise ValueError("cumulative legacy index contains duplicate scopes")
    rows.sort(key=lambda row: (row["outer_fold"], row["context_epoch"]))
    body = {
        "schema_version": STAGE1_CUMULATIVE_LEGACY_NATIVE_INDEX_SCHEMA,
        "request_sha256": str(request_sha256),
        "schedule_sha256": str(schedule_sha256),
        "split_registry_content_sha256": str(split_registry_content_sha256),
        "registered_families": list(PRODUCTION_CUMULATIVE_LEGACY_NATIVE_FAMILY_ADAPTERS),
        "cumulative_scope_count": len(rows),
        "sealed_text_available_to_producers": False,
        "sealed_labels_available_to_producers": False,
        "scopes": rows,
    }
    target = Path(index_path)
    if not target.is_absolute():
        target = root / target
    _write_immutable_json(target, {**body, "content_sha256": _sha256_json(body)})
    return _component_file_registration(target, component_root=root)


def _validate_legacy_cumulative_spent_native_index(
    *,
    component_root: Path,
    index_registration: Mapping[str, Any],
    expected_requests: Mapping[str, CumulativeSpentStage1FamilyRequest],
    expected_configuration_by_scope: Mapping[
        str,
        Mapping[str, Mapping[str, Any]],
    ],
    request_sha256: str,
    schedule_sha256: str,
    split_registry_content_sha256: str,
    htr_model_path: Path | None = None,
    htr_model_sha256: str | None = None,
    device: torch.device | str = "cpu",
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    """Reconstruct every cumulative legacy producer from its component bytes."""

    root = Path(component_root).resolve(strict=True)
    file_fields = {"relative_path", "size", "sha256"}
    directory_fields = {"relative_path", "kind", "file_count", "size", "sha256"}
    if not isinstance(index_registration, Mapping) or set(index_registration) != file_fields:
        raise ValueError("cumulative legacy index registration is not closed")
    index_path = _validate_component_native_registration(root, index_registration)
    index = _read_json_object_reject_duplicates(
        index_path,
        field_name="cumulative legacy native index",
    )
    index_fields = {
        "schema_version",
        "request_sha256",
        "schedule_sha256",
        "split_registry_content_sha256",
        "registered_families",
        "cumulative_scope_count",
        "sealed_text_available_to_producers",
        "sealed_labels_available_to_producers",
        "scopes",
        "content_sha256",
    }
    body = {key: value for key, value in index.items() if key != "content_sha256"}
    families = list(PRODUCTION_CUMULATIVE_LEGACY_NATIVE_FAMILY_ADAPTERS)
    scopes = index.get("scopes")
    if (
        set(index) != index_fields
        or index.get("schema_version") != STAGE1_CUMULATIVE_LEGACY_NATIVE_INDEX_SCHEMA
        or index.get("request_sha256") != request_sha256
        or index.get("schedule_sha256") != schedule_sha256
        or index.get("split_registry_content_sha256") != split_registry_content_sha256
        or index.get("registered_families") != families
        or index.get("sealed_text_available_to_producers") is not False
        or index.get("sealed_labels_available_to_producers") is not False
        or not isinstance(scopes, list)
        or int(index.get("cumulative_scope_count", -1)) != len(scopes)
        or index.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("cumulative legacy native index has an invalid closed envelope")
    scope_fields = {
        "scope_id",
        "outer_fold",
        "context_epoch",
        "provider_inner_fold",
        "spent_row_ids",
        "sealed_row_ids",
        "split_scope_fingerprint",
        "data_projection_sha256",
        "catalog_sha256",
        "content_sha256",
        "registration",
    }
    indexed = {
        str(row.get("scope_id")): row
        for row in scopes
        if isinstance(row, Mapping) and set(row) == scope_fields
    }
    if (
        len(indexed) != len(scopes)
        or set(indexed) != set(expected_requests)
        or set(expected_configuration_by_scope) != set(expected_requests)
    ):
        raise ValueError("cumulative legacy native index has incomplete scope coverage")
    registration_fields = {
        "schema_version",
        "request_sha256",
        "schedule_sha256",
        "scope_id",
        "outer_fold",
        "context_epoch",
        "provider_inner_fold",
        "spent_row_ids",
        "sealed_row_ids",
        "split_scope_fingerprint",
        "data_projection_sha256",
        "catalog_sha256",
        "registered_families",
        "replay_canary",
        "sealed_text_accessed",
        "sealed_labels_accessed",
        "oracle_fields_accessed",
        "secrets_accessed",
        "family_proofs",
        "content_sha256",
    }
    family_fields = {
        "family",
        "request_binding_sha256",
        "fit_semantics",
        "evidence_item_count",
        "producer_configuration",
        "producer_identity",
        "fit_audit",
        "replay_canary",
        "evidence_payload",
        "execution_record",
        "model_artifact",
        "source_artifact",
        "content_sha256",
    }
    producers_by_scope: dict[str, dict[str, Any]] = {}
    for scope_id, expected_request in expected_requests.items():
        row = indexed[scope_id]
        descriptor = row.get("registration")
        if not isinstance(descriptor, Mapping) or set(descriptor) != file_fields:
            raise ValueError(f"cumulative legacy scope descriptor is open: {scope_id}")
        path = _validate_component_native_registration(root, descriptor)
        registration = _read_json_object_reject_duplicates(
            path,
            field_name=f"cumulative legacy scope registration {scope_id}",
        )
        registration_body = {
            key: value for key, value in registration.items() if key != "content_sha256"
        }
        proofs = registration.get("family_proofs")
        expected_configs = expected_configuration_by_scope[scope_id]
        if (
            set(registration) != registration_fields
            or registration.get("schema_version") != STAGE1_CUMULATIVE_LEGACY_NATIVE_SCOPE_SCHEMA
            or registration.get("request_sha256") != expected_request.request_sha256
            or registration.get("schedule_sha256") != expected_request.schedule_sha256
            or registration.get("scope_id") != scope_id
            or int(registration.get("outer_fold", 0)) != expected_request.outer_fold
            or int(registration.get("context_epoch", -1)) != expected_request.context_epoch
            or int(registration.get("provider_inner_fold", 0))
            != expected_request.provider_inner_fold
            or tuple(map(int, registration.get("spent_row_ids") or ()))
            != expected_request.spent_row_ids
            or tuple(map(int, registration.get("sealed_row_ids") or ()))
            != expected_request.sealed_row_ids
            or registration.get("split_scope_fingerprint")
            != expected_request.split_scope_fingerprint
            or registration.get("data_projection_sha256") != expected_request.data_projection_sha256
            or registration.get("registered_families") != families
            or registration.get("sealed_text_accessed") is not False
            or registration.get("sealed_labels_accessed") is not False
            or registration.get("oracle_fields_accessed") is not False
            or registration.get("secrets_accessed") is not False
            or registration.get("content_sha256") != _sha256_json(registration_body)
            or row.get("content_sha256") != registration.get("content_sha256")
            or row.get("data_projection_sha256") != expected_request.data_projection_sha256
            or not isinstance(proofs, list)
            or len(proofs) != len(families)
            or set(expected_configs) != set(families)
        ):
            raise ValueError(f"cumulative legacy scope registration is invalid: {scope_id}")
        canary = CumulativeSpentReplayCanary.from_request(expected_request)
        if registration.get("replay_canary") != canary.binding:
            raise ValueError("cumulative legacy replay canary changed after fitting")
        scope_producers: dict[str, Any] = {}
        for family, raw_proof in zip(families, proofs):
            if not isinstance(raw_proof, Mapping) or set(raw_proof) != family_fields:
                raise ValueError(f"cumulative legacy family proof is open: {scope_id}/{family}")
            proof_body = {key: value for key, value in raw_proof.items() if key != "content_sha256"}
            family_request = _cumulative_request_for_family(
                expected_request,
                family=family,
            )
            configuration = copy.deepcopy(dict(expected_configs[family]))
            identity = cumulative_spent_native_family_identity(
                family=family,
                configuration=configuration,
            )
            if (
                raw_proof.get("family") != family
                or raw_proof.get("request_binding_sha256") != family_request.binding_sha256
                or raw_proof.get("fit_semantics") != CUMULATIVE_SPENT_REFIT
                or raw_proof.get("producer_configuration") != configuration
                or raw_proof.get("producer_identity") != identity
                or raw_proof.get("replay_canary") != canary.binding
                or raw_proof.get("content_sha256") != _sha256_json(proof_body)
            ):
                raise ValueError(
                    f"cumulative legacy family proof identity is invalid: {scope_id}/{family}"
                )
            descriptors = {
                key: raw_proof.get(key)
                for key in (
                    "evidence_payload",
                    "execution_record",
                    "model_artifact",
                    "source_artifact",
                )
            }
            for key, artifact_descriptor in descriptors.items():
                expected_fields = directory_fields if key == "model_artifact" else file_fields
                if (
                    not isinstance(artifact_descriptor, Mapping)
                    or set(artifact_descriptor) != expected_fields
                ):
                    raise ValueError(
                        f"cumulative legacy {key} descriptor is open: {scope_id}/{family}"
                    )
            paths = {
                key: _validate_component_native_registration(root, artifact_descriptor)
                for key, artifact_descriptor in descriptors.items()
            }
            if paths["source_artifact"] != paths["evidence_payload"]:
                raise ValueError("cumulative legacy source must be its family payload")
            payload = _read_json_object_reject_duplicates(
                paths["evidence_payload"],
                field_name=f"cumulative legacy payload {scope_id}/{family}",
            )
            execution = _read_json_object_reject_duplicates(
                paths["execution_record"],
                field_name=f"cumulative legacy execution {scope_id}/{family}",
            )
            if (
                execution.get("schema_version") != CUMULATIVE_SPENT_NATIVE_EXECUTION_RECORD_SCHEMA
                or execution.get("request_binding_sha256") != family_request.binding_sha256
                or execution.get("fit_semantics") != CUMULATIVE_SPENT_REFIT
                or execution.get("sealed_text_accessed") is not False
                or execution.get("sealed_labels_accessed") is not False
                or execution.get("oracle_fields_accessed") is not False
                or execution.get("secrets_accessed") is not False
                or execution.get("replay_canary_contributes_to_concept_evidence") is not False
                or execution.get("executable_serialization_used") is not False
            ):
                raise ValueError(
                    f"cumulative legacy execution security is invalid: {scope_id}/{family}"
                )
            producer = bind_cumulative_spent_native_family_producer(
                request=family_request,
                producer_identity=identity,
                evidence_payload=payload,
                evidence_item_count=int(raw_proof["evidence_item_count"]),
                replay_canary=canary,
                capture_artifact_path=paths["model_artifact"],
                source_artifact_path=paths["source_artifact"],
                execution_record_path=paths["execution_record"],
                htr_model_path=(
                    htr_model_path if family in {HTR_NEURAL, MATCHED_PAIR_UPLIFT} else None
                ),
                expected_htr_model_tree_sha256=(
                    htr_model_sha256 if family in {HTR_NEURAL, MATCHED_PAIR_UPLIFT} else None
                ),
                device=device,
            )
            draft = producer.produce_cumulative_spent(family_request)
            if (
                draft.evidence_payload != payload
                or int(draft.evidence_item_count) != int(raw_proof["evidence_item_count"])
                or draft.fit_audit != raw_proof.get("fit_audit")
            ):
                raise RuntimeError(f"cumulative legacy family replay changed: {scope_id}/{family}")
            scope_producers[family] = producer
        producers_by_scope[scope_id] = scope_producers
    return copy.deepcopy(index), producers_by_scope


def _htr_capture_family_bindings(capture: Mapping[str, Any]) -> Mapping[str, Any]:
    inventory = capture.get("array_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("HTR capture has no numerical inventory")

    def numerical(key: Any) -> Mapping[str, Any]:
        name = str(key or "")
        row = inventory.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"HTR capture lacks numerical state: {name}")
        return copy.deepcopy(dict(row))

    def model_binding(model: Mapping[str, Any]) -> Mapping[str, Any]:
        tensors = model.get("state_tensors")
        if not isinstance(tensors, list) or not tensors:
            raise ValueError("HTR captured model has no tensor state")
        return {
            "kind": model.get("kind"),
            "class_name": model.get("class_name"),
            "hidden_dim": model.get("hidden_dim"),
            "outcome_type": model.get("outcome_type"),
            "extractor": copy.deepcopy(model.get("extractor")),
            "state_sha256": model.get("state_sha256"),
            "state_tensors": [
                {
                    "state_key": row.get("state_key"),
                    "torch_dtype": row.get("torch_dtype"),
                    "shape": copy.deepcopy(row.get("shape")),
                    "numerical": numerical(row.get("array")),
                }
                for row in tensors
            ],
        }

    def calibrator_binding(row: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if row is None:
            return None
        result = {
            "class_name": row.get("class_name"),
            "method": row.get("method"),
            "temperature": row.get("temperature"),
            "isotonic": row.get("isotonic"),
        }
        if row.get("isotonic") is True:
            result["x_thresholds"] = numerical(row.get("x_thresholds"))
            result["y_thresholds"] = numerical(row.get("y_thresholds"))
        return result

    nuisance_keys = (
        "fit_treatment",
        "fit_outcome",
        "validation_treatment",
        "validation_outcome",
        "fit_e_raw",
        "fit_m_raw",
        "validation_e_raw",
        "validation_m_raw",
        "validation_e_hat",
        "validation_m_hat",
        "heldout_e_raw",
        "heldout_m_raw",
        "heldout_e_hat",
        "heldout_m_hat",
    )
    nuisance_rows = []
    for row in capture.get("nuisance_fold_states") or ():
        if not isinstance(row, Mapping):
            raise ValueError("HTR nuisance fold binding is malformed")
        nuisance_rows.append(
            {
                "fold": row.get("fold"),
                "objective": row.get("objective"),
                "split_seed": row.get("split_seed"),
                "fit_row_ids": copy.deepcopy(row.get("fit_row_ids")),
                "validation_row_ids": copy.deepcopy(row.get("validation_row_ids")),
                "fit_row_fingerprint": row.get("fit_row_fingerprint"),
                "validation_row_fingerprint": row.get("validation_row_fingerprint"),
                "model": model_binding(row.get("model") or {}),
                "propensity_calibrator": calibrator_binding(row.get("propensity_calibrator")),
                "outcome_calibrator": calibrator_binding(row.get("outcome_calibrator")),
                "numerical": {key: numerical(row.get(key)) for key in nuisance_keys},
                "heldout_labels_accessed": False,
            }
        )
    effect_keys = (
        "treatment",
        "outcome",
        "e_hat",
        "m_hat",
        "e_clipped",
        "y_residual",
        "t_residual",
        "r_pseudo_outcome",
        "train_eligible",
        "validation_raw_effect",
        "validation_tau",
        "validation_r_loss",
        "validation_effect_loss",
        "heldout_raw_effect",
        "heldout_tau",
    )
    effect_rows = []
    for row in capture.get("effect_fold_states") or ():
        if not isinstance(row, Mapping):
            raise ValueError("HTR effect fold binding is malformed")
        effect_rows.append(
            {
                "fold": row.get("fold"),
                "objective": row.get("objective"),
                "effect_objective": row.get("effect_objective"),
                "split_seed": row.get("split_seed"),
                "fit_row_ids": copy.deepcopy(row.get("fit_row_ids")),
                "eligible_fit_row_ids": copy.deepcopy(row.get("eligible_fit_row_ids")),
                "validation_row_ids": copy.deepcopy(row.get("validation_row_ids")),
                "fit_row_fingerprint": row.get("fit_row_fingerprint"),
                "eligible_fit_row_fingerprint": row.get("eligible_fit_row_fingerprint"),
                "validation_row_fingerprint": row.get("validation_row_fingerprint"),
                "r_stage_min_propensity": row.get("r_stage_min_propensity"),
                "r_stage_max_propensity": row.get("r_stage_max_propensity"),
                "model": model_binding(row.get("model") or {}),
                "numerical": {key: numerical(row.get(key)) for key in effect_keys},
                "heldout_labels_accessed": False,
            }
        )
    scope_outputs = capture.get("scope_outputs")
    if not isinstance(scope_outputs, Mapping):
        raise ValueError("HTR capture has no final scope outputs")
    return {
        "schema_version": HTR_NATIVE_CAPTURE_SCHEMA,
        "family": HTR_NEURAL,
        "extractor_identity": copy.deepcopy(capture.get("extractor_identity")),
        "model_tree_sha256": capture.get("model_tree_sha256"),
        "e_clip": capture.get("e_clip"),
        "nuisance_folds": capture.get("nuisance_folds"),
        "effect_folds": capture.get("effect_folds"),
        "prediction_batch_size": capture.get("prediction_batch_size"),
        "seed": capture.get("seed"),
        "nuisance_fold_states": nuisance_rows,
        "effect_fold_states": effect_rows,
        "scope_numerical_outputs": {
            str(name): {
                "role": row.get("role"),
                **numerical(row.get("array")),
            }
            for name, row in sorted(scope_outputs.items())
            if isinstance(row, Mapping)
        },
        "heldout_labels_accessed": False,
    }


def _register_htr_native_family_proof(
    *,
    component_root: Path,
    proof_directory: Path,
    scope_id: str,
    catalog: RoleNeutralEvidenceCatalog,
    capture_artifact_path: Path,
    source_artifact_path: Path,
    outer_fold: int,
    inner_fold: int,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    fit_texts: Sequence[str],
    heldout_texts: Sequence[str],
    fit_treatment: Sequence[float],
    fit_outcome: Sequence[float],
    split_scope_fingerprint: str,
    data_projection_sha256: str,
    configuration: Mapping[str, Any],
    htr_model_path: Path,
    htr_model_sha256: str,
    device: torch.device | str,
) -> Mapping[str, Any]:
    """Register the real nested HTR nuisance/effect producer after replay."""

    if int(outer_fold) < 1 or int(inner_fold) < 1:
        raise ValueError("HTR native proof registration requires an exact-inner scope")
    root = Path(component_root).resolve(strict=True)
    fit_rows = tuple(map(int, fit_row_ids))
    heldout_rows = tuple(map(int, heldout_row_ids))
    source_path = _component_regular_file(
        root,
        source_artifact_path,
        field_name="HTR raw evidence sidecar",
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("schema_version") != STAGE1_RAW_EVIDENCE_SIDECAR_SCHEMA
        or source.get("scope_id") != str(scope_id)
        or int(source.get("outer_fold", 0)) != int(outer_fold)
        or int(source.get("inner_fold", 0)) != int(inner_fold)
        or source.get("fit_row_fingerprint") != row_set_fingerprint(fit_rows)
        or source.get("heldout_row_fingerprint") != row_set_fingerprint(heldout_rows)
        or source.get("prompt_grounding_allowed") is not False
    ):
        raise ValueError("HTR source artifact changed its exact-inner scope")
    model_registration = _component_native_artifact_registration(
        Path(capture_artifact_path),
        component_root=root,
    )
    if model_registration["kind"] != "directory":
        raise ValueError("HTR native capture must be one directory artifact")
    model_path = root / str(model_registration["relative_path"])
    capture = validate_htr_native_capture(
        model_path,
        expected_scope_id=scope_id,
        expected_fit_row_ids=fit_rows,
        expected_heldout_row_ids=heldout_rows,
        fit_texts=fit_texts,
        heldout_texts=heldout_texts,
        expected_fit_treatment=fit_treatment,
        expected_fit_outcome=fit_outcome,
        htr_model_path=htr_model_path,
        expected_model_tree_sha256=htr_model_sha256,
        device=device,
    )
    if (
        capture.get("outer_fold") != int(outer_fold)
        or capture.get("inner_fold") != int(inner_fold)
        or configuration.get("scope_id") != str(scope_id)
        or configuration.get("heldout_label_policy") != "id_and_text_only"
        or configuration.get("capture_schema_version") != HTR_NATIVE_CAPTURE_SCHEMA
        or configuration.get("htr_model_tree_sha256") != htr_model_sha256
    ):
        raise ValueError("HTR native configuration or capture changed scope identity")
    proof_root = Path(proof_directory)
    if not proof_root.is_absolute():
        proof_root = root / proof_root
    if proof_root.exists() and proof_root.is_symlink():
        raise ValueError("HTR proof directory cannot be a symlink")
    proof_root.mkdir(parents=True, exist_ok=True)
    try:
        proof_root.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError("HTR proof directory escapes its component") from exc
    evidence_payload, evidence_item_count = family_payload_from_catalog(
        catalog,
        family=HTR_NEURAL,
    )
    if int(evidence_item_count) < 1:
        raise RuntimeError("HTR native scope has no catalog evidence")
    payload_path = proof_root / f"{HTR_NEURAL}.evidence_payload.json"
    metadata_path = proof_root / f"{HTR_NEURAL}.fit_metadata.json"
    execution_path = proof_root / f"{HTR_NEURAL}.execution.json"
    _write_immutable_json(payload_path, evidence_payload)
    fit_metadata_body = {
        "schema_version": STAGE1_HTR_NATIVE_FIT_METADATA_SCHEMA,
        "family": HTR_NEURAL,
        "scope_id": str(scope_id),
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "fit_semantics": EXACT_INNER_REFIT,
        "fit_row_ids": list(fit_rows),
        "heldout_row_ids": list(heldout_rows),
        "fit_row_fingerprint": row_set_fingerprint(fit_rows),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_rows),
        "split_scope_fingerprint": str(split_scope_fingerprint),
        "data_projection_sha256": str(data_projection_sha256),
        "capture_schema_version": HTR_NATIVE_CAPTURE_SCHEMA,
        "capture_content_sha256": capture["content_sha256"],
        "capture_artifact_sha256": model_registration["sha256"],
        "source_artifact_sha256": _sha256_file(source_path),
        "configuration": copy.deepcopy(dict(configuration)),
        "family_state_bindings": _htr_capture_family_bindings(capture),
        "heldout_columns_read": ["_oci_row_id", capture["text_column"]],
        "heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "executable_checkpoint_retained": False,
        "torch_checkpoint_loaded": False,
    }
    fit_metadata = {
        **fit_metadata_body,
        "content_sha256": _sha256_json(fit_metadata_body),
    }
    _write_immutable_json(metadata_path, fit_metadata)
    semantics = (
        "non-executable JSON/NPZ exact HierarchicalTransformerExtractor tensor "
        "state with nested calibrator, nuisance, effect-objective, validation, "
        "and ID/text-only heldout replay"
    )
    execution_record = native_family_execution_record(
        family=HTR_NEURAL,
        fit_semantics=EXACT_INNER_REFIT,
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        split_scope_fingerprint=split_scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        evidence_payload=evidence_payload,
        configuration=configuration,
        native_fit_metadata_path=metadata_path,
        model_artifact_path=model_path,
        source_artifact_path=source_path,
        model_artifact_semantics=semantics,
    )
    _write_immutable_json(execution_path, execution_record)
    proof = bind_native_family_fit_proof(
        family=HTR_NEURAL,
        fit_semantics=EXACT_INNER_REFIT,
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        split_scope_fingerprint=split_scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        evidence_payload=evidence_payload,
        configuration=configuration,
        native_fit_metadata_path=metadata_path,
        native_execution_record_path=execution_path,
        model_artifact_path=model_path,
        source_artifact_path=source_path,
        model_artifact_semantics=semantics,
    )
    proof.verify_artifact_bytes()
    family_row = {
        "family": HTR_NEURAL,
        "evidence_item_count": int(evidence_item_count),
        "proof": proof.as_dict(),
        "evidence_payload": _component_file_registration(
            payload_path,
            component_root=root,
        ),
        "native_execution_record": _component_file_registration(
            execution_path,
            component_root=root,
        ),
        "native_fit_metadata": _component_file_registration(
            metadata_path,
            component_root=root,
        ),
        "model_artifact": copy.deepcopy(model_registration),
        "source_artifact": _component_file_registration(
            source_path,
            component_root=root,
        ),
    }
    registration_body = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
        "scope_id": str(scope_id),
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "fit_row_ids": list(fit_rows),
        "heldout_row_ids": list(heldout_rows),
        "split_scope_fingerprint": str(split_scope_fingerprint),
        "data_projection_sha256": str(data_projection_sha256),
        "fit_semantics": EXACT_INNER_REFIT,
        "registered_families": list(PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS),
        "heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "family_proofs": [family_row],
    }
    registration = {
        **registration_body,
        "content_sha256": _sha256_json(registration_body),
    }
    registration_path = proof_root / "registration.json"
    _write_immutable_json(registration_path, registration)
    return {
        **registration,
        "registration": _component_file_registration(
            registration_path,
            component_root=root,
        ),
    }


def _validate_htr_native_family_proof_index(
    *,
    component_root: Path,
    index_registration: Mapping[str, Any],
    expected_inner_scopes: Mapping[str, Mapping[str, Any]],
    split_registry_content_sha256: str,
    modeling_data: pd.DataFrame,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    htr_model_path: Path,
    htr_model_sha256: str,
    device: torch.device | str,
    reloaded_native_by_scope: dict[str, dict[str, Mapping[str, Any]]] | None = None,
) -> Mapping[str, Any]:
    root = Path(component_root).resolve(strict=True)
    index_path = _validate_component_native_registration(root, index_registration)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in index.items() if key != "content_sha256"}
    scopes = index.get("scopes")
    if (
        index.get("schema_version") != STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA
        or index.get("split_registry_content_sha256") != split_registry_content_sha256
        or index.get("registered_families")
        != list(PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS)
        or index.get("executable_checkpoint_files_retained") is not False
        or index.get("content_sha256") != _sha256_json(body)
        or not isinstance(scopes, list)
        or int(index.get("exact_inner_scope_count", -1)) != len(scopes)
    ):
        raise ValueError("HTR native proof index has an invalid closed envelope")
    indexed = {str(row.get("scope_id")): row for row in scopes if isinstance(row, Mapping)}
    if len(indexed) != len(scopes) or set(indexed) != set(expected_inner_scopes):
        raise ValueError("HTR native proof index has incomplete exact-inner coverage")
    semantics = (
        "non-executable JSON/NPZ exact HierarchicalTransformerExtractor tensor "
        "state with nested calibrator, nuisance, effect-objective, validation, "
        "and ID/text-only heldout replay"
    )
    for scope_id, expected in expected_inner_scopes.items():
        row = indexed[scope_id]
        registration_path = _validate_component_native_registration(
            root,
            row.get("registration") or {},
        )
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        registration_body = {
            key: value for key, value in registration.items() if key != "content_sha256"
        }
        family_rows = registration.get("family_proofs")
        if (
            int(row.get("outer_fold", 0)) != int(expected["outer_fold"])
            or int(row.get("inner_fold", 0)) != int(expected["inner_fold"])
            or row.get("registered_families")
            != list(PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS)
            or registration.get("scope_id") != scope_id
            or registration.get("registered_families")
            != list(PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS)
            or registration.get("content_sha256") != _sha256_json(registration_body)
            or registration.get("content_sha256") != row.get("content_sha256")
            or tuple(map(int, registration.get("fit_row_ids") or ()))
            != tuple(map(int, expected["fit_row_ids"]))
            or tuple(map(int, registration.get("heldout_row_ids") or ()))
            != tuple(map(int, expected["heldout_row_ids"]))
            or registration.get("heldout_labels_accessed") is not False
            or not isinstance(family_rows, list)
            or len(family_rows) != 1
            or family_rows[0].get("family") != HTR_NEURAL
        ):
            raise ValueError(f"HTR native proof registration is invalid: {scope_id}")
        family_row = family_rows[0]
        paths = {
            key: _validate_component_native_registration(
                root,
                family_row.get(key) or {},
            )
            for key in (
                "evidence_payload",
                "native_execution_record",
                "native_fit_metadata",
                "model_artifact",
                "source_artifact",
            )
        }
        fit_rows = tuple(map(int, expected["fit_row_ids"]))
        heldout_rows = tuple(map(int, expected["heldout_row_ids"]))
        fit_texts = tuple(
            str(value) for value in modeling_data.iloc[list(fit_rows)][text_column]
        )
        heldout_texts = tuple(
            str(value) for value in modeling_data.iloc[list(heldout_rows)][text_column]
        )
        fit_treatment = tuple(
            modeling_data.iloc[list(fit_rows)][treatment_column].to_numpy(dtype=float)
        )
        fit_outcome = tuple(
            modeling_data.iloc[list(fit_rows)][outcome_column].to_numpy(dtype=float)
        )
        capture = validate_htr_native_capture(
            paths["model_artifact"],
            expected_scope_id=scope_id,
            expected_fit_row_ids=fit_rows,
            expected_heldout_row_ids=heldout_rows,
            fit_texts=fit_texts,
            heldout_texts=heldout_texts,
            expected_fit_treatment=fit_treatment,
            expected_fit_outcome=fit_outcome,
            htr_model_path=htr_model_path,
            expected_model_tree_sha256=htr_model_sha256,
            device=device,
        )
        metadata = json.loads(paths["native_fit_metadata"].read_text(encoding="utf-8"))
        metadata_body = {key: value for key, value in metadata.items() if key != "content_sha256"}
        if (
            metadata.get("schema_version") != STAGE1_HTR_NATIVE_FIT_METADATA_SCHEMA
            or metadata.get("family") != HTR_NEURAL
            or metadata.get("capture_content_sha256") != capture["content_sha256"]
            or metadata.get("content_sha256") != _sha256_json(metadata_body)
            or metadata.get("heldout_labels_accessed") is not False
            or metadata.get("family_state_bindings") != _htr_capture_family_bindings(capture)
        ):
            raise ValueError(f"HTR native fit metadata is invalid: {scope_id}")
        evidence_payload = json.loads(paths["evidence_payload"].read_text(encoding="utf-8"))
        rebound = bind_native_family_fit_proof(
            family=HTR_NEURAL,
            fit_semantics=EXACT_INNER_REFIT,
            outer_fold=int(expected["outer_fold"]),
            inner_fold=int(expected["inner_fold"]),
            split_scope_fingerprint=str(registration["split_scope_fingerprint"]),
            data_projection_sha256=str(registration["data_projection_sha256"]),
            fit_row_ids=fit_rows,
            heldout_row_ids=heldout_rows,
            evidence_payload=evidence_payload,
            configuration=metadata["configuration"],
            native_fit_metadata_path=paths["native_fit_metadata"],
            native_execution_record_path=paths["native_execution_record"],
            model_artifact_path=paths["model_artifact"],
            source_artifact_path=paths["source_artifact"],
            model_artifact_semantics=semantics,
        )
        if rebound.as_dict() != family_row.get("proof"):
            raise RuntimeError(f"HTR native proof identity changed: {scope_id}")
        _record_reloaded_exact_inner_family(
            reloaded_native_by_scope,
            scope_id=scope_id,
            family=HTR_NEURAL,
            proof=rebound,
            evidence_payload=evidence_payload,
            artifact_paths=paths,
        )
    return copy.deepcopy(index)


def _matched_pair_capture_family_bindings(
    capture: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind both native pair subproducers and every retained numerical array."""

    inventory = capture.get("array_inventory")
    if (
        capture.get("schema_version") != MATCHED_PAIR_NATIVE_CAPTURE_SCHEMA
        or capture.get("subproducer_coverage") != ["bow", "htr"]
        or not isinstance(inventory, Mapping)
        or not capture.get("bow_fold_states")
        or not capture.get("bow_full_fit_states")
        or not capture.get("htr_fold_states")
        or not isinstance(capture.get("scope_outputs"), Mapping)
        or capture.get("heldout_labels_accessed") is not False
    ):
        raise ValueError("matched-pair capture lacks complete native subproducer state")
    return {
        "schema_version": MATCHED_PAIR_NATIVE_CAPTURE_SCHEMA,
        "family": MATCHED_PAIR_UPLIFT,
        "capture_content_sha256": capture.get("content_sha256"),
        "subproducer_coverage": ["bow", "htr"],
        "effect_folds": capture.get("effect_folds"),
        "view_configs": copy.deepcopy(capture.get("view_configs")),
        "matching_configuration": copy.deepcopy(capture.get("matching_configuration")),
        "htr_model_tree_sha256": capture.get("htr_model_tree_sha256"),
        "htr_extractor_identity": copy.deepcopy(capture.get("htr_extractor_identity")),
        "scope_inputs": copy.deepcopy(capture.get("scope_inputs")),
        "bow_fold_states": copy.deepcopy(capture.get("bow_fold_states")),
        "bow_full_fit_states": copy.deepcopy(capture.get("bow_full_fit_states")),
        "htr_fold_states": copy.deepcopy(capture.get("htr_fold_states")),
        "scope_outputs": copy.deepcopy(capture.get("scope_outputs")),
        "array_inventory": copy.deepcopy(dict(sorted(inventory.items()))),
        "heldout_labels_accessed": False,
    }


def _require_matched_pair_source_proofs(
    source: Mapping[str, Any],
    *,
    scope_id: str,
) -> Mapping[str, Any]:
    proofs = source.get("matched_pair_subproducer_proofs")
    subproducers = proofs.get("subproducers") if isinstance(proofs, Mapping) else None
    if (
        not isinstance(proofs, Mapping)
        or proofs.get("schema_version") != STAGE1_MATCHED_PAIR_PROOF_SCHEMA
        or proofs.get("scope_id") != str(scope_id)
        or proofs.get("all_required_subproducers_succeeded") is not True
        or not isinstance(subproducers, Mapping)
        or set(subproducers) != {"bow", "htr"}
        or any(
            not isinstance(subproducers[name], Mapping)
            or subproducers[name].get("subproducer") != name
            or subproducers[name].get("success") is not True
            or subproducers[name].get("artifact_semantics")
            != "sealed_model_outputs_and_concept_evidence"
            for name in ("bow", "htr")
        )
    ):
        raise ValueError("matched-pair source lacks both successful subproducer proofs")
    proof_body = {name: copy.deepcopy(subproducers[name]) for name in ("bow", "htr")}
    if proofs.get("content_sha256") != _sha256_json(proof_body):
        raise ValueError("matched-pair source subproducer proof digest changed")
    return copy.deepcopy(dict(proofs))


def _register_matched_pair_native_family_proof(
    *,
    component_root: Path,
    proof_directory: Path,
    scope_id: str,
    catalog: RoleNeutralEvidenceCatalog,
    capture_artifact_path: Path,
    source_artifact_path: Path,
    outer_fold: int,
    inner_fold: int,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    fit_texts: Sequence[str],
    heldout_texts: Sequence[str],
    fit_treatment: Sequence[float],
    fit_outcome: Sequence[float],
    split_scope_fingerprint: str,
    data_projection_sha256: str,
    configuration: Mapping[str, Any],
    htr_model_path: Path | None,
    htr_model_sha256: str | None,
    device: torch.device | str,
) -> Mapping[str, Any]:
    """Register one genuine paired BoW+HTR matched-uplift producer."""

    if int(outer_fold) < 1 or int(inner_fold) < 1:
        raise ValueError("matched-pair native proof requires an exact-inner scope")
    root = Path(component_root).resolve(strict=True)
    fit_rows = tuple(map(int, fit_row_ids))
    heldout_rows = tuple(map(int, heldout_row_ids))
    source_path = _component_regular_file(
        root,
        source_artifact_path,
        field_name="matched-pair raw evidence sidecar",
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("schema_version") != STAGE1_RAW_EVIDENCE_SIDECAR_SCHEMA
        or source.get("scope_id") != str(scope_id)
        or int(source.get("outer_fold", 0)) != int(outer_fold)
        or int(source.get("inner_fold", 0)) != int(inner_fold)
        or source.get("fit_row_fingerprint") != row_set_fingerprint(fit_rows)
        or source.get("heldout_row_fingerprint") != row_set_fingerprint(heldout_rows)
        or source.get("prompt_grounding_allowed") is not False
    ):
        raise ValueError("matched-pair source changed its exact-inner scope")
    source_subproducer_proofs = _require_matched_pair_source_proofs(
        source,
        scope_id=scope_id,
    )
    model_registration = _component_native_artifact_registration(
        Path(capture_artifact_path),
        component_root=root,
    )
    if model_registration["kind"] != "directory":
        raise ValueError("matched-pair native capture must be one directory artifact")
    model_path = root / str(model_registration["relative_path"])
    capture = validate_matched_pair_native_capture(
        model_path,
        expected_scope_id=scope_id,
        expected_fit_row_ids=fit_rows,
        expected_heldout_row_ids=heldout_rows,
        fit_texts=fit_texts,
        heldout_texts=heldout_texts,
        expected_fit_treatment=fit_treatment,
        expected_fit_outcome=fit_outcome,
        htr_model_path=htr_model_path,
        expected_htr_model_tree_sha256=htr_model_sha256,
        device=device,
    )
    if (
        capture.get("outer_fold") != int(outer_fold)
        or capture.get("inner_fold") != int(inner_fold)
        or configuration.get("scope_id") != str(scope_id)
        or configuration.get("heldout_label_policy") != "id_and_text_only"
        or configuration.get("capture_schema_version") != MATCHED_PAIR_NATIVE_CAPTURE_SCHEMA
        or configuration.get("required_subproducers") != ["bow", "htr"]
        or configuration.get("htr_model_tree_sha256") != htr_model_sha256
    ):
        raise ValueError("matched-pair native configuration or capture changed scope")
    proof_root = Path(proof_directory)
    if not proof_root.is_absolute():
        proof_root = root / proof_root
    if proof_root.exists() and proof_root.is_symlink():
        raise ValueError("matched-pair proof directory cannot be a symlink")
    proof_root.mkdir(parents=True, exist_ok=True)
    try:
        proof_root.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError("matched-pair proof directory escapes its component") from exc
    evidence_payload, evidence_item_count = family_payload_from_catalog(
        catalog,
        family=MATCHED_PAIR_UPLIFT,
    )
    if int(evidence_item_count) < 1:
        raise RuntimeError("matched-pair native scope has no catalog evidence")
    payload_path = proof_root / f"{MATCHED_PAIR_UPLIFT}.evidence_payload.json"
    metadata_path = proof_root / f"{MATCHED_PAIR_UPLIFT}.fit_metadata.json"
    execution_path = proof_root / f"{MATCHED_PAIR_UPLIFT}.execution.json"
    _write_immutable_json(payload_path, evidence_payload)
    bindings = _matched_pair_capture_family_bindings(capture)
    fit_metadata_body = {
        "schema_version": STAGE1_MATCHED_PAIR_NATIVE_FIT_METADATA_SCHEMA,
        "family": MATCHED_PAIR_UPLIFT,
        "scope_id": str(scope_id),
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "fit_semantics": EXACT_INNER_REFIT,
        "fit_row_ids": list(fit_rows),
        "heldout_row_ids": list(heldout_rows),
        "fit_row_fingerprint": row_set_fingerprint(fit_rows),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_rows),
        "split_scope_fingerprint": str(split_scope_fingerprint),
        "data_projection_sha256": str(data_projection_sha256),
        "capture_schema_version": MATCHED_PAIR_NATIVE_CAPTURE_SCHEMA,
        "capture_content_sha256": capture["content_sha256"],
        "capture_artifact_sha256": model_registration["sha256"],
        "source_artifact_sha256": _sha256_file(source_path),
        "source_subproducer_proofs_sha256": source_subproducer_proofs["content_sha256"],
        "configuration": copy.deepcopy(dict(configuration)),
        "family_state_bindings": bindings,
        "required_subproducers": ["bow", "htr"],
        "heldout_columns_read": ["_oci_row_id", capture["text_column"]],
        "heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "executable_checkpoint_retained": False,
        "joblib_pickle_or_torch_checkpoint_loaded": False,
    }
    fit_metadata = {
        **fit_metadata_body,
        "content_sha256": _sha256_json(fit_metadata_body),
    }
    _write_immutable_json(metadata_path, fit_metadata)
    semantics = (
        "non-executable JSON/NPZ paired native TF-IDF offset/Ridge and exact "
        "HierarchicalTransformerExtractor uplift state with deterministic match, "
        "fold, validation, aggregate, and ID/text-only heldout replay"
    )
    execution_record = native_family_execution_record(
        family=MATCHED_PAIR_UPLIFT,
        fit_semantics=EXACT_INNER_REFIT,
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        split_scope_fingerprint=split_scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        evidence_payload=evidence_payload,
        configuration=configuration,
        native_fit_metadata_path=metadata_path,
        model_artifact_path=model_path,
        source_artifact_path=source_path,
        model_artifact_semantics=semantics,
    )
    _write_immutable_json(execution_path, execution_record)
    proof = bind_native_family_fit_proof(
        family=MATCHED_PAIR_UPLIFT,
        fit_semantics=EXACT_INNER_REFIT,
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        split_scope_fingerprint=split_scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        evidence_payload=evidence_payload,
        configuration=configuration,
        native_fit_metadata_path=metadata_path,
        native_execution_record_path=execution_path,
        model_artifact_path=model_path,
        source_artifact_path=source_path,
        model_artifact_semantics=semantics,
    )
    proof.verify_artifact_bytes()
    family_row = {
        "family": MATCHED_PAIR_UPLIFT,
        "evidence_item_count": int(evidence_item_count),
        "proof": proof.as_dict(),
        "evidence_payload": _component_file_registration(
            payload_path,
            component_root=root,
        ),
        "native_execution_record": _component_file_registration(
            execution_path,
            component_root=root,
        ),
        "native_fit_metadata": _component_file_registration(
            metadata_path,
            component_root=root,
        ),
        "model_artifact": copy.deepcopy(model_registration),
        "source_artifact": _component_file_registration(
            source_path,
            component_root=root,
        ),
    }
    registration_body = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
        "scope_id": str(scope_id),
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "fit_row_ids": list(fit_rows),
        "heldout_row_ids": list(heldout_rows),
        "split_scope_fingerprint": str(split_scope_fingerprint),
        "data_projection_sha256": str(data_projection_sha256),
        "fit_semantics": EXACT_INNER_REFIT,
        "registered_families": list(PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS),
        "heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "family_proofs": [family_row],
    }
    registration = {
        **registration_body,
        "content_sha256": _sha256_json(registration_body),
    }
    registration_path = proof_root / "registration.json"
    _write_immutable_json(registration_path, registration)
    return {
        **registration,
        "registration": _component_file_registration(
            registration_path,
            component_root=root,
        ),
    }


def _validate_matched_pair_native_family_proof_index(
    *,
    component_root: Path,
    index_registration: Mapping[str, Any],
    expected_inner_scopes: Mapping[str, Mapping[str, Any]],
    split_registry_content_sha256: str,
    modeling_data: pd.DataFrame,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    htr_model_path: Path | None,
    htr_model_sha256: str | None,
    device: torch.device | str,
    reloaded_native_by_scope: dict[str, dict[str, Mapping[str, Any]]] | None = None,
) -> Mapping[str, Any]:
    root = Path(component_root).resolve(strict=True)
    index_path = _validate_component_native_registration(root, index_registration)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in index.items() if key != "content_sha256"}
    scopes = index.get("scopes")
    registered = list(PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS)
    if (
        index.get("schema_version") != STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA
        or index.get("split_registry_content_sha256") != split_registry_content_sha256
        or index.get("registered_families") != registered
        or index.get("executable_checkpoint_files_retained") is not False
        or index.get("content_sha256") != _sha256_json(body)
        or not isinstance(scopes, list)
        or int(index.get("exact_inner_scope_count", -1)) != len(scopes)
    ):
        raise ValueError("matched-pair native proof index has an invalid envelope")
    indexed = {str(row.get("scope_id")): row for row in scopes if isinstance(row, Mapping)}
    if len(indexed) != len(scopes) or set(indexed) != set(expected_inner_scopes):
        raise ValueError("matched-pair native proof index has incomplete exact-inner coverage")
    semantics = (
        "non-executable JSON/NPZ paired native TF-IDF offset/Ridge and exact "
        "HierarchicalTransformerExtractor uplift state with deterministic match, "
        "fold, validation, aggregate, and ID/text-only heldout replay"
    )
    for scope_id, expected in expected_inner_scopes.items():
        row = indexed[scope_id]
        registration_path = _validate_component_native_registration(
            root,
            row.get("registration") or {},
        )
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        registration_body = {
            key: value for key, value in registration.items() if key != "content_sha256"
        }
        family_rows = registration.get("family_proofs")
        if (
            int(row.get("outer_fold", 0)) != int(expected["outer_fold"])
            or int(row.get("inner_fold", 0)) != int(expected["inner_fold"])
            or row.get("registered_families") != registered
            or registration.get("scope_id") != scope_id
            or int(registration.get("outer_fold", 0)) != int(expected["outer_fold"])
            or int(registration.get("inner_fold", 0)) != int(expected["inner_fold"])
            or registration.get("registered_families") != registered
            or registration.get("content_sha256") != _sha256_json(registration_body)
            or registration.get("content_sha256") != row.get("content_sha256")
            or tuple(map(int, registration.get("fit_row_ids") or ()))
            != tuple(map(int, expected["fit_row_ids"]))
            or tuple(map(int, registration.get("heldout_row_ids") or ()))
            != tuple(map(int, expected["heldout_row_ids"]))
            or registration.get("heldout_labels_accessed") is not False
            or not isinstance(family_rows, list)
            or len(family_rows) != 1
            or family_rows[0].get("family") != MATCHED_PAIR_UPLIFT
        ):
            raise ValueError(f"matched-pair native registration is invalid: {scope_id}")
        family_row = family_rows[0]
        paths = {
            key: _validate_component_native_registration(
                root,
                family_row.get(key) or {},
            )
            for key in (
                "evidence_payload",
                "native_execution_record",
                "native_fit_metadata",
                "model_artifact",
                "source_artifact",
            )
        }
        fit_rows = tuple(map(int, expected["fit_row_ids"]))
        heldout_rows = tuple(map(int, expected["heldout_row_ids"]))
        fit_texts = tuple(_normalize_texts(modeling_data.iloc[list(fit_rows)][text_column]))
        heldout_texts = tuple(_normalize_texts(modeling_data.iloc[list(heldout_rows)][text_column]))
        fit_treatment = tuple(
            modeling_data.iloc[list(fit_rows)][treatment_column].to_numpy(dtype=float)
        )
        fit_outcome = tuple(
            modeling_data.iloc[list(fit_rows)][outcome_column].to_numpy(dtype=float)
        )
        capture = validate_matched_pair_native_capture(
            paths["model_artifact"],
            expected_scope_id=scope_id,
            expected_fit_row_ids=fit_rows,
            expected_heldout_row_ids=heldout_rows,
            fit_texts=fit_texts,
            heldout_texts=heldout_texts,
            expected_fit_treatment=fit_treatment,
            expected_fit_outcome=fit_outcome,
            htr_model_path=htr_model_path,
            expected_htr_model_tree_sha256=htr_model_sha256,
            device=device,
        )
        source = json.loads(paths["source_artifact"].read_text(encoding="utf-8"))
        source_proofs = _require_matched_pair_source_proofs(source, scope_id=scope_id)
        metadata = json.loads(paths["native_fit_metadata"].read_text(encoding="utf-8"))
        metadata_body = {key: value for key, value in metadata.items() if key != "content_sha256"}
        if (
            metadata.get("schema_version") != STAGE1_MATCHED_PAIR_NATIVE_FIT_METADATA_SCHEMA
            or metadata.get("family") != MATCHED_PAIR_UPLIFT
            or metadata.get("capture_content_sha256") != capture["content_sha256"]
            or metadata.get("source_subproducer_proofs_sha256") != source_proofs["content_sha256"]
            or metadata.get("required_subproducers") != ["bow", "htr"]
            or metadata.get("content_sha256") != _sha256_json(metadata_body)
            or metadata.get("heldout_labels_accessed") is not False
            or metadata.get("family_state_bindings")
            != _matched_pair_capture_family_bindings(capture)
        ):
            raise ValueError(f"matched-pair native fit metadata is invalid: {scope_id}")
        evidence_payload = json.loads(paths["evidence_payload"].read_text(encoding="utf-8"))
        rebound = bind_native_family_fit_proof(
            family=MATCHED_PAIR_UPLIFT,
            fit_semantics=EXACT_INNER_REFIT,
            outer_fold=int(expected["outer_fold"]),
            inner_fold=int(expected["inner_fold"]),
            split_scope_fingerprint=str(registration["split_scope_fingerprint"]),
            data_projection_sha256=str(registration["data_projection_sha256"]),
            fit_row_ids=fit_rows,
            heldout_row_ids=heldout_rows,
            evidence_payload=evidence_payload,
            configuration=metadata["configuration"],
            native_fit_metadata_path=paths["native_fit_metadata"],
            native_execution_record_path=paths["native_execution_record"],
            model_artifact_path=paths["model_artifact"],
            source_artifact_path=paths["source_artifact"],
            model_artifact_semantics=semantics,
        )
        if rebound.as_dict() != family_row.get("proof"):
            raise RuntimeError(f"matched-pair native proof identity changed: {scope_id}")
        _record_reloaded_exact_inner_family(
            reloaded_native_by_scope,
            scope_id=scope_id,
            family=MATCHED_PAIR_UPLIFT,
            proof=rebound,
            evidence_payload=evidence_payload,
            artifact_paths=paths,
        )
    return copy.deepcopy(index)


def _register_tfidf_native_family_proofs(
    *,
    component_root: Path,
    proof_directory: Path,
    scope_id: str,
    catalog: RoleNeutralEvidenceCatalog,
    tfidf_discovery: Mapping[str, Any],
    outer_fold: int,
    inner_fold: int,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    fit_treatment: Sequence[float],
    fit_outcome: Sequence[float],
    split_scope_fingerprint: str,
    data_projection_sha256: str,
    configuration: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind the real nested TF-IDF context to its two native family adapters.

    This accepts only the fitted context, score-selection JSON, and exact
    context metadata already emitted by ``run_tfidf_topic_stage1``.  The
    adapter payload is projected from the architecture catalog and persisted;
    no synthetic model file, placeholder digest, or output-only proof is
    manufactured here.
    """

    if int(outer_fold) < 1 or int(inner_fold) < 1:
        raise ValueError("TF-IDF native proof registration requires an exact-inner scope")
    if not isinstance(tfidf_discovery, Mapping):
        raise TypeError("TF-IDF discovery metadata must be a mapping")
    if str(tfidf_discovery.get("scope_id") or "") != str(scope_id):
        raise ValueError("TF-IDF discovery metadata is bound to another scope")
    fit_rows = tuple(map(int, fit_row_ids))
    heldout_rows = tuple(map(int, heldout_row_ids))
    if (
        tuple(map(int, tfidf_discovery.get("fit_row_ids") or ())) != fit_rows
        or tuple(map(int, tfidf_discovery.get("heldout_row_ids") or ())) != heldout_rows
    ):
        raise ValueError("TF-IDF discovery metadata changed its exact row scope")
    canonical_treatment = np.asarray(fit_treatment, dtype=float)
    canonical_outcome = np.asarray(fit_outcome, dtype=float)
    if (
        canonical_treatment.shape != (len(fit_rows),)
        or canonical_outcome.shape != (len(fit_rows),)
        or not np.isfinite(canonical_treatment).all()
        or not np.isfinite(canonical_outcome).all()
        or tfidf_discovery.get("registered_fit_treatment_sha256")
        != _float_hex_sha256(canonical_treatment)
        or tfidf_discovery.get("registered_fit_outcome_sha256")
        != _float_hex_sha256(canonical_outcome)
    ):
        raise ValueError("TF-IDF fitted context differs from canonical fit labels")
    nesting = tfidf_discovery.get("selection_nesting")
    if not isinstance(nesting, Mapping):
        raise ValueError("TF-IDF fitted context lacks nested-calibration lineage")
    position_by_row = {row_id: position for position, row_id in enumerate(fit_rows)}
    model_rows = tuple(map(int, nesting.get("model_fit_row_ids") or ()))
    calibration_rows = tuple(map(int, nesting.get("calibration_row_ids") or ()))
    if (
        set(model_rows) & set(calibration_rows)
        or set(model_rows) | set(calibration_rows) != set(fit_rows)
        or tfidf_discovery.get("nested_model_fit_treatment_sha256")
        != _float_hex_sha256(
            canonical_treatment[[position_by_row[row_id] for row_id in model_rows]]
        )
        or tfidf_discovery.get("nested_model_fit_outcome_sha256")
        != _float_hex_sha256(canonical_outcome[[position_by_row[row_id] for row_id in model_rows]])
        or tfidf_discovery.get("nested_calibration_treatment_sha256")
        != _float_hex_sha256(
            canonical_treatment[[position_by_row[row_id] for row_id in calibration_rows]]
        )
        or tfidf_discovery.get("nested_calibration_outcome_sha256")
        != _float_hex_sha256(
            canonical_outcome[[position_by_row[row_id] for row_id in calibration_rows]]
        )
    ):
        raise ValueError("TF-IDF nested partitions differ from canonical fit labels")
    artifacts = tfidf_discovery.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("TF-IDF discovery metadata has no native artifacts")
    root = Path(component_root).resolve(strict=True)
    model_path = _component_regular_file(
        root,
        str(artifacts.get("fitted_context") or ""),
        field_name="TF-IDF fitted context",
    )
    source_path = _component_regular_file(
        root,
        str(artifacts.get("topic_score_tests") or ""),
        field_name="TF-IDF score-selection artifact",
    )
    metadata_path = _component_regular_file(
        root,
        model_path.parent / "context_metadata.json",
        field_name="TF-IDF context metadata",
    )
    try:
        persisted_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("TF-IDF context metadata is not valid JSON") from exc
    if persisted_metadata != copy.deepcopy(dict(tfidf_discovery)):
        raise ValueError("TF-IDF handoff metadata differs from its native context artifact")
    if configuration.get("scope_id") != scope_id:
        raise ValueError("TF-IDF native configuration changed its scope identity")
    if configuration.get("score_selection_label_policy") != "nested_fit_calibration":
        raise ValueError("TF-IDF native registration requires nested fit-only calibration")

    proof_root = Path(proof_directory)
    if not proof_root.is_absolute():
        proof_root = root / proof_root
    if proof_root.exists() and proof_root.is_symlink():
        raise ValueError("TF-IDF proof directory cannot be a symlink")
    proof_root.mkdir(parents=True, exist_ok=True)
    try:
        proof_root.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError("TF-IDF proof directory escapes its component") from exc

    registrations: list[dict[str, Any]] = []
    for family in PRODUCTION_TFIDF_REGISTERED_NATIVE_FAMILY_ADAPTERS:
        evidence_payload, evidence_item_count = family_payload_from_catalog(
            catalog,
            family=family,
        )
        payload_path = proof_root / f"{family}.evidence_payload.json"
        _write_immutable_json(payload_path, evidence_payload)
        semantics = (
            "native nested-calibration fitted TF-IDF topic context"
            if family == TFIDF_TOPICS
            else (
                "native nested-calibration fitted TF-IDF context with fit-side "
                "orphan-term exclusion"
            )
        )
        execution_record = native_family_execution_record(
            family=family,
            fit_semantics=EXACT_INNER_REFIT,
            outer_fold=int(outer_fold),
            inner_fold=int(inner_fold),
            split_scope_fingerprint=split_scope_fingerprint,
            data_projection_sha256=data_projection_sha256,
            fit_row_ids=fit_rows,
            heldout_row_ids=heldout_rows,
            evidence_payload=evidence_payload,
            configuration=configuration,
            native_fit_metadata_path=metadata_path,
            model_artifact_path=model_path,
            source_artifact_path=source_path,
            model_artifact_semantics=semantics,
        )
        execution_path = proof_root / f"{family}.execution.json"
        _write_immutable_json(execution_path, execution_record)
        proof: NativeFamilyFitProof = bind_native_family_fit_proof(
            family=family,
            fit_semantics=EXACT_INNER_REFIT,
            outer_fold=int(outer_fold),
            inner_fold=int(inner_fold),
            split_scope_fingerprint=split_scope_fingerprint,
            data_projection_sha256=data_projection_sha256,
            fit_row_ids=fit_rows,
            heldout_row_ids=heldout_rows,
            evidence_payload=evidence_payload,
            configuration=configuration,
            native_fit_metadata_path=metadata_path,
            native_execution_record_path=execution_path,
            model_artifact_path=model_path,
            source_artifact_path=source_path,
            model_artifact_semantics=semantics,
        )
        proof.verify_artifact_bytes()
        registrations.append(
            {
                "family": family,
                "evidence_item_count": int(evidence_item_count),
                "proof": proof.as_dict(),
                "evidence_payload": _component_file_registration(
                    payload_path,
                    component_root=root,
                ),
                "native_execution_record": _component_file_registration(
                    execution_path,
                    component_root=root,
                ),
                "native_fit_metadata": _component_file_registration(
                    metadata_path,
                    component_root=root,
                ),
                "model_artifact": _component_file_registration(
                    model_path,
                    component_root=root,
                ),
                "source_artifact": _component_file_registration(
                    source_path,
                    component_root=root,
                ),
            }
        )
    body = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
        "scope_id": scope_id,
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "fit_row_ids": list(fit_rows),
        "heldout_row_ids": list(heldout_rows),
        "split_scope_fingerprint": split_scope_fingerprint,
        "data_projection_sha256": data_projection_sha256,
        "fit_semantics": EXACT_INNER_REFIT,
        "registered_families": list(PRODUCTION_TFIDF_REGISTERED_NATIVE_FAMILY_ADAPTERS),
        "heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "secrets_accessed": False,
        "family_proofs": registrations,
    }
    payload = {**body, "content_sha256": _sha256_json(body)}
    registration_path = proof_root / "registration.json"
    _write_immutable_json(registration_path, payload)
    return {
        **payload,
        "registration": _component_file_registration(
            registration_path,
            component_root=root,
        ),
    }


def _validate_tfidf_native_family_proof_index(
    *,
    component_root: Path,
    index_registration: Mapping[str, Any],
    expected_inner_scopes: Mapping[str, Mapping[str, Any]],
    expected_configuration_by_scope: Mapping[str, Mapping[str, Any]],
    split_registry_content_sha256: str,
    modeling_data: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    reloaded_native_by_scope: dict[str, dict[str, Mapping[str, Any]]] | None = None,
) -> Mapping[str, Any]:
    """Reload the two exact TF-IDF families from their nested-fit artifacts."""

    root = Path(component_root).resolve(strict=True)
    index_path = _validate_component_native_registration(root, index_registration)
    index = _read_json_object_reject_duplicates(
        index_path,
        field_name="TF-IDF native family proof index",
    )
    index_body = {key: value for key, value in index.items() if key != "content_sha256"}
    scopes = index.get("scopes")
    families = list(PRODUCTION_TFIDF_REGISTERED_NATIVE_FAMILY_ADAPTERS)
    if (
        index.get("schema_version") != STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA
        or index.get("split_registry_content_sha256") != split_registry_content_sha256
        or index.get("registered_families") != families
        or index.get("executable_checkpoint_files_retained") is not False
        or index.get("content_sha256") != _sha256_json(index_body)
        or not isinstance(scopes, list)
        or int(index.get("exact_inner_scope_count", -1)) != len(scopes)
    ):
        raise ValueError("TF-IDF native proof index has an invalid closed envelope")
    indexed = {str(row.get("scope_id")): row for row in scopes if isinstance(row, Mapping)}
    if (
        len(indexed) != len(scopes)
        or set(indexed) != set(expected_inner_scopes)
        or set(expected_configuration_by_scope) != set(expected_inner_scopes)
    ):
        raise ValueError("TF-IDF native proof index has incomplete exact-inner coverage")
    semantics = {
        TFIDF_TOPICS: "native nested-calibration fitted TF-IDF topic context",
        TFIDF_ORPHAN_NGRAMS: (
            "native nested-calibration fitted TF-IDF context with fit-side " "orphan-term exclusion"
        ),
    }
    for scope_id, expected in expected_inner_scopes.items():
        row = indexed[scope_id]
        registration_path = _validate_component_native_registration(
            root,
            row.get("registration") or {},
        )
        registration = _read_json_object_reject_duplicates(
            registration_path,
            field_name=f"TF-IDF native family proof registration {scope_id}",
        )
        registration_body = {
            key: value for key, value in registration.items() if key != "content_sha256"
        }
        family_rows = registration.get("family_proofs")
        fit_rows = tuple(map(int, expected["fit_row_ids"]))
        heldout_rows = tuple(map(int, expected["heldout_row_ids"]))
        if (
            int(row.get("outer_fold", 0)) != int(expected["outer_fold"])
            or int(row.get("inner_fold", 0)) != int(expected["inner_fold"])
            or row.get("registered_families") != families
            or registration.get("schema_version") != STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA
            or registration.get("scope_id") != scope_id
            or int(registration.get("outer_fold", 0)) != int(expected["outer_fold"])
            or int(registration.get("inner_fold", 0)) != int(expected["inner_fold"])
            or registration.get("registered_families") != families
            or registration.get("fit_semantics") != EXACT_INNER_REFIT
            or registration.get("heldout_labels_accessed") is not False
            or registration.get("oracle_fields_accessed") is not False
            or registration.get("secrets_accessed") is not False
            or tuple(map(int, registration.get("fit_row_ids") or ())) != fit_rows
            or tuple(map(int, registration.get("heldout_row_ids") or ())) != heldout_rows
            or registration.get("content_sha256") != _sha256_json(registration_body)
            or registration.get("content_sha256") != row.get("content_sha256")
            or not isinstance(family_rows, list)
            or [item.get("family") for item in family_rows if isinstance(item, Mapping)] != families
        ):
            raise ValueError(f"TF-IDF native proof registration is invalid: {scope_id}")
        fit_treatment = modeling_data.iloc[list(fit_rows)][treatment_column].to_numpy(dtype=float)
        fit_outcome = modeling_data.iloc[list(fit_rows)][outcome_column].to_numpy(dtype=float)
        expected_configuration = expected_configuration_by_scope[scope_id]
        for family_row in family_rows:
            family = str(family_row["family"])
            paths = {
                key: _validate_component_native_registration(
                    root,
                    family_row.get(key) or {},
                )
                for key in (
                    "evidence_payload",
                    "native_execution_record",
                    "native_fit_metadata",
                    "model_artifact",
                    "source_artifact",
                )
            }
            metadata = _read_json_object_reject_duplicates(
                paths["native_fit_metadata"],
                field_name=f"TF-IDF native metadata {scope_id}/{family}",
            )
            if (
                metadata.get("scope_id") != scope_id
                or metadata.get("registered_fit_treatment_sha256")
                != _float_hex_sha256(fit_treatment)
                or metadata.get("registered_fit_outcome_sha256") != _float_hex_sha256(fit_outcome)
                or expected_configuration.get("scope_id") != scope_id
                or expected_configuration.get("score_selection_label_policy")
                != "nested_fit_calibration"
            ):
                raise ValueError(f"TF-IDF metadata differs from canonical fit labels: {scope_id}")
            evidence_payload = _read_json_object_reject_duplicates(
                paths["evidence_payload"],
                field_name=f"TF-IDF native payload {scope_id}/{family}",
            )
            if int(family_row.get("evidence_item_count", 0)) != len(
                evidence_payload.get("architecture_evidence") or ()
            ):
                raise ValueError(f"TF-IDF native evidence count changed: {scope_id}/{family}")
            rebound = bind_native_family_fit_proof(
                family=family,
                fit_semantics=EXACT_INNER_REFIT,
                outer_fold=int(expected["outer_fold"]),
                inner_fold=int(expected["inner_fold"]),
                split_scope_fingerprint=str(registration["split_scope_fingerprint"]),
                data_projection_sha256=str(registration["data_projection_sha256"]),
                fit_row_ids=fit_rows,
                heldout_row_ids=heldout_rows,
                evidence_payload=evidence_payload,
                configuration=expected_configuration,
                native_fit_metadata_path=paths["native_fit_metadata"],
                native_execution_record_path=paths["native_execution_record"],
                model_artifact_path=paths["model_artifact"],
                source_artifact_path=paths["source_artifact"],
                model_artifact_semantics=semantics[family],
            )
            if rebound.as_dict() != family_row.get("proof"):
                raise RuntimeError(f"TF-IDF native proof identity changed: {scope_id}/{family}")
            _record_reloaded_exact_inner_family(
                reloaded_native_by_scope,
                scope_id=scope_id,
                family=family,
                proof=rebound,
                evidence_payload=evidence_payload,
                artifact_paths=paths,
            )
    return copy.deepcopy(index)


def _load_serialized_mapping(path: Path) -> Mapping[str, Any]:
    payload = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is present in the lock
            raise RuntimeError("PyYAML is required to read a YAML Stage 1 config") from exc
        value = yaml.safe_load(payload)
    else:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ValueError("Stage 1 config is not valid JSON") from exc
            value = yaml.safe_load(payload)
    if not isinstance(value, Mapping):
        raise ValueError("Stage 1 config must contain one object")
    return value


def load_applied_stage1_config(path: Path | str) -> AppliedInferenceConfig:
    """Load a raw, experiment, or historical Stage 1 config without its secrets."""

    requested = Path(path).resolve(strict=True)
    payload = copy.deepcopy(dict(_load_serialized_mapping(requested)))
    if isinstance(payload.get("config"), Mapping):
        applied = copy.deepcopy(dict(payload["config"]))
    elif isinstance(payload.get("applied_inference"), Mapping):
        applied = copy.deepcopy(dict(payload["applied_inference"]))
    else:
        applied = payload
    architecture = applied.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("Stage 1 config has no architecture object")
    architecture = copy.deepcopy(dict(architecture))
    # Historical snapshots duplicate the same builder config under this legacy
    # key.  Keeping both makes the generic dataclass parser instantiate a second,
    # potentially divergent object.
    architecture.pop("multi_model_agentic_forest", None)
    applied["architecture"] = architecture
    return ExperimentConfig.from_dict(
        {"applied_inference": _sanitize_secrets(applied)}
    ).applied_inference


def _source_identity() -> Mapping[str, Any]:
    """Authenticate the complete local Python behavior and locked environment.

    A hand-maintained list of a few directly imported modules is not a closed
    behavior dependency.  The production request therefore binds every Python
    source file in ``oci``, the entry point, both packaging/lock files, the full
    installed distribution version set, and the Python/native ML runtime.
    """

    repository_root = Path(__file__).resolve().parents[2]
    candidates = sorted((repository_root / "oci").rglob("*.py"))
    for relative in (
        Path("scripts/build_all_evidence_stage1_bundle.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ):
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required behavior dependency is absent: {path}")
        candidates.append(path)
    source_files = []
    for path in sorted(set(candidates)):
        digest, stat_identity = _read_stable_sha256(path)
        source_files.append(
            {
                "relative_path": path.relative_to(repository_root).as_posix(),
                "size": int(stat_identity[2]),
                "sha256": digest,
            }
        )

    distributions: set[tuple[str, str]] = set()
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").strip().casefold()
        version = str(distribution.version or "").strip()
        if name and version:
            distributions.add((name, version))
    package_versions = [
        {"name": name, "version": version} for name, version in sorted(distributions)
    ]
    runtime = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": str(sys.implementation.cache_tag),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "torch_cudnn_version": (
            None if not torch.backends.cudnn.is_available() else int(torch.backends.cudnn.version())
        ),
    }
    body = {
        "schema_version": STAGE1_BEHAVIOR_IDENTITY_SCHEMA,
        "source_files": source_files,
        "source_file_count": len(source_files),
        "source_tree_sha256": _sha256_json(source_files),
        "installed_distributions": package_versions,
        "installed_distributions_sha256": _sha256_json(package_versions),
        "runtime": runtime,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _validate_binary(values: pd.Series, *, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or set(np.unique(numeric).tolist()) != {0.0, 1.0}:
        raise ValueError(f"{name} must contain finite binary 0/1 values with both arms")
    return numeric


def _load_local_htr_tokenizer(model_path: Path) -> Any:
    """Load the same local-only tokenizer family used by the HTR runtime."""

    try:
        from transformers import AutoTokenizer, BertTokenizer
    except ImportError as exc:  # pragma: no cover - production dependency
        raise ImportError("transformers is required for the HTR no-truncation audit") from exc

    config_path = model_path / "config.json"
    config: Mapping[str, Any] = {}
    if config_path.is_file():
        try:
            decoded = json.loads(config_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("local HTR model config is invalid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ValueError("local HTR model config must contain one JSON object")
        config = decoded
    legacy_bert = (model_path / "vocab.txt").is_file() and config.get("model_type") in {
        None,
        "bert",
    }
    if legacy_bert:
        return BertTokenizer.from_pretrained(str(model_path), local_files_only=True)

    failures: list[Exception] = []
    for use_fast in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                str(model_path),
                use_fast=use_fast,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:  # pragma: no cover - depends on local model family
            failures.append(exc)
    if (model_path / "vocab.txt").is_file():
        try:
            return BertTokenizer.from_pretrained(str(model_path), local_files_only=True)
        except Exception as exc:  # pragma: no cover - malformed local model
            failures.append(exc)
    raise ValueError(
        "local HTR tokenizer could not be loaded without network access"
    ) from failures[-1]


def _htr_model_sequence_limit(model_path: Path, tokenizer: Any) -> int | None:
    candidates: list[int] = []
    config_path = model_path / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("local HTR model config is invalid JSON") from exc
        if not isinstance(config, Mapping):
            raise ValueError("local HTR model config must contain one JSON object")
        for field_name in (
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
            "model_max_length",
        ):
            value = config.get(field_name)
            if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 1_000_000_000:
                candidates.append(int(value))
    tokenizer_maximum = getattr(tokenizer, "model_max_length", None)
    if (
        isinstance(tokenizer_maximum, int)
        and not isinstance(tokenizer_maximum, bool)
        and 0 < tokenizer_maximum < 1_000_000_000
    ):
        candidates.append(int(tokenizer_maximum))
    return min(candidates) if candidates else None


def _htr_token_lengths_without_truncation(
    tokenizer: Any,
    chunks: Sequence[str],
) -> tuple[int, ...]:
    try:
        encoded = tokenizer(
            list(chunks),
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_length=True,
        )
    except Exception as exc:
        raise ValueError("local HTR tokenizer could not perform a no-truncation audit") from exc
    if not isinstance(encoded, Mapping):
        raise ValueError("local HTR tokenizer returned an invalid audit payload")
    raw_lengths = encoded.get("length")
    if raw_lengths is not None:
        if hasattr(raw_lengths, "tolist"):
            raw_lengths = raw_lengths.tolist()
        if isinstance(raw_lengths, int) and not isinstance(raw_lengths, bool):
            raw_lengths = [raw_lengths]
        try:
            lengths = tuple(raw_lengths)
        except TypeError as exc:
            raise ValueError("local HTR tokenizer returned invalid token lengths") from exc
    else:
        raw_input_ids = encoded.get("input_ids")
        if hasattr(raw_input_ids, "tolist"):
            raw_input_ids = raw_input_ids.tolist()
        if not isinstance(raw_input_ids, Sequence) or isinstance(raw_input_ids, (str, bytes)):
            raise ValueError("local HTR tokenizer omitted auditable token lengths")
        lengths = tuple(
            (
                len(value)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
                else -1
            )
            for value in raw_input_ids
        )
    if len(lengths) != len(chunks) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in lengths
    ):
        raise ValueError("local HTR tokenizer returned invalid token lengths")
    return tuple(int(value) for value in lengths)


def validate_htr_input_nontruncation_audit(
    audit: Mapping[str, Any],
    *,
    config: AppliedInferenceConfig | Mapping[str, Any],
    expected_rows: int,
    expected_htr_model_tree_sha256: str,
) -> Mapping[str, Any]:
    if not isinstance(audit, Mapping) or set(audit) != set(_HTR_INPUT_AUDIT_FIELDS):
        raise ValueError("HTR input no-truncation audit is not a closed schema")
    body = copy.deepcopy(dict(audit))
    content_sha256 = str(body.pop("content_sha256", ""))
    if isinstance(config, Mapping):
        architecture = config.get("architecture")
        if not isinstance(architecture, Mapping):
            raise ValueError("HTR input audit config has no architecture mapping")

        def architecture_value(name: str) -> Any:
            return architecture.get(name)

    else:
        architecture = config.architecture

        def architecture_value(name: str) -> Any:
            return getattr(architecture, name)

    configured_maximum = int(architecture_value("htr_max_chunk_length"))
    model_maximum = audit.get("model_max_sequence_length")
    effective_maximum = audit.get("effective_max_chunk_length")
    valid_model_maximum = model_maximum is None or (
        isinstance(model_maximum, int)
        and not isinstance(model_maximum, bool)
        and model_maximum >= 1
    )
    expected_effective = (
        configured_maximum if model_maximum is None else min(configured_maximum, model_maximum)
    )
    total_chunks = audit.get("total_chunks")
    max_observed = audit.get("max_observed_token_count")
    vocab_size = audit.get("tokenizer_vocab_size")
    if (
        audit.get("schema_version") != STAGE1_HTR_INPUT_NONTRUNCATION_AUDIT_SCHEMA
        or audit.get("row_count") != expected_rows
        or audit.get("chunk_size_words") != int(architecture_value("htr_chunk_size_words"))
        or audit.get("chunk_overlap_words") != int(architecture_value("htr_chunk_overlap_words"))
        or audit.get("max_chunks") != int(architecture_value("htr_max_chunks"))
        or audit.get("configured_max_chunk_length") != configured_maximum
        or not valid_model_maximum
        or effective_maximum != expected_effective
        or not isinstance(total_chunks, int)
        or isinstance(total_chunks, bool)
        or total_chunks < expected_rows
        or audit.get("uncapped_total_chunks") != total_chunks
        or not isinstance(max_observed, int)
        or isinstance(max_observed, bool)
        or not 1 <= max_observed <= effective_maximum
        or audit.get("chunk_cap_nonbinding") is not True
        or audit.get("all_chunks_within_effective_max_length") is not True
        or audit.get("semantic_truncation_allowed") is not False
        or audit.get("tokenizer_truncation_allowed") is not False
        or not isinstance(audit.get("tokenizer_class"), str)
        or not audit["tokenizer_class"]
        or not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size < 1
        or audit.get("htr_model_tree_sha256") != expected_htr_model_tree_sha256
        or audit.get("applies_to_families") != [HTR_NEURAL, MATCHED_PAIR_UPLIFT]
        or _HEX_SHA256.fullmatch(content_sha256) is None
        or _sha256_json(body) != content_sha256
    ):
        raise ValueError("HTR input no-truncation audit changed its authenticated policy")
    for field_name in (
        "normalized_text_projection_sha256",
        "ordered_chunk_counts_sha256",
        "ordered_token_counts_sha256",
        "htr_model_tree_sha256",
    ):
        if _HEX_SHA256.fullmatch(str(audit.get(field_name) or "")) is None:
            raise ValueError(f"HTR input audit {field_name} is not a SHA-256")
    return copy.deepcopy(dict(audit))


def _build_htr_input_nontruncation_audit(
    *,
    texts: Sequence[str],
    config: AppliedInferenceConfig,
    htr_model_path: Path,
    htr_model_tree_sha256: str,
) -> Mapping[str, Any]:
    """Prove full-cohort HTR inputs are word- and tokenizer-lossless."""

    architecture = config.architecture
    chunk_size = int(architecture.htr_chunk_size_words)
    overlap = int(architecture.htr_chunk_overlap_words)
    maximum_chunks = int(architecture.htr_max_chunks)
    configured_token_limit = int(architecture.htr_max_chunk_length)
    if (
        chunk_size < 1
        or overlap < 0
        or overlap >= chunk_size
        or maximum_chunks < 1
        or configured_token_limit < 1
    ):
        raise ValueError("production HTR chunk configuration is invalid")
    normalized_texts = tuple(_normalize_text(value) for value in texts)
    sample_chunks: list[tuple[str, ...]] = []
    chunk_counts: list[int] = []
    cap_offenders: list[tuple[int, int]] = []
    stride = chunk_size - overlap
    for row_index, text in enumerate(normalized_texts):
        word_count = sum(1 for _match in re.finditer(r"\S+", text))
        uncapped_count = max(1, (word_count + stride - 1) // stride)
        if uncapped_count > maximum_chunks:
            cap_offenders.append((row_index, uncapped_count))
            continue
        chunks = tuple(
            split_text_into_word_chunks(
                text,
                chunk_size_words=chunk_size,
                chunk_overlap_words=overlap,
                max_chunks=maximum_chunks,
            )
        )
        if len(chunks) != uncapped_count:
            raise RuntimeError("HTR word chunker differs from its uncapped source projection")
        sample_chunks.append(chunks)
        chunk_counts.append(len(chunks))
    if cap_offenders:
        preview = ", ".join(
            f"row {row_index}: {count} chunks" for row_index, count in cap_offenders[:8]
        )
        suffix = "" if len(cap_offenders) <= 8 else f", plus {len(cap_offenders) - 8} more rows"
        raise ValueError(
            "production HTR max_chunks would cause semantic truncation; "
            f"configured max_chunks={maximum_chunks}, offending rows: {preview}{suffix}. "
            "Raise htr_max_chunks so the cap is nonbinding."
        )
    flat_chunks = tuple(chunk for chunks in sample_chunks for chunk in chunks)
    row_chunk_coordinates = tuple(
        (row_index, chunk_index)
        for row_index, chunks in enumerate(sample_chunks)
        for chunk_index, _chunk in enumerate(chunks)
    )
    tokenizer = _load_local_htr_tokenizer(htr_model_path)
    model_token_limit = _htr_model_sequence_limit(htr_model_path, tokenizer)
    effective_token_limit = (
        configured_token_limit
        if model_token_limit is None
        else min(configured_token_limit, model_token_limit)
    )
    token_counts: list[int] = []
    for start in range(0, len(flat_chunks), _HTR_TOKENIZATION_AUDIT_BATCH_SIZE):
        stop = min(start + _HTR_TOKENIZATION_AUDIT_BATCH_SIZE, len(flat_chunks))
        token_counts.extend(
            _htr_token_lengths_without_truncation(tokenizer, flat_chunks[start:stop])
        )
    token_offenders = [
        (flat_index, row_chunk_coordinates[flat_index], count)
        for flat_index, count in enumerate(token_counts)
        if count > effective_token_limit
    ]
    if token_offenders:
        preview = ", ".join(
            f"flat chunk {flat_index} (row {coordinate[0]}, row chunk {coordinate[1]}): "
            f"{count} tokens"
            for flat_index, coordinate, count in token_offenders[:8]
        )
        suffix = (
            "" if len(token_offenders) <= 8 else f", plus {len(token_offenders) - 8} more chunks"
        )
        raise ValueError(
            "production HTR tokenizer would cause semantic truncation; "
            f"effective_max_chunk_length={effective_token_limit}, offending chunks: "
            f"{preview}{suffix}. Reduce htr_chunk_size_words or repair the source cohort."
        )
    raw_vocab_size = getattr(tokenizer, "vocab_size", None)
    if (
        not isinstance(raw_vocab_size, int)
        or isinstance(raw_vocab_size, bool)
        or raw_vocab_size < 1
    ):
        try:
            raw_vocab_size = len(tokenizer)
        except Exception as exc:
            raise ValueError("local HTR tokenizer has no auditable vocabulary size") from exc
    body = {
        "schema_version": STAGE1_HTR_INPUT_NONTRUNCATION_AUDIT_SCHEMA,
        "row_count": len(normalized_texts),
        "normalized_text_projection_sha256": _sha256_json(
            {
                "schema_version": "production_htr_normalized_text_projection_v1",
                "texts": normalized_texts,
            }
        ),
        "chunk_size_words": chunk_size,
        "chunk_overlap_words": overlap,
        "max_chunks": maximum_chunks,
        "configured_max_chunk_length": configured_token_limit,
        "model_max_sequence_length": model_token_limit,
        "effective_max_chunk_length": effective_token_limit,
        "total_chunks": len(flat_chunks),
        "uncapped_total_chunks": sum(chunk_counts),
        "ordered_chunk_counts_sha256": _sha256_json(chunk_counts),
        "ordered_token_counts_sha256": _sha256_json(token_counts),
        "max_observed_token_count": max(token_counts),
        "chunk_cap_nonbinding": True,
        "all_chunks_within_effective_max_length": True,
        "semantic_truncation_allowed": False,
        "tokenizer_truncation_allowed": False,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(raw_vocab_size),
        "htr_model_tree_sha256": htr_model_tree_sha256,
        "applies_to_families": [HTR_NEURAL, MATCHED_PAIR_UPLIFT],
    }
    audit = {**body, "content_sha256": _sha256_json(body)}
    return validate_htr_input_nontruncation_audit(
        audit,
        config=config,
        expected_rows=len(normalized_texts),
        expected_htr_model_tree_sha256=htr_model_tree_sha256,
    )


def _validate_effective_config(
    config: AppliedInferenceConfig,
    *,
    dataset_path: Path,
    embedding_cache_dir: Path,
    config_dir: Path,
    seed: int,
) -> tuple[AppliedInferenceConfig, Path]:
    config = copy.deepcopy(config)
    config.dataset_path = str(dataset_path)
    setattr(config, "seed", int(seed))
    if config.outcome_type.strip().lower() != "binary":
        raise ValueError(
            "production all-ten Stage 1 currently requires a binary outcome because "
            "matched-pair uplift is part of the mandatory architecture contract"
        )
    if int(config.cv_folds) < 2:
        raise ValueError("production Stage 1 requires at least two honest outer folds")
    architecture = config.architecture
    if architecture.model_type != "multi_model_forest":
        raise ValueError("architecture.model_type must be 'multi_model_forest'")
    nn_config = architecture.multi_model_forest
    methods = {
        str(value).strip().lower().replace("-", "_")
        for value in (nn_config.feature_discovery_methods or ())
    }
    missing_methods = {"bow", "htr", "embedding_contrast"} - methods
    if missing_methods:
        raise ValueError(
            "Stage 1 config omits mandatory legacy methods: " + ", ".join(sorted(missing_methods))
        )
    if not nn_config.bow_discovery_enabled or not nn_config.bow_views:
        raise ValueError("production Stage 1 requires non-empty BoW nuisance/R views")
    if not nn_config.htr_evidence_enabled:
        raise ValueError("production Stage 1 requires HTR evidence")
    if bool(architecture.htr_freeze_sentence_encoder):
        raise ValueError("the production HTR sentence encoder must remain unfrozen")
    if not (
        nn_config.matched_pair_uplift_enabled
        and nn_config.matched_pair_bow_enabled
        and nn_config.matched_pair_htr_enabled
    ):
        raise ValueError("production Stage 1 requires both BoW and HTR matched-pair uplift")
    embedding = nn_config.embedding_contrast
    if not embedding.enabled or not embedding.include_cluster_contrast_vectors:
        raise ValueError("production Stage 1 requires whole-cohort and clustered embeddings")
    if str(embedding.chunk_selection).strip().lower() != "last":
        raise ValueError(
            "the authenticated production embedding cache requires last-chunk selection"
        )
    if list(embedding.residualize_columns):
        raise ValueError(
            "embedding residualization columns are not allowed in the projected production input"
        )
    if not nn_config.require_honest_outer_split:
        raise ValueError("multi_model_forest.require_honest_outer_split must be true")
    if not nn_config.candidate_consistency_enabled:
        raise ValueError("exact-inner candidate-consistency evidence must be enabled")
    if int(nn_config.candidate_consistency_inner_folds) < 4:
        raise ValueError(
            "hierarchical Stage 1 requires at least four candidate-consistency inner "
            "folds: three initially spent partitions and at least one review gate"
        )
    if str(nn_config.structured_effect_estimator).strip().lower() != "causal_forest":
        raise ValueError("the structured effect estimator must remain causal_forest")
    forest = architecture.explicit_feature_forest
    if not forest.honest or not forest.inference:
        raise ValueError("the configured causal forest must be honest with inference enabled")
    if str(nn_config.agent_context_mode).strip().lower() != "evidence_digest":
        raise ValueError("production legacy handoffs require compact evidence_digest context mode")
    tfidf = nn_config.tfidf_topic
    if not tfidf.score_test_enabled:
        raise ValueError("production TF-IDF Stage 1 requires honest score testing")
    tfidf.score_selection_label_policy = "nested_fit_calibration"
    if not tfidf.orphan_ngram_enabled or int(tfidf.orphan_ngram_min_selected_clusters) < 1:
        raise ValueError("production Stage 1 requires a non-empty orphan-ngram architecture")
    if int(tfidf.score_test_min_topics_per_bank) < 1:
        raise ValueError("production Stage 1 requires non-empty TF-IDF topic banks")

    raw_htr = Path(str(architecture.htr_sentence_model)).expanduser()
    if not raw_htr.is_absolute():
        raw_htr = config_dir / raw_htr
    htr_path = raw_htr.resolve()
    if not htr_path.is_dir():
        raise FileNotFoundError(f"local HTR sentence-model directory does not exist: {htr_path}")
    architecture.htr_sentence_model = str(htr_path)
    architecture.htr_require_live_unfrozen_encoder_attestation = True

    # Scope construction is serial because the authenticated cache provider and
    # private HTR tree are process-local.  Inner model parallelism remains under
    # the existing Stage 1 configuration.
    nn_config.outer_parallelism = "1"
    nn_config.require_honest_outer_split = True
    embedding.cache_dir = str(embedding_cache_dir)
    embedding.external_corpus_cache_dirs = []
    # Novel concept strings cannot be encoded by the frozen row cache.  Raw
    # retrieval witnesses remain available and are converted into the separate
    # TF-IDF semantic-retrieval architecture downstream.
    embedding.include_bow_phrases_as_concepts = False
    embedding.concept_phrases = []
    architecture.multi_model_agentic_forest = nn_config
    return config, htr_path


def _embedding_chunk_configuration(config: AppliedInferenceConfig) -> Mapping[str, Any]:
    embedding = config.architecture.multi_model_forest.embedding_contrast
    return {
        "chunk_size_words": int(embedding.chunk_size_words),
        "chunk_overlap_words": int(embedding.chunk_overlap_words),
        "max_chunks": int(embedding.max_chunks),
        "chunk_selection": str(embedding.chunk_selection),
        "normalize_embeddings": bool(embedding.normalize_embeddings),
        "max_seq_length": (
            None if embedding.max_seq_length is None else int(embedding.max_seq_length)
        ),
    }


def _validate_cache_configuration(
    cache: SpentOnlyFrozenChunkEmbeddingCache,
    config: AppliedInferenceConfig,
) -> None:
    metadata = cache.metadata
    embedding = config.architecture.multi_model_forest.embedding_contrast
    expected = {
        "sentence_model_name": str(embedding.model_name),
        **_embedding_chunk_configuration(config),
    }
    mismatches = {
        key: {"expected": expected_value, "observed": metadata.get(key)}
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    # The selection direction changes the actual chunk population whenever a
    # row exceeds ``max_chunks``.  Missing metadata is therefore not a legacy
    # synonym for the production policy: it is unauthenticated configuration.
    if mismatches:
        raise ValueError(
            "frozen embedding cache does not match effective Stage 1 config: "
            + _canonical_json(mismatches)
        )


def build_canonical_split_registry(
    *,
    data: pd.DataFrame,
    config: AppliedInferenceConfig,
    seed: int,
) -> Mapping[str, Any]:
    """Create the sole authoritative outer/inner fold registry."""

    outer_splits, outer_metadata = make_joint_treatment_outcome_splits(
        data,
        treatment_column=config.treatment_column,
        outcome_column=config.outcome_column,
        outcome_type=config.outcome_type,
        n_splits=int(config.cv_folds),
        seed=int(seed),
    )
    folds: list[dict[str, Any]] = []
    seen_outer: dict[int, int] = {}
    all_ids = set(range(len(data)))
    inner_count = int(config.architecture.multi_model_forest.candidate_consistency_inner_folds)
    outer_heldout_by_fold: dict[int, tuple[int, ...]] = {}
    outer_fit_by_fold: dict[int, tuple[int, ...]] = {}
    for outer_fold, (fit_raw, heldout_raw) in enumerate(outer_splits, start=1):
        fit_ids = tuple(int(value) for value in np.asarray(fit_raw, dtype=int))
        heldout_ids = tuple(int(value) for value in np.asarray(heldout_raw, dtype=int))
        if set(fit_ids) & set(heldout_ids) or set(fit_ids) | set(heldout_ids) != all_ids:
            raise RuntimeError("canonical outer split does not partition the cohort")
        for row_id in heldout_ids:
            seen_outer[row_id] = seen_outer.get(row_id, 0) + 1
        outer_fit_by_fold[outer_fold] = fit_ids
        outer_heldout_by_fold[outer_fold] = heldout_ids
    if set(seen_outer) != all_ids or set(seen_outer.values()) != {1}:
        raise RuntimeError("canonical outer heldouts do not cover the cohort exactly once")

    # The exact-inner contract, rather than a wrapper-local KFold clone, owns
    # every inner partition and its row order.
    exact_registry = CanonicalStage1SplitRegistry.build(
        dataset_row_ids=tuple(range(len(data))),
        outer_heldout_row_ids=outer_heldout_by_fold,
        inner_fold_count=inner_count,
        inner_seed_base=51_000,
    )
    indexed_data = data.reset_index(drop=True).copy()
    indexed_data["_oci_row_id"] = np.arange(len(indexed_data), dtype=int)

    def validate_nested_tfidf_scope(
        fit_ids: Sequence[int],
        *,
        outer_fold: int,
        inner_fold: int | None,
    ) -> None:
        scope_name = (
            f"outer_{outer_fold:03d}_full"
            if inner_fold is None
            else f"outer_{outer_fold:03d}_inner_{inner_fold:03d}"
        )
        registered_fit = indexed_data.iloc[list(map(int, fit_ids))].copy()
        for column in (config.treatment_column, config.outcome_column):
            if set(registered_fit[column].astype(int).unique().tolist()) != {0, 1}:
                raise ValueError(
                    f"{scope_name} lacks both binary classes required by all-ten Stage 1"
                )
        try:
            model_fit, calibration, _plan = _nested_calibration_plan(
                registered_fit,
                config=config,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"{scope_name} is infeasible for production nested TF-IDF calibration: {exc}"
            ) from exc
        for partition_name, partition in (
            ("model-fit", model_fit),
            ("calibration", calibration),
        ):
            for column in (config.treatment_column, config.outcome_column):
                if set(partition[column].astype(int).unique().tolist()) != {0, 1}:
                    raise ValueError(
                        f"{scope_name} is infeasible for production nested TF-IDF "
                        f"calibration: {partition_name} partition lacks both classes in {column}"
                    )

    for exact_outer in exact_registry.outer_splits:
        outer_fold = int(exact_outer.outer_fold)
        fit_ids = outer_fit_by_fold[outer_fold]
        heldout_ids = outer_heldout_by_fold[outer_fold]
        if exact_outer.train_row_ids != fit_ids or exact_outer.heldout_row_ids != heldout_ids:
            raise RuntimeError("exact-inner contract changed the authoritative outer row order")
        validate_nested_tfidf_scope(
            fit_ids,
            outer_fold=outer_fold,
            inner_fold=None,
        )
        inner_metadata = {
            "method": "canonical_kfold",
            "n_splits": inner_count,
            "shuffle": True,
            "random_state": 51_000 + outer_fold,
            "authority": "stage1_exact_inner_evidence.CanonicalStage1SplitRegistry",
        }
        inner_rows: list[dict[str, Any]] = []
        for exact_inner in exact_outer.inner_splits:
            inner_fold = int(exact_inner.inner_fold)
            inner_fit = list(exact_inner.fit_row_ids)
            inner_heldout = list(exact_inner.heldout_row_ids)
            validate_nested_tfidf_scope(
                inner_fit,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
            )
            inner_rows.append(
                {
                    "inner_fold": inner_fold,
                    "fit_row_ids": inner_fit,
                    "heldout_row_ids": inner_heldout,
                    "fit_row_fingerprint": row_set_fingerprint(inner_fit),
                    "heldout_row_fingerprint": row_set_fingerprint(inner_heldout),
                    "split_method": inner_metadata,
                }
            )
        folds.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": list(fit_ids),
                "heldout_row_ids": list(heldout_ids),
                "fit_row_fingerprint": row_set_fingerprint(fit_ids),
                "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
                "inner_folds": inner_rows,
            }
        )
    return {
        "schema_version": TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
        "dataset_row_count": len(data),
        "outer_split_method": outer_metadata,
        "inner_seed_base": 51_000,
        "exact_inner_contract_registry_content_sha256": exact_registry.content_sha256,
        "outer_folds": folds,
    }


def _registry_scopes(registry: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    scopes: list[dict[str, Any]] = []
    for outer in registry["outer_folds"]:
        outer_fold = int(outer["outer_fold"])
        scopes.append(
            {
                "scope_id": f"outer_{outer_fold:03d}_full",
                "outer_fold": outer_fold,
                "scope": "full_outer_train",
                "inner_fold": None,
                "fit_row_ids": list(map(int, outer["fit_row_ids"])),
                "heldout_row_ids": list(map(int, outer["heldout_row_ids"])),
            }
        )
        for inner in outer["inner_folds"]:
            inner_fold = int(inner["inner_fold"])
            scopes.append(
                {
                    "scope_id": f"outer_{outer_fold:03d}_inner_{inner_fold:03d}",
                    "outer_fold": outer_fold,
                    "scope": "candidate_consistency_inner_train",
                    "inner_fold": inner_fold,
                    "fit_row_ids": list(map(int, inner["fit_row_ids"])),
                    "heldout_row_ids": list(map(int, inner["heldout_row_ids"])),
                }
            )
    return tuple(scopes)


def _canonical_exact_registry_from_wrapper(
    registry: Mapping[str, Any],
) -> CanonicalStage1SplitRegistry:
    return CanonicalStage1SplitRegistry.build(
        dataset_row_ids=tuple(range(int(registry["dataset_row_count"]))),
        outer_heldout_row_ids={
            int(fold["outer_fold"]): tuple(map(int, fold["heldout_row_ids"]))
            for fold in registry["outer_folds"]
        },
        inner_fold_count=len(registry["outer_folds"][0]["inner_folds"]),
        inner_seed_base=int(registry.get("inner_seed_base", 51_000)),
    )


def _exact_inner_contract_registry_status(registry: Mapping[str, Any]) -> Mapping[str, Any]:
    """Verify that the wrapper registry is byte-semantically compatible with P0."""

    exact = _canonical_exact_registry_from_wrapper(registry)
    expected = {
        (
            int(outer.outer_fold),
            int(inner.inner_fold),
        ): (list(inner.fit_row_ids), list(inner.heldout_row_ids))
        for outer in exact.outer_splits
        for inner in outer.inner_splits
    }
    observed = {
        (int(outer["outer_fold"]), int(inner["inner_fold"])): (
            list(map(int, inner["fit_row_ids"])),
            list(map(int, inner["heldout_row_ids"])),
        )
        for outer in registry["outer_folds"]
        for inner in outer["inner_folds"]
    }
    if observed != expected:
        raise RuntimeError("canonical wrapper splits differ from the exact-inner P0 contract")
    declared = registry.get("exact_inner_contract_registry_content_sha256")
    if declared != exact.content_sha256:
        raise RuntimeError("wrapper registry changed its exact-inner contract identity")
    return {
        "contract_module_available": True,
        "registry_matches_contract": True,
        "contract_registry_content_sha256": exact.content_sha256,
        "contract_registry": exact.as_dict(),
    }


def _exact_inner_projection_sha256(
    *,
    modeling_data: pd.DataFrame,
    config: AppliedInferenceConfig,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
) -> str:
    fit_rows = tuple(
        Stage1FitRow(
            row_id=int(row_id),
            text=str(modeling_data.iloc[int(row_id)][config.text_column]),
            treatment=float(modeling_data.iloc[int(row_id)][config.treatment_column]),
            outcome=float(modeling_data.iloc[int(row_id)][config.outcome_column]),
        )
        for row_id in fit_row_ids
    )
    heldout_rows = tuple(
        Stage1HeldoutRow(
            row_id=int(row_id),
            text=str(modeling_data.iloc[int(row_id)][config.text_column]),
        )
        for row_id in heldout_row_ids
    )
    return exact_inner_data_projection_sha256(
        fit_rows=fit_rows,
        heldout_rows=heldout_rows,
    )


def exact_inner_family_adapter_gate() -> Mapping[str, Any]:
    """Return the non-bypassable production readiness gate for family adapters.

    The native adapter implementations cover all ten existing architectures.
    Every native component now emits genuine immutable proof registrations for
    all ten families, and the cumulative all-ten root/cache substrate is wired.
    Candidate bundle construction is distinct from final one-shot certification.
    """

    required_families = set(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    if set(FAMILY_NATIVE_BACKEND) != required_families or set(FAMILY_NATIVE_APIS) != (
        required_families
    ):
        raise RuntimeError("native exact-inner adapter implementation has incomplete coverage")
    label_access_policy = {
        family: {
            "fit_text_available": True,
            "fit_treatment_available": True,
            "fit_outcome_available": True,
            "heldout_text_available": True,
            "heldout_treatment_available": False,
            "heldout_outcome_available": False,
            "oracle_fields_available": False,
            "secrets_available": False,
        }
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    registered = frozenset(PRODUCTION_REGISTERED_NATIVE_FAMILY_ADAPTERS)
    unregistered = tuple(
        family for family in ACTIVE_STAGE1_CONCEPT_FAMILIES if family not in registered
    )
    family_adapter_blockers: dict[str, list[str]] = {}
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        if family in registered:
            family_adapter_blockers[family] = []
        else:
            family_adapter_blockers[family] = [
                "native ExactInnerStage1FamilyProducer adapter is not registered by the " "wrapper",
                "component does not yet emit the adapter's scope-bound execution record, "
                "fit metadata, retained native model/source artifacts, and proof index",
            ]
    candidate_bundle_build_ready = not unregistered
    return {
        "schema_version": STAGE1_EXACT_INNER_ADAPTER_GATE_SCHEMA,
        "production_execution_ready": candidate_bundle_build_ready,
        "candidate_bundle_build_ready": candidate_bundle_build_ready,
        "genuine_one_shot_e2e_certified": False,
        "native_exact_inner_registration_complete": not unregistered,
        "registered_component_proof_family_count": len(registered),
        "registered_component_proof_families": list(PRODUCTION_REGISTERED_NATIVE_FAMILY_ADAPTERS),
        "unregistered_component_proof_families": list(unregistered),
        # Backward-compatible field name for pre-existing dry-run consumers.
        "missing_registered_family_producers": list(unregistered),
        "required_contract": "stage1_exact_inner_evidence.ExactInnerStage1FamilyProducer",
        "native_adapter_implementation_by_family": {
            family: {
                "backend": FAMILY_NATIVE_BACKEND[family],
                "fit_apis": list(FAMILY_NATIVE_APIS[family]),
                "implementation_available": True,
                "production_wrapper_registered": family in registered,
            }
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        "required_canaries": [
            "per_family_producer_identity",
            "per_family_fit_audit",
            "per_family_model_artifact_sha256",
            "per_family_code_and_configuration_sha256",
            "persisted_catalog_artifact_sha256",
            "full_outer_payload_clone_rejection",
            "heldout_label_oracle_secret_access_false",
        ],
        "family_label_access_policy": label_access_policy,
        "family_adapter_blockers": family_adapter_blockers,
        "resolved_integration_hardening": [
            "all ten native architecture families have exact-inner adapter identities and "
            "scope/payload/code/config/artifact-bound proof validation",
            "native exact-inner scopes and full-outer clone canaries are derived from validated "
            "persisted catalogs, with artifact bytes rechecked when adapters run",
            "topic and orphan TF-IDF paths bind deterministic nested fit/calibration label "
            "selection; semantic retrieval binds label-free training-only replay partitions "
            "that select nothing and preserve the exhaustive exact-fit projection",
            "TF-IDF topic and orphan-n-gram exact-inner adapters are registered from the "
            "native context metadata, fitted-context model, and score-selection artifacts",
            "neural-query exact-inner adapters are registered from trusted in-memory fitted "
            "query arrays, fit activations, safe evidence, and ID/text-only heldout moments",
            "BoW nuisance and R-loss exact-inner adapters are registered from replayable "
            "non-executable vectorizer/learner arrays, exact fold targets and weights, "
            "full-fit importance learners, and ID/text-only heldout transforms",
            "semantic-retrieval TF-IDF projection is label-free, uncapped, and exhaustive "
            "after its supervised fit-scope embedding directions are frozen",
            "the wrapper avoids prompt-oriented compaction and registers immutable raw-evidence "
            "sidecars for authenticated drill-back",
            "dataset, Stage 1 config, query config, HTR model tree, embedding cache, and source "
            "identities are rechecked before sealing",
            "the request binds the complete in-repository Python behavior surface, lock and "
            "package metadata, installed distribution versions, and runtime identities",
            "partial TF-IDF checkpoint reuse is prohibited; only a complete sealed component "
            "may be reused",
            "the read-only hierarchy loader authenticates the root byte graph, exact-inner "
            "artifacts, raw sidecars, and all-ten coverage",
        ],
        "integration_substrate_blockers": [],
        "certification_blockers": [
            "no genuine arbitrary-cohort one-shot run has yet validated emitted direct and "
            "cumulative proof graphs through hierarchy execution",
        ],
        "residual_hardening_notes": [
            "installed dependencies are version-bound; installed package/native bytes are not "
            "independently content-addressed",
        ],
        "uncontracted_composite_execution_allowed": False,
    }


def _require_exact_inner_family_adapters() -> None:
    gate = exact_inner_family_adapter_gate()
    if gate["candidate_bundle_build_ready"] is not True:
        raise RuntimeError(
            "production Stage 1 candidate bundle construction is fail-closed because its "
            "all-ten integration substrate is incomplete: "
            + "; ".join(
                [
                    *gate["integration_substrate_blockers"],
                    *gate["unregistered_component_proof_families"],
                ]
            )
        )


def _tree_inventory(root: Path, *, exclude_names: Sequence[str] = ()) -> list[dict[str, Any]]:
    excluded = set(exclude_names)
    rows: list[dict[str, Any]] = []
    for candidate in sorted(item for item in root.rglob("*") if item.is_file()):
        if candidate.name in excluded:
            continue
        rows.append(
            {
                "relative_path": candidate.relative_to(root).as_posix(),
                "size": int(candidate.stat().st_size),
                "sha256": _sha256_file(candidate),
            }
        )
    return rows


def _validate_inventory(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    expected_paths = {str(row["relative_path"]) for row in rows}
    observed_paths = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != "component_manifest.json"
    }
    if observed_paths != expected_paths:
        raise RuntimeError(f"authenticated component file set changed: {root}")
    for row in rows:
        path = root / str(row["relative_path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size"])
            or _sha256_file(path) != str(row["sha256"])
        ):
            raise RuntimeError(f"authenticated component file changed: {path}")


def _seal_component(root: Path, *, request_sha256: str, component: str) -> Mapping[str, Any]:
    manifest_path = root / "component_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if set(payload) != {
            "schema_version",
            "request_sha256",
            "component",
            "files",
            "content_sha256",
        }:
            raise RuntimeError(f"component manifest uses an unsupported schema: {manifest_path}")
        if (
            payload.get("schema_version") != STAGE1_COMPONENT_MANIFEST_SCHEMA
            or payload.get("request_sha256") != request_sha256
            or payload.get("component") != component
        ):
            raise RuntimeError(f"component manifest identity mismatch: {manifest_path}")
        body = dict(payload)
        declared_content_sha256 = body.pop("content_sha256", None)
        if (
            not _HEX_SHA256.fullmatch(str(declared_content_sha256 or ""))
            or _sha256_json(body) != declared_content_sha256
        ):
            raise RuntimeError(f"component manifest content hash is invalid: {manifest_path}")
        inventory = payload.get("files")
        if not isinstance(inventory, list):
            raise ValueError("component manifest has no file inventory")
        _validate_inventory(root, inventory)
        return payload
    inventory = _tree_inventory(root, exclude_names=("component_manifest.json",))
    if not inventory:
        raise RuntimeError(f"cannot seal an empty Stage 1 component: {root}")
    body = {
        "schema_version": STAGE1_COMPONENT_MANIFEST_SCHEMA,
        "request_sha256": request_sha256,
        "component": component,
        "files": inventory,
    }
    payload = {**body, "content_sha256": _sha256_json(body)}
    _write_immutable_json(manifest_path, payload)
    return payload


def _load_component_manifest(
    root: Path, *, request_sha256: str, component: str
) -> Mapping[str, Any] | None:
    manifest = root / "component_manifest.json"
    if not manifest.exists():
        return None
    return _seal_component(root, request_sha256=request_sha256, component=component)


@dataclass(frozen=True)
class Stage1BundleBuildOptions:
    dataset_path: Path
    config_path: Path
    embedding_cache_dir: Path | None
    output_dir: Path
    unit_id_column: str
    embedding_local_model_path: Path | None = None
    embedding_cache_output_dir: Path | None = None
    seed: int = 42
    device: str = "cpu"
    gpu_ids: tuple[int, ...] = ()
    num_workers: int = 1
    tfidf_workers: int = 1
    tfidf_parallel_backend: str = "threads"
    query_devices: tuple[str, ...] = ("cpu",)
    query_nuisance_folds: int = 3
    query_config_path: Path | None = None
    resume: bool = False
    dry_run: bool = False


@dataclass
class _PreparedBuild:
    options: Stage1BundleBuildOptions
    output_path: Path
    data: pd.DataFrame
    modeling_data: pd.DataFrame
    config: AppliedInferenceConfig
    htr_model_path: Path
    htr_model_sha256: str
    htr_input_nontruncation_audit: Mapping[str, Any]
    embedding_cache_path: Path
    embedding_cache: SpentOnlyFrozenChunkEmbeddingCache
    embedding_cache_identity: Mapping[str, Any]
    embedding_cache_input_identity: Mapping[str, Any]
    registry: Mapping[str, Any]
    registry_content_sha256: str
    exact_inner_contract_status: Mapping[str, Any]
    query_config: NeuralQueryAgenticForestConfig
    query_config_identity: Mapping[str, Any]
    input_file_identities: Mapping[str, Mapping[str, Any]]
    behavior_identity: Mapping[str, Any]
    hierarchical_discovery_contract_identity: Mapping[str, Any]
    request: Mapping[str, Any]
    request_sha256: str


class ProductionStage1BundleBuilder:
    """Build or authenticate one arbitrary-cohort Stage 1 bundle."""

    def __init__(self, options: Stage1BundleBuildOptions) -> None:
        self.options = options

    def prepare(self) -> _PreparedBuild:
        options = self.options
        dataset_path = options.dataset_path.resolve(strict=True)
        config_path = options.config_path.resolve(strict=True)
        if options.output_dir.is_symlink():
            raise ValueError("output directory cannot be a symlink")
        output_path = options.output_dir.resolve()
        existing_cache = options.embedding_cache_dir
        fresh_cache_target = options.embedding_cache_output_dir
        fresh_model_path = options.embedding_local_model_path
        if existing_cache is not None:
            if fresh_cache_target is not None or fresh_model_path is not None:
                raise ValueError("existing and fresh embedding-cache inputs are mutually exclusive")
            if existing_cache.is_symlink():
                raise ValueError("embedding cache directory cannot be a symlink")
            cache_dir = existing_cache.resolve(strict=True)
            if not cache_dir.is_dir():
                raise FileNotFoundError(f"embedding cache is not a directory: {cache_dir}")
            build_fresh_cache = False
        else:
            if fresh_cache_target is None or fresh_model_path is None:
                raise ValueError(
                    "provide either embedding_cache_dir or both embedding_cache_output_dir "
                    "and embedding_local_model_path"
                )
            if options.resume:
                raise ValueError(
                    "resume requires an existing embedding cache; fresh caches are never resumed"
                )
            if options.dry_run:
                raise ValueError(
                    "dry-run cannot publish a fresh embedding cache; prebuild it or use "
                    "--embedding-cache-dir"
                )
            if not fresh_cache_target.is_absolute():
                raise ValueError("embedding_cache_output_dir must be an absolute path")
            if not fresh_model_path.is_absolute():
                raise ValueError("embedding_local_model_path must be an absolute path")
            if fresh_cache_target.is_symlink() or fresh_cache_target.exists():
                raise FileExistsError(
                    "embedding_cache_output_dir must be fresh and cannot be a symlink"
                )
            cache_dir = fresh_cache_target
            build_fresh_cache = True
        canonical_cache_path = cache_dir.resolve(strict=not build_fresh_cache)
        if (
            canonical_cache_path == output_path
            or canonical_cache_path in output_path.parents
            or output_path in canonical_cache_path.parents
        ):
            raise ValueError("embedding cache and Stage 1 output trees must be disjoint")
        if not options.unit_id_column.strip():
            raise ValueError("unit_id_column must be non-empty")
        if not _DEVICE.fullmatch(options.device):
            raise ValueError("device must be cpu or one explicit cuda:N device")
        if not options.query_devices or any(
            not _DEVICE.fullmatch(value) for value in options.query_devices
        ):
            raise ValueError("query_devices must contain explicit cpu/cuda:N values")
        if options.num_workers < 1 or options.tfidf_workers < 1:
            raise ValueError("worker counts must be positive")
        if options.query_nuisance_folds < 2:
            raise ValueError("query_nuisance_folds must be at least two")
        if options.tfidf_parallel_backend not in {
            "threads",
            "processes",
            "multiprocessing",
            "fork",
        }:
            raise ValueError("unsupported TF-IDF parallel backend")

        dataset_sha, dataset_stat = _read_stable_sha256(dataset_path)
        config_sha, config_stat = _read_stable_sha256(config_path)
        source_config = load_applied_stage1_config(config_path)
        config_sha_after_parse, config_stat_after_parse = _read_stable_sha256(config_path)
        if (config_sha_after_parse, config_stat_after_parse) != (config_sha, config_stat):
            raise RuntimeError("Stage 1 config changed while it was being parsed")
        config, htr_model_path = _validate_effective_config(
            source_config,
            dataset_path=dataset_path,
            embedding_cache_dir=cache_dir,
            config_dir=config_path.parent,
            seed=options.seed,
        )
        projected_columns = list(
            dict.fromkeys(
                [
                    options.unit_id_column,
                    config.text_column,
                    config.treatment_column,
                    config.outcome_column,
                ]
            )
        )
        if len(projected_columns) != 4:
            raise ValueError("unit ID, text, treatment, and outcome columns must be distinct")
        if any(_ORACLE_COLUMN.search(column) for column in projected_columns):
            raise ValueError("oracle/ground-truth columns cannot be configured as Stage 1 inputs")
        data = pd.read_parquet(dataset_path, columns=projected_columns).reset_index(drop=True)
        dataset_sha_after_parse, dataset_stat_after_parse = _read_stable_sha256(dataset_path)
        if (dataset_sha_after_parse, dataset_stat_after_parse) != (dataset_sha, dataset_stat):
            raise RuntimeError("cohort changed while its configured projection was being read")
        if len(data) < int(config.cv_folds):
            raise ValueError("cohort has fewer rows than configured outer folds")
        if (
            data[options.unit_id_column].isna().any()
            or data[options.unit_id_column].duplicated().any()
        ):
            raise ValueError("unit ID column must be complete and unique")
        if data[config.text_column].isna().any():
            raise ValueError("text column must be complete; encode missing records explicitly")
        if not data[config.text_column].map(lambda value: isinstance(value, str)).all():
            raise TypeError("text column must contain strings")
        for row_index, value in enumerate(data[config.text_column]):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(f"text column row {row_index} must contain valid UTF-8") from exc
        if not data[config.text_column].map(lambda value: bool(value.strip())).all():
            raise ValueError("text column must contain nonempty explicit records")
        _validate_binary(data[config.treatment_column], name=config.treatment_column)
        _validate_binary(data[config.outcome_column], name=config.outcome_column)
        modeling_data = data[
            [config.text_column, config.treatment_column, config.outcome_column]
        ].copy()
        if any(_ORACLE_COLUMN.search(column) for column in modeling_data.columns):
            raise RuntimeError("oracle column entered the Stage 1 modeling projection")

        htr_sha = _directory_tree_sha256(htr_model_path)
        htr_input_nontruncation_audit = _build_htr_input_nontruncation_audit(
            texts=tuple(modeling_data[config.text_column].tolist()),
            config=config,
            htr_model_path=htr_model_path,
            htr_model_tree_sha256=htr_sha,
        )

        fresh_result_identity: Mapping[str, Any] | None = None
        if build_fresh_cache:
            assert fresh_model_path is not None
            result = build_production_embedding_cache(
                dataset_path=dataset_path,
                text_column=config.text_column,
                local_model_path=fresh_model_path,
                sentence_model_name=str(
                    config.architecture.multi_model_forest.embedding_contrast.model_name
                ),
                chunk_configuration=_embedding_chunk_configuration(config),
                target_dir=cache_dir,
                device=options.device,
                batch_size=int(
                    config.architecture.multi_model_forest.embedding_contrast.batch_size
                ),
            )
            cache_dir = result.cache_path
            fresh_result_identity = result.identity()
        embedding_cache = SpentOnlyFrozenChunkEmbeddingCache(cache_dir)
        if embedding_cache.row_count != len(modeling_data):
            raise ValueError("embedding cache row count does not match the cohort")
        _validate_cache_configuration(embedding_cache, config)
        all_rows = tuple(range(len(modeling_data)))
        all_texts = tuple(modeling_data[config.text_column].tolist())
        # This binds every cache row to the exact projected cohort text without
        # exposing treatment/outcome or constructing an embedding model.
        embedding_cache.bind_spent(all_rows, all_texts)
        cache_identity = embedding_cache.identity()
        cache_input_identity = validate_published_production_embedding_cache(
            cache_dir=cache_dir,
            dataset_path=dataset_path,
            text_column=config.text_column,
            sentence_model_name=str(
                config.architecture.multi_model_forest.embedding_contrast.model_name
            ),
            chunk_configuration=_embedding_chunk_configuration(config),
            expected_local_model_path=(fresh_model_path if build_fresh_cache else None),
        )
        if fresh_result_identity is not None and fresh_result_identity != cache_input_identity:
            raise RuntimeError("fresh embedding-cache result differs from its read-only validation")
        if cache_input_identity.get("provider_identity") != cache_identity:
            raise RuntimeError(
                "published embedding-cache validation differs from the active provider"
            )

        registry = build_canonical_split_registry(
            data=modeling_data,
            config=config,
            seed=options.seed,
        )
        registry_sha = _sha256_json(registry)
        exact_inner_contract_status = _exact_inner_contract_registry_status(registry)
        query_config, query_config_identity = self._load_query_config(options.query_config_path)
        hierarchical_discovery_contract_identity = (
            current_production_stage1_hierarchy_contract_identity()
        )
        behavior_identity = _source_identity()
        input_file_identities = {
            "dataset": {
                "path": str(dataset_path),
                "sha256": dataset_sha,
                "stat_identity": list(dataset_stat),
            },
            "source_config": {
                "path": str(config_path),
                "sha256": config_sha,
                "stat_identity": list(config_stat),
            },
        }
        if query_config_identity["provided"]:
            input_file_identities["query_config"] = copy.deepcopy(query_config_identity)
        unit_ids = [
            {"type": type(value).__name__, "value": repr(value)}
            for value in data[options.unit_id_column].tolist()
        ]
        effective_config_payload = _sanitize_secrets(asdict(config))
        architecture_contract = production_stage1_hierarchy_architecture_bindings(
            hierarchical_discovery_contract_identity
        )
        if architecture_contract.get("tfidf_resume_policy") != STAGE1_TFIDF_RESUME_POLICY:
            raise RuntimeError("hierarchy contract changed the sealed-only TF-IDF resume policy")
        hierarchy_spent_evidence_contract = _hierarchy_spent_evidence_contract(
            registry=registry,
            config=config,
            hierarchical_discovery_contract_identity_sha256=(
                hierarchical_discovery_contract_identity["content_sha256"]
            ),
        )
        request_body = {
            "schema_version": STAGE1_BUNDLE_REQUEST_SCHEMA,
            "dataset": {
                "path": str(dataset_path),
                "sha256": dataset_sha,
                "row_count": len(data),
                "columns_read": projected_columns,
                "ordered_unit_id_sha256": _sha256_json(unit_ids),
            },
            "source_config": {"path": str(config_path), "sha256": config_sha},
            "effective_stage1_config": effective_config_payload,
            "embedding_cache": {
                "path": str(cache_dir),
                "identity": cache_identity,
                "production_cache_build_identity": cache_input_identity,
            },
            "htr_model": {
                "path": str(htr_model_path),
                "tree_sha256": htr_sha,
                "sentence_encoder_unfrozen": True,
            },
            "htr_input_nontruncation_audit": htr_input_nontruncation_audit,
            "split_registry_content_sha256": registry_sha,
            "exact_inner_contract": {
                **exact_inner_contract_status,
                "family_adapter_gate": exact_inner_family_adapter_gate(),
            },
            "query_config": {
                "effective": asdict(query_config),
                "source": query_config_identity,
            },
            "runtime": {
                "device": options.device,
                "gpu_ids": list(options.gpu_ids),
                "num_workers": options.num_workers,
                "tfidf_workers": options.tfidf_workers,
                "tfidf_parallel_backend": options.tfidf_parallel_backend,
                "query_devices": list(options.query_devices),
                "query_nuisance_folds": options.query_nuisance_folds,
            },
            "behavior_identity": behavior_identity,
            "hierarchical_discovery_contract_identity": (hierarchical_discovery_contract_identity),
            "architecture_contract": architecture_contract,
            "hierarchy_spent_evidence_contract": hierarchy_spent_evidence_contract,
            "security": {
                "remote_clients_constructed": False,
                "remote_calls_allowed": False,
                "oracle_columns_decoded_or_materialized": False,
                "whole_parquet_container_authenticated": True,
                "plaintext_secrets_persisted": False,
                "manual_digest_approval_required": False,
                "raw_evidence_sidecars_visible_to_prompts": False,
                "partial_tfidf_checkpoint_reuse_allowed": False,
                "htr_source_word_truncation_allowed": False,
                "htr_tokenizer_truncation_allowed": False,
            },
        }
        validate_production_stage1_hierarchy_request_bindings(request_body)
        request_sha = _sha256_json(request_body)
        request = {**request_body, "request_sha256": request_sha}
        self._revalidate_input_files(input_file_identities)
        if _source_identity() != behavior_identity:
            raise RuntimeError("Stage 1 behavior dependencies changed during preflight")
        return _PreparedBuild(
            options=options,
            output_path=output_path,
            data=data,
            modeling_data=modeling_data,
            config=config,
            htr_model_path=htr_model_path,
            htr_model_sha256=htr_sha,
            htr_input_nontruncation_audit=htr_input_nontruncation_audit,
            embedding_cache_path=cache_dir,
            embedding_cache=embedding_cache,
            embedding_cache_identity=cache_identity,
            embedding_cache_input_identity=cache_input_identity,
            registry=registry,
            registry_content_sha256=registry_sha,
            exact_inner_contract_status=exact_inner_contract_status,
            query_config=query_config,
            query_config_identity=query_config_identity,
            input_file_identities=input_file_identities,
            behavior_identity=behavior_identity,
            hierarchical_discovery_contract_identity=(hierarchical_discovery_contract_identity),
            request=request,
            request_sha256=request_sha,
        )

    @staticmethod
    def _load_query_config(
        path: Path | None,
    ) -> tuple[NeuralQueryAgenticForestConfig, Mapping[str, Any]]:
        if path is None:
            config = NeuralQueryAgenticForestConfig()
            identity: Mapping[str, Any] = {
                "provided": False,
                "path": None,
                "sha256": None,
                "stat_identity": None,
            }
        else:
            resolved = path.resolve(strict=True)
            digest, stat_identity = _read_stable_sha256(resolved)
            payload = _load_serialized_mapping(resolved)
            digest_after, stat_after = _read_stable_sha256(resolved)
            if (digest_after, stat_after) != (digest, stat_identity):
                raise RuntimeError("neural-query config changed while it was being parsed")
            unknown = set(payload) - set(asdict(NeuralQueryAgenticForestConfig()))
            if unknown:
                raise ValueError(
                    "neural-query config contains unknown fields: " + ", ".join(sorted(unknown))
                )
            config = NeuralQueryAgenticForestConfig(**dict(payload))
            identity = {
                "provided": True,
                "path": str(resolved),
                "sha256": digest,
                "stat_identity": list(stat_identity),
            }
        config.validate()
        return config, identity

    @staticmethod
    def _revalidate_input_files(
        identities: Mapping[str, Mapping[str, Any]],
    ) -> None:
        for label, identity in identities.items():
            path = Path(str(identity["path"]))
            digest, stat_identity = _read_stable_sha256(path)
            if digest != identity.get("sha256") or list(stat_identity) != list(
                identity.get("stat_identity") or ()
            ):
                raise RuntimeError(f"{label} changed after its authenticated read")

    @staticmethod
    def _revalidate_prepared_inputs(prepared: _PreparedBuild) -> None:
        ProductionStage1BundleBuilder._revalidate_input_files(prepared.input_file_identities)
        if _source_identity() != prepared.behavior_identity:
            raise RuntimeError("Stage 1 behavior dependencies changed after preflight")
        current_hierarchy_identity = validate_production_stage1_hierarchy_request_bindings(
            prepared.request
        )
        if current_hierarchy_identity != prepared.hierarchical_discovery_contract_identity:
            raise RuntimeError("hierarchical discovery contract changed after preflight")
        if _directory_tree_sha256(prepared.htr_model_path) != prepared.htr_model_sha256:
            raise RuntimeError("HTR model tree changed after preflight")
        validated_htr_audit = validate_htr_input_nontruncation_audit(
            prepared.htr_input_nontruncation_audit,
            config=prepared.config,
            expected_rows=len(prepared.modeling_data),
            expected_htr_model_tree_sha256=prepared.htr_model_sha256,
        )
        if validated_htr_audit != prepared.request.get("htr_input_nontruncation_audit"):
            raise RuntimeError("HTR input no-truncation audit changed after preflight")
        current_cache_identity = prepared.embedding_cache.identity()
        if current_cache_identity != prepared.embedding_cache_identity:
            raise RuntimeError("frozen embedding cache changed after preflight")
        current_cache_input_identity = validate_published_production_embedding_cache(
            cache_dir=prepared.embedding_cache_path,
            dataset_path=Path(prepared.input_file_identities["dataset"]["path"]),
            text_column=prepared.config.text_column,
            sentence_model_name=str(
                prepared.config.architecture.multi_model_forest.embedding_contrast.model_name
            ),
            chunk_configuration=_embedding_chunk_configuration(prepared.config),
        )
        if current_cache_input_identity != prepared.embedding_cache_input_identity:
            raise RuntimeError("embedding cache provenance changed after preflight")

    def build(self) -> Mapping[str, Any]:
        prepared = self.prepare()
        if prepared.options.dry_run:
            return {
                "status": "ready_for_candidate_bundle_build_pending_e2e_certification",
                "request_sha256": prepared.request_sha256,
                "row_count": len(prepared.data),
                "outer_fold_count": int(prepared.config.cv_folds),
                "exact_scope_count": len(_registry_scopes(prepared.registry)),
                "required_families": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
                "candidate_bundle_build_ready": True,
                "production_execution_ready": False,
                "production_hierarchy_ready": False,
                "genuine_one_shot_e2e_certified": False,
                "exact_inner_contract": prepared.request["exact_inner_contract"],
                "htr_input_nontruncation_audit": (prepared.htr_input_nontruncation_audit),
                "hierarchical_discovery_contract_identity_sha256": (
                    prepared.hierarchical_discovery_contract_identity["content_sha256"]
                ),
                "manual_digest_approval_required": False,
                "remote_clients_constructed": False,
                "remote_calls_made": False,
                "oracle_columns_decoded_or_materialized": False,
                "whole_parquet_container_authenticated": True,
                "outputs_written": False,
            }
        _require_exact_inner_family_adapters()
        self._revalidate_prepared_inputs(prepared)
        output = prepared.output_path
        self._initialize_output(output, prepared)
        bundle_manifest_path = output / "bundle_manifest.json"
        if bundle_manifest_path.exists():
            result = self._validate_complete_bundle(output, prepared)
            return {**result, "status": "reused_authenticated_bundle"}

        legacy_root = output / f"legacy_all_source_{prepared.request_sha256}"
        tfidf_root = output / f"tfidf_{prepared.request_sha256}"
        query_root = output / f"neural_query_{prepared.request_sha256}"

        legacy_manifest = _load_component_manifest(
            legacy_root,
            request_sha256=prepared.request_sha256,
            component="legacy_all_source",
        )
        if legacy_manifest is None:
            if legacy_root.exists() and any(legacy_root.iterdir()):
                raise RuntimeError(
                    "legacy Stage 1 is incomplete and cannot be resumed safely; use a fresh "
                    "output directory (completed components are reusable only after sealing)"
                )
            self._run_legacy_component(legacy_root, prepared)
            legacy_manifest = _seal_component(
                legacy_root,
                request_sha256=prepared.request_sha256,
                component="legacy_all_source",
            )

        tfidf_manifest = _load_component_manifest(
            tfidf_root,
            request_sha256=prepared.request_sha256,
            component="tfidf",
        )
        if tfidf_manifest is None:
            if tfidf_root.exists() and any(tfidf_root.iterdir()):
                raise RuntimeError(
                    "partial TF-IDF Stage 1 reuse is disabled because every native "
                    "checkpoint is not independently cryptographically registered; "
                    "use a fresh output directory"
                )
            self._run_tfidf_component(tfidf_root, output, prepared)
            tfidf_manifest = _seal_component(
                tfidf_root,
                request_sha256=prepared.request_sha256,
                component="tfidf",
            )

        query_manifest = _load_component_manifest(
            query_root,
            request_sha256=prepared.request_sha256,
            component="neural_query",
        )
        if query_manifest is None:
            if query_root.exists() and any(query_root.iterdir()):
                raise RuntimeError(
                    "neural-query Stage 1 is incomplete and executable checkpoints are never "
                    "loaded across processes; use a fresh output directory"
                )
            self._run_query_component(query_root, output, prepared)
            query_manifest = _seal_component(
                query_root,
                request_sha256=prepared.request_sha256,
                component="neural_query",
            )

        coverage = self._validate_all_scope_coverage(
            legacy_root=legacy_root,
            tfidf_root=tfidf_root,
            query_root=query_root,
            prepared=prepared,
            emit_root_graph=True,
        )
        self._revalidate_prepared_inputs(prepared)
        bundle_body = {
            "schema_version": STAGE1_BUNDLE_MANIFEST_SCHEMA,
            "request_sha256": prepared.request_sha256,
            "hierarchical_discovery_contract_identity_sha256": (
                prepared.hierarchical_discovery_contract_identity["content_sha256"]
            ),
            "manual_digest_approval_required": False,
            "dataset_path": str(prepared.options.dataset_path.resolve()),
            "row_count": len(prepared.data),
            "unit_id_column": prepared.options.unit_id_column,
            "immutable_build_request": self._register_file(
                output / "immutable_build_request.json", output
            ),
            "stage1_config": self._register_file(output / "stage1_config.json", output),
            "split_registry": self._register_file(output / "split_registry.json", output),
            "primary_splits": self._register_file(output / "primary_predictions.parquet", output),
            "row_registry": self._register_file(output / "row_registry.parquet", output),
            "legacy_handoff": self._register_file(
                legacy_root / "handoff" / "discovery_contexts.jsonl", output
            ),
            "tfidf_handoff": self._register_file(
                tfidf_root / "handoff" / "discovery_contexts.jsonl", output
            ),
            "neural_query_artifact_index": self._register_file(
                query_root / "query_artifact_index.json", output
            ),
            "exact_inner_evidence_index": self._register_file(
                output / "exact_inner_evidence_index.json", output
            ),
            "cumulative_all_ten_root_index": self._register_file(
                output / "cumulative_all_ten_root_graph" / "index.json", output
            ),
            "hierarchy_spent_evidence_index": self._register_file(
                output / "hierarchy_spent_evidence_index.json", output
            ),
            "embedding_cache": {
                "path": str(prepared.embedding_cache_path),
                "identity": prepared.embedding_cache_identity,
                "production_cache_build_identity": (prepared.embedding_cache_input_identity),
            },
            "components": {
                "legacy_all_source": {
                    "relative_path": legacy_root.relative_to(output).as_posix(),
                    "manifest_sha256": _sha256_file(legacy_root / "component_manifest.json"),
                    "content_sha256": legacy_manifest["content_sha256"],
                },
                "tfidf": {
                    "relative_path": tfidf_root.relative_to(output).as_posix(),
                    "manifest_sha256": _sha256_file(tfidf_root / "component_manifest.json"),
                    "content_sha256": tfidf_manifest["content_sha256"],
                },
                "neural_query": {
                    "relative_path": query_root.relative_to(output).as_posix(),
                    "manifest_sha256": _sha256_file(query_root / "component_manifest.json"),
                    "content_sha256": query_manifest["content_sha256"],
                },
            },
            "coverage": coverage,
            "security": prepared.request["security"],
        }
        bundle = {**bundle_body, "bundle_sha256": _sha256_json(bundle_body)}
        _write_immutable_json(bundle_manifest_path, bundle)
        result = self._validate_complete_bundle(output, prepared)
        return {**result, "status": "completed"}

    def _initialize_output(self, output: Path, prepared: _PreparedBuild) -> None:
        if output.exists() and output.is_symlink():
            raise ValueError("output directory cannot be a symlink")
        if output.exists() and not output.is_dir():
            raise ValueError("output path must be a directory")
        if output.exists() and any(item.is_symlink() for item in output.rglob("*")):
            raise ValueError("Stage 1 output trees cannot contain symlinks")
        if output.exists() and any(output.iterdir()) and not prepared.options.resume:
            raise ValueError("output directory must be empty unless --resume is supplied")
        output.mkdir(parents=True, exist_ok=True)
        request_path = output / "immutable_build_request.json"
        if request_path.exists():
            existing = json.loads(request_path.read_text(encoding="utf-8"))
            if existing != prepared.request:
                raise RuntimeError("resume request does not match immutable Stage 1 build inputs")
        else:
            _write_immutable_json(request_path, prepared.request)
        config_payload = {
            "model_type": "multi_model_forest",
            "stage": "stage1_bundle",
            "config": _sanitize_secrets(asdict(prepared.config)),
            "production_contract": prepared.request["architecture_contract"],
        }
        _write_immutable_json(output / "stage1_config.json", config_payload)
        _write_immutable_json(output / "split_registry.json", prepared.registry)
        self._write_row_and_split_registries(output, prepared)
        load_tfidf_topic_split_registry(
            output / "split_registry.json",
            dataset_row_count=len(prepared.data),
            outer_fold_count=int(prepared.config.cv_folds),
            inner_fold_count=int(
                prepared.config.architecture.multi_model_forest.candidate_consistency_inner_folds
            ),
        )

    @staticmethod
    def _write_row_and_split_registries(output: Path, prepared: _PreparedBuild) -> None:
        row_registry = pd.DataFrame(
            {
                "_oci_row_id": np.arange(len(prepared.data), dtype=np.int64),
                "unit_id": prepared.data[prepared.options.unit_id_column].to_numpy(),
            }
        )
        by_row: dict[int, int] = {}
        for outer in prepared.registry["outer_folds"]:
            for row_id in outer["heldout_row_ids"]:
                by_row[int(row_id)] = int(outer["outer_fold"])
        splits = pd.DataFrame(
            {
                "_oci_row_id": np.arange(len(prepared.data), dtype=np.int64),
                "outer_fold": [by_row[row_id] for row_id in range(len(prepared.data))],
            }
        )
        splits["cv_fold"] = splits["outer_fold"]
        for path, frame in (
            (output / "row_registry.parquet", row_registry),
            (output / "primary_predictions.parquet", splits),
        ):
            if path.exists():
                existing = pd.read_parquet(path)
                if not existing.equals(frame):
                    raise RuntimeError(f"refusing to mutate immutable registry: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=path.parent, suffix=".parquet", delete=False
                ) as h:
                    temporary = Path(h.name)
                try:
                    frame.to_parquet(temporary, index=False)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)

    def _run_legacy_cumulative_spent_scope(
        self,
        *,
        root: Path,
        prepared: _PreparedBuild,
        request: CumulativeSpentStage1FamilyRequest,
        htr_snapshot: PrivateHTRModelTreeSnapshot,
        bow_models_dir: Path,
        htr_models_dir: Path,
        matched_pair_models_dir: Path,
        proofs_dir: Path,
        embedding_models_dir: Path,
        embedding_execution_dir: Path,
        embedding_proofs_dir: Path,
    ) -> tuple[
        Mapping[str, Any],
        Mapping[str, Mapping[str, Any]],
        Mapping[str, Any],
    ]:
        """Fit seven shared legacy/embedding families without sealed text."""

        canary = CumulativeSpentReplayCanary.from_request(request)
        spent_ids = list(request.spent_row_ids)
        runtime_config = copy.deepcopy(prepared.config)
        runtime_config.architecture.htr_sentence_model = str(htr_snapshot.path)
        runtime_config.architecture.multi_model_agentic_forest = (
            runtime_config.architecture.multi_model_forest
        )
        embedding_config = runtime_config.architecture.multi_model_forest.embedding_contrast
        if not bool(embedding_config.enabled):
            raise ValueError("cumulative all-architecture fit requires embedding contrast")
        runtime_dataset = prepared.modeling_data.copy()
        non_spent = sorted(set(range(len(runtime_dataset))) - set(spent_ids))
        runtime_dataset.loc[
            non_spent,
            [runtime_config.treatment_column, runtime_config.outcome_column],
        ] = np.nan
        runtime_dataset.loc[non_spent, runtime_config.text_column] = ""
        raw_fit_texts, normalized_fit_texts = _native_scope_text_projections(
            row.text for row in request.spent_rows
        )
        raw_canary_texts, normalized_canary_texts = _native_scope_text_projections(
            (canary.text,)
        )
        bow_path = bow_models_dir / request.scope_id
        htr_path = htr_models_dir / request.scope_id
        pair_path = matched_pair_models_dir / request.scope_id
        embedding_path = embedding_models_dir / request.scope_id
        pair_config = runtime_config.architecture.multi_model_forest
        htr_config = runtime_config.architecture.agentic_attention_variable_forest
        bow_capture = NativeBoWProofCaptureSink(
            artifact_dir=bow_path,
            scope_id=request.scope_id,
            outer_fold=request.outer_fold,
            inner_fold=request.provider_inner_fold,
            fit_row_ids=request.spent_row_ids,
            heldout_row_ids=(canary.alias_row_id,),
            fit_texts=normalized_fit_texts,
            heldout_texts=normalized_canary_texts,
            text_column=runtime_config.text_column,
            outcome_type=runtime_config.outcome_type,
            e_clip=float(pair_config.e_clip),
            nuisance_folds=max(2, min(int(pair_config.nuisance_folds), len(spent_ids))),
            effect_folds=max(2, min(int(pair_config.effect_folds), len(spent_ids))),
            view_configs=tuple(asdict(view) for view in pair_config.bow_views),
        )
        htr_capture = NativeHTRProofCaptureSink(
            artifact_dir=htr_path,
            scope_id=request.scope_id,
            outer_fold=request.outer_fold,
            inner_fold=request.provider_inner_fold,
            fit_row_ids=request.spent_row_ids,
            heldout_row_ids=(canary.alias_row_id,),
            fit_texts=raw_fit_texts,
            heldout_texts=raw_canary_texts,
            text_column=runtime_config.text_column,
            treatment_column=runtime_config.treatment_column,
            outcome_column=runtime_config.outcome_column,
            outcome_type=runtime_config.outcome_type,
            e_clip=float(htr_config.e_clip),
            nuisance_folds=max(2, min(int(htr_config.nuisance_folds), len(spent_ids))),
            effect_folds=max(2, min(int(htr_config.effect_folds), len(spent_ids))),
            model_tree_sha256=prepared.htr_model_sha256,
            prediction_batch_size=int(runtime_config.training.batch_size),
            seed=int(prepared.options.seed),
        )
        pair_batch_size = runtime_config.training.effect_batch_size
        if pair_batch_size is None:
            pair_batch_size = runtime_config.training.batch_size
        matched_pair_capture = NativeMatchedPairProofCaptureSink(
            artifact_dir=pair_path,
            scope_id=request.scope_id,
            outer_fold=request.outer_fold,
            inner_fold=request.provider_inner_fold,
            fit_row_ids=request.spent_row_ids,
            heldout_row_ids=(canary.alias_row_id,),
            fit_texts=normalized_fit_texts,
            heldout_texts=normalized_canary_texts,
            text_column=runtime_config.text_column,
            effect_folds=max(2, min(int(pair_config.effect_folds), len(spent_ids))),
            view_configs=tuple(asdict(view) for view in pair_config.bow_views),
            propensity_caliper=float(pair_config.matched_pair_propensity_caliper),
            outcome_caliper=float(pair_config.matched_pair_outcome_caliper),
            max_controls_per_candidate=int(pair_config.matched_pair_max_controls_per_candidate),
            nearest_fallback_controls=int(pair_config.matched_pair_nearest_fallback_controls),
            htr_model_tree_sha256=prepared.htr_model_sha256,
            htr_prediction_batch_size=int(pair_batch_size),
            seed=int(prepared.options.seed),
        )
        with tempfile.TemporaryDirectory(
            prefix=f"production-cumulative-legacy-{request.scope_id}-"
        ) as temporary:
            runner = MultiModelForestStage1Runner(
                dataset=runtime_dataset,
                config=runtime_config,
                output_path=Path(temporary) / "unused_predictions.parquet",
                device=torch.device(prepared.options.device),
                gpu_ids=prepared.options.gpu_ids or None,
                num_workers=prepared.options.num_workers,
                bow_native_capture_sink=bow_capture,
                htr_native_capture_sink=htr_capture,
                matched_pair_native_capture_sink=matched_pair_capture,
            )
            bound_embedding_cache = prepared.embedding_cache.bind_spent(
                request.spent_row_ids,
                raw_fit_texts,
            )
            embedding_generator = _FrozenCacheEmbeddingEvidenceGenerator(
                config=runtime_config,
                embedding_provider=bound_embedding_cache,
                dataset_row_count=len(prepared.modeling_data),
                output_dir=runner.artifact_dir,
            )
            embedding_capture = NativeEmbeddingProofCaptureSink(
                artifact_dir=embedding_path,
                scope_id=request.scope_id,
                outer_fold=request.outer_fold,
                inner_fold=request.provider_inner_fold,
                fit_row_ids=request.spent_row_ids,
                heldout_row_ids=(canary.alias_row_id,),
                fit_texts=raw_fit_texts,
                expected_fit_treatment=np.asarray(
                    [row.treatment for row in request.spent_rows], dtype=float
                ),
                expected_fit_outcome=np.asarray(
                    [row.outcome for row in request.spent_rows], dtype=float
                ),
                text_column=runtime_config.text_column,
                outcome_type=runtime_config.outcome_type,
                embedding_provider=bound_embedding_cache,
                embedding_config=embedding_generator.embedding_config,
                tfidf_nested_calibration_folds=int(
                    runtime_config.architecture.multi_model_forest.tfidf_nested_calibration_folds
                ),
                seed=int(prepared.options.seed),
            )
            embedding_generator._native_embedding_proof_observer = embedding_capture
            runner.embedding_evidence_generator = embedding_generator
            fit_df = runner.dataset.iloc[spent_ids].reset_index(drop=True)
            bundle = runner._build_feature_bundle(
                train_df=fit_df,
                test_df=canary.transform_frame(text_column=runtime_config.text_column),
                outer_fold=request.outer_fold,
            )
        bow_capture.finalize()
        htr_capture.finalize()
        matched_pair_capture.finalize()
        if bundle.handoff_evidence is None:
            raise RuntimeError("cumulative legacy fit produced no native evidence")
        fitted = copy.deepcopy(bundle.handoff_evidence)
        digest = _catalog_ready_legacy_digest(
            importance=fitted.get("importance") or {},
            embedding_evidence={},
            htr_evidence=_htr_concepts_only(
                fitted.get("htr_evidence") or {},
                phrases_per_attention_row=None,
                fallback_contrastive_term_limit=None,
            ),
        )
        provenance = FoldEvidenceProvenance(
            outer_fold=request.outer_fold,
            train_row_ids=request.spent_row_ids,
            heldout_row_ids=request.sealed_row_ids,
            scope="inner_train",
            inner_fold=request.provider_inner_fold,
            artifact_id=f"production-cumulative-legacy-{request.scope_id}",
        )
        if provenance.split_fingerprint != request.split_scope_fingerprint:
            raise RuntimeError("cumulative legacy fit changed its canonical split")
        catalog = build_role_neutral_evidence_catalog(
            (
                FoldEvidenceInput(
                    LEGACY_ALL_SOURCE,
                    {
                        "outer_fold": request.outer_fold,
                        "inner_fold": request.provider_inner_fold,
                        "scope": "inner_train",
                        "n_rows": len(spent_ids),
                        "context": {"evidence_digest": digest},
                    },
                    provenance,
                ),
            ),
            require_all_source_kinds=False,
            require_all_architecture_families=False,
            require_upstream_completeness=False,
        )
        missing = [
            family
            for family in PRODUCTION_CUMULATIVE_LEGACY_NATIVE_FAMILY_ADAPTERS
            if not catalog.family_atoms(family)
        ]
        if missing:
            raise RuntimeError(
                "cumulative legacy scope lacks native family evidence: " + ", ".join(missing)
            )
        configurations = _cumulative_legacy_configuration_by_family(
            config=runtime_config,
            scope_id=request.scope_id,
            split_registry_content_sha256=prepared.registry_content_sha256,
            htr_model_tree_sha256=prepared.htr_model_sha256,
            seed=int(prepared.options.seed),
        )
        registration = _register_legacy_cumulative_spent_native_scope(
            component_root=root,
            proof_directory=proofs_dir / request.scope_id,
            request=request,
            catalog=catalog,
            replay_canary=canary,
            capture_artifact_by_family={
                BOW_NUISANCE: bow_path,
                BOW_R_LOSS: bow_path,
                HTR_NEURAL: htr_path,
                MATCHED_PAIR_UPLIFT: pair_path,
            },
            configuration_by_family=configurations,
            htr_model_path=htr_snapshot.path,
            htr_model_sha256=prepared.htr_model_sha256,
            device=torch.device(prepared.options.device),
        )
        embedding_requests = {
            family: _cumulative_request_for_family(request, family=family)
            for family in PRODUCTION_CUMULATIVE_EMBEDDING_NATIVE_FAMILY_ADAPTERS
        }
        embedding_emissions = emit_cumulative_spent_embedding_capture(
            requests=embedding_requests,
            replay_canary=canary,
            capture_sink=embedding_capture,
            execution_record_dir=embedding_execution_dir / request.scope_id,
        )
        embedding_registration = _register_cumulative_spent_embedding_scope(
            component_root=root,
            proof_directory=embedding_proofs_dir / request.scope_id,
            requests=embedding_requests,
            replay_canary=canary,
            emissions=embedding_emissions,
        )
        return registration, configurations, embedding_registration

    def _run_legacy_component(self, root: Path, prepared: _PreparedBuild) -> None:
        root.mkdir(parents=True, exist_ok=False)
        handoff_dir = root / "handoff"
        handoff_dir.mkdir(parents=True, exist_ok=False)
        raw_sidecar_dir = root / "raw_evidence_sidecars"
        raw_sidecar_dir.mkdir(parents=True, exist_ok=False)
        native_bow_models_dir = root / "native_bow_models"
        native_bow_models_dir.mkdir(parents=True, exist_ok=False)
        native_htr_models_dir = root / "native_htr_models"
        native_htr_models_dir.mkdir(parents=True, exist_ok=False)
        native_matched_pair_models_dir = root / "native_matched_pair_models"
        native_matched_pair_models_dir.mkdir(parents=True, exist_ok=False)
        native_embedding_models_dir = root / "native_embedding_models"
        native_embedding_models_dir.mkdir(parents=True, exist_ok=False)
        cumulative_bow_models_dir = root / "cumulative_native_bow_models"
        cumulative_bow_models_dir.mkdir(parents=True, exist_ok=False)
        cumulative_htr_models_dir = root / "cumulative_native_htr_models"
        cumulative_htr_models_dir.mkdir(parents=True, exist_ok=False)
        cumulative_matched_pair_models_dir = root / "cumulative_native_matched_pair_models"
        cumulative_matched_pair_models_dir.mkdir(parents=True, exist_ok=False)
        cumulative_proofs_dir = root / "cumulative_legacy_family_proofs"
        cumulative_proofs_dir.mkdir(parents=True, exist_ok=False)
        cumulative_embedding_models_dir = root / "cumulative_native_embedding_models"
        cumulative_embedding_models_dir.mkdir(parents=True, exist_ok=False)
        cumulative_embedding_execution_dir = root / "cumulative_embedding_execution_records"
        cumulative_embedding_execution_dir.mkdir(parents=True, exist_ok=False)
        cumulative_embedding_proofs_dir = root / "cumulative_embedding_family_proofs"
        cumulative_embedding_proofs_dir.mkdir(parents=True, exist_ok=False)
        scopes = _registry_scopes(prepared.registry)
        handoff_rows: list[Mapping[str, Any]] = []
        scope_index: list[Mapping[str, Any]] = []
        native_bow_proof_rows: list[Mapping[str, Any]] = []
        native_htr_proof_rows: list[Mapping[str, Any]] = []
        native_matched_pair_proof_rows: list[Mapping[str, Any]] = []
        native_embedding_proof_rows: list[Mapping[str, Any]] = []
        exact_registry = _canonical_exact_registry_from_wrapper(prepared.registry)
        review_rounds = int(exact_registry.inner_fold_count) - 3
        if review_rounds < 1:
            raise ValueError(
                "cumulative hierarchy emission requires at least four canonical inner "
                "partitions: three initially spent plus one review gate"
            )
        # Imported lazily because the authenticated handoff imports this module's
        # hash helpers. At execution time this module is already fully initialized.
        from .production_stage1_hierarchy_handoff import CanonicalHierarchySpentSchedule

        cumulative_schedule = CanonicalHierarchySpentSchedule.build(
            registry=exact_registry,
            review_rounds=review_rounds,
        )
        cumulative_registrations: list[Mapping[str, Any]] = []
        cumulative_embedding_registrations: list[Mapping[str, Any]] = []
        cumulative_expected_requests: dict[str, CumulativeSpentStage1FamilyRequest] = {}
        cumulative_expected_configurations: dict[
            str,
            Mapping[str, Mapping[str, Any]],
        ] = {}
        htr_snapshot = PrivateHTRModelTreeSnapshot(prepared.htr_model_path)
        try:
            if htr_snapshot.sha256 != prepared.htr_model_sha256:
                raise RuntimeError("HTR model tree differs from the authenticated request")
            runtime_config = copy.deepcopy(prepared.config)
            runtime_config.architecture.htr_sentence_model = str(htr_snapshot.path)
            runtime_config.architecture.multi_model_agentic_forest = (
                runtime_config.architecture.multi_model_forest
            )
            for scope in scopes:
                fit_ids = list(map(int, scope["fit_row_ids"]))
                heldout_ids = list(map(int, scope["heldout_row_ids"]))
                scope_ids = fit_ids + heldout_ids
                raw_fit_texts, normalized_fit_texts = _native_scope_text_projections(
                    prepared.modeling_data.iloc[fit_ids][prepared.config.text_column]
                )
                raw_heldout_texts, normalized_heldout_texts = _native_scope_text_projections(
                    prepared.modeling_data.iloc[heldout_ids][prepared.config.text_column]
                )
                # Embedding discovery is an exact-fit operation.  Its native
                # provider is never bound to registered heldout rows, even for
                # text-only transforms used by other Stage 1 architectures.
                bound_cache = prepared.embedding_cache.bind_spent(fit_ids, raw_fit_texts)
                # Keep global row positions for the authenticated cache while
                # physically removing labels outside the current fit partition.
                # Rows outside an inner scope also have their text blanked.
                runtime_dataset = prepared.modeling_data.copy()
                non_fit = sorted(set(range(len(runtime_dataset))) - set(fit_ids))
                runtime_dataset.loc[
                    non_fit,
                    [runtime_config.treatment_column, runtime_config.outcome_column],
                ] = np.nan
                outside_scope = sorted(set(range(len(runtime_dataset))) - set(scope_ids))
                if outside_scope:
                    runtime_dataset.loc[outside_scope, runtime_config.text_column] = ""
                with tempfile.TemporaryDirectory(
                    prefix=f"production-stage1-{scope['scope_id']}-"
                ) as raw_scope_dir:
                    scope_dir = Path(raw_scope_dir)
                    runner = MultiModelForestStage1Runner(
                        dataset=runtime_dataset,
                        config=copy.deepcopy(runtime_config),
                        output_path=scope_dir / "unused_predictions.parquet",
                        device=torch.device(prepared.options.device),
                        gpu_ids=prepared.options.gpu_ids or None,
                        num_workers=prepared.options.num_workers,
                    )
                    runner.embedding_evidence_generator = _FrozenCacheEmbeddingEvidenceGenerator(
                        config=runtime_config,
                        embedding_provider=bound_cache,
                        dataset_row_count=len(prepared.modeling_data),
                        output_dir=runner.artifact_dir,
                    )
                    fit_df = runner.dataset.iloc[fit_ids].reset_index(drop=True)
                    # The evidence builder receives text and identity only for
                    # transform rows. Held-out treatment/outcome values do not
                    # exist anywhere in this scope runner.
                    heldout_df = runner.dataset.iloc[heldout_ids][
                        ["_oci_row_id", runtime_config.text_column]
                    ].reset_index(drop=True)
                    is_inner = scope["inner_fold"] is not None
                    bow_capture = None
                    bow_capture_metadata = None
                    htr_capture = None
                    htr_capture_metadata = None
                    matched_pair_capture = None
                    matched_pair_capture_metadata = None
                    embedding_capture = None
                    embedding_capture_metadata = None
                    if is_inner:
                        embedding_capture = NativeEmbeddingProofCaptureSink(
                            artifact_dir=(native_embedding_models_dir / str(scope["scope_id"])),
                            scope_id=str(scope["scope_id"]),
                            outer_fold=int(scope["outer_fold"]),
                            inner_fold=int(scope["inner_fold"]),
                            fit_row_ids=fit_ids,
                            heldout_row_ids=heldout_ids,
                            fit_texts=raw_fit_texts,
                            expected_fit_treatment=fit_df[runtime_config.treatment_column].to_numpy(
                                dtype=float
                            ),
                            expected_fit_outcome=fit_df[runtime_config.outcome_column].to_numpy(
                                dtype=float
                            ),
                            text_column=runtime_config.text_column,
                            outcome_type=runtime_config.outcome_type,
                            embedding_provider=bound_cache,
                            embedding_config=(runner.embedding_evidence_generator.embedding_config),
                            tfidf_nested_calibration_folds=int(
                                runtime_config.architecture.multi_model_forest.tfidf_nested_calibration_folds
                            ),
                            seed=int(prepared.options.seed),
                        )
                        runner.embedding_evidence_generator._native_embedding_proof_observer = (
                            embedding_capture
                        )
                        bow_capture = NativeBoWProofCaptureSink(
                            artifact_dir=native_bow_models_dir / str(scope["scope_id"]),
                            scope_id=str(scope["scope_id"]),
                            outer_fold=int(scope["outer_fold"]),
                            inner_fold=int(scope["inner_fold"]),
                            fit_row_ids=fit_ids,
                            heldout_row_ids=heldout_ids,
                            fit_texts=normalized_fit_texts,
                            heldout_texts=normalized_heldout_texts,
                            text_column=runtime_config.text_column,
                            outcome_type=runtime_config.outcome_type,
                            e_clip=float(runtime_config.architecture.multi_model_forest.e_clip),
                            nuisance_folds=int(
                                runtime_config.architecture.multi_model_forest.nuisance_folds
                            ),
                            effect_folds=int(
                                runtime_config.architecture.multi_model_forest.effect_folds
                            ),
                            view_configs=tuple(
                                asdict(view)
                                for view in runtime_config.architecture.multi_model_forest.bow_views
                            ),
                        )
                        runner.bow_native_capture_sink = bow_capture
                        htr_config = runtime_config.architecture.agentic_attention_variable_forest
                        htr_capture = NativeHTRProofCaptureSink(
                            artifact_dir=native_htr_models_dir / str(scope["scope_id"]),
                            scope_id=str(scope["scope_id"]),
                            outer_fold=int(scope["outer_fold"]),
                            inner_fold=int(scope["inner_fold"]),
                            fit_row_ids=fit_ids,
                            heldout_row_ids=heldout_ids,
                            fit_texts=raw_fit_texts,
                            heldout_texts=raw_heldout_texts,
                            text_column=runtime_config.text_column,
                            treatment_column=runtime_config.treatment_column,
                            outcome_column=runtime_config.outcome_column,
                            outcome_type=runtime_config.outcome_type,
                            e_clip=float(htr_config.e_clip),
                            nuisance_folds=max(
                                2,
                                min(int(htr_config.nuisance_folds), len(fit_df)),
                            ),
                            effect_folds=max(
                                2,
                                min(int(htr_config.effect_folds), len(fit_df)),
                            ),
                            model_tree_sha256=prepared.htr_model_sha256,
                            prediction_batch_size=int(runtime_config.training.batch_size),
                            seed=int(prepared.options.seed),
                        )
                        runner.htr_native_capture_sink = htr_capture
                        pair_config = runtime_config.architecture.multi_model_forest
                        pair_batch_size = runtime_config.training.effect_batch_size
                        if pair_batch_size is None:
                            pair_batch_size = runtime_config.training.batch_size
                        matched_pair_capture = NativeMatchedPairProofCaptureSink(
                            artifact_dir=(native_matched_pair_models_dir / str(scope["scope_id"])),
                            scope_id=str(scope["scope_id"]),
                            outer_fold=int(scope["outer_fold"]),
                            inner_fold=int(scope["inner_fold"]),
                            fit_row_ids=fit_ids,
                            heldout_row_ids=heldout_ids,
                            fit_texts=normalized_fit_texts,
                            heldout_texts=normalized_heldout_texts,
                            text_column=runtime_config.text_column,
                            effect_folds=max(
                                2,
                                min(int(pair_config.effect_folds), len(fit_df)),
                            ),
                            view_configs=tuple(asdict(view) for view in pair_config.bow_views),
                            propensity_caliper=float(pair_config.matched_pair_propensity_caliper),
                            outcome_caliper=float(pair_config.matched_pair_outcome_caliper),
                            max_controls_per_candidate=int(
                                pair_config.matched_pair_max_controls_per_candidate
                            ),
                            nearest_fallback_controls=int(
                                pair_config.matched_pair_nearest_fallback_controls
                            ),
                            htr_model_tree_sha256=prepared.htr_model_sha256,
                            htr_prediction_batch_size=int(pair_batch_size),
                            seed=int(prepared.options.seed),
                        )
                        runner.matched_pair_native_capture_sink = matched_pair_capture
                    bundle = runner._build_feature_bundle(
                        train_df=fit_df,
                        test_df=heldout_df,
                        outer_fold=int(scope["outer_fold"]),
                    )
                    if bow_capture is not None:
                        bow_capture_metadata = bow_capture.finalize()
                    if htr_capture is not None:
                        htr_capture_metadata = htr_capture.finalize()
                    if matched_pair_capture is not None:
                        matched_pair_capture_metadata = matched_pair_capture.finalize()
                    if embedding_capture is not None:
                        embedding_capture_metadata = embedding_capture.finalize()
                    if bundle.handoff_evidence is None:
                        raise RuntimeError(
                            f"legacy scope produced no evidence: {scope['scope_id']}"
                        )
                    fitted = copy.deepcopy(bundle.handoff_evidence)
                    safe_embedding = _embedding_concepts_only(
                        fitted.get("embedding_contrast_evidence") or {},
                        contrastive_term_limit=None,
                    )
                    safe_htr = _htr_concepts_only(
                        fitted.get("htr_evidence") or {},
                        phrases_per_attention_row=None,
                        fallback_contrastive_term_limit=None,
                    )
                    digest = _catalog_ready_legacy_digest(
                        importance=fitted.get("importance") or {},
                        embedding_evidence=safe_embedding,
                        htr_evidence=safe_htr,
                    )
                    matched_pair_proofs = _matched_pair_subproducer_proofs(
                        bundle=bundle,
                        expected_bow_views=tuple(
                            str(view.name)
                            for view in runtime_config.architecture.multi_model_forest.bow_views
                        ),
                        scope_id=str(scope["scope_id"]),
                        fit_row_ids=fit_ids,
                        heldout_row_ids=heldout_ids,
                    )
                    raw_sidecar = _write_raw_evidence_sidecar(
                        raw_sidecar_dir / f"{scope['scope_id']}.json",
                        component_root=root,
                        scope=scope,
                        split_registry_content_sha256=prepared.registry_content_sha256,
                        raw_evidence=fitted,
                        matched_pair_proofs=matched_pair_proofs,
                    )
                    native_bow_registration = None
                    native_htr_registration = None
                    native_matched_pair_registration = None
                    native_embedding_registration = None
                    if is_inner:
                        if bow_capture_metadata is None:
                            raise RuntimeError("exact-inner BoW scope produced no native capture")
                        if embedding_capture_metadata is None:
                            raise RuntimeError(
                                "exact-inner embedding scope produced no native capture"
                            )
                        outer_fold = int(scope["outer_fold"])
                        inner_fold = int(scope["inner_fold"])
                        provenance = FoldEvidenceProvenance(
                            outer_fold=outer_fold,
                            train_row_ids=tuple(fit_ids),
                            heldout_row_ids=tuple(heldout_ids),
                            scope="inner_train",
                            inner_fold=inner_fold,
                            artifact_id=(f"production-stage1-bow-native-proof-{scope['scope_id']}"),
                        )
                        catalog = build_role_neutral_evidence_catalog(
                            (
                                FoldEvidenceInput(
                                    LEGACY_ALL_SOURCE,
                                    {
                                        "outer_fold": outer_fold,
                                        "inner_fold": inner_fold,
                                        "scope": "inner_train",
                                        "n_rows": len(fit_ids),
                                        "context": {"evidence_digest": digest},
                                    },
                                    provenance,
                                ),
                            ),
                            require_all_source_kinds=False,
                            require_all_architecture_families=False,
                            require_upstream_completeness=False,
                        )
                        missing_bow = [
                            family
                            for family in PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS
                            if not catalog.family_atoms(family)
                        ]
                        if missing_bow:
                            raise RuntimeError(
                                f"BoW native scope {scope['scope_id']} lacks family evidence: "
                                + ", ".join(missing_bow)
                            )
                        if not catalog.family_atoms(HTR_NEURAL):
                            raise RuntimeError(
                                f"HTR native scope {scope['scope_id']} lacks family evidence"
                            )
                        if not catalog.family_atoms(MATCHED_PAIR_UPLIFT):
                            raise RuntimeError(
                                f"matched-pair native scope {scope['scope_id']} lacks "
                                "family evidence"
                            )
                        missing_embedding = [
                            family
                            for family in PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS
                            if not catalog.family_atoms(family)
                        ]
                        if missing_embedding:
                            raise RuntimeError(
                                f"embedding native scope {scope['scope_id']} lacks family "
                                "evidence: " + ", ".join(missing_embedding)
                            )
                        split = exact_registry.inner_split(outer_fold, inner_fold)
                        if split.fit_row_ids != tuple(fit_ids) or split.heldout_row_ids != tuple(
                            heldout_ids
                        ):
                            raise RuntimeError(
                                "BoW native proof changed the canonical exact-inner split"
                            )
                        bow_configuration = {
                            "schema_version": (STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA),
                            "scope_id": str(scope["scope_id"]),
                            "text_column": runtime_config.text_column,
                            "treatment_column": runtime_config.treatment_column,
                            "outcome_column": runtime_config.outcome_column,
                            "outcome_type": runtime_config.outcome_type,
                            "e_clip": float(runtime_config.architecture.multi_model_forest.e_clip),
                            "nuisance_folds": int(
                                runtime_config.architecture.multi_model_forest.nuisance_folds
                            ),
                            "effect_folds": int(
                                runtime_config.architecture.multi_model_forest.effect_folds
                            ),
                            "bow_views": [
                                asdict(view)
                                for view in runtime_config.architecture.multi_model_forest.bow_views
                            ],
                            "capture_schema_version": BOW_NATIVE_CAPTURE_SCHEMA,
                            "heldout_label_policy": "id_and_text_only",
                            "r_loss_nuisance_source": "ensemble_mean_nuisance",
                            "split_registry_content_sha256": (prepared.registry_content_sha256),
                        }
                        native_bow_registration = _register_bow_native_family_proofs(
                            component_root=root,
                            proof_directory=(Path("native_family_proofs") / str(scope["scope_id"])),
                            scope_id=str(scope["scope_id"]),
                            catalog=catalog,
                            capture_artifact_path=(native_bow_models_dir / str(scope["scope_id"])),
                            source_artifact_path=(root / str(raw_sidecar["relative_path"])),
                            outer_fold=outer_fold,
                            inner_fold=inner_fold,
                            fit_row_ids=fit_ids,
                            heldout_row_ids=heldout_ids,
                            fit_texts=tuple(_normalize_texts(fit_df[runtime_config.text_column])),
                            heldout_texts=tuple(
                                _normalize_texts(heldout_df[runtime_config.text_column])
                            ),
                            fit_treatment=fit_df[runtime_config.treatment_column].to_numpy(
                                dtype=float
                            ),
                            fit_outcome=fit_df[runtime_config.outcome_column].to_numpy(dtype=float),
                            split_scope_fingerprint=split.scope_fingerprint,
                            data_projection_sha256=_exact_inner_projection_sha256(
                                modeling_data=prepared.modeling_data,
                                config=prepared.config,
                                fit_row_ids=fit_ids,
                                heldout_row_ids=heldout_ids,
                            ),
                            configuration=bow_configuration,
                        )
                        native_bow_proof_rows.append(
                            {
                                "scope_id": str(scope["scope_id"]),
                                "outer_fold": outer_fold,
                                "inner_fold": inner_fold,
                                "registered_families": list(
                                    PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS
                                ),
                                "content_sha256": native_bow_registration["content_sha256"],
                                "registration": native_bow_registration["registration"],
                            }
                        )
                        embedding_configuration = {
                            "schema_version": (STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA),
                            "scope_id": str(scope["scope_id"]),
                            "text_column": runtime_config.text_column,
                            "treatment_column": runtime_config.treatment_column,
                            "outcome_column": runtime_config.outcome_column,
                            "outcome_type": runtime_config.outcome_type,
                            "embedding_config": copy.deepcopy(
                                embedding_capture_metadata["embedding_config"]
                            ),
                            "capture_schema_version": EMBEDDING_NATIVE_CAPTURE_SCHEMA,
                            "semantic_policy_schema_version": (
                                SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA
                            ),
                            "tfidf_nested_calibration_folds": int(
                                runtime_config.architecture.multi_model_forest.tfidf_nested_calibration_folds
                            ),
                            "heldout_label_policy": "id_only_no_transform",
                            "seed": int(prepared.options.seed),
                            "split_registry_content_sha256": (prepared.registry_content_sha256),
                        }
                        native_embedding_registration = _register_embedding_native_family_proofs(
                            component_root=root,
                            proof_directory=(
                                Path("native_embedding_family_proofs") / str(scope["scope_id"])
                            ),
                            scope_id=str(scope["scope_id"]),
                            catalog=catalog,
                            capture_artifact_path=(
                                native_embedding_models_dir / str(scope["scope_id"])
                            ),
                            outer_fold=outer_fold,
                            inner_fold=inner_fold,
                            fit_row_ids=fit_ids,
                            heldout_row_ids=heldout_ids,
                            modeling_data=prepared.modeling_data,
                            text_column=runtime_config.text_column,
                            treatment_column=runtime_config.treatment_column,
                            outcome_column=runtime_config.outcome_column,
                            embedding_provider=bound_cache,
                            split_scope_fingerprint=split.scope_fingerprint,
                            data_projection_sha256=_exact_inner_projection_sha256(
                                modeling_data=prepared.modeling_data,
                                config=prepared.config,
                                fit_row_ids=fit_ids,
                                heldout_row_ids=heldout_ids,
                            ),
                            configuration=embedding_configuration,
                        )
                        native_embedding_proof_rows.append(
                            {
                                "scope_id": str(scope["scope_id"]),
                                "outer_fold": outer_fold,
                                "inner_fold": inner_fold,
                                "registered_families": list(
                                    PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS
                                ),
                                "content_sha256": native_embedding_registration["content_sha256"],
                                "registration": native_embedding_registration["registration"],
                            }
                        )
                        if htr_capture_metadata is None:
                            raise RuntimeError("exact-inner HTR scope produced no native capture")
                        htr_configuration = {
                            "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
                            "scope_id": str(scope["scope_id"]),
                            "text_column": runtime_config.text_column,
                            "treatment_column": runtime_config.treatment_column,
                            "outcome_column": runtime_config.outcome_column,
                            "outcome_type": runtime_config.outcome_type,
                            "e_clip": float(
                                runtime_config.architecture.agentic_attention_variable_forest.e_clip
                            ),
                            "nuisance_folds": int(
                                runtime_config.architecture.agentic_attention_variable_forest.nuisance_folds
                            ),
                            "effect_folds": int(
                                runtime_config.architecture.agentic_attention_variable_forest.effect_folds
                            ),
                            "effect_objectives": list(("pseudo_outcome_mse", "squared_r_loss")),
                            "nuisance_calibration": str(
                                runtime_config.architecture.agentic_attention_variable_forest.nuisance_calibration
                            ),
                            "capture_schema_version": HTR_NATIVE_CAPTURE_SCHEMA,
                            "htr_model_tree_sha256": prepared.htr_model_sha256,
                            "heldout_label_policy": "id_and_text_only",
                            "split_registry_content_sha256": (prepared.registry_content_sha256),
                        }
                        native_htr_registration = _register_htr_native_family_proof(
                            component_root=root,
                            proof_directory=(
                                Path("native_htr_family_proofs") / str(scope["scope_id"])
                            ),
                            scope_id=str(scope["scope_id"]),
                            catalog=catalog,
                            capture_artifact_path=(native_htr_models_dir / str(scope["scope_id"])),
                            source_artifact_path=(root / str(raw_sidecar["relative_path"])),
                            outer_fold=outer_fold,
                            inner_fold=inner_fold,
                            fit_row_ids=fit_ids,
                            heldout_row_ids=heldout_ids,
                            fit_texts=raw_fit_texts,
                            heldout_texts=raw_heldout_texts,
                            fit_treatment=fit_df[runtime_config.treatment_column].to_numpy(
                                dtype=float
                            ),
                            fit_outcome=fit_df[runtime_config.outcome_column].to_numpy(dtype=float),
                            split_scope_fingerprint=split.scope_fingerprint,
                            data_projection_sha256=_exact_inner_projection_sha256(
                                modeling_data=prepared.modeling_data,
                                config=prepared.config,
                                fit_row_ids=fit_ids,
                                heldout_row_ids=heldout_ids,
                            ),
                            configuration=htr_configuration,
                            htr_model_path=htr_snapshot.path,
                            htr_model_sha256=prepared.htr_model_sha256,
                            device=torch.device(prepared.options.device),
                        )
                        native_htr_proof_rows.append(
                            {
                                "scope_id": str(scope["scope_id"]),
                                "outer_fold": outer_fold,
                                "inner_fold": inner_fold,
                                "registered_families": list(
                                    PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS
                                ),
                                "content_sha256": native_htr_registration["content_sha256"],
                                "registration": native_htr_registration["registration"],
                            }
                        )
                        if matched_pair_capture_metadata is None:
                            raise RuntimeError(
                                "exact-inner matched-pair scope produced no native capture"
                            )
                        pair_config = runtime_config.architecture.multi_model_forest
                        matched_pair_configuration = {
                            "schema_version": (STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA),
                            "scope_id": str(scope["scope_id"]),
                            "text_column": runtime_config.text_column,
                            "outcome_type": runtime_config.outcome_type,
                            "effect_folds": max(
                                2,
                                min(int(pair_config.effect_folds), len(fit_df)),
                            ),
                            "bow_views": [asdict(view) for view in pair_config.bow_views],
                            "matching_configuration": {
                                "propensity_caliper": float(
                                    pair_config.matched_pair_propensity_caliper
                                ),
                                "outcome_caliper": float(pair_config.matched_pair_outcome_caliper),
                                "max_controls_per_candidate": int(
                                    pair_config.matched_pair_max_controls_per_candidate
                                ),
                                "nearest_fallback_controls": int(
                                    pair_config.matched_pair_nearest_fallback_controls
                                ),
                            },
                            "required_subproducers": ["bow", "htr"],
                            "capture_schema_version": MATCHED_PAIR_NATIVE_CAPTURE_SCHEMA,
                            "htr_model_tree_sha256": prepared.htr_model_sha256,
                            "heldout_label_policy": "id_and_text_only",
                            "split_registry_content_sha256": (prepared.registry_content_sha256),
                        }
                        native_matched_pair_registration = (
                            _register_matched_pair_native_family_proof(
                                component_root=root,
                                proof_directory=(
                                    Path("native_matched_pair_family_proofs")
                                    / str(scope["scope_id"])
                                ),
                                scope_id=str(scope["scope_id"]),
                                catalog=catalog,
                                capture_artifact_path=(
                                    native_matched_pair_models_dir / str(scope["scope_id"])
                                ),
                                source_artifact_path=(root / str(raw_sidecar["relative_path"])),
                                outer_fold=outer_fold,
                                inner_fold=inner_fold,
                                fit_row_ids=fit_ids,
                                heldout_row_ids=heldout_ids,
                                fit_texts=tuple(
                                    _normalize_texts(fit_df[runtime_config.text_column])
                                ),
                                heldout_texts=tuple(
                                    _normalize_texts(heldout_df[runtime_config.text_column])
                                ),
                                fit_treatment=fit_df[runtime_config.treatment_column].to_numpy(
                                    dtype=float
                                ),
                                fit_outcome=fit_df[runtime_config.outcome_column].to_numpy(
                                    dtype=float
                                ),
                                split_scope_fingerprint=split.scope_fingerprint,
                                data_projection_sha256=_exact_inner_projection_sha256(
                                    modeling_data=prepared.modeling_data,
                                    config=prepared.config,
                                    fit_row_ids=fit_ids,
                                    heldout_row_ids=heldout_ids,
                                ),
                                configuration=matched_pair_configuration,
                                htr_model_path=htr_snapshot.path,
                                htr_model_sha256=prepared.htr_model_sha256,
                                device=torch.device(prepared.options.device),
                            )
                        )
                        native_matched_pair_proof_rows.append(
                            {
                                "scope_id": str(scope["scope_id"]),
                                "outer_fold": outer_fold,
                                "inner_fold": inner_fold,
                                "registered_families": list(
                                    PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS
                                ),
                                "content_sha256": native_matched_pair_registration[
                                    "content_sha256"
                                ],
                                "registration": native_matched_pair_registration["registration"],
                            }
                        )
                    metrics = {
                        "evidence_fit_rows": len(fit_ids),
                        "evidence_heldout_rows": len(heldout_ids),
                        "heldout_treatment_read": False,
                        "heldout_outcome_read": False,
                        "oracle_value_materialized": False,
                    }
                    handoff_result = {
                        "metrics": metrics,
                        "importance": copy.deepcopy(fitted.get("importance") or {}),
                        "embedding_contrast_evidence": safe_embedding,
                        "htr_evidence": safe_htr,
                        "context": {"evidence_digest": digest},
                    }
                    fold_key = (
                        int(scope["outer_fold"]) * 1000 + int(scope["inner_fold"])
                        if is_inner
                        else int(scope["outer_fold"])
                    )
                    row = _agentic_discovery_handoff_row(
                        handoff_result,
                        fold_key=fold_key,
                        outer_fold=int(scope["outer_fold"]),
                        scope=(
                            "candidate_consistency_inner_train" if is_inner else "full_outer_train"
                        ),
                        n_rows=len(fit_ids),
                        inner_fold=(int(scope["inner_fold"]) if is_inner else None),
                        heldout_rows=(len(heldout_ids) if is_inner else None),
                    )
                    row.update(
                        {
                            "fit_row_ids": fit_ids,
                            "heldout_row_ids": heldout_ids,
                            "fit_row_fingerprint": row_set_fingerprint(fit_ids),
                            "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
                            "split_registry_content_sha256": prepared.registry_content_sha256,
                            "evidence_scope_fit_was_executed": True,
                            "evidence_reused_from_fold_key": None,
                            "heldout_labels_supplied_to_evidence_builder": False,
                            "lossless_concept_catalog_projection": True,
                            "prompt_compactor_used": False,
                            "raw_evidence_sidecar_sha256": raw_sidecar["sha256"],
                        }
                    )
                    handoff_rows.append(row)
                    scope_index.append(
                        {
                            "scope_id": scope["scope_id"],
                            "outer_fold": scope["outer_fold"],
                            "inner_fold": scope["inner_fold"],
                            "fit_row_fingerprint": row["fit_row_fingerprint"],
                            "heldout_row_fingerprint": row["heldout_row_fingerprint"],
                            "evidence_sha256": _sha256_json(row),
                            "fit_was_executed": True,
                            "reused_full_outer_evidence": False,
                            "heldout_labels_supplied": False,
                            "lossless_concept_catalog_projection": True,
                            "prompt_compactor_used": False,
                            "raw_evidence_sidecar": raw_sidecar,
                            "matched_pair_subproducer_proofs_sha256": matched_pair_proofs[
                                "content_sha256"
                            ],
                            "native_bow_family_proof_registration": (
                                None
                                if native_bow_registration is None
                                else native_bow_registration["registration"]
                            ),
                            "native_htr_family_proof_registration": (
                                None
                                if native_htr_registration is None
                                else native_htr_registration["registration"]
                            ),
                            "native_matched_pair_family_proof_registration": (
                                None
                                if native_matched_pair_registration is None
                                else native_matched_pair_registration["registration"]
                            ),
                            "native_embedding_family_proof_registration": (
                                None
                                if native_embedding_registration is None
                                else native_embedding_registration["registration"]
                            ),
                        }
                    )
                    if not is_inner:
                        _atomic_write_npz(
                            root / f"direct_numerical_outer_{int(scope['outer_fold']):03d}.npz",
                            fit_row_ids=np.asarray(fit_ids, dtype=np.int64),
                            heldout_row_ids=np.asarray(heldout_ids, dtype=np.int64),
                            x_train=np.asarray(bundle.x_train, dtype=np.float32),
                            x_heldout=np.asarray(bundle.x_test, dtype=np.float32),
                            w_train=np.asarray(bundle.w_train, dtype=np.float32),
                            w_heldout=np.asarray(bundle.w_test, dtype=np.float32),
                            x_feature_names=np.asarray(bundle.x_names, dtype=str),
                            w_feature_names=np.asarray(bundle.w_names, dtype=str),
                        )
                htr_snapshot.verify()
            for cumulative_scope in cumulative_schedule.scopes:
                cumulative_request = _cumulative_spent_request_from_modeling_data(
                    family=BOW_NUISANCE,
                    modeling_data=prepared.modeling_data,
                    request_sha256=prepared.request_sha256,
                    schedule_sha256=cumulative_schedule.schedule_sha256,
                    scope_id=cumulative_scope.scope_id,
                    outer_fold=cumulative_scope.outer_fold,
                    context_epoch=cumulative_scope.context_epoch,
                    provider_inner_fold=cumulative_scope.provider_inner_fold,
                    split_scope_fingerprint=cumulative_scope.split_fingerprint,
                    spent_row_ids=cumulative_scope.spent_row_ids,
                    sealed_row_ids=cumulative_scope.sealed_row_ids,
                    text_column=prepared.config.text_column,
                    treatment_column=prepared.config.treatment_column,
                    outcome_column=prepared.config.outcome_column,
                )
                registration, configurations, embedding_registration = (
                    self._run_legacy_cumulative_spent_scope(
                        root=root,
                        prepared=prepared,
                        request=cumulative_request,
                        htr_snapshot=htr_snapshot,
                        bow_models_dir=cumulative_bow_models_dir,
                        htr_models_dir=cumulative_htr_models_dir,
                        matched_pair_models_dir=cumulative_matched_pair_models_dir,
                        proofs_dir=cumulative_proofs_dir,
                        embedding_models_dir=cumulative_embedding_models_dir,
                        embedding_execution_dir=cumulative_embedding_execution_dir,
                        embedding_proofs_dir=cumulative_embedding_proofs_dir,
                    )
                )
                cumulative_registrations.append(registration)
                cumulative_embedding_registrations.append(embedding_registration)
                cumulative_expected_requests[cumulative_request.scope_id] = cumulative_request
                cumulative_expected_configurations[cumulative_request.scope_id] = configurations
                htr_snapshot.verify()
        finally:
            try:
                htr_snapshot.verify()
            finally:
                # PrivateHTRModelTreeSnapshot owns a TemporaryDirectory but has
                # no context-manager/close API in the current implementation.
                htr_snapshot._temporary_directory.cleanup()
        if _directory_tree_sha256(prepared.htr_model_path) != prepared.htr_model_sha256:
            raise RuntimeError("source HTR model tree changed during Stage 1")
        prepared.embedding_cache.identity()
        cumulative_index_registration = _write_legacy_cumulative_spent_native_index(
            component_root=root,
            index_path=Path("cumulative_legacy_native_family_proof_index.json"),
            request_sha256=prepared.request_sha256,
            schedule_sha256=cumulative_schedule.schedule_sha256,
            split_registry_content_sha256=prepared.registry_content_sha256,
            scope_registrations=cumulative_registrations,
        )
        _validate_legacy_cumulative_spent_native_index(
            component_root=root,
            index_registration=cumulative_index_registration,
            expected_requests=cumulative_expected_requests,
            expected_configuration_by_scope=cumulative_expected_configurations,
            request_sha256=prepared.request_sha256,
            schedule_sha256=cumulative_schedule.schedule_sha256,
            split_registry_content_sha256=prepared.registry_content_sha256,
            htr_model_path=prepared.htr_model_path,
            htr_model_sha256=prepared.htr_model_sha256,
            device=torch.device(prepared.options.device),
        )
        cumulative_embedding_index_registration = _write_cumulative_spent_embedding_index(
            component_root=root,
            index_path=Path("cumulative_embedding_native_family_proof_index.json"),
            request_sha256=prepared.request_sha256,
            schedule_sha256=cumulative_schedule.schedule_sha256,
            split_registry_content_sha256=prepared.registry_content_sha256,
            scope_registrations=cumulative_embedding_registrations,
        )
        _validate_cumulative_spent_embedding_index(
            component_root=root,
            index_registration=cumulative_embedding_index_registration,
            expected_requests=cumulative_expected_requests,
            request_sha256=prepared.request_sha256,
            schedule_sha256=cumulative_schedule.schedule_sha256,
            split_registry_content_sha256=prepared.registry_content_sha256,
            embedding_cache=prepared.embedding_cache,
        )
        handoff_rows.sort(
            key=lambda row: (
                int(row["outer_fold"]),
                0 if row["scope"] == "full_outer_train" else int(row["inner_fold"]),
            )
        )
        handoff_path = handoff_dir / "discovery_contexts.jsonl"
        _atomic_write_bytes(
            handoff_path,
            b"".join(
                (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
                for row in handoff_rows
            ),
        )
        _write_immutable_json(
            handoff_dir / "manifest.json",
            {
                "schema_version": "multi_model_agentic_discovery_handoff_v1",
                "handoff_file": handoff_path.name,
                "handoff_sha256": _sha256_file(handoff_path),
                "row_count": len(handoff_rows),
                "exact_scope_count": len(scopes),
                "split_registry_content_sha256": prepared.registry_content_sha256,
                "raw_evidence_sidecar_count": len(scope_index),
                "raw_evidence_sidecars_prompt_visible": False,
                "prompt_compactor_used": False,
                "full_outer_evidence_reused_for_inner": False,
                "heldout_labels_supplied_to_evidence_builder": False,
            },
        )
        bow_index_body = {
            "schema_version": STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
            "split_registry_content_sha256": prepared.registry_content_sha256,
            "registered_families": list(PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS),
            "exact_inner_scope_count": len(native_bow_proof_rows),
            "executable_checkpoint_files_retained": False,
            "scopes": native_bow_proof_rows,
        }
        _write_immutable_json(
            root / "bow_native_family_proof_index.json",
            {**bow_index_body, "content_sha256": _sha256_json(bow_index_body)},
        )
        htr_index_body = {
            "schema_version": STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
            "split_registry_content_sha256": prepared.registry_content_sha256,
            "registered_families": list(PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS),
            "exact_inner_scope_count": len(native_htr_proof_rows),
            "executable_checkpoint_files_retained": False,
            "scopes": native_htr_proof_rows,
        }
        _write_immutable_json(
            root / "htr_native_family_proof_index.json",
            {**htr_index_body, "content_sha256": _sha256_json(htr_index_body)},
        )
        matched_pair_index_body = {
            "schema_version": STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
            "split_registry_content_sha256": prepared.registry_content_sha256,
            "registered_families": list(PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS),
            "exact_inner_scope_count": len(native_matched_pair_proof_rows),
            "executable_checkpoint_files_retained": False,
            "scopes": native_matched_pair_proof_rows,
        }
        _write_immutable_json(
            root / "matched_pair_native_family_proof_index.json",
            {
                **matched_pair_index_body,
                "content_sha256": _sha256_json(matched_pair_index_body),
            },
        )
        embedding_index_body = {
            "schema_version": STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
            "split_registry_content_sha256": prepared.registry_content_sha256,
            "registered_families": list(PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS),
            "exact_inner_scope_count": len(native_embedding_proof_rows),
            "executable_checkpoint_files_retained": False,
            "scopes": native_embedding_proof_rows,
        }
        _write_immutable_json(
            root / "embedding_native_family_proof_index.json",
            {
                **embedding_index_body,
                "content_sha256": _sha256_json(embedding_index_body),
            },
        )
        if any(
            path.name.lower().endswith((".joblib", ".pkl", ".pickle", ".pt", ".pth", ".ckpt"))
            for path in root.rglob("*")
        ):
            raise RuntimeError("executable native serialization entered the legacy component")
        _write_immutable_json(
            root / "exact_scope_index.json",
            {
                "schema_version": STAGE1_SCOPE_INDEX_SCHEMA,
                "split_registry_content_sha256": prepared.registry_content_sha256,
                "registered_native_families": list(
                    (
                        *PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                        *PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                        *PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                        *PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                    )
                ),
                "native_bow_family_proof_index": _component_file_registration(
                    root / "bow_native_family_proof_index.json",
                    component_root=root,
                ),
                "native_htr_family_proof_index": _component_file_registration(
                    root / "htr_native_family_proof_index.json",
                    component_root=root,
                ),
                "native_matched_pair_family_proof_index": (
                    _component_file_registration(
                        root / "matched_pair_native_family_proof_index.json",
                        component_root=root,
                    )
                ),
                "native_embedding_family_proof_index": _component_file_registration(
                    root / "embedding_native_family_proof_index.json",
                    component_root=root,
                ),
                "native_cumulative_legacy_family_proof_index": copy.deepcopy(
                    dict(cumulative_index_registration)
                ),
                "native_cumulative_embedding_family_proof_index": copy.deepcopy(
                    dict(cumulative_embedding_index_registration)
                ),
                "scopes": scope_index,
            },
        )
        self._validate_legacy_scope_lineage(handoff_path, prepared)
        load_legacy_full_outer_evidence(handoff_path)

    @staticmethod
    def _validate_legacy_scope_lineage(
        handoff_path: Path, prepared: _PreparedBuild
    ) -> Mapping[str, Mapping[str, Any]]:
        component_root = handoff_path.parent.parent
        scope_index_path = component_root / "exact_scope_index.json"
        if not scope_index_path.is_file():
            raise ValueError("legacy component has no authenticated exact-scope index")
        scope_index = json.loads(scope_index_path.read_text(encoding="utf-8"))
        if (
            scope_index.get("schema_version") != STAGE1_SCOPE_INDEX_SCHEMA
            or scope_index.get("split_registry_content_sha256") != prepared.registry_content_sha256
            or not isinstance(scope_index.get("scopes"), list)
        ):
            raise ValueError("legacy exact-scope index has an invalid registry binding")
        indexed_scopes = {
            str(row.get("scope_id")): row
            for row in scope_index["scopes"]
            if isinstance(row, Mapping)
        }
        if len(indexed_scopes) != len(scope_index["scopes"]):
            raise ValueError("legacy exact-scope index contains duplicates or malformed rows")
        rows: dict[str, Mapping[str, Any]] = {}
        with handoff_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                outer_fold = int(row["outer_fold"])
                inner_fold = row.get("inner_fold")
                scope_id = (
                    f"outer_{outer_fold:03d}_inner_{int(inner_fold):03d}"
                    if inner_fold is not None
                    else f"outer_{outer_fold:03d}_full"
                )
                if scope_id in rows:
                    raise ValueError(f"duplicate legacy scope {scope_id}")
                if row.get("evidence_reused_from_fold_key") is not None:
                    raise ValueError("legacy exact-inner evidence was reused instead of refit")
                if not row.get("evidence_scope_fit_was_executed"):
                    raise ValueError("legacy evidence scope lacks an executed-fit attestation")
                if row.get("heldout_labels_supplied_to_evidence_builder") is not False:
                    raise ValueError("legacy evidence scope received held-out labels")
                if (
                    row.get("lossless_concept_catalog_projection") is not True
                    or row.get("prompt_compactor_used") is not False
                ):
                    raise ValueError("legacy evidence scope did not use the lossless projection")
                if row.get("split_registry_content_sha256") != prepared.registry_content_sha256:
                    raise ValueError("legacy evidence scope has the wrong split-registry binding")
                rows[scope_id] = row
        expected = {str(scope["scope_id"]): scope for scope in _registry_scopes(prepared.registry)}
        if set(rows) != set(expected) or set(indexed_scopes) != set(expected):
            raise ValueError("legacy handoff does not match the canonical scope registry")
        for scope_id, scope in expected.items():
            row = rows[scope_id]
            indexed = indexed_scopes[scope_id]
            is_inner = scope["inner_fold"] is not None
            expected_scope = "candidate_consistency_inner_train" if is_inner else "full_outer_train"
            expected_fold_key = (
                int(scope["outer_fold"]) * 1000 + int(scope["inner_fold"])
                if is_inner
                else int(scope["outer_fold"])
            )
            if (
                row.get("scope") != expected_scope
                or int(row.get("fold_key", 0)) != expected_fold_key
                or int(row.get("n_rows", 0)) != len(scope["fit_row_ids"])
                or (is_inner and int(row.get("heldout_rows", 0)) != len(scope["heldout_row_ids"]))
                or list(map(int, row.get("fit_row_ids") or ())) != scope["fit_row_ids"]
                or list(map(int, row.get("heldout_row_ids") or ())) != scope["heldout_row_ids"]
                or row.get("fit_row_fingerprint") != row_set_fingerprint(scope["fit_row_ids"])
                or row.get("heldout_row_fingerprint")
                != row_set_fingerprint(scope["heldout_row_ids"])
            ):
                raise ValueError(f"legacy scope lineage mismatch: {scope_id}")
            raw_registration = indexed.get("raw_evidence_sidecar")
            if not isinstance(raw_registration, Mapping):
                raise ValueError(f"legacy scope lacks a raw evidence sidecar: {scope_id}")
            raw_path = component_root / str(raw_registration.get("relative_path") or "")
            try:
                raw_path.resolve(strict=True).relative_to(component_root.resolve(strict=True))
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(f"legacy raw sidecar escapes its component: {scope_id}") from exc
            sidecar = _validate_raw_evidence_sidecar(
                raw_path,
                registration=raw_registration,
                scope=scope,
                split_registry_content_sha256=prepared.registry_content_sha256,
            )
            if row.get("raw_evidence_sidecar_sha256") != raw_registration.get(
                "sha256"
            ) or indexed.get("matched_pair_subproducer_proofs_sha256") != (
                sidecar.get("matched_pair_subproducer_proofs") or {}
            ).get(
                "content_sha256"
            ):
                raise ValueError(f"legacy scope raw sidecar linkage mismatch: {scope_id}")
            bow_registration = indexed.get("native_bow_family_proof_registration")
            if scope_index.get("native_bow_family_proof_index") is not None and (
                (is_inner and not isinstance(bow_registration, Mapping))
                or (not is_inner and bow_registration is not None)
            ):
                raise ValueError(f"legacy scope BoW native-proof linkage mismatch: {scope_id}")
            htr_registration = indexed.get("native_htr_family_proof_registration")
            if scope_index.get("native_htr_family_proof_index") is not None and (
                (is_inner and not isinstance(htr_registration, Mapping))
                or (not is_inner and htr_registration is not None)
            ):
                raise ValueError(f"legacy scope HTR native-proof linkage mismatch: {scope_id}")
            matched_pair_registration = indexed.get("native_matched_pair_family_proof_registration")
            if scope_index.get("native_matched_pair_family_proof_index") is not None and (
                (is_inner and not isinstance(matched_pair_registration, Mapping))
                or (not is_inner and matched_pair_registration is not None)
            ):
                raise ValueError(
                    f"legacy scope matched-pair native-proof linkage mismatch: {scope_id}"
                )
            embedding_registration = indexed.get("native_embedding_family_proof_registration")
            if scope_index.get("native_embedding_family_proof_index") is not None and (
                (is_inner and not isinstance(embedding_registration, Mapping))
                or (not is_inner and embedding_registration is not None)
            ):
                raise ValueError(
                    f"legacy scope embedding native-proof linkage mismatch: {scope_id}"
                )
        expected_inner = {
            scope_id: scope
            for scope_id, scope in expected.items()
            if scope["inner_fold"] is not None
        }
        bow_index_registration = scope_index.get("native_bow_family_proof_index")
        if bow_index_registration is not None:
            _validate_bow_native_family_proof_index(
                component_root=component_root,
                index_registration=bow_index_registration,
                expected_inner_scopes=expected_inner,
                split_registry_content_sha256=prepared.registry_content_sha256,
                modeling_data=prepared.modeling_data,
                text_column=prepared.config.text_column,
                treatment_column=prepared.config.treatment_column,
                outcome_column=prepared.config.outcome_column,
            )
        htr_index_registration = scope_index.get("native_htr_family_proof_index")
        if htr_index_registration is not None:
            _validate_htr_native_family_proof_index(
                component_root=component_root,
                index_registration=htr_index_registration,
                expected_inner_scopes=expected_inner,
                split_registry_content_sha256=prepared.registry_content_sha256,
                modeling_data=prepared.modeling_data,
                text_column=prepared.config.text_column,
                treatment_column=prepared.config.treatment_column,
                outcome_column=prepared.config.outcome_column,
                htr_model_path=prepared.htr_model_path,
                htr_model_sha256=prepared.htr_model_sha256,
                device=torch.device(prepared.options.device),
            )
        matched_pair_index_registration = scope_index.get("native_matched_pair_family_proof_index")
        if matched_pair_index_registration is not None:
            _validate_matched_pair_native_family_proof_index(
                component_root=component_root,
                index_registration=matched_pair_index_registration,
                expected_inner_scopes=expected_inner,
                split_registry_content_sha256=prepared.registry_content_sha256,
                modeling_data=prepared.modeling_data,
                text_column=prepared.config.text_column,
                treatment_column=prepared.config.treatment_column,
                outcome_column=prepared.config.outcome_column,
                htr_model_path=prepared.htr_model_path,
                htr_model_sha256=prepared.htr_model_sha256,
                device=torch.device(prepared.options.device),
            )
        embedding_index_registration = scope_index.get("native_embedding_family_proof_index")
        if embedding_index_registration is not None:
            _validate_embedding_native_family_proof_index(
                component_root=component_root,
                index_registration=embedding_index_registration,
                expected_inner_scopes=expected_inner,
                split_registry_content_sha256=prepared.registry_content_sha256,
                modeling_data=prepared.modeling_data,
                text_column=prepared.config.text_column,
                treatment_column=prepared.config.treatment_column,
                outcome_column=prepared.config.outcome_column,
                embedding_cache=prepared.embedding_cache,
            )
        cumulative_index_registration = scope_index.get(
            "native_cumulative_legacy_family_proof_index"
        )
        if cumulative_index_registration is not None:
            from .production_stage1_hierarchy_handoff import (
                CanonicalHierarchySpentSchedule,
            )

            exact_registry = _canonical_exact_registry_from_wrapper(prepared.registry)
            review_rounds = int(exact_registry.inner_fold_count) - 3
            if review_rounds < 1:
                raise ValueError(
                    "cumulative legacy proof index has no valid canonical review schedule"
                )
            schedule = CanonicalHierarchySpentSchedule.build(
                registry=exact_registry,
                review_rounds=review_rounds,
            )
            expected_requests: dict[str, CumulativeSpentStage1FamilyRequest] = {}
            expected_configurations: dict[
                str,
                Mapping[str, Mapping[str, Any]],
            ] = {}
            for cumulative_scope in schedule.scopes:
                request = _cumulative_spent_request_from_modeling_data(
                    family=BOW_NUISANCE,
                    modeling_data=prepared.modeling_data,
                    request_sha256=prepared.request_sha256,
                    schedule_sha256=schedule.schedule_sha256,
                    scope_id=cumulative_scope.scope_id,
                    outer_fold=cumulative_scope.outer_fold,
                    context_epoch=cumulative_scope.context_epoch,
                    provider_inner_fold=cumulative_scope.provider_inner_fold,
                    split_scope_fingerprint=cumulative_scope.split_fingerprint,
                    spent_row_ids=cumulative_scope.spent_row_ids,
                    sealed_row_ids=cumulative_scope.sealed_row_ids,
                    text_column=prepared.config.text_column,
                    treatment_column=prepared.config.treatment_column,
                    outcome_column=prepared.config.outcome_column,
                )
                expected_requests[request.scope_id] = request
                expected_configurations[request.scope_id] = (
                    _cumulative_legacy_configuration_by_family(
                        config=prepared.config,
                        scope_id=request.scope_id,
                        split_registry_content_sha256=prepared.registry_content_sha256,
                        htr_model_tree_sha256=prepared.htr_model_sha256,
                        seed=int(prepared.options.seed),
                    )
                )
            _validate_legacy_cumulative_spent_native_index(
                component_root=component_root,
                index_registration=cumulative_index_registration,
                expected_requests=expected_requests,
                expected_configuration_by_scope=expected_configurations,
                request_sha256=prepared.request_sha256,
                schedule_sha256=schedule.schedule_sha256,
                split_registry_content_sha256=prepared.registry_content_sha256,
                htr_model_path=prepared.htr_model_path,
                htr_model_sha256=prepared.htr_model_sha256,
                device=torch.device(prepared.options.device),
            )
            cumulative_embedding_index = scope_index.get(
                "native_cumulative_embedding_family_proof_index"
            )
            if not isinstance(cumulative_embedding_index, Mapping):
                raise ValueError("legacy component lacks cumulative embedding proof index")
            _validate_cumulative_spent_embedding_index(
                component_root=component_root,
                index_registration=cumulative_embedding_index,
                expected_requests=expected_requests,
                request_sha256=prepared.request_sha256,
                schedule_sha256=schedule.schedule_sha256,
                split_registry_content_sha256=prepared.registry_content_sha256,
                embedding_cache=prepared.embedding_cache,
            )
        return rows

    def _run_tfidf_component(self, root: Path, output: Path, prepared: _PreparedBuild) -> None:
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise RuntimeError(
                "TF-IDF component must start empty; partial checkpoint reuse is disabled"
            )
        config = copy.deepcopy(prepared.config)
        nn_config = config.architecture.multi_model_forest
        nn_config.feature_discovery_methods = ["bow", "tfidf_topic_contrast"]
        nn_config.htr_evidence_enabled = False
        nn_config.matched_pair_uplift_enabled = False
        nn_config.matched_pair_bow_enabled = False
        nn_config.matched_pair_htr_enabled = False
        nn_config.embedding_contrast.enabled = False
        nn_config.split_registry_path = str((output / "split_registry.json").resolve())
        nn_config.cpus_total = int(prepared.options.tfidf_workers)
        backend = prepared.options.tfidf_parallel_backend
        nn_config.outer_parallel_backend = "multiprocessing" if backend == "fork" else backend
        config.architecture.multi_model_agentic_forest = nn_config
        handoff_path = root / "handoff" / "discovery_contexts.jsonl"
        prediction_path = root / "primary_predictions.parquet"
        run_tfidf_topic_stage1(
            dataset=prepared.modeling_data,
            config=config,
            output_path=prediction_path,
            artifact_dir=root,
            handoff_path=handoff_path,
        )
        if not tfidf_topic_stage1_cache_is_valid(
            dataset=prepared.modeling_data,
            config=config,
            output_path=prediction_path,
            handoff_path=handoff_path,
        ):
            raise RuntimeError("TF-IDF Stage 1 did not produce an authenticated complete cache")
        validate_tfidf_topic_stage2_handoff(
            dataset=prepared.modeling_data,
            config=config,
            handoff_path=handoff_path,
        )
        load_resealed_tfidf_handoff(
            handoff_path,
            dataset_row_count=len(prepared.modeling_data),
            require_registry_seal=True,
        )
        self._register_tfidf_component_native_family_proofs(
            root=root,
            handoff_path=handoff_path,
            prepared=prepared,
        )
        cumulative_artifacts = root / "cumulative_native_tfidf_artifacts"
        cumulative_records = root / "cumulative_tfidf_execution_records"
        cumulative_proofs = root / "cumulative_tfidf_family_proofs"
        cumulative_artifacts.mkdir(parents=True, exist_ok=False)
        cumulative_records.mkdir(parents=True, exist_ok=False)
        cumulative_proofs.mkdir(parents=True, exist_ok=False)
        schedule = _canonical_cumulative_spent_schedule(prepared.registry)
        registrations: list[Mapping[str, Any]] = []
        expected_requests: dict[str, CumulativeSpentStage1FamilyRequest] = {}
        for scope in schedule.scopes:
            reference = _cumulative_spent_request_from_modeling_data(
                family=TFIDF_TOPICS,
                modeling_data=prepared.modeling_data,
                request_sha256=prepared.request_sha256,
                schedule_sha256=schedule.schedule_sha256,
                scope_id=scope.scope_id,
                outer_fold=scope.outer_fold,
                context_epoch=scope.context_epoch,
                provider_inner_fold=scope.provider_inner_fold,
                split_scope_fingerprint=scope.split_fingerprint,
                spent_row_ids=scope.spent_row_ids,
                sealed_row_ids=scope.sealed_row_ids,
                text_column=prepared.config.text_column,
                treatment_column=prepared.config.treatment_column,
                outcome_column=prepared.config.outcome_column,
            )
            requests = {
                family: _cumulative_request_for_family(reference, family=family)
                for family in PRODUCTION_CUMULATIVE_TFIDF_NATIVE_FAMILY_ADAPTERS
            }
            canary = CumulativeSpentReplayCanary.from_request(reference)
            emissions = emit_cumulative_spent_tfidf_capture(
                requests=requests,
                replay_canary=canary,
                config=prepared.config,
                artifact_dir=cumulative_artifacts / scope.scope_id,
                execution_record_dir=cumulative_records / scope.scope_id,
            )
            registrations.append(
                _register_cumulative_spent_remaining_scope(
                    component_root=root,
                    proof_directory=cumulative_proofs / scope.scope_id,
                    requests=requests,
                    replay_canary=canary,
                    emissions=emissions,
                    families=PRODUCTION_CUMULATIVE_TFIDF_NATIVE_FAMILY_ADAPTERS,
                )
            )
            expected_requests[scope.scope_id] = reference
        index_registration = _write_cumulative_spent_remaining_index(
            component_root=root,
            index_path=Path("cumulative_tfidf_native_family_proof_index.json"),
            index_schema=STAGE1_CUMULATIVE_TFIDF_NATIVE_INDEX_SCHEMA,
            families=PRODUCTION_CUMULATIVE_TFIDF_NATIVE_FAMILY_ADAPTERS,
            request_sha256=prepared.request_sha256,
            schedule_sha256=schedule.schedule_sha256,
            split_registry_content_sha256=prepared.registry_content_sha256,
            scope_registrations=registrations,
        )
        _validate_cumulative_spent_tfidf_index(
            component_root=root,
            index_registration=index_registration,
            expected_requests=expected_requests,
            request_sha256=prepared.request_sha256,
            schedule_sha256=schedule.schedule_sha256,
            split_registry_content_sha256=prepared.registry_content_sha256,
            config=prepared.config,
        )

    def _register_tfidf_component_native_family_proofs(
        self,
        *,
        root: Path,
        handoff_path: Path,
        prepared: _PreparedBuild,
    ) -> Mapping[str, Any]:
        """Persist genuine topic/orphan proofs for every canonical inner scope."""

        rows = self._raw_tfidf_scope_rows(handoff_path)
        exact_registry = _canonical_exact_registry_from_wrapper(prepared.registry)
        expected_scopes = {
            str(scope["scope_id"]): scope
            for scope in _registry_scopes(prepared.registry)
            if scope["inner_fold"] is not None
        }
        registrations: list[dict[str, Any]] = []
        for scope_id, scope in expected_scopes.items():
            try:
                row = rows[scope_id]
            except KeyError as exc:
                raise ValueError(f"TF-IDF proof registration lacks scope {scope_id}") from exc
            outer_fold = int(scope["outer_fold"])
            inner_fold = int(scope["inner_fold"])
            fit_row_ids = tuple(map(int, scope["fit_row_ids"]))
            heldout_row_ids = tuple(map(int, scope["heldout_row_ids"]))
            if (
                int(row.get("outer_fold", 0)) != outer_fold
                or int(row.get("inner_fold", 0)) != inner_fold
                or tuple(map(int, row.get("fit_row_ids") or ())) != fit_row_ids
                or tuple(map(int, row.get("heldout_row_ids") or ())) != heldout_row_ids
            ):
                raise ValueError(f"TF-IDF proof registration scope mismatch: {scope_id}")
            native_discovery = copy.deepcopy(row.get("discovery") or {})
            catalog_discovery = _catalog_ready_tfidf_discovery(native_discovery)
            provenance = FoldEvidenceProvenance(
                outer_fold=outer_fold,
                train_row_ids=fit_row_ids,
                heldout_row_ids=heldout_row_ids,
                scope="inner_train",
                inner_fold=inner_fold,
                artifact_id=f"production-stage1-tfidf-native-proof-{scope_id}",
            )
            catalog = build_role_neutral_evidence_catalog(
                (
                    FoldEvidenceInput(
                        TFIDF_TOPIC_SOURCE,
                        {
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "scope": "inner_train",
                            "discovery": catalog_discovery,
                        },
                        provenance,
                    ),
                ),
                require_all_source_kinds=False,
                require_all_architecture_families=False,
                require_upstream_completeness=False,
            )
            missing = [
                family
                for family in PRODUCTION_TFIDF_REGISTERED_NATIVE_FAMILY_ADAPTERS
                if not catalog.family_atoms(family)
            ]
            if missing:
                raise RuntimeError(
                    f"TF-IDF native scope {scope_id} lacks registered family evidence: "
                    + ", ".join(missing)
                )
            split = exact_registry.inner_split(outer_fold, inner_fold)
            if split.fit_row_ids != fit_row_ids or split.heldout_row_ids != heldout_row_ids:
                raise RuntimeError("TF-IDF native proof changed the canonical exact-inner split")
            configuration = {
                "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
                "scope_id": scope_id,
                "text_column": prepared.config.text_column,
                "tfidf_nested_calibration_folds": int(
                    prepared.config.architecture.multi_model_forest.tfidf_nested_calibration_folds
                ),
                "score_selection_label_policy": "nested_fit_calibration",
                "stage1_config_hash": native_discovery.get("stage1_config_hash"),
                "topic_configuration_hash": native_discovery.get("config_hash"),
            }
            registration = _register_tfidf_native_family_proofs(
                component_root=root,
                proof_directory=Path("native_family_proofs") / scope_id,
                scope_id=scope_id,
                catalog=catalog,
                tfidf_discovery=native_discovery,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                fit_row_ids=fit_row_ids,
                heldout_row_ids=heldout_row_ids,
                fit_treatment=prepared.modeling_data.iloc[list(fit_row_ids)][
                    prepared.config.treatment_column
                ].to_numpy(dtype=float),
                fit_outcome=prepared.modeling_data.iloc[list(fit_row_ids)][
                    prepared.config.outcome_column
                ].to_numpy(dtype=float),
                split_scope_fingerprint=split.scope_fingerprint,
                data_projection_sha256=_exact_inner_projection_sha256(
                    modeling_data=prepared.modeling_data,
                    config=prepared.config,
                    fit_row_ids=fit_row_ids,
                    heldout_row_ids=heldout_row_ids,
                ),
                configuration=configuration,
            )
            registrations.append(
                {
                    "scope_id": scope_id,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "registered_families": list(PRODUCTION_TFIDF_REGISTERED_NATIVE_FAMILY_ADAPTERS),
                    "content_sha256": registration["content_sha256"],
                    "registration": registration["registration"],
                }
            )
        index_body = {
            "schema_version": STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
            "split_registry_content_sha256": prepared.registry_content_sha256,
            "registered_families": list(PRODUCTION_TFIDF_REGISTERED_NATIVE_FAMILY_ADAPTERS),
            "exact_inner_scope_count": len(registrations),
            "scopes": registrations,
        }
        index = {**index_body, "content_sha256": _sha256_json(index_body)}
        _write_immutable_json(root / "native_family_proof_index.json", index)
        return index

    def _run_query_component(self, root: Path, output: Path, prepared: _PreparedBuild) -> None:
        root.mkdir(parents=True, exist_ok=False)
        artifacts_dir = root / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=False)
        native_models_dir = root / "native_models"
        native_models_dir.mkdir(parents=True, exist_ok=False)
        rows: list[dict[str, Any]] = []
        native_proof_rows: list[dict[str, Any]] = []
        indexed = prepared.modeling_data.copy()
        indexed["_oci_row_id"] = np.arange(len(indexed), dtype=np.int64)
        exact_registry = _canonical_exact_registry_from_wrapper(prepared.registry)
        # Executable joblib checkpoints are a same-process optimization only.
        # They are deliberately kept outside the sealed component and deleted
        # after safe JSON evidence and non-executable native state have been
        # materialized. No joblib bytes enter the component.
        with tempfile.TemporaryDirectory(prefix="production-stage1-neural-query-") as cache_dir:
            service = ContextFitNeuralQueryService(
                cache_dir=Path(cache_dir) / "executable_cache",
                dataset_path=prepared.options.dataset_path,
                text_column=prepared.config.text_column,
                embedding_cache=prepared.embedding_cache,
                stage1_config_path=output / "stage1_config.json",
                query_config=prepared.query_config,
                nuisance_folds=int(prepared.options.query_nuisance_folds),
                devices=prepared.options.query_devices,
                seed=int(prepared.options.seed),
                outcome_type=prepared.config.outcome_type,
            )
            backend = NeuralQueryContextBackend(service)
            for scope in _registry_scopes(prepared.registry):
                fit_ids = tuple(map(int, scope["fit_row_ids"]))
                heldout_ids = tuple(map(int, scope["heldout_row_ids"]))
                fit = indexed.iloc[list(fit_ids)]
                texts = tuple(fit[prepared.config.text_column].tolist())
                discovery, cache_key = service.discovery_for_context(
                    outer_fold=int(scope["outer_fold"]),
                    context_row_ids=fit_ids,
                    context_texts=texts,
                    context_treatment=fit[prepared.config.treatment_column].to_numpy(dtype=float),
                    context_outcome=fit[prepared.config.outcome_column].to_numpy(dtype=float),
                )
                model_artifact_dir = native_models_dir / str(scope["scope_id"])
                model_artifact_dir.mkdir(parents=True, exist_ok=False)
                owned_snapshot = service.write_owned_discovery_snapshot(
                    cache_key=cache_key,
                    output_dir=model_artifact_dir / "owned_snapshot",
                )
                evidence = service.safe_evidence(
                    discovery=discovery,
                    context_row_ids=fit_ids,
                    context_texts=texts,
                    device_offset=int(scope["inner_fold"] or 0),
                )
                if not evidence:
                    raise RuntimeError(
                        f"neural-query scope produced no evidence: {scope['scope_id']}"
                    )
                heldout_projection = indexed.iloc[list(heldout_ids)][
                    ["_oci_row_id", prepared.config.text_column]
                ].copy()
                if tuple(map(int, heldout_projection["_oci_row_id"].tolist())) != heldout_ids:
                    raise RuntimeError(
                        "neural-query heldout projection changed canonical row order"
                    )
                heldout_texts = tuple(heldout_projection[prepared.config.text_column].tolist())
                prediction = backend.fit_predict(
                    outer_fold=int(scope["outer_fold"]),
                    context_row_ids=fit_ids,
                    context_texts=texts,
                    context_treatment=fit[prepared.config.treatment_column].to_numpy(dtype=float),
                    context_outcome=fit[prepared.config.outcome_column].to_numpy(dtype=float),
                    gate_row_ids=heldout_ids,
                    gate_texts=heldout_texts,
                    work_dir=model_artifact_dir,
                )
                data_projection_sha256 = _exact_inner_projection_sha256(
                    modeling_data=prepared.modeling_data,
                    config=prepared.config,
                    fit_row_ids=fit_ids,
                    heldout_row_ids=heldout_ids,
                )
                if scope["inner_fold"] is not None:
                    split = exact_registry.inner_split(
                        int(scope["outer_fold"]),
                        int(scope["inner_fold"]),
                    )
                    if split.fit_row_ids != fit_ids or split.heldout_row_ids != heldout_ids:
                        raise RuntimeError(
                            "neural-query native proof changed the canonical exact-inner split"
                        )
                    split_scope_fingerprint = split.scope_fingerprint
                else:
                    split_scope_fingerprint = FoldEvidenceProvenance(
                        outer_fold=int(scope["outer_fold"]),
                        train_row_ids=fit_ids,
                        heldout_row_ids=heldout_ids,
                        scope="outer_train",
                        artifact_id=f"production-stage1-query-{scope['scope_id']}",
                    ).split_fingerprint
                moment_metadata = _write_neural_query_moment_artifact(
                    model_artifact_dir,
                    scope_id=str(scope["scope_id"]),
                    outer_fold=int(scope["outer_fold"]),
                    inner_fold=(None if scope["inner_fold"] is None else int(scope["inner_fold"])),
                    fit_row_ids=fit_ids,
                    heldout_row_ids=heldout_ids,
                    split_scope_fingerprint=split_scope_fingerprint,
                    data_projection_sha256=data_projection_sha256,
                    query_cache_key=cache_key,
                    owned_snapshot_metadata=owned_snapshot,
                    text_column=prepared.config.text_column,
                    prediction=prediction,
                )
                model_registration = _component_native_artifact_registration(
                    model_artifact_dir,
                    component_root=root,
                )
                payload = {
                    "schema_version": STAGE1_QUERY_ARTIFACT_SCHEMA,
                    "source_kind": NEURAL_QUERY_SOURCE,
                    "source_family": NEURAL_QUERY_MOMENTS,
                    "adapter_mode": "authenticated_neural_query_artifact",
                    "scope_id": str(scope["scope_id"]),
                    "outer_fold": int(scope["outer_fold"]),
                    "scope": ("outer_train" if scope["inner_fold"] is None else "inner_train"),
                    "fit_row_ids": list(fit_ids),
                    "heldout_row_ids": list(heldout_ids),
                    "fit_row_fingerprint": row_set_fingerprint(fit_ids),
                    "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
                    "split_registry_content_sha256": prepared.registry_content_sha256,
                    "query_cache_key": cache_key,
                    "heldout_labels_supplied": False,
                    "heldout_columns_read": ["_oci_row_id", prepared.config.text_column],
                    "native_model_artifact": {
                        "relative_path": model_registration["relative_path"],
                        "sha256": model_registration["sha256"],
                    },
                    "heldout_moment_artifact": {
                        "relative_path": (model_artifact_dir / "heldout_moments.npz")
                        .relative_to(root)
                        .as_posix(),
                        "sha256": moment_metadata["arrays_sha256"],
                        "content_sha256": moment_metadata["content_sha256"],
                    },
                    "query_evidence": evidence,
                }
                if scope["inner_fold"] is not None:
                    payload["inner_fold"] = int(scope["inner_fold"])
                path = artifacts_dir / f"{scope['scope_id']}.json"
                _write_immutable_json(path, payload)
                native_registration: Mapping[str, Any] | None = None
                if scope["inner_fold"] is not None:
                    provenance = FoldEvidenceProvenance(
                        outer_fold=int(scope["outer_fold"]),
                        train_row_ids=fit_ids,
                        heldout_row_ids=heldout_ids,
                        scope="inner_train",
                        inner_fold=int(scope["inner_fold"]),
                        artifact_id=(
                            f"production-stage1-neural-query-native-proof-{scope['scope_id']}"
                        ),
                    )
                    catalog = build_role_neutral_evidence_catalog(
                        (
                            FoldEvidenceInput(
                                NEURAL_QUERY_SOURCE,
                                payload,
                                provenance,
                            ),
                        ),
                        require_all_source_kinds=False,
                        require_all_architecture_families=False,
                        require_upstream_completeness=False,
                    )
                    if not catalog.family_atoms(NEURAL_QUERY_MOMENTS):
                        raise RuntimeError(
                            f"neural-query native scope {scope['scope_id']} lacks concept evidence"
                        )
                    configuration = {
                        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
                        "scope_id": str(scope["scope_id"]),
                        "text_column": prepared.config.text_column,
                        "query_config": asdict(prepared.query_config),
                        "query_nuisance_folds": int(prepared.options.query_nuisance_folds),
                        "query_devices": list(prepared.options.query_devices),
                        "seed": int(prepared.options.seed),
                        "outcome_type": prepared.config.outcome_type,
                        "split_registry_content_sha256": prepared.registry_content_sha256,
                        "service_identity_sha256": owned_snapshot["service_identity_sha256"],
                        "heldout_label_policy": "id_and_text_only",
                    }
                    native_registration = _register_neural_query_native_family_proof(
                        component_root=root,
                        proof_directory=Path("native_family_proofs") / str(scope["scope_id"]),
                        scope_id=str(scope["scope_id"]),
                        catalog=catalog,
                        query_artifact_path=path,
                        model_artifact_path=model_artifact_dir,
                        outer_fold=int(scope["outer_fold"]),
                        inner_fold=int(scope["inner_fold"]),
                        fit_row_ids=fit_ids,
                        heldout_row_ids=heldout_ids,
                        fit_treatment=fit[prepared.config.treatment_column].to_numpy(dtype=float),
                        fit_outcome=fit[prepared.config.outcome_column].to_numpy(dtype=float),
                        split_scope_fingerprint=split_scope_fingerprint,
                        data_projection_sha256=data_projection_sha256,
                        configuration=configuration,
                    )
                    native_proof_rows.append(
                        {
                            "scope_id": str(scope["scope_id"]),
                            "outer_fold": int(scope["outer_fold"]),
                            "inner_fold": int(scope["inner_fold"]),
                            "registered_families": list(
                                PRODUCTION_QUERY_REGISTERED_NATIVE_FAMILY_ADAPTERS
                            ),
                            "content_sha256": native_registration["content_sha256"],
                            "registration": native_registration["registration"],
                        }
                    )
                rows.append(
                    {
                        "scope_id": scope["scope_id"],
                        "outer_fold": scope["outer_fold"],
                        "inner_fold": scope["inner_fold"],
                        "path": path.relative_to(root).as_posix(),
                        "sha256": _sha256_file(path),
                        "fit_row_fingerprint": payload["fit_row_fingerprint"],
                        "heldout_row_fingerprint": payload["heldout_row_fingerprint"],
                        "query_count": len(evidence),
                        "heldout_moment_feature_count": int(moment_metadata["feature_count"]),
                        "heldout_labels_supplied": False,
                        "native_model_artifact": model_registration,
                        "owned_snapshot_metadata": _component_file_registration(
                            model_artifact_dir / "owned_snapshot" / "metadata.json",
                            component_root=root,
                        ),
                        "owned_snapshot_arrays": _component_file_registration(
                            model_artifact_dir / "owned_snapshot" / "arrays.npz",
                            component_root=root,
                        ),
                        "heldout_moment_metadata": _component_file_registration(
                            model_artifact_dir / "heldout_moments.metadata.json",
                            component_root=root,
                        ),
                        "heldout_moment_arrays": _component_file_registration(
                            model_artifact_dir / "heldout_moments.npz",
                            component_root=root,
                        ),
                        "native_family_proof_registration": (
                            None
                            if native_registration is None
                            else native_registration["registration"]
                        ),
                    }
                )
            service.identity()
            backend.identity()
        cumulative_artifacts = root / "cumulative_native_query_artifacts"
        cumulative_records = root / "cumulative_query_execution_records"
        cumulative_proofs = root / "cumulative_query_family_proofs"
        cumulative_artifacts.mkdir(parents=True, exist_ok=False)
        cumulative_records.mkdir(parents=True, exist_ok=False)
        cumulative_proofs.mkdir(parents=True, exist_ok=False)
        schedule = _canonical_cumulative_spent_schedule(prepared.registry)
        cumulative_registrations: list[Mapping[str, Any]] = []
        cumulative_requests: dict[str, CumulativeSpentStage1FamilyRequest] = {}
        with tempfile.TemporaryDirectory(
            prefix="production-stage1-cumulative-neural-query-"
        ) as cumulative_cache_dir:
            cumulative_service = ContextFitNeuralQueryService(
                cache_dir=Path(cumulative_cache_dir) / "executable_cache",
                dataset_path=prepared.options.dataset_path,
                text_column=prepared.config.text_column,
                embedding_cache=prepared.embedding_cache,
                stage1_config_path=output / "stage1_config.json",
                query_config=prepared.query_config,
                nuisance_folds=int(prepared.options.query_nuisance_folds),
                devices=prepared.options.query_devices,
                seed=int(prepared.options.seed),
                outcome_type=prepared.config.outcome_type,
            )
            cumulative_service_identity = cumulative_service.identity()
            for scope in schedule.scopes:
                request = _cumulative_spent_request_from_modeling_data(
                    family=NEURAL_QUERY_MOMENTS,
                    modeling_data=prepared.modeling_data,
                    request_sha256=prepared.request_sha256,
                    schedule_sha256=schedule.schedule_sha256,
                    scope_id=scope.scope_id,
                    outer_fold=scope.outer_fold,
                    context_epoch=scope.context_epoch,
                    provider_inner_fold=scope.provider_inner_fold,
                    split_scope_fingerprint=scope.split_fingerprint,
                    spent_row_ids=scope.spent_row_ids,
                    sealed_row_ids=scope.sealed_row_ids,
                    text_column=prepared.config.text_column,
                    treatment_column=prepared.config.treatment_column,
                    outcome_column=prepared.config.outcome_column,
                )
                canary = CumulativeSpentReplayCanary.from_request(request)
                emission = emit_cumulative_spent_neural_query_capture(
                    request=request,
                    replay_canary=canary,
                    service=cumulative_service,
                    artifact_dir=cumulative_artifacts / scope.scope_id,
                    execution_record_dir=cumulative_records / scope.scope_id,
                )
                cumulative_registrations.append(
                    _register_cumulative_spent_remaining_scope(
                        component_root=root,
                        proof_directory=cumulative_proofs / scope.scope_id,
                        requests={NEURAL_QUERY_MOMENTS: request},
                        replay_canary=canary,
                        emissions={NEURAL_QUERY_MOMENTS: emission},
                        families=PRODUCTION_CUMULATIVE_QUERY_NATIVE_FAMILY_ADAPTERS,
                    )
                )
                cumulative_requests[scope.scope_id] = request
            if cumulative_service.identity() != cumulative_service_identity:
                raise RuntimeError("cumulative neural-query service identity changed")
        cumulative_index_registration = _write_cumulative_spent_remaining_index(
            component_root=root,
            index_path=Path("cumulative_query_native_family_proof_index.json"),
            index_schema=STAGE1_CUMULATIVE_QUERY_NATIVE_INDEX_SCHEMA,
            families=PRODUCTION_CUMULATIVE_QUERY_NATIVE_FAMILY_ADAPTERS,
            request_sha256=prepared.request_sha256,
            schedule_sha256=schedule.schedule_sha256,
            split_registry_content_sha256=prepared.registry_content_sha256,
            scope_registrations=cumulative_registrations,
        )
        _validate_cumulative_spent_query_index(
            component_root=root,
            index_registration=cumulative_index_registration,
            expected_requests=cumulative_requests,
            request_sha256=prepared.request_sha256,
            schedule_sha256=schedule.schedule_sha256,
            split_registry_content_sha256=prepared.registry_content_sha256,
            service_identity=cumulative_service_identity,
        )
        if any(path.suffix == ".joblib" for path in root.rglob("*")):
            raise RuntimeError("executable neural-query checkpoint entered the sealed component")
        proof_index_body = {
            "schema_version": STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
            "split_registry_content_sha256": prepared.registry_content_sha256,
            "registered_families": list(PRODUCTION_QUERY_REGISTERED_NATIVE_FAMILY_ADAPTERS),
            "exact_inner_scope_count": len(native_proof_rows),
            "scopes": native_proof_rows,
            "executable_checkpoint_files_retained": False,
        }
        proof_index = {
            **proof_index_body,
            "content_sha256": _sha256_json(proof_index_body),
        }
        _write_immutable_json(root / "native_family_proof_index.json", proof_index)
        _write_immutable_json(
            root / "query_artifact_index.json",
            {
                "schema_version": STAGE1_SCOPE_INDEX_SCHEMA,
                "split_registry_content_sha256": prepared.registry_content_sha256,
                "registered_native_families": list(
                    PRODUCTION_QUERY_REGISTERED_NATIVE_FAMILY_ADAPTERS
                ),
                "native_family_proof_index": _component_file_registration(
                    root / "native_family_proof_index.json",
                    component_root=root,
                ),
                "heldout_labels_supplied": False,
                "executable_checkpoint_files_retained": False,
                "scopes": rows,
            },
        )

    @staticmethod
    def _raw_tfidf_scope_rows(path: Path) -> Mapping[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                outer = int(row["outer_fold"])
                inner = row.get("inner_fold")
                scope_id = (
                    f"outer_{outer:03d}_inner_{int(inner):03d}"
                    if inner is not None
                    else f"outer_{outer:03d}_full"
                )
                if scope_id in result:
                    raise ValueError(f"duplicate TF-IDF Stage 1 scope: {scope_id}")
                result[scope_id] = row
        return result

    def _load_cumulative_all_ten_producers(
        self,
        *,
        legacy_root: Path,
        tfidf_root: Path,
        query_root: Path,
        prepared: _PreparedBuild,
    ) -> Mapping[str, Any]:
        """Reload all component indexes into ten live producers per hierarchy scope."""

        schedule = _canonical_cumulative_spent_schedule(prepared.registry)
        base_requests: dict[str, CumulativeSpentStage1FamilyRequest] = {}
        legacy_configurations: dict[str, Mapping[str, Mapping[str, Any]]] = {}
        for scope in schedule.scopes:
            request = _cumulative_spent_request_from_modeling_data(
                family=BOW_NUISANCE,
                modeling_data=prepared.modeling_data,
                request_sha256=prepared.request_sha256,
                schedule_sha256=schedule.schedule_sha256,
                scope_id=scope.scope_id,
                outer_fold=scope.outer_fold,
                context_epoch=scope.context_epoch,
                provider_inner_fold=scope.provider_inner_fold,
                split_scope_fingerprint=scope.split_fingerprint,
                spent_row_ids=scope.spent_row_ids,
                sealed_row_ids=scope.sealed_row_ids,
                text_column=prepared.config.text_column,
                treatment_column=prepared.config.treatment_column,
                outcome_column=prepared.config.outcome_column,
            )
            base_requests[scope.scope_id] = request
            legacy_configurations[scope.scope_id] = _cumulative_legacy_configuration_by_family(
                config=prepared.config,
                scope_id=scope.scope_id,
                split_registry_content_sha256=prepared.registry_content_sha256,
                htr_model_tree_sha256=prepared.htr_model_sha256,
                seed=int(prepared.options.seed),
            )
        legacy_scope_index = _read_json_object_reject_duplicates(
            legacy_root / "exact_scope_index.json",
            field_name="legacy exact-scope index",
        )
        legacy_index_registration = legacy_scope_index.get(
            "native_cumulative_legacy_family_proof_index"
        )
        embedding_index_registration = legacy_scope_index.get(
            "native_cumulative_embedding_family_proof_index"
        )
        if not isinstance(legacy_index_registration, Mapping) or not isinstance(
            embedding_index_registration,
            Mapping,
        ):
            raise ValueError("legacy component lacks both cumulative native indexes")
        _legacy_index, legacy_producers = _validate_legacy_cumulative_spent_native_index(
            component_root=legacy_root,
            index_registration=legacy_index_registration,
            expected_requests=base_requests,
            expected_configuration_by_scope=legacy_configurations,
            request_sha256=prepared.request_sha256,
            schedule_sha256=schedule.schedule_sha256,
            split_registry_content_sha256=prepared.registry_content_sha256,
            htr_model_path=prepared.htr_model_path,
            htr_model_sha256=prepared.htr_model_sha256,
            device=torch.device(prepared.options.device),
        )
        _embedding_index, embedding_producers = _validate_cumulative_spent_embedding_index(
            component_root=legacy_root,
            index_registration=embedding_index_registration,
            expected_requests=base_requests,
            request_sha256=prepared.request_sha256,
            schedule_sha256=schedule.schedule_sha256,
            split_registry_content_sha256=prepared.registry_content_sha256,
            embedding_cache=prepared.embedding_cache,
        )
        tfidf_index_registration = _component_file_registration(
            tfidf_root / "cumulative_tfidf_native_family_proof_index.json",
            component_root=tfidf_root,
        )
        _tfidf_index, tfidf_producers = _validate_cumulative_spent_tfidf_index(
            component_root=tfidf_root,
            index_registration=tfidf_index_registration,
            expected_requests=base_requests,
            request_sha256=prepared.request_sha256,
            schedule_sha256=schedule.schedule_sha256,
            split_registry_content_sha256=prepared.registry_content_sha256,
            config=prepared.config,
        )
        query_index_registration = _component_file_registration(
            query_root / "cumulative_query_native_family_proof_index.json",
            component_root=query_root,
        )
        with tempfile.TemporaryDirectory(
            prefix="production-stage1-cumulative-query-reload-"
        ) as cache_dir:
            service = ContextFitNeuralQueryService(
                cache_dir=Path(cache_dir) / "executable_cache",
                dataset_path=prepared.options.dataset_path,
                text_column=prepared.config.text_column,
                embedding_cache=prepared.embedding_cache,
                stage1_config_path=legacy_root.parent / "stage1_config.json",
                query_config=prepared.query_config,
                nuisance_folds=int(prepared.options.query_nuisance_folds),
                devices=prepared.options.query_devices,
                seed=int(prepared.options.seed),
                outcome_type=prepared.config.outcome_type,
            )
            service_identity = service.identity()
            _query_index, query_producers = _validate_cumulative_spent_query_index(
                component_root=query_root,
                index_registration=query_index_registration,
                expected_requests=base_requests,
                request_sha256=prepared.request_sha256,
                schedule_sha256=schedule.schedule_sha256,
                split_registry_content_sha256=prepared.registry_content_sha256,
                service_identity=service_identity,
            )
            if service.identity() != service_identity:
                raise RuntimeError("cumulative query reload service identity changed")

        all_ten: dict[str, dict[str, Any]] = {}
        for scope_id in base_requests:
            merged = {
                **legacy_producers[scope_id],
                **embedding_producers[scope_id],
                **tfidf_producers[scope_id],
                **query_producers[scope_id],
            }
            if set(merged) != set(ACTIVE_STAGE1_CONCEPT_FAMILIES):
                raise RuntimeError(f"cumulative producer reload is not all-ten: {scope_id}")
            all_ten[scope_id] = {
                family: merged[family] for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            }
        return {
            "schedule": schedule,
            "requests": base_requests,
            "producers": all_ten,
        }

    @staticmethod
    def _cumulative_producer_native_paths(producer: Any) -> Mapping[str, Path | None]:
        """Expose only the sealed artifact addresses needed by the root proof graph."""

        execution_raw = getattr(producer, "_execution_record_path", None)
        model_raw = getattr(producer, "_model_artifact_path", None)
        if model_raw is None:
            model_raw = getattr(producer, "_capture_artifact_path", None)
        source_raw = getattr(producer, "_source_artifact_path", None)
        if not isinstance(execution_raw, str) or not isinstance(model_raw, str):
            raise TypeError("cumulative producer does not expose its authenticated artifacts")
        execution = Path(execution_raw)
        model = Path(model_raw)
        if source_raw is None and hasattr(producer, "_embedding_provider"):
            source_raw = model / "semantic_full_scope_evidence.json"
        if source_raw is None:
            raise TypeError("cumulative producer does not expose its authenticated source artifact")
        source = Path(str(source_raw))
        if execution.is_symlink() or not execution.is_file():
            raise ValueError("cumulative execution record must be one real regular file")
        if model.is_symlink() or not model.exists():
            raise ValueError("cumulative native model artifact is absent or a symlink")
        if source.is_symlink() or not source.exists():
            raise ValueError("cumulative native source artifact is absent or a symlink")
        return {
            "execution": execution.resolve(strict=True),
            "model": model.resolve(strict=True),
            "source": source.resolve(strict=True),
        }

    def _emit_exact_inner_all_ten_root_index(
        self,
        *,
        output: Path,
        prepared: _PreparedBuild,
        catalogs_by_scope: Mapping[str, RoleNeutralEvidenceCatalog],
        reloaded_native_by_scope: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> Path:
        """Invoke the typed exact-inner boundary from ten persisted native proofs."""

        root = output / "exact_inner_all_ten_root_graph"
        if root.exists():
            raise RuntimeError("exact-inner all-ten root graph already exists before emission")
        catalog_dir = root / "catalogs"
        bundle_dir = root / "typed_bundles"
        catalog_dir.mkdir(parents=True, exist_ok=False)
        bundle_dir.mkdir(parents=True, exist_ok=False)
        contract_registry = _canonical_exact_registry_from_wrapper(prepared.registry)
        expected_scope_ids = {
            f"outer_{outer.outer_fold:03d}_inner_{inner.inner_fold:03d}"
            for outer in contract_registry.outer_splits
            for inner in outer.inner_splits
        }
        if set(reloaded_native_by_scope) != expected_scope_ids:
            raise ValueError("reloaded exact-inner native proofs have incomplete scope coverage")
        dataset = prepared.modeling_data.reset_index(drop=True).copy()
        dataset["_oci_row_id"] = np.arange(len(dataset), dtype=np.int64)
        full_outer_registry_by_outer: dict[int, Any] = {}
        for outer in contract_registry.outer_splits:
            scope_id = f"outer_{outer.outer_fold:03d}_full"
            catalog = catalogs_by_scope.get(scope_id)
            if not isinstance(catalog, RoleNeutralEvidenceCatalog):
                raise ValueError(f"exact root graph lacks full-outer catalog: {scope_id}")
            catalog_path = catalog_dir / f"{scope_id}.json"
            _write_immutable_json(catalog_path, catalog.as_dict())
            full_outer_registry_by_outer[outer.outer_fold] = (
                native_full_outer_payload_registry_from_catalog(
                    catalog=catalog,
                    outer_fold=outer.outer_fold,
                    fit_row_ids=outer.train_row_ids,
                    heldout_row_ids=outer.heldout_row_ids,
                    catalog_artifact_path=catalog_path,
                )
            )

        scope_rows: list[dict[str, Any]] = []
        for outer in contract_registry.outer_splits:
            full_outer_registry = full_outer_registry_by_outer[outer.outer_fold]
            for split in outer.inner_splits:
                scope_id = f"outer_{outer.outer_fold:03d}_inner_{split.inner_fold:03d}"
                catalog = catalogs_by_scope.get(scope_id)
                native_rows = reloaded_native_by_scope[scope_id]
                if (
                    not isinstance(catalog, RoleNeutralEvidenceCatalog)
                    or tuple(native_rows) != ACTIVE_STAGE1_CONCEPT_FAMILIES
                ):
                    raise ValueError(f"exact root scope is not cataloged all-ten: {scope_id}")
                catalog_path = catalog_dir / f"{scope_id}.json"
                _write_immutable_json(catalog_path, catalog.as_dict())
                proofs: dict[str, NativeFamilyFitProof] = {}
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
                    native_row = native_rows[family]
                    proof = native_row.get("proof")
                    persisted_payload = native_row.get("evidence_payload")
                    catalog_payload, catalog_count = family_payload_from_catalog(
                        catalog,
                        family=family,
                    )
                    if (
                        not isinstance(proof, NativeFamilyFitProof)
                        or persisted_payload != catalog_payload
                        or catalog_count != len(catalog_payload["architecture_evidence"])
                    ):
                        raise RuntimeError(
                            f"exact native proof differs from the lossless catalog: "
                            f"{scope_id}/{family}"
                        )
                    proofs[family] = proof
                data_projection_sha256 = _exact_inner_projection_sha256(
                    modeling_data=prepared.modeling_data,
                    config=prepared.config,
                    fit_row_ids=split.fit_row_ids,
                    heldout_row_ids=split.heldout_row_ids,
                )
                native_scope = native_scope_from_catalog(
                    catalog=catalog,
                    catalog_artifact_path=catalog_path,
                    full_outer_registry=full_outer_registry,
                    outer_fold=outer.outer_fold,
                    inner_fold=split.inner_fold,
                    split_scope_fingerprint=split.scope_fingerprint,
                    data_projection_sha256=data_projection_sha256,
                    fit_row_ids=split.fit_row_ids,
                    heldout_row_ids=split.heldout_row_ids,
                    fit_proof_by_family=proofs,
                )
                producers = family_producers_for_native_scope(native_scope)
                typed_bundle = produce_exact_inner_stage1_evidence_bundle(
                    dataset=dataset,
                    registry=contract_registry,
                    outer_fold=outer.outer_fold,
                    inner_fold=split.inner_fold,
                    producers=producers,
                    full_outer_payload_sha256_by_family=(
                        full_outer_registry.payload_sha256_by_family
                    ),
                    text_column=prepared.config.text_column,
                    treatment_column=prepared.config.treatment_column,
                    outcome_column=prepared.config.outcome_column,
                )
                bundle_path = bundle_dir / f"{scope_id}.json"
                _write_immutable_json(bundle_path, typed_bundle)
                scope_rows.append(
                    {
                        **_component_file_registration(bundle_path, component_root=output),
                        "outer_fold": outer.outer_fold,
                        "inner_fold": split.inner_fold,
                        "data_projection_sha256": typed_bundle["data_projection_sha256"],
                        "producer_identity_sha256_by_family": copy.deepcopy(
                            typed_bundle["producer_identity_sha256_by_family"]
                        ),
                        "full_outer_payload_sha256_by_family": copy.deepcopy(
                            typed_bundle["full_outer_payload_sha256_by_family"]
                        ),
                        "catalog": _component_file_registration(
                            catalog_path,
                            component_root=output,
                        ),
                        "catalog_sha256": catalog.catalog_sha256,
                    }
                )
        index_body = {
            "schema_version": STAGE1_EXACT_INNER_ROOT_INDEX_SCHEMA,
            "split_registry_content_sha256": prepared.registry_content_sha256,
            "contract_registry_content_sha256": contract_registry.content_sha256,
            "contract_registry": contract_registry.as_dict(),
            "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "scope_identity_registries_are_local": True,
            "scopes": scope_rows,
        }
        index = {**index_body, "content_sha256": _sha256_json(index_body)}
        index_path = output / "exact_inner_evidence_index.json"
        _write_immutable_json(index_path, index)
        return index_path

    def _emit_cumulative_all_ten_root_graph(
        self,
        *,
        output: Path,
        prepared: _PreparedBuild,
        cumulative: Mapping[str, Any],
        exact_inner_index_path: Path,
    ) -> Mapping[str, Mapping[str, Any]]:
        """Persist typed all-ten bundles, lossless catalogs, and hierarchy proofs."""

        from .production_stage1_hierarchy_handoff import (
            STAGE1_HIERARCHY_NATIVE_MODEL_DESCRIPTOR_SCHEMA,
            STAGE1_HIERARCHY_SPENT_FAMILY_PROOF_SCHEMA,
            STAGE1_HIERARCHY_SPENT_INDEX_SCHEMA,
            STAGE1_HIERARCHY_SPENT_PROOF_BUNDLE_SCHEMA,
        )

        root = output / "cumulative_all_ten_root_graph"
        if root.exists():
            raise RuntimeError("cumulative all-ten root graph already exists before emission")
        bundle_dir = root / "typed_bundles"
        catalog_dir = root / "catalogs"
        proof_dir = root / "proof_bundles"
        descriptor_dir = root / "native_model_descriptors"
        for path in (bundle_dir, catalog_dir, proof_dir, descriptor_dir):
            path.mkdir(parents=True, exist_ok=False)

        schedule = cumulative.get("schedule")
        requests = cumulative.get("requests")
        producers = cumulative.get("producers")
        if (
            schedule is None
            or not isinstance(requests, Mapping)
            or not isinstance(producers, Mapping)
            or tuple(requests) != tuple(scope.scope_id for scope in schedule.scopes)
            or tuple(producers) != tuple(requests)
        ):
            raise ValueError("cumulative all-ten reload changed the canonical schedule order")
        dataset = prepared.modeling_data.reset_index(drop=True).copy()
        dataset["_oci_row_id"] = np.arange(len(dataset), dtype=np.int64)
        interaction_inner_folds = int(
            prepared.config.architecture.explicit_feature_forest.interaction_inner_folds
        )
        tfidf_nested_calibration_folds = int(
            prepared.config.architecture.multi_model_forest.tfidf_nested_calibration_folds
        )
        root_scope_rows: list[dict[str, Any]] = []
        hierarchy_scope_rows: list[dict[str, Any]] = []
        for scope in schedule.scopes:
            scope_id = scope.scope_id
            family_producers = producers[scope_id]
            if tuple(family_producers) != ACTIVE_STAGE1_CONCEPT_FAMILIES:
                raise RuntimeError(f"cumulative producer order is not canonical: {scope_id}")
            typed_bundle = produce_cumulative_spent_stage1_evidence_bundle(
                dataset=dataset,
                request_sha256=prepared.request_sha256,
                schedule_sha256=schedule.schedule_sha256,
                scope_id=scope_id,
                outer_fold=scope.outer_fold,
                context_epoch=scope.context_epoch,
                provider_inner_fold=scope.provider_inner_fold,
                split_scope_fingerprint=scope.split_fingerprint,
                spent_row_ids=scope.spent_row_ids,
                sealed_row_ids=scope.sealed_row_ids,
                producers=family_producers,
                text_column=prepared.config.text_column,
                treatment_column=prepared.config.treatment_column,
                outcome_column=prepared.config.outcome_column,
            )
            artifacts = typed_bundle.get("family_artifacts")
            if (
                not isinstance(artifacts, list)
                or tuple(str(item.get("family")) for item in artifacts if isinstance(item, Mapping))
                != ACTIVE_STAGE1_CONCEPT_FAMILIES
            ):
                raise RuntimeError(f"typed cumulative bundle is not all-ten: {scope_id}")
            artifact_by_family = {
                str(item["family"]): item for item in artifacts if isinstance(item, Mapping)
            }
            family_payloads: dict[str, Mapping[str, Any]] = {}
            family_artifact_hashes: dict[str, str] = {}
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
                artifact = artifact_by_family[family]
                payload = artifact.get("evidence_payload")
                evidence = (
                    payload.get("architecture_evidence") if isinstance(payload, Mapping) else None
                )
                if (
                    not isinstance(payload, Mapping)
                    or not isinstance(evidence, list)
                    or int(artifact.get("evidence_item_count", -1)) != len(evidence)
                    or not evidence
                ):
                    raise RuntimeError(
                        f"typed cumulative artifact count differs from its payload: "
                        f"{scope_id}/{family}"
                    )
                family_payloads[family] = copy.deepcopy(dict(payload))
                family_artifact_hashes[family] = str(artifact.get("artifact_sha256") or "")
            reference_request = requests[scope_id]
            catalog = assemble_cumulative_spent_role_neutral_catalog(
                family_payload_by_family=family_payloads,
                family_artifact_sha256_by_family=family_artifact_hashes,
                scope_binding_sha256=reference_request.binding_sha256,
                scope_id=scope_id,
                outer_fold=scope.outer_fold,
                provider_inner_fold=scope.provider_inner_fold,
                split_fingerprint=scope.split_fingerprint,
            )
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
                projected, projected_count = family_payload_from_catalog(catalog, family=family)
                if projected != family_payloads[family] or projected_count != len(
                    family_payloads[family]["architecture_evidence"]
                ):
                    raise RuntimeError(
                        f"lossless cumulative catalog roundtrip failed: {scope_id}/{family}"
                    )

            bundle_path = bundle_dir / f"{scope_id}.json"
            catalog_path = catalog_dir / f"{scope_id}.json"
            _write_immutable_json(bundle_path, typed_bundle)
            _write_immutable_json(catalog_path, catalog.as_dict())
            family_proofs: list[dict[str, Any]] = []
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
                artifact = artifact_by_family[family]
                producer = family_producers[family]
                paths = self._cumulative_producer_native_paths(producer)
                actual_model_registration = _component_native_artifact_registration(
                    paths["model"],
                    component_root=output,
                )
                source_path = paths["source"]
                actual_source_registration = (
                    None
                    if source_path is None
                    else _component_native_artifact_registration(
                        source_path,
                        component_root=output,
                    )
                )
                descriptor_body = {
                    "schema_version": STAGE1_HIERARCHY_NATIVE_MODEL_DESCRIPTOR_SCHEMA,
                    "scope_id": scope_id,
                    "family": family,
                    "typed_family_artifact_sha256": artifact["artifact_sha256"],
                    "producer_identity_sha256": artifact["producer_identity_sha256"],
                    "native_model_artifact": actual_model_registration,
                    "native_source_artifact": actual_source_registration,
                    "fit_audit": copy.deepcopy(artifact["fit_audit"]),
                }
                descriptor = {
                    **descriptor_body,
                    "content_sha256": _sha256_json(descriptor_body),
                }
                family_descriptor_dir = descriptor_dir / scope_id
                family_descriptor_dir.mkdir(parents=True, exist_ok=True)
                descriptor_path = family_descriptor_dir / f"{family}.json"
                _write_immutable_json(descriptor_path, descriptor)
                execution_registration = _component_file_registration(
                    paths["execution"],
                    component_root=output,
                )
                descriptor_registration = _component_file_registration(
                    descriptor_path,
                    component_root=output,
                )
                catalog_payload, _count = family_payload_from_catalog(catalog, family=family)
                catalog_payload_sha256 = _sha256_json(catalog_payload)
                identity = artifact.get("producer_identity")
                fit_audit = artifact.get("fit_audit")
                if not isinstance(identity, Mapping) or not isinstance(fit_audit, Mapping):
                    raise RuntimeError(
                        f"typed cumulative identity/audit is absent: {scope_id}/{family}"
                    )
                proof_body = {
                    "schema_version": STAGE1_HIERARCHY_SPENT_FAMILY_PROOF_SCHEMA,
                    "family": family,
                    "scope_id": scope_id,
                    "input_binding_sha256": artifact["request_binding_sha256"],
                    "split_fingerprint": scope.split_fingerprint,
                    "fit_semantics": artifact["fit_semantics"],
                    "producer_identity_sha256": artifact["producer_identity_sha256"],
                    "producer_code_sha256": identity["code_sha256"],
                    "configuration_sha256": identity["configuration_sha256"],
                    "fit_execution_sha256": execution_registration["sha256"],
                    "model_artifact_sha256": descriptor_registration["sha256"],
                    "execution_record": execution_registration,
                    "model_artifact": descriptor_registration,
                    "catalog_family_payload_sha256": catalog_payload_sha256,
                    "evidence_payload_sha256": artifact["evidence_payload_sha256"],
                    "tfidf_training_scope_policy": copy.deepcopy(
                        fit_audit.get("tfidf_training_scope_policy")
                    ),
                    "heldout_labels_accessed": False,
                    "oracle_fields_accessed": False,
                    "secrets_accessed": False,
                }
                family_proofs.append({**proof_body, "content_sha256": _sha256_json(proof_body)})
            proof_body = {
                "schema_version": STAGE1_HIERARCHY_SPENT_PROOF_BUNDLE_SCHEMA,
                "request_sha256": prepared.request_sha256,
                "schedule_sha256": schedule.schedule_sha256,
                "scope_id": scope_id,
                "outer_fold": scope.outer_fold,
                "context_epoch": scope.context_epoch,
                "provider_inner_fold": scope.provider_inner_fold,
                "split_fingerprint": scope.split_fingerprint,
                "spent_row_order_fingerprint": row_order_fingerprint(scope.spent_row_ids),
                "sealed_row_order_fingerprint": row_order_fingerprint(scope.sealed_row_ids),
                "data_projection_sha256": typed_bundle["data_projection_sha256"],
                "catalog_sha256": catalog.catalog_sha256,
                "interaction_inner_folds": interaction_inner_folds,
                "tfidf_nested_calibration_folds": tfidf_nested_calibration_folds,
                "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
                "family_proofs": family_proofs,
                "sealed_text_available_to_producers": False,
                "sealed_labels_available_to_producers": False,
            }
            proof_bundle = {**proof_body, "content_sha256": _sha256_json(proof_body)}
            proof_path = proof_dir / f"{scope_id}.json"
            _write_immutable_json(proof_path, proof_bundle)
            bundle_registration = _component_file_registration(bundle_path, component_root=output)
            catalog_registration = _component_file_registration(catalog_path, component_root=output)
            proof_registration = _component_file_registration(proof_path, component_root=output)
            root_scope_rows.append(
                {
                    "scope_id": scope_id,
                    "outer_fold": scope.outer_fold,
                    "context_epoch": scope.context_epoch,
                    "provider_inner_fold": scope.provider_inner_fold,
                    "split_fingerprint": scope.split_fingerprint,
                    "typed_bundle": bundle_registration,
                    "typed_bundle_sha256": typed_bundle["bundle_sha256"],
                    "catalog": catalog_registration,
                    "catalog_sha256": catalog.catalog_sha256,
                    "proof_bundle": proof_registration,
                }
            )
            hierarchy_scope_rows.append(
                {
                    "scope_id": scope_id,
                    "outer_fold": scope.outer_fold,
                    "context_epoch": scope.context_epoch,
                    "provider_inner_fold": scope.provider_inner_fold,
                    "spent_row_ids": list(scope.spent_row_ids),
                    "sealed_row_ids": list(scope.sealed_row_ids),
                    "split_fingerprint": scope.split_fingerprint,
                    "catalog": catalog_registration,
                    "catalog_sha256": catalog.catalog_sha256,
                    "proof_bundle": proof_registration,
                }
            )

        exact_index_registration = _component_file_registration(
            exact_inner_index_path,
            component_root=output,
        )
        root_index_body = {
            "schema_version": STAGE1_CUMULATIVE_ALL_TEN_ROOT_INDEX_SCHEMA,
            "request_sha256": prepared.request_sha256,
            "split_registry_content_sha256": prepared.registry_content_sha256,
            "schedule_sha256": schedule.schedule_sha256,
            "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "exact_inner_evidence_index": exact_index_registration,
            "scopes": root_scope_rows,
            "manual_digest_approval_required": False,
        }
        root_index = {**root_index_body, "content_sha256": _sha256_json(root_index_body)}
        root_index_path = root / "index.json"
        _write_immutable_json(root_index_path, root_index)
        contract_registry = _canonical_exact_registry_from_wrapper(prepared.registry)
        contract = prepared.request["hierarchy_spent_evidence_contract"]
        hierarchy_index_body = {
            "schema_version": STAGE1_HIERARCHY_SPENT_INDEX_SCHEMA,
            "request_sha256": prepared.request_sha256,
            "wrapper_split_registry_content_sha256": prepared.registry_content_sha256,
            "contract_split_registry_sha256": contract_registry.content_sha256,
            "schedule_sha256": schedule.schedule_sha256,
            "review_rounds": int(contract["review_rounds"]),
            "initial_spent_partition_count": 3,
            "canonical_hierarchy_partition_count": int(
                contract["canonical_hierarchy_partition_count"]
            ),
            "interaction_inner_folds": interaction_inner_folds,
            "tfidf_nested_calibration_folds": tfidf_nested_calibration_folds,
            "fold_domains_are_distinct": True,
            "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "hierarchical_discovery_contract_identity_sha256": (
                prepared.hierarchical_discovery_contract_identity["content_sha256"]
            ),
            "exact_inner_evidence_index_file_sha256": exact_index_registration["sha256"],
            "scopes": hierarchy_scope_rows,
            "independent_runtime_stage1_refit_allowed": False,
            "manual_digest_approval_required": False,
        }
        hierarchy_index = {
            **hierarchy_index_body,
            "content_sha256": _sha256_json(hierarchy_index_body),
        }
        hierarchy_index_path = output / "hierarchy_spent_evidence_index.json"
        _write_immutable_json(hierarchy_index_path, hierarchy_index)
        return {
            "cumulative_all_ten_root_index": _component_file_registration(
                root_index_path,
                component_root=output,
            ),
            "hierarchy_spent_evidence_index": _component_file_registration(
                hierarchy_index_path,
                component_root=output,
            ),
        }

    def _validate_all_scope_coverage(
        self,
        *,
        legacy_root: Path,
        tfidf_root: Path,
        query_root: Path,
        prepared: _PreparedBuild,
        emit_root_graph: bool = False,
    ) -> Mapping[str, Any]:
        cumulative = (
            self._load_cumulative_all_ten_producers(
                legacy_root=legacy_root,
                tfidf_root=tfidf_root,
                query_root=query_root,
                prepared=prepared,
            )
            if emit_root_graph
            else None
        )
        legacy_rows = self._validate_legacy_scope_lineage(
            legacy_root / "handoff" / "discovery_contexts.jsonl", prepared
        )
        tfidf_handoff_path = tfidf_root / "handoff" / "discovery_contexts.jsonl"
        resealed_tfidf = load_resealed_tfidf_handoff(
            tfidf_handoff_path,
            dataset_row_count=len(prepared.modeling_data),
            require_registry_seal=True,
        )
        if resealed_tfidf.split_registry_content_hash != prepared.registry_content_sha256:
            raise ValueError("TF-IDF evidence is sealed to a different split registry")
        tfidf_rows: dict[str, Mapping[str, Any]] = {}
        for outer_fold, fold_rows in resealed_tfidf.rows_by_outer_fold.items():
            for row in fold_rows:
                inner_fold = row.get("inner_fold")
                scope_id = (
                    f"outer_{int(outer_fold):03d}_inner_{int(inner_fold):03d}"
                    if inner_fold is not None
                    else f"outer_{int(outer_fold):03d}_full"
                )
                if scope_id in tfidf_rows:
                    raise ValueError(f"duplicate TF-IDF Stage 1 scope: {scope_id}")
                tfidf_rows[scope_id] = row
        query_index = json.loads(
            (query_root / "query_artifact_index.json").read_text(encoding="utf-8")
        )
        if (
            query_index.get("schema_version") != STAGE1_SCOPE_INDEX_SCHEMA
            or query_index.get("split_registry_content_sha256") != prepared.registry_content_sha256
            or query_index.get("registered_native_families")
            != list(PRODUCTION_QUERY_REGISTERED_NATIVE_FAMILY_ADAPTERS)
            or query_index.get("heldout_labels_supplied") is not False
            or query_index.get("executable_checkpoint_files_retained") is not False
            or not isinstance(query_index.get("scopes"), list)
        ):
            raise ValueError("neural-query scope index has an invalid registry binding")
        query_rows = {str(row["scope_id"]): row for row in query_index["scopes"]}
        if len(query_rows) != len(query_index["scopes"]):
            raise ValueError("neural-query scope index contains duplicates")
        expected_scopes = {
            str(scope["scope_id"]): scope for scope in _registry_scopes(prepared.registry)
        }
        if set(tfidf_rows) != set(expected_scopes) or set(query_rows) != set(expected_scopes):
            raise ValueError("Stage 1 component scope sets do not match the canonical registry")
        expected_inner_scopes = {
            scope_id: scope
            for scope_id, scope in expected_scopes.items()
            if scope["inner_fold"] is not None
        }
        reloaded_native_by_scope: dict[str, dict[str, Mapping[str, Any]]] | None = (
            {} if emit_root_graph else None
        )
        tfidf_configuration_by_scope: dict[str, Mapping[str, Any]] = {}
        query_configuration_by_scope: dict[str, Mapping[str, Any]] = {}
        if emit_root_graph:
            for scope_id in expected_inner_scopes:
                tfidf_discovery = tfidf_rows[scope_id].get("discovery") or {}
                tfidf_configuration_by_scope[scope_id] = {
                    "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
                    "scope_id": scope_id,
                    "text_column": prepared.config.text_column,
                    "tfidf_nested_calibration_folds": int(
                        prepared.config.architecture.multi_model_forest.tfidf_nested_calibration_folds
                    ),
                    "score_selection_label_policy": "nested_fit_calibration",
                    "stage1_config_hash": tfidf_discovery.get("stage1_config_hash"),
                    "topic_configuration_hash": tfidf_discovery.get("config_hash"),
                }
                query_registration = query_rows[scope_id]
                owned_metadata_path = _validate_component_native_registration(
                    query_root,
                    query_registration.get("owned_snapshot_metadata") or {},
                )
                owned_metadata = _read_json_object_reject_duplicates(
                    owned_metadata_path,
                    field_name=f"{scope_id} owned query snapshot metadata",
                )
                query_configuration_by_scope[scope_id] = {
                    "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
                    "scope_id": scope_id,
                    "text_column": prepared.config.text_column,
                    "query_config": asdict(prepared.query_config),
                    "query_nuisance_folds": int(prepared.options.query_nuisance_folds),
                    "query_devices": list(prepared.options.query_devices),
                    "seed": int(prepared.options.seed),
                    "outcome_type": prepared.config.outcome_type,
                    "split_registry_content_sha256": prepared.registry_content_sha256,
                    "service_identity_sha256": owned_metadata["service_identity_sha256"],
                    "heldout_label_policy": "id_and_text_only",
                }
        legacy_scope_index = json.loads(
            (legacy_root / "exact_scope_index.json").read_text(encoding="utf-8")
        )
        expected_legacy_native_families = list(
            (
                *PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                *PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                *PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                *PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS,
            )
        )
        if (
            legacy_scope_index.get("registered_native_families") != expected_legacy_native_families
            or not isinstance(legacy_scope_index.get("native_bow_family_proof_index"), Mapping)
            or not isinstance(legacy_scope_index.get("native_htr_family_proof_index"), Mapping)
            or not isinstance(
                legacy_scope_index.get("native_matched_pair_family_proof_index"),
                Mapping,
            )
            or not isinstance(
                legacy_scope_index.get("native_embedding_family_proof_index"),
                Mapping,
            )
        ):
            raise ValueError(
                "legacy component lacks registered BoW/HTR/matched-pair/embedding " "native proofs"
            )
        _validate_bow_native_family_proof_index(
            component_root=legacy_root,
            index_registration=legacy_scope_index["native_bow_family_proof_index"],
            expected_inner_scopes=expected_inner_scopes,
            split_registry_content_sha256=prepared.registry_content_sha256,
            modeling_data=prepared.modeling_data,
            text_column=prepared.config.text_column,
            treatment_column=prepared.config.treatment_column,
            outcome_column=prepared.config.outcome_column,
            reloaded_native_by_scope=reloaded_native_by_scope,
        )
        _validate_htr_native_family_proof_index(
            component_root=legacy_root,
            index_registration=legacy_scope_index["native_htr_family_proof_index"],
            expected_inner_scopes=expected_inner_scopes,
            split_registry_content_sha256=prepared.registry_content_sha256,
            modeling_data=prepared.modeling_data,
            text_column=prepared.config.text_column,
            treatment_column=prepared.config.treatment_column,
            outcome_column=prepared.config.outcome_column,
            htr_model_path=prepared.htr_model_path,
            htr_model_sha256=prepared.htr_model_sha256,
            device=torch.device(prepared.options.device),
            reloaded_native_by_scope=reloaded_native_by_scope,
        )
        _validate_matched_pair_native_family_proof_index(
            component_root=legacy_root,
            index_registration=legacy_scope_index["native_matched_pair_family_proof_index"],
            expected_inner_scopes=expected_inner_scopes,
            split_registry_content_sha256=prepared.registry_content_sha256,
            modeling_data=prepared.modeling_data,
            text_column=prepared.config.text_column,
            treatment_column=prepared.config.treatment_column,
            outcome_column=prepared.config.outcome_column,
            htr_model_path=prepared.htr_model_path,
            htr_model_sha256=prepared.htr_model_sha256,
            device=torch.device(prepared.options.device),
            reloaded_native_by_scope=reloaded_native_by_scope,
        )
        _validate_embedding_native_family_proof_index(
            component_root=legacy_root,
            index_registration=legacy_scope_index["native_embedding_family_proof_index"],
            expected_inner_scopes=expected_inner_scopes,
            split_registry_content_sha256=prepared.registry_content_sha256,
            modeling_data=prepared.modeling_data,
            text_column=prepared.config.text_column,
            treatment_column=prepared.config.treatment_column,
            outcome_column=prepared.config.outcome_column,
            embedding_cache=prepared.embedding_cache,
            reloaded_native_by_scope=reloaded_native_by_scope,
        )
        if emit_root_graph:
            _validate_tfidf_native_family_proof_index(
                component_root=tfidf_root,
                index_registration=_component_file_registration(
                    tfidf_root / "native_family_proof_index.json",
                    component_root=tfidf_root,
                ),
                expected_inner_scopes=expected_inner_scopes,
                expected_configuration_by_scope=tfidf_configuration_by_scope,
                split_registry_content_sha256=prepared.registry_content_sha256,
                modeling_data=prepared.modeling_data,
                treatment_column=prepared.config.treatment_column,
                outcome_column=prepared.config.outcome_column,
                reloaded_native_by_scope=reloaded_native_by_scope,
            )
        _validate_neural_query_native_family_proof_index(
            component_root=query_root,
            index_registration=query_index.get("native_family_proof_index") or {},
            expected_inner_scopes=expected_inner_scopes,
            split_registry_content_sha256=prepared.registry_content_sha256,
            modeling_data=prepared.modeling_data,
            treatment_column=prepared.config.treatment_column,
            outcome_column=prepared.config.outcome_column,
            expected_configuration_by_scope=(
                query_configuration_by_scope if emit_root_graph else None
            ),
            reloaded_native_by_scope=reloaded_native_by_scope,
        )
        output: list[dict[str, Any]] = []
        catalogs_by_scope: dict[str, RoleNeutralEvidenceCatalog] = {}
        for scope_id, scope in expected_scopes.items():
            legacy_raw = legacy_rows[scope_id]
            tfidf_raw = tfidf_rows[scope_id]
            if (
                list(map(int, tfidf_raw.get("fit_row_ids") or ())) != scope["fit_row_ids"]
                or list(map(int, tfidf_raw.get("heldout_row_ids") or ()))
                != scope["heldout_row_ids"]
                or tfidf_raw.get("fit_row_fingerprint") != row_set_fingerprint(scope["fit_row_ids"])
                or tfidf_raw.get("heldout_row_fingerprint")
                != row_set_fingerprint(scope["heldout_row_ids"])
            ):
                raise ValueError(f"TF-IDF scope lineage mismatch: {scope_id}")
            digest = _catalog_ready_legacy_digest(
                importance=legacy_raw.get("importance") or {},
                embedding_evidence=legacy_raw.get("embedding_contrast_evidence") or {},
                htr_evidence=legacy_raw.get("htr_evidence") or {},
            )
            legacy_payload = {
                "outer_fold": int(scope["outer_fold"]),
                "scope": ("outer_train" if scope["inner_fold"] is None else "inner_train"),
                "n_rows": len(scope["fit_row_ids"]),
                "context": {"evidence_digest": digest},
            }
            if scope["inner_fold"] is not None:
                legacy_payload["inner_fold"] = int(scope["inner_fold"])
            # The raw full row intentionally has no held-out score tests and no
            # orphan branch.  The authenticated loader derives the full-outer
            # orphan recurrence from the independently fitted exact-inner rows.
            # Supplying the raw row here silently drops that architecture.
            tfidf_evidence_row = tfidf_raw
            if scope["inner_fold"] is None:
                try:
                    tfidf_evidence_row = resealed_tfidf.full_rows_by_outer_fold[
                        int(scope["outer_fold"])
                    ]
                except KeyError as exc:
                    raise ValueError(f"missing resealed full TF-IDF scope: {scope_id}") from exc
                if (
                    list(map(int, tfidf_evidence_row.get("fit_row_ids") or ()))
                    != scope["fit_row_ids"]
                    or list(map(int, tfidf_evidence_row.get("heldout_row_ids") or ()))
                    != scope["heldout_row_ids"]
                ):
                    raise ValueError(f"resealed TF-IDF scope lineage mismatch: {scope_id}")
            tfidf_discovery = _catalog_ready_tfidf_discovery(
                tfidf_evidence_row.get("discovery") or {}
            )
            tfidf_payload = {
                "outer_fold": int(scope["outer_fold"]),
                "scope": legacy_payload["scope"],
                "discovery": tfidf_discovery,
            }
            query_registration = query_rows[scope_id]
            query_path = query_root / str(query_registration["path"])
            if _sha256_file(query_path) != query_registration["sha256"]:
                raise ValueError(f"neural-query artifact changed: {scope_id}")
            query_payload = json.loads(query_path.read_text(encoding="utf-8"))
            if (
                query_payload.get("split_registry_content_sha256")
                != prepared.registry_content_sha256
                or query_payload.get("heldout_labels_supplied") is not False
                or list(map(int, query_payload.get("fit_row_ids") or ())) != scope["fit_row_ids"]
                or list(map(int, query_payload.get("heldout_row_ids") or ()))
                != scope["heldout_row_ids"]
                or query_payload.get("fit_row_fingerprint")
                != row_set_fingerprint(scope["fit_row_ids"])
                or query_payload.get("heldout_row_fingerprint")
                != row_set_fingerprint(scope["heldout_row_ids"])
                or query_payload.get("scope_id") != scope_id
                or query_registration.get("heldout_labels_supplied") is not False
            ):
                raise ValueError(f"neural-query scope lineage mismatch: {scope_id}")
            model_path = _validate_component_native_registration(
                query_root,
                query_registration.get("native_model_artifact") or {},
            )
            snapshot_metadata_path = _validate_component_native_registration(
                query_root,
                query_registration.get("owned_snapshot_metadata") or {},
            )
            snapshot_arrays_path = _validate_component_native_registration(
                query_root,
                query_registration.get("owned_snapshot_arrays") or {},
            )
            moment_metadata_path = _validate_component_native_registration(
                query_root,
                query_registration.get("heldout_moment_metadata") or {},
            )
            moment_arrays_path = _validate_component_native_registration(
                query_root,
                query_registration.get("heldout_moment_arrays") or {},
            )
            if (
                snapshot_metadata_path != model_path / "owned_snapshot" / "metadata.json"
                or snapshot_arrays_path != model_path / "owned_snapshot" / "arrays.npz"
                or moment_metadata_path != model_path / "heldout_moments.metadata.json"
                or moment_arrays_path != model_path / "heldout_moments.npz"
            ):
                raise ValueError(f"neural-query native artifact layout mismatch: {scope_id}")
            snapshot = validate_owned_discovery_snapshot(
                model_path / "owned_snapshot",
                expected_cache_key=str(query_payload.get("query_cache_key") or ""),
            )
            moments = _validate_neural_query_moment_artifact(
                moment_metadata_path,
                expected_scope_id=scope_id,
                expected_fit_row_ids=scope["fit_row_ids"],
                expected_heldout_row_ids=scope["heldout_row_ids"],
                expected_query_cache_key=str(query_payload.get("query_cache_key") or ""),
                expected_snapshot_content_sha256=str(snapshot["content_sha256"]),
            )
            declared_model = query_payload.get("native_model_artifact")
            declared_moments = query_payload.get("heldout_moment_artifact")
            if (
                not isinstance(declared_model, Mapping)
                or declared_model.get("relative_path")
                != query_registration["native_model_artifact"]["relative_path"]
                or declared_model.get("sha256")
                != query_registration["native_model_artifact"]["sha256"]
                or not isinstance(declared_moments, Mapping)
                or declared_moments.get("relative_path")
                != query_registration["heldout_moment_arrays"]["relative_path"]
                or declared_moments.get("sha256") != moments["arrays_sha256"]
                or int(query_registration.get("heldout_moment_feature_count", 0))
                != int(moments["feature_count"])
                or (
                    scope["inner_fold"] is None
                    and query_registration.get("native_family_proof_registration") is not None
                )
                or (
                    scope["inner_fold"] is not None
                    and not isinstance(
                        query_registration.get("native_family_proof_registration"),
                        Mapping,
                    )
                )
            ):
                raise ValueError(f"neural-query native artifact linkage mismatch: {scope_id}")
            provenance = FoldEvidenceProvenance(
                outer_fold=int(scope["outer_fold"]),
                train_row_ids=tuple(scope["fit_row_ids"]),
                heldout_row_ids=tuple(scope["heldout_row_ids"]),
                scope=("outer_train" if scope["inner_fold"] is None else "inner_train"),
                inner_fold=(None if scope["inner_fold"] is None else int(scope["inner_fold"])),
                artifact_id=f"production-stage1-{scope_id}",
            )
            catalog = build_role_neutral_evidence_catalog(
                (
                    FoldEvidenceInput(LEGACY_ALL_SOURCE, legacy_payload, provenance),
                    FoldEvidenceInput(TFIDF_TOPIC_SOURCE, tfidf_payload, provenance),
                    FoldEvidenceInput(NEURAL_QUERY_SOURCE, query_payload, provenance),
                )
            )
            counts = {
                str(key): int(value) for key, value in catalog.audit["atom_count_by_family"].items()
            }
            missing = [
                family for family in ACTIVE_STAGE1_CONCEPT_FAMILIES if counts.get(family, 0) < 1
            ]
            if missing:
                raise RuntimeError(
                    f"Stage 1 scope {scope_id} has zero concept evidence for: " + ", ".join(missing)
                )
            catalogs_by_scope[scope_id] = catalog
            output.append(
                {
                    "scope_id": scope_id,
                    "outer_fold": scope["outer_fold"],
                    "inner_fold": scope["inner_fold"],
                    "family_counts": counts,
                    "semantic_member_counts": catalog.audit["semantic_member_count_by_family"],
                    "catalog_sha256": catalog.catalog_sha256,
                    "all_ten_families_nonzero": True,
                }
            )
        if emit_root_graph:
            assert reloaded_native_by_scope is not None
            assert cumulative is not None
            bundle_output = legacy_root.parent
            exact_index_path = self._emit_exact_inner_all_ten_root_index(
                output=bundle_output,
                prepared=prepared,
                catalogs_by_scope=catalogs_by_scope,
                reloaded_native_by_scope=reloaded_native_by_scope,
            )
            self._emit_cumulative_all_ten_root_graph(
                output=bundle_output,
                prepared=prepared,
                cumulative=cumulative,
                exact_inner_index_path=exact_index_path,
            )
        return {
            "required_families": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "all_ten_families_nonzero_in_every_scope": True,
            "all_ten_cumulative_producers_reloaded": bool(emit_root_graph),
            "cumulative_scope_count": (0 if cumulative is None else len(cumulative["requests"])),
            "root_graph_emitted": bool(emit_root_graph),
            "scope_count": len(output),
            "scopes": output,
        }

    @staticmethod
    def _register_file(path: Path, root: Path) -> Mapping[str, Any]:
        return {
            "relative_path": path.relative_to(root).as_posix(),
            "size": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }

    def _validate_complete_bundle(
        self, output: Path, prepared: _PreparedBuild
    ) -> Mapping[str, Any]:
        self._revalidate_prepared_inputs(prepared)
        manifest_path = output / "bundle_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != STAGE1_BUNDLE_MANIFEST_SCHEMA
            or payload.get("request_sha256") != prepared.request_sha256
            or payload.get("hierarchical_discovery_contract_identity_sha256")
            != prepared.hierarchical_discovery_contract_identity["content_sha256"]
            or payload.get("manual_digest_approval_required") is not False
        ):
            raise RuntimeError("bundle manifest identity does not match this request")
        body = dict(payload)
        declared = body.pop("bundle_sha256", None)
        if not _HEX_SHA256.fullmatch(str(declared or "")) or _sha256_json(body) != declared:
            raise RuntimeError("bundle manifest content hash is invalid")
        for key in (
            "immutable_build_request",
            "stage1_config",
            "split_registry",
            "primary_splits",
            "row_registry",
            "legacy_handoff",
            "tfidf_handoff",
            "neural_query_artifact_index",
            "exact_inner_evidence_index",
            "cumulative_all_ten_root_index",
            "hierarchy_spent_evidence_index",
        ):
            registration = payload[key]
            path = output / str(registration["relative_path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(registration["size"])
                or _sha256_file(path) != str(registration["sha256"])
            ):
                raise RuntimeError(f"bundle file registration changed: {key}")
        for component, registration in payload["components"].items():
            root = output / str(registration["relative_path"])
            component_manifest = _seal_component(
                root,
                request_sha256=prepared.request_sha256,
                component=component,
            )
            if (
                _sha256_file(root / "component_manifest.json") != registration["manifest_sha256"]
                or component_manifest["content_sha256"] != registration["content_sha256"]
            ):
                raise RuntimeError(f"bundle component registration changed: {component}")
        self._revalidate_prepared_inputs(prepared)
        splits = load_outer_splits_from_primary_predictions(
            output / payload["primary_splits"]["relative_path"],
            dataset_row_count=len(prepared.data),
        )
        if set(splits) != set(range(1, int(prepared.config.cv_folds) + 1)):
            raise RuntimeError("completed bundle primary splits are incomplete")
        coverage = payload.get("coverage") or {}
        if coverage.get("all_ten_families_nonzero_in_every_scope") is not True:
            raise RuntimeError("completed bundle lacks mandatory all-ten coverage")
        hierarchy_contract = prepared.request["hierarchy_spent_evidence_contract"]
        from .production_stage1_hierarchy_handoff import (
            load_production_stage1_hierarchy_handoff,
        )

        hierarchy_handoff = load_production_stage1_hierarchy_handoff(
            manifest_path,
            review_rounds=int(hierarchy_contract["review_rounds"]),
            interaction_inner_folds=int(hierarchy_contract["interaction_inner_folds"]),
            tfidf_nested_calibration_folds=int(
                hierarchy_contract["tfidf_nested_calibration_folds"]
            ),
        )
        hierarchy_handoff.provider.identity()
        return {
            "bundle_manifest": str(manifest_path),
            "bundle_sha256": declared,
            "request_sha256": prepared.request_sha256,
            "row_count": len(prepared.data),
            "outer_fold_count": len(splits),
            "exact_scope_count": int(coverage["scope_count"]),
            "all_ten_families_nonzero_in_every_scope": True,
            "hierarchy_root_graph_authenticated": True,
            "production_hierarchy_execution_ready": False,
            "manual_digest_approval_required": False,
            "remote_clients_constructed": False,
            "remote_calls_made": False,
            "oracle_columns_decoded_or_materialized": False,
            "whole_parquet_container_authenticated": True,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an authenticated all-ten Stage 1 bundle for an arbitrary cohort, including "
            "exact-inner and cumulative-spent hierarchy roots; final one-shot hierarchy "
            "certification remains separate"
        ),
        epilog=(
            "Digests are internal integrity controls; this command never requests manual "
            "digest approval and has no bypass for authenticated producer or cache validation."
        ),
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stage1-config", type=Path, required=True)
    cache_mode = parser.add_mutually_exclusive_group(required=True)
    cache_mode.add_argument(
        "--embedding-cache-dir",
        type=Path,
        help="Use an existing authenticated four-file embedding cache.",
    )
    cache_mode.add_argument(
        "--embedding-cache-output-dir",
        type=Path,
        help=(
            "Build and atomically publish a fresh arbitrary-cohort embedding cache at "
            "this path before Stage 1."
        ),
    )
    parser.add_argument(
        "--embedding-local-model-path",
        type=Path,
        help=(
            "Absolute local sentence-embedding model tree; required with "
            "--embedding-cache-output-dir and forbidden with --embedding-cache-dir."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unit-id-column", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-id", type=int, action="append", default=[])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--tfidf-workers", type=int, default=1)
    parser.add_argument(
        "--tfidf-parallel-backend",
        choices=("threads", "processes", "multiprocessing", "fork"),
        default="threads",
    )
    parser.add_argument("--query-device", action="append", default=[])
    parser.add_argument("--query-nuisance-folds", type=int, default=3)
    parser.add_argument("--query-config", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only byte-verified sealed components from the identical build request.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def options_from_args(args: argparse.Namespace) -> Stage1BundleBuildOptions:
    if args.embedding_cache_output_dir is not None and args.embedding_local_model_path is None:
        raise ValueError(
            "--embedding-local-model-path is required with --embedding-cache-output-dir"
        )
    if args.embedding_cache_dir is not None and args.embedding_local_model_path is not None:
        raise ValueError("--embedding-local-model-path is forbidden with --embedding-cache-dir")
    if args.resume and args.embedding_cache_output_dir is not None:
        raise ValueError(
            "--resume requires --embedding-cache-dir; a fresh cache build is never resumed"
        )
    if args.dry_run and args.embedding_cache_output_dir is not None:
        raise ValueError(
            "--dry-run cannot publish a fresh cache; prebuild it and pass " "--embedding-cache-dir"
        )
    return Stage1BundleBuildOptions(
        dataset_path=args.dataset,
        config_path=args.stage1_config,
        embedding_cache_dir=args.embedding_cache_dir,
        embedding_local_model_path=args.embedding_local_model_path,
        embedding_cache_output_dir=args.embedding_cache_output_dir,
        output_dir=args.output_dir,
        unit_id_column=args.unit_id_column,
        seed=int(args.seed),
        device=str(args.device),
        gpu_ids=tuple(args.gpu_id),
        num_workers=int(args.num_workers),
        tfidf_workers=int(args.tfidf_workers),
        tfidf_parallel_backend=str(args.tfidf_parallel_backend),
        query_devices=tuple(args.query_device or [args.device]),
        query_nuisance_folds=int(args.query_nuisance_folds),
        query_config_path=args.query_config,
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = options_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    result = ProductionStage1BundleBuilder(options).build()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 2 if str(result.get("status") or "").startswith("blocked_") else 0


__all__ = [
    "PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS",
    "PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS",
    "PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS",
    "PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS",
    "PRODUCTION_QUERY_REGISTERED_NATIVE_FAMILY_ADAPTERS",
    "PRODUCTION_REGISTERED_NATIVE_FAMILY_ADAPTERS",
    "PRODUCTION_TFIDF_REGISTERED_NATIVE_FAMILY_ADAPTERS",
    "STAGE1_BUNDLE_MANIFEST_SCHEMA",
    "STAGE1_BUNDLE_REQUEST_SCHEMA",
    "STAGE1_HTR_INPUT_NONTRUNCATION_AUDIT_SCHEMA",
    "ProductionStage1BundleBuilder",
    "Stage1BundleBuildOptions",
    "build_canonical_split_registry",
    "build_parser",
    "exact_inner_family_adapter_gate",
    "load_applied_stage1_config",
    "main",
    "options_from_args",
    "validate_htr_input_nontruncation_audit",
]
