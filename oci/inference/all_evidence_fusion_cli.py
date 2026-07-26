"""Remote-only command line entry point for all-evidence fusion benchmarks.

This module deliberately keeps benchmark execution behind a narrow boundary:

* production initial discovery is hierarchical and architecture-at-a-time,
  using strict JSON jobs against one explicit endpoint/model (never the legacy
  staged prompt agent);
* selector reasoning is enabled with exactly 5000 tokens while hierarchy
  extraction-definition reasoning and patient extraction reasoning are off;
* every outer fold is prepared into one inspectable batch before the exact
  batch SHA-256 can authorize cache lookup or remote hierarchy jobs;
* the legacy staged discovery mode remains only for tests and historical
  ablations;
* extraction is fixed to ``vllm_mode='server'`` and cannot start or import a
  local model through a CLI option;
* the causal runner sees only text, treatment, and observed outcome columns;
* synthetic oracle effects are loaded only after frozen predictions exist and
  only when post-hoc evaluation was explicitly requested.

``--dry-run`` authenticates the input structure and reports the complete
configuration without constructing any agent, extraction provider, runner, or
network client. ``--prepare-hierarchical-discovery`` may materialize local
Stage-1/cache artifacts in a fresh scratch output, but makes no remote call and
writes no prediction or final run manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import socket
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import psutil

from ..config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
)
from ..extraction import CONTRACT_LEXICAL_CONTEXT_VERSION, EXTRACTION_GROUPING_VERSION
from ..extraction.llm_routing import parse_server_urls
from .agentic_explicit_feature_forest import (
    EXTRACTION_PROMPT_VERSION,
    OpenAICompatibleFeatureSearchAgent,
    VLLMExplicitFeatureExtractionProvider,
)
from .all_evidence_fusion import (
    FoldEvidenceProvenance,
    NEURAL_QUERY_MOMENTS,
    source_text_temporal_policy_audit,
)
from .all_evidence_fusion_runner import (
    AllEvidenceFusionRunResult,
    AllEvidenceFusionRunner,
    AllEvidenceFusionRunnerConfig,
    DEFAULT_POST_EXTRACTION_REVIEW_ROUNDS,
    QueryEvidenceArtifact,
    TfidfOrphanNgramArtifact,
    evaluate_frozen_all_evidence_predictions,
    load_legacy_full_outer_evidence,
    load_outer_splits_from_primary_predictions,
    load_resealed_tfidf_handoff,
    load_sanitized_dataset,
)
from .approved_hierarchical_discovery_batch import (
    FrozenReviewEvidencePolicyBinding,
)
from .adaptive_hierarchical_stage1_reconsideration import (
    AdaptiveReconsiderationConfig,
    adaptive_hierarchical_stage1_reconsideration_identity,
)
from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .frozen_extraction_cache_overlay import FrozenExtractionCacheOverlay, sha256_file
from .first_untouched_gate_direct_numerical_preparation import (
    FirstUntouchedGatePreparationBounds,
)
from .hierarchical_discovery_job_cache import (
    HierarchicalDiscoveryJobCacheConfig,
)
from .context_fit_upstream_gate_provider import (
    CompositeContextFitUpstreamBackend,
    ContextFitUpstreamGateProvider,
)
from .context_fit_upstream_cache_overlay import (
    AuthenticatedContextFitCacheSource,
    AuthenticatedContextFitGateCacheOverlay,
    AuthenticatedFinalContextFitCacheOverlay,
    authenticate_context_fit_cache_index_registrations,
)
from .neural_query_agentic_forest import NeuralQueryAgenticForestConfig
from .neural_query_context_backend import (
    ContextFitNeuralQueryService,
    NeuralQueryContextBackend,
    NeuralQuerySpentDiscoveryBackend,
)
from .final_context_fit_upstream_bank import FinalContextFitUpstreamProducer
from .coordinate_preserving_context_fit_upstream_backend import (
    CoordinatePreservingContextFitUpstreamBackend,
    CoordinatePreservingUpstreamSchemaConfig,
)
from .final_context_fit_causal_forest_adapter import (
    FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID,
    FixedCausalForestHeadBackend,
)
from .review_spent_evidence_provider import (
    ALL_NON_QUERY_DISCOVERY_FAMILIES,
    ContextFitReviewSpentEvidenceProvider,
    HistoricalStage1SpentDiscoveryBackend,
    SemanticWitnessScientificConfig,
    SpentOnlyFrozenChunkEmbeddingCache,
    TfidfTopicOrphanSpentDiscoveryBackend,
)
from .tfidf_orphan_evidence_adapter import (
    orphan_ngram_adapter_config_from_tfidf_topic,
)
from .review_spent_evidence_cache_overlay import (
    AuthenticatedReviewSpentCacheSource,
    AuthenticatedReviewSpentEvidenceCacheOverlay,
    authenticate_review_spent_cache_registrations,
)
from .stable_context_fit_upstream_backend import (
    CrossFitStableUpstreamSchemaConfig,
    PrecommittedCalibratedSource,
    PrecommittedRawFeatureFamily,
)
from .production_coordinate_preserving_upstream_schema import (
    build_production_coordinate_preserving_schema,
    production_coordinate_preserving_registry_audit,
)
from .stage1_upstream_gate_backend import (
    HistoricalStage1ConfigSnapshot,
    HistoricalStage1ContextBackend,
    PrivateHTRModelTreeSnapshot,
    _minimal_historical_applied_config,
    _resolve_htr_model_path,
)
from .shared_tfidf_context_fit_service import (
    SHARED_TFIDF_CONTEXT_BACKEND_ID,
    SHARED_TFIDF_RUNTIME_GRAPH_ID,
    UNWRAPPED_TFIDF_RUNTIME_GRAPH_ID,
    build_shared_tfidf_context_fit_backends,
    classify_tfidf_context_member_identity,
)
from .tfidf_upstream_gate_backend import (
    TFIDF_CONTEXT_BACKEND_ID,
    TfidfTopicOrphanContextBackend,
)
from .query_moment_evidence_adapter import load_query_moment_evidence_artifact
from .frozen_hierarchical_review_evidence import (
    frozen_hierarchical_review_evidence_identity,
)
from .hierarchical_all_architecture_discovery import (
    MAX_RENDERED_DISCOVERY_PROMPT_BYTES,
    HierarchicalDiscoveryConfig,
)
from .hierarchical_discovery_response_contract import (
    HIERARCHICAL_DISCOVERY_MAX_ATOMS_PER_INTERPRET_JOB,
    HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB,
)
from .lossless_stage1_evidence_catalog import (
    DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK,
)
from .openai_compatible_json_discovery_job_runner import (
    MINIMUM_DISCOVERY_MAX_TOKENS,
    OPENAI_JSON_DISCOVERY_RUNNER_VERSION,
    OpenAICompatibleJsonDiscoveryJobRunner,
)
from .offline_hierarchical_discovery_review_packet import (
    AuthenticatedPromptFile,
    build_offline_extraction_definition_prompt_preview,
    compose_offline_hierarchical_discovery_review_packet,
)
from .staged_all_evidence_fusion_agent import StagedAllEvidenceFusionAgent
from .tfidf_topic_discovery import row_set_fingerprint

_BENCHMARK_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SERVER_MODE = "server"
_FUSION_ENABLE_THINKING = True
_FUSION_THINKING_TOKEN_BUDGET = 5000
_EXTRACTION_ENABLE_THINKING = False
_HIERARCHICAL_DISCOVERY_MODE = "hierarchical"
_LEGACY_STAGED_DISCOVERY_MODE = "legacy-staged-test-only"
_HIERARCHICAL_JOB_CACHE_DIRNAME = "hierarchical_job_cache"
_DEFAULT_FROZEN_REVIEW_MAX_EVIDENCE_IDS = 512
_DEFAULT_FROZEN_REVIEW_MAX_EVIDENCE_BYTES = 2_000_000
_ORACLE_ITE_COLUMN = "true_ite_prob"
_REVIEW_EMBEDDING_CACHE_FILES = (
    "metadata.json",
    "chunk_embeddings.npy",
    "offsets.npy",
    "chunk_texts.jsonl",
)
_EXPLICIT_DEVICE = re.compile(r"^(?:cpu|cuda:[0-9]+)$")
_REQUIRED_REVIEW_DISCOVERY_FAMILIES = frozenset(
    set(ALL_NON_QUERY_DISCOVERY_FAMILIES) | {NEURAL_QUERY_MOMENTS}
)
_FINAL_UPSTREAM_NAMESPACE = "all_evidence_upstream"
_FINAL_UPSTREAM_RAW_FAMILY_ROLE_KEYS = (
    ("bow_nuisance", PROPENSITY_NUISANCE_FEATURE_ROLE),
    ("bow_nuisance", OUTCOME_NUISANCE_FEATURE_ROLE),
    ("htr_nuisance", PROPENSITY_NUISANCE_FEATURE_ROLE),
    ("htr_nuisance", OUTCOME_NUISANCE_FEATURE_ROLE),
    ("bow_r_loss", UNCALIBRATED_EFFECT_MODIFIER_ROLE),
    ("htr_neural", UNCALIBRATED_EFFECT_MODIFIER_ROLE),
    ("matched_pair_uplift", UNCALIBRATED_EFFECT_MODIFIER_ROLE),
    ("embedding_whole_cohort", PROPENSITY_NUISANCE_FEATURE_ROLE),
    ("embedding_whole_cohort", OUTCOME_NUISANCE_FEATURE_ROLE),
    ("embedding_whole_cohort", UNCALIBRATED_EFFECT_MODIFIER_ROLE),
    ("embedding_clustered", PROPENSITY_NUISANCE_FEATURE_ROLE),
    ("embedding_clustered", UNCALIBRATED_EFFECT_MODIFIER_ROLE),
    ("tfidf_topics", PROPENSITY_NUISANCE_FEATURE_ROLE),
    ("tfidf_topics", OUTCOME_NUISANCE_FEATURE_ROLE),
    ("tfidf_topic_contrast", UNCALIBRATED_EFFECT_MODIFIER_ROLE),
    ("tfidf_orphan_ngrams", UNCALIBRATED_EFFECT_MODIFIER_ROLE),
    ("neural_query_treatment_moments", PROPENSITY_NUISANCE_FEATURE_ROLE),
    ("neural_query_outcome_moments", OUTCOME_NUISANCE_FEATURE_ROLE),
    ("neural_query_effect_moments", UNCALIBRATED_EFFECT_MODIFIER_ROLE),
)
_NEURAL_QUERY_BANK_BY_RAW_SOURCE_KIND = {
    "neural_query_treatment_moments": "treatment",
    "neural_query_outcome_moments": "outcome",
    "neural_query_effect_moments": "effect",
}


def _neural_query_moment_feature_names(bank: str, query_count: int) -> tuple[str, ...]:
    return (
        f"neural_query_{bank}_signed_mean",
        f"neural_query_{bank}_absolute_max",
        *(f"neural_query_{bank}_signed_order_{rank:02d}" for rank in range(1, query_count + 1)),
    )


@dataclass(frozen=True)
class ValidatedBenchmarkInputs:
    """Read-only validation result used to construct the live runner."""

    dataset_path: Path
    legacy_handoff_path: Path
    tfidf_handoff_path: Path
    primary_splits_path: Path
    output_dir: Path
    cache_index_paths: tuple[Path, ...]
    orphan_ngram_artifacts_by_fold: Mapping[int, TfidfOrphanNgramArtifact]
    row_count: int
    outer_folds: tuple[int, ...]
    neural_query_moment_artifacts_by_fold: Mapping[int, QueryEvidenceArtifact] = field(
        default_factory=dict
    )
    review_stage1_config_path: Path | None = None
    review_semantic_witness_scientific_config: (
        SemanticWitnessScientificConfig | None
    ) = None
    review_embedding_cache_dir: Path | None = None
    review_neural_query_cache_dir: Path | None = None
    authenticated_review_spent_cache_sources: tuple[AuthenticatedReviewSpentCacheSource, ...] = ()
    authenticated_context_fit_cache_sources: tuple[AuthenticatedContextFitCacheSource, ...] = ()
    hierarchical_preparation_dir: Path | None = None
    hierarchical_job_cache_root: Path | None = None
    hierarchical_offline_review_packet_dir: Path | None = None
    historical_discovery_prompt: AuthenticatedPromptFile | None = None
    old_hierarchy_prompt: AuthenticatedPromptFile | None = None


_SHARED_TFIDF_GRAPH_DEFAULT_SELECTION = "default_wrapped_no_context_fit_source_v1"
_SHARED_TFIDF_GRAPH_ATTESTED_SELECTION = "authenticated_context_fit_run_attestation_v1"


def _tfidf_graph_from_stable_backend_identity(value: Any, *, path: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} has no stable backend identity")
    child = value.get("child")
    if not isinstance(child, Mapping):
        raise ValueError(f"{path} has no composite child identity")
    members = child.get("members")
    if isinstance(members, (str, bytes, Mapping)) or not isinstance(members, Sequence):
        raise ValueError(f"{path} has no authenticated composite member identities")
    candidates = [
        member
        for member in members
        if isinstance(member, Mapping)
        and member.get("backend") in {TFIDF_CONTEXT_BACKEND_ID, SHARED_TFIDF_CONTEXT_BACKEND_ID}
    ]
    if len(candidates) != 1:
        raise ValueError(f"{path} must contain exactly one recognized TF-IDF context member")
    return classify_tfidf_context_member_identity(candidates[0])


def _select_tfidf_context_backend_graph(
    sources: Sequence[AuthenticatedContextFitCacheSource],
) -> tuple[str, str]:
    """Select the live graph from authenticated historical run attestations."""

    exact_sources = tuple(sources)
    if not exact_sources:
        return SHARED_TFIDF_RUNTIME_GRAPH_ID, _SHARED_TFIDF_GRAPH_DEFAULT_SELECTION
    graphs: list[str] = []
    for index, source in enumerate(exact_sources):
        run_attestation = getattr(source, "run_attestation", None)
        final_identity = getattr(run_attestation, "final_producer_identity", None)
        if not isinstance(final_identity, Mapping):
            raise ValueError(
                f"authenticated context-fit source {index} lacks final producer attestation"
            )
        graphs.append(
            _tfidf_graph_from_stable_backend_identity(
                final_identity.get("backend_identity"),
                path=f"authenticated_context_fit_sources[{index}]",
            )
        )
    distinct = set(graphs)
    if len(distinct) != 1:
        raise ValueError(
            "authenticated context-fit cache sources mix wrapped and unwrapped "
            "TF-IDF backend graphs"
        )
    return graphs[0], _SHARED_TFIDF_GRAPH_ATTESTED_SELECTION


def _shared_tfidf_runtime_audit(
    *,
    review_enabled: bool,
    graph: str,
    selection: str,
) -> dict[str, Any]:
    active_graph = graph if review_enabled else None
    return {
        "shared_tfidf_context_backend_graph": active_graph,
        "shared_tfidf_context_backend_graph_selection": (
            selection if review_enabled else "inactive_no_post_extraction_review"
        ),
        "shared_tfidf_context_fit_service_enabled": bool(
            review_enabled and graph == SHARED_TFIDF_RUNTIME_GRAPH_ID
        ),
        "shared_tfidf_disabled_to_preserve_authenticated_cache_identity": bool(
            review_enabled and graph == UNWRAPPED_TFIDF_RUNTIME_GRAPH_ID
        ),
        "shared_tfidf_disabled_to_preserve_authenticated_spent_cache_identity": False,
        "authenticated_spent_cache_influences_tfidf_graph_selection": False,
    }


def _validate_numeric_configuration(args: argparse.Namespace) -> None:
    if int(args.expected_outer_folds) < 2:
        raise ValueError("--expected-outer-folds must be at least 2")
    if int(args.interaction_inner_folds) < 2:
        raise ValueError("--interaction-inner-folds must be at least 2")
    if not 1 <= int(args.max_candidates) <= 64:
        raise ValueError("--max-candidates must be in [1, 64]")
    if int(args.proposal_max_tokens) < 1 or int(args.extraction_max_tokens) < 1:
        raise ValueError("proposal/extraction token limits must be positive")
    if (
        str(args.discovery_mode) == _HIERARCHICAL_DISCOVERY_MODE
        and int(args.proposal_max_tokens) < MINIMUM_DISCOVERY_MAX_TOKENS
    ):
        raise ValueError(
            "hierarchical --proposal-max-tokens must cover the authenticated "
            "visible response plus reasoning reserve: at least "
            f"{MINIMUM_DISCOVERY_MAX_TOKENS}"
        )
    if not 0 <= int(args.request_max_retries) <= 8:
        raise ValueError("--request-max-retries must be in [0, 8]")
    if int(args.extraction_batch_size) < 1:
        raise ValueError("--extraction-batch-size must be positive")
    if not 1 <= int(args.max_variables_per_extraction_request) <= 10:
        raise ValueError("--max-variables-per-extraction-request must be in [1, 10]")
    if int(args.extraction_max_text_length) < 1:
        raise ValueError("--extraction-max-text-length must be positive")
    if not 1 <= int(args.post_extraction_review_rounds) <= 8:
        raise ValueError("--post-extraction-review-rounds must be in [1, 8]")
    if (
        int(args.post_extraction_review_rounds) > 0
        and int(args.max_variables_per_extraction_request) != 1
    ):
        raise ValueError(
            "adaptive post-extraction review requires "
            "--max-variables-per-extraction-request 1 so changed-only extraction "
            "has contract-local request semantics"
        )
    if not 1 <= int(args.post_extraction_review_max_operations) <= 32:
        raise ValueError("--post-extraction-review-max-operations must be in [1, 32]")
    if not 0 <= int(args.post_extraction_review_max_quality_retries) <= 8:
        raise ValueError("--post-extraction-review-max-quality-retries must be in [0, 8]")
    if int(args.post_extraction_review_min_partition_rows) < 2:
        raise ValueError("--post-extraction-review-min-partition-rows must be at least 2")
    if int(args.review_neural_query_nuisance_folds) < 2:
        raise ValueError("--review-neural-query-nuisance-folds must be at least 2")
    bow_workers = args.review_stage1_bow_fold_parallelism
    if (
        isinstance(bow_workers, (bool, np.bool_))
        or not isinstance(bow_workers, (int, np.integer))
        or int(bow_workers) < 1
    ):
        raise ValueError("--review-stage1-bow-fold-parallelism must be a positive integer")
    meta_folds = args.final_upstream_meta_inner_folds
    if (
        isinstance(meta_folds, (bool, np.bool_))
        or not isinstance(meta_folds, (int, np.integer))
        or int(meta_folds) < 2
    ):
        raise ValueError("--final-upstream-meta-inner-folds must be an integer at least 2")
    head_regularization = args.final_upstream_head_regularization
    if (
        isinstance(head_regularization, (bool, np.bool_))
        or not isinstance(head_regularization, (int, float, np.integer, np.floating))
        or not np.isfinite(float(head_regularization))
        or float(head_regularization) <= 0.0
    ):
        raise ValueError("--final-upstream-head-regularization must be positive and finite")
    _review_stage1_device(args)
    _review_neural_query_devices(args)
    if int(args.post_extraction_review_rounds) > 0:
        _final_upstream_max_orphan_features(args)
        if args.review_stage1_config is None:
            raise ValueError(
                "--review-stage1-config is required when post-extraction review is enabled"
            )
        if getattr(args, "review_semantic_witness_scientific_config", None) is None:
            raise ValueError(
                "--review-semantic-witness-scientific-config is required when "
                "post-extraction review is enabled"
            )
        if args.review_embedding_cache_dir is None:
            raise ValueError(
                "--review-embedding-cache-dir is required when post-extraction review is enabled"
            )
        if str(args.outcome_type).strip().lower() != "binary":
            raise ValueError(
                "adaptive review final-upstream production currently requires a binary "
                "outcome because matched-pair uplift is a required family"
            )
        if not bool(args.modifier_interactions_only):
            raise ValueError(
                "adaptive review with the high-dimensional role-aware upstream bank "
                "requires --modifier-interactions-only"
            )
    if (
        str(args.extraction_context_strategy) == "contract_lexical_rag"
        and int(args.extraction_max_text_length) < 256
    ):
        raise ValueError(
            "--extraction-max-text-length must be at least 256 for contract_lexical_rag"
        )
    if str(args.extraction_prompt_version) != EXTRACTION_PROMPT_VERSION:
        raise ValueError(
            "--extraction-prompt-version must match the provider's actual prompt version "
            f"{EXTRACTION_PROMPT_VERSION!r}"
        )

    if str(args.discovery_mode) == _HIERARCHICAL_DISCOVERY_MODE:
        # Construction performs the complete closed-bound validation.  Keeping
        # this here makes dry-run and live execution share one exact policy.
        build_hierarchical_discovery_config(args)
        build_frozen_review_evidence_policy(args)


def _final_upstream_max_orphan_features(args: argparse.Namespace) -> int:
    value = getattr(args, "final_upstream_max_orphan_features", None)
    if value is None:
        raise ValueError(
            "--final-upstream-max-orphan-features is required when "
            "post-extraction review is enabled; there is no production default"
        )
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("--final-upstream-max-orphan-features must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError("--final-upstream-max-orphan-features must be positive")
    return result


def build_hierarchical_discovery_config(
    args: argparse.Namespace,
) -> HierarchicalDiscoveryConfig:
    """Build the closed architecture-at-a-time discovery policy."""

    return HierarchicalDiscoveryConfig(
        max_rendered_prompt_bytes=MAX_RENDERED_DISCOVERY_PROMPT_BYTES,
        max_semantic_member_ids_per_chunk=int(args.hierarchical_max_semantic_member_ids_per_chunk),
        max_cross_architecture_lookback_ids_per_group=int(
            args.hierarchical_max_cross_architecture_lookback_ids
        ),
        max_cross_architecture_lookback_bytes_per_group=int(
            args.hierarchical_max_cross_architecture_lookback_bytes
        ),
        max_extraction_lookback_ids_per_feature=int(
            args.hierarchical_max_extraction_lookback_ids_per_feature
        ),
        max_extraction_lookback_bytes_per_feature=int(
            args.hierarchical_max_extraction_lookback_bytes_per_feature
        ),
        max_rejection_lookback_ids_per_candidate=int(
            args.hierarchical_max_rejection_lookback_ids_per_candidate
        ),
        max_rejection_lookback_bytes_per_candidate=int(
            args.hierarchical_max_rejection_lookback_bytes_per_candidate
        ),
        max_integrated_features=int(args.max_candidates),
    )


def build_frozen_review_evidence_policy(
    args: argparse.Namespace,
) -> FrozenReviewEvidencePolicyBinding:
    """Bind review to exact support of hierarchy-accepted features only."""

    adaptive_config = AdaptiveReconsiderationConfig(
        max_atoms_per_chunk=int(args.hierarchical_max_atoms_per_chunk),
        max_bytes_per_chunk=int(args.hierarchical_max_bytes_per_chunk),
        max_semantic_member_ids_per_chunk=int(args.hierarchical_max_semantic_member_ids_per_chunk),
        max_operations=int(args.post_extraction_review_max_operations),
    )
    return FrozenReviewEvidencePolicyBinding(
        max_evidence_ids=int(args.hierarchical_review_max_evidence_ids),
        max_evidence_bytes=int(args.hierarchical_review_max_evidence_bytes),
        review_materializer_identity=frozen_hierarchical_review_evidence_identity(),
        adaptive_reconsideration_identity=(
            adaptive_hierarchical_stage1_reconsideration_identity(adaptive_config)
        ),
        accepted_support_only=True,
    )


def extraction_prompt_cache_identity(args: argparse.Namespace) -> str:
    """Bind every prompt, packing, and context semantic into overlay identity."""
    base = str(args.extraction_prompt_version)
    semantics = {
        "prompt_template_version": base,
        "grouping_strategy": str(args.extraction_grouping_strategy),
        "grouping_version": EXTRACTION_GROUPING_VERSION,
        "max_variables_per_request": int(args.max_variables_per_extraction_request),
        "context_strategy": str(args.extraction_context_strategy),
        "context_compactor_version": CONTRACT_LEXICAL_CONTEXT_VERSION,
        "max_text_length": int(args.extraction_max_text_length),
        "vllm_enable_thinking": _EXTRACTION_ENABLE_THINKING,
        "source_text_temporal_policy": source_text_temporal_policy_audit(),
    }
    digest = hashlib.sha256(
        json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"{base}+extraction_semantics:{digest}"


def _canonical_ip_literal(hostname: str) -> str | None:
    """Return one canonical address for standard and legacy numeric hosts."""

    candidate = str(hostname).strip()
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        # Browsers and libc commonly accept historical IPv4 spellings such as
        # ``127.1``, a single decimal integer, octal components, and hex.  They
        # must not fall through and be treated as innocuous DNS names.
        if ":" in candidate or not re.fullmatch(r"[0-9A-Fa-fXx.]+", candidate):
            return None
        try:
            address = ipaddress.ip_address(socket.inet_aton(candidate))
        except OSError:
            return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return str(address)


def _resolve_host_addresses(hostname: str) -> frozenset[str]:
    """Resolve a hostname to canonical IPs, returning empty on lookup failure."""

    try:
        records = socket.getaddrinfo(
            hostname,
            None,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except OSError:
        return frozenset()
    addresses = {
        canonical
        for record in records
        if record[4]
        for canonical in (_canonical_ip_literal(str(record[4][0])),)
        if canonical is not None
    }
    return frozenset(addresses)


def _current_host_interface_addresses(machine_names: Sequence[str]) -> frozenset[str]:
    """Return addresses assigned locally or advertised for this host's names."""

    addresses: set[str] = set()
    try:
        interface_map = psutil.net_if_addrs()
    except OSError:
        # Restricted containers can forbid the underlying netlink query.  The
        # hostname-resolution facts below remain available as a conservative
        # fallback and endpoint aliases are still checked by their own lookup.
        interface_map = {}
    for interface_addresses in interface_map.values():
        for interface_address in interface_addresses:
            if interface_address.family not in {socket.AF_INET, socket.AF_INET6}:
                continue
            canonical = _canonical_ip_literal(str(interface_address.address).split("%", 1)[0])
            if canonical is not None:
                addresses.add(canonical)
    for name in dict.fromkeys(str(value).strip() for value in machine_names if value):
        addresses.update(_resolve_host_addresses(name))
    return frozenset(addresses)


def validate_remote_endpoint_pool(value: Any) -> str:
    """Validate and normalize a remote OpenAI-compatible endpoint pool.

    Loopback and wildcard hosts are rejected by parsed hostname, rather than a
    substring check that could be evaded with credentials or URL casing.
    Private network names remain valid because an explicitly supplied cluster
    host (for example ``camus``) is still remote from this process.
    """

    urls = parse_server_urls(value, default="")
    if not urls:
        raise ValueError("at least one remote endpoint is required")
    machine_hostname = socket.gethostname().rstrip(".").lower()
    machine_fqdn = socket.getfqdn().rstrip(".").lower()
    local_machine_names = {
        value
        for value in (
            machine_hostname,
            machine_hostname.split(".", 1)[0],
            machine_fqdn,
            machine_fqdn.split(".", 1)[0],
        )
        if value
    }
    local_interface_addresses = _current_host_interface_addresses(
        tuple(sorted(local_machine_names))
    )
    normalized: list[str] = []
    for raw_url in urls:
        url = str(raw_url).strip()
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"remote endpoint must be an absolute HTTP(S) URL: {url!r}")
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if not hostname:
            raise ValueError(f"remote endpoint has no hostname: {url!r}")
        is_local_name = hostname == "localhost" or hostname.endswith(".localhost")
        literal_address = _canonical_ip_literal(hostname)
        resolved_addresses = (
            frozenset((literal_address,))
            if literal_address is not None
            else _resolve_host_addresses(hostname)
        )
        is_local_address = any(
            (address := ipaddress.ip_address(value)).is_loopback or address.is_unspecified
            for value in resolved_addresses
        )
        is_current_machine = hostname in local_machine_names or bool(
            resolved_addresses & local_interface_addresses
        )
        if is_local_name or is_local_address or is_current_machine:
            raise ValueError(
                "proposal/extraction endpoint must be remote; the current machine, "
                "localhost, loopback, and wildcard listener addresses are forbidden"
            )
        normalized.append(url.rstrip("/"))
    return ",".join(dict.fromkeys(normalized))


def _review_stage1_device(args: argparse.Namespace) -> str:
    device = str(args.review_stage1_device).strip()
    if not _EXPLICIT_DEVICE.fullmatch(device):
        raise ValueError("--review-stage1-device must be 'cpu' or an explicit cuda:N device")
    return device


def _review_neural_query_devices(args: argparse.Namespace) -> tuple[str, ...]:
    raw_devices = tuple(args.review_neural_query_device or ("cuda:0",))
    devices = tuple(dict.fromkeys(str(value).strip() for value in raw_devices))
    if not devices or any(not _EXPLICIT_DEVICE.fullmatch(value) for value in devices):
        raise ValueError(
            "--review-neural-query-device must be repeated with 'cpu' or explicit cuda:N values"
        )
    return devices


def build_agent_config(args: argparse.Namespace) -> AgenticFeatureSearchConfig:
    """Build the fixed reasoning-enabled proposal/selection configuration."""

    _validate_numeric_configuration(args)
    endpoint = validate_remote_endpoint_pool(args.endpoint)
    model = str(args.model or "").strip()
    if not model:
        raise ValueError("--model must be non-empty")
    return AgenticFeatureSearchConfig(
        outer_folds=max(2, int(args.expected_outer_folds)),
        inner_folds=max(2, int(args.interaction_inner_folds)),
        max_iterations=1,
        max_additions_per_iter=int(args.max_candidates),
        agent_server_url=endpoint,
        agent_model_name=model,
        agent_api_key=str(args.api_key),
        agent_temperature=0.0,
        agent_max_tokens=int(args.proposal_max_tokens),
        agent_enable_thinking=_FUSION_ENABLE_THINKING,
        agent_thinking_token_budget=_FUSION_THINKING_TOKEN_BUDGET,
        agent_schema_repair_attempts=int(args.proposal_schema_repair_attempts),
        agent_request_max_retries=int(args.request_max_retries),
        agent_request_timeout=float(args.request_timeout),
        agent_provider="openai",
        save_agent_context=False,
        save_agent_raw_output=False,
    )


def build_applied_inference_config(
    args: argparse.Namespace,
    *,
    vllm_mode: str = _SERVER_MODE,
) -> AppliedInferenceConfig:
    """Build extraction config while making non-server execution impossible."""

    if str(vllm_mode).strip().lower() != _SERVER_MODE:
        raise ValueError("remote-only benchmark extraction requires vllm_mode='server'")
    _validate_numeric_configuration(args)
    endpoint = validate_remote_endpoint_pool(args.endpoint)
    model = str(args.model or "").strip()
    if not model:
        raise ValueError("--model must be non-empty")
    cache_dir = Path(args.output_dir).expanduser().resolve() / "current_extraction_cache"
    explicit = ExplicitFeatureExtractionConfig(
        enabled=True,
        features=[],
        vllm_mode=_SERVER_MODE,
        vllm_server_url=endpoint,
        vllm_model_name=model,
        vllm_api_key=str(args.api_key),
        vllm_tensor_parallel_size=1,
        vllm_enable_thinking=_EXTRACTION_ENABLE_THINKING,
        extraction_batch_size=int(args.extraction_batch_size),
        max_variables_per_extraction_request=int(args.max_variables_per_extraction_request),
        extraction_max_retries=int(args.request_max_retries),
        extraction_request_timeout=float(args.request_timeout),
        extraction_temperature=0.0,
        extraction_max_tokens=int(args.extraction_max_tokens),
        extraction_max_text_length=int(args.extraction_max_text_length),
        extraction_grouping_strategy=str(args.extraction_grouping_strategy),
        extraction_context_strategy=str(args.extraction_context_strategy),
        extraction_provider="openai",
        source_text_temporally_valid_by_design=True,
        cache_enabled=True,
        cache_dir=str(cache_dir),
    )
    config = AppliedInferenceConfig(
        outcome_type=str(args.outcome_type),
        dataset_path=str(Path(args.dataset).expanduser().resolve()),
        text_column=str(args.text_column),
        treatment_column=str(args.treatment_column),
        outcome_column=str(args.outcome_column),
        explicit_features=explicit,
    )
    assert config.explicit_features.vllm_mode == _SERVER_MODE
    return config


def _validate_discovery_cli_mode(args: argparse.Namespace) -> None:
    mode = str(args.discovery_mode)
    hierarchical = mode == _HIERARCHICAL_DISCOVERY_MODE
    if mode not in {_HIERARCHICAL_DISCOVERY_MODE, _LEGACY_STAGED_DISCOVERY_MODE}:
        raise ValueError(f"unsupported discovery mode: {mode!r}")
    if bool(args.prepare_hierarchical_discovery) and not hierarchical:
        raise ValueError("--prepare-hierarchical-discovery requires hierarchical mode")
    approved = args.hierarchical_approved_batch_sha256
    if approved is not None:
        raw_approved = str(approved).strip()
        if re.fullmatch(r"[0-9a-f]{64}", raw_approved) is None:
            raise ValueError("--hierarchical-approved-batch-sha256 must be one lowercase SHA-256")
        args.hierarchical_approved_batch_sha256 = raw_approved
    if bool(args.prepare_hierarchical_discovery) and approved is not None:
        raise ValueError(
            "prepare-only mode cannot also provide --hierarchical-approved-batch-sha256"
        )
    if bool(args.prepare_hierarchical_discovery) and (
        args.historical_discovery_prompt is None or args.old_hierarchy_prompt is None
    ):
        raise ValueError(
            "prepare-only mode requires both --historical-discovery-prompt "
            "PATH::SHA256 and --old-hierarchy-prompt PATH::SHA256"
        )
    if bool(args.dry_run) and approved is not None:
        raise ValueError("--dry-run cannot approve a hierarchical batch")
    if bool(args.dry_run) and bool(args.prepare_hierarchical_discovery):
        raise ValueError("--dry-run and --prepare-hierarchical-discovery are mutually exclusive")
    hierarchy_resource_options = {
        "--hierarchical-job-cache-max-entry-bytes": (
            args.hierarchical_job_cache_max_entry_bytes
        ),
        "--first-untouched-gate-max-initial-spent-rows": (
            args.first_untouched_gate_max_initial_spent_rows
        ),
        "--first-untouched-gate-max-first-gate-rows": (
            args.first_untouched_gate_max_first_gate_rows
        ),
        "--first-untouched-gate-max-total-text-utf8-bytes": (
            args.first_untouched_gate_max_total_text_utf8_bytes
        ),
        "--first-untouched-gate-max-catalog-atoms": (
            args.first_untouched_gate_max_catalog_atoms
        ),
        "--first-untouched-gate-max-source-manifest-bytes": (
            args.first_untouched_gate_max_source_manifest_bytes
        ),
        "--first-untouched-gate-max-direct-numerical-signals": (
            args.first_untouched_gate_max_direct_numerical_signals
        ),
        "--first-untouched-gate-max-single-matrix-file-bytes": (
            args.first_untouched_gate_max_single_matrix_file_bytes
        ),
        "--first-untouched-gate-max-total-matrix-file-bytes": (
            args.first_untouched_gate_max_total_matrix_file_bytes
        ),
    }
    if hierarchical:
        missing = sorted(
            name
            for name, value in hierarchy_resource_options.items()
            if value is None
        )
        if missing:
            raise ValueError(
                "hierarchical discovery requires explicit resource bounds: "
                + ", ".join(missing)
            )
        invalid = sorted(
            name
            for name, value in hierarchy_resource_options.items()
            if isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        )
        if invalid:
            raise ValueError(
                "hierarchical resource bounds must be positive integers: "
                + ", ".join(invalid)
            )
    if not hierarchical:
        unexpected = {
            "--hierarchical-preparation-dir": args.hierarchical_preparation_dir,
            "--hierarchical-job-cache-root": args.hierarchical_job_cache_root,
            "--hierarchical-approved-batch-sha256": approved,
            "--historical-discovery-prompt": args.historical_discovery_prompt,
            "--old-hierarchy-prompt": args.old_hierarchy_prompt,
            "--hierarchical-offline-review-packet-dir": (
                args.hierarchical_offline_review_packet_dir
            ),
            **hierarchy_resource_options,
        }
        present = sorted(name for name, value in unexpected.items() if value is not None)
        if present:
            raise ValueError(
                "legacy staged discovery does not accept hierarchical runtime options: "
                + ", ".join(present)
            )


def _required_file(path: Path | str, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def _load_semantic_witness_scientific_config(
    path: Path | str,
) -> SemanticWitnessScientificConfig:
    requested = _required_file(
        path,
        label="review semantic-witness scientific config",
    )
    before = sha256_file(requested)

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(
                    f"review semantic-witness config contains duplicate key {key!r}"
                )
            output[key] = value
        return output

    try:
        raw = json.loads(
            requested.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(
                    "review semantic-witness config contains non-finite "
                    f"JSON value {token}"
                )
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "review semantic-witness scientific config is not strict JSON"
        ) from exc
    after = sha256_file(requested)
    if before != after:
        raise RuntimeError(
            "review semantic-witness scientific config changed while being parsed"
        )
    return SemanticWitnessScientificConfig.from_mapping(
        raw,
        label="review semantic-witness scientific config",
    )


def _required_directory(path: Path | str, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {resolved}")
    return resolved


def _authenticated_prompt_registration(
    value: str | None,
    *,
    label: str,
) -> AuthenticatedPromptFile | None:
    """Parse and immediately authenticate one exact ``PATH::SHA256`` prompt."""

    if value is None:
        return None
    raw_path, separator, raw_sha256 = str(value).rpartition("::")
    if not separator or not raw_path:
        raise ValueError(f"{label} must use PATH::SHA256")
    sha256 = raw_sha256.strip()
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError(f"{label} must declare one lowercase SHA-256")
    path = _required_file(raw_path, label=label)
    registration = AuthenticatedPromptFile(
        path=path,
        expected_sha256=sha256,
        display_name=path.name,
    )
    # Re-read during packet composition so mutation after validation fails too.
    registration.snapshot(artifact_kind="cli_prevalidation_only")
    return registration


def build_review_neural_query_config(
    args: argparse.Namespace,
) -> NeuralQueryAgenticForestConfig:
    """Load an optional closed JSON override for context-fitted query moments."""

    config_path = args.review_neural_query_config
    if config_path is None:
        config = NeuralQueryAgenticForestConfig()
    else:
        path = _required_file(config_path, label="review neural-query config")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("review neural-query config must be valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("review neural-query config must be one JSON object")
        allowed = {item.name for item in dataclass_fields(NeuralQueryAgenticForestConfig)}
        unknown = sorted(set(map(str, payload)) - allowed)
        if unknown:
            raise ValueError(
                "review neural-query config contains unknown fields: " + ", ".join(unknown)
            )
        try:
            config = NeuralQueryAgenticForestConfig(**dict(payload))
        except TypeError as exc:
            raise ValueError("review neural-query config has invalid field values") from exc
    try:
        config.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"review neural-query config is invalid: {exc}") from exc
    return config


def build_final_upstream_schema_config(
    stage1_config_path: Path | str,
    *,
    stage1_config_snapshot: HistoricalStage1ConfigSnapshot | None = None,
    neural_query_config: NeuralQueryAgenticForestConfig | None = None,
    signed_order_width: int,
) -> CrossFitStableUpstreamSchemaConfig:
    """Precommit one shared gate/final schema from exact upstream configs."""

    path = _required_file(stage1_config_path, label="historical Stage-1 review config")
    snapshot = stage1_config_snapshot or HistoricalStage1ConfigSnapshot.from_path(path)
    if snapshot.source_path != path:
        raise ValueError("final upstream schema config path does not match exact snapshot")
    snapshot.verify_source()
    applied = snapshot.applied_config()
    views = tuple(applied.architecture.multi_model_forest.bow_views)
    if not views:
        raise ValueError("historical Stage-1 config contains no configured BoW views")
    view_names = tuple(str(view.name).strip() for view in views)
    if any(not name for name in view_names) or len(view_names) != len(set(view_names)):
        raise ValueError("historical Stage-1 BoW view names must be non-empty and unique")
    query_config = neural_query_config or NeuralQueryAgenticForestConfig()
    if not isinstance(query_config, NeuralQueryAgenticForestConfig):
        raise TypeError("neural_query_config must be NeuralQueryAgenticForestConfig")
    query_config.validate()
    if isinstance(signed_order_width, (bool, np.bool_)) or not isinstance(
        signed_order_width, (int, np.integer)
    ):
        raise TypeError("signed_order_width must be an integer")
    stable_width = int(signed_order_width)
    if stable_width < 1:
        raise ValueError("signed_order_width must be positive")

    calibrated_sources = tuple(
        PrecommittedCalibratedSource(
            child_name=(f"stage1_calibrated__bow__{view_name}__effect_weighted_r_tau_pred"),
            source_kind="nested_calibrated_bow_weighted_r",
        )
        for view_name in view_names
    ) + (
        PrecommittedCalibratedSource(
            child_name="stage1_calibrated__htr__effect_weighted_r_tau_pred",
            source_kind="nested_calibrated_htr_weighted_r",
        ),
    )
    raw_families = tuple(
        PrecommittedRawFeatureFamily(
            source_kind=source_kind,
            consumer_role=consumer_role,
            signed_order_width=(
                query_config.query_count(_NEURAL_QUERY_BANK_BY_RAW_SOURCE_KIND[source_kind])
                if source_kind in _NEURAL_QUERY_BANK_BY_RAW_SOURCE_KIND
                else stable_width
            ),
            required=True,
            exact_passthrough_feature_names=(
                _neural_query_moment_feature_names(
                    _NEURAL_QUERY_BANK_BY_RAW_SOURCE_KIND[source_kind],
                    query_config.query_count(_NEURAL_QUERY_BANK_BY_RAW_SOURCE_KIND[source_kind]),
                )
                if source_kind in _NEURAL_QUERY_BANK_BY_RAW_SOURCE_KIND
                else ()
            ),
        )
        for source_kind, consumer_role in _FINAL_UPSTREAM_RAW_FAMILY_ROLE_KEYS
    )
    return CrossFitStableUpstreamSchemaConfig(
        namespace=_FINAL_UPSTREAM_NAMESPACE,
        calibrated_sources=calibrated_sources,
        raw_families=raw_families,
        reject_unconfigured_calibrated_sources=True,
        reject_unconfigured_raw_families=True,
        source_config_sha256=snapshot.sha256,
    )


def build_coordinate_preserving_final_upstream_schema_config(
    stage1_config_path: Path | str,
    *,
    stage1_config_snapshot: HistoricalStage1ConfigSnapshot | None = None,
    neural_query_config: NeuralQueryAgenticForestConfig | None = None,
    max_orphan_features: int,
) -> CoordinatePreservingUpstreamSchemaConfig:
    """Precommit the production v3 schema without fitting or reading patient rows."""

    path = _required_file(stage1_config_path, label="historical Stage-1 review config")
    snapshot = stage1_config_snapshot or HistoricalStage1ConfigSnapshot.from_path(path)
    if snapshot.source_path != path:
        raise ValueError("final upstream schema config path does not match exact snapshot")
    snapshot.verify_source()
    applied = snapshot.applied_config()
    forest = applied.architecture.multi_model_forest
    required_flags = {
        "bow_discovery_enabled": bool(forest.bow_discovery_enabled),
        "htr_evidence_enabled": bool(forest.htr_evidence_enabled),
        "embedding_contrast.enabled": bool(forest.embedding_contrast.enabled),
        "embedding_contrast.include_cluster_contrast_vectors": bool(
            forest.embedding_contrast.include_cluster_contrast_vectors
        ),
        "matched_pair_uplift_enabled": bool(forest.matched_pair_uplift_enabled),
        "matched_pair_bow_enabled": bool(forest.matched_pair_bow_enabled),
        "matched_pair_htr_enabled": bool(forest.matched_pair_htr_enabled),
    }
    disabled = sorted(name for name, enabled in required_flags.items() if not enabled)
    if disabled:
        raise ValueError(
            "all-architecture coordinate registry requires enabled Stage-1 paths: "
            + ", ".join(disabled)
        )
    if bool(applied.architecture.htr_freeze_sentence_encoder):
        raise ValueError(
            "the all-architecture benchmark requires the HTR sentence encoder to be trainable"
        )
    views = tuple(str(view.name).strip() for view in forest.bow_views)
    query_config = neural_query_config or NeuralQueryAgenticForestConfig()
    if not isinstance(query_config, NeuralQueryAgenticForestConfig):
        raise TypeError("neural_query_config must be NeuralQueryAgenticForestConfig")
    query_config.validate()
    return build_production_coordinate_preserving_schema(
        namespace=_FINAL_UPSTREAM_NAMESPACE,
        bow_view_names=views,
        source_config_sha256=snapshot.sha256,
        cluster_max_components=int(forest.embedding_contrast.cluster_contrast_max_components),
        tfidf_topic_count=int(forest.tfidf_topic.topic_count),
        max_orphan_features=max_orphan_features,
        neural_query_counts={
            bank: query_config.query_count(bank) for bank in ("treatment", "outcome", "effect")
        },
    )


def _validate_review_embedding_cache(
    cache_dir: Path | str,
    *,
    expected_row_count: int,
) -> Path:
    resolved = _required_directory(cache_dir, label="review frozen embedding cache")
    for filename in _REVIEW_EMBEDDING_CACHE_FILES:
        _required_file(
            resolved / filename,
            label=f"review frozen embedding cache {filename}",
        )
    try:
        metadata = json.loads((resolved / "metadata.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("review frozen embedding cache metadata is invalid JSON") from exc
    try:
        row_count = int(metadata.get("num_samples", -1)) if isinstance(metadata, Mapping) else -1
    except (TypeError, ValueError):
        row_count = -1
    if row_count != int(expected_row_count):
        raise ValueError("review frozen embedding cache row count does not match the dataset")
    return resolved


def _validated_output_path(path: Path | str) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output}")
    parent = output
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        raise ValueError(f"output path has no existing directory ancestor: {output}")
    return output


def _validated_hierarchical_layout(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Resolve the separate resumable preparation and stable job-cache roots."""

    if args.hierarchical_preparation_dir is None:
        raise ValueError("hierarchical discovery requires --hierarchical-preparation-dir")
    preparation = _validated_output_path(args.hierarchical_preparation_dir)
    output = output_dir.resolve()
    if (
        preparation == output
        or preparation.is_relative_to(output)
        or output.is_relative_to(preparation)
    ):
        raise ValueError(
            "--hierarchical-preparation-dir and --output-dir must be separate "
            "non-nested directories"
        )
    requested_cache = args.hierarchical_job_cache_root
    job_cache_root = _validated_output_path(
        requested_cache
        if requested_cache is not None
        else preparation / _HIERARCHICAL_JOB_CACHE_DIRNAME
    )
    if job_cache_root == preparation or not job_cache_root.is_relative_to(preparation):
        raise ValueError(
            "--hierarchical-job-cache-root must be a child of " "--hierarchical-preparation-dir"
        )
    review_packet_dir = _validated_output_path(
        args.hierarchical_offline_review_packet_dir
        if args.hierarchical_offline_review_packet_dir is not None
        else preparation / "offline_review_packet"
    )
    if review_packet_dir.parent != preparation:
        raise ValueError(
            "--hierarchical-offline-review-packet-dir must be a direct child of "
            "--hierarchical-preparation-dir"
        )
    if bool(args.prepare_hierarchical_discovery) and review_packet_dir.exists():
        raise ValueError(
            "--hierarchical-offline-review-packet-dir must be absent before "
            "prepare-only immutable persistence"
        )
    return preparation, job_cache_root, review_packet_dir


def _validate_fresh_review_neural_query_cache(
    path: Path | str,
    *,
    cache_parent: Path,
    require_empty: bool = True,
    parent_option: str = "--output-dir",
) -> Path:
    """Bind the executable query cache to one explicit local cache directory.

    Unlike the frozen embedding input, this cache contains executable joblib
    checkpoints produced while the adaptive review is running.  It is
    therefore never accepted outside its explicitly selected parent.  Every
    live invocation uses a fresh output-local directory.  Cross-process
    hierarchical replay uses separately authenticated read-only overlays,
    never a previously populated executable neural-query cache.
    """

    parent = _validated_output_path(cache_parent)
    raw = Path(path).expanduser()
    cursor = raw if raw.is_absolute() else Path.cwd() / raw
    while True:
        if cursor.is_symlink():
            raise ValueError("--review-neural-query-cache-dir cannot contain a symlink component")
        if cursor.resolve() == parent or cursor.parent == cursor:
            break
        cursor = cursor.parent
    cache = _validated_output_path(raw)
    if cache.parent != parent:
        raise ValueError(
            "--review-neural-query-cache-dir must be a direct child of "
            f"{'the fresh ' if parent_option == '--output-dir' else ''}{parent_option}"
        )
    if require_empty and cache.exists() and any(cache.iterdir()):
        raise ValueError(
            "--review-neural-query-cache-dir must be nonexistent or empty; "
            "pre-populated executable checkpoints are forbidden"
        )
    return cache


def _validate_fresh_benchmark_output(
    output_dir: Path,
    *,
    allowed_empty_cache_dir: Path | None = None,
) -> None:
    """Reject all prior output except the explicitly validated empty cache root."""

    if not output_dir.exists():
        return
    entries = tuple(output_dir.iterdir())
    if not entries:
        return
    if (
        allowed_empty_cache_dir is not None
        and allowed_empty_cache_dir.exists()
        and entries == (allowed_empty_cache_dir,)
        and not any(allowed_empty_cache_dir.iterdir())
    ):
        return
    raise ValueError(
        "--output-dir must be nonexistent or empty because live extraction/cache "
        "artifacts are not accepted as authenticated inputs"
    )


def parse_orphan_ngram_artifact_registrations(
    entries: Sequence[str],
) -> dict[int, TfidfOrphanNgramArtifact]:
    """Parse and authenticate repeatable ``FOLD=PATH[::SHA256]`` overrides."""

    registrations: dict[int, TfidfOrphanNgramArtifact] = {}
    for raw_entry in entries:
        entry = str(raw_entry).strip()
        if "=" not in entry:
            raise ValueError("--orphan-ngram-artifact must use FOLD=PATH or FOLD=PATH::SHA256")
        raw_fold, raw_artifact = entry.split("=", 1)
        try:
            fold = int(raw_fold)
        except ValueError as exc:
            raise ValueError("orphan n-gram artifact fold must be an integer") from exc
        if fold < 1:
            raise ValueError("orphan n-gram artifact fold must be positive")
        if fold in registrations:
            raise ValueError(f"duplicate orphan n-gram artifact registration for fold {fold}")
        raw_path, separator, declared_sha = raw_artifact.rpartition("::")
        if not separator:
            raw_path = raw_artifact
            declared_sha = ""
        artifact_path = _required_file(raw_path, label=f"fold {fold} orphan n-gram artifact")
        actual_sha = sha256_file(artifact_path)
        if declared_sha and declared_sha.strip().lower() != actual_sha:
            raise ValueError(f"orphan n-gram artifact SHA-256 mismatch for fold {fold}")
        registrations[fold] = TfidfOrphanNgramArtifact(
            path=artifact_path,
            artifact_sha256=actual_sha,
        )
    return registrations


def parse_neural_query_moment_artifact_registrations(
    entries: Sequence[str],
    *,
    full_outer_rows_by_fold: Mapping[int, Mapping[str, Any]],
    require_declared_partition: bool = False,
) -> dict[int, QueryEvidenceArtifact]:
    """Authenticate learned neural query evidence against authoritative folds.

    A historical bare ``query_evidence.json`` has no complete partition
    declaration, so the CLI binds its bytes to the resealed TF-IDF registry and
    validates every retrieved row through the strict query-evidence adapter.
    New fold-scoped bundles may additionally declare their complete fit and
    held-out row lists; the same adapter checks those lists exactly.
    """

    registrations: dict[int, QueryEvidenceArtifact] = {}
    for raw_entry in entries:
        entry = str(raw_entry).strip()
        if "=" not in entry:
            raise ValueError(
                "--neural-query-moment-artifact must use " "FOLD=PATH or FOLD=PATH::SHA256"
            )
        raw_fold, raw_artifact = entry.split("=", 1)
        try:
            fold = int(raw_fold)
        except ValueError as exc:
            raise ValueError("neural query-moment artifact fold must be an integer") from exc
        if fold < 1:
            raise ValueError("neural query-moment artifact fold must be positive")
        if fold in registrations:
            raise ValueError(f"duplicate neural query-moment artifact registration for fold {fold}")
        if fold not in full_outer_rows_by_fold:
            raise ValueError(
                f"neural query-moment artifact registration has unknown outer fold {fold}"
            )
        raw_path, separator, declared_sha = raw_artifact.rpartition("::")
        if not separator:
            raw_path = raw_artifact
            declared_sha = ""
        artifact_path = _required_file(
            raw_path,
            label=f"fold {fold} neural query-moment artifact",
        )
        actual_sha = sha256_file(artifact_path)
        if declared_sha and declared_sha.strip().lower() != actual_sha:
            raise ValueError(f"neural query-moment artifact SHA-256 mismatch for fold {fold}")
        full = full_outer_rows_by_fold[fold]
        fit_ids = tuple(map(int, full.get("fit_row_ids") or ()))
        heldout_ids = tuple(map(int, full.get("heldout_row_ids") or ()))
        provenance = FoldEvidenceProvenance(
            outer_fold=fold,
            train_row_ids=fit_ids,
            heldout_row_ids=heldout_ids,
            scope="outer_train",
            artifact_id=f"cli-neural-query-moments-{fold}",
        )
        registration = QueryEvidenceArtifact(
            path=artifact_path,
            outer_fold=fold,
            artifact_sha256=actual_sha,
            fit_row_fingerprint=row_set_fingerprint(fit_ids),
            heldout_row_fingerprint=row_set_fingerprint(heldout_ids),
            scope="outer_train",
        )
        # Validate now for side-effect-free dry runs, then validate again in the
        # live runner so a post-validation mutation cannot enter a prompt.
        adapted = load_query_moment_evidence_artifact(
            artifact_path,
            provenance=provenance,
            expected_sha256=registration.artifact_sha256,
            registered_fit_row_ids=fit_ids,
            registered_heldout_row_ids=heldout_ids,
        )
        if require_declared_partition and not adapted.audit["artifact_declared_full_partition"]:
            raise ValueError(
                "required neural query-moment evidence must be a hashed fold-scoped "
                "bundle that declares exact fit and heldout row IDs"
            )
        registrations[fold] = registration
    return registrations


def validate_benchmark_inputs(args: argparse.Namespace) -> ValidatedBenchmarkInputs:
    """Authenticate input structure without constructing remote dependencies."""

    _validate_discovery_cli_mode(args)
    benchmark = str(args.benchmark_name or "").strip()
    if not _BENCHMARK_NAME.fullmatch(benchmark):
        raise ValueError("--benchmark-name must use only letters, numbers, '.', '_' or '-'")
    # Construct both configs during dry-run validation.  These dataclasses do
    # not construct clients; they also prove the execution mode is fixed.
    agent_config = build_agent_config(args)
    applied_config = build_applied_inference_config(args)
    if agent_config.agent_provider != "openai":  # pragma: no cover - fixed above
        raise RuntimeError("proposal provider escaped the remote-only boundary")
    if (
        applied_config.explicit_features.extraction_provider != "openai"
        or applied_config.explicit_features.vllm_mode != _SERVER_MODE
    ):  # pragma: no cover - fixed above
        raise RuntimeError("extraction provider escaped the remote-only boundary")

    dataset_path = _required_file(args.dataset, label="dataset")
    legacy_path = _required_file(args.legacy_handoff, label="legacy handoff")
    tfidf_path = _required_file(args.resealed_tfidf_handoff, label="resealed TF-IDF handoff")
    primary_path = _required_file(args.primary_splits, label="primary split predictions")
    cache_paths = tuple(
        _required_file(path, label="read-only extraction cache index")
        for path in args.read_only_cache_index
    )
    review_spent_cache_sources = authenticate_review_spent_cache_registrations(
        args.read_only_review_spent_evidence_cache
    )
    context_fit_cache_sources = authenticate_context_fit_cache_index_registrations(
        args.read_only_context_fit_cache_index
    )
    _select_tfidf_context_backend_graph(context_fit_cache_sources)
    if review_spent_cache_sources and int(args.post_extraction_review_rounds) <= 0:
        raise ValueError("--read-only-review-spent-evidence-cache requires post-extraction review")
    if context_fit_cache_sources and int(args.post_extraction_review_rounds) <= 0:
        raise ValueError("--read-only-context-fit-cache-index requires post-extraction review")
    orphan_artifacts = parse_orphan_ngram_artifact_registrations(args.orphan_ngram_artifact)
    output_dir = _validated_output_path(args.output_dir)
    hierarchical = str(args.discovery_mode) == _HIERARCHICAL_DISCOVERY_MODE
    hierarchical_preparation_dir: Path | None = None
    hierarchical_job_cache_root: Path | None = None
    hierarchical_offline_review_packet_dir: Path | None = None
    if hierarchical:
        (
            hierarchical_preparation_dir,
            hierarchical_job_cache_root,
            hierarchical_offline_review_packet_dir,
        ) = _validated_hierarchical_layout(args, output_dir=output_dir)
    historical_discovery_prompt = _authenticated_prompt_registration(
        args.historical_discovery_prompt,
        label="historical discovery prompt",
    )
    old_hierarchy_prompt = _authenticated_prompt_registration(
        args.old_hierarchy_prompt,
        label="old hierarchy prompt",
    )
    review_stage1_config_path = None
    review_semantic_witness_scientific_config = None
    review_embedding_cache_dir = None
    review_neural_query_cache_dir = None
    if int(args.post_extraction_review_rounds) > 0:
        requested_query_cache = (
            args.review_neural_query_cache_dir
            if args.review_neural_query_cache_dir is not None
            else output_dir / "post_extraction_review_neural_query_cache"
        )
        review_neural_query_cache_dir = _validate_fresh_review_neural_query_cache(
            requested_query_cache,
            cache_parent=output_dir,
            require_empty=True,
            parent_option="--output-dir",
        )
        _validate_fresh_benchmark_output(
            output_dir,
            allowed_empty_cache_dir=review_neural_query_cache_dir,
        )
        review_stage1_config_path = _required_file(
            args.review_stage1_config,
            label="historical Stage-1 review config",
        )
        review_semantic_witness_scientific_config = (
            _load_semantic_witness_scientific_config(
                args.review_semantic_witness_scientific_config
            )
        )
        # Parse and validate any closed query override during dry-run, before
        # a service or model-bearing backend can be constructed.
        review_query_config = build_review_neural_query_config(args)
        build_coordinate_preserving_final_upstream_schema_config(
            review_stage1_config_path,
            neural_query_config=review_query_config,
            max_orphan_features=_final_upstream_max_orphan_features(args),
        )
    else:
        _validate_fresh_benchmark_output(output_dir)

    # The loader performs a Parquet projection of exactly the three model
    # columns.  No prompt, event-timeline, identifier, or oracle field enters
    # the process during prediction-side validation.
    data = load_sanitized_dataset(
        dataset_path,
        text_column=applied_config.text_column,
        treatment_column=applied_config.treatment_column,
        outcome_column=applied_config.outcome_column,
    )
    if int(args.post_extraction_review_rounds) > 0:
        review_embedding_cache_dir = _validate_review_embedding_cache(
            args.review_embedding_cache_dir,
            expected_row_count=len(data),
        )
    if cache_paths:
        # Parsing an overlay validates its versioned index and declared cache
        # identities without opening a network client or writing to cache
        # roots. Exact artifact authentication still occurs per requested
        # extraction contract in the runner.
        FrozenExtractionCacheOverlay(
            cache_paths,
            expected_row_count=len(data),
            row_id_column="_oci_row_id",
            text_column=applied_config.text_column,
        )
    legacy = load_legacy_full_outer_evidence(legacy_path)
    tfidf = load_resealed_tfidf_handoff(
        tfidf_path,
        dataset_row_count=len(data),
        require_registry_seal=True,
    )
    neural_query_artifacts = parse_neural_query_moment_artifact_registrations(
        args.neural_query_moment_artifact,
        full_outer_rows_by_fold=tfidf.full_rows_by_outer_fold,
        require_declared_partition=(
            bool(args.require_neural_query_moments) and int(args.post_extraction_review_rounds) == 0
        ),
    )
    primary = load_outer_splits_from_primary_predictions(
        primary_path,
        dataset_row_count=len(data),
    )
    folds = tuple(sorted(tfidf.full_rows_by_outer_fold))
    if not folds:
        raise ValueError("resealed TF-IDF handoff contains no full outer folds")
    if set(legacy.rows_by_outer_fold) != set(folds) or set(primary) != set(folds):
        raise ValueError("legacy, TF-IDF, and primary-split outer folds must match exactly")
    unknown_orphan_folds = set(orphan_artifacts) - set(folds)
    if unknown_orphan_folds:
        raise ValueError(
            "orphan n-gram artifact registrations contain unknown outer folds: "
            f"{sorted(unknown_orphan_folds)}"
        )
    for fold in folds:
        tfidf_ids = tuple(sorted(map(int, tfidf.full_rows_by_outer_fold[fold]["heldout_row_ids"])))
        if tuple(sorted(map(int, primary[fold]))) != tfidf_ids:
            raise ValueError(f"primary and TF-IDF heldout rows differ for fold {fold}")
        # Fingerprint computation catches duplicate or malformed row IDs and
        # records the same set semantics expected by the runner.
        row_set_fingerprint(tfidf_ids)
    if len(folds) != int(args.expected_outer_folds):
        raise ValueError(
            f"benchmark has {len(folds)} outer folds; expected " f"{int(args.expected_outer_folds)}"
        )
    missing_neural_query_folds = set(folds) - set(neural_query_artifacts)
    if (
        bool(args.require_neural_query_moments)
        and int(args.post_extraction_review_rounds) == 0
        and missing_neural_query_folds
    ):
        raise ValueError(
            "--require-neural-query-moments is set but no authenticated artifact was "
            f"registered for outer folds {sorted(missing_neural_query_folds)}"
        )
    return ValidatedBenchmarkInputs(
        dataset_path=dataset_path,
        legacy_handoff_path=legacy_path,
        tfidf_handoff_path=tfidf_path,
        primary_splits_path=primary_path,
        output_dir=output_dir,
        cache_index_paths=cache_paths,
        orphan_ngram_artifacts_by_fold=orphan_artifacts,
        row_count=len(data),
        outer_folds=folds,
        neural_query_moment_artifacts_by_fold=neural_query_artifacts,
        review_stage1_config_path=review_stage1_config_path,
        review_semantic_witness_scientific_config=(
            review_semantic_witness_scientific_config
        ),
        review_embedding_cache_dir=review_embedding_cache_dir,
        review_neural_query_cache_dir=review_neural_query_cache_dir,
        authenticated_review_spent_cache_sources=review_spent_cache_sources,
        authenticated_context_fit_cache_sources=context_fit_cache_sources,
        hierarchical_preparation_dir=hierarchical_preparation_dir,
        hierarchical_job_cache_root=hierarchical_job_cache_root,
        hierarchical_offline_review_packet_dir=hierarchical_offline_review_packet_dir,
        historical_discovery_prompt=historical_discovery_prompt,
        old_hierarchy_prompt=old_hierarchy_prompt,
    )


def load_posthoc_oracle_projection(dataset_path: Path | str) -> pd.DataFrame:
    """Load only the synthetic ITE column, then attach canonical row identity.

    The named benchmark Parquet files do not persist ``_oci_row_id``.  The
    prediction runner defines it as the zero-based Parquet row position, so the
    same deterministic identity is attached after projecting only
    ``true_ite_prob``.  The returned frame contains exactly the two columns
    accepted by the post-hoc evaluator.
    """

    oracle = pd.read_parquet(
        Path(dataset_path),
        columns=[_ORACLE_ITE_COLUMN],
    ).reset_index(drop=True)
    values = pd.to_numeric(oracle[_ORACLE_ITE_COLUMN], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("true_ite_prob must contain only finite numeric values")
    oracle[_ORACLE_ITE_COLUMN] = values.astype(float)
    oracle.insert(0, "_oci_row_id", np.arange(len(oracle), dtype=int))
    return oracle[["_oci_row_id", _ORACLE_ITE_COLUMN]]


def _dry_run_summary(
    args: argparse.Namespace,
    validated: ValidatedBenchmarkInputs,
) -> dict[str, Any]:
    review_enabled = int(args.post_extraction_review_rounds) > 0
    hierarchical = str(args.discovery_mode) == _HIERARCHICAL_DISCOVERY_MODE
    if not review_enabled:
        raise ValueError(
            "the v24 benchmark requires adaptive post-extraction review and the "
            "honest final causal forest"
        )
    review_query_config = build_review_neural_query_config(args) if review_enabled else None
    final_schema = (
        build_coordinate_preserving_final_upstream_schema_config(
            validated.review_stage1_config_path,
            neural_query_config=review_query_config,
            max_orphan_features=_final_upstream_max_orphan_features(args),
        )
        if review_enabled and validated.review_stage1_config_path is not None
        else None
    )
    final_registry_audit = (
        production_coordinate_preserving_registry_audit(final_schema)
        if final_schema is not None
        else None
    )
    sparse_query_fallback_enabled = False
    tfidf_graph, tfidf_graph_selection = _select_tfidf_context_backend_graph(
        validated.authenticated_context_fit_cache_sources
    )
    shared_tfidf_audit = _shared_tfidf_runtime_audit(
        review_enabled=review_enabled,
        graph=tfidf_graph,
        selection=tfidf_graph_selection,
    )
    hierarchical_config = build_hierarchical_discovery_config(args) if hierarchical else None
    review_policy = build_frozen_review_evidence_policy(args) if hierarchical else None
    per_fold_job_cache_roots = (
        {
            str(fold): str(validated.hierarchical_job_cache_root / f"outer_fold_{fold:03d}")
            for fold in validated.outer_folds
        }
        if hierarchical and validated.hierarchical_job_cache_root is not None
        else {}
    )
    return {
        "status": "validated_dry_run",
        "source_text_temporal_policy": source_text_temporal_policy_audit(),
        "benchmark_name": str(args.benchmark_name),
        "dataset_path": str(validated.dataset_path),
        "row_count": validated.row_count,
        "outer_folds": list(validated.outer_folds),
        "remote_endpoint_pool": validate_remote_endpoint_pool(args.endpoint),
        "remote_model": str(args.model),
        "discovery_mode": str(args.discovery_mode),
        "benchmark_default_discovery_mode": _HIERARCHICAL_DISCOVERY_MODE,
        "initial_discovery_agent": (
            "OpenAICompatibleJsonDiscoveryJobRunner"
            if hierarchical
            else "StagedAllEvidenceFusionAgent_legacy_test_only"
        ),
        "legacy_staged_initial_discovery_active": not hierarchical,
        "hierarchical_architecture_at_a_time": hierarchical,
        "hierarchical_all_active_stage1_architectures_required": hierarchical,
        "hierarchical_prepare_only_requested": bool(args.prepare_hierarchical_discovery),
        "hierarchical_approved_batch_sha256_supplied": bool(
            args.hierarchical_approved_batch_sha256
        ),
        "hierarchical_preparation_dir": (
            str(validated.hierarchical_preparation_dir) if hierarchical else None
        ),
        "hierarchical_job_cache_root": (
            str(validated.hierarchical_job_cache_root) if hierarchical else None
        ),
        "hierarchical_offline_review_packet_dir": (
            str(validated.hierarchical_offline_review_packet_dir) if hierarchical else None
        ),
        "historical_discovery_prompt_registration": (
            {
                "path": str(validated.historical_discovery_prompt.path),
                "sha256": validated.historical_discovery_prompt.expected_sha256,
            }
            if validated.historical_discovery_prompt is not None
            else None
        ),
        "old_hierarchy_prompt_registration": (
            {
                "path": str(validated.old_hierarchy_prompt.path),
                "sha256": validated.old_hierarchy_prompt.expected_sha256,
            }
            if validated.old_hierarchy_prompt is not None
            else None
        ),
        "hierarchical_per_fold_job_cache_roots": per_fold_job_cache_roots,
        "hierarchical_preparation_scratch_output_dir": (
            str(validated.output_dir) if hierarchical else None
        ),
        "hierarchical_provider_writable_caches_output_local": hierarchical,
        "hierarchical_cross_process_provider_replay_requires_authenticated_overlays": (
            hierarchical
        ),
        "hierarchical_fresh_executable_neural_query_cache_required": hierarchical,
        "hierarchical_discovery_config": (
            hierarchical_config.as_dict() if hierarchical_config is not None else None
        ),
        "hierarchical_architecture_chunk_limits": (
            {
                "max_atoms_per_chunk": int(args.hierarchical_max_atoms_per_chunk),
                "max_bytes_per_chunk": int(args.hierarchical_max_bytes_per_chunk),
                "max_semantic_member_ids_per_chunk": int(
                    args.hierarchical_max_semantic_member_ids_per_chunk
                ),
            }
            if hierarchical
            else None
        ),
        "hierarchical_review_evidence_policy": (
            review_policy.as_dict() if review_policy is not None else None
        ),
        "hierarchical_runner_schema_version": (
            OPENAI_JSON_DISCOVERY_RUNNER_VERSION if hierarchical else None
        ),
        "hierarchical_runner_temperature": 0 if hierarchical else None,
        "hierarchical_runner_response_format": (
            "strict_json_schema_from_authenticated_discovery_job_v1" if hierarchical else None
        ),
        "hierarchical_runner_explicit_model_no_autodiscovery": hierarchical,
        "hierarchical_runner_max_tokens": (int(args.proposal_max_tokens) if hierarchical else None),
        "hierarchical_runner_request_timeout": (
            float(args.request_timeout) if hierarchical else None
        ),
        "hierarchical_runner_max_retries": (
            int(args.request_max_retries) if hierarchical else None
        ),
        "hierarchical_selector_thinking_enabled": hierarchical,
        "hierarchical_selector_thinking_token_budget": (
            _FUSION_THINKING_TOKEN_BUDGET if hierarchical else None
        ),
        "hierarchical_extraction_definition_thinking_enabled": False,
        "hierarchical_extraction_definition_thinking_token_budget_field": "omitted",
        "hierarchical_max_rendered_prompt_bytes": (
            MAX_RENDERED_DISCOVERY_PROMPT_BYTES if hierarchical else None
        ),
        "hierarchical_json_runner_constructed": False,
        "proposal_provider": (
            "openai_compatible_strict_json_hierarchy"
            if hierarchical
            else "openai_compatible_remote_legacy_staged"
        ),
        "fusion_enable_thinking": _FUSION_ENABLE_THINKING,
        "fusion_max_tokens": int(args.proposal_max_tokens),
        "fusion_thinking_token_budget": _FUSION_THINKING_TOKEN_BUDGET,
        "extraction_provider": "openai_compatible_remote",
        "extraction_vllm_mode": _SERVER_MODE,
        "extraction_grouping_strategy": str(args.extraction_grouping_strategy),
        "extraction_grouping_version": EXTRACTION_GROUPING_VERSION,
        "extraction_context_strategy": str(args.extraction_context_strategy),
        "extraction_context_compactor_version": CONTRACT_LEXICAL_CONTEXT_VERSION,
        "extraction_max_text_length": int(args.extraction_max_text_length),
        "extraction_batch_size": int(args.extraction_batch_size),
        "max_variables_per_extraction_request": int(args.max_variables_per_extraction_request),
        "adaptive_review_contract_local_extraction_verified": bool(
            not review_enabled or int(args.max_variables_per_extraction_request) == 1
        ),
        "extraction_enable_thinking": _EXTRACTION_ENABLE_THINKING,
        "extraction_source_text_temporally_valid_by_design": True,
        "extraction_prompt_cache_identity": extraction_prompt_cache_identity(args),
        "post_extraction_review_rounds": int(args.post_extraction_review_rounds),
        "post_extraction_review_max_quality_retries": int(
            args.post_extraction_review_max_quality_retries
        ),
        "post_extraction_review_agent_is_base_reasoning_agent": bool(review_enabled),
        "post_extraction_review_source_signals_required": review_enabled,
        "post_extraction_review_feature_banks_required": review_enabled,
        "post_extraction_review_spent_discovery_families": (
            sorted(_REQUIRED_REVIEW_DISCOVERY_FAMILIES) if review_enabled else []
        ),
        "read_only_review_spent_cache_source_count": len(
            validated.authenticated_review_spent_cache_sources
        ),
        "read_only_review_spent_cache_sources": [
            source.identity() for source in validated.authenticated_review_spent_cache_sources
        ],
        "read_only_context_fit_cache_source_count": len(
            validated.authenticated_context_fit_cache_sources
        ),
        "read_only_context_fit_cache_sources": [
            source.identity() for source in validated.authenticated_context_fit_cache_sources
        ],
        **shared_tfidf_audit,
        "post_extraction_review_gate_provider": (
            "shared_context_fit_all_upstream" if review_enabled else None
        ),
        "precomputed_recursive_review_feature_banks_enabled": False,
        "review_stage1_config_path": (
            str(validated.review_stage1_config_path) if review_enabled else None
        ),
        "review_semantic_witness_scientific_config": (
            None
            if validated.review_semantic_witness_scientific_config is None
            else validated.review_semantic_witness_scientific_config.as_dict()
        ),
        "review_semantic_witness_scientific_config_sha256": (
            None
            if validated.review_semantic_witness_scientific_config is None
            else validated.review_semantic_witness_scientific_config.identity_sha256
        ),
        "review_embedding_cache_dir": (
            str(validated.review_embedding_cache_dir) if review_enabled else None
        ),
        "review_neural_query_cache_dir": (
            str(validated.review_neural_query_cache_dir) if review_enabled else None
        ),
        "review_stage1_device": _review_stage1_device(args) if review_enabled else None,
        "review_stage1_bow_fold_parallelism": (
            int(args.review_stage1_bow_fold_parallelism) if review_enabled else None
        ),
        "review_stage1_bow_parallel_backend": (
            str(args.review_stage1_bow_parallel_backend) if review_enabled else None
        ),
        "review_neural_query_devices": (
            list(_review_neural_query_devices(args)) if review_enabled else []
        ),
        "review_neural_query_config": (
            asdict(review_query_config) if review_query_config is not None else None
        ),
        "final_upstream_inputs_required": review_enabled,
        "final_upstream_neural_query_inputs_required": review_enabled,
        "final_upstream_producer_constructed": False,
        "final_causal_forest_required": True,
        "final_causal_forest_active": True,
        "final_ite_estimator": FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID,
        "final_causal_forest_backend": FixedCausalForestHeadBackend(
            random_state=int(args.seed)
        ).identity(),
        "nonforest_final_model_fallback_allowed": False,
        "final_causal_forest_backend_injected": False,
        "raw_final_upstream_runtime_constructed": False,
        "raw_final_upstream_runtime_retained_separately_from_cache_overlay": (review_enabled),
        "final_upstream_meta_inner_folds": int(args.final_upstream_meta_inner_folds),
        "final_upstream_head_regularization": float(args.final_upstream_head_regularization),
        "final_upstream_max_orphan_features": (
            _final_upstream_max_orphan_features(args) if review_enabled else None
        ),
        "final_upstream_schema_namespace": (
            final_schema.namespace if final_schema is not None else None
        ),
        "final_upstream_signed_order_width": (None),
        "final_upstream_volatile_signed_order_widths": (
            {
                f"{item.source_kind}::{item.consumer_role}": item.signed_order_width
                for item in final_schema.volatile_raw_families
            }
            if final_schema is not None
            else {}
        ),
        "final_upstream_neural_query_signed_order_widths": (
            {
                bank: review_query_config.query_count(bank)
                for bank in ("treatment", "outcome", "effect")
            }
            if review_query_config is not None
            else {}
        ),
        "final_upstream_calibrated_sources": (
            [item.identity() for item in final_schema.calibrated_sources]
            if final_schema is not None
            else []
        ),
        "final_upstream_raw_family_roles": (
            [item.identity() for item in final_schema.volatile_raw_families]
            if final_schema is not None
            else []
        ),
        "final_upstream_raw_family_role_count": (
            len(final_schema.volatile_raw_families) if final_schema is not None else 0
        ),
        "final_upstream_named_raw_coordinates": (
            [item.identity() for item in final_schema.named_raw_coordinates]
            if final_schema is not None
            else []
        ),
        "final_upstream_named_raw_coordinate_count": (
            len(final_schema.named_raw_coordinates) if final_schema is not None else 0
        ),
        "final_upstream_raw_column_count": (
            len(final_schema.raw_output_schema()) if final_schema is not None else 0
        ),
        "final_upstream_coordinate_registry": final_registry_audit,
        "modifier_interactions_only_required_for_final_upstream": review_enabled,
        "modifier_interactions_only": bool(args.modifier_interactions_only),
        "neural_query_moments_required": (
            review_enabled or bool(args.require_neural_query_moments)
        ),
        "neural_query_moment_requirement_flag_set": bool(args.require_neural_query_moments),
        "neural_query_moment_requirement_mode": (
            "adaptive_context_fit"
            if review_enabled
            else (
                "authenticated_fold_artifact"
                if bool(args.require_neural_query_moments)
                else "optional"
            )
        ),
        "adaptive_context_fit_neural_query_path_required": review_enabled,
        "authenticated_neural_query_artifacts_required": (
            bool(args.require_neural_query_moments) and not review_enabled
        ),
        "authenticated_neural_query_moment_folds": sorted(
            validated.neural_query_moment_artifacts_by_fold
        ),
        "registered_neural_query_artifact_usage": (
            "adaptive_audit_only_excluded_from_selector_and_model_inputs"
            if review_enabled
            else "nonadaptive_selector_evidence"
        ),
        "sparse_query_moment_fallback_enabled": sparse_query_fallback_enabled,
        "query_moment_fallback_enabled": sparse_query_fallback_enabled,
        "sparse_query_moment_fallback_folds": (
            sorted(
                set(validated.outer_folds) - set(validated.neural_query_moment_artifacts_by_fold)
            )
            if sparse_query_fallback_enabled
            else []
        ),
        "tfidf_orphan_adapter_enabled": True,
        "read_only_cache_index_count": len(validated.cache_index_paths),
        "explicit_orphan_artifact_folds": sorted(validated.orphan_ngram_artifacts_by_fold),
        "clients_constructed": False,
        "remote_calls_made": False,
        "oracle_columns_read": False,
    }


def _persist_hierarchical_offline_review_packet(
    *,
    prepared: Any,
    validated: ValidatedBenchmarkInputs,
) -> tuple[Any, Any]:
    """Compose the mandatory human-review packet from one real prepared fold."""

    if (
        validated.historical_discovery_prompt is None
        or validated.old_hierarchy_prompt is None
        or validated.hierarchical_offline_review_packet_dir is None
    ):
        raise RuntimeError("validated prepare-only comparison artifacts are incomplete")
    folds = tuple(prepared.folds)
    if not folds:
        raise RuntimeError("prepared hierarchy contains no outer fold for offline review")
    representative = folds[0]
    atoms = tuple(representative.catalog.atoms)
    if not atoms:
        raise RuntimeError("representative prepared fold has no evidence atoms")
    atom = atoms[0]
    evidence = atom.as_discovery_item()
    extraction_preview = build_offline_extraction_definition_prompt_preview(
        canonical_name=f"preview_{atom.source_family}_supported_concept",
        evidence=(evidence,),
        supporting_evidence_ids=(evidence.evidence_id,),
        value_shape_hypothesis="ambiguous",
    )
    packet = compose_offline_hierarchical_discovery_review_packet(
        batch_precommit=prepared.coordinator.precommit,
        representative_outer_fold=representative.outer_fold,
        representative_family=atom.source_family,
        extraction_definition_preview=extraction_preview,
        extraction_preview_outer_fold=representative.outer_fold,
        historical_prompt=validated.historical_discovery_prompt,
        old_hierarchy_prompt=validated.old_hierarchy_prompt,
    )
    if packet.approval_ready is not True:
        raise RuntimeError("offline hierarchical review packet is not approval-ready")
    persisted = packet.persist(
        preparation_directory=validated.hierarchical_offline_review_packet_dir
    )
    persisted.validate_authentication()
    return packet, persisted


def _prepared_review_spent_cache_registrations(
    *,
    validated: ValidatedBenchmarkInputs,
    expected_fold_count: int,
) -> tuple[str, ...]:
    """Authenticate the fold-local spent caches exported by preparation."""

    root = validated.output_dir / "post_extraction_review_spent_evidence_cache"
    if not root.is_dir():
        raise RuntimeError("prepare-only did not materialize the spent-evidence cache root")
    paths = tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_file() and not path.is_symlink() and path.suffix == ".json"
            ),
            key=lambda path: path.name,
        )
    )
    registrations = tuple(f"{path}::{sha256_file(path)}" for path in paths)
    authenticated = authenticate_review_spent_cache_registrations(registrations)
    if len(authenticated) != expected_fold_count:
        raise RuntimeError(
            "prepare-only must export exactly one authenticated initial-spent cache "
            "per outer fold"
        )
    return registrations


def _approval_preserving_review_spent_cache_registrations(
    validated: ValidatedBenchmarkInputs,
) -> tuple[str, ...]:
    """Render the exact spent-cache sources already bound into preparation."""

    return tuple(
        f"{source.source_path}::{source.registered_sha256}"
        for source in validated.authenticated_review_spent_cache_sources
    )


def _approval_preserving_context_fit_cache_registrations(
    validated: ValidatedBenchmarkInputs,
) -> tuple[str, ...]:
    """Render the exact context-cache indexes already bound into preparation."""

    registrations: list[str] = []
    seen: set[tuple[Path, str]] = set()
    for source in validated.authenticated_context_fit_cache_sources:
        key = (Path(source.index_path), str(source.index_sha256))
        if key in seen:
            continue
        seen.add(key)
        registrations.append(f"{key[0]}::{key[1]}")
    return tuple(registrations)


def run_benchmark(args: argparse.Namespace) -> Mapping[str, Any]:
    """Validate and run one benchmark, or return a side-effect-free dry-run."""

    validated = validate_benchmark_inputs(args)
    if bool(args.dry_run):
        return _dry_run_summary(args, validated)
    hierarchical = str(args.discovery_mode) == _HIERARCHICAL_DISCOVERY_MODE
    if (
        hierarchical
        and not bool(args.prepare_hierarchical_discovery)
        and args.hierarchical_approved_batch_sha256 is None
    ):
        raise ValueError(
            "hierarchical execution requires the exact "
            "--hierarchical-approved-batch-sha256 printed by prepare-only mode"
        )
    tfidf_graph, tfidf_graph_selection = _select_tfidf_context_backend_graph(
        validated.authenticated_context_fit_cache_sources
    )

    agent_config = build_agent_config(args)
    base_agent = OpenAICompatibleFeatureSearchAgent(agent_config)
    hierarchical_runner = None
    if hierarchical:
        hierarchical_runner = OpenAICompatibleJsonDiscoveryJobRunner(
            server_urls=validate_remote_endpoint_pool(args.endpoint),
            model_name=str(args.model),
            api_key=str(args.api_key),
            request_timeout=float(args.request_timeout),
            max_retries=int(args.request_max_retries),
            max_tokens=int(args.proposal_max_tokens),
        )
        fusion_agent = None
    else:
        fusion_agent = StagedAllEvidenceFusionAgent(
            base_agent,
            final_max_candidates=int(args.max_candidates),
        )

    applied_config = build_applied_inference_config(args, vllm_mode=_SERVER_MODE)
    if applied_config.explicit_features.vllm_mode != _SERVER_MODE:
        raise RuntimeError("non-server extraction mode is forbidden")
    extraction_provider = VLLMExplicitFeatureExtractionProvider(
        applied_config,
        validated.output_dir / "remote_extraction",
    )
    review_rounds = int(args.post_extraction_review_rounds)
    if review_rounds < 1:
        raise RuntimeError(
            "the v24 benchmark requires adaptive post-extraction review and the "
            "honest final causal forest"
        )
    review_spent_evidence_provider = None
    raw_review_spent_evidence_provider = None
    review_gate_provider = None
    raw_review_gate_provider = None
    final_upstream_producer = None
    raw_final_upstream_producer = None
    coordinate_preserving_nuisance_view_names: tuple[str, ...] | None = None
    if review_rounds > 0:
        if (
            validated.review_stage1_config_path is None
            or validated.review_embedding_cache_dir is None
            or validated.review_neural_query_cache_dir is None
        ):
            raise RuntimeError("validated adaptive-review dependencies are incomplete")
        stage1_config_snapshot = HistoricalStage1ConfigSnapshot.from_path(
            validated.review_stage1_config_path
        )
        htr_config = stage1_config_snapshot.applied_config()
        coordinate_preserving_nuisance_view_names = tuple(
            str(view.name).strip() for view in htr_config.architecture.multi_model_forest.bow_views
        )
        htr_model_snapshot = PrivateHTRModelTreeSnapshot(_resolve_htr_model_path(htr_config))
        shared_embedding_cache = SpentOnlyFrozenChunkEmbeddingCache(
            validated.review_embedding_cache_dir
        )
        if shared_embedding_cache.row_count != validated.row_count:
            raise RuntimeError(
                "authenticated review embedding cache row count changed after validation"
            )
        review_query_config = build_review_neural_query_config(args)
        query_service = ContextFitNeuralQueryService(
            cache_dir=validated.review_neural_query_cache_dir,
            dataset_path=validated.dataset_path,
            text_column=applied_config.text_column,
            embedding_cache_dir=validated.review_embedding_cache_dir,
            stage1_config_path=validated.review_stage1_config_path,
            embedding_cache=shared_embedding_cache,
            stage1_config_snapshot=stage1_config_snapshot,
            query_config=review_query_config,
            nuisance_folds=int(args.review_neural_query_nuisance_folds),
            devices=_review_neural_query_devices(args),
            seed=int(args.seed),
            outcome_type=applied_config.outcome_type,
        )
        tfidf_spent_backend = TfidfTopicOrphanSpentDiscoveryBackend(
            stage1_config_path=validated.review_stage1_config_path,
            stage1_config_snapshot=stage1_config_snapshot,
            outcome_type=applied_config.outcome_type,
            orphan_config=orphan_ngram_adapter_config_from_tfidf_topic(
                htr_config.architecture.multi_model_forest.tfidf_topic
            ),
        )
        tfidf_context_backend = TfidfTopicOrphanContextBackend(
            stage1_config_path=validated.review_stage1_config_path,
            stage1_config_snapshot=stage1_config_snapshot,
            outcome_type=applied_config.outcome_type,
            max_orphan_features=_final_upstream_max_orphan_features(args),
        )
        # Context-fit run attestations select their exact historical graph.
        # With no such source, prefer the current wrapped graph; a spent-only
        # overlay must then validate against that exact current provider.
        if tfidf_graph == SHARED_TFIDF_RUNTIME_GRAPH_ID:
            shared_tfidf = build_shared_tfidf_context_fit_backends(
                spent_discovery_backend=tfidf_spent_backend,
                context_backend=tfidf_context_backend,
            )
            tfidf_spent_backend = shared_tfidf.spent_discovery_backend
            tfidf_context_backend = shared_tfidf.context_backend
        elif tfidf_graph != UNWRAPPED_TFIDF_RUNTIME_GRAPH_ID:
            raise RuntimeError("selected an unsupported TF-IDF context backend graph")
        raw_review_spent_evidence_provider = ContextFitReviewSpentEvidenceProvider(
            backends=(
                HistoricalStage1SpentDiscoveryBackend(
                    dataset_path=validated.dataset_path,
                    stage1_config_path=validated.review_stage1_config_path,
                    embedding_cache_dir=validated.review_embedding_cache_dir,
                    stage1_config_snapshot=stage1_config_snapshot,
                    embedding_cache=shared_embedding_cache,
                    htr_model_snapshot=htr_model_snapshot,
                    semantic_witness_scientific_config=(
                        validated.review_semantic_witness_scientific_config
                    ),
                    device=_review_stage1_device(args),
                    bow_fold_parallelism=int(args.review_stage1_bow_fold_parallelism),
                    bow_parallel_backend=str(args.review_stage1_bow_parallel_backend),
                ),
                tfidf_spent_backend,
                NeuralQuerySpentDiscoveryBackend(query_service),
            ),
            cache_dir=validated.output_dir / "post_extraction_review_spent_evidence_cache",
            required_source_families=tuple(sorted(_REQUIRED_REVIEW_DISCOVERY_FAMILIES)),
        )
        review_spent_evidence_provider = raw_review_spent_evidence_provider
        if validated.authenticated_review_spent_cache_sources:
            review_spent_evidence_provider = AuthenticatedReviewSpentEvidenceCacheOverlay(
                provider=review_spent_evidence_provider,
                sources=validated.authenticated_review_spent_cache_sources,
                output_root=validated.output_dir,
            )
        review_backend = CompositeContextFitUpstreamBackend(
            (
                HistoricalStage1ContextBackend(
                    dataset_path=validated.dataset_path,
                    stage1_config_path=validated.review_stage1_config_path,
                    embedding_cache_dir=validated.review_embedding_cache_dir,
                    stage1_config_snapshot=stage1_config_snapshot,
                    embedding_cache=shared_embedding_cache,
                    htr_model_snapshot=htr_model_snapshot,
                    device=_review_stage1_device(args),
                    bow_fold_parallelism=int(args.review_stage1_bow_fold_parallelism),
                    bow_parallel_backend=str(args.review_stage1_bow_parallel_backend),
                ),
                tfidf_context_backend,
                NeuralQueryContextBackend(query_service),
            )
        )
        stable_review_backend = CoordinatePreservingContextFitUpstreamBackend(
            review_backend,
            config=build_coordinate_preserving_final_upstream_schema_config(
                validated.review_stage1_config_path,
                stage1_config_snapshot=stage1_config_snapshot,
                neural_query_config=review_query_config,
                max_orphan_features=_final_upstream_max_orphan_features(args),
            ),
        )
        raw_review_gate_provider = ContextFitUpstreamGateProvider(
            validated.output_dir / "post_extraction_review_gate_cache",
            backend=stable_review_backend,
        )
        raw_final_upstream_producer = FinalContextFitUpstreamProducer(
            validated.output_dir / "final_context_fit_upstream_cache",
            backend=stable_review_backend,
        )
        review_gate_provider = raw_review_gate_provider
        final_upstream_producer = raw_final_upstream_producer
        context_fit_sources = validated.authenticated_context_fit_cache_sources
        if any(source.kind == "review_gate" for source in context_fit_sources):
            overlay_kwargs = {
                "provider": raw_review_gate_provider,
                "runtime_producer": raw_final_upstream_producer,
                "sources": context_fit_sources,
                "output_root": validated.output_dir,
            }
            if hierarchical:
                overlay_kwargs["hierarchical_first_gate_preparation"] = True
            review_gate_provider = AuthenticatedContextFitGateCacheOverlay(**overlay_kwargs)
        if any(source.kind == "final_upstream" for source in context_fit_sources):
            final_upstream_producer = AuthenticatedFinalContextFitCacheOverlay(
                producer=raw_final_upstream_producer,
                sources=context_fit_sources,
                output_root=validated.output_dir,
            )
    if (
        review_spent_evidence_provider is None
        or review_gate_provider is None
        or final_upstream_producer is None
        or raw_final_upstream_producer is None
    ):
        raise RuntimeError(
            "the v24 benchmark cannot construct its required adaptive-review and "
            "honest causal-forest runtime"
        )
    overlay = None
    if validated.cache_index_paths:
        overlay = FrozenExtractionCacheOverlay(
            validated.cache_index_paths,
            expected_row_count=validated.row_count,
            row_id_column="_oci_row_id",
            text_column=applied_config.text_column,
        )

    runner = AllEvidenceFusionRunner(
        dataset_path=validated.dataset_path,
        legacy_handoff_path=validated.legacy_handoff_path,
        tfidf_handoff_path=validated.tfidf_handoff_path,
        output_dir=validated.output_dir,
        fusion_agent=fusion_agent,
        extraction_provider=extraction_provider,
        review_agent=base_agent,
        review_spent_evidence_provider=review_spent_evidence_provider,
        review_partition_provider=None,
        review_gate_source_provider=review_gate_provider,
        review_gate_feature_bank_provider=review_gate_provider,
        final_upstream_producer=final_upstream_producer,
        raw_final_upstream_producer=raw_final_upstream_producer,
        coordinate_preserving_nuisance_view_names=(coordinate_preserving_nuisance_view_names),
        legacy_primary_predictions_path=validated.primary_splits_path,
        tfidf_orphan_artifacts_by_fold=validated.orphan_ngram_artifacts_by_fold,
        cache_overlay=overlay,
        hierarchical_discovery_runner=hierarchical_runner,
        hierarchical_discovery_config=(
            build_hierarchical_discovery_config(args) if hierarchical else None
        ),
        hierarchical_discovery_job_cache_root=(
            validated.hierarchical_job_cache_root if hierarchical else None
        ),
        hierarchical_discovery_job_cache_config=(
            HierarchicalDiscoveryJobCacheConfig(
                max_entry_bytes=int(
                    args.hierarchical_job_cache_max_entry_bytes
                )
            )
            if hierarchical
            else None
        ),
        first_untouched_gate_preparation_bounds=(
            FirstUntouchedGatePreparationBounds(
                max_initial_spent_rows=int(
                    args.first_untouched_gate_max_initial_spent_rows
                ),
                max_first_gate_rows=int(
                    args.first_untouched_gate_max_first_gate_rows
                ),
                max_total_text_utf8_bytes=int(
                    args.first_untouched_gate_max_total_text_utf8_bytes
                ),
                max_catalog_atoms=int(
                    args.first_untouched_gate_max_catalog_atoms
                ),
                max_source_manifest_bytes=int(
                    args.first_untouched_gate_max_source_manifest_bytes
                ),
                max_direct_numerical_signals=int(
                    args.first_untouched_gate_max_direct_numerical_signals
                ),
                max_single_matrix_file_bytes=int(
                    args.first_untouched_gate_max_single_matrix_file_bytes
                ),
                max_total_matrix_file_bytes=int(
                    args.first_untouched_gate_max_total_matrix_file_bytes
                ),
            )
            if hierarchical
            else None
        ),
        hierarchical_discovery_approved_batch_sha256=(
            args.hierarchical_approved_batch_sha256 if hierarchical else None
        ),
        hierarchical_review_evidence_policy=(
            build_frozen_review_evidence_policy(args) if hierarchical else None
        ),
        hierarchical_preparation_dir=(
            validated.hierarchical_preparation_dir if hierarchical else None
        ),
        hierarchical_max_atoms_per_chunk=int(args.hierarchical_max_atoms_per_chunk),
        hierarchical_max_bytes_per_chunk=int(args.hierarchical_max_bytes_per_chunk),
        hierarchical_max_semantic_member_ids_per_chunk=int(
            args.hierarchical_max_semantic_member_ids_per_chunk
        ),
        config=AllEvidenceFusionRunnerConfig(
            text_column=applied_config.text_column,
            treatment_column=applied_config.treatment_column,
            outcome_column=applied_config.outcome_column,
            outcome_type=applied_config.outcome_type,
            max_candidates=int(args.max_candidates),
            interaction_inner_folds=int(args.interaction_inner_folds),
            interact_all_features=not bool(args.modifier_interactions_only),
            random_state=int(args.seed),
            fusion_model_identity=str(args.model),
            fusion_enable_thinking=bool(agent_config.agent_enable_thinking),
            fusion_max_tokens=agent_config.agent_max_tokens,
            fusion_thinking_token_budget=agent_config.agent_thinking_token_budget,
            extraction_model_identity=str(args.model),
            remote_endpoint_pool_identity=validate_remote_endpoint_pool(args.endpoint),
            extraction_prompt_template_version=extraction_prompt_cache_identity(args),
            extraction_enable_thinking=bool(applied_config.explicit_features.vllm_enable_thinking),
            extraction_grouping_strategy=str(args.extraction_grouping_strategy),
            extraction_grouping_version=EXTRACTION_GROUPING_VERSION,
            extraction_context_strategy=str(args.extraction_context_strategy),
            extraction_context_compactor_version=CONTRACT_LEXICAL_CONTEXT_VERSION,
            extraction_max_text_length=int(args.extraction_max_text_length),
            extraction_batch_size=int(args.extraction_batch_size),
            max_variables_per_extraction_request=int(args.max_variables_per_extraction_request),
            post_extraction_review_rounds=review_rounds,
            post_extraction_review_max_operations=int(args.post_extraction_review_max_operations),
            post_extraction_review_max_quality_retries=int(
                args.post_extraction_review_max_quality_retries
            ),
            post_extraction_review_min_partition_rows=int(
                args.post_extraction_review_min_partition_rows
            ),
            require_review_source_signals=True,
            require_review_feature_banks=True,
            require_final_upstream_inputs=True,
            require_final_upstream_neural_query_inputs=True,
            require_final_causal_forest=True,
            final_upstream_meta_inner_folds=int(args.final_upstream_meta_inner_folds),
            final_upstream_head_regularization=float(args.final_upstream_head_regularization),
            require_registry_seal=True,
            include_tfidf_orphan_ngrams=True,
            require_tfidf_orphan_ngrams=bool(args.require_orphan_ngrams),
            orphan_ngram_adapter=orphan_ngram_adapter_config_from_tfidf_topic(
                applied_config.architecture.multi_model_forest.tfidf_topic
            ),
            derive_sparse_query_moments_when_missing=False,
            require_neural_query_moments=bool(args.require_neural_query_moments),
            neural_query_moment_artifacts_by_fold=(validated.neural_query_moment_artifacts_by_fold),
        ),
    )

    if bool(args.prepare_hierarchical_discovery):
        if not hierarchical or hierarchical_runner is None:  # pragma: no cover - mode guard
            raise RuntimeError("prepare-only mode escaped the hierarchical boundary")
        prepared = runner.prepare_hierarchical_discovery_batch()
        if hierarchical_runner.execution_metadata:
            raise RuntimeError("prepare-only mode executed a hierarchical JSON job")
        if getattr(hierarchical_runner, "_pool", None) is not None:
            raise RuntimeError("prepare-only mode constructed an OpenAI client pool")
        if (
            getattr(base_agent, "_client", None) is not None
            or getattr(base_agent, "_client_pool", None) is not None
        ):
            raise RuntimeError("prepare-only mode constructed a review-agent client")
        review_packet, persisted_review = _persist_hierarchical_offline_review_packet(
            prepared=prepared,
            validated=validated,
        )
        spent_cache_registrations = _prepared_review_spent_cache_registrations(
            validated=validated,
            expected_fold_count=len(prepared.folds),
        )
        if len(spent_cache_registrations) != len(prepared.folds):
            raise RuntimeError("preparation must retain exactly one spent cache per fold")
        # Approval binds the exact authenticated input overlays, including
        # their source paths and source count.  Newly copied output-local cache
        # files are useful inputs to a *new* preparation, but substituting them
        # into execution changes the provider identity and invalidates the
        # approved batch before any remote call.
        authoritative_spent_registrations = _approval_preserving_review_spent_cache_registrations(
            validated
        )
        authoritative_context_registrations = _approval_preserving_context_fit_cache_registrations(
            validated
        )
        return {
            "status": "hierarchical_discovery_prepared_awaiting_approval",
            "benchmark_name": str(args.benchmark_name),
            "approval_sha256": prepared.approval_sha256,
            "batch_packet_path": str(prepared.batch_packet_path),
            "input_manifest_path": str(prepared.input_manifest_path),
            "input_manifest_sha256": prepared.input_manifest_sha256,
            "dataset_sha256": prepared.dataset_sha256,
            "context_fit_overlay_companion_path": str(prepared.context_fit_overlay_companion_path),
            "context_fit_overlay_companion_sha256": (prepared.context_fit_overlay_companion_sha256),
            "first_gate_materialization_intent_index_path": str(
                prepared.first_gate_materialization_intent_index_path
            ),
            "first_gate_materialization_intent_index_sha256": (
                prepared.first_gate_materialization_intent_index_sha256
            ),
            "first_gate_context_fit_cache_registration": None,
            "first_gate_numerical_materialization_deferred": True,
            "review_spent_evidence_cache_registrations": list(spent_cache_registrations),
            "new_preparation_review_spent_registrations": list(spent_cache_registrations),
            "hierarchical_preparation_cache_replay_exported": False,
            "hierarchical_preparation_cache_replay_registration": None,
            "authoritative_replay_review_spent_registrations": list(
                authoritative_spent_registrations
            ),
            "authoritative_replay_context_fit_index_registration": None,
            "authoritative_execution_review_spent_registrations": list(
                authoritative_spent_registrations
            ),
            "authoritative_execution_context_fit_index_registrations": list(
                authoritative_context_registrations
            ),
            "authoritative_execution_replay_arguments": [
                *(
                    "--read-only-review-spent-evidence-cache=" + registration
                    for registration in authoritative_spent_registrations
                ),
                *(
                    "--read-only-context-fit-cache-index=" + registration
                    for registration in authoritative_context_registrations
                ),
            ],
            "next_authenticated_provider_replay_arguments": [
                *(
                    "--read-only-review-spent-evidence-cache=" + registration
                    for registration in spent_cache_registrations
                ),
            ],
            "new_preparation_replay_arguments": [
                *(
                    "--read-only-review-spent-evidence-cache=" + registration
                    for registration in spent_cache_registrations
                ),
            ],
            "hierarchical_preparation_dir": str(validated.hierarchical_preparation_dir),
            "hierarchical_job_cache_root": str(validated.hierarchical_job_cache_root),
            "hierarchical_per_fold_job_cache_roots": {
                str(fold.outer_fold): str(
                    validated.hierarchical_job_cache_root / f"outer_fold_{fold.outer_fold:03d}"
                )
                for fold in prepared.folds
            },
            "next_execution_argument": (
                "--hierarchical-approved-batch-sha256=" + prepared.approval_sha256
            ),
            "offline_review_packet_approval_ready": review_packet.approval_ready,
            "offline_review_packet_sha256": review_packet.packet_sha256,
            "offline_review_packet_json_path": str(persisted_review.packet_json_path),
            "offline_review_packet_json_sha256": (persisted_review.packet_json_sha256),
            "offline_review_packet_markdown_path": str(persisted_review.packet_markdown_path),
            "offline_review_packet_markdown_sha256": (persisted_review.packet_markdown_sha256),
            "offline_review_packet_manifest_path": str(persisted_review.manifest_path),
            "offline_review_packet_manifest_sha256": (persisted_review.manifest_sha256),
            "remote_execution_authorized_by_review_packet": False,
            "remote_clients_constructed": False,
            "remote_calls_made": False,
            "hierarchical_json_jobs_executed": 0,
            "preparation_scratch_output_dir": str(validated.output_dir),
            "preparation_scratch_local_caches_materialized": (
                validated.output_dir.exists() and any(validated.output_dir.iterdir())
            ),
            "predictions_written": False,
            "final_run_manifest_written": False,
            "execution_requires_fresh_output_dir": True,
            "cross_process_provider_replay_requires_authenticated_read_only_overlays": True,
            "first_gate_context_fit_replay_required_before_approval": False,
            "oracle_columns_read": False,
        }

    # This call must finish and freeze oracle-free predictions before the
    # evaluation flag can cause any oracle column to be read.
    result: AllEvidenceFusionRunResult = runner.run()
    response: dict[str, Any] = {
        "status": "completed",
        "discovery_mode": str(args.discovery_mode),
        "hierarchical_batch_approval_sha256": (
            args.hierarchical_approved_batch_sha256 if hierarchical else None
        ),
        "source_text_temporal_policy": source_text_temporal_policy_audit(),
        "benchmark_name": str(args.benchmark_name),
        "prediction_path": str(result.prediction_path),
        "prediction_sha256": result.prediction_sha256,
        "run_manifest_path": str(result.run_manifest_path),
        "final_ite_estimator": FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID,
        "final_causal_forest_active": True,
        "final_causal_forest_backend": FixedCausalForestHeadBackend(
            random_state=int(args.seed)
        ).identity(),
        "nonforest_final_model_fallback_allowed": False,
        "raw_final_upstream_runtime_retained_separately_from_cache_overlay": True,
        **_shared_tfidf_runtime_audit(
            review_enabled=review_rounds > 0,
            graph=tfidf_graph,
            selection=tfidf_graph_selection,
        ),
        "posthoc_oracle_evaluation_performed": False,
    }
    if bool(args.evaluate_oracle_posthoc):
        declared_prediction_sha256 = str(result.prediction_sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", declared_prediction_sha256):
            raise ValueError("runner returned an invalid frozen prediction SHA-256")
        actual_prediction_sha256 = sha256_file(result.prediction_path)
        if actual_prediction_sha256 != declared_prediction_sha256:
            raise ValueError(
                "frozen prediction SHA-256 does not match the runner result; "
                "oracle projection is forbidden"
            )
        oracle = load_posthoc_oracle_projection(validated.dataset_path)
        if len(oracle) != validated.row_count:
            raise ValueError("post-hoc oracle row count differs from sanitized dataset")
        metrics = evaluate_frozen_all_evidence_predictions(
            prediction_path=result.prediction_path,
            expected_prediction_sha256=declared_prediction_sha256,
            oracle_frame=oracle,
            output_dir=validated.output_dir / "posthoc_oracle_evaluation",
            oracle_ite_column=_ORACLE_ITE_COLUMN,
        )
        response["posthoc_oracle_evaluation_performed"] = True
        response["posthoc_oracle_metrics"] = metrics
    return response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-name", "--benchmark", required=True)
    parser.add_argument(
        "--discovery-mode",
        choices=(_HIERARCHICAL_DISCOVERY_MODE, _LEGACY_STAGED_DISCOVERY_MODE),
        default=_HIERARCHICAL_DISCOVERY_MODE,
        help=(
            "Initial feature discovery mode. Production defaults to the "
            "architecture-at-a-time hierarchy; the staged mode exists only for "
            "legacy tests and historical ablations."
        ),
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--legacy-handoff", required=True, type=Path)
    parser.add_argument(
        "--resealed-tfidf-handoff",
        "--tfidf-handoff",
        required=True,
        type=Path,
    )
    parser.add_argument("--primary-splits", required=True, type=Path)
    parser.add_argument("--output-dir", "--output", required=True, type=Path)
    parser.add_argument(
        "--hierarchical-preparation-dir",
        type=Path,
        help=(
            "Separate non-nested directory for immutable all-fold approval packets "
            "and resumable hierarchical JSON-job caches. Required in hierarchical mode."
        ),
    )
    parser.add_argument(
        "--hierarchical-job-cache-root",
        type=Path,
        help=(
            "Stable job-cache root below hierarchical-preparation-dir. Defaults to "
            "<preparation-dir>/hierarchical_job_cache; each outer fold has a stable "
            "outer_fold_NNN child."
        ),
    )
    parser.add_argument(
        "--hierarchical-job-cache-max-entry-bytes",
        type=int,
        help=(
            "Required in hierarchical mode. Maximum authenticated cache-entry "
            "size in bytes; there is no production default."
        ),
    )
    for field_name in (
        "max_initial_spent_rows",
        "max_first_gate_rows",
        "max_total_text_utf8_bytes",
        "max_catalog_atoms",
        "max_source_manifest_bytes",
        "max_direct_numerical_signals",
        "max_single_matrix_file_bytes",
        "max_total_matrix_file_bytes",
    ):
        parser.add_argument(
            "--first-untouched-gate-" + field_name.replace("_", "-"),
            dest="first_untouched_gate_" + field_name,
            type=int,
            help=(
                "Required in hierarchical mode. Explicit first-untouched-gate "
                "resource bound; there is no production default."
            ),
        )
    parser.add_argument(
        "--hierarchical-offline-review-packet-dir",
        type=Path,
        help=(
            "Fresh direct child of hierarchical-preparation-dir used for immutable "
            "human-readable and canonical offline review packets. Defaults to "
            "<preparation-dir>/offline_review_packet."
        ),
    )
    parser.add_argument(
        "--historical-discovery-prompt",
        metavar="PATH::SHA256",
        help=(
            "Byte-exact historical model-facing prompt and its known SHA-256. "
            "Required by prepare-only mode; bytes are embedded without normalization."
        ),
    )
    parser.add_argument(
        "--old-hierarchy-prompt",
        metavar="PATH::SHA256",
        help=(
            "Byte-exact old hierarchy prompt retained only for the prompt-quality "
            "ablation. Required by prepare-only mode."
        ),
    )
    parser.add_argument(
        "--prepare-hierarchical-discovery",
        "--hierarchical-prepare-only",
        dest="prepare_hierarchical_discovery",
        action="store_true",
        help=(
            "Prepare every fold and write the offline batch approval packet without "
            "consulting the hierarchy job cache or making a remote call. Use a fresh "
            "scratch output-dir for local provider caches."
        ),
    )
    parser.add_argument(
        "--hierarchical-approved-batch-sha256",
        help=(
            "Exact approval SHA-256 printed by prepare-only mode. Required for "
            "hierarchical execution and checked before any hierarchy cache lookup or "
            "remote JSON job."
        ),
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--outcome-type", choices=("binary", "continuous"), default="binary")
    parser.add_argument("--expected-outer-folds", type=int, default=5)
    parser.add_argument("--interaction-inner-folds", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument(
        "--hierarchical-max-atoms-per-chunk",
        type=int,
        default=HIERARCHICAL_DISCOVERY_MAX_ATOMS_PER_INTERPRET_JOB,
        help="Hard cap on complete semantic evidence atoms in one architecture chunk.",
    )
    parser.add_argument(
        "--hierarchical-max-bytes-per-chunk",
        type=int,
        default=DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK,
        help="Hard cap on canonical input bytes in one architecture chunk.",
    )
    parser.add_argument(
        "--hierarchical-max-semantic-member-ids-per-chunk",
        type=int,
        default=HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB,
        help=(
            "Hard cap on semantic member IDs in one architecture chunk; complete atoms "
            "are never split and an oversized atom is rejected before a model call."
        ),
    )
    parser.add_argument(
        "--hierarchical-max-cross-architecture-lookback-ids",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--hierarchical-max-cross-architecture-lookback-bytes",
        type=int,
        default=96_000,
    )
    parser.add_argument(
        "--hierarchical-max-rejection-lookback-ids-per-candidate",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--hierarchical-max-extraction-lookback-ids-per-feature",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--hierarchical-max-extraction-lookback-bytes-per-feature",
        type=int,
        default=96_000,
    )
    parser.add_argument(
        "--hierarchical-max-rejection-lookback-bytes-per-candidate",
        type=int,
        default=48_000,
    )
    parser.add_argument(
        "--hierarchical-review-max-evidence-ids",
        type=int,
        default=_DEFAULT_FROZEN_REVIEW_MAX_EVIDENCE_IDS,
    )
    parser.add_argument(
        "--hierarchical-review-max-evidence-bytes",
        type=int,
        default=_DEFAULT_FROZEN_REVIEW_MAX_EVIDENCE_BYTES,
    )
    parser.add_argument(
        "--post-extraction-review-rounds",
        type=int,
        default=DEFAULT_POST_EXTRACTION_REVIEW_ROUNDS,
        help=(
            "Bounded sequential outer-train review proposals. The v24 benchmark "
            "requires at least one round and defaults to two; the context-fit Stage-1 "
            "and frozen-embedding arguments below are required."
        ),
    )
    parser.add_argument("--post-extraction-review-max-operations", type=int, default=4)
    parser.add_argument(
        "--post-extraction-review-max-quality-retries",
        type=int,
        default=2,
        help=(
            "Spent-only extraction-quality correction attempts per sealed review gate. "
            "Failed attempts never inspect or consume the gate."
        ),
    )
    parser.add_argument("--post-extraction-review-min-partition-rows", type=int, default=8)
    parser.add_argument(
        "--review-stage1-config",
        type=Path,
        help=(
            "Historical all-source Stage-1 YAML/JSON config used to refit BoW, HTR, "
            "matched-pair, embedding, and TF-IDF evidence on each spent context."
        ),
    )
    parser.add_argument(
        "--review-semantic-witness-scientific-config",
        type=Path,
        help=(
            "Closed JSON scientific configuration for spent-review embedding "
            "retrieval and HTR semantic-witness TF-IDF projection. Required "
            "when post-extraction review is enabled; no defaults are applied."
        ),
    )
    parser.add_argument(
        "--review-embedding-cache-dir",
        type=Path,
        help=(
            "Frozen chunk-embedding cache. It is row-bound after proposals and cannot "
            "launch a language model on this host."
        ),
    )
    parser.add_argument(
        "--review-stage1-device",
        default="cuda:0",
        help="Explicit cpu or cuda:N device for context-fitted Stage-1/HTR work.",
    )
    parser.add_argument(
        "--review-stage1-bow-fold-parallelism",
        type=int,
        default=1,
        help=(
            "CPU workers used only across context-fitted BoW nuisance folds. HTR "
            "folds remain serial on the selected GPU."
        ),
    )
    parser.add_argument(
        "--review-stage1-bow-parallel-backend",
        choices=("threads", "processes"),
        default="threads",
        help=(
            "Execution backend for parallel BoW folds. Threads avoid child-interpreter "
            "model-library loading; HTR remains serial either way."
        ),
    )
    parser.add_argument(
        "--review-neural-query-cache-dir",
        type=Path,
        help=(
            "Fresh executable exact-context neural-query cache. It must be a direct "
            "child of output-dir and nonexistent or empty; defaults to a dedicated "
            "directory under output-dir. Existing checkpoints are never inputs; "
            "cross-process hierarchy runs use authenticated read-only cache overlays."
        ),
    )
    parser.add_argument(
        "--review-neural-query-config",
        type=Path,
        help="Optional JSON object overriding NeuralQueryAgenticForestConfig fields.",
    )
    parser.add_argument(
        "--review-neural-query-nuisance-folds",
        type=int,
        default=3,
        help="Cross-fitting folds for spent-context neural-query nuisance models.",
    )
    parser.add_argument(
        "--review-neural-query-device",
        action="append",
        default=[],
        metavar="DEVICE",
        help=(
            "Explicit cpu or cuda:N neural-query device; repeat to distribute the three "
            "query banks. Defaults to cuda:0."
        ),
    )
    parser.add_argument(
        "--final-upstream-meta-inner-folds",
        type=int,
        default=3,
        help=("Precommitted meta-inner folds for final outer-train OOF upstream banks."),
    )
    parser.add_argument(
        "--final-upstream-head-regularization",
        type=float,
        default=1.0,
        help="Fixed positive ridge penalty for the final upstream fusion head.",
    )
    parser.add_argument(
        "--final-upstream-max-orphan-features",
        type=int,
        help=(
            "Required with post-extraction review. Explicit rectangular-schema "
            "capacity for context-fitted orphan n-grams. If the capacity would "
            "bind, execution aborts before omitting any eligible feature."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proposal-max-tokens", type=int, default=25000)
    parser.add_argument("--extraction-max-tokens", type=int, default=25000)
    parser.add_argument("--proposal-schema-repair-attempts", type=int, default=2)
    parser.add_argument("--request-max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--extraction-batch-size", type=int, default=32)
    parser.add_argument("--max-variables-per-extraction-request", type=int, default=1)
    parser.add_argument(
        "--extraction-grouping-strategy",
        choices=("clinical_domain", "packed"),
        default="clinical_domain",
    )
    parser.add_argument(
        "--extraction-context-strategy",
        choices=("tail", "contract_lexical_rag"),
        default="tail",
    )
    parser.add_argument(
        "--extraction-max-text-length",
        required=True,
        type=int,
        help="Explicit context bound for this legacy extraction mode.",
    )
    parser.add_argument(
        "--extraction-prompt-version",
        default=EXTRACTION_PROMPT_VERSION,
    )
    parser.add_argument(
        "--read-only-cache-index",
        "--cache-index",
        action="append",
        type=Path,
        default=[],
        help="Authenticated FrozenExtractionCacheOverlay v2 index; may be repeated.",
    )
    parser.add_argument(
        "--read-only-review-spent-evidence-cache",
        action="append",
        default=[],
        metavar="PATH::SHA256",
        help=(
            "Authenticated historical context-fit spent-evidence cache entry; repeatable. "
            "The exact registered bytes are copied only on a current exact-binding hit "
            "into the fresh output-local writable cache."
        ),
    )
    parser.add_argument(
        "--read-only-context-fit-cache-index",
        action="append",
        default=[],
        metavar="INDEX_PATH::SHA256",
        help=(
            "Externally hashed closed index of complete context-fit review-gate and/or "
            "final-upstream cache bundles. Entries explicitly hash every matrix and "
            "their companion immutable run-input manifest; repeatable."
        ),
    )
    parser.add_argument(
        "--orphan-ngram-artifact",
        action="append",
        default=[],
        metavar="FOLD=PATH[::SHA256]",
        help=(
            "Trusted full-outer effect n-gram score artifact for one fold. "
            "Use this repeatable override when a resealed handoff retains a stale "
            "source path; bytes are SHA-256-bound even when no hash is supplied."
        ),
    )
    parser.add_argument(
        "--neural-query-moment-artifact",
        "--neural-query-artifact",
        action="append",
        default=[],
        metavar="FOLD=PATH[::SHA256]",
        help=(
            "Learned fold-local neural cohort-query evidence. The repeatable "
            "registration is bound to its file SHA-256 and the authoritative "
            "outer-train/held-out split before any remote client is constructed."
        ),
    )
    parser.add_argument(
        "--require-neural-query-moments",
        action="store_true",
        help=(
            "Require neural query moments without sparse substitution. In adaptive-review "
            "mode this requires the fold-local context-fit neural path; otherwise it "
            "requires an authenticated learned artifact for every outer fold."
        ),
    )
    parser.add_argument(
        "--require-orphan-ngrams",
        action="store_true",
        help="Fail if no effect orphan n-gram source can be resolved for any fold.",
    )
    parser.add_argument(
        "--modifier-interactions-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Interact only contracts assigned the effect-modifier role. This is on by "
            "default and required by the v24 causal-forest runtime."
        ),
    )
    parser.add_argument(
        "--evaluate-oracle-posthoc",
        action="store_true",
        help=(
            "After oracle-free predictions are frozen, project true_ite_prob "
            "and write separate post-hoc synthetic-benchmark metrics."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs/configuration without constructing clients or writing output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_benchmark(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


__all__ = [
    "_minimal_historical_applied_config",
    "ValidatedBenchmarkInputs",
    "build_agent_config",
    "build_applied_inference_config",
    "build_frozen_review_evidence_policy",
    "build_final_upstream_schema_config",
    "build_hierarchical_discovery_config",
    "build_parser",
    "build_review_neural_query_config",
    "extraction_prompt_cache_identity",
    "load_posthoc_oracle_projection",
    "main",
    "parse_neural_query_moment_artifact_registrations",
    "parse_orphan_ngram_artifact_registrations",
    "run_benchmark",
    "validate_benchmark_inputs",
    "validate_remote_endpoint_pool",
]
