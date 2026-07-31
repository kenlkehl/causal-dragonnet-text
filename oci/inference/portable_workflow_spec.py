"""Typed, path-neutral configuration for the portable all-evidence workflow.

The production runner historically accepted one large collection of command
line options.  These types separate scientific choices from deployment
locators and operational controls.  The immutable *run* request may bind all
three layers, while :meth:`ScientificWorkflowSpec.identity_payload` is stable
under relocation, device reassignment, worker-count changes, and pause/resume
controls.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .portable_identity import canonical_json, identity_sha256
from .stage1_execution_topology_policy import (
    Stage1ExecutionTopologyPolicy,
)
from .stage1_htr_operational_controls import (
    RoleNeutralHTROperationalControls,
)
from .neural_query_operational_controls import (
    RoleNeutralNeuralQueryOperationalControls,
)

from ..models.strict_causal_forest_runtime import (
    CAUSAL_FOREST_IMPLEMENTATION,
    STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
    StrictCausalForestDMLSpec,
    StrictCausalForestOperationalSpec,
    StrictCausalForestRuntimeConfig,
    StrictRandomForestClassifierSpec,
    StrictRandomForestRegressorSpec,
    StrictStratifiedKFoldSpec,
)
from .hierarchical_discovery_response_contract import (
    HierarchyWireBudget,
)
from .all_evidence_post_extraction_review import (
    CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY,
    GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
)
from .openai_compatible_json_discovery_job_runner import (
    Stage2GenerationPolicy,
)
from .post_extraction_scientific_policy import (
    PostExtractionScientificPolicy,
)

PORTABLE_SPEC_VERSION = "portable_all_evidence_scientific_workflow_v11"
DEPLOYMENT_PROFILE_VERSION = "portable_all_evidence_deployment_profile_v9"
STAGE1_EXECUTION_PROFILE_VERSION = "portable_stage1_execution_profile_v8"
STAGE1_PREFLIGHT_EXECUTION_POLICY_VERSION = (
    "portable_stage1_preflight_execution_policy_v1"
)
RESOURCE_PERFORMANCE_SAFETY_VERSION = "portable_resource_performance_safety_policy_v2"
RUN_CONTROL_VERSION = "portable_all_evidence_run_control_v2"
BINARY_PROBABILITY_DIFFERENCE = "binary_treatment_binary_outcome_probability_difference_v1"

EVIDENCE_FAMILIES = (
    "word_treatment_outcome",
    "word_residual_effect",
    "hierarchical_transformer",
    "matched_patient_uplift",
    "whole_cohort_embeddings",
    "cluster_local_embeddings",
    "lexical_semantic_retrieval",
    "tfidf_topics",
    "residual_tfidf_ngrams",
    "learned_neural_queries",
)

STRICT_FOREST_IMPLEMENTATION = CAUSAL_FOREST_IMPLEMENTATION
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CUDA_DEVICE = re.compile(r"^cuda:[0-9]+$")
_ARCHITECTURE_OPERATIONAL_KEYS = frozenset(
    {
        "api_key",
        "cache_dir",
        "completion_order",
        "dataset_path",
        "device",
        "devices",
        "endpoint",
        "gpu_id",
        "gpu_ids",
        "host",
        "hostname",
        "model_locator",
        "model_path",
        "n_jobs",
        "output_dir",
        "pid",
        "runtime_compatibility_class",
        "scratch_root",
        "server_url",
        "work_root",
        "worker",
        "workers",
    }
)


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            output[key] = value
        return output

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite JSON value {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable canonical JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _validate_path_neutral_architecture_profile(
    value: Any,
    *,
    path: str,
) -> None:
    """Reject deployment controls accidentally placed in scientific config."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if (
                lowered in _ARCHITECTURE_OPERATIONAL_KEYS
                or lowered.endswith("_locator")
                or lowered.endswith("worker_count")
            ):
                raise ValueError(
                    f"{path}.{key} is deployment metadata and cannot appear "
                    "in an architecture scientific profile"
                )
            _validate_path_neutral_architecture_profile(
                child,
                path=f"{path}.{key}",
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_path_neutral_architecture_profile(
                child,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(value, str) and (
        value.startswith("/")
        or _CUDA_DEVICE.fullmatch(value.lower()) is not None
        or value.startswith(("http://", "https://"))
    ):
        raise ValueError(f"{path} contains a deployment locator in a scientific profile")


@dataclass(frozen=True)
class EstimandDefinition:
    """Registered estimand interface.

    The registry makes unsupported estimands fail in configuration preflight
    while leaving artifact identities extensible to later implementations.
    """

    name: str
    treatment_type: str
    outcome_type: str
    scale: str
    implementation_version: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


class EstimandRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, EstimandDefinition] = {}

    def register(self, definition: EstimandDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"estimand is already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def resolve(self, name: str) -> EstimandDefinition:
        try:
            return self._definitions[str(name)]
        except KeyError as exc:
            raise ValueError(
                f"unsupported estimand {name!r}; available implementations are "
                f"{sorted(self._definitions)}"
            ) from exc

    @property
    def definitions(self) -> Mapping[str, EstimandDefinition]:
        return MappingProxyType(self._definitions)


ESTIMAND_REGISTRY = EstimandRegistry()
ESTIMAND_REGISTRY.register(
    EstimandDefinition(
        name=BINARY_PROBABILITY_DIFFERENCE,
        treatment_type="binary",
        outcome_type="binary",
        scale="probability_difference",
        implementation_version="strict_causal_forest_binary_v1",
    )
)


@dataclass(frozen=True)
class WorkflowColumns:
    unit_id: str
    text: str
    treatment: str
    outcome: str

    def __post_init__(self) -> None:
        values = tuple(
            _require_nonempty(value, label=f"columns.{name}")
            for name, value in asdict(self).items()
        )
        if len(values) != len(set(values)):
            raise ValueError("workflow column names must be distinct")


@dataclass(frozen=True)
class TextPreprocessingSpec:
    empty_text_policy: str
    repeated_character_policy: str
    repeated_character_threshold: int
    source_text_temporally_valid_by_design: bool

    def __post_init__(self) -> None:
        if self.empty_text_policy != "marker":
            raise ValueError("portable production preprocessing requires marker empty-text policy")
        if self.repeated_character_policy != "marker":
            raise ValueError(
                "portable production preprocessing requires marker repeated-character policy"
            )
        if int(self.repeated_character_threshold) < 1:
            raise ValueError("repeated_character_threshold must be positive")
        if not isinstance(self.source_text_temporally_valid_by_design, bool):
            raise ValueError("source_text_temporally_valid_by_design must be boolean")


@dataclass(frozen=True)
class SentenceEmbeddingEncoderSpec:
    """Closed scientific controls for cached sentence embeddings.

    Device placement and encode batch size are deliberately absent: they are
    deployment controls and must not change this identity.  Conversely, every
    setting here can change the stored numerical vectors or the exact text
    presented to the authenticated model and therefore has no production
    default.
    """

    prompt_policy: str
    prompt_name: str | None
    output_value: str
    precision: str
    convert_to_numpy: bool
    convert_to_tensor: bool
    truncate_dim: None
    pooling_output_policy: str
    model_dtype: str
    stored_array_dtype: str
    zero_vector_policy: str

    def __post_init__(self) -> None:
        if self.prompt_policy not in {"disabled", "authenticated_model_prompt_name"}:
            raise ValueError(
                "embedding_encoder.prompt_policy must be 'disabled' or "
                "'authenticated_model_prompt_name'"
            )
        if self.prompt_policy == "disabled":
            if self.prompt_name is not None:
                raise ValueError(
                    "embedding_encoder.prompt_name must be null when prompts are disabled"
                )
        elif (
            not isinstance(self.prompt_name, str)
            or not self.prompt_name
            or self.prompt_name != self.prompt_name.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in self.prompt_name)
        ):
            raise ValueError(
                "embedding_encoder.prompt_name must be one exact non-empty model prompt name"
            )
        if self.output_value != "sentence_embedding":
            raise ValueError(
                "portable embedding-cache v1 supports only output_value='sentence_embedding'"
            )
        if self.precision != "float32":
            raise ValueError(
                "portable embedding-cache v1 supports only precision='float32'; "
                "quantized encode precision can make deployment batch size scientific"
            )
        if self.convert_to_numpy is not True or self.convert_to_tensor is not False:
            raise ValueError(
                "portable embedding-cache v1 requires convert_to_numpy=true and "
                "convert_to_tensor=false"
            )
        if self.truncate_dim is not None:
            raise ValueError("embedding_encoder.truncate_dim must be null; truncation is forbidden")
        if self.pooling_output_policy != "single_process_sentence_embedding_v1":
            raise ValueError(
                "embedding_encoder.pooling_output_policy must be "
                "'single_process_sentence_embedding_v1'"
            )
        if self.model_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("embedding_encoder.model_dtype must be float32, float16, or bfloat16")
        if self.stored_array_dtype != "float32":
            raise ValueError(
                "portable embedding-cache v1 supports only stored_array_dtype='float32'"
            )
        if self.zero_vector_policy not in {"reject", "preserve"}:
            raise ValueError("embedding_encoder.zero_vector_policy must be 'reject' or 'preserve'")

    def as_configuration(self, *, normalize_embeddings: bool) -> dict[str, Any]:
        if not isinstance(normalize_embeddings, bool):
            raise TypeError("normalize_embeddings must be boolean")
        return {
            **asdict(self),
            "normalize_embeddings": normalize_embeddings,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SentenceEmbeddingEncoderSpec":
        if not isinstance(value, Mapping):
            raise TypeError("text_windows.embedding_encoder must be one object")
        expected = {field.name for field in fields(cls)}
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        if missing or extra:
            raise ValueError(
                "scientific text_windows.embedding_encoder must be explicitly and "
                f"exactly configured; missing={missing}, extra={extra}"
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class LosslessTextWindowSpec:
    """Configured request/chunk geometry with fail-closed nontruncation.

    ``embedding_max_chunks`` is a declared allocation bound, not a sampling
    rule.  The embedding builder must prove it is nonbinding for every note
    before fitting.  Likewise, tokenized chunks must fit the effective encoder
    sequence length or the run aborts.
    """

    complete_page_core_chars: int
    complete_page_context_chars: int
    complete_page_max_chars: int
    reconciliation_fan_in: int
    embedding_chunk_size_words: int
    embedding_chunk_overlap_words: int
    embedding_max_chunks: int
    embedding_chunk_selection: str
    embedding_max_seq_length: int | None
    embedding_normalize: bool
    embedding_encoder: SentenceEmbeddingEncoderSpec

    def __post_init__(self) -> None:
        integer_fields = (
            "complete_page_core_chars",
            "complete_page_context_chars",
            "complete_page_max_chars",
            "reconciliation_fan_in",
            "embedding_chunk_size_words",
            "embedding_chunk_overlap_words",
            "embedding_max_chunks",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"text_windows.{name} must be an integer")
        if self.complete_page_core_chars < 1:
            raise ValueError("complete_page_core_chars must be positive")
        if self.complete_page_context_chars < 0:
            raise ValueError("complete_page_context_chars must be nonnegative")
        if (
            self.complete_page_core_chars + 2 * self.complete_page_context_chars
            > self.complete_page_max_chars
        ):
            raise ValueError("complete-page core plus context exceeds complete_page_max_chars")
        if self.reconciliation_fan_in < 2:
            raise ValueError("reconciliation_fan_in must be at least two")
        if self.embedding_chunk_size_words < 1:
            raise ValueError("embedding_chunk_size_words must be positive")
        if not (0 <= self.embedding_chunk_overlap_words < self.embedding_chunk_size_words):
            raise ValueError(
                "embedding chunk overlap must be nonnegative and smaller than chunk size"
            )
        if self.embedding_max_chunks < 1:
            raise ValueError("embedding_max_chunks must be positive")
        if self.embedding_chunk_selection not in {"first", "last"}:
            raise ValueError("embedding_chunk_selection must be 'first' or 'last'")
        if self.embedding_max_seq_length is not None and (
            isinstance(self.embedding_max_seq_length, bool)
            or not isinstance(self.embedding_max_seq_length, int)
            or self.embedding_max_seq_length < 1
        ):
            raise ValueError("embedding_max_seq_length must be null or a positive integer")
        if not isinstance(self.embedding_normalize, bool):
            raise TypeError("embedding_normalize must be boolean")
        if not isinstance(self.embedding_encoder, SentenceEmbeddingEncoderSpec):
            raise TypeError(
                "embedding_encoder must be a fully configured SentenceEmbeddingEncoderSpec"
            )

    @property
    def complete_page_geometry(self) -> dict[str, int]:
        return {
            "core_chars": self.complete_page_core_chars,
            "context_chars": self.complete_page_context_chars,
            "max_page_chars": self.complete_page_max_chars,
        }

    @property
    def embedding_chunk_configuration(self) -> dict[str, Any]:
        return {
            "chunk_size_words": self.embedding_chunk_size_words,
            "chunk_overlap_words": self.embedding_chunk_overlap_words,
            "max_chunks": self.embedding_max_chunks,
            "chunk_selection": self.embedding_chunk_selection,
            "max_seq_length": self.embedding_max_seq_length,
            **self.embedding_encoder.as_configuration(
                normalize_embeddings=self.embedding_normalize
            ),
        }


@dataclass(frozen=True)
class HierarchyWireBudgetSpec:
    """Closed, versioned model-response capacity chosen by configuration."""

    budget_version: str
    max_opaque_identifier_chars: int
    max_generated_name_chars: int
    max_description_chars: int
    max_reason_chars: int
    max_ambiguity_chars: int
    max_free_text_chars: int
    max_generated_list_items: int
    max_feature_names_per_member: int
    max_findings_per_atomic_review: int
    max_pair_relation_peers_per_page: int
    max_definition_fold_inputs: int
    max_group_lookback_ids: int
    max_adaptive_review_targets: int
    max_interpret_atoms_per_job: int
    max_interpret_members_per_job: int
    max_interpret_name_chars: int
    max_interpret_description_chars: int
    max_interpret_ambiguity_chars: int
    max_interpret_reason_chars: int
    max_interpret_canonical_json_bytes: int
    max_interpret_transport_bytes: int
    interpret_generation_token_budget: int
    max_response_transport_bytes: int
    generation_token_budget: int

    def __post_init__(self) -> None:
        HierarchyWireBudget.from_mapping(asdict(self))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "HierarchyWireBudgetSpec":
        if not isinstance(value, Mapping):
            raise ValueError("hierarchy_wire_budget must be one object")
        HierarchyWireBudget.from_mapping(value)
        return cls(**dict(value))

    def runtime_budget(self) -> HierarchyWireBudget:
        return HierarchyWireBudget.from_mapping(asdict(self))


@dataclass(frozen=True)
class Stage2PromptProtocolSpec:
    """Configured Stage 2 response and lossless evidence-paging bounds.

    These values affect model-facing request composition and therefore belong
    to scientific identity.  Evidence-size bounds must page or fail closed;
    they never authorize dropping evidence atoms or prepared-note text.
    """

    proposal_max_tokens: int
    extraction_max_tokens: int
    model_context_window_tokens: int
    post_extraction_review_max_operations: int
    post_extraction_review_max_quality_retries: int
    post_extraction_review_min_partition_rows: int
    hierarchical_max_atoms_per_chunk: int
    hierarchical_max_bytes_per_chunk: int
    hierarchical_max_semantic_member_ids_per_chunk: int
    hierarchical_max_cross_architecture_lookback_ids: int
    hierarchical_max_cross_architecture_lookback_bytes: int
    hierarchical_max_extraction_lookback_ids_per_feature: int
    hierarchical_max_extraction_lookback_bytes_per_feature: int
    hierarchical_max_rejection_lookback_ids_per_candidate: int
    hierarchical_max_rejection_lookback_bytes_per_candidate: int
    hierarchical_review_max_evidence_ids: int
    hierarchical_review_max_evidence_bytes: int
    max_rendered_discovery_prompt_bytes: int
    selector_thinking_token_budget: int
    final_upstream_max_orphan_features: int
    review_neural_query_nuisance_folds: int
    final_upstream_meta_inner_folds: int
    final_upstream_head_regularization: float
    query_moment_max_queries: int
    query_moment_max_terms_per_query: int
    query_moment_max_chunks_per_query: int
    query_moment_fallback_chunks_per_query: int
    query_moment_max_excerpt_chars: int
    query_moment_max_term_chars: int
    query_moment_max_ngram_tokens: int
    extraction_grouping_strategy: str
    extraction_context_strategy: str
    extraction_prompt_version: str
    hierarchy_wire_budget: HierarchyWireBudgetSpec
    generation_policy: Stage2GenerationPolicy

    def __post_init__(self) -> None:
        integer_fields = set(self.__dataclass_fields__) - {
            "final_upstream_head_regularization",
            "extraction_grouping_strategy",
            "extraction_context_strategy",
            "extraction_prompt_version",
            "hierarchy_wire_budget",
            "generation_policy",
        }
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"stage2_prompt_protocol.{name} must be an integer")
        regularization = self.final_upstream_head_regularization
        if (
            isinstance(regularization, bool)
            or not isinstance(regularization, (int, float))
            or not math.isfinite(float(regularization))
            or float(regularization) <= 0.0
        ):
            raise ValueError(
                "stage2_prompt_protocol.final_upstream_head_regularization "
                "must be positive and finite"
            )
        object.__setattr__(
            self,
            "final_upstream_head_regularization",
            float(regularization),
        )
        if not isinstance(
            self.hierarchy_wire_budget,
            HierarchyWireBudgetSpec,
        ):
            raise TypeError(
                "stage2_prompt_protocol.hierarchy_wire_budget must be the "
                "closed HierarchyWireBudgetSpec"
            )
        if not isinstance(self.generation_policy, Stage2GenerationPolicy):
            raise TypeError(
                "stage2_prompt_protocol.generation_policy must be the closed "
                "Stage2GenerationPolicy"
            )
        proposal_families = (
            "interpret_architecture_chunk",
            "consolidate_architecture_candidates",
            "audit_architecture_coverage",
            "plan_cross_architecture_integration",
            "integrate_cross_architecture_candidates",
            "audit_rejected_candidates",
            "feature_proposal_review",
        )
        extraction_families = (
            "define_one_extraction_feature",
            "patient_feature_extraction",
        )
        for family in proposal_families:
            parameters = self.generation_policy.for_family(family)
            if parameters.max_tokens != self.proposal_max_tokens:
                raise ValueError(
                    "stage2_prompt_protocol proposal_max_tokens conflicts "
                    f"with generation family {family!r}"
                )
        for family in extraction_families:
            parameters = self.generation_policy.for_family(family)
            if parameters.max_tokens != self.extraction_max_tokens:
                raise ValueError(
                    "stage2_prompt_protocol extraction_max_tokens conflicts "
                    f"with generation family {family!r}"
                )
        for family, parameters in self.generation_policy.as_dict().items():
            if family == "schema_version":
                continue
            if (
                parameters["thinking_enabled"]
                and parameters["thinking_token_budget"] != self.selector_thinking_token_budget
            ):
                raise ValueError(
                    "stage2_prompt_protocol selector_thinking_token_budget "
                    f"conflicts with generation family {family!r}"
                )
            if not parameters["thinking_enabled"] and parameters["thinking_token_budget"] != 0:
                raise ValueError(
                    "disabled Stage 2 thinking must have zero budget for "
                    f"generation family {family!r}"
                )
            if (
                parameters["transport_max_retries"] != 0
                or parameters["schema_repair_attempts"] != 1
            ):
                raise ValueError(
                    "portable production Stage 2 requires zero transport "
                    "retries and exactly one schema-repair attempt for every "
                    f"generation family; invalid family={family!r}"
                )
        if self.extraction_grouping_strategy not in {
            "clinical_domain",
            "packed",
        }:
            raise ValueError("stage2_prompt_protocol.extraction_grouping_strategy is unsupported")
        if self.extraction_context_strategy != "complete_paged_v1":
            raise ValueError("portable v1 requires configured complete_paged_v1 extraction")
        _require_nonempty(
            self.extraction_prompt_version,
            label="stage2_prompt_protocol.extraction_prompt_version",
        )
        positive = integer_fields - {"post_extraction_review_max_quality_retries"}
        if any(getattr(self, name) < 1 for name in positive):
            raise ValueError("Stage 2 token, paging, lookback, and review bounds must be positive")
        if self.post_extraction_review_max_quality_retries < 0:
            raise ValueError("post_extraction_review_max_quality_retries must be nonnegative")
        if (
            self.post_extraction_review_max_operations
            > self.hierarchy_wire_budget.max_adaptive_review_targets
        ):
            raise ValueError(
                "post_extraction_review_max_operations exceeds the configured "
                "v1 response-contract capability"
            )
        if self.post_extraction_review_max_quality_retries > 8:
            raise ValueError(
                "post_extraction_review_max_quality_retries exceeds the v1 engine capability"
            )
        if self.post_extraction_review_min_partition_rows < 2:
            raise ValueError("post_extraction_review_min_partition_rows must be at least two")
        if self.review_neural_query_nuisance_folds < 2:
            raise ValueError("review_neural_query_nuisance_folds must be at least two")
        if self.final_upstream_meta_inner_folds < 2:
            raise ValueError("final_upstream_meta_inner_folds must be at least two")
        if self.query_moment_fallback_chunks_per_query > self.query_moment_max_chunks_per_query:
            raise ValueError(
                "query_moment_fallback_chunks_per_query cannot exceed "
                "query_moment_max_chunks_per_query"
            )
        if (
            self.hierarchical_max_atoms_per_chunk
            > self.hierarchy_wire_budget.max_interpret_atoms_per_job
        ):
            raise ValueError(
                "hierarchical_max_atoms_per_chunk exceeds the configured v1 "
                "response-contract capability"
            )
        if (
            self.hierarchical_max_semantic_member_ids_per_chunk
            > self.hierarchy_wire_budget.max_interpret_members_per_job
        ):
            raise ValueError(
                "hierarchical_max_semantic_member_ids_per_chunk exceeds the "
                "configured v1 response-contract capability"
            )
        required_proposal_tokens = (
            self.hierarchy_wire_budget.generation_token_budget + self.selector_thinking_token_budget
        )
        if self.proposal_max_tokens < required_proposal_tokens:
            raise ValueError(
                "proposal_max_tokens must cover the v1 visible-response budget "
                "plus selector_thinking_token_budget "
                f"({required_proposal_tokens})"
            )
        if max(self.proposal_max_tokens, self.extraction_max_tokens) >= (
            self.model_context_window_tokens
        ):
            raise ValueError(
                "model_context_window_tokens must be larger than every configured "
                "generation budget so an authenticated nonempty prompt can fit"
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "Stage2PromptProtocolSpec":
        if not isinstance(value, Mapping):
            raise ValueError("Stage 2 prompt protocol must be one object")
        expected = set(cls.__dataclass_fields__)
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        if missing or extra:
            raise ValueError(
                "Stage 2 prompt protocol must be explicitly and exactly "
                f"configured; missing={missing}, extra={extra}"
            )
        normalized = dict(value)
        normalized["hierarchy_wire_budget"] = HierarchyWireBudgetSpec.from_mapping(
            normalized["hierarchy_wire_budget"]
        )
        normalized["generation_policy"] = Stage2GenerationPolicy.from_mapping(
            normalized["generation_policy"]
        )
        return cls(**normalized)

    @classmethod
    def from_json(cls, path: Path | str) -> "Stage2PromptProtocolSpec":
        return cls.from_mapping(_strict_json(Path(path), label="Stage 2 prompt protocol"))

    def as_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["generation_policy"] = self.generation_policy.as_dict()
        return output


@dataclass(frozen=True)
class PostExtractionCausalReviewSpec:
    """Every scientific choice used by the bounded causal-review gate."""

    upstream_review_policy: str
    e_clip: float
    nuisance_ridge_alpha: float
    effect_ridge_alpha: float
    contract_complexity_penalty: float
    encoded_column_complexity_penalty: float
    minimum_score_improvement: float
    nuisance_relative_tolerance: float
    source_preservation_tolerance: float
    source_context_r_loss_relative_tolerance: float
    feature_bank_preservation_tolerance: float
    scientific_policy: PostExtractionScientificPolicy

    def __post_init__(self) -> None:
        if self.upstream_review_policy not in {
            CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY,
            GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
        }:
            raise ValueError(
                "post_extraction_causal_review.upstream_review_policy must be "
                "an explicitly registered review policy"
            )
        if not isinstance(
            self.scientific_policy,
            PostExtractionScientificPolicy,
        ):
            raise TypeError(
                "post_extraction_causal_review.scientific_policy must be the "
                "closed PostExtractionScientificPolicy"
            )
        numerical_fields = set(self.__dataclass_fields__) - {
            "upstream_review_policy",
            "scientific_policy",
        }
        for name in numerical_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"post_extraction_causal_review.{name} must be finite")
            object.__setattr__(self, name, float(value))
        if not 0.0 < self.e_clip < 0.5:
            raise ValueError("post_extraction_causal_review.e_clip must be in (0, 0.5)")
        if self.nuisance_ridge_alpha <= 0.0:
            raise ValueError("post_extraction_causal_review.nuisance_ridge_alpha must be positive")
        nonnegative = numerical_fields - {
            "e_clip",
            "nuisance_ridge_alpha",
        }
        for name in nonnegative:
            if getattr(self, name) < 0.0:
                raise ValueError(f"post_extraction_causal_review.{name} must be nonnegative")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "PostExtractionCausalReviewSpec":
        if not isinstance(value, Mapping):
            raise ValueError("post-extraction causal review must be one object")
        expected = set(cls.__dataclass_fields__)
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        if missing or extra:
            raise ValueError(
                "post-extraction causal review must be explicitly and exactly "
                f"configured; missing={missing}, extra={extra}"
            )
        normalized = dict(value)
        normalized["scientific_policy"] = PostExtractionScientificPolicy.from_mapping(
            normalized["scientific_policy"]
        )
        return cls(**normalized)

    @classmethod
    def from_json(
        cls,
        path: Path | str,
    ) -> "PostExtractionCausalReviewSpec":
        return cls.from_mapping(_strict_json(Path(path), label="post-extraction causal review"))

    def as_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["scientific_policy"] = self.scientific_policy.as_dict()
        return output


@dataclass(frozen=True)
class FoldReviewSpec:
    outer_folds: int
    review_rounds: int
    initial_training_partitions: int
    interaction_inner_folds: int
    tfidf_nested_calibration_folds: int

    def __post_init__(self) -> None:
        if int(self.outer_folds) < 2:
            raise ValueError("outer_folds must be at least two")
        if int(self.review_rounds) < 1:
            raise ValueError("review_rounds must be positive")
        if int(self.initial_training_partitions) < 1:
            raise ValueError("initial_training_partitions must be positive")
        if int(self.interaction_inner_folds) < 2:
            raise ValueError("interaction_inner_folds must be at least two")
        if int(self.tfidf_nested_calibration_folds) < 2:
            raise ValueError("tfidf_nested_calibration_folds must be at least two")

    @property
    def inner_partitions(self) -> int:
        return int(self.initial_training_partitions) + int(self.review_rounds)

    @property
    def logical_context_count(self) -> int:
        return int(self.outer_folds) * (1 + self.inner_partitions + int(self.review_rounds))


# Public API continuity: the authoritative portable representation is the
# exhaustive nested DML specification shared with the runtime layer.  The old
# flat dataclass no longer exists.
StrictCausalForestSpec = StrictCausalForestDMLSpec


@dataclass(frozen=True)
class ScientificWorkflowSpec:
    """All choices that can change the scientific result."""

    columns: WorkflowColumns
    clinical_question: str
    architecture_profiles: Mapping[str, Mapping[str, Any]]
    text_windows: LosslessTextWindowSpec
    stage2_prompt_protocol: Stage2PromptProtocolSpec
    post_extraction_causal_review: PostExtractionCausalReviewSpec
    max_candidate_variables: int
    causal_estimator: StrictCausalForestSpec
    estimand: str
    preprocessing: TextPreprocessingSpec
    folds: FoldReviewSpec
    seed: int
    seed_policy: str
    prompt_identities: Mapping[str, str]
    compatibility_version: str

    def __post_init__(self) -> None:
        _require_nonempty(self.clinical_question, label="clinical_question")
        ESTIMAND_REGISTRY.resolve(self.estimand)
        if self.compatibility_version != PORTABLE_SPEC_VERSION:
            raise ValueError(
                f"unsupported scientific compatibility version {self.compatibility_version!r}"
            )
        if int(self.seed) < 0:
            raise ValueError("scientific seed must be nonnegative")
        if (
            isinstance(self.max_candidate_variables, bool)
            or not isinstance(self.max_candidate_variables, int)
            or not 1 <= self.max_candidate_variables <= 20
        ):
            raise ValueError("max_candidate_variables must be an integer in [1, 20]")
        if self.seed_policy != "canonical_group_sha256_v1":
            raise ValueError("v1 requires content-derived canonical_group_sha256_v1 seeds")
        if int(self.stage2_prompt_protocol.final_upstream_meta_inner_folds) != int(
            self.folds.inner_partitions
        ):
            raise ValueError(
                "stage2_prompt_protocol.final_upstream_meta_inner_folds must "
                "equal the configured Stage 1 inner partition count so the "
                "final outer-train OOF bank is assembled from authenticated "
                "exact-inner transforms without a Stage 2 refit"
            )
        observed = set(self.architecture_profiles)
        required = set(EVIDENCE_FAMILIES)
        if observed != required:
            raise ValueError(
                "architecture_profiles must contain exactly all ten evidence families; "
                f"missing={sorted(required - observed)}, extra={sorted(observed - required)}"
            )
        for family in EVIDENCE_FAMILIES:
            profile = self.architecture_profiles[family]
            if not isinstance(profile, Mapping) or not profile:
                raise ValueError(f"architecture profile {family!r} must be a nonempty object")
            _validate_path_neutral_architecture_profile(
                profile,
                path=f"architecture_profiles.{family}",
            )
            canonical_json(profile)
        for name, digest in self.prompt_identities.items():
            _require_nonempty(name, label="prompt identity name")
            if _SHA256.fullmatch(str(digest)) is None:
                raise ValueError(f"prompt identity {name!r} must be one lowercase SHA-256")

    def identity_payload(self) -> dict[str, Any]:
        """Return scientific settings only, with no physical locator metadata."""

        return {
            "schema_version": PORTABLE_SPEC_VERSION,
            "columns": asdict(self.columns),
            "clinical_question": self.clinical_question,
            "estimand": ESTIMAND_REGISTRY.resolve(self.estimand).as_dict(),
            "preprocessing": asdict(self.preprocessing),
            "folds": {
                **asdict(self.folds),
                "inner_partitions": self.folds.inner_partitions,
                "logical_context_count": self.folds.logical_context_count,
            },
            "architecture_profiles": {
                family: copy.deepcopy(dict(self.architecture_profiles[family]))
                for family in EVIDENCE_FAMILIES
            },
            "text_windows": asdict(self.text_windows),
            "stage2_prompt_protocol": self.stage2_prompt_protocol.as_dict(),
            "post_extraction_causal_review": (self.post_extraction_causal_review.as_dict()),
            "max_candidate_variables": self.max_candidate_variables,
            "causal_estimator": self.causal_estimator.as_dict(),
            "seed": int(self.seed),
            "seed_policy": self.seed_policy,
            "prompt_identities": dict(sorted(self.prompt_identities.items())),
        }

    @property
    def scientific_sha256(self) -> str:
        return identity_sha256(self.identity_payload())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScientificWorkflowSpec":
        if not isinstance(value, Mapping):
            raise ValueError("scientific spec must be one object")
        required_sections = {
            "architecture_profiles",
            "causal_estimator",
            "clinical_question",
            "columns",
            "compatibility_version",
            "estimand",
            "folds",
            "max_candidate_variables",
            "preprocessing",
            "prompt_identities",
            "seed",
            "seed_policy",
            "stage2_prompt_protocol",
            "post_extraction_causal_review",
            "text_windows",
        }
        missing_sections = sorted(required_sections - set(value))
        if missing_sections:
            raise ValueError(
                "scientific spec omits required configured fields: " + ", ".join(missing_sections)
            )
        extra_sections = sorted(set(value) - required_sections)
        if extra_sections:
            raise ValueError(
                "scientific spec contains unsupported fields: " + ", ".join(extra_sections)
            )
        columns = value.get("columns")
        if not isinstance(columns, Mapping):
            raise ValueError("scientific spec requires columns")
        folds = value.get("folds")
        preprocessing = value.get("preprocessing")
        forest = value.get("causal_estimator")
        if not isinstance(folds, Mapping) or not isinstance(preprocessing, Mapping):
            raise ValueError("scientific folds and preprocessing must be objects")
        if not isinstance(forest, Mapping):
            raise ValueError("scientific causal_estimator must be an object")
        text_windows = value.get("text_windows")
        if not isinstance(text_windows, Mapping):
            raise ValueError(
                "scientific spec requires text_windows; production has no "
                "hard-coded paging or embedding window defaults"
            )
        stage2_prompt_protocol = value.get("stage2_prompt_protocol")
        if not isinstance(stage2_prompt_protocol, Mapping):
            raise ValueError(
                "scientific spec requires stage2_prompt_protocol; production "
                "has no hard-coded evidence paging or response-token defaults"
            )
        post_extraction_causal_review = value.get("post_extraction_causal_review")
        if not isinstance(post_extraction_causal_review, Mapping):
            raise ValueError(
                "scientific spec requires post_extraction_causal_review; "
                "production has no causal-review threshold defaults"
            )
        expected_columns = {"unit_id", "text", "treatment", "outcome"}
        expected_folds = {field.name for field in FoldReviewSpec.__dataclass_fields__.values()}
        expected_preprocessing = {
            "empty_text_policy",
            "repeated_character_policy",
            "repeated_character_threshold",
            "source_text_temporally_valid_by_design",
        }
        expected_forest = set(StrictCausalForestSpec.__dataclass_fields__)
        expected_text_windows = {
            field.name for field in LosslessTextWindowSpec.__dataclass_fields__.values()
        }
        expected_stage2_protocol = {
            field.name for field in Stage2PromptProtocolSpec.__dataclass_fields__.values()
        }
        for label, observed, expected in (
            ("columns", columns, expected_columns),
            ("folds", folds, expected_folds),
            ("preprocessing", preprocessing, expected_preprocessing),
            ("causal_estimator", forest, expected_forest),
            ("text_windows", text_windows, expected_text_windows),
            (
                "stage2_prompt_protocol",
                stage2_prompt_protocol,
                expected_stage2_protocol,
            ),
            (
                "post_extraction_causal_review",
                post_extraction_causal_review,
                set(PostExtractionCausalReviewSpec.__dataclass_fields__),
            ),
        ):
            missing = sorted(expected - set(observed))
            extra = sorted(set(observed) - expected)
            if missing or extra:
                raise ValueError(
                    f"scientific {label} must be explicitly and exactly configured; "
                    f"missing={missing}, extra={extra}"
                )
        parsed_text_windows = dict(text_windows)
        parsed_text_windows["embedding_encoder"] = SentenceEmbeddingEncoderSpec.from_mapping(
            parsed_text_windows["embedding_encoder"]
        )
        return cls(
            columns=WorkflowColumns(
                unit_id=columns["unit_id"],
                text=columns["text"],
                treatment=columns["treatment"],
                outcome=columns["outcome"],
            ),
            clinical_question=value.get("clinical_question"),
            architecture_profiles=copy.deepcopy(value.get("architecture_profiles") or {}),
            text_windows=LosslessTextWindowSpec(**parsed_text_windows),
            stage2_prompt_protocol=Stage2PromptProtocolSpec.from_mapping(stage2_prompt_protocol),
            post_extraction_causal_review=(
                PostExtractionCausalReviewSpec.from_mapping(post_extraction_causal_review)
            ),
            max_candidate_variables=value.get("max_candidate_variables"),
            estimand=value.get("estimand"),
            preprocessing=TextPreprocessingSpec(**dict(preprocessing)),
            folds=FoldReviewSpec(**dict(folds)),
            causal_estimator=StrictCausalForestSpec.from_mapping(forest),
            seed=int(value.get("seed")),
            seed_policy=value.get("seed_policy"),
            prompt_identities=dict(value.get("prompt_identities") or {}),
            compatibility_version=value.get("compatibility_version"),
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "ScientificWorkflowSpec":
        return cls.from_mapping(_strict_json(Path(path), label="scientific workflow spec"))


def normalize_device_policy(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = tuple(part for part in value.replace(",", " ").split() if part)
    else:
        raw = tuple(str(part).strip() for part in value if str(part).strip())
    if not raw:
        return ("auto",)
    lowered = tuple(part.lower() for part in raw)
    if "auto" in lowered:
        if len(lowered) != 1:
            raise ValueError("'auto' cannot be combined with explicit devices")
        return ("auto",)
    if "cpu" in lowered:
        if len(lowered) != 1:
            raise ValueError("'cpu' cannot be combined with accelerator devices")
        return ("cpu",)
    if len(lowered) != len(set(lowered)):
        raise ValueError("device policy cannot contain duplicates")
    if any(_CUDA_DEVICE.fullmatch(part) is None for part in lowered):
        raise ValueError("devices must be 'auto', 'cpu', or explicit cuda:N values")
    return lowered


@dataclass(frozen=True)
class ResourcePerformanceSafetyPolicy:
    """Operational resource and benchmark gates, never scientific settings."""

    gpu_max_allocation_fraction: float
    gpu_minimum_headroom_bytes: int
    minimum_multi_device_throughput_ratio: float
    maximum_coordination_proof_overhead_ratio: float
    maximum_ordinary_read_amplification: float
    minimum_benchmark_repetitions_per_scope: int
    read_counter_source: str
    fail_on_external_gpu_occupants: bool
    hierarchical_job_cache_max_entry_bytes: int
    first_untouched_gate_max_initial_spent_rows: int
    first_untouched_gate_max_first_gate_rows: int
    first_untouched_gate_max_total_text_utf8_bytes: int
    first_untouched_gate_max_catalog_atoms: int
    first_untouched_gate_max_source_manifest_bytes: int
    first_untouched_gate_max_direct_numerical_signals: int
    first_untouched_gate_max_single_matrix_file_bytes: int
    first_untouched_gate_max_total_matrix_file_bytes: int
    schema_version: str = RESOURCE_PERFORMANCE_SAFETY_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESOURCE_PERFORMANCE_SAFETY_VERSION:
            raise ValueError("unsupported resource/performance safety policy version")
        fraction = float(self.gpu_max_allocation_fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("gpu_max_allocation_fraction must be finite and in (0, 1]")
        object.__setattr__(self, "gpu_max_allocation_fraction", fraction)
        if (
            isinstance(self.gpu_minimum_headroom_bytes, bool)
            or not isinstance(self.gpu_minimum_headroom_bytes, int)
            or self.gpu_minimum_headroom_bytes < 0
        ):
            raise ValueError("gpu_minimum_headroom_bytes must be a nonnegative integer")
        for name in (
            "minimum_multi_device_throughput_ratio",
            "maximum_coordination_proof_overhead_ratio",
            "maximum_ordinary_read_amplification",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.minimum_benchmark_repetitions_per_scope, bool)
            or not isinstance(
                self.minimum_benchmark_repetitions_per_scope,
                int,
            )
            or self.minimum_benchmark_repetitions_per_scope < 2
        ):
            raise ValueError("minimum_benchmark_repetitions_per_scope must be at least two")
        if self.read_counter_source not in {
            "logical_read_bytes",
            "process_read_bytes",
        }:
            raise ValueError("read_counter_source must select logical or process bytes")
        if not isinstance(self.fail_on_external_gpu_occupants, bool):
            raise TypeError("fail_on_external_gpu_occupants must be explicitly boolean")
        capacity_fields = (
            "hierarchical_job_cache_max_entry_bytes",
            "first_untouched_gate_max_initial_spent_rows",
            "first_untouched_gate_max_first_gate_rows",
            "first_untouched_gate_max_total_text_utf8_bytes",
            "first_untouched_gate_max_catalog_atoms",
            "first_untouched_gate_max_source_manifest_bytes",
            "first_untouched_gate_max_direct_numerical_signals",
            "first_untouched_gate_max_single_matrix_file_bytes",
            "first_untouched_gate_max_total_matrix_file_bytes",
        )
        for name in capacity_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.first_untouched_gate_max_single_matrix_file_bytes
            > self.first_untouched_gate_max_total_matrix_file_bytes
        ):
            raise ValueError(
                "first_untouched_gate_max_single_matrix_file_bytes cannot exceed "
                "first_untouched_gate_max_total_matrix_file_bytes"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def content_sha256(self) -> str:
        return identity_sha256(self.as_dict())

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ResourcePerformanceSafetyPolicy":
        if not isinstance(value, Mapping):
            raise TypeError("resource_performance_safety must be one mapping")
        required = {field.name for field in fields(cls)}
        if set(value) != required:
            raise ValueError(
                "resource_performance_safety must configure every field "
                f"exactly; missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class Stage1PreflightExecutionPolicy:
    """Deployment-only bounds for reusable Stage 1 precomputation.

    These values select operational concurrency only.  They are intentionally
    absent from every Stage 1 scientific identity.
    """

    max_parallel_owners: int
    memory_budget_bytes: int
    estimated_owner_peak_bytes: int
    input_io_lane_cap: int
    publication_io_lane_cap: int
    authentication_io_lane_cap: int
    schema_version: str = STAGE1_PREFLIGHT_EXECUTION_POLICY_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != STAGE1_PREFLIGHT_EXECUTION_POLICY_VERSION
        ):
            raise ValueError(
                "unsupported Stage 1 preflight execution policy version"
            )
        for name in (
            "max_parallel_owners",
            "memory_budget_bytes",
            "estimated_owner_peak_bytes",
            "input_io_lane_cap",
            "publication_io_lane_cap",
            "authentication_io_lane_cap",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"Stage 1 preflight {name} must be a positive integer"
                )
        if self.estimated_owner_peak_bytes > self.memory_budget_bytes:
            raise ValueError(
                "Stage 1 preflight estimated_owner_peak_bytes exceeds "
                "its deployment memory budget"
            )

    @property
    def memory_lane_cap(self) -> int:
        return max(
            1,
            int(self.memory_budget_bytes)
            // int(self.estimated_owner_peak_bytes),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "Stage1PreflightExecutionPolicy":
        if not isinstance(value, Mapping):
            raise TypeError(
                "stage1_execution.preflight_execution_policy must be "
                "one mapping"
            )
        required = {row.name for row in fields(cls)}
        if set(value) != required:
            raise ValueError(
                "preflight_execution_policy must configure every field "
                f"exactly; missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        return cls(**dict(value))


def _backward_compatible_preflight_execution_policy() -> (
    Stage1PreflightExecutionPolicy
):
    """Conservative operational default for pre-policy deployment profiles."""

    return Stage1PreflightExecutionPolicy(
        max_parallel_owners=1,
        memory_budget_bytes=1,
        estimated_owner_peak_bytes=1,
        input_io_lane_cap=1,
        publication_io_lane_cap=1,
        authentication_io_lane_cap=1,
    )


@dataclass(frozen=True)
class Stage1ExecutionProfile:
    """Complete deployment-only Stage 1 execution selection."""

    resource_kind: str
    device_count: int
    scope_workers_per_device: int
    max_parallel_owners: int
    executor_mode: str
    persistent_slot_startup_timeout_seconds: float
    neural_query_topology: Stage1ExecutionTopologyPolicy
    htr_operational_controls: RoleNeutralHTROperationalControls
    neural_query_operational_controls: (
        RoleNeutralNeuralQueryOperationalControls
    )
    tfidf_parallel_backend: str
    selection_method: str
    benchmark_evidence_kind: str
    selected_candidate: str | None
    benchmark_result_sha256: str | None
    benchmark_result_locator: Path | None
    benchmark_workload_deployment_sha256: str | None
    benchmark_workload_deployment_locator: Path | None
    benchmark_publication_sha256: str | None
    benchmark_publication_locator: Path | None
    preflight_execution_policy: Stage1PreflightExecutionPolicy = field(
        default_factory=(
            _backward_compatible_preflight_execution_policy
        )
    )
    schema_version: str = STAGE1_EXECUTION_PROFILE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE1_EXECUTION_PROFILE_VERSION:
            raise ValueError("unsupported Stage 1 execution profile version")
        if self.resource_kind not in {"cpu", "accelerator"}:
            raise ValueError(
                "Stage 1 execution resource_kind must be cpu or accelerator"
            )
        for name in (
            "device_count",
            "scope_workers_per_device",
            "max_parallel_owners",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    "Stage 1 execution device count and per-device scope "
                    "concurrency must be positive integers"
                )
        if self.resource_kind == "cpu" and self.device_count != 1:
            raise ValueError("CPU Stage 1 execution requires exactly one device")
        if not isinstance(
            self.neural_query_topology,
            Stage1ExecutionTopologyPolicy,
        ):
            raise TypeError(
                "Stage 1 execution requires a typed neural-query topology"
            )
        topology_capacity = (
            self.neural_query_topology.effective_parallel_owners_for_shape(
                resource_kind=self.resource_kind,
                device_count=self.device_count,
                workers_per_device=self.scope_workers_per_device,
            )
        )
        if self.max_parallel_owners > topology_capacity:
            raise ValueError(
                "Stage 1 max_parallel_owners exceeds the effective "
                "device-topology capacity"
            )
        if not isinstance(
            self.preflight_execution_policy,
            Stage1PreflightExecutionPolicy,
        ):
            raise TypeError(
                "Stage 1 execution requires a typed preflight execution "
                "policy"
            )
        if (
            self.preflight_execution_policy.max_parallel_owners
            > self.max_parallel_owners
        ):
            raise ValueError(
                "Stage 1 preflight max_parallel_owners exceeds the "
                "deployment Stage 1 owner cap"
            )
        if not isinstance(
            self.htr_operational_controls,
            RoleNeutralHTROperationalControls,
        ):
            raise TypeError(
                "Stage 1 execution requires typed HTR operational controls"
            )
        htr_controls = self.htr_operational_controls
        devices_per_owner = (
            self.device_count
            if self.neural_query_topology.mode
            == "one_context_spanning_all_selected_devices"
            else 1
        )
        htr_slot_capacity = (
            devices_per_owner * htr_controls.fold_slots_per_device
        )
        if htr_controls.fold_parallelism > htr_slot_capacity:
            raise ValueError(
                "Stage 1 HTR fold parallelism exceeds the configured "
                "per-owner lease fold-slot capacity"
            )
        if (
            devices_per_owner > 1
            and htr_controls.fold_parallelism < devices_per_owner
        ):
            raise ValueError(
                "Stage 1 HTR fold parallelism must use every device in the "
                "owner lease"
            )
        if (
            htr_controls.fold_parallelism > 1
            and htr_controls.fold_parallel_backend != "processes"
        ):
            raise ValueError(
                "parallel HTR fold execution requires the "
                "process-isolated backend"
            )
        if (
            (
                htr_controls.fold_parallel_backend == "processes"
                or htr_controls.fold_parallelism > 1
            )
            and not htr_controls.reuse_tokenizer_and_chunk_plans
        ):
            raise ValueError(
                "process or parallel HTR fold execution requires one "
                "reusable complete tokenizer/chunk plan"
            )
        if not isinstance(
            self.neural_query_operational_controls,
            RoleNeutralNeuralQueryOperationalControls,
        ):
            raise TypeError(
                "Stage 1 execution requires typed neural-query operational "
                "controls"
            )
        neural_controls = self.neural_query_operational_controls
        neural_slot_capacity = (
            devices_per_owner
            * neural_controls.fold_slots_per_device
        )
        if (
            neural_controls.inner_fold_parallelism
            > neural_slot_capacity
            or neural_controls.bank_parallelism > neural_slot_capacity
        ):
            raise ValueError(
                "Stage 1 neural-query task parallelism exceeds the "
                "per-owner device-slot capacity"
            )
        if (
            devices_per_owner > 1
            and (
                neural_controls.inner_fold_parallelism
                < devices_per_owner
                or neural_controls.bank_parallelism
                < devices_per_owner
            )
        ):
            raise ValueError(
                "Stage 1 neural-query task parallelism must make every "
                "reserved device schedulable"
            )
        if (
            self.resource_kind == "accelerator"
            and max(
                neural_controls.inner_fold_parallelism,
                neural_controls.bank_parallelism,
            )
            > 1
            and neural_controls.fold_parallel_backend != "processes"
        ):
            raise ValueError(
                "parallel CUDA neural-query execution requires the "
                "process-isolated backend"
            )
        tfidf_backend = str(self.tfidf_parallel_backend).strip().lower()
        if tfidf_backend not in {"threads", "processes"}:
            raise ValueError(
                "Stage 1 TF-IDF parallel backend must be 'threads' or "
                "'processes'"
            )
        object.__setattr__(
            self,
            "tfidf_parallel_backend",
            tfidf_backend,
        )
        if self.executor_mode not in {
            "fresh_per_fit",
            "persistent_slots",
        }:
            raise ValueError(
                "Stage 1 execution executor_mode must be fresh_per_fit or "
                "persistent_slots"
            )
        if (
            isinstance(self.persistent_slot_startup_timeout_seconds, bool)
            or not isinstance(
                self.persistent_slot_startup_timeout_seconds,
                (int, float),
            )
            or not math.isfinite(
                float(self.persistent_slot_startup_timeout_seconds)
            )
            or float(self.persistent_slot_startup_timeout_seconds) <= 0
        ):
            raise ValueError(
                "persistent_slot_startup_timeout_seconds must be a finite "
                "positive deployment value"
            )
        evidence = (
            self.selected_candidate,
            self.benchmark_result_sha256,
            self.benchmark_result_locator,
            self.benchmark_workload_deployment_sha256,
            self.benchmark_workload_deployment_locator,
            self.benchmark_publication_sha256,
            self.benchmark_publication_locator,
        )
        if self.selection_method != "operator_configured":
            raise ValueError("unsupported Stage 1 execution selection method")
        if (
            self.benchmark_evidence_kind != "none"
            or any(value is not None for value in evidence)
        ):
            raise ValueError(
                "operator-configured Stage 1 execution cannot claim "
                "benchmark selection evidence"
            )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in (
            "benchmark_result_locator",
            "benchmark_workload_deployment_locator",
            "benchmark_publication_locator",
        ):
            if value[name] is not None:
                value[name] = str(value[name])
        return value

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "Stage1ExecutionProfile":
        if not isinstance(value, Mapping):
            raise TypeError("stage1_execution must be one mapping")
        required = {field.name for field in fields(cls)}
        backward_optional = {"preflight_execution_policy"}
        if (
            set(value) - backward_optional
            != required - backward_optional
            or set(value) - required
        ):
            raise ValueError(
                "stage1_execution must configure every field exactly; "
                f"missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        normalized = dict(value)
        preflight_policy = normalized.get(
            "preflight_execution_policy"
        )
        if preflight_policy is None:
            normalized["preflight_execution_policy"] = (
                _backward_compatible_preflight_execution_policy()
            )
        elif not isinstance(preflight_policy, Mapping):
            raise ValueError(
                "stage1_execution must configure "
                "preflight_execution_policy as one mapping"
            )
        else:
            normalized["preflight_execution_policy"] = (
                Stage1PreflightExecutionPolicy.from_mapping(
                    preflight_policy
                )
            )
        topology = normalized.get("neural_query_topology")
        if not isinstance(topology, Mapping):
            raise ValueError(
                "stage1_execution must explicitly configure "
                "neural_query_topology"
            )
        normalized["neural_query_topology"] = (
            Stage1ExecutionTopologyPolicy.from_mapping(topology)
        )
        htr_controls = normalized.get("htr_operational_controls")
        if not isinstance(htr_controls, Mapping):
            raise ValueError(
                "stage1_execution must explicitly configure "
                "htr_operational_controls"
            )
        normalized["htr_operational_controls"] = (
            RoleNeutralHTROperationalControls.from_mapping(htr_controls)
        )
        neural_controls = normalized.get(
            "neural_query_operational_controls"
        )
        if not isinstance(neural_controls, Mapping):
            raise ValueError(
                "stage1_execution must explicitly configure "
                "neural_query_operational_controls"
            )
        normalized["neural_query_operational_controls"] = (
            RoleNeutralNeuralQueryOperationalControls.from_mapping(
                neural_controls
            )
        )
        for name in (
            "benchmark_result_locator",
            "benchmark_workload_deployment_locator",
            "benchmark_publication_locator",
        ):
            if normalized.get(name) is not None:
                normalized[name] = Path(str(normalized[name]))
        return cls(**normalized)


@dataclass(frozen=True)
class DeploymentProfile:
    """Physical locators and resource/storage policy.

    This object is intentionally absent from the scientific identity payload.
    Content hashes derived from its dataset/model locators are added separately
    when a concrete run request is compiled.
    """

    dataset_path: Path
    durable_artifact_root: Path
    scratch_root: Path
    embedding_model_locator: Path
    htr_model_locator: Path
    stage1_profile_locator: Path
    query_profile_locator: Path
    embedding_batch_size: int
    resource_performance_safety: ResourcePerformanceSafetyPolicy
    forest_operational: StrictCausalForestOperationalSpec
    stage1_execution: Stage1ExecutionProfile
    cluster_preflight_parquet_compression: str
    embedding_model_name: str | None = None
    endpoint: str | None = None
    endpoint_model: str | None = None
    stage2_tokenizer_locator: Path | None = None
    devices: tuple[str, ...] = ("auto",)
    cpu_budget: int = 1
    response_concurrency: int = 1
    storage_backend: str = "posix"
    oracle_source: Path | None = None
    oracle_unit_id_column: str | None = None
    oracle_ite_column: str | None = None
    runtime_compatibility_class: str = "portable_python_posix_v1"
    schema_version: str = DEPLOYMENT_PROFILE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DEPLOYMENT_PROFILE_VERSION:
            raise ValueError(f"unsupported deployment profile version {self.schema_version!r}")
        if not isinstance(
            self.resource_performance_safety,
            ResourcePerformanceSafetyPolicy,
        ):
            raise TypeError("deployment profile requires typed resource_performance_safety")
        if not isinstance(
            self.forest_operational,
            StrictCausalForestOperationalSpec,
        ):
            raise TypeError("deployment profile requires typed forest_operational")
        if not isinstance(self.stage1_execution, Stage1ExecutionProfile):
            raise TypeError("deployment profile requires typed stage1_execution")
        if self.cluster_preflight_parquet_compression not in {
            "none",
            "zstd",
        }:
            raise ValueError(
                "cluster_preflight_parquet_compression must be explicitly "
                "configured as 'none' or 'zstd'"
            )
        object.__setattr__(self, "devices", normalize_device_policy(self.devices))
        if (
            self.stage1_execution.resource_kind == "cpu"
            and self.devices != ("cpu",)
        ) or (
            self.stage1_execution.resource_kind == "accelerator"
            and self.devices == ("cpu",)
        ):
            raise ValueError(
                "Stage 1 execution resource_kind conflicts with the device policy"
            )
        for name in (
            "cpu_budget",
            "response_concurrency",
            "embedding_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    "CPU budget, response concurrency, and embedding batch "
                    "size must be positive integers"
                )
        if int(self.forest_operational.requested_host_cpu_budget) != int(self.cpu_budget):
            raise ValueError(
                "forest_operational.requested_host_cpu_budget must equal "
                "the deployment cpu_budget"
            )
        if self.stage1_execution.max_parallel_owners > self.cpu_budget:
            raise ValueError(
                "Stage 1 max_parallel_owners exceeds the configured global "
                "host CPU budget"
            )
        if (
            math.floor(
                self.resource_performance_safety
                .maximum_ordinary_read_amplification
            )
            < 1
        ):
            raise ValueError(
                "maximum_ordinary_read_amplification must permit at least "
                "one Stage 1 preflight input lane"
            )
        owner_cpu_budget = (
            self.cpu_budget
            // self.stage1_execution.max_parallel_owners
        )
        if (
            self.stage1_execution.htr_operational_controls.fold_parallelism
            > owner_cpu_budget
        ):
            raise ValueError(
                "HTR fold parallelism exceeds one parallel owner's host "
                "CPU lease"
            )
        neural_controls = (
            self.stage1_execution.neural_query_operational_controls
        )
        if (
            max(
                neural_controls.inner_fold_parallelism,
                neural_controls.bank_parallelism,
            )
            * neural_controls.worker_cpu_threads
            > owner_cpu_budget
        ):
            raise ValueError(
                "neural-query task CPU threads exceed one parallel owner's "
                "host CPU lease"
            )
        if self.storage_backend not in {"posix", "local_posix", "sshfs"}:
            raise ValueError("unsupported storage backend")
        _require_nonempty(
            self.runtime_compatibility_class,
            label="runtime_compatibility_class",
        )
        oracle_values = (
            self.oracle_source,
            self.oracle_unit_id_column,
            self.oracle_ite_column,
        )
        if any(value is not None for value in oracle_values) and not all(
            value is not None for value in oracle_values
        ):
            raise ValueError("optional oracle source requires its ID and ITE columns")
        if (self.endpoint is None) != (self.endpoint_model is None):
            raise ValueError("Stage 2 endpoint and exact model must be supplied together")
        if self.endpoint is not None and self.stage2_tokenizer_locator is None:
            raise ValueError(
                "a Stage 2 endpoint requires stage2_tokenizer_locator for "
                "fail-closed local prompt-token accounting"
            )
        if self.embedding_model_name is not None:
            _require_nonempty(self.embedding_model_name, label="embedding_model_name")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DeploymentProfile":
        if "schema_version" not in value:
            raise ValueError("typed deployment profile must explicitly declare schema_version")
        aliases = {
            "dataset": "dataset_path",
            "work_root": "durable_artifact_root",
            "artifact_root": "durable_artifact_root",
            "embedding_local_model_path": "embedding_model_locator",
            "htr_local_model_path": "htr_model_locator",
            "stage1_profile_path": "stage1_profile_locator",
            "query_profile_path": "query_profile_locator",
            "stage2_tokenizer_path": "stage2_tokenizer_locator",
            "model": "endpoint_model",
            "model_name": "endpoint_model",
            "oracle_dataset": "oracle_source",
            "oracle_dataset_path": "oracle_source",
        }
        normalized = dict(value)
        for source, target in aliases.items():
            if source in normalized and target not in normalized:
                normalized[target] = normalized.pop(source)
        required = {field.name for field in fields(cls)}
        if set(normalized) != required:
            raise ValueError(
                "typed deployment profile must configure every field exactly; "
                f"missing={sorted(required - set(normalized))}, "
                f"extra={sorted(set(normalized) - required)}"
            )
        path_fields = (
            "dataset_path",
            "durable_artifact_root",
            "scratch_root",
            "embedding_model_locator",
            "htr_model_locator",
            "stage1_profile_locator",
            "query_profile_locator",
            "stage2_tokenizer_locator",
            "oracle_source",
        )
        for name in path_fields:
            if normalized.get(name) is not None:
                normalized[name] = Path(str(normalized[name]))
        safety = normalized.get("resource_performance_safety")
        if not isinstance(safety, Mapping):
            raise ValueError(
                "typed deployment profile must explicitly configure " "resource_performance_safety"
            )
        normalized["resource_performance_safety"] = ResourcePerformanceSafetyPolicy.from_mapping(
            safety
        )
        forest_operational = normalized.get("forest_operational")
        if not isinstance(forest_operational, Mapping):
            raise ValueError(
                "typed deployment profile must explicitly configure " "forest_operational"
            )
        normalized["forest_operational"] = StrictCausalForestOperationalSpec.from_mapping(
            forest_operational
        )
        stage1_execution = normalized.get("stage1_execution")
        if not isinstance(stage1_execution, Mapping):
            raise ValueError(
                "typed deployment profile must explicitly configure "
                "stage1_execution"
            )
        normalized["stage1_execution"] = Stage1ExecutionProfile.from_mapping(
            stage1_execution
        )
        if "devices" in normalized:
            normalized["devices"] = normalize_device_policy(normalized["devices"])
        return cls(**normalized)

    @classmethod
    def from_json(cls, path: Path | str) -> "DeploymentProfile":
        return cls.from_mapping(_strict_json(Path(path), label="deployment profile"))


def compile_strict_causal_forest_runtime(
    *,
    scientific: ScientificWorkflowSpec,
    deployment: DeploymentProfile,
) -> StrictCausalForestRuntimeConfig:
    """Bind exhaustive scientific settings to deployment-only operations."""

    if not isinstance(scientific, ScientificWorkflowSpec):
        raise TypeError("scientific must be ScientificWorkflowSpec")
    if not isinstance(deployment, DeploymentProfile):
        raise TypeError("deployment must be DeploymentProfile")
    return StrictCausalForestRuntimeConfig(
        schema_version=STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
        causal_forest=scientific.causal_estimator,
        operational=deployment.forest_operational,
    )


@dataclass(frozen=True)
class RunControl:
    """Operational controls excluded from scientific compatibility.

    ``validation_depth`` is a requested minimum. Production acceptance may
    enforce a stronger floor and must attest both the request and the depth
    actually achieved. ``log_level`` controls orchestrator lifecycle logging;
    durable progress and scientific validation are never suppressed by it.
    """

    resume: bool = False
    stop_after: str | None = None
    adopt_checkpoints: tuple[Path, ...] = ()
    trust_prior_adoption_attestations: tuple[Path, ...] = ()
    log_level: str = "INFO"
    validation_depth: str = "full"
    schema_version: str = RUN_CONTROL_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_CONTROL_VERSION:
            raise ValueError(f"unsupported run-control version {self.schema_version!r}")
        if not isinstance(self.resume, bool):
            raise TypeError("run-control resume must be explicitly boolean")
        if self.stop_after is not None:
            stop_after = _require_nonempty(
                self.stop_after,
                label="run-control stop_after",
            )
            object.__setattr__(self, "stop_after", stop_after)
        if self.validation_depth not in {"standard", "full", "fresh_terminal_audit"}:
            raise ValueError("unsupported validation depth")
        log_level = _require_nonempty(
            self.log_level,
            label="run-control log_level",
        ).upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("unsupported log level")
        object.__setattr__(self, "log_level", log_level)
        if isinstance(self.adopt_checkpoints, (str, bytes)) or not isinstance(
            self.adopt_checkpoints, Sequence
        ):
            raise TypeError("run-control adopt_checkpoints must be one ordered sequence")
        checkpoints: list[Path] = []
        for value in self.adopt_checkpoints:
            if isinstance(value, str) and not value.strip():
                raise ValueError("run-control checkpoint paths cannot be empty")
            checkpoint = Path(value)
            if not str(checkpoint).strip():
                raise ValueError("run-control checkpoint paths cannot be empty")
            checkpoints.append(checkpoint)
        if len(checkpoints) != len(set(map(str, checkpoints))):
            raise ValueError("run-control checkpoint paths cannot be duplicated")
        object.__setattr__(
            self,
            "adopt_checkpoints",
            tuple(checkpoints),
        )
        if isinstance(
            self.trust_prior_adoption_attestations,
            (str, bytes),
        ) or not isinstance(
            self.trust_prior_adoption_attestations,
            Sequence,
        ):
            raise TypeError(
                "run-control trusted prior adoption attestations must be "
                "one ordered sequence"
            )
        trusted_attestations: list[Path] = []
        for value in self.trust_prior_adoption_attestations:
            if isinstance(value, str) and not value.strip():
                raise ValueError(
                    "run-control trusted prior adoption attestation paths "
                    "cannot be empty"
                )
            attestation = Path(value)
            if not str(attestation).strip():
                raise ValueError(
                    "run-control trusted prior adoption attestation paths "
                    "cannot be empty"
                )
            trusted_attestations.append(attestation)
        if len(trusted_attestations) != len(
            set(map(str, trusted_attestations))
        ):
            raise ValueError(
                "run-control trusted prior adoption attestation paths "
                "cannot be duplicated"
            )
        object.__setattr__(
            self,
            "trust_prior_adoption_attestations",
            tuple(trusted_attestations),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "resume": self.resume,
            "stop_after": self.stop_after,
            "adopt_checkpoints": [str(value) for value in self.adopt_checkpoints],
            "trust_prior_adoption_attestations": [
                str(value)
                for value in self.trust_prior_adoption_attestations
            ],
            "log_level": self.log_level,
            "validation_depth": self.validation_depth,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunControl":
        if not isinstance(value, Mapping):
            raise TypeError("run control must be one mapping")
        expected = {field.name for field in fields(cls)}
        if set(value) != expected:
            raise ValueError(
                "run control must configure every field exactly; "
                f"missing={sorted(expected - set(value))}, "
                f"extra={sorted(set(value) - expected)}"
            )
        checkpoints = value.get("adopt_checkpoints")
        if isinstance(checkpoints, (str, bytes)) or not isinstance(checkpoints, Sequence):
            raise TypeError("run-control adopt_checkpoints must be one ordered list")
        trusted_attestations = value.get(
            "trust_prior_adoption_attestations"
        )
        if isinstance(
            trusted_attestations,
            (str, bytes),
        ) or not isinstance(trusted_attestations, Sequence):
            raise TypeError(
                "run-control trusted prior adoption attestations must be "
                "one ordered list"
            )
        return cls(
            resume=value.get("resume"),
            stop_after=value.get("stop_after"),
            adopt_checkpoints=tuple(checkpoints),
            trust_prior_adoption_attestations=tuple(
                trusted_attestations
            ),
            log_level=value.get("log_level"),
            validation_depth=value.get("validation_depth"),
            schema_version=value.get("schema_version"),
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "RunControl":
        return cls.from_mapping(_strict_json(Path(path), label="run control"))


__all__ = [
    "BINARY_PROBABILITY_DIFFERENCE",
    "DEPLOYMENT_PROFILE_VERSION",
    "EVIDENCE_FAMILIES",
    "ESTIMAND_REGISTRY",
    "EstimandDefinition",
    "EstimandRegistry",
    "FoldReviewSpec",
    "HierarchyWireBudgetSpec",
    "LosslessTextWindowSpec",
    "PORTABLE_SPEC_VERSION",
    "PostExtractionCausalReviewSpec",
    "RESOURCE_PERFORMANCE_SAFETY_VERSION",
    "ResourcePerformanceSafetyPolicy",
    "RoleNeutralNeuralQueryOperationalControls",
    "RUN_CONTROL_VERSION",
    "RunControl",
    "SentenceEmbeddingEncoderSpec",
    "STRICT_FOREST_IMPLEMENTATION",
    "ScientificWorkflowSpec",
    "STAGE1_EXECUTION_PROFILE_VERSION",
    "STAGE1_PREFLIGHT_EXECUTION_POLICY_VERSION",
    "Stage1PreflightExecutionPolicy",
    "Stage1ExecutionProfile",
    "Stage2PromptProtocolSpec",
    "StrictCausalForestDMLSpec",
    "StrictCausalForestOperationalSpec",
    "StrictCausalForestRuntimeConfig",
    "StrictCausalForestSpec",
    "StrictRandomForestClassifierSpec",
    "StrictRandomForestRegressorSpec",
    "StrictStratifiedKFoldSpec",
    "TextPreprocessingSpec",
    "WorkflowColumns",
    "DeploymentProfile",
    "canonical_json",
    "compile_strict_causal_forest_runtime",
    "identity_sha256",
    "normalize_device_policy",
]
