"""Resumable public orchestration for the all-evidence causal workflow."""

from __future__ import annotations

import argparse
import ast
import copy
import functools
import hashlib
import inspect
import importlib.metadata
import json
import logging
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence

from ..config import ClusterLocalEmbeddingScientificConfig
from .production_authenticated_tree_cache import (
    AUTHENTICATED_DIRECTORY_TREE_POLICY,
    AuthenticatedDirectoryTreeDriftError,
    authenticate_directory_tree,
)
from .production_stage1_bundle import (
    ProductionStage1BundleBuilder,
    STAGE1_REUSABLE_ASSEMBLED_PREFLIGHT_PRODUCER_IDENTITY,
    Stage1BundleBuildOptions,
)
from .production_text_preparation import (
    TextPreparationOptions,
    prepare_modeling_cohort,
    stable_file_sha256,
)
from .portable_artifacts import (
    ArtifactCompatibility,
    COMPLETE_PAYLOAD_TREE,
    MANIFEST_NAME,
    REGISTERED_PAYLOAD_PATHS_ONLY,
    ValidatedPortableArtifact,
    adopt_checkpoint,
    assert_validated_artifact_unchanged,
    materialize_portable_phase,
    publish_portable_reference_artifact,
    validate_checkpoint_adoption,
    validate_portable_artifact,
)
from .operator_trusted_checkpoint_adoption import (
    OPERATOR_TRUSTED_VALIDATION_POLICY,
    OperatorTrustedCheckpoint,
    adopt_checkpoint_from_prior_full_byte_attestation,
    validate_operator_trusted_checkpoint_adoption,
    validate_operator_trusted_portable_artifact,
)
from .performance_telemetry import TelemetryLedger
from .portable_resource_scheduler import (
    _logical_to_physical_cuda_indices,
)
from .portable_workflow_spec import (
    BINARY_PROBABILITY_DIFFERENCE,
    DeploymentProfile,
    EVIDENCE_FAMILIES,
    PostExtractionCausalReviewSpec,
    ResourcePerformanceSafetyPolicy,
    RunControl,
    ScientificWorkflowSpec,
    SentenceEmbeddingEncoderSpec,
    Stage1ExecutionProfile,
    Stage1PreflightExecutionPolicy,
    Stage2PromptProtocolSpec,
    StrictCausalForestOperationalSpec,
    StrictCausalForestRuntimeConfig,
    TextPreprocessingSpec,
    WorkflowColumns,
    compile_strict_causal_forest_runtime,
    identity_sha256,
    normalize_device_policy,
)
from .scientific_profile_identity import (
    scientific_profile_file_identity,
)
from .stage1_execution_topology_policy import (
    SUPPORTED_STAGE1_EXECUTION_TOPOLOGY_MODES,
    Stage1ExecutionTopologyPolicy,
)
from .stage1_htr_operational_controls import (
    RoleNeutralHTROperationalControls,
)
from .neural_query_operational_controls import (
    ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA,
    RoleNeutralNeuralQueryOperationalControls,
)

LOGGER = logging.getLogger(__name__)

WORKFLOW_SCHEMA = "production_all_evidence_workflow_v5"
PHASES = (
    "input_preparation",
    "embedding_cache",
    "stage1_preflight",
    "stage1_modeling",
    "handoff_validation",
    "stage2_canary",
    "stage2_inference",
    "oracle_evaluation",
    "terminal_validation",
)
STAGE1_ONLY_PHASES = PHASES[:5] + ("terminal_validation",)
IN_PLACE_RESUMABLE_PHASES = frozenset(
    {"stage1_modeling", "stage2_canary", "stage2_inference"}
)
EMBEDDING_CACHE_PHASE_SCHEMA = "production_embedding_cache_phase_result_v1"
STAGE1_PREFLIGHT_PHASE_SCHEMA = "production_stage1_preflight_phase_result_v2"
PORTABLE_ROLE_NEUTRAL_STAGE1_PHASE_SCHEMA = (
    "production_portable_role_neutral_stage1_phase_result_v2"
)
PORTABLE_ROLE_NEUTRAL_STAGE1_HANDOFF_BINDING_SCHEMA = (
    "production_portable_role_neutral_stage1_handoff_binding_v2"
)
STAGE1_COMPONENT_STORE_SCHEMA = (
    "production_stage1_scientific_component_store_v2"
)
LEGACY_STAGE1_COMPONENT_STORE_SCHEMAS = frozenset(
    {"production_stage1_scientific_component_store_v1"}
)
STAGE1_COMPONENT_STORE_MANIFEST = "component_store_manifest.json"
WORKFLOW_PROGRESS_SCHEMA = "production_all_evidence_workflow_progress_v1"
WORKFLOW_PHASE_MANIFEST_SCHEMA = "production_workflow_phase_manifest_v2"
WORKFLOW_ADOPTED_PHASE_MANIFEST_SCHEMA = "production_workflow_adopted_phase_manifest_v1"
WORKFLOW_CHECKPOINT_PUBLICATION_ATTESTATION_SCHEMA = (
    "production_workflow_checkpoint_publication_attestation_v1"
)
WORKFLOW_CHECKPOINT_DAG_VALIDATION_SCHEMA = "production_workflow_checkpoint_dag_validation_v1"
OPERATOR_TRUSTED_LEGACY_PHASE_PROJECTION_SCHEMA = (
    "operator_trusted_legacy_phase_compatibility_projection_v1"
)
WORKFLOW_GRANULAR_CHECKPOINT_INDEX_SCHEMA = (
    "production_workflow_granular_checkpoint_index_v1"
)
WORKFLOW_GRANULAR_CHECKPOINT_NODE_SCHEMA = (
    "production_workflow_granular_checkpoint_node_v1"
)
WORKFLOW_GRANULAR_CHECKPOINT_LOCATOR_SCHEMA = (
    "production_workflow_granular_checkpoint_locator_v1"
)
WORKFLOW_EXPECTED_GRANULAR_PLAN_SCHEMA = (
    "production_workflow_expected_granular_checkpoint_plan_v2"
)
GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS: Mapping[str, str] = {
    "prepared_stage1_context": (
        "production_prepared_stage1_context_checkpoint_v1"
    ),
    "tfidf_component": "production_stage1_tfidf_component_checkpoint_v1",
    "neural_query_component": (
        "production_stage1_neural_query_component_checkpoint_v1"
    ),
    "physical_scope_fit": (
        "production_stage1_physical_scope_fit_checkpoint_v1"
    ),
    "logical_scope_bindings": (
        "production_stage1_logical_scope_binding_checkpoint_v1"
    ),
    "row_map": "production_stage1_row_map_checkpoint_v1",
    "stage2_response_component": (
        "production_stage2_response_component_checkpoint_v1"
    ),
    "stage2_extraction_component": (
        "production_stage2_extraction_component_checkpoint_v1"
    ),
    "stage2_review_component": (
        "production_stage2_review_component_checkpoint_v1"
    ),
    "stage2_fold": "production_stage2_fold_checkpoint_v1",
}
WORKFLOW_LEGACY_PREFLIGHT_DECISION_SCHEMA = (
    "production_workflow_legacy_preflight_recompute_decision_v1"
)
WORKFLOW_TERMINAL_VALIDATION_SCHEMA = "production_all_evidence_fresh_terminal_validation_v1"
WORKFLOW_RUN_CONTROL_SELECTION_SCHEMA = (
    "production_all_evidence_run_control_selection_v1"
)
WORKFLOW_VALIDATION_ACHIEVEMENT_SCHEMA = (
    "production_all_evidence_validation_achievement_v1"
)
WORKFLOW_VALIDATION_POLICY_SCHEMA = (
    "production_all_evidence_validation_minimum_policy_v1"
)
WORKFLOW_STRUCTURED_LOG_EVENT_SCHEMA = (
    "production_all_evidence_structured_log_event_v1"
)
SOURCE_SNAPSHOT_EXECUTION_ENV = "OCI_PRODUCTION_SOURCE_SNAPSHOT_SHA256"

# Stage 1 component artifacts already seal their own implementation,
# configuration, model, row, and seed identities.  Keep the scope-plan
# namespace stable across orchestration-only changes so that a completed
# component remains reusable when an unrelated component or resume plumbing
# changes.  This value is the namespace used by the first released all-ten
# role-neutral production plan; changing scientific component code still
# changes that component's own producer identity and therefore its seal.
ROLE_NEUTRAL_STAGE1_COMPONENT_PLAN_NAMESPACE_IDENTITY = (
    "195ebe1a8229410f20144ddecef0b5cea"
    "3b73f7948813938b5eadf5c0a90f45d"
)

ADOPTABLE_PHASE_BY_ARTIFACT_KIND = {
    "prepared_cohort": "input_preparation",
    "embedding_cache": "embedding_cache",
    "clustered_preflight": "stage1_preflight",
    "stage1_handoff": "stage1_modeling",
    "stage2_canary": "stage2_canary",
    "frozen_prediction": "stage2_inference",
}
CHECKPOINT_COMPATIBILITY_PHASE_BY_ARTIFACT_KIND = {
    "prepared_cohort": "input_preparation",
    "embedding_cache": "embedding_cache",
    "clustered_preflight": "stage1_preflight",
    "prepared_stage1_context": "stage1_preflight",
    "physical_scope_fit": "stage1_modeling",
    "logical_scope_bindings": "stage1_modeling",
    "tfidf_component": "stage1_modeling",
    "neural_query_component": "stage1_modeling",
    "stage1_handoff": "stage1_modeling",
    "row_map": "stage1_modeling",
    "stage2_canary": "stage2_canary",
    "stage2_response_component": "stage2_inference",
    "stage2_extraction_component": "stage2_inference",
    "stage2_review_component": "stage2_inference",
    "stage2_fold": "stage2_inference",
    "frozen_prediction": "stage2_inference",
    "oracle_evaluation": "oracle_evaluation",
}
_REQUIRED_ADOPTED_ANCESTOR_KIND = {
    "embedding_cache": "prepared_cohort",
    "clustered_preflight": "embedding_cache",
    "stage1_handoff": "clustered_preflight",
    "stage2_canary": "stage1_handoff",
    "frozen_prediction": "stage1_handoff",
}

# These are scientific DAG edges, not execution-order metadata.  In
# particular, the frozen prediction is downstream of the accepted Stage 2
# canary as well as the Stage 1 handoff, and an oracle evaluation is
# downstream of the already-frozen prediction.
PORTABLE_CHECKPOINT_PHASE_SPECS: Mapping[str, Mapping[str, Any]] = {
    "input_preparation": {
        "artifact_kind": "prepared_cohort",
        "artifact_schema": "production_prepared_cohort_checkpoint_v1",
        "upstream_phases": (),
    },
    "embedding_cache": {
        "artifact_kind": "embedding_cache",
        "artifact_schema": "production_embedding_cache_checkpoint_v1",
        "upstream_phases": ("input_preparation",),
    },
    "stage1_preflight": {
        "artifact_kind": "clustered_preflight",
        "artifact_schema": "production_clustered_preflight_checkpoint_v1",
        "upstream_phases": ("embedding_cache",),
    },
    "stage1_modeling": {
        "artifact_kind": "stage1_handoff",
        "artifact_schema": "production_stage1_handoff_checkpoint_v1",
        "upstream_phases": ("stage1_preflight",),
    },
    "stage2_canary": {
        "artifact_kind": "stage2_canary",
        "artifact_schema": "production_stage2_canary_checkpoint_v1",
        "upstream_phases": ("stage1_modeling",),
    },
    "stage2_inference": {
        "artifact_kind": "frozen_prediction",
        "artifact_schema": "production_frozen_prediction_checkpoint_v1",
        "upstream_phases": ("stage1_modeling", "stage2_canary"),
    },
    "oracle_evaluation": {
        "artifact_kind": "oracle_evaluation",
        "artifact_schema": "production_oracle_evaluation_checkpoint_v1",
        "upstream_phases": ("stage2_inference",),
    },
}


def _authenticated_adopted_compact_preflight_parquet_compression(
    artifact: ValidatedPortableArtifact,
) -> str:
    """Return the deployment-only codec bound by an adopted preflight.

    The workflow phase binding is part of the portable artifact's
    content-addressed manifest.  ``materialize_portable_phase`` first checks
    the authenticated handle and then decodes only registered payload
    locators, so the physical-storage claim below is authenticated without
    adding it to scientific compatibility.
    """

    if artifact.manifest.get("artifact_kind") != "clustered_preflight":
        raise ValueError(
            "adopted compact Stage 1 preflight has the wrong artifact kind"
        )
    materialized = materialize_portable_phase(
        artifact,
        expected_phase="stage1_preflight",
    )
    result = materialized.get("result")
    identity = (
        result.get("cluster_preflight_identity")
        if isinstance(result, Mapping)
        else None
    )
    physical_storage = (
        identity.get("physical_storage")
        if isinstance(identity, Mapping)
        else None
    )
    required_storage = {
        "owner_concept_payload_format",
        "parquet_compression",
        "parquet_use_dictionary",
        "parquet_write_statistics",
        "parquet_data_page_version",
    }
    from .production_stage1_cluster_preflight_artifact_v2 import (
        PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME,
        PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_SCHEMA,
        PORTABLE_CLUSTER_PREFLIGHT_RESULT_SCHEMA,
    )

    raw_manifest_path = (
        result.get("cluster_preflight_manifest_path")
        if isinstance(result, Mapping)
        else None
    )
    manifest_path = (
        Path(raw_manifest_path)
        if isinstance(raw_manifest_path, str)
        else Path()
    )
    registered_paths = {
        Path(str(row["path"])).resolve(strict=True)
        for row in materialized.get("artifacts", ())
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    identity_body = (
        {
            key: copy.deepcopy(value)
            for key, value in identity.items()
            if key != "content_sha256"
        }
        if isinstance(identity, Mapping)
        else {}
    )
    if (
        not isinstance(result, Mapping)
        or result.get("schema_version") != STAGE1_PREFLIGHT_PHASE_SCHEMA
        or result.get("scientific_cluster_preflight")
        != "accepted_portable_compact_lossless_v2"
        or not isinstance(identity, Mapping)
        or identity.get("schema_version")
        != PORTABLE_CLUSTER_PREFLIGHT_RESULT_SCHEMA
        or identity.get("content_sha256")
        != identity_sha256(identity_body)
        or not isinstance(physical_storage, Mapping)
        or set(physical_storage) != required_storage
        or physical_storage.get("owner_concept_payload_format") != "parquet"
        or physical_storage.get("parquet_compression") not in {"none", "zstd"}
        or physical_storage.get("parquet_use_dictionary") is not False
        or physical_storage.get("parquet_write_statistics") is not False
        or physical_storage.get("parquet_data_page_version") != "1.0"
        or not manifest_path.is_absolute()
        or manifest_path.name != PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME
        or manifest_path.resolve(strict=True) not in registered_paths
    ):
        raise ValueError(
            "adopted compact Stage 1 preflight lacks authenticated physical "
            "Parquet storage metadata"
        )
    compact_manifest = _read_json_object(
        manifest_path,
        label="adopted compact Stage 1 preflight manifest",
    )
    compact_body = {
        key: copy.deepcopy(value)
        for key, value in compact_manifest.items()
        if key != "content_sha256"
    }
    if (
        compact_manifest.get("schema_version")
        != PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_SCHEMA
        or compact_manifest.get("status") != "complete"
        or compact_manifest.get("physical_storage")
        != dict(physical_storage)
        or compact_manifest.get("content_sha256")
        != identity_sha256(compact_body)
    ):
        raise ValueError(
            "adopted compact Stage 1 preflight manifest physical storage "
            "differs from its authenticated phase identity"
        )
    return str(physical_storage["parquet_compression"])


def _require_adopted_compact_preflight_parquet_compression(
    artifact: ValidatedPortableArtifact,
    *,
    expected: Any,
) -> None:
    if expected not in {"none", "zstd"}:
        raise ValueError(
            "workflow request lacks an explicit clustered-preflight "
            "Parquet compression"
        )
    observed = (
        _authenticated_adopted_compact_preflight_parquet_compression(
            artifact
        )
    )
    if observed != expected:
        raise ValueError(
            "adopted compact Stage 1 preflight Parquet compression "
            f"{observed!r} differs from requested deployment compression "
            f"{expected!r}"
        )


def _reconstruct_granular_checkpoint_index_from_artifacts(
    *,
    phase: str,
    artifacts: Sequence[ValidatedPortableArtifact],
) -> Mapping[str, Any]:
    """Rebuild the path-neutral granular index from authenticated nodes."""

    descriptors: list[dict[str, Any]] = []
    for ordinal, artifact in enumerate(artifacts):
        metadata = dict(artifact.artifact_metadata)
        descriptors.append(
            {
                "node_ordinal": ordinal,
                "node_key": metadata.get("node_key"),
                "artifact_id": artifact.artifact_id,
                "artifact_kind": artifact.manifest["artifact_kind"],
                "artifact_schema": artifact.manifest["artifact_schema"],
                "upstream_artifact_ids": list(
                    artifact.manifest["upstream_artifact_ids"]
                ),
                "artifact_metadata": metadata,
            }
        )
    coverage = _granular_checkpoint_coverage(descriptors)
    body = {
        "schema_version": WORKFLOW_GRANULAR_CHECKPOINT_INDEX_SCHEMA,
        "phase": phase,
        "node_count": len(descriptors),
        "nodes": descriptors,
        "coverage": coverage,
        "relative_filesystem_layout_included": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _validate_primary_granular_binding_digests(
    *,
    phase: str,
    primary_metadata: Mapping[str, Any],
    artifacts: Sequence[ValidatedPortableArtifact],
) -> Mapping[str, Any]:
    """Recompute both primary digest claims from authenticated nodes."""

    reconstructed = (
        _reconstruct_granular_checkpoint_index_from_artifacts(
            phase=phase,
            artifacts=artifacts,
        )
    )
    coverage = reconstructed["coverage"]
    if (
        list(
            primary_metadata.get("granular_artifact_ids") or ()
        )
        != [artifact.artifact_id for artifact in artifacts]
        or primary_metadata.get(
            "granular_index_content_sha256"
        )
        != reconstructed["content_sha256"]
        or primary_metadata.get(
            "granular_coverage_content_sha256"
        )
        != coverage["content_sha256"]
        or primary_metadata.get(
            "granular_artifact_kind_counts"
        )
        != coverage["artifact_kind_counts"]
    ):
        raise ValueError(
            f"{phase} primary granular digest binding changed"
        )
    return reconstructed


def _validated_stage1_granular_physical_fit_key(
    *,
    metadata: Mapping[str, Any],
    expected_identity: Any,
    expected_key_record: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    from .physical_fit_deduplication import PhysicalFitKey

    key_record = metadata.get("physical_fit_key_record")
    if not isinstance(key_record, Mapping):
        raise ValueError(
            "Stage 1 granular artifact lacks its complete physical-fit key"
        )
    key_body = {
        key: value
        for key, value in key_record.items()
        if key != "content_sha256"
    }
    key = PhysicalFitKey(**key_body)
    owner = str(metadata.get("physical_owner_scope_id", ""))
    if (
        not owner
        or key.as_dict() != dict(key_record)
        or metadata.get("physical_fit_key") != key.key
        or key.architecture_identity
        != expected_identity.architecture_identity
        or key.target != expected_identity.target
        or key.scientific_configuration_identity
        != expected_identity.scientific_configuration_identity
        or key.producer_identity
        != expected_identity.producer_identity
        or key.runtime_compatibility_class
        != expected_identity.runtime_compatibility_class
        or dict(key_record) != dict(expected_key_record)
    ):
        raise ValueError(
            "Stage 1 granular full physical-fit key changed"
        )
    return owner, copy.deepcopy(dict(key_record))


def _stage1_scope_plan_granular_expectations(
    *,
    scope_plan: Any,
    expected_granular_checkpoint_plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Project exact owner mappings and full keys from one validated plan."""

    granular_plan = _validate_expected_granular_checkpoint_plan(
        expected_granular_checkpoint_plan
    )
    physical_owners = [
        scope.scope_id for scope in scope_plan.physical_scopes
    ]
    logical_scopes = [scope.scope_id for scope in scope_plan.scopes]
    logical_to_owner = {
        scope.scope_id: scope_plan.physical_owner(scope.scope_id).scope_id
        for scope in scope_plan.scopes
    }
    if (
        physical_owners
        != list(granular_plan["stage1_physical_owner_scope_ids"])
        or logical_scopes
        != list(granular_plan["stage1_logical_scope_ids"])
        or logical_to_owner
        != dict(
            granular_plan["stage1_logical_to_physical_owner"]
        )
    ):
        raise ValueError(
            "authenticated Stage 1 content groups differ from the request plan"
        )
    return {
        "physical_owner_scope_ids": physical_owners,
        "logical_scope_ids": logical_scopes,
        "logical_to_physical_owner": logical_to_owner,
        "physical_fit_key_records_by_owner": {
            owner: scope_plan.physical_fit_key(owner).as_dict()
            for owner in physical_owners
        },
    }


def _load_authenticated_current_stage1_scope_plan(
    *,
    prepared_context_artifact: ValidatedPortableArtifact,
    expected_granular_checkpoint_plan: Mapping[str, Any],
    expected_stage1_physical_fit_identity: Mapping[str, Any],
    expected_global_seed: int,
) -> Any:
    """Rebuild the current row-level plan from an authenticated context."""

    from .prepared_stage1_context import (
        PREPARED_STAGE1_CONTEXT_MANIFEST_NAME,
        load_prepared_stage1_context,
    )
    from .production_stage1_scope_scheduler import (
        Stage1PhysicalFitIdentity,
        build_canonical_stage1_scope_plan,
    )

    assert_validated_artifact_unchanged(prepared_context_artifact)
    if (
        prepared_context_artifact.manifest.get("artifact_kind")
        != "prepared_stage1_context"
    ):
        raise ValueError(
            "Stage 1 full-key validation requires its prepared context"
        )
    manifest_rows = [
        row
        for row in prepared_context_artifact.payloads
        if row.relative_path == PREPARED_STAGE1_CONTEXT_MANIFEST_NAME
    ]
    if len(manifest_rows) != 1:
        raise ValueError(
            "prepared Stage 1 context has no unique authenticated manifest"
        )
    context = load_prepared_stage1_context(
        prepared_context_artifact.payload_root
        / manifest_rows[0].relative_path
    )
    granular_plan = _validate_expected_granular_checkpoint_plan(
        expected_granular_checkpoint_plan
    )
    scientific = context.scientific_identity
    projection = scientific.get(
        "stage1_request_scientific_projection"
    )
    exact_request = context.execution_locators.get(
        "exact_stage1_request"
    )
    registry = scientific.get("split_registry")
    raw_scope_plan = (
        exact_request.get("stage1_scope_plan")
        if isinstance(exact_request, Mapping)
        else None
    )
    projected_scope_plan = (
        projection.get("stage1_scope_plan")
        if isinstance(projection, Mapping)
        else None
    )
    if (
        not isinstance(registry, Mapping)
        or not isinstance(raw_scope_plan, Mapping)
        or not isinstance(projected_scope_plan, Mapping)
    ):
        raise ValueError(
            "prepared Stage 1 context lacks its authenticated scope plan"
        )
    expected_identity = Stage1PhysicalFitIdentity.from_mapping(
        expected_stage1_physical_fit_identity
    )
    rebuilt = build_canonical_stage1_scope_plan(
        registry=registry,
        registry_content_sha256=str(
            scientific["split_registry_content_sha256"]
        ),
        global_seed=int(expected_global_seed),
        physical_fit_identity=expected_identity,
        gpu_ids=(),
        review_rounds=int(granular_plan["review_rounds"]),
        initial_training_partitions=int(
            granular_plan["initial_training_partitions"]
        ),
        scope_workers_per_gpu=1,
        expected_outer_fold_count=int(
            granular_plan["outer_fold_count"]
        ),
        expected_inner_fold_count=int(
            granular_plan["inner_partition_count"]
        ),
    )
    if (
        raw_scope_plan.get("scientific_content_sha256")
        != rebuilt.scientific_content_sha256
        or projected_scope_plan.get("scientific_content_sha256")
        != rebuilt.scientific_content_sha256
    ):
        raise ValueError(
            "prepared Stage 1 context scope plan differs from the current request"
        )
    _stage1_scope_plan_granular_expectations(
        scope_plan=rebuilt,
        expected_granular_checkpoint_plan=granular_plan,
    )
    return rebuilt


def _validate_adopted_checkpoint_graph(
    artifacts: Sequence[ValidatedPortableArtifact],
    *,
    allowed_phases: Sequence[str],
    expected_granular_checkpoint_plan: Mapping[str, Any],
    expected_stage1_physical_fit_identity: Mapping[str, Any],
    expected_global_seed: int,
    require_prepared_stage1_context: bool,
) -> Mapping[str, str]:
    """Validate a closed portable DAG and return phase-to-artifact bindings."""

    granular_plan = _validate_expected_granular_checkpoint_plan(
        expected_granular_checkpoint_plan
    )
    from .production_stage1_scope_scheduler import (
        Stage1PhysicalFitIdentity,
    )

    expected_fit_identity = Stage1PhysicalFitIdentity.from_mapping(
        expected_stage1_physical_fit_identity
    )
    by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    if len(by_id) != len(artifacts):
        raise ValueError("checkpoint adoption cannot register duplicate artifact content")
    allowed = set(allowed_phases)
    phase_artifacts: dict[str, str] = {}
    kind_artifacts: dict[str, ValidatedPortableArtifact] = {}
    for artifact in artifacts:
        kind = str(artifact.manifest["artifact_kind"])
        if kind == "oracle_evaluation":
            raise ValueError(
                "oracle-evaluation checkpoint adoption must occur only after "
                "the consumer prediction is frozen; pre-run phase substitution "
                "is forbidden"
            )
        phase = ADOPTABLE_PHASE_BY_ARTIFACT_KIND.get(kind)
        binding = artifact.phase_binding
        if phase is None:
            if binding is not None:
                raise ValueError("component checkpoint cannot claim a workflow phase binding")
            continue
        if kind in kind_artifacts:
            raise ValueError(f"checkpoint adoption has multiple candidates for {kind}")
        kind_artifacts[kind] = artifact
        if binding is None:
            raise ValueError(f"{kind} checkpoint lacks an authenticated phase binding")
        if binding.get("phase") != phase:
            raise ValueError(f"{kind} checkpoint is bound to the wrong workflow phase")
        if phase not in allowed:
            raise ValueError(f"checkpoint phase is outside this workflow request: {phase}")
        phase_artifacts[phase] = artifact.artifact_id

    for artifact in artifacts:
        upstream = tuple(artifact.manifest["upstream_artifact_ids"])
        missing = sorted(set(upstream) - set(by_id))
        if missing:
            raise ValueError(
                "checkpoint upstream dependencies are absent from the adoption "
                f"request: {missing}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(artifact_id: str) -> None:
        if artifact_id in visited:
            return
        if artifact_id in visiting:
            raise ValueError("checkpoint adoption DAG contains a dependency cycle")
        visiting.add(artifact_id)
        for upstream_id in by_id[artifact_id].manifest["upstream_artifact_ids"]:
            visit(str(upstream_id))
        visiting.remove(artifact_id)
        visited.add(artifact_id)

    for artifact_id in by_id:
        visit(artifact_id)

    prepared_contexts = [
        artifact
        for artifact in artifacts
        if artifact.manifest.get("artifact_kind")
        == "prepared_stage1_context"
    ]
    preflight_id = phase_artifacts.get("stage1_preflight")
    authenticated_prepared_scope_plan: Any | None = None
    if require_prepared_stage1_context and preflight_id is not None:
        context = (
            prepared_contexts[0]
            if len(prepared_contexts) == 1
            else None
        )
        metadata = (
            {}
            if context is None
            else dict(context.artifact_metadata)
        )
        scientific_root = metadata.get(
            "scientific_content_root_sha256"
        )
        if (
            context is None
            or context.manifest.get("artifact_schema")
            != GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS[
                "prepared_stage1_context"
            ]
            or tuple(
                str(value)
                for value in context.manifest[
                    "upstream_artifact_ids"
                ]
            )
            != (preflight_id,)
            or set(metadata)
            != {
                "schema_version",
                "producer_phase",
                "node_ordinal",
                "node_key",
                "coverage_role",
                "scientific_content_root_sha256",
            }
            or metadata.get("schema_version")
            != WORKFLOW_GRANULAR_CHECKPOINT_NODE_SCHEMA
            or metadata.get("producer_phase") != "stage1_preflight"
            or metadata.get("node_ordinal") != 0
            or metadata.get("node_key")
            != "prepared_stage1_context"
            or metadata.get("coverage_role")
            != "prepared_stage1_context"
            or not isinstance(scientific_root, str)
            or len(scientific_root) != 64
            or any(
                character not in "0123456789abcdef"
                for character in scientific_root
            )
        ):
            raise ValueError(
                "adopted prepared Stage 1 context binding is invalid"
            )
        authenticated_prepared_scope_plan = (
            _load_authenticated_current_stage1_scope_plan(
                prepared_context_artifact=context,
                expected_granular_checkpoint_plan=granular_plan,
                expected_stage1_physical_fit_identity=(
                    expected_stage1_physical_fit_identity
                ),
                expected_global_seed=expected_global_seed,
            )
        )
    elif prepared_contexts:
        raise ValueError(
            "prepared Stage 1 context requires its typed adopted preflight"
        )

    for phase in ("stage1_modeling", "stage2_inference"):
        primary_id = phase_artifacts.get(phase)
        if primary_id is None:
            continue
        expected_stage1_plan_projection: Mapping[str, Any] | None = None
        expected_component_parent_ids: tuple[str, ...]
        if phase == "stage1_modeling":
            if len(prepared_contexts) != 1:
                raise ValueError(
                    "adopted Stage 1 requires one authenticated prepared context"
                )
            clustered_preflight_id = phase_artifacts.get(
                "stage1_preflight"
            )
            if (
                clustered_preflight_id is None
                or tuple(
                    str(value)
                    for value in prepared_contexts[
                        0
                    ].manifest["upstream_artifact_ids"]
                )
                != (clustered_preflight_id,)
            ):
                raise ValueError(
                    "adopted prepared Stage 1 context upstream edge changed"
                )
            authenticated_scope_plan = (
                authenticated_prepared_scope_plan
            )
            if authenticated_scope_plan is None:
                authenticated_scope_plan = (
                    _load_authenticated_current_stage1_scope_plan(
                        prepared_context_artifact=prepared_contexts[0],
                        expected_granular_checkpoint_plan=granular_plan,
                        expected_stage1_physical_fit_identity=(
                            expected_stage1_physical_fit_identity
                        ),
                        expected_global_seed=expected_global_seed,
                    )
                )
            expected_stage1_plan_projection = (
                _stage1_scope_plan_granular_expectations(
                    scope_plan=authenticated_scope_plan,
                    expected_granular_checkpoint_plan=granular_plan,
                )
            )
            expected_component_parent_ids = (
                prepared_contexts[0].artifact_id,
            )
        else:
            missing_parent_phases = [
                parent
                for parent in PORTABLE_CHECKPOINT_PHASE_SPECS[phase][
                    "upstream_phases"
                ]
                if parent not in phase_artifacts
            ]
            if missing_parent_phases:
                raise ValueError(
                    f"adopted {phase} lacks exact workflow parents: "
                    f"{missing_parent_phases}"
                )
            expected_component_parent_ids = tuple(
                phase_artifacts[parent]
                for parent in PORTABLE_CHECKPOINT_PHASE_SPECS[phase][
                    "upstream_phases"
                ]
            )
        primary = by_id[primary_id]
        metadata = primary.artifact_metadata
        required_metadata_fields = {
            "schema_version",
            "producer_phase",
            "granular_index_content_sha256",
            "granular_coverage_content_sha256",
            "granular_artifact_ids",
            "granular_terminal_artifact_ids",
            "granular_artifact_kind_counts",
        }
        all_ids = metadata.get("granular_artifact_ids")
        terminal_ids = metadata.get(
            "granular_terminal_artifact_ids"
        )
        kind_counts = metadata.get(
            "granular_artifact_kind_counts"
        )
        if (
            set(metadata) != required_metadata_fields
            or metadata.get("schema_version")
            != "workflow_primary_granular_coverage_binding_v1"
            or metadata.get("producer_phase") != phase
            or not isinstance(all_ids, list)
            or not all_ids
            or len(all_ids) != len(set(all_ids))
            or any(artifact_id not in by_id for artifact_id in all_ids)
            or not isinstance(terminal_ids, list)
            or not terminal_ids
            or not set(terminal_ids).issubset(set(all_ids))
            or not isinstance(kind_counts, Mapping)
        ):
            raise ValueError(
                f"adopted {phase} granular coverage binding is invalid"
            )
        phase_granular_ids = {
            artifact.artifact_id
            for artifact in artifacts
            if str(artifact.manifest["artifact_kind"])
            in GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS
            and CHECKPOINT_COMPATIBILITY_PHASE_BY_ARTIFACT_KIND.get(
                str(artifact.manifest["artifact_kind"])
            )
            == phase
        }
        if set(str(value) for value in all_ids) != phase_granular_ids:
            raise ValueError(
                f"adopted {phase} granular index omits or adds nodes"
            )
        ordered_granular = tuple(
            by_id[str(artifact_id)] for artifact_id in all_ids
        )
        reconstructed_index = (
            _validate_primary_granular_binding_digests(
                phase=phase,
                primary_metadata=metadata,
                artifacts=ordered_granular,
            )
        )
        reconstructed_coverage = reconstructed_index["coverage"]
        expected_counts = dict(
            granular_plan[
                (
                    "stage1_artifact_kind_counts"
                    if phase == "stage1_modeling"
                    else "stage2_artifact_kind_counts"
                )
            ]
        )
        if dict(kind_counts) != expected_counts:
            raise ValueError(
                f"adopted {phase} is self-consistent but incomplete"
            )

        observed_counts: dict[str, int] = {}
        observed_ordinals: list[int] = []
        physical_key_by_owner: dict[str, Mapping[str, Any]] = {}
        owners_by_kind: dict[str, list[str]] = {}
        logical_to_owner: dict[str, str] = {}
        stage2_folds: list[int] = []
        stage2_review_folds: list[int] = []
        for artifact_id in all_ids:
            granular = by_id[str(artifact_id)]
            granular_phase = (
                CHECKPOINT_COMPATIBILITY_PHASE_BY_ARTIFACT_KIND.get(
                    str(granular.manifest["artifact_kind"])
                )
            )
            granular_metadata = granular.artifact_metadata
            if (
                granular_phase != phase
                or granular_metadata.get("producer_phase") != phase
                or not isinstance(
                    granular_metadata.get("node_ordinal"),
                    int,
                )
            ):
                raise ValueError(
                    f"adopted {phase} granular artifact changed domain"
                )
            observed_ordinals.append(
                int(granular_metadata["node_ordinal"])
            )
            kind = str(granular.manifest["artifact_kind"])
            if kind in {
                "tfidf_component",
                "neural_query_component",
                "physical_scope_fit",
                "logical_scope_bindings",
            }:
                owner_hint = str(
                    granular_metadata.get(
                        "physical_owner_scope_id", ""
                    )
                )
                expected_key_records = (
                    {}
                    if expected_stage1_plan_projection is None
                    else expected_stage1_plan_projection[
                        "physical_fit_key_records_by_owner"
                    ]
                )
                expected_key_record = expected_key_records.get(
                    owner_hint
                )
                if not isinstance(expected_key_record, Mapping):
                    raise ValueError(
                        "adopted Stage 1 owner is absent from the current "
                        "authenticated scope plan"
                    )
                owner, validated_key_record = (
                    _validated_stage1_granular_physical_fit_key(
                        metadata=granular_metadata,
                        expected_identity=expected_fit_identity,
                        expected_key_record=expected_key_record,
                    )
                )
                prior_key = physical_key_by_owner.setdefault(
                    owner, validated_key_record
                )
                if prior_key != validated_key_record:
                    raise ValueError(
                        "adopted Stage 1 owner has inconsistent full "
                        "physical-fit keys"
                    )
                owners_by_kind.setdefault(kind, []).append(owner)
                if kind == "logical_scope_bindings":
                    logical_id = str(
                        granular_metadata.get(
                            "logical_scope_id", ""
                        )
                    )
                    if (
                        not logical_id
                        or logical_id in logical_to_owner
                    ):
                        raise ValueError(
                            "adopted Stage 1 logical scope is absent or duplicated"
                        )
                    logical_to_owner[logical_id] = owner
            elif kind in {
                "stage2_fold",
                "stage2_review_component",
            }:
                outer_fold = granular_metadata.get("outer_fold")
                if (
                    isinstance(outer_fold, bool)
                    or not isinstance(outer_fold, int)
                ):
                    raise ValueError(
                        "adopted Stage 2 fold identity is invalid"
                    )
                (
                    stage2_folds
                    if kind == "stage2_fold"
                    else stage2_review_folds
                ).append(int(outer_fold))
            observed_counts[kind] = observed_counts.get(kind, 0) + 1
        if (
            observed_ordinals != list(range(len(all_ids)))
            or dict(sorted(observed_counts.items()))
            != dict(kind_counts)
        ):
            raise ValueError(
                f"adopted {phase} granular coverage is incomplete"
            )
        if phase == "stage1_modeling":
            expected_owners = list(
                granular_plan[
                    "stage1_physical_owner_scope_ids"
                ]
            )
            expected_logical = list(
                granular_plan["stage1_logical_scope_ids"]
            )
            expected_logical_to_owner = dict(
                granular_plan[
                    "stage1_logical_to_physical_owner"
                ]
            )
            if (
                owners_by_kind.get("tfidf_component")
                != expected_owners
                or owners_by_kind.get("neural_query_component")
                != expected_owners
                or owners_by_kind.get("physical_scope_fit")
                != expected_owners
                or list(logical_to_owner) != expected_logical
                or logical_to_owner != expected_logical_to_owner
                or set(physical_key_by_owner) != set(
                    expected_owners
                )
            ):
                raise ValueError(
                    "adopted Stage 1 physical/logical plan changed"
                )
            row_maps = [
                by_id[str(artifact_id)]
                for artifact_id in all_ids
                if by_id[str(artifact_id)].manifest[
                    "artifact_kind"
                ]
                == "row_map"
            ]
            logical_ids = [
                str(artifact_id)
                for artifact_id in all_ids
                if by_id[str(artifact_id)].manifest[
                    "artifact_kind"
                ]
                == "logical_scope_bindings"
            ]
            if (
                len(row_maps) != 1
                or row_maps[0].artifact_metadata.get(
                    "logical_scope_count"
                )
                != len(expected_logical)
                or list(
                    row_maps[0].manifest[
                        "upstream_artifact_ids"
                    ]
                )
                != logical_ids
            ):
                raise ValueError(
                    "adopted Stage 1 row-map coverage changed"
                )
        else:
            expected_folds = list(
                granular_plan["stage2_fold_ids"]
            )
            expected_reviews = list(
                granular_plan["stage2_review_fold_ids"]
            )
            if (
                stage2_folds != expected_folds
                or stage2_review_folds != expected_reviews
            ):
                raise ValueError(
                    "adopted Stage 2 fold/review coverage changed"
                )
        _validate_exact_granular_upstream_edges(
            phase=phase,
            artifacts=ordered_granular,
            expected_plan=granular_plan,
            expected_external_upstream_artifact_ids=(
                expected_component_parent_ids
            ),
        )
        expected_terminal_kinds = (
            {"logical_scope_bindings", "row_map"}
            if phase == "stage1_modeling"
            else {"stage2_fold"}
        )
        expected_terminal_ids = [
            str(artifact_id)
            for artifact_id in all_ids
            if str(
                by_id[str(artifact_id)].manifest["artifact_kind"]
            )
            in expected_terminal_kinds
        ]
        if list(terminal_ids) != expected_terminal_ids:
            raise ValueError(
                f"adopted {phase} granular terminal kinds changed"
            )
        missing_primary_parent_phases = [
            parent
            for parent in PORTABLE_CHECKPOINT_PHASE_SPECS[phase][
                "upstream_phases"
            ]
            if parent not in phase_artifacts
        ]
        if missing_primary_parent_phases:
            raise ValueError(
                f"adopted {phase} lacks exact primary workflow parents: "
                f"{missing_primary_parent_phases}"
            )
        parent_ids = tuple(
            phase_artifacts[parent]
            for parent in PORTABLE_CHECKPOINT_PHASE_SPECS[phase][
                "upstream_phases"
            ]
        )
        if tuple(primary.manifest["upstream_artifact_ids"]) != (
            *parent_ids,
            *tuple(str(value) for value in terminal_ids),
        ):
            raise ValueError(
                f"adopted {phase} primary terminal edges changed"
            )

    def ancestor_ids(artifact_id: str) -> set[str]:
        output: set[str] = set()
        pending = list(by_id[artifact_id].manifest["upstream_artifact_ids"])
        while pending:
            upstream_id = str(pending.pop())
            if upstream_id in output:
                continue
            output.add(upstream_id)
            pending.extend(by_id[upstream_id].manifest["upstream_artifact_ids"])
        return output

    for kind, artifact in kind_artifacts.items():
        if kind == "prepared_cohort" and artifact.manifest["upstream_artifact_ids"]:
            raise ValueError("prepared cohort checkpoint must be a DAG root")
        required_kind = _REQUIRED_ADOPTED_ANCESTOR_KIND.get(kind)
        if required_kind is None:
            continue
        required = kind_artifacts.get(required_kind)
        if required is None or required.artifact_id not in ancestor_ids(artifact.artifact_id):
            raise ValueError(
                f"{kind} checkpoint lacks its authenticated " f"{required_kind} ancestor"
            )
    phase_ids = set(phase_artifacts.values())
    connected = set(phase_ids)
    for artifact_id in phase_ids:
        connected.update(ancestor_ids(artifact_id))
    # A typed prepared context is the one deliberate downstream component of
    # an adopted preflight checkpoint.  Its exact schema, edge, metadata, and
    # scope plan were authenticated above; arbitrary descendants remain
    # unrelated and fail closed below.
    if authenticated_prepared_scope_plan is not None:
        connected.add(prepared_contexts[0].artifact_id)
    unrelated = sorted(set(by_id) - connected)
    if unrelated:
        raise ValueError(
            "component checkpoints must be authenticated ancestors of one "
            f"substituted workflow phase; unrelated={unrelated}"
        )
    return phase_artifacts


class WorkflowPhaseHook(Protocol):
    """Injected implementation for one expensive production phase.

    Hooks receive paths and immutable scalar configuration only.  They must
    return a mapping with ``terminal_files`` so the workflow can independently
    hash and seal every published result.
    """

    def __call__(
        self,
        attempt_dir: Path,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class RoleNeutralStage1ProducerFactoriesBuilder(Protocol):
    """Bind all six real role-neutral producers to one prepared Stage 1 input."""

    def __call__(self, prepared: Any) -> Any: ...


class RoleNeutralStage1HandoffPublisher(Protocol):
    """Adapt an authenticated role-neutral execution to the Stage 2 loader."""

    def __call__(
        self,
        *,
        target_dir: Path,
        prepared: Any,
        role_neutral_execution_root: Path,
        role_neutral_execution_manifest: Mapping[str, Any],
    ) -> "RoleNeutralStage1HandoffPublication": ...


@dataclass(frozen=True)
class RoleNeutralStage1HandoffPublication:
    """Closed return type for the deployment-specific Stage 2 handoff adapter."""

    bundle_manifest_path: Path
    source_role_neutral_execution_content_sha256: str
    legacy_bundle_build_invoked: bool
    all_ten_role_neutral_execution_is_exclusive_evidence_source: bool
    handoff_kind: str | None = None
    handoff_scientific_content_sha256: str | None = None
    bundle_sha256: str | None = None
    stage2_provider: Any | None = None

    def __post_init__(self) -> None:
        digest = str(self.source_role_neutral_execution_content_sha256)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("handoff publication requires the role-neutral execution hash")
        if self.legacy_bundle_build_invoked is not False:
            raise ValueError(
                "portable role-neutral handoff adapters cannot invoke the "
                "legacy Stage 1 bundle build"
            )
        if self.all_ten_role_neutral_execution_is_exclusive_evidence_source is not True:
            raise ValueError(
                "portable handoff publication must use the authenticated "
                "role-neutral all-ten execution as its exclusive evidence source"
            )
        for label, value in (
            (
                "handoff scientific content",
                self.handoff_scientific_content_sha256,
            ),
            ("handoff bundle", self.bundle_sha256),
        ):
            if value is not None and (
                len(str(value)) != 64
                or any(character not in "0123456789abcdef" for character in str(value))
            ):
                raise ValueError(f"{label} must be one lowercase SHA-256")

    def as_dict(self) -> Mapping[str, Any]:
        """Return a path-safe offline-validation summary.

        The live provider handle is intentionally excluded.  This summary
        retains the established fresh-process report fields so terminal
        validation can recognize both the legacy and direct-loader modes.
        """

        manifest = _read_json_object(
            Path(self.bundle_manifest_path),
            label="role-neutral handoff manifest",
        )
        bundle_sha256 = self.bundle_sha256
        if bundle_sha256 is None:
            bundle_sha256 = str(manifest.get("bundle_sha256") or "")
        htr_preflight = manifest.get("htr_stage2_call_plan_preflight")
        body = {
            "schema_version": ("production_role_neutral_stage1_handoff_publication_v2"),
            "handoff_kind": self.handoff_kind,
            "stage1_inputs": {
                "bundle_sha256": bundle_sha256,
                "source_role_neutral_execution_content_sha256": (
                    self.source_role_neutral_execution_content_sha256
                ),
            },
            "handoff_scientific_content_sha256": (self.handoff_scientific_content_sha256),
            "all_ten_architectures_required": True,
            "per_architecture_interpretation_required": True,
            "raw_all_architecture_prompt_allowed": False,
            "independent_runtime_stage1_refit_allowed": False,
            "manual_digest_approval_required": False,
            "legacy_bundle_build_invoked": False,
            "evidence_payloads_copied": False,
            "derived_htr_aggregate_payloads_materialized": True,
            "raw_htr_token_arrays_materialized": False,
            "raw_htr_chunk_atoms_model_facing": False,
            "htr_stage2_call_plan_preflight": copy.deepcopy(
                htr_preflight
            ),
            "offline_handoff_validation_complete": True,
            "full_stage2_one_shot_runtime_complete": False,
        }
        return {**body, "content_sha256": _sha(body)}


@dataclass(frozen=True)
class ProductionRoleNeutralStage1Integration:
    """Deployment-bound execution and handoff seam for typed portable runs.

    The workflow owns scope planning, resource planning, all-six execution, and
    fresh validation.  Deployment code supplies only the scientific producer
    bindings, an executor implementation, and the currently deployment-specific
    adapter to the Stage 2 bundle loader.
    """

    producer_factories_builder: RoleNeutralStage1ProducerFactoriesBuilder
    executor: Any
    handoff_publisher: RoleNeutralStage1HandoffPublisher
    producer_factories_scientific_identity: Mapping[str, Any] | None = None
    physical_owner_executor_scientific_identity: Mapping[str, Any] | None = None
    handoff_publisher_scientific_identity: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not callable(self.producer_factories_builder):
            raise TypeError("role-neutral integration requires a producer-factories builder")
        if not callable(getattr(self.executor, "execute", None)):
            raise TypeError(
                "role-neutral integration requires a configured physical-owner " "executor"
            )
        if not callable(self.handoff_publisher):
            raise TypeError("role-neutral integration requires a Stage 2 handoff publisher")
        for label, identity in (
            (
                "producer-factories",
                self.producer_factories_scientific_identity,
            ),
            (
                "physical-owner executor",
                self.physical_owner_executor_scientific_identity,
            ),
            (
                "handoff publisher",
                self.handoff_publisher_scientific_identity,
            ),
        ):
            if identity is not None:
                _closed_explicit_callable_identity(
                    identity,
                    label=f"{label} scientific identity",
                )


@dataclass(frozen=True)
class ProductionAllEvidenceWorkflowHooks:
    """Optional cache/preflight/scheduler integrations.

    The public command uses the built-in implementations unless a hook is
    supplied by an embedding-cache relocator or parallel Stage 1 scheduler.
    Hooks are code-identity-bound into the immutable workflow request.
    """

    embedding_cache: WorkflowPhaseHook | None = None
    stage1_preflight: WorkflowPhaseHook | None = None
    stage1_modeling: WorkflowPhaseHook | None = None
    role_neutral_stage1: ProductionRoleNeutralStage1Integration | None = None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


_VALIDATION_DEPTH_RANK: Mapping[str, int] = {
    "standard": 0,
    "full": 1,
    "fresh_terminal_audit": 2,
}
_PRODUCTION_VALIDATION_MINIMUM = "fresh_terminal_audit"


def _resolve_validation_depth_policy(
    requested: str,
) -> Mapping[str, Any]:
    """Lift an operational request to the non-bypassable acceptance floor."""

    if requested not in _VALIDATION_DEPTH_RANK:
        raise ValueError("unsupported validation depth")
    effective = max(
        (requested, _PRODUCTION_VALIDATION_MINIMUM),
        key=_VALIDATION_DEPTH_RANK.__getitem__,
    )
    return {
        "schema_version": WORKFLOW_VALIDATION_POLICY_SCHEMA,
        "requested_minimum": requested,
        "production_minimum": _PRODUCTION_VALIDATION_MINIMUM,
        "effective_minimum": effective,
        "fresh_path_only_terminal_audit_required": True,
        "terminal_phase_override_can_satisfy_minimum": False,
    }


def _configure_cli_logging(log_level: str) -> None:
    """Configure only the public command's stderr logging threshold."""

    numeric_level = getattr(logging, str(log_level), None)
    if not isinstance(numeric_level, int):
        raise ValueError("unsupported log level")
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
    )
    LOGGER.setLevel(numeric_level)


def _emit_structured_workflow_log(
    *,
    configured_threshold: str,
    event_level: int,
    payload: Mapping[str, Any],
) -> bool:
    """Emit one canonical lifecycle record when RunControl permits it."""

    threshold = getattr(logging, str(configured_threshold), None)
    if not isinstance(threshold, int):
        raise ValueError("unsupported log level")
    if int(event_level) < threshold:
        return False
    LOGGER.log(int(event_level), _canonical(payload))
    return True


_PHASE_PRODUCER_ROOTS: Mapping[str, tuple[str, ...]] = {
    "input_preparation": ("oci/inference/production_text_preparation.py",),
    "embedding_cache": (
        "oci/inference/production_authenticated_tree_cache.py",
        "oci/inference/production_embedding_cache_builder.py",
        "oci/inference/production_embedding_cache_process.py",
        "oci/inference/production_embedding_cache_relocation.py",
    ),
    "stage1_preflight": (
        "oci/inference/production_stage1_bundle.py",
        "oci/inference/production_embedding_cache_phase_publication.py",
        "oci/inference/production_stage1_cluster_preflight_artifact_v2.py",
        "oci/inference/prepared_stage1_context.py",
        "oci/inference/role_neutral_embedding_group_execution.py",
    ),
    "stage1_modeling": (
        "oci/inference/production_embedding_cache_phase_publication.py",
        "oci/inference/production_stage1_role_neutral_execution.py",
        "oci/inference/production_role_neutral_process_executor.py",
        "oci/inference/production_role_neutral_persistent_executor.py",
        "oci/inference/production_role_neutral_producer_factories.py",
        "oci/inference/role_neutral_all_ten_binding.py",
        "oci/inference/role_neutral_bow_group_execution.py",
        "oci/inference/htr_attention_evidence_schema.py",
        "oci/inference/role_neutral_htr_group_execution.py",
        "oci/inference/role_neutral_matched_pair_group_execution.py",
        "oci/inference/role_neutral_embedding_group_execution.py",
        "oci/inference/role_neutral_tfidf_group_execution.py",
        "oci/inference/role_neutral_neural_query_group_execution.py",
        "oci/inference/production_role_neutral_stage2_handoff.py",
        "oci/inference/direct_upstream_numerical_reference_bank.py",
    ),
    "stage2_canary": (
        "scripts/canary_production_stage1_hierarchy.py",
        "oci/inference/production_stage1_hierarchy_one_shot.py",
        "oci/inference/openai_compatible_json_discovery_job_runner.py",
    ),
    "stage2_inference": (
        "oci/inference/production_stage1_hierarchy_one_shot.py",
        "oci/inference/hierarchical_all_architecture_discovery.py",
        "oci/inference/all_evidence_fusion_runner.py",
        "oci/inference/all_evidence_post_extraction_review.py",
        "oci/inference/final_context_fit_causal_forest_adapter.py",
        "oci/extraction/complete_paged.py",
        "oci/models/causal_forest_head.py",
        "oci/models/strict_causal_forest_runtime.py",
    ),
    "oracle_evaluation": ("oci/inference/production_oracle_evaluation.py",),
}
_SHARED_CHECKPOINT_PRODUCER_ROOTS = (
    "oci/inference/portable_artifacts.py",
    "oci/inference/portable_identity.py",
)
_SHARED_DEPENDENCY_LOCK_FILES = (
    "pyproject.toml",
    "uv.lock",
)
# These modules expose narrow utilities to the embedding-cache producer but
# also host unrelated downstream orchestration.  Hash the supplying module
# itself, while stopping traversal through imports that the selected cache
# utility never executes.  This prevents Stage 2-only modules from becoming
# accidental preparation/cache dependencies.
_PHASE_TRANSITIVE_IMPORT_LEAVES: Mapping[str, frozenset[str]] = {
    "embedding_cache": frozenset(
        {
            "oci/inference/__init__.py",
            "oci/inference/production_stage1_scope_scheduler.py",
            "oci/inference/review_spent_evidence_provider.py",
        }
    ),
    # This artifact module contains both the preflight sealer and the later
    # worker-side reconstruction method.  Preflight executes only the local
    # sealing/option-wire code; its scientific dependencies are already
    # explicit phase roots.  Do not traverse reconstruction-only imports into
    # the six modeling producers.
    "stage1_preflight": frozenset(
        {"oci/inference/prepared_stage1_context.py"}
    ),
}
_PHASE_WORKFLOW_METHODS: Mapping[str, tuple[str, ...]] = {
    "input_preparation": (),
    "embedding_cache": ("_run_embedding_cache_phase",),
    "stage1_preflight": ("_stage1_build_options", "_effective_stage1_profile"),
    "stage1_modeling": (
        "_stage1_build_options",
        "_run_portable_role_neutral_stage1_modeling",
    ),
    "stage2_canary": ("_stage2_options",),
    "stage2_inference": ("_stage2_options",),
    "oracle_evaluation": (),
}
_SHARED_CHECKPOINT_WORKFLOW_METHODS = (
    "_complete",
    "_publish_completed_phase_checkpoint",
)
_SHARED_CHECKPOINT_MODULE_CALLABLES = (
    "_bind_workflow_scientific_identity",
    "_derive_expected_granular_checkpoint_plan",
    "_validate_adopted_checkpoint_graph",
    "validate_published_workflow_checkpoint_dag",
)
_PHASE_CHECKPOINT_MODULE_CALLABLES: Mapping[
    str, tuple[str, ...]
] = {
    phase: (
        (
            "_validate_granular_checkpoint_index_from_paths",
            "_granular_checkpoint_coverage",
        )
        if phase
        in {
            "stage1_preflight",
            "stage1_modeling",
            "stage2_inference",
        }
        else ()
    )
    + (
        (
            "_granular_primary_metadata_from_index",
            "_reconstruct_granular_checkpoint_index_from_artifacts",
        )
        if phase in {"stage1_modeling", "stage2_inference"}
        else ()
    )
    for phase in PORTABLE_CHECKPOINT_PHASE_SPECS
}
_MODULE_CALLABLE_PHASE_DOMAINS: Mapping[str, frozenset[str]] = {
    "_validate_granular_checkpoint_index_from_paths": frozenset(
        {
            "stage1_preflight",
            "stage1_modeling",
            "stage2_inference",
        }
    ),
    "_validate_granular_handles_against_plan": frozenset(
        {"stage1_modeling", "stage2_inference"}
    ),
    "_granular_primary_metadata_from_index": frozenset(
        {"stage1_modeling", "stage2_inference"}
    ),
    "_reconstruct_granular_checkpoint_index_from_artifacts": (
        frozenset({"stage1_modeling", "stage2_inference"})
    ),
    "_validate_primary_granular_binding_digests": frozenset(
        {"stage1_modeling", "stage2_inference"}
    ),
}


def _phase_predicate_value(node: ast.AST, *, phase: str) -> bool | None:
    """Evaluate only closed predicates over the workflow ``phase`` name."""

    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return bool(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _phase_predicate_value(node.operand, phase=phase)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [
            _phase_predicate_value(child, phase=phase)
            for child in node.values
        ]
        if isinstance(node.op, ast.And):
            if False in values:
                return False
            return True if all(value is True for value in values) else None
        if isinstance(node.op, ast.Or):
            if True in values:
                return True
            return False if all(value is False for value in values) else None
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and len(node.comparators) == 1
    ):
        left = node.left
        right = node.comparators[0]
        if isinstance(left, ast.Name) and left.id == "phase":
            if isinstance(right, ast.Constant) and isinstance(
                right.value, str
            ):
                if isinstance(node.ops[0], ast.Eq):
                    return phase == right.value
                if isinstance(node.ops[0], ast.NotEq):
                    return phase != right.value
            if isinstance(
                right,
                (ast.Tuple, ast.List, ast.Set),
            ) and all(
                isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                for child in right.elts
            ):
                choices = {
                    str(child.value) for child in right.elts
                }
                if isinstance(node.ops[0], ast.In):
                    return phase in choices
                if isinstance(node.ops[0], ast.NotIn):
                    return phase not in choices
    return None


class _PhaseLocalAstPruner(ast.NodeTransformer):
    """Discard branches that provably belong to a different phase."""

    def __init__(self, phase: str) -> None:
        self.phase = phase

    def visit_If(self, node: ast.If) -> Any:  # noqa: N802
        value = _phase_predicate_value(node.test, phase=self.phase)
        if value is True:
            return [
                transformed
                for child in node.body
                for transformed in self._visit_statement(child)
            ]
        if value is False:
            return [
                transformed
                for child in node.orelse
                for transformed in self._visit_statement(child)
            ]
        return self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> Any:  # noqa: N802
        value = _phase_predicate_value(node.test, phase=self.phase)
        if value is True:
            return self.visit(node.body)
        if value is False:
            return self.visit(node.orelse)
        return self.generic_visit(node)

    def _visit_statement(self, node: ast.stmt) -> list[ast.stmt]:
        value = self.visit(node)
        if value is None:
            return []
        if isinstance(value, list):
            return [child for child in value if isinstance(child, ast.stmt)]
        return [value] if isinstance(value, ast.stmt) else []


def _phase_local_callable_tree(value: Any, *, phase: str) -> ast.Module:
    try:
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(value))
        )
    except (OSError, TypeError, IndentationError, SyntaxError):
        code = getattr(value, "__code__", None)
        if code is None:
            raise ValueError(
                "phase producer dependency lacks stable callable source"
            )

        def normalized_code(child: Any) -> Any:
            if hasattr(child, "co_code") and hasattr(
                child, "co_consts"
            ):
                return {
                    "argcount": int(child.co_argcount),
                    "posonlyargcount": int(
                        child.co_posonlyargcount
                    ),
                    "kwonlyargcount": int(
                        child.co_kwonlyargcount
                    ),
                    "flags": int(child.co_flags),
                    "bytecode_hex": bytes(child.co_code).hex(),
                    "names": list(child.co_names),
                    "varnames": list(child.co_varnames),
                    "freevars": list(child.co_freevars),
                    "cellvars": list(child.co_cellvars),
                    "constants": [
                        normalized_code(constant)
                        for constant in child.co_consts
                    ],
                }
            accepted, normalized = _closed_code_constant(child)
            if accepted:
                return normalized
            return {
                "type": (
                    f"{type(child).__module__}."
                    f"{type(child).__qualname__}"
                )
            }

        fallback_digest = _sha(
            {
                "schema_version": (
                    "normalized_callable_code_fallback_v1"
                ),
                "module": str(
                    getattr(value, "__module__", "")
                ),
                "qualname": str(
                    getattr(value, "__qualname__", "")
                ),
                "code": normalized_code(code),
            }
        )
        tree = ast.parse(
            "def _source_fragment_fallback():\n"
            f"    return {fallback_digest!r}\n"
        )
    transformed = _PhaseLocalAstPruner(phase).visit(tree)
    assert isinstance(transformed, ast.Module)
    return ast.fix_missing_locations(transformed)


def _ast_tree_sha256(
    tree: ast.AST,
    *,
    schema_version: str,
) -> str:
    return _sha(
        {
            "schema_version": schema_version,
            "ast": ast.dump(
                tree,
                annotate_fields=True,
                include_attributes=False,
            ),
        }
    )


def _closed_code_constant(value: Any) -> tuple[bool, Any]:
    """Return a canonical closed value when a global is code configuration."""

    if value is None or isinstance(value, (str, bool, int)):
        return True, value
    if isinstance(value, float):
        return (math.isfinite(value), value)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, child in sorted(
            value.items(), key=lambda row: str(row[0])
        ):
            if not isinstance(raw_key, str):
                return False, None
            accepted, normalized = _closed_code_constant(child)
            if not accepted:
                return False, None
            output[raw_key] = normalized
        return True, output
    if isinstance(value, (tuple, list)):
        output_list: list[Any] = []
        for child in value:
            accepted, normalized = _closed_code_constant(child)
            if not accepted:
                return False, None
            output_list.append(normalized)
        return True, output_list
    if isinstance(value, (set, frozenset)):
        output_set: list[Any] = []
        for child in value:
            accepted, normalized = _closed_code_constant(child)
            if not accepted:
                return False, None
            output_set.append(normalized)
        return True, sorted(output_set, key=_canonical)
    return False, None


def _phase_local_constant_value(
    *,
    name: str,
    value: Any,
    phase: str,
) -> tuple[bool, Any]:
    """Project cross-phase globals to the current compatibility domain."""

    if name == "PORTABLE_CHECKPOINT_PHASE_SPECS":
        value = {phase: value[phase]}
    elif name == "CHECKPOINT_COMPATIBILITY_PHASE_BY_ARTIFACT_KIND":
        value = {
            kind: domain
            for kind, domain in value.items()
            if domain == phase
        }
    elif name == "ADOPTABLE_PHASE_BY_ARTIFACT_KIND":
        value = {
            kind: domain
            for kind, domain in value.items()
            if domain == phase
        }
    elif name == "GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS":
        relevant = {
            kind
            for kind, domain in (
                CHECKPOINT_COMPATIBILITY_PHASE_BY_ARTIFACT_KIND.items()
            )
            if domain == phase
        }
        value = {
            kind: schema
            for kind, schema in value.items()
            if kind in relevant
        }
    elif name in {"PHASES", "STAGE1_ONLY_PHASES"}:
        sequence = tuple(str(child) for child in value)
        value = {
            "phase": phase,
            "included": phase in sequence,
            "ordinal": (
                sequence.index(phase) if phase in sequence else None
            ),
        }
    elif name == "EVIDENCE_FAMILIES" and phase in {
        "input_preparation",
        "embedding_cache",
        "oracle_evaluation",
    }:
        value = {"used_by_phase": False}
    return _closed_code_constant(value)


def _workflow_same_file_dependency_identity(
    *,
    workflow_type: type,
    phase: str,
    include_default_phase_producer: bool,
) -> Mapping[str, Any]:
    """Hash the phase-reachable same-file callable/constant closure."""

    queue: list[tuple[str, Any, ast.Module]] = []
    method_names = list(_SHARED_CHECKPOINT_WORKFLOW_METHODS)
    if include_default_phase_producer:
        method_names.extend(_PHASE_WORKFLOW_METHODS[phase])
    for name in method_names:
        target = getattr(workflow_type, name)
        queue.append(
            (
                f"workflow_method:{name}",
                target,
                _phase_local_callable_tree(target, phase=phase),
            )
        )
    if include_default_phase_producer:
        dispatcher = workflow_type._run_default
        dispatcher_tree = ast.parse(
            textwrap.dedent(inspect.getsource(dispatcher))
        )
        branch_matches: list[ast.If] = []
        for node in ast.walk(dispatcher_tree):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and any(
                    isinstance(candidate, ast.Name)
                    and candidate.id == "phase"
                    for candidate in (test.left, test.comparators[0])
                )
                and any(
                    isinstance(candidate, ast.Constant)
                    and candidate.value == phase
                    for candidate in (test.left, test.comparators[0])
                )
            ):
                branch_matches.append(node)
        if len(branch_matches) != 1:
            raise RuntimeError(
                f"workflow dispatcher must expose one {phase!r} dependency root"
            )
        branch_tree = ast.Module(
            body=copy.deepcopy(branch_matches[0].body),
            type_ignores=[],
        )
        transformed_branch = _PhaseLocalAstPruner(phase).visit(
            branch_tree
        )
        assert isinstance(transformed_branch, ast.Module)
        queue.append(
            (
                "workflow_method:phase_dispatch_branch",
                dispatcher,
                ast.fix_missing_locations(transformed_branch),
            )
        )
    module_globals = vars(sys.modules[__name__])
    for name in (
        *_SHARED_CHECKPOINT_MODULE_CALLABLES,
        *_PHASE_CHECKPOINT_MODULE_CALLABLES[phase],
    ):
        target = module_globals[name]
        queue.append(
            (
                f"module_callable:{name}",
                target,
                _phase_local_callable_tree(target, phase=phase),
            )
        )

    callable_hashes: dict[str, str] = {}
    constant_values: dict[str, Any] = {}
    visited: set[tuple[str, str]] = set()
    while queue:
        label, target, tree = queue.pop()
        module_name = str(
            getattr(target, "__module__", type(target).__module__)
        )
        qualname = str(
            getattr(target, "__qualname__", type(target).__qualname__)
        )
        identity = (module_name, qualname)
        if identity in visited:
            continue
        visited.add(identity)
        logical_name = f"{module_name}:{qualname}"
        callable_hashes[logical_name] = _ast_tree_sha256(
            tree,
            schema_version="phase_local_callable_ast_v1",
        )
        target_globals = getattr(target, "__globals__", {})
        target_source = inspect.getsourcefile(target)
        target_source_path = (
            None
            if target_source is None
            else Path(target_source).resolve()
        )
        referenced_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
        }
        referenced_attributes = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id
            in {
                "self",
                "cls",
                workflow_type.__name__,
                "ProductionAllEvidenceWorkflow",
            }
        }
        for name in sorted(referenced_attributes):
            if not hasattr(workflow_type, name):
                continue
            dependency = getattr(workflow_type, name)
            if inspect.isfunction(dependency) or inspect.ismethod(
                dependency
            ):
                queue.append(
                    (
                        f"workflow_method:{name}",
                        dependency,
                        _phase_local_callable_tree(
                            dependency,
                            phase=phase,
                        ),
                    )
                )
            else:
                accepted, normalized = _phase_local_constant_value(
                    name=f"{workflow_type.__name__}.{name}",
                    value=dependency,
                    phase=phase,
                )
                if accepted:
                    constant_values[
                        f"{workflow_type.__name__}.{name}"
                    ] = normalized
        for name in sorted(referenced_names):
            if name not in target_globals:
                continue
            allowed_domains = _MODULE_CALLABLE_PHASE_DOMAINS.get(
                name
            )
            if (
                allowed_domains is not None
                and phase not in allowed_domains
            ):
                continue
            dependency = target_globals[name]
            if inspect.isfunction(dependency) or inspect.ismethod(
                dependency
            ):
                try:
                    dependency_source = inspect.getsourcefile(
                        dependency
                    )
                except (TypeError, OSError):
                    dependency_source = None
                if (
                    target_source_path is not None
                    and dependency_source is not None
                    and Path(dependency_source).resolve()
                    == target_source_path
                ):
                    queue.append(
                        (
                            f"module_callable:{name}",
                            dependency,
                            _phase_local_callable_tree(
                                dependency,
                                phase=phase,
                            ),
                        )
                    )
                continue
            accepted, normalized = _phase_local_constant_value(
                name=name,
                value=dependency,
                phase=phase,
            )
            if accepted:
                constant_values[f"{module_name}:{name}"] = normalized
    body = {
        "schema_version": (
            "phase_local_same_file_dependency_identity_v1"
        ),
        "phase": phase,
        "callable_ast_sha256": dict(sorted(callable_hashes.items())),
        "referenced_constants": dict(sorted(constant_values.items())),
    }
    return {**body, "content_sha256": _sha(body)}


def _normalized_callable_ast_sha256(value: Any) -> str:
    """Hash executable structure without source paths or formatting."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(value)))
    return _sha(
        {
            "schema_version": "normalized_callable_ast_v1",
            "ast": ast.dump(tree, annotate_fields=True, include_attributes=False),
        }
    )


def _phase_branch_ast_sha256(value: Any, phase: str) -> str:
    """Hash only one explicit phase branch of the monolithic dispatcher."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(value)))
    matches: list[ast.If] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
        ):
            candidates = (test.left, test.comparators[0])
            names = [item for item in candidates if isinstance(item, ast.Name)]
            constants = [
                item.value
                for item in candidates
                if isinstance(item, ast.Constant)
            ]
            if any(item.id == "phase" for item in names) and phase in constants:
                matches.append(node)
    if len(matches) != 1:
        raise RuntimeError(
            f"workflow dispatcher must have exactly one explicit {phase!r} branch"
        )
    return _sha(
        {
            "schema_version": "normalized_phase_branch_ast_v1",
            "phase": phase,
            "body_ast": [
                ast.dump(item, annotate_fields=True, include_attributes=False)
                for item in matches[0].body
            ],
        }
    )


_IdentityStatToken = tuple[int, int, int, int, int, int, int]


def _identity_stat_token(path: Path) -> _IdentityStatToken:
    """Return the content-relevant stat guard for one authenticated path."""

    value = Path(path).stat()
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _parse_local_import_module_names(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[str, ...]:
    """Parse every import name while leaving repository resolution separate."""

    relative = path.relative_to(repository_root)
    package_parts = relative.with_suffix("").parts[:-1]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError(f"cannot parse scientific producer source: {relative}") from exc
    module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_names.update(
                alias.name
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - (int(node.level) - 1)
                if keep < 0:
                    continue
                prefix = package_parts[:keep]
                suffix = tuple(str(node.module).split(".")) if node.module else ()
                resolved_parts = (*prefix, *suffix)
                if resolved_parts:
                    base_module = ".".join(resolved_parts)
                    module_names.add(base_module)
                    module_names.update(
                        f"{base_module}.{alias.name}"
                        for alias in node.names
                        if alias.name != "*"
                    )
            elif node.module:
                base_module = str(node.module)
                module_names.add(base_module)
                module_names.update(
                    f"{base_module}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
    return tuple(sorted(module_names))


@dataclass(frozen=True)
class _ParsedImportCacheEntry:
    repository_root: Path
    source_stat: _IdentityStatToken
    parser: Any
    module_names: tuple[str, ...]


@dataclass(frozen=True)
class _FileDigestCacheEntry:
    repository_root: Path
    source_stat: _IdentityStatToken
    hasher: Any
    digest: str
    size_bytes: int


@dataclass(frozen=True)
class _DirectoryListingCacheEntry:
    repository_root: Path
    directory_stat: _IdentityStatToken
    entries: Mapping[str, str]


class _ScientificIdentityMemo:
    """Process-local authenticated handles guarded by current file metadata.

    Nothing in this cache is serialized or accepted across a fresh process.
    Every hit reopens the relevant stat metadata. Parser/hasher replacement,
    repository relocation, or a content-relevant stat change forces the exact
    uncached operation again.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._parsed_imports: dict[
            tuple[Path, Path], _ParsedImportCacheEntry
        ] = {}
        self._file_digests: dict[
            tuple[Path, Path], _FileDigestCacheEntry
        ] = {}
        self._directory_listings: dict[
            tuple[Path, Path], _DirectoryListingCacheEntry
        ] = {}

    def clear(self) -> None:
        """Drop all process-local authenticated handles."""

        with self._lock:
            self._parsed_imports.clear()
            self._file_digests.clear()
            self._directory_listings.clear()

    @staticmethod
    def _resolved_pair(
        *,
        repository_root: Path,
        path: Path,
    ) -> tuple[Path, Path]:
        root = Path(repository_root).resolve(strict=True)
        resolved = Path(path).resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "scientific identity cache path escaped the repository"
            ) from exc
        return root, resolved

    def parsed_import_module_names(
        self,
        path: Path,
        *,
        repository_root: Path,
    ) -> tuple[str, ...]:
        root, resolved = self._resolved_pair(
            repository_root=repository_root,
            path=path,
        )
        source_stat = _identity_stat_token(resolved)
        parser = _parse_local_import_module_names
        key = (root, resolved)
        with self._lock:
            cached = self._parsed_imports.get(key)
            if (
                cached is not None
                and cached.repository_root == root
                and cached.source_stat == source_stat
                and cached.parser is parser
            ):
                return cached.module_names
        module_names = parser(
            resolved,
            repository_root=root,
        )
        terminal_stat = _identity_stat_token(resolved)
        if terminal_stat != source_stat:
            raise RuntimeError(
                "scientific producer source changed while imports were parsed"
            )
        entry = _ParsedImportCacheEntry(
            repository_root=root,
            source_stat=terminal_stat,
            parser=parser,
            module_names=module_names,
        )
        with self._lock:
            self._parsed_imports[key] = entry
        return module_names

    def file_digest(
        self,
        path: Path,
        *,
        repository_root: Path,
    ) -> tuple[str, int]:
        root, resolved = self._resolved_pair(
            repository_root=repository_root,
            path=path,
        )
        source_stat = _identity_stat_token(resolved)
        hasher = stable_file_sha256
        key = (root, resolved)
        with self._lock:
            cached = self._file_digests.get(key)
            if (
                cached is not None
                and cached.repository_root == root
                and cached.source_stat == source_stat
                and cached.hasher is hasher
            ):
                return cached.digest, cached.size_bytes
        digest, size = hasher(resolved)
        terminal_stat = _identity_stat_token(resolved)
        if terminal_stat != source_stat:
            raise RuntimeError(
                "scientific producer source changed while it was hashed"
            )
        entry = _FileDigestCacheEntry(
            repository_root=root,
            source_stat=terminal_stat,
            hasher=hasher,
            digest=str(digest),
            size_bytes=int(size),
        )
        with self._lock:
            self._file_digests[key] = entry
        return entry.digest, entry.size_bytes

    def directory_entries(
        self,
        directory: Path,
        *,
        repository_root: Path,
    ) -> Mapping[str, str]:
        root, resolved = self._resolved_pair(
            repository_root=repository_root,
            path=directory,
        )
        directory_stat = _identity_stat_token(resolved)
        if not stat.S_ISDIR(directory_stat[2]):
            raise NotADirectoryError(resolved)
        key = (root, resolved)
        with self._lock:
            cached = self._directory_listings.get(key)
            if (
                cached is not None
                and cached.repository_root == root
                and cached.directory_stat == directory_stat
            ):
                return cached.entries
        entries: dict[str, str] = {}
        with os.scandir(resolved) as iterator:
            for child in iterator:
                if child.is_file(follow_symlinks=True):
                    entries[child.name] = "file"
                elif child.is_dir(follow_symlinks=True):
                    entries[child.name] = "directory"
                else:
                    entries[child.name] = "other"
        terminal_stat = _identity_stat_token(resolved)
        if terminal_stat != directory_stat:
            raise RuntimeError(
                "repository directory changed while imports were resolved"
            )
        entry = _DirectoryListingCacheEntry(
            repository_root=root,
            directory_stat=terminal_stat,
            entries=dict(sorted(entries.items())),
        )
        with self._lock:
            self._directory_listings[key] = entry
        return entry.entries


_PROCESS_SCIENTIFIC_IDENTITY_MEMO = _ScientificIdentityMemo()


def _memoized_scientific_file_digest(
    path: Path,
    *,
    identity_memo: _ScientificIdentityMemo,
) -> tuple[str, int]:
    """Reuse one authenticated file inside the current process trust handle."""

    resolved = Path(path).resolve(strict=True)
    repository_root = Path(__file__).resolve().parents[2]
    try:
        resolved.relative_to(repository_root)
        trust_root = repository_root
    except ValueError:
        trust_root = resolved.parent
    return identity_memo.file_digest(
        resolved,
        repository_root=trust_root,
    )


def _resolve_repository_module_path(
    module_name: str,
    *,
    repository_root: Path,
    identity_memo: _ScientificIdentityMemo,
    directory_views: MutableMapping[Path, Mapping[str, str]],
) -> Path | None:
    """Resolve one module using guarded directory views, including misses."""

    parts = tuple(part for part in module_name.split(".") if part)
    if not parts:
        return None

    def entries(directory: Path) -> Mapping[str, str]:
        resolved = Path(directory).resolve(strict=True)
        cached = directory_views.get(resolved)
        if cached is None:
            cached = identity_memo.directory_entries(
                resolved,
                repository_root=repository_root,
            )
            directory_views[resolved] = cached
        return cached

    parent = repository_root
    for component in parts[:-1]:
        current = entries(parent)
        if current.get(component) != "directory":
            return None
        parent = parent / component
    current = entries(parent)
    leaf = parts[-1]
    module_file = f"{leaf}.py"
    if current.get(module_file) == "file":
        return (parent / module_file).resolve(strict=True)
    if current.get(leaf) != "directory":
        return None
    package = parent / leaf
    if entries(package).get("__init__.py") == "file":
        return (package / "__init__.py").resolve(strict=True)
    return None


def _local_import_paths(path: Path, *, repository_root: Path) -> tuple[Path, ...]:
    """Resolve all repository-local imports with stat-guarded parse reuse."""

    root = Path(repository_root).resolve(strict=True)
    resolved_source = Path(path).resolve(strict=True)
    module_names = (
        _PROCESS_SCIENTIFIC_IDENTITY_MEMO.parsed_import_module_names(
            resolved_source,
            repository_root=root,
        )
    )
    directory_views: dict[Path, Mapping[str, str]] = {}
    output: set[Path] = set()
    workflow_source = Path(__file__).resolve()
    for module_name in module_names:
        resolved = _resolve_repository_module_path(
            module_name,
            repository_root=root,
            identity_memo=_PROCESS_SCIENTIFIC_IDENTITY_MEMO,
            directory_views=directory_views,
        )
        if resolved is not None and resolved != workflow_source:
            output.add(resolved)
    return tuple(sorted(output))


def _transitive_local_source_inventory(
    *,
    repository_root: Path,
    roots: Sequence[str],
    import_leaf_paths: frozenset[str] = frozenset(),
    import_cache: MutableMapping[Path, tuple[Path, ...]] | None = None,
    file_identity_cache: MutableMapping[Path, Mapping[str, Any]] | None = None,
    identity_memo: _ScientificIdentityMemo | None = None,
) -> tuple[dict[str, Any], ...]:
    import_cache = {} if import_cache is None else import_cache
    file_identity_cache = (
        {} if file_identity_cache is None else file_identity_cache
    )
    pending = [(repository_root / relative).resolve(strict=True) for relative in roots]
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        try:
            path.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError("producer source escaped the repository") from exc
        observed.add(path)
        relative = path.relative_to(repository_root).as_posix()
        if relative in import_leaf_paths:
            imports = ()
        else:
            imports = import_cache.get(path)
            if imports is None:
                imports = _local_import_paths(
                    path,
                    repository_root=repository_root,
                )
                import_cache[path] = imports
        pending.extend(
            candidate
            for candidate in imports
            if candidate not in observed
        )
    rows: list[dict[str, Any]] = []
    for path in sorted(observed):
        row = file_identity_cache.get(path)
        if row is None:
            digest, size = (
                stable_file_sha256(path)
                if identity_memo is None
                else identity_memo.file_digest(
                    path,
                    repository_root=repository_root,
                )
            )
            row = {
                "relative_path": path.relative_to(repository_root).as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
            file_identity_cache[path] = row
        rows.append(copy.deepcopy(dict(row)))
    return tuple(rows)


def _repository_local_callable_import_closure(
    target: Any,
    *,
    repository_root: Path | None = None,
    identity_memo: _ScientificIdentityMemo | None = None,
) -> tuple[dict[str, Any], ...]:
    """Bind an injected callable's repository-local transitive imports."""

    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve(strict=True)
    )
    try:
        source = inspect.getsourcefile(target)
    except (TypeError, OSError):
        source = None
    if source is None:
        raise ValueError(
            "injected workflow callables must have inspectable source"
        )
    resolved = Path(source).resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            "injected workflow callable source must be inside the authenticated "
            "repository; external callable sources are not adoptable"
        ) from exc
    closure = _transitive_local_source_inventory(
        repository_root=root,
        roots=(relative,),
        identity_memo=(
            _PROCESS_SCIENTIFIC_IDENTITY_MEMO
            if identity_memo is None
            else identity_memo
        ),
    )
    if not closure:
        raise RuntimeError(
            "injected workflow callable dependency inventory is empty"
        )
    return closure


_EXPLICIT_CALLABLE_SCIENTIFIC_IDENTITY = (
    "__portable_workflow_scientific_identity__"
)


def _closed_explicit_callable_identity(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    """Validate one caller-authored, path-neutral scientific identity."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one closed mapping")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values") from exc
    if not isinstance(normalized, dict) or any(
        not isinstance(key, str) or not key
        for key in normalized
    ):
        raise ValueError(f"{label} keys must be nonempty strings")

    def reject_absolute_locator(item: Any) -> None:
        if isinstance(item, Mapping):
            for child in item.values():
                reject_absolute_locator(child)
            return
        if isinstance(item, list):
            for child in item:
                reject_absolute_locator(child)
            return
        if isinstance(item, str) and Path(item).is_absolute():
            raise ValueError(
                f"{label} cannot contain an absolute deployment locator"
            )

    reject_absolute_locator(normalized)
    return normalized


def _callable_explicit_scientific_identity(
    value: Any,
) -> Mapping[str, Any] | None:
    try:
        supplied = getattr(
            value,
            _EXPLICIT_CALLABLE_SCIENTIFIC_IDENTITY,
        )
    except AttributeError:
        return None
    if callable(supplied):
        raise TypeError(
            "explicit callable scientific identity must be data, not "
            "another executable provider"
        )
    return _closed_explicit_callable_identity(
        supplied,
        label="explicit callable scientific identity",
    )


def _closed_callable_state(
    value: Any,
    *,
    label: str,
    active_callables: set[int],
    active_values: set[int],
    identity_memo: _ScientificIdentityMemo,
) -> Any:
    """Encode behavior-affecting callable state without lossy string casts."""

    if value is None or type(value) in {str, bool, int}:  # noqa: E721
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return value
    if type(value) is bytes:
        return {
            "state_type": "bytes",
            "hex": value.hex(),
        }
    if isinstance(value, Path):
        raise ValueError(
            f"{label} closes over a deployment path; supply an explicit "
            "path-neutral scientific identity"
        )
    if callable(value):
        return {
            "state_type": "callable",
            "identity": _callable_behavior_identity(
                value,
                active_callables=active_callables,
                identity_memo=identity_memo,
            ),
        }

    tracked = isinstance(
        value,
        (Mapping, list, tuple, set, frozenset),
    )
    marker = id(value)
    if tracked:
        if marker in active_values:
            raise ValueError(f"{label} contains a recursive state value")
        active_values.add(marker)
    try:
        if isinstance(value, Mapping):
            if any(
                not isinstance(key, str) or not key
                for key in value
            ):
                raise ValueError(
                    f"{label} mapping keys must be nonempty strings"
                )
            return {
                "state_type": "mapping",
                "items": [
                    [
                        key,
                        _closed_callable_state(
                            value[key],
                            label=f"{label}.{key}",
                            active_callables=active_callables,
                            active_values=active_values,
                            identity_memo=identity_memo,
                        ),
                    ]
                    for key in sorted(value)
                ],
            }
        if isinstance(value, (list, tuple)):
            return {
                "state_type": (
                    "tuple" if isinstance(value, tuple) else "list"
                ),
                "items": [
                    _closed_callable_state(
                        child,
                        label=f"{label}[{index}]",
                        active_callables=active_callables,
                        active_values=active_values,
                        identity_memo=identity_memo,
                    )
                    for index, child in enumerate(value)
                ],
            }
        if isinstance(value, (set, frozenset)):
            children = [
                _closed_callable_state(
                    child,
                    label=f"{label} member",
                    active_callables=active_callables,
                    active_values=active_values,
                    identity_memo=identity_memo,
                )
                for child in value
            ]
            return {
                "state_type": (
                    "frozenset"
                    if isinstance(value, frozenset)
                    else "set"
                ),
                "items": sorted(children, key=_canonical),
            }
    finally:
        if tracked:
            active_values.remove(marker)
    raise TypeError(
        f"{label} contains unclosed state of type "
        f"{type(value).__module__}.{type(value).__qualname__}; supply an "
        "explicit closed scientific identity"
    )


def _function_behavior_state(
    function: Any,
    *,
    active_callables: set[int],
    identity_memo: _ScientificIdentityMemo,
) -> Mapping[str, Any]:
    closure = function.__closure__ or ()
    freevars = tuple(function.__code__.co_freevars)
    if len(closure) != len(freevars):
        raise RuntimeError("callable closure metadata is inconsistent")
    active_values: set[int] = set()
    attributes = {
        key: child
        for key, child in vars(function).items()
        if key != _EXPLICIT_CALLABLE_SCIENTIFIC_IDENTITY
    }
    return {
        "state_policy": "closed_function_state_v1",
        "defaults": _closed_callable_state(
            function.__defaults__,
            label="callable positional defaults",
            active_callables=active_callables,
            active_values=active_values,
            identity_memo=identity_memo,
        ),
        "keyword_defaults": _closed_callable_state(
            function.__kwdefaults__,
            label="callable keyword defaults",
            active_callables=active_callables,
            active_values=active_values,
            identity_memo=identity_memo,
        ),
        "closure_cells": [
            {
                "name": name,
                "value": _closed_callable_state(
                    cell.cell_contents,
                    label=f"callable closure cell {name!r}",
                    active_callables=active_callables,
                    active_values=active_values,
                    identity_memo=identity_memo,
                ),
            }
            for name, cell in zip(freevars, closure, strict=True)
        ],
        "function_attributes": _closed_callable_state(
            attributes,
            label="callable function attributes",
            active_callables=active_callables,
            active_values=active_values,
            identity_memo=identity_memo,
        ),
    }


def _callable_instance_state(
    value: Any,
    *,
    active_callables: set[int],
    identity_memo: _ScientificIdentityMemo,
) -> Mapping[str, Any]:
    state: dict[str, Any] = {}
    try:
        state.update(vars(value))
    except TypeError:
        pass
    for owner in type(value).__mro__:
        raw_slots = owner.__dict__.get("__slots__", ())
        slots = (
            (raw_slots,)
            if isinstance(raw_slots, str)
            else tuple(raw_slots)
        )
        for name in slots:
            if name in {"__dict__", "__weakref__"} or name in state:
                continue
            if hasattr(value, name):
                state[name] = getattr(value, name)
    state.pop(_EXPLICIT_CALLABLE_SCIENTIFIC_IDENTITY, None)
    return {
        "state_policy": "closed_callable_instance_state_v1",
        "instance_state": _closed_callable_state(
            state,
            label="callable instance state",
            active_callables=active_callables,
            active_values=set(),
            identity_memo=identity_memo,
        ),
    }


def _callable_source_identity(
    target: Any,
    *,
    identity_memo: _ScientificIdentityMemo,
) -> Mapping[str, Any]:
    try:
        source = inspect.getsourcefile(target)
    except (TypeError, OSError):
        source = None
    if source is None:
        raise ValueError(
            "injected workflow callable must have inspectable repository "
            "source"
        )
    resolved = Path(source).resolve(strict=True)
    repository_root = Path(__file__).resolve().parents[2]
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(
            "injected workflow callable source must be inside the "
            "authenticated repository; external callable sources are not "
            "adoptable"
        ) from exc
    digest, size = identity_memo.file_digest(
        resolved,
        repository_root=repository_root,
    )
    closure = _repository_local_callable_import_closure(
        target,
        identity_memo=identity_memo,
    )
    return {
        "source_file": {
            "path": str(resolved),
            "sha256": digest,
            "size_bytes": size,
        },
        "repository_local_import_closure": list(closure),
    }


def _callable_behavior_identity(
    value: Any,
    *,
    explicit_scientific_identity: Mapping[str, Any] | None = None,
    active_callables: set[int] | None = None,
    identity_memo: _ScientificIdentityMemo | None = None,
) -> Mapping[str, Any]:
    """Bind source, dependency bytes, and every closed callable state axis."""

    if not callable(value):
        raise TypeError("injected workflow capability is not callable")
    memo = (
        _PROCESS_SCIENTIFIC_IDENTITY_MEMO
        if identity_memo is None
        else identity_memo
    )
    active = set() if active_callables is None else active_callables
    marker = id(value)
    if marker in active:
        raise ValueError("injected workflow callable state is recursive")
    active.add(marker)
    try:
        supplied = (
            _callable_explicit_scientific_identity(value)
            if explicit_scientific_identity is None
            else _closed_explicit_callable_identity(
                explicit_scientific_identity,
                label="injected callable scientific identity",
            )
        )
        if isinstance(value, functools.partial):
            target = value.func
            source_identity = _callable_source_identity(
                target,
                identity_memo=memo,
            )
            if supplied is None:
                state = {
                    "state_policy": "closed_partial_state_v1",
                    "wrapped_callable": _callable_behavior_identity(
                        target,
                        active_callables=active,
                        identity_memo=memo,
                    ),
                    "positional_arguments": _closed_callable_state(
                        value.args,
                        label="partial positional arguments",
                        active_callables=active,
                        active_values=set(),
                        identity_memo=memo,
                    ),
                    "keyword_arguments": _closed_callable_state(
                        value.keywords or {},
                        label="partial keyword arguments",
                        active_callables=active,
                        active_values=set(),
                        identity_memo=memo,
                    ),
                    "partial_attributes": _closed_callable_state(
                        vars(value),
                        label="partial attributes",
                        active_callables=active,
                        active_values=set(),
                        identity_memo=memo,
                    ),
                }
            else:
                state = {
                    "state_policy": (
                        "explicit_closed_scientific_identity_v1"
                    ),
                    "scientific_identity": supplied,
                }
            callable_kind = "functools.partial"
            module = str(
                getattr(target, "__module__", type(target).__module__)
            )
            qualname = str(
                getattr(target, "__qualname__", type(target).__qualname__)
            )
        elif inspect.ismethod(value):
            target = value.__func__
            source_identity = _callable_source_identity(
                target,
                identity_memo=memo,
            )
            if supplied is None:
                bound = value.__self__
                if inspect.isclass(bound):
                    raise TypeError(
                        "class-bound injected methods require an explicit "
                        "closed scientific identity"
                    )
                state = {
                    "state_policy": "closed_bound_method_state_v1",
                    "function_state": _function_behavior_state(
                        target,
                        active_callables=active,
                        identity_memo=memo,
                    ),
                    "bound_instance_state": _callable_instance_state(
                        bound,
                        active_callables=active,
                        identity_memo=memo,
                    ),
                }
            else:
                state = {
                    "state_policy": (
                        "explicit_closed_scientific_identity_v1"
                    ),
                    "scientific_identity": supplied,
                }
            callable_kind = "bound_method"
            module = str(target.__module__)
            qualname = str(target.__qualname__)
        elif inspect.isfunction(value):
            target = value
            source_identity = _callable_source_identity(
                target,
                identity_memo=memo,
            )
            state = (
                _function_behavior_state(
                    target,
                    active_callables=active,
                    identity_memo=memo,
                )
                if supplied is None
                else {
                    "state_policy": (
                        "explicit_closed_scientific_identity_v1"
                    ),
                    "scientific_identity": supplied,
                }
            )
            callable_kind = "function"
            module = str(target.__module__)
            qualname = str(target.__qualname__)
        else:
            target = getattr(type(value), "__call__", None)
            if target is None or not callable(target):
                raise TypeError(
                    "injected callable instance has no inspectable __call__"
                )
            source_identity = _callable_source_identity(
                target,
                identity_memo=memo,
            )
            state = (
                _callable_instance_state(
                    value,
                    active_callables=active,
                    identity_memo=memo,
                )
                if supplied is None
                else {
                    "state_policy": (
                        "explicit_closed_scientific_identity_v1"
                    ),
                    "scientific_identity": supplied,
                }
            )
            callable_kind = "callable_instance"
            module = str(type(value).__module__)
            qualname = str(type(value).__qualname__)
        body = {
            "schema_version": "closed_callable_behavior_identity_v1",
            "callable_kind": callable_kind,
            "module": module,
            "qualname": qualname,
            **source_identity,
            "behavior_state": state,
        }
        neutral_body = _path_neutral_injected_identity(body)
        return {
            **body,
            "content_sha256": identity_sha256(neutral_body),
        }
    finally:
        active.remove(marker)


def _repository_import_closure_rows(
    value: Any,
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        closure = value.get("repository_local_import_closure")
        if isinstance(closure, list):
            rows.extend(
                row for row in closure if isinstance(row, Mapping)
            )
        for child in value.values():
            rows.extend(_repository_import_closure_rows(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_repository_import_closure_rows(child))
    return tuple(rows)


def _path_neutral_injected_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, child in sorted(value.items()):
            if key == "source_file" and isinstance(child, Mapping):
                output[str(key)] = {
                    str(source_key): _path_neutral_injected_identity(
                        source_value
                    )
                    for source_key, source_value in sorted(
                        child.items()
                    )
                    if source_key != "path"
                }
            else:
                output[str(key)] = _path_neutral_injected_identity(
                    child
                )
        return output
    if isinstance(value, list):
        return [_path_neutral_injected_identity(child) for child in value]
    return value


def _stage1_preflight_integration_identity(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Project the all-ten integration to what preflight actually consumes.

    The default preflight stores only the producer-factory architecture
    profiles and runtime compatibility class in the prepared-context locator.
    It does not invoke a component factory, physical-owner executor, or Stage 2
    handoff publisher.  When the producer builder supplies the existing
    explicit closed scientific identity, bind that configuration and its
    stable callable interface without binding later modeling implementation
    bytes.  Custom integrations without that explicit identity retain the full
    source-bound identity as a fail-closed fallback.
    """

    if value is None:
        return None
    builder = value.get("producer_factories_builder")
    if not isinstance(builder, Mapping):
        return copy.deepcopy(dict(value))
    behavior = builder.get("behavior_state")
    if (
        not isinstance(behavior, Mapping)
        or behavior.get("state_policy")
        != "explicit_closed_scientific_identity_v1"
        or not isinstance(behavior.get("scientific_identity"), Mapping)
    ):
        return copy.deepcopy(dict(value))
    interface = {
        key: builder.get(key)
        for key in ("callable_kind", "module", "qualname")
    }
    if not all(
        isinstance(child, str) and child
        for child in interface.values()
    ):
        return copy.deepcopy(dict(value))
    body = {
        "schema_version": (
            "production_stage1_preflight_integration_identity_v1"
        ),
        "producer_factories_builder_interface": interface,
        "producer_factories_scientific_identity": copy.deepcopy(
            dict(behavior["scientific_identity"])
        ),
        "component_factories_invoked_during_preflight": False,
        "physical_owner_executor_invoked_during_preflight": False,
        "stage2_handoff_publisher_invoked_during_preflight": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _phase_transitive_producer_code_records(
    *,
    workflow_type: type,
    integration_hooks: Mapping[str, Any],
    phase_overrides: Mapping[str, Any],
    identity_memo: _ScientificIdentityMemo | None = None,
) -> dict[str, dict[str, Any]]:
    repository_root = Path(__file__).resolve().parents[2]
    memo = (
        _PROCESS_SCIENTIFIC_IDENTITY_MEMO
        if identity_memo is None
        else identity_memo
    )
    output: dict[str, dict[str, Any]] = {}
    dependency_lock_inventory: list[dict[str, Any]] = []
    for relative in _SHARED_DEPENDENCY_LOCK_FILES:
        path = (repository_root / relative).resolve(strict=True)
        digest, size = memo.file_digest(
            path,
            repository_root=repository_root,
        )
        dependency_lock_inventory.append(
            {
                "relative_path": relative,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    for phase in PORTABLE_CHECKPOINT_PHASE_SPECS:
        override = phase_overrides.get(phase)
        hook = (
            integration_hooks.get(phase)
            if phase in {"embedding_cache", "stage1_preflight", "stage1_modeling"}
            else None
        )
        injected = override if override is not None else hook
        roots = list(_SHARED_CHECKPOINT_PRODUCER_ROOTS)
        if injected is None:
            roots.extend(_PHASE_PRODUCER_ROOTS[phase])
        roots = list(dict.fromkeys(roots))
        file_inventory = _transitive_local_source_inventory(
            repository_root=repository_root,
            roots=roots,
            import_leaf_paths=_PHASE_TRANSITIVE_IMPORT_LEAVES.get(
                phase,
                frozenset(),
            ),
            identity_memo=memo,
        )
        same_file_dependencies = (
            _workflow_same_file_dependency_identity(
                workflow_type=workflow_type,
                phase=phase,
                include_default_phase_producer=(
                    injected is None
                ),
            )
        )
        callable_identities = dict(
            same_file_dependencies["callable_ast_sha256"]
        )
        role_neutral = None
        if phase == "stage1_preflight":
            role_neutral = _stage1_preflight_integration_identity(
                integration_hooks.get("role_neutral_stage1")
            )
        elif phase == "stage1_modeling":
            role_neutral = _path_neutral_injected_identity(
                integration_hooks.get("role_neutral_stage1")
            )
        phase_constant_body = {
            "schema_version": (
                "phase_workflow_constant_identity_v1"
            ),
            "phase": phase,
            "workflow_schema": WORKFLOW_SCHEMA,
            "workflow_phase_manifest_schema": (
                WORKFLOW_PHASE_MANIFEST_SCHEMA
            ),
            "portable_checkpoint_phase_spec": copy.deepcopy(
                dict(PORTABLE_CHECKPOINT_PHASE_SPECS[phase])
            ),
            "artifact_kind_compatibility_domains": {
                kind: domain
                for kind, domain in sorted(
                    CHECKPOINT_COMPATIBILITY_PHASE_BY_ARTIFACT_KIND.items()
                )
                if domain == phase
            },
            "granular_artifact_schemas": {
                kind: GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS[kind]
                for kind, domain in sorted(
                    CHECKPOINT_COMPATIBILITY_PHASE_BY_ARTIFACT_KIND.items()
                )
                if domain == phase
                and kind in GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS
            },
            "granular_index_schema": (
                WORKFLOW_GRANULAR_CHECKPOINT_INDEX_SCHEMA
            ),
            "granular_node_schema": (
                WORKFLOW_GRANULAR_CHECKPOINT_NODE_SCHEMA
            ),
            "adoptable_artifact_kinds": {
                kind: domain
                for kind, domain in sorted(
                    ADOPTABLE_PHASE_BY_ARTIFACT_KIND.items()
                )
                if domain == phase
            },
            "required_adopted_ancestor_kind": (
                _REQUIRED_ADOPTED_ANCESTOR_KIND.get(
                    str(
                        PORTABLE_CHECKPOINT_PHASE_SPECS[phase][
                            "artifact_kind"
                        ]
                    )
                )
            ),
            "phase_sequence_position": PHASES.index(phase),
            "included_in_stage1_only_workflow": (
                phase in STAGE1_ONLY_PHASES
            ),
            "evidence_family_order": (
                list(EVIDENCE_FAMILIES)
                if phase
                in {
                    "stage1_preflight",
                    "stage1_modeling",
                    "stage2_canary",
                    "stage2_inference",
                }
                else None
            ),
        }
        phase_constant_identity = {
            **phase_constant_body,
            "content_sha256": _sha(phase_constant_body),
        }
        body = {
            "schema_version": "phase_transitive_producer_code_identity_v1",
            "phase": phase,
            "root_modules": roots,
            "transitive_source_inventory": list(file_inventory),
            "dependency_lock_inventory": copy.deepcopy(
                dependency_lock_inventory
            ),
            "workflow_callable_ast_sha256": callable_identities,
            "workflow_same_file_dependency_identity": (
                same_file_dependencies
            ),
            "workflow_constant_identity": phase_constant_identity,
            "injected_phase_producer": _path_neutral_injected_identity(injected),
            "role_neutral_stage1_integration": role_neutral,
        }
        output[phase] = {**body, "content_sha256": _sha(body)}
    return output


def _bind_workflow_scientific_identity(
    *,
    scientific_configuration_body: Mapping[str, Any],
    phase_code_records: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Bind global identity to all code while retaining phase-local reuse."""

    if set(phase_code_records) != set(
        PORTABLE_CHECKPOINT_PHASE_SPECS
    ):
        raise ValueError(
            "phase producer identities do not cover every checkpoint phase"
        )
    phase_ids: dict[str, str] = {}
    for phase, record in phase_code_records.items():
        body = {
            key: copy.deepcopy(value)
            for key, value in record.items()
            if key != "content_sha256"
        }
        if (
            record.get("phase") != phase
            or record.get("content_sha256") != _sha(body)
        ):
            raise ValueError(
                f"{phase} producer-code record is invalid"
            )
        phase_ids[phase] = str(record["content_sha256"])
    scientific_configuration_sha256 = identity_sha256(
        scientific_configuration_body
    )
    workflow_producer_code_identity = identity_sha256(
        {
            "schema_version": (
                "workflow_phase_producer_code_aggregate_v1"
            ),
            "phase_producer_code_identities": phase_ids,
        }
    )
    scientific_body = {
        "schema_version": (
            "portable_all_evidence_scientific_identity_v3"
        ),
        "scientific_configuration_sha256": (
            scientific_configuration_sha256
        ),
        "workflow_producer_code_identity": (
            workflow_producer_code_identity
        ),
        "phase_producer_code_identities": phase_ids,
    }
    return {
        "scientific_configuration_identity": {
            **copy.deepcopy(dict(scientific_configuration_body)),
            "scientific_configuration_sha256": (
                scientific_configuration_sha256
            ),
        },
        "phase_producer_code_identities": phase_ids,
        "workflow_producer_code_identity": (
            workflow_producer_code_identity
        ),
        "scientific_identity": {
            **scientific_body,
            "scientific_sha256": identity_sha256(scientific_body),
        },
    }


def _path_neutral_identity(value: Any) -> Any:
    """Remove locator/execution fields from a content identity recursively."""

    locator_keys = {
        "path",
        "root",
        "absolute_path",
        "manifest_path",
        "cache_path",
        "hostname",
        "pid",
        "gpu_id",
        "gpu_ids",
        "device",
        "devices",
        "worker_count",
        "workers",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _path_neutral_identity(child)
            for key, child in value.items()
            if str(key) not in locator_keys
        }
    if isinstance(value, (list, tuple)):
        return [_path_neutral_identity(child) for child in value]
    return value


def _reusable_preflight_cache_selector(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Select only scientific cache fields, excluding publication paths."""

    # A relocated/adopted cache wraps the original cache-build identity in a
    # path-bearing relocation record.  The completed phase has already
    # authenticated that wrapper; preflight compatibility is intentionally
    # keyed by the same nested cache science as a fresh build.
    nested = value.get("cache_build_identity")
    selected = (
        nested
        if isinstance(nested, Mapping)
        else value
    )
    required = (
        "schema_version",
        "dataset_sha256",
        "ordered_text_sha256",
        "sentence_model_name",
        "local_model_tree_sha256",
        "chunk_configuration_sha256",
        "cache_configuration_sha256",
        "row_count",
        "chunk_count",
        "hidden_size",
        "cache_files",
        "provider_identity",
    )
    if any(name not in selected for name in required):
        raise ValueError(
            "embedding-cache phase identity lacks reusable-preflight science"
        )
    return {
        name: _path_neutral_identity(
            copy.deepcopy(selected[name])
        )
        for name in required
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace one JSON control file in its existing parent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create one fsync'ed JSON attestation, or verify its exact prior bytes."""

    payload = (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        state = os.lstat(path)
        if (
            stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or int(state.st_nlink) != 1
            or path.read_bytes() != payload
        ):
            raise RuntimeError(f"immutable JSON attestation conflicts: {path}")
        return
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _checkpoint_publication_attestation_value(
    *,
    producer_request_sha256: str,
    phase: str,
    phase_manifest_path: Path,
    phase_manifest: Mapping[str, Any],
    artifact: ValidatedPortableArtifact,
) -> dict[str, Any]:
    phase_manifest_sha256, phase_manifest_size = stable_file_sha256(
        phase_manifest_path.resolve(strict=True)
    )
    body = {
        "schema_version": WORKFLOW_CHECKPOINT_PUBLICATION_ATTESTATION_SCHEMA,
        "status": "complete",
        "producer_request_sha256": producer_request_sha256,
        "phase": phase,
        "phase_manifest_content_sha256": phase_manifest["content_sha256"],
        "phase_manifest_sha256": phase_manifest_sha256,
        "phase_manifest_size_bytes": phase_manifest_size,
        "artifact_id": artifact.artifact_id,
        "artifact_kind": artifact.manifest["artifact_kind"],
        "compatibility_key": artifact.compatibility_key,
        "upstream_artifact_ids": list(artifact.manifest["upstream_artifact_ids"]),
        # Physical locators are operational evidence and deliberately live
        # outside the path-neutral scientific artifact manifest.
        "artifact_control_root": str(artifact.root),
        "payload_root": str(artifact.payload_root),
    }
    return {**body, "content_sha256": _sha(body)}


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Read an immutable control through the same strict private-file gate."""

    return _read_private_json_object(Path(path), label=label)


def _read_private_json_object(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    """Read one private JSON file through a stable no-follow descriptor."""

    target = Path(path)
    before_path = os.lstat(target)
    if (
        stat.S_ISLNK(before_path.st_mode)
        or not stat.S_ISREG(before_path.st_mode)
        or int(before_path.st_nlink) != 1
    ):
        raise ValueError(
            f"{label} must be a private non-symlink regular file"
        )
    descriptor = os.open(
        target,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    chunks: list[bytes] = []
    try:
        before_fd = os.fstat(descriptor)
        before_identity = (
            int(before_path.st_dev),
            int(before_path.st_ino),
            int(before_path.st_mode),
            int(before_path.st_nlink),
            int(before_path.st_size),
            int(before_path.st_mtime_ns),
            int(before_path.st_ctime_ns),
        )
        fd_identity = (
            int(before_fd.st_dev),
            int(before_fd.st_ino),
            int(before_fd.st_mode),
            int(before_fd.st_nlink),
            int(before_fd.st_size),
            int(before_fd.st_mtime_ns),
            int(before_fd.st_ctime_ns),
        )
        if fd_identity != before_identity:
            raise RuntimeError(f"{label} changed while being opened")
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(target)
    after_fd_identity = (
        int(after_fd.st_dev),
        int(after_fd.st_ino),
        int(after_fd.st_mode),
        int(after_fd.st_nlink),
        int(after_fd.st_size),
        int(after_fd.st_mtime_ns),
        int(after_fd.st_ctime_ns),
    )
    after_path_identity = (
        int(after_path.st_dev),
        int(after_path.st_ino),
        int(after_path.st_mode),
        int(after_path.st_nlink),
        int(after_path.st_size),
        int(after_path.st_mtime_ns),
        int(after_path.st_ctime_ns),
    )
    if (
        after_fd_identity != before_identity
        or after_path_identity != before_identity
    ):
        raise RuntimeError(f"{label} changed while being read")

    def reject_duplicates(
        pairs: Sequence[tuple[str, Any]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(
                    f"{label} contains duplicate JSON key: {key}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(
                    f"{label} contains non-finite value {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _granular_checkpoint_index_paths(
    *,
    work_root: Path,
    phase: str,
) -> tuple[Path, Path, Path]:
    root = work_root / "portable_granular_checkpoints" / phase
    return (
        root,
        root / "granular_index.json",
        root / "granular_index_locator.json",
    )


def _derive_expected_granular_checkpoint_plan(
    *,
    outer_folds: int,
    initial_training_partitions: int,
    review_rounds: int,
) -> Mapping[str, Any]:
    """Derive exact logical/physical/fold coverage from the fold contract."""

    for label, raw in (
        ("outer_folds", outer_folds),
        ("initial_training_partitions", initial_training_partitions),
        ("review_rounds", review_rounds),
    ):
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int)
            or int(raw) < 1
        ):
            raise ValueError(
                f"granular checkpoint plan requires positive {label}"
            )
    inner_partitions = int(initial_training_partitions) + int(review_rounds)
    # Derive equivalence from ordered abstract partition content.  These tokens
    # describe the fold contract only; the terminal validators separately bind
    # every resulting owner to the exact row-level PhysicalFitKey reconstructed
    # from the authenticated prepared Stage 1 context.
    logical_rows: list[tuple[str, tuple[str, ...]]] = []
    for outer_fold in range(1, int(outer_folds) + 1):
        full = f"outer_{outer_fold:03d}_full"
        partitions = tuple(
            f"outer_{outer_fold:03d}:partition_{partition:03d}"
            for partition in range(1, inner_partitions + 1)
        )
        logical_rows.append((full, partitions))
        for inner_fold in range(1, inner_partitions + 1):
            scope = (
                f"outer_{outer_fold:03d}_inner_{inner_fold:03d}"
            )
            logical_rows.append(
                (
                    scope,
                    tuple(
                        partition
                        for partition_index, partition in enumerate(
                            partitions,
                            start=1,
                        )
                        if partition_index != inner_fold
                    ),
                )
            )
    for outer_fold in range(1, int(outer_folds) + 1):
        partitions = tuple(
            f"outer_{outer_fold:03d}:partition_{partition:03d}"
            for partition in range(1, inner_partitions + 1)
        )
        for epoch in range(int(review_rounds)):
            scope = (
                f"outer_{outer_fold:03d}_hierarchy_epoch_{epoch:03d}"
            )
            spent_partition_count = (
                int(initial_training_partitions) + epoch
            )
            logical_rows.append(
                (scope, partitions[:spent_partition_count])
            )
    logical_scope_ids = [scope_id for scope_id, _rows in logical_rows]
    physical_owner_scope_ids: list[str] = []
    logical_to_physical_owner: dict[str, str] = {}
    owner_by_ordered_content: dict[tuple[str, ...], str] = {}
    for scope_id, ordered_content in logical_rows:
        owner = owner_by_ordered_content.get(ordered_content)
        if owner is None:
            owner = scope_id
            owner_by_ordered_content[ordered_content] = owner
            physical_owner_scope_ids.append(owner)
        logical_to_physical_owner[scope_id] = owner
    physical_count = len(physical_owner_scope_ids)
    logical_count = len(logical_scope_ids)
    fold_ids = list(range(1, int(outer_folds) + 1))
    body = {
        "schema_version": WORKFLOW_EXPECTED_GRANULAR_PLAN_SCHEMA,
        "outer_fold_count": int(outer_folds),
        "initial_training_partitions": int(initial_training_partitions),
        "review_rounds": int(review_rounds),
        "inner_partition_count": inner_partitions,
        "outer_fold_ids": fold_ids,
        "stage1_physical_owner_scope_ids": (
            physical_owner_scope_ids
        ),
        "stage1_logical_scope_ids": logical_scope_ids,
        "stage1_logical_to_physical_owner": (
            logical_to_physical_owner
        ),
        "stage1_physical_fit_count": physical_count,
        "stage1_logical_scope_count": logical_count,
        "stage1_artifact_kind_counts": {
            "logical_scope_bindings": logical_count,
            "neural_query_component": physical_count,
            "physical_scope_fit": physical_count,
            "row_map": 1,
            "tfidf_component": physical_count,
        },
        "stage2_fold_ids": fold_ids,
        "stage2_review_fold_ids": fold_ids,
        "stage2_artifact_kind_counts": {
            "stage2_extraction_component": 1,
            "stage2_fold": len(fold_ids),
            "stage2_response_component": 1,
            "stage2_review_component": len(fold_ids),
        },
    }
    return {**body, "content_sha256": _sha(body)}


def _validate_expected_granular_checkpoint_plan(
    value: Any,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "workflow request lacks its expected granular checkpoint plan"
        )
    required = {
        "schema_version",
        "outer_fold_count",
        "initial_training_partitions",
        "review_rounds",
        "inner_partition_count",
        "outer_fold_ids",
        "stage1_physical_owner_scope_ids",
        "stage1_logical_scope_ids",
        "stage1_logical_to_physical_owner",
        "stage1_physical_fit_count",
        "stage1_logical_scope_count",
        "stage1_artifact_kind_counts",
        "stage2_fold_ids",
        "stage2_review_fold_ids",
        "stage2_artifact_kind_counts",
        "content_sha256",
    }
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    physical = value.get("stage1_physical_owner_scope_ids")
    logical = value.get("stage1_logical_scope_ids")
    logical_to_owner = value.get(
        "stage1_logical_to_physical_owner"
    )
    folds = value.get("outer_fold_ids")
    stage2_folds = value.get("stage2_fold_ids")
    review_folds = value.get("stage2_review_fold_ids")
    if (
        set(value) != required
        or value.get("schema_version")
        != WORKFLOW_EXPECTED_GRANULAR_PLAN_SCHEMA
        or value.get("content_sha256") != _sha(body)
        or not isinstance(folds, list)
        or not folds
        or any(
            isinstance(fold, bool) or not isinstance(fold, int)
            for fold in folds
        )
        or isinstance(value.get("outer_fold_count"), bool)
        or not isinstance(value.get("outer_fold_count"), int)
        or int(value["outer_fold_count"]) < 1
        or value.get("outer_fold_count") != len(folds)
        or isinstance(value.get("initial_training_partitions"), bool)
        or not isinstance(value.get("initial_training_partitions"), int)
        or int(value["initial_training_partitions"]) < 1
        or isinstance(value.get("review_rounds"), bool)
        or not isinstance(value.get("review_rounds"), int)
        or int(value["review_rounds"]) < 1
        or isinstance(value.get("inner_partition_count"), bool)
        or not isinstance(value.get("inner_partition_count"), int)
        or value.get("inner_partition_count")
        != int(value["initial_training_partitions"])
        + int(value["review_rounds"])
        or not isinstance(physical, list)
        or not physical
        or len(physical) != len(set(physical))
        or not isinstance(logical, list)
        or not logical
        or len(logical) != len(set(logical))
        or not isinstance(logical_to_owner, Mapping)
        or set(logical_to_owner) != set(logical)
        or any(owner not in set(physical) for owner in logical_to_owner.values())
        or value.get("stage1_physical_fit_count") != len(physical)
        or value.get("stage1_logical_scope_count") != len(logical)
        or folds != list(range(1, len(folds) + 1))
        or stage2_folds != folds
        or review_folds != folds
    ):
        raise ValueError(
            "expected granular checkpoint plan is invalid"
        )
    expected_stage1_counts = {
        "logical_scope_bindings": len(logical),
        "neural_query_component": len(physical),
        "physical_scope_fit": len(physical),
        "row_map": 1,
        "tfidf_component": len(physical),
    }
    expected_stage2_counts = {
        "stage2_extraction_component": 1,
        "stage2_fold": len(folds),
        "stage2_response_component": 1,
        "stage2_review_component": len(folds),
    }
    if (
        value.get("stage1_artifact_kind_counts")
        != expected_stage1_counts
        or value.get("stage2_artifact_kind_counts")
        != expected_stage2_counts
    ):
        raise ValueError(
            "expected granular checkpoint kind counts changed"
        )
    return copy.deepcopy(dict(value))


def _granular_checkpoint_coverage(
    nodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    artifact_ids: list[str] = []
    for node in nodes:
        kind = str(node["artifact_kind"])
        counts[kind] = counts.get(kind, 0) + 1
        artifact_ids.append(str(node["artifact_id"]))
    body = {
        "schema_version": "granular_checkpoint_coverage_v1",
        "artifact_kind_counts": dict(sorted(counts.items())),
        "ordered_artifact_ids": artifact_ids,
        "node_count": len(nodes),
        "payload_bytes_referenced_without_copy": True,
    }
    return {**body, "content_sha256": _sha(body)}


def _granular_primary_metadata_from_index(
    *,
    phase: str,
    index: Mapping[str, Any],
) -> Mapping[str, Any]:
    if phase == "stage1_modeling":
        terminal_kinds = {"logical_scope_bindings", "row_map"}
    elif phase == "stage2_inference":
        terminal_kinds = {"stage2_fold"}
    else:
        raise ValueError(
            f"{phase} has no granular primary checkpoint binding"
        )
    nodes = index.get("nodes")
    coverage = index.get("coverage")
    if not isinstance(nodes, list) or not isinstance(coverage, Mapping):
        raise ValueError(f"{phase} granular index is incomplete")
    return {
        "schema_version": (
            "workflow_primary_granular_coverage_binding_v1"
        ),
        "producer_phase": phase,
        "granular_index_content_sha256": index["content_sha256"],
        "granular_coverage_content_sha256": coverage[
            "content_sha256"
        ],
        "granular_artifact_ids": [
            node["artifact_id"] for node in nodes
        ],
        "granular_terminal_artifact_ids": [
            node["artifact_id"]
            for node in nodes
            if node["artifact_kind"] in terminal_kinds
        ],
        "granular_artifact_kind_counts": copy.deepcopy(
            coverage["artifact_kind_counts"]
        ),
    }


def _validate_exact_granular_upstream_edges(
    *,
    phase: str,
    artifacts: Sequence[ValidatedPortableArtifact],
    expected_plan: Mapping[str, Any],
    expected_external_upstream_artifact_ids: Sequence[str],
) -> None:
    """Require the exact scientific dependency edge for every granular node."""

    plan = _validate_expected_granular_checkpoint_plan(expected_plan)
    external = tuple(
        str(value) for value in expected_external_upstream_artifact_ids
    )

    def artifacts_of_kind(
        artifact_kind: str,
    ) -> tuple[ValidatedPortableArtifact, ...]:
        return tuple(
            artifact
            for artifact in artifacts
            if artifact.manifest["artifact_kind"] == artifact_kind
        )

    def exact_upstream(
        artifact: ValidatedPortableArtifact,
        expected: Sequence[str],
        *,
        label: str,
    ) -> None:
        observed = tuple(
            str(value)
            for value in artifact.manifest["upstream_artifact_ids"]
        )
        if observed != tuple(str(value) for value in expected):
            raise ValueError(
                f"{phase} granular {label} upstream edge changed"
            )

    if phase == "stage1_modeling":
        if len(external) != 1:
            raise ValueError(
                "Stage 1 granular validation requires exactly one "
                "authenticated prepared-context parent"
            )
        expected_owners = tuple(
            str(value)
            for value in plan["stage1_physical_owner_scope_ids"]
        )

        def owner_map(
            artifact_kind: str,
        ) -> dict[str, ValidatedPortableArtifact]:
            output: dict[str, ValidatedPortableArtifact] = {}
            for artifact in artifacts_of_kind(artifact_kind):
                owner = str(
                    artifact.artifact_metadata.get(
                        "physical_owner_scope_id", ""
                    )
                )
                if not owner or owner in output:
                    raise ValueError(
                        "Stage 1 granular physical-owner edge coverage "
                        "is absent or duplicated"
                    )
                output[owner] = artifact
            if tuple(output) != expected_owners:
                raise ValueError(
                    "Stage 1 granular physical-owner edge coverage changed"
                )
            return output

        tfidf_by_owner = owner_map("tfidf_component")
        neural_by_owner = owner_map("neural_query_component")
        physical_by_owner = owner_map("physical_scope_fit")
        for owner in expected_owners:
            tfidf = tfidf_by_owner[owner]
            neural = neural_by_owner[owner]
            physical = physical_by_owner[owner]
            exact_upstream(
                tfidf,
                external,
                label=f"TF-IDF component {owner}",
            )
            exact_upstream(
                neural,
                external,
                label=f"neural-query component {owner}",
            )
            exact_upstream(
                physical,
                (
                    *external,
                    tfidf.artifact_id,
                    neural.artifact_id,
                ),
                label=f"physical fit {owner}",
            )

        logical_by_id: dict[str, ValidatedPortableArtifact] = {}
        for artifact in artifacts_of_kind("logical_scope_bindings"):
            logical_id = str(
                artifact.artifact_metadata.get("logical_scope_id", "")
            )
            if not logical_id or logical_id in logical_by_id:
                raise ValueError(
                    "Stage 1 granular logical edge coverage is absent or "
                    "duplicated"
                )
            logical_by_id[logical_id] = artifact
        expected_logical = tuple(
            str(value) for value in plan["stage1_logical_scope_ids"]
        )
        if tuple(logical_by_id) != expected_logical:
            raise ValueError(
                "Stage 1 granular logical edge coverage changed"
            )
        expected_logical_to_owner = dict(
            plan["stage1_logical_to_physical_owner"]
        )
        for logical_id in expected_logical:
            owner = str(expected_logical_to_owner[logical_id])
            exact_upstream(
                logical_by_id[logical_id],
                (physical_by_owner[owner].artifact_id,),
                label=f"logical binding {logical_id}",
            )

        row_maps = artifacts_of_kind("row_map")
        if len(row_maps) != 1:
            raise ValueError(
                "Stage 1 granular row-map edge coverage changed"
            )
        exact_upstream(
            row_maps[0],
            tuple(
                logical_by_id[logical_id].artifact_id
                for logical_id in expected_logical
            ),
            label="row map",
        )
        return

    if phase != "stage2_inference":
        return
    if len(external) != len(
        PORTABLE_CHECKPOINT_PHASE_SPECS[phase]["upstream_phases"]
    ):
        raise ValueError(
            "Stage 2 granular validation requires its exact authenticated "
            "workflow parents"
        )
    responses = artifacts_of_kind("stage2_response_component")
    extractions = artifacts_of_kind("stage2_extraction_component")
    if len(responses) != 1 or len(extractions) != 1:
        raise ValueError(
            "Stage 2 granular response/extraction edge coverage changed"
        )
    response = responses[0]
    extraction = extractions[0]
    exact_upstream(
        response,
        external,
        label="response component",
    )
    exact_upstream(
        extraction,
        (response.artifact_id,),
        label="extraction component",
    )

    def fold_map(
        artifact_kind: str,
    ) -> dict[int, ValidatedPortableArtifact]:
        output: dict[int, ValidatedPortableArtifact] = {}
        for artifact in artifacts_of_kind(artifact_kind):
            outer_fold = artifact.artifact_metadata.get("outer_fold")
            if (
                isinstance(outer_fold, bool)
                or not isinstance(outer_fold, int)
                or outer_fold in output
            ):
                raise ValueError(
                    "Stage 2 granular fold edge coverage is invalid"
                )
            output[int(outer_fold)] = artifact
        return output

    reviews = fold_map("stage2_review_component")
    folds = fold_map("stage2_fold")
    expected_reviews = tuple(
        int(value) for value in plan["stage2_review_fold_ids"]
    )
    expected_folds = tuple(
        int(value) for value in plan["stage2_fold_ids"]
    )
    if tuple(reviews) != expected_reviews or tuple(folds) != expected_folds:
        raise ValueError(
            "Stage 2 granular fold/review edge coverage changed"
        )
    for outer_fold in expected_reviews:
        exact_upstream(
            reviews[outer_fold],
            (extraction.artifact_id,),
            label=f"review component outer fold {outer_fold}",
        )
    for outer_fold in expected_folds:
        exact_upstream(
            folds[outer_fold],
            (
                response.artifact_id,
                extraction.artifact_id,
                reviews[outer_fold].artifact_id,
            ),
            label=f"fold outer fold {outer_fold}",
        )


def _validate_granular_handles_against_plan(
    *,
    phase: str,
    artifacts: Sequence[ValidatedPortableArtifact],
    expected_plan: Mapping[str, Any],
    expected_stage1_scope_plan: Any | None = None,
    expected_external_upstream_artifact_ids: Sequence[str] | None = None,
) -> None:
    plan = _validate_expected_granular_checkpoint_plan(expected_plan)
    counts: dict[str, int] = {}
    for artifact in artifacts:
        kind = str(artifact.manifest["artifact_kind"])
        counts[kind] = counts.get(kind, 0) + 1
    expected_counts = dict(
        plan[
            (
                "stage1_artifact_kind_counts"
                if phase == "stage1_modeling"
                else "stage2_artifact_kind_counts"
            )
        ]
    )
    if phase not in {"stage1_modeling", "stage2_inference"}:
        return
    if dict(sorted(counts.items())) != expected_counts:
        raise ValueError(
            f"{phase} granular checkpoint differs from the request plan"
        )
    if phase == "stage1_modeling":
        if expected_stage1_scope_plan is None:
            raise ValueError(
                "Stage 1 granular validation requires the authenticated "
                "current scope plan"
            )
        exact_projection = _stage1_scope_plan_granular_expectations(
            scope_plan=expected_stage1_scope_plan,
            expected_granular_checkpoint_plan=plan,
        )
        expected_key_records = exact_projection[
            "physical_fit_key_records_by_owner"
        ]
        for artifact in artifacts:
            if artifact.manifest["artifact_kind"] not in {
                "tfidf_component",
                "neural_query_component",
                "physical_scope_fit",
                "logical_scope_bindings",
            }:
                continue
            owner = str(
                artifact.artifact_metadata.get(
                    "physical_owner_scope_id", ""
                )
            )
            expected_record = expected_key_records.get(owner)
            if not isinstance(expected_record, Mapping):
                raise ValueError(
                    "Stage 1 granular owner is absent from the "
                    "authenticated current scope plan"
                )
            _validated_stage1_granular_physical_fit_key(
                metadata=artifact.artifact_metadata,
                expected_identity=(
                    expected_stage1_scope_plan.physical_fit_identity
                ),
                expected_key_record=expected_record,
            )
        expected_owners = list(
            plan["stage1_physical_owner_scope_ids"]
        )
        for kind in (
            "tfidf_component",
            "neural_query_component",
            "physical_scope_fit",
        ):
            owners = [
                str(
                    artifact.artifact_metadata.get(
                        "physical_owner_scope_id"
                    )
                )
                for artifact in artifacts
                if artifact.manifest["artifact_kind"] == kind
            ]
            if owners != expected_owners:
                raise ValueError(
                    "Stage 1 granular physical-owner coverage changed"
                )
        logical_to_owner = {
            str(artifact.artifact_metadata.get("logical_scope_id")): str(
                artifact.artifact_metadata.get(
                    "physical_owner_scope_id"
                )
            )
            for artifact in artifacts
            if artifact.manifest["artifact_kind"]
            == "logical_scope_bindings"
        }
        if (
            list(logical_to_owner)
            != list(plan["stage1_logical_scope_ids"])
            or logical_to_owner
            != dict(plan["stage1_logical_to_physical_owner"])
        ):
            raise ValueError(
                "Stage 1 granular logical coverage changed"
            )
    else:
        folds = [
            artifact.artifact_metadata.get("outer_fold")
            for artifact in artifacts
            if artifact.manifest["artifact_kind"] == "stage2_fold"
        ]
        reviews = [
            artifact.artifact_metadata.get("outer_fold")
            for artifact in artifacts
            if artifact.manifest["artifact_kind"]
            == "stage2_review_component"
        ]
        if (
            folds != list(plan["stage2_fold_ids"])
            or reviews != list(plan["stage2_review_fold_ids"])
        ):
            raise ValueError(
                "Stage 2 granular fold/review coverage changed"
            )
    if expected_external_upstream_artifact_ids is None:
        raise ValueError(
            f"{phase} granular validation lacks its authenticated external "
            "parents"
        )
    _validate_exact_granular_upstream_edges(
        phase=phase,
        artifacts=artifacts,
        expected_plan=plan,
        expected_external_upstream_artifact_ids=(
            expected_external_upstream_artifact_ids
        ),
    )


def _validate_granular_checkpoint_index_from_paths(
    *,
    work_root: Path,
    phase: str,
    compatibility: ArtifactCompatibility,
    payload_authentication_cache: MutableMapping[
        str, tuple[tuple[int, ...], str, int]
    ]
    | None = None,
    expected_granular_checkpoint_plan: Mapping[str, Any] | None = None,
    expected_stage1_scope_plan: Any | None = None,
    expected_external_upstream_artifact_ids: Sequence[str] | None = None,
) -> tuple[Mapping[str, Any], tuple[ValidatedPortableArtifact, ...]]:
    """Freshly reopen one locator-separated granular checkpoint index."""

    root, index_path, locator_path = _granular_checkpoint_index_paths(
        work_root=work_root,
        phase=phase,
    )
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{phase} granular checkpoint root is invalid")
    index = _read_private_json_object(
        index_path,
        label=f"{phase} granular checkpoint index",
    )
    index_fields = {
        "schema_version",
        "phase",
        "node_count",
        "nodes",
        "coverage",
        "relative_filesystem_layout_included",
        "content_sha256",
    }
    index_body = {
        key: copy.deepcopy(value)
        for key, value in index.items()
        if key != "content_sha256"
    }
    nodes = index.get("nodes")
    if (
        set(index) != index_fields
        or index.get("schema_version")
        != WORKFLOW_GRANULAR_CHECKPOINT_INDEX_SCHEMA
        or index.get("phase") != phase
        or not isinstance(nodes, list)
        or not nodes
        or index.get("node_count") != len(nodes)
        or index.get("relative_filesystem_layout_included") is not False
        or index.get("content_sha256") != _sha(index_body)
    ):
        raise ValueError(f"{phase} granular checkpoint index is invalid")
    expected_node_fields = {
        "node_ordinal",
        "node_key",
        "artifact_id",
        "artifact_kind",
        "artifact_schema",
        "upstream_artifact_ids",
        "artifact_metadata",
    }
    artifact_ids: list[str] = []
    node_keys: list[str] = []
    for ordinal, node in enumerate(nodes):
        if (
            not isinstance(node, Mapping)
            or set(node) != expected_node_fields
            or node.get("node_ordinal") != ordinal
            or not isinstance(node.get("node_key"), str)
            or not str(node["node_key"])
            or str(node.get("artifact_kind"))
            not in GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS
            or node.get("artifact_schema")
            != GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS[
                str(node["artifact_kind"])
            ]
            or CHECKPOINT_COMPATIBILITY_PHASE_BY_ARTIFACT_KIND.get(
                str(node["artifact_kind"])
            )
            != phase
            or not isinstance(node.get("upstream_artifact_ids"), list)
            or not isinstance(node.get("artifact_metadata"), Mapping)
        ):
            raise ValueError(
                f"{phase} granular checkpoint node descriptor is invalid"
            )
        artifact_ids.append(str(node["artifact_id"]))
        node_keys.append(str(node["node_key"]))
    if (
        len(set(artifact_ids)) != len(artifact_ids)
        or len(set(node_keys)) != len(node_keys)
        or index.get("coverage") != _granular_checkpoint_coverage(nodes)
    ):
        raise ValueError(
            f"{phase} granular checkpoint coverage is duplicated or changed"
        )

    locator = _read_private_json_object(
        locator_path,
        label=f"{phase} granular checkpoint locator",
    )
    locator_fields = {
        "schema_version",
        "phase",
        "index_content_sha256",
        "index_path",
        "phase_manifest_path",
        "phase_manifest_sha256",
        "phase_manifest_size_bytes",
        "node_controls",
        "content_sha256",
    }
    locator_body = {
        key: copy.deepcopy(value)
        for key, value in locator.items()
        if key != "content_sha256"
    }
    controls = locator.get("node_controls")
    if (
        set(locator) != locator_fields
        or locator.get("schema_version")
        != WORKFLOW_GRANULAR_CHECKPOINT_LOCATOR_SCHEMA
        or locator.get("phase") != phase
        or locator.get("index_content_sha256")
        != index["content_sha256"]
        or locator.get("index_path")
        != str(index_path.resolve(strict=True))
        or not isinstance(controls, list)
        or len(controls) != len(nodes)
        or locator.get("content_sha256") != _sha(locator_body)
    ):
        raise ValueError(f"{phase} granular checkpoint locator is invalid")
    phase_manifest_path = Path(str(locator["phase_manifest_path"]))
    phase_manifest_sha, phase_manifest_size = stable_file_sha256(
        phase_manifest_path.resolve(strict=True)
    )
    if (
        phase_manifest_path
        != (work_root / "phases" / phase / "complete_manifest.json").resolve(
            strict=True
        )
        or phase_manifest_sha != locator.get("phase_manifest_sha256")
        or phase_manifest_size
        != int(locator.get("phase_manifest_size_bytes", -1))
    ):
        raise ValueError(
            f"{phase} granular checkpoint phase-manifest locator changed"
        )
    expected_control_fields = {
        "node_ordinal",
        "artifact_id",
        "control_root",
    }
    handles: list[ValidatedPortableArtifact] = []
    observed_control_names: set[str] = set()
    prior_granular_ids: set[str] = set()
    all_granular_ids = set(artifact_ids)
    for ordinal, (node, control) in enumerate(
        zip(nodes, controls, strict=True)
    ):
        if (
            not isinstance(control, Mapping)
            or set(control) != expected_control_fields
            or control.get("node_ordinal") != ordinal
            or control.get("artifact_id") != node["artifact_id"]
        ):
            raise ValueError(
                f"{phase} granular checkpoint control mapping is invalid"
            )
        control_root = Path(str(control["control_root"]))
        if (
            not control_root.is_absolute()
            or control_root.is_symlink()
            or control_root.resolve(strict=True).parent
            != (root / "nodes").resolve(strict=True)
        ):
            raise ValueError(
                f"{phase} granular checkpoint control escaped its index"
            )
        observed_control_names.add(control_root.name)
        upstream = tuple(str(value) for value in node["upstream_artifact_ids"])
        if (set(upstream) & all_granular_ids) - prior_granular_ids:
            raise ValueError(
                f"{phase} granular checkpoint index is not topological"
            )
        artifact = validate_portable_artifact(
            control_root,
            expected_kind=str(node["artifact_kind"]),
            expected_compatibility_key=compatibility.key,
            expected_upstream_artifact_ids=upstream,
            payload_authentication_cache=payload_authentication_cache,
        )
        if (
            artifact.artifact_id != node["artifact_id"]
            or artifact.manifest.get("artifact_schema")
            != node["artifact_schema"]
            or dict(artifact.artifact_metadata)
            != dict(node["artifact_metadata"])
        ):
            raise ValueError(
                f"{phase} granular checkpoint artifact changed"
            )
        handles.append(artifact)
        prior_granular_ids.add(artifact.artifact_id)
    nodes_root = root / "nodes"
    if nodes_root.is_symlink() or not nodes_root.is_dir():
        raise ValueError(
            f"{phase} granular checkpoint controls are invalid"
        )
    actual_control_names = {
        child.name
        for child in nodes_root.iterdir()
        if child.is_dir() and not child.is_symlink()
    }
    if (
        actual_control_names != observed_control_names
        or any(
            child.is_symlink() or not child.is_dir()
            for child in nodes_root.iterdir()
        )
        or {
            child.name for child in root.iterdir()
        }
        != {
            "nodes",
            index_path.name,
            locator_path.name,
        }
    ):
        raise ValueError(
            f"{phase} granular checkpoint controls contain missing or extra entries"
        )
    if expected_granular_checkpoint_plan is not None:
        _validate_granular_handles_against_plan(
            phase=phase,
            artifacts=tuple(handles),
            expected_plan=expected_granular_checkpoint_plan,
            expected_stage1_scope_plan=expected_stage1_scope_plan,
            expected_external_upstream_artifact_ids=(
                expected_external_upstream_artifact_ids
            ),
        )
    return copy.deepcopy(index), tuple(handles)


def _persist_legacy_preflight_recompute_decision(
    *,
    attempt: Path,
    consumer_request_sha256: str,
    source_candidate_identity: Mapping[str, Any],
    migration: Mapping[str, Any],
    expected_logical_scope_count: int,
    expected_physical_fit_count: int,
) -> tuple[Path, Mapping[str, Any]]:
    """Validate and seal one audit-only legacy-preflight recompute decision."""

    migration_fields = {
        "schema_version",
        "decision",
        "source_legacy_preflight_status",
        "source_legacy_preflight_directly_reusable",
        "accounting",
        "logical_scope_count",
        "physical_fit_count",
        "deduplicated_group_count",
        "recompute_physical_fit_count",
        "recompute_reason_codes",
        "dependency_proof",
        "migration_is_reference_only_no_refit",
        "legacy_tree_mutation_allowed",
        "content_sha256",
    }
    accounting_fields = {
        "schema_version",
        "source_manifest_content_sha256",
        "source_audit",
        "source_stage1_request",
        "logical_scope_count",
        "physical_fit_count",
        "deduplicated_group_count",
        "physical_records",
        "logical_bindings",
        "superseded_duplicate_outputs",
        "legacy_payloads_authenticated_once_at_fresh_trust_boundary",
        "canonical_owners_selected_by_content_and_earliest_index",
        "canonical_owner_row_order_and_requested_seed_retained",
        "source_tree_mutated",
        "legacy_payload_copies_materialized",
        "content_sha256",
    }
    dependency_fields = {
        "source_manifest_structure_and_registry_validated",
        "registered_payload_bytes_freshly_authenticated",
        "requested_logical_scope_inventory_matches",
        "requested_fit_row_orders_match_legacy_records",
        "canonical_owner_scope_seed_registered_by_legacy_producer",
        "safe_kmeans_svd_state_payload_inventory_present",
        "legacy_internal_payload_graph_replayed_under_current_schema",
        "requested_current_compatibility_key_proved",
        "all_dependencies_and_evidence_identities_proved",
    }
    candidate_fields = {
        "selection_source",
        "manifest_path",
        "manifest_sha256",
        "manifest_size_bytes",
        "manifest_content_sha256",
        "registered_payloads",
        "registered_payload_bytes_authenticated_during_request",
        "direct_reuse_allowed",
    }
    accounting = migration.get("accounting")
    dependency_proof = migration.get("dependency_proof")
    expected_duplicates = int(expected_logical_scope_count) - int(expected_physical_fit_count)
    if (
        set(migration) != migration_fields
        or migration.get("content_sha256")
        != identity_sha256(
            {
                key: copy.deepcopy(value)
                for key, value in migration.items()
                if key != "content_sha256"
            }
        )
        or not isinstance(accounting, Mapping)
        or set(accounting) != accounting_fields
        or accounting.get("content_sha256")
        != identity_sha256(
            {
                key: copy.deepcopy(value)
                for key, value in accounting.items()
                if key != "content_sha256"
            }
        )
        or not isinstance(dependency_proof, Mapping)
        or set(dependency_proof) != dependency_fields
        or set(source_candidate_identity) != candidate_fields
        or source_candidate_identity.get("direct_reuse_allowed") is not False
        or source_candidate_identity.get("registered_payload_bytes_authenticated_during_request")
        is not False
        or accounting.get("source_manifest_content_sha256")
        != source_candidate_identity.get("manifest_content_sha256")
        or migration.get("decision") != "recompute_required"
        or migration.get("source_legacy_preflight_directly_reusable") is not False
        or migration.get("logical_scope_count") != int(expected_logical_scope_count)
        or migration.get("physical_fit_count") != int(expected_physical_fit_count)
        or migration.get("recompute_physical_fit_count") != int(expected_physical_fit_count)
        or migration.get("deduplicated_group_count") != expected_duplicates
        or accounting.get("logical_scope_count") != int(expected_logical_scope_count)
        or accounting.get("physical_fit_count") != int(expected_physical_fit_count)
        or accounting.get("deduplicated_group_count") != expected_duplicates
        or len(accounting.get("superseded_duplicate_outputs") or ()) != expected_duplicates
        or accounting.get("legacy_payloads_authenticated_once_at_fresh_trust_boundary") is not True
        or dependency_proof.get("registered_payload_bytes_freshly_authenticated") is not True
        or dependency_proof.get("all_dependencies_and_evidence_identities_proved") is not False
    ):
        raise RuntimeError(
            "legacy preflight migration did not prove the complete "
            "request-derived recompute decision"
        )
    decision_body = {
        "schema_version": WORKFLOW_LEGACY_PREFLIGHT_DECISION_SCHEMA,
        "consumer_request_sha256": consumer_request_sha256,
        "source_candidate_identity": copy.deepcopy(dict(source_candidate_identity)),
        "migration_decision": copy.deepcopy(dict(migration)),
        "adoption_disposition": "audit_only_not_checkpoint_adoption",
        "current_preflight_recomputed": True,
        "legacy_fitted_output_reused": False,
        "terminal_registration_required": True,
        "source_tree_mutated": False,
    }
    decision = {
        **decision_body,
        "content_sha256": _sha(decision_body),
    }
    path = Path(attempt) / "legacy_preflight_migration_decision.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError("legacy preflight migration decision already exists")
    _atomic_write_json(path, decision)
    if (
        _read_json_object(
            path,
            label="legacy preflight migration decision",
        )
        != decision
    ):
        raise RuntimeError("legacy preflight migration decision changed while sealing")
    return path.resolve(strict=True), decision


def _validate_preflight_candidate_selector(
    path: Path,
) -> tuple[Mapping[str, Any], str, Path | None]:
    """Validate a preflight selector without reading its bulk payloads."""

    candidate = Path(path).resolve(strict=True)
    value = _read_json_object(
        candidate,
        label="selected preflight manifest",
    )
    if value.get("schema_version") == (
        "production_stage1_reusable_preflight_reference_v1"
    ):
        body = {
            key: copy.deepcopy(child)
            for key, child in value.items()
            if key != "content_sha256"
        }
        assembled_path = Path(
            str(value.get("assembled_terminal_path", ""))
        )
        if (
            value.get("status") != "complete"
            or value.get("content_sha256") != _sha(body)
            or value.get("owner_payloads_copied") is not False
            or value.get("locator_is_operational_not_scientific")
            is not True
            or not assembled_path.is_absolute()
            or assembled_path.name
            != "assembled_preflight_terminal.json"
            or not assembled_path.is_file()
            or assembled_path.is_symlink()
        ):
            raise ValueError(
                "reusable preflight selector terminal is invalid"
            )
        assembled = _read_json_object(
            assembled_path,
            label="selected reusable assembled preflight terminal",
        )
        assembled_body = {
            key: copy.deepcopy(child)
            for key, child in assembled.items()
            if key != "content_sha256"
        }
        if (
            assembled.get("schema_version")
            != "production_stage1_reusable_assembled_preflight_artifact_v2"
            or assembled.get("status") != "complete"
            or assembled.get("content_sha256")
            != _sha(assembled_body)
            or assembled.get("content_sha256")
            != value.get("assembled_terminal_content_sha256")
            or assembled.get(
                "artifact_scientific_content_sha256"
            )
            != value.get(
                "assembled_scientific_content_sha256"
            )
        ):
            raise ValueError(
                "reusable preflight selector assembled binding changed"
            )
        state_manifest = (
            candidate.parent.parent
            / "cluster_preflight_states"
            / "cluster_state_bundle_manifest.json"
        ).resolve(strict=True)
        state_value = _read_json_object(
            state_manifest,
            label="selected reusable preflight state reference",
        )
        state_body = {
            key: copy.deepcopy(child)
            for key, child in state_value.items()
            if key != "content_sha256"
        }
        prepared_context_manifest = (
            candidate.parent.parent
            / "prepared_stage1_context"
            / "prepared_stage1_context_manifest.json"
        ).resolve(strict=True)
        if (
            state_value.get("schema_version")
            != "production_stage1_reusable_cluster_state_bundle_reference_v1"
            or state_value.get("status") != "complete"
            or state_value.get("content_sha256") != _sha(state_body)
            or state_value.get("cluster_refit_performed") is not False
            or state_value.get("owner_payloads_copied") is not False
            or state_value.get("assembled_terminal_path")
            != str(assembled_path)
            or not prepared_context_manifest.is_file()
            or prepared_context_manifest.is_symlink()
        ):
            raise ValueError(
                "reusable preflight selector state/context binding changed"
            )
        return (
            {
                "manifest": copy.deepcopy(value),
                "payloads": {},
                "assembled_terminal_path": str(assembled_path),
                "prepared_context_manifest_path": str(
                    prepared_context_manifest
                ),
            },
            "reusable_v1",
            state_manifest,
        )
    if value.get("schema_version") != (
        "production_stage1_cluster_preflight_manifest_v2"
    ):
        from .legacy_checkpoint_migration import (
            validate_legacy_preflight_manifest,
        )

        return (
            validate_legacy_preflight_manifest(
                candidate,
                authenticate_registered_payload_bytes=False,
            ),
            "legacy_v4",
            None,
        )
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    files = value.get("files")
    if (
        value.get("artifact_version")
        != "production_stage1_cluster_preflight_artifact_v2"
        or value.get("status") != "complete"
        or value.get("content_sha256") != _sha(body)
        or not isinstance(files, list)
        or not files
        or value.get("logical_scope_count") is None
        or value.get("physical_fit_count") is None
    ):
        raise ValueError(
            "portable-v2 preflight selector terminal is invalid"
        )
    payloads: dict[str, dict[str, Any]] = {}
    for row in files:
        relative = (
            row.get("relative_path")
            if isinstance(row, Mapping)
            else None
        )
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in payloads
        ):
            raise ValueError(
                "portable-v2 preflight selector inventory is invalid"
            )
        payload_path = candidate.parent / relative
        state = os.lstat(payload_path)
        if (
            stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or int(state.st_nlink) != 1
            or int(state.st_size) != int(row.get("size_bytes", -1))
        ):
            raise ValueError(
                "portable-v2 preflight selector payload is absent or linked"
            )
        payloads[relative] = {
            "path": str(payload_path.resolve(strict=True)),
            "sha256": str(row.get("sha256")),
            "size_bytes": int(row["size_bytes"]),
        }
    state_manifest = (
        candidate.parent.parent
        / "cluster_preflight_states"
        / "cluster_state_bundle_manifest.json"
    ).resolve(strict=True)
    state_value = _read_json_object(
        state_manifest,
        label="selected portable-v2 preflight state bundle",
    )
    state_body = {
        key: copy.deepcopy(child)
        for key, child in state_value.items()
        if key != "content_sha256"
    }
    if (
        state_value.get("schema_version")
        != "production_canonical_clustered_preflight_state_bundle_v2"
        or state_value.get("status") != "complete"
        or state_value.get("cluster_refit_performed") is not False
        or state_value.get("content_sha256") != _sha(state_body)
        or state_value.get("physical_owner_scope_order")
        != value.get("physical_scope_order")
    ):
        raise ValueError(
            "portable-v2 preflight state selector is incompatible"
        )
    return (
        {
            "manifest": copy.deepcopy(value),
            "payloads": payloads,
        },
        "portable_v2",
        state_manifest,
    )


def _attempt_tree_artifacts(attempt_dir: Path) -> list[dict[str, Any]]:
    """Return the exact closed regular-file inventory for one phase attempt."""

    if attempt_dir.is_symlink() or not attempt_dir.is_dir():
        raise ValueError(f"phase attempt must be one real directory: {attempt_dir}")
    root = attempt_dir.resolve(strict=True)
    artifacts: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        state = os.lstat(candidate)
        if stat.S_ISLNK(state.st_mode):
            raise ValueError(f"phase attempt contains a symlink: {candidate}")
        if stat.S_ISDIR(state.st_mode):
            continue
        if not stat.S_ISREG(state.st_mode):
            raise ValueError(f"phase attempt contains a special file: {candidate}")
        if state.st_nlink != 1:
            raise ValueError(f"phase attempt contains a hard-linked file: {candidate}")
        resolved = candidate.resolve(strict=True)
        digest, size = stable_file_sha256(resolved)
        artifacts.append(
            {
                "relative_path": resolved.relative_to(root).as_posix(),
                "path": str(resolved),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return artifacts


def _complete_regular_file_tree(root: Path) -> tuple[str, ...]:
    tree = Path(root).resolve(strict=True)
    if tree.is_symlink() or not tree.is_dir():
        raise ValueError("terminal payload tree must be a real directory")
    output: list[str] = []
    for path in sorted(tree.rglob("*")):
        state = os.lstat(path)
        if stat.S_ISLNK(state.st_mode):
            raise ValueError("terminal payload tree contains a symlink")
        if stat.S_ISDIR(state.st_mode):
            continue
        if not stat.S_ISREG(state.st_mode) or int(state.st_nlink) != 1:
            raise ValueError(
                "terminal payload tree contains non-private data"
            )
        output.append(str(path.resolve(strict=True)))
    return tuple(output)


def _portable_stage1_terminal_file_inventory(
    *,
    execution_root: Path,
    bundle_root: Path,
    numerical_bank_root: Path,
    binding_path: Path,
) -> tuple[str, ...]:
    """Return every Stage 1 byte claimed by granular/handoff checkpoints."""

    values = (
        *_complete_regular_file_tree(execution_root),
        *_complete_regular_file_tree(bundle_root),
        *_complete_regular_file_tree(numerical_bank_root),
        str(Path(binding_path).resolve(strict=True)),
    )
    if len(values) != len(set(values)):
        raise RuntimeError(
            "Stage 1 terminal payload roots overlap or duplicate bytes"
        )
    return tuple(values)


def _portable_stage2_terminal_file_inventory(
    *,
    result: Mapping[str, Any],
    prediction_path: Path,
    run_manifest_path: Path,
    attestation_path: Path,
) -> tuple[str, ...]:
    """Return exact Stage 2-owned response/review/extraction/fold bytes."""

    required_lists = {
        "fold_manifest_paths": result.get("fold_manifest_paths"),
        "fold_prediction_paths": result.get("fold_prediction_paths"),
        "complete_paged_ledger_artifact_paths": result.get(
            "complete_paged_ledger_artifact_paths"
        ),
    }
    if any(
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) for path in paths)
        for paths in required_lists.values()
    ):
        raise RuntimeError(
            "direct Stage 2 result omitted fold terminal artifacts"
        )
    batch_result_path = Path(
        str(result["hierarchical_batch_result_path"])
    ).resolve(strict=True)
    review_terminals: list[str] = []
    for raw_fold_manifest in required_lists["fold_manifest_paths"]:
        review_root = (
            Path(raw_fold_manifest).resolve(strict=True).parent
            / "post_extraction_review"
        )
        if review_root.exists():
            review_terminals.extend(
                _complete_regular_file_tree(review_root)
            )
    raw = [
        str(
            Path(
                str(result["runner_input_manifest_path"])
            ).resolve(strict=True)
        ),
        *_complete_regular_file_tree(batch_result_path.parent),
        *[
            str(Path(path).resolve(strict=True))
            for path in required_lists["fold_manifest_paths"]
        ],
        *[
            str(Path(path).resolve(strict=True))
            for path in required_lists["fold_prediction_paths"]
        ],
        *[
            str(Path(path).resolve(strict=True))
            for path in required_lists[
                "complete_paged_ledger_artifact_paths"
            ]
        ],
        *review_terminals,
        str(Path(prediction_path).resolve(strict=True)),
        str(Path(run_manifest_path).resolve(strict=True)),
        str(Path(attestation_path).resolve(strict=True)),
    ]
    values = tuple(dict.fromkeys(raw))
    if len(values) != len(set(values)):
        raise RuntimeError(
            "Stage 2 terminal payload inventory is duplicated"
        )
    return values


def _rewrite_attempt_locators(
    value: Any,
    *,
    source_root: Path,
    published_root: Path,
) -> Any:
    """Rewrite only absolute locators rooted in one phase attempt.

    Scientific identities never depend on either locator.  This traversal is
    deliberately conservative: ordinary strings and paths outside the attempt
    are left byte-for-byte unchanged.
    """

    if isinstance(value, Mapping):
        return {
            key: _rewrite_attempt_locators(
                child,
                source_root=source_root,
                published_root=published_root,
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_attempt_locators(
                child,
                source_root=source_root,
                published_root=published_root,
            )
            for child in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _rewrite_attempt_locators(
                child,
                source_root=source_root,
                published_root=published_root,
            )
            for child in value
        )
    if not isinstance(value, (str, Path)):
        return value
    raw = str(value)
    candidate = Path(raw)
    if not candidate.is_absolute():
        return value
    try:
        relative = candidate.relative_to(source_root)
    except ValueError:
        return value
    rewritten = published_root / relative
    return rewritten if isinstance(value, Path) else str(rewritten)


def _assert_attempt_inventory_unchanged(
    attempt_dir: Path,
    artifacts: Sequence[Mapping[str, Any]],
) -> None:
    """Check the closed tree without rereading payload bytes."""

    expected = {str(row["relative_path"]): int(row["size_bytes"]) for row in artifacts}
    observed: dict[str, int] = {}
    for candidate in sorted(attempt_dir.rglob("*")):
        state = os.lstat(candidate)
        if stat.S_ISLNK(state.st_mode):
            raise ValueError(f"phase attempt contains a symlink: {candidate}")
        if stat.S_ISDIR(state.st_mode):
            continue
        if not stat.S_ISREG(state.st_mode):
            raise ValueError(f"phase attempt contains a special file: {candidate}")
        observed[candidate.relative_to(attempt_dir).as_posix()] = int(state.st_size)
    if observed != expected:
        raise ValueError("phase attempt changed during durable publication")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_owned_scratch_tree_after_publication(path: Path) -> None:
    """Best-effort cleanup after the durable tree has already committed.

    Terminal producers may deliberately seal nested artifacts read-only.  The
    workflow owns its scratch attempt, so make only its directories removable
    before deleting it.  A cleanup failure after the durable rename must not
    turn the already-committed publication into a failed phase; the uniquely
    named scratch attempt can safely remain for an operator to inspect.
    """

    try:
        for current, directory_names, _file_names in os.walk(
            path,
            topdown=True,
            followlinks=False,
        ):
            current_path = Path(current)
            current_state = os.lstat(current_path)
            if (
                stat.S_ISLNK(current_state.st_mode)
                or not stat.S_ISDIR(current_state.st_mode)
            ):
                raise OSError(
                    f"owned scratch cleanup encountered a non-directory: "
                    f"{current_path}"
                )
            os.chmod(
                current_path,
                stat.S_IMODE(current_state.st_mode) | stat.S_IRWXU,
            )
            for name in directory_names:
                child = current_path / name
                child_state = os.lstat(child)
                if (
                    stat.S_ISLNK(child_state.st_mode)
                    or not stat.S_ISDIR(child_state.st_mode)
                ):
                    raise OSError(
                        "owned scratch cleanup encountered a non-directory: "
                        f"{child}"
                    )
        shutil.rmtree(path)
        _fsync_directory(path.parent)
    except OSError:
        LOGGER.warning(
            "durable phase publication committed but scratch cleanup was "
            "incomplete: %s",
            path,
            exc_info=True,
        )


def _phase_payload_stat_inventory(
    root: Path,
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, ...]]:
    """Capture the nonserializable same-process guard for authenticated bytes."""

    inventory: dict[str, tuple[int, ...]] = {}
    for row in artifacts:
        relative = str(row["relative_path"])
        state = os.lstat(root / relative)
        if (
            stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or int(state.st_nlink) != 1
            or int(state.st_size) != int(row["size_bytes"])
        ):
            raise RuntimeError(f"published phase payload changed before checkpointing: {relative}")
        inventory[relative] = (
            int(state.st_dev),
            int(state.st_ino),
            int(state.st_mode),
            int(state.st_nlink),
            int(state.st_size),
            int(state.st_mtime_ns),
            int(state.st_ctime_ns),
        )
    return inventory


def _phase_payload_proof_store_root(work_root: Path) -> Path:
    """Return the protected operational proof store for durable phase bytes."""

    return (
        Path(work_root)
        / "execution_attestations"
        / "phase_payload_authentication"
    )


def _phase_payload_proof_key(
    *,
    phase: str,
    request_sha256: str,
    terminal_content_sha256: str,
) -> str:
    return _sha(
        {
            "schema_version": (
                "production_workflow_phase_payload_proof_key_v1"
            ),
            "phase": str(phase),
            "request_sha256": str(request_sha256),
            "terminal_content_sha256": str(
                terminal_content_sha256
            ),
        }
    )


def _phase_payload_stat_inventory_from_proof(
    *,
    proof: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, ...]]:
    """Project one protected proof into the existing stat-only phase guard."""

    rows = proof.get("tree_stat_inventory")
    if not isinstance(rows, list):
        raise ValueError("phase payload proof has no stat inventory")
    files = {
        str(row.get("relative_path")): row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("kind") == "file"
    }
    expected = {
        str(row["relative_path"])
        for row in artifacts
    }
    if set(files) != expected:
        raise ValueError(
            "phase payload proof coverage differs from its manifest"
        )
    output: dict[str, tuple[int, ...]] = {}
    for registration in artifacts:
        relative = str(registration["relative_path"])
        row = files[relative]
        if (
            int(row.get("size_bytes", -1))
            != int(registration["size_bytes"])
            or int(row.get("link_count", -1)) != 1
        ):
            raise ValueError(
                "phase payload proof metadata differs from its manifest"
            )
        output[relative] = (
            int(row["device"]),
            int(row["inode"]),
            int(row["mode"]),
            int(row["link_count"]),
            int(row["size_bytes"]),
            int(row["mtime_ns"]),
            int(row["ctime_ns"]),
        )
    return output


def _publish_attempt_tree(
    *,
    attempt_dir: Path,
    durable_phase_root: Path,
    artifacts: Sequence[Mapping[str, Any]],
) -> tuple[Path, Mapping[str, int], Mapping[str, tuple[int, ...]]]:
    """Publish one authenticated attempt exactly once.

    A same-filesystem scratch tree is atomically renamed.  Across filesystems,
    every destination byte is hashed while it is copied, synchronized, and
    atomically renamed into place.  The owned scratch tree is removed only
    after the durable tree is complete.
    """

    source = attempt_dir.resolve(strict=True)
    durable_phase_root.mkdir(parents=True, exist_ok=True)
    durable_root = durable_phase_root.resolve(strict=True)
    published = durable_root / source.name
    if published.exists() or published.is_symlink():
        raise FileExistsError(f"durable phase attempt already exists: {published}")
    _assert_attempt_inventory_unchanged(source, artifacts)
    total_bytes = sum(int(row["size_bytes"]) for row in artifacts)
    counters = {
        "read": 0,
        "written": 0,
        "copied": 0,
        "hashed": 0,
        "fsynced": 0,
    }

    if os.stat(source).st_dev == os.stat(durable_root).st_dev:
        for row in artifacts:
            payload = source / str(row["relative_path"])
            descriptor = os.open(payload, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        counters["fsynced"] = total_bytes
        _fsync_directory(source)
        os.replace(source, published)
        _fsync_directory(durable_root)
        published = published.resolve(strict=True)
        return (
            published,
            counters,
            _phase_payload_stat_inventory(published, artifacts),
        )

    temporary = durable_root / (
        f".{source.name}.publishing_{os.getpid()}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    temporary.mkdir()
    try:
        for row in artifacts:
            relative = Path(str(row["relative_path"]))
            source_file = source / relative
            destination_file = temporary / relative
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            source_mode = stat.S_IMODE(os.lstat(source_file).st_mode)
            digest = hashlib.sha256()
            copied = 0
            with source_file.open("rb") as source_handle, destination_file.open(
                "xb"
            ) as destination_handle:
                while True:
                    block = source_handle.read(8 * 1024 * 1024)
                    if not block:
                        break
                    destination_handle.write(block)
                    digest.update(block)
                    copied += len(block)
                destination_handle.flush()
                os.fchmod(destination_handle.fileno(), source_mode)
                os.fsync(destination_handle.fileno())
            if copied != int(row["size_bytes"]) or digest.hexdigest() != str(row["sha256"]):
                raise RuntimeError(f"phase payload changed while publishing: {relative.as_posix()}")
            counters["read"] += copied
            counters["written"] += copied
            counters["copied"] += copied
            counters["hashed"] += copied
            counters["fsynced"] += copied
        _assert_attempt_inventory_unchanged(source, artifacts)
        directories = [temporary]
        directories.extend(candidate for candidate in temporary.rglob("*") if candidate.is_dir())
        for directory in sorted(
            directories,
            key=lambda item: len(item.relative_to(temporary).parts),
            reverse=True,
        ):
            source_directory = source / directory.relative_to(temporary)
            source_state = os.lstat(source_directory)
            if (
                stat.S_ISLNK(source_state.st_mode)
                or not stat.S_ISDIR(source_state.st_mode)
            ):
                raise ValueError(
                    "phase attempt directory changed during durable "
                    f"publication: {source_directory}"
                )
            os.chmod(directory, stat.S_IMODE(source_state.st_mode))
            _fsync_directory(directory)
        os.replace(temporary, published)
        _fsync_directory(durable_root)
        _remove_owned_scratch_tree_after_publication(source)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    published = published.resolve(strict=True)
    return (
        published,
        counters,
        _phase_payload_stat_inventory(published, artifacts),
    )


def _operator_trusted_adoption_selected(
    record: Mapping[str, Any],
) -> bool:
    policy = record.get("adoption_validation_policy")
    if policy is None:
        return False
    if policy != OPERATOR_TRUSTED_VALIDATION_POLICY:
        raise ValueError("checkpoint adoption validation policy is unsupported")
    prior = record.get("prior_adoption_attestation_path")
    if not isinstance(prior, str) or not prior.strip():
        raise ValueError(
            "operator-trusted checkpoint adoption lacks its prior attestation"
        )
    if record.get("payload_bytes_reauthenticated") is not False:
        raise ValueError(
            "operator-trusted checkpoint adoption has an invalid byte-audit claim"
        )
    return True


def _operator_trusted_legacy_migration_expectation(
    artifact: ValidatedPortableArtifact,
    *,
    expected_phase: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    """Open the small sealed typed expectation without reading payload bytes."""

    binding = artifact.phase_binding
    result = (
        binding.get("result_template")
        if isinstance(binding, Mapping)
        else None
    )
    migration = (
        result.get("legacy_terminal_migration_identity")
        if isinstance(result, Mapping)
        else None
    )
    if not isinstance(migration, Mapping):
        raise ValueError(
            f"operator-trusted {expected_phase} artifact lacks its sealed "
            "legacy migration identity"
        )
    migration_body = {
        key: copy.deepcopy(value)
        for key, value in migration.items()
        if key != "content_sha256"
    }
    typed = migration.get("typed_expectation")
    typed_identity = migration.get("typed_expectation_identity")
    if (
        migration.get("schema_version")
        != "legacy_terminal_typed_request_migration_identity_v1"
        or migration.get("phase") != expected_phase
        or migration.get("content_sha256") != _sha(migration_body)
        or not isinstance(typed, Mapping)
        or typed_identity != identity_sha256(typed)
        or migration.get("source_tree_mutated") is not False
        or migration.get("legacy_payload_copies_materialized") is not False
    ):
        raise ValueError(
            f"operator-trusted {expected_phase} migration identity is invalid"
        )
    if expected_phase == "input_preparation":
        required_true = (
            "byte_affecting_preprocessing_policy_matched",
            "configured_columns_reopened_exactly",
            "current_preparation_transform_replayed",
            "prepared_projection_recomputed",
            "unit_id_order_recomputed",
        )
    elif expected_phase == "embedding_cache":
        required_true = (
            "chunk_and_tokenization_capacity_nonbinding",
            "dense_array_shape_dtype_and_finiteness_reopened",
            "ordered_text_identity_recomputed",
            "prepared_projection_recomputed",
            "upstream_prepared_identity_reauthenticated",
            "word_chunk_registry_recomputed_exactly",
        )
    else:
        raise ValueError(
            "operator-trusted legacy phase projection supports only "
            "input preparation and embedding cache"
        )
    if any(migration.get(field) is not True for field in required_true):
        raise ValueError(
            f"operator-trusted {expected_phase} migration proof is incomplete"
        )
    return migration, typed, str(typed_identity)


def _operator_trusted_legacy_phase_projection_proof(
    *,
    artifact: ValidatedPortableArtifact,
    request: Mapping[str, Any],
    adopted_artifacts: Mapping[str, ValidatedPortableArtifact],
) -> Mapping[str, Any]:
    """Prove current phase inputs against a trusted V5 typed expectation.

    Historical V5 portable manifests used one whole-workflow configuration
    digest for every phase.  This proof permits that legacy digest alone to
    differ after checking every input that can affect preparation or cached
    embeddings.  It deliberately does not authenticate payload bytes again.
    """

    artifact_kind = str(artifact.manifest.get("artifact_kind") or "")
    phase_by_kind = {
        "prepared_cohort": "input_preparation",
        "embedding_cache": "embedding_cache",
    }
    phase = phase_by_kind.get(artifact_kind)
    if phase is None:
        raise ValueError(
            "operator-trusted legacy phase projection received an "
            "unsupported artifact kind"
        )
    compatibility_rows = request.get(
        "expected_checkpoint_compatibilities_by_phase"
    )
    expected_compatibility = (
        compatibility_rows.get(phase)
        if isinstance(compatibility_rows, Mapping)
        else None
    )
    if not isinstance(expected_compatibility, Mapping):
        raise ValueError(
            f"operator-trusted {phase} projection lacks request compatibility"
        )
    migration, typed, typed_identity = (
        _operator_trusted_legacy_migration_expectation(
            artifact,
            expected_phase=phase,
        )
    )
    dataset_path = Path(str(request.get("dataset_path") or ""))
    if (
        not dataset_path.is_absolute()
        or dataset_path.is_symlink()
        or not dataset_path.is_file()
    ):
        raise ValueError(
            "operator-trusted legacy phase projection lacks the current dataset"
        )
    dataset_size = int(dataset_path.stat().st_size)
    columns = {
        "unit_id": request.get("unit_id_column"),
        "text": request.get("text_column"),
        "treatment": request.get("treatment_column"),
        "outcome": request.get("outcome_column"),
    }
    preprocessing = {
        "empty_text_policy": request.get("empty_text_policy"),
        "repeated_character_policy": request.get(
            "repeated_character_policy"
        ),
        "repeated_character_threshold": request.get(
            "repeated_character_threshold"
        ),
        "source_text_temporally_valid_by_design": request.get(
            "source_text_temporally_valid_by_design"
        ),
    }
    prepared_dependencies = {
        "schema_version": "legacy_prepared_migration_expectation_v1",
        "dataset_sha256": request.get("source_sha256"),
        "dataset_size_bytes": dataset_size,
        "columns": columns,
        "preprocessing": preprocessing,
        "row_order_identity": expected_compatibility.get(
            "row_order_identity"
        ),
    }
    if request.get("outcome_type") != "binary":
        raise ValueError(
            "operator-trusted legacy preparation/cache reuse requires the "
            "binary v1 workflow"
        )

    if phase == "input_preparation":
        dependencies = prepared_dependencies
    else:
        upstream_ids = tuple(
            str(value)
            for value in artifact.manifest.get(
                "upstream_artifact_ids"
            )
            or ()
        )
        if len(upstream_ids) != 1:
            raise ValueError(
                "operator-trusted embedding cache must name exactly one "
                "prepared upstream artifact"
            )
        prepared_artifact = adopted_artifacts.get(upstream_ids[0])
        if (
            prepared_artifact is None
            or prepared_artifact.manifest.get("artifact_kind")
            != "prepared_cohort"
        ):
            raise ValueError(
                "operator-trusted embedding cache lacks its exact adopted "
                "prepared artifact"
            )
        (
            _prepared_migration,
            prepared_typed,
            prepared_typed_identity,
        ) = _operator_trusted_legacy_migration_expectation(
            prepared_artifact,
            expected_phase="input_preparation",
        )
        if any(
            prepared_typed.get(key) != value
            for key, value in prepared_dependencies.items()
        ):
            raise ValueError(
                "operator-trusted prepared artifact differs from the current "
                "preparation inputs"
            )
        raw_encoder = request.get("embedding_encoder")
        if not isinstance(raw_encoder, Mapping):
            raise ValueError(
                "operator-trusted embedding-cache projection lacks its "
                "encoder configuration"
            )
        chunk_configuration = {
            "chunk_size_words": request.get(
                "embedding_chunk_size_words"
            ),
            "chunk_overlap_words": request.get(
                "embedding_chunk_overlap_words"
            ),
            "max_chunks": request.get("embedding_max_chunks"),
            "chunk_selection": request.get(
                "embedding_chunk_selection"
            ),
            "normalize_embeddings": request.get(
                "embedding_normalize"
            ),
            "max_seq_length": request.get(
                "embedding_max_seq_length"
            ),
            **copy.deepcopy(dict(raw_encoder)),
        }
        dependencies = {
            "schema_version": (
                "legacy_embedding_cache_migration_expectation_v2"
            ),
            "prepared_expectation_identity": prepared_typed_identity,
            "embedding_model_name": request.get(
                "embedding_model_name"
            ),
            "embedding_model_tree_sha256": request.get(
                "embedding_model_builder_tree_sha256"
            ),
            "chunk_configuration": chunk_configuration,
        }
        if (
            migration.get("upstream_prepared_artifact_id")
            != upstream_ids[0]
        ):
            raise ValueError(
                "operator-trusted embedding-cache migration names the wrong "
                "prepared artifact"
            )

    if any(typed.get(key) != value for key, value in dependencies.items()):
        raise ValueError(
            f"operator-trusted {phase} artifact differs from the current "
            "phase-specific scientific inputs"
        )
    observed_compatibility = artifact.manifest.get("compatibility")
    if not isinstance(observed_compatibility, Mapping):
        raise ValueError(
            "operator-trusted legacy phase artifact lacks compatibility"
        )
    body = {
        "schema_version": (
            OPERATOR_TRUSTED_LEGACY_PHASE_PROJECTION_SCHEMA
        ),
        "status": "exact_phase_dependencies_match",
        "phase": phase,
        "artifact_id": artifact.artifact_id,
        "artifact_kind": artifact_kind,
        "migration_identity_sha256": migration.get(
            "content_sha256"
        ),
        "typed_expectation_identity": typed_identity,
        "phase_dependency_projection": dependencies,
        "superseded_legacy_compatibility_fields": [
            "configuration_identity"
        ],
        "observed_legacy_configuration_identity": (
            observed_compatibility.get("configuration_identity")
        ),
        "requested_configuration_identity": (
            expected_compatibility.get("configuration_identity")
        ),
        "payload_bytes_reauthenticated": False,
        "global_release_certified": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _validate_operator_trusted_legacy_phase_projection_record(
    *,
    artifact: ValidatedPortableArtifact,
    requested: Mapping[str, Any],
    record: Mapping[str, Any],
) -> bool:
    proof = record.get(
        "legacy_phase_compatibility_projection_proof"
    )
    if proof is None:
        return False
    if not isinstance(proof, Mapping):
        raise ValueError(
            "operator-trusted legacy phase projection proof is invalid"
        )
    body = {
        key: copy.deepcopy(value)
        for key, value in proof.items()
        if key != "content_sha256"
    }
    expected_keys = {
        "schema_version",
        "status",
        "phase",
        "artifact_id",
        "artifact_kind",
        "migration_identity_sha256",
        "typed_expectation_identity",
        "phase_dependency_projection",
        "superseded_legacy_compatibility_fields",
        "observed_legacy_configuration_identity",
        "requested_configuration_identity",
        "payload_bytes_reauthenticated",
        "global_release_certified",
        "content_sha256",
    }
    observed_compatibility = artifact.manifest.get("compatibility")
    phase = CHECKPOINT_COMPATIBILITY_PHASE_BY_ARTIFACT_KIND.get(
        str(artifact.manifest.get("artifact_kind") or "")
    )
    if (
        set(proof) != expected_keys
        or proof.get("schema_version")
        != OPERATOR_TRUSTED_LEGACY_PHASE_PROJECTION_SCHEMA
        or proof.get("status") != "exact_phase_dependencies_match"
        or proof.get("phase") != phase
        or proof.get("artifact_id") != artifact.artifact_id
        or proof.get("artifact_kind")
        != artifact.manifest.get("artifact_kind")
        or proof.get("superseded_legacy_compatibility_fields")
        != ["configuration_identity"]
        or proof.get("observed_legacy_configuration_identity")
        != (
            observed_compatibility.get("configuration_identity")
            if isinstance(observed_compatibility, Mapping)
            else None
        )
        or proof.get("requested_configuration_identity")
        != requested.get("configuration_identity")
        or proof.get("payload_bytes_reauthenticated") is not False
        or proof.get("global_release_certified") is not False
        or proof.get("content_sha256") != _sha(body)
    ):
        raise ValueError(
            "operator-trusted legacy phase projection proof failed validation"
        )
    return True


def _adopted_compatibility_matches_request(
    *,
    artifact: ValidatedPortableArtifact,
    expected: Mapping[str, Any],
    record: Mapping[str, Any],
) -> bool:
    observed = dict(artifact.manifest["compatibility"])
    requested = dict(expected)
    if observed == requested:
        return True
    if not _operator_trusted_adoption_selected(record):
        return False
    # The trusted artifact was produced by the earlier frozen source snapshot.
    # Producer-code identity may differ, while a sealed legacy phase projection
    # may additionally prove that the historical whole-workflow configuration
    # digest and the downstream-only Stage 2 model name are irrelevant to
    # preparation/cache bytes. Every data, split, row, seed, prompt, runtime,
    # embedding, HTR, and tokenizer axis remains exact.
    observed.pop("producer_code_identity", None)
    requested.pop("producer_code_identity", None)
    legacy_projection_validated = False
    if observed.get("configuration_identity") != requested.get(
        "configuration_identity"
    ):
        if not _validate_operator_trusted_legacy_phase_projection_record(
            artifact=artifact,
            requested=requested,
            record=record,
        ):
            return False
        legacy_projection_validated = True
        observed.pop("configuration_identity", None)
        requested.pop("configuration_identity", None)
    observed_models = observed.get("model_identities")
    requested_models = requested.get("model_identities")
    if observed_models != requested_models:
        if (
            not legacy_projection_validated
            and not _validate_operator_trusted_legacy_phase_projection_record(
                artifact=artifact,
                requested=requested,
                record=record,
            )
        ):
            return False
        if not isinstance(observed_models, Mapping) or not isinstance(
            requested_models,
            Mapping,
        ):
            return False
        observed_upstream_models = dict(observed_models)
        requested_upstream_models = dict(requested_models)
        observed_upstream_models.pop("stage2_model_name", None)
        requested_upstream_models.pop("stage2_model_name", None)
        if observed_upstream_models != requested_upstream_models:
            return False
        observed["model_identities"] = observed_upstream_models
        requested["model_identities"] = requested_upstream_models
    return observed == requested


def _open_adopted_artifact(
    *,
    record: Mapping[str, Any],
    locator: Path,
    expected_compatibility: Mapping[str, Any],
    payload_authentication_cache: MutableMapping[
        str, tuple[tuple[int, ...], str, int]
    ]
    | None = None,
) -> ValidatedPortableArtifact:
    kind = str(record.get("artifact_kind") or "")
    upstream = tuple(
        str(value)
        for value in record.get("upstream_artifact_ids") or ()
    )
    if _operator_trusted_adoption_selected(record):
        trusted = validate_operator_trusted_portable_artifact(
            source=locator,
            prior_attestation_path=Path(
                str(record["prior_adoption_attestation_path"])
            ),
            expected_kind=kind,
            expected_upstream_artifact_ids=upstream,
        )
        artifact = trusted.artifact
    else:
        artifact = validate_portable_artifact(
            locator,
            expected_kind=kind,
            expected_compatibility_key=ArtifactCompatibility(
                **dict(expected_compatibility)
            ).key,
            expected_upstream_artifact_ids=upstream,
            payload_authentication_cache=payload_authentication_cache,
        )
    if not _adopted_compatibility_matches_request(
        artifact=artifact,
        expected=expected_compatibility,
        record=record,
    ):
        raise ValueError(
            "adopted checkpoint compatibility differs from its request"
        )
    return artifact


def _validate_adoption_attestation_for_record(
    *,
    attestation_path: Path,
    artifact: ValidatedPortableArtifact,
    record: Mapping[str, Any],
    consumer_request_sha256: str,
) -> Mapping[str, Any]:
    if _operator_trusted_adoption_selected(record):
        return validate_operator_trusted_checkpoint_adoption(
            attestation_path=attestation_path,
            source=artifact.root,
            prior_attestation_path=Path(
                str(record["prior_adoption_attestation_path"])
            ),
            consumer_request_sha256=consumer_request_sha256,
            expected_kind=str(record["artifact_kind"]),
            expected_upstream_artifact_ids=tuple(
                str(value)
                for value in record.get("upstream_artifact_ids") or ()
            ),
        )
    return validate_checkpoint_adoption(
        attestation_path=attestation_path,
        artifact=artifact,
        consumer_request_sha256=consumer_request_sha256,
    )


def _validate_adopted_phase_manifest_from_paths(
    *,
    work_root: Path,
    phase: str,
    request_sha256: str,
    value: Mapping[str, Any],
    authenticated_adoptions: Mapping[str, ValidatedPortableArtifact] | None = None,
) -> dict[str, Any]:
    """Freshly validate and materialize one immutable adopted-phase reference."""

    required = {
        "schema_version",
        "phase",
        "status",
        "request_sha256",
        "artifact_id",
        "artifact_kind",
        "compatibility_key",
        "artifact_locator",
        "adoption_attestation_path",
        "upstream_artifact_ids",
        "content_sha256",
    }
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        set(value) != required
        or value.get("schema_version") != WORKFLOW_ADOPTED_PHASE_MANIFEST_SCHEMA
        or value.get("phase") != phase
        or value.get("status") != "complete"
        or value.get("request_sha256") != request_sha256
        or value.get("content_sha256") != _sha(body)
        or ADOPTABLE_PHASE_BY_ARTIFACT_KIND.get(str(value.get("artifact_kind", ""))) != phase
    ):
        raise ValueError(f"adopted phase manifest failed validation: {phase}")

    request_path = work_root / "immutable_run_request.json"
    request = _read_json_object(
        request_path,
        label="immutable workflow request for adopted phase",
    )
    request_body = {key: item for key, item in request.items() if key != "request_sha256"}
    if request.get("request_sha256") != request_sha256 or _sha(request_body) != request_sha256:
        raise ValueError("immutable workflow request changed for adopted phase")
    records = request.get("requested_checkpoint_adoptions")
    locators = request.get("checkpoint_adoption_locators")
    if (
        not isinstance(records, list)
        or not isinstance(locators, list)
        or len(records) != len(locators)
    ):
        raise ValueError("immutable request has invalid checkpoint adoptions")
    matching = [
        (record, locator)
        for record, locator in zip(records, locators)
        if isinstance(record, Mapping) and record.get("artifact_id") == value.get("artifact_id")
    ]
    if len(matching) != 1:
        raise ValueError("adopted phase is absent or duplicated in its request")
    expected, locator = matching[0]
    if (
        expected.get("artifact_kind") != value.get("artifact_kind")
        or expected.get("compatibility_key") != value.get("compatibility_key")
        or expected.get("substituted_phase") != phase
        or expected.get("upstream_artifact_ids") != value.get("upstream_artifact_ids")
        or str(locator) != value.get("artifact_locator")
    ):
        raise ValueError("adopted phase differs from its immutable request")

    artifact_id = str(value["artifact_id"])
    artifact = None if authenticated_adoptions is None else authenticated_adoptions.get(artifact_id)
    compatibility_rows = request.get(
        "expected_checkpoint_compatibilities_by_phase"
    )
    expected_compatibility = (
        compatibility_rows.get(phase)
        if isinstance(compatibility_rows, Mapping)
        else None
    )
    if not isinstance(expected_compatibility, Mapping):
        raise ValueError("adopted phase compatibility is absent from its request")
    if artifact is None:
        artifact = _open_adopted_artifact(
            record=expected,
            locator=Path(str(locator)),
            expected_compatibility=expected_compatibility,
        )
    else:
        assert_validated_artifact_unchanged(artifact)
    if artifact.artifact_id != artifact_id:
        raise ValueError("adopted phase artifact ID changed")
    if (
        not _adopted_compatibility_matches_request(
            artifact=artifact,
            expected=expected_compatibility,
            record=expected,
        )
    ):
        raise ValueError("adopted phase compatibility differs from its request")
    if phase == "stage1_preflight":
        _require_adopted_compact_preflight_parquet_compression(
            artifact,
            expected=request.get(
                "cluster_preflight_parquet_compression"
            ),
        )
    expected_attestation = (
        work_root / "checkpoint_adoptions" / f"{artifact_id}.adoption.json"
    ).resolve(strict=True)
    supplied_attestation = Path(str(value["adoption_attestation_path"])).resolve(strict=True)
    if supplied_attestation != expected_attestation:
        raise ValueError("adopted phase attestation locator was substituted")
    _validate_adoption_attestation_for_record(
        attestation_path=expected_attestation,
        artifact=artifact,
        record=expected,
        consumer_request_sha256=request_sha256,
    )
    materialized = materialize_portable_phase(
        artifact,
        expected_phase=phase,
    )
    return {
        "schema_version": WORKFLOW_ADOPTED_PHASE_MANIFEST_SCHEMA,
        "phase": phase,
        "status": "complete",
        "request_sha256": request_sha256,
        "attempt_dir": materialized["attempt_dir"],
        "result": materialized["result"],
        "artifacts": materialized["artifacts"],
        "content_sha256": value["content_sha256"],
        "adopted_checkpoint": {
            "artifact_id": artifact.artifact_id,
            "artifact_kind": artifact.manifest["artifact_kind"],
            "compatibility_key": artifact.compatibility_key,
            "upstream_artifact_ids": list(artifact.manifest["upstream_artifact_ids"]),
            "adoption_attestation_path": str(expected_attestation),
            "fresh_full_byte_validation": (
                authenticated_adoptions is None
                and not _operator_trusted_adoption_selected(expected)
            ),
            "operator_trusted_prior_full_byte_attestation": (
                _operator_trusted_adoption_selected(expected)
            ),
            "payload_bytes_reauthenticated": (
                not _operator_trusted_adoption_selected(expected)
            ),
        },
    }


def _validate_phase_manifest_from_paths(
    *,
    work_root: Path,
    phase: str,
    request_sha256: str,
    authenticated_adoptions: Mapping[str, ValidatedPortableArtifact] | None = None,
) -> dict[str, Any]:
    """Validate one completed phase without a live workflow runner."""

    manifest_path = work_root / "phases" / phase / "complete_manifest.json"
    value = _read_json_object(manifest_path, label=f"{phase} phase manifest")
    if value.get("schema_version") == WORKFLOW_ADOPTED_PHASE_MANIFEST_SCHEMA:
        return _validate_adopted_phase_manifest_from_paths(
            work_root=work_root,
            phase=phase,
            request_sha256=request_sha256,
            value=value,
            authenticated_adoptions=authenticated_adoptions,
        )
    if set(value) != {
        "schema_version",
        "phase",
        "status",
        "request_sha256",
        "attempt_dir",
        "result",
        "artifacts",
        "content_sha256",
    }:
        raise ValueError(f"completed phase manifest is not closed: {phase}")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("schema_version") != WORKFLOW_PHASE_MANIFEST_SCHEMA
        or value.get("phase") != phase
        or value.get("request_sha256") != request_sha256
        or value.get("content_sha256") != _sha(body)
        or value.get("status") != "complete"
        or not isinstance(value.get("result"), Mapping)
        or not isinstance(value.get("artifacts"), list)
    ):
        raise ValueError(f"completed phase manifest failed validation: {phase}")
    attempt = Path(str(value.get("attempt_dir", "")))
    expected_phase_root = (work_root / "phases" / phase).resolve(strict=True)
    if (
        not attempt.is_absolute()
        or attempt.is_symlink()
        or not attempt.is_dir()
        or attempt.resolve(strict=True).parent != expected_phase_root
        or not attempt.name.startswith("attempt_")
    ):
        raise ValueError(f"completed phase attempt path is invalid: {phase}")
    observed = _attempt_tree_artifacts(attempt)
    if value["artifacts"] != observed:
        raise ValueError(f"completed phase attempt tree changed: {phase}")
    terminal_files = value["result"].get("terminal_files", [])
    if (
        not isinstance(terminal_files, list)
        or any(not isinstance(path, str) for path in terminal_files)
        or len(terminal_files) != len(set(terminal_files))
    ):
        raise ValueError(f"completed phase terminal-file registry is invalid: {phase}")
    registered = {row["path"] for row in observed}
    for raw in terminal_files:
        terminal = Path(raw)
        if not terminal.is_absolute() or str(terminal.resolve(strict=True)) not in registered:
            raise ValueError(
                f"completed phase terminal file escaped its sealed attempt: {terminal}"
            )
    return value


def _validate_real_stage2_terminal_artifacts(
    *,
    request: Mapping[str, Any],
    phase_records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Delegate real artifact validation to the path-only closed validator."""

    from .production_terminal_artifact_validation import (
        validate_real_stage2_terminal_artifacts,
    )

    return validate_real_stage2_terminal_artifacts(
        request=request,
        phase_records=phase_records,
    )


def validate_completed_workflow_prefix(
    *,
    work_root: Path | str,
    expected_request_sha256: str,
    expected_phases: Sequence[str],
) -> dict[str, Any]:
    """Fresh path-only validation for every phase completed before terminal seal."""

    root = Path(work_root).resolve(strict=True)
    request_path = root / "immutable_run_request.json"
    request = _read_json_object(request_path, label="immutable workflow request")
    request_body = {key: value for key, value in request.items() if key != "request_sha256"}
    phases = tuple(str(value) for value in expected_phases)
    if (
        request.get("request_sha256") != expected_request_sha256
        or request.get("request_sha256") != _sha(request_body)
        or tuple(request.get("phase_sequence") or ()) != (*phases, "terminal_validation")
    ):
        raise ValueError("immutable workflow request failed fresh validation")
    _revalidate_request_bound_external_inputs(request)
    phase_records = [
        _validate_phase_manifest_from_paths(
            work_root=root,
            phase=phase,
            request_sha256=expected_request_sha256,
        )
        for phase in phases
    ]
    source_snapshot = request.get("source_snapshot")
    if source_snapshot is not None:
        if not isinstance(source_snapshot, Mapping):
            raise ValueError("source snapshot identity is invalid")
        from .production_source_snapshot import validate_production_source_snapshot

        validated_snapshot = validate_production_source_snapshot(
            Path(str(source_snapshot.get("root", "")))
        ).as_dict()
        if validated_snapshot != dict(source_snapshot):
            raise ValueError("source snapshot changed before terminal validation")
    handoff = next(
        (row for row in phase_records if row.get("phase") == "handoff_validation"),
        None,
    )
    handoff_result = None if handoff is None else handoff["result"].get("fresh_process_validation")
    reported_handoff_validated = bool(
        isinstance(handoff_result, Mapping)
        and handoff_result.get("schema_version") == "production_stage1_fresh_handoff_validation_v1"
        and handoff_result.get("status") == "accepted"
        and handoff_result.get("remote_clients_constructed") is False
        and handoff_result.get("remote_calls_made") is False
    )
    stage2_terminal = _validate_real_stage2_terminal_artifacts(
        request=request,
        phase_records=phase_records,
    )
    stage1_terminal = stage2_terminal.get("stage1_handoff_validation")
    if not isinstance(stage1_terminal, Mapping):
        from .production_terminal_artifact_validation import (
            validate_real_stage1_handoff,
        )

        stage1_terminal = validate_real_stage1_handoff(
            request=request,
            phase_records=phase_records,
        )
    handoff_validated = bool(
        stage1_terminal.get("real_stage1_handoff_detected") is True or reported_handoff_validated
    )
    body = {
        "schema_version": WORKFLOW_TERMINAL_VALIDATION_SCHEMA,
        "status": "accepted",
        "request_path": str(request_path.resolve(strict=True)),
        "request_sha256": expected_request_sha256,
        "validated_phases": list(phases),
        "validated_phase_manifest_sha256": {
            row["phase"]: stable_file_sha256(
                root / "phases" / row["phase"] / "complete_manifest.json"
            )[0]
            for row in phase_records
        },
        "validated_artifact_count": sum(len(row["artifacts"]) for row in phase_records),
        "stage1_handoff_validated_in_fresh_process": handoff_validated,
        "stage1_terminal_validation": stage1_terminal,
        "stage2_terminal_validation": stage2_terminal,
        "source_snapshot": source_snapshot,
        "request_bound_external_inputs_revalidated": True,
        "live_runner_objects_received": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _stable_path_identity(
    path: Path,
    *,
    reuse_process_authenticated_tree: bool = False,
) -> Mapping[str, Any]:
    """Bind one file or directory tree without trusting names alone.

    Callers may reuse a PID-scoped content identity after one full provenance
    authentication while every logical check still compares the complete
    filesystem inventory. Callers that do not opt in retain full byte-tree
    reauthentication.
    """

    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError(f"identity-bound path cannot be a symlink: {supplied}")
    resolved = supplied.resolve(strict=True)
    if resolved.is_file():
        digest, size = stable_file_sha256(resolved)
        return {
            "kind": "file",
            "path": str(resolved),
            "sha256": digest,
            "size_bytes": size,
        }
    if not resolved.is_dir():
        raise ValueError(f"identity-bound path is not a file or directory: {resolved}")
    if reuse_process_authenticated_tree:
        return authenticate_directory_tree(resolved).workflow_path_identity()
    inventory: list[dict[str, Any]] = []
    for candidate in sorted(resolved.rglob("*")):
        if candidate.is_symlink():
            raise ValueError(f"identity-bound tree cannot contain symlinks: {candidate}")
        if not candidate.is_file():
            continue
        digest, size = stable_file_sha256(candidate)
        inventory.append(
            {
                "relative_path": candidate.relative_to(resolved).as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    if not inventory:
        raise ValueError(f"identity-bound directory has no files: {resolved}")
    return {
        "kind": "directory",
        "path": str(resolved),
        "file_count": len(inventory),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in inventory),
        "tree_sha256": _sha(inventory),
        "files": inventory,
    }


def _embedding_builder_tree_sha256(
    *,
    root: Path,
    workflow_tree_identity: Mapping[str, Any],
) -> str:
    """Project the current model tree into the cache-builder identity schema."""

    supplied = Path(root)
    if supplied.is_symlink():
        raise ValueError("embedding model tree cannot be symlinked")
    resolved = supplied.resolve(strict=True)
    files = workflow_tree_identity.get("files")
    if (
        workflow_tree_identity.get("kind") != "directory"
        or not isinstance(files, list)
        or not files
    ):
        raise ValueError("embedding model workflow identity is not one directory tree")
    directories: list[str] = []
    observed_files: set[str] = set()
    for candidate in sorted(resolved.rglob("*")):
        state = os.lstat(candidate)
        if stat.S_ISLNK(state.st_mode):
            raise ValueError("embedding model tree cannot contain symlinks")
        relative = candidate.relative_to(resolved).as_posix()
        if stat.S_ISDIR(state.st_mode):
            directories.append(relative)
        elif stat.S_ISREG(state.st_mode):
            observed_files.add(relative)
        else:
            raise ValueError("embedding model tree contains a special entry")
    registered_files = {str(row.get("relative_path")) for row in files if isinstance(row, Mapping)}
    if len(registered_files) != len(files) or observed_files != registered_files:
        raise RuntimeError("embedding model tree changed during identity projection")
    builder_body = {
        "directories": directories,
        "files": [
            {
                "path": str(row["relative_path"]),
                "sha256": str(row["sha256"]),
                "size_bytes": int(row["size_bytes"]),
            }
            for row in files
        ],
    }
    return _sha(builder_body)


def _stage1_bundle_model_tree_sha256_from_workflow_identity(
    workflow_tree_identity: Mapping[str, Any],
) -> str:
    """Project an already authenticated model tree into the Stage 1 digest.

    ``stage1_upstream_gate_backend._directory_tree_sha256`` hashes rows named
    ``relative_path``, ``size``, and ``sha256``.  The immutable workflow
    request has already authenticated the same bytes under the workflow tree
    schema, whose size field is named ``size_bytes``.  Re-projecting that
    closed inventory avoids a second read of every model byte during a
    reusable-preflight reopen.
    """

    files = workflow_tree_identity.get("files")
    if (
        workflow_tree_identity.get("kind") != "directory"
        or not isinstance(files, list)
        or not files
        or int(workflow_tree_identity.get("file_count", -1))
        != len(files)
    ):
        raise ValueError(
            "workflow HTR model identity is not one closed directory tree"
        )
    workflow_rows: list[dict[str, Any]] = []
    stage1_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in files:
        if (
            not isinstance(raw, Mapping)
            or set(raw)
            != {"relative_path", "sha256", "size_bytes"}
        ):
            raise ValueError(
                "workflow HTR model inventory row is malformed"
            )
        relative = str(raw["relative_path"])
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in seen
        ):
            raise ValueError(
                "workflow HTR model inventory path is invalid or duplicated"
            )
        seen.add(relative)
        size = int(raw["size_bytes"])
        digest = str(raw["sha256"])
        if (
            size < 0
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "workflow HTR model inventory content identity is invalid"
            )
        workflow_rows.append(
            {
                "relative_path": relative,
                "sha256": digest,
                "size_bytes": size,
            }
        )
        stage1_rows.append(
            {
                "relative_path": relative,
                "size": size,
                "sha256": digest,
            }
        )
    workflow_rows.sort(key=lambda row: row["relative_path"])
    stage1_rows.sort(key=lambda row: row["relative_path"])
    if (
        sum(int(row["size_bytes"]) for row in workflow_rows)
        != int(workflow_tree_identity.get("total_size_bytes", -1))
        or _sha(workflow_rows)
        != workflow_tree_identity.get("tree_sha256")
    ):
        raise ValueError(
            "workflow HTR model inventory does not match its terminal identity"
        )
    return _sha(stage1_rows)


def _revalidate_request_bound_external_inputs(
    request: Mapping[str, Any],
    *,
    authenticated_adoptions: Mapping[str, ValidatedPortableArtifact] | None = None,
    identity_memo: _ScientificIdentityMemo | None = None,
) -> None:
    """Reopen every external input whose bytes were bound into the run request."""

    authenticated_files: dict[Path, tuple[str, int]] = {}

    def authenticate_file(path: Path) -> tuple[str, int]:
        resolved = Path(path).resolve(strict=True)
        cached = authenticated_files.get(resolved)
        if cached is not None:
            return cached
        observed = (
            stable_file_sha256(resolved)
            if identity_memo is None
            else _memoized_scientific_file_digest(
                resolved,
                identity_memo=identity_memo,
            )
        )
        normalized = (str(observed[0]), int(observed[1]))
        authenticated_files[resolved] = normalized
        return normalized

    expected_granular_plan = _derive_expected_granular_checkpoint_plan(
        outer_folds=int(request["outer_folds"]),
        initial_training_partitions=int(
            request["initial_training_partitions"]
        ),
        review_rounds=int(request["review_rounds"]),
    )
    if (
        _validate_expected_granular_checkpoint_plan(
            request.get("expected_granular_checkpoint_plan")
        )
        != expected_granular_plan
    ):
        raise ValueError(
            "immutable workflow request granular plan changed"
        )

    def require_file_hash(
        *,
        path_field: str,
        sha_field: str,
        label: str,
    ) -> None:
        raw_path = request.get(path_field)
        expected = request.get(sha_field)
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ValueError(f"immutable workflow request lacks {label} identity")
        observed, _size = authenticate_file(Path(raw_path))
        if observed != expected:
            raise RuntimeError(f"{label} changed after workflow initialization")

    require_file_hash(
        path_field="dataset_path",
        sha_field="source_sha256",
        label="source dataset",
    )
    require_file_hash(
        path_field="stage1_profile_path",
        sha_field="stage1_profile_sha256",
        label="Stage 1 profile",
    )
    require_file_hash(
        path_field="query_profile_path",
        sha_field="query_profile_sha256",
        label="neural-query profile",
    )
    if request.get("scientific_spec_path") is not None:
        require_file_hash(
            path_field="scientific_spec_path",
            sha_field="scientific_spec_source_sha256",
            label="scientific workflow spec",
        )
    if request.get("deployment_profile_path") is not None:
        require_file_hash(
            path_field="deployment_profile_path",
            sha_field="deployment_profile_source_sha256",
            label="deployment profile",
        )
    adoption_records = request.get("requested_checkpoint_adoptions") or []
    adoption_locators = request.get("checkpoint_adoption_locators") or []
    if (
        not isinstance(adoption_records, list)
        or not isinstance(adoption_locators, list)
        or len(adoption_records) != len(adoption_locators)
    ):
        raise ValueError("immutable workflow request has invalid checkpoint adoptions")
    compatibility_rows = request.get(
        "expected_checkpoint_compatibilities_by_phase"
    )
    if not isinstance(compatibility_rows, Mapping):
        raise ValueError(
            "immutable workflow request lacks checkpoint compatibilities"
        )
    reopened_adoptions: dict[str, ValidatedPortableArtifact] = {}
    for locator, expected in zip(adoption_locators, adoption_records):
        if not isinstance(expected, Mapping):
            raise ValueError("checkpoint adoption request is invalid")
        artifact_id = str(expected.get("artifact_id", ""))
        artifact = (
            None if authenticated_adoptions is None else authenticated_adoptions.get(artifact_id)
        )
        if artifact is None:
            compatibility_phase = str(
                expected.get("compatibility_phase") or ""
            )
            expected_compatibility = compatibility_rows.get(
                compatibility_phase
            )
            if not isinstance(expected_compatibility, Mapping):
                raise ValueError(
                    "checkpoint adoption compatibility phase is invalid"
                )
            artifact = _open_adopted_artifact(
                record=expected,
                locator=Path(str(locator)),
                expected_compatibility=expected_compatibility,
            )
        else:
            assert_validated_artifact_unchanged(artifact)
        reopened_adoptions[artifact.artifact_id] = artifact
        if (
            artifact.artifact_id != expected.get("artifact_id")
            or artifact.manifest["artifact_kind"] != expected.get("artifact_kind")
            or artifact.compatibility_key != expected.get("compatibility_key")
            or list(artifact.manifest["upstream_artifact_ids"])
            != expected.get("upstream_artifact_ids")
            or expected.get("compatibility_phase")
            != CHECKPOINT_COMPATIBILITY_PHASE_BY_ARTIFACT_KIND.get(
                str(artifact.manifest["artifact_kind"])
            )
            or expected.get("artifact_metadata")
            != dict(artifact.artifact_metadata)
        ):
            raise RuntimeError("adopted checkpoint changed after workflow initialization")
    for expected in adoption_records:
        if (
            not isinstance(expected, Mapping)
            or not _operator_trusted_adoption_selected(expected)
        ):
            continue
        artifact = reopened_adoptions.get(
            str(expected.get("artifact_id") or "")
        )
        if artifact is None:
            raise RuntimeError(
                "operator-trusted legacy phase projection lost its artifact"
            )
        recomputed_projection = (
            _operator_trusted_legacy_phase_projection_proof(
                artifact=artifact,
                request=request,
                adopted_artifacts=reopened_adoptions,
            )
        )
        if (
            expected.get(
                "legacy_phase_compatibility_projection_proof"
            )
            != recomputed_projection
        ):
            raise RuntimeError(
                "operator-trusted legacy phase projection changed after "
                "workflow initialization"
            )

    legacy_migrations = request.get("legacy_checkpoint_migration_sources")
    if not isinstance(legacy_migrations, list):
        raise ValueError("immutable workflow request lacks its legacy-migration source ledger")
    adopted_ids = {
        str(row.get("artifact_id")) for row in adoption_records if isinstance(row, Mapping)
    }
    for row in legacy_migrations:
        required = {
            "phase",
            "legacy_manifest_path",
            "legacy_manifest_sha256",
            "legacy_manifest_size_bytes",
            "legacy_manifest_content_sha256",
            "legacy_request_sha256",
            "typed_expectation_identity",
            "migrated_artifact_id",
            "portable_control_root",
            "source_tree_mutated",
            "payload_copies_materialized",
        }
        if (
            not isinstance(row, Mapping)
            or set(row) != required
            or row.get("phase") not in {"input_preparation", "embedding_cache"}
            or row.get("migrated_artifact_id") not in adopted_ids
            or row.get("source_tree_mutated") is not False
            or row.get("payload_copies_materialized") is not False
        ):
            raise ValueError("immutable legacy-migration source ledger is invalid")
        manifest_path = Path(str(row["legacy_manifest_path"])).resolve(strict=True)
        observed_sha, observed_size = authenticate_file(manifest_path)
        if (
            observed_sha != row["legacy_manifest_sha256"]
            or observed_size != row["legacy_manifest_size_bytes"]
        ):
            raise RuntimeError("legacy terminal manifest changed after workflow initialization")
        from .legacy_checkpoint_migration import (
            validate_legacy_terminal_phase_manifest,
        )

        validated_legacy = validate_legacy_terminal_phase_manifest(
            manifest_path,
            expected_phase=str(row["phase"]),
        )
        if (
            validated_legacy["manifest"]["content_sha256"] != row["legacy_manifest_content_sha256"]
            or validated_legacy["manifest"]["request_sha256"] != row["legacy_request_sha256"]
        ):
            raise RuntimeError("legacy terminal manifest identity changed after initialization")

    legacy_preflight_identity = request.get("legacy_preflight_candidate_identity")
    if legacy_preflight_identity is not None:
        if not isinstance(legacy_preflight_identity, Mapping):
            raise ValueError("immutable legacy preflight identity is invalid")
        path = Path(str(legacy_preflight_identity.get("manifest_path", ""))).resolve(strict=True)
        observed_sha, observed_size = authenticate_file(path)
        (
            observed,
            observed_kind,
            observed_state_manifest,
        ) = _validate_preflight_candidate_selector(
            path,
        )
        if (
            observed_sha != legacy_preflight_identity.get("manifest_sha256")
            or observed_size != legacy_preflight_identity.get("manifest_size_bytes")
            or observed["manifest"]["content_sha256"]
            != legacy_preflight_identity.get("manifest_content_sha256")
            or observed_kind
            != legacy_preflight_identity.get("candidate_kind")
            or (
                None
                if observed_state_manifest is None
                else str(observed_state_manifest)
            )
            != legacy_preflight_identity.get(
                "state_bundle_manifest_path"
            )
            or (
                None
                if observed_state_manifest is None
                else stable_file_sha256(observed_state_manifest)[0]
            )
            != legacy_preflight_identity.get(
                "state_bundle_manifest_sha256"
            )
            or observed.get("prepared_context_manifest_path")
            != legacy_preflight_identity.get(
                "prepared_context_manifest_path"
            )
            or (
                None
                if observed.get("prepared_context_manifest_path")
                is None
                else stable_file_sha256(
                    Path(
                        str(
                            observed[
                                "prepared_context_manifest_path"
                            ]
                        )
                    )
                )[0]
            )
            != legacy_preflight_identity.get(
                "prepared_context_manifest_sha256"
            )
        ):
            raise RuntimeError(
                "selected preflight candidate changed after workflow initialization"
            )

    cache_inputs = request.get("embedding_cache_import_inputs")
    expected_model_policy = AUTHENTICATED_DIRECTORY_TREE_POLICY
    if request.get("embedding_model_revalidation_policy") != expected_model_policy:
        raise ValueError(
            "immutable workflow request has an invalid embedding-model " "revalidation policy"
        )

    model_tree_fields = [
        ("embedding_model_tree", "embedding model tree"),
        ("htr_model_tree", "HTR model tree"),
    ]
    if request.get("stage2_tokenizer_tree") is not None:
        model_tree_fields.append(("stage2_tokenizer_tree", "Stage 2 tokenizer tree"))
    elif request.get("stage1_only") is not True:
        raise ValueError("immutable full-workflow request lacks its Stage 2 tokenizer identity")
    for field, label in model_tree_fields:
        expected = request.get(field)
        if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
            raise ValueError(f"immutable workflow request lacks {label} identity")
        try:
            observed_model_tree = _stable_path_identity(
                Path(str(expected["path"])),
                reuse_process_authenticated_tree=(
                    field == "embedding_model_tree"
                ),
            )
        except AuthenticatedDirectoryTreeDriftError as exc:
            raise RuntimeError(
                f"{label} changed after workflow initialization"
            ) from exc
        if observed_model_tree != dict(expected):
            raise RuntimeError(f"{label} changed after workflow initialization")
    expected_embedding_tree = request.get("embedding_model_tree")
    expected_builder_tree_sha256 = request.get("embedding_model_builder_tree_sha256")
    if (
        not isinstance(expected_embedding_tree, Mapping)
        or not isinstance(expected_builder_tree_sha256, str)
        or _embedding_builder_tree_sha256(
            root=Path(str(expected_embedding_tree.get("path", ""))),
            workflow_tree_identity=expected_embedding_tree,
        )
        != expected_builder_tree_sha256
    ):
        raise RuntimeError("embedding model builder identity changed after workflow initialization")

    if cache_inputs is not None:
        if not isinstance(cache_inputs, Mapping) or set(cache_inputs) != {
            "cache",
            "prepared_cohort",
            "preparation_manifest",
        }:
            raise ValueError("immutable workflow request has an invalid cache-import identity")
        for name, expected in cache_inputs.items():
            if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
                raise ValueError(f"immutable workflow request lacks cache-import {name} identity")
            if _stable_path_identity(
                Path(str(expected["path"])),
                reuse_process_authenticated_tree=(name == "cache"),
            ) != dict(expected):
                raise RuntimeError(f"cache-import {name} changed after workflow initialization")

    implementation_files = request.get("implementation_files")
    if not isinstance(implementation_files, Mapping) or not implementation_files:
        raise ValueError("immutable workflow request lacks implementation identities")
    phase_code_records = request.get(
        "phase_transitive_producer_code"
    )
    phase_code_ids = request.get("phase_producer_code_identities")
    if (
        not isinstance(phase_code_records, Mapping)
        or set(phase_code_records) != set(
            PORTABLE_CHECKPOINT_PHASE_SPECS
        )
        or not isinstance(phase_code_ids, Mapping)
        or set(phase_code_ids) != set(
            PORTABLE_CHECKPOINT_PHASE_SPECS
        )
    ):
        raise ValueError(
            "immutable workflow request lacks phase producer identities"
        )
    repository_root = Path(__file__).resolve().parents[2]
    for phase, raw_record in phase_code_records.items():
        if not isinstance(raw_record, Mapping):
            raise ValueError(
                "immutable phase producer record is invalid"
            )
        record_body = {
            key: copy.deepcopy(value)
            for key, value in raw_record.items()
            if key != "content_sha256"
        }
        constant_identity = raw_record.get(
            "workflow_constant_identity"
        )
        if (
            raw_record.get("phase") != phase
            or raw_record.get("content_sha256")
            != _sha(record_body)
            or phase_code_ids.get(phase)
            != raw_record.get("content_sha256")
            or not isinstance(constant_identity, Mapping)
        ):
            raise ValueError(
                f"immutable {phase} producer identity is invalid"
            )
        constant_body = {
            key: copy.deepcopy(value)
            for key, value in constant_identity.items()
            if key != "content_sha256"
        }
        if constant_identity.get("content_sha256") != _sha(
            constant_body
        ):
            raise ValueError(
                f"immutable {phase} workflow constants are invalid"
            )
        for inventory_name in (
            "transitive_source_inventory",
            "dependency_lock_inventory",
        ):
            inventory = raw_record.get(inventory_name)
            if not isinstance(inventory, list) or not inventory:
                raise ValueError(
                    f"immutable {phase} {inventory_name} is invalid"
                )
            for row in inventory:
                if (
                    not isinstance(row, Mapping)
                    or set(row)
                    != {
                        "relative_path",
                        "sha256",
                        "size_bytes",
                    }
                ):
                    raise ValueError(
                        f"immutable {phase} producer inventory is invalid"
                    )
                path = (
                    repository_root / str(row["relative_path"])
                ).resolve(strict=True)
                if implementation_files.get(str(path)) != row.get(
                    "sha256"
                ):
                    raise ValueError(
                        f"immutable {phase} producer inventory is unbound"
                    )
    aggregate_identity = identity_sha256(
        {
            "schema_version": (
                "workflow_phase_producer_code_aggregate_v1"
            ),
            "phase_producer_code_identities": dict(phase_code_ids),
        }
    )
    if request.get("workflow_producer_code_identity") != aggregate_identity:
        raise ValueError(
            "immutable aggregate workflow producer identity is invalid"
        )
    scientific_configuration = request.get(
        "scientific_configuration_identity"
    )
    scientific_identity = request.get("scientific_identity")
    if (
        not isinstance(scientific_configuration, Mapping)
        or not isinstance(scientific_identity, Mapping)
    ):
        raise ValueError(
            "immutable workflow scientific identities are invalid"
        )
    configuration_body = {
        key: copy.deepcopy(value)
        for key, value in scientific_configuration.items()
        if key != "scientific_configuration_sha256"
    }
    scientific_body = {
        key: copy.deepcopy(value)
        for key, value in scientific_identity.items()
        if key != "scientific_sha256"
    }
    if (
        scientific_configuration.get(
            "scientific_configuration_sha256"
        )
        != identity_sha256(configuration_body)
        or scientific_identity.get("scientific_sha256")
        != identity_sha256(scientific_body)
        or scientific_identity.get(
            "scientific_configuration_sha256"
        )
        != scientific_configuration.get(
            "scientific_configuration_sha256"
        )
        or scientific_identity.get(
            "workflow_producer_code_identity"
        )
        != aggregate_identity
        or scientific_identity.get(
            "phase_producer_code_identities"
        )
        != phase_code_ids
    ):
        raise ValueError(
            "immutable workflow scientific identity binding is invalid"
        )
    compatibility_rows = request.get(
        "expected_checkpoint_compatibilities_by_phase"
    )
    if (
        not isinstance(compatibility_rows, Mapping)
        or set(compatibility_rows)
        != set(PORTABLE_CHECKPOINT_PHASE_SPECS)
    ):
        raise ValueError(
            "immutable checkpoint compatibility domains are invalid"
        )
    for phase, raw_compatibility in compatibility_rows.items():
        if (
            not isinstance(raw_compatibility, Mapping)
            or raw_compatibility.get("configuration_identity")
            != scientific_configuration[
                "scientific_configuration_sha256"
            ]
            or raw_compatibility.get("producer_code_identity")
            != phase_code_ids[phase]
        ):
            raise ValueError(
                f"immutable {phase} checkpoint compatibility is invalid"
            )
    for raw_path, expected_sha in implementation_files.items():
        observed_sha, _size = authenticate_file(Path(str(raw_path)))
        if observed_sha != expected_sha:
            raise RuntimeError("workflow implementation changed after workflow initialization")

    for collection_name in ("integration_hooks", "phase_overrides"):
        collection = request.get(collection_name)
        if not isinstance(collection, Mapping):
            raise ValueError(f"immutable workflow request lacks {collection_name} identities")
        for identity in collection.values():
            if identity is None:
                continue
            if not isinstance(identity, Mapping):
                raise ValueError(
                    f"immutable workflow request has invalid {collection_name} identity"
                )
            source_file = identity.get("source_file")
            if source_file is not None:
                if not isinstance(source_file, Mapping) or not isinstance(source_file.get("path"), str):
                    raise ValueError(f"immutable workflow request has invalid {collection_name} source")
                observed_sha, observed_size = authenticate_file(
                    Path(str(source_file["path"]))
                )
                if observed_sha != source_file.get("sha256") or observed_size != int(
                    source_file.get("size_bytes", -1)
                ):
                    raise RuntimeError(f"{collection_name} implementation changed after initialization")
            for row in _repository_import_closure_rows(identity):
                if set(row) != {
                    "relative_path",
                    "sha256",
                    "size_bytes",
                }:
                    raise ValueError(
                        f"immutable {collection_name} import closure is invalid"
                    )
                closure_path = (
                    repository_root / str(row["relative_path"])
                ).resolve(strict=True)
                observed_sha, observed_size = authenticate_file(
                    closure_path
                )
                if (
                    observed_sha != row["sha256"]
                    or observed_size != int(row["size_bytes"])
                ):
                    raise RuntimeError(
                        f"{collection_name} dependency changed after initialization"
                    )

    source_snapshot = request.get("source_snapshot")
    if source_snapshot is not None:
        if not isinstance(source_snapshot, Mapping):
            raise ValueError("immutable workflow request has invalid source snapshot")
        from .production_source_snapshot import validate_production_source_snapshot

        observed_snapshot = validate_production_source_snapshot(
            Path(str(source_snapshot.get("root", "")))
        ).as_dict()
        if observed_snapshot != dict(source_snapshot):
            raise RuntimeError("source snapshot changed after workflow initialization")


def _canary_stage1_gpu_ids_from_request(
    request: Mapping[str, Any],
) -> tuple[int, int]:
    raw = request.get("resolved_stage1_gpu_ids")
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(type(value) is not int or value < 0 for value in raw)
        or len(set(raw)) != 2
    ):
        raise ValueError(
            "canary preparation request must bind exactly two ordered, "
            "distinct nonnegative Stage 1 GPU IDs"
        )
    return int(raw[0]), int(raw[1])


def _select_configured_canary_descriptor(
    descriptors: Mapping[str, Any],
    *,
    configured_gpu_ids: tuple[int, int],
) -> Any:
    if not isinstance(descriptors, Mapping) or not descriptors:
        raise ValueError("canonical canary descriptor set is empty")
    assignment_ids: set[int] = set()
    for descriptor in descriptors.values():
        gpu_id = getattr(getattr(descriptor, "assignment", None), "gpu_id", None)
        if type(gpu_id) is not int or gpu_id < 0:
            raise ValueError(
                "canonical canary descriptor set contains an invalid GPU assignment"
            )
        assignment_ids.add(gpu_id)
    if assignment_ids != set(configured_gpu_ids):
        raise ValueError(
            "canonical canary descriptor assignments disagree with the "
            "configured Stage 1 GPU inventory"
        )
    selected_gpu_id = configured_gpu_ids[0]
    selected = next(
        (
            descriptor
            for descriptor in descriptors.values()
            if descriptor.scope.scope_kind == "full_outer"
            and int(descriptor.assignment.gpu_id) == selected_gpu_id
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            "canonical canary descriptor set has no full-outer scope assigned "
            f"to configured cuda:{selected_gpu_id}"
        )
    return selected


def validate_stage1_canary_descriptor_preparation(
    work_root: Path | str,
) -> Mapping[str, Any]:
    """Fresh path-only validation of the pre-fit canary preparation boundary."""

    supplied = Path(work_root)
    if (
        not supplied.is_absolute()
        or supplied.is_symlink()
        or not supplied.is_dir()
        or supplied.resolve(strict=True) != supplied
    ):
        raise ValueError("canary preparation work root is invalid")
    root = supplied
    request = _read_json_object(
        root / "immutable_run_request.json",
        label="immutable workflow request",
    )
    request_body = {key: value for key, value in request.items() if key != "request_sha256"}
    request_sha = request.get("request_sha256")
    if (
        request_sha != _sha(request_body)
        or request.get("stage1_only") is not True
        or request.get("phase_sequence") != list(STAGE1_ONLY_PHASES)
        or not isinstance(request.get("source_snapshot"), Mapping)
    ):
        raise ValueError("canary preparation workflow request is invalid")
    configured_gpu_ids = _canary_stage1_gpu_ids_from_request(request)
    _revalidate_request_bound_external_inputs(request)
    prefix = ("input_preparation", "embedding_cache", "stage1_preflight")
    phase_records = {
        phase: _validate_phase_manifest_from_paths(
            work_root=root,
            phase=phase,
            request_sha256=str(request_sha),
        )
        for phase in prefix
    }
    path = root / "recovery" / "canary_descriptor_preparation_manifest.json"
    manifest = _read_json_object(
        path,
        label="canary descriptor preparation manifest",
    )
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    expected_fields = {
        "schema_version",
        "status",
        "workflow_request_sha256",
        "stage1_request_sha256",
        "source_snapshot",
        "completed_workflow_prefix",
        "cluster_preflight_manifest",
        "stage1_preflight_phase_manifest",
        "descriptor_set_manifest",
        "descriptor_set_content_sha256",
        "descriptor_count",
        "selected_scope_id",
        "selected_scope_kind",
        "selected_configured_gpu_id",
        "selected_descriptor_manifest",
        "supervised_stage1_fits_started",
        "tfidf_component_started",
        "neural_query_component_started",
        "remote_clients_constructed",
        "remote_calls_made",
        "content_sha256",
    }
    if (
        set(manifest) != expected_fields
        or manifest.get("schema_version") != "production_stage1_canary_descriptor_preparation_v2"
        or manifest.get("status") != "complete"
        or manifest.get("content_sha256") != _sha(body)
        or manifest.get("workflow_request_sha256") != request_sha
        or manifest.get("source_snapshot") != request.get("source_snapshot")
        or manifest.get("completed_workflow_prefix") != list(prefix)
        or manifest.get("selected_scope_kind") != "full_outer"
        or type(manifest.get("selected_configured_gpu_id")) is not int
        or manifest.get("selected_configured_gpu_id") != configured_gpu_ids[0]
        or manifest.get("supervised_stage1_fits_started") is not False
        or manifest.get("tfidf_component_started") is not False
        or manifest.get("neural_query_component_started") is not False
        or manifest.get("remote_clients_constructed") is not False
        or manifest.get("remote_calls_made") is not False
    ):
        raise ValueError("canary descriptor preparation manifest is invalid")

    def validate_registration(
        value: Any,
        *,
        label: str,
    ) -> Path:
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"{label} registration is invalid")
        registered = Path(str(value["path"]))
        if (
            not registered.is_absolute()
            or registered.is_symlink()
            or not registered.is_file()
            or registered.resolve(strict=True) != registered
        ):
            raise ValueError(f"{label} path is invalid")
        digest, size = stable_file_sha256(registered)
        if digest != value.get("sha256") or size != int(value.get("size_bytes", -1)):
            raise ValueError(f"{label} changed")
        return registered

    preflight_artifact = validate_registration(
        manifest["cluster_preflight_manifest"],
        label="cluster preflight manifest",
    )
    preflight_phase = validate_registration(
        manifest["stage1_preflight_phase_manifest"],
        label="Stage 1 preflight phase manifest",
    )
    descriptor_set_manifest = validate_registration(
        manifest["descriptor_set_manifest"],
        label="descriptor-set manifest",
    )
    selected_manifest = validate_registration(
        manifest["selected_descriptor_manifest"],
        label="selected descriptor manifest",
    )
    registered_preflight_files = {
        Path(row["path"]).resolve(strict=True)
        for row in phase_records["stage1_preflight"]["artifacts"]
    }
    if (
        preflight_artifact not in registered_preflight_files
        or preflight_phase != root / "phases" / "stage1_preflight" / "complete_manifest.json"
    ):
        raise ValueError("canary preparation preflight registration changed")

    from .production_stage1_legacy_scope_adapter import (
        validate_legacy_stage1_scope_descriptor_set,
    )

    descriptor_set = validate_legacy_stage1_scope_descriptor_set(
        descriptor_root=descriptor_set_manifest.parent,
        expected_stage1_request_sha256=str(manifest["stage1_request_sha256"]),
    )
    inner_partition_count = int(request["initial_training_partitions"]) + int(
        request["review_rounds"]
    )
    expected_count = int(request["outer_folds"]) * (
        1 + inner_partition_count + int(request["review_rounds"])
    )
    selected_scope_id = str(manifest["selected_scope_id"])
    selected = descriptor_set.descriptors.get(selected_scope_id)
    expected_selected = _select_configured_canary_descriptor(
        descriptor_set.descriptors,
        configured_gpu_ids=configured_gpu_ids,
    )
    if (
        len(descriptor_set.descriptors) != expected_count
        or manifest.get("descriptor_count") != expected_count
        or descriptor_set.manifest.get("content_sha256")
        != manifest.get("descriptor_set_content_sha256")
        or selected is None
        or selected.scope_id != expected_selected.scope_id
        or selected.manifest_path != selected_manifest
        or selected.scope.scope_kind != "full_outer"
        or int(selected.assignment.gpu_id) != configured_gpu_ids[0]
    ):
        raise ValueError("canary descriptor set or selected scope changed")
    if (root / "phases" / "stage1_modeling").exists() or (
        root / "recovery" / "tfidf_component_recovery"
    ).exists():
        raise ValueError("supervised Stage 1 work began before the canary")
    return copy.deepcopy(manifest)


def validate_published_workflow_checkpoint_dag(
    *,
    work_root: Path,
    expected_request_sha256: str,
    expected_phases: Sequence[str],
) -> Mapping[str, Any]:
    """Freshly reopen every published checkpoint and prove its exact DAG."""

    root = Path(work_root).resolve(strict=True)
    request = _read_json_object(
        root / "immutable_run_request.json",
        label="immutable request for checkpoint DAG validation",
    )
    request_body = {key: item for key, item in request.items() if key != "request_sha256"}
    if (
        request.get("request_sha256") != expected_request_sha256
        or _sha(request_body) != expected_request_sha256
    ):
        raise ValueError("checkpoint DAG request identity changed")
    expected_plan = _derive_expected_granular_checkpoint_plan(
        outer_folds=int(request["outer_folds"]),
        initial_training_partitions=int(
            request["initial_training_partitions"]
        ),
        review_rounds=int(request["review_rounds"]),
    )
    if (
        _validate_expected_granular_checkpoint_plan(
            request.get("expected_granular_checkpoint_plan")
        )
        != expected_plan
    ):
        raise ValueError(
            "checkpoint DAG granular plan differs from fold configuration"
        )
    phases = tuple(str(value) for value in expected_phases)
    configured_sequence = STAGE1_ONLY_PHASES if request.get("stage1_only") is True else PHASES
    if (
        not phases
        or len(phases) != len(set(phases))
        or phases != configured_sequence[: len(phases)]
    ):
        raise ValueError("checkpoint DAG phases must be an ordered workflow prefix")
    compatibility_rows = request.get(
        "expected_checkpoint_compatibilities_by_phase"
    )
    if not isinstance(compatibility_rows, Mapping):
        raise ValueError(
            "checkpoint DAG request lacks phase-specific compatibilities"
        )
    compatibilities = {
        phase: ArtifactCompatibility(**dict(value))
        for phase, value in compatibility_rows.items()
        if isinstance(value, Mapping)
    }
    if set(compatibilities) != set(PORTABLE_CHECKPOINT_PHASE_SPECS):
        raise ValueError(
            "checkpoint DAG phase-specific compatibility coverage changed"
        )
    payload_authentication_cache: dict[
        str, tuple[tuple[int, ...], str, int]
    ] = {}
    artifacts_by_phase: dict[str, ValidatedPortableArtifact] = {}
    local_phases: list[str] = []
    skipped_phases: list[str] = []
    phase_overrides = request.get("phase_overrides")
    if not isinstance(phase_overrides, Mapping):
        raise ValueError("checkpoint DAG request lacks phase identities")
    adoption_records = request.get("requested_checkpoint_adoptions")
    adoption_locators = request.get("checkpoint_adoption_locators")
    if (
        not isinstance(adoption_records, list)
        or not isinstance(adoption_locators, list)
        or len(adoption_records) != len(adoption_locators)
    ):
        raise ValueError("checkpoint DAG request has invalid adoptions")
    adopted_handles: dict[str, ValidatedPortableArtifact] = {}
    for record, locator in zip(
        adoption_records,
        adoption_locators,
        strict=True,
    ):
        if not isinstance(record, Mapping):
            raise ValueError(
                "checkpoint DAG request has an invalid adoption record"
            )
        kind = str(record.get("artifact_kind") or "")
        compatibility_phase = (
            CHECKPOINT_COMPATIBILITY_PHASE_BY_ARTIFACT_KIND.get(kind)
        )
        if (
            compatibility_phase is None
            or record.get("compatibility_phase")
            != compatibility_phase
        ):
            raise ValueError(
                "adopted checkpoint compatibility domain changed"
            )
        artifact = _open_adopted_artifact(
            record=record,
            locator=Path(str(locator)),
            expected_compatibility=compatibilities[
                compatibility_phase
            ].as_dict(),
            payload_authentication_cache=payload_authentication_cache,
        )
        if (
            artifact.artifact_id != record.get("artifact_id")
            or artifact.compatibility_key
            != record.get("compatibility_key")
            or dict(artifact.artifact_metadata)
            != dict(record.get("artifact_metadata") or {})
            or artifact.artifact_id in adopted_handles
        ):
            raise ValueError(
                "adopted checkpoint record changed during DAG validation"
            )
        attestation_path = (
            root
            / "checkpoint_adoptions"
            / f"{artifact.artifact_id}.adoption.json"
        )
        _validate_adoption_attestation_for_record(
            attestation_path=attestation_path,
            artifact=artifact,
            record=record,
            consumer_request_sha256=expected_request_sha256,
        )
        adopted_handles[artifact.artifact_id] = artifact
    if adopted_handles:
        _validate_adopted_checkpoint_graph(
            tuple(adopted_handles.values()),
            allowed_phases=phases,
            expected_granular_checkpoint_plan=(
                _validate_expected_granular_checkpoint_plan(
                    request.get(
                        "expected_granular_checkpoint_plan"
                    )
                )
            ),
            expected_stage1_physical_fit_identity=dict(
                request.get("stage1_physical_fit_identity") or {}
            ),
            expected_global_seed=int(request["seed"]),
            require_prepared_stage1_context=(
                request.get("portable_typed_workflow") is True
            ),
        )

    typed_workflow = request.get("portable_typed_workflow") is True
    granular_indexes: dict[str, Mapping[str, Any]] = {}
    granular_handles_by_phase: dict[
        str, tuple[ValidatedPortableArtifact, ...]
    ] = {}

    for phase in phases:
        spec = PORTABLE_CHECKPOINT_PHASE_SPECS.get(phase)
        if not isinstance(spec, Mapping):
            continue
        phase_manifest = _validate_phase_manifest_from_paths(
            work_root=root,
            phase=phase,
            request_sha256=expected_request_sha256,
        )
        adopted = phase_manifest.get("adopted_checkpoint")
        rows = phase_manifest.get("artifacts")
        if not isinstance(adopted, Mapping) and not isinstance(rows, list):
            raise ValueError(f"{phase} phase has no authenticated artifact inventory")
        if not isinstance(adopted, Mapping) and not rows:
            skipped_by_configuration = (
                phase == "oracle_evaluation"
                and phase_manifest.get("result", {}).get("skipped_by_configuration") is True
            )
            if phase_overrides.get(phase) is None and not (skipped_by_configuration):
                raise ValueError(f"{phase} lacks its required portable checkpoint")
            skipped_control = root / "portable_checkpoints" / phase
            if skipped_control.exists() or skipped_control.is_symlink():
                raise ValueError(f"{phase} has a checkpoint despite an empty phase tree")
            skipped_phases.append(phase)
            continue
        missing_upstream = [
            parent for parent in spec["upstream_phases"] if parent not in artifacts_by_phase
        ]
        if missing_upstream:
            raise ValueError(
                f"{phase} checkpoint lacks upstream phase checkpoints: " f"{missing_upstream}"
            )
        base_upstream = tuple(
            artifacts_by_phase[parent].artifact_id for parent in spec["upstream_phases"]
        )
        local_granular_index: Mapping[str, Any] | None = None
        local_granular_handles: tuple[
            ValidatedPortableArtifact, ...
        ] = ()
        if (
            typed_workflow
            and phase in {"stage1_modeling", "stage2_inference"}
            and not isinstance(adopted, Mapping)
        ):
            current_stage1_scope_plan = None
            granular_external_upstream = base_upstream
            if phase == "stage1_modeling":
                prepared_contexts = tuple(
                    granular_handles_by_phase.get(
                        "stage1_preflight", ()
                    )
                )
                if (
                    len(prepared_contexts) != 1
                    or prepared_contexts[0].manifest.get(
                        "artifact_kind"
                    )
                    != "prepared_stage1_context"
                ):
                    raise ValueError(
                        "local Stage 1 validation lacks its authenticated "
                        "prepared context"
                    )
                current_stage1_scope_plan = (
                    _load_authenticated_current_stage1_scope_plan(
                        prepared_context_artifact=prepared_contexts[0],
                        expected_granular_checkpoint_plan=expected_plan,
                        expected_stage1_physical_fit_identity=dict(
                            request.get(
                                "stage1_physical_fit_identity"
                            )
                            or {}
                        ),
                        expected_global_seed=int(request["seed"]),
                    )
                )
                granular_external_upstream = (
                    prepared_contexts[0].artifact_id,
                )
            (
                local_granular_index,
                local_granular_handles,
            ) = _validate_granular_checkpoint_index_from_paths(
                work_root=root,
                phase=phase,
                compatibility=compatibilities[phase],
                payload_authentication_cache=(
                    payload_authentication_cache
                ),
                expected_granular_checkpoint_plan=expected_plan,
                expected_stage1_scope_plan=(
                    current_stage1_scope_plan
                ),
                expected_external_upstream_artifact_ids=(
                    granular_external_upstream
                ),
            )
            primary_metadata = _granular_primary_metadata_from_index(
                phase=phase,
                index=local_granular_index,
            )
            expected_upstream = (
                *base_upstream,
                *tuple(
                    primary_metadata[
                        "granular_terminal_artifact_ids"
                    ]
                ),
            )
        elif typed_workflow and phase in {
            "stage1_modeling",
            "stage2_inference",
        }:
            matching_primary = [
                handle
                for handle in adopted_handles.values()
                if handle.artifact_id == adopted.get("artifact_id")
            ]
            if len(matching_primary) != 1:
                raise ValueError(
                    f"{phase} adopted primary checkpoint is absent"
                )
            primary_metadata = dict(
                matching_primary[0].artifact_metadata
            )
            all_ids = primary_metadata.get(
                "granular_artifact_ids"
            )
            terminal_ids = primary_metadata.get(
                "granular_terminal_artifact_ids"
            )
            if (
                not isinstance(all_ids, list)
                or not all_ids
                or not isinstance(terminal_ids, list)
                or not terminal_ids
                or any(
                    artifact_id not in adopted_handles
                    for artifact_id in all_ids
                )
                or not set(terminal_ids).issubset(set(all_ids))
            ):
                raise ValueError(
                    f"{phase} adopted granular coverage is incomplete"
                )
            local_granular_handles = tuple(
                adopted_handles[str(artifact_id)]
                for artifact_id in all_ids
            )
            expected_upstream = (
                *base_upstream,
                *tuple(str(value) for value in terminal_ids),
            )
        else:
            expected_upstream = base_upstream
        if isinstance(adopted, Mapping):
            matching = [
                (record, locator)
                for record, locator in zip(
                    adoption_records,
                    adoption_locators,
                )
                if isinstance(record, Mapping)
                and record.get("artifact_id") == adopted.get("artifact_id")
                and record.get("substituted_phase") == phase
            ]
            if len(matching) != 1:
                raise ValueError(f"{phase} adopted checkpoint is not uniquely requested")
            artifact = adopted_handles.get(
                str(adopted["artifact_id"])
            )
            if artifact is None:
                raise ValueError(
                    f"{phase} adopted checkpoint is absent"
                )
            adoption_record = matching[0][0]
            if (
                artifact.manifest.get("artifact_kind")
                != spec["artifact_kind"]
                or not _adopted_compatibility_matches_request(
                    artifact=artifact,
                    expected=compatibilities[phase].as_dict(),
                    record=adoption_record,
                )
                or tuple(
                    artifact.manifest["upstream_artifact_ids"]
                )
                != expected_upstream
            ):
                raise ValueError(
                    f"{phase} adopted checkpoint DAG changed"
                )
        else:
            assert isinstance(rows, list) and rows
            control_root = root / "portable_checkpoints" / phase
            artifact = validate_portable_artifact(
                control_root,
                expected_kind=str(spec["artifact_kind"]),
                expected_compatibility_key=compatibilities[phase].key,
                expected_upstream_artifact_ids=expected_upstream,
                payload_authentication_cache=payload_authentication_cache,
            )
            if (
                artifact.root != control_root.resolve(strict=True)
                or artifact.payload_root
                != Path(str(phase_manifest["attempt_dir"])).resolve(strict=True)
                or artifact.manifest.get("artifact_schema") != spec["artifact_schema"]
                or not isinstance(artifact.phase_binding, Mapping)
                or artifact.phase_binding.get("phase") != phase
            ):
                raise ValueError(f"{phase} portable checkpoint binding is invalid")
            if (
                local_granular_index is not None
                and dict(artifact.artifact_metadata)
                != _granular_primary_metadata_from_index(
                    phase=phase,
                    index=local_granular_index,
                )
            ):
                raise ValueError(
                    f"{phase} primary granular coverage binding changed"
                )
            expected_inventory = [
                (
                    str(row["relative_path"]),
                    str(row["sha256"]),
                    int(row["size_bytes"]),
                )
                for row in rows
            ]
            observed_inventory = [
                (
                    row.relative_path,
                    row.sha256,
                    row.size_bytes,
                )
                for row in artifact.payloads
            ]
            if observed_inventory != expected_inventory:
                raise ValueError(f"{phase} checkpoint inventory differs from its phase")
            materialized = materialize_portable_phase(
                artifact,
                expected_phase=phase,
            )
            if materialized["artifacts"] != rows:
                raise ValueError(f"{phase} checkpoint does not materialize its phase tree")
            expected_attestation = _checkpoint_publication_attestation_value(
                producer_request_sha256=expected_request_sha256,
                phase=phase,
                phase_manifest_path=(root / "phases" / phase / "complete_manifest.json"),
                phase_manifest=phase_manifest,
                artifact=artifact,
            )
            observed_attestation = _read_json_object(
                root
                / "execution_attestations"
                / "portable_checkpoint_publications"
                / f"{phase}.json",
                label=f"{phase} checkpoint publication attestation",
            )
            if observed_attestation != expected_attestation:
                raise ValueError(f"{phase} checkpoint publication attestation changed")
            local_phases.append(phase)
        artifacts_by_phase[phase] = artifact
        if typed_workflow and phase == "stage1_preflight":
            if isinstance(adopted, Mapping):
                prepared = tuple(
                    handle
                    for handle in adopted_handles.values()
                    if handle.manifest.get("artifact_kind")
                    == "prepared_stage1_context"
                    and handle.artifact_metadata.get("producer_phase")
                    == phase
                )
                if (
                    len(prepared) != 1
                    or tuple(
                        prepared[0].manifest[
                            "upstream_artifact_ids"
                        ]
                    )
                    != (artifact.artifact_id,)
                ):
                    raise ValueError(
                        "adopted prepared Stage 1 context binding changed"
                    )
                local_granular_handles = prepared
            else:
                (
                    local_granular_index,
                    local_granular_handles,
                ) = _validate_granular_checkpoint_index_from_paths(
                    work_root=root,
                    phase=phase,
                    compatibility=compatibilities[phase],
                    payload_authentication_cache=(
                        payload_authentication_cache
                    ),
                    expected_granular_checkpoint_plan=expected_plan,
                )
                if (
                    len(local_granular_handles) != 1
                    or local_granular_handles[
                        0
                    ].manifest.get("artifact_kind")
                    != "prepared_stage1_context"
                    or tuple(
                        local_granular_handles[
                            0
                        ].manifest["upstream_artifact_ids"]
                    )
                    != (artifact.artifact_id,)
                ):
                    raise ValueError(
                        "prepared Stage 1 context granular DAG changed"
                    )
        if local_granular_index is not None:
            granular_indexes[phase] = local_granular_index
        if local_granular_handles:
            granular_handles_by_phase[phase] = (
                local_granular_handles
            )

    checkpoint_root = root / "portable_checkpoints"
    observed_controls: set[str] = set()
    if checkpoint_root.exists() or checkpoint_root.is_symlink():
        if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
            raise ValueError("portable checkpoint root is invalid")
        for child in checkpoint_root.iterdir():
            if child.is_symlink() or not child.is_dir():
                raise ValueError("portable checkpoint root contains an invalid entry")
            observed_controls.add(child.name)
    if observed_controls != set(local_phases):
        raise ValueError("portable checkpoint controls contain missing or extra phases")
    granular_root = root / "portable_granular_checkpoints"
    observed_granular_phases: set[str] = set()
    if granular_root.exists() or granular_root.is_symlink():
        if granular_root.is_symlink() or not granular_root.is_dir():
            raise ValueError(
                "portable granular checkpoint root is invalid"
            )
        for child in granular_root.iterdir():
            if child.is_symlink() or not child.is_dir():
                raise ValueError(
                    "portable granular checkpoint root contains an invalid entry"
                )
            observed_granular_phases.add(child.name)
    if observed_granular_phases != set(granular_indexes):
        raise ValueError(
            "portable granular checkpoint indexes contain missing or extra phases"
        )
    attestation_root = root / "execution_attestations" / "portable_checkpoint_publications"
    observed_attestations: set[str] = set()
    if attestation_root.exists() or attestation_root.is_symlink():
        if attestation_root.is_symlink() or not attestation_root.is_dir():
            raise ValueError("portable checkpoint publication attestation root is invalid")
        for child in attestation_root.iterdir():
            state = os.lstat(child)
            if (
                stat.S_ISLNK(state.st_mode)
                or not stat.S_ISREG(state.st_mode)
                or int(state.st_nlink) != 1
            ):
                raise ValueError(
                    "portable checkpoint publication attestations contain " "an invalid entry"
                )
            observed_attestations.add(child.name)
    if observed_attestations != {f"{phase}.json" for phase in local_phases}:
        raise ValueError(
            "portable checkpoint publication attestations contain missing " "or extra phases"
        )

    operator_trusted_phases = [
        str(record.get("substituted_phase"))
        for record in adoption_records
        if isinstance(record, Mapping)
        and _operator_trusted_adoption_selected(record)
        and record.get("substituted_phase") in phases
    ]
    body = {
        "schema_version": WORKFLOW_CHECKPOINT_DAG_VALIDATION_SCHEMA,
        "status": "accepted",
        "request_sha256": expected_request_sha256,
        "validated_phases": list(phases),
        "checkpoint_artifact_ids": {
            phase: artifacts_by_phase[phase].artifact_id
            for phase in phases
            if phase in artifacts_by_phase
        },
        "granular_checkpoint_artifact_ids": {
            phase: [
                artifact.artifact_id
                for artifact in granular_handles_by_phase[phase]
            ]
            for phase in phases
            if phase in granular_handles_by_phase
        },
        "granular_checkpoint_indexes": {
            phase: {
                "content_sha256": granular_indexes[phase][
                    "content_sha256"
                ],
                "coverage_content_sha256": granular_indexes[phase][
                    "coverage"
                ]["content_sha256"],
                "node_count": granular_indexes[phase][
                    "node_count"
                ],
            }
            for phase in phases
            if phase in granular_indexes
        },
        "local_publication_phases": local_phases,
        "adopted_publication_phases": [
            phase for phase in phases if phase in artifacts_by_phase and phase not in local_phases
        ],
        "checkpoint_skipped_phases": skipped_phases,
        "oracle_evaluation_after_frozen_prediction": (
            "oracle_evaluation" not in artifacts_by_phase
            or tuple(artifacts_by_phase["oracle_evaluation"].manifest["upstream_artifact_ids"])
            == (artifacts_by_phase["stage2_inference"].artifact_id,)
        ),
        "fresh_full_byte_validation": not operator_trusted_phases,
        "operator_trusted_checkpoint_reuse": bool(
            operator_trusted_phases
        ),
        "operator_trusted_checkpoint_phases": operator_trusted_phases,
        "payload_bytes_reauthenticated_for_all_adoptions": (
            not operator_trusted_phases
        ),
        "global_release_certified": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _hook_identity(
    hook: WorkflowPhaseHook | None,
    *,
    identity_memo: _ScientificIdentityMemo | None = None,
) -> Mapping[str, Any] | None:
    if hook is None:
        return None
    return _callable_behavior_identity(
        hook,
        identity_memo=identity_memo,
    )


def _scientific_callable_identity(
    value: Any,
    *,
    method_name: str | None = None,
    explicit_scientific_identity: Mapping[str, Any] | None = None,
    identity_memo: _ScientificIdentityMemo | None = None,
) -> Mapping[str, Any]:
    """Return a path-neutral source identity for injected scientific code."""

    target = getattr(value, method_name) if method_name is not None else value
    return _path_neutral_injected_identity(
        _callable_behavior_identity(
            target,
            explicit_scientific_identity=(
                explicit_scientific_identity
            ),
            identity_memo=identity_memo,
        )
    )


def _role_neutral_stage1_integration_identity(
    integration: ProductionRoleNeutralStage1Integration | None,
    *,
    identity_memo: _ScientificIdentityMemo | None = None,
) -> Mapping[str, Any] | None:
    if integration is None:
        return None
    body = {
        "schema_version": (
            "production_role_neutral_stage1_integration_code_identity_v2"
        ),
        "producer_factories_builder": _scientific_callable_identity(
            integration.producer_factories_builder,
            explicit_scientific_identity=(
                integration.producer_factories_scientific_identity
            ),
            identity_memo=identity_memo,
        ),
        "physical_owner_executor": _scientific_callable_identity(
            integration.executor,
            method_name="execute",
            explicit_scientific_identity=(
                integration.physical_owner_executor_scientific_identity
            ),
            identity_memo=identity_memo,
        ),
        "stage2_handoff_publisher": _scientific_callable_identity(
            integration.handoff_publisher,
            explicit_scientific_identity=(
                integration.handoff_publisher_scientific_identity
            ),
            identity_memo=identity_memo,
        ),
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _validate_portable_role_neutral_stage1_phase_result(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the closed phase/adoption claim before it can name a handoff."""

    required = {
        "schema_version",
        "execution_mode",
        "prepared_stage1_request_sha256",
        "stage1_scope_plan_scientific_content_sha256",
        "role_neutral_execution_root",
        "role_neutral_execution_manifest_path",
        "role_neutral_execution_content_sha256",
        "role_neutral_handoff_binding_path",
        "bundle_manifest_path",
        "bundle_sha256",
        "direct_numerical_bank_manifest_path",
        "direct_numerical_bank_locator_path",
        "direct_numerical_bank_content_sha256",
        "physical_fit_count",
        "logical_scope_count",
        "deduplicated_fit_count",
        "every_physical_owner_executed_once",
        "productive_compute_canary_completed",
        "selected_canary_replica_adopted_as_production",
        "compute_canary_scientific_equality",
        "all_ten_families_bound_per_logical_context",
        "legacy_bundle_build_invoked",
        "stage2_handoff_derived_exclusively_from_role_neutral_execution",
        "resource_preflight",
        "terminal_files",
    }
    result = copy.deepcopy(dict(value))
    digests = (
        result.get("prepared_stage1_request_sha256"),
        result.get("stage1_scope_plan_scientific_content_sha256"),
        result.get("role_neutral_execution_content_sha256"),
        result.get("bundle_sha256"),
        result.get("direct_numerical_bank_content_sha256"),
    )
    path_fields = (
        "role_neutral_execution_root",
        "role_neutral_execution_manifest_path",
        "role_neutral_handoff_binding_path",
        "bundle_manifest_path",
        "direct_numerical_bank_manifest_path",
        "direct_numerical_bank_locator_path",
    )
    terminals = result.get("terminal_files")
    if (
        set(result) != required
        or result.get("schema_version") != PORTABLE_ROLE_NEUTRAL_STAGE1_PHASE_SCHEMA
        or result.get("execution_mode") != "deduplicated_role_neutral_all_ten_v1"
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        )
        or any(
            not isinstance(result.get(field), str) or not Path(str(result[field])).is_absolute()
            for field in path_fields
        )
        or isinstance(result.get("physical_fit_count"), bool)
        or not isinstance(result.get("physical_fit_count"), int)
        or int(result["physical_fit_count"]) < 1
        or isinstance(result.get("logical_scope_count"), bool)
        or not isinstance(result.get("logical_scope_count"), int)
        or int(result["logical_scope_count"]) < int(result["physical_fit_count"])
        or result.get("deduplicated_fit_count")
        != int(result["logical_scope_count"]) - int(result["physical_fit_count"])
        or result.get("every_physical_owner_executed_once") is not True
        or result.get("productive_compute_canary_completed") is not False
        or result.get("selected_canary_replica_adopted_as_production") is not False
        or result.get("compute_canary_scientific_equality") is not None
        or result.get("all_ten_families_bound_per_logical_context") is not True
        or result.get("legacy_bundle_build_invoked") is not False
        or result.get("stage2_handoff_derived_exclusively_from_role_neutral_execution") is not True
        or not isinstance(result.get("resource_preflight"), Mapping)
        or not isinstance(terminals, list)
        or any(not isinstance(path, str) for path in terminals)
        or len(terminals) != len(set(terminals))
        or set(terminals)
        != {
            result["role_neutral_execution_manifest_path"],
            result["role_neutral_handoff_binding_path"],
            result["bundle_manifest_path"],
            result["direct_numerical_bank_manifest_path"],
            result["direct_numerical_bank_locator_path"],
        }
    ):
        raise ValueError(
            "portable Stage 1 handoff lacks a closed authenticated "
            "role-neutral all-ten execution claim"
        )
    execution_root = Path(result["role_neutral_execution_root"])
    execution_manifest = Path(result["role_neutral_execution_manifest_path"])
    if execution_manifest.parent != execution_root:
        raise ValueError("role-neutral execution manifest is outside its execution root")
    return result


def _resolved_stage1_gpu_ids(options: "ProductionAllEvidenceWorkflowOptions") -> tuple[int, ...]:
    plural = tuple(int(value) for value in options.stage1_gpu_ids)
    singular = None if options.gpu_id is None else int(options.gpu_id)
    if plural and singular is not None and plural != (singular,):
        raise ValueError("--gpu-id conflicts with ordered --stage1-gpu-id values")
    resolved = plural or (() if singular is None else (singular,))
    if not resolved and str(options.stage1_device).startswith("cuda:"):
        try:
            resolved = (int(str(options.stage1_device).split(":", 1)[1]),)
        except ValueError as exc:
            raise ValueError("stage1_device must name one explicit CUDA index") from exc
    if any(value < 0 for value in resolved) or len(set(resolved)) != len(resolved):
        raise ValueError("Stage 1 GPU IDs must be nonnegative, unique, and ordered")
    return resolved


def _resolved_query_devices(options: "ProductionAllEvidenceWorkflowOptions") -> tuple[str, ...]:
    plural = tuple(str(value) for value in options.query_devices)
    singular = None if options.query_device is None else str(options.query_device)
    if plural and singular is not None and plural != (singular,):
        raise ValueError("query_device conflicts with ordered query_devices")
    resolved = plural or (() if singular is None else (singular,))
    if resolved:
        return resolved
    if options.stage1_device is None:
        raise ValueError("query devices require an explicit stage1_device or device policy")
    return (str(options.stage1_device),)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nullable_positive_int_argument(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "null"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return parsed


def _forest_max_features_argument(
    value: str,
) -> str | float | int | None:
    normalized = str(value).strip()
    if normalized.lower() in {"none", "null"}:
        return None
    try:
        return int(normalized)
    except ValueError:
        try:
            return float(normalized)
        except ValueError:
            return normalized


@dataclass(frozen=True)
class ProductionAllEvidenceWorkflowOptions:
    dataset_path: Path
    work_root: Path
    stage1_profile_path: Path
    query_profile_path: Path
    unit_id_column: str
    text_column: str
    treatment_column: str
    outcome_column: str
    outcome_type: str
    clinical_question: str
    embedding_model_name: str
    embedding_local_model_path: Path
    htr_local_model_path: Path
    resource_performance_safety: ResourcePerformanceSafetyPolicy
    run_control: RunControl
    endpoint: str | None = None
    model_name: str | None = None
    stage2_tokenizer_locator: Path | None = None
    outer_folds: int | None = None
    review_rounds: int | None = None
    initial_training_partitions: int | None = None
    interaction_inner_folds: int | None = None
    tfidf_nested_calibration_folds: int | None = None
    stage1_device: str | None = None
    query_device: str | None = None
    query_devices: tuple[str, ...] = ()
    review_device: str | None = None
    gpu_id: int | None = None
    stage1_gpu_ids: tuple[int, ...] = ()
    stage1_execution_device_count: int = 1
    stage1_scope_workers_per_gpu: int = 1
    stage1_execution_profile: Stage1ExecutionProfile | None = None
    stage1_preflight_workers: int = 8
    stage1_preflight_execution_attestation: Mapping[str, Any] | None = None
    stage1_seed_policy: str | None = None
    num_workers: int = 1
    tfidf_workers: int = 8
    tfidf_parallel_backend: str = "processes"
    seed: int | None = None
    empty_text_policy: str | None = None
    repeated_character_policy: str | None = None
    repeated_character_threshold: int | None = None
    source_text_temporally_valid_by_design: bool | None = None
    evaluate_oracle_posthoc: bool = False
    oracle_dataset_path: Path | None = None
    oracle_unit_id_column: str | None = None
    oracle_ite_column: str | None = None
    embedding_cache_import: Path | None = None
    embedding_cache_import_source_prepared_path: Path | None = None
    embedding_cache_import_source_preparation_manifest_path: Path | None = None
    source_snapshot_root: Path | None = None
    stage1_only: bool = False
    scratch_root: Path | None = None
    device_policy: tuple[str, ...] = ()
    cpu_budget: int = 1
    response_concurrency: int = 1
    storage_backend: str = "posix"
    cluster_preflight_parquet_compression: str | None = None
    runtime_compatibility_class: str = "portable_python_posix_v1"
    legacy_preflight_candidate: Path | None = None
    portable_scientific_spec: Mapping[str, Any] | None = None
    scientific_spec_path: Path | None = None
    deployment_profile_path: Path | None = None
    forest_runtime_config: StrictCausalForestRuntimeConfig | None = None
    # The flat forest fields are retained only for non-portable internal
    # compatibility. Typed portable requests must leave all of them null.
    forest_n_estimators: int | None = None
    forest_max_depth: int | None = None
    forest_min_samples_leaf: int | None = None
    forest_max_features: str | float | int | None = None
    forest_honest: bool | None = None
    forest_inference: bool | None = None
    forest_subforest_size: int | None = None
    forest_tune_model: bool | None = None
    forest_nuisance_n_estimators: int | None = None
    forest_nuisance_max_depth: int | None = None
    forest_nuisance_min_samples_leaf: int | None = None
    forest_nuisance_treatment_max_features: str | float | int | None = None
    forest_nuisance_outcome_max_features: str | float | int | None = None
    forest_random_seed: int | None = None
    max_candidate_variables: int | None = None
    stage2_prompt_protocol: Stage2PromptProtocolSpec | None = None
    post_extraction_causal_review: PostExtractionCausalReviewSpec | None = None
    complete_page_core_chars: int | None = None
    complete_page_context_chars: int | None = None
    complete_page_max_chars: int | None = None
    complete_reconciliation_fan_in: int | None = None
    embedding_chunk_size_words: int | None = None
    embedding_chunk_overlap_words: int | None = None
    embedding_max_chunks: int | None = None
    embedding_chunk_selection: str | None = None
    embedding_max_seq_length: int | None = None
    embedding_normalize: bool | None = None
    embedding_encoder: SentenceEmbeddingEncoderSpec | None = None
    embedding_batch_size: int | None = None


class ProductionAllEvidenceWorkflow:
    """Fail-closed phase runner; completed phases are content-addressed."""

    def __init__(
        self,
        options: ProductionAllEvidenceWorkflowOptions,
        *,
        phase_overrides: Mapping[str, Callable[[Path], Mapping[str, Any]]] | None = None,
        hooks: ProductionAllEvidenceWorkflowHooks | None = None,
    ) -> None:
        self.options = options
        self.phase_overrides = dict(phase_overrides or {})
        self.hooks = hooks or ProductionAllEvidenceWorkflowHooks()
        if (
            self.hooks.role_neutral_stage1 is not None
            and self.options.portable_scientific_spec is None
        ):
            raise ValueError(
                "the role-neutral Stage 1 integration seam is reserved for "
                "typed portable workflow requests"
            )
        if self.hooks.role_neutral_stage1 is not None and self.hooks.stage1_modeling is not None:
            raise ValueError(
                "role-neutral and generic Stage 1 modeling integrations are " "mutually exclusive"
            )
        if self.options.portable_scientific_spec is not None and (
            self.hooks.stage1_modeling is not None or "stage1_modeling" in self.phase_overrides
        ):
            raise ValueError(
                "typed portable Stage 1 forbids generic modeling hooks and "
                "phase overrides; configure the explicit role-neutral "
                "integration seam"
            )
        legacy_preflight_requested = self.options.legacy_preflight_candidate is not None or any(
            Path(source).name == "cluster_preflight_manifest.json"
            for source in self.options.run_control.adopt_checkpoints
        )
        if legacy_preflight_requested and (
            self.hooks.stage1_preflight is not None or "stage1_preflight" in self.phase_overrides
        ):
            raise ValueError(
                "legacy preflight migration accounting requires the current "
                "built-in preflight producer; hooks and phase overrides cannot "
                "bypass its full-byte recompute decision"
            )
        self.request: dict[str, Any] = {}
        self._scientific_identity_memo = _ScientificIdentityMemo()
        self._adopted_artifact_handles: dict[str, ValidatedPortableArtifact] = {}
        self._operator_trusted_checkpoint_handles: dict[
            str, OperatorTrustedCheckpoint
        ] = {}
        self._published_checkpoint_handles: dict[str, ValidatedPortableArtifact] = {}
        self._published_granular_checkpoint_handles: dict[
            str, ValidatedPortableArtifact
        ] = {}
        self._published_granular_checkpoint_indexes: dict[
            str, Mapping[str, Any]
        ] = {}
        self._phase_payload_stat_inventories: dict[str, Mapping[str, tuple[int, ...]]] = {}
        self._validate_options()
        self._validation_policy = _resolve_validation_depth_policy(
            self.options.run_control.validation_depth
        )
        self._run_control_selection_attestation: Mapping[str, Any] | None = None
        self._run_control_selection_attestation_path: Path | None = None
        self._validation_achievement_attestation: Mapping[str, Any] | None = None
        self._validation_achievement_attestation_path: Path | None = None
        telemetry_devices = tuple(f"cuda:{gpu_id}" for gpu_id in self.stage1_gpu_ids)
        if not telemetry_devices and self.options.stage1_device == "cpu":
            telemetry_devices = ("cpu",)
        self.telemetry = TelemetryLedger(devices=telemetry_devices)

    @property
    def stage1_gpu_ids(self) -> tuple[int, ...]:
        return _resolved_stage1_gpu_ids(self.options)

    @property
    def query_devices(self) -> tuple[str, ...]:
        return _resolved_query_devices(self.options)

    def _phase_sequence(self) -> tuple[str, ...]:
        return STAGE1_ONLY_PHASES if self.options.stage1_only else PHASES

    def _resolved_cache_import_sources(self) -> tuple[Path, Path] | None:
        o = self.options
        if o.embedding_cache_import is None:
            return None
        if (
            o.embedding_cache_import_source_prepared_path is not None
            and o.embedding_cache_import_source_preparation_manifest_path is not None
        ):
            return (
                o.embedding_cache_import_source_prepared_path.resolve(strict=True),
                o.embedding_cache_import_source_preparation_manifest_path.resolve(strict=True),
            )
        metadata_path = o.embedding_cache_import / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        provenance = metadata.get("production_provenance")
        dataset = provenance.get("dataset") if isinstance(provenance, Mapping) else None
        source = dataset.get("path") if isinstance(dataset, Mapping) else None
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "embedding-cache metadata does not identify its prepared cohort; "
                "supply the two explicit source-preparation options"
            )
        prepared = Path(source).resolve(strict=True)
        manifest = (prepared.parent / "preparation_manifest.json").resolve(strict=True)
        return prepared, manifest

    def _validate_options(self) -> None:
        o = self.options
        stage1_gpu_ids = _resolved_stage1_gpu_ids(o)
        query_devices = _resolved_query_devices(o)
        valid_device = lambda value: value == "cpu" or (
            value.startswith("cuda:") and value.split(":", 1)[1].isdigit()
        )
        if o.stage1_device is None or not valid_device(str(o.stage1_device)):
            raise ValueError("stage1_device must be explicitly configured as cpu or cuda:N")
        if (
            not query_devices
            or any(not valid_device(value) for value in query_devices)
            or len(query_devices) != len(set(query_devices))
        ):
            raise ValueError("query devices must contain unique explicit cpu/cuda:N values")
        if o.review_device is None or not valid_device(str(o.review_device)):
            raise ValueError("review_device must be explicitly configured as cpu or cuda:N")
        cuda_devices = {
            int(value.split(":", 1)[1])
            for value in (str(o.stage1_device), *query_devices)
            if value.startswith("cuda:")
        }
        if not o.stage1_only and str(o.review_device).startswith("cuda:"):
            cuda_devices.add(int(str(o.review_device).split(":", 1)[1]))
        if not cuda_devices.issubset(set(stage1_gpu_ids)):
            raise ValueError(
                "every Stage 1/query CUDA device must be included in the exclusive "
                "Stage 1 GPU IDs"
            )
        if any(
            value < 1
            for value in (
                o.stage1_scope_workers_per_gpu,
                o.stage1_execution_device_count,
                o.stage1_preflight_workers,
                o.num_workers,
                o.tfidf_workers,
                o.cpu_budget,
                o.response_concurrency,
            )
        ):
            raise ValueError("Stage 1 and TF-IDF worker counts must be positive")
        if o.portable_scientific_spec is None and o.stage1_scope_workers_per_gpu != 1:
            raise ValueError(
                "historical Stage 1 requires exactly one scope worker per GPU; "
                "typed portable Stage 1 accepts the deployment-selected "
                "positive concurrency"
            )
        if (
            o.portable_scientific_spec is not None
            and o.stage1_execution_device_count != len(query_devices)
        ):
            raise ValueError(
                "typed portable Stage 1 execution-device count must equal the "
                "resolved deployment-selected device inventory"
            )
        if o.portable_scientific_spec is not None and (
            not isinstance(o.stage1_execution_profile, Stage1ExecutionProfile)
            or o.stage1_execution_profile.device_count
            != o.stage1_execution_device_count
            or o.stage1_execution_profile.scope_workers_per_device
            != o.stage1_scope_workers_per_gpu
        ):
            raise ValueError(
                "typed portable Stage 1 requires its complete deployment "
                "execution-selection profile"
            )
        if o.portable_scientific_spec is not None:
            profile = o.stage1_execution_profile
            assert isinstance(profile, Stage1ExecutionProfile)
            policy = profile.preflight_execution_policy
            caps = {
                "cpu_budget": int(o.cpu_budget),
                "stage1_owner_cap": int(
                    profile.max_parallel_owners
                ),
                "preflight_owner_cap": int(
                    policy.max_parallel_owners
                ),
                "memory_lane_cap": int(policy.memory_lane_cap),
                "input_io_lane_cap": int(
                    policy.input_io_lane_cap
                ),
                "publication_io_lane_cap": int(
                    policy.publication_io_lane_cap
                ),
                "authentication_io_lane_cap": int(
                    policy.authentication_io_lane_cap
                ),
                "ordinary_read_amplification_lane_cap": math.floor(
                    o.resource_performance_safety
                    .maximum_ordinary_read_amplification
                ),
            }
            body = {
                "schema_version": (
                    "production_stage1_preflight_execution_attestation_v1"
                ),
                "policy": policy.as_dict(),
                "derived_caps": caps,
                "effective_preflight_owner_lanes_before_scope_cap": min(
                    caps.values()
                ),
                "physical_owner_count_applied_by_preflight_executor": True,
                "resource_assignment_in_scientific_identity": False,
                "completion_order_in_scientific_identity": False,
            }
            expected_attestation = {
                **body,
                "content_sha256": _sha(body),
            }
            if (
                not isinstance(
                    o.stage1_preflight_execution_attestation,
                    Mapping,
                )
                or dict(
                    o.stage1_preflight_execution_attestation
                )
                != expected_attestation
                or int(o.stage1_preflight_workers)
                != int(
                    expected_attestation[
                        "effective_preflight_owner_lanes_before_scope_cap"
                    ]
                )
            ):
                raise ValueError(
                    "Stage 1 preflight worker count or execution "
                    "attestation differs from the compiled deployment policy"
                )
        if o.stage1_seed_policy != "canonical_group_sha256_v1":
            raise ValueError("stage1_seed_policy must be canonical_group_sha256_v1")
        scientific_scalars = {
            "outer_folds": o.outer_folds,
            "review_rounds": o.review_rounds,
            "initial_training_partitions": o.initial_training_partitions,
            "interaction_inner_folds": o.interaction_inner_folds,
            "tfidf_nested_calibration_folds": (o.tfidf_nested_calibration_folds),
            "seed": o.seed,
            "repeated_character_threshold": o.repeated_character_threshold,
        }
        missing_scientific_scalars = sorted(
            name for name, value in scientific_scalars.items() if value is None
        )
        if missing_scientific_scalars:
            raise ValueError(
                "scientific fold, preprocessing, and seed values must be "
                "explicitly configured: " + ", ".join(missing_scientific_scalars)
            )
        if (
            int(o.outer_folds) < 2
            or int(o.review_rounds) < 1
            or int(o.initial_training_partitions) < 1
            or int(o.interaction_inner_folds) < 2
            or int(o.tfidf_nested_calibration_folds) < 2
            or int(o.repeated_character_threshold) < 1
            or int(o.seed) < 0
        ):
            raise ValueError(
                "configured fold, initial-partition, review, threshold, or "
                "seed settings are invalid"
            )
        if o.outcome_type != "binary":
            raise ValueError("this production workflow currently requires binary outcomes")
        if o.empty_text_policy != "marker" or o.repeated_character_policy != "marker":
            raise ValueError("production text preparation requires both neutral marker policies")
        if o.source_text_temporally_valid_by_design is not True:
            raise ValueError(
                "source_text_temporally_valid_by_design must be explicitly true "
                "for the v1 decision-time text estimand"
            )
        if o.tfidf_parallel_backend not in {"threads", "processes"}:
            raise ValueError("unsupported TF-IDF parallel backend")
        if o.storage_backend not in {"posix", "local_posix", "sshfs"}:
            raise ValueError("unsupported portable artifact storage backend")
        if o.cluster_preflight_parquet_compression not in {
            None,
            "none",
            "zstd",
        }:
            raise ValueError(
                "cluster_preflight_parquet_compression must be 'none' or "
                "'zstd'"
            )
        if (
            o.portable_scientific_spec is not None
            and o.cluster_preflight_parquet_compression is None
        ):
            raise ValueError(
                "typed portable Stage 1 requires an explicit deployment "
                "cluster_preflight_parquet_compression"
            )
        if not isinstance(
            o.resource_performance_safety,
            ResourcePerformanceSafetyPolicy,
        ):
            raise TypeError("workflow options require typed " "resource_performance_safety")
        if not isinstance(o.run_control, RunControl):
            raise TypeError("workflow options require typed RunControl")
        if not str(o.runtime_compatibility_class).strip():
            raise ValueError("runtime compatibility class is required")
        if o.device_policy and tuple(o.device_policy) != normalize_device_policy(o.device_policy):
            raise ValueError("device_policy is not canonical")
        if (
            o.run_control.stop_after is not None
            and o.run_control.stop_after not in self._phase_sequence()
        ):
            raise ValueError("stop_after is outside this workflow request")
        if o.legacy_preflight_candidate is not None:
            candidate = Path(o.legacy_preflight_candidate)
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or candidate.name != "cluster_preflight_manifest.json"
            ):
                raise ValueError(
                    "legacy preflight candidate must be one complete "
                    "cluster_preflight_manifest.json file"
                )
        from .final_context_fit_causal_forest_adapter import (
            FixedCausalForestHeadBackend,
        )

        flat_forest_fields = (
            "forest_n_estimators",
            "forest_max_depth",
            "forest_min_samples_leaf",
            "forest_max_features",
            "forest_honest",
            "forest_inference",
            "forest_subforest_size",
            "forest_tune_model",
            "forest_nuisance_n_estimators",
            "forest_nuisance_max_depth",
            "forest_nuisance_min_samples_leaf",
            "forest_nuisance_treatment_max_features",
            "forest_nuisance_outcome_max_features",
            "forest_random_seed",
        )
        if o.portable_scientific_spec is not None:
            if not isinstance(
                o.forest_runtime_config,
                StrictCausalForestRuntimeConfig,
            ):
                raise TypeError("typed portable workflow requires forest_runtime_config")
            populated_legacy_fields = sorted(
                name for name in flat_forest_fields if getattr(o, name) is not None
            )
            if populated_legacy_fields:
                raise ValueError(
                    "typed portable workflow forbids duplicate flat forest "
                    f"fields: {populated_legacy_fields}"
                )
            portable_forest = o.portable_scientific_spec.get("causal_estimator")
            if portable_forest != o.forest_runtime_config.causal_forest.as_dict() or int(
                o.forest_runtime_config.operational.requested_host_cpu_budget
            ) != int(o.cpu_budget):
                raise ValueError(
                    "forest_runtime_config differs from the portable "
                    "scientific/deployment request"
                )
            portable_backend = FixedCausalForestHeadBackend(runtime_config=o.forest_runtime_config)
            if portable_backend.identity().get("configuration_mode") != (
                "portable_strict_runtime_config_v1"
            ):
                raise RuntimeError("typed portable workflow selected a legacy forest backend")
        else:
            if o.forest_runtime_config is not None:
                raise ValueError(
                    "non-portable compatibility options cannot supply the "
                    "portable forest runtime"
                )
            if any(
                getattr(o, name) is None
                for name in (
                    "forest_n_estimators",
                    "forest_min_samples_leaf",
                    "forest_max_features",
                    "forest_honest",
                    "forest_inference",
                    "forest_subforest_size",
                    "forest_tune_model",
                    "forest_nuisance_n_estimators",
                    "forest_nuisance_min_samples_leaf",
                    "forest_random_seed",
                )
            ):
                raise ValueError(
                    "legacy final estimator requires every non-nullable flat "
                    "strict-forest setting"
                )
            FixedCausalForestHeadBackend(
                n_estimators=o.forest_n_estimators,
                max_depth=o.forest_max_depth,
                min_samples_leaf=o.forest_min_samples_leaf,
                max_features=o.forest_max_features,
                honest=o.forest_honest,
                inference=o.forest_inference,
                subforest_size=o.forest_subforest_size,
                tune_model=o.forest_tune_model,
                nuisance_n_estimators=o.forest_nuisance_n_estimators,
                nuisance_max_depth=o.forest_nuisance_max_depth,
                nuisance_min_samples_leaf=(o.forest_nuisance_min_samples_leaf),
                nuisance_treatment_max_features=(o.forest_nuisance_treatment_max_features),
                nuisance_outcome_max_features=(o.forest_nuisance_outcome_max_features),
                random_state=o.forest_random_seed,
                n_jobs=o.cpu_budget,
            )
        from ..extraction.complete_paged import CompletePagingGeometry

        if not isinstance(o.stage2_prompt_protocol, Stage2PromptProtocolSpec):
            raise ValueError(
                "stage2_prompt_protocol must be explicitly configured with "
                "every Stage 2 scientific bound"
            )
        if not isinstance(
            o.post_extraction_causal_review,
            PostExtractionCausalReviewSpec,
        ):
            raise ValueError(
                "post_extraction_causal_review must explicitly configure every "
                "causal-review threshold"
            )
        if o.max_candidate_variables is None or not (1 <= int(o.max_candidate_variables) <= 20):
            raise ValueError("max_candidate_variables must be explicitly configured in [1, 20]")
        if any(
            value is None
            for value in (
                o.complete_page_core_chars,
                o.complete_page_context_chars,
                o.complete_page_max_chars,
                o.complete_reconciliation_fan_in,
            )
        ):
            raise ValueError("complete-page geometry must be explicitly configured")
        CompletePagingGeometry(
            core_chars=int(o.complete_page_core_chars),
            context_chars=int(o.complete_page_context_chars),
            max_page_chars=int(o.complete_page_max_chars),
        )
        if int(o.complete_reconciliation_fan_in) < 2:
            raise ValueError("complete_reconciliation_fan_in must be at least two")
        if any(
            value is None
            for value in (
                o.embedding_chunk_size_words,
                o.embedding_chunk_overlap_words,
                o.embedding_max_chunks,
                o.embedding_chunk_selection,
                o.embedding_normalize,
                o.embedding_encoder,
                o.embedding_batch_size,
            )
        ):
            raise ValueError("embedding chunk geometry must be explicitly configured")
        if (
            int(o.embedding_chunk_size_words) < 1
            or not 0 <= int(o.embedding_chunk_overlap_words) < int(o.embedding_chunk_size_words)
            or int(o.embedding_max_chunks) < 1
            or int(o.embedding_batch_size) < 1
            or str(o.embedding_chunk_selection) not in {"first", "last"}
            or not isinstance(o.embedding_normalize, bool)
            or not isinstance(o.embedding_encoder, SentenceEmbeddingEncoderSpec)
            or (o.embedding_max_seq_length is not None and int(o.embedding_max_seq_length) < 1)
        ):
            raise ValueError("configured embedding chunk geometry is invalid")
        if o.stage1_only:
            if o.evaluate_oracle_posthoc:
                raise ValueError("Stage-1-only mode cannot request oracle evaluation")
        elif not (
            isinstance(o.endpoint, str)
            and o.endpoint.strip()
            and isinstance(o.model_name, str)
            and o.model_name.strip()
            and isinstance(o.stage2_tokenizer_locator, Path)
        ):
            raise ValueError(
                "the full workflow requires one endpoint, exact model name, and "
                "stage2_tokenizer_locator for prompt nontruncation proofs"
            )
        auxiliary_import_values = (
            o.embedding_cache_import_source_prepared_path,
            o.embedding_cache_import_source_preparation_manifest_path,
        )
        if o.embedding_cache_import is None and any(
            value is not None for value in auxiliary_import_values
        ):
            raise ValueError(
                "embedding-cache source-preparation options require " "--embedding-cache-import"
            )
        if sum(value is not None for value in auxiliary_import_values) == 1:
            raise ValueError(
                "embedding-cache import source preparation requires both its cohort " "and manifest"
            )
        if o.evaluate_oracle_posthoc and not all(
            (o.oracle_dataset_path, o.oracle_unit_id_column, o.oracle_ite_column)
        ):
            raise ValueError("post-hoc oracle evaluation requires its dataset, ID, and ITE column")
        if o.source_snapshot_root is not None:
            from .production_source_snapshot import validate_production_source_snapshot

            snapshot = validate_production_source_snapshot(o.source_snapshot_root)
            loaded_root = Path(__file__).resolve().parents[2]
            marker = os.environ.get(SOURCE_SNAPSHOT_EXECUTION_ENV)
            if loaded_root != snapshot.root or marker != snapshot.content_sha256:
                raise ValueError(
                    "source_snapshot_root requires execution from that authenticated "
                    "snapshot; use the public CLI so it can re-exec safely"
                )

    def _legacy_migration_control_root(self, phase: str) -> Path:
        if phase not in {"input_preparation", "embedding_cache"}:
            raise ValueError("unsupported legacy migration phase")
        work_root = Path(self.options.work_root)
        return work_root.parent / f".{work_root.name}.legacy_checkpoint_sources" / phase

    def _resolve_requested_checkpoint_sources(
        self,
        *,
        expected_compatibilities_by_phase: Mapping[str, Mapping[str, Any]],
        embedding_model_builder_tree_sha256: str,
    ) -> tuple[
        list[ValidatedPortableArtifact],
        list[dict[str, Any]],
        dict[str, Any] | None,
    ]:
        """Authenticate portable nodes and narrowly migrate legacy terminals.

        Legacy discovery is deliberately not recursive.  The public option
        must name an exact terminal ``complete_manifest.json`` for preparation
        or cache; an attempt directory, marker, partial preflight, or loose
        model file is never interpreted as a checkpoint.
        """

        from .legacy_checkpoint_migration import (
            derive_legacy_embedding_cache_migration_expectation,
            derive_legacy_prepared_migration_expectation,
            migrate_legacy_terminal_phase_reference,
            validate_legacy_preflight_manifest,
            validate_legacy_terminal_phase_manifest,
            validate_migrated_legacy_terminal_phase_reference,
        )

        compatibilities = {
            phase: ArtifactCompatibility(**dict(value))
            for phase, value in expected_compatibilities_by_phase.items()
        }
        if set(compatibilities) != set(PORTABLE_CHECKPOINT_PHASE_SPECS):
            raise ValueError(
                "checkpoint compatibility mapping does not cover every producer phase"
            )
        portable: list[ValidatedPortableArtifact] = []
        payload_authentication_cache: dict[
            str, tuple[tuple[int, ...], str, int]
        ] = {}
        legacy: dict[str, tuple[Path, Mapping[str, Any]]] = {}
        selected_preflight: tuple[
            Path,
            Mapping[str, Any],
            str,
            str,
            Path | None,
        ] | None = None
        for raw_attestation in (
            self.options.run_control
            .trust_prior_adoption_attestations
        ):
            prior_path = Path(raw_attestation)
            if prior_path.is_symlink() or not prior_path.is_file():
                raise ValueError(
                    "operator-trusted checkpoint reuse requires an exact "
                    "prior adoption attestation file"
                )
            prior = _read_json_object(
                prior_path,
                label="operator-trusted prior adoption attestation selector",
            )
            producer_locator = prior.get("producer_locator")
            if (
                not isinstance(producer_locator, str)
                or not Path(producer_locator).is_absolute()
            ):
                raise ValueError(
                    "operator-trusted prior adoption attestation lacks its "
                    "producer locator"
                )
            locator_path = Path(producer_locator)
            if locator_path.name != "artifact_locator.json":
                raise ValueError(
                    "operator-trusted prior adoption attestation names an "
                    "invalid producer locator"
                )
            trusted = validate_operator_trusted_portable_artifact(
                source=locator_path.parent,
                prior_attestation_path=prior_path,
            )
            artifact = trusted.artifact
            if artifact.manifest["artifact_kind"] not in {
                "prepared_cohort",
                "embedding_cache",
            }:
                raise ValueError(
                    "operator-trusted reuse is narrowly limited to prepared "
                    "cohort and embedding-cache checkpoints"
                )
            if (
                artifact.artifact_id
                in self._operator_trusted_checkpoint_handles
            ):
                raise ValueError(
                    "operator-trusted checkpoint was selected more than once"
                )
            self._operator_trusted_checkpoint_handles[
                artifact.artifact_id
            ] = trusted
            portable.append(artifact)
        for raw_source in self.options.run_control.adopt_checkpoints:
            source = Path(raw_source)
            if source.is_symlink():
                raise ValueError("checkpoint adoption source cannot be a symlink")
            if source.name == "cluster_preflight_manifest.json":
                if not source.is_file():
                    raise ValueError(
                        "legacy preflight adoption requires an exact complete "
                        "cluster_preflight_manifest.json file"
                    )
                if selected_preflight is not None:
                    raise ValueError("legacy preflight candidate was selected more than once")
                (
                    validated_preflight,
                    candidate_kind,
                    state_manifest,
                ) = _validate_preflight_candidate_selector(
                    source,
                )
                selected_preflight = (
                    source.resolve(strict=True),
                    validated_preflight,
                    "adopt_checkpoint",
                    candidate_kind,
                    state_manifest,
                )
                continue
            if source.name == "complete_manifest.json":
                if not source.is_file():
                    raise ValueError(
                        "legacy checkpoint adoption requires an exact terminal "
                        "complete_manifest.json file"
                    )
                control = _read_json_object(
                    source,
                    label="legacy terminal phase manifest selector",
                )
                phase = control.get("phase")
                if phase not in {"input_preparation", "embedding_cache"}:
                    raise ValueError(
                        "legacy adopt-checkpoint supports only terminal "
                        "input_preparation and embedding_cache manifests; use "
                        "--legacy-preflight-candidate for a complete V4 preflight"
                    )
                if str(phase) in legacy:
                    raise ValueError(f"legacy checkpoint adoption has multiple {phase} candidates")
                validated = validate_legacy_terminal_phase_manifest(
                    source,
                    expected_phase=str(phase),
                )
                legacy[str(phase)] = (source.resolve(strict=True), validated)
                continue
            if source.is_file() and source.name != MANIFEST_NAME:
                raise ValueError(
                    "checkpoint adoption files must be artifact_manifest.json "
                    "or a supported legacy complete_manifest.json"
                )
            portable.append(
                validate_portable_artifact(
                    source,
                    payload_authentication_cache=payload_authentication_cache,
                )
            )

        artifact_ids = [artifact.artifact_id for artifact in portable]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError(
                "checkpoint adoption selected the same artifact through "
                "multiple trust paths"
            )

        if self.options.legacy_preflight_candidate is not None:
            if selected_preflight is not None:
                raise ValueError(
                    "legacy preflight candidate cannot be selected through "
                    "both --adopt-checkpoint and --legacy-preflight-candidate"
                )
            alias_path = Path(self.options.legacy_preflight_candidate).resolve(strict=True)
            (
                validated_preflight,
                candidate_kind,
                state_manifest,
            ) = _validate_preflight_candidate_selector(alias_path)
            selected_preflight = (
                alias_path,
                validated_preflight,
                "deprecated_legacy_preflight_candidate_alias",
                candidate_kind,
                state_manifest,
            )

        portable_kinds = [str(artifact.manifest["artifact_kind"]) for artifact in portable]
        if "input_preparation" in legacy and "prepared_cohort" in portable_kinds:
            raise ValueError("checkpoint adoption cannot mix portable and legacy prepared cohorts")
        if "embedding_cache" in legacy and "embedding_cache" in portable_kinds:
            raise ValueError("checkpoint adoption cannot mix portable and legacy embedding caches")
        if "embedding_cache" in legacy and "input_preparation" not in legacy:
            raise ValueError(
                "legacy embedding-cache migration requires its legacy prepared "
                "terminal manifest in the same adoption request"
            )

        columns = WorkflowColumns(
            unit_id=self.options.unit_id_column,
            text=self.options.text_column,
            treatment=self.options.treatment_column,
            outcome=self.options.outcome_column,
        )
        preprocessing = TextPreprocessingSpec(
            empty_text_policy=str(self.options.empty_text_policy),
            repeated_character_policy=str(self.options.repeated_character_policy),
            repeated_character_threshold=int(self.options.repeated_character_threshold),
            source_text_temporally_valid_by_design=bool(
                self.options.source_text_temporally_valid_by_design
            ),
        )
        migration_records: list[dict[str, Any]] = []
        prepared_expectation = None
        migrated_prepared: ValidatedPortableArtifact | None = None

        for phase in ("input_preparation", "embedding_cache"):
            candidate = legacy.get(phase)
            if candidate is None:
                continue
            manifest_path, validated_legacy = candidate
            spec = PORTABLE_CHECKPOINT_PHASE_SPECS[phase]
            compatibility = compatibilities[phase]
            if phase == "input_preparation":
                prepared_expectation = derive_legacy_prepared_migration_expectation(
                    manifest_path=manifest_path,
                    current_dataset_path=self.options.dataset_path,
                    columns=columns,
                    preprocessing=preprocessing,
                    compatibility=compatibility,
                )
                typed_expectation = prepared_expectation
                upstream_ids: tuple[str, ...] = ()
                upstream_prepared = None
            else:
                if prepared_expectation is None or migrated_prepared is None:
                    raise RuntimeError("legacy cache migration lost its prepared dependency")
                typed_expectation = derive_legacy_embedding_cache_migration_expectation(
                    manifest_path=manifest_path,
                    prepared_expectation=prepared_expectation,
                    upstream_prepared_artifact=migrated_prepared,
                    embedding_model_name=self.options.embedding_model_name,
                    embedding_model_tree_sha256=(embedding_model_builder_tree_sha256),
                    chunk_configuration=self._embedding_chunk_configuration(),
                )
                upstream_ids = (migrated_prepared.artifact_id,)
                upstream_prepared = migrated_prepared

            control_root = self._legacy_migration_control_root(phase)
            if control_root.exists() or control_root.is_symlink():
                if control_root.is_symlink() or not control_root.is_dir():
                    raise ValueError("existing legacy migration control is not a real directory")
                artifact = validate_portable_artifact(
                    control_root,
                    expected_kind=str(spec["artifact_kind"]),
                    expected_compatibility_key=compatibility.key,
                    expected_upstream_artifact_ids=upstream_ids,
                )
                artifact = validate_migrated_legacy_terminal_phase_reference(
                    artifact=artifact,
                    manifest_path=manifest_path,
                    expected_phase=phase,
                    artifact_kind=str(spec["artifact_kind"]),
                    artifact_schema=str(spec["artifact_schema"]),
                    compatibility=compatibility,
                    upstream_artifact_ids=upstream_ids,
                    typed_expectation=typed_expectation,
                    upstream_prepared_artifact=upstream_prepared,
                )
            else:
                control_root.parent.mkdir(parents=True, exist_ok=True)
                artifact = migrate_legacy_terminal_phase_reference(
                    manifest_path=manifest_path,
                    expected_phase=phase,
                    control_root=control_root,
                    artifact_kind=str(spec["artifact_kind"]),
                    artifact_schema=str(spec["artifact_schema"]),
                    compatibility=compatibility,
                    upstream_artifact_ids=upstream_ids,
                    typed_expectation=typed_expectation,
                    upstream_prepared_artifact=upstream_prepared,
                )
            if phase == "input_preparation":
                migrated_prepared = artifact
            manifest_sha256, manifest_size = stable_file_sha256(manifest_path)
            migration_records.append(
                {
                    "phase": phase,
                    "legacy_manifest_path": str(manifest_path),
                    "legacy_manifest_sha256": manifest_sha256,
                    "legacy_manifest_size_bytes": manifest_size,
                    "legacy_manifest_content_sha256": validated_legacy["manifest"][
                        "content_sha256"
                    ],
                    "legacy_request_sha256": validated_legacy["manifest"]["request_sha256"],
                    "typed_expectation_identity": typed_expectation.identity,
                    "migrated_artifact_id": artifact.artifact_id,
                    "portable_control_root": str(artifact.root),
                    "source_tree_mutated": False,
                    "payload_copies_materialized": False,
                }
            )
            portable.append(artifact)
        preflight_identity: dict[str, Any] | None = None
        if selected_preflight is not None:
            (
                manifest_path,
                validated_preflight,
                selection_source,
                candidate_kind,
                state_manifest,
            ) = selected_preflight
            manifest_sha256, manifest_size = stable_file_sha256(manifest_path)
            preflight_identity = {
                "selection_source": selection_source,
                "candidate_kind": candidate_kind,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "manifest_size_bytes": manifest_size,
                "manifest_content_sha256": validated_preflight["manifest"]["content_sha256"],
                "registered_payloads": {
                    name: {
                        "path": row["path"],
                        "sha256": row["sha256"],
                        "size_bytes": row["size_bytes"],
                    }
                    for name, row in validated_preflight["payloads"].items()
                },
                "registered_payload_bytes_authenticated_during_request": False,
                "direct_reuse_allowed": (
                    candidate_kind
                    in {"portable_v2", "reusable_v1"}
                ),
                "state_bundle_manifest_path": (
                    None
                    if state_manifest is None
                    else str(state_manifest)
                ),
                "state_bundle_manifest_sha256": (
                    None
                    if state_manifest is None
                    else stable_file_sha256(state_manifest)[0]
                ),
                "prepared_context_manifest_path": (
                    validated_preflight.get(
                        "prepared_context_manifest_path"
                    )
                ),
                "prepared_context_manifest_sha256": (
                    None
                    if validated_preflight.get(
                        "prepared_context_manifest_path"
                    )
                    is None
                    else stable_file_sha256(
                        Path(
                            str(
                                validated_preflight[
                                    "prepared_context_manifest_path"
                                ]
                            )
                        )
                    )[0]
                ),
            }
        return portable, migration_records, preflight_identity

    def _request_body(self) -> dict[str, Any]:
        self._adopted_artifact_handles.clear()
        self._operator_trusted_checkpoint_handles.clear()
        identity_memo = self._scientific_identity_memo
        values = json.loads(json.dumps(asdict(self.options), default=str))
        values.pop("run_control")
        if self.options.endpoint is not None:
            from ..extraction.llm_routing import (
                resolve_stage2_endpoint_authentication,
                resolve_stage2_endpoint_transport,
                validate_stage2_endpoint_runtime_configuration,
            )

            endpoint_auth = resolve_stage2_endpoint_authentication()
            endpoint_transport = resolve_stage2_endpoint_transport()
            validate_stage2_endpoint_runtime_configuration(
                authentication=endpoint_auth,
                transport=endpoint_transport,
            )
            if endpoint_auth.identity["mode"] != "none":
                values["stage2_endpoint_authentication"] = dict(
                    endpoint_auth.identity
                )
            if endpoint_transport.mode != "vllm":
                values["stage2_endpoint_transport"] = dict(
                    endpoint_transport.identity
                )
        values["schema_version"] = WORKFLOW_SCHEMA
        # ``dataclasses.asdict`` exposes internal tuple-backed fields from the
        # generation parameter objects and omits the policy's wire schema.
        # Persist the exact closed protocol representation used by requests.
        values["stage2_prompt_protocol"] = (
            self.options.stage2_prompt_protocol.as_dict()
        )
        generation_policy = self.options.stage2_prompt_protocol.generation_policy
        values["transport_retries"] = (
            generation_policy.interpret_architecture_chunk.transport_max_retries
        )
        values["schema_repairs"] = generation_policy.feature_proposal_review.schema_repair_attempts
        values["extraction_context_strategy"] = "complete_paged_v1"
        values["final_estimator"] = "strict_outer_honest_final_context_fit_causal_forest_v2"
        values["phase_sequence"] = list(self._phase_sequence())
        values["expected_granular_checkpoint_plan"] = (
            _derive_expected_granular_checkpoint_plan(
                outer_folds=int(self.options.outer_folds),
                initial_training_partitions=int(
                    self.options.initial_training_partitions
                ),
                review_rounds=int(self.options.review_rounds),
            )
        )
        values["resolved_stage1_gpu_ids"] = list(self.stage1_gpu_ids)
        values["resolved_query_devices"] = list(self.query_devices)
        values["stage1_resource_contract"] = {
            "execution_device_count": self.options.stage1_execution_device_count,
            "scope_workers_per_gpu": self.options.stage1_scope_workers_per_gpu,
            "execution_selection": (
                None
                if self.options.stage1_execution_profile is None
                else self.options.stage1_execution_profile.as_dict()
            ),
            "preflight_workers": self.options.stage1_preflight_workers,
            "preflight_execution_attestation": copy.deepcopy(
                self.options.stage1_preflight_execution_attestation
            ),
            "tfidf_workers": self.options.tfidf_workers,
            "tfidf_parallel_backend": self.options.tfidf_parallel_backend,
            "seed": self.options.seed,
            "scope_seed_policy": self.options.stage1_seed_policy,
            "exclusive_gpu_preflight_required": bool(self.stage1_gpu_ids),
        }
        values["source_sha256"] = _memoized_scientific_file_digest(
            self.options.dataset_path,
            identity_memo=identity_memo,
        )[0]
        values["stage1_profile_sha256"] = (
            _memoized_scientific_file_digest(
                self.options.stage1_profile_path,
                identity_memo=identity_memo,
            )[0]
        )
        values["query_profile_sha256"] = (
            _memoized_scientific_file_digest(
                self.options.query_profile_path,
                identity_memo=identity_memo,
            )[0]
        )
        values["stage1_profile_scientific_identity"] = scientific_profile_file_identity(
            self.options.stage1_profile_path,
            profile_kind="stage1",
        )
        values["query_profile_scientific_identity"] = scientific_profile_file_identity(
            self.options.query_profile_path,
            profile_kind="neural_query",
        )
        if self.options.scientific_spec_path is not None:
            values["scientific_spec_source_sha256"] = (
                _memoized_scientific_file_digest(
                    self.options.scientific_spec_path,
                    identity_memo=identity_memo,
                )[0]
            )
        if self.options.deployment_profile_path is not None:
            values["deployment_profile_source_sha256"] = (
                _memoized_scientific_file_digest(
                    self.options.deployment_profile_path,
                    identity_memo=identity_memo,
                )[0]
            )
        imported_embedding_cache = self.options.embedding_cache_import is not None
        values["embedding_model_revalidation_policy"] = (
            AUTHENTICATED_DIRECTORY_TREE_POLICY
        )
        values["embedding_model_tree"] = _stable_path_identity(
            self.options.embedding_local_model_path,
            reuse_process_authenticated_tree=True,
        )
        values["embedding_model_builder_tree_sha256"] = _embedding_builder_tree_sha256(
            root=self.options.embedding_local_model_path,
            workflow_tree_identity=values["embedding_model_tree"],
        )
        values["htr_model_tree"] = _stable_path_identity(
            self.options.htr_local_model_path,
        )
        if self.options.stage2_tokenizer_locator is not None:
            values["stage2_tokenizer_tree"] = _stable_path_identity(
                self.options.stage2_tokenizer_locator,
            )
        if imported_embedding_cache:
            source_prepared, source_manifest = self._resolved_cache_import_sources()
            values["embedding_cache_import_inputs"] = {
                "cache": _stable_path_identity(
                    self.options.embedding_cache_import,
                    reuse_process_authenticated_tree=True,
                ),
                "prepared_cohort": _stable_path_identity(source_prepared),
                "preparation_manifest": _stable_path_identity(source_manifest),
            }
        if self.options.source_snapshot_root is not None:
            from .production_source_snapshot import validate_production_source_snapshot

            values["source_snapshot"] = validate_production_source_snapshot(
                self.options.source_snapshot_root
            ).as_dict()
        values["integration_hooks"] = {
            "embedding_cache": _hook_identity(
                self.hooks.embedding_cache,
                identity_memo=identity_memo,
            ),
            "stage1_preflight": _hook_identity(
                self.hooks.stage1_preflight,
                identity_memo=identity_memo,
            ),
            "stage1_modeling": _hook_identity(
                self.hooks.stage1_modeling,
                identity_memo=identity_memo,
            ),
            "role_neutral_stage1": _role_neutral_stage1_integration_identity(
                self.hooks.role_neutral_stage1,
                identity_memo=identity_memo,
            ),
        }
        values["phase_overrides"] = {
            phase: _hook_identity(
                self.phase_overrides.get(phase),
                identity_memo=identity_memo,
            )
            for phase in self._phase_sequence()
        }
        phase_code_records = _phase_transitive_producer_code_records(
            workflow_type=type(self),
            integration_hooks=values["integration_hooks"],
            phase_overrides=values["phase_overrides"],
            identity_memo=identity_memo,
        )
        values["phase_transitive_producer_code"] = phase_code_records
        values["phase_producer_code_identities"] = {
            phase: record["content_sha256"]
            for phase, record in phase_code_records.items()
        }
        repository_root = Path(__file__).resolve().parents[2]
        implementation_files: dict[str, str] = {
            str(Path(__file__).resolve()): (
                identity_memo.file_digest(
                    Path(__file__).resolve(),
                    repository_root=repository_root,
                )[0]
            )
        }

        def register_implementation_file(
            path: Path,
            digest: str,
        ) -> None:
            key = str(path)
            prior = implementation_files.setdefault(key, str(digest))
            if prior != str(digest):
                raise RuntimeError(
                    "workflow implementation changed while its transitive "
                    "producer identity was being constructed"
                )

        for record in phase_code_records.values():
            for row in (
                *record["transitive_source_inventory"],
                *record["dependency_lock_inventory"],
            ):
                path = (
                    repository_root / str(row["relative_path"])
                ).resolve(strict=True)
                register_implementation_file(
                    path,
                    str(row["sha256"]),
                )
        for collection_name in (
            "integration_hooks",
            "phase_overrides",
        ):
            for row in _repository_import_closure_rows(
                values[collection_name]
            ):
                path = (
                    repository_root / str(row["relative_path"])
                ).resolve(strict=True)
                register_implementation_file(
                    path,
                    str(row["sha256"]),
                )
        values["implementation_files"] = dict(
            sorted(implementation_files.items())
        )
        values["stage1_recovery_contract"] = {
            "scope_attempt_root": str(
                (self.options.work_root / "recovery" / "stage1_scope_attempts").resolve()
            ),
            "scope_progress_path": str(
                (self.options.work_root / "recovery" / "stage1_scope_progress.json").resolve()
            ),
            "scope_reuse_policy": "individually_sealed_matching_scope_attempts_only_v1",
        }
        if self.options.portable_scientific_spec is not None:
            scientific_settings = copy.deepcopy(dict(self.options.portable_scientific_spec))
        else:
            scientific_settings = {
                "schema_version": "compiled_direct_flag_scientific_workflow_v1",
                "columns": {
                    "unit_id": self.options.unit_id_column,
                    "text": self.options.text_column,
                    "treatment": self.options.treatment_column,
                    "outcome": self.options.outcome_column,
                },
                "clinical_question": self.options.clinical_question,
                "estimand": BINARY_PROBABILITY_DIFFERENCE,
                "preprocessing": {
                    "empty_text_policy": self.options.empty_text_policy,
                    "repeated_character_policy": (self.options.repeated_character_policy),
                    "repeated_character_threshold": (self.options.repeated_character_threshold),
                    "source_text_temporally_valid_by_design": (
                        self.options.source_text_temporally_valid_by_design
                    ),
                },
                "folds": {
                    "outer_folds": self.options.outer_folds,
                    "review_rounds": self.options.review_rounds,
                    "initial_training_partitions": (self.options.initial_training_partitions),
                    "interaction_inner_folds": (self.options.interaction_inner_folds),
                    "tfidf_nested_calibration_folds": (self.options.tfidf_nested_calibration_folds),
                },
                "causal_estimator": {
                    "implementation": ("strict_outer_honest_final_context_fit_causal_forest_v2"),
                    "n_estimators": self.options.forest_n_estimators,
                    "max_depth": self.options.forest_max_depth,
                    "min_samples_leaf": self.options.forest_min_samples_leaf,
                    "max_features": self.options.forest_max_features,
                    "honest": self.options.forest_honest,
                    "inference": self.options.forest_inference,
                    "subforest_size": self.options.forest_subforest_size,
                    "tune_model": self.options.forest_tune_model,
                    "nuisance_n_estimators": (self.options.forest_nuisance_n_estimators),
                    "nuisance_max_depth": (self.options.forest_nuisance_max_depth),
                    "nuisance_min_samples_leaf": (self.options.forest_nuisance_min_samples_leaf),
                    "nuisance_treatment_max_features": (
                        self.options.forest_nuisance_treatment_max_features
                    ),
                    "nuisance_outcome_max_features": (
                        self.options.forest_nuisance_outcome_max_features
                    ),
                    "random_seed": self.options.forest_random_seed,
                },
                "text_windows": {
                    "complete_page_core_chars": (self.options.complete_page_core_chars),
                    "complete_page_context_chars": (self.options.complete_page_context_chars),
                    "complete_page_max_chars": (self.options.complete_page_max_chars),
                    "reconciliation_fan_in": (self.options.complete_reconciliation_fan_in),
                    "embedding_chunk_size_words": (self.options.embedding_chunk_size_words),
                    "embedding_chunk_overlap_words": (self.options.embedding_chunk_overlap_words),
                    "embedding_max_chunks": self.options.embedding_max_chunks,
                    "embedding_chunk_selection": (self.options.embedding_chunk_selection),
                    "embedding_max_seq_length": (self.options.embedding_max_seq_length),
                    "embedding_normalize": self.options.embedding_normalize,
                    "embedding_encoder": asdict(self.options.embedding_encoder),
                },
                "stage2_prompt_protocol": (self.options.stage2_prompt_protocol.as_dict()),
                "post_extraction_causal_review": (
                    self.options.post_extraction_causal_review.as_dict()
                ),
                "max_candidate_variables": self.options.max_candidate_variables,
                "seed": self.options.seed,
                "seed_policy": self.options.stage1_seed_policy,
            }
        values["portable_typed_workflow"] = (
            self.options.portable_scientific_spec is not None
        )
        scientific_configuration_body = {
            "schema_version": (
                "portable_all_evidence_scientific_configuration_identity_v1"
            ),
            "scientific_settings": scientific_settings,
            "dataset_content_sha256": values["source_sha256"],
            "stage1_profile_scientific_identity": (values["stage1_profile_scientific_identity"]),
            "query_profile_scientific_identity": (values["query_profile_scientific_identity"]),
            "embedding_model": _path_neutral_identity(values["embedding_model_tree"]),
            "htr_model": _path_neutral_identity(values["htr_model_tree"]),
            "stage2_tokenizer": (
                _path_neutral_identity(values["stage2_tokenizer_tree"])
                if isinstance(values.get("stage2_tokenizer_tree"), Mapping)
                else None
            ),
            "embedding_model_name": self.options.embedding_model_name,
            "stage2_model_name": self.options.model_name,
            "runtime_compatibility_class": (self.options.runtime_compatibility_class),
        }
        identity_binding = _bind_workflow_scientific_identity(
            scientific_configuration_body=(
                scientific_configuration_body
            ),
            phase_code_records=phase_code_records,
        )
        values.update(identity_binding)
        workflow_producer_code_identity = str(
            identity_binding["workflow_producer_code_identity"]
        )
        folds_identity = identity_sha256(scientific_settings.get("folds", {}))
        seed_identity = identity_sha256(
            {
                "seed": self.options.seed,
                "seed_policy": self.options.stage1_seed_policy,
            }
        )
        neutral_embedding_model = _path_neutral_identity(values["embedding_model_tree"])
        neutral_htr_model = _path_neutral_identity(values["htr_model_tree"])
        neutral_stage2_tokenizer = (
            _path_neutral_identity(values["stage2_tokenizer_tree"])
            if isinstance(values.get("stage2_tokenizer_tree"), Mapping)
            else None
        )

        def model_content_digest(value: Mapping[str, Any]) -> str:
            registered = value.get("tree_sha256", value.get("sha256"))
            return str(registered) if isinstance(registered, str) else identity_sha256(value)

        expected_model_identities = {
            "embedding_model_tree": model_content_digest(neutral_embedding_model),
            "embedding_model_builder_tree": values["embedding_model_builder_tree_sha256"],
            "htr_model_tree": model_content_digest(neutral_htr_model),
            "stage2_model_name": identity_sha256({"model_name": self.options.model_name}),
        }
        if neutral_stage2_tokenizer is not None:
            expected_model_identities["stage2_tokenizer_tree"] = model_content_digest(
                neutral_stage2_tokenizer
            )
        expected_prompt_identities = dict(scientific_settings.get("prompt_identities") or {})
        expected_checkpoint_compatibility_base = {
            "dataset_identity": values["source_sha256"],
            "split_identity": folds_identity,
            "row_order_identity": identity_sha256(
                {
                    "dataset_content_sha256": values["source_sha256"],
                    "unit_id_column": self.options.unit_id_column,
                    "row_order_policy": (
                        "source_parquet_physical_order_bound_by_complete_file_hash_v1"
                    ),
                }
            ),
            "model_identities": expected_model_identities,
            "prompt_identities": expected_prompt_identities,
            "configuration_identity": values[
                "scientific_configuration_identity"
            ]["scientific_configuration_sha256"],
            "seed_identity": seed_identity,
            "runtime_compatibility_class": (self.options.runtime_compatibility_class),
        }
        expected_checkpoint_compatibilities_by_phase = {
            phase: {
                **expected_checkpoint_compatibility_base,
                "producer_code_identity": values[
                    "phase_producer_code_identities"
                ][phase],
            }
            for phase in PORTABLE_CHECKPOINT_PHASE_SPECS
        }
        expected_checkpoint_compatibility = {
            **expected_checkpoint_compatibility_base,
            "producer_code_identity": workflow_producer_code_identity,
        }
        values["expected_checkpoint_compatibility"] = expected_checkpoint_compatibility
        values["expected_checkpoint_compatibilities_by_phase"] = (
            expected_checkpoint_compatibilities_by_phase
        )
        from .production_stage1_scope_scheduler import (
            Stage1PhysicalFitIdentity,
        )

        architecture_profiles = scientific_settings.get(
            "architecture_profiles"
        )
        if isinstance(architecture_profiles, Mapping):
            architecture_identity = identity_sha256(
                {
                    "schema_version": "all_ten_architecture_profiles_v1",
                    "family_order": list(EVIDENCE_FAMILIES),
                    "architecture_profiles": {
                        family: copy.deepcopy(
                            dict(architecture_profiles[family])
                        )
                        for family in EVIDENCE_FAMILIES
                    },
                }
            )
        else:
            architecture_identity = identity_sha256(
                {
                    "schema_version": (
                        "compiled_legacy_all_ten_architecture_profiles_v1"
                    ),
                    "stage1_profile": values[
                        "stage1_profile_scientific_identity"
                    ],
                    "query_profile": values[
                        "query_profile_scientific_identity"
                    ],
                }
            )
        values["stage1_physical_fit_identity"] = Stage1PhysicalFitIdentity(
            architecture_identity=architecture_identity,
            target="all_ten_stage1_context_fit_v1",
            scientific_configuration_identity=values[
                "scientific_configuration_identity"
            ]["scientific_configuration_sha256"],
            producer_identity=(
                ROLE_NEUTRAL_STAGE1_COMPONENT_PLAN_NAMESPACE_IDENTITY
            ),
            runtime_compatibility_class=(
                self.options.runtime_compatibility_class
            ),
        ).as_dict()
        (
            validated_adoptions,
            legacy_migration_records,
            legacy_preflight_identity,
        ) = self._resolve_requested_checkpoint_sources(
            expected_compatibilities_by_phase=(
                expected_checkpoint_compatibilities_by_phase
            ),
            embedding_model_builder_tree_sha256=values["embedding_model_builder_tree_sha256"],
        )
        if legacy_preflight_identity is not None:
            values["legacy_preflight_candidate_identity"] = legacy_preflight_identity
        validated_adoptions_by_id = {
            artifact.artifact_id: artifact
            for artifact in validated_adoptions
        }
        trusted_legacy_projection_proofs: dict[
            str, Mapping[str, Any]
        ] = {}
        for artifact in validated_adoptions:
            if (
                artifact.artifact_id
                in self._operator_trusted_checkpoint_handles
            ):
                trusted_legacy_projection_proofs[
                    artifact.artifact_id
                ] = (
                    _operator_trusted_legacy_phase_projection_proof(
                        artifact=artifact,
                        request=values,
                        adopted_artifacts=validated_adoptions_by_id,
                    )
                )
        for artifact in validated_adoptions:
            artifact_kind = str(artifact.manifest["artifact_kind"])
            compatibility_phase = (
                CHECKPOINT_COMPATIBILITY_PHASE_BY_ARTIFACT_KIND.get(
                    artifact_kind
                )
            )
            if compatibility_phase is None:
                raise ValueError(
                    "adopted checkpoint kind has no producer compatibility domain"
                )
            expected_artifact_compatibility = (
                expected_checkpoint_compatibilities_by_phase[
                    compatibility_phase
                ]
            )
            trusted_checkpoint = (
                self._operator_trusted_checkpoint_handles.get(
                    artifact.artifact_id
                )
            )
            compatibility_record = {
                "adoption_validation_policy": (
                    None
                    if trusted_checkpoint is None
                    else OPERATOR_TRUSTED_VALIDATION_POLICY
                ),
                "prior_adoption_attestation_path": (
                    None
                    if trusted_checkpoint is None
                    else str(
                        trusted_checkpoint.prior_attestation_path
                    )
                ),
                "payload_bytes_reauthenticated": (
                    trusted_checkpoint is None
                ),
                "legacy_phase_compatibility_projection_proof": (
                    trusted_legacy_projection_proofs.get(
                        artifact.artifact_id
                    )
                ),
            }
            if not _adopted_compatibility_matches_request(
                artifact=artifact,
                expected=expected_artifact_compatibility,
                record=compatibility_record,
            ):
                raise ValueError(
                    "adopted checkpoint is incompatible with the configured "
                    "dataset, splits, models, prompts, scientific settings, "
                    "seed policy, producer code, or runtime class"
                )
            if (
                self.options.portable_scientific_spec is not None
                and artifact.manifest.get("artifact_kind") == "stage1_handoff"
            ):
                materialized = materialize_portable_phase(
                    artifact,
                    expected_phase="stage1_modeling",
                )
                result = materialized.get("result")
                if not isinstance(result, Mapping):
                    raise ValueError("adopted portable Stage 1 handoff has no phase result")
                _validate_portable_role_neutral_stage1_phase_result(result)
        phase_artifact_ids = _validate_adopted_checkpoint_graph(
            validated_adoptions,
            allowed_phases=self._phase_sequence(),
            expected_granular_checkpoint_plan=values[
                "expected_granular_checkpoint_plan"
            ],
            expected_stage1_physical_fit_identity=values[
                "stage1_physical_fit_identity"
            ],
            expected_global_seed=int(self.options.seed),
            require_prepared_stage1_context=bool(
                values["portable_typed_workflow"]
            ),
        )
        artifact_phases = {artifact_id: phase for phase, artifact_id in phase_artifact_ids.items()}
        adopted: list[dict[str, Any]] = []
        adoption_locators: list[str] = []
        for artifact in sorted(
            validated_adoptions,
            key=lambda value: value.artifact_id,
        ):
            substituted_phase = artifact_phases.get(artifact.artifact_id)
            trusted_checkpoint = (
                self._operator_trusted_checkpoint_handles.get(
                    artifact.artifact_id
                )
            )
            if substituted_phase == "stage1_preflight":
                self._require_adopted_preflight_storage_compatibility(
                    artifact
                )
            adopted.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "artifact_kind": artifact.manifest["artifact_kind"],
                    "compatibility_key": artifact.compatibility_key,
                    "upstream_artifact_ids": list(artifact.manifest["upstream_artifact_ids"]),
                    "substituted_phase": substituted_phase,
                    "compatibility_phase": (
                        CHECKPOINT_COMPATIBILITY_PHASE_BY_ARTIFACT_KIND[
                            str(artifact.manifest["artifact_kind"])
                        ]
                    ),
                    "artifact_metadata": copy.deepcopy(
                        dict(artifact.artifact_metadata)
                    ),
                    "adoption_validation_policy": (
                        None
                        if trusted_checkpoint is None
                        else OPERATOR_TRUSTED_VALIDATION_POLICY
                    ),
                    "prior_adoption_attestation_path": (
                        None
                        if trusted_checkpoint is None
                        else str(
                            trusted_checkpoint.prior_attestation_path
                        )
                    ),
                    "payload_bytes_reauthenticated": (
                        trusted_checkpoint is None
                    ),
                    "legacy_phase_compatibility_projection_proof": (
                        trusted_legacy_projection_proofs.get(
                            artifact.artifact_id
                        )
                    ),
                }
            )
            self._adopted_artifact_handles[artifact.artifact_id] = artifact
            adoption_locators.append(str(artifact.root))
        values["requested_checkpoint_adoptions"] = adopted
        values["checkpoint_adoption_locators"] = adoption_locators
        values["legacy_checkpoint_migration_sources"] = legacy_migration_records
        try:
            normalized = json.loads(
                json.dumps(
                    values,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TypeError(
                "immutable workflow request must be closed finite JSON"
            ) from exc
        if not isinstance(normalized, dict):
            raise TypeError(
                "immutable workflow request must be one JSON object"
            )
        return normalized

    def _run_control_attestation_root(self) -> Path:
        parent = self.options.work_root / "execution_attestations"
        target = parent / "run_control"
        for candidate in (parent, target):
            if candidate.exists() or candidate.is_symlink():
                if candidate.is_symlink() or not candidate.is_dir():
                    raise ValueError(
                        "run-control attestation root is not a regular directory"
                    )
        target.mkdir(parents=True, exist_ok=True)
        return target.resolve(strict=True)

    def _write_run_control_selection_attestation(
        self,
    ) -> Mapping[str, Any]:
        request_sha256 = self.request.get("request_sha256")
        if not isinstance(request_sha256, str):
            raise RuntimeError(
                "run-control selection requires an initialized request"
            )
        body = {
            "schema_version": WORKFLOW_RUN_CONTROL_SELECTION_SCHEMA,
            "request_sha256": request_sha256,
            "run_control_schema_version": (
                self.options.run_control.schema_version
            ),
            "resume_requested": self.options.run_control.resume,
            "stop_after": self.options.run_control.stop_after,
            "log_level": self.options.run_control.log_level,
            "validation_policy": copy.deepcopy(
                dict(self._validation_policy)
            ),
            "terminal_phase_override_present": (
                "terminal_validation" in self.phase_overrides
            ),
            "scientific_request_identity_affected": False,
            "portable_artifact_identity_affected": False,
            "achievement_requires_separate_fresh_validation_attestation": (
                True
            ),
        }
        record = {**body, "content_sha256": _sha(body)}
        path = (
            self._run_control_attestation_root()
            / f"selection.{record['content_sha256']}.json"
        )
        _write_immutable_json(path, record)
        if (
            _read_json_object(
                path,
                label="run-control selection attestation",
            )
            != record
        ):
            raise RuntimeError(
                "run-control selection attestation changed after writing"
            )
        self._run_control_selection_attestation = record
        self._run_control_selection_attestation_path = path
        return record

    def _write_validation_achievement_attestation(
        self,
        terminal_phase_manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Bind the published terminal phase and its fresh validator report."""

        if "terminal_validation" in self.phase_overrides:
            raise RuntimeError(
                "a terminal phase override cannot satisfy the fresh terminal "
                "audit minimum"
            )
        selection = self._run_control_selection_attestation
        if selection is None:
            raise RuntimeError(
                "validation achievement lacks its run-control selection"
            )
        manifest_body = {
            key: copy.deepcopy(value)
            for key, value in terminal_phase_manifest.items()
            if key != "content_sha256"
        }
        result = terminal_phase_manifest.get("result")
        artifacts = terminal_phase_manifest.get("artifacts")
        if (
            terminal_phase_manifest.get("schema_version")
            != WORKFLOW_PHASE_MANIFEST_SCHEMA
            or terminal_phase_manifest.get("phase")
            != "terminal_validation"
            or terminal_phase_manifest.get("status") != "complete"
            or terminal_phase_manifest.get("request_sha256")
            != self.request["request_sha256"]
            or terminal_phase_manifest.get("content_sha256")
            != _sha(manifest_body)
            or not isinstance(result, Mapping)
            or not isinstance(artifacts, list)
        ):
            raise RuntimeError(
                "validation achievement requires the published terminal "
                "phase manifest"
            )
        complete_manifest_path = self._phase_manifest(
            "terminal_validation"
        ).resolve(strict=True)
        reopened_complete_manifest = _read_json_object(
            complete_manifest_path,
            label="published terminal phase manifest",
        )
        if reopened_complete_manifest != dict(
            terminal_phase_manifest
        ):
            raise RuntimeError(
                "published terminal phase manifest differs from the "
                "process-authenticated result"
            )
        complete_manifest_sha256, complete_manifest_size = (
            stable_file_sha256(complete_manifest_path)
        )
        terminal_files = result.get("terminal_files")
        if (
            not isinstance(terminal_files, list)
            or len(terminal_files) != 1
            or not isinstance(terminal_files[0], str)
        ):
            raise RuntimeError(
                "published terminal validation must register one report"
            )
        matching_artifacts = [
            row
            for row in artifacts
            if isinstance(row, Mapping)
            and row.get("path") == terminal_files[0]
            and row.get("relative_path") == "validation.json"
        ]
        if len(matching_artifacts) != 1:
            raise RuntimeError(
                "published terminal report is not uniquely registered"
            )
        report_artifact = matching_artifacts[0]
        report_path = Path(str(report_artifact["path"]))
        observed_sha256, observed_size = stable_file_sha256(
            report_path.resolve(strict=True)
        )
        if (
            observed_sha256 != report_artifact.get("sha256")
            or observed_size != report_artifact.get("size_bytes")
        ):
            raise RuntimeError(
                "published terminal report changed after phase publication"
            )
        report = _read_json_object(
            report_path,
            label="published fresh terminal validation report",
        )
        result_report = {
            key: copy.deepcopy(value)
            for key, value in result.items()
            if key != "terminal_files"
        }
        if result_report != report:
            raise RuntimeError(
                "published terminal phase result differs from its report"
            )
        report_body = {
            key: copy.deepcopy(value)
            for key, value in report.items()
            if key != "content_sha256"
        }
        checkpoint_validation = report.get(
            "portable_checkpoint_dag_validation"
        )
        prefix_validation = report.get("read_only_prefix_validation")
        checkpoint_content_sha256 = (
            None
            if not isinstance(checkpoint_validation, Mapping)
            else checkpoint_validation.get("content_sha256")
        )
        operator_trusted_reuse = (
            isinstance(checkpoint_validation, Mapping)
            and checkpoint_validation.get(
                "operator_trusted_checkpoint_reuse"
            )
            is True
        )
        if (
            report.get("schema_version")
            != "production_all_evidence_fresh_terminal_validation_report_v2"
            or report.get("execution_completed") is not True
            or report.get("run_validation_status") != "accepted"
            or report.get("global_release_certified") is not False
            or report.get("validated_phase_sequence")
            != list(self._phase_sequence())
            or report.get("live_runner_objects_received") is not False
            or report.get("content_sha256") != _sha(report_body)
            or not isinstance(prefix_validation, Mapping)
            or prefix_validation.get("status") != "accepted"
            or not isinstance(checkpoint_validation, Mapping)
            or checkpoint_validation.get("status") != "accepted"
            or checkpoint_validation.get(
                "fresh_full_byte_validation"
            )
            is operator_trusted_reuse
            or checkpoint_validation.get(
                "payload_bytes_reauthenticated_for_all_adoptions"
            )
            is operator_trusted_reuse
            or checkpoint_validation.get(
                "global_release_certified"
            )
            is not False
            or (
                operator_trusted_reuse
                and not checkpoint_validation.get(
                    "operator_trusted_checkpoint_phases"
                )
            )
            or checkpoint_validation.get(
                "oracle_evaluation_after_frozen_prediction"
            )
            is not True
            or not isinstance(checkpoint_content_sha256, str)
            or len(checkpoint_content_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in checkpoint_content_sha256
            )
        ):
            raise RuntimeError(
                "terminal validation report does not match its declared "
                "fresh or operator-trusted checkpoint policy"
            )
        achieved_minimum = (
            "full_operator_trusted_terminal_audit"
            if operator_trusted_reuse
            else "fresh_terminal_audit"
        )
        body = {
            "schema_version": WORKFLOW_VALIDATION_ACHIEVEMENT_SCHEMA,
            "request_sha256": self.request["request_sha256"],
            "run_control_selection_content_sha256": selection[
                "content_sha256"
            ],
            "requested_minimum": self._validation_policy[
                "requested_minimum"
            ],
            "production_minimum": self._validation_policy[
                "production_minimum"
            ],
            "effective_minimum": self._validation_policy[
                "effective_minimum"
            ],
            "achieved_minimum": achieved_minimum,
            "effective_minimum_satisfied": (
                not operator_trusted_reuse
            ),
            "fresh_path_only_terminal_audit_achieved": (
                not operator_trusted_reuse
            ),
            "operator_trusted_checkpoint_reuse": (
                operator_trusted_reuse
            ),
            "payload_bytes_reauthenticated_for_all_adoptions": (
                not operator_trusted_reuse
            ),
            "fresh_terminal_validation_report_content_sha256": report[
                "content_sha256"
            ],
            "published_terminal_phase_manifest_content_sha256": (
                terminal_phase_manifest["content_sha256"]
            ),
            "published_terminal_complete_manifest_sha256": (
                complete_manifest_sha256
            ),
            "published_terminal_complete_manifest_size_bytes": (
                complete_manifest_size
            ),
            "published_terminal_report_sha256": report_artifact[
                "sha256"
            ],
            "published_terminal_report_size_bytes": report_artifact[
                "size_bytes"
            ],
            "published_checkpoint_dag_validation_content_sha256": (
                checkpoint_content_sha256
            ),
            "terminal_phase_portable_checkpoint_published": False,
            "terminal_phase_portable_checkpoint_artifact_id": None,
            "terminal_phase_portable_checkpoint_content_root": None,
            "terminal_publication_identity_policy": (
                "complete_manifest_report_and_checkpoint_dag_v1"
            ),
            "execution_completed": True,
            "run_validation_status": "accepted",
            "global_release_certified": False,
            "terminal_phase_override_present": False,
            "scientific_request_identity_affected": False,
            "portable_artifact_identity_affected": False,
        }
        record = {**body, "content_sha256": _sha(body)}
        path = (
            self._run_control_attestation_root()
            / f"achievement.{record['content_sha256']}.json"
        )
        _write_immutable_json(path, record)
        if (
            _read_json_object(
                path,
                label="validation achievement attestation",
            )
            != record
        ):
            raise RuntimeError(
                "validation achievement attestation changed after writing"
            )
        self._validation_achievement_attestation = record
        self._validation_achievement_attestation_path = path
        return record

    def _initialize(self) -> None:
        root = self.options.work_root
        request_path = root / "immutable_run_request.json"
        if root.is_symlink():
            raise ValueError("work root cannot be a symlink")
        if root.exists():
            if not self.options.run_control.resume or not request_path.is_file():
                raise ValueError("work root must be fresh unless --resume validates its request")
        else:
            # Legacy reference controls are published beside, never inside,
            # the not-yet-visible immutable run root.
            root.parent.mkdir(parents=True, exist_ok=True)
        body = self._request_body()
        request = {**body, "request_sha256": _sha(body)}
        if root.exists():
            if not self.options.run_control.resume or not request_path.is_file():
                raise ValueError("work root must be fresh unless --resume validates its request")
            existing = _read_json_object(request_path, label="immutable workflow request")
            if existing != request:
                raise ValueError("--resume request differs from the immutable run request")
        else:
            initialization_attempt = Path(
                tempfile.mkdtemp(
                    prefix=f".{root.name}.initialization_attempt_",
                    dir=root.parent,
                )
            )
            staged_request = initialization_attempt / request_path.name
            # Preserve an interrupted initialization attempt for audit, while
            # keeping the requested work root absent and therefore reusable.
            # The root becomes visible only after the immutable request has
            # been durably written and reopened byte-for-byte.
            _atomic_write_json(staged_request, request)
            if (
                _read_json_object(
                    staged_request,
                    label="staged immutable workflow request",
                )
                != request
            ):
                raise RuntimeError("staged immutable workflow request changed")
            attempt_fd = os.open(
                initialization_attempt,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(attempt_fd)
            finally:
                os.close(attempt_fd)
            if root.exists() or root.is_symlink():
                raise ValueError("work root was populated during initialization")
            os.rename(initialization_attempt, root)
            parent_fd = os.open(
                root.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        self.request = request
        self._write_run_control_selection_attestation()
        self._publish_checkpoint_adoptions()
        self._write_progress(status="initialized", completed=(), current_phase=None)

    def _publish_checkpoint_adoptions(self) -> None:
        records = self.request.get("requested_checkpoint_adoptions") or []
        locators = self.request.get("checkpoint_adoption_locators") or []
        for source, expected in zip(locators, records):
            handle = self._adopted_artifact_handles.get(str(expected["artifact_id"]))
            if handle is None:
                raise RuntimeError("checkpoint adoption lost its process-authenticated handle")
            if expected.get("substituted_phase") == "stage1_preflight":
                self._require_adopted_preflight_storage_compatibility(
                    handle
                )
            trusted = self._operator_trusted_checkpoint_handles.get(
                str(expected["artifact_id"])
            )
            if _operator_trusted_adoption_selected(expected):
                if trusted is None:
                    raise RuntimeError(
                        "operator-trusted checkpoint lost its stat-guarded handle"
                    )
                attestation = (
                    adopt_checkpoint_from_prior_full_byte_attestation(
                        source=Path(str(source)),
                        prior_attestation_path=(
                            trusted.prior_attestation_path
                        ),
                        attestation_root=(
                            self.options.work_root
                            / "checkpoint_adoptions"
                        ),
                        consumer_request_sha256=self.request[
                            "request_sha256"
                        ],
                        expected_kind=str(expected["artifact_kind"]),
                        expected_upstream_artifact_ids=tuple(
                            expected["upstream_artifact_ids"]
                        ),
                        trusted_checkpoint=trusted,
                    )
                )
            else:
                if trusted is not None:
                    raise RuntimeError(
                        "operator-trusted handle lacks its immutable request policy"
                    )
                attestation = adopt_checkpoint(
                    source=Path(str(source)),
                    attestation_root=self.options.work_root / "checkpoint_adoptions",
                    consumer_request_sha256=self.request["request_sha256"],
                    expected_kind=str(expected["artifact_kind"]),
                    expected_compatibility_key=str(expected["compatibility_key"]),
                    expected_upstream_artifact_ids=tuple(expected["upstream_artifact_ids"]),
                    validated_artifact=handle,
                )
            if attestation.get("producer_artifact_id") != expected["artifact_id"]:
                raise RuntimeError("checkpoint adoption attestation bound the wrong artifact")

    def _checkpoint_compatibility(self, phase: str) -> ArtifactCompatibility:
        rows = self.request.get("expected_checkpoint_compatibilities_by_phase")
        raw = rows.get(phase) if isinstance(rows, Mapping) else None
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                f"immutable request lacks {phase} checkpoint compatibility"
            )
        return ArtifactCompatibility(**dict(raw))

    def _checkpoint_control_root(self, phase: str) -> Path:
        return self.options.work_root / "portable_checkpoints" / phase

    def _phase_payload_authentication_cache(
        self,
        *,
        phase: str,
        phase_manifest: Mapping[str, Any],
    ) -> dict[str, tuple[tuple[int, ...], str, int]]:
        payload_root = Path(str(phase_manifest["attempt_dir"])).resolve(
            strict=True
        )
        stats = self._phase_payload_stat_inventories.get(phase)
        if stats is None:
            return {}
        output: dict[str, tuple[tuple[int, ...], str, int]] = {}
        for row in phase_manifest.get("artifacts") or ():
            relative = str(row["relative_path"])
            state = stats.get(relative)
            if state is None:
                raise RuntimeError(
                    f"{phase} authenticated stat inventory is incomplete"
                )
            output[str((payload_root / relative).resolve(strict=True))] = (
                tuple(int(value) for value in state),
                str(row["sha256"]),
                int(row["size_bytes"]),
            )
        return output

    def _expected_granular_checkpoint_plan(
        self,
    ) -> Mapping[str, Any]:
        return _validate_expected_granular_checkpoint_plan(
            self.request.get("expected_granular_checkpoint_plan")
        )

    def _publish_granular_checkpoint_node(
        self,
        *,
        phase: str,
        phase_manifest: Mapping[str, Any],
        node_ordinal: int,
        node_key: str,
        artifact_kind: str,
        payload_root: Path,
        payload_files: Sequence[Path],
        upstream_artifact_ids: Sequence[str],
        artifact_metadata: Mapping[str, Any],
        payload_inventory_policy: str,
    ) -> ValidatedPortableArtifact:
        if artifact_kind not in GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS:
            raise ValueError(
                f"unsupported granular checkpoint kind: {artifact_kind}"
            )
        phase_root = Path(str(phase_manifest["attempt_dir"])).resolve(
            strict=True
        )
        root = Path(payload_root).resolve(strict=True)
        try:
            root.relative_to(phase_root)
        except ValueError as exc:
            raise ValueError(
                f"{phase} granular payload root escaped its immutable phase"
            ) from exc
        phase_rows = phase_manifest.get("artifacts")
        if not isinstance(phase_rows, list):
            raise RuntimeError(
                f"{phase} phase has no authenticated artifact inventory"
            )
        registration_by_path = {
            Path(str(row["path"])).resolve(strict=True): row
            for row in phase_rows
            if isinstance(row, Mapping)
        }
        files = tuple(
            sorted(
                {Path(path).resolve(strict=True) for path in payload_files},
                key=lambda path: path.as_posix(),
            )
        )
        if not files:
            raise ValueError(
                f"{phase}/{node_key} granular payload inventory is empty"
            )
        if any(path not in registration_by_path for path in files):
            raise ValueError(
                f"{phase}/{node_key} granular payload was not phase-authenticated"
            )
        relative_paths: list[str] = []
        expected: dict[str, tuple[str, int]] = {}
        trusted: dict[str, tuple[int, ...]] = {}
        phase_stats = self._phase_payload_stat_inventories.get(phase)
        for path in files:
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"{phase}/{node_key} granular payload escaped its root"
                ) from exc
            row = registration_by_path[path]
            phase_relative = path.relative_to(phase_root).as_posix()
            relative_paths.append(relative)
            expected[relative] = (
                str(row["sha256"]),
                int(row["size_bytes"]),
            )
            if phase_stats is not None:
                state = phase_stats.get(phase_relative)
                if state is None:
                    raise RuntimeError(
                        f"{phase}/{node_key} lost its authenticated stat"
                    )
                trusted[relative] = tuple(int(value) for value in state)
        metadata = {
            "schema_version": WORKFLOW_GRANULAR_CHECKPOINT_NODE_SCHEMA,
            "producer_phase": phase,
            "node_ordinal": int(node_ordinal),
            "node_key": str(node_key),
            **copy.deepcopy(dict(artifact_metadata)),
        }
        granular_root, _index_path, _locator_path = (
            _granular_checkpoint_index_paths(
                work_root=self.options.work_root,
                phase=phase,
            )
        )
        controls_root = granular_root / "nodes"
        controls_root.mkdir(parents=True, exist_ok=True)
        control_root = controls_root / (
            f"{int(node_ordinal):05d}-"
            f"{_sha({'node_key': str(node_key)})[:16]}"
        )
        if control_root.exists() or control_root.is_symlink():
            artifact = validate_portable_artifact(
                control_root,
                expected_kind=artifact_kind,
                expected_compatibility_key=self._checkpoint_compatibility(
                    phase
                ).key,
                expected_upstream_artifact_ids=tuple(
                    str(value) for value in upstream_artifact_ids
                ),
            )
            if (
                artifact.manifest.get("artifact_schema")
                != GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS[artifact_kind]
                or dict(artifact.artifact_metadata) != metadata
            ):
                raise RuntimeError(
                    f"{phase}/{node_key} granular checkpoint conflicts"
                )
        else:
            artifact = publish_portable_reference_artifact(
                control_root=control_root,
                payload_root=root,
                artifact_kind=artifact_kind,
                artifact_schema=(
                    GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS[artifact_kind]
                ),
                compatibility=self._checkpoint_compatibility(phase),
                upstream_artifact_ids=tuple(
                    str(value) for value in upstream_artifact_ids
                ),
                payload_paths=tuple(relative_paths),
                expected_payload_identities=expected,
                process_authenticated_stat_inventory=(
                    trusted if phase_stats is not None else None
                ),
                artifact_metadata=metadata,
                payload_inventory_policy=payload_inventory_policy,
            )
        self._published_granular_checkpoint_handles[
            artifact.artifact_id
        ] = artifact
        return artifact

    def _seal_granular_checkpoint_index(
        self,
        *,
        phase: str,
        phase_manifest: Mapping[str, Any],
        nodes: Sequence[ValidatedPortableArtifact],
        expected_external_upstream_artifact_ids: Sequence[str] | None = None,
    ) -> Mapping[str, Any]:
        descriptors: list[dict[str, Any]] = []
        for ordinal, artifact in enumerate(nodes):
            metadata = dict(artifact.artifact_metadata)
            if (
                metadata.get("producer_phase") != phase
                or metadata.get("node_ordinal") != ordinal
            ):
                raise RuntimeError(
                    f"{phase} granular checkpoint node order changed"
                )
            descriptors.append(
                {
                    "node_ordinal": ordinal,
                    "node_key": metadata["node_key"],
                    "artifact_id": artifact.artifact_id,
                    "artifact_kind": artifact.manifest["artifact_kind"],
                    "artifact_schema": artifact.manifest[
                        "artifact_schema"
                    ],
                    "upstream_artifact_ids": list(
                        artifact.manifest["upstream_artifact_ids"]
                    ),
                    "artifact_metadata": metadata,
                }
            )
        coverage = _granular_checkpoint_coverage(descriptors)
        body = {
            "schema_version": WORKFLOW_GRANULAR_CHECKPOINT_INDEX_SCHEMA,
            "phase": phase,
            "node_count": len(descriptors),
            "nodes": descriptors,
            "coverage": coverage,
            "relative_filesystem_layout_included": False,
        }
        index = {**body, "content_sha256": _sha(body)}
        root, index_path, locator_path = _granular_checkpoint_index_paths(
            work_root=self.options.work_root,
            phase=phase,
        )
        phase_manifest_path = self._phase_manifest(phase).resolve(
            strict=True
        )
        phase_manifest_sha, phase_manifest_size = stable_file_sha256(
            phase_manifest_path
        )
        locator_body = {
            "schema_version": (
                WORKFLOW_GRANULAR_CHECKPOINT_LOCATOR_SCHEMA
            ),
            "phase": phase,
            "index_content_sha256": index["content_sha256"],
            "index_path": str(index_path.resolve()),
            "phase_manifest_path": str(phase_manifest_path),
            "phase_manifest_sha256": phase_manifest_sha,
            "phase_manifest_size_bytes": phase_manifest_size,
            "node_controls": [
                {
                    "node_ordinal": ordinal,
                    "artifact_id": artifact.artifact_id,
                    "control_root": str(artifact.root),
                }
                for ordinal, artifact in enumerate(nodes)
            ],
        }
        locator = {
            **locator_body,
            "content_sha256": _sha(locator_body),
        }
        root.mkdir(parents=True, exist_ok=True)
        _write_immutable_json(index_path, index)
        _write_immutable_json(locator_path, locator)
        validated_index, validated_nodes = (
            _validate_granular_checkpoint_index_from_paths(
                work_root=self.options.work_root.resolve(strict=True),
                phase=phase,
                compatibility=self._checkpoint_compatibility(phase),
                payload_authentication_cache=(
                    self._phase_payload_authentication_cache(
                        phase=phase,
                        phase_manifest=phase_manifest,
                    )
                ),
                expected_granular_checkpoint_plan=(
                    self._expected_granular_checkpoint_plan()
                ),
                expected_stage1_scope_plan=(
                    self._authenticated_current_stage1_scope_plan()
                    if phase == "stage1_modeling"
                    else None
                ),
                expected_external_upstream_artifact_ids=(
                    expected_external_upstream_artifact_ids
                ),
            )
        )
        if (
            validated_index != index
            or tuple(
                artifact.artifact_id for artifact in validated_nodes
            )
            != tuple(artifact.artifact_id for artifact in nodes)
        ):
            raise RuntimeError(
                f"{phase} granular checkpoint index changed after sealing"
            )
        self._published_granular_checkpoint_indexes[phase] = (
            validated_index
        )
        for artifact in validated_nodes:
            self._published_granular_checkpoint_handles[
                artifact.artifact_id
            ] = artifact
        return validated_index

    def _granular_checkpoint_index(
        self,
        phase: str,
        *,
        required: bool,
    ) -> Mapping[str, Any] | None:
        cached = self._published_granular_checkpoint_indexes.get(phase)
        if cached is not None:
            return cached
        root, _index_path, _locator_path = (
            _granular_checkpoint_index_paths(
                work_root=self.options.work_root,
                phase=phase,
            )
        )
        if not root.exists() and not root.is_symlink():
            if required:
                raise RuntimeError(
                    f"required granular checkpoint index is absent: {phase}"
                )
            return None
        expected_external_upstream_artifact_ids: (
            tuple[str, ...] | None
        ) = None
        if phase == "stage1_modeling":
            prepared_context = self._granular_artifact_for_kind(
                phase="stage1_preflight",
                artifact_kind="prepared_stage1_context",
                required=True,
            )
            assert prepared_context is not None
            expected_external_upstream_artifact_ids = (
                prepared_context.artifact_id,
            )
        elif phase == "stage2_inference":
            stage1 = self._checkpoint_artifact_for_phase(
                "stage1_modeling",
                required=True,
            )
            canary = self._checkpoint_artifact_for_phase(
                "stage2_canary",
                required=True,
            )
            assert stage1 is not None and canary is not None
            expected_external_upstream_artifact_ids = (
                stage1.artifact_id,
                canary.artifact_id,
            )
        index, handles = _validate_granular_checkpoint_index_from_paths(
            work_root=self.options.work_root.resolve(strict=True),
            phase=phase,
            compatibility=self._checkpoint_compatibility(phase),
            expected_granular_checkpoint_plan=(
                self._expected_granular_checkpoint_plan()
            ),
            expected_stage1_scope_plan=(
                self._authenticated_current_stage1_scope_plan()
                if phase == "stage1_modeling"
                else None
            ),
            expected_external_upstream_artifact_ids=(
                expected_external_upstream_artifact_ids
            ),
        )
        self._published_granular_checkpoint_indexes[phase] = index
        for artifact in handles:
            self._published_granular_checkpoint_handles[
                artifact.artifact_id
            ] = artifact
        return index

    def _granular_artifact_for_kind(
        self,
        *,
        phase: str,
        artifact_kind: str,
        required: bool,
    ) -> ValidatedPortableArtifact | None:
        index = self._granular_checkpoint_index(
            phase,
            required=False,
        )
        candidates: list[ValidatedPortableArtifact] = []
        if index is not None:
            candidates.extend(
                self._published_granular_checkpoint_handles[
                    str(node["artifact_id"])
                ]
                for node in index["nodes"]
                if node["artifact_kind"] == artifact_kind
            )
        candidates.extend(
            artifact
            for artifact in self._adopted_artifact_handles.values()
            if artifact.manifest.get("artifact_kind") == artifact_kind
            and artifact.artifact_metadata.get("producer_phase") == phase
            and artifact not in candidates
        )
        if len(candidates) > 1:
            raise RuntimeError(
                f"{phase} has multiple granular {artifact_kind} artifacts"
            )
        if not candidates:
            if required:
                raise RuntimeError(
                    f"{phase} lacks granular {artifact_kind}"
                )
            return None
        assert_validated_artifact_unchanged(candidates[0])
        return candidates[0]

    def _authenticated_current_stage1_scope_plan(self) -> Any:
        prepared_context = self._granular_artifact_for_kind(
            phase="stage1_preflight",
            artifact_kind="prepared_stage1_context",
            required=True,
        )
        assert prepared_context is not None
        return _load_authenticated_current_stage1_scope_plan(
            prepared_context_artifact=prepared_context,
            expected_granular_checkpoint_plan=(
                self._expected_granular_checkpoint_plan()
            ),
            expected_stage1_physical_fit_identity=self.request[
                "stage1_physical_fit_identity"
            ],
            expected_global_seed=int(self.options.seed),
        )

    def _publish_prepared_stage1_context_checkpoint(
        self,
        *,
        phase_manifest: Mapping[str, Any],
        clustered_preflight: ValidatedPortableArtifact,
    ) -> Mapping[str, Any]:
        result = phase_manifest.get("result")
        if not isinstance(result, Mapping):
            raise RuntimeError("Stage 1 preflight result is invalid")
        raw_manifest = result.get(
            "prepared_stage1_context_manifest_path"
        )
        if not isinstance(raw_manifest, str) or not raw_manifest:
            raise RuntimeError(
                "typed Stage 1 preflight omitted its prepared context"
            )
        context_manifest = Path(raw_manifest).resolve(strict=True)
        context_root = context_manifest.parent
        files = tuple(
            path.resolve(strict=True)
            for path in sorted(context_root.rglob("*"))
            if path.is_file()
        )
        artifact = self._publish_granular_checkpoint_node(
            phase="stage1_preflight",
            phase_manifest=phase_manifest,
            node_ordinal=0,
            node_key="prepared_stage1_context",
            artifact_kind="prepared_stage1_context",
            payload_root=context_root,
            payload_files=files,
            upstream_artifact_ids=(clustered_preflight.artifact_id,),
            artifact_metadata={
                "coverage_role": "prepared_stage1_context",
                "scientific_content_root_sha256": result.get(
                    "prepared_stage1_context_scientific_content_root_sha256"
                ),
            },
            payload_inventory_policy=COMPLETE_PAYLOAD_TREE,
        )
        return self._seal_granular_checkpoint_index(
            phase="stage1_preflight",
            phase_manifest=phase_manifest,
            nodes=(artifact,),
        )

    def _publish_stage1_modeling_granular_checkpoints(
        self,
        *,
        phase_manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = phase_manifest.get("result")
        if (
            not isinstance(result, Mapping)
            or result.get("schema_version")
            != PORTABLE_ROLE_NEUTRAL_STAGE1_PHASE_SCHEMA
        ):
            raise RuntimeError(
                "typed Stage 1 modeling lacks its role-neutral result"
            )
        prepared_context = self._granular_artifact_for_kind(
            phase="stage1_preflight",
            artifact_kind="prepared_stage1_context",
            required=True,
        )
        assert prepared_context is not None
        phase_root = Path(str(phase_manifest["attempt_dir"])).resolve(
            strict=True
        )
        execution_root = Path(
            str(result["role_neutral_execution_root"])
        ).resolve(strict=True)
        try:
            execution_root.relative_to(phase_root)
        except ValueError as exc:
            raise RuntimeError(
                "role-neutral Stage 1 execution escaped its phase"
            ) from exc
        binding_root = (
            execution_root
            / "coordination_gate"
            / "scientific_bindings"
        )
        binding = _read_private_json_object(
            binding_root / "role_neutral_binding_set.json",
            label="role-neutral Stage 1 granular binding set",
        )
        physical_rows = binding.get("physical_payloads")
        logical_rows = binding.get("logical_views")
        granular_plan = self._expected_granular_checkpoint_plan()
        expected_physical_owners = list(
            granular_plan["stage1_physical_owner_scope_ids"]
        )
        expected_logical_scopes = list(
            granular_plan["stage1_logical_scope_ids"]
        )
        if (
            not isinstance(physical_rows, list)
            or not physical_rows
            or not isinstance(logical_rows, list)
            or not logical_rows
            or len(physical_rows)
            != int(
                granular_plan["stage1_physical_fit_count"]
            )
            or len(logical_rows)
            != int(
                granular_plan["stage1_logical_scope_count"]
            )
            or result.get("physical_fit_count")
            != granular_plan["stage1_physical_fit_count"]
            or result.get("logical_scope_count")
            != granular_plan["stage1_logical_scope_count"]
            or [
                str(row.get("physical_owner_scope_id"))
                for row in physical_rows
                if isinstance(row, Mapping)
            ]
            != expected_physical_owners
            or [
                str(row.get("logical_scope_id"))
                for row in logical_rows
                if isinstance(row, Mapping)
            ]
            != expected_logical_scopes
        ):
            raise RuntimeError(
                "role-neutral Stage 1 granular binding coverage changed"
            )
        nodes: list[ValidatedPortableArtifact] = []
        component_by_owner: dict[
            str, dict[str, ValidatedPortableArtifact]
        ] = {}
        physical_by_owner: dict[str, ValidatedPortableArtifact] = {}
        from .physical_fit_deduplication import (
            PhysicalFitKey,
            ordered_row_identity,
        )
        from .production_stage1_scope_scheduler import (
            Stage1PhysicalFitIdentity,
        )

        physical_identity = Stage1PhysicalFitIdentity.from_mapping(
            self.request["stage1_physical_fit_identity"]
        )
        authenticated_scope_plan = (
            self._authenticated_current_stage1_scope_plan()
        )
        exact_scope_projection = (
            _stage1_scope_plan_granular_expectations(
                scope_plan=authenticated_scope_plan,
                expected_granular_checkpoint_plan=granular_plan,
            )
        )
        expected_key_records = exact_scope_projection[
            "physical_fit_key_records_by_owner"
        ]
        physical_key_by_owner: dict[str, PhysicalFitKey] = {}
        for physical_row in physical_rows:
            if not isinstance(physical_row, Mapping):
                raise RuntimeError(
                    "role-neutral physical binding is invalid"
                )
            owner = str(physical_row["physical_owner_scope_id"])
            physical_descriptor = (
                binding_root / str(physical_row["relative_path"])
            ).resolve(strict=True)
            physical_payload = _read_private_json_object(
                physical_descriptor,
                label=f"{owner} physical fit binding",
            )
            fit_row_ids = tuple(
                physical_payload.get("fit_row_ids") or ()
            )
            physical_key = PhysicalFitKey(
                architecture_identity=(
                    physical_identity.architecture_identity
                ),
                target=physical_identity.target,
                fit_row_order_identity=ordered_row_identity(
                    fit_row_ids
                ),
                scientific_configuration_identity=(
                    physical_identity.scientific_configuration_identity
                ),
                canonical_group_seed=physical_payload[
                    "canonical_group_seed"
                ],
                producer_identity=(
                    physical_identity.producer_identity
                ),
                runtime_compatibility_class=(
                    physical_identity.runtime_compatibility_class
                ),
            )
            if (
                physical_payload.get("physical_owner_scope_id")
                != owner
                or physical_payload.get(
                    "fit_row_order_fingerprint"
                )
                != physical_key.fit_row_order_identity
                or (
                    physical_payload.get(
                        "physical_fit_key_record"
                    )
                    is not None
                    and physical_payload.get(
                        "physical_fit_key_record"
                    )
                    != physical_key.as_dict()
                )
                or physical_key.as_dict()
                != expected_key_records.get(owner)
            ):
                raise RuntimeError(
                    f"{owner} physical fit key differs from its binding"
                )
            physical_key_by_owner[owner] = physical_key
            component_by_owner[owner] = {}
            for component, kind in (
                ("tfidf", "tfidf_component"),
                ("neural_query", "neural_query_component"),
            ):
                component_root = (
                    execution_root / "components" / owner / component
                ).resolve(strict=True)
                files = tuple(
                    path.resolve(strict=True)
                    for path in sorted(component_root.rglob("*"))
                    if path.is_file()
                )
                artifact = self._publish_granular_checkpoint_node(
                    phase="stage1_modeling",
                    phase_manifest=phase_manifest,
                    node_ordinal=len(nodes),
                    node_key=f"{kind}:{owner}",
                    artifact_kind=kind,
                    payload_root=component_root,
                    payload_files=files,
                    upstream_artifact_ids=(
                        prepared_context.artifact_id,
                    ),
                    artifact_metadata={
                        "coverage_role": kind,
                        "physical_owner_scope_id": owner,
                        "physical_fit_key": physical_key.key,
                        "physical_fit_key_record": (
                            physical_key.as_dict()
                        ),
                    },
                    payload_inventory_policy=COMPLETE_PAYLOAD_TREE,
                )
                nodes.append(artifact)
                component_by_owner[owner][component] = artifact
            owner_root = (
                execution_root / "components" / owner
            ).resolve(strict=True)
            owner_files = [
                path.resolve(strict=True)
                for path in sorted(owner_root.rglob("*"))
                if path.is_file()
            ]
            artifact = self._publish_granular_checkpoint_node(
                phase="stage1_modeling",
                phase_manifest=phase_manifest,
                node_ordinal=len(nodes),
                node_key=f"physical_scope_fit:{owner}",
                artifact_kind="physical_scope_fit",
                payload_root=phase_root,
                payload_files=(
                    *owner_files,
                    physical_descriptor,
                ),
                upstream_artifact_ids=(
                    prepared_context.artifact_id,
                    component_by_owner[owner]["tfidf"].artifact_id,
                    component_by_owner[owner][
                        "neural_query"
                    ].artifact_id,
                ),
                artifact_metadata={
                    "coverage_role": "physical_scope_fit",
                    "physical_owner_scope_id": owner,
                    "physical_fit_key": physical_key.key,
                    "physical_fit_key_record": (
                        physical_key.as_dict()
                    ),
                },
                payload_inventory_policy=(
                    REGISTERED_PAYLOAD_PATHS_ONLY
                ),
            )
            nodes.append(artifact)
            physical_by_owner[owner] = artifact

        logical_nodes: list[ValidatedPortableArtifact] = []
        for logical_row in logical_rows:
            if not isinstance(logical_row, Mapping):
                raise RuntimeError(
                    "role-neutral logical binding is invalid"
                )
            logical_id = str(logical_row["logical_scope_id"])
            logical_path = (
                binding_root / str(logical_row["relative_path"])
            ).resolve(strict=True)
            logical = _read_private_json_object(
                logical_path,
                label=f"{logical_id} logical binding",
            )
            owner = str(logical["physical_owner_scope_id"])
            physical = physical_by_owner.get(owner)
            expected_owner = granular_plan[
                "stage1_logical_to_physical_owner"
            ][logical_id]
            if (
                physical is None
                or owner != expected_owner
                or logical.get("logical_scope_id") != logical_id
            ):
                raise RuntimeError(
                    f"{logical_id} references a changed physical owner"
                )
            artifact = self._publish_granular_checkpoint_node(
                phase="stage1_modeling",
                phase_manifest=phase_manifest,
                node_ordinal=len(nodes),
                node_key=f"logical_scope_binding:{logical_id}",
                artifact_kind="logical_scope_bindings",
                payload_root=phase_root,
                payload_files=(logical_path,),
                upstream_artifact_ids=(physical.artifact_id,),
                artifact_metadata={
                    "coverage_role": "logical_scope_binding",
                    "logical_scope_id": logical_id,
                    "physical_owner_scope_id": owner,
                    "physical_fit_key": (
                        physical_key_by_owner[owner].key
                    ),
                    "physical_fit_key_record": (
                        physical_key_by_owner[owner].as_dict()
                    ),
                    "logical_purpose": logical.get(
                        "logical_purpose"
                    ),
                },
                payload_inventory_policy=(
                    REGISTERED_PAYLOAD_PATHS_ONLY
                ),
            )
            nodes.append(artifact)
            logical_nodes.append(artifact)

        bundle_manifest = Path(
            str(result["bundle_manifest_path"])
        ).resolve(strict=True)
        row_map = (bundle_manifest.parent / "row_registry.parquet").resolve(
            strict=True
        )
        row_map_artifact = self._publish_granular_checkpoint_node(
            phase="stage1_modeling",
            phase_manifest=phase_manifest,
            node_ordinal=len(nodes),
            node_key="stage1_row_map",
            artifact_kind="row_map",
            payload_root=phase_root,
            payload_files=(row_map,),
            upstream_artifact_ids=tuple(
                artifact.artifact_id for artifact in logical_nodes
            ),
            artifact_metadata={
                "coverage_role": "row_map",
                "logical_scope_count": len(logical_nodes),
            },
            payload_inventory_policy=REGISTERED_PAYLOAD_PATHS_ONLY,
        )
        nodes.append(row_map_artifact)
        observed_counts = _granular_checkpoint_coverage(
            [
                {
                    "artifact_kind": artifact.manifest[
                        "artifact_kind"
                    ],
                    "artifact_id": artifact.artifact_id,
                }
                for artifact in nodes
            ]
        )["artifact_kind_counts"]
        if observed_counts != granular_plan[
            "stage1_artifact_kind_counts"
        ]:
            raise RuntimeError(
                "Stage 1 granular component coverage differs from the plan"
            )
        return self._seal_granular_checkpoint_index(
            phase="stage1_modeling",
            phase_manifest=phase_manifest,
            nodes=tuple(nodes),
            expected_external_upstream_artifact_ids=(
                prepared_context.artifact_id,
            ),
        )

    def _publish_stage2_inference_granular_checkpoints(
        self,
        *,
        phase_manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = phase_manifest.get("result")
        if (
            not isinstance(result, Mapping)
            or result.get("mode")
            != "reference_only_role_neutral_stage2"
        ):
            raise RuntimeError(
                "typed Stage 2 inference lacks its reference-only result"
            )
        phase_root = Path(str(phase_manifest["attempt_dir"])).resolve(
            strict=True
        )
        granular_plan = self._expected_granular_checkpoint_plan()
        stage1 = self._checkpoint_artifact_for_phase(
            "stage1_modeling",
            required=True,
        )
        canary = self._checkpoint_artifact_for_phase(
            "stage2_canary",
            required=True,
        )
        assert stage1 is not None and canary is not None
        base_upstream = (stage1.artifact_id, canary.artifact_id)
        nodes: list[ValidatedPortableArtifact] = []

        batch_path = Path(
            str(result["hierarchical_batch_result_path"])
        ).resolve(strict=True)
        response_root = batch_path.parent
        response_files = tuple(
            path.resolve(strict=True)
            for path in sorted(response_root.rglob("*"))
            if path.is_file()
        )
        response = self._publish_granular_checkpoint_node(
            phase="stage2_inference",
            phase_manifest=phase_manifest,
            node_ordinal=len(nodes),
            node_key="stage2_response_component",
            artifact_kind="stage2_response_component",
            payload_root=response_root,
            payload_files=response_files,
            upstream_artifact_ids=base_upstream,
            artifact_metadata={
                "coverage_role": "stage2_response_component",
            },
            payload_inventory_policy=COMPLETE_PAYLOAD_TREE,
        )
        nodes.append(response)

        ledger_paths = result.get(
            "complete_paged_ledger_artifact_paths"
        )
        if (
            not isinstance(ledger_paths, list)
            or not ledger_paths
            or any(not isinstance(path, str) for path in ledger_paths)
        ):
            raise RuntimeError(
                "Stage 2 inference omitted complete extraction ledgers"
            )
        extraction = self._publish_granular_checkpoint_node(
            phase="stage2_inference",
            phase_manifest=phase_manifest,
            node_ordinal=len(nodes),
            node_key="stage2_extraction_component",
            artifact_kind="stage2_extraction_component",
            payload_root=phase_root,
            payload_files=tuple(
                Path(path).resolve(strict=True) for path in ledger_paths
            ),
            upstream_artifact_ids=(response.artifact_id,),
            artifact_metadata={
                "coverage_role": "stage2_extraction_component",
                "ledger_artifact_count": len(ledger_paths),
            },
            payload_inventory_policy=REGISTERED_PAYLOAD_PATHS_ONLY,
        )
        nodes.append(extraction)

        manifest_paths = result.get("fold_manifest_paths")
        prediction_paths = result.get("fold_prediction_paths")
        if (
            not isinstance(manifest_paths, list)
            or not isinstance(prediction_paths, list)
            or not manifest_paths
            or len(manifest_paths) != len(prediction_paths)
        ):
            raise RuntimeError(
                "Stage 2 inference fold registrations are incomplete"
            )
        fold_records: list[tuple[int, Path, Path]] = []
        for raw_manifest, raw_prediction in zip(
            manifest_paths,
            prediction_paths,
            strict=True,
        ):
            manifest_path = Path(str(raw_manifest)).resolve(strict=True)
            wrapper = _read_private_json_object(
                manifest_path,
                label="Stage 2 fold manifest",
            )
            fold_body = wrapper.get("body")
            if not isinstance(fold_body, Mapping):
                raise RuntimeError(
                    "Stage 2 fold manifest wrapper is invalid"
                )
            outer_fold = int(fold_body["outer_fold"])
            fold_records.append(
                (
                    outer_fold,
                    manifest_path,
                    Path(str(raw_prediction)).resolve(strict=True),
                )
            )
        fold_records.sort(key=lambda row: row[0])
        if [row[0] for row in fold_records] != list(
            granular_plan["stage2_fold_ids"]
        ):
            raise RuntimeError(
                "Stage 2 fold registrations are incomplete, duplicated, "
                "or differ from the request-derived plan"
            )

        reviews: dict[int, ValidatedPortableArtifact] = {}
        for outer_fold, manifest_path, _prediction_path in fold_records:
            review_root = manifest_path.parent / "post_extraction_review"
            if not review_root.exists():
                raise RuntimeError(
                    f"outer fold {outer_fold} review tree is absent"
                )
            review_root = review_root.resolve(strict=True)
            review_files = tuple(
                path.resolve(strict=True)
                for path in sorted(review_root.rglob("*"))
                if path.is_file()
            )
            if not review_files:
                raise RuntimeError(
                    f"outer fold {outer_fold} review tree is empty"
                )
            review = self._publish_granular_checkpoint_node(
                phase="stage2_inference",
                phase_manifest=phase_manifest,
                node_ordinal=len(nodes),
                node_key=(
                    f"stage2_review_component:outer_{outer_fold:03d}"
                ),
                artifact_kind="stage2_review_component",
                payload_root=review_root,
                payload_files=review_files,
                upstream_artifact_ids=(extraction.artifact_id,),
                artifact_metadata={
                    "coverage_role": "stage2_review_component",
                    "outer_fold": outer_fold,
                },
                payload_inventory_policy=COMPLETE_PAYLOAD_TREE,
            )
            nodes.append(review)
            reviews[outer_fold] = review
        if list(reviews) != list(
            granular_plan["stage2_review_fold_ids"]
        ):
            raise RuntimeError(
                "Stage 2 review coverage differs from the request-derived plan"
            )

        for outer_fold, manifest_path, prediction_path in fold_records:
            upstream = [
                response.artifact_id,
                extraction.artifact_id,
            ]
            upstream.append(reviews[outer_fold].artifact_id)
            fold = self._publish_granular_checkpoint_node(
                phase="stage2_inference",
                phase_manifest=phase_manifest,
                node_ordinal=len(nodes),
                node_key=f"stage2_fold:outer_{outer_fold:03d}",
                artifact_kind="stage2_fold",
                payload_root=phase_root,
                payload_files=(manifest_path, prediction_path),
                upstream_artifact_ids=tuple(upstream),
                artifact_metadata={
                    "coverage_role": "stage2_fold",
                    "outer_fold": outer_fold,
                },
                payload_inventory_policy=(
                    REGISTERED_PAYLOAD_PATHS_ONLY
                ),
            )
            nodes.append(fold)
        observed_counts = _granular_checkpoint_coverage(
            [
                {
                    "artifact_kind": artifact.manifest[
                        "artifact_kind"
                    ],
                    "artifact_id": artifact.artifact_id,
                }
                for artifact in nodes
            ]
        )["artifact_kind_counts"]
        if observed_counts != granular_plan[
            "stage2_artifact_kind_counts"
        ]:
            raise RuntimeError(
                "Stage 2 granular component coverage differs from the plan"
            )
        return self._seal_granular_checkpoint_index(
            phase="stage2_inference",
            phase_manifest=phase_manifest,
            nodes=tuple(nodes),
            expected_external_upstream_artifact_ids=base_upstream,
        )

    def _checkpoint_publication_attestation_path(self, phase: str) -> Path:
        return (
            self.options.work_root
            / "execution_attestations"
            / "portable_checkpoint_publications"
            / f"{phase}.json"
        )

    def _require_adopted_preflight_storage_compatibility(
        self,
        artifact: ValidatedPortableArtifact,
    ) -> None:
        _require_adopted_compact_preflight_parquet_compression(
            artifact,
            expected=self.options.cluster_preflight_parquet_compression,
        )

    def _adopted_checkpoint_handle_for_phase(
        self,
        phase: str,
    ) -> ValidatedPortableArtifact | None:
        adopted = self._adopted_record_for_phase(phase)
        if adopted is None:
            return None
        record, _locator = adopted
        handle = self._adopted_artifact_handles.get(str(record["artifact_id"]))
        if handle is None:
            raise RuntimeError("adopted checkpoint lost its authenticated handle")
        assert_validated_artifact_unchanged(handle)
        if phase == "stage1_preflight":
            self._require_adopted_preflight_storage_compatibility(handle)
        return handle

    def _expected_checkpoint_upstream_ids(self, phase: str) -> tuple[str, ...]:
        spec = PORTABLE_CHECKPOINT_PHASE_SPECS.get(phase)
        if not isinstance(spec, Mapping):
            raise ValueError(f"workflow phase has no portable checkpoint: {phase}")
        upstream_phases = tuple(str(value) for value in spec["upstream_phases"])
        base_upstream = tuple(
            self._checkpoint_artifact_for_phase(
                upstream_phase,
                required=True,
            ).artifact_id
            for upstream_phase in upstream_phases
        )
        if (
            self.request.get("portable_typed_workflow") is not True
            or phase not in {"stage1_modeling", "stage2_inference"}
        ):
            return base_upstream
        index = self._granular_checkpoint_index(
            phase,
            required=False,
        )
        if index is not None:
            terminal_kinds = (
                {"logical_scope_bindings", "row_map"}
                if phase == "stage1_modeling"
                else {"stage2_fold"}
            )
            granular_upstream = tuple(
                str(node["artifact_id"])
                for node in index["nodes"]
                if node["artifact_kind"] in terminal_kinds
            )
        else:
            adopted = self._adopted_record_for_phase(phase)
            metadata = (
                adopted[0].get("artifact_metadata")
                if adopted is not None
                else None
            )
            raw = (
                metadata.get("granular_terminal_artifact_ids")
                if isinstance(metadata, Mapping)
                else None
            )
            if (
                not isinstance(raw, list)
                or not raw
                or any(not isinstance(value, str) for value in raw)
            ):
                raise RuntimeError(
                    f"{phase} lacks its granular terminal dependencies"
                )
            granular_upstream = tuple(raw)
        if (
            not granular_upstream
            or len(granular_upstream) != len(set(granular_upstream))
        ):
            raise RuntimeError(
                f"{phase} granular terminal dependencies are invalid"
            )
        return (*base_upstream, *granular_upstream)

    def _primary_granular_artifact_metadata(
        self,
        phase: str,
    ) -> Mapping[str, Any] | None:
        if (
            self.request.get("portable_typed_workflow") is not True
            or phase not in {"stage1_modeling", "stage2_inference"}
        ):
            return None
        index = self._granular_checkpoint_index(
            phase,
            required=True,
        )
        assert index is not None
        return _granular_primary_metadata_from_index(
            phase=phase,
            index=index,
        )

    def _expected_primary_artifact_metadata(
        self,
        phase: str,
    ) -> Mapping[str, Any]:
        if (
            self.request.get("portable_typed_workflow") is True
            and phase in {"stage1_modeling", "stage2_inference"}
        ):
            index = self._granular_checkpoint_index(
                phase,
                required=False,
            )
            if index is not None:
                return _granular_primary_metadata_from_index(
                    phase=phase,
                    index=index,
                )
        adopted = self._adopted_record_for_phase(phase)
        metadata = (
            adopted[0].get("artifact_metadata")
            if adopted is not None
            else None
        )
        return (
            {}
            if not isinstance(metadata, Mapping)
            else copy.deepcopy(dict(metadata))
        )

    def _validate_checkpoint_publication_attestation(
        self,
        *,
        phase: str,
        phase_manifest: Mapping[str, Any],
        artifact: ValidatedPortableArtifact,
    ) -> Mapping[str, Any]:
        phase_manifest_path = self._phase_manifest(phase).resolve(strict=True)
        expected = _checkpoint_publication_attestation_value(
            producer_request_sha256=self.request["request_sha256"],
            phase=phase,
            phase_manifest_path=phase_manifest_path,
            phase_manifest=phase_manifest,
            artifact=artifact,
        )
        target = self._checkpoint_publication_attestation_path(phase)
        _write_immutable_json(target, expected)
        reopened = _read_json_object(
            target,
            label=f"{phase} checkpoint publication attestation",
        )
        if reopened != expected:
            raise RuntimeError(f"{phase} checkpoint publication attestation changed")
        return reopened

    def _checkpoint_artifact_for_phase(
        self,
        phase: str,
        *,
        required: bool,
    ) -> ValidatedPortableArtifact | None:
        spec = PORTABLE_CHECKPOINT_PHASE_SPECS.get(phase)
        if not isinstance(spec, Mapping):
            if required:
                raise ValueError(f"workflow phase has no portable checkpoint: {phase}")
            return None
        upstream_ids = self._expected_checkpoint_upstream_ids(phase)
        expected_metadata = self._expected_primary_artifact_metadata(
            phase
        )
        adopted_reference = self._adopted_record_for_phase(phase)
        adopted = self._adopted_checkpoint_handle_for_phase(phase)
        if adopted is not None:
            if adopted_reference is None:
                raise RuntimeError(
                    f"adopted {phase} checkpoint lost its immutable request record"
                )
            adopted_record, _adopted_locator = adopted_reference
            expected_compatibility = self._checkpoint_compatibility(phase)
            if (
                adopted.manifest.get("artifact_kind") != spec["artifact_kind"]
                or str(adopted_record.get("artifact_id"))
                != adopted.artifact_id
                or not _adopted_compatibility_matches_request(
                    artifact=adopted,
                    expected=expected_compatibility.as_dict(),
                    record=adopted_record,
                )
                or tuple(adopted.manifest.get("upstream_artifact_ids") or ()) != upstream_ids
                or dict(adopted.artifact_metadata)
                != dict(expected_metadata)
                or not isinstance(adopted.phase_binding, Mapping)
                or adopted.phase_binding.get("phase") != phase
            ):
                raise RuntimeError(f"adopted {phase} checkpoint differs from the requested DAG")
            self._published_checkpoint_handles[phase] = adopted
            return adopted
        cached = self._published_checkpoint_handles.get(phase)
        if cached is not None:
            assert_validated_artifact_unchanged(cached)
            if (
                cached.manifest.get("artifact_kind") != spec["artifact_kind"]
                or cached.manifest.get("artifact_schema") != spec["artifact_schema"]
                or cached.compatibility_key != self._checkpoint_compatibility(phase).key
                or tuple(cached.manifest.get("upstream_artifact_ids") or ()) != upstream_ids
                or dict(cached.artifact_metadata)
                != dict(expected_metadata)
            ):
                raise RuntimeError(f"cached {phase} checkpoint differs from the requested DAG")
            return cached
        control_root = self._checkpoint_control_root(phase)
        if not control_root.exists() and not control_root.is_symlink():
            if required:
                raise RuntimeError(f"required portable checkpoint is absent: {phase}")
            return None
        artifact = validate_portable_artifact(
            control_root,
            expected_kind=str(spec["artifact_kind"]),
            expected_compatibility_key=self._checkpoint_compatibility(phase).key,
            expected_upstream_artifact_ids=upstream_ids,
        )
        authenticated_payload_bytes = sum(row.size_bytes for row in artifact.payloads)
        self.telemetry.count_bytes(
            read=authenticated_payload_bytes,
            hashed=authenticated_payload_bytes,
        )
        if (
            artifact.manifest.get("artifact_schema") != spec["artifact_schema"]
            or dict(artifact.artifact_metadata)
            != dict(expected_metadata)
            or not isinstance(artifact.phase_binding, Mapping)
            or artifact.phase_binding.get("phase") != phase
        ):
            raise RuntimeError(f"published {phase} checkpoint has the wrong schema or phase")
        self._published_checkpoint_handles[phase] = artifact
        return artifact

    def _publish_completed_phase_checkpoint(
        self,
        phase: str,
        phase_manifest: Mapping[str, Any],
    ) -> ValidatedPortableArtifact | None:
        """Reference one durable phase tree as an immutable scientific DAG node."""

        spec = PORTABLE_CHECKPOINT_PHASE_SPECS.get(phase)
        if not isinstance(spec, Mapping):
            return None
        adopted = self._adopted_checkpoint_handle_for_phase(phase)
        if adopted is not None:
            return adopted
        artifacts = phase_manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise RuntimeError(f"{phase} phase manifest has no artifact inventory")
        if not artifacts:
            if phase in self.phase_overrides or (
                phase == "oracle_evaluation"
                and phase_manifest.get("result", {}).get("skipped_by_configuration") is True
            ):
                return None
            raise RuntimeError(f"{phase} completed without checkpointable payload bytes")
        typed = self.request.get("portable_typed_workflow") is True
        if typed and phase == "stage1_modeling":
            if self._granular_checkpoint_index(
                phase,
                required=False,
            ) is None:
                self._publish_stage1_modeling_granular_checkpoints(
                    phase_manifest=phase_manifest,
                )
        if typed and phase == "stage2_inference":
            if self._granular_checkpoint_index(
                phase,
                required=False,
            ) is None:
                self._publish_stage2_inference_granular_checkpoints(
                    phase_manifest=phase_manifest,
                )
        upstream_ids = self._expected_checkpoint_upstream_ids(phase)
        existing = self._checkpoint_artifact_for_phase(
            phase,
            required=False,
        )
        if existing is not None:
            self._validate_checkpoint_publication_attestation(
                phase=phase,
                phase_manifest=phase_manifest,
                artifact=existing,
            )
            if (
                typed
                and phase == "stage1_preflight"
                and self._granular_checkpoint_index(
                    phase,
                    required=False,
                )
                is None
            ):
                self._publish_prepared_stage1_context_checkpoint(
                    phase_manifest=phase_manifest,
                    clustered_preflight=existing,
                )
            return existing
        payload_root = Path(str(phase_manifest["attempt_dir"])).resolve(strict=True)
        payload_paths: list[str] = []
        expected_payload_identities: dict[str, tuple[str, int]] = {}
        for row in artifacts:
            if not isinstance(row, Mapping):
                raise RuntimeError(f"{phase} phase artifact inventory is invalid")
            relative = str(row.get("relative_path") or "")
            payload_paths.append(relative)
            expected_payload_identities[relative] = (
                str(row.get("sha256") or ""),
                int(row.get("size_bytes", -1)),
            )
        if len(payload_paths) != len(set(payload_paths)) or set(expected_payload_identities) != set(
            payload_paths
        ):
            raise RuntimeError(f"{phase} phase artifact inventory contains duplicates")
        result = phase_manifest.get("result")
        if not isinstance(result, Mapping):
            raise RuntimeError(f"{phase} phase result is invalid")
        artifact = publish_portable_reference_artifact(
            control_root=self._checkpoint_control_root(phase),
            payload_root=payload_root,
            artifact_kind=str(spec["artifact_kind"]),
            artifact_schema=str(spec["artifact_schema"]),
            compatibility=self._checkpoint_compatibility(phase),
            upstream_artifact_ids=upstream_ids,
            payload_paths=tuple(payload_paths),
            expected_payload_identities=expected_payload_identities,
            process_authenticated_stat_inventory=(self._phase_payload_stat_inventories.get(phase)),
            workflow_phase=phase,
            workflow_phase_result=result,
            artifact_metadata=(
                self._primary_granular_artifact_metadata(phase)
            ),
        )
        if phase not in self._phase_payload_stat_inventories:
            published_payload_bytes = sum(int(row["size_bytes"]) for row in artifacts)
            self.telemetry.count_bytes(
                read=published_payload_bytes,
                hashed=published_payload_bytes,
            )
        self._published_checkpoint_handles[phase] = artifact
        self._validate_checkpoint_publication_attestation(
            phase=phase,
            phase_manifest=phase_manifest,
            artifact=artifact,
        )
        if typed and phase == "stage1_preflight":
            self._publish_prepared_stage1_context_checkpoint(
                phase_manifest=phase_manifest,
                clustered_preflight=artifact,
            )
        return artifact

    def _adopted_record_for_phase(
        self,
        phase: str,
    ) -> tuple[Mapping[str, Any], str] | None:
        records = self.request.get("requested_checkpoint_adoptions") or []
        locators = self.request.get("checkpoint_adoption_locators") or []
        matching = [
            (record, str(locator))
            for record, locator in zip(records, locators)
            if isinstance(record, Mapping) and record.get("substituted_phase") == phase
        ]
        if len(matching) > 1:
            raise RuntimeError("immutable request has duplicate adopted phases")
        return None if not matching else matching[0]

    def _publish_adopted_phase_reference(
        self,
        phase: str,
    ) -> Mapping[str, Any]:
        adopted = self._adopted_record_for_phase(phase)
        if adopted is None:
            raise ValueError(f"workflow phase has no adopted checkpoint: {phase}")
        record, locator = adopted
        artifact_id = str(record["artifact_id"])
        handle = self._adopted_artifact_handles.get(artifact_id)
        if handle is None:
            raise RuntimeError("adopted phase lost its authenticated artifact handle")
        assert_validated_artifact_unchanged(handle)
        if phase == "stage1_preflight":
            self._require_adopted_preflight_storage_compatibility(handle)
        phase_root = self.options.work_root / "phases" / phase
        target = phase_root / "complete_manifest.json"
        if target.is_file() and not target.is_symlink():
            existing = self._validated_complete(phase)
            if existing is None:
                raise RuntimeError("existing adopted phase failed validation")
            return existing
        if phase_root.exists() or phase_root.is_symlink():
            raise ValueError(f"adopted phase root is partial or substituted: {phase}")
        attestation = (
            self.options.work_root / "checkpoint_adoptions" / f"{artifact_id}.adoption.json"
        ).resolve(strict=True)
        body = {
            "schema_version": WORKFLOW_ADOPTED_PHASE_MANIFEST_SCHEMA,
            "phase": phase,
            "status": "complete",
            "request_sha256": self.request["request_sha256"],
            "artifact_id": artifact_id,
            "artifact_kind": record["artifact_kind"],
            "compatibility_key": record["compatibility_key"],
            "artifact_locator": locator,
            "adoption_attestation_path": str(attestation),
            "upstream_artifact_ids": list(record["upstream_artifact_ids"]),
        }
        phase_root.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(
            target,
            {**body, "content_sha256": _sha(body)},
        )
        completed = self._validated_complete(phase)
        if completed is None:
            raise RuntimeError("adopted phase publication did not validate")
        return completed

    def _write_progress(
        self,
        *,
        status: str,
        completed: Sequence[str],
        current_phase: str | None,
        error: str | None = None,
    ) -> None:
        sequence = self._phase_sequence()
        body = {
            "schema_version": WORKFLOW_PROGRESS_SCHEMA,
            "request_sha256": self.request.get("request_sha256"),
            "status": status,
            "phase_sequence": list(sequence),
            "planned_phase_count": len(sequence),
            "completed_phases": list(completed),
            "completed_phase_count": len(completed),
            "current_phase": current_phase,
            "remaining_phase_count": len(sequence) - len(completed),
            "stage1_gpu_ids": list(self.stage1_gpu_ids),
            "stage1_execution_device_count": (
                self.options.stage1_execution_device_count
            ),
            "stage1_execution_profile": (
                None
                if self.options.stage1_execution_profile is None
                else self.options.stage1_execution_profile.as_dict()
            ),
            "stage1_scope_workers_per_gpu": self.options.stage1_scope_workers_per_gpu,
            "stage1_preflight_workers": self.options.stage1_preflight_workers,
            "tfidf_workers": self.options.tfidf_workers,
            "updated_at": _utc_now(),
            "error": error,
        }
        target = self.options.work_root / "workflow_progress.json"
        _atomic_write_json(target, body)
        achievement = self._validation_achievement_attestation
        event = {
            "schema_version": WORKFLOW_STRUCTURED_LOG_EVENT_SCHEMA,
            "event": "workflow_progress",
            "request_sha256": self.request.get("request_sha256"),
            "status": status,
            "current_phase": current_phase,
            "completed_phase_count": len(completed),
            "remaining_phase_count": len(sequence) - len(completed),
            "updated_at": body["updated_at"],
            "error": error,
            "configured_log_level": (
                self.options.run_control.log_level
            ),
            "run_control_selection_content_sha256": (
                None
                if self._run_control_selection_attestation is None
                else self._run_control_selection_attestation[
                    "content_sha256"
                ]
            ),
            "validation_requested_minimum": self._validation_policy[
                "requested_minimum"
            ],
            "validation_effective_minimum": self._validation_policy[
                "effective_minimum"
            ],
            "fresh_path_only_terminal_audit_required": True,
            "fresh_path_only_terminal_audit_achieved": (
                False
                if achievement is None
                else bool(
                    achievement[
                        "fresh_path_only_terminal_audit_achieved"
                    ]
                )
            ),
            "validation_achievement_content_sha256": (
                None
                if achievement is None
                else achievement["content_sha256"]
            ),
            "terminal_phase_override_present": (
                "terminal_validation" in self.phase_overrides
            ),
        }
        event_level = (
            logging.ERROR
            if error is not None or status == "failed"
            else logging.WARNING
            if status == "paused"
            else logging.INFO
        )
        _emit_structured_workflow_log(
            configured_threshold=self.options.run_control.log_level,
            event_level=event_level,
            payload=event,
        )

    def _phase_manifest(self, phase: str) -> Path:
        return self.options.work_root / "phases" / phase / "complete_manifest.json"

    def _validated_complete(self, phase: str) -> Mapping[str, Any] | None:
        path = self._phase_manifest(phase)
        if not path.is_file():
            return None
        process_stats = self._phase_payload_stat_inventories.get(
            phase
        )
        proof_start_probe: int | None = None
        if process_stats is None:
            candidate = _read_json_object(
                path,
                label=f"{phase} phase manifest",
            )
            candidate_body = {
                key: item
                for key, item in candidate.items()
                if key != "content_sha256"
            }
            attempt = Path(
                str(candidate.get("attempt_dir", ""))
            )
            expected_phase_root = (
                self.options.work_root / "phases" / phase
            ).resolve(strict=True)
            if (
                candidate.get("schema_version")
                == WORKFLOW_PHASE_MANIFEST_SCHEMA
                and candidate.get("phase") == phase
                and candidate.get("status") == "complete"
                and candidate.get("request_sha256")
                == self.request["request_sha256"]
                and candidate.get("content_sha256")
                == _sha(candidate_body)
                and isinstance(
                    candidate.get("artifacts"),
                    list,
                )
                and attempt.is_absolute()
                and not attempt.is_symlink()
                and attempt.is_dir()
                and attempt.resolve(strict=True).parent
                == expected_phase_root
            ):
                from .production_stage1_reusable_preflight import (
                    _authentication_probe_ctime_ns,
                    _load_fast_proof,
                )

                proof_store = _phase_payload_proof_store_root(
                    self.options.work_root
                )
                key = _phase_payload_proof_key(
                    phase=phase,
                    request_sha256=self.request[
                        "request_sha256"
                    ],
                    terminal_content_sha256=str(
                        candidate["content_sha256"]
                    ),
                )
                proof = _load_fast_proof(
                    store_root=proof_store,
                    artifact_kind=(
                        f"workflow_phase_payload_{phase}"
                    ),
                    scientific_key=key,
                    artifact_root=attempt.resolve(strict=True),
                    terminal_content_sha256=str(
                        candidate["content_sha256"]
                    ),
                    producer_identity=(
                        "production_workflow_phase_publication_v1"
                    ),
                    schema_identity=WORKFLOW_PHASE_MANIFEST_SCHEMA,
                )
                if proof is not None:
                    self._phase_payload_stat_inventories[phase] = (
                        _phase_payload_stat_inventory_from_proof(
                            proof=proof[0],
                            artifacts=candidate["artifacts"],
                        )
                    )
                    return self._validated_complete(phase)
                proof_start_probe = (
                    _authentication_probe_ctime_ns(
                        proof_store
                    )
                )
        if process_stats is not None:
            value = _read_json_object(
                path,
                label=f"{phase} phase manifest",
            )
            body = {
                key: item
                for key, item in value.items()
                if key != "content_sha256"
            }
            artifacts = value.get("artifacts")
            attempt = Path(str(value.get("attempt_dir", "")))
            expected_phase_root = (
                self.options.work_root / "phases" / phase
            ).resolve(strict=True)
            if (
                set(value)
                != {
                    "schema_version",
                    "phase",
                    "status",
                    "request_sha256",
                    "attempt_dir",
                    "result",
                    "artifacts",
                    "content_sha256",
                }
                or value.get("schema_version")
                != WORKFLOW_PHASE_MANIFEST_SCHEMA
                or value.get("phase") != phase
                or value.get("status") != "complete"
                or value.get("request_sha256")
                != self.request["request_sha256"]
                or value.get("content_sha256") != _sha(body)
                or not isinstance(value.get("result"), Mapping)
                or not isinstance(artifacts, list)
                or not attempt.is_absolute()
                or attempt.is_symlink()
                or not attempt.is_dir()
                or attempt.resolve(strict=True).parent
                != expected_phase_root
            ):
                raise ValueError(
                    f"completed phase fast-stat manifest failed: {phase}"
                )
            registered_relatives = [
                str(row.get("relative_path", ""))
                for row in artifacts
                if isinstance(row, Mapping)
            ]
            observed_relatives = [
                candidate.relative_to(attempt).as_posix()
                for candidate in sorted(attempt.rglob("*"))
                if candidate.is_file()
            ]
            if (
                len(registered_relatives) != len(artifacts)
                or len(registered_relatives)
                != len(set(registered_relatives))
                or registered_relatives != observed_relatives
                or set(process_stats)
                != set(registered_relatives)
            ):
                raise ValueError(
                    f"completed phase fast-stat inventory changed: {phase}"
                )
            registered_paths: set[str] = set()
            for row in artifacts:
                relative = str(row["relative_path"])
                payload = attempt / relative
                state = os.lstat(payload)
                observed_state = (
                    int(state.st_dev),
                    int(state.st_ino),
                    int(state.st_mode),
                    int(state.st_nlink),
                    int(state.st_size),
                    int(state.st_mtime_ns),
                    int(state.st_ctime_ns),
                )
                if (
                    observed_state
                    != tuple(process_stats[relative])
                    or stat.S_ISLNK(state.st_mode)
                    or not stat.S_ISREG(state.st_mode)
                    or int(state.st_nlink) != 1
                    or int(state.st_size)
                    != int(row.get("size_bytes", -1))
                    or str(payload.resolve(strict=True))
                    != str(row.get("path"))
                ):
                    raise ValueError(
                        f"completed phase fast-stat payload changed: "
                        f"{phase}/{relative}"
                    )
                registered_paths.add(str(payload.resolve(strict=True)))
            terminal_files = value["result"].get(
                "terminal_files",
                [],
            )
            if (
                not isinstance(terminal_files, list)
                or any(
                    not isinstance(item, str)
                    or item not in registered_paths
                    for item in terminal_files
                )
                or len(terminal_files) != len(set(terminal_files))
            ):
                raise ValueError(
                    f"completed phase fast-stat terminal changed: {phase}"
                )
            return value
        validated = _validate_phase_manifest_from_paths(
            work_root=self.options.work_root.resolve(strict=True),
            phase=phase,
            request_sha256=self.request["request_sha256"],
            authenticated_adoptions=self._adopted_artifact_handles,
        )
        # Deep validation has already hashed every ordinary phase payload.
        # Preserve that process-local authority so subsequent selector/path/
        # option lookups validate only exact inode/stat continuity instead of
        # rereading the same (potentially multi-gigabyte) tree.
        if validated.get("schema_version") == WORKFLOW_PHASE_MANIFEST_SCHEMA:
            attempt = Path(
                str(validated["attempt_dir"])
            ).resolve(strict=True)
            if proof_start_probe is not None:
                from .production_stage1_reusable_preflight import (
                    _publish_optional_full_auth_proof,
                )

                proof_store = _phase_payload_proof_store_root(
                    self.options.work_root
                )
                key = _phase_payload_proof_key(
                    phase=phase,
                    request_sha256=self.request[
                        "request_sha256"
                    ],
                    terminal_content_sha256=str(
                        validated["content_sha256"]
                    ),
                )
                authenticated = {
                    str(row["relative_path"]): (
                        str(row["sha256"]),
                        int(row["size_bytes"]),
                    )
                    for row in validated["artifacts"]
                }
                published_proof = _publish_optional_full_auth_proof(
                    store_root=proof_store,
                    artifact_kind=(
                        f"workflow_phase_payload_{phase}"
                    ),
                    scientific_key=key,
                    artifact_root=attempt,
                    terminal_content_sha256=str(
                        validated["content_sha256"]
                    ),
                    artifact_scientific_content_sha256=str(
                        validated["content_sha256"]
                    ),
                    producer_identity=(
                        "production_workflow_phase_publication_v1"
                    ),
                    schema_identity=(
                        WORKFLOW_PHASE_MANIFEST_SCHEMA
                    ),
                    full_authentication_start_probe_ctime_ns=(
                        proof_start_probe
                    ),
                    authenticated_byte_inventory=authenticated,
                )
                if published_proof is not None:
                    self._phase_payload_stat_inventories[
                        phase
                    ] = _phase_payload_stat_inventory_from_proof(
                        proof=published_proof,
                        artifacts=validated["artifacts"],
                    )
                else:
                    self._phase_payload_stat_inventories.pop(
                        phase,
                        None,
                    )
        return validated

    def _attempt_dir(self, phase: str) -> Path:
        configured = self.options.scratch_root
        if configured is None:
            configured = self.options.work_root.parent / f".{self.options.work_root.name}.scratch"
        phase_root = (
            Path(configured)
            / "production_all_evidence_workflow"
            / str(self.request["request_sha256"])
            / phase
        )
        phase_root.mkdir(parents=True, exist_ok=True)
        if (
            self.options.run_control.resume
            and phase in IN_PLACE_RESUMABLE_PHASES
        ):
            prior_attempts = sorted(
                path
                for path in phase_root.glob("attempt_*")
                if path.is_dir() and not path.is_symlink()
            )
            if prior_attempts:
                return prior_attempts[-1]
        attempt = phase_root / (
            "attempt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
        attempt.mkdir()
        return attempt

    def _complete(
        self,
        phase: str,
        result: Mapping[str, Any],
        *,
        attempt_dir: Path,
    ) -> Mapping[str, Any]:
        if not isinstance(result, Mapping):
            raise TypeError(f"phase {phase} did not return one result mapping")
        target = self._phase_manifest(phase)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"completed phase manifest already exists: {phase}")
        result_copy = dict(result)
        terminal_files = result_copy.get("terminal_files", [])
        if (
            not isinstance(terminal_files, list)
            or any(not isinstance(path, str) for path in terminal_files)
            or len(terminal_files) != len(set(terminal_files))
        ):
            raise ValueError(f"phase {phase} returned an invalid terminal_files list")
        proof_start_probe: int | None = None
        try:
            from .production_stage1_reusable_preflight import (
                _authentication_probe_ctime_ns,
            )

            proof_start_probe = _authentication_probe_ctime_ns(
                _phase_payload_proof_store_root(
                    self.options.work_root
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            LOGGER.warning(
                "could not start phase stat-continuity proof: %s",
                phase,
                exc_info=True,
            )
        artifacts = _attempt_tree_artifacts(attempt_dir)
        registered = {row["path"] for row in artifacts}
        for raw in terminal_files:
            terminal = Path(raw)
            if not terminal.is_absolute() or str(terminal.resolve(strict=True)) not in registered:
                raise ValueError(f"phase {phase} terminal file escaped its attempt: {terminal}")
        authenticated_bytes = sum(int(row["size_bytes"]) for row in artifacts)
        self.telemetry.count_bytes(
            read=authenticated_bytes,
            hashed=authenticated_bytes,
        )
        source_root = attempt_dir.resolve(strict=True)
        (
            published_root,
            publication_counters,
            process_authenticated_stats,
        ) = _publish_attempt_tree(
            attempt_dir=source_root,
            durable_phase_root=self.options.work_root / "phases" / phase,
            artifacts=artifacts,
        )
        self.telemetry.count_bytes(**dict(publication_counters))
        result_copy = dict(
            _rewrite_attempt_locators(
                result_copy,
                source_root=source_root,
                published_root=published_root,
            )
        )
        published_artifacts = [
            {
                **dict(row),
                "path": str(published_root / str(row["relative_path"])),
            }
            for row in artifacts
        ]
        body = {
            "schema_version": WORKFLOW_PHASE_MANIFEST_SCHEMA,
            "phase": phase,
            "status": "complete",
            "request_sha256": self.request["request_sha256"],
            "attempt_dir": str(published_root),
            "result": result_copy,
            "artifacts": published_artifacts,
        }
        manifest = {**body, "content_sha256": _sha(body)}
        _atomic_write_json(target, manifest)
        if proof_start_probe is not None:
            from .production_stage1_reusable_preflight import (
                _publish_optional_full_auth_proof,
            )

            key = _phase_payload_proof_key(
                phase=phase,
                request_sha256=self.request["request_sha256"],
                terminal_content_sha256=manifest[
                    "content_sha256"
                ],
            )
            published_proof = _publish_optional_full_auth_proof(
                store_root=_phase_payload_proof_store_root(
                    self.options.work_root
                ),
                artifact_kind=(
                    f"workflow_phase_payload_{phase}"
                ),
                scientific_key=key,
                artifact_root=published_root,
                terminal_content_sha256=manifest[
                    "content_sha256"
                ],
                artifact_scientific_content_sha256=manifest[
                    "content_sha256"
                ],
                producer_identity=(
                    "production_workflow_phase_publication_v1"
                ),
                schema_identity=WORKFLOW_PHASE_MANIFEST_SCHEMA,
                full_authentication_start_probe_ctime_ns=(
                    proof_start_probe
                ),
                authenticated_byte_inventory={
                    str(row["relative_path"]): (
                        str(row["sha256"]),
                        int(row["size_bytes"]),
                    )
                    for row in published_artifacts
                },
            )
            if published_proof is not None:
                self._phase_payload_stat_inventories[
                    phase
                ] = _phase_payload_stat_inventory_from_proof(
                    proof=published_proof,
                    artifacts=published_artifacts,
                )
            else:
                self._phase_payload_stat_inventories.pop(
                    phase,
                    None,
                )
        return manifest

    def _gpu_preflight(self) -> Mapping[str, Any]:
        requested = self.stage1_gpu_ids
        safety = self.options.resource_performance_safety
        if not requested:
            return {
                "status": "accepted",
                "requested_gpu_ids": [],
                "exclusive_gpu_check_required": False,
                "resource_performance_safety": safety.as_dict(),
                "resource_performance_safety_sha256": safety.content_sha256,
                "checked_at": _utc_now(),
            }
        logical_to_physical = _logical_to_physical_cuda_indices(requested)
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        physical_resources: dict[int, dict[str, Any]] = {}
        for line in gpu.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if (
                len(fields) != 5
                or not fields[0].isdigit()
                or not fields[2].isdigit()
                or not fields[3].isdigit()
                or not fields[4].isdigit()
            ):
                continue
            physical_resources[int(fields[0])] = {
                "uuid": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "utilization_percent": int(fields[4]),
            }
        resources = {
            logical_id: physical_resources[physical_id]
            for logical_id, physical_id in logical_to_physical.items()
            if physical_id in physical_resources
        }
        mapping = {gpu_id: str(resource["uuid"]) for gpu_id, resource in resources.items()}
        missing = [gpu_id for gpu_id in requested if gpu_id not in mapping]
        if missing:
            raise RuntimeError(f"requested Stage 1 GPU IDs are unavailable: {missing}")
        active: dict[str, list[dict[str, Any]]] = {}
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 3:
                continue
            uuid, pid_text, used_memory = fields[:3]
            if not pid_text.isdigit() or int(pid_text) == os.getpid():
                continue
            active.setdefault(uuid, []).append(
                {"pid": int(pid_text), "used_memory_mib": used_memory}
            )
        observed_compute_processes = {
            gpu_id: active[mapping[gpu_id]] for gpu_id in requested if active.get(mapping[gpu_id])
        }
        unsafe_allocation_state: dict[int, Mapping[str, Any]] = {}
        for gpu_id in requested:
            resource = resources[gpu_id]
            total = int(resource["memory_total_mib"])
            used = int(resource["memory_used_mib"])
            free_bytes = max(0, total - used) * 1024**2
            allocation_fraction = used / total if total > 0 else math.inf
            reasons: list[str] = []
            if allocation_fraction >= safety.gpu_max_allocation_fraction:
                reasons.append("existing_allocation_exceeds_configured_fraction")
            if free_bytes < safety.gpu_minimum_headroom_bytes:
                reasons.append("less_than_configured_headroom")
            if reasons:
                unsafe_allocation_state[gpu_id] = {
                    **resource,
                    "allocation_fraction": allocation_fraction,
                    "free_memory_bytes": free_bytes,
                    "reasons": reasons,
                }
        rejected_compute_processes = (
            observed_compute_processes if safety.fail_on_external_gpu_occupants else {}
        )
        if rejected_compute_processes or unsafe_allocation_state:
            raise RuntimeError(
                "Stage 1 GPUs do not satisfy the configured resource safety "
                "policy; no external process was killed: "
                + _canonical(
                    {
                        "compute_processes": observed_compute_processes,
                        "compute_processes_rejected": bool(rejected_compute_processes),
                        "unsafe_allocation_state": unsafe_allocation_state,
                        "resource_performance_safety": safety.as_dict(),
                    }
                )
            )
        return {
            "status": "accepted",
            "requested_gpu_ids": list(requested),
            "gpu_uuids": {str(gpu_id): mapping[gpu_id] for gpu_id in requested},
            "gpu_resources": {str(gpu_id): resources[gpu_id] for gpu_id in requested},
            "observed_compute_processes": observed_compute_processes,
            "exclusive_gpu_check_required": (safety.fail_on_external_gpu_occupants),
            "resource_performance_safety": safety.as_dict(),
            "resource_performance_safety_sha256": safety.content_sha256,
            "checked_at": _utc_now(),
        }

    def _effective_stage1_profile(
        self,
        attempt: Path,
        *,
        dataset_path: Path,
        embedding_cache_dir: Path,
    ) -> Path:
        raw = json.loads(self.options.stage1_profile_path.read_text(encoding="utf-8"))
        config = raw.get("config", raw)
        architecture = config.get("architecture")
        if not isinstance(architecture, dict):
            raise ValueError("Stage 1 profile lacks architecture configuration")
        required_htr_window_fields = {
            "htr_chunk_size_words",
            "htr_chunk_overlap_words",
            "htr_max_chunks",
            "htr_max_chunk_length",
        }
        missing_htr_window_fields = sorted(required_htr_window_fields - set(architecture))
        if missing_htr_window_fields:
            raise ValueError(
                "Stage 1 profile must explicitly configure every HTR text-window "
                "hyperparameter; missing: " + ", ".join(missing_htr_window_fields)
            )
        config["dataset_path"] = str(dataset_path.resolve(strict=True))
        config["text_column"] = self.options.text_column
        config["treatment_column"] = self.options.treatment_column
        config["outcome_column"] = self.options.outcome_column
        config["outcome_type"] = self.options.outcome_type
        config["clinical_question"] = self.options.clinical_question
        config["cv_folds"] = self.options.outer_folds
        architecture["htr_sentence_model"] = str(self.options.htr_local_model_path.resolve())
        inner_partition_count = (
            int(self.options.initial_training_partitions) + self.options.review_rounds
        )
        for section_name in (
            "multi_model_forest",
            "multi_model_agentic_forest",
        ):
            section = architecture.get(section_name)
            if not isinstance(section, dict):
                raise ValueError(f"Stage 1 profile lacks architecture.{section_name}")
            section["candidate_consistency_inner_folds"] = inner_partition_count
            section["tfidf_nested_calibration_folds"] = self.options.tfidf_nested_calibration_folds
        explicit_forest = architecture.get("explicit_feature_forest")
        if not isinstance(explicit_forest, dict):
            raise ValueError("Stage 1 profile lacks architecture.explicit_feature_forest")
        explicit_forest["interaction_inner_folds"] = self.options.interaction_inner_folds

        cluster_local_scientific: dict[str, Any] | None = None
        if self.options.portable_scientific_spec is not None:
            architecture_profiles = self.options.portable_scientific_spec.get(
                "architecture_profiles"
            )
            cluster_profile = (
                architecture_profiles.get("cluster_local_embeddings")
                if isinstance(architecture_profiles, Mapping)
                else None
            )
            cluster_configuration = (
                cluster_profile.get("producer_configuration")
                if isinstance(cluster_profile, Mapping)
                else None
            )
            cluster_local_scientific = (
                ClusterLocalEmbeddingScientificConfig.from_mapping(
                    cluster_configuration
                ).as_dict()
            )

        def bind_embedding_sections(value: Any) -> None:
            if not isinstance(value, dict):
                return
            embedding = value.get("embedding_contrast")
            if isinstance(embedding, dict):
                cache_configuration = self._embedding_chunk_configuration()
                embedding.update(
                    {
                        key: cache_configuration[key]
                        for key in (
                            "chunk_size_words",
                            "chunk_overlap_words",
                            "max_chunks",
                            "chunk_selection",
                            "normalize_embeddings",
                            "max_seq_length",
                        )
                    }
                )
                embedding.update(
                    {
                        "model_name": self.options.embedding_model_name,
                        "cache_dir": str(embedding_cache_dir.resolve(strict=True)),
                        "device": self.options.stage1_device,
                    }
                )
                if cluster_local_scientific is not None:
                    embedding["cluster_local_scientific"] = copy.deepcopy(
                        cluster_local_scientific
                    )
            for child in value.values():
                bind_embedding_sections(child)

        bind_embedding_sections(architecture)
        forest = architecture["causal_forest"]
        if self.options.forest_runtime_config is not None:
            final_forest = self.options.forest_runtime_config.causal_forest
            final_n_estimators = final_forest.n_estimators
            final_min_samples_leaf = final_forest.min_samples_leaf
            final_max_features = final_forest.max_features
            final_honest = final_forest.honest
            final_inference = final_forest.inference
        else:
            final_n_estimators = self.options.forest_n_estimators
            final_min_samples_leaf = self.options.forest_min_samples_leaf
            final_max_features = self.options.forest_max_features
            final_honest = self.options.forest_honest
            final_inference = self.options.forest_inference
        forest.update(
            {
                "n_estimators": final_n_estimators,
                "min_samples_leaf": final_min_samples_leaf,
                "max_features": final_max_features,
                "honest": final_honest,
                "inference": final_inference,
            }
        )
        path = attempt / "effective_stage1_profile.json"
        path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _embedding_chunk_configuration(self) -> Mapping[str, Any]:
        return {
            "chunk_size_words": int(self.options.embedding_chunk_size_words),
            "chunk_overlap_words": int(self.options.embedding_chunk_overlap_words),
            "max_chunks": int(self.options.embedding_max_chunks),
            "chunk_selection": str(self.options.embedding_chunk_selection),
            "normalize_embeddings": bool(self.options.embedding_normalize),
            "max_seq_length": self.options.embedding_max_seq_length,
            **self.options.embedding_encoder.as_configuration(
                normalize_embeddings=bool(self.options.embedding_normalize)
            ),
        }

    def _input_preparation_paths(self) -> tuple[Path, Path]:
        preparation = self._validated_complete("input_preparation")
        if preparation is None:
            raise RuntimeError("input preparation is not complete")
        output = Path(preparation["result"]["output"]["path"]).resolve(strict=True)
        manifest = next(
            Path(row["path"]).resolve(strict=True)
            for row in preparation["artifacts"]
            if Path(row["path"]).name == "preparation_manifest.json"
        )
        return output, manifest

    def _input_preparation_validation_paths(
        self,
    ) -> tuple[Path, Path]:
        """Restore the path-bound scratch cohort for strict relocation checks."""

        prepared, manifest_path = self._input_preparation_paths()
        manifest = _read_json_object(
            manifest_path,
            label="published input-preparation manifest",
        )
        output = manifest.get("output")
        raw_historical = (
            output.get("path")
            if isinstance(output, Mapping)
            else None
        )
        if not isinstance(raw_historical, str) or not raw_historical.strip():
            raise RuntimeError(
                "published input preparation lacks its historical output path"
            )
        historical = Path(raw_historical)
        if historical == prepared:
            return prepared, manifest_path
        phase = self._validated_complete("input_preparation")
        if (
            phase is None
            or phase.get("schema_version") != WORKFLOW_PHASE_MANIFEST_SCHEMA
        ):
            raise RuntimeError(
                "an adopted path-relocated input preparation requires a "
                "fresh local preparation before cache import"
            )
        published_root = Path(str(phase["attempt_dir"])).resolve(strict=True)
        try:
            relative = prepared.relative_to(published_root)
        except ValueError as exc:
            raise RuntimeError(
                "published prepared cohort escaped its phase attempt"
            ) from exc
        configured_scratch = self.options.scratch_root
        if configured_scratch is None:
            configured_scratch = (
                self.options.work_root.parent
                / f".{self.options.work_root.name}.scratch"
            )
        configured_scratch.mkdir(parents=True, exist_ok=True)
        scratch_root = configured_scratch.resolve(strict=True)
        if configured_scratch.is_symlink() or scratch_root != configured_scratch:
            raise RuntimeError(
                "input-preparation provenance recovery requires one canonical "
                "non-symlink scratch root"
            )
        expected = (
            scratch_root
            / "production_all_evidence_workflow"
            / str(self.request["request_sha256"])
            / "input_preparation"
            / published_root.name
            / relative
        )
        if (
            not historical.is_absolute()
            or Path(os.path.normpath(str(historical))) != historical
            or historical != expected
        ):
            raise RuntimeError(
                "input-preparation historical path is not owned by this "
                "workflow request"
            )
        current = historical.parent
        while current != scratch_root:
            if current.exists() or current.is_symlink():
                state = os.lstat(current)
                if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(
                    state.st_mode
                ):
                    raise RuntimeError(
                        "input-preparation provenance recovery encountered a "
                        "non-directory scratch ancestor"
                    )
            current = current.parent
        historical.parent.mkdir(parents=True, exist_ok=True)
        if historical.parent.resolve(strict=True) != historical.parent:
            raise RuntimeError(
                "input-preparation provenance recovery parent is not canonical"
            )
        prepared_sha256, prepared_size = stable_file_sha256(prepared)
        output_body = {
            "path": str(historical),
            "sha256": prepared_sha256,
            "size_bytes": prepared_size,
        }
        manifest_body = {
            key: copy.deepcopy(value)
            for key, value in manifest.items()
            if key != "content_sha256"
        }
        manifest_digest = hashlib.sha256(
            json.dumps(
                manifest_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
                default=(
                    lambda item: (
                        item.item()
                        if hasattr(item, "item")
                        else str(item)
                    )
                ),
            ).encode("utf-8")
        ).hexdigest()
        if (
            manifest.get("content_sha256") != manifest_digest
            or output != output_body
        ):
            raise RuntimeError(
                "published input-preparation manifest is invalid"
            )
        if historical.exists() or historical.is_symlink():
            if historical.is_symlink() or not historical.is_file():
                raise RuntimeError(
                    "input-preparation recovery target is not one regular file"
                )
        else:
            temporary = historical.parent / (
                f".{historical.name}.rehydrating_{os.getpid()}_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            )
            try:
                with prepared.open("rb") as source, temporary.open(
                    "xb"
                ) as destination:
                    shutil.copyfileobj(
                        source,
                        destination,
                        length=8 * 1024 * 1024,
                    )
                    destination.flush()
                    os.fsync(destination.fileno())
                try:
                    os.link(temporary, historical)
                except FileExistsError:
                    pass
                _fsync_directory(historical.parent)
            finally:
                temporary.unlink(missing_ok=True)
        historical_sha256, historical_size = stable_file_sha256(historical)
        prepared_final = stable_file_sha256(prepared)
        if (
            (historical_sha256, historical_size)
            != (prepared_sha256, prepared_size)
            or prepared_final != (prepared_sha256, prepared_size)
        ):
            raise RuntimeError(
                "input-preparation historical and durable cohorts differ"
            )
        proof_body = {
            "schema_version": (
                "production_input_preparation_dataset_provenance_rehydration_v1"
            ),
            "status": "complete",
            "request_sha256": self.request["request_sha256"],
            "input_preparation_phase_content_sha256": phase[
                "content_sha256"
            ],
            "historical_dataset_path": str(historical),
            "durable_dataset_path": str(prepared),
            "preparation_manifest_path": str(manifest_path),
            "dataset_sha256": prepared_sha256,
            "dataset_size_bytes": prepared_size,
            "copy_policy": "atomic_private_hardlink_publication_v1",
        }
        _atomic_write_json(
            self.options.work_root
            / "recovery"
            / "input_preparation_dataset_provenance_rehydration.json",
            {
                **proof_body,
                "content_sha256": _sha(proof_body),
            },
        )
        return historical, manifest_path

    def _stage1_input_preparation_validation_paths(
        self,
    ) -> tuple[Path, Path]:
        """Resolve relocation inputs without changing embedding-phase identity."""

        phase = self._validated_complete("input_preparation")
        if phase is None:
            raise RuntimeError("input-preparation phase is not complete")
        if phase.get("schema_version") != WORKFLOW_ADOPTED_PHASE_MANIFEST_SCHEMA:
            return self._input_preparation_validation_paths()

        prepared, manifest_path = self._input_preparation_paths()
        manifest = _read_json_object(
            manifest_path,
            label="adopted input-preparation manifest",
        )
        output = manifest.get("output")
        raw_historical = (
            output.get("path")
            if isinstance(output, Mapping)
            else None
        )
        if not isinstance(raw_historical, str) or not raw_historical.strip():
            raise RuntimeError(
                "adopted input preparation lacks its historical output path"
            )
        historical = Path(raw_historical)
        if historical == prepared:
            return prepared, manifest_path
        if (
            not historical.is_absolute()
            or Path(os.path.normpath(str(historical))) != historical
        ):
            raise RuntimeError(
                "adopted input-preparation historical path is not canonical"
            )

        published_root = Path(str(phase["attempt_dir"])).resolve(strict=True)
        try:
            prepared.relative_to(published_root)
        except ValueError as exc:
            raise RuntimeError(
                "adopted prepared cohort escaped its authenticated payload"
            ) from exc

        prepared_sha256, prepared_size = stable_file_sha256(prepared)
        output_body = {
            "path": str(historical),
            "sha256": prepared_sha256,
            "size_bytes": prepared_size,
        }
        manifest_body = {
            key: copy.deepcopy(value)
            for key, value in manifest.items()
            if key != "content_sha256"
        }
        manifest_digest = hashlib.sha256(
            json.dumps(
                manifest_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
                default=(
                    lambda item: (
                        item.item()
                        if hasattr(item, "item")
                        else str(item)
                    )
                ),
            ).encode("utf-8")
        ).hexdigest()
        if (
            manifest.get("content_sha256") != manifest_digest
            or output != output_body
        ):
            raise RuntimeError(
                "adopted input-preparation manifest is invalid"
            )
        if (
            not historical.exists()
            or historical.is_symlink()
            or not historical.is_file()
        ):
            raise RuntimeError(
                "adopted input-preparation historical cohort is unavailable; "
                "use a fresh local input-preparation phase"
            )
        historical_sha256, historical_size = stable_file_sha256(historical)
        prepared_final = stable_file_sha256(prepared)
        if (
            (historical_sha256, historical_size)
            != (prepared_sha256, prepared_size)
            or prepared_final != (prepared_sha256, prepared_size)
        ):
            raise RuntimeError(
                "adopted historical and durable prepared cohorts differ"
            )

        proof_body = {
            "schema_version": (
                "production_input_preparation_dataset_provenance_rehydration_v1"
            ),
            "status": "complete",
            "request_sha256": self.request["request_sha256"],
            "input_preparation_phase_content_sha256": phase[
                "content_sha256"
            ],
            "historical_dataset_path": str(historical),
            "durable_dataset_path": str(prepared),
            "preparation_manifest_path": str(manifest_path),
            "dataset_sha256": prepared_sha256,
            "dataset_size_bytes": prepared_size,
            "copy_policy": "adopted_producer_path_reopened_v1",
        }
        _atomic_write_json(
            self.options.work_root
            / "recovery"
            / "input_preparation_dataset_provenance_rehydration.json",
            {
                **proof_body,
                "content_sha256": _sha(proof_body),
            },
        )
        return historical, manifest_path

    def _embedding_cache_paths(self) -> tuple[Path, Path]:
        phase = self._validated_complete("embedding_cache")
        if phase is None:
            raise RuntimeError("embedding-cache phase is not complete")
        result = phase["result"]
        if result.get("schema_version") != EMBEDDING_CACHE_PHASE_SCHEMA:
            raise RuntimeError("embedding-cache phase has an unsupported result schema")
        cache = Path(result["cache_path"]).resolve(strict=True)
        prepared = Path(result["prepared_cohort_path"]).resolve(strict=True)
        phase_root = (
            Path(str(phase["attempt_dir"])).resolve(strict=True)
            if phase.get("schema_version") == WORKFLOW_ADOPTED_PHASE_MANIFEST_SCHEMA
            else (self.options.work_root / "phases" / "embedding_cache").resolve()
        )
        if any(path != phase_root and phase_root not in path.parents for path in (cache, prepared)):
            raise RuntimeError("embedding-cache outputs escaped their authenticated phase payload")
        registered = {Path(row["path"]).resolve(strict=True) for row in phase["artifacts"]}
        actual_cache_files = {
            path.resolve(strict=True) for path in cache.rglob("*") if path.is_file()
        }
        if not actual_cache_files or not actual_cache_files.issubset(registered):
            raise RuntimeError("embedding-cache files are not fully terminally registered")
        if prepared not in registered:
            raise RuntimeError("cache-bound prepared cohort is not terminally registered")
        return cache, prepared

    def _embedding_cache_dataset_provenance_path(
        self,
        *,
        cache: Path,
    ) -> Path:
        """Recover the cache's authenticated pre-publication cohort locator.

        Phase payload bytes are authenticated before the scratch attempt is
        atomically published under the durable root.  Embedded provenance is
        deliberately immutable payload, so its dataset locator continues to
        name that historical scratch file even after every byte has moved.
        Stage 1 still freshly authenticates the durable cohort's bytes, row
        order, and full text projection; this value supplies only the exact
        historical path expected in the already sealed cache metadata.
        """

        metadata_path = (cache / "metadata.json").resolve(strict=True)
        metadata = _read_json_object(
            metadata_path,
            label="published embedding-cache metadata",
        )
        provenance = metadata.get("production_provenance")
        dataset = (
            provenance.get("dataset")
            if isinstance(provenance, Mapping)
            else None
        )
        raw_path = (
            dataset.get("path")
            if isinstance(dataset, Mapping)
            else None
        )
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise RuntimeError(
                "published embedding cache lacks its historical cohort path"
            )
        historical = Path(raw_path)
        if (
            not historical.is_absolute()
            or Path(os.path.normpath(str(historical))) != historical
        ):
            raise RuntimeError(
                "published embedding cache has a noncanonical historical "
                "cohort path"
            )
        return historical

    def _embedding_cache_validation_dataset_path(
        self,
        *,
        cache: Path,
        prepared: Path,
    ) -> Path | None:
        """Rehydrate the exact scratch cohort named by fresh-cache metadata.

        The generic phase publisher moves authenticated bytes from scratch to
        durable storage without rewriting nested scientific payloads.  A
        freshly built embedding cache therefore retains the path of its
        byte-identical scratch cohort.  Restore only that workflow-owned cohort
        file so the cache's original strict path-bound validator can run
        unchanged.  Imported/relocated caches use their existing relocation
        proof instead and never enter this path.
        """

        historical = self._embedding_cache_dataset_provenance_path(
            cache=cache,
        )
        if historical == prepared:
            return None
        phase = self._validated_complete("embedding_cache")
        if (
            phase is None
            or phase.get("schema_version") != WORKFLOW_PHASE_MANIFEST_SCHEMA
        ):
            raise RuntimeError(
                "an adopted path-relocated embedding cache requires the "
                "authenticated cache-import relocation path"
            )
        published_root = Path(str(phase["attempt_dir"])).resolve(strict=True)
        try:
            prepared_relative = prepared.relative_to(published_root)
        except ValueError as exc:
            raise RuntimeError(
                "published cache-bound cohort escaped its phase attempt"
            ) from exc
        configured_scratch = self.options.scratch_root
        if configured_scratch is None:
            configured_scratch = (
                self.options.work_root.parent
                / f".{self.options.work_root.name}.scratch"
            )
        configured_scratch.mkdir(parents=True, exist_ok=True)
        scratch_root = configured_scratch.resolve(strict=True)
        if configured_scratch.is_symlink() or scratch_root != configured_scratch:
            raise RuntimeError(
                "embedding-cache provenance recovery requires one canonical "
                "non-symlink scratch root"
            )
        expected_historical = (
            scratch_root
            / "production_all_evidence_workflow"
            / str(self.request["request_sha256"])
            / "embedding_cache"
            / published_root.name
            / prepared_relative
        )
        if historical != expected_historical:
            raise RuntimeError(
                "embedding-cache historical cohort path is not owned by this "
                "workflow request; use authenticated cache import/relocation"
            )
        current = historical.parent
        while current != scratch_root:
            if current.exists() or current.is_symlink():
                state = os.lstat(current)
                if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(
                    state.st_mode
                ):
                    raise RuntimeError(
                        "embedding-cache provenance recovery encountered a "
                        "non-directory scratch ancestor"
                    )
            current = current.parent
        historical.parent.mkdir(parents=True, exist_ok=True)
        if historical.parent.resolve(strict=True) != historical.parent:
            raise RuntimeError(
                "embedding-cache provenance recovery parent is not canonical"
            )

        prepared_sha256, prepared_size = stable_file_sha256(prepared)
        if historical.exists() or historical.is_symlink():
            if historical.is_symlink() or not historical.is_file():
                raise RuntimeError(
                    "embedding-cache historical cohort recovery target is not "
                    "one regular file"
                )
            historical_sha256, historical_size = stable_file_sha256(
                historical
            )
        else:
            temporary = historical.parent / (
                f".{historical.name}.rehydrating_{os.getpid()}_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            )
            digest = hashlib.sha256()
            copied = 0
            try:
                with prepared.open("rb") as source, temporary.open(
                    "xb"
                ) as destination:
                    while True:
                        block = source.read(8 * 1024 * 1024)
                        if not block:
                            break
                        destination.write(block)
                        digest.update(block)
                        copied += len(block)
                    destination.flush()
                    os.fsync(destination.fileno())
                if (
                    copied != prepared_size
                    or digest.hexdigest() != prepared_sha256
                ):
                    raise RuntimeError(
                        "durable cohort changed during provenance recovery"
                    )
                try:
                    os.link(temporary, historical)
                except FileExistsError:
                    pass
                _fsync_directory(historical.parent)
            finally:
                temporary.unlink(missing_ok=True)
            historical_sha256, historical_size = stable_file_sha256(
                historical
            )
        prepared_final_sha256, prepared_final_size = stable_file_sha256(
            prepared
        )
        if (
            (prepared_final_sha256, prepared_final_size)
            != (prepared_sha256, prepared_size)
            or (historical_sha256, historical_size)
            != (prepared_sha256, prepared_size)
        ):
            raise RuntimeError(
                "embedding-cache historical and durable cohorts differ"
            )
        preparation_cohort, preparation_manifest_path = (
            self._input_preparation_paths()
        )
        preparation_sha256, preparation_size = stable_file_sha256(
            preparation_cohort
        )
        if (preparation_sha256, preparation_size) != (
            prepared_sha256,
            prepared_size,
        ):
            raise RuntimeError(
                "embedding-cache cohort differs from its authenticated text "
                "preparation output"
            )
        source_preparation_manifest = _read_json_object(
            preparation_manifest_path,
            label="embedding-cache source preparation manifest",
        )
        source_preparation_body = {
            key: copy.deepcopy(value)
            for key, value in source_preparation_manifest.items()
            if key != "content_sha256"
        }
        preparation_digest = lambda value: hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
                default=(
                    lambda item: (
                        item.item()
                        if hasattr(item, "item")
                        else str(item)
                    )
                ),
            ).encode("utf-8")
        ).hexdigest()
        source_output = source_preparation_manifest.get("output")
        input_preparation_phase = self._validated_complete(
            "input_preparation"
        )
        if (
            input_preparation_phase is None
            or input_preparation_phase.get("schema_version")
            != WORKFLOW_PHASE_MANIFEST_SCHEMA
        ):
            raise RuntimeError(
                "embedding-cache source preparation must be a directly "
                "published workflow phase"
            )
        input_preparation_root = Path(
            str(input_preparation_phase["attempt_dir"])
        ).resolve(strict=True)
        try:
            preparation_relative = preparation_cohort.relative_to(
                input_preparation_root
            )
        except ValueError as exc:
            raise RuntimeError(
                "text-preparation cohort escaped its published phase"
            ) from exc
        expected_preparation_output_path = (
            scratch_root
            / "production_all_evidence_workflow"
            / str(self.request["request_sha256"])
            / "input_preparation"
            / input_preparation_root.name
            / preparation_relative
        )
        if (
            source_preparation_manifest.get("content_sha256")
            != preparation_digest(source_preparation_body)
            or not isinstance(source_output, Mapping)
            or source_output
            != {
                "path": str(expected_preparation_output_path),
                "sha256": preparation_sha256,
                "size_bytes": preparation_size,
            }
        ):
            raise RuntimeError(
                "embedding-cache source preparation manifest is invalid"
            )
        recovered_preparation_body = copy.deepcopy(
            source_preparation_body
        )
        recovered_preparation_body["output"] = {
            "path": str(historical),
            "sha256": historical_sha256,
            "size_bytes": historical_size,
        }
        recovered_preparation_manifest = {
            **recovered_preparation_body,
            "content_sha256": preparation_digest(
                recovered_preparation_body
            ),
        }
        recovered_preparation_path = (
            historical.parent / "preparation_manifest.json"
        )
        durable_recovered_preparation_path = (
            self.options.work_root
            / "recovery"
            / "embedding_cache_source_preparation_manifest.json"
        )
        _write_immutable_json(
            durable_recovered_preparation_path,
            recovered_preparation_manifest,
        )
        _write_immutable_json(
            recovered_preparation_path,
            recovered_preparation_manifest,
        )
        proof_body = {
            "schema_version": (
                "production_embedding_cache_dataset_provenance_rehydration_v1"
            ),
            "status": "complete",
            "request_sha256": self.request["request_sha256"],
            "embedding_cache_phase_content_sha256": phase["content_sha256"],
            "historical_dataset_path": str(historical),
            "durable_dataset_path": str(prepared),
            "dataset_sha256": prepared_sha256,
            "dataset_size_bytes": prepared_size,
            "historical_preparation_manifest_path": str(
                recovered_preparation_path
            ),
            "durable_recovered_preparation_manifest_path": str(
                durable_recovered_preparation_path
            ),
            "historical_preparation_manifest_content_sha256": (
                recovered_preparation_manifest["content_sha256"]
            ),
            "copy_policy": "atomic_private_hardlink_publication_v1",
        }
        proof = {**proof_body, "content_sha256": _sha(proof_body)}
        _atomic_write_json(
            self.options.work_root
            / "recovery"
            / "embedding_cache_dataset_provenance_rehydration.json",
            proof,
        )
        return historical

    def _embedding_cache_relocation_options(
        self,
        *,
        cache: Path,
        prepared: Path,
    ) -> Any | None:
        """Reconstruct the exact relocation proof input consumed by Stage 1."""

        if self.options.embedding_cache_import is None:
            return None
        from .production_embedding_cache_relocation import (
            ProductionEmbeddingCacheRelocationOptions,
        )

        phase = self._validated_complete("embedding_cache")
        if phase is None:
            raise RuntimeError("embedding-cache relocation phase is not complete")
        result = phase["result"]
        identity = result.get("cache_identity")
        if result.get("mode") != "authenticated_relocation" or not isinstance(identity, Mapping):
            raise RuntimeError("cache import did not produce an authenticated relocation")
        target = Path(str(identity.get("root", ""))).resolve(strict=True)
        if cache.parent != target or prepared.parent.parent != target:
            raise RuntimeError("relocated cache result paths differ from its sealed root")
        fresh_prepared, fresh_manifest = (
            self._stage1_input_preparation_validation_paths()
        )
        source_prepared, source_manifest = self._resolved_cache_import_sources()
        return ProductionEmbeddingCacheRelocationOptions(
            source_cache_dir=self.options.embedding_cache_import,
            source_prepared_cohort_path=source_prepared,
            source_preparation_manifest_path=source_manifest,
            fresh_prepared_cohort_path=fresh_prepared,
            fresh_preparation_manifest_path=fresh_manifest,
            local_model_path=self.options.embedding_local_model_path,
            target_dir=target,
            unit_id_column=self.options.unit_id_column,
            text_column=self.options.text_column,
            treatment_column=self.options.treatment_column,
            outcome_column=self.options.outcome_column,
            sentence_model_name=self.options.embedding_model_name,
            chunk_configuration=self._embedding_chunk_configuration(),
        )

    def _embedding_cache_relocation_prepublication_root(
        self,
        *,
        cache: Path,
    ) -> Path | None:
        """Return the exact root sealed before generic phase publication."""

        phase = self._validated_complete("embedding_cache")
        if phase is None:
            raise RuntimeError("embedding-cache phase is not complete")
        result = phase.get("result")
        if (
            not isinstance(result, Mapping)
            or result.get("mode") != "authenticated_relocation"
        ):
            return None
        durable_root = cache.parent
        terminal_path = (
            durable_root
            / "complete_manifest.json"
        ).resolve(strict=True)
        terminal = _read_json_object(
            terminal_path,
            label="published embedding-cache relocation terminal",
        )
        raw_root = terminal.get("root")
        if not isinstance(raw_root, str) or not raw_root.strip():
            raise RuntimeError(
                "published embedding-cache relocation lacks its producer root"
            )
        historical = Path(raw_root)
        if (
            not historical.is_absolute()
            or Path(os.path.normpath(str(historical))) != historical
        ):
            raise RuntimeError(
                "published embedding-cache relocation has a noncanonical "
                "producer root"
            )
        if historical == durable_root:
            return None
        if phase.get("schema_version") == WORKFLOW_PHASE_MANIFEST_SCHEMA:
            configured_scratch = self.options.scratch_root
            if configured_scratch is None:
                configured_scratch = (
                    self.options.work_root.parent
                    / f".{self.options.work_root.name}.scratch"
                )
            scratch_root = configured_scratch.resolve(strict=True)
            published_attempt = Path(
                str(phase["attempt_dir"])
            ).resolve(strict=True)
            expected = (
                scratch_root
                / "production_all_evidence_workflow"
                / str(self.request["request_sha256"])
                / "embedding_cache"
                / published_attempt.name
                / durable_root.relative_to(published_attempt)
            )
            if historical != expected:
                raise RuntimeError(
                    "embedding-cache relocation producer root is not the "
                    "exact workflow publication source"
                )
        elif (
            phase.get("schema_version")
            != WORKFLOW_ADOPTED_PHASE_MANIFEST_SCHEMA
        ):
            raise RuntimeError(
                "embedding-cache relocation phase schema is unsupported"
            )
        return historical

    def _stage1_preflight_paths(self) -> tuple[Path, Path]:
        phase = self._validated_complete("stage1_preflight")
        if phase is None:
            raise RuntimeError("Stage 1 scientific preflight is not complete")
        result = phase["result"]
        if result.get("schema_version") != STAGE1_PREFLIGHT_PHASE_SCHEMA:
            raise RuntimeError("Stage 1 preflight phase has an unsupported schema")
        profile = Path(str(result.get("effective_profile_path", ""))).resolve(strict=True)
        manifest = Path(str(result.get("cluster_preflight_manifest_path", ""))).resolve(strict=True)
        registered = {Path(row["path"]).resolve(strict=True) for row in phase["artifacts"]}
        if (
            profile not in registered
            or manifest not in registered
            or profile.name != "effective_stage1_profile.json"
            or manifest.name != "cluster_preflight_manifest.json"
        ):
            raise RuntimeError("Stage 1 preflight profile/artifact is not terminally sealed")
        return profile, manifest

    def _stage1_preflight_state_bundle_path(self) -> Path:
        phase = self._validated_complete("stage1_preflight")
        if phase is None:
            raise RuntimeError("Stage 1 scientific preflight is not complete")
        result = phase["result"]
        supplied = result.get("cluster_preflight_state_bundle_manifest_path")
        if not isinstance(supplied, str) or not supplied:
            raise RuntimeError(
                "Stage 1 preflight is a legacy audit-only checkpoint and "
                "has no reusable KMeans/SVD state"
            )
        manifest = Path(supplied).resolve(strict=True)
        registered = {Path(row["path"]).resolve(strict=True) for row in phase["artifacts"]}
        if (
            manifest not in registered
            or manifest.name != "cluster_state_bundle_manifest.json"
            or result.get("cluster_preflight_states_are_canonical_no_refit") is not True
        ):
            raise RuntimeError("Stage 1 clustered fitted-state bundle is not terminally sealed")
        return manifest

    def _stage1_prepared_context_path(self) -> Path:
        """Return the terminally registered reusable prepared-context manifest."""

        phase = self._validated_complete("stage1_preflight")
        if phase is None:
            raise RuntimeError("Stage 1 scientific preflight is not complete")
        result = phase["result"]
        supplied = result.get("prepared_stage1_context_manifest_path")
        if not isinstance(supplied, str) or not supplied:
            raise RuntimeError(
                "Stage 1 preflight has no reusable prepared context"
            )
        manifest = Path(supplied).resolve(strict=True)
        registered = {
            Path(row["path"]).resolve(strict=True)
            for row in phase["artifacts"]
        }
        if (
            manifest not in registered
            or manifest.name
            != "prepared_stage1_context_manifest.json"
        ):
            raise RuntimeError(
                "prepared Stage 1 context is not terminally registered"
            )
        from .prepared_stage1_context import (
            load_prepared_stage1_context,
            rebind_prepared_stage1_context_locators,
            serialize_stage1_build_options,
        )

        context = load_prepared_stage1_context(manifest)
        cache, prepared = self._embedding_cache_paths()
        profile, preflight_manifest = self._stage1_preflight_paths()
        state_manifest = self._stage1_preflight_state_bundle_path()
        current_options = self._stage1_build_options(
            dataset=prepared,
            profile=profile,
            cache=cache,
            output=(
                self.options.work_root
                / "recovery"
                / "prepared_stage1_context_runtime"
            ).resolve(),
            dry_run=False,
            cluster_preflight_manifest_path=preflight_manifest,
            cluster_preflight_state_bundle_manifest_path=state_manifest,
        )
        current_mapping = serialize_stage1_build_options(
            current_options
        )
        sealed_mapping = context.execution_locators[
            "stage1_build_options"
        ]
        # Scientific equality is enforced by the context root independently.
        # Compare the complete execution-locator envelope here: an adopted run
        # may reuse byte-identical inputs while requiring new output,
        # descriptor, attempt, and progress roots. Returning the producer's
        # locator artifact in that case would let the consumer write into the
        # producer checkpoint.
        if sealed_mapping == current_mapping:
            return manifest

        exact_request = copy.deepcopy(
            dict(context.execution_locators["exact_stage1_request"])
        )
        exact_request["dataset"]["path"] = str(
            Path(current_options.dataset_path).resolve(strict=True)
        )
        exact_request["source_config"]["path"] = str(
            Path(current_options.config_path).resolve(strict=True)
        )
        exact_request["embedding_cache"]["path"] = str(
            Path(current_options.embedding_cache_dir).resolve(strict=True)
        )
        exact_request["htr_model"]["path"] = str(
            self.options.htr_local_model_path.resolve(strict=True)
        )
        runtime = exact_request["runtime"]
        runtime.update(
            {
                "device": current_options.device,
                "gpu_ids": list(current_options.gpu_ids),
                "num_workers": current_options.num_workers,
                "tfidf_workers": current_options.tfidf_workers,
                "tfidf_parallel_backend": (
                    current_options.tfidf_parallel_backend
                ),
                "query_devices": list(current_options.query_devices),
                "query_nuisance_folds": (
                    current_options.query_nuisance_folds
                ),
                "scope_workers_per_gpu": (
                    current_options.scope_workers_per_gpu
                ),
                "preflight_workers": current_options.preflight_workers,
                "scope_descriptor_root": str(
                    Path(
                        current_options.stage1_scope_descriptor_root
                        or (
                            Path(current_options.output_dir)
                            / "stage1_scope_recovery"
                            / "descriptor"
                        )
                    ).resolve()
                ),
                "scope_attempt_root": str(
                    Path(
                        current_options.stage1_scope_attempt_root
                        or (
                            Path(current_options.output_dir)
                            / "stage1_scope_recovery"
                            / "attempts"
                        )
                    ).resolve()
                ),
                "scope_progress_path": str(
                    Path(
                        current_options.stage1_scope_progress_path
                        or (
                            Path(current_options.output_dir)
                            / "stage1_scope_recovery"
                            / "progress.json"
                        )
                    ).resolve()
                ),
            }
        )
        request_body = {
            key: copy.deepcopy(value)
            for key, value in exact_request.items()
            if key != "request_sha256"
        }
        exact_request["request_sha256"] = _sha(request_body)
        rebound_root = (
            self.options.work_root
            / "recovery"
            / "adopted_prepared_stage1_context"
        ).resolve()
        rebound = rebind_prepared_stage1_context_locators(
            source_manifest_path=manifest,
            output_root=rebound_root,
            stage1_build_options=current_mapping,
            exact_stage1_request=exact_request,
        )
        if rebound.content_root_sha256 != context.content_root_sha256:
            raise RuntimeError(
                "adopted prepared-context scientific root changed"
            )
        return rebound.manifest_path

    def _reusable_preflight_store_root(self) -> Path:
        configured = self.options.scratch_root
        if configured is None:
            configured = (
                self.options.work_root.parent
                / ".production_all_evidence_reusable"
            )
        return (
            Path(configured)
            / "production_all_evidence_workflow"
            / "stage1_reusable_preflight_store_v2"
        ).resolve()

    def _reusable_preflight_accepted_input_selector(
        self,
    ) -> dict[str, Any]:
        """Derive a path/resource/Stage2-neutral preflight lookup key."""

        if self.request.get("portable_typed_workflow") is not True:
            raise RuntimeError(
                "reusable preflight accepted-input selection requires a "
                "typed portable workflow"
            )
        cache_phase = self._validated_complete("embedding_cache")
        cache_identity = (
            cache_phase.get("result", {}).get("cache_identity")
            if isinstance(cache_phase, Mapping)
            else None
        )
        settings = self.request.get("scientific_settings")
        architectures = (
            settings.get("architecture_profiles")
            if isinstance(settings, Mapping)
            else None
        )
        if (
            not isinstance(cache_identity, Mapping)
            or not isinstance(settings, Mapping)
            or not isinstance(architectures, Mapping)
        ):
            raise RuntimeError(
                "workflow request lacks reusable-preflight scientific inputs"
            )
        htr_profile = architectures.get("hierarchical_transformer")
        cluster_profile = architectures.get(
            "cluster_local_embeddings"
        )
        cluster_producer_configuration = (
            cluster_profile.get("producer_configuration")
            if isinstance(cluster_profile, Mapping)
            else None
        )
        if (
            not isinstance(htr_profile, Mapping)
            or not isinstance(cluster_profile, Mapping)
            or not isinstance(
                cluster_producer_configuration,
                Mapping,
            )
        ):
            raise RuntimeError(
                "workflow request lacks HTR/cluster preflight science"
            )
        raw_stage1_profile = json.loads(
            self.options.stage1_profile_path.read_text(
                encoding="utf-8"
            )
        )
        raw_stage1_config = raw_stage1_profile.get(
            "config",
            raw_stage1_profile,
        )
        raw_architecture = (
            raw_stage1_config.get("architecture")
            if isinstance(raw_stage1_config, Mapping)
            else None
        )
        raw_forest = (
            raw_architecture.get("multi_model_forest")
            if isinstance(raw_architecture, Mapping)
            else None
        )
        raw_embedding = (
            raw_forest.get("embedding_contrast")
            if isinstance(raw_forest, Mapping)
            else None
        )
        if not all(
            isinstance(value, Mapping)
            for value in (
                raw_architecture,
                raw_forest,
                raw_embedding,
            )
        ):
            raise RuntimeError(
                "Stage 1 profile lacks preflight architecture inputs"
            )
        htr_chunking = {
            name: copy.deepcopy(raw_architecture[name])
            for name in (
                "htr_chunk_size_words",
                "htr_chunk_overlap_words",
                "htr_max_chunks",
                "htr_max_chunk_length",
            )
        }
        embedding_for_preflight = copy.deepcopy(
            dict(raw_embedding)
        )
        embedding_for_preflight.update(
            {
                **{
                    name: copy.deepcopy(value)
                    for name, value in (
                        self._embedding_chunk_configuration()
                    ).items()
                    if name
                    in {
                        "chunk_size_words",
                        "chunk_overlap_words",
                        "max_chunks",
                        "chunk_selection",
                        "normalize_embeddings",
                        "max_seq_length",
                    }
                },
                "model_name": self.options.embedding_model_name,
                "cache_dir": "reusable-preflight://frozen-cache",
                "cluster_local_scientific": copy.deepcopy(
                    dict(cluster_producer_configuration)
                ),
            }
        )
        from .production_stage1_bundle import (
            STAGE1_REUSABLE_CLUSTER_OWNER_PRODUCER_IDENTITY,
            STAGE1_REUSABLE_GLOBAL_AUDIT_PRODUCER_IDENTITY,
            _embedding_cluster_preflight_scientific_configuration,
            _htr_tokenizer_scientific_identity,
        )
        cluster_preflight_configuration = (
            _embedding_cluster_preflight_scientific_configuration(
                {
                    "architecture": {
                        "multi_model_forest": {
                            "embedding_contrast": (
                                embedding_for_preflight
                            )
                        }
                    }
                }
            )
        )

        htr_tokenizer_identity = (
            _htr_tokenizer_scientific_identity(
                self.options.htr_local_model_path
            )
        )
        semantic_witness_identity = None
        if self.options.portable_scientific_spec is not None:
            from .review_spent_evidence_provider import (
                semantic_witness_config_from_portable_scientific_spec,
            )

            semantic_witness_identity = (
                semantic_witness_config_from_portable_scientific_spec(
                    self.options.portable_scientific_spec
                ).as_dict()
            )
        body = {
            "schema_version": (
                "production_stage1_preflight_accepted_input_selector_v2"
            ),
            "prepared_dataset_and_embedding_cache": (
                _reusable_preflight_cache_selector(cache_identity)
            ),
            "source_dataset_content_sha256": self.request[
                "source_sha256"
            ],
            "row_order_identity": copy.deepcopy(
                self.request[
                    "expected_checkpoint_compatibility"
                ]["row_order_identity"]
            ),
            "columns": copy.deepcopy(settings["columns"]),
            "preprocessing": copy.deepcopy(
                settings["preprocessing"]
            ),
            "folds": copy.deepcopy(settings["folds"]),
            "seed": settings["seed"],
            "seed_policy": settings["seed_policy"],
            "split_and_owner_derivation": {
                "row_order_identity": copy.deepcopy(
                    self.request[
                        "expected_checkpoint_compatibility"
                    ]["row_order_identity"]
                ),
                "folds": copy.deepcopy(settings["folds"]),
                "global_seed": settings["seed"],
                "seed_policy": settings["seed_policy"],
                "initial_training_partitions": int(
                    self.options.initial_training_partitions
                ),
                "review_rounds": int(self.options.review_rounds),
                "deduplication_policy": (
                    "identical_ordered_fit_rows_and_canonical_seed_"
                    "earliest_scope_owner_v1"
                ),
                "all_ten_physical_fit_identity_included": False,
            },
            "htr_nontruncation_configuration": htr_chunking,
            "cluster_preflight_scientific_configuration": (
                cluster_preflight_configuration
            ),
            "semantic_witness_scientific_configuration": (
                semantic_witness_identity
            ),
            "htr_model": _path_neutral_identity(
                self.request["htr_model_tree"]
            ),
            "htr_tokenizer_identity": htr_tokenizer_identity,
            "numerical_runtime_class": {
                "numpy": importlib.metadata.version("numpy"),
                "sklearn": importlib.metadata.version(
                    "scikit-learn"
                ),
                "runtime_compatibility_class": (
                    self.options.runtime_compatibility_class
                ),
            },
            "producer_and_schema_identities": {
                "global_audit": (
                    STAGE1_REUSABLE_GLOBAL_AUDIT_PRODUCER_IDENTITY
                ),
                "cluster_owner": (
                    STAGE1_REUSABLE_CLUSTER_OWNER_PRODUCER_IDENTITY
                ),
                "assembled": (
                    STAGE1_REUSABLE_ASSEMBLED_PREFLIGHT_PRODUCER_IDENTITY
                ),
                "store": (
                    "production_stage1_reusable_preflight_store_v2"
                ),
            },
            "stage2_identity_included": False,
            "all_ten_physical_fit_identity_included": False,
            "operational_paths_included": False,
            "resource_assignment_included": False,
        }
        return {**body, "content_sha256": _sha(body)}

    def _stage1_build_options(
        self,
        *,
        dataset: Path,
        profile: Path,
        cache: Path,
        output: Path,
        dry_run: bool,
        cluster_preflight_manifest_path: Path | None = None,
        cluster_preflight_state_bundle_manifest_path: Path | None = None,
        reusable_preflight_fast_reopen: bool = False,
    ) -> Stage1BundleBuildOptions:
        from .production_stage1_scope_scheduler import (
            Stage1PhysicalFitIdentity,
        )

        physical_fit_identity = self.request.get(
            "stage1_physical_fit_identity"
        )
        if not isinstance(physical_fit_identity, Mapping):
            raise RuntimeError(
                "immutable request lacks its Stage 1 physical-fit identity"
            )
        cache_phase = self._validated_complete("embedding_cache")
        if cache_phase is None:
            raise RuntimeError(
                "Stage 1 options require the completed embedding-cache phase"
            )
        cache_phase_result = cache_phase.get("result")
        if not isinstance(cache_phase_result, Mapping):
            raise RuntimeError(
                "embedding-cache phase result is invalid"
            )
        legacy_migration_identity = cache_phase_result.get(
            "legacy_terminal_migration_identity"
        )
        if (
            legacy_migration_identity is not None
            and not isinstance(legacy_migration_identity, Mapping)
        ):
            raise RuntimeError(
                "embedding-cache legacy migration identity is invalid"
            )
        trusted_cache_read_proof: Mapping[str, Any] | None = None
        adopted_cache = self._adopted_record_for_phase(
            "embedding_cache"
        )
        if adopted_cache is not None:
            adopted_cache_record, _locator = adopted_cache
            if _operator_trusted_adoption_selected(
                adopted_cache_record
            ):
                artifact_id = str(
                    adopted_cache_record["artifact_id"]
                )
                trusted_checkpoint = (
                    self._operator_trusted_checkpoint_handles.get(
                        artifact_id
                    )
                )
                cache_identity = cache_phase_result.get(
                    "cache_identity"
                )
                cache_build_identity = (
                    cache_identity.get("cache_build_identity")
                    if isinstance(cache_identity, Mapping)
                    else None
                )
                provider_identity = (
                    cache_build_identity.get("provider_identity")
                    if isinstance(cache_build_identity, Mapping)
                    else None
                )
                if (
                    trusted_checkpoint is None
                    or not isinstance(cache_build_identity, Mapping)
                    or not isinstance(provider_identity, Mapping)
                    or not isinstance(
                        legacy_migration_identity,
                        Mapping,
                    )
                ):
                    raise RuntimeError(
                        "operator-trusted embedding-cache phase lacks its "
                        "cache identity, migration proof, or live trust handle"
                    )
                from .operator_trusted_embedding_cache_reader import (
                    build_operator_trusted_cache_read_proof,
                )

                trusted_cache_read_proof = (
                    build_operator_trusted_cache_read_proof(
                        trusted_checkpoint,
                        cache_dir=cache,
                        cache_build_identity=cache_build_identity,
                        provider_identity=provider_identity,
                        migration_identity=legacy_migration_identity,
                    )
                )
        cache_relocation = None
        if (
            trusted_cache_read_proof is None
            and not reusable_preflight_fast_reopen
        ):
            cache_relocation = self._embedding_cache_relocation_options(
                cache=cache,
                prepared=dataset,
            )
        values: dict[str, Any] = {
            "dataset_path": dataset,
            "config_path": profile,
            "embedding_cache_dir": cache,
            "embedding_local_model_path": None,
            "embedding_cache_output_dir": None,
            "embedding_cache_relocation_prepublication_root": (
                None
                if (
                    cache_relocation is None
                    or reusable_preflight_fast_reopen
                )
                else self._embedding_cache_relocation_prepublication_root(
                    cache=cache,
                )
            ),
            "embedding_cache_validation_dataset_path": (
                None
                if trusted_cache_read_proof is not None
                or cache_relocation is not None
                or reusable_preflight_fast_reopen
                else self._embedding_cache_validation_dataset_path(
                    cache=cache,
                    prepared=dataset,
                )
            ),
            "embedding_cache_configuration": copy.deepcopy(
                dict(self._embedding_chunk_configuration())
            ),
            "embedding_cache_legacy_migration_identity": (
                None
                if legacy_migration_identity is None
                else copy.deepcopy(dict(legacy_migration_identity))
            ),
            "embedding_cache_operator_trusted_read_proof": (
                None
                if trusted_cache_read_proof is None
                else copy.deepcopy(dict(trusted_cache_read_proof))
            ),
            "output_dir": output,
            "unit_id_column": self.options.unit_id_column,
            "physical_fit_identity": (
                Stage1PhysicalFitIdentity.from_mapping(
                    physical_fit_identity
                )
            ),
            "seed": self.options.seed,
            "initial_training_partitions": (self.options.initial_training_partitions),
            "device": self.options.stage1_device,
            "gpu_ids": self.stage1_gpu_ids,
            "num_workers": self.options.num_workers,
            "tfidf_workers": self.options.tfidf_workers,
            "tfidf_parallel_backend": self.options.tfidf_parallel_backend,
            "query_devices": self.query_devices,
            "query_nuisance_folds": self.options.interaction_inner_folds,
            "query_config_path": self.options.query_profile_path,
            "resume": False,
            "dry_run": dry_run,
        }
        if "embedding_cache_relocation" in Stage1BundleBuildOptions.__dataclass_fields__:
            values["embedding_cache_relocation"] = (
                cache_relocation
            )
        # Parallel scheduler fields are passed automatically as soon as the
        # Stage1BundleBuildOptions API exposes them.  This keeps the workflow
        # interface independently testable while that implementation lands.
        available = Stage1BundleBuildOptions.__dataclass_fields__
        if len(self.stage1_gpu_ids) > 1 and "scope_workers_per_gpu" not in available:
            raise RuntimeError(
                "multiple Stage 1 GPUs require the canonical scope scheduler; "
                "this builder does not expose it"
            )
        if (
            cluster_preflight_manifest_path is not None
            and "cluster_preflight_manifest_path" not in available
        ):
            raise RuntimeError(
                "supervised Stage 1 requires the independently sealed scientific "
                "preflight consumer API"
            )
        optional_bindings = {
            "scope_workers_per_gpu": self.options.stage1_scope_workers_per_gpu,
            "preflight_workers": self.options.stage1_preflight_workers,
            "preflight_execution_attestation": copy.deepcopy(
                self.options.stage1_preflight_execution_attestation
            ),
            "portable_cluster_preflight_v2": (
                self.options.portable_scientific_spec is not None
            ),
            "scope_seed_policy": self.options.stage1_seed_policy,
            "cluster_preflight_manifest_path": cluster_preflight_manifest_path,
            "cluster_preflight_state_bundle_manifest_path": (
                cluster_preflight_state_bundle_manifest_path
            ),
            "reusable_preflight_store_root": (
                self._reusable_preflight_store_root()
            ),
            "stage1_scope_attempt_root": (
                self.options.work_root / "recovery" / "stage1_scope_attempts"
            ).resolve(),
            "stage1_scope_progress_path": (
                self.options.work_root / "recovery" / "stage1_scope_progress.json"
            ).resolve(),
        }
        selected_preflight = self.request.get(
            "legacy_preflight_candidate_identity"
        )
        if (
            isinstance(selected_preflight, Mapping)
            and selected_preflight.get("candidate_kind")
            == "portable_v2"
            and cluster_preflight_manifest_path is None
        ):
            optional_bindings.update(
                {
                    "reusable_preflight_import_manifest_path": Path(
                        str(selected_preflight["manifest_path"])
                    ).resolve(strict=True),
                    "reusable_preflight_import_state_bundle_manifest_path": (
                        Path(
                            str(
                                selected_preflight[
                                    "state_bundle_manifest_path"
                                ]
                            )
                        ).resolve(strict=True)
                    ),
                }
            )
        if (
            self.options.portable_scientific_spec is not None
            and "semantic_witness_scientific_config" in available
        ):
            from .review_spent_evidence_provider import (
                semantic_witness_config_from_portable_scientific_spec,
            )

            optional_bindings["semantic_witness_scientific_config"] = (
                semantic_witness_config_from_portable_scientific_spec(
                    self.options.portable_scientific_spec
                )
            )
        values.update({key: value for key, value in optional_bindings.items() if key in available})
        return Stage1BundleBuildOptions(**values)

    def _run_embedding_cache_phase(self, attempt: Path) -> Mapping[str, Any]:
        o = self.options
        resource = self._gpu_preflight()
        fresh_prepared, fresh_preparation_manifest = self._input_preparation_paths()
        worker_execution: Mapping[str, Any] | None = None
        if o.embedding_cache_import is not None:
            (
                fresh_prepared,
                fresh_preparation_manifest,
            ) = self._input_preparation_validation_paths()
            from .production_embedding_cache_relocation import (
                ProductionEmbeddingCacheRelocationOptions,
                relocate_authenticated_production_embedding_cache,
                validate_relocated_production_embedding_cache,
            )

            source_prepared, source_preparation_manifest = self._resolved_cache_import_sources()
            relocation_options = ProductionEmbeddingCacheRelocationOptions(
                source_cache_dir=o.embedding_cache_import,
                source_prepared_cohort_path=source_prepared,
                source_preparation_manifest_path=source_preparation_manifest,
                fresh_prepared_cohort_path=fresh_prepared,
                fresh_preparation_manifest_path=fresh_preparation_manifest,
                local_model_path=o.embedding_local_model_path,
                target_dir=(attempt / "relocated_cache").resolve(),
                unit_id_column=o.unit_id_column,
                text_column=o.text_column,
                treatment_column=o.treatment_column,
                outcome_column=o.outcome_column,
                sentence_model_name=o.embedding_model_name,
                chunk_configuration=self._embedding_chunk_configuration(),
            )
            built = relocate_authenticated_production_embedding_cache(relocation_options)
            validated = validate_relocated_production_embedding_cache(relocation_options)
            if built.identity() != validated.identity():
                raise RuntimeError("relocated embedding cache changed during fresh validation")
            identity = validated.identity()
            cache_path = validated.cache_dir
            prepared_path = validated.prepared_cohort_path
            terminal_files = [
                str(path)
                for path in (
                    *sorted(validated.cache_dir.iterdir()),
                    validated.prepared_cohort_path,
                    validated.attestation_path,
                    validated.terminal_manifest_path,
                )
                if path.is_file()
            ]
            mode = "authenticated_relocation"
        else:
            from .production_embedding_cache_process import (
                build_production_embedding_cache_in_spawned_worker,
            )

            cache_path = (attempt / "embedding_cache").resolve()
            prepared_copy = attempt / "prepared"
            prepared_copy.mkdir()
            prepared_path = prepared_copy / "modeling_cohort.parquet"
            # Build and validate against the exact artifact Stage 1 will read;
            # the production cache provenance intentionally binds its path.
            import shutil

            shutil.copyfile(fresh_prepared, prepared_path)
            built = build_production_embedding_cache_in_spawned_worker(
                dataset_path=prepared_path,
                text_column=o.text_column,
                local_model_path=o.embedding_local_model_path,
                sentence_model_name=o.embedding_model_name,
                chunk_configuration=self._embedding_chunk_configuration(),
                target_dir=cache_path,
                device=o.stage1_device,
                batch_size=int(o.embedding_batch_size),
                cpu_budget=int(o.cpu_budget),
            )
            identity = built.identity()
            worker_execution = copy.deepcopy(
                dict(built.execution_attestation)
            )
            terminal_files = [
                *(str(path) for path in sorted(cache_path.iterdir()) if path.is_file()),
                str(prepared_path),
            ]
            mode = "fresh_build"
        return {
            "schema_version": EMBEDDING_CACHE_PHASE_SCHEMA,
            "mode": mode,
            "cache_path": str(Path(cache_path).resolve(strict=True)),
            "prepared_cohort_path": str(Path(prepared_path).resolve(strict=True)),
            "cache_identity": identity,
            "resource_preflight": resource,
            "embedding_model_materialized_in_workflow_process": False,
            "embedding_model_materialized_in_short_lived_worker": (
                mode == "fresh_build"
            ),
            "embedding_cache_worker_execution": worker_execution,
            "cuda_memory_release_by_worker_exit": mode == "fresh_build",
            "terminal_files": terminal_files,
        }

    def _phase_hook_context(self, phase: str, attempt: Path) -> Mapping[str, Any]:
        prepared, preparation_manifest = self._input_preparation_paths()
        import_sources = self._resolved_cache_import_sources()
        cache: Path | None = None
        cache_prepared: Path | None = None
        cache_phase_identity: Mapping[str, Any] | None = None
        preflight_profile: Path | None = None
        cluster_preflight_manifest: Path | None = None
        if phase in {"stage1_preflight", "stage1_modeling"}:
            cache, cache_prepared = self._embedding_cache_paths()
            cache_phase = self._validated_complete("embedding_cache")
            assert cache_phase is not None
            raw_identity = cache_phase["result"].get("cache_identity")
            if not isinstance(raw_identity, Mapping):
                raise RuntimeError("embedding-cache phase did not expose a cache identity")
            cache_phase_identity = dict(raw_identity)
        if phase == "stage1_modeling":
            preflight_profile, cluster_preflight_manifest = self._stage1_preflight_paths()
        return {
            "schema_version": "production_workflow_phase_hook_context_v1",
            "phase": phase,
            "attempt_dir": str(attempt.resolve()),
            "request_sha256": self.request["request_sha256"],
            "prepared_cohort_path": str(prepared),
            "preparation_manifest_path": str(preparation_manifest),
            "embedding_cache_target_dir": str((attempt / "embedding_cache").resolve()),
            "embedding_cache_import": (
                None
                if self.options.embedding_cache_import is None
                else str(self.options.embedding_cache_import.resolve(strict=True))
            ),
            "embedding_cache_import_source_prepared_path": (
                None if import_sources is None else str(import_sources[0])
            ),
            "embedding_cache_import_source_preparation_manifest_path": (
                None if import_sources is None else str(import_sources[1])
            ),
            "embedding_chunk_configuration": dict(self._embedding_chunk_configuration()),
            "embedding_cache_path": None if cache is None else str(cache),
            "cache_bound_prepared_cohort_path": (
                None if cache_prepared is None else str(cache_prepared)
            ),
            "embedding_cache_phase_identity": cache_phase_identity,
            "effective_stage1_profile_path": (
                None if preflight_profile is None else str(preflight_profile)
            ),
            "cluster_preflight_manifest_path": (
                None if cluster_preflight_manifest is None else str(cluster_preflight_manifest)
            ),
            "stage1_profile_path": str(self.options.stage1_profile_path.resolve(strict=True)),
            "query_profile_path": str(self.options.query_profile_path.resolve(strict=True)),
            "stage1_gpu_ids": list(self.stage1_gpu_ids),
            "query_devices": list(self.query_devices),
            "stage1_execution_device_count": (
                self.options.stage1_execution_device_count
            ),
            "stage1_execution_profile": (
                None
                if self.options.stage1_execution_profile is None
                else self.options.stage1_execution_profile.as_dict()
            ),
            "stage1_scope_workers_per_gpu": self.options.stage1_scope_workers_per_gpu,
            "stage1_preflight_workers": self.options.stage1_preflight_workers,
            "stage1_seed_policy": self.options.stage1_seed_policy,
            "stage1_scope_attempt_root": str(
                (self.options.work_root / "recovery" / "stage1_scope_attempts").resolve()
            ),
            "stage1_scope_progress_path": str(
                (self.options.work_root / "recovery" / "stage1_scope_progress.json").resolve()
            ),
            "tfidf_workers": self.options.tfidf_workers,
            "tfidf_parallel_backend": self.options.tfidf_parallel_backend,
            "seed": self.options.seed,
            "resource_preflight": self._gpu_preflight(),
        }

    def _validate_handoff_in_fresh_process(
        self,
        *,
        bundle_manifest: Path,
        report_path: Path,
        portable_role_neutral: bool = False,
    ) -> Mapping[str, Any]:
        script = r"""
import hashlib
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1]).resolve(strict=True)
report = Path(sys.argv[2])
review_rounds = int(sys.argv[3])
initial_training_partitions = int(sys.argv[4])
interaction_folds = int(sys.argv[5])
tfidf_folds = int(sys.argv[6])
portable_role_neutral = json.loads(sys.argv[7])
if portable_role_neutral:
    import oci.inference.production_role_neutral_stage2_handoff as handoff_module
    from oci.inference.production_role_neutral_stage2_handoff import (
        load_reference_only_role_neutral_stage1_handoff,
    )
    handoff = load_reference_only_role_neutral_stage1_handoff(manifest)
else:
    import oci.inference.production_stage1_hierarchy_handoff as handoff_module
    from oci.inference.production_stage1_hierarchy_handoff import (
        load_production_stage1_hierarchy_handoff,
    )
    handoff = load_production_stage1_hierarchy_handoff(
        manifest,
        review_rounds=review_rounds,
        initial_training_partitions=initial_training_partitions,
        interaction_inner_folds=interaction_folds,
        tfidf_nested_calibration_folds=tfidf_folds,
    )
body = {
    "schema_version": "production_stage1_fresh_handoff_validation_v1",
    "status": "accepted",
    "bundle_manifest_path": str(manifest),
    "review_rounds": review_rounds,
    "initial_training_partitions": initial_training_partitions,
    "interaction_inner_folds": interaction_folds,
    "tfidf_nested_calibration_folds": tfidf_folds,
    "handoff": handoff.as_dict(),
    "remote_clients_constructed": False,
    "remote_calls_made": False,
    "loader_module_path": str(Path(handoff_module.__file__).resolve(strict=True)),
}
canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
payload = {**body, "content_sha256": hashlib.sha256(canonical.encode()).hexdigest()}
report.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
"""
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        interpreter = [sys.executable]
        if self.request.get("source_snapshot") is not None:
            interpreter.append("-P")
        subprocess.run(
            [
                *interpreter,
                "-c",
                script,
                str(bundle_manifest.resolve(strict=True)),
                str(report_path.resolve()),
                str(self.options.review_rounds),
                str(self.options.initial_training_partitions),
                str(self.options.interaction_inner_folds),
                str(self.options.tfidf_nested_calibration_folds),
                json.dumps(bool(portable_role_neutral)),
            ],
            check=True,
            env=environment,
        )
        value = json.loads(report_path.read_text(encoding="utf-8"))
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("schema_version") != "production_stage1_fresh_handoff_validation_v1"
            or value.get("status") != "accepted"
            or value.get("bundle_manifest_path") != str(bundle_manifest.resolve(strict=True))
            or int(value.get("initial_training_partitions", -1))
            != int(self.options.initial_training_partitions)
            or value.get("content_sha256") != _sha(body)
            or value.get("remote_clients_constructed") is not False
            or value.get("remote_calls_made") is not False
            or (
                portable_role_neutral
                and (
                    not isinstance(value.get("handoff"), Mapping)
                    or value["handoff"].get("full_stage2_one_shot_runtime_complete") is not False
                    or value["handoff"].get("offline_handoff_validation_complete") is not True
                )
            )
        ):
            raise RuntimeError("fresh Stage 1 handoff validation report is invalid")
        source_snapshot = self.request.get("source_snapshot")
        if source_snapshot is not None:
            loaded = Path(str(value.get("loader_module_path", ""))).resolve(strict=True)
            snapshot_root = Path(str(source_snapshot["root"])).resolve(strict=True)
            try:
                loaded.relative_to(snapshot_root)
            except ValueError as exc:
                raise RuntimeError(
                    "fresh handoff loader did not execute from source snapshot"
                ) from exc
        return value

    def _validate_terminal_in_fresh_process(
        self,
        *,
        report_path: Path,
    ) -> Mapping[str, Any]:
        """Reopen the immutable request and every prior phase from paths only."""

        script = r"""
import hashlib
import json
import sys
from pathlib import Path
import oci.inference.production_all_evidence_workflow as workflow_module
from oci.inference.production_all_evidence_workflow import (
    validate_completed_workflow_prefix,
    validate_published_workflow_checkpoint_dag,
)

root = Path(sys.argv[1]).resolve(strict=True)
request_sha256 = sys.argv[2]
phases = json.loads(sys.argv[3])
stage1_only = json.loads(sys.argv[4])
report = Path(sys.argv[5])
validation = validate_completed_workflow_prefix(
    work_root=root,
    expected_request_sha256=request_sha256,
    expected_phases=phases,
)
checkpoint_validation = validate_published_workflow_checkpoint_dag(
    work_root=root,
    expected_request_sha256=request_sha256,
    expected_phases=phases,
)
body = {
    "schema_version": "production_all_evidence_fresh_terminal_validation_report_v2",
    "execution_completed": True,
    "run_validation_status": "accepted",
    "global_release_certified": False,
    "stage1_only": stage1_only,
    "validated_phase_sequence": [*phases, "terminal_validation"],
    "stage1_handoff_validated_in_fresh_process": validation[
        "stage1_handoff_validated_in_fresh_process"
    ],
    "read_only_prefix_validation": validation,
    "portable_checkpoint_dag_validation": checkpoint_validation,
    "live_runner_objects_received": False,
    "validator_module_path": str(Path(workflow_module.__file__).resolve(strict=True)),
}
canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
payload = {**body, "content_sha256": hashlib.sha256(canonical.encode()).hexdigest()}
report.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
"""
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        prior_phases = list(self._phase_sequence()[:-1])
        interpreter = [sys.executable]
        if self.request.get("source_snapshot") is not None:
            interpreter.append("-P")
        subprocess.run(
            [
                *interpreter,
                "-c",
                script,
                str(self.options.work_root.resolve(strict=True)),
                self.request["request_sha256"],
                json.dumps(prior_phases),
                json.dumps(self.options.stage1_only),
                str(report_path.resolve()),
            ],
            check=True,
            env=environment,
        )
        value = _read_json_object(report_path, label="fresh terminal validation report")
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        expected_keys = {
            "schema_version",
            "execution_completed",
            "run_validation_status",
            "global_release_certified",
            "stage1_only",
            "validated_phase_sequence",
            "stage1_handoff_validated_in_fresh_process",
            "read_only_prefix_validation",
            "portable_checkpoint_dag_validation",
            "live_runner_objects_received",
            "validator_module_path",
            "content_sha256",
        }
        prefix_validation = value.get("read_only_prefix_validation")
        checkpoint_validation = value.get("portable_checkpoint_dag_validation")
        operator_trusted_reuse = any(
            isinstance(record, Mapping)
            and _operator_trusted_adoption_selected(record)
            for record in (
                self.request.get("requested_checkpoint_adoptions")
                or ()
            )
        )
        if (
            set(value) != expected_keys
            or value.get("schema_version")
            != "production_all_evidence_fresh_terminal_validation_report_v2"
            or value.get("execution_completed") is not True
            or value.get("run_validation_status") != "accepted"
            or value.get("global_release_certified") is not False
            or value.get("stage1_only") is not self.options.stage1_only
            or value.get("validated_phase_sequence") != list(self._phase_sequence())
            or value.get("content_sha256") != _sha(body)
            or value.get("live_runner_objects_received") is not False
            or not isinstance(prefix_validation, Mapping)
            or prefix_validation.get("status") != "accepted"
            or prefix_validation.get("request_sha256") != self.request["request_sha256"]
            or not isinstance(checkpoint_validation, Mapping)
            or checkpoint_validation.get("status") != "accepted"
            or checkpoint_validation.get("request_sha256") != self.request["request_sha256"]
            or checkpoint_validation.get(
                "fresh_full_byte_validation"
            )
            is operator_trusted_reuse
            or checkpoint_validation.get(
                "operator_trusted_checkpoint_reuse"
            )
            is not operator_trusted_reuse
            or checkpoint_validation.get(
                "payload_bytes_reauthenticated_for_all_adoptions"
            )
            is operator_trusted_reuse
            or checkpoint_validation.get(
                "global_release_certified"
            )
            is not False
            or checkpoint_validation.get("oracle_evaluation_after_frozen_prediction") is not True
        ):
            raise RuntimeError("fresh terminal validation report is invalid")
        source_snapshot = self.request.get("source_snapshot")
        if source_snapshot is not None:
            loaded = Path(str(value.get("validator_module_path", ""))).resolve(strict=True)
            snapshot_root = Path(str(source_snapshot["root"])).resolve(strict=True)
            try:
                loaded.relative_to(snapshot_root)
            except ValueError as exc:
                raise RuntimeError(
                    "fresh terminal validator did not execute from source snapshot"
                ) from exc
        if (
            "handoff_validation" not in self.phase_overrides
            and value.get("stage1_handoff_validated_in_fresh_process") is not True
        ):
            raise RuntimeError("default Stage 1 handoff was not freshly validated")
        return value

    def _stage1_component_store_root(
        self,
        *,
        prepared_context: Any,
        plan: Any,
        integration: ProductionRoleNeutralStage1Integration,
    ) -> Path:
        """Resolve a resource-neutral, cross-request component namespace."""

        integration_identity = (
            _role_neutral_stage1_integration_identity(integration)
        )
        if integration_identity is None:
            raise RuntimeError(
                "Stage 1 component store requires the closed production "
                "integration identity"
            )
        producer_compatibility = integration_identity.get(
            "producer_factories_builder"
        )
        scientific_identity = getattr(
            prepared_context,
            "scientific_identity",
            None,
        )
        prepared_projection = (
            scientific_identity.get(
                "stage1_request_scientific_projection"
            )
            if isinstance(scientific_identity, Mapping)
            else None
        )
        producer_behavior = (
            producer_compatibility.get("behavior_state")
            if isinstance(producer_compatibility, Mapping)
            else None
        )
        producer_scientific_identity = (
            producer_behavior.get("scientific_identity")
            if isinstance(producer_behavior, Mapping)
            and producer_behavior.get("state_policy")
            == "explicit_closed_scientific_identity_v1"
            else None
        )
        plan_sha256 = getattr(
            plan,
            "scientific_content_sha256",
            None,
        )
        component_input_fields = (
            "dataset",
            "effective_stage1_config",
            "embedding_cache",
            "exact_inner_contract",
            "htr_input_nontruncation_audit",
            "htr_model",
            "query_config",
            "semantic_witness_scientific_config",
            "source_config",
            "split_registry_content_sha256",
            "stage1_scope_plan",
        )
        if (
            not isinstance(producer_compatibility, Mapping)
            or not isinstance(prepared_projection, Mapping)
            or any(
                field not in prepared_projection
                for field in component_input_fields
            )
            or not isinstance(producer_scientific_identity, Mapping)
            or not isinstance(plan_sha256, str)
            or len(plan_sha256) != 64
        ):
            raise RuntimeError(
                "Stage 1 component store compatibility identity is "
                "incomplete"
            )
        component_input_projection = {
            field: copy.deepcopy(prepared_projection[field])
            for field in component_input_fields
        }
        compatibility = {
            "schema_version": (
                "production_stage1_component_store_compatibility_v2"
            ),
            "prepared_stage1_component_input_projection": (
                component_input_projection
            ),
            "prepared_stage1_component_input_projection_sha256": (
                _sha(component_input_projection)
            ),
            "stage1_scope_plan_scientific_content_sha256": (
                plan_sha256
            ),
            "component_plan_namespace_identity": (
                ROLE_NEUTRAL_STAGE1_COMPONENT_PLAN_NAMESPACE_IDENTITY
            ),
            "component_producer_scientific_identity": copy.deepcopy(
                dict(producer_scientific_identity)
            ),
            "evidence_family_order": list(EVIDENCE_FAMILIES),
            "component_authentication_is_final_reuse_authority": True,
            "stage2_handoff_publisher_identity_included": False,
            "stage2_catalog_identity_included": False,
            "repository_source_closure_included": False,
            "resource_assignment_included": False,
            "cpu_budget_included": False,
            "owner_concurrency_included": False,
        }
        component_store_key = _sha(compatibility)
        configured_scratch = self.options.scratch_root
        if configured_scratch is None:
            configured_scratch = (
                self.options.work_root.parent
                / f".{self.options.work_root.name}.scratch"
            )
        root = (
            Path(configured_scratch)
            / "production_all_evidence_workflow"
            / "stage1_component_store"
            / component_store_key
        ).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if (
            root.is_symlink()
            or root.resolve(strict=True) != root
            or not root.is_dir()
        ):
            raise ValueError(
                "Stage 1 component store namespace is not canonical"
            )
        manifest_body = {
            "schema_version": STAGE1_COMPONENT_STORE_SCHEMA,
            "component_store_key": component_store_key,
            "compatibility": compatibility,
            "components_relative_path": "components",
            "successful_component_marker": (
                "execution_manifest.json"
            ),
            "incomplete_attempts_preserved_for_recovery": True,
        }
        manifest = {
            **manifest_body,
            "content_sha256": _sha(manifest_body),
        }
        manifest_path = root / STAGE1_COMPONENT_STORE_MANIFEST
        if manifest_path.is_file() and not manifest_path.is_symlink():
            if _read_json_object(
                manifest_path,
                label="Stage 1 component store manifest",
            ) != manifest:
                raise ValueError(
                    "Stage 1 component store manifest changed"
                )
        elif manifest_path.exists() or manifest_path.is_symlink():
            raise ValueError(
                "Stage 1 component store manifest is not a regular file"
            )
        else:
            _atomic_write_json(manifest_path, manifest)
        components = (root / "components").resolve()
        components.mkdir(exist_ok=True)
        if components.is_symlink() or components.resolve(strict=True) != components:
            raise ValueError(
                "Stage 1 component store payload root is not canonical"
            )
        return components

    def _stage1_component_reuse_roots(
        self,
        *,
        component_store_root: Path,
        plan: Any,
    ) -> tuple[Path, ...]:
        """Discover prior stores; per-component producers remain authoritative."""

        current = Path(component_store_root).resolve(strict=True)
        namespace = current.parent.parent
        if (
            current.is_symlink()
            or not current.is_dir()
            or current.name != "components"
            or namespace.is_symlink()
            or namespace.resolve(strict=True) != namespace
        ):
            raise ValueError(
                "Stage 1 component reuse namespace is not canonical"
            )
        current_manifest = _read_json_object(
            current.parent / STAGE1_COMPONENT_STORE_MANIFEST,
            label="current Stage 1 component store manifest",
        )
        current_compatibility = current_manifest.get("compatibility")
        current_input_sha256 = (
            current_compatibility.get(
                "prepared_stage1_component_input_projection_sha256"
            )
            if isinstance(current_compatibility, Mapping)
            else None
        )
        plan_sha256 = getattr(
            plan,
            "scientific_content_sha256",
            None,
        )
        if (
            not isinstance(current_input_sha256, str)
            or len(current_input_sha256) != 64
            or not isinstance(plan_sha256, str)
            or len(plan_sha256) != 64
        ):
            raise ValueError(
                "current Stage 1 component store compatibility is incomplete"
            )

        required_manifest_fields = {
            "schema_version",
            "component_store_key",
            "compatibility",
            "components_relative_path",
            "successful_component_marker",
            "incomplete_attempts_preserved_for_recovery",
            "content_sha256",
        }
        accepted_schemas = {
            STAGE1_COMPONENT_STORE_SCHEMA,
            *LEGACY_STAGE1_COMPONENT_STORE_SCHEMAS,
        }
        roots: list[Path] = []
        for candidate_store in sorted(
            namespace.iterdir(),
            key=lambda path: path.name,
        ):
            if candidate_store == current.parent:
                continue
            manifest_path = (
                candidate_store / STAGE1_COMPONENT_STORE_MANIFEST
            )
            if not manifest_path.is_file() or manifest_path.is_symlink():
                continue
            if (
                candidate_store.is_symlink()
                or not candidate_store.is_dir()
                or candidate_store.resolve(strict=True)
                != candidate_store
            ):
                raise ValueError(
                    "prior Stage 1 component store is not canonical"
                )
            manifest = _read_json_object(
                manifest_path,
                label="prior Stage 1 component store manifest",
            )
            body = {
                key: copy.deepcopy(value)
                for key, value in manifest.items()
                if key != "content_sha256"
            }
            compatibility = manifest.get("compatibility")
            if (
                set(manifest) != required_manifest_fields
                or manifest.get("schema_version")
                not in accepted_schemas
                or manifest.get("component_store_key")
                != candidate_store.name
                or manifest.get("components_relative_path")
                != "components"
                or manifest.get("successful_component_marker")
                != "execution_manifest.json"
                or manifest.get(
                    "incomplete_attempts_preserved_for_recovery"
                )
                is not True
                or manifest.get("content_sha256") != _sha(body)
                or not isinstance(compatibility, Mapping)
            ):
                raise ValueError(
                    "prior Stage 1 component store manifest is invalid"
                )
            if (
                compatibility.get(
                    "stage1_scope_plan_scientific_content_sha256"
                )
                != plan_sha256
            ):
                continue
            if (
                manifest.get("schema_version")
                == STAGE1_COMPONENT_STORE_SCHEMA
                and compatibility.get(
                    "prepared_stage1_component_input_projection_sha256"
                )
                != current_input_sha256
            ):
                continue
            candidate_components_path = candidate_store / "components"
            if (
                candidate_components_path.is_symlink()
                or not candidate_components_path.is_dir()
            ):
                raise ValueError(
                    "prior Stage 1 component payload root is not canonical"
                )
            candidate_components = (
                candidate_components_path.resolve(strict=True)
            )
            if candidate_components.parent != candidate_store:
                raise ValueError(
                    "prior Stage 1 component payload root escaped its store"
                )
            roots.append(candidate_components)
        return tuple(roots)

    def _run_portable_role_neutral_stage1_modeling(
        self,
        attempt: Path,
    ) -> Mapping[str, Any]:
        """Run the deduplicated all-ten path without entering legacy build()."""

        integration = self.hooks.role_neutral_stage1
        if integration is None:
            raise RuntimeError(
                "typed portable Stage 1 requires the explicit role-neutral "
                "producer/executor/handoff integration; the legacy 40-attempt "
                "bundle build is forbidden"
            )
        from .portable_resource_scheduler import plan_resources
        from .direct_upstream_numerical_reference_bank import (
            DIRECT_NUMERICAL_REFERENCE_LOCATOR,
            DIRECT_NUMERICAL_REFERENCE_MANIFEST,
            publish_role_neutral_direct_numerical_reference_bank,
        )
        from .production_role_neutral_stage2_handoff import (
            ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
            load_reference_only_role_neutral_stage1_handoff,
        )
        from .production_stage1_role_neutral_execution import (
            ROLE_NEUTRAL_EXECUTION_MANIFEST,
            RoleNeutralProducerFactories,
            RoleNeutralStage1ExecutionPolicy,
            execute_and_publish_role_neutral_stage1,
        )

        cache, prepared_cohort = self._embedding_cache_paths()
        profile, cluster_preflight_manifest = self._stage1_preflight_paths()
        cluster_preflight_state_bundle_manifest = self._stage1_preflight_state_bundle_path()
        prepared_context_manifest = self._stage1_prepared_context_path()
        from .prepared_stage1_context import (
            load_prepared_stage1_context,
        )

        prepared_context = load_prepared_stage1_context(
            prepared_context_manifest
        )
        prepared, producer_factories = prepared_context.reconstruct(
            slot_cpu_budget=int(self.options.cpu_budget),
        )
        if (
            prepared.options.dataset_path.resolve(strict=True)
            != prepared_cohort
            or prepared.options.config_path.resolve(strict=True) != profile
            or prepared.embedding_cache_path.resolve(strict=True) != cache
            or prepared.cluster_preflight_manifest_path
            != cluster_preflight_manifest
            or Path(
                prepared.options
                .cluster_preflight_state_bundle_manifest_path
            ).resolve(strict=True)
            != cluster_preflight_state_bundle_manifest
        ):
            raise RuntimeError(
                "reusable prepared Stage 1 context changed its phase locators"
            )
        plan = prepared.stage1_scope_plan
        state_bundle = prepared.cluster_preflight_state_bundle
        if state_bundle is None or set(state_bundle.states) != {
            scope.scope_id for scope in plan.physical_scopes
        }:
            raise RuntimeError(
                "typed portable Stage 1 requires one authenticated "
                "no-refit clustered state per physical owner"
            )
        registry_outer_folds = prepared.registry.get("outer_folds")
        if (
            not isinstance(registry_outer_folds, list)
            or not registry_outer_folds
            or plan.review_rounds != int(self.options.review_rounds)
            or plan.initial_training_partitions != int(self.options.initial_training_partitions)
            or not plan.physical_scopes
            or len(plan.physical_scopes) > len(plan.scopes)
        ):
            raise RuntimeError(
                "prepared Stage 1 scope plan does not cover the configured " "logical contexts"
            )
        expected_scope_partition: dict[
            tuple[int, str],
            int,
        ] = {}
        for outer in registry_outer_folds:
            if not isinstance(outer, Mapping) or not isinstance(
                outer.get("inner_folds"),
                list,
            ):
                raise RuntimeError("prepared split registry has an invalid fold partition")
            outer_fold = int(outer["outer_fold"])
            expected_scope_partition[(outer_fold, "full_outer")] = 1
            expected_scope_partition[(outer_fold, "exact_inner")] = len(outer["inner_folds"])
            expected_scope_partition[(outer_fold, "cumulative_spent")] = int(plan.review_rounds)
        observed_scope_partition: dict[
            tuple[int, str],
            int,
        ] = {}
        for scope in plan.scopes:
            key = (int(scope.outer_fold), str(scope.scope_kind))
            observed_scope_partition[key] = observed_scope_partition.get(key, 0) + 1
        if observed_scope_partition != expected_scope_partition:
            raise RuntimeError(
                "prepared Stage 1 scope plan differs from the validated " "registry fold partition"
            )

        configured_resource_policy: str | Sequence[str]
        if self.options.device_policy:
            configured_resource_policy = self.options.device_policy
        else:
            configured_resource_policy = tuple(
                dict.fromkeys(
                    (
                        str(self.options.stage1_device),
                        *self.query_devices,
                    )
                )
            )
        resource_plan = plan_resources(
            policy=configured_resource_policy,
            cpu_budget=int(self.options.cpu_budget),
            requested_device_count=int(
                self.options.stage1_execution_device_count
            ),
            cpu_supported=True,
            resource_performance_safety=(self.options.resource_performance_safety),
        )
        stage1_execution_profile = self.options.stage1_execution_profile
        if not isinstance(
            stage1_execution_profile,
            Stage1ExecutionProfile,
        ):
            raise RuntimeError(
                "typed portable Stage 1 requires its complete execution profile"
            )
        if (
            len(resource_plan.devices)
            != stage1_execution_profile.device_count
            or int(self.options.stage1_scope_workers_per_gpu)
            != stage1_execution_profile.scope_workers_per_device
            or stage1_execution_profile.max_parallel_owners
            > int(resource_plan.cpu_budget)
        ):
            raise RuntimeError(
                "resolved Stage 1 resources differ from the authenticated "
                "execution profile"
            )
        neural_query_topologies = (
            stage1_execution_profile.neural_query_topology
            .runtime_topologies(resource_plan.devices)
        )
        execution_policy = RoleNeutralStage1ExecutionPolicy(
            resource_plan=resource_plan,
            max_parallel_owners=(
                stage1_execution_profile.max_parallel_owners
            ),
            neural_query_execution_topologies=(
                neural_query_topologies
            ),
            htr_operational_controls=(
                stage1_execution_profile.htr_operational_controls
            ),
            neural_query_operational_controls=(
                stage1_execution_profile
                .neural_query_operational_controls
            ),
            # Every owner is authenticated at the ordinary component and
            # handoff boundaries.  Production owner parallelism must not be
            # preceded by a serial calibration/replica gate.
            first_owner_validation=None,
        )
        execution_executor = integration.executor
        bind_context = getattr(execution_executor, "bind_context", None)
        if not callable(bind_context):
            raise RuntimeError(
                "typed portable Stage 1 executor cannot consume the sealed "
                "prepared context"
            )
        execution_executor = bind_context(prepared_context_manifest)
        if not isinstance(producer_factories, RoleNeutralProducerFactories):
            raise TypeError(
                "role-neutral producer-factories builder returned an "
                "untyped or incomplete six-producer binding"
            )
        component_store_root = self._stage1_component_store_root(
            prepared_context=prepared_context,
            plan=plan,
            integration=integration,
        )
        component_reuse_roots = self._stage1_component_reuse_roots(
            component_store_root=component_store_root,
            plan=plan,
        )

        execution_root = (attempt / "role_neutral_stage1_execution").resolve()
        execution_manifest = execute_and_publish_role_neutral_stage1(
            root=execution_root,
            plan=plan,
            producer_factories=producer_factories,
            policy=execution_policy,
            executor=execution_executor,
            resume=self.options.run_control.resume,
            component_store_root=component_store_root,
            component_reuse_roots=component_reuse_roots,
        )
        if (
            int(execution_manifest.get("physical_fit_count", -1)) != len(plan.physical_scopes)
            or int(execution_manifest.get("logical_scope_count", -1)) != len(plan.scopes)
            or execution_manifest.get("legacy_bundle_build_invoked") is not False
            or execution_manifest.get("every_physical_owner_executed_once") is not True
            or execution_manifest.get("every_component_executed_and_authenticated_once_per_owner")
            is not True
            or execution_manifest.get("productive_compute_canary_completed") is not False
            or execution_manifest.get("selected_canary_replica_adopted_as_production") is not False
            or execution_manifest.get("compute_canary_scientific_equality") is not None
        ):
            raise RuntimeError(
                "role-neutral Stage 1 execution returned an incomplete "
                "physical/logical coverage claim"
            )

        numerical_bank_root = (attempt / "direct_upstream_numerical_reference_bank").resolve()
        numerical_bank = publish_role_neutral_direct_numerical_reference_bank(
            root=numerical_bank_root,
            execution_root=execution_root,
            plan=plan,
            execution_manifest=execution_manifest,
        )
        numerical_bank_identity = numerical_bank.identity()
        numerical_bank_manifest_path = numerical_bank_root / DIRECT_NUMERICAL_REFERENCE_MANIFEST
        numerical_bank_locator_path = numerical_bank_root / DIRECT_NUMERICAL_REFERENCE_LOCATOR
        if (
            numerical_bank.manifest_path != numerical_bank_manifest_path
            or numerical_bank_identity.get("plan_scientific_content_sha256")
            != plan.scientific_content_sha256
            or numerical_bank_identity.get("source_execution_content_sha256")
            != execution_manifest["content_sha256"]
        ):
            raise RuntimeError("direct numerical reference bank changed its Stage 1 binding")

        bundle_root = (attempt / "stage1_bundle").resolve()
        publication = integration.handoff_publisher(
            target_dir=bundle_root,
            prepared=prepared,
            role_neutral_execution_root=execution_root,
            role_neutral_execution_manifest=execution_manifest,
        )
        if not isinstance(publication, RoleNeutralStage1HandoffPublication):
            raise TypeError(
                "role-neutral Stage 2 handoff publisher returned an untyped " "publication"
            )
        execution_content_sha256 = str(execution_manifest["content_sha256"])
        if publication.source_role_neutral_execution_content_sha256 != execution_content_sha256:
            raise ValueError(
                "Stage 2 handoff publisher bound a different role-neutral " "execution"
            )
        bundle_manifest_path = Path(publication.bundle_manifest_path)
        if (
            not bundle_manifest_path.is_absolute()
            or bundle_manifest_path.is_symlink()
            or bundle_manifest_path.resolve(strict=True) != bundle_root / "bundle_manifest.json"
        ):
            raise ValueError(
                "role-neutral Stage 2 handoff publisher did not publish the "
                "requested canonical bundle manifest"
            )
        bundle_manifest = _read_json_object(
            bundle_manifest_path,
            label="role-neutral Stage 2 bundle manifest",
        )
        bundle_sha256 = str(bundle_manifest.get("bundle_sha256") or "")
        if (
            len(bundle_sha256) != 64
            or any(character not in "0123456789abcdef" for character in bundle_sha256)
            or bundle_manifest.get("request_sha256") != prepared.request_sha256
        ):
            raise ValueError(
                "role-neutral Stage 2 adapter published an incompatible " "bundle manifest"
            )
        loaded_publication = load_reference_only_role_neutral_stage1_handoff(
            bundle_manifest_path,
        )
        if (
            loaded_publication.handoff_kind != ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
            or loaded_publication.stage2_provider is None
        ):
            raise ValueError(
                "portable Stage 1 handoff did not reopen its authenticated "
                "reference-only provider"
            )

        execution_manifest_path = execution_root / ROLE_NEUTRAL_EXECUTION_MANIFEST
        execution_manifest_bytes_sha256, execution_manifest_size = stable_file_sha256(
            execution_manifest_path
        )
        bundle_manifest_bytes_sha256, bundle_manifest_size = stable_file_sha256(
            bundle_manifest_path
        )
        integration_identity = _role_neutral_stage1_integration_identity(integration)
        assert integration_identity is not None
        binding_body = {
            "schema_version": (PORTABLE_ROLE_NEUTRAL_STAGE1_HANDOFF_BINDING_SCHEMA),
            "workflow_request_sha256": self.request["request_sha256"],
            "prepared_stage1_request_sha256": prepared.request_sha256,
            "stage1_scope_plan_scientific_content_sha256": (plan.scientific_content_sha256),
            "role_neutral_execution_manifest": {
                "relative_path": (execution_manifest_path.relative_to(attempt).as_posix()),
                "sha256": execution_manifest_bytes_sha256,
                "size_bytes": execution_manifest_size,
                "content_sha256": execution_content_sha256,
            },
            "stage2_bundle_manifest": {
                "relative_path": (bundle_manifest_path.relative_to(attempt).as_posix()),
                "sha256": bundle_manifest_bytes_sha256,
                "size_bytes": bundle_manifest_size,
                "bundle_sha256": bundle_sha256,
            },
            "direct_numerical_reference_bank": {
                "relative_path": (numerical_bank_manifest_path.relative_to(attempt).as_posix()),
                "content_sha256": numerical_bank_identity["manifest_content_sha256"],
                "source_execution_content_sha256": numerical_bank_identity[
                    "source_execution_content_sha256"
                ],
                "combined_npy_payloads_persisted": False,
            },
            "integration_code_identity": integration_identity,
            "physical_fit_count": len(plan.physical_scopes),
            "logical_scope_count": len(plan.scopes),
            "deduplicated_fit_count": (len(plan.scopes) - len(plan.physical_scopes)),
            "productive_compute_canary_completed": False,
            "selected_canary_replica_adopted_as_production": False,
            "compute_canary_scientific_equality": None,
            "legacy_bundle_build_invoked": False,
            "all_ten_role_neutral_execution_is_exclusive_evidence_source": True,
            "stage2_loader_validation": ("reference_only_role_neutral_provider_accepted"),
        }
        binding = {
            **binding_body,
            "content_sha256": _sha(binding_body),
        }
        binding_path = attempt / "role_neutral_handoff_binding.json"
        _atomic_write_json(binding_path, binding)

        stage1_terminal_paths = list(
            _portable_stage1_terminal_file_inventory(
                execution_root=execution_root,
                bundle_root=bundle_root,
                numerical_bank_root=numerical_bank_root,
                binding_path=binding_path,
            )
        )
        result = {
            "schema_version": PORTABLE_ROLE_NEUTRAL_STAGE1_PHASE_SCHEMA,
            "execution_mode": "deduplicated_role_neutral_all_ten_v1",
            "prepared_stage1_request_sha256": prepared.request_sha256,
            "stage1_scope_plan_scientific_content_sha256": (plan.scientific_content_sha256),
            "role_neutral_execution_root": str(execution_root),
            "role_neutral_execution_manifest_path": str(execution_manifest_path),
            "role_neutral_execution_content_sha256": (execution_content_sha256),
            "role_neutral_handoff_binding_path": str(binding_path.resolve()),
            "bundle_manifest_path": str(bundle_manifest_path),
            "bundle_sha256": bundle_sha256,
            "direct_numerical_bank_manifest_path": str(numerical_bank_manifest_path),
            "direct_numerical_bank_locator_path": str(numerical_bank_locator_path),
            "direct_numerical_bank_content_sha256": numerical_bank_identity[
                "manifest_content_sha256"
            ],
            "physical_fit_count": len(plan.physical_scopes),
            "logical_scope_count": len(plan.scopes),
            "deduplicated_fit_count": (len(plan.scopes) - len(plan.physical_scopes)),
            "every_physical_owner_executed_once": True,
            "productive_compute_canary_completed": False,
            "selected_canary_replica_adopted_as_production": False,
            "compute_canary_scientific_equality": None,
            "all_ten_families_bound_per_logical_context": True,
            "legacy_bundle_build_invoked": False,
            "stage2_handoff_derived_exclusively_from_role_neutral_execution": (True),
            "resource_preflight": resource_plan.execution_attestation(),
            "terminal_files": stage1_terminal_paths,
        }
        return _validate_portable_role_neutral_stage1_phase_result(result)

    def _compile_current_stage1_request_from_reused_preflight(
        self,
        *,
        accepted: Any,
        current_options: Stage1BundleBuildOptions,
        current_plan: Any,
    ) -> Mapping[str, Any]:
        """Compile today's exact Stage 1 request without cohort/cache reads."""

        from .production_stage1_bundle import (
            STAGE1_BUNDLE_REQUEST_SCHEMA,
            STAGE1_TFIDF_RESUME_POLICY,
            ProductionStage1BundleBuilder,
            _exact_inner_contract_registry_status,
            _hierarchy_spent_evidence_contract,
            _read_stable_sha256,
            _sanitize_secrets,
            _scientific_query_config_identity,
            _source_identity,
            _validate_effective_config,
            exact_inner_family_adapter_gate,
            load_applied_stage1_config,
        )
        from .production_stage1_config_wire import (
            production_stage1_effective_config_payload,
        )
        from .production_stage1_hierarchy_contract import (
            current_production_stage1_hierarchy_contract_identity,
            production_stage1_hierarchy_architecture_bindings,
            validate_production_stage1_hierarchy_request_bindings,
        )

        old_request = copy.deepcopy(
            dict(
                accepted.prepared_context.execution_locators[
                    "exact_stage1_request"
                ]
            )
        )
        registry = accepted.prepared_context.scientific_identity[
            "split_registry"
        ]
        registry_sha = str(
            accepted.prepared_context.scientific_identity[
                "split_registry_content_sha256"
            ]
        )
        profile_path = Path(current_options.config_path).resolve(
            strict=True
        )
        dataset_path = Path(current_options.dataset_path).resolve(
            strict=True
        )
        cache_path = Path(
            current_options.embedding_cache_dir
        ).resolve(strict=True)
        config_sha, _config_stat = _read_stable_sha256(
            profile_path
        )
        source_config = load_applied_stage1_config(
            profile_path,
            require_explicit_scientific_fields=True,
        )
        config, htr_model_path = _validate_effective_config(
            source_config,
            dataset_path=dataset_path,
            embedding_cache_dir=cache_path,
            config_dir=profile_path.parent,
            seed=current_options.seed,
        )
        if (
            htr_model_path.resolve(strict=True)
            != self.options.htr_local_model_path.resolve(strict=True)
        ):
            raise ValueError(
                "current effective profile selected another HTR model"
            )
        effective_config = _sanitize_secrets(
            production_stage1_effective_config_payload(config)
        )
        query_config, query_identity = (
            ProductionStage1BundleBuilder._load_query_config(
                current_options.query_config_path
            )
        )
        query_request_identity = _scientific_query_config_identity(
            query_identity
        )
        semantic = current_options.semantic_witness_scientific_config
        if isinstance(semantic, Mapping):
            from .review_spent_evidence_provider import (
                SemanticWitnessScientificConfig,
            )

            semantic = SemanticWitnessScientificConfig.from_mapping(
                semantic
            )
        if semantic is None:
            raise ValueError(
                "current Stage 1 request lacks semantic-witness science"
            )
        semantic_mapping = semantic.as_dict()
        hierarchy_identity = (
            current_production_stage1_hierarchy_contract_identity()
        )
        architecture_contract = (
            production_stage1_hierarchy_architecture_bindings(
                hierarchy_identity
            )
        )
        if (
            architecture_contract.get("tfidf_resume_policy")
            != STAGE1_TFIDF_RESUME_POLICY
        ):
            raise RuntimeError(
                "current hierarchy changed TF-IDF resume policy"
            )
        exact_inner = _exact_inner_contract_registry_status(
            registry
        )
        hierarchy_spent = _hierarchy_spent_evidence_contract(
            registry=registry,
            config=config,
            initial_training_partitions=(
                current_options.initial_training_partitions
            ),
            hierarchical_discovery_contract_identity_sha256=(
                hierarchy_identity["content_sha256"]
            ),
        )

        cache_phase = self._validated_complete("embedding_cache")
        cache_phase_result = (
            cache_phase.get("result")
            if isinstance(cache_phase, Mapping)
            else None
        )
        raw_cache_identity = (
            cache_phase_result.get("cache_identity")
            if isinstance(cache_phase_result, Mapping)
            else None
        )
        cache_phase_mode = (
            cache_phase_result.get("mode")
            if isinstance(cache_phase_result, Mapping)
            else None
        )
        if not isinstance(raw_cache_identity, Mapping):
            raise RuntimeError(
                "current embedding-cache phase lacks its identity"
            )
        old_cache = old_request.get("embedding_cache")
        if not isinstance(old_cache, Mapping):
            raise RuntimeError(
                "accepted Stage 1 request lacks cache provenance"
            )
        trusted_cache_route = (
            current_options.embedding_cache_operator_trusted_read_proof
            is not None
        )
        if trusted_cache_route:
            # Match the ordinary builder: the prior byte-authentication proof
            # is an access capability, not a relocation scientific record.
            cache_build_identity = copy.deepcopy(
                dict(
                    old_cache[
                        "production_cache_build_identity"
                    ]
                )
            )
            provider_identity = copy.deepcopy(
                dict(old_cache["identity"])
            )
            authenticated_relocation = None
        elif cache_phase_mode == "authenticated_relocation":
            from .production_embedding_cache_relocation import (
                PRODUCTION_EMBEDDING_CACHE_RELOCATION_RESULT_SCHEMA,
            )

            relocation_fields = {
                "schema_version",
                "relocator_version",
                "relocator_code_sha256",
                "authenticated_tree_code_sha256",
                "root",
                "cache_dir",
                "prepared_cohort_path",
                "attestation_path",
                "terminal_manifest_path",
                "row_count",
                "prepared_projection_sha256",
                "source_cache_identity_sha256",
                "cache_build_identity",
                "attestation_sha256",
                "terminal_manifest_sha256",
            }
            relocation_hash_fields = {
                "relocator_code_sha256",
                "authenticated_tree_code_sha256",
                "prepared_projection_sha256",
                "source_cache_identity_sha256",
                "attestation_sha256",
                "terminal_manifest_sha256",
            }
            cache_build = raw_cache_identity.get(
                "cache_build_identity"
            )
            if (
                set(raw_cache_identity) != relocation_fields
                or raw_cache_identity.get("schema_version")
                != PRODUCTION_EMBEDDING_CACHE_RELOCATION_RESULT_SCHEMA
                or not isinstance(cache_build, Mapping)
                or any(
                    not isinstance(raw_cache_identity.get(name), str)
                    or len(str(raw_cache_identity[name])) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in str(
                            raw_cache_identity[name]
                        )
                    )
                    for name in relocation_hash_fields
                )
                or Path(
                    str(raw_cache_identity.get("cache_dir", ""))
                ).resolve(strict=True)
                != cache_path
                or Path(
                    str(
                        raw_cache_identity.get(
                            "prepared_cohort_path",
                            "",
                        )
                    )
                ).resolve(strict=True)
                != dataset_path
            ):
                raise RuntimeError(
                    "completed embedding-cache phase has an invalid "
                    "relocation identity"
                )
            cache_build_identity = copy.deepcopy(
                dict(cache_build)
            )
            raw_provider = cache_build_identity.get(
                "provider_identity"
            )
            if not isinstance(raw_provider, Mapping):
                raise RuntimeError(
                    "relocated cache identity lacks its provider science"
                )
            provider_identity = copy.deepcopy(dict(raw_provider))
            authenticated_relocation = copy.deepcopy(
                dict(raw_cache_identity)
            )
        elif cache_phase_mode == "fresh_build":
            cache_build_identity = copy.deepcopy(
                dict(raw_cache_identity)
            )
            provider_identity = cache_build_identity.get(
                "provider_identity"
            )
            if not isinstance(provider_identity, Mapping):
                raise RuntimeError(
                    "current non-relocated embedding cache lacks its "
                    "provider identity"
                )
            provider_identity = copy.deepcopy(
                dict(provider_identity)
            )
            authenticated_relocation = None
        else:
            raise RuntimeError(
                "completed embedding-cache phase has an unsupported mode"
            )
        # The immutable request has already authenticated this complete tree.
        # Project its closed inventory into the bundle schema instead of
        # rereading every HTR model byte during the fast reopen.
        workflow_htr_tree = self.request.get("htr_model_tree")
        if (
            not isinstance(workflow_htr_tree, Mapping)
            or Path(str(workflow_htr_tree.get("path", ""))).resolve(
                strict=True
            )
            != htr_model_path.resolve(strict=True)
        ):
            raise ValueError(
                "immutable workflow request selected another HTR model tree"
            )
        htr_tree_sha = (
            _stage1_bundle_model_tree_sha256_from_workflow_identity(
                workflow_htr_tree
            )
        )
        accepted_audit = copy.deepcopy(
            dict(old_request["htr_input_nontruncation_audit"])
        )
        if (
            htr_tree_sha
            != accepted_audit.get("htr_model_tree_sha256")
        ):
            raise ValueError(
                "reused global audit belongs to another HTR tree"
            )
        dataset = copy.deepcopy(dict(old_request["dataset"]))
        dataset["path"] = str(dataset_path)
        source_profile = {
            "path": str(profile_path),
            "sha256": config_sha,
        }
        embedding_cache = {
            "path": str(cache_path),
            "identity": copy.deepcopy(dict(provider_identity)),
            "production_cache_build_identity": (
                cache_build_identity
            ),
            "authenticated_relocation": (
                authenticated_relocation
            ),
            "legacy_terminal_migration_identity": copy.deepcopy(
                current_options.embedding_cache_legacy_migration_identity
            ),
        }
        runtime = {
            "device": current_options.device,
            "gpu_ids": list(current_options.gpu_ids),
            "num_workers": current_options.num_workers,
            "tfidf_workers": current_options.tfidf_workers,
            "tfidf_parallel_backend": (
                current_options.tfidf_parallel_backend
            ),
            "query_devices": list(current_options.query_devices),
            "query_nuisance_folds": (
                current_options.query_nuisance_folds
            ),
            "scope_workers_per_gpu": (
                current_options.scope_workers_per_gpu
            ),
            "preflight_workers": current_options.preflight_workers,
            "scope_descriptor_root": str(
                Path(
                    current_options.stage1_scope_descriptor_root
                    or (
                        Path(current_options.output_dir)
                        / "stage1_scope_recovery"
                        / "descriptor"
                    )
                ).resolve()
            ),
            "scope_attempt_root": str(
                Path(
                    current_options.stage1_scope_attempt_root
                    or (
                        Path(current_options.output_dir)
                        / "stage1_scope_recovery"
                        / "attempts"
                    )
                ).resolve()
            ),
            "scope_progress_path": str(
                Path(
                    current_options.stage1_scope_progress_path
                    or (
                        Path(current_options.output_dir)
                        / "stage1_scope_recovery"
                        / "progress.json"
                    )
                ).resolve()
            ),
        }
        request_body = {
            "schema_version": STAGE1_BUNDLE_REQUEST_SCHEMA,
            "dataset": dataset,
            "source_config": source_profile,
            "effective_stage1_config": effective_config,
            "embedding_cache": embedding_cache,
            "htr_model": {
                "path": str(htr_model_path),
                "tree_sha256": str(htr_tree_sha),
                "sentence_encoder_unfrozen": True,
            },
            "htr_input_nontruncation_audit": accepted_audit,
            "embedding_cluster_feasibility_audit": copy.deepcopy(
                dict(accepted.preflight.reference)
            ),
            "split_registry_content_sha256": registry_sha,
            "stage1_scope_plan": current_plan.as_dict(),
            "exact_inner_contract": {
                **exact_inner,
                "family_adapter_gate": (
                    exact_inner_family_adapter_gate()
                ),
            },
            "query_config": {
                "effective": asdict(query_config),
                "source": query_request_identity,
            },
            "semantic_witness_scientific_config": (
                semantic_mapping
            ),
            "runtime": runtime,
            "behavior_identity": _source_identity(),
            "hierarchical_discovery_contract_identity": (
                hierarchy_identity
            ),
            "architecture_contract": architecture_contract,
            "hierarchy_spent_evidence_contract": hierarchy_spent,
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
        validate_production_stage1_hierarchy_request_bindings(
            request_body
        )
        request = {
            **request_body,
            "request_sha256": _sha(request_body),
        }
        accepted.preflight.require_stage1_request(request)
        return request

    def _try_fast_reopen_stage1_preflight(
        self,
        attempt: Path,
    ) -> Mapping[str, Any] | None:
        """Fail closed to ordinary preparation while preserving attempts."""

        try:
            return self._fast_reopen_stage1_preflight_or_raise(
                attempt
            )
        except (
            FileNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            candidates = tuple(
                attempt / name
                for name in (
                    "effective_stage1_profile.json",
                    "cluster_preflight",
                    "cluster_preflight_states",
                    "prepared_stage1_context",
                    "stage1_preflight_report.json",
                )
            )
            present = tuple(
                path
                for path in candidates
                if path.exists() or path.is_symlink()
            )
            if present:
                recovery = (
                    attempt
                    / "fast_reopen_recovery"
                    / f"attempt-{time.time_ns()}"
                )
                recovery.mkdir(parents=True, exist_ok=False)
                for path in present:
                    os.rename(path, recovery / path.name)
            return None

    def _fast_reopen_stage1_preflight_or_raise(
        self,
        attempt: Path,
    ) -> Mapping[str, Any] | None:
        """Reopen sealed precomputation before constructing the bulk builder."""

        if self.request.get("portable_typed_workflow") is not True:
            return None
        from .production_stage1_bundle import (
            STAGE1_REUSABLE_CLUSTER_OWNER_PRODUCER_IDENTITY,
            STAGE1_REUSABLE_GLOBAL_AUDIT_PRODUCER_IDENTITY,
        )
        from .production_stage1_reusable_preflight import (
            publish_reusable_preflight_references,
            try_load_reusable_preflight_acceptance,
        )

        selector = self._reusable_preflight_accepted_input_selector()
        started = time.perf_counter()
        try:
            accepted = try_load_reusable_preflight_acceptance(
                store_root=self._reusable_preflight_store_root(),
                selector=selector,
                producer_identity=(
                    STAGE1_REUSABLE_ASSEMBLED_PREFLIGHT_PRODUCER_IDENTITY
                ),
                owner_producer_identity=(
                    STAGE1_REUSABLE_CLUSTER_OWNER_PRODUCER_IDENTITY
                ),
                global_audit_producer_identity=(
                    STAGE1_REUSABLE_GLOBAL_AUDIT_PRODUCER_IDENTITY
                ),
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            # A changed or incomplete candidate can never use the shortcut.
            # The ordinary builder will deeply authenticate compatible pieces
            # and recompute only what is missing or invalid.
            return None
        if accepted is None:
            return None

        cache, prepared = self._embedding_cache_paths()
        profile = self._effective_stage1_profile(
            attempt,
            dataset_path=prepared,
            embedding_cache_dir=cache,
        )
        from .production_stage1_scope_scheduler import (
            Stage1PhysicalFitIdentity,
            build_canonical_stage1_scope_plan,
        )

        registry = accepted.prepared_context.scientific_identity[
            "split_registry"
        ]
        registry_sha = accepted.prepared_context.scientific_identity[
            "split_registry_content_sha256"
        ]
        current_plan = build_canonical_stage1_scope_plan(
            registry=registry,
            registry_content_sha256=registry_sha,
            global_seed=int(self.options.seed),
            physical_fit_identity=(
                Stage1PhysicalFitIdentity.from_mapping(
                    self.request["stage1_physical_fit_identity"]
                )
            ),
            gpu_ids=self.stage1_gpu_ids,
            review_rounds=int(self.options.review_rounds),
            initial_training_partitions=int(
                self.options.initial_training_partitions
            ),
            scope_workers_per_gpu=int(
                self.options.stage1_scope_workers_per_gpu
            ),
            expected_outer_fold_count=int(
                self.options.outer_folds
            ),
            expected_inner_fold_count=(
                int(self.options.initial_training_partitions)
                + int(self.options.review_rounds)
            ),
        )
        from .production_stage1_reusable_preflight import (
            preflight_scope_plan_projection,
        )

        if (
            preflight_scope_plan_projection(current_plan)
            != preflight_scope_plan_projection(
                accepted.state_bundle.plan
            )
        ):
            raise RuntimeError(
                "fast-reopened preflight differs scientifically from the "
                "current deployment-neutral scope plan"
            )
        from .production_stage1_reusable_preflight import (
            ReusableClusterPreflightStateBundle,
        )

        current_state_bundle = ReusableClusterPreflightStateBundle(
            preflight=accepted.preflight,
            plan=current_plan,
        )
        preflight_root = (attempt / "cluster_preflight").resolve()
        state_root = (
            attempt / "cluster_preflight_states"
        ).resolve()
        artifact, state_bundle = publish_reusable_preflight_references(
            preflight_output_root=preflight_root,
            state_output_root=state_root,
            artifact=accepted.preflight,
            state_bundle=current_state_bundle,
        )
        preflight_manifest = (
            preflight_root / "cluster_preflight_manifest.json"
        ).resolve(strict=True)
        state_manifest = (
            state_root / "cluster_state_bundle_manifest.json"
        ).resolve(strict=True)
        from .prepared_stage1_context import (
            seal_prepared_stage1_context_from_authenticated_parts,
            serialize_stage1_build_options,
        )

        current_options = self._stage1_build_options(
            dataset=prepared,
            profile=profile.resolve(strict=True),
            cache=cache,
            output=attempt / "preflight_no_model_output",
            dry_run=False,
            cluster_preflight_manifest_path=preflight_manifest,
            cluster_preflight_state_bundle_manifest_path=state_manifest,
            reusable_preflight_fast_reopen=True,
        )
        current_mapping = serialize_stage1_build_options(
            current_options
        )
        exact_request = (
            self._compile_current_stage1_request_from_reused_preflight(
                accepted=accepted,
                current_options=current_options,
                current_plan=current_plan,
            )
        )
        integration = self.hooks.role_neutral_stage1
        if integration is None:
            raise RuntimeError(
                "fast preflight reopen requires all-ten integration"
            )
        prepared_context = (
            seal_prepared_stage1_context_from_authenticated_parts(
                root=(
                    attempt / "prepared_stage1_context"
                ).resolve(),
                stage1_build_options=current_mapping,
                architecture_profiles=(
                    integration.producer_factories_builder.architecture_profiles
                ),
                runtime_compatibility_class=(
                    integration.producer_factories_builder.runtime_compatibility_class
                ),
                exact_stage1_request=exact_request,
                registry=registry,
                registry_content_sha256=registry_sha,
            )
        )
        plan = current_plan
        artifact_identity = artifact.identity()
        if (
            artifact_identity["scope_count"] != len(plan.scopes)
            or artifact_identity["physical_fit_count"]
            != len(plan.physical_scopes)
        ):
            raise RuntimeError(
                "fast-reopened preflight changed scope coverage"
            )
        counts: dict[str, int] = {}
        for scope in plan.scopes:
            counts[scope.scope_kind] = (
                counts.get(scope.scope_kind, 0) + 1
            )
        telemetry = {
            "schema_version": (
                "production_stage1_reusable_preflight_telemetry_v1"
            ),
            "reopen_route": (
                "accepted_input_prior_proof_and_stat_continuity"
                if accepted.authentication_mode
                == "prior_proof_stat_continuity"
                else "accepted_input_full_byte_reauthentication"
            ),
            "global_audit_seconds": 0.0,
            "global_audit_authentication_seconds": (
                accepted.global_audit_authentication_seconds
            ),
            "global_audit_authentication_mode": (
                accepted.global_audit_authentication_mode
            ),
            "global_audit_payload_bytes_read": (
                accepted.global_audit_payload_bytes_read
            ),
            "owner_total_count": len(plan.physical_scopes),
            "owner_reused_count": len(plan.physical_scopes),
            "owner_recomputed_count": 0,
            "owner_incomplete_count": 0,
            "owner_fast_stat_count": int(
                accepted.preflight.authentication[
                    "owner_fast_stat_count"
                ]
            ),
            "owner_deep_auth_count": int(
                accepted.preflight.authentication[
                    "owner_deep_auth_count"
                ]
            ),
            "actual_worker_concurrency": 0,
            "scope_input_publication_seconds": 0.0,
            "scope_input_publication_bytes": 0,
            "authentication_seconds": (
                time.perf_counter() - started
            ),
            "authentication_payload_bytes_read": int(
                accepted.payload_bytes_read
                + accepted.global_audit_payload_bytes_read
                + accepted.preflight.authentication[
                    "payload_bytes_read"
                ]
            ),
            "htr_retokenization_performed": False,
            "kmeans_or_svd_refit_performed": False,
            "bulk_preflight_payload_read_on_fast_path": (
                accepted.authentication_mode
                != "prior_proof_stat_continuity"
                or accepted.global_audit_authentication_mode
                != "prior_proof_stat_continuity"
                or accepted.preflight.authentication[
                    "assembled_authentication_mode"
                ]
                != "prior_proof_stat_continuity"
                or int(
                    accepted.preflight.authentication[
                        "owner_deep_auth_count"
                    ]
                )
                != 0
            ),
            "deployment_execution_attestation": copy.deepcopy(
                self.options.stage1_preflight_execution_attestation
            ),
            "actual_worker_concurrency_within_every_derived_cap": True,
        }
        scope_input_body = {
            "schema_version": (
                "production_stage1_reused_scope_inputs_v1"
            ),
            "scope_count": 0,
            "scope_inputs_republished": False,
            "all_physical_owners_reused": True,
            "portable_v2_no_refit_import_used": False,
        }
        scope_input_identity = {
            **scope_input_body,
            "content_sha256": _sha(scope_input_body),
        }
        resource = self._gpu_preflight()
        payload = {
            "schema_version": STAGE1_PREFLIGHT_PHASE_SCHEMA,
            "resource_preflight": resource,
            "cache_phase_reopened_and_rehashed": False,
            "effective_profile_path": str(
                profile.resolve(strict=True)
            ),
            "cluster_preflight_manifest_path": str(
                preflight_manifest
            ),
            "cluster_preflight_identity": artifact_identity,
            "cluster_preflight_state_bundle_manifest_path": str(
                state_manifest
            ),
            "cluster_preflight_state_bundle_content_sha256": (
                state_bundle.content_sha256
            ),
            "cluster_preflight_physical_state_count": len(
                state_bundle.states
            ),
            "cluster_preflight_states_are_canonical_no_refit": True,
            "prepared_stage1_context_manifest_path": str(
                prepared_context.manifest_path
            ),
            "prepared_stage1_context_scientific_content_root_sha256": (
                prepared_context.content_root_sha256
            ),
            "prepared_stage1_context_sealed_during_preflight": True,
            "prepared_stage1_context_checkpoint_placement": (
                "nested_terminal_payload_in_stage1_preflight_portable_dag_v1"
            ),
            "cluster_preflight_scope_inputs_identity": (
                scope_input_identity
            ),
            "planned_scope_counts": {
                "full_outer": counts.get("full_outer", 0),
                "exact_inner": counts.get("exact_inner", 0),
                "cumulative_review": counts.get(
                    "cumulative_spent",
                    0,
                ),
                "total": len(plan.scopes),
            },
            "scientific_cluster_preflight": (
                "accepted_reusable_global_owner_assembled_v1"
            ),
            "reusable_preflight_telemetry": telemetry,
            "preflight_computation_distinguished_from_authentication": True,
            "global_audit_artifact_reusable": True,
            "per_physical_owner_cluster_artifacts_reusable": True,
            "assembled_context_references_owner_payloads": True,
            "scientific_preflight_recomputed_during_supervised_modeling": False,
            "supervised_fit_may_begin_before_scientific_preflight_acceptance": False,
            "stage1_gpu_ids": list(self.stage1_gpu_ids),
            "stage1_execution_device_count": (
                self.options.stage1_execution_device_count
            ),
            "stage1_execution_profile": (
                self.options.stage1_execution_profile.as_dict()
                if isinstance(
                    self.options.stage1_execution_profile,
                    Stage1ExecutionProfile,
                )
                else None
            ),
            "scope_workers_per_gpu": (
                self.options.stage1_scope_workers_per_gpu
            ),
            "preflight_workers": (
                self.options.stage1_preflight_workers
            ),
            "preflight_execution_attestation": copy.deepcopy(
                self.options.stage1_preflight_execution_attestation
            ),
            "seed_policy": self.options.stage1_seed_policy,
            "accepted_context_fast_reopen_used": True,
        }
        report = attempt / "stage1_preflight_report.json"
        report.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        return {
            **payload,
            "terminal_files": [
                str(profile.resolve(strict=True)),
                *[
                    str(path.resolve(strict=True))
                    for root in (
                        preflight_root,
                        state_root,
                        prepared_context.root,
                    )
                    for path in sorted(root.rglob("*"))
                    if path.is_file()
                ],
                str(report.resolve(strict=True)),
            ],
        }

    def _run_default(self, phase: str, attempt: Path) -> Mapping[str, Any]:
        o = self.options
        if phase == "input_preparation":
            prepared = attempt / "prepared"
            result = prepare_modeling_cohort(
                TextPreparationOptions(
                    o.dataset_path,
                    prepared,
                    o.unit_id_column,
                    o.text_column,
                    o.treatment_column,
                    o.outcome_column,
                    o.outcome_type,
                    o.repeated_character_threshold,
                    o.empty_text_policy,
                    o.repeated_character_policy,
                )
            )
            return {
                **result,
                "terminal_files": [
                    result["output"]["path"],
                    str(prepared / "preparation_manifest.json"),
                ],
            }
        if phase == "embedding_cache":
            return self._run_embedding_cache_phase(attempt)
        if phase == "stage1_preflight":
            reopened = self._try_fast_reopen_stage1_preflight(
                attempt
            )
            if reopened is not None:
                return reopened
            resource = self._gpu_preflight()
            cache, prepared = self._embedding_cache_paths()
            profile = self._effective_stage1_profile(
                attempt,
                dataset_path=prepared,
                embedding_cache_dir=cache,
            )
            preflight_builder = ProductionStage1BundleBuilder(
                self._stage1_build_options(
                    dataset=prepared,
                    profile=profile,
                    cache=cache,
                    output=attempt / "preflight_no_model_output",
                    dry_run=True,
                )
            )
            prepared_build = preflight_builder.prepare()
            legacy_migration_decision_path: Path | None = None
            legacy_migration_decision: Mapping[str, Any] | None = None
            legacy_preflight_identity = self.request.get("legacy_preflight_candidate_identity")
            if (
                legacy_preflight_identity is not None
                and isinstance(
                    legacy_preflight_identity,
                    Mapping,
                )
                and legacy_preflight_identity.get(
                    "candidate_kind"
                )
                == "legacy_v4"
            ):
                from .legacy_checkpoint_migration import (
                    plan_legacy_v4_preflight_migration,
                )
                from .physical_fit_deduplication import LogicalContext

                architecture_identity = identity_sha256(
                    prepared_build.request["architecture_contract"]
                )
                configuration_identity = identity_sha256(
                    prepared_build.request["effective_stage1_config"]
                )
                producer_identity = str(
                    self.request["expected_checkpoint_compatibilities_by_phase"][
                        "stage1_preflight"
                    ]["producer_code_identity"]
                )
                logical_contexts = tuple(
                    LogicalContext(
                        canonical_index=int(scope.canonical_index),
                        scope_id=scope.scope_id,
                        purpose=scope.scope_kind,
                        outer_fold=int(scope.outer_fold),
                        fit_row_ids=tuple(scope.fit_row_ids),
                        heldout_row_ids=tuple(scope.heldout_row_ids),
                        architecture_identity=architecture_identity,
                        target="cluster_preflight",
                        scientific_configuration_identity=(configuration_identity),
                        scope_seed=int(scope.scope_seed),
                        producer_identity=producer_identity,
                        runtime_compatibility_class=(o.runtime_compatibility_class),
                    )
                    for scope in prepared_build.stage1_scope_plan.scopes
                )
                migration = plan_legacy_v4_preflight_migration(
                    manifest_path=Path(str(legacy_preflight_identity["manifest_path"])),
                    logical_contexts=logical_contexts,
                    authenticate_registered_payload_bytes=True,
                )
                expected_logical = len(prepared_build.stage1_scope_plan.scopes)
                expected_physical = len(prepared_build.stage1_scope_plan.physical_scopes)
                (
                    legacy_migration_decision_path,
                    legacy_migration_decision,
                ) = _persist_legacy_preflight_recompute_decision(
                    attempt=attempt,
                    consumer_request_sha256=self.request["request_sha256"],
                    source_candidate_identity={
                        name: copy.deepcopy(
                            legacy_preflight_identity[name]
                        )
                        for name in (
                            "selection_source",
                            "manifest_path",
                            "manifest_sha256",
                            "manifest_size_bytes",
                            "manifest_content_sha256",
                            "registered_payloads",
                            "registered_payload_bytes_authenticated_during_request",
                            "direct_reuse_allowed",
                        )
                    },
                    migration=migration,
                    expected_logical_scope_count=expected_logical,
                    expected_physical_fit_count=expected_physical,
                )
            scope_input_identity = prepared_build.cluster_preflight_scope_input_set_identity
            if not isinstance(scope_input_identity, Mapping):
                raise RuntimeError(
                    "Stage 1 preflight omitted its recoverable row-restricted "
                    "scope-input identity"
                )
            from .production_stage1_reusable_preflight import (
                ReusableProductionStage1ClusterPreflightArtifact,
                load_reusable_preflight_reference,
                publish_reusable_preflight_references,
            )

            reusable_preflight = isinstance(
                prepared_build.cluster_preflight_artifact_handle,
                ReusableProductionStage1ClusterPreflightArtifact,
            )
            if reusable_preflight:
                artifact = (
                    prepared_build.cluster_preflight_artifact_handle
                )
                state_bundle = (
                    prepared_build.cluster_preflight_state_bundle
                )
                if state_bundle is None:
                    raise RuntimeError(
                        "reusable preflight omitted its state bundle"
                    )
                publish_reusable_preflight_references(
                    preflight_output_root=(
                        attempt / "cluster_preflight"
                    ).resolve(),
                    state_output_root=(
                        attempt / "cluster_preflight_states"
                    ).resolve(),
                    artifact=artifact,
                    state_bundle=state_bundle,
                )
                # The phase-local manifests are the portable capabilities.
                artifact = load_reusable_preflight_reference(
                    manifest_path=(
                        attempt
                        / "cluster_preflight"
                        / "cluster_preflight_manifest.json"
                    ).resolve(),
                    expected_stage1_request=prepared_build.request,
                    plan=prepared_build.stage1_scope_plan,
                    producer_identity=(
                        STAGE1_REUSABLE_ASSEMBLED_PREFLIGHT_PRODUCER_IDENTITY
                    ),
                )
            elif prepared_build.options.portable_cluster_preflight_v2:
                from .production_stage1_cluster_preflight_artifact_v2 import (
                    seal_portable_production_stage1_cluster_preflight_artifact,
                )

                artifact = (
                    seal_portable_production_stage1_cluster_preflight_artifact(
                        output_dir=(attempt / "cluster_preflight").resolve(),
                        audit=(
                            prepared_build.embedding_cluster_feasibility_audit
                        ),
                        stage1_request=prepared_build.request,
                        config=prepared_build.config,
                        registry=prepared_build.registry,
                        registry_content_sha256=(
                            prepared_build.registry_content_sha256
                        ),
                        embedding_cache_identity=(
                            prepared_build.embedding_cache_identity
                        ),
                        parquet_compression=str(
                            o.cluster_preflight_parquet_compression
                        ),
                    )
                )
            else:
                from .production_stage1_cluster_preflight_artifact import (
                    seal_production_stage1_cluster_preflight_artifact,
                )

                artifact = seal_production_stage1_cluster_preflight_artifact(
                    output_dir=(attempt / "cluster_preflight").resolve(),
                    audit=prepared_build.embedding_cluster_feasibility_audit,
                    stage1_request=prepared_build.request,
                    config=prepared_build.config,
                    registry=prepared_build.registry,
                    registry_content_sha256=prepared_build.registry_content_sha256,
                    embedding_cache_identity=prepared_build.embedding_cache_identity,
                )
            if reusable_preflight:
                state_bundle_manifest = (
                    attempt
                    / "cluster_preflight_states"
                    / "cluster_state_bundle_manifest.json"
                ).resolve(strict=True)
            else:
                captured_states = prepared_build.cluster_preflight_canonical_scope_states
                if not isinstance(captured_states, Mapping):
                    raise RuntimeError(
                        "fresh clustered preflight omitted its canonical "
                        "physical-owner fitted states"
                    )
                from .role_neutral_embedding_group_execution import (
                    seal_canonical_clustered_preflight_state_bundle,
                )

                state_bundle = seal_canonical_clustered_preflight_state_bundle(
                    output_root=(attempt / "cluster_preflight_states").resolve(),
                    preflight=artifact,
                    plan=prepared_build.stage1_scope_plan,
                    captured_scope_states=captured_states,
                )
                state_bundle_manifest = state_bundle.root / "cluster_state_bundle_manifest.json"
            prepared_context = None
            accepted_context = None
            if o.portable_scientific_spec is not None:
                integration = self.hooks.role_neutral_stage1
                if integration is None:
                    raise RuntimeError(
                        "typed portable preflight requires its all-ten "
                        "producer integration"
                    )
                from .prepared_stage1_context import (
                    seal_prepared_stage1_context,
                )

                reusable_options = replace(
                    prepared_build.options,
                    dry_run=False,
                    cluster_preflight_manifest_path=(
                        (
                            attempt
                            / "cluster_preflight"
                            / "cluster_preflight_manifest.json"
                        ).resolve(strict=True)
                        if reusable_preflight
                        else artifact.manifest_path
                    ),
                    cluster_preflight_state_bundle_manifest_path=(
                        state_bundle_manifest
                    ),
                )
                reusable_prepared = replace(
                    prepared_build,
                    options=reusable_options,
                    cluster_preflight_manifest_path=(
                        (
                            attempt
                            / "cluster_preflight"
                            / "cluster_preflight_manifest.json"
                        ).resolve(strict=True)
                        if reusable_preflight
                        else artifact.manifest_path
                    ),
                    cluster_preflight_artifact_identity=(
                        artifact.identity()
                    ),
                    cluster_preflight_artifact_handle=artifact,
                    cluster_preflight_state_bundle=state_bundle,
                )
                prepared_context = seal_prepared_stage1_context(
                    root=(
                        attempt / "prepared_stage1_context"
                    ).resolve(),
                    prepared=reusable_prepared,
                    producer_factories_builder=(
                        integration.producer_factories_builder
                    ),
                )
                if reusable_preflight:
                    from .production_stage1_bundle import (
                        STAGE1_REUSABLE_CLUSTER_OWNER_PRODUCER_IDENTITY,
                        STAGE1_REUSABLE_GLOBAL_AUDIT_PRODUCER_IDENTITY,
                    )
                    from .production_stage1_reusable_preflight import (
                        publish_reusable_preflight_acceptance,
                    )

                    accepted_context = (
                        publish_reusable_preflight_acceptance(
                            store_root=(
                                self._reusable_preflight_store_root()
                            ),
                            selector=(
                                self._reusable_preflight_accepted_input_selector()
                            ),
                            artifact=artifact,
                            prepared_context_manifest_path=(
                                prepared_context.manifest_path
                            ),
                            producer_identity=(
                                STAGE1_REUSABLE_ASSEMBLED_PREFLIGHT_PRODUCER_IDENTITY
                            ),
                            owner_producer_identity=(
                                STAGE1_REUSABLE_CLUSTER_OWNER_PRODUCER_IDENTITY
                            ),
                            global_audit_producer_identity=(
                                STAGE1_REUSABLE_GLOBAL_AUDIT_PRODUCER_IDENTITY
                            ),
                        )
                    )
            report = attempt / "stage1_preflight_report.json"
            inner_partition_count = int(o.initial_training_partitions) + o.review_rounds
            planned_scope_count = o.outer_folds * (1 + inner_partition_count + o.review_rounds)
            artifact_identity = artifact.identity()
            if artifact_identity["scope_count"] != planned_scope_count:
                raise RuntimeError("scientific Stage 1 preflight did not cover every planned scope")
            payload = {
                "schema_version": STAGE1_PREFLIGHT_PHASE_SCHEMA,
                "resource_preflight": resource,
                "cache_phase_reopened_and_rehashed": True,
                "effective_profile_path": str(profile.resolve(strict=True)),
                "cluster_preflight_manifest_path": str(
                    (
                        attempt
                        / "cluster_preflight"
                        / "cluster_preflight_manifest.json"
                    ).resolve(strict=True)
                    if reusable_preflight
                    else artifact.manifest_path
                ),
                "cluster_preflight_identity": artifact_identity,
                "cluster_preflight_state_bundle_manifest_path": str(state_bundle_manifest),
                "cluster_preflight_state_bundle_content_sha256": (state_bundle.content_sha256),
                "cluster_preflight_physical_state_count": len(state_bundle.states),
                "cluster_preflight_states_are_canonical_no_refit": True,
                "prepared_stage1_context_manifest_path": (
                    None
                    if prepared_context is None
                    else str(prepared_context.manifest_path)
                ),
                "prepared_stage1_context_scientific_content_root_sha256": (
                    None
                    if prepared_context is None
                    else prepared_context.content_root_sha256
                ),
                "prepared_stage1_context_sealed_during_preflight": (
                    prepared_context is not None
                ),
                "prepared_stage1_context_checkpoint_placement": (
                    "nested_terminal_payload_in_stage1_preflight_portable_dag_v1"
                    if prepared_context is not None
                    else None
                ),
                "reusable_preflight_accepted_context_terminal_path": (
                    None
                    if accepted_context is None
                    else str(
                        accepted_context.root
                        / "accepted_context_terminal.json"
                    )
                ),
                "reusable_preflight_accepted_context_authentication_mode": (
                    None
                    if accepted_context is None
                    else accepted_context.authentication_mode
                ),
                "cluster_preflight_scope_inputs_identity": copy.deepcopy(
                    dict(scope_input_identity)
                ),
                "planned_scope_counts": {
                    "full_outer": o.outer_folds,
                    "exact_inner": o.outer_folds * inner_partition_count,
                    "cumulative_review": o.outer_folds * o.review_rounds,
                    "total": planned_scope_count,
                },
                "scientific_cluster_preflight": (
                    "accepted_reusable_global_owner_assembled_v1"
                    if reusable_preflight
                    else (
                    "accepted_portable_compact_lossless_v2"
                    if prepared_build.options.portable_cluster_preflight_v2
                    else "accepted_and_independently_sealed_v1"
                    )
                ),
                "reusable_preflight_telemetry": copy.deepcopy(
                    dict(prepared_build.reusable_preflight_telemetry)
                ),
                "preflight_computation_distinguished_from_authentication": True,
                "global_audit_artifact_reusable": reusable_preflight,
                "per_physical_owner_cluster_artifacts_reusable": reusable_preflight,
                "assembled_context_references_owner_payloads": reusable_preflight,
                "scientific_preflight_recomputed_during_supervised_modeling": False,
                "supervised_fit_may_begin_before_scientific_preflight_acceptance": False,
                "stage1_gpu_ids": list(self.stage1_gpu_ids),
                "stage1_execution_device_count": (
                    o.stage1_execution_device_count
                ),
                "stage1_execution_profile": (
                    None
                    if o.stage1_execution_profile is None
                    else o.stage1_execution_profile.as_dict()
                ),
                "scope_workers_per_gpu": o.stage1_scope_workers_per_gpu,
            "preflight_workers": o.stage1_preflight_workers,
            "preflight_execution_attestation": copy.deepcopy(
                o.stage1_preflight_execution_attestation
            ),
                "seed_policy": o.stage1_seed_policy,
            }
            if legacy_migration_decision_path is not None and legacy_migration_decision is not None:
                payload.update(
                    {
                        "legacy_preflight_migration_decision_path": str(
                            legacy_migration_decision_path.resolve(strict=True)
                        ),
                        "legacy_preflight_migration_decision_content_sha256": (
                            legacy_migration_decision["content_sha256"]
                        ),
                        "legacy_preflight_directly_reused": False,
                        "legacy_preflight_physical_fit_recompute_count": (
                            legacy_migration_decision["migration_decision"][
                                "recompute_physical_fit_count"
                            ]
                        ),
                        "legacy_preflight_superseded_duplicate_count": (
                            legacy_migration_decision["migration_decision"][
                                "deduplicated_group_count"
                            ]
                        ),
                    }
                )
            report.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            return {
                **payload,
                "terminal_files": [
                    str(profile),
                    *[
                        str(path)
                        for path in sorted(
                            (
                                attempt / "cluster_preflight"
                                if reusable_preflight
                                else artifact.root
                            ).rglob("*")
                        )
                        if path.is_file()
                    ],
                    *[
                        str(path)
                        for path in sorted(
                            (
                                attempt / "cluster_preflight_states"
                                if reusable_preflight
                                else state_bundle.root
                            ).rglob("*")
                        )
                        if path.is_file()
                    ],
                    *(
                        [
                            str(path)
                            for path in sorted(
                                prepared_context.root.rglob("*")
                            )
                            if path.is_file()
                        ]
                        if prepared_context is not None
                        else []
                    ),
                    *(
                        [str(legacy_migration_decision_path)]
                        if legacy_migration_decision_path is not None
                        else []
                    ),
                    str(report),
                ],
            }
        if phase == "stage1_modeling":
            if self.options.portable_scientific_spec is not None:
                return self._run_portable_role_neutral_stage1_modeling(attempt)
            resource = self._gpu_preflight()
            cache, prepared = self._embedding_cache_paths()
            profile, cluster_preflight_manifest = self._stage1_preflight_paths()
            bundle = attempt / "stage1_bundle"
            result = ProductionStage1BundleBuilder(
                self._stage1_build_options(
                    dataset=prepared,
                    profile=profile,
                    cache=cache,
                    output=bundle,
                    dry_run=False,
                    cluster_preflight_manifest_path=cluster_preflight_manifest,
                )
            ).build()
            manifest = bundle / "bundle_manifest.json"
            return {
                **result,
                "resource_preflight": resource,
                "stage1_gpu_ids": list(self.stage1_gpu_ids),
                "effective_profile_reused_from_preflight": str(profile),
                "cluster_preflight_manifest_consumed": str(cluster_preflight_manifest),
                "scientific_cluster_preflight_recomputed": False,
                "terminal_files": [str(manifest)],
            }
        if phase == "handoff_validation":
            stage1 = self._validated_complete("stage1_modeling")
            manifest = next(
                Path(row["path"])
                for row in stage1["artifacts"]
                if Path(row["path"]).name == "bundle_manifest.json"
            )
            report = attempt / "fresh_handoff_validation.json"
            validation = self._validate_handoff_in_fresh_process(
                bundle_manifest=manifest,
                report_path=report,
                portable_role_neutral=(self.options.portable_scientific_spec is not None),
            )
            return {
                "fresh_process_validation": validation,
                "source_snapshot": self.request.get("source_snapshot"),
                "terminal_files": [str(report)],
            }
        if phase == "stage2_canary":
            from scripts.canary_production_stage1_hierarchy import run_canary

            options = self._stage2_options(attempt, prefix="canary")
            result = run_canary(options)
            return {**result, "terminal_files": [result["report_path"]]}
        if phase == "stage2_inference":
            from .production_stage1_hierarchy_one_shot import (
                run_production_stage1_hierarchy_one_shot,
            )

            options = self._stage2_options(attempt, prefix="full")
            result = run_production_stage1_hierarchy_one_shot(options)
            prediction = options.output_dir / "frozen_predictions.parquet"
            manifest = options.output_dir / "immutable_run_manifest.json"
            attestation = Path(str(result["attestation_path"])).resolve(strict=True)
            if result.get("mode") == "reference_only_role_neutral_stage2":
                direct_terminals = list(
                    _portable_stage2_terminal_file_inventory(
                        result=result,
                        prediction_path=prediction,
                        run_manifest_path=manifest,
                        attestation_path=attestation,
                    )
                )
                return {
                    **result,
                    "terminal_files": direct_terminals,
                }
            return {
                **result,
                "terminal_files": [
                    str(prediction),
                    str(manifest),
                    str(attestation),
                ],
            }
        if phase == "oracle_evaluation":
            if not o.evaluate_oracle_posthoc:
                return {"skipped_by_configuration": True, "terminal_files": []}
            from .production_oracle_evaluation import evaluate_frozen_predictions_posthoc

            inference = self._validated_complete("stage2_inference")
            files = [Path(row["path"]) for row in inference["artifacts"]]
            prediction = next(path for path in files if path.name == "frozen_predictions.parquet")
            manifest = next(path for path in files if path.name == "immutable_run_manifest.json")
            stage1 = self._validated_complete("stage1_modeling")
            bundle_manifest = next(
                Path(row["path"])
                for row in stage1["artifacts"]
                if Path(row["path"]).name == "bundle_manifest.json"
            )
            row_map = bundle_manifest.parent / "row_registry.parquet"
            result = evaluate_frozen_predictions_posthoc(
                predictions_path=prediction,
                prediction_manifest_path=manifest,
                unit_id_map_path=row_map,
                oracle_dataset_path=o.oracle_dataset_path,
                output_dir=attempt / "evaluation",
                unit_id_column=o.unit_id_column,
                oracle_unit_id_column=o.oracle_unit_id_column,
                oracle_ite_column=o.oracle_ite_column,
            )
            return {
                **result,
                "terminal_files": [
                    result["joined_path"],
                    str(attempt / "evaluation/evaluation_metrics.json"),
                ],
            }
        if phase == "terminal_validation":
            report = attempt / "validation.json"
            validation = self._validate_terminal_in_fresh_process(report_path=report)
            return {**validation, "terminal_files": [str(report)]}
        raise AssertionError(phase)

    def _stage2_options(self, attempt: Path, *, prefix: str) -> Any:
        from .all_evidence_post_extraction_review import CausalReviewConfig
        from .first_untouched_gate_direct_numerical_preparation import (
            FirstUntouchedGatePreparationBounds,
        )
        from .hierarchical_discovery_job_cache import (
            HierarchicalDiscoveryJobCacheConfig,
        )
        from .hierarchical_discovery_response_contract import (
            HierarchyWireBudget,
        )
        from .production_stage1_hierarchy_one_shot import (
            ProductionStage1HierarchyOneShotOptions,
            Stage2HierarchyPromptProtocol,
        )

        o = self.options
        if not isinstance(o.endpoint, str) or not isinstance(o.model_name, str):
            raise RuntimeError("Stage 2 options were requested without endpoint/model identity")
        from ..extraction.llm_routing import (
            resolve_stage2_endpoint_authentication,
            resolve_stage2_endpoint_transport,
            validate_stage2_endpoint_runtime_configuration,
        )

        endpoint_auth = resolve_stage2_endpoint_authentication()
        endpoint_transport = resolve_stage2_endpoint_transport()
        validate_stage2_endpoint_runtime_configuration(
            authentication=endpoint_auth,
            transport=endpoint_transport,
        )
        expected_auth = self.request.get(
            "stage2_endpoint_authentication"
        )
        if endpoint_auth.identity["mode"] == "none":
            if expected_auth is not None:
                raise RuntimeError(
                    "Stage 2 endpoint authentication changed after request creation"
                )
        elif expected_auth != dict(endpoint_auth.identity):
            raise RuntimeError(
                "Stage 2 endpoint authentication changed after request creation"
            )
        expected_transport = self.request.get("stage2_endpoint_transport")
        if endpoint_transport.mode == "vllm":
            if expected_transport is not None:
                raise RuntimeError(
                    "Stage 2 endpoint transport changed after request creation"
                )
        elif expected_transport != dict(endpoint_transport.identity):
            raise RuntimeError(
                "Stage 2 endpoint transport changed after request creation"
            )
        stage1 = self._validated_complete("stage1_modeling")
        stage1_artifact_paths = tuple(Path(row["path"]) for row in stage1["artifacts"])
        bundle_manifest = next(
            path for path in stage1_artifact_paths if path.name == "bundle_manifest.json"
        )
        direct_numerical_bank_manifest: Path | None = None
        prepared_cohort_path: Path | None = None
        upstream_review_policy: str | None = None
        causal_review_values = o.post_extraction_causal_review.as_dict()
        upstream_review_policy = str(causal_review_values.pop("upstream_review_policy"))
        causal_review_values.pop("scientific_policy")
        if o.portable_scientific_spec is not None:
            direct_numerical_bank_manifest = next(
                (
                    path
                    for path in stage1_artifact_paths
                    if path.name == "direct_upstream_numerical_manifest.json"
                ),
                None,
            )
            if direct_numerical_bank_manifest is None:
                raise RuntimeError(
                    "portable Stage 1 checkpoint lacks its direct numerical "
                    "reference bank manifest"
                )
            _cache, prepared_cohort_path = self._embedding_cache_paths()
        protocol_values = o.stage2_prompt_protocol.as_dict()
        protocol_values["hierarchy_wire_budget"] = HierarchyWireBudget.from_mapping(
            protocol_values["hierarchy_wire_budget"]
        )
        generation_policy = o.stage2_prompt_protocol.generation_policy
        protocol_values["generation_policy"] = generation_policy
        return ProductionStage1HierarchyOneShotOptions(
            bundle_manifest_path=bundle_manifest,
            output_dir=attempt / f"{prefix}_output",
            preparation_dir=attempt / f"{prefix}_preparation",
            attestation_dir=attempt / f"{prefix}_attestation",
            endpoint=o.endpoint,
            model_name=o.model_name,
            endpoint_api_key=endpoint_auth.api_key,
            stage2_tokenizer_locator=o.stage2_tokenizer_locator,
            review_rounds=o.review_rounds,
            initial_training_partitions=o.initial_training_partitions,
            stage2_protocol=Stage2HierarchyPromptProtocol(**protocol_values),
            post_extraction_scientific_policy=(o.post_extraction_causal_review.scientific_policy),
            post_extraction_review_config=CausalReviewConfig(
                estimator_policy=(
                    o.post_extraction_causal_review.scientific_policy.review_estimator
                ),
                **causal_review_values,
            ),
            source_text_temporally_valid_by_design=(o.source_text_temporally_valid_by_design),
            interaction_inner_folds=o.interaction_inner_folds,
            tfidf_nested_calibration_folds=o.tfidf_nested_calibration_folds,
            review_stage1_device=o.review_device,
            review_neural_query_devices=(o.review_device,),
            hierarchical_discovery_job_cache_config=(
                HierarchicalDiscoveryJobCacheConfig(
                    max_entry_bytes=(
                        o.resource_performance_safety
                        .hierarchical_job_cache_max_entry_bytes
                    )
                )
            ),
            first_untouched_gate_preparation_bounds=(
                FirstUntouchedGatePreparationBounds(
                    max_initial_spent_rows=(
                        o.resource_performance_safety
                        .first_untouched_gate_max_initial_spent_rows
                    ),
                    max_first_gate_rows=(
                        o.resource_performance_safety
                        .first_untouched_gate_max_first_gate_rows
                    ),
                    max_total_text_utf8_bytes=(
                        o.resource_performance_safety
                        .first_untouched_gate_max_total_text_utf8_bytes
                    ),
                    max_catalog_atoms=(
                        o.resource_performance_safety
                        .first_untouched_gate_max_catalog_atoms
                    ),
                    max_source_manifest_bytes=(
                        o.resource_performance_safety
                        .first_untouched_gate_max_source_manifest_bytes
                    ),
                    max_direct_numerical_signals=(
                        o.resource_performance_safety
                        .first_untouched_gate_max_direct_numerical_signals
                    ),
                    max_single_matrix_file_bytes=(
                        o.resource_performance_safety
                        .first_untouched_gate_max_single_matrix_file_bytes
                    ),
                    max_total_matrix_file_bytes=(
                        o.resource_performance_safety
                        .first_untouched_gate_max_total_matrix_file_bytes
                    ),
                )
            ),
            max_candidates=int(o.max_candidate_variables),
            seed=o.seed,
            forest_runtime_config=o.forest_runtime_config,
            forest_n_estimators=o.forest_n_estimators,
            forest_max_depth=o.forest_max_depth,
            forest_min_samples_leaf=o.forest_min_samples_leaf,
            forest_max_features=o.forest_max_features,
            forest_honest=o.forest_honest,
            forest_inference=o.forest_inference,
            forest_subforest_size=o.forest_subforest_size,
            forest_tune_model=o.forest_tune_model,
            forest_nuisance_n_estimators=o.forest_nuisance_n_estimators,
            forest_nuisance_max_depth=o.forest_nuisance_max_depth,
            forest_nuisance_min_samples_leaf=(o.forest_nuisance_min_samples_leaf),
            forest_nuisance_treatment_max_features=(o.forest_nuisance_treatment_max_features),
            forest_nuisance_outcome_max_features=(o.forest_nuisance_outcome_max_features),
            forest_random_seed=o.forest_random_seed,
            forest_n_jobs=(None if o.forest_runtime_config is not None else o.cpu_budget),
            proposal_schema_repair_attempts=(
                generation_policy.feature_proposal_review.schema_repair_attempts
            ),
            request_max_retries=(
                generation_policy.interpret_architecture_chunk.transport_max_retries
            ),
            extraction_batch_size=o.response_concurrency,
            extraction_max_text_length=int(o.complete_page_max_chars),
            complete_page_core_chars=int(o.complete_page_core_chars),
            complete_page_context_chars=int(o.complete_page_context_chars),
            complete_page_max_chars=int(o.complete_page_max_chars),
            complete_reconciliation_fan_in=int(o.complete_reconciliation_fan_in),
            prepared_cohort_path=prepared_cohort_path,
            unit_id_column=(o.unit_id_column if prepared_cohort_path is not None else None),
            text_column=(o.text_column if prepared_cohort_path is not None else None),
            treatment_column=(o.treatment_column if prepared_cohort_path is not None else None),
            outcome_column=(o.outcome_column if prepared_cohort_path is not None else None),
            outcome_type=(o.outcome_type if prepared_cohort_path is not None else None),
            direct_numerical_bank_manifest_path=(direct_numerical_bank_manifest),
            upstream_review_policy=upstream_review_policy,
            resume=o.run_control.resume,
        )

    def _execute_phase_sequence(
        self,
        sequence: Sequence[str],
    ) -> dict[str, Mapping[str, Any]]:
        """Execute or authenticate an ordered workflow prefix."""

        completed: dict[str, Any] = {}
        hook_by_phase: Mapping[str, WorkflowPhaseHook | None] = {
            "embedding_cache": self.hooks.embedding_cache,
            "stage1_preflight": self.hooks.stage1_preflight,
            "stage1_modeling": self.hooks.stage1_modeling,
        }
        for phase in sequence:
            if phase not in self._phase_sequence():
                raise ValueError(f"phase is outside this workflow request: {phase}")
            _revalidate_request_bound_external_inputs(
                self.request,
                authenticated_adoptions=self._adopted_artifact_handles,
                identity_memo=self._scientific_identity_memo,
            )
            adopted = self._adopted_record_for_phase(phase)
            if adopted is not None:
                with self.telemetry.subphase(f"{phase}.checkpoint_adoption_substitution"):
                    existing = self._validated_complete(phase)
                    if existing is None:
                        existing = self._publish_adopted_phase_reference(phase)
                    self._checkpoint_artifact_for_phase(
                        phase,
                        required=True,
                    )
                completed[phase] = existing
                self._write_progress(
                    status="running",
                    completed=tuple(completed),
                    current_phase=None,
                )
                self._write_performance_telemetry()
                continue
            if self.options.run_control.resume:
                with self.telemetry.subphase(f"{phase}.resume_authentication"):
                    existing = self._validated_complete(phase)
            else:
                existing = None
            if existing is not None:
                with self.telemetry.subphase(f"{phase}.checkpoint_publication_authentication"):
                    self._publish_completed_phase_checkpoint(
                        phase,
                        existing,
                    )
                completed[phase] = existing
                self._write_progress(
                    status="running",
                    completed=tuple(completed),
                    current_phase=None,
                )
                self._write_performance_telemetry()
                continue
            attempt = self._attempt_dir(phase)
            if phase == "oracle_evaluation" and self.options.evaluate_oracle_posthoc:
                # This full-byte-authenticated handle must exist before any
                # oracle evaluator can be constructed or the oracle opened.
                self._checkpoint_artifact_for_phase(
                    "stage2_inference",
                    required=True,
                )
            self._write_progress(
                status="running",
                completed=tuple(completed),
                current_phase=phase,
            )
            try:
                with self.telemetry.subphase(f"{phase}.compute"):
                    if phase in self.phase_overrides:
                        result = self.phase_overrides[phase](attempt)
                    elif hook_by_phase.get(phase) is not None:
                        hook = hook_by_phase[phase]
                        assert hook is not None
                        result = hook(attempt, self._phase_hook_context(phase, attempt))
                    else:
                        result = self._run_default(phase, attempt)
                with self.telemetry.subphase(f"{phase}.proof_and_publication"):
                    _revalidate_request_bound_external_inputs(
                        self.request,
                        authenticated_adoptions=self._adopted_artifact_handles,
                        identity_memo=self._scientific_identity_memo,
                    )
                    completed_manifest = self._complete(
                        phase,
                        result,
                        attempt_dir=attempt,
                    )
                    self._publish_completed_phase_checkpoint(
                        phase,
                        completed_manifest,
                    )
                    completed[phase] = completed_manifest
                self._write_performance_telemetry()
            except BaseException as exc:
                self._write_performance_telemetry()
                self._write_progress(
                    status="failed",
                    completed=tuple(completed),
                    current_phase=phase,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            self._write_progress(
                status="running",
                completed=tuple(completed),
                current_phase=None,
            )
        return completed

    def _write_performance_telemetry(self) -> None:
        """Publish operational telemetry outside scientific artifact identity."""

        if not self.options.work_root.is_dir():
            return
        payload = dict(self.telemetry.as_dict())
        safety = self.options.resource_performance_safety
        payload["resource_performance_safety"] = safety.as_dict()
        payload["resource_performance_safety_sha256"] = safety.content_sha256
        _atomic_write_json(
            self.options.work_root / "execution_attestations" / "performance_telemetry.json",
            payload,
        )

    @staticmethod
    def _registered_file_identity(path: Path) -> Mapping[str, Any]:
        resolved = path.resolve(strict=True)
        digest, size = stable_file_sha256(resolved)
        return {
            "path": str(resolved),
            "sha256": digest,
            "size_bytes": size,
        }

    def _validate_canary_preparation_in_fresh_process(
        self,
    ) -> Mapping[str, Any]:
        script = r"""
import json
import os
from pathlib import Path
import sys
import oci.inference.production_all_evidence_workflow as workflow_module
from oci.inference.production_all_evidence_workflow import validate_stage1_canary_descriptor_preparation

result = validate_stage1_canary_descriptor_preparation(Path(sys.argv[1]))
print(json.dumps({
    "result": result,
    "validator_module_path": str(Path(workflow_module.__file__).resolve(strict=True)),
    "source_snapshot_marker": os.environ.get(
        workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV
    ),
    "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    "python_path": os.environ.get("PYTHONPATH"),
    "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
}, sort_keys=True, allow_nan=False))
"""
        source_snapshot = self.request.get("source_snapshot")
        if not isinstance(source_snapshot, Mapping):
            raise RuntimeError("canary preparation fresh validation requires a source snapshot")
        snapshot_root = Path(str(source_snapshot.get("root", ""))).resolve(strict=True)
        snapshot_sha = str(source_snapshot.get("content_sha256") or "")
        expected_hash_seed = str(int(self.options.seed))
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONPATH"] = str(snapshot_root)
        environment[SOURCE_SNAPSHOT_EXECUTION_ENV] = snapshot_sha
        environment["PYTHONHASHSEED"] = expected_hash_seed
        completed = subprocess.run(
            [
                sys.executable,
                "-P",
                "-c",
                script,
                str(self.options.work_root.resolve(strict=True)),
            ],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
        payload = json.loads(completed.stdout)
        if (
            not isinstance(payload, Mapping)
            or not isinstance(payload.get("result"), Mapping)
            or not isinstance(payload.get("validator_module_path"), str)
            or payload.get("source_snapshot_marker") != snapshot_sha
            or payload.get("python_hash_seed") != expected_hash_seed
            or payload.get("python_path") != str(snapshot_root)
            or payload.get("python_no_user_site") != "1"
        ):
            raise RuntimeError("fresh canary preparation validator returned invalid output")
        loaded = Path(payload["validator_module_path"]).resolve(strict=True)
        try:
            loaded.relative_to(snapshot_root)
        except ValueError as exc:
            raise RuntimeError(
                "canary preparation validator did not execute from source snapshot"
            ) from exc
        return copy.deepcopy(dict(payload["result"]))

    def prepare_stage1_canary_descriptors_only(self) -> Mapping[str, Any]:
        """Seal the exact final-run prefix and descriptors without fitting."""

        if not self.options.stage1_only or len(self.stage1_gpu_ids) != 2:
            raise ValueError(
                "canary descriptor preparation requires Stage-1-only mode and "
                "exactly two Stage 1 GPUs"
            )
        if self.options.source_snapshot_root is None:
            raise ValueError(
                "canary descriptor preparation requires one authenticated " "source snapshot"
            )
        self._initialize()
        _revalidate_request_bound_external_inputs(
            self.request,
            authenticated_adoptions=self._adopted_artifact_handles,
            identity_memo=self._scientific_identity_memo,
        )
        prefix = ("input_preparation", "embedding_cache", "stage1_preflight")
        completed = self._execute_phase_sequence(prefix)
        self._write_progress(
            status="preparing_canary_descriptors",
            completed=tuple(completed),
            current_phase="canary_descriptor_preparation",
        )
        try:
            cache, prepared_path = self._embedding_cache_paths()
            profile, cluster_preflight_manifest = self._stage1_preflight_paths()
            prepared = ProductionStage1BundleBuilder(
                self._stage1_build_options(
                    dataset=prepared_path,
                    profile=profile,
                    cache=cache,
                    output=(
                        self.options.work_root / "recovery" / "canary_descriptor_no_model_output"
                    ).resolve(),
                    dry_run=False,
                    cluster_preflight_manifest_path=cluster_preflight_manifest,
                )
            ).prepare()
            from .production_stage1_legacy_scope_adapter import (
                LEGACY_STAGE1_SCOPE_DESCRIPTOR_SET_MANIFEST,
                publish_legacy_stage1_scope_descriptor,
                validate_legacy_stage1_scope_descriptor_set,
            )

            descriptor_set = publish_legacy_stage1_scope_descriptor(
                prepared=prepared,
                descriptor_root=prepared.scope_descriptor_root,
            )
            descriptor_set = validate_legacy_stage1_scope_descriptor_set(
                descriptor_root=descriptor_set.root,
                expected_stage1_request_sha256=prepared.request_sha256,
                prepared=prepared,
            )
            configured_gpu_ids = _canary_stage1_gpu_ids_from_request(
                self.request
            )
            if configured_gpu_ids != tuple(self.stage1_gpu_ids):
                raise RuntimeError(
                    "immutable canary GPU inventory differs from resolved workflow devices"
                )
            selected = _select_configured_canary_descriptor(
                descriptor_set.descriptors,
                configured_gpu_ids=configured_gpu_ids,
            )
            descriptor_set_manifest = (
                descriptor_set.root / LEGACY_STAGE1_SCOPE_DESCRIPTOR_SET_MANIFEST
            )
            preflight_phase_manifest = self._phase_manifest("stage1_preflight")
            body = {
                "schema_version": ("production_stage1_canary_descriptor_preparation_v2"),
                "status": "complete",
                "workflow_request_sha256": self.request["request_sha256"],
                "stage1_request_sha256": prepared.request_sha256,
                "source_snapshot": copy.deepcopy(self.request.get("source_snapshot")),
                "completed_workflow_prefix": list(prefix),
                "cluster_preflight_manifest": self._registered_file_identity(
                    cluster_preflight_manifest
                ),
                "stage1_preflight_phase_manifest": (
                    self._registered_file_identity(preflight_phase_manifest)
                ),
                "descriptor_set_manifest": self._registered_file_identity(descriptor_set_manifest),
                "descriptor_set_content_sha256": descriptor_set.manifest["content_sha256"],
                "descriptor_count": len(descriptor_set.descriptors),
                "selected_scope_id": selected.scope_id,
                "selected_scope_kind": selected.scope.scope_kind,
                "selected_configured_gpu_id": int(selected.assignment.gpu_id),
                "selected_descriptor_manifest": self._registered_file_identity(
                    selected.manifest_path
                ),
                "supervised_stage1_fits_started": False,
                "tfidf_component_started": False,
                "neural_query_component_started": False,
                "remote_clients_constructed": False,
                "remote_calls_made": False,
            }
            manifest = {**body, "content_sha256": _sha(body)}
            target = (
                self.options.work_root / "recovery" / "canary_descriptor_preparation_manifest.json"
            )
            if target.exists() or target.is_symlink():
                observed = _read_json_object(
                    target,
                    label="canary descriptor preparation manifest",
                )
                if observed != manifest:
                    raise RuntimeError("existing canary descriptor preparation manifest changed")
            else:
                _atomic_write_json(target, manifest)
            reopened = _read_json_object(
                target,
                label="canary descriptor preparation manifest",
            )
            if reopened != manifest:
                raise RuntimeError("canary descriptor preparation manifest failed fresh validation")
            reopened = self._validate_canary_preparation_in_fresh_process()
            if reopened != manifest:
                raise RuntimeError("fresh process changed the canary preparation result")
            _revalidate_request_bound_external_inputs(
                self.request,
                authenticated_adoptions=self._adopted_artifact_handles,
                identity_memo=self._scientific_identity_memo,
            )
        except BaseException as exc:
            self._write_progress(
                status="failed",
                completed=tuple(completed),
                current_phase="canary_descriptor_preparation",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._write_progress(
            status="canary_descriptors_ready",
            completed=tuple(completed),
            current_phase=None,
        )
        return reopened

    def run(self) -> Mapping[str, Any]:
        self._initialize()
        _revalidate_request_bound_external_inputs(
            self.request,
            authenticated_adoptions=self._adopted_artifact_handles,
            identity_memo=self._scientific_identity_memo,
        )
        full_sequence = self._phase_sequence()
        if self.options.run_control.stop_after is None:
            execution_sequence = full_sequence
        else:
            stop_index = full_sequence.index(self.options.run_control.stop_after)
            execution_sequence = full_sequence[: stop_index + 1]
        completed = self._execute_phase_sequence(execution_sequence)
        if execution_sequence[-1] != "terminal_validation":
            self._write_progress(
                status="paused",
                completed=tuple(completed),
                current_phase=None,
            )
            return {
                "schema_version": "production_all_evidence_operational_pause_v1",
                "status": "paused",
                "stop_after": execution_sequence[-1],
                "completed_phases": list(completed),
                "request_sha256": self.request["request_sha256"],
                "scientific_sha256": self.request["scientific_identity"]["scientific_sha256"],
                "resume_requires_identical_immutable_request": True,
            }
        terminal_result = completed["terminal_validation"].get("result")
        if not isinstance(terminal_result, Mapping):
            raise RuntimeError(
                "terminal validation phase lacks one closed result"
            )
        self._write_progress(
            status="complete",
            completed=tuple(completed),
            current_phase=None,
        )
        if "terminal_validation" not in self.phase_overrides:
            try:
                achievement = (
                    self._write_validation_achievement_attestation(
                        completed["terminal_validation"]
                    )
                )
            except BaseException as exc:
                self._write_progress(
                    status="failed",
                    completed=tuple(completed),
                    current_phase=None,
                    error=(
                        "validation achievement attestation failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
                raise
            _emit_structured_workflow_log(
                configured_threshold=(
                    self.options.run_control.log_level
                ),
                event_level=logging.INFO,
                payload={
                    "schema_version": (
                        WORKFLOW_STRUCTURED_LOG_EVENT_SCHEMA
                    ),
                    "event": "validation_achievement",
                    "request_sha256": self.request[
                        "request_sha256"
                    ],
                    "status": "accepted",
                    "validation_requested_minimum": (
                        self._validation_policy[
                            "requested_minimum"
                        ]
                    ),
                    "validation_effective_minimum": (
                        self._validation_policy[
                            "effective_minimum"
                        ]
                    ),
                    "validation_achieved_minimum": achievement[
                        "achieved_minimum"
                    ],
                    "validation_achievement_content_sha256": (
                        achievement["content_sha256"]
                    ),
                    "global_release_certified": False,
                },
            )
        return terminal_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--scientific-spec",
        type=Path,
        help=(
            "Required typed path-neutral ScientificWorkflowSpec JSON. Direct "
            "flags cannot synthesize scientific architecture settings."
        ),
    )
    parser.add_argument(
        "--deployment-profile",
        type=Path,
        help="Typed DeploymentProfile JSON containing physical locators/resources.",
    )
    for flag in (
        "dataset",
        "work-root",
        "stage1-profile",
        "query-profile",
        "embedding-local-model-path",
        "htr-local-model-path",
    ):
        parser.add_argument("--" + flag, type=Path)
    for flag in (
        "unit-id-column",
        "text-column",
        "treatment-column",
        "outcome-column",
        "outcome-type",
        "clinical-question",
        "embedding-model-name",
    ):
        parser.add_argument("--" + flag)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        help="Local POSIX scratch root for active work.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        help="Portable resource policy: auto, cpu, or ordered explicit cuda:N devices.",
    )
    parser.add_argument(
        "--stage1-device-count",
        type=int,
        help=(
            "Deployment-selected number of Stage 1 execution devices; "
            "normally populated from the measured benchmark result."
        ),
    )
    parser.add_argument(
        "--stage1-scope-workers-per-device",
        type=int,
        help=(
            "Deployment-selected simultaneous Stage 1 scopes per execution "
            "device; excluded from scientific identity."
        ),
    )
    parser.add_argument(
        "--stage1-persistent-slot-startup-timeout-seconds",
        type=float,
        help=(
            "Finite positive deployment-only deadline for persistent Stage 1 "
            "slots to reconstruct their authenticated context and report ready."
        ),
    )
    parser.add_argument(
        "--stage1-max-parallel-owners",
        type=int,
        help=(
            "Effective complete-owner concurrency after accounting for "
            "multi-device learned-query reservations."
        ),
    )
    parser.add_argument("--stage1-preflight-max-parallel-owners", type=int)
    parser.add_argument("--stage1-preflight-memory-budget-bytes", type=int)
    parser.add_argument(
        "--stage1-preflight-estimated-owner-peak-bytes",
        type=int,
    )
    parser.add_argument("--stage1-preflight-input-io-lane-cap", type=int)
    parser.add_argument(
        "--stage1-preflight-publication-io-lane-cap",
        type=int,
    )
    parser.add_argument(
        "--stage1-preflight-authentication-io-lane-cap",
        type=int,
    )
    parser.add_argument(
        "--stage1-neural-query-topology",
        choices=tuple(sorted(SUPPORTED_STAGE1_EXECUTION_TOPOLOGY_MODES)),
        help=(
            "Deployment-only learned-query context topology; explicit in "
            "direct deployment mode."
        ),
    )
    parser.add_argument("--stage1-htr-training-batch-size", type=int)
    parser.add_argument("--stage1-htr-sentence-encoder-batch-size", type=int)
    parser.add_argument("--stage1-htr-data-loader-workers", type=int)
    parser.add_argument("--stage1-htr-fold-parallelism", type=int)
    parser.add_argument(
        "--stage1-htr-fold-parallel-backend",
        choices=("threads", "processes"),
    )
    parser.add_argument("--stage1-htr-fold-slots-per-device", type=int)
    parser.add_argument(
        "--stage1-htr-reuse-tokenizer-and-chunk-plans",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--stage1-htr-chunk-plan-cache-max-entries", type=int)
    parser.add_argument(
        "--stage1-htr-tokenized-chunk-cache-max-entries",
        type=int,
    )
    parser.add_argument(
        "--stage1-neural-query-inner-fold-parallelism",
        type=int,
    )
    parser.add_argument(
        "--stage1-neural-query-fold-parallel-backend",
        choices=("threads", "processes"),
    )
    parser.add_argument(
        "--stage1-neural-query-fold-slots-per-device",
        type=int,
    )
    parser.add_argument(
        "--stage1-neural-query-bank-parallelism",
        type=int,
    )
    parser.add_argument(
        "--stage1-neural-query-worker-cpu-threads",
        type=int,
    )
    parser.add_argument("--cpu-budget", type=int)
    parser.add_argument(
        "--forest-operational",
        type=Path,
        help=(
            "Closed StrictCausalForestOperationalSpec JSON for direct "
            "deployment mode; separate from the scientific forest spec."
        ),
    )
    parser.add_argument("--response-concurrency", type=int)
    parser.add_argument("--gpu-max-allocation-fraction", type=float)
    parser.add_argument("--gpu-minimum-headroom-bytes", type=int)
    parser.add_argument("--minimum-multi-device-throughput-ratio", type=float)
    parser.add_argument(
        "--maximum-coordination-proof-overhead-ratio",
        type=float,
    )
    parser.add_argument("--maximum-ordinary-read-amplification", type=float)
    parser.add_argument(
        "--minimum-benchmark-repetitions-per-scope",
        type=int,
    )
    parser.add_argument(
        "--performance-read-counter-source",
        choices=("logical_read_bytes", "process_read_bytes"),
    )
    parser.add_argument(
        "--fail-on-external-gpu-occupants",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--hierarchical-job-cache-max-entry-bytes",
        type=int,
    )
    parser.add_argument(
        "--first-untouched-gate-max-initial-spent-rows",
        type=int,
    )
    parser.add_argument(
        "--first-untouched-gate-max-first-gate-rows",
        type=int,
    )
    parser.add_argument(
        "--first-untouched-gate-max-total-text-utf8-bytes",
        type=int,
    )
    parser.add_argument(
        "--first-untouched-gate-max-catalog-atoms",
        type=int,
    )
    parser.add_argument(
        "--first-untouched-gate-max-source-manifest-bytes",
        type=int,
    )
    parser.add_argument(
        "--first-untouched-gate-max-direct-numerical-signals",
        type=int,
    )
    parser.add_argument(
        "--first-untouched-gate-max-single-matrix-file-bytes",
        type=int,
    )
    parser.add_argument(
        "--first-untouched-gate-max-total-matrix-file-bytes",
        type=int,
    )
    parser.add_argument("--max-candidate-variables", type=int)
    parser.add_argument(
        "--stage2-prompt-protocol",
        type=Path,
        help=(
            "Closed JSON object containing every Stage 2 scientific prompt, "
            "coverage, review, and upstream-fit bound. There are no production "
            "defaults."
        ),
    )
    parser.add_argument(
        "--post-extraction-causal-review",
        type=Path,
        help=(
            "Closed JSON object containing every causal-review fitting and "
            "acceptance threshold. There are no production defaults."
        ),
    )
    parser.add_argument("--complete-page-core-chars", type=int)
    parser.add_argument("--complete-page-context-chars", type=int)
    parser.add_argument("--complete-page-max-chars", type=int)
    parser.add_argument("--complete-reconciliation-fan-in", type=int)
    parser.add_argument("--embedding-chunk-size-words", type=int)
    parser.add_argument("--embedding-chunk-overlap-words", type=int)
    parser.add_argument("--embedding-max-chunks", type=int)
    parser.add_argument(
        "--embedding-chunk-selection",
        choices=("first", "last"),
    )
    parser.add_argument("--embedding-max-seq-length", type=int)
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument(
        "--embedding-normalize",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--forest-n-estimators", type=int)
    parser.add_argument(
        "--forest-max-depth",
        type=_nullable_positive_int_argument,
    )
    parser.add_argument("--forest-min-samples-leaf", type=int)
    parser.add_argument(
        "--forest-max-features",
        type=_forest_max_features_argument,
    )
    parser.add_argument(
        "--forest-honest",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--forest-inference",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--forest-subforest-size", type=int)
    parser.add_argument(
        "--forest-tune-model",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--forest-nuisance-n-estimators", type=int)
    parser.add_argument(
        "--forest-nuisance-max-depth",
        type=_nullable_positive_int_argument,
    )
    parser.add_argument("--forest-nuisance-min-samples-leaf", type=int)
    parser.add_argument(
        "--forest-nuisance-treatment-max-features",
        type=_forest_max_features_argument,
    )
    parser.add_argument(
        "--forest-nuisance-outcome-max-features",
        type=_forest_max_features_argument,
    )
    parser.add_argument("--forest-random-seed", type=int)
    parser.add_argument(
        "--storage-backend",
        choices=("posix", "local_posix", "sshfs"),
    )
    parser.add_argument(
        "--cluster-preflight-parquet-compression",
        choices=("none", "zstd"),
        help=(
            "Explicit deployment-only compression for compact clustered-"
            "preflight Parquet payloads."
        ),
    )
    parser.add_argument(
        "--runtime-compatibility-class",
        help="Explicit runtime/ABI compatibility class for direct deployment mode.",
    )
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument(
        "--stage2-tokenizer-locator",
        type=Path,
        help=(
            "Local immutable tokenizer/chat-template files matching the exact "
            "Stage 2 endpoint model; required for full-workflow prompt-length proofs."
        ),
    )
    parser.add_argument("--outer-folds", type=int)
    parser.add_argument("--review-rounds", type=int)
    parser.add_argument("--initial-training-partitions", type=int)
    parser.add_argument("--interaction-inner-folds", type=int)
    parser.add_argument("--tfidf-nested-calibration-folds", type=int)
    parser.add_argument("--stage1-device")
    parser.add_argument(
        "--query-device",
        action="append",
        help="Ordered Stage 1 neural-query device; repeat to use multiple devices.",
    )
    parser.add_argument("--review-device")
    parser.add_argument(
        "--stage1-gpu-id",
        type=int,
        action="append",
        help="Ordered exclusive Stage 1 GPU; repeat once per GPU.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        help="Backward-compatible singular alias for one --stage1-gpu-id.",
    )
    parser.add_argument("--stage1-scope-workers-per-gpu", type=int)
    parser.add_argument("--stage1-preflight-workers", type=int)
    parser.add_argument("--stage1-seed-policy")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--tfidf-workers", type=int)
    parser.add_argument(
        "--tfidf-parallel-backend",
        choices=("threads", "processes"),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--empty-text-policy")
    parser.add_argument("--repeated-character-policy")
    parser.add_argument("--repeated-character-threshold", type=int)
    parser.add_argument(
        "--source-text-temporally-valid-by-design",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--embedding-cache-import",
        type=Path,
        help="Authenticated source cache to relocate into the fresh work root.",
    )
    parser.add_argument(
        "--embedding-cache-import-source-prepared",
        dest="embedding_cache_import_source_prepared_path",
        type=Path,
    )
    parser.add_argument(
        "--embedding-cache-import-source-preparation-manifest",
        dest="embedding_cache_import_source_preparation_manifest_path",
        type=Path,
    )
    parser.add_argument("--source-snapshot-root", type=Path)
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help=(
            "Stop after a fresh-process Stage 1 handoff validation; endpoint/model "
            "are not required and no Stage 2 client is imported or constructed."
        ),
    )
    parser.add_argument(
        "--prepare-stage1-canary-descriptors-only",
        action="store_true",
        help=(
            "Operational pre-launch mode: seal input/cache/preflight and the "
            "exact private descriptor set, then exit before any Stage 1 fit. "
            "This flag is excluded from the immutable scientific request."
        ),
    )
    parser.add_argument("--evaluate-oracle-posthoc", action="store_true")
    parser.add_argument("--oracle-dataset", type=Path)
    parser.add_argument("--oracle-unit-id-column")
    parser.add_argument("--oracle-ite-column")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=PHASES,
        help="Operational pause boundary; excluded from scientific identity.",
    )
    parser.add_argument(
        "--adopt-checkpoint",
        action="append",
        default=[],
        type=Path,
        help=(
            "Complete portable artifact checkpoint, or a complete legacy "
            "input-preparation/embedding-cache terminal manifest, or an exact "
            "legacy cluster_preflight_manifest.json audit candidate; "
            "repeatable. Legacy preflight fitted state is never adopted."
        ),
    )
    parser.add_argument(
        "--trust-prior-adoption-attestation",
        action="append",
        default=[],
        type=Path,
        help=(
            "Explicitly reuse the producer artifact named by a prior v3 "
            "full-byte adoption attestation without rereading its payload "
            "bytes. Repeatable. This operator-trusted research mode validates "
            "controls and filesystem-stat continuity, records "
            "payload_bytes_reauthenticated=false, and cannot satisfy fresh "
            "full-byte or global-release certification."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help=(
            "Orchestrator lifecycle logging threshold; durable progress is "
            "always written and the setting is excluded from scientific "
            "identity."
        ),
    )
    parser.add_argument(
        "--validation-depth",
        choices=("standard", "full", "fresh_terminal_audit"),
        help=(
            "Requested operational validation minimum. Production acceptance "
            "always enforces and separately attests a fresh path-only terminal "
            "audit; excluded from scientific identity."
        ),
    )
    parser.add_argument(
        "--legacy-preflight-candidate",
        type=Path,
        help=(
            "Deprecated compatibility alias for one complete legacy clustered-"
            "preflight manifest. Prefer repeatable --adopt-checkpoint. The "
            "candidate is authenticated and accounted for before the "
            "configuration-derived physical/logical preflight is recomputed. "
            "Legacy fitted state is never adopted without a complete current "
            "dependency proof."
        ),
    )
    return parser


_DIRECT_SCIENTIFIC_SHIMS = (
    "unit_id_column",
    "text_column",
    "treatment_column",
    "outcome_column",
    "outcome_type",
    "clinical_question",
    "max_candidate_variables",
    "stage2_prompt_protocol",
    "post_extraction_causal_review",
    "complete_page_core_chars",
    "complete_page_context_chars",
    "complete_page_max_chars",
    "complete_reconciliation_fan_in",
    "embedding_chunk_size_words",
    "embedding_chunk_overlap_words",
    "embedding_max_chunks",
    "embedding_chunk_selection",
    "embedding_max_seq_length",
    "embedding_normalize",
    "forest_n_estimators",
    "forest_max_depth",
    "forest_min_samples_leaf",
    "forest_max_features",
    "forest_honest",
    "forest_inference",
    "forest_subforest_size",
    "forest_tune_model",
    "forest_nuisance_n_estimators",
    "forest_nuisance_max_depth",
    "forest_nuisance_min_samples_leaf",
    "forest_nuisance_treatment_max_features",
    "forest_nuisance_outcome_max_features",
    "forest_random_seed",
    "outer_folds",
    "review_rounds",
    "initial_training_partitions",
    "interaction_inner_folds",
    "tfidf_nested_calibration_folds",
    "seed",
    "stage1_seed_policy",
    "empty_text_policy",
    "repeated_character_policy",
    "repeated_character_threshold",
    "source_text_temporally_valid_by_design",
)

_DIRECT_RESOURCE_SAFETY_FIELDS = (
    "gpu_max_allocation_fraction",
    "gpu_minimum_headroom_bytes",
    "minimum_multi_device_throughput_ratio",
    "maximum_coordination_proof_overhead_ratio",
    "maximum_ordinary_read_amplification",
    "minimum_benchmark_repetitions_per_scope",
    "performance_read_counter_source",
    "fail_on_external_gpu_occupants",
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

_DIRECT_DEPLOYMENT_SHIMS = (
    "dataset",
    "stage1_profile",
    "query_profile",
    "embedding_model_name",
    "embedding_local_model_path",
    "htr_local_model_path",
    "embedding_batch_size",
    "devices",
    "stage1_device_count",
    "stage1_scope_workers_per_device",
    "stage1_persistent_slot_startup_timeout_seconds",
    "stage1_max_parallel_owners",
    "stage1_preflight_max_parallel_owners",
    "stage1_preflight_memory_budget_bytes",
    "stage1_preflight_estimated_owner_peak_bytes",
    "stage1_preflight_input_io_lane_cap",
    "stage1_preflight_publication_io_lane_cap",
    "stage1_preflight_authentication_io_lane_cap",
    "stage1_neural_query_topology",
    "stage1_htr_training_batch_size",
    "stage1_htr_sentence_encoder_batch_size",
    "stage1_htr_data_loader_workers",
    "stage1_htr_fold_parallelism",
    "stage1_htr_fold_parallel_backend",
    "stage1_htr_fold_slots_per_device",
    "stage1_htr_reuse_tokenizer_and_chunk_plans",
    "stage1_htr_chunk_plan_cache_max_entries",
    "stage1_htr_tokenized_chunk_cache_max_entries",
    "stage1_neural_query_inner_fold_parallelism",
    "stage1_neural_query_fold_parallel_backend",
    "stage1_neural_query_fold_slots_per_device",
    "stage1_neural_query_bank_parallelism",
    "stage1_neural_query_worker_cpu_threads",
    "tfidf_parallel_backend",
    "cpu_budget",
    "forest_operational",
    "response_concurrency",
    "storage_backend",
    "cluster_preflight_parquet_compression",
    "runtime_compatibility_class",
    "endpoint",
    "model",
    "stage2_tokenizer_locator",
    "oracle_dataset",
    "oracle_unit_id_column",
    "oracle_ite_column",
    *_DIRECT_RESOURCE_SAFETY_FIELDS,
)

_TYPED_DEPLOYMENT_OPERATIONAL_ROOT_OVERRIDES: Mapping[str, str] = {
    "work_root": "durable_artifact_root",
    "scratch_root": "scratch_root",
}

_UNSUPPORTED_LEGACY_EXECUTION_SHIMS = (
    "stage1_device",
    "query_device",
    "review_device",
    "stage1_gpu_id",
    "gpu_id",
    "stage1_scope_workers_per_gpu",
    "stage1_preflight_workers",
    "num_workers",
    "tfidf_workers",
)


def _specified_fields(
    values: Mapping[str, Any],
    names: Sequence[str],
) -> list[str]:
    return sorted(name for name in names if name in values)


def _run_control_from_namespace(values: Mapping[str, Any]) -> RunControl:
    defaults = RunControl()
    return RunControl(
        resume=values.get("resume", defaults.resume),
        stop_after=values.get("stop_after", defaults.stop_after),
        adopt_checkpoints=tuple(values.get("adopt_checkpoint", ())),
        trust_prior_adoption_attestations=tuple(
            values.get("trust_prior_adoption_attestation", ())
        ),
        log_level=values.get("log_level", defaults.log_level),
        validation_depth=values.get(
            "validation_depth",
            defaults.validation_depth,
        ),
        schema_version=defaults.schema_version,
    )


def _stage1_preflight_execution_attestation(
    deployment: DeploymentProfile,
) -> dict[str, Any]:
    """Compile one deployment-only, science-neutral preflight lane cap."""

    policy = (
        deployment.stage1_execution.preflight_execution_policy
    )
    read_amplification_lane_cap = math.floor(
        deployment.resource_performance_safety
        .maximum_ordinary_read_amplification
    )
    caps = {
        "cpu_budget": int(deployment.cpu_budget),
        "stage1_owner_cap": int(
            deployment.stage1_execution.max_parallel_owners
        ),
        "preflight_owner_cap": int(policy.max_parallel_owners),
        "memory_lane_cap": int(policy.memory_lane_cap),
        "input_io_lane_cap": int(policy.input_io_lane_cap),
        "publication_io_lane_cap": int(
            policy.publication_io_lane_cap
        ),
        "authentication_io_lane_cap": int(
            policy.authentication_io_lane_cap
        ),
        "ordinary_read_amplification_lane_cap": int(
            read_amplification_lane_cap
        ),
    }
    if any(value < 1 for value in caps.values()):
        raise ValueError(
            "Stage 1 preflight deployment policy leaves no executable lane"
        )
    effective = min(caps.values())
    body = {
        "schema_version": (
            "production_stage1_preflight_execution_attestation_v1"
        ),
        "policy": policy.as_dict(),
        "derived_caps": caps,
        "effective_preflight_owner_lanes_before_scope_cap": effective,
        "physical_owner_count_applied_by_preflight_executor": True,
        "resource_assignment_in_scientific_identity": False,
        "completion_order_in_scientific_identity": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _compile_direct_deployment_profile(
    values: Mapping[str, Any],
) -> DeploymentProfile:
    required = (
        "dataset",
        "work_root",
        "scratch_root",
        "stage1_profile",
        "query_profile",
        "embedding_model_name",
        "embedding_local_model_path",
        "htr_local_model_path",
        "embedding_batch_size",
        "devices",
        "stage1_device_count",
        "stage1_scope_workers_per_device",
        "stage1_persistent_slot_startup_timeout_seconds",
        "stage1_max_parallel_owners",
        "stage1_preflight_max_parallel_owners",
        "stage1_preflight_memory_budget_bytes",
        "stage1_preflight_estimated_owner_peak_bytes",
        "stage1_preflight_input_io_lane_cap",
        "stage1_preflight_publication_io_lane_cap",
        "stage1_preflight_authentication_io_lane_cap",
        "stage1_neural_query_topology",
        "stage1_htr_training_batch_size",
        "stage1_htr_sentence_encoder_batch_size",
        "stage1_htr_data_loader_workers",
        "stage1_htr_fold_parallelism",
        "stage1_htr_fold_parallel_backend",
        "stage1_htr_fold_slots_per_device",
        "stage1_htr_reuse_tokenizer_and_chunk_plans",
        "stage1_htr_chunk_plan_cache_max_entries",
        "stage1_htr_tokenized_chunk_cache_max_entries",
        "stage1_neural_query_inner_fold_parallelism",
        "stage1_neural_query_fold_parallel_backend",
        "stage1_neural_query_fold_slots_per_device",
        "stage1_neural_query_bank_parallelism",
        "stage1_neural_query_worker_cpu_threads",
        "tfidf_parallel_backend",
        "cpu_budget",
        "forest_operational",
        "response_concurrency",
        "storage_backend",
        "cluster_preflight_parquet_compression",
        "runtime_compatibility_class",
        *_DIRECT_RESOURCE_SAFETY_FIELDS,
    )
    missing = sorted(name for name in required if name not in values or values[name] is None)
    if missing:
        raise ValueError(
            "scientific-spec direct-deployment mode is missing explicit "
            f"DeploymentProfile fields: {missing}"
        )

    endpoint_fields = ("endpoint", "model", "stage2_tokenizer_locator")
    endpoint_present = _specified_fields(values, endpoint_fields)
    if endpoint_present and len(endpoint_present) != len(endpoint_fields):
        raise ValueError(
            "direct deployment requires endpoint, model, and " "stage2-tokenizer-locator together"
        )
    oracle_fields = (
        "oracle_dataset",
        "oracle_unit_id_column",
        "oracle_ite_column",
    )
    oracle_present = _specified_fields(values, oracle_fields)
    if oracle_present and len(oracle_present) != len(oracle_fields):
        raise ValueError(
            "direct deployment requires oracle dataset, unit-ID column, and " "ITE column together"
        )

    safety = ResourcePerformanceSafetyPolicy(
        gpu_max_allocation_fraction=values["gpu_max_allocation_fraction"],
        gpu_minimum_headroom_bytes=values["gpu_minimum_headroom_bytes"],
        minimum_multi_device_throughput_ratio=(values["minimum_multi_device_throughput_ratio"]),
        maximum_coordination_proof_overhead_ratio=(
            values["maximum_coordination_proof_overhead_ratio"]
        ),
        maximum_ordinary_read_amplification=(values["maximum_ordinary_read_amplification"]),
        minimum_benchmark_repetitions_per_scope=(values["minimum_benchmark_repetitions_per_scope"]),
        read_counter_source=values["performance_read_counter_source"],
        fail_on_external_gpu_occupants=(values["fail_on_external_gpu_occupants"]),
        hierarchical_job_cache_max_entry_bytes=(
            values["hierarchical_job_cache_max_entry_bytes"]
        ),
        first_untouched_gate_max_initial_spent_rows=(
            values["first_untouched_gate_max_initial_spent_rows"]
        ),
        first_untouched_gate_max_first_gate_rows=(
            values["first_untouched_gate_max_first_gate_rows"]
        ),
        first_untouched_gate_max_total_text_utf8_bytes=(
            values["first_untouched_gate_max_total_text_utf8_bytes"]
        ),
        first_untouched_gate_max_catalog_atoms=(
            values["first_untouched_gate_max_catalog_atoms"]
        ),
        first_untouched_gate_max_source_manifest_bytes=(
            values["first_untouched_gate_max_source_manifest_bytes"]
        ),
        first_untouched_gate_max_direct_numerical_signals=(
            values["first_untouched_gate_max_direct_numerical_signals"]
        ),
        first_untouched_gate_max_single_matrix_file_bytes=(
            values["first_untouched_gate_max_single_matrix_file_bytes"]
        ),
        first_untouched_gate_max_total_matrix_file_bytes=(
            values["first_untouched_gate_max_total_matrix_file_bytes"]
        ),
    )
    forest_operational = StrictCausalForestOperationalSpec.from_mapping(
        _read_json_object(
            Path(values["forest_operational"]),
            label="direct forest operational policy",
        )
    )
    direct_devices = normalize_device_policy(values["devices"])
    return DeploymentProfile(
        dataset_path=Path(values["dataset"]).resolve(),
        durable_artifact_root=Path(values["work_root"]).resolve(),
        scratch_root=Path(values["scratch_root"]).resolve(),
        embedding_model_locator=Path(values["embedding_local_model_path"]).resolve(),
        htr_model_locator=Path(values["htr_local_model_path"]).resolve(),
        stage1_profile_locator=Path(values["stage1_profile"]).resolve(),
        query_profile_locator=Path(values["query_profile"]).resolve(),
        embedding_batch_size=values["embedding_batch_size"],
        resource_performance_safety=safety,
        forest_operational=forest_operational,
        stage1_execution=Stage1ExecutionProfile(
            resource_kind=(
                "cpu" if direct_devices == ("cpu",) else "accelerator"
            ),
            device_count=values["stage1_device_count"],
            scope_workers_per_device=(
                values["stage1_scope_workers_per_device"]
            ),
            max_parallel_owners=values["stage1_max_parallel_owners"],
            preflight_execution_policy=(
                Stage1PreflightExecutionPolicy(
                    max_parallel_owners=values[
                        "stage1_preflight_max_parallel_owners"
                    ],
                    memory_budget_bytes=values[
                        "stage1_preflight_memory_budget_bytes"
                    ],
                    estimated_owner_peak_bytes=values[
                        "stage1_preflight_estimated_owner_peak_bytes"
                    ],
                    input_io_lane_cap=values[
                        "stage1_preflight_input_io_lane_cap"
                    ],
                    publication_io_lane_cap=values[
                        "stage1_preflight_publication_io_lane_cap"
                    ],
                    authentication_io_lane_cap=values[
                        "stage1_preflight_authentication_io_lane_cap"
                    ],
                )
            ),
            executor_mode="persistent_slots",
            persistent_slot_startup_timeout_seconds=values[
                "stage1_persistent_slot_startup_timeout_seconds"
            ],
            neural_query_topology=Stage1ExecutionTopologyPolicy(
                mode=values["stage1_neural_query_topology"],
            ),
            htr_operational_controls=(
                RoleNeutralHTROperationalControls(
                    training_batch_size=values[
                        "stage1_htr_training_batch_size"
                    ],
                    sentence_encoder_batch_size=values[
                        "stage1_htr_sentence_encoder_batch_size"
                    ],
                    data_loader_workers=values[
                        "stage1_htr_data_loader_workers"
                    ],
                    fold_parallelism=values[
                        "stage1_htr_fold_parallelism"
                    ],
                    fold_parallel_backend=values[
                        "stage1_htr_fold_parallel_backend"
                    ],
                    fold_slots_per_device=values[
                        "stage1_htr_fold_slots_per_device"
                    ],
                    reuse_tokenizer_and_chunk_plans=values[
                        "stage1_htr_reuse_tokenizer_and_chunk_plans"
                    ],
                    chunk_plan_cache_max_entries=values[
                        "stage1_htr_chunk_plan_cache_max_entries"
                    ],
                    tokenized_chunk_cache_max_entries=values[
                        "stage1_htr_tokenized_chunk_cache_max_entries"
                    ],
                )
            ),
            neural_query_operational_controls=(
                RoleNeutralNeuralQueryOperationalControls(
                    inner_fold_parallelism=values[
                        "stage1_neural_query_inner_fold_parallelism"
                    ],
                    fold_parallel_backend=values[
                        "stage1_neural_query_fold_parallel_backend"
                    ],
                    fold_slots_per_device=values[
                        "stage1_neural_query_fold_slots_per_device"
                    ],
                    bank_parallelism=values[
                        "stage1_neural_query_bank_parallelism"
                    ],
                    worker_cpu_threads=values[
                        "stage1_neural_query_worker_cpu_threads"
                    ],
                    schema_version=(
                        ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA
                    ),
                )
            ),
            tfidf_parallel_backend=values["tfidf_parallel_backend"],
            selection_method="operator_configured",
            benchmark_evidence_kind="none",
            selected_candidate=None,
            benchmark_result_sha256=None,
            benchmark_result_locator=None,
            benchmark_workload_deployment_sha256=None,
            benchmark_workload_deployment_locator=None,
            benchmark_publication_sha256=None,
            benchmark_publication_locator=None,
        ),
        embedding_model_name=values["embedding_model_name"],
        endpoint=values.get("endpoint"),
        endpoint_model=values.get("model"),
        stage2_tokenizer_locator=(
            None
            if "stage2_tokenizer_locator" not in values
            else Path(values["stage2_tokenizer_locator"]).resolve()
        ),
        devices=direct_devices,
        cpu_budget=values["cpu_budget"],
        response_concurrency=values["response_concurrency"],
        storage_backend=values["storage_backend"],
        cluster_preflight_parquet_compression=(
            values["cluster_preflight_parquet_compression"]
        ),
        oracle_source=(
            None if "oracle_dataset" not in values else Path(values["oracle_dataset"]).resolve()
        ),
        oracle_unit_id_column=values.get("oracle_unit_id_column"),
        oracle_ite_column=values.get("oracle_ite_column"),
        runtime_compatibility_class=values["runtime_compatibility_class"],
    )


def _apply_typed_deployment_operational_root_overrides(
    *,
    values: Mapping[str, Any],
    deployment: DeploymentProfile,
) -> DeploymentProfile:
    """Return a copied profile with only its run-local roots replaced.

    Dataset/model/profile locators and every resource or scientific setting
    remain owned by the typed deployment profile.  These two roots are
    operational locators: resolving them at the CLI boundary changes the
    immutable run request and resume target, but not scientific compatibility.
    """

    replacements: dict[str, Path] = {}
    for argument_name, profile_field in (
        _TYPED_DEPLOYMENT_OPERATIONAL_ROOT_OVERRIDES.items()
    ):
        if argument_name in values:
            replacements[profile_field] = Path(
                values[argument_name]
            ).resolve()
    if not replacements:
        return deployment
    return replace(deployment, **replacements)


def _compile_production_options(
    *,
    values: Mapping[str, Any],
    scientific: ScientificWorkflowSpec,
    deployment: DeploymentProfile,
    scientific_path: Path,
    deployment_path: Path | None,
    run_control: RunControl,
) -> ProductionAllEvidenceWorkflowOptions:
    if deployment.embedding_model_name is None:
        raise ValueError(
            "deployment must explicitly configure embedding_model_name; "
            "model locator basenames are not scientific model identities"
        )
    def deployment_locator(path: Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute() and deployment_path is not None:
            candidate = deployment_path.parent / candidate
        return candidate.resolve()

    from .portable_resource_scheduler import plan_resources

    policy = normalize_device_policy(deployment.devices)
    resource_plan = plan_resources(
        policy=policy,
        cpu_budget=deployment.cpu_budget,
        requested_device_count=deployment.stage1_execution.device_count,
        cpu_supported=(deployment.stage1_execution.resource_kind == "cpu"),
        resource_performance_safety=(deployment.resource_performance_safety),
    )
    selected_devices = tuple(resource_plan.devices)
    if (
        deployment.stage1_execution.resource_kind == "cpu"
        and selected_devices != ("cpu",)
    ) or (
        deployment.stage1_execution.resource_kind == "accelerator"
        and any(value == "cpu" for value in selected_devices)
    ):
        raise RuntimeError(
            "resolved Stage 1 resources differ from the deployment resource_kind"
        )
    resolved_owner_capacity = (
        deployment.stage1_execution.neural_query_topology
        .effective_parallel_owners(
            devices=selected_devices,
            workers_per_device=(
                deployment.stage1_execution.scope_workers_per_device
            ),
        )
    )
    if (
        deployment.stage1_execution.max_parallel_owners
        > deployment.cpu_budget
        or deployment.stage1_execution.max_parallel_owners
        > resolved_owner_capacity
    ):
        raise ValueError(
            "Stage 1 effective owner concurrency exceeds the resolved "
            "resource topology or host CPU budget"
        )
    learned_query_profile = scientific.architecture_profiles.get(
        "learned_neural_queries"
    )
    learned_query_configuration = (
        learned_query_profile.get("producer_configuration")
        if isinstance(learned_query_profile, Mapping)
        else None
    )
    learned_query_scientific = (
        learned_query_configuration.get("query_config")
        if isinstance(learned_query_configuration, Mapping)
        else None
    )
    query_inner_folds = (
        learned_query_scientific.get("query_inner_folds")
        if isinstance(learned_query_scientific, Mapping)
        else None
    )
    neural_controls = (
        deployment.stage1_execution.neural_query_operational_controls
    )
    if (
        isinstance(query_inner_folds, bool)
        or not isinstance(query_inner_folds, int)
        or query_inner_folds < 2
        or neural_controls.inner_fold_parallelism > query_inner_folds
        or neural_controls.bank_parallelism > 3
    ):
        raise ValueError(
            "Stage 1 neural-query concurrency exceeds the configured "
            "scientific inner-fold or three-bank task count"
        )
    # Bind the repeated optimizer batch to the authenticated scientific HTR
    # profile before any output root is created.
    htr_profile = scientific.architecture_profiles.get(
        "hierarchical_transformer"
    )
    htr_producer_configuration = (
        htr_profile.get("producer_configuration")
        if isinstance(htr_profile, Mapping)
        else None
    )
    htr_training_batch = (
        htr_producer_configuration.get("batch_size")
        if isinstance(htr_producer_configuration, Mapping)
        else None
    )
    if (
        isinstance(htr_training_batch, bool)
        or not isinstance(htr_training_batch, int)
        or htr_training_batch < 1
        or deployment.stage1_execution.htr_operational_controls
        .training_batch_size
        != htr_training_batch
    ):
        raise ValueError(
            "Stage 1 HTR operational training batch differs from the "
            "authenticated scientific optimizer batch"
        )
    first_device = selected_devices[0]
    gpu_ids = tuple(
        int(value.split(":", 1)[1]) for value in selected_devices if value.startswith("cuda:")
    )
    oracle_configured = deployment.oracle_source is not None
    forest_runtime_config = compile_strict_causal_forest_runtime(
        scientific=scientific,
        deployment=deployment,
    )
    options = ProductionAllEvidenceWorkflowOptions(
        dataset_path=deployment_locator(deployment.dataset_path),
        work_root=deployment_locator(deployment.durable_artifact_root),
        stage1_profile_path=deployment_locator(deployment.stage1_profile_locator),
        query_profile_path=deployment_locator(deployment.query_profile_locator),
        unit_id_column=scientific.columns.unit_id,
        text_column=scientific.columns.text,
        treatment_column=scientific.columns.treatment,
        outcome_column=scientific.columns.outcome,
        outcome_type="binary",
        clinical_question=scientific.clinical_question,
        embedding_model_name=deployment.embedding_model_name,
        embedding_local_model_path=deployment_locator(deployment.embedding_model_locator),
        htr_local_model_path=deployment_locator(deployment.htr_model_locator),
        resource_performance_safety=(deployment.resource_performance_safety),
        run_control=run_control,
        endpoint=deployment.endpoint,
        model_name=deployment.endpoint_model,
        stage2_tokenizer_locator=(
            None
            if deployment.stage2_tokenizer_locator is None
            else deployment_locator(deployment.stage2_tokenizer_locator)
        ),
        outer_folds=scientific.folds.outer_folds,
        review_rounds=scientific.folds.review_rounds,
        initial_training_partitions=(scientific.folds.initial_training_partitions),
        interaction_inner_folds=(scientific.folds.interaction_inner_folds),
        tfidf_nested_calibration_folds=(scientific.folds.tfidf_nested_calibration_folds),
        stage1_device=first_device,
        query_device=None,
        query_devices=selected_devices,
        review_device=first_device,
        gpu_id=None,
        stage1_gpu_ids=gpu_ids,
        stage1_execution_device_count=(
            deployment.stage1_execution.device_count
        ),
        stage1_scope_workers_per_gpu=(
            deployment.stage1_execution.scope_workers_per_device
        ),
        stage1_execution_profile=deployment.stage1_execution,
        stage1_preflight_workers=int(
            _stage1_preflight_execution_attestation(deployment)[
                "effective_preflight_owner_lanes_before_scope_cap"
            ]
        ),
        stage1_preflight_execution_attestation=(
            _stage1_preflight_execution_attestation(deployment)
        ),
        stage1_seed_policy=scientific.seed_policy,
        num_workers=deployment.cpu_budget,
        tfidf_workers=deployment.cpu_budget,
        tfidf_parallel_backend=(
            deployment.stage1_execution.tfidf_parallel_backend
        ),
        seed=scientific.seed,
        empty_text_policy=scientific.preprocessing.empty_text_policy,
        repeated_character_policy=(scientific.preprocessing.repeated_character_policy),
        repeated_character_threshold=(scientific.preprocessing.repeated_character_threshold),
        source_text_temporally_valid_by_design=(
            scientific.preprocessing.source_text_temporally_valid_by_design
        ),
        evaluate_oracle_posthoc=(
            bool(values.get("evaluate_oracle_posthoc", False)) or oracle_configured
        ),
        oracle_dataset_path=(
            None
            if deployment.oracle_source is None
            else deployment_locator(deployment.oracle_source)
        ),
        oracle_unit_id_column=deployment.oracle_unit_id_column,
        oracle_ite_column=deployment.oracle_ite_column,
        embedding_cache_import=values.get("embedding_cache_import"),
        embedding_cache_import_source_prepared_path=values.get(
            "embedding_cache_import_source_prepared_path"
        ),
        embedding_cache_import_source_preparation_manifest_path=values.get(
            "embedding_cache_import_source_preparation_manifest_path"
        ),
        source_snapshot_root=values.get("source_snapshot_root"),
        stage1_only=bool(values.get("stage1_only", False)),
        scratch_root=deployment_locator(deployment.scratch_root),
        device_policy=policy,
        cpu_budget=deployment.cpu_budget,
        response_concurrency=deployment.response_concurrency,
        storage_backend=deployment.storage_backend,
        cluster_preflight_parquet_compression=(
            deployment.cluster_preflight_parquet_compression
        ),
        runtime_compatibility_class=(deployment.runtime_compatibility_class),
        legacy_preflight_candidate=values.get("legacy_preflight_candidate"),
        portable_scientific_spec=scientific.identity_payload(),
        scientific_spec_path=scientific_path,
        deployment_profile_path=deployment_path,
        forest_runtime_config=forest_runtime_config,
        max_candidate_variables=scientific.max_candidate_variables,
        stage2_prompt_protocol=scientific.stage2_prompt_protocol,
        post_extraction_causal_review=(scientific.post_extraction_causal_review),
        complete_page_core_chars=(scientific.text_windows.complete_page_core_chars),
        complete_page_context_chars=(scientific.text_windows.complete_page_context_chars),
        complete_page_max_chars=(scientific.text_windows.complete_page_max_chars),
        complete_reconciliation_fan_in=(scientific.text_windows.reconciliation_fan_in),
        embedding_chunk_size_words=(scientific.text_windows.embedding_chunk_size_words),
        embedding_chunk_overlap_words=(scientific.text_windows.embedding_chunk_overlap_words),
        embedding_max_chunks=(scientific.text_windows.embedding_max_chunks),
        embedding_chunk_selection=(scientific.text_windows.embedding_chunk_selection),
        embedding_max_seq_length=(scientific.text_windows.embedding_max_seq_length),
        embedding_normalize=(scientific.text_windows.embedding_normalize),
        embedding_encoder=(scientific.text_windows.embedding_encoder),
        embedding_batch_size=deployment.embedding_batch_size,
    )
    # Validate every public option before hashing inputs or creating work_root.
    ProductionAllEvidenceWorkflow(options)
    return options


def options_from_args(
    args: argparse.Namespace,
) -> ProductionAllEvidenceWorkflowOptions:
    """Compile the public CLI into the three typed configuration layers."""

    values = vars(args).copy()
    values.pop("prepare_stage1_canary_descriptors_only", None)
    scientific_value = values.pop("scientific_spec", None)
    deployment_value = values.pop("deployment_profile", None)
    if scientific_value is None:
        raise ValueError(
            "--scientific-spec is required; direct flags cannot define all "
            "ten architecture profiles, prompt identities, estimand, and "
            "result-changing text-window settings without hidden defaults"
        )

    scientific_path = Path(scientific_value).resolve(strict=True)
    scientific = ScientificWorkflowSpec.from_json(scientific_path)
    run_control = _run_control_from_namespace(values)

    scientific_conflicts = _specified_fields(
        values,
        _DIRECT_SCIENTIFIC_SHIMS,
    )
    if scientific_conflicts:
        raise ValueError(
            "--scientific-spec cannot be mixed with direct scientific "
            f"shims: {scientific_conflicts}"
        )

    legacy_conflicts = _specified_fields(
        values,
        _UNSUPPORTED_LEGACY_EXECUTION_SHIMS,
    )
    if legacy_conflicts:
        raise ValueError(
            "typed public workflow forbids legacy device/worker shims; use "
            f"DeploymentProfile devices and CPU budget: {legacy_conflicts}"
        )

    if deployment_value is not None:
        deployment_conflicts = _specified_fields(
            values,
            _DIRECT_DEPLOYMENT_SHIMS,
        )
        if deployment_conflicts:
            raise ValueError(
                "--deployment-profile cannot be mixed with direct deployment "
                f"shims: {deployment_conflicts}"
            )
        deployment_path = Path(deployment_value).resolve(strict=True)
        deployment = DeploymentProfile.from_json(deployment_path)
        deployment = _apply_typed_deployment_operational_root_overrides(
            values=values,
            deployment=deployment,
        )
    else:
        deployment_path = None
        deployment = _compile_direct_deployment_profile(values)

    return _compile_production_options(
        values=values,
        scientific=scientific,
        deployment=deployment,
        scientific_path=scientific_path,
        deployment_path=deployment_path,
        run_control=run_control,
    )


def _reexec_from_source_snapshot(
    *,
    parsed_args: argparse.Namespace,
    raw_argv: Sequence[str],
) -> None:
    """Replace the current process so all subsequent imports use the snapshot."""

    snapshot_root = getattr(parsed_args, "source_snapshot_root", None)
    if snapshot_root is None:
        return
    typed_scientific_path = getattr(parsed_args, "scientific_spec", None)
    if typed_scientific_path is None:
        raise ValueError("--scientific-spec is required before source-snapshot execution")
    from .production_source_snapshot import validate_production_source_snapshot

    snapshot = validate_production_source_snapshot(snapshot_root)
    loaded_root = Path(__file__).resolve().parents[2]
    marker = os.environ.get(SOURCE_SNAPSHOT_EXECUTION_ENV)
    requested_hash_seed = ScientificWorkflowSpec.from_json(typed_scientific_path).seed
    if requested_hash_seed < 0:
        raise ValueError("source-snapshot execution requires a nonnegative seed")
    expected_hash_seed = str(requested_hash_seed)
    if marker is not None:
        if (
            marker != snapshot.content_sha256
            or loaded_root != snapshot.root
            or os.environ.get("PYTHONHASHSEED") != expected_hash_seed
        ):
            raise RuntimeError(
                "source-snapshot execution marker, loaded source tree, or "
                "PYTHONHASHSEED does not match the requested run"
            )
        return
    entrypoint = snapshot.root / "scripts" / "run_production_all_evidence_workflow.py"
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise FileNotFoundError("source snapshot lacks the production workflow entry point")
    environment = os.environ.copy()
    environment[SOURCE_SNAPSHOT_EXECUTION_ENV] = snapshot.content_sha256
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(snapshot.root)
    # Python reads this setting only while starting the interpreter.  Binding
    # it to the configured global seed before exec makes parent-side hashing
    # deterministic as well as the per-scope child seeds enforced later by
    # the Stage 1 scheduler.
    environment["PYTHONHASHSEED"] = expected_hash_seed
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-P",
            "-u",
            str(entrypoint),
            *[str(value) for value in raw_argv],
        ],
        environment,
    )


def _default_portable_role_neutral_hooks(
    options: ProductionAllEvidenceWorkflowOptions,
) -> ProductionAllEvidenceWorkflowHooks:
    """Bind the concrete local role-neutral path for typed public runs.

    Explicitly injected hooks remain authoritative in tests and embedding
    deployments. The public CLI always supplies a typed scientific request;
    programmatic historical options remain outside this public binding.
    """

    scientific = options.portable_scientific_spec
    if scientific is None:
        return ProductionAllEvidenceWorkflowHooks()
    if not isinstance(scientific, Mapping):
        raise TypeError("typed portable workflow requires its scientific identity mapping")
    architecture_profiles = scientific.get("architecture_profiles")
    if not isinstance(architecture_profiles, Mapping):
        raise ValueError("typed portable workflow lacks configured architecture profiles")
    if set(architecture_profiles) != set(EVIDENCE_FAMILIES):
        missing = sorted(set(EVIDENCE_FAMILIES) - set(architecture_profiles))
        extra = sorted(set(architecture_profiles) - set(EVIDENCE_FAMILIES))
        raise ValueError(
            "producer factory architecture profiles differ from all ten; "
            f"missing={missing}, extra={extra}"
        )
    stage2_protocol = scientific.get("stage2_prompt_protocol")
    if not isinstance(stage2_protocol, Mapping):
        raise ValueError("typed portable workflow lacks its Stage 2 prompt protocol")
    semantic_member_batch_size = stage2_protocol.get(
        "hierarchical_max_semantic_member_ids_per_chunk"
    )
    embedding_profile = architecture_profiles.get("whole_cohort_embeddings")
    embedding_configuration = (
        embedding_profile.get("producer_configuration")
        if isinstance(embedding_profile, Mapping)
        else None
    )
    if (
        isinstance(semantic_member_batch_size, bool)
        or not isinstance(semantic_member_batch_size, int)
        or semantic_member_batch_size < 1
        or not isinstance(embedding_configuration, Mapping)
        or embedding_configuration.get("semantic_member_batch_size") != semantic_member_batch_size
    ):
        raise ValueError(
            "the embedding semantic_member_batch_size must be an explicit "
            "positive integer identical to the configured hierarchy semantic "
            "member bound"
        )
    cluster_profile = architecture_profiles.get("cluster_local_embeddings")
    cluster_configuration = (
        cluster_profile.get("producer_configuration")
        if isinstance(cluster_profile, Mapping)
        else None
    )
    ClusterLocalEmbeddingScientificConfig.from_mapping(cluster_configuration)
    from .production_role_neutral_producer_factories import (
        PreparedBuildRoleNeutralProducerFactoriesBuilder,
    )
    from .production_role_neutral_stage2_handoff import (
        ReferenceOnlyRoleNeutralStage1HandoffPublisher,
    )
    from .production_role_neutral_persistent_executor import (
        PersistentSpawnRoleNeutralPhysicalOwnerExecutor,
    )
    from .production_role_neutral_process_executor import (
        ProcessIsolatedRoleNeutralPhysicalOwnerExecutor,
    )

    execution_profile = options.stage1_execution_profile
    if execution_profile is None:
        raise ValueError(
            "typed portable workflow requires one Stage 1 execution profile"
        )
    if execution_profile.executor_mode == "persistent_slots":
        physical_executor = PersistentSpawnRoleNeutralPhysicalOwnerExecutor(
            max_workers_per_resource=(
                options.stage1_scope_workers_per_gpu
            ),
            startup_timeout_seconds=(
                execution_profile.persistent_slot_startup_timeout_seconds
            ),
        )
    elif execution_profile.executor_mode == "fresh_per_fit":
        physical_executor = ProcessIsolatedRoleNeutralPhysicalOwnerExecutor(
            max_workers_per_resource=(
                options.stage1_scope_workers_per_gpu
            ),
        )
    else:  # The typed profile rejects this before hook construction.
        raise ValueError("unsupported typed Stage 1 executor mode")

    integration = ProductionRoleNeutralStage1Integration(
        producer_factories_builder=(
            PreparedBuildRoleNeutralProducerFactoriesBuilder(
                architecture_profiles=architecture_profiles,
                runtime_compatibility_class=(options.runtime_compatibility_class),
            )
        ),
        executor=physical_executor,
        handoff_publisher=(
            ReferenceOnlyRoleNeutralStage1HandoffPublisher(
                semantic_member_batch_size=semantic_member_batch_size,
            )
        ),
        producer_factories_scientific_identity={
            "schema_version": (
                "prepared_role_neutral_factories_scientific_identity_v1"
            ),
            "architecture_profiles": copy.deepcopy(
                dict(architecture_profiles)
            ),
            "runtime_compatibility_class": (
                options.runtime_compatibility_class
            ),
        },
        physical_owner_executor_scientific_identity={
            "schema_version": (
                "role_neutral_executor_scientific_identity_v1"
            ),
            "executor_mode": execution_profile.executor_mode,
            "worker_target": str(physical_executor.worker_target),
            "production_worker_required": bool(
                physical_executor.production_worker_required
            ),
            "worker_lifecycle_mode": getattr(
                physical_executor,
                "worker_lifecycle_mode",
                "fresh_process_per_physical_fit_v1",
            ),
            "process_isolated_physical_owners": bool(
                physical_executor.process_isolated_physical_owners
            ),
            "operational_state_fields_excluded": [
                "max_workers_per_resource",
                "poll_interval_seconds",
                "startup_timeout_seconds",
                "worker_parameters",
            ],
        },
        handoff_publisher_scientific_identity={
            "schema_version": (
                "role_neutral_handoff_publisher_scientific_identity_v2"
            ),
            "semantic_member_batch_size": semantic_member_batch_size,
            "htr_catalog_representation": (
                "authenticated_semantic_aggregates_with_complete_reverse_index_v2"
            ),
            "raw_htr_token_arrays_model_facing": False,
        },
    )
    return ProductionAllEvidenceWorkflowHooks(
        role_neutral_stage1=integration,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    try:
        _reexec_from_source_snapshot(parsed_args=args, raw_argv=raw_argv)
        options = options_from_args(args)
        _configure_cli_logging(options.run_control.log_level)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        hooks = _default_portable_role_neutral_hooks(options)
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    workflow = ProductionAllEvidenceWorkflow(
        options,
        hooks=hooks,
    )
    if bool(getattr(args, "prepare_stage1_canary_descriptors_only", False)):
        result = workflow.prepare_stage1_canary_descriptors_only()
    else:
        result = workflow.run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "EMBEDDING_CACHE_PHASE_SCHEMA",
    "PHASES",
    "PORTABLE_ROLE_NEUTRAL_STAGE1_HANDOFF_BINDING_SCHEMA",
    "PORTABLE_ROLE_NEUTRAL_STAGE1_PHASE_SCHEMA",
    "STAGE1_PREFLIGHT_PHASE_SCHEMA",
    "STAGE1_ONLY_PHASES",
    "ProductionAllEvidenceWorkflow",
    "ProductionAllEvidenceWorkflowHooks",
    "ProductionAllEvidenceWorkflowOptions",
    "ProductionRoleNeutralStage1Integration",
    "RoleNeutralStage1HandoffPublication",
    "RoleNeutralStage1HandoffPublisher",
    "RoleNeutralStage1ProducerFactoriesBuilder",
    "WorkflowPhaseHook",
    "build_parser",
    "main",
    "options_from_args",
    "validate_stage1_canary_descriptor_preparation",
    "validate_completed_workflow_prefix",
]
