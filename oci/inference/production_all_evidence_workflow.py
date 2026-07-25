"""Resumable public orchestration for the all-evidence causal workflow."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..config import ClusterLocalEmbeddingScientificConfig
from .production_authenticated_tree_cache import (
    AUTHENTICATED_DIRECTORY_TREE_POLICY,
    authenticate_directory_tree,
)
from .production_stage1_bundle import (
    ProductionStage1BundleBuilder,
    Stage1BundleBuildOptions,
)
from .production_text_preparation import (
    TextPreparationOptions,
    prepare_modeling_cohort,
    stable_file_sha256,
)
from .portable_artifacts import (
    ArtifactCompatibility,
    MANIFEST_NAME,
    ValidatedPortableArtifact,
    adopt_checkpoint,
    assert_validated_artifact_unchanged,
    materialize_portable_phase,
    publish_portable_reference_artifact,
    validate_checkpoint_adoption,
    validate_portable_artifact,
)
from .performance_telemetry import TelemetryLedger
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
EMBEDDING_CACHE_PHASE_SCHEMA = "production_embedding_cache_phase_result_v1"
STAGE1_PREFLIGHT_PHASE_SCHEMA = "production_stage1_preflight_phase_result_v2"
PORTABLE_ROLE_NEUTRAL_STAGE1_PHASE_SCHEMA = (
    "production_portable_role_neutral_stage1_phase_result_v1"
)
PORTABLE_ROLE_NEUTRAL_STAGE1_HANDOFF_BINDING_SCHEMA = (
    "production_portable_role_neutral_stage1_handoff_binding_v1"
)
WORKFLOW_PROGRESS_SCHEMA = "production_all_evidence_workflow_progress_v1"
WORKFLOW_PHASE_MANIFEST_SCHEMA = "production_workflow_phase_manifest_v2"
WORKFLOW_ADOPTED_PHASE_MANIFEST_SCHEMA = "production_workflow_adopted_phase_manifest_v1"
WORKFLOW_CHECKPOINT_PUBLICATION_ATTESTATION_SCHEMA = (
    "production_workflow_checkpoint_publication_attestation_v1"
)
WORKFLOW_CHECKPOINT_DAG_VALIDATION_SCHEMA = "production_workflow_checkpoint_dag_validation_v1"
WORKFLOW_LEGACY_PREFLIGHT_DECISION_SCHEMA = (
    "production_workflow_legacy_preflight_recompute_decision_v1"
)
WORKFLOW_TERMINAL_VALIDATION_SCHEMA = "production_all_evidence_fresh_terminal_validation_v1"
SOURCE_SNAPSHOT_EXECUTION_ENV = "OCI_PRODUCTION_SOURCE_SNAPSHOT_SHA256"

ADOPTABLE_PHASE_BY_ARTIFACT_KIND = {
    "prepared_cohort": "input_preparation",
    "embedding_cache": "embedding_cache",
    "clustered_preflight": "stage1_preflight",
    "stage1_handoff": "stage1_modeling",
    "stage2_canary": "stage2_canary",
    "frozen_prediction": "stage2_inference",
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


def _validate_adopted_checkpoint_graph(
    artifacts: Sequence[ValidatedPortableArtifact],
    *,
    allowed_phases: Sequence[str],
) -> Mapping[str, str]:
    """Validate a closed portable DAG and return phase-to-artifact bindings."""

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

        bundle_sha256 = self.bundle_sha256
        if bundle_sha256 is None:
            manifest = _read_json_object(
                Path(self.bundle_manifest_path),
                label="role-neutral handoff manifest",
            )
            bundle_sha256 = str(manifest.get("bundle_sha256") or "")
        body = {
            "schema_version": ("production_role_neutral_stage1_handoff_publication_v1"),
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

    def __post_init__(self) -> None:
        if not callable(self.producer_factories_builder):
            raise TypeError("role-neutral integration requires a producer-factories builder")
        if not callable(getattr(self.executor, "execute", None)):
            raise TypeError(
                "role-neutral integration requires a configured physical-owner " "executor"
            )
        if not callable(self.handoff_publisher):
            raise TypeError("role-neutral integration requires a Stage 2 handoff publisher")


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
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{label} contains duplicate JSON key: {key}")
            output[key] = value
        return output

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one real regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


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
            _fsync_directory(directory)
        os.replace(temporary, published)
        _fsync_directory(durable_root)
        shutil.rmtree(source)
        _fsync_directory(source.parent)
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
    if artifact is None:
        artifact = validate_portable_artifact(
            Path(str(locator)),
            expected_kind=str(value["artifact_kind"]),
            expected_compatibility_key=str(value["compatibility_key"]),
            expected_upstream_artifact_ids=tuple(value["upstream_artifact_ids"]),
        )
    else:
        assert_validated_artifact_unchanged(artifact)
    if artifact.artifact_id != artifact_id:
        raise ValueError("adopted phase artifact ID changed")
    if dict(artifact.manifest["compatibility"]) != dict(
        request.get("expected_checkpoint_compatibility") or {}
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
    validate_checkpoint_adoption(
        attestation_path=expected_attestation,
        artifact=artifact,
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
            "fresh_full_byte_validation": authenticated_adoptions is None,
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

    Imported embedding caches never load the embedding model after its first
    full provenance authentication. For that one lifecycle, callers may reuse
    a PID-scoped content identity while every logical check still compares the
    complete filesystem inventory. Fresh cache builds and live HTR models
    always retain full byte-tree reauthentication.
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


def _revalidate_request_bound_external_inputs(
    request: Mapping[str, Any],
    *,
    authenticated_adoptions: Mapping[str, ValidatedPortableArtifact] | None = None,
) -> None:
    """Reopen every external input whose bytes were bound into the run request."""

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
        observed, _size = stable_file_sha256(Path(raw_path).resolve(strict=True))
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
    raw_execution_profile = request.get("stage1_execution_profile")
    if (
        isinstance(raw_execution_profile, Mapping)
        and raw_execution_profile.get("selection_method")
        == "measured_role_neutral_benchmark_v1"
    ):
        scientific_spec_path = request.get("scientific_spec_path")
        raw_safety = request.get("resource_performance_safety")
        if (
            not isinstance(scientific_spec_path, str)
            or not scientific_spec_path
            or not isinstance(raw_safety, Mapping)
        ):
            raise ValueError(
                "benchmark-selected request lacks scientific or safety authority"
            )
        from .role_neutral_benchmark_deployment_selection import (
            validate_benchmarked_stage1_execution_profile,
        )

        validate_benchmarked_stage1_execution_profile(
            profile=Stage1ExecutionProfile.from_mapping(
                raw_execution_profile
            ),
            scientific_spec_path=Path(scientific_spec_path),
            resource_performance_safety=(
                ResourcePerformanceSafetyPolicy.from_mapping(raw_safety)
            ),
            cpu_budget=int(request["cpu_budget"]),
        )
    adoption_records = request.get("requested_checkpoint_adoptions") or []
    adoption_locators = request.get("checkpoint_adoption_locators") or []
    if (
        not isinstance(adoption_records, list)
        or not isinstance(adoption_locators, list)
        or len(adoption_records) != len(adoption_locators)
    ):
        raise ValueError("immutable workflow request has invalid checkpoint adoptions")
    for locator, expected in zip(adoption_locators, adoption_records):
        if not isinstance(expected, Mapping):
            raise ValueError("checkpoint adoption request is invalid")
        artifact_id = str(expected.get("artifact_id", ""))
        artifact = (
            None if authenticated_adoptions is None else authenticated_adoptions.get(artifact_id)
        )
        if artifact is None:
            artifact = validate_portable_artifact(Path(str(locator)))
        else:
            assert_validated_artifact_unchanged(artifact)
        if (
            artifact.artifact_id != expected.get("artifact_id")
            or artifact.manifest["artifact_kind"] != expected.get("artifact_kind")
            or artifact.compatibility_key != expected.get("compatibility_key")
            or list(artifact.manifest["upstream_artifact_ids"])
            != expected.get("upstream_artifact_ids")
        ):
            raise RuntimeError("adopted checkpoint changed after workflow initialization")

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
        observed_sha, observed_size = stable_file_sha256(manifest_path)
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
        from .legacy_checkpoint_migration import validate_legacy_preflight_manifest

        path = Path(str(legacy_preflight_identity.get("manifest_path", ""))).resolve(strict=True)
        observed_sha, observed_size = stable_file_sha256(path)
        observed = validate_legacy_preflight_manifest(
            path,
            authenticate_registered_payload_bytes=False,
        )
        if (
            observed_sha != legacy_preflight_identity.get("manifest_sha256")
            or observed_size != legacy_preflight_identity.get("manifest_size_bytes")
            or observed["manifest"]["content_sha256"]
            != legacy_preflight_identity.get("manifest_content_sha256")
        ):
            raise RuntimeError("legacy preflight candidate changed after workflow initialization")

    cache_inputs = request.get("embedding_cache_import_inputs")
    imported_embedding_cache = cache_inputs is not None
    expected_model_policy = (
        AUTHENTICATED_DIRECTORY_TREE_POLICY
        if imported_embedding_cache
        else "full_byte_tree_reauthentication_v1"
    )
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
        if _stable_path_identity(
            Path(str(expected["path"])),
            reuse_process_authenticated_tree=(
                field == "embedding_model_tree" and imported_embedding_cache
            ),
        ) != dict(expected):
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
    for raw_path, expected_sha in implementation_files.items():
        observed_sha, _size = stable_file_sha256(Path(str(raw_path)).resolve(strict=True))
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
            if source_file is None:
                continue
            if not isinstance(source_file, Mapping) or not isinstance(source_file.get("path"), str):
                raise ValueError(f"immutable workflow request has invalid {collection_name} source")
            observed_sha, observed_size = stable_file_sha256(
                Path(str(source_file["path"])).resolve(strict=True)
            )
            if observed_sha != source_file.get("sha256") or observed_size != int(
                source_file.get("size_bytes", -1)
            ):
                raise RuntimeError(f"{collection_name} implementation changed after initialization")

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
        "selected_logical_gpu_id",
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
        or manifest.get("schema_version") != "production_stage1_canary_descriptor_preparation_v1"
        or manifest.get("status") != "complete"
        or manifest.get("content_sha256") != _sha(body)
        or manifest.get("workflow_request_sha256") != request_sha
        or manifest.get("source_snapshot") != request.get("source_snapshot")
        or manifest.get("completed_workflow_prefix") != list(prefix)
        or manifest.get("selected_scope_kind") != "full_outer"
        or manifest.get("selected_logical_gpu_id") != 0
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
    if (
        len(descriptor_set.descriptors) != expected_count
        or manifest.get("descriptor_count") != expected_count
        or descriptor_set.manifest.get("content_sha256")
        != manifest.get("descriptor_set_content_sha256")
        or selected is None
        or selected.manifest_path != selected_manifest
        or selected.scope.scope_kind != "full_outer"
        or int(selected.assignment.gpu_id) != 0
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
    phases = tuple(str(value) for value in expected_phases)
    configured_sequence = STAGE1_ONLY_PHASES if request.get("stage1_only") is True else PHASES
    if (
        not phases
        or len(phases) != len(set(phases))
        or phases != configured_sequence[: len(phases)]
    ):
        raise ValueError("checkpoint DAG phases must be an ordered workflow prefix")
    compatibility = ArtifactCompatibility(
        **dict(request.get("expected_checkpoint_compatibility") or {})
    )
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
        expected_upstream = tuple(
            artifacts_by_phase[parent].artifact_id for parent in spec["upstream_phases"]
        )
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
            artifact = validate_portable_artifact(
                Path(str(matching[0][1])),
                expected_kind=str(spec["artifact_kind"]),
                expected_compatibility_key=compatibility.key,
                expected_upstream_artifact_ids=expected_upstream,
            )
        else:
            assert isinstance(rows, list) and rows
            control_root = root / "portable_checkpoints" / phase
            artifact = validate_portable_artifact(
                control_root,
                expected_kind=str(spec["artifact_kind"]),
                expected_compatibility_key=compatibility.key,
                expected_upstream_artifact_ids=expected_upstream,
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
        "fresh_full_byte_validation": True,
    }
    return {**body, "content_sha256": _sha(body)}


def _hook_identity(hook: WorkflowPhaseHook | None) -> Mapping[str, Any] | None:
    if hook is None:
        return None
    target = hook if inspect.isfunction(hook) else hook.__call__
    source = inspect.getsourcefile(target)
    identity: dict[str, Any] = {
        "module": str(getattr(target, "__module__", type(hook).__module__)),
        "qualname": str(getattr(target, "__qualname__", type(hook).__qualname__)),
    }
    if source is not None and Path(source).is_file():
        digest, size = stable_file_sha256(Path(source).resolve())
        identity["source_file"] = {
            "path": str(Path(source).resolve()),
            "sha256": digest,
            "size_bytes": size,
        }
    return identity


def _scientific_callable_identity(
    value: Any,
    *,
    method_name: str | None = None,
) -> Mapping[str, Any]:
    """Return a path-neutral source identity for injected scientific code."""

    target = getattr(value, method_name) if method_name is not None else value
    if not callable(target):
        raise TypeError("scientific integration capability is not callable")
    identity: dict[str, Any] = {
        "module": str(getattr(target, "__module__", type(value).__module__)),
        "qualname": str(getattr(target, "__qualname__", type(value).__qualname__)),
    }
    try:
        source = inspect.getsourcefile(target)
    except (TypeError, OSError):
        source = None
    if source is None:
        raise ValueError(
            "portable role-neutral integration callables must have an " "inspectable source file"
        )
    resolved = Path(source).resolve(strict=True)
    digest, size = stable_file_sha256(resolved)
    identity["source_sha256"] = digest
    identity["source_size_bytes"] = size
    return identity


def _role_neutral_stage1_integration_identity(
    integration: ProductionRoleNeutralStage1Integration | None,
) -> Mapping[str, Any] | None:
    if integration is None:
        return None
    body = {
        "schema_version": ("production_role_neutral_stage1_integration_code_identity_v1"),
        "producer_factories_builder": _scientific_callable_identity(
            integration.producer_factories_builder
        ),
        "physical_owner_executor": _scientific_callable_identity(
            integration.executor,
            method_name="execute",
        ),
        "stage2_handoff_publisher": _scientific_callable_identity(integration.handoff_publisher),
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
        or result.get("productive_compute_canary_completed") is not True
        or result.get("selected_canary_replica_adopted_as_production") is not True
        or result.get("compute_canary_scientific_equality") is not True
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
        self._adopted_artifact_handles: dict[str, ValidatedPortableArtifact] = {}
        self._published_checkpoint_handles: dict[str, ValidatedPortableArtifact] = {}
        self._phase_payload_stat_inventories: dict[str, Mapping[str, tuple[int, ...]]] = {}
        self._validate_options()
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
        expected_compatibility: Mapping[str, Any],
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

        compatibility = ArtifactCompatibility(**dict(expected_compatibility))
        portable: list[ValidatedPortableArtifact] = []
        legacy: dict[str, tuple[Path, Mapping[str, Any]]] = {}
        selected_preflight: tuple[Path, Mapping[str, Any], str] | None = None
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
                validated_preflight = validate_legacy_preflight_manifest(
                    source,
                    authenticate_registered_payload_bytes=False,
                )
                selected_preflight = (
                    source.resolve(strict=True),
                    validated_preflight,
                    "adopt_checkpoint",
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
            portable.append(validate_portable_artifact(source))

        if self.options.legacy_preflight_candidate is not None:
            if selected_preflight is not None:
                raise ValueError(
                    "legacy preflight candidate cannot be selected through "
                    "both --adopt-checkpoint and --legacy-preflight-candidate"
                )
            alias_path = Path(self.options.legacy_preflight_candidate).resolve(strict=True)
            selected_preflight = (
                alias_path,
                validate_legacy_preflight_manifest(
                    alias_path,
                    authenticate_registered_payload_bytes=False,
                ),
                "deprecated_legacy_preflight_candidate_alias",
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
            manifest_path, validated_preflight, selection_source = selected_preflight
            manifest_sha256, manifest_size = stable_file_sha256(manifest_path)
            preflight_identity = {
                "selection_source": selection_source,
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
                "direct_reuse_allowed": False,
            }
        return portable, migration_records, preflight_identity

    def _request_body(self) -> dict[str, Any]:
        self._adopted_artifact_handles.clear()
        values = json.loads(json.dumps(asdict(self.options), default=str))
        values.pop("run_control")
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
            "tfidf_workers": self.options.tfidf_workers,
            "tfidf_parallel_backend": self.options.tfidf_parallel_backend,
            "seed": self.options.seed,
            "scope_seed_policy": self.options.stage1_seed_policy,
            "exclusive_gpu_preflight_required": bool(self.stage1_gpu_ids),
        }
        values["source_sha256"] = stable_file_sha256(self.options.dataset_path)[0]
        values["stage1_profile_sha256"] = stable_file_sha256(self.options.stage1_profile_path)[0]
        values["query_profile_sha256"] = stable_file_sha256(self.options.query_profile_path)[0]
        values["stage1_profile_scientific_identity"] = scientific_profile_file_identity(
            self.options.stage1_profile_path,
            profile_kind="stage1",
        )
        values["query_profile_scientific_identity"] = scientific_profile_file_identity(
            self.options.query_profile_path,
            profile_kind="neural_query",
        )
        if self.options.scientific_spec_path is not None:
            values["scientific_spec_source_sha256"] = stable_file_sha256(
                self.options.scientific_spec_path
            )[0]
        if self.options.deployment_profile_path is not None:
            values["deployment_profile_source_sha256"] = stable_file_sha256(
                self.options.deployment_profile_path
            )[0]
        imported_embedding_cache = self.options.embedding_cache_import is not None
        values["embedding_model_revalidation_policy"] = (
            AUTHENTICATED_DIRECTORY_TREE_POLICY
            if imported_embedding_cache
            else "full_byte_tree_reauthentication_v1"
        )
        values["embedding_model_tree"] = _stable_path_identity(
            self.options.embedding_local_model_path,
            reuse_process_authenticated_tree=imported_embedding_cache,
        )
        values["embedding_model_builder_tree_sha256"] = _embedding_builder_tree_sha256(
            root=self.options.embedding_local_model_path,
            workflow_tree_identity=values["embedding_model_tree"],
        )
        values["htr_model_tree"] = _stable_path_identity(self.options.htr_local_model_path)
        if self.options.stage2_tokenizer_locator is not None:
            values["stage2_tokenizer_tree"] = _stable_path_identity(
                self.options.stage2_tokenizer_locator
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
            "embedding_cache": _hook_identity(self.hooks.embedding_cache),
            "stage1_preflight": _hook_identity(self.hooks.stage1_preflight),
            "stage1_modeling": _hook_identity(self.hooks.stage1_modeling),
            "role_neutral_stage1": _role_neutral_stage1_integration_identity(
                self.hooks.role_neutral_stage1
            ),
        }
        values["phase_overrides"] = {
            phase: _hook_identity(self.phase_overrides.get(phase))
            for phase in self._phase_sequence()
        }
        implementation_files = (
            Path(__file__).resolve(),
            Path(__file__).with_name("production_text_preparation.py").resolve(),
            Path(__file__).with_name("production_oracle_evaluation.py").resolve(),
            Path(__file__).with_name("production_authenticated_tree_cache.py").resolve(),
            Path(__file__).with_name("production_embedding_cache_builder.py").resolve(),
            Path(__file__).with_name("production_embedding_cache_process.py").resolve(),
            Path(__file__).with_name("production_embedding_cache_relocation.py").resolve(),
            Path(__file__).parents[1] / "extraction" / "complete_paged.py",
            Path(__file__).with_name("production_source_snapshot.py").resolve(),
            Path(__file__).with_name("production_stage1_cluster_preflight_artifact.py").resolve(),
            Path(__file__).with_name(
                "production_stage1_cluster_preflight_artifact_v2.py"
            ).resolve(),
            Path(__file__).with_name("production_stage1_bundle.py").resolve(),
            Path(__file__).with_name("production_stage1_scope_scheduler.py").resolve(),
            Path(__file__).with_name("production_stage1_role_neutral_execution.py").resolve(),
            Path(__file__).with_name("production_stage1_role_neutral_coordinator.py").resolve(),
            Path(__file__).with_name(
                "production_role_neutral_process_executor.py"
            ).resolve(),
            Path(__file__).with_name(
                "production_role_neutral_persistent_executor.py"
            ).resolve(),
            Path(__file__).with_name(
                "production_role_neutral_producer_factories.py"
            ).resolve(),
            Path(__file__).with_name("prepared_stage1_context.py").resolve(),
            Path(__file__).with_name("role_neutral_all_ten_binding.py").resolve(),
            Path(__file__).with_name("role_neutral_bow_group_execution.py").resolve(),
            Path(__file__).with_name("role_neutral_htr_group_execution.py").resolve(),
            Path(__file__).with_name("role_neutral_matched_pair_group_execution.py").resolve(),
            Path(__file__).with_name("role_neutral_embedding_group_execution.py").resolve(),
            Path(__file__).with_name("role_neutral_tfidf_group_execution.py").resolve(),
            Path(__file__).with_name("role_neutral_neural_query_group_execution.py").resolve(),
            Path(__file__).with_name("production_neural_query_binary_layout.py").resolve(),
            Path(__file__).with_name("production_role_neutral_stage2_handoff.py").resolve(),
            Path(__file__).with_name("direct_upstream_numerical_reference_bank.py").resolve(),
            Path(__file__).with_name("tfidf_safe_artifacts.py").resolve(),
            Path(__file__).with_name("production_stage1_legacy_scope_adapter.py").resolve(),
            Path(__file__).with_name("production_stage1_legacy_scope_fragments.py").resolve(),
            Path(__file__).with_name("portable_artifacts.py").resolve(),
            Path(__file__).with_name("portable_workflow_spec.py").resolve(),
            Path(__file__).with_name("physical_fit_deduplication.py").resolve(),
            Path(__file__).with_name("legacy_checkpoint_migration.py").resolve(),
            Path(__file__).with_name("portable_resource_scheduler.py").resolve(),
            Path(__file__).with_name("scoped_embedding_cache.py").resolve(),
            Path(__file__).with_name("performance_telemetry.py").resolve(),
            Path(__file__).with_name("scientific_profile_identity.py").resolve(),
            Path(__file__).with_name("production_stage1_hierarchy_one_shot.py").resolve(),
            Path(__file__).with_name("production_stage1_hierarchy_handoff.py").resolve(),
            Path(__file__).with_name("production_stage1_hierarchy_contract.py").resolve(),
            Path(__file__).with_name("hierarchical_all_architecture_discovery.py").resolve(),
            Path(__file__).with_name("hierarchical_discovery_response_contract.py").resolve(),
            Path(__file__).with_name("openai_compatible_json_discovery_job_runner.py").resolve(),
            Path(__file__).with_name("adaptive_hierarchical_stage1_reconsideration.py").resolve(),
            Path(__file__).with_name("stage2_prompt_nontruncation.py").resolve(),
            Path(__file__).with_name("all_evidence_post_extraction_review.py").resolve(),
            Path(__file__).with_name("production_terminal_artifact_validation.py").resolve(),
            Path(__file__).with_name("all_evidence_fusion_runner.py").resolve(),
            Path(__file__).with_name("final_context_fit_causal_forest_adapter.py").resolve(),
            Path(__file__).parents[1] / "models" / "causal_forest_head.py",
            Path(__file__).parents[1] / "models" / "strict_causal_forest_runtime.py",
            Path(__file__).parents[1] / "models" / "lossless_tokenization.py",
            Path(__file__).parents[1] / "config.py",
            Path(__file__).parents[2] / "scripts" / "run_production_all_evidence_workflow.py",
            Path(__file__).parents[2] / "scripts" / "canary_production_stage1_hierarchy.py",
        )
        values["implementation_files"] = {
            str(path.resolve()): stable_file_sha256(path.resolve())[0]
            for path in implementation_files
        }
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
        producer_code_inputs: dict[str, Any] = {
            "implementation_file_sha256": sorted(values["implementation_files"].values())
        }
        role_neutral_integration_identity = values["integration_hooks"].get("role_neutral_stage1")
        if role_neutral_integration_identity is not None:
            producer_code_inputs["role_neutral_stage1_integration"] = (
                role_neutral_integration_identity
            )
        producer_code_identity = identity_sha256(producer_code_inputs)
        scientific_body = {
            "schema_version": "portable_all_evidence_scientific_identity_v2",
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
            "producer_code_identity": producer_code_identity,
            "source_snapshot_content_sha256": (
                values.get("source_snapshot", {}).get("content_sha256")
                if isinstance(values.get("source_snapshot"), Mapping)
                else None
            ),
            "runtime_compatibility_class": (self.options.runtime_compatibility_class),
        }
        values["scientific_identity"] = {
            **scientific_body,
            "scientific_sha256": identity_sha256(scientific_body),
        }
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
        expected_checkpoint_compatibility = {
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
            "configuration_identity": values["scientific_identity"]["scientific_sha256"],
            "seed_identity": seed_identity,
            "producer_code_identity": producer_code_identity,
            "runtime_compatibility_class": (self.options.runtime_compatibility_class),
        }
        values["expected_checkpoint_compatibility"] = expected_checkpoint_compatibility
        (
            validated_adoptions,
            legacy_migration_records,
            legacy_preflight_identity,
        ) = self._resolve_requested_checkpoint_sources(
            expected_compatibility=expected_checkpoint_compatibility,
            embedding_model_builder_tree_sha256=values["embedding_model_builder_tree_sha256"],
        )
        if legacy_preflight_identity is not None:
            values["legacy_preflight_candidate_identity"] = legacy_preflight_identity
        for artifact in validated_adoptions:
            compatibility = artifact.manifest["compatibility"]
            observed_compatibility = {
                key: compatibility.get(key) for key in expected_checkpoint_compatibility
            }
            if observed_compatibility != expected_checkpoint_compatibility:
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
        )
        artifact_phases = {artifact_id: phase for phase, artifact_id in phase_artifact_ids.items()}
        adopted: list[dict[str, Any]] = []
        adoption_locators: list[str] = []
        for artifact in sorted(
            validated_adoptions,
            key=lambda value: value.artifact_id,
        ):
            substituted_phase = artifact_phases.get(artifact.artifact_id)
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
                }
            )
            self._adopted_artifact_handles[artifact.artifact_id] = artifact
            adoption_locators.append(str(artifact.root))
        values["requested_checkpoint_adoptions"] = adopted
        values["checkpoint_adoption_locators"] = adoption_locators
        values["legacy_checkpoint_migration_sources"] = legacy_migration_records
        return values

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

    def _checkpoint_compatibility(self) -> ArtifactCompatibility:
        raw = self.request.get("expected_checkpoint_compatibility")
        if not isinstance(raw, Mapping):
            raise RuntimeError("immutable request lacks checkpoint compatibility")
        return ArtifactCompatibility(**dict(raw))

    def _checkpoint_control_root(self, phase: str) -> Path:
        return self.options.work_root / "portable_checkpoints" / phase

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
        return tuple(
            self._checkpoint_artifact_for_phase(
                upstream_phase,
                required=True,
            ).artifact_id
            for upstream_phase in upstream_phases
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
        adopted = self._adopted_checkpoint_handle_for_phase(phase)
        if adopted is not None:
            if (
                adopted.manifest.get("artifact_kind") != spec["artifact_kind"]
                or adopted.compatibility_key != self._checkpoint_compatibility().key
                or tuple(adopted.manifest.get("upstream_artifact_ids") or ()) != upstream_ids
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
                or cached.compatibility_key != self._checkpoint_compatibility().key
                or tuple(cached.manifest.get("upstream_artifact_ids") or ()) != upstream_ids
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
            expected_compatibility_key=self._checkpoint_compatibility().key,
            expected_upstream_artifact_ids=upstream_ids,
        )
        authenticated_payload_bytes = sum(row.size_bytes for row in artifact.payloads)
        self.telemetry.count_bytes(
            read=authenticated_payload_bytes,
            hashed=authenticated_payload_bytes,
        )
        if (
            artifact.manifest.get("artifact_schema") != spec["artifact_schema"]
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
            compatibility=self._checkpoint_compatibility(),
            upstream_artifact_ids=upstream_ids,
            payload_paths=tuple(payload_paths),
            expected_payload_identities=expected_payload_identities,
            process_authenticated_stat_inventory=(self._phase_payload_stat_inventories.get(phase)),
            workflow_phase=phase,
            workflow_phase_result=result,
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

    def _phase_manifest(self, phase: str) -> Path:
        return self.options.work_root / "phases" / phase / "complete_manifest.json"

    def _validated_complete(self, phase: str) -> Mapping[str, Any] | None:
        path = self._phase_manifest(phase)
        if not path.is_file():
            return None
        return _validate_phase_manifest_from_paths(
            work_root=self.options.work_root.resolve(strict=True),
            phase=phase,
            request_sha256=self.request["request_sha256"],
            authenticated_adoptions=self._adopted_artifact_handles,
        )

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
        self._phase_payload_stat_inventories[phase] = process_authenticated_stats
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
        resources: dict[int, dict[str, Any]] = {}
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
            resources[int(fields[0])] = {
                "uuid": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "utilization_percent": int(fields[4]),
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
        fresh_prepared, fresh_manifest = self._input_preparation_paths()
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
    ) -> Stage1BundleBuildOptions:
        values: dict[str, Any] = {
            "dataset_path": dataset,
            "config_path": profile,
            "embedding_cache_dir": cache,
            "embedding_local_model_path": None,
            "embedding_cache_output_dir": None,
            "embedding_cache_configuration": copy.deepcopy(
                dict(self._embedding_chunk_configuration())
            ),
            "output_dir": output,
            "unit_id_column": self.options.unit_id_column,
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
            values["embedding_cache_relocation"] = self._embedding_cache_relocation_options(
                cache=cache,
                prepared=dataset,
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
            "portable_cluster_preflight_v2": (
                self.options.portable_scientific_spec is not None
            ),
            "scope_seed_policy": self.options.stage1_seed_policy,
            "cluster_preflight_manifest_path": cluster_preflight_manifest_path,
            "cluster_preflight_state_bundle_manifest_path": (
                cluster_preflight_state_bundle_manifest_path
            ),
            "stage1_scope_attempt_root": (
                self.options.work_root / "recovery" / "stage1_scope_attempts"
            ).resolve(),
            "stage1_scope_progress_path": (
                self.options.work_root / "recovery" / "stage1_scope_progress.json"
            ).resolve(),
        }
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
            or checkpoint_validation.get("fresh_full_byte_validation") is not True
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
            DISTINCT_RESOURCE_CANARY_REPLICA_POLICY,
            EARLIEST_CANONICAL_OWNER_CANARY_SELECTION,
            ROLE_NEUTRAL_EXECUTION_MANIFEST,
            RoleNeutralComputeCanaryPolicy,
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
        max_parallel_owners = min(
            int(resource_plan.cpu_budget),
            len(resource_plan.devices) * int(self.options.stage1_scope_workers_per_gpu),
        )
        execution_policy = RoleNeutralStage1ExecutionPolicy(
            resource_plan=resource_plan,
            max_parallel_owners=max_parallel_owners,
            compute_canary=RoleNeutralComputeCanaryPolicy(
                canonical_scope_selection=(EARLIEST_CANONICAL_OWNER_CANARY_SELECTION),
                replica_resource_selection=(DISTINCT_RESOURCE_CANARY_REPLICA_POLICY),
            ),
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

        execution_root = (attempt / "role_neutral_stage1_execution").resolve()
        execution_manifest = execute_and_publish_role_neutral_stage1(
            root=execution_root,
            plan=plan,
            producer_factories=producer_factories,
            policy=execution_policy,
            executor=execution_executor,
        )
        if (
            int(execution_manifest.get("physical_fit_count", -1)) != len(plan.physical_scopes)
            or int(execution_manifest.get("logical_scope_count", -1)) != len(plan.scopes)
            or execution_manifest.get("legacy_bundle_build_invoked") is not False
            or execution_manifest.get("every_physical_owner_executed_once") is not True
            or execution_manifest.get("every_component_executed_and_authenticated_once_per_owner")
            is not True
            or execution_manifest.get("productive_compute_canary_completed") is not True
            or execution_manifest.get("selected_canary_replica_adopted_as_production") is not True
            or execution_manifest.get("compute_canary_scientific_equality") is not True
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
            "productive_compute_canary_completed": True,
            "selected_canary_replica_adopted_as_production": True,
            "compute_canary_scientific_equality": True,
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
            "productive_compute_canary_completed": True,
            "selected_canary_replica_adopted_as_production": True,
            "compute_canary_scientific_equality": True,
            "all_ten_families_bound_per_logical_context": True,
            "legacy_bundle_build_invoked": False,
            "stage2_handoff_derived_exclusively_from_role_neutral_execution": (True),
            "resource_preflight": resource_plan.execution_attestation(),
            "terminal_files": [
                str(execution_manifest_path),
                str(binding_path.resolve()),
                str(bundle_manifest_path),
                str(numerical_bank_manifest_path),
                str(numerical_bank_locator_path),
            ],
        }
        return _validate_portable_role_neutral_stage1_phase_result(result)

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
            if legacy_preflight_identity is not None:
                if not isinstance(legacy_preflight_identity, Mapping):
                    raise RuntimeError("immutable legacy preflight identity is invalid")
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
                    self.request["expected_checkpoint_compatibility"]["producer_code_identity"]
                )
                logical_contexts = tuple(
                    LogicalContext(
                        canonical_index=int(scope.canonical_index),
                        scope_id=scope.scope_id,
                        purpose=scope.scope_kind,
                        outer_fold=int(scope.outer_fold),
                        fit_row_ids=tuple(str(row_id) for row_id in scope.fit_row_ids),
                        heldout_row_ids=tuple(str(row_id) for row_id in scope.heldout_row_ids),
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
                    source_candidate_identity=legacy_preflight_identity,
                    migration=migration,
                    expected_logical_scope_count=expected_logical,
                    expected_physical_fit_count=expected_physical,
                )
            scope_input_identity = prepared_build.cluster_preflight_scope_input_set_identity
            if not isinstance(scope_input_identity, Mapping):
                raise RuntimeError(
                    "Stage 1 preflight omitted its recoverable private " "scope-input identity"
                )
            if prepared_build.options.portable_cluster_preflight_v2:
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
                        artifact.manifest_path
                    ),
                    cluster_preflight_state_bundle_manifest_path=(
                        state_bundle_manifest
                    ),
                )
                reusable_prepared = replace(
                    prepared_build,
                    options=reusable_options,
                    cluster_preflight_manifest_path=(
                        artifact.manifest_path
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
                "cluster_preflight_manifest_path": str(artifact.manifest_path),
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
                    "accepted_portable_compact_lossless_v2"
                    if prepared_build.options.portable_cluster_preflight_v2
                    else "accepted_and_independently_sealed_v1"
                ),
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
                        for path in sorted(artifact.root.rglob("*"))
                        if path.is_file()
                    ],
                    *[str(path) for path in sorted(state_bundle.root.rglob("*")) if path.is_file()],
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
                    raise RuntimeError("direct Stage 2 result omitted fold terminal artifacts")
                direct_terminals = [
                    str(Path(str(result["runner_input_manifest_path"])).resolve(strict=True)),
                    str(Path(str(result["hierarchical_batch_result_path"])).resolve(strict=True)),
                    str(Path(str(result["prepared_cohort_path"])).resolve(strict=True)),
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
                        for path in required_lists["complete_paged_ledger_artifact_paths"]
                    ],
                    str(prediction.resolve(strict=True)),
                    str(manifest.resolve(strict=True)),
                    str(attestation),
                ]
                if len(direct_terminals) != len(set(direct_terminals)):
                    raise RuntimeError("direct Stage 2 terminal artifact inventory is duplicated")
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
            selected = next(
                (
                    descriptor
                    for descriptor in descriptor_set.descriptors.values()
                    if descriptor.scope.scope_kind == "full_outer"
                    and int(descriptor.assignment.gpu_id) == 0
                ),
                None,
            )
            if selected is None:
                raise RuntimeError(
                    "the canonical descriptor set has no full-outer logical cuda:0 scope"
                )
            descriptor_set_manifest = (
                descriptor_set.root / LEGACY_STAGE1_SCOPE_DESCRIPTOR_SET_MANIFEST
            )
            preflight_phase_manifest = self._phase_manifest("stage1_preflight")
            body = {
                "schema_version": ("production_stage1_canary_descriptor_preparation_v1"),
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
                "selected_logical_gpu_id": int(selected.assignment.gpu_id),
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
        self._write_progress(
            status="complete",
            completed=tuple(completed),
            current_phase=None,
        )
        return completed["terminal_validation"]["result"]


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
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Operational logging threshold; excluded from scientific identity.",
    )
    parser.add_argument(
        "--validation-depth",
        choices=("standard", "full", "fresh_terminal_audit"),
        help="Operational validation depth; excluded from scientific identity.",
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
)

_DIRECT_DEPLOYMENT_SHIMS = (
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
    "tfidf_parallel_backend",
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
        log_level=values.get("log_level", defaults.log_level),
        validation_depth=values.get(
            "validation_depth",
            defaults.validation_depth,
        ),
        schema_version=defaults.schema_version,
    )


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
            executor_mode="persistent_slots",
            selection_method="operator_configured",
            selected_candidate=None,
            benchmark_result_sha256=None,
            benchmark_result_locator=None,
            benchmark_workload_deployment_sha256=None,
            benchmark_workload_deployment_locator=None,
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
    if (
        deployment.stage1_execution.selection_method
        == "measured_role_neutral_benchmark_v1"
    ):
        from .role_neutral_benchmark_deployment_selection import (
            validate_benchmarked_stage1_execution_profile,
        )

        validate_benchmarked_stage1_execution_profile(
            profile=deployment.stage1_execution,
            scientific_spec_path=scientific_path,
            resource_performance_safety=(
                deployment.resource_performance_safety
            ),
            cpu_budget=deployment.cpu_budget,
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
        stage1_preflight_workers=deployment.cpu_budget,
        stage1_seed_policy=scientific.seed_policy,
        num_workers=deployment.cpu_budget,
        tfidf_workers=deployment.cpu_budget,
        tfidf_parallel_backend="processes",
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
