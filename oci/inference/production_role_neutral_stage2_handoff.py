"""Authenticated, non-materializing Stage 1-to-Stage 2 bridge.

The role-neutral Stage 1 execution is the sole scientific authority accepted
here.  Validation reopens the execution, its coordination gate, every retained
component tree, and the complete physical/logical binding tree before
constructing a typed in-memory bridge.

This module intentionally does *not* manufacture the historical Stage 1 bundle
layout.  Its positive reference-only publisher writes only the split registry,
scope plan, row map, a path-neutral scientific manifest, and a separate locator
attestation.  The direct loader reconstructs the plan and opens the original
authenticated all-ten producer trees through
``AuthenticatedRoleNeutralStage2Provider``.  Evidence is neither copied nor
recomputed.

The older fail-closed publisher remains as an explicit compatibility diagnostic
for callers that still request conversion into the legacy hierarchy layout.
No cohort size, fold count, note length, page size, context overlap, device
count, or other dataset/deployment hyperparameter is fixed in this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .all_evidence_fusion import FoldEvidenceProvenance
from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
)
from .lossless_stage1_evidence_catalog import (
    NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
    RoleNeutralEvidenceCatalog,
    assemble_cumulative_spent_role_neutral_catalog,
)
from .htr_attention_evidence_schema import (
    ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA,
    ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_PACKAGE_SCHEMA,
)
from .htr_stage2_complete_semantic_aggregation import (
    HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA,
    HTR_STAGE2_SCOPE_MANIFEST_SCHEMA,
    HTR_STAGE2_STORE_MANIFEST_SCHEMA,
    HtrSemanticAggregationResult,
    build_htr_semantic_aggregation_scope,
    summarize_htr_call_plan,
    validate_htr_semantic_aggregation_scope,
)
from .portable_workflow_spec import EVIDENCE_FAMILIES
from .production_stage1_role_neutral_coordinator import (
    ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION,
    ROLE_NEUTRAL_COORDINATION_MANIFEST,
    ROLE_NEUTRAL_SCIENTIFIC_BINDING_DIRECTORY,
    validate_role_neutral_stage1_coordination_gate,
)
from .production_stage1_role_neutral_execution import (
    ROLE_NEUTRAL_COORDINATION_DIRECTORY,
    ROLE_NEUTRAL_EXECUTION_MANIFEST,
    validate_role_neutral_stage1_execution,
)
from .production_stage1_scope_scheduler import (
    Stage1PhysicalFitIdentity,
    Stage1ScopePlan,
    validate_stage1_scope_plan,
)
from .production_stage1_legacy_scope_fragments import (
    ROLE_NEUTRAL_FIT_ONLY_FAMILY_PRIOR_AUTH_REFERENCE_SCHEMA,
    ROLE_NEUTRAL_FIT_ONLY_FAMILY_SEAL_REFERENCE_SCHEMA,
    ROLE_NEUTRAL_FIT_ONLY_FAMILY_SEAL_REFERENCE_SCHEMAS,
)
from .role_neutral_all_ten_binding import (
    EXPECTED_COMPONENT_FAMILIES,
    PORTABLE_TO_NATIVE_FAMILY,
    validate_complete_role_neutral_stage1_bindings,
)

ROLE_NEUTRAL_STAGE2_BRIDGE_SCHEMA = "authenticated_role_neutral_stage1_to_stage2_bridge_v4"
ROLE_NEUTRAL_STAGE2_LOADER_REQUIREMENTS_SCHEMA = "role_neutral_stage2_direct_loader_requirements_v4"
ROLE_NEUTRAL_STAGE2_COMPONENT_EXPORT_INDEX_SCHEMA = "role_neutral_stage2_component_export_index_v2"
ROLE_NEUTRAL_STAGE2_DIRECT_HANDOFF_SCHEMA = "role_neutral_stage2_direct_handoff_v4"
ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_PROOF_SCHEMA = "role_neutral_stage2_fit_projection_proof_v1"
ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD = "stage2_fit_projection_proof"
ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_SCHEMA = "production_role_neutral_stage1_reference_handoff_v5"
ROLE_NEUTRAL_STAGE1_REFERENCE_LOCATOR_SCHEMA = (
    "production_role_neutral_stage1_reference_locator_attestation_v1"
)
ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND = "authenticated_role_neutral_all_ten_reference_only_v1"
ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST = "bundle_manifest.json"
ROLE_NEUTRAL_STAGE1_REFERENCE_LOCATOR = "locator_attestation.json"
ROLE_NEUTRAL_STAGE1_REFERENCE_REGISTRY = "split_registry.json"
ROLE_NEUTRAL_STAGE1_REFERENCE_PLAN = "stage1_scope_plan.json"
ROLE_NEUTRAL_STAGE1_REFERENCE_ROW_MAP = "row_registry.parquet"
ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_DIRECTORY = "htr_semantic_aggregation"
ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_MANIFEST = "store_manifest.json"

_HEX = frozenset("0123456789abcdef")
_PREPARED_PROJECTION_BINDING_ISSUER = object()
_DIRECT_RUNTIME_BINDING_ISSUER = object()
_DIRECT_HIERARCHY_AUTHORIZATION_ISSUER = object()


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


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _require_semantic_member_batch_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "semantic_member_batch_size must be an explicitly configured "
            "positive integer"
        )
    return int(value)


@dataclass(frozen=True)
class AuthenticatedPreparedCohortProjectionBinding:
    """Raw-value-free proof that one prepared cohort matches every Stage 1 fit."""

    plan_scientific_content_sha256: str
    prepared_request_sha256: str
    source_execution_content_sha256: str
    provider_identity_sha256: str
    prepared_cohort_artifact_sha256: str
    row_map_sha256: str
    row_count: int
    unit_id_column: str
    text_column: str
    treatment_column: str
    outcome_column: str
    physical_owner_projection_proofs: tuple[tuple[str, str], ...]
    content_sha256: str
    _issuer: object

    def __post_init__(self) -> None:
        if self._issuer is not _PREPARED_PROJECTION_BINDING_ISSUER:
            raise TypeError(
                "prepared-cohort projection bindings are issued only by the "
                "authenticated role-neutral provider"
            )

    def as_dict(self) -> Mapping[str, Any]:
        body = {
            "schema_version": (
                "authenticated_role_neutral_prepared_cohort_projection_binding_v1"
            ),
            "plan_scientific_content_sha256": self.plan_scientific_content_sha256,
            "prepared_request_sha256": self.prepared_request_sha256,
            "source_execution_content_sha256": self.source_execution_content_sha256,
            "provider_identity_sha256": self.provider_identity_sha256,
            "prepared_cohort_artifact_sha256": (
                self.prepared_cohort_artifact_sha256
            ),
            "row_map_sha256": self.row_map_sha256,
            "row_count": int(self.row_count),
            "unit_id_column": self.unit_id_column,
            "text_column": self.text_column,
            "treatment_column": self.treatment_column,
            "outcome_column": self.outcome_column,
            "physical_owner_projection_proofs": [
                {
                    "physical_owner_scope_id": scope_id,
                    "projection_proof_content_sha256": proof_sha256,
                }
                for scope_id, proof_sha256 in self.physical_owner_projection_proofs
            ],
            "all_physical_fit_projections_verified": True,
            "raw_text_persisted": False,
            "raw_treatment_persisted": False,
            "raw_outcome_persisted": False,
            "text_truncation_applied": False,
        }
        if _sha256_json(body) != self.content_sha256:
            raise RuntimeError("prepared-cohort projection binding changed")
        return {**body, "content_sha256": self.content_sha256}


def validate_authenticated_prepared_projection_binding(
    value: Any,
    *,
    expected_plan_scientific_content_sha256: str,
    expected_source_execution_content_sha256: str,
) -> Mapping[str, Any]:
    """Validate the exact provider-issued binding at a downstream boundary."""

    if type(value) is not AuthenticatedPreparedCohortProjectionBinding:
        raise TypeError(
            "direct numerical consumption requires the exact authenticated "
            "prepared-cohort projection binding"
        )
    assert isinstance(value, AuthenticatedPreparedCohortProjectionBinding)
    payload = value.as_dict()
    if (
        payload["plan_scientific_content_sha256"]
        != _require_sha256(
            expected_plan_scientific_content_sha256,
            label="expected prepared projection plan",
        )
        or payload["source_execution_content_sha256"]
        != _require_sha256(
            expected_source_execution_content_sha256,
            label="expected prepared projection execution",
        )
    ):
        raise ValueError(
            "prepared-cohort projection binding belongs to another Stage 1 graph"
        )
    return payload


@dataclass(frozen=True)
class AuthenticatedRoleNeutralStage2RuntimeBinding:
    """Opaque direct-runtime authority over one exact prepared cohort graph."""

    plan_scientific_content_sha256: str
    prepared_request_sha256: str
    source_execution_content_sha256: str
    provider_identity_sha256: str
    runner_dataset_artifact_sha256: str
    prepared_projection_binding_content_sha256: str
    row_map_sha256: str
    fold_bindings: tuple[
        tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        ...,
    ]
    content_sha256: str
    _prepared_projection_binding: AuthenticatedPreparedCohortProjectionBinding
    _issuer: object

    def __post_init__(self) -> None:
        if self._issuer is not _DIRECT_RUNTIME_BINDING_ISSUER:
            raise TypeError(
                "direct Stage 2 runtime bindings are issued only by the "
                "authenticated role-neutral provider"
            )

    def _body(self) -> dict[str, Any]:
        projection = self._prepared_projection_binding.as_dict()
        if (
            projection["content_sha256"]
            != self.prepared_projection_binding_content_sha256
            or projection["prepared_cohort_artifact_sha256"]
            != self.runner_dataset_artifact_sha256
        ):
            raise RuntimeError(
                "prepared projection changed after direct runtime authorization"
            )
        return {
            "schema_version": (
                "authenticated_role_neutral_stage2_runtime_binding_v1"
            ),
            "plan_scientific_content_sha256": (
                self.plan_scientific_content_sha256
            ),
            "prepared_request_sha256": self.prepared_request_sha256,
            "source_execution_content_sha256": (
                self.source_execution_content_sha256
            ),
            "provider_identity_sha256": self.provider_identity_sha256,
            "runner_dataset_artifact_sha256": (
                self.runner_dataset_artifact_sha256
            ),
            "prepared_projection_binding_content_sha256": (
                self.prepared_projection_binding_content_sha256
            ),
            "row_map_sha256": self.row_map_sha256,
            "fold_bindings": [
                {
                    "outer_fold": outer_fold,
                    "outer_train_row_ids": list(train_rows),
                    "outer_heldout_row_ids": list(heldout_rows),
                    "meta_inner_fold_ids": list(meta_ids),
                    "outer_train_row_count": len(train_rows),
                    "outer_heldout_row_count": len(heldout_rows),
                }
                for outer_fold, train_rows, heldout_rows, meta_ids in (
                    self.fold_bindings
                )
            ],
            "runner_dataset_matches_prepared_projection": True,
            "fold_row_order_and_meta_assignments_precommitted": True,
            "per_fold_text_treatment_outcome_rehash_required": False,
            "outer_heldout_labels_authorized": False,
        }

    def as_dict(self) -> Mapping[str, Any]:
        body = self._body()
        if _sha256_json(body) != self.content_sha256:
            raise RuntimeError("direct Stage 2 runtime binding changed")
        return {**body, "content_sha256": self.content_sha256}

    def authorize_final_fold_shapes(
        self,
        *,
        outer_fold: int,
        exact_outer_train_row_ids: Sequence[Any],
        exact_outer_heldout_row_ids: Sequence[Any],
        exact_meta_inner_fold_ids: Sequence[Any],
        outer_train_text_count: int,
        outer_train_treatment_count: int,
        outer_train_outcome_count: int,
        outer_heldout_text_count: int,
        runner_dataset_artifact_sha256: str,
    ) -> Mapping[str, Any]:
        """Authorize rows/meta/shapes without rehashing already-bound values."""

        self.as_dict()
        if (
            _require_sha256(
                runner_dataset_artifact_sha256,
                label="runner dataset artifact",
            )
            != self.runner_dataset_artifact_sha256
        ):
            raise ValueError(
                "runner dataset artifact differs from direct runtime binding"
            )
        matches = tuple(
            row for row in self.fold_bindings if row[0] == int(outer_fold)
        )
        if len(matches) != 1:
            raise ValueError("direct runtime binding has no requested outer fold")
        _fold, train_rows, heldout_rows, meta_ids = matches[0]
        supplied_train = _ordered_fit_rows(exact_outer_train_row_ids)
        supplied_heldout = _ordered_fit_rows(exact_outer_heldout_row_ids)
        supplied_meta = tuple(int(value) for value in exact_meta_inner_fold_ids)
        counts = (
            int(outer_train_text_count),
            int(outer_train_treatment_count),
            int(outer_train_outcome_count),
            int(outer_heldout_text_count),
        )
        if (
            supplied_train != train_rows
            or supplied_heldout != heldout_rows
            or supplied_meta != meta_ids
            or counts
            != (
                len(train_rows),
                len(train_rows),
                len(train_rows),
                len(heldout_rows),
            )
        ):
            raise ValueError(
                "runner fold assignments differ: rows, meta assignments, or "
                "observable shapes changed from the direct runtime binding"
            )
        return {
            "schema_version": (
                "authenticated_role_neutral_final_fold_shape_authorization_v1"
            ),
            "runtime_binding_content_sha256": self.content_sha256,
            "outer_fold": int(outer_fold),
            "outer_train_row_count": len(train_rows),
            "outer_heldout_row_count": len(heldout_rows),
            "runner_dataset_artifact_sha256": (
                self.runner_dataset_artifact_sha256
            ),
            "row_order_and_meta_assignments_verified": True,
            "per_fold_text_treatment_outcome_rehashed": False,
            "outer_heldout_labels_authorized": False,
        }


def validate_authenticated_role_neutral_stage2_runtime_binding(
    value: Any,
    *,
    expected_plan_scientific_content_sha256: str,
    expected_source_execution_content_sha256: str,
) -> Mapping[str, Any]:
    if type(value) is not AuthenticatedRoleNeutralStage2RuntimeBinding:
        raise TypeError(
            "direct Stage 2 requires the exact provider-issued runtime binding"
        )
    assert isinstance(value, AuthenticatedRoleNeutralStage2RuntimeBinding)
    payload = value.as_dict()
    if (
        payload["plan_scientific_content_sha256"]
        != _require_sha256(
            expected_plan_scientific_content_sha256,
            label="expected direct runtime plan",
        )
        or payload["source_execution_content_sha256"]
        != _require_sha256(
            expected_source_execution_content_sha256,
            label="expected direct runtime execution",
        )
    ):
        raise ValueError("direct runtime binding belongs to another Stage 1 graph")
    return payload


class AuthenticatedRoleNeutralHierarchyExecutionAuthorization:
    """Opaque one-shot hierarchy authority for the direct Stage 1 provider."""

    def __init__(
        self,
        *,
        issuer: object,
        provider: "AuthenticatedRoleNeutralStage2Provider",
        runtime_binding: AuthenticatedRoleNeutralStage2RuntimeBinding,
        prepared_batch: Any,
        runner: Any,
    ) -> None:
        if issuer is not _DIRECT_HIERARCHY_AUTHORIZATION_ISSUER:
            raise TypeError(
                "direct hierarchy authorizations are issued internally only"
            )
        self._provider = provider
        self._runtime_binding = runtime_binding
        self._prepared_batch = prepared_batch
        self._prepared_folds = prepared_batch.folds
        self._runner = runner
        self._coordinator = prepared_batch.coordinator
        self._coordinator_precommit = prepared_batch.coordinator.precommit
        self._coordinator_execute = type(self._coordinator).execute
        self._approval_sha256 = prepared_batch.approval_sha256
        self._input_manifest_sha256 = prepared_batch.input_manifest_sha256
        provider_identity = provider.identity()
        runtime_payload = runtime_binding.as_dict()
        body = {
            "schema_version": (
                "authenticated_role_neutral_hierarchy_execution_authorization_v1"
            ),
            "provider_identity_sha256": provider_identity["identity_sha256"],
            "runtime_binding_content_sha256": runtime_payload[
                "content_sha256"
            ],
            "prepared_batch_sha256": self._approval_sha256,
            "preparation_input_manifest_sha256": (
                self._input_manifest_sha256
            ),
            "prepared_fold_catalogs": [
                {
                    "outer_fold": fold.outer_fold,
                    "catalog_sha256": fold.catalog.catalog_sha256,
                    "chunk_plan_sha256": fold.chunk_plan.plan_sha256,
                }
                for fold in prepared_batch.folds
            ],
            "caller_digest_authority": False,
            "legacy_handoff_authorization_used": False,
        }
        self._payload = {**body, "content_sha256": _sha256_json(body)}
        self._lock = threading.Lock()
        self._consumed = False

    def as_dict(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._payload)

    def _assert_binding(self, *, prepared_batch: Any, runner: Any) -> None:
        from .all_evidence_fusion_runner import (
            AllEvidenceFusionRunner,
            PreparedHierarchicalDiscoveryBatch,
        )
        from .approved_hierarchical_discovery_batch import (
            ApprovedHierarchicalDiscoveryBatchCoordinator,
        )

        if type(prepared_batch) is not PreparedHierarchicalDiscoveryBatch:
            raise TypeError(
                "direct hierarchy authorization requires the concrete "
                "prepared batch"
            )
        if type(runner) is not AllEvidenceFusionRunner:
            raise TypeError(
                "direct hierarchy authorization requires the concrete runner"
            )
        if (
            prepared_batch is not self._prepared_batch
            or runner is not self._runner
            or prepared_batch.folds is not self._prepared_folds
            or prepared_batch.coordinator is not self._coordinator
            or prepared_batch.coordinator.precommit
            is not self._coordinator_precommit
        ):
            raise ValueError(
                "direct hierarchy authorization owns another runner or batch"
            )
        if (
            type(self._coordinator)
            is not ApprovedHierarchicalDiscoveryBatchCoordinator
            or type(self._coordinator).execute is not self._coordinator_execute
            or "execute" in vars(self._coordinator)
        ):
            raise RuntimeError(
                "direct hierarchy coordinator execution surface changed"
            )
        if (
            prepared_batch.approval_sha256 != self._approval_sha256
            or prepared_batch.input_manifest_sha256
            != self._input_manifest_sha256
            or runner.reference_only_stage1_provider is not self._provider
            or runner.reference_only_stage1_runtime_binding
            is not self._runtime_binding
        ):
            raise ValueError(
                "direct hierarchy preparation or retained runtime changed"
            )
        provider_identity = self._provider.identity()
        runtime_payload = self._runtime_binding.as_dict()
        if (
            provider_identity["identity_sha256"]
            != self._payload["provider_identity_sha256"]
            or runtime_payload["content_sha256"]
            != self._payload["runtime_binding_content_sha256"]
        ):
            raise ValueError(
                "direct hierarchy provider identity changed after authorization"
            )

    def _execute_for_prepared_batch(
        self,
        *,
        prepared_batch: Any,
        runner: Any,
    ) -> Any:
        from .approved_hierarchical_discovery_batch import (
            ApprovedHierarchicalDiscoveryBatchResult,
        )

        with self._lock:
            if self._consumed:
                raise RuntimeError(
                    "direct hierarchy authorization is already consumed"
                )
            self._assert_binding(
                prepared_batch=prepared_batch,
                runner=runner,
            )
            self._consumed = True
        result = self._coordinator_execute(
            self._coordinator,
            approved_batch_sha256=self._approval_sha256,
        )
        if type(result) is not ApprovedHierarchicalDiscoveryBatchResult:
            raise TypeError(
                "direct hierarchy coordinator returned a noncanonical result"
            )
        result.validate_authentication()
        return result

    def _consumed_runtime_binding_for_runner(
        self,
        *,
        prepared_batch: Any,
        runner: Any,
    ) -> Mapping[str, Any]:
        with self._lock:
            if not self._consumed:
                raise RuntimeError(
                    "direct hierarchy authorization has not been consumed"
                )
            self._assert_binding(
                prepared_batch=prepared_batch,
                runner=runner,
            )
            runtime_payload = self._runtime_binding.as_dict()
            return {
                "schema_version": (
                    "authenticated_role_neutral_hierarchy_runtime_transfer_v1"
                ),
                "dataset_artifact": {
                    "sha256": runtime_payload[
                        "runner_dataset_artifact_sha256"
                    ]
                },
                "reference_only_runtime_binding_content_sha256": (
                    runtime_payload["content_sha256"]
                ),
                "provider_identity_sha256": runtime_payload[
                    "provider_identity_sha256"
                ],
                "legacy_handoff_artifact": None,
                "tfidf_handoff_artifact": None,
                "legacy_primary_predictions_artifact": None,
                "independent_stage1_refit_allowed": False,
            }


def authorize_reference_only_role_neutral_hierarchy_execution(
    *,
    provider: "AuthenticatedRoleNeutralStage2Provider",
    runtime_binding: AuthenticatedRoleNeutralStage2RuntimeBinding,
    prepared_batch: Any,
    runner: Any,
) -> AuthenticatedRoleNeutralHierarchyExecutionAuthorization:
    """Issue provider-neutral same-process authority without a user digest."""

    from .all_evidence_fusion_runner import (
        AllEvidenceFusionRunner,
        PreparedHierarchicalDiscoveryBatch,
        _claim_prepared_hierarchy_capability,
    )

    if type(provider) is not AuthenticatedRoleNeutralStage2Provider:
        raise TypeError(
            "direct hierarchy authorization requires the exact provider"
        )
    if type(runtime_binding) is not AuthenticatedRoleNeutralStage2RuntimeBinding:
        raise TypeError(
            "direct hierarchy authorization requires the exact runtime binding"
        )
    if type(runner) is not AllEvidenceFusionRunner:
        raise TypeError(
            "direct hierarchy authorization requires the concrete runner"
        )
    if type(prepared_batch) is not PreparedHierarchicalDiscoveryBatch:
        raise TypeError(
            "direct hierarchy authorization requires the concrete prepared batch"
        )
    if (
        runner.reference_only_stage1_provider is not provider
        or runner.reference_only_stage1_runtime_binding is not runtime_binding
        or tuple(fold.outer_fold for fold in prepared_batch.folds)
        != tuple(sorted(provider.get_outer_fold_assignments()))
    ):
        raise ValueError(
            "direct hierarchy preparation belongs to another provider graph"
        )
    _claim_prepared_hierarchy_capability(prepared_batch)
    return AuthenticatedRoleNeutralHierarchyExecutionAuthorization(
        issuer=_DIRECT_HIERARCHY_AUTHORIZATION_ISSUER,
        provider=provider,
        runtime_binding=runtime_binding,
        prepared_batch=prepared_batch,
        runner=runner,
    )


def _duplicate_rejecting_object(
    pairs: list[tuple[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"{label} contains duplicate key {key!r}")
        value[key] = child
    return value


def _read_registered_json(
    path: Path,
    registration: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Reopen one validator-authenticated JSON registration without TOCTOU."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one regular JSON file")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise ValueError(f"{label} must be private regular data")
    payload = path.read_bytes()
    after = os.lstat(path)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(int(getattr(before, field)) != int(getattr(after, field)) for field in stable_fields):
        raise ValueError(f"{label} changed while it was reopened")
    digest = hashlib.sha256(payload).hexdigest()
    if registration.get("sha256") != digest or registration.get("size_bytes") != len(payload):
        raise ValueError(f"{label} differs from its authenticated registration")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=lambda pairs: _duplicate_rejecting_object(
                pairs,
                label=label,
            ),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not closed UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    body = {key: copy.deepcopy(child) for key, child in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != registration.get("content_sha256") or value.get(
        "content_sha256"
    ) != _sha256_json(body):
        raise ValueError(f"{label} content identity changed")
    return value


def _read_content_addressed_json(
    path: Path,
    *,
    expected_content_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Reopen closed JSON whose content identity is registered by its parent."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one regular JSON file")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise ValueError(f"{label} must be private regular data")
    payload = path.read_bytes()
    after = os.lstat(path)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(int(getattr(before, field)) != int(getattr(after, field)) for field in stable_fields):
        raise ValueError(f"{label} changed while it was reopened")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=lambda pairs: _duplicate_rejecting_object(
                pairs,
                label=label,
            ),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not closed UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    body = {key: copy.deepcopy(child) for key, child in value.items() if key != "content_sha256"}
    expected = _require_sha256(
        expected_content_sha256,
        label=f"{label} registered content",
    )
    if value.get("content_sha256") != expected or value.get("content_sha256") != _sha256_json(body):
        raise ValueError(f"{label} content identity changed")
    return value


def _sha256_file(path: Path, *, label: str) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one regular file")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise ValueError(f"{label} must be private regular data")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    after = os.lstat(path)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        int(getattr(before, field)) != int(getattr(after, field)) for field in stable_fields
    ) or size != int(after.st_size):
        raise RuntimeError(f"{label} changed while it was hashed")
    return digest.hexdigest(), size


def _validate_file_registration(
    path: Path,
    registration: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, int]:
    digest, size = _sha256_file(path, label=label)
    if (
        set(registration)
        != {
            "relative_path",
            "sha256",
            "size_bytes",
            "content_sha256",
        }
        or registration.get("sha256") != digest
        or registration.get("size_bytes") != size
    ):
        raise ValueError(f"{label} differs from its immutable registration")
    _require_sha256(
        registration.get("content_sha256"),
        label=f"{label} content identity",
    )
    return digest, size


def _write_new_bytes(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace immutable file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
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
    _write_new_bytes(path, payload)


def _file_registration(
    path: Path,
    *,
    root: Path,
    content_sha256: str,
) -> dict[str, Any]:
    digest, size = _sha256_file(path, label=path.name)
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "size_bytes": size,
        "content_sha256": _require_sha256(
            content_sha256,
            label=f"{path.name} content identity",
        ),
    }


def _read_closed_json_file(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one regular JSON file")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise ValueError(f"{label} must be private regular data")
    payload = path.read_bytes()
    after = os.lstat(path)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(int(getattr(before, field)) != int(getattr(after, field)) for field in stable_fields):
        raise ValueError(f"{label} changed while it was reopened")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=lambda pairs: _duplicate_rejecting_object(
                pairs,
                label=label,
            ),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not closed UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _ordered_fit_rows(
    row_ids: Sequence[Any],
) -> tuple[int, ...]:
    if isinstance(row_ids, (str, bytes)):
        raise TypeError("fit row IDs must be one explicit sequence")
    rows: list[int] = []
    for value in row_ids:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError("fit row IDs must be integers")
        row_id = int(value)
        if row_id < 0:
            raise ValueError("fit row IDs must be unique nonnegative integers")
        rows.append(row_id)
    result = tuple(rows)
    if not result or len(result) != len(set(result)):
        raise ValueError("fit row IDs must be unique and nonempty")
    return result


def _fit_text_sha256(
    row_ids: tuple[int, ...],
    texts: Sequence[Any],
) -> str:
    values = tuple(texts)
    if len(values) != len(row_ids) or any(
        not isinstance(text, str) or not text.strip() for text in values
    ):
        raise ValueError("fit projection requires one explicit nonempty text per row")
    digest = hashlib.sha256()
    digest.update(b"production-bow-text-binding-v1\0")
    for row_id, text in zip(row_ids, values, strict=True):
        encoded = text.encode("utf-8")
        digest.update(row_id.to_bytes(8, byteorder="little", signed=False))
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _binary_projection_sha256(
    values: Sequence[Any],
    *,
    expected_length: int,
    label: str,
) -> str:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.shape != (expected_length,)
        or not np.isfinite(array).all()
        or not set(np.unique(array)).issubset({0.0, 1.0})
    ):
        raise ValueError(f"{label} must be one finite binary vector aligned to fit rows")
    return _sha256_json([float(value).hex() for value in array])


def build_role_neutral_stage2_fit_projection_proof(
    *,
    plan_scientific_content_sha256: str,
    physical_owner_scope_id: str,
    fit_row_ids: Sequence[Any],
    fit_texts: Sequence[Any],
    fit_treatment: Sequence[Any],
    fit_outcome: Sequence[Any],
) -> dict[str, Any]:
    """Build the raw-value-free proof a producer must seal with its fit.

    The proof contains ordered row identity plus hashes of complete prepared
    text, treatment, and outcome vectors.  It carries no row-level text or
    labels and has no capacity/truncation parameter.
    """

    plan_sha256 = _require_sha256(
        plan_scientific_content_sha256,
        label="fit projection plan",
    )
    owner_id = str(physical_owner_scope_id)
    if not owner_id:
        raise ValueError("fit projection physical owner must be explicit")
    rows = _ordered_fit_rows(fit_row_ids)
    text_sha256 = _fit_text_sha256(rows, fit_texts)
    treatment_sha256 = _binary_projection_sha256(
        fit_treatment,
        expected_length=len(rows),
        label="fit treatment",
    )
    outcome_sha256 = _binary_projection_sha256(
        fit_outcome,
        expected_length=len(rows),
        label="fit outcome",
    )
    combined = {
        "fit_row_ids": list(rows),
        "fit_text_sha256": text_sha256,
        "fit_treatment_sha256": treatment_sha256,
        "fit_outcome_sha256": outcome_sha256,
    }
    body = {
        "schema_version": (ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_PROOF_SCHEMA),
        "plan_scientific_content_sha256": plan_sha256,
        "physical_owner_scope_id": owner_id,
        "fit_row_ids": list(rows),
        "fit_row_order_fingerprint": _sha256_json(list(rows)),
        "fit_text_sha256": text_sha256,
        "fit_treatment_sha256": treatment_sha256,
        "fit_outcome_sha256": outcome_sha256,
        "fit_data_projection_sha256": _sha256_json(combined),
        "raw_text_persisted": False,
        "raw_treatment_persisted": False,
        "raw_outcome_persisted": False,
        "text_truncation_applied": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def validate_role_neutral_stage2_fit_projection_proof(
    value: Any,
    *,
    expected_plan_scientific_content_sha256: str,
    expected_physical_owner_scope_id: str,
    expected_fit_row_ids: Sequence[Any],
    expected_fit_text_sha256: str | None = None,
    expected_fit_treatment_sha256: str | None = None,
    expected_fit_outcome_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Stage 2 fit projection proof must be a mapping")
    proof = copy.deepcopy(dict(value))
    expected_fields = {
        "schema_version",
        "plan_scientific_content_sha256",
        "physical_owner_scope_id",
        "fit_row_ids",
        "fit_row_order_fingerprint",
        "fit_text_sha256",
        "fit_treatment_sha256",
        "fit_outcome_sha256",
        "fit_data_projection_sha256",
        "raw_text_persisted",
        "raw_treatment_persisted",
        "raw_outcome_persisted",
        "text_truncation_applied",
        "content_sha256",
    }
    plan_sha256 = _require_sha256(
        expected_plan_scientific_content_sha256,
        label="expected fit projection plan",
    )
    owner_id = str(expected_physical_owner_scope_id)
    if not owner_id:
        raise ValueError("expected fit projection owner must be explicit")
    rows = _ordered_fit_rows(expected_fit_row_ids)
    body = {key: copy.deepcopy(child) for key, child in proof.items() if key != "content_sha256"}
    combined = {
        "fit_row_ids": list(rows),
        "fit_text_sha256": proof.get("fit_text_sha256"),
        "fit_treatment_sha256": proof.get("fit_treatment_sha256"),
        "fit_outcome_sha256": proof.get("fit_outcome_sha256"),
    }
    if (
        set(proof) != expected_fields
        or proof.get("schema_version") != ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_PROOF_SCHEMA
        or proof.get("plan_scientific_content_sha256") != plan_sha256
        or proof.get("physical_owner_scope_id") != owner_id
        or proof.get("fit_row_ids") != list(rows)
        or proof.get("fit_row_order_fingerprint") != _sha256_json(list(rows))
        or proof.get("fit_data_projection_sha256") != _sha256_json(combined)
        or proof.get("raw_text_persisted") is not False
        or proof.get("raw_treatment_persisted") is not False
        or proof.get("raw_outcome_persisted") is not False
        or proof.get("text_truncation_applied") is not False
        or proof.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError(f"{owner_id} Stage 2 fit projection proof is invalid")
    for field_name in (
        "fit_text_sha256",
        "fit_treatment_sha256",
        "fit_outcome_sha256",
        "fit_data_projection_sha256",
        "content_sha256",
    ):
        _require_sha256(
            proof.get(field_name),
            label=f"{owner_id} {field_name}",
        )
    expected_hashes = {
        "fit_text_sha256": expected_fit_text_sha256,
        "fit_treatment_sha256": expected_fit_treatment_sha256,
        "fit_outcome_sha256": expected_fit_outcome_sha256,
    }
    for field_name, expected in expected_hashes.items():
        if expected is not None and proof[field_name] != _require_sha256(
            expected,
            label=f"expected {owner_id} {field_name}",
        ):
            raise ValueError(
                f"{owner_id} Stage 2 fit projection proof differs from " f"sealed {field_name}"
            )
    return proof


def _validate_fit_projection_proof(
    value: Any,
    *,
    plan: Stage1ScopePlan,
    physical_owner_scope_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RoleNeutralStage2ProjectionProofUnavailable(
            physical_owner_scope_id=physical_owner_scope_id,
        )
    owner = plan.scope(physical_owner_scope_id)
    return validate_role_neutral_stage2_fit_projection_proof(
        value,
        expected_plan_scientific_content_sha256=(plan.scientific_content_sha256),
        expected_physical_owner_scope_id=owner.scope_id,
        expected_fit_row_ids=owner.fit_row_ids,
    )


@dataclass(frozen=True)
class RoleNeutralStage2PhysicalFit:
    """Path-neutral identity of one canonical physical all-ten fit."""

    physical_owner_scope_id: str
    physical_owner_scope_sha256: str
    physical_fit_content_sha256: str
    fit_artifact_sha256: str
    fit_row_order_fingerprint: str
    canonical_group_seed: int
    family_fit_artifact_sha256: tuple[tuple[str, str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "physical_owner_scope_id": self.physical_owner_scope_id,
            "physical_owner_scope_sha256": self.physical_owner_scope_sha256,
            "physical_fit_content_sha256": self.physical_fit_content_sha256,
            "fit_artifact_sha256": self.fit_artifact_sha256,
            "fit_row_order_fingerprint": self.fit_row_order_fingerprint,
            "canonical_group_seed": int(self.canonical_group_seed),
            "family_fit_artifact_sha256": [
                {
                    "portable_family": portable,
                    "native_family": native,
                    "content_sha256": digest,
                }
                for portable, native, digest in self.family_fit_artifact_sha256
            ],
        }


@dataclass(frozen=True)
class RoleNeutralStage2LogicalContext:
    """Purpose-specific logical view bound to one physical all-ten fit."""

    logical_scope_id: str
    logical_scope_sha256: str
    logical_purpose: str
    physical_owner_scope_id: str
    physical_fit_content_sha256: str
    logical_view_artifact_sha256: str
    logical_view_content_sha256: str
    heldout_row_order_fingerprint: str
    view_input_policy: str
    reuses_physical_fit: bool
    family_fit_artifact_sha256: tuple[tuple[str, str, str], ...]
    family_logical_view_content_sha256: tuple[tuple[str, str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_scope_id": self.logical_scope_id,
            "logical_scope_sha256": self.logical_scope_sha256,
            "logical_purpose": self.logical_purpose,
            "physical_owner_scope_id": self.physical_owner_scope_id,
            "physical_fit_content_sha256": (self.physical_fit_content_sha256),
            "logical_view_artifact_sha256": (self.logical_view_artifact_sha256),
            "logical_view_content_sha256": (self.logical_view_content_sha256),
            "heldout_row_order_fingerprint": (self.heldout_row_order_fingerprint),
            "view_input_policy": self.view_input_policy,
            "reuses_physical_fit": self.reuses_physical_fit,
            "family_fit_artifact_sha256": [
                {
                    "portable_family": portable,
                    "native_family": native,
                    "content_sha256": digest,
                }
                for portable, native, digest in self.family_fit_artifact_sha256
            ],
            "family_logical_view_content_sha256": [
                {
                    "portable_family": portable,
                    "native_family": native,
                    "content_sha256": digest,
                }
                for portable, native, digest in (self.family_logical_view_content_sha256)
            ],
        }


@dataclass(frozen=True)
class AuthenticatedRoleNeutralStage2Bridge:
    """Validated in-memory authority for a future direct Stage 2 reader.

    ``execution_root`` is an operational locator and is deliberately excluded
    from :meth:`scientific_identity`.  All nested scientific fields are tuples,
    so callers cannot mutate an authenticated bridge in place.
    """

    execution_root: Path
    source_execution_content_sha256: str
    source_coordination_content_sha256: str
    plan_scientific_content_sha256: str
    coordination_scientific_identity_sha256: str
    binding_terminal_content_sha256: str
    physical_fits: tuple[RoleNeutralStage2PhysicalFit, ...]
    logical_contexts: tuple[RoleNeutralStage2LogicalContext, ...]
    bridge_scientific_content_sha256: str

    def _scientific_body(self) -> dict[str, Any]:
        return {
            "schema_version": ROLE_NEUTRAL_STAGE2_BRIDGE_SCHEMA,
            "plan_scientific_content_sha256": (self.plan_scientific_content_sha256),
            "coordination_scientific_identity_sha256": (
                self.coordination_scientific_identity_sha256
            ),
            "binding_terminal_content_sha256": (self.binding_terminal_content_sha256),
            "portable_family_order": list(EVIDENCE_FAMILIES),
            "native_family_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "portable_to_native_family": dict(PORTABLE_TO_NATIVE_FAMILY),
            "canonical_component_family_partition": {
                component: list(families)
                for component, families in EXPECTED_COMPONENT_FAMILIES.items()
            },
            "physical_fit_count": len(self.physical_fits),
            "logical_scope_count": len(self.logical_contexts),
            "deduplicated_fit_count": (len(self.logical_contexts) - len(self.physical_fits)),
            "physical_fits": [row.as_dict() for row in self.physical_fits],
            "logical_contexts": [row.as_dict() for row in self.logical_contexts],
            "all_ten_nonempty_fit_families_authenticated": True,
            "whole_cohort_and_cluster_local_embeddings_independent": True,
            "heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
            "text_truncation_applied": False,
            "lossy_evidence_selection_applied": False,
            "evidence_payloads_copied": False,
            "evidence_payloads_recomputed": False,
            "htr_native_evidence_schema": (
                ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA
            ),
            "htr_token_evidence_package_schema": (
                ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_PACKAGE_SCHEMA
            ),
            "complete_htr_token_and_chunk_evidence_authenticated": True,
            "complete_htr_raw_token_sidecars_retained_in_source_graph": True,
            "htr_readable_spans_consumed_through_staged_catalogs": True,
            "legacy_bundle_build_invoked": False,
            "legacy_hierarchy_loader_compatible": False,
        }

    def scientific_identity(self) -> dict[str, Any]:
        body = self._scientific_body()
        digest = _sha256_json(body)
        if digest != self.bridge_scientific_content_sha256:
            raise RuntimeError("authenticated role-neutral Stage 2 bridge was mutated")
        return {
            **copy.deepcopy(body),
            "content_sha256": digest,
        }

    def as_dict(self) -> dict[str, Any]:
        """Return scientific identity plus a separate exact-source locator."""

        return {
            "schema_version": ROLE_NEUTRAL_STAGE2_BRIDGE_SCHEMA,
            "scientific_identity": self.scientific_identity(),
            "source_execution_attestation": {
                "root_locator": str(self.execution_root),
                "execution_manifest_content_sha256": (self.source_execution_content_sha256),
                "coordination_manifest_content_sha256": (self.source_coordination_content_sha256),
                "locator_in_scientific_identity": False,
            },
            "materialized_stage2_bundle_published": False,
        }


@dataclass(frozen=True)
class RoleNeutralStage2LoaderRequirements:
    """Closed schema evolution required for zero-copy Stage 2 consumption."""

    component_export_index_schema: str = ROLE_NEUTRAL_STAGE2_COMPONENT_EXPORT_INDEX_SCHEMA
    direct_handoff_schema: str = ROLE_NEUTRAL_STAGE2_DIRECT_HANDOFF_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": (ROLE_NEUTRAL_STAGE2_LOADER_REQUIREMENTS_SCHEMA),
            "component_export_index_schema": (self.component_export_index_schema),
            "direct_handoff_schema": self.direct_handoff_schema,
            "required_component_export_registrations": [
                "logical_scope_and_family",
                "relative_path_within_authenticated_component_root",
                "payload_kind_and_schema",
                "payload_size_and_full_byte_sha256",
                "scientific_content_sha256",
                "native_numerical_bank",
                "lossless_evidence_atoms_or_catalog",
                "native_fit_proof",
                "native_model_descriptor",
                ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD,
            ],
            "required_direct_loader_capabilities": [
                "accept_authenticated_role_neutral_execution_root",
                "open_component_payloads_in_place",
                "bind_prepared_request_dataset_split_model_prompt_and_seed_identity",
                "account_for_every_family_atom_exactly_once",
                "keep_whole_cohort_and_cluster_local_embeddings_distinct",
                "serve_all_logical_contexts_without_refit",
                "verify_runtime_spent_projection_against_sealed_fit_proof",
                "authenticate_complete_htr_raw_sidecars_without_prompt_copy",
                "consume_complete_htr_semantic_aggregates_with_reverse_index",
                "deliver_content_addressed_htr_batches_exactly_once",
            ],
            "forbidden_compatibility_actions": [
                "legacy_bundle_build",
                "raw_evidence_copy",
                "scientific_htr_refit_or_evidence_recomputation",
                "loose_file_or_manual_digest_adoption",
                "lossy_top_k_or_text_truncation",
            ],
        }
        return {**body, "content_sha256": _sha256_json(body)}


ROLE_NEUTRAL_STAGE2_LOADER_REQUIREMENTS = RoleNeutralStage2LoaderRequirements()


class RoleNeutralStage2LoaderContractUnavailable(RuntimeError):
    """The source is valid, but no safe zero-copy Stage 2 loader exists."""

    def __init__(
        self,
        *,
        bridge: AuthenticatedRoleNeutralStage2Bridge,
        requirements: RoleNeutralStage2LoaderRequirements,
    ) -> None:
        self.bridge = bridge
        self.requirements = requirements
        super().__init__(
            "authenticated role-neutral Stage 1 execution validated, but "
            "the current hierarchy loader requires a different root-local "
            "bundle graph and the execution has no standardized Stage 2 "
            "component export index; publication aborted without copying or "
            "recomputing evidence"
        )


class RoleNeutralStage2ProjectionProofUnavailable(RuntimeError):
    """A valid execution predates the exact spent-data projection proof."""

    def __init__(self, *, physical_owner_scope_id: str) -> None:
        self.physical_owner_scope_id = str(physical_owner_scope_id)
        body = {
            "schema_version": ("role_neutral_stage2_projection_proof_schema_addition_v1"),
            "producer_component": "bow",
            "terminal_field": (ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD),
            "field_schema": (ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_PROOF_SCHEMA),
            "required_producer_validation": [
                "derive_from_the_exact_ordered_fit_rows",
                "cross_check_against_sealed_fit_state_text_hash",
                "cross_check_against_sealed_fit_state_treatment_hash",
                "cross_check_against_sealed_fit_state_outcome_hash",
                "publish_before_terminal_content_hash",
            ],
            "raw_values_persisted": False,
            "compatibility_default_allowed": False,
        }
        self.required_schema_addition = {
            **body,
            "content_sha256": _sha256_json(body),
        }
        super().__init__(
            "authenticated role-neutral execution lacks the sealed "
            f"{ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD!r} "
            f"for physical owner {self.physical_owner_scope_id}; runtime "
            "spent row/text/treatment/outcome equality cannot be inferred"
        )


def _family_rows(
    family_fit_artifact_sha256: Mapping[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    if set(family_fit_artifact_sha256) != set(ACTIVE_STAGE1_CONCEPT_FAMILIES):
        raise ValueError(
            "role-neutral bridge source does not contain exactly ten "
            "native family fit identities"
        )
    rows = tuple(
        (
            portable,
            native,
            _require_sha256(
                family_fit_artifact_sha256[native],
                label=f"{native} fit artifact",
            ),
        )
        for portable, native in PORTABLE_TO_NATIVE_FAMILY.items()
    )
    if tuple(portable for portable, _native, _digest in rows) != tuple(EVIDENCE_FAMILIES):
        raise RuntimeError("portable family order changed during handoff")
    return rows


def _family_logical_view_identities(
    locator_attestation: Mapping[str, Any],
    *,
    plan: Stage1ScopePlan,
) -> dict[str, tuple[tuple[str, str, str], ...]]:
    """Project the authenticated six-component receipts into ten-family views."""

    registrations = locator_attestation.get("registrations")
    if not isinstance(registrations, list):
        raise ValueError("component locator attestation lacks receipt registrations")
    by_scope: dict[str, dict[str, str]] = {scope.scope_id: {} for scope in plan.scopes}
    for registration in registrations:
        if not isinstance(registration, Mapping):
            raise ValueError("component receipt registration is malformed")
        component = str(registration.get("component"))
        if component not in EXPECTED_COMPONENT_FAMILIES:
            raise ValueError("component receipt registration is substituted")
        scientific = registration.get("component_scientific_receipt")
        if not isinstance(scientific, Mapping):
            raise ValueError("component scientific receipt is missing")
        family_views = scientific.get("family_logical_view_content_sha256")
        logical_scope_ids = scientific.get("logical_scope_ids")
        if not isinstance(family_views, Mapping) or not isinstance(logical_scope_ids, list):
            raise ValueError("component scientific receipt is incomplete")
        for native_family in EXPECTED_COMPONENT_FAMILIES[component]:
            views = family_views.get(native_family)
            if not isinstance(views, Mapping):
                raise ValueError("component receipt lacks a family logical-view index")
            for scope_id in logical_scope_ids:
                target = by_scope.get(str(scope_id))
                if target is None or native_family in target:
                    raise ValueError(
                        "component receipts duplicate or substitute a " "logical family view"
                    )
                target[native_family] = _require_sha256(
                    views.get(scope_id),
                    label=(f"{scope_id}/{native_family} logical family view"),
                )
    output: dict[str, tuple[tuple[str, str, str], ...]] = {}
    for scope in plan.scopes:
        views = by_scope[scope.scope_id]
        if set(views) != set(ACTIVE_STAGE1_CONCEPT_FAMILIES):
            raise ValueError("component receipts do not cover all ten logical families")
        output[scope.scope_id] = tuple(
            (portable, native, views[native])
            for portable, native in PORTABLE_TO_NATIVE_FAMILY.items()
        )
    return output


def validate_role_neutral_stage2_bridge(
    *,
    execution_root: Path | str,
    plan: Stage1ScopePlan,
    execution_manifest: Mapping[str, Any],
) -> AuthenticatedRoleNeutralStage2Bridge:
    """Freshly validate a complete execution and return its typed identity.

    This is a positive validation operation: a complete role-neutral execution
    returns a bridge.  It does not assert that the legacy hierarchy loader can
    consume the retained producer trees.
    """

    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError("role-neutral Stage 2 bridge requires a scope plan")
    if not isinstance(execution_manifest, Mapping):
        raise TypeError("role-neutral Stage 2 bridge requires an execution manifest")
    root = Path(execution_root)
    fresh_execution = validate_role_neutral_stage1_execution(
        root=root,
        plan=plan,
    )
    if dict(execution_manifest) != fresh_execution:
        raise ValueError(
            "supplied role-neutral execution manifest differs from fresh " "path-only validation"
        )

    coordination_root = root / ROLE_NEUTRAL_COORDINATION_DIRECTORY
    coordination = validate_role_neutral_stage1_coordination_gate(
        root=coordination_root,
        plan=plan,
    )
    binding_terminal = validate_complete_role_neutral_stage1_bindings(
        root=(coordination_root / ROLE_NEUTRAL_SCIENTIFIC_BINDING_DIRECTORY),
        plan=plan,
    )
    locator_registration = coordination.get("component_locator_attestation")
    if not isinstance(locator_registration, Mapping):
        raise ValueError("coordination gate lacks its component locator attestation")
    locator_attestation = _read_registered_json(
        (coordination_root / ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION),
        locator_registration,
        label="role-neutral component locator attestation",
    )
    family_logical_views = _family_logical_view_identities(
        locator_attestation,
        plan=plan,
    )
    scientific_bindings = binding_terminal.get("scientific_bindings")
    if not isinstance(scientific_bindings, Mapping):
        raise ValueError("authenticated coordination gate lacks scientific bindings")
    if (
        fresh_execution.get("legacy_bundle_build_invoked") is not False
        or coordination.get("legacy_role_specific_fragments_adopted") is not False
        or scientific_bindings.get("heldout_labels_supplied") is not False
    ):
        raise ValueError("role-neutral bridge rejects legacy or held-out-label evidence")

    physical_rows = scientific_bindings.get("physical_fits")
    logical_rows = scientific_bindings.get("logical_views")
    if not isinstance(physical_rows, list) or not isinstance(logical_rows, list):
        raise ValueError("authenticated scientific bindings are incomplete")

    physical_fits = tuple(
        RoleNeutralStage2PhysicalFit(
            physical_owner_scope_id=str(row["physical_owner_scope_id"]),
            physical_owner_scope_sha256=_require_sha256(
                row["physical_owner_scope_sha256"],
                label="physical owner scope",
            ),
            physical_fit_content_sha256=_require_sha256(
                row["content_sha256"],
                label="physical fit content",
            ),
            fit_artifact_sha256=_require_sha256(
                row["fit_artifact_sha256"],
                label="physical fit artifact",
            ),
            fit_row_order_fingerprint=_require_sha256(
                row["fit_row_order_fingerprint"],
                label="physical fit row order",
            ),
            canonical_group_seed=int(row["canonical_group_seed"]),
            family_fit_artifact_sha256=_family_rows(row["family_fit_artifact_sha256"]),
        )
        for row in physical_rows
    )
    physical_by_owner = {row.physical_owner_scope_id: row for row in physical_fits}
    if len(physical_fits) != len(plan.physical_scopes) or tuple(physical_by_owner) != tuple(
        scope.scope_id for scope in plan.physical_scopes
    ):
        raise ValueError("role-neutral bridge physical-fit order changed")

    logical_contexts: list[RoleNeutralStage2LogicalContext] = []
    for registration, row in zip(
        binding_terminal["logical_views"],
        logical_rows,
        strict=True,
    ):
        if not isinstance(registration, Mapping) or not isinstance(row, Mapping):
            raise ValueError("role-neutral logical binding is malformed")
        owner_id = str(row["physical_owner_scope_id"])
        owner = physical_by_owner.get(owner_id)
        if owner is None:
            raise ValueError("logical context names an unknown physical fit owner")
        family_rows = _family_rows(row["family_fit_artifact_sha256"])
        if family_rows != owner.family_fit_artifact_sha256:
            raise ValueError("logical context changed an authenticated family fit")
        logical_contexts.append(
            RoleNeutralStage2LogicalContext(
                logical_scope_id=str(row["logical_scope_id"]),
                logical_scope_sha256=_require_sha256(
                    row["logical_scope_sha256"],
                    label="logical scope",
                ),
                logical_purpose=str(row["logical_purpose"]),
                physical_owner_scope_id=owner_id,
                physical_fit_content_sha256=_require_sha256(
                    row["physical_fit_content_sha256"],
                    label="logical physical-fit binding",
                ),
                logical_view_artifact_sha256=_require_sha256(
                    row["logical_view_artifact_sha256"],
                    label="logical view artifact",
                ),
                logical_view_content_sha256=_require_sha256(
                    registration["content_sha256"],
                    label="logical view binding content",
                ),
                heldout_row_order_fingerprint=_require_sha256(
                    row["logical_heldout_row_order_fingerprint"],
                    label="logical held-out row order",
                ),
                view_input_policy=str(row["view_input_policy"]),
                reuses_physical_fit=bool(row["reuses_physical_fit"]),
                family_fit_artifact_sha256=family_rows,
                family_logical_view_content_sha256=(
                    family_logical_views[str(row["logical_scope_id"])]
                ),
            )
        )
    logical_tuple = tuple(logical_contexts)
    if len(logical_tuple) != len(plan.scopes) or tuple(
        row.logical_scope_id for row in logical_tuple
    ) != tuple(scope.scope_id for scope in plan.scopes):
        raise ValueError("role-neutral bridge logical-context order changed")

    coordination_scientific = coordination.get("scientific_identity")
    if not isinstance(coordination_scientific, Mapping):
        raise ValueError("coordination scientific identity is missing")
    bridge = AuthenticatedRoleNeutralStage2Bridge(
        execution_root=root,
        source_execution_content_sha256=_require_sha256(
            fresh_execution["content_sha256"],
            label="role-neutral execution content",
        ),
        source_coordination_content_sha256=_require_sha256(
            coordination["content_sha256"],
            label="role-neutral coordination content",
        ),
        plan_scientific_content_sha256=plan.scientific_content_sha256,
        coordination_scientific_identity_sha256=_require_sha256(
            coordination_scientific["content_sha256"],
            label="coordination scientific identity",
        ),
        binding_terminal_content_sha256=_require_sha256(
            binding_terminal["content_sha256"],
            label="role-neutral binding terminal",
        ),
        physical_fits=physical_fits,
        logical_contexts=logical_tuple,
        bridge_scientific_content_sha256="0" * 64,
    )
    digest = _sha256_json(bridge._scientific_body())
    bridge = AuthenticatedRoleNeutralStage2Bridge(
        execution_root=bridge.execution_root,
        source_execution_content_sha256=(bridge.source_execution_content_sha256),
        source_coordination_content_sha256=(bridge.source_coordination_content_sha256),
        plan_scientific_content_sha256=(bridge.plan_scientific_content_sha256),
        coordination_scientific_identity_sha256=(bridge.coordination_scientific_identity_sha256),
        binding_terminal_content_sha256=(bridge.binding_terminal_content_sha256),
        physical_fits=bridge.physical_fits,
        logical_contexts=bridge.logical_contexts,
        bridge_scientific_content_sha256=digest,
    )
    bridge.scientific_identity()
    return bridge


def _reopen_provider_source_graph(
    *,
    execution_root: Path,
    plan: Stage1ScopePlan,
    execution_manifest: Mapping[str, Any],
) -> tuple[
    AuthenticatedRoleNeutralStage2Bridge,
    dict[str, Any],
    dict[str, Any],
]:
    """Validate the full graph once, then reopen its two compact indexes."""

    bridge = validate_role_neutral_stage2_bridge(
        execution_root=execution_root,
        plan=plan,
        execution_manifest=execution_manifest,
    )
    coordination_root = execution_root / ROLE_NEUTRAL_COORDINATION_DIRECTORY
    gate_registration = execution_manifest.get("coordination_gate")
    if not isinstance(gate_registration, Mapping):
        raise ValueError("execution manifest lacks its coordination gate")
    coordination = _read_content_addressed_json(
        coordination_root / ROLE_NEUTRAL_COORDINATION_MANIFEST,
        expected_content_sha256=str(gate_registration.get("manifest_content_sha256")),
        label="role-neutral coordination manifest",
    )
    if coordination.get("content_sha256") != bridge.source_coordination_content_sha256:
        raise RuntimeError("coordination manifest changed after bridge validation")
    binding_terminal = validate_complete_role_neutral_stage1_bindings(
        root=(coordination_root / ROLE_NEUTRAL_SCIENTIFIC_BINDING_DIRECTORY),
        plan=plan,
    )
    if binding_terminal.get("content_sha256") != bridge.binding_terminal_content_sha256:
        raise RuntimeError("scientific binding changed after bridge validation")
    locator_registration = coordination.get("component_locator_attestation")
    if not isinstance(locator_registration, Mapping):
        raise ValueError("coordination manifest lacks its component locator attestation")
    locator_attestation = _read_registered_json(
        (coordination_root / ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION),
        locator_registration,
        label="role-neutral component locator attestation",
    )
    return bridge, binding_terminal, locator_attestation


def _projection_proofs_by_owner(
    *,
    execution_root: Path,
    plan: Stage1ScopePlan,
    locator_attestation: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    registrations = locator_attestation.get("registrations")
    if not isinstance(registrations, list):
        raise ValueError("component locator attestation lacks registrations")
    bow_by_owner: dict[str, Mapping[str, Any]] = {}
    for registration in registrations:
        if not isinstance(registration, Mapping):
            raise ValueError("component locator registration is malformed")
        if registration.get("component") != "bow":
            continue
        owner_id = str(registration.get("physical_owner_scope_id"))
        if owner_id in bow_by_owner:
            raise ValueError("physical owner has two BoW projection sources")
        bow_by_owner[owner_id] = registration
    expected_owners = tuple(owner.scope_id for owner in plan.physical_scopes)
    if set(bow_by_owner) != set(expected_owners):
        raise ValueError("BoW projection-source coverage is incomplete")

    proofs: dict[str, dict[str, Any]] = {}
    for owner_id in expected_owners:
        registration = bow_by_owner[owner_id]
        component_root = Path(str(registration.get("absolute_root_locator")))
        expected_root = execution_root / "components" / owner_id / "bow"
        if component_root != expected_root:
            raise ValueError("BoW projection source points outside the execution tree")
        terminal = _read_content_addressed_json(
            component_root / "execution_manifest.json",
            expected_content_sha256=str(registration.get("source_terminal_content_sha256")),
            label=f"{owner_id} BoW execution terminal",
        )
        proofs[owner_id] = _validate_fit_projection_proof(
            terminal.get(ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD),
            plan=plan,
            physical_owner_scope_id=owner_id,
        )
    return proofs


def _physical_family_seals(
    *,
    execution_root: Path,
    plan: Stage1ScopePlan,
    binding_terminal: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    binding_root = (
        execution_root
        / ROLE_NEUTRAL_COORDINATION_DIRECTORY
        / ROLE_NEUTRAL_SCIENTIFIC_BINDING_DIRECTORY
    )
    registrations = binding_terminal.get("physical_payloads")
    if not isinstance(registrations, list):
        raise ValueError("scientific binding lacks physical payload registrations")
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for owner, registration in zip(
        plan.physical_scopes,
        registrations,
        strict=True,
    ):
        if not isinstance(registration, Mapping):
            raise ValueError("physical family-seal registration is malformed")
        payload = _read_registered_json(
            binding_root / str(registration.get("relative_path")),
            registration,
            label=f"{owner.scope_id} physical all-ten payload",
        )
        seals = payload.get("family_fit_seals")
        if not isinstance(seals, Mapping) or set(seals) != set(ACTIVE_STAGE1_CONCEPT_FAMILIES):
            raise ValueError(f"{owner.scope_id} physical payload is not all-ten")
        output[owner.scope_id] = {
            family: copy.deepcopy(dict(seals[family])) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        }
    if len(output) != len(plan.physical_scopes):
        raise ValueError("physical family-seal coverage changed")
    return output


@dataclass(frozen=True)
class AuthenticatedHtrSemanticAggregationStore:
    root: Path
    manifest: Mapping[str, Any]
    scope_results: Mapping[str, HtrSemanticAggregationResult]

    @property
    def content_sha256(self) -> str:
        return _require_sha256(
            self.manifest.get("content_sha256"),
            label="HTR semantic aggregation store",
        )

    @property
    def preflight_report(self) -> Mapping[str, Any]:
        value = self.manifest.get("preflight_report")
        if not isinstance(value, Mapping):
            raise ValueError("HTR semantic aggregation store lacks its preflight report")
        return copy.deepcopy(dict(value))


def _cumulative_scope_rows(
    *,
    plan: Stage1ScopePlan,
    bridge: AuthenticatedRoleNeutralStage2Bridge,
) -> tuple[tuple[Any, Any, Any], ...]:
    logical_by_scope = {
        row.logical_scope_id: row for row in bridge.logical_contexts
    }
    rows: list[tuple[Any, Any, Any]] = []
    for scope in plan.scopes:
        if scope.scope_kind != "cumulative_spent":
            continue
        if scope.context_epoch is None or scope.provider_inner_fold is None:
            raise ValueError("cumulative scope lacks its hierarchy binding")
        logical = logical_by_scope.get(scope.scope_id)
        if logical is None:
            raise ValueError("cumulative scope lacks its logical-view binding")
        owner = plan.physical_owner(scope.scope_id)
        rows.append((scope, owner, logical))
    rows.sort(key=lambda row: (int(row[0].outer_fold), int(row[0].context_epoch)))
    expected = int(plan.review_rounds) * len(
        {
            scope.outer_fold
            for scope in plan.scopes
            if scope.scope_kind == "full_outer"
        }
    )
    if len(rows) != expected:
        raise ValueError("cumulative HTR aggregation scope coverage is incomplete")
    return tuple(rows)


def _htr_fit_seal(
    *,
    execution_root: Path,
    physical_owner_scope_id: str,
) -> tuple[Path, dict[str, Any]]:
    path = (
        execution_root
        / "components"
        / physical_owner_scope_id
        / "htr"
        / "fit_only_family_seal.json"
    )
    seal = _read_closed_json_file(
        path,
        label=f"{physical_owner_scope_id} HTR fit-only family seal",
    )
    body = {
        key: copy.deepcopy(child)
        for key, child in seal.items()
        if key != "content_sha256"
    }
    payload = seal.get("evidence_payload")
    if (
        seal.get("family") != "htr_neural"
        or seal.get("physical_owner_scope_id") != physical_owner_scope_id
        or seal.get("content_sha256") != _sha256_json(body)
        or not isinstance(payload, Mapping)
        or seal.get("evidence_payload_sha256") != _sha256_json(payload)
        or payload.get("schema_version")
        != ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA
        or not isinstance(payload.get("token_attention_evidence"), Mapping)
        or payload["token_attention_evidence"].get("schema_version")
        != ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_PACKAGE_SCHEMA
        or payload["token_attention_evidence"].get("sentence_pooling")
        != "token_attention"
        or payload["token_attention_evidence"].get(
            "effective_sentence_pooling"
        )
        != "token_attention"
        or payload["token_attention_evidence"].get(
            "all_raw_token_occurrences_authenticated"
        )
        is not True
        or payload["token_attention_evidence"].get("exact_oof_note_coverage")
        is not True
    ):
        raise ValueError(
            f"{physical_owner_scope_id} HTR fit seal is not a complete "
            "token-attention source"
        )
    return path, seal


def _build_htr_semantic_aggregation_store(
    *,
    root: Path,
    execution_root: Path,
    execution_content_sha256: str,
    plan: Stage1ScopePlan,
    bridge: AuthenticatedRoleNeutralStage2Bridge,
) -> AuthenticatedHtrSemanticAggregationStore:
    target = Path(root)
    if not target.is_absolute():
        raise ValueError("HTR semantic store target must be absolute")
    if target.exists() or target.is_symlink():
        raise FileExistsError("HTR semantic store target must be fresh")
    execution_sha = _require_sha256(
        execution_content_sha256,
        label="HTR semantic store source execution",
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            dir=target.parent,
        )
    )
    scope_results: dict[str, HtrSemanticAggregationResult] = {}
    scope_rows: list[dict[str, Any]] = []
    try:
        for scope, owner, logical in _cumulative_scope_rows(
            plan=plan,
            bridge=bridge,
        ):
            fit_path, seal = _htr_fit_seal(
                execution_root=execution_root,
                physical_owner_scope_id=owner.scope_id,
            )
            scope_root = staging / scope.scope_id
            result = build_htr_semantic_aggregation_scope(
                root=scope_root,
                source_payload=seal["evidence_payload"],
                source_array_store_root=(
                    fit_path.parent / "fit_state" / "arrays"
                ),
                source_fit_seal_content_sha256=seal["content_sha256"],
                source_payload_content_sha256=seal[
                    "evidence_payload_sha256"
                ],
                source_fit_seal_locator=fit_path.relative_to(
                    execution_root
                ).as_posix(),
                logical_scope_id=scope.scope_id,
                physical_owner_scope_id=owner.scope_id,
                outer_fold=int(scope.outer_fold),
                context_epoch=int(scope.context_epoch),
                scope_binding_sha256=(
                    logical.logical_view_content_sha256
                ),
            )
            scope_results[scope.scope_id] = result
            scope_rows.append(
                {
                    "logical_scope_id": scope.scope_id,
                    "physical_owner_scope_id": owner.scope_id,
                    "outer_fold": int(scope.outer_fold),
                    "context_epoch": int(scope.context_epoch),
                    "scope_binding_sha256": (
                        logical.logical_view_content_sha256
                    ),
                    "scope_root_relative_path": scope.scope_id,
                    "scope_manifest": _file_registration(
                        result.scope_manifest_path,
                        root=staging,
                        content_sha256=result.scope_manifest[
                            "content_sha256"
                        ],
                    ),
                }
            )
        preflight = summarize_htr_call_plan(
            result.scope_manifest for result in scope_results.values()
        )
        body = {
            "schema_version": HTR_STAGE2_STORE_MANIFEST_SCHEMA,
            "source_execution_content_sha256": execution_sha,
            "scope_plan_scientific_content_sha256": (
                plan.scientific_content_sha256
            ),
            "scope_count": len(scope_rows),
            "scopes": scope_rows,
            "preflight_report": preflight,
            "raw_htr_token_arrays_copied": False,
            "raw_htr_chunk_records_copied_to_model_prompts": False,
            "aggregate_reverse_indexes_complete": True,
            "derived_store_only": True,
        }
        manifest = {**body, "content_sha256": _sha256_json(body)}
        _write_new_json(
            staging / ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_MANIFEST,
            manifest,
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    relocated_results = {
        scope_id: HtrSemanticAggregationResult(
            payload=result.payload,
            scope_manifest=result.scope_manifest,
            scope_manifest_path=(
                target / scope_id / "scope_manifest.json"
            ),
        )
        for scope_id, result in scope_results.items()
    }
    return AuthenticatedHtrSemanticAggregationStore(
        root=target,
        manifest=manifest,
        scope_results=relocated_results,
    )


def _validate_htr_semantic_aggregation_store(
    *,
    root: Path,
    execution_root: Path,
    execution_content_sha256: str,
    plan: Stage1ScopePlan,
    bridge: AuthenticatedRoleNeutralStage2Bridge,
) -> AuthenticatedHtrSemanticAggregationStore:
    target = Path(root)
    if (
        not target.is_absolute()
        or target.is_symlink()
        or target.resolve(strict=True) != target
        or not target.is_dir()
    ):
        raise ValueError("HTR semantic aggregation store root is not canonical")
    manifest_path = target / ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_MANIFEST
    manifest = _read_closed_json_file(
        manifest_path,
        label="HTR semantic aggregation store manifest",
    )
    body = {
        key: copy.deepcopy(child)
        for key, child in manifest.items()
        if key != "content_sha256"
    }
    rows = manifest.get("scopes")
    expected_scope_rows = _cumulative_scope_rows(plan=plan, bridge=bridge)
    if (
        manifest.get("schema_version") != HTR_STAGE2_STORE_MANIFEST_SCHEMA
        or manifest.get("content_sha256") != _sha256_json(body)
        or manifest.get("source_execution_content_sha256")
        != _require_sha256(
            execution_content_sha256,
            label="expected HTR semantic source execution",
        )
        or manifest.get("scope_plan_scientific_content_sha256")
        != plan.scientific_content_sha256
        or not isinstance(rows, list)
        or len(rows) != len(expected_scope_rows)
        or manifest.get("scope_count") != len(rows)
        or manifest.get("raw_htr_token_arrays_copied") is not False
        or manifest.get("raw_htr_chunk_records_copied_to_model_prompts")
        is not False
    ):
        raise ValueError("HTR semantic aggregation store manifest is invalid")
    results: dict[str, HtrSemanticAggregationResult] = {}
    for registration, (scope, owner, logical) in zip(
        rows,
        expected_scope_rows,
        strict=True,
    ):
        if not isinstance(registration, Mapping):
            raise ValueError("HTR semantic scope registration is malformed")
        expected_root = target / scope.scope_id
        if (
            registration.get("logical_scope_id") != scope.scope_id
            or registration.get("physical_owner_scope_id")
            != owner.scope_id
            or registration.get("outer_fold") != int(scope.outer_fold)
            or registration.get("context_epoch")
            != int(scope.context_epoch)
            or registration.get("scope_binding_sha256")
            != logical.logical_view_content_sha256
            or registration.get("scope_root_relative_path")
            != scope.scope_id
            or not isinstance(registration.get("scope_manifest"), Mapping)
        ):
            raise ValueError("HTR semantic scope registration changed")
        fit_path, seal = _htr_fit_seal(
            execution_root=execution_root,
            physical_owner_scope_id=owner.scope_id,
        )
        scope_manifest_registration = registration["scope_manifest"]
        _validate_file_registration(
            expected_root / "scope_manifest.json",
            scope_manifest_registration,
            label=f"{scope.scope_id} HTR semantic scope manifest",
        )
        result = validate_htr_semantic_aggregation_scope(
            root=expected_root,
            source_payload=seal["evidence_payload"],
            source_array_store_root=(
                fit_path.parent / "fit_state" / "arrays"
            ),
            expected_source_fit_seal_content_sha256=seal[
                "content_sha256"
            ],
            expected_source_payload_content_sha256=seal[
                "evidence_payload_sha256"
            ],
            expected_scope_binding_sha256=(
                logical.logical_view_content_sha256
            ),
        )
        if (
            result.scope_manifest["source_fit_seal_locator"]
            != fit_path.relative_to(execution_root).as_posix()
            or result.scope_manifest["content_sha256"]
            != scope_manifest_registration["content_sha256"]
        ):
            raise ValueError("HTR semantic scope source locator changed")
        results[scope.scope_id] = result
    preflight = summarize_htr_call_plan(
        result.scope_manifest for result in results.values()
    )
    if preflight != manifest.get("preflight_report"):
        raise ValueError("HTR semantic aggregation preflight report changed")
    return AuthenticatedHtrSemanticAggregationStore(
        root=target,
        manifest=manifest,
        scope_results=results,
    )


def _review_partition_assignments(
    plan: Stage1ScopePlan,
    *,
    outer_fold: int,
) -> dict[int, tuple[int, ...]]:
    rows = {
        int(scope.inner_fold): tuple(scope.heldout_row_ids)
        for scope in plan.scopes
        if scope.outer_fold == int(outer_fold)
        and scope.scope_kind == "exact_inner"
        and scope.inner_fold is not None
    }
    expected = tuple(
        range(
            1,
            int(plan.initial_training_partitions) + int(plan.review_rounds) + 1,
        )
    )
    if tuple(sorted(rows)) != expected:
        raise ValueError("role-neutral plan review partition coverage is incomplete")
    return {partition: rows[partition] for partition in expected}


def _component_root_index(
    locator_attestation: Mapping[str, Any],
) -> dict[tuple[str, str], Path]:
    registrations = locator_attestation.get("registrations")
    if not isinstance(registrations, list):
        raise ValueError(
            "component locator attestation lacks its registrations"
        )
    output: dict[tuple[str, str], Path] = {}
    for registration in registrations:
        if not isinstance(registration, Mapping):
            raise ValueError("component locator registration is malformed")
        owner_id = str(
            registration.get("physical_owner_scope_id")
        )
        component = str(registration.get("component"))
        key = (owner_id, component)
        root = Path(str(registration.get("absolute_root_locator")))
        if (
            not owner_id
            or component not in EXPECTED_COMPONENT_FAMILIES
            or key in output
            or not root.is_absolute()
        ):
            raise ValueError(
                "component locator registration index is invalid"
            )
        output[key] = root
    return output


def _resolve_fit_seal_reference(
    *,
    owner_scope_id: str,
    family: str,
    seal_or_reference: Mapping[str, Any],
    component_roots: Mapping[tuple[str, str], Path],
) -> dict[str, Any]:
    """Open one complete seal only when its Stage 2 family is consumed."""

    reference_schema = seal_or_reference.get("schema_version")
    if reference_schema not in (
        ROLE_NEUTRAL_FIT_ONLY_FAMILY_SEAL_REFERENCE_SCHEMAS
    ):
        return copy.deepcopy(dict(seal_or_reference))
    components = [
        component
        for component, families in EXPECTED_COMPONENT_FAMILIES.items()
        if family in families
    ]
    if len(components) != 1:
        raise RuntimeError("native family has no unique producer component")
    component = components[0]
    root = component_roots.get((owner_scope_id, component))
    registration = seal_or_reference.get(
        "source_seal_registration"
    )
    if root is None or not isinstance(registration, Mapping):
        raise ValueError(
            f"{owner_scope_id}/{family} seal reference lacks its source"
        )
    relative = Path(str(registration.get("relative_path")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"{owner_scope_id}/{family} seal reference path is unsafe"
        )
    source = _read_registered_json(
        root / relative,
        registration,
        label=f"{owner_scope_id}/{family} complete fit-only seal",
    )
    if (
        source.get("content_sha256")
        != seal_or_reference.get("content_sha256")
        or source.get("physical_owner_scope_id") != owner_scope_id
        or source.get("family") != family
    ):
        raise ValueError(
            f"{owner_scope_id}/{family} seal reference changed its "
            "authenticated source identity"
        )
    if (
        reference_schema
        == ROLE_NEUTRAL_FIT_ONLY_FAMILY_SEAL_REFERENCE_SCHEMA
    ):
        for field_name in (
            "producer_identity_sha256",
            "configuration_identity_sha256",
            "fit_state_artifact_sha256",
        ):
            if source.get(field_name) == seal_or_reference.get(
                field_name
            ):
                continue
            raise ValueError(
                f"{owner_scope_id}/{family} seal reference changed "
                f"{field_name}"
            )
    elif (
        reference_schema
        != ROLE_NEUTRAL_FIT_ONLY_FAMILY_PRIOR_AUTH_REFERENCE_SCHEMA
    ):
        raise ValueError(
            f"{owner_scope_id}/{family} seal reference schema is invalid"
        )
    projection = seal_or_reference.get("source_evidence_projection")
    if projection == "identity_evidence_payload_v1":
        payload = source.get("evidence_payload")
    elif (
        projection
        == "matched_pair_subproducer_normalization_v1"
        and family == MATCHED_PAIR_UPLIFT
    ):
        proofs = source.get("subproducer_proofs")
        if not isinstance(proofs, list) or not proofs:
            raise ValueError(
                f"{owner_scope_id}/{family} source proofs are incomplete"
            )
        payload = {
            "schema_version": (
                NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION
            ),
            "family": MATCHED_PAIR_UPLIFT,
            "architecture_evidence": [
                {
                    "source_family_seal_content_sha256": source[
                        "content_sha256"
                    ],
                    "subproducer": proof["subproducer"],
                    "evidence_payload_sha256": proof[
                        "evidence_payload_sha256"
                    ],
                    "evidence_payload": copy.deepcopy(
                        proof["evidence_payload"]
                    ),
                }
                for proof in proofs
            ],
        }
    else:
        raise ValueError(
            f"{owner_scope_id}/{family} source projection is invalid"
        )
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"{owner_scope_id}/{family} referenced evidence is invalid"
        )
    if reference_schema == (
        ROLE_NEUTRAL_FIT_ONLY_FAMILY_PRIOR_AUTH_REFERENCE_SCHEMA
    ):
        if (
            projection == "identity_evidence_payload_v1"
            and _sha256_json(payload)
            != _require_sha256(
                source.get("evidence_payload_sha256"),
                label=(
                    f"{owner_scope_id}/{family} source evidence payload"
                ),
            )
        ):
            raise ValueError(
                f"{owner_scope_id}/{family} referenced evidence is "
                "invalid"
            )
    elif _sha256_json(payload) != _require_sha256(
        seal_or_reference.get("evidence_payload_sha256"),
        label=f"{owner_scope_id}/{family} evidence payload",
    ):
        raise ValueError(
            f"{owner_scope_id}/{family} referenced evidence is invalid"
        )
    resolved = copy.deepcopy(dict(source))
    # The source seal may bind an earlier aggregate all-ten plan.  The
    # compact receipt re-expresses that fit under the current plan without
    # changing its independently authenticated evidence payload.
    resolved["content_sha256"] = _require_sha256(
        seal_or_reference.get("content_sha256"),
        label=f"{owner_scope_id}/{family} current fit seal",
    )
    return resolved


def _cumulative_catalogs(
    *,
    execution_root: Path,
    plan: Stage1ScopePlan,
    bridge: AuthenticatedRoleNeutralStage2Bridge,
    binding_terminal: Mapping[str, Any],
    locator_attestation: Mapping[str, Any],
    htr_aggregation_store: AuthenticatedHtrSemanticAggregationStore,
    semantic_member_batch_size: int,
) -> dict[tuple[int, int], RoleNeutralEvidenceCatalog]:
    batch_size = _require_semantic_member_batch_size(
        semantic_member_batch_size
    )
    logical_by_scope = {row.logical_scope_id: row for row in bridge.logical_contexts}
    physical_registrations = binding_terminal.get("physical_payloads")
    if (
        not isinstance(physical_registrations, list)
        or len(physical_registrations) != len(plan.physical_scopes)
    ):
        raise ValueError("scientific binding physical payload coverage changed")
    registration_by_owner = {
        owner.scope_id: registration
        for owner, registration in zip(
            plan.physical_scopes,
            physical_registrations,
            strict=True,
        )
    }
    if any(
        not isinstance(registration, Mapping)
        for registration in registration_by_owner.values()
    ):
        raise ValueError("scientific binding physical registration is malformed")
    binding_root = (
        execution_root
        / ROLE_NEUTRAL_COORDINATION_DIRECTORY
        / ROLE_NEUTRAL_SCIENTIFIC_BINDING_DIRECTORY
    )
    component_roots = _component_root_index(locator_attestation)
    cached_non_htr: dict[
        str,
        tuple[dict[str, Mapping[str, Any]], dict[str, str], str],
    ] = {}
    catalogs: dict[tuple[int, int], RoleNeutralEvidenceCatalog] = {}
    for scope in plan.scopes:
        if scope.scope_kind != "cumulative_spent":
            continue
        if scope.context_epoch is None or scope.provider_inner_fold is None:
            raise ValueError("cumulative scope lacks its hierarchy epoch binding")
        owner = plan.physical_owner(scope.scope_id)
        cached = cached_non_htr.get(owner.scope_id)
        if cached is None:
            registration = registration_by_owner.get(owner.scope_id)
            if registration is None:
                raise ValueError("cumulative scope lacks its physical family seals")
            physical_payload = _read_registered_json(
                binding_root / str(registration.get("relative_path")),
                registration,
                label=f"{owner.scope_id} physical all-ten payload",
            )
            owner_seals = physical_payload.get("family_fit_seals")
            if (
                not isinstance(owner_seals, Mapping)
                or set(owner_seals) != set(ACTIVE_STAGE1_CONCEPT_FAMILIES)
            ):
                raise ValueError(f"{owner.scope_id} physical payload is not all-ten")
            non_htr_payloads: dict[str, Mapping[str, Any]] = {}
            artifact_hashes: dict[str, str] = {}
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
                seal = owner_seals[family]
                if not isinstance(seal, Mapping):
                    raise ValueError(f"{owner.scope_id}/{family} fit seal is malformed")
                artifact_hashes[family] = _require_sha256(
                    seal.get("content_sha256"),
                    label=f"{owner.scope_id}/{family} fit seal",
                )
                if family == HTR_NEURAL:
                    continue
                resolved_seal = _resolve_fit_seal_reference(
                    owner_scope_id=owner.scope_id,
                    family=family,
                    seal_or_reference=seal,
                    component_roots=component_roots,
                )
                source_payload = resolved_seal.get("evidence_payload")
                if not isinstance(source_payload, Mapping):
                    raise ValueError(
                        f"{owner.scope_id}/{family} has no sealed evidence payload"
                    )
                non_htr_payloads[family] = copy.deepcopy(
                    dict(source_payload)
                )
            cached = (
                non_htr_payloads,
                artifact_hashes,
                artifact_hashes[HTR_NEURAL],
            )
            cached_non_htr[owner.scope_id] = cached
        non_htr_payloads, artifact_hashes, _htr_artifact_hash = cached
        aggregate_result = htr_aggregation_store.scope_results.get(
            scope.scope_id
        )
        if aggregate_result is None:
            raise ValueError("cumulative scope lacks its HTR aggregate payload")
        payloads = {
            **{
                family: copy.deepcopy(dict(payload))
                for family, payload in non_htr_payloads.items()
            },
            HTR_NEURAL: copy.deepcopy(dict(aggregate_result.payload)),
        }
        provenance = FoldEvidenceProvenance(
            outer_fold=scope.outer_fold,
            train_row_ids=scope.fit_row_ids,
            heldout_row_ids=scope.heldout_row_ids,
            scope="inner_train",
            inner_fold=scope.provider_inner_fold,
            artifact_id=(f"role-neutral-stage1-{scope.scope_id}"),
        )
        logical = logical_by_scope.get(scope.scope_id)
        if logical is None:
            raise ValueError("cumulative scope lacks its authenticated logical binding")
        catalog = assemble_cumulative_spent_role_neutral_catalog(
            family_payload_by_family=payloads,
            family_artifact_sha256_by_family=artifact_hashes,
            scope_binding_sha256=(logical.logical_view_content_sha256),
            scope_id=scope.scope_id,
            outer_fold=scope.outer_fold,
            provider_inner_fold=scope.provider_inner_fold,
            split_fingerprint=provenance.split_fingerprint,
            semantic_member_batch_size=batch_size,
        )
        key = (scope.outer_fold, scope.context_epoch)
        if key in catalogs:
            raise ValueError("cumulative catalog scope is duplicated")
        catalogs[key] = catalog
    expected_count = int(plan.review_rounds) * len(
        {scope.outer_fold for scope in plan.scopes if scope.scope_kind == "full_outer"}
    )
    if len(catalogs) != expected_count:
        raise ValueError("cumulative all-ten catalog coverage is incomplete")
    return catalogs


def _tree_stat_inventory(
    root: Path,
) -> tuple[tuple[Any, ...], ...]:
    """Guard an authenticated in-process handle without rereading payloads."""

    if (
        not root.is_absolute()
        or root.is_symlink()
        or root.resolve(strict=True) != root
        or not root.is_dir()
    ):
        raise ValueError("authenticated handle root must remain one canonical directory")
    paths = (
        root,
        *sorted(root.rglob("*"), key=lambda path: path.as_posix()),
    )
    inventory: list[tuple[Any, ...]] = []
    for path in paths:
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("authenticated handle tree cannot contain symbolic links")
        if stat.S_ISREG(metadata.st_mode):
            kind = "file"
            if int(metadata.st_nlink) != 1:
                raise ValueError("authenticated handle files must remain private")
        elif stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
        else:
            raise ValueError("authenticated handle tree contains a non-file entry")
        relative = "." if path == root else path.relative_to(root).as_posix()
        inventory.append(
            (
                relative,
                kind,
                int(metadata.st_dev),
                int(metadata.st_ino),
                int(metadata.st_mode),
                int(metadata.st_nlink),
                int(metadata.st_size),
                int(metadata.st_mtime_ns),
                int(metadata.st_ctime_ns),
            )
        )
    return tuple(inventory)


def _private_regular_file_stat(path: Path, *, label: str) -> tuple[int, ...]:
    """Return the immutable in-process guard for one authenticated payload."""

    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must remain one canonical regular file")
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_nlink) != 1:
        raise ValueError(f"{label} must remain private regular data")
    return tuple(
        int(getattr(metadata, field))
        for field in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


class AuthenticatedRoleNeutralStage2Provider:
    """Direct, lossless cumulative-catalog provider over role-neutral bytes.

    Construction succeeds only when each physical BoW producer terminal
    contains the standardized exact fit-projection proof.  Current terminals
    that predate that proof fail closed with
    :class:`RoleNeutralStage2ProjectionProofUnavailable`.
    """

    def __init__(
        self,
        *,
        execution_root: Path | str,
        plan: Stage1ScopePlan,
        execution_manifest: Mapping[str, Any],
        semantic_member_batch_size: int,
        htr_aggregation_store_root: Path | str,
        authenticated_row_map_path: Path | str | None = None,
        authenticated_row_map_registration: Mapping[str, Any] | None = None,
        prepared_request_sha256: str | None = None,
    ) -> None:
        if not isinstance(plan, Stage1ScopePlan):
            raise TypeError("role-neutral Stage 2 provider requires a scope plan")
        if not isinstance(execution_manifest, Mapping):
            raise TypeError("role-neutral Stage 2 provider requires an execution manifest")
        self._execution_root = Path(execution_root)
        self._htr_aggregation_store_root = Path(
            htr_aggregation_store_root
        )
        self._plan = plan
        self._semantic_member_batch_size = (
            _require_semantic_member_batch_size(
                semantic_member_batch_size
            )
        )
        self._execution_manifest_json = _canonical_json(dict(execution_manifest))
        self._authenticated_row_map_path: Path | None = None
        self._authenticated_row_map_stat: tuple[int, ...] | None = None
        self._authenticated_row_map: pd.DataFrame | None = None
        self._authenticated_row_map_sha256: str | None = None
        self._prepared_request_sha256: str | None = None
        self._prepared_projection_binding: (
            AuthenticatedPreparedCohortProjectionBinding | None
        ) = None
        self._direct_runtime_binding: (
            AuthenticatedRoleNeutralStage2RuntimeBinding | None
        ) = None
        if (authenticated_row_map_path is None) != (
            authenticated_row_map_registration is None
        ):
            raise ValueError(
                "authenticated row-map path and registration must be supplied together"
            )
        if authenticated_row_map_path is not None:
            if prepared_request_sha256 is None:
                raise ValueError(
                    "authenticated row-map providers require the prepared request identity"
                )
            self._prepared_request_sha256 = _require_sha256(
                prepared_request_sha256,
                label="prepared Stage 1 request",
            )
            row_map_path = Path(authenticated_row_map_path)
            if (
                not row_map_path.is_absolute()
                or row_map_path.is_symlink()
                or row_map_path.resolve(strict=True) != row_map_path
            ):
                raise ValueError(
                    "authenticated row map must be one canonical absolute file"
                )
            full_scopes = tuple(
                scope for scope in plan.scopes if scope.scope_kind == "full_outer"
            )
            if not full_scopes:
                raise ValueError("role-neutral scope plan has no full-outer scopes")
            dataset_rows = set(full_scopes[0].fit_row_ids) | set(
                full_scopes[0].heldout_row_ids
            )
            if dataset_rows != set(range(len(dataset_rows))):
                raise ValueError(
                    "role-neutral scope plan does not bind a canonical prepared row map"
                )
            row_map_sha256, _row_map_size, row_map = _validate_row_map(
                row_map_path,
                authenticated_row_map_registration,
                expected_row_count=len(dataset_rows),
            )
            self._authenticated_row_map_path = row_map_path
            self._authenticated_row_map_stat = _private_regular_file_stat(
                row_map_path,
                label="authenticated role-neutral row map",
            )
            self._authenticated_row_map = row_map.copy(deep=True)
            self._authenticated_row_map_sha256 = row_map_sha256
        before = _tree_stat_inventory(self._execution_root)
        aggregate_before = _tree_stat_inventory(
            self._htr_aggregation_store_root
        )
        (
            bridge,
            proofs,
            catalogs,
            identity,
        ) = self._load_current()
        after = _tree_stat_inventory(self._execution_root)
        aggregate_after = _tree_stat_inventory(
            self._htr_aggregation_store_root
        )
        if before != after:
            raise RuntimeError("role-neutral execution changed across provider " "authentication")
        if aggregate_before != aggregate_after:
            raise RuntimeError(
                "HTR semantic aggregation store changed across provider "
                "authentication"
            )
        self._bridge = bridge
        self._proofs = proofs
        self._catalogs = catalogs
        self._identity = identity
        self._authenticated_stat_inventory = after
        self._authenticated_htr_aggregate_stat_inventory = (
            aggregate_after
        )

    def _execution_manifest(self) -> dict[str, Any]:
        value = json.loads(self._execution_manifest_json)
        if not isinstance(value, dict):
            raise RuntimeError("bound role-neutral execution manifest was corrupted")
        return value

    def _load_current(
        self,
    ) -> tuple[
        AuthenticatedRoleNeutralStage2Bridge,
        dict[str, dict[str, Any]],
        dict[tuple[int, int], RoleNeutralEvidenceCatalog],
        dict[str, Any],
    ]:
        manifest = self._execution_manifest()
        (
            bridge,
            binding_terminal,
            locator_attestation,
        ) = _reopen_provider_source_graph(
            execution_root=self._execution_root,
            plan=self._plan,
            execution_manifest=manifest,
        )
        proofs = _projection_proofs_by_owner(
            execution_root=self._execution_root,
            plan=self._plan,
            locator_attestation=locator_attestation,
        )
        htr_aggregation_store = _validate_htr_semantic_aggregation_store(
            root=self._htr_aggregation_store_root,
            execution_root=self._execution_root,
            execution_content_sha256=(
                bridge.source_execution_content_sha256
            ),
            plan=self._plan,
            bridge=bridge,
        )
        catalogs = _cumulative_catalogs(
            execution_root=self._execution_root,
            plan=self._plan,
            bridge=bridge,
            binding_terminal=binding_terminal,
            locator_attestation=locator_attestation,
            htr_aggregation_store=htr_aggregation_store,
            semantic_member_batch_size=self._semantic_member_batch_size,
        )
        body = {
            "schema_version": ("authenticated_role_neutral_stage2_provider_v4"),
            "bridge_scientific_content_sha256": (bridge.bridge_scientific_content_sha256),
            "plan_scientific_content_sha256": (self._plan.scientific_content_sha256),
            "portable_family_order": list(EVIDENCE_FAMILIES),
            "native_family_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "semantic_member_batch_size": (
                self._semantic_member_batch_size
            ),
            "review_partition_assignments": {
                str(outer_fold): {
                    str(partition): list(rows)
                    for partition, rows in (
                        _review_partition_assignments(
                            self._plan,
                            outer_fold=outer_fold,
                        ).items()
                    )
                }
                for outer_fold in sorted({scope.outer_fold for scope in self._plan.scopes})
            },
            "physical_fit_projection_proofs": [
                {
                    "physical_owner_scope_id": owner.scope_id,
                    "content_sha256": proofs[owner.scope_id]["content_sha256"],
                }
                for owner in self._plan.physical_scopes
            ],
            "cumulative_catalogs": [
                {
                    "outer_fold": outer_fold,
                    "context_epoch": context_epoch,
                    "catalog_sha256": catalogs[(outer_fold, context_epoch)].catalog_sha256,
                    "split_fingerprint": catalogs[(outer_fold, context_epoch)].split_fingerprint,
                }
                for outer_fold, context_epoch in sorted(catalogs)
            ],
            "all_ten_architectures_required": True,
            "catalogs_assembled_losslessly_from_fit_only_seals": True,
            "complete_source_byte_graph_revalidated": True,
            "subsequent_calls_use_guarded_stat_inventory": True,
            "raw_spent_evidence_input_fallback_available": False,
            "independent_runtime_stage1_refit_allowed": False,
            "evidence_payloads_materialized_to_legacy_graph": False,
            "htr_native_evidence_schema": (
                ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA
            ),
            "htr_token_evidence_package_schema": (
                ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_PACKAGE_SCHEMA
            ),
            "htr_stage2_aggregate_payload_schema": (
                HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA
            ),
            "htr_semantic_aggregation_store_content_sha256": (
                htr_aggregation_store.content_sha256
            ),
            "htr_stage2_call_plan_preflight": (
                htr_aggregation_store.preflight_report
            ),
            "complete_htr_token_and_chunk_evidence_authenticated": True,
            "complete_htr_raw_token_sidecars_opened_in_place": True,
            "complete_htr_semantic_reverse_index_authenticated": True,
            "raw_htr_token_arrays_copied_to_handoff": False,
            "raw_htr_chunk_atoms_copied_to_model_prompts": False,
            "text_truncation_applied": False,
            "lossy_evidence_selection_applied": False,
        }
        identity = {**body, "identity_sha256": _sha256_json(body)}
        return bridge, proofs, catalogs, identity

    def _assert_current(
        self,
    ) -> tuple[
        AuthenticatedRoleNeutralStage2Bridge,
        dict[str, dict[str, Any]],
        dict[tuple[int, int], RoleNeutralEvidenceCatalog],
    ]:
        if _tree_stat_inventory(self._execution_root) != self._authenticated_stat_inventory:
            raise RuntimeError(
                "authenticated role-neutral Stage 2 provider byte/stat " "inventory changed"
            )
        if (
            _tree_stat_inventory(self._htr_aggregation_store_root)
            != self._authenticated_htr_aggregate_stat_inventory
        ):
            raise RuntimeError(
                "authenticated HTR semantic aggregate byte/stat inventory "
                "changed"
            )
        if self._authenticated_row_map_path is not None and (
            _private_regular_file_stat(
                self._authenticated_row_map_path,
                label="authenticated role-neutral row map",
            )
            != self._authenticated_row_map_stat
        ):
            raise RuntimeError(
                "authenticated role-neutral prepared row-map inventory changed"
            )
        return self._bridge, self._proofs, self._catalogs

    def identity(self) -> Mapping[str, Any]:
        self._assert_current()
        return copy.deepcopy(self._identity)

    def get_htr_stage2_preflight_catalogs(
        self,
    ) -> tuple[tuple[int, int, RoleNeutralEvidenceCatalog], ...]:
        """Return every authenticated cumulative catalog for offline planning."""

        _bridge, _proofs, catalogs = self._assert_current()
        return tuple(
            (
                int(outer_fold),
                int(context_epoch),
                catalogs[(outer_fold, context_epoch)],
            )
            for outer_fold, context_epoch in sorted(catalogs)
        )

    def get_outer_fold_assignments(
        self,
    ) -> Mapping[int, Mapping[str, tuple[int, ...]]]:
        """Return exact authenticated outer-train/held-out row assignments.

        Counts and fold labels are derived exclusively from the sealed scope
        plan.  No benchmark cohort size or fold count is assumed.
        """

        self._assert_current()
        full_scopes = tuple(
            scope for scope in self._plan.scopes if scope.scope_kind == "full_outer"
        )
        assignments = {
            int(scope.outer_fold): {
                "fit_row_ids": tuple(scope.fit_row_ids),
                "heldout_row_ids": tuple(scope.heldout_row_ids),
            }
            for scope in full_scopes
        }
        if len(assignments) != len(full_scopes):
            raise RuntimeError("role-neutral scope plan duplicates one outer fold")
        return copy.deepcopy(assignments)

    def authenticated_scope_plan(self) -> Stage1ScopePlan:
        """Return the already-authenticated immutable plan for bank validation."""

        self._assert_current()
        # Stage1ScopePlan is frozen and its nested members are frozen tuples.
        # Returning the exact handle also lets downstream bank validators prove
        # they consume the same trust-boundary object, without reconstructing
        # plan identities from operational paths.
        return self._plan

    def bind_prepared_row_map(
        self,
        *,
        prepared: pd.DataFrame,
        unit_id_column: str,
    ) -> Mapping[str, Any]:
        """Authenticate a prepared cohort's exact row order against the handoff.

        This is deliberately a row-map binding only.  The direct runtime must
        separately require its configured text, treatment, and outcome columns,
        and cumulative fit-projection proofs authenticate their exact values
        before review evidence is consumed.
        """

        self._assert_current()
        if self._authenticated_row_map is None:
            raise RuntimeError(
                "this provider was not opened through a reference-only handoff "
                "with an authenticated prepared row map"
            )
        if not isinstance(prepared, pd.DataFrame):
            raise TypeError("prepared cohort must be a pandas DataFrame")
        column = str(unit_id_column)
        if not column or column not in prepared:
            raise ValueError("configured unit-ID column is absent from prepared cohort")
        expected = self._authenticated_row_map
        observed = prepared.loc[:, [column]].reset_index(drop=True)
        observed.columns = ["unit_id"]
        if (
            len(observed) != len(expected)
            or observed["unit_id"].isna().any()
            or observed["unit_id"].duplicated().any()
            or not observed.equals(expected.loc[:, ["unit_id"]])
        ):
            raise ValueError(
                "prepared cohort unit IDs or row order differ from the "
                "authenticated Stage 1 row map"
            )
        return {
            "schema_version": "authenticated_prepared_row_map_binding_v1",
            "row_count": len(expected),
            "row_ids": tuple(
                int(value)
                for value in expected["_oci_row_id"].to_numpy(dtype=np.int64)
            ),
            "row_map_sha256": self._authenticated_row_map_sha256,
            "configured_unit_id_column": column,
            "cohort_path_in_scientific_identity": False,
            "exact_unit_id_order_verified": True,
        }

    def bind_prepared_cohort_projection(
        self,
        *,
        prepared: pd.DataFrame,
        prepared_cohort_artifact_sha256: str,
        unit_id_column: str,
        text_column: str,
        treatment_column: str,
        outcome_column: str,
    ) -> AuthenticatedPreparedCohortProjectionBinding:
        """Verify complete prepared text/T/Y projections for every physical fit."""

        _bridge, proofs, _catalogs = self._assert_current()
        artifact_sha256 = _require_sha256(
            prepared_cohort_artifact_sha256,
            label="prepared cohort artifact",
        )
        row_binding = self.bind_prepared_row_map(
            prepared=prepared,
            unit_id_column=unit_id_column,
        )
        columns = (
            str(unit_id_column),
            str(text_column),
            str(treatment_column),
            str(outcome_column),
        )
        if any(not column or column not in prepared for column in columns):
            raise ValueError(
                "prepared cohort lacks one or more configured projection columns"
            )
        if len(set(columns)) != len(columns):
            raise ValueError("prepared cohort projection columns must be distinct")
        if len(prepared) != int(row_binding["row_count"]):
            raise ValueError("prepared cohort row count differs from authenticated row map")
        texts = tuple(prepared[text_column].tolist())
        treatment = np.asarray(prepared[treatment_column], dtype=np.float64)
        outcome = np.asarray(prepared[outcome_column], dtype=np.float64)
        owner_proofs: list[tuple[str, str]] = []
        for owner in self._plan.physical_scopes:
            positions = np.asarray(owner.fit_row_ids, dtype=np.int64)
            observed = build_role_neutral_stage2_fit_projection_proof(
                plan_scientific_content_sha256=(
                    self._plan.scientific_content_sha256
                ),
                physical_owner_scope_id=owner.scope_id,
                fit_row_ids=owner.fit_row_ids,
                fit_texts=tuple(texts[int(position)] for position in positions),
                fit_treatment=treatment[positions],
                fit_outcome=outcome[positions],
            )
            sealed = proofs.get(owner.scope_id)
            if observed != sealed:
                raise ValueError(
                    f"prepared cohort projection differs from sealed physical "
                    f"fit {owner.scope_id}"
                )
            owner_proofs.append(
                (owner.scope_id, str(observed["content_sha256"]))
            )
        body = {
            "schema_version": (
                "authenticated_role_neutral_prepared_cohort_projection_binding_v1"
            ),
            "plan_scientific_content_sha256": (
                self._plan.scientific_content_sha256
            ),
            "prepared_request_sha256": self._prepared_request_sha256,
            "source_execution_content_sha256": (
                self._bridge.source_execution_content_sha256
            ),
            "provider_identity_sha256": self._identity["identity_sha256"],
            "prepared_cohort_artifact_sha256": artifact_sha256,
            "row_map_sha256": self._authenticated_row_map_sha256,
            "row_count": int(row_binding["row_count"]),
            "unit_id_column": str(unit_id_column),
            "text_column": str(text_column),
            "treatment_column": str(treatment_column),
            "outcome_column": str(outcome_column),
            "physical_owner_projection_proofs": [
                {
                    "physical_owner_scope_id": scope_id,
                    "projection_proof_content_sha256": proof_sha256,
                }
                for scope_id, proof_sha256 in owner_proofs
            ],
            "all_physical_fit_projections_verified": True,
            "raw_text_persisted": False,
            "raw_treatment_persisted": False,
            "raw_outcome_persisted": False,
            "text_truncation_applied": False,
        }
        binding = AuthenticatedPreparedCohortProjectionBinding(
            plan_scientific_content_sha256=(
                self._plan.scientific_content_sha256
            ),
            prepared_request_sha256=str(self._prepared_request_sha256),
            source_execution_content_sha256=(
                self._bridge.source_execution_content_sha256
            ),
            provider_identity_sha256=self._identity["identity_sha256"],
            prepared_cohort_artifact_sha256=artifact_sha256,
            row_map_sha256=str(self._authenticated_row_map_sha256),
            row_count=int(row_binding["row_count"]),
            unit_id_column=str(unit_id_column),
            text_column=str(text_column),
            treatment_column=str(treatment_column),
            outcome_column=str(outcome_column),
            physical_owner_projection_proofs=tuple(owner_proofs),
            content_sha256=_sha256_json(body),
            _issuer=_PREPARED_PROJECTION_BINDING_ISSUER,
        )
        if self._prepared_projection_binding is not None and (
            self._prepared_projection_binding.as_dict() != binding.as_dict()
        ):
            raise RuntimeError(
                "provider was already bound to a different prepared cohort projection"
            )
        self._prepared_projection_binding = binding
        return binding

    def issue_direct_runtime_binding(
        self,
        *,
        prepared_projection_binding: AuthenticatedPreparedCohortProjectionBinding,
    ) -> AuthenticatedRoleNeutralStage2RuntimeBinding:
        """Bind direct runner dataset identity, rows, and meta-fold shapes once."""

        self._assert_current()
        call_plan = self._identity.get("htr_stage2_call_plan_preflight")
        if (
            not isinstance(call_plan, Mapping)
            or call_plan.get("stage2_endpoint_launch_allowed") is not True
            or call_plan.get(
                "call_plan_on_order_of_hundreds_of_thousands"
            )
            is not False
        ):
            raise RuntimeError(
                "HTR Stage 2 call-plan preflight forbids endpoint launch; "
                "the remaining semantic redundancy must be reported first"
            )
        if (
            self._prepared_projection_binding is None
            or prepared_projection_binding is not self._prepared_projection_binding
        ):
            raise ValueError(
                "direct runtime authorization requires this provider's exact "
                "prepared-cohort projection binding"
            )
        projection = dict(
            validate_authenticated_prepared_projection_binding(
                prepared_projection_binding,
                expected_plan_scientific_content_sha256=(
                    self._plan.scientific_content_sha256
                ),
                expected_source_execution_content_sha256=(
                    self._bridge.source_execution_content_sha256
                ),
            )
        )
        if projection["provider_identity_sha256"] != self._identity[
            "identity_sha256"
        ]:
            raise ValueError(
                "prepared projection binding belongs to another provider"
            )
        fold_bindings: list[
            tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]
        ] = []
        full_scopes = tuple(
            scope for scope in self._plan.scopes if scope.scope_kind == "full_outer"
        )
        for full_scope in full_scopes:
            inner_scopes = tuple(
                scope
                for scope in self._plan.scopes
                if scope.scope_kind == "exact_inner"
                and scope.outer_fold == full_scope.outer_fold
            )
            meta_by_row: dict[int, int] = {}
            for inner_scope in inner_scopes:
                if inner_scope.inner_fold is None:
                    raise RuntimeError("exact-inner scope lacks its fold identity")
                for row_id in inner_scope.heldout_row_ids:
                    if row_id in meta_by_row:
                        raise RuntimeError(
                            "exact-inner scopes duplicate one outer-training row"
                        )
                    meta_by_row[int(row_id)] = int(inner_scope.inner_fold)
            if set(meta_by_row) != set(full_scope.fit_row_ids):
                raise RuntimeError(
                    "exact-inner scopes do not partition one full outer fit"
                )
            fold_bindings.append(
                (
                    int(full_scope.outer_fold),
                    tuple(full_scope.fit_row_ids),
                    tuple(full_scope.heldout_row_ids),
                    tuple(
                        meta_by_row[int(row_id)]
                        for row_id in full_scope.fit_row_ids
                    ),
                )
            )
        body = {
            "schema_version": (
                "authenticated_role_neutral_stage2_runtime_binding_v1"
            ),
            "plan_scientific_content_sha256": (
                self._plan.scientific_content_sha256
            ),
            "prepared_request_sha256": projection[
                "prepared_request_sha256"
            ],
            "source_execution_content_sha256": (
                self._bridge.source_execution_content_sha256
            ),
            "provider_identity_sha256": self._identity["identity_sha256"],
            "runner_dataset_artifact_sha256": projection[
                "prepared_cohort_artifact_sha256"
            ],
            "prepared_projection_binding_content_sha256": projection[
                "content_sha256"
            ],
            "row_map_sha256": projection["row_map_sha256"],
            "fold_bindings": [
                {
                    "outer_fold": outer_fold,
                    "outer_train_row_ids": list(train_rows),
                    "outer_heldout_row_ids": list(heldout_rows),
                    "meta_inner_fold_ids": list(meta_ids),
                    "outer_train_row_count": len(train_rows),
                    "outer_heldout_row_count": len(heldout_rows),
                }
                for outer_fold, train_rows, heldout_rows, meta_ids in fold_bindings
            ],
            "runner_dataset_matches_prepared_projection": True,
            "fold_row_order_and_meta_assignments_precommitted": True,
            "per_fold_text_treatment_outcome_rehash_required": False,
            "outer_heldout_labels_authorized": False,
        }
        binding = AuthenticatedRoleNeutralStage2RuntimeBinding(
            plan_scientific_content_sha256=(
                self._plan.scientific_content_sha256
            ),
            prepared_request_sha256=projection[
                "prepared_request_sha256"
            ],
            source_execution_content_sha256=(
                self._bridge.source_execution_content_sha256
            ),
            provider_identity_sha256=self._identity["identity_sha256"],
            runner_dataset_artifact_sha256=projection[
                "prepared_cohort_artifact_sha256"
            ],
            prepared_projection_binding_content_sha256=projection[
                "content_sha256"
            ],
            row_map_sha256=projection["row_map_sha256"],
            fold_bindings=tuple(fold_bindings),
            content_sha256=_sha256_json(body),
            _prepared_projection_binding=prepared_projection_binding,
            _issuer=_DIRECT_RUNTIME_BINDING_ISSUER,
        )
        if self._direct_runtime_binding is not None:
            if self._direct_runtime_binding.as_dict() != binding.as_dict():
                raise RuntimeError(
                    "provider was already issued for another direct runtime"
                )
            return self._direct_runtime_binding
        self._direct_runtime_binding = binding
        return binding

    def get_review_partition_assignments(
        self,
        *,
        outer_fold: int,
        exact_outer_train_row_ids: tuple[int, ...],
    ) -> Mapping[int, Sequence[int]]:
        self._assert_current()
        assignments = _review_partition_assignments(
            self._plan,
            outer_fold=int(outer_fold),
        )
        supplied = _ordered_fit_rows(exact_outer_train_row_ids)
        flattened = tuple(row_id for rows in assignments.values() for row_id in rows)
        if set(supplied) != set(flattened):
            raise ValueError(
                "hierarchy runner outer-train rows differ from the " "role-neutral split plan"
            )
        return copy.deepcopy(assignments)

    def get_spent_evidence_catalog(
        self,
        *,
        outer_fold: int,
        review_round: int,
        exact_spent_row_ids: tuple[int, ...],
        exact_sealed_row_ids: tuple[int, ...],
        spent_texts: tuple[str, ...],
        spent_treatment: np.ndarray,
        spent_outcome: np.ndarray,
    ) -> RoleNeutralEvidenceCatalog:
        _bridge, proofs, catalogs = self._assert_current()
        matches = [
            scope
            for scope in self._plan.scopes
            if scope.scope_kind == "cumulative_spent"
            and scope.outer_fold == int(outer_fold)
            and scope.context_epoch == int(review_round)
        ]
        if len(matches) != 1:
            raise ValueError("requested cumulative role-neutral scope is absent")
        scope = matches[0]
        spent_ids = _ordered_fit_rows(exact_spent_row_ids)
        sealed_ids = _ordered_fit_rows(exact_sealed_row_ids)
        if spent_ids != scope.fit_row_ids:
            raise ValueError("hierarchy requested a noncanonical accumulated-spent " "row scope")
        if sealed_ids != scope.heldout_row_ids:
            raise ValueError("hierarchy requested a noncanonical still-sealed row scope")
        texts = tuple(spent_texts)
        treatment = np.asarray(spent_treatment)
        outcome = np.asarray(spent_outcome)
        if (
            len(texts) != len(spent_ids)
            or treatment.shape != (len(spent_ids),)
            or outcome.shape != (len(spent_ids),)
        ):
            raise ValueError("runtime spent projection columns are misaligned")
        if self._direct_runtime_binding is not None:
            # Complete text/T/Y values were authenticated once when the
            # provider issued this exact prepared-cohort runtime token.
            # Ordinary catalog access is row/shape authorized and does not
            # reread or rehash the same values for each family/review wrapper.
            self._direct_runtime_binding.as_dict()
            return catalogs[(scope.outer_fold, scope.context_epoch)]
        position = {row_id: index for index, row_id in enumerate(spent_ids)}
        owner = self._plan.physical_owner(scope.scope_id)
        if set(owner.fit_row_ids) != set(spent_ids):
            raise RuntimeError("cumulative scope no longer matches its physical fit")
        owner_positions = [position[row_id] for row_id in owner.fit_row_ids]
        observed = build_role_neutral_stage2_fit_projection_proof(
            plan_scientific_content_sha256=(self._plan.scientific_content_sha256),
            physical_owner_scope_id=owner.scope_id,
            fit_row_ids=owner.fit_row_ids,
            fit_texts=tuple(texts[index] for index in owner_positions),
            fit_treatment=treatment[owner_positions],
            fit_outcome=outcome[owner_positions],
        )
        if observed != proofs[owner.scope_id]:
            raise ValueError(
                "runtime spent row/text/treatment/outcome projection "
                "differs from the sealed role-neutral producer proof"
            )
        catalog = catalogs[(scope.outer_fold, scope.context_epoch)]
        return catalog

    def get_spent_evidence_inputs(
        self,
        **_kwargs: Any,
    ) -> Sequence[Any]:
        raise RuntimeError(
            "authenticated role-neutral Stage 2 requires direct prefit "
            "catalog consumption; raw-input and independent-refit fallback "
            "paths are forbidden"
        )


def _registry_shape(registry: Mapping[str, Any]) -> tuple[int, int, int]:
    outer = registry.get("outer_folds")
    if not isinstance(outer, list) or not outer:
        raise ValueError("handoff split registry has no outer folds")
    inner_counts = {
        len(row.get("inner_folds"))
        for row in outer
        if isinstance(row, Mapping) and isinstance(row.get("inner_folds"), list)
    }
    if len(inner_counts) != 1 or len(outer) != sum(isinstance(row, Mapping) for row in outer):
        raise ValueError("handoff split registry has inconsistent inner folds")
    inner_count = next(iter(inner_counts))
    row_count = int(registry.get("dataset_row_count", -1))
    if inner_count < 1 or row_count < 1:
        raise ValueError("handoff split registry is empty")
    return row_count, len(outer), inner_count


def _reconstruct_scope_plan(
    *,
    registry: Mapping[str, Any],
    plan_value: Mapping[str, Any],
) -> Stage1ScopePlan:
    registry_sha256 = _sha256_json(registry)
    row_count, outer_count, inner_count = _registry_shape(registry)
    if row_count != int(registry.get("dataset_row_count", -1)):
        raise RuntimeError("registry row-count validation changed")
    return validate_stage1_scope_plan(
        plan_value,
        registry=registry,
        registry_content_sha256=registry_sha256,
        global_seed=int(plan_value.get("global_seed")),
        physical_fit_identity=Stage1PhysicalFitIdentity.from_mapping(
            plan_value.get("physical_fit_identity") or {}
        ),
        gpu_ids=tuple(plan_value.get("gpu_ids") or ()),
        review_rounds=int(plan_value.get("review_rounds")),
        initial_training_partitions=int(plan_value.get("initial_training_partitions")),
        scope_workers_per_gpu=int(plan_value.get("scope_workers_per_gpu")),
        expected_outer_fold_count=outer_count,
        expected_inner_fold_count=inner_count,
    )


def _write_row_map(
    *,
    path: Path,
    prepared: Any,
    expected_row_count: int,
) -> tuple[str, int]:
    data = getattr(prepared, "data", None)
    options = getattr(prepared, "options", None)
    unit_id_column = getattr(options, "unit_id_column", None)
    if not isinstance(data, pd.DataFrame) or not isinstance(
        unit_id_column,
        str,
    ):
        raise TypeError(
            "reference handoff requires prepared data and its configured " "unit-ID column"
        )
    if unit_id_column not in data or len(data) != expected_row_count:
        raise ValueError("prepared row map differs from the split-registry cohort")
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(expected_row_count, dtype=np.int64),
            "unit_id": data[unit_id_column].to_numpy(copy=True),
        }
    )
    if frame["unit_id"].isna().any() or frame["unit_id"].duplicated().any():
        raise ValueError("prepared unit IDs must be complete and unique")
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("row-map staging file already exists")
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path, label="role-neutral row map")


def _validate_row_map(
    path: Path,
    registration: Mapping[str, Any],
    *,
    expected_row_count: int,
) -> tuple[str, int, pd.DataFrame]:
    digest, size = _validate_file_registration(
        path,
        registration,
        label="role-neutral row map",
    )
    if registration.get("content_sha256") != digest:
        raise ValueError("row-map content identity must authenticate full bytes")
    frame = pd.read_parquet(path)
    expected_rows = np.arange(expected_row_count, dtype=np.int64)
    if (
        list(frame.columns) != ["_oci_row_id", "unit_id"]
        or len(frame) != expected_row_count
        or not np.array_equal(
            frame["_oci_row_id"].to_numpy(dtype=np.int64),
            expected_rows,
        )
        or frame["unit_id"].isna().any()
        or frame["unit_id"].duplicated().any()
    ):
        raise ValueError("role-neutral row map is incomplete or substituted")
    return digest, size, frame


def _direct_handoff_scientific_body(
    *,
    prepared_request_sha256: str,
    registry_content_sha256: str,
    plan: Stage1ScopePlan,
    row_map_sha256: str,
    source_execution_content_sha256: str,
    provider_identity_sha256: str,
    semantic_member_batch_size: int,
    htr_aggregation_store_content_sha256: str,
    htr_call_plan_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(htr_call_plan_preflight, Mapping)
        or htr_call_plan_preflight.get("schema_version")
        != "production_htr_stage2_call_plan_preflight_v2"
    ):
        raise ValueError("direct handoff lacks its HTR call-plan preflight")
    return {
        "schema_version": ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_SCHEMA,
        "handoff_kind": ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
        "request_sha256": _require_sha256(
            prepared_request_sha256,
            label="prepared Stage 1 request",
        ),
        "registry_content_sha256": _require_sha256(
            registry_content_sha256,
            label="split registry",
        ),
        "scope_plan_scientific_content_sha256": (plan.scientific_content_sha256),
        "row_map_sha256": _require_sha256(
            row_map_sha256,
            label="row map",
        ),
        "source_role_neutral_execution_content_sha256": _require_sha256(
            source_execution_content_sha256,
            label="role-neutral execution",
        ),
        "stage2_provider_identity_sha256": _require_sha256(
            provider_identity_sha256,
            label="role-neutral Stage 2 provider",
        ),
        "htr_semantic_aggregation_store_content_sha256": _require_sha256(
            htr_aggregation_store_content_sha256,
            label="HTR semantic aggregation store",
        ),
        "htr_stage2_call_plan_preflight": copy.deepcopy(
            dict(htr_call_plan_preflight)
        ),
        "semantic_member_batch_size": _require_semantic_member_batch_size(
            semantic_member_batch_size
        ),
        "physical_fit_count": len(plan.physical_scopes),
        "logical_scope_count": len(plan.scopes),
        "deduplicated_fit_count": (len(plan.scopes) - len(plan.physical_scopes)),
        "portable_family_order": list(EVIDENCE_FAMILIES),
        "native_family_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
        "execution_locator_in_scientific_identity": False,
        "evidence_payloads_copied": False,
        "derived_htr_aggregate_payloads_materialized_here": True,
        "raw_htr_token_arrays_materialized_here": False,
        "raw_htr_chunk_atoms_model_facing": False,
        "legacy_bundle_build_invoked": False,
        "independent_stage1_refit_invoked": False,
        "all_ten_role_neutral_execution_is_exclusive_evidence_source": True,
        "text_truncation_applied": False,
        "lossy_evidence_selection_applied": False,
        "offline_handoff_validation_complete": True,
        "full_stage2_one_shot_runtime_complete": False,
    }


def _validate_direct_handoff_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("role-neutral direct handoff manifest must be a mapping")
    manifest = copy.deepcopy(dict(value))
    expected = {
        "schema_version",
        "handoff_kind",
        "request_sha256",
        "registry_content_sha256",
        "scope_plan_scientific_content_sha256",
        "row_map_sha256",
        "source_role_neutral_execution_content_sha256",
        "stage2_provider_identity_sha256",
        "htr_semantic_aggregation_store_content_sha256",
        "htr_stage2_call_plan_preflight",
        "semantic_member_batch_size",
        "physical_fit_count",
        "logical_scope_count",
        "deduplicated_fit_count",
        "portable_family_order",
        "native_family_order",
        "execution_locator_in_scientific_identity",
        "evidence_payloads_copied",
        "derived_htr_aggregate_payloads_materialized_here",
        "raw_htr_token_arrays_materialized_here",
        "raw_htr_chunk_atoms_model_facing",
        "legacy_bundle_build_invoked",
        "independent_stage1_refit_invoked",
        "all_ten_role_neutral_execution_is_exclusive_evidence_source",
        "text_truncation_applied",
        "lossy_evidence_selection_applied",
        "offline_handoff_validation_complete",
        "full_stage2_one_shot_runtime_complete",
        "content_sha256",
        "bundle_sha256",
    }
    scientific_body = {
        key: copy.deepcopy(child)
        for key, child in manifest.items()
        if key not in {"content_sha256", "bundle_sha256"}
    }
    bundle_body = {
        key: copy.deepcopy(child) for key, child in manifest.items() if key != "bundle_sha256"
    }
    digest_fields = (
        "request_sha256",
        "registry_content_sha256",
        "scope_plan_scientific_content_sha256",
        "row_map_sha256",
        "source_role_neutral_execution_content_sha256",
        "stage2_provider_identity_sha256",
        "htr_semantic_aggregation_store_content_sha256",
        "content_sha256",
        "bundle_sha256",
    )
    if (
        set(manifest) != expected
        or manifest.get("schema_version") != ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_SCHEMA
        or manifest.get("handoff_kind") != ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
        or any(
            _require_sha256(
                manifest.get(field),
                label=f"direct handoff {field}",
            )
            != manifest.get(field)
            for field in digest_fields
        )
        or manifest.get("content_sha256") != _sha256_json(scientific_body)
        or manifest.get("bundle_sha256") != _sha256_json(bundle_body)
        or manifest.get("portable_family_order") != list(EVIDENCE_FAMILIES)
        or manifest.get("native_family_order") != list(ACTIVE_STAGE1_CONCEPT_FAMILIES)
        or _require_semantic_member_batch_size(
            manifest.get("semantic_member_batch_size")
        )
        != manifest.get("semantic_member_batch_size")
        or manifest.get("execution_locator_in_scientific_identity") is not False
        or manifest.get("evidence_payloads_copied") is not False
        or manifest.get(
            "derived_htr_aggregate_payloads_materialized_here"
        )
        is not True
        or manifest.get("raw_htr_token_arrays_materialized_here") is not False
        or manifest.get("raw_htr_chunk_atoms_model_facing") is not False
        or manifest.get("legacy_bundle_build_invoked") is not False
        or manifest.get("independent_stage1_refit_invoked") is not False
        or manifest.get("all_ten_role_neutral_execution_is_exclusive_evidence_source") is not True
        or manifest.get("text_truncation_applied") is not False
        or manifest.get("lossy_evidence_selection_applied") is not False
        or manifest.get("offline_handoff_validation_complete") is not True
        or manifest.get("full_stage2_one_shot_runtime_complete") is not False
        or not isinstance(
            manifest.get("htr_stage2_call_plan_preflight"),
            Mapping,
        )
        or manifest["htr_stage2_call_plan_preflight"].get(
            "schema_version"
        )
        != "production_htr_stage2_call_plan_preflight_v2"
        or isinstance(manifest.get("physical_fit_count"), bool)
        or int(manifest.get("physical_fit_count", -1)) < 1
        or isinstance(manifest.get("logical_scope_count"), bool)
        or int(manifest.get("logical_scope_count", -1))
        < int(manifest.get("physical_fit_count", -1))
        or manifest.get("deduplicated_fit_count")
        != int(manifest["logical_scope_count"]) - int(manifest["physical_fit_count"])
    ):
        raise ValueError("role-neutral direct handoff manifest is invalid")
    return manifest


def _validate_direct_handoff_locator(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("role-neutral direct handoff locator must be a mapping")
    locator = copy.deepcopy(dict(value))
    expected = {
        "schema_version",
        "handoff_kind",
        "scientific_manifest",
        "metadata_files",
        "role_neutral_execution",
        "references_only",
        "evidence_payloads_materialized_here",
        "derived_htr_aggregate_payloads_materialized_here",
        "raw_htr_token_arrays_materialized_here",
        "content_sha256",
    }
    body = {key: copy.deepcopy(child) for key, child in locator.items() if key != "content_sha256"}
    if (
        set(locator) != expected
        or locator.get("schema_version") != ROLE_NEUTRAL_STAGE1_REFERENCE_LOCATOR_SCHEMA
        or locator.get("handoff_kind") != ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
        or locator.get("references_only") is not True
        or locator.get("evidence_payloads_materialized_here") is not False
        or locator.get(
            "derived_htr_aggregate_payloads_materialized_here"
        )
        is not True
        or locator.get("raw_htr_token_arrays_materialized_here") is not False
        or locator.get("content_sha256") != _sha256_json(body)
        or not isinstance(locator.get("scientific_manifest"), Mapping)
        or not isinstance(locator.get("metadata_files"), Mapping)
        or not isinstance(locator.get("role_neutral_execution"), Mapping)
    ):
        raise ValueError("role-neutral direct handoff locator is invalid")
    return locator


def load_reference_only_role_neutral_stage1_handoff(
    bundle_manifest_path: Path | str,
    *,
    role_neutral_execution_root: Path | str | None = None,
) -> Any:
    """Freshly reopen a reference-only handoff and its exclusive provider."""

    supplied_manifest = Path(bundle_manifest_path)
    if (
        not supplied_manifest.is_absolute()
        or supplied_manifest.is_symlink()
        or supplied_manifest.name != ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST
    ):
        raise ValueError("role-neutral direct handoff requires its absolute canonical manifest")
    manifest_path = supplied_manifest.resolve(strict=True)
    root = manifest_path.parent
    manifest = _validate_direct_handoff_manifest(
        _read_closed_json_file(
            manifest_path,
            label="role-neutral direct scientific manifest",
        )
    )
    locator = _validate_direct_handoff_locator(
        _read_closed_json_file(
            root / ROLE_NEUTRAL_STAGE1_REFERENCE_LOCATOR,
            label="role-neutral direct locator attestation",
        )
    )
    scientific_registration = locator["scientific_manifest"]
    if (
        scientific_registration.get("relative_path") != ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST
        or scientific_registration.get("content_sha256") != manifest["content_sha256"]
    ):
        raise ValueError("locator substituted the direct scientific manifest")
    _validate_file_registration(
        manifest_path,
        scientific_registration,
        label="role-neutral direct scientific manifest",
    )

    metadata = locator["metadata_files"]
    expected_metadata = {
        "split_registry": ROLE_NEUTRAL_STAGE1_REFERENCE_REGISTRY,
        "scope_plan": ROLE_NEUTRAL_STAGE1_REFERENCE_PLAN,
        "row_map": ROLE_NEUTRAL_STAGE1_REFERENCE_ROW_MAP,
        "htr_semantic_aggregation": (
            f"{ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_DIRECTORY}/"
            f"{ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_MANIFEST}"
        ),
    }
    if set(metadata) != set(expected_metadata):
        raise ValueError("direct handoff metadata inventory is incomplete")
    for name, expected_relative in expected_metadata.items():
        registration = metadata[name]
        if (
            not isinstance(registration, Mapping)
            or registration.get("relative_path") != expected_relative
        ):
            raise ValueError("direct handoff metadata path was substituted")

    registry_path = root / ROLE_NEUTRAL_STAGE1_REFERENCE_REGISTRY
    plan_path = root / ROLE_NEUTRAL_STAGE1_REFERENCE_PLAN
    registry_registration = metadata["split_registry"]
    plan_registration = metadata["scope_plan"]
    _validate_file_registration(
        registry_path,
        registry_registration,
        label="role-neutral split registry",
    )
    _validate_file_registration(
        plan_path,
        plan_registration,
        label="role-neutral scope plan",
    )
    registry = _read_closed_json_file(
        registry_path,
        label="role-neutral split registry",
    )
    plan_value = _read_closed_json_file(
        plan_path,
        label="role-neutral scope plan",
    )
    registry_sha256 = _sha256_json(registry)
    if (
        registry_registration.get("content_sha256") != registry_sha256
        or registry_sha256 != manifest["registry_content_sha256"]
    ):
        raise ValueError("direct handoff split-registry identity changed")
    plan = _reconstruct_scope_plan(
        registry=registry,
        plan_value=plan_value,
    )
    if (
        plan_registration.get("content_sha256") != plan.content_sha256
        or plan.scientific_content_sha256 != manifest["scope_plan_scientific_content_sha256"]
        or len(plan.physical_scopes) != manifest["physical_fit_count"]
        or len(plan.scopes) != manifest["logical_scope_count"]
    ):
        raise ValueError("direct handoff scope-plan identity changed")
    row_map_path = root / ROLE_NEUTRAL_STAGE1_REFERENCE_ROW_MAP
    if metadata["row_map"].get("content_sha256") != manifest["row_map_sha256"]:
        raise ValueError("direct handoff row-map identity changed")
    aggregation_store_root = (
        root / ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_DIRECTORY
    )
    aggregation_registration = metadata[
        "htr_semantic_aggregation"
    ]
    _validate_file_registration(
        aggregation_store_root
        / ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_MANIFEST,
        aggregation_registration,
        label="HTR semantic aggregation store manifest",
    )
    if (
        aggregation_registration.get("content_sha256")
        != manifest[
            "htr_semantic_aggregation_store_content_sha256"
        ]
    ):
        raise ValueError("direct handoff HTR aggregate identity changed")

    execution_reference = locator["role_neutral_execution"]
    if set(execution_reference) != {
        "relative_root_locator",
        "manifest_relative_path",
        "manifest_sha256",
        "manifest_size_bytes",
        "manifest_content_sha256",
        "execution_tree_materialized_here",
    }:
        raise ValueError("role-neutral execution reference is malformed")
    if (
        execution_reference.get("manifest_relative_path") != ROLE_NEUTRAL_EXECUTION_MANIFEST
        or execution_reference.get("execution_tree_materialized_here") is not False
    ):
        raise ValueError("role-neutral execution reference is not reference-only")
    if role_neutral_execution_root is None:
        relative_locator = execution_reference.get("relative_root_locator")
        if (
            not isinstance(relative_locator, str)
            or not relative_locator
            or Path(relative_locator).is_absolute()
        ):
            raise ValueError("role-neutral execution locator must be relative")
        execution_root = (root / relative_locator).resolve(strict=True)
    else:
        supplied_root = Path(role_neutral_execution_root)
        if not supplied_root.is_absolute() or supplied_root.is_symlink():
            raise ValueError("execution-root override must be canonical and absolute")
        execution_root = supplied_root.resolve(strict=True)
    execution_manifest_path = execution_root / ROLE_NEUTRAL_EXECUTION_MANIFEST
    execution_digest, execution_size = _sha256_file(
        execution_manifest_path,
        label="role-neutral execution manifest",
    )
    if execution_digest != execution_reference.get(
        "manifest_sha256"
    ) or execution_size != execution_reference.get("manifest_size_bytes"):
        raise ValueError("referenced role-neutral execution manifest changed")
    execution_manifest = _read_closed_json_file(
        execution_manifest_path,
        label="role-neutral execution manifest",
    )
    execution_content_sha256 = _require_sha256(
        execution_manifest.get("content_sha256"),
        label="role-neutral execution content",
    )
    if (
        execution_content_sha256 != execution_reference.get("manifest_content_sha256")
        or execution_content_sha256 != manifest["source_role_neutral_execution_content_sha256"]
    ):
        raise ValueError("referenced role-neutral execution identity changed")
    provider = AuthenticatedRoleNeutralStage2Provider(
        execution_root=execution_root,
        plan=plan,
        execution_manifest=execution_manifest,
        semantic_member_batch_size=manifest[
            "semantic_member_batch_size"
        ],
        htr_aggregation_store_root=aggregation_store_root,
        authenticated_row_map_path=row_map_path,
        authenticated_row_map_registration=metadata["row_map"],
        prepared_request_sha256=manifest["request_sha256"],
    )
    provider_identity = provider.identity()
    if provider_identity.get("identity_sha256") != manifest["stage2_provider_identity_sha256"]:
        raise ValueError("direct handoff Stage 2 provider identity changed")
    if (
        provider_identity.get("htr_stage2_call_plan_preflight")
        != manifest["htr_stage2_call_plan_preflight"]
    ):
        raise ValueError("direct handoff HTR call-plan preflight changed")

    from .production_all_evidence_workflow import (
        RoleNeutralStage1HandoffPublication,
    )

    return RoleNeutralStage1HandoffPublication(
        bundle_manifest_path=manifest_path,
        source_role_neutral_execution_content_sha256=(execution_content_sha256),
        legacy_bundle_build_invoked=False,
        all_ten_role_neutral_execution_is_exclusive_evidence_source=True,
        handoff_kind=ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
        handoff_scientific_content_sha256=manifest["content_sha256"],
        bundle_sha256=manifest["bundle_sha256"],
        stage2_provider=provider,
    )


@dataclass(frozen=True)
class ReferenceOnlyRoleNeutralStage1HandoffPublisher:
    """Publish compact metadata and references, then validate them afresh."""

    semantic_member_batch_size: int

    def __post_init__(self) -> None:
        _require_semantic_member_batch_size(
            self.semantic_member_batch_size
        )

    def __call__(
        self,
        *,
        target_dir: Path,
        prepared: Any,
        role_neutral_execution_root: Path,
        role_neutral_execution_manifest: Mapping[str, Any],
    ) -> Any:
        target = Path(target_dir)
        if not target.is_absolute():
            raise ValueError("role-neutral direct handoff target must be absolute")
        if target.exists() or target.is_symlink():
            raise FileExistsError("role-neutral direct handoff target must be fresh")
        if target.parent.resolve(strict=True) != target.parent:
            raise ValueError("role-neutral direct handoff parent must be canonical")
        plan = getattr(prepared, "stage1_scope_plan", None)
        registry = getattr(prepared, "registry", None)
        registry_content_sha256 = str(getattr(prepared, "registry_content_sha256", ""))
        prepared_request_sha256 = str(getattr(prepared, "request_sha256", ""))
        if not isinstance(plan, Stage1ScopePlan) or not isinstance(
            registry,
            Mapping,
        ):
            raise TypeError("prepared Stage 1 input lacks its scope plan or registry")
        registry_value = copy.deepcopy(dict(registry))
        if (
            _sha256_json(registry_value) != registry_content_sha256
            or plan.registry_content_sha256 != registry_content_sha256
        ):
            raise ValueError("prepared split-registry identity changed")
        execution_root = Path(role_neutral_execution_root).resolve(strict=True)
        source_execution_content_sha256 = _require_sha256(
            role_neutral_execution_manifest.get("content_sha256"),
            label="role-neutral execution",
        )
        row_count, _outer_count, _inner_count = _registry_shape(registry_value)

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.",
                dir=target.parent,
            )
        )
        try:
            bridge = validate_role_neutral_stage2_bridge(
                execution_root=execution_root,
                plan=plan,
                execution_manifest=role_neutral_execution_manifest,
            )
            aggregation_store = _build_htr_semantic_aggregation_store(
                root=(
                    staging
                    / ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_DIRECTORY
                ),
                execution_root=execution_root,
                execution_content_sha256=(
                    source_execution_content_sha256
                ),
                plan=plan,
                bridge=bridge,
            )
            provider = AuthenticatedRoleNeutralStage2Provider(
                execution_root=execution_root,
                plan=plan,
                execution_manifest=role_neutral_execution_manifest,
                semantic_member_batch_size=self.semantic_member_batch_size,
                htr_aggregation_store_root=aggregation_store.root,
            )
            provider_identity = provider.identity()
            registry_path = staging / ROLE_NEUTRAL_STAGE1_REFERENCE_REGISTRY
            plan_path = staging / ROLE_NEUTRAL_STAGE1_REFERENCE_PLAN
            row_map_path = staging / ROLE_NEUTRAL_STAGE1_REFERENCE_ROW_MAP
            _write_new_json(registry_path, registry_value)
            _write_new_json(plan_path, plan.as_dict())
            row_map_sha256, _row_map_size = _write_row_map(
                path=row_map_path,
                prepared=prepared,
                expected_row_count=row_count,
            )
            scientific_body = _direct_handoff_scientific_body(
                prepared_request_sha256=prepared_request_sha256,
                registry_content_sha256=registry_content_sha256,
                plan=plan,
                row_map_sha256=row_map_sha256,
                source_execution_content_sha256=(source_execution_content_sha256),
                provider_identity_sha256=provider_identity["identity_sha256"],
                semantic_member_batch_size=self.semantic_member_batch_size,
                htr_aggregation_store_content_sha256=(
                    aggregation_store.content_sha256
                ),
                htr_call_plan_preflight=(
                    aggregation_store.preflight_report
                ),
            )
            content_sha256 = _sha256_json(scientific_body)
            manifest_without_bundle = {
                **scientific_body,
                "content_sha256": content_sha256,
            }
            manifest = {
                **manifest_without_bundle,
                "bundle_sha256": _sha256_json(manifest_without_bundle),
            }
            manifest_path = staging / ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST
            _write_new_json(manifest_path, manifest)

            execution_manifest_path = execution_root / ROLE_NEUTRAL_EXECUTION_MANIFEST
            execution_manifest_sha256, execution_manifest_size = _sha256_file(
                execution_manifest_path,
                label="role-neutral execution manifest",
            )
            locator_body = {
                "schema_version": (ROLE_NEUTRAL_STAGE1_REFERENCE_LOCATOR_SCHEMA),
                "handoff_kind": (ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND),
                "scientific_manifest": _file_registration(
                    manifest_path,
                    root=staging,
                    content_sha256=content_sha256,
                ),
                "metadata_files": {
                    "split_registry": _file_registration(
                        registry_path,
                        root=staging,
                        content_sha256=registry_content_sha256,
                    ),
                    "scope_plan": _file_registration(
                        plan_path,
                        root=staging,
                        content_sha256=plan.content_sha256,
                    ),
                    "row_map": _file_registration(
                        row_map_path,
                        root=staging,
                        content_sha256=row_map_sha256,
                    ),
                    "htr_semantic_aggregation": _file_registration(
                        aggregation_store.root
                        / ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_MANIFEST,
                        root=staging,
                        content_sha256=(
                            aggregation_store.content_sha256
                        ),
                    ),
                },
                "role_neutral_execution": {
                    "relative_root_locator": os.path.relpath(
                        execution_root,
                        start=target,
                    ),
                    "manifest_relative_path": (ROLE_NEUTRAL_EXECUTION_MANIFEST),
                    "manifest_sha256": execution_manifest_sha256,
                    "manifest_size_bytes": execution_manifest_size,
                    "manifest_content_sha256": (source_execution_content_sha256),
                    "execution_tree_materialized_here": False,
                },
                "references_only": True,
                "evidence_payloads_materialized_here": False,
                "derived_htr_aggregate_payloads_materialized_here": True,
                "raw_htr_token_arrays_materialized_here": False,
            }
            locator = {
                **locator_body,
                "content_sha256": _sha256_json(locator_body),
            }
            _write_new_json(
                staging / ROLE_NEUTRAL_STAGE1_REFERENCE_LOCATOR,
                locator,
            )
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return load_reference_only_role_neutral_stage1_handoff(
            target / ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST,
        )


@dataclass(frozen=True)
class FailClosedRoleNeutralStage2HandoffPublisher:
    """Workflow-compatible publisher that refuses unsafe legacy conversion."""

    requirements: RoleNeutralStage2LoaderRequirements = ROLE_NEUTRAL_STAGE2_LOADER_REQUIREMENTS

    def __call__(
        self,
        *,
        target_dir: Path,
        prepared: Any,
        role_neutral_execution_root: Path,
        role_neutral_execution_manifest: Mapping[str, Any],
    ) -> Any:
        target = Path(target_dir)
        if not target.is_absolute():
            raise ValueError("role-neutral Stage 2 target must be absolute")
        if target.exists() or target.is_symlink():
            raise FileExistsError("role-neutral Stage 2 target must be fresh")
        if target.parent.resolve(strict=True) != target.parent:
            raise ValueError("role-neutral Stage 2 target parent must be canonical")
        plan = getattr(prepared, "stage1_scope_plan", None)
        if not isinstance(plan, Stage1ScopePlan):
            raise TypeError("prepared Stage 1 context lacks its typed scope plan")
        bridge = validate_role_neutral_stage2_bridge(
            execution_root=role_neutral_execution_root,
            plan=plan,
            execution_manifest=role_neutral_execution_manifest,
        )
        # No directory or file is created before this typed incompatibility.
        raise RoleNeutralStage2LoaderContractUnavailable(
            bridge=bridge,
            requirements=self.requirements,
        )


__all__ = [
    "AuthenticatedPreparedCohortProjectionBinding",
    "AuthenticatedRoleNeutralHierarchyExecutionAuthorization",
    "AuthenticatedRoleNeutralStage2RuntimeBinding",
    "AuthenticatedRoleNeutralStage2Bridge",
    "AuthenticatedRoleNeutralStage2Provider",
    "FailClosedRoleNeutralStage2HandoffPublisher",
    "ReferenceOnlyRoleNeutralStage1HandoffPublisher",
    "ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND",
    "ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_SCHEMA",
    "ROLE_NEUTRAL_STAGE1_REFERENCE_LOCATOR",
    "ROLE_NEUTRAL_STAGE1_REFERENCE_LOCATOR_SCHEMA",
    "ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST",
    "ROLE_NEUTRAL_STAGE1_REFERENCE_PLAN",
    "ROLE_NEUTRAL_STAGE1_REFERENCE_REGISTRY",
    "ROLE_NEUTRAL_STAGE1_REFERENCE_ROW_MAP",
    "ROLE_NEUTRAL_STAGE2_BRIDGE_SCHEMA",
    "ROLE_NEUTRAL_STAGE2_COMPONENT_EXPORT_INDEX_SCHEMA",
    "ROLE_NEUTRAL_STAGE2_DIRECT_HANDOFF_SCHEMA",
    "ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_PROOF_SCHEMA",
    "ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD",
    "ROLE_NEUTRAL_STAGE2_LOADER_REQUIREMENTS",
    "ROLE_NEUTRAL_STAGE2_LOADER_REQUIREMENTS_SCHEMA",
    "RoleNeutralStage2LoaderContractUnavailable",
    "RoleNeutralStage2LoaderRequirements",
    "RoleNeutralStage2LogicalContext",
    "RoleNeutralStage2PhysicalFit",
    "RoleNeutralStage2ProjectionProofUnavailable",
    "build_role_neutral_stage2_fit_projection_proof",
    "authorize_reference_only_role_neutral_hierarchy_execution",
    "load_reference_only_role_neutral_stage1_handoff",
    "validate_authenticated_prepared_projection_binding",
    "validate_authenticated_role_neutral_stage2_runtime_binding",
    "validate_role_neutral_stage2_fit_projection_proof",
    "validate_role_neutral_stage2_bridge",
]
