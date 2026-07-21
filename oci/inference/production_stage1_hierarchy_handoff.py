"""Authenticated Stage-1-to-hierarchy handoff for an arbitrary cohort.

This module deliberately does not translate a sealed Stage 1 bundle back into
the historical legacy/TF-IDF command-line inputs.  That compatibility route
would authenticate the bundle and then let hierarchical discovery refit a
different, independently generated accumulated-spent schedule.

The production boundary defined here is stricter:

* review partitions are the canonical inner held-out partitions, in registry
  order;
* Stage 1 must prefit every cumulative-spent context used by the hierarchy;
* every context persists one lossless all-ten catalog and one all-ten native
  fit-proof bundle;
* the index, catalogs, proof bundles, model artifacts, and execution records
  are root-registered by the Stage 1 manifest; and
* the provider has no ``get_spent_evidence_inputs`` compatibility fallback.

Schema authentication is not genuine native-proof validation. Catalog serving
and internal digest authorization therefore remain hard-gated until the family
records are reconstructed with the native binders and the same-process one-shot
runner binds this exact provider and its exact runtime objects into preparation.

The current production Stage 1 builder does not yet emit this graph.  Loading a
current bundle through :func:`load_production_stage1_hierarchy_handoff` therefore
fails closed before a hierarchy runner or remote client can be constructed.
"""

from __future__ import annotations

import copy
import ctypes
import fcntl
import hashlib
import json
import os
import re
import secrets
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)
from .all_evidence_fusion import FoldEvidenceProvenance
from .approved_hierarchical_discovery_batch import (
    ApprovedHierarchicalDiscoveryBatchCoordinator,
    ApprovedHierarchicalDiscoveryBatchResult,
)
from .lossless_stage1_evidence_catalog import (
    NON_GROUNDING_SUMMARY_SCHEMA_VERSION,
    ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
    NonGroundingNumericalSummary,
    RoleNeutralEvidenceCatalog,
    Stage1EvidenceAtom,
    validate_role_neutral_catalog,
)
from .production_stage1_bundle import (
    _HEX_SHA256,
    _sha256_json,
)
from .production_stage1_hierarchy_loader import (
    AuthenticatedStage1HierarchyInputs,
    _BundleRootCapability,
    _StableFileSnapshot,
    _contract_registry,
    _load_json_snapshot,
    _registered_file_hash,
    _registered_json,
    load_authenticated_stage1_bundle_for_hierarchy,
)
from .production_stage1_hierarchy_contract import (
    validate_production_stage1_hierarchy_contract_identity,
)
from .stage1_cumulative_spent_evidence import (
    CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA,
    CUMULATIVE_SPENT_CACHE_REPLAY,
    CUMULATIVE_SPENT_REFIT,
    CUMULATIVE_SPENT_REQUEST_SCHEMA,
    _validate_fit_audit,
    cumulative_spent_data_projection_sha256,
)
from .stage1_exact_inner_evidence import (
    CanonicalStage1SplitRegistry,
    Stage1FitRow,
    row_order_fingerprint,
)
from .stage1_exact_inner_family_adapters import family_payload_from_catalog

STAGE1_HIERARCHY_SPENT_CONTRACT_SCHEMA = "production_stage1_hierarchy_spent_contract_v2"
STAGE1_HIERARCHY_SPENT_INDEX_SCHEMA = "production_stage1_hierarchy_spent_index_v2"
STAGE1_HIERARCHY_SPENT_SCHEDULE_SCHEMA = "production_stage1_canonical_hierarchy_spent_schedule_v1"
STAGE1_HIERARCHY_SPENT_PROOF_BUNDLE_SCHEMA = "production_stage1_hierarchy_spent_fit_proof_bundle_v2"
STAGE1_HIERARCHY_SPENT_FAMILY_PROOF_SCHEMA = "production_stage1_hierarchy_spent_family_fit_proof_v2"
STAGE1_HIERARCHY_NATIVE_MODEL_DESCRIPTOR_SCHEMA = (
    "production_stage1_cumulative_native_model_descriptor_v1"
)
STAGE1_HIERARCHY_SPENT_REQUEST_SCHEMA = CUMULATIVE_SPENT_REQUEST_SCHEMA
STAGE1_HIERARCHY_PROVIDER_IDENTITY_SCHEMA = (
    "authenticated_production_stage1_hierarchy_catalog_provider_v4"
)
STAGE1_HIERARCHY_HANDOFF_SCHEMA = "authenticated_production_stage1_hierarchy_handoff_v5"
INTERNAL_HIERARCHY_AUTHORIZATION_SCHEMA = "production_internal_hierarchy_execution_authorization_v5"
INTERNAL_HIERARCHY_PREPARATION_BINDING_SCHEMA = (
    "production_internal_hierarchy_preparation_binding_v2"
)
HIERARCHICAL_PREPARATION_INPUT_WRAPPER_SCHEMA = (
    "hierarchical_all_evidence_runner_preparation_input_v2"
)
HIERARCHICAL_PREPARATION_BATCH_WRAPPER_SCHEMA = "hierarchical_all_evidence_runner_batch_packet_v1"

# The cumulative-spent component runners now emit native model/source artifacts,
# typed family bundles, and descriptor-anchored proof objects.  Loading a v2
# root graph reconstructs and validates that complete byte graph before a
# provider can be created.  This implementation gate is intentionally immutable
# and is distinct from the cohort-level certification flag below: the genuine
# one-shot path must be executable in order to perform the E2E that earns that
# certification.
NATIVE_PROOF_VALIDATION_SUBSTRATE_READY = True

# This reports only the final, frozen-source one-shot E2E certification state.
# It must remain false until a genuine arbitrary-cohort, all-ten, no-approval
# execution succeeds.  It does not disable the authenticated candidate path.
GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY = False

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TFIDF_FAMILIES = frozenset({TFIDF_SEMANTIC_RETRIEVAL, TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS})
_REPLAY_ARGUMENT_PREFIXES = (
    "--read-only-review-spent-evidence-cache=",
    "--read-only-context-fit-cache-index=",
)
_CANONICAL_COORDINATOR_EXECUTE = ApprovedHierarchicalDiscoveryBatchCoordinator.execute
_CANONICAL_COORDINATOR_ASSERT_UNCHANGED = (
    ApprovedHierarchicalDiscoveryBatchCoordinator._assert_unchanged
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    result = str(value or "")
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return result


def _load_hash_wrapper_snapshot(
    snapshot: _StableFileSnapshot,
    *,
    label: str,
    expected_schema_version: str,
    expected_content_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _load_json_snapshot(snapshot, label=label)
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"schema_version", "content_sha256", "body"}
        or raw.get("schema_version") != expected_schema_version
        or not isinstance(raw.get("body"), Mapping)
    ):
        raise ValueError(f"{label} is not the pinned closed hash-wrapper schema")
    body = copy.deepcopy(dict(raw["body"]))
    declared = _require_sha256(raw.get("content_sha256"), label=f"{label} content hash")
    if _sha(body) != declared:
        raise ValueError(f"{label} content hash is invalid")
    if expected_content_sha256 is not None and declared != _require_sha256(
        expected_content_sha256,
        label=f"expected {label} content hash",
    ):
        raise ValueError(f"{label} differs from its prepared content binding")
    return copy.deepcopy(dict(raw)), body


def _ordered_unique_rows(values: Sequence[Any], *, label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{label} must be a sequence of integer row IDs")
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains a non-integer row ID") from exc
    if not result or any(value < 0 for value in result) or len(set(result)) != len(result):
        raise ValueError(f"{label} must be non-empty, nonnegative, and unique")
    return result


@dataclass(frozen=True)
class _AuthenticatedReplaySource:
    prefix: str
    argument: str
    path: Path
    sha256: str
    snapshot: _StableFileSnapshot
    snapshot_descriptor: int
    execution_argument: str


def _sealed_replay_snapshot_descriptor(snapshot: _StableFileSnapshot, *, label: str) -> int:
    name = f"stage1-replay-{hashlib.sha256(label.encode()).hexdigest()[:16]}"
    flags = getattr(os, "MFD_CLOEXEC", 0x0001) | getattr(os, "MFD_ALLOW_SEALING", 0x0002)
    if hasattr(os, "memfd_create"):
        descriptor = os.memfd_create(name, flags)
    else:
        # Some hermetic CPython builds omit the Linux wrapper and constants even
        # though the host kernel and libc expose memfd_create(2).  Use libc's
        # exact syscall wrapper; never downgrade immutable replay to a pathname.
        libc = ctypes.CDLL(None, use_errno=True)
        creator = getattr(libc, "memfd_create", None)
        if creator is None:
            raise RuntimeError("sealed replay snapshots require memfd_create")
        creator.argtypes = (ctypes.c_char_p, ctypes.c_uint)
        creator.restype = ctypes.c_int
        descriptor = int(creator(name.encode("utf-8"), flags))
        if descriptor < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
    try:
        view = memoryview(snapshot.payload)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = (
            getattr(fcntl, "F_SEAL_SEAL", 0x0001)
            | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
            | getattr(fcntl, "F_SEAL_GROW", 0x0004)
            | getattr(fcntl, "F_SEAL_WRITE", 0x0008)
        )
        fcntl.fcntl(descriptor, getattr(fcntl, "F_ADD_SEALS", 1033), seals)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _validated_authoritative_replay_arguments(
    value: Any,
    *,
    preparation_root: _BundleRootCapability,
) -> tuple[_AuthenticatedReplaySource, ...]:
    """Authenticate legacy cross-process replay registrations for audit tooling.

    This helper is not accepted by the production same-process capability or
    one-shot API and cannot mint production execution authority.
    """

    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("hierarchy preparation has no authoritative replay argument list")
    arguments = tuple(value)
    if len(arguments) != len(set(arguments)):
        raise ValueError("hierarchy authoritative replay arguments cannot contain duplicates")
    registrations: dict[str, _AuthenticatedReplaySource] = {}
    try:
        for argument in arguments:
            prefix = next(
                (
                    candidate
                    for candidate in _REPLAY_ARGUMENT_PREFIXES
                    if argument.startswith(candidate)
                ),
                None,
            )
            if prefix is None:
                raise ValueError("hierarchy authoritative replay argument is not read-only")
            if prefix in registrations:
                raise ValueError("hierarchy authoritative replay argument kind is duplicated")
            registration = argument[len(prefix) :]
            try:
                source_path, digest = registration.rsplit("::", 1)
            except ValueError as exc:
                raise ValueError(
                    "hierarchy authoritative replay argument lacks a registered SHA-256"
                ) from exc
            requested = Path(source_path)
            if not source_path or not requested.is_absolute() or _SHA256.fullmatch(digest) is None:
                raise ValueError("hierarchy authoritative replay registration is malformed")
            absolute = Path(os.path.abspath(os.fspath(requested)))
            try:
                relative = absolute.relative_to(preparation_root.path)
            except ValueError as exc:
                raise ValueError("hierarchy replay source is outside the preparation root") from exc
            snapshot = preparation_root.snapshot(
                relative.as_posix(),
                label=f"hierarchy replay source {prefix}",
            )
            if snapshot.sha256 != digest:
                raise ValueError("hierarchy replay source bytes differ from the registered digest")
            snapshot_descriptor = _sealed_replay_snapshot_descriptor(snapshot, label=prefix)
            registrations[prefix] = _AuthenticatedReplaySource(
                prefix=prefix,
                argument=argument,
                path=snapshot.path,
                sha256=digest,
                snapshot=snapshot,
                snapshot_descriptor=snapshot_descriptor,
                execution_argument=(f"{prefix}/proc/self/fd/{snapshot_descriptor}::{digest}"),
            )
        if set(registrations) != set(_REPLAY_ARGUMENT_PREFIXES):
            raise ValueError("hierarchy replay registrations must cover each read-only source once")
    except Exception:
        for source in registrations.values():
            os.close(source.snapshot_descriptor)
        raise
    return tuple(registrations[prefix] for prefix in _REPLAY_ARGUMENT_PREFIXES)


@dataclass(frozen=True)
class CanonicalHierarchySpentScope:
    outer_fold: int
    context_epoch: int
    provider_inner_fold: int
    spent_partition_ids: tuple[int, ...]
    sealed_partition_ids: tuple[int, ...]
    spent_row_ids: tuple[int, ...]
    sealed_row_ids: tuple[int, ...]
    split_fingerprint: str

    @property
    def scope_id(self) -> str:
        return f"outer_{self.outer_fold:03d}_hierarchy_epoch_{self.context_epoch:03d}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "outer_fold": self.outer_fold,
            "context_epoch": self.context_epoch,
            "provider_inner_fold": self.provider_inner_fold,
            "spent_partition_ids": list(self.spent_partition_ids),
            "sealed_partition_ids": list(self.sealed_partition_ids),
            "spent_row_ids": list(self.spent_row_ids),
            "sealed_row_ids": list(self.sealed_row_ids),
            "spent_row_order_fingerprint": row_order_fingerprint(self.spent_row_ids),
            "sealed_row_order_fingerprint": row_order_fingerprint(self.sealed_row_ids),
            "split_fingerprint": self.split_fingerprint,
        }


@dataclass(frozen=True)
class CanonicalHierarchySpentSchedule:
    split_registry_sha256: str
    review_rounds: int
    initial_spent_partition_count: int
    partitions_by_outer_fold: Mapping[int, Mapping[int, tuple[int, ...]]]
    scopes: tuple[CanonicalHierarchySpentScope, ...]
    schedule_sha256: str

    @classmethod
    def build(
        cls,
        *,
        registry: CanonicalStage1SplitRegistry,
        review_rounds: int,
    ) -> "CanonicalHierarchySpentSchedule":
        if not isinstance(registry, CanonicalStage1SplitRegistry):
            raise TypeError("registry must be CanonicalStage1SplitRegistry")
        rounds = int(review_rounds)
        if rounds < 1:
            raise ValueError("hierarchical adaptive review requires at least one round")
        # Exactly three initial partitions mirrors the production review
        # contract and prevents a caller from silently changing the amount of
        # evidence available to initial discovery.
        if registry.inner_fold_count != rounds + 3:
            raise ValueError(
                "the canonical Stage 1 inner-fold count must equal review_rounds + 3 "
                "so the hierarchy has exactly three initial-spent partitions"
            )

        partitions: dict[int, Mapping[int, tuple[int, ...]]] = {}
        scopes: list[CanonicalHierarchySpentScope] = []
        for outer in registry.outer_splits:
            by_partition = {
                int(inner.inner_fold): tuple(map(int, inner.heldout_row_ids))
                for inner in outer.inner_splits
            }
            expected_partition_ids = tuple(range(1, registry.inner_fold_count + 1))
            if tuple(sorted(by_partition)) != expected_partition_ids:
                raise ValueError("canonical inner partitions are incomplete or reordered")
            flattened = [
                row_id
                for partition_id in expected_partition_ids
                for row_id in by_partition[partition_id]
            ]
            if len(flattened) != len(set(flattened)) or set(flattened) != set(outer.train_row_ids):
                raise ValueError("canonical inner held-outs do not partition outer train")
            partitions[int(outer.outer_fold)] = MappingProxyType(dict(by_partition))
            for epoch in range(rounds):
                spent_partition_ids = expected_partition_ids[: 3 + epoch]
                sealed_partition_ids = expected_partition_ids[3 + epoch :]
                spent_rows = tuple(
                    row_id
                    for partition_id in spent_partition_ids
                    for row_id in by_partition[partition_id]
                )
                sealed_rows = tuple(
                    row_id
                    for partition_id in sealed_partition_ids
                    for row_id in by_partition[partition_id]
                )
                provenance = FoldEvidenceProvenance(
                    outer_fold=int(outer.outer_fold),
                    train_row_ids=spent_rows,
                    heldout_row_ids=sealed_rows,
                    scope="inner_train",
                    inner_fold=epoch + 1,
                    artifact_id=(f"production-stage1-hierarchy-{int(outer.outer_fold)}-{epoch}"),
                )
                scopes.append(
                    CanonicalHierarchySpentScope(
                        outer_fold=int(outer.outer_fold),
                        context_epoch=epoch,
                        provider_inner_fold=epoch + 1,
                        spent_partition_ids=spent_partition_ids,
                        sealed_partition_ids=sealed_partition_ids,
                        spent_row_ids=spent_rows,
                        sealed_row_ids=sealed_rows,
                        split_fingerprint=provenance.split_fingerprint,
                    )
                )
        body = {
            "schema_version": STAGE1_HIERARCHY_SPENT_SCHEDULE_SCHEMA,
            "split_registry_sha256": registry.content_sha256,
            "review_rounds": rounds,
            "initial_spent_partition_count": 3,
            "partitions_by_outer_fold": {
                str(outer_fold): {
                    str(partition_id): list(row_ids)
                    for partition_id, row_ids in sorted(rows.items())
                }
                for outer_fold, rows in sorted(partitions.items())
            },
            "scopes": [scope.as_dict() for scope in scopes],
        }
        return cls(
            split_registry_sha256=registry.content_sha256,
            review_rounds=rounds,
            initial_spent_partition_count=3,
            partitions_by_outer_fold=MappingProxyType(partitions),
            scopes=tuple(scopes),
            schedule_sha256=_sha(body),
        )

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": STAGE1_HIERARCHY_SPENT_SCHEDULE_SCHEMA,
            "split_registry_sha256": self.split_registry_sha256,
            "review_rounds": self.review_rounds,
            "initial_spent_partition_count": self.initial_spent_partition_count,
            "partitions_by_outer_fold": {
                str(outer_fold): {
                    str(partition_id): list(row_ids)
                    for partition_id, row_ids in sorted(rows.items())
                }
                for outer_fold, rows in sorted(self.partitions_by_outer_fold.items())
            },
            "scopes": [scope.as_dict() for scope in self.scopes],
        }
        if _sha(body) != self.schedule_sha256:
            raise RuntimeError("canonical hierarchy schedule mutated after construction")
        return {**body, "schedule_sha256": self.schedule_sha256}

    def scope(self, outer_fold: int, context_epoch: int) -> CanonicalHierarchySpentScope:
        for scope in self.scopes:
            if scope.outer_fold == int(outer_fold) and scope.context_epoch == int(context_epoch):
                return scope
        raise ValueError("requested canonical accumulated-spent scope is absent")


def hierarchy_spent_data_projection_sha256(
    *,
    outer_fold: int,
    context_epoch: int,
    spent_row_ids: Sequence[int],
    sealed_row_ids: Sequence[int],
    spent_texts: Sequence[str],
    spent_treatment: Sequence[float],
    spent_outcome: Sequence[float],
) -> str:
    """Hash the only labeled projection a cumulative-spent producer may see."""

    spent = _ordered_unique_rows(spent_row_ids, label="spent_row_ids")
    sealed = _ordered_unique_rows(sealed_row_ids, label="sealed_row_ids")
    if set(spent) & set(sealed):
        raise ValueError("spent and sealed hierarchy rows overlap")
    texts = tuple(str(value) for value in spent_texts)
    treatment = np.asarray(spent_treatment, dtype=float).reshape(-1)
    outcome = np.asarray(spent_outcome, dtype=float).reshape(-1)
    if len(texts) != len(spent) or len(treatment) != len(spent) or len(outcome) != len(spent):
        raise ValueError("spent hierarchy projection columns have inconsistent lengths")
    if any(not text.strip() for text in texts):
        raise ValueError("spent hierarchy text must be explicit and non-empty")
    if not np.isfinite(treatment).all() or not np.isfinite(outcome).all():
        raise ValueError("spent hierarchy labels must be finite")
    return cumulative_spent_data_projection_sha256(
        outer_fold=int(outer_fold),
        context_epoch=int(context_epoch),
        spent_rows=tuple(
            Stage1FitRow(
                row_id=row_id,
                text=text,
                treatment=float(a),
                outcome=float(y),
            )
            for row_id, text, a, y in zip(spent, texts, treatment, outcome)
        ),
        sealed_row_ids=sealed,
    )


def role_neutral_catalog_from_dict(value: Mapping[str, Any]) -> RoleNeutralEvidenceCatalog:
    """Rehydrate and revalidate one exact persisted role-neutral catalog."""

    if not isinstance(value, Mapping):
        raise TypeError("persisted role-neutral catalog must be one mapping")
    raw = copy.deepcopy(dict(value))
    expected_keys = {
        "schema_version",
        "outer_fold",
        "scope",
        "inner_fold",
        "split_fingerprint",
        "catalog_sha256",
        "atoms",
        "non_grounding_numerical_summaries",
        "audit",
    }
    if (
        set(raw) != expected_keys
        or raw.get("schema_version") != ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION
    ):
        raise ValueError("persisted role-neutral catalog has an unsupported closed schema")
    atoms_raw = raw.get("atoms")
    summaries_raw = raw.get("non_grounding_numerical_summaries")
    if not isinstance(atoms_raw, list) or not isinstance(summaries_raw, list):
        raise TypeError("persisted catalog atoms and summaries must be lists")
    atoms: list[Stage1EvidenceAtom] = []
    for index, item in enumerate(atoms_raw):
        if not isinstance(item, Mapping):
            raise TypeError(f"persisted catalog atom {index} must be a mapping")
        row = dict(item)
        required = {
            "schema_version",
            "evidence_id",
            "atom_kind",
            "source_kind",
            "source_family",
            "observable_axes",
            "member_ids",
            "split_fingerprint",
            "origin_sha256",
            "content_sha256",
            "origin",
            "content",
        }
        if set(row) != required or row.get("schema_version") != ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION:
            raise ValueError(f"persisted catalog atom {index} has an unsupported schema")
        if not isinstance(row["origin"], Mapping) or not isinstance(row["content"], Mapping):
            raise TypeError(f"persisted catalog atom {index} has malformed content")
        atoms.append(
            Stage1EvidenceAtom(
                evidence_id=str(row["evidence_id"]),
                atom_kind=str(row["atom_kind"]),
                source_kind=str(row["source_kind"]),
                source_family=str(row["source_family"]),
                observable_axes=tuple(map(str, row["observable_axes"])),
                member_ids=tuple(map(str, row["member_ids"])),
                split_fingerprint=str(row["split_fingerprint"]),
                origin_sha256=str(row["origin_sha256"]),
                content_sha256=str(row["content_sha256"]),
                _origin_json=_canonical_json(row["origin"]),
                _content_json=_canonical_json(row["content"]),
            )
        )
    summaries: list[NonGroundingNumericalSummary] = []
    for index, item in enumerate(summaries_raw):
        if not isinstance(item, Mapping):
            raise TypeError(f"persisted numerical summary {index} must be a mapping")
        row = dict(item)
        required = {
            "schema_version",
            "summary_id",
            "source_kind",
            "source_family",
            "observable_axes",
            "split_fingerprint",
            "metrics",
            "concept_grounding_allowed",
        }
        if (
            set(row) != required
            or row.get("schema_version") != NON_GROUNDING_SUMMARY_SCHEMA_VERSION
            or row.get("concept_grounding_allowed") is not False
            or not isinstance(row.get("metrics"), Mapping)
        ):
            raise ValueError(f"persisted numerical summary {index} has an unsupported schema")
        summaries.append(
            NonGroundingNumericalSummary(
                summary_id=str(row["summary_id"]),
                source_kind=str(row["source_kind"]),
                source_family=str(row["source_family"]),
                observable_axes=tuple(map(str, row["observable_axes"])),
                split_fingerprint=str(row["split_fingerprint"]),
                _metrics_json=_canonical_json(row["metrics"]),
            )
        )
    if not isinstance(raw.get("audit"), Mapping):
        raise TypeError("persisted catalog audit must be a mapping")
    catalog = RoleNeutralEvidenceCatalog(
        outer_fold=int(raw["outer_fold"]),
        scope=str(raw["scope"]),
        inner_fold=(None if raw["inner_fold"] is None else int(raw["inner_fold"])),
        split_fingerprint=str(raw["split_fingerprint"]),
        atoms=tuple(atoms),
        non_grounding_numerical_summaries=tuple(summaries),
        catalog_sha256=str(raw["catalog_sha256"]),
        _audit_json=_canonical_json(raw["audit"]),
    )
    validate_role_neutral_catalog(catalog)
    if catalog.as_dict() != raw:
        raise ValueError("persisted catalog changed bytes while being rehydrated")
    family_counts = {
        family: len(catalog.family_atoms(family)) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    if any(count < 1 for count in family_counts.values()):
        raise ValueError("persisted hierarchy catalog lacks one or more active architectures")
    return catalog


def _scope_request_binding(
    *,
    request_sha256: str,
    schedule_sha256: str,
    scope: CanonicalHierarchySpentScope,
    data_projection_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": STAGE1_HIERARCHY_SPENT_REQUEST_SCHEMA,
        "request_sha256": request_sha256,
        "schedule_sha256": schedule_sha256,
        "scope_id": scope.scope_id,
        "outer_fold": scope.outer_fold,
        "context_epoch": scope.context_epoch,
        "provider_inner_fold": scope.provider_inner_fold,
        "split_fingerprint": scope.split_fingerprint,
        "spent_row_order_fingerprint": row_order_fingerprint(scope.spent_row_ids),
        "sealed_row_order_fingerprint": row_order_fingerprint(scope.sealed_row_ids),
        "data_projection_sha256": data_projection_sha256,
        "sealed_text_available": False,
        "sealed_labels_available": False,
    }


def _validate_tfidf_training_scope_policy(
    value: Any,
    *,
    family: str,
    scope: CanonicalHierarchySpentScope,
    configured_fold_count: int,
) -> None:
    if family not in _TFIDF_FAMILIES:
        if value is not None:
            raise ValueError(f"{family} cannot claim a TF-IDF training-scope policy")
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"{family} must bind a truthful training-scope policy")
    row = dict(value)

    if family == TFIDF_SEMANTIC_RETRIEVAL:
        expected_keys = {
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
        if set(row) != expected_keys:
            raise ValueError(f"{family} exhaustive no-selection policy is not a closed schema")
        model_rows = _ordered_unique_rows(
            row.get("model_fit_row_ids") or (),
            label=f"{family} replay-canary model rows",
        )
        calibration_rows = _ordered_unique_rows(
            row.get("calibration_row_ids") or (),
            label=f"{family} replay-canary calibration rows",
        )
        try:
            seed = int(row.get("seed"))
            fold_count = int(row.get("fold_count"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{family} replay-canary seed/fold count is invalid") from exc
        if (
            isinstance(row.get("seed"), bool)
            or seed != row.get("seed")
            or isinstance(row.get("fold_count"), bool)
            or fold_count != row.get("fold_count")
            or row.get("schema_version")
            != "semantic_retrieval_training_only_exhaustive_no_selection_v1"
            or row.get("policy") != "training_only_exhaustive_no_selection"
            or row.get("selection_kind") != "none_deterministic_exhaustive"
            or row.get("nested_calibration_applicability") != "no_label_or_hyperparameter_selection"
            or row.get("fold_parameter") != "tfidf_nested_calibration_folds"
            or int(row.get("configured_fold_count", 0)) != int(configured_fold_count)
            or not 2 <= fold_count <= int(configured_fold_count)
            or row.get("split_method") != "ordered_row_positions_seeded_label_free_partition"
            or set(model_rows) & set(calibration_rows)
            or set(model_rows) | set(calibration_rows) != set(scope.spent_row_ids)
            or row.get("model_fit_row_order_fingerprint") != row_order_fingerprint(model_rows)
            or row.get("calibration_row_order_fingerprint")
            != row_order_fingerprint(calibration_rows)
            or row.get("partitions_are_replay_canaries_only") is not True
            or row.get("partition_canaries_select_or_drop_terms") is not False
            or row.get("authoritative_projection_scope") != "all_exact_fit_frozen_retrieval_tails"
            or row.get("projection_vocabulary_max_features") is not None
            or row.get("projection_output_limit") is not None
            or row.get("all_nonzero_sanitized_terms_preserved") is not True
            or row.get("upstream_embedding_directions_and_retrieval_use_exact_fit_labels_only")
            is not True
            or row.get("nested_calibration_labels_accessed") is not False
            or row.get("registered_heldout_labels_accessed") is not False
            or row.get("registered_heldout_text_accessed") is not False
            or row.get("registered_heldout_transform_performed") is not False
            or row.get("selection_frozen_before_registered_heldout_use") is not True
            or row.get("projection_frozen_before_registered_heldout_use") is not True
            or row.get("canonical_hierarchy_partition_count_used_as_calibration_folds") is not False
            or row.get("interaction_inner_folds_used_as_calibration_folds") is not False
        ):
            raise ValueError(
                f"{family} policy must be exhaustive, label-free, uncapped, and "
                "independent of registered heldout rows"
            )
        return

    expected_keys = {
        "policy",
        "fold_parameter",
        "configured_fold_count",
        "effective_fold_count",
        "selected_fold",
        "model_fit_row_ids",
        "calibration_row_ids",
        "model_fit_row_order_fingerprint",
        "calibration_row_order_fingerprint",
        "registered_sealed_labels_accessed",
        "nested_calibration_labels_accessed",
        "selection_frozen_before_registered_sealed_transform",
        "canonical_hierarchy_partition_count_used_as_calibration_folds",
        "interaction_inner_folds_used_as_calibration_folds",
    }
    if set(row) != expected_keys:
        raise ValueError(f"{family} nested calibration proof is not a closed schema")
    model_rows = _ordered_unique_rows(
        row.get("model_fit_row_ids") or (), label=f"{family} nested model rows"
    )
    calibration_rows = _ordered_unique_rows(
        row.get("calibration_row_ids") or (), label=f"{family} nested calibration rows"
    )
    effective = int(row.get("effective_fold_count", 0))
    selected = int(row.get("selected_fold", 0))
    if (
        row.get("policy") != "nested_training_only_calibration"
        or row.get("fold_parameter") != "tfidf_nested_calibration_folds"
        or int(row.get("configured_fold_count", 0)) != int(configured_fold_count)
        or not 2 <= effective <= int(configured_fold_count)
        or not 1 <= selected <= effective
        or set(model_rows) & set(calibration_rows)
        or set(model_rows) | set(calibration_rows) != set(scope.spent_row_ids)
        or row.get("model_fit_row_order_fingerprint") != row_order_fingerprint(model_rows)
        or row.get("calibration_row_order_fingerprint") != row_order_fingerprint(calibration_rows)
        or row.get("registered_sealed_labels_accessed") is not False
        or row.get("nested_calibration_labels_accessed") is not True
        or row.get("selection_frozen_before_registered_sealed_transform") is not True
        or row.get("canonical_hierarchy_partition_count_used_as_calibration_folds") is not False
        or row.get("interaction_inner_folds_used_as_calibration_folds") is not False
    ):
        raise ValueError(f"{family} nested calibration proof violates training-only isolation")


_NATIVE_ARTIFACT_REGISTRATION_KEYS = {
    "relative_path",
    "kind",
    "file_count",
    "size",
    "sha256",
}


def _authenticated_native_artifact_registration(
    root: _BundleRootCapability,
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    """Re-hash one registered native file or tree through the root capability."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} registration must be a mapping")
    registration = copy.deepcopy(dict(value))
    if set(registration) != _NATIVE_ARTIFACT_REGISTRATION_KEYS:
        raise ValueError(f"{label} registration is not a closed schema")
    relative_path = registration.get("relative_path")
    kind = registration.get("kind")
    raw_count = registration.get("file_count")
    raw_size = registration.get("size")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or kind not in {"file", "directory"}
        or isinstance(raw_count, bool)
        or not isinstance(raw_count, int)
        or raw_count < 1
        or isinstance(raw_size, bool)
        or not isinstance(raw_size, int)
        or raw_size < 0
    ):
        raise ValueError(f"{label} registration has invalid shape or counts")
    expected_sha256 = _require_sha256(
        registration.get("sha256"),
        label=f"{label} registered hash",
    )

    if kind == "file":
        if raw_count != 1:
            raise ValueError(f"{label} file registration must have file_count=1")
        first = root.snapshot(relative_path, label=label)
        second = root.snapshot(relative_path, label=label)
        if (
            first.sha256 != second.sha256
            or first.stat_identity != second.stat_identity
            or first.sha256 != expected_sha256
            or first.stat_identity[2] != raw_size
        ):
            raise RuntimeError(f"{label} bytes changed or differ from their registration")
        return registration

    def scan_tree() -> tuple[tuple[dict[str, Any], ...], int, str]:
        relative_files = root.walk_regular_files(relative_path, label=label)
        if not relative_files:
            raise ValueError(f"{label} directory cannot be empty")
        inventory: list[dict[str, Any]] = []
        total_size = 0
        prefix = relative_path.rstrip("/")
        for child in relative_files:
            snapshot = root.snapshot(f"{prefix}/{child}", label=f"{label}/{child}")
            size = int(snapshot.stat_identity[2])
            total_size += size
            inventory.append(
                {
                    "relative_path": child,
                    "size": size,
                    "sha256": snapshot.sha256,
                }
            )
        frozen_inventory = tuple(inventory)
        return frozen_inventory, total_size, _sha({"artifact_tree": inventory})

    first_inventory, first_size, first_sha256 = scan_tree()
    second_inventory, second_size, second_sha256 = scan_tree()
    if (
        first_inventory != second_inventory
        or first_size != second_size
        or first_sha256 != second_sha256
    ):
        raise RuntimeError(f"{label} directory changed while it was being authenticated")
    if (
        len(first_inventory) != raw_count
        or first_size != raw_size
        or first_sha256 != expected_sha256
    ):
        raise ValueError(f"{label} directory differs from its registered tree")
    return registration


def _validate_native_model_descriptor(
    *,
    root: _BundleRootCapability,
    registration: Mapping[str, Any],
    family: str,
    scope: CanonicalHierarchySpentScope,
    proof_body: Mapping[str, Any],
    execution_record_sha256: str,
    execution_record_content_sha256: str,
) -> dict[str, Any]:
    """Authenticate the descriptor and every native byte named inside it."""

    descriptor_path, descriptor_sha256, _descriptor_stat = _registered_file_hash(
        root,
        registration,
        label=f"{family} hierarchy native model descriptor",
    )
    snapshot = root.snapshot(
        registration.get("relative_path"),
        label=f"{family} hierarchy native model descriptor",
    )
    if snapshot.sha256 != descriptor_sha256:
        raise RuntimeError(f"{family} native model descriptor changed while loading")
    descriptor = _load_json_snapshot(
        snapshot,
        label=f"{family} hierarchy native model descriptor",
    )
    body = dict(descriptor)
    declared = _require_sha256(
        body.pop("content_sha256", None),
        label=f"{family} native model descriptor content hash",
    )
    expected_keys = {
        "schema_version",
        "scope_id",
        "family",
        "typed_family_artifact_sha256",
        "producer_identity_sha256",
        "native_model_artifact",
        "native_source_artifact",
        "fit_audit",
    }
    if set(body) != expected_keys or _sha(body) != declared:
        raise ValueError(f"{family} native model descriptor is not a closed hash-bound schema")
    if (
        body.get("schema_version") != STAGE1_HIERARCHY_NATIVE_MODEL_DESCRIPTOR_SCHEMA
        or body.get("scope_id") != scope.scope_id
        or body.get("family") != family
        or body.get("producer_identity_sha256") != proof_body.get("producer_identity_sha256")
    ):
        raise ValueError(f"{family} native model descriptor changed its fit identity")
    _require_sha256(
        body.get("typed_family_artifact_sha256"),
        label=f"{family} typed family artifact hash",
    )
    model_registration = _authenticated_native_artifact_registration(
        root,
        body.get("native_model_artifact"),
        label=f"{family} native model artifact",
    )
    source_registration = _authenticated_native_artifact_registration(
        root,
        body.get("native_source_artifact"),
        label=f"{family} native source artifact",
    )
    fit_audit = _validate_fit_audit(
        body.get("fit_audit") or {},
        family=family,
        input_binding_sha256=str(proof_body.get("input_binding_sha256") or ""),
        scope_id=scope.scope_id,
        split_scope_fingerprint=scope.split_fingerprint,
        fit_semantics=str(proof_body.get("fit_semantics") or ""),
    )
    if fit_audit.get("schema_version") != CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA:
        raise ValueError(f"{family} native descriptor has another fit-audit schema")
    if (
        fit_audit.get("tfidf_training_scope_policy")
        != proof_body.get("tfidf_training_scope_policy")
        or fit_audit.get("model_artifact_sha256") != model_registration["sha256"]
        or fit_audit.get("source_artifact_sha256") != source_registration["sha256"]
        or fit_audit.get("fit_execution_sha256")
        not in {execution_record_sha256, execution_record_content_sha256}
    ):
        raise ValueError(f"{family} native descriptor differs from its family fit proof")
    return {
        "descriptor_path": str(descriptor_path),
        "descriptor_sha256": descriptor_sha256,
        "descriptor_content_sha256": declared,
        "native_model_artifact": model_registration,
        "native_source_artifact": source_registration,
        "fit_audit": fit_audit,
    }


def _validate_scope_proof_bundle(
    value: Mapping[str, Any],
    *,
    root: _BundleRootCapability,
    request_sha256: str,
    schedule: CanonicalHierarchySpentSchedule,
    scope: CanonicalHierarchySpentScope,
    catalog: RoleNeutralEvidenceCatalog,
    interaction_inner_folds: int,
    tfidf_nested_calibration_folds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError("hierarchy spent proof bundle must be a mapping")
    raw = copy.deepcopy(dict(value))
    body = dict(raw)
    declared = _require_sha256(body.pop("content_sha256", None), label="proof bundle hash")
    if _sha(body) != declared:
        raise ValueError("hierarchy spent proof bundle content hash is invalid")
    expected_keys = {
        "schema_version",
        "request_sha256",
        "schedule_sha256",
        "scope_id",
        "outer_fold",
        "context_epoch",
        "provider_inner_fold",
        "split_fingerprint",
        "spent_row_order_fingerprint",
        "sealed_row_order_fingerprint",
        "data_projection_sha256",
        "catalog_sha256",
        "interaction_inner_folds",
        "tfidf_nested_calibration_folds",
        "architecture_order",
        "family_proofs",
        "sealed_text_available_to_producers",
        "sealed_labels_available_to_producers",
    }
    if set(body) != expected_keys:
        raise ValueError("hierarchy spent proof bundle is not a closed schema")
    if (
        body.get("schema_version") != STAGE1_HIERARCHY_SPENT_PROOF_BUNDLE_SCHEMA
        or body.get("request_sha256") != request_sha256
        or body.get("schedule_sha256") != schedule.schedule_sha256
        or body.get("scope_id") != scope.scope_id
        or int(body.get("outer_fold", 0)) != scope.outer_fold
        or int(body.get("context_epoch", -1)) != scope.context_epoch
        or int(body.get("provider_inner_fold", 0)) != scope.provider_inner_fold
        or body.get("split_fingerprint") != scope.split_fingerprint
        or body.get("spent_row_order_fingerprint") != row_order_fingerprint(scope.spent_row_ids)
        or body.get("sealed_row_order_fingerprint") != row_order_fingerprint(scope.sealed_row_ids)
        or body.get("catalog_sha256") != catalog.catalog_sha256
        or int(body.get("interaction_inner_folds", 0)) != int(interaction_inner_folds)
        or int(body.get("tfidf_nested_calibration_folds", 0)) != int(tfidf_nested_calibration_folds)
        or tuple(body.get("architecture_order") or ()) != ACTIVE_STAGE1_CONCEPT_FAMILIES
        or body.get("sealed_text_available_to_producers") is not False
        or body.get("sealed_labels_available_to_producers") is not False
    ):
        raise ValueError("hierarchy spent proof bundle changed its scope or security binding")
    data_projection_sha256 = _require_sha256(
        body.get("data_projection_sha256"), label="spent data projection hash"
    )
    request_binding_sha256 = _sha(
        _scope_request_binding(
            request_sha256=request_sha256,
            schedule_sha256=schedule.schedule_sha256,
            scope=scope,
            data_projection_sha256=data_projection_sha256,
        )
    )
    family_proofs = body.get("family_proofs")
    if not isinstance(family_proofs, list) or len(family_proofs) != len(
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    ):
        raise ValueError("hierarchy spent proof bundle must contain all ten family proofs")
    if (
        tuple(str(row.get("family")) for row in family_proofs if isinstance(row, Mapping))
        != ACTIVE_STAGE1_CONCEPT_FAMILIES
    ):
        raise ValueError("hierarchy spent family proofs are missing, duplicated, or reordered")

    registrations: dict[str, Any] = {}
    for family, raw_proof in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, family_proofs):
        if not isinstance(raw_proof, Mapping):
            raise TypeError(f"{family} hierarchy spent proof must be a mapping")
        proof = copy.deepcopy(dict(raw_proof))
        proof_body = dict(proof)
        proof_hash = _require_sha256(
            proof_body.pop("content_sha256", None), label=f"{family} proof hash"
        )
        if _sha(proof_body) != proof_hash:
            raise ValueError(f"{family} hierarchy spent proof content hash is invalid")
        expected_proof_keys = {
            "schema_version",
            "family",
            "scope_id",
            "input_binding_sha256",
            "split_fingerprint",
            "fit_semantics",
            "producer_identity_sha256",
            "producer_code_sha256",
            "configuration_sha256",
            "fit_execution_sha256",
            "model_artifact_sha256",
            "execution_record",
            "model_artifact",
            "catalog_family_payload_sha256",
            "evidence_payload_sha256",
            "tfidf_training_scope_policy",
            "heldout_labels_accessed",
            "oracle_fields_accessed",
            "secrets_accessed",
        }
        if set(proof_body) != expected_proof_keys:
            raise ValueError(f"{family} hierarchy spent proof is not a closed schema")
        if (
            proof_body.get("schema_version") != STAGE1_HIERARCHY_SPENT_FAMILY_PROOF_SCHEMA
            or proof_body.get("family") != family
            or proof_body.get("scope_id") != scope.scope_id
            or proof_body.get("input_binding_sha256") != request_binding_sha256
            or proof_body.get("split_fingerprint") != scope.split_fingerprint
            or proof_body.get("fit_semantics")
            not in {CUMULATIVE_SPENT_REFIT, CUMULATIVE_SPENT_CACHE_REPLAY}
        ):
            raise ValueError(f"{family} hierarchy spent proof changed its fit binding")
        for flag in ("heldout_labels_accessed", "oracle_fields_accessed", "secrets_accessed"):
            if proof_body.get(flag) is not False:
                raise ValueError(f"{family} hierarchy spent proof must attest {flag}=false")
        _validate_tfidf_training_scope_policy(
            proof_body.get("tfidf_training_scope_policy"),
            family=family,
            scope=scope,
            configured_fold_count=tfidf_nested_calibration_folds,
        )
        for key in (
            "producer_identity_sha256",
            "producer_code_sha256",
            "configuration_sha256",
        ):
            _require_sha256(proof_body.get(key), label=f"{family} {key}")
        execution_path, execution_sha256, _execution_stat = _registered_file_hash(
            root,
            proof_body.get("execution_record"),
            label=f"{family} hierarchy execution record",
        )
        execution_snapshot = root.snapshot(
            proof_body["execution_record"].get("relative_path"),
            label=f"{family} hierarchy execution record",
        )
        if execution_snapshot.sha256 != execution_sha256:
            raise RuntimeError(f"{family} hierarchy execution record changed while loading")
        execution_record = _load_json_snapshot(
            execution_snapshot,
            label=f"{family} hierarchy execution record",
        )
        execution_record_content_sha256 = _sha(execution_record)
        model_path, model_sha256, _model_stat = _registered_file_hash(
            root,
            proof_body.get("model_artifact"),
            label=f"{family} hierarchy native model descriptor",
        )
        if proof_body.get("fit_execution_sha256") != execution_sha256:
            raise ValueError(f"{family} hierarchy execution record hash is invalid")
        if proof_body.get("model_artifact_sha256") != model_sha256:
            raise ValueError(f"{family} hierarchy model artifact hash is invalid")
        family_payload, _count = family_payload_from_catalog(catalog, family=family)
        expected_payload_sha256 = _sha(family_payload)
        if (
            proof_body.get("catalog_family_payload_sha256") != expected_payload_sha256
            or proof_body.get("evidence_payload_sha256") != expected_payload_sha256
        ):
            raise ValueError(f"{family} hierarchy proof differs from the persisted catalog")
        descriptor_validation = _validate_native_model_descriptor(
            root=root,
            registration=proof_body["model_artifact"],
            family=family,
            scope=scope,
            proof_body=proof_body,
            execution_record_sha256=execution_sha256,
            execution_record_content_sha256=execution_record_content_sha256,
        )
        registrations[family] = {
            "proof_content_sha256": proof_hash,
            "execution_record_path": str(execution_path),
            "execution_record_sha256": execution_sha256,
            "model_descriptor_path": str(model_path),
            "model_descriptor_sha256": model_sha256,
            "native_model_artifact": descriptor_validation["native_model_artifact"],
            "native_source_artifact": descriptor_validation["native_source_artifact"],
            "catalog_family_payload_sha256": expected_payload_sha256,
        }
    return raw, {
        "proof_bundle_content_sha256": declared,
        "data_projection_sha256": data_projection_sha256,
        "request_binding_sha256": request_binding_sha256,
        "families": registrations,
    }


class AuthenticatedProductionStage1HierarchyProvider:
    """Canonical partition and prefit-catalog provider for the hierarchy.

    The hierarchy integration must call :meth:`get_spent_evidence_catalog`.
    Calling the historical raw-input method is an explicit error so an
    independently scheduled Stage 1 refit cannot become a silent fallback.
    """

    def __init__(
        self,
        *,
        bundle_root: _BundleRootCapability,
        bundle_sha256: str,
        request_sha256: str,
        hierarchical_discovery_contract_identity: Mapping[str, Any],
        index_path: Path,
        index_file_sha256: str,
        schedule: CanonicalHierarchySpentSchedule,
        interaction_inner_folds: int,
        tfidf_nested_calibration_folds: int,
        scope_registrations: Mapping[tuple[int, int], Mapping[str, Any]],
    ) -> None:
        if not isinstance(bundle_root, _BundleRootCapability):
            raise TypeError("bundle_root must be the loader's descriptor-anchored capability")
        self._bundle_root = bundle_root
        self._bundle_sha256 = _require_sha256(bundle_sha256, label="Stage 1 bundle hash")
        self._request_sha256 = _require_sha256(request_sha256, label="Stage 1 request hash")
        self._hierarchical_discovery_contract_identity = (
            validate_production_stage1_hierarchy_contract_identity(
                hierarchical_discovery_contract_identity
            )
        )
        self._hierarchical_discovery_contract_identity_sha256 = _require_sha256(
            self._hierarchical_discovery_contract_identity["content_sha256"],
            label="hierarchical discovery contract identity hash",
        )
        self._index_path = Path(index_path)
        self._index_file_sha256 = _require_sha256(
            index_file_sha256, label="hierarchy spent index file hash"
        )
        self._schedule = schedule
        self._interaction_inner_folds = int(interaction_inner_folds)
        self._tfidf_nested_calibration_folds = int(tfidf_nested_calibration_folds)
        if self._interaction_inner_folds < 2 or self._tfidf_nested_calibration_folds < 2:
            raise ValueError("interaction and TF-IDF calibration fold counts must be at least two")
        self._scope_registrations = {
            (int(key[0]), int(key[1])): copy.deepcopy(dict(value))
            for key, value in scope_registrations.items()
        }
        expected = {(scope.outer_fold, scope.context_epoch) for scope in schedule.scopes}
        if set(self._scope_registrations) != expected:
            raise ValueError("hierarchy provider scope registrations are incomplete")
        graph: list[dict[str, Any]] = []
        for scope in schedule.scopes:
            _catalog, _proof, binding = self._load_scope(scope)
            graph.append({"scope": scope.as_dict(), **binding})
        identity_body = {
            "schema_version": STAGE1_HIERARCHY_PROVIDER_IDENTITY_SCHEMA,
            "bundle_sha256": self._bundle_sha256,
            "request_sha256": self._request_sha256,
            "hierarchical_discovery_contract_identity_sha256": (
                self._hierarchical_discovery_contract_identity_sha256
            ),
            "index_path": str(self._index_path),
            "index_file_sha256": self._index_file_sha256,
            "schedule": schedule.as_dict(),
            "fold_domains": {
                "hierarchy_schedule": {
                    "partition_count": schedule.review_rounds + 3,
                    "review_rounds": schedule.review_rounds,
                    "purpose": "three_initial_spent_partitions_then_one_sealed_gate_per_round",
                },
                "interaction_crossfit": {
                    "fold_count": self._interaction_inner_folds,
                    "purpose": "downstream_interaction_and_final_effect_estimator_crossfit",
                    "reused_for_hierarchy_schedule": False,
                    "reused_for_tfidf_calibration": False,
                },
                "tfidf_nested_training_only_calibration": {
                    "configured_fold_count": self._tfidf_nested_calibration_folds,
                    "families": list(ACTIVE_STAGE1_CONCEPT_FAMILIES[6:9]),
                    "label_based_nested_selection_families": list(
                        ACTIVE_STAGE1_CONCEPT_FAMILIES[7:9]
                    ),
                    "deterministic_exhaustive_no_selection_families": [
                        ACTIVE_STAGE1_CONCEPT_FAMILIES[6]
                    ],
                    "purpose": ("truthful_per_family_training_scope_selection_or_applicability"),
                    "registered_sealed_treatment_or_outcome_available": False,
                    "reused_for_hierarchy_schedule": False,
                    "reused_for_interaction_crossfit": False,
                },
            },
            "scope_graph": graph,
            "all_ten_architectures_required": True,
            "catalogs_prefit_on_exact_cumulative_spent_scopes": True,
            "schema_level_proof_graph_authenticated": True,
            "native_proof_validation_substrate_ready": (NATIVE_PROOF_VALIDATION_SUBSTRATE_READY),
            "genuine_native_component_proofs_validated": (NATIVE_PROOF_VALIDATION_SUBSTRATE_READY),
            "genuine_one_shot_e2e_certified": (GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY),
            "independent_runtime_stage1_refit_allowed": False,
            "raw_all_architecture_prompt_allowed": False,
            "manual_digest_approval_required": False,
        }
        self._identity = {
            **identity_body,
            "identity_sha256": _sha(identity_body),
        }

    @property
    def schedule(self) -> CanonicalHierarchySpentSchedule:
        return self._schedule

    def _assert_hierarchical_discovery_contract_current(self) -> None:
        current = validate_production_stage1_hierarchy_contract_identity(
            self._hierarchical_discovery_contract_identity
        )
        if current["content_sha256"] != self._hierarchical_discovery_contract_identity_sha256:
            raise RuntimeError("hierarchical discovery contract changed after provider binding")

    def _load_scope(
        self, scope: CanonicalHierarchySpentScope
    ) -> tuple[RoleNeutralEvidenceCatalog, Mapping[str, Any], dict[str, Any]]:
        self._assert_hierarchical_discovery_contract_current()
        registration = self._scope_registrations[(scope.outer_fold, scope.context_epoch)]
        catalog_path, catalog_raw, catalog_snapshot = _registered_json(
            self._bundle_root,
            registration.get("catalog"),
            label=f"{scope.scope_id} catalog",
        )
        proof_path, proof_raw, proof_snapshot = _registered_json(
            self._bundle_root,
            registration.get("proof_bundle"),
            label=f"{scope.scope_id} proof bundle",
        )
        catalog = role_neutral_catalog_from_dict(catalog_raw)
        if (
            catalog.outer_fold != scope.outer_fold
            or catalog.scope != "inner_train"
            or catalog.inner_fold != scope.provider_inner_fold
            or catalog.split_fingerprint != scope.split_fingerprint
            or registration.get("catalog_sha256") != catalog.catalog_sha256
        ):
            raise ValueError(f"{scope.scope_id} catalog changed its canonical scope")
        proof, proof_binding = _validate_scope_proof_bundle(
            proof_raw,
            root=self._bundle_root,
            request_sha256=self._request_sha256,
            schedule=self._schedule,
            scope=scope,
            catalog=catalog,
            interaction_inner_folds=self._interaction_inner_folds,
            tfidf_nested_calibration_folds=self._tfidf_nested_calibration_folds,
        )
        binding = {
            "scope_id": scope.scope_id,
            "catalog_path": str(catalog_path),
            "catalog_file_sha256": catalog_snapshot.sha256,
            "catalog_sha256": catalog.catalog_sha256,
            "proof_bundle_path": str(proof_path),
            "proof_bundle_file_sha256": proof_snapshot.sha256,
            **proof_binding,
        }
        return catalog, proof, binding

    def identity(self) -> Mapping[str, Any]:
        # Recheck the complete byte graph before returning the bound identity.
        for scope in self._schedule.scopes:
            _catalog, _proof, current = self._load_scope(scope)
            expected = next(
                row for row in self._identity["scope_graph"] if row["scope_id"] == scope.scope_id
            )
            if {key: value for key, value in expected.items() if key != "scope"} != current:
                raise RuntimeError(f"{scope.scope_id} proof graph changed after binding")
        return copy.deepcopy(self._identity)

    def get_review_partition_assignments(
        self,
        *,
        outer_fold: int,
        exact_outer_train_row_ids: tuple[int, ...],
    ) -> Mapping[int, Sequence[int]]:
        self._assert_hierarchical_discovery_contract_current()
        outer = int(outer_fold)
        partitions = self._schedule.partitions_by_outer_fold.get(outer)
        if partitions is None:
            raise ValueError("requested outer fold is absent from the Stage 1 hierarchy schedule")
        flattened = {row_id for rows in partitions.values() for row_id in rows}
        if flattened != set(map(int, exact_outer_train_row_ids)):
            raise ValueError("hierarchy runner outer-train rows differ from the Stage 1 registry")
        return {partition_id: tuple(rows) for partition_id, rows in sorted(partitions.items())}

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
        self._assert_hierarchical_discovery_contract_current()
        if not NATIVE_PROOF_VALIDATION_SUBSTRATE_READY:
            raise RuntimeError(
                "production hierarchy catalog serving is blocked because cumulative-spent "
                "native proof validation is not implemented"
            )
        scope = self._schedule.scope(int(outer_fold), int(review_round))
        if tuple(map(int, exact_spent_row_ids)) != scope.spent_row_ids:
            raise ValueError("hierarchy requested a noncanonical accumulated-spent row scope")
        if tuple(map(int, exact_sealed_row_ids)) != scope.sealed_row_ids:
            raise ValueError("hierarchy requested a noncanonical still-sealed row scope")
        catalog, proof, _binding = self._load_scope(scope)
        observed_projection = hierarchy_spent_data_projection_sha256(
            outer_fold=scope.outer_fold,
            context_epoch=scope.context_epoch,
            spent_row_ids=scope.spent_row_ids,
            sealed_row_ids=scope.sealed_row_ids,
            spent_texts=spent_texts,
            spent_treatment=spent_treatment,
            spent_outcome=spent_outcome,
        )
        if proof.get("data_projection_sha256") != observed_projection:
            raise ValueError("runtime spent data differ from the component-emitted Stage 1 proof")
        return catalog

    def get_spent_evidence_inputs(self, **_kwargs: Any) -> Sequence[Any]:
        raise RuntimeError(
            "authenticated production Stage 1 requires direct prefit catalog consumption; "
            "the historical independently refitted evidence-input path is forbidden"
        )


@dataclass(frozen=True)
class AuthenticatedProductionStage1HierarchyHandoff:
    inputs: AuthenticatedStage1HierarchyInputs
    provider: AuthenticatedProductionStage1HierarchyProvider

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": STAGE1_HIERARCHY_HANDOFF_SCHEMA,
            "stage1_inputs": self.inputs.as_dict(),
            "provider_identity": self.provider.identity(),
            "hierarchical_discovery_contract_identity_sha256": (
                self.inputs.hierarchical_discovery_contract_identity["content_sha256"]
            ),
            "discovery_mode": "hierarchical",
            "all_ten_architectures_required": True,
            "per_architecture_interpretation_required": True,
            "cross_architecture_integration_uses_bounded_id_lookback": True,
            "raw_all_architecture_prompt_allowed": False,
            "independent_runtime_stage1_refit_allowed": False,
            "manual_digest_approval_required": False,
            "production_one_shot_same_process_runner_required": True,
            "production_one_shot_caller_replay_registrations_accepted": False,
            "production_one_shot_exact_coordinator_and_result_required": True,
            "native_proof_validation_substrate_ready": (NATIVE_PROOF_VALIDATION_SUBSTRATE_READY),
            "genuine_native_component_proofs_validated": (NATIVE_PROOF_VALIDATION_SUBSTRATE_READY),
            "genuine_one_shot_e2e_certified": (GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY),
            "production_hierarchy_ready": (GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY),
        }
        return {**body, "content_sha256": _sha(body)}


def load_production_stage1_hierarchy_handoff(
    manifest_path: Path | str,
    *,
    review_rounds: int,
    interaction_inner_folds: int = 3,
    tfidf_nested_calibration_folds: int = 3,
) -> AuthenticatedProductionStage1HierarchyHandoff:
    """Load the honest hierarchy graph or fail before constructing live clients."""

    interaction_folds = int(interaction_inner_folds)
    tfidf_folds = int(tfidf_nested_calibration_folds)
    if interaction_folds < 2:
        raise ValueError("interaction_inner_folds must be at least two")
    if tfidf_folds < 2:
        raise ValueError("tfidf_nested_calibration_folds must be at least two")
    inputs = load_authenticated_stage1_bundle_for_hierarchy(manifest_path)
    root = inputs._bundle_root_capability
    manifest = inputs._authenticated_manifest()
    request = inputs._authenticated_registered_json("immutable_build_request")
    contract = request.get("hierarchy_spent_evidence_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError(
            "Stage 1 bundle has no canonical accumulated-spent hierarchy contract; "
            "compatibility handoffs cannot satisfy production hierarchy discovery"
        )
    expected_contract_keys = {
        "schema_version",
        "review_rounds",
        "partition_authority",
        "initial_spent_partition_count",
        "canonical_hierarchy_partition_count",
        "interaction_inner_folds",
        "tfidf_nested_calibration_folds",
        "fold_domains_are_distinct",
        "required_families",
        "hierarchical_discovery_contract_identity_sha256",
        "schedule_sha256",
        "component_emitted_catalogs_and_proofs_required",
        "independent_runtime_stage1_refit_allowed",
        "manual_digest_approval_required",
    }
    if (
        set(contract) != expected_contract_keys
        or contract.get("schema_version") != STAGE1_HIERARCHY_SPENT_CONTRACT_SCHEMA
        or int(contract.get("review_rounds", 0)) != int(review_rounds)
        or contract.get("partition_authority")
        != "canonical_stage1_inner_heldout_partitions_in_registry_order"
        or int(contract.get("initial_spent_partition_count", 0)) != 3
        or int(contract.get("canonical_hierarchy_partition_count", 0)) != int(review_rounds) + 3
        or int(contract.get("interaction_inner_folds", 0)) != interaction_folds
        or int(contract.get("tfidf_nested_calibration_folds", 0)) != tfidf_folds
        or contract.get("fold_domains_are_distinct") is not True
        or tuple(contract.get("required_families") or ()) != ACTIVE_STAGE1_CONCEPT_FAMILIES
        or contract.get("hierarchical_discovery_contract_identity_sha256")
        != inputs.hierarchical_discovery_contract_identity["content_sha256"]
        or contract.get("component_emitted_catalogs_and_proofs_required") is not True
        or contract.get("independent_runtime_stage1_refit_allowed") is not False
        or contract.get("manual_digest_approval_required") is not False
    ):
        raise ValueError("Stage 1 hierarchy spent-evidence contract is invalid")

    wrapper_registry = inputs._authenticated_registered_json("split_registry")
    registry = _contract_registry(wrapper_registry)
    schedule = CanonicalHierarchySpentSchedule.build(
        registry=registry,
        review_rounds=int(review_rounds),
    )
    if contract.get("schedule_sha256") != schedule.schedule_sha256:
        raise ValueError("Stage 1 request is bound to another canonical hierarchy schedule")

    index_path, index, index_snapshot = _registered_json(
        root,
        manifest.get("hierarchy_spent_evidence_index"),
        label="hierarchy accumulated-spent evidence index",
    )
    index_file_sha256 = index_snapshot.sha256
    index_body = dict(index)
    index_content_sha256 = _require_sha256(
        index_body.pop("content_sha256", None), label="hierarchy spent index content hash"
    )
    if _sha(index_body) != index_content_sha256:
        raise ValueError("hierarchy spent index content hash is invalid")
    expected_index_keys = {
        "schema_version",
        "request_sha256",
        "wrapper_split_registry_content_sha256",
        "contract_split_registry_sha256",
        "schedule_sha256",
        "review_rounds",
        "initial_spent_partition_count",
        "canonical_hierarchy_partition_count",
        "interaction_inner_folds",
        "tfidf_nested_calibration_folds",
        "fold_domains_are_distinct",
        "architecture_order",
        "hierarchical_discovery_contract_identity_sha256",
        "exact_inner_evidence_index_file_sha256",
        "scopes",
        "independent_runtime_stage1_refit_allowed",
        "manual_digest_approval_required",
    }
    if set(index_body) != expected_index_keys:
        raise ValueError("hierarchy spent index is not a closed schema")
    exact_index_sha = inputs._authenticated_registered_snapshot("exact_inner_evidence_index").sha256
    if (
        index_body.get("schema_version") != STAGE1_HIERARCHY_SPENT_INDEX_SCHEMA
        or index_body.get("request_sha256") != inputs.request_sha256
        or index_body.get("wrapper_split_registry_content_sha256") != _sha256_json(wrapper_registry)
        or index_body.get("contract_split_registry_sha256") != registry.content_sha256
        or index_body.get("schedule_sha256") != schedule.schedule_sha256
        or int(index_body.get("review_rounds", 0)) != int(review_rounds)
        or int(index_body.get("initial_spent_partition_count", 0)) != 3
        or int(index_body.get("canonical_hierarchy_partition_count", 0)) != int(review_rounds) + 3
        or int(index_body.get("interaction_inner_folds", 0)) != interaction_folds
        or int(index_body.get("tfidf_nested_calibration_folds", 0)) != tfidf_folds
        or index_body.get("fold_domains_are_distinct") is not True
        or tuple(index_body.get("architecture_order") or ()) != ACTIVE_STAGE1_CONCEPT_FAMILIES
        or index_body.get("hierarchical_discovery_contract_identity_sha256")
        != inputs.hierarchical_discovery_contract_identity["content_sha256"]
        or index_body.get("exact_inner_evidence_index_file_sha256") != exact_index_sha
        or index_body.get("independent_runtime_stage1_refit_allowed") is not False
        or index_body.get("manual_digest_approval_required") is not False
    ):
        raise ValueError("hierarchy spent index changed its root or schedule binding")
    raw_scopes = index_body.get("scopes")
    if not isinstance(raw_scopes, list):
        raise ValueError("hierarchy spent index has no scope registrations")
    registrations: dict[tuple[int, int], Mapping[str, Any]] = {}
    expected_scopes = {(scope.outer_fold, scope.context_epoch) for scope in schedule.scopes}
    for raw_scope in raw_scopes:
        if not isinstance(raw_scope, Mapping):
            raise TypeError("hierarchy spent scope registration must be a mapping")
        row = copy.deepcopy(dict(raw_scope))
        expected_keys = {
            "scope_id",
            "outer_fold",
            "context_epoch",
            "provider_inner_fold",
            "spent_row_ids",
            "sealed_row_ids",
            "split_fingerprint",
            "catalog",
            "catalog_sha256",
            "proof_bundle",
        }
        if set(row) != expected_keys:
            raise ValueError("hierarchy spent scope registration is not a closed schema")
        key = (int(row.get("outer_fold", 0)), int(row.get("context_epoch", -1)))
        if key in registrations:
            raise ValueError("hierarchy spent scope index contains duplicates")
        expected_scope = schedule.scope(*key)
        if (
            row.get("scope_id") != expected_scope.scope_id
            or int(row.get("provider_inner_fold", 0)) != expected_scope.provider_inner_fold
            or tuple(map(int, row.get("spent_row_ids") or ())) != expected_scope.spent_row_ids
            or tuple(map(int, row.get("sealed_row_ids") or ())) != expected_scope.sealed_row_ids
            or row.get("split_fingerprint") != expected_scope.split_fingerprint
            or _HEX_SHA256.fullmatch(str(row.get("catalog_sha256") or "")) is None
        ):
            raise ValueError("hierarchy spent scope registration changed canonical rows")
        registrations[key] = row
    if set(registrations) != expected_scopes:
        raise ValueError("hierarchy spent scope index does not cover every required epoch")

    provider = AuthenticatedProductionStage1HierarchyProvider(
        bundle_root=root,
        bundle_sha256=inputs.bundle_sha256,
        request_sha256=inputs.request_sha256,
        hierarchical_discovery_contract_identity=(inputs.hierarchical_discovery_contract_identity),
        index_path=index_path,
        index_file_sha256=index_file_sha256,
        schedule=schedule,
        interaction_inner_folds=interaction_folds,
        tfidf_nested_calibration_folds=tfidf_folds,
        scope_registrations=registrations,
    )
    return AuthenticatedProductionStage1HierarchyHandoff(inputs=inputs, provider=provider)


_PREPARED_EXECUTION_CAPABILITY_ISSUER = object()
_PREPARED_EXECUTION_CAPABILITY_LOCK = threading.Lock()
_PREPARED_EXECUTION_CAPABILITIES: dict[
    int,
    tuple[weakref.ReferenceType[object], str],
] = {}
_GENERIC_PREPARATION_REQUIRED_BODY_KEYS = {
    "dataset",
    "effective_runner_config",
    "extraction_cache_overlay",
    "final_causal_forest_backend",
    "final_upstream_producer",
    "frozen_review_evidence_policy",
    "hierarchical_architecture_chunk_limits",
    "hierarchical_discovery_config",
    "hierarchical_runner_identity",
    "legacy_handoff",
    "outer_folds",
    "raw_final_upstream_producer",
    "shared_first_gate_provider",
    "spent_evidence_provider",
    "tfidf_handoff",
}


class PreparedProductionStage1HierarchyExecutionCapability:
    """Opaque, process-local, one-shot authority over one prepared batch."""

    def __init__(
        self,
        *,
        issuer: object,
        prepared_batch: object,
        handoff_content_sha256: str,
        provider_identity_sha256: str,
        contract_identity_sha256: str,
        approval_sha256: str,
        input_manifest_sha256: str,
        input_wrapper: Mapping[str, Any],
        batch_wrapper: Mapping[str, Any],
        production_binding: Mapping[str, Any],
        runner: object,
        runtime_binding: Mapping[str, Any],
        runtime_objects: Sequence[tuple[str, object]],
        coordinator: object,
        coordinator_precommit: object,
        coordinator_execute_function: object,
        coordinator_assert_unchanged_function: object,
    ) -> None:
        if issuer is not _PREPARED_EXECUTION_CAPABILITY_ISSUER:
            raise TypeError("prepared execution capabilities are issued internally only")
        self._prepared_batch = prepared_batch
        self._prepared_folds = prepared_batch.folds
        self._prepared_fold_objects = tuple(prepared_batch.folds)
        self.handoff_content_sha256 = _require_sha256(
            handoff_content_sha256,
            label="prepared capability handoff hash",
        )
        self.provider_identity_sha256 = _require_sha256(
            provider_identity_sha256,
            label="prepared capability provider hash",
        )
        self.contract_identity_sha256 = _require_sha256(
            contract_identity_sha256,
            label="prepared capability contract hash",
        )
        self.approval_sha256 = _require_sha256(
            approval_sha256,
            label="prepared capability batch approval hash",
        )
        self.input_manifest_sha256 = _require_sha256(
            input_manifest_sha256,
            label="prepared capability input manifest hash",
        )
        self.input_wrapper = copy.deepcopy(dict(input_wrapper))
        self.batch_wrapper = copy.deepcopy(dict(batch_wrapper))
        self.production_binding = copy.deepcopy(dict(production_binding))
        self._runner = runner
        self.runtime_binding = copy.deepcopy(dict(runtime_binding))
        self._runtime_objects = tuple(runtime_objects)
        self._coordinator = coordinator
        self._coordinator_precommit = coordinator_precommit
        self._coordinator_execute_function = coordinator_execute_function
        self._coordinator_assert_unchanged_function = coordinator_assert_unchanged_function
        self._capability_token = secrets.token_hex(32)
        identifier = id(self)

        def discard(reference: weakref.ReferenceType[object]) -> None:
            with _PREPARED_EXECUTION_CAPABILITY_LOCK:
                registered = _PREPARED_EXECUTION_CAPABILITIES.get(identifier)
                if registered is not None and registered[0] is reference:
                    _PREPARED_EXECUTION_CAPABILITIES.pop(identifier, None)

        reference = weakref.ref(self, discard)
        with _PREPARED_EXECUTION_CAPABILITY_LOCK:
            _PREPARED_EXECUTION_CAPABILITIES[identifier] = (
                reference,
                self._capability_token,
            )

    def _assert_fresh(self) -> None:
        with _PREPARED_EXECUTION_CAPABILITY_LOCK:
            registered = _PREPARED_EXECUTION_CAPABILITIES.get(id(self))
            if (
                registered is None
                or registered[0]() is not self
                or registered[1] != getattr(self, "_capability_token", None)
            ):
                raise RuntimeError("prepared hierarchy execution capability is absent or consumed")

    def _assert_execution_binding(self, *, prepared_batch: object, runner: object) -> None:
        from .all_evidence_fusion_runner import (
            AllEvidenceFusionRunner,
            PreparedHierarchicalDiscoveryBatch,
            PreparedHierarchicalDiscoveryFold,
            _current_production_hierarchy_runtime_binding,
        )

        if type(prepared_batch) is not PreparedHierarchicalDiscoveryBatch:
            raise TypeError("authorized execution requires the concrete prepared batch")
        if type(runner) is not AllEvidenceFusionRunner:
            raise TypeError("authorized execution requires the concrete production runner")
        if self._prepared_batch is not prepared_batch or self._runner is not runner:
            raise ValueError("hierarchy authorization owns another runner or prepared batch")
        if (
            prepared_batch.folds is not self._prepared_folds
            or len(prepared_batch.folds) != len(self._prepared_fold_objects)
            or any(
                type(current) is not PreparedHierarchicalDiscoveryFold or current is not expected
                for current, expected in zip(
                    prepared_batch.folds,
                    self._prepared_fold_objects,
                )
            )
        ):
            raise ValueError("prepared hierarchy fold objects changed after authorization")
        if (
            prepared_batch.approval_sha256 != self.approval_sha256
            or prepared_batch.input_manifest_sha256 != self.input_manifest_sha256
        ):
            raise ValueError("prepared hierarchy binding changed after authorization")

        current_runtime_binding, current_runtime_objects = (
            _current_production_hierarchy_runtime_binding(runner)
        )
        if current_runtime_binding != self.runtime_binding:
            raise ValueError("production hierarchy runtime identity changed after preparation")
        expected_objects = dict(self._runtime_objects)
        if tuple(name for name, _value in current_runtime_objects) != tuple(expected_objects):
            raise RuntimeError("production hierarchy runtime object schema changed")
        for name, current in current_runtime_objects:
            if current is not expected_objects[name]:
                raise ValueError(f"production hierarchy runtime object changed: {name}")

        coordinator = prepared_batch.coordinator
        if (
            coordinator is not self._coordinator
            or type(coordinator) is not ApprovedHierarchicalDiscoveryBatchCoordinator
            or coordinator.precommit is not self._coordinator_precommit
        ):
            raise ValueError("prepared hierarchy coordinator changed after authorization")
        if "execute" in vars(coordinator) or "_assert_unchanged" in vars(coordinator):
            raise ValueError("prepared hierarchy coordinator has an instance method override")
        if (
            ApprovedHierarchicalDiscoveryBatchCoordinator.execute
            is not self._coordinator_execute_function
            or ApprovedHierarchicalDiscoveryBatchCoordinator._assert_unchanged
            is not self._coordinator_assert_unchanged_function
            or self._coordinator_execute_function is not _CANONICAL_COORDINATOR_EXECUTE
            or self._coordinator_assert_unchanged_function
            is not _CANONICAL_COORDINATOR_ASSERT_UNCHANGED
        ):
            raise RuntimeError("prepared hierarchy coordinator implementation changed")
        coordinator.precommit.__post_init__()
        expected_packet = self.batch_wrapper["body"]["packet"]
        if (
            coordinator.precommit.approval_sha256 != self.approval_sha256
            or coordinator.precommit.packet != expected_packet
        ):
            raise ValueError("prepared hierarchy coordinator precommit changed")

    def _consume_once(self) -> None:
        with _PREPARED_EXECUTION_CAPABILITY_LOCK:
            registered = _PREPARED_EXECUTION_CAPABILITIES.get(id(self))
            if (
                registered is None
                or registered[0]() is not self
                or registered[1] != self._capability_token
            ):
                raise RuntimeError("prepared hierarchy execution capability is absent or consumed")
            _PREPARED_EXECUTION_CAPABILITIES.pop(id(self), None)


def prepare_internal_hierarchy_execution_capability(
    *,
    handoff: AuthenticatedProductionStage1HierarchyHandoff,
    runner: object,
    prepared_batch: object,
) -> PreparedProductionStage1HierarchyExecutionCapability:
    """Bind one genuine generic preparation to its exact same-process runner."""

    if not isinstance(handoff, AuthenticatedProductionStage1HierarchyHandoff):
        raise TypeError("handoff must be an authenticated production Stage 1 hierarchy handoff")
    from .all_evidence_fusion_runner import (
        AllEvidenceFusionRunner,
        PreparedHierarchicalDiscoveryBatch,
        PreparedHierarchicalDiscoveryFold,
        _claim_prepared_hierarchy_capability,
        _current_production_hierarchy_runtime_binding,
    )

    if type(runner) is not AllEvidenceFusionRunner:
        raise TypeError("runner must be the concrete AllEvidenceFusionRunner")
    if type(prepared_batch) is not PreparedHierarchicalDiscoveryBatch:
        raise TypeError("prepared_batch must be a concrete PreparedHierarchicalDiscoveryBatch")
    if (
        not isinstance(prepared_batch.folds, tuple)
        or not prepared_batch.folds
        or any(type(fold) is not PreparedHierarchicalDiscoveryFold for fold in prepared_batch.folds)
    ):
        raise TypeError("prepared batch must retain exact prepared fold objects")
    if (
        runner.review_spent_evidence_provider is not handoff.provider
        or runner.review_partition_provider is not handoff.provider
    ):
        raise ValueError(
            "production runner must use the handoff provider for catalogs and partitions"
        )
    if type(prepared_batch.coordinator) is not ApprovedHierarchicalDiscoveryBatchCoordinator:
        raise TypeError("prepared batch must retain the exact hierarchy coordinator")
    if (
        ApprovedHierarchicalDiscoveryBatchCoordinator.execute is not _CANONICAL_COORDINATOR_EXECUTE
        or ApprovedHierarchicalDiscoveryBatchCoordinator._assert_unchanged
        is not _CANONICAL_COORDINATOR_ASSERT_UNCHANGED
        or "execute" in vars(prepared_batch.coordinator)
        or "_assert_unchanged" in vars(prepared_batch.coordinator)
    ):
        raise RuntimeError("prepared batch coordinator execution surface is not canonical")
    _claim_prepared_hierarchy_capability(prepared_batch)
    preparation_root = _BundleRootCapability(prepared_batch.input_manifest_path.parent)
    try:
        input_absolute = Path(os.path.abspath(os.fspath(prepared_batch.input_manifest_path)))
        batch_absolute = Path(os.path.abspath(os.fspath(prepared_batch.batch_packet_path)))
        try:
            input_relative = input_absolute.relative_to(preparation_root.path)
            batch_relative = batch_absolute.relative_to(preparation_root.path)
        except ValueError as exc:
            raise ValueError(
                "prepared hierarchy artifacts must share one preparation root"
            ) from exc
        input_snapshot = preparation_root.snapshot(
            input_relative.as_posix(),
            label="hierarchy preparation input manifest",
        )
        batch_snapshot = preparation_root.snapshot(
            batch_relative.as_posix(),
            label="hierarchy preparation batch packet",
        )
        input_wrapper, input_body = _load_hash_wrapper_snapshot(
            input_snapshot,
            label="hierarchy preparation input manifest",
            expected_schema_version=HIERARCHICAL_PREPARATION_INPUT_WRAPPER_SCHEMA,
            expected_content_sha256=prepared_batch.input_manifest_sha256,
        )
        if not _GENERIC_PREPARATION_REQUIRED_BODY_KEYS <= set(input_body):
            raise ValueError("generic hierarchy preparation input manifest is incomplete")
        handoff_dict = handoff.as_dict()
        provider_identity = handoff.provider.identity()
        expected_provider_wrapper = {
            "identity": provider_identity,
            "identity_sha256": _sha(provider_identity),
        }
        request = handoff.inputs._authenticated_registered_json("immutable_build_request")
        runtime_binding, runtime_objects = _current_production_hierarchy_runtime_binding(runner)
        runtime_body = runtime_binding["body"]
        dataset = input_body.get("dataset")
        legacy_handoff = input_body.get("legacy_handoff")
        tfidf_handoff = input_body.get("tfidf_handoff")
        outer_folds = input_body.get("outer_folds")
        expected_outer_folds = tuple(sorted(handoff.provider.schedule.partitions_by_outer_fold))
        if tuple(fold.outer_fold for fold in prepared_batch.folds) != expected_outer_folds:
            raise ValueError("prepared hierarchy folds differ from the Stage 1 schedule")
        observed_outer_folds = (
            tuple(int(row.get("outer_fold", 0)) for row in outer_folds)
            if isinstance(outer_folds, list)
            and len(outer_folds) == len(expected_outer_folds)
            and all(isinstance(row, Mapping) for row in outer_folds)
            else ()
        )
        if (
            not isinstance(dataset, Mapping)
            or not isinstance(legacy_handoff, Mapping)
            or not isinstance(tfidf_handoff, Mapping)
            or dataset.get("sha256") != (request.get("dataset") or {}).get("sha256")
            or dataset.get("sha256") != runtime_body["dataset_artifact"]["sha256"]
            or legacy_handoff.get("sha256") != runtime_body["legacy_handoff_artifact"]["sha256"]
            or tfidf_handoff.get("sha256") != runtime_body["tfidf_handoff_artifact"]["sha256"]
            or input_body.get("spent_evidence_provider") != expected_provider_wrapper
            or runtime_body["spent_evidence_provider"] != expected_provider_wrapper
            or runtime_body["review_partition_provider"] != expected_provider_wrapper
            or input_body.get("hierarchical_discovery_config")
            != handoff.inputs.hierarchical_discovery_contract_identity[
                "hierarchical_discovery_config"
            ]
            or any(
                input_body.get(field) != runtime_body[field]
                for field in (
                    "effective_runner_config",
                    "extraction_cache_overlay",
                    "final_causal_forest_backend",
                    "final_upstream_producer",
                    "frozen_review_evidence_policy",
                    "hierarchical_architecture_chunk_limits",
                    "hierarchical_discovery_config",
                    "hierarchical_runner_identity",
                    "raw_final_upstream_producer",
                    "shared_first_gate_provider",
                    "spent_evidence_provider",
                )
            )
            or observed_outer_folds != expected_outer_folds
        ):
            raise ValueError("generic hierarchy preparation is not bound to this Stage 1 provider")
        batch_wrapper, batch_body = _load_hash_wrapper_snapshot(
            batch_snapshot,
            label="hierarchy preparation batch packet",
            expected_schema_version=HIERARCHICAL_PREPARATION_BATCH_WRAPPER_SCHEMA,
        )
        packet = batch_body.get("packet")
        if (
            set(batch_body) != {"approval_sha256", "packet"}
            or not isinstance(packet, Mapping)
            or batch_body.get("approval_sha256") != prepared_batch.approval_sha256
            or _sha(packet) != prepared_batch.approval_sha256
            or packet.get("input_manifest_sha256") != prepared_batch.input_manifest_sha256
            or packet != prepared_batch.coordinator.precommit.packet
        ):
            raise ValueError(
                "hierarchy preparation batch digest is not bound to its authenticated "
                "generic input manifest"
            )
        production_binding_body = {
            "generic_preparation_input_content_sha256": input_wrapper["content_sha256"],
            "generic_preparation_batch_content_sha256": batch_wrapper["content_sha256"],
            "production_stage1_hierarchy_handoff_content_sha256": handoff_dict["content_sha256"],
            "production_stage1_hierarchical_discovery_contract_identity_sha256": (
                handoff.inputs.hierarchical_discovery_contract_identity["content_sha256"]
            ),
            "production_stage1_hierarchy_provider_identity_sha256": provider_identity[
                "identity_sha256"
            ],
            "same_process_runner_runtime_binding_schema_version": runtime_binding["schema_version"],
            "same_process_runner_runtime_binding_content_sha256": runtime_binding["content_sha256"],
            "caller_replay_registrations_accepted": False,
            "execution_sources": "exact_runner_held_authenticated_providers",
            "end_user_digest_entry_required": False,
            "manual_digest_approval_required": False,
        }
        production_binding = {
            "schema_version": INTERNAL_HIERARCHY_PREPARATION_BINDING_SCHEMA,
            "content_sha256": _sha(production_binding_body),
            "body": production_binding_body,
        }
        capability = PreparedProductionStage1HierarchyExecutionCapability(
            issuer=_PREPARED_EXECUTION_CAPABILITY_ISSUER,
            prepared_batch=prepared_batch,
            handoff_content_sha256=handoff_dict["content_sha256"],
            provider_identity_sha256=provider_identity["identity_sha256"],
            contract_identity_sha256=(
                handoff.inputs.hierarchical_discovery_contract_identity["content_sha256"]
            ),
            approval_sha256=prepared_batch.approval_sha256,
            input_manifest_sha256=prepared_batch.input_manifest_sha256,
            input_wrapper=input_wrapper,
            batch_wrapper=batch_wrapper,
            production_binding=production_binding,
            runner=runner,
            runtime_binding=runtime_binding,
            runtime_objects=runtime_objects,
            coordinator=prepared_batch.coordinator,
            coordinator_precommit=prepared_batch.coordinator.precommit,
            coordinator_execute_function=_CANONICAL_COORDINATOR_EXECUTE,
            coordinator_assert_unchanged_function=(_CANONICAL_COORDINATOR_ASSERT_UNCHANGED),
        )
        return capability
    finally:
        preparation_root.close()


_EXECUTION_AUTHORIZATION_ISSUER = object()


class AuthenticatedProductionStage1HierarchyExecutionAuthorization(Mapping[str, Any]):
    """Opaque one-shot authority over one exact same-process runner and batch."""

    def __init__(
        self,
        *,
        issuer: object,
        prepared_capability: PreparedProductionStage1HierarchyExecutionCapability,
        body: Mapping[str, Any],
    ) -> None:
        if issuer is not _EXECUTION_AUTHORIZATION_ISSUER:
            raise TypeError("hierarchy execution authorizations are issued internally only")
        payload = copy.deepcopy(dict(body))
        self._payload = {**payload, "content_sha256": _sha(payload)}
        self._prepared_capability = prepared_capability
        self._lock = threading.Lock()
        self._consumed = False

    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self._payload[key])

    def __iter__(self):
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)

    def consume_for_execution(
        self,
    ) -> "AuthenticatedProductionStage1HierarchyExecutionAuthorization":
        """Consume without executing (retained for low-level audit tests)."""

        with self._lock:
            if self._consumed:
                raise RuntimeError("hierarchy execution authorization is already consumed")
            self._consumed = True
        return self

    def _execute_for_prepared_batch(
        self,
        *,
        prepared_batch: object,
        runner: object,
    ) -> Any:
        """Consume authority and invoke the retained canonical coordinator exactly once."""

        with self._lock:
            if self._consumed:
                raise RuntimeError("hierarchy execution authorization is already consumed")
            capability = self._prepared_capability
            capability._assert_execution_binding(
                prepared_batch=prepared_batch,
                runner=runner,
            )
            if (
                self._payload.get("prepared_batch_sha256") != capability.approval_sha256
                or self._payload.get("preparation_input_manifest_sha256")
                != capability.input_manifest_sha256
            ):
                raise ValueError("hierarchy execution authorization payload changed its binding")
            self._consumed = True
        result = capability._coordinator_execute_function(
            capability._coordinator,
            approved_batch_sha256=str(self._payload["prepared_batch_sha256"]),
        )
        if type(result) is not ApprovedHierarchicalDiscoveryBatchResult:
            raise TypeError("hierarchy coordinator returned a noncanonical batch result")
        result.validate_authentication()
        return result

    def _consumed_runtime_binding_for_runner(
        self,
        *,
        prepared_batch: object,
        runner: object,
    ) -> dict[str, Any]:
        """Return the immutable expected file identities after exact execution transfer."""

        with self._lock:
            if not self._consumed:
                raise RuntimeError("hierarchy execution authorization is not consumed")
            capability = self._prepared_capability
            if capability._prepared_batch is not prepared_batch or capability._runner is not runner:
                raise ValueError("hierarchy runtime binding belongs to another execution")
            capability._assert_execution_binding(
                prepared_batch=prepared_batch,
                runner=runner,
            )
            return copy.deepcopy(dict(capability.runtime_binding["body"]))


def internal_hierarchy_execution_authorization(
    *,
    handoff: AuthenticatedProductionStage1HierarchyHandoff,
    prepared_capability: PreparedProductionStage1HierarchyExecutionCapability,
) -> AuthenticatedProductionStage1HierarchyExecutionAuthorization:
    """Consume one opaque prepared-batch capability; never accept user digests."""

    if not isinstance(handoff, AuthenticatedProductionStage1HierarchyHandoff):
        raise TypeError("handoff must be an authenticated production Stage 1 hierarchy handoff")
    if type(prepared_capability) is not PreparedProductionStage1HierarchyExecutionCapability:
        raise TypeError("prepared_capability must be the concrete in-memory capability")
    prepared_capability._assert_fresh()
    handoff_dict = handoff.as_dict()
    provider_identity = handoff.provider.identity()
    if (
        prepared_capability.handoff_content_sha256 != handoff_dict["content_sha256"]
        or prepared_capability.provider_identity_sha256 != provider_identity["identity_sha256"]
        or prepared_capability.contract_identity_sha256
        != handoff.inputs.hierarchical_discovery_contract_identity["content_sha256"]
    ):
        raise ValueError("prepared hierarchy capability is bound to another Stage 1 handoff")
    if not NATIVE_PROOF_VALIDATION_SUBSTRATE_READY:
        raise RuntimeError(
            "internal digest carry is blocked because cumulative-spent native proof "
            "validation is not implemented"
        )
    prepared_capability._assert_execution_binding(
        prepared_batch=prepared_capability._prepared_batch,
        runner=prepared_capability._runner,
    )
    prepared_capability._consume_once()
    body = {
        "schema_version": INTERNAL_HIERARCHY_AUTHORIZATION_SCHEMA,
        "stage1_handoff_content_sha256": handoff_dict["content_sha256"],
        "provider_identity_sha256": provider_identity["identity_sha256"],
        "hierarchical_discovery_contract_identity_sha256": (
            handoff.inputs.hierarchical_discovery_contract_identity["content_sha256"]
        ),
        "prepared_batch_sha256": prepared_capability.approval_sha256,
        "preparation_input_manifest_sha256": prepared_capability.input_manifest_sha256,
        "preparation_input_wrapper_content_sha256": (
            prepared_capability.input_wrapper["content_sha256"]
        ),
        "preparation_batch_wrapper_content_sha256": (
            prepared_capability.batch_wrapper["content_sha256"]
        ),
        "preparation_input_wrapper_schema_version": (
            prepared_capability.input_wrapper["schema_version"]
        ),
        "preparation_batch_wrapper_schema_version": (
            prepared_capability.batch_wrapper["schema_version"]
        ),
        "production_preparation_binding_schema_version": (
            prepared_capability.production_binding["schema_version"]
        ),
        "production_preparation_binding_content_sha256": (
            prepared_capability.production_binding["content_sha256"]
        ),
        "same_process_runner_runtime_binding_schema_version": (
            prepared_capability.runtime_binding["schema_version"]
        ),
        "same_process_runner_runtime_binding_content_sha256": (
            prepared_capability.runtime_binding["content_sha256"]
        ),
        "caller_replay_registrations_accepted": False,
        "execution_sources": "exact_runner_held_authenticated_providers",
        "exact_coordinator_object_bound": True,
        "canonical_unbound_coordinator_execute_required": True,
        "exact_batch_result_type_required": True,
        "authorization_source": "single_authorized_production_cohort_invocation",
        "one_shot_capability_consumed": True,
        "low_level_digest_check_retained": True,
        "end_user_digest_entry_required": False,
        "manual_digest_approval_required": False,
    }
    return AuthenticatedProductionStage1HierarchyExecutionAuthorization(
        issuer=_EXECUTION_AUTHORIZATION_ISSUER,
        prepared_capability=prepared_capability,
        body=body,
    )


def run_internal_production_stage1_hierarchy_one_shot(
    *,
    handoff: AuthenticatedProductionStage1HierarchyHandoff,
    runner: object,
) -> Any:
    """Prepare, internally authorize, and execute one production cohort run."""

    from .all_evidence_fusion_runner import AllEvidenceFusionRunner

    if not isinstance(handoff, AuthenticatedProductionStage1HierarchyHandoff):
        raise TypeError("handoff must be an authenticated production Stage 1 hierarchy handoff")
    if type(runner) is not AllEvidenceFusionRunner:
        raise TypeError("runner must be the concrete AllEvidenceFusionRunner")
    if not NATIVE_PROOF_VALIDATION_SUBSTRATE_READY:
        raise RuntimeError(
            "one-shot production hierarchy execution is blocked because cumulative-spent "
            "native proof validation is not implemented"
        )
    if (
        runner.review_spent_evidence_provider is not handoff.provider
        or runner.review_partition_provider is not handoff.provider
    ):
        raise ValueError(
            "production runner must use the authenticated handoff provider for both "
            "spent catalogs and canonical review partitions"
        )
    if runner.hierarchical_discovery_approved_batch_sha256 is not None:
        raise ValueError("production one-shot runner cannot accept a caller-supplied digest")
    prepared = runner.prepare_hierarchical_discovery_batch()
    capability = prepare_internal_hierarchy_execution_capability(
        handoff=handoff,
        runner=runner,
        prepared_batch=prepared,
    )
    authorization = internal_hierarchy_execution_authorization(
        handoff=handoff,
        prepared_capability=capability,
    )
    return runner.run(
        prepared_hierarchical_batch=prepared,
        hierarchy_execution_authorization=authorization,
    )


__all__ = [
    "AuthenticatedProductionStage1HierarchyExecutionAuthorization",
    "AuthenticatedProductionStage1HierarchyHandoff",
    "AuthenticatedProductionStage1HierarchyProvider",
    "CanonicalHierarchySpentSchedule",
    "CanonicalHierarchySpentScope",
    "GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY",
    "NATIVE_PROOF_VALIDATION_SUBSTRATE_READY",
    "HIERARCHICAL_PREPARATION_BATCH_WRAPPER_SCHEMA",
    "HIERARCHICAL_PREPARATION_INPUT_WRAPPER_SCHEMA",
    "INTERNAL_HIERARCHY_AUTHORIZATION_SCHEMA",
    "PreparedProductionStage1HierarchyExecutionCapability",
    "STAGE1_HIERARCHY_HANDOFF_SCHEMA",
    "STAGE1_HIERARCHY_NATIVE_MODEL_DESCRIPTOR_SCHEMA",
    "STAGE1_HIERARCHY_PROVIDER_IDENTITY_SCHEMA",
    "STAGE1_HIERARCHY_SPENT_CONTRACT_SCHEMA",
    "STAGE1_HIERARCHY_SPENT_FAMILY_PROOF_SCHEMA",
    "STAGE1_HIERARCHY_SPENT_INDEX_SCHEMA",
    "STAGE1_HIERARCHY_SPENT_PROOF_BUNDLE_SCHEMA",
    "STAGE1_HIERARCHY_SPENT_REQUEST_SCHEMA",
    "STAGE1_HIERARCHY_SPENT_SCHEDULE_SCHEMA",
    "hierarchy_spent_data_projection_sha256",
    "internal_hierarchy_execution_authorization",
    "load_production_stage1_hierarchy_handoff",
    "prepare_internal_hierarchy_execution_capability",
    "role_neutral_catalog_from_dict",
    "run_internal_production_stage1_hierarchy_one_shot",
]
