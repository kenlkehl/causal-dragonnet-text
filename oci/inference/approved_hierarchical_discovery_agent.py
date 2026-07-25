"""Approved, offline-first boundary for hierarchical feature discovery.

The lower-level hierarchy deliberately separates packet construction from model
transport.  This module adds the fold-level approval boundary used by callers:
it binds the complete semantic catalog, its lossless chunk plan, exactly one
authenticated direct-numerical contract (a pre-fit first-gate materialization
intent or an already-realized manifest), the exact ten compact dossier bindings,
the JSON runner identity, and the deterministic compiler policy.

Only semantic evidence and non-grounding numerical provenance/counts enter the
rendered packet.  Coordinate names, matrix metadata, and row-level numerical
values remain behind the local authentication boundary.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
    DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT,
    DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST,
    canonical_json,
    content_sha256,
)
from .direct_upstream_numerical_manifest import (
    DirectUpstreamNumericalManifest,
    validate_architecture_dossier_numerical_binding,
)
from .hierarchical_all_architecture_discovery import (
    CompletedHierarchicalDiscovery,
    DirectNumericalDossierBinding,
    HierarchicalAllArchitectureDiscoveryOrchestrator,
    HierarchicalDiscoveryConfig,
    JsonDiscoveryJobRunner,
    RAW_TRANSPORT_BUDGET_FAILURE,
    STRICT_JSON_PARSE_FAILURE,
    VALIDATED_RESPONSE,
)
from .hierarchical_discovery_job_cache import (
    HIERARCHICAL_DISCOVERY_JOB_CACHE_ENTRY_VERSION,
    HIERARCHICAL_DISCOVERY_JOB_CACHE_HIT_VERSION,
    HIERARCHICAL_DISCOVERY_JOB_CACHE_LOOKUP_VERSION,
    AuthenticatedHierarchicalDiscoveryJobCache,
)
from .hierarchical_discovery_compiler import (
    CompiledHierarchicalFeatureRegistry,
    HierarchicalDiscoveryCompiler,
)
from .first_gate_materialization_contract import FirstGateMaterializationIntent
from .lossless_stage1_evidence_catalog import (
    DEFAULT_MAX_ATOMS_PER_ARCHITECTURE_CHUNK,
    DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK,
    DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK,
    ArchitectureChunkPlan,
    FoldEvidenceInput,
    RoleNeutralEvidenceCatalog,
    audit_complete_architecture_delivery,
    build_complete_architecture_chunks,
    build_role_neutral_evidence_catalog,
    validate_role_neutral_catalog,
)

APPROVED_HIERARCHICAL_DISCOVERY_AGENT_VERSION = "approved_hierarchical_discovery_agent_v9"
APPROVED_HIERARCHICAL_DISCOVERY_PRECOMMIT_VERSION = "approved_hierarchical_discovery_precommit_v9"
AUTHENTICATED_RUNNER_EXECUTION_TRACE_VERSION = (
    "authenticated_json_discovery_runner_and_cache_execution_trace_v8"
)
REFERENCE_ONLY_DIRECT_NUMERICAL_CONTRACT_SCHEMA = (
    "authenticated_reference_only_direct_numerical_contract_v1"
)
_REFERENCE_ONLY_DIRECT_NUMERICAL_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "outer_fold",
        "context_epoch",
        "plan_scientific_content_sha256",
        "source_execution_content_sha256",
        "reference_manifest_content_sha256",
        "runtime_binding_content_sha256",
        "provider_identity_sha256",
        "projection_content_sha256",
        "spent_row_ids",
        "spent_row_ids_sha256",
        "gate_row_ids",
        "gate_row_ids_sha256",
        "semantic_catalog",
        "family_coverage",
        "already_fit_stage1_projection",
        "conditional_fit_or_refit_performed",
        "row_values_included",
        "coordinate_to_semantic_atom_linkage",
        "concept_grounding_allowed",
    }
)
_REFERENCE_ONLY_SEMANTIC_CATALOG_KEYS = frozenset(
    {
        "catalog_sha256",
        "scope",
        "inner_fold",
        "split_fingerprint",
        "atom_count",
    }
)
_REFERENCE_ONLY_FAMILY_COVERAGE_KEYS = frozenset(
    {
        "source_family",
        "coordinate_ids",
        "coordinate_ids_sha256",
        "semantic_atom_ids",
        "semantic_atom_ids_sha256",
    }
)
APPROVED_HIERARCHICAL_DISCOVERY_RESULT_VERSION = "approved_hierarchical_discovery_result_v7"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SUCCESS_METADATA_KEYS = frozenset(
    {
        "job_id",
        "job_kind",
        "request_sha256",
        "runner_identity_sha256",
        "outcome",
        "parsed_response_sha256",
        "attempts",
    }
)
_INVALID_RESPONSE_METADATA_KEYS = _SUCCESS_METADATA_KEYS - {"parsed_response_sha256"}
_REMOTE_TRANSPORT_FAILURES = frozenset({STRICT_JSON_PARSE_FAILURE, RAW_TRANSPORT_BUDGET_FAILURE})
_REMOTE_RESPONSE_REPAIR_SEQUENCE_VERSION = "authenticated_remote_response_repair_sequence_v3"
_REMOTE_RESPONSE_REPAIR_SEQUENCE_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "job_id",
        "job_kind",
        "response_attempt_trace",
        "response_attempt_trace_sha256",
        "validated_response_sha256",
        "remote_records",
        "outcome",
        "record_sha256",
    }
)
_ATTEMPT_REQUIRED_KEYS = frozenset(
    {
        "attempt_number",
        "endpoint",
        "model",
        "request_sha256",
        "runner_identity_sha256",
        "outcome",
        "retryable",
        "will_retry",
    }
)
_ATTEMPT_ALLOWED_KEYS = _ATTEMPT_REQUIRED_KEYS | {
    "exception_type",
    "status_code",
    "retry_delay_seconds",
    "response_id",
    "response_model",
    "finish_reason",
    "usage",
    "content_sha256",
    "reasoning_hashes",
    "raw_transport_bytes",
    "parsed_response_sha256",
}
_CACHE_HIT_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "job_id",
        "job_kind",
        "job_sha256",
        "runner_identity_sha256",
        "hierarchy_inner_precommit_sha256",
        "validator_code_sha256",
        "cache_identity_sha256",
        "cache_lookup_sha256",
        "cache_entry_sha256",
        "wire_response",
        "wire_response_sha256",
        "validated_response_sha256",
        "response_attempt_trace_sha256",
        "outcome",
        "record_sha256",
    }
)


def _clone(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _closed_mapping(value: Any, *, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one JSON object")
    if set(value) != keys:
        raise ValueError(f"{label} has an unexpected closed schema")
    return value


def _component_binding(component: Any, identity: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _clone(identity)
    if not isinstance(normalized, Mapping) or not normalized:
        raise ValueError("component identity must be one non-empty JSON object")
    source = inspect.getsourcefile(component.__class__)
    if source is None:
        raise ValueError("component implementation source cannot be authenticated")
    path = Path(source).resolve(strict=True)
    return {
        "class": f"{component.__class__.__module__}.{component.__class__.__qualname__}",
        "identity": normalized,
        "identity_sha256": content_sha256(normalized),
        "implementation_file_sha256": _sha256_bytes(path.read_bytes()),
    }


def _validated_runner_identity(runner: JsonDiscoveryJobRunner) -> dict[str, Any]:
    if not callable(getattr(runner, "identity", None)):
        raise TypeError("runner must expose identity()")
    identity = _clone(runner.identity())
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("runner identity must be one non-empty JSON object")
    declared = _require_sha256(
        identity.get("identity_sha256"),
        label="runner identity_sha256",
    )
    body = {key: value for key, value in identity.items() if key != "identity_sha256"}
    if declared != content_sha256(body):
        raise ValueError("runner identity_sha256 does not authenticate runner identity")
    retry = identity.get("retry")
    if not isinstance(retry, Mapping):
        raise ValueError("runner identity must authenticate its retry policy")
    max_attempts = retry.get("max_attempts")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("runner identity retry.max_attempts must be positive")
    return dict(identity)


def _validated_cache_identity(
    cache: AuthenticatedHierarchicalDiscoveryJobCache,
) -> dict[str, Any]:
    if not isinstance(cache, AuthenticatedHierarchicalDiscoveryJobCache):
        raise TypeError("job cache must be AuthenticatedHierarchicalDiscoveryJobCache")
    identity = _clone(cache.identity())
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("job cache identity must be one non-empty JSON object")
    declared = _require_sha256(
        identity.get("identity_sha256"),
        label="job cache identity_sha256",
    )
    body = {key: value for key, value in identity.items() if key != "identity_sha256"}
    if declared != content_sha256(body):
        raise ValueError("job cache identity_sha256 does not authenticate its identity")
    root = identity.get("root_envelope")
    if not isinstance(root, Mapping) or set(root) != {"kind", "absolute_path"}:
        raise ValueError("job cache identity must bind its exact root envelope")
    if root.get("kind") != "machine_local_absolute_path" or not isinstance(
        root.get("absolute_path"), str
    ):
        raise ValueError("job cache root envelope is invalid")
    return dict(identity)


def _cache_binding(
    cache: AuthenticatedHierarchicalDiscoveryJobCache | None,
    *,
    validator_code_sha256: str,
) -> dict[str, Any]:
    validator_sha256 = _require_sha256(
        validator_code_sha256,
        label="cache validator_code_sha256",
    )
    if cache is None:
        return {
            "mode": "disabled",
            "cache_lookup_allowed": False,
            "cache_write_allowed": False,
            "validator_code_sha256": validator_sha256,
        }
    identity = _validated_cache_identity(cache)
    return {
        "mode": "authenticated_immutable",
        "validator_code_sha256": validator_sha256,
        **_component_binding(cache, identity),
    }


def _reauthenticate_full_manifest(manifest: DirectUpstreamNumericalManifest) -> None:
    """Re-run every nested manifest authentication boundary without trusting init."""

    if not isinstance(manifest, DirectUpstreamNumericalManifest):
        raise TypeError("manifest must be DirectUpstreamNumericalManifest")
    declared_sha256 = manifest.content_sha256
    for matrix in manifest.matrices:
        matrix.__post_init__()
    for coordinate in manifest.coordinates:
        coordinate.__post_init__()
    for coverage in manifest.family_coverage:
        coverage.__post_init__()
    manifest.__post_init__()
    if manifest.content_sha256 != declared_sha256:
        raise ValueError("direct numerical manifest mutated after authentication")


def _validate_reference_only_direct_numerical_contract_body(
    value: Any,
) -> Mapping[str, Any]:
    body = _closed_mapping(
        value,
        keys=_REFERENCE_ONLY_DIRECT_NUMERICAL_CONTRACT_KEYS,
        label="reference-only direct numerical contract",
    )
    if body["schema_version"] != REFERENCE_ONLY_DIRECT_NUMERICAL_CONTRACT_SCHEMA:
        raise ValueError("reference-only direct numerical contract schema changed")
    outer_fold = body["outer_fold"]
    context_epoch = body["context_epoch"]
    if (
        isinstance(outer_fold, bool)
        or not isinstance(outer_fold, int)
        or outer_fold < 1
        or isinstance(context_epoch, bool)
        or not isinstance(context_epoch, int)
        or context_epoch < 0
    ):
        raise ValueError("reference-only direct numerical scope is invalid")
    for key in (
        "plan_scientific_content_sha256",
        "source_execution_content_sha256",
        "reference_manifest_content_sha256",
        "runtime_binding_content_sha256",
        "provider_identity_sha256",
        "projection_content_sha256",
    ):
        _require_sha256(body[key], label=f"reference-only {key}")

    row_scopes: dict[str, tuple[int, ...]] = {}
    for scope in ("spent", "gate"):
        values = body[f"{scope}_row_ids"]
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(row_id, bool)
                or not isinstance(row_id, int)
                or row_id < 0
                for row_id in values
            )
            or len(values) != len(set(values))
            or body[f"{scope}_row_ids_sha256"] != content_sha256(values)
        ):
            raise ValueError(
                f"reference-only direct numerical {scope} rows are invalid"
            )
        row_scopes[scope] = tuple(values)
    if set(row_scopes["spent"]) & set(row_scopes["gate"]):
        raise ValueError("reference-only spent and gate scopes overlap")

    semantic = _closed_mapping(
        body["semantic_catalog"],
        keys=_REFERENCE_ONLY_SEMANTIC_CATALOG_KEYS,
        label="reference-only semantic catalog",
    )
    _require_sha256(
        semantic["catalog_sha256"],
        label="reference-only semantic catalog",
    )
    _require_sha256(
        semantic["split_fingerprint"],
        label="reference-only semantic split",
    )
    if not isinstance(semantic["scope"], str) or not semantic["scope"].strip():
        raise ValueError("reference-only semantic catalog scope is invalid")
    inner_fold = semantic["inner_fold"]
    if (
        inner_fold is not None
        and (
            isinstance(inner_fold, bool)
            or not isinstance(inner_fold, int)
            or inner_fold < 1
        )
    ):
        raise ValueError("reference-only semantic inner fold is invalid")
    atom_count = semantic["atom_count"]
    if (
        isinstance(atom_count, bool)
        or not isinstance(atom_count, int)
        or atom_count < 1
    ):
        raise ValueError("reference-only semantic atom count is invalid")

    rows = body["family_coverage"]
    if (
        not isinstance(rows, list)
        or len(rows) != len(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    ):
        raise ValueError(
            "reference-only direct numerical family coverage is incomplete"
        )
    coordinate_ids_seen: set[str] = set()
    semantic_ids_seen: set[str] = set()
    for family, raw_row in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, rows):
        row = _closed_mapping(
            raw_row,
            keys=_REFERENCE_ONLY_FAMILY_COVERAGE_KEYS,
            label=f"reference-only family coverage {family}",
        )
        if row["source_family"] != family:
            raise ValueError(
                "reference-only direct numerical family order changed"
            )
        coordinate_ids = row["coordinate_ids"]
        semantic_ids = row["semantic_atom_ids"]
        if (
            not isinstance(coordinate_ids, list)
            or not coordinate_ids
            or any(
                not isinstance(coordinate_id, str)
                or not coordinate_id.strip()
                for coordinate_id in coordinate_ids
            )
            or len(coordinate_ids) != len(set(coordinate_ids))
            or coordinate_ids_seen.intersection(coordinate_ids)
            or row["coordinate_ids_sha256"]
            != content_sha256(coordinate_ids)
        ):
            raise ValueError(
                f"reference-only numerical coordinates are invalid for {family}"
            )
        if (
            not isinstance(semantic_ids, list)
            or not semantic_ids
            or any(
                not isinstance(evidence_id, str)
                or not evidence_id.strip()
                for evidence_id in semantic_ids
            )
            or len(semantic_ids) != len(set(semantic_ids))
            or semantic_ids_seen.intersection(semantic_ids)
            or row["semantic_atom_ids_sha256"]
            != content_sha256(semantic_ids)
        ):
            raise ValueError(
                f"reference-only semantic atoms are invalid for {family}"
            )
        coordinate_ids_seen.update(coordinate_ids)
        semantic_ids_seen.update(semantic_ids)
    if len(semantic_ids_seen) != atom_count:
        raise ValueError("reference-only semantic atom count changed")
    if (
        body["already_fit_stage1_projection"] is not True
        or body["conditional_fit_or_refit_performed"] is not False
        or body["row_values_included"] is not False
        or body["coordinate_to_semantic_atom_linkage"] is not False
        or body["concept_grounding_allowed"] is not False
    ):
        raise ValueError(
            "reference-only direct numerical safety declaration changed"
        )
    return body


@dataclass(frozen=True)
class AuthenticatedReferenceOnlyDirectNumericalContract:
    """Non-grounding hierarchy binding to an already-fit reference projection.

    Unlike :class:`FirstGateMaterializationIntent`, this contract never
    represents a deferred conditional fit.  It binds the semantic catalog to
    the exact cumulative projection already authenticated by the role-neutral
    Stage 1 reference bank.
    """

    _body_json: str = field(repr=False)
    content_sha256: str

    def __post_init__(self) -> None:
        body = _validate_reference_only_direct_numerical_contract_body(
            self.body
        )
        if content_sha256(body) != _require_sha256(
            self.content_sha256,
            label="reference-only numerical contract",
        ):
            raise ValueError(
                "reference-only direct numerical contract is unauthenticated"
            )
        object.__setattr__(self, "_body_json", canonical_json(body))

    @classmethod
    def create(
        cls,
        *,
        outer_fold: int,
        context_epoch: int,
        plan_scientific_content_sha256: str,
        source_execution_content_sha256: str,
        reference_manifest_content_sha256: str,
        runtime_binding_content_sha256: str,
        provider_identity_sha256: str,
        spent_row_ids: Sequence[int],
        gate_row_ids: Sequence[int],
        catalog: RoleNeutralEvidenceCatalog,
        family_coordinate_ids: Mapping[str, Sequence[str]],
        projection_content_sha256: str,
    ) -> "AuthenticatedReferenceOnlyDirectNumericalContract":
        validate_role_neutral_catalog(catalog)
        spent = tuple(int(value) for value in spent_row_ids)
        gate = tuple(int(value) for value in gate_row_ids)
        if (
            isinstance(outer_fold, bool)
            or not isinstance(outer_fold, int)
            or outer_fold < 1
            or isinstance(context_epoch, bool)
            or not isinstance(context_epoch, int)
            or context_epoch < 0
            or not spent
            or not gate
            or len(spent) != len(set(spent))
            or len(gate) != len(set(gate))
            or set(spent) & set(gate)
            or catalog.outer_fold != outer_fold
        ):
            raise ValueError("reference-only direct numerical scope is invalid")
        if set(family_coordinate_ids) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError(
                "reference-only direct numerical family coordinates are incomplete"
            )
        family_rows = []
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
            coordinates = tuple(
                str(value) for value in family_coordinate_ids.get(family, ())
            )
            if not coordinates or len(coordinates) != len(set(coordinates)):
                raise ValueError(
                    f"reference-only direct numerical coordinates are invalid for {family}"
                )
            semantic_ids = tuple(
                atom.evidence_id for atom in catalog.family_atoms(family)
            )
            if not semantic_ids:
                raise ValueError(
                    f"reference-only semantic catalog is empty for {family}"
                )
            family_rows.append(
                {
                    "source_family": family,
                    "coordinate_ids": list(coordinates),
                    "coordinate_ids_sha256": content_sha256(list(coordinates)),
                    "semantic_atom_ids": list(semantic_ids),
                    "semantic_atom_ids_sha256": content_sha256(
                        list(semantic_ids)
                    ),
                }
            )
        body = {
            "schema_version": REFERENCE_ONLY_DIRECT_NUMERICAL_CONTRACT_SCHEMA,
            "outer_fold": outer_fold,
            "context_epoch": context_epoch,
            "plan_scientific_content_sha256": _require_sha256(
                plan_scientific_content_sha256,
                label="reference-only scope plan",
            ),
            "source_execution_content_sha256": _require_sha256(
                source_execution_content_sha256,
                label="reference-only source execution",
            ),
            "reference_manifest_content_sha256": _require_sha256(
                reference_manifest_content_sha256,
                label="reference-only numerical manifest",
            ),
            "runtime_binding_content_sha256": _require_sha256(
                runtime_binding_content_sha256,
                label="reference-only runtime binding",
            ),
            "provider_identity_sha256": _require_sha256(
                provider_identity_sha256,
                label="reference-only provider identity",
            ),
            "projection_content_sha256": _require_sha256(
                projection_content_sha256,
                label="reference-only cumulative projection",
            ),
            "spent_row_ids": list(spent),
            "spent_row_ids_sha256": content_sha256(list(spent)),
            "gate_row_ids": list(gate),
            "gate_row_ids_sha256": content_sha256(list(gate)),
            "semantic_catalog": {
                "catalog_sha256": catalog.catalog_sha256,
                "scope": catalog.scope,
                "inner_fold": catalog.inner_fold,
                "split_fingerprint": catalog.split_fingerprint,
                "atom_count": len(catalog.atoms),
            },
            "family_coverage": family_rows,
            "already_fit_stage1_projection": True,
            "conditional_fit_or_refit_performed": False,
            "row_values_included": False,
            "coordinate_to_semantic_atom_linkage": False,
            "concept_grounding_allowed": False,
        }
        return cls(
            _body_json=canonical_json(body),
            content_sha256=content_sha256(body),
        )

    @property
    def body(self) -> dict[str, Any]:
        value = json.loads(self._body_json)
        if not isinstance(value, dict):
            raise RuntimeError("reference-only numerical contract was corrupted")
        return value

    def as_dict(self) -> dict[str, Any]:
        return {**self.body, "content_sha256": self.content_sha256}

    def verify(
        self,
        *,
        catalog: RoleNeutralEvidenceCatalog,
    ) -> None:
        validate_role_neutral_catalog(catalog)
        body = _validate_reference_only_direct_numerical_contract_body(
            self.body
        )
        if (
            body.get("schema_version")
            != REFERENCE_ONLY_DIRECT_NUMERICAL_CONTRACT_SCHEMA
            or content_sha256(body) != self.content_sha256
            or body.get("outer_fold") != catalog.outer_fold
            or body.get("semantic_catalog")
            != {
                "catalog_sha256": catalog.catalog_sha256,
                "scope": catalog.scope,
                "inner_fold": catalog.inner_fold,
                "split_fingerprint": catalog.split_fingerprint,
                "atom_count": len(catalog.atoms),
            }
            or body.get("already_fit_stage1_projection") is not True
            or body.get("conditional_fit_or_refit_performed") is not False
            or body.get("row_values_included") is not False
            or body.get("coordinate_to_semantic_atom_linkage") is not False
            or body.get("concept_grounding_allowed") is not False
        ):
            raise ValueError(
                "reference-only direct numerical contract is invalid"
            )
        rows = body.get("family_coverage")
        if (
            not isinstance(rows, list)
            or [row.get("source_family") for row in rows if isinstance(row, Mapping)]
            != list(ACTIVE_STAGE1_CONCEPT_FAMILIES)
        ):
            raise ValueError(
                "reference-only direct numerical family coverage is incomplete"
            )
        for family, row in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, rows):
            if not isinstance(row, Mapping):
                raise TypeError("reference-only family coverage is malformed")
            coordinates = row.get("coordinate_ids")
            semantic_ids = [
                atom.evidence_id for atom in catalog.family_atoms(family)
            ]
            if (
                not isinstance(coordinates, list)
                or not coordinates
                or len(coordinates) != len(set(coordinates))
                or row.get("coordinate_ids_sha256")
                != content_sha256(coordinates)
                or row.get("semantic_atom_ids") != semantic_ids
                or row.get("semantic_atom_ids_sha256")
                != content_sha256(semantic_ids)
            ):
                raise ValueError(
                    f"reference-only numerical family binding changed for {family}"
                )


def direct_numerical_bindings_from_reference_contract(
    contract: AuthenticatedReferenceOnlyDirectNumericalContract,
    *,
    catalog: RoleNeutralEvidenceCatalog,
) -> tuple[DirectNumericalDossierBinding, ...]:
    if type(contract) is not AuthenticatedReferenceOnlyDirectNumericalContract:
        raise TypeError("reference-only numerical contract has the wrong type")
    contract.verify(catalog=catalog)
    coverage = contract.body["family_coverage"]
    return tuple(
        DirectNumericalDossierBinding(
            source_family=family,
            signal_count=len(row["coordinate_ids"]),
            zero_reason="",
            direct_numerical_contract_kind=(
                DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
            ),
            direct_numerical_contract_sha256=contract.content_sha256,
            manifest_sha256=contract.content_sha256,
        )
        for family, row in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, coverage)
    )


def direct_numerical_bindings_from_manifest(
    manifest: DirectUpstreamNumericalManifest,
) -> tuple[DirectNumericalDossierBinding, ...]:
    """Derive the only ten dossier bindings compatible with ``manifest``."""

    _reauthenticate_full_manifest(manifest)
    return tuple(
        DirectNumericalDossierBinding(
            source_family=coverage.source_family,
            signal_count=len(coverage.coordinate_ids),
            zero_reason=coverage.numerical_zero_reason,
            direct_numerical_contract_kind=(DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST),
            direct_numerical_contract_sha256=manifest.content_sha256,
            manifest_sha256=manifest.content_sha256,
        )
        for coverage in manifest.family_coverage
    )


def _validate_catalog_manifest_binding(
    *,
    catalog: RoleNeutralEvidenceCatalog,
    manifest: DirectUpstreamNumericalManifest,
) -> None:
    validate_role_neutral_catalog(catalog)
    if manifest.semantic_catalog_sha256 != catalog.catalog_sha256:
        raise ValueError("direct numerical manifest binds a different semantic catalog")
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        catalog_ids = tuple(atom.evidence_id for atom in catalog.family_atoms(family))
        coverage = manifest.family(family)
        if coverage.semantic_atom_ids != catalog_ids:
            raise ValueError(f"direct numerical manifest evidence IDs differ for {family}")
        if len(coverage.semantic_atom_ids) != len(catalog_ids):
            raise ValueError(f"direct numerical manifest evidence count differs for {family}")


def _intent_family_coverage(
    intent: FirstGateMaterializationIntent,
    *,
    catalog: RoleNeutralEvidenceCatalog,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(intent, FirstGateMaterializationIntent):
        raise TypeError("first_gate_materialization_intent has the wrong type")
    intent.verify()
    validate_role_neutral_catalog(catalog)
    body = intent.body
    semantic = body.get("semantic_catalog")
    schema = body.get("coordinate_schema")
    if not isinstance(semantic, Mapping) or not isinstance(schema, Mapping):
        raise ValueError("first-gate intent omits semantic or coordinate bindings")
    exact_catalog_fields = {
        "catalog_sha256": catalog.catalog_sha256,
        "scope": catalog.scope,
        "inner_fold": catalog.inner_fold,
        "split_fingerprint": catalog.split_fingerprint,
        "atom_count": len(catalog.atoms),
    }
    for name, expected in exact_catalog_fields.items():
        if semantic.get(name) != expected:
            raise ValueError(f"first-gate intent binds a different catalog {name}")
    if body.get("outer_fold") != catalog.outer_fold:
        raise ValueError("first-gate intent binds a different outer fold")
    family_bindings = semantic.get("family_bindings")
    if not isinstance(family_bindings, list) or [
        row.get("source_family") for row in family_bindings if isinstance(row, Mapping)
    ] != list(ACTIVE_STAGE1_CONCEPT_FAMILIES):
        raise ValueError("first-gate intent semantic families are incomplete or unordered")
    for family, row in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, family_bindings):
        if not isinstance(row, Mapping):
            raise TypeError("first-gate intent family binding must be one JSON object")
        expected_ids = tuple(atom.evidence_id for atom in catalog.family_atoms(family))
        if tuple(row.get("semantic_atom_ids") or ()) != expected_ids:
            raise ValueError(f"first-gate intent evidence IDs differ for {family}")
        if row.get("semantic_atom_ids_sha256") != content_sha256(list(expected_ids)):
            raise ValueError(f"first-gate intent evidence hash differs for {family}")
    coverage = schema.get("family_coverage")
    if not isinstance(coverage, list) or [
        row.get("source_family") for row in coverage if isinstance(row, Mapping)
    ] != list(ACTIVE_STAGE1_CONCEPT_FAMILIES):
        raise ValueError("first-gate intent coordinate coverage is incomplete or unordered")
    for family, semantic_row, coverage_row in zip(
        ACTIVE_STAGE1_CONCEPT_FAMILIES,
        family_bindings,
        coverage,
    ):
        if not isinstance(coverage_row, Mapping):
            raise TypeError("first-gate intent coverage row must be one JSON object")
        if coverage_row.get("semantic_atom_ids") != semantic_row.get("semantic_atom_ids"):
            raise ValueError(f"first-gate intent coverage atom IDs differ for {family}")
        if coverage_row.get("semantic_atom_ids_sha256") != semantic_row.get(
            "semantic_atom_ids_sha256"
        ):
            raise ValueError(f"first-gate intent coverage atom hash differs for {family}")
        coordinate_ids = coverage_row.get("coordinate_ids")
        if not isinstance(coordinate_ids, list) or len(coordinate_ids) != len(set(coordinate_ids)):
            raise ValueError(f"first-gate intent coordinate IDs are invalid for {family}")
        zero_reason = coverage_row.get("numerical_zero_reason")
        if not isinstance(zero_reason, str):
            raise TypeError("first-gate intent numerical zero reason must be a string")
        if bool(coordinate_ids) == bool(zero_reason):
            raise ValueError(
                f"first-gate intent signal count/zero reason is inconsistent for {family}"
            )
    return tuple(coverage)


def direct_numerical_bindings_from_intent(
    intent: FirstGateMaterializationIntent,
    *,
    catalog: RoleNeutralEvidenceCatalog,
) -> tuple[DirectNumericalDossierBinding, ...]:
    """Derive all ten dossier bindings from an authenticated pre-fit intent."""

    coverage = _intent_family_coverage(intent, catalog=catalog)
    return tuple(
        DirectNumericalDossierBinding(
            source_family=family,
            signal_count=len(row["coordinate_ids"]),
            zero_reason=row["numerical_zero_reason"],
            direct_numerical_contract_kind=(DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT),
            direct_numerical_contract_sha256=intent.content_sha256,
        )
        for family, row in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, coverage)
    )


def _validate_explicit_bindings(
    *,
    supplied: Sequence[DirectNumericalDossierBinding],
    expected: Sequence[DirectNumericalDossierBinding],
) -> tuple[DirectNumericalDossierBinding, ...]:
    bindings = tuple(supplied)
    if len(bindings) != len(ACTIVE_STAGE1_CONCEPT_FAMILIES):
        raise ValueError("direct numerical bindings must contain exactly ten entries")
    if not all(isinstance(row, DirectNumericalDossierBinding) for row in bindings):
        raise TypeError("direct numerical bindings contain an invalid entry")
    if tuple(row.source_family for row in bindings) != ACTIVE_STAGE1_CONCEPT_FAMILIES:
        raise ValueError("direct numerical bindings must use canonical architecture order")
    if tuple(row.as_dossier_dict() for row in bindings) != tuple(
        row.as_dossier_dict() for row in expected
    ):
        raise ValueError("direct numerical bindings differ from the approved contract")
    return bindings


def _safe_manifest_binding(manifest: DirectUpstreamNumericalManifest) -> dict[str, Any]:
    """Project a full local manifest into an offline-safe non-grounding binding."""

    return {
        "direct_numerical_contract_kind": (DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST),
        "direct_numerical_contract_sha256": manifest.content_sha256,
        "source_cache_schema": manifest.source_cache_schema,
        "source_cache_key": manifest.source_cache_key,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "producer_identity_sha256": manifest.producer_identity_sha256,
        "stable_output_schema_sha256": manifest.stable_output_schema_sha256,
        "semantic_catalog_sha256": manifest.semantic_catalog_sha256,
        "shared_lineage_sha256": manifest.shared_lineage_sha256,
        "lineage_scope": manifest.lineage_scope,
        "signal_count": manifest.signal_count,
        "families": [
            {
                "source_family": coverage.source_family,
                "semantic_atom_ids": list(coverage.semantic_atom_ids),
                "semantic_atom_ids_sha256": coverage.semantic_atom_ids_sha256,
                "semantic_atom_count": len(coverage.semantic_atom_ids),
                "signal_count": len(coverage.coordinate_ids),
                "numerical_zero_reason": coverage.numerical_zero_reason,
            }
            for coverage in manifest.family_coverage
        ],
        "row_values_included": False,
        "matrix_metadata_included": False,
        "coordinate_metadata_included": False,
        "coordinate_to_semantic_atom_linkage": False,
        "concept_grounding_allowed": False,
    }


def _safe_intent_binding(
    intent: FirstGateMaterializationIntent,
    *,
    catalog: RoleNeutralEvidenceCatalog,
) -> dict[str, Any]:
    coverage = _intent_family_coverage(intent, catalog=catalog)
    body = intent.body
    semantic = body["semantic_catalog"]
    schema = body["coordinate_schema"]
    return {
        "direct_numerical_contract_kind": (DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT),
        "direct_numerical_contract_sha256": intent.content_sha256,
        "source_cache_key": body["source_cache_key"],
        "stable_output_schema_sha256": schema["stable_output_schema_sha256"],
        "semantic_catalog_sha256": semantic["catalog_sha256"],
        "expected_shared_lineage_sha256": schema["expected_shared_lineage_sha256"],
        "lineage_scope": schema["lineage_scope"],
        "signal_count": sum(len(row["coordinate_ids"]) for row in coverage),
        "families": [
            {
                "source_family": family,
                "semantic_atom_ids": list(row["semantic_atom_ids"]),
                "semantic_atom_ids_sha256": row["semantic_atom_ids_sha256"],
                "semantic_atom_count": len(row["semantic_atom_ids"]),
                "signal_count": len(row["coordinate_ids"]),
                "numerical_zero_reason": row["numerical_zero_reason"],
            }
            for family, row in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, coverage)
        ],
        "materialization_state": "deferred_until_after_approval_and_proposal_freeze",
        "row_values_included": False,
        "matrix_metadata_included": False,
        "coordinate_metadata_included": False,
        "coordinate_to_semantic_atom_linkage": False,
        "concept_grounding_allowed": False,
    }


def _safe_reference_contract_binding(
    contract: AuthenticatedReferenceOnlyDirectNumericalContract,
    *,
    catalog: RoleNeutralEvidenceCatalog,
) -> dict[str, Any]:
    contract.verify(catalog=catalog)
    body = contract.body
    return {
        "direct_numerical_contract_kind": (
            DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
        ),
        "direct_numerical_contract_sha256": contract.content_sha256,
        "reference_manifest_content_sha256": body[
            "reference_manifest_content_sha256"
        ],
        "projection_content_sha256": body["projection_content_sha256"],
        "semantic_catalog_sha256": body["semantic_catalog"][
            "catalog_sha256"
        ],
        "signal_count": sum(
            len(row["coordinate_ids"])
            for row in body["family_coverage"]
        ),
        "families": [
            {
                "source_family": row["source_family"],
                "semantic_atom_ids": list(row["semantic_atom_ids"]),
                "semantic_atom_ids_sha256": row[
                    "semantic_atom_ids_sha256"
                ],
                "semantic_atom_count": len(row["semantic_atom_ids"]),
                "signal_count": len(row["coordinate_ids"]),
                "numerical_zero_reason": "",
            }
            for row in body["family_coverage"]
        ],
        "materialization_state": "already_fit_stage1_reference_projection",
        "conditional_fit_or_refit_performed": False,
        "row_values_included": False,
        "matrix_metadata_included": False,
        "coordinate_metadata_included": False,
        "coordinate_to_semantic_atom_linkage": False,
        "concept_grounding_allowed": False,
    }


@dataclass(frozen=True)
class ApprovedHierarchicalDiscoveryPrecommit:
    approval_sha256: str
    _packet_json: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_sha256(self.approval_sha256, label="approval_sha256")
        try:
            packet = json.loads(self._packet_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("approved discovery packet is invalid JSON") from exc
        if not isinstance(packet, Mapping):
            raise TypeError("approved discovery packet must be one JSON object")
        if self.approval_sha256 != content_sha256(packet):
            raise ValueError("approval_sha256 does not authenticate the offline packet")

    @classmethod
    def create(cls, packet: Mapping[str, Any]) -> "ApprovedHierarchicalDiscoveryPrecommit":
        detached = _clone(packet)
        return cls(
            approval_sha256=content_sha256(detached),
            _packet_json=canonical_json(detached),
        )

    @property
    def packet(self) -> dict[str, Any]:
        return json.loads(self._packet_json)

    def render_json(self, *, indent: int = 2) -> str:
        if isinstance(indent, bool) or not isinstance(indent, int) or indent < 0:
            raise ValueError("indent must be a non-negative integer")
        return json.dumps(
            {"approval_sha256": self.approval_sha256, "packet": self.packet},
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )


@runtime_checkable
class MetadataJsonDiscoveryJobRunner(JsonDiscoveryJobRunner, Protocol):
    """JSON runner that exposes detached metadata for every attempted job."""

    @property
    def execution_metadata(self) -> Sequence[Mapping[str, Any]]:
        raise NotImplementedError


def _validate_reasoning_hashes(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("runner reasoning_hashes must be one JSON object")
    for key, digest in value.items():
        if not isinstance(key, str) or not key.endswith("_sha256"):
            raise ValueError("runner reasoning metadata may contain hashes only")
        _require_sha256(digest, label=f"reasoning_hashes.{key}")


def _validate_remote_response_record(
    *,
    record: Mapping[str, Any],
    expected_job_id: str,
    expected_job_kind: str,
    expected_validation_outcome: str,
    expected_raw_response_projection_sha256: str,
    runner_identity_sha256: str,
    endpoints: Sequence[str],
    model_name: str,
    max_attempts: int,
    label: str,
) -> None:
    if expected_validation_outcome in _REMOTE_TRANSPORT_FAILURES:
        row = _closed_mapping(record, keys=_INVALID_RESPONSE_METADATA_KEYS, label=label)
        if row["outcome"] != "invalid_response":
            raise ValueError("strict-JSON failure requires invalid-response runner metadata")
    else:
        row = _closed_mapping(record, keys=_SUCCESS_METADATA_KEYS, label=label)
        if row["outcome"] != "success":
            raise ValueError("parsed response metadata must report success")
        if row["parsed_response_sha256"] != expected_raw_response_projection_sha256:
            raise ValueError("runner parsed response differs from its authenticated raw projection")
    if row["job_id"] != expected_job_id or row["job_kind"] != expected_job_kind:
        raise ValueError("runner metadata job binding differs from response attempt trace")
    request_sha256 = _require_sha256(row["request_sha256"], label="runner request_sha256")
    if row["runner_identity_sha256"] != runner_identity_sha256:
        raise ValueError("runner metadata cites a different runner identity")
    attempts = row["attempts"]
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("runner metadata requires at least one transport attempt")
    if len(attempts) > max_attempts:
        raise ValueError("runner attempts exceed the authenticated transport-retry bound")
    for attempt_index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping):
            raise TypeError("runner transport attempt must be one JSON object")
        if not _ATTEMPT_REQUIRED_KEYS <= set(attempt) <= _ATTEMPT_ALLOWED_KEYS:
            raise ValueError("runner transport attempt has an unexpected closed schema")
        if attempt["attempt_number"] != attempt_index:
            raise ValueError("runner transport attempt numbers must be contiguous")
        if attempt["endpoint"] not in endpoints or attempt["model"] != model_name:
            raise ValueError("runner attempt endpoint/model differs from runner identity")
        if (
            attempt["request_sha256"] != request_sha256
            or attempt["runner_identity_sha256"] != runner_identity_sha256
        ):
            raise ValueError("runner attempt authentication differs from its record")
        final = attempt_index == len(attempts)
        expected_outcome = (
            "invalid_response"
            if final and expected_validation_outcome in _REMOTE_TRANSPORT_FAILURES
            else "success" if final else "transport_error"
        )
        if attempt["outcome"] != expected_outcome:
            raise ValueError("runner attempt outcome differs from the response attempt trace")
        if final:
            if attempt["retryable"] is not False or attempt["will_retry"] is not False:
                raise ValueError("final response attempt cannot request a transport retry")
            if attempt.get("response_model") != model_name:
                raise ValueError("runner response model differs from the exact requested model")
            if attempt.get("finish_reason") != "stop":
                raise ValueError("runner response finish_reason must be exactly 'stop'")
            content_sha256 = _require_sha256(
                attempt.get("content_sha256"), label="attempt content_sha256"
            )
            raw_transport_bytes = attempt.get("raw_transport_bytes")
            if (
                isinstance(raw_transport_bytes, bool)
                or not isinstance(raw_transport_bytes, int)
                or raw_transport_bytes < 1
            ):
                raise ValueError("final runner attempt requires a positive raw byte count")
            if expected_validation_outcome in _REMOTE_TRANSPORT_FAILURES:
                if content_sha256 != expected_raw_response_projection_sha256:
                    raise ValueError("invalid response content hash differs from repair trace")
            else:
                if attempt.get("parsed_response_sha256") != (
                    expected_raw_response_projection_sha256
                ):
                    raise ValueError("final parsed response hash differs from repair trace")
            _validate_reasoning_hashes(attempt.get("reasoning_hashes"))
            if not isinstance(attempt.get("usage"), Mapping):
                raise TypeError("final runner attempt requires usage metadata")
        elif attempt["retryable"] is not True or attempt["will_retry"] is not True:
            raise ValueError("an intermediate transport failure must be retryable")


class _PerCallMetadataAuthenticatingRunner:
    """Authenticate each remote wire object before it reaches the cache boundary.

    The approval wrapper's final execution trace remains a complete second
    validation.  This earlier boundary is necessary because the hierarchy may
    persist one semantically valid job before the complete wrapper result is
    assembled.  A response whose runner metadata does not authenticate the raw
    parsed object must therefore fail before semantic validation or cache write.
    """

    def __init__(
        self,
        *,
        runner: MetadataJsonDiscoveryJobRunner,
        runner_identity: Mapping[str, Any],
    ) -> None:
        self._runner = runner
        self._runner_identity = _clone(runner_identity)
        endpoints = self._runner_identity.get("endpoint_urls")
        model = self._runner_identity.get("model")
        retry = self._runner_identity.get("retry")
        if not isinstance(endpoints, list) or not all(isinstance(row, str) for row in endpoints):
            raise ValueError("runner identity endpoint_urls are invalid")
        if not isinstance(model, Mapping) or not isinstance(model.get("name"), str):
            raise ValueError("runner identity model is invalid")
        if not isinstance(retry, Mapping):
            raise ValueError("runner identity retry policy is invalid")
        max_attempts = retry.get("max_attempts")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("runner identity transport retry bound is invalid")
        self._endpoints = tuple(endpoints)
        self._model_name = model["name"]
        self._max_attempts = max_attempts

    def identity(self) -> Mapping[str, Any]:
        current = _validated_runner_identity(self._runner)
        if canonical_json(current) != canonical_json(self._runner_identity):
            raise ValueError("runner identity changed during per-call authentication")
        return _clone(current)

    @property
    def execution_metadata(self) -> Sequence[Mapping[str, Any]]:
        return self._runner.execution_metadata

    def _single_appended_record(
        self,
        *,
        before: Sequence[Mapping[str, Any]],
        label: str,
    ) -> Mapping[str, Any]:
        after = tuple(_clone(row) for row in self._runner.execution_metadata)
        if tuple(after[: len(before)]) != tuple(before):
            raise ValueError("runner execution metadata mutated during remote call")
        if len(after) != len(before) + 1:
            raise ValueError("runner must append exactly one metadata record per remote call")
        record = after[-1]
        if not isinstance(record, Mapping):
            raise TypeError(f"{label} must be one JSON object")
        return record

    def _validate_record(
        self,
        *,
        record: Mapping[str, Any],
        job: Any,
        validation_outcome: str,
        raw_response_projection_sha256: str,
        label: str,
    ) -> None:
        _validate_remote_response_record(
            record=record,
            expected_job_id=job.job_id,
            expected_job_kind=job.job_kind,
            expected_validation_outcome=validation_outcome,
            expected_raw_response_projection_sha256=(raw_response_projection_sha256),
            runner_identity_sha256=self._runner_identity["identity_sha256"],
            endpoints=self._endpoints,
            model_name=self._model_name,
            max_attempts=self._max_attempts,
            label=label,
        )

    def run_json(self, *, job: Any) -> Mapping[str, Any]:
        self.identity()
        before = tuple(_clone(row) for row in self._runner.execution_metadata)
        try:
            response = self._runner.run_json(job=job)
        except Exception as exc:
            category = getattr(exc, "discovery_response_failure_category", None)
            failed_content = getattr(exc, "failed_response_content", None)
            if category in _REMOTE_TRANSPORT_FAILURES and isinstance(failed_content, str):
                record = self._single_appended_record(
                    before=before,
                    label="strict-JSON runner metadata",
                )
                self._validate_record(
                    record=record,
                    job=job,
                    validation_outcome=category,
                    raw_response_projection_sha256=hashlib.sha256(
                        failed_content.encode("utf-8")
                    ).hexdigest(),
                    label="strict-JSON runner metadata",
                )
                self.identity()
            raise
        wire = _clone(response)
        if not isinstance(wire, Mapping):
            raise TypeError("runner must return one JSON object")
        record = self._single_appended_record(
            before=before,
            label="successful runner metadata",
        )
        self._validate_record(
            record=record,
            job=job,
            validation_outcome=VALIDATED_RESPONSE,
            raw_response_projection_sha256=content_sha256(wire),
            label="successful runner metadata",
        )
        self.identity()
        return wire


def _validate_execution_records(
    *,
    completed: CompletedHierarchicalDiscovery,
    runner_identity: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    validator_code_sha256: str,
    cache_identity: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    detached = tuple(_clone(row) for row in records)
    jobs = completed.execution_ledger.job_ledger.jobs
    results = completed.execution_ledger.results
    if len(detached) != len(jobs):
        raise ValueError("runner metadata must contain exactly one record per hierarchy job")
    runner_identity_sha256 = _require_sha256(
        runner_identity.get("identity_sha256"), label="runner identity_sha256"
    )
    validator_sha256 = _require_sha256(
        validator_code_sha256,
        label="validator_code_sha256",
    )
    normalized_cache_identity = None if cache_identity is None else _clone(cache_identity)
    if normalized_cache_identity is not None:
        if not isinstance(normalized_cache_identity, Mapping):
            raise TypeError("cache identity must be one JSON object")
        cache_identity_sha256 = _require_sha256(
            normalized_cache_identity.get("identity_sha256"),
            label="cache identity_sha256",
        )
        cache_body = {
            key: value
            for key, value in normalized_cache_identity.items()
            if key != "identity_sha256"
        }
        if cache_identity_sha256 != content_sha256(cache_body):
            raise ValueError("cache identity_sha256 does not authenticate its identity")
    else:
        cache_identity_sha256 = None
    retry = runner_identity.get("retry")
    if not isinstance(retry, Mapping):
        raise ValueError("runner retry identity is missing")
    max_attempts = retry.get("max_attempts")
    endpoints = runner_identity.get("endpoint_urls")
    model = runner_identity.get("model")
    if not isinstance(endpoints, list) or not all(isinstance(row, str) for row in endpoints):
        raise ValueError("runner identity endpoint_urls are invalid")
    if not isinstance(model, Mapping) or not isinstance(model.get("name"), str):
        raise ValueError("runner identity model is invalid")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("runner identity transport retry bound is invalid")

    for index, (record, job, result) in enumerate(zip(detached, jobs, results)):
        if record.get("record_type") == "authenticated_cache_hit":
            row = _closed_mapping(
                record,
                keys=_CACHE_HIT_METADATA_KEYS,
                label=f"cache execution metadata[{index}]",
            )
            if normalized_cache_identity is None or cache_identity_sha256 is None:
                raise ValueError("cache-hit metadata is forbidden when caching is disabled")
            if (
                row["schema_version"] != HIERARCHICAL_DISCOVERY_JOB_CACHE_HIT_VERSION
                or row["outcome"] != "cache_hit"
            ):
                raise ValueError("cache-hit metadata has the wrong schema or outcome")
            if row["job_id"] != job.job_id or row["job_kind"] != job.job_kind:
                raise ValueError("cache metadata job binding differs from execution ledger")
            if row["job_sha256"] != content_sha256(job.as_dict()):
                raise ValueError("cache metadata does not authenticate the exact job")
            if row["runner_identity_sha256"] != runner_identity_sha256:
                raise ValueError("cache metadata cites a different runner identity")
            if row["hierarchy_inner_precommit_sha256"] != completed.precommit_sha256:
                raise ValueError("cache metadata cites a different hierarchy precommit")
            if row["validator_code_sha256"] != validator_sha256:
                raise ValueError("cache metadata cites different semantic-validator code")
            if row["cache_identity_sha256"] != cache_identity_sha256:
                raise ValueError("cache metadata cites a different cache identity")
            if row["validated_response_sha256"] != result.response_sha256:
                raise ValueError("cache response hash differs from validated hierarchy result")
            wire_response = row["wire_response"]
            if not isinstance(wire_response, Mapping):
                raise TypeError("cache wire_response must be one JSON object")
            if row["wire_response_sha256"] != content_sha256(wire_response):
                raise ValueError("cache wire response hash does not authenticate its content")
            final_attempt = result.response_attempt_trace["attempts"][-1]
            if row["wire_response_sha256"] != final_attempt["raw_response_projection_sha256"]:
                raise ValueError("cache wire response differs from hierarchy response trace")
            if row["response_attempt_trace_sha256"] != (result.response_attempt_trace_sha256):
                raise ValueError("cache response-attempt trace differs from hierarchy result")
            lookup = {
                "schema_version": HIERARCHICAL_DISCOVERY_JOB_CACHE_LOOKUP_VERSION,
                "cache_identity_sha256": cache_identity_sha256,
                "hierarchy_inner_precommit_sha256": completed.precommit_sha256,
                "runner_identity": _clone(runner_identity),
                "runner_identity_sha256": runner_identity_sha256,
                "validator_code_sha256": validator_sha256,
                "job": job.as_dict(),
                "job_id": job.job_id,
            }
            lookup_sha256 = content_sha256(lookup)
            if row["cache_lookup_sha256"] != lookup_sha256:
                raise ValueError("cache metadata lookup hash differs from the exact job context")
            entry_body = {
                "schema_version": HIERARCHICAL_DISCOVERY_JOB_CACHE_ENTRY_VERSION,
                "lookup_identity": lookup,
                "lookup_sha256": lookup_sha256,
                "wire_response": wire_response,
                "wire_response_sha256": row["wire_response_sha256"],
                "validated_response": result.response,
                "validated_response_sha256": result.response_sha256,
                "response_attempt_trace": result.response_attempt_trace,
                "response_attempt_trace_sha256": result.response_attempt_trace_sha256,
            }
            if row["cache_entry_sha256"] != content_sha256(entry_body):
                raise ValueError("cache metadata entry hash differs from the validated result")
            record_body = {key: value for key, value in row.items() if key != "record_sha256"}
            if row["record_sha256"] != content_sha256(record_body):
                raise ValueError("cache record_sha256 does not authenticate its metadata")
            continue
        trace = result.response_attempt_trace
        trace_attempts = trace["attempts"]
        if trace_attempts[-1]["normalized_validated_response_sha256"] != result.response_sha256:
            raise ValueError("response trace normalized hash differs from hierarchy result")
        if record.get("record_type") == "authenticated_remote_response_repair_sequence":
            row = _closed_mapping(
                record,
                keys=_REMOTE_RESPONSE_REPAIR_SEQUENCE_KEYS,
                label=f"remote response-repair sequence[{index}]",
            )
            if (
                row["schema_version"] != _REMOTE_RESPONSE_REPAIR_SEQUENCE_VERSION
                or row["outcome"] != "success_after_one_response_repair"
            ):
                raise ValueError("remote response-repair sequence has the wrong schema/outcome")
            if row["job_id"] != job.job_id or row["job_kind"] != job.job_kind:
                raise ValueError("remote response-repair sequence changed its logical job")
            if (
                row["response_attempt_trace"] != trace
                or row["response_attempt_trace_sha256"] != result.response_attempt_trace_sha256
            ):
                raise ValueError("remote response-repair sequence changed its attempt trace")
            if row["validated_response_sha256"] != result.response_sha256:
                raise ValueError("remote response-repair sequence changed its final response")
            remote_rows = row["remote_records"]
            if not isinstance(remote_rows, list) or len(remote_rows) != 2:
                raise ValueError("remote response repair requires exactly two response records")
            if len(trace_attempts) != 2:
                raise ValueError("remote repair record requires one authenticated trace repair")
            for attempt_index, (remote, response_attempt) in enumerate(
                zip(remote_rows, trace_attempts), start=1
            ):
                _validate_remote_response_record(
                    record=remote,
                    expected_job_id=response_attempt["job_id"],
                    expected_job_kind=response_attempt["job_kind"],
                    expected_validation_outcome=response_attempt["validation_outcome"],
                    expected_raw_response_projection_sha256=response_attempt[
                        "raw_response_projection_sha256"
                    ],
                    runner_identity_sha256=runner_identity_sha256,
                    endpoints=endpoints,
                    model_name=model["name"],
                    max_attempts=max_attempts,
                    label=f"remote response repair[{index}].attempt[{attempt_index}]",
                )
            record_body = {key: value for key, value in row.items() if key != "record_sha256"}
            if row["record_sha256"] != content_sha256(record_body):
                raise ValueError("remote response-repair record SHA-256 does not authenticate")
            continue
        if "record_type" in record:
            raise ValueError("execution metadata has an unsupported record_type")
        if len(trace_attempts) != 1:
            raise ValueError("a repaired result requires composite runner metadata")
        response_attempt = trace_attempts[0]
        _validate_remote_response_record(
            record=record,
            expected_job_id=job.job_id,
            expected_job_kind=job.job_kind,
            expected_validation_outcome=VALIDATED_RESPONSE,
            expected_raw_response_projection_sha256=response_attempt[
                "raw_response_projection_sha256"
            ],
            runner_identity_sha256=runner_identity_sha256,
            endpoints=endpoints,
            model_name=model["name"],
            max_attempts=max_attempts,
            label=f"runner execution metadata[{index}]",
        )
    return detached


@dataclass(frozen=True)
class AuthenticatedRunnerExecutionTrace:
    hierarchy_execution_sha256: str
    runner_identity_sha256: str
    validator_code_sha256: str
    cache_identity_sha256: str | None
    trace_sha256: str
    _runner_identity_json: str = field(repr=False)
    _cache_identity_json: str = field(repr=False)
    _records_json: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_sha256(self.hierarchy_execution_sha256, label="hierarchy_execution_sha256")
        _require_sha256(self.runner_identity_sha256, label="runner_identity_sha256")
        _require_sha256(self.validator_code_sha256, label="validator_code_sha256")
        _require_sha256(self.trace_sha256, label="trace_sha256")
        identity = json.loads(self._runner_identity_json)
        cache_identity = json.loads(self._cache_identity_json)
        records = json.loads(self._records_json)
        if not isinstance(identity, Mapping) or not isinstance(records, list):
            raise ValueError("runner trace contains invalid authenticated JSON")
        if cache_identity is None:
            if self.cache_identity_sha256 is not None:
                raise ValueError("disabled cache trace cannot cite a cache identity SHA-256")
        else:
            if not isinstance(cache_identity, Mapping):
                raise ValueError("cache trace identity must be one JSON object or null")
            _require_sha256(self.cache_identity_sha256, label="cache_identity_sha256")
            if cache_identity.get("identity_sha256") != self.cache_identity_sha256:
                raise ValueError("cache trace identity differs from cache_identity_sha256")
        body = {
            "schema_version": AUTHENTICATED_RUNNER_EXECUTION_TRACE_VERSION,
            "hierarchy_execution_sha256": self.hierarchy_execution_sha256,
            "runner_identity": identity,
            "runner_identity_sha256": self.runner_identity_sha256,
            "validator_code_sha256": self.validator_code_sha256,
            "cache_identity": cache_identity,
            "cache_identity_sha256": self.cache_identity_sha256,
            "records": records,
        }
        if self.trace_sha256 != content_sha256(body):
            raise ValueError("trace_sha256 does not authenticate remote/cache execution records")

    @classmethod
    def create(
        cls,
        *,
        completed: CompletedHierarchicalDiscovery,
        runner_identity: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
        validator_code_sha256: str,
        cache_identity: Mapping[str, Any] | None,
    ) -> "AuthenticatedRunnerExecutionTrace":
        identity = _clone(runner_identity)
        normalized_cache = None if cache_identity is None else _clone(cache_identity)
        validated = _validate_execution_records(
            completed=completed,
            runner_identity=identity,
            records=records,
            validator_code_sha256=validator_code_sha256,
            cache_identity=normalized_cache,
        )
        identity_sha256 = identity["identity_sha256"]
        cache_identity_sha256 = (
            None if normalized_cache is None else normalized_cache["identity_sha256"]
        )
        body = {
            "schema_version": AUTHENTICATED_RUNNER_EXECUTION_TRACE_VERSION,
            "hierarchy_execution_sha256": completed.execution_ledger.execution_sha256,
            "runner_identity": identity,
            "runner_identity_sha256": identity_sha256,
            "validator_code_sha256": validator_code_sha256,
            "cache_identity": normalized_cache,
            "cache_identity_sha256": cache_identity_sha256,
            "records": list(validated),
        }
        return cls(
            hierarchy_execution_sha256=completed.execution_ledger.execution_sha256,
            runner_identity_sha256=identity_sha256,
            validator_code_sha256=validator_code_sha256,
            cache_identity_sha256=cache_identity_sha256,
            trace_sha256=content_sha256(body),
            _runner_identity_json=canonical_json(identity),
            _cache_identity_json=canonical_json(normalized_cache),
            _records_json=canonical_json(validated),
        )

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(self._records_json))

    @property
    def runner_identity(self) -> dict[str, Any]:
        return json.loads(self._runner_identity_json)

    @property
    def cache_identity(self) -> dict[str, Any] | None:
        value = json.loads(self._cache_identity_json)
        return None if value is None else dict(value)

    def validate_against(self, completed: CompletedHierarchicalDiscovery) -> None:
        self.__post_init__()
        if self.hierarchy_execution_sha256 != completed.execution_ledger.execution_sha256:
            raise ValueError("runner trace cites a different hierarchy execution")
        _validate_execution_records(
            completed=completed,
            runner_identity=self.runner_identity,
            records=self.records,
            validator_code_sha256=self.validator_code_sha256,
            cache_identity=self.cache_identity,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUTHENTICATED_RUNNER_EXECUTION_TRACE_VERSION,
            "hierarchy_execution_sha256": self.hierarchy_execution_sha256,
            "runner_identity_sha256": self.runner_identity_sha256,
            "validator_code_sha256": self.validator_code_sha256,
            "cache_identity_sha256": self.cache_identity_sha256,
            "records": list(self.records),
            "trace_sha256": self.trace_sha256,
        }


def _merge_current_execution_records(
    *,
    completed: CompletedHierarchicalDiscovery,
    remote_records: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    jobs = completed.execution_ledger.job_ledger.jobs
    results = completed.execution_ledger.results
    expected = tuple(job.job_id for job in jobs)
    attempt_owner: dict[str, str] = {}
    attempts_by_logical: dict[str, tuple[dict[str, Any], ...]] = {}
    for job, result in zip(jobs, results):
        attempts = tuple(_clone(row) for row in result.response_attempt_trace["attempts"])
        attempts_by_logical[job.job_id] = attempts
        for attempt in attempts:
            attempt_job_id = attempt["job_id"]
            if attempt_job_id in attempt_owner:
                raise ValueError("one response-attempt job belongs to multiple logical jobs")
            attempt_owner[attempt_job_id] = job.job_id

    remote_by_logical: dict[str, list[dict[str, Any]]] = {job_id: [] for job_id in expected}
    cache_by_logical: dict[str, dict[str, Any]] = {}
    for record in remote_records:
        detached = _clone(record)
        if not isinstance(detached, Mapping):
            raise TypeError("execution record must be one JSON object")
        job_id = detached.get("job_id")
        logical_job_id = attempt_owner.get(job_id)
        if logical_job_id is None:
            raise ValueError("remote execution record cites an unauthenticated response attempt")
        remote_by_logical[logical_job_id].append(dict(detached))
    for record in cache_records:
        detached = _clone(record)
        if not isinstance(detached, Mapping):
            raise TypeError("cache execution record must be one JSON object")
        job_id = detached.get("job_id")
        if not isinstance(job_id, str) or job_id not in remote_by_logical:
            raise ValueError("cache execution record cites a job outside the completed hierarchy")
        if job_id in cache_by_logical:
            raise ValueError("one hierarchy job has duplicate cache execution records")
        cache_by_logical[job_id] = dict(detached)

    merged: list[dict[str, Any]] = []
    job_by_id = {job.job_id: job for job in jobs}
    result_by_id = {result.job_id: result for result in results}
    for logical_job_id in expected:
        remotes = remote_by_logical[logical_job_id]
        cached = cache_by_logical.get(logical_job_id)
        if cached is not None:
            if remotes:
                raise ValueError("one hierarchy job cannot be both remote and a cache hit")
            merged.append(cached)
            continue
        attempts = attempts_by_logical[logical_job_id]
        expected_attempt_job_ids = [row["job_id"] for row in attempts]
        remote_by_attempt_id = {row.get("job_id"): row for row in remotes}
        if len(remote_by_attempt_id) != len(remotes) or set(remote_by_attempt_id) != set(
            expected_attempt_job_ids
        ):
            raise ValueError("remote records do not cover the exact response-attempt trace")
        ordered = [remote_by_attempt_id[job_id] for job_id in expected_attempt_job_ids]
        if len(ordered) == 1:
            merged.append(ordered[0])
            continue
        if len(ordered) != 2:
            raise ValueError("response repair is bounded to one additional remote record")
        job = job_by_id[logical_job_id]
        result = result_by_id[logical_job_id]
        body = {
            "schema_version": _REMOTE_RESPONSE_REPAIR_SEQUENCE_VERSION,
            "record_type": "authenticated_remote_response_repair_sequence",
            "job_id": logical_job_id,
            "job_kind": job.job_kind,
            "response_attempt_trace": result.response_attempt_trace,
            "response_attempt_trace_sha256": result.response_attempt_trace_sha256,
            "validated_response_sha256": result.response_sha256,
            "remote_records": ordered,
            "outcome": "success_after_one_response_repair",
        }
        merged.append({**body, "record_sha256": content_sha256(body)})
    return tuple(merged)


def _dossier_numerical_audit(
    *,
    completed: CompletedHierarchicalDiscovery,
    direct_numerical_contract_kind: str,
    direct_numerical_contract_sha256: str,
    expected_bindings: Sequence[DirectNumericalDossierBinding],
    manifest: DirectUpstreamNumericalManifest | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    binding_by_family = {row.source_family: row for row in expected_bindings}
    for dossier in completed.dossiers:
        binding = binding_by_family[dossier.source_family]
        if manifest is not None:
            validate_architecture_dossier_numerical_binding(dossier, manifest)
        if dossier.direct_numerical_contract_kind != direct_numerical_contract_kind:
            raise ValueError("dossier direct numerical contract kind changed")
        if dossier.direct_numerical_contract_sha256 != direct_numerical_contract_sha256:
            raise ValueError("dossier direct numerical contract SHA-256 changed")
        if dossier.direct_numerical_signal_count != binding.signal_count:
            raise ValueError("dossier direct numerical signal count changed")
        if dossier.direct_numerical_zero_reason != binding.zero_reason:
            raise ValueError("dossier direct numerical zero reason changed")
        rows.append(
            {
                "source_family": dossier.source_family,
                "catalog_sha256": dossier.catalog_sha256,
                "catalog_evidence_ids": list(dossier.catalog_evidence_ids),
                "catalog_evidence_count": len(dossier.catalog_evidence_ids),
                "semantic_atom_ids_sha256": content_sha256(list(dossier.catalog_evidence_ids)),
                "direct_numerical_contract_kind": (dossier.direct_numerical_contract_kind),
                "direct_numerical_contract_sha256": (dossier.direct_numerical_contract_sha256),
                "direct_numerical_signal_count": dossier.direct_numerical_signal_count,
                "direct_numerical_zero_reason": dossier.direct_numerical_zero_reason,
                "validated_against_approved_contract": True,
                "validated_against_full_manifest": manifest is not None,
            }
        )
    return rows


@dataclass(frozen=True)
class ApprovedHierarchicalDiscoveryResult:
    wrapper_approval_sha256: str
    inner_precommit_sha256: str
    direct_numerical_contract_kind: str
    direct_numerical_contract_sha256: str
    completed: CompletedHierarchicalDiscovery
    compiled_registry: CompiledHierarchicalFeatureRegistry
    runner_trace: AuthenticatedRunnerExecutionTrace
    numerical_binding_audit_sha256: str
    result_sha256: str
    _numerical_binding_audit_json: str = field(repr=False)

    def __post_init__(self) -> None:
        self.validate_authentication()

    @property
    def numerical_binding_audit(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(self._numerical_binding_audit_json))

    @property
    def direct_numerical_manifest_sha256(self) -> str | None:
        """Compatibility view available only when the approved contract is realized."""

        if self.direct_numerical_contract_kind != DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST:
            return None
        return self.direct_numerical_contract_sha256

    def _identity_without_sha(self) -> dict[str, Any]:
        return {
            "schema_version": APPROVED_HIERARCHICAL_DISCOVERY_RESULT_VERSION,
            "wrapper_approval_sha256": self.wrapper_approval_sha256,
            "inner_precommit_sha256": self.inner_precommit_sha256,
            "direct_numerical_contract_kind": self.direct_numerical_contract_kind,
            "direct_numerical_contract_sha256": self.direct_numerical_contract_sha256,
            "completion_sha256": self.completed.completion_sha256,
            "compiled_registry_sha256": self.compiled_registry.registry_sha256,
            "runner_trace_sha256": self.runner_trace.trace_sha256,
            "numerical_binding_audit_sha256": self.numerical_binding_audit_sha256,
        }

    def validate_authentication(self) -> None:
        for label, value in (
            ("wrapper_approval_sha256", self.wrapper_approval_sha256),
            ("inner_precommit_sha256", self.inner_precommit_sha256),
            ("direct_numerical_contract_sha256", self.direct_numerical_contract_sha256),
            (
                "numerical_binding_audit_sha256",
                self.numerical_binding_audit_sha256,
            ),
            ("result_sha256", self.result_sha256),
        ):
            _require_sha256(value, label=label)
        if self.direct_numerical_contract_kind not in {
            DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST,
            DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT,
        }:
            raise ValueError("result direct_numerical_contract_kind is unsupported")
        self.completed.__post_init__()
        self.compiled_registry.validate_authentication()
        self.runner_trace.validate_against(self.completed)
        if self.completed.precommit_sha256 != self.inner_precommit_sha256:
            raise ValueError("completed hierarchy cites a different inner precommit")
        if self.compiled_registry.source_completion_sha256 != self.completed.completion_sha256:
            raise ValueError("compiled registry cites a different hierarchy completion")
        audit = list(self.numerical_binding_audit)
        if content_sha256(audit) != self.numerical_binding_audit_sha256:
            raise ValueError("numerical binding audit SHA-256 changed")
        if [row.get("source_family") for row in audit] != list(ACTIVE_STAGE1_CONCEPT_FAMILIES):
            raise ValueError("numerical binding audit does not cover all architectures")
        if any(
            row.get("direct_numerical_contract_kind") != self.direct_numerical_contract_kind
            or row.get("direct_numerical_contract_sha256") != self.direct_numerical_contract_sha256
            for row in audit
        ):
            raise ValueError("numerical binding audit cites a different contract")
        if self.result_sha256 != content_sha256(self._identity_without_sha()):
            raise ValueError("result_sha256 does not authenticate approved discovery result")

    @classmethod
    def create(
        cls,
        *,
        wrapper_approval_sha256: str,
        inner_precommit_sha256: str,
        direct_numerical_contract_kind: str,
        direct_numerical_contract_sha256: str,
        expected_bindings: Sequence[DirectNumericalDossierBinding],
        manifest: DirectUpstreamNumericalManifest | None,
        completed: CompletedHierarchicalDiscovery,
        compiled_registry: CompiledHierarchicalFeatureRegistry,
        runner_trace: AuthenticatedRunnerExecutionTrace,
    ) -> "ApprovedHierarchicalDiscoveryResult":
        audit = _dossier_numerical_audit(
            completed=completed,
            direct_numerical_contract_kind=direct_numerical_contract_kind,
            direct_numerical_contract_sha256=direct_numerical_contract_sha256,
            expected_bindings=expected_bindings,
            manifest=manifest,
        )
        audit_sha256 = content_sha256(audit)
        identity = {
            "schema_version": APPROVED_HIERARCHICAL_DISCOVERY_RESULT_VERSION,
            "wrapper_approval_sha256": wrapper_approval_sha256,
            "inner_precommit_sha256": inner_precommit_sha256,
            "direct_numerical_contract_kind": direct_numerical_contract_kind,
            "direct_numerical_contract_sha256": direct_numerical_contract_sha256,
            "completion_sha256": completed.completion_sha256,
            "compiled_registry_sha256": compiled_registry.registry_sha256,
            "runner_trace_sha256": runner_trace.trace_sha256,
            "numerical_binding_audit_sha256": audit_sha256,
        }
        return cls(
            wrapper_approval_sha256=wrapper_approval_sha256,
            inner_precommit_sha256=inner_precommit_sha256,
            direct_numerical_contract_kind=direct_numerical_contract_kind,
            direct_numerical_contract_sha256=direct_numerical_contract_sha256,
            completed=completed,
            compiled_registry=compiled_registry,
            runner_trace=runner_trace,
            numerical_binding_audit_sha256=audit_sha256,
            result_sha256=content_sha256(identity),
            _numerical_binding_audit_json=canonical_json(audit),
        )


class ApprovedHierarchicalDiscoveryAgent:
    """Prepare an inspectable packet and execute only its exact approval."""

    def __init__(
        self,
        *,
        catalog: RoleNeutralEvidenceCatalog,
        chunk_plan: ArchitectureChunkPlan,
        family_explanations: Mapping[str, str],
        runner: MetadataJsonDiscoveryJobRunner,
        direct_numerical_manifest: DirectUpstreamNumericalManifest | None = None,
        first_gate_materialization_intent: FirstGateMaterializationIntent | None = None,
        reference_only_direct_numerical_contract: (
            AuthenticatedReferenceOnlyDirectNumericalContract | None
        ) = None,
        direct_numerical_bindings: Sequence[DirectNumericalDossierBinding] | None = None,
        config: HierarchicalDiscoveryConfig | None = None,
        compiler: HierarchicalDiscoveryCompiler | None = None,
        job_cache: AuthenticatedHierarchicalDiscoveryJobCache | None = None,
    ) -> None:
        if not isinstance(runner, MetadataJsonDiscoveryJobRunner):
            raise TypeError("runner must expose identity, run_json, and execution_metadata")
        self.catalog = catalog
        self.chunk_plan = chunk_plan
        self.family_explanations = dict(family_explanations)
        self.direct_numerical_manifest = direct_numerical_manifest
        self.first_gate_materialization_intent = first_gate_materialization_intent
        self.reference_only_direct_numerical_contract = (
            reference_only_direct_numerical_contract
        )
        supplied_contracts = sum(
            value is not None
            for value in (
                direct_numerical_manifest,
                first_gate_materialization_intent,
                reference_only_direct_numerical_contract,
            )
        )
        if supplied_contracts != 1:
            raise ValueError(
                "provide exactly one direct numerical manifest, first-gate "
                "materialization intent, or reference-only direct contract"
            )
        if direct_numerical_manifest is not None:
            expected_bindings = direct_numerical_bindings_from_manifest(direct_numerical_manifest)
            self.direct_numerical_contract_kind = DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
            self.direct_numerical_contract_sha256 = direct_numerical_manifest.content_sha256
        elif first_gate_materialization_intent is not None:
            assert first_gate_materialization_intent is not None
            expected_bindings = direct_numerical_bindings_from_intent(
                first_gate_materialization_intent,
                catalog=catalog,
            )
            self.direct_numerical_contract_kind = DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
            self.direct_numerical_contract_sha256 = first_gate_materialization_intent.content_sha256
        else:
            assert reference_only_direct_numerical_contract is not None
            expected_bindings = direct_numerical_bindings_from_reference_contract(
                reference_only_direct_numerical_contract,
                catalog=catalog,
            )
            self.direct_numerical_contract_kind = (
                DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
            )
            self.direct_numerical_contract_sha256 = (
                reference_only_direct_numerical_contract.content_sha256
            )
        self.direct_numerical_bindings = _validate_explicit_bindings(
            supplied=(
                expected_bindings
                if direct_numerical_bindings is None
                else direct_numerical_bindings
            ),
            expected=expected_bindings,
        )
        self.runner = runner
        self.config = config or HierarchicalDiscoveryConfig()
        if job_cache is not None and not isinstance(
            job_cache, AuthenticatedHierarchicalDiscoveryJobCache
        ):
            raise TypeError("job_cache must be AuthenticatedHierarchicalDiscoveryJobCache")
        self.job_cache = job_cache
        self.compiler = compiler or HierarchicalDiscoveryCompiler(
            max_candidates=self.config.max_integrated_features
        )
        if not isinstance(self.config, HierarchicalDiscoveryConfig):
            raise TypeError("config must be HierarchicalDiscoveryConfig")
        if not isinstance(self.compiler, HierarchicalDiscoveryCompiler):
            raise TypeError("compiler must be HierarchicalDiscoveryCompiler")

        self._validate_bound_inputs()
        runner_identity = _validated_runner_identity(self.runner)
        orchestrator = self._new_orchestrator(runner_identity=runner_identity)
        self._runner_identity_json = canonical_json(runner_identity)
        self._inner_precommit_json = canonical_json(orchestrator.precommit.packet)
        self._inner_precommit_sha256 = orchestrator.precommit.precommit_sha256
        self._hierarchy_validator_code_sha256 = orchestrator.implementation_bundle_sha256
        self._catalog_json = canonical_json(self.catalog.as_dict())
        self._chunk_plan_json = canonical_json(self.chunk_plan.as_dict())
        self._direct_numerical_contract_json = canonical_json(
            self._direct_numerical_contract_as_dict()
        )
        self._bindings_json = canonical_json(
            [row.as_dossier_dict() for row in self.direct_numerical_bindings]
        )
        self._compiler_binding_json = canonical_json(
            _component_binding(self.compiler, self.compiler.identity)
        )
        self._cache_binding_json = canonical_json(
            _cache_binding(
                self.job_cache,
                validator_code_sha256=self._hierarchy_validator_code_sha256,
            )
        )
        self.precommit = ApprovedHierarchicalDiscoveryPrecommit.create(
            self._offline_packet(orchestrator=orchestrator)
        )

    @classmethod
    def prepare_from_evidence_inputs(
        cls,
        *,
        evidence_inputs: Sequence[FoldEvidenceInput],
        family_explanations: Mapping[str, str],
        runner: MetadataJsonDiscoveryJobRunner,
        direct_numerical_manifest: DirectUpstreamNumericalManifest | None = None,
        first_gate_materialization_intent: FirstGateMaterializationIntent | None = None,
        reference_only_direct_numerical_contract: (
            AuthenticatedReferenceOnlyDirectNumericalContract | None
        ) = None,
        config: HierarchicalDiscoveryConfig | None = None,
        compiler: HierarchicalDiscoveryCompiler | None = None,
        job_cache: AuthenticatedHierarchicalDiscoveryJobCache | None = None,
        max_atoms_per_chunk: int = DEFAULT_MAX_ATOMS_PER_ARCHITECTURE_CHUNK,
        max_bytes_per_chunk: int = DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK,
        max_semantic_member_ids_per_chunk: int | None = None,
    ) -> "ApprovedHierarchicalDiscoveryAgent":
        if config is not None and not isinstance(config, HierarchicalDiscoveryConfig):
            raise TypeError("config must be HierarchicalDiscoveryConfig")
        chosen_member_bound = (
            (
                DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK
                if config is None
                else config.max_semantic_member_ids_per_chunk
            )
            if max_semantic_member_ids_per_chunk is None
            else max_semantic_member_ids_per_chunk
        )
        chosen_config = config or HierarchicalDiscoveryConfig(
            max_semantic_member_ids_per_chunk=chosen_member_bound
        )
        if not isinstance(chosen_config, HierarchicalDiscoveryConfig):
            raise TypeError("config must be HierarchicalDiscoveryConfig")
        if chosen_config.max_semantic_member_ids_per_chunk != chosen_member_bound:
            raise ValueError(
                "config and chunk builder must use the same " "max_semantic_member_ids_per_chunk"
            )
        catalog = build_role_neutral_evidence_catalog(evidence_inputs)
        plan = build_complete_architecture_chunks(
            catalog,
            max_atoms_per_chunk=max_atoms_per_chunk,
            max_bytes_per_chunk=max_bytes_per_chunk,
            max_semantic_member_ids_per_chunk=chosen_member_bound,
        )
        return cls(
            catalog=catalog,
            chunk_plan=plan,
            family_explanations=family_explanations,
            direct_numerical_manifest=direct_numerical_manifest,
            first_gate_materialization_intent=first_gate_materialization_intent,
            reference_only_direct_numerical_contract=(
                reference_only_direct_numerical_contract
            ),
            runner=runner,
            config=chosen_config,
            compiler=compiler,
            job_cache=job_cache,
        )

    @property
    def inner_precommit_sha256(self) -> str:
        return self._inner_precommit_sha256

    def _direct_numerical_contract_as_dict(self) -> dict[str, Any]:
        if self.direct_numerical_manifest is not None:
            return {
                "direct_numerical_contract_kind": (
                    DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
                ),
                "direct_numerical_contract_sha256": (self.direct_numerical_manifest.content_sha256),
                "contract": self.direct_numerical_manifest.as_dict(),
            }
        if self.reference_only_direct_numerical_contract is not None:
            return {
                "direct_numerical_contract_kind": (
                    DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
                ),
                "direct_numerical_contract_sha256": (
                    self.reference_only_direct_numerical_contract.content_sha256
                ),
                "contract": (
                    self.reference_only_direct_numerical_contract.as_dict()
                ),
            }
        if self.first_gate_materialization_intent is None:
            raise RuntimeError("direct numerical contract is missing")
        return {
            "direct_numerical_contract_kind": (DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT),
            "direct_numerical_contract_sha256": (
                self.first_gate_materialization_intent.content_sha256
            ),
            "contract": self.first_gate_materialization_intent.as_dict(),
        }

    def _validate_bound_inputs(self) -> None:
        validate_role_neutral_catalog(self.catalog)
        audit = audit_complete_architecture_delivery(self.catalog, self.chunk_plan)
        if audit.get("all_catalog_atoms_delivered_exactly_once") is not True:
            raise ValueError("chunk plan does not losslessly deliver the catalog")
        if set(self.family_explanations) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("family explanations must cover exactly all architectures")
        if self.direct_numerical_manifest is not None:
            if not isinstance(self.direct_numerical_manifest, DirectUpstreamNumericalManifest):
                raise TypeError("direct_numerical_manifest must be DirectUpstreamNumericalManifest")
            _reauthenticate_full_manifest(self.direct_numerical_manifest)
            _validate_catalog_manifest_binding(
                catalog=self.catalog,
                manifest=self.direct_numerical_manifest,
            )
            expected = direct_numerical_bindings_from_manifest(self.direct_numerical_manifest)
        elif self.reference_only_direct_numerical_contract is not None:
            expected = direct_numerical_bindings_from_reference_contract(
                self.reference_only_direct_numerical_contract,
                catalog=self.catalog,
            )
        else:
            if self.first_gate_materialization_intent is None:
                raise ValueError("first-gate materialization intent is missing")
            expected = direct_numerical_bindings_from_intent(
                self.first_gate_materialization_intent,
                catalog=self.catalog,
            )
        _validate_explicit_bindings(
            supplied=self.direct_numerical_bindings,
            expected=expected,
        )
        if {row.direct_numerical_contract_kind for row in self.direct_numerical_bindings} != {
            self.direct_numerical_contract_kind
        }:
            raise ValueError("dossier bindings changed direct numerical contract kind")
        if {row.direct_numerical_contract_sha256 for row in self.direct_numerical_bindings} != {
            self.direct_numerical_contract_sha256
        }:
            raise ValueError("dossier bindings changed direct numerical contract SHA-256")

    def _new_orchestrator(
        self, *, runner_identity: Mapping[str, Any]
    ) -> HierarchicalAllArchitectureDiscoveryOrchestrator:
        return HierarchicalAllArchitectureDiscoveryOrchestrator(
            catalog=self.catalog,
            chunk_plan=self.chunk_plan,
            family_explanations=self.family_explanations,
            direct_numerical_bindings=self.direct_numerical_bindings,
            runner_identity=runner_identity,
            config=self.config,
            job_cache=self.job_cache,
        )

    def _offline_packet(
        self, *, orchestrator: HierarchicalAllArchitectureDiscoveryOrchestrator
    ) -> dict[str, Any]:
        compiler_binding = _component_binding(self.compiler, self.compiler.identity)
        return {
            "schema_version": APPROVED_HIERARCHICAL_DISCOVERY_PRECOMMIT_VERSION,
            "agent_version": APPROVED_HIERARCHICAL_DISCOVERY_AGENT_VERSION,
            "agent_implementation_file_sha256": _sha256_bytes(Path(__file__).read_bytes()),
            "catalog_binding": {
                "catalog_sha256": self.catalog.catalog_sha256,
                "outer_fold": self.catalog.outer_fold,
                "scope": self.catalog.scope,
                "inner_fold": self.catalog.inner_fold,
                "split_fingerprint": self.catalog.split_fingerprint,
                "atom_count": len(self.catalog.atoms),
                "family_atom_counts": {
                    family: len(self.catalog.family_atoms(family))
                    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
                },
            },
            "chunk_plan_binding": {
                "plan_sha256": self.chunk_plan.plan_sha256,
                "chunk_count": len(self.chunk_plan.chunks),
                "max_atoms_per_chunk": self.chunk_plan.max_atoms_per_chunk,
                "max_bytes_per_chunk": self.chunk_plan.max_bytes_per_chunk,
                "max_semantic_member_ids_per_chunk": (
                    self.chunk_plan.max_semantic_member_ids_per_chunk
                ),
            },
            "direct_numerical_contract_binding": (
                _safe_manifest_binding(self.direct_numerical_manifest)
                if self.direct_numerical_manifest is not None
                else (
                    _safe_reference_contract_binding(
                        self.reference_only_direct_numerical_contract,
                        catalog=self.catalog,
                    )
                    if self.reference_only_direct_numerical_contract
                    is not None
                    else _safe_intent_binding(
                        self.first_gate_materialization_intent,
                        catalog=self.catalog,
                    )
                )
            ),
            "direct_numerical_dossier_bindings": [
                row.as_dossier_dict() for row in self.direct_numerical_bindings
            ],
            "hierarchy_precommit": {
                "precommit_sha256": orchestrator.precommit.precommit_sha256,
                "packet": orchestrator.precommit.packet,
            },
            "runner_identity": _clone(self.runner.identity()),
            "job_cache_binding": _cache_binding(
                self.job_cache,
                validator_code_sha256=orchestrator.implementation_bundle_sha256,
            ),
            "compiler_binding": compiler_binding,
            "config_bounds": self.config.as_dict(),
            "assurances": {
                "all_active_architectures_bound": True,
                "all_catalog_atoms_delivered_exactly_once": True,
                "direct_numerical_contract_authenticated_locally_in_full": True,
                "direct_numerical_contract_kind": self.direct_numerical_contract_kind,
                "direct_numerical_contract_materialized": (
                    self.direct_numerical_manifest is not None
                    or self.reference_only_direct_numerical_contract is not None
                ),
                "direct_row_level_numerical_values_in_packet": False,
                "direct_coordinate_metadata_in_packet": False,
                "unapproved_remote_execution_allowed": False,
                "final_dossiers_revalidated_against_approved_contract": True,
                "runner_retry_records_authenticated_in_final_result": True,
                "cache_hits_authenticated_in_final_result": True,
                "cache_lookup_before_wrapper_approval_allowed": False,
                "cache_write_before_semantic_validation_allowed": False,
                "runner_wire_hash_authenticated_before_cache_write": True,
                "cache_validator_identity_is_hierarchy_implementation_bundle": True,
            },
        }

    def _assert_unchanged(
        self,
    ) -> tuple[dict[str, Any], HierarchicalAllArchitectureDiscoveryOrchestrator]:
        self.precommit.__post_init__()
        self._validate_bound_inputs()
        if canonical_json(self.catalog.as_dict()) != self._catalog_json:
            raise ValueError("semantic catalog mutated after offline approval preparation")
        if canonical_json(self.chunk_plan.as_dict()) != self._chunk_plan_json:
            raise ValueError("architecture chunk plan mutated after approval preparation")
        if (
            canonical_json(self._direct_numerical_contract_as_dict())
            != self._direct_numerical_contract_json
        ):
            raise ValueError("direct numerical contract mutated after approval preparation")
        if (
            canonical_json([row.as_dossier_dict() for row in self.direct_numerical_bindings])
            != self._bindings_json
        ):
            raise ValueError("direct numerical dossier bindings mutated after preparation")
        runner_identity = _validated_runner_identity(self.runner)
        if canonical_json(runner_identity) != self._runner_identity_json:
            raise ValueError("runner identity mutated after offline approval preparation")
        if (
            canonical_json(_component_binding(self.compiler, self.compiler.identity))
            != self._compiler_binding_json
        ):
            raise ValueError("compiler identity mutated after offline approval preparation")
        orchestrator = self._new_orchestrator(runner_identity=runner_identity)
        if orchestrator.implementation_bundle_sha256 != self._hierarchy_validator_code_sha256:
            raise ValueError("hierarchy validator bundle changed after preparation")
        if (
            canonical_json(
                _cache_binding(
                    self.job_cache,
                    validator_code_sha256=orchestrator.implementation_bundle_sha256,
                )
            )
            != self._cache_binding_json
        ):
            raise ValueError("job cache or validator identity mutated after preparation")
        if (
            orchestrator.precommit.precommit_sha256 != self._inner_precommit_sha256
            or canonical_json(orchestrator.precommit.packet) != self._inner_precommit_json
        ):
            raise ValueError("hierarchy precommit mutated after offline approval preparation")
        regenerated = self._offline_packet(orchestrator=orchestrator)
        if content_sha256(regenerated) != self.precommit.approval_sha256 or canonical_json(
            regenerated
        ) != canonical_json(self.precommit.packet):
            raise ValueError("wrapper offline packet mutated after approval preparation")
        return runner_identity, orchestrator

    def validate_precommit_unchanged(self) -> None:
        """Reauthenticate static approval bindings without cache lookup or transport."""

        self._assert_unchanged()

    def render_offline_precommit(self, *, indent: int = 2) -> str:
        return self.precommit.render_json(indent=indent)

    def execute(self, *, approved_wrapper_sha256: str) -> ApprovedHierarchicalDiscoveryResult:
        """Run the hierarchy only after exact wrapper-packet approval."""

        if approved_wrapper_sha256 != self.precommit.approval_sha256:
            raise ValueError("approved wrapper SHA-256 does not match the offline packet")
        runner_identity, orchestrator = self._assert_unchanged()
        before = tuple(_clone(row) for row in self.runner.execution_metadata)
        authenticated_runner = _PerCallMetadataAuthenticatingRunner(
            runner=self.runner,
            runner_identity=runner_identity,
        )
        completed = orchestrator.execute(
            runner=authenticated_runner,
            approved_precommit_sha256=self._inner_precommit_sha256,
        )
        after = tuple(_clone(row) for row in self.runner.execution_metadata)
        if after[: len(before)] != before:
            raise ValueError("runner execution metadata mutated during hierarchy execution")
        new_remote_records = after[len(before) :]
        cache_records = orchestrator.cache_execution_metadata
        if canonical_json(_validated_runner_identity(self.runner)) != self._runner_identity_json:
            raise ValueError("runner identity changed during hierarchy execution")
        cache_identity = (
            None if self.job_cache is None else _validated_cache_identity(self.job_cache)
        )
        if (
            canonical_json(
                _cache_binding(
                    self.job_cache,
                    validator_code_sha256=orchestrator.implementation_bundle_sha256,
                )
            )
            != self._cache_binding_json
        ):
            raise ValueError("job cache or validator identity changed during execution")
        binding_by_family = {row.source_family: row for row in self.direct_numerical_bindings}
        for dossier in completed.dossiers:
            expected_ids = tuple(
                atom.evidence_id for atom in self.catalog.family_atoms(dossier.source_family)
            )
            if dossier.catalog_sha256 != self.catalog.catalog_sha256:
                raise ValueError("completed dossier changed the semantic catalog SHA-256")
            if dossier.catalog_evidence_ids != expected_ids:
                raise ValueError("completed dossier changed exact catalog atom bindings")
            binding = binding_by_family[dossier.source_family]
            if (
                dossier.direct_numerical_contract_kind != binding.direct_numerical_contract_kind
                or dossier.direct_numerical_contract_sha256
                != binding.direct_numerical_contract_sha256
                or dossier.direct_numerical_signal_count != binding.signal_count
                or dossier.direct_numerical_zero_reason != binding.zero_reason
            ):
                raise ValueError("completed dossier changed its direct numerical binding")
            if self.direct_numerical_manifest is not None:
                validate_architecture_dossier_numerical_binding(
                    dossier, self.direct_numerical_manifest
                )
        compiled = self.compiler.compile(completed)
        if (
            canonical_json(_component_binding(self.compiler, self.compiler.identity))
            != self._compiler_binding_json
        ):
            raise ValueError("compiler identity changed during hierarchy compilation")
        trace = AuthenticatedRunnerExecutionTrace.create(
            completed=completed,
            runner_identity=runner_identity,
            records=_merge_current_execution_records(
                completed=completed,
                remote_records=new_remote_records,
                cache_records=cache_records,
            ),
            validator_code_sha256=orchestrator.implementation_bundle_sha256,
            cache_identity=cache_identity,
        )
        return ApprovedHierarchicalDiscoveryResult.create(
            wrapper_approval_sha256=self.precommit.approval_sha256,
            inner_precommit_sha256=self._inner_precommit_sha256,
            direct_numerical_contract_kind=self.direct_numerical_contract_kind,
            direct_numerical_contract_sha256=self.direct_numerical_contract_sha256,
            expected_bindings=self.direct_numerical_bindings,
            manifest=self.direct_numerical_manifest,
            completed=completed,
            compiled_registry=compiled,
            runner_trace=trace,
        )


__all__ = [
    "APPROVED_HIERARCHICAL_DISCOVERY_AGENT_VERSION",
    "APPROVED_HIERARCHICAL_DISCOVERY_PRECOMMIT_VERSION",
    "APPROVED_HIERARCHICAL_DISCOVERY_RESULT_VERSION",
    "AUTHENTICATED_RUNNER_EXECUTION_TRACE_VERSION",
    "AuthenticatedReferenceOnlyDirectNumericalContract",
    "ApprovedHierarchicalDiscoveryAgent",
    "ApprovedHierarchicalDiscoveryPrecommit",
    "ApprovedHierarchicalDiscoveryResult",
    "AuthenticatedRunnerExecutionTrace",
    "MetadataJsonDiscoveryJobRunner",
    "direct_numerical_bindings_from_intent",
    "direct_numerical_bindings_from_manifest",
    "direct_numerical_bindings_from_reference_contract",
]
