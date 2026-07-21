"""Run-level approval boundary for all outer-fold discovery agents.

The fold-level approval wrapper prevents an individual hierarchy from running
without review.  This module adds the stronger run-level invariant needed by a
frozen benchmark: every outer fold, and every complete fold wrapper packet, is
committed in one inspectable packet before any fold may consult a job cache or
make a remote model call.

The coordinator intentionally does not implement persistence or transport.  A
fold agent's ``validate_precommit_unchanged`` method is the side-effect-free
static preflight; its ``execute`` method is reached only after the exact batch
digest is supplied and *all* folds pass that preflight.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .all_evidence_discovery_interfaces import (
    DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST,
    DIRECT_NUMERICAL_CONTRACT_KINDS,
    canonical_json,
    content_sha256,
)
from .approved_hierarchical_discovery_agent import (
    ApprovedHierarchicalDiscoveryAgent,
    ApprovedHierarchicalDiscoveryResult,
)
from .adaptive_hierarchical_stage1_reconsideration import (
    AdaptiveReconsiderationConfig,
    adaptive_hierarchical_stage1_reconsideration_identity,
)
from .frozen_hierarchical_review_evidence import (
    FrozenHierarchicalReviewEvidenceConfig,
    frozen_hierarchical_review_evidence_identity,
)

APPROVED_HIERARCHICAL_DISCOVERY_BATCH_COORDINATOR_VERSION = (
    "approved_hierarchical_discovery_batch_coordinator_v4"
)
APPROVED_HIERARCHICAL_DISCOVERY_BATCH_PRECOMMIT_VERSION = (
    "approved_hierarchical_discovery_batch_precommit_v4"
)
APPROVED_HIERARCHICAL_DISCOVERY_BATCH_RESULT_VERSION = (
    "approved_hierarchical_discovery_batch_result_v4"
)
FROZEN_REVIEW_EVIDENCE_POLICY_VERSION = "frozen_review_evidence_policy_v2"

_ARCHITECTURE_CHUNK_LIMIT_KEYS = (
    "max_atoms_per_chunk",
    "max_bytes_per_chunk",
    "max_semantic_member_ids_per_chunk",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_POLICY_KEYS = frozenset(
    {
        "max_evidence_ids",
        "max_evidence_bytes",
        "accepted_support_only",
        "review_materializer_identity",
        "adaptive_reconsideration_identity",
    }
)
_ADAPTIVE_RECONSIDERATION_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "authenticated_execution_version",
        "executable_bridge_version",
        "implementation_file_sha256",
        "implementation_bundle",
        "config",
        "config_sha256",
        "prompt_contract",
        "phase_policy",
    }
)
_REVIEW_MATERIALIZER_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "policy_version",
        "implementation_file_sha256",
        "accepted_routed_support_only",
        "original_content_addressed_ids_preserved",
        "rejected_only_excluded",
        "planner_lookback_only_excluded",
        "bounds_fail_closed_without_truncation",
    }
)


def _clone(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one JSON object")
    return value


def _contract_identity_from_binding(value: Any) -> tuple[str, str]:
    row = _mapping(value, label="direct numerical contract binding")
    kind = row.get("direct_numerical_contract_kind")
    sha256 = row.get("direct_numerical_contract_sha256")
    if kind is None and sha256 is None and "manifest_sha256" in row:
        kind = DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
        sha256 = row.get("manifest_sha256")
    if kind not in DIRECT_NUMERICAL_CONTRACT_KINDS:
        raise ValueError("direct numerical contract kind is unsupported")
    return kind, _require_sha256(sha256, label="direct_numerical_contract_sha256")


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validated_review_materializer_identity(value: Any) -> dict[str, Any]:
    row = _mapping(value, label="review materializer identity")
    if set(row) != _REVIEW_MATERIALIZER_IDENTITY_KEYS:
        raise ValueError("review materializer identity has an unexpected closed schema")
    for label in ("schema_version", "policy_version"):
        if not isinstance(row[label], str) or not row[label]:
            raise ValueError(f"review materializer identity {label} must be non-empty")
    _require_sha256(
        row["implementation_file_sha256"],
        label="review materializer implementation_file_sha256",
    )
    for label in _REVIEW_MATERIALIZER_IDENTITY_KEYS - {
        "schema_version",
        "policy_version",
        "implementation_file_sha256",
    }:
        if row[label] is not True:
            raise ValueError(f"review materializer identity {label} must be true")
    normalized = dict(_clone(row))
    current = frozen_hierarchical_review_evidence_identity()
    if canonical_json(normalized) != canonical_json(current):
        raise ValueError(
            "review materializer identity differs from the current closed implementation"
        )
    return normalized


def _validated_adaptive_reconsideration_identity(value: Any) -> dict[str, Any]:
    row = _mapping(value, label="adaptive reconsideration identity")
    if set(row) != _ADAPTIVE_RECONSIDERATION_IDENTITY_KEYS:
        raise ValueError("adaptive reconsideration identity has an unexpected closed schema")
    for label in (
        "schema_version",
        "authenticated_execution_version",
        "executable_bridge_version",
    ):
        if not isinstance(row[label], str) or not row[label]:
            raise ValueError(f"adaptive reconsideration identity {label} must be non-empty")
    _require_sha256(
        row["implementation_file_sha256"],
        label="adaptive reconsideration implementation_file_sha256",
    )
    _require_sha256(
        row["config_sha256"],
        label="adaptive reconsideration config_sha256",
    )
    config = _mapping(row["config"], label="adaptive reconsideration config")
    chosen = AdaptiveReconsiderationConfig(**dict(config))
    normalized = dict(_clone(row))
    current = adaptive_hierarchical_stage1_reconsideration_identity(chosen)
    if canonical_json(normalized) != canonical_json(current):
        raise ValueError(
            "adaptive reconsideration identity differs from the current closed implementation"
        )
    return normalized


@dataclass(frozen=True)
class FrozenReviewEvidencePolicyBinding:
    """Closed phased policy for initial support review and later rediscovery."""

    max_evidence_ids: int
    max_evidence_bytes: int
    review_materializer_identity: Mapping[str, Any] = field(repr=False)
    adaptive_reconsideration_identity: Mapping[str, Any] = field(
        default_factory=adaptive_hierarchical_stage1_reconsideration_identity,
        repr=False,
    )
    accepted_support_only: bool = True
    policy_sha256: str = field(init=False)
    _review_materializer_identity_json: str = field(init=False, repr=False)
    _adaptive_reconsideration_identity_json: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _positive_int(
            self.max_evidence_ids,
            label="max_evidence_ids",
        )
        _positive_int(self.max_evidence_bytes, label="max_evidence_bytes")
        if self.accepted_support_only is not True:
            raise ValueError("frozen review evidence must be accepted-support-only")
        materializer = _validated_review_materializer_identity(self.review_materializer_identity)
        adaptive = _validated_adaptive_reconsideration_identity(
            self.adaptive_reconsideration_identity
        )
        materializer_json = canonical_json(materializer)
        stored_materializer_json = getattr(self, "_review_materializer_identity_json", None)
        if stored_materializer_json is None:
            object.__setattr__(
                self,
                "review_materializer_identity",
                materializer,
            )
            object.__setattr__(
                self,
                "_review_materializer_identity_json",
                materializer_json,
            )
        elif stored_materializer_json != materializer_json:
            raise ValueError("review materializer identity mutated after binding")
        adaptive_json = canonical_json(adaptive)
        stored_adaptive_json = getattr(self, "_adaptive_reconsideration_identity_json", None)
        if stored_adaptive_json is None:
            object.__setattr__(self, "adaptive_reconsideration_identity", adaptive)
            object.__setattr__(
                self,
                "_adaptive_reconsideration_identity_json",
                adaptive_json,
            )
        elif stored_adaptive_json != adaptive_json:
            raise ValueError("adaptive reconsideration identity mutated after binding")
        expected = content_sha256(self._identity_without_sha())
        current = getattr(self, "policy_sha256", None)
        if current is None:
            object.__setattr__(self, "policy_sha256", expected)
        elif current != expected:
            raise ValueError("policy_sha256 does not authenticate the closed review policy")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenReviewEvidencePolicyBinding":
        row = _mapping(value, label="frozen review evidence policy")
        if set(row) != _POLICY_KEYS:
            raise ValueError("frozen review evidence policy has an unexpected closed schema")
        return cls(
            max_evidence_ids=row["max_evidence_ids"],
            max_evidence_bytes=row["max_evidence_bytes"],
            review_materializer_identity=row["review_materializer_identity"],
            adaptive_reconsideration_identity=row["adaptive_reconsideration_identity"],
            accepted_support_only=row["accepted_support_only"],
        )

    def _identity_without_sha(self) -> dict[str, Any]:
        return {
            "schema_version": FROZEN_REVIEW_EVIDENCE_POLICY_VERSION,
            "max_evidence_ids": self.max_evidence_ids,
            "max_evidence_bytes": self.max_evidence_bytes,
            "accepted_support_only": self.accepted_support_only,
            "review_materializer_identity": json.loads(self._review_materializer_identity_json),
            "adaptive_reconsideration_identity": json.loads(
                self._adaptive_reconsideration_identity_json
            ),
            "evidence_selection_rule": (
                "round_1_exact_supporting_evidence_ids_of_hierarchy_accepted_features_only"
            ),
            "architecture_wide_single_prompt_evidence_dump_allowed": False,
            "round_1_feature_rediscovery_allowed": False,
            "later_round_feature_rediscovery_allowed": True,
            "same_frozen_evidence_used_for_every_round": False,
        }

    def validate_authentication(self) -> None:
        self.__post_init__()

    def as_dict(self) -> dict[str, Any]:
        self.validate_authentication()
        return {**self._identity_without_sha(), "policy_sha256": self.policy_sha256}

    def materializer_config(self) -> FrozenHierarchicalReviewEvidenceConfig:
        """Construct the exact fail-closed bounds committed by this policy."""

        self.validate_authentication()
        return FrozenHierarchicalReviewEvidenceConfig(
            max_evidence_ids=self.max_evidence_ids,
            max_evidence_bytes=self.max_evidence_bytes,
        )

    def adaptive_config(self) -> AdaptiveReconsiderationConfig:
        """Construct the exact later-round hierarchy bounds committed by this policy."""

        self.validate_authentication()
        identity = json.loads(self._adaptive_reconsideration_identity_json)
        return AdaptiveReconsiderationConfig(**dict(identity["config"]))


@dataclass(frozen=True)
class OrderedFoldDiscoveryAgent:
    """Explicit outer-fold label paired with its prepared fold agent."""

    outer_fold: int
    agent: ApprovedHierarchicalDiscoveryAgent

    def __post_init__(self) -> None:
        _positive_int(self.outer_fold, label="outer_fold")
        for method in ("validate_precommit_unchanged", "execute"):
            if not callable(getattr(self.agent, method, None)):
                raise TypeError(f"fold agent must implement {method}()")
        catalog = getattr(self.agent, "catalog", None)
        if catalog is None or getattr(catalog, "outer_fold", None) != self.outer_fold:
            raise ValueError("fold label does not match agent.catalog.outer_fold")


@dataclass(frozen=True)
class ApprovedHierarchicalDiscoveryBatchPrecommit:
    approval_sha256: str
    _packet_json: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_sha256(self.approval_sha256, label="approval_sha256")
        try:
            packet = json.loads(self._packet_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("batch approval packet is invalid JSON") from exc
        if not isinstance(packet, Mapping):
            raise TypeError("batch approval packet must be one JSON object")
        if self.approval_sha256 != content_sha256(packet):
            raise ValueError("approval_sha256 does not authenticate the batch packet")

    @classmethod
    def create(cls, packet: Mapping[str, Any]) -> "ApprovedHierarchicalDiscoveryBatchPrecommit":
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


def _wrapper_packet(agent: Any) -> tuple[str, dict[str, Any]]:
    precommit = getattr(agent, "precommit", None)
    if precommit is None:
        raise TypeError("fold agent must expose precommit")
    if callable(getattr(precommit, "__post_init__", None)):
        precommit.__post_init__()
    approval_sha256 = _require_sha256(
        getattr(precommit, "approval_sha256", None),
        label="fold wrapper approval_sha256",
    )
    packet = _clone(getattr(precommit, "packet", None))
    if not isinstance(packet, Mapping):
        raise TypeError("fold wrapper packet must be one JSON object")
    if content_sha256(packet) != approval_sha256:
        raise ValueError("fold wrapper approval does not authenticate its complete packet")
    required = {
        "catalog_binding",
        "chunk_plan_binding",
        "hierarchy_precommit",
        "runner_identity",
        "compiler_binding",
        "config_bounds",
    }
    if not required <= set(packet):
        raise ValueError("fold wrapper packet is missing a required batch binding")
    if not {
        "direct_numerical_contract_binding",
        "direct_numerical_manifest_binding",
    } & set(packet):
        raise ValueError("fold wrapper packet is missing a direct numerical contract")
    if {
        "direct_numerical_contract_binding",
        "direct_numerical_manifest_binding",
    } <= set(packet):
        raise ValueError("fold wrapper packet has ambiguous direct numerical contracts")
    return approval_sha256, dict(packet)


def _hierarchy_common_identity(wrapper_packet: Mapping[str, Any]) -> dict[str, Any]:
    hierarchy = _mapping(
        wrapper_packet["hierarchy_precommit"],
        label="hierarchy_precommit",
    )
    inner = _mapping(hierarchy.get("packet"), label="hierarchy precommit packet")
    required = {
        "schema_version",
        "orchestrator_version",
        "orchestrator_implementation_file_sha256",
        "orchestrator_implementation_bundle",
        "orchestrator_implementation_bundle_sha256",
        "runner_identity",
        "config",
        "downstream_contract",
        "assurances",
    }
    if not required <= set(inner):
        raise ValueError("hierarchy precommit is missing its common identity")
    if canonical_json(inner["runner_identity"]) != canonical_json(
        wrapper_packet["runner_identity"]
    ):
        raise ValueError("wrapper and hierarchy runner identities differ")
    if canonical_json(inner["config"]) != canonical_json(wrapper_packet["config_bounds"]):
        raise ValueError("wrapper and hierarchy config identities differ")
    bundle = _mapping(
        inner["orchestrator_implementation_bundle"],
        label="hierarchy implementation bundle",
    )
    declared_bundle_sha256 = _require_sha256(
        inner["orchestrator_implementation_bundle_sha256"],
        label="hierarchy implementation bundle SHA-256",
    )
    if bundle.get("implementation_bundle_sha256") != declared_bundle_sha256:
        raise ValueError("hierarchy implementation bundle SHA fields differ")
    bundle_body = {
        key: value for key, value in bundle.items() if key != "implementation_bundle_sha256"
    }
    if content_sha256(bundle_body) != declared_bundle_sha256:
        raise ValueError("hierarchy implementation bundle SHA-256 does not authenticate")
    bundle_files = _mapping(bundle.get("files"), label="hierarchy implementation bundle files")
    if (
        bundle_files.get("hierarchical_all_architecture_discovery.py")
        != inner["orchestrator_implementation_file_sha256"]
    ):
        raise ValueError("hierarchy primary implementation differs from its bundle")
    return {
        "schema_version": inner["schema_version"],
        "orchestrator_version": inner["orchestrator_version"],
        "orchestrator_implementation_file_sha256": inner["orchestrator_implementation_file_sha256"],
        "orchestrator_implementation_bundle": _clone(bundle),
        "orchestrator_implementation_bundle_sha256": declared_bundle_sha256,
        "downstream_contract": inner["downstream_contract"],
        "assurances": inner["assurances"],
    }


def _fold_snapshot(binding: OrderedFoldDiscoveryAgent) -> dict[str, Any]:
    """Capture and cross-check one complete, already-static wrapper packet."""

    binding.__post_init__()
    agent = binding.agent
    approval_sha256, packet = _wrapper_packet(agent)
    catalog_binding = _mapping(packet["catalog_binding"], label="catalog_binding")
    chunk_binding = _mapping(packet["chunk_plan_binding"], label="chunk_plan_binding")
    numerical_binding = _mapping(
        packet.get(
            "direct_numerical_contract_binding",
            packet.get("direct_numerical_manifest_binding"),
        ),
        label="direct_numerical_contract_binding",
    )
    hierarchy = _mapping(packet["hierarchy_precommit"], label="hierarchy_precommit")

    catalog = getattr(agent, "catalog", None)
    chunk_plan = getattr(agent, "chunk_plan", None)
    contract_kind, contract_sha256 = _contract_identity_from_binding(numerical_binding)
    agent_contract_kind = getattr(agent, "direct_numerical_contract_kind", None)
    agent_contract_sha256 = getattr(agent, "direct_numerical_contract_sha256", None)
    if agent_contract_kind is None and agent_contract_sha256 is None:
        manifest = getattr(agent, "direct_numerical_manifest", None)
        agent_contract_kind = DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
        agent_contract_sha256 = getattr(manifest, "content_sha256", None)
    if catalog_binding.get("outer_fold") != binding.outer_fold:
        raise ValueError("wrapper catalog binding cites a different outer fold")
    if catalog_binding.get("split_fingerprint") != getattr(catalog, "split_fingerprint", None):
        raise ValueError("wrapper split binding differs from the fold catalog")
    if catalog_binding.get("catalog_sha256") != getattr(catalog, "catalog_sha256", None):
        raise ValueError("wrapper catalog SHA differs from the fold catalog")
    if chunk_binding.get("plan_sha256") != getattr(chunk_plan, "plan_sha256", None):
        raise ValueError("wrapper chunk-plan SHA differs from the fold agent")
    architecture_chunk_limits = {
        key: _positive_int(chunk_binding.get(key), label=f"chunk_plan_binding.{key}")
        for key in _ARCHITECTURE_CHUNK_LIMIT_KEYS
    }
    for key, value in architecture_chunk_limits.items():
        if getattr(chunk_plan, key, None) != value:
            raise ValueError(f"wrapper {key} differs from the fold chunk plan")
    inner_packet = _mapping(hierarchy.get("packet"), label="hierarchy precommit packet")
    inner_chunk = _mapping(
        inner_packet.get("chunk_plan_binding"), label="inner chunk_plan_binding"
    )
    if (
        inner_chunk.get("max_semantic_member_ids_per_chunk")
        != architecture_chunk_limits["max_semantic_member_ids_per_chunk"]
    ):
        raise ValueError("inner and outer semantic-member chunk bounds differ")
    config_bounds = _mapping(packet.get("config_bounds"), label="config_bounds")
    if (
        config_bounds.get("max_semantic_member_ids_per_chunk")
        != architecture_chunk_limits["max_semantic_member_ids_per_chunk"]
    ):
        raise ValueError("hierarchy config and chunk plan semantic-member bounds differ")
    if (contract_kind, contract_sha256) != (
        agent_contract_kind,
        agent_contract_sha256,
    ):
        raise ValueError("wrapper direct numerical contract differs from the fold agent")

    split_sha256 = _require_sha256(
        catalog_binding.get("split_fingerprint"), label="split_fingerprint"
    )
    catalog_sha256 = _require_sha256(catalog_binding.get("catalog_sha256"), label="catalog_sha256")
    chunk_plan_sha256 = _require_sha256(chunk_binding.get("plan_sha256"), label="chunk_plan_sha256")
    inner_sha256 = _require_sha256(
        hierarchy.get("precommit_sha256"), label="hierarchy precommit_sha256"
    )
    if inner_sha256 != getattr(agent, "inner_precommit_sha256", None):
        raise ValueError("wrapper hierarchy SHA differs from the fold agent")

    return {
        "ordinal": binding.outer_fold,
        "outer_fold": binding.outer_fold,
        "split_fingerprint_sha256": split_sha256,
        "catalog_sha256": catalog_sha256,
        "chunk_plan_sha256": chunk_plan_sha256,
        "architecture_chunk_limits": architecture_chunk_limits,
        "direct_numerical_contract_kind": contract_kind,
        "direct_numerical_contract_sha256": contract_sha256,
        "hierarchy_precommit_sha256": inner_sha256,
        "wrapper_approval_sha256": approval_sha256,
        "wrapper_packet": packet,
    }


def _common_bindings(fold_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not fold_rows:
        raise ValueError("at least one outer-fold agent is required")
    first_packet = fold_rows[0]["wrapper_packet"]
    first = {
        "runner_identity": _clone(first_packet["runner_identity"]),
        "compiler_binding": _clone(first_packet["compiler_binding"]),
        "hierarchy_config_identity": _clone(first_packet["config_bounds"]),
        "architecture_chunk_limits": _clone(fold_rows[0]["architecture_chunk_limits"]),
        "hierarchy_implementation_identity": _hierarchy_common_identity(first_packet),
        "direct_numerical_contract_kind": fold_rows[0]["direct_numerical_contract_kind"],
    }
    for row in fold_rows[1:]:
        packet = row["wrapper_packet"]
        current = {
            "runner_identity": _clone(packet["runner_identity"]),
            "compiler_binding": _clone(packet["compiler_binding"]),
            "hierarchy_config_identity": _clone(packet["config_bounds"]),
            "architecture_chunk_limits": _clone(row["architecture_chunk_limits"]),
            "hierarchy_implementation_identity": _hierarchy_common_identity(packet),
            "direct_numerical_contract_kind": row["direct_numerical_contract_kind"],
        }
        for key in (
            "runner_identity",
            "compiler_binding",
            "hierarchy_config_identity",
            "architecture_chunk_limits",
            "hierarchy_implementation_identity",
            "direct_numerical_contract_kind",
        ):
            if canonical_json(current[key]) != canonical_json(first[key]):
                raise ValueError(f"outer folds have mixed {key.replace('_', ' ')}")
    return {
        **first,
        "runner_identity_sha256": content_sha256(first["runner_identity"]),
        "compiler_binding_sha256": content_sha256(first["compiler_binding"]),
        "hierarchy_config_identity_sha256": content_sha256(first["hierarchy_config_identity"]),
        "architecture_chunk_limits_sha256": content_sha256(
            first["architecture_chunk_limits"]
        ),
        "hierarchy_implementation_identity_sha256": content_sha256(
            first["hierarchy_implementation_identity"]
        ),
    }


def _result_binding(result: Any, *, outer_fold: int) -> dict[str, Any]:
    validator = getattr(result, "validate_authentication", None)
    if not callable(validator):
        raise TypeError("fold discovery result must expose validate_authentication()")
    validator()
    completed = getattr(result, "completed", None)
    registry = getattr(result, "compiled_registry", None)
    trace = getattr(result, "runner_trace", None)
    row = {
        "outer_fold": outer_fold,
        "wrapper_approval_sha256": _require_sha256(
            getattr(result, "wrapper_approval_sha256", None),
            label="result wrapper_approval_sha256",
        ),
        "fold_result_sha256": _require_sha256(
            getattr(result, "result_sha256", None), label="fold result_sha256"
        ),
        "inner_precommit_sha256": _require_sha256(
            getattr(result, "inner_precommit_sha256", None),
            label="result inner_precommit_sha256",
        ),
        "direct_numerical_contract_kind": getattr(
            result,
            "direct_numerical_contract_kind",
            DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST,
        ),
        "direct_numerical_contract_sha256": _require_sha256(
            getattr(
                result,
                "direct_numerical_contract_sha256",
                getattr(result, "direct_numerical_manifest_sha256", None),
            ),
            label="result direct_numerical_contract_sha256",
        ),
        "completion_sha256": _require_sha256(
            getattr(completed, "completion_sha256", None),
            label="result completion_sha256",
        ),
        "compiled_registry_sha256": _require_sha256(
            getattr(registry, "registry_sha256", None),
            label="result compiled_registry_sha256",
        ),
        "runner_trace_sha256": _require_sha256(
            getattr(trace, "trace_sha256", None), label="result runner_trace_sha256"
        ),
        "numerical_binding_audit_sha256": _require_sha256(
            getattr(result, "numerical_binding_audit_sha256", None),
            label="result numerical_binding_audit_sha256",
        ),
    }
    if row["direct_numerical_contract_kind"] not in DIRECT_NUMERICAL_CONTRACT_KINDS:
        raise ValueError("result direct numerical contract kind is unsupported")
    return row


@dataclass(frozen=True)
class OrderedFoldDiscoveryResult:
    outer_fold: int
    expected_wrapper_approval_sha256: str
    expected_direct_numerical_contract_kind: str
    expected_direct_numerical_contract_sha256: str
    result: ApprovedHierarchicalDiscoveryResult
    _binding_json: str = field(repr=False)

    @classmethod
    def create(
        cls,
        *,
        outer_fold: int,
        expected_wrapper_approval_sha256: str,
        expected_direct_numerical_contract_kind: str,
        expected_direct_numerical_contract_sha256: str,
        result: ApprovedHierarchicalDiscoveryResult,
    ) -> "OrderedFoldDiscoveryResult":
        binding = _result_binding(result, outer_fold=outer_fold)
        if binding["wrapper_approval_sha256"] != expected_wrapper_approval_sha256:
            raise ValueError("fold result cites a different reviewed wrapper")
        if binding["direct_numerical_contract_kind"] != (
            expected_direct_numerical_contract_kind
        ) or binding["direct_numerical_contract_sha256"] != (
            expected_direct_numerical_contract_sha256
        ):
            raise ValueError("fold result cites a different approved numerical contract")
        return cls(
            outer_fold=outer_fold,
            expected_wrapper_approval_sha256=expected_wrapper_approval_sha256,
            expected_direct_numerical_contract_kind=(expected_direct_numerical_contract_kind),
            expected_direct_numerical_contract_sha256=(expected_direct_numerical_contract_sha256),
            result=result,
            _binding_json=canonical_json(binding),
        )

    @property
    def binding(self) -> dict[str, Any]:
        return json.loads(self._binding_json)

    def validate_authentication(self) -> None:
        _positive_int(self.outer_fold, label="result outer_fold")
        _require_sha256(
            self.expected_wrapper_approval_sha256,
            label="expected_wrapper_approval_sha256",
        )
        if self.expected_direct_numerical_contract_kind not in DIRECT_NUMERICAL_CONTRACT_KINDS:
            raise ValueError("expected direct numerical contract kind is unsupported")
        _require_sha256(
            self.expected_direct_numerical_contract_sha256,
            label="expected_direct_numerical_contract_sha256",
        )
        current = _result_binding(self.result, outer_fold=self.outer_fold)
        if current["wrapper_approval_sha256"] != self.expected_wrapper_approval_sha256:
            raise ValueError("fold result wrapper differs from the batch precommit")
        if current["direct_numerical_contract_kind"] != (
            self.expected_direct_numerical_contract_kind
        ) or current["direct_numerical_contract_sha256"] != (
            self.expected_direct_numerical_contract_sha256
        ):
            raise ValueError("fold result numerical contract differs from batch precommit")
        if canonical_json(current) != self._binding_json:
            raise ValueError("authenticated fold result mutated after batch assembly")


@dataclass(frozen=True)
class ApprovedHierarchicalDiscoveryBatchResult:
    batch_approval_sha256: str
    input_manifest_sha256: str
    frozen_review_policy_sha256: str
    ordered_fold_results: tuple[OrderedFoldDiscoveryResult, ...]
    result_sha256: str
    _precommit_packet_json: str = field(repr=False)

    def __post_init__(self) -> None:
        self.validate_authentication()

    @property
    def precommit_packet(self) -> dict[str, Any]:
        return json.loads(self._precommit_packet_json)

    def _identity_without_sha(self) -> dict[str, Any]:
        return {
            "schema_version": APPROVED_HIERARCHICAL_DISCOVERY_BATCH_RESULT_VERSION,
            "batch_approval_sha256": self.batch_approval_sha256,
            "input_manifest_sha256": self.input_manifest_sha256,
            "frozen_review_policy_sha256": self.frozen_review_policy_sha256,
            "ordered_fold_result_bindings": [row.binding for row in self.ordered_fold_results],
        }

    def validate_authentication(self) -> None:
        _require_sha256(self.batch_approval_sha256, label="batch_approval_sha256")
        _require_sha256(self.input_manifest_sha256, label="input_manifest_sha256")
        _require_sha256(
            self.frozen_review_policy_sha256,
            label="frozen_review_policy_sha256",
        )
        _require_sha256(self.result_sha256, label="batch result_sha256")
        packet = self.precommit_packet
        if content_sha256(packet) != self.batch_approval_sha256:
            raise ValueError("batch result contains a tampered precommit packet")
        expected_folds = tuple(range(1, len(self.ordered_fold_results) + 1))
        if tuple(row.outer_fold for row in self.ordered_fold_results) != expected_folds:
            raise ValueError("batch results are missing, duplicated, or out of fold order")
        packet_rows = packet.get("ordered_folds")
        if not isinstance(packet_rows, list) or len(packet_rows) != len(self.ordered_fold_results):
            raise ValueError("batch result fold count differs from the precommit")
        for row, packet_row in zip(self.ordered_fold_results, packet_rows):
            row.validate_authentication()
            if packet_row.get("outer_fold") != row.outer_fold:
                raise ValueError("batch result fold differs from precommit order")
            if packet_row.get("wrapper_approval_sha256") != row.expected_wrapper_approval_sha256:
                raise ValueError("batch result wrapper differs from precommit")
            if packet_row.get("direct_numerical_contract_kind") != (
                row.expected_direct_numerical_contract_kind
            ) or packet_row.get("direct_numerical_contract_sha256") != (
                row.expected_direct_numerical_contract_sha256
            ):
                raise ValueError("batch result numerical contract differs from precommit")
        if packet.get("input_manifest_sha256") != self.input_manifest_sha256:
            raise ValueError("batch result input manifest differs from precommit")
        policy = _mapping(
            packet.get("frozen_review_evidence_policy"),
            label="precommit frozen review evidence policy",
        )
        if policy.get("policy_sha256") != self.frozen_review_policy_sha256:
            raise ValueError("batch result frozen review policy differs from precommit")
        if content_sha256(self._identity_without_sha()) != self.result_sha256:
            raise ValueError("result_sha256 does not authenticate the ordered batch results")

    @classmethod
    def create(
        cls,
        *,
        precommit: ApprovedHierarchicalDiscoveryBatchPrecommit,
        input_manifest_sha256: str,
        review_policy: FrozenReviewEvidencePolicyBinding,
        ordered_fold_results: Sequence[OrderedFoldDiscoveryResult],
    ) -> "ApprovedHierarchicalDiscoveryBatchResult":
        rows = tuple(ordered_fold_results)
        identity = {
            "schema_version": APPROVED_HIERARCHICAL_DISCOVERY_BATCH_RESULT_VERSION,
            "batch_approval_sha256": precommit.approval_sha256,
            "input_manifest_sha256": input_manifest_sha256,
            "frozen_review_policy_sha256": review_policy.policy_sha256,
            "ordered_fold_result_bindings": [row.binding for row in rows],
        }
        return cls(
            batch_approval_sha256=precommit.approval_sha256,
            input_manifest_sha256=input_manifest_sha256,
            frozen_review_policy_sha256=review_policy.policy_sha256,
            ordered_fold_results=rows,
            result_sha256=content_sha256(identity),
            _precommit_packet_json=canonical_json(precommit.packet),
        )


class ApprovedHierarchicalDiscoveryBatchCoordinator:
    """Prepare one run packet and enforce all-fold preflight before execution."""

    def __init__(
        self,
        *,
        input_manifest_sha256: str,
        fold_agents: Sequence[OrderedFoldDiscoveryAgent],
        frozen_review_evidence_policy: FrozenReviewEvidencePolicyBinding,
    ) -> None:
        self.input_manifest_sha256 = _require_sha256(
            input_manifest_sha256, label="input_manifest_sha256"
        )
        if not isinstance(frozen_review_evidence_policy, FrozenReviewEvidencePolicyBinding):
            raise TypeError(
                "frozen_review_evidence_policy must be FrozenReviewEvidencePolicyBinding"
            )
        frozen_review_evidence_policy.validate_authentication()
        self.frozen_review_evidence_policy = frozen_review_evidence_policy
        self.fold_agents = tuple(fold_agents)
        self._validate_fold_order()

        # Preparation is static.  It cannot inspect cache entries or invoke a
        # remote runner, and it ensures the rendered packet starts from fully
        # reauthenticated fold wrappers.
        rows = self._preflight_all_folds()
        common = _common_bindings(rows)
        self._validate_adaptive_chunk_limits(common)
        self._fold_rows_json = canonical_json(rows)
        self._common_bindings_json = canonical_json(common)
        self._policy_json = canonical_json(frozen_review_evidence_policy.as_dict())
        self._implementation_file_sha256 = _implementation_sha256()
        self.precommit = ApprovedHierarchicalDiscoveryBatchPrecommit.create(
            self._offline_packet(fold_rows=rows, common_bindings=common)
        )

    def _validate_adaptive_chunk_limits(self, common: Mapping[str, Any]) -> None:
        adaptive = self.frozen_review_evidence_policy.adaptive_config()
        adaptive_limits = {
            key: getattr(adaptive, key) for key in _ARCHITECTURE_CHUNK_LIMIT_KEYS
        }
        initial_limits = dict(
            _mapping(
                common.get("architecture_chunk_limits"),
                label="common architecture_chunk_limits",
            )
        )
        if initial_limits != adaptive_limits:
            raise ValueError(
                "initial and adaptive architecture chunk limits differ; "
                f"initial={initial_limits}, adaptive={adaptive_limits}"
            )

    def _validate_fold_order(self) -> None:
        if not self.fold_agents:
            raise ValueError("at least one outer-fold agent is required")
        for row in self.fold_agents:
            if not isinstance(row, OrderedFoldDiscoveryAgent):
                raise TypeError("fold_agents must contain OrderedFoldDiscoveryAgent entries")
            row.__post_init__()
        observed = tuple(row.outer_fold for row in self.fold_agents)
        if len(set(observed)) != len(observed):
            raise ValueError("outer-fold agents contain a duplicate fold")
        expected = tuple(range(1, len(observed) + 1))
        if observed != expected:
            raise ValueError(
                "outer-fold agents must be complete and ordered contiguously from fold 1"
            )

    def _preflight_all_folds(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for binding in self.fold_agents:
            # This public fold-agent method is deliberately cache- and
            # transport-free.  Do not replace it with execute or cache probing.
            binding.agent.validate_precommit_unchanged()
            rows.append(_fold_snapshot(binding))
        return rows

    def _offline_packet(
        self,
        *,
        fold_rows: Sequence[Mapping[str, Any]],
        common_bindings: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": APPROVED_HIERARCHICAL_DISCOVERY_BATCH_PRECOMMIT_VERSION,
            "coordinator_version": (APPROVED_HIERARCHICAL_DISCOVERY_BATCH_COORDINATOR_VERSION),
            "coordinator_code_identity": {
                "class": (f"{self.__class__.__module__}.{self.__class__.__qualname__}"),
                "implementation_file_sha256": self._implementation_file_sha256,
            },
            "input_manifest_sha256": self.input_manifest_sha256,
            "frozen_review_evidence_policy": (self.frozen_review_evidence_policy.as_dict()),
            "ordered_outer_folds": [row.outer_fold for row in self.fold_agents],
            "ordered_folds": [_clone(row) for row in fold_rows],
            "common_bindings": _clone(common_bindings),
            "assurances": {
                "all_fold_wrapper_packets_included_in_full": True,
                "outer_folds_unique_complete_and_one_based": True,
                "all_fold_static_preflights_before_first_cache_lookup": True,
                "all_fold_static_preflights_before_first_remote_call": True,
                "wrong_or_missing_batch_approval_rejected_before_preflight": True,
                "per_fold_execution_uses_exact_wrapper_approval_sha256": True,
                "mixed_runner_compiler_or_hierarchy_config_allowed": False,
                "round_1_frozen_review_evidence_is_accepted_support_only": True,
                "architecture_wide_review_evidence_dump_allowed": False,
                "later_round_feature_rediscovery_uses_fresh_exact_spent_catalog": True,
                "later_round_all_ten_architectures_required": True,
                "later_round_architecture_at_a_time_interpretation_required": True,
                "later_round_compact_ten_dossier_planner_required": True,
                "later_round_bounded_requested_id_lookback_only": True,
                "later_round_executable_definition_uses_requested_atoms_only": True,
                "later_round_proposal_frozen_before_next_gate": True,
                "same_frozen_review_evidence_used_for_every_round": False,
                "adaptive_reconsideration_identity_authenticated": True,
                "ordered_batch_results_content_authenticated": True,
            },
        }

    def _assert_unchanged(self) -> None:
        self.precommit.__post_init__()
        self._validate_fold_order()
        self.frozen_review_evidence_policy.validate_authentication()
        if canonical_json(self.frozen_review_evidence_policy.as_dict()) != self._policy_json:
            raise ValueError("frozen review evidence policy mutated after preparation")
        if _implementation_sha256() != self._implementation_file_sha256:
            raise ValueError("batch coordinator implementation changed after preparation")
        rows = self._preflight_all_folds()
        common = _common_bindings(rows)
        self._validate_adaptive_chunk_limits(common)
        if canonical_json(rows) != self._fold_rows_json:
            raise ValueError("one or more fold wrappers mutated after batch preparation")
        if canonical_json(common) != self._common_bindings_json:
            raise ValueError("common fold identities mutated after batch preparation")
        regenerated = self._offline_packet(fold_rows=rows, common_bindings=common)
        if content_sha256(regenerated) != self.precommit.approval_sha256 or canonical_json(
            regenerated
        ) != canonical_json(self.precommit.packet):
            raise ValueError("batch offline packet mutated after preparation")

    def render_offline_precommit(self, *, indent: int = 2) -> str:
        return self.precommit.render_json(indent=indent)

    def execute(self, *, approved_batch_sha256: str) -> ApprovedHierarchicalDiscoveryBatchResult:
        """Execute folds only after exact batch approval and all-fold preflight."""

        # This comparison is intentionally the first operation.  In
        # particular, no fold preflight, cache access, or runner call may occur
        # for a missing or incorrect approval.
        if approved_batch_sha256 != self.precommit.approval_sha256:
            raise ValueError("approved batch SHA-256 does not match the offline packet")

        # Complete the static validation loop before entering the first fold's
        # execute method, which is the first place cache/remote work is allowed.
        self._assert_unchanged()

        ordered_results: list[OrderedFoldDiscoveryResult] = []
        stored_rows = json.loads(self._fold_rows_json)
        for binding, stored in zip(self.fold_agents, stored_rows):
            result = binding.agent.execute(
                approved_wrapper_sha256=stored["wrapper_approval_sha256"]
            )
            ordered_results.append(
                OrderedFoldDiscoveryResult.create(
                    outer_fold=binding.outer_fold,
                    expected_wrapper_approval_sha256=stored["wrapper_approval_sha256"],
                    expected_direct_numerical_contract_kind=stored[
                        "direct_numerical_contract_kind"
                    ],
                    expected_direct_numerical_contract_sha256=stored[
                        "direct_numerical_contract_sha256"
                    ],
                    result=result,
                )
            )
        return ApprovedHierarchicalDiscoveryBatchResult.create(
            precommit=self.precommit,
            input_manifest_sha256=self.input_manifest_sha256,
            review_policy=self.frozen_review_evidence_policy,
            ordered_fold_results=ordered_results,
        )


__all__ = [
    "APPROVED_HIERARCHICAL_DISCOVERY_BATCH_COORDINATOR_VERSION",
    "APPROVED_HIERARCHICAL_DISCOVERY_BATCH_PRECOMMIT_VERSION",
    "APPROVED_HIERARCHICAL_DISCOVERY_BATCH_RESULT_VERSION",
    "FROZEN_REVIEW_EVIDENCE_POLICY_VERSION",
    "ApprovedHierarchicalDiscoveryBatchCoordinator",
    "ApprovedHierarchicalDiscoveryBatchPrecommit",
    "ApprovedHierarchicalDiscoveryBatchResult",
    "FrozenReviewEvidencePolicyBinding",
    "OrderedFoldDiscoveryAgent",
    "OrderedFoldDiscoveryResult",
]
