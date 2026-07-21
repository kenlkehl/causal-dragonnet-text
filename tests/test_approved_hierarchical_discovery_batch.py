from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT,
    canonical_json,
    content_sha256,
)
from oci.inference.approved_hierarchical_discovery_batch import (
    ApprovedHierarchicalDiscoveryBatchCoordinator,
    FrozenReviewEvidencePolicyBinding,
    OrderedFoldDiscoveryAgent,
)
from oci.inference.adaptive_hierarchical_stage1_reconsideration import (
    AdaptiveReconsiderationConfig,
    adaptive_hierarchical_stage1_reconsideration_identity,
)
from oci.inference.frozen_hierarchical_review_evidence import (
    frozen_hierarchical_review_evidence_identity,
)


def _digest(label: str) -> str:
    return content_sha256({"label": label})


def _review_materializer_identity() -> dict[str, Any]:
    return frozen_hierarchical_review_evidence_identity()


class _StubPrecommit:
    def __init__(self, packet: dict[str, Any]) -> None:
        self._packet_json = canonical_json(packet)
        self.approval_sha256 = content_sha256(packet)

    @property
    def packet(self) -> dict[str, Any]:
        return json.loads(self._packet_json)

    def __post_init__(self) -> None:
        assert content_sha256(self.packet) == self.approval_sha256


class _StubFoldResult:
    def __init__(
        self,
        *,
        outer_fold: int,
        wrapper_sha256: str,
        contract_sha256: str | None = None,
    ) -> None:
        self.wrapper_approval_sha256 = wrapper_sha256
        self.inner_precommit_sha256 = _digest(f"inner-{outer_fold}")
        self.direct_numerical_contract_kind = DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
        self.direct_numerical_contract_sha256 = (
            _digest(f"intent-{outer_fold}") if contract_sha256 is None else contract_sha256
        )
        self.completed = SimpleNamespace(completion_sha256=_digest(f"completion-{outer_fold}"))
        self.compiled_registry = SimpleNamespace(registry_sha256=_digest(f"registry-{outer_fold}"))
        self.runner_trace = SimpleNamespace(trace_sha256=_digest(f"trace-{outer_fold}"))
        self.numerical_binding_audit_sha256 = _digest(f"audit-{outer_fold}")
        self.result_sha256 = self._expected_sha256()

    def _identity(self) -> dict[str, Any]:
        return {
            "wrapper_approval_sha256": self.wrapper_approval_sha256,
            "inner_precommit_sha256": self.inner_precommit_sha256,
            "direct_numerical_contract_kind": self.direct_numerical_contract_kind,
            "direct_numerical_contract_sha256": self.direct_numerical_contract_sha256,
            "completion_sha256": self.completed.completion_sha256,
            "compiled_registry_sha256": self.compiled_registry.registry_sha256,
            "runner_trace_sha256": self.runner_trace.trace_sha256,
            "numerical_binding_audit_sha256": self.numerical_binding_audit_sha256,
        }

    def _expected_sha256(self) -> str:
        return content_sha256(self._identity())

    def validate_authentication(self) -> None:
        if self.result_sha256 != self._expected_sha256():
            raise ValueError("stub result mutated")


class _StubFoldAgent:
    def __init__(
        self,
        *,
        outer_fold: int,
        events: list[str],
        runner_seed: str = "common-runner",
        compiler_seed: str = "common-compiler",
        config_max_features: int = 16,
        max_atoms_per_chunk: int = 2,
        max_bytes_per_chunk: int = 48_000,
        max_semantic_member_ids_per_chunk: int = 3,
        result_contract_sha256: str | None = None,
    ) -> None:
        self.outer_fold = outer_fold
        self.events = events
        self.result_contract_sha256 = result_contract_sha256
        self.catalog = SimpleNamespace(
            outer_fold=outer_fold,
            split_fingerprint=_digest(f"split-{outer_fold}"),
            catalog_sha256=_digest(f"catalog-{outer_fold}"),
        )
        self.chunk_plan = SimpleNamespace(
            plan_sha256=_digest(f"chunks-{outer_fold}"),
            max_atoms_per_chunk=max_atoms_per_chunk,
            max_bytes_per_chunk=max_bytes_per_chunk,
            max_semantic_member_ids_per_chunk=max_semantic_member_ids_per_chunk,
        )
        self.direct_numerical_contract_kind = DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
        self.direct_numerical_contract_sha256 = _digest(f"intent-{outer_fold}")
        self.inner_precommit_sha256 = _digest(f"inner-{outer_fold}")
        runner_identity = {
            "runner": runner_seed,
            "identity_sha256": _digest(f"runner-identity-{runner_seed}"),
        }
        compiler_binding = {
            "compiler": compiler_seed,
            "identity_sha256": _digest(f"compiler-identity-{compiler_seed}"),
            "implementation_file_sha256": _digest("compiler-implementation"),
        }
        config = {
            "max_integrated_features": config_max_features,
            "max_semantic_member_ids_per_chunk": max_semantic_member_ids_per_chunk,
            "selector_thinking_token_budget": 5000,
        }
        hierarchy_implementation_sha256 = _digest("hierarchy-implementation")
        hierarchy_bundle_body = {
            "schema_version": "hierarchical-discovery-test-bundle-v1",
            "files": {
                "hierarchical_all_architecture_discovery.py": (hierarchy_implementation_sha256),
                "all_evidence_discovery_interfaces.py": _digest("interfaces-implementation"),
                "hierarchical_discovery_response_contract.py": _digest(
                    "response-contract-implementation"
                ),
                "lossless_stage1_evidence_catalog.py": _digest("catalog-implementation"),
            },
        }
        hierarchy_bundle = {
            **hierarchy_bundle_body,
            "implementation_bundle_sha256": content_sha256(hierarchy_bundle_body),
        }
        hierarchy_packet = {
            "schema_version": "hierarchy-precommit-v1",
            "orchestrator_version": "hierarchy-v1",
            "orchestrator_implementation_file_sha256": hierarchy_implementation_sha256,
            "orchestrator_implementation_bundle": hierarchy_bundle,
            "orchestrator_implementation_bundle_sha256": hierarchy_bundle[
                "implementation_bundle_sha256"
            ],
            "runner_identity": runner_identity,
            "config": config,
            "chunk_plan_binding": {
                "plan_sha256": self.chunk_plan.plan_sha256,
                "max_semantic_member_ids_per_chunk": max_semantic_member_ids_per_chunk,
            },
            "downstream_contract": {
                "architecture_order": ["all-ten-in-canonical-order"],
                "one_architecture_at_a_time": True,
            },
            "assurances": {"raw_evidence_hierarchical": True},
            "fold_specific_job_ledger": _digest(f"ledger-{outer_fold}"),
        }
        wrapper_packet = {
            "schema_version": "wrapper-precommit-v1",
            "catalog_binding": {
                "outer_fold": outer_fold,
                "split_fingerprint": self.catalog.split_fingerprint,
                "catalog_sha256": self.catalog.catalog_sha256,
            },
            "chunk_plan_binding": {
                "plan_sha256": self.chunk_plan.plan_sha256,
                "max_atoms_per_chunk": max_atoms_per_chunk,
                "max_bytes_per_chunk": max_bytes_per_chunk,
                "max_semantic_member_ids_per_chunk": max_semantic_member_ids_per_chunk,
            },
            "direct_numerical_contract_binding": {
                "direct_numerical_contract_kind": self.direct_numerical_contract_kind,
                "direct_numerical_contract_sha256": (self.direct_numerical_contract_sha256),
            },
            "hierarchy_precommit": {
                "precommit_sha256": self.inner_precommit_sha256,
                "packet": hierarchy_packet,
            },
            "runner_identity": runner_identity,
            "compiler_binding": compiler_binding,
            "config_bounds": config,
            "assurances": {"remote_requires_wrapper_approval": True},
        }
        self.precommit = _StubPrecommit(wrapper_packet)
        self.execute_calls = 0
        self.cache_lookups = 0
        self.remote_calls = 0

    def validate_precommit_unchanged(self) -> None:
        self.events.append(f"preflight:{self.outer_fold}")
        self.precommit.__post_init__()

    def execute(self, *, approved_wrapper_sha256: str) -> _StubFoldResult:
        self.events.append(f"execute:{self.outer_fold}")
        self.execute_calls += 1
        self.cache_lookups += 1
        self.events.append(f"cache:{self.outer_fold}")
        if approved_wrapper_sha256 != self.precommit.approval_sha256:
            raise ValueError("wrong fold approval")
        self.remote_calls += 1
        self.events.append(f"remote:{self.outer_fold}")
        return _StubFoldResult(
            outer_fold=self.outer_fold,
            wrapper_sha256=approved_wrapper_sha256,
            contract_sha256=self.result_contract_sha256,
        )


def _policy(
    *,
    max_atoms_per_chunk: int = 2,
    max_bytes_per_chunk: int = 48_000,
    max_semantic_member_ids_per_chunk: int = 3,
) -> FrozenReviewEvidencePolicyBinding:
    return FrozenReviewEvidencePolicyBinding(
        max_evidence_ids=32,
        max_evidence_bytes=64_000,
        review_materializer_identity=_review_materializer_identity(),
        adaptive_reconsideration_identity=(
            adaptive_hierarchical_stage1_reconsideration_identity(
                AdaptiveReconsiderationConfig(
                    max_atoms_per_chunk=max_atoms_per_chunk,
                    max_bytes_per_chunk=max_bytes_per_chunk,
                    max_semantic_member_ids_per_chunk=(max_semantic_member_ids_per_chunk),
                )
            )
        ),
        accepted_support_only=True,
    )


def _coordinator(
    *agents: _StubFoldAgent,
) -> ApprovedHierarchicalDiscoveryBatchCoordinator:
    return ApprovedHierarchicalDiscoveryBatchCoordinator(
        input_manifest_sha256=_digest("immutable-input-manifest"),
        fold_agents=tuple(
            OrderedFoldDiscoveryAgent(outer_fold=index, agent=agent)  # type: ignore[arg-type]
            for index, agent in enumerate(agents, start=1)
        ),
        frozen_review_evidence_policy=_policy(),
    )


def test_wrong_or_missing_batch_approval_has_zero_cache_or_remote_calls() -> None:
    events: list[str] = []
    agents = (_StubFoldAgent(outer_fold=1, events=events),)
    coordinator = _coordinator(*agents)
    prepared_events = tuple(events)

    with pytest.raises(ValueError, match="approved batch SHA-256"):
        coordinator.execute(approved_batch_sha256=_digest("wrong"))
    with pytest.raises(ValueError, match="approved batch SHA-256"):
        coordinator.execute(approved_batch_sha256=None)  # type: ignore[arg-type]

    assert tuple(events) == prepared_events
    assert agents[0].cache_lookups == 0
    assert agents[0].remote_calls == 0


def test_every_fold_is_preflighted_before_first_cache_or_remote_call() -> None:
    events: list[str] = []
    agents = (
        _StubFoldAgent(outer_fold=1, events=events),
        _StubFoldAgent(outer_fold=2, events=events),
        _StubFoldAgent(outer_fold=3, events=events),
    )
    coordinator = _coordinator(*agents)
    events.clear()

    result = coordinator.execute(approved_batch_sha256=coordinator.precommit.approval_sha256)

    assert events[:3] == ["preflight:1", "preflight:2", "preflight:3"]
    assert events[3] == "execute:1"
    assert [row.outer_fold for row in result.ordered_fold_results] == [1, 2, 3]
    result.validate_authentication()


def test_fold_two_mutation_is_detected_before_fold_one_cache_or_remote_call() -> None:
    events: list[str] = []
    first = _StubFoldAgent(outer_fold=1, events=events)
    second = _StubFoldAgent(outer_fold=2, events=events)
    coordinator = _coordinator(first, second)
    events.clear()
    second.catalog.catalog_sha256 = _digest("mutated-fold-two-catalog")

    with pytest.raises(ValueError, match="wrapper catalog SHA"):
        coordinator.execute(approved_batch_sha256=coordinator.precommit.approval_sha256)

    assert events == ["preflight:1", "preflight:2"]
    assert first.execute_calls == 0
    assert first.cache_lookups == 0
    assert first.remote_calls == 0


@pytest.mark.parametrize("mixed", ["runner", "compiler", "config", "chunk_limits"])
def test_mixed_common_fold_identity_is_rejected(mixed: str) -> None:
    events: list[str] = []
    first = _StubFoldAgent(outer_fold=1, events=events)
    kwargs: dict[str, Any] = {}
    if mixed == "runner":
        kwargs["runner_seed"] = "different-runner"
    elif mixed == "compiler":
        kwargs["compiler_seed"] = "different-compiler"
    elif mixed == "config":
        kwargs["config_max_features"] = 15
    else:
        kwargs["max_atoms_per_chunk"] = 1
    second = _StubFoldAgent(outer_fold=2, events=events, **kwargs)

    with pytest.raises(ValueError, match="mixed"):
        _coordinator(first, second)

    assert first.remote_calls == second.remote_calls == 0


def test_duplicate_missing_or_mislabeled_fold_is_rejected() -> None:
    events: list[str] = []
    first = _StubFoldAgent(outer_fold=1, events=events)
    duplicate = _StubFoldAgent(outer_fold=1, events=events)
    with pytest.raises(ValueError, match="duplicate"):
        ApprovedHierarchicalDiscoveryBatchCoordinator(
            input_manifest_sha256=_digest("manifest"),
            fold_agents=(
                OrderedFoldDiscoveryAgent(outer_fold=1, agent=first),  # type: ignore[arg-type]
                OrderedFoldDiscoveryAgent(outer_fold=1, agent=duplicate),  # type: ignore[arg-type]
            ),
            frozen_review_evidence_policy=_policy(),
        )

    third = _StubFoldAgent(outer_fold=3, events=events)
    with pytest.raises(ValueError, match="complete and ordered"):
        ApprovedHierarchicalDiscoveryBatchCoordinator(
            input_manifest_sha256=_digest("manifest"),
            fold_agents=(
                OrderedFoldDiscoveryAgent(outer_fold=1, agent=first),  # type: ignore[arg-type]
                OrderedFoldDiscoveryAgent(outer_fold=3, agent=third),  # type: ignore[arg-type]
            ),
            frozen_review_evidence_policy=_policy(),
        )

    with pytest.raises(ValueError, match="agent.catalog.outer_fold"):
        OrderedFoldDiscoveryAgent(outer_fold=2, agent=first)  # type: ignore[arg-type]


def test_batch_packet_order_and_digest_are_deterministic_and_include_full_wrappers() -> None:
    events_one: list[str] = []
    agents_one = (
        _StubFoldAgent(outer_fold=1, events=events_one),
        _StubFoldAgent(outer_fold=2, events=events_one),
    )
    first = _coordinator(*agents_one)
    events_two: list[str] = []
    agents_two = (
        _StubFoldAgent(outer_fold=1, events=events_two),
        _StubFoldAgent(outer_fold=2, events=events_two),
    )
    second = _coordinator(*agents_two)

    assert first.precommit.approval_sha256 == second.precommit.approval_sha256
    packet = first.precommit.packet
    assert packet["ordered_outer_folds"] == [1, 2]
    assert [row["outer_fold"] for row in packet["ordered_folds"]] == [1, 2]
    assert packet["ordered_folds"][0]["wrapper_packet"] == agents_one[0].precommit.packet
    assert packet["ordered_folds"][1]["wrapper_packet"] == agents_one[1].precommit.packet
    assert packet["ordered_folds"][0]["direct_numerical_contract_kind"] == (
        DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
    )
    assert packet["ordered_folds"][0]["direct_numerical_contract_sha256"] == (
        agents_one[0].direct_numerical_contract_sha256
    )
    assert "direct_numerical_manifest_sha256" not in packet["ordered_folds"][0]
    assert packet["frozen_review_evidence_policy"]["accepted_support_only"] is True
    assert packet["common_bindings"]["architecture_chunk_limits"] == {
        "max_atoms_per_chunk": 2,
        "max_bytes_per_chunk": 48_000,
        "max_semantic_member_ids_per_chunk": 3,
    }
    assert all(
        row["architecture_chunk_limits"] == packet["common_bindings"]["architecture_chunk_limits"]
        for row in packet["ordered_folds"]
    )
    assert packet["assurances"]["all_fold_static_preflights_before_first_remote_call"]


def test_batch_rejects_forged_or_adaptively_divergent_chunk_limits_before_execution() -> None:
    events: list[str] = []
    forged = _StubFoldAgent(outer_fold=1, events=events)
    forged.chunk_plan.max_semantic_member_ids_per_chunk = 63
    with pytest.raises(ValueError, match="wrapper max_semantic_member_ids_per_chunk"):
        _coordinator(forged)
    assert forged.remote_calls == 0

    divergent = _StubFoldAgent(
        outer_fold=1,
        events=events,
        max_semantic_member_ids_per_chunk=2,
    )
    with pytest.raises(ValueError, match="initial and adaptive architecture chunk limits differ"):
        _coordinator(divergent)
    assert divergent.remote_calls == 0


def test_batch_result_detects_even_a_rehashed_nested_result_mutation() -> None:
    events: list[str] = []
    coordinator = _coordinator(_StubFoldAgent(outer_fold=1, events=events))
    result = coordinator.execute(approved_batch_sha256=coordinator.precommit.approval_sha256)
    assert (
        result.ordered_fold_results[0].binding["direct_numerical_contract_kind"]
        == DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
    )
    assert "direct_numerical_manifest_sha256" not in (result.ordered_fold_results[0].binding)
    fold_result = result.ordered_fold_results[0].result
    fold_result.compiled_registry.registry_sha256 = _digest("tampered-registry")
    fold_result.result_sha256 = fold_result._expected_sha256()

    with pytest.raises(ValueError, match="mutated after batch assembly"):
        result.validate_authentication()


def test_batch_rejects_result_that_changes_the_approved_numerical_contract() -> None:
    events: list[str] = []
    coordinator = _coordinator(
        _StubFoldAgent(
            outer_fold=1,
            events=events,
            result_contract_sha256=_digest("different-intent"),
        )
    )

    with pytest.raises(ValueError, match="different approved numerical contract"):
        coordinator.execute(approved_batch_sha256=coordinator.precommit.approval_sha256)


def test_frozen_review_policy_uses_a_closed_phased_adaptive_schema() -> None:
    policy = FrozenReviewEvidencePolicyBinding.from_mapping(
        {
            "max_evidence_ids": 1,
            "max_evidence_bytes": 1,
            "accepted_support_only": True,
            "review_materializer_identity": _review_materializer_identity(),
            "adaptive_reconsideration_identity": (
                adaptive_hierarchical_stage1_reconsideration_identity()
            ),
        }
    )
    policy.validate_authentication()
    body = policy.as_dict()
    assert body["architecture_wide_single_prompt_evidence_dump_allowed"] is False
    assert body["round_1_feature_rediscovery_allowed"] is False
    assert body["later_round_feature_rediscovery_allowed"] is True
    assert body["same_frozen_evidence_used_for_every_round"] is False
    assert (
        body["adaptive_reconsideration_identity"]["prompt_contract"][
            "dynamic_fold_content_in_static_contract"
        ]
        is False
    )
    assert policy.materializer_config().max_evidence_ids == 1
    assert policy.adaptive_config().max_operations == 4

    with pytest.raises(ValueError, match="unexpected closed schema"):
        FrozenReviewEvidencePolicyBinding.from_mapping(
            {
                "max_evidence_ids": 1,
                "max_evidence_bytes": 10,
                "accepted_support_only": True,
                "review_materializer_identity": _review_materializer_identity(),
                "adaptive_reconsideration_identity": (
                    adaptive_hierarchical_stage1_reconsideration_identity()
                ),
                "extra": "not allowed",
            }
        )
    with pytest.raises(ValueError, match="accepted-support-only"):
        FrozenReviewEvidencePolicyBinding(
            max_evidence_ids=1,
            max_evidence_bytes=10,
            review_materializer_identity=_review_materializer_identity(),
            accepted_support_only=False,
        )

    bad_identity = _review_materializer_identity()
    bad_identity["planner_lookback_only_excluded"] = False
    with pytest.raises(ValueError, match="planner_lookback_only_excluded must be true"):
        FrozenReviewEvidencePolicyBinding(
            max_evidence_ids=1,
            max_evidence_bytes=10,
            review_materializer_identity=bad_identity,
        )

    stale_identity = _review_materializer_identity()
    stale_identity["implementation_file_sha256"] = _digest("stale-materializer")
    with pytest.raises(ValueError, match="differs from the current closed implementation"):
        FrozenReviewEvidencePolicyBinding(
            max_evidence_ids=1,
            max_evidence_bytes=10,
            review_materializer_identity=stale_identity,
        )
