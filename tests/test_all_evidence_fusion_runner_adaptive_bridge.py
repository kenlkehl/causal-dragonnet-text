from __future__ import annotations

import ast
import inspect
import json
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

import oci.inference.all_evidence_fusion_runner as runner_module
from oci.inference.adaptive_hierarchical_stage1_reconsideration import (
    AdaptiveCurrentFeature,
)
from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    OUTCOME_AXIS,
    canonical_json,
    content_sha256,
)
from oci.inference.all_evidence_fusion_runner import AllEvidenceFusionRunner
from oci.inference.frozen_hierarchical_review_evidence import (
    FROZEN_HIERARCHICAL_REVIEW_EVIDENCE_SCHEMA_VERSION,
    FrozenHierarchicalReviewEvidence,
    frozen_hierarchical_review_evidence_identity,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
    SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION,
    RoleNeutralEvidenceCatalog,
    Stage1EvidenceAtom,
    validate_role_neutral_catalog,
)


def _catalog() -> RoleNeutralEvidenceCatalog:
    split_fingerprint = "1" * 64
    atoms: list[Stage1EvidenceAtom] = []
    for ordinal, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES, start=1):
        member_id = f"member_{ordinal:03d}"
        origin = {"closed_source": family, "ordinal": ordinal}
        content = {
            "terms": [
                {
                    "member_id": member_id,
                    "term": f"documented {family} patient clue",
                }
            ]
        }
        origin_sha256 = content_sha256(origin)
        content_sha256_value = content_sha256(content)
        identity = {
            "atom_kind": "test_semantic_atom",
            "source_kind": f"closed_test_source_{ordinal:02d}",
            "source_family": family,
            "observable_axes": (OUTCOME_AXIS,),
            "member_ids": (member_id,),
            "split_fingerprint": split_fingerprint,
            "origin_sha256": origin_sha256,
            "content_sha256": content_sha256_value,
        }
        atoms.append(
            Stage1EvidenceAtom(
                evidence_id=f"evidence_{content_sha256(identity)}",
                atom_kind="test_semantic_atom",
                source_kind=f"closed_test_source_{ordinal:02d}",
                source_family=family,
                observable_axes=(OUTCOME_AXIS,),
                member_ids=(member_id,),
                split_fingerprint=split_fingerprint,
                origin_sha256=origin_sha256,
                content_sha256=content_sha256_value,
                _origin_json=canonical_json(origin),
                _content_json=canonical_json(content),
            )
        )
    semantic_member_batching = {
        "schema_version": SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION,
        "semantic_member_batch_size": 1,
        "selection_or_truncation_authorized": False,
        "complete_member_coverage_required": True,
    }
    catalog_identity = {
        "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
        "semantic_member_batching": semantic_member_batching,
        "outer_fold": 1,
        "scope": "outer_train",
        "inner_fold": None,
        "split_fingerprint": split_fingerprint,
        "atoms": [atom.as_dict() for atom in atoms],
        "non_grounding_numerical_summaries": [],
    }
    catalog = RoleNeutralEvidenceCatalog(
        outer_fold=1,
        scope="outer_train",
        inner_fold=None,
        split_fingerprint=split_fingerprint,
        atoms=tuple(atoms),
        non_grounding_numerical_summaries=(),
        catalog_sha256=content_sha256(catalog_identity),
        _audit_json=canonical_json(
            {
                "semantic_member_batching": semantic_member_batching,
                "semantic_member_batch_size": 1,
            }
        ),
    )
    validate_role_neutral_catalog(catalog)
    return catalog


def _continuous(
    name: str,
    *,
    roles: tuple[str, ...] = ("confounder",),
    description: str | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "type": "continuous",
        "roles": list(roles),
        "description": description or f"Baseline numeric value for {name} before treatment.",
    }


def _frozen_review_evidence(
    catalog: RoleNeutralEvidenceCatalog,
    support_by_name: dict[str, tuple[str, ...]],
) -> FrozenHierarchicalReviewEvidence:
    atom_by_id = {atom.evidence_id: atom for atom in catalog.atoms}
    accepted_ids = {
        evidence_id for evidence_ids in support_by_name.values() for evidence_id in evidence_ids
    }
    ordered_ids = tuple(
        atom.evidence_id for atom in catalog.atoms if atom.evidence_id in accepted_ids
    )
    rows = [
        {
            "evidence_id": evidence_id,
            "source_families": [atom_by_id[evidence_id].source_family],
            "role_hint": "",
            "content": atom_by_id[evidence_id].as_discovery_item().content,
        }
        for evidence_id in ordered_ids
    ]
    audit = {
        "accepted_feature_support": [
            {
                "canonical_name": name,
                "supporting_evidence_ids": list(evidence_ids),
            }
            for name, evidence_ids in support_by_name.items()
        ]
    }
    review_evidence_sha256 = content_sha256(rows)
    evidence_bytes = len(canonical_json(rows).encode("utf-8"))
    identity = {
        "schema_version": FROZEN_HIERARCHICAL_REVIEW_EVIDENCE_SCHEMA_VERSION,
        "materializer_identity": frozen_hierarchical_review_evidence_identity(),
        "catalog_sha256": catalog.catalog_sha256,
        "completion_sha256": "a" * 64,
        "precommit_sha256": "b" * 64,
        "ordered_evidence_ids": list(ordered_ids),
        "evidence_count": len(rows),
        "evidence_bytes": evidence_bytes,
        "review_evidence_sha256": review_evidence_sha256,
        "audit": audit,
    }
    return FrozenHierarchicalReviewEvidence(
        catalog_sha256=catalog.catalog_sha256,
        completion_sha256="a" * 64,
        precommit_sha256="b" * 64,
        ordered_evidence_ids=ordered_ids,
        evidence_count=len(rows),
        evidence_bytes=evidence_bytes,
        review_evidence_sha256=review_evidence_sha256,
        binding_sha256=content_sha256(identity),
        _review_rows_json=canonical_json(rows),
        _audit_json=canonical_json(audit),
    )


def _initial_registry(
    catalog: RoleNeutralEvidenceCatalog,
) -> tuple[tuple[AdaptiveCurrentFeature, ...], dict[str, str]]:
    support_by_name = {
        "alpha_measure": (catalog.atoms[0].evidence_id,),
        "beta_measure": (catalog.atoms[1].evidence_id,),
    }
    registry, family_by_id, _audit = AllEvidenceFusionRunner._initial_adaptive_registry(
        specs=[_continuous("alpha_measure"), _continuous("beta_measure")],
        frozen_review_evidence=_frozen_review_evidence(catalog, support_by_name),
        initial_catalog=catalog,
    )
    return registry, family_by_id


def _diagnostic_adapter_audit(
    *,
    score: float = 0.25,
) -> dict[str, object]:
    catalog = _catalog()
    registry, _family_by_id = _initial_registry(catalog)
    _adapted, audit = AllEvidenceFusionRunner._adaptive_diagnostics(
        [
            {
                "diagnostic_id": "diagnostic_0001",
                "kind": "feature_quality",
                "feature_name": "alpha_measure",
                "missingness_rate": score,
            }
        ],
        current_registry=registry,
    )
    return audit


def _support_by_name(
    registry: tuple[AdaptiveCurrentFeature, ...],
) -> dict[str, tuple[str, ...]]:
    return {item.feature_name: item.supporting_evidence_ids for item in registry}


def test_initial_adaptive_registry_uses_exact_frozen_support_and_excludes_roleless() -> None:
    catalog = _catalog()
    support_by_name = {
        "alpha_measure": (catalog.atoms[0].evidence_id,),
        "beta_measure": (
            catalog.atoms[1].evidence_id,
            catalog.atoms[2].evidence_id,
        ),
        "accepted_but_roleless": (catalog.atoms[3].evidence_id,),
    }

    registry, family_by_id, audit = AllEvidenceFusionRunner._initial_adaptive_registry(
        specs=[_continuous("alpha_measure"), _continuous("beta_measure")],
        frozen_review_evidence=_frozen_review_evidence(catalog, support_by_name),
        initial_catalog=catalog,
    )

    assert [item.feature_name for item in registry] == ["alpha_measure", "beta_measure"]
    assert _support_by_name(registry) == {
        "alpha_measure": support_by_name["alpha_measure"],
        "beta_measure": support_by_name["beta_measure"],
    }
    assert registry[1].source_families == (
        catalog.atoms[1].source_family,
        catalog.atoms[2].source_family,
    )
    assert family_by_id == {atom.evidence_id: atom.source_family for atom in catalog.atoms}
    assert audit["frozen_accepted_feature_count"] == 3
    assert audit["modeled_feature_count"] == 2
    assert audit["excluded_nonmodeled_accepted_feature_count"] == 1
    assert audit["modeled_specs_are_unique_subset_of_frozen_accepted_support"] is True
    assert audit["excluded_nonmodeled_features_treated_as_executable"] is False


def test_initial_adaptive_registry_rejects_another_catalog_binding() -> None:
    catalog = _catalog()
    frozen = _frozen_review_evidence(
        catalog,
        {"alpha_measure": (catalog.atoms[0].evidence_id,)},
    )

    with pytest.raises(ValueError, match="cites another initial catalog"):
        AllEvidenceFusionRunner._initial_adaptive_registry(
            specs=[_continuous("alpha_measure")],
            frozen_review_evidence=frozen,
            initial_catalog=replace(catalog, catalog_sha256="f" * 64),
        )


def test_initial_adaptive_registry_rejects_duplicate_modeled_names() -> None:
    catalog = _catalog()
    frozen = _frozen_review_evidence(
        catalog,
        {"alpha_measure": (catalog.atoms[0].evidence_id,)},
    )

    with pytest.raises(ValueError, match="duplicate feature names"):
        AllEvidenceFusionRunner._initial_adaptive_registry(
            specs=[_continuous("alpha_measure"), _continuous("alpha_measure")],
            frozen_review_evidence=frozen,
            initial_catalog=catalog,
        )


@pytest.mark.parametrize(
    ("operation_audit", "after_specs", "expected_support"),
    (
        (
            (
                {
                    "adaptive_operation": "add",
                    "target_names": ["gamma_measure"],
                    "contract": _continuous("gamma_measure"),
                    "supporting_evidence_ids": [],
                },
            ),
            (
                _continuous("alpha_measure"),
                _continuous("beta_measure"),
                _continuous("gamma_measure"),
            ),
            {"gamma_measure": (2,)},
        ),
        (
            (
                {
                    "adaptive_operation": "drop",
                    "target_names": ["alpha_measure"],
                    "contract": None,
                    "supporting_evidence_ids": [],
                },
            ),
            (_continuous("beta_measure"),),
            {"beta_measure": (1,)},
        ),
        (
            (
                {
                    "adaptive_operation": "split",
                    "target_names": ["alpha_measure"],
                    "contract": _continuous("gamma_measure"),
                    "supporting_evidence_ids": [],
                },
            ),
            (
                _continuous("alpha_measure"),
                _continuous("gamma_measure"),
                _continuous("beta_measure"),
            ),
            {"alpha_measure": (0,), "gamma_measure": (2,)},
        ),
        (
            (
                {
                    "adaptive_operation": "rename",
                    "target_names": ["alpha_measure"],
                    "contract": _continuous("gamma_measure"),
                    "supporting_evidence_ids": [],
                },
            ),
            (_continuous("gamma_measure"), _continuous("beta_measure")),
            {"gamma_measure": (0, 2)},
        ),
        (
            (
                {
                    "adaptive_operation": "revise_definition",
                    "target_names": ["alpha_measure"],
                    "contract": _continuous(
                        "alpha_measure",
                        description="Revised baseline alpha measurement before treatment.",
                    ),
                    "supporting_evidence_ids": [],
                },
            ),
            (
                _continuous(
                    "alpha_measure",
                    description="Revised baseline alpha measurement before treatment.",
                ),
                _continuous("beta_measure"),
            ),
            {"alpha_measure": (0, 2)},
        ),
        (
            (
                {
                    "action": "revise",
                    "target_names": ["alpha_measure"],
                    "contract": _continuous(
                        "alpha_measure",
                        description="Legacy revised alpha measurement before treatment.",
                    ),
                    "supporting_evidence_ids": [],
                },
            ),
            (
                _continuous(
                    "alpha_measure",
                    description="Legacy revised alpha measurement before treatment.",
                ),
                _continuous("beta_measure"),
            ),
            {"alpha_measure": (0, 2)},
        ),
        (
            (
                {
                    "adaptive_operation": "merge",
                    "target_names": ["alpha_measure", "beta_measure"],
                    "contract": _continuous("gamma_measure"),
                    "supporting_evidence_ids": [],
                },
            ),
            (_continuous("gamma_measure"),),
            {"gamma_measure": (0, 1, 2)},
        ),
        (
            (
                {
                    "action": "re_role",
                    "target_names": ["alpha_measure"],
                    "contract": _continuous("alpha_measure", roles=("effect_modifier",)),
                    "supporting_evidence_ids": [],
                },
            ),
            (
                _continuous("alpha_measure", roles=("effect_modifier",)),
                _continuous("beta_measure"),
            ),
            {"alpha_measure": (0,)},
        ),
    ),
)
def test_transition_adaptive_registry_preserves_exact_provenance_algebra(
    operation_audit: tuple[dict[str, object], ...],
    after_specs: tuple[dict[str, object], ...],
    expected_support: dict[str, tuple[int, ...]],
) -> None:
    catalog = _catalog()
    before, family_by_id = _initial_registry(catalog)
    current_evidence_id = catalog.atoms[2].evidence_id
    materialized_operations = tuple(
        {
            **operation,
            "supporting_evidence_ids": (
                []
                if operation[
                    "adaptive_operation" if "adaptive_operation" in operation else "action"
                ]
                in {"drop", "re_role"}
                else [current_evidence_id]
            ),
        }
        for operation in operation_audit
    )

    transitioned = AllEvidenceFusionRunner._transition_adaptive_registry(
        before=before,
        after_specs=after_specs,
        operation_audit=materialized_operations,
        evidence_family_by_id=family_by_id,
    )

    observed = _support_by_name(transitioned)
    for name, atom_indices in expected_support.items():
        assert observed[name] == tuple(catalog.atoms[index].evidence_id for index in atom_indices)
    assert set(observed) == {str(spec["name"]) for spec in after_specs}
    for item in transitioned:
        expected_families = {
            family_by_id[evidence_id] for evidence_id in item.supporting_evidence_ids
        }
        assert set(item.source_families) == expected_families


def test_re_role_ignores_valid_new_citations_and_preserves_historical_support() -> None:
    catalog = _catalog()
    before, family_by_id = _initial_registry(catalog)

    transitioned = AllEvidenceFusionRunner._transition_adaptive_registry(
        before=before,
        after_specs=(
            _continuous("alpha_measure", roles=("effect_modifier",)),
            _continuous("beta_measure"),
        ),
        operation_audit=(
            {
                "action": "re_role",
                "target_names": ["alpha_measure"],
                "contract": _continuous("alpha_measure", roles=("effect_modifier",)),
                "supporting_evidence_ids": [catalog.atoms[2].evidence_id],
            },
        ),
        evidence_family_by_id=family_by_id,
    )

    assert _support_by_name(transitioned)["alpha_measure"] == (catalog.atoms[0].evidence_id,)


def test_transition_adaptive_registry_fails_closed_on_unknown_current_support() -> None:
    catalog = _catalog()
    before, family_by_id = _initial_registry(catalog)

    with pytest.raises(ValueError, match="cites unavailable provenance"):
        AllEvidenceFusionRunner._transition_adaptive_registry(
            before=before,
            after_specs=(
                _continuous("alpha_measure"),
                _continuous("beta_measure"),
                _continuous("gamma_measure"),
            ),
            operation_audit=(
                {
                    "adaptive_operation": "add",
                    "target_names": ["gamma_measure"],
                    "contract": _continuous("gamma_measure"),
                    "supporting_evidence_ids": ["evidence_unavailable"],
                },
            ),
            evidence_family_by_id=family_by_id,
        )


def test_adaptive_diagnostics_maps_every_kind_and_redacts_historical_targets() -> None:
    catalog = _catalog()
    registry, _family_by_id = _initial_registry(catalog)
    diagnostics = [
        {
            "diagnostic_id": "diagnostic_0001",
            "kind": "feature_quality",
            "feature_name": "alpha_measure",
            "missingness_rate": 0.25,
            "provider_sha256": "a" * 64,
        },
        {
            "diagnostic_id": "diagnostic_0002",
            "kind": "extraction_text_grounding",
            "contract_name": "alpha_measure",
            "validity_rate": 0.75,
        },
        {
            "diagnostic_id": "diagnostic_0003",
            "kind": "redundancy",
            "feature_names": ["alpha_measure", "beta_measure"],
            "association_score": 0.81,
        },
        {
            "diagnostic_id": "diagnostic_0004",
            "kind": "nested_observable_causal_quality",
            "nuisance_loss": 0.43,
        },
        {
            "diagnostic_id": "diagnostic_0005",
            "kind": "contract_ablation",
            "contract_name": "beta_measure",
            "importance_delta": 0.09,
        },
        {
            "diagnostic_id": "diagnostic_0006",
            "kind": "prior_gate_feedback",
            "prior_operations": [{"target_names": ["removed_measure", "alpha_measure"]}],
            "candidate_score": 0.55,
            "gate_score": 0.99,
        },
        {
            "diagnostic_id": "diagnostic_0007",
            "kind": "candidate_quality_retry_feedback",
            "failed_contract_names": ["removed_measure"],
            "failure_count": 1,
        },
        {
            "diagnostic_id": "diagnostic_0008",
            "kind": "retained_registry_ontology_retry_feedback",
            "failed_contract_names": ["removed_measure"],
            "ontology_mismatched_contract_names": [
                "older_measure",
                "beta_measure",
            ],
            "failure_count": 2,
        },
        {
            "diagnostic_id": "diagnostic_0009",
            "kind": "review_response_validation_retry_feedback",
            "failed_contract_names": ["older_measure"],
            "failure_count": 1,
        },
    ]

    adapted, audit = AllEvidenceFusionRunner._adaptive_diagnostics(
        diagnostics,
        current_registry=registry,
    )

    assert [item.diagnostic_kind for item in adapted] == [
        "extraction_missingness",
        "extraction_validity",
        "redundancy",
        "nuisance_residual",
        "heterogeneity",
        "source_preservation",
        "extraction_validity",
        "extraction_validity",
        "extraction_validity",
    ]
    by_id = {item.diagnostic_id: item for item in adapted}
    assert by_id["diagnostic_0006"].affected_features == ("alpha_measure",)
    assert by_id["diagnostic_0007"].affected_features == ()
    assert by_id["diagnostic_0008"].affected_features == ("beta_measure",)
    assert by_id["diagnostic_0009"].affected_features == ()
    assert by_id["diagnostic_0001"].aggregate_metrics == {"missingness_rate": 0.25}
    assert by_id["diagnostic_0006"].aggregate_metrics == {"candidate_score": 0.55}
    assert audit["input_diagnostic_count"] == 9
    assert audit["adapted_diagnostic_count"] == 9
    assert audit["excluded_historical_target_count"] == 5
    assert audit["excluded_historical_targets_by_diagnostic"] == {
        "diagnostic_0006": ["removed_measure"],
        "diagnostic_0007": ["removed_measure"],
        "diagnostic_0008": ["older_measure", "removed_measure"],
        "diagnostic_0009": ["older_measure"],
    }
    assert audit["unknown_current_diagnostic_targets_fail_closed"] is True
    assert audit["model_context_contains_excluded_historical_names"] is False
    assert audit["metric_coverage_proof_count"] == 9
    assert audit["total_eligible_metric_count"] == audit["total_emitted_metric_count"]
    assert audit["every_eligible_metric_emitted_once"] is True
    runner_module._validate_adaptive_diagnostic_adapter_audit(audit)
    serialized_model_context = canonical_json([item.as_prompt_item() for item in adapted])
    assert "removed_measure" not in serialized_model_context
    assert "older_measure" not in serialized_model_context
    assert "provider_sha256" not in serialized_model_context
    assert "gate_score" not in serialized_model_context


def test_adaptive_diagnostics_preserves_more_than_32_metrics_and_long_distinct_keys() -> None:
    catalog = _catalog()
    registry, _family_by_id = _initial_registry(catalog)
    shared_prefix = "score_" + ("clinically_meaningful_component_" * 5)
    long_left = f"{shared_prefix}left"
    long_right = f"{shared_prefix}right"
    assert len(long_left) > 96
    assert long_left[:96] == long_right[:96]
    metric_values = {f"score_component_{index:03d}": float(index) for index in range(40)}
    metric_values[long_left] = 101.0
    metric_values[long_right] = 202.0

    adapted, audit = AllEvidenceFusionRunner._adaptive_diagnostics(
        [
            {
                "diagnostic_id": "diagnostic_0001",
                "kind": "feature_quality",
                "feature_name": "alpha_measure",
                **metric_values,
            }
        ],
        current_registry=registry,
    )

    metrics = dict(adapted[0].aggregate_metrics)
    assert len(metrics) == 42
    assert metrics[long_left] == 101.0
    assert metrics[long_right] == 202.0
    assert set(metrics) == set(metric_values)
    proof = audit["metric_coverage_proofs"][0]
    assert proof["eligible_metric_count"] == 42
    assert proof["emitted_metric_count"] == 42
    assert proof["ordered_metric_keys"] == sorted(metric_values)
    assert audit["total_eligible_metric_count"] == 42
    assert audit["total_emitted_metric_count"] == 42


def test_adaptive_diagnostic_path_encoding_keeps_flat_and_nested_metrics_distinct() -> None:
    catalog = _catalog()
    registry, _family_by_id = _initial_registry(catalog)

    adapted, audit = AllEvidenceFusionRunner._adaptive_diagnostics(
        [
            {
                "diagnostic_id": "diagnostic_0001",
                "kind": "feature_quality",
                "feature_name": "alpha_measure",
                "quality_score": 1.0,
                "quality": {"score": 2.0},
                "Score_metric_count": 3,
                "score_metric_count": 4,
            }
        ],
        current_registry=registry,
    )

    assert adapted[0].aggregate_metrics == {
        runner_module._encode_adaptive_metric_path_segment("Score_metric_count"): 3,
        "quality.score": 2.0,
        "quality_score": 1.0,
        "score_metric_count": 4,
    }
    runner_module._validate_adaptive_diagnostic_adapter_audit(audit)
    runner_module._validate_adaptive_diagnostic_adapter_audit(
        json.loads(canonical_json(audit))
    )


def test_adaptive_diagnostic_metric_coverage_proof_rejects_duplicate_and_tampering() -> None:
    audit = _diagnostic_adapter_audit()
    proof = audit["metric_coverage_proofs"][0]

    duplicate = json.loads(canonical_json(proof))
    duplicate["ordered_metric_keys"].append(duplicate["ordered_metric_keys"][0])
    with pytest.raises(ValueError, match="contain duplicates"):
        runner_module._validate_adaptive_diagnostic_metric_coverage_proof(duplicate)

    omitted = json.loads(canonical_json(proof))
    omitted["aggregate_metrics"] = {}
    with pytest.raises(ValueError, match="key order is inconsistent"):
        runner_module._validate_adaptive_diagnostic_metric_coverage_proof(omitted)


def test_adaptive_diagnostics_rejects_metric_like_non_scalar_collections() -> None:
    catalog = _catalog()
    registry, _family_by_id = _initial_registry(catalog)

    with pytest.raises(ValueError, match="must be pre-aggregated to scalar"):
        AllEvidenceFusionRunner._adaptive_diagnostics(
            [
                {
                    "diagnostic_id": "diagnostic_0001",
                    "kind": "feature_quality",
                    "feature_name": "alpha_measure",
                    "quality_score_history": [0.1, 0.2],
                }
            ],
            current_registry=registry,
        )


def test_adaptive_diagnostics_rejects_unknown_target_for_current_diagnostic() -> None:
    catalog = _catalog()
    registry, _family_by_id = _initial_registry(catalog)

    with pytest.raises(ValueError, match="targets absent registry features"):
        AllEvidenceFusionRunner._adaptive_diagnostics(
            [
                {
                    "diagnostic_id": "diagnostic_0001",
                    "kind": "feature_quality",
                    "feature_name": "removed_measure",
                    "missingness_rate": 0.25,
                }
            ],
            current_registry=registry,
        )


def _authenticated_execution_artifact_body(
    *, request_sha256: str, outer_fold: int = 1
) -> dict[str, object]:
    proposal = {"operations": [], "converged": True}
    frozen_without_hash = {
        "schema_version": "adaptive_reconsideration_round_freeze_test_v1",
        "proposal_sha256": runner_module._content_sha256(proposal),
        "proposal": proposal,
    }
    frozen = {
        **frozen_without_hash,
        "freeze_sha256": runner_module._content_sha256(frozen_without_hash),
    }
    specs = [_continuous("alpha_measure")]
    applied = {
        "specs": specs,
        "reextract_specs": [],
        "removed_names": [],
        "added_names": [],
        "extraction_changed_names": [],
        "role_only_changed_names": [],
        "operation_audit": [],
    }
    executable_without_hash = {
        "schema_version": "adaptive_executable_bridge_test_v1",
        "proposal_freeze_sha256": frozen["freeze_sha256"],
        "applied_specs_sha256": runner_module._content_sha256(specs),
        "applied": applied,
    }
    executable = {
        **executable_without_hash,
        "executable_freeze_sha256": runner_module._content_sha256(executable_without_hash),
    }
    audit = {"proposal_and_executable_frozen_before_gate": True}
    execution = {
        "schema_version": "authenticated_adaptive_execution_test_v1",
        "frozen_round": frozen,
        "executable_revision": executable,
        "dossier_sha256s": ["c" * 64],
        "lookback": {"lookback_sha256": "d" * 64},
        "runner_identity_sha256": "e" * 64,
        "cache_identity_sha256": "f" * 64,
        "audit": audit,
    }
    execution_identity = {
        "schema_version": execution["schema_version"],
        "freeze_sha256": frozen["freeze_sha256"],
        "executable_freeze_sha256": executable["executable_freeze_sha256"],
        "dossier_sha256s": execution["dossier_sha256s"],
        "lookback_sha256": execution["lookback"]["lookback_sha256"],
        "runner_identity_sha256": execution["runner_identity_sha256"],
        "cache_identity_sha256": execution["cache_identity_sha256"],
        "audit": audit,
    }
    execution["execution_sha256"] = runner_module._content_sha256(execution_identity)
    return {
        "outer_fold": outer_fold,
        "review_round": 2,
        "review_attempt": 1,
        "request_sha256": request_sha256,
        "diagnostic_adapter_audit": _diagnostic_adapter_audit(),
        "authenticated_execution": execution,
        "proposal_frozen_before_executable_bridge": True,
        "executable_revision_frozen_before_gate": True,
        "complete_catalog_sent_to_legacy_review_agent": False,
        "raw_response_persisted": False,
        "raw_reasoning_persisted": False,
        "gate_accessed": False,
    }


def test_post_revalidation_adaptive_artifact_consistency_loader_preserves_frozen_application(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authenticated_adaptive_hierarchy.json"
    request_sha256 = "9" * 64
    body = _authenticated_execution_artifact_body(request_sha256=request_sha256)
    runner_module._write_immutable_json(
        path,
        body,
        schema=runner_module.ADAPTIVE_HIERARCHICAL_REVIEW_EXECUTION_SCHEMA_VERSION,
    )

    proposal, applied, execution = runner_module._load_request_bound_adaptive_execution(
        path,
        outer_fold=1,
        request_sha256=request_sha256,
        review_round=2,
        review_attempt=1,
    )

    assert proposal == {"operations": [], "converged": True}
    assert applied.specs == (_continuous("alpha_measure"),)
    assert applied.operation_audit == ()
    assert execution["frozen_round"]["freeze_sha256"] == (
        execution["executable_revision"]["proposal_freeze_sha256"]
    )


def test_adaptive_execution_loader_rejects_metric_audit_different_from_fresh_extraction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authenticated_adaptive_hierarchy.json"
    request_sha256 = "9" * 64
    body = _authenticated_execution_artifact_body(request_sha256=request_sha256)
    runner_module._write_immutable_json(
        path,
        body,
        schema=runner_module.ADAPTIVE_HIERARCHICAL_REVIEW_EXECUTION_SCHEMA_VERSION,
    )

    with pytest.raises(RuntimeError, match="differs from fresh extraction"):
        runner_module._load_request_bound_adaptive_execution(
            path,
            outer_fold=1,
            request_sha256=request_sha256,
            review_round=2,
            review_attempt=1,
            expected_diagnostic_adapter_audit=_diagnostic_adapter_audit(score=0.75),
        )


def test_adaptive_execution_precedes_outer_artifact_comparison_and_bypasses_legacy_validation() -> (
    None
):
    source = textwrap.dedent(inspect.getsource(AllEvidenceFusionRunner._run_post_extraction_review))
    tree = ast.parse(source)
    execute_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute_authenticated"
    ]
    consistency_load_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_load_request_bound_adaptive_execution"
    ]

    assert len(execute_lines) == 1
    assert len(consistency_load_lines) == 1
    assert execute_lines[0] < consistency_load_lines[0]

    adaptive_apply_branches = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "adaptive_attempt"
            and node.orelse
        ):
            continue
        adaptive_branch = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        legacy_branch = ast.dump(ast.Module(body=node.orelse, type_ignores=[]))
        if (
            "adaptive_applied" in adaptive_branch
            and "validate_post_extraction_review_response" in legacy_branch
        ):
            adaptive_apply_branches.append((adaptive_branch, legacy_branch))

    assert len(adaptive_apply_branches) == 1
    adaptive_branch, legacy_branch = adaptive_apply_branches[0]
    assert "validate_post_extraction_review_response" not in adaptive_branch
    assert "adaptive_applied" not in legacy_branch


def test_authenticated_adaptive_artifact_is_request_bound_and_rejects_unsafe_flags(
    tmp_path: Path,
) -> None:
    request_sha256 = "9" * 64
    path = tmp_path / "request_bound.json"
    runner_module._write_immutable_json(
        path,
        _authenticated_execution_artifact_body(request_sha256=request_sha256),
        schema=runner_module.ADAPTIVE_HIERARCHICAL_REVIEW_EXECUTION_SCHEMA_VERSION,
    )

    with pytest.raises(RuntimeError, match="belongs to another request"):
        runner_module._load_request_bound_adaptive_execution(
            path,
            outer_fold=1,
            request_sha256="8" * 64,
            review_round=2,
            review_attempt=1,
        )

    unsafe_path = tmp_path / "unsafe.json"
    unsafe = _authenticated_execution_artifact_body(request_sha256=request_sha256)
    unsafe["gate_accessed"] = True
    runner_module._write_immutable_json(
        unsafe_path,
        unsafe,
        schema=runner_module.ADAPTIVE_HIERARCHICAL_REVIEW_EXECUTION_SCHEMA_VERSION,
    )
    with pytest.raises(RuntimeError, match="unsafe flag gate_accessed"):
        runner_module._load_request_bound_adaptive_execution(
            unsafe_path,
            outer_fold=1,
            request_sha256=request_sha256,
            review_round=2,
            review_attempt=1,
        )


def test_authenticated_adaptive_artifact_rejects_tampered_nested_freeze(
    tmp_path: Path,
) -> None:
    request_sha256 = "9" * 64
    body = _authenticated_execution_artifact_body(request_sha256=request_sha256)
    body["authenticated_execution"]["frozen_round"]["proposal"]["converged"] = False
    path = tmp_path / "tampered_nested_freeze.json"
    runner_module._write_immutable_json(
        path,
        body,
        schema=runner_module.ADAPTIVE_HIERARCHICAL_REVIEW_EXECUTION_SCHEMA_VERSION,
    )

    with pytest.raises(RuntimeError, match="adaptive proposal freeze hash is invalid"):
        runner_module._load_request_bound_adaptive_execution(
            path,
            outer_fold=1,
            request_sha256=request_sha256,
            review_round=2,
            review_attempt=1,
        )
