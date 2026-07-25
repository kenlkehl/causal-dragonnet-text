from __future__ import annotations

from dataclasses import dataclass

import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    AS_DOCUMENTED_UNIT,
    MECHANICAL_MENTION_CATEGORIES,
    ArchitectureDossier,
    RoleRoutingResult,
    TFIDF_SEMANTIC_RETRIEVAL,
    canonical_json,
    content_sha256,
)
from oci.inference.all_evidence_fusion import CandidateContract
from oci.inference.hierarchical_all_architecture_discovery import (
    COMPLETED_HIERARCHICAL_DISCOVERY_VERSION,
    EXTRACTION_DEFINITION_JOB,
    CompletedHierarchicalDiscovery,
    DiscoveryExecutionLedger,
    DiscoveryJobSettings,
    DiscoveryJsonJob,
    IntegratedCanonicalFeature,
    RoutedIntegratedFeature,
    ValidatedDiscoveryJobResult,
)
from oci.inference.hierarchical_discovery_compiler import (
    DOWNSTREAM_ADJUSTMENT_SLOT_AUDIT,
    MODELED_DISPOSITION,
    NON_MODEL_EXTRACTION_ONLY,
    NON_MODEL_ROLELESS,
    NON_MODEL_TREATMENT_AND_EXTRACTION_ONLY,
    NON_MODEL_TREATMENT_ONLY,
    HierarchicalDiscoveryCompiler,
    compile_hierarchical_discovery,
)
from oci.inference.hierarchical_discovery_response_contract import (
    attach_hierarchical_discovery_response_contract,
)


@dataclass(frozen=True)
class _FeatureCase:
    name: str
    definition: dict
    adjustment_roles: tuple[str, ...] = ()
    effect_modifier: bool = False
    treatment_support: bool = False
    extraction_support: bool = False
    source_families: tuple[str, ...] = ("bow_nuisance",)


def _continuous_definition(name: str) -> dict:
    return {
        "feature_name": name,
        "measurement": f"Extract the documented {name.replace('_', ' ')} measurement.",
        "representation": {"kind": "continuous", "unit": "unitless", "categories": []},
        "aliases": [],
        "distinguish_from": [],
        "missing_or_ambiguous": "Return null when absent or ambiguous.",
        "supporting_evidence_ids": [f"evidence_{name}"],
    }


def _categorical_definition(name: str, categories: list[str]) -> dict:
    return {
        "feature_name": name,
        "measurement": f"Extract the documented {name.replace('_', ' ')} category.",
        "representation": {"kind": "categorical", "unit": "", "categories": categories},
        "aliases": [],
        "distinguish_from": [],
        "missing_or_ambiguous": "Return null when absent or ambiguous.",
        "supporting_evidence_ids": [f"evidence_{name}"],
    }


def _completed(cases: tuple[_FeatureCase, ...]) -> CompletedHierarchicalDiscovery:
    dossiers = tuple(
        ArchitectureDossier(
            source_family=family,
            catalog_sha256=f"{index + 1:064x}",
            catalog_evidence_ids=(),
            coverage_disposition_ids=(),
            coverage_audit_sha256=f"{index + 101:064x}",
            architecture_candidates=(),
            direct_numerical_manifest_sha256=f"{index + 201:064x}",
            direct_numerical_signal_count=0,
            direct_numerical_zero_reason="No direct signal in this compiler fixture.",
        )
        for index, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    )
    routed = tuple(
        RoutedIntegratedFeature(
            feature=IntegratedCanonicalFeature(
                canonical_name=case.name,
                description=f"Integrated definition for {case.name}.",
                member_candidate_ids=(f"candidate_{case.name}",),
                supporting_evidence_ids=(f"evidence_{case.name}",),
                source_families=case.source_families,
                value_shape_hypothesis="ambiguous",
                unresolved_ambiguity="Resolved by the authenticated extraction definition.",
            ),
            role_routing=RoleRoutingResult(
                observable_axes=(),
                adjustment_roles=case.adjustment_roles,
                effect_modifier=case.effect_modifier,
                treatment_prediction_support=case.treatment_support,
                extraction_definition_support=case.extraction_support,
                applied_rules=(),
            ),
        )
        for case in cases
    )
    definitions = {case.name: case.definition for case in cases}
    jobs = []
    results = []
    for case in cases:
        request = attach_hierarchical_discovery_response_contract(
            job_kind=EXTRACTION_DEFINITION_JOB,
            request={
                "job": "define_one_extraction_feature",
                "canonical_name": case.name,
                "value_shape_hypothesis": (
                    "categorical"
                    if case.definition["representation"]["kind"] == "categorical"
                    else "continuous"
                ),
                "supporting_evidence_ids": [f"evidence_{case.name}"],
            },
        )
        job = DiscoveryJsonJob.create(
            job_kind=EXTRACTION_DEFINITION_JOB,
            scope=case.name,
            dependencies=(),
            settings=DiscoveryJobSettings.extraction(),
            messages=(
                {"role": "system", "content": "Return one extraction definition as JSON."},
                {"role": "user", "content": canonical_json(request)},
            ),
            input_bindings={"canonical_name": case.name},
        )
        jobs.append(job)
        results.append(
            ValidatedDiscoveryJobResult.create(
                job=job,
                validated_response=case.definition,
            )
        )
    execution = DiscoveryExecutionLedger.build(jobs=jobs, results=results)
    planner = {}
    integration = {
        "candidate_dispositions": [
            {
                "candidate_id": f"candidate_{case.name}",
                "decision": "accept",
            }
            for case in cases
        ]
    }
    rejection = {}
    identity = {
        "schema_version": COMPLETED_HIERARCHICAL_DISCOVERY_VERSION,
        "precommit_sha256": "a" * 64,
        "dossiers": [row.as_authenticated_dict() for row in dossiers],
        "planner_response": planner,
        "requested_lookback_evidence_ids": [],
        "integration_response": integration,
        "rejected_candidate_ids": [],
        "rejection_critic_response": rejection,
        "routed_features": [row.as_dict() for row in routed],
        "extraction_definitions": definitions,
        "execution_sha256": execution.execution_sha256,
    }
    return CompletedHierarchicalDiscovery(
        precommit_sha256="a" * 64,
        dossiers=dossiers,
        routed_features=routed,
        rejected_candidate_ids=(),
        requested_lookback_evidence_ids=(),
        extraction_job_ids=tuple(job.job_id for job in jobs),
        execution_ledger=execution,
        completion_sha256=content_sha256(identity),
        _planner_response_json=canonical_json(planner),
        _integration_response_json=canonical_json(integration),
        _rejection_critic_response_json=canonical_json(rejection),
        _extraction_definitions_json=canonical_json(definitions),
    )


def test_happy_path_compiles_mixed_roles_and_preserves_names_and_families():
    cases = (
        _FeatureCase(
            name="baseline_ratio",
            definition=_continuous_definition("baseline_ratio"),
            adjustment_roles=("prognostic_adjustment",),
            effect_modifier=True,
            source_families=(TFIDF_SEMANTIC_RETRIEVAL, "bow_r_loss"),
        ),
        _FeatureCase(
            name="biomarker_status",
            definition=_categorical_definition("biomarker_status", ["negative", "positive"]),
            effect_modifier=True,
            source_families=("matched_pair_uplift",),
        ),
    )
    completed = _completed(cases)
    registry = compile_hierarchical_discovery(completed, max_candidates=2)

    assert registry.source_completion_sha256 == completed.completion_sha256
    assert registry.registry_sha256 == content_sha256(
        {key: value for key, value in registry.as_dict().items() if key != "registry_sha256"}
    )
    assert all(isinstance(contract, CandidateContract) for contract in registry.contracts)
    assert [spec["name"] for spec in registry.specs] == [
        "baseline_ratio",
        "biomarker_status",
    ]
    assert registry.specs[0]["roles"] == ["confounder", "effect_modifier"]
    assert registry.specs[1]["roles"] == ["effect_modifier"]
    assert registry.contracts[0].source_families == (
        TFIDF_SEMANTIC_RETRIEVAL,
        "bow_r_loss",
    )
    audit = registry.disposition_audit
    assert [row["disposition"] for row in audit] == [
        MODELED_DISPOSITION,
        MODELED_DISPOSITION,
    ]
    assert audit[0]["adjustment_slot_audit"] == DOWNSTREAM_ADJUSTMENT_SLOT_AUDIT
    assert "not a causal-confounder claim" in audit[0]["adjustment_slot_audit"]
    assert audit[1]["adjustment_slot_audit"] == ""

    detached = registry.specs
    detached[0]["roles"].clear()
    assert registry.specs[0]["roles"] == ["confounder", "effect_modifier"]


def test_unresolved_representation_fails_closed():
    definition = _continuous_definition("unresolved_measure")
    definition["representation"] = {"kind": "unresolved", "unit": "", "categories": []}
    completed = _completed(
        (
            _FeatureCase(
                name="unresolved_measure",
                definition=definition,
                effect_modifier=True,
            ),
        )
    )
    with pytest.raises(ValueError, match="unresolved extraction representation"):
        compile_hierarchical_discovery(completed)


def test_reserved_extraction_mechanics_remain_explicit_in_compiled_contracts():
    continuous = _continuous_definition("documented_ratio")
    continuous["representation"]["unit"] = AS_DOCUMENTED_UNIT
    categorical = _categorical_definition(
        "supported_language_observed",
        list(MECHANICAL_MENTION_CATEGORIES),
    )
    completed = _completed(
        (
            _FeatureCase(
                name="documented_ratio",
                definition=continuous,
                effect_modifier=True,
            ),
            _FeatureCase(
                name="supported_language_observed",
                definition=categorical,
                effect_modifier=True,
            ),
        )
    )

    registry = compile_hierarchical_discovery(completed)

    assert "not a clinical unit assertion" in registry.specs[0]["description"]
    assert "not a clinical status ontology" in registry.specs[1]["description"]
    assert (
        registry.compiler_identity["representation_policy"]["extraction_vocabulary_grounding"][
            "mechanical_encodings_are_clinical_ontology"
        ]
        is False
    )


def test_compiler_rechecks_complete_extraction_support_set():
    definition = _continuous_definition("supported_measure")
    definition["supporting_evidence_ids"] = ["evidence_different_measure"]
    completed = _completed(
        (
            _FeatureCase(
                name="supported_measure",
                definition=definition,
                effect_modifier=True,
            ),
        )
    )

    with pytest.raises(ValueError, match="preserve complete feature support"):
        compile_hierarchical_discovery(completed)


@pytest.mark.parametrize(
    "categories",
    (
        ["positive"],
        ["high-risk", "high risk"],
        ["category 1", "positive"],
        "positive,negative",
    ),
)
def test_candidate_contract_rejects_invalid_categorical_values(categories):
    completed = _completed(
        (
            _FeatureCase(
                name="invalid_status",
                definition=_categorical_definition("invalid_status", categories),
                adjustment_roles=("prognostic_adjustment",),
            ),
        )
    )
    with pytest.raises((TypeError, ValueError), match="categories"):
        compile_hierarchical_discovery(completed)


def test_compiler_preserves_more_than_eight_categorical_values_losslessly():
    categories = [f"state_{index}" for index in range(12)]
    completed = _completed(
        (
            _FeatureCase(
                name="detailed_biomarker_status",
                definition=_categorical_definition(
                    "detailed_biomarker_status",
                    categories,
                ),
                effect_modifier=True,
            ),
        )
    )

    registry = compile_hierarchical_discovery(completed, max_candidates=1)

    assert registry.specs[0]["categories"] == categories
    assert registry.contracts[0].extraction_spec["categories"] == categories
    assert (
        registry.compiler_identity["representation_policy"]["categorical_categories"]
        == "CandidateContract concrete distinct nonempty validation with at least "
        "two values and no compiler category-count cap"
    )


@pytest.mark.parametrize(
    ("treatment", "extraction", "expected"),
    (
        (True, False, NON_MODEL_TREATMENT_ONLY),
        (False, True, NON_MODEL_EXTRACTION_ONLY),
        (True, True, NON_MODEL_TREATMENT_AND_EXTRACTION_ONLY),
        (False, False, NON_MODEL_ROLELESS),
    ),
)
def test_roleless_features_receive_explicit_non_model_dispositions(treatment, extraction, expected):
    completed = _completed(
        (
            _FeatureCase(
                name="support_only_measure",
                definition=_continuous_definition("support_only_measure"),
                treatment_support=treatment,
                extraction_support=extraction,
            ),
        )
    )
    registry = compile_hierarchical_discovery(completed, max_candidates=0)

    assert registry.specs == []
    assert registry.contracts == ()
    assert registry.disposition_audit[0]["disposition"] == expected
    assert registry.disposition_audit[0]["modeled"] is False
    assert registry.disposition_audit[0]["legacy_roles"] == []
    assert registry.disposition_audit[0]["reason"]


def test_candidate_cap_fails_instead_of_truncating():
    completed = _completed(
        (
            _FeatureCase(
                name="first_modifier",
                definition=_continuous_definition("first_modifier"),
                effect_modifier=True,
            ),
            _FeatureCase(
                name="second_modifier",
                definition=_continuous_definition("second_modifier"),
                effect_modifier=True,
            ),
        )
    )
    with pytest.raises(ValueError, match="refusing to truncate 2 candidates to 1"):
        HierarchicalDiscoveryCompiler(max_candidates=1).compile(completed)


def test_source_completion_records_dispositions_and_registry_detect_tampering():
    completed = _completed(
        (
            _FeatureCase(
                name="stable_modifier",
                definition=_continuous_definition("stable_modifier"),
                effect_modifier=True,
            ),
        )
    )
    registry = compile_hierarchical_discovery(completed)

    original_spec = registry.modeled_candidates[0].extraction_spec
    original_spec["description"] += " Changed after compilation."
    object.__setattr__(
        registry.modeled_candidates[0],
        "_extraction_spec_json",
        canonical_json(original_spec),
    )
    with pytest.raises(ValueError, match="does not authenticate the spec"):
        registry.validate_authentication()

    fresh_registry = compile_hierarchical_discovery(completed)
    object.__setattr__(fresh_registry.dispositions[0], "reason", "tampered reason")
    with pytest.raises(ValueError, match="does not authenticate the disposition"):
        fresh_registry.validate_authentication()

    object.__setattr__(completed, "completion_sha256", "0" * 64)
    with pytest.raises(ValueError, match="does not authenticate completed discovery"):
        compile_hierarchical_discovery(completed)
