from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    AS_DOCUMENTED_UNIT,
    BOW_NUISANCE,
    DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT,
    DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST,
    DIRECT_UPSTREAM_NUMERICAL_CHANNEL,
    EMBEDDING_CLUSTERED,
    EXTRACTION_SUPPORT_AXIS,
    HETEROGENEITY_AXIS,
    MECHANICAL_MENTION_CATEGORIES,
    OUTCOME_AXIS,
    TREATMENT_AXIS,
    ArchitectureDossier,
    DiscoveryCandidate,
    DiscoveryEvidenceItem,
    ExtractionDefinitionRequest,
    bounded_candidate_relation_pages,
    candidate_definition_fold_batches,
    compile_complete_link_candidate_groups,
    consolidate_candidate_context,
    cross_architecture_planner_context,
    render_cross_architecture_planner_messages,
    render_interpret_evidence_chunk_messages,
    revalidate_normalized_consolidation_response,
    revalidate_normalized_extraction_definition_response,
    resolve_raw_evidence_lookback,
    route_concept_roles,
    validate_consolidation_response,
    validate_candidate_relation_page_response,
    validate_cross_architecture_planner_response,
    validate_extraction_definition_response,
    validate_interpret_evidence_chunk_response,
    canonical_json,
)
from oci.inference.hierarchical_discovery_response_contract import (
    LEGACY_HIERARCHY_WIRE_BUDGET,
)


def test_bounded_pair_pages_cover_every_unordered_pair_once_and_compile_complete_link():
    candidates = tuple(_candidate(BOW_NUISANCE, index) for index in range(10))
    pages = bounded_candidate_relation_pages(candidates)
    assert sum(page["pair_count"] for page in pages) == 45
    assert max(page["pair_count"] for page in pages) == 7

    same_pairs = {
        frozenset(("candidate_000", "candidate_001")),
        frozenset(("candidate_001", "candidate_002")),
    }
    explicit_distinct = frozenset(("candidate_000", "candidate_002"))
    normalized_pages = []
    for page in pages:
        anchor = page["anchor_candidate_id"]
        response = {"comparisons": {}}
        for peer in page["peer_candidate_ids"]:
            pair = frozenset((anchor, peer))
            response["comparisons"][peer] = {
                "relation": "same_construct" if pair in same_pairs else "distinct",
                "reason": "bounded pair judgment",
            }
        normalized_pages.append(
            validate_candidate_relation_page_response(
                response,
                anchor_candidate_id=anchor,
                peer_candidate_ids=page["peer_candidate_ids"],
            )
        )
    assert explicit_distinct not in same_pairs
    compiled = compile_complete_link_candidate_groups(
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        relation_pages=normalized_pages,
    )
    groups = [row["member_candidate_ids"] for row in compiled["groups"]]
    assert groups[:2] == [
        ["candidate_000", "candidate_001"],
        ["candidate_002"],
    ]
    assert compiled["pair_relation_audit"]["cross_group_same_construct_edges"] == [
        {
            "left_candidate_id": "candidate_001",
            "right_candidate_id": "candidate_002",
            "reason": "bounded pair judgment",
        }
    ]


def test_definition_fold_batches_terminate_without_dropping_group_members():
    members = tuple(f"candidate_{index:03d}" for index in range(20))
    batches = candidate_definition_fold_batches(
        group_id="candidate_group_test",
        member_candidate_ids=members,
    )
    assert [len(batch["member_candidate_ids"]) for batch in batches] == [8, 7, 5]
    assert [batch["uses_prior_accumulator"] for batch in batches] == [False, True, True]
    assert tuple(member for batch in batches for member in batch["member_candidate_ids"]) == members


def _evidence(
    family: str,
    index: int,
    *,
    axes: tuple[str, ...] = (OUTCOME_AXIS,),
    raw_marker: str | None = None,
) -> DiscoveryEvidenceItem:
    return DiscoveryEvidenceItem(
        evidence_id=f"evidence_{index:03d}",
        source_family=family,
        observable_axes=axes,
        content={"terms": [raw_marker or f"clinical term {index}"]},
    )


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"])
def test_canonical_json_rejects_values_that_cannot_be_utf8_encoded(surrogate):
    with pytest.raises(ValueError, match="valid UTF-8"):
        canonical_json({"arbitrary_cohort_value": surrogate})


def _candidate(
    family: str,
    index: int,
    *,
    evidence_id: str | None = None,
    shape: str = "continuous",
) -> DiscoveryCandidate:
    return DiscoveryCandidate(
        candidate_id=f"candidate_{index:03d}",
        feature_name=f"patient_measure_{index}",
        description=f"Patient-level clinical measurement {index}.",
        supporting_evidence_ids=(evidence_id or f"evidence_{index:03d}",),
        source_families=(family,),
        value_shape_hypothesis=shape,
    )


def _all_dossiers() -> tuple[tuple[ArchitectureDossier, ...], dict[str, DiscoveryEvidenceItem]]:
    dossiers: list[ArchitectureDossier] = []
    catalog: dict[str, DiscoveryEvidenceItem] = {}
    for index, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES, start=1):
        item = _evidence(family, index, raw_marker=f"secret raw clue {index}")
        candidate = _candidate(family, index)
        catalog[item.evidence_id] = item
        dossiers.append(
            ArchitectureDossier(
                source_family=family,
                catalog_sha256=f"{index:064x}",
                catalog_evidence_ids=(item.evidence_id,),
                coverage_disposition_ids=(item.evidence_id,),
                coverage_audit_sha256=f"{index + 100:064x}",
                architecture_candidates=(candidate,),
                direct_numerical_manifest_sha256="f" * 64,
                direct_numerical_signal_count=1,
            )
        )
    return tuple(dossiers), catalog


def _planner_response(
    dossiers: tuple[ArchitectureDossier, ...], *, maximum_lookback_ids: int = 4
) -> dict:
    candidates = [
        candidate for dossier in dossiers for candidate in dossier.architecture_candidates
    ]
    group_slots = [f"planner_group_slot_{index:03d}" for index in range(1, len(candidates) + 1)]
    evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for candidate in candidates
            for evidence_id in candidate.supporting_evidence_ids
        )
    )
    lookback_slots = [
        f"planner_lookback_slot_{index:03d}"
        for index in range(1, min(maximum_lookback_ids, len(evidence_ids)) + 1)
    ]
    return {
        "candidate_assignments": {
            candidate.candidate_id: {
                "group_slot": group_slots[index],
            }
            for index, candidate in enumerate(candidates)
        },
        "group_slot_definitions": {
            slot: {
                "provisional_name": candidates[index].feature_name,
                "reason": "Keep the architecture-level measurement distinct.",
            }
            for index, slot in enumerate(group_slots)
        },
        "lookback_slot_definitions": {
            slot: {
                "selection": "unused",
                "question": "Check whether compact evidence is sufficient.",
                "reason": "Use raw evidence only when the compact dossier is ambiguous.",
            }
            for slot in lookback_slots
        },
    }


def test_active_architecture_contract_is_complete_and_excludes_sparse_fallback():
    assert ACTIVE_STAGE1_CONCEPT_FAMILIES == (
        "bow_nuisance",
        "bow_r_loss",
        "htr_neural",
        "matched_pair_uplift",
        "embedding_whole_cohort",
        "embedding_clustered",
        "tfidf_semantic_retrieval_contrasts",
        "tfidf_topics",
        "tfidf_orphan_ngrams",
        "neural_query_moments",
    )
    assert "sparse_query_moments" not in ACTIVE_STAGE1_CONCEPT_FAMILIES


def test_interpretation_prompt_rejects_mixed_architectures():
    with pytest.raises(ValueError, match="exactly one architecture"):
        render_interpret_evidence_chunk_messages(
            family_explanation="Two architectures must not share a discovery pass.",
            evidence=(
                _evidence(BOW_NUISANCE, 1),
                _evidence(EMBEDDING_CLUSTERED, 2),
            ),
        )


def test_interpretation_validation_requires_every_evidence_disposition_and_exact_support():
    evidence = (
        _evidence(BOW_NUISANCE, 1),
        _evidence(BOW_NUISANCE, 2),
    )
    finding = {
        "feature_name": "baseline_age",
        "description": "Age measured for one patient at baseline.",
        "value_shape_hypothesis": "continuous",
        "unresolved_ambiguity": "",
    }
    response = {
        "evidence_dispositions": {
            "evidence_001": {
                "evidence_findings": [finding],
                "member_dispositions": {},
                "reason": "The clue explicitly names age.",
            },
            "evidence_002": {
                "evidence_findings": [finding],
                "member_dispositions": {},
                "reason": "The second clue also names age.",
            },
        },
    }
    normalized = validate_interpret_evidence_chunk_response(response, evidence=evidence)
    assert [row["evidence_id"] for row in normalized["evidence_dispositions"]] == [
        "evidence_001",
        "evidence_002",
    ]
    assert [row["supporting_evidence_ids"] for row in normalized["concepts"]] == [
        ["evidence_001"],
        ["evidence_002"],
    ]
    duplicate_groups = normalized["wire_normalization_audit"]["duplicate_feature_name_groups"]
    assert len(duplicate_groups) == 1
    assert duplicate_groups[0]["feature_name"] == "baseline_age"

    missing = deepcopy(response)
    missing["evidence_dispositions"].pop("evidence_002")
    with pytest.raises(ValueError, match="keys differ"):
        validate_interpret_evidence_chunk_response(missing, evidence=evidence)

    role_first = deepcopy(response)
    role_first["evidence_dispositions"]["evidence_001"]["evidence_findings"][0][
        "role"
    ] = "confounder"
    with pytest.raises(ValueError, match="keys differ"):
        validate_interpret_evidence_chunk_response(role_first, evidence=evidence)


def test_interpretation_requires_a_disposition_for_every_container_member():
    item = DiscoveryEvidenceItem(
        evidence_id="evidence_001",
        source_family=BOW_NUISANCE,
        observable_axes=(OUTCOME_AXIS,),
        member_ids=("member_001", "member_002"),
        content={
            "terms": [
                {"member_id": "member_001", "term": "baseline age"},
                {"member_id": "member_002", "term": "administrative header"},
            ]
        },
    )
    response = {
        "evidence_dispositions": {
            "evidence_001": {
                "evidence_findings": [],
                "member_dispositions": {
                    "member_001": {
                        "findings": [
                            {
                                "feature_name": "baseline_age",
                                "description": "Patient age measured at baseline.",
                                "value_shape_hypothesis": "continuous",
                                "unresolved_ambiguity": "",
                            }
                        ]
                    },
                    "member_002": {"findings": []},
                },
                "reason": "One term grounds age; the other is non-specific.",
            }
        },
    }
    normalized = validate_interpret_evidence_chunk_response(response, evidence=(item,))
    assert [
        row["member_id"] for row in normalized["evidence_dispositions"][0]["member_dispositions"]
    ] == ["member_001", "member_002"]
    incomplete = deepcopy(response)
    incomplete["evidence_dispositions"]["evidence_001"]["member_dispositions"].pop("member_002")
    with pytest.raises(ValueError, match="keys differ"):
        validate_interpret_evidence_chunk_response(incomplete, evidence=(item,))


def test_discovery_evidence_rejects_raw_numerical_and_identifier_payloads():
    with pytest.raises(ValueError, match="forbidden"):
        DiscoveryEvidenceItem(
            evidence_id="evidence_001",
            source_family=BOW_NUISANCE,
            observable_axes=(OUTCOME_AXIS,),
            content={"raw_vector": [0.1, 0.2]},
        )
    with pytest.raises(ValueError, match="forbidden"):
        DiscoveryEvidenceItem(
            evidence_id="evidence_001",
            source_family=BOW_NUISANCE,
            observable_axes=(OUTCOME_AXIS,),
            content={"row_ids": [1, 2]},
        )


def test_within_architecture_consolidation_preserves_candidates_evidence_and_family():
    candidates = (
        _candidate(BOW_NUISANCE, 1),
        DiscoveryCandidate(
            candidate_id="candidate_002",
            feature_name="age_at_baseline",
            description="Patient age at the baseline encounter.",
            supporting_evidence_ids=("evidence_002",),
            source_families=(BOW_NUISANCE,),
            value_shape_hypothesis="continuous",
        ),
    )
    context = consolidate_candidate_context(
        source_family=BOW_NUISANCE,
        candidates=candidates,
    )
    assert context["source_family"] == BOW_NUISANCE
    response = {
        "candidate_assignments": {
            "candidate_001": {
                "cluster_slot": "consolidation_slot_001",
                "reason": "Formatting alias.",
            },
            "candidate_002": {
                "cluster_slot": "consolidation_slot_001",
                "reason": "Formatting alias.",
            },
        },
        "slot_definitions": {
            "consolidation_slot_001": {
                "canonical_name": "baseline_age",
                "description": "Patient age at baseline.",
                "unresolved_ambiguity": "",
            },
            "consolidation_slot_002": {
                "canonical_name": "unused_consolidation_slot_002",
                "description": "Unused compiler-owned consolidation slot.",
                "unresolved_ambiguity": "",
            },
        },
    }
    normalized = validate_consolidation_response(
        response,
        source_family=BOW_NUISANCE,
        candidates=candidates,
    )
    assert normalized["canonical_concepts"][0]["member_candidate_ids"] == [
        "candidate_001",
        "candidate_002",
    ]
    assert normalized["canonical_concepts"][0]["supporting_evidence_ids"] == [
        "evidence_001",
        "evidence_002",
    ]
    assert normalized["canonical_concepts"][0]["source_families"] == [BOW_NUISANCE]

    dropped = deepcopy(response)
    dropped["candidate_assignments"].pop("candidate_002")
    with pytest.raises(ValueError, match="keys differ"):
        validate_consolidation_response(
            dropped,
            source_family=BOW_NUISANCE,
            candidates=candidates,
        )


def test_normalized_consolidation_revalidation_requires_configured_name_budget():
    candidates = (
        _candidate(BOW_NUISANCE, 1),
        _candidate(BOW_NUISANCE, 2),
    )
    proposed_name = "patient_treatment_response_measure"
    wire_response = {
        "candidate_assignments": {
            "candidate_001": {
                "cluster_slot": "consolidation_slot_001",
                "reason": "Keep the first measurement.",
            },
            "candidate_002": {
                "cluster_slot": "consolidation_slot_002",
                "reason": "Keep the second measurement distinct.",
            },
        },
        "slot_definitions": {
            slot: {
                "canonical_name": proposed_name,
                "description": f"Definition for {slot}.",
                "unresolved_ambiguity": "",
            }
            for slot in ("consolidation_slot_001", "consolidation_slot_002")
        },
    }
    wire_budget = replace(
        LEGACY_HIERARCHY_WIRE_BUDGET,
        max_generated_name_chars=40,
        max_interpret_name_chars=40,
    )
    normalized = validate_consolidation_response(
        wire_response,
        source_family=BOW_NUISANCE,
        candidates=candidates,
        wire_budget=wire_budget,
    )

    revalidated = revalidate_normalized_consolidation_response(
        normalized,
        source_family=BOW_NUISANCE,
        candidates=candidates,
        wire_budget=wire_budget,
    )
    assert revalidated == normalized
    assert len(revalidated["canonical_concepts"][1]["canonical_name"]) == 40
    with pytest.raises(ValueError, match="deterministic projection"):
        revalidate_normalized_consolidation_response(
            normalized,
            source_family=BOW_NUISANCE,
            candidates=candidates,
            wire_budget=LEGACY_HIERARCHY_WIRE_BUDGET,
        )


def test_architecture_dossier_fails_if_any_raw_atom_lacks_a_disposition():
    with pytest.raises(ValueError, match="disposition for every catalog atom"):
        ArchitectureDossier(
            source_family=BOW_NUISANCE,
            catalog_sha256="a" * 64,
            catalog_evidence_ids=("evidence_001", "evidence_002"),
            coverage_disposition_ids=("evidence_001",),
            coverage_audit_sha256="b" * 64,
            architecture_candidates=(_candidate(BOW_NUISANCE, 1),),
            direct_numerical_manifest_sha256="c" * 64,
            direct_numerical_signal_count=1,
        )


def test_cross_architecture_prompt_has_every_dossier_but_no_raw_evidence_dump():
    dossiers, _ = _all_dossiers()
    context = cross_architecture_planner_context(dossiers)
    assert [row["source_family"] for row in context["architecture_dossiers"]] == list(
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    assert all("direct_numerical_channel" not in row for row in context["architecture_dossiers"])
    assert all(
        row["direct_numerical_channel"]["channel"] == DIRECT_UPSTREAM_NUMERICAL_CHANNEL
        for row in (dossier.as_authenticated_dict() for dossier in dossiers)
    )
    assert all(
        row["direct_numerical_channel"]["direct_numerical_contract_kind"]
        == DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
        for row in (dossier.as_authenticated_dict() for dossier in dossiers)
    )
    serialized = "\n".join(
        message["content"]
        for message in render_cross_architecture_planner_messages(
            dossiers,
            maximum_raw_evidence_lookback_ids=4,
        )
    )
    assert "secret raw clue" not in serialized
    assert "temporal_policy" not in serialized
    assert "sparse_query" not in serialized

    with pytest.raises(ValueError, match="requires every active architecture"):
        cross_architecture_planner_context(dossiers[:-1])


def test_intent_dossier_never_mislabels_its_digest_as_a_manifest():
    dossier = ArchitectureDossier(
        source_family=BOW_NUISANCE,
        catalog_sha256="a" * 64,
        catalog_evidence_ids=("evidence_001",),
        coverage_disposition_ids=("evidence_001",),
        coverage_audit_sha256="b" * 64,
        architecture_candidates=(_candidate(BOW_NUISANCE, 1),),
        direct_numerical_signal_count=1,
        direct_numerical_contract_kind=(DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT),
        direct_numerical_contract_sha256="c" * 64,
    )
    authenticated = dossier.as_authenticated_dict()["direct_numerical_channel"]
    assert authenticated["direct_numerical_contract_sha256"] == "c" * 64
    assert "manifest_sha256" not in authenticated
    assert dossier.direct_numerical_manifest_sha256 == ""
    with pytest.raises(ValueError, match="cannot expose a manifest"):
        ArchitectureDossier(
            source_family=BOW_NUISANCE,
            catalog_sha256="a" * 64,
            catalog_evidence_ids=("evidence_001",),
            coverage_disposition_ids=("evidence_001",),
            coverage_audit_sha256="b" * 64,
            architecture_candidates=(_candidate(BOW_NUISANCE, 1),),
            direct_numerical_signal_count=1,
            direct_numerical_contract_kind=(DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT),
            direct_numerical_contract_sha256="c" * 64,
            direct_numerical_manifest_sha256="c" * 64,
        )


def test_cross_architecture_raw_lookback_is_explicit_exact_and_bounded():
    dossiers, catalog = _all_dossiers()
    response = _planner_response(dossiers)
    response["lookback_slot_definitions"]["planner_lookback_slot_001"] = {
        "selection": "evidence_001",
        "question": "Does this clue denote age or a treatment duration?",
        "reason": "The compact candidate description is ambiguous.",
    }
    normalized = validate_cross_architecture_planner_response(
        response,
        dossiers=dossiers,
        maximum_raw_evidence_lookback_ids=4,
    )
    assert [
        member
        for group in normalized["provisional_groups"]
        for member in group["member_candidate_ids"]
    ] == [
        candidate.candidate_id
        for dossier in dossiers
        for candidate in dossier.architecture_candidates
    ]
    lookback = resolve_raw_evidence_lookback(
        planner_response=normalized,
        dossiers=dossiers,
        catalog=catalog,
        maximum_raw_evidence_lookback_ids=4,
    )
    assert [row["evidence_id"] for row in lookback] == ["evidence_001"]
    assert "secret raw clue 1" in str(lookback[0])
    assert "secret raw clue 2" not in str(lookback)

    invented = deepcopy(response)
    invented["lookback_slot_definitions"]["planner_lookback_slot_001"]["selection"] = "evidence_999"
    with pytest.raises(ValueError, match="not dossier-owned"):
        validate_cross_architecture_planner_response(
            invented,
            dossiers=dossiers,
            maximum_raw_evidence_lookback_ids=4,
        )


def test_role_routing_occurs_after_discovery_and_preserves_overlapping_support():
    evidence = (
        _evidence(BOW_NUISANCE, 1, axes=(TREATMENT_AXIS,)),
        _evidence(BOW_NUISANCE, 2, axes=(OUTCOME_AXIS,)),
        _evidence(BOW_NUISANCE, 3, axes=(HETEROGENEITY_AXIS, EXTRACTION_SUPPORT_AXIS)),
    )
    result = route_concept_roles(
        evidence=evidence,
        supporting_evidence_ids=("evidence_001", "evidence_002", "evidence_003"),
    )
    assert result.adjustment_roles == ("confounder_adjustment",)
    assert result.effect_modifier is True
    assert result.treatment_prediction_support is True
    assert result.extraction_definition_support is True

    prognostic = route_concept_roles(
        evidence=evidence,
        supporting_evidence_ids=("evidence_002",),
    )
    assert prognostic.adjustment_roles == ("prognostic_adjustment",)
    assert prognostic.effect_modifier is False


def test_extraction_definition_grounds_vocabulary_in_complete_support_evidence():
    evidence = (
        _evidence(
            BOW_NUISANCE,
            1,
            axes=(EXTRACTION_SUPPORT_AXIS,),
            raw_marker="Baseline NLR reported as a unitless ratio.",
        ),
    )
    request = ExtractionDefinitionRequest(
        canonical_name="baseline_nlr",
        evidence=evidence,
        supporting_evidence_ids=("evidence_001",),
        value_shape_hypothesis="continuous",
    )
    response = {
        "feature_name": "baseline_nlr",
        "measurement": "Read the baseline neutrophil-to-lymphocyte ratio.",
        "representation": {"kind": "continuous", "unit": "unitless", "categories": []},
        "aliases": ["nlr"],
        "distinguish_from": [],
        "missing_or_ambiguous": "Return null when absent or ambiguous.",
        "supporting_evidence_reviewed": True,
    }
    normalized = validate_extraction_definition_response(response, request=request)
    assert normalized["supporting_evidence_ids"] == ["evidence_001"]
    assert (
        revalidate_normalized_extraction_definition_response(normalized, request=request)
        == normalized
    )
    invented = deepcopy(response)
    invented["representation"]["unit"] = "mg/dL"
    normalized_invented = validate_extraction_definition_response(invented, request=request)
    assert normalized_invented["representation"] == {
        "kind": "unresolved",
        "unit": "",
        "categories": [],
    }
    assert (
        normalized_invented["vocabulary_normalization_audit"]["normalization_events"][0]["action"]
        == "representation_set_unresolved"
    )

    invented_alias = deepcopy(response)
    invented_alias["aliases"] = ["inflammatory index"]
    normalized_alias = validate_extraction_definition_response(
        invented_alias,
        request=request,
    )
    assert normalized_alias["aliases"] == []
    assert (
        normalized_alias["vocabulary_normalization_audit"]["normalization_events"][0]["action"]
        == "filtered"
    )


def test_extraction_reserved_mechanics_are_executable_but_not_clinical_ontology():
    evidence = (
        _evidence(
            BOW_NUISANCE,
            1,
            axes=(EXTRACTION_SUPPORT_AXIS,),
            raw_marker="Documented age measurement",
        ),
    )
    continuous_request = ExtractionDefinitionRequest(
        canonical_name="documented_age",
        evidence=evidence,
        supporting_evidence_ids=("evidence_001",),
        value_shape_hypothesis="continuous",
    )
    continuous = {
        "feature_name": "documented_age",
        "measurement": "Extract the documented age value without changing its scale.",
        "representation": {
            "kind": "continuous",
            "unit": AS_DOCUMENTED_UNIT,
            "categories": [],
        },
        "aliases": [],
        "distinguish_from": [],
        "missing_or_ambiguous": "Return null when absent or ambiguous.",
        "supporting_evidence_reviewed": True,
    }
    assert validate_extraction_definition_response(
        continuous,
        request=continuous_request,
    )[
        "supporting_evidence_ids"
    ] == ["evidence_001"]

    categorical_request = ExtractionDefinitionRequest(
        canonical_name="age_language_observed",
        evidence=evidence,
        supporting_evidence_ids=("evidence_001",),
        value_shape_hypothesis="categorical",
    )
    categorical = {
        **continuous,
        "feature_name": "age_language_observed",
        "measurement": "Record whether the supported age language is mentioned.",
        "representation": {
            "kind": "categorical",
            "unit": "",
            "categories": list(MECHANICAL_MENTION_CATEGORIES),
        },
    }
    assert validate_extraction_definition_response(
        categorical,
        request=categorical_request,
    )[
        "supporting_evidence_ids"
    ] == ["evidence_001"]

    mixed = deepcopy(categorical)
    mixed["representation"]["categories"] = ["not_mentioned", "documented"]
    normalized_mixed = validate_extraction_definition_response(
        mixed,
        request=categorical_request,
    )
    assert normalized_mixed["representation"]["kind"] == "unresolved"


def test_extraction_clinical_categories_require_two_or_more_literal_values():
    evidence = (
        _evidence(
            BOW_NUISANCE,
            1,
            axes=(EXTRACTION_SUPPORT_AXIS,),
            raw_marker="Biomarker result: negative or positive",
        ),
    )
    request = ExtractionDefinitionRequest(
        canonical_name="biomarker_result",
        evidence=evidence,
        supporting_evidence_ids=("evidence_001",),
        value_shape_hypothesis="categorical",
    )
    response = {
        "feature_name": "biomarker_result",
        "measurement": "Extract the documented biomarker result.",
        "representation": {
            "kind": "categorical",
            "unit": "",
            "categories": ["negative", "positive"],
        },
        "aliases": [],
        "distinguish_from": [],
        "missing_or_ambiguous": "Return null when absent or ambiguous.",
        "supporting_evidence_reviewed": True,
    }
    assert validate_extraction_definition_response(
        response,
        request=request,
    )[
        "supporting_evidence_ids"
    ] == ["evidence_001"]

    expanded_categories = [f"state {index}" for index in range(12)]
    expanded_evidence = (
        _evidence(
            BOW_NUISANCE,
            1,
            axes=(EXTRACTION_SUPPORT_AXIS,),
            raw_marker="; ".join(expanded_categories),
        ),
    )
    expanded_request = ExtractionDefinitionRequest(
        canonical_name="detailed_biomarker_result",
        evidence=expanded_evidence,
        supporting_evidence_ids=("evidence_001",),
        value_shape_hypothesis="categorical",
    )
    expanded = {
        **response,
        "feature_name": "detailed_biomarker_result",
        "representation": {
            "kind": "categorical",
            "unit": "",
            "categories": expanded_categories,
        },
    }

    validated_expanded = validate_extraction_definition_response(
        expanded,
        request=expanded_request,
    )

    assert validated_expanded["representation"]["categories"] == expanded_categories

    too_few = deepcopy(response)
    too_few["representation"]["categories"] = ["positive"]
    with pytest.raises(ValueError, match="at least two"):
        validate_extraction_definition_response(too_few, request=request)

    duplicate = deepcopy(response)
    duplicate["representation"]["categories"] = [
        "negative",
        "Positive",
        "positive",
    ]
    with pytest.raises(ValueError, match="distinct after case/spacing normalization"):
        validate_extraction_definition_response(duplicate, request=request)

    wrong_shape = deepcopy(response)
    wrong_shape["representation"] = {
        "kind": "continuous",
        "unit": AS_DOCUMENTED_UNIT,
        "categories": [],
    }
    with pytest.raises(ValueError, match="categorical value shape"):
        validate_extraction_definition_response(wrong_shape, request=request)


def test_extraction_vocabulary_cannot_be_assembled_across_evidence_fields():
    evidence = (
        DiscoveryEvidenceItem(
            evidence_id="evidence_001",
            source_family=BOW_NUISANCE,
            observable_axes=(EXTRACTION_SUPPORT_AXIS,),
            content={"left": "milligrams", "right": "per deciliter"},
        ),
    )
    with pytest.raises(ValueError, match="not literally grounded"):
        ExtractionDefinitionRequest(
            canonical_name="laboratory_measure",
            evidence=evidence,
            supporting_evidence_ids=("evidence_001",),
            allowed_units=("milligrams per deciliter",),
        )


def test_extraction_definition_compiler_restores_complete_support_set():
    evidence = (
        _evidence(BOW_NUISANCE, 1, axes=(EXTRACTION_SUPPORT_AXIS,)),
        _evidence(BOW_NUISANCE, 2, axes=(EXTRACTION_SUPPORT_AXIS,)),
    )
    request = ExtractionDefinitionRequest(
        canonical_name="combined_measure",
        evidence=evidence,
        supporting_evidence_ids=("evidence_001", "evidence_002"),
        value_shape_hypothesis="continuous",
    )
    response = {
        "feature_name": "combined_measure",
        "measurement": "Extract the documented combined measurement.",
        "representation": {
            "kind": "continuous",
            "unit": AS_DOCUMENTED_UNIT,
            "categories": [],
        },
        "aliases": [],
        "distinguish_from": [],
        "missing_or_ambiguous": "Return null when absent or ambiguous.",
        "supporting_evidence_reviewed": True,
    }
    normalized = validate_extraction_definition_response(response, request=request)
    assert normalized["supporting_evidence_ids"] == ["evidence_001", "evidence_002"]

    response["supporting_evidence_reviewed"] = False
    with pytest.raises(ValueError, match="did not review"):
        validate_extraction_definition_response(response, request=request)
