from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator

import oci.inference.adaptive_hierarchical_stage1_reconsideration as adaptive_module
from oci.inference.adaptive_hierarchical_stage1_reconsideration import (
    NEW_MISSING_CONSTRUCT,
    AdaptiveCoverageRequiresRevision,
    AdaptiveCurrentFeature,
    AdaptiveDiagnostic,
    AdaptiveHierarchicalStage1Reconsideration,
    AdaptiveReconsiderationConfig,
    AdaptiveArchitectureDossier,
    ExactSpentCatalogAuthentication,
    FrozenAdaptiveReconsiderationRound,
    adaptive_hierarchical_stage1_reconsideration_identity,
)
from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    MECHANICAL_MENTION_CATEGORIES,
    OUTCOME_AXIS,
    DiscoveryCandidate,
    canonical_json,
    content_sha256,
)
from oci.inference.hierarchical_all_architecture_discovery import (
    CONSOLIDATE_ARCHITECTURE_JOB,
    CROSS_ARCHITECTURE_INTEGRATION_JOB,
    CROSS_ARCHITECTURE_PLANNER_JOB,
    COVERAGE_CRITIC_JOB,
    DiscoveryJobSettings,
    EXTRACTION_DEFINITION_JOB,
    INTERPRET_CHUNK_JOB,
    SELECTOR_THINKING_TOKEN_BUDGET,
)
from oci.inference.hierarchical_discovery_job_cache import (
    AuthenticatedHierarchicalDiscoveryJobCache,
)
from oci.inference.hierarchical_discovery_response_contract import (
    LEGACY_HIERARCHY_WIRE_BUDGET,
)
from oci.inference.openai_compatible_json_discovery_job_runner import (
    InvalidDiscoveryJsonResponse,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
    SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION,
    RoleNeutralEvidenceCatalog,
    Stage1EvidenceAtom,
    validate_role_neutral_catalog,
)
from tests.hierarchy_resource_test_support import HIERARCHY_JOB_CACHE_CONFIG


def _catalog(
    *, first_family_atom_count: int = 2, outer_fold: int = 1
) -> RoleNeutralEvidenceCatalog:
    split_fingerprint = f"{outer_fold:x}" * 64
    semantic_member_batch_size = 1
    semantic_member_batching = {
        "schema_version": SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION,
        "semantic_member_batch_size": semantic_member_batch_size,
        "selection_or_truncation_authorized": False,
        "complete_member_coverage_required": True,
    }
    atoms = []
    ordinal = 0
    for family_index, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES, start=1):
        count = first_family_atom_count if family_index == 1 else 1
        for family_ordinal in range(1, count + 1):
            ordinal += 1
            member_id = f"member_{ordinal:03d}"
            origin = {"closed_source": family, "ordinal": ordinal}
            content = {
                "terms": [
                    {
                        "member_id": member_id,
                        "term": f"documented {family} clue {family_ordinal:03d}",
                    }
                ]
            }
            origin_sha = content_sha256(origin)
            content_sha = content_sha256(content)
            identity = {
                "atom_kind": "test_semantic_atom",
                "source_kind": f"closed_test_source_{family_index:02d}",
                "source_family": family,
                "observable_axes": (OUTCOME_AXIS,),
                "member_ids": (member_id,),
                "split_fingerprint": split_fingerprint,
                "origin_sha256": origin_sha,
                "content_sha256": content_sha,
            }
            atoms.append(
                Stage1EvidenceAtom(
                    evidence_id=f"evidence_{content_sha256(identity)}",
                    atom_kind="test_semantic_atom",
                    source_kind=f"closed_test_source_{family_index:02d}",
                    source_family=family,
                    observable_axes=(OUTCOME_AXIS,),
                    member_ids=(member_id,),
                    split_fingerprint=split_fingerprint,
                    origin_sha256=origin_sha,
                    content_sha256=content_sha,
                    _origin_json=canonical_json(origin),
                    _content_json=canonical_json(content),
                )
            )
    identity = {
        "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
        "semantic_member_batching": semantic_member_batching,
        "outer_fold": outer_fold,
        "scope": "outer_train",
        "inner_fold": None,
        "split_fingerprint": split_fingerprint,
        "atoms": [atom.as_dict() for atom in atoms],
        "non_grounding_numerical_summaries": [],
    }
    result = RoleNeutralEvidenceCatalog(
        outer_fold=outer_fold,
        scope="outer_train",
        inner_fold=None,
        split_fingerprint=split_fingerprint,
        atoms=tuple(atoms),
        non_grounding_numerical_summaries=(),
        catalog_sha256=content_sha256(identity),
        _audit_json=canonical_json(
            {
                "semantic_member_batching": semantic_member_batching,
                "semantic_member_batch_size": semantic_member_batch_size,
            }
        ),
    )
    validate_role_neutral_catalog(result)
    return result


def _authentication(catalog: RoleNeutralEvidenceCatalog) -> ExactSpentCatalogAuthentication:
    return ExactSpentCatalogAuthentication.create(
        catalog=catalog,
        accumulated_spent_scope_sha256="a" * 64,
        accumulated_spent_row_count=640,
        consumed_gate_fingerprints=("b" * 64,),
        still_sealed_gate_fingerprint="c" * 64,
        upstream_authentication_sha256="d" * 64,
    )


def _builder(
    *,
    catalog: RoleNeutralEvidenceCatalog | None = None,
    config: AdaptiveReconsiderationConfig | None = None,
) -> AdaptiveHierarchicalStage1Reconsideration:
    selected_catalog = catalog or _catalog()
    first_atom = selected_catalog.atoms[0]
    return AdaptiveHierarchicalStage1Reconsideration(
        catalog=selected_catalog,
        exact_spent_authentication=_authentication(selected_catalog),
        family_explanations={
            family: f"Interpret concept-bearing clues from {family}."
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        current_registry=(
            AdaptiveCurrentFeature(
                feature_name="existing_measure",
                description="A currently extracted patient measurement.",
                value_shape_hypothesis="ambiguous",
                source_families=(first_atom.source_family,),
                supporting_evidence_ids=(first_atom.evidence_id,),
                definition_summary="Extract the documented patient measurement.",
            ),
        ),
        diagnostics=(
            AdaptiveDiagnostic(
                diagnostic_id="diagnostic_001",
                diagnostic_kind="extraction_missingness",
                affected_features=("existing_measure",),
                summary="The retained measurement has elevated aggregate missingness.",
                aggregate_metrics={"missing_fraction": 0.31, "observed_count": 442},
            ),
        ),
        config=config
        or AdaptiveReconsiderationConfig(
            max_atoms_per_chunk=1,
            max_bytes_per_chunk=20_000,
        ),
    )


def _builder_with_registry_count(
    *,
    registry_count: int,
    catalog: RoleNeutralEvidenceCatalog | None = None,
) -> AdaptiveHierarchicalStage1Reconsideration:
    selected_catalog = catalog or _catalog()
    first_atom = selected_catalog.atoms[0]
    registry = tuple(
        AdaptiveCurrentFeature(
            feature_name=f"existing_measure_{index:03d}",
            description=f"Current patient measurement {index:03d}.",
            value_shape_hypothesis="ambiguous",
            source_families=(first_atom.source_family,),
            supporting_evidence_ids=(first_atom.evidence_id,),
            definition_summary=f"Extract current patient measurement {index:03d}.",
        )
        for index in range(1, registry_count + 1)
    )
    return AdaptiveHierarchicalStage1Reconsideration(
        catalog=selected_catalog,
        exact_spent_authentication=_authentication(selected_catalog),
        family_explanations={
            family: f"Interpret concept-bearing clues from {family}."
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        current_registry=registry,
        diagnostics=(
            AdaptiveDiagnostic(
                diagnostic_id="diagnostic_001",
                diagnostic_kind="extraction_missingness",
                affected_features=tuple(item.feature_name for item in registry),
                summary="The retained measurements need aggregate reconsideration.",
                aggregate_metrics={"observed_count": 442},
            ),
        ),
        config=AdaptiveReconsiderationConfig(
            max_atoms_per_chunk=1,
            max_bytes_per_chunk=20_000,
        ),
    )


def _manual_complete_dossiers(builder):
    dossiers = []
    for family_index, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES, start=1):
        evidence_ids = tuple(
            atom.evidence_id for atom in builder.catalog.atoms if atom.source_family == family
        )
        candidate = DiscoveryCandidate(
            candidate_id=f"manual_candidate_{family_index:03d}",
            feature_name=f"manual_measure_{family_index:03d}",
            description="A compact architecture-local patient measurement.",
            supporting_evidence_ids=evidence_ids,
            source_families=(family,),
            value_shape_hypothesis="ambiguous",
            unresolved_ambiguity="The exact representation remains uncertain.",
        )
        dossiers.append(
            AdaptiveArchitectureDossier.create(
                source_family=family,
                catalog_sha256=builder.catalog.catalog_sha256,
                catalog_evidence_ids=evidence_ids,
                coverage_disposition_ids=evidence_ids,
                architecture_candidates=(candidate,),
                interpretation_job_ids=(
                    f"job_{content_sha256({'kind': 'interpret', 'family': family})}",
                ),
                consolidation_job_id=f"manual_consolidation_{family_index:03d}",
                coverage_job_ids=(f"job_{content_sha256({'kind': 'coverage', 'family': family})}",),
            )
        )
    return tuple(dossiers)


def _manual_frozen_round(*, builder, dossiers, lookback, proposal):
    audit = {
        "schema_version": adaptive_module.ADAPTIVE_ROUND_FREEZE_VERSION,
        "manual_lossless_test_fixture": True,
        "proposal_frozen_before_next_gate": True,
    }
    values = {
        "schema_version": adaptive_module.ADAPTIVE_ROUND_FREEZE_VERSION,
        "exact_spent_authentication_sha256": (
            builder.exact_spent_authentication.authentication_sha256
        ),
        "catalog_sha256": builder.catalog.catalog_sha256,
        "chunk_plan_sha256": builder.chunk_plan.plan_sha256,
        "dossier_sha256s": [item.dossier_sha256 for item in dossiers],
        "current_registry_sha256": content_sha256(builder._registry_private_items()),
        "diagnostics_sha256": content_sha256(builder._diagnostic_prompt_items()),
        "planner_job_id": "manual_planner_compilation",
        "planner_response_sha256": content_sha256({"manual": "planner"}),
        "lookback_sha256": lookback.lookback_sha256,
        "proposer_job_id": "manual_proposer_compilation",
        "proposal_sha256": content_sha256(proposal),
        "still_sealed_gate_fingerprint": (
            builder.exact_spent_authentication.still_sealed_gate_fingerprint
        ),
        "proposal": proposal,
        "audit": audit,
    }
    return FrozenAdaptiveReconsiderationRound(
        exact_spent_authentication_sha256=values["exact_spent_authentication_sha256"],
        catalog_sha256=values["catalog_sha256"],
        chunk_plan_sha256=values["chunk_plan_sha256"],
        dossier_sha256s=tuple(values["dossier_sha256s"]),
        current_registry_sha256=values["current_registry_sha256"],
        diagnostics_sha256=values["diagnostics_sha256"],
        planner_job_id=values["planner_job_id"],
        planner_response_sha256=values["planner_response_sha256"],
        lookback_sha256=values["lookback_sha256"],
        proposer_job_id=values["proposer_job_id"],
        proposal_sha256=values["proposal_sha256"],
        still_sealed_gate_fingerprint=values["still_sealed_gate_fingerprint"],
        freeze_sha256=content_sha256(values),
        _proposal_json=canonical_json(proposal),
        _audit_json=canonical_json(audit),
    )


def test_current_registry_rejects_reserved_missing_construct_name():
    catalog = _catalog()
    first_atom = catalog.atoms[0]
    reserved = AdaptiveCurrentFeature(
        feature_name=NEW_MISSING_CONSTRUCT,
        description="A conflicting current feature name.",
        value_shape_hypothesis="ambiguous",
        source_families=(first_atom.source_family,),
        supporting_evidence_ids=(first_atom.evidence_id,),
        definition_summary="This name conflicts with the adaptive planner sentinel.",
    )
    with pytest.raises(ValueError, match="reserved adaptive missing-construct"):
        AdaptiveHierarchicalStage1Reconsideration(
            catalog=catalog,
            exact_spent_authentication=_authentication(catalog),
            family_explanations={
                family: f"Interpret concept-bearing clues from {family}."
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            },
            current_registry=(reserved,),
            diagnostics=(
                AdaptiveDiagnostic(
                    diagnostic_id="diagnostic_001",
                    diagnostic_kind="extraction_missingness",
                    affected_features=(NEW_MISSING_CONSTRUCT,),
                    summary="The reserved-name feature has aggregate missingness.",
                ),
            ),
            config=AdaptiveReconsiderationConfig(
                max_atoms_per_chunk=1,
                max_bytes_per_chunk=20_000,
            ),
        )


def test_adaptive_semantic_member_bound_is_authenticated_and_applied_to_chunks():
    config = AdaptiveReconsiderationConfig(
        max_atoms_per_chunk=2,
        max_bytes_per_chunk=20_000,
        max_semantic_member_ids_per_chunk=1,
    )
    builder = _builder(config=config)
    identity = adaptive_hierarchical_stage1_reconsideration_identity(config)

    assert identity["config"]["max_semantic_member_ids_per_chunk"] == 1
    assert identity["config_sha256"] == content_sha256(identity["config"])
    assert builder.chunk_plan.max_semantic_member_ids_per_chunk == 1
    assert all(
        sum(len(item["member_ids"]) for item in chunk.evidence) <= 1
        for chunk in builder.chunk_plan.chunks
    )
    assert builder.delivery_audit["all_catalog_semantic_member_ids_delivered_exactly_once"] is True
    prompt_contract = identity["prompt_contract"]
    assert all(
        "hierarchy_wire_budget" in stage["user_payload_top_level_keys"]
        and "hierarchy_wire_budget" in stage["dynamic_user_payload_paths"]
        for stage in [
            *prompt_contract["stages"],
            *prompt_contract["phased_stage_variants"],
        ]
    )
    assert all(
        json.loads(job.messages[1]["content"])["hierarchy_wire_budget"]
        == config.wire_budget.as_dict()
        for job in builder.interpret_jobs
    )


def _interpretation_responses(builder):
    responses = {}
    for ordinal, job in enumerate(builder.interpret_jobs, start=1):
        request = json.loads(job.messages[1]["content"])
        dispositions = {}
        for evidence_index, evidence in enumerate(request["evidence"], start=1):
            feature_name = f"architecture_measure_{ordinal:03d}_{evidence_index:03d}"
            dispositions[evidence["evidence_id"]] = {
                "evidence_findings": [
                    {
                        "feature_name": feature_name,
                        "description": "A patient-level measurement supported by this clue.",
                        "value_shape_hypothesis": "ambiguous",
                        "unresolved_ambiguity": "The representation remains uncertain.",
                    }
                ],
                "member_dispositions": {
                    member_id: {"findings": []} for member_id in evidence["member_ids"]
                },
                "reason": "The supplied semantic member supports the measurement.",
            }
        responses[job.job_id] = {
            "evidence_dispositions": dispositions,
        }
    return responses


def _consolidation_responses(builder, interpretations):
    responses = {}
    for job in builder.build_consolidation_jobs(interpretations):
        request = json.loads(job.messages[1]["content"])
        candidates = request["candidates"]
        slots = request["identifier_ownership"]["identifier_domains"]["cluster_slots"]
        slot_by_name = {}
        for candidate in candidates:
            slot_by_name.setdefault(candidate["feature_name"], slots[len(slot_by_name)])
        candidate_by_slot = {
            slot_by_name[candidate["feature_name"]]: candidate for candidate in candidates
        }
        responses[job.job_id] = {
            "candidate_assignments": {
                candidate["candidate_id"]: {
                    "cluster_slot": slot_by_name[candidate["feature_name"]],
                    "reason": "Keep this architecture-local measurement distinct.",
                }
                for candidate in candidates
            },
            "slot_definitions": {
                slot: {
                    "canonical_name": (
                        candidate_by_slot[slot]["feature_name"]
                        if slot in candidate_by_slot
                        else f"unused_{slot}"
                    ),
                    "description": (
                        candidate_by_slot[slot]["description"]
                        if slot in candidate_by_slot
                        else "Unused compiler-owned consolidation slot."
                    ),
                    "unresolved_ambiguity": (
                        candidate_by_slot[slot]["unresolved_ambiguity"]
                        if slot in candidate_by_slot
                        else ""
                    ),
                }
                for slot in slots
            },
        }
    return responses


def _coverage_responses(builder, interpretations, consolidations):
    responses = {}
    jobs = builder.build_coverage_jobs(
        interpretation_responses=interpretations,
        consolidation_responses=consolidations,
    )
    for job in jobs:
        request = json.loads(job.messages[1]["content"])
        responses[job.job_id] = {
            "findings": [],
            "reviewed_evidence_ids": {row["evidence_id"]: True for row in request["evidence"]},
        }
    return responses


def _dense_interpretation_responses(builder, *, dense_candidate_count):
    dense_family = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    dense_generated = 0
    feature_ordinal = 0
    responses = {}
    for job in builder.interpret_jobs:
        request = json.loads(job.messages[1]["content"])
        dispositions = {}
        for evidence in request["evidence"]:
            family = evidence["source_family"]
            finding_count = (
                min(8, max(0, dense_candidate_count - dense_generated))
                if family == dense_family
                else 1
            )
            evidence_count = min(4, finding_count)
            member_count = finding_count - evidence_count

            def finding():
                nonlocal dense_generated, feature_ordinal
                feature_ordinal += 1
                if family == dense_family:
                    dense_generated += 1
                return {
                    "feature_name": f"dense_measure_{feature_ordinal:03d}",
                    "description": "A distinct patient-level measurement from this clue.",
                    "value_shape_hypothesis": "ambiguous",
                    "unresolved_ambiguity": "The compact representation remains uncertain.",
                }

            evidence_findings = [finding() for _ in range(evidence_count)]
            member_dispositions = {}
            for member_index, member_id in enumerate(evidence["member_ids"]):
                member_dispositions[member_id] = {
                    "findings": (
                        [finding() for _ in range(member_count)] if member_index == 0 else []
                    )
                }
            dispositions[evidence["evidence_id"]] = {
                "evidence_findings": evidence_findings,
                "member_dispositions": member_dispositions,
                "reason": "Review every exact semantic member without truncation.",
            }
        responses[job.job_id] = {"evidence_dispositions": dispositions}
    return responses


class _PhasedResponseCallback:
    def __init__(self, *, relation):
        self.relation = relation
        self.jobs = []

    def __call__(self, job, validator):
        request = json.loads(job.messages[1]["content"])
        self.jobs.append(job)
        if request["job"] == "compare_adaptive_candidate_relations":
            response = {
                "comparisons": {
                    peer_id: {
                        "relation": self.relation,
                        "reason": "Apply the precommitted pair judgment.",
                    }
                    for peer_id in request["peer_candidate_ids"]
                }
            }
        elif request["job"] == "fold_adaptive_group_definition":
            prior = request["prior_accumulator"]
            first = request["fresh_candidates"][0]
            response = {
                "canonical_name": (
                    prior["canonical_name"] if prior is not None else first["feature_name"]
                ),
                "description": (
                    prior["description"] if prior is not None else first["description"]
                ),
                "unresolved_ambiguity": (
                    prior["unresolved_ambiguity"]
                    if prior is not None
                    else first["unresolved_ambiguity"]
                ),
                "reason": "Fold all compiler-proven group members.",
            }
        elif request["job"] == "audit_adaptive_architecture_coverage":
            response = {
                "findings": [],
                "reviewed_evidence_ids": {row["evidence_id"]: True for row in request["evidence"]},
            }
        elif request["job"] == "audit_adaptive_atomic_coverage":
            response = {"findings": [], "reviewed_atomic_review": True}
        else:
            raise AssertionError(f"unexpected phased job: {request['job']}")
        return validator(response)


class _ExhaustiveAdaptiveReviewCallback:
    def __init__(self, *, selected_evidence_id=None, emit_singleton_drops=False):
        self.selected_evidence_id = selected_evidence_id
        self.emit_singleton_drops = emit_singleton_drops
        self.jobs = []

    def __call__(self, job, validator):
        request = json.loads(job.messages[1]["content"])
        self.jobs.append(job)
        request_job = request["job"]
        if request_job == "plan_adaptive_stage1_reconsideration":
            target = (
                request["current_registry"][0]["feature_name"]
                if request["current_registry"]
                else NEW_MISSING_CONSTRUCT
            )
            evidence_id = next(
                evidence_id
                for dossier in request["architecture_dossiers"]
                for evidence_id in dossier["coverage"]["lookback_evidence_ids"]
            )
            should_review = self.selected_evidence_id is None or (
                target != NEW_MISSING_CONSTRUCT and evidence_id == self.selected_evidence_id
            )
            if should_review:
                owning_family = next(
                    dossier["source_family"]
                    for dossier in request["architecture_dossiers"]
                    if evidence_id in dossier["coverage"]["lookback_evidence_ids"]
                )
                response = {
                    "review_targets": [
                        {
                            "target": target,
                            "problem": "This bounded page supports explicit reconsideration.",
                            "relevant_architectures": [owning_family],
                            "requested_evidence_ids": [evidence_id],
                            "reason": "Retain this exact page decision for recursive integration.",
                        }
                    ],
                    "no_lookback_needed": False,
                }
            else:
                response = {"review_targets": [], "no_lookback_needed": True}
        elif request_job == "fold_cross_architecture_group_definition":
            prior = request["prior_accumulator"]
            first = request["fresh_candidates"][0]
            response = {
                "canonical_name": (
                    prior["canonical_name"] if prior is not None else first["feature_name"]
                ),
                "description": (
                    prior["description"] if prior is not None else first["description"]
                ),
                "unresolved_ambiguity": (
                    prior["unresolved_ambiguity"]
                    if prior is not None
                    else first["unresolved_ambiguity"]
                ),
                "reason": "Fold every exact bounded review input.",
            }
        elif request_job == "propose_adaptive_registry_revision":
            targets = [row["target"] for row in request["review_plan"]["review_targets"]]
            existing_targets = [target for target in targets if target != NEW_MISSING_CONSTRUCT]
            if self.emit_singleton_drops and len(existing_targets) == 1 and len(targets) == 1:
                response = {
                    "operations": [
                        {
                            "operation": "drop",
                            "targets": existing_targets,
                            "proposed_feature": {},
                            "supporting_evidence_ids": [],
                            "diagnostic_ids": [request["diagnostics"][0]["diagnostic_id"]],
                            "reason": "This exact singleton target warrants a drop review.",
                        }
                    ],
                    "converged": False,
                }
            else:
                response = {"operations": [], "converged": True}
        elif request_job == "integrate_cross_architecture_group":
            operation = request["proposal"]
            canonical_name = (
                operation["targets"][0]
                if operation["operation"] == "drop"
                else operation["proposed_feature"]["feature_name"]
            )
            response = {
                "decision": "accept",
                "canonical_name": canonical_name,
                "description": "An explicitly judged patient-feature revision.",
                "unresolved_ambiguity": "",
                "reason": "Accept this exact proposal for compiler conflict resolution.",
            }
        elif request_job == "compare_cross_architecture_candidate_relations":
            response = {
                "comparisons": {
                    peer_id: {
                        "relation": "distinct",
                        "reason": "Keep distinct revision measurements separate.",
                    }
                    for peer_id in request["peer_candidate_ids"]
                }
            }
        elif request_job == "review_extraction_feature_evidence":
            response = {
                "measurement_observation": "One documented patient-level measurement.",
                "shape_observation": "ambiguous",
                "literal_aliases": [],
                "literal_units": [],
                "literal_categories": [],
                "literal_distinctions": [],
                "missing_or_ambiguous": "The exact representation remains unresolved.",
                "reviewed_evidence": True,
            }
        elif request_job == "fold_extraction_evidence_definitions":
            response = {
                "feature_name": request["canonical_name"],
                "measurement": "The documented patient-level measurement.",
                "representation": {
                    "kind": "unresolved",
                    "unit": "",
                    "categories": [],
                },
                "aliases": [],
                "distinguish_from": [],
                "missing_or_ambiguous": "The exact representation remains unresolved.",
                "input_dispositions": {
                    review_id: {
                        "action": "integrated",
                        "reason": "Integrate this exact page or prior accumulator.",
                    }
                    for review_id in request["review_input_ids"]
                },
                "supporting_evidence_reviewed": True,
            }
        else:
            raise AssertionError(f"unexpected exhaustive adaptive job: {request_job}")
        return validator(response)


def _completed_hierarchy(builder):
    interpretations = _interpretation_responses(builder)
    consolidations = _consolidation_responses(builder, interpretations)
    coverage = _coverage_responses(builder, interpretations, consolidations)
    dossiers = builder.compile_dossiers(
        interpretation_responses=interpretations,
        consolidation_responses=consolidations,
        coverage_responses=coverage,
    )
    return interpretations, consolidations, coverage, dossiers


def _planner_response(dossiers):
    family = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    dossier = next(row for row in dossiers if row.source_family == family)
    requested_id = dossier.catalog_evidence_ids[-1]
    return {
        "review_targets": [
            {
                "target": NEW_MISSING_CONSTRUCT,
                "problem": "A missing patient characteristic may explain the diagnostic.",
                "relevant_architectures": [family],
                "requested_evidence_ids": [requested_id],
                "reason": "The compact architecture candidate needs exact semantic confirmation.",
            },
            {
                "target": "existing_measure",
                "problem": "The existing definition has elevated aggregate missingness.",
                "relevant_architectures": [family],
                "requested_evidence_ids": [],
                "reason": "The registry and diagnostic are enough to consider a drop.",
            },
        ],
        "no_lookback_needed": False,
    }


def _proposer_response(builder, lookback):
    family = builder.catalog.atoms[-1].source_family
    requested_atom = builder._atom_by_id[lookback.requested_evidence_ids[0]]
    family = requested_atom.source_family
    return {
        "operations": [
            {
                "operation": "add",
                "targets": ["newly_surfaced_measure"],
                "proposed_feature": {
                    "feature_name": "newly_surfaced_measure",
                    "description": "A newly surfaced patient-level measurement.",
                    "value_shape_hypothesis": "ambiguous",
                    "definition_summary": "Extract the documented characteristic.",
                    "source_families": [family],
                },
                "supporting_evidence_ids": list(lookback.requested_evidence_ids),
                "diagnostic_ids": ["diagnostic_001"],
                "reason": "The requested clue grounds a construct absent from the registry.",
            },
            {
                "operation": "drop",
                "targets": ["existing_measure"],
                "proposed_feature": {},
                "supporting_evidence_ids": [],
                "diagnostic_ids": ["diagnostic_001"],
                "reason": "The existing extraction remains unusably sparse.",
            },
        ],
        "converged": False,
    }


def _existing_measure_specs():
    return (
        {
            "name": "existing_measure",
            "type": "categorical",
            "roles": ["confounder"],
            "description": "Extract the documented patient measurement.",
            "categories": list(MECHANICAL_MENTION_CATEGORIES),
        },
    )


class _ConvergedAdaptiveRunner:
    def __init__(self):
        identity_body = {"schema_version": "test_converged_adaptive_runner_v1"}
        self._identity = {
            **identity_body,
            "identity_sha256": content_sha256(identity_body),
        }
        self.calls = []
        self.jobs = []
        self._execution_metadata = []

    def identity(self):
        return json.loads(canonical_json(self._identity))

    @property
    def execution_metadata(self):
        return tuple(json.loads(canonical_json(row)) for row in self._execution_metadata)

    def _record_response(self, *, job, response):
        metadata = {
            "job_id": job.job_id,
            "runner_identity_sha256": self._identity["identity_sha256"],
            "outcome": "success",
            "parsed_response_sha256": content_sha256(response),
        }
        self.calls.append(job.job_id)
        self.jobs.append(job)
        self._execution_metadata.append(metadata)
        return response

    def run_json(self, *, job):
        request = json.loads(job.messages[1]["content"])
        request_job = request["job"]
        if request_job == "compare_adaptive_candidate_relations":
            response = {
                "comparisons": {
                    peer_id: {
                        "relation": "distinct",
                        "reason": "Keep the compact measurements distinct.",
                    }
                    for peer_id in request["peer_candidate_ids"]
                }
            }
        elif request_job == "fold_adaptive_group_definition":
            prior = request["prior_accumulator"]
            first = request["fresh_candidates"][0]
            response = {
                "canonical_name": (
                    prior["canonical_name"] if prior is not None else first["feature_name"]
                ),
                "description": (
                    prior["description"] if prior is not None else first["description"]
                ),
                "unresolved_ambiguity": (
                    prior["unresolved_ambiguity"]
                    if prior is not None
                    else first["unresolved_ambiguity"]
                ),
                "reason": "Fold every compiler-proven group member.",
            }
        elif request_job == "audit_adaptive_atomic_coverage":
            response = {"findings": [], "reviewed_atomic_review": True}
        elif job.job_kind == INTERPRET_CHUNK_JOB:
            dispositions = {}
            for index, evidence in enumerate(request["evidence"], start=1):
                feature_name = f"runner_measure_{len(self.calls) + 1:03d}_{index:03d}"
                dispositions[evidence["evidence_id"]] = {
                    "evidence_findings": [
                        {
                            "feature_name": feature_name,
                            "description": "A patient-level measurement supported by this clue.",
                            "value_shape_hypothesis": "ambiguous",
                            "unresolved_ambiguity": "The representation remains uncertain.",
                        }
                    ],
                    "member_dispositions": {
                        member_id: {"findings": []} for member_id in evidence["member_ids"]
                    },
                    "reason": "The semantic member supports this measurement.",
                }
            response = {
                "evidence_dispositions": dispositions,
            }
        elif job.job_kind == CONSOLIDATE_ARCHITECTURE_JOB:
            candidates = request["candidates"]
            slots = request["identifier_ownership"]["identifier_domains"]["cluster_slots"]
            slot_by_name = {}
            for candidate in candidates:
                slot_by_name.setdefault(candidate["feature_name"], slots[len(slot_by_name)])
            candidate_by_slot = {
                slot_by_name[candidate["feature_name"]]: candidate for candidate in candidates
            }
            response = {
                "candidate_assignments": {
                    candidate["candidate_id"]: {
                        "cluster_slot": slot_by_name[candidate["feature_name"]],
                        "reason": "Keep this architecture-local measurement distinct.",
                    }
                    for candidate in candidates
                },
                "slot_definitions": {
                    slot: {
                        "canonical_name": (
                            candidate_by_slot[slot]["feature_name"]
                            if slot in candidate_by_slot
                            else f"unused_{slot}"
                        ),
                        "description": (
                            candidate_by_slot[slot]["description"]
                            if slot in candidate_by_slot
                            else "Unused compiler-owned consolidation slot."
                        ),
                        "unresolved_ambiguity": (
                            candidate_by_slot[slot]["unresolved_ambiguity"]
                            if slot in candidate_by_slot
                            else ""
                        ),
                    }
                    for slot in slots
                },
            }
        elif job.job_kind == COVERAGE_CRITIC_JOB:
            response = {
                "findings": [],
                "reviewed_evidence_ids": {
                    evidence["evidence_id"]: True for evidence in request["evidence"]
                },
            }
        elif job.job_kind == CROSS_ARCHITECTURE_PLANNER_JOB:
            response = {"review_targets": [], "no_lookback_needed": True}
        elif job.job_kind == CROSS_ARCHITECTURE_INTEGRATION_JOB:
            response = {"operations": [], "converged": True}
        else:
            raise AssertionError(f"unexpected adaptive job kind: {job.job_kind}")
        return self._record_response(job=job, response=response)


class _MissingParsedResponseShaRunner(_ConvergedAdaptiveRunner):
    def run_json(self, *, job):
        response = super().run_json(job=job)
        self._execution_metadata[-1].pop("parsed_response_sha256")
        return response


class _DenseAdaptiveRunner(_ConvergedAdaptiveRunner):
    def __init__(self, *, dense_candidate_count):
        super().__init__()
        self.dense_candidate_count = dense_candidate_count
        self._dense_emitted = 0
        self._feature_ordinal = 0
        identity_body = {
            "schema_version": "test_dense_adaptive_runner_v1",
            "dense_candidate_count": dense_candidate_count,
        }
        self._identity = {
            **identity_body,
            "identity_sha256": content_sha256(identity_body),
        }

    def run_json(self, *, job):
        request = json.loads(job.messages[1]["content"])
        if job.job_kind != INTERPRET_CHUNK_JOB:
            return super().run_json(job=job)
        evidence = request["evidence"]
        family = evidence[0]["source_family"]
        dense_family = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
        finding_count = (
            min(8, self.dense_candidate_count - self._dense_emitted)
            if family == dense_family
            else 1
        )
        finding_count = max(0, finding_count)
        dispositions = {}
        for item in evidence:
            evidence_count = min(4, finding_count)
            member_count = finding_count - evidence_count

            def finding():
                self._feature_ordinal += 1
                if family == dense_family:
                    self._dense_emitted += 1
                return {
                    "feature_name": f"runner_dense_measure_{self._feature_ordinal:03d}",
                    "description": "A distinct patient-level measurement from this clue.",
                    "value_shape_hypothesis": "ambiguous",
                    "unresolved_ambiguity": "The representation remains uncertain.",
                }

            dispositions[item["evidence_id"]] = {
                "evidence_findings": [finding() for _ in range(evidence_count)],
                "member_dispositions": {
                    member_id: {
                        "findings": (
                            [finding() for _ in range(member_count)] if member_index == 0 else []
                        )
                    }
                    for member_index, member_id in enumerate(item["member_ids"])
                },
                "reason": "Review every exact semantic member without truncation.",
            }
        return self._record_response(
            job=job,
            response={"evidence_dispositions": dispositions},
        )


class _InvalidFirstAdaptiveRunner(_ConvergedAdaptiveRunner):
    def __init__(self):
        super().__init__()
        self.failed = False

    def run_json(self, *, job):
        if self.failed:
            return super().run_json(job=job)
        self.failed = True
        content = "not one JSON object"
        content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        metadata = {
            "job_id": job.job_id,
            "runner_identity_sha256": self._identity["identity_sha256"],
            "outcome": "invalid_response",
            "attempts": [{"content_sha256": content_sha}],
        }
        self.calls.append(job.job_id)
        self.jobs.append(job)
        self._execution_metadata.append(metadata)
        raise InvalidDiscoveryJsonResponse(failed_response_content=content)


def _executable_builder(
    *,
    config: AdaptiveReconsiderationConfig | None = None,
):
    catalog = _catalog()
    first_family, second_family = ACTIVE_STAGE1_CONCEPT_FAMILIES[:2]
    first_atom = next(atom for atom in catalog.atoms if atom.source_family == first_family)
    second_atom = next(atom for atom in catalog.atoms if atom.source_family == second_family)
    feature_names = (
        "documented_bow_nuisance_clue",
        "documented_bow_r_loss_clue",
    )
    return AdaptiveHierarchicalStage1Reconsideration(
        catalog=catalog,
        exact_spent_authentication=_authentication(catalog),
        family_explanations={
            family: f"Interpret concept-bearing clues from {family}."
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        current_registry=(
            AdaptiveCurrentFeature(
                feature_name=feature_names[0],
                description="A documented nuisance-family clue.",
                value_shape_hypothesis="categorical",
                source_families=(first_family,),
                supporting_evidence_ids=(first_atom.evidence_id,),
                definition_summary="Extract whether the nuisance clue is documented.",
            ),
            AdaptiveCurrentFeature(
                feature_name=feature_names[1],
                description="A documented R-loss-family clue.",
                value_shape_hypothesis="categorical",
                source_families=(second_family,),
                supporting_evidence_ids=(second_atom.evidence_id,),
                definition_summary="Extract whether the R-loss clue is documented.",
            ),
        ),
        diagnostics=(
            AdaptiveDiagnostic(
                diagnostic_id="diagnostic_001",
                diagnostic_kind="extraction_missingness",
                affected_features=feature_names,
                summary="The retained clue definitions need bounded reconsideration.",
                aggregate_metrics={"missing_fraction": 0.31, "observed_count": 442},
            ),
        ),
        config=config
        or AdaptiveReconsiderationConfig(
            max_atoms_per_chunk=1,
            max_bytes_per_chunk=20_000,
        ),
    )


def _executable_specs():
    return (
        {
            "name": "documented_bow_nuisance_clue",
            "type": "categorical",
            "roles": ["confounder"],
            "description": "Original nuisance-clue extraction definition.",
            "categories": list(MECHANICAL_MENTION_CATEGORIES),
        },
        {
            "name": "documented_bow_r_loss_clue",
            "type": "categorical",
            "roles": ["confounder"],
            "description": "Original R-loss-clue extraction definition.",
            "categories": list(MECHANICAL_MENTION_CATEGORIES),
        },
    )


def _operation_plan_and_proposal(builder, operation_kind):
    first_name, second_name = (
        "documented_bow_nuisance_clue",
        "documented_bow_r_loss_clue",
    )
    first_family, second_family = ACTIVE_STAGE1_CONCEPT_FAMILIES[:2]
    first_atom = next(atom for atom in builder.catalog.atoms if atom.source_family == first_family)
    second_atom = next(
        atom for atom in builder.catalog.atoms if atom.source_family == second_family
    )

    def review_target(*, target, family, evidence_ids):
        return {
            "target": target,
            "problem": "A bounded registry reconsideration is warranted.",
            "relevant_architectures": [family],
            "requested_evidence_ids": list(evidence_ids),
            "reason": "Inspect only the exact architecture-local supporting clue.",
        }

    if operation_kind == "add":
        proposed_name = "documented_bow_nuisance_clue_002"
        review_targets = [
            review_target(
                target=NEW_MISSING_CONSTRUCT,
                family=first_family,
                evidence_ids=(first_atom.evidence_id,),
            )
        ]
        targets = [proposed_name]
        support = [first_atom.evidence_id]
        source_families = [first_family]
    elif operation_kind == "drop":
        proposed_name = ""
        review_targets = [review_target(target=first_name, family=first_family, evidence_ids=())]
        targets = [first_name]
        support = []
        source_families = []
    elif operation_kind == "merge":
        proposed_name = "documented_clue"
        review_targets = [
            review_target(
                target=first_name,
                family=first_family,
                evidence_ids=(first_atom.evidence_id,),
            ),
            review_target(
                target=second_name,
                family=second_family,
                evidence_ids=(second_atom.evidence_id,),
            ),
        ]
        targets = [first_name, second_name]
        support = [first_atom.evidence_id, second_atom.evidence_id]
        source_families = [first_family, second_family]
    else:
        proposed_name = (
            first_name
            if operation_kind == "revise_definition"
            else "documented_bow_nuisance_clue_002"
        )
        review_targets = [
            review_target(
                target=first_name,
                family=first_family,
                evidence_ids=(first_atom.evidence_id,),
            )
        ]
        targets = [first_name]
        support = [first_atom.evidence_id]
        source_families = [first_family]

    planner_response = {
        "review_targets": review_targets,
        "no_lookback_needed": operation_kind == "drop",
    }
    if operation_kind == "drop":
        proposed_feature = {}
    else:
        proposed_feature = {
            "feature_name": proposed_name,
            "description": "A patient-level clue represented in source text.",
            "value_shape_hypothesis": "categorical",
            "definition_summary": "Extract whether the clue is documented.",
            "source_families": source_families,
        }
    proposer_response = {
        "operations": [
            {
                "operation": operation_kind,
                "targets": targets,
                "proposed_feature": proposed_feature,
                "supporting_evidence_ids": support,
                "diagnostic_ids": ["diagnostic_001"],
                "reason": "Apply the bounded evidence-grounded registry operation.",
            }
        ],
        "converged": False,
    }
    return planner_response, proposer_response


def _definition_response(request):
    return {
        "feature_name": request.canonical_name,
        "measurement": "Extract the documented clue.",
        "representation": {
            "kind": "categorical",
            "unit": "",
            "categories": list(MECHANICAL_MENTION_CATEGORIES),
        },
        "aliases": [],
        "distinguish_from": [],
        "missing_or_ambiguous": ("Handle ambiguity using the precommitted observation encoding."),
        "supporting_evidence_reviewed": True,
    }


def _prepare_operation_case(
    operation_kind,
    *,
    config: AdaptiveReconsiderationConfig | None = None,
):
    builder = _executable_builder(config=config)
    _, _, _, dossiers = _completed_hierarchy(builder)
    planner_response, proposer_response = _operation_plan_and_proposal(builder, operation_kind)
    planner_job = builder.build_planner_job(dossiers)
    lookback = builder.resolve_requested_evidence(
        dossiers=dossiers,
        planner_response=planner_response,
    )
    proposer_job = builder.build_proposer_job(
        dossiers=dossiers,
        planner_job=planner_job,
        planner_response=planner_response,
        lookback=lookback,
    )
    frozen_round = builder.freeze_round(
        dossiers=dossiers,
        planner_job=planner_job,
        planner_response=planner_response,
        lookback=lookback,
        proposer_job=proposer_job,
        proposer_response=proposer_response,
    )
    definition_jobs = builder.build_extraction_definition_jobs(
        frozen_round=frozen_round,
        lookback=lookback,
    )
    return builder, dossiers, frozen_round, lookback, definition_jobs


def _recursive_keys(value):
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(_recursive_keys(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_recursive_keys(child))
        return result
    return set()


def test_architecture_local_jobs_cover_all_ten_families_without_mixing():
    builder = _builder()

    assert builder.offline_contract["assurances"]["all_ten_architectures_required"] is True
    assert len(builder.interpret_jobs) == len(builder.catalog.atoms)
    assert {job.job_kind for job in builder.interpret_jobs} == {INTERPRET_CHUNK_JOB}
    assert all(
        job.settings.thinking_token_budget == SELECTOR_THINKING_TOKEN_BUDGET
        for job in builder.interpret_jobs
    )
    delivered = []
    for job in builder.interpret_jobs:
        request = json.loads(job.messages[1]["content"])
        assert len({row["source_family"] for row in request["evidence"]}) == 1
        delivered.extend(row["evidence_id"] for row in request["evidence"])
    assert set(delivered) == {atom.evidence_id for atom in builder.catalog.atoms}
    assert len(delivered) == len(set(delivered))

    interpretations = _interpretation_responses(builder)
    consolidation_jobs = builder.build_consolidation_jobs(interpretations)
    assert len(consolidation_jobs) == 10
    assert {job.job_kind for job in consolidation_jobs} == {CONSOLIDATE_ARCHITECTURE_JOB}
    assert [
        json.loads(job.messages[1]["content"])["source_family"] for job in consolidation_jobs
    ] == list(ACTIVE_STAGE1_CONCEPT_FAMILIES)


def test_phased_family_consolidation_pages_every_pair_and_folds_to_termination():
    builder = _builder(catalog=_catalog(first_family_atom_count=2))
    interpretations = _dense_interpretation_responses(
        builder,
        dense_candidate_count=9,
    )
    callback = _PhasedResponseCallback(relation="same_construct")

    artifacts, compiler_records = builder._execute_phased_family_consolidations(
        interpretation_responses=interpretations,
        run_job=callback,
    )

    family = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    artifact = artifacts[family]
    relation_jobs = [
        job
        for job in callback.jobs
        if json.loads(job.messages[1]["content"])["job"] == "compare_adaptive_candidate_relations"
    ]
    fold_jobs = [
        job
        for job in callback.jobs
        if json.loads(job.messages[1]["content"])["job"] == "fold_adaptive_group_definition"
    ]
    observed_pairs = set()
    for job in relation_jobs:
        request = json.loads(job.messages[1]["content"])
        assert len(request["peer_candidate_ids"]) <= 7
        observed_pairs.update(
            (request["anchor_candidate_id"], peer_id) for peer_id in request["peer_candidate_ids"]
        )
        assert (
            request["identifier_ownership"]["ownership"]["wire_response_budget"][
                "maximum_canonical_json_bytes"
            ]
            <= 20_000
        )
    assert len(artifact.candidate_ids) == 9
    assert len(observed_pairs) == 9 * 8 // 2
    assert len(fold_jobs) == 2
    assert [
        len(json.loads(job.messages[1]["content"])["member_candidate_ids"]) for job in fold_jobs
    ] == [8, 1]
    assert len(artifact.normalized_response["canonical_concepts"]) == 1
    assert set(
        artifact.normalized_response["canonical_concepts"][0]["supporting_evidence_ids"]
    ) == {atom.evidence_id for atom in builder.catalog.atoms if atom.source_family == family}
    assert artifact.compiler_audit["candidate_or_decision_truncation_applied"] is False
    assert any(row["consolidation_id"] == artifact.consolidation_id for row in compiler_records)


def test_phased_atomic_coverage_pages_every_relevant_name_without_loss():
    builder = _builder(catalog=_catalog(first_family_atom_count=1))
    interpretations = _dense_interpretation_responses(
        builder,
        dense_candidate_count=8,
    )
    consolidation_callback = _PhasedResponseCallback(relation="distinct")
    consolidations, _ = builder._execute_phased_family_consolidations(
        interpretation_responses=interpretations,
        run_job=consolidation_callback,
    )
    coverage_callback = _PhasedResponseCallback(relation="distinct")

    coverages, _ = builder._execute_phased_chunk_coverage(
        interpretation_responses=interpretations,
        family_consolidations=consolidations,
        run_job=coverage_callback,
    )

    family = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    family_coverage = next(row for row in coverages if row.source_family == family)
    atomic_jobs = [
        job
        for job in coverage_callback.jobs
        if json.loads(job.messages[1]["content"])["job"] == "audit_adaptive_atomic_coverage"
    ]
    assert family_coverage.compiler_audit["coverage_mode"] == ("atomic_evidence_name_pages_v1")
    assert len(atomic_jobs) == 2
    observed_names = []
    for job in atomic_jobs:
        request = json.loads(job.messages[1]["content"])
        assert len(request["canonical_names"]) <= 4
        observed_names.extend(request["canonical_names"])
        assert (
            request["identifier_ownership"]["ownership"]["wire_response_budget"][
                "maximum_canonical_json_bytes"
            ]
            <= 20_000
        )
    assert len(observed_names) == 8
    assert len(observed_names) == len(set(observed_names))
    assert family_coverage.normalized_response["reviewed_evidence_ids"] == [
        next(atom.evidence_id for atom in builder.catalog.atoms if atom.source_family == family)
    ]
    assert family_coverage.compiler_audit["evidence_or_model_decision_truncation_applied"] is False


def test_phased_planner_preserves_targets_and_evidence_above_every_old_cap():
    catalog = _catalog(first_family_atom_count=16)
    builder = _builder_with_registry_count(registry_count=5, catalog=catalog)
    dossiers = _manual_complete_dossiers(builder)
    callback = _ExhaustiveAdaptiveReviewCallback()

    planner, lookback, planner_id, compiler_record = builder._execute_phased_adaptive_planner(
        dossiers=dossiers,
        run_job=callback,
    )

    assert len(planner["review_targets"]) == 6
    assert len(planner["review_targets"]) > 4
    assert all(
        len(row["requested_evidence_ids"]) > builder.config.max_lookback_ids_per_target
        for row in planner["review_targets"]
    )
    assert lookback.requested_evidence_ids == tuple(
        atom.evidence_id for atom in builder.catalog.atoms
    )
    assert len(lookback.requested_evidence_ids) > builder.config.max_total_lookback_ids
    assert compiler_record["planner_compilation_id"] == planner_id
    assert compiler_record["expected_page_count"] == len(compiler_record["page_records"])
    assert all(
        row["disposition"] in {"review_requested", "no_review_requested"}
        for row in compiler_record["page_records"]
    )
    assert compiler_record["target_or_evidence_truncation_applied"] is False
    planner_page_jobs = [
        job
        for job in callback.jobs
        if json.loads(job.messages[1]["content"])["job"] == "plan_adaptive_stage1_reconsideration"
    ]
    assert planner_page_jobs
    assert all(
        sum(
            len(dossier["coverage"]["lookback_evidence_ids"])
            for dossier in json.loads(job.messages[1]["content"])["architecture_dossiers"]
        )
        == 1
        for job in planner_page_jobs
    )
    assert all(
        len(job.rendered_messages_bytes) <= builder.config.max_rendered_prompt_bytes
        for job in callback.jobs
    )


def test_phased_proposer_judges_more_than_operation_cap_before_capacity_dispositions():
    builder = _builder_with_registry_count(registry_count=6)
    dossiers = _manual_complete_dossiers(builder)
    selected_evidence_id = builder.catalog.atoms[0].evidence_id
    callback = _ExhaustiveAdaptiveReviewCallback(
        selected_evidence_id=selected_evidence_id,
        emit_singleton_drops=True,
    )
    planner, lookback, planner_id, planner_record = builder._execute_phased_adaptive_planner(
        dossiers=dossiers,
        run_job=callback,
    )

    proposal, proposer_id, proposer_record = builder._execute_phased_adaptive_proposer(
        dossiers=dossiers,
        planner_response=planner,
        lookback=lookback,
        planner_compilation_id=planner_id,
        planner_dependency_ids=(
            *planner_record["page_job_ids"],
            *planner_record["fold_job_ids"],
        ),
        run_job=callback,
    )

    assert planner_record["planner_compilation_id"] == planner_id
    assert len(proposer_record["raw_proposals"]) == 6
    assert len(proposer_record["raw_proposals"]) > builder.config.max_operations
    assert len(proposer_record["proposal_judgments"]) == 6
    assert len(proposer_record["proposal_dispositions"]) == 6
    assert len(proposal["operations"]) == builder.config.max_operations
    capacity_rejections = [
        row
        for row in proposer_record["proposal_dispositions"]
        if row["reason"] == "round_capacity_after_exhaustive_operation_validation"
    ]
    assert len(capacity_rejections) == 2
    assert all(
        row["disposition"] == "rejected_after_exhaustive_compilation" for row in capacity_rejections
    )
    assert proposer_record["proposer_compilation_id"] == proposer_id
    assert proposer_record["operation_slice_or_semantic_truncation_applied"] is False
    assert proposer_record["every_page_and_proposal_has_an_explicit_disposition"] is True
    assert all(
        len(job.rendered_messages_bytes) <= builder.config.max_rendered_prompt_bytes
        for job in callback.jobs
    )


def test_phased_extraction_reviews_more_than_eight_support_items_and_folds_all():
    catalog = _catalog(first_family_atom_count=10)
    builder = _builder(catalog=catalog)
    dossiers = _manual_complete_dossiers(builder)
    family = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    support = tuple(
        atom.evidence_id for atom in builder.catalog.atoms if atom.source_family == family
    )
    planner_wire = {
        "review_targets": [
            {
                "target": NEW_MISSING_CONSTRUCT,
                "problem": "Define one feature from every exact support page.",
                "relevant_architectures": [family],
                "requested_evidence_ids": list(support),
                "reason": "The complete support relation must remain compiler-owned.",
            }
        ],
        "no_lookback_needed": False,
    }
    lookback = builder.resolve_requested_evidence(
        dossiers=dossiers,
        planner_response=planner_wire,
    )
    proposal = {
        "operations": [
            {
                "operation": "add",
                "targets": ["lossless_support_measure"],
                "proposed_feature": {
                    "feature_name": "lossless_support_measure",
                    "description": "A patient feature with extensive exact support.",
                    "value_shape_hypothesis": "ambiguous",
                    "definition_summary": "Integrate every exact support item.",
                    "source_families": [family],
                },
                "supporting_evidence_ids": list(support),
                "diagnostic_ids": ["diagnostic_001"],
                "reason": "Every support item warrants grounded extraction review.",
            }
        ],
        "converged": False,
        "wire_normalization_audit": {"manual_lossless_test_fixture": True},
    }
    frozen = _manual_frozen_round(
        builder=builder,
        dossiers=dossiers,
        lookback=lookback,
        proposal=proposal,
    )
    callback = _ExhaustiveAdaptiveReviewCallback()

    artifacts, compiler_record = builder._execute_phased_extraction_definitions(
        frozen_round=frozen,
        lookback=lookback,
        proposer_compilation_id="manual_proposer_compilation",
        proposer_dependency_ids=(),
        run_job=callback,
    )

    artifact = artifacts["lossless_support_measure"]
    feature_record = compiler_record["feature_records"][0]
    page_jobs = [
        job
        for job in callback.jobs
        if json.loads(job.messages[1]["content"])["job"] == "review_extraction_feature_evidence"
    ]
    fold_jobs = [
        job
        for job in callback.jobs
        if json.loads(job.messages[1]["content"])["job"] == "fold_extraction_evidence_definitions"
    ]
    assert len(page_jobs) == len(support) == 10
    assert len(page_jobs) > 8
    assert len(fold_jobs) == 2
    assert [
        len(json.loads(job.messages[1]["content"])["review_input_ids"]) for job in fold_jobs
    ] == [8, 3]
    assert tuple(artifact["definition"]["supporting_evidence_ids"]) == support
    assert [row["evidence_id"] for row in feature_record["page_records"]] == list(support)
    assert all(
        row["disposition"] == "reviewed_exactly_once" for row in feature_record["page_records"]
    )
    assert feature_record["all_page_reviews_transitively_folded"] is True
    assert compiler_record["complete_support_single_prompt_present"] is False
    assert compiler_record["semantic_truncation_applied"] is False
    assert all(
        len(job.rendered_messages_bytes) <= builder.config.max_rendered_prompt_bytes
        for job in callback.jobs
    )


def test_completed_family_coverage_builds_exactly_ten_numerical_free_dossiers():
    builder = _builder()
    _, _, _, dossiers = _completed_hierarchy(builder)

    assert len(dossiers) == 10
    assert [row.source_family for row in dossiers] == list(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    assert all(
        set(row.catalog_evidence_ids) == set(row.coverage_disposition_ids) for row in dossiers
    )
    prompt_items = [row.as_prompt_item() for row in dossiers]
    keys = _recursive_keys(prompt_items)
    assert "content" not in keys
    assert "evidence" not in keys
    assert "direct_upstream_numerical" not in keys
    assert "non_grounding_numerical_summaries" not in keys


def test_planner_gets_only_ten_compact_dossiers_registry_and_diagnostics():
    builder = _builder()
    _, _, _, dossiers = _completed_hierarchy(builder)

    job = builder.build_planner_job(dossiers)
    request = json.loads(job.messages[1]["content"])
    audit = builder.audit_planner_prompt(job=job, dossiers=dossiers)

    assert job.job_kind == CROSS_ARCHITECTURE_PLANNER_JOB
    assert set(request) == {
        "job",
        "architecture_dossiers",
        "current_registry",
        "diagnostics",
        "lookback_bounds",
        "hierarchy_wire_budget",
        "identifier_ownership",
        "output_schema",
    }
    assert request["hierarchy_wire_budget"] == builder.config.wire_budget.as_dict()
    assert len(request["architecture_dossiers"]) == 10
    assert audit["raw_atom_count"] == 0
    assert audit["complete_catalog_dump_present"] is False
    assert audit["direct_numerical_channel_present"] is False
    assert '"content":' not in canonical_json(request["architecture_dossiers"])


def test_bounded_id_resolution_returns_only_requested_atom_without_rewriting():
    builder = _builder()
    _, _, _, dossiers = _completed_hierarchy(builder)
    response = _planner_response(dossiers)

    lookback = builder.resolve_requested_evidence(
        dossiers=dossiers,
        planner_response=response,
    )

    assert len(lookback.items) == 1
    assert tuple(row["evidence_id"] for row in lookback.items) == (lookback.requested_evidence_ids)
    expected = (
        builder._atom_by_id[lookback.requested_evidence_ids[0]].as_discovery_item().as_prompt_item()
    )
    assert lookback.items[0] == expected
    assert lookback.audit()["all_catalog_atoms_returned"] is False


def test_proposer_supports_new_addition_and_targeted_drop_then_freezes_round():
    builder = _builder()
    _, _, _, dossiers = _completed_hierarchy(builder)
    planner_job = builder.build_planner_job(dossiers)
    planner_response = _planner_response(dossiers)
    lookback = builder.resolve_requested_evidence(
        dossiers=dossiers,
        planner_response=planner_response,
    )
    proposer_job = builder.build_proposer_job(
        dossiers=dossiers,
        planner_job=planner_job,
        planner_response=planner_response,
        lookback=lookback,
    )
    proposer_response = _proposer_response(builder, lookback)

    validated = builder.validate_proposer_response(
        dossiers=dossiers,
        planner_response=planner_response,
        lookback=lookback,
        response=proposer_response,
    )
    frozen = builder.freeze_round(
        dossiers=dossiers,
        planner_job=planner_job,
        planner_response=planner_response,
        lookback=lookback,
        proposer_job=proposer_job,
        proposer_response=proposer_response,
    )

    assert proposer_job.job_kind == CROSS_ARCHITECTURE_INTEGRATION_JOB
    assert [row["operation"] for row in validated["operations"]] == ["add", "drop"]
    assert frozen.proposal == validated
    assert frozen.audit["proposal_frozen_before_next_gate"] is True
    assert frozen.audit["complete_catalog_dump_present"] is False
    assert frozen.still_sealed_gate_fingerprint == "c" * 64


def test_schema_valid_adaptive_planner_wires_normalize_totally_and_revalidate():
    builder = _builder()
    _, _, _, dossiers = _completed_hierarchy(builder)
    job = builder.build_planner_job(dossiers)
    validator = Draft202012Validator(json.loads(job.messages[1]["content"])["output_schema"])
    baseline = _planner_response(dossiers)
    family = baseline["review_targets"][0]["relevant_architectures"][0]
    other_family = next(value for value in ACTIVE_STAGE1_CONCEPT_FAMILIES if value != family)
    evidence_id = baseline["review_targets"][0]["requested_evidence_ids"][0]

    cases = []
    duplicate_family = json.loads(canonical_json(baseline))
    duplicate_family["review_targets"][0]["relevant_architectures"] = [family, family]
    cases.append(duplicate_family)

    duplicate_evidence = json.loads(canonical_json(baseline))
    duplicate_evidence["review_targets"][0]["requested_evidence_ids"] = [
        evidence_id,
        evidence_id,
    ]
    cases.append(duplicate_evidence)

    cross_target_duplicate = json.loads(canonical_json(baseline))
    cross_target_duplicate["review_targets"][1]["requested_evidence_ids"] = [evidence_id]
    cases.append(cross_target_duplicate)

    ownership_mismatch = json.loads(canonical_json(baseline))
    ownership_mismatch["review_targets"][0]["relevant_architectures"] = [other_family]
    cases.append(ownership_mismatch)

    false_without_requests = json.loads(canonical_json(baseline))
    false_without_requests["review_targets"][0]["requested_evidence_ids"] = []
    cases.append(false_without_requests)

    model_authored_policy_word = json.loads(canonical_json(baseline))
    model_authored_policy_word["review_targets"][0]["reason"] = "This is not an oracle assessment."
    cases.append(model_authored_policy_word)

    astral_text = json.loads(canonical_json(baseline))
    astral_text["review_targets"][0]["problem"] = "Review the documented signal 🚀."
    cases.append(astral_text)

    for wire in cases:
        validator.validate(wire)
        normalized = builder.validate_planner_response(dossiers=dossiers, response=wire)
        assert (
            builder.validate_planner_response(dossiers=dossiers, response=normalized) == normalized
        )
        assert (
            normalized["wire_normalization_audit"]["wire_review_targets"] == wire["review_targets"]
        )

    normalized_owner = builder.validate_planner_response(
        dossiers=dossiers,
        response=ownership_mismatch,
    )
    assert family in normalized_owner["review_targets"][0]["relevant_architectures"]
    normalized_empty = builder.validate_planner_response(
        dossiers=dossiers,
        response=false_without_requests,
    )
    assert normalized_empty["no_lookback_needed"] is True
    normalized_policy_word = builder.validate_planner_response(
        dossiers=dossiers,
        response=model_authored_policy_word,
    )
    assert (
        "oracle"
        not in canonical_json(builder._planner_model_view(normalized_policy_word)).casefold()
    )
    policy_lookback = builder.resolve_requested_evidence(
        dossiers=dossiers,
        planner_response=normalized_policy_word,
    )
    builder.build_proposer_job(
        dossiers=dossiers,
        planner_job=job,
        planner_response=normalized_policy_word,
        lookback=policy_lookback,
    )

    true_with_requests = json.loads(canonical_json(baseline))
    true_with_requests["no_lookback_needed"] = True
    assert validator.is_valid(true_with_requests) is False
    over_per_target_bound = json.loads(canonical_json(baseline))
    over_per_target_bound["review_targets"][0]["requested_evidence_ids"] = [
        evidence_id for _ in range(builder.config.max_lookback_ids_per_target + 1)
    ]
    assert validator.is_valid(over_per_target_bound) is False
    empty_problem = json.loads(canonical_json(baseline))
    empty_problem["review_targets"][0]["problem"] = ""
    assert validator.is_valid(empty_problem) is False
    unknown_family = json.loads(canonical_json(baseline))
    unknown_family["review_targets"][0]["relevant_architectures"] = ["unknown_family"]
    assert validator.is_valid(unknown_family) is False
    control_text = json.loads(canonical_json(baseline))
    control_text["review_targets"][0]["reason"] = "line one\nline two"
    assert validator.is_valid(control_text) is False
    surrogate_text = json.loads(canonical_json(baseline))
    surrogate_text["review_targets"][0]["reason"] = "\ud800"
    assert validator.is_valid(surrogate_text) is False


def test_schema_valid_adaptive_proposer_wires_normalize_totally_and_revalidate():
    builder = _builder()
    _, _, _, dossiers = _completed_hierarchy(builder)
    planner_response = _planner_response(dossiers)
    planner_job = builder.build_planner_job(dossiers)
    lookback = builder.resolve_requested_evidence(
        dossiers=dossiers,
        planner_response=planner_response,
    )
    job = builder.build_proposer_job(
        dossiers=dossiers,
        planner_job=planner_job,
        planner_response=planner_response,
        lookback=lookback,
    )
    validator = Draft202012Validator(json.loads(job.messages[1]["content"])["output_schema"])
    baseline = _proposer_response(builder, lookback)
    add = baseline["operations"][0]
    drop = baseline["operations"][1]
    cited_family = add["proposed_feature"]["source_families"][0]
    other_family = next(value for value in ACTIVE_STAGE1_CONCEPT_FAMILIES if value != cited_family)

    cases = []
    target_mismatch = json.loads(canonical_json(baseline))
    target_mismatch["operations"][0]["targets"] = ["different_generated_name"]
    cases.append(target_mismatch)

    registry_collision = json.loads(canonical_json(baseline))
    registry_collision["operations"][0]["targets"] = ["existing_measure"]
    registry_collision["operations"][0]["proposed_feature"]["feature_name"] = "existing_measure"
    cases.append(registry_collision)

    reserved_result = json.loads(canonical_json(baseline))
    reserved_result["operations"][0]["targets"] = [NEW_MISSING_CONSTRUCT]
    reserved_result["operations"][0]["proposed_feature"]["feature_name"] = NEW_MISSING_CONSTRUCT
    cases.append(reserved_result)

    provenance_mismatch = json.loads(canonical_json(baseline))
    provenance_mismatch["operations"][0]["proposed_feature"]["source_families"] = [other_family]
    cases.append(provenance_mismatch)

    duplicate_families = json.loads(canonical_json(baseline))
    duplicate_families["operations"][0]["proposed_feature"]["source_families"] = [
        cited_family,
        cited_family,
    ]
    cases.append(duplicate_families)

    duplicate_results = {
        "operations": [
            json.loads(canonical_json(add)),
            json.loads(canonical_json(add)),
        ],
        "converged": False,
    }
    cases.append(duplicate_results)

    reused_target = {
        "operations": [
            json.loads(canonical_json(drop)),
            json.loads(canonical_json(drop)),
        ],
        "converged": False,
    }
    cases.append(reused_target)

    for kind in ("rename", "split", "revise_definition"):
        relation = json.loads(canonical_json(add))
        relation["operation"] = kind
        relation["targets"] = ["existing_measure"]
        relation["proposed_feature"]["feature_name"] = (
            "changed_measure" if kind == "revise_definition" else "existing_measure"
        )
        cases.append({"operations": [relation], "converged": False})

    model_authored_policy_word = json.loads(canonical_json(baseline))
    model_authored_policy_word["operations"][0]["reason"] = "This is not an oracle assessment."
    cases.append(model_authored_policy_word)

    astral_text = json.loads(canonical_json(baseline))
    astral_text["operations"][0]["reason"] = "Retain the documented signal 🚀."
    cases.append(astral_text)

    for wire in cases:
        validator.validate(wire)
        normalized = builder.validate_proposer_response(
            dossiers=dossiers,
            planner_response=planner_response,
            lookback=lookback,
            response=wire,
        )
        assert (
            builder.validate_proposer_response(
                dossiers=dossiers,
                planner_response=planner_response,
                lookback=lookback,
                response=normalized,
            )
            == normalized
        )

    normalized_collision = builder.validate_proposer_response(
        dossiers=dossiers,
        planner_response=planner_response,
        lookback=lookback,
        response=registry_collision,
    )
    assert normalized_collision["operations"][0]["targets"] == [
        normalized_collision["operations"][0]["proposed_feature"]["feature_name"]
    ]
    assert normalized_collision["operations"][0]["targets"] != ["existing_measure"]
    normalized_reserved = builder.validate_proposer_response(
        dossiers=dossiers,
        planner_response=planner_response,
        lookback=lookback,
        response=reserved_result,
    )
    assert (
        normalized_reserved["operations"][0]["proposed_feature"]["feature_name"]
        != NEW_MISSING_CONSTRUCT
    )

    normalized_reuse = builder.validate_proposer_response(
        dossiers=dossiers,
        planner_response=planner_response,
        lookback=lookback,
        response=reused_target,
    )
    assert len(normalized_reuse["operations"]) == 1
    assert normalized_reuse["wire_normalization_audit"]["dropped_operation_slots"] == [
        {
            "operation_index": 1,
            "reason": "existing_target_already_used_by_earlier_slot",
        }
    ]

    assert validator.is_valid({"operations": [], "converged": False}) is False
    true_with_operations = json.loads(canonical_json(baseline))
    true_with_operations["converged"] = True
    assert validator.is_valid(true_with_operations) is False
    too_many = json.loads(canonical_json(baseline))
    too_many["operations"] = [
        json.loads(canonical_json(add)) for _ in range(builder.config.max_operations + 1)
    ]
    assert validator.is_valid(too_many) is False
    empty_description = json.loads(canonical_json(baseline))
    empty_description["operations"][0]["proposed_feature"]["description"] = ""
    assert validator.is_valid(empty_description) is False
    unknown_operation = json.loads(canonical_json(baseline))
    unknown_operation["operations"][0]["operation"] = "unknown_operation"
    assert validator.is_valid(unknown_operation) is False
    unknown_family = json.loads(canonical_json(baseline))
    unknown_family["operations"][0]["proposed_feature"]["source_families"] = ["unknown_family"]
    assert validator.is_valid(unknown_family) is False
    control_text = json.loads(canonical_json(baseline))
    control_text["operations"][0]["reason"] = "line one\nline two"
    assert validator.is_valid(control_text) is False
    surrogate_text = json.loads(canonical_json(baseline))
    surrogate_text["operations"][0]["reason"] = "\ud800"
    assert validator.is_valid(surrogate_text) is False


def test_proposer_prompt_contains_only_deterministically_requested_raw_atoms():
    builder = _builder()
    _, _, _, dossiers = _completed_hierarchy(builder)
    planner_job = builder.build_planner_job(dossiers)
    planner_response = _planner_response(dossiers)
    lookback = builder.resolve_requested_evidence(
        dossiers=dossiers,
        planner_response=planner_response,
    )
    proposer_job = builder.build_proposer_job(
        dossiers=dossiers,
        planner_job=planner_job,
        planner_response=planner_response,
        lookback=lookback,
    )

    request = json.loads(proposer_job.messages[1]["content"])
    audit = builder.audit_proposer_prompt(
        job=proposer_job,
        dossiers=dossiers,
        lookback=lookback,
    )
    assert [row["evidence_id"] for row in request["requested_evidence"]] == list(
        lookback.requested_evidence_ids
    )
    assert audit["raw_atom_count"] == 1
    assert audit["only_requested_raw_atoms_present"] is True
    assert audit["complete_catalog_dump_present"] is False
    assert '"content":' not in canonical_json(request["architecture_dossiers"])


def test_unknown_lookback_is_rejected_and_complete_catalog_is_preserved_losslessly():
    builder = _builder()
    _, _, _, dossiers = _completed_hierarchy(builder)
    response = _planner_response(dossiers)
    response["review_targets"][0]["requested_evidence_ids"] = ["evidence_unknown"]
    with pytest.raises(ValueError, match="absent from the authenticated dossiers"):
        builder.resolve_requested_evidence(dossiers=dossiers, planner_response=response)

    permissive = _builder(
        config=AdaptiveReconsiderationConfig(
            max_atoms_per_chunk=1,
            max_bytes_per_chunk=20_000,
            max_lookback_ids_per_target=20,
            max_total_lookback_ids=20,
        )
    )
    _, _, _, permissive_dossiers = _completed_hierarchy(permissive)
    all_ids = [atom.evidence_id for atom in permissive.catalog.atoms]
    all_response = {
        "review_targets": [
            {
                "target": NEW_MISSING_CONSTRUCT,
                "problem": "Request every clue.",
                "relevant_architectures": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
                "requested_evidence_ids": all_ids,
                "reason": "Attempt a complete evidence transfer.",
            }
        ],
        "no_lookback_needed": False,
    }
    normalized = permissive.validate_planner_response(
        dossiers=permissive_dossiers,
        response=all_response,
    )
    lookback = permissive.resolve_requested_evidence(
        dossiers=permissive_dossiers,
        planner_response=normalized,
    )
    assert lookback.requested_evidence_ids == tuple(all_ids)
    assert lookback.audit()["all_catalog_atoms_returned"] is True
    assert not any(
        "truncation" in row["action"]
        for row in normalized["wire_normalization_audit"]["normalization_events"]
    )


def test_legacy_byte_bound_never_discards_a_planner_evidence_decision():
    builder = _builder(
        config=AdaptiveReconsiderationConfig(
            max_atoms_per_chunk=1,
            max_bytes_per_chunk=20_000,
            max_total_lookback_bytes=50,
        )
    )
    _, _, _, dossiers = _completed_hierarchy(builder)
    normalized = builder.validate_planner_response(
        dossiers=dossiers,
        response=_planner_response(dossiers),
    )
    lookback = builder.resolve_requested_evidence(
        dossiers=dossiers,
        planner_response=normalized,
    )
    assert lookback.requested_evidence_ids
    assert normalized["no_lookback_needed"] is False
    assert lookback.canonical_size_bytes > 50
    assert not any(
        "truncation" in row["action"]
        for row in normalized["wire_normalization_audit"]["normalization_events"]
    )


def test_planner_normalization_measures_prompt_without_popping_targets_or_evidence():
    builder = _builder(
        config=AdaptiveReconsiderationConfig(
            max_atoms_per_chunk=1,
            max_bytes_per_chunk=20_000,
            max_rendered_prompt_bytes=30_000,
        )
    )
    _, _, _, dossiers = _completed_hierarchy(builder)
    normalized = builder.validate_planner_response(
        dossiers=dossiers,
        response=_planner_response(dossiers),
    )
    lookback = builder.resolve_requested_evidence(
        dossiers=dossiers,
        planner_response=normalized,
    )
    rendered_bytes = builder._proposer_rendered_byte_count(
        dossiers=dossiers,
        planner_response=normalized,
        requested_evidence_ids=lookback.requested_evidence_ids,
    )
    assert not any(
        "truncation" in row["action"]
        for row in normalized["wire_normalization_audit"]["normalization_events"]
    )
    assert normalized["wire_normalization_audit"]["rendered_proposer_bytes"] == rendered_bytes


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"aggregate_metrics": {"true_ite_mean": 0.4}}, "forbidden"),
        ({"aggregate_metrics": {"raw_rows": 640}}, "forbidden"),
        ({"summary": "Compare against the current date."}, "forbidden"),
    ],
)
def test_diagnostics_reject_oracle_row_and_temporal_policy_context(kwargs, match):
    base = {
        "diagnostic_id": "diagnostic_unsafe",
        "diagnostic_kind": "heterogeneity",
        "affected_features": (),
        "summary": "A safe aggregate diagnostic.",
        "aggregate_metrics": {},
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        AdaptiveDiagnostic(**base)


def test_exact_spent_authentication_cannot_be_reused_for_another_catalog():
    first = _catalog(outer_fold=1)
    second = _catalog(outer_fold=2)
    with pytest.raises(ValueError, match="another catalog"):
        AdaptiveHierarchicalStage1Reconsideration(
            catalog=second,
            exact_spent_authentication=_authentication(first),
            family_explanations={
                family: f"Interpret {family}." for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            },
            current_registry=(),
            diagnostics=(
                AdaptiveDiagnostic(
                    diagnostic_id="diagnostic_001",
                    diagnostic_kind="source_preservation",
                    affected_features=(),
                    summary="One aggregate architecture check.",
                ),
            ),
        )


def test_actionable_family_coverage_blocks_cross_family_dossiers():
    builder = _builder()
    interpretations = _interpretation_responses(builder)
    consolidations = _consolidation_responses(builder, interpretations)
    coverage = _coverage_responses(builder, interpretations, consolidations)
    first_job = builder.build_coverage_jobs(
        interpretation_responses=interpretations,
        consolidation_responses=consolidations,
    )[0]
    request = json.loads(first_job.messages[1]["content"])
    first_name = request["family_consolidation"]["canonical_concepts"][0]["canonical_name"]
    coverage[first_job.job_id] = {
        "findings": [
            {
                "action": "split_concept",
                "affected_canonical_names": [first_name],
                "proposed_name": f"{first_name}_subtype",
                "description": "A distinct patient-level measurement.",
                "supporting_evidence_ids": [request["evidence"][0]["evidence_id"]],
                "reason": "The architecture-local audit found an improper merge.",
            }
        ],
        "reviewed_evidence_ids": {row["evidence_id"]: True for row in request["evidence"]},
    }
    with pytest.raises(AdaptiveCoverageRequiresRevision):
        builder.compile_dossiers(
            interpretation_responses=interpretations,
            consolidation_responses=consolidations,
            coverage_responses=coverage,
        )


def test_addition_must_cite_current_requested_evidence():
    builder = _builder()
    _, _, _, dossiers = _completed_hierarchy(builder)
    planner_response = _planner_response(dossiers)
    lookback = builder.resolve_requested_evidence(
        dossiers=dossiers,
        planner_response=planner_response,
    )
    response = _proposer_response(builder, lookback)
    response["operations"][0]["supporting_evidence_ids"] = [
        builder.current_registry[0].supporting_evidence_ids[0]
    ]
    with pytest.raises(ValueError, match="current exact-scope requested evidence"):
        builder.validate_proposer_response(
            dossiers=dossiers,
            planner_response=planner_response,
            lookback=lookback,
            response=response,
        )


def test_targeted_definition_revision_rejects_unmaterialized_historical_support():
    builder = _builder()
    _, _, _, dossiers = _completed_hierarchy(builder)
    family = builder.current_registry[0].source_families[0]
    planner_response = {
        "review_targets": [
            {
                "target": "existing_measure",
                "problem": "The current extraction definition needs a bounded revision.",
                "relevant_architectures": [family],
                "requested_evidence_ids": [],
                "reason": "The registry and aggregate diagnostic are sufficient.",
            }
        ],
        "no_lookback_needed": True,
    }
    lookback = builder.resolve_requested_evidence(
        dossiers=dossiers,
        planner_response=planner_response,
    )
    response = {
        "operations": [
            {
                "operation": "revise_definition",
                "targets": ["existing_measure"],
                "proposed_feature": {
                    "feature_name": "existing_measure",
                    "description": "A currently extracted patient measurement.",
                    "value_shape_hypothesis": "ambiguous",
                    "definition_summary": "Use a more explicit documented mention rule.",
                    "source_families": [family],
                },
                "supporting_evidence_ids": list(
                    builder.current_registry[0].supporting_evidence_ids
                ),
                "diagnostic_ids": ["diagnostic_001"],
                "reason": "The bounded definition revision addresses aggregate missingness.",
            }
        ],
        "converged": False,
    }

    assert lookback.items == ()
    with pytest.raises(ValueError, match="current exact-scope requested evidence"):
        builder.validate_proposer_response(
            dossiers=dossiers,
            planner_response=planner_response,
            lookback=lookback,
            response=response,
        )


@pytest.mark.parametrize(
    ("operation_kind", "expected_names"),
    [
        (
            "add",
            (
                "documented_bow_nuisance_clue",
                "documented_bow_r_loss_clue",
                "documented_bow_nuisance_clue_002",
            ),
        ),
        ("drop", ("documented_bow_r_loss_clue",)),
        (
            "split",
            (
                "documented_bow_nuisance_clue",
                "documented_bow_nuisance_clue_002",
                "documented_bow_r_loss_clue",
            ),
        ),
        (
            "rename",
            (
                "documented_bow_nuisance_clue_002",
                "documented_bow_r_loss_clue",
            ),
        ),
        (
            "revise_definition",
            (
                "documented_bow_nuisance_clue",
                "documented_bow_r_loss_clue",
            ),
        ),
        ("merge", ("documented_clue",)),
    ],
)
def test_all_six_operations_compile_into_frozen_executable_contracts(
    operation_kind, expected_names
):
    builder, _, frozen_round, lookback, definition_jobs = _prepare_operation_case(operation_kind)
    expected_definition_count = 0 if operation_kind == "drop" else 1

    assert len(definition_jobs) == expected_definition_count
    for job, request in definition_jobs:
        assert job.job_kind == EXTRACTION_DEFINITION_JOB
        assert job.settings == DiscoveryJobSettings.extraction()
        assert job.input_bindings["definition_thinking_enabled"] is False
        assert job.input_bindings["supporting_evidence_ids"] == list(
            request.supporting_evidence_ids
        )
    definition_responses = {
        job.job_id: _definition_response(request) for job, request in definition_jobs
    }
    revision = builder.freeze_executable_revision(
        current_specs=_executable_specs(),
        frozen_round=frozen_round,
        lookback=lookback,
        definition_responses=definition_responses,
        max_contracts=4,
    )

    assert tuple(spec["name"] for spec in revision.applied.specs) == expected_names
    assert revision.audit["definition_jobs_thinking_enabled"] is False
    assert revision.audit["roles_routed_from_exact_cited_evidence_axes"] is True
    assert revision.applied.operation_audit[0]["adaptive_operation"] == operation_kind
    if operation_kind == "drop":
        assert revision.definition_job_ids == ()
        assert revision.applied.operation_audit[0]["contract"] is None
    else:
        operation_audit = revision.applied.operation_audit[0]
        assert operation_audit["contract"]["roles"] == ["confounder"]
        assert all(row["supported"] for row in operation_audit["evidence_contract_grounding"])


def test_executable_definition_job_uses_configured_nonlegacy_response_contract():
    wire_budget = replace(
        LEGACY_HIERARCHY_WIRE_BUDGET,
        max_generated_name_chars=40,
        max_interpret_name_chars=40,
        max_free_text_chars=73,
    )
    config = AdaptiveReconsiderationConfig(
        max_atoms_per_chunk=1,
        max_bytes_per_chunk=20_000,
        wire_budget=wire_budget,
    )
    builder, _, _, _, definition_jobs = _prepare_operation_case(
        "add",
        config=config,
    )

    assert builder.config.wire_budget == wire_budget
    assert len(definition_jobs) == 1
    request = json.loads(definition_jobs[0][0].messages[1]["content"])
    assert request["hierarchy_wire_budget"] == wire_budget.as_dict()
    assert request["hierarchy_wire_budget"] != LEGACY_HIERARCHY_WIRE_BUDGET.as_dict()
    assert request["output_schema"]["properties"]["measurement"]["maxLength"] == 73
    assert (
        request["output_schema"]["properties"]["aliases"]["items"]["maxLength"]
        == 73
    )


def test_executable_freeze_revalidates_normalized_definition_without_weakening_wire_shape():
    builder, _, frozen_round, lookback, definition_jobs = _prepare_operation_case("add")
    job, request = definition_jobs[0]
    normalized = builder.validate_extraction_definition_job_response(
        job=job,
        request=request,
        response=_definition_response(request),
    )
    assert isinstance(normalized["supporting_evidence_ids"], list)

    revision = builder.freeze_executable_revision(
        current_specs=_executable_specs(),
        frozen_round=frozen_round,
        lookback=lookback,
        definition_responses={job.job_id: normalized},
        max_contracts=4,
    )
    assert "documented_bow_nuisance_clue_002" in {spec["name"] for spec in revision.applied.specs}


def test_empty_consolidation_keeps_wire_and_normalized_shapes_separate():
    builder = _builder()
    wire = {"candidate_assignments": {}, "slot_definitions": {}}
    normalized = {
        "canonical_concepts": [],
        "candidate_dispositions": [],
        "wire_normalization_audit": {
            "audit_version": "fixed_slot_consolidation_normalization_audit_v1",
            "slot_policy": "exact_candidate_assignment_then_compiler_derived_groups_v1",
            "derived_relation_fields": [
                "active_slots",
                "canonical_concepts.member_candidate_ids",
                "canonical_concepts.supporting_evidence_ids",
                "canonical_concepts.source_families",
                "canonical_concepts.value_shape_hypothesis",
                "candidate_dispositions.canonical_name",
            ],
            "candidate_slot_assignments": [],
            "slot_definitions": [],
            "active_slots": [],
            "unused_slots": [],
            "canonical_name_disambiguations": [],
        },
    }
    assert (
        builder._validate_empty_consolidation_wire(wire) == normalized
    )
    with pytest.raises(ValueError, match="consolidation response keys differ"):
        builder._validate_empty_consolidation_wire(normalized)
    assert (
        builder._revalidate_empty_consolidation_projection(normalized)
        == normalized
    )


def test_converged_noop_has_no_definition_jobs_and_preserves_specs():
    builder = _executable_builder()
    _, _, _, dossiers = _completed_hierarchy(builder)
    planner_response = {"review_targets": [], "no_lookback_needed": True}
    planner_job = builder.build_planner_job(dossiers)
    lookback = builder.resolve_requested_evidence(
        dossiers=dossiers,
        planner_response=planner_response,
    )
    proposer_job = builder.build_proposer_job(
        dossiers=dossiers,
        planner_job=planner_job,
        planner_response=planner_response,
        lookback=lookback,
    )
    frozen_round = builder.freeze_round(
        dossiers=dossiers,
        planner_job=planner_job,
        planner_response=planner_response,
        lookback=lookback,
        proposer_job=proposer_job,
        proposer_response={"operations": [], "converged": True},
    )

    assert lookback.items == ()
    assert frozen_round.proposal["operations"] == []
    assert frozen_round.proposal["converged"] is True
    assert frozen_round.proposal["wire_normalization_audit"]["wire_operations"] == []
    assert (
        builder.build_extraction_definition_jobs(
            frozen_round=frozen_round,
            lookback=lookback,
        )
        == ()
    )
    revision = builder.freeze_executable_revision(
        current_specs=_executable_specs(),
        frozen_round=frozen_round,
        lookback=lookback,
        definition_responses={},
        max_contracts=4,
    )

    assert revision.definition_job_ids == ()
    assert revision.definition_response_sha256s == ()
    assert revision.applied.specs == _executable_specs()
    assert revision.applied.operation_audit == ()


def test_thinking_off_definition_rejects_temporal_policy_text():
    builder, _, _, _, definition_jobs = _prepare_operation_case("add")
    job, request = definition_jobs[0]
    response = _definition_response(request)
    response["measurement"] = "Compare the clue against the current date."

    assert job.settings == DiscoveryJobSettings.extraction()
    with pytest.raises(ValueError, match="forbidden adaptive model text"):
        builder.validate_extraction_definition_job_response(
            job=job,
            request=request,
            response=response,
        )


def test_definition_prompt_rejects_mutated_static_vocabulary_policy():
    _, _, _, _, definition_jobs = _prepare_operation_case("add")
    job, _ = definition_jobs[0]
    payload = json.loads(job.messages[1]["content"])
    payload["vocabulary_grounding_policy"]["clinical_vocabulary"] = "mutated_policy"
    messages = (
        job.messages[0],
        {"role": "user", "content": canonical_json(payload)},
    )

    with pytest.raises(ValueError, match="static vocabulary_grounding_policy literal"):
        adaptive_module._assert_adaptive_job_prompt_contract(
            job_kind=job.job_kind,
            messages=messages,
            settings=job.settings,
            selector_thinking_token_budget=SELECTOR_THINKING_TOKEN_BUDGET,
        )


def test_merge_with_incomplete_family_support_is_compiler_dropped():
    builder = _executable_builder()
    _, _, _, dossiers = _completed_hierarchy(builder)
    planner_response, proposer_response = _operation_plan_and_proposal(builder, "merge")
    planner_response["review_targets"][1]["requested_evidence_ids"] = []
    proposer_response["operations"][0]["supporting_evidence_ids"] = list(
        planner_response["review_targets"][0]["requested_evidence_ids"]
    )
    lookback = builder.resolve_requested_evidence(
        dossiers=dossiers,
        planner_response=planner_response,
    )

    normalized = builder.validate_proposer_response(
        dossiers=dossiers,
        planner_response=planner_response,
        lookback=lookback,
        response=proposer_response,
    )
    assert normalized["operations"] == []
    assert normalized["converged"] is True
    assert normalized["wire_normalization_audit"]["dropped_operation_slots"] == [
        {
            "operation_index": 0,
            "reason": "merge_lacks_evidence_for_retained_architectures",
        }
    ]


def test_execute_authenticated_cache_miss_then_hit_revalidates_semantics(tmp_path, monkeypatch):
    cache_root = tmp_path / "adaptive-cache"
    runner = _ConvergedAdaptiveRunner()
    first_builder = _builder()
    first_cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=cache_root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    approved_adaptive = adaptive_hierarchical_stage1_reconsideration_identity(first_builder.config)
    approved_runner = runner.identity()
    approved_cache = first_cache.identity()

    first = first_builder.execute_authenticated(
        runner=runner,
        job_cache=first_cache,
        approved_adaptive_identity=approved_adaptive,
        approved_runner_identity=approved_runner,
        approved_cache_identity=approved_cache,
        current_specs=_existing_measure_specs(),
        max_contracts=4,
    )
    remote_call_count = len(runner.calls)

    assert remote_call_count > 0
    assert all(
        row["outcome"] == "remote_validated_and_cached" for row in first.audit["job_records"]
    )
    assert tuple(spec["name"] for spec in first.executable_revision.applied.specs) == (
        "existing_measure",
    )

    second_builder = _builder()
    second_cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=cache_root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    semantic_revalidations = []
    original_validator = second_builder.validate_interpretation_job_response

    def counting_validator(*, job, response):
        semantic_revalidations.append(job.job_id)
        return original_validator(job=job, response=response)

    monkeypatch.setattr(
        second_builder,
        "validate_interpretation_job_response",
        counting_validator,
    )
    second = second_builder.execute_authenticated(
        runner=runner,
        job_cache=second_cache,
        approved_adaptive_identity=approved_adaptive,
        approved_runner_identity=approved_runner,
        approved_cache_identity=approved_cache,
        current_specs=_existing_measure_specs(),
        max_contracts=4,
    )

    assert len(runner.calls) == remote_call_count
    assert semantic_revalidations == [job.job_id for job in second_builder.interpret_jobs]
    assert all(row["outcome"] == "authenticated_cache_hit" for row in second.audit["job_records"])
    assert second.audit["remote_execution_record_count"] == 0
    assert second.execution_sha256 != first.execution_sha256


def test_oversized_adaptive_family_uses_phased_jobs_and_replays_from_cache(tmp_path):
    catalog = _catalog(first_family_atom_count=2)
    first_builder = _builder(catalog=catalog)
    dense_manual = _dense_interpretation_responses(
        first_builder,
        dense_candidate_count=16,
    )
    with pytest.raises(ValueError, match="compile bounded pages"):
        first_builder.build_consolidation_jobs(dense_manual)

    cache_root = tmp_path / "adaptive-dense-cache"
    runner = _DenseAdaptiveRunner(dense_candidate_count=16)
    first_cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=cache_root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    approved_adaptive = adaptive_hierarchical_stage1_reconsideration_identity(first_builder.config)
    approved_runner = runner.identity()
    approved_cache = first_cache.identity()
    first = first_builder.execute_authenticated(
        runner=runner,
        job_cache=first_cache,
        approved_adaptive_identity=approved_adaptive,
        approved_runner_identity=approved_runner,
        approved_cache_identity=approved_cache,
        current_specs=_existing_measure_specs(),
        max_contracts=4,
    )
    remote_call_count = len(runner.calls)
    relation_jobs = [
        job
        for job in runner.jobs
        if json.loads(job.messages[1]["content"])["job"] == "compare_adaptive_candidate_relations"
    ]
    atomic_jobs = [
        job
        for job in runner.jobs
        if json.loads(job.messages[1]["content"])["job"] == "audit_adaptive_atomic_coverage"
    ]
    assert relation_jobs
    assert atomic_jobs
    assert first.audit["all_candidate_pairs_and_coverage_pages_compiled_without_truncation"] is True
    assert sum(
        record["record_type"] == "adaptive_family_consolidation_compilation"
        for record in first.audit["compiler_records"]
    ) == len(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    assert (
        first_builder.offline_contract["assurances"]["candidate_or_coverage_decision_truncation"]
        is False
    )

    second_builder = _builder(catalog=catalog)
    second_cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=cache_root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    second = second_builder.execute_authenticated(
        runner=runner,
        job_cache=second_cache,
        approved_adaptive_identity=approved_adaptive,
        approved_runner_identity=approved_runner,
        approved_cache_identity=approved_cache,
        current_specs=_existing_measure_specs(),
        max_contracts=4,
    )
    assert len(runner.calls) == remote_call_count
    assert second.audit["remote_execution_record_count"] == 0
    assert all(row["outcome"] == "authenticated_cache_hit" for row in second.audit["job_records"])


def test_execute_authenticated_uses_one_projection_bound_response_repair(tmp_path):
    builder = _builder()
    runner = _InvalidFirstAdaptiveRunner()
    cache_root = tmp_path / "adaptive-repair-cache"
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=cache_root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )

    result = builder.execute_authenticated(
        runner=runner,
        job_cache=cache,
        approved_adaptive_identity=(
            adaptive_hierarchical_stage1_reconsideration_identity(builder.config)
        ),
        approved_runner_identity=runner.identity(),
        approved_cache_identity=cache.identity(),
        current_specs=_existing_measure_specs(),
        max_contracts=4,
    )

    first = result.audit["job_records"][0]
    assert len(first["remote_record_sha256s"]) == 2
    assert len(runner.calls) == len(result.audit["job_records"]) + 1
    assert result.audit["remote_execution_record_count"] == len(runner.calls)
    assert all(
        "not one JSON object" not in path.read_text(encoding="utf-8")
        for path in cache_root.glob("*/*.json")
    )


def test_execute_authenticated_rejects_adaptive_approval_mismatch_before_cache_or_runner(
    tmp_path,
):
    builder = _builder()
    runner = _ConvergedAdaptiveRunner()
    cache_root = tmp_path / "unused-adaptive-cache"
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=cache_root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    mismatched = adaptive_hierarchical_stage1_reconsideration_identity(builder.config)
    mismatched["implementation_file_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="offline-approved identity"):
        builder.execute_authenticated(
            runner=runner,
            job_cache=cache,
            approved_adaptive_identity=mismatched,
            approved_runner_identity=runner.identity(),
            approved_cache_identity=cache.identity(),
            current_specs=_existing_measure_specs(),
            max_contracts=4,
        )

    assert runner.calls == []
    assert not cache_root.exists()


def test_execute_authenticated_requires_runner_raw_wire_sha_before_cache_write(tmp_path):
    builder = _builder()
    runner = _MissingParsedResponseShaRunner()
    cache_root = tmp_path / "missing-raw-wire-sha-cache"
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=cache_root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )

    with pytest.raises(ValueError, match="parsed_response_sha256"):
        builder.execute_authenticated(
            runner=runner,
            job_cache=cache,
            approved_adaptive_identity=(
                adaptive_hierarchical_stage1_reconsideration_identity(builder.config)
            ),
            approved_runner_identity=runner.identity(),
            approved_cache_identity=cache.identity(),
            current_specs=_existing_measure_specs(),
            max_contracts=4,
        )

    assert len(runner.calls) == 1
    if cache_root.exists():
        assert not tuple(cache_root.glob("*/*.json"))


def test_execute_authenticated_rejects_implementation_bundle_mismatch_fail_closed(
    tmp_path, monkeypatch
):
    builder = _builder()
    original_bundle = adaptive_module.adaptive_hierarchical_implementation_bundle

    def mismatched_bundle():
        current = original_bundle()
        files = dict(current["files"])
        files["all_evidence_fusion.py"] = "0" * 64
        body = {
            "schema_version": current["schema_version"],
            "files": files,
        }
        return {**body, "implementation_bundle_sha256": content_sha256(body)}

    monkeypatch.setattr(
        adaptive_module,
        "adaptive_hierarchical_implementation_bundle",
        mismatched_bundle,
    )
    approved_adaptive = adaptive_hierarchical_stage1_reconsideration_identity(builder.config)
    runner = _ConvergedAdaptiveRunner()
    cache_root = tmp_path / "bundle-mismatch-cache"
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=cache_root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )

    with pytest.raises(ValueError, match="dependency bundle changed before"):
        builder.execute_authenticated(
            runner=runner,
            job_cache=cache,
            approved_adaptive_identity=approved_adaptive,
            approved_runner_identity=runner.identity(),
            approved_cache_identity=cache.identity(),
            current_specs=_existing_measure_specs(),
            max_contracts=4,
        )

    assert runner.calls == []
    assert not cache_root.exists()
