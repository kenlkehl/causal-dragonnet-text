from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import oci.inference.hierarchical_all_architecture_discovery as hierarchy_module

from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    AS_DOCUMENTED_UNIT,
    OUTCOME_AXIS,
    DiscoveryCandidate,
    canonical_json,
    content_sha256,
    validate_interpret_evidence_chunk_response,
)
from oci.inference.hierarchical_discovery_compiler import (
    compile_hierarchical_discovery,
)
from oci.inference.frozen_hierarchical_review_evidence import (
    FrozenHierarchicalReviewEvidenceConfig,
    freeze_hierarchical_review_evidence,
)
from oci.inference.hierarchical_all_architecture_discovery import (
    AUTHENTICATED_MESSAGE_ENVELOPE_BINDING,
    AUTHENTICATED_RESPONSE_CONTRACT_BINDING,
    CONSOLIDATE_ARCHITECTURE_JOB,
    COVERAGE_CRITIC_JOB,
    CROSS_ARCHITECTURE_INTEGRATION_JOB,
    CROSS_ARCHITECTURE_PLANNER_JOB,
    EXTRACTION_DEFINITION_JOB,
    HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING,
    INTERPRET_CHUNK_JOB,
    MAX_RENDERED_DISCOVERY_PROMPT_BYTES,
    REJECTION_CRITIC_JOB,
    SELECTOR_THINKING_TOKEN_BUDGET,
    CoverageCriticRequiresRevision,
    DirectNumericalDossierBinding,
    DiscoveryJsonJob,
    DiscoveryResponseRepairExhausted,
    HierarchicalAllArchitectureDiscoveryOrchestrator,
    HierarchicalDiscoveryConfig,
    ValidatedDiscoveryJobResult,
    hierarchical_discovery_implementation_bundle,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
    SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION,
    RoleNeutralEvidenceCatalog,
    Stage1EvidenceAtom,
    build_complete_architecture_chunks,
    validate_role_neutral_catalog,
)
from oci.inference.stage1_architecture_explanations import (
    PRODUCTION_STAGE1_FAMILY_EXPLANATIONS,
    production_stage1_family_explanations,
)
from oci.inference.openai_compatible_json_discovery_job_runner import (
    InvalidDiscoveryJsonResponse,
)


def _catalog(*, first_family_atom_count: int = 1) -> RoleNeutralEvidenceCatalog:
    split_fingerprint = "1" * 64
    semantic_member_batch_size = 1
    semantic_member_batching = {
        "schema_version": SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION,
        "semantic_member_batch_size": semantic_member_batch_size,
        "selection_or_truncation_authorized": False,
        "complete_member_coverage_required": True,
    }
    atoms = []
    atom_ordinal = 0
    for family_index, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES, start=1):
        family_atom_count = first_family_atom_count if family_index == 1 else 1
        for family_atom_index in range(1, family_atom_count + 1):
            atom_ordinal += 1
            member_id = f"member_{atom_ordinal:03d}"
            origin = {
                "source": f"closed_{family}",
                "ordinal": atom_ordinal,
                "family_atom_index": family_atom_index,
            }
            content = {
                "terms": [
                    {
                        "member_id": member_id,
                        "term": (f"{family} patient measurement clue {family_atom_index:03d}"),
                    }
                ]
            }
            origin_sha = content_sha256(origin)
            content_sha = content_sha256(content)
            identity = {
                "atom_kind": "test_term_atom",
                "source_kind": f"{family}_test_source",
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
                    atom_kind="test_term_atom",
                    source_kind=f"{family}_test_source",
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
                "semantic_member_batch_size": semantic_member_batch_size,
            }
        ),
    )
    validate_role_neutral_catalog(catalog)
    return catalog


def _bindings():
    return tuple(
        DirectNumericalDossierBinding(
            source_family=family,
            manifest_sha256="f" * 64,
            signal_count=index,
        )
        for index, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES, start=1)
    )


def _orchestrator(
    *,
    max_lookback: int = 1,
    first_family_atom_count: int = 1,
    config: HierarchicalDiscoveryConfig | None = None,
) -> HierarchicalAllArchitectureDiscoveryOrchestrator:
    catalog = _catalog(first_family_atom_count=first_family_atom_count)
    selected_config = (
        config
        if config is not None
        else HierarchicalDiscoveryConfig(max_cross_architecture_lookback_ids_per_group=max_lookback)
    )
    plan = build_complete_architecture_chunks(
        catalog,
        max_atoms_per_chunk=1,
        max_bytes_per_chunk=20_000,
        max_semantic_member_ids_per_chunk=(selected_config.max_semantic_member_ids_per_chunk),
    )
    return HierarchicalAllArchitectureDiscoveryOrchestrator(
        catalog=catalog,
        chunk_plan=plan,
        family_explanations={
            family: f"Interpret the concept-bearing evidence emitted by {family}."
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        direct_numerical_bindings=_bindings(),
        runner_identity={"runner": "deterministic_offline_test_v1", "model": "fixed_json"},
        config=selected_config,
    )


def _json_keys(value):
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_json_keys(child))
        return keys
    if isinstance(value, list):
        keys = set()
        for child in value:
            keys.update(_json_keys(child))
        return keys
    return set()


class _DeterministicRunner:
    def __init__(
        self,
        *,
        bad_first_member_disposition: bool = False,
        coverage_revision: bool = False,
        coverage_revision_once: bool = False,
        omit_integration_disposition: bool = False,
        omit_rejection_reconsideration: bool = False,
        rejection_revision: str | None = None,
    ) -> None:
        self.bad_first_member_disposition = bad_first_member_disposition
        self.coverage_revision = coverage_revision
        self.coverage_revision_once = coverage_revision_once
        self.coverage_revision_count = 0
        self.omit_integration_disposition = omit_integration_disposition
        self.omit_rejection_reconsideration = omit_rejection_reconsideration
        self.rejection_revision = rejection_revision
        self.calls = []

    def identity(self):
        return {"runner": "deterministic_offline_test_v1", "model": "fixed_json"}

    def run_json(self, *, job):
        self.calls.append(job)
        request = json.loads(job.messages[1]["content"])
        if request["job"] == "compare_consolidation_candidate_relations":
            anchor = request["anchor_candidate"]
            return {
                "comparisons": {
                    peer["candidate_id"]: {
                        "relation": (
                            "same_construct"
                            if peer["feature_name"] == anchor["feature_name"]
                            else "distinct"
                        ),
                        "reason": "Compare the bounded candidate descriptors directly.",
                    }
                    for peer in request["peer_candidates"]
                }
            }
        if request["job"] == "fold_consolidation_group_definition":
            prior = request["prior_accumulator"]
            first = request["fresh_candidates"][0]
            return {
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
                "reason": "Fold every member of the compiler-proven group.",
            }
        if request["job"] == "compare_cross_architecture_candidate_relations":
            anchor = request["anchor_candidate"]
            return {
                "comparisons": {
                    peer["candidate_id"]: {
                        "relation": (
                            "same_construct"
                            if peer["feature_name"] == anchor["feature_name"]
                            else "distinct"
                        ),
                        "reason": "Compare compact cross-architecture descriptors.",
                    }
                    for peer in request["peer_candidates"]
                }
            }
        if request["job"] == "fold_cross_architecture_group_definition":
            prior = request["prior_accumulator"]
            first = request["fresh_candidates"][0]
            return {
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
                "reason": "Fold every cross-architecture group member.",
            }
        if request["job"] == "integrate_cross_architecture_group":
            family = request["compiler_owned_relations"]["source_families"][0]
            rejected = family == ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
            provisional = request["provisional_definition"]
            return {
                "decision": "reject" if rejected else "accept",
                "canonical_name": "" if rejected else provisional["canonical_name"],
                "description": "" if rejected else provisional["description"],
                "unresolved_ambiguity": ("" if rejected else provisional["unresolved_ambiguity"]),
                "reason": (
                    "The clue remains nonspecific after bounded group review."
                    if rejected
                    else "The group is a supported patient measurement."
                ),
            }
        if request["job"] == "review_integration_group_evidence":
            return {
                "relationship": "supports_group",
                "proposed_distinct_name": "",
                "measurement_summary": "The raw item supports the provisional measurement.",
                "unresolved_ambiguity": "",
                "reason": "Exercise one exact raw-support review page.",
                "reviewed_evidence": True,
            }
        if request["job"] == "fold_integration_group_evidence_reviews":
            provisional = request["provisional_definition"]
            rejected = provisional["canonical_name"].startswith(ACTIVE_STAGE1_CONCEPT_FAMILIES[0])
            return {
                "decision": "reject" if rejected else "accept",
                "canonical_name": "" if rejected else provisional["canonical_name"],
                "description": "" if rejected else provisional["description"],
                "unresolved_ambiguity": ("" if rejected else provisional["unresolved_ambiguity"]),
                "input_dispositions": (
                    {}
                    if self.omit_integration_disposition
                    else {
                        review_id: {
                            "action": "integrated",
                            "reason": "The exact review input was folded.",
                        }
                        for review_id in request["review_input_ids"]
                    }
                ),
                "complete_support_reviewed": True,
                "reason": (
                    "The clue remains nonspecific after exhaustive review."
                    if rejected
                    else "The complete support establishes one patient measurement."
                ),
            }
        if request["job"] == "review_rejection_candidate_evidence":
            decision = self.rejection_revision
            return {
                "signal": (f"supports_{decision}" if decision is not None else "supports_uphold"),
                "proposed_name": ("restored_patient_measurement" if decision is not None else ""),
                "measurement_summary": "The raw item was independently reconsidered.",
                "reason": "Exercise one exact rejection-support review page.",
                "reviewed_evidence": True,
            }
        if request["job"] == "fold_rejection_candidate_evidence_reviews":
            decision = self.rejection_revision or "uphold"
            return {
                "decision": decision,
                "proposed_name": (
                    "restored_patient_measurement" if self.rejection_revision else ""
                ),
                "measurement_summary": "Every candidate-support page was folded.",
                "input_dispositions": (
                    {}
                    if self.omit_rejection_reconsideration
                    else {
                        review_id: {
                            "action": "integrated",
                            "reason": "The exact rejection review input was folded.",
                        }
                        for review_id in request["review_input_ids"]
                    }
                ),
                "complete_support_reviewed": True,
                "reason": (
                    "Complete raw support warrants deterministic restoration."
                    if self.rejection_revision
                    else "Complete raw support agrees that rejection should be upheld."
                ),
            }
        if request["job"] == "review_extraction_feature_evidence":
            return {
                "measurement_observation": "One supported patient measurement is present.",
                "shape_observation": "unresolved",
                "literal_aliases": [],
                "literal_units": [],
                "literal_categories": [],
                "literal_distinctions": [],
                "missing_or_ambiguous": "The exact representation remains unresolved.",
                "reviewed_evidence": True,
            }
        if request["job"] == "fold_extraction_evidence_definitions":
            return {
                "feature_name": request["canonical_name"],
                "measurement": "Extract this one supported patient measurement.",
                "representation": {"kind": "unresolved", "unit": "", "categories": []},
                "aliases": [],
                "distinguish_from": [],
                "missing_or_ambiguous": "Return null when absent or ambiguous.",
                "input_dispositions": {
                    review_id: {
                        "action": "integrated",
                        "reason": "The exact extraction review input was folded.",
                    }
                    for review_id in request["review_input_ids"]
                },
                "supporting_evidence_reviewed": True,
            }
        if request["job"] == "audit_architecture_atomic_coverage":
            return {
                "findings": [],
                "reviewed_atomic_review": True,
            }
        if job.job_kind == INTERPRET_CHUNK_JOB:
            evidence = request["evidence"]
            item = evidence[0]
            family = item["source_family"]
            name = f"{family}_measure"
            member_dispositions = {member_id: {"findings": []} for member_id in item["member_ids"]}
            if self.bad_first_member_disposition and len(self.calls) == 1:
                member_dispositions = {}
            return {
                "evidence_dispositions": {
                    item["evidence_id"]: {
                        "evidence_findings": [
                            {
                                "feature_name": name,
                                "description": (
                                    f"Patient-level measurement suggested by {family}."
                                ),
                                "value_shape_hypothesis": "ambiguous",
                                "unresolved_ambiguity": (
                                    "Exact representation remains unresolved."
                                ),
                            }
                        ],
                        "member_dispositions": member_dispositions,
                        "reason": "The supplied member grounds this measurement.",
                    }
                },
            }
        if job.job_kind == CONSOLIDATE_ARCHITECTURE_JOB:
            candidates = request["candidates"]
            slots = request["identifier_ownership"]["identifier_domains"]["cluster_slots"]
            slot_by_name = {}
            for candidate in candidates:
                slot_by_name.setdefault(candidate["feature_name"], slots[len(slot_by_name)])
            candidate_by_slot = {
                slot_by_name[candidate["feature_name"]]: candidate for candidate in candidates
            }
            return {
                "candidate_assignments": {
                    candidate["candidate_id"]: {
                        "cluster_slot": slot_by_name[candidate["feature_name"]],
                        "reason": "The architecture candidate remains distinct.",
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
        if job.job_kind == COVERAGE_CRITIC_JOB:
            if self.coverage_revision and (
                not self.coverage_revision_once or self.coverage_revision_count == 0
            ):
                self.coverage_revision_count += 1
                first = request["consolidation"]["canonical_concepts"][0]
                return {
                    "findings": [
                        {
                            "action": "split_concept",
                            "affected_canonical_names": [first["canonical_name"]],
                            "proposed_name": f"{first['canonical_name']}_subtype",
                            "description": "A distinct patient-level subtype.",
                            "supporting_evidence_ids": [request["evidence"][0]["evidence_id"]],
                            "reason": "The local audit requests an explicit split.",
                        }
                    ],
                    "reviewed_evidence_ids": {
                        row["evidence_id"]: True for row in request["evidence"]
                    },
                }
            return {
                "findings": [],
                "reviewed_evidence_ids": {row["evidence_id"]: True for row in request["evidence"]},
            }
        if job.job_kind == CROSS_ARCHITECTURE_PLANNER_JOB:
            dossiers = request["architecture_dossiers"]
            candidates = [
                candidate
                for dossier in dossiers
                for candidate in dossier["architecture_candidates"]
            ]
            group_slots = request["identifier_ownership"]["identifier_domains"][
                "planner_group_slots"
            ]
            lookback_slots = request["identifier_ownership"]["identifier_domains"][
                "planner_lookback_slots"
            ]
            group_slot_by_candidate = {
                candidate["candidate_id"]: group_slots[index]
                for index, candidate in enumerate(candidates)
            }
            candidate_by_group_slot = {
                group_slot_by_candidate[candidate["candidate_id"]]: candidate
                for candidate in candidates
            }
            return {
                "candidate_assignments": {
                    candidate["candidate_id"]: {
                        "group_slot": group_slot_by_candidate[candidate["candidate_id"]],
                    }
                    for candidate in candidates
                },
                "group_slot_definitions": {
                    slot: {
                        "provisional_name": candidate_by_group_slot[slot]["feature_name"],
                        "reason": "Keep the patient measurement distinct.",
                    }
                    for slot in group_slots
                },
                "lookback_slot_definitions": {
                    slot: {
                        "selection": (
                            candidates[0]["supporting_evidence_ids"][0] if index == 0 else "unused"
                        ),
                        "question": "Confirm the compact candidate grouping.",
                        "reason": "Exercise the exact bounded lookback path.",
                    }
                    for index, slot in enumerate(lookback_slots)
                },
            }
        if job.job_kind == CROSS_ARCHITECTURE_INTEGRATION_JOB:
            dossiers = request["architecture_context"]["architecture_dossiers"]
            candidates = [
                candidate
                for dossier in dossiers
                for candidate in dossier["architecture_candidates"]
            ]
            rejected_id = candidates[0]["candidate_id"]
            accepted = [row for row in candidates if row["candidate_id"] != rejected_id]
            slots = request["identifier_ownership"]["identifier_domains"]["integration_slots"]
            slot_by_candidate = {
                candidate["candidate_id"]: slots[index % len(slots)]
                for index, candidate in enumerate(accepted)
            }
            candidate_by_slot = {}
            for candidate in accepted:
                candidate_by_slot.setdefault(
                    slot_by_candidate[candidate["candidate_id"]], candidate
                )
            response = {
                "candidate_routes": {
                    candidate["candidate_id"]: {
                        "route": (
                            "reject"
                            if candidate["candidate_id"] == rejected_id
                            else slot_by_candidate[candidate["candidate_id"]]
                        ),
                        "reason": (
                            "The clue is too nonspecific after independent review."
                            if candidate["candidate_id"] == rejected_id
                            else "The candidate is a supported patient measurement."
                        ),
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
                            else "Unused compiler-owned integration slot."
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
            if self.omit_integration_disposition:
                response["candidate_routes"].pop(candidates[-1]["candidate_id"])
            return response
        if job.job_kind == REJECTION_CRITIC_JOB:
            if self.omit_rejection_reconsideration:
                return {"reconsiderations": {}}
            return {
                "reconsiderations": {
                    candidate["candidate_id"]: {
                        "decision": self.rejection_revision or "uphold",
                        "proposed_name": (
                            "restored_patient_measurement" if self.rejection_revision else ""
                        ),
                        "supporting_evidence_ids": (
                            candidate["supporting_evidence_ids"][:1]
                            if self.rejection_revision
                            else []
                        ),
                        "reason": (
                            "Bounded raw support warrants deterministic restoration."
                            if self.rejection_revision
                            else "Independent review agrees the clue is nonspecific."
                        ),
                    }
                    for candidate in request["rejected_candidates"]
                }
            }
        if job.job_kind == EXTRACTION_DEFINITION_JOB:
            return {
                "feature_name": request["canonical_name"],
                "measurement": "Extract this one supported patient measurement.",
                "representation": {"kind": "unresolved", "unit": "", "categories": []},
                "aliases": [],
                "distinguish_from": [],
                "missing_or_ambiguous": "Return null when absent or ambiguous.",
                "supporting_evidence_reviewed": True,
            }
        raise AssertionError(f"unexpected job kind: {job.job_kind}")


class _ObservedInterpretFailureRunner(_DeterministicRunner):
    def __init__(self, *, failure: str, exhaust: bool = False) -> None:
        super().__init__()
        if failure not in {
            "duplicate_key",
            "empty_response",
            "whitespace_response",
            "unsupplied_evidence",
        }:
            raise ValueError("unsupported observed failure")
        self.failure = failure
        self.exhaust = exhaust

    def run_json(self, *, job):
        is_target = job.job_kind == INTERPRET_CHUNK_JOB and (
            not self.calls or (self.exhaust and len(self.calls) == 1 and len(job.messages) == 4)
        )
        if self.failure in {"duplicate_key", "empty_response", "whitespace_response"} and is_target:
            self.calls.append(job)
            failed_content = {
                "duplicate_key": (
                    '{"concepts":[],"evidence_dispositions":['
                    '{"feature_names":[],"feature_names":[]}]}'
                ),
                "empty_response": "",
                "whitespace_response": " \n\t ",
            }[self.failure]
            raise InvalidDiscoveryJsonResponse(failed_response_content=failed_content)
        response = super().run_json(job=job)
        if self.failure == "unsupplied_evidence" and is_target:
            disposition = next(iter(response["evidence_dispositions"].values()))
            disposition["member_dispositions"]["member_model_invented_999"] = {"findings": []}
        return response


class _ExecutableNoPlannerLookbackRunner(_DeterministicRunner):
    """Resolve each feature from its own support, never planner raw lookback."""

    def run_json(self, *, job):
        response = super().run_json(job=job)
        request = json.loads(job.messages[1]["content"])
        if job.job_kind == INTERPRET_CHUNK_JOB:
            for disposition in response["evidence_dispositions"].values():
                for finding in disposition["evidence_findings"]:
                    finding["value_shape_hypothesis"] = "continuous"
            return response
        if request["job"] == "fold_extraction_evidence_definitions":
            response.update(
                {
                    "measurement": ("Extract the one supported patient measurement as documented."),
                    "representation": {
                        "kind": "continuous",
                        "unit": AS_DOCUMENTED_UNIT,
                        "categories": [],
                    },
                }
            )
            return response
        return response


class _DistinctWithinFamilyPagingRunner(_DeterministicRunner):
    def run_json(self, *, job):
        request = json.loads(job.messages[1]["content"])
        if request["job"] == "compare_consolidation_candidate_relations":
            self.calls.append(job)
            return {
                "comparisons": {
                    peer_id: {
                        "relation": "distinct",
                        "reason": "Exercise arbitrary-count cross-architecture paging.",
                    }
                    for peer_id in request["peer_candidate_ids"]
                }
            }
        return super().run_json(job=job)


class _TwoRejectionRunner(_DeterministicRunner):
    """Reject the first two exhaustively reviewed provisional groups."""

    def __init__(self) -> None:
        super().__init__()
        self.integration_fold_count = 0

    def run_json(self, *, job):
        response = super().run_json(job=job)
        request = json.loads(job.messages[1]["content"])
        if request["job"] == "fold_integration_group_evidence_reviews":
            self.integration_fold_count += 1
            if self.integration_fold_count <= 2:
                response.update(
                    {
                        "decision": "reject",
                        "canonical_name": "",
                        "description": "",
                        "unresolved_ambiguity": "",
                        "reason": "Exhaustive review found this group nonspecific.",
                    }
                )
        return response


def test_production_family_explanations_cover_all_ten_architectures_in_plain_language():
    explanations = production_stage1_family_explanations()

    assert (
        tuple(PRODUCTION_STAGE1_FAMILY_EXPLANATIONS)
        == tuple(explanations)
        == (ACTIVE_STAGE1_CONCEPT_FAMILIES)
    )
    assert len(explanations) == 10
    assert all(len(explanation.split()) >= 20 for explanation in explanations.values())
    rendered = canonical_json(explanations).casefold()
    assert "temporal_policy" not in rendered
    assert "temporal policy" not in rendered
    assert "current_date" not in rendered
    assert "schema_version" not in rendered


def test_offline_precommit_is_complete_lossless_and_has_exact_settings():
    orchestrator = _orchestrator()
    packet = orchestrator.precommit.packet
    ledger = orchestrator.initial_job_ledger

    assert len(ledger.jobs) == len(ACTIVE_STAGE1_CONCEPT_FAMILIES) == 10
    assert [job.input_bindings["source_family"] for job in ledger.jobs] == list(
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    evidence_ids = []
    for job in ledger.jobs:
        request = json.loads(job.messages[1]["content"])
        assert {row["source_family"] for row in request["evidence"]} == {
            job.input_bindings["source_family"]
        }
        evidence_ids.extend(row["evidence_id"] for row in request["evidence"])
        assert job.settings.thinking_enabled is True
        assert job.settings.thinking_token_budget == SELECTOR_THINKING_TOKEN_BUDGET
    assert len(evidence_ids) == len(set(evidence_ids)) == len(orchestrator.catalog.atoms)
    assert packet["assurances"]["direct_row_level_numerical_values_accepted"] is False
    assert packet["assurances"]["bounded_response_repair_implemented"] is True
    assert packet["assurances"]["unvalidated_response_cache_write_allowed"] is False
    assert packet["config"]["max_semantic_member_ids_per_chunk"] == 3
    assert packet["chunk_plan_binding"]["max_semantic_member_ids_per_chunk"] == 3
    repair_policy = packet["response_repair_policy"]
    assert repair_policy["maximum_repair_attempts"] == 1
    assert repair_policy["message_sequence"] == ["system", "user", "assistant", "user"]
    assert set(repair_policy["repair_prompts"]) == {
        "local_json_schema_validation_failure",
        "raw_transport_budget_failure",
        "strict_json_parse_failure",
    }
    assert len(packet["dossier_direct_numerical_bindings"]) == 10
    assert packet["direct_numerical_contract_binding"] == {
        "direct_numerical_contract_kind": "direct_upstream_numerical_manifest",
        "direct_numerical_contract_sha256": "f" * 64,
        "model_facing": False,
    }
    assert packet["downstream_contract"]["raw_lookback"] == {
        "cross_architecture_integration": (
            "every_group_support_item_once_then_recursive_8_input_folds"
        ),
        "rejection_critic": (
            "every_rejected_candidate_support_item_once_then_recursive_8_input_folds"
        ),
        "extraction_definition": (
            "every_accepted_feature_support_item_once_then_recursive_8_input_folds"
        ),
    }
    assert packet["downstream_contract"]["rejection_review"] == (
        "one_page_per_support_item_and_complete_fold_disposition_per_rejected_candidate"
    )
    assert (
        packet["downstream_contract"]["lossless_raw_evidence_hierarchy"][
            "semantic_sampling_or_truncation"
        ]
        is False
    )
    rendered = orchestrator.render_offline_precommit().casefold()
    assert "temporal_policy" not in rendered
    assert "temporal policy" not in rendered
    assert "sparse_query_moments" not in rendered
    assert "non_grounding_numerical_summaries" not in "\n".join(
        job.messages[1]["content"] for job in ledger.jobs
    )
    assert _orchestrator().precommit.precommit_sha256 == orchestrator.precommit.precommit_sha256


def test_base_implementation_bundle_authenticates_dependencies_precommit_and_jobs():
    orchestrator = _orchestrator()
    bundle = hierarchical_discovery_implementation_bundle()
    packet = orchestrator.precommit.packet
    expected_files = {
        "hierarchical_all_architecture_discovery.py",
        "all_evidence_discovery_interfaces.py",
        "hierarchical_discovery_response_contract.py",
        "lossless_stage1_evidence_catalog.py",
    }

    assert orchestrator.implementation_bundle == bundle
    assert set(bundle["files"]) == expected_files
    source_root = Path(hierarchy_module.__file__).resolve().parent
    for filename in expected_files:
        assert (
            bundle["files"][filename]
            == hashlib.sha256((source_root / filename).read_bytes()).hexdigest()
        )
    bundle_body = {
        key: value for key, value in bundle.items() if key != "implementation_bundle_sha256"
    }
    assert bundle["implementation_bundle_sha256"] == content_sha256(bundle_body)
    assert packet["orchestrator_implementation_bundle"] == bundle
    assert packet["orchestrator_implementation_bundle_sha256"] == (
        bundle["implementation_bundle_sha256"]
    )
    assert {
        job.input_bindings[HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING]
        for job in orchestrator.initial_job_ledger.jobs
    } == {bundle["implementation_bundle_sha256"]}


def test_helper_only_bundle_change_rekeys_precommit_and_jobs_and_blocks_stale_execution(
    monkeypatch,
):
    original = _orchestrator()
    changed_bundle = original.implementation_bundle
    changed_bundle["files"]["hierarchical_discovery_response_contract.py"] = "0" * 64
    changed_body = {
        key: value for key, value in changed_bundle.items() if key != "implementation_bundle_sha256"
    }
    changed_bundle["implementation_bundle_sha256"] = content_sha256(changed_body)
    monkeypatch.setattr(
        hierarchy_module,
        "hierarchical_discovery_implementation_bundle",
        lambda: json.loads(canonical_json(changed_bundle)),
    )

    rekeyed = _orchestrator()
    assert rekeyed.precommit.precommit_sha256 != original.precommit.precommit_sha256
    assert rekeyed.initial_job_ledger.jobs[0].job_id != original.initial_job_ledger.jobs[0].job_id
    assert (
        rekeyed.initial_job_ledger.jobs[0].input_bindings[
            HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING
        ]
        == changed_bundle["implementation_bundle_sha256"]
    )

    runner = _DeterministicRunner()
    with pytest.raises(
        ValueError, match="dependency bundle changed before authenticated execution"
    ):
        original.execute(
            runner=runner,
            approved_precommit_sha256=original.precommit.precommit_sha256,
        )
    assert runner.calls == []


def test_wrong_precommit_or_runner_identity_makes_zero_calls():
    orchestrator = _orchestrator()
    runner = _DeterministicRunner()
    with pytest.raises(ValueError, match="does not match"):
        orchestrator.execute(runner=runner, approved_precommit_sha256="0" * 64)
    assert runner.calls == []

    class WrongIdentityRunner(_DeterministicRunner):
        def identity(self):
            return {"runner": "changed_runner", "model": "fixed_json"}

    wrong = WrongIdentityRunner()
    with pytest.raises(ValueError, match="runner identity differs"):
        orchestrator.execute(
            runner=wrong,
            approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
        )
    assert wrong.calls == []


def test_complete_hierarchical_execution_builds_ten_dossiers_and_full_dag():
    orchestrator = _orchestrator()
    runner = _DeterministicRunner()
    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    assert [dossier.source_family for dossier in completed.dossiers] == list(
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    assert all(
        set(dossier.catalog_evidence_ids) == set(dossier.coverage_disposition_ids)
        for dossier in completed.dossiers
    )
    assert {dossier.catalog_sha256 for dossier in completed.dossiers} == {
        orchestrator.catalog.catalog_sha256
    }
    assert set(completed.requested_lookback_evidence_ids) == {
        atom.evidence_id for atom in orchestrator.catalog.atoms
    }
    assert len(completed.rejected_candidate_ids) == 1
    assert len(completed.routed_features) == 9
    assert len(completed.extraction_job_ids) == 2 * len(completed.routed_features)
    assert set(completed.extraction_definitions) == {
        item.feature.canonical_name for item in completed.routed_features
    }
    assert all(
        routed.role_routing.adjustment_roles == ("prognostic_adjustment",)
        for routed in completed.routed_features
    )
    jobs = completed.execution_ledger.job_ledger.jobs
    assert len(jobs) == len(completed.execution_ledger.results) == len(runner.calls) == 81
    assert [job.job_id for job in jobs] == [job.job_id for job in runner.calls]
    assert [job.job_kind for job in jobs].count(INTERPRET_CHUNK_JOB) == 10
    assert [job.job_kind for job in jobs].count(CONSOLIDATE_ARCHITECTURE_JOB) == 10
    assert [job.job_kind for job in jobs].count(COVERAGE_CRITIC_JOB) == 10
    assert [job.job_kind for job in jobs].count(CROSS_ARCHITECTURE_PLANNER_JOB) == 11
    assert [job.job_kind for job in jobs].count(CROSS_ARCHITECTURE_INTEGRATION_JOB) == 20
    assert [job.job_kind for job in jobs].count(REJECTION_CRITIC_JOB) == 2
    assert [job.job_kind for job in jobs].count(EXTRACTION_DEFINITION_JOB) == 18
    local_jobs = [
        job
        for job in jobs
        if job.job_kind in {INTERPRET_CHUNK_JOB, CONSOLIDATE_ARCHITECTURE_JOB, COVERAGE_CRITIC_JOB}
    ]
    assert [job.job_kind for job in local_jobs] == [
        job_kind
        for _family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        for job_kind in (
            INTERPRET_CHUNK_JOB,
            CONSOLIDATE_ARCHITECTURE_JOB,
            COVERAGE_CRITIC_JOB,
        )
    ]
    planner_job = next(job for job in jobs if job.job_kind == CROSS_ARCHITECTURE_PLANNER_JOB)
    assert set(planner_job.dependencies) == {
        job.job_id for job in jobs if job.job_kind == COVERAGE_CRITIC_JOB
    }
    seen = set()
    for job in jobs:
        assert set(job.dependencies) <= seen
        seen.add(job.job_id)
        if job.job_kind == EXTRACTION_DEFINITION_JOB:
            assert job.settings.thinking_enabled is False
            assert job.settings.thinking_token_budget == 0
            request = json.loads(job.messages[1]["content"])
            assert isinstance(request["canonical_name"], str)
        else:
            assert job.settings.thinking_enabled is True
            assert job.settings.thinking_token_budget == 5000
    local_job_text = "\n".join(
        message["content"]
        for job in jobs
        if job.job_kind in {INTERPRET_CHUNK_JOB, CONSOLIDATE_ARCHITECTURE_JOB, COVERAGE_CRITIC_JOB}
        for message in job.messages
    )
    assert "direct_upstream_numerical" not in local_job_text

    with pytest.raises(ValueError, match="differ from the authenticated integration"):
        replace(completed, rejected_candidate_ids=())


def test_frozen_review_evidence_contains_only_accepted_support_with_original_ids():
    orchestrator = _orchestrator()
    completed = orchestrator.execute(
        runner=_DeterministicRunner(),
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    frozen = freeze_hierarchical_review_evidence(
        catalog=orchestrator.catalog,
        completed=completed,
        config=FrozenHierarchicalReviewEvidenceConfig(),
    )

    expected_support = {
        evidence_id
        for routed in completed.routed_features
        for evidence_id in routed.feature.supporting_evidence_ids
    }
    candidate_by_id = {
        candidate.candidate_id: candidate
        for dossier in completed.dossiers
        for candidate in dossier.architecture_candidates
    }
    rejected_support = {
        evidence_id
        for candidate_id in completed.rejected_candidate_ids
        for evidence_id in candidate_by_id[candidate_id].supporting_evidence_ids
    }
    requested_lookback = set(completed.requested_lookback_evidence_ids)
    rejected_only = rejected_support - expected_support
    planner_only = requested_lookback - expected_support - rejected_support
    assert not expected_support.intersection(rejected_only)
    assert not expected_support.intersection(planner_only)
    assert not rejected_only.intersection(planner_only)
    assert expected_support | rejected_only | planner_only == (
        expected_support | rejected_support | requested_lookback
    )
    assert set(frozen.ordered_evidence_ids) == expected_support
    assert len(frozen.review_rows) == len(expected_support) == 9
    assert all(row["evidence_id"].startswith("evidence_") for row in frozen.review_rows)
    assert all(row["role_hint"] == "" for row in frozen.review_rows)
    atom_by_id = {atom.evidence_id: atom for atom in orchestrator.catalog.atoms}
    for row in frozen.review_rows:
        atom = atom_by_id[row["evidence_id"]]
        assert row["source_families"] == [atom.source_family]
        assert row["content"] == atom.as_discovery_item().content
    assert frozen.audit["rejected_only_evidence_id_count_excluded"] == 1
    assert frozen.audit["planner_lookback_only_evidence_id_count_excluded"] == 0
    assert frozen.audit["architecture_wide_evidence_dumped_to_review"] is False


def test_large_family_uses_exhaustive_bounded_pair_pages_and_preserves_all_support():
    orchestrator = _orchestrator(first_family_atom_count=9)
    runner = _DeterministicRunner()
    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    first_dossier = completed.dossiers[0]
    assert len(first_dossier.architecture_candidates) == 1
    assert set(first_dossier.architecture_candidates[0].supporting_evidence_ids) == set(
        first_dossier.catalog_evidence_ids
    )
    relation_jobs = [
        job
        for job in completed.execution_ledger.job_ledger.jobs
        if json.loads(job.messages[1]["content"])["job"]
        == "compare_consolidation_candidate_relations"
    ]
    assert relation_jobs
    assert (
        sum(
            len(json.loads(job.messages[1]["content"])["peer_candidate_ids"])
            for job in relation_jobs
        )
        == 36
    )
    assert all(
        len(json.loads(job.messages[1]["content"])["peer_candidate_ids"]) <= 7
        for job in relation_jobs
    )
    assert all(
        job.identifier_ownership["ownership"]["wire_response_budget"][
            "maximum_canonical_json_bytes"
        ]
        < 20_000
        for job in relation_jobs
    )


def test_large_cross_architecture_stage_pages_every_pair_and_integrates_every_group():
    orchestrator = _orchestrator(first_family_atom_count=9)
    runner = _DistinctWithinFamilyPagingRunner()
    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    cross_relation_jobs = [
        job
        for job in completed.execution_ledger.job_ledger.jobs
        if json.loads(job.messages[1]["content"])["job"]
        == "compare_cross_architecture_candidate_relations"
    ]
    assert (
        sum(
            len(json.loads(job.messages[1]["content"])["peer_candidate_ids"])
            for job in cross_relation_jobs
        )
        == 153
    )  # 18 candidates choose 2.
    assert all(
        len(json.loads(job.messages[1]["content"])["peer_candidate_ids"]) <= 7
        for job in cross_relation_jobs
    )
    evidence_page_jobs = [
        job
        for job in completed.execution_ledger.job_ledger.jobs
        if json.loads(job.messages[1]["content"])["job"] == "review_integration_group_evidence"
    ]
    fold_jobs = [
        job
        for job in completed.execution_ledger.job_ledger.jobs
        if json.loads(job.messages[1]["content"])["job"]
        == "fold_integration_group_evidence_reviews"
    ]
    expected_support_reviews = sum(
        len(request["evidence_ids"])
        for request in completed.planner_response["raw_evidence_requests"]
    )
    assert len(evidence_page_jobs) == expected_support_reviews
    assert len(fold_jobs) >= len(completed.planner_response["provisional_groups"])
    assert all(
        len(json.loads(job.messages[1]["content"])["review_input_ids"]) <= 8 for job in fold_jobs
    )
    assert (
        completed.integration_response["wire_normalization_audit"][
            "all_provisional_groups_integrated_exactly_once"
        ]
        is True
    )
    assert (
        completed.integration_response["wire_normalization_audit"][
            "global_integrated_feature_truncation"
        ]
        is False
    )
    assert (
        completed.integration_response["wire_normalization_audit"]["raw_support_sampling"] is False
    )
    assert all(
        job.identifier_ownership["ownership"]["wire_response_budget"][
            "maximum_canonical_json_bytes"
        ]
        < 20_000
        for job in (*cross_relation_jobs, *evidence_page_jobs, *fold_jobs)
    )


@pytest.mark.parametrize(
    "config",
    (
        FrozenHierarchicalReviewEvidenceConfig(max_evidence_ids=8),
        FrozenHierarchicalReviewEvidenceConfig(max_evidence_bytes=1),
    ),
)
def test_frozen_review_evidence_bounds_fail_without_truncation(config):
    orchestrator = _orchestrator()
    completed = orchestrator.execute(
        runner=_DeterministicRunner(),
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    with pytest.raises(ValueError, match="refusing to truncate"):
        freeze_hierarchical_review_evidence(
            catalog=orchestrator.catalog,
            completed=completed,
            config=config,
        )


def test_frozen_review_evidence_rejects_a_different_catalog():
    orchestrator = _orchestrator()
    completed = orchestrator.execute(
        runner=_DeterministicRunner(),
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    with pytest.raises(ValueError, match="different evidence catalog"):
        freeze_hierarchical_review_evidence(
            catalog=_catalog(first_family_atom_count=2),
            completed=completed,
            config=FrozenHierarchicalReviewEvidenceConfig(),
        )


def test_every_job_hides_machine_metadata_and_authenticates_exact_message_bytes():
    orchestrator = _orchestrator()
    runner = _DeterministicRunner()
    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    hidden_keys = {
        "schema_version",
        "catalog_sha256",
        "coverage_audit_sha256",
        "manifest_sha256",
        "split_fingerprint",
        "producer_identity",
        "producer_id",
        "cache_id",
        "cache_key",
        "deterministic_role_routing",
        "role_routing_sha256",
    }
    jobs = completed.execution_ledger.job_ledger.jobs
    assert jobs
    for job in jobs:
        messages = list(job.messages)
        request = json.loads(messages[1]["content"])
        assert not hidden_keys & _json_keys(request)
        assert all(token not in messages[0]["content"] for token in hidden_keys)

        envelope = job.input_bindings[AUTHENTICATED_MESSAGE_ENVELOPE_BINDING]
        assert envelope["byte_count"] == len(job.rendered_messages_bytes)
        assert envelope["sha256"] == content_sha256(messages)
        assert envelope["byte_limit_binding"] == (
            "content_addressed_orchestrator_runtime_config_v1"
        )
        assert envelope["schema_version"].endswith(("_v1", "_v2", "_v3", "_v4", "_v5"))

    dossier_jobs = [
        job
        for job in jobs
        if job.job_kind in {CROSS_ARCHITECTURE_PLANNER_JOB, CROSS_ARCHITECTURE_INTEGRATION_JOB}
    ]
    assert len(dossier_jobs) == 31
    for job in dossier_jobs:
        request = json.loads(job.messages[1]["content"])
        assert "architecture_dossiers" not in request
        assert "architecture_context" not in request
        assert "direct_numerical_channel" not in _json_keys(request)
        assert "dossier_sha256" in job.input_bindings

    extraction_jobs = [job for job in jobs if job.job_kind == EXTRACTION_DEFINITION_JOB]
    assert extraction_jobs
    for job in extraction_jobs:
        request = json.loads(job.messages[1]["content"])
        assert "observable_axes" not in _json_keys(request)
        assert "deterministic_role_routing" not in _json_keys(request)
        assert "deterministic_role_routing" in job.input_bindings
        assert "schema_version" in job.input_bindings["deterministic_role_routing"]


def test_configured_prompt_byte_guard_fails_before_runner_invocation():
    catalog = _catalog()
    plan = build_complete_architecture_chunks(
        catalog,
        max_atoms_per_chunk=1,
        max_bytes_per_chunk=20_000,
    )
    explanations = production_stage1_family_explanations()
    explanations[ACTIVE_STAGE1_CONCEPT_FAMILIES[0]] = "clue " * (
        MAX_RENDERED_DISCOVERY_PROMPT_BYTES // 4
    )
    orchestrator = HierarchicalAllArchitectureDiscoveryOrchestrator(
        catalog=catalog,
        chunk_plan=plan,
        family_explanations=explanations,
        direct_numerical_bindings=_bindings(),
        runner_identity={"runner": "deterministic_offline_test_v1", "model": "fixed_json"},
    )
    first_job = orchestrator.initial_job_ledger.jobs[0]
    assert len(first_job.rendered_messages_bytes) > MAX_RENDERED_DISCOVERY_PROMPT_BYTES

    runner = _DeterministicRunner()
    with pytest.raises(ValueError, match="220000-byte guard"):
        orchestrator.execute(
            runner=runner,
            approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
        )
    assert runner.calls == []

    assert (
        HierarchicalDiscoveryConfig(
            max_rendered_prompt_bytes=MAX_RENDERED_DISCOVERY_PROMPT_BYTES + 130_000
        ).max_rendered_prompt_bytes
        == 350_000
    )


def test_semantic_member_chunk_cap_is_positive_and_must_match_the_plan():
    with pytest.raises(ValueError, match="max_semantic_member_ids_per_chunk"):
        HierarchicalDiscoveryConfig(max_semantic_member_ids_per_chunk=0)

    catalog = _catalog()
    plan = build_complete_architecture_chunks(
        catalog,
        max_atoms_per_chunk=1,
        max_bytes_per_chunk=20_000,
        max_semantic_member_ids_per_chunk=2,
    )
    with pytest.raises(ValueError, match="differs from the hierarchy config"):
        HierarchicalAllArchitectureDiscoveryOrchestrator(
            catalog=catalog,
            chunk_plan=plan,
            family_explanations={
                family: f"Interpret evidence emitted by {family}."
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            },
            direct_numerical_bindings=_bindings(),
            runner_identity={"runner": "deterministic_offline_test_v1"},
            config=HierarchicalDiscoveryConfig(max_semantic_member_ids_per_chunk=3),
        )


def test_cross_architecture_plan_schedules_arbitrary_complete_support_without_sampling():
    catalog = _catalog(first_family_atom_count=20)
    evidence_by_id = {atom.evidence_id: atom.as_discovery_item() for atom in catalog.atoms}
    complete_support = tuple(
        evidence_id
        for evidence_id, evidence in evidence_by_id.items()
        if evidence.source_family == ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    )
    candidate = DiscoveryCandidate(
        candidate_id="candidate.lossless",
        feature_name="lossless_measure",
        description="One patient measurement with arbitrary complete support.",
        supporting_evidence_ids=complete_support,
        source_families=(ACTIVE_STAGE1_CONCEPT_FAMILIES[0],),
        value_shape_hypothesis="ambiguous",
    )
    group_id = "candidate_group.lossless"
    planner, support_by_group = hierarchy_module._compile_bounded_cross_architecture_plan(
        candidates=(candidate,),
        grouped={
            "groups": [
                {
                    "group_id": group_id,
                    "member_candidate_ids": [candidate.candidate_id],
                }
            ],
            "pair_relation_audit": {"expected_pair_count": 0},
        },
        definitions_by_group_id={
            group_id: {
                "canonical_name": candidate.feature_name,
                "description": candidate.description,
                "unresolved_ambiguity": "",
                "reason": "Preserve the singleton definition.",
            }
        },
        evidence_by_id=evidence_by_id,
        wire_budget=HierarchicalDiscoveryConfig().wire_budget,
    )

    assert len(complete_support) == 20
    assert support_by_group[group_id] == complete_support
    assert planner["raw_evidence_requests"][0]["evidence_ids"] == list(complete_support)
    audit = planner["wire_normalization_audit"]
    assert audit["raw_support_sampling"] is False
    assert audit["every_group_support_item_is_page_scheduled"] is True


def test_legacy_zero_planner_lookback_does_not_truncate_grounded_feature_contracts():
    orchestrator = _orchestrator(max_lookback=0)
    runner = _ExecutableNoPlannerLookbackRunner()
    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    assert set(completed.requested_lookback_evidence_ids) == {
        atom.evidence_id for atom in orchestrator.catalog.atoms
    }
    extraction_jobs = [job for job in runner.calls if job.job_kind == EXTRACTION_DEFINITION_JOB]
    page_jobs = [
        job
        for job in extraction_jobs
        if json.loads(job.messages[1]["content"])["job"] == "review_extraction_feature_evidence"
    ]
    fold_jobs = [
        job
        for job in extraction_jobs
        if json.loads(job.messages[1]["content"])["job"] == "fold_extraction_evidence_definitions"
    ]
    assert len(page_jobs) == len(fold_jobs) == len(completed.routed_features) == 9
    expected_raw_by_id = {
        atom.evidence_id: {
            key: value
            for key, value in atom.as_discovery_item().as_prompt_item().items()
            if key != "observable_axes"
        }
        for atom in orchestrator.catalog.atoms
    }
    for job in page_jobs:
        request = json.loads(job.messages[1]["content"])
        support_ids = [request["evidence_id"]]
        assert request["raw_evidence"] == expected_raw_by_id[request["evidence_id"]]
        assert job.input_bindings["complete_supporting_evidence_ids"] == support_ids
        assert "observable_axes" not in job.messages[1]["content"]
        assert "deterministic_role_routing" not in job.messages[1]["content"]
        assert "deterministic_role_routing" in job.input_bindings
    for job in fold_jobs:
        request = json.loads(job.messages[1]["content"])
        assert request["planner_lookback_constraints"] == {
            "aliases": [],
            "units": [],
            "categories": [],
            "distinguish_from": [],
        }
        assert request["vocabulary_grounding_policy"]["continuous_scale_fallback"] == (
            AS_DOCUMENTED_UNIT
        )
        assert (
            request["vocabulary_grounding_policy"]["mechanical_encodings_are_clinical_ontology"]
            is False
        )
        assert "observable_axes" not in job.messages[1]["content"]
        assert "deterministic_role_routing" not in job.messages[1]["content"]
        assert "deterministic_role_routing" in job.input_bindings

    registry = compile_hierarchical_discovery(completed)
    assert len(registry.specs) == 9
    assert {spec["type"] for spec in registry.specs} == {"continuous"}
    assert all("not a clinical unit assertion" in spec["description"] for spec in registry.specs)


def test_member_disposition_and_coverage_findings_fail_closed():
    orchestrator = _orchestrator()
    incomplete = _DeterministicRunner(bad_first_member_disposition=True)
    repaired = orchestrator.execute(
        runner=incomplete,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )
    assert len(incomplete.calls[0].messages) == 2
    assert len(incomplete.calls[1].messages) == 4
    assert tuple(row["role"] for row in incomplete.calls[1].messages) == (
        "system",
        "user",
        "assistant",
        "user",
    )
    assert len(repaired.execution_ledger.results[0].response_attempt_trace["attempts"]) == 2

    revision = _DeterministicRunner(coverage_revision=True)
    with pytest.raises(CoverageCriticRequiresRevision, match="unresolved coverage"):
        orchestrator.execute(
            runner=revision,
            approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
        )
    # The first architecture is interpreted, consolidated, and audited before
    # any evidence from the second architecture is sent.
    assert [job.job_kind for job in revision.calls] == [
        INTERPRET_CHUNK_JOB,
        CONSOLIDATE_ARCHITECTURE_JOB,
        COVERAGE_CRITIC_JOB,
        COVERAGE_CRITIC_JOB,
    ]


def test_coverage_revision_is_compiled_and_reaudited_before_cross_architecture_work():
    orchestrator = _orchestrator()
    runner = _DeterministicRunner(
        coverage_revision=True,
        coverage_revision_once=True,
    )

    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    first_dossier = completed.dossiers[0]
    assert any(
        candidate.feature_name.endswith("_subtype")
        for candidate in first_dossier.architecture_candidates
    )
    first_cross_index = next(
        index
        for index, job in enumerate(runner.calls)
        if job.job_kind == CROSS_ARCHITECTURE_PLANNER_JOB
    )
    first_family_coverage = [
        job
        for job in runner.calls[:first_cross_index]
        if job.job_kind == COVERAGE_CRITIC_JOB
        and job.scope.startswith(ACTIVE_STAGE1_CONCEPT_FAMILIES[0])
    ]
    assert len(first_family_coverage) == 2
    assert first_family_coverage[0].job_id in first_family_coverage[1].dependencies


def test_keyed_wire_hash_and_normalized_interpretation_hash_are_distinct_and_bound():
    orchestrator = _orchestrator()
    runner = _DeterministicRunner()
    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    first = completed.execution_ledger.results[0]
    normalized = first.response
    wire_dispositions = {}
    for disposition in normalized["evidence_dispositions"]:
        evidence_id = disposition["evidence_id"]
        wire_dispositions[evidence_id] = {
            "evidence_findings": [],
            "member_dispositions": {
                member["member_id"]: {"findings": []}
                for member in disposition["member_dispositions"]
            },
            "reason": disposition["reason"],
        }
    for concept in normalized["concepts"]:
        origin = concept["origin"]
        finding = {
            "feature_name": concept["feature_name"],
            "description": concept["description"],
            "value_shape_hypothesis": concept["value_shape_hypothesis"],
            "unresolved_ambiguity": concept["unresolved_ambiguity"],
        }
        if origin["member_id"]:
            target = wire_dispositions[origin["evidence_id"]]["member_dispositions"][
                origin["member_id"]
            ]["findings"]
        else:
            target = wire_dispositions[origin["evidence_id"]]["evidence_findings"]
        assert origin["finding_ordinal"] == len(target)
        target.append(finding)
    wire = {"evidence_dispositions": wire_dispositions}
    final_attempt = first.response_attempt_trace["attempts"][-1]
    assert final_attempt["raw_response_projection_sha256"] == content_sha256(wire)
    assert final_attempt["normalized_validated_response_sha256"] == first.response_sha256
    assert first.raw_wire_response_sha256 == content_sha256(wire)
    assert first.raw_wire_response_sha256 != first.response_sha256


def test_raw_wire_is_detached_before_semantic_validator_runs():
    orchestrator = _orchestrator()
    runner = _DeterministicRunner()
    job = orchestrator.initial_job_ledger.jobs[0]
    request = json.loads(job.messages[1]["content"])
    evidence = tuple(
        orchestrator._evidence_by_id[row["evidence_id"]] for row in request["evidence"]
    )
    observed = {}

    def mutating_validator(raw):
        original = json.loads(canonical_json(raw))
        observed["wire"] = original
        raw.clear()
        return validate_interpret_evidence_chunk_response(original, evidence=evidence)

    _response, result = orchestrator._run(
        runner=runner,
        job=job,
        validator=mutating_validator,
    )
    assert result.raw_wire_response_sha256 == content_sha256(observed["wire"])
    assert observed["wire"]["evidence_dispositions"]


@pytest.mark.parametrize(
    ("failure", "expected_category"),
    [
        ("duplicate_key", "strict_json_parse_failure"),
        ("empty_response", "strict_json_parse_failure"),
        ("whitespace_response", "strict_json_parse_failure"),
        ("unsupplied_evidence", "local_json_schema_validation_failure"),
    ],
)
def test_observed_interpret_failures_receive_one_authenticated_sanitized_repair(
    failure,
    expected_category,
):
    orchestrator = _orchestrator()
    runner = _ObservedInterpretFailureRunner(failure=failure)

    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    assert len(runner.calls[0].messages) == 2
    repair_job = runner.calls[1]
    assert tuple(row["role"] for row in repair_job.messages) == (
        "system",
        "user",
        "assistant",
        "user",
    )
    binding = repair_job.input_bindings["authenticated_bounded_response_repair"]
    assert binding["failure_category"] == expected_category
    assert binding["repair_attempt_number"] == 1
    assert repair_job.response_schema == runner.calls[0].response_schema
    assert repair_job.identifier_ownership == runner.calls[0].identifier_ownership
    assert (
        repair_job.input_bindings[HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING]
        == runner.calls[0].input_bindings[HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING]
    )
    repair_instruction = repair_job.messages[-1]["content"]
    repair_assistant = repair_job.messages[-2]["content"]
    assert "evidence_model_invented_999" not in repair_instruction
    assert "evidence_model_invented_999" not in repair_assistant
    assert "feature_names" not in repair_instruction
    assert "feature_names" not in repair_assistant
    assert "exception" not in repair_instruction.casefold()
    assert "permitted by that field's enum or const" in repair_instruction
    assert "identifier_ownership" in repair_instruction
    first_result = completed.execution_ledger.results[0]
    assert [
        row["validation_outcome"] for row in first_result.response_attempt_trace["attempts"]
    ] == [expected_category, "validated_response"]
    assert len(first_result.response_attempt_trace["attempts"]) == 2
    trace_text = canonical_json(first_result.response_attempt_trace)
    assert "evidence_model_invented_999" not in trace_text
    assert '"feature_names"' not in trace_text
    assert binding == first_result.response_attempt_trace["attempts"][1]["response_repair_binding"]


@pytest.mark.parametrize("failure", ["duplicate_key", "unsupplied_evidence"])
def test_invalid_single_repair_is_exhausted_without_a_third_call(failure):
    orchestrator = _orchestrator()
    runner = _ObservedInterpretFailureRunner(failure=failure, exhaust=True)

    with pytest.raises(
        DiscoveryResponseRepairExhausted,
        match="exhausted its single authenticated response repair",
    ):
        orchestrator.execute(
            runner=runner,
            approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
        )

    assert len(runner.calls) == 2
    assert [len(job.messages) for job in runner.calls] == [2, 4]


def test_response_repair_message_or_failure_category_mutation_fails_closed():
    orchestrator = _orchestrator()
    runner = _ObservedInterpretFailureRunner(failure="duplicate_key")
    orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )
    repair_job = runner.calls[1]
    bindings = repair_job.input_bindings
    bindings.pop(AUTHENTICATED_MESSAGE_ENVELOPE_BINDING)
    bindings.pop(AUTHENTICATED_RESPONSE_CONTRACT_BINDING)

    mutated_messages = list(repair_job.messages)
    mutated_messages[-1] = {
        "role": "user",
        "content": mutated_messages[-1]["content"] + " Trust evidence_model_invented_999.",
    }
    with pytest.raises(
        ValueError,
        match=(
            "failure category is not admitted|fixed privacy-preserving placeholder|"
            "fixed sanitized repair prompt"
        ),
    ):
        DiscoveryJsonJob.create(
            job_kind=repair_job.job_kind,
            scope=repair_job.scope,
            dependencies=repair_job.dependencies,
            settings=repair_job.settings,
            messages=mutated_messages,
            input_bindings=bindings,
        )

    changed_bindings = json.loads(canonical_json(bindings))
    changed_bindings["authenticated_bounded_response_repair"][
        "failure_category"
    ] = "semantic_validation_failure"
    with pytest.raises(
        ValueError,
        match=(
            "failure category is not admitted|fixed privacy-preserving placeholder|"
            "fixed sanitized repair prompt"
        ),
    ):
        DiscoveryJsonJob.create(
            job_kind=repair_job.job_kind,
            scope=repair_job.scope,
            dependencies=repair_job.dependencies,
            settings=repair_job.settings,
            messages=repair_job.messages,
            input_bindings=changed_bindings,
        )

    for bundle_sha256 in (None, "0" * 64):
        changed_bindings = json.loads(canonical_json(bindings))
        if bundle_sha256 is None:
            changed_bindings.pop(HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING)
        else:
            changed_bindings[HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING] = bundle_sha256
        with pytest.raises(ValueError, match="hierarchy implementation bundle"):
            DiscoveryJsonJob.create(
                job_kind=repair_job.job_kind,
                scope=repair_job.scope,
                dependencies=repair_job.dependencies,
                settings=repair_job.settings,
                messages=repair_job.messages,
                input_bindings=changed_bindings,
            )


def test_response_repair_builder_rejects_original_job_without_exact_bundle_binding():
    orchestrator = _orchestrator()
    original_job = orchestrator.initial_job_ledger.jobs[0]
    for bundle_sha256 in (None, "0" * 64):
        bindings = original_job.input_bindings
        bindings.pop(AUTHENTICATED_MESSAGE_ENVELOPE_BINDING)
        bindings.pop(AUTHENTICATED_RESPONSE_CONTRACT_BINDING)
        if bundle_sha256 is None:
            bindings.pop(HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING)
        else:
            bindings[HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING] = bundle_sha256
        forged_original = DiscoveryJsonJob.create(
            job_kind=original_job.job_kind,
            scope=original_job.scope,
            dependencies=original_job.dependencies,
            settings=original_job.settings,
            messages=original_job.messages,
            input_bindings=bindings,
        )
        with pytest.raises(ValueError, match="current hierarchy implementation bundle"):
            hierarchy_module._build_response_repair_job_from_projection_sha256(
                original_job=forged_original,
                prior_response_content_sha256="a" * 64,
                failure_category="strict_json_parse_failure",
            )


@pytest.mark.parametrize(
    "mutation",
    [
        "job_id",
        "job_sha256",
        "input_bindings_sha256",
        "messages_sha256",
        "policy_binding",
        "failure_category_binding",
        "prior_response_binding",
        "assistant_placeholder_binding",
    ],
)
def test_fabricated_second_attempt_trace_identity_or_binding_fails_closed(mutation):
    orchestrator = _orchestrator()
    runner = _ObservedInterpretFailureRunner(failure="duplicate_key")
    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )
    logical_job = completed.execution_ledger.job_ledger.jobs[0]
    result = completed.execution_ledger.results[0]
    trace = json.loads(canonical_json(result.response_attempt_trace))
    second = trace["attempts"][1]
    if mutation == "job_id":
        second["job_id"] = "job_" + "0" * 64
    elif mutation in {
        "job_sha256",
        "input_bindings_sha256",
        "messages_sha256",
    }:
        second[mutation] = "0" * 64
    elif mutation == "policy_binding":
        second["response_repair_binding"]["policy_sha256"] = "0" * 64
    elif mutation == "failure_category_binding":
        second["response_repair_binding"]["failure_category"] = "semantic_validation_failure"
    elif mutation == "prior_response_binding":
        second["response_repair_binding"]["prior_response_content_sha256"] = "0" * 64
    else:
        second["response_repair_binding"]["assistant_placeholder_sha256"] = "0" * 64
    trace_body = {key: value for key, value in trace.items() if key != "trace_sha256"}
    trace["trace_sha256"] = content_sha256(trace_body)

    with pytest.raises(ValueError, match="exact deterministic repair job"):
        ValidatedDiscoveryJobResult.create(
            job=logical_job,
            validated_response=result.response,
            response_attempt_trace=trace,
        )


def test_multichunk_coverage_is_raw_evidence_local_and_fully_aggregated():
    orchestrator = _orchestrator(first_family_atom_count=2)
    runner = _DeterministicRunner()
    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    coverage_jobs = [job for job in runner.calls if job.job_kind == COVERAGE_CRITIC_JOB]
    assert len(coverage_jobs) == len(orchestrator.chunk_plan.chunks) == 11
    all_evidence_ids = {atom.evidence_id for atom in orchestrator.catalog.atoms}
    for job in coverage_jobs:
        request = json.loads(job.messages[1]["content"])
        expected_ids = tuple(job.input_bindings["expected_reviewed_evidence_ids"])
        assert tuple(row["evidence_id"] for row in request["evidence"]) == expected_ids
        assert len(request["chunk_interpretations"]) == 1
        assert {
            row["evidence_id"]
            for row in request["chunk_interpretations"][0]["evidence_dispositions"]
        } == set(expected_ids)
        prompt_text = job.messages[1]["content"]
        for outside_id in all_evidence_ids - set(expected_ids):
            assert outside_id not in prompt_text

    first_family = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    first_family_jobs = [
        job
        for job in runner.calls
        if job.scope == first_family or job.scope.startswith(f"{first_family}.chunk_")
    ]
    assert [job.job_kind for job in first_family_jobs] == [
        INTERPRET_CHUNK_JOB,
        INTERPRET_CHUNK_JOB,
        CONSOLIDATE_ARCHITECTURE_JOB,
        COVERAGE_CRITIC_JOB,
        COVERAGE_CRITIC_JOB,
    ]
    planner_job = next(
        job
        for job in completed.execution_ledger.job_ledger.jobs
        if job.job_kind == CROSS_ARCHITECTURE_PLANNER_JOB
    )
    assert set(planner_job.dependencies) == {job.job_id for job in coverage_jobs}
    first_dossier = completed.dossiers[0]
    assert first_dossier.catalog_sha256 == orchestrator.catalog.catalog_sha256
    assert set(first_dossier.coverage_disposition_ids) == {
        atom.evidence_id
        for atom in orchestrator.catalog.atoms
        if atom.source_family == first_family
    }


def test_legacy_planner_lookback_bound_does_not_limit_lossless_page_schedule():
    orchestrator = _orchestrator(max_lookback=0)
    runner = _DeterministicRunner()
    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )
    assert set(completed.requested_lookback_evidence_ids) == {
        atom.evidence_id for atom in orchestrator.catalog.atoms
    }
    page_jobs = [
        job
        for job in runner.calls
        if json.loads(job.messages[1]["content"])["job"] == "review_integration_group_evidence"
    ]
    assert {json.loads(job.messages[1]["content"])["evidence_id"] for job in page_jobs} == set(
        completed.requested_lookback_evidence_ids
    )


def test_integration_and_rejection_critics_require_every_candidate_disposition():
    orchestrator = _orchestrator()
    incomplete_integration = _DeterministicRunner(omit_integration_disposition=True)
    with pytest.raises(
        DiscoveryResponseRepairExhausted,
        match="single authenticated response repair",
    ):
        orchestrator.execute(
            runner=incomplete_integration,
            approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
        )
    assert incomplete_integration.calls[-1].job_kind == CROSS_ARCHITECTURE_INTEGRATION_JOB

    incomplete_rejection = _DeterministicRunner(omit_rejection_reconsideration=True)
    with pytest.raises(
        DiscoveryResponseRepairExhausted,
        match="single authenticated response repair",
    ):
        orchestrator.execute(
            runner=incomplete_rejection,
            approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
        )
    assert incomplete_rejection.calls[-1].job_kind == REJECTION_CRITIC_JOB


def test_each_rejection_critic_receives_exact_candidate_support_raw_evidence():
    orchestrator = _orchestrator()
    runner = _TwoRejectionRunner()
    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    assert set(completed.requested_lookback_evidence_ids) == {
        atom.evidence_id for atom in orchestrator.catalog.atoms
    }

    page_jobs = [
        job
        for job in runner.calls
        if json.loads(job.messages[1]["content"])["job"] == "review_rejection_candidate_evidence"
    ]
    fold_jobs = [
        job
        for job in runner.calls
        if json.loads(job.messages[1]["content"])["job"]
        == "fold_rejection_candidate_evidence_reviews"
    ]
    assert len(page_jobs) == len(fold_jobs) == len(completed.rejected_candidate_ids) == 2
    for job, rejected_id in zip(page_jobs, completed.rejected_candidate_ids):
        request = json.loads(job.messages[1]["content"])
        assert request["candidate_id"] == rejected_id
        assert request["raw_evidence"]["evidence_id"] == request["evidence_id"]
        support_ids = job.input_bindings["complete_rejected_candidate_supporting_evidence_ids"]
        assert request["evidence_id"] in support_ids
        assert set(support_ids) <= set(completed.requested_lookback_evidence_ids)
    assert all(
        1 <= len(json.loads(job.messages[1]["content"])["review_input_ids"]) <= 8
        for job in fold_jobs
    )


@pytest.mark.parametrize(
    "config",
    [
        HierarchicalDiscoveryConfig(
            max_rejection_lookback_ids_per_candidate=0,
        ),
        HierarchicalDiscoveryConfig(
            max_rejection_lookback_bytes_per_candidate=1,
        ),
    ],
)
def test_legacy_rejection_lookback_bounds_do_not_truncate_lossless_review(config):
    orchestrator = _orchestrator(config=config)
    runner = _DeterministicRunner()
    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    page_jobs = [
        job
        for job in runner.calls
        if json.loads(job.messages[1]["content"])["job"] == "review_rejection_candidate_evidence"
    ]
    assert len(page_jobs) == len(completed.rejected_candidate_ids) == 1
    request = json.loads(page_jobs[0].messages[1]["content"])
    assert request["raw_evidence"]["evidence_id"] == request["evidence_id"]
    assert page_jobs[0].input_bindings["complete_rejected_candidate_supporting_evidence_ids"] == [
        request["evidence_id"]
    ]
    assert any(job.job_kind == EXTRACTION_DEFINITION_JOB for job in runner.calls)


@pytest.mark.parametrize("decision", ["restore", "split"])
def test_rejection_revision_is_compiled_into_integration_before_extraction(decision):
    orchestrator = _orchestrator()
    runner = _DeterministicRunner(rejection_revision=decision)

    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    assert completed.rejected_candidate_ids == ()
    assert any(
        routed.feature.canonical_name == "restored_patient_measurement"
        for routed in completed.routed_features
    )
    events = completed.integration_response["wire_normalization_audit"][
        "rejection_reconsideration_events"
    ]
    assert len(events) == 1
    assert events[0]["decision"] == decision
    assert any(
        job.job_kind == EXTRACTION_DEFINITION_JOB
        and job.scope.startswith("restored_patient_measurement.")
        for job in runner.calls
    )


def test_legacy_cross_architecture_byte_bound_does_not_drop_support_pages():
    orchestrator = _orchestrator(
        config=HierarchicalDiscoveryConfig(max_cross_architecture_lookback_bytes_per_group=1)
    )
    runner = _DeterministicRunner()
    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )
    page_jobs = [
        job
        for job in runner.calls
        if json.loads(job.messages[1]["content"])["job"] == "review_integration_group_evidence"
    ]
    assert {json.loads(job.messages[1]["content"])["evidence_id"] for job in page_jobs} == set(
        completed.requested_lookback_evidence_ids
    )


def test_legacy_integrated_feature_cap_does_not_truncate_paged_groups():
    orchestrator = _orchestrator(config=HierarchicalDiscoveryConfig(max_integrated_features=1))
    runner = _DeterministicRunner()
    completed = orchestrator.execute(
        runner=runner,
        approved_precommit_sha256=orchestrator.precommit.precommit_sha256,
    )

    assert len(completed.routed_features) == 9
    assert (
        completed.integration_response["wire_normalization_audit"][
            "global_integrated_feature_truncation"
        ]
        is False
    )
