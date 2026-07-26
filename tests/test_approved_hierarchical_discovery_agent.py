from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

import oci.inference.approved_hierarchical_discovery_agent as approved_module
from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT,
    DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST,
    HETEROGENEITY_AXIS,
    OUTCOME_AXIS,
    TFIDF_SEMANTIC_RETRIEVAL,
    canonical_json,
    content_sha256,
)
from oci.inference.approved_hierarchical_discovery_agent import (
    ApprovedHierarchicalDiscoveryAgent,
    AuthenticatedReferenceOnlyDirectNumericalContract,
    AuthenticatedRunnerExecutionTrace,
    direct_numerical_bindings_from_intent,
    direct_numerical_bindings_from_manifest,
    direct_numerical_bindings_from_reference_contract,
)
from oci.inference.direct_upstream_numerical_manifest import (
    CALIBRATED_SOURCES_BLOCK,
    EXACT_PRECOMMITTED_ALIGNMENT,
    MATRIX_BLOCKS,
    NESTED_CALIBRATED_STATUS,
    PERMUTATION_SUMMARY_ALIGNMENT,
    RAW_FEATURES_BLOCK,
    ROW_SCOPES,
    UNCALIBRATED_BASIS_STATUS,
    AuthenticatedMatrixBinding,
    DirectNumericalCoordinate,
    DirectNumericalFamilyCoverage,
    DirectUpstreamNumericalManifest,
)
from oci.inference.hierarchical_all_architecture_discovery import (
    CONSOLIDATE_ARCHITECTURE_JOB,
    COVERAGE_CRITIC_JOB,
    CROSS_ARCHITECTURE_INTEGRATION_JOB,
    CROSS_ARCHITECTURE_PLANNER_JOB,
    EXTRACTION_DEFINITION_JOB,
    INTERPRET_CHUNK_JOB,
    HierarchicalDiscoveryConfig,
)
from oci.inference.first_gate_materialization_contract import (
    FirstGateMaterializationIntent,
)
from oci.inference.hierarchical_discovery_job_cache import (
    AuthenticatedHierarchicalDiscoveryJobCache,
)
from oci.inference.openai_compatible_json_discovery_job_runner import (
    InvalidDiscoveryJsonResponse,
    InvalidDiscoveryTransportResponse,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
    RoleNeutralEvidenceCatalog,
    Stage1EvidenceAtom,
    build_complete_architecture_chunks,
    validate_role_neutral_catalog,
)
from tests.hierarchy_resource_test_support import HIERARCHY_JOB_CACHE_CONFIG
from tests.semantic_member_batching_test_support import (
    semantic_member_batching_audit,
    semantic_member_batching_identity,
)


def _catalog() -> RoleNeutralEvidenceCatalog:
    split_fingerprint = "1" * 64
    semantic_member_batch_size = 1
    batching = semantic_member_batching_identity(
        semantic_member_batch_size=semantic_member_batch_size,
    )
    atoms = []
    for ordinal, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES, start=1):
        member_id = f"member_{ordinal:03d}"
        origin = {"source": f"closed_{family}", "ordinal": ordinal}
        content = {
            "terms": [
                {
                    "member_id": member_id,
                    "term": (f"{family} patient measurement clue in " "documented_clinical_unit"),
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
    identity = {
        "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
        "semantic_member_batching": batching,
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
        catalog_sha256=content_sha256(identity),
        _audit_json=canonical_json(
            semantic_member_batching_audit(
                semantic_member_batch_size=semantic_member_batch_size,
            )
        ),
    )
    validate_role_neutral_catalog(catalog)
    return catalog


def _coordinate(
    *,
    family: str,
    ordinal: int,
    matrix_block: str,
    column_index: int,
) -> DirectNumericalCoordinate:
    calibrated = matrix_block == CALIBRATED_SOURCES_BLOCK
    coordinate_name = f"closed_coordinate_{ordinal:03d}"
    source_kind = f"closed_kind_{ordinal:03d}"
    identity_fields = {
        "matrix_block": matrix_block,
        "column_index": column_index,
        "coordinate_name": coordinate_name,
        "source_family": family,
        "source_kind": source_kind,
        "producer_subarchitecture": f"closed_producer_{ordinal:03d}",
        "consumer_role": "closed_effect_basis",
        "observable_axes": [HETEROGENEITY_AXIS],
        "calibration_status": (
            NESTED_CALIBRATED_STATUS if calibrated else UNCALIBRATED_BASIS_STATUS
        ),
        "statistic_kind": "direct_prediction" if calibrated else "signed_mean",
        "statistic_rank": None,
        "statistic_width": 1,
        "alignment_mode": (
            EXACT_PRECOMMITTED_ALIGNMENT if calibrated else PERMUTATION_SUMMARY_ALIGNMENT
        ),
        "output_coordinate_identity_stable": True,
        "source_coordinate_identity_preserved": calibrated,
        "concept_grounding_allowed": False,
    }
    coordinate_identity_sha256 = content_sha256(identity_fields)
    signal_fields = {
        "coordinate_identity_sha256": coordinate_identity_sha256,
        "source_cache_key": "2" * 64,
        "matrix_binding_sha256": "3" * 64,
        "column_values_sha256": f"{ordinal + 20:064x}",
        "shared_lineage_sha256": "4" * 64,
        "lineage_scope": "outer_train_context_oof_and_prediction_rows",
    }
    return DirectNumericalCoordinate(
        coordinate_id=f"direct_{ordinal:03d}",
        matrix_block=matrix_block,
        column_index=column_index,
        coordinate_name=coordinate_name,
        source_family=family,
        source_kind=source_kind,
        producer_subarchitecture=f"closed_producer_{ordinal:03d}",
        consumer_role="closed_effect_basis",
        observable_axes=(HETEROGENEITY_AXIS,),
        calibration_status=(NESTED_CALIBRATED_STATUS if calibrated else UNCALIBRATED_BASIS_STATUS),
        statistic_kind="direct_prediction" if calibrated else "signed_mean",
        statistic_rank=None,
        statistic_width=1,
        alignment_mode=(
            EXACT_PRECOMMITTED_ALIGNMENT if calibrated else PERMUTATION_SUMMARY_ALIGNMENT
        ),
        output_coordinate_identity_stable=True,
        source_coordinate_identity_preserved=calibrated,
        source_cache_key="2" * 64,
        matrix_binding_sha256="3" * 64,
        column_values_sha256=f"{ordinal + 20:064x}",
        context_nonzero_count=1,
        prediction_nonzero_count=1,
        combined_standard_deviation=1.0,
        observed_nonzero=True,
        observed_varying=True,
        shared_lineage_sha256="4" * 64,
        lineage_scope="outer_train_context_oof_and_prediction_rows",
        concept_grounding_allowed=False,
        coordinate_identity_sha256=coordinate_identity_sha256,
        signal_instance_sha256=content_sha256(signal_fields),
    )


def _manifest(catalog: RoleNeutralEvidenceCatalog) -> DirectUpstreamNumericalManifest:
    nonzero_families = [
        family for family in ACTIVE_STAGE1_CONCEPT_FAMILIES if family != TFIDF_SEMANTIC_RETRIEVAL
    ]
    coordinates = []
    raw_column = 0
    for ordinal, family in enumerate(nonzero_families, start=1):
        if ordinal == 1:
            block = CALIBRATED_SOURCES_BLOCK
            column = 0
        else:
            block = RAW_FEATURES_BLOCK
            column = raw_column
            raw_column += 1
        coordinates.append(
            _coordinate(
                family=family,
                ordinal=ordinal,
                matrix_block=block,
                column_index=column,
            )
        )
    widths = {CALIBRATED_SOURCES_BLOCK: 1, RAW_FEATURES_BLOCK: raw_column}
    matrices = tuple(
        AuthenticatedMatrixBinding(
            matrix_block=block,
            row_scope=scope,
            filename=f"{block}_{scope}.npy",
            sha256=content_sha256([block, scope]),
            shape=(2, widths[block]),
        )
        for block in MATRIX_BLOCKS
        for scope in ROW_SCOPES
    )
    by_family = {row.source_family: row for row in coordinates}
    coverages = []
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        coordinate = by_family.get(family)
        atom_ids = tuple(atom.evidence_id for atom in catalog.family_atoms(family))
        coverages.append(
            DirectNumericalFamilyCoverage(
                source_family=family,
                coordinate_ids=(() if coordinate is None else (coordinate.coordinate_id,)),
                source_kinds=(() if coordinate is None else (coordinate.source_kind,)),
                semantic_atom_ids=atom_ids,
                semantic_atom_ids_sha256=content_sha256(list(atom_ids)),
                numerical_zero_reason=(
                    "semantic_retrieval_has_no_independent_row_signal" if coordinate is None else ""
                ),
            )
        )
    return DirectUpstreamNumericalManifest(
        source_cache_schema="closed_test_cache_v1",
        source_cache_key="2" * 64,
        source_manifest_sha256="5" * 64,
        producer_identity_sha256="6" * 64,
        stable_output_schema_sha256="7" * 64,
        semantic_catalog_sha256=catalog.catalog_sha256,
        shared_lineage_sha256="4" * 64,
        lineage_scope="outer_train_context_oof_and_prediction_rows",
        matrices=matrices,
        coordinates=tuple(coordinates),
        family_coverage=tuple(coverages),
    )


def _reference_contract(
    catalog: RoleNeutralEvidenceCatalog,
) -> AuthenticatedReferenceOnlyDirectNumericalContract:
    return AuthenticatedReferenceOnlyDirectNumericalContract.create(
        outer_fold=catalog.outer_fold,
        context_epoch=0,
        plan_scientific_content_sha256="8" * 64,
        source_execution_content_sha256="9" * 64,
        reference_manifest_content_sha256="a" * 64,
        runtime_binding_content_sha256="b" * 64,
        provider_identity_sha256="c" * 64,
        spent_row_ids=(0, 1),
        gate_row_ids=(2, 3),
        catalog=catalog,
        family_coordinate_ids={
            family: (f"coordinate_{index:03d}",)
            for index, family in enumerate(
                ACTIVE_STAGE1_CONCEPT_FAMILIES,
                start=1,
            )
        },
        projection_content_sha256="d" * 64,
    )


def test_reference_only_direct_numerical_contract_binds_all_ten_families():
    catalog = _catalog()
    contract = _reference_contract(catalog)

    contract.verify(catalog=catalog)
    bindings = direct_numerical_bindings_from_reference_contract(
        contract,
        catalog=catalog,
    )

    assert tuple(row.source_family for row in bindings) == (
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    assert all(row.signal_count == 1 for row in bindings)
    assert all(
        row.direct_numerical_contract_sha256 == contract.content_sha256
        for row in bindings
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "spent_hash",
        "safety_flag",
        "duplicate_coordinate",
    ],
)
def test_reference_only_direct_numerical_contract_rejects_rehashed_forgery(
    mutation,
):
    wire = _reference_contract(_catalog()).as_dict()
    wire.pop("content_sha256")
    if mutation == "spent_hash":
        wire["spent_row_ids_sha256"] = "0" * 64
    elif mutation == "safety_flag":
        wire["row_values_included"] = True
    else:
        wire["family_coverage"][1]["coordinate_ids"] = list(
            wire["family_coverage"][0]["coordinate_ids"]
        )
        wire["family_coverage"][1]["coordinate_ids_sha256"] = content_sha256(
            wire["family_coverage"][1]["coordinate_ids"]
        )

    with pytest.raises(ValueError, match="reference-only"):
        AuthenticatedReferenceOnlyDirectNumericalContract(
            _body_json=canonical_json(wire),
            content_sha256=content_sha256(wire),
        )


class _StructuralIntentFixture(FirstGateMaterializationIntent):
    """Small approval-layer fixture; the intent module tests its own full schema."""

    def __init__(self, body):
        object.__setattr__(self, "_body_json", canonical_json(body))
        object.__setattr__(self, "content_sha256", content_sha256(body))

    def verify(self) -> None:
        return None


def _intent(catalog: RoleNeutralEvidenceCatalog) -> FirstGateMaterializationIntent:
    family_bindings = [
        {
            "source_family": family,
            "semantic_atom_ids": [atom.evidence_id for atom in catalog.family_atoms(family)],
            "semantic_atom_ids_sha256": content_sha256(
                [atom.evidence_id for atom in catalog.family_atoms(family)]
            ),
        }
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    ]
    family_coverage = [
        {
            **row,
            "coordinate_ids": (
                [] if row["source_family"] == TFIDF_SEMANTIC_RETRIEVAL else [f"num_{index:03d}"]
            ),
            "numerical_zero_reason": (
                "semantic_retrieval_has_no_independent_row_signal"
                if row["source_family"] == TFIDF_SEMANTIC_RETRIEVAL
                else ""
            ),
        }
        for index, row in enumerate(family_bindings, start=1)
    ]
    return _StructuralIntentFixture(
        {
            "outer_fold": catalog.outer_fold,
            "source_cache_key": "2" * 64,
            "semantic_catalog": {
                "catalog_sha256": catalog.catalog_sha256,
                "scope": catalog.scope,
                "inner_fold": catalog.inner_fold,
                "split_fingerprint": catalog.split_fingerprint,
                "atom_count": len(catalog.atoms),
                "family_bindings": family_bindings,
            },
            "coordinate_schema": {
                "family_coverage": family_coverage,
                "stable_output_schema_sha256": "3" * 64,
                "expected_shared_lineage_sha256": "4" * 64,
                "lineage_scope": "closed_first_gate_intent_test_scope",
            },
        }
    )


class _MetadataRunner:
    def __init__(self) -> None:
        body = {
            "schema_version": "closed_metadata_runner_v1",
            "endpoint_urls": ["http://offline.test/v1"],
            "model": {"name": "closed-json-model", "resolution": "explicit"},
            "retry": {"max_attempts": 1},
        }
        self._identity = {**body, "identity_sha256": content_sha256(body)}
        self.calls = []
        self._metadata = []

    def identity(self):
        return json.loads(canonical_json(self._identity))

    @property
    def execution_metadata(self):
        return tuple(json.loads(canonical_json(row)) for row in self._metadata)

    def _response(self, job):
        request = json.loads(job.messages[1]["content"])
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
            return {
                "decision": "accept",
                "canonical_name": provisional["canonical_name"],
                "description": provisional["description"],
                "unresolved_ambiguity": provisional["unresolved_ambiguity"],
                "input_dispositions": {
                    review_id: {
                        "action": "integrated",
                        "reason": "The exact review input was folded.",
                    }
                    for review_id in request["review_input_ids"]
                },
                "complete_support_reviewed": True,
                "reason": "The complete support establishes one patient measurement.",
            }
        if request["job"] == "review_extraction_feature_evidence":
            return {
                "measurement_observation": "One supported patient measurement is present.",
                "shape_observation": "continuous",
                "literal_aliases": [],
                "literal_units": ["documented_clinical_unit"],
                "literal_categories": [],
                "literal_distinctions": [],
                "missing_or_ambiguous": "The exact representation remains unresolved.",
                "reviewed_evidence": True,
            }
        if request["job"] == "fold_extraction_evidence_definitions":
            return {
                "feature_name": request["canonical_name"],
                "measurement": "Extract this one supported patient measurement.",
                "representation": {
                    "kind": "continuous",
                    "unit": "documented_clinical_unit",
                    "categories": [],
                },
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
            return {
                "evidence_dispositions": {
                    item["evidence_id"]: {
                        "evidence_findings": [
                            {
                                "feature_name": f"{item['source_family']}_measure",
                                "description": (
                                    "Patient measurement supported by this architecture."
                                ),
                                "value_shape_hypothesis": "continuous",
                                "unresolved_ambiguity": "",
                            }
                        ],
                        "member_dispositions": {
                            member: {"findings": []} for member in item["member_ids"]
                        },
                        "reason": "The complete atom supports this measurement.",
                    }
                    for item in request["evidence"]
                },
            }
        if job.job_kind == CONSOLIDATE_ARCHITECTURE_JOB:
            candidates = request["candidates"]
            slots = request["identifier_ownership"]["identifier_domains"]["cluster_slots"]
            slot_by_candidate = {
                candidate["candidate_id"]: slots[index]
                for index, candidate in enumerate(candidates)
            }
            candidate_by_slot = {
                slot_by_candidate[candidate["candidate_id"]]: candidate for candidate in candidates
            }
            return {
                "candidate_assignments": {
                    candidate["candidate_id"]: {
                        "cluster_slot": slot_by_candidate[candidate["candidate_id"]],
                        "reason": "Keep this architecture-local candidate.",
                    }
                    for candidate in candidates
                },
                "slot_definitions": {
                    slot: {
                        "canonical_name": candidate_by_slot[slot]["feature_name"],
                        "description": candidate_by_slot[slot]["description"],
                        "unresolved_ambiguity": candidate_by_slot[slot]["unresolved_ambiguity"],
                    }
                    for slot in slots
                },
            }
        if job.job_kind == COVERAGE_CRITIC_JOB:
            return {
                "findings": [],
                "reviewed_evidence_ids": {row["evidence_id"]: True for row in request["evidence"]},
            }
        if job.job_kind == CROSS_ARCHITECTURE_PLANNER_JOB:
            candidates = [
                candidate
                for dossier in request["architecture_dossiers"]
                for candidate in dossier["architecture_candidates"]
            ]
            group_slots = request["identifier_ownership"]["identifier_domains"][
                "planner_group_slots"
            ]
            lookback_slots = request["identifier_ownership"]["identifier_domains"][
                "planner_lookback_slots"
            ]
            slot_by_candidate = {
                candidate["candidate_id"]: group_slots[index]
                for index, candidate in enumerate(candidates)
            }
            candidate_by_slot = {
                slot_by_candidate[candidate["candidate_id"]]: candidate for candidate in candidates
            }
            evidence_ids = [
                evidence_id
                for candidate in candidates
                for evidence_id in candidate["supporting_evidence_ids"]
            ]
            return {
                "candidate_assignments": {
                    candidate["candidate_id"]: {
                        "group_slot": slot_by_candidate[candidate["candidate_id"]],
                    }
                    for candidate in candidates
                },
                "group_slot_definitions": {
                    slot: {
                        "provisional_name": candidate_by_slot[slot]["feature_name"],
                        "reason": "Keep this supported measurement distinct.",
                    }
                    for slot in group_slots
                },
                "lookback_slot_definitions": {
                    slot: {
                        "selection": evidence_ids[index] if index < len(evidence_ids) else "unused",
                        "question": "Confirm the documented clinical unit.",
                        "reason": "Ground the bounded extraction unit vocabulary.",
                    }
                    for index, slot in enumerate(lookback_slots)
                },
            }
        if job.job_kind == CROSS_ARCHITECTURE_INTEGRATION_JOB:
            candidates = [
                candidate
                for dossier in request["architecture_context"]["architecture_dossiers"]
                for candidate in dossier["architecture_candidates"]
            ]
            slots = request["identifier_ownership"]["identifier_domains"]["integration_slots"]
            slot_by_candidate = {
                candidate["candidate_id"]: slots[index]
                for index, candidate in enumerate(candidates)
            }
            candidate_by_slot = {
                slot_by_candidate[candidate["candidate_id"]]: candidate for candidate in candidates
            }
            return {
                "candidate_routes": {
                    candidate["candidate_id"]: {
                        "route": slot_by_candidate[candidate["candidate_id"]],
                        "reason": "The candidate is supported and distinct.",
                    }
                    for candidate in candidates
                },
                "slot_definitions": {
                    slot: {
                        "canonical_name": candidate_by_slot[slot]["feature_name"],
                        "description": candidate_by_slot[slot]["description"],
                        "unresolved_ambiguity": candidate_by_slot[slot]["unresolved_ambiguity"],
                    }
                    for slot in slots
                },
            }
        if job.job_kind == EXTRACTION_DEFINITION_JOB:
            return {
                "feature_name": request["canonical_name"],
                "measurement": "Extract the supported patient measurement.",
                "representation": {
                    "kind": "continuous",
                    "unit": "documented_clinical_unit",
                    "categories": [],
                },
                "aliases": [],
                "distinguish_from": [],
                "missing_or_ambiguous": "Return null when absent or ambiguous.",
                "supporting_evidence_reviewed": True,
            }
        raise AssertionError(f"unexpected job kind: {job.job_kind}")

    def run_json(self, *, job):
        self.calls.append(job)
        response = self._response(job)
        request_sha = content_sha256(job.as_dict())
        response_sha = content_sha256(response)
        raw = canonical_json(response)
        identity_sha = self._identity["identity_sha256"]
        attempt = {
            "attempt_number": 1,
            "endpoint": "http://offline.test/v1",
            "model": "closed-json-model",
            "response_model": "closed-json-model",
            "finish_reason": "stop",
            "request_sha256": request_sha,
            "runner_identity_sha256": identity_sha,
            "outcome": "success",
            "retryable": False,
            "will_retry": False,
            "usage": {},
            "content_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "raw_transport_bytes": len(raw.encode("utf-8")),
            "reasoning_hashes": {},
            "parsed_response_sha256": response_sha,
        }
        self._metadata.append(
            {
                "job_id": job.job_id,
                "job_kind": job.job_kind,
                "request_sha256": request_sha,
                "runner_identity_sha256": identity_sha,
                "outcome": "success",
                "parsed_response_sha256": response_sha,
                "attempts": [attempt],
            }
        )
        return response


def _agent(*, runner=None, job_cache=None):
    catalog = _catalog()
    manifest = _manifest(catalog)
    runner = runner or _MetadataRunner()
    agent = ApprovedHierarchicalDiscoveryAgent(
        catalog=catalog,
        chunk_plan=build_complete_architecture_chunks(
            catalog, max_atoms_per_chunk=1, max_bytes_per_chunk=20_000
        ),
        family_explanations={
            family: f"Interpret complete evidence from {family}."
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        direct_numerical_manifest=manifest,
        direct_numerical_bindings=direct_numerical_bindings_from_manifest(manifest),
        runner=runner,
        config=HierarchicalDiscoveryConfig(max_integrated_features=10),
        job_cache=job_cache,
    )
    return agent, runner


class _FailOnceRunner(_MetadataRunner):
    def __init__(self, *, fail_on_remote_attempt):
        super().__init__()
        self.fail_on_remote_attempt = fail_on_remote_attempt
        self.remote_attempts = []
        self.failed = False

    def run_json(self, *, job):
        self.remote_attempts.append(job)
        if len(self.remote_attempts) == self.fail_on_remote_attempt and not self.failed:
            self.failed = True
            raise RuntimeError("one later remote job failed")
        return super().run_json(job=job)


class _InvalidFirstSemanticRunner(_MetadataRunner):
    def __init__(self):
        super().__init__()
        self.response_count = 0

    def _response(self, job):
        self.response_count += 1
        response = super()._response(job)
        if self.response_count == 1:
            first = next(iter(response["evidence_dispositions"].values()))
            first["member_dispositions"]["member_malicious"] = {"findings": []}
        return response


class _PoisonedSuccessfulMetadataRunner(_MetadataRunner):
    def run_json(self, *, job):
        response = super().run_json(job=job)
        self._metadata[-1]["parsed_response_sha256"] = "0" * 64
        return response


class _PoisonedRepairMetadataRunner(_InvalidFirstSemanticRunner):
    def run_json(self, *, job):
        response = super().run_json(job=job)
        if self.response_count == 2:
            self._metadata[-1]["attempts"][-1]["parsed_response_sha256"] = "0" * 64
        return response


def _tamper_response_envelope_metadata(*, attempt, field, mode):
    if mode == "missing":
        attempt.pop(field, None)
        return
    if field == "response_model":
        attempt[field] = "substituted-model"
        return
    if field == "finish_reason" and mode in {"length", "content_filter"}:
        attempt[field] = mode
        return
    raise AssertionError("unsupported response-envelope metadata tamper")


class _PoisonedSuccessfulResponseEnvelopeRunner(_MetadataRunner):
    def __init__(self, *, field, mode):
        super().__init__()
        self.field = field
        self.mode = mode

    def run_json(self, *, job):
        response = super().run_json(job=job)
        _tamper_response_envelope_metadata(
            attempt=self._metadata[-1]["attempts"][-1],
            field=self.field,
            mode=self.mode,
        )
        return response


class _PoisonedRepairResponseEnvelopeRunner(_InvalidFirstSemanticRunner):
    def __init__(self, *, field, mode):
        super().__init__()
        self.field = field
        self.mode = mode

    def run_json(self, *, job):
        response = super().run_json(job=job)
        if self.response_count == 2:
            _tamper_response_envelope_metadata(
                attempt=self._metadata[-1]["attempts"][-1],
                field=self.field,
                mode=self.mode,
            )
        return response


class _InvalidFirstStrictJsonRunner(_MetadataRunner):
    def __init__(self):
        super().__init__()
        self.failed = False

    def run_json(self, *, job):
        if self.failed:
            return super().run_json(job=job)
        self.failed = True
        self.calls.append(job)
        content = "not one JSON object"
        content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        request_sha = content_sha256(job.as_dict())
        identity_sha = self._identity["identity_sha256"]
        attempt = {
            "attempt_number": 1,
            "endpoint": "http://offline.test/v1",
            "model": "closed-json-model",
            "response_model": "closed-json-model",
            "finish_reason": "stop",
            "request_sha256": request_sha,
            "runner_identity_sha256": identity_sha,
            "outcome": "invalid_response",
            "retryable": False,
            "will_retry": False,
            "exception_type": "InvalidDiscoveryJsonResponse",
            "usage": {},
            "content_sha256": content_sha,
            "raw_transport_bytes": len(content.encode("utf-8")),
            "reasoning_hashes": {},
        }
        self._metadata.append(
            {
                "job_id": job.job_id,
                "job_kind": job.job_kind,
                "request_sha256": request_sha,
                "runner_identity_sha256": identity_sha,
                "outcome": "invalid_response",
                "attempts": [attempt],
            }
        )
        raise InvalidDiscoveryJsonResponse(failed_response_content=content)


class _InvalidFirstRawTransportRunner(_MetadataRunner):
    def __init__(self):
        super().__init__()
        self.failed = False

    def run_json(self, *, job):
        if self.failed:
            return super().run_json(job=job)
        self.failed = True
        self.calls.append(job)
        content = "oversized raw transport response"
        content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        request_sha = content_sha256(job.as_dict())
        identity_sha = self._identity["identity_sha256"]
        attempt = {
            "attempt_number": 1,
            "endpoint": "http://offline.test/v1",
            "model": "closed-json-model",
            "response_model": "closed-json-model",
            "finish_reason": "stop",
            "request_sha256": request_sha,
            "runner_identity_sha256": identity_sha,
            "outcome": "invalid_response",
            "retryable": False,
            "will_retry": False,
            "exception_type": "InvalidDiscoveryTransportResponse",
            "usage": {},
            "content_sha256": content_sha,
            "raw_transport_bytes": len(content.encode("utf-8")),
            "reasoning_hashes": {},
        }
        self._metadata.append(
            {
                "job_id": job.job_id,
                "job_kind": job.job_kind,
                "request_sha256": request_sha,
                "runner_identity_sha256": identity_sha,
                "outcome": "invalid_response",
                "attempts": [attempt],
            }
        )
        raise InvalidDiscoveryTransportResponse(failed_response_content=content)


class _PoisonedInvalidResponseEnvelopeRunner(_InvalidFirstStrictJsonRunner):
    def __init__(self, *, field, mode):
        super().__init__()
        self.field = field
        self.mode = mode

    def run_json(self, *, job):
        try:
            return super().run_json(job=job)
        except InvalidDiscoveryJsonResponse:
            _tamper_response_envelope_metadata(
                attempt=self._metadata[-1]["attempts"][-1],
                field=self.field,
                mode=self.mode,
            )
            raise


def test_offline_wrapper_binds_every_architecture_without_direct_values():
    agent, _runner = _agent()
    packet = agent.precommit.packet
    numerical = packet["direct_numerical_contract_binding"]

    assert numerical["direct_numerical_contract_kind"] == (
        DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
    )
    assert numerical["direct_numerical_contract_sha256"] == (
        agent.direct_numerical_manifest.content_sha256
    )
    assert len(numerical["families"]) == 10
    assert [row["source_family"] for row in numerical["families"]] == list(
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    assert numerical["row_values_included"] is False
    assert numerical["coordinate_metadata_included"] is False
    assert "coordinates" not in numerical
    assert "matrix_authentication" not in numerical
    assert packet["hierarchy_precommit"]["precommit_sha256"] == (agent.inner_precommit_sha256)
    assert packet["chunk_plan_binding"]["max_semantic_member_ids_per_chunk"] == 3
    assert (
        packet["hierarchy_precommit"]["packet"]["chunk_plan_binding"][
            "max_semantic_member_ids_per_chunk"
        ]
        == 3
    )
    assert len(packet["direct_numerical_dossier_bindings"]) == 10
    assert packet["assurances"]["runner_wire_hash_authenticated_before_cache_write"] is True
    assert _agent()[0].precommit.approval_sha256 == agent.precommit.approval_sha256


def test_prepare_threads_semantic_member_bound_into_plan_config_and_both_packets(monkeypatch):
    catalog = _catalog()
    monkeypatch.setattr(
        approved_module,
        "build_role_neutral_evidence_catalog",
        lambda _inputs: catalog,
    )
    config = HierarchicalDiscoveryConfig(
        max_integrated_features=10,
        max_semantic_member_ids_per_chunk=1,
    )
    agent = ApprovedHierarchicalDiscoveryAgent.prepare_from_evidence_inputs(
        evidence_inputs=(),
        family_explanations={
            family: f"Interpret complete evidence from {family}."
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        direct_numerical_manifest=_manifest(catalog),
        runner=_MetadataRunner(),
        config=config,
        max_atoms_per_chunk=2,
        max_bytes_per_chunk=20_000,
    )

    packet = agent.precommit.packet
    assert agent.chunk_plan.max_semantic_member_ids_per_chunk == 1
    assert packet["chunk_plan_binding"]["max_semantic_member_ids_per_chunk"] == 1
    assert packet["config_bounds"]["max_semantic_member_ids_per_chunk"] == 1
    assert (
        packet["hierarchy_precommit"]["packet"]["chunk_plan_binding"][
            "max_semantic_member_ids_per_chunk"
        ]
        == 1
    )

    with pytest.raises(ValueError, match="same max_semantic_member_ids_per_chunk"):
        ApprovedHierarchicalDiscoveryAgent.prepare_from_evidence_inputs(
            evidence_inputs=(),
            family_explanations={
                family: f"Interpret complete evidence from {family}."
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            },
            direct_numerical_manifest=_manifest(catalog),
            runner=_MetadataRunner(),
            config=config,
            max_semantic_member_ids_per_chunk=2,
        )

    with pytest.raises(TypeError, match="config must be HierarchicalDiscoveryConfig"):
        ApprovedHierarchicalDiscoveryAgent.prepare_from_evidence_inputs(
            evidence_inputs=(),
            family_explanations={},
            direct_numerical_manifest=_manifest(catalog),
            runner=_MetadataRunner(),
            config=object(),  # type: ignore[arg-type]
        )


def test_pre_fit_intent_drives_exact_bindings_and_remains_honestly_named():
    catalog = _catalog()
    intent = _intent(catalog)
    bindings = direct_numerical_bindings_from_intent(intent, catalog=catalog)
    runner = _MetadataRunner()
    agent = ApprovedHierarchicalDiscoveryAgent(
        catalog=catalog,
        chunk_plan=build_complete_architecture_chunks(
            catalog, max_atoms_per_chunk=1, max_bytes_per_chunk=20_000
        ),
        family_explanations={
            family: f"Interpret complete evidence from {family}."
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        first_gate_materialization_intent=intent,
        direct_numerical_bindings=bindings,
        runner=runner,
        config=HierarchicalDiscoveryConfig(max_integrated_features=10),
    )

    assert [row.source_family for row in bindings] == list(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    assert [row.signal_count for row in bindings] == [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    assert bindings[6].zero_reason == ("semantic_retrieval_has_no_independent_row_signal")
    packet = agent.precommit.packet
    numerical = packet["direct_numerical_contract_binding"]
    assert numerical["direct_numerical_contract_kind"] == (
        DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
    )
    assert numerical["direct_numerical_contract_sha256"] == intent.content_sha256
    assert "manifest_sha256" not in numerical
    assert "direct_numerical_manifest_binding" not in packet
    assert numerical["row_values_included"] is False
    assert numerical["coordinate_metadata_included"] is False

    result = agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)
    assert result.direct_numerical_contract_kind == (
        DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
    )
    assert result.direct_numerical_contract_sha256 == intent.content_sha256
    assert result.direct_numerical_manifest_sha256 is None
    assert all(
        row["validated_against_approved_contract"]
        and not row["validated_against_full_manifest"]
        and row["direct_numerical_contract_kind"]
        == DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
        for row in result.numerical_binding_audit
    )
    assert all(
        dossier.direct_numerical_manifest_sha256 == ""
        and dossier.direct_numerical_contract_sha256 == intent.content_sha256
        for dossier in result.completed.dossiers
    )
    for job in result.completed.execution_ledger.job_ledger.jobs:
        rendered = canonical_json(job.messages)
        assert intent.content_sha256 not in rendered
        assert "direct_numerical_contract" not in rendered


def test_agent_requires_exactly_one_contract_and_intent_atom_bindings_are_exact():
    catalog = _catalog()
    intent = _intent(catalog)
    manifest = _manifest(catalog)
    common = {
        "catalog": catalog,
        "chunk_plan": build_complete_architecture_chunks(
            catalog, max_atoms_per_chunk=1, max_bytes_per_chunk=20_000
        ),
        "family_explanations": {
            family: f"Interpret complete evidence from {family}."
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        "runner": _MetadataRunner(),
    }
    with pytest.raises(ValueError, match="exactly one"):
        ApprovedHierarchicalDiscoveryAgent(**common)
    with pytest.raises(ValueError, match="exactly one"):
        ApprovedHierarchicalDiscoveryAgent(
            **common,
            direct_numerical_manifest=manifest,
            first_gate_materialization_intent=intent,
        )

    body = intent.body
    body["semantic_catalog"]["family_bindings"][0]["semantic_atom_ids"] = ["evidence_wrong"]
    wrong = _StructuralIntentFixture(body)
    with pytest.raises(ValueError, match="evidence IDs differ"):
        direct_numerical_bindings_from_intent(wrong, catalog=catalog)


def test_wrong_wrapper_approval_makes_no_job_calls():
    agent, runner = _agent()

    with pytest.raises(ValueError, match="does not match"):
        agent.execute(approved_wrapper_sha256="0" * 64)

    assert runner.calls == []
    assert runner.execution_metadata == ()


def test_cache_binding_static_preflight_and_wrong_approval_do_not_read_cache(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "cache_symlink"
    root.symlink_to(target, target_is_directory=True)
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    agent, runner = _agent(job_cache=cache)

    binding = agent.precommit.packet["job_cache_binding"]
    hierarchy_bundle_sha256 = agent.precommit.packet["hierarchy_precommit"]["packet"][
        "orchestrator_implementation_bundle_sha256"
    ]
    assert binding["mode"] == "authenticated_immutable"
    assert binding["validator_code_sha256"] == hierarchy_bundle_sha256
    assert binding["identity"]["root_envelope"]["absolute_path"] == str(root)
    agent.validate_precommit_unchanged()
    with pytest.raises(ValueError, match="does not match"):
        agent.execute(approved_wrapper_sha256="0" * 64)
    assert runner.calls == []

    with pytest.raises(ValueError, match="root cannot be a symlink"):
        agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)
    assert runner.calls == []


def test_invalid_semantic_response_is_repaired_then_only_validated_output_is_cached(tmp_path):
    cache_root = tmp_path / "cache"
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=cache_root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    runner = _InvalidFirstSemanticRunner()
    agent, _ = _agent(runner=runner, job_cache=cache)

    result = agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)

    assert cache_root.is_dir()
    assert runner.response_count == len(result.completed.execution_ledger.results) + 1
    first = result.completed.execution_ledger.results[0]
    assert [row["validation_outcome"] for row in first.response_attempt_trace["attempts"]] == [
        "local_json_schema_validation_failure",
        "validated_response",
    ]
    trace_record = result.runner_trace.records[0]
    assert trace_record["record_type"] == "authenticated_remote_response_repair_sequence"
    assert trace_record["response_attempt_trace_sha256"] == (first.response_attempt_trace_sha256)
    # The malicious invalid citation is carried only as a hashed response
    # projection; the immutable cache exposes the final validated response.
    assert trace_record["validated_response_sha256"] == first.response_sha256
    malicious = "evidence_malicious_ignore_prior_instructions_999"
    repair_job = runner.calls[1]
    assert malicious not in canonical_json(repair_job.messages)
    assert malicious not in canonical_json(first.response_attempt_trace)
    assert malicious not in canonical_json(trace_record)
    assert all(
        malicious not in path.read_text(encoding="utf-8") for path in cache_root.glob("*/*.json")
    )
    result.validate_authentication()


def test_poisoned_raw_metadata_fails_before_cache_write_and_cannot_seed_replay(tmp_path):
    cache_root = tmp_path / "cache"
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=cache_root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    poisoned_agent, poisoned_runner = _agent(
        runner=_PoisonedSuccessfulMetadataRunner(),
        job_cache=cache,
    )

    with pytest.raises(ValueError, match="authenticated raw projection"):
        poisoned_agent.execute(approved_wrapper_sha256=poisoned_agent.precommit.approval_sha256)

    assert len(poisoned_runner.calls) == 1
    assert list(cache_root.glob("*/*.json")) == []

    clean_runner = _MetadataRunner()
    replay_agent, _ = _agent(runner=clean_runner, job_cache=cache)
    result = replay_agent.execute(approved_wrapper_sha256=replay_agent.precommit.approval_sha256)

    assert clean_runner.calls
    assert all(
        record.get("record_type") != "authenticated_cache_hit"
        for record in result.runner_trace.records
    )
    result.validate_authentication()


def test_poisoned_repair_raw_metadata_fails_before_cache_write(tmp_path):
    cache_root = tmp_path / "cache"
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=cache_root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    agent, runner = _agent(runner=_PoisonedRepairMetadataRunner(), job_cache=cache)

    with pytest.raises(ValueError, match="final parsed response hash"):
        agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)

    assert runner.response_count == 2
    assert len(runner.calls) == 2
    assert list(cache_root.glob("*/*.json")) == []


@pytest.mark.parametrize(
    ("field", "mode", "message"),
    (
        ("response_model", "wrong", "response model differs"),
        ("response_model", "missing", "response model differs"),
        ("finish_reason", "length", "finish_reason must be exactly 'stop'"),
        ("finish_reason", "content_filter", "finish_reason must be exactly 'stop'"),
        ("finish_reason", "missing", "finish_reason must be exactly 'stop'"),
    ),
)
def test_initial_response_envelope_metadata_fails_before_cache_write(
    tmp_path, field, mode, message
):
    cache_root = tmp_path / "cache"
    runner = _PoisonedSuccessfulResponseEnvelopeRunner(field=field, mode=mode)
    agent, _ = _agent(
        runner=runner,
        job_cache=AuthenticatedHierarchicalDiscoveryJobCache(
            root=cache_root,
            config=HIERARCHY_JOB_CACHE_CONFIG,
        ),
    )

    with pytest.raises(ValueError, match=message):
        agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)

    assert len(runner.calls) == 1
    assert list(cache_root.glob("*/*.json")) == []


@pytest.mark.parametrize(
    ("field", "mode", "message"),
    (
        ("response_model", "wrong", "response model differs"),
        ("response_model", "missing", "response model differs"),
        ("finish_reason", "length", "finish_reason must be exactly 'stop'"),
        ("finish_reason", "content_filter", "finish_reason must be exactly 'stop'"),
        ("finish_reason", "missing", "finish_reason must be exactly 'stop'"),
    ),
)
def test_repair_response_envelope_metadata_fails_before_cache_write(tmp_path, field, mode, message):
    cache_root = tmp_path / "cache"
    runner = _PoisonedRepairResponseEnvelopeRunner(field=field, mode=mode)
    agent, _ = _agent(
        runner=runner,
        job_cache=AuthenticatedHierarchicalDiscoveryJobCache(
            root=cache_root,
            config=HIERARCHY_JOB_CACHE_CONFIG,
        ),
    )

    with pytest.raises(ValueError, match=message):
        agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)

    assert runner.response_count == 2
    assert len(runner.calls) == 2
    assert list(cache_root.glob("*/*.json")) == []


@pytest.mark.parametrize(
    ("field", "mode", "message"),
    (
        ("response_model", "wrong", "response model differs"),
        ("finish_reason", "length", "finish_reason must be exactly 'stop'"),
    ),
)
def test_invalid_initial_response_envelope_is_rejected_before_repair_or_cache(
    tmp_path, field, mode, message
):
    cache_root = tmp_path / "cache"
    runner = _PoisonedInvalidResponseEnvelopeRunner(field=field, mode=mode)
    agent, _ = _agent(
        runner=runner,
        job_cache=AuthenticatedHierarchicalDiscoveryJobCache(
            root=cache_root,
            config=HIERARCHY_JOB_CACHE_CONFIG,
        ),
    )

    with pytest.raises(ValueError, match=message):
        agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)

    assert len(runner.calls) == 1
    assert list(cache_root.glob("*/*.json")) == []


def test_wrong_response_model_cannot_seed_a_later_cache_replay(tmp_path):
    cache_root = tmp_path / "cache"
    poisoned_runner = _PoisonedSuccessfulResponseEnvelopeRunner(
        field="response_model",
        mode="wrong",
    )
    poisoned_agent, _ = _agent(
        runner=poisoned_runner,
        job_cache=AuthenticatedHierarchicalDiscoveryJobCache(
            root=cache_root,
            config=HIERARCHY_JOB_CACHE_CONFIG,
        ),
    )
    with pytest.raises(ValueError, match="response model differs"):
        poisoned_agent.execute(approved_wrapper_sha256=poisoned_agent.precommit.approval_sha256)
    assert list(cache_root.glob("*/*.json")) == []

    clean_runner = _FailOnceRunner(fail_on_remote_attempt=2)
    replay_agent, _ = _agent(
        runner=clean_runner,
        job_cache=AuthenticatedHierarchicalDiscoveryJobCache(
            root=cache_root,
            config=HIERARCHY_JOB_CACHE_CONFIG,
        ),
    )
    with pytest.raises(RuntimeError, match="one later remote job failed"):
        replay_agent.execute(approved_wrapper_sha256=replay_agent.precommit.approval_sha256)

    assert len(clean_runner.remote_attempts) == 2
    assert len(clean_runner.calls) == 1


def test_strict_json_failure_metadata_is_authenticated_before_successful_repair(tmp_path):
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=tmp_path / "cache",
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    agent, runner = _agent(runner=_InvalidFirstStrictJsonRunner(), job_cache=cache)

    result = agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)

    first = result.completed.execution_ledger.results[0]
    assert [row["validation_outcome"] for row in first.response_attempt_trace["attempts"]] == [
        "strict_json_parse_failure",
        "validated_response",
    ]
    assert len(runner.calls) == len(result.completed.execution_ledger.results) + 1
    assert result.runner_trace.records[0]["record_type"] == (
        "authenticated_remote_response_repair_sequence"
    )
    result.validate_authentication()


def test_raw_transport_budget_failure_is_authenticated_before_successful_repair(tmp_path):
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=tmp_path / "cache",
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    agent, runner = _agent(runner=_InvalidFirstRawTransportRunner(), job_cache=cache)

    result = agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)

    first = result.completed.execution_ledger.results[0]
    assert [row["validation_outcome"] for row in first.response_attempt_trace["attempts"]] == [
        "raw_transport_budget_failure",
        "validated_response",
    ]
    assert len(runner.calls) == len(result.completed.execution_ledger.results) + 1
    assert result.runner_trace.records[0]["record_type"] == (
        "authenticated_remote_response_repair_sequence"
    )
    result.validate_authentication()


def test_fully_rehashed_fabricated_repair_trace_fails_on_cache_replay(tmp_path):
    cache_root = tmp_path / "cache"
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=cache_root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    agent, _ = _agent(runner=_InvalidFirstSemanticRunner(), job_cache=cache)
    agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)

    repaired_entries = []
    for path in cache_root.glob("*/*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if len(entry["response_attempt_trace"]["attempts"]) == 2:
            repaired_entries.append((path, entry))
    assert len(repaired_entries) == 1
    old_path, entry = repaired_entries[0]
    trace = entry["response_attempt_trace"]
    trace["attempts"][1]["response_repair_binding"]["policy_sha256"] = "0" * 64
    trace_body = {key: value for key, value in trace.items() if key != "trace_sha256"}
    trace["trace_sha256"] = content_sha256(trace_body)
    entry["response_attempt_trace_sha256"] = content_sha256(trace)
    entry_body = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = content_sha256(entry_body)
    new_path = old_path.with_name(f"entry_{entry['entry_sha256']}.json")
    old_path.rename(new_path)
    new_path.write_text(canonical_json(entry), encoding="utf-8")

    replay_runner = _MetadataRunner()
    replay_agent, _ = _agent(runner=replay_runner, job_cache=cache)
    with pytest.raises(ValueError, match="exact deterministic repair job"):
        replay_agent.execute(approved_wrapper_sha256=replay_agent.precommit.approval_sha256)
    assert replay_runner.calls == []


def test_dynamic_jobs_resume_after_later_failure_and_trace_cache_hits(tmp_path):
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=tmp_path / "cache",
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    runner = _FailOnceRunner(fail_on_remote_attempt=5)
    agent, _ = _agent(runner=runner, job_cache=cache)

    with pytest.raises(RuntimeError, match="later remote job failed"):
        agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)
    first_successful_job_ids = tuple(job.job_id for job in runner.calls)
    assert len(first_successful_job_ids) == 4

    result = agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)

    records = result.runner_trace.records
    assert len(records) == len(result.completed.execution_ledger.job_ledger.jobs)
    assert tuple(row["job_id"] for row in records[:4]) == first_successful_job_ids
    assert all(row.get("record_type") == "authenticated_cache_hit" for row in records[:4])
    assert all("attempts" not in row for row in records[:4])
    assert all(row.get("outcome") == "success" for row in records[4:])
    assert tuple(job.job_id for job in runner.remote_attempts[5:]) == tuple(
        job.job_id for job in result.completed.execution_ledger.job_ledger.jobs[4:]
    )
    assert result.runner_trace.cache_identity == cache.identity()
    assert (
        result.runner_trace.validator_code_sha256
        == agent.precommit.packet["hierarchy_precommit"]["packet"][
            "orchestrator_implementation_bundle_sha256"
        ]
    )
    assert all(
        row["validator_code_sha256"] == result.runner_trace.validator_code_sha256
        for row in records[:4]
    )
    result.validate_authentication()

    altered = list(records)
    altered[0]["cache_entry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="entry hash differs"):
        AuthenticatedRunnerExecutionTrace.create(
            completed=result.completed,
            runner_identity=result.runner_trace.runner_identity,
            records=altered,
            validator_code_sha256=result.runner_trace.validator_code_sha256,
            cache_identity=result.runner_trace.cache_identity,
        )


def test_execute_revalidates_dossiers_compiles_and_authenticates_retry_records():
    agent, runner = _agent()
    result = agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)

    assert len(result.completed.dossiers) == 10
    assert len(result.compiled_registry.contracts) == 10
    assert len(result.runner_trace.records) == len(runner.calls)
    assert [row["job_id"] for row in result.runner_trace.records] == [
        job.job_id for job in result.completed.execution_ledger.job_ledger.jobs
    ]
    assert [row["source_family"] for row in result.numerical_binding_audit] == list(
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    assert all(row["validated_against_full_manifest"] for row in result.numerical_binding_audit)
    first_result = result.completed.execution_ledger.results[0]
    first_wire = runner._response(runner.calls[0])
    first_wire_sha256 = content_sha256(first_wire)
    first_record = result.runner_trace.records[0]
    assert first_record["parsed_response_sha256"] == first_wire_sha256
    assert first_record["attempts"][-1]["parsed_response_sha256"] == first_wire_sha256
    assert first_result.raw_wire_response_sha256 == first_wire_sha256
    assert first_result.response_sha256 != first_wire_sha256
    result.validate_authentication()
    with pytest.raises(ValueError, match="does not authenticate"):
        replace(result.runner_trace, trace_sha256="0" * 64)
    altered_records = list(result.runner_trace.records)
    altered_records[0]["attempts"][0]["parsed_response_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="response hash differs"):
        AuthenticatedRunnerExecutionTrace.create(
            completed=result.completed,
            runner_identity=result.runner_trace.runner_identity,
            records=altered_records,
            validator_code_sha256=result.runner_trace.validator_code_sha256,
            cache_identity=result.runner_trace.cache_identity,
        )


def test_global_catalog_family_ids_counts_and_explicit_bindings_fail_closed():
    catalog = _catalog()
    manifest = _manifest(catalog)
    plan = build_complete_architecture_chunks(
        catalog, max_atoms_per_chunk=1, max_bytes_per_chunk=20_000
    )
    explanations = {
        family: f"Interpret complete evidence from {family}."
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    bindings = direct_numerical_bindings_from_manifest(manifest)

    with pytest.raises(ValueError, match="exactly ten"):
        ApprovedHierarchicalDiscoveryAgent(
            catalog=catalog,
            chunk_plan=plan,
            family_explanations=explanations,
            direct_numerical_manifest=manifest,
            direct_numerical_bindings=bindings[:-1],
            runner=_MetadataRunner(),
        )

    changed = replace(bindings[0], signal_count=bindings[0].signal_count + 1)
    with pytest.raises(ValueError, match="differ from the approved contract"):
        ApprovedHierarchicalDiscoveryAgent(
            catalog=catalog,
            chunk_plan=plan,
            family_explanations=explanations,
            direct_numerical_manifest=manifest,
            direct_numerical_bindings=(changed, *bindings[1:]),
            runner=_MetadataRunner(),
        )

    first = manifest.family_coverage[0]
    wrong_ids = ("evidence_wrong",)
    wrong_coverage = replace(
        first,
        semantic_atom_ids=wrong_ids,
        semantic_atom_ids_sha256=content_sha256(list(wrong_ids)),
    )
    wrong_manifest = replace(
        manifest,
        family_coverage=(wrong_coverage, *manifest.family_coverage[1:]),
    )
    with pytest.raises(ValueError, match="evidence IDs differ"):
        ApprovedHierarchicalDiscoveryAgent(
            catalog=catalog,
            chunk_plan=plan,
            family_explanations=explanations,
            direct_numerical_manifest=wrong_manifest,
            direct_numerical_bindings=direct_numerical_bindings_from_manifest(wrong_manifest),
            runner=_MetadataRunner(),
        )


def test_manifest_mutation_after_preparation_fails_before_job_call():
    agent, runner = _agent()
    object.__setattr__(
        agent.direct_numerical_manifest,
        "semantic_catalog_sha256",
        "f" * 64,
    )

    with pytest.raises(ValueError, match="different semantic catalog|mutated"):
        agent.execute(approved_wrapper_sha256=agent.precommit.approval_sha256)

    assert runner.calls == []
