#!/usr/bin/env python3
"""One-shot, nonpersisting diagnostic for the retired v5 first hierarchy job.

The retired packet is used only to authenticate which evidence payload failed.
The model request is compiled afresh from the current hierarchy implementation,
dynamic keyed response contract, and production family explanation.  The script
never creates a hierarchy cache, executes the full fusion runner, or writes an
artifact.  Its stdout contains hashes, counts, and transport metadata only.

Running without ``--execute`` is offline and cannot create a network client.
Execution additionally requires the exact preflight digest printed by the
offline run, so source or input drift fails before the remote boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from oci.inference.all_evidence_discovery_interfaces import (
    BOW_NUISANCE,
    DISCOVERY_WIRE_NORMALIZATION_VERSION,
    DiscoveryEvidenceItem,
    canonical_json,
    content_sha256,
    render_interpret_evidence_chunk_messages,
    validate_interpret_evidence_chunk_response,
)
from oci.inference.hierarchical_all_architecture_discovery import (
    HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING,
    INTERPRET_CHUNK_JOB,
    SELECTOR_THINKING_TOKEN_BUDGET,
    DiscoveryJobSettings,
    DiscoveryJsonJob,
    hierarchical_discovery_implementation_bundle,
)
from oci.inference.hierarchical_discovery_response_contract import (
    HIERARCHICAL_DISCOVERY_EXACT_COVERAGE_REPRESENTATION,
    HIERARCHICAL_DISCOVERY_RESPONSE_CONTRACT_VERSION,
    HierarchyWireBudget,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    NON_GROUNDING_SUMMARY_SCHEMA_VERSION,
    ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
    SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION,
    ArchitectureChunkPlan,
    ArchitectureEvidenceChunk,
    NonGroundingNumericalSummary,
    RoleNeutralEvidenceCatalog,
    Stage1EvidenceAtom,
    audit_complete_architecture_delivery,
    build_complete_architecture_chunks,
    validate_role_neutral_catalog,
)
from oci.inference.openai_compatible_json_discovery_job_runner import (
    OpenAICompatibleJsonDiscoveryJobRunner,
    parse_strict_json_object,
)
from oci.inference.stage1_architecture_explanations import (
    production_stage1_family_explanations,
)

PROBE_SCHEMA_VERSION = "hierarchical_keyed_first_job_live_diagnostic_v1"
PREFLIGHT_SCHEMA_VERSION = "hierarchical_keyed_first_job_preflight_v1"

ENDPOINT = "http://camus:8010/v1"
MODEL = "RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
MAX_TOKENS = 25_000
MAX_RETRIES = 0
REQUEST_TIMEOUT_SECONDS = 900.0
REQUIRED_INTERPRETER = "/home/klkehl/thisenv/bin/python"

# This is an explicit compatibility profile for one authenticated retired
# diagnostic target.  Production callers do not import or select it, and the
# current 2-atom/3-member production profile continues to reject this old
# 7-atom/61-member request.
RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET = HierarchyWireBudget(
    max_opaque_identifier_chars=128,
    max_generated_name_chars=1,
    max_description_chars=1,
    max_reason_chars=64,
    max_ambiguity_chars=1,
    max_free_text_chars=1,
    max_generated_list_items=4,
    max_feature_names_per_member=1,
    max_findings_per_atomic_review=1,
    max_pair_relation_peers_per_page=7,
    max_definition_fold_inputs=8,
    max_group_lookback_ids=8,
    max_adaptive_review_targets=4,
    max_interpret_atoms_per_job=7,
    max_interpret_members_per_job=61,
    max_interpret_name_chars=1,
    max_interpret_description_chars=1,
    max_interpret_ambiguity_chars=1,
    max_interpret_reason_chars=64,
    max_interpret_canonical_json_bytes=20_000,
    max_interpret_transport_bytes=20_000,
    interpret_generation_token_budget=20_000,
    max_response_transport_bytes=20_000,
    generation_token_budget=20_000,
)
EXPECTED_RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET_SHA256 = (
    "09fe08ebf047a7d98291f91b88c35453047099f5cddcd8531e303ff4f4c2e2c6"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREPARATION_ROOT = REPOSITORY_ROOT / (
    "artifacts/all_evidence_fusion/"
    "hierarchical_all_arch_one_20260720_v5_preparation/outer_fold_001"
)

EXPECTED_FILES = {
    "role_neutral_evidence_catalog.json": {
        "file_sha256": "755a8ac052896a7fefbe7ec5d5f9e4f21bc2a181a787ff2e0a73b8ae89392357",
        "content_sha256": "183f23c5c56cdfae4bb2a74d5f6907de799e131955c85ed17233c3c614531c15",
        "envelope_schema": "role_neutral_evidence_catalog_preparation_envelope_v1",
    },
    "architecture_chunk_plan.json": {
        "file_sha256": "7f5d9e32a3eb1065c52dbd4c0217b68d966fdab616cedbb93379d030ad523f98",
        "content_sha256": "485567ed29a58c3dc4cd0dc59c5a0641a2e42b9f17bea8e2b85b05b5243cda7c",
        "envelope_schema": "architecture_chunk_plan_preparation_envelope_v1",
    },
    "approved_hierarchical_wrapper_precommit.json": {
        "file_sha256": "46da744ed2271a590cea6c4da733ca979878624925e160564d61910daac73312",
        "content_sha256": "c55ad674d3c458af92f791f2343f975492e2955c531d0577e39639795c95c539",
        "envelope_schema": "hierarchical_all_evidence_runner_batch_packet_v1",
    },
}

RETIRED_CATALOG_SCHEMA_VERSION = "role_neutral_stage1_evidence_catalog_v1"
RETIRED_CHUNK_SCHEMA_VERSION = "role_neutral_architecture_chunk_v1"
RETIRED_CHUNK_PLAN_SCHEMA_VERSION = "complete_architecture_chunk_plan_v2"
EXPECTED_RETIRED_CATALOG_SHA256 = (
    "24562e2b83fc3d9defbcaf4e3edbdf5f5748907d898ce5580b58b008befd86e4"
)
EXPECTED_RETIRED_PLAN_SHA256 = (
    "52cb8176f9c7b1e58036fa4b2bb73d17394b67cb66b74d06545e3bfd0bcb4c1f"
)
EXPECTED_CURRENT_CATALOG_SHA256 = (
    "54ca35db68608846b34e18db830d1d66fbe4fd049e1ba22a1557e17971cbf70c"
)
EXPECTED_CURRENT_CATALOG_CONTENT_SHA256 = (
    "11f5736f01fea35bc1b68c694a5034829532e425adc22022a45c36af089584fe"
)
EXPECTED_CURRENT_PLAN_SHA256 = (
    "c1bf054ef2a6f12ace65a210c63ffa1c2306924b86e10612122b070588a266bb"
)
EXPECTED_CURRENT_PLAN_CONTENT_SHA256 = (
    "a39ce1f3bfb29382029c88e4e4627b2cc330a7de3796b34695bdcf1cb89d5b80"
)
EXPECTED_RETIRED_JOB_ID = "job_7096d1629cfff17c149963786b8bca5c58dbcf2bffcd57e80a173704c7ade598"
EXPECTED_RETIRED_CHUNK_ID = (
    "chunk_fe031c1020bb83a671460a0b2ccb9f32ff93152bd010fc9e63f469109f6cea0b"
)
EXPECTED_CURRENT_CHUNK_ID = (
    "chunk_55160e4437d55883f40e21e4d197685b5ac4a9182d590aec0bc37cc81a05d064"
)
EXPECTED_OWNER_MEMBER_COUNTS = (11, 9, 6, 11, 7, 9, 8)
EXPECTED_EVIDENCE_ID = "evidence_220e6715b7cc98d3780f023d6cf9b6df09f0a0e15cacc98a7bf437bfa845b13c"
EXPECTED_TARGET_MEMBER_ID = (
    "member_799c5503136021d95239c8e113963c10100d72f60f96891c8aa9e42afcfc614a"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _strict_mapping_file(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not UTF-8") from exc
    parsed = parse_strict_json_object(text)
    return parsed, _sha256_bytes(raw)


def _authenticated_envelope(name: str) -> dict[str, Any]:
    expected = EXPECTED_FILES[name]
    payload, file_sha256 = _strict_mapping_file(PREPARATION_ROOT / name)
    _exact_keys(
        payload,
        {"schema_version", "content_sha256", "body"},
        label=name,
    )
    if file_sha256 != expected["file_sha256"]:
        raise ValueError(f"{name} byte digest differs from the authenticated target")
    if payload["schema_version"] != expected["envelope_schema"]:
        raise ValueError(f"{name} envelope schema differs from the authenticated target")
    body = payload["body"]
    if not isinstance(body, Mapping):
        raise TypeError(f"{name}.body must be one object")
    body_sha256 = content_sha256(body)
    if body_sha256 != expected["content_sha256"]:
        raise ValueError(f"{name} content differs from the authenticated target")
    if payload["content_sha256"] != body_sha256:
        raise ValueError(f"{name} envelope content digest is invalid")
    return json.loads(canonical_json(body))


def _catalog_from_authenticated_body(body: Mapping[str, Any]) -> RoleNeutralEvidenceCatalog:
    _exact_keys(
        body,
        {
            "schema_version",
            "outer_fold",
            "scope",
            "inner_fold",
            "split_fingerprint",
            "catalog_sha256",
            "atoms",
            "non_grounding_numerical_summaries",
            "audit",
        },
        label="retired catalog body",
    )
    if body.get("schema_version") != RETIRED_CATALOG_SCHEMA_VERSION:
        raise ValueError("retired catalog body schema version differs")
    retired_identity = {
        key: body[key]
        for key in (
            "schema_version",
            "outer_fold",
            "scope",
            "inner_fold",
            "split_fingerprint",
            "atoms",
            "non_grounding_numerical_summaries",
        )
    }
    if (
        content_sha256(retired_identity) != EXPECTED_RETIRED_CATALOG_SHA256
        or body.get("catalog_sha256") != EXPECTED_RETIRED_CATALOG_SHA256
    ):
        raise ValueError("retired catalog identity does not authenticate")
    atoms: list[Stage1EvidenceAtom] = []
    for row in body.get("atoms", []):
        if not isinstance(row, Mapping):
            raise TypeError("catalog atom must be one object")
        _exact_keys(
            row,
            {
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
            },
            label="retired catalog atom",
        )
        if row.get("schema_version") != RETIRED_CATALOG_SCHEMA_VERSION:
            raise ValueError("retired catalog atom schema version differs")
        atoms.append(
            Stage1EvidenceAtom(
                evidence_id=str(row["evidence_id"]),
                atom_kind=str(row["atom_kind"]),
                source_kind=str(row["source_kind"]),
                source_family=str(row["source_family"]),
                observable_axes=tuple(row["observable_axes"]),
                member_ids=tuple(row["member_ids"]),
                split_fingerprint=str(row["split_fingerprint"]),
                origin_sha256=str(row["origin_sha256"]),
                content_sha256=str(row["content_sha256"]),
                _origin_json=canonical_json(row["origin"]),
                _content_json=canonical_json(row["content"]),
            )
        )
    summaries: list[NonGroundingNumericalSummary] = []
    for row in body.get("non_grounding_numerical_summaries", []):
        if not isinstance(row, Mapping):
            raise TypeError("numerical summary must be one object")
        _exact_keys(
            row,
            {
                "schema_version",
                "summary_id",
                "source_kind",
                "source_family",
                "observable_axes",
                "split_fingerprint",
                "metrics",
                "concept_grounding_allowed",
            },
            label="retired numerical summary",
        )
        if row.get("schema_version") != NON_GROUNDING_SUMMARY_SCHEMA_VERSION:
            raise ValueError("numerical summary schema version differs")
        if row.get("concept_grounding_allowed") is not False:
            raise ValueError("numerical summary unexpectedly permits concept grounding")
        summaries.append(
            NonGroundingNumericalSummary(
                summary_id=str(row["summary_id"]),
                source_kind=str(row["source_kind"]),
                source_family=str(row["source_family"]),
                observable_axes=tuple(row["observable_axes"]),
                split_fingerprint=str(row["split_fingerprint"]),
                _metrics_json=canonical_json(row["metrics"]),
            )
        )
    retired_audit = body.get("audit")
    if not isinstance(retired_audit, Mapping):
        raise TypeError("retired catalog audit must be one object")
    semantic_member_batch_size = retired_audit.get("semantic_member_batch_size")
    if (
        isinstance(semantic_member_batch_size, bool)
        or not isinstance(semantic_member_batch_size, int)
        or semantic_member_batch_size < 1
        or retired_audit.get("semantic_member_batches_truncated") is not False
    ):
        raise ValueError(
            "retired catalog does not prove complete configured semantic-member batching"
        )
    semantic_member_batching = {
        "schema_version": SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION,
        "semantic_member_batch_size": semantic_member_batch_size,
        "selection_or_truncation_authorized": False,
        "complete_member_coverage_required": True,
    }
    current_identity = {
        "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
        "semantic_member_batching": semantic_member_batching,
        "outer_fold": int(body["outer_fold"]),
        "scope": str(body["scope"]),
        "inner_fold": (None if body["inner_fold"] is None else int(body["inner_fold"])),
        "split_fingerprint": str(body["split_fingerprint"]),
        "atoms": [atom.as_dict() for atom in atoms],
        "non_grounding_numerical_summaries": [summary.as_dict() for summary in summaries],
    }
    current_catalog_sha256 = content_sha256(current_identity)
    if current_catalog_sha256 != EXPECTED_CURRENT_CATALOG_SHA256:
        raise ValueError("current catalog migration content differs")
    current_audit = json.loads(canonical_json(retired_audit))
    current_audit["schema_version"] = ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION
    current_audit["catalog_sha256"] = current_catalog_sha256
    current_audit["semantic_member_batching"] = semantic_member_batching
    catalog = RoleNeutralEvidenceCatalog(
        outer_fold=int(body["outer_fold"]),
        scope=str(body["scope"]),
        inner_fold=(None if body["inner_fold"] is None else int(body["inner_fold"])),
        split_fingerprint=str(body["split_fingerprint"]),
        atoms=tuple(atoms),
        non_grounding_numerical_summaries=tuple(summaries),
        catalog_sha256=current_catalog_sha256,
        _audit_json=canonical_json(current_audit),
    )
    validate_role_neutral_catalog(catalog)
    if content_sha256(catalog.as_dict()) != EXPECTED_CURRENT_CATALOG_CONTENT_SHA256:
        raise ValueError("migrated catalog is not its authenticated current typed projection")
    return catalog


def _plan_from_authenticated_body(
    body: Mapping[str, Any],
    *,
    catalog: RoleNeutralEvidenceCatalog,
) -> ArchitectureChunkPlan:
    _exact_keys(
        body,
        {
            "schema_version",
            "catalog_sha256",
            "max_atoms_per_chunk",
            "max_bytes_per_chunk",
            "max_semantic_member_ids_per_chunk",
            "plan_sha256",
            "chunks",
            "audit",
        },
        label="retired architecture chunk plan",
    )
    if body.get("schema_version") != RETIRED_CHUNK_PLAN_SCHEMA_VERSION:
        raise ValueError("retired architecture chunk plan schema differs")
    retired_identity = {
        key: body[key]
        for key in (
            "schema_version",
            "catalog_sha256",
            "max_atoms_per_chunk",
            "max_bytes_per_chunk",
            "max_semantic_member_ids_per_chunk",
            "chunks",
        )
    }
    if (
        content_sha256(retired_identity) != EXPECTED_RETIRED_PLAN_SHA256
        or body.get("plan_sha256") != EXPECTED_RETIRED_PLAN_SHA256
        or body.get("catalog_sha256") != EXPECTED_RETIRED_CATALOG_SHA256
    ):
        raise ValueError("retired architecture chunk plan identity does not authenticate")
    retired_chunks = body.get("chunks")
    if (
        not isinstance(retired_chunks, list)
        or not retired_chunks
        or not isinstance(retired_chunks[0], Mapping)
        or retired_chunks[0].get("chunk_id") != EXPECTED_RETIRED_CHUNK_ID
    ):
        raise ValueError("retired architecture chunk plan changed its designated first target")
    retired_projection: list[dict[str, Any]] = []
    for row in retired_chunks:
        if not isinstance(row, Mapping):
            raise TypeError("chunk must be one object")
        _exact_keys(
            row,
            {
                "schema_version",
                "source_family",
                "chunk_index",
                "chunk_count",
                "chunk_id",
                "evidence",
            },
            label="retired architecture chunk",
        )
        if row.get("schema_version") != RETIRED_CHUNK_SCHEMA_VERSION:
            raise ValueError("retired architecture chunk schema differs")
        chunk_identity = {
            "schema_version": RETIRED_CHUNK_SCHEMA_VERSION,
            "catalog_sha256": EXPECTED_RETIRED_CATALOG_SHA256,
            "source_family": row["source_family"],
            "chunk_index": row["chunk_index"],
            "chunk_count": row["chunk_count"],
            "evidence": row["evidence"],
        }
        if row.get("chunk_id") != f"chunk_{content_sha256(chunk_identity)}":
            raise ValueError("retired architecture chunk identity does not authenticate")
        retired_projection.append(
            {
                "source_family": row["source_family"],
                "chunk_index": row["chunk_index"],
                "chunk_count": row["chunk_count"],
                "evidence": row["evidence"],
            }
        )
    plan = build_complete_architecture_chunks(
        catalog,
        max_atoms_per_chunk=int(body["max_atoms_per_chunk"]),
        max_bytes_per_chunk=int(body["max_bytes_per_chunk"]),
        max_semantic_member_ids_per_chunk=int(body["max_semantic_member_ids_per_chunk"]),
    )
    current_projection = [
        {
            "source_family": row.source_family,
            "chunk_index": row.chunk_index,
            "chunk_count": row.chunk_count,
            "evidence": row.evidence,
        }
        for row in plan.chunks
    ]
    if canonical_json(current_projection) != canonical_json(retired_projection):
        raise ValueError("current chunk migration changed evidence ownership or grouping")
    if (
        plan.plan_sha256 != EXPECTED_CURRENT_PLAN_SHA256
        or content_sha256(plan.as_dict()) != EXPECTED_CURRENT_PLAN_CONTENT_SHA256
    ):
        raise ValueError("migrated chunk plan is not its authenticated current projection")
    return plan


def _authenticate_retired_target(
    *, wrapper_body: Mapping[str, Any], target_evidence: Sequence[DiscoveryEvidenceItem]
) -> dict[str, Any]:
    _exact_keys(wrapper_body, {"approval_sha256", "packet"}, label="wrapper body")
    packet = wrapper_body["packet"]
    if not isinstance(packet, Mapping):
        raise TypeError("wrapper packet must be one object")
    if wrapper_body["approval_sha256"] != content_sha256(packet):
        raise ValueError("wrapper approval does not authenticate its retired packet")

    hierarchy = packet.get("hierarchy_precommit")
    if not isinstance(hierarchy, Mapping):
        raise TypeError("retired hierarchy precommit must be one object")
    _exact_keys(hierarchy, {"packet", "precommit_sha256"}, label="hierarchy precommit")
    inner_packet = hierarchy["packet"]
    if not isinstance(inner_packet, Mapping):
        raise TypeError("retired inner packet must be one object")
    if hierarchy["precommit_sha256"] != content_sha256(inner_packet):
        raise ValueError("retired hierarchy precommit digest is invalid")
    ledger = inner_packet.get("initial_job_ledger")
    if not isinstance(ledger, Mapping):
        raise TypeError("retired initial job ledger must be one object")
    jobs = ledger.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("retired initial job ledger is empty")
    ledger_identity = {
        "schema_version": ledger.get("schema_version"),
        "jobs": jobs,
    }
    if ledger.get("ledger_sha256") != content_sha256(ledger_identity):
        raise ValueError("retired initial job ledger digest is invalid")

    historical = jobs[0]
    if not isinstance(historical, Mapping):
        raise TypeError("retired first job must be one object")
    historical_identity = {key: value for key, value in historical.items() if key != "job_id"}
    if historical.get("job_id") != f"job_{content_sha256(historical_identity)}":
        raise ValueError("retired first job ID does not authenticate its content")
    if historical.get("job_id") != EXPECTED_RETIRED_JOB_ID:
        raise ValueError("retired first job is not the known failing target")
    if historical.get("job_kind") != INTERPRET_CHUNK_JOB:
        raise ValueError("retired first job kind differs from the failing target")
    if historical.get("scope") != "bow_nuisance.chunk_001":
        raise ValueError("retired first job scope differs from the failing target")
    if historical.get("settings") != DiscoveryJobSettings.selector().as_dict():
        raise ValueError("retired first job selector settings differ")
    messages = historical.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("retired first job does not contain one initial message pair")
    user = messages[1]
    if not isinstance(user, Mapping) or user.get("role") != "user":
        raise ValueError("retired first job user message is invalid")
    historical_request = parse_strict_json_object(str(user.get("content")))
    expected_prompt_evidence = [item.as_prompt_item() for item in target_evidence]
    if historical_request.get("job") != "interpret_evidence_chunk":
        raise ValueError("retired request job differs from the failing target")
    if historical_request.get("evidence") != expected_prompt_evidence:
        raise ValueError("retired request evidence differs from the authenticated chunk")
    if historical_request.get("family_explanation") != (
        production_stage1_family_explanations()[BOW_NUISANCE]
    ):
        raise ValueError("production family explanation drifted from the target request")
    return {
        "wrapper_approval_sha256": wrapper_body["approval_sha256"],
        "hierarchy_precommit_sha256": hierarchy["precommit_sha256"],
        "retired_initial_job_ledger_sha256": ledger["ledger_sha256"],
        "retired_job_id": historical["job_id"],
        "retired_designated_request_sha256": content_sha256(
            {
                "job": historical_request["job"],
                "family_explanation": historical_request["family_explanation"],
                "evidence": historical_request["evidence"],
            }
        ),
    }


def _target_chunk(
    *, catalog: RoleNeutralEvidenceCatalog, plan: ArchitectureChunkPlan
) -> tuple[ArchitectureEvidenceChunk, tuple[DiscoveryEvidenceItem, ...]]:
    if catalog.catalog_sha256 != EXPECTED_CURRENT_CATALOG_SHA256:
        raise ValueError("catalog is not the authenticated current migration")
    if plan.plan_sha256 != EXPECTED_CURRENT_PLAN_SHA256:
        raise ValueError("chunk plan is not the authenticated current migration")
    if (catalog.outer_fold, catalog.scope, catalog.inner_fold) != (1, "inner_train", 1):
        raise ValueError("catalog fold scope is not outer-1 inner-train inner-1")
    audit = audit_complete_architecture_delivery(catalog, plan)
    if audit != plan.audit:
        raise ValueError("stored architecture delivery audit is not current and exact")
    chunk = plan.chunks[0]
    if (
        chunk.source_family,
        chunk.chunk_index,
        chunk.chunk_count,
        chunk.chunk_id,
    ) != (BOW_NUISANCE, 1, 5, EXPECTED_CURRENT_CHUNK_ID):
        raise ValueError("first chunk is not the authenticated failing target")
    if len(chunk.evidence) != 7:
        raise ValueError("target chunk must contain exactly seven evidence owners")
    owner_counts = tuple(len(row["member_ids"]) for row in chunk.evidence)
    if owner_counts != EXPECTED_OWNER_MEMBER_COUNTS or sum(owner_counts) != 61:
        raise ValueError("target chunk semantic-member ownership differs")
    if chunk.evidence[1]["evidence_id"] != EXPECTED_EVIDENCE_ID:
        raise ValueError("target evidence owner differs")
    if chunk.evidence[1]["member_ids"][2] != EXPECTED_TARGET_MEMBER_ID:
        raise ValueError("known duplicated member is not at its authenticated location")
    all_member_ids = [member for row in chunk.evidence for member in row["member_ids"]]
    if len(all_member_ids) != len(set(all_member_ids)):
        raise ValueError("authenticated target input itself contains duplicate member IDs")
    if all_member_ids.count(EXPECTED_TARGET_MEMBER_ID) != 1:
        raise ValueError("known duplicated-response target must occur once in the input")

    by_id = {atom.evidence_id: atom.as_discovery_item() for atom in catalog.atoms}
    evidence = tuple(by_id[str(row["evidence_id"])] for row in chunk.evidence)
    if [item.as_prompt_item() for item in evidence] != chunk.evidence:
        raise ValueError("typed catalog evidence differs from the authenticated chunk payload")
    return chunk, evidence


def _compile_current_job(
    *,
    catalog: RoleNeutralEvidenceCatalog,
    plan: ArchitectureChunkPlan,
    chunk: ArchitectureEvidenceChunk,
    evidence: Sequence[DiscoveryEvidenceItem],
) -> DiscoveryJsonJob:
    bundle = hierarchical_discovery_implementation_bundle()
    bundle_sha256 = bundle.get("implementation_bundle_sha256")
    if not isinstance(bundle_sha256, str):
        raise ValueError("current hierarchy implementation bundle is not content addressed")
    messages = render_interpret_evidence_chunk_messages(
        family_explanation=production_stage1_family_explanations()[BOW_NUISANCE],
        evidence=evidence,
        wire_budget=RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET,
    )
    job = DiscoveryJsonJob.create(
        job_kind=INTERPRET_CHUNK_JOB,
        scope="bow_nuisance.chunk_001",
        dependencies=(),
        settings=DiscoveryJobSettings.selector(),
        messages=messages,
        input_bindings={
            "catalog_sha256": catalog.catalog_sha256,
            "chunk_plan_sha256": plan.plan_sha256,
            "chunk_id": chunk.chunk_id,
            "source_family": chunk.source_family,
            HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING: bundle_sha256,
        },
    )
    if job.job_id == EXPECTED_RETIRED_JOB_ID:
        raise ValueError("current job unexpectedly reuses the retired response contract")
    current_request = parse_strict_json_object(job.messages[1]["content"])
    if current_request.get("evidence") != [item.as_prompt_item() for item in evidence]:
        raise ValueError("fresh current job changed the authenticated evidence payload")
    response_contract = current_request.get("identifier_ownership")
    if not isinstance(response_contract, Mapping):
        raise ValueError("fresh current job is missing its dynamic response contract")
    if response_contract.get("contract_version") != (
        HIERARCHICAL_DISCOVERY_RESPONSE_CONTRACT_VERSION
    ):
        raise ValueError("fresh job does not use the current response contract")
    if response_contract.get("exact_coverage_representation") != (
        HIERARCHICAL_DISCOVERY_EXACT_COVERAGE_REPRESENTATION
    ):
        raise ValueError("fresh job does not use keyed exact coverage")
    if response_contract.get("hierarchy_wire_budget") != (
        RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET.as_dict()
    ):
        raise ValueError("fresh diagnostic job does not bind its explicit wire profile")
    _assert_exact_keyed_target_schema(job.response_schema, evidence=evidence)
    return job


def _assert_exact_keyed_target_schema(
    schema: Mapping[str, Any], *, evidence: Sequence[DiscoveryEvidenceItem]
) -> None:
    """Independently confirm the dynamic schema closes both identifier levels."""

    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("current response schema root is not one closed object")
    if schema.get("required") != ["evidence_dispositions"]:
        raise ValueError("current response schema root requirements differ")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        raise TypeError("current response schema properties are malformed")
    _exact_keys(
        properties,
        {"evidence_dispositions"},
        label="current response schema properties",
    )
    dispositions = properties["evidence_dispositions"]
    if not isinstance(dispositions, Mapping):
        raise TypeError("evidence disposition schema is malformed")
    expected_evidence_ids = [item.evidence_id for item in evidence]
    if (
        dispositions.get("type") != "object"
        or dispositions.get("additionalProperties") is not False
        or dispositions.get("required") != expected_evidence_ids
    ):
        raise ValueError("evidence disposition schema is not exact keyed coverage")
    disposition_properties = dispositions.get("properties")
    if not isinstance(disposition_properties, Mapping):
        raise TypeError("evidence disposition keyed properties are malformed")
    if set(disposition_properties) != set(expected_evidence_ids):
        raise ValueError("evidence disposition schema keys differ from the request")
    for item in evidence:
        disposition = disposition_properties.get(item.evidence_id)
        if not isinstance(disposition, Mapping):
            raise TypeError("one evidence disposition schema is malformed")
        disposition_fields = disposition.get("properties")
        if not isinstance(disposition_fields, Mapping):
            raise TypeError("one evidence disposition field schema is malformed")
        _exact_keys(
            disposition_fields,
            {"evidence_findings", "member_dispositions", "reason"},
            label="one evidence disposition fields",
        )
        if disposition.get("required") != [
            "evidence_findings",
            "member_dispositions",
            "reason",
        ]:
            raise ValueError("one evidence disposition requirements differ")
        members = disposition_fields.get("member_dispositions")
        if not isinstance(members, Mapping):
            raise TypeError("one member disposition schema is malformed")
        if (
            members.get("type") != "object"
            or members.get("additionalProperties") is not False
            or members.get("required") != list(item.member_ids)
        ):
            raise ValueError("member disposition schema is not exact owner-keyed coverage")
        member_properties = members.get("properties")
        if not isinstance(member_properties, Mapping) or set(member_properties) != set(
            item.member_ids
        ):
            raise ValueError("member disposition schema keys differ from owner membership")
        for member_id in item.member_ids:
            member = member_properties.get(member_id)
            if (
                not isinstance(member, Mapping)
                or member.get("additionalProperties") is not False
                or member.get("required") != ["findings"]
                or set(member.get("properties") or {}) != {"findings"}
            ):
                raise ValueError("member disposition value schema is not current and closed")


def _build_preflight() -> tuple[
    dict[str, Any],
    DiscoveryJsonJob,
    tuple[DiscoveryEvidenceItem, ...],
    OpenAICompatibleJsonDiscoveryJobRunner,
]:
    if sys.executable != REQUIRED_INTERPRETER:
        raise ValueError("diagnostic requires the exact production Python interpreter")
    if sys.dont_write_bytecode is not True:
        raise ValueError("diagnostic requires PYTHONDONTWRITEBYTECODE=1")
    if RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET.content_sha256 != (
        EXPECTED_RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET_SHA256
    ):
        raise ValueError("explicit retired-target diagnostic wire profile differs")
    catalog_body = _authenticated_envelope("role_neutral_evidence_catalog.json")
    plan_body = _authenticated_envelope("architecture_chunk_plan.json")
    wrapper_body = _authenticated_envelope("approved_hierarchical_wrapper_precommit.json")
    catalog = _catalog_from_authenticated_body(catalog_body)
    plan = _plan_from_authenticated_body(plan_body, catalog=catalog)
    chunk, evidence = _target_chunk(catalog=catalog, plan=plan)
    retired = _authenticate_retired_target(
        wrapper_body=wrapper_body,
        target_evidence=evidence,
    )
    job = _compile_current_job(
        catalog=catalog,
        plan=plan,
        chunk=chunk,
        evidence=evidence,
    )
    runner = OpenAICompatibleJsonDiscoveryJobRunner(
        server_urls=(ENDPOINT,),
        model_name=MODEL,
        api_key="EMPTY",
        request_timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        max_tokens=MAX_TOKENS,
    )
    if runner.last_execution_metadata is not None or runner.execution_metadata:
        raise RuntimeError("runner executed during offline preflight")
    runner_identity = runner.identity()
    request_kwargs = runner._request_kwargs(job)  # exact production transport projection
    response_format = request_kwargs.get("response_format")
    requested_schema = (
        response_format.get("json_schema", {}).get("schema")
        if isinstance(response_format, Mapping)
        else None
    )
    if requested_schema != job.response_schema:
        raise ValueError("transport request does not carry the exact authenticated job schema")
    json_schema_wrapper = (
        response_format.get("json_schema") if isinstance(response_format, Mapping) else None
    )
    if (
        not isinstance(json_schema_wrapper, Mapping)
        or json_schema_wrapper.get("strict") is not True
    ):
        raise ValueError("transport request does not require strict JSON Schema generation")
    if request_kwargs.get("model") != MODEL or request_kwargs.get("max_tokens") != MAX_TOKENS:
        raise ValueError("transport request model or max_tokens differs")
    extra_body = request_kwargs.get("extra_body")
    if not isinstance(extra_body, Mapping):
        raise ValueError("transport request is missing selector settings")
    if extra_body.get("thinking_token_budget") != SELECTOR_THINKING_TOKEN_BUDGET:
        raise ValueError("transport request does not use the exact selector budget")
    chat_template = extra_body.get("chat_template_kwargs")
    if not isinstance(chat_template, Mapping) or chat_template.get("enable_thinking") is not True:
        raise ValueError("transport request does not enable selector thinking")
    owner_ids = tuple(item.evidence_id for item in evidence)
    owner_counts = tuple(len(item.member_ids) for item in evidence)
    preflight_body = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "probe_implementation_file_sha256": _sha256_bytes(Path(__file__).resolve().read_bytes()),
        "python_interpreter": sys.executable,
        "python_bytecode_writes_disabled": sys.dont_write_bytecode,
        "authenticated_input_envelopes": {
            name: {
                "file_sha256": values["file_sha256"],
                "content_sha256": values["content_sha256"],
                "envelope_schema": values["envelope_schema"],
            }
            for name, values in EXPECTED_FILES.items()
        },
        "current_schema_migration": {
            "retired_catalog_schema_version": RETIRED_CATALOG_SCHEMA_VERSION,
            "retired_catalog_sha256": EXPECTED_RETIRED_CATALOG_SHA256,
            "current_catalog_schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
            "current_catalog_sha256": catalog.catalog_sha256,
            "current_catalog_content_sha256": content_sha256(catalog.as_dict()),
            "retired_chunk_plan_schema_version": RETIRED_CHUNK_PLAN_SCHEMA_VERSION,
            "retired_chunk_plan_sha256": EXPECTED_RETIRED_PLAN_SHA256,
            "current_chunk_plan_schema_version": plan.as_dict()["schema_version"],
            "current_chunk_plan_sha256": plan.plan_sha256,
            "current_chunk_plan_content_sha256": content_sha256(plan.as_dict()),
            "evidence_ownership_and_grouping_preserved": True,
        },
        "retired_target_authentication": retired,
        "fresh_current_job_id": job.job_id,
        "fresh_current_job_sha256": content_sha256(job.as_dict()),
        "fresh_current_message_envelope_sha256": content_sha256(list(job.messages)),
        "fresh_current_response_schema_sha256": content_sha256(job.response_schema),
        "fresh_current_identifier_ownership_sha256": content_sha256(job.identifier_ownership),
        "fresh_current_implementation_bundle_sha256": job.input_bindings[
            HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING
        ],
        "fresh_current_request_sha256": content_sha256(request_kwargs),
        "fresh_current_runner_identity_sha256": runner_identity["identity_sha256"],
        "contract_version": HIERARCHICAL_DISCOVERY_RESPONSE_CONTRACT_VERSION,
        "exact_coverage_representation": (HIERARCHICAL_DISCOVERY_EXACT_COVERAGE_REPRESENTATION),
        "wire_normalization_version": DISCOVERY_WIRE_NORMALIZATION_VERSION,
        "diagnostic_wire_budget_sha256": (
            RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET.content_sha256
        ),
        "endpoint": ENDPOINT,
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "selector_thinking_enabled": True,
        "selector_thinking_token_budget": SELECTOR_THINKING_TOKEN_BUDGET,
        "transport_retry_count": MAX_RETRIES,
        "strict_duplicate_key_parser": True,
        "strict_json_schema_generation": True,
        "finish_reason_required": "stop",
        "outer_fold": 1,
        "scope": "inner_train",
        "inner_fold": 1,
        "source_family": BOW_NUISANCE,
        "chunk_id": chunk.chunk_id,
        "evidence_owner_count": len(owner_ids),
        "evidence_owner_ids_sha256": content_sha256(list(owner_ids)),
        "owner_member_counts": list(owner_counts),
        "semantic_member_count": sum(owner_counts),
        "semantic_member_ids_sha256": content_sha256(
            [member for item in evidence for member in item.member_ids]
        ),
        "known_target_member_input_occurrences": 1,
        "persistence_policy": {
            "hierarchy_job_cache_constructed": False,
            "full_fusion_runner_constructed": False,
            "prediction_path_constructed": False,
            "manifest_writer_constructed": False,
            "oracle_path_constructed": False,
            "raw_response_printed_or_written": False,
            "stdout_content": "hashes_counts_and_transport_metadata_only",
        },
    }
    return preflight_body, job, evidence, runner


def _wire_coverage_counts(
    response: Mapping[str, Any], *, evidence: Sequence[DiscoveryEvidenceItem]
) -> dict[str, Any]:
    expected_owner_ids = tuple(item.evidence_id for item in evidence)
    dispositions = response.get("evidence_dispositions")
    if not isinstance(dispositions, Mapping):
        raise TypeError("wire evidence_dispositions is not one keyed object")
    if set(dispositions) != set(expected_owner_ids):
        raise ValueError("wire evidence owner keys are not exact")
    observed_counts: list[int] = []
    observed_members: list[str] = []
    for item in evidence:
        disposition = dispositions.get(item.evidence_id)
        if not isinstance(disposition, Mapping):
            raise TypeError("wire evidence disposition is not one object")
        members = disposition.get("member_dispositions")
        if not isinstance(members, Mapping):
            raise TypeError("wire member_dispositions is not one keyed object")
        if set(members) != set(item.member_ids):
            raise ValueError("wire member ownership keys are not exact")
        observed_counts.append(len(members))
        observed_members.extend(members)
    return {
        "evidence_owner_count": len(dispositions),
        "owner_member_counts": observed_counts,
        "semantic_member_count": len(observed_members),
        "unique_semantic_member_count": len(set(observed_members)),
        "known_target_member_occurrences": observed_members.count(EXPECTED_TARGET_MEMBER_ID),
    }


def _execute(
    *,
    expected_probe_sha256: str,
    preflight_body: Mapping[str, Any],
    job: DiscoveryJsonJob,
    evidence: Sequence[DiscoveryEvidenceItem],
    runner: OpenAICompatibleJsonDiscoveryJobRunner,
) -> dict[str, Any]:
    probe_sha256 = content_sha256(preflight_body)
    if expected_probe_sha256 != probe_sha256:
        runner.close()
        raise ValueError("execution digest differs from the exact offline preflight")
    response: Mapping[str, Any] | None = None
    try:
        response = runner.run_json(job=job)
        if (
            _sha256_bytes(Path(__file__).resolve().read_bytes())
            != preflight_body["probe_implementation_file_sha256"]
        ):
            raise ValueError("probe implementation changed across the remote boundary")
        current_bundle_sha256 = hierarchical_discovery_implementation_bundle().get(
            "implementation_bundle_sha256"
        )
        if current_bundle_sha256 != preflight_body["fresh_current_implementation_bundle_sha256"]:
            raise ValueError("hierarchy implementation changed across the remote boundary")
        metadata = runner.last_execution_metadata
        if not isinstance(metadata, Mapping):
            raise RuntimeError("runner did not retain hash-only execution metadata")
        expected_request_sha256 = preflight_body["fresh_current_request_sha256"]
        expected_runner_sha256 = preflight_body["fresh_current_runner_identity_sha256"]
        if metadata.get("request_sha256") != expected_request_sha256:
            raise ValueError("executed request digest differs from the offline preflight")
        if metadata.get("runner_identity_sha256") != expected_runner_sha256:
            raise ValueError("executed runner identity differs from the offline preflight")
        attempts = metadata.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 1:
            raise ValueError("diagnostic must make exactly one transport attempt")
        attempt = attempts[0]
        if not isinstance(attempt, Mapping):
            raise TypeError("transport attempt metadata is malformed")
        if attempt.get("request_sha256") != expected_request_sha256:
            raise ValueError("transport attempt request digest differs from preflight")
        if attempt.get("runner_identity_sha256") != expected_runner_sha256:
            raise ValueError("transport attempt runner identity differs from preflight")
        if attempt.get("endpoint") != ENDPOINT:
            raise ValueError("transport used a non-target endpoint")
        if attempt.get("model") != MODEL or attempt.get("response_model") != MODEL:
            raise ValueError("request or response model differs from the exact target")
        if attempt.get("finish_reason") != "stop":
            raise ValueError("diagnostic accepts only finish_reason=stop")
        coverage = _wire_coverage_counts(response, evidence=evidence)
        validated = validate_interpret_evidence_chunk_response(
            response,
            evidence=evidence,
            wire_budget=RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET,
        )
        raw_wire_sha256 = content_sha256(response)
        if metadata.get("parsed_response_sha256") != raw_wire_sha256:
            raise ValueError("runner raw-wire response digest is inconsistent")
        if attempt.get("parsed_response_sha256") != raw_wire_sha256:
            raise ValueError("attempt raw-wire response digest is inconsistent")
        if coverage != {
            "evidence_owner_count": 7,
            "owner_member_counts": list(EXPECTED_OWNER_MEMBER_COUNTS),
            "semantic_member_count": 61,
            "unique_semantic_member_count": 61,
            "known_target_member_occurrences": 1,
        }:
            raise ValueError("keyed response did not achieve exact target coverage")
        return {
            "schema_version": PROBE_SCHEMA_VERSION,
            "status": "accepted",
            "preflight_sha256": probe_sha256,
            "job_id": job.job_id,
            "raw_wire_response_sha256": raw_wire_sha256,
            "normalized_validated_response_sha256": content_sha256(validated),
            "coverage": coverage,
            "semantic_validation": "passed",
            "finish_reason": "stop",
            "transport_metadata": metadata,
            "raw_response_retained": False,
        }
    except Exception as exc:
        metadata = runner.last_execution_metadata
        failure: dict[str, Any] = {
            "schema_version": PROBE_SCHEMA_VERSION,
            "status": "rejected",
            "preflight_sha256": probe_sha256,
            "job_id": job.job_id,
            "failure_type": exc.__class__.__name__,
            "transport_metadata": metadata,
            "raw_response_retained": False,
        }
        if response is not None:
            failure["raw_wire_response_sha256"] = content_sha256(response)
        return failure
    finally:
        response = None
        runner.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="make the single remote call after an exact preflight digest is supplied",
    )
    parser.add_argument(
        "--expected-probe-sha256",
        default="",
        help="exact digest printed by the immediately preceding offline preflight",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        preflight_body, job, evidence, runner = _build_preflight()
        preflight_sha256 = content_sha256(preflight_body)
        if not args.execute:
            if args.expected_probe_sha256:
                raise ValueError("--expected-probe-sha256 is valid only with --execute")
            output = {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "status": "offline_preflight_passed_no_network_client_created",
                "preflight_sha256": preflight_sha256,
                "preflight": preflight_body,
            }
            runner.close()
            print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if not args.expected_probe_sha256:
            raise ValueError("--execute requires --expected-probe-sha256")
        result = _execute(
            expected_probe_sha256=args.expected_probe_sha256,
            preflight_body=preflight_body,
            job=job,
            evidence=evidence,
            runner=runner,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "accepted" else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": PROBE_SCHEMA_VERSION,
                    "status": "preflight_rejected_before_remote_execution",
                    "failure_type": exc.__class__.__name__,
                    "raw_response_retained": False,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
