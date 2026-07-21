#!/usr/bin/env python3
"""Run one current-contract hierarchy canary against one explicit endpoint.

This diagnostic is deliberately narrower than the production one-shot.  It
authenticates a completed production Stage-1 bundle, builds the exact production
runtime, performs the ordinary transport-free all-fold hierarchy preparation,
and selects the smallest real architecture-pure initial interpretation job by
rendered message bytes.  It then executes exactly that one logical job through
the production per-call metadata authenticator and hierarchy ``_run`` boundary,
which provides local JSON-Schema validation, deterministic normalization, the
single bounded schema repair, and validated-only immutable caching.

The endpoint and served-model name are invocation inputs, not hard-coded trust
anchors.  The canary binds one exact endpoint and model with no routing pool or
fallback, then requires every response (including a schema-invalid response and
its sole repair) to report that exact model and ``finish_reason=stop`` before
semantic validation or a cache write.

The canary never calls the fusion runner's prediction path and never constructs
an oracle evaluation.  Its published report contains hashes, counts, settings,
and transport metadata only; raw model content and normalized clinical findings
are not emitted.  Optional deployment/version records are informational only and
are never execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from oci.inference.all_evidence_discovery_interfaces import (
    DiscoveryEvidenceItem,
    canonical_json,
    content_sha256,
    validate_interpret_evidence_chunk_response,
)
from oci.inference.approved_hierarchical_discovery_agent import (
    _PerCallMetadataAuthenticatingRunner,
)
from oci.inference.hierarchical_all_architecture_discovery import (
    AUTHENTICATED_RESPONSE_CONTRACT_BINDING,
    INTERPRET_CHUNK_JOB,
    MAX_DISCOVERY_RESPONSE_REPAIR_ATTEMPTS,
    SELECTOR_THINKING_TOKEN_BUDGET,
    DiscoveryJsonJob,
    discovery_response_repair_policy_identity,
    local_json_schema_validator_identity,
)
from oci.inference.openai_compatible_json_discovery_job_runner import (
    parse_strict_json_object,
)
from oci.inference.production_stage1_hierarchy_handoff import (
    load_production_stage1_hierarchy_handoff,
)
from oci.inference.production_stage1_hierarchy_one_shot import (
    ProductionStage1HierarchyOneShotOptions,
    _validate_fresh_roots,
    _validate_options,
    build_production_stage1_hierarchy_runner,
    validate_single_openai_compatible_endpoint,
)

CANARY_REPORT_SCHEMA = "production_stage1_hierarchy_runtime_canary_report_v1"
CANARY_FAILURE_SCHEMA = "production_stage1_hierarchy_runtime_canary_failure_v1"
EXACT_MAX_TOKENS = 25_000
EXACT_TRANSPORT_RETRIES = 0
EXACT_SCHEMA_REPAIR_ATTEMPTS = 1

_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "content",
        "reasoning_content",
        "messages",
        "response",
        "wire_response",
        "validated_response",
        "raw_response",
    }
)


def _clone(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_identity_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{label} must be one non-empty content-addressed identity")
    identity = _clone(value)
    declared = identity.pop("identity_sha256", None)
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or any(character not in "0123456789abcdef" for character in declared)
        or declared != content_sha256(identity)
    ):
        raise ValueError(f"{label} identity_sha256 does not authenticate its identity")
    return declared


def _assert_hash_metadata_only(value: Any, *, path: str = "result") -> None:
    """Reject raw prompt/response-bearing fields from the public result."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key in _FORBIDDEN_OUTPUT_KEYS:
                raise RuntimeError(f"canary output contains forbidden raw field at {path}.{key}")
            _assert_hash_metadata_only(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_hash_metadata_only(child, path=f"{path}[{index}]")


def _response_contract_hashes(job: DiscoveryJsonJob) -> dict[str, str]:
    binding = job.input_bindings.get(AUTHENTICATED_RESPONSE_CONTRACT_BINDING)
    if not isinstance(binding, Mapping):
        raise ValueError("selected job lacks its authenticated dynamic response contract")
    validator = binding.get("local_json_schema_validator")
    if not isinstance(validator, Mapping):
        raise ValueError("selected job lacks its local JSON-Schema validator identity")
    return {
        "response_schema_sha256": str(binding.get("response_schema_sha256") or ""),
        "identifier_ownership_sha256": str(binding.get("identifier_ownership_sha256") or ""),
        "response_contract_binding_sha256": str(binding.get("binding_sha256") or ""),
        "local_json_schema_validator_identity_sha256": content_sha256(validator),
    }


@dataclass(frozen=True)
class _SelectedInterpretationJob:
    outer_fold: int
    source_family: str
    chunk_id: str
    job: DiscoveryJsonJob
    evidence: tuple[DiscoveryEvidenceItem, ...]
    agent: Any
    orchestrator: Any
    runner_identity: Mapping[str, Any]

    @property
    def rendered_message_bytes(self) -> int:
        return len(self.job.rendered_messages_bytes)


def _select_smallest_initial_interpretation_job(
    *,
    prepared_batch: Any,
    production_hierarchy_runner: Any,
) -> _SelectedInterpretationJob:
    """Select the deterministic smallest real initial architecture job."""

    folds = tuple(getattr(prepared_batch, "folds", ()))
    if not folds:
        raise ValueError("prepared production hierarchy has no outer folds")
    candidates: list[_SelectedInterpretationJob] = []
    for fold in folds:
        outer_fold = getattr(fold, "outer_fold", None)
        if isinstance(outer_fold, bool) or not isinstance(outer_fold, int) or outer_fold < 1:
            raise ValueError("prepared canary fold has an invalid outer-fold label")
        agent = getattr(fold, "agent", None)
        if agent is None or getattr(agent, "runner", None) is not production_hierarchy_runner:
            raise ValueError("prepared canary fold does not retain the exact production runner")
        assert_unchanged = getattr(agent, "_assert_unchanged", None)
        if not callable(assert_unchanged):
            raise TypeError("prepared canary agent lacks its authentication boundary")
        runner_identity, orchestrator = assert_unchanged()
        ledger = getattr(orchestrator, "initial_job_ledger", None)
        jobs = tuple(getattr(ledger, "jobs", ()))
        if not jobs:
            raise ValueError("prepared canary fold has no initial interpretation jobs")
        catalog = getattr(fold, "catalog", None)
        atoms = tuple(getattr(catalog, "atoms", ()))
        evidence_by_id = {
            atom.evidence_id: atom.as_discovery_item()
            for atom in atoms
            if callable(getattr(atom, "as_discovery_item", None))
        }
        chunk_plan = getattr(fold, "chunk_plan", None)
        chunks = tuple(getattr(chunk_plan, "chunks", ()))
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        for job in jobs:
            if not isinstance(job, DiscoveryJsonJob) or job.job_kind != INTERPRET_CHUNK_JOB:
                raise ValueError("initial canary ledger contains a non-interpretation job")
            job.settings.validate_for(INTERPRET_CHUNK_JOB)
            chunk_id = str(job.input_bindings.get("chunk_id") or "")
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                raise ValueError("initial canary job cites an unknown architecture chunk")
            source_family = str(job.input_bindings.get("source_family") or "")
            if source_family != chunk.source_family:
                raise ValueError("initial canary job and chunk cite different architectures")
            chunk_evidence = tuple(getattr(chunk, "evidence", ()))
            evidence_ids = tuple(str(row["evidence_id"]) for row in chunk_evidence)
            if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
                raise ValueError("initial canary chunk has invalid evidence ownership")
            try:
                evidence = tuple(evidence_by_id[evidence_id] for evidence_id in evidence_ids)
            except KeyError as exc:
                raise ValueError("initial canary chunk cites evidence outside its catalog") from exc
            if {item.source_family for item in evidence} != {source_family}:
                raise ValueError("initial canary job mixes Stage-1 architectures")
            request = parse_strict_json_object(job.messages[1]["content"])
            if request.get("job") != "interpret_evidence_chunk":
                raise ValueError("initial canary job has the wrong model-facing operation")
            if request.get("evidence") != [item.as_prompt_item() for item in evidence]:
                raise ValueError("initial canary job does not preserve its exact chunk evidence")
            candidates.append(
                _SelectedInterpretationJob(
                    outer_fold=outer_fold,
                    source_family=source_family,
                    chunk_id=chunk_id,
                    job=job,
                    evidence=evidence,
                    agent=agent,
                    orchestrator=orchestrator,
                    runner_identity=_clone(runner_identity),
                )
            )
    if not candidates:
        raise ValueError("production hierarchy has no eligible interpretation canary job")
    return min(
        candidates,
        key=lambda item: (
            item.rendered_message_bytes,
            item.outer_fold,
            item.job.job_id,
        ),
    )


def _validate_exact_runner_identity(
    identity: Mapping[str, Any],
    *,
    endpoint: str,
    model_name: str,
) -> None:
    if identity.get("endpoint_urls") != [endpoint]:
        raise ValueError("canary hierarchy runner does not bind only the supplied endpoint")
    model = identity.get("model")
    if not isinstance(model, Mapping) or model != {
        "name": model_name,
        "resolution": "explicit_only_no_autodiscovery",
    }:
        raise ValueError("canary hierarchy runner does not bind the exact supplied model")
    if identity.get("max_tokens") != EXACT_MAX_TOKENS:
        raise ValueError("canary hierarchy runner max_tokens differs from 25000")
    retry = identity.get("retry")
    if not isinstance(retry, Mapping) or (
        retry.get("max_retries") != EXACT_TRANSPORT_RETRIES or retry.get("max_attempts") != 1
    ):
        raise ValueError("canary hierarchy runner must disable transport retries")
    semantics = identity.get("response_semantics")
    selector = semantics.get("selector_thinking") if isinstance(semantics, Mapping) else None
    extraction = semantics.get("extraction_thinking") if isinstance(semantics, Mapping) else None
    if not isinstance(selector, Mapping) or selector != {
        "enabled": True,
        "thinking_token_budget": SELECTOR_THINKING_TOKEN_BUDGET,
    }:
        raise ValueError("canary selector thinking contract differs from exact production")
    if not isinstance(extraction, Mapping) or extraction != {
        "enabled": False,
        "thinking_token_budget_field": "omitted",
    }:
        raise ValueError("canary extraction thinking contract differs from exact production")


def _publish_report(*, target: Path, body: Mapping[str, Any]) -> Path:
    _assert_hash_metadata_only(body)
    wrapper = {
        "schema_version": CANARY_REPORT_SCHEMA,
        "content_sha256": content_sha256(body),
        "body": _clone(body),
    }
    parent = target.parent
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
    try:
        path = temporary / "production_stage1_hierarchy_runtime_canary.json"
        serialized = json.dumps(wrapper, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise FileExistsError("canary report directory appeared before publication")
        temporary.rename(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target / "production_stage1_hierarchy_runtime_canary.json"


def run_canary(options: ProductionStage1HierarchyOneShotOptions) -> Mapping[str, Any]:
    """Run exactly one current production hierarchy interpretation job."""

    if not isinstance(options, ProductionStage1HierarchyOneShotOptions):
        raise TypeError("options must be ProductionStage1HierarchyOneShotOptions")
    implementation_path = Path(__file__).resolve(strict=True)
    implementation_sha256 = _file_sha256(implementation_path)
    _validate_options(options)
    _validate_fresh_roots(options)
    endpoint = validate_single_openai_compatible_endpoint(options.endpoint)
    model_name = str(options.model_name)
    if not model_name or model_name != model_name.strip():
        raise ValueError("canary model must be one explicit nonempty canonical name")
    if (
        options.proposal_max_tokens != EXACT_MAX_TOKENS
        or options.extraction_max_tokens != EXACT_MAX_TOKENS
        or options.request_max_retries != EXACT_TRANSPORT_RETRIES
        or options.proposal_schema_repair_attempts != EXACT_SCHEMA_REPAIR_ATTEMPTS
    ):
        raise ValueError("canary token, retry, or repair settings differ from the fixed contract")
    if MAX_DISCOVERY_RESPONSE_REPAIR_ATTEMPTS != EXACT_SCHEMA_REPAIR_ATTEMPTS:
        raise RuntimeError("hierarchy no longer has exactly one bounded response repair")

    handoff = load_production_stage1_hierarchy_handoff(
        options.bundle_manifest_path,
        review_rounds=options.review_rounds,
        interaction_inner_folds=options.interaction_inner_folds,
        tfidf_nested_calibration_folds=options.tfidf_nested_calibration_folds,
    )
    handoff_before = handoff.as_dict()
    if (
        handoff_before.get("manual_digest_approval_required") is not False
        or handoff_before.get("raw_all_architecture_prompt_allowed") is not False
        or handoff_before.get("per_architecture_interpretation_required") is not True
    ):
        raise RuntimeError("authenticated Stage-1 bundle is not the production hierarchy handoff")

    production_runner: Any | None = None
    try:
        production_runner = build_production_stage1_hierarchy_runner(
            handoff=handoff,
            options=options,
            endpoint=endpoint,
        )
        hierarchy_runner = production_runner.hierarchical_discovery_runner
        if production_runner.hierarchical_discovery_approved_batch_sha256 is not None:
            raise RuntimeError("canary runner unexpectedly accepts a caller approval digest")
        if tuple(hierarchy_runner.execution_metadata):
            raise RuntimeError("canary hierarchy runner executed before local preparation")
        runner_identity = _clone(hierarchy_runner.identity())
        _validate_exact_runner_identity(
            runner_identity,
            endpoint=endpoint,
            model_name=model_name,
        )
        config = production_runner.config
        if (
            config.fusion_enable_thinking is not True
            or config.fusion_thinking_token_budget != SELECTOR_THINKING_TOKEN_BUDGET
            or config.fusion_max_tokens != EXACT_MAX_TOKENS
            or config.extraction_enable_thinking is not False
        ):
            raise ValueError("production fusion/extraction settings differ from canary contract")

        prepared = production_runner.prepare_hierarchical_discovery_batch()
        if tuple(hierarchy_runner.execution_metadata):
            raise RuntimeError("hierarchy preparation made a remote model call")
        selected = _select_smallest_initial_interpretation_job(
            prepared_batch=prepared,
            production_hierarchy_runner=hierarchy_runner,
        )
        if selected.runner_identity != runner_identity:
            raise ValueError("selected interpretation job binds a different runner identity")
        orchestrator = selected.orchestrator
        assert_implementation = getattr(
            orchestrator,
            "_assert_implementation_bundle_unchanged",
            None,
        )
        if not callable(assert_implementation):
            raise TypeError("selected hierarchy lacks its implementation authentication boundary")
        assert_implementation(
            context="before production hierarchy runtime canary",
            refresh_local_validator=True,
        )
        job_cache = getattr(orchestrator, "job_cache", None)
        if job_cache is None:
            raise RuntimeError("production runtime canary requires the authenticated job cache")
        if tuple(job_cache.execution_metadata):
            raise RuntimeError("canary job cache was consulted before execution")
        cache_identity_sha256 = _validated_identity_sha256(
            job_cache.identity(),
            label="hierarchical discovery job cache",
        )
        job_cache.begin_execution(
            hierarchy_inner_precommit_sha256=orchestrator.precommit.precommit_sha256,
            runner_identity=selected.runner_identity,
        )
        authenticated_runner = _PerCallMetadataAuthenticatingRunner(
            runner=hierarchy_runner,
            runner_identity=selected.runner_identity,
        )
        before_records = tuple(hierarchy_runner.execution_metadata)
        normalized, result = orchestrator._run(
            runner=authenticated_runner,
            job=selected.job,
            validator=lambda raw: validate_interpret_evidence_chunk_response(
                raw,
                evidence=selected.evidence,
            ),
        )
        normalized_sha256 = content_sha256(normalized)
        normalized = None
        after_records = tuple(hierarchy_runner.execution_metadata)
        if after_records[: len(before_records)] != before_records:
            raise RuntimeError("canary runner metadata mutated across execution")
        remote_records = tuple(_clone(row) for row in after_records[len(before_records) :])
        if len(remote_records) not in {1, 2}:
            raise RuntimeError("canary exceeded one initial response plus one schema repair")
        trace = result.response_attempt_trace
        attempts = trace.get("attempts") if isinstance(trace, Mapping) else None
        if not isinstance(attempts, list) or len(attempts) not in {1, 2}:
            raise RuntimeError("canary response trace exceeded its exact logical-call bound")
        if len(attempts) != len(remote_records):
            raise RuntimeError("canary logical response trace and remote records differ")
        for record in remote_records:
            transport_attempts = record.get("attempts")
            if not isinstance(transport_attempts, list) or len(transport_attempts) != 1:
                raise RuntimeError("canary transport retries were not exactly disabled")
            transport = transport_attempts[0]
            if (
                transport.get("endpoint") != endpoint
                or transport.get("model") != model_name
                or transport.get("response_model") != model_name
                or transport.get("finish_reason") != "stop"
            ):
                raise RuntimeError(
                    "canary response metadata differs from the supplied endpoint/model/stop"
                )
        if handoff.as_dict() != handoff_before:
            raise RuntimeError("authenticated Stage-1 handoff changed during canary")
        if _file_sha256(implementation_path) != implementation_sha256:
            raise RuntimeError("canary implementation changed during execution")
        if (options.output_dir / "frozen_predictions.parquet").exists() or (
            options.output_dir / "immutable_run_manifest.json"
        ).exists():
            raise RuntimeError("canary unexpectedly constructed prediction outputs")

        contract_hashes = _response_contract_hashes(selected.job)
        body = {
            "status": "accepted",
            "canary_kind": "one_real_architecture_pure_initial_interpretation_job",
            "authorization_role": "non_authorizing_operational_runtime_check",
            "stage1_bundle": {
                "manifest_path": str(options.bundle_manifest_path),
                "bundle_sha256": handoff.inputs.bundle_sha256,
                "handoff_content_sha256": handoff_before["content_sha256"],
            },
            "endpoint": endpoint,
            "model": model_name,
            "runner_identity_sha256": runner_identity["identity_sha256"],
            "settings": {
                "max_tokens": EXACT_MAX_TOKENS,
                "transport_retries": EXACT_TRANSPORT_RETRIES,
                "selector_thinking_enabled": True,
                "selector_thinking_token_budget": SELECTOR_THINKING_TOKEN_BUDGET,
                "extraction_thinking_enabled": False,
                "maximum_schema_repairs": EXACT_SCHEMA_REPAIR_ATTEMPTS,
            },
            "selected_job": {
                "selection_order": [
                    "rendered_message_bytes",
                    "outer_fold",
                    "job_id",
                ],
                "outer_fold": selected.outer_fold,
                "source_family": selected.source_family,
                "scope": selected.job.scope,
                "chunk_id_sha256": hashlib.sha256(selected.chunk_id.encode("utf-8")).hexdigest(),
                "job_id": selected.job.job_id,
                "job_sha256": content_sha256(selected.job.as_dict()),
                "rendered_message_bytes": selected.rendered_message_bytes,
                "evidence_owner_count": len(selected.evidence),
                "evidence_owner_ids_sha256": content_sha256(
                    [item.evidence_id for item in selected.evidence]
                ),
                "semantic_member_count": sum(len(item.member_ids) for item in selected.evidence),
                **contract_hashes,
            },
            "validation": {
                "normalized_response_sha256": normalized_sha256,
                "raw_wire_response_sha256": result.raw_wire_response_sha256,
                "response_attempt_trace_sha256": result.response_attempt_trace_sha256,
                "response_attempt_outcomes": [
                    str(attempt["validation_outcome"]) for attempt in attempts
                ],
                "local_json_schema_validator_identity_sha256": content_sha256(
                    local_json_schema_validator_identity(refresh=True)
                ),
                "response_repair_policy_sha256": discovery_response_repair_policy_identity()[
                    "policy_sha256"
                ],
                "job_cache_identity_sha256": cache_identity_sha256,
                "validated_only_cache_enabled": True,
            },
            "remote_response_count": len(remote_records),
            "transport_metadata": list(remote_records),
            "raw_prompt_emitted": False,
            "raw_response_emitted": False,
            "normalized_findings_emitted": False,
            "prediction_path_constructed": False,
            "oracle_path_constructed": False,
            "full_fusion_runner_executed": False,
            "canary_implementation_file_sha256": implementation_sha256,
        }
        _assert_hash_metadata_only(body)
        report_path = _publish_report(target=options.attestation_dir, body=body)
        result_summary = {
            "schema_version": CANARY_REPORT_SCHEMA,
            "status": "accepted",
            "report_path": str(report_path),
            "report_content_sha256": content_sha256(body),
            "selected_job_id": selected.job.job_id,
            "remote_response_count": len(remote_records),
            "raw_response_emitted": False,
            "prediction_path_constructed": False,
            "oracle_path_constructed": False,
        }
        _assert_hash_metadata_only(result_summary)
        return result_summary
    finally:
        if production_runner is not None:
            close = getattr(production_runner.hierarchical_discovery_runner, "close", None)
            if callable(close):
                close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", required=True, type=Path)
    parser.add_argument("--scratch-output-dir", required=True, type=Path)
    parser.add_argument("--hierarchical-preparation-dir", required=True, type=Path)
    parser.add_argument(
        "--report-dir",
        "--attestation-dir",
        dest="report_dir",
        required=True,
        type=Path,
        help="fresh output directory for the non-authorizing runtime canary report",
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--review-rounds", required=True, type=int)
    parser.add_argument("--interaction-inner-folds", type=int, default=3)
    parser.add_argument("--tfidf-nested-calibration-folds", type=int, default=3)
    parser.add_argument("--review-stage1-device", default="cuda:0")
    parser.add_argument("--review-neural-query-device", action="append", default=[])
    parser.add_argument("--review-neural-query-nuisance-folds", type=int, default=3)
    parser.add_argument("--review-stage1-bow-fold-parallelism", type=int, default=1)
    parser.add_argument(
        "--review-stage1-bow-parallel-backend",
        choices=("threads", "processes"),
        default="threads",
    )
    parser.add_argument("--request-timeout", type=float, default=1_800.0)
    return parser


def options_from_args(args: argparse.Namespace) -> ProductionStage1HierarchyOneShotOptions:
    endpoint = validate_single_openai_compatible_endpoint(str(args.endpoint))
    model = str(args.model)
    if not model or model != model.strip():
        raise ValueError("--model must be one explicit nonempty canonical name")
    options = ProductionStage1HierarchyOneShotOptions(
        bundle_manifest_path=args.bundle_manifest,
        output_dir=args.scratch_output_dir,
        preparation_dir=args.hierarchical_preparation_dir,
        attestation_dir=args.report_dir,
        endpoint=endpoint,
        model_name=model,
        review_rounds=int(args.review_rounds),
        interaction_inner_folds=int(args.interaction_inner_folds),
        tfidf_nested_calibration_folds=int(args.tfidf_nested_calibration_folds),
        review_stage1_device=str(args.review_stage1_device),
        review_neural_query_devices=tuple(args.review_neural_query_device or ("cuda:0",)),
        review_neural_query_nuisance_folds=int(args.review_neural_query_nuisance_folds),
        review_stage1_bow_fold_parallelism=int(args.review_stage1_bow_fold_parallelism),
        review_stage1_bow_parallel_backend=str(args.review_stage1_bow_parallel_backend),
        proposal_max_tokens=EXACT_MAX_TOKENS,
        extraction_max_tokens=EXACT_MAX_TOKENS,
        proposal_schema_repair_attempts=EXACT_SCHEMA_REPAIR_ATTEMPTS,
        request_max_retries=EXACT_TRANSPORT_RETRIES,
        request_timeout=float(args.request_timeout),
    )
    _validate_options(options)
    return options


def main(argv: Sequence[str] | None = None) -> int:
    try:
        options = options_from_args(build_parser().parse_args(argv))
        result = run_canary(options)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": CANARY_FAILURE_SCHEMA,
                    "status": "rejected",
                    "failure_type": exc.__class__.__name__,
                    "raw_response_emitted": False,
                    "prediction_path_constructed": False,
                    "oracle_path_constructed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
