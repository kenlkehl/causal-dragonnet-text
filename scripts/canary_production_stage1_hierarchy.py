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
from dataclasses import asdict, dataclass
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
    DiscoveryJsonJob,
    discovery_response_repair_policy_identity,
    local_json_schema_validator_identity,
)
from oci.inference.hierarchical_discovery_job_cache import (
    HierarchicalDiscoveryJobCacheConfig,
)
from oci.inference.first_untouched_gate_direct_numerical_preparation import (
    FirstUntouchedGatePreparationBounds,
)
from oci.inference.openai_compatible_json_discovery_job_runner import (
    Stage2GenerationPolicy,
    parse_strict_json_object,
)
from oci.inference.post_extraction_scientific_policy import (
    PostExtractionScientificPolicy,
)
from oci.inference.production_stage1_hierarchy_handoff import (
    load_production_stage1_hierarchy_handoff,
)
from oci.inference.production_stage1_hierarchy_one_shot import (
    ProductionStage1HierarchyOneShotOptions,
    _forest_max_features_argument,
    _nullable_positive_int_argument,
    _validate_fresh_roots,
    _validate_options,
    add_post_extraction_causal_review_arguments,
    add_stage2_hierarchy_prompt_protocol_arguments,
    build_production_stage1_hierarchy_runner,
    build_reference_only_role_neutral_stage2_runner,
    post_extraction_causal_review_from_namespace,
    stage2_hierarchy_prompt_protocol_from_namespace,
    validate_single_openai_compatible_endpoint,
)
from oci.inference.production_role_neutral_stage2_handoff import (
    ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
    load_reference_only_role_neutral_stage1_handoff,
)

CANARY_REPORT_SCHEMA = "production_stage1_hierarchy_runtime_canary_report_v2"
ROLE_NEUTRAL_STAGE2_CANARY_REPORT_SCHEMA = (
    "production_role_neutral_stage2_runtime_canary_report_v1"
)
ROLE_NEUTRAL_STAGE2_CANARY_REPORT_FILENAME = (
    "production_role_neutral_stage2_runtime_canary.json"
)
CANARY_FAILURE_SCHEMA = "production_stage1_hierarchy_runtime_canary_failure_v1"
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
    generation_policy: Stage2GenerationPolicy,
    model_context_window_tokens: int,
) -> None:
    if identity.get("endpoint_urls") != [endpoint]:
        raise ValueError("canary hierarchy runner does not bind only the supplied endpoint")
    model = identity.get("model")
    if not isinstance(model, Mapping) or model != {
        "name": model_name,
        "resolution": "explicit_only_no_autodiscovery",
    }:
        raise ValueError("canary hierarchy runner does not bind the exact supplied model")
    if not isinstance(generation_policy, Stage2GenerationPolicy):
        raise TypeError("canary generation policy must be Stage2GenerationPolicy")
    if (
        identity.get("generation_policy") != generation_policy.as_dict()
        or identity.get("generation_policy_sha256")
        != generation_policy.content_sha256
        or identity.get("generation_policy_resolution")
        != "explicit_closed_policy"
    ):
        raise ValueError(
            "canary hierarchy runner generation policy differs from the "
            "configured protocol"
        )
    retry = identity.get("retry")
    if not isinstance(retry, Mapping) or (
        retry.get("max_retries") != EXACT_TRANSPORT_RETRIES or retry.get("max_attempts") != 1
    ):
        raise ValueError("canary hierarchy runner must disable transport retries")
    prompt_guard = identity.get("prompt_nontruncation_guard")
    if (
        not isinstance(prompt_guard, Mapping)
        or prompt_guard.get("model_name") != model_name
        or prompt_guard.get("model_context_window_tokens")
        != model_context_window_tokens
    ):
        raise ValueError(
            "canary hierarchy runner does not bind the configured exact "
            "tokenizer/context nontruncation guard"
        )
    accounting = prompt_guard.get("accounting")
    if not isinstance(accounting, Mapping) or accounting != {
        "apply_chat_template": True,
        "tokenize": True,
        "add_generation_prompt": True,
        "truncation": False,
        "endpoint_prompt_usage_exact_match_required": True,
        "request_truncation_controls_allowed": False,
    }:
        raise ValueError("canary hierarchy runner prompt nontruncation contract drifted")


def _publish_report(
    *,
    target: Path,
    body: Mapping[str, Any],
    schema_version: str = CANARY_REPORT_SCHEMA,
    filename: str = "production_stage1_hierarchy_runtime_canary.json",
) -> Path:
    _assert_hash_metadata_only(body)
    wrapper = {
        "schema_version": schema_version,
        "content_sha256": content_sha256(body),
        "body": _clone(body),
    }
    parent = target.parent
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
    try:
        path = temporary / filename
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
    return target / filename


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
        options.request_max_retries != EXACT_TRANSPORT_RETRIES
        or options.proposal_schema_repair_attempts != EXACT_SCHEMA_REPAIR_ATTEMPTS
    ):
        raise ValueError("canary retry or repair settings differ from the fixed contract")
    if MAX_DISCOVERY_RESPONSE_REPAIR_ATTEMPTS != EXACT_SCHEMA_REPAIR_ATTEMPTS:
        raise RuntimeError("hierarchy no longer has exactly one bounded response repair")

    try:
        manifest_discriminator = json.loads(
            options.bundle_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "canary bundle manifest is not readable closed JSON"
        ) from exc
    reference_only_mode = bool(
        isinstance(manifest_discriminator, Mapping)
        and manifest_discriminator.get("handoff_kind")
        == ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
    )
    if reference_only_mode:
        handoff = load_reference_only_role_neutral_stage1_handoff(
            options.bundle_manifest_path
        )
    else:
        handoff = load_production_stage1_hierarchy_handoff(
            options.bundle_manifest_path,
            review_rounds=options.review_rounds,
            initial_training_partitions=options.initial_training_partitions,
            interaction_inner_folds=options.interaction_inner_folds,
            tfidf_nested_calibration_folds=options.tfidf_nested_calibration_folds,
        )
    handoff_before = handoff.as_dict()
    if reference_only_mode:
        if (
            getattr(handoff, "handoff_kind", None)
            != ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
            or getattr(handoff, "stage2_provider", None) is None
            or handoff_before.get("offline_handoff_validation_complete")
            is not True
            or handoff_before.get("independent_runtime_stage1_refit_allowed")
            is not False
            or handoff_before.get("legacy_bundle_build_invoked") is not False
        ):
            raise RuntimeError(
                "authenticated reference-only Stage 1 handoff is invalid"
            )
    elif (
        handoff_before.get("manual_digest_approval_required") is not False
        or handoff_before.get("raw_all_architecture_prompt_allowed") is not False
        or handoff_before.get("per_architecture_interpretation_required") is not True
    ):
        raise RuntimeError(
            "authenticated Stage-1 bundle is not the production hierarchy handoff"
        )

    production_runner: Any | None = None
    try:
        production_runner = (
            build_reference_only_role_neutral_stage2_runner(
                handoff=handoff,
                options=options,
                endpoint=endpoint,
            )
            if reference_only_mode
            else build_production_stage1_hierarchy_runner(
                handoff=handoff,
                options=options,
                endpoint=endpoint,
            )
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
            generation_policy=options.stage2_protocol.generation_policy,
            model_context_window_tokens=(
                options.model_context_window_tokens
            ),
        )
        config = production_runner.config
        proposal_generation = (
            options.stage2_protocol.generation_policy.feature_proposal_review
        )
        patient_generation = (
            options.stage2_protocol.generation_policy.patient_feature_extraction
        )
        if (
            config.fusion_enable_thinking
            != proposal_generation.thinking_enabled
            or config.fusion_thinking_token_budget
            != proposal_generation.thinking_token_budget
            or config.fusion_max_tokens != proposal_generation.max_tokens
            or config.extraction_enable_thinking
            != patient_generation.thinking_enabled
            or config.post_extraction_review_config
            != options.post_extraction_review_config
            or config.post_extraction_scientific_policy
            != options.post_extraction_scientific_policy
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
        stage1_reference = (
            {
                "manifest_path": str(options.bundle_manifest_path),
                "handoff_kind": getattr(handoff, "handoff_kind", None),
                "bundle_sha256": getattr(handoff, "bundle_sha256", None),
                "handoff_content_sha256": getattr(
                    handoff,
                    "handoff_scientific_content_sha256",
                    None,
                ),
                "source_execution_content_sha256": getattr(
                    handoff,
                    "source_role_neutral_execution_content_sha256",
                    None,
                ),
                "provider_identity_sha256": handoff.stage2_provider.identity()[
                    "identity_sha256"
                ],
                "reference_only_all_ten": True,
                "legacy_stage1_loader_invoked": False,
                "independent_stage1_refit_performed": False,
            }
            if reference_only_mode
            else {
                "manifest_path": str(options.bundle_manifest_path),
                "bundle_sha256": handoff.inputs.bundle_sha256,
                "handoff_content_sha256": handoff_before["content_sha256"],
            }
        )
        body = {
            "status": "accepted",
            "canary_kind": "one_real_architecture_pure_initial_interpretation_job",
            "authorization_role": "non_authorizing_operational_runtime_check",
            "stage1_bundle": stage1_reference,
            "endpoint": endpoint,
            "model": model_name,
            "runner_identity_sha256": runner_identity["identity_sha256"],
            "settings": {
                "proposal_max_tokens": options.proposal_max_tokens,
                "extraction_max_tokens": options.extraction_max_tokens,
                "stage2_hierarchy_prompt_protocol": (
                    options.stage2_protocol.as_dict()
                ),
                "stage2_hierarchy_prompt_protocol_sha256": (
                    options.stage2_protocol.content_sha256
                ),
                "stage2_generation_policy": (
                    options.stage2_protocol.generation_policy.as_dict()
                ),
                "stage2_generation_policy_sha256": (
                    options.stage2_protocol.generation_policy.content_sha256
                ),
                "post_extraction_causal_review": asdict(
                    options.post_extraction_review_config
                ),
                "post_extraction_causal_review_sha256": content_sha256(
                    asdict(options.post_extraction_review_config)
                ),
                "post_extraction_scientific_policy": (
                    options.post_extraction_scientific_policy.as_dict()
                ),
                "post_extraction_scientific_policy_sha256": content_sha256(
                    options.post_extraction_scientific_policy.as_dict()
                ),
                "prompt_nontruncation_guard_identity_sha256": (
                    runner_identity["prompt_nontruncation_guard"][
                        "identity_sha256"
                    ]
                ),
                "transport_retries": (
                    options.stage2_protocol.generation_policy
                    .interpret_architecture_chunk.transport_max_retries
                ),
                "selector_thinking_enabled": (
                    options.stage2_protocol.generation_policy
                    .interpret_architecture_chunk.thinking_enabled
                ),
                "selector_thinking_token_budget": (
                    options.stage2_protocol.generation_policy
                    .interpret_architecture_chunk.thinking_token_budget
                ),
                "max_rendered_discovery_prompt_bytes": (
                    options.max_rendered_discovery_prompt_bytes
                ),
                "final_upstream_max_orphan_features": (
                    options.final_upstream_max_orphan_features
                ),
                "review_neural_query_nuisance_folds": (
                    options.review_neural_query_nuisance_folds
                ),
                "final_upstream_meta_inner_folds": (
                    options.final_upstream_meta_inner_folds
                ),
                "final_upstream_head_regularization": (
                    options.final_upstream_head_regularization
                ),
                "extraction_thinking_enabled": (
                    options.stage2_protocol.generation_policy
                    .patient_feature_extraction.thinking_enabled
                ),
                "maximum_schema_repairs": (
                    options.stage2_protocol.generation_policy
                    .interpret_architecture_chunk.schema_repair_attempts
                ),
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
            "reference_only_role_neutral_stage1": reference_only_mode,
            "legacy_stage1_loader_invoked": False
            if reference_only_mode
            else None,
            "independent_stage1_refit_performed": False
            if reference_only_mode
            else None,
            "canary_implementation_file_sha256": implementation_sha256,
        }
        _assert_hash_metadata_only(body)
        report_schema = (
            ROLE_NEUTRAL_STAGE2_CANARY_REPORT_SCHEMA
            if reference_only_mode
            else CANARY_REPORT_SCHEMA
        )
        report_path = _publish_report(
            target=options.attestation_dir,
            body=body,
            schema_version=report_schema,
            filename=(
                ROLE_NEUTRAL_STAGE2_CANARY_REPORT_FILENAME
                if reference_only_mode
                else "production_stage1_hierarchy_runtime_canary.json"
            ),
        )
        result_summary = {
            "schema_version": report_schema,
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
    parser.add_argument("--stage2-tokenizer-locator", required=True, type=Path)
    parser.add_argument("--review-rounds", required=True, type=int)
    parser.add_argument("--initial-training-partitions", required=True, type=int)
    parser.add_argument(
        "--hierarchical-job-cache-max-entry-bytes",
        required=True,
        type=int,
    )
    for field_name in (
        "max_initial_spent_rows",
        "max_first_gate_rows",
        "max_total_text_utf8_bytes",
        "max_catalog_atoms",
        "max_source_manifest_bytes",
        "max_direct_numerical_signals",
        "max_single_matrix_file_bytes",
        "max_total_matrix_file_bytes",
    ):
        parser.add_argument(
            "--first-untouched-gate-" + field_name.replace("_", "-"),
            dest="first_untouched_gate_" + field_name,
            required=True,
            type=int,
        )
    parser.add_argument(
        "--source-text-temporally-valid-by-design",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument("--max-candidate-variables", required=True, type=int)
    parser.add_argument("--complete-page-core-chars", required=True, type=int)
    parser.add_argument("--complete-page-context-chars", required=True, type=int)
    parser.add_argument("--complete-page-max-chars", required=True, type=int)
    parser.add_argument(
        "--complete-reconciliation-fan-in",
        required=True,
        type=int,
    )
    parser.add_argument("--forest-n-estimators", required=True, type=int)
    parser.add_argument(
        "--forest-max-depth",
        required=True,
        type=_nullable_positive_int_argument,
    )
    parser.add_argument("--forest-min-samples-leaf", required=True, type=int)
    parser.add_argument(
        "--forest-max-features",
        required=True,
        type=_forest_max_features_argument,
    )
    parser.add_argument(
        "--forest-honest",
        required=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--forest-inference",
        required=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--forest-subforest-size", required=True, type=int)
    parser.add_argument(
        "--forest-tune-model",
        required=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--forest-nuisance-n-estimators",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--forest-nuisance-max-depth",
        required=True,
        type=_nullable_positive_int_argument,
    )
    parser.add_argument(
        "--forest-nuisance-min-samples-leaf",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--forest-nuisance-treatment-max-features",
        required=True,
        type=_forest_max_features_argument,
    )
    parser.add_argument(
        "--forest-nuisance-outcome-max-features",
        required=True,
        type=_forest_max_features_argument,
    )
    parser.add_argument("--forest-random-seed", required=True, type=int)
    parser.add_argument("--forest-n-jobs", required=True, type=int)
    parser.add_argument("--interaction-inner-folds", type=int, default=3)
    parser.add_argument("--tfidf-nested-calibration-folds", type=int, default=3)
    parser.add_argument("--review-stage1-device", required=True)
    parser.add_argument(
        "--review-neural-query-device",
        action="append",
        required=True,
    )
    parser.add_argument("--review-stage1-bow-fold-parallelism", type=int, default=1)
    parser.add_argument(
        "--review-stage1-bow-parallel-backend",
        choices=("threads", "processes"),
        default="threads",
    )
    parser.add_argument("--request-timeout", type=float, default=1_800.0)
    add_stage2_hierarchy_prompt_protocol_arguments(parser)
    add_post_extraction_causal_review_arguments(parser)
    parser.add_argument(
        "--post-extraction-scientific-policy",
        required=True,
        type=Path,
    )
    return parser


def options_from_args(args: argparse.Namespace) -> ProductionStage1HierarchyOneShotOptions:
    endpoint = validate_single_openai_compatible_endpoint(str(args.endpoint))
    model = str(args.model)
    if not model or model != model.strip():
        raise ValueError("--model must be one explicit nonempty canonical name")
    scientific_policy = PostExtractionScientificPolicy.from_mapping(
        json.loads(
            Path(args.post_extraction_scientific_policy).read_text(
                encoding="utf-8"
            )
        )
    )
    options = ProductionStage1HierarchyOneShotOptions(
        bundle_manifest_path=args.bundle_manifest,
        output_dir=args.scratch_output_dir,
        preparation_dir=args.hierarchical_preparation_dir,
        attestation_dir=args.report_dir,
        endpoint=endpoint,
        model_name=model,
        review_rounds=int(args.review_rounds),
        initial_training_partitions=int(args.initial_training_partitions),
        stage2_protocol=stage2_hierarchy_prompt_protocol_from_namespace(args),
        stage2_tokenizer_locator=args.stage2_tokenizer_locator,
        hierarchical_discovery_job_cache_config=(
            HierarchicalDiscoveryJobCacheConfig(
                max_entry_bytes=int(
                    args.hierarchical_job_cache_max_entry_bytes
                )
            )
        ),
        first_untouched_gate_preparation_bounds=(
            FirstUntouchedGatePreparationBounds(
                max_initial_spent_rows=int(
                    args.first_untouched_gate_max_initial_spent_rows
                ),
                max_first_gate_rows=int(
                    args.first_untouched_gate_max_first_gate_rows
                ),
                max_total_text_utf8_bytes=int(
                    args.first_untouched_gate_max_total_text_utf8_bytes
                ),
                max_catalog_atoms=int(
                    args.first_untouched_gate_max_catalog_atoms
                ),
                max_source_manifest_bytes=int(
                    args.first_untouched_gate_max_source_manifest_bytes
                ),
                max_direct_numerical_signals=int(
                    args.first_untouched_gate_max_direct_numerical_signals
                ),
                max_single_matrix_file_bytes=int(
                    args.first_untouched_gate_max_single_matrix_file_bytes
                ),
                max_total_matrix_file_bytes=int(
                    args.first_untouched_gate_max_total_matrix_file_bytes
                ),
            )
        ),
        post_extraction_review_config=(
            post_extraction_causal_review_from_namespace(
                args,
                scientific_policy=scientific_policy,
            )
        ),
        post_extraction_scientific_policy=scientific_policy,
        source_text_temporally_valid_by_design=(
            args.source_text_temporally_valid_by_design
        ),
        interaction_inner_folds=int(args.interaction_inner_folds),
        tfidf_nested_calibration_folds=int(args.tfidf_nested_calibration_folds),
        review_stage1_device=str(args.review_stage1_device),
        review_neural_query_devices=tuple(args.review_neural_query_device),
        review_stage1_bow_fold_parallelism=int(args.review_stage1_bow_fold_parallelism),
        review_stage1_bow_parallel_backend=str(args.review_stage1_bow_parallel_backend),
        max_candidates=int(args.max_candidate_variables),
        forest_n_estimators=int(args.forest_n_estimators),
        forest_max_depth=args.forest_max_depth,
        forest_min_samples_leaf=int(args.forest_min_samples_leaf),
        forest_max_features=args.forest_max_features,
        forest_honest=bool(args.forest_honest),
        forest_inference=bool(args.forest_inference),
        forest_subforest_size=int(args.forest_subforest_size),
        forest_tune_model=bool(args.forest_tune_model),
        forest_nuisance_n_estimators=int(
            args.forest_nuisance_n_estimators
        ),
        forest_nuisance_max_depth=args.forest_nuisance_max_depth,
        forest_nuisance_min_samples_leaf=int(
            args.forest_nuisance_min_samples_leaf
        ),
        forest_nuisance_treatment_max_features=(
            args.forest_nuisance_treatment_max_features
        ),
        forest_nuisance_outcome_max_features=(
            args.forest_nuisance_outcome_max_features
        ),
        forest_random_seed=int(args.forest_random_seed),
        forest_n_jobs=int(args.forest_n_jobs),
        proposal_schema_repair_attempts=EXACT_SCHEMA_REPAIR_ATTEMPTS,
        request_max_retries=EXACT_TRANSPORT_RETRIES,
        request_timeout=float(args.request_timeout),
        extraction_max_text_length=int(args.complete_page_max_chars),
        complete_page_core_chars=int(args.complete_page_core_chars),
        complete_page_context_chars=int(args.complete_page_context_chars),
        complete_page_max_chars=int(args.complete_page_max_chars),
        complete_reconciliation_fan_in=int(
            args.complete_reconciliation_fan_in
        ),
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
