#!/usr/bin/env python3
"""One-repair, nonpersisting continuation of the keyed first-job diagnostic.

The initial live call is authenticated by a hash-only control record.  This
script reconstructs the exact privacy-preserving repair job used by production
from the current original job and the initial canonical response projection
SHA-256.  It can make at most one transport call and never includes the prior
response content in the repair messages, stdout, or a filesystem artifact.

Running without ``--execute`` is offline and does not create a network client.
Execution requires the exact digest printed by that offline preflight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from oci.inference.all_evidence_discovery_interfaces import (
    DiscoveryEvidenceItem,
    canonical_json,
    content_sha256,
    validate_interpret_evidence_chunk_response,
)
from oci.inference.hierarchical_all_architecture_discovery import (
    AUTHENTICATED_RESPONSE_REPAIR_BINDING,
    HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING,
    LOCAL_JSON_SCHEMA_VALIDATION_FAILURE,
    SEMANTIC_VALIDATION_FAILURE,
    STRICT_JSON_PARSE_FAILURE,
    VALIDATED_RESPONSE,
    DiscoveryJsonJob,
    _build_response_repair_job_from_projection_sha256,
    _response_attempt_entry,
    _response_attempt_trace,
    _validated_response_attempt_trace,
    discovery_response_repair_policy_identity,
    hierarchical_discovery_implementation_bundle,
)
from oci.inference.openai_compatible_json_discovery_job_runner import (
    InvalidDiscoveryJsonResponse,
    OpenAICompatibleJsonDiscoveryJobRunner,
    parse_strict_json_object,
)

if __package__:
    from scripts.probe_hierarchical_keyed_first_job_remote import (
        ENDPOINT,
        EXPECTED_OWNER_MEMBER_COUNTS,
        MAX_RETRIES,
        MAX_TOKENS,
        MODEL,
        REQUIRED_INTERPRETER,
        RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET,
        _build_preflight as _build_initial_preflight,
        _sha256_bytes,
        _wire_coverage_counts,
    )
else:
    from probe_hierarchical_keyed_first_job_remote import (
        ENDPOINT,
        EXPECTED_OWNER_MEMBER_COUNTS,
        MAX_RETRIES,
        MAX_TOKENS,
        MODEL,
        REQUIRED_INTERPRETER,
        RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET,
        _build_preflight as _build_initial_preflight,
        _sha256_bytes,
        _wire_coverage_counts,
    )

REPAIR_PROBE_SCHEMA_VERSION = "hierarchical_keyed_first_job_repair_diagnostic_v1"
REPAIR_PREFLIGHT_SCHEMA_VERSION = "hierarchical_keyed_first_job_repair_preflight_v1"
INITIAL_RECORD_SCHEMA_VERSION = "hierarchical_keyed_first_job_initial_live_rejection_hash_only_v2"
INITIAL_RECORD_ENVELOPE_VERSION = "hierarchical_keyed_first_job_initial_live_rejection_envelope_v2"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INITIAL_PROBE_PATH = Path(__file__).with_name("probe_hierarchical_keyed_first_job_remote.py")
INITIAL_RECORD_PATH = REPOSITORY_ROOT / (
    "artifacts/probe_controls/hierarchical_keyed_first_job_20260720/"
    "initial_live_rejection_hash_only_v2.json"
)

EXPECTED_INITIAL_RECORD_FILE_SHA256 = (
    "8a1a073f9c577f9c02d1b68081b7683b40d5235f8259f8d4bcc9d05209592b4b"
)
EXPECTED_INITIAL_RECORD_CONTENT_SHA256 = (
    "78fb4524386a58a85a508c68e9c7394946c7ece076e6c1a8abbe96fdd18ac33e"
)
EXPECTED_INITIAL_PREFLIGHT_SHA256 = (
    "610ad8622faca91deb17188da72228d9d4f0185a41f2d071b1ecc6375ad094e0"
)
EXPECTED_INITIAL_PROBE_FILE_SHA256 = (
    "6bc51064bdae435afb3deaab53cb064a646cb14b630c3eb729b0320691fc81ce"
)
EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256 = (
    "d88d49b65123e270a74919e1415e9081e08d2b21d1bd60781d2cfac03e37e77e"
)
EXPECTED_INITIAL_CONTENT_SHA256 = "262a7d6f3121d6b589d09fd3ffe6728bed7b28e45930dfcc4f8aa48f5e43f7a1"
EXPECTED_INITIAL_FAILURE_STAGE = "post_response_exact_coverage_or_semantic_validation_not_persisted"

EXPECTED_COVERAGE = {
    "evidence_owner_count": 7,
    "owner_member_counts": list(EXPECTED_OWNER_MEMBER_COUNTS),
    "semantic_member_count": 61,
    "unique_semantic_member_count": 61,
    "known_target_member_occurrences": 1,
}


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _assert_sha256_fields(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key.endswith("_sha256"):
                if (
                    not isinstance(child, str)
                    or len(child) != 64
                    or any(character not in "0123456789abcdef" for character in child)
                ):
                    raise ValueError(f"{child_path} must be one lowercase SHA-256")
            _assert_sha256_fields(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_sha256_fields(child, path=f"{path}[{index}]")


def _authenticated_initial_record() -> tuple[dict[str, Any], str]:
    raw = INITIAL_RECORD_PATH.read_bytes()
    file_sha256 = _sha256_bytes(raw)
    if file_sha256 != EXPECTED_INITIAL_RECORD_FILE_SHA256:
        raise ValueError("initial rejection control record byte digest differs")
    try:
        envelope = parse_strict_json_object(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("initial rejection control record is not UTF-8") from exc
    _exact_keys(
        envelope,
        {"schema_version", "content_sha256", "body"},
        label="initial rejection control envelope",
    )
    _assert_sha256_fields(envelope, path="initial_rejection_control")
    if envelope["schema_version"] != INITIAL_RECORD_ENVELOPE_VERSION:
        raise ValueError("initial rejection control envelope schema differs")
    body = envelope["body"]
    if not isinstance(body, Mapping):
        raise TypeError("initial rejection control body must be one object")
    body_sha256 = content_sha256(body)
    if body_sha256 != EXPECTED_INITIAL_RECORD_CONTENT_SHA256:
        raise ValueError("initial rejection control content digest differs")
    if envelope["content_sha256"] != body_sha256:
        raise ValueError("initial rejection control envelope digest is invalid")
    if body.get("schema_version") != INITIAL_RECORD_SCHEMA_VERSION:
        raise ValueError("initial rejection control body schema differs")
    correction = body.get("transcription_correction")
    if not isinstance(correction, Mapping):
        raise TypeError("initial rejection control must carry one correction record")
    if (
        correction.get("correction_kind") != "prior_local_record_omitted_final_hex_nibble"
        or correction.get("corrected_field")
        != "transport.canonical_parsed_response_projection_sha256"
        or correction.get("stdout_derived_corrected_sha256")
        != EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256
        or correction.get("omitted_final_nibble") != "e"
        or correction.get("retired_v1_record_body_sha256")
        != "83c8f78386bfa487e870d675d91256633d794f0d8a60a42dff3ae011b06654a4"
        or correction.get("retired_v1_record_file_sha256")
        != "c06d23fb218dd1064c686216151bd5664948ab73e72d921a7a7fc45ebe9827d4"
        or correction.get("retired_v1_record_used_for_network_or_repair") is not False
    ):
        raise ValueError("initial rejection transcription correction is not exact")
    if body.get("initial_probe_preflight_sha256") != EXPECTED_INITIAL_PREFLIGHT_SHA256:
        raise ValueError("initial rejection control cites a different preflight")
    if body.get("post_execution_offline_preflight_sha256") != (EXPECTED_INITIAL_PREFLIGHT_SHA256):
        raise ValueError("post-execution source preflight was not unchanged")
    if body.get("probe_implementation_file_sha256") != (EXPECTED_INITIAL_PROBE_FILE_SHA256):
        raise ValueError("initial rejection control cites a different probe")
    transport = body.get("transport")
    if not isinstance(transport, Mapping):
        raise TypeError("initial rejection transport record must be one object")
    if transport.get("canonical_parsed_response_projection_sha256") != (
        EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256
    ):
        raise ValueError("initial parsed response projection digest differs")
    if transport.get("content_sha256") != EXPECTED_INITIAL_CONTENT_SHA256:
        raise ValueError("initial transport content digest differs")
    if (
        transport.get("strict_json_parse_succeeded") is not True
        or transport.get("outcome") != "success"
        or transport.get("attempt_count") != 1
        or transport.get("maximum_transport_retries") != 0
        or transport.get("endpoint") != ENDPOINT
        or transport.get("requested_model") != MODEL
        or transport.get("response_model") != MODEL
        or transport.get("finish_reason") != "stop"
    ):
        raise ValueError("initial transport facts do not authenticate the repair premise")
    local_result = body.get("local_result")
    if not isinstance(local_result, Mapping):
        raise TypeError("initial local result must be one object")
    if (
        local_result.get("status") != "rejected"
        or local_result.get("failure_stage") != EXPECTED_INITIAL_FAILURE_STAGE
        or local_result.get("failure_category") is not None
        or local_result.get("semantic_failure_claimed") is not False
    ):
        raise ValueError("initial local result does not preserve the unresolved stage")
    privacy = body.get("privacy_and_persistence")
    if not isinstance(privacy, Mapping):
        raise TypeError("initial privacy record must be one object")
    for label in (
        "raw_response_retained",
        "raw_response_printed_or_written",
        "exception_text_retained",
        "response_values_included",
        "hierarchy_job_cache_constructed",
        "full_fusion_runner_constructed",
        "prediction_or_oracle_path_constructed",
        "benchmark_output_or_preparation_root_used",
    ):
        if privacy.get(label) is not False:
            raise ValueError(f"initial privacy premise differs for {label}")
    return json.loads(canonical_json(body)), file_sha256


def _assert_exact_repair_job(
    *,
    original_job: DiscoveryJsonJob,
    repair_job: DiscoveryJsonJob,
) -> Mapping[str, Any]:
    if repair_job.job_kind != original_job.job_kind:
        raise ValueError("repair job changed the original job kind")
    if repair_job.scope != f"{original_job.scope}.response_repair_001":
        raise ValueError("repair job scope is not the single production repair scope")
    if repair_job.dependencies:
        raise ValueError("repair continuation unexpectedly has dependencies")
    if repair_job.settings != original_job.settings:
        raise ValueError("repair job changed the original inference settings")
    if repair_job.response_schema != original_job.response_schema:
        raise ValueError("repair job changed the authenticated response schema")
    if repair_job.identifier_ownership != original_job.identifier_ownership:
        raise ValueError("repair job changed authenticated identifier ownership")
    if len(repair_job.messages) != 4:
        raise ValueError("repair job must have exactly four cumulative messages")
    if tuple(repair_job.messages[:2]) != tuple(original_job.messages):
        raise ValueError("repair job changed the original authenticated messages")
    if [message.get("role") for message in repair_job.messages] != [
        "system",
        "user",
        "assistant",
        "user",
    ]:
        raise ValueError("repair job role sequence differs from production")
    binding = repair_job.input_bindings.get(AUTHENTICATED_RESPONSE_REPAIR_BINDING)
    if not isinstance(binding, Mapping):
        raise TypeError("repair job lacks its authenticated repair binding")
    if (
        binding.get("original_job_id") != original_job.job_id
        or binding.get("repair_attempt_number") != 1
        or binding.get("failure_category") != LOCAL_JSON_SCHEMA_VALIDATION_FAILURE
        or binding.get("prior_response_content_sha256")
        != EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256
        or binding.get("policy_sha256")
        != discovery_response_repair_policy_identity()["policy_sha256"]
    ):
        raise ValueError("repair job binding differs from the production continuation")
    rendered_messages = canonical_json(list(repair_job.messages))
    for private_hash in (
        EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256,
        EXPECTED_INITIAL_CONTENT_SHA256,
    ):
        if private_hash in rendered_messages:
            raise ValueError("a private initial response digest entered model-visible messages")
    return binding


def _build_repair_preflight() -> tuple[
    dict[str, Any],
    DiscoveryJsonJob,
    DiscoveryJsonJob,
    tuple[DiscoveryEvidenceItem, ...],
    OpenAICompatibleJsonDiscoveryJobRunner,
]:
    if sys.executable != REQUIRED_INTERPRETER:
        raise ValueError("repair diagnostic requires the exact production interpreter")
    if sys.dont_write_bytecode is not True:
        raise ValueError("repair diagnostic requires PYTHONDONTWRITEBYTECODE=1")
    initial_record, initial_record_file_sha256 = _authenticated_initial_record()
    initial_preflight, original_job, evidence, runner = _build_initial_preflight()
    initial_preflight_sha256 = content_sha256(initial_preflight)
    current_initial_probe_file_sha256 = _sha256_bytes(INITIAL_PROBE_PATH.read_bytes())
    if current_initial_probe_file_sha256 != initial_preflight.get(
        "probe_implementation_file_sha256"
    ):
        runner.close()
        raise ValueError("current initial probe differs from its fresh preflight")
    job_record = initial_record.get("job")
    if not isinstance(job_record, Mapping):
        runner.close()
        raise TypeError("initial rejection job record must be one object")
    if (
        job_record.get("job_kind") != original_job.job_kind
        or job_record.get("job_id") == original_job.job_id
        or job_record.get("request_sha256")
        == initial_preflight["fresh_current_request_sha256"]
        or initial_record.get("hierarchy_implementation_bundle_sha256")
        == initial_preflight["fresh_current_implementation_bundle_sha256"]
    ):
        runner.close()
        raise ValueError(
            "historical rejection and current schema-migrated job identities are not distinct"
        )
    migration = initial_preflight.get("current_schema_migration")
    if (
        not isinstance(migration, Mapping)
        or migration.get("evidence_ownership_and_grouping_preserved") is not True
        or initial_preflight.get("diagnostic_wire_budget_sha256")
        != RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET.content_sha256
    ):
        runner.close()
        raise ValueError("current original job lacks its authenticated schema migration")

    repair_job = _build_response_repair_job_from_projection_sha256(
        original_job=original_job,
        prior_response_content_sha256=EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256,
        failure_category=LOCAL_JSON_SCHEMA_VALIDATION_FAILURE,
    )
    repair_binding = _assert_exact_repair_job(
        original_job=original_job,
        repair_job=repair_job,
    )
    runner_identity = runner.identity()
    request_kwargs = runner._request_kwargs(repair_job)
    response_format = request_kwargs.get("response_format")
    json_schema = (
        response_format.get("json_schema") if isinstance(response_format, Mapping) else None
    )
    if (
        not isinstance(json_schema, Mapping)
        or json_schema.get("strict") is not True
        or json_schema.get("schema") != repair_job.response_schema
    ):
        runner.close()
        raise ValueError("repair transport does not carry the exact strict response schema")
    if (
        request_kwargs.get("model") != MODEL
        or request_kwargs.get("messages") != repair_job.messages
        or request_kwargs.get("max_tokens") != MAX_TOKENS
    ):
        runner.close()
        raise ValueError("repair transport changed model, messages, or token ceiling")
    if MAX_RETRIES != 0:
        runner.close()
        raise ValueError("repair transport must permit exactly one socket attempt")

    preflight = {
        "schema_version": REPAIR_PREFLIGHT_SCHEMA_VERSION,
        "repair_probe_implementation_file_sha256": _sha256_bytes(
            Path(__file__).resolve().read_bytes()
        ),
        "initial_probe_implementation_file_sha256": current_initial_probe_file_sha256,
        "historical_initial_probe_implementation_file_sha256": (
            EXPECTED_INITIAL_PROBE_FILE_SHA256
        ),
        "initial_rejection_record_file_sha256": initial_record_file_sha256,
        "initial_rejection_record_content_sha256": content_sha256(initial_record),
        "initial_probe_preflight_sha256": initial_preflight_sha256,
        "historical_initial_probe_preflight_sha256": EXPECTED_INITIAL_PREFLIGHT_SHA256,
        "continuation_kind": (
            "current_schema_migration_from_authenticated_historical_rejection_v1"
        ),
        "initial_failure_stage": EXPECTED_INITIAL_FAILURE_STAGE,
        "initial_failure_category_persisted": None,
        "repair_category_basis": (
            "strict_parse_succeeded_and_historical_exact_coverage_failure_maps_to_"
            "current_closed_json_schema_validation_failure"
        ),
        "repair_failure_category": LOCAL_JSON_SCHEMA_VALIDATION_FAILURE,
        "initial_response_projection_sha256": (EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256),
        "original_job_id": original_job.job_id,
        "repair_job_id": repair_job.job_id,
        "repair_job_sha256": content_sha256(repair_job.as_dict()),
        "repair_message_envelope_sha256": content_sha256(list(repair_job.messages)),
        "repair_response_schema_sha256": content_sha256(repair_job.response_schema),
        "repair_identifier_ownership_sha256": content_sha256(repair_job.identifier_ownership),
        "repair_binding_sha256": content_sha256(repair_binding),
        "repair_policy_sha256": discovery_response_repair_policy_identity()["policy_sha256"],
        "hierarchy_implementation_bundle_sha256": repair_job.input_bindings[
            HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING
        ],
        "repair_request_sha256": content_sha256(request_kwargs),
        "runner_identity_sha256": runner_identity["identity_sha256"],
        "python_interpreter": sys.executable,
        "python_bytecode_writes_disabled": sys.dont_write_bytecode,
        "endpoint": ENDPOINT,
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "transport_retry_count": MAX_RETRIES,
        "continuation_transport_attempt_count": 1,
        "combined_logical_response_attempt_limit": 2,
        "strict_duplicate_key_parser": True,
        "strict_json_schema_generation": True,
        "finish_reason_required": "stop",
        "evidence_owner_count": len(evidence),
        "owner_member_counts": [len(item.member_ids) for item in evidence],
        "semantic_member_count": sum(len(item.member_ids) for item in evidence),
        "persistence_policy": {
            "hierarchy_job_cache_constructed": False,
            "full_fusion_runner_constructed": False,
            "prediction_path_constructed": False,
            "manifest_writer_constructed": False,
            "oracle_path_constructed": False,
            "raw_response_printed_or_written": False,
            "prior_response_content_model_visible": False,
            "stdout_content": "hashes_counts_sanitized_stages_and_transport_metadata_only",
        },
    }
    if runner.last_execution_metadata is not None or runner.execution_metadata:
        runner.close()
        raise RuntimeError("runner executed during offline repair preflight")
    return preflight, original_job, repair_job, evidence, runner


def _sanitized_transport_metadata(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    top_keys = (
        "job_id",
        "job_kind",
        "request_sha256",
        "runner_identity_sha256",
        "outcome",
        "parsed_response_sha256",
    )
    attempt_keys = (
        "attempt_number",
        "endpoint",
        "model",
        "request_sha256",
        "runner_identity_sha256",
        "response_id",
        "response_model",
        "finish_reason",
        "usage",
        "content_sha256",
        "reasoning_hashes",
        "outcome",
        "exception_type",
        "retryable",
        "will_retry",
        "status_code",
    )
    result = {key: value[key] for key in top_keys if key in value}
    attempts = value.get("attempts")
    if isinstance(attempts, list):
        result["attempts"] = [
            {key: attempt[key] for key in attempt_keys if key in attempt}
            for attempt in attempts
            if isinstance(attempt, Mapping)
        ]
    return json.loads(canonical_json(result))


def _assert_post_transport_sources(preflight: Mapping[str, Any]) -> None:
    if _sha256_bytes(Path(__file__).resolve().read_bytes()) != preflight.get(
        "repair_probe_implementation_file_sha256"
    ):
        raise ValueError("repair probe implementation changed across remote execution")
    if _sha256_bytes(INITIAL_PROBE_PATH.read_bytes()) != preflight.get(
        "initial_probe_implementation_file_sha256"
    ):
        raise ValueError("initial probe implementation changed across repair execution")
    if _sha256_bytes(INITIAL_RECORD_PATH.read_bytes()) != preflight.get(
        "initial_rejection_record_file_sha256"
    ):
        raise ValueError("initial rejection control record changed across repair execution")
    current_bundle = hierarchical_discovery_implementation_bundle().get(
        "implementation_bundle_sha256"
    )
    if current_bundle != preflight.get("hierarchy_implementation_bundle_sha256"):
        raise ValueError("hierarchy implementation changed across repair execution")


def _assert_transport_metadata(
    *,
    metadata: Any,
    preflight: Mapping[str, Any],
    repair_job: DiscoveryJsonJob,
    require_success: bool,
) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise RuntimeError("runner did not retain sanitized execution metadata")
    if metadata.get("job_id") != repair_job.job_id:
        raise ValueError("transport metadata cites a different repair job")
    if metadata.get("job_kind") != repair_job.job_kind:
        raise ValueError("transport metadata cites a different repair job kind")
    if metadata.get("request_sha256") != preflight.get("repair_request_sha256"):
        raise ValueError("executed repair request differs from offline preflight")
    if metadata.get("runner_identity_sha256") != preflight.get("runner_identity_sha256"):
        raise ValueError("executed runner identity differs from offline preflight")
    attempts = metadata.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 1:
        raise ValueError("repair continuation must make exactly one transport attempt")
    attempt = attempts[0]
    if not isinstance(attempt, Mapping):
        raise TypeError("repair transport attempt metadata is malformed")
    if (
        attempt.get("attempt_number") != 1
        or attempt.get("endpoint") != ENDPOINT
        or attempt.get("model") != MODEL
        or attempt.get("request_sha256") != preflight.get("repair_request_sha256")
        or attempt.get("runner_identity_sha256") != preflight.get("runner_identity_sha256")
    ):
        raise ValueError("repair transport attempt differs from the authenticated target")
    if attempt.get("response_model") is not None and attempt.get("response_model") != MODEL:
        raise ValueError("repair response model differs from the exact target")
    if attempt.get("finish_reason") is not None and attempt.get("finish_reason") != "stop":
        raise ValueError("repair diagnostic accepts only finish_reason=stop")
    if require_success:
        if (
            metadata.get("outcome") != "success"
            or attempt.get("outcome") != "success"
            or attempt.get("response_model") != MODEL
            or attempt.get("finish_reason") != "stop"
        ):
            raise ValueError("repair transport did not return one accepted response envelope")
    return metadata


def _rejection(
    *,
    preflight_sha256: str,
    repair_job: DiscoveryJsonJob,
    failure_stage: str,
    failure_category: str,
    failure_type: str,
    metadata: Any,
    response_projection_sha256: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": REPAIR_PROBE_SCHEMA_VERSION,
        "status": "rejected",
        "preflight_sha256": preflight_sha256,
        "repair_job_id": repair_job.job_id,
        "failure_stage": failure_stage,
        "failure_category": failure_category,
        "failure_type": failure_type,
        "transport_metadata": _sanitized_transport_metadata(metadata),
        "combined_logical_response_attempt_count": 2,
        "raw_response_retained": False,
        "exception_text_retained": False,
    }
    if response_projection_sha256 is not None:
        result["repair_response_projection_sha256"] = response_projection_sha256
    return result


def _execute_repair(
    *,
    expected_probe_sha256: str,
    preflight_body: Mapping[str, Any],
    original_job: DiscoveryJsonJob,
    repair_job: DiscoveryJsonJob,
    evidence: Sequence[DiscoveryEvidenceItem],
    runner: OpenAICompatibleJsonDiscoveryJobRunner,
) -> dict[str, Any]:
    preflight_sha256 = content_sha256(preflight_body)
    if expected_probe_sha256 != preflight_sha256:
        runner.close()
        raise ValueError("execution digest differs from the exact repair preflight")
    response: Mapping[str, Any] | None = None
    try:
        try:
            response = runner.run_json(job=repair_job)
        except InvalidDiscoveryJsonResponse as exc:
            metadata = runner.last_execution_metadata
            exc.failed_response_content = ""
            try:
                _assert_post_transport_sources(preflight_body)
            except Exception as authentication_exc:
                return _rejection(
                    preflight_sha256=preflight_sha256,
                    repair_job=repair_job,
                    failure_stage="repair_source_authentication",
                    failure_category="local_authentication_failure",
                    failure_type=authentication_exc.__class__.__name__,
                    metadata=metadata,
                )
            try:
                _assert_transport_metadata(
                    metadata=metadata,
                    preflight=preflight_body,
                    repair_job=repair_job,
                    require_success=False,
                )
            except Exception as authentication_exc:
                return _rejection(
                    preflight_sha256=preflight_sha256,
                    repair_job=repair_job,
                    failure_stage="repair_transport_contract",
                    failure_category="transport_contract_failure",
                    failure_type=authentication_exc.__class__.__name__,
                    metadata=metadata,
                )
            return _rejection(
                preflight_sha256=preflight_sha256,
                repair_job=repair_job,
                failure_stage="repair_strict_json_parse",
                failure_category=STRICT_JSON_PARSE_FAILURE,
                failure_type=exc.__class__.__name__,
                metadata=metadata,
            )
        except Exception as exc:
            metadata = runner.last_execution_metadata
            try:
                _assert_post_transport_sources(preflight_body)
            except Exception as authentication_exc:
                return _rejection(
                    preflight_sha256=preflight_sha256,
                    repair_job=repair_job,
                    failure_stage="repair_source_authentication",
                    failure_category="local_authentication_failure",
                    failure_type=authentication_exc.__class__.__name__,
                    metadata=metadata,
                )
            try:
                _assert_transport_metadata(
                    metadata=metadata,
                    preflight=preflight_body,
                    repair_job=repair_job,
                    require_success=False,
                )
            except Exception as authentication_exc:
                return _rejection(
                    preflight_sha256=preflight_sha256,
                    repair_job=repair_job,
                    failure_stage="repair_transport_contract",
                    failure_category="transport_contract_failure",
                    failure_type=authentication_exc.__class__.__name__,
                    metadata=metadata,
                )
            return _rejection(
                preflight_sha256=preflight_sha256,
                repair_job=repair_job,
                failure_stage="repair_transport",
                failure_category="transport_failure",
                failure_type=exc.__class__.__name__,
                metadata=metadata,
            )

        metadata = runner.last_execution_metadata
        try:
            _assert_post_transport_sources(preflight_body)
        except Exception as exc:
            return _rejection(
                preflight_sha256=preflight_sha256,
                repair_job=repair_job,
                failure_stage="repair_source_authentication",
                failure_category="local_authentication_failure",
                failure_type=exc.__class__.__name__,
                metadata=metadata,
                response_projection_sha256=content_sha256(response),
            )
        try:
            _assert_transport_metadata(
                metadata=metadata,
                preflight=preflight_body,
                repair_job=repair_job,
                require_success=True,
            )
        except Exception as exc:
            return _rejection(
                preflight_sha256=preflight_sha256,
                repair_job=repair_job,
                failure_stage="repair_transport_contract",
                failure_category="transport_contract_failure",
                failure_type=exc.__class__.__name__,
                metadata=metadata,
                response_projection_sha256=content_sha256(response),
            )

        repair_projection_sha256 = content_sha256(response)
        attempt = metadata["attempts"][0]
        if (
            metadata.get("parsed_response_sha256") != repair_projection_sha256
            or attempt.get("parsed_response_sha256") != repair_projection_sha256
        ):
            return _rejection(
                preflight_sha256=preflight_sha256,
                repair_job=repair_job,
                failure_stage="repair_transport_contract",
                failure_category="transport_contract_failure",
                failure_type="ValueError",
                metadata=metadata,
                response_projection_sha256=repair_projection_sha256,
            )

        try:
            validated = validate_interpret_evidence_chunk_response(
                response,
                evidence=evidence,
                wire_budget=RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET,
            )
        except (TypeError, ValueError) as exc:
            return _rejection(
                preflight_sha256=preflight_sha256,
                repair_job=repair_job,
                failure_stage="repair_semantic_validation",
                failure_category=SEMANTIC_VALIDATION_FAILURE,
                failure_type=exc.__class__.__name__,
                metadata=metadata,
                response_projection_sha256=repair_projection_sha256,
            )

        try:
            coverage = _wire_coverage_counts(response, evidence=evidence)
            if coverage != EXPECTED_COVERAGE:
                raise ValueError("validated repair differs from the fixed coverage audit")
        except (TypeError, ValueError) as exc:
            return _rejection(
                preflight_sha256=preflight_sha256,
                repair_job=repair_job,
                failure_stage="repair_exact_coverage_audit",
                failure_category="local_invariant_failure",
                failure_type=exc.__class__.__name__,
                metadata=metadata,
                response_projection_sha256=repair_projection_sha256,
            )

        normalized_sha256 = content_sha256(validated)
        trace = _response_attempt_trace(
            logical_job=original_job,
            attempts=(
                _response_attempt_entry(
                    job=original_job,
                    validation_outcome=LOCAL_JSON_SCHEMA_VALIDATION_FAILURE,
                    raw_response_projection_sha256=(EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256),
                ),
                _response_attempt_entry(
                    job=repair_job,
                    validation_outcome=VALIDATED_RESPONSE,
                    raw_response_projection_sha256=repair_projection_sha256,
                    normalized_validated_response_sha256=normalized_sha256,
                ),
            ),
        )
        trace = _validated_response_attempt_trace(
            logical_job=original_job,
            validated_response_sha256=normalized_sha256,
            trace=trace,
        )
        return {
            "schema_version": REPAIR_PROBE_SCHEMA_VERSION,
            "status": "accepted",
            "preflight_sha256": preflight_sha256,
            "initial_probe_preflight_sha256": preflight_body[
                "initial_probe_preflight_sha256"
            ],
            "initial_response_projection_sha256": (EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256),
            "original_job_id": original_job.job_id,
            "repair_job_id": repair_job.job_id,
            "repair_response_projection_sha256": repair_projection_sha256,
            "normalized_validated_response_sha256": normalized_sha256,
            "logical_response_attempt_trace_sha256": trace["trace_sha256"],
            "combined_logical_response_attempt_count": len(trace["attempts"]),
            "repair_semantic_validation": "passed",
            "coverage": coverage,
            "finish_reason": "stop",
            "transport_metadata": _sanitized_transport_metadata(metadata),
            "raw_response_retained": False,
            "exception_text_retained": False,
        }
    finally:
        response = None
        runner.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="make the single repair call after an exact preflight digest is supplied",
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
        preflight, original_job, repair_job, evidence, runner = _build_repair_preflight()
        preflight_sha256 = content_sha256(preflight)
        if not args.execute:
            if args.expected_probe_sha256:
                raise ValueError("--expected-probe-sha256 is valid only with --execute")
            output = {
                "schema_version": REPAIR_PREFLIGHT_SCHEMA_VERSION,
                "status": "offline_preflight_passed_no_network_client_created",
                "preflight_sha256": preflight_sha256,
                "preflight": preflight,
            }
            runner.close()
            print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if not args.expected_probe_sha256:
            raise ValueError("--execute requires --expected-probe-sha256")
        result = _execute_repair(
            expected_probe_sha256=args.expected_probe_sha256,
            preflight_body=preflight,
            original_job=original_job,
            repair_job=repair_job,
            evidence=evidence,
            runner=runner,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "accepted" else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": REPAIR_PROBE_SCHEMA_VERSION,
                    "status": "preflight_rejected_before_remote_execution",
                    "failure_stage": "offline_preflight_authentication",
                    "failure_category": "local_authentication_failure",
                    "failure_type": exc.__class__.__name__,
                    "raw_response_retained": False,
                    "exception_text_retained": False,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
