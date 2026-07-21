from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    canonical_json,
    content_sha256,
)
from oci.inference.hierarchical_all_architecture_discovery import (
    AUTHENTICATED_RESPONSE_REPAIR_BINDING,
    SEMANTIC_VALIDATION_FAILURE,
    STRICT_JSON_PARSE_FAILURE,
)
from oci.inference.openai_compatible_json_discovery_job_runner import (
    InvalidDiscoveryJsonResponse,
)
from scripts.probe_hierarchical_keyed_first_job_repair_remote import (
    ENDPOINT,
    EXPECTED_COVERAGE,
    EXPECTED_INITIAL_CONTENT_SHA256,
    EXPECTED_INITIAL_RECORD_CONTENT_SHA256,
    EXPECTED_INITIAL_RECORD_FILE_SHA256,
    EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256,
    INITIAL_RECORD_PATH,
    MODEL,
    _build_repair_preflight,
    _execute_repair,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def prepared_repair_probe():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        preflight, original_job, repair_job, evidence, runner = _build_repair_preflight()
    finally:
        sys.dont_write_bytecode = previous
    runner.close()
    assert runner.execution_metadata == ()
    return preflight, original_job, repair_job, evidence


def _valid_wire_response(evidence) -> dict[str, Any]:
    return {
        "concepts": [],
        "evidence_dispositions": {
            item.evidence_id: {
                "status": "reviewed_no_specific_concept",
                "feature_names": [],
                "member_dispositions": {
                    member_id: {"feature_names": []} for member_id in item.member_ids
                },
                "reason": "No specific concept is supported.",
            }
            for item in evidence
        },
    }


class _OneCallHashOnlyRunner:
    def __init__(
        self,
        *,
        preflight: Mapping[str, Any],
        response: Mapping[str, Any] | None = None,
        invalid_content: str | None = None,
        transport_exception: Exception | None = None,
        finish_reason: str = "stop",
        response_model: str = MODEL,
    ) -> None:
        modes = sum(value is not None for value in (response, invalid_content, transport_exception))
        if modes != 1:
            raise ValueError("fake runner requires exactly one response mode")
        self.preflight = preflight
        self.response = response
        self.invalid_content = invalid_content
        self.transport_exception = transport_exception
        self.finish_reason = finish_reason
        self.response_model = response_model
        self.last_execution_metadata: dict[str, Any] | None = None
        self._execution_metadata: list[dict[str, Any]] = []
        self.calls: list[Any] = []
        self.closed = False

    @property
    def execution_metadata(self):
        return tuple(self._execution_metadata)

    def _record(self, value: Mapping[str, Any]) -> None:
        detached = json.loads(canonical_json(value))
        self.last_execution_metadata = detached
        self._execution_metadata.append(detached)

    def run_json(self, *, job):
        self.calls.append(job)
        common_attempt = {
            "attempt_number": 1,
            "endpoint": ENDPOINT,
            "model": MODEL,
            "request_sha256": self.preflight["repair_request_sha256"],
            "runner_identity_sha256": self.preflight["runner_identity_sha256"],
        }
        common_top = {
            "job_id": job.job_id,
            "job_kind": job.job_kind,
            "request_sha256": self.preflight["repair_request_sha256"],
            "runner_identity_sha256": self.preflight["runner_identity_sha256"],
        }
        if self.transport_exception is not None:
            attempt = {
                **common_attempt,
                "outcome": "transport_error",
                "exception_type": self.transport_exception.__class__.__name__,
                "retryable": False,
                "will_retry": False,
            }
            self._record(
                {
                    **common_top,
                    "outcome": "transport_error",
                    "attempts": [attempt],
                }
            )
            raise self.transport_exception
        if self.invalid_content is not None:
            attempt = {
                **common_attempt,
                "response_id": "fake-invalid-response",
                "response_model": self.response_model,
                "finish_reason": self.finish_reason,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "content_sha256": hashlib.sha256(self.invalid_content.encode("utf-8")).hexdigest(),
                "reasoning_hashes": {},
                "outcome": "invalid_response",
                "exception_type": "InvalidDiscoveryJsonResponse",
                "retryable": False,
                "will_retry": False,
            }
            self._record(
                {
                    **common_top,
                    "outcome": "invalid_response",
                    "attempts": [attempt],
                }
            )
            raise InvalidDiscoveryJsonResponse(failed_response_content=self.invalid_content)
        assert self.response is not None
        projection_sha256 = content_sha256(self.response)
        attempt = {
            **common_attempt,
            "response_id": "fake-success-response",
            "response_model": self.response_model,
            "finish_reason": self.finish_reason,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "content_sha256": hashlib.sha256(
                canonical_json(self.response).encode("utf-8")
            ).hexdigest(),
            "reasoning_hashes": {},
            "outcome": "success",
            "retryable": False,
            "will_retry": False,
            "parsed_response_sha256": projection_sha256,
        }
        self._record(
            {
                **common_top,
                "outcome": "success",
                "parsed_response_sha256": projection_sha256,
                "attempts": [attempt],
            }
        )
        return self.response

    def close(self) -> None:
        self.closed = True


def _run(prepared, runner):
    preflight, original_job, repair_job, evidence = prepared
    return _execute_repair(
        expected_probe_sha256=content_sha256(preflight),
        preflight_body=preflight,
        original_job=original_job,
        repair_job=repair_job,
        evidence=evidence,
        runner=runner,
    )


def test_preflight_reconstructs_exact_private_one_repair_job(prepared_repair_probe):
    preflight, original_job, repair_job, _evidence = prepared_repair_probe
    binding = repair_job.input_bindings[AUTHENTICATED_RESPONSE_REPAIR_BINDING]

    assert len(EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256) == 64
    assert len(repair_job.messages) == 4
    assert tuple(repair_job.messages[:2]) == tuple(original_job.messages)
    assert [message["role"] for message in repair_job.messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert binding["failure_category"] == SEMANTIC_VALIDATION_FAILURE
    assert binding["repair_attempt_number"] == 1
    assert binding["prior_response_content_sha256"] == (EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256)
    rendered = canonical_json(list(repair_job.messages))
    assert EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256 not in rendered
    assert EXPECTED_INITIAL_CONTENT_SHA256 not in rendered
    assert preflight["continuation_transport_attempt_count"] == 1
    assert preflight["combined_logical_response_attempt_limit"] == 2
    assert preflight["persistence_policy"]["prior_response_content_model_visible"] is False


def test_corrected_v2_record_closes_and_validates_every_preserved_digest():
    envelope = json.loads(INITIAL_RECORD_PATH.read_text(encoding="utf-8"))
    observed: dict[str, str] = {}

    def collect(value, path):
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if key.endswith("_sha256"):
                    observed[child_path] = child
                collect(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect(child, f"{path}[{index}]")

    collect(envelope, "record")
    assert set(observed) == {
        "record.content_sha256",
        "record.body.transcription_correction.stdout_derived_corrected_sha256",
        "record.body.transcription_correction.retired_v1_record_body_sha256",
        "record.body.transcription_correction.retired_v1_record_file_sha256",
        "record.body.initial_probe_preflight_sha256",
        "record.body.post_execution_offline_preflight_sha256",
        "record.body.probe_implementation_file_sha256",
        "record.body.hierarchy_implementation_bundle_sha256",
        "record.body.job.request_sha256",
        "record.body.job.runner_identity_sha256",
        "record.body.transport.content_sha256",
        "record.body.transport.canonical_parsed_response_projection_sha256",
        "record.body.transport.reasoning_sha256",
    }
    assert all(isinstance(value, str) and _SHA256.fullmatch(value) for value in observed.values())
    assert envelope["content_sha256"] == EXPECTED_INITIAL_RECORD_CONTENT_SHA256
    assert hashlib.sha256(INITIAL_RECORD_PATH.read_bytes()).hexdigest() == (
        EXPECTED_INITIAL_RECORD_FILE_SHA256
    )
    body = envelope["body"]
    assert content_sha256(body) == EXPECTED_INITIAL_RECORD_CONTENT_SHA256
    correction = body["transcription_correction"]
    assert correction["stdout_derived_corrected_sha256"] == (
        EXPECTED_INITIAL_RESPONSE_PROJECTION_SHA256
    )
    assert correction["retired_v1_record_used_for_network_or_repair"] is False
    assert re.fullmatch(r"job_[0-9a-f]{64}", body["job"]["job_id"])


def test_one_repair_accepts_valid_response_and_builds_two_attempt_trace(
    prepared_repair_probe,
):
    preflight, _original_job, repair_job, evidence = prepared_repair_probe
    wire = _valid_wire_response(evidence)
    runner = _OneCallHashOnlyRunner(preflight=preflight, response=wire)

    result = _run(prepared_repair_probe, runner)

    assert result["status"] == "accepted"
    assert result["coverage"] == EXPECTED_COVERAGE
    assert result["combined_logical_response_attempt_count"] == 2
    assert len(result["logical_response_attempt_trace_sha256"]) == 64
    assert result["repair_job_id"] == repair_job.job_id
    assert result["raw_response_retained"] is False
    assert result["exception_text_retained"] is False
    assert len(runner.calls) == 1
    assert runner.calls[0].job_id == repair_job.job_id
    assert runner.closed is True
    serialized = json.dumps(result, sort_keys=True)
    assert json.dumps(wire, sort_keys=True) not in serialized
    assert "evidence_dispositions" not in serialized


def test_semantically_invalid_repair_is_sanitized_exhaustion(prepared_repair_probe):
    preflight, _original_job, _repair_job, evidence = prepared_repair_probe
    wire = _valid_wire_response(evidence)
    first = evidence[0]
    wire["evidence_dispositions"][first.evidence_id]["member_dispositions"].pop(first.member_ids[0])
    runner = _OneCallHashOnlyRunner(preflight=preflight, response=wire)

    result = _run(prepared_repair_probe, runner)

    assert result["status"] == "rejected"
    assert result["failure_stage"] == "repair_semantic_validation"
    assert result["failure_category"] == SEMANTIC_VALIDATION_FAILURE
    assert result["combined_logical_response_attempt_count"] == 2
    assert result["raw_response_retained"] is False
    assert len(runner.calls) == 1
    serialized = json.dumps(result, sort_keys=True)
    assert json.dumps(wire, sort_keys=True) not in serialized
    assert "member_dispositions" not in serialized


def test_strict_parse_failure_on_repair_is_sanitized_exhaustion(
    prepared_repair_probe,
):
    preflight, _original_job, _repair_job, _evidence = prepared_repair_probe
    private_invalid_content = "PRIVATE BROKEN MODEL CONTENT {"
    runner = _OneCallHashOnlyRunner(
        preflight=preflight,
        invalid_content=private_invalid_content,
    )

    result = _run(prepared_repair_probe, runner)

    assert result["status"] == "rejected"
    assert result["failure_stage"] == "repair_strict_json_parse"
    assert result["failure_category"] == STRICT_JSON_PARSE_FAILURE
    assert result["failure_type"] == "InvalidDiscoveryJsonResponse"
    assert result["raw_response_retained"] is False
    assert result["exception_text_retained"] is False
    assert len(runner.calls) == 1
    assert private_invalid_content not in json.dumps(result, sort_keys=True)


@pytest.mark.parametrize(
    ("finish_reason", "response_model"),
    (("length", MODEL), ("stop", "wrong-model")),
)
def test_repair_transport_contract_is_checked_before_semantic_acceptance(
    prepared_repair_probe,
    finish_reason,
    response_model,
):
    preflight, _original_job, _repair_job, evidence = prepared_repair_probe
    runner = _OneCallHashOnlyRunner(
        preflight=preflight,
        response=_valid_wire_response(evidence),
        finish_reason=finish_reason,
        response_model=response_model,
    )

    result = _run(prepared_repair_probe, runner)

    assert result["status"] == "rejected"
    assert result["failure_stage"] == "repair_transport_contract"
    assert result["failure_category"] == "transport_contract_failure"
    assert len(runner.calls) == 1


def test_transport_exception_is_sanitized_and_not_retried(prepared_repair_probe):
    preflight, _original_job, _repair_job, _evidence = prepared_repair_probe
    runner = _OneCallHashOnlyRunner(
        preflight=preflight,
        transport_exception=RuntimeError("PRIVATE TRANSPORT ERROR TEXT"),
    )

    result = _run(prepared_repair_probe, runner)

    assert result["status"] == "rejected"
    assert result["failure_stage"] == "repair_transport"
    assert result["failure_category"] == "transport_failure"
    assert result["failure_type"] == "RuntimeError"
    assert len(runner.calls) == 1
    assert "PRIVATE TRANSPORT ERROR TEXT" not in json.dumps(result, sort_keys=True)


def test_offline_repair_preflight_attempts_no_socket_or_filesystem_write(monkeypatch):
    original_path_open = Path.open
    original_os_open = os.open

    def guarded_path_open(path, mode="r", *args, **kwargs):
        if any(marker in mode for marker in ("w", "a", "x", "+")):
            raise AssertionError(f"repair preflight attempted a filesystem write: {path}")
        return original_path_open(path, mode, *args, **kwargs)

    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC | os.O_EXCL

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & write_flags:
            raise AssertionError(f"repair preflight attempted a low-level write: {path}")
        return original_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("repair preflight attempted a socket connection")
        ),
    )
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    preflight, _original_job, _repair_job, _evidence, runner = _build_repair_preflight()
    try:
        assert runner.execution_metadata == ()
        assert preflight["persistence_policy"]["hierarchy_job_cache_constructed"] is False
        assert preflight["persistence_policy"]["full_fusion_runner_constructed"] is False
    finally:
        runner.close()
