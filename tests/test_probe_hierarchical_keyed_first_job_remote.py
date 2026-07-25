from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    content_sha256,
    validate_interpret_evidence_chunk_response,
)
from oci.inference.openai_compatible_json_discovery_job_runner import (
    parse_strict_json_object,
)
from scripts.probe_hierarchical_keyed_first_job_remote import (
    ENDPOINT,
    EXPECTED_OWNER_MEMBER_COUNTS,
    EXPECTED_RETIRED_JOB_ID,
    EXPECTED_TARGET_MEMBER_ID,
    MODEL,
    RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET,
    _build_preflight,
    _execute,
)


@pytest.fixture(scope="module")
def prepared_probe():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        preflight, job, evidence, runner = _build_preflight()
    finally:
        sys.dont_write_bytecode = previous
    runner.close()
    assert runner.execution_metadata == ()
    return preflight, job, evidence


def _valid_wire_response(evidence) -> dict[str, Any]:
    return {
        "evidence_dispositions": {
            item.evidence_id: {
                "evidence_findings": [],
                "member_dispositions": {
                    member_id: {"findings": []} for member_id in item.member_ids
                },
                "reason": "No specific concept is supported.",
            }
            for item in evidence
        },
    }


class _HashOnlyFakeRunner:
    def __init__(
        self,
        *,
        response: Mapping[str, Any],
        preflight: Mapping[str, Any],
        finish_reason: str = "stop",
    ) -> None:
        self.response = response
        self.preflight = preflight
        self.finish_reason = finish_reason
        self.last_execution_metadata: dict[str, Any] | None = None
        self.closed = False

    def run_json(self, *, job):
        raw_sha256 = content_sha256(self.response)
        attempt = {
            "attempt_number": 1,
            "endpoint": ENDPOINT,
            "model": MODEL,
            "response_model": MODEL,
            "finish_reason": self.finish_reason,
            "request_sha256": self.preflight["fresh_current_request_sha256"],
            "runner_identity_sha256": self.preflight["fresh_current_runner_identity_sha256"],
            "parsed_response_sha256": raw_sha256,
            "outcome": "success",
        }
        self.last_execution_metadata = {
            "job_id": job.job_id,
            "job_kind": job.job_kind,
            "request_sha256": self.preflight["fresh_current_request_sha256"],
            "runner_identity_sha256": self.preflight["fresh_current_runner_identity_sha256"],
            "parsed_response_sha256": raw_sha256,
            "outcome": "success",
            "attempts": [attempt],
        }
        return self.response

    def close(self) -> None:
        self.closed = True


def test_offline_preflight_authenticates_exact_failing_target(prepared_probe):
    preflight, job, evidence = prepared_probe

    assert preflight["retired_target_authentication"]["retired_job_id"] == (EXPECTED_RETIRED_JOB_ID)
    assert job.job_id != EXPECTED_RETIRED_JOB_ID
    assert len(evidence) == 7
    assert tuple(len(item.member_ids) for item in evidence) == (EXPECTED_OWNER_MEMBER_COUNTS)
    assert sum(len(item.member_ids) for item in evidence) == 61
    assert (
        sum(
            member_id == EXPECTED_TARGET_MEMBER_ID
            for item in evidence
            for member_id in item.member_ids
        )
        == 1
    )
    assert preflight["persistence_policy"] == {
        "hierarchy_job_cache_constructed": False,
        "full_fusion_runner_constructed": False,
        "prediction_path_constructed": False,
        "manifest_writer_constructed": False,
        "oracle_path_constructed": False,
        "raw_response_printed_or_written": False,
        "stdout_content": "hashes_counts_and_transport_metadata_only",
    }


def test_fake_transport_accepts_exact_keyed_coverage_without_raw_output(prepared_probe):
    preflight, job, evidence = prepared_probe
    wire = _valid_wire_response(evidence)
    runner = _HashOnlyFakeRunner(response=wire, preflight=preflight)

    result = _execute(
        expected_probe_sha256=content_sha256(preflight),
        preflight_body=preflight,
        job=job,
        evidence=evidence,
        runner=runner,
    )

    assert result["status"] == "accepted"
    assert result["coverage"] == {
        "evidence_owner_count": 7,
        "owner_member_counts": list(EXPECTED_OWNER_MEMBER_COUNTS),
        "semantic_member_count": 61,
        "unique_semantic_member_count": 61,
        "known_target_member_occurrences": 1,
    }
    assert result["raw_response_retained"] is False
    assert "concepts" not in result
    assert "evidence_dispositions" not in result
    assert json.dumps(wire, sort_keys=True) not in json.dumps(result, sort_keys=True)
    assert runner.closed is True


def test_duplicate_json_object_key_is_rejected_before_semantics():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_strict_json_object('{"concepts":[],"concepts":[]}')


def test_missing_member_is_rejected_by_exact_semantics_and_probe(prepared_probe):
    preflight, job, evidence = prepared_probe
    wire = _valid_wire_response(evidence)
    first = evidence[0]
    wire["evidence_dispositions"][first.evidence_id]["member_dispositions"].pop(first.member_ids[0])

    with pytest.raises(ValueError, match="keys differ"):
        validate_interpret_evidence_chunk_response(
            wire,
            evidence=evidence,
            wire_budget=RETIRED_TARGET_DIAGNOSTIC_WIRE_BUDGET,
        )

    runner = _HashOnlyFakeRunner(response=wire, preflight=preflight)
    result = _execute(
        expected_probe_sha256=content_sha256(preflight),
        preflight_body=preflight,
        job=job,
        evidence=evidence,
        runner=runner,
    )
    assert result["status"] == "rejected"
    assert result["failure_type"] == "ValueError"
    assert result["raw_response_retained"] is False
    assert "evidence_dispositions" not in result


def test_finish_reason_length_is_rejected_before_semantic_acceptance(prepared_probe):
    preflight, job, evidence = prepared_probe
    runner = _HashOnlyFakeRunner(
        response=_valid_wire_response(evidence),
        preflight=preflight,
        finish_reason="length",
    )

    result = _execute(
        expected_probe_sha256=content_sha256(preflight),
        preflight_body=preflight,
        job=job,
        evidence=evidence,
        runner=runner,
    )

    assert result["status"] == "rejected"
    assert result["failure_type"] == "ValueError"
    assert result["raw_response_retained"] is False


def test_preflight_attempts_no_socket_connection_or_filesystem_write(monkeypatch):
    original_path_open = Path.open
    original_os_open = os.open

    def guarded_path_open(path, mode="r", *args, **kwargs):
        if any(marker in mode for marker in ("w", "a", "x", "+")):
            raise AssertionError(f"preflight attempted a filesystem write: {path}")
        return original_path_open(path, mode, *args, **kwargs)

    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC | os.O_EXCL

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & write_flags:
            raise AssertionError(f"preflight attempted a low-level filesystem write: {path}")
        return original_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preflight attempted a socket connection")
        ),
    )
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    preflight, _job, _evidence, runner = _build_preflight()
    try:
        assert runner.execution_metadata == ()
        assert preflight["persistence_policy"]["hierarchy_job_cache_constructed"] is False
    finally:
        runner.close()
