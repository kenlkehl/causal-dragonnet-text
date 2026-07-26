"""Fresh path-only validation of real Stage 1/Stage 2 terminal artifacts.

This module deliberately accepts plain mappings and paths rather than live
workflow objects.  It is imported by the short-lived terminal-validation
process after the generic phase manifests have authenticated every registered
byte.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fold_honest_signal_fusion import row_set_fingerprint
from .portable_workflow_spec import (
    PostExtractionCausalReviewSpec,
    ResourcePerformanceSafetyPolicy,
    Stage1ExecutionProfile,
    Stage2PromptProtocolSpec,
)
from .production_text_preparation import stable_file_sha256

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_PREDICTION_COLUMNS = (
    "_oci_row_id",
    "outer_fold",
    "pred_y0_prob",
    "pred_y1_prob",
    "pred_ite_prob",
)
_DIRECT_CATE_PREDICTION_COLUMNS = (
    "_oci_row_id",
    "outer_fold",
    "pred_ite_prob",
)
_ORACLE_EVENTS = (
    "frozen_prediction_bytes_authenticated",
    "prediction_manifest_bytes_authenticated",
    "stage1_row_map_bytes_authenticated",
    "prediction_manifest_schema_and_row_map_validated",
    "oracle_source_opened",
)
_STAGE2_HIERARCHY_PROMPT_PROTOCOL_SCHEMA = "stage2_hierarchy_prompt_protocol_v3"
_STAGE2_PROMPT_NONTRUNCATION_SCHEMA = "stage2_prompt_nontruncation_v1"
_STAGE2_PROMPT_NONTRUNCATION_EXECUTION_AUDIT_SCHEMA = (
    "stage2_prompt_nontruncation_execution_audit_v1"
)
_STAGE2_ONE_SHOT_ATTESTATION_SCHEMA = "production_stage1_hierarchy_one_shot_attestation_v2"
_HIERARCHICAL_BATCH_RESULT_SCHEMA = "hierarchical_all_evidence_runner_batch_result_v1"
_STAGE2_RUN_MANIFEST_SCHEMA = "all_evidence_fusion_predictions_v5"
_PORTABLE_STAGE1_HANDOFF_BINDING_SCHEMA = (
    "production_portable_role_neutral_stage1_handoff_binding_v1"
)
_PROMPT_GUARD_ACCOUNTING = {
    "apply_chat_template": True,
    "tokenize": True,
    "add_generation_prompt": True,
    "truncation": False,
    "endpoint_prompt_usage_exact_match_required": True,
    "request_truncation_controls_allowed": False,
}
_STAGE2_CLIENT_PATHS = {
    "explicit_feature_extraction",
    "hierarchical_discovery",
    "proposal_and_post_extraction_review",
}


def _validate_benchmarked_execution_authority(
    request: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    raw_profile = request.get("stage1_execution_profile")
    if (
        not isinstance(raw_profile, Mapping)
        or raw_profile.get("selection_method")
        != "measured_role_neutral_benchmark_v1"
    ):
        return None
    scientific_spec_path = request.get("scientific_spec_path")
    raw_safety = request.get("resource_performance_safety")
    if (
        not isinstance(scientific_spec_path, str)
        or not scientific_spec_path
        or not isinstance(raw_safety, Mapping)
    ):
        raise ValueError(
            "benchmark-selected terminal request lacks its authorities"
        )
    from .role_neutral_benchmark_deployment_selection import (
        validate_benchmarked_stage1_execution_profile,
    )

    return validate_benchmarked_stage1_execution_profile(
        profile=Stage1ExecutionProfile.from_mapping(raw_profile),
        scientific_spec_path=Path(scientific_spec_path),
        resource_performance_safety=(
            ResourcePerformanceSafetyPolicy.from_mapping(raw_safety)
        ),
        cpu_budget=int(request["cpu_budget"]),
    )


def _canonical_sha256(
    value: Any,
    *,
    ensure_ascii: bool = False,
) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicates(
        pairs: Sequence[tuple[str, Any]],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            output[key] = value
        return output

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one real regular file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite JSON value {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _phase_map(
    phase_records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for record in phase_records:
        phase = str(record.get("phase") or "")
        if not phase or phase in output:
            raise ValueError("terminal phase records have an invalid phase identity")
        output[phase] = record
    return output


def _validate_terminal_phase_order(
    phase_records: Sequence[Mapping[str, Any]],
    *,
    oracle_configured: bool,
) -> Mapping[str, int]:
    """Prove that frozen inference preceded any post-hoc oracle phase."""

    phases = [str(record.get("phase") or "") for record in phase_records]
    required = (
        "stage1_modeling",
        "handoff_validation",
        "stage2_canary",
        "stage2_inference",
    )
    if any(phase not in phases for phase in required):
        raise ValueError("terminal phase ordering lacks a required Stage 1 or Stage 2 phase")
    positions = {phase: phases.index(phase) for phase in required}
    if [positions[phase] for phase in required] != sorted(positions.values()):
        raise ValueError(
            "terminal phase ordering does not freeze Stage 2 after its "
            "Stage 1 graph, handoff, and canary"
        )
    if "oracle_evaluation" in phases:
        positions["oracle_evaluation"] = phases.index("oracle_evaluation")
        if positions["oracle_evaluation"] <= positions["stage2_inference"]:
            raise ValueError(
                "terminal phase ordering opened oracle evaluation before "
                "frozen Stage 2 inference"
            )
    elif oracle_configured:
        raise ValueError("configured oracle evaluation phase is absent")
    return positions


def _artifact_paths(record: Mapping[str, Any] | None) -> set[Path]:
    if record is None:
        return set()
    rows = record.get("artifacts")
    if not isinstance(rows, list):
        raise ValueError("terminal phase record lacks its artifact inventory")
    output: set[Path] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ValueError("terminal phase artifact registration is invalid")
        declared_size = row.get("size_bytes")
        if (
            isinstance(declared_size, bool)
            or not isinstance(declared_size, int)
            or declared_size < 0
        ):
            raise ValueError("terminal phase artifact size registration is invalid")
        _require_sha256(
            row.get("sha256"),
            label="terminal phase artifact content hash",
        )
        raw = Path(str(row["path"]))
        if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
            raise ValueError("terminal phase artifact is not a real absolute file")
        resolved = raw.resolve(strict=True)
        before = resolved.lstat()
        if raw != resolved or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("terminal phase artifact is symlinked, hard-linked, or noncanonical")
        digest, size = stable_file_sha256(resolved)
        after = resolved.lstat()
        if (
            digest != row.get("sha256")
            or size != declared_size
            or (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or resolved in output
        ):
            raise ValueError("terminal phase artifact bytes or registration changed")
        output.add(resolved)
    return output


def _unique_named(
    paths: set[Path],
    name: str,
    *,
    label: str,
) -> Path:
    matches = [path for path in paths if path.name == name]
    if len(matches) != 1:
        raise ValueError(f"{label} must contain exactly one {name}")
    return matches[0]


def _require_sha256(value: Any, *, label: str) -> str:
    rendered = str(value or "")
    if _HEX_SHA256.fullmatch(rendered) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return rendered


def _validate_content_hashed_body(
    value: Mapping[str, Any],
    *,
    schema: str,
    label: str,
) -> Mapping[str, Any]:
    if set(value) != {"schema_version", "content_sha256", "body"}:
        raise ValueError(f"{label} wrapper is not closed")
    body = value.get("body")
    if (
        value.get("schema_version") != schema
        or not isinstance(body, Mapping)
        or value.get("content_sha256") != _canonical_sha256(body)
    ):
        raise ValueError(f"{label} failed content validation")
    return body


def _validated_stage2_request_contract(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    """Validate and freshly authenticate the immutable Stage 2 request."""

    try:
        protocol_spec = Stage2PromptProtocolSpec.from_mapping(request.get("stage2_prompt_protocol"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "immutable workflow request has an invalid Stage 2 prompt protocol"
        ) from exc
    try:
        causal_review_spec = PostExtractionCausalReviewSpec.from_mapping(
            request.get("post_extraction_causal_review")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "immutable workflow request has an invalid post-extraction "
            "causal-review configuration"
        ) from exc
    protocol = {
        "schema_version": _STAGE2_HIERARCHY_PROMPT_PROTOCOL_SCHEMA,
        **protocol_spec.as_dict(),
    }
    causal_review = causal_review_spec.as_dict()
    tokenizer_identity = _fresh_stage2_tokenizer_content_identity(
        request.get("stage2_tokenizer_tree")
    )
    return protocol, causal_review, tokenizer_identity


def _fresh_stage2_tokenizer_content_identity(value: Any) -> dict[str, Any]:
    """Reopen every tokenizer byte and return its path-neutral identity."""

    if not isinstance(value, Mapping):
        raise ValueError("immutable full-workflow request lacks its Stage 2 tokenizer identity")
    expected_tree_keys = {
        "kind",
        "path",
        "file_count",
        "total_size_bytes",
        "tree_sha256",
        "files",
    }
    if set(value) != expected_tree_keys or value.get("kind") != "directory":
        raise ValueError("immutable Stage 2 tokenizer tree identity is not closed")
    declared_files = value.get("files")
    declared_file_count = value.get("file_count")
    declared_total_size = value.get("total_size_bytes")
    if (
        isinstance(declared_file_count, bool)
        or not isinstance(declared_file_count, int)
        or declared_file_count < 1
        or isinstance(declared_total_size, bool)
        or not isinstance(declared_total_size, int)
        or declared_total_size < 0
        or not isinstance(declared_files, list)
        or len(declared_files) != declared_file_count
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"relative_path", "sha256", "size_bytes"}
            or not isinstance(row.get("relative_path"), str)
            or not row["relative_path"]
            or _HEX_SHA256.fullmatch(str(row.get("sha256") or "")) is None
            or isinstance(row.get("size_bytes"), bool)
            or not isinstance(row.get("size_bytes"), int)
            or row["size_bytes"] < 0
            for row in declared_files
        )
    ):
        raise ValueError("immutable Stage 2 tokenizer inventory is invalid")
    _require_sha256(
        value.get("tree_sha256"),
        label="immutable Stage 2 tokenizer content root",
    )
    raw_path = value.get("path")
    if not isinstance(raw_path, str):
        raise ValueError("immutable Stage 2 tokenizer tree lacks its locator")
    supplied = Path(raw_path)
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ValueError("immutable Stage 2 tokenizer locator must be one real absolute directory")
    root = supplied.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("immutable Stage 2 tokenizer locator is not a directory")

    inventory: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise ValueError("Stage 2 tokenizer tree contains a symlink")
        if not candidate.is_file():
            continue
        before = candidate.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("Stage 2 tokenizer tree contains a non-regular or hard-linked member")
        digest, size = stable_file_sha256(candidate)
        after = candidate.lstat()
        before_key = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_key = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_key != after_key or size != after.st_size:
            raise RuntimeError("Stage 2 tokenizer tree changed during terminal authentication")
        inventory.append(
            {
                "relative_path": candidate.relative_to(root).as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    if not inventory:
        raise ValueError("Stage 2 tokenizer tree contains no files")
    observed = {
        "kind": "directory",
        "path": str(root),
        "file_count": len(inventory),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in inventory),
        # The immutable workflow request predates the Stage 2 guard and uses
        # JSON's historical ASCII-escaped canonical form.
        "tree_sha256": _canonical_sha256(inventory, ensure_ascii=True),
        "files": inventory,
    }
    if dict(value) != observed:
        raise ValueError("Stage 2 tokenizer bytes differ from the immutable workflow request")
    path_neutral = {key: child for key, child in observed.items() if key != "path"}
    # The prompt guard deliberately uses UTF-8 canonical JSON.  Re-derive its
    # root from the already byte-authenticated request inventory rather than
    # trusting either producer's independently supplied digest.
    path_neutral["tree_sha256"] = _canonical_sha256(inventory)
    return path_neutral


def _validate_prompt_guard_identity(
    value: Any,
    *,
    request_tokenizer_identity: Mapping[str, Any],
    model_name: Any,
    model_context_window_tokens: int,
    label: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "model_name",
        "model_context_window_tokens",
        "tokenizer_content_identity",
        "chat_template_sha256",
        "tokenizer_class",
        "accounting",
        "identity_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(f"{label} prompt-guard identity is not closed")
    body = {key: child for key, child in value.items() if key != "identity_sha256"}
    tokenizer_class = value.get("tokenizer_class")
    accounting = value.get("accounting")
    if (
        value.get("schema_version") != _STAGE2_PROMPT_NONTRUNCATION_SCHEMA
        or value.get("model_name") != model_name
        or value.get("model_context_window_tokens") != model_context_window_tokens
        or value.get("tokenizer_content_identity") != dict(request_tokenizer_identity)
        or not isinstance(accounting, Mapping)
        or set(accounting) != set(_PROMPT_GUARD_ACCOUNTING)
        or any(
            accounting.get(name) is not expected
            for name, expected in _PROMPT_GUARD_ACCOUNTING.items()
        )
        or value.get("identity_sha256") != _canonical_sha256(body)
        or not isinstance(tokenizer_class, Mapping)
        or set(tokenizer_class) != {"module", "qualname"}
        or any(
            not isinstance(tokenizer_class.get(name), str) or not tokenizer_class[name]
            for name in ("module", "qualname")
        )
    ):
        raise ValueError(f"{label} prompt-guard identity is invalid")
    _require_sha256(
        value.get("chat_template_sha256"),
        label=f"{label} prompt-guard chat template",
    )
    return dict(value)


def _validate_prompt_nontruncation_audit(
    value: Any,
    *,
    guard_identity_sha256: str,
    model_context_window_tokens: int,
    permitted_generation_budgets: set[int],
    expected_request_sha256: Any | None,
    expected_client_path: str | None,
    label: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "guard_identity_sha256",
        "request_sha256",
        "client_path",
        "local_prompt_tokens",
        "maximum_generation_tokens",
        "required_context_tokens",
        "model_context_window_tokens",
        "context_headroom_tokens",
        "truncation_controls_present",
        "tokenizer_truncation_enabled",
        "endpoint_prompt_tokens",
        "endpoint_prompt_tokens_exact_match",
        "status",
        "audit_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(f"{label} prompt nontruncation audit is not closed")
    body = {key: child for key, child in value.items() if key != "audit_sha256"}
    local = value.get("local_prompt_tokens")
    generation = value.get("maximum_generation_tokens")
    required = value.get("required_context_tokens")
    context = value.get("model_context_window_tokens")
    headroom = value.get("context_headroom_tokens")
    endpoint = value.get("endpoint_prompt_tokens")
    integer_values = (local, generation, required, context, endpoint)
    if (
        value.get("schema_version") != _STAGE2_PROMPT_NONTRUNCATION_SCHEMA
        or value.get("guard_identity_sha256") != guard_identity_sha256
        or _HEX_SHA256.fullmatch(str(value.get("request_sha256") or "")) is None
        or value.get("client_path") not in _STAGE2_CLIENT_PATHS
        or (expected_client_path is not None and value.get("client_path") != expected_client_path)
        or (
            expected_request_sha256 is not None
            and value.get("request_sha256") != expected_request_sha256
        )
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in integer_values
        )
        or isinstance(headroom, bool)
        or not isinstance(headroom, int)
        or headroom < 0
        or generation not in permitted_generation_budgets
        or context != model_context_window_tokens
        or required != local + generation
        or headroom != context - required
        or endpoint != local
        or value.get("truncation_controls_present") is not False
        or value.get("tokenizer_truncation_enabled") is not False
        or value.get("endpoint_prompt_tokens_exact_match") is not True
        or value.get("status") != "accepted_nontruncated"
        or value.get("audit_sha256") != _canonical_sha256(body)
    ):
        raise ValueError(f"{label} prompt nontruncation audit is invalid")
    return dict(value)


def _validate_prompt_nontruncation_execution_audit(
    value: Any,
    *,
    guard_identity_sha256: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "guard_identity_sha256",
        "record_count",
        "records",
        "records_sha256",
        "record_counts_by_client_path",
        "unclassified_record_count",
        "all_records_status",
        "all_endpoint_prompt_tokens_exact_match",
        "all_request_audits_authenticated",
        "all_guard_identities_exact_match",
        "all_requests_forbid_truncation_controls",
        "audit_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("Stage 2 prompt nontruncation execution audit is not closed")
    body = {key: child for key, child in value.items() if key != "audit_sha256"}
    records = value.get("records")
    counts = value.get("record_counts_by_client_path")
    count = value.get("record_count")
    unclassified = value.get("unclassified_record_count")
    if (
        value.get("schema_version") != _STAGE2_PROMPT_NONTRUNCATION_EXECUTION_AUDIT_SCHEMA
        or value.get("guard_identity_sha256") != guard_identity_sha256
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not isinstance(records, list)
        or len(records) != count
        or value.get("records_sha256") != _canonical_sha256(records)
        or not isinstance(counts, Mapping)
        or set(counts) != _STAGE2_CLIENT_PATHS
        or any(
            isinstance(child, bool) or not isinstance(child, int) or child < 0
            for child in counts.values()
        )
        or sum(counts.values()) != count
        or isinstance(unclassified, bool)
        or not isinstance(unclassified, int)
        or unclassified != 0
        or value.get("all_records_status") != "accepted_nontruncated"
        or value.get("all_endpoint_prompt_tokens_exact_match") is not True
        or value.get("all_request_audits_authenticated") is not True
        or value.get("all_guard_identities_exact_match") is not True
        or value.get("all_requests_forbid_truncation_controls") is not True
        or value.get("audit_sha256") != _canonical_sha256(body)
    ):
        raise ValueError("Stage 2 prompt nontruncation execution audit is invalid")
    observed_counts = {path: 0 for path in _STAGE2_CLIENT_PATHS}
    permitted_generation_budgets = {
        int(protocol["proposal_max_tokens"]),
        int(protocol["extraction_max_tokens"]),
    }
    for index, record in enumerate(records):
        validated = _validate_prompt_nontruncation_audit(
            record,
            guard_identity_sha256=guard_identity_sha256,
            model_context_window_tokens=int(protocol["model_context_window_tokens"]),
            permitted_generation_budgets=permitted_generation_budgets,
            expected_request_sha256=None,
            expected_client_path=None,
            label=f"Stage 2 one-shot execution record {index}",
        )
        observed_counts[str(validated["client_path"])] += 1
    if dict(counts) != dict(sorted(observed_counts.items())):
        raise ValueError("Stage 2 prompt nontruncation client-path counts are invalid")
    return dict(value)


def _validate_portable_stage1_handoff_binding(
    *,
    request: Mapping[str, Any],
    stage1_paths: set[Path],
    bundle_path: Path,
    bundle: Mapping[str, Any],
    numerical_manifest_path: Path,
    numerical_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Authenticate the workflow-to-reference-graph binding from paths only."""

    from .direct_upstream_numerical_reference_bank import (
        DIRECT_NUMERICAL_REFERENCE_LOCATOR,
    )
    from .production_stage1_role_neutral_execution import (
        ROLE_NEUTRAL_EXECUTION_MANIFEST,
    )

    binding_path = _unique_named(
        stage1_paths,
        "role_neutral_handoff_binding.json",
        label="portable Stage 1 phase",
    )
    execution_path = _unique_named(
        stage1_paths,
        ROLE_NEUTRAL_EXECUTION_MANIFEST,
        label="portable Stage 1 phase",
    )
    numerical_locator_path = _unique_named(
        stage1_paths,
        DIRECT_NUMERICAL_REFERENCE_LOCATOR,
        label="portable Stage 1 phase",
    )
    if stage1_paths != {
        binding_path,
        execution_path,
        bundle_path,
        numerical_manifest_path,
        numerical_locator_path,
    }:
        raise ValueError(
            "portable Stage 1 phase contains an incomplete or unrelated "
            "terminal artifact inventory"
        )
    binding = _read_json_object(
        binding_path,
        label="portable Stage 1 workflow handoff binding",
    )
    expected_keys = {
        "schema_version",
        "workflow_request_sha256",
        "prepared_stage1_request_sha256",
        "stage1_scope_plan_scientific_content_sha256",
        "role_neutral_execution_manifest",
        "stage2_bundle_manifest",
        "direct_numerical_reference_bank",
        "integration_code_identity",
        "physical_fit_count",
        "logical_scope_count",
        "deduplicated_fit_count",
        "productive_compute_canary_completed",
        "selected_canary_replica_adopted_as_production",
        "compute_canary_scientific_equality",
        "legacy_bundle_build_invoked",
        "all_ten_role_neutral_execution_is_exclusive_evidence_source",
        "stage2_loader_validation",
        "content_sha256",
    }
    body = {key: value for key, value in binding.items() if key != "content_sha256"}
    execution = binding.get("role_neutral_execution_manifest")
    bundle_registration = binding.get("stage2_bundle_manifest")
    numerical = binding.get("direct_numerical_reference_bank")
    integration = binding.get("integration_code_identity")
    if (
        set(binding) != expected_keys
        or binding.get("schema_version") != _PORTABLE_STAGE1_HANDOFF_BINDING_SCHEMA
        or binding.get("content_sha256") != _canonical_sha256(body)
        or binding.get("workflow_request_sha256") != request.get("request_sha256")
        or binding.get("prepared_stage1_request_sha256") != bundle.get("request_sha256")
        or binding.get("stage1_scope_plan_scientific_content_sha256")
        != bundle.get("scope_plan_scientific_content_sha256")
        or binding.get("physical_fit_count") != bundle.get("physical_fit_count")
        or binding.get("logical_scope_count") != bundle.get("logical_scope_count")
        or binding.get("deduplicated_fit_count")
        != int(bundle["logical_scope_count"]) - int(bundle["physical_fit_count"])
        or binding.get("productive_compute_canary_completed") is not False
        or binding.get("selected_canary_replica_adopted_as_production") is not False
        or binding.get("compute_canary_scientific_equality") is not None
        or binding.get("legacy_bundle_build_invoked") is not False
        or binding.get("all_ten_role_neutral_execution_is_exclusive_evidence_source") is not True
        or binding.get("stage2_loader_validation")
        != "reference_only_role_neutral_provider_accepted"
        or not isinstance(execution, Mapping)
        or set(execution) != {"relative_path", "sha256", "size_bytes", "content_sha256"}
        or not isinstance(bundle_registration, Mapping)
        or set(bundle_registration) != {"relative_path", "sha256", "size_bytes", "bundle_sha256"}
        or not isinstance(numerical, Mapping)
        or set(numerical)
        != {
            "relative_path",
            "content_sha256",
            "source_execution_content_sha256",
            "combined_npy_payloads_persisted",
        }
        or not isinstance(integration, Mapping)
    ):
        raise ValueError("portable Stage 1 workflow handoff binding is invalid")
    integration_body = {key: value for key, value in integration.items() if key != "content_sha256"}
    if (
        set(integration)
        != {
            "schema_version",
            "producer_factories_builder",
            "physical_owner_executor",
            "stage2_handoff_publisher",
            "content_sha256",
        }
        or integration.get("schema_version")
        != "production_role_neutral_stage1_integration_code_identity_v1"
        or integration.get("content_sha256") != _canonical_sha256(integration_body)
    ):
        raise ValueError("portable Stage 1 integration-code identity is invalid")

    def registered_relative(
        registration: Mapping[str, Any],
        expected: Path,
        *,
        label: str,
    ) -> tuple[str, int]:
        relative = registration.get("relative_path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"{label} lacks its relative path")
        raw = Path(relative)
        if raw.is_absolute() or ".." in raw.parts:
            raise ValueError(f"{label} relative path escaped its phase")
        resolved = (binding_path.parent / raw).resolve(strict=True)
        if resolved != expected:
            raise ValueError(f"{label} path was substituted")
        return stable_file_sha256(resolved)

    execution_sha, execution_size = registered_relative(
        execution,
        execution_path,
        label="portable role-neutral execution manifest",
    )
    bundle_sha, bundle_size = registered_relative(
        bundle_registration,
        bundle_path,
        label="portable Stage 1 reference manifest",
    )
    numerical_sha, _numerical_size = registered_relative(
        numerical,
        numerical_manifest_path,
        label="portable direct numerical manifest",
    )
    # The numerical registration intentionally binds its path-neutral manifest
    # content identity, not an unrelated caller-provided file digest.  The
    # loader above independently reopened both manifest and locator bytes.
    del numerical_sha
    if (
        execution.get("sha256") != execution_sha
        or execution.get("size_bytes") != execution_size
        or execution.get("content_sha256")
        != bundle.get("source_role_neutral_execution_content_sha256")
        or bundle_registration.get("sha256") != bundle_sha
        or bundle_registration.get("size_bytes") != bundle_size
        or bundle_registration.get("bundle_sha256") != bundle.get("bundle_sha256")
        or numerical.get("content_sha256") != numerical_identity.get("manifest_content_sha256")
        or numerical.get("source_execution_content_sha256")
        != bundle.get("source_role_neutral_execution_content_sha256")
        or numerical.get("combined_npy_payloads_persisted") is not False
    ):
        raise ValueError("portable Stage 1 workflow handoff references changed")
    return {
        "path": str(binding_path),
        "content_sha256": binding["content_sha256"],
        "execution_manifest_path": str(execution_path),
        "numerical_locator_path": str(numerical_locator_path),
        "closed_terminal_inventory_validated": True,
    }


def validate_real_stage1_handoff(
    *,
    request: Mapping[str, Any],
    phase_records: Sequence[Mapping[str, Any]],
    _validated_phase_map: Mapping[str, Mapping[str, Any]] | None = None,
    _validated_path_inventory: Mapping[str, set[Path]] | None = None,
) -> Mapping[str, Any]:
    """Reopen the bundle and the independently produced handoff report."""

    by_phase = (
        dict(_validated_phase_map)
        if _validated_phase_map is not None
        else _phase_map(phase_records)
    )
    paths_by_phase = (
        dict(_validated_path_inventory)
        if _validated_path_inventory is not None
        else {phase: _artifact_paths(record) for phase, record in by_phase.items()}
    )
    stage1_paths = paths_by_phase.get("stage1_modeling", set())
    bundles = [path for path in stage1_paths if path.name == "bundle_manifest.json"]
    if not bundles:
        return {
            "real_stage1_handoff_detected": False,
            "reason": "injected_stage1_phase_without_bundle_manifest",
        }
    if len(bundles) != 1:
        raise ValueError("terminal validation found multiple Stage 1 bundle manifests")
    bundle_path = bundles[0]
    bundle = _read_json_object(bundle_path, label="Stage 1 bundle manifest")
    bundle_body = dict(bundle)
    bundle_sha = _require_sha256(
        bundle_body.pop("bundle_sha256", None),
        label="Stage 1 bundle content hash",
    )
    if _canonical_sha256(bundle_body) != bundle_sha:
        raise ValueError("Stage 1 bundle manifest content hash is invalid")

    handoff_paths = paths_by_phase.get("handoff_validation", set())
    report_path = _unique_named(
        handoff_paths,
        "fresh_handoff_validation.json",
        label="Stage 1 handoff phase",
    )
    report = _read_json_object(
        report_path,
        label="fresh Stage 1 handoff validation report",
    )
    expected_keys = {
        "schema_version",
        "status",
        "bundle_manifest_path",
        "review_rounds",
        "initial_training_partitions",
        "interaction_inner_folds",
        "tfidf_nested_calibration_folds",
        "handoff",
        "remote_clients_constructed",
        "remote_calls_made",
        "loader_module_path",
        "content_sha256",
    }
    report_body = {key: value for key, value in report.items() if key != "content_sha256"}
    handoff = report.get("handoff")
    if (
        set(report) != expected_keys
        or report.get("schema_version") != "production_stage1_fresh_handoff_validation_v1"
        or report.get("status") != "accepted"
        or Path(str(report.get("bundle_manifest_path", ""))).resolve(strict=True) != bundle_path
        or int(report.get("review_rounds", -1)) != int(request["review_rounds"])
        or int(report.get("initial_training_partitions", -1))
        != int(request["initial_training_partitions"])
        or int(report.get("interaction_inner_folds", -1)) != int(request["interaction_inner_folds"])
        or int(report.get("tfidf_nested_calibration_folds", -1))
        != int(request["tfidf_nested_calibration_folds"])
        or report.get("remote_clients_constructed") is not False
        or report.get("remote_calls_made") is not False
        # This report is produced by the workflow subprocess, whose immutable
        # request/report canonicalizer intentionally retains JSON's historical
        # ASCII-escaping behavior.
        or report.get("content_sha256") != _canonical_sha256(report_body, ensure_ascii=True)
        or not isinstance(handoff, Mapping)
    ):
        raise ValueError("fresh Stage 1 handoff validation report is invalid")
    handoff_body = {key: value for key, value in handoff.items() if key != "content_sha256"}
    stage1_inputs = handoff.get("stage1_inputs")
    if (
        handoff.get("content_sha256") != _canonical_sha256(handoff_body)
        or handoff.get("all_ten_architectures_required") is not True
        or handoff.get("per_architecture_interpretation_required") is not True
        or handoff.get("raw_all_architecture_prompt_allowed") is not False
        or handoff.get("independent_runtime_stage1_refit_allowed") is not False
        or handoff.get("manual_digest_approval_required") is not False
        or not isinstance(stage1_inputs, Mapping)
        or stage1_inputs.get("bundle_sha256") != bundle_sha
    ):
        raise ValueError("fresh Stage 1 handoff content is invalid")

    result: dict[str, Any] = {
        "real_stage1_handoff_detected": True,
        "bundle_manifest_path": str(bundle_path),
        "bundle_sha256": bundle_sha,
        "handoff_report_path": str(report_path),
        "handoff_content_sha256": handoff["content_sha256"],
        "all_ten_architectures_required": True,
    }
    if bundle.get("schema_version") != ("production_role_neutral_stage1_reference_handoff_v1"):
        return result

    # The portable path is authenticated by reopening its complete reference
    # graph in this fresh validator process.  Do not reinterpret the direct
    # manifest as a legacy bundle or accept a manually supplied digest.
    from .direct_upstream_numerical_reference_bank import (
        DIRECT_NUMERICAL_REFERENCE_MANIFEST,
        load_role_neutral_direct_numerical_reference_bank,
    )
    from .production_role_neutral_stage2_handoff import (
        ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
        load_reference_only_role_neutral_stage1_handoff,
    )

    publication = load_reference_only_role_neutral_stage1_handoff(bundle_path)
    reopened_handoff = dict(publication.as_dict())
    source_execution_sha256 = bundle.get("source_role_neutral_execution_content_sha256")
    row_map_path = bundle_path.parent / "row_registry.parquet"
    row_map_sha256, _row_map_size = stable_file_sha256(row_map_path)
    numerical_paths = [
        path for path in stage1_paths if path.name == DIRECT_NUMERICAL_REFERENCE_MANIFEST
    ]
    if (
        handoff != reopened_handoff
        or handoff.get("handoff_kind") != ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
        or handoff.get("handoff_scientific_content_sha256") != bundle.get("content_sha256")
        or stage1_inputs.get("source_role_neutral_execution_content_sha256")
        != source_execution_sha256
        or publication.source_role_neutral_execution_content_sha256 != source_execution_sha256
        or publication.legacy_bundle_build_invoked is not False
        or publication.all_ten_role_neutral_execution_is_exclusive_evidence_source is not True
        or bundle.get("legacy_bundle_build_invoked") is not False
        or bundle.get("independent_stage1_refit_invoked") is not False
        or bundle.get("all_ten_role_neutral_execution_is_exclusive_evidence_source") is not True
        or bundle.get("text_truncation_applied") is not False
        or bundle.get("lossy_evidence_selection_applied") is not False
        or bundle.get("offline_handoff_validation_complete") is not True
        or bundle.get("full_stage2_one_shot_runtime_complete") is not False
        or row_map_sha256 != bundle.get("row_map_sha256")
        or len(numerical_paths) != 1
        or publication.stage2_provider is None
    ):
        raise ValueError("fresh portable Stage 1 reference handoff is invalid")
    provider = publication.stage2_provider
    plan = provider.authenticated_scope_plan()
    outer_assignments = provider.get_outer_fold_assignments()
    normalized_assignments = {
        int(fold): {
            "fit_row_ids": list(map(int, assignment["fit_row_ids"])),
            "heldout_row_ids": list(map(int, assignment["heldout_row_ids"])),
        }
        for fold, assignment in outer_assignments.items()
    }
    if tuple(sorted(normalized_assignments)) != tuple(
        range(1, len(normalized_assignments) + 1)
    ) or any(
        not row["fit_row_ids"]
        or not row["heldout_row_ids"]
        or len(row["fit_row_ids"]) != len(set(row["fit_row_ids"]))
        or len(row["heldout_row_ids"]) != len(set(row["heldout_row_ids"]))
        or set(row["fit_row_ids"]) & set(row["heldout_row_ids"])
        for row in normalized_assignments.values()
    ):
        raise ValueError("fresh portable Stage 1 outer-fold assignments are invalid")
    heldout_coverage = [
        row_id
        for fold in sorted(normalized_assignments)
        for row_id in normalized_assignments[fold]["heldout_row_ids"]
    ]
    if len(heldout_coverage) != len(set(heldout_coverage)) or set(heldout_coverage) != set(
        range(len(heldout_coverage))
    ):
        raise ValueError("fresh portable Stage 1 outer-heldout coverage is incomplete")
    numerical_bank = load_role_neutral_direct_numerical_reference_bank(
        manifest_path=numerical_paths[0],
        plan=plan,
    )
    numerical_identity = numerical_bank.identity()
    if (
        numerical_identity.get("plan_scientific_content_sha256")
        != bundle.get("scope_plan_scientific_content_sha256")
        or numerical_identity.get("source_execution_content_sha256") != source_execution_sha256
    ):
        raise ValueError("fresh portable Stage 1 numerical bank belongs to another graph")
    workflow_binding = _validate_portable_stage1_handoff_binding(
        request=request,
        stage1_paths=stage1_paths,
        bundle_path=bundle_path,
        bundle=bundle,
        numerical_manifest_path=numerical_paths[0],
        numerical_identity=numerical_identity,
    )
    result.update(
        {
            "handoff_kind": ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
            "handoff_scientific_content_sha256": bundle["content_sha256"],
            "source_role_neutral_execution_content_sha256": (source_execution_sha256),
            "stage2_provider_identity_sha256": bundle["stage2_provider_identity_sha256"],
            "scope_plan_scientific_content_sha256": bundle["scope_plan_scientific_content_sha256"],
            "physical_fit_count": int(bundle["physical_fit_count"]),
            "logical_scope_count": int(bundle["logical_scope_count"]),
            "outer_fold_assignments": normalized_assignments,
            "row_map_path": str(row_map_path.resolve(strict=True)),
            "row_map_sha256": row_map_sha256,
            "direct_numerical_bank_manifest_path": str(numerical_paths[0]),
            "direct_numerical_bank_content_sha256": numerical_identity["manifest_content_sha256"],
            "reference_only_graph_reopened_in_fresh_process": True,
            "legacy_bundle_loader_invoked": False,
            "portable_workflow_binding": workflow_binding,
        }
    )
    return result


def validate_real_stage2_canary(
    *,
    request: Mapping[str, Any],
    phase_records: Sequence[Mapping[str, Any]],
    handoff_validation: Mapping[str, Any],
    _validated_phase_map: Mapping[str, Mapping[str, Any]] | None = None,
    _validated_path_inventory: Mapping[str, set[Path]] | None = None,
    _validated_request_contract: (
        tuple[dict[str, Any], dict[str, float], dict[str, Any]] | None
    ) = None,
) -> Mapping[str, Any]:
    """Validate the one-call remote canary and its exact endpoint metadata."""

    if handoff_validation.get("real_stage1_handoff_detected") is not True:
        raise ValueError("real Stage 2 canary lacks a validated Stage 1 handoff")
    by_phase = (
        dict(_validated_phase_map)
        if _validated_phase_map is not None
        else _phase_map(phase_records)
    )
    paths_by_phase = (
        dict(_validated_path_inventory)
        if _validated_path_inventory is not None
        else {phase: _artifact_paths(record) for phase, record in by_phase.items()}
    )
    paths = paths_by_phase.get("stage2_canary", set())
    from .production_role_neutral_stage2_handoff import (
        ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
    )

    portable_direct = (
        handoff_validation.get("handoff_kind") == ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
    )
    if portable_direct:
        from scripts import (
            canary_production_stage1_hierarchy as canary_module,
        )

        report_name = canary_module.ROLE_NEUTRAL_STAGE2_CANARY_REPORT_FILENAME
        report_schema = canary_module.ROLE_NEUTRAL_STAGE2_CANARY_REPORT_SCHEMA
    else:
        report_name = "production_stage1_hierarchy_runtime_canary.json"
        report_schema = "production_stage1_hierarchy_runtime_canary_report_v2"
    report_path = _unique_named(
        paths,
        report_name,
        label="Stage 2 canary phase",
    )
    wrapper = _read_json_object(report_path, label="Stage 2 canary report")
    body = _validate_content_hashed_body(
        wrapper,
        schema=report_schema,
        label="Stage 2 canary report",
    )
    expected_body_keys = {
        "status",
        "canary_kind",
        "authorization_role",
        "stage1_bundle",
        "endpoint",
        "model",
        "runner_identity_sha256",
        "settings",
        "selected_job",
        "validation",
        "remote_response_count",
        "transport_metadata",
        "raw_prompt_emitted",
        "raw_response_emitted",
        "normalized_findings_emitted",
        "prediction_path_constructed",
        "oracle_path_constructed",
        "full_fusion_runner_executed",
        "canary_implementation_file_sha256",
    }
    direct_body_keys = {
        "reference_only_role_neutral_stage1",
        "legacy_stage1_loader_invoked",
        "independent_stage1_refit_performed",
    }
    stage1_bundle = body.get("stage1_bundle")
    settings = body.get("settings")
    selected = body.get("selected_job")
    validation = body.get("validation")
    transports = body.get("transport_metadata")
    protocol = (
        settings.get("stage2_hierarchy_prompt_protocol") if isinstance(settings, Mapping) else None
    )
    (
        configured_protocol,
        configured_causal_review,
        _request_tokenizer_identity,
    ) = (
        _validated_stage2_request_contract(request)
        if _validated_request_contract is None
        else _validated_request_contract
    )
    prompt_guard_identity_sha256 = (
        settings.get("prompt_nontruncation_guard_identity_sha256")
        if isinstance(settings, Mapping)
        else None
    )
    if portable_direct:
        direct_stage1_keys = {
            "manifest_path",
            "handoff_kind",
            "bundle_sha256",
            "handoff_content_sha256",
            "source_execution_content_sha256",
            "provider_identity_sha256",
            "reference_only_all_ten",
            "legacy_stage1_loader_invoked",
            "independent_stage1_refit_performed",
        }
        stage1_bundle_valid = bool(
            isinstance(stage1_bundle, Mapping)
            and set(stage1_bundle) == direct_stage1_keys
            and stage1_bundle.get("handoff_kind") == ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
            and stage1_bundle.get("handoff_content_sha256")
            == handoff_validation.get("handoff_scientific_content_sha256")
            and stage1_bundle.get("source_execution_content_sha256")
            == handoff_validation.get("source_role_neutral_execution_content_sha256")
            and stage1_bundle.get("provider_identity_sha256")
            == handoff_validation.get("stage2_provider_identity_sha256")
            and stage1_bundle.get("reference_only_all_ten") is True
            and stage1_bundle.get("legacy_stage1_loader_invoked") is False
            and stage1_bundle.get("independent_stage1_refit_performed") is False
        )
    else:
        stage1_bundle_valid = bool(
            isinstance(stage1_bundle, Mapping)
            and set(stage1_bundle) == {"manifest_path", "bundle_sha256", "handoff_content_sha256"}
            and stage1_bundle.get("handoff_content_sha256")
            == handoff_validation.get("handoff_content_sha256")
        )
    if isinstance(stage1_bundle, Mapping):
        try:
            stage1_bundle_path = Path(str(stage1_bundle.get("manifest_path", ""))).resolve(
                strict=True
            )
        except (OSError, RuntimeError):
            stage1_bundle_path = None
    else:
        stage1_bundle_path = None
    stage1_bundle_valid = bool(
        stage1_bundle_valid
        and stage1_bundle_path == Path(str(handoff_validation["bundle_manifest_path"]))
        and stage1_bundle.get("bundle_sha256") == handoff_validation.get("bundle_sha256")
    )
    if (
        (
            set(body)
            != (expected_body_keys | direct_body_keys if portable_direct else expected_body_keys)
            and not (
                not portable_direct
                and set(body) == expected_body_keys | direct_body_keys
                and body.get("reference_only_role_neutral_stage1") is False
                and body.get("legacy_stage1_loader_invoked") is None
                and body.get("independent_stage1_refit_performed") is None
            )
        )
        or body.get("status") != "accepted"
        or body.get("canary_kind") != "one_real_architecture_pure_initial_interpretation_job"
        or body.get("authorization_role") != "non_authorizing_operational_runtime_check"
        or body.get("endpoint") != request.get("endpoint")
        or body.get("model") != request.get("model_name")
        or not stage1_bundle_valid
        or (
            portable_direct
            and (
                body.get("reference_only_role_neutral_stage1") is not True
                or body.get("legacy_stage1_loader_invoked") is not False
                or body.get("independent_stage1_refit_performed") is not False
            )
        )
        or not isinstance(settings, Mapping)
        or set(settings)
        != {
            "proposal_max_tokens",
            "extraction_max_tokens",
            "stage2_hierarchy_prompt_protocol",
            "stage2_hierarchy_prompt_protocol_sha256",
            "post_extraction_causal_review",
            "post_extraction_causal_review_sha256",
            "prompt_nontruncation_guard_identity_sha256",
            "transport_retries",
            "selector_thinking_enabled",
            "selector_thinking_token_budget",
            "max_rendered_discovery_prompt_bytes",
            "final_upstream_max_orphan_features",
            "review_neural_query_nuisance_folds",
            "final_upstream_meta_inner_folds",
            "final_upstream_head_regularization",
            "extraction_thinking_enabled",
            "maximum_schema_repairs",
        }
        or not isinstance(protocol, Mapping)
        or dict(protocol) != configured_protocol
        or settings.get("stage2_hierarchy_prompt_protocol_sha256") != _canonical_sha256(protocol)
        or settings.get("post_extraction_causal_review") != configured_causal_review
        or settings.get("post_extraction_causal_review_sha256")
        != _canonical_sha256(configured_causal_review)
        or settings.get("proposal_max_tokens") != protocol.get("proposal_max_tokens")
        or settings.get("extraction_max_tokens") != protocol.get("extraction_max_tokens")
        or settings.get("selector_thinking_token_budget")
        != protocol.get("selector_thinking_token_budget")
        or settings.get("max_rendered_discovery_prompt_bytes")
        != protocol.get("max_rendered_discovery_prompt_bytes")
        or settings.get("final_upstream_max_orphan_features")
        != protocol.get("final_upstream_max_orphan_features")
        or settings.get("review_neural_query_nuisance_folds")
        != protocol.get("review_neural_query_nuisance_folds")
        or settings.get("final_upstream_meta_inner_folds")
        != protocol.get("final_upstream_meta_inner_folds")
        or settings.get("final_upstream_head_regularization")
        != protocol.get("final_upstream_head_regularization")
        or isinstance(settings.get("proposal_max_tokens"), bool)
        or not isinstance(settings.get("proposal_max_tokens"), int)
        or int(settings["proposal_max_tokens"]) < 1
        or isinstance(settings.get("extraction_max_tokens"), bool)
        or not isinstance(settings.get("extraction_max_tokens"), int)
        or int(settings["extraction_max_tokens"]) < 1
        or settings.get("transport_retries") != 0
        or settings.get("maximum_schema_repairs") != 1
        or settings.get("selector_thinking_enabled") is not True
        or settings.get("extraction_thinking_enabled") is not False
        or not isinstance(selected, Mapping)
        or not isinstance(validation, Mapping)
        or not isinstance(transports, list)
        or isinstance(body.get("remote_response_count"), bool)
        or not isinstance(body.get("remote_response_count"), int)
        or body.get("remote_response_count") not in {1, 2}
        or len(transports) != body["remote_response_count"]
        or body.get("raw_prompt_emitted") is not False
        or body.get("raw_response_emitted") is not False
        or body.get("normalized_findings_emitted") is not False
        or body.get("prediction_path_constructed") is not False
        or body.get("oracle_path_constructed") is not False
        or body.get("full_fusion_runner_executed") is not False
    ):
        raise ValueError("Stage 2 canary scientific/transport contract is invalid")
    if portable_direct:
        from .all_evidence_discovery_interfaces import (
            ACTIVE_STAGE1_CONCEPT_FAMILIES,
        )

        selected_keys = {
            "selection_order",
            "outer_fold",
            "source_family",
            "scope",
            "chunk_id_sha256",
            "job_id",
            "job_sha256",
            "rendered_message_bytes",
            "evidence_owner_count",
            "evidence_owner_ids_sha256",
            "semantic_member_count",
            "response_schema_sha256",
            "identifier_ownership_sha256",
            "response_contract_binding_sha256",
            "local_json_schema_validator_identity_sha256",
        }
        if (
            set(selected) != selected_keys
            or selected.get("selection_order") != ["rendered_message_bytes", "outer_fold", "job_id"]
            or selected.get("source_family") not in set(ACTIVE_STAGE1_CONCEPT_FAMILIES)
            or isinstance(selected.get("outer_fold"), bool)
            or not isinstance(selected.get("outer_fold"), int)
            or int(selected["outer_fold"]) < 1
            or int(selected["outer_fold"])
            not in set(
                map(
                    int,
                    (handoff_validation.get("outer_fold_assignments") or {}),
                )
            )
            or not isinstance(selected.get("scope"), str)
            or not selected["scope"]
            or not isinstance(selected.get("job_id"), str)
            or not selected["job_id"]
            or isinstance(selected.get("rendered_message_bytes"), bool)
            or not isinstance(selected.get("rendered_message_bytes"), int)
            or int(selected["rendered_message_bytes"]) < 1
            or isinstance(selected.get("evidence_owner_count"), bool)
            or not isinstance(selected.get("evidence_owner_count"), int)
            or int(selected["evidence_owner_count"]) < 1
            or isinstance(selected.get("semantic_member_count"), bool)
            or not isinstance(selected.get("semantic_member_count"), int)
            or int(selected["semantic_member_count"]) < 0
        ):
            raise ValueError(
                "direct Stage 2 canary did not select one real "
                "architecture-pure initial interpretation job"
            )
        for field in (
            "chunk_id_sha256",
            "job_sha256",
            "evidence_owner_ids_sha256",
            "response_schema_sha256",
            "identifier_ownership_sha256",
            "response_contract_binding_sha256",
            "local_json_schema_validator_identity_sha256",
        ):
            _require_sha256(
                selected.get(field),
                label=f"direct Stage 2 canary {field}",
            )
    for field in (
        "runner_identity_sha256",
        "canary_implementation_file_sha256",
    ):
        _require_sha256(body.get(field), label=f"Stage 2 canary {field}")
    if portable_direct:
        current_canary_sha256, _current_canary_size = stable_file_sha256(
            Path(canary_module.__file__).resolve(strict=True)
        )
        if body.get("canary_implementation_file_sha256") != current_canary_sha256:
            raise ValueError("direct Stage 2 canary implementation identity changed")
    _require_sha256(
        prompt_guard_identity_sha256,
        label="Stage 2 canary prompt guard identity",
    )
    for field in (
        "normalized_response_sha256",
        "raw_wire_response_sha256",
        "response_attempt_trace_sha256",
        "local_json_schema_validator_identity_sha256",
        "response_repair_policy_sha256",
        "job_cache_identity_sha256",
    ):
        _require_sha256(validation.get(field), label=f"Stage 2 canary {field}")
    outcomes = validation.get("response_attempt_outcomes")
    if (
        validation.get("validated_only_cache_enabled") is not True
        or not isinstance(outcomes, list)
        or len(outcomes) != len(transports)
        or not outcomes
        or outcomes[-1] != "validated_response"
    ):
        raise ValueError("Stage 2 canary response validation trace is invalid")
    for record in transports:
        attempts = record.get("attempts") if isinstance(record, Mapping) else None
        if not isinstance(attempts, list) or len(attempts) != 1:
            raise ValueError("Stage 2 canary performed a transport retry")
        attempt = attempts[0]
        audit = attempt.get("prompt_nontruncation_audit") if isinstance(attempt, Mapping) else None
        if (
            not isinstance(attempt, Mapping)
            or record.get("outcome") != "success"
            or record.get("runner_identity_sha256") != body.get("runner_identity_sha256")
            or record.get("request_sha256") != attempt.get("request_sha256")
            or attempt.get("endpoint") != request.get("endpoint")
            or attempt.get("model") != request.get("model_name")
            or attempt.get("response_model") != request.get("model_name")
            or attempt.get("finish_reason") != "stop"
            or attempt.get("outcome") != "success"
            or not isinstance(attempt.get("usage"), Mapping)
        ):
            raise ValueError("Stage 2 canary response metadata is invalid")
        validated_audit = _validate_prompt_nontruncation_audit(
            audit,
            guard_identity_sha256=prompt_guard_identity_sha256,
            model_context_window_tokens=int(protocol["model_context_window_tokens"]),
            permitted_generation_budgets={int(protocol["proposal_max_tokens"])},
            expected_request_sha256=attempt.get("request_sha256"),
            expected_client_path="hierarchical_discovery",
            label="Stage 2 canary response",
        )
        if attempt["usage"].get("prompt_tokens") != validated_audit["endpoint_prompt_tokens"]:
            raise ValueError("Stage 2 canary usage metadata differs from its prompt audit")
    return {
        "report_path": str(report_path),
        "report_content_sha256": wrapper["content_sha256"],
        "endpoint": body["endpoint"],
        "model": body["model"],
        "remote_response_count": len(transports),
        "transport_retries": 0,
        "maximum_schema_repairs": 1,
        "stage2_hierarchy_prompt_protocol_sha256": settings[
            "stage2_hierarchy_prompt_protocol_sha256"
        ],
        "post_extraction_causal_review_sha256": settings["post_extraction_causal_review_sha256"],
        "prompt_nontruncation_guard_identity_sha256": (prompt_guard_identity_sha256),
        "prompt_nontruncation_execution_audits_validated": len(transports),
        "finish_reason_stop_proven": True,
        "reference_only_role_neutral_stage1": portable_direct,
        "legacy_stage1_loader_invoked": False if portable_direct else None,
        "independent_stage1_refit_performed": (False if portable_direct else None),
    }


def _parse_strict_json_cell(value: Any, *, label: str) -> Any:
    if not isinstance(value, str):
        raise ValueError(f"{label} is not serialized JSON text")

    def reject_duplicates(
        pairs: Sequence[tuple[str, Any]],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, child in pairs:
            if key in output:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            output[key] = child
        return output

    try:
        return json.loads(
            value,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {token}")
            ),
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc


def _validate_complete_paged_transport_audit(
    value: Any,
    *,
    configured_model: str,
    label: str,
) -> Mapping[str, Any]:
    from ..extraction.complete_paged import (
        COMPLETE_PAGED_TRANSPORT_SCHEMA,
    )

    expected_keys = {
        "schema_version",
        "transport_retry_count",
        "schema_repair_count",
        "configured_model",
        "attempts",
        "content_sha256",
    }
    body = (
        {key: child for key, child in value.items() if key != "content_sha256"}
        if isinstance(value, Mapping)
        else None
    )
    attempts = value.get("attempts") if isinstance(value, Mapping) else None
    repair_count = value.get("schema_repair_count") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or value.get("schema_version") != COMPLETE_PAGED_TRANSPORT_SCHEMA
        or value.get("transport_retry_count") != 0
        or repair_count not in {0, 1}
        or value.get("configured_model") != configured_model
        or not isinstance(attempts, list)
        or len(attempts) != 1 + int(repair_count)
        or value.get("content_sha256") != _canonical_sha256(body)
    ):
        raise ValueError(f"{label} transport audit is invalid")
    for index, attempt in enumerate(attempts):
        if (
            not isinstance(attempt, Mapping)
            or set(attempt)
            != {
                "kind",
                "request_sha256",
                "response_sha256",
                "model",
                "finish_reason",
            }
            or attempt.get("kind") != ("initial" if index == 0 else "fixed_schema_repair")
            or attempt.get("model") != configured_model
            or attempt.get("finish_reason") != "stop"
        ):
            raise ValueError(f"{label} transport attempt is invalid")
        _require_sha256(
            attempt.get("request_sha256"),
            label=f"{label} request identity",
        )
        _require_sha256(
            attempt.get("response_sha256"),
            label=f"{label} response identity",
        )
    return dict(value)


def _validate_complete_paged_reconciliation_ledger(
    value: Any,
    *,
    leaf_responses: Sequence[tuple[str, Mapping[str, Any]]],
    final_response: Mapping[str, Any],
    prepared_text: str,
    configured_fan_in: int,
    label: str,
) -> int:
    from ..extraction.complete_paged import (
        COMPLETE_PAGED_RECONCILIATION_SCHEMA,
        CompletePageResponse,
    )

    expected_keys = {
        "schema_version",
        "leaf_count",
        "fan_in",
        "nodes",
        "root_node_id",
        "every_child_referenced_exactly_once",
        "content_sha256",
    }
    body = (
        {key: child for key, child in value.items() if key != "content_sha256"}
        if isinstance(value, Mapping)
        else None
    )
    nodes = value.get("nodes") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or value.get("schema_version") != COMPLETE_PAGED_RECONCILIATION_SCHEMA
        or value.get("leaf_count") != len(leaf_responses)
        or value.get("fan_in") != configured_fan_in
        or configured_fan_in < 2
        or not isinstance(nodes, list)
        or value.get("every_child_referenced_exactly_once") is not True
        or value.get("content_sha256") != _canonical_sha256(body)
    ):
        raise ValueError(f"{label} reconciliation ledger is invalid")

    level = [
        {"node_id": request_id, "response": dict(response)}
        for request_id, response in leaf_responses
    ]
    consumed = 0
    depth = 0
    while len(level) > 1:
        next_level: list[dict[str, Any]] = []
        for start in range(0, len(level), configured_fan_in):
            children = level[start : start + configured_fan_in]
            if consumed >= len(nodes):
                raise ValueError(f"{label} reconciliation omitted a node")
            node = nodes[consumed]
            consumed += 1
            expected_child_ids = [str(child["node_id"]) for child in children]
            if (
                not isinstance(node, Mapping)
                or set(node) != {"depth", "child_ids", "response", "node_id"}
                or node.get("depth") != depth
                or node.get("child_ids") != expected_child_ids
                or not isinstance(node.get("response"), Mapping)
            ):
                raise ValueError(f"{label} reconciliation reordered or omitted children")
            response = CompletePageResponse.validate(
                node["response"],
                text=prepared_text,
                page=None,
                authenticated_citations=True,
            ).as_dict()
            allowed_citations = {
                (
                    int(citation["start"]),
                    int(citation["end"]),
                    str(citation["text"]),
                    str(citation["sha256"]),
                )
                for child in children
                for citation in child["response"]["citations"]
            }
            observed_citations = {
                (
                    int(citation["start"]),
                    int(citation["end"]),
                    str(citation["text"]),
                    str(citation["sha256"]),
                )
                for citation in response["citations"]
            }
            node_body = {
                "depth": depth,
                "child_ids": expected_child_ids,
                "response": response,
            }
            if not observed_citations <= allowed_citations or node.get(
                "node_id"
            ) != _canonical_sha256(node_body):
                raise ValueError(f"{label} reconciliation invented evidence or changed identity")
            next_level.append(
                {
                    "node_id": node["node_id"],
                    "response": response,
                }
            )
        level = next_level
        depth += 1
    if (
        consumed != len(nodes)
        or value.get("root_node_id") != level[0]["node_id"]
        or dict(final_response) != level[0]["response"]
    ):
        raise ValueError(f"{label} reconciliation root or node coverage is invalid")
    return consumed


def _registered_reference(
    value: Any,
    *,
    registered_paths: set[Path],
    label: str,
) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} path is missing")
    raw = Path(value)
    if not raw.is_absolute() or raw.is_symlink():
        raise ValueError(f"{label} path is not one real canonical absolute path")
    resolved = raw.resolve(strict=True)
    if raw != resolved or resolved not in registered_paths:
        raise ValueError(f"{label} is not terminally registered")
    return resolved


def _validate_direct_stage2_one_shot_attestation(
    *,
    request: Mapping[str, Any],
    inference_paths: set[Path],
    handoff_validation: Mapping[str, Any],
    canary_validation: Mapping[str, Any],
    request_contract: tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, Any],
    ],
) -> Mapping[str, Any]:
    """Reopen the closed reference-only Stage 2 terminal inventory."""

    from .production_role_neutral_stage2_handoff import (
        ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
    )
    from .production_stage1_hierarchy_one_shot import (
        PRODUCTION_ROLE_NEUTRAL_STAGE2_ONE_SHOT_ATTESTATION_FILENAME,
        PRODUCTION_ROLE_NEUTRAL_STAGE2_ONE_SHOT_ATTESTATION_SCHEMA,
    )

    attestation_path = _unique_named(
        inference_paths,
        PRODUCTION_ROLE_NEUTRAL_STAGE2_ONE_SHOT_ATTESTATION_FILENAME,
        label="portable Stage 2 inference phase",
    )
    attestation = _read_json_object(
        attestation_path,
        label="portable Stage 2 one-shot attestation",
    )
    body = {key: value for key, value in attestation.items() if key != "content_sha256"}
    expected_keys = {
        "schema_version",
        "status",
        "handoff_kind",
        "stage1_reference_handoff",
        "remote_runtime_identity",
        "stage2_hierarchy_prompt_protocol",
        "post_extraction_causal_review",
        "hierarchical_batch_result",
        "folds",
        "fold_count",
        "runner_input_manifest",
        "prepared_cohort",
        "complete_paged_extraction_ledgers",
        "immutable_run_manifest",
        "frozen_predictions",
        "phase_artifact_inventory",
        "one_shot_implementation_sha256",
        "legacy_stage1_loader_invoked",
        "tfidf_handoff_loader_invoked",
        "independent_stage1_refit_performed",
        "structured_or_nonforest_fallback_used",
        "outer_heldout_labels_used_during_discovery_or_review",
        "oracle_source_opened",
        "global_release_certified",
        "content_sha256",
    }
    if (
        set(attestation) != expected_keys
        or attestation.get("content_sha256") != _canonical_sha256(body)
        or body.get("schema_version") != PRODUCTION_ROLE_NEUTRAL_STAGE2_ONE_SHOT_ATTESTATION_SCHEMA
        or body.get("status") != "completed"
        or body.get("handoff_kind") != ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
        or body.get("legacy_stage1_loader_invoked") is not False
        or body.get("tfidf_handoff_loader_invoked") is not False
        or body.get("independent_stage1_refit_performed") is not False
        or body.get("structured_or_nonforest_fallback_used") is not False
        or body.get("outer_heldout_labels_used_during_discovery_or_review") is not False
        or body.get("oracle_source_opened") is not False
        or body.get("global_release_certified") is not False
    ):
        raise ValueError("portable Stage 2 one-shot attestation is invalid")

    reference = body.get("stage1_reference_handoff")
    expected_reference_keys = {
        "manifest_path",
        "scientific_content_sha256",
        "bundle_sha256",
        "source_execution_content_sha256",
        "provider_identity_sha256",
        "runtime_binding_content_sha256",
        "prepared_projection_binding_content_sha256",
        "prepared_cohort_artifact_sha256",
        "row_map_sha256",
        "direct_numerical_bank_manifest_content_sha256",
        "offline_handoff_validation_complete",
    }
    if (
        not isinstance(reference, Mapping)
        or set(reference) != expected_reference_keys
        or Path(str(reference.get("manifest_path", ""))).resolve(strict=True)
        != Path(str(handoff_validation["bundle_manifest_path"]))
        or reference.get("scientific_content_sha256")
        != handoff_validation.get("handoff_scientific_content_sha256")
        or reference.get("bundle_sha256") != handoff_validation.get("bundle_sha256")
        or reference.get("source_execution_content_sha256")
        != handoff_validation.get("source_role_neutral_execution_content_sha256")
        or reference.get("provider_identity_sha256")
        != handoff_validation.get("stage2_provider_identity_sha256")
        or reference.get("row_map_sha256") != handoff_validation.get("row_map_sha256")
        or reference.get("direct_numerical_bank_manifest_content_sha256")
        != handoff_validation.get("direct_numerical_bank_content_sha256")
        or reference.get("offline_handoff_validation_complete") is not True
    ):
        raise ValueError("portable Stage 2 attestation belongs to another Stage 1 graph")
    for field in (
        "runtime_binding_content_sha256",
        "prepared_projection_binding_content_sha256",
        "prepared_cohort_artifact_sha256",
    ):
        _require_sha256(
            reference.get(field),
            label=f"portable Stage 2 {field}",
        )

    (
        configured_protocol,
        configured_causal_review,
        request_tokenizer_identity,
    ) = request_contract
    if (
        body.get("stage2_hierarchy_prompt_protocol") != configured_protocol
        or body.get("post_extraction_causal_review") != configured_causal_review
    ):
        raise ValueError("portable Stage 2 scientific prompt/review configuration changed")
    runtime = body.get("remote_runtime_identity")
    runtime_keys = {
        "endpoint_urls",
        "model",
        "hierarchical_runner_identity_sha256",
        "prompt_nontruncation_guard",
        "prompt_nontruncation_execution_audit",
        "required_finish_reason",
        "endpoint_pool_or_fallback_allowed",
        "model_substitution_allowed",
    }
    guard = (
        runtime.get("prompt_nontruncation_guard")
        if isinstance(
            runtime,
            Mapping,
        )
        else None
    )
    validated_guard = _validate_prompt_guard_identity(
        guard,
        request_tokenizer_identity=request_tokenizer_identity,
        model_name=request.get("model_name"),
        model_context_window_tokens=int(configured_protocol["model_context_window_tokens"]),
        label="portable Stage 2 one-shot",
    )
    prompt_audit = _validate_prompt_nontruncation_execution_audit(
        (
            runtime.get("prompt_nontruncation_execution_audit")
            if isinstance(runtime, Mapping)
            else None
        ),
        guard_identity_sha256=validated_guard["identity_sha256"],
        protocol=configured_protocol,
    )
    prompt_counts = prompt_audit["record_counts_by_client_path"]
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != runtime_keys
        or runtime.get("endpoint_urls") != [request.get("endpoint")]
        or runtime.get("model") != {"name": request.get("model_name")}
        or runtime.get("required_finish_reason") != "stop"
        or runtime.get("endpoint_pool_or_fallback_allowed") is not False
        or runtime.get("model_substitution_allowed") is not False
        or any(int(count) < 1 for count in prompt_counts.values())
        or canary_validation.get("prompt_nontruncation_guard_identity_sha256")
        != validated_guard["identity_sha256"]
        or canary_validation.get("stage2_hierarchy_prompt_protocol_sha256")
        != _canonical_sha256(configured_protocol)
        or canary_validation.get("post_extraction_causal_review_sha256")
        != _canonical_sha256(configured_causal_review)
        or canary_validation.get("reference_only_role_neutral_stage1") is not True
    ):
        raise ValueError("portable Stage 2 remote runtime identity is invalid")
    _require_sha256(
        runtime.get("hierarchical_runner_identity_sha256"),
        label="portable Stage 2 hierarchy runner identity",
    )

    def registered_file(
        registration: Any,
        *,
        expected_keys: set[str],
        label: str,
    ) -> tuple[Path, str, int]:
        if not isinstance(registration, Mapping) or set(registration) != expected_keys:
            raise ValueError(f"{label} registration is not closed")
        path = _registered_reference(
            registration.get("path"),
            registered_paths=inference_paths,
            label=label,
        )
        digest, size = stable_file_sha256(path)
        if registration.get("sha256") != digest or registration.get("size", size) != size:
            raise ValueError(f"{label} bytes changed")
        return path, digest, size

    prepared_registration = body.get("prepared_cohort")
    (
        prepared_cohort_path,
        _prepared_cohort_sha,
        _prepared_cohort_size,
    ) = registered_file(
        prepared_registration,
        expected_keys={
            "path",
            "size",
            "sha256",
            "row_count",
            "text_column",
        },
        label="portable Stage 2 prepared cohort",
    )
    import pandas as pd

    prepared_frame = pd.read_parquet(prepared_cohort_path)
    text_column = request.get("text_column")
    if (
        prepared_registration.get("sha256") != reference["prepared_cohort_artifact_sha256"]
        or prepared_registration.get("row_count") != len(prepared_frame)
        or prepared_registration.get("text_column") != text_column
        or not isinstance(text_column, str)
        or text_column not in prepared_frame.columns
        or "_oci_row_id" not in prepared_frame.columns
        or prepared_frame["_oci_row_id"].duplicated().any()
    ):
        raise ValueError("portable Stage 2 prepared cohort registration is invalid")
    prepared_row_ids = [int(value) for value in prepared_frame["_oci_row_id"].tolist()]
    if prepared_row_ids != list(range(len(prepared_row_ids))) or any(
        not isinstance(value, str) for value in prepared_frame[text_column].tolist()
    ):
        raise ValueError("portable Stage 2 prepared cohort row/text projection changed")
    prepared_by_row = prepared_frame.set_index("_oci_row_id", drop=False)

    from ..extraction.complete_paged import (
        CompleteFeatureContract,
        CompletePageResponse,
        CompletePagingGeometry,
        build_complete_page_prompt,
        build_complete_paged_coverage_ledger,
        plan_complete_paged_requests,
    )

    ledger_registrations = body.get("complete_paged_extraction_ledgers")
    if not isinstance(ledger_registrations, list) or not ledger_registrations:
        raise ValueError("portable Stage 2 omitted complete-paged extraction ledgers")
    ledger_inventory: list[dict[str, Any]] = []
    ledger_manifest_paths: list[Path] = []
    ledger_payload_paths: list[Path] = []
    complete_paged_planned_requests = 0
    complete_paged_remote_requests = 0
    complete_paged_transport_request_sha256s: list[str] = []
    expected_geometry = {
        "core_chars": int(request["complete_page_core_chars"]),
        "context_chars": int(request["complete_page_context_chars"]),
        "max_page_chars": int(request["complete_page_max_chars"]),
    }
    expected_fan_in = int(request["complete_reconciliation_fan_in"])
    page_columns = [
        "request_index",
        "request_id",
        "patient_local_id",
        "oci_row_id",
        "note_sha256",
        "feature_name",
        "feature_contract_sha256",
        "page_index",
        "core_start",
        "core_end",
        "context_start",
        "context_end",
        "page_text_sha256",
        "core_sha256",
        "prompt_sha256",
        "prompt",
        "normalized_response_json",
        "normalized_response_sha256",
        "transport_audit_json",
        "transport_audit_sha256",
    ]
    reconciliation_columns = [
        "patient_local_id",
        "oci_row_id",
        "final_response_json",
        "final_response_sha256",
        "reconciliation_ledger_json",
        "reconciliation_ledger_sha256",
        "transport_audits_json",
        "transport_audits_sha256",
    ]
    for invocation_index, ledger_registration in enumerate(ledger_registrations):
        if (
            not isinstance(ledger_registration, Mapping)
            or set(ledger_registration) != {"invocation_index", "manifest", "payloads"}
            or ledger_registration.get("invocation_index") != invocation_index
        ):
            raise ValueError("portable Stage 2 extraction ledger order is invalid")
        manifest_registration = ledger_registration["manifest"]
        manifest_path, _manifest_sha, _manifest_size = registered_file(
            manifest_registration,
            expected_keys={
                "path",
                "size",
                "sha256",
                "content_sha256",
            },
            label=(f"portable Stage 2 extraction ledger {invocation_index} " "manifest"),
        )
        manifest = _read_json_object(
            manifest_path,
            label=(f"portable Stage 2 extraction ledger {invocation_index} " "manifest"),
        )
        manifest_body = {key: value for key, value in manifest.items() if key != "content_sha256"}
        manifest_keys = {
            "schema_version",
            "feature_contract",
            "feature_contract_sha256",
            "configured_model",
            "geometry",
            "geometry_sha256",
            "ordered_oci_row_ids",
            "ordered_oci_row_ids_sha256",
            "ordered_note_sha256",
            "request_plan_content_sha256",
            "coverage_content_sha256",
            "planned_page_request_count",
            "completed_page_request_count",
            "patient_count",
            "page_table",
            "reconciliation_table",
            "one_feature_contract_per_page_request",
            "configured_reconciliation_fan_in",
            "all_pages_reconciled_with_configured_fan_in",
            "transport_retries",
            "maximum_schema_repairs_per_request",
            "exact_prompts_persisted",
            "canonical_row_ids_persisted",
            "raw_note_copies_persisted",
            "content_sha256",
        }
        from .production_stage1_hierarchy_one_shot import (
            PRODUCTION_COMPLETE_PAGED_EXTRACTION_LEDGER_SCHEMA,
        )

        if (
            set(manifest) != manifest_keys
            or manifest.get("schema_version") != PRODUCTION_COMPLETE_PAGED_EXTRACTION_LEDGER_SCHEMA
            or manifest.get("content_sha256") != _canonical_sha256(manifest_body)
            or manifest_registration.get("content_sha256") != manifest["content_sha256"]
            or manifest.get("configured_model") != request.get("model_name")
            or manifest.get("geometry") != expected_geometry
            or manifest.get("geometry_sha256") != _canonical_sha256(expected_geometry)
            or manifest.get("configured_reconciliation_fan_in") != expected_fan_in
            or manifest.get("one_feature_contract_per_page_request") is not True
            or manifest.get("all_pages_reconciled_with_configured_fan_in") is not True
            or manifest.get("transport_retries") != 0
            or manifest.get("maximum_schema_repairs_per_request") != 1
            or manifest.get("exact_prompts_persisted") is not True
            or manifest.get("canonical_row_ids_persisted") is not True
            or manifest.get("raw_note_copies_persisted") is not False
        ):
            raise ValueError("portable Stage 2 complete-paged ledger manifest is invalid")
        feature_value = manifest.get("feature_contract")
        if (
            not isinstance(feature_value, Mapping)
            or set(feature_value)
            != {
                "name",
                "value_type",
                "description",
                "temporal_rule",
                "aggregation_rule",
                "categories",
            }
            or not isinstance(feature_value.get("categories"), list)
        ):
            raise ValueError("portable Stage 2 complete-paged feature contract is invalid")
        feature = CompleteFeatureContract(
            name=str(feature_value["name"]),
            value_type=str(feature_value["value_type"]),
            description=str(feature_value["description"]),
            temporal_rule=str(feature_value["temporal_rule"]),
            aggregation_rule=str(feature_value["aggregation_rule"]),
            categories=tuple(map(str, feature_value["categories"])),
        )
        if manifest.get("feature_contract_sha256") != feature.contract_sha256:
            raise ValueError("portable Stage 2 complete-paged feature identity changed")
        ordered_ids = manifest.get("ordered_oci_row_ids")
        if (
            not isinstance(ordered_ids, list)
            or not ordered_ids
            or any(isinstance(value, bool) or not isinstance(value, int) for value in ordered_ids)
            or len(ordered_ids) != len(set(ordered_ids))
            or manifest.get("ordered_oci_row_ids_sha256") != _canonical_sha256(ordered_ids)
            or not set(ordered_ids) <= set(prepared_row_ids)
            or manifest.get("patient_count") != len(ordered_ids)
        ):
            raise ValueError("portable Stage 2 complete-paged row scope is invalid")
        texts = [str(prepared_by_row.loc[row_id, text_column]) for row_id in ordered_ids]
        note_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        if manifest.get("ordered_note_sha256") != note_hashes:
            raise ValueError("portable Stage 2 complete-paged prepared notes changed")
        geometry = CompletePagingGeometry(**expected_geometry)
        notes = {str(index): text for index, text in enumerate(texts)}
        request_plan = plan_complete_paged_requests(
            notes,
            (feature,),
            geometry=geometry,
        )
        request_plan_value = request_plan.as_dict()
        planned = len(request_plan.requests)
        if (
            manifest.get("request_plan_content_sha256") != request_plan_value["content_sha256"]
            or manifest.get("planned_page_request_count") != planned
            or manifest.get("completed_page_request_count") != planned
        ):
            raise ValueError("portable Stage 2 complete-paged request plan is incomplete")

        payload_registrations = ledger_registration.get("payloads")
        if not isinstance(payload_registrations, list) or [
            row.get("kind") for row in payload_registrations
        ] != ["page_table", "reconciliation_table"]:
            raise ValueError("portable Stage 2 complete-paged payload inventory is invalid")
        payload_paths: dict[str, Path] = {}
        for payload_registration in payload_registrations:
            payload_kind = str(payload_registration["kind"])
            payload_path, payload_sha, payload_size = registered_file(
                payload_registration,
                expected_keys={"kind", "path", "size", "sha256"},
                label=(f"portable Stage 2 extraction ledger {invocation_index} " f"{payload_kind}"),
            )
            manifest_payload = manifest.get(payload_kind)
            if (
                not isinstance(manifest_payload, Mapping)
                or set(manifest_payload) != {"relative_path", "row_count", "size", "sha256"}
                or manifest_payload.get("relative_path") != payload_path.name
                or payload_path.parent != manifest_path.parent
                or manifest_payload.get("sha256") != payload_sha
                or manifest_payload.get("size") != payload_size
            ):
                raise ValueError("portable Stage 2 complete-paged payload binding changed")
            payload_paths[payload_kind] = payload_path

        page_frame = pd.read_parquet(payload_paths["page_table"])
        if (
            list(page_frame.columns) != page_columns
            or len(page_frame) != planned
            or manifest["page_table"].get("row_count") != planned
        ):
            raise ValueError("portable Stage 2 complete-paged page table is incomplete")
        normalized_responses: dict[str, Mapping[str, Any]] = {}
        for request_index, (request_row, request_spec) in enumerate(
            zip(
                page_frame.to_dict(orient="records"),
                request_plan.requests,
                strict=True,
            )
        ):
            patient_index = int(request_spec.patient_id)
            prompt = build_complete_page_prompt(
                texts[patient_index],
                page=request_spec.page,
                feature=feature,
                geometry=geometry,
            )
            expected_page_fields = {
                "request_index": request_index,
                "request_id": request_spec.request_id,
                "patient_local_id": request_spec.patient_id,
                "oci_row_id": ordered_ids[patient_index],
                "note_sha256": request_spec.note_sha256,
                "feature_name": request_spec.feature_name,
                "feature_contract_sha256": (request_spec.feature_contract_sha256),
                "page_index": request_spec.page.page_index,
                "core_start": request_spec.page.core_start,
                "core_end": request_spec.page.core_end,
                "context_start": request_spec.page.context_start,
                "context_end": request_spec.page.context_end,
                "page_text_sha256": request_spec.page.text_sha256,
                "core_sha256": request_spec.page.core_sha256,
                "prompt_sha256": request_spec.prompt_sha256,
                "prompt": prompt,
            }
            if any(
                request_row.get(key) != expected for key, expected in expected_page_fields.items()
            ):
                raise ValueError("portable Stage 2 complete-paged request/prompt changed")
            response = _parse_strict_json_cell(
                request_row["normalized_response_json"],
                label=(
                    f"complete-paged invocation {invocation_index} " f"response {request_index}"
                ),
            )
            normalized = CompletePageResponse.validate(
                response,
                text=texts[patient_index],
                page=request_spec.page,
                authenticated_citations=True,
            ).as_dict()
            if response != normalized or request_row.get(
                "normalized_response_sha256"
            ) != _canonical_sha256(normalized):
                raise ValueError("portable Stage 2 complete-paged response changed")
            transport = _parse_strict_json_cell(
                request_row["transport_audit_json"],
                label=(
                    f"complete-paged invocation {invocation_index} " f"transport {request_index}"
                ),
            )
            validated_transport = _validate_complete_paged_transport_audit(
                transport,
                configured_model=str(request["model_name"]),
                label=(f"complete-paged invocation {invocation_index} " f"page {request_index}"),
            )
            if request_row.get("transport_audit_sha256") != _canonical_sha256(validated_transport):
                raise ValueError("portable Stage 2 complete-paged transport changed")
            complete_paged_remote_requests += len(validated_transport["attempts"])
            complete_paged_transport_request_sha256s.extend(
                str(attempt["request_sha256"])
                for attempt in validated_transport["attempts"]
            )
            normalized_responses[request_spec.request_id] = normalized
        coverage = build_complete_paged_coverage_ledger(
            request_plan,
            normalized_responses,
        )
        if manifest.get("coverage_content_sha256") != coverage["content_sha256"]:
            raise ValueError("portable Stage 2 complete-paged coverage proof changed")

        reconciliation_frame = pd.read_parquet(payload_paths["reconciliation_table"])
        if (
            list(reconciliation_frame.columns) != reconciliation_columns
            or len(reconciliation_frame) != len(ordered_ids)
            or manifest["reconciliation_table"].get("row_count") != len(ordered_ids)
        ):
            raise ValueError("portable Stage 2 reconciliation table is incomplete")
        for patient_index, reconciliation_row in enumerate(
            reconciliation_frame.to_dict(orient="records")
        ):
            patient_id = str(patient_index)
            if (
                reconciliation_row.get("patient_local_id") != patient_id
                or reconciliation_row.get("oci_row_id") != ordered_ids[patient_index]
            ):
                raise ValueError("portable Stage 2 reconciliation row order changed")
            final_response_value = _parse_strict_json_cell(
                reconciliation_row["final_response_json"],
                label=(
                    f"complete-paged invocation {invocation_index} "
                    f"patient {patient_id} final response"
                ),
            )
            final_response = CompletePageResponse.validate(
                final_response_value,
                text=texts[patient_index],
                page=None,
                authenticated_citations=True,
            ).as_dict()
            if final_response_value != final_response or reconciliation_row.get(
                "final_response_sha256"
            ) != _canonical_sha256(final_response):
                raise ValueError("portable Stage 2 final extraction response changed")
            reconciliation = _parse_strict_json_cell(
                reconciliation_row["reconciliation_ledger_json"],
                label=(
                    f"complete-paged invocation {invocation_index} "
                    f"patient {patient_id} reconciliation"
                ),
            )
            if reconciliation_row.get("reconciliation_ledger_sha256") != _canonical_sha256(
                reconciliation
            ):
                raise ValueError("portable Stage 2 reconciliation ledger changed")
            patient_requests = [
                request_spec
                for request_spec in request_plan.requests
                if request_spec.patient_id == patient_id
            ]
            leaf_responses = [
                (
                    request_spec.request_id,
                    normalized_responses[request_spec.request_id],
                )
                for request_spec in patient_requests
            ]
            node_count = _validate_complete_paged_reconciliation_ledger(
                reconciliation,
                leaf_responses=leaf_responses,
                final_response=final_response,
                prepared_text=texts[patient_index],
                configured_fan_in=expected_fan_in,
                label=(f"complete-paged invocation {invocation_index} " f"patient {patient_id}"),
            )
            reconciliation_transports = _parse_strict_json_cell(
                reconciliation_row["transport_audits_json"],
                label=(
                    f"complete-paged invocation {invocation_index} "
                    f"patient {patient_id} reconciliation transports"
                ),
            )
            if (
                not isinstance(reconciliation_transports, list)
                or len(reconciliation_transports) != node_count
                or reconciliation_row.get("transport_audits_sha256")
                != _canonical_sha256(reconciliation_transports)
            ):
                raise ValueError("portable Stage 2 reconciliation transport coverage changed")
            for transport_index, transport in enumerate(reconciliation_transports):
                validated_transport = _validate_complete_paged_transport_audit(
                    transport,
                    configured_model=str(request["model_name"]),
                    label=(
                        f"complete-paged invocation "
                        f"{invocation_index} patient {patient_id} "
                        f"reconciliation transport {transport_index}"
                    ),
                )
                complete_paged_remote_requests += len(validated_transport["attempts"])
                complete_paged_transport_request_sha256s.extend(
                    str(attempt["request_sha256"])
                    for attempt in validated_transport["attempts"]
                )
        complete_paged_planned_requests += planned
        ledger_manifest_paths.append(manifest_path)
        ledger_payload_paths.extend(payload_paths.values())
        ledger_inventory.extend(
            [
                {
                    "kind": "complete_paged_page_table",
                    "invocation_index": invocation_index,
                    **{key: payload_registrations[0][key] for key in ("path", "size", "sha256")},
                },
                {
                    "kind": "complete_paged_reconciliation_table",
                    "invocation_index": invocation_index,
                    **{key: payload_registrations[1][key] for key in ("path", "size", "sha256")},
                },
                {
                    "kind": "complete_paged_ledger_manifest",
                    "invocation_index": invocation_index,
                    **dict(manifest_registration),
                },
            ]
        )
    explicit_prompt_request_sha256s = [
        str(record["request_sha256"])
        for record in prompt_audit["records"]
        if record["client_path"] == "explicit_feature_extraction"
    ]
    if (
        complete_paged_planned_requests < len(prepared_frame)
        or complete_paged_remote_requests
        != int(prompt_counts["explicit_feature_extraction"])
        or sorted(complete_paged_transport_request_sha256s)
        != sorted(explicit_prompt_request_sha256s)
    ):
        raise ValueError(
            "portable Stage 2 complete-paged planned/remote request ledgers "
            "do not equal the prompt execution audit by count and request identity"
        )

    batch_registration = body.get("hierarchical_batch_result")
    batch_path, _batch_sha, _batch_size = registered_file(
        batch_registration,
        expected_keys={
            "path",
            "size",
            "sha256",
            "content_sha256",
            "all_fold_discovery_completed_before_per_fold_modeling",
        },
        label="portable Stage 2 hierarchical batch result",
    )
    batch_wrapper = _read_json_object(
        batch_path,
        label="portable Stage 2 hierarchical batch result",
    )
    batch_body = _validate_content_hashed_body(
        batch_wrapper,
        schema=_HIERARCHICAL_BATCH_RESULT_SCHEMA,
        label="portable Stage 2 hierarchical batch result",
    )
    outer_fold_count = int(request["outer_folds"])
    ordered_batch_rows = batch_body.get("ordered_fold_results")
    if (
        batch_registration.get("content_sha256") != batch_wrapper.get("content_sha256")
        or batch_registration.get("all_fold_discovery_completed_before_per_fold_modeling")
        is not True
        or batch_body.get("all_fold_discovery_completed_before_per_fold_modeling") is not True
        or not isinstance(ordered_batch_rows, list)
        or len(ordered_batch_rows) != outer_fold_count
        or any(not isinstance(row, Mapping) for row in ordered_batch_rows)
        or [row.get("outer_fold") for row in ordered_batch_rows]
        != list(range(1, outer_fold_count + 1))
    ):
        raise ValueError("portable Stage 2 hierarchical batch coverage is incomplete")

    input_registration = body.get("runner_input_manifest")
    input_path, _input_sha, _input_size = registered_file(
        input_registration,
        expected_keys={"path", "size", "sha256", "content_sha256"},
        label="portable Stage 2 runner input manifest",
    )
    input_wrapper = _read_json_object(
        input_path,
        label="portable Stage 2 runner input manifest",
    )
    input_body = _validate_content_hashed_body(
        input_wrapper,
        schema="all_evidence_fusion_outer_runner_v20",
        label="portable Stage 2 runner input manifest",
    )
    input_source = _validate_direct_stage1_source_identity(
        input_body.get("stage1_reference_source"),
        handoff_validation=handoff_validation,
    )
    if (
        input_registration.get("content_sha256") != input_wrapper.get("content_sha256")
        or input_body.get("legacy_handoff_path") is not None
        or input_body.get("legacy_handoff_sha256") is not None
        or input_body.get("tfidf_handoff_path") is not None
        or input_body.get("tfidf_handoff_sha256") is not None
        or input_source["runtime_binding_content_sha256"]
        != reference["runtime_binding_content_sha256"]
        or input_source["prepared_projection_binding_content_sha256"]
        != reference["prepared_projection_binding_content_sha256"]
        or input_source["prepared_cohort_artifact_sha256"]
        != reference["prepared_cohort_artifact_sha256"]
    ):
        raise ValueError(
            "portable Stage 2 runner input contains a legacy or substituted " "Stage 1 source"
        )

    folds = body.get("folds")
    assignments = handoff_validation.get("outer_fold_assignments")
    if (
        not isinstance(folds, list)
        or not isinstance(assignments, Mapping)
        or len(folds) != outer_fold_count
        or body.get("fold_count") != outer_fold_count
    ):
        raise ValueError("portable Stage 2 attestation fold inventory is incomplete")
    fold_manifest_paths: list[Path] = []
    fold_prediction_paths: list[Path] = []
    expected_inventory: list[dict[str, Any]] = []
    for expected_fold, row in enumerate(folds, start=1):
        expected_row_keys = {
            "outer_fold",
            "fit_row_count",
            "heldout_row_count",
            "manifest",
            "prediction",
            "strict_forest_receipt_content_sha256",
        }
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_row_keys
            or row.get("outer_fold") != expected_fold
            or row.get("fit_row_count") != len(assignments[expected_fold]["fit_row_ids"])
            or row.get("heldout_row_count") != len(assignments[expected_fold]["heldout_row_ids"])
        ):
            raise ValueError("portable Stage 2 attestation fold row is invalid")
        manifest_path, _manifest_sha, _manifest_size = registered_file(
            row.get("manifest"),
            expected_keys={"path", "size", "sha256", "content_sha256"},
            label=f"portable Stage 2 fold {expected_fold} manifest",
        )
        prediction_path, _prediction_sha, _prediction_size = registered_file(
            row.get("prediction"),
            expected_keys={"path", "size", "sha256"},
            label=f"portable Stage 2 fold {expected_fold} prediction",
        )
        manifest_wrapper = _read_json_object(
            manifest_path,
            label=f"portable Stage 2 fold {expected_fold} manifest",
        )
        fold_body = _validate_content_hashed_body(
            manifest_wrapper,
            schema="all_evidence_fusion_frozen_fold_v20",
            label=f"portable Stage 2 fold {expected_fold} manifest",
        )
        receipt = (
            fold_body.get("final_ite_estimator", {}).get("forest_receipt")
            if isinstance(fold_body.get("final_ite_estimator"), Mapping)
            else None
        )
        if (
            row["manifest"].get("content_sha256") != manifest_wrapper.get("content_sha256")
            or fold_body.get("prediction_path") != str(prediction_path)
            or fold_body.get("prediction_sha256") != row["prediction"].get("sha256")
            or not isinstance(receipt, Mapping)
            or row.get("strict_forest_receipt_content_sha256") != receipt.get("content_sha256")
        ):
            raise ValueError("portable Stage 2 attestation fold references changed")
        fold_manifest_paths.append(manifest_path)
        fold_prediction_paths.append(prediction_path)
        expected_inventory.extend(
            (
                {
                    "kind": "fold_manifest",
                    "outer_fold": expected_fold,
                    **dict(row["manifest"]),
                },
                {
                    "kind": "fold_prediction",
                    "outer_fold": expected_fold,
                    **dict(row["prediction"]),
                },
            )
        )

    run_registration = body.get("immutable_run_manifest")
    run_path, _run_sha, run_size = registered_file(
        run_registration,
        expected_keys={"path", "sha256", "content_sha256"},
        label="portable Stage 2 immutable run manifest",
    )
    run_wrapper = _read_json_object(
        run_path,
        label="portable Stage 2 immutable run manifest",
    )
    _validate_content_hashed_body(
        run_wrapper,
        schema=_STAGE2_RUN_MANIFEST_SCHEMA,
        label="portable Stage 2 immutable run manifest",
    )
    if run_registration.get("content_sha256") != run_wrapper.get("content_sha256"):
        raise ValueError("portable Stage 2 run-manifest content identity changed")
    prediction_registration = body.get("frozen_predictions")
    prediction_path, _prediction_sha, prediction_size = registered_file(
        prediction_registration,
        expected_keys={
            "path",
            "size",
            "sha256",
            "columns",
            "row_count",
            "probability_difference_bounds",
            "probability_difference_validation_tolerance",
            "probability_difference_bounds_validated",
            "values_clipped",
        },
        label="portable Stage 2 combined frozen CATE",
    )
    import numpy as np
    import pandas as pd

    prediction_frame = pd.read_parquet(prediction_path)
    _validate_prediction_frame(
        prediction_frame,
        label="portable Stage 2 combined frozen CATE",
        prediction_columns=_DIRECT_CATE_PREDICTION_COLUMNS,
    )
    tolerance = float(64 * np.finfo(np.float64).eps)
    if (
        prediction_registration.get("columns") != list(_DIRECT_CATE_PREDICTION_COLUMNS)
        or prediction_registration.get("row_count") != len(prediction_frame)
        or prediction_registration.get("probability_difference_bounds") != [-1.0, 1.0]
        or prediction_registration.get("probability_difference_validation_tolerance") != tolerance
        or prediction_registration.get("probability_difference_bounds_validated") is not True
        or prediction_registration.get("values_clipped") is not False
    ):
        raise ValueError("portable Stage 2 frozen CATE estimand contract changed")

    expected_inventory.extend(ledger_inventory)
    expected_inventory.append(
        {
            "kind": "prepared_cohort",
            **{key: prepared_registration[key] for key in ("path", "size", "sha256")},
        }
    )
    expected_inventory.extend(
        (
            {
                "kind": "hierarchical_batch_result",
                **{
                    key: batch_registration[key]
                    for key in ("path", "size", "sha256", "content_sha256")
                },
            },
            {
                "kind": "runner_input_manifest",
                **dict(input_registration),
            },
            {
                "kind": "combined_prediction",
                **{key: prediction_registration[key] for key in ("path", "size", "sha256")},
            },
            {
                "kind": "run_manifest",
                "path": str(run_path),
                "size": run_size,
                "sha256": run_registration["sha256"],
                "content_sha256": run_registration["content_sha256"],
            },
        )
    )
    if body.get("phase_artifact_inventory") != expected_inventory or inference_paths != {
        attestation_path,
        prepared_cohort_path,
        batch_path,
        input_path,
        run_path,
        prediction_path,
        *ledger_manifest_paths,
        *ledger_payload_paths,
        *fold_manifest_paths,
        *fold_prediction_paths,
    }:
        raise ValueError(
            "portable Stage 2 phase artifact inventory is incomplete or "
            "contains unrelated artifacts"
        )
    implementation_sha256 = _require_sha256(
        body.get("one_shot_implementation_sha256"),
        label="portable Stage 2 one-shot implementation",
    )
    from . import production_stage1_hierarchy_one_shot as one_shot_module

    current_implementation_sha256, _size = stable_file_sha256(
        Path(one_shot_module.__file__).resolve(strict=True)
    )
    if implementation_sha256 != current_implementation_sha256:
        raise ValueError("portable Stage 2 one-shot implementation changed")
    return {
        "attestation_path": str(attestation_path),
        "attestation_content_sha256": attestation["content_sha256"],
        "stage2_hierarchy_prompt_protocol_sha256": _canonical_sha256(configured_protocol),
        "post_extraction_causal_review_sha256": _canonical_sha256(configured_causal_review),
        "prompt_nontruncation_guard_identity_sha256": validated_guard["identity_sha256"],
        "prompt_nontruncation_execution_audit_sha256": prompt_audit["audit_sha256"],
        "prompt_nontruncation_execution_record_count": prompt_audit["record_count"],
        "hierarchical_batch_result_path": str(batch_path),
        "runner_input_manifest_path": str(input_path),
        "run_manifest_path": str(run_path),
        "prediction_path": str(prediction_path),
        "fold_manifest_paths": [str(path) for path in fold_manifest_paths],
        "fold_prediction_paths": [str(path) for path in fold_prediction_paths],
        "prepared_cohort_path": str(prepared_cohort_path),
        "complete_paged_ledger_manifest_paths": [str(path) for path in ledger_manifest_paths],
        "complete_paged_ledger_payload_paths": [str(path) for path in ledger_payload_paths],
        "complete_paged_planned_page_request_count": (complete_paged_planned_requests),
        "complete_paged_remote_request_count": (complete_paged_remote_requests),
        "complete_paged_exact_prompt_and_citation_validation": True,
        "direct_source_binding": dict(reference),
        "legacy_stage1_loader_invoked": False,
        "tfidf_handoff_loader_invoked": False,
        "independent_stage1_refit_performed": False,
        "structured_or_nonforest_fallback_used": False,
        "oracle_source_opened": False,
        "global_release_certified": False,
    }


def validate_real_stage2_one_shot_attestation(
    *,
    request: Mapping[str, Any],
    phase_records: Sequence[Mapping[str, Any]],
    handoff_validation: Mapping[str, Any],
    canary_validation: Mapping[str, Any],
    _validated_phase_map: Mapping[str, Mapping[str, Any]] | None = None,
    _validated_path_inventory: Mapping[str, set[Path]] | None = None,
    _validated_request_contract: (
        tuple[dict[str, Any], dict[str, float], dict[str, Any]] | None
    ) = None,
) -> Mapping[str, Any]:
    """Authenticate the sealed one-shot execution and every referenced byte."""

    if handoff_validation.get("real_stage1_handoff_detected") is not True:
        raise ValueError("Stage 2 one-shot lacks a validated Stage 1 handoff")
    by_phase = (
        dict(_validated_phase_map)
        if _validated_phase_map is not None
        else _phase_map(phase_records)
    )
    paths_by_phase = (
        dict(_validated_path_inventory)
        if _validated_path_inventory is not None
        else {phase: _artifact_paths(record) for phase, record in by_phase.items()}
    )
    inference_paths = paths_by_phase.get("stage2_inference", set())
    from .production_role_neutral_stage2_handoff import (
        ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
    )

    if handoff_validation.get("handoff_kind") == ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND:
        request_contract = (
            _validated_stage2_request_contract(request)
            if _validated_request_contract is None
            else _validated_request_contract
        )
        return _validate_direct_stage2_one_shot_attestation(
            request=request,
            inference_paths=inference_paths,
            handoff_validation=handoff_validation,
            canary_validation=canary_validation,
            request_contract=request_contract,
        )
    attestation_path = _unique_named(
        inference_paths,
        "production_stage1_hierarchy_one_shot_result.json",
        label="Stage 2 inference phase",
    )
    attestation = _read_json_object(
        attestation_path,
        label="Stage 2 one-shot attestation",
    )
    body = {key: value for key, value in attestation.items() if key != "content_sha256"}
    expected_keys = {
        "schema_version",
        "status",
        "stage1_bundle_manifest_path",
        "stage1_bundle_sha256",
        "stage1_handoff_content_sha256",
        "stage1_provider_identity_sha256",
        "production_endpoint",
        "production_model",
        "stage2_hierarchy_prompt_protocol",
        "stage2_hierarchy_prompt_protocol_sha256",
        "post_extraction_causal_review",
        "post_extraction_causal_review_sha256",
        "remote_runtime_identity",
        "prompt_nontruncation_execution_audit",
        "hierarchical_runner_identity_sha256",
        "preparation_dir",
        "hierarchical_batch_result",
        "final_output_dir",
        "immutable_run_manifest",
        "frozen_predictions",
        "fold_manifests",
        "one_shot_implementation_sha256",
        "run_result_audit_record_is_authorization",
        "architecture_at_a_time_hierarchy_required",
        "same_handoff_provider_used_for_spent_and_partitions",
        "genuine_one_shot_e2e_certified",
        "global_certification_mutated",
    }
    if (
        set(attestation) != expected_keys | {"content_sha256"}
        or attestation.get("content_sha256") != _canonical_sha256(body)
        or body.get("schema_version") != _STAGE2_ONE_SHOT_ATTESTATION_SCHEMA
        or body.get("status") != "completed"
        or body.get("production_endpoint") != request.get("endpoint")
        or body.get("production_model") != request.get("model_name")
        or body.get("stage1_bundle_sha256") != handoff_validation.get("bundle_sha256")
        or body.get("stage1_handoff_content_sha256")
        != handoff_validation.get("handoff_content_sha256")
        or body.get("run_result_audit_record_is_authorization") is not False
        or body.get("architecture_at_a_time_hierarchy_required") is not True
        or body.get("same_handoff_provider_used_for_spent_and_partitions") is not True
        or body.get("genuine_one_shot_e2e_certified") is not False
        or body.get("global_certification_mutated") is not False
    ):
        raise ValueError("Stage 2 one-shot attestation is invalid")
    bundle_path = Path(str(body.get("stage1_bundle_manifest_path", ""))).resolve(strict=True)
    if bundle_path != Path(str(handoff_validation["bundle_manifest_path"])):
        raise ValueError("Stage 2 one-shot used a different Stage 1 bundle")

    (
        configured_protocol,
        configured_causal_review,
        request_tokenizer_identity,
    ) = (
        _validated_stage2_request_contract(request)
        if _validated_request_contract is None
        else _validated_request_contract
    )
    if (
        body.get("stage2_hierarchy_prompt_protocol") != configured_protocol
        or body.get("stage2_hierarchy_prompt_protocol_sha256")
        != _canonical_sha256(configured_protocol)
        or body.get("post_extraction_causal_review") != configured_causal_review
        or body.get("post_extraction_causal_review_sha256")
        != _canonical_sha256(configured_causal_review)
    ):
        raise ValueError("Stage 2 one-shot scientific prompt/review configuration changed")

    runtime = body.get("remote_runtime_identity")
    expected_runtime_keys = {
        "endpoint_urls",
        "model",
        "guarded_client_paths",
        "endpoint_pool_or_fallback_allowed",
        "model_autodiscovery_or_substitution_allowed",
        "required_response_model",
        "required_finish_reason",
        "response_metadata_checked_before_content_semantics_and_cache",
        "prompt_nontruncation_guard",
        "local_prompt_tokens_plus_generation_within_context_required",
        "endpoint_prompt_token_usage_exact_match_required",
        "request_prompt_truncation_controls_allowed",
        "served_deployment_metadata_required",
        "caller_digest_authority",
    }
    prompt_guard = (
        runtime.get("prompt_nontruncation_guard") if isinstance(runtime, Mapping) else None
    )
    validated_guard = _validate_prompt_guard_identity(
        prompt_guard,
        request_tokenizer_identity=request_tokenizer_identity,
        model_name=request.get("model_name"),
        model_context_window_tokens=int(configured_protocol["model_context_window_tokens"]),
        label="Stage 2 one-shot",
    )
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != expected_runtime_keys
        or runtime.get("endpoint_urls") != [request.get("endpoint")]
        or runtime.get("model") != {"name": request.get("model_name")}
        or runtime.get("guarded_client_paths")
        != [
            "hierarchical_discovery",
            "proposal_and_post_extraction_review",
            "explicit_feature_extraction",
        ]
        or runtime.get("endpoint_pool_or_fallback_allowed") is not False
        or runtime.get("model_autodiscovery_or_substitution_allowed") is not False
        or runtime.get("required_response_model") != request.get("model_name")
        or runtime.get("required_finish_reason") != "stop"
        or runtime.get("response_metadata_checked_before_content_semantics_and_cache") is not True
        or runtime.get("local_prompt_tokens_plus_generation_within_context_required") is not True
        or runtime.get("endpoint_prompt_token_usage_exact_match_required") is not True
        or runtime.get("request_prompt_truncation_controls_allowed") is not False
        or runtime.get("served_deployment_metadata_required") is not False
        or runtime.get("caller_digest_authority") is not False
        or canary_validation.get("prompt_nontruncation_guard_identity_sha256")
        != validated_guard["identity_sha256"]
        or canary_validation.get("stage2_hierarchy_prompt_protocol_sha256")
        != body.get("stage2_hierarchy_prompt_protocol_sha256")
        or canary_validation.get("post_extraction_causal_review_sha256")
        != body.get("post_extraction_causal_review_sha256")
    ):
        raise ValueError("Stage 2 one-shot remote runtime identity is invalid")
    prompt_execution_audit = _validate_prompt_nontruncation_execution_audit(
        body.get("prompt_nontruncation_execution_audit"),
        guard_identity_sha256=validated_guard["identity_sha256"],
        protocol=configured_protocol,
    )

    for field in (
        "stage1_provider_identity_sha256",
        "hierarchical_runner_identity_sha256",
        "one_shot_implementation_sha256",
    ):
        _require_sha256(
            body.get(field),
            label=f"Stage 2 one-shot {field}",
        )

    batch = body.get("hierarchical_batch_result")
    run_manifest = body.get("immutable_run_manifest")
    prediction = body.get("frozen_predictions")
    fold_rows = body.get("fold_manifests")
    if (
        not isinstance(batch, Mapping)
        or set(batch) != {"path", "sha256"}
        or not isinstance(run_manifest, Mapping)
        or set(run_manifest) != {"path", "sha256", "content_sha256"}
        or not isinstance(prediction, Mapping)
        or set(prediction) != {"path", "size", "sha256"}
        or not isinstance(fold_rows, list)
        or not fold_rows
    ):
        raise ValueError("Stage 2 one-shot artifact references are not closed")
    batch_path = _registered_reference(
        batch.get("path"),
        registered_paths=inference_paths,
        label="Stage 2 hierarchical batch result",
    )
    run_manifest_path = _registered_reference(
        run_manifest.get("path"),
        registered_paths=inference_paths,
        label="Stage 2 immutable run manifest",
    )
    prediction_path = _registered_reference(
        prediction.get("path"),
        registered_paths=inference_paths,
        label="Stage 2 frozen predictions",
    )
    preparation_dir = Path(str(body.get("preparation_dir", "")))
    final_output_dir = Path(str(body.get("final_output_dir", "")))
    if (
        not preparation_dir.is_absolute()
        or preparation_dir.is_symlink()
        or preparation_dir.resolve(strict=True) != preparation_dir
        or batch_path.parent != preparation_dir
        or not final_output_dir.is_absolute()
        or final_output_dir.is_symlink()
        or final_output_dir.resolve(strict=True) != final_output_dir
        or run_manifest_path.parent != final_output_dir
        or prediction_path.parent != final_output_dir
    ):
        raise ValueError("Stage 2 one-shot artifact roots are invalid")

    batch_sha, _batch_size = stable_file_sha256(batch_path)
    run_sha, _run_size = stable_file_sha256(run_manifest_path)
    prediction_sha, prediction_size = stable_file_sha256(prediction_path)
    batch_wrapper = _read_json_object(
        batch_path,
        label="Stage 2 hierarchical batch result",
    )
    _validate_content_hashed_body(
        batch_wrapper,
        schema=_HIERARCHICAL_BATCH_RESULT_SCHEMA,
        label="Stage 2 hierarchical batch result",
    )
    run_wrapper = _read_json_object(
        run_manifest_path,
        label="Stage 2 immutable run manifest",
    )
    run_body = _validate_content_hashed_body(
        run_wrapper,
        schema=_STAGE2_RUN_MANIFEST_SCHEMA,
        label="Stage 2 immutable run manifest",
    )
    if (
        batch.get("sha256") != batch_sha
        or run_manifest.get("sha256") != run_sha
        or run_manifest.get("content_sha256") != run_wrapper.get("content_sha256")
        or prediction.get("sha256") != prediction_sha
        or isinstance(prediction.get("size"), bool)
        or not isinstance(prediction.get("size"), int)
        or prediction.get("size") != prediction_size
    ):
        raise ValueError("Stage 2 one-shot referenced artifact bytes changed")

    fold_paths: list[Path] = []
    for index, row in enumerate(fold_rows):
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "size",
            "sha256",
        }:
            raise ValueError("Stage 2 one-shot fold inventory is not closed")
        path = _registered_reference(
            row.get("path"),
            registered_paths=inference_paths,
            label=f"Stage 2 fold manifest {index}",
        )
        digest, size = stable_file_sha256(path)
        if (
            not path.is_relative_to(final_output_dir)
            or path in fold_paths
            or row.get("sha256") != digest
            or isinstance(row.get("size"), bool)
            or not isinstance(row.get("size"), int)
            or row.get("size") != size
        ):
            raise ValueError("Stage 2 one-shot fold manifest inventory changed")
        fold_paths.append(path)
    expected_fold_paths = [
        Path(str(path)).resolve(strict=True) for path in run_body.get("fold_manifest_paths", ())
    ]
    if fold_paths != expected_fold_paths:
        raise ValueError("Stage 2 one-shot fold manifests differ from the immutable run manifest")
    return {
        "attestation_path": str(attestation_path),
        "attestation_content_sha256": attestation["content_sha256"],
        "stage2_hierarchy_prompt_protocol_sha256": body["stage2_hierarchy_prompt_protocol_sha256"],
        "post_extraction_causal_review_sha256": body["post_extraction_causal_review_sha256"],
        "prompt_nontruncation_guard_identity_sha256": validated_guard["identity_sha256"],
        "prompt_nontruncation_execution_audit_sha256": (prompt_execution_audit["audit_sha256"]),
        "prompt_nontruncation_execution_record_count": (prompt_execution_audit["record_count"]),
        "hierarchical_batch_result_path": str(batch_path),
        "run_manifest_path": str(run_manifest_path),
        "prediction_path": str(prediction_path),
        "fold_manifest_paths": [str(path) for path in fold_paths],
    }


def _integer_series(frame: Any, column: str) -> list[int]:
    values = frame[column].tolist()
    output: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{column} contains a boolean row identity")
        integer = int(value)
        if float(value) != float(integer):
            raise ValueError(f"{column} contains a non-integral value")
        output.append(integer)
    return output


def _validate_prediction_frame(
    frame: Any,
    *,
    label: str,
    prediction_columns: Sequence[str] = _LEGACY_PREDICTION_COLUMNS,
) -> None:
    import numpy as np

    expected_columns = tuple(map(str, prediction_columns))
    if (
        expected_columns not in {_LEGACY_PREDICTION_COLUMNS, _DIRECT_CATE_PREDICTION_COLUMNS}
        or list(frame.columns) != list(expected_columns)
        or frame.empty
    ):
        raise ValueError(f"{label} has an invalid closed prediction schema")
    row_ids = _integer_series(frame, "_oci_row_id")
    folds = _integer_series(frame, "outer_fold")
    if len(row_ids) != len(set(row_ids)) or any(fold < 1 for fold in folds):
        raise ValueError(f"{label} has duplicated rows or invalid fold labels")
    numeric = frame[
        [column for column in expected_columns if column not in {"_oci_row_id", "outer_fold"}]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} contains non-finite predictions")
    if expected_columns == _LEGACY_PREDICTION_COLUMNS:
        if (
            (numeric[:, :2] < -1e-12).any()
            or (numeric[:, :2] > 1.0 + 1e-12).any()
            or not np.allclose(
                numeric[:, 2],
                numeric[:, 1] - numeric[:, 0],
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(f"{label} is not a probability-scale treatment effect")
    else:
        tolerance = float(64 * np.finfo(np.float64).eps)
        if (numeric[:, 0] < (-1.0 - tolerance)).any() or (numeric[:, 0] > (1.0 + tolerance)).any():
            raise ValueError(f"{label} is outside binary probability-difference bounds")


def _validate_strict_forest(
    *,
    fold_body: Mapping[str, Any],
    request: Mapping[str, Any],
    portable_direct: bool = False,
    freshly_authenticated_repository_runtime: Mapping[str, Any] | None = None,
    expected_direct_numerical_manifest_sha256: str | None = None,
) -> Mapping[str, Any] | None:
    fold_estimator = fold_body.get("final_ite_estimator")
    backend = (
        fold_estimator.get("forest_backend_identity")
        if isinstance(fold_estimator, Mapping)
        else None
    )
    identity = backend.get("identity") if isinstance(backend, Mapping) else None
    if (
        not isinstance(fold_estimator, Mapping)
        or fold_estimator.get("mode") != "strict_outer_honest_final_context_fit_causal_forest_v2"
        or fold_estimator.get("strict_causal_forest_active") is not True
        or fold_estimator.get("strict_causal_forest_required") is not True
        or fold_estimator.get("structured_interaction_head_constructed") is not False
        or fold_estimator.get("outer_heldout_labels_used") is not False
        or not isinstance(identity, Mapping)
    ):
        raise ValueError("Stage 2 fold strict-forest identity is invalid")
    if not portable_direct:
        if (
            int(identity.get("n_estimators", -1)) != int(request["forest_n_estimators"])
            or int(identity.get("min_samples_leaf", -1)) != int(request["forest_min_samples_leaf"])
            or identity.get("max_features") != request["forest_max_features"]
            or identity.get("honest") is not True
            or identity.get("inference") is not True
            or int(identity.get("random_state", -1)) != int(request["forest_random_seed"])
        ):
            raise ValueError("Stage 2 fold strict-forest identity is invalid")
        return None

    import numpy as np
    from ..models.strict_causal_forest_runtime import (
        StrictCausalForestRuntimeConfig,
    )

    # The portable request has exactly one authoritative forest
    # configuration.  Legacy flat flags must remain null, so neither the
    # workflow nor a terminal validator can accidentally reconstruct the v3
    # compatibility shim.
    legacy_forest_fields = {
        "forest_n_estimators",
        "forest_max_depth",
        "forest_min_samples_leaf",
        "forest_max_features",
        "forest_honest",
        "forest_inference",
        "forest_subforest_size",
        "forest_tune_model",
        "forest_nuisance_n_estimators",
        "forest_nuisance_max_depth",
        "forest_nuisance_min_samples_leaf",
        "forest_nuisance_treatment_max_features",
        "forest_nuisance_outcome_max_features",
        "forest_random_seed",
        "forest_n_jobs",
    }
    runtime_value = request.get("forest_runtime_config")
    try:
        runtime = StrictCausalForestRuntimeConfig.from_mapping(runtime_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "portable Stage 2 request lacks its closed strict-forest runtime " "configuration"
        ) from exc
    if (
        any(request.get(name) is not None for name in legacy_forest_fields)
        or isinstance(request.get("cpu_budget"), bool)
        or not isinstance(request.get("cpu_budget"), int)
        or int(request["cpu_budget"]) < 1
        or (runtime.operational.requested_host_cpu_budget != int(request["cpu_budget"]))
    ):
        raise ValueError(
            "portable Stage 2 request mixes legacy forest fields or has an "
            "inconsistent operational CPU budget"
        )

    scientific = runtime.causal_forest
    expected_effective_parameters = {
        "n_estimators": int(scientific.n_estimators),
        "max_depth": scientific.max_depth,
        "min_samples_leaf": scientific.min_samples_leaf,
        "max_features": scientific.max_features,
        "honest": bool(scientific.honest),
        "inference": bool(scientific.inference),
        "subforest_size": int(scientific.subforest_size),
        "random_state": int(scientific.random_seed),
    }
    expected_treatment_parameters = runtime.treatment_constructor_kwargs()
    expected_outcome_parameters = runtime.outcome_constructor_kwargs()
    expected_crossfit_parameters = runtime.crossfit_constructor_kwargs()
    expected_top_level_parameters = {
        **scientific.scientific_constructor_kwargs(),
        "n_jobs": 1,
        "verbose": int(runtime.operational.verbose),
        "use_ray": bool(runtime.operational.use_ray),
        "ray_remote_func_options": (runtime.operational.ray_remote_func_options),
    }
    expected_unfitted_audit = {
        "top_level_attributes": expected_top_level_parameters,
        "model_t_parameters": expected_treatment_parameters,
        "model_y_parameters": expected_outcome_parameters,
        "crossfit_parameters": expected_crossfit_parameters,
    }
    expected_grf_parameters = {
        "criterion": scientific.criterion,
        "fit_intercept": bool(scientific.fit_intercept),
        "honest": bool(scientific.honest),
        "inference": bool(scientific.inference),
        "max_depth": scientific.max_depth,
        "max_features": scientific.max_features,
        "max_samples": scientific.max_samples,
        "min_balancedness_tol": float(scientific.min_balancedness_tol),
        "min_impurity_decrease": float(scientific.min_impurity_decrease),
        "min_samples_leaf": scientific.min_samples_leaf,
        "min_samples_split": scientific.min_samples_split,
        "min_var_fraction_leaf": scientific.min_var_fraction_leaf,
        "min_var_leaf_on_val": bool(scientific.min_var_leaf_on_val),
        "min_weight_fraction_leaf": float(scientific.min_weight_fraction_leaf),
        "n_estimators": int(scientific.n_estimators),
        "n_jobs": 1,
        "random_state": int(scientific.random_seed),
        "subforest_size": int(scientific.subforest_size),
        "verbose": int(runtime.operational.verbose),
        "warm_start": False,
    }
    repetitions = 1
    folds = int(scientific.crossfit.n_splits)
    expected_fitted_audit = {
        "unfitted_estimator_graph": expected_unfitted_audit,
        "fitted_treatment_models": [
            [dict(expected_treatment_parameters) for _fold in range(folds)]
            for _repetition in range(repetitions)
        ],
        "fitted_outcome_models": [
            [dict(expected_outcome_parameters) for _fold in range(folds)]
            for _repetition in range(repetitions)
        ],
        "model_cate_template_parameters": expected_grf_parameters,
        "fitted_grf_parameters": [expected_grf_parameters],
    }
    operational_attestation = runtime.operational_attestation()
    expected_identity = {
        "backend": "repository_strict_causal_forest_path_v4",
        "configuration_mode": "portable_strict_runtime_config_v1",
        "strict_runtime_scientific_identity": (runtime.scientific_identity()),
        "strict_runtime_scientific_identity_sha256": (runtime.scientific_identity_sha256()),
        "operational_settings_excluded_from_scientific_identity": True,
        "exact_nuisance_used_as_fixed_internal_predictions": False,
        "tuning_labels": "outer_train_only",
        "outer_heldout_labels_accepted": False,
        "repository_runtime": (
            None
            if freshly_authenticated_repository_runtime is None
            else dict(freshly_authenticated_repository_runtime)
        ),
    }
    if (
        freshly_authenticated_repository_runtime is None
        or not isinstance(backend, Mapping)
        or set(backend) != {"identity", "identity_sha256"}
        or identity != expected_identity
        or backend.get("identity_sha256") != _canonical_sha256(identity)
        or fold_estimator.get("reference_only_role_neutral_runtime") is not True
        or fold_estimator.get("potential_outcome_reconstruction")
        != "not_emitted_direct_cate_estimand_only"
    ):
        raise ValueError(
            "portable Stage 2 fold did not use the authenticated strict v4 " "causal-forest runtime"
        )

    receipt = fold_estimator.get("forest_receipt")
    fit_audit = receipt.get("backend_fit_audit") if isinstance(receipt, Mapping) else None
    expected_fit_keys = {
        "configuration_mode",
        "runtime_schema_version",
        "scientific_identity",
        "scientific_identity_sha256",
        "operational_attestation",
        "operational_parameters",
        "tuning_configured",
        "tuning_attempted",
        "tuning_succeeded",
        "tuning_failure_fell_back_to_configured_parameters",
        "tuning_params",
        "crossfit_split_audit",
        "unfitted_estimator_audit",
        "fitted_estimator_audit",
        "fit_call_contract",
        "prediction_contrast",
        "effective_parameters",
        "effective_nuisance_parameters",
        "outer_train_labels_only",
        "outer_heldout_labels_accepted",
        "repository_runtime",
    }
    receipt_body = (
        {key: value for key, value in receipt.items() if key != "content_sha256"}
        if isinstance(receipt, Mapping)
        else None
    )
    expected_receipt_keys = {
        "schema_version",
        "outer_fold",
        "backend_identity",
        "backend_fit_audit",
        "reference_manifest_content_sha256",
        "effect_train_sha256",
        "effect_heldout_sha256",
        "control_train_sha256",
        "control_heldout_sha256",
        "treatment_sha256",
        "outcome_sha256",
        "tau_sha256",
        "probability_difference_bounds",
        "probability_difference_validation_tolerance",
        "probability_difference_bounds_validated",
        "probability_difference_values_clipped",
        "effect_column_count",
        "control_column_count",
        "explicit_effect_column_count",
        "explicit_control_column_count",
        "fit_row_count",
        "prediction_row_count",
        "strict_causal_forest_only",
        "structured_or_nonforest_fallback_used",
        "outer_heldout_labels_used",
        "potential_outcome_columns_emitted",
        "content_sha256",
    }
    split_audit = fit_audit.get("crossfit_split_audit") if isinstance(fit_audit, Mapping) else None
    split_records = split_audit.get("splits") if isinstance(split_audit, Mapping) else None
    split_body = (
        {key: value for key, value in split_audit.items() if key != "split_plan_sha256"}
        if isinstance(split_audit, Mapping)
        else None
    )
    fit_row_count = fold_body.get("train_row_count")
    split_records_valid = (
        isinstance(split_records, list)
        and len(split_records) == folds
        and isinstance(fit_row_count, int)
        and not isinstance(fit_row_count, bool)
        and all(
            isinstance(row, Mapping)
            and set(row)
            == {
                "fold_index",
                "train_count",
                "test_count",
                "train_index_sha256",
                "test_index_sha256",
            }
            and row.get("fold_index") == fold_index
            and isinstance(row.get("train_count"), int)
            and not isinstance(row.get("train_count"), bool)
            and isinstance(row.get("test_count"), int)
            and not isinstance(row.get("test_count"), bool)
            and int(row["train_count"]) > 0
            and int(row["test_count"]) > 0
            and int(row["train_count"]) + int(row["test_count"]) == int(fit_row_count)
            and _HEX_SHA256.fullmatch(str(row.get("train_index_sha256") or "")) is not None
            and _HEX_SHA256.fullmatch(str(row.get("test_index_sha256") or "")) is not None
            for fold_index, row in enumerate(split_records)
        )
        and sum(int(row["test_count"]) for row in split_records) == int(fit_row_count)
    )
    expected_fit_call_contract = dict(runtime.scientific_identity()["fit_contract"])
    expected_fit_call_contract.pop("prediction_contrast")
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != expected_receipt_keys
        or receipt.get("schema_version") != "role_neutral_direct_strict_causal_forest_receipt_v1"
        or receipt.get("outer_fold") != fold_body.get("outer_fold")
        or receipt.get("backend_identity") != backend
        or receipt.get("reference_manifest_content_sha256")
        != expected_direct_numerical_manifest_sha256
        or not isinstance(fit_audit, Mapping)
        or set(fit_audit) != expected_fit_keys
        or fit_audit.get("configuration_mode") != "portable_strict_runtime_config_v1"
        or fit_audit.get("runtime_schema_version") != runtime.schema_version
        or fit_audit.get("scientific_identity") != runtime.scientific_identity()
        or fit_audit.get("scientific_identity_sha256") != runtime.scientific_identity_sha256()
        or fit_audit.get("operational_attestation") != operational_attestation
        or fit_audit.get("effective_parameters") != expected_effective_parameters
        or fit_audit.get("effective_nuisance_parameters")
        != {
            "treatment_model": expected_treatment_parameters,
            "outcome_model": expected_outcome_parameters,
        }
        or fit_audit.get("operational_parameters") != operational_attestation
        or fit_audit.get("unfitted_estimator_audit") != expected_unfitted_audit
        or fit_audit.get("fitted_estimator_audit") != expected_fitted_audit
        or fit_audit.get("fit_call_contract") != expected_fit_call_contract
        or fit_audit.get("prediction_contrast") != {"T0": 0, "T1": 1}
        or not isinstance(split_audit, Mapping)
        or set(split_audit)
        != {
            "implementation",
            "parameters",
            "splits",
            "split_plan_sha256",
        }
        or split_audit.get("implementation") != scientific.crossfit.implementation
        or split_audit.get("parameters") != expected_crossfit_parameters
        or not split_records_valid
        or split_audit.get("split_plan_sha256") != _canonical_sha256(split_body)
        or fit_audit.get("tuning_configured") is not False
        or fit_audit.get("tuning_attempted") is not False
        or fit_audit.get("tuning_succeeded") is not None
        or fit_audit.get("tuning_failure_fell_back_to_configured_parameters") is not False
        or fit_audit.get("tuning_params") is not None
        or fit_audit.get("outer_train_labels_only") is not True
        or fit_audit.get("outer_heldout_labels_accepted") is not False
        or fit_audit.get("repository_runtime") != dict(freshly_authenticated_repository_runtime)
        or receipt.get("strict_causal_forest_only") is not True
        or receipt.get("structured_or_nonforest_fallback_used") is not False
        or receipt.get("outer_heldout_labels_used") is not False
        or receipt.get("potential_outcome_columns_emitted") is not False
        or receipt.get("probability_difference_bounds") != [-1.0, 1.0]
        or receipt.get("probability_difference_bounds_validated") is not True
        or receipt.get("probability_difference_values_clipped") is not False
        or receipt.get("fit_row_count") != fold_body.get("train_row_count")
        or receipt.get("prediction_row_count") != fold_body.get("heldout_row_count")
        or receipt.get("content_sha256") != _canonical_sha256(receipt_body)
        or any(
            not isinstance(receipt.get(name), int)
            or isinstance(receipt.get(name), bool)
            or int(receipt[name]) < 0
            for name in (
                "effect_column_count",
                "control_column_count",
                "explicit_effect_column_count",
                "explicit_control_column_count",
            )
        )
        or any(
            _HEX_SHA256.fullmatch(str(receipt.get(name) or "")) is None
            for name in (
                "effect_train_sha256",
                "effect_heldout_sha256",
                "control_train_sha256",
                "control_heldout_sha256",
                "treatment_sha256",
                "outcome_sha256",
                "tau_sha256",
                "reference_manifest_content_sha256",
            )
        )
        or not isinstance(
            receipt.get("probability_difference_validation_tolerance"),
            (int, float),
        )
        or isinstance(
            receipt.get("probability_difference_validation_tolerance"),
            bool,
        )
        or float(receipt["probability_difference_validation_tolerance"])
        != float(64 * np.finfo(np.float64).eps)
    ):
        raise ValueError("portable Stage 2 strict-forest fit audit is incomplete or changed")
    return dict(receipt)


def _validate_direct_stage1_source_identity(
    value: Any,
    *,
    handoff_validation: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate one fold's exact reference-only Stage 1 source declaration."""

    expected_keys = {
        "mode",
        "provider_identity_sha256",
        "plan_scientific_content_sha256",
        "source_execution_content_sha256",
        "runtime_binding_content_sha256",
        "prepared_projection_binding_content_sha256",
        "prepared_cohort_artifact_sha256",
        "row_map_sha256",
        "direct_numerical_bank_manifest_content_sha256",
        "legacy_stage1_loader_invoked",
        "tfidf_handoff_loader_invoked",
        "independent_stage1_refit_performed",
        "text_truncation_applied",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or value.get("mode") != "authenticated_role_neutral_all_ten_reference_only_v1"
        or value.get("provider_identity_sha256")
        != handoff_validation.get("stage2_provider_identity_sha256")
        or value.get("plan_scientific_content_sha256")
        != handoff_validation.get("scope_plan_scientific_content_sha256")
        or value.get("source_execution_content_sha256")
        != handoff_validation.get("source_role_neutral_execution_content_sha256")
        or value.get("row_map_sha256") != handoff_validation.get("row_map_sha256")
        or value.get("direct_numerical_bank_manifest_content_sha256")
        != handoff_validation.get("direct_numerical_bank_content_sha256")
        or value.get("legacy_stage1_loader_invoked") is not False
        or value.get("tfidf_handoff_loader_invoked") is not False
        or value.get("independent_stage1_refit_performed") is not False
        or value.get("text_truncation_applied") is not False
    ):
        raise ValueError("portable Stage 2 fold used an invalid or legacy Stage 1 source")
    for field in (
        "runtime_binding_content_sha256",
        "prepared_projection_binding_content_sha256",
        "prepared_cohort_artifact_sha256",
    ):
        _require_sha256(
            value.get(field),
            label=f"portable Stage 2 {field}",
        )
    return dict(value)


def _metric_values(frame: Any, *, truth: str) -> dict[str, Any]:
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    y = frame[truth].to_numpy(dtype=float)
    estimate = frame["pred_ite_prob"].to_numpy(dtype=float)
    truth_variance = float(np.var(y))
    estimate_variance = float(np.var(estimate))
    error = estimate - y
    return {
        "row_count": len(frame),
        "pearson_correlation_primary": (
            float(pearsonr(y, estimate).statistic)
            if truth_variance > 0 and estimate_variance > 0
            else None
        ),
        "spearman_correlation_secondary": (
            float(spearmanr(y, estimate).statistic)
            if truth_variance > 0 and estimate_variance > 0
            else None
        ),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(np.square(error)))),
        "mean_signed_error": float(np.mean(error)),
        "truth_variance": truth_variance,
        "estimate_variance": estimate_variance,
    }


def _metrics_equal(observed: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(observed, Mapping) or set(observed) != set(expected):
        return False
    for key, expected_value in expected.items():
        observed_value = observed[key]
        if expected_value is None:
            if observed_value is not None:
                return False
        elif key == "row_count":
            if int(observed_value) != int(expected_value):
                return False
        elif not math.isclose(
            float(observed_value),
            float(expected_value),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return False
    return True


def _validate_oracle_evaluation(
    *,
    request: Mapping[str, Any],
    prediction_frame: Any,
    prediction_sha256: str,
    prediction_manifest_path: Path,
    row_map_sha256: str,
    evaluation_paths: set[Path],
    prediction_columns: Sequence[str] = _LEGACY_PREDICTION_COLUMNS,
) -> Mapping[str, Any]:
    metrics_path = _unique_named(
        evaluation_paths,
        "evaluation_metrics.json",
        label="oracle evaluation phase",
    )
    joined_path = _unique_named(
        evaluation_paths,
        "predictions_with_oracle.parquet",
        label="oracle evaluation phase",
    )
    metrics = _read_json_object(
        metrics_path,
        label="post-freeze oracle evaluation metrics",
    )
    expected_metric_keys = {
        "schema_version",
        "prediction_sha256_before_oracle_read",
        "prediction_size_bytes",
        "prediction_manifest_sha256_before_oracle_read",
        "prediction_manifest_size_bytes",
        "stage1_row_map_sha256_before_oracle_read",
        "stage1_row_map_size_bytes",
        "event_order",
        "oracle_open_sequence",
        "freeze_validation_completed_sequence",
        "oracle_access_only_after_prediction_manifest_and_row_map_validation",
        "overall",
        "per_fold",
        "oracle_join_performed_posthoc",
        "joined_path",
    }
    manifest_sha, manifest_size = stable_file_sha256(prediction_manifest_path)
    events = metrics.get("event_order")
    if (
        set(metrics) != expected_metric_keys
        or metrics.get("schema_version") != "posthoc_oracle_ite_evaluation_v1"
        or metrics.get("prediction_sha256_before_oracle_read") != prediction_sha256
        or metrics.get("prediction_manifest_sha256_before_oracle_read") != manifest_sha
        or int(metrics.get("prediction_manifest_size_bytes", -1)) != manifest_size
        or metrics.get("stage1_row_map_sha256_before_oracle_read") != row_map_sha256
        or metrics.get("oracle_open_sequence") != 5
        or metrics.get("freeze_validation_completed_sequence") != 4
        or metrics.get("oracle_access_only_after_prediction_manifest_and_row_map_validation")
        is not True
        or metrics.get("oracle_join_performed_posthoc") is not True
        or Path(str(metrics.get("joined_path", ""))).resolve(strict=True) != joined_path
        or not isinstance(events, list)
        or len(events) != len(_ORACLE_EVENTS)
        or [event.get("sequence") for event in events] != [1, 2, 3, 4, 5]
        or [event.get("event") for event in events] != list(_ORACLE_EVENTS)
        or events[0].get("sha256") != prediction_sha256
        or events[1].get("sha256") != manifest_sha
        or events[2].get("sha256") != row_map_sha256
        or events[3].get("oracle_opened") is not False
        or events[4].get("all_freeze_validations_preceded_oracle_open") is not True
    ):
        raise ValueError("oracle-open event ordering was not proven")

    import numpy as np
    import pandas as pd

    joined = pd.read_parquet(joined_path)
    oracle_column = str(request.get("oracle_ite_column") or "")
    if (
        not oracle_column
        or oracle_column not in joined
        or joined["_oci_row_id"].duplicated().any()
        or _integer_series(joined, "_oci_row_id")
        != _integer_series(prediction_frame, "_oci_row_id")
        or not np.isfinite(joined[oracle_column].to_numpy(dtype=float)).all()
    ):
        raise ValueError("post-freeze oracle join is invalid")
    for column in prediction_columns:
        if column not in joined or not np.array_equal(
            joined[column].to_numpy(),
            prediction_frame[column].to_numpy(),
            equal_nan=False,
        ):
            raise ValueError("oracle join changed frozen prediction values")
    if not _metrics_equal(
        metrics.get("overall"),
        _metric_values(joined, truth=oracle_column),
    ):
        raise ValueError("overall oracle metrics do not match the joined rows")
    expected_per_fold = [
        {
            "outer_fold": int(fold),
            **_metric_values(group, truth=oracle_column),
        }
        for fold, group in joined.groupby("outer_fold", sort=True)
    ]
    observed_per_fold = metrics.get("per_fold")
    if not isinstance(observed_per_fold, list) or len(observed_per_fold) != len(expected_per_fold):
        raise ValueError("per-fold oracle metric coverage is incomplete")
    for observed, expected in zip(observed_per_fold, expected_per_fold, strict=True):
        if (
            not isinstance(observed, Mapping)
            or int(observed.get("outer_fold", -1)) != expected["outer_fold"]
            or not _metrics_equal(
                {key: value for key, value in observed.items() if key != "outer_fold"},
                {key: value for key, value in expected.items() if key != "outer_fold"},
            )
        ):
            raise ValueError("per-fold oracle metrics do not match the joined rows")
    return {
        "evaluation_metrics_path": str(metrics_path),
        "joined_path": str(joined_path),
        "oracle_open_order_proven": True,
        "metric_row_count": len(joined),
    }


def validate_real_stage2_terminal_artifacts(
    *,
    request: Mapping[str, Any],
    phase_records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Reopen all real Stage 2 artifacts and prove their terminal contract."""

    benchmark_execution_validation = (
        _validate_benchmarked_execution_authority(request)
    )
    by_phase = _phase_map(phase_records)
    if "stage2_inference" not in by_phase:
        return {
            "real_stage2_artifacts_detected": False,
            "reason": "stage2_inference_phase_absent",
            "benchmark_execution_validation": (
                benchmark_execution_validation
            ),
        }
    # Authenticate the inference publication first.  Evaluation artifacts can
    # contain oracle values, so do not reopen those bytes until the phase
    # sequence itself proves that frozen inference preceded evaluation.
    inference_paths = _artifact_paths(by_phase["stage2_inference"])
    manifests = [path for path in inference_paths if path.name == "immutable_run_manifest.json"]
    prediction_candidates = [
        path for path in inference_paths if path.name == "frozen_predictions.parquet"
    ]
    if not manifests and not prediction_candidates:
        return {
            "real_stage2_artifacts_detected": False,
            "reason": "injected_stage2_phase_without_real_prediction_outputs",
            "benchmark_execution_validation": (
                benchmark_execution_validation
            ),
        }
    phase_positions = _validate_terminal_phase_order(
        phase_records,
        oracle_configured=bool(request.get("evaluate_oracle_posthoc")),
    )
    paths_by_phase = {
        phase: (
            inference_paths
            if phase == "stage2_inference"
            else set() if phase == "oracle_evaluation" else _artifact_paths(record)
        )
        for phase, record in by_phase.items()
    }
    request_contract = _validated_stage2_request_contract(request)
    _configured_protocol, configured_causal_review, _tokenizer_identity = request_contract
    if len(manifests) != 1:
        raise ValueError("Stage 2 must publish exactly one run manifest")
    predictions = [path for path in prediction_candidates if path.parent == manifests[0].parent]
    if len(predictions) != 1:
        raise ValueError(
            "Stage 2 must publish exactly one run manifest and combined prediction file"
        )
    handoff_validation = validate_real_stage1_handoff(
        request=request,
        phase_records=phase_records,
        _validated_phase_map=by_phase,
        _validated_path_inventory=paths_by_phase,
    )
    from .production_role_neutral_stage2_handoff import (
        ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
    )

    portable_direct = (
        handoff_validation.get("handoff_kind") == ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
    )
    prediction_columns = (
        _DIRECT_CATE_PREDICTION_COLUMNS if portable_direct else _LEGACY_PREDICTION_COLUMNS
    )
    freshly_authenticated_repository_runtime: Mapping[str, Any] | None = None
    if portable_direct:
        from .final_context_fit_causal_forest_adapter import (
            _repository_causal_forest_runtime_attestation,
        )

        freshly_authenticated_repository_runtime = dict(
            _repository_causal_forest_runtime_attestation()
        )
    canary_validation = validate_real_stage2_canary(
        request=request,
        phase_records=phase_records,
        handoff_validation=handoff_validation,
        _validated_phase_map=by_phase,
        _validated_path_inventory=paths_by_phase,
        _validated_request_contract=request_contract,
    )
    one_shot_validation = validate_real_stage2_one_shot_attestation(
        request=request,
        phase_records=phase_records,
        handoff_validation=handoff_validation,
        canary_validation=canary_validation,
        _validated_phase_map=by_phase,
        _validated_path_inventory=paths_by_phase,
        _validated_request_contract=request_contract,
    )
    run_manifest_path = manifests[0]
    prediction_path = predictions[0]
    if one_shot_validation.get("run_manifest_path") != str(
        run_manifest_path
    ) or one_shot_validation.get("prediction_path") != str(prediction_path):
        raise ValueError("Stage 2 one-shot attestation does not bind the terminal outputs")
    run_wrapper = _read_json_object(
        run_manifest_path,
        label="immutable Stage 2 run manifest",
    )
    run_body = _validate_content_hashed_body(
        run_wrapper,
        schema=_STAGE2_RUN_MANIFEST_SCHEMA,
        label="immutable Stage 2 run manifest",
    )
    prediction_sha, prediction_size = stable_file_sha256(prediction_path)
    estimator = run_body.get("final_ite_estimator")
    if (
        run_body.get("prediction_path") != str(prediction_path)
        or run_body.get("prediction_sha256") != prediction_sha
        or int(run_body.get("prediction_row_count", -1)) < 1
        or run_body.get("prediction_columns") != list(prediction_columns)
        or run_body.get("outer_test_rows_predicted_once") is not True
        or run_body.get("oracle_columns_written") is not False
        or not isinstance(estimator, Mapping)
        or estimator.get("mode") != "strict_outer_honest_final_context_fit_causal_forest_v2"
        or estimator.get("strict_causal_forest_active_for_every_fold") is not True
        or estimator.get("strict_causal_forest_required") is not True
        or estimator.get("fixed_prior_working_backend_active") is not True
        or (portable_direct and estimator.get("reference_only_role_neutral_runtime") is not True)
    ):
        raise ValueError("frozen Stage 2 run manifest is invalid")

    import pandas as pd

    prediction_frame = pd.read_parquet(prediction_path)
    _validate_prediction_frame(
        prediction_frame,
        label="combined frozen Stage 2 predictions",
        prediction_columns=prediction_columns,
    )
    if len(prediction_frame) != int(run_body["prediction_row_count"]) or _integer_series(
        prediction_frame, "_oci_row_id"
    ) != sorted(_integer_series(prediction_frame, "_oci_row_id")):
        raise ValueError("combined frozen predictions changed row coverage or order")
    folds = _integer_series(prediction_frame, "outer_fold")
    outer_fold_count = int(request["outer_folds"])
    if set(folds) != set(range(1, outer_fold_count + 1)):
        raise ValueError("combined frozen predictions omit an outer fold")
    direct_outer_assignments = (
        handoff_validation.get("outer_fold_assignments") if portable_direct else None
    )
    if portable_direct and (
        not isinstance(direct_outer_assignments, Mapping)
        or {int(fold) for fold in direct_outer_assignments} != set(range(1, outer_fold_count + 1))
    ):
        raise ValueError(
            "portable Stage 2 fold count differs from the authenticated " "Stage 1 scope plan"
        )

    fold_manifest_paths = [
        Path(str(value)).resolve(strict=True) for value in run_body.get("fold_manifest_paths", ())
    ]
    if (
        len(fold_manifest_paths) != outer_fold_count
        or len(fold_manifest_paths) != int(run_body.get("fold_count", -1))
        or len(fold_manifest_paths) != len(set(fold_manifest_paths))
        or any(path not in inference_paths for path in fold_manifest_paths)
    ):
        raise ValueError("Stage 2 fold manifest inventory is incomplete")
    seen_folds: set[int] = set()
    fold_prediction_paths: set[Path] = set()
    direct_source_identity: Mapping[str, Any] | None = None
    for fold_manifest_path in fold_manifest_paths:
        wrapper = _read_json_object(
            fold_manifest_path,
            label="immutable Stage 2 fold manifest",
        )
        body = _validate_content_hashed_body(
            wrapper,
            schema="all_evidence_fusion_frozen_fold_v20",
            label="immutable Stage 2 fold manifest",
        )
        fold = int(body.get("outer_fold", 0))
        if fold < 1 or fold > outer_fold_count or fold in seen_folds:
            raise ValueError("Stage 2 fold manifest identity is invalid")
        if portable_direct:
            source_identity = _validate_direct_stage1_source_identity(
                body.get("stage1_reference_source"),
                handoff_validation=handoff_validation,
            )
            if direct_source_identity is not None and source_identity != direct_source_identity:
                raise ValueError(
                    "portable Stage 2 folds disagree on their reference-only " "Stage 1 source"
                )
            direct_attested_source = one_shot_validation.get("direct_source_binding")
            if not isinstance(direct_attested_source, Mapping) or any(
                source_identity[field] != direct_attested_source[field]
                for field in (
                    "provider_identity_sha256",
                    "runtime_binding_content_sha256",
                    "prepared_projection_binding_content_sha256",
                    "prepared_cohort_artifact_sha256",
                    "row_map_sha256",
                    "direct_numerical_bank_manifest_content_sha256",
                )
            ):
                raise ValueError(
                    "portable Stage 2 fold source differs from its direct " "one-shot attestation"
                )
            direct_source_identity = source_identity
            if (
                body.get("legacy_handoff_sha256") is not None
                or body.get("tfidf_handoff_sha256") is not None
            ):
                raise ValueError("portable Stage 2 fold registered a legacy handoff")
        direct_forest_receipt = _validate_strict_forest(
            fold_body=body,
            request=request,
            portable_direct=portable_direct,
            freshly_authenticated_repository_runtime=(freshly_authenticated_repository_runtime),
            expected_direct_numerical_manifest_sha256=(
                handoff_validation.get("direct_numerical_bank_content_sha256")
                if portable_direct
                else None
            ),
        )
        fold_rows = prediction_frame[prediction_frame["outer_fold"] == fold]
        fold_ids = _integer_series(fold_rows, "_oci_row_id")
        expected_direct_assignment = (
            direct_outer_assignments.get(fold)
            if isinstance(direct_outer_assignments, Mapping)
            else None
        )
        fold_prediction_path = Path(str(body.get("prediction_path", ""))).resolve(strict=True)
        if (
            int(body.get("heldout_row_count", -1)) != len(fold_rows)
            or body.get("heldout_row_fingerprint") != row_set_fingerprint(fold_ids)
            or int(body.get("train_row_count", -1)) + len(fold_rows) != len(prediction_frame)
            or body.get("outer_heldout_outcomes_used") is not False
            or body.get("oracle_columns_written") is not False
            or body.get("prediction_columns") != list(prediction_columns)
            or fold_prediction_path not in inference_paths
            or fold_prediction_path in fold_prediction_paths
            or (
                portable_direct
                and (
                    not isinstance(expected_direct_assignment, Mapping)
                    or set(fold_ids)
                    != set(
                        map(
                            int,
                            expected_direct_assignment["heldout_row_ids"],
                        )
                    )
                    or body.get("train_row_fingerprint")
                    != row_set_fingerprint(expected_direct_assignment["fit_row_ids"])
                    or int(body.get("train_row_count", -1))
                    != len(expected_direct_assignment["fit_row_ids"])
                )
            )
        ):
            raise ValueError("Stage 2 fold heldout coverage is invalid")
        fold_sha, _fold_size = stable_file_sha256(fold_prediction_path)
        fold_frame = pd.read_parquet(fold_prediction_path)
        _validate_prediction_frame(
            fold_frame,
            label=f"outer-fold {fold} frozen predictions",
            prediction_columns=prediction_columns,
        )
        if fold_sha != body.get("prediction_sha256") or not fold_frame.sort_values(
            "_oci_row_id"
        ).reset_index(drop=True).equals(
            fold_rows.sort_values("_oci_row_id").reset_index(drop=True)
        ):
            raise ValueError("Stage 2 fold predictions differ from the combined file")
        if portable_direct:
            if not isinstance(direct_forest_receipt, Mapping):
                raise ValueError("portable Stage 2 fold lacks a strict forest receipt")
            from .all_evidence_fusion_runner import (
                _numerical_array_sha256,
            )

            if direct_forest_receipt.get("tau_sha256") != _numerical_array_sha256(
                fold_frame["pred_ite_prob"].to_numpy(dtype=float)
            ):
                raise ValueError(
                    "portable Stage 2 strict forest receipt differs from " "the frozen CATE values"
                )
        seen_folds.add(fold)
        fold_prediction_paths.add(fold_prediction_path)
    if seen_folds != set(range(1, outer_fold_count + 1)):
        raise ValueError("Stage 2 fold identities are incomplete")
    if portable_direct and direct_source_identity is None:
        raise ValueError("portable Stage 2 has no authenticated reference-only fold source")
    if portable_direct and (
        one_shot_validation.get("fold_manifest_paths")
        != [str(path) for path in fold_manifest_paths]
        or set(one_shot_validation.get("fold_prediction_paths") or ())
        != {str(path) for path in fold_prediction_paths}
    ):
        raise ValueError(
            "portable Stage 2 one-shot fold inventory differs from the run " "manifest"
        )

    stage1_paths = paths_by_phase.get("stage1_modeling", set())
    direct_row_map_path = handoff_validation.get("row_map_path")
    if direct_row_map_path is None:
        row_map_path = _unique_named(
            stage1_paths,
            "row_registry.parquet",
            label="Stage 1 bundle",
        )
    else:
        row_map_path = Path(str(direct_row_map_path))
        if (
            not row_map_path.is_absolute()
            or row_map_path.is_symlink()
            or row_map_path.resolve(strict=True) != row_map_path
        ):
            raise ValueError("portable Stage 1 handoff returned a noncanonical row map")
    row_map_sha, _row_map_size = stable_file_sha256(row_map_path)
    row_map = pd.read_parquet(row_map_path)
    if (
        "_oci_row_id" not in row_map
        or row_map["_oci_row_id"].duplicated().any()
        or _integer_series(row_map, "_oci_row_id")
        != _integer_series(prediction_frame, "_oci_row_id")
    ):
        raise ValueError("Stage 1 row map order does not match frozen predictions")

    oracle_validation: Mapping[str, Any] | None = None
    evaluation_paths = _artifact_paths(by_phase.get("oracle_evaluation"))
    if bool(request.get("evaluate_oracle_posthoc")):
        oracle_validation = _validate_oracle_evaluation(
            request=request,
            prediction_frame=prediction_frame,
            prediction_sha256=prediction_sha,
            prediction_manifest_path=run_manifest_path,
            row_map_sha256=row_map_sha,
            evaluation_paths=evaluation_paths,
            prediction_columns=prediction_columns,
        )
        oracle_validation = {
            **dict(oracle_validation),
            "workflow_phase_order_proven": True,
            "stage1_graph_handoff_and_canary_preceded_oracle": True,
            "all_configured_strict_folds_and_attestation_preceded_oracle": (True),
            "configured_strict_fold_count_preceded_oracle": (outer_fold_count),
        }
    else:
        if any(
            path.name in {"evaluation_metrics.json", "predictions_with_oracle.parquet"}
            for path in evaluation_paths
        ):
            raise ValueError("unconfigured oracle evaluation artifacts were published")
    return {
        "real_stage2_artifacts_detected": True,
        "execution_completed": True,
        "run_validation_status": "accepted",
        "global_release_certified": False,
        "workflow_phase_order_validated": True,
        "terminal_phase_positions": dict(phase_positions),
        "portable_reference_only_stage2_validated": portable_direct,
        "portable_reference_only_stage1_source": direct_source_identity,
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha,
        "prediction_size_bytes": prediction_size,
        "prediction_row_count": len(prediction_frame),
        "fold_manifest_count": len(fold_manifest_paths),
        "fold_prediction_count": len(fold_prediction_paths),
        "stage1_handoff_validation": handoff_validation,
        "stage2_canary_validation": canary_validation,
        "stage2_one_shot_validation": one_shot_validation,
        "strict_forest_identity_validated_per_fold": True,
        "post_extraction_causal_review_configuration": (configured_causal_review),
        "probability_scale_identity_validated": True,
        "stage1_row_map_path": str(row_map_path),
        "stage1_row_map_sha256": row_map_sha,
        "row_order_validated": True,
        "oracle_validation": oracle_validation,
        "benchmark_execution_validation": benchmark_execution_validation,
    }


__all__ = [
    "validate_real_stage1_handoff",
    "validate_real_stage2_canary",
    "validate_real_stage2_one_shot_attestation",
    "validate_real_stage2_terminal_artifacts",
]
