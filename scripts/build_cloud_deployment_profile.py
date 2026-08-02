#!/usr/bin/env python3
"""Build and validate the canonical eight-GPU cloud deployment profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from oci.inference.portable_workflow_spec import DeploymentProfile


GPU_COUNT = 8


def _matches_legacy_profile_without_owner_capacity(
    *,
    existing: object,
    candidate: dict[str, object],
) -> bool:
    """Compare the last pre-capacity schema with its current equivalent."""

    if not isinstance(existing, dict):
        return False
    existing_stage1 = existing.get("stage1_execution")
    candidate_stage1 = candidate.get("stage1_execution")
    if (
        existing.get("schema_version")
        != "portable_all_evidence_deployment_profile_v9"
        or not isinstance(existing_stage1, dict)
        or existing_stage1.get("schema_version")
        != "portable_stage1_execution_profile_v8"
        or "owner_capacity_policy" in existing_stage1
        or not isinstance(candidate_stage1, dict)
    ):
        return False

    normalized_existing = dict(existing)
    normalized_existing_stage1 = dict(existing_stage1)
    normalized_existing["schema_version"] = candidate.get(
        "schema_version"
    )
    normalized_existing_stage1["schema_version"] = (
        candidate_stage1.get("schema_version")
    )
    normalized_existing["stage1_execution"] = (
        normalized_existing_stage1
    )

    normalized_candidate = dict(candidate)
    normalized_candidate_stage1 = dict(candidate_stage1)
    normalized_candidate_stage1.pop("owner_capacity_policy", None)
    normalized_candidate["stage1_execution"] = (
        normalized_candidate_stage1
    )
    return normalized_existing == normalized_candidate


def build_profile(args: argparse.Namespace) -> DeploymentProfile:
    profile = json.loads(args.base.read_text(encoding="utf-8"))
    profile.update(
        {
            "dataset_path": str(args.dataset),
            "durable_artifact_root": str(args.durable_root),
            "scratch_root": str(args.scratch_root),
            "embedding_model_locator": str(args.embedding_model),
            "htr_model_locator": str(args.htr_model),
            "stage2_tokenizer_locator": str(args.stage2_tokenizer),
            "stage1_profile_locator": str(args.stage1_profile),
            "query_profile_locator": str(args.query_profile),
            "embedding_model_name": "Qwen/Qwen3-Embedding-8B",
            "embedding_batch_size": args.embedding_batch_size,
            "devices": [f"cuda:{index}" for index in range(GPU_COUNT)],
            "cpu_budget": args.cpu_budget,
            "response_concurrency": GPU_COUNT,
            "storage_backend": "local_posix",
            "endpoint": args.endpoint,
            "endpoint_model": args.endpoint_model,
            "oracle_source": str(args.dataset),
            "oracle_unit_id_column": "patient_id",
            "oracle_ite_column": "true_ite_prob",
        }
    )
    profile["forest_operational"]["requested_host_cpu_budget"] = args.cpu_budget
    safety = profile["resource_performance_safety"]
    safety["fail_on_external_gpu_occupants"] = True
    safety["maximum_ordinary_read_amplification"] = float(
        args.preflight_lanes
    )
    stage1 = profile["stage1_execution"]
    owner_ceiling = min(
        int(args.cpu_budget),
        GPU_COUNT * int(args.max_workers_per_device),
    )
    stage1.update(
        {
            "resource_kind": "accelerator",
            "device_count": GPU_COUNT,
            "scope_workers_per_device": int(
                args.max_workers_per_device
            ),
            "max_parallel_owners": owner_ceiling,
        }
    )
    stage1["owner_capacity_policy"] = {
        "schema_version": (
            "portable_stage1_owner_capacity_policy_v1"
        ),
        "mode": "resource_autodetect",
        "estimated_device_memory_bytes_per_owner": int(
            args.estimated_device_memory_per_owner
        ),
        "device_memory_reserve_bytes": int(
            args.device_memory_reserve
        ),
        "estimated_host_memory_bytes_per_owner": int(
            args.estimated_host_memory_per_owner
        ),
        "host_memory_budget_fraction": float(
            args.host_memory_budget_fraction
        ),
        "minimum_cpu_threads_per_owner": int(
            args.minimum_cpu_threads_per_owner
        ),
    }
    stage1["neural_query_topology"]["mode"] = (
        "one_context_per_selected_device"
    )
    stage1["preflight_execution_policy"] = {
        "schema_version": "portable_stage1_preflight_execution_policy_v1",
        "max_parallel_owners": args.preflight_lanes,
        "memory_budget_bytes": args.preflight_memory_budget,
        "estimated_owner_peak_bytes": args.preflight_owner_peak,
        "input_io_lane_cap": args.preflight_lanes,
        "publication_io_lane_cap": args.preflight_lanes,
        "authentication_io_lane_cap": args.preflight_lanes,
    }
    htr = stage1["htr_operational_controls"]
    htr["fold_parallelism"] = 1
    htr["fold_slots_per_device"] = 1
    htr["sentence_encoder_batch_size"] = 16
    neural = stage1["neural_query_operational_controls"]
    neural["inner_fold_parallelism"] = 1
    neural["bank_parallelism"] = 1
    neural["fold_slots_per_device"] = 1

    encoded = (
        json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    args.target.parent.mkdir(parents=True, exist_ok=True)
    reused_legacy_profile = False
    try:
        descriptor = os.open(
            args.target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        existing_bytes = args.target.read_bytes()
        if existing_bytes == encoded:
            pass
        else:
            compiled_existing = DeploymentProfile.from_json(args.target)
            try:
                existing_mapping = json.loads(
                    existing_bytes.decode("utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "existing current deployment profile is invalid"
                ) from exc
            if not _matches_legacy_profile_without_owner_capacity(
                existing=existing_mapping,
                candidate=profile,
            ):
                raise ValueError(
                    "existing current deployment profile differs; choose "
                    "fresh run and scratch roots"
                )
            # Keep the authenticated v9 bytes in place. Loading them applies
            # the supported fixed-capacity migration without rewriting the
            # deployment profile named by an interrupted run.
            reused_legacy_profile = True
            compiled = compiled_existing
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    if not reused_legacy_profile:
        compiled = DeploymentProfile.from_json(args.target)
    expected_capacity_mode = (
        "fixed" if reused_legacy_profile else "resource_autodetect"
    )
    execution = compiled.stage1_execution
    if (
        tuple(compiled.devices)
        != tuple(f"cuda:{index}" for index in range(GPU_COUNT))
        or execution.device_count != GPU_COUNT
        or execution.scope_workers_per_device
        != int(args.max_workers_per_device)
        or execution.max_parallel_owners != owner_ceiling
        or execution.owner_capacity_policy.mode
        != expected_capacity_mode
        or execution.htr_operational_controls.fold_parallelism != 1
        or execution.htr_operational_controls.fold_slots_per_device != 1
        or execution.htr_operational_controls.sentence_encoder_batch_size != 16
        or execution.neural_query_topology.mode
        != "one_context_per_selected_device"
    ):
        raise RuntimeError(
            "compiled cloud deployment does not expose eight disjoint lanes"
        )
    return compiled


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError(
            "expected a fraction in (0, 1]"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for name in (
        "base",
        "target",
        "dataset",
        "durable-root",
        "scratch-root",
        "embedding-model",
        "htr-model",
        "stage2-tokenizer",
        "stage1-profile",
        "query-profile",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cpu-budget", type=_positive, required=True)
    parser.add_argument("--preflight-memory-budget", type=_positive, required=True)
    parser.add_argument("--preflight-owner-peak", type=_positive, required=True)
    parser.add_argument("--preflight-lanes", type=_positive, required=True)
    parser.add_argument("--embedding-batch-size", type=_positive, required=True)
    parser.add_argument(
        "--max-workers-per-device",
        type=_positive,
        required=True,
    )
    parser.add_argument(
        "--estimated-device-memory-per-owner",
        type=_positive,
        required=True,
    )
    parser.add_argument(
        "--device-memory-reserve",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--estimated-host-memory-per-owner",
        type=_positive,
        required=True,
    )
    parser.add_argument(
        "--host-memory-budget-fraction",
        type=_fraction,
        required=True,
    )
    parser.add_argument(
        "--minimum-cpu-threads-per-owner",
        type=_positive,
        required=True,
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--endpoint-model", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.device_memory_reserve < 0:
        raise ValueError("device memory reserve cannot be negative")
    if args.preflight_lanes > GPU_COUNT:
        raise ValueError("preflight lanes cannot exceed the eight-GPU owner cap")
    build_profile(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
