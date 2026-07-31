#!/usr/bin/env python3
"""Build and validate the canonical eight-GPU cloud deployment profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from oci.inference.portable_workflow_spec import DeploymentProfile


GPU_COUNT = 8


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
    stage1.update(
        {
            "resource_kind": "accelerator",
            "device_count": GPU_COUNT,
            "scope_workers_per_device": 1,
            "max_parallel_owners": GPU_COUNT,
        }
    )
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
    try:
        descriptor = os.open(
            args.target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        if args.target.read_bytes() != encoded:
            raise ValueError(
                "existing current deployment profile differs; choose fresh "
                "run and scratch roots"
            )
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    compiled = DeploymentProfile.from_json(args.target)
    execution = compiled.stage1_execution
    if (
        tuple(compiled.devices)
        != tuple(f"cuda:{index}" for index in range(GPU_COUNT))
        or execution.device_count != GPU_COUNT
        or execution.scope_workers_per_device != 1
        or execution.max_parallel_owners != GPU_COUNT
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
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--endpoint-model", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.preflight_lanes > GPU_COUNT:
        raise ValueError("preflight lanes cannot exceed the eight-GPU owner cap")
    build_profile(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
