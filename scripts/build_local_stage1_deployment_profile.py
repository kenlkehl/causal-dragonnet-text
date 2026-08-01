#!/usr/bin/env python3
"""Build one generic accelerator deployment for local Stage 1 production."""

from __future__ import annotations

import argparse
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

from oci.inference.portable_workflow_spec import DeploymentProfile


_CUDA_DEVICE = re.compile(r"cuda:(0|[1-9][0-9]*)")


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _free_fraction(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(
            "expected a decimal GPU free-memory fraction"
        ) from exc
    if (
        not parsed.is_finite()
        or not Decimal("0") < parsed < Decimal("1")
    ):
        raise argparse.ArgumentTypeError(
            "GPU free-memory fraction must be in (0, 1)"
        )
    return parsed


def _devices(values: Iterable[str]) -> tuple[str, ...]:
    devices = tuple(str(value) for value in values)
    if (
        not devices
        or len(devices) != len(set(devices))
        or any(_CUDA_DEVICE.fullmatch(value) is None for value in devices)
    ):
        raise ValueError("devices must be unique explicit cuda:N locators")
    return devices


def build_profile(args: argparse.Namespace) -> DeploymentProfile:
    devices = _devices(args.device)
    owner_cap = (
        len(devices)
        if args.max_parallel_owners is None
        else int(args.max_parallel_owners)
    )
    capacity = len(devices) * int(args.scope_workers_per_device)
    if owner_cap > capacity or owner_cap > int(args.cpu_budget):
        raise ValueError(
            "max_parallel_owners exceeds device or CPU capacity"
        )
    if int(args.preflight_lanes) > owner_cap:
        raise ValueError(
            "preflight lanes cannot exceed the Stage 1 owner cap"
        )

    profile = json.loads(args.base.read_text(encoding="utf-8"))
    profile.update(
        {
            "dataset_path": str(args.dataset),
            "durable_artifact_root": str(args.durable_root),
            "scratch_root": str(args.scratch_root),
            "embedding_model_locator": str(args.embedding_model),
            "htr_model_locator": str(args.htr_model),
            "stage2_tokenizer_locator": None,
            "stage1_profile_locator": str(args.stage1_profile),
            "query_profile_locator": str(args.query_profile),
            "embedding_model_name": "Qwen/Qwen3-Embedding-8B",
            "embedding_batch_size": int(args.embedding_batch_size),
            "devices": list(devices),
            "cpu_budget": int(args.cpu_budget),
            "response_concurrency": max(1, owner_cap),
            "storage_backend": "local_posix",
            "endpoint": None,
            "endpoint_model": None,
            "oracle_source": None,
            "oracle_unit_id_column": None,
            "oracle_ite_column": None,
        }
    )
    profile["forest_operational"]["requested_host_cpu_budget"] = int(
        args.cpu_budget
    )
    safety = profile["resource_performance_safety"]
    # Existing process presence is not itself unsafe. Admission is based on
    # aggregate VRAM occupancy, irrespective of which processes own it.
    safety["fail_on_external_gpu_occupants"] = False
    safety["gpu_max_allocation_fraction"] = float(
        Decimal("1") - args.gpu_minimum_free_fraction
    )
    safety["gpu_minimum_headroom_bytes"] = 0
    safety["maximum_ordinary_read_amplification"] = float(
        args.preflight_lanes
    )

    stage1 = profile["stage1_execution"]
    stage1.update(
        {
            "resource_kind": "accelerator",
            "device_count": len(devices),
            "scope_workers_per_device": int(
                args.scope_workers_per_device
            ),
            "max_parallel_owners": owner_cap,
        }
    )
    stage1["neural_query_topology"]["mode"] = (
        "one_context_per_selected_device"
    )
    stage1["preflight_execution_policy"] = {
        "schema_version": (
            "portable_stage1_preflight_execution_policy_v1"
        ),
        "max_parallel_owners": int(args.preflight_lanes),
        "memory_budget_bytes": int(args.preflight_memory_budget),
        "estimated_owner_peak_bytes": int(args.preflight_owner_peak),
        "input_io_lane_cap": int(args.preflight_lanes),
        "publication_io_lane_cap": int(args.preflight_lanes),
        "authentication_io_lane_cap": int(args.preflight_lanes),
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
        json.dumps(
            profile,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
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
                "existing deployment profile differs; choose a fresh "
                "profile path"
            )
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    compiled = DeploymentProfile.from_json(args.target)
    execution = compiled.stage1_execution
    if (
        tuple(compiled.devices) != devices
        or execution.device_count != len(devices)
        or execution.scope_workers_per_device
        != int(args.scope_workers_per_device)
        or execution.max_parallel_owners != owner_cap
        or execution.htr_operational_controls.fold_parallelism != 1
        or execution.htr_operational_controls.fold_slots_per_device != 1
        or execution.neural_query_topology.mode
        != "one_context_per_selected_device"
        or compiled.resource_performance_safety.fail_on_external_gpu_occupants
        or compiled.resource_performance_safety.gpu_max_allocation_fraction
        != float(Decimal("1") - args.gpu_minimum_free_fraction)
        or compiled.resource_performance_safety.gpu_minimum_headroom_bytes != 0
        or compiled.endpoint is not None
        or compiled.oracle_source is not None
    ):
        raise RuntimeError(
            "compiled local deployment differs from its requested lanes"
        )
    return compiled


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
        "stage1-profile",
        "query-profile",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--device",
        action="append",
        required=True,
        help="Logical CUDA locator; repeat once per selected device.",
    )
    parser.add_argument("--scope-workers-per-device", type=_positive, default=1)
    parser.add_argument("--max-parallel-owners", type=_positive)
    parser.add_argument("--cpu-budget", type=_positive, required=True)
    parser.add_argument(
        "--preflight-memory-budget",
        type=_positive,
        required=True,
    )
    parser.add_argument(
        "--preflight-owner-peak",
        type=_positive,
        required=True,
    )
    parser.add_argument("--preflight-lanes", type=_positive, required=True)
    parser.add_argument(
        "--embedding-batch-size",
        type=_positive,
        required=True,
    )
    parser.add_argument(
        "--gpu-minimum-free-fraction",
        type=_free_fraction,
        required=True,
        help=(
            "Admit selected GPUs when this fraction of aggregate VRAM is "
            "free; external process presence alone is permitted."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_profile(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
