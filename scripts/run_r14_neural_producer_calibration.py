#!/usr/bin/env python3
"""Bounded R14 production-kernel GPU calibration.

This command is deliberately not a miniature Stage 1 run.  It authenticates
one sealed exact-inner owner, asks production code to construct the canonical
upstream tasks and serial-prefix-conditioned downstream tasks, and executes
only an explicit optimizer-step prefix.  The prefix is useful for placement,
memory, and step-throughput calibration only.  The first complete Stage 1
owner remains the full scientific/text validation gate.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import copy
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import stat
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


R14_KERNEL_CALIBRATION_SCHEMA = (
    "production_r14_bounded_kernel_calibration_attestation_v8"
)
R14_KERNEL_RUN_SCHEMA = "production_r14_bounded_kernel_calibration_run_v8"
_PHASES = (
    "htr_nuisance",
    "htr_effect",
    "matched_pair_htr",
    "neural_inner_folds",
    "neural_final_banks",
)
_EXPECTED_TASK_COUNTS = {
    "htr_nuisance": 5,
    "htr_effect": 5,
    "matched_pair_htr": 5,
    "neural_inner_folds": 5,
    "neural_final_banks": 3,
}
_EXPECTED_CANDIDATE_PARALLELISM = {
    "htr_nuisance": 4,
    "htr_effect": 4,
    "matched_pair_htr": 4,
    "neural_inner_folds": 4,
    "neural_final_banks": 3,
}
_REQUIRED_GPU_SAMPLER_BACKEND = "persistent_pynvml_nvml_session_v1"
_CUDA_DEVICE = re.compile(r"^cuda:[0-9]+$")
_PREPARED_MANIFEST_NAME = "prepared_stage1_context_manifest.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _closed_json_state(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _closed_json_state(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _closed_json_state(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_closed_json_state(item) for item in value]
    if isinstance(value, np.ndarray):
        return _closed_json_state(value.tolist())
    if isinstance(value, np.generic):
        return _closed_json_state(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "calibration plan contains a value without a closed JSON identity: "
        f"{type(value).__name__}"
    )


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("calibration identity cannot hash an object array")
    return hashlib.sha256(
        array.dtype.str.encode("ascii")
        + _canonical_json(list(array.shape)).encode("ascii")
        + array.tobytes(order="C")
    ).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    before = os.lstat(path)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise ValueError(f"calibration file is not private regular data: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    after = os.lstat(path)
    identity = lambda value: (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )
    if identity(before) != identity(after) or size != int(after.st_size):
        raise RuntimeError(f"calibration file changed while hashing: {path}")
    return digest.hexdigest(), size


def _write_self_hashed_json(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    value = {**copy.deepcopy(dict(body)), "content_sha256": _sha256_json(body)}
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("calibration JSON write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    reopened = json.loads(path.read_text(encoding="utf-8"))
    reopened_body = {
        key: item for key, item in reopened.items() if key != "content_sha256"
    }
    if reopened != value or reopened["content_sha256"] != _sha256_json(
        reopened_body
    ):
        raise RuntimeError("calibration JSON changed after publication")
    return value


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed < 1 or str(parsed) != value.strip():
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a nonnegative integer"
        ) from exc
    if parsed < 0 or str(parsed) != value.strip():
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be finite and positive") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be finite and nonnegative"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def _open_unit_fraction(value: str) -> float:
    parsed = _positive_float(value)
    if parsed >= 1.0:
        raise argparse.ArgumentTypeError("value must be below one")
    return parsed


def _explicit_boolean(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("value must be explicitly true or false")


def _cuda_device(value: str) -> str:
    normalized = str(value).strip().lower()
    if _CUDA_DEVICE.fullmatch(normalized) is None:
        raise argparse.ArgumentTypeError("device must use the form cuda:N")
    return normalized


def _parallel_backend(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"threads", "processes"}:
        raise argparse.ArgumentTypeError(
            "parallel backend must be 'threads' or 'processes'"
        )
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one serial and two concurrent bounded optimizer-prefix "
            "calibrations for the R14 production GPU kernels."
        )
    )
    parser.add_argument("--prepared-context-manifest", required=True, type=Path)
    parser.add_argument("--source-snapshot-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--htr-model-path", required=True, type=Path)
    parser.add_argument(
        "--ordinary-full-byte-cache-fallback",
        required=True,
        type=_explicit_boolean,
    )
    parser.add_argument(
        "--candidate-device",
        action="append",
        dest="candidate_devices",
        required=True,
        type=_cuda_device,
    )
    parser.add_argument("--cpu-budget", required=True, type=_positive_integer)
    parser.add_argument(
        "--warmup-optimizer-steps",
        required=True,
        type=_positive_integer,
    )
    parser.add_argument(
        "--measured-optimizer-steps",
        required=True,
        type=_positive_integer,
    )
    parser.add_argument(
        "--prefix-relative-tolerance",
        required=True,
        type=_nonnegative_float,
    )
    parser.add_argument(
        "--prefix-absolute-tolerance",
        required=True,
        type=_nonnegative_float,
    )
    parser.add_argument(
        "--gpu-max-allocation-fraction",
        required=True,
        type=_open_unit_fraction,
    )
    parser.add_argument(
        "--gpu-minimum-headroom-bytes",
        required=True,
        type=_positive_integer,
    )
    parser.add_argument(
        "--gpu-sample-interval-seconds",
        required=True,
        type=_positive_float,
    )
    parser.add_argument(
        "--minimum-throughput-ratio",
        required=True,
        type=_positive_float,
    )
    parser.add_argument(
        "--candidate-slot-cap-per-device",
        required=True,
        type=_positive_integer,
    )
    parser.add_argument(
        "--htr-training-batch-size",
        required=True,
        type=_positive_integer,
    )
    parser.add_argument(
        "--htr-sentence-encoder-batch-size",
        required=True,
        type=_positive_integer,
    )
    parser.add_argument(
        "--htr-data-loader-workers",
        required=True,
        type=_nonnegative_integer,
    )
    parser.add_argument(
        "--htr-candidate-fold-parallelism",
        required=True,
        type=_positive_integer,
    )
    parser.add_argument(
        "--htr-fold-parallel-backend",
        required=True,
        type=_parallel_backend,
    )
    parser.add_argument(
        "--htr-reuse-tokenizer-and-chunk-plans",
        required=True,
        type=_explicit_boolean,
    )
    parser.add_argument(
        "--htr-chunk-plan-cache-max-entries",
        required=True,
        type=_nonnegative_integer,
    )
    parser.add_argument(
        "--htr-tokenized-chunk-cache-max-entries",
        required=True,
        type=_nonnegative_integer,
    )
    parser.add_argument(
        "--neural-candidate-inner-fold-parallelism",
        required=True,
        type=_positive_integer,
    )
    parser.add_argument(
        "--neural-fold-parallel-backend",
        required=True,
        type=_parallel_backend,
    )
    parser.add_argument(
        "--neural-candidate-bank-parallelism",
        required=True,
        type=_positive_integer,
    )
    parser.add_argument(
        "--neural-worker-cpu-threads",
        required=True,
        type=_positive_integer,
    )
    return parser


def _validate_main_args(args: argparse.Namespace) -> tuple[str, str]:
    devices = tuple(args.candidate_devices)
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("calibration requires exactly two distinct CUDA devices")
    if (
        not args.prepared_context_manifest.is_absolute()
        or args.prepared_context_manifest.name != _PREPARED_MANIFEST_NAME
    ):
        raise ValueError("prepared context must be an absolute canonical manifest")
    if not args.source_snapshot_root.is_absolute():
        raise ValueError("source snapshot root must be absolute")
    if not args.output_root.is_absolute() or not args.htr_model_path.is_absolute():
        raise ValueError("output root and HTR model path must be absolute")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError("calibration output root must be fresh")
    parent = args.output_root.parent
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise ValueError("calibration output parent must be canonical")
    if args.cpu_budget > 64:
        raise ValueError("R14 calibration CPU budget cannot exceed 64")
    if args.cpu_budget > len(os.sched_getaffinity(0)):
        raise ValueError("CPU budget exceeds the process affinity mask")
    if (
        args.warmup_optimizer_steps != 1
        or args.measured_optimizer_steps != 32
    ):
        raise ValueError(
            "R14 calibration requires one warm-up and 32 measured "
            "optimizer steps"
        )
    if args.minimum_throughput_ratio < 1.0:
        raise ValueError("minimum throughput ratio must be at least one")
    if args.gpu_sample_interval_seconds > 0.001:
        raise ValueError(
            "bounded-prefix GPU sampling interval must be <= 0.001s"
        )
    if args.candidate_slot_cap_per_device != 2:
        raise ValueError(
            "R14 candidate requires exactly two safe slots per device"
        )
    if args.htr_candidate_fold_parallelism != 4:
        raise ValueError(
            "R14 HTR calibration requires four active leases for five "
            "canonical tasks"
        )
    if args.neural_candidate_inner_fold_parallelism != 4:
        raise ValueError(
            "R14 neural-inner calibration requires four active leases for "
            "five canonical tasks"
        )
    if args.neural_candidate_bank_parallelism != 3:
        raise ValueError("R14 final-bank calibration requires three leases")
    if (
        args.htr_fold_parallel_backend != "processes"
        or args.neural_fold_parallel_backend != "processes"
    ):
        raise ValueError("parallel CUDA calibration requires spawned processes")
    if args.neural_worker_cpu_threads != 1:
        raise ValueError("neural calibration workers require one CPU thread")
    return devices  # type: ignore[return-value]


def _canonical_exact_inner_group(plan: Any) -> tuple[Any, tuple[Any, ...]]:
    matches = tuple(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if any(member.scope_kind == "exact_inner" for member in members)
    )
    if not matches:
        raise ValueError("prepared Stage 1 plan has no exact-inner owner")
    owner, members = matches[0]
    if any(
        candidate.canonical_index < owner.canonical_index
        for candidate, candidate_members in plan.physical_scope_groups
        if any(member.scope_kind == "exact_inner" for member in candidate_members)
    ):
        raise RuntimeError("exact-inner owner selection was not canonical")
    return owner, members


def _controls(
    args: argparse.Namespace,
    *,
    devices: Sequence[str],
    slots_per_device: int,
    baseline: bool,
) -> tuple[Any, Any]:
    from oci.inference.neural_query_operational_controls import (
        ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA,
        RoleNeutralNeuralQueryOperationalControls,
    )
    from oci.inference.stage1_htr_operational_controls import (
        ROLE_NEUTRAL_HTR_OPERATIONAL_CONTROLS_SCHEMA,
        RoleNeutralHTROperationalControls,
    )

    capacity = len(tuple(devices)) * int(slots_per_device)
    htr_parallelism = (
        1
        if baseline
        else min(int(args.htr_candidate_fold_parallelism), capacity)
    )
    neural_inner = (
        1
        if baseline
        else min(int(args.neural_candidate_inner_fold_parallelism), capacity)
    )
    neural_banks = (
        1
        if baseline
        else min(int(args.neural_candidate_bank_parallelism), capacity)
    )
    htr = RoleNeutralHTROperationalControls(
        training_batch_size=int(args.htr_training_batch_size),
        sentence_encoder_batch_size=int(args.htr_sentence_encoder_batch_size),
        data_loader_workers=int(args.htr_data_loader_workers),
        fold_parallelism=htr_parallelism,
        fold_parallel_backend=str(args.htr_fold_parallel_backend),
        fold_slots_per_device=int(slots_per_device),
        reuse_tokenizer_and_chunk_plans=bool(
            args.htr_reuse_tokenizer_and_chunk_plans
        ),
        chunk_plan_cache_max_entries=int(args.htr_chunk_plan_cache_max_entries),
        tokenized_chunk_cache_max_entries=int(
            args.htr_tokenized_chunk_cache_max_entries
        ),
        schema_version=ROLE_NEUTRAL_HTR_OPERATIONAL_CONTROLS_SCHEMA,
    )
    neural = RoleNeutralNeuralQueryOperationalControls(
        inner_fold_parallelism=neural_inner,
        fold_parallel_backend=str(args.neural_fold_parallel_backend),
        fold_slots_per_device=int(slots_per_device),
        bank_parallelism=neural_banks,
        worker_cpu_threads=int(args.neural_worker_cpu_threads),
        schema_version=ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA,
    )
    return htr, neural


def _authenticated_prefix_tolerances(
    bindings: Any,
) -> dict[str, dict[str, float]]:
    neural = bindings.neural_query_configuration
    values = {
        "htr_nuisance": {
            "relative_tolerance": float(
                bindings.htr.replay_relative_tolerance
            ),
            "absolute_tolerance": float(
                bindings.htr.replay_absolute_tolerance
            ),
        },
        "htr_effect": {
            "relative_tolerance": float(
                bindings.htr.replay_relative_tolerance
            ),
            "absolute_tolerance": float(
                bindings.htr.replay_absolute_tolerance
            ),
        },
        "matched_pair_htr": {
            "relative_tolerance": float(
                bindings.matched_pair.replay_relative_tolerance
            ),
            "absolute_tolerance": float(
                bindings.matched_pair.replay_absolute_tolerance
            ),
        },
        "neural_inner_folds": {
            "relative_tolerance": float(
                neural["replay_relative_tolerance"]
            ),
            "absolute_tolerance": float(
                neural["replay_absolute_tolerance"]
            ),
        },
        "neural_final_banks": {
            "relative_tolerance": float(
                neural["replay_relative_tolerance"]
            ),
            "absolute_tolerance": float(
                neural["replay_absolute_tolerance"]
            ),
        },
    }
    if any(
        not math.isfinite(row[key]) or row[key] < 0.0
        for row in values.values()
        for key in ("relative_tolerance", "absolute_tolerance")
    ):
        raise ValueError("authenticated replay tolerances are invalid")
    return values


class _CapturedCanonicalTasks(BaseException):
    def __init__(self, tasks: Sequence[Any], *, phase: str) -> None:
        super().__init__(phase)
        self.tasks = tuple(tasks)
        self.phase = str(phase)


@dataclass(frozen=True)
class _PrefixTask:
    phase: str
    canonical_index: int
    canonical_task: Any
    canonical_identity: Mapping[str, Any]
    ready_barrier: Any
    warmup_steps: int
    measured_steps: int
    parameter_bundle_path: str | None
    parameter_bundle_relative_path: str | None


@dataclass(frozen=True)
class _PreparedMatchedTask:
    canonical_task: Any
    complete_input_plan_content_sha256: str


@dataclass(frozen=True)
class _PreparedHTREffectTask:
    canonical_task: Any
    source_nuisance_oof_e_sha256: str
    source_nuisance_oof_m_sha256: str


@dataclass(frozen=True)
class _PreparedNeuralInnerTask:
    canonical_task: Any
    train_e: np.ndarray
    train_m: np.ndarray
    validation_e: np.ndarray
    validation_m: np.ndarray
    nuisance_identity: Mapping[str, Any]
    complete_input_plan_content_sha256: str


@dataclass(frozen=True)
class _PreparedNeuralFinalTask:
    canonical_task: Any
    complete_input_plan_content_sha256: str


class _HTREffectFixture(NamedTuple):
    tasks: tuple[Any, ...]
    preparation: Mapping[str, Any]


class _NeuralFinalFixture(NamedTuple):
    tasks: tuple[Any, ...]
    preparation: Mapping[str, Any]


class _PrefixFinished(BaseException):
    pass


@dataclass
class _StepObservation:
    warmup_steps: int
    measured_steps: int
    losses: list[Any] = dataclasses.field(default_factory=list)
    completed_steps: int = 0
    measured_started_monotonic_ns: int | None = None
    measured_finished_monotonic_ns: int | None = None
    terminal_shapes: list[list[int]] = dataclasses.field(default_factory=list)
    terminal_dtypes: list[str] = dataclasses.field(default_factory=list)
    terminal_samples: list[float] = dataclasses.field(default_factory=list)
    terminal_sha256s: list[str] = dataclasses.field(default_factory=list)
    terminal_element_counts: list[int] = dataclasses.field(default_factory=list)
    terminal_all_finite: list[bool] = dataclasses.field(default_factory=list)
    terminal_group_indices: list[int] = dataclasses.field(default_factory=list)
    terminal_parameter_indices_within_group: list[int] = (
        dataclasses.field(default_factory=list)
    )
    terminal_arrays: list[np.ndarray] = dataclasses.field(default_factory=list)
    initial_full_tensor: np.ndarray | None = None
    terminal_full_tensor: np.ndarray | None = None
    ready_wait_started_monotonic_ns: int | None = None
    ready_wait_finished_monotonic_ns: int | None = None
    completed_warmup_steps_at_interval_start: int | None = None
    optimizer_state_verified_monotonic_ns: int | None = None
    optimizer_state_at_interval_start: Mapping[str, Any] | None = None
    optimizer_state_at_interval_finish: Mapping[str, Any] | None = None
    optimizer_state_finish_verified_monotonic_ns: int | None = None
    optimizer_state_persistence_observed: bool | None = None


def _adamw_state_observation(
    optimizer: Any,
    *,
    expected_step: int,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    import torch

    parameter_rows: list[tuple[int, Any, bool]] = []
    seen: set[int] = set()
    for group_index, group in enumerate(optimizer.param_groups):
        amsgrad = bool(group.get("amsgrad", False))
        for parameter in group["params"]:
            identity = id(parameter)
            if identity in seen:
                raise RuntimeError(
                    "AdamW state reused one parameter across optimizer groups"
                )
            seen.add(identity)
            parameter_rows.append((group_index, parameter, amsgrad))
    if not parameter_rows:
        raise RuntimeError("AdamW prefix optimizer has no parameters")

    tensor_count = 0
    tensor_bytes = 0
    state_object_count = 0
    state_parameter_count = 0
    stateless_parameter_count = 0
    layout: list[dict[str, Any]] = []
    storage_signature: list[Any] = []
    for parameter_index, (group_index, parameter, amsgrad) in enumerate(
        parameter_rows
    ):
        if parameter not in optimizer.state:
            if parameter.grad is not None:
                raise RuntimeError(
                    "AdamW parameter has a gradient but no optimizer state"
                )
            stateless_parameter_count += 1
            layout.append(
                {
                    "parameter_index": parameter_index,
                    "group_index": group_index,
                    "amsgrad": amsgrad,
                    "parameter_shape": [
                        int(item) for item in parameter.shape
                    ],
                    "parameter_dtype": str(parameter.dtype),
                    "parameter_device_type": parameter.device.type,
                    "state_status": "stateless_no_gradient",
                    "gradient_present": False,
                    "optimizer_state_present": False,
                    "entries": [],
                }
            )
            storage_signature.append(
                (parameter_index, "stateless_no_gradient")
            )
            continue
        parameter_state = optimizer.state[parameter]
        if not isinstance(parameter_state, Mapping) or not parameter_state:
            raise RuntimeError(
                "AdamW optimizer contains an empty or malformed state object"
            )
        gradient = parameter.grad
        if (
            not isinstance(gradient, torch.Tensor)
            or tuple(gradient.shape) != tuple(parameter.shape)
            or gradient.dtype != parameter.dtype
            or gradient.device != parameter.device
            or not bool(torch.isfinite(gradient).all().to(device="cpu"))
        ):
            raise RuntimeError(
                "stateful AdamW parameter omitted its finite boundary gradient"
            )
        state_parameter_count += 1
        required = {"step", "exp_avg", "exp_avg_sq"}
        if amsgrad:
            required.add("max_exp_avg_sq")
        if set(parameter_state) != required:
            raise RuntimeError(
                "AdamW optimizer-state object layout changed"
            )
        entries: list[dict[str, Any]] = []
        for key in sorted(required):
            value = parameter_state[key]
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    "AdamW optimizer state must contain only tensors"
                )
            if not bool(torch.isfinite(value).all().to(device="cpu")):
                raise RuntimeError(
                    "AdamW optimizer state contains a non-finite tensor"
                )
            if key == "step":
                if value.numel() != 1:
                    raise RuntimeError("AdamW step state is not scalar")
                observed_step = float(value.detach().to(device="cpu").item())
                if observed_step != float(expected_step):
                    raise RuntimeError(
                        "AdamW state step does not match the prefix boundary"
                    )
            elif (
                tuple(value.shape) != tuple(parameter.shape)
                or value.dtype != parameter.dtype
                or value.device != parameter.device
            ):
                raise RuntimeError(
                    "AdamW moment state changed parameter shape/device/dtype"
                )
            storage = value.untyped_storage()
            nbytes = int(value.numel()) * int(value.element_size())
            tensor_count += 1
            tensor_bytes += nbytes
            state_object_count += 1
            entries.append(
                {
                    "key": key,
                    "kind": "tensor",
                    "shape": [int(item) for item in value.shape],
                    "stride": [int(item) for item in value.stride()],
                    "dtype": str(value.dtype),
                    "device_type": value.device.type,
                    "storage_offset": int(value.storage_offset()),
                    "storage_nbytes": int(storage.nbytes()),
                }
            )
            storage_signature.append(
                (
                    parameter_index,
                    key,
                    int(storage.data_ptr()),
                    int(storage.nbytes()),
                    int(value.storage_offset()),
                    tuple(int(item) for item in value.stride()),
                )
            )
        layout.append(
            {
                "parameter_index": parameter_index,
                "group_index": group_index,
                "amsgrad": amsgrad,
                "parameter_shape": [
                    int(item) for item in parameter.shape
                ],
                "parameter_dtype": str(parameter.dtype),
                "parameter_device_type": parameter.device.type,
                "state_status": "stateful_with_finite_gradient",
                "gradient_present": True,
                "optimizer_state_present": True,
                "entries": entries,
            }
        )
    layout_json = _canonical_json(layout)
    summary = {
        "schema_version": "adamw_optimizer_state_boundary_observation_v2",
        "expected_optimizer_step": int(expected_step),
        "optimizer_parameter_count": len(parameter_rows),
        "state_parameter_count": state_parameter_count,
        "stateless_parameter_count": stateless_parameter_count,
        "state_object_count": state_object_count,
        "state_tensor_count": tensor_count,
        "state_tensor_bytes": tensor_bytes,
        "all_optimizer_parameters_classified": (
            state_parameter_count + stateless_parameter_count
            == len(parameter_rows)
        ),
        "all_stateless_parameters_have_no_gradient": True,
        "all_stateless_parameters_have_no_optimizer_state": True,
        "all_stateful_parameters_have_finite_gradients": True,
        "all_required_state_keys_observed": True,
        "all_state_tensors_finite": True,
        "object_layout_canonical_json": layout_json,
        "object_layout_sha256": hashlib.sha256(
            layout_json.encode("utf-8")
        ).hexdigest(),
    }
    if tensor_count <= 0 or tensor_bytes <= 0:
        raise RuntimeError("AdamW optimizer state has no tensor storage")
    return summary, tuple(storage_signature)


def _publish_parameter_bundle(
    path: Path,
    *,
    relative_path: str,
    arrays: Sequence[np.ndarray],
    group_indices: Sequence[int],
    parameter_indices_within_group: Sequence[int],
) -> dict[str, Any]:
    if (
        not arrays
        or len(group_indices) != len(arrays)
        or len(parameter_indices_within_group) != len(arrays)
        or Path(relative_path).is_absolute()
        or ".." in Path(relative_path).parts
    ):
        raise ValueError("parameter bundle path or inventory is invalid")
    parent = path.parent
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or parent.resolve(strict=True) != parent
        or path.exists()
        or path.is_symlink()
    ):
        raise ValueError("parameter bundle target is not fresh canonical data")
    temporary = parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    digest = hashlib.sha256()
    offset = 0
    inventory: list[dict[str, Any]] = []
    published = False
    try:
        for parameter_index, source in enumerate(arrays):
            array = np.ascontiguousarray(np.asarray(source))
            payload = memoryview(array).cast("B")
            nbytes = len(payload)
            inventory.append(
                {
                    "parameter_index": parameter_index,
                    "group_index": int(group_indices[parameter_index]),
                    "parameter_index_within_group": int(
                        parameter_indices_within_group[parameter_index]
                    ),
                    "offset_bytes": offset,
                    "nbytes": nbytes,
                }
            )
            while payload:
                written = os.write(descriptor, payload)
                if written < 1:
                    raise OSError("parameter bundle write made no progress")
                digest.update(payload[:written])
                payload = payload[written:]
                offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        published = True
        directory_descriptor = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            os.unlink(temporary)
    if not published:
        raise RuntimeError("parameter bundle was not atomically published")
    observed_sha256, observed_size = _sha256_file(path)
    if observed_sha256 != digest.hexdigest() or observed_size != offset:
        raise RuntimeError("parameter bundle changed after publication")
    return {
        "schema_version": "packed_raw_optimizer_parameter_bundle_v1",
        "relative_path": relative_path,
        "sha256": observed_sha256,
        "size_bytes": observed_size,
        "parameter_count": len(inventory),
        "inventory": inventory,
        "o_exclusive_private_write": True,
        "fsync_before_atomic_publish": True,
    }


def _terminal_parameter_state(
    optimizer: Any,
    *,
    capture_full_single_tensor: bool,
) -> tuple[
    list[list[int]],
    list[str],
    list[float],
    list[str],
    list[int],
    list[bool],
    list[int],
    list[int],
    list[np.ndarray],
    np.ndarray | None,
]:
    rows = [
        (group_index, parameter_index, parameter.detach())
        for group_index, group in enumerate(optimizer.param_groups)
        for parameter_index, parameter in enumerate(group["params"])
    ]
    shapes: list[list[int]] = []
    dtypes: list[str] = []
    samples: list[float] = []
    sha256s: list[str] = []
    element_counts: list[int] = []
    all_finite: list[bool] = []
    arrays: list[np.ndarray] = []
    group_indices: list[int] = []
    parameter_indices_within_group: list[int] = []
    for group_index, parameter_index, tensor in rows:
        group_indices.append(group_index)
        parameter_indices_within_group.append(parameter_index)
        array = tensor.to(device="cpu").contiguous().numpy().copy()
        arrays.append(array)
        shapes.append([int(value) for value in array.shape])
        dtypes.append(array.dtype.str)
        sha256s.append(_array_sha256(array))
        element_counts.append(int(array.size))
        all_finite.append(bool(np.isfinite(array).all()))
        flattened = array.reshape(-1)
        if flattened.size:
            positions = np.linspace(
                0,
                flattened.size - 1,
                num=min(32, flattened.size),
                dtype=np.int64,
            )
            samples.extend(
                np.asarray(flattened[positions], dtype=np.float64).tolist()
            )
    full: np.ndarray | None = None
    if capture_full_single_tensor:
        if len(rows) != 1:
            raise RuntimeError("neural prefix optimizer changed parameter topology")
        full = rows[0][2].to(device="cpu").contiguous().numpy().copy()
    return (
        shapes,
        dtypes,
        samples,
        sha256s,
        element_counts,
        all_finite,
        group_indices,
        parameter_indices_within_group,
        arrays,
        full,
    )


class _ObserveAdamWPrefix:
    """Observe the real production AdamW loop and stop at one prefix."""

    def __init__(
        self,
        *,
        device: str,
        ready_barrier: Any,
        warmup_steps: int,
        measured_steps: int,
        stop_on_next_step: bool,
        capture_full_single_tensor: bool,
        capture_exact_parameter_hashes: bool,
        parameter_bundle_path: str | None,
        parameter_bundle_relative_path: str | None,
        continue_after_terminal_projection: bool = False,
    ) -> None:
        self.device = str(device)
        self.ready_barrier = ready_barrier
        self.stop_on_next_step = bool(stop_on_next_step)
        self.capture_full_single_tensor = bool(capture_full_single_tensor)
        self.capture_exact_parameter_hashes = bool(
            capture_exact_parameter_hashes
        )
        self.continue_after_terminal_projection = bool(
            continue_after_terminal_projection
        )
        self.parameter_bundle_path = (
            None
            if parameter_bundle_path is None
            else Path(parameter_bundle_path)
        )
        self.parameter_bundle_relative_path = (
            None
            if parameter_bundle_relative_path is None
            else str(parameter_bundle_relative_path)
        )
        if self.capture_exact_parameter_hashes != (
            self.parameter_bundle_path is not None
            and self.parameter_bundle_relative_path is not None
        ):
            raise ValueError(
                "complete parameter capture requires one closed bundle target"
            )
        self.observation = _StepObservation(
            warmup_steps=int(warmup_steps),
            measured_steps=int(measured_steps),
        )
        self._original_step: Any = None
        self._original_zero_grad: Any = None
        self._original_backward: Any = None
        self._original_copy: Any = None
        self._terminal_optimizer: Any = None
        self._terminal_projection_copies = 0
        self._optimizer_storage_at_interval_start: tuple[Any, ...] | None = (
            None
        )

    @property
    def total_steps(self) -> int:
        return (
            self.observation.warmup_steps
            + self.observation.measured_steps
        )

    def _synchronize(self) -> None:
        import torch

        if self.device.startswith("cuda:"):
            torch.cuda.synchronize(self.device)

    def _capture(self, optimizer: Any) -> None:
        (
            finish_state,
            finish_storage,
        ) = _adamw_state_observation(
            optimizer,
            expected_step=self.total_steps,
        )
        start_state = self.observation.optimizer_state_at_interval_start
        if (
            start_state is None
            or self._optimizer_storage_at_interval_start is None
        ):
            raise RuntimeError(
                "AdamW prefix omitted its optimizer-state start boundary"
            )
        persistence = bool(
            start_state["optimizer_parameter_count"]
            == finish_state["optimizer_parameter_count"]
            and start_state["state_parameter_count"]
            == finish_state["state_parameter_count"]
            and start_state["state_object_count"]
            == finish_state["state_object_count"]
            and start_state["state_tensor_count"]
            == finish_state["state_tensor_count"]
            and start_state["state_tensor_bytes"]
            == finish_state["state_tensor_bytes"]
            and start_state["object_layout_sha256"]
            == finish_state["object_layout_sha256"]
            and self._optimizer_storage_at_interval_start == finish_storage
        )
        if not persistence:
            raise RuntimeError(
                "AdamW optimizer state/storage topology did not persist"
            )
        self.observation.optimizer_state_at_interval_finish = finish_state
        self.observation.optimizer_state_finish_verified_monotonic_ns = (
            time.monotonic_ns()
        )
        self.observation.optimizer_state_persistence_observed = persistence
        (
            self.observation.terminal_shapes,
            self.observation.terminal_dtypes,
            self.observation.terminal_samples,
            self.observation.terminal_sha256s,
            self.observation.terminal_element_counts,
            self.observation.terminal_all_finite,
            self.observation.terminal_group_indices,
            self.observation.terminal_parameter_indices_within_group,
            self.observation.terminal_arrays,
            self.observation.terminal_full_tensor,
        ) = _terminal_parameter_state(
            optimizer,
            capture_full_single_tensor=self.capture_full_single_tensor,
        )

    def __enter__(self) -> "_ObserveAdamWPrefix":
        import torch

        if self._original_step is not None:
            raise RuntimeError("AdamW prefix observer cannot be entered twice")
        self._original_step = torch.optim.AdamW.step
        self._original_zero_grad = torch.optim.AdamW.zero_grad
        self._original_backward = torch.Tensor.backward
        self._original_copy = torch.Tensor.copy_
        observer = self

        def backward(tensor: Any, *args: Any, **kwargs: Any) -> Any:
            if tensor.numel() == 1:
                # Retain only the detached scalar. Converting it to a host
                # float here would synchronize every optimizer iteration and
                # turn the benchmark into an instrumentation microbenchmark.
                observer.observation.losses.append(tensor.detach())
            return observer._original_backward(tensor, *args, **kwargs)

        def zero_grad(optimizer: Any, *args: Any, **kwargs: Any) -> Any:
            state = observer.observation
            if (
                observer.capture_full_single_tensor
                and state.completed_steps == 0
                and state.initial_full_tensor is None
            ):
                parameters = [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ]
                if len(parameters) != 1:
                    raise RuntimeError(
                        "neural prefix optimizer changed initial parameter topology"
                    )
                initial = (
                    parameters[0]
                    .detach()
                    .to(device="cpu")
                    .contiguous()
                    .numpy()
                    .copy()
                )
                if not np.isfinite(initial).all():
                    raise RuntimeError(
                        "neural prefix optimizer has non-finite initial queries"
                    )
                state.initial_full_tensor = initial
            if (
                state.completed_steps == state.warmup_steps
                and state.measured_started_monotonic_ns is None
            ):
                state.completed_warmup_steps_at_interval_start = (
                    state.completed_steps
                )
                (
                    state.optimizer_state_at_interval_start,
                    observer._optimizer_storage_at_interval_start,
                ) = _adamw_state_observation(
                    optimizer,
                    expected_step=state.warmup_steps,
                )
                state.optimizer_state_verified_monotonic_ns = (
                    time.monotonic_ns()
                )
                state.ready_wait_started_monotonic_ns = time.monotonic_ns()
                observer.ready_barrier.wait(timeout=900.0)
                state.ready_wait_finished_monotonic_ns = time.monotonic_ns()
                observer._synchronize()
                # Start before zeroing gradients, forward, backward, clipping,
                # and AdamW so every operation in the first measured
                # optimizer iteration is included.
                state.measured_started_monotonic_ns = time.monotonic_ns()
            return observer._original_zero_grad(
                optimizer,
                *args,
                **kwargs,
            )

        def step(optimizer: Any, *args: Any, **kwargs: Any) -> Any:
            state = observer.observation
            if observer.stop_on_next_step and state.completed_steps == observer.total_steps:
                raise RuntimeError(
                    "neural optimizer entered another step before its "
                    "production normalization/projection boundary"
                )
            if (
                state.completed_steps >= state.warmup_steps
                and state.measured_started_monotonic_ns is None
            ):
                raise RuntimeError(
                    "production optimizer skipped the measured "
                    "iteration's zero_grad boundary"
                )
            result = observer._original_step(optimizer, *args, **kwargs)
            state.completed_steps += 1
            if (
                observer.stop_on_next_step
                and state.completed_steps == observer.total_steps
            ):
                observer._terminal_optimizer = optimizer
            if (
                not observer.stop_on_next_step
                and state.completed_steps == observer.total_steps
            ):
                observer._synchronize()
                state.measured_finished_monotonic_ns = time.monotonic_ns()
                observer._capture(optimizer)
                raise _PrefixFinished()
            return result

        def copy_(tensor: Any, source: Any, *args: Any, **kwargs: Any) -> Any:
            result = observer._original_copy(
                tensor,
                source,
                *args,
                **kwargs,
            )
            if (
                observer.stop_on_next_step
                and observer.observation.completed_steps == observer.total_steps
            ):
                observer._terminal_projection_copies += 1
                # The production neural kernel performs two query copies:
                # unit normalization followed by the bounded-drift
                # projection. Capture only after both have completed.
                if observer._terminal_projection_copies == 2:
                    if observer._terminal_optimizer is None:
                        raise RuntimeError(
                            "neural prefix lost its terminal optimizer"
                        )
                    observer._synchronize()
                    observer.observation.measured_finished_monotonic_ns = (
                        time.monotonic_ns()
                    )
                    observer._capture(observer._terminal_optimizer)
                    if not observer.continue_after_terminal_projection:
                        raise _PrefixFinished()
            return result

        torch.optim.AdamW.zero_grad = zero_grad
        torch.Tensor.backward = backward
        torch.optim.AdamW.step = step
        torch.Tensor.copy_ = copy_
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        import torch

        torch.optim.AdamW.step = self._original_step
        torch.optim.AdamW.zero_grad = self._original_zero_grad
        torch.Tensor.backward = self._original_backward
        torch.Tensor.copy_ = self._original_copy
        self._original_step = None
        self._original_zero_grad = None
        self._original_backward = None
        self._original_copy = None
        self._terminal_optimizer = None
        self._optimizer_storage_at_interval_start = None

    def result(self) -> dict[str, Any]:
        state = self.observation
        total = self.total_steps
        if (
            state.completed_steps != total
            or state.measured_started_monotonic_ns is None
            or state.measured_finished_monotonic_ns is None
            or state.measured_finished_monotonic_ns
            <= state.measured_started_monotonic_ns
            or len(state.losses) < total
            or not state.terminal_shapes
            or len(state.terminal_sha256s) != len(state.terminal_shapes)
            or len(state.terminal_element_counts) != len(state.terminal_shapes)
            or len(state.terminal_all_finite) != len(state.terminal_shapes)
            or len(state.terminal_group_indices) != len(state.terminal_shapes)
            or len(state.terminal_parameter_indices_within_group)
            != len(state.terminal_shapes)
            or (
                self.capture_full_single_tensor
                and state.initial_full_tensor is None
            )
            or state.completed_warmup_steps_at_interval_start
            != state.warmup_steps
            or state.optimizer_state_verified_monotonic_ns is None
            or state.optimizer_state_at_interval_start is None
            or state.optimizer_state_at_interval_finish is None
            or state.optimizer_state_finish_verified_monotonic_ns is None
            or state.optimizer_state_persistence_observed is not True
            or state.ready_wait_started_monotonic_ns is None
            or state.ready_wait_finished_monotonic_ns is None
            or not (
                state.optimizer_state_verified_monotonic_ns
                <= state.ready_wait_started_monotonic_ns
                <= state.ready_wait_finished_monotonic_ns
                <= state.measured_started_monotonic_ns
                < state.measured_finished_monotonic_ns
                <= state.optimizer_state_finish_verified_monotonic_ns
            )
        ):
            raise RuntimeError("production optimizer did not complete the bounded prefix")
        result = {
            "warmup_optimizer_steps": state.warmup_steps,
            "measured_optimizer_steps": state.measured_steps,
            "measured_started_monotonic_ns": (
                state.measured_started_monotonic_ns
            ),
            "measured_finished_monotonic_ns": (
                state.measured_finished_monotonic_ns
            ),
            "ready_wait_started_monotonic_ns": (
                state.ready_wait_started_monotonic_ns
            ),
            "ready_wait_finished_monotonic_ns": (
                state.ready_wait_finished_monotonic_ns
            ),
            "completed_warmup_optimizer_steps_at_interval_start": (
                state.completed_warmup_steps_at_interval_start
            ),
            "optimizer_state_verified_monotonic_ns": (
                state.optimizer_state_verified_monotonic_ns
            ),
            "optimizer_state_at_interval_start": copy.deepcopy(
                dict(state.optimizer_state_at_interval_start)
            ),
            "optimizer_state_at_interval_finish": copy.deepcopy(
                dict(state.optimizer_state_at_interval_finish)
            ),
            "optimizer_state_finish_verified_monotonic_ns": (
                state.optimizer_state_finish_verified_monotonic_ns
            ),
            "optimizer_state_persistence_observed": (
                state.optimizer_state_persistence_observed
            ),
            "loss_prefix": [
                float(value.to(device="cpu"))
                for value in state.losses[
                    state.warmup_steps : total
                ]
            ],
            "terminal_parameter_shapes": state.terminal_shapes,
            "terminal_parameter_dtypes": state.terminal_dtypes,
            "terminal_parameter_samples": state.terminal_samples,
        }
        if not all(state.terminal_all_finite):
            raise RuntimeError(
                "optimizer prefix emitted a non-finite terminal parameter"
            )
        if self.capture_exact_parameter_hashes:
            bundle = _publish_parameter_bundle(
                self.parameter_bundle_path,
                relative_path=self.parameter_bundle_relative_path,
                arrays=state.terminal_arrays,
                group_indices=state.terminal_group_indices,
                parameter_indices_within_group=(
                    state.terminal_parameter_indices_within_group
                ),
            )
            result.update(
                {
                    "terminal_parameter_count": len(
                        state.terminal_sha256s
                    ),
                    "terminal_parameter_sha256s": state.terminal_sha256s,
                    "terminal_parameter_element_counts": (
                        state.terminal_element_counts
                    ),
                    "terminal_parameter_all_finite": (
                        state.terminal_all_finite
                    ),
                    "terminal_parameter_group_indices": (
                        state.terminal_group_indices
                    ),
                    "terminal_parameter_indices_within_group": (
                        state.terminal_parameter_indices_within_group
                    ),
                    "terminal_all_parameters_finite": True,
                    "terminal_parameter_bundle": bundle,
                }
            )
        if state.terminal_full_tensor is not None:
            if (
                state.initial_full_tensor is None
                or state.initial_full_tensor.shape
                != state.terminal_full_tensor.shape
                or state.initial_full_tensor.dtype
                != state.terminal_full_tensor.dtype
            ):
                raise RuntimeError(
                    "neural prefix initial and terminal query tensors differ"
                )
            result["initial_query_tensor"] = (
                state.initial_full_tensor.tolist()
            )
            result["initial_query_tensor_dtype"] = (
                state.initial_full_tensor.dtype.str
            )
            result["initial_query_tensor_shape"] = [
                int(value) for value in state.initial_full_tensor.shape
            ]
            result["initial_query_tensor_sha256"] = _array_sha256(
                state.initial_full_tensor
            )
            result["terminal_query_tensor"] = (
                state.terminal_full_tensor.tolist()
            )
            result["terminal_query_tensor_dtype"] = (
                state.terminal_full_tensor.dtype.str
            )
            result["terminal_query_tensor_shape"] = [
                int(value) for value in state.terminal_full_tensor.shape
            ]
        numerical = [
            *result["loss_prefix"],
            *result["terminal_parameter_samples"],
        ]
        if state.terminal_full_tensor is not None:
            numerical.extend(
                np.asarray(
                    state.initial_full_tensor,
                    dtype=np.float64,
                ).reshape(-1).tolist()
            )
            numerical.extend(
                np.asarray(
                    state.terminal_full_tensor,
                    dtype=np.float64,
                ).reshape(-1).tolist()
            )
        if not np.isfinite(np.asarray(numerical, dtype=np.float64)).all():
            raise RuntimeError("optimizer prefix emitted non-finite terminal state")
        return result


def _htr_batch_hashes(task: Any, *, steps: int) -> list[str]:
    from oci.inference import role_neutral_htr_group_execution as htr

    result: list[str] = []
    epoch = 0
    while len(result) < int(steps):
        for positions in htr._batch_positions(
            np.asarray(task.fit_positions, dtype=np.int64),
            batch_size=int(task.config.batch_size),
            seed=int(task.model_seed),
            epoch=epoch,
        ):
            result.append(_array_sha256(positions))
            if len(result) == int(steps):
                break
        epoch += 1
        if epoch > int(task.config.nuisance_epochs):
            raise RuntimeError("HTR production schedule is shorter than prefix")
    return result


def _htr_effect_batch_hashes(task: Any, *, steps: int) -> list[str]:
    from oci.inference import role_neutral_htr_group_execution as htr

    result: list[str] = []
    epoch = 0
    while len(result) < int(steps):
        for positions in htr._batch_positions(
            np.asarray(task.eligible_fit_positions, dtype=np.int64),
            batch_size=int(task.config.batch_size),
            seed=int(task.model_seed),
            epoch=epoch,
        ):
            result.append(_array_sha256(positions))
            if len(result) == int(steps):
                break
        epoch += 1
        if epoch > int(task.config.effect_epochs):
            raise RuntimeError("HTR effect schedule is shorter than prefix")
    return result


def _run_htr_prefix(task: _PrefixTask, device: str) -> Mapping[str, Any]:
    import torch
    from oci.inference import role_neutral_htr_group_execution as htr

    canonical = task.canonical_task
    values, row_ids, coverage, reusable = htr._resolve_fold_text_authority(
        canonical.text_authority
    )
    fit_positions = np.asarray(canonical.fit_positions, dtype=np.int64)
    torch_device = torch.device(device)
    htr._set_model_seed(int(canonical.model_seed), torch_device)
    extractor, _attestation = htr._prepare_fold_extractor(
        config=canonical.config,
        model_marker=canonical.model_marker,
        device=torch_device,
        texts=values,
        row_ids=row_ids,
        fit_positions=fit_positions,
        coverage=coverage,
        reusable_plan=reusable,
        operational_controls=canonical.operational_controls,
        preflight_complete_text=False,
    )
    model = htr._NuisanceNet(
        extractor=extractor,
        hidden_dim=canonical.config.hidden_dim,
        outcome_type=canonical.config.outcome_type,
        head_depth=canonical.config.nuisance_head_depth,
        head_activation=canonical.config.nuisance_head_activation,
        head_dropout=canonical.config.nuisance_head_dropout,
        head_layer_norm=canonical.config.nuisance_head_layer_norm,
        head_bias=canonical.config.nuisance_head_bias,
    ).to(torch_device)
    observer = _ObserveAdamWPrefix(
        device=device,
        ready_barrier=task.ready_barrier,
        warmup_steps=task.warmup_steps,
        measured_steps=task.measured_steps,
        stop_on_next_step=False,
        capture_full_single_tensor=False,
        capture_exact_parameter_hashes=True,
        parameter_bundle_path=task.parameter_bundle_path,
        parameter_bundle_relative_path=(
            task.parameter_bundle_relative_path
        ),
    )
    try:
        try:
            with observer:
                htr._train_nuisance(
                    model,
                    texts=values,
                    treatment=canonical.treatment,
                    outcome=canonical.outcome,
                    positions=fit_positions,
                    config=canonical.config,
                    seed=canonical.model_seed,
                    device=torch_device,
                )
        except _PrefixFinished:
            pass
        prefix = observer.result()
        validation_positions = np.asarray(
            canonical.validation_positions,
            dtype=np.int64,
        )
        fit_raw_e, fit_raw_m = htr._predict_model(
            model,
            [values[int(position)] for position in fit_positions],
            kind="nuisance",
            outcome_type=canonical.config.outcome_type,
            batch_size=canonical.config.prediction_batch_size,
        )
        validation_raw_e, validation_raw_m = htr._predict_model(
            model,
            [values[int(position)] for position in validation_positions],
            kind="nuisance",
            outcome_type=canonical.config.outcome_type,
            batch_size=canonical.config.prediction_batch_size,
        )
        propensity_calibrator = htr.BinaryProbabilityCalibrator.fit(
            fit_raw_e,
            canonical.treatment[fit_positions],
            method=canonical.config.nuisance_calibration,
        )
        outcome_calibrator = htr.BinaryProbabilityCalibrator.fit(
            fit_raw_m,
            canonical.outcome[fit_positions],
            method=canonical.config.nuisance_calibration,
        )
        validation_e = np.ascontiguousarray(
            propensity_calibrator.transform(validation_raw_e),
            dtype=np.float64,
        )
        validation_m = np.ascontiguousarray(
            outcome_calibrator.transform(validation_raw_m),
            dtype=np.float64,
        )
        if (
            validation_e.shape != validation_positions.shape
            or validation_m.shape != validation_positions.shape
            or not np.isfinite(validation_e).all()
            or not np.isfinite(validation_m).all()
        ):
            raise RuntimeError(
                "HTR nuisance prefix could not produce complete calibrated OOF rows"
            )
        prefix = {
            **dict(prefix),
            "prefix_conditioned_nuisance_oof": {
                "validation_positions": validation_positions.tolist(),
                "validation_e_hat": validation_e.tolist(),
                "validation_e_hat_dtype": validation_e.dtype.str,
                "validation_m_hat": validation_m.tolist(),
                "validation_m_hat_dtype": validation_m.dtype.str,
            },
        }
        return {
            "phase": task.phase,
            "canonical_index": task.canonical_index,
            "canonical_identity": copy.deepcopy(dict(task.canonical_identity)),
            "batch_position_sha256s": _htr_batch_hashes(
                canonical,
                steps=task.warmup_steps + task.measured_steps,
            ),
            "prefix_output": prefix,
            "prefix_conditioned_nuisance_oof": {
                "validation_positions": validation_positions.tolist(),
                "validation_e_hat": validation_e.tolist(),
                "validation_e_hat_dtype": validation_e.dtype.str,
                "validation_e_hat_sha256": _array_sha256(validation_e),
                "validation_m_hat": validation_m.tolist(),
                "validation_m_hat_dtype": validation_m.dtype.str,
                "validation_m_hat_sha256": _array_sha256(validation_m),
                "production_probability_calibration_applied": True,
                "complete_fit_and_validation_prediction_applied": True,
            },
            "complete_plan_authenticated": True,
            "complete_text_optimizer_execution_claimed": False,
        }
    finally:
        del model
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()


def _run_htr_effect_prefix(
    task: _PrefixTask,
    device: str,
) -> Mapping[str, Any]:
    import torch
    from oci.inference import role_neutral_htr_group_execution as htr

    prepared = task.canonical_task
    if not isinstance(prepared, _PreparedHTREffectTask):
        raise TypeError("HTR effect prefix requires one prepared effect task")
    canonical = prepared.canonical_task
    values, row_ids, coverage, reusable = htr._resolve_fold_text_authority(
        canonical.text_authority
    )
    fit_positions = np.asarray(
        canonical.eligible_fit_positions,
        dtype=np.int64,
    )
    torch_device = torch.device(device)
    htr._set_model_seed(int(canonical.model_seed), torch_device)
    extractor, _attestation = htr._prepare_fold_extractor(
        config=canonical.config,
        model_marker=canonical.model_marker,
        device=torch_device,
        texts=values,
        row_ids=row_ids,
        fit_positions=fit_positions,
        coverage=coverage,
        reusable_plan=reusable,
        operational_controls=canonical.operational_controls,
        preflight_complete_text=False,
    )
    model = htr._EffectNet(
        extractor=extractor,
        hidden_dim=canonical.config.hidden_dim,
        head_depth=canonical.config.effect_head_depth,
        head_activation=canonical.config.effect_head_activation,
        head_dropout=canonical.config.effect_head_dropout,
        head_layer_norm=canonical.config.effect_head_layer_norm,
        head_bias=canonical.config.effect_head_bias,
    ).to(torch_device)
    observer = _ObserveAdamWPrefix(
        device=device,
        ready_barrier=task.ready_barrier,
        warmup_steps=task.warmup_steps,
        measured_steps=task.measured_steps,
        stop_on_next_step=False,
        capture_full_single_tensor=False,
        capture_exact_parameter_hashes=True,
        parameter_bundle_path=task.parameter_bundle_path,
        parameter_bundle_relative_path=(
            task.parameter_bundle_relative_path
        ),
    )
    try:
        try:
            with observer:
                htr._train_effect(
                    model,
                    texts=values,
                    positions=fit_positions,
                    y_residual=canonical.y_residual,
                    t_residual=canonical.t_residual,
                    pseudo_outcome=canonical.pseudo_outcome,
                    objective=canonical.objective,
                    config=canonical.config,
                    seed=canonical.model_seed,
                    device=torch_device,
                )
        except _PrefixFinished:
            pass
        return {
            "phase": task.phase,
            "canonical_index": task.canonical_index,
            "canonical_identity": copy.deepcopy(
                dict(task.canonical_identity)
            ),
            "batch_position_sha256s": _htr_effect_batch_hashes(
                canonical,
                steps=task.warmup_steps + task.measured_steps,
            ),
            "prefix_output": observer.result(),
            "prefix_conditioned_on_bounded_nuisance_oof": True,
            "source_nuisance_oof_e_sha256": (
                prepared.source_nuisance_oof_e_sha256
            ),
            "source_nuisance_oof_m_sha256": (
                prepared.source_nuisance_oof_m_sha256
            ),
            "complete_plan_authenticated": True,
            "complete_text_optimizer_execution_claimed": False,
        }
    finally:
        del model
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()


def _matched_batch_hashes(source: Any, *, pair_count: int, steps: int) -> list[str]:
    from oci.inference import role_neutral_matched_pair_group_execution as matched

    batch_size = int(source.config["htr_batch_size"])
    result: list[str] = []
    epoch = 1
    while len(result) < int(steps):
        order = np.arange(pair_count, dtype=np.int64)
        rng = np.random.default_rng(
            matched._derived_seed(
                source.htr_seed,
                purpose="htr_epoch_order",
                fold=epoch,
                view="htr",
            )
        )
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            result.append(_array_sha256(order[start : start + batch_size]))
            if len(result) == int(steps):
                break
        epoch += 1
        if epoch > int(source.config["htr_epochs"]):
            raise RuntimeError("matched HTR schedule is shorter than prefix")
    return result


def _run_matched_prefix(task: _PrefixTask, device: str) -> Mapping[str, Any]:
    import torch
    from oci.inference import role_neutral_matched_pair_group_execution as matched

    prepared = task.canonical_task
    if not isinstance(prepared, _PreparedMatchedTask):
        raise TypeError("matched prefix requires one prepared matched task")
    source = prepared.canonical_task
    frame = matched._make_frame(source.owner_fit_row_ids)
    fit_pos = np.asarray(source.fit_positions, dtype=np.int64)
    fit_frame = frame.iloc[fit_pos].reset_index(drop=True)
    pairs = matched.build_training_pairs(
        fit_frame,
        texts=[source.fit_texts[int(position)] for position in fit_pos],
        treatment=source.treatment[fit_pos],
        outcome=source.outcome[fit_pos],
        propensity=source.propensity_probability[fit_pos],
        outcome_prob=source.outcome_nuisance_probability[fit_pos],
        **matched._matching_training_config(source.config),
    )
    if pairs.empty or set(pairs["label"].astype(int)) != {0, 1}:
        raise RuntimeError("matched calibration task cannot fit both outcomes")
    initialization_texts = (
        pairs["control_text"].astype(str).tolist()
        + pairs["treated_text"].astype(str).tolist()
    )
    matched._assert_text_capacity(
        initialization_texts,
        extractor_config=source.config["htr_extractor"],
        stage="matched calibration prefix",
    )
    torch_device = torch.device(device)
    observer = _ObserveAdamWPrefix(
        device=device,
        ready_barrier=task.ready_barrier,
        warmup_steps=task.warmup_steps,
        measured_steps=task.measured_steps,
        stop_on_next_step=False,
        capture_full_single_tensor=False,
        capture_exact_parameter_hashes=True,
        parameter_bundle_path=task.parameter_bundle_path,
        parameter_bundle_relative_path=(
            task.parameter_bundle_relative_path
        ),
    )
    try:
        with observer:
            matched._train_htr(
                pairs=pairs,
                config=source.config,
                seed=source.htr_seed,
                extractor_factory=lambda value: matched._new_fold_htr_extractor(
                    config=source.config,
                    htr_model_path=source.htr_model_path,
                    device=value,
                ),
                device=torch_device,
            )
    except _PrefixFinished:
        pass
    fit_pair_count = int(len(pairs))
    fit_pair_text_sha256 = _sha256_json(initialization_texts)
    prefix = {
        **observer.result(),
        # These are optimizer-input identities, so they belong inside the
        # compared prefix envelope rather than only in surrounding telemetry.
        "fit_pair_count": fit_pair_count,
        "fit_pair_text_sha256": fit_pair_text_sha256,
    }
    return {
        "phase": task.phase,
        "canonical_index": task.canonical_index,
        "canonical_identity": copy.deepcopy(dict(task.canonical_identity)),
        "batch_position_sha256s": _matched_batch_hashes(
            source,
            pair_count=len(pairs),
            steps=task.warmup_steps + task.measured_steps,
        ),
        "fit_pair_count": fit_pair_count,
        "fit_pair_text_sha256": fit_pair_text_sha256,
        "complete_input_plan_content_sha256": (
            prepared.complete_input_plan_content_sha256
        ),
        "prefix_output": prefix,
        "complete_plan_authenticated": True,
        "complete_text_optimizer_execution_claimed": False,
    }


def _neural_bank_prefix(
    *,
    bank: str,
    bank_index: int,
    chunks: Sequence[np.ndarray],
    treatment: np.ndarray,
    outcome: np.ndarray,
    fit_e: np.ndarray,
    fit_m: np.ndarray,
    outcome_binary: bool,
    config: Any,
    seed: int,
    task: _PrefixTask,
    device: str,
    final_refit: bool,
    initial_queries: np.ndarray | None = None,
) -> Mapping[str, Any]:
    from oci.inference import neural_query_discovery_runtime as runtime
    from oci.inference.neural_cohort_witness import (
        cohort_contribution,
        fit_soft_contrast_queries,
        fit_soft_target_queries,
    )

    witness = runtime._witness_config(
        config,
        bank,
        final_refit=final_refit,
    )
    total_steps = task.warmup_steps + task.measured_steps
    if int(witness.epochs) < total_steps:
        raise ValueError(
            "neural production epoch schedule is shorter than prefix boundary"
        )
    bounded_witness = dataclasses.replace(
        witness,
        epochs=int(total_steps),
    )
    observer = _ObserveAdamWPrefix(
        device=device,
        ready_barrier=task.ready_barrier,
        warmup_steps=task.warmup_steps,
        measured_steps=task.measured_steps,
        stop_on_next_step=True,
        capture_full_single_tensor=True,
        capture_exact_parameter_hashes=False,
        parameter_bundle_path=None,
        parameter_bundle_relative_path=None,
        continue_after_terminal_projection=True,
    )
    fit_result: Mapping[str, Any] | None = None
    try:
        with observer:
            if bank == "treatment":
                fit_result = fit_soft_target_queries(
                    chunks,
                    treatment,
                    binary=True,
                    config=bounded_witness,
                    seed=seed,
                    device=device,
                    initial_queries=initial_queries,
                    target_name="treatment",
                )
            elif bank == "outcome":
                fit_result = fit_soft_target_queries(
                    chunks,
                    outcome,
                    binary=bool(outcome_binary),
                    config=bounded_witness,
                    seed=seed,
                    device=device,
                    initial_queries=initial_queries,
                    target_name="outcome",
                )
            else:
                train_u = treatment - fit_e
                train_v = outcome - fit_m
                contribution, constant = cohort_contribution(train_u, train_v)
                mutable_result = fit_soft_contrast_queries(
                    chunks,
                    contribution,
                    center_weights=np.square(train_u),
                    config=bounded_witness,
                    seed=seed,
                    device=device,
                    initial_queries=initial_queries,
                    objective_name=(
                        "constant_effect_orthogonalized_cohort_contrast"
                    ),
                )
                mutable_result["constant_effect"] = float(constant)
                fit_result = mutable_result
    except _PrefixFinished:
        raise RuntimeError(
            "bounded neural production fit aborted before natural scoring"
        )
    if not isinstance(fit_result, Mapping):
        raise RuntimeError("bounded neural production fit returned no result")
    queries = np.ascontiguousarray(
        np.asarray(fit_result.get("queries"), dtype=np.float32)
    )
    train_activations = np.ascontiguousarray(
        np.asarray(
            fit_result.get("train_activations"),
            dtype=np.float32,
        )
    )
    train_scores = np.ascontiguousarray(
        np.asarray(
            fit_result.get("train_standardized_scores"),
            dtype=np.float64,
        )
    )
    query_drift = np.ascontiguousarray(
        np.asarray(fit_result.get("query_drift"), dtype=np.float64)
    )
    if (
        queries.ndim != 2
        or train_activations.shape
        != (len(chunks), len(queries))
        or train_scores.shape != (len(queries),)
        or query_drift.shape != (len(queries),)
        or not np.isfinite(queries).all()
        or not np.isfinite(train_activations).all()
        or not np.isfinite(train_scores).all()
        or not np.isfinite(query_drift).all()
    ):
        raise RuntimeError("bounded neural production fit result is invalid")
    prefix = observer.result()
    return {
        **prefix,
        "production_fit_queries": queries.tolist(),
        "production_fit_query_dtype": queries.dtype.str,
        "production_fit_query_shape": [
            int(value) for value in queries.shape
        ],
        "production_train_activations": train_activations.tolist(),
        "production_train_activation_dtype": train_activations.dtype.str,
        "production_train_activation_shape": [
            int(value) for value in train_activations.shape
        ],
        "production_train_standardized_scores": train_scores.tolist(),
        "production_query_drift": query_drift.tolist(),
        "production_loss_history": [
            float(value) for value in fit_result["loss_history"]
        ],
        "production_objective": str(fit_result["objective"]),
        "production_constant_effect": (
            None
            if fit_result.get("constant_effect") is None
            else float(fit_result["constant_effect"])
        ),
        "bounded_execution_epoch_count": int(total_steps),
        "scientific_epoch_configuration_preserved_in_canonical_identity": True,
        "natural_production_post_loop_scoring_completed": True,
    }


def _production_prefix_candidate_scores(
    *,
    bank: str,
    train_chunks: Sequence[np.ndarray],
    validation_chunks: Sequence[np.ndarray],
    train_treatment: np.ndarray,
    train_outcome: np.ndarray,
    validation_treatment: np.ndarray,
    validation_outcome: np.ndarray,
    train_e: np.ndarray,
    train_m: np.ndarray,
    validation_e: np.ndarray,
    validation_m: np.ndarray,
    outcome_binary: bool,
    query_config: Any,
    initial_queries: np.ndarray,
    terminal_queries: np.ndarray,
    production_queries: np.ndarray,
    production_train_activations: np.ndarray,
    production_train_standardized_scores: np.ndarray,
    production_query_drift: np.ndarray,
    production_constant_effect: float | None,
    device: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Mapping[str, Any],
]:
    import torch
    from oci.inference import neural_query_discovery_runtime as runtime
    from oci.inference import neural_cohort_witness as witness

    observer_terminal = np.ascontiguousarray(
        terminal_queries,
        dtype=np.float32,
    )
    production_query_values = np.ascontiguousarray(
        production_queries,
        dtype=np.float32,
    )
    candidate_queries = production_query_values.copy()
    initial = np.ascontiguousarray(initial_queries, dtype=np.float32)
    if (
        production_query_values.ndim != 2
        or production_query_values.shape != initial.shape
        or observer_terminal.shape != production_query_values.shape
        or not np.isfinite(production_query_values).all()
        or not np.isfinite(observer_terminal).all()
        or not np.isfinite(initial).all()
        or not np.allclose(
            observer_terminal,
            production_query_values,
            rtol=3e-5,
            atol=3e-6,
        )
    ):
        raise RuntimeError("neural prefix query scoring received invalid tensors")
    bank_config = runtime._witness_config(
        query_config,
        bank,
        final_refit=False,
    )
    recomputed_train_activations = np.ascontiguousarray(
        witness.soft_retrieval_activations(
            train_chunks,
            production_query_values.copy(),
            temperature=float(query_config.temperature),
            device=device,
            patient_batch_size=int(
                query_config.retrieval_patient_batch_size
            ),
        ),
        dtype=np.float32,
    )
    train_activations = np.ascontiguousarray(
        production_train_activations,
        dtype=np.float32,
    )
    if (
        train_activations.shape != recomputed_train_activations.shape
        or not np.isfinite(train_activations).all()
        or not np.allclose(
            train_activations,
            recomputed_train_activations,
            rtol=3e-5,
            atol=3e-6,
        )
    ):
        raise RuntimeError(
            "natural production train activations failed independent replay"
        )
    validation_activations = np.ascontiguousarray(
        witness.soft_retrieval_activations(
            validation_chunks,
            candidate_queries,
            temperature=float(query_config.temperature),
            device=device,
            patient_batch_size=int(
                query_config.retrieval_patient_batch_size
            ),
        ),
        dtype=np.float32,
    )
    train_t = np.asarray(train_treatment, dtype=np.float64)
    train_y = np.asarray(train_outcome, dtype=np.float64)
    validation_t = np.asarray(validation_treatment, dtype=np.float64)
    validation_y = np.asarray(validation_outcome, dtype=np.float64)
    if bank == "treatment":
        contribution = witness.direct_target_contribution(
            train_t,
            binary=True,
        )
        weights = np.ones_like(contribution)
        validation_audit = witness.standardized_direct_target_contrasts(
            validation_activations,
            validation_t,
            binary=True,
        )
        constant_effect: float | None = None
    elif bank == "outcome":
        contribution = witness.direct_target_contribution(
            train_y,
            binary=bool(outcome_binary),
        )
        weights = np.ones_like(contribution)
        validation_audit = witness.standardized_direct_target_contrasts(
            validation_activations,
            validation_y,
            binary=bool(outcome_binary),
        )
        constant_effect = None
    elif bank == "effect":
        train_u = train_t - np.asarray(train_e, dtype=np.float64)
        train_v = train_y - np.asarray(train_m, dtype=np.float64)
        validation_u = validation_t - np.asarray(
            validation_e,
            dtype=np.float64,
        )
        validation_v = validation_y - np.asarray(
            validation_m,
            dtype=np.float64,
        )
        contribution, constant_effect = witness.cohort_contribution(
            train_u,
            train_v,
        )
        if (
            production_constant_effect is None
            or not math.isclose(
                float(production_constant_effect),
                float(constant_effect),
                rel_tol=3e-5,
                abs_tol=3e-6,
            )
        ):
            raise RuntimeError(
                "natural production constant effect failed replay"
            )
        weights = np.square(train_u)
        validation_audit = witness.standardized_cohort_moments(
            validation_activations,
            validation_u,
            validation_v,
            constant_effect=float(constant_effect),
        )
    else:
        raise ValueError("neural prefix scoring received an unknown bank")

    torch_device = torch.device(device)
    activation_tensor = torch.as_tensor(
        train_activations,
        dtype=torch.float32,
        device=torch_device,
    )
    contribution_tensor = torch.as_tensor(
        np.asarray(contribution, dtype=np.float32),
        dtype=torch.float32,
        device=torch_device,
    )
    weight_tensor = torch.as_tensor(
        np.asarray(weights, dtype=np.float32),
        dtype=torch.float32,
        device=torch_device,
    )
    with torch.no_grad():
        weight_denominator = torch.clamp(
            weight_tensor.sum(),
            min=float(bank_config.epsilon),
        )
        recomputed_train_scores = (
            witness._torch_standardized_moments(
                activation_tensor,
                contribution_tensor,
                weight_tensor,
                weight_denominator,
                epsilon=float(bank_config.epsilon),
            )
            .detach()
            .to(device="cpu")
            .numpy()
            .astype(np.float64, copy=False)
        )
        recomputed_query_drift = (
            torch.linalg.vector_norm(
                torch.as_tensor(
                    production_query_values,
                    dtype=torch.float32,
                    device=torch_device,
                )
                - torch.as_tensor(
                    initial,
                    dtype=torch.float32,
                    device=torch_device,
                ),
                dim=1,
            )
            .detach()
            .to(device="cpu")
            .numpy()
            .astype(np.float64, copy=False)
        )
    train_scores = np.ascontiguousarray(
        np.asarray(
            production_train_standardized_scores,
            dtype=np.float64,
        )
    )
    query_drift = np.ascontiguousarray(
        np.asarray(production_query_drift, dtype=np.float64)
    )
    validation_scores = np.asarray(
        validation_audit["standardized_scores"],
        dtype=np.float64,
    )
    if (
        train_scores.shape != (len(candidate_queries),)
        or recomputed_train_scores.shape != (len(candidate_queries),)
        or validation_scores.shape != (len(candidate_queries),)
        or query_drift.shape != (len(candidate_queries),)
        or recomputed_query_drift.shape != (len(candidate_queries),)
        or not np.isfinite(train_scores).all()
        or not np.isfinite(recomputed_train_scores).all()
        or not np.isfinite(validation_scores).all()
        or not np.isfinite(query_drift).all()
        or not np.isfinite(recomputed_query_drift).all()
        or np.any(query_drift < 0.0)
        or not np.allclose(
            train_scores,
            recomputed_train_scores,
            rtol=3e-5,
            atol=3e-6,
        )
        or not np.allclose(
            query_drift,
            recomputed_query_drift,
            rtol=3e-5,
            atol=3e-6,
        )
    ):
        raise RuntimeError("neural prefix production scoring is incomplete")
    proof_body = {
        "schema_version": (
            "production_r14_neural_prefix_candidate_scoring_v1"
        ),
        "bank": bank,
        "train_activation_shape": [
            int(value) for value in train_activations.shape
        ],
        "train_activation_dtype": train_activations.dtype.str,
        "train_activation_sha256": _array_sha256(train_activations),
        "independently_recomputed_train_activation_sha256": _array_sha256(
            recomputed_train_activations
        ),
        "validation_activation_shape": [
            int(value) for value in validation_activations.shape
        ],
        "validation_activation_dtype": validation_activations.dtype.str,
        "validation_activation_sha256": _array_sha256(
            validation_activations
        ),
        "train_contribution_sha256": _array_sha256(
            np.asarray(contribution, dtype=np.float32)
        ),
        "train_center_weights_sha256": _array_sha256(
            np.asarray(weights, dtype=np.float32)
        ),
        "train_standardized_score_sha256": _array_sha256(train_scores),
        "validation_audit_standardized_score_sha256": _array_sha256(
            validation_scores
        ),
        "query_drift_sha256": _array_sha256(query_drift),
        "initial_query_sha256": _array_sha256(initial),
        "observer_terminal_query_sha256": _array_sha256(
            observer_terminal
        ),
        "natural_production_query_sha256": _array_sha256(
            production_query_values
        ),
        "candidate_query_after_validation_activation_sha256": (
            _array_sha256(candidate_queries)
        ),
        "constant_effect": constant_effect,
        "training_score_policy": (
            "torch_population_std_weighted_center_production_v1"
        ),
        "validation_score_policy": (
            "numpy_sample_std_production_audit_v1"
        ),
        "epsilon": float(bank_config.epsilon),
        "temperature": float(query_config.temperature),
        "production_scoring_recomputed_after_prefix_outside_measured_window": (
            True
        ),
        "natural_production_post_loop_outputs_are_authoritative": True,
        "independent_train_activation_score_and_drift_replay_accepted": True,
    }
    return (
        candidate_queries,
        train_scores,
        validation_scores,
        query_drift,
        {
            **proof_body,
            "content_sha256": _sha256_json(proof_body),
        },
    )


def _run_neural_inner_prefix(
    task: _PrefixTask,
    device: str,
) -> Mapping[str, Any]:
    from oci.inference import neural_query_discovery_runtime as runtime

    prepared = task.canonical_task
    if not isinstance(prepared, _PreparedNeuralInnerTask):
        raise TypeError("neural inner prefix requires one prepared inner task")
    canonical = prepared.canonical_task
    arguments = dict(canonical.arguments)
    resolved = runtime._resolve_task_chunks(
        row_ids=arguments["row_ids"],
        texts=arguments["texts"],
        chunks=arguments["chunks"],
        embedding_task_reference=arguments["embedding_task_reference"],
    )
    train_indices = np.asarray(arguments["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(
        arguments["validation_indices"],
        dtype=np.int64,
    )
    chunks = [resolved[int(index)] for index in train_indices]
    validation_chunks = [
        resolved[int(index)] for index in validation_indices
    ]
    treatment = np.asarray(
        arguments["treatment"][train_indices],
        dtype=np.float64,
    )
    outcome = np.asarray(
        arguments["outcome"][train_indices],
        dtype=np.float64,
    )
    validation_treatment = np.asarray(
        arguments["treatment"][validation_indices],
        dtype=np.float64,
    )
    validation_outcome = np.asarray(
        arguments["outcome"][validation_indices],
        dtype=np.float64,
    )
    rows = tuple(int(arguments["row_ids"][index]) for index in train_indices)
    bank_results: dict[str, Mapping[str, Any]] = {}
    candidates_by_bank: dict[str, dict[str, Any]] = {}
    scoring_proofs: dict[str, Mapping[str, Any]] = {}
    for bank_index, bank in enumerate(runtime.BANKS):
        bank_seed = int(arguments["seed"] + 100 * bank_index)
        prefix = _neural_bank_prefix(
            bank=bank,
            bank_index=bank_index,
            chunks=chunks,
            treatment=treatment,
            outcome=outcome,
            fit_e=np.asarray(prepared.train_e, dtype=np.float64),
            fit_m=np.asarray(prepared.train_m, dtype=np.float64),
            outcome_binary=bool(arguments["outcome_binary"]),
            config=arguments["config"],
            seed=bank_seed,
            task=task,
            device=device,
            final_refit=False,
        )
        observer_terminal_query_tensor = np.asarray(
            prefix["terminal_query_tensor"],
            dtype=np.float32,
        )
        initial_query_tensor = np.asarray(
            prefix["initial_query_tensor"],
            dtype=np.float32,
        )
        (
            query_tensor,
            train_scores,
            validation_scores,
            query_drift,
            scoring_proof,
        ) = _production_prefix_candidate_scores(
            bank=bank,
            train_chunks=chunks,
            validation_chunks=validation_chunks,
            train_treatment=treatment,
            train_outcome=outcome,
            validation_treatment=validation_treatment,
            validation_outcome=validation_outcome,
            train_e=np.asarray(prepared.train_e, dtype=np.float64),
            train_m=np.asarray(prepared.train_m, dtype=np.float64),
            validation_e=np.asarray(
                prepared.validation_e,
                dtype=np.float64,
            ),
            validation_m=np.asarray(
                prepared.validation_m,
                dtype=np.float64,
            ),
            outcome_binary=bool(arguments["outcome_binary"]),
            query_config=arguments["config"],
            initial_queries=initial_query_tensor,
            terminal_queries=observer_terminal_query_tensor,
            production_queries=np.asarray(
                prefix["production_fit_queries"],
                dtype=np.float32,
            ),
            production_train_activations=np.asarray(
                prefix["production_train_activations"],
                dtype=np.float32,
            ),
            production_train_standardized_scores=np.asarray(
                prefix["production_train_standardized_scores"],
                dtype=np.float64,
            ),
            production_query_drift=np.asarray(
                prefix["production_query_drift"],
                dtype=np.float64,
            ),
            production_constant_effect=prefix[
                "production_constant_effect"
            ],
            device=device,
        )
        prefix = {
            **dict(prefix),
            "production_candidate_score_vectors": {
                "train_standardized_scores": train_scores.tolist(),
                "validation_audit_standardized_scores": (
                    validation_scores.tolist()
                ),
                "query_drift": query_drift.tolist(),
            },
        }
        candidates = []
        for index in range(len(query_tensor)):
            query = np.ascontiguousarray(
                query_tensor[index],
                dtype=np.float32,
            )
            if query.ndim != 1 or not np.isfinite(query).all():
                raise RuntimeError(
                    "neural inner prefix produced an invalid candidate query"
                )
            candidates.append(
                {
                    "candidate_id": (
                        f"{bank}_fold_{int(arguments['fold']):02d}_"
                        f"query_{index + 1:03d}"
                    ),
                    "bank": bank,
                    "subfold": int(arguments["fold"]),
                    "query": query.tolist(),
                    "query_dtype": query.dtype.str,
                    "query_shape": [
                        int(value) for value in query.shape
                    ],
                    "query_sha256": _array_sha256(query),
                    "train_standardized_score": float(
                        train_scores[index]
                    ),
                    "validation_audit_standardized_score": float(
                        validation_scores[index]
                    ),
                    "validation_audit_only_not_used_for_gating": True,
                    "query_drift": float(query_drift[index]),
                    (
                        "calibration_prefix_derived_with_"
                        "production_scoring"
                    ): True,
                }
            )
        bank_results[bank] = prefix
        candidates_by_bank[bank] = {"candidates": candidates}
        scoring_proofs[bank] = scoring_proof
    starts = [
        int(value["measured_started_monotonic_ns"])
        for value in bank_results.values()
    ]
    finishes = [
        int(value["measured_finished_monotonic_ns"])
        for value in bank_results.values()
    ]
    return {
        "phase": task.phase,
        "canonical_index": task.canonical_index,
        "canonical_identity": copy.deepcopy(dict(task.canonical_identity)),
        "prefix_output": {
            "banks": copy.deepcopy(bank_results),
            "measured_started_monotonic_ns": min(starts),
            "measured_finished_monotonic_ns": max(finishes),
            "measured_optimizer_steps": (
                task.measured_steps * len(bank_results)
            ),
            "optimizer_row_order_sha256": _sha256_json(list(rows)),
        },
        "fold": int(arguments["fold"]),
        "identity_payload": {
            "fold": int(arguments["fold"]),
            "seed": int(arguments["seed"]),
            "train_row_ids": list(rows),
            "validation_row_ids": [
                int(arguments["row_ids"][index])
                for index in arguments["validation_indices"]
            ],
        },
        "banks": candidates_by_bank,
        "candidate_scoring_proofs": copy.deepcopy(scoring_proofs),
        "complete_input_plan_content_sha256": (
            prepared.complete_input_plan_content_sha256
        ),
        "complete_plan_authenticated": True,
        "complete_text_optimizer_execution_claimed": False,
    }


def _run_neural_final_prefix(
    task: _PrefixTask,
    device: str,
) -> Mapping[str, Any]:
    from oci.inference import neural_query_discovery_runtime as runtime
    from oci.inference.neural_cohort_witness import (
        build_ungated_consensus_query_bank,
        soft_retrieval_activations,
    )

    prepared = task.canonical_task
    if not isinstance(prepared, _PreparedNeuralFinalTask):
        raise TypeError("neural final prefix requires one prepared final task")
    canonical = prepared.canonical_task
    arguments = dict(canonical.arguments)
    bank = str(arguments["bank"])
    bank_index = int(arguments["bank_index"])
    chunks = runtime._resolve_task_chunks(
        row_ids=arguments["row_ids"],
        texts=arguments["texts"],
        chunks=arguments["chunks"],
        embedding_task_reference=arguments["embedding_task_reference"],
    )
    candidate_queries = _validated_calibration_candidate_queries(
        arguments["candidates"]
    )
    candidate_activations = soft_retrieval_activations(
        chunks,
        candidate_queries,
        temperature=float(arguments["config"].temperature),
        device=device,
        patient_batch_size=int(
            arguments["config"].retrieval_patient_batch_size
        ),
    )
    consensus_config = runtime._witness_config(
        arguments["config"],
        bank,
        final_refit=False,
    )
    consensus_seed = int(arguments["seed"] + 1000 + bank_index)
    consensus = build_ungated_consensus_query_bank(
        arguments["candidates"],
        candidate_activations=candidate_activations,
        n_queries=arguments["config"].query_count(bank),
        bank=bank,
        seed=consensus_seed,
        config=consensus_config,
    )
    initial_queries = np.asarray(
        consensus["queries"],
        dtype=np.float32,
    )
    prefix = _neural_bank_prefix(
        bank=bank,
        bank_index=bank_index,
        chunks=chunks,
        treatment=np.asarray(arguments["treatment"], dtype=np.float64),
        outcome=np.asarray(arguments["outcome"], dtype=np.float64),
        fit_e=np.asarray(arguments["fit_e"], dtype=np.float64),
        fit_m=np.asarray(arguments["fit_m"], dtype=np.float64),
        outcome_binary=bool(arguments["outcome_binary"]),
        config=arguments["config"],
        seed=int(arguments["seed"] + 2000 + bank_index),
        task=task,
        device=device,
        final_refit=True,
        initial_queries=initial_queries,
    )
    prefix = {
        **dict(prefix),
        "prefix_input_candidate_queries": candidate_queries.tolist(),
        "prefix_input_candidate_query_dtype": candidate_queries.dtype.str,
        "prefix_input_candidate_query_shape": [
            int(value) for value in candidate_queries.shape
        ],
    }
    return {
        "phase": task.phase,
        "canonical_index": task.canonical_index,
        "canonical_identity": copy.deepcopy(dict(task.canonical_identity)),
        "prefix_output": prefix,
        "bank": bank,
        "bank_index": bank_index,
        "consensus_seed": consensus_seed,
        "final_refit_seed": int(arguments["seed"] + 2000 + bank_index),
        "final_inputs_derived_from_serial_inner_prefix_candidates": True,
        "serial_prefix_conditioned_calibration_task": True,
        "complete_input_plan_content_sha256": (
            prepared.complete_input_plan_content_sha256
        ),
        "complete_plan_authenticated": True,
        "complete_text_optimizer_execution_claimed": False,
    }


_WORKER_BY_PHASE: Mapping[str, Callable[[Any, str], Mapping[str, Any]]] = {
    "htr_nuisance": _run_htr_prefix,
    "htr_effect": _run_htr_effect_prefix,
    "matched_pair_htr": _run_matched_prefix,
    "neural_inner_folds": _run_neural_inner_prefix,
    "neural_final_banks": _run_neural_final_prefix,
}


def _htr_task_identity(task: Any) -> dict[str, Any]:
    return {
        "objective": "joint_treatment_outcome_nuisance",
        "fold": int(task.fold),
        "split_seed": int(task.split_seed),
        "model_seed": int(task.model_seed),
        "fit_positions_sha256": _array_sha256(task.fit_positions),
        "validation_positions_sha256": _array_sha256(
            task.validation_positions
        ),
        "config": task.config.as_dict(),
        "complete_plan_content_sha256": (
            task.text_authority.materialized_plan.content_sha256
        ),
        "complete_input_plan_content_sha256": (
            task.text_authority.materialized_plan.content_sha256
        ),
    }


def _htr_effect_task_identity(task: Any) -> dict[str, Any]:
    if not isinstance(task, _PreparedHTREffectTask):
        raise TypeError("HTR effect identity requires one prepared effect task")
    prepared = task
    task = prepared.canonical_task
    return {
        "objective": str(task.objective),
        "fold": int(task.fold),
        "split_seed": int(task.split_seed),
        "model_seed": int(task.model_seed),
        "fit_positions_sha256": _array_sha256(task.fit_positions),
        "eligible_fit_positions_sha256": _array_sha256(
            task.eligible_fit_positions
        ),
        "validation_positions_sha256": _array_sha256(
            task.validation_positions
        ),
        "y_residual_sha256": _array_sha256(task.y_residual),
        "t_residual_sha256": _array_sha256(task.t_residual),
        "pseudo_outcome_sha256": _array_sha256(task.pseudo_outcome),
        "config": task.config.as_dict(),
        "complete_plan_content_sha256": (
            task.text_authority.materialized_plan.content_sha256
        ),
        "complete_input_plan_content_sha256": (
            task.text_authority.materialized_plan.content_sha256
        ),
        "source_nuisance_oof_e_sha256": (
            prepared.source_nuisance_oof_e_sha256
        ),
        "source_nuisance_oof_m_sha256": (
            prepared.source_nuisance_oof_m_sha256
        ),
        "prefix_conditioned_on_bounded_nuisance_oof": True,
    }


def _matched_task_identity(task: Any) -> dict[str, Any]:
    if not isinstance(task, _PreparedMatchedTask):
        raise TypeError("matched identity requires one prepared matched task")
    prepared = task
    task = prepared.canonical_task
    if (
        re.fullmatch(
            r"[0-9a-f]{64}",
            prepared.complete_input_plan_content_sha256,
        )
        is None
    ):
        raise ValueError("matched input-plan identity is invalid")
    return {
        "objective": str(task.objective),
        "fold": int(task.fold),
        "split_seed": int(task.split_seed),
        "htr_seed": int(task.htr_seed),
        "owner_scope_seed": int(task.owner_scope_seed),
        "owner_fit_row_ids_sha256": _sha256_json(
            list(task.owner_fit_row_ids)
        ),
        "fit_text_sha256": _sha256_json(list(task.fit_texts)),
        "treatment_sha256": _array_sha256(task.treatment),
        "outcome_sha256": _array_sha256(task.outcome),
        "propensity_sha256": _array_sha256(task.propensity_probability),
        "outcome_nuisance_sha256": _array_sha256(
            task.outcome_nuisance_probability
        ),
        "fit_positions_sha256": _array_sha256(task.fit_positions),
        "validation_positions_sha256": _array_sha256(
            task.validation_positions
        ),
        "config": copy.deepcopy(dict(task.config)),
        "complete_input_plan_content_sha256": (
            prepared.complete_input_plan_content_sha256
        ),
    }


def _neural_inner_task_identity(task: _PreparedNeuralInnerTask) -> dict[str, Any]:
    if (
        not isinstance(task, _PreparedNeuralInnerTask)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            task.complete_input_plan_content_sha256,
        )
        is None
    ):
        raise ValueError("neural-inner input-plan identity is invalid")
    arguments = dict(task.canonical_task.arguments)
    return {
        "fold": int(arguments["fold"]),
        "seed": int(arguments["seed"]),
        "train_positions_sha256": _array_sha256(arguments["train_indices"]),
        "validation_positions_sha256": _array_sha256(
            arguments["validation_indices"]
        ),
        "row_ids_sha256": _sha256_json(list(arguments["row_ids"])),
        "texts_sha256": _sha256_json(list(arguments["texts"])),
        "treatment_sha256": _array_sha256(arguments["treatment"]),
        "outcome_sha256": _array_sha256(arguments["outcome"]),
        "outcome_binary": bool(arguments["outcome_binary"]),
        "parent_input_binding_sha256": str(
            arguments["parent_input_binding_sha256"]
        ),
        "nuisance_views_sha256": _sha256_json(
            _closed_json_state(arguments["nuisance_views"])
        ),
        "nuisance_folds": int(arguments["nuisance_folds"]),
        "nuisance_stack_config_sha256": _sha256_json(
            _closed_json_state(arguments["nuisance_stack_config"])
        ),
        "query_config": arguments["config"].to_dict(),
        "nuisance_identity": copy.deepcopy(dict(task.nuisance_identity)),
        "complete_input_plan_content_sha256": (
            task.complete_input_plan_content_sha256
        ),
    }


def _validated_calibration_candidate_queries(
    candidates: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    required = {
        "candidate_id",
        "bank",
        "subfold",
        "query",
        "query_dtype",
        "query_shape",
        "query_sha256",
        "train_standardized_score",
        "validation_audit_standardized_score",
        "validation_audit_only_not_used_for_gating",
        "query_drift",
        "calibration_prefix_derived_with_production_scoring",
    }
    arrays: list[np.ndarray] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or set(candidate) != required:
            raise RuntimeError(
                "calibration candidate query record changed schema"
            )
        query = np.ascontiguousarray(
            np.asarray(candidate["query"], dtype=np.float32)
        )
        if (
            query.ndim != 1
            or query.size < 1
            or not np.isfinite(query).all()
            or candidate["query_dtype"] != query.dtype.str
            or candidate["query_shape"]
            != [int(value) for value in query.shape]
            or candidate["query_sha256"] != _array_sha256(query)
            or candidate[
                "calibration_prefix_derived_with_production_scoring"
            ]
            is not True
            or candidate[
                "validation_audit_only_not_used_for_gating"
            ]
            is not True
            or isinstance(candidate["subfold"], bool)
            or not isinstance(candidate["subfold"], int)
            or not isinstance(candidate["candidate_id"], str)
            or not candidate["candidate_id"]
            or not isinstance(candidate["bank"], str)
            or not candidate["bank"]
            or not isinstance(
                candidate["train_standardized_score"],
                (int, float),
            )
            or isinstance(
                candidate["train_standardized_score"],
                bool,
            )
            or not math.isfinite(
                float(candidate["train_standardized_score"])
            )
            or isinstance(
                candidate["validation_audit_standardized_score"],
                bool,
            )
            or not isinstance(
                candidate["validation_audit_standardized_score"],
                (int, float),
            )
            or not math.isfinite(
                float(candidate["validation_audit_standardized_score"])
            )
            or isinstance(candidate["query_drift"], bool)
            or not isinstance(candidate["query_drift"], (int, float))
            or not math.isfinite(float(candidate["query_drift"]))
            or float(candidate["query_drift"]) < 0.0
        ):
            raise RuntimeError(
                f"calibration candidate query {index} failed authentication"
            )
        arrays.append(query)
    if not arrays or len({value.shape for value in arrays}) != 1:
        raise RuntimeError(
            "calibration candidate queries are empty or ragged"
        )
    return np.ascontiguousarray(np.vstack(arrays), dtype=np.float32)


def _neural_final_task_identity(task: Any) -> dict[str, Any]:
    if (
        not isinstance(task, _PreparedNeuralFinalTask)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            task.complete_input_plan_content_sha256,
        )
        is None
    ):
        raise ValueError("neural-final input-plan identity is invalid")
    prepared = task
    arguments = dict(prepared.canonical_task.arguments)
    candidates = arguments["candidates"]
    candidate_queries = _validated_calibration_candidate_queries(candidates)
    return {
        "bank": str(arguments["bank"]),
        "bank_index": int(arguments["bank_index"]),
        "consensus_seed": int(
            arguments["seed"] + 1000 + int(arguments["bank_index"])
        ),
        "final_refit_seed": int(
            arguments["seed"] + 2000 + int(arguments["bank_index"])
        ),
        "row_ids_sha256": _sha256_json(list(arguments["row_ids"])),
        "texts_sha256": _sha256_json(list(arguments["texts"])),
        "treatment_sha256": _array_sha256(arguments["treatment"]),
        "outcome_sha256": _array_sha256(arguments["outcome"]),
        "outcome_binary": bool(arguments["outcome_binary"]),
        "fit_e_sha256": _array_sha256(arguments["fit_e"]),
        "fit_m_sha256": _array_sha256(arguments["fit_m"]),
        "candidate_ids": [
            str(candidate["candidate_id"]) for candidate in candidates
        ],
        "candidate_query_shape": [
            int(value) for value in candidate_queries.shape
        ],
        "candidate_query_dtype": candidate_queries.dtype.str,
        "candidate_query_sha256": _array_sha256(candidate_queries),
        "query_config": arguments["config"].to_dict(),
        "complete_input_plan_content_sha256": (
            prepared.complete_input_plan_content_sha256
        ),
        "inputs_derived_from_serial_inner_prefix_candidates": True,
        "serial_prefix_conditioned_calibration_task": True,
    }


def _capture_htr_tasks(
    *,
    output_root: Path,
    owner: Any,
    inputs: Any,
    config: Any,
    controls: Any,
    model_path: Path,
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    import torch
    from oci.inference import role_neutral_htr_group_execution as htr
    from oci.inference.stage1_htr_operational_controls import (
        RoleNeutralHTRFoldResourcePlan,
    )

    coverage = htr._coverage_plan(
        texts=inputs.fit_texts,
        config=config,
        phase="fit",
    )
    marker = htr._resolve_model_marker(
        config=config,
        htr_model_path=model_path,
    )
    capture_resource = RoleNeutralHTRFoldResourcePlan(
        devices=("cpu",),
        fold_parallelism=1,
        fold_slots_per_device=1,
        owner_cpu_budget=1,
        fold_parallel_backend="threads",
    )
    original = htr._execute_htr_fold_tasks

    def capture(tasks: Sequence[Any], **_kwargs: Any) -> tuple[Any, ...]:
        raise _CapturedCanonicalTasks(tasks, phase="htr_nuisance")

    htr._execute_htr_fold_tasks = capture
    try:
        try:
            htr._fit_owner_htr_folds(
                owner=owner,
                texts=inputs.fit_texts,
                treatment=np.asarray(inputs.fit_treatment, dtype=np.float64),
                outcome=np.asarray(inputs.fit_outcome, dtype=np.float64),
                coverage=coverage,
                config=config,
                model_marker=marker,
                store=htr._SafeArrayStore(),
                operational_controls=controls,
                resource_plan=capture_resource,
                scratch_parent=output_root,
                external_event_sink=None,
            )
        except _CapturedCanonicalTasks as captured:
            tasks = captured.tasks
        else:  # pragma: no cover - fail-closed production seam
            raise RuntimeError("HTR production builder did not expose canonical tasks")
    finally:
        htr._execute_htr_fold_tasks = original
    if len(tasks) != _EXPECTED_TASK_COUNTS["htr_nuisance"]:
        raise RuntimeError("HTR canonical task count changed")
    authority = tasks[0].text_authority
    if (
        authority.texts is None
        or authority.coverage is None
        or authority.reusable_plan is None
    ):
        raise RuntimeError("captured HTR task omitted its complete reusable plan")
    descriptor = htr._materialize_reusable_text_plan(
        root=output_root / "htr_complete_authenticated_plan",
        plan=authority.reusable_plan,
        coverage=authority.coverage,
        texts=authority.texts,
        row_ids=authority.row_ids,
    )
    materialized = htr._FoldTextAuthority.materialized(descriptor)
    rebound = tuple(
        dataclasses.replace(task, text_authority=materialized)
        for task in tasks
    )
    del torch
    return rebound, descriptor.attestation()


def _factory_invocation(
    *,
    prepared: Any,
    owner: Any,
    members: tuple[Any, ...],
    output_root: Path,
    component: str,
    devices: tuple[str, ...],
    htr_controls: Any,
    neural_controls: Any,
    cpu_budget: int,
) -> Any:
    from oci.inference.neural_query_execution_topology import (
        NeuralQueryExecutionTopology,
    )
    from oci.inference.production_stage1_role_neutral_execution import (
        RoleNeutralComponentInvocation,
    )

    return RoleNeutralComponentInvocation(
        plan=prepared.stage1_scope_plan,
        physical_owner=owner,
        logical_members=members,
        component=component,
        output_root=output_root,
        resource=devices[0],
        neural_query_execution_topology=NeuralQueryExecutionTopology(
            devices=devices
        ),
        htr_operational_controls=htr_controls,
        neural_query_operational_controls=neural_controls,
        htr_fold_devices=devices,
        owner_cpu_budget=int(cpu_budget),
    )


def _prepare_bow_and_capture_matched(
    *,
    setup_root: Path,
    prepared: Any,
    factories: Any,
    owner: Any,
    members: tuple[Any, ...],
    devices: tuple[str, ...],
    htr_controls: Any,
    neural_controls: Any,
    cpu_budget: int,
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    from oci.inference import role_neutral_matched_pair_group_execution as matched

    mapping = factories.as_mapping()
    bow_root = setup_root / "bow"
    bow = mapping["bow"](
        _factory_invocation(
            prepared=prepared,
            owner=owner,
            members=members,
            output_root=bow_root,
            component="bow",
            devices=devices,
            htr_controls=htr_controls,
            neural_controls=neural_controls,
            cpu_budget=cpu_budget,
        )
    )
    bow.execute()
    bow_receipt = bow.authenticate()
    original = matched._execute_htr_fold_tasks

    def capture(tasks: Sequence[Any], **_kwargs: Any) -> tuple[Any, ...]:
        raise _CapturedCanonicalTasks(tasks, phase="matched_pair_htr")

    matched._execute_htr_fold_tasks = capture
    try:
        bound = mapping["matched_pair"](
            _factory_invocation(
                prepared=prepared,
                owner=owner,
                members=members,
                output_root=setup_root / "matched_capture",
                component="matched_pair",
                devices=devices,
                htr_controls=htr_controls,
                neural_controls=neural_controls,
                cpu_budget=cpu_budget,
            )
        )
        try:
            bound.execute()
        except _CapturedCanonicalTasks as captured:
            tasks = captured.tasks
        else:  # pragma: no cover
            raise RuntimeError(
                "matched production builder did not expose canonical tasks"
            )
    finally:
        matched._execute_htr_fold_tasks = original
    if len(tasks) != _EXPECTED_TASK_COUNTS["matched_pair_htr"]:
        raise RuntimeError("matched canonical task count changed")
    return tasks, {
        "bow_preparation_outside_measured_gpu_window": True,
        "bow_authenticated_receipt": dataclasses.asdict(bow_receipt),
    }


def _published_json_descriptor(
    path: Path,
    value: Mapping[str, Any],
    *,
    producer: str,
) -> dict[str, Any]:
    digest, size = _sha256_file(path)
    content_sha256 = str(value.get("content_sha256") or "")
    if (
        re.fullmatch(r"[0-9a-f]{64}", content_sha256) is None
        or value.get("content_sha256")
        != _sha256_json(
            {
                key: item
                for key, item in value.items()
                if key != "content_sha256"
            }
        )
        or stat.S_IMODE(os.lstat(path).st_mode) != 0o444
    ):
        raise RuntimeError("published calibration input plan is not immutable")
    return {
        "producer": str(producer),
        "path": str(path.resolve(strict=True)),
        "sha256": digest,
        "size_bytes": size,
        "content_sha256": content_sha256,
        "self_hash_reopened_and_verified": True,
    }


def _reopen_published_json_descriptor(
    descriptor: Mapping[str, Any],
    *,
    expected_root: Path,
    expected_producer: str,
) -> Mapping[str, Any]:
    required = {
        "producer",
        "path",
        "sha256",
        "size_bytes",
        "content_sha256",
        "self_hash_reopened_and_verified",
    }
    if not isinstance(descriptor, Mapping) or set(descriptor) != required:
        raise ValueError("published plan descriptor changed schema")
    root = expected_root.resolve(strict=True)
    supplied = Path(str(descriptor["path"]))
    if (
        not supplied.is_absolute()
        or supplied.resolve(strict=True) != supplied
        or descriptor["producer"] != expected_producer
        or descriptor["self_hash_reopened_and_verified"] is not True
    ):
        raise ValueError("published plan descriptor changed authority")
    try:
        supplied.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "published plan escaped the calibration output root"
        ) from exc
    state = os.lstat(supplied)
    digest, size = _sha256_file(supplied)
    if (
        stat.S_IMODE(state.st_mode) != 0o444
        or digest != descriptor["sha256"]
        or size != descriptor["size_bytes"]
    ):
        raise ValueError("published plan file failed raw authentication")
    reopened = json.loads(supplied.read_text(encoding="utf-8"))
    if not isinstance(reopened, Mapping):
        raise ValueError("published plan JSON is not one object")
    body = {
        key: value
        for key, value in reopened.items()
        if key != "content_sha256"
    }
    if (
        reopened.get("content_sha256") != _sha256_json(body)
        or reopened.get("content_sha256")
        != descriptor["content_sha256"]
    ):
        raise ValueError("published plan JSON failed self-authentication")
    return dict(reopened)


def _materialize_matched_complete_input_plan(
    *,
    root: Path,
    tasks: Sequence[Any],
    stage1_plan_scientific_content_sha256: str,
    htr_complete_plan_content_sha256: str,
    htr_model_tree_sha256: str,
) -> tuple[tuple[_PreparedMatchedTask, ...], Mapping[str, Any]]:
    from oci.inference import role_neutral_matched_pair_group_execution as matched
    from oci.models.hierarchical_transformer_extractor import (
        split_text_into_word_chunks,
    )

    canonical = tuple(tasks)
    if len(canonical) != _EXPECTED_TASK_COUNTS["matched_pair_htr"]:
        raise RuntimeError(
            "matched complete plan requires every canonical fold"
        )
    for value, label in (
        (
            stage1_plan_scientific_content_sha256,
            "Stage 1 plan",
        ),
        (htr_complete_plan_content_sha256, "HTR complete plan"),
        (htr_model_tree_sha256, "HTR model tree"),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(value)) is None:
            raise ValueError(f"matched complete plan {label} SHA is invalid")

    catalog_index: dict[str, int] = {}
    catalog: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for task in canonical:
        frame = matched._make_frame(task.owner_fit_row_ids)
        fit_positions = np.asarray(task.fit_positions, dtype=np.int64)
        pairs = matched.build_training_pairs(
            frame.iloc[fit_positions].reset_index(drop=True),
            texts=[
                task.fit_texts[int(position)]
                for position in fit_positions
            ],
            treatment=task.treatment[fit_positions],
            outcome=task.outcome[fit_positions],
            propensity=task.propensity_probability[fit_positions],
            outcome_prob=task.outcome_nuisance_probability[fit_positions],
            **matched._matching_training_config(task.config),
        )
        if pairs.empty or set(pairs["label"].astype(int)) != {0, 1}:
            raise RuntimeError(
                "matched complete plan could not reconstruct production pairs"
            )
        initialization_texts = (
            pairs["control_text"].astype(str).tolist()
            + pairs["treated_text"].astype(str).tolist()
        )
        matched._assert_text_capacity(
            initialization_texts,
            extractor_config=task.config["htr_extractor"],
            stage="matched complete authenticated input plan",
        )
        extractor = task.config["htr_extractor"]
        task_catalog_positions: list[int] = []
        for text_value in initialization_texts:
            text = str(text_value)
            position = catalog_index.get(text)
            if position is None:
                position = len(catalog)
                catalog_index[text] = position
                chunks = tuple(
                    split_text_into_word_chunks(
                        text,
                        int(extractor["chunk_size_words"]),
                        int(extractor["chunk_overlap_words"]),
                        int(extractor["max_chunks"]),
                    )
                )
                catalog.append(
                    {
                        "catalog_index": position,
                        "text_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        "word_count": len(re.findall(r"\S+", text)),
                        "chunk_count": len(chunks),
                        "chunks_sha256": _sha256_json(list(chunks)),
                    }
                )
            task_catalog_positions.append(position)
        pair_columns: dict[str, Any] = {}
        for column in pairs.columns:
            values = pairs[column]
            if values.dtype.kind in {"b", "i", "u", "f", "c"}:
                pair_columns[str(column)] = {
                    "kind": "numeric_array",
                    "dtype": np.asarray(values).dtype.str,
                    "shape": [int(len(values))],
                    "content_sha256": _array_sha256(
                        np.ascontiguousarray(np.asarray(values))
                    ),
                }
            else:
                exact = values.astype(str).tolist()
                pair_columns[str(column)] = {
                    "kind": "ordered_utf8_values",
                    "count": len(exact),
                    "content_sha256": _sha256_json(exact),
                }
        task_rows.append(
            {
                "fold": int(task.fold),
                "objective": str(task.objective),
                "split_seed": int(task.split_seed),
                "htr_seed": int(task.htr_seed),
                "owner_scope_seed": int(task.owner_scope_seed),
                "owner_fit_row_ids_sha256": _sha256_json(
                    list(task.owner_fit_row_ids)
                ),
                "owner_fit_texts_sha256": _sha256_json(
                    list(task.fit_texts)
                ),
                "treatment_sha256": _array_sha256(task.treatment),
                "outcome_sha256": _array_sha256(task.outcome),
                "propensity_probability_sha256": _array_sha256(
                    task.propensity_probability
                ),
                "outcome_nuisance_probability_sha256": _array_sha256(
                    task.outcome_nuisance_probability
                ),
                "fit_positions_sha256": _array_sha256(
                    task.fit_positions
                ),
                "validation_positions_sha256": _array_sha256(
                    task.validation_positions
                ),
                "pair_count": int(len(pairs)),
                "pair_fingerprint": matched._pair_fingerprint(pairs),
                "pair_columns": pair_columns,
                "initialization_text_count": len(initialization_texts),
                "initialization_text_sha256": _sha256_json(
                    initialization_texts
                ),
                "initialization_text_catalog_positions_sha256": (
                    _sha256_json(task_catalog_positions)
                ),
                "htr_extractor": copy.deepcopy(
                    dict(task.config["htr_extractor"])
                ),
                "htr_training_configuration": (
                    matched._htr_training_configuration(task.config)
                ),
            }
        )
    body = {
        "schema_version": (
            "production_r14_matched_complete_input_plan_v1"
        ),
        "producer": "matched_pair_htr",
        "stage1_plan_scientific_content_sha256": (
            stage1_plan_scientific_content_sha256
        ),
        "htr_owner_materialized_plan_content_sha256": (
            htr_complete_plan_content_sha256
        ),
        "htr_model_tree_sha256": htr_model_tree_sha256,
        "canonical_task_count": len(canonical),
        "canonical_tasks": task_rows,
        "unique_initialization_text_catalog": catalog,
        "unique_initialization_text_count": len(catalog),
        "all_production_pair_rows_reconstructed": True,
        "all_initialization_texts_covered": True,
        "all_production_word_chunks_reconstructed_and_hashed": True,
        "worker_consumed_materialized_chunk_plan": False,
        "worker_recomputes_exact_authenticated_production_chunks": True,
        "semantic_truncation_applied": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    path = root / "matched_pair_complete_input_plan.json"
    plan = _write_self_hashed_json(path, body)
    descriptor = _published_json_descriptor(
        path,
        plan,
        producer="matched_pair_htr",
    )
    wrapped = tuple(
        _PreparedMatchedTask(
            canonical_task=task,
            complete_input_plan_content_sha256=plan["content_sha256"],
        )
        for task in canonical
    )
    return wrapped, descriptor


def _neural_cache_reopen_inventory_worker(
    payload: tuple[Any, tuple[int, ...], tuple[str, ...]],
) -> Mapping[str, Any]:
    reference, rows, texts = payload
    bound = reference.open_bound(row_ids=rows, texts=texts)
    matrices = tuple(bound.chunk_matrices(rows))
    chunk_text_rows = tuple(
        tuple(str(value) for value in row)
        for row in bound.chunk_texts(rows)
    )
    if len(matrices) != len(rows) or len(chunk_text_rows) != len(rows):
        raise RuntimeError(
            "fresh neural cache reopen changed complete row coverage"
        )
    inventory: list[dict[str, Any]] = []
    for row_id, text, matrix, chunk_texts in zip(
        rows,
        texts,
        matrices,
        chunk_text_rows,
        strict=True,
    ):
        value = np.ascontiguousarray(np.asarray(matrix))
        if (
            value.dtype != np.dtype(np.float32)
            or value.ndim != 2
            or value.shape[0] != len(chunk_texts)
            or not np.isfinite(value).all()
        ):
            raise RuntimeError(
                "fresh neural cache reopen produced an invalid row"
            )
        inventory.append(
            {
                "row_id": int(row_id),
                "text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "matrix_dtype": value.dtype.str,
                "matrix_shape": [
                    int(item) for item in value.shape
                ],
                "matrix_sha256": _array_sha256(value),
                "chunk_text_count": len(chunk_texts),
                "chunk_texts_sha256": _sha256_json(list(chunk_texts)),
            }
        )
    return {
        "worker_pid": os.getpid(),
        "row_inventory": inventory,
        "row_inventory_sha256": _sha256_json(inventory),
        "fresh_process_open_bound_completed": True,
    }


def _materialize_neural_complete_input_plan(
    *,
    root: Path,
    prepared_tasks: Sequence[_PreparedNeuralInnerTask],
    stage1_plan_scientific_content_sha256: str,
    discovery_kwargs: Mapping[str, Any],
) -> tuple[tuple[_PreparedNeuralInnerTask, ...], Mapping[str, Any]]:
    from oci.inference.neural_query_task_execution import (
        NeuralQueryAuthenticatedCacheReference,
    )
    from oci.inference.neural_query_discovery_runtime import _stable_hash

    tasks = tuple(prepared_tasks)
    if len(tasks) != _EXPECTED_TASK_COUNTS["neural_inner_folds"]:
        raise RuntimeError(
            "neural complete plan requires every canonical inner fold"
        )
    first_arguments = dict(tasks[0].canonical_task.arguments)
    reference = first_arguments.get("embedding_task_reference")
    rows = tuple(map(int, first_arguments["row_ids"]))
    texts = tuple(first_arguments["texts"])
    if (
        not isinstance(reference, NeuralQueryAuthenticatedCacheReference)
        or tuple(reference.allowed_row_ids) != rows
        or len(texts) != len(rows)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(stage1_plan_scientific_content_sha256),
        )
        is None
    ):
        raise RuntimeError(
            "neural complete plan lacks its authenticated cache authority"
        )
    global_treatment = np.ascontiguousarray(
        np.asarray(discovery_kwargs.get("treatment"), dtype=np.float64)
    )
    global_outcome = np.ascontiguousarray(
        np.asarray(discovery_kwargs.get("outcome"), dtype=np.float64)
    )
    global_fit_e = np.ascontiguousarray(
        np.asarray(discovery_kwargs.get("fit_e"), dtype=np.float64)
    )
    global_fit_m = np.ascontiguousarray(
        np.asarray(discovery_kwargs.get("fit_m"), dtype=np.float64)
    )
    if (
        global_treatment.shape != (len(rows),)
        or global_outcome.shape != (len(rows),)
        or global_fit_e.shape != (len(rows),)
        or global_fit_m.shape != (len(rows),)
        or not np.isfinite(global_treatment).all()
        or not np.isfinite(global_outcome).all()
        or not np.isfinite(global_fit_e).all()
        or not np.isfinite(global_fit_m).all()
        or tuple(map(int, discovery_kwargs.get("fit_ids", ())))
        != rows
        or tuple(discovery_kwargs.get("fit_texts", ())) != texts
        or not np.array_equal(
            np.asarray(discovery_kwargs.get("treatment")),
            np.asarray(first_arguments["treatment"]),
        )
        or not np.array_equal(
            np.asarray(discovery_kwargs.get("outcome")),
            np.asarray(first_arguments["outcome"]),
        )
        or bool(discovery_kwargs.get("outcome_binary"))
        != bool(first_arguments["outcome_binary"])
        or _closed_json_state(discovery_kwargs.get("nuisance_views"))
        != _closed_json_state(first_arguments["nuisance_views"])
        or _closed_json_state(
            discovery_kwargs.get("nuisance_stack_config")
        )
        != _closed_json_state(first_arguments["nuisance_stack_config"])
        or _closed_json_state(discovery_kwargs.get("config"))
        != _closed_json_state(first_arguments["config"])
        or int(discovery_kwargs.get("nuisance_folds", -1))
        != int(first_arguments["nuisance_folds"])
    ):
        raise RuntimeError(
            "neural complete plan differs from its production parent inputs"
        )
    runtime_texts_sha256 = _stable_hash(list(texts))
    parent_binding = _stable_hash(
        {
            "scope": "production_in_memory_no_executable_checkpoint_io",
            "runtime": "neural_query_in_memory_discovery_runtime_v2",
            "row_ids": list(rows),
            "texts_sha256": runtime_texts_sha256,
            "treatment": global_treatment.tolist(),
            "outcome": global_outcome.tolist(),
            "fit_e": global_fit_e.tolist(),
            "fit_m": global_fit_m.tolist(),
            "outcome_binary": bool(first_arguments["outcome_binary"]),
            "nuisance_views_sha256": _stable_hash(
                list(first_arguments["nuisance_views"])
            ),
            "nuisance_stack_scientific": _closed_json_state(
                first_arguments["nuisance_stack_config"]
            ),
            "query_config": first_arguments["config"].to_dict(),
        }
    )
    if parent_binding != first_arguments["parent_input_binding_sha256"]:
        raise RuntimeError(
            "neural complete plan could not replay its production parent "
            "input binding"
        )
    shared = copy.deepcopy(dict(reference.shared_cache_reference))
    shared_body = {
        key: value for key, value in shared.items() if key != "content_sha256"
    }
    if (
        shared.get("content_sha256") != _sha256_json(shared_body)
        or reference.logical_identity_sha256
        != _sha256_json(shared["logical_identity"])
        or reference.allowed_row_order_sha256
        != _sha256_json(list(rows))
    ):
        raise RuntimeError(
            "neural complete plan cache reference failed authentication"
        )
    for prepared in tasks[1:]:
        arguments = dict(prepared.canonical_task.arguments)
        other = arguments.get("embedding_task_reference")
        if (
            not isinstance(
                other,
                NeuralQueryAuthenticatedCacheReference,
            )
            or other.shared_cache_reference["content_sha256"]
            != shared["content_sha256"]
            or tuple(other.allowed_row_ids) != rows
            or tuple(arguments["row_ids"]) != rows
            or tuple(arguments["texts"]) != texts
            or arguments["parent_input_binding_sha256"]
            != first_arguments["parent_input_binding_sha256"]
            or not np.array_equal(
                arguments["treatment"],
                first_arguments["treatment"],
            )
            or not np.array_equal(
                arguments["outcome"],
                first_arguments["outcome"],
            )
            or bool(arguments["outcome_binary"])
            != bool(first_arguments["outcome_binary"])
            or _closed_json_state(arguments["nuisance_views"])
            != _closed_json_state(first_arguments["nuisance_views"])
            or int(arguments["nuisance_folds"])
            != int(first_arguments["nuisance_folds"])
            or _closed_json_state(
                arguments["nuisance_stack_config"]
            )
            != _closed_json_state(
                first_arguments["nuisance_stack_config"]
            )
            or arguments["config"].to_dict()
            != first_arguments["config"].to_dict()
        ):
            raise RuntimeError(
                "neural inner tasks changed their complete input authority"
            )
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        mp_context=context,
    ) as pool:
        fresh = pool.submit(
            _neural_cache_reopen_inventory_worker,
            (reference, rows, texts),
        ).result()
    task_rows: list[dict[str, Any]] = []
    for prepared in tasks:
        arguments = dict(prepared.canonical_task.arguments)
        task_rows.append(
            {
                "fold": int(arguments["fold"]),
                "seed": int(arguments["seed"]),
                "train_indices_sha256": _array_sha256(
                    arguments["train_indices"]
                ),
                "validation_indices_sha256": _array_sha256(
                    arguments["validation_indices"]
                ),
                "parent_input_binding_sha256": str(
                    arguments["parent_input_binding_sha256"]
                ),
                "outcome_binary": bool(arguments["outcome_binary"]),
                "treatment_sha256": _array_sha256(
                    arguments["treatment"]
                ),
                "outcome_sha256": _array_sha256(arguments["outcome"]),
                "nuisance_views_sha256": _sha256_json(
                    _closed_json_state(arguments["nuisance_views"])
                ),
                "nuisance_folds": int(arguments["nuisance_folds"]),
                "nuisance_stack_config_sha256": _sha256_json(
                    _closed_json_state(
                        arguments["nuisance_stack_config"]
                    )
                ),
                "query_config": arguments["config"].to_dict(),
                "nuisance_identity": copy.deepcopy(
                    dict(prepared.nuisance_identity)
                ),
            }
        )
    body = {
        "schema_version": (
            "production_r14_neural_complete_input_plan_v1"
        ),
        "producer": "neural_queries",
        "stage1_plan_scientific_content_sha256": (
            stage1_plan_scientific_content_sha256
        ),
        "parent_input_binding_sha256": str(
            first_arguments["parent_input_binding_sha256"]
        ),
        "ordered_row_ids": list(rows),
        "ordered_row_ids_sha256": _sha256_json(list(rows)),
        "ordered_texts_sha256": _sha256_json(list(texts)),
        "runtime_ordered_texts_sha256": runtime_texts_sha256,
        "treatment_sha256": _array_sha256(global_treatment),
        "outcome_sha256": _array_sha256(global_outcome),
        "outcome_binary": bool(first_arguments["outcome_binary"]),
        "global_treatment": {
            "dtype": global_treatment.dtype.str,
            "shape": [
                int(value) for value in global_treatment.shape
            ],
            "values": global_treatment.tolist(),
            "content_sha256": _array_sha256(global_treatment),
        },
        "global_outcome": {
            "dtype": global_outcome.dtype.str,
            "shape": [int(value) for value in global_outcome.shape],
            "values": global_outcome.tolist(),
            "content_sha256": _array_sha256(global_outcome),
        },
        "global_fit_e": {
            "dtype": global_fit_e.dtype.str,
            "shape": [int(value) for value in global_fit_e.shape],
            "values": global_fit_e.tolist(),
            "content_sha256": _array_sha256(global_fit_e),
        },
        "global_fit_m": {
            "dtype": global_fit_m.dtype.str,
            "shape": [int(value) for value in global_fit_m.shape],
            "values": global_fit_m.tolist(),
            "content_sha256": _array_sha256(global_fit_m),
        },
        "nuisance_views": _closed_json_state(
            first_arguments["nuisance_views"]
        ),
        "nuisance_stack_config": _closed_json_state(
            first_arguments["nuisance_stack_config"]
        ),
        "query_config": first_arguments["config"].to_dict(),
        "nuisance_folds": int(first_arguments["nuisance_folds"]),
        "shared_cache_reference": shared,
        "shared_cache_reference_content_sha256": shared["content_sha256"],
        "logical_identity_sha256": reference.logical_identity_sha256,
        "allowed_row_order_sha256": (
            reference.allowed_row_order_sha256
        ),
        "cache_file_inventory": copy.deepcopy(shared["cache_files"]),
        "complete_row_chunk_inventory": copy.deepcopy(
            list(fresh["row_inventory"])
        ),
        "complete_row_chunk_inventory_sha256": fresh[
            "row_inventory_sha256"
        ],
        "fresh_reopen_worker_pid": int(fresh["worker_pid"]),
        "fresh_process_open_bound_completed": fresh[
            "fresh_process_open_bound_completed"
        ],
        "canonical_inner_tasks": task_rows,
        "canonical_inner_task_count": len(task_rows),
        "all_owner_rows_and_chunks_reopened": True,
        "materialized_matrix_or_text_payload_copied_into_plan": False,
        "semantic_truncation_applied": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    path = root / "neural_complete_input_plan.json"
    plan = _write_self_hashed_json(path, body)
    descriptor = _published_json_descriptor(
        path,
        plan,
        producer="neural_queries",
    )
    wrapped = tuple(
        dataclasses.replace(
            prepared,
            complete_input_plan_content_sha256=plan["content_sha256"],
        )
        for prepared in tasks
    )
    return wrapped, descriptor


def _materialize_complete_input_plan_inventory(
    *,
    path: Path,
    htr_plan_root: Path,
    htr_plan_attestation: Mapping[str, Any],
    matched_plan: Mapping[str, Any],
    neural_plan: Mapping[str, Any],
    neural_final_fixture: _NeuralFinalFixture,
    stage1_plan_scientific_content_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    from oci.inference import role_neutral_htr_group_execution as htr

    htr_root = htr_plan_root.resolve(strict=True)
    descriptor = htr._MaterializedReusableTextPlan(
        root=str(htr_root),
        manifest_sha256=str(htr_plan_attestation["manifest_sha256"]),
        manifest_size_bytes=int(
            htr_plan_attestation["manifest_size_bytes"]
        ),
        content_sha256=str(htr_plan_attestation["content_sha256"]),
    )
    texts, row_ids, coverage, reusable = (
        htr._load_materialized_reusable_text_plan(descriptor)
    )
    if (
        not texts
        or len(texts) != len(row_ids)
        or len(coverage.chunks_by_note) != len(texts)
    ):
        raise RuntimeError("HTR complete input plan failed a fresh reopen")
    htr_manifest_path = htr_root / "manifest.json"
    htr_manifest_sha256, htr_manifest_size = _sha256_file(
        htr_manifest_path
    )
    final_plans = copy.deepcopy(
        list(neural_final_fixture.preparation["complete_input_plans"])
    )
    plan_descriptors = [
        copy.deepcopy(dict(matched_plan)),
        copy.deepcopy(dict(neural_plan)),
        *final_plans,
    ]
    if (
        len(plan_descriptors) != 5
        or any(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(value.get("content_sha256")),
            )
            is None
            for value in plan_descriptors
        )
    ):
        raise RuntimeError(
            "complete input-plan inventory omitted a producer plan"
        )
    body = {
        "schema_version": (
            "production_r14_complete_input_plan_inventory_v1"
        ),
        "stage1_plan_scientific_content_sha256": (
            stage1_plan_scientific_content_sha256
        ),
        "htr": {
            "producer": "htr_nuisance_and_prefix_conditioned_effect",
            "root": str(htr_root),
            "manifest_path": str(htr_manifest_path),
            "manifest_sha256": htr_manifest_sha256,
            "manifest_size_bytes": htr_manifest_size,
            "content_sha256": descriptor.content_sha256,
            "attestation": copy.deepcopy(dict(htr_plan_attestation)),
            "fresh_reopen_row_count": len(row_ids),
            "fresh_reopen_ordered_row_ids_sha256": _sha256_json(
                list(row_ids)
            ),
            "fresh_reopen_ordered_texts_sha256": _sha256_json(
                list(texts)
            ),
            "fresh_reopen_chunk_plan_sha256": _sha256_json(
                [list(value) for value in coverage.chunks_by_note]
            ),
            "fresh_reopen_reusable_plan_content_sha256": (
                reusable.content_sha256
            ),
            "worker_consumed_materialized_text_chunk_token_plan": True,
            "fresh_reopen_completed": True,
        },
        "matched_pair": copy.deepcopy(dict(matched_plan)),
        "neural_inner": copy.deepcopy(dict(neural_plan)),
        "neural_final_prefix_conditioned": final_plans,
        "producer_plan_count": 6,
        "canonical_phase_plan_bindings": {
            "htr_nuisance": descriptor.content_sha256,
            "htr_effect": descriptor.content_sha256,
            "matched_pair_htr": matched_plan["content_sha256"],
            "neural_inner_folds": neural_plan["content_sha256"],
            "neural_final_banks": [
                value["content_sha256"] for value in final_plans
            ],
        },
        "all_complete_inputs_bound_to_authenticated_plans": True,
        "all_plan_json_files_self_hashed_and_reopened": True,
        "semantic_truncation_applied": False,
    }
    inventory = _write_self_hashed_json(path, body)
    inventory_descriptor = _published_json_descriptor(
        path,
        inventory,
        producer="all_calibration_kernels",
    )
    return inventory, inventory_descriptor


def _capture_neural_inner(
    *,
    setup_root: Path,
    prepared: Any,
    factories: Any,
    owner: Any,
    members: tuple[Any, ...],
    devices: tuple[str, ...],
    htr_controls: Any,
    neural_controls: Any,
    cpu_budget: int,
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    from oci.inference import neural_query_context_backend as backend
    from oci.inference import neural_query_discovery_runtime as runtime

    captured_kwargs: dict[str, Any] = {}
    original_backend_fit = backend.fit_in_memory_query_discovery
    original_execute = runtime.execute_bounded_neural_query_tasks

    def observe_fit(**kwargs: Any) -> Mapping[str, Any]:
        captured_kwargs.update(kwargs)
        return runtime.fit_in_memory_query_discovery(**kwargs)

    def capture(tasks: Sequence[Any], **kwargs: Any) -> tuple[Any, ...]:
        raise _CapturedCanonicalTasks(tasks, phase=str(kwargs.get("phase")))

    backend.fit_in_memory_query_discovery = observe_fit
    runtime.execute_bounded_neural_query_tasks = capture
    try:
        bound = factories.as_mapping()["neural_query"](
            _factory_invocation(
                prepared=prepared,
                owner=owner,
                members=members,
                output_root=setup_root / "neural_capture",
                component="neural_query",
                devices=devices,
                htr_controls=htr_controls,
                neural_controls=neural_controls,
                cpu_budget=cpu_budget,
            )
        )
        try:
            bound.execute()
        except _CapturedCanonicalTasks as captured:
            tasks = captured.tasks
        else:  # pragma: no cover
            raise RuntimeError(
                "neural production builder did not expose canonical tasks"
            )
    finally:
        backend.fit_in_memory_query_discovery = original_backend_fit
        runtime.execute_bounded_neural_query_tasks = original_execute
    if (
        len(tasks) != _EXPECTED_TASK_COUNTS["neural_inner_folds"]
        or not captured_kwargs
    ):
        raise RuntimeError("neural canonical task capture changed")
    return tasks, captured_kwargs


def _prepare_one_neural_inner_nuisance(
    task: Any,
) -> tuple[_PreparedNeuralInnerTask, Mapping[str, Any]]:
    from oci.inference.neural_query_context_backend import _strata
    from oci.inference.tfidf_topic_discovery import (
        fit_joint_cross_fitted_nuisance_stacks,
    )
    from threadpoolctl import threadpool_limits

    started = time.monotonic_ns()
    worker_pid = os.getpid()
    with threadpool_limits(limits=1):
        arguments = dict(task.arguments)
        train = np.asarray(arguments["train_indices"], dtype=np.int64)
        validation = np.asarray(
            arguments["validation_indices"],
            dtype=np.int64,
        )
        treatment = np.asarray(arguments["treatment"][train], dtype=float)
        outcome = np.asarray(arguments["outcome"][train], dtype=float)
        texts = [arguments["texts"][int(index)] for index in train]
        validation_texts = [
            arguments["texts"][int(index)] for index in validation
        ]
        nuisance = fit_joint_cross_fitted_nuisance_stacks(
            texts=texts,
            treatment=treatment,
            outcome=outcome,
            outcome_binary=bool(arguments["outcome_binary"]),
            strata=_strata(
                treatment,
                outcome,
                outcome_binary=bool(arguments["outcome_binary"]),
            ),
            views=arguments["nuisance_views"],
            folds=int(arguments["nuisance_folds"]),
            random_state=int(arguments["seed"] + 10_000),
            nuisance_stack_config=arguments["nuisance_stack_config"],
            tfidf_workers=1,
            tfidf_parallel_backend="threads",
            owner_cpu_budget=1,
        )
        validation_e, _ = nuisance["treatment"]["fitted"].predict(
            validation_texts
        )
        validation_m, _ = nuisance["outcome"]["fitted"].predict(
            validation_texts
        )
        fit_e = np.asarray(
            nuisance["treatment"]["stacked_oof"],
            dtype=np.float64,
        )
        fit_m = np.asarray(
            nuisance["outcome"]["stacked_oof"],
            dtype=np.float64,
        )
        validation_e = np.asarray(validation_e, dtype=np.float64)
        validation_m = np.asarray(validation_m, dtype=np.float64)
    del nuisance
    finished = time.monotonic_ns()
    return (
        _PreparedNeuralInnerTask(
            canonical_task=task,
            train_e=fit_e,
            train_m=fit_m,
            validation_e=validation_e,
            validation_m=validation_m,
            nuisance_identity={
                "fit_e_sha256": _array_sha256(fit_e),
                "fit_m_sha256": _array_sha256(fit_m),
                "validation_e_sha256": _array_sha256(validation_e),
                "validation_m_sha256": _array_sha256(validation_m),
                "prepared_once_outside_measured_gpu_window": True,
                "production_nuisance_configuration_preserved": True,
            },
            complete_input_plan_content_sha256="pending_neural_input_plan",
        ),
        {
            "worker_pid": worker_pid,
            "started_monotonic_ns": started,
            "finished_monotonic_ns": finished,
        },
    )


def _prepare_neural_inner_nuisance(
    tasks: Sequence[Any],
    *,
    cpu_budget: int,
) -> tuple[tuple[_PreparedNeuralInnerTask, ...], Mapping[str, Any]]:
    canonical_tasks = tuple(tasks)
    if len(canonical_tasks) != _EXPECTED_TASK_COUNTS["neural_inner_folds"]:
        raise RuntimeError(
            "neural nuisance preparation requires all canonical inner tasks"
        )
    worker_count = min(len(canonical_tasks), int(cpu_budget))
    if worker_count != len(canonical_tasks):
        raise RuntimeError(
            "CPU budget cannot concurrently prepare every neural inner task"
        )
    preparation_started = time.monotonic_ns()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp.get_context("spawn"),
    ) as pool:
        outcomes = tuple(
            pool.map(
                _prepare_one_neural_inner_nuisance,
                canonical_tasks,
                chunksize=1,
            )
        )
    preparation_finished = time.monotonic_ns()
    prepared = tuple(value[0] for value in outcomes)
    worker_rows = tuple(dict(value[1]) for value in outcomes)
    intervals = [
        (
            int(row["started_monotonic_ns"]),
            int(row["finished_monotonic_ns"]),
        )
        for row in worker_rows
    ]
    if (
        len({int(row["worker_pid"]) for row in worker_rows}) != worker_count
        or _maximum_overlap(intervals) != worker_count
        or any(finish <= start for start, finish in intervals)
    ):
        raise RuntimeError(
            "neural nuisance preparation did not overlap every CPU worker"
        )
    return prepared, {
        "schema_version": (
            "production_r14_neural_nuisance_parallel_preparation_v1"
        ),
        "backend": "spawn_processes",
        "configured_worker_count": worker_count,
        "canonical_task_count": len(canonical_tasks),
        "maximum_concurrent_workers": _maximum_overlap(intervals),
        "worker_intervals": list(worker_rows),
        "started_monotonic_ns": preparation_started,
        "finished_monotonic_ns": preparation_finished,
        "wall_seconds": (
            preparation_finished - preparation_started
        )
        / 1e9,
        "per_worker_threadpool_limit": 1,
        "production_nuisance_configuration_preserved": True,
    }


def _capture_neural_final_tasks(
    *,
    discovery_kwargs: Mapping[str, Any],
    inner_prefix_results: Sequence[Mapping[str, Any]],
) -> tuple[Any, ...]:
    from oci.inference import neural_query_discovery_runtime as runtime

    original = runtime.execute_bounded_neural_query_tasks

    def replay_or_capture(
        tasks: Sequence[Any],
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        phase = str(kwargs.get("phase"))
        if phase == "inner_folds":
            return tuple(inner_prefix_results), {
                "calibration_prefix_injected": True,
            }
        raise _CapturedCanonicalTasks(tasks, phase=phase)

    runtime.execute_bounded_neural_query_tasks = replay_or_capture
    try:
        try:
            runtime.fit_in_memory_query_discovery(
                **copy.deepcopy(dict(discovery_kwargs))
            )
        except _CapturedCanonicalTasks as captured:
            tasks = captured.tasks
        else:  # pragma: no cover
            raise RuntimeError("neural final task builder was not captured")
    finally:
        runtime.execute_bounded_neural_query_tasks = original
    if len(tasks) != _EXPECTED_TASK_COUNTS["neural_final_banks"]:
        raise RuntimeError("neural final-bank canonical task count changed")
    return tasks


def _build_serial_neural_final_fixture(
    *,
    root: Path,
    discovery_kwargs: Mapping[str, Any],
    inner_prefix_results: Sequence[Mapping[str, Any]],
    base_complete_input_plan_content_sha256: str,
) -> _NeuralFinalFixture:
    results = tuple(inner_prefix_results)
    if (
        len(results) != _EXPECTED_TASK_COUNTS["neural_inner_folds"]
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(base_complete_input_plan_content_sha256),
        )
        is None
    ):
        raise RuntimeError(
            "neural final fixture requires every serial inner prefix and "
            "one complete base input plan"
        )
    latest_inner_finish = max(
        int(row["prefix_output"]["measured_finished_monotonic_ns"])
        for row in results
    )
    captured = _capture_neural_final_tasks(
        discovery_kwargs=discovery_kwargs,
        inner_prefix_results=results,
    )
    built_after_barrier_ns = time.monotonic_ns()
    if built_after_barrier_ns <= latest_inner_finish:
        raise RuntimeError(
            "neural final tasks crossed the serial inner-prefix barrier"
        )
    root.mkdir(parents=True, exist_ok=False)
    wrapped: list[_PreparedNeuralFinalTask] = []
    descriptors: list[dict[str, Any]] = []
    for task in captured:
        arguments = dict(task.arguments)
        candidates = list(arguments["candidates"])
        candidate_queries = _validated_calibration_candidate_queries(
            candidates
        )
        body = {
            "schema_version": (
                "production_r14_neural_final_prefix_conditioned_"
                "input_plan_v1"
            ),
            "producer": "neural_final_bank",
            "bank": str(arguments["bank"]),
            "bank_index": int(arguments["bank_index"]),
            "base_neural_complete_input_plan_content_sha256": (
                base_complete_input_plan_content_sha256
            ),
            "source": "serial_bounded_inner_prefix_candidates",
            "source_inner_prefix_results_sha256": _sha256_json(results),
            "row_ids_sha256": _sha256_json(
                list(arguments["row_ids"])
            ),
            "texts_sha256": _sha256_json(list(arguments["texts"])),
            "treatment_sha256": _array_sha256(
                arguments["treatment"]
            ),
            "outcome_sha256": _array_sha256(arguments["outcome"]),
            "outcome_binary": bool(arguments["outcome_binary"]),
            "fit_e_sha256": _array_sha256(arguments["fit_e"]),
            "fit_m_sha256": _array_sha256(arguments["fit_m"]),
            "candidate_count": len(candidates),
            "candidate_payload_sha256": _sha256_json(candidates),
            "candidate_ids": [
                str(value["candidate_id"]) for value in candidates
            ],
            "candidate_query_shape": [
                int(value) for value in candidate_queries.shape
            ],
            "candidate_query_dtype": candidate_queries.dtype.str,
            "candidate_query_sha256": _array_sha256(candidate_queries),
            "query_config": arguments["config"].to_dict(),
            "consensus_seed": int(
                arguments["seed"]
                + 1000
                + int(arguments["bank_index"])
            ),
            "final_refit_seed": int(
                arguments["seed"]
                + 2000
                + int(arguments["bank_index"])
            ),
            "production_consensus_builder_consumes_real_scores": True,
            "serial_prefix_conditioned_calibration_task": True,
            "full_inner_fit_or_scientific_final_output_claimed": False,
            "semantic_truncation_applied": False,
        }
        path = root / (
            f"neural_final_{int(arguments['bank_index']):03d}_"
            f"{str(arguments['bank'])}_input_plan.json"
        )
        plan = _write_self_hashed_json(path, body)
        descriptors.append(
            _published_json_descriptor(
                path,
                plan,
                producer=(
                    f"neural_final_bank:{str(arguments['bank'])}"
                ),
            )
        )
        wrapped.append(
            _PreparedNeuralFinalTask(
                canonical_task=task,
                complete_input_plan_content_sha256=plan[
                    "content_sha256"
                ],
            )
        )
    identities = [
        _neural_final_task_identity(task) for task in wrapped
    ]
    preparation_body = {
        "schema_version": (
            "production_r14_serial_neural_final_fixture_v1"
        ),
        "source": "serial_bounded_inner_prefix_candidates",
        "base_neural_complete_input_plan_content_sha256": (
            base_complete_input_plan_content_sha256
        ),
        "source_inner_prefix_results_sha256": _sha256_json(results),
        "latest_inner_prefix_finished_monotonic_ns": latest_inner_finish,
        "fixture_built_monotonic_ns": built_after_barrier_ns,
        "strict_inner_to_final_barrier_observed": True,
        "production_final_task_builder_used": True,
        "task_count": len(wrapped),
        "task_identities": identities,
        "complete_input_plans": descriptors,
        "same_typed_tasks_reused_by_both_candidates": True,
        "prefix_conditioned_not_full_inner_fit": True,
    }
    return _NeuralFinalFixture(
        tasks=tuple(wrapped),
        preparation={
            **preparation_body,
            "content_sha256": _sha256_json(preparation_body),
        },
    )


def _validated_optimizer_state_proof(
    prefix: Mapping[str, Any],
) -> dict[str, Any]:
    warmup = prefix.get("warmup_optimizer_steps")
    measured = prefix.get("measured_optimizer_steps")
    completed_warmup = prefix.get(
        "completed_warmup_optimizer_steps_at_interval_start"
    )
    start_state = prefix.get("optimizer_state_at_interval_start")
    finish_state = prefix.get("optimizer_state_at_interval_finish")
    state_verified = prefix.get("optimizer_state_verified_monotonic_ns")
    ready_started = prefix.get("ready_wait_started_monotonic_ns")
    ready_finished = prefix.get("ready_wait_finished_monotonic_ns")
    measured_started = prefix.get("measured_started_monotonic_ns")
    measured_finished = prefix.get("measured_finished_monotonic_ns")
    finish_verified = prefix.get(
        "optimizer_state_finish_verified_monotonic_ns"
    )
    integer_values = (
        warmup,
        measured,
        completed_warmup,
        state_verified,
        ready_started,
        ready_finished,
        measured_started,
        measured_finished,
        finish_verified,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_values
    ):
        raise RuntimeError("optimizer-state proof omitted integer boundaries")
    if (
        warmup < 1
        or measured < 1
        or completed_warmup != warmup
        or not (
            state_verified
            <= ready_started
            <= ready_finished
            <= measured_started
            < measured_finished
            <= finish_verified
        )
        or not isinstance(start_state, Mapping)
        or not isinstance(finish_state, Mapping)
    ):
        raise RuntimeError("optimizer-state proof changed boundary ordering")

    expected_counts = (
        "optimizer_parameter_count",
        "state_parameter_count",
        "stateless_parameter_count",
        "state_object_count",
        "state_tensor_count",
        "state_tensor_bytes",
    )
    for label, value, expected_step in (
        ("start", start_state, warmup),
        ("finish", finish_state, warmup + measured),
    ):
        if (
            value.get("schema_version")
            != "adamw_optimizer_state_boundary_observation_v2"
            or value.get("expected_optimizer_step") != expected_step
            or any(
                isinstance(value.get(key), bool)
                or not isinstance(value.get(key), int)
                or int(value[key]) <= 0
                for key in expected_counts
                if key != "stateless_parameter_count"
            )
            or isinstance(value.get("stateless_parameter_count"), bool)
            or not isinstance(value.get("stateless_parameter_count"), int)
            or int(value["stateless_parameter_count"]) < 0
            or value.get("optimizer_parameter_count")
            != (
                value.get("state_parameter_count")
                + value.get("stateless_parameter_count")
            )
            or value.get("all_optimizer_parameters_classified") is not True
            or value.get(
                "all_stateless_parameters_have_no_gradient"
            )
            is not True
            or value.get(
                "all_stateless_parameters_have_no_optimizer_state"
            )
            is not True
            or value.get(
                "all_stateful_parameters_have_finite_gradients"
            )
            is not True
            or value.get("all_required_state_keys_observed") is not True
            or value.get("all_state_tensors_finite") is not True
            or not isinstance(
                value.get("object_layout_canonical_json"), str
            )
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(value.get("object_layout_sha256")),
            )
            is None
            or hashlib.sha256(
                value["object_layout_canonical_json"].encode("utf-8")
            ).hexdigest()
            != value["object_layout_sha256"]
        ):
            raise RuntimeError(
                f"optimizer-state {label} boundary is not authenticated"
            )
    if (
        any(start_state[key] != finish_state[key] for key in expected_counts)
        or start_state["object_layout_sha256"]
        != finish_state["object_layout_sha256"]
        or prefix.get("optimizer_state_persistence_observed") is not True
    ):
        raise RuntimeError("optimizer-state persistence was not observed")
    return {
        "schema_version": "adamw_optimizer_state_persistence_proof_v2",
        "required_warmup_optimizer_steps": warmup,
        "completed_warmup_optimizer_steps": completed_warmup,
        "measured_optimizer_steps": measured,
        "start_expected_optimizer_step": start_state[
            "expected_optimizer_step"
        ],
        "finish_expected_optimizer_step": finish_state[
            "expected_optimizer_step"
        ],
        "optimizer_parameter_count": start_state[
            "optimizer_parameter_count"
        ],
        "state_parameter_count": start_state["state_parameter_count"],
        "stateless_parameter_count": start_state[
            "stateless_parameter_count"
        ],
        "state_tensor_count": start_state["state_tensor_count"],
        "state_tensor_bytes": start_state["state_tensor_bytes"],
        "all_optimizer_parameters_classified": True,
        "all_stateless_parameters_have_no_gradient": True,
        "all_stateless_parameters_have_no_optimizer_state": True,
        "all_stateful_parameters_have_finite_gradients": True,
        "object_layout_sha256": start_state["object_layout_sha256"],
        "state_verified_monotonic_ns": state_verified,
        "ready_wait_started_monotonic_ns": ready_started,
        "ready_wait_finished_monotonic_ns": ready_finished,
        "measured_started_monotonic_ns": measured_started,
        "measured_finished_monotonic_ns": measured_finished,
        "finish_state_verified_monotonic_ns": finish_verified,
        "accepted": True,
    }


def _sample_acquisition_window(
    row: Mapping[str, Any],
) -> tuple[int, int]:
    started = row.get("gpu_sample_acquisition_started_monotonic_ns")
    finished = row.get("gpu_sample_acquisition_finished_monotonic_ns")
    completion_seconds = row.get("sample_monotonic_seconds")
    if (
        isinstance(started, bool)
        or not isinstance(started, int)
        or isinstance(finished, bool)
        or not isinstance(finished, int)
        or finished < started
        or isinstance(completion_seconds, bool)
        or not isinstance(completion_seconds, (int, float))
        or not math.isfinite(float(completion_seconds))
        or abs(float(completion_seconds) * 1e9 - finished) > 16.0
    ):
        raise RuntimeError("GPU sample acquisition window is invalid")
    return started, finished


def _maximum_concurrent_child_peak_sum(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Bound simultaneous child allocation by charging each lease its peak."""

    boundaries: list[tuple[int, int, int, int, int]] = []
    for row in rows:
        start = row.get("started_monotonic_ns")
        finish = row.get("finished_monotonic_ns")
        allocated = row.get("gpu_peak_allocated_bytes")
        reserved = row.get("gpu_peak_reserved_bytes")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(finish, bool)
            or not isinstance(finish, int)
            or finish <= start
            or isinstance(allocated, bool)
            or not isinstance(allocated, int)
            or allocated <= 0
            or isinstance(reserved, bool)
            or not isinstance(reserved, int)
            or reserved <= 0
        ):
            raise RuntimeError(
                "child memory bound requires advancing task intervals "
                "and positive Torch peaks"
            )
        # Finish events sort before start events at an equal timestamp, so
        # adjacent half-open leases are not counted as concurrent.
        charge = max(allocated, reserved)
        boundaries.append((start, 1, allocated, reserved, charge))
        boundaries.append(
            (finish, -1, -allocated, -reserved, -charge)
        )
    active_count = 0
    active_allocated = 0
    active_reserved = 0
    active_charge = 0
    maximum_charge_count = 0
    maximum_allocated = 0
    maximum_reserved = 0
    maximum_charge = 0
    for (
        _timestamp,
        count_delta,
        allocated_delta,
        reserved_delta,
        charge_delta,
    ) in sorted(
        boundaries,
        key=lambda value: (value[0], value[1]),
    ):
        active_count += count_delta
        active_allocated += allocated_delta
        active_reserved += reserved_delta
        active_charge += charge_delta
        if (
            active_count < 0
            or active_allocated < 0
            or active_reserved < 0
            or active_charge < 0
        ):
            raise RuntimeError("child memory interval ledger did not close")
        maximum_allocated = max(maximum_allocated, active_allocated)
        maximum_reserved = max(maximum_reserved, active_reserved)
        if active_charge > maximum_charge:
            maximum_charge = active_charge
            maximum_charge_count = active_count
    if any(
        value != 0
        for value in (
            active_count,
            active_allocated,
            active_reserved,
            active_charge,
        )
    ):
        raise RuntimeError("child memory interval ledger did not close")
    return {
        "maximum_concurrent_task_count": maximum_charge_count,
        "maximum_concurrent_child_allocated_peak_sum_bytes": (
            maximum_allocated
        ),
        "maximum_concurrent_child_reserved_peak_sum_bytes": (
            maximum_reserved
        ),
        "maximum_concurrent_child_allocator_charge_sum_bytes": (
            maximum_charge
        ),
    }


def _conservative_child_peak_bounds(
    phase_rows: Mapping[str, Mapping[str, Any]],
    *,
    devices: Sequence[str],
) -> dict[str, dict[str, Any]]:
    selected = tuple(str(value) for value in devices)
    if len(set(selected)) != len(selected):
        raise ValueError("child memory bound devices must be unique")
    by_phase: dict[str, dict[str, Any]] = {}
    observed_devices: set[str] = set()
    for phase in _PHASES:
        try:
            task_intervals = phase_rows[phase][
                "task_execution_attestation"
            ]["task_intervals"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"{phase} omitted child task memory intervals"
            ) from exc
        by_phase[phase] = {}
        for device in selected:
            rows = [
                dict(row)
                for row in task_intervals
                if str(row.get("device")) == device
            ]
            bounds = _maximum_concurrent_child_peak_sum(rows)
            by_phase[phase][device] = {
                "task_count": len(rows),
                **bounds,
            }
        observed_devices.update(
            str(row.get("device")) for row in task_intervals
        )
    unexpected = observed_devices - set(selected)
    if unexpected:
        raise RuntimeError(
            "child memory intervals used an unselected device: "
            f"{sorted(unexpected)}"
        )

    result: dict[str, dict[str, Any]] = {}
    for device in selected:
        phase_bounds = {
            phase: copy.deepcopy(by_phase[phase][device])
            for phase in _PHASES
        }
        bounding_phase = max(
            _PHASES,
            key=lambda phase: int(
                phase_bounds[phase][
                    "maximum_concurrent_child_allocator_charge_sum_bytes"
                ]
            ),
        )
        result[device] = {
            "bounding_phase": bounding_phase,
            "maximum_concurrent_child_allocated_peak_sum_bytes": max(
                int(
                    phase_bounds[phase][
                        "maximum_concurrent_child_allocated_peak_sum_bytes"
                    ]
                )
                for phase in _PHASES
            ),
            "maximum_concurrent_child_reserved_peak_sum_bytes": max(
                int(
                    phase_bounds[phase][
                        "maximum_concurrent_child_reserved_peak_sum_bytes"
                    ]
                )
                for phase in _PHASES
            ),
            "maximum_concurrent_child_allocator_charge_sum_bytes": int(
                phase_bounds[bounding_phase][
                    "maximum_concurrent_child_allocator_charge_sum_bytes"
                ]
            ),
            "per_phase": phase_bounds,
        }
    return result


def _gpu_summary(
    samples: Sequence[Mapping[str, Any]],
    *,
    phase_rows: Mapping[str, Mapping[str, Any]],
    post_warmup_sampling_rows: Sequence[Mapping[str, Any]],
    sampler_completed_monotonic_ns: int,
    run_finished_monotonic_ns: int,
    devices: Sequence[str],
    maximum_fraction: float,
    minimum_headroom_bytes: int,
) -> dict[str, Any]:
    def valid_sampling_proof(row: Mapping[str, Any]) -> bool:
        start = row.get("measured_started_monotonic_ns")
        finish = row.get("measured_finished_monotonic_ns")
        device = row.get("device")
        count = row.get(
            "host_gpu_acquisition_windows_wholly_inside_interval"
        )
        proof = row.get("optimizer_state_proof")
        if not isinstance(proof, Mapping):
            return False
        proof_integers = (
            proof.get("required_warmup_optimizer_steps"),
            proof.get("completed_warmup_optimizer_steps"),
            proof.get("measured_optimizer_steps"),
            proof.get("start_expected_optimizer_step"),
            proof.get("finish_expected_optimizer_step"),
            proof.get("optimizer_parameter_count"),
            proof.get("state_parameter_count"),
            proof.get("stateless_parameter_count"),
            proof.get("state_tensor_count"),
            proof.get("state_tensor_bytes"),
            proof.get("state_verified_monotonic_ns"),
            proof.get("ready_wait_started_monotonic_ns"),
            proof.get("ready_wait_finished_monotonic_ns"),
            proof.get("measured_started_monotonic_ns"),
            proof.get("measured_finished_monotonic_ns"),
            proof.get("finish_state_verified_monotonic_ns"),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in proof_integers
        ):
            return False
        warmup = int(proof["required_warmup_optimizer_steps"])
        measured = int(proof["measured_optimizer_steps"])
        if not isinstance(device, str) or device not in devices:
            return False
        try:
            observed_count = sum(
                int(
                    start <= acquisition_start
                    and acquisition_start <= acquisition_finish
                    and acquisition_finish <= finish
                )
                for sample in samples
                if str(sample.get("device")) == device
                for acquisition_start, acquisition_finish in (
                    _sample_acquisition_window(sample),
                )
            )
        except (RuntimeError, TypeError, ValueError):
            return False
        return bool(
            row.get("accepted") is True
            and proof.get("schema_version")
            == "adamw_optimizer_state_persistence_proof_v2"
            and proof.get("accepted") is True
            and warmup >= 1
            and measured >= 1
            and proof["completed_warmup_optimizer_steps"] == warmup
            and proof["start_expected_optimizer_step"] == warmup
            and proof["finish_expected_optimizer_step"]
            == warmup + measured
            and proof["optimizer_parameter_count"]
            == (
                proof["state_parameter_count"]
                + proof["stateless_parameter_count"]
            )
            and proof["state_parameter_count"] >= 1
            and proof["stateless_parameter_count"] >= 0
            and proof["state_tensor_count"] >= 1
            and proof["state_tensor_bytes"] >= 1
            and proof.get("all_optimizer_parameters_classified") is True
            and proof.get(
                "all_stateless_parameters_have_no_gradient"
            )
            is True
            and proof.get(
                "all_stateless_parameters_have_no_optimizer_state"
            )
            is True
            and proof.get(
                "all_stateful_parameters_have_finite_gradients"
            )
            is True
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(proof.get("object_layout_sha256")),
            )
            is not None
            and isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(finish, int)
            and not isinstance(finish, bool)
            and finish > start
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count == observed_count
            and observed_count >= 1
            and proof["measured_started_monotonic_ns"] == start
            and proof["measured_finished_monotonic_ns"] == finish
            and proof["state_verified_monotonic_ns"]
            <= proof["ready_wait_started_monotonic_ns"]
            <= proof["ready_wait_finished_monotonic_ns"]
            <= start
            < finish
            <= proof["finish_state_verified_monotonic_ns"]
        )

    coverage_keys = [
        (
            row.get("phase"),
            row.get("canonical_task_index"),
            row.get("atomic_prefix_index"),
            row.get("device"),
        )
        for row in post_warmup_sampling_rows
    ]
    sampling_proof_observed = bool(post_warmup_sampling_rows) and (
        len(set(coverage_keys)) == len(coverage_keys)
    ) and all(
        valid_sampling_proof(row) for row in post_warmup_sampling_rows
    )
    if (
        not post_warmup_sampling_rows
        or not sampling_proof_observed
    ):
        raise RuntimeError(
            "memory acceptance requires a host sample inside every "
            "post-warmup optimizer interval"
        )
    child_bounds = _conservative_child_peak_bounds(
        phase_rows,
        devices=devices,
    )
    task_intervals = [
        row
        for phase in _PHASES
        for row in phase_rows[phase]["task_execution_attestation"][
            "task_intervals"
        ]
    ]
    if not task_intervals:
        raise RuntimeError("memory acceptance omitted all task intervals")
    first_task_started = min(
        int(row["started_monotonic_ns"]) for row in task_intervals
    )
    last_task_finished = max(
        int(row["finished_monotonic_ns"]) for row in task_intervals
    )
    if (
        isinstance(sampler_completed_monotonic_ns, bool)
        or not isinstance(sampler_completed_monotonic_ns, int)
        or isinstance(run_finished_monotonic_ns, bool)
        or not isinstance(run_finished_monotonic_ns, int)
        or sampler_completed_monotonic_ns < last_task_finished
        or run_finished_monotonic_ns < sampler_completed_monotonic_ns
    ):
        raise RuntimeError("GPU sampler completion ordering is invalid")
    rows: list[dict[str, Any]] = []
    accepted = True
    for device in devices:
        observed = sorted(
            (
                dict(row)
                for row in samples
                if str(row.get("device")) == device
            ),
            key=lambda row: _sample_acquisition_window(row)[1],
        )
        if len(observed) < 2:
            raise RuntimeError("continuous GPU sampler omitted a selected device")
        totals = {int(row["memory_total_bytes"]) for row in observed}
        uuids = {str(row["uuid"]) for row in observed}
        if len(totals) != 1 or len(uuids) != 1:
            raise RuntimeError("GPU identity changed during calibration")
        total = next(iter(totals))
        if total <= 0:
            raise RuntimeError("GPU sampler reported nonpositive total memory")
        used_values = [
            int(row["memory_used_bytes"]) for row in observed
        ]
        if any(value < 0 or value > total for value in used_values):
            raise RuntimeError("GPU sampler reported invalid used memory")
        pre_task_started_ns, pre_task_finished_ns = (
            _sample_acquisition_window(observed[0])
        )
        final_sample_started_ns, final_sample_finished_ns = (
            _sample_acquisition_window(observed[-1])
        )
        if pre_task_finished_ns > first_task_started:
            raise RuntimeError(
                "GPU sampler omitted a pre-task external-memory baseline"
            )
        if (
            final_sample_started_ns < last_task_finished
            or final_sample_finished_ns > sampler_completed_monotonic_ns
        ):
            raise RuntimeError(
                "GPU sampler omitted a post-task terminal acquisition"
            )
        external_baseline = used_values[0]
        host_peak = max(used_values)
        child_incremental_bound = int(
            child_bounds[str(device)][
                "maximum_concurrent_child_allocator_charge_sum_bytes"
            ]
        )
        child_total_bound = external_baseline + child_incremental_bound
        acceptance_peak = max(host_peak, child_total_bound)
        headroom = total - acceptance_peak
        fraction = acceptance_peak / total
        safe = (
            fraction <= float(maximum_fraction)
            and headroom >= int(minimum_headroom_bytes)
        )
        accepted = accepted and safe
        if host_peak > child_total_bound:
            peak_source = "host_peak"
        elif child_total_bound > host_peak:
            peak_source = "external_baseline_plus_child_peak_bound"
        else:
            peak_source = "host_and_child_bound_tie"
        rows.append(
            {
                "device": device,
                "uuid": next(iter(uuids)),
                "sample_count": len(observed),
                "memory_total_bytes": total,
                "pre_task_sample_acquisition_started_monotonic_ns": (
                    pre_task_started_ns
                ),
                "pre_task_sample_acquisition_finished_monotonic_ns": (
                    pre_task_finished_ns
                ),
                "first_task_started_monotonic_ns": first_task_started,
                "last_task_finished_monotonic_ns": last_task_finished,
                "final_sample_acquisition_started_monotonic_ns": (
                    final_sample_started_ns
                ),
                "final_sample_acquisition_finished_monotonic_ns": (
                    final_sample_finished_ns
                ),
                "sampler_completed_monotonic_ns": (
                    sampler_completed_monotonic_ns
                ),
                "run_finished_monotonic_ns": run_finished_monotonic_ns,
                "pre_task_external_baseline_memory_used_bytes": (
                    external_baseline
                ),
                "initial_memory_used_bytes": external_baseline,
                "host_peak_memory_used_bytes": host_peak,
                "peak_memory_used_bytes": host_peak,
                "aggregate_same_gpu_incremental_peak_bytes": max(
                    0, host_peak - external_baseline
                ),
                "conservative_concurrent_child_peak_bound": copy.deepcopy(
                    child_bounds[str(device)]
                ),
                "conservative_child_incremental_bound_bytes": (
                    child_incremental_bound
                ),
                "conservative_child_allocated_peak_sum_bytes": int(
                    child_bounds[str(device)][
                        "maximum_concurrent_child_allocated_peak_sum_bytes"
                    ]
                ),
                "conservative_child_reserved_peak_sum_bytes": int(
                    child_bounds[str(device)][
                        "maximum_concurrent_child_reserved_peak_sum_bytes"
                    ]
                ),
                "pre_task_external_plus_child_bound_bytes": (
                    child_total_bound
                ),
                "memory_acceptance_peak_bytes": acceptance_peak,
                "memory_acceptance_peak_source": peak_source,
                "peak_allocation_fraction": fraction,
                "minimum_headroom_bytes": headroom,
                "memory_safety_accepted": safe,
            }
        )
    return {
        "devices": rows,
        "continuous_host_level_sampling": True,
        "same_gpu_process_allocations_aggregated": True,
        "pre_task_external_baseline_sampled_before_task_launch": True,
        "post_task_terminal_acquisition_completed_before_sampler_exit": True,
        "child_bound_sums_concurrent_task_torch_peaks_by_device": True,
        "child_bound_reports_allocated_and_reserved_peaks": True,
        "child_bound_charges_maximum_allocated_or_reserved_per_task": True,
        "child_bound_takes_maximum_across_phases": True,
        "memory_acceptance_uses_maximum_of_host_and_child_bound": True,
        "host_sample_inside_every_post_warmup_optimizer_interval": (
            sampling_proof_observed
        ),
        "post_warmup_optimizer_state_persistent_during_sampled_intervals": (
            sampling_proof_observed
        ),
        "post_warmup_interval_count": len(post_warmup_sampling_rows),
        "memory_safety_accepted": accepted,
    }


def _maximum_overlap(intervals: Sequence[tuple[int, int]]) -> int:
    boundaries = [
        (start, 1) for start, _finish in intervals
    ] + [
        (finish, -1) for _start, finish in intervals
    ]
    active = 0
    peak = 0
    for _timestamp, delta in sorted(
        boundaries,
        key=lambda value: (value[0], value[1]),
    ):
        active += delta
        peak = max(peak, active)
    if active != 0:
        raise RuntimeError("prefix interval ledger did not close")
    return peak


def _prefix_intervals(
    value: Mapping[str, Any],
) -> tuple[tuple[int, int, int], ...]:
    prefix = value["prefix_output"]
    atomic = (
        tuple(prefix["banks"].values())
        if value.get("phase") == "neural_inner_folds"
        else (prefix,)
    )
    return tuple(
        (
            int(row["measured_started_monotonic_ns"]),
            int(row["measured_finished_monotonic_ns"]),
            int(row["measured_optimizer_steps"]),
        )
        for row in atomic
    )


def _interval_union_seconds(
    intervals: Sequence[tuple[int, int]],
) -> float:
    ordered = sorted((int(start), int(finish)) for start, finish in intervals)
    if not ordered or any(finish <= start for start, finish in ordered):
        raise ValueError("measured interval union requires advancing intervals")
    total = 0
    active_start, active_finish = ordered[0]
    for start, finish in ordered[1:]:
        if start <= active_finish:
            active_finish = max(active_finish, finish)
        else:
            total += active_finish - active_start
            active_start, active_finish = start, finish
    total += active_finish - active_start
    return total / 1e9


def _phase_summary(
    results: Sequence[Mapping[str, Any]],
    *,
    execution_attestation: Mapping[str, Any],
    expected_parallelism: int,
) -> dict[str, Any]:
    atomic_prefixes = [
        prefix
        for value in results
        for prefix in (
            tuple(value["prefix_output"]["banks"].values())
            if value.get("phase") == "neural_inner_folds"
            else (value["prefix_output"],)
        )
    ]
    state_proofs = [
        _validated_optimizer_state_proof(prefix)
        for prefix in atomic_prefixes
    ]
    state_persistence_observed = bool(state_proofs) and all(
        proof["accepted"] is True for proof in state_proofs
    )
    if not state_persistence_observed:
        raise RuntimeError(
            "one or more optimizer prefixes omitted warmup/state persistence"
        )
    intervals = [
        interval
        for value in results
        for interval in _prefix_intervals(value)
    ]
    starts = [value[0] for value in intervals]
    finishes = [value[1] for value in intervals]
    total_steps = sum(value[2] for value in intervals)
    makespan = (max(finishes) - min(starts)) / 1e9
    union_seconds = _interval_union_seconds(
        [(start, finish) for start, finish, _steps in intervals]
    )
    if makespan <= 0.0:
        raise RuntimeError("measured optimizer makespan did not advance")
    overlap = _maximum_overlap(
        [(start, finish) for start, finish, _steps in intervals]
    )
    if int(expected_parallelism) > 1 and overlap != int(expected_parallelism):
        raise RuntimeError(
            "post-warmup measured prefixes did not overlap every configured lease"
        )
    if (
        int(execution_attestation["maximum_concurrent_leases"])
        != int(expected_parallelism)
    ):
        raise RuntimeError("task leases did not reach configured concurrency")
    child_allocated_peaks = [
        row.get("gpu_peak_allocated_bytes")
        for row in execution_attestation["task_intervals"]
        if str(row.get("device")).startswith("cuda:")
    ]
    child_reserved_peaks = [
        row.get("gpu_peak_reserved_bytes")
        for row in execution_attestation["task_intervals"]
        if str(row.get("device")).startswith("cuda:")
    ]
    if (
        len(child_allocated_peaks) != len(results)
        or len(child_reserved_peaks) != len(results)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in (
                *child_allocated_peaks,
                *child_reserved_peaks,
            )
        )
    ):
        raise RuntimeError(
            "one or more CUDA prefix tasks omitted an allocated/reserved "
            "child Torch memory peak"
        )
    return {
        "canonical_task_count": len(results),
        "configured_parallelism": int(expected_parallelism),
        "measured_optimizer_steps": total_steps,
        "measured_makespan_seconds": makespan,
        "measured_interval_union_seconds": union_seconds,
        "aggregate_measured_optimizer_steps_per_second": (
            total_steps / union_seconds
        ),
        "measured_optimizer_maximum_concurrency": overlap,
        "ready_barrier_enforced_after_warmup": (
            state_persistence_observed
        ),
        "optimizer_state_persistence_observed_for_every_atomic_prefix": (
            state_persistence_observed
        ),
        "optimizer_state_proof_count": len(state_proofs),
        "cuda_synchronized_at_measured_window_boundaries": True,
        "instrumentation_added_per_step_cuda_synchronization": False,
        "every_cuda_task_reported_positive_allocated_and_reserved_torch_peaks": True,
        "maximum_child_torch_peak_allocated_bytes": max(
            child_allocated_peaks
        ),
        "maximum_child_torch_peak_reserved_bytes": max(
            child_reserved_peaks
        ),
        "task_execution_attestation": copy.deepcopy(
            dict(execution_attestation)
        ),
        "prefix_results": copy.deepcopy(list(results)),
    }


def _barrier_wrapped_tasks(
    *,
    context: Any,
    run_root: Path,
    phase: str,
    tasks: Sequence[Any],
    identities: Sequence[Mapping[str, Any]],
    parallelism: int,
    warmup_steps: int,
    measured_steps: int,
) -> tuple[_PrefixTask, ...]:
    if len(tasks) != len(identities):
        raise ValueError("canonical tasks and identities differ")
    barriers: dict[int, Any] = {}
    for start in range(0, len(tasks), int(parallelism)):
        parties = min(int(parallelism), len(tasks) - start)
        barriers[start // int(parallelism)] = context.Barrier(parties)
    captures_complete_parameters = phase in {
        "htr_nuisance",
        "htr_effect",
        "matched_pair_htr",
    }
    bundle_directory = run_root / "parameter_bundles" / phase
    if captures_complete_parameters:
        bundle_directory.mkdir(parents=True, exist_ok=False)
    return tuple(
        _PrefixTask(
            phase=phase,
            canonical_index=index,
            canonical_task=canonical,
            canonical_identity=copy.deepcopy(dict(identities[index])),
            ready_barrier=barriers[index // int(parallelism)],
            warmup_steps=int(warmup_steps),
            measured_steps=int(measured_steps),
            parameter_bundle_path=(
                str(bundle_directory / f"task_{index:03d}.bin")
                if captures_complete_parameters
                else None
            ),
            parameter_bundle_relative_path=(
                (
                    Path("parameter_bundles")
                    / phase
                    / f"task_{index:03d}.bin"
                ).as_posix()
                if captures_complete_parameters
                else None
            ),
        )
        for index, canonical in enumerate(tasks)
    )


def _execute_phase(
    *,
    manager: Any,
    run_root: Path,
    phase: str,
    tasks: Sequence[Any],
    identities: Sequence[Mapping[str, Any]],
    resource_plan: Any,
    parallelism: int,
    warmup_steps: int,
    measured_steps: int,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    from oci.inference.neural_query_task_execution import (
        execute_bounded_neural_query_tasks,
    )

    wrapped = _barrier_wrapped_tasks(
        context=manager,
        run_root=run_root,
        phase=phase,
        tasks=tasks,
        identities=identities,
        parallelism=parallelism,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )
    values, attestation = execute_bounded_neural_query_tasks(
        wrapped,
        task_names=tuple(
            f"{phase}_{index:03d}" for index in range(len(wrapped))
        ),
        resource_plan=resource_plan,
        worker=_WORKER_BY_PHASE[phase],
        parallelism=int(parallelism),
        phase=phase,
    )
    return tuple(values), attestation


def _build_prefix_conditioned_htr_effect_fixture(
    *,
    nuisance_tasks: Sequence[Any],
    nuisance_results: Sequence[Mapping[str, Any]],
    owner_scope_seed: int,
) -> _HTREffectFixture:
    from oci.inference import role_neutral_htr_group_execution as htr

    canonical_tasks = tuple(nuisance_tasks)
    results = tuple(nuisance_results)
    expected = _EXPECTED_TASK_COUNTS["htr_nuisance"]
    if len(canonical_tasks) != expected or len(results) != expected:
        raise RuntimeError(
            "HTR effect fixture requires every canonical nuisance prefix"
        )
    row_count = int(len(canonical_tasks[0].treatment))
    nuisance_oof_e = np.full(row_count, np.nan, dtype=np.float64)
    nuisance_oof_m = np.full(row_count, np.nan, dtype=np.float64)
    assignment_count = np.zeros(row_count, dtype=np.int64)
    fold_rows: list[dict[str, Any]] = []
    latest_nuisance_finish = 0
    for canonical_index, (canonical, result) in enumerate(
        zip(canonical_tasks, results, strict=True)
    ):
        if (
            int(result.get("canonical_index", -1)) != canonical_index
            or result.get("phase") != "htr_nuisance"
            or result.get("canonical_identity") != _htr_task_identity(canonical)
        ):
            raise RuntimeError(
                "HTR nuisance prefix changed canonical order before effect barrier"
            )
        payload = result.get("prefix_conditioned_nuisance_oof")
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                "HTR nuisance prefix omitted calibrated validation predictions"
            )
        validation_positions = np.asarray(
            payload.get("validation_positions"),
            dtype=np.int64,
        )
        validation_e = np.ascontiguousarray(
            np.asarray(payload.get("validation_e_hat"), dtype=np.float64)
        )
        validation_m = np.ascontiguousarray(
            np.asarray(payload.get("validation_m_hat"), dtype=np.float64)
        )
        if (
            not np.array_equal(
                validation_positions,
                np.asarray(canonical.validation_positions, dtype=np.int64),
            )
            or validation_e.shape != validation_positions.shape
            or validation_m.shape != validation_positions.shape
            or not np.isfinite(validation_e).all()
            or not np.isfinite(validation_m).all()
            or payload.get("validation_e_hat_dtype") != validation_e.dtype.str
            or payload.get("validation_m_hat_dtype") != validation_m.dtype.str
            or payload.get("validation_e_hat_sha256")
            != _array_sha256(validation_e)
            or payload.get("validation_m_hat_sha256")
            != _array_sha256(validation_m)
            or payload.get("production_probability_calibration_applied")
            is not True
            or payload.get("complete_fit_and_validation_prediction_applied")
            is not True
        ):
            raise RuntimeError(
                "HTR nuisance prefix validation prediction proof changed"
            )
        nuisance_oof_e[validation_positions] = validation_e
        nuisance_oof_m[validation_positions] = validation_m
        assignment_count[validation_positions] += 1
        latest_nuisance_finish = max(
            latest_nuisance_finish,
            int(result["prefix_output"]["measured_finished_monotonic_ns"]),
        )
        fold_rows.append(
            {
                "fold": int(canonical.fold),
                "validation_positions_sha256": _array_sha256(
                    validation_positions
                ),
                "validation_e_hat_sha256": _array_sha256(validation_e),
                "validation_m_hat_sha256": _array_sha256(validation_m),
            }
        )
    if (
        not np.array_equal(
            assignment_count,
            np.ones(row_count, dtype=np.int64),
        )
        or not np.isfinite(nuisance_oof_e).all()
        or not np.isfinite(nuisance_oof_m).all()
    ):
        raise RuntimeError(
            "HTR nuisance prefixes did not assemble one complete OOF prediction"
        )
    first = canonical_tasks[0]
    first_plan = first.text_authority.materialized_plan
    if first_plan is None:
        raise RuntimeError("HTR nuisance task lacks its materialized input plan")
    if any(
        task.config != first.config
        or task.model_marker != first.model_marker
        or task.text_authority.materialized_plan is None
        or task.text_authority.materialized_plan.content_sha256
        != first_plan.content_sha256
        or not np.array_equal(task.treatment, first.treatment)
        or not np.array_equal(task.outcome, first.outcome)
        for task in canonical_tasks[1:]
    ):
        raise RuntimeError("HTR nuisance task authorities changed across folds")
    built_after_barrier_ns = time.monotonic_ns()
    if built_after_barrier_ns <= latest_nuisance_finish:
        raise RuntimeError("HTR effect tasks crossed the nuisance prefix barrier")
    plan = htr._build_effect_fold_tasks(
        owner_scope_seed=int(owner_scope_seed),
        text_count=row_count,
        treatment=first.treatment,
        outcome=first.outcome,
        nuisance_oof_e=nuisance_oof_e,
        nuisance_oof_m=nuisance_oof_m,
        config=first.config,
        model_marker=first.model_marker,
        operational_controls=first.operational_controls,
        text_authority=first.text_authority,
    )
    if len(plan.tasks) != _EXPECTED_TASK_COUNTS["htr_effect"]:
        raise RuntimeError("production HTR effect task count changed")
    oof_e_sha256 = _array_sha256(nuisance_oof_e)
    oof_m_sha256 = _array_sha256(nuisance_oof_m)
    wrapped = tuple(
        _PreparedHTREffectTask(
            canonical_task=task,
            source_nuisance_oof_e_sha256=oof_e_sha256,
            source_nuisance_oof_m_sha256=oof_m_sha256,
        )
        for task in plan.tasks
    )
    body = {
        "schema_version": (
            "production_r14_htr_effect_prefix_input_preparation_v1"
        ),
        "source": "serial_bounded_nuisance_prefix",
        "production_builder": (
            "oci.inference.role_neutral_htr_group_execution."
            "_build_effect_fold_tasks"
        ),
        "nuisance_fold_count": len(canonical_tasks),
        "nuisance_fold_validation_outputs": fold_rows,
        "row_count": row_count,
        "assignment_count_sha256": _array_sha256(assignment_count),
        "nuisance_oof_e_sha256": oof_e_sha256,
        "nuisance_oof_m_sha256": oof_m_sha256,
        "clipped_e_sha256": _array_sha256(plan.clipped_e),
        "y_residual_sha256": _array_sha256(plan.y_residual),
        "t_residual_sha256": _array_sha256(plan.t_residual),
        "pseudo_outcome_sha256": _array_sha256(plan.pseudo_outcome),
        "eligible_sha256": _array_sha256(plan.eligible),
        "effect_task_identities": [
            _htr_effect_task_identity(task) for task in wrapped
        ],
        "latest_nuisance_prefix_finished_monotonic_ns": (
            latest_nuisance_finish
        ),
        "effect_fixture_built_monotonic_ns": built_after_barrier_ns,
        "strict_nuisance_to_effect_barrier_observed": True,
        "prefix_conditioned_not_full_nuisance_fit": True,
    }
    return _HTREffectFixture(
        tasks=wrapped,
        preparation={**body, "content_sha256": _sha256_json(body)},
    )


def _run_prefix_suite(
    *,
    run_root: Path,
    prepared_tasks: Mapping[str, Sequence[Any]],
    discovery_kwargs: Mapping[str, Any],
    execution_devices: tuple[str, ...],
    sampler_devices: tuple[str, ...],
    slots_per_device: int,
    args: argparse.Namespace,
    baseline: bool,
    owner_scope_seed: int,
    htr_effect_fixture: _HTREffectFixture | None,
    neural_final_fixture: _NeuralFinalFixture | None,
    neural_final_plan_root: Path | None,
) -> tuple[
    dict[str, Any],
    _HTREffectFixture,
    _NeuralFinalFixture,
]:
    from oci.inference.neural_query_operational_controls import (
        RoleNeutralNeuralQueryTaskResourcePlan,
    )
    from oci.inference.role_neutral_performance_benchmark import (
        _CandidateGpuSampler,
    )

    run_root.mkdir(parents=True, exist_ok=False)
    capacity = len(execution_devices) * int(slots_per_device)
    configured = {
        "htr_nuisance": (
            1 if baseline else min(args.htr_candidate_fold_parallelism, capacity)
        ),
        "htr_effect": (
            1 if baseline else min(args.htr_candidate_fold_parallelism, capacity)
        ),
        "matched_pair_htr": (
            1 if baseline else min(args.htr_candidate_fold_parallelism, capacity)
        ),
        "neural_inner_folds": (
            1
            if baseline
            else min(args.neural_candidate_inner_fold_parallelism, capacity)
        ),
        "neural_final_banks": (
            1
            if baseline
            else min(args.neural_candidate_bank_parallelism, capacity)
        ),
    }
    if not baseline and configured != _EXPECTED_CANDIDATE_PARALLELISM:
        raise RuntimeError("candidate slots cannot exercise required R14 leases")
    sampler = _CandidateGpuSampler(
        devices=sampler_devices,
        interval_seconds=float(args.gpu_sample_interval_seconds),
    )
    phase_rows: dict[str, Any] = {}
    active_effect_fixture = htr_effect_fixture
    active_final_fixture = neural_final_fixture
    started = time.monotonic_ns()
    context = mp.get_context("spawn")
    with sampler, context.Manager() as manager:
        for phase in _PHASES:
            if phase == "htr_effect":
                if active_effect_fixture is None:
                    active_effect_fixture = (
                        _build_prefix_conditioned_htr_effect_fixture(
                            nuisance_tasks=prepared_tasks["htr_nuisance"],
                            nuisance_results=phase_rows["htr_nuisance"][
                                "prefix_results"
                            ],
                            owner_scope_seed=int(owner_scope_seed),
                        )
                    )
                tasks = active_effect_fixture.tasks
                if not baseline:
                    candidate_controls = prepared_tasks["htr_nuisance"][
                        0
                    ].operational_controls
                    tasks = tuple(
                        dataclasses.replace(
                            task,
                            canonical_task=dataclasses.replace(
                                task.canonical_task,
                                operational_controls=candidate_controls,
                            ),
                        )
                        for task in tasks
                    )
                identities = tuple(
                    _htr_effect_task_identity(task) for task in tasks
                )
            elif phase == "neural_final_banks":
                if active_final_fixture is None:
                    if neural_final_plan_root is None:
                        raise RuntimeError(
                            "serial neural final fixture lacks a plan root"
                        )
                    inner_tasks = tuple(
                        prepared_tasks["neural_inner_folds"]
                    )
                    if (
                        not inner_tasks
                        or not isinstance(
                            inner_tasks[0],
                            _PreparedNeuralInnerTask,
                        )
                    ):
                        raise RuntimeError(
                            "neural final fixture lacks its base input plan"
                        )
                    active_final_fixture = (
                        _build_serial_neural_final_fixture(
                            root=neural_final_plan_root,
                            discovery_kwargs=discovery_kwargs,
                            inner_prefix_results=phase_rows[
                                "neural_inner_folds"
                            ]["prefix_results"],
                            base_complete_input_plan_content_sha256=(
                                inner_tasks[
                                    0
                                ].complete_input_plan_content_sha256
                            ),
                        )
                    )
                tasks = active_final_fixture.tasks
                identities = tuple(
                    _neural_final_task_identity(task) for task in tasks
                )
            else:
                tasks = tuple(prepared_tasks[phase])
                if phase == "htr_nuisance":
                    identities = tuple(_htr_task_identity(task) for task in tasks)
                elif phase == "matched_pair_htr":
                    identities = tuple(
                        _matched_task_identity(task) for task in tasks
                    )
                else:
                    identities = tuple(
                        _neural_inner_task_identity(task) for task in tasks
                    )
            parallelism = int(configured[phase])
            plan = RoleNeutralNeuralQueryTaskResourcePlan(
                devices=execution_devices,
                inner_fold_parallelism=parallelism,
                fold_parallel_backend="processes",
                fold_slots_per_device=int(slots_per_device),
                bank_parallelism=parallelism,
                worker_cpu_threads=1,
                owner_cpu_budget=int(args.cpu_budget),
            )
            results, execution = _execute_phase(
                manager=manager,
                run_root=run_root,
                phase=phase,
                tasks=tasks,
                identities=identities,
                resource_plan=plan,
                parallelism=parallelism,
                warmup_steps=int(args.warmup_optimizer_steps),
                measured_steps=int(args.measured_optimizer_steps),
            )
            phase_rows[phase] = _phase_summary(
                results,
                execution_attestation=execution,
                expected_parallelism=parallelism,
            )
    if active_effect_fixture is None:
        raise RuntimeError("HTR effect fixture was not built")
    if active_final_fixture is None:
        raise RuntimeError("neural final fixture was not built")
    sampler_completed = sampler.completed_monotonic_ns
    sampler_backend = sampler.sampling_backend
    if sampler_backend != _REQUIRED_GPU_SAMPLER_BACKEND:
        raise RuntimeError(
            "bounded calibration did not use the required persistent NVML "
            "sampling backend"
        )
    finished = time.monotonic_ns()
    samples = tuple(copy.deepcopy(dict(row)) for row in sampler.samples)
    sample_windows_by_device = {
        device: [
            _sample_acquisition_window(row)
            for row in samples
            if str(row.get("device")) == device
        ]
        for device in sampler_devices
    }
    prefix_sampling_rows: list[dict[str, Any]] = []
    for phase_name, phase in phase_rows.items():
        task_intervals = phase["task_execution_attestation"][
            "task_intervals"
        ]
        results = phase["prefix_results"]
        for result, task_interval in zip(
            results,
            task_intervals,
            strict=True,
        ):
            device = str(task_interval["device"])
            prefix = result["prefix_output"]
            atomic = (
                tuple(prefix["banks"].values())
                if phase_name == "neural_inner_folds"
                else (prefix,)
            )
            for atomic_index, interval in enumerate(atomic):
                start = int(interval["measured_started_monotonic_ns"])
                finish = int(interval["measured_finished_monotonic_ns"])
                state_proof = _validated_optimizer_state_proof(interval)
                inside = sum(
                    start <= acquisition_start
                    and acquisition_start <= acquisition_finish
                    and acquisition_finish <= finish
                    for acquisition_start, acquisition_finish in (
                        sample_windows_by_device[device]
                    )
                )
                prefix_sampling_rows.append(
                    {
                        "phase": phase_name,
                        "canonical_task_index": int(
                            result["canonical_index"]
                        ),
                        "atomic_prefix_index": atomic_index,
                        "device": device,
                        "measured_started_monotonic_ns": start,
                        "measured_finished_monotonic_ns": finish,
                        "host_gpu_acquisition_windows_wholly_inside_interval": (
                            inside
                        ),
                        "optimizer_state_proof": state_proof,
                        "accepted": bool(
                            inside >= 1
                            and state_proof["accepted"] is True
                        ),
                    }
                )
    sampling_bounds_measured = bool(prefix_sampling_rows) and all(
        row["accepted"] for row in prefix_sampling_rows
    )
    if not sampling_bounds_measured:
        raise RuntimeError(
            "host GPU sampler omitted an individual measured prefix interval"
        )
    gpu = _gpu_summary(
        samples,
        phase_rows=phase_rows,
        post_warmup_sampling_rows=prefix_sampling_rows,
        sampler_completed_monotonic_ns=sampler_completed,
        run_finished_monotonic_ns=finished,
        devices=sampler_devices,
        maximum_fraction=float(args.gpu_max_allocation_fraction),
        minimum_headroom_bytes=int(args.gpu_minimum_headroom_bytes),
    )
    body = {
        "schema_version": R14_KERNEL_RUN_SCHEMA,
        "execution_devices": list(execution_devices),
        "sampler_devices": list(sampler_devices),
        "slots_per_device": int(slots_per_device),
        "serial_baseline": bool(baseline),
        "started_monotonic_ns": started,
        "finished_monotonic_ns": finished,
        "gpu_sampler_completed_monotonic_ns": sampler_completed,
        "gpu_sampler_backend": sampler_backend,
        "gpu_sampler_interval_seconds": float(
            args.gpu_sample_interval_seconds
        ),
        "wall_seconds": (finished - started) / 1e9,
        "phase_order": list(_PHASES),
        "htr_effect_input_preparation": copy.deepcopy(
            dict(active_effect_fixture.preparation)
        ),
        "neural_final_input_preparation": copy.deepcopy(
            dict(active_final_fixture.preparation)
        ),
        "phases": phase_rows,
        "gpu_samples": list(samples),
        "gpu_summary": gpu,
        "gpu_sampler_sampled_inside_every_individual_measured_prefix": (
            sampling_bounds_measured
        ),
        "individual_prefix_gpu_sample_coverage": prefix_sampling_rows,
        "pre_task_external_gpu_memory_baseline_sampled": bool(
            gpu["pre_task_external_baseline_sampled_before_task_launch"]
        ),
        "post_warmup_optimizer_state_persistence_temporally_sampled": bool(
            sampling_bounds_measured
            and gpu[
                "post_warmup_optimizer_state_persistent_during_sampled_intervals"
            ]
        ),
        "status": "completed",
    }
    telemetry_path = run_root / "run_telemetry.json"
    telemetry = _write_self_hashed_json(telemetry_path, body)
    digest, size = _sha256_file(telemetry_path)
    record = {
        "root": str(run_root),
        "status": "completed",
        "wall_seconds": body["wall_seconds"],
        "htr_effect_input_preparation": copy.deepcopy(
            dict(active_effect_fixture.preparation)
        ),
        "neural_final_input_preparation": copy.deepcopy(
            dict(active_final_fixture.preparation)
        ),
        "phases": phase_rows,
        "gpu_summary": gpu,
        "telemetry": {
            "path": str(telemetry_path),
            "sha256": digest,
            "size_bytes": size,
            "content_sha256": telemetry["content_sha256"],
        },
    }
    return record, active_effect_fixture, active_final_fixture


def _compare_float_values(
    left: Any,
    right: Any,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    label: str,
    summary: dict[str, Any],
) -> None:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype.kind != right_array.dtype.kind:
        raise ValueError(f"{label} changed prefix shape or dtype family")
    if left_array.dtype.kind not in {"f", "c"}:
        if not np.array_equal(left_array, right_array):
            raise ValueError(f"{label} changed discrete prefix state")
        summary["discrete_values_compared"] += int(left_array.size)
        return
    left_float = np.asarray(left_array, dtype=np.float64)
    right_float = np.asarray(right_array, dtype=np.float64)
    if (
        not np.array_equal(np.isnan(left_float), np.isnan(right_float))
        or not np.array_equal(np.isposinf(left_float), np.isposinf(right_float))
        or not np.array_equal(np.isneginf(left_float), np.isneginf(right_float))
    ):
        raise ValueError(f"{label} changed finite masks")
    finite = np.isfinite(left_float)
    if not bool(np.all(finite)) or not bool(np.isfinite(right_float).all()):
        raise ValueError(f"{label} contains non-finite prefix state")
    if not np.allclose(
        left_float[finite],
        right_float[finite],
        rtol=float(relative_tolerance),
        atol=float(absolute_tolerance),
    ):
        raise ValueError(f"{label} exceeds prefix tolerance")
    if np.any(finite):
        differences = np.abs(left_float[finite] - right_float[finite])
        summary["maximum_absolute_difference"] = max(
            summary["maximum_absolute_difference"],
            float(np.max(differences)),
        )
    summary["floating_values_compared"] += int(np.count_nonzero(finite))


def _validated_parameter_bundle(
    run: Mapping[str, Any],
    prefix: Mapping[str, Any],
    *,
    label: str,
) -> tuple[np.memmap, tuple[np.ndarray, ...], tuple[str, ...]]:
    descriptor = prefix.get("terminal_parameter_bundle")
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "schema_version",
        "relative_path",
        "sha256",
        "size_bytes",
        "parameter_count",
        "inventory",
        "o_exclusive_private_write",
        "fsync_before_atomic_publish",
    }:
        raise ValueError(f"{label} parameter bundle descriptor is not closed")
    if (
        descriptor["schema_version"]
        != "packed_raw_optimizer_parameter_bundle_v1"
        or descriptor["o_exclusive_private_write"] is not True
        or descriptor["fsync_before_atomic_publish"] is not True
    ):
        raise ValueError(f"{label} parameter bundle publication changed")
    relative = descriptor["relative_path"]
    if not isinstance(relative, str):
        raise ValueError(f"{label} parameter bundle path changed type")
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.as_posix() != relative
    ):
        raise ValueError(f"{label} parameter bundle escaped its run root")
    root_value = run.get("root")
    if not isinstance(root_value, str):
        raise ValueError(f"{label} run root is missing")
    root = Path(root_value)
    if (
        not root.is_absolute()
        or root.is_symlink()
        or root.resolve(strict=True) != root
    ):
        raise ValueError(f"{label} run root is not canonical")
    path = root / relative_path
    if (
        path.is_symlink()
        or path.resolve(strict=True) != path
        or not path.is_relative_to(root)
    ):
        raise ValueError(f"{label} parameter bundle path is not canonical")
    bundle_sha256, bundle_size = _sha256_file(path)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(descriptor["sha256"])) is None
        or bundle_sha256 != descriptor["sha256"]
        or isinstance(descriptor["size_bytes"], bool)
        or not isinstance(descriptor["size_bytes"], int)
        or bundle_size != descriptor["size_bytes"]
        or stat.S_IMODE(os.lstat(path).st_mode) != 0o444
    ):
        raise ValueError(f"{label} parameter bundle failed authentication")

    inventory = descriptor["inventory"]
    count = descriptor["parameter_count"]
    shapes = prefix.get("terminal_parameter_shapes")
    dtypes = prefix.get("terminal_parameter_dtypes")
    element_counts = prefix.get("terminal_parameter_element_counts")
    sha256s = prefix.get("terminal_parameter_sha256s")
    finite = prefix.get("terminal_parameter_all_finite")
    group_indices = prefix.get("terminal_parameter_group_indices")
    within_group = prefix.get(
        "terminal_parameter_indices_within_group"
    )
    sequences = (
        inventory,
        shapes,
        dtypes,
        element_counts,
        sha256s,
        finite,
        group_indices,
        within_group,
    )
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or prefix.get("terminal_parameter_count") != count
        or descriptor["parameter_count"] != count
        or any(
            not isinstance(value, (list, tuple)) or len(value) != count
            for value in sequences
        )
        or prefix.get("terminal_all_parameters_finite") is not True
    ):
        raise ValueError(f"{label} parameter bundle inventory changed")

    raw = np.memmap(path, mode="r", dtype=np.uint8)
    arrays: list[np.ndarray] = []
    observed_hashes: list[str] = []
    expected_offset = 0
    try:
        for index in range(count):
            item = inventory[index]
            if not isinstance(item, Mapping) or set(item) != {
                "parameter_index",
                "group_index",
                "parameter_index_within_group",
                "offset_bytes",
                "nbytes",
            }:
                raise ValueError(
                    f"{label} parameter bundle inventory is not closed"
                )
            shape = shapes[index]
            dtype_value = dtypes[index]
            if (
                not isinstance(shape, (list, tuple))
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in shape
                )
                or not isinstance(dtype_value, str)
            ):
                raise ValueError(f"{label} parameter topology is invalid")
            dtype = np.dtype(dtype_value)
            if dtype.hasobject or dtype.kind not in {"f", "c"}:
                raise ValueError(
                    f"{label} parameter dtype is not finite numeric data"
                )
            expected_count = int(math.prod(shape))
            expected_nbytes = expected_count * int(dtype.itemsize)
            if (
                any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in (
                        item.get("parameter_index"),
                        item.get("group_index"),
                        item.get("parameter_index_within_group"),
                        item.get("offset_bytes"),
                        item.get("nbytes"),
                        group_indices[index],
                        within_group[index],
                        element_counts[index],
                    )
                )
                or item["parameter_index"] != index
                or item["group_index"] != group_indices[index]
                or item["parameter_index_within_group"]
                != within_group[index]
                or item["offset_bytes"] != expected_offset
                or item["nbytes"] != expected_nbytes
                or element_counts[index] != expected_count
                or finite[index] is not True
                or re.fullmatch(r"[0-9a-f]{64}", str(sha256s[index]))
                is None
            ):
                raise ValueError(
                    f"{label} parameter bundle topology changed"
                )
            finish = expected_offset + expected_nbytes
            if finish > bundle_size:
                raise ValueError(
                    f"{label} parameter bundle inventory exceeds its file"
                )
            array = np.ndarray(
                shape=tuple(int(value) for value in shape),
                dtype=dtype,
                buffer=raw,
                offset=expected_offset,
                order="C",
            )
            digest = hashlib.sha256(
                dtype.str.encode("ascii")
                + _canonical_json(list(array.shape)).encode("ascii")
            )
            bytes_view = raw[expected_offset:finish]
            for start in range(0, expected_nbytes, 8 * 1024 * 1024):
                digest.update(
                    memoryview(
                        bytes_view[
                            start : min(start + 8 * 1024 * 1024, expected_nbytes)
                        ]
                    )
                )
            observed = digest.hexdigest()
            if observed != sha256s[index]:
                raise ValueError(
                    f"{label} parameter SHA-256 does not authenticate its bytes"
                )
            flattened = array.reshape(-1)
            for start in range(0, expected_count, 1_000_000):
                if not bool(
                    np.isfinite(
                        flattened[
                            start : min(start + 1_000_000, expected_count)
                        ]
                    ).all()
                ):
                    raise ValueError(
                        f"{label} parameter bundle contains non-finite data"
                    )
            arrays.append(array)
            observed_hashes.append(observed)
            expected_offset = finish
        if expected_offset != bundle_size:
            raise ValueError(
                f"{label} parameter bundle has unauthenticated trailing bytes"
            )
    except BaseException:
        del arrays
        raw._mmap.close()
        raise
    return raw, tuple(arrays), tuple(observed_hashes)


def _compare_complete_parameter_bundles(
    left_run: Mapping[str, Any],
    right_run: Mapping[str, Any],
    left_prefix: Mapping[str, Any],
    right_prefix: Mapping[str, Any],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    label: str,
    summary: dict[str, Any],
) -> None:
    left_raw, left_arrays, left_hashes = _validated_parameter_bundle(
        left_run,
        left_prefix,
        label=f"{label}.reference",
    )
    try:
        right_raw, right_arrays, right_hashes = _validated_parameter_bundle(
            right_run,
            right_prefix,
            label=f"{label}.comparison",
        )
    except BaseException:
        del left_arrays
        left_raw._mmap.close()
        raise
    try:
        if len(left_arrays) != len(right_arrays):
            raise ValueError(f"{label} changed optimizer parameter count")
        for index, (left, right) in enumerate(
            zip(left_arrays, right_arrays, strict=True)
        ):
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(
                    f"{label}.parameter[{index}] changed exact topology"
                )
            left_flat = left.reshape(-1)
            right_flat = right.reshape(-1)
            for start in range(0, left_flat.size, 1_000_000):
                finish = min(start + 1_000_000, left_flat.size)
                left_chunk = np.asarray(
                    left_flat[start:finish],
                    dtype=np.float64,
                )
                right_chunk = np.asarray(
                    right_flat[start:finish],
                    dtype=np.float64,
                )
                if (
                    not np.isfinite(left_chunk).all()
                    or not np.isfinite(right_chunk).all()
                    or not np.allclose(
                        left_chunk,
                        right_chunk,
                        rtol=float(relative_tolerance),
                        atol=float(absolute_tolerance),
                    )
                ):
                    raise ValueError(
                        f"{label}.parameter[{index}] exceeds complete "
                        "parameter tolerance"
                    )
                if left_chunk.size:
                    summary["maximum_absolute_difference"] = max(
                        summary["maximum_absolute_difference"],
                        float(
                            np.max(np.abs(left_chunk - right_chunk))
                        ),
                    )
                    summary["floating_values_compared"] += int(
                        left_chunk.size
                    )
            summary["complete_parameter_tensors_compared"] += 1
            if left_hashes[index] == right_hashes[index]:
                summary["cross_run_parameter_sha256_matches"] += 1
            else:
                summary["cross_run_parameter_sha256_differences_tolerance_accepted"] += (
                    1
                )
    finally:
        left = right = None
        left_flat = right_flat = None
        left_chunk = right_chunk = None
        del left_arrays
        del right_arrays
        left_raw._mmap.close()
        right_raw._mmap.close()


def _compare_prefix_outputs(
    reference: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    summary = {
        "floating_values_compared": 0,
        "discrete_values_compared": 0,
        "maximum_absolute_difference": 0.0,
        "complete_parameter_tensors_compared": 0,
        "cross_run_parameter_sha256_matches": 0,
        "cross_run_parameter_sha256_differences_tolerance_accepted": 0,
        "relative_tolerance": float(relative_tolerance),
        "absolute_tolerance": float(absolute_tolerance),
    }
    for phase in _PHASES:
        left_rows = reference["phases"][phase]["prefix_results"]
        right_rows = comparison["phases"][phase]["prefix_results"]
        if len(left_rows) != len(right_rows):
            raise ValueError(f"{phase} changed canonical task count")
        for index, (left, right) in enumerate(
            zip(left_rows, right_rows, strict=True)
        ):
            if (
                left["canonical_index"] != right["canonical_index"]
                or left["canonical_identity"] != right["canonical_identity"]
                or left.get("batch_position_sha256s")
                != right.get("batch_position_sha256s")
            ):
                raise ValueError(f"{phase}[{index}] changed canonical identity")
            left_prefix = left["prefix_output"]
            right_prefix = right["prefix_output"]
            if (
                left.get("complete_plan_authenticated")
                != right.get("complete_plan_authenticated")
                or left.get("complete_input_plan_content_sha256")
                != right.get("complete_input_plan_content_sha256")
            ):
                raise ValueError(
                    f"{phase}[{index}] changed complete-plan binding"
                )
            complete_parameter_keys: set[str] = set()
            if phase in {
                "htr_nuisance",
                "htr_effect",
                "matched_pair_htr",
            }:
                _compare_complete_parameter_bundles(
                    reference,
                    comparison,
                    left_prefix,
                    right_prefix,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                    label=f"{phase}[{index}]",
                    summary=summary,
                )
                complete_parameter_keys = {
                    "terminal_parameter_bundle",
                    "terminal_parameter_sha256s",
                }

            def compare_node(a: Any, b: Any, label: str) -> None:
                if isinstance(a, Mapping) or isinstance(b, Mapping):
                    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
                        raise ValueError(f"{label} changed type")
                    ignored = {
                        "measured_started_monotonic_ns",
                        "measured_finished_monotonic_ns",
                        "ready_wait_started_monotonic_ns",
                        "ready_wait_finished_monotonic_ns",
                        "optimizer_state_verified_monotonic_ns",
                        "optimizer_state_finish_verified_monotonic_ns",
                        *complete_parameter_keys,
                    }
                    left_keys = set(a) - ignored
                    right_keys = set(b) - ignored
                    if left_keys != right_keys:
                        raise ValueError(f"{label} changed keys")
                    for key in sorted(left_keys):
                        compare_node(a[key], b[key], f"{label}.{key}")
                    return
                if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
                    _compare_float_values(
                        a,
                        b,
                        relative_tolerance=relative_tolerance,
                        absolute_tolerance=absolute_tolerance,
                        label=label,
                        summary=summary,
                    )
                    return
                if isinstance(a, float) or isinstance(b, float):
                    _compare_float_values(
                        [a],
                        [b],
                        relative_tolerance=relative_tolerance,
                        absolute_tolerance=absolute_tolerance,
                        label=label,
                        summary=summary,
                    )
                    return
                if type(a) is not type(b) or a != b:
                    raise ValueError(f"{label} changed discrete state")
                summary["discrete_values_compared"] += 1

            compare_node(left_prefix, right_prefix, f"{phase}[{index}]")
            if phase == "neural_inner_folds":
                left_banks = left.get("banks")
                right_banks = right.get("banks")
                if left_banks is None and right_banks is None:
                    continue
                if (
                    not isinstance(left_banks, Mapping)
                    or not isinstance(right_banks, Mapping)
                    or set(left_banks) != set(right_banks)
                    or set(left_banks)
                    != {"treatment", "outcome", "effect"}
                    or left.get("fold") != right.get("fold")
                    or left.get("identity_payload")
                    != right.get("identity_payload")
                ):
                    raise ValueError(
                        f"{phase}[{index}] changed candidate identity"
                    )
                for bank in ("treatment", "outcome", "effect"):
                    left_candidates = left_banks[bank].get(
                        "candidates"
                    )
                    right_candidates = right_banks[bank].get(
                        "candidates"
                    )
                    if (
                        not isinstance(left_candidates, list)
                        or not isinstance(right_candidates, list)
                        or len(left_candidates) != len(right_candidates)
                    ):
                        raise ValueError(
                            f"{phase}[{index}].{bank} changed candidates"
                        )
                    _validated_calibration_candidate_queries(
                        left_candidates
                    )
                    _validated_calibration_candidate_queries(
                        right_candidates
                    )
                    for candidate_index, (
                        left_candidate,
                        right_candidate,
                    ) in enumerate(
                        zip(
                            left_candidates,
                            right_candidates,
                            strict=True,
                        )
                    ):
                        for key in (
                            "candidate_id",
                            "bank",
                            "subfold",
                            "query_dtype",
                            "query_shape",
                            "validation_audit_only_not_used_for_gating",
                            (
                                "calibration_prefix_derived_with_"
                                "production_scoring"
                            ),
                        ):
                            if left_candidate[key] != right_candidate[key]:
                                raise ValueError(
                                    f"{phase}[{index}].{bank}["
                                    f"{candidate_index}] changed {key}"
                                )
                            summary["discrete_values_compared"] += 1
                        compare_node(
                            left_candidate["query"],
                            right_candidate["query"],
                            (
                                f"{phase}[{index}].{bank}["
                                f"{candidate_index}].query"
                            ),
                        )
                        for key in (
                            "train_standardized_score",
                            "validation_audit_standardized_score",
                            "query_drift",
                        ):
                            compare_node(
                                float(left_candidate[key]),
                                float(right_candidate[key]),
                                (
                                    f"{phase}[{index}].{bank}["
                                    f"{candidate_index}].{key}"
                                ),
                            )
    return summary


def _throughput_ratios(
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    minimum_ratio: float,
) -> dict[str, Any]:
    if len(candidates) != 2:
        raise ValueError("calibration requires exactly two candidate repetitions")
    result: dict[str, Any] = {}
    for phase in _PHASES:
        baseline_rate = float(
            baseline["phases"][phase][
                "aggregate_measured_optimizer_steps_per_second"
            ]
        )
        candidate_rates = [
            float(
                run["phases"][phase][
                    "aggregate_measured_optimizer_steps_per_second"
                ]
            )
            for run in candidates
        ]
        candidate_rate = min(candidate_rates)
        ratio = candidate_rate / baseline_rate
        result[phase] = {
            "serial_baseline_measured_steps_per_second": baseline_rate,
            "candidate_repetition_measured_steps_per_second": candidate_rates,
            "conservative_candidate_measured_steps_per_second": candidate_rate,
            "single_gpu_baseline_throughput_ratio": ratio,
            "minimum_required_ratio": float(minimum_ratio),
            "throughput_threshold_met": ratio >= float(minimum_ratio),
        }
    return result


def _required_kernel_coverage_gate(
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    htr_effect_fixture: _HTREffectFixture,
    neural_final_fixture: _NeuralFinalFixture,
) -> bool:
    runs = (baseline, *tuple(candidates))
    if len(runs) != 3:
        return False
    for run_index, run in enumerate(runs):
        phases = run.get("phases")
        if (
            not isinstance(phases, Mapping)
            or list(phases) != list(_PHASES)
        ):
            return False
        for phase in _PHASES:
            row = phases.get(phase)
            expected_parallelism = (
                1
                if run_index == 0
                else _EXPECTED_CANDIDATE_PARALLELISM[phase]
            )
            if (
                not isinstance(row, Mapping)
                or row.get("canonical_task_count")
                != _EXPECTED_TASK_COUNTS[phase]
                or row.get("configured_parallelism")
                != expected_parallelism
                or row.get("measured_optimizer_maximum_concurrency")
                != expected_parallelism
                or len(row.get("prefix_results") or ())
                != _EXPECTED_TASK_COUNTS[phase]
                or row.get(
                    "ready_barrier_enforced_after_warmup"
                )
                is not True
                or row.get(
                    "optimizer_state_persistence_observed_for_every_"
                    "atomic_prefix"
                )
                is not True
            ):
                return False
        if (
            run.get("htr_effect_input_preparation")
            != htr_effect_fixture.preparation
            or run.get("neural_final_input_preparation")
            != neural_final_fixture.preparation
        ):
            return False
    return bool(
        len(htr_effect_fixture.tasks)
        == _EXPECTED_TASK_COUNTS["htr_effect"]
        and len(neural_final_fixture.tasks)
        == _EXPECTED_TASK_COUNTS["neural_final_banks"]
        and htr_effect_fixture.preparation.get(
            "strict_nuisance_to_effect_barrier_observed"
        )
        is True
        and neural_final_fixture.preparation.get(
            "strict_inner_to_final_barrier_observed"
        )
        is True
    )


def _complete_input_plan_bindings_gate(
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    inventory: Mapping[str, Any],
    inventory_descriptor: Mapping[str, Any],
    expected_plan_root: Path,
    neural_final_fixture: _NeuralFinalFixture,
) -> bool:
    try:
        reopened_inventory = _reopen_published_json_descriptor(
            inventory_descriptor,
            expected_root=expected_plan_root,
            expected_producer="all_calibration_kernels",
        )
        inventory_keys = {
            "schema_version",
            "stage1_plan_scientific_content_sha256",
            "htr",
            "matched_pair",
            "neural_inner",
            "neural_final_prefix_conditioned",
            "producer_plan_count",
            "canonical_phase_plan_bindings",
            "all_complete_inputs_bound_to_authenticated_plans",
            "all_plan_json_files_self_hashed_and_reopened",
            "semantic_truncation_applied",
            "content_sha256",
        }
        if (
            reopened_inventory != dict(inventory)
            or set(reopened_inventory) != inventory_keys
            or reopened_inventory.get("schema_version")
            != "production_r14_complete_input_plan_inventory_v1"
            or reopened_inventory.get("producer_plan_count") != 6
            or reopened_inventory.get(
                "all_complete_inputs_bound_to_authenticated_plans"
            )
            is not True
            or reopened_inventory.get(
                "all_plan_json_files_self_hashed_and_reopened"
            )
            is not True
            or reopened_inventory.get("semantic_truncation_applied") is not False
        ):
            return False

        matched_descriptor = reopened_inventory.get("matched_pair")
        neural_descriptor = reopened_inventory.get("neural_inner")
        final_descriptors = reopened_inventory.get(
            "neural_final_prefix_conditioned"
        )
        if (
            not isinstance(matched_descriptor, Mapping)
            or not isinstance(neural_descriptor, Mapping)
            or not isinstance(final_descriptors, list)
            or len(final_descriptors)
            != _EXPECTED_TASK_COUNTS["neural_final_banks"]
        ):
            return False
        matched_plan = _reopen_published_json_descriptor(
            matched_descriptor,
            expected_root=expected_plan_root,
            expected_producer="matched_pair_htr",
        )
        neural_plan = _reopen_published_json_descriptor(
            neural_descriptor,
            expected_root=expected_plan_root,
            expected_producer="neural_queries",
        )
        expected_banks = ("treatment", "outcome", "effect")
        final_plans = tuple(
            _reopen_published_json_descriptor(
                descriptor,
                expected_root=expected_plan_root,
                expected_producer=f"neural_final_bank:{bank}",
            )
            for descriptor, bank in zip(
                final_descriptors,
                expected_banks,
                strict=True,
            )
        )
        stage1_sha256 = reopened_inventory.get(
            "stage1_plan_scientific_content_sha256"
        )
        if (
            re.fullmatch(r"[0-9a-f]{64}", str(stage1_sha256)) is None
            or matched_plan.get("schema_version")
            != "production_r14_matched_complete_input_plan_v1"
            or neural_plan.get("schema_version")
            != "production_r14_neural_complete_input_plan_v1"
            or matched_plan.get(
                "stage1_plan_scientific_content_sha256"
            )
            != stage1_sha256
            or neural_plan.get(
                "stage1_plan_scientific_content_sha256"
            )
            != stage1_sha256
            or matched_plan.get("canonical_task_count")
            != _EXPECTED_TASK_COUNTS["matched_pair_htr"]
            or len(matched_plan.get("canonical_tasks") or ())
            != _EXPECTED_TASK_COUNTS["matched_pair_htr"]
            or matched_plan.get(
                "all_production_pair_rows_reconstructed"
            )
            is not True
            or matched_plan.get("all_initialization_texts_covered") is not True
            or matched_plan.get(
                "all_production_word_chunks_reconstructed_and_hashed"
            )
            is not True
            or matched_plan.get("worker_consumed_materialized_chunk_plan")
            is not False
            or matched_plan.get(
                "worker_recomputes_exact_authenticated_production_chunks"
            )
            is not True
            or matched_plan.get("semantic_truncation_applied") is not False
        ):
            return False

        htr_inventory = reopened_inventory.get("htr")
        if not isinstance(htr_inventory, Mapping):
            return False
        from oci.inference import role_neutral_htr_group_execution as htr

        htr_root = Path(str(htr_inventory["root"]))
        htr_manifest_path = Path(str(htr_inventory["manifest_path"]))
        expected_htr_root = (
            expected_plan_root.resolve(strict=True).parent
            / "htr_complete_authenticated_plan"
        )
        if (
            not htr_root.is_absolute()
            or htr_root.resolve(strict=True) != htr_root
            or htr_root != expected_htr_root
            or htr_manifest_path != htr_root / "manifest.json"
            or htr_manifest_path.resolve(strict=True) != htr_manifest_path
        ):
            return False
        htr_manifest_sha256, htr_manifest_size = _sha256_file(
            htr_manifest_path
        )
        if (
            htr_manifest_sha256 != htr_inventory.get("manifest_sha256")
            or htr_manifest_size != htr_inventory.get("manifest_size_bytes")
        ):
            return False
        htr_descriptor = htr._MaterializedReusableTextPlan(
            root=str(htr_root),
            manifest_sha256=str(htr_manifest_sha256),
            manifest_size_bytes=int(htr_manifest_size),
            content_sha256=str(htr_inventory["content_sha256"]),
        )
        texts, row_ids, coverage, reusable = (
            htr._load_materialized_reusable_text_plan(htr_descriptor)
        )
        if (
            len(texts) != htr_inventory.get("fresh_reopen_row_count")
            or _sha256_json(list(row_ids))
            != htr_inventory.get("fresh_reopen_ordered_row_ids_sha256")
            or _sha256_json(list(texts))
            != htr_inventory.get("fresh_reopen_ordered_texts_sha256")
            or _sha256_json(
                [list(value) for value in coverage.chunks_by_note]
            )
            != htr_inventory.get("fresh_reopen_chunk_plan_sha256")
            or reusable.content_sha256
            != htr_inventory.get(
                "fresh_reopen_reusable_plan_content_sha256"
            )
            or htr_inventory.get(
                "worker_consumed_materialized_text_chunk_token_plan"
            )
            is not True
            or htr_inventory.get("fresh_reopen_completed") is not True
            or matched_plan.get(
                "htr_owner_materialized_plan_content_sha256"
            )
            != htr_inventory.get("content_sha256")
        ):
            return False

        rows = neural_plan.get("ordered_row_ids")
        neural_tasks = neural_plan.get("canonical_inner_tasks")
        row_inventory = neural_plan.get("complete_row_chunk_inventory")
        shared = neural_plan.get("shared_cache_reference")
        if (
            not isinstance(rows, list)
            or not rows
            or not isinstance(neural_tasks, list)
            or len(neural_tasks)
            != _EXPECTED_TASK_COUNTS["neural_inner_folds"]
            or neural_plan.get("canonical_inner_task_count")
            != len(neural_tasks)
            or not isinstance(row_inventory, list)
            or len(row_inventory) != len(rows)
            or not isinstance(shared, Mapping)
            or neural_plan.get("ordered_row_ids_sha256")
            != _sha256_json(rows)
            or neural_plan.get("allowed_row_order_sha256")
            != _sha256_json(rows)
            or neural_plan.get("complete_row_chunk_inventory_sha256")
            != _sha256_json(row_inventory)
            or neural_plan.get("shared_cache_reference_content_sha256")
            != shared.get("content_sha256")
            or shared.get("content_sha256")
            != _sha256_json(
                {
                    key: value
                    for key, value in shared.items()
                    if key != "content_sha256"
                }
            )
            or neural_plan.get("logical_identity_sha256")
            != _sha256_json(shared.get("logical_identity"))
            or neural_plan.get("cache_file_inventory")
            != shared.get("cache_files")
            or neural_plan.get("fresh_process_open_bound_completed")
            is not True
            or neural_plan.get("all_owner_rows_and_chunks_reopened")
            is not True
            or neural_plan.get(
                "materialized_matrix_or_text_payload_copied_into_plan"
            )
            is not False
            or neural_plan.get("semantic_truncation_applied") is not False
        ):
            return False
        global_arrays: dict[str, np.ndarray] = {}
        for label in (
            "global_treatment",
            "global_outcome",
            "global_fit_e",
            "global_fit_m",
        ):
            payload = neural_plan.get(label)
            if not isinstance(payload, Mapping) or set(payload) != {
                "dtype",
                "shape",
                "values",
                "content_sha256",
            }:
                return False
            values = np.ascontiguousarray(
                np.asarray(payload["values"], dtype=np.dtype(payload["dtype"]))
            )
            if (
                payload["shape"]
                != [int(value) for value in values.shape]
                or values.shape != (len(rows),)
                or not np.isfinite(values).all()
                or payload["content_sha256"] != _array_sha256(values)
            ):
                return False
            global_arrays[label] = values
        if (
            neural_plan.get("treatment_sha256")
            != _array_sha256(global_arrays["global_treatment"])
            or neural_plan.get("outcome_sha256")
            != _array_sha256(global_arrays["global_outcome"])
        ):
            return False
        nuisance_views_sha256 = _sha256_json(
            neural_plan.get("nuisance_views")
        )
        nuisance_config_sha256 = _sha256_json(
            neural_plan.get("nuisance_stack_config")
        )
        for task in neural_tasks:
            if (
                not isinstance(task, Mapping)
                or task.get("parent_input_binding_sha256")
                != neural_plan.get("parent_input_binding_sha256")
                or task.get("outcome_binary")
                != neural_plan.get("outcome_binary")
                or task.get("treatment_sha256")
                != neural_plan.get("treatment_sha256")
                or task.get("outcome_sha256")
                != neural_plan.get("outcome_sha256")
                or task.get("nuisance_views_sha256")
                != nuisance_views_sha256
                or task.get("nuisance_folds")
                != neural_plan.get("nuisance_folds")
                or task.get("nuisance_stack_config_sha256")
                != nuisance_config_sha256
                or task.get("query_config")
                != neural_plan.get("query_config")
            ):
                return False
        from oci.inference.production_stage1_preflight_scope_inputs import (
            _authenticated_cache_identity,
            _load_shared_cache,
        )
        from oci.inference.neural_query_discovery_runtime import _stable_hash

        reopened_cache = _load_shared_cache(shared)
        if _authenticated_cache_identity(reopened_cache) != shared.get(
            "logical_identity"
        ):
            return False
        runtime_texts_sha256 = neural_plan.get(
            "runtime_ordered_texts_sha256"
        )
        if (
            re.fullmatch(r"[0-9a-f]{64}", str(runtime_texts_sha256)) is None
            or _stable_hash(
                {
                    "scope": (
                        "production_in_memory_no_executable_checkpoint_io"
                    ),
                    "runtime": (
                        "neural_query_in_memory_discovery_runtime_v2"
                    ),
                    "row_ids": rows,
                    "texts_sha256": runtime_texts_sha256,
                    "treatment": global_arrays[
                        "global_treatment"
                    ].tolist(),
                    "outcome": global_arrays["global_outcome"].tolist(),
                    "fit_e": global_arrays["global_fit_e"].tolist(),
                    "fit_m": global_arrays["global_fit_m"].tolist(),
                    "outcome_binary": neural_plan.get("outcome_binary"),
                    "nuisance_views_sha256": _stable_hash(
                        neural_plan.get("nuisance_views")
                    ),
                    "nuisance_stack_scientific": neural_plan.get(
                        "nuisance_stack_config"
                    ),
                    "query_config": neural_plan.get("query_config"),
                }
            )
            != neural_plan.get("parent_input_binding_sha256")
        ):
            return False
        replayed_row_inventory: list[dict[str, Any]] = []
        for row_id, expected_row in zip(
            rows,
            row_inventory,
            strict=True,
        ):
            if (
                isinstance(row_id, bool)
                or not isinstance(row_id, int)
                or not isinstance(expected_row, Mapping)
                or expected_row.get("row_id") != row_id
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(expected_row.get("text_sha256")),
                )
                is None
            ):
                return False
            start = int(reopened_cache._offsets[row_id])
            stop = int(reopened_cache._offsets[row_id + 1])
            matrix = np.ascontiguousarray(
                np.asarray(
                    reopened_cache._embeddings[start:stop],
                    dtype=np.float32,
                )
            )
            chunks = tuple(reopened_cache._cached_chunks(row_id))
            if (
                matrix.ndim != 2
                or matrix.shape[0] != len(chunks)
                or not np.isfinite(matrix).all()
            ):
                return False
            replayed_row_inventory.append(
                {
                    "row_id": row_id,
                    "text_sha256": expected_row["text_sha256"],
                    "matrix_dtype": matrix.dtype.str,
                    "matrix_shape": [
                        int(value) for value in matrix.shape
                    ],
                    "matrix_sha256": _array_sha256(matrix),
                    "chunk_text_count": len(chunks),
                    "chunk_texts_sha256": _sha256_json(list(chunks)),
                }
            )
        if (
            replayed_row_inventory != row_inventory
            or _sha256_json(replayed_row_inventory)
            != neural_plan.get("complete_row_chunk_inventory_sha256")
        ):
            return False

        fixture = neural_final_fixture.preparation
        fixture_body = {
            key: value
            for key, value in fixture.items()
            if key != "content_sha256"
        }
        if (
            fixture.get("content_sha256") != _sha256_json(fixture_body)
            or fixture.get("complete_input_plans") != final_descriptors
            or fixture.get(
                "base_neural_complete_input_plan_content_sha256"
            )
            != neural_plan.get("content_sha256")
            or fixture.get("task_count")
            != _EXPECTED_TASK_COUNTS["neural_final_banks"]
            or fixture.get("production_final_task_builder_used") is not True
            or fixture.get("strict_inner_to_final_barrier_observed") is not True
            or fixture.get(
                "same_typed_tasks_reused_by_both_candidates"
            )
            is not True
            or fixture.get("prefix_conditioned_not_full_inner_fit") is not True
        ):
            return False
        source_inner_sha256 = _sha256_json(
            baseline["phases"]["neural_inner_folds"]["prefix_results"]
        )
        fixture_identities = fixture.get("task_identities")
        if (
            fixture_identities
            != [
                _neural_final_task_identity(task)
                for task in neural_final_fixture.tasks
            ]
        ):
            return False
        inner_prefix_results = baseline["phases"][
            "neural_inner_folds"
        ]["prefix_results"]
        for index, (bank, plan) in enumerate(
            zip(expected_banks, final_plans, strict=True)
        ):
            candidates_for_bank = [
                candidate
                for inner_result in inner_prefix_results
                for candidate in inner_result["banks"][bank]["candidates"]
            ]
            candidate_queries = _validated_calibration_candidate_queries(
                candidates_for_bank
            )
            identity = fixture_identities[index]
            if (
                plan.get("schema_version")
                != (
                    "production_r14_neural_final_prefix_conditioned_"
                    "input_plan_v1"
                )
                or plan.get("bank") != bank
                or plan.get("bank_index") != index
                or plan.get(
                    "base_neural_complete_input_plan_content_sha256"
                )
                != neural_plan.get("content_sha256")
                or plan.get("source_inner_prefix_results_sha256")
                != source_inner_sha256
                or plan.get("row_ids_sha256")
                != neural_plan.get("ordered_row_ids_sha256")
                or plan.get("texts_sha256")
                != neural_plan.get("ordered_texts_sha256")
                or plan.get("treatment_sha256")
                != neural_plan.get("treatment_sha256")
                or plan.get("outcome_sha256")
                != neural_plan.get("outcome_sha256")
                or plan.get("outcome_binary")
                != neural_plan.get("outcome_binary")
                or plan.get("fit_e_sha256")
                != _array_sha256(global_arrays["global_fit_e"])
                or plan.get("fit_m_sha256")
                != _array_sha256(global_arrays["global_fit_m"])
                or plan.get("query_config")
                != neural_plan.get("query_config")
                or plan.get("candidate_count")
                != len(candidates_for_bank)
                or plan.get("candidate_payload_sha256")
                != _sha256_json(candidates_for_bank)
                or plan.get("candidate_ids")
                != [
                    str(value["candidate_id"])
                    for value in candidates_for_bank
                ]
                or plan.get("candidate_query_shape")
                != [int(value) for value in candidate_queries.shape]
                or plan.get("candidate_query_dtype")
                != candidate_queries.dtype.str
                or plan.get("candidate_query_sha256")
                != _array_sha256(candidate_queries)
                or not isinstance(identity, Mapping)
                or identity.get("outcome_binary")
                != plan.get("outcome_binary")
                or identity.get("candidate_query_sha256")
                != plan.get("candidate_query_sha256")
                or identity.get("complete_input_plan_content_sha256")
                != plan.get("content_sha256")
                or plan.get(
                    "production_consensus_builder_consumes_real_scores"
                )
                is not True
                or plan.get("serial_prefix_conditioned_calibration_task")
                is not True
                or plan.get(
                    "full_inner_fit_or_scientific_final_output_claimed"
                )
                is not False
                or plan.get("semantic_truncation_applied") is not False
            ):
                return False

        bindings = reopened_inventory.get("canonical_phase_plan_bindings")
        expected_bindings = {
            "htr_nuisance": htr_inventory.get("content_sha256"),
            "htr_effect": htr_inventory.get("content_sha256"),
            "matched_pair_htr": matched_plan.get("content_sha256"),
            "neural_inner_folds": neural_plan.get("content_sha256"),
            "neural_final_banks": [
                value.get("content_sha256") for value in final_plans
            ],
        }
        if bindings != expected_bindings:
            return False
        for run in (baseline, *tuple(candidates)):
            phases = run.get("phases")
            if not isinstance(phases, Mapping) or set(phases) != set(_PHASES):
                return False
            for phase in _PHASES:
                phase_row = phases[phase]
                rows = phase_row.get("prefix_results")
                if (
                    not isinstance(rows, list)
                    or len(rows) != _EXPECTED_TASK_COUNTS[phase]
                ):
                    return False
                for index, row in enumerate(rows):
                    identity = row.get("canonical_identity")
                    expected = (
                        expected_bindings[phase][index]
                        if phase == "neural_final_banks"
                        else expected_bindings[phase]
                    )
                    if (
                        row.get("phase") != phase
                        or row.get("canonical_index") != index
                        or not isinstance(identity, Mapping)
                        or identity.get(
                            "complete_input_plan_content_sha256"
                        )
                        != expected
                        or row.get("complete_plan_authenticated") is not True
                    ):
                        return False
                    result_binding = row.get(
                        "complete_input_plan_content_sha256"
                    )
                    if phase in {
                        "matched_pair_htr",
                        "neural_inner_folds",
                        "neural_final_banks",
                    }:
                        if result_binding != expected:
                            return False
                    elif result_binding is not None:
                        return False
        return True
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False


def _neural_candidate_provenance_gate(
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    neural_final_fixture: _NeuralFinalFixture,
) -> bool:
    if (
        neural_final_fixture.preparation.get(
            "source"
        )
        != "serial_bounded_inner_prefix_candidates"
        or neural_final_fixture.preparation.get(
            "same_typed_tasks_reused_by_both_candidates"
        )
        is not True
    ):
        return False
    try:
        for run in (baseline, *tuple(candidates)):
            inner_rows = run["phases"]["neural_inner_folds"][
                "prefix_results"
            ]
            if len(inner_rows) != _EXPECTED_TASK_COUNTS[
                "neural_inner_folds"
            ]:
                return False
            for row in inner_rows:
                proofs = row.get("candidate_scoring_proofs")
                banks = row.get("banks")
                if (
                    not isinstance(proofs, Mapping)
                    or not isinstance(banks, Mapping)
                    or set(proofs) != {"treatment", "outcome", "effect"}
                    or set(banks) != set(proofs)
                ):
                    return False
                for bank, proof in proofs.items():
                    proof_body = {
                        key: value
                        for key, value in proof.items()
                        if key != "content_sha256"
                    }
                    candidates_for_bank = banks[bank].get("candidates")
                    if (
                        proof.get("content_sha256")
                        != _sha256_json(proof_body)
                        or proof.get(
                            "natural_production_post_loop_outputs_"
                            "are_authoritative"
                        )
                        is not True
                        or proof.get(
                            "independent_train_activation_score_and_"
                            "drift_replay_accepted"
                        )
                        is not True
                        or not isinstance(candidates_for_bank, list)
                    ):
                        return False
                    _validated_calibration_candidate_queries(
                        candidates_for_bank
                    )
        for prepared in neural_final_fixture.tasks:
            _neural_final_task_identity(prepared)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _calibration_acceptance(
    *,
    equality_accepted: bool,
    memory_accepted: bool,
    all_phase_throughput_thresholds_met: bool,
    required_kernel_coverage_accepted: bool,
    complete_input_plan_bindings_accepted: bool,
    neural_candidate_provenance_accepted: bool,
) -> dict[str, Any]:
    gates = {
        "equality_accepted": bool(equality_accepted),
        "memory_accepted": bool(memory_accepted),
        "all_phase_throughput_thresholds_met": bool(
            all_phase_throughput_thresholds_met
        ),
        "required_kernel_coverage_accepted": bool(
            required_kernel_coverage_accepted
        ),
        "complete_input_plan_bindings_accepted": bool(
            complete_input_plan_bindings_accepted
        ),
        "neural_candidate_provenance_accepted": bool(
            neural_candidate_provenance_accepted
        ),
    }
    accepted = all(gates.values())
    return {
        **gates,
        "policy": "all_required_gates_conjunction_v1",
        "calibration_valid": accepted,
        "multi_gpu_step_throughput_acceleration_claimed": accepted,
        "deployment_recommendation": (
            "proceed_to_first_complete_owner_validation_gate"
            if accepted
            else "do_not_adopt_kernel_calibration"
        ),
        "process_exit_code": 0 if accepted else 2,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    devices = _validate_main_args(args)
    from oci.inference.prepared_stage1_context import (
        load_prepared_stage1_context,
    )
    from oci.inference.production_role_neutral_producer_factories import (
        _group_inputs,
        _scientific_bindings,
    )
    from oci.inference.production_source_snapshot import (
        validate_production_source_snapshot,
    )

    snapshot = validate_production_source_snapshot(
        args.source_snapshot_root
    )
    expected_runner = (
        snapshot.root / "scripts" / Path(__file__).name
    )
    if Path(__file__).resolve(strict=True) != expected_runner:
        raise RuntimeError(
            "calibration must execute its runner from the authenticated "
            "immutable source snapshot"
        )
    snapshot_manifest_sha256, snapshot_manifest_size = _sha256_file(
        snapshot.manifest_path
    )
    artifact = load_prepared_stage1_context(args.prepared_context_manifest)
    prepared, factories = artifact.reconstruct(
        slot_cpu_budget=int(args.cpu_budget),
        ordinary_full_byte_cache_fallback=(
            args.ordinary_full_byte_cache_fallback
        ),
        absent_htr_model_path_rebinding=args.htr_model_path,
    )
    owner, members = _canonical_exact_inner_group(prepared.stage1_scope_plan)
    profiles = artifact.execution_locators["architecture_profiles"]
    bindings = _scientific_bindings(prepared=prepared, profiles=profiles)
    source_tolerances = _authenticated_prefix_tolerances(bindings)
    requested_tolerance = {
        "relative_tolerance": float(args.prefix_relative_tolerance),
        "absolute_tolerance": float(args.prefix_absolute_tolerance),
    }
    if any(
        row != requested_tolerance for row in source_tolerances.values()
    ):
        raise ValueError(
            "calibration prefix tolerances differ from one or more "
            "authenticated production replay tolerances"
        )
    inputs = _group_inputs(
        prepared,
        SimpleNamespace(
            plan=prepared.stage1_scope_plan,
            physical_owner=owner,
        ),
    )
    baseline_htr, baseline_neural = _controls(
        args,
        devices=(devices[0],),
        slots_per_device=1,
        baseline=True,
    )
    candidate_htr, candidate_neural = _controls(
        args,
        devices=devices,
        slots_per_device=int(args.candidate_slot_cap_per_device),
        baseline=False,
    )
    args.output_root.mkdir(parents=False, exist_ok=False)
    setup_root = args.output_root / "authenticated_preparation"
    setup_root.mkdir()
    complete_plan_root = setup_root / "complete_input_plans"
    complete_plan_root.mkdir()
    htr_tasks, htr_plan = _capture_htr_tasks(
        output_root=setup_root,
        owner=owner,
        inputs=inputs,
        config=bindings.htr,
        controls=candidate_htr,
        model_path=prepared.htr_model_path,
    )
    raw_matched_tasks, bow_preparation = _prepare_bow_and_capture_matched(
        setup_root=setup_root,
        prepared=prepared,
        factories=factories,
        owner=owner,
        members=members,
        devices=devices,
        htr_controls=candidate_htr,
        neural_controls=candidate_neural,
        cpu_budget=int(args.cpu_budget),
    )
    inner_tasks, discovery_kwargs = _capture_neural_inner(
        setup_root=setup_root,
        prepared=prepared,
        factories=factories,
        owner=owner,
        members=members,
        devices=devices,
        htr_controls=candidate_htr,
        neural_controls=candidate_neural,
        cpu_budget=int(args.cpu_budget),
    )
    pending_inner, neural_nuisance_preparation = (
        _prepare_neural_inner_nuisance(
            inner_tasks,
            cpu_budget=int(args.cpu_budget),
        )
    )
    matched_tasks, matched_complete_plan = (
        _materialize_matched_complete_input_plan(
            root=complete_plan_root / "matched_pair",
            tasks=raw_matched_tasks,
            stage1_plan_scientific_content_sha256=(
                prepared.stage1_scope_plan.scientific_content_sha256
            ),
            htr_complete_plan_content_sha256=str(
                htr_plan["content_sha256"]
            ),
            htr_model_tree_sha256=str(prepared.htr_model_sha256),
        )
    )
    prepared_inner, neural_complete_plan = (
        _materialize_neural_complete_input_plan(
            root=complete_plan_root / "neural",
            prepared_tasks=pending_inner,
            stage1_plan_scientific_content_sha256=(
                prepared.stage1_scope_plan.scientific_content_sha256
            ),
            discovery_kwargs=discovery_kwargs,
        )
    )
    prepared_tasks = {
        "htr_nuisance": htr_tasks,
        "matched_pair_htr": matched_tasks,
        "neural_inner_folds": prepared_inner,
    }
    baseline_tasks = {
        **prepared_tasks,
        "htr_nuisance": tuple(
            dataclasses.replace(
                task,
                operational_controls=baseline_htr,
            )
            for task in htr_tasks
        ),
    }
    (
        baseline,
        htr_effect_fixture,
        neural_final_fixture,
    ) = _run_prefix_suite(
        run_root=args.output_root / "serial_baseline",
        prepared_tasks=baseline_tasks,
        discovery_kwargs=discovery_kwargs,
        execution_devices=(devices[0],),
        sampler_devices=devices,
        slots_per_device=1,
        args=args,
        baseline=True,
        owner_scope_seed=int(owner.scope_seed),
        htr_effect_fixture=None,
        neural_final_fixture=None,
        neural_final_plan_root=(
            complete_plan_root / "neural_final_prefix_conditioned"
        ),
    )
    (
        complete_input_plan_inventory,
        complete_input_plan_inventory_descriptor,
    ) = _materialize_complete_input_plan_inventory(
        path=complete_plan_root / "complete_input_plan_inventory.json",
        htr_plan_root=setup_root / "htr_complete_authenticated_plan",
        htr_plan_attestation=htr_plan,
        matched_plan=matched_complete_plan,
        neural_plan=neural_complete_plan,
        neural_final_fixture=neural_final_fixture,
        stage1_plan_scientific_content_sha256=(
            prepared.stage1_scope_plan.scientific_content_sha256
        ),
    )
    candidates: list[dict[str, Any]] = []
    for index in range(2):
        (
            candidate,
            reused_effect_fixture,
            reused_final_fixture,
        ) = _run_prefix_suite(
            run_root=(
                args.output_root
                / "concurrent_candidates"
                / f"repetition_{index:03d}"
            ),
            prepared_tasks=prepared_tasks,
            discovery_kwargs=discovery_kwargs,
            execution_devices=devices,
            sampler_devices=devices,
            slots_per_device=int(args.candidate_slot_cap_per_device),
            args=args,
            baseline=False,
            owner_scope_seed=int(owner.scope_seed),
            htr_effect_fixture=htr_effect_fixture,
            neural_final_fixture=neural_final_fixture,
            neural_final_plan_root=None,
        )
        if (
            reused_effect_fixture.preparation
            != htr_effect_fixture.preparation
        ):
            raise RuntimeError(
                "candidate changed the serially frozen HTR effect fixture"
            )
        if (
            reused_final_fixture.preparation
            != neural_final_fixture.preparation
        ):
            raise RuntimeError(
                "candidate changed the serially frozen neural final fixture"
            )
        candidates.append(candidate)
    comparisons: list[dict[str, Any]] = []
    comparison_pairs = [
        ("baseline_to_candidate_000", baseline, candidates[0]),
        ("baseline_to_candidate_001", baseline, candidates[1]),
        ("candidate_000_to_candidate_001", candidates[0], candidates[1]),
    ]
    equality_accepted = True
    for name, left, right in comparison_pairs:
        try:
            detail = _compare_prefix_outputs(
                left,
                right,
                relative_tolerance=float(args.prefix_relative_tolerance),
                absolute_tolerance=float(args.prefix_absolute_tolerance),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            equality_accepted = False
            comparisons.append(
                {"comparison": name, "accepted": False, "reason": str(exc)}
            )
        else:
            comparisons.append(
                {"comparison": name, "accepted": True, "detail": detail}
            )
    throughput = _throughput_ratios(
        baseline,
        candidates,
        minimum_ratio=float(args.minimum_throughput_ratio),
    )
    memory_accepted = all(
        run["gpu_summary"]["memory_safety_accepted"]
        for run in (baseline, *candidates)
    )
    threshold_met = all(
        value["throughput_threshold_met"] for value in throughput.values()
    )
    required_kernel_coverage_accepted = _required_kernel_coverage_gate(
        baseline,
        candidates,
        htr_effect_fixture=htr_effect_fixture,
        neural_final_fixture=neural_final_fixture,
    )
    complete_input_plan_bindings_accepted = (
        _complete_input_plan_bindings_gate(
            baseline,
            candidates,
            inventory=complete_input_plan_inventory,
            inventory_descriptor=(
                complete_input_plan_inventory_descriptor
            ),
            expected_plan_root=complete_plan_root,
            neural_final_fixture=neural_final_fixture,
        )
    )
    neural_candidate_provenance_accepted = (
        _neural_candidate_provenance_gate(
            baseline,
            candidates,
            neural_final_fixture=neural_final_fixture,
        )
    )
    acceptance = _calibration_acceptance(
        equality_accepted=equality_accepted,
        memory_accepted=memory_accepted,
        all_phase_throughput_thresholds_met=threshold_met,
        required_kernel_coverage_accepted=(
            required_kernel_coverage_accepted
        ),
        complete_input_plan_bindings_accepted=(
            complete_input_plan_bindings_accepted
        ),
        neural_candidate_provenance_accepted=(
            neural_candidate_provenance_accepted
        ),
    )
    valid = bool(acceptance["calibration_valid"])
    manifest_sha256, manifest_size = _sha256_file(artifact.manifest_path)
    body = {
        "schema_version": R14_KERNEL_CALIBRATION_SCHEMA,
        "prepared_stage1_context": {
            "manifest_path": str(artifact.manifest_path),
            "manifest_sha256": manifest_sha256,
            "manifest_size_bytes": manifest_size,
            "content_root_sha256": artifact.content_root_sha256,
            "scientific_compatibility_sha256": (
                artifact.scientific_compatibility_sha256
            ),
            "ordinary_full_byte_cache_fallback": (
                args.ordinary_full_byte_cache_fallback
            ),
            "htr_model_path_rebinding": str(prepared.htr_model_path),
            "htr_model_tree_sha256": prepared.htr_model_sha256,
        },
        "immutable_source_snapshot": {
            "root": str(snapshot.root),
            "manifest_path": str(snapshot.manifest_path),
            "manifest_sha256": snapshot_manifest_sha256,
            "manifest_size_bytes": snapshot_manifest_size,
            "content_sha256": snapshot.content_sha256,
            "file_count": snapshot.file_count,
            "executed_runner_path": str(
                Path(__file__).resolve(strict=True)
            ),
            "executed_runner_is_registered_frozen_copy": True,
            "complete_file_inventory_reauthenticated": True,
        },
        "canonical_exact_inner_owner": {
            "scope_id": owner.scope_id,
            "canonical_index": int(owner.canonical_index),
            "fit_row_count": int(owner.fit_row_count),
            "logical_scope_ids": [member.scope_id for member in members],
        },
        "calibration_scope": {
            "calibration_only": True,
            "complete_owner_executed": False,
            "completed_producer_or_checkpoint_claimed": False,
            "scientific_equivalence_claimed": False,
            "step_throughput_and_memory_only": True,
            "first_complete_stage1_owner_is_required_full_run_gate": True,
            "complete_plan_authenticated": True,
            "complete_text_optimizer_execution_claimed": False,
            "prefix_batches_only_consumed_by_optimizer": True,
            "full_text_and_scientific_execution_deferred_to_first_owner": True,
            "htr_scope": (
                "all_five_canonical_nuisance_prefixes_and_all_five_"
                "serial_prefix_conditioned_production_effect_tasks"
            ),
            "htr_effect_inputs_derived_from_serial_bounded_nuisance_oof": True,
            "htr_effect_prefix_does_not_claim_full_nuisance_fit": True,
            "matched_scope": "all_five_canonical_matched_htr_folds",
            "neural_inner_scope": "all_five_canonical_inner_fold_tasks",
            "neural_final_scope": (
                "all_three_serial_prefix_conditioned_production_final_"
                "bank_tasks"
            ),
            "final_bank_inputs_derived_from_serial_inner_prefix_candidates": True,
            "same_serial_final_bank_tasks_reused_by_both_candidates": True,
            "final_bank_prefix_does_not_claim_full_inner_fit": True,
        },
        "authenticated_preparation": {
            "htr_complete_plan": copy.deepcopy(dict(htr_plan)),
            "htr_effect_prefix_conditioned_fixture": copy.deepcopy(
                dict(htr_effect_fixture.preparation)
            ),
            "matched_bow_nuisance": copy.deepcopy(dict(bow_preparation)),
            "matched_complete_input_plan": copy.deepcopy(
                dict(matched_complete_plan)
            ),
            "neural_cpu_nuisances_prepared_outside_measured_gpu_window": True,
            "neural_cpu_nuisance_parallel_preparation": copy.deepcopy(
                dict(neural_nuisance_preparation)
            ),
            "neural_complete_input_plan": copy.deepcopy(
                dict(neural_complete_plan)
            ),
            "neural_final_prefix_conditioned_fixture": copy.deepcopy(
                dict(neural_final_fixture.preparation)
            ),
            "complete_input_plan_inventory": copy.deepcopy(
                dict(complete_input_plan_inventory_descriptor)
            ),
        },
        "optimizer_prefix": {
            "warmup_steps_per_kernel": int(args.warmup_optimizer_steps),
            "measured_steps_per_kernel": int(args.measured_optimizer_steps),
            "ready_barrier_after_warmup_observed": all(
                phase["ready_barrier_enforced_after_warmup"]
                for run in (baseline, *candidates)
                for phase in run["phases"].values()
            ),
            "cuda_sync_at_measured_window_start_and_terminal_boundary": True,
            "instrumentation_added_per_step_cuda_synchronization": False,
            "neural_terminal_capture_after_production_normalization_projection": True,
            "loss_prefix_and_terminal_tensor_state_compared": True,
            "htr_and_matched_parameter_bundle_sha256s_validated_independently": True,
            "htr_and_matched_complete_parameter_values_compared_with_tolerance": True,
            "cross_run_parameter_sha256_difference_requires_tolerance_pass": True,
            "htr_and_matched_all_parameter_finiteness_proved": True,
            "matched_pair_count_and_text_identity_compared": True,
            "all_optimizer_parameters_classified_at_both_boundaries": True,
            (
                "stateful_parameters_require_finite_gradients_and_"
                "persistent_adamw_state"
            ): True,
            (
                "stateless_parameters_require_no_gradient_or_optimizer_"
                "state_throughout_the_prefix"
            ): True,
            (
                "terminal_parameter_bundles_include_stateful_and_"
                "stateless_parameters"
            ): True,
            "relative_tolerance": float(args.prefix_relative_tolerance),
            "absolute_tolerance": float(args.prefix_absolute_tolerance),
            "authenticated_source_tolerance_by_phase": (
                copy.deepcopy(source_tolerances)
            ),
        },
        "resources": {
            "candidate_devices": list(devices),
            "baseline_device": devices[0],
            "cpu_budget": int(args.cpu_budget),
            "candidate_slots_per_device": int(
                args.candidate_slot_cap_per_device
            ),
            "canonical_phase_task_counts": copy.deepcopy(
                _EXPECTED_TASK_COUNTS
            ),
            "required_candidate_phase_concurrency": copy.deepcopy(
                _EXPECTED_CANDIDATE_PARALLELISM
            ),
            "gpu_max_allocation_fraction": float(
                args.gpu_max_allocation_fraction
            ),
            "gpu_minimum_headroom_bytes": int(
                args.gpu_minimum_headroom_bytes
            ),
            "gpu_sample_interval_seconds": float(
                args.gpu_sample_interval_seconds
            ),
            "required_gpu_sampler_backend": (
                _REQUIRED_GPU_SAMPLER_BACKEND
            ),
            "memory_acceptance_uses_host_and_conservative_child_bounds": True,
            "child_allocator_bounds_include_allocated_and_reserved_peaks": True,
            "gpu_sample_acquisition_window_must_be_wholly_contained": True,
            "host_sample_required_inside_every_post_warmup_interval": True,
        },
        "serial_baseline": {
            "repetition_count": 1,
            "run": baseline,
            "htr_operational_controls": baseline_htr.as_dict(),
            "neural_operational_controls": baseline_neural.as_dict(),
        },
        "concurrent_candidate": {
            "repetition_count": 2,
            "runs": candidates,
            "htr_operational_controls": candidate_htr.as_dict(),
            "neural_operational_controls": candidate_neural.as_dict(),
        },
        "prefix_output_equality": {
            "accepted": equality_accepted,
            "comparisons": comparisons,
            "exact_discrete_shapes_dtypes_and_finite_masks_required": True,
            "exact_parameter_bundle_and_per_parameter_sha256_integrity_required": True,
            "complete_htr_and_matched_parameter_values_tolerance_compared": True,
            "matched_pair_optimizer_input_identity_required": True,
        },
        "phase_step_throughput": throughput,
        "memory_safety_accepted": memory_accepted,
        "all_phase_throughput_thresholds_met": threshold_met,
        "required_kernel_coverage_accepted": (
            required_kernel_coverage_accepted
        ),
        "complete_input_plan_bindings_accepted": (
            complete_input_plan_bindings_accepted
        ),
        "neural_candidate_provenance_accepted": (
            neural_candidate_provenance_accepted
        ),
        "acceptance_gate": copy.deepcopy(acceptance),
        "multi_gpu_step_throughput_acceleration_claimed": acceptance[
            "multi_gpu_step_throughput_acceleration_claimed"
        ],
        "calibration_valid": valid,
        "deployment_recommendation": acceptance[
            "deployment_recommendation"
        ],
        "first_owner_failure_policy": (
            "stop_and_discard_if_serialized_memory_unsafe_or_science_text_changed"
        ),
    }
    attestation_path = args.output_root / "calibration_attestation.json"
    attestation = _write_self_hashed_json(attestation_path, body)
    print(attestation_path)
    print(f"calibration_content_sha256={attestation['content_sha256']}")
    print(f"calibration_valid={str(valid).lower()}")
    print(
        "multi_gpu_step_throughput_acceleration_claimed="
        f"{str(bool(acceptance['multi_gpu_step_throughput_acceleration_claimed'])).lower()}"
    )
    for phase, row in throughput.items():
        print(
            f"{phase}_single_gpu_baseline_throughput_ratio="
            f"{row['single_gpu_baseline_throughput_ratio']:.6f}"
        )
    return int(acceptance["process_exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
