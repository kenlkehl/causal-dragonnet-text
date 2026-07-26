"""Authenticated dual-GPU reproducibility canary for one Stage 1 scope.

The canary deliberately executes the production legacy scientific worker,
``run_legacy_stage1_scope_worker``, twice from the same private scope
descriptor.  The two replicas have the same scope, seed, model/profile inputs,
and scientific attempt identity.  Only their operational CUDA device differs.

PID, timing, direct GPU assignment, and resource telemetry are kept outside
the within-descriptor scientific comparison.  Legacy descriptor, request, and
plan hashes remain deployment-specific provenance, so this canary makes no
cross-descriptor device-neutrality claim.  Success requires independently
authenticated fragment accumulators and artifact inventories to be
byte-identical.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Mapping, Sequence

from .production_stage1_legacy_scope_adapter import (
    LEGACY_STAGE1_SCOPE_WORKER_TARGET,
    validate_legacy_stage1_scope_descriptor,
)
from .production_stage1_legacy_scope_fragments import (
    validate_legacy_stage1_scope_fragment,
)
from .production_stage1_scope_scheduler import (
    Stage1ScopeExecutionRequest,
    _establish_worker_process_group,
    _enforce_stage1_torch_determinism,
    _observe_stage1_torch_determinism,
    _resolve_worker_target,
    _start_spawned_process_with_scope_hash_seed,
    _terminate_process_and_descendants,
    _validate_torch_determinism_observation,
    seed_stage1_scope_rngs,
    stage1_torch_determinism_policy,
)
from .production_source_snapshot import validate_production_source_snapshot


STAGE1_DUAL_GPU_CANARY_REQUEST_SCHEMA = "production_stage1_dual_gpu_canary_request_v3"
STAGE1_DUAL_GPU_CANARY_REPLICA_SCHEMA = "production_stage1_dual_gpu_canary_replica_v3"
STAGE1_DUAL_GPU_CANARY_RESOURCE_LEDGER_SCHEMA = (
    "production_stage1_dual_gpu_canary_resource_ledger_v2"
)
STAGE1_DUAL_GPU_CANARY_MANIFEST_SCHEMA = "production_stage1_dual_gpu_canary_manifest_v3"
STAGE1_DUAL_GPU_CANARY_SCIENTIFIC_IDENTITY_SCHEMA = (
    "production_stage1_dual_gpu_canary_scientific_identity_v2"
)
STAGE1_DUAL_GPU_CANARY_TEST_DESCRIPTOR_SCHEMA = (
    "production_stage1_dual_gpu_canary_test_descriptor_v3"
)
SOURCE_SNAPSHOT_EXECUTION_ENV = "OCI_PRODUCTION_SOURCE_SNAPSHOT_SHA256"

CANARY_REQUEST_NAME = "canary_request.json"
CANARY_RESOURCE_LEDGER_NAME = "resource_ledger.json"
CANARY_TERMINAL_MANIFEST_NAME = "canary_manifest.json"
REPLICA_TERMINAL_MANIFEST_NAME = "replica_manifest.json"
REPLICA_WORKER_RESULT_NAME = "worker_result.json"
REPLICA_PROCESS_GROUP_MARKER_NAME = "process_group_ready.json"

_SHA256_CHARS = frozenset("0123456789abcdef")
_DEFAULT_MINIMUM_HEADROOM_BYTES = 6 * 1024**3


class _CanaryParentSignal(BaseException):
    """Translate SIGTERM into the canary's owned-child cleanup path."""


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _strict_json_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"JSON object contains duplicate key: {key!r}")
        value[key] = child
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant: {value}")


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_CHARS for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _stable_file(
    path: Path,
    *,
    label: str,
    include_payload: bool = True,
) -> tuple[str, int, bytes]:
    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError(f"{label} must be one real regular file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(supplied, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ValueError(f"{label} must be singly linked")
        payload = bytearray()
        bytes_read = 0
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            bytes_read += len(block)
            if include_payload:
                payload.extend(block)
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        tuple(getattr(before, field) for field in fields)
        != tuple(getattr(after, field) for field in fields)
        or bytes_read != int(after.st_size)
    ):
        raise RuntimeError(f"{label} changed while it was read")
    named = os.stat(supplied, follow_symlinks=False)
    if (
        not stat.S_ISREG(named.st_mode)
        or int(named.st_nlink) != 1
        or (int(named.st_dev), int(named.st_ino))
        != (int(after.st_dev), int(after.st_ino))
    ):
        raise RuntimeError(f"{label} path was substituted while it was read")
    return digest.hexdigest(), int(after.st_size), bytes(payload)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _digest, _size, payload = _stable_file(path, label=label)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one object")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any], *, immutable: bool) -> None:
    if immutable and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"immutable canary file already exists: {path}")
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, temporary_text = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
        )
        temporary = Path(temporary_text)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short canary manifest write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if immutable and (path.exists() or path.is_symlink()):
            raise FileExistsError(f"immutable canary file already exists: {path}")
        os.replace(temporary, path)
        temporary = None
        parent = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _inventory_tree(root: Path, *, label: str) -> list[dict[str, Any]]:
    supplied = Path(root)
    if supplied.is_symlink() or not supplied.is_dir():
        raise ValueError(f"{label} must be one real directory")
    resolved = supplied.resolve(strict=True)
    if resolved != supplied:
        raise ValueError(f"{label} must have a canonical path")
    rows: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("*")):
        state = os.stat(path, follow_symlinks=False)
        if stat.S_ISLNK(state.st_mode):
            raise ValueError(f"{label} contains a symlink")
        if stat.S_ISDIR(state.st_mode):
            continue
        if not stat.S_ISREG(state.st_mode) or int(state.st_nlink) != 1:
            raise ValueError(f"{label} contains a special or linked file")
        digest, size, _payload = _stable_file(
            path,
            label=f"{label} file",
            include_payload=False,
        )
        rows.append(
            {
                "relative_path": path.relative_to(resolved).as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return rows


def _fsync_tree(root: Path) -> None:
    files: list[Path] = []
    directories: list[Path] = [root]
    for path in root.rglob("*"):
        state = os.stat(path, follow_symlinks=False)
        if stat.S_ISLNK(state.st_mode):
            raise ValueError("canary output cannot contain symlinks")
        if stat.S_ISDIR(state.st_mode):
            directories.append(path)
        elif stat.S_ISREG(state.st_mode) and int(state.st_nlink) == 1:
            files.append(path)
        else:
            raise ValueError("canary output contains a special or linked file")
    for path in files:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@dataclass(frozen=True)
class Stage1DualGpuCanaryOptions:
    descriptor_manifest_path: Path
    source_snapshot_root: Path
    output_root: Path
    gpu_ids: tuple[int, int]
    scope_id: str = "outer_001_full"
    resource_poll_seconds: float = 5.0
    maximum_reservation_fraction: float = 0.85
    minimum_headroom_bytes: int = _DEFAULT_MINIMUM_HEADROOM_BYTES
    minimum_concurrency_factor: float = 1.5
    seed: int = 42
    worker_target: str = LEGACY_STAGE1_SCOPE_WORKER_TARGET
    production_worker_required: bool = True
    require_source_snapshot_execution: bool = True
    test_sleep_seconds: float = 0.0
    test_fail_gpu_id: int | None = None
    test_descendant_sentinel_path: Path | None = None


@dataclass(frozen=True)
class _DescriptorBinding:
    manifest_path: Path
    manifest_sha256: str
    manifest_size_bytes: int
    stage1_request_sha256: str
    plan_content_sha256: str
    scope: Mapping[str, Any]
    configured_assignment: Mapping[str, Any]
    replica_logical_gpu_id: int
    descriptor: Any | None

    def __post_init__(self) -> None:
        if (
            type(self.replica_logical_gpu_id) is not int
            or self.replica_logical_gpu_id != 0
        ):
            raise ValueError(
                "isolated canary descriptors must attest replica-local GPU ID zero"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "manifest_size_bytes": self.manifest_size_bytes,
            "stage1_request_sha256": self.stage1_request_sha256,
            "plan_content_sha256": self.plan_content_sha256,
            "scope": copy.deepcopy(dict(self.scope)),
            "configured_assignment": copy.deepcopy(
                dict(self.configured_assignment)
            ),
            "replica_logical_gpu_id": int(self.replica_logical_gpu_id),
        }


@dataclass(frozen=True)
class _CanaryReplicaExecutionRequest(Stage1ScopeExecutionRequest):
    """Preserve configured authority while exposing one replica-local CUDA ID."""

    @property
    def gpu_id(self) -> int:
        return 0


def _validate_options(options: Stage1DualGpuCanaryOptions) -> None:
    if not isinstance(options, Stage1DualGpuCanaryOptions):
        raise TypeError("options must be Stage1DualGpuCanaryOptions")
    if (
        len(options.gpu_ids) != 2
        or any(type(gpu_id) is not int for gpu_id in options.gpu_ids)
        or len(set(options.gpu_ids)) != 2
    ):
        raise ValueError("the reproducibility canary requires exactly two distinct GPUs")
    if any(gpu_id < 0 for gpu_id in options.gpu_ids):
        raise ValueError("canary GPU IDs must be nonnegative")
    if not options.scope_id:
        raise ValueError("canary scope_id cannot be empty")
    if (
        not math.isfinite(float(options.resource_poll_seconds))
        or float(options.resource_poll_seconds) <= 0.0
    ):
        raise ValueError("resource_poll_seconds must be finite and positive")
    if not 0.0 < float(options.maximum_reservation_fraction) < 1.0:
        raise ValueError("maximum_reservation_fraction must be between zero and one")
    if int(options.minimum_headroom_bytes) < 0:
        raise ValueError("minimum_headroom_bytes cannot be negative")
    if (
        not math.isfinite(float(options.minimum_concurrency_factor))
        or float(options.minimum_concurrency_factor) <= 0.0
    ):
        raise ValueError("minimum_concurrency_factor must be finite and positive")
    if (
        isinstance(options.seed, bool)
        or not isinstance(options.seed, int)
        or int(options.seed) < 0
    ):
        raise ValueError("canary seed must be one nonnegative integer")
    if options.production_worker_required:
        if options.worker_target != LEGACY_STAGE1_SCOPE_WORKER_TARGET:
            raise ValueError("production canary cannot switch scientific workers")
        if (
            options.test_fail_gpu_id is not None
            or options.test_sleep_seconds != 0.0
            or options.test_descendant_sentinel_path is not None
        ):
            raise ValueError("test-only controls cannot enter a production canary")
    output = Path(options.output_root)
    if not output.is_absolute():
        raise ValueError("canary output_root must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError("canary output_root must be fresh")
    parent = output.parent.resolve(strict=True)
    if parent != output.parent:
        raise ValueError("canary output_root parent must be canonical")


def _validate_snapshot_binding(
    options: Stage1DualGpuCanaryOptions,
) -> Mapping[str, Any]:
    snapshot = validate_production_source_snapshot(options.source_snapshot_root)
    if options.require_source_snapshot_execution:
        loaded_root = Path(__file__).resolve().parents[2]
        marker = os.environ.get(SOURCE_SNAPSHOT_EXECUTION_ENV)
        if (
            loaded_root != snapshot.root
            or marker != snapshot.content_sha256
            or os.environ.get("PYTHONHASHSEED") != str(int(options.seed))
        ):
            raise ValueError(
                "the canary must execute from the authenticated source snapshot"
            )
    return snapshot.as_dict()


def _descriptor_binding(
    options: Stage1DualGpuCanaryOptions,
) -> _DescriptorBinding:
    supplied = Path(options.descriptor_manifest_path)
    if not supplied.is_absolute():
        raise ValueError("descriptor manifest path must be absolute")
    manifest_path = supplied.resolve(strict=True)
    if supplied != manifest_path:
        raise ValueError("descriptor manifest path must be canonical")
    digest, size, _payload = _stable_file(
        manifest_path,
        label="canary scope descriptor manifest",
    )
    manifest = _load_json(
        manifest_path,
        label="canary scope descriptor manifest",
    )
    request_sha = _require_sha256(
        manifest.get("stage1_request_sha256"),
        label="descriptor stage1_request_sha256",
    )
    if options.production_worker_required:
        descriptor = validate_legacy_stage1_scope_descriptor(
            descriptor_manifest_path=manifest_path,
            expected_stage1_request_sha256=request_sha,
            expected_scope_id=options.scope_id,
            retain_embedding_cache=False,
        )
        configured_gpu_id = descriptor.assignment.gpu_id
        if (
            type(configured_gpu_id) is not int
            or configured_gpu_id not in options.gpu_ids
        ):
            raise ValueError(
                "the dual-GPU canary descriptor assignment is not in the "
                "configured canary GPU inventory"
            )
        if int(descriptor.scope.global_seed) != int(options.seed):
            raise ValueError("canary seed differs from the descriptor global seed")
        return _DescriptorBinding(
            manifest_path=manifest_path,
            manifest_sha256=digest,
            manifest_size_bytes=size,
            stage1_request_sha256=request_sha,
            plan_content_sha256=descriptor.plan_content_sha256,
            scope=descriptor.scope.as_dict(),
            configured_assignment=descriptor.assignment.as_dict(),
            replica_logical_gpu_id=0,
            descriptor=descriptor,
        )
    if (
        set(manifest)
        != {
            "schema_version",
            "stage1_request_sha256",
            "plan_content_sha256",
            "scope",
            "assignment",
            "content_sha256",
        }
        or manifest.get("schema_version")
        != STAGE1_DUAL_GPU_CANARY_TEST_DESCRIPTOR_SCHEMA
    ):
        raise ValueError("test canary descriptor is not closed")
    body = dict(manifest)
    declared = body.pop("content_sha256", None)
    if declared != _sha256_json(body):
        raise ValueError("test canary descriptor identity changed")
    plan_sha = _require_sha256(
        manifest.get("plan_content_sha256"),
        label="test descriptor plan_content_sha256",
    )
    scope = manifest.get("scope")
    assignment = manifest.get("assignment")
    configured_gpu_id = (
        assignment.get("gpu_id")
        if isinstance(assignment, Mapping)
        else None
    )
    if (
        not isinstance(scope, Mapping)
        or scope.get("scope_id") != options.scope_id
        or int(scope.get("scope_seed", -1)) < 0
        or int(scope.get("global_seed", -1)) != int(options.seed)
        or not isinstance(assignment, Mapping)
        or assignment.get("scope_id") != options.scope_id
    ):
        raise ValueError("test canary descriptor has an invalid scope")
    if (
        type(configured_gpu_id) is not int
        or configured_gpu_id not in options.gpu_ids
    ):
        raise ValueError(
            "the test canary descriptor assignment is not in the "
            "configured canary GPU inventory"
        )
    return _DescriptorBinding(
        manifest_path=manifest_path,
        manifest_sha256=digest,
        manifest_size_bytes=size,
        stage1_request_sha256=request_sha,
        plan_content_sha256=plan_sha,
        scope=copy.deepcopy(dict(scope)),
        configured_assignment=copy.deepcopy(dict(assignment)),
        replica_logical_gpu_id=0,
        descriptor=None,
    )


def _build_request(
    *,
    options: Stage1DualGpuCanaryOptions,
    snapshot: Mapping[str, Any],
    descriptor: _DescriptorBinding,
) -> dict[str, Any]:
    scientific_body = {
        "schema_version": "production_stage1_dual_gpu_canary_scientific_request_v3",
        "comparison_domain": "same_authenticated_descriptor_only_v1",
        "source_snapshot_content_sha256": snapshot["content_sha256"],
        "descriptor_manifest_sha256": descriptor.manifest_sha256,
        "stage1_request_sha256": descriptor.stage1_request_sha256,
        "plan_content_sha256": descriptor.plan_content_sha256,
        "scope": copy.deepcopy(dict(descriptor.scope)),
        "worker_target": options.worker_target,
        "global_seed": int(options.seed),
        "determinism_policy": stage1_torch_determinism_policy(),
    }
    scientific_sha = _sha256_json(scientific_body)
    body = {
        "schema_version": STAGE1_DUAL_GPU_CANARY_REQUEST_SCHEMA,
        "output_root": str(Path(options.output_root)),
        "source_snapshot": copy.deepcopy(dict(snapshot)),
        "descriptor": descriptor.as_dict(),
        "scope_id": options.scope_id,
        "gpu_ids": list(map(int, options.gpu_ids)),
        "worker_target": options.worker_target,
        "production_worker_required": bool(options.production_worker_required),
        "strict_spawn": True,
        "native_threads_per_replica": 1,
        "replicas_start_concurrently": True,
        "scientific_request": scientific_body,
        "scientific_request_sha256": scientific_sha,
        "resource_contract": {
            "poll_seconds": float(options.resource_poll_seconds),
            "maximum_reservation_fraction": float(
                options.maximum_reservation_fraction
            ),
            "minimum_headroom_bytes": int(options.minimum_headroom_bytes),
            "minimum_concurrency_factor": float(
                options.minimum_concurrency_factor
            ),
            "external_gpu_process_policy": "abort_canary_without_external_kill",
        },
        "test_controls": (
            None
            if options.production_worker_required
            else {
                "sleep_seconds": float(options.test_sleep_seconds),
                "fail_gpu_id": options.test_fail_gpu_id,
                "descendant_sentinel_path": (
                    None
                    if options.test_descendant_sentinel_path is None
                    else str(options.test_descendant_sentinel_path)
                ),
            }
        ),
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _replica_worker_parameters(
    *,
    options: Stage1DualGpuCanaryOptions,
    descriptor: _DescriptorBinding,
) -> dict[str, Any]:
    if options.production_worker_required:
        return {
            "descriptor_manifest_path": str(descriptor.manifest_path),
            "stage1_request_sha256": descriptor.stage1_request_sha256,
            "scope_id": options.scope_id,
        }
    return {
        "descriptor_manifest_path": str(descriptor.manifest_path),
        "stage1_request_sha256": descriptor.stage1_request_sha256,
        "scope_id": options.scope_id,
        "test_sleep_seconds": float(options.test_sleep_seconds),
        "test_fail_gpu_id": options.test_fail_gpu_id,
        "test_descendant_sentinel_path": (
            None
            if options.test_descendant_sentinel_path is None
            else str(options.test_descendant_sentinel_path)
        ),
    }


def _replica_request(
    *,
    root: Path,
    physical_gpu_id: int,
    canary_request: Mapping[str, Any],
    descriptor: _DescriptorBinding,
    options: Stage1DualGpuCanaryOptions,
) -> Stage1ScopeExecutionRequest:
    del physical_gpu_id
    parameters = _replica_worker_parameters(options=options, descriptor=descriptor)
    return _CanaryReplicaExecutionRequest(
        attempt_dir=str(root),
        plan_content_sha256=descriptor.plan_content_sha256,
        scope=copy.deepcopy(dict(descriptor.scope)),
        # Preserve the descriptor's authenticated configured assignment.  The
        # private request subclass exposes replica-local gpu_id 0 only at the
        # runtime device boundary after CUDA_VISIBLE_DEVICES isolation.
        assignment=copy.deepcopy(dict(descriptor.configured_assignment)),
        worker_target=options.worker_target,
        worker_parameters=parameters,
        worker_parameters_sha256=_sha256_json(parameters),
        # Deliberately identical across replicas. The device is operational.
        attempt_request_sha256=str(canary_request["scientific_request_sha256"]),
    )


def _cpu_fake_canary_worker(
    request: Stage1ScopeExecutionRequest,
) -> Mapping[str, Any]:
    """Importable CPU-only worker used solely by the canary contract tests."""

    parameters = request.worker_parameters
    sentinel = parameters.get("test_descendant_sentinel_path")
    if sentinel is not None:
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import pathlib,sys,time;"
                    "time.sleep(1.5);"
                    "pathlib.Path(sys.argv[1]).write_text("
                    "'orphaned',encoding='utf-8')"
                ),
                str(sentinel),
            ],
            close_fds=True,
        )
    delay = float(parameters.get("test_sleep_seconds", 0.0))
    if delay:
        time.sleep(delay)
    proof = {
        "schema_version": "production_stage1_dual_gpu_canary_fake_proof_v1",
        "scope_id": request.scope_id,
        "scope_seed": int(request.scope_seed),
        "scientific_request_sha256": request.attempt_request_sha256,
        "heldout_labels_supplied": False,
    }
    _atomic_write_json(
        request.payload_dir / "fake_scientific_proof.json",
        proof,
        immutable=True,
    )
    return {
        "scope_id": request.scope_id,
        "heldout_labels_supplied": False,
        "_canary_test_peak_gpu_allocated_bytes": 512 * 1024**2,
        "_canary_test_peak_gpu_reserved_bytes": 1024 * 1024**2,
    }


def _canary_replica_child(
    request: Stage1ScopeExecutionRequest,
    *,
    physical_gpu_id: int,
    canary_request_sha256: str,
    production_worker_required: bool,
    messages: Any,
) -> None:
    attempt = Path(request.attempt_dir)
    started = time.monotonic()
    terminal_published = False
    try:
        _establish_worker_process_group(
            attempt / REPLICA_PROCESS_GROUP_MARKER_NAME
        )
        if (
            not isinstance(physical_gpu_id, int)
            or isinstance(physical_gpu_id, bool)
            or physical_gpu_id < 0
        ):
            raise ValueError("canary physical GPU ID is invalid")
        request.payload_dir.mkdir(parents=False, exist_ok=False)
        before = _enforce_stage1_torch_determinism()
        seed_stage1_scope_rngs(
            request.scope_seed,
            gpu_id=(request.gpu_id if production_worker_required else None),
        )
        import torch
        from threadpoolctl import threadpool_limits

        if production_worker_required and (
            int(request.gpu_id) != 0
            or os.environ.get("CUDA_VISIBLE_DEVICES") is None
            or torch.cuda.device_count() != 1
        ):
            raise RuntimeError(
                "production canary replicas must each expose one logical cuda:0"
            )
        fail_gpu = request.worker_parameters.get("test_fail_gpu_id")
        if (
            not production_worker_required
            and fail_gpu is not None
            and int(fail_gpu) == int(physical_gpu_id)
        ):
            raise RuntimeError("intentional canary peer-failure fixture")
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        messages.put(
            {
                "event": "started",
                "gpu_id": physical_gpu_id,
                "pid": os.getpid(),
            }
        )
        target = _resolve_worker_target(request.worker_target)
        with threadpool_limits(limits=1):
            raw_result = target(request)
        if raw_result is None:
            raw_result = {}
        if not isinstance(raw_result, Mapping):
            raise TypeError("canary scientific worker must return a mapping")
        after = _observe_stage1_torch_determinism()
        comparable_before = {
            key: value
            for key, value in before.items()
            if key not in {"torch_version", "cuda_runtime_version"}
        }
        comparable_after = {
            key: value
            for key, value in after.items()
            if key not in {"torch_version", "cuda_runtime_version"}
        }
        if (
            after.get("policy_active") is not True
            or comparable_before != comparable_after
        ):
            raise RuntimeError("canary worker weakened strict Torch determinism")
        if production_worker_required:
            peak_allocated = int(torch.cuda.max_memory_allocated(int(request.gpu_id)))
            peak_reserved = int(torch.cuda.max_memory_reserved(int(request.gpu_id)))
        else:
            peak_allocated = int(
                raw_result.get("_canary_test_peak_gpu_allocated_bytes", 0)
            )
            peak_reserved = int(
                raw_result.get("_canary_test_peak_gpu_reserved_bytes", 0)
            )
        elapsed = max(time.monotonic() - started, 1e-9)
        sanitized_result = {
            str(key): value
            for key, value in raw_result.items()
            if not str(key).startswith("_canary_test_")
        }
        worker_result = {
            "schema_version": "production_stage1_dual_gpu_canary_worker_result_v2",
            "physical_gpu_id": int(physical_gpu_id),
            "logical_gpu_id": int(request.gpu_id),
            "elapsed_seconds": elapsed,
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "result": sanitized_result,
        }
        _atomic_write_json(
            attempt / REPLICA_WORKER_RESULT_NAME,
            worker_result,
            immutable=True,
        )
        _fsync_tree(attempt)
        payload_files = _inventory_tree(
            request.payload_dir,
            label="canary replica payload",
        )
        worker_sha, worker_size, _worker_bytes = _stable_file(
            attempt / REPLICA_WORKER_RESULT_NAME,
            label="canary replica worker result",
        )
        marker_sha, marker_size, _marker_bytes = _stable_file(
            attempt / REPLICA_PROCESS_GROUP_MARKER_NAME,
            label="canary replica process-group marker",
        )
        body = {
            "schema_version": STAGE1_DUAL_GPU_CANARY_REPLICA_SCHEMA,
            "status": "complete",
            "canary_request_sha256": canary_request_sha256,
            "scientific_request_sha256": request.attempt_request_sha256,
            "plan_content_sha256": request.plan_content_sha256,
            "scope": copy.deepcopy(dict(request.scope)),
            "worker_target": request.worker_target,
            "worker_parameters_sha256": request.worker_parameters_sha256,
            "physical_gpu_id": int(physical_gpu_id),
            "logical_gpu_id": int(request.gpu_id),
            "heldout_labels_supplied": False,
            "determinism_policy": stage1_torch_determinism_policy(),
            "determinism_observed": copy.deepcopy(dict(after)),
            "payload_files": payload_files,
            "worker_result": {
                "relative_path": REPLICA_WORKER_RESULT_NAME,
                "sha256": worker_sha,
                "size_bytes": worker_size,
            },
            "process_group": {
                "relative_path": REPLICA_PROCESS_GROUP_MARKER_NAME,
                "sha256": marker_sha,
                "size_bytes": marker_size,
            },
        }
        replica_manifest = {**body, "content_sha256": _sha256_json(body)}
        _fsync_tree(attempt)
        # Sole per-replica terminal marker, always written last.
        _atomic_write_json(
            attempt / REPLICA_TERMINAL_MANIFEST_NAME,
            replica_manifest,
            immutable=True,
        )
        terminal_published = True
        try:
            messages.put(
                {
                    "event": "completed",
                    "gpu_id": physical_gpu_id,
                    "pid": os.getpid(),
                    "elapsed_seconds": elapsed,
                }
            )
        except BaseException:
            # The terminal manifest, not best-effort telemetry, is
            # authoritative after successful publication.
            pass
    except BaseException as exc:
        if not terminal_published and not (
            attempt / REPLICA_TERMINAL_MANIFEST_NAME
        ).exists():
            failure_body = {
                "schema_version": "production_stage1_dual_gpu_canary_failure_v2",
                "physical_gpu_id": physical_gpu_id,
                "logical_gpu_id": int(request.gpu_id),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            try:
                _atomic_write_json(
                    attempt / "failure.json",
                    {
                        **failure_body,
                        "content_sha256": _sha256_json(failure_body),
                    },
                    immutable=True,
                )
            finally:
                messages.put(
                    {
                        "event": "failed",
                        "gpu_id": physical_gpu_id,
                        "pid": os.getpid(),
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        raise


def _default_resource_sample(
    gpu_ids: Sequence[int],
    *,
    child_pids: Sequence[int],
) -> Mapping[str, Any]:
    del child_pids
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    requested = set(map(int, gpu_ids))
    rows: list[dict[str, Any]] = []
    uuid_to_id: dict[str, int] = {}
    for line in gpu.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6 or not fields[0].isdigit():
            continue
        gpu_id = int(fields[0])
        if gpu_id not in requested:
            continue
        row = {
            "gpu_id": gpu_id,
            "uuid": fields[1],
            "total_mib": int(fields[2]),
            "used_mib": int(fields[3]),
            "free_mib": int(fields[4]),
            "utilization_percent": int(fields[5]),
        }
        rows.append(row)
        uuid_to_id[fields[1]] = gpu_id
    if {row["gpu_id"] for row in rows} != requested:
        raise RuntimeError("nvidia-smi did not report every requested canary GPU")
    applications = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    compute_apps: list[dict[str, Any]] = []
    for line in applications.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if (
            len(fields) < 3
            or fields[0] not in uuid_to_id
            or not fields[1].isdigit()
        ):
            continue
        memory_text = fields[2].split()[0]
        compute_apps.append(
            {
                "gpu_id": uuid_to_id[fields[0]],
                "pid": int(fields[1]),
                "used_memory_mib": (
                    int(memory_text) if memory_text.isdigit() else None
                ),
            }
        )
    return {
        "sampled_at": _utc_now(),
        "gpus": sorted(rows, key=lambda row: int(row["gpu_id"])),
        "compute_apps": sorted(
            compute_apps,
            key=lambda row: (int(row["gpu_id"]), int(row["pid"])),
        ),
    }


def _validate_resource_sample(
    sample: Mapping[str, Any],
    *,
    gpu_ids: Sequence[int],
) -> dict[str, Any]:
    if not isinstance(sample, Mapping):
        raise TypeError("GPU resource sampler must return a mapping")
    rows = sample.get("gpus")
    apps = sample.get("compute_apps")
    if not isinstance(rows, list) or not isinstance(apps, list):
        raise ValueError("GPU resource sample lacks gpus/compute_apps")
    expected = set(map(int, gpu_ids))
    observed: set[int] = set()
    observed_uuids: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("GPU resource row is not an object")
        gpu_id = int(row.get("gpu_id", -1))
        values = {
            "gpu_id": gpu_id,
            "uuid": str(row.get("uuid", "")),
            "total_mib": int(row.get("total_mib", -1)),
            "used_mib": int(row.get("used_mib", -1)),
            "free_mib": int(row.get("free_mib", -1)),
            "utilization_percent": int(row.get("utilization_percent", -1)),
        }
        if (
            gpu_id not in expected
            or gpu_id in observed
            or not values["uuid"]
            or values["uuid"] in observed_uuids
            or values["total_mib"] <= 0
            or min(values["used_mib"], values["free_mib"]) < 0
            or not 0 <= values["utilization_percent"] <= 100
        ):
            raise ValueError("GPU resource row is invalid")
        observed.add(gpu_id)
        observed_uuids.add(values["uuid"])
        normalized_rows.append(values)
    if observed != expected:
        raise ValueError("GPU resource sample has incomplete GPU coverage")
    normalized_apps: list[dict[str, Any]] = []
    for row in apps:
        if not isinstance(row, Mapping):
            raise ValueError("GPU compute-app row is not an object")
        gpu_id = int(row.get("gpu_id", -1))
        pid = int(row.get("pid", -1))
        if gpu_id not in expected or pid <= 0:
            raise ValueError("GPU compute-app row is invalid")
        memory = row.get("used_memory_mib")
        normalized_apps.append(
            {
                "gpu_id": gpu_id,
                "pid": pid,
                "used_memory_mib": None if memory is None else int(memory),
            }
        )
    return {
        "sampled_at": str(sample.get("sampled_at") or _utc_now()),
        "gpus": sorted(normalized_rows, key=lambda row: int(row["gpu_id"])),
        "compute_apps": sorted(
            normalized_apps,
            key=lambda row: (int(row["gpu_id"]), int(row["pid"])),
        ),
    }


def _write_resource_ledger(
    *,
    path: Path,
    canary_request_sha256: str,
    gpu_ids: Sequence[int],
    status: str,
    samples: Sequence[Mapping[str, Any]],
    failure: Mapping[str, Any] | None = None,
) -> None:
    body = {
        "schema_version": STAGE1_DUAL_GPU_CANARY_RESOURCE_LEDGER_SCHEMA,
        "canary_request_sha256": canary_request_sha256,
        "gpu_ids": list(map(int, gpu_ids)),
        "status": str(status),
        "sample_count": len(samples),
        "samples": copy.deepcopy(list(samples)),
        "failure": None if failure is None else copy.deepcopy(dict(failure)),
        "updated_at": _utc_now(),
    }
    _atomic_write_json(
        path,
        {**body, "content_sha256": _sha256_json(body)},
        immutable=False,
    )


def _require_initial_idle_resources(sample: Mapping[str, Any]) -> None:
    """Apply the workflow's conservative no-occupant gate before spawning."""

    if sample.get("compute_apps"):
        raise RuntimeError(
            "canary GPUs are occupied; no external process will be killed"
        )
    unexpected: list[dict[str, Any]] = []
    for row in sample["gpus"]:
        total = int(row["total_mib"])
        used = int(row["used_mib"])
        utilization = int(row["utilization_percent"])
        idle_memory_limit = max(512, int(math.ceil(total * 0.02)))
        if (
            used > idle_memory_limit
            or total - used < 6 * 1024
            or utilization > 1
        ):
            unexpected.append(
                {
                    **dict(row),
                    "idle_memory_limit_mib": idle_memory_limit,
                    "minimum_headroom_mib": 6 * 1024,
                }
            )
    if unexpected:
        raise RuntimeError(
            "canary GPUs are not physically idle: "
            + _canonical_json(unexpected)
        )


def _start_canary_replica(
    process: mp.Process,
    *,
    scope_seed: int,
    physical_gpu_uuid: str,
    production_worker_required: bool,
) -> None:
    """Spawn one worker with a private physical-to-logical CUDA mapping."""

    if not production_worker_required:
        _start_spawned_process_with_scope_hash_seed(
            process,
            scope_seed=scope_seed,
        )
        return
    uuid = str(physical_gpu_uuid)
    if not uuid.startswith("GPU-") or any(character.isspace() for character in uuid):
        raise ValueError("physical GPU UUID is invalid")
    updates = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": uuid,
    }
    previous = {
        key: (key in os.environ, os.environ.get(key))
        for key in updates
    }
    os.environ.update(updates)
    try:
        _start_spawned_process_with_scope_hash_seed(
            process,
            scope_seed=scope_seed,
        )
    finally:
        for key, (present, value) in previous.items():
            if present:
                assert value is not None
                os.environ[key] = value
            else:
                os.environ.pop(key, None)


def _validate_replica(
    *,
    attempt: Path,
    gpu_id: int,
    canary_request: Mapping[str, Any],
    descriptor: _DescriptorBinding,
    options: Stage1DualGpuCanaryOptions,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    manifest = _load_json(
        attempt / REPLICA_TERMINAL_MANIFEST_NAME,
        label=f"GPU {gpu_id} canary replica manifest",
    )
    body = dict(manifest)
    declared = body.pop("content_sha256", None)
    request = _replica_request(
        root=attempt,
        physical_gpu_id=gpu_id,
        canary_request=canary_request,
        descriptor=descriptor,
        options=options,
    )
    expected_fields = {
        "schema_version",
        "status",
        "canary_request_sha256",
        "scientific_request_sha256",
        "plan_content_sha256",
        "scope",
        "worker_target",
        "worker_parameters_sha256",
        "physical_gpu_id",
        "logical_gpu_id",
        "heldout_labels_supplied",
        "determinism_policy",
        "determinism_observed",
        "payload_files",
        "worker_result",
        "process_group",
        "content_sha256",
    }
    if (
        set(manifest) != expected_fields
        or manifest.get("schema_version") != STAGE1_DUAL_GPU_CANARY_REPLICA_SCHEMA
        or manifest.get("status") != "complete"
        or declared != _sha256_json(body)
        or manifest.get("canary_request_sha256")
        != canary_request["content_sha256"]
        or manifest.get("scientific_request_sha256")
        != canary_request["scientific_request_sha256"]
        or manifest.get("plan_content_sha256") != descriptor.plan_content_sha256
        or manifest.get("scope") != dict(descriptor.scope)
        or manifest.get("worker_target") != options.worker_target
        or manifest.get("worker_parameters_sha256")
        != request.worker_parameters_sha256
        or manifest.get("physical_gpu_id") != int(gpu_id)
        or manifest.get("logical_gpu_id")
        != int(descriptor.replica_logical_gpu_id)
        or manifest.get("heldout_labels_supplied") is not False
        or manifest.get("determinism_policy") != stage1_torch_determinism_policy()
    ):
        raise ValueError(f"GPU {gpu_id} replica terminal binding changed")
    _validate_torch_determinism_observation(
        manifest.get("determinism_observed")
    )
    payload_inventory = _inventory_tree(
        attempt / "payload",
        label=f"GPU {gpu_id} canary payload",
    )
    if manifest.get("payload_files") != payload_inventory:
        raise ValueError(f"GPU {gpu_id} canary payload changed")
    worker_result = _load_json(
        attempt / REPLICA_WORKER_RESULT_NAME,
        label=f"GPU {gpu_id} canary worker result",
    )
    if (
        set(worker_result)
        != {
            "schema_version",
            "physical_gpu_id",
            "logical_gpu_id",
            "elapsed_seconds",
            "peak_gpu_allocated_bytes",
            "peak_gpu_reserved_bytes",
            "result",
        }
        or worker_result.get("schema_version")
        != "production_stage1_dual_gpu_canary_worker_result_v2"
        or worker_result.get("physical_gpu_id") != int(gpu_id)
        or worker_result.get("logical_gpu_id")
        != int(descriptor.replica_logical_gpu_id)
        or not isinstance(worker_result.get("elapsed_seconds"), (int, float))
        or isinstance(worker_result.get("elapsed_seconds"), bool)
        or not math.isfinite(float(worker_result["elapsed_seconds"]))
        or float(worker_result["elapsed_seconds"]) <= 0.0
        or type(worker_result.get("peak_gpu_allocated_bytes")) is not int
        or type(worker_result.get("peak_gpu_reserved_bytes")) is not int
        or int(worker_result["peak_gpu_allocated_bytes"]) < 0
        or int(worker_result["peak_gpu_reserved_bytes"]) < 0
        or int(worker_result["peak_gpu_allocated_bytes"])
        > int(worker_result["peak_gpu_reserved_bytes"])
        or not isinstance(worker_result.get("result"), Mapping)
        or worker_result["result"].get("heldout_labels_supplied") is not False
    ):
        raise ValueError(f"GPU {gpu_id} canary worker result is invalid")
    worker_sha, worker_size, _worker_payload = _stable_file(
        attempt / REPLICA_WORKER_RESULT_NAME,
        label=f"GPU {gpu_id} canary worker result",
    )
    if manifest.get("worker_result") != {
        "relative_path": REPLICA_WORKER_RESULT_NAME,
        "sha256": worker_sha,
        "size_bytes": worker_size,
    }:
        raise ValueError(f"GPU {gpu_id} canary worker result changed")
    marker_path = attempt / REPLICA_PROCESS_GROUP_MARKER_NAME
    marker = _load_json(
        marker_path,
        label=f"GPU {gpu_id} canary process-group marker",
    )
    marker_body = dict(marker)
    marker_declared = marker_body.pop("content_sha256", None)
    marker_pid = marker.get("pid")
    if (
        set(marker)
        != {
            "schema_version",
            "pid",
            "process_group_id",
            "process_start_time_ticks",
            "content_sha256",
        }
        or marker.get("schema_version")
        != "production_stage1_worker_process_group_ready_v2"
        or type(marker_pid) is not int
        or int(marker_pid) <= 0
        or marker.get("process_group_id") != marker_pid
        or type(marker.get("process_start_time_ticks")) is not int
        or int(marker["process_start_time_ticks"]) < 0
        or marker_declared != _sha256_json(marker_body)
    ):
        raise ValueError(f"GPU {gpu_id} canary process-group marker changed")
    marker_sha, marker_size, _marker_payload = _stable_file(
        marker_path,
        label=f"GPU {gpu_id} canary process-group marker",
    )
    if manifest.get("process_group") != {
        "relative_path": REPLICA_PROCESS_GROUP_MARKER_NAME,
        "sha256": marker_sha,
        "size_bytes": marker_size,
    }:
        raise ValueError(f"GPU {gpu_id} process-group registration changed")
    observed_root_entries = {path.name for path in attempt.iterdir()}
    if observed_root_entries != {
        "payload",
        REPLICA_WORKER_RESULT_NAME,
        REPLICA_PROCESS_GROUP_MARKER_NAME,
        REPLICA_TERMINAL_MANIFEST_NAME,
    }:
        raise ValueError(f"GPU {gpu_id} canary replica contains extra entries")
    if options.production_worker_required:
        assert descriptor.descriptor is not None
        fragment = validate_legacy_stage1_scope_fragment(
            fragment_root=attempt / "payload" / "legacy_fragment",
            scope_authority=descriptor.descriptor.scope,
            plan_content_sha256=descriptor.plan_content_sha256,
            scope_id=options.scope_id,
            stage1_request_sha256=descriptor.stage1_request_sha256,
            scope_attempt_request_sha256=canary_request[
                "scientific_request_sha256"
            ],
        )
        scientific_body = {
            "schema_version": STAGE1_DUAL_GPU_CANARY_SCIENTIFIC_IDENTITY_SCHEMA,
            "scope_id": options.scope_id,
            "scope_seed": int(descriptor.scope["scope_seed"]),
            "fragment_manifest_content_sha256": (
                fragment.manifest_content_sha256
            ),
            "accumulator_content_sha256": str(
                fragment.accumulator["content_sha256"]
            ),
            "accumulator": copy.deepcopy(dict(fragment.accumulator)),
            "artifacts": copy.deepcopy(list(fragment.artifacts)),
        }
    else:
        scientific_body = {
            "schema_version": STAGE1_DUAL_GPU_CANARY_SCIENTIFIC_IDENTITY_SCHEMA,
            "scope_id": options.scope_id,
            "scope_seed": int(descriptor.scope["scope_seed"]),
            "fragment_manifest_content_sha256": None,
            "accumulator_content_sha256": None,
            "payload_files": payload_inventory,
        }
    scientific = {
        **scientific_body,
        "content_sha256": _sha256_json(scientific_body),
    }
    return manifest, worker_result, scientific, int(marker_pid)


def _terminate_and_join(
    processes: Sequence[mp.Process],
    requests: Mapping[int, Stage1ScopeExecutionRequest],
) -> None:
    for process in processes:
        if process.pid is None:
            continue
        gpu_id = int(process.name.rsplit("-", 1)[-1])
        request = requests.get(gpu_id)
        _terminate_process_and_descendants(
            process,
            process_group_marker_path=(
                None
                if request is None
                else Path(request.attempt_dir)
                / REPLICA_PROCESS_GROUP_MARKER_NAME
            ),
        )


def _resource_summary(
    *,
    samples: Sequence[Mapping[str, Any]],
    worker_results: Mapping[int, Mapping[str, Any]],
    gpu_ids: Sequence[int],
    maximum_reservation_fraction: float,
    minimum_headroom_bytes: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for gpu_id in map(int, gpu_ids):
        gpu_samples = [
            row
            for sample in samples
            for row in sample["gpus"]
            if int(row["gpu_id"]) == gpu_id
        ]
        if not gpu_samples:
            raise RuntimeError(f"no resource samples were recorded for GPU {gpu_id}")
        totals = {int(row["total_mib"]) * 1024**2 for row in gpu_samples}
        uuids = {str(row["uuid"]) for row in gpu_samples}
        if len(totals) != 1 or len(uuids) != 1:
            raise RuntimeError(
                f"GPU {gpu_id} identity or total memory changed during canary"
            )
        total = totals.pop()
        uuid = uuids.pop()
        peak_observed = max(int(row["used_mib"]) * 1024**2 for row in gpu_samples)
        result = worker_results[gpu_id]
        peak_reserved = int(result["peak_gpu_reserved_bytes"])
        peak_allocated = int(result["peak_gpu_allocated_bytes"])
        conservative_peak = max(peak_observed, peak_reserved)
        fraction = conservative_peak / total
        headroom = total - conservative_peak
        accepted = (
            fraction < float(maximum_reservation_fraction)
            and headroom >= int(minimum_headroom_bytes)
        )
        rows.append(
            {
                "gpu_id": gpu_id,
                "uuid": uuid,
                "total_bytes": total,
                "peak_observed_used_bytes": peak_observed,
                "peak_torch_allocated_bytes": peak_allocated,
                "peak_torch_reserved_bytes": peak_reserved,
                "conservative_peak_bytes": conservative_peak,
                "conservative_reservation_fraction": fraction,
                "headroom_bytes": headroom,
                "accepted": accepted,
            }
        )
    if not all(row["accepted"] for row in rows):
        raise RuntimeError("dual-GPU canary exceeded its memory/headroom contract")
    return {
        "maximum_reservation_fraction": float(maximum_reservation_fraction),
        "minimum_headroom_bytes": int(minimum_headroom_bytes),
        "gpus": rows,
        "accepted": True,
    }


def run_stage1_dual_gpu_canary(
    options: Stage1DualGpuCanaryOptions,
    *,
    resource_sampler: Callable[..., Mapping[str, Any]] | None = None,
    cancellation_event: Any | None = None,
) -> Mapping[str, Any]:
    """Run two concurrent replicas and seal an accepted canary result."""

    _validate_options(options)
    if (
        cancellation_event is not None
        and not callable(getattr(cancellation_event, "is_set", None))
    ):
        raise TypeError("cancellation_event must expose is_set()")
    snapshot = _validate_snapshot_binding(options)
    descriptor = _descriptor_binding(options)
    sampler = resource_sampler or _default_resource_sample
    output = Path(options.output_root)
    output.mkdir(parents=False, exist_ok=False)
    replicas_root = output / "replicas"
    replicas_root.mkdir(parents=False, exist_ok=False)
    canary_request = _build_request(
        options=options,
        snapshot=snapshot,
        descriptor=descriptor,
    )
    _atomic_write_json(
        output / CANARY_REQUEST_NAME,
        canary_request,
        immutable=True,
    )
    resource_path = output / CANARY_RESOURCE_LEDGER_NAME
    samples: list[Mapping[str, Any]] = []
    processes: list[mp.Process] = []
    requests: dict[int, Stage1ScopeExecutionRequest] = {}
    messages: Any | None = None
    failure: dict[str, Any] | None = None
    process_groups_cleaned = False
    global_terminal_published = False
    previous_sigterm_handler: Any = None
    sigterm_installed = False
    start_wall = time.monotonic()
    try:
        initial = _validate_resource_sample(
            sampler(options.gpu_ids, child_pids=()),
            gpu_ids=options.gpu_ids,
        )
        samples.append(initial)
        _write_resource_ledger(
            path=resource_path,
            canary_request_sha256=canary_request["content_sha256"],
            gpu_ids=options.gpu_ids,
            status="running",
            samples=samples,
        )
        _require_initial_idle_resources(initial)
        physical_uuid_by_gpu = {
            int(row["gpu_id"]): str(row["uuid"])
            for row in initial["gpus"]
        }
        context = mp.get_context("spawn")
        messages = context.Queue()
        for gpu_id in map(int, options.gpu_ids):
            attempt = replicas_root / f"gpu_{gpu_id:03d}"
            attempt.mkdir(parents=False, exist_ok=False)
            request = _replica_request(
                root=attempt,
                physical_gpu_id=gpu_id,
                canary_request=canary_request,
                descriptor=descriptor,
                options=options,
            )
            requests[gpu_id] = request
            process = context.Process(
                target=_canary_replica_child,
                kwargs={
                    "request": request,
                    "physical_gpu_id": gpu_id,
                    "canary_request_sha256": canary_request["content_sha256"],
                    "production_worker_required": options.production_worker_required,
                    "messages": messages,
                },
                name=f"stage1-canary-gpu-{gpu_id}",
            )
            processes.append(process)
        if threading.current_thread() is threading.main_thread():
            previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

            def _sigterm(signum: int, _frame: Any) -> None:
                raise _CanaryParentSignal(
                    f"dual-GPU canary received signal {signum}"
                )

            signal.signal(signal.SIGTERM, _sigterm)
            sigterm_installed = True
        # Starts are serialized only for the few milliseconds required by
        # multiprocessing. Both long-running scientific workers then overlap.
        for process, gpu_id in zip(processes, options.gpu_ids, strict=True):
            _start_canary_replica(
                process,
                scope_seed=int(requests[int(gpu_id)].scope_seed),
                physical_gpu_uuid=physical_uuid_by_gpu[int(gpu_id)],
                production_worker_required=options.production_worker_required,
            )
        child_pids = tuple(int(process.pid) for process in processes)
        child_pid_by_gpu = {
            int(gpu_id): int(process.pid)
            for process, gpu_id in zip(
                processes,
                options.gpu_ids,
                strict=True,
            )
        }
        completed_messages: set[int] = set()
        last_sample = 0.0
        while True:
            if cancellation_event is not None and cancellation_event.is_set():
                raise RuntimeError("dual-GPU canary cancelled")
            try:
                message = messages.get(timeout=min(options.resource_poll_seconds, 0.25))
            except Empty:
                message = None
            if isinstance(message, Mapping):
                event = str(message.get("event") or "")
                gpu_id = int(message.get("gpu_id", -1))
                if event == "failed":
                    raise RuntimeError(
                        f"GPU {gpu_id} canary replica failed: "
                        f"{message.get('exception_type')}: {message.get('message')}"
                    )
                if event == "completed":
                    completed_messages.add(gpu_id)
            now = time.monotonic()
            if now - last_sample >= float(options.resource_poll_seconds):
                sample = _validate_resource_sample(
                    sampler(options.gpu_ids, child_pids=child_pids),
                    gpu_ids=options.gpu_ids,
                )
                samples.append(sample)
                _write_resource_ledger(
                    path=resource_path,
                    canary_request_sha256=canary_request["content_sha256"],
                    gpu_ids=options.gpu_ids,
                    status="running",
                    samples=samples,
                )
                allowed = set(child_pids)
                external = [
                    app
                    for app in sample["compute_apps"]
                    if int(app["pid"]) not in allowed
                    or int(app["pid"])
                    != child_pid_by_gpu[int(app["gpu_id"])]
                ]
                if external:
                    raise RuntimeError(
                        "an external process entered a canary GPU; aborting "
                        "without killing it"
                    )
                last_sample = now
            exited = [not process.is_alive() for process in processes]
            for process, has_exited in zip(processes, exited, strict=True):
                if has_exited:
                    process.join(timeout=0)
                    if process.exitcode != 0:
                        raise RuntimeError(
                            f"canary child {process.name} exited with "
                            f"code {process.exitcode}"
                        )
            if all(exited):
                break
        # A successful scientific target must not leave background
        # descendants. Authenticate and retire both private groups before any
        # global success marker can be published.
        _terminate_and_join(processes, requests)
        process_groups_cleaned = True
        final_sample = _validate_resource_sample(
            sampler(options.gpu_ids, child_pids=child_pids),
            gpu_ids=options.gpu_ids,
        )
        external = [
            app
            for app in final_sample["compute_apps"]
            if int(app["pid"]) not in set(child_pids)
            or int(app["pid"])
            != child_pid_by_gpu[int(app["gpu_id"])]
        ]
        if external:
            raise RuntimeError(
                "an external process occupied a canary GPU at completion"
            )
        samples.append(final_sample)
        elapsed_wall = max(time.monotonic() - start_wall, 1e-9)
        replica_manifests: dict[int, Mapping[str, Any]] = {}
        worker_results: dict[int, Mapping[str, Any]] = {}
        scientific_identities: dict[int, Mapping[str, Any]] = {}
        for gpu_id in map(int, options.gpu_ids):
            replica, worker, scientific, _process_group_pid = _validate_replica(
                attempt=Path(requests[gpu_id].attempt_dir),
                gpu_id=gpu_id,
                canary_request=canary_request,
                descriptor=descriptor,
                options=options,
            )
            replica_manifests[gpu_id] = replica
            worker_results[gpu_id] = worker
            scientific_identities[gpu_id] = scientific
        first_gpu, second_gpu = map(int, options.gpu_ids)
        if scientific_identities[first_gpu] != scientific_identities[second_gpu]:
            raise RuntimeError(
                "dual-GPU replicas produced different scientific identities"
            )
        sum_replica_seconds = sum(
            float(worker_results[gpu_id]["elapsed_seconds"])
            for gpu_id in map(int, options.gpu_ids)
        )
        concurrency_factor = sum_replica_seconds / elapsed_wall
        if concurrency_factor < float(options.minimum_concurrency_factor):
            raise RuntimeError(
                "dual-GPU canary did not demonstrate useful concurrent throughput"
            )
        resource_summary = _resource_summary(
            samples=samples,
            worker_results=worker_results,
            gpu_ids=options.gpu_ids,
            maximum_reservation_fraction=options.maximum_reservation_fraction,
            minimum_headroom_bytes=options.minimum_headroom_bytes,
        )
        _write_resource_ledger(
            path=resource_path,
            canary_request_sha256=canary_request["content_sha256"],
            gpu_ids=options.gpu_ids,
            status="complete",
            samples=samples,
        )
        request_sha, request_size, _request_bytes = _stable_file(
            output / CANARY_REQUEST_NAME,
            label="canary request",
        )
        resource_sha, resource_size, _resource_bytes = _stable_file(
            resource_path,
            label="canary resource ledger",
        )
        replicas = []
        for gpu_id in map(int, options.gpu_ids):
            manifest_path = (
                Path(requests[gpu_id].attempt_dir)
                / REPLICA_TERMINAL_MANIFEST_NAME
            )
            manifest_sha, manifest_size, _manifest_bytes = _stable_file(
                manifest_path,
                label=f"GPU {gpu_id} replica manifest",
            )
            replicas.append(
                {
                    "gpu_id": gpu_id,
                    "relative_manifest_path": manifest_path.relative_to(
                        output
                    ).as_posix(),
                    "manifest_sha256": manifest_sha,
                    "manifest_size_bytes": manifest_size,
                    "replica_manifest_content_sha256": replica_manifests[gpu_id][
                        "content_sha256"
                    ],
                    "scientific_identity_content_sha256": scientific_identities[
                        gpu_id
                    ]["content_sha256"],
                    "elapsed_seconds": float(
                        worker_results[gpu_id]["elapsed_seconds"]
                    ),
                }
            )
        body = {
            "schema_version": STAGE1_DUAL_GPU_CANARY_MANIFEST_SCHEMA,
            "status": "accepted",
            "canary_request_sha256": canary_request["content_sha256"],
            "request_file": {
                "relative_path": CANARY_REQUEST_NAME,
                "sha256": request_sha,
                "size_bytes": request_size,
            },
            "resource_ledger": {
                "relative_path": CANARY_RESOURCE_LEDGER_NAME,
                "sha256": resource_sha,
                "size_bytes": resource_size,
            },
            "source_snapshot": copy.deepcopy(dict(snapshot)),
            "descriptor": descriptor.as_dict(),
            "scope_id": options.scope_id,
            "worker_target": options.worker_target,
            "production_worker_required": options.production_worker_required,
            "scientific_identity": copy.deepcopy(
                dict(scientific_identities[first_gpu])
            ),
            "scientific_equality": True,
            "replicas": replicas,
            "execution": {
                "wall_seconds": elapsed_wall,
                "sum_replica_seconds": sum_replica_seconds,
                "concurrency_factor": concurrency_factor,
                "minimum_concurrency_factor": float(
                    options.minimum_concurrency_factor
                ),
                "completion_messages_received": sorted(completed_messages),
            },
            "resource_summary": resource_summary,
            "external_processes_killed": False,
        }
        terminal = {**body, "content_sha256": _sha256_json(body)}
        _fsync_tree(output)
        # Sole global terminal marker, always written last.
        _atomic_write_json(
            output / CANARY_TERMINAL_MANIFEST_NAME,
            terminal,
            immutable=True,
        )
        global_terminal_published = True
        return validate_stage1_dual_gpu_canary(
            output,
            allow_test_worker=not options.production_worker_required,
            require_source_snapshot_execution=(
                options.require_source_snapshot_execution
            ),
        )
    except BaseException as exc:
        failure = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        if not process_groups_cleaned:
            _terminate_and_join(processes, requests)
            process_groups_cleaned = True
        if (
            not global_terminal_published
            and output.exists()
            and resource_path.exists()
        ):
            try:
                _write_resource_ledger(
                    path=resource_path,
                    canary_request_sha256=canary_request["content_sha256"],
                    gpu_ids=options.gpu_ids,
                    status="failed",
                    samples=samples,
                    failure=failure,
                )
            except BaseException:
                pass
        raise
    finally:
        if not process_groups_cleaned:
            _terminate_and_join(processes, requests)
        if messages is not None:
            try:
                messages.close()
                messages.join_thread()
            except BaseException:
                pass
        if sigterm_installed:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)


def validate_stage1_dual_gpu_canary(
    output_root: Path | str,
    *,
    allow_test_worker: bool = False,
    require_source_snapshot_execution: bool = True,
) -> Mapping[str, Any]:
    """Fresh, path-only authentication of a completed canary."""

    supplied = Path(output_root)
    if not supplied.is_absolute() or supplied.is_symlink() or not supplied.is_dir():
        raise ValueError("canary output must be one absolute real directory")
    root = supplied.resolve(strict=True)
    if root != supplied:
        raise ValueError("canary output path must be canonical")
    request = _load_json(root / CANARY_REQUEST_NAME, label="canary request")
    request_body = dict(request)
    request_declared = request_body.pop("content_sha256", None)
    if (
        request.get("schema_version") != STAGE1_DUAL_GPU_CANARY_REQUEST_SCHEMA
        or request_declared != _sha256_json(request_body)
        or request.get("output_root") != str(root)
        or not isinstance(request.get("descriptor"), Mapping)
        or not isinstance(request.get("source_snapshot"), Mapping)
    ):
        raise ValueError("canary request identity changed")
    production = bool(request.get("production_worker_required"))
    if not production and not allow_test_worker:
        raise ValueError("test-worker canary is not a production result")
    options = Stage1DualGpuCanaryOptions(
        descriptor_manifest_path=Path(request["descriptor"]["manifest_path"]),
        source_snapshot_root=Path(request["source_snapshot"]["root"]),
        output_root=root,
        scope_id=str(request["scope_id"]),
        gpu_ids=tuple(map(int, request["gpu_ids"])),  # type: ignore[arg-type]
        resource_poll_seconds=float(request["resource_contract"]["poll_seconds"]),
        maximum_reservation_fraction=float(
            request["resource_contract"]["maximum_reservation_fraction"]
        ),
        minimum_headroom_bytes=int(
            request["resource_contract"]["minimum_headroom_bytes"]
        ),
        minimum_concurrency_factor=float(
            request["resource_contract"]["minimum_concurrency_factor"]
        ),
        seed=int(request["scientific_request"]["global_seed"]),
        worker_target=str(request["worker_target"]),
        production_worker_required=production,
        require_source_snapshot_execution=require_source_snapshot_execution,
        test_sleep_seconds=(
            0.0
            if production
            else float(request["test_controls"]["sleep_seconds"])
        ),
        test_fail_gpu_id=(
            None if production else request["test_controls"]["fail_gpu_id"]
        ),
        test_descendant_sentinel_path=(
            None
            if production
            or request["test_controls"]["descendant_sentinel_path"] is None
            else Path(request["test_controls"]["descendant_sentinel_path"])
        ),
    )
    snapshot = _validate_snapshot_binding(options)
    if snapshot != request["source_snapshot"]:
        raise ValueError("canary source snapshot changed")
    descriptor = _descriptor_binding(options)
    if descriptor.as_dict() != request["descriptor"]:
        raise ValueError("canary descriptor changed")
    expected_request = _build_request(
        options=options,
        snapshot=snapshot,
        descriptor=descriptor,
    )
    if expected_request != request:
        raise ValueError("canary request does not reconstruct exactly")
    terminal = _load_json(
        root / CANARY_TERMINAL_MANIFEST_NAME,
        label="canary terminal manifest",
    )
    terminal_body = dict(terminal)
    terminal_declared = terminal_body.pop("content_sha256", None)
    if (
        terminal.get("schema_version") != STAGE1_DUAL_GPU_CANARY_MANIFEST_SCHEMA
        or terminal.get("status") != "accepted"
        or terminal_declared != _sha256_json(terminal_body)
        or terminal.get("canary_request_sha256") != request["content_sha256"]
        or terminal.get("source_snapshot") != snapshot
        or terminal.get("descriptor") != descriptor.as_dict()
        or terminal.get("scope_id") != options.scope_id
        or terminal.get("worker_target") != options.worker_target
        or terminal.get("production_worker_required") is not production
        or terminal.get("scientific_equality") is not True
        or terminal.get("external_processes_killed") is not False
    ):
        raise ValueError("canary terminal manifest binding changed")
    request_sha, request_size, _request_bytes = _stable_file(
        root / CANARY_REQUEST_NAME,
        label="canary request",
    )
    resource_sha, resource_size, _resource_bytes = _stable_file(
        root / CANARY_RESOURCE_LEDGER_NAME,
        label="canary resource ledger",
    )
    if terminal.get("request_file") != {
        "relative_path": CANARY_REQUEST_NAME,
        "sha256": request_sha,
        "size_bytes": request_size,
    } or terminal.get("resource_ledger") != {
        "relative_path": CANARY_RESOURCE_LEDGER_NAME,
        "sha256": resource_sha,
        "size_bytes": resource_size,
    }:
        raise ValueError("canary terminal file registrations changed")
    ledger = _load_json(
        root / CANARY_RESOURCE_LEDGER_NAME,
        label="canary resource ledger",
    )
    ledger_body = dict(ledger)
    ledger_declared = ledger_body.pop("content_sha256", None)
    if (
        ledger.get("schema_version")
        != STAGE1_DUAL_GPU_CANARY_RESOURCE_LEDGER_SCHEMA
        or ledger.get("status") != "complete"
        or ledger_declared != _sha256_json(ledger_body)
        or ledger.get("canary_request_sha256") != request["content_sha256"]
        or ledger.get("gpu_ids") != list(options.gpu_ids)
        or ledger.get("sample_count") != len(ledger.get("samples") or ())
        or ledger.get("failure") is not None
    ):
        raise ValueError("canary resource ledger changed")
    normalized_samples = [
        _validate_resource_sample(sample, gpu_ids=options.gpu_ids)
        for sample in ledger["samples"]
    ]
    if normalized_samples != ledger["samples"]:
        raise ValueError("canary resource samples are not canonical")
    if normalized_samples[0]["compute_apps"]:
        raise ValueError("canary began on an occupied GPU")
    expected_replicas: list[dict[str, Any]] = []
    scientific: list[Mapping[str, Any]] = []
    worker_results: dict[int, Mapping[str, Any]] = {}
    process_group_pids: set[int] = set()
    process_group_pid_by_gpu: dict[int, int] = {}
    for gpu_id in options.gpu_ids:
        attempt = root / "replicas" / f"gpu_{gpu_id:03d}"
        replica, worker, identity, process_group_pid = _validate_replica(
            attempt=attempt,
            gpu_id=gpu_id,
            canary_request=request,
            descriptor=descriptor,
            options=options,
        )
        manifest_path = attempt / REPLICA_TERMINAL_MANIFEST_NAME
        manifest_sha, manifest_size, _manifest_bytes = _stable_file(
            manifest_path,
            label=f"GPU {gpu_id} replica manifest",
        )
        expected_replicas.append(
            {
                "gpu_id": gpu_id,
                "relative_manifest_path": manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": manifest_sha,
                "manifest_size_bytes": manifest_size,
                "replica_manifest_content_sha256": replica["content_sha256"],
                "scientific_identity_content_sha256": identity["content_sha256"],
                "elapsed_seconds": float(worker["elapsed_seconds"]),
            }
        )
        scientific.append(identity)
        worker_results[int(gpu_id)] = worker
        process_group_pids.add(process_group_pid)
        process_group_pid_by_gpu[int(gpu_id)] = process_group_pid
    if (
        len(scientific) != 2
        or scientific[0] != scientific[1]
        or terminal.get("scientific_identity") != scientific[0]
        or terminal.get("replicas") != expected_replicas
    ):
        raise ValueError("canary scientific equality proof changed")
    expected_resource_summary = _resource_summary(
        samples=normalized_samples,
        worker_results=worker_results,
        gpu_ids=options.gpu_ids,
        maximum_reservation_fraction=options.maximum_reservation_fraction,
        minimum_headroom_bytes=options.minimum_headroom_bytes,
    )
    if terminal.get("resource_summary") != expected_resource_summary:
        raise ValueError("canary resource acceptance proof changed")
    if len(process_group_pids) != 2 or any(
        int(app["pid"]) not in process_group_pids
        or int(app["pid"])
        != process_group_pid_by_gpu[int(app["gpu_id"])]
        for sample in normalized_samples[1:]
        for app in sample["compute_apps"]
    ):
        raise ValueError("canary resource ledger contains an external GPU process")
    execution = terminal.get("execution")
    sum_replica_seconds = sum(
        float(worker_results[int(gpu_id)]["elapsed_seconds"])
        for gpu_id in options.gpu_ids
    )
    if not isinstance(execution, Mapping):
        raise ValueError("canary execution summary is absent")
    wall_seconds = execution.get("wall_seconds")
    concurrency_factor = execution.get("concurrency_factor")
    completion_messages = execution.get("completion_messages_received")
    if (
        not isinstance(wall_seconds, (int, float))
        or isinstance(wall_seconds, bool)
        or not math.isfinite(float(wall_seconds))
        or float(wall_seconds) <= 0.0
        or execution.get("sum_replica_seconds") != sum_replica_seconds
        or execution.get("minimum_concurrency_factor")
        != float(options.minimum_concurrency_factor)
        or not isinstance(concurrency_factor, (int, float))
        or isinstance(concurrency_factor, bool)
        or not math.isfinite(float(concurrency_factor))
        or not math.isclose(
            float(concurrency_factor),
            sum_replica_seconds / float(wall_seconds),
            rel_tol=1e-12,
            abs_tol=0.0,
        )
        or float(concurrency_factor)
        < float(options.minimum_concurrency_factor)
        or not isinstance(completion_messages, list)
        or completion_messages != sorted(set(map(int, completion_messages)))
        or not set(map(int, completion_messages)).issubset(set(options.gpu_ids))
    ):
        raise ValueError("canary concurrency proof changed")
    observed_files = {
        row["relative_path"]
        for row in _inventory_tree(root, label="completed canary tree")
    }
    expected_files = {
        CANARY_REQUEST_NAME,
        CANARY_RESOURCE_LEDGER_NAME,
        CANARY_TERMINAL_MANIFEST_NAME,
    }
    for gpu_id in options.gpu_ids:
        attempt = root / "replicas" / f"gpu_{gpu_id:03d}"
        expected_files.update(
            f"replicas/gpu_{gpu_id:03d}/{row['relative_path']}"
            for row in _inventory_tree(
                attempt,
                label=f"GPU {gpu_id} completed replica",
            )
        )
    if observed_files != expected_files:
        raise ValueError("completed canary contains unregistered files")
    return copy.deepcopy(terminal)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one full-profile Stage 1 scope concurrently on two GPUs and "
            "require exact scientific reproducibility."
        )
    )
    parser.add_argument("--descriptor-manifest", required=True, type=Path)
    parser.add_argument("--source-snapshot-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scope-id", default="outer_001_full")
    parser.add_argument(
        "--gpu-id",
        action="append",
        required=True,
        type=int,
        help="Repeat exactly twice; each replica receives one GPU.",
    )
    parser.add_argument("--resource-poll-seconds", type=float, default=5.0)
    parser.add_argument("--maximum-reservation-fraction", type=float, default=0.85)
    parser.add_argument(
        "--minimum-headroom-gib",
        type=float,
        default=6.0,
    )
    parser.add_argument("--minimum-concurrency-factor", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_stage1_dual_gpu_canary_args(
    argv: Sequence[str] | None = None,
) -> Stage1DualGpuCanaryOptions:
    parser = _parser()
    values = parser.parse_args(argv)
    if len(values.gpu_id) != 2 or len(set(values.gpu_id)) != 2:
        parser.error("--gpu-id must be supplied exactly twice with distinct IDs")
    return Stage1DualGpuCanaryOptions(
        descriptor_manifest_path=values.descriptor_manifest,
        source_snapshot_root=values.source_snapshot_root,
        output_root=values.output_root,
        scope_id=str(values.scope_id),
        gpu_ids=tuple(values.gpu_id),  # type: ignore[arg-type]
        resource_poll_seconds=float(values.resource_poll_seconds),
        maximum_reservation_fraction=float(values.maximum_reservation_fraction),
        minimum_headroom_bytes=int(float(values.minimum_headroom_gib) * 1024**3),
        minimum_concurrency_factor=float(values.minimum_concurrency_factor),
        seed=int(values.seed),
    )


def _reexec_from_source_snapshot(
    *,
    source_snapshot_root: Path,
    seed: int,
    raw_argv: Sequence[str],
) -> None:
    snapshot = validate_production_source_snapshot(source_snapshot_root)
    loaded_root = Path(__file__).resolve().parents[2]
    marker = os.environ.get(SOURCE_SNAPSHOT_EXECUTION_ENV)
    expected_hash_seed = str(int(seed))
    if marker is not None:
        if (
            marker != snapshot.content_sha256
            or loaded_root != snapshot.root
            or os.environ.get("PYTHONHASHSEED") != expected_hash_seed
        ):
            raise RuntimeError(
                "source-snapshot marker, loaded canary source, or "
                "PYTHONHASHSEED does not match"
            )
        return
    entrypoint = (
        snapshot.root
        / "scripts"
        / "run_stage1_dual_gpu_reproducibility_canary.py"
    )
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise FileNotFoundError("source snapshot lacks the dual-GPU canary CLI")
    environment = os.environ.copy()
    environment[SOURCE_SNAPSHOT_EXECUTION_ENV] = snapshot.content_sha256
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(snapshot.root)
    environment["PYTHONHASHSEED"] = expected_hash_seed
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-P",
            "-u",
            str(entrypoint),
            *map(str, raw_argv),
        ],
        environment,
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    options = parse_stage1_dual_gpu_canary_args(raw_argv)
    _reexec_from_source_snapshot(
        source_snapshot_root=options.source_snapshot_root,
        seed=options.seed,
        raw_argv=raw_argv,
    )
    result = run_stage1_dual_gpu_canary(options)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
