"""Deterministic, manifest-sealed orchestration for production Stage 1 scopes.

This module contains no scientific model implementation.  It turns the
authoritative outer/inner registry into a closed execution plan, assigns that
plan to explicit GPU slots independently of completion order, and provides a
spawn-only execution substrate whose reusable unit is one fully sealed
physical-fit-owner attempt. Distinct logical purposes are retained as
authenticated references to those attempts.

The scheduler deliberately keeps labels out of its scope/request types.  A
worker receives fit and held-out row identities, but a scientific worker is
still responsible for projecting fit labels and exposing only held-out text.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import signal
import stat
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .physical_fit_deduplication import PhysicalFitKey, ordered_row_identity

STAGE1_SCOPE_PLAN_SCHEMA = "production_stage1_scope_plan_v7"
STAGE1_SCOPE_SCIENTIFIC_PLAN_SCHEMA = (
    "production_stage1_scope_scientific_plan_v3"
)
STAGE1_PHYSICAL_FIT_IDENTITY_SCHEMA = (
    "production_stage1_physical_fit_identity_v1"
)
STAGE1_SCOPE_ATTEMPT_REQUEST_SCHEMA = "production_stage1_scope_attempt_request_v4"
STAGE1_SCOPE_ATTEMPT_MANIFEST_SCHEMA = "production_stage1_scope_attempt_manifest_v4"
STAGE1_SCOPE_PROGRESS_SCHEMA = "production_stage1_scope_progress_v2"
STAGE1_SCOPE_WORKER_RESULT_SCHEMA = "production_stage1_scope_worker_result_v3"
STAGE1_LOGICAL_SCOPE_BINDING_SET_SCHEMA = (
    "production_stage1_logical_scope_binding_set_v3"
)
STAGE1_LOGICAL_SCOPE_BINDING_SCHEMA = (
    "production_stage1_logical_scope_binding_v3"
)
STAGE1_LOGICAL_SCOPE_BINDING_FILENAME = "logical_scope_bindings.json"
STAGE1_TORCH_DETERMINISM_POLICY_SCHEMA = (
    "production_stage1_torch_determinism_policy_v1"
)
STAGE1_ATTEMPT_FILESYSTEM_IDENTITY_SCHEMA = (
    "production_stage1_attempt_filesystem_identity_v1"
)
STAGE1_ATTEMPT_STORE_SCHEMA = "production_stage1_scope_attempt_store_v2"

_SCOPE_ID = re.compile(
    r"^outer_[0-9]{3}_(?:full|inner_[0-9]{3}|hierarchy_epoch_[0-9]{3})$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WORKER_TARGET = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
)
_ATTEMPT_NAME = re.compile(
    r"^attempt_[0-9]{8}T[0-9]{12}Z_[0-9a-f]{32}$"
)
_UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:"
    r"[0-9]{2}\.[0-9]{6}Z$"
)
_FORBIDDEN_WORKER_PARAMETER = re.compile(
    r"(?:^|_)(?:oracle|ground_truth|true_(?:age|pdl1|ite)|"
    r"heldout_(?:treatment|outcome)|held_out_(?:treatment|outcome))(?:_|$)",
    re.IGNORECASE,
)
_TERMINAL_SCOPE_STATUSES = frozenset({"completed"})
_PROGRESS_STATUSES = frozenset(
    {"pending", "running", "sealing", "completed", "failed"}
)


class _Stage1ParentSignal(BaseException):
    """Turn a parent SIGTERM into normal orchestrator cleanup."""


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


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"JSON object contains duplicate key: {key!r}")
        value[key] = child
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains a non-finite constant: {value}")


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _absolute_path(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path))))


def _open_safe_directory(path: Path | str, *, label: str) -> tuple[int, Path]:
    absolute = _absolute_path(path)
    try:
        resolved = absolute.resolve(strict=True)
        metadata = os.lstat(absolute)
    except OSError as exc:
        raise ValueError(f"{label} does not exist as a safe directory") from exc
    if (
        resolved != absolute
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ValueError(f"{label} must be a symlink-free directory")
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ValueError(f"{label} could not be opened safely") from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
    ):
        os.close(descriptor)
        raise ValueError(f"{label} changed while it was opened")
    return descriptor, absolute


def _directory_inode_binding(
    path: Path | str,
    *,
    label: str,
) -> dict[str, Any]:
    descriptor, absolute = _open_safe_directory(path, label=label)
    try:
        metadata = os.fstat(descriptor)
        body = {
            "schema_version": "production_stage1_directory_inode_binding_v1",
            "absolute_path": str(absolute),
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
        }
        return {**body, "content_sha256": _sha256_json(body)}
    finally:
        os.close(descriptor)


def _validate_directory_inode_binding(
    value: Any,
    *,
    path: Path | str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} lacks its directory inode binding")
    expected = _directory_inode_binding(path, label=label)
    if dict(value) != expected:
        raise ValueError(f"{label} directory was atomically substituted")
    return expected


def _read_regular_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[bytes, str, tuple[int, ...]]:
    if not name or "/" in name or name in {".", ".."}:
        raise ValueError(f"{label} has an unsafe basename")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f"{label} is not a safe regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                f"{label} must be a regular file with exactly one hard link"
            )
        blocks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        if _stable_stat_identity(before) != _stable_stat_identity(after):
            raise ValueError(f"{label} changed while it was read")
        payload = b"".join(blocks)
        if len(payload) != int(before.st_size):
            raise ValueError(f"{label} size changed while it was read")
        return payload, digest.hexdigest(), _stable_stat_identity(before)
    finally:
        os.close(descriptor)


def _hash_regular_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[int, str, tuple[int, ...]]:
    """Hash one regular file without materializing its bytes."""

    if not name or "/" in name or name in {".", ".."}:
        raise ValueError(f"{label} has an unsafe basename")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f"{label} is not a safe regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                f"{label} must be a regular file with exactly one hard link"
            )
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
        after = os.fstat(descriptor)
        if _stable_stat_identity(before) != _stable_stat_identity(after):
            raise ValueError(f"{label} changed while it was hashed")
        if size != int(before.st_size):
            raise ValueError(f"{label} size changed while it was hashed")
        return size, digest.hexdigest(), _stable_stat_identity(before)
    finally:
        os.close(descriptor)


def _read_regular_file(
    path: Path | str,
    *,
    label: str,
) -> tuple[bytes, str, tuple[int, ...]]:
    absolute = _absolute_path(path)
    directory_fd, parent = _open_safe_directory(
        absolute.parent,
        label=f"{label} parent",
    )
    try:
        payload, digest, identity = _read_regular_at(
            directory_fd,
            absolute.name,
            label=label,
        )
        try:
            after = os.stat(
                absolute.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ValueError(f"{label} disappeared after it was read") from exc
        if _stable_stat_identity(after) != identity:
            raise ValueError(f"{label} changed after it was read")
        return payload, digest, identity
    finally:
        os.close(directory_fd)


def _sha256_file(path: Path) -> str:
    return _read_regular_file(path, label=f"file {path}")[1]


def _load_strict_json_file(path: Path | str, *, label: str) -> Any:
    payload, _digest, _identity = _read_regular_file(path, label=label)
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def stage1_torch_determinism_policy() -> dict[str, Any]:
    """Return the immutable deterministic-compute contract for every scope."""

    return {
        "schema_version": STAGE1_TORCH_DETERMINISM_POLICY_SCHEMA,
        "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "deterministic_debug_mode": "error",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "failure_policy": "abort_scope_on_unsupported_nondeterministic_operation",
    }


def _enforce_stage1_torch_determinism() -> Mapping[str, Any]:
    """Activate and verify the plan's strict Torch determinism policy."""

    policy = stage1_torch_determinism_policy()
    # This must be set before a CUDA BLAS workspace is initialized.  Each
    # scope runs in a fresh spawned process, so changing it cannot affect a
    # peer scope or the parent.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(
        policy["cublas_workspace_config"]
    )
    try:
        import torch
    except ImportError:  # pragma: no cover - Torch is a production dependency.
        return {
            **policy,
            "torch_available": False,
            "policy_active": True,
        }

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_deterministic_debug_mode("error")
    observed = _observe_stage1_torch_determinism()
    if not observed["policy_active"]:
        raise RuntimeError("strict Stage 1 Torch determinism could not be enabled")
    return observed


def _observe_stage1_torch_determinism() -> Mapping[str, Any]:
    """Read back the effective policy, including after a worker target runs."""

    policy = stage1_torch_determinism_policy()
    try:
        import torch
    except ImportError:  # pragma: no cover - Torch is a production dependency.
        return {
            **policy,
            "torch_available": False,
            "policy_active": True,
        }
    active = bool(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        == policy["cublas_workspace_config"]
        and torch.backends.cudnn.benchmark is False
        and torch.backends.cudnn.deterministic is True
        and torch.are_deterministic_algorithms_enabled()
        and torch.is_deterministic_algorithms_warn_only_enabled() is False
        and str(torch.get_deterministic_debug_mode()) == "2"
        and torch.backends.cuda.matmul.allow_tf32 is False
        and torch.backends.cudnn.allow_tf32 is False
    )
    return {
        **policy,
        "torch_available": True,
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "policy_active": active,
    }


def _validate_torch_determinism_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Torch determinism observation must be an object")
    expected_policy = stage1_torch_determinism_policy()
    required = {
        *expected_policy,
        "torch_available",
        "policy_active",
    }
    allowed = {
        *required,
        "torch_version",
        "cuda_runtime_version",
    }
    if set(value) != required and set(value) != allowed:
        raise ValueError("Torch determinism observation has an invalid schema")
    if any(value.get(key) != expected for key, expected in expected_policy.items()):
        raise ValueError("Torch determinism observation changed the declared policy")
    if value.get("policy_active") is not True:
        raise ValueError("Torch determinism observation is not active")
    available = value.get("torch_available")
    if available is not True and available is not False:
        raise ValueError("Torch determinism observation has an invalid availability")
    if available is False and set(value) != required:
        raise ValueError("Torch-unavailable observation contains runtime fields")
    if available is True:
        if set(value) != allowed or not isinstance(value.get("torch_version"), str):
            raise ValueError("Torch determinism observation lacks runtime identity")
        cuda_version = value.get("cuda_runtime_version")
        if cuda_version is not None and not isinstance(cuda_version, str):
            raise ValueError("Torch CUDA runtime identity is invalid")
    return dict(value)


def _atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    immutable: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    directory_fd, _parent = _open_safe_directory(
        path.parent,
        label=f"{path.name} parent",
    )
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        try:
            existing, _digest, _identity = _read_regular_at(
                directory_fd,
                path.name,
                label=f"existing {path.name}",
            )
        except ValueError:
            try:
                existing_stat = os.stat(
                    path.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                existing = None
            else:
                if existing_stat:
                    raise
        if immutable and existing is not None:
            if existing != payload:
                raise RuntimeError(f"immutable scheduler file changed: {path}")
            return
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=directory_fd,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("atomic scheduler write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        temporary_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(temporary_stat.st_mode)
            or temporary_stat.st_nlink != 1
            or temporary_stat.st_size != len(payload)
        ):
            raise RuntimeError("atomic scheduler temporary file is invalid")
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
        written_payload, _digest, _identity = _read_regular_at(
            directory_fd,
            path.name,
            label=f"published {path.name}",
        )
        if written_payload != payload:
            raise RuntimeError(f"atomic scheduler publication changed: {path}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, payload, immutable=False)


def _write_immutable_json(path: Path, value: Any) -> None:
    expected = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, expected, immutable=True)


def _closed_json(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if not key or key in output:
                raise ValueError(f"{path} contains an empty or colliding key")
            if _FORBIDDEN_WORKER_PARAMETER.search(key):
                raise ValueError(f"{path}.{key} is forbidden in a scope worker request")
            output[key] = _closed_json(child, path=f"{path}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [
            _closed_json(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, np.generic):
        return _closed_json(value.item(), path=path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    raise TypeError(f"{path} contains a non-JSON value")


def _integer_rows(values: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{label} must be a row-ID sequence")
    rows: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{label} must contain integer row IDs")
        row_id = int(value)
        if row_id < 0:
            raise ValueError(f"{label} contains a negative row ID")
        rows.append(row_id)
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"{label} must be nonempty and unique")
    return tuple(rows)


def _row_order_fingerprint(rows: Sequence[int]) -> str:
    return _sha256_json([int(value) for value in rows])


def derive_stage1_scope_seed(global_seed: int, scope_id: str) -> int:
    """Derive the legacy schedule-independent seed from a scope name.

    New portable production plans use :func:`derive_stage1_group_seed`, which
    is content-derived and therefore assigns one seed to equivalent logical
    contexts.  This legacy helper remains available only for authenticating
    artifacts produced under the older scope-name policy.
    """

    seed = int(global_seed)
    if seed < 0:
        raise ValueError("global Stage 1 seed must be nonnegative")
    scope = str(scope_id)
    if _SCOPE_ID.fullmatch(scope) is None:
        raise ValueError("scope_id is not canonical")
    digest = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": "production_stage1_scope_seed_v1",
                "global_seed": seed,
                "scope_id": scope,
            }
        ).encode("utf-8")
    ).digest()
    # Keep the result in the range accepted by NumPy's legacy RNG and common
    # sklearn random_state parameters.  Zero is valid but replacing it with one
    # makes accidental falsy checks harmless.
    result = int.from_bytes(digest[:8], "big") % (2**31 - 1)
    return result or 1


def derive_stage1_group_seed(
    global_seed: int,
    fit_row_ids: Sequence[int],
) -> int:
    """Derive one schedule/name-independent seed from ordered physical rows."""

    seed = int(global_seed)
    if seed < 0:
        raise ValueError("global Stage 1 seed must be nonnegative")
    rows = _integer_rows(fit_row_ids, label="physical-fit seed rows")
    digest = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": "production_stage1_group_seed_v2",
                "global_seed": seed,
                "ordered_fit_rows": list(rows),
            }
        ).encode("utf-8")
    ).digest()
    result = int.from_bytes(digest[:8], "big") % (2**31 - 1)
    return result or 1


def seed_stage1_scope_rngs(seed: int, *, gpu_id: int | None = None) -> None:
    """Reset Python, NumPy, and Torch RNGs at one isolated scope boundary."""

    resolved = int(seed)
    if not 0 <= resolved < 2**31:
        raise ValueError("scope seed must be a nonnegative 31-bit integer")
    random.seed(resolved)
    np.random.seed(resolved)
    try:
        import torch
    except ImportError:  # pragma: no cover - Torch is a production dependency.
        return
    # ``torch.manual_seed`` delegates to every visible accelerator. In a
    # one-process-per-GPU scheduler that can initialize or mutate RNG state on
    # a peer GPU. Seed the CPU generator directly, then seed only the selected
    # current CUDA device.
    torch.default_generator.manual_seed(resolved)
    if gpu_id is not None:
        selected = int(gpu_id)
        if selected < 0:
            raise ValueError("gpu_id must be nonnegative")
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA Stage 1 scope was assigned without CUDA")
        torch.cuda.set_device(selected)
        torch.cuda.manual_seed(resolved)


@dataclass(frozen=True)
class Stage1ScopeSpec:
    canonical_index: int
    scope_id: str
    scope_kind: str
    outer_fold: int
    inner_fold: int | None
    context_epoch: int | None
    provider_inner_fold: int | None
    fit_row_ids: tuple[int, ...]
    heldout_row_ids: tuple[int, ...]
    global_seed: int
    scope_seed: int

    @property
    def fit_row_count(self) -> int:
        return len(self.fit_row_ids)

    @property
    def heldout_row_count(self) -> int:
        return len(self.heldout_row_ids)

    def as_dict(self) -> dict[str, Any]:
        body = {
            "canonical_index": int(self.canonical_index),
            "scope_id": self.scope_id,
            "scope_kind": self.scope_kind,
            "outer_fold": int(self.outer_fold),
            "inner_fold": self.inner_fold,
            "context_epoch": self.context_epoch,
            "provider_inner_fold": self.provider_inner_fold,
            "fit_row_ids": list(self.fit_row_ids),
            "heldout_row_ids": list(self.heldout_row_ids),
            "fit_row_count": self.fit_row_count,
            "heldout_row_count": self.heldout_row_count,
            "fit_row_order_fingerprint": _row_order_fingerprint(self.fit_row_ids),
            "heldout_row_order_fingerprint": _row_order_fingerprint(
                self.heldout_row_ids
            ),
            "global_seed": int(self.global_seed),
            "scope_seed": int(self.scope_seed),
            "heldout_labels_supplied": False,
        }
        return {**body, "scope_sha256": _sha256_json(body)}


@dataclass(frozen=True)
class Stage1ScopeAssignment:
    scope_id: str
    gpu_id: int | None
    execution_rank: int
    fit_row_count: int
    assigned_gpu_load_after: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "gpu_id": self.gpu_id,
            "execution_rank": int(self.execution_rank),
            "fit_row_count": int(self.fit_row_count),
            "assigned_gpu_load_after": int(self.assigned_gpu_load_after),
        }


@dataclass(frozen=True)
class Stage1PhysicalFitIdentity:
    """Scientific axes shared by every physical fit in one scope plan.

    Deployment locators and scheduling choices are intentionally absent.
    These values must be supplied by the immutable workflow request; the
    scheduler never invents a producer, architecture, target, configuration,
    or runtime compatibility identity.
    """

    architecture_identity: str
    target: str
    scientific_configuration_identity: str
    producer_identity: str
    runtime_compatibility_class: str
    schema_version: str = STAGE1_PHYSICAL_FIT_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STAGE1_PHYSICAL_FIT_IDENTITY_SCHEMA:
            raise ValueError("unsupported Stage 1 physical-fit identity schema")
        for name in (
            "architecture_identity",
            "scientific_configuration_identity",
            "producer_identity",
        ):
            if _SHA256.fullmatch(str(getattr(self, name))) is None:
                raise ValueError(f"{name} must be one lowercase SHA-256")
        target = str(self.target).strip()
        runtime = str(self.runtime_compatibility_class).strip()
        if not target:
            raise ValueError("physical-fit target must be nonempty")
        if not runtime:
            raise ValueError(
                "physical-fit runtime compatibility class must be nonempty"
            )
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "runtime_compatibility_class", runtime)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "Stage1PhysicalFitIdentity":
        required = {
            "schema_version",
            "architecture_identity",
            "target",
            "scientific_configuration_identity",
            "producer_identity",
            "runtime_compatibility_class",
            "content_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                "Stage 1 physical-fit identity must be one closed record"
            )
        identity = cls(
            schema_version=str(value["schema_version"]),
            architecture_identity=str(value["architecture_identity"]),
            target=str(value["target"]),
            scientific_configuration_identity=str(
                value["scientific_configuration_identity"]
            ),
            producer_identity=str(value["producer_identity"]),
            runtime_compatibility_class=str(
                value["runtime_compatibility_class"]
            ),
        )
        if value["content_sha256"] != identity.content_sha256:
            raise ValueError("Stage 1 physical-fit identity content changed")
        return identity

    @property
    def content_sha256(self) -> str:
        return _sha256_json(
            {
                "schema_version": self.schema_version,
                "architecture_identity": self.architecture_identity,
                "target": self.target,
                "scientific_configuration_identity": (
                    self.scientific_configuration_identity
                ),
                "producer_identity": self.producer_identity,
                "runtime_compatibility_class": (
                    self.runtime_compatibility_class
                ),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": self.schema_version,
            "architecture_identity": self.architecture_identity,
            "target": self.target,
            "scientific_configuration_identity": (
                self.scientific_configuration_identity
            ),
            "producer_identity": self.producer_identity,
            "runtime_compatibility_class": (
                self.runtime_compatibility_class
            ),
        }
        return {**body, "content_sha256": self.content_sha256}

    def key_for_scope(self, scope: Stage1ScopeSpec) -> PhysicalFitKey:
        if not isinstance(scope, Stage1ScopeSpec):
            raise TypeError("physical-fit keys require a Stage1ScopeSpec")
        return PhysicalFitKey(
            architecture_identity=self.architecture_identity,
            target=self.target,
            fit_row_order_identity=ordered_row_identity(
                scope.fit_row_ids
            ),
            scientific_configuration_identity=(
                self.scientific_configuration_identity
            ),
            canonical_group_seed=int(scope.scope_seed),
            producer_identity=self.producer_identity,
            runtime_compatibility_class=self.runtime_compatibility_class,
        )


def _normalize_physical_fit_identity(
    value: Stage1PhysicalFitIdentity | Mapping[str, Any],
) -> Stage1PhysicalFitIdentity:
    if isinstance(value, Stage1PhysicalFitIdentity):
        # Round-trip its closed record so subclass/proxy behavior cannot enter
        # a scientific plan.
        return Stage1PhysicalFitIdentity.from_mapping(value.as_dict())
    return Stage1PhysicalFitIdentity.from_mapping(value)


def _physical_fit_groups(
    *,
    scopes: Sequence[Stage1ScopeSpec],
    physical_fit_identity: Stage1PhysicalFitIdentity,
) -> tuple[
    tuple[PhysicalFitKey, Stage1ScopeSpec, tuple[Stage1ScopeSpec, ...]],
    ...,
]:
    """Group exact scientific equivalents and select the earliest owner."""

    if not scopes:
        raise ValueError("Stage 1 scope plan cannot be empty")
    canonical_indices = tuple(int(scope.canonical_index) for scope in scopes)
    scope_ids = tuple(scope.scope_id for scope in scopes)
    if (
        len(canonical_indices) != len(set(canonical_indices))
        or len(scope_ids) != len(set(scope_ids))
    ):
        raise ValueError("Stage 1 scopes have duplicate identities")

    # Seed is validated inside, rather than used to split, a scientific
    # equivalence group. A changed canonical seed must never silently turn an
    # otherwise identical fit into a second accepted group.
    grouped: dict[
        tuple[str, int],
        list[Stage1ScopeSpec],
    ] = {}
    for scope in sorted(scopes, key=lambda value: value.canonical_index):
        row_identity = ordered_row_identity(scope.fit_row_ids)
        grouped.setdefault(
            (row_identity, len(scope.fit_row_ids)),
            [],
        ).append(scope)

    output: list[
        tuple[PhysicalFitKey, Stage1ScopeSpec, tuple[Stage1ScopeSpec, ...]]
    ] = []
    for members in grouped.values():
        owner = min(members, key=lambda value: value.canonical_index)
        owner_rows = tuple(owner.fit_row_ids)
        owner_seed = int(owner.scope_seed)
        key = physical_fit_identity.key_for_scope(owner)
        for member in members:
            if (
                tuple(member.fit_row_ids) != owner_rows
                or ordered_row_identity(member.fit_row_ids)
                != key.fit_row_order_identity
            ):
                raise RuntimeError(
                    "physical-fit row-order equivalence changed"
                )
            if int(member.scope_seed) != owner_seed:
                raise ValueError(
                    "ordered-equivalent logical scopes changed their "
                    "canonical group seed"
                )
            if physical_fit_identity.key_for_scope(member) != key:
                raise RuntimeError("physical-fit scientific key changed")
        output.append((key, owner, tuple(members)))
    return tuple(
        sorted(output, key=lambda value: value[1].canonical_index)
    )


def _stage1_scope_scientific_plan_body(
    *,
    registry_content_sha256: str,
    global_seed: int,
    review_rounds: int,
    initial_training_partitions: int,
    physical_fit_identity: Stage1PhysicalFitIdentity,
    scopes: tuple[Stage1ScopeSpec, ...],
) -> dict[str, Any]:
    """Return the fold/row/seed plan with no execution-resource metadata."""

    groups = _physical_fit_groups(
        scopes=scopes,
        physical_fit_identity=physical_fit_identity,
    )
    physical_scopes = tuple(owner for _key, owner, _members in groups)
    owner_by_scope = {
        member.scope_id: owner.scope_id
        for _key, owner, members in groups
        for member in members
    }
    key_by_scope = {
        member.scope_id: key
        for key, _owner, members in groups
        for member in members
    }
    return {
        "schema_version": STAGE1_SCOPE_SCIENTIFIC_PLAN_SCHEMA,
        "registry_content_sha256": registry_content_sha256,
        "global_seed": int(global_seed),
        "physical_fit_identity": physical_fit_identity.as_dict(),
        "scope_seed_derivation": (
            "sha256(global_seed,canonical_ordered_fit_rows)_31bit_v2"
        ),
        "review_rounds": int(review_rounds),
        "initial_training_partitions": int(initial_training_partitions),
        "canonical_scope_count": len(scopes),
        "logical_scope_count": len(scopes),
        "physical_scope_count": len(physical_scopes),
        "deduplicated_physical_fit_count": len(scopes) - len(physical_scopes),
        "canonical_scope_order": [scope.scope_id for scope in scopes],
        "physical_scope_order": [scope.scope_id for scope in physical_scopes],
        "physical_fit_groups": [
            {
                "physical_fit_key": key.key,
                "physical_fit_key_record": key.as_dict(),
                "canonical_owner_scope_id": owner.scope_id,
                "logical_scope_ids": [
                    member.scope_id for member in members
                ],
            }
            for key, owner, members in groups
        ],
        "logical_physical_bindings": [
            {
                "logical_scope_id": scope.scope_id,
                "physical_owner_scope_id": owner_by_scope[scope.scope_id],
                "physical_fit_key": key_by_scope[scope.scope_id].key,
                "reuses_physical_fit": (
                    owner_by_scope[scope.scope_id] != scope.scope_id
                ),
            }
            for scope in scopes
        ],
        "scopes": [scope.as_dict() for scope in scopes],
        "heldout_labels_present_in_scheduler_requests": False,
    }


@dataclass(frozen=True)
class Stage1ScopePlan:
    registry_content_sha256: str
    global_seed: int
    review_rounds: int
    initial_training_partitions: int
    physical_fit_identity: Stage1PhysicalFitIdentity
    gpu_ids: tuple[int, ...]
    scope_workers_per_gpu: int
    scopes: tuple[Stage1ScopeSpec, ...]
    assignments: tuple[Stage1ScopeAssignment, ...]
    content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "physical_fit_identity",
            _normalize_physical_fit_identity(self.physical_fit_identity),
        )

    @cached_property
    def scientific_content_sha256(self) -> str:
        return _sha256_json(
            _stage1_scope_scientific_plan_body(
                registry_content_sha256=self.registry_content_sha256,
                global_seed=self.global_seed,
                review_rounds=self.review_rounds,
                initial_training_partitions=self.initial_training_partitions,
                physical_fit_identity=self.physical_fit_identity,
                scopes=self.scopes,
            )
        )

    @cached_property
    def physical_fit_groups(
        self,
    ) -> tuple[
        tuple[
            PhysicalFitKey,
            Stage1ScopeSpec,
            tuple[Stage1ScopeSpec, ...],
        ],
        ...,
    ]:
        return _physical_fit_groups(
            scopes=self.scopes,
            physical_fit_identity=self.physical_fit_identity,
        )

    @property
    def physical_scope_groups(
        self,
    ) -> tuple[tuple[Stage1ScopeSpec, tuple[Stage1ScopeSpec, ...]], ...]:
        """Return content-derived groups in earliest-owner order.

        Logical purpose and held-out rows remain distinct. Physical
        equivalence requires the complete :class:`PhysicalFitKey`, including
        architecture, target, ordered rows, scientific configuration,
        canonical group seed, producer, and runtime compatibility class.
        """

        return tuple(
            (owner, members)
            for _key, owner, members in self.physical_fit_groups
        )

    @property
    def physical_scopes(self) -> tuple[Stage1ScopeSpec, ...]:
        return tuple(owner for owner, _members in self.physical_scope_groups)

    def physical_owner(self, scope_id: str) -> Stage1ScopeSpec:
        requested = self.scope(scope_id)
        for _key, owner, members in self.physical_fit_groups:
            if requested in members:
                return owner
        raise RuntimeError("logical scope has no physical owner")

    def physical_fit_key(self, scope_id: str) -> PhysicalFitKey:
        requested = self.scope(scope_id)
        for key, _owner, members in self.physical_fit_groups:
            if requested in members:
                return key
        raise RuntimeError("logical scope has no physical-fit key")

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": STAGE1_SCOPE_PLAN_SCHEMA,
            "scientific_content_sha256": self.scientific_content_sha256,
            "registry_content_sha256": self.registry_content_sha256,
            "global_seed": int(self.global_seed),
            "physical_fit_identity": self.physical_fit_identity.as_dict(),
            "scope_seed_derivation": (
                "sha256(global_seed,canonical_ordered_fit_rows)_31bit_v2"
            ),
            "python_hash_seed_policy": (
                "set_scope_seed_immediately_before_spawn_then_restore_parent_v1"
            ),
            "review_rounds": int(self.review_rounds),
            "initial_training_partitions": int(
                self.initial_training_partitions
            ),
            "gpu_ids": list(self.gpu_ids),
            "scope_workers_per_gpu": int(self.scope_workers_per_gpu),
            "scheduling_policy": (
                "largest_fit_row_count_first_then_canonical_index_"
                "least_loaded_gpu_then_gpu_id_v1"
            ),
            "spawn_start_method": "spawn",
            "worker_process_group_policy": (
                "new_posix_session_per_scope_terminate_group_on_abort_v1"
            ),
            "native_threads_per_scope": 1,
            "nested_native_parallelism_allowed": False,
            "torch_determinism_policy": stage1_torch_determinism_policy(),
            "maximum_concurrent_scopes_per_gpu": int(
                self.scope_workers_per_gpu
            ),
            "canonical_scope_count": len(self.scopes),
            "logical_scope_count": len(self.scopes),
            "physical_scope_count": len(self.physical_scopes),
            "deduplicated_physical_fit_count": (
                len(self.scopes) - len(self.physical_scopes)
            ),
            "full_outer_scope_count": sum(
                scope.scope_kind == "full_outer" for scope in self.scopes
            ),
            "exact_inner_scope_count": sum(
                scope.scope_kind == "exact_inner" for scope in self.scopes
            ),
            "cumulative_spent_scope_count": sum(
                scope.scope_kind == "cumulative_spent" for scope in self.scopes
            ),
            "canonical_scope_order": [scope.scope_id for scope in self.scopes],
            "physical_scope_order": [
                scope.scope_id for scope in self.physical_scopes
            ],
            "physical_fit_groups": [
                {
                    "physical_fit_key": key.key,
                    "physical_fit_key_record": key.as_dict(),
                    "canonical_owner_scope_id": owner.scope_id,
                    "logical_scope_ids": [
                        member.scope_id for member in members
                    ],
                }
                for key, owner, members in self.physical_fit_groups
            ],
            "logical_physical_bindings": [
                {
                    "logical_scope_id": scope.scope_id,
                    "physical_owner_scope_id": self.physical_owner(
                        scope.scope_id
                    ).scope_id,
                    "physical_fit_key": self.physical_fit_key(
                        scope.scope_id
                    ).key,
                    "reuses_physical_fit": (
                        self.physical_owner(scope.scope_id).scope_id
                        != scope.scope_id
                    ),
                }
                for scope in self.scopes
            ],
            "scopes": [scope.as_dict() for scope in self.scopes],
            "assignments": [
                assignment.as_dict() for assignment in self.assignments
            ],
            "heldout_labels_present_in_scheduler_requests": False,
        }
        if _sha256_json(body) != self.content_sha256:
            raise RuntimeError("Stage 1 scope plan changed after construction")
        return {**body, "content_sha256": self.content_sha256}

    def scope(self, scope_id: str) -> Stage1ScopeSpec:
        for scope in self.scopes:
            if scope.scope_id == str(scope_id):
                return scope
        raise ValueError(f"unknown Stage 1 scope: {scope_id}")

    def assignment(self, scope_id: str) -> Stage1ScopeAssignment:
        for assignment in self.assignments:
            if assignment.scope_id == str(scope_id):
                return assignment
        raise ValueError(f"unassigned Stage 1 scope: {scope_id}")

    @property
    def execution_order(self) -> tuple[str, ...]:
        return tuple(
            assignment.scope_id
            for assignment in sorted(
                self.assignments, key=lambda item: item.execution_rank
            )
        )

    @property
    def physical_execution_order(self) -> tuple[str, ...]:
        """Return only canonical owners in their precomputed execution order."""

        owners = {scope.scope_id for scope in self.physical_scopes}
        return tuple(
            scope_id for scope_id in self.execution_order if scope_id in owners
        )


def _stage1_scope_plan_body(
    *,
    registry_content_sha256: str,
    global_seed: int,
    review_rounds: int,
    initial_training_partitions: int,
    physical_fit_identity: Stage1PhysicalFitIdentity,
    gpu_ids: tuple[int, ...],
    scope_workers_per_gpu: int,
    scopes: tuple[Stage1ScopeSpec, ...],
    assignments: tuple[Stage1ScopeAssignment, ...],
) -> dict[str, Any]:
    # Mirror ``as_dict`` without invoking its integrity check.
    groups = _physical_fit_groups(
        scopes=scopes,
        physical_fit_identity=physical_fit_identity,
    )
    physical_scopes = tuple(owner for _key, owner, _members in groups)
    owner_by_scope = {
        member.scope_id: owner.scope_id
        for _key, owner, members in groups
        for member in members
    }
    key_by_scope = {
        member.scope_id: key
        for key, _owner, members in groups
        for member in members
    }
    return {
        "schema_version": STAGE1_SCOPE_PLAN_SCHEMA,
        "scientific_content_sha256": _sha256_json(
            _stage1_scope_scientific_plan_body(
                registry_content_sha256=registry_content_sha256,
                global_seed=global_seed,
                review_rounds=review_rounds,
                initial_training_partitions=initial_training_partitions,
                physical_fit_identity=physical_fit_identity,
                scopes=scopes,
            )
        ),
        "registry_content_sha256": registry_content_sha256,
        "global_seed": int(global_seed),
        "physical_fit_identity": physical_fit_identity.as_dict(),
        "scope_seed_derivation": (
            "sha256(global_seed,canonical_ordered_fit_rows)_31bit_v2"
        ),
        "python_hash_seed_policy": (
            "set_scope_seed_immediately_before_spawn_then_restore_parent_v1"
        ),
        "review_rounds": int(review_rounds),
        "initial_training_partitions": int(initial_training_partitions),
        "gpu_ids": list(gpu_ids),
        "scope_workers_per_gpu": int(scope_workers_per_gpu),
        "scheduling_policy": (
            "largest_fit_row_count_first_then_canonical_index_"
            "least_loaded_gpu_then_gpu_id_v1"
        ),
        "spawn_start_method": "spawn",
        "worker_process_group_policy": (
            "new_posix_session_per_scope_terminate_group_on_abort_v1"
        ),
        "native_threads_per_scope": 1,
        "nested_native_parallelism_allowed": False,
        "torch_determinism_policy": stage1_torch_determinism_policy(),
        "maximum_concurrent_scopes_per_gpu": int(scope_workers_per_gpu),
        "canonical_scope_count": len(scopes),
        "logical_scope_count": len(scopes),
        "physical_scope_count": len(physical_scopes),
        "deduplicated_physical_fit_count": len(scopes) - len(physical_scopes),
        "full_outer_scope_count": sum(
            scope.scope_kind == "full_outer" for scope in scopes
        ),
        "exact_inner_scope_count": sum(
            scope.scope_kind == "exact_inner" for scope in scopes
        ),
        "cumulative_spent_scope_count": sum(
            scope.scope_kind == "cumulative_spent" for scope in scopes
        ),
        "canonical_scope_order": [scope.scope_id for scope in scopes],
        "physical_scope_order": [scope.scope_id for scope in physical_scopes],
        "physical_fit_groups": [
            {
                "physical_fit_key": key.key,
                "physical_fit_key_record": key.as_dict(),
                "canonical_owner_scope_id": owner.scope_id,
                "logical_scope_ids": [
                    member.scope_id for member in members
                ],
            }
            for key, owner, members in groups
        ],
        "logical_physical_bindings": [
            {
                "logical_scope_id": scope.scope_id,
                "physical_owner_scope_id": owner_by_scope[scope.scope_id],
                "physical_fit_key": key_by_scope[scope.scope_id].key,
                "reuses_physical_fit": (
                    owner_by_scope[scope.scope_id] != scope.scope_id
                ),
            }
            for scope in scopes
        ],
        "scopes": [scope.as_dict() for scope in scopes],
        "assignments": [assignment.as_dict() for assignment in assignments],
        "heldout_labels_present_in_scheduler_requests": False,
    }


def build_canonical_stage1_scope_plan(
    *,
    registry: Mapping[str, Any],
    registry_content_sha256: str,
    global_seed: int,
    physical_fit_identity: Stage1PhysicalFitIdentity | Mapping[str, Any],
    gpu_ids: Sequence[int] = (),
    review_rounds: int,
    initial_training_partitions: int,
    scope_workers_per_gpu: int = 1,
    expected_outer_fold_count: int | None = None,
    expected_inner_fold_count: int | None = None,
) -> Stage1ScopePlan:
    """Build the exact full/inner/cumulative task graph from one registry."""

    resolved_physical_fit_identity = _normalize_physical_fit_identity(
        physical_fit_identity
    )
    if (
        not isinstance(registry_content_sha256, str)
        or _SHA256.fullmatch(registry_content_sha256) is None
    ):
        raise ValueError("registry_content_sha256 must be one lowercase SHA-256")
    if not isinstance(registry, Mapping):
        raise TypeError("registry must be a mapping")
    outer_rows = registry.get("outer_folds")
    if not isinstance(outer_rows, list) or not outer_rows:
        raise ValueError("registry has no canonical outer folds")
    rounds = int(review_rounds)
    if rounds < 1:
        raise ValueError("review_rounds must be positive")
    initial_partitions = int(initial_training_partitions)
    if initial_partitions < 1:
        raise ValueError("initial_training_partitions must be positive")
    workers_per_gpu = int(scope_workers_per_gpu)
    if (
        isinstance(scope_workers_per_gpu, bool)
        or workers_per_gpu < 1
        or workers_per_gpu != scope_workers_per_gpu
    ):
        raise ValueError("scope_workers_per_gpu must be a positive integer")
    resolved_gpus = tuple(int(value) for value in gpu_ids)
    if any(value < 0 for value in resolved_gpus) or len(resolved_gpus) != len(
        set(resolved_gpus)
    ):
        raise ValueError("gpu_ids must be unique nonnegative integers")
    if expected_outer_fold_count is not None and len(outer_rows) != int(
        expected_outer_fold_count
    ):
        raise ValueError("registry outer-fold count differs from the requested plan")
    dataset_row_count = int(registry.get("dataset_row_count", -1))
    if dataset_row_count < 1:
        raise ValueError("registry has no positive dataset row count")
    dataset_rows = set(range(dataset_row_count))
    expected_outer_ids = tuple(range(1, len(outer_rows) + 1))
    observed_outer_ids = tuple(int(row.get("outer_fold", -1)) for row in outer_rows)
    if observed_outer_ids != expected_outer_ids:
        raise ValueError("registry outer folds are missing, duplicated, or reordered")

    specs: list[Stage1ScopeSpec] = []
    cumulative_inputs: list[
        tuple[int, tuple[int, ...], dict[int, tuple[int, ...]]]
    ] = []
    outer_heldout_counts: dict[int, int] = {}

    def add_scope(
        *,
        scope_id: str,
        scope_kind: str,
        outer_fold: int,
        inner_fold: int | None,
        context_epoch: int | None,
        provider_inner_fold: int | None,
        fit_rows: tuple[int, ...],
        heldout_rows: tuple[int, ...],
    ) -> None:
        if _SCOPE_ID.fullmatch(scope_id) is None:
            raise ValueError(f"generated noncanonical scope ID: {scope_id}")
        if set(fit_rows) & set(heldout_rows):
            raise ValueError(f"{scope_id} fit and held-out rows overlap")
        specs.append(
            Stage1ScopeSpec(
                canonical_index=len(specs),
                scope_id=scope_id,
                scope_kind=scope_kind,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                context_epoch=context_epoch,
                provider_inner_fold=provider_inner_fold,
                fit_row_ids=fit_rows,
                heldout_row_ids=heldout_rows,
                global_seed=int(global_seed),
                scope_seed=derive_stage1_group_seed(global_seed, fit_rows),
            )
        )

    for outer in outer_rows:
        outer_fold = int(outer["outer_fold"])
        outer_fit = _integer_rows(
            outer.get("fit_row_ids"), label=f"outer {outer_fold} fit rows"
        )
        outer_heldout = _integer_rows(
            outer.get("heldout_row_ids"),
            label=f"outer {outer_fold} held-out rows",
        )
        if set(outer_fit) & set(outer_heldout):
            raise ValueError(f"outer {outer_fold} fit and held-out rows overlap")
        if set(outer_fit) | set(outer_heldout) != dataset_rows:
            raise ValueError(f"outer {outer_fold} does not partition the cohort")
        for row_id in outer_heldout:
            outer_heldout_counts[row_id] = outer_heldout_counts.get(row_id, 0) + 1
        add_scope(
            scope_id=f"outer_{outer_fold:03d}_full",
            scope_kind="full_outer",
            outer_fold=outer_fold,
            inner_fold=None,
            context_epoch=None,
            provider_inner_fold=None,
            fit_rows=outer_fit,
            heldout_rows=outer_heldout,
        )
        inner_rows = outer.get("inner_folds")
        if (
            not isinstance(inner_rows, list)
            or len(inner_rows) != rounds + initial_partitions
        ):
            raise ValueError(
                f"outer {outer_fold} must have review_rounds + "
                "initial_training_partitions inner folds"
            )
        if expected_inner_fold_count is not None and len(inner_rows) != int(
            expected_inner_fold_count
        ):
            raise ValueError("registry inner-fold count differs from the requested plan")
        by_partition: dict[int, tuple[int, ...]] = {}
        for expected_inner, inner in enumerate(inner_rows, start=1):
            if int(inner.get("inner_fold", -1)) != expected_inner:
                raise ValueError(
                    f"outer {outer_fold} inner folds are missing or reordered"
                )
            inner_fit = _integer_rows(
                inner.get("fit_row_ids"),
                label=f"outer {outer_fold} inner {expected_inner} fit rows",
            )
            inner_heldout = _integer_rows(
                inner.get("heldout_row_ids"),
                label=f"outer {outer_fold} inner {expected_inner} held-out rows",
            )
            if (
                set(inner_fit) & set(inner_heldout)
                or set(inner_fit) | set(inner_heldout) != set(outer_fit)
            ):
                raise ValueError(
                    f"outer {outer_fold} inner {expected_inner} does not partition outer fit"
                )
            by_partition[expected_inner] = inner_heldout
            add_scope(
                scope_id=f"outer_{outer_fold:03d}_inner_{expected_inner:03d}",
                scope_kind="exact_inner",
                outer_fold=outer_fold,
                inner_fold=expected_inner,
                context_epoch=None,
                provider_inner_fold=None,
                fit_rows=inner_fit,
                heldout_rows=inner_heldout,
            )
        flattened = tuple(
            row_id
            for partition_id in range(1, len(inner_rows) + 1)
            for row_id in by_partition[partition_id]
        )
        if len(flattened) != len(set(flattened)) or set(flattened) != set(
            outer_fit
        ):
            raise ValueError(
                f"outer {outer_fold} inner held-outs do not partition outer fit"
            )
        cumulative_inputs.append((outer_fold, outer_fit, by_partition))

    if set(outer_heldout_counts) != dataset_rows or set(
        outer_heldout_counts.values()
    ) != {1}:
        raise ValueError("outer held-outs do not cover the cohort exactly once")

    # Preserve the scientific order already used by clustered-embedding
    # preflight: all full/exact-inner scopes first, then cumulative scopes.
    for outer_fold, outer_fit_order, by_partition in cumulative_inputs:
        partition_ids = tuple(sorted(by_partition))
        for epoch in range(rounds):
            gate = initial_partitions + epoch
            spent_partition_ids = partition_ids[:gate]
            sealed_partition_ids = partition_ids[gate:]
            spent_rows = {
                row_id
                for partition_id in spent_partition_ids
                for row_id in by_partition[partition_id]
            }
            sealed_rows = {
                row_id
                for partition_id in sealed_partition_ids
                for row_id in by_partition[partition_id]
            }
            # A cumulative purpose inherits the canonical outer-fit row order
            # rather than the incidental order in which partition blocks are
            # concatenated.  This makes an exact-inner/cumulative alias a
            # physical-fit equivalence only when their ordered fit inputs are
            # genuinely identical.
            fit_rows = tuple(
                row_id
                for row_id in outer_fit_order
                if row_id in spent_rows
            )
            heldout_rows = tuple(
                row_id
                for row_id in outer_fit_order
                if row_id in sealed_rows
            )
            add_scope(
                scope_id=(
                    f"outer_{outer_fold:03d}_hierarchy_epoch_{epoch:03d}"
                ),
                scope_kind="cumulative_spent",
                outer_fold=outer_fold,
                inner_fold=None,
                context_epoch=epoch,
                provider_inner_fold=epoch + 1,
                fit_rows=fit_rows,
                heldout_rows=heldout_rows,
            )

    if len({scope.scope_id for scope in specs}) != len(specs):
        raise RuntimeError("canonical Stage 1 plan contains duplicate scopes")
    seeds_by_fit_order: dict[tuple[int, ...], int] = {}
    fit_order_by_seed: dict[int, tuple[int, ...]] = {}
    for scope in specs:
        fit_order = tuple(scope.fit_row_ids)
        prior_seed = seeds_by_fit_order.setdefault(fit_order, scope.scope_seed)
        if prior_seed != scope.scope_seed:
            raise RuntimeError(
                "ordered-equivalent Stage 1 scopes received different seeds"
            )
        prior_fit_order = fit_order_by_seed.setdefault(scope.scope_seed, fit_order)
        if prior_fit_order != fit_order:
            raise RuntimeError("Stage 1 physical-fit seed derivation collided")

    execution_specs = sorted(
        specs, key=lambda scope: (-scope.fit_row_count, scope.canonical_index)
    )
    assignments: list[Stage1ScopeAssignment] = []
    loads: dict[int | None, int]
    if resolved_gpus:
        loads = {gpu_id: 0 for gpu_id in resolved_gpus}
    else:
        loads = {None: 0}
    for execution_rank, scope in enumerate(execution_specs):
        selected = min(
            loads,
            key=lambda gpu_id: (
                loads[gpu_id],
                -1 if gpu_id is None else int(gpu_id),
            ),
        )
        loads[selected] += scope.fit_row_count
        assignments.append(
            Stage1ScopeAssignment(
                scope_id=scope.scope_id,
                gpu_id=selected,
                execution_rank=execution_rank,
                fit_row_count=scope.fit_row_count,
                assigned_gpu_load_after=loads[selected],
            )
        )
    assignment_by_scope = {row.scope_id: row for row in assignments}
    canonical_assignments = tuple(
        assignment_by_scope[scope.scope_id] for scope in specs
    )
    scope_tuple = tuple(specs)
    body = _stage1_scope_plan_body(
        registry_content_sha256=registry_content_sha256,
        global_seed=int(global_seed),
        review_rounds=rounds,
        initial_training_partitions=initial_partitions,
        physical_fit_identity=resolved_physical_fit_identity,
        gpu_ids=resolved_gpus,
        scope_workers_per_gpu=workers_per_gpu,
        scopes=scope_tuple,
        assignments=canonical_assignments,
    )
    return Stage1ScopePlan(
        registry_content_sha256=registry_content_sha256,
        global_seed=int(global_seed),
        review_rounds=rounds,
        initial_training_partitions=initial_partitions,
        physical_fit_identity=resolved_physical_fit_identity,
        gpu_ids=resolved_gpus,
        scope_workers_per_gpu=workers_per_gpu,
        scopes=scope_tuple,
        assignments=canonical_assignments,
        content_sha256=_sha256_json(body),
    )


def validate_stage1_scope_plan(
    value: Any,
    *,
    registry: Mapping[str, Any],
    registry_content_sha256: str,
    global_seed: int,
    physical_fit_identity: Stage1PhysicalFitIdentity | Mapping[str, Any],
    gpu_ids: Sequence[int] = (),
    review_rounds: int,
    initial_training_partitions: int,
    scope_workers_per_gpu: int = 1,
    expected_outer_fold_count: int | None = None,
    expected_inner_fold_count: int | None = None,
) -> Stage1ScopePlan:
    """Rebuild and byte-for-byte compare a persisted canonical scope plan."""

    if not isinstance(value, Mapping):
        raise TypeError("persisted Stage 1 scope plan must be a mapping")
    expected = build_canonical_stage1_scope_plan(
        registry=registry,
        registry_content_sha256=registry_content_sha256,
        global_seed=global_seed,
        physical_fit_identity=physical_fit_identity,
        gpu_ids=gpu_ids,
        review_rounds=review_rounds,
        initial_training_partitions=initial_training_partitions,
        scope_workers_per_gpu=scope_workers_per_gpu,
        expected_outer_fold_count=expected_outer_fold_count,
        expected_inner_fold_count=expected_inner_fold_count,
    )
    if dict(value) != expected.as_dict():
        raise ValueError("persisted Stage 1 scope plan changed or was substituted")
    return expected


def write_stage1_scope_plan(path: Path | str, plan: Stage1ScopePlan) -> None:
    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError("plan must be a Stage1ScopePlan")
    _write_immutable_json(Path(path), plan.as_dict())


@dataclass(frozen=True)
class Stage1ScopeExecutionRequest:
    attempt_dir: str
    plan_content_sha256: str
    scope: Mapping[str, Any]
    assignment: Mapping[str, Any]
    worker_target: str
    worker_parameters: Mapping[str, Any]
    worker_parameters_sha256: str
    attempt_request_sha256: str

    @property
    def scope_id(self) -> str:
        return str(self.scope["scope_id"])

    @property
    def scope_seed(self) -> int:
        return int(self.scope["scope_seed"])

    @property
    def gpu_id(self) -> int | None:
        value = self.assignment.get("gpu_id")
        return None if value is None else int(value)

    @property
    def payload_dir(self) -> Path:
        return Path(self.attempt_dir) / "payload"


@dataclass(frozen=True)
class ValidatedStage1ScopeAttempt:
    """One authenticated terminal attempt plus its canonical local directory."""

    scope_id: str
    attempt_dir: Path
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class ValidatedStage1LogicalScopeBindings:
    """Authenticated 40-logical-to-physical-attempt reference set."""

    path: Path
    manifest: Mapping[str, Any]

    @property
    def bindings(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            copy.deepcopy(dict(row))
            for row in self.manifest["logical_bindings"]
        )


def _attempt_inventory(
    attempt_dir: Path,
    *,
    expected_attempt_binding: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Stream a tree inventory through one continuously held root descriptor."""

    root_descriptor, root = _open_safe_directory(
        attempt_dir,
        label="scope attempt",
    )
    root_before = os.fstat(root_descriptor)
    if expected_attempt_binding is not None:
        expected = dict(expected_attempt_binding)
        if (
            expected.get("absolute_path") != str(root)
            or int(expected.get("device", -1)) != int(root_before.st_dev)
            or int(expected.get("inode", -1)) != int(root_before.st_ino)
        ):
            os.close(root_descriptor)
            raise ValueError("scope attempt directory was atomically substituted")
    inventory: list[dict[str, Any]] = []

    def walk(directory_fd: int, relative_parts: tuple[str, ...]) -> None:
        directory_before = os.fstat(directory_fd)
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise ValueError("scope attempt changed during inventory") from exc
        for name in names:
            if not name or "/" in name or name in {".", ".."}:
                raise ValueError("scope attempt contains an unsafe entry name")
            try:
                metadata = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ValueError(
                    "scope attempt changed during inventory"
                ) from exc
            relative = "/".join((*relative_parts, name))
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("scope attempts cannot contain symlinks")
            if stat.S_ISDIR(metadata.st_mode):
                flags = (
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                try:
                    child_fd = os.open(name, flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise ValueError(
                        "scope attempt directory changed during inventory"
                    ) from exc
                try:
                    opened = os.fstat(child_fd)
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or (opened.st_dev, opened.st_ino)
                        != (metadata.st_dev, metadata.st_ino)
                    ):
                        raise ValueError(
                            "scope attempt directory was substituted during inventory"
                        )
                    walk(child_fd, (*relative_parts, name))
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("scope attempts cannot contain special files")
            if relative == "attempt_manifest.json":
                continue
            size, digest, opened_identity = _hash_regular_at(
                directory_fd,
                name,
                label=f"scope attempt file {relative}",
            )
            if (
                opened_identity[0],
                opened_identity[1],
            ) != (int(metadata.st_dev), int(metadata.st_ino)):
                raise ValueError(
                    "scope attempt file was substituted during inventory"
                )
            inventory.append(
                {
                    "relative_path": relative,
                    "size": size,
                    "sha256": digest,
                }
            )
        directory_after = os.fstat(directory_fd)
        if _stable_stat_identity(directory_before) != _stable_stat_identity(
            directory_after
        ):
            raise ValueError("scope attempt directory changed during inventory")

    try:
        walk(root_descriptor, ())
        root_after = os.fstat(root_descriptor)
        if _stable_stat_identity(root_before) != _stable_stat_identity(
            root_after
        ):
            raise ValueError("scope attempt root changed during inventory")
        if (
            expected_attempt_binding is not None
            and _directory_inode_binding(
                root,
                label="scope attempt post-inventory",
            )
            != dict(expected_attempt_binding)
        ):
            raise ValueError(
                "scope attempt path was atomically substituted during inventory"
            )
        return sorted(inventory, key=lambda row: str(row["relative_path"]))
    finally:
        os.close(root_descriptor)


def _durably_sync_attempt_tree(attempt_dir: Path) -> None:
    """Fsync every scope-owned file and directory before terminal sealing."""

    root_fd, root = _open_safe_directory(
        attempt_dir,
        label="scope attempt durability root",
    )
    os.close(root_fd)
    directories = [root]
    files: list[Path] = []
    for path in root.rglob("*"):
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("scope attempt durability tree contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(path)
        elif stat.S_ISREG(metadata.st_mode):
            files.append(path)
        else:
            raise ValueError("scope attempt durability tree contains a special file")
    for path in sorted(files):
        directory_fd, _parent = _open_safe_directory(
            path.parent,
            label=f"durability parent for {path.name}",
        )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor: int | None = None
        try:
            descriptor = os.open(path.name, flags, dir_fd=directory_fd)
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError(
                    "scope attempt durability file is linked or non-regular"
                )
            os.fsync(descriptor)
            after = os.fstat(descriptor)
            if _stable_stat_identity(before) != _stable_stat_identity(after):
                raise ValueError(
                    "scope attempt file changed during durability synchronization"
                )
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(directory_fd)
    for path in sorted(
        directories,
        key=lambda item: len(item.relative_to(root).parts),
        reverse=True,
    ):
        descriptor, _opened = _open_safe_directory(
            path,
            label="scope attempt durability directory",
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _durably_sync_attempt_parent_chain(attempt_dir: Path) -> None:
    """Persist the attempt-directory entry before publishing its terminal marker."""

    absolute = _absolute_path(attempt_dir)
    # The scope directory owns the attempt entry; the attempt-store directory
    # owns the scope entry. Its parent owns the attempt-store entry. Sync all
    # three bottom-up so a completed manifest cannot outlive its path after a
    # host crash.
    for path in (
        absolute.parent,
        absolute.parent.parent,
        absolute.parent.parent.parent,
    ):
        descriptor, _opened = _open_safe_directory(
            path,
            label="scope attempt durability parent",
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class Stage1ScopeAttemptStore:
    """Create, seal, and authenticate immutable per-scope attempts."""

    def __init__(self, root: Path | str, plan: Stage1ScopePlan) -> None:
        if not isinstance(plan, Stage1ScopePlan):
            raise TypeError("plan must be a Stage1ScopePlan")
        self.root = _absolute_path(root)
        self.plan = plan
        if self.root.is_symlink():
            raise ValueError("scope-attempt root cannot be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self._root_inode_binding = _directory_inode_binding(
            self.root,
            label="scope-attempt root",
        )

    def identity(self) -> dict[str, Any]:
        _validate_directory_inode_binding(
            self._root_inode_binding,
            path=self.root,
            label="scope-attempt root",
        )
        body = {
            "schema_version": STAGE1_ATTEMPT_STORE_SCHEMA,
            "plan_content_sha256": self.plan.content_sha256,
            "root": str(_absolute_path(self.root)),
            "root_inode_binding": dict(self._root_inode_binding),
        }
        return {**body, "content_sha256": _sha256_json(body)}

    def _fresh_attempt_capability(
        self,
        *,
        scope_id: str,
        attempt_dir: Path | str,
    ) -> ValidatedStage1ScopeAttempt:
        """Reopen one attempt using only its closed, persisted request."""

        path = self._validated_attempt_path(attempt_dir, scope_id=scope_id)
        request = _load_strict_json_file(
            path / "attempt_request.json",
            label=f"{scope_id} logical-binding attempt request",
        )
        if not isinstance(request, Mapping):
            raise ValueError("logical-binding attempt request is not an object")
        target = request.get("worker_target")
        parameters = request.get("worker_parameters")
        if (
            not isinstance(target, str)
            or _WORKER_TARGET.fullmatch(target) is None
            or not isinstance(parameters, Mapping)
        ):
            raise ValueError("logical-binding attempt capability is malformed")
        manifest = self.validate_completed(
            path,
            scope_id=scope_id,
            worker_target=target,
            worker_parameters=parameters,
        )
        return ValidatedStage1ScopeAttempt(
            scope_id=scope_id,
            attempt_dir=path,
            manifest=manifest,
        )

    def _logical_binding_manifest(
        self,
        physical_attempts: Sequence[ValidatedStage1ScopeAttempt],
    ) -> dict[str, Any]:
        by_scope: dict[str, ValidatedStage1ScopeAttempt] = {}
        for supplied in physical_attempts:
            if not isinstance(supplied, ValidatedStage1ScopeAttempt):
                raise TypeError(
                    "physical attempts must be authenticated Stage 1 attempts"
                )
            if supplied.scope_id in by_scope:
                raise ValueError("physical attempt owner is duplicated")
            by_scope[supplied.scope_id] = self._fresh_attempt_capability(
                scope_id=supplied.scope_id,
                attempt_dir=supplied.attempt_dir,
            )
        expected_owners = tuple(
            scope.scope_id for scope in self.plan.physical_scopes
        )
        if set(by_scope) != set(expected_owners):
            raise ValueError(
                "physical attempts do not cover exactly the canonical owners"
            )

        physical_rows: list[dict[str, Any]] = []
        for owner_id in expected_owners:
            attempt = by_scope[owner_id]
            manifest = dict(attempt.manifest)
            physical_key = self.plan.physical_fit_key(owner_id)
            relative = attempt.attempt_dir.relative_to(self.root).as_posix()
            if relative != f"{owner_id}/{attempt.attempt_dir.name}":
                raise ValueError("physical attempt capability is noncanonical")
            physical_rows.append(
                {
                    "physical_owner_scope_id": owner_id,
                    "physical_fit_key": physical_key.key,
                    "physical_fit_key_record": physical_key.as_dict(),
                    "physical_attempt_relative_path": relative,
                    "attempt_request_sha256": manifest[
                        "attempt_request_sha256"
                    ],
                    "attempt_manifest_content_sha256": manifest[
                        "content_sha256"
                    ],
                }
            )
        physical_by_owner = {
            row["physical_owner_scope_id"]: row for row in physical_rows
        }

        logical_rows: list[dict[str, Any]] = []
        for logical in self.plan.scopes:
            owner = self.plan.physical_owner(logical.scope_id)
            physical_key = self.plan.physical_fit_key(logical.scope_id)
            if (
                tuple(logical.fit_row_ids) != tuple(owner.fit_row_ids)
                or logical.scope_seed != owner.scope_seed
                or physical_key
                != self.plan.physical_fit_key(owner.scope_id)
            ):
                raise RuntimeError(
                    "logical scope no longer matches its physical-fit owner"
                )
            body = {
                "schema_version": STAGE1_LOGICAL_SCOPE_BINDING_SCHEMA,
                "plan_content_sha256": self.plan.content_sha256,
                "logical_scope_id": logical.scope_id,
                "logical_scope_sha256": logical.as_dict()["scope_sha256"],
                "logical_purpose": logical.scope_kind,
                "physical_owner_scope_id": owner.scope_id,
                "physical_owner_scope_sha256": owner.as_dict()[
                    "scope_sha256"
                ],
                **physical_by_owner[owner.scope_id],
                "physical_fit_identity": (
                    self.plan.physical_fit_identity.as_dict()
                ),
                "fit_row_order_fingerprint": _row_order_fingerprint(
                    logical.fit_row_ids
                ),
                "fit_row_order_identity": (
                    physical_key.fit_row_order_identity
                ),
                "canonical_group_seed": int(owner.scope_seed),
                "reuses_physical_fit": logical.scope_id != owner.scope_id,
                "heldout_labels_supplied_to_physical_worker": False,
            }
            logical_rows.append(
                {**body, "content_sha256": _sha256_json(body)}
            )
        top_body = {
            "schema_version": STAGE1_LOGICAL_SCOPE_BINDING_SET_SCHEMA,
            "plan_content_sha256": self.plan.content_sha256,
            "canonical_logical_scope_order": [
                scope.scope_id for scope in self.plan.scopes
            ],
            "physical_owner_scope_order": list(expected_owners),
            "logical_scope_count": len(logical_rows),
            "physical_fit_count": len(physical_rows),
            "deduplicated_fit_count": len(logical_rows) - len(physical_rows),
            "physical_attempts": physical_rows,
            "logical_bindings": logical_rows,
            "physical_fit_identity": (
                self.plan.physical_fit_identity.as_dict()
            ),
            "fit_equivalence_proven_from_complete_physical_fit_key": True,
            "evidence_family_equality_proof_location": (
                "downstream_sealed_stage1_handoff"
            ),
            "heldout_labels_supplied_to_physical_workers": False,
        }
        return {**top_body, "content_sha256": _sha256_json(top_body)}

    def seal_logical_bindings(
        self,
        physical_attempts: Sequence[ValidatedStage1ScopeAttempt],
    ) -> ValidatedStage1LogicalScopeBindings:
        """Publish logical references only after every physical owner seals."""

        expected = self._logical_binding_manifest(physical_attempts)
        path = self.root / STAGE1_LOGICAL_SCOPE_BINDING_FILENAME
        _write_immutable_json(path, expected)
        descriptor, _root = _open_safe_directory(
            self.root,
            label="scope-attempt root for logical-binding durability",
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return self.validate_logical_bindings()

    def validate_logical_bindings(
        self,
    ) -> ValidatedStage1LogicalScopeBindings:
        """Freshly reopen all owner attempts named by the logical references."""

        path = self.root / STAGE1_LOGICAL_SCOPE_BINDING_FILENAME
        manifest = _load_strict_json_file(
            path,
            label="Stage 1 logical-scope binding set",
        )
        if not isinstance(manifest, Mapping):
            raise ValueError("logical-scope binding set is not an object")
        body = dict(manifest)
        declared = body.pop("content_sha256", None)
        physical_rows = manifest.get("physical_attempts")
        if (
            manifest.get("schema_version")
            != STAGE1_LOGICAL_SCOPE_BINDING_SET_SCHEMA
            or _SHA256.fullmatch(str(declared or "")) is None
            or _sha256_json(body) != declared
            or manifest.get("plan_content_sha256")
            != self.plan.content_sha256
            or not isinstance(physical_rows, list)
        ):
            raise ValueError("logical-scope binding set has an invalid binding")
        attempts: list[ValidatedStage1ScopeAttempt] = []
        expected_owners = [
            scope.scope_id for scope in self.plan.physical_scopes
        ]
        for owner_id, row in zip(
            expected_owners, physical_rows, strict=True
        ):
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "physical_owner_scope_id",
                    "physical_fit_key",
                    "physical_fit_key_record",
                    "physical_attempt_relative_path",
                    "attempt_request_sha256",
                    "attempt_manifest_content_sha256",
                }
                or row.get("physical_owner_scope_id") != owner_id
                or row.get("physical_fit_key")
                != self.plan.physical_fit_key(owner_id).key
                or row.get("physical_fit_key_record")
                != self.plan.physical_fit_key(owner_id).as_dict()
            ):
                raise ValueError(
                    "logical-scope binding physical capability is malformed"
                )
            relative = str(row["physical_attempt_relative_path"])
            parts = Path(relative).parts
            if (
                Path(relative).is_absolute()
                or len(parts) != 2
                or parts[0] != owner_id
                or _ATTEMPT_NAME.fullmatch(parts[1]) is None
            ):
                raise ValueError(
                    "logical-scope binding physical capability is noncanonical"
                )
            attempt = self._fresh_attempt_capability(
                scope_id=owner_id,
                attempt_dir=self.root / relative,
            )
            if (
                row.get("attempt_request_sha256")
                != attempt.manifest["attempt_request_sha256"]
                or row.get("attempt_manifest_content_sha256")
                != attempt.manifest["content_sha256"]
            ):
                raise ValueError(
                    "logical-scope binding physical attempt changed"
                )
            attempts.append(attempt)
        if len(physical_rows) != len(expected_owners):
            raise ValueError(
                "logical-scope binding physical coverage changed"
            )
        expected = self._logical_binding_manifest(attempts)
        if dict(manifest) != expected:
            raise ValueError("logical-scope binding set changed")
        return ValidatedStage1LogicalScopeBindings(
            path=path.resolve(),
            manifest=copy.deepcopy(dict(manifest)),
        )

    def _attempt_filesystem_identity(
        self,
        *,
        scope_root: Path,
        attempt_dir: Path,
    ) -> dict[str, Any]:
        body = {
            "schema_version": STAGE1_ATTEMPT_FILESYSTEM_IDENTITY_SCHEMA,
            "attempt_store_root": dict(self._root_inode_binding),
            "scope_directory": _directory_inode_binding(
                scope_root,
                label="scope-attempt scope directory",
            ),
            "attempt_directory": _directory_inode_binding(
                attempt_dir,
                label="scope-attempt directory",
            ),
        }
        return {**body, "content_sha256": _sha256_json(body)}

    def _validated_attempt_path(
        self,
        attempt_dir: Path | str,
        *,
        scope_id: str,
    ) -> Path:
        root_fd, root = _open_safe_directory(
            self.root,
            label="scope-attempt root",
        )
        os.close(root_fd)
        scope_root_fd, scope_root = _open_safe_directory(
            root / str(scope_id),
            label=f"scope-attempt directory {scope_id}",
        )
        os.close(scope_root_fd)
        path = _absolute_path(attempt_dir)
        if (
            _ATTEMPT_NAME.fullmatch(path.name) is None
            or path.parent != scope_root
        ):
            raise ValueError("scope attempt is outside its canonical scope directory")
        attempt_fd, opened = _open_safe_directory(
            path,
            label=f"scope attempt {scope_id}",
        )
        os.close(attempt_fd)
        return opened

    def begin(
        self,
        *,
        scope_id: str,
        worker_target: str,
        worker_parameters: Mapping[str, Any] | None = None,
    ) -> Stage1ScopeExecutionRequest:
        if _WORKER_TARGET.fullmatch(str(worker_target)) is None:
            raise ValueError("worker_target must be one module:function import path")
        scope = self.plan.scope(scope_id)
        assignment = self.plan.assignment(scope_id)
        parameters = _closed_json(
            dict(worker_parameters or {}), path="worker_parameters"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        root_fd, root = _open_safe_directory(
            self.root,
            label="scope-attempt root",
        )
        os.close(root_fd)
        scope_root = root / scope.scope_id
        scope_root.mkdir(parents=True, exist_ok=True)
        scope_fd, scope_root = _open_safe_directory(
            scope_root,
            label=f"scope-attempt directory {scope.scope_id}",
        )
        os.close(scope_fd)
        attempt_name = (
            "attempt_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + "_"
            + uuid.uuid4().hex
        )
        attempt_dir = scope_root / attempt_name
        attempt_dir.mkdir(parents=False, exist_ok=False)
        filesystem_identity = self._attempt_filesystem_identity(
            scope_root=scope_root,
            attempt_dir=attempt_dir,
        )
        body = {
            "schema_version": STAGE1_SCOPE_ATTEMPT_REQUEST_SCHEMA,
            "plan_content_sha256": self.plan.content_sha256,
            "attempt_store_identity": self.identity(),
            "attempt_filesystem_identity": filesystem_identity,
            "scope": scope.as_dict(),
            "assignment": assignment.as_dict(),
            "worker_target": str(worker_target),
            "worker_parameters": parameters,
            "worker_parameters_sha256": _sha256_json(parameters),
            "created_at": _utc_now(),
            "heldout_labels_supplied": False,
            "torch_determinism_policy": stage1_torch_determinism_policy(),
        }
        payload = {**body, "attempt_request_sha256": _sha256_json(body)}
        _write_immutable_json(attempt_dir / "attempt_request.json", payload)
        return Stage1ScopeExecutionRequest(
            attempt_dir=str(attempt_dir.resolve()),
            plan_content_sha256=self.plan.content_sha256,
            scope=scope.as_dict(),
            assignment=assignment.as_dict(),
            worker_target=str(worker_target),
            worker_parameters=parameters,
            worker_parameters_sha256=_sha256_json(parameters),
            attempt_request_sha256=payload["attempt_request_sha256"],
        )

    def validate_completed(
        self,
        attempt_dir: Path | str,
        *,
        scope_id: str,
        worker_target: str,
        worker_parameters: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        path = self._validated_attempt_path(
            attempt_dir,
            scope_id=scope_id,
        )
        filesystem_identity = self._attempt_filesystem_identity(
            scope_root=path.parent,
            attempt_dir=path,
        )
        manifest_path = path / "attempt_manifest.json"
        manifest = _load_strict_json_file(
            manifest_path,
            label="scope attempt terminal manifest",
        )
        if not isinstance(manifest, Mapping):
            raise ValueError("scope attempt manifest is not an object")
        manifest_fields = {
            "schema_version",
            "plan_content_sha256",
            "attempt_request_sha256",
            "attempt_store_identity",
            "attempt_filesystem_identity",
            "scope",
            "assignment",
            "worker_target",
            "worker_parameters_sha256",
            "status",
            "heldout_labels_supplied",
            "torch_determinism_policy",
            "torch_determinism_observed",
            "files",
            "content_sha256",
        }
        body = dict(manifest)
        declared = body.pop("content_sha256", None)
        expected_parameters = _closed_json(
            dict(worker_parameters or {}), path="worker_parameters"
        )
        scope = self.plan.scope(scope_id)
        assignment = self.plan.assignment(scope_id)
        if (
            set(manifest) != manifest_fields
            or manifest.get("schema_version")
            != STAGE1_SCOPE_ATTEMPT_MANIFEST_SCHEMA
            or _SHA256.fullmatch(str(declared or "")) is None
            or _sha256_json(body) != declared
            or manifest.get("plan_content_sha256") != self.plan.content_sha256
            or manifest.get("attempt_store_identity") != self.identity()
            or manifest.get("attempt_filesystem_identity")
            != filesystem_identity
            or manifest.get("scope") != scope.as_dict()
            or manifest.get("assignment") != assignment.as_dict()
            or manifest.get("worker_target") != str(worker_target)
            or manifest.get("worker_parameters_sha256")
            != _sha256_json(expected_parameters)
            or manifest.get("status") != "completed"
            or manifest.get("heldout_labels_supplied") is not False
            or manifest.get("torch_determinism_policy")
            != stage1_torch_determinism_policy()
        ):
            raise ValueError("scope attempt terminal manifest has an invalid binding")
        observed = _validate_torch_determinism_observation(
            manifest.get("torch_determinism_observed")
        )
        request_path = path / "attempt_request.json"
        request = _load_strict_json_file(
            request_path,
            label="scope attempt request",
        )
        if not isinstance(request, Mapping):
            raise ValueError("scope attempt request is not an object")
        request_fields = {
            "schema_version",
            "plan_content_sha256",
            "attempt_store_identity",
            "attempt_filesystem_identity",
            "scope",
            "assignment",
            "worker_target",
            "worker_parameters",
            "worker_parameters_sha256",
            "created_at",
            "heldout_labels_supplied",
            "torch_determinism_policy",
            "attempt_request_sha256",
        }
        request_body = dict(request)
        request_sha = request_body.pop("attempt_request_sha256", None)
        if (
            set(request) != request_fields
            or request.get("schema_version")
            != STAGE1_SCOPE_ATTEMPT_REQUEST_SCHEMA
            or _SHA256.fullmatch(str(request_sha or "")) is None
            or _sha256_json(request_body) != request_sha
            or request.get("plan_content_sha256") != self.plan.content_sha256
            or request.get("attempt_store_identity") != self.identity()
            or request.get("attempt_filesystem_identity")
            != filesystem_identity
            or request.get("scope") != scope.as_dict()
            or request.get("assignment") != assignment.as_dict()
            or request.get("worker_target") != str(worker_target)
            or request.get("worker_parameters") != expected_parameters
            or request.get("worker_parameters_sha256")
            != _sha256_json(expected_parameters)
            or request.get("heldout_labels_supplied") is not False
            or request.get("torch_determinism_policy")
            != stage1_torch_determinism_policy()
            or _UTC_TIMESTAMP.fullmatch(str(request.get("created_at") or ""))
            is None
            or manifest.get("attempt_request_sha256") != request_sha
        ):
            raise ValueError("scope attempt request differs from its terminal binding")
        inventory = _attempt_inventory(
            path,
            expected_attempt_binding=filesystem_identity[
                "attempt_directory"
            ],
        )
        if manifest.get("files") != inventory:
            raise ValueError("scope attempt file inventory changed")
        result_path = path / "worker_result.json"
        result = _load_strict_json_file(
            result_path,
            label="scope worker result",
        )
        if not isinstance(result, Mapping):
            raise ValueError("scope worker result is not an object")
        result_fields = {
            "schema_version",
            "scope_id",
            "scope_seed",
            "gpu_id",
            "torch_determinism_policy",
            "torch_determinism_observed",
            "result",
            "content_sha256",
        }
        result_body = dict(result)
        result_sha = result_body.pop("content_sha256", None)
        if (
            set(result) != result_fields
            or result.get("schema_version")
            != STAGE1_SCOPE_WORKER_RESULT_SCHEMA
            or _SHA256.fullmatch(str(result_sha or "")) is None
            or _sha256_json(result_body) != result_sha
            or result.get("scope_id") != scope.scope_id
            or result.get("scope_seed") != scope.scope_seed
            or result.get("gpu_id") != assignment.gpu_id
            or result.get("torch_determinism_policy")
            != stage1_torch_determinism_policy()
            or _validate_torch_determinism_observation(
                result.get("torch_determinism_observed")
            )
            != observed
            or not isinstance(result.get("result"), Mapping)
            or _closed_json(dict(result["result"]), path="worker_result")
            != result["result"]
        ):
            raise ValueError("scope worker result has an invalid binding")
        if (
            self._attempt_filesystem_identity(
                scope_root=path.parent,
                attempt_dir=path,
            )
            != filesystem_identity
        ):
            raise ValueError(
                "scope attempt filesystem identity changed during validation"
            )
        return dict(manifest)

    def seal(
        self,
        request: Stage1ScopeExecutionRequest,
        *,
        worker_result: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Seal one locally executed request, primarily for custom workers/tests."""

        if not isinstance(request, Stage1ScopeExecutionRequest):
            raise TypeError("request must be a Stage1ScopeExecutionRequest")
        path = self._validated_attempt_path(
            request.attempt_dir,
            scope_id=request.scope_id,
        )
        if request.plan_content_sha256 != self.plan.content_sha256:
            raise ValueError("scope attempt belongs to another execution plan")
        persisted_request = _load_strict_json_file(
            path / "attempt_request.json",
            label="scope attempt request",
        )
        if (
            not isinstance(persisted_request, Mapping)
            or persisted_request.get("attempt_request_sha256")
            != request.attempt_request_sha256
            or persisted_request.get("scope") != dict(request.scope)
            or persisted_request.get("assignment") != dict(request.assignment)
            or persisted_request.get("worker_target") != request.worker_target
            or persisted_request.get("worker_parameters")
            != dict(request.worker_parameters)
        ):
            raise ValueError("scope execution request differs from its immutable file")
        observed = _enforce_stage1_torch_determinism()
        return _seal_scope_attempt(
            request,
            worker_result=dict(worker_result or {}),
            torch_determinism_observed=observed,
        )

    def reusable_attempt(
        self,
        *,
        scope_id: str,
        worker_target: str,
        worker_parameters: Mapping[str, Any] | None = None,
    ) -> ValidatedStage1ScopeAttempt | None:
        if not self.root.exists():
            return None
        root_fd, root = _open_safe_directory(
            self.root,
            label="scope-attempt root",
        )
        os.close(root_fd)
        scope_root = root / str(scope_id)
        if not scope_root.exists():
            return None
        scope_fd, scope_root = _open_safe_directory(
            scope_root,
            label=f"scope-attempt directory {scope_id}",
        )
        os.close(scope_fd)
        completed: list[ValidatedStage1ScopeAttempt] = []
        for attempt in sorted(scope_root.iterdir()):
            attempt_fd, attempt = _open_safe_directory(
                attempt,
                label=f"scope attempt candidate {scope_id}",
            )
            os.close(attempt_fd)
            if _ATTEMPT_NAME.fullmatch(attempt.name) is None:
                raise ValueError("scope-attempt directory name is not canonical")
            if not (attempt / "attempt_manifest.json").exists():
                # Interrupted and failed attempts are intentionally preserved.
                continue
            manifest = self.validate_completed(
                attempt,
                scope_id=scope_id,
                worker_target=worker_target,
                worker_parameters=worker_parameters,
            )
            completed.append(
                ValidatedStage1ScopeAttempt(
                    scope_id=str(scope_id),
                    attempt_dir=attempt.resolve(),
                    manifest=manifest,
                )
            )
        if len(completed) > 1:
            raise RuntimeError("scope has multiple terminal completed attempts")
        return None if not completed else completed[0]

    def reusable(
        self,
        *,
        scope_id: str,
        worker_target: str,
        worker_parameters: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        """Compatibility view returning only the terminal manifest."""

        attempt = self.reusable_attempt(
            scope_id=scope_id,
            worker_target=worker_target,
            worker_parameters=worker_parameters,
        )
        return None if attempt is None else attempt.manifest


def _seal_scope_attempt(
    request: Stage1ScopeExecutionRequest,
    *,
    worker_result: Mapping[str, Any],
    torch_determinism_observed: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    attempt_dir = Path(request.attempt_dir)
    persisted_request = _load_strict_json_file(
        attempt_dir / "attempt_request.json",
        label="scope attempt request before sealing",
    )
    if (
        not isinstance(persisted_request, Mapping)
        or persisted_request.get("attempt_request_sha256")
        != request.attempt_request_sha256
        or not isinstance(
            persisted_request.get("attempt_store_identity"), Mapping
        )
        or not isinstance(
            persisted_request.get("attempt_filesystem_identity"), Mapping
        )
    ):
        raise ValueError("scope attempt request changed before terminal sealing")
    attempt_store_identity = copy.deepcopy(
        dict(persisted_request["attempt_store_identity"])
    )
    attempt_filesystem_identity = copy.deepcopy(
        dict(persisted_request["attempt_filesystem_identity"])
    )
    _validate_directory_inode_binding(
        attempt_store_identity.get("root_inode_binding"),
        path=attempt_dir.parent.parent,
        label="scope-attempt root before sealing",
    )
    _validate_directory_inode_binding(
        attempt_filesystem_identity.get("scope_directory"),
        path=attempt_dir.parent,
        label="scope-attempt scope directory before sealing",
    )
    _validate_directory_inode_binding(
        attempt_filesystem_identity.get("attempt_directory"),
        path=attempt_dir,
        label="scope-attempt directory before sealing",
    )
    observed = _validate_torch_determinism_observation(
        _closed_json(
        dict(
            torch_determinism_observed
            if torch_determinism_observed is not None
            else _observe_stage1_torch_determinism()
        ),
        path="torch_determinism_observed",
        )
    )
    result_body = {
        "schema_version": STAGE1_SCOPE_WORKER_RESULT_SCHEMA,
        "scope_id": request.scope_id,
        "scope_seed": request.scope_seed,
        "gpu_id": request.gpu_id,
        "torch_determinism_policy": stage1_torch_determinism_policy(),
        "torch_determinism_observed": observed,
        "result": _closed_json(dict(worker_result), path="worker_result"),
    }
    result_payload = {**result_body, "content_sha256": _sha256_json(result_body)}
    _write_immutable_json(attempt_dir / "worker_result.json", result_payload)
    _durably_sync_attempt_tree(attempt_dir)
    _durably_sync_attempt_parent_chain(attempt_dir)
    inventory = _attempt_inventory(
        attempt_dir,
        expected_attempt_binding=attempt_filesystem_identity[
            "attempt_directory"
        ],
    )
    if _attempt_inventory(
        attempt_dir,
        expected_attempt_binding=attempt_filesystem_identity[
            "attempt_directory"
        ],
    ) != inventory:
        raise RuntimeError("scope attempt changed after durability synchronization")
    body = {
        "schema_version": STAGE1_SCOPE_ATTEMPT_MANIFEST_SCHEMA,
        "plan_content_sha256": request.plan_content_sha256,
        "attempt_request_sha256": request.attempt_request_sha256,
        "attempt_store_identity": attempt_store_identity,
        "attempt_filesystem_identity": attempt_filesystem_identity,
        "scope": dict(request.scope),
        "assignment": dict(request.assignment),
        "worker_target": request.worker_target,
        "worker_parameters_sha256": request.worker_parameters_sha256,
        "status": "completed",
        "heldout_labels_supplied": False,
        "torch_determinism_policy": stage1_torch_determinism_policy(),
        "torch_determinism_observed": observed,
        "files": inventory,
    }
    manifest = {**body, "content_sha256": _sha256_json(body)}
    # The terminal manifest is always written last.
    _write_immutable_json(attempt_dir / "attempt_manifest.json", manifest)
    return manifest


class Stage1ScopeProgressLedger:
    """Atomic, non-authoritative operational progress for one closed plan."""

    def __init__(
        self,
        path: Path | str,
        plan: Stage1ScopePlan,
        *,
        execution_binding: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.plan = plan
        supplied_binding = (
            {
                "schema_version": "production_stage1_scope_execution_binding_v1",
                "mode": "standalone_progress_ledger",
                "plan_content_sha256": plan.content_sha256,
                "progress_path": str(_absolute_path(self.path)),
            }
            if execution_binding is None
            else dict(execution_binding)
        )
        self.execution_binding = _closed_json(
            supplied_binding,
            path="execution_binding",
        )
        self.execution_binding_sha256 = _sha256_json(self.execution_binding)
        if self.path.exists():
            self._load()
        else:
            rows = []
            for scope in plan.scopes:
                assignment = plan.assignment(scope.scope_id)
                owner = plan.physical_owner(scope.scope_id)
                rows.append(
                    {
                        "scope_id": scope.scope_id,
                        "canonical_index": scope.canonical_index,
                        "scope_kind": scope.scope_kind,
                        "physical_owner_scope_id": owner.scope_id,
                        "execution_mode": (
                            "physical_fit"
                            if owner.scope_id == scope.scope_id
                            else "logical_reference"
                        ),
                        "fit_row_count": scope.fit_row_count,
                        "gpu_id": assignment.gpu_id,
                        "scope_seed": scope.scope_seed,
                        "status": "pending",
                        "attempt_dir": None,
                        "pid": None,
                        "started_at": None,
                        "heartbeat_at": None,
                        "finished_at": None,
                        "elapsed_seconds": None,
                        "peak_gpu_allocated_bytes": None,
                        "peak_gpu_reserved_bytes": None,
                        "output_bytes": None,
                        "throughput_fit_rows_per_second": None,
                        "logical_reference_sha256": None,
                        "failure": None,
                    }
                )
            self._write(rows)

    def _load(self) -> dict[str, Any]:
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("Stage 1 progress ledger must be a regular file")
        value = _load_strict_json_file(
            self.path,
            label="Stage 1 progress ledger",
        )
        if not isinstance(value, Mapping):
            raise ValueError("Stage 1 progress ledger is not an object")
        body = dict(value)
        declared = body.pop("content_sha256", None)
        if (
            value.get("schema_version") != STAGE1_SCOPE_PROGRESS_SCHEMA
            or value.get("plan_content_sha256") != self.plan.content_sha256
            or value.get("execution_binding") != self.execution_binding
            or value.get("execution_binding_sha256")
            != self.execution_binding_sha256
            or _SHA256.fullmatch(str(declared or "")) is None
            or _sha256_json(body) != declared
            or [row.get("scope_id") for row in value.get("scopes") or ()]
            != [scope.scope_id for scope in self.plan.scopes]
        ):
            raise ValueError("Stage 1 progress ledger has an invalid binding")
        return dict(value)

    def _write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        counts = {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted(_PROGRESS_STATUSES)
        }
        body = {
            "schema_version": STAGE1_SCOPE_PROGRESS_SCHEMA,
            "plan_content_sha256": self.plan.content_sha256,
            "execution_binding": self.execution_binding,
            "execution_binding_sha256": self.execution_binding_sha256,
            "planned_scope_count": len(self.plan.scopes),
            "planned_logical_scope_count": len(self.plan.scopes),
            "planned_physical_fit_count": len(self.plan.physical_scopes),
            "counts": counts,
            "completed_fit_row_units": sum(
                int(row["fit_row_count"])
                for row in rows
                if row.get("status") == "completed"
                and row.get("execution_mode") == "physical_fit"
            ),
            "planned_fit_row_units": sum(
                scope.fit_row_count for scope in self.plan.physical_scopes
            ),
            "updated_at": _utc_now(),
            "scopes": [dict(row) for row in rows],
        }
        _atomic_write_json(
            self.path, {**body, "content_sha256": _sha256_json(body)}
        )

    def update(self, scope_id: str, status: str, **fields: Any) -> None:
        if status not in _PROGRESS_STATUSES:
            raise ValueError("unsupported Stage 1 scope progress status")
        current = self._load()
        rows = [dict(row) for row in current["scopes"]]
        selected = next(
            (row for row in rows if row["scope_id"] == str(scope_id)), None
        )
        if selected is None:
            raise ValueError("progress update names an unknown scope")
        previous = str(selected["status"])
        if previous in _TERMINAL_SCOPE_STATUSES and status != previous:
            raise RuntimeError("a completed Stage 1 scope cannot change status")
        allowed = {
            "pending": {"pending", "running", "completed"},
            "running": {"running", "sealing", "failed"},
            "sealing": {"sealing", "completed", "failed"},
            "failed": {"failed", "running", "completed"},
            "completed": {"completed"},
        }
        if status not in allowed[previous]:
            raise RuntimeError(
                f"invalid Stage 1 progress transition: {previous} -> {status}"
            )
        selected["status"] = status
        for key, value in fields.items():
            if key not in selected:
                raise ValueError(f"unknown Stage 1 progress field: {key}")
            selected[key] = _closed_json(value, path=f"progress.{key}")
        now = _utc_now()
        if status == "running":
            selected["started_at"] = selected["started_at"] or now
            selected["heartbeat_at"] = now
        if status in {"completed", "failed"}:
            selected["finished_at"] = now
            selected["heartbeat_at"] = now
        self._write(rows)

    def reconcile_logical_references(
        self,
        bindings: ValidatedStage1LogicalScopeBindings,
    ) -> None:
        """Mark all logical purposes complete from sealed owner references."""

        if not isinstance(bindings, ValidatedStage1LogicalScopeBindings):
            raise TypeError(
                "bindings must be authenticated Stage 1 logical references"
            )
        manifest = dict(bindings.manifest)
        body = dict(manifest)
        declared = body.pop("content_sha256", None)
        rows = manifest.get("logical_bindings")
        if (
            manifest.get("schema_version")
            != STAGE1_LOGICAL_SCOPE_BINDING_SET_SCHEMA
            or manifest.get("plan_content_sha256")
            != self.plan.content_sha256
            or _SHA256.fullmatch(str(declared or "")) is None
            or _sha256_json(body) != declared
            or not isinstance(rows, list)
            or [row.get("logical_scope_id") for row in rows]
            != [scope.scope_id for scope in self.plan.scopes]
        ):
            raise ValueError("logical references do not match the progress plan")
        for scope, binding in zip(self.plan.scopes, rows, strict=True):
            owner = self.plan.physical_owner(scope.scope_id)
            binding_body = dict(binding)
            binding_sha = binding_body.pop("content_sha256", None)
            if (
                binding.get("schema_version")
                != STAGE1_LOGICAL_SCOPE_BINDING_SCHEMA
                or binding.get("logical_scope_sha256")
                != scope.as_dict()["scope_sha256"]
                or binding.get("physical_owner_scope_id") != owner.scope_id
                or binding.get("physical_owner_scope_sha256")
                != owner.as_dict()["scope_sha256"]
                or binding.get("canonical_group_seed") != owner.scope_seed
                or binding.get("heldout_labels_supplied_to_physical_worker")
                is not False
                or _SHA256.fullmatch(str(binding_sha or "")) is None
                or _sha256_json(binding_body) != binding_sha
            ):
                raise ValueError("logical reference row has an invalid binding")
            attempt_path = (
                bindings.path.parent
                / str(binding["physical_attempt_relative_path"])
            ).resolve(strict=True)
            fields: dict[str, Any] = {
                "attempt_dir": str(attempt_path),
                "logical_reference_sha256": str(binding_sha),
                "failure": None,
            }
            if scope.scope_id != owner.scope_id:
                fields.update(
                    {
                        "pid": None,
                        "output_bytes": 0,
                        "throughput_fit_rows_per_second": None,
                    }
                )
            self.update(scope.scope_id, "completed", **fields)

    def reconcile_authenticated_completion(
        self,
        attempt: ValidatedStage1ScopeAttempt,
    ) -> None:
        """Repair stale operational state from an independently sealed attempt.

        The progress ledger is deliberately non-authoritative. A process may
        crash after a worker durably publishes and authenticates its terminal
        manifest but before the parent records ``completed``. This method is
        the sole transition that permits ``running`` or ``sealing`` directly
        to ``completed`` during resume, and it requires the manifest to be
        freshly reopened and bound to this exact plan and attempt-store root.
        """

        if not isinstance(attempt, ValidatedStage1ScopeAttempt):
            raise TypeError(
                "attempt must be an authenticated Stage 1 scope attempt"
            )
        scope = self.plan.scope(attempt.scope_id)
        assignment = self.plan.assignment(attempt.scope_id)
        path = _absolute_path(attempt.attempt_dir)
        persisted = _load_strict_json_file(
            path / "attempt_manifest.json",
            label="reconciled Stage 1 scope terminal manifest",
        )
        if not isinstance(persisted, Mapping) or dict(persisted) != dict(
            attempt.manifest
        ):
            raise ValueError(
                "reconciled Stage 1 scope manifest changed after authentication"
            )
        manifest = dict(persisted)
        manifest_body = dict(manifest)
        declared = manifest_body.pop("content_sha256", None)
        expected_store_identity = self.execution_binding.get(
            "attempt_store_identity"
        )
        if (
            manifest.get("schema_version")
            != STAGE1_SCOPE_ATTEMPT_MANIFEST_SCHEMA
            or _SHA256.fullmatch(str(declared or "")) is None
            or _sha256_json(manifest_body) != declared
            or manifest.get("plan_content_sha256")
            != self.plan.content_sha256
            or manifest.get("attempt_store_identity")
            != expected_store_identity
            or not isinstance(
                manifest.get("attempt_filesystem_identity"), Mapping
            )
            or manifest.get("scope") != scope.as_dict()
            or manifest.get("assignment") != assignment.as_dict()
            or manifest.get("status") != "completed"
            or manifest.get("heldout_labels_supplied") is not False
            or not isinstance(manifest.get("files"), list)
        ):
            raise ValueError(
                "reconciled Stage 1 scope completion has an invalid binding"
            )
        if (
            not isinstance(expected_store_identity, Mapping)
            or path.parent.parent
            != _absolute_path(str(expected_store_identity.get("root") or ""))
        ):
            raise ValueError(
                "reconciled Stage 1 scope attempt is outside its bound store"
            )
        filesystem_identity = manifest["attempt_filesystem_identity"]
        _validate_directory_inode_binding(
            expected_store_identity.get("root_inode_binding"),
            path=path.parent.parent,
            label="reconciled scope-attempt root",
        )
        _validate_directory_inode_binding(
            filesystem_identity.get("scope_directory"),
            path=path.parent,
            label="reconciled scope directory",
        )
        _validate_directory_inode_binding(
            filesystem_identity.get("attempt_directory"),
            path=path,
            label="reconciled attempt directory",
        )
        current = self._load()
        rows = [dict(row) for row in current["scopes"]]
        selected = next(
            (
                row
                for row in rows
                if row["scope_id"] == str(attempt.scope_id)
            ),
            None,
        )
        if selected is None:
            raise ValueError("reconciled completion names an unknown scope")
        if (
            selected.get("status") == "completed"
            and selected.get("attempt_dir") not in {None, str(path)}
        ):
            raise RuntimeError(
                "completed Stage 1 progress names another attempt directory"
            )
        selected.update(
            {
                "status": "completed",
                "attempt_dir": str(path),
                "pid": None,
                "gpu_id": assignment.gpu_id,
                "heartbeat_at": _utc_now(),
                "finished_at": _utc_now(),
                "output_bytes": sum(
                    int(row["size"]) for row in manifest["files"]
                ),
                "failure": None,
            }
        )
        self._write(rows)

    def snapshot(self) -> Mapping[str, Any]:
        return self._load()


def _resolve_worker_target(target: str) -> Callable[[Stage1ScopeExecutionRequest], Any]:
    if _WORKER_TARGET.fullmatch(target) is None:
        raise ValueError("worker target import path is malformed")
    module_name, attribute = target.split(":", 1)
    module = importlib.import_module(module_name)
    resolved = getattr(module, attribute, None)
    if not callable(resolved):
        raise TypeError("scope worker target is not callable")
    return resolved


def _start_spawned_process_with_scope_hash_seed(
    process: mp.Process,
    *,
    scope_seed: int,
) -> None:
    """Start one spawn child with deterministic hash randomization only."""

    resolved = int(scope_seed)
    if not 0 <= resolved < 2**31:
        raise ValueError("scope hash seed must be a nonnegative 31-bit integer")
    key = "PYTHONHASHSEED"
    was_present = key in os.environ
    previous = os.environ.get(key)
    os.environ[key] = str(resolved)
    try:
        process.start()
    finally:
        if was_present:
            assert previous is not None
            os.environ[key] = previous
        else:
            os.environ.pop(key, None)


_WORKER_PROCESS_GROUP_MARKER_SCHEMA = (
    "production_stage1_worker_process_group_ready_v2"
)


def _linux_process_start_time_ticks(pid: int) -> int | None:
    """Read Linux's non-repeating process start identity when the PID exists."""

    path = Path(f"/proc/{int(pid)}/stat")
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"cannot authenticate process identity for PID {pid}") from exc
    closing = payload.rfind(")")
    fields = payload[closing + 1 :].strip().split() if closing >= 0 else []
    # The suffix begins at field 3 (state); Linux starttime is field 22.
    if len(fields) <= 19 or not fields[19].isdigit():
        raise RuntimeError(f"cannot parse process identity for PID {pid}")
    return int(fields[19])


def _establish_worker_process_group(
    marker_path: Path | str | None = None,
) -> None:
    """Place a spawned worker and all future descendants in a new session."""

    if os.name != "posix":
        raise RuntimeError(
            "production Stage 1 descendant cleanup requires POSIX process groups"
        )
    os.setsid()
    if os.getpgrp() != os.getpid():
        raise RuntimeError("spawned Stage 1 worker did not become group leader")
    if marker_path is not None:
        start_time_ticks = _linux_process_start_time_ticks(os.getpid())
        if start_time_ticks is None:
            raise RuntimeError("spawned Stage 1 worker lacks a live process identity")
        body = {
            "schema_version": _WORKER_PROCESS_GROUP_MARKER_SCHEMA,
            "pid": int(os.getpid()),
            "process_group_id": int(os.getpgrp()),
            "process_start_time_ticks": start_time_ticks,
        }
        _write_immutable_json(
            Path(marker_path),
            {**body, "content_sha256": _sha256_json(body)},
        )


def _authenticated_private_process_group_marker(
    marker_path: Path | str | None,
    *,
    pid: int,
) -> bool:
    if marker_path is None:
        return False
    try:
        value = _load_strict_json_file(
            Path(marker_path),
            label="worker process-group marker",
        )
    except (FileNotFoundError, OSError, ValueError):
        return False
    if not isinstance(value, Mapping):
        return False
    fields = {
        "schema_version",
        "pid",
        "process_group_id",
        "process_start_time_ticks",
        "content_sha256",
    }
    body = dict(value)
    declared = body.pop("content_sha256", None)
    valid = bool(
        set(value) == fields
        and value.get("schema_version")
        == _WORKER_PROCESS_GROUP_MARKER_SCHEMA
        and value.get("pid") == int(pid)
        and value.get("process_group_id") == int(pid)
        and type(value.get("process_start_time_ticks")) is int
        and int(value["process_start_time_ticks"]) >= 0
        and _sha256_json(body) == declared
    )
    if not valid:
        return False
    current_start = _linux_process_start_time_ticks(int(pid))
    return bool(
        current_start is None
        or current_start == int(value["process_start_time_ticks"])
    )


def _terminate_process_and_descendants(
    process: mp.Process,
    *,
    process_group_marker_path: Path | str | None = None,
    timeout_seconds: float = 10.0,
) -> None:
    """Terminate one spawned worker group, then escalate without touching peers."""

    pid = process.pid
    if pid is None:
        return
    group_id: int | None = None
    if os.name == "posix":
        marker_authenticated = _authenticated_private_process_group_marker(
            process_group_marker_path,
            pid=int(pid),
        )
        try:
            observed_group = os.getpgid(int(pid))
        except ProcessLookupError:
            observed_group = None
        # A live PID is trusted only when it is still the authenticated
        # leader, or while the multiprocessing handle confirms it is our
        # still-running child. This prevents killing an unrelated reused PID.
        live_owned_leader = bool(
            observed_group == int(pid)
            and (
                marker_authenticated
                or (
                    process.is_alive()
                    and getattr(process, "exitcode", None) is None
                )
            )
        )
        if live_owned_leader or (
            observed_group is None and marker_authenticated
        ):
            group_id = int(pid)
            try:
                os.killpg(group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
    if process.is_alive() and group_id is None:
        process.terminate()
    process.join(timeout=float(timeout_seconds))
    # The group leader may exit before a descendant. Always issue the group
    # escalation when a private group was established.
    if group_id is not None:
        try:
            os.killpg(group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.is_alive():
        process.kill()
    process.join(timeout=float(timeout_seconds))
    if process.is_alive():
        raise RuntimeError("spawned Stage 1 worker survived forced termination")


def _spawned_scope_worker(
    request: Stage1ScopeExecutionRequest,
    messages: Any,
) -> None:
    started = time.monotonic()
    attempt_dir = Path(request.attempt_dir)
    terminal_published = False
    try:
        _establish_worker_process_group(
            attempt_dir / "process_group_ready.json"
        )
        request.payload_dir.mkdir(parents=False, exist_ok=False)
        determinism = _enforce_stage1_torch_determinism()
        seed_stage1_scope_rngs(request.scope_seed, gpu_id=request.gpu_id)
        import torch
        from threadpoolctl import threadpool_limits

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # A worker target may import a library that initialized the
            # inter-op pool before entry. The spawned process remains isolated,
            # and BLAS/OpenMP plus Torch intra-op work are still bounded.
            pass
        messages.put(
            {
                "event": "started",
                "scope_id": request.scope_id,
                "pid": os.getpid(),
                "at": _utc_now(),
            }
        )
        target = _resolve_worker_target(request.worker_target)
        with threadpool_limits(limits=1):
            raw_result = target(request)
        determinism_after = _observe_stage1_torch_determinism()
        if (
            determinism_after.get("policy_active") is not True
            or {
                key: value
                for key, value in determinism.items()
                if key not in {"torch_version", "cuda_runtime_version"}
            }
            != {
                key: value
                for key, value in determinism_after.items()
                if key not in {"torch_version", "cuda_runtime_version"}
            }
        ):
            raise RuntimeError(
                "scope worker weakened or changed the strict Torch determinism policy"
            )
        if raw_result is None:
            raw_result = {}
        if not isinstance(raw_result, Mapping):
            raise TypeError("scope worker must return a mapping or None")
        elapsed = max(time.monotonic() - started, 1e-9)
        peak_allocated = None
        peak_reserved = None
        if request.gpu_id is not None:
            import torch

            peak_allocated = int(
                torch.cuda.max_memory_allocated(int(request.gpu_id))
            )
            peak_reserved = int(
                torch.cuda.max_memory_reserved(int(request.gpu_id))
            )
        messages.put(
            {
                "event": "sealing",
                "scope_id": request.scope_id,
                "at": _utc_now(),
            }
        )
        manifest = _seal_scope_attempt(
            request,
            worker_result=dict(raw_result),
            torch_determinism_observed=determinism_after,
        )
        terminal_published = True
        output_bytes = sum(int(row["size"]) for row in manifest["files"])
        messages.put(
            {
                "event": "completed",
                "scope_id": request.scope_id,
                "pid": os.getpid(),
                "elapsed_seconds": elapsed,
                "peak_gpu_allocated_bytes": peak_allocated,
                "peak_gpu_reserved_bytes": peak_reserved,
                "output_bytes": output_bytes,
                "throughput_fit_rows_per_second": (
                    int(request.scope["fit_row_count"]) / elapsed
                ),
                "at": _utc_now(),
            }
        )
    except BaseException as exc:
        # Once the immutable terminal marker exists, it is the authoritative
        # result. A Queue feeder/transport failure after publication must not
        # add ``failure.json`` and thereby invalidate the sealed inventory.
        if terminal_published or os.path.lexists(
            attempt_dir / "attempt_manifest.json"
        ):
            raise
        failure_body = {
            "schema_version": "production_stage1_scope_failure_v1",
            "scope_id": request.scope_id,
            "pid": os.getpid(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            _atomic_write_json(
                attempt_dir / "failure.json",
                {**failure_body, "content_sha256": _sha256_json(failure_body)},
            )
        finally:
            messages.put(
                {
                    "event": "failed",
                    "scope_id": request.scope_id,
                    "pid": os.getpid(),
                    "failure": {
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "at": _utc_now(),
                }
            )
        raise


class SpawnedStage1ScopeOrchestrator:
    """Run sealed scopes with one spawn-only child per explicit GPU slot."""

    def __init__(
        self,
        *,
        plan: Stage1ScopePlan,
        attempt_root: Path | str,
        progress_path: Path | str,
        worker_target: str,
        worker_parameters: Mapping[str, Any] | None = None,
        worker_parameters_by_scope: Mapping[
            str, Mapping[str, Any]
        ] | None = None,
        poll_interval_seconds: float = 0.1,
        post_exit_message_grace_seconds: float = 2.0,
    ) -> None:
        if _WORKER_TARGET.fullmatch(str(worker_target)) is None:
            raise ValueError("worker_target must be one module:function import path")
        if plan.scope_workers_per_gpu != 1:
            raise ValueError("orchestrator permits one active scope per GPU")
        self.plan = plan
        self.store = Stage1ScopeAttemptStore(attempt_root, plan)
        self.worker_target = str(worker_target)
        if worker_parameters_by_scope is not None and worker_parameters is not None:
            raise ValueError(
                "global and per-scope worker parameters are mutually exclusive"
            )
        expected_scope_ids = {
            scope.scope_id for scope in plan.physical_scopes
        }
        if worker_parameters_by_scope is None:
            self.worker_parameters: Mapping[str, Any] | None = _closed_json(
                dict(worker_parameters or {}),
                path="worker_parameters",
            )
            self.worker_parameters_by_scope: Mapping[
                str, Mapping[str, Any]
            ] | None = None
            parameter_mode = "global"
            parameter_binding: Mapping[str, Any] = {
                "worker_parameters_sha256": _sha256_json(
                    self.worker_parameters
                ),
            }
        else:
            if set(map(str, worker_parameters_by_scope)) != expected_scope_ids:
                raise ValueError(
                    "per-scope worker parameters must cover exactly the canonical plan"
                )
            normalized: dict[str, Mapping[str, Any]] = {}
            for scope in plan.physical_scopes:
                raw = worker_parameters_by_scope.get(scope.scope_id)
                if not isinstance(raw, Mapping):
                    raise TypeError(
                        f"worker parameters for {scope.scope_id} must be a mapping"
                    )
                normalized[scope.scope_id] = _closed_json(
                    dict(raw),
                    path=f"worker_parameters_by_scope.{scope.scope_id}",
                )
            self.worker_parameters = None
            self.worker_parameters_by_scope = normalized
            parameter_mode = "per_scope"
            parameter_binding = {
                "worker_parameters_by_scope_sha256": _sha256_json(normalized),
                "worker_parameter_scope_order": [
                    scope.scope_id for scope in plan.physical_scopes
                ],
            }
        execution_binding = {
            "schema_version": "production_stage1_scope_execution_binding_v1",
            "mode": "spawned_scope_orchestrator",
            "plan_content_sha256": plan.content_sha256,
            "attempt_store_identity": self.store.identity(),
            "progress_path": str(_absolute_path(progress_path)),
            "worker_target": self.worker_target,
            "worker_parameter_mode": parameter_mode,
            **parameter_binding,
        }
        self.ledger = Stage1ScopeProgressLedger(
            progress_path,
            plan,
            execution_binding=execution_binding,
        )
        poll_interval = float(poll_interval_seconds)
        post_exit_grace = float(post_exit_message_grace_seconds)
        if (
            not math.isfinite(poll_interval)
            or poll_interval <= 0.0
            or not math.isfinite(post_exit_grace)
            or post_exit_grace <= 0.0
        ):
            raise ValueError(
                "scope polling and post-exit grace must be finite and positive"
            )
        self.poll_interval_seconds = max(0.01, poll_interval)
        self.post_exit_message_grace_seconds = max(
            self.poll_interval_seconds,
            post_exit_grace,
        )

    def worker_parameters_for_scope(self, scope_id: str) -> Mapping[str, Any]:
        """Return only the immutable parameter projection for one scope."""

        self.plan.scope(scope_id)
        if self.worker_parameters_by_scope is not None:
            return dict(self.worker_parameters_by_scope[str(scope_id)])
        assert self.worker_parameters is not None
        return dict(self.worker_parameters)

    def run(
        self,
        *,
        cancellation_event: Any | None = None,
    ) -> tuple[ValidatedStage1ScopeAttempt, ...]:
        """Execute each physical owner once and seal all logical references."""

        if (
            cancellation_event is not None
            and not callable(getattr(cancellation_event, "is_set", None))
        ):
            raise TypeError("cancellation_event must expose is_set()")
        completed: dict[str, ValidatedStage1ScopeAttempt] = {}
        pending: list[str] = []
        for scope_id in self.plan.physical_execution_order:
            parameters = self.worker_parameters_for_scope(scope_id)
            reusable = self.store.reusable_attempt(
                scope_id=scope_id,
                worker_target=self.worker_target,
                worker_parameters=parameters,
            )
            if reusable is None:
                pending.append(scope_id)
            else:
                completed[scope_id] = reusable
                self.ledger.reconcile_authenticated_completion(reusable)
        if not pending:
            ordered = tuple(
                completed[scope.scope_id]
                for scope in self.plan.physical_scopes
            )
            bindings = self.store.seal_logical_bindings(ordered)
            self.ledger.reconcile_logical_references(bindings)
            return ordered

        context = mp.get_context("spawn")
        messages = context.Queue()
        active: dict[int | None, tuple[str, mp.Process, Stage1ScopeExecutionRequest]] = {}
        queues: dict[int | None, list[str]] = {}
        completion_messages: dict[str, Mapping[str, Any]] = {}
        clean_exit_seen_at: dict[str, float] = {}
        last_heartbeat: dict[str, float] = {}
        for scope_id in pending:
            slot = self.plan.assignment(scope_id).gpu_id
            queues.setdefault(slot, []).append(scope_id)
        failure: tuple[str, Mapping[str, Any]] | None = None
        interruption: BaseException | None = None
        previous_sigterm_handler: Any = None
        sigterm_handler_installed = False

        try:
            if threading.current_thread() is threading.main_thread():
                previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

                def _interrupt_parent(signum: int, _frame: Any) -> None:
                    raise _Stage1ParentSignal(
                        f"Stage 1 scope orchestrator received signal {signum}"
                    )

                signal.signal(signal.SIGTERM, _interrupt_parent)
                sigterm_handler_installed = True
            while queues or active:
                if (
                    cancellation_event is not None
                    and cancellation_event.is_set()
                ):
                    failure = (
                        "external_component",
                        {
                            "exception_type": "PeerComponentFailure",
                            "message": (
                                "Stage 1 scope lane cancelled after its peer "
                                "component failed"
                            ),
                        },
                    )
                    break
                for slot in sorted(
                    list(queues),
                    key=lambda value: -1 if value is None else int(value),
                ):
                    if slot in active:
                        continue
                    queue = queues[slot]
                    if not queue:
                        del queues[slot]
                        continue
                    scope_id = queue.pop(0)
                    parameters = self.worker_parameters_for_scope(scope_id)
                    request = self.store.begin(
                        scope_id=scope_id,
                        worker_target=self.worker_target,
                        worker_parameters=parameters,
                    )
                    process = context.Process(
                        target=_spawned_scope_worker,
                        args=(request, messages),
                        name=f"stage1-{scope_id}",
                    )
                    _start_spawned_process_with_scope_hash_seed(
                        process,
                        scope_seed=request.scope_seed,
                    )
                    active[slot] = (scope_id, process, request)
                    last_heartbeat[scope_id] = time.monotonic()
                    self.ledger.update(
                        scope_id,
                        "running",
                        attempt_dir=request.attempt_dir,
                        pid=int(process.pid),
                        gpu_id=slot,
                    )

                try:
                    message = messages.get(timeout=self.poll_interval_seconds)
                except Empty:
                    message = None
                if isinstance(message, Mapping):
                    scope_id = str(message.get("scope_id") or "")
                    event = str(message.get("event") or "")
                    if event == "started":
                        self.ledger.update(
                            scope_id,
                            "running",
                            pid=int(message["pid"]),
                        )
                    elif event == "sealing":
                        self.ledger.update(scope_id, "sealing")
                    elif event == "completed":
                        # Completion is authoritative only after the process has
                        # exited cleanly and the parent has reopened the terminal
                        # manifest.  Until then the operational state is sealing.
                        if scope_id not in completed:
                            completion_messages[scope_id] = dict(message)
                    elif event == "failed":
                        failure = (scope_id, dict(message.get("failure") or {}))
                        self.ledger.update(
                            scope_id,
                            "failed",
                            pid=int(message["pid"]),
                            failure=dict(message.get("failure") or {}),
                        )

                for slot, (scope_id, process, request) in list(active.items()):
                    if process.is_alive():
                        # Parent-owned heartbeat; no worker writes the shared ledger.
                        now = time.monotonic()
                        if now - last_heartbeat.get(scope_id, 0.0) >= 30.0:
                            snapshot = self.ledger.snapshot()
                            status = next(
                                row["status"]
                                for row in snapshot["scopes"]
                                if row["scope_id"] == scope_id
                            )
                            if status == "running":
                                self.ledger.update(scope_id, "running")
                            last_heartbeat[scope_id] = now
                        continue
                    process.join()
                    if process.exitcode != 0:
                        del active[slot]
                        if failure is None:
                            failure = (
                                scope_id,
                                {
                                    "exception_type": "WorkerProcessError",
                                    "message": (
                                        f"spawned worker exited with code {process.exitcode}"
                                    ),
                                },
                            )
                            self.ledger.update(
                                scope_id,
                                "failed",
                                failure=failure[1],
                            )
                        continue
                    if scope_id not in completion_messages:
                        # ``multiprocessing.Queue`` uses a feeder thread, so
                        # allow a bounded delivery grace. After that, the
                        # terminal manifest—not Queue delivery—is authoritative.
                        now = time.monotonic()
                        first_seen = clean_exit_seen_at.setdefault(
                            scope_id, now
                        )
                        if (
                            now - first_seen
                            < self.post_exit_message_grace_seconds
                        ):
                            continue
                        del active[slot]
                        clean_exit_seen_at.pop(scope_id, None)
                        try:
                            manifest = self.store.validate_completed(
                                request.attempt_dir,
                                scope_id=scope_id,
                                worker_target=self.worker_target,
                                worker_parameters=(
                                    self.worker_parameters_for_scope(scope_id)
                                ),
                            )
                        except Exception as exc:
                            failure = (
                                scope_id,
                                {
                                    "exception_type": "WorkerProtocolError",
                                    "message": (
                                        "cleanly exited worker sent no completion "
                                        "message and its terminal attempt failed "
                                        f"authentication: {type(exc).__name__}: {exc}"
                                    ),
                                },
                            )
                            self.ledger.update(
                                scope_id,
                                "failed",
                                failure=failure[1],
                            )
                            continue
                        authenticated = ValidatedStage1ScopeAttempt(
                            scope_id=scope_id,
                            attempt_dir=Path(request.attempt_dir),
                            manifest=manifest,
                        )
                        completed[scope_id] = authenticated
                        self.ledger.reconcile_authenticated_completion(
                            authenticated
                        )
                        continue
                    del active[slot]
                    clean_exit_seen_at.pop(scope_id, None)
                    manifest = self.store.validate_completed(
                        request.attempt_dir,
                        scope_id=scope_id,
                        worker_target=self.worker_target,
                        worker_parameters=self.worker_parameters_for_scope(
                            scope_id
                        ),
                    )
                    completion = completion_messages.pop(scope_id, None)
                    if completion is None:
                        failure = (
                            scope_id,
                            {
                                "exception_type": "WorkerProtocolError",
                                "message": "worker exited without a completion message",
                            },
                        )
                        self.ledger.update(
                            scope_id,
                            "failed",
                            failure=failure[1],
                        )
                        continue
                    completed[scope_id] = ValidatedStage1ScopeAttempt(
                        scope_id=scope_id,
                        attempt_dir=Path(request.attempt_dir),
                        manifest=manifest,
                    )
                    self.ledger.update(
                        scope_id,
                        "completed",
                        pid=int(completion["pid"]),
                        elapsed_seconds=float(completion["elapsed_seconds"]),
                        peak_gpu_allocated_bytes=completion[
                            "peak_gpu_allocated_bytes"
                        ],
                        peak_gpu_reserved_bytes=completion[
                            "peak_gpu_reserved_bytes"
                        ],
                        output_bytes=int(completion["output_bytes"]),
                        throughput_fit_rows_per_second=float(
                            completion["throughput_fit_rows_per_second"]
                        ),
                        failure=None,
                    )
                if failure is not None:
                    break
        except BaseException as exc:
            interruption = exc
            raise
        finally:
            if failure is not None or interruption is not None:
                cleanup_failure = (
                    {
                        "exception_type": "ParentInterruption",
                        "message": (
                            f"{type(interruption).__name__}: {interruption}"
                        ),
                    }
                    if interruption is not None
                    else {
                        "exception_type": "PeerScopeFailure",
                        "message": "terminated after another scope failed",
                    }
                )
                for scope_id, process, _request in active.values():
                    _terminate_process_and_descendants(
                        process,
                        process_group_marker_path=(
                            Path(_request.attempt_dir)
                            / "process_group_ready.json"
                        ),
                    )
                    try:
                        snapshot = self.ledger.snapshot()
                        status = next(
                            row["status"]
                            for row in snapshot["scopes"]
                            if row["scope_id"] == scope_id
                        )
                        if status in {"running", "sealing"}:
                            self.ledger.update(
                                scope_id,
                                "failed",
                                failure=cleanup_failure,
                            )
                    except BaseException:
                        # Child termination has priority. A malformed progress
                        # ledger is non-authoritative and will fail closed when
                        # the identical command is resumed.
                        pass
                active.clear()
            try:
                messages.close()
                messages.join_thread()
            finally:
                if sigterm_handler_installed:
                    signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if failure is not None:
            raise RuntimeError(
                f"Stage 1 scope failed: {failure[0]}: "
                f"{failure[1].get('exception_type')}: {failure[1].get('message')}"
            )
        expected_owners = {
            scope.scope_id for scope in self.plan.physical_scopes
        }
        if set(completed) != expected_owners:
            missing = sorted(
                expected_owners - set(completed)
            )
            raise RuntimeError(
                "Stage 1 physical-fit orchestration ended with incomplete coverage: "
                + ", ".join(missing)
            )
        ordered = tuple(
            completed[scope.scope_id] for scope in self.plan.physical_scopes
        )
        bindings = self.store.seal_logical_bindings(ordered)
        self.ledger.reconcile_logical_references(bindings)
        return ordered


__all__ = [
    "STAGE1_LOGICAL_SCOPE_BINDING_FILENAME",
    "STAGE1_LOGICAL_SCOPE_BINDING_SCHEMA",
    "STAGE1_LOGICAL_SCOPE_BINDING_SET_SCHEMA",
    "STAGE1_SCOPE_ATTEMPT_MANIFEST_SCHEMA",
    "STAGE1_SCOPE_ATTEMPT_REQUEST_SCHEMA",
    "STAGE1_SCOPE_PLAN_SCHEMA",
    "STAGE1_SCOPE_PROGRESS_SCHEMA",
    "STAGE1_TORCH_DETERMINISM_POLICY_SCHEMA",
    "SpawnedStage1ScopeOrchestrator",
    "Stage1ScopeAssignment",
    "Stage1ScopeAttemptStore",
    "Stage1ScopeExecutionRequest",
    "Stage1ScopePlan",
    "Stage1PhysicalFitIdentity",
    "Stage1ScopeProgressLedger",
    "Stage1ScopeSpec",
    "ValidatedStage1LogicalScopeBindings",
    "ValidatedStage1ScopeAttempt",
    "build_canonical_stage1_scope_plan",
    "derive_stage1_group_seed",
    "derive_stage1_scope_seed",
    "seed_stage1_scope_rngs",
    "stage1_torch_determinism_policy",
    "validate_stage1_scope_plan",
    "write_stage1_scope_plan",
]
