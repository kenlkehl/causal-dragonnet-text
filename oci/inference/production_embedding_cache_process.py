"""Short-lived process boundary for fresh production embedding-cache builds.

Loading the sentence encoder in the long-lived workflow process can retain
Python, Torch, and CUDA allocations after the cache has been published.  This
module keeps model construction inside one ``spawn`` child.  The parent
receives only closed JSON, waits for the worker process to exit, and then
freshly validates the published cache bytes without loading the model.

Multiprocessing transport is ephemeral only.  No pickle is persisted as a
scientific artifact.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import re
import resource
import stat
import sys
import sysconfig
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .production_stage1_scope_scheduler import (
    _establish_worker_process_group,
    _terminate_process_and_descendants,
)


SPAWNED_EMBEDDING_CACHE_BUILD_SCHEMA = (
    "production_embedding_cache_spawn_build_v1"
)
SPAWNED_EMBEDDING_CACHE_EXECUTION_SCHEMA = (
    "production_embedding_cache_spawn_execution_v1"
)
PRODUCTION_EMBEDDING_CACHE_WORKER_TARGET = (
    "oci.inference.production_embedding_cache_process:"
    "_production_embedding_cache_build_target"
)

_TARGET = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
)
_START_ENVIRONMENT_LOCK = threading.Lock()
_NATIVE_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


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


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TypeError(f"{label} must be closed finite JSON") from exc


def _resolve_target(value: str) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    target = str(value)
    if _TARGET.fullmatch(target) is None:
        raise ValueError("embedding-cache worker target is malformed")
    module_name, attribute = target.split(":", 1)
    resolved = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(resolved):
        raise TypeError("embedding-cache worker target is not callable")
    return resolved


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _native_thread_environment(cpu_budget: int) -> dict[str, str]:
    count = _positive_integer(cpu_budget, label="embedding-cache CPU budget")
    return {name: str(count) for name in _NATIVE_THREAD_ENVIRONMENT}


def _active_environment_runtime_library_directories() -> tuple[Path, ...]:
    """Find packaged native-library directories owned by this interpreter."""

    prefix = Path(sys.prefix).resolve()
    site_roots: list[Path] = []
    configured_paths = sysconfig.get_paths()
    for name in ("purelib", "platlib"):
        raw = configured_paths.get(name)
        if not raw:
            continue
        candidate = Path(raw).resolve()
        if (
            candidate.is_dir()
            and candidate.is_relative_to(prefix)
            and candidate not in site_roots
        ):
            site_roots.append(candidate)

    discovered: list[Path] = []

    def add(candidate: Path) -> None:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            return
        if (
            resolved.is_dir()
            and resolved.is_relative_to(prefix)
            and resolved not in discovered
        ):
            discovered.append(resolved)

    for site_root in site_roots:
        add(site_root / "PyNvVideoCodec")
        add(site_root / "torch" / "lib")
        nvidia = site_root / "nvidia"
        # Prefer the active CUDA-major bundle before dependency packages that
        # may also contain libraries for another installed CUDA major.
        if nvidia.is_dir():
            cuda_major_bundles = sorted(
                (
                    path
                    for path in nvidia.glob("cu*/lib")
                    if path.parent.name[2:].isdigit()
                ),
                key=lambda path: int(path.parent.name[2:]),
                reverse=True,
            )
            for bundle in cuda_major_bundles:
                add(bundle)
            for dependency_lib in sorted(nvidia.glob("*/lib")):
                add(dependency_lib)
    return tuple(discovered)


def _spawn_environment(cpu_budget: int) -> dict[str, str]:
    replacements = _native_thread_environment(cpu_budget)
    runtime_libraries = _active_environment_runtime_library_directories()
    if not runtime_libraries:
        return replacements
    entries = [str(path) for path in runtime_libraries]
    entries.extend(
        entry
        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if entry
    )
    replacements["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(entries))
    return replacements


def _process_io_counters() -> dict[str, int] | None:
    path = Path("/proc/self/io")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return None
    parsed: dict[str, int] = {}
    for line in lines:
        key, separator, raw = line.partition(":")
        if not separator or not raw.strip().isdigit():
            return None
        parsed[key.strip()] = int(raw.strip())
    required = {"rchar", "wchar", "read_bytes", "write_bytes"}
    if not required.issubset(parsed):
        return None
    return {name: parsed[name] for name in sorted(required)}


def _process_io_delta(
    before: Mapping[str, int] | None,
    after: Mapping[str, int] | None,
) -> dict[str, int] | None:
    if before is None or after is None or set(before) != set(after):
        return None
    return {
        name: max(0, int(after[name]) - int(before[name]))
        for name in sorted(before)
    }


def _gpu_peak(device: Any) -> dict[str, int | None]:
    if not isinstance(device, str) or not device.startswith("cuda"):
        return {
            "peak_gpu_allocated_bytes": None,
            "peak_gpu_reserved_bytes": None,
        }
    try:
        import torch

        index = (
            int(device.split(":", 1)[1])
            if ":" in device
            else int(torch.cuda.current_device())
        )
        return {
            "peak_gpu_allocated_bytes": int(
                torch.cuda.max_memory_allocated(index)
            ),
            "peak_gpu_reserved_bytes": int(
                torch.cuda.max_memory_reserved(index)
            ),
        }
    except (ImportError, RuntimeError, ValueError):
        # A successful cache build may use a backend whose peak allocator
        # counters are unavailable. The absence is operational telemetry, not
        # a scientific substitute.
        return {
            "peak_gpu_allocated_bytes": None,
            "peak_gpu_reserved_bytes": None,
        }


def _production_embedding_cache_build_target(
    parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    required = {
        "dataset_path",
        "text_column",
        "local_model_path",
        "sentence_model_name",
        "chunk_configuration",
        "target_dir",
        "device",
        "batch_size",
    }
    if not isinstance(parameters, Mapping) or set(parameters) != required:
        raise ValueError(
            "production embedding-cache worker parameters are not closed"
        )
    from .production_embedding_cache_builder import (
        build_production_embedding_cache,
    )

    result = build_production_embedding_cache(
        dataset_path=Path(str(parameters["dataset_path"])),
        text_column=str(parameters["text_column"]),
        local_model_path=Path(str(parameters["local_model_path"])),
        sentence_model_name=str(parameters["sentence_model_name"]),
        chunk_configuration=copy.deepcopy(
            dict(parameters["chunk_configuration"])
        ),
        target_dir=Path(str(parameters["target_dir"])),
        device=parameters["device"],
        batch_size=_positive_integer(
            parameters["batch_size"],
            label="embedding-cache encode batch size",
        ),
    )
    # The constructor already authenticated the just-published tree. Avoid a
    # redundant child-side replay merely to detach its closed identity; the
    # parent performs the fresh trust-boundary validation below.
    identity = _json_copy(
        dict(
            result._identity  # noqa: SLF001 - same-package authenticated handle
        ),
        label="embedding-cache child build identity",
    )
    return {
        "schema_version": SPAWNED_EMBEDDING_CACHE_BUILD_SCHEMA,
        "cache_path": str(result.cache_path),
        "build_identity": identity,
        "model_materialized_in_worker_process": True,
        "model_materialized_in_parent_process": False,
    }


def _worker_entry(
    *,
    worker_target: str,
    worker_parameters: Mapping[str, Any],
    cpu_budget: int,
    process_group_marker_path: str,
    connection: Any,
) -> None:
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    io_before = _process_io_counters()
    try:
        os.environ.update(_native_thread_environment(cpu_budget))
        _establish_worker_process_group(process_group_marker_path)
        target = _resolve_target(worker_target)
        result = _json_copy(
            target(
                _json_copy(
                    worker_parameters,
                    label="embedding-cache worker parameters",
                )
            ),
            label="embedding-cache worker result",
        )
        usage_after = resource.getrusage(resource.RUSAGE_SELF)
        io_after = _process_io_counters()
        device = worker_parameters.get("device")
        telemetry = {
            "worker_pid": int(os.getpid()),
            "wall_seconds": max(0.0, time.monotonic() - started_wall),
            "cpu_seconds": max(0.0, time.process_time() - started_cpu),
            "filesystem_input_blocks": max(
                0,
                int(usage_after.ru_inblock - usage_before.ru_inblock),
            ),
            "filesystem_output_blocks": max(
                0,
                int(usage_after.ru_oublock - usage_before.ru_oublock),
            ),
            "process_io_deltas": _process_io_delta(io_before, io_after),
            "peak_resident_kib": max(0, int(usage_after.ru_maxrss)),
            **_gpu_peak(device),
        }
        connection.send(
            {
                "status": "completed",
                "result": result,
                "telemetry": telemetry,
            }
        )
    except BaseException as exc:
        try:
            connection.send(
                {
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            raise
    finally:
        connection.close()


def _start_with_environment(
    process: mp.Process,
    *,
    cpu_budget: int,
) -> None:
    replacements = _spawn_environment(cpu_budget)
    with _START_ENVIRONMENT_LOCK:
        prior = {name: os.environ.get(name) for name in replacements}
        try:
            os.environ.update(replacements)
            process.start()
        finally:
            for name, value in prior.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _run_spawned_target(
    *,
    worker_target: str,
    worker_parameters: Mapping[str, Any],
    cpu_budget: int,
) -> Mapping[str, Any]:
    """Execute one importable target and return only after its child exits."""

    _resolve_target(worker_target)
    parameters = _json_copy(
        worker_parameters,
        label="embedding-cache worker parameters",
    )
    budget = _positive_integer(
        cpu_budget,
        label="embedding-cache CPU budget",
    )
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    message: Mapping[str, Any] | None = None
    with tempfile.TemporaryDirectory(
        prefix="production-embedding-cache-worker-"
    ) as control_root:
        marker = Path(control_root) / "process_group_ready.json"
        process = context.Process(
            target=_worker_entry,
            kwargs={
                "worker_target": worker_target,
                "worker_parameters": parameters,
                "cpu_budget": budget,
                "process_group_marker_path": str(marker),
                "connection": child_connection,
            },
            name="production-embedding-cache-builder",
            daemon=False,
        )
        try:
            _start_with_environment(process, cpu_budget=budget)
            child_connection.close()
            while process.is_alive():
                if parent_connection.poll(0.1):
                    message = parent_connection.recv()
                    break
            if message is None and parent_connection.poll():
                message = parent_connection.recv()
            process.join()
            if process.is_alive():
                raise RuntimeError(
                    "embedding-cache worker remained alive after join"
                )
            if (
                message is None
                or not isinstance(message, Mapping)
                or message.get("status") not in {"completed", "failed"}
            ):
                raise RuntimeError(
                    "embedding-cache worker exited without a closed result"
                )
            if message.get("status") == "failed":
                raise RuntimeError(
                    "embedding-cache worker failed: "
                    f"{message.get('exception_type')}: "
                    f"{message.get('message')}\n"
                    f"{message.get('traceback')}"
                )
            if process.exitcode != 0:
                raise RuntimeError(
                    "embedding-cache worker reported success but exited "
                    f"with code {process.exitcode}"
                )
        except BaseException:
            if process.pid is not None and process.is_alive():
                _terminate_process_and_descendants(
                    process,
                    process_group_marker_path=marker,
                )
            raise
        finally:
            parent_connection.close()
            try:
                child_connection.close()
            except OSError:
                pass
    assert message is not None
    return _json_copy(message, label="embedding-cache worker message")


def _stat_inventory(
    cache_path: Path,
    *,
    identity: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    registrations = identity.get("cache_files")
    if not isinstance(registrations, Mapping) or not registrations:
        raise ValueError(
            "validated embedding-cache identity has no file inventory"
        )
    result: dict[str, tuple[int, ...]] = {}
    for name in sorted(registrations):
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
        ):
            raise ValueError(
                "validated embedding-cache inventory name is unsafe"
            )
        path = cache_path / name
        value = os.lstat(path)
        if (
            stat.S_ISLNK(value.st_mode)
            or not stat.S_ISREG(value.st_mode)
            or int(value.st_nlink) != 1
        ):
            raise ValueError(
                "validated embedding-cache payload is not private data"
            )
        result[name] = (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_mode),
            int(value.st_nlink),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )
    return result


@dataclass(frozen=True)
class SpawnedProductionEmbeddingCacheBuildResult:
    """Parent-authenticated result guarded by a same-process stat inventory."""

    cache_path: Path
    _identity: Mapping[str, Any] = field(repr=False)
    execution_attestation: Mapping[str, Any]
    _authenticated_stats: Mapping[str, tuple[int, ...]] = field(repr=False)

    def __post_init__(self) -> None:
        path = Path(self.cache_path).resolve(strict=True)
        identity = _json_copy(
            self._identity,
            label="spawned embedding-cache validated identity",
        )
        attestation = _json_copy(
            self.execution_attestation,
            label="spawned embedding-cache execution attestation",
        )
        stats = {
            str(name): tuple(int(value) for value in values)
            for name, values in self._authenticated_stats.items()
        }
        if (
            attestation.get("schema_version")
            != SPAWNED_EMBEDDING_CACHE_EXECUTION_SCHEMA
            or attestation.get("content_sha256")
            != _sha256_json(
                {
                    key: value
                    for key, value in attestation.items()
                    if key != "content_sha256"
                }
            )
            or attestation.get("worker_exit_confirmed") is not True
            or attestation.get("model_materialized_in_parent_process")
            is not False
            or attestation.get("parent_fresh_byte_validation") is not True
            or str(identity.get("cache_path")) != str(path)
            or set(stats) != set(identity.get("cache_files") or {})
        ):
            raise ValueError(
                "spawned embedding-cache result is not closed and validated"
            )
        object.__setattr__(self, "cache_path", path)
        object.__setattr__(self, "_identity", MappingProxyType(identity))
        object.__setattr__(
            self,
            "execution_attestation",
            MappingProxyType(attestation),
        )
        object.__setattr__(
            self,
            "_authenticated_stats",
            MappingProxyType(stats),
        )

    def identity(self) -> dict[str, Any]:
        for name, expected in self._authenticated_stats.items():
            value = os.lstat(self.cache_path / name)
            observed = (
                int(value.st_dev),
                int(value.st_ino),
                int(value.st_mode),
                int(value.st_nlink),
                int(value.st_size),
                int(value.st_mtime_ns),
                int(value.st_ctime_ns),
            )
            if observed != expected:
                raise RuntimeError(
                    "spawned embedding-cache bytes changed after parent "
                    f"authentication: {name}"
                )
        return copy.deepcopy(dict(self._identity))


def build_production_embedding_cache_in_spawned_worker(
    *,
    dataset_path: Path | str,
    text_column: str,
    local_model_path: Path | str,
    sentence_model_name: str,
    chunk_configuration: Mapping[str, Any],
    target_dir: Path | str,
    device: str | None,
    batch_size: int,
    cpu_budget: int,
) -> SpawnedProductionEmbeddingCacheBuildResult:
    """Build in a short-lived child and freshly authenticate in the parent."""

    target = Path(target_dir)
    if not target.is_absolute():
        raise ValueError(
            "spawned embedding-cache target must be an absolute path"
        )
    parameters = {
        "dataset_path": str(Path(dataset_path)),
        "text_column": str(text_column),
        "local_model_path": str(Path(local_model_path)),
        "sentence_model_name": str(sentence_model_name),
        "chunk_configuration": _json_copy(
            chunk_configuration,
            label="embedding-cache chunk configuration",
        ),
        "target_dir": str(target),
        "device": device,
        "batch_size": _positive_integer(
            batch_size,
            label="embedding-cache encode batch size",
        ),
    }
    started = time.monotonic()
    message = _run_spawned_target(
        worker_target=PRODUCTION_EMBEDDING_CACHE_WORKER_TARGET,
        worker_parameters=parameters,
        cpu_budget=cpu_budget,
    )
    child_result = message.get("result")
    telemetry = message.get("telemetry")
    if (
        not isinstance(child_result, Mapping)
        or set(child_result)
        != {
            "schema_version",
            "cache_path",
            "build_identity",
            "model_materialized_in_worker_process",
            "model_materialized_in_parent_process",
        }
        or child_result.get("schema_version")
        != SPAWNED_EMBEDDING_CACHE_BUILD_SCHEMA
        or child_result.get("cache_path") != str(target.resolve(strict=True))
        or child_result.get("model_materialized_in_worker_process")
        is not True
        or child_result.get("model_materialized_in_parent_process")
        is not False
        or not isinstance(child_result.get("build_identity"), Mapping)
        or not isinstance(telemetry, Mapping)
    ):
        raise RuntimeError(
            "embedding-cache worker returned an invalid success envelope"
        )
    from .production_embedding_cache_builder import (
        validate_published_production_embedding_cache,
    )

    validated = validate_published_production_embedding_cache(
        cache_dir=target,
        dataset_path=Path(dataset_path),
        text_column=text_column,
        sentence_model_name=sentence_model_name,
        chunk_configuration=chunk_configuration,
        expected_local_model_path=Path(local_model_path),
    )
    child_identity = _json_copy(
        child_result["build_identity"],
        label="embedding-cache child build identity",
    )
    if child_identity != validated:
        raise RuntimeError(
            "parent-authenticated embedding cache differs from the child "
            "build identity"
        )
    stats = _stat_inventory(target.resolve(strict=True), identity=validated)
    attestation_body = {
        "schema_version": SPAWNED_EMBEDDING_CACHE_EXECUTION_SCHEMA,
        "worker_target": PRODUCTION_EMBEDDING_CACHE_WORKER_TARGET,
        "worker_start_method": "spawn",
        "worker_exit_confirmed": True,
        "model_materialized_in_worker_process": True,
        "model_materialized_in_parent_process": False,
        "parent_fresh_byte_validation": True,
        "cpu_budget": _positive_integer(
            cpu_budget,
            label="embedding-cache CPU budget",
        ),
        "parent_elapsed_seconds": max(0.0, time.monotonic() - started),
        "worker_telemetry": _json_copy(
            telemetry,
            label="embedding-cache worker telemetry",
        ),
    }
    if not math.isfinite(float(attestation_body["parent_elapsed_seconds"])):
        raise RuntimeError(
            "embedding-cache worker elapsed time is not finite"
        )
    attestation = {
        **attestation_body,
        "content_sha256": _sha256_json(attestation_body),
    }
    return SpawnedProductionEmbeddingCacheBuildResult(
        cache_path=target,
        _identity=validated,
        execution_attestation=attestation,
        _authenticated_stats=stats,
    )


__all__ = [
    "PRODUCTION_EMBEDDING_CACHE_WORKER_TARGET",
    "SPAWNED_EMBEDDING_CACHE_BUILD_SCHEMA",
    "SPAWNED_EMBEDDING_CACHE_EXECUTION_SCHEMA",
    "SpawnedProductionEmbeddingCacheBuildResult",
    "build_production_embedding_cache_in_spawned_worker",
]
