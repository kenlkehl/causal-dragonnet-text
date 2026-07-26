"""Durable, compact publication for a completed role-neutral benchmark.

The performance benchmark deliberately executes on local POSIX scratch.  Its
complete fit replicas are useful at the fresh terminal trust boundary, but
copying all of them to a durable (often networked) artifact root would recreate
the proof-amplification problem that the benchmark is intended to measure.

This module therefore publishes two deliberately separate records:

* byte-exact benchmark result, request, and checkpoint JSON files.  These retain
  historical scratch locators and are explicitly non-authoritative after
  publication; and
* path-neutral checkpoint/result evidence plus one canonical scientific
  artifact manifest and full tree inventory.  This is the durable authority.

All complete scratch trees are reopened and matched to their checkpoint and
terminal-audit registrations before publication.  They are not retained.  A
future reader can authenticate the durable evidence and the one canonical
manifest, but cannot replay the omitted model artifacts.  The manifest states
that limitation rather than claiming otherwise.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .compact_preflight_compression_benchmark import (
    publish_compact_preflight_compression_benchmark_result,
    validate_compact_preflight_compression_benchmark_result,
)
from .portable_workflow_spec import identity_sha256
from .production_stage1_role_neutral_execution import (
    ROLE_NEUTRAL_EXECUTION_MANIFEST,
    ROLE_NEUTRAL_STAGE1_EXECUTION_SCHEMA,
)
from .role_neutral_performance_benchmark import (
    ROLE_NEUTRAL_BENCHMARK_INTERRUPTED_OBSERVATION_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_OBSERVATION_CHECKPOINT_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_REQUEST_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_RESULT_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_WORKLOAD_BINDING_SCHEMA,
    RoleNeutralBenchmarkConfig,
    RoleNeutralBenchmarkSourceBinding,
)
from .role_neutral_benchmark_workload_provider import (
    RoleNeutralBenchmarkWorkloadDeployment,
    _authenticate_paused_stage1_preflight,
)


ROLE_NEUTRAL_BENCHMARK_PUBLICATION_SCHEMA = (
    "portable_role_neutral_performance_benchmark_publication_v1"
)
ROLE_NEUTRAL_BENCHMARK_LOGICAL_CHECKPOINT_SCHEMA = (
    "portable_role_neutral_benchmark_logical_checkpoint_proof_v1"
)
ROLE_NEUTRAL_BENCHMARK_PATH_NEUTRAL_RESULT_SCHEMA = (
    "portable_role_neutral_benchmark_path_neutral_result_v1"
)
ROLE_NEUTRAL_BENCHMARK_CANONICAL_ARTIFACT_REFERENCE_SCHEMA = (
    "portable_role_neutral_benchmark_canonical_artifact_reference_v1"
)
ROLE_NEUTRAL_BENCHMARK_SCIENTIFIC_WORKFLOW_BINDING_SCHEMA = (
    "portable_role_neutral_benchmark_scientific_workflow_binding_v1"
)
ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST = "publication_manifest.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_NAME = re.compile(r"^observation_([0-9]{6})\.json$")
_INTERRUPTED_NAME = re.compile(
    r"^observation_([0-9]{6})_attempt_([0-9]{3})\.json$"
)
_ABSOLUTE_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")

_BENCHMARK_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "config",
        "config_sha256",
        "workload_binding",
        "resource_inventory",
        "execution_schedule",
        "warmup_observations",
        "warmup_telemetry",
        "warmup_observations_excluded_from_selection",
        "benchmark_observations",
        "observation_telemetry",
        "terminal_audit",
        "terminal_audit_telemetry",
        "ordinary_observations_exclude_terminal_audit",
        "candidate_results",
        "preflight_compression_benchmark",
        "benchmark_matrix_coverage",
        "selected_candidate",
        "selection_policy",
        "scientific_result_identity_sha256",
        "accepted",
        "content_sha256",
    }
)
_BENCHMARK_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "config",
        "config_sha256",
        "workload_binding",
        "immutable_inputs_by_scope",
        "compression_source",
        "resource_resume_compatibility",
        "candidate_device_assignments",
        "execution_schedule",
        "producer_code_evidence",
        "content_sha256",
    }
)
_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "request_sha256",
        "schedule_entry",
        "observation",
        "detail",
        "observation_tree",
        "complete_artifacts",
        "content_sha256",
    }
)
_INTERRUPTED_FIELDS = frozenset(
    {
        "schema_version",
        "request_sha256",
        "schedule_entry",
        "preserved_relative_root",
        "tree_sha256",
        "total_file_bytes",
        "file_count",
        "content_sha256",
    }
)
_BENCHMARK_BINDING_FIELDS = frozenset(
    {
        "publication_producer_code_sha256",
        "benchmark_result",
        "benchmark_request",
        "config_sha256",
        "workload_binding_content_sha256",
        "execution_schedule_content_sha256",
        "scientific_result_identity_sha256",
        "selected_candidate",
        "workflow_request_sha256",
        "workflow_scientific_sha256",
        "workload_deployment_sha256",
        "stage1_preflight_phase_content_sha256",
        "prepared_stage1_context_content_root_sha256",
        "scientific_workflow_binding_content_sha256",
    }
)
_PATH_NEUTRAL_NORMALIZED_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "config",
        "config_sha256",
        "scientific_workload_binding",
        "execution_schedule",
        "candidate_results",
        "benchmark_matrix_coverage",
        "selected_candidate",
        "selection_policy",
        "scientific_result_identity_sha256",
        "accepted",
        "warmup_observations_excluded_from_selection",
        "ordinary_observations_exclude_terminal_audit",
        "preflight_compression_benchmark",
        "historical_result_content_identity_retained",
        "physical_observation_and_terminal_audit_records_retained",
    }
)


def _strict_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(
                f"benchmark publication JSON repeats key {key!r}"
            )
        output[key] = value
    return output


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _canonical_private_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    state = os.lstat(path)
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISDIR(state.st_mode)
        or path.resolve(strict=True) != path
    ):
        raise ValueError(f"{label} must be canonical and symlink-free")
    return path


def _read_private_bytes(path: Path, *, label: str) -> tuple[bytes, str]:
    before = os.lstat(path)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise ValueError(f"{label} must be one private regular file")
    canonical = path.resolve(strict=True)
    if canonical != Path(os.path.abspath(os.fspath(path))):
        raise ValueError(f"{label} path must be canonical and symlink-free")
    descriptor = os.open(
        canonical,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    blocks: list[bytes] = []
    try:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            blocks.append(block)
    finally:
        os.close(descriptor)
    after = os.lstat(canonical)
    payload = b"".join(blocks)
    if (
        _stat_identity(before) != _stat_identity(after)
        or len(payload) != int(before.st_size)
    ):
        raise RuntimeError(f"{label} changed while it was being read")
    return payload, digest.hexdigest()


def _decode_closed_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not closed UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _read_private_json(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], str, int]:
    payload, digest = _read_private_bytes(path, label=label)
    return _decode_closed_json(payload, label=label), digest, len(payload)


def _closed_content_identity(
    value: Mapping[str, Any],
    *,
    label: str,
) -> str:
    supplied = _require_sha256(
        value.get("content_sha256"),
        label=f"{label} content identity",
    )
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    if supplied != identity_sha256(body):
        raise ValueError(f"{label} content identity is invalid")
    return supplied


def _safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    relative = PurePosixPath(str(value))
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or "." in relative.parts
        or "\x00" in str(value)
    ):
        raise ValueError(f"{label} must be one traversal-safe relative path")
    return relative


def _inventory(
    root: Path,
    *,
    exclude_relative_paths: Sequence[str] = (),
    require_read_only: bool,
) -> tuple[list[dict[str, Any]], str, int]:
    excluded = set(exclude_relative_paths)
    rows: list[dict[str, Any]] = []
    total = 0
    for path in sorted(
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        state = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(state.st_mode):
            raise ValueError(
                "benchmark publication inventory encountered a symlink"
            )
        if stat.S_ISDIR(state.st_mode):
            continue
        if (
            not stat.S_ISREG(state.st_mode)
            or int(state.st_nlink) != 1
            or (
                require_read_only
                and stat.S_IMODE(state.st_mode) & 0o222
            )
        ):
            raise ValueError(
                "benchmark publication inventory requires private "
                "read-only regular files"
            )
        if relative in excluded:
            continue
        payload, digest = _read_private_bytes(
            path,
            label=f"benchmark publication payload {relative!r}",
        )
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": len(payload),
                "sha256": digest,
            }
        )
        total += len(payload)
    return rows, identity_sha256(rows), total


def _subtree_inventory(
    inventory: Sequence[Mapping[str, Any]],
    *,
    relative_root: PurePosixPath,
) -> list[dict[str, Any]]:
    prefix = relative_root.as_posix() + "/"
    rows: list[dict[str, Any]] = []
    for row in inventory:
        path = str(row["relative_path"])
        if not path.startswith(prefix):
            continue
        rows.append(
            {
                "relative_path": path[len(prefix) :],
                "size_bytes": int(row["size_bytes"]),
                "sha256": str(row["sha256"]),
            }
        )
    return rows


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
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
                raise OSError(
                    "benchmark publication JSON write made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _copy_once(
    source: Path,
    destination: Path,
    *,
    label: str,
) -> dict[str, Any]:
    before = os.lstat(source)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise ValueError(f"{label} must be private regular source data")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_descriptor = os.open(
        source,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    destination_descriptor = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_descriptor, view)
                if written < 1:
                    raise OSError(
                        "benchmark publication copy made no progress"
                    )
                view = view[written:]
        os.fsync(destination_descriptor)
    finally:
        os.close(source_descriptor)
        os.close(destination_descriptor)
    after = os.lstat(source)
    if _stat_identity(before) != _stat_identity(after) or size != before.st_size:
        raise RuntimeError(f"{label} changed while it was copied")
    os.chmod(destination, 0o444)
    _fsync_directory(destination.parent)
    return {
        "relative_path": destination.name,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publication_producer_code_sha256() -> str:
    path = Path(__file__).resolve(strict=True)
    before = os.stat(path)
    payload = path.read_bytes()
    after = os.stat(path)
    if not payload:
        raise RuntimeError("benchmark publication producer code is empty")
    if (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    ) != (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    ):
        raise RuntimeError(
            "benchmark publication producer code changed while hashing"
        )
    return hashlib.sha256(payload).hexdigest()


def _redact_absolute_locators(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _redact_absolute_locators(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_absolute_locators(child) for child in value]
    if isinstance(value, tuple):
        return [_redact_absolute_locators(child) for child in value]
    if isinstance(value, str) and (
        value.startswith("/") or _ABSOLUTE_WINDOWS_PATH.match(value)
    ):
        return "__historical_absolute_locator_redacted__"
    return copy.deepcopy(value)


def _logical_observation_root(entry: Mapping[str, Any]) -> str:
    kind = str(entry["observation_kind"])
    index = int(entry["observation_index"])
    return (
        Path("warmups" if kind == "warmup" else "runs")
        / str(entry["candidate_name"])
        / str(entry["scope_label"])
        / (
            f"warmup_{index:03d}"
            if kind == "warmup"
            else f"repetition_{index:03d}"
        )
    ).as_posix()


def _compression_path_neutral_evidence(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_compact_preflight_compression_benchmark_result(
        value,
        reopen_artifacts=False,
    )
    source = validated["source"]
    body = {
        "config": copy.deepcopy(validated["config"]),
        "execution_schedule": copy.deepcopy(
            validated["execution_schedule"]
        ),
        "source_path_neutral_scientific_content_sha256": source[
            "path_neutral_scientific_content_sha256"
        ],
        "warmup_observations": [
            {
                key: copy.deepcopy(child)
                for key, child in row.items()
                if key
                not in {
                    "artifact_manifest_path",
                    "artifact_content_sha256",
                    "content_sha256",
                }
            }
            for row in validated["warmup_observations"]
        ],
        "measured_observations": [
            {
                key: copy.deepcopy(child)
                for key, child in row.items()
                if key
                not in {
                    "artifact_manifest_path",
                    "artifact_content_sha256",
                    "content_sha256",
                }
            }
            for row in validated["measured_observations"]
        ],
        "codec_results": copy.deepcopy(validated["codec_results"]),
        "selected_parquet_compression": validated[
            "selected_parquet_compression"
        ],
        "selection_policy": validated["selection_policy"],
        "accepted": validated["accepted"],
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _validate_source_result(
    path: Path,
) -> tuple[dict[str, Any], str, int, RoleNeutralBenchmarkConfig]:
    # This existing fresh validator checks the current closed result schema,
    # benchmark matrix, and all compact-preflight replica bytes.  Importing it
    # lazily avoids making durable validation depend on historical locators.
    from .role_neutral_benchmark_deployment_selection import _read_result

    result, file_sha256, config = _read_result(path)
    payload, independently_hashed = _read_private_bytes(
        path,
        label="completed benchmark result",
    )
    if independently_hashed != file_sha256:
        raise RuntimeError("benchmark result changed across fresh validation")
    return result, file_sha256, len(payload), config


def _validate_result_record_without_historical_reopen(
    value: Mapping[str, Any],
) -> RoleNeutralBenchmarkConfig:
    if (
        not isinstance(value, Mapping)
        or set(value) != _BENCHMARK_RESULT_FIELDS
        or value.get("schema_version") != ROLE_NEUTRAL_BENCHMARK_RESULT_SCHEMA
    ):
        raise ValueError("published benchmark result schema is invalid")
    _closed_content_identity(value, label="published benchmark result")
    config = RoleNeutralBenchmarkConfig.from_mapping(value["config"])
    if (
        value.get("config_sha256") != identity_sha256(value["config"])
        or value.get("status") != "complete"
        or value.get("accepted") is not True
        or value.get("ordinary_observations_exclude_terminal_audit") is not True
        or value.get("warmup_observations_excluded_from_selection") is not True
        or _SHA256.fullmatch(
            str(value.get("scientific_result_identity_sha256"))
        )
        is None
    ):
        raise ValueError("published benchmark result envelope is invalid")
    return config


def _validate_workload_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("benchmark workload binding must be an object")
    required = {
        "schema_version",
        "source",
        "representative_scope_plans",
        "content_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version")
        != ROLE_NEUTRAL_BENCHMARK_WORKLOAD_BINDING_SCHEMA
    ):
        raise ValueError("benchmark workload binding schema is invalid")
    _closed_content_identity(value, label="benchmark workload binding")
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("benchmark workload source binding is invalid")
    RoleNeutralBenchmarkSourceBinding(**dict(source))
    plans = value.get("representative_scope_plans")
    if not isinstance(plans, list) or not plans:
        raise ValueError("benchmark workload binding lacks scope plans")
    labels: set[str] = set()
    for row in plans:
        if not isinstance(row, Mapping) or set(row) != {
            "scope_label",
            "fit_row_count",
            "plan_scientific_content_sha256",
            "physical_owner_scope_id",
        }:
            raise ValueError("benchmark workload scope-plan row is invalid")
        label = str(row["scope_label"])
        if (
            not label
            or label in labels
            or isinstance(row["fit_row_count"], bool)
            or not isinstance(row["fit_row_count"], int)
            or row["fit_row_count"] < 1
        ):
            raise ValueError("benchmark workload scope-plan coverage is invalid")
        labels.add(label)
        _require_sha256(
            row["plan_scientific_content_sha256"],
            label="benchmark scope-plan identity",
        )
    return copy.deepcopy(dict(value))


def _path_neutral_workload_binding(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    validated = _validate_workload_binding(value)
    source = validated["source"]
    body = {
        "schema_version": (
            "portable_role_neutral_benchmark_scientific_workload_binding_v1"
        ),
        "workflow_scientific_sha256": source[
            "workflow_scientific_sha256"
        ],
        "stage1_preflight_phase_content_sha256": source[
            "stage1_preflight_phase_content_sha256"
        ],
        "prepared_stage1_context_content_root_sha256": source[
            "prepared_stage1_context_content_root_sha256"
        ],
        "representative_scope_plans": copy.deepcopy(
            validated["representative_scope_plans"]
        ),
        "workflow_request_identity_retained": False,
        "workload_deployment_identity_retained": False,
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _scientific_workflow_binding(
    *,
    source: "_ValidatedSource",
    workload_deployment_path: Path,
) -> dict[str, Any]:
    deployment_value, deployment_sha256, _deployment_size = (
        _read_private_json(
            workload_deployment_path,
            label="benchmark workload deployment",
        )
    )
    deployment = RoleNeutralBenchmarkWorkloadDeployment.from_mapping(
        deployment_value
    )
    source_binding = RoleNeutralBenchmarkSourceBinding(
        **dict(source.result["workload_binding"]["source"])
    )
    if (
        deployment_sha256 != source_binding.workload_deployment_sha256
        or deployment.expected_workflow_request_sha256
        != source_binding.workflow_request_sha256
        or deployment.expected_benchmark_config_sha256
        != source.result["config_sha256"]
    ):
        raise ValueError(
            "benchmark workload deployment differs from the completed result"
        )
    authenticated = _authenticate_paused_stage1_preflight(
        deployment,
        require_fresh_prepared_context=False,
    )
    request = authenticated.request
    portable = request.get("portable_scientific_spec")
    configuration = request.get("scientific_configuration_identity")
    phase_code = request.get("phase_producer_code_identities")
    workflow_code = request.get("workflow_producer_code_identity")
    scientific = request.get("scientific_identity")
    spec_source_sha256 = request.get("scientific_spec_source_sha256")
    if (
        request.get("request_sha256")
        != source_binding.workflow_request_sha256
        or not isinstance(portable, Mapping)
        or not isinstance(configuration, Mapping)
        or not isinstance(phase_code, Mapping)
        or not isinstance(scientific, Mapping)
        or _SHA256.fullmatch(str(workflow_code)) is None
        or _SHA256.fullmatch(str(spec_source_sha256)) is None
        or scientific.get("scientific_sha256")
        != source_binding.workflow_scientific_sha256
    ):
        raise ValueError(
            "authenticated benchmark workflow lacks its scientific binding"
        )
    body = {
        "schema_version": (
            ROLE_NEUTRAL_BENCHMARK_SCIENTIFIC_WORKFLOW_BINDING_SCHEMA
        ),
        "portable_scientific_spec": copy.deepcopy(dict(portable)),
        "portable_scientific_spec_sha256": identity_sha256(portable),
        "scientific_spec_source_sha256": str(spec_source_sha256),
        "scientific_configuration_identity": copy.deepcopy(
            dict(configuration)
        ),
        "phase_producer_code_identities": copy.deepcopy(
            dict(phase_code)
        ),
        "workflow_producer_code_identity": str(workflow_code),
        "workflow_scientific_identity": copy.deepcopy(dict(scientific)),
    }
    binding = {**body, "content_sha256": identity_sha256(body)}
    _validate_scientific_workflow_binding(
        binding,
        expected_workflow_scientific_sha256=(
            source_binding.workflow_scientific_sha256
        ),
    )
    return binding


def _validate_scientific_workflow_binding(
    value: Any,
    *,
    expected_workflow_scientific_sha256: str,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "portable_scientific_spec",
        "portable_scientific_spec_sha256",
        "scientific_spec_source_sha256",
        "scientific_configuration_identity",
        "phase_producer_code_identities",
        "workflow_producer_code_identity",
        "workflow_scientific_identity",
        "content_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("schema_version")
        != ROLE_NEUTRAL_BENCHMARK_SCIENTIFIC_WORKFLOW_BINDING_SCHEMA
    ):
        raise ValueError(
            "benchmark scientific workflow binding is not closed"
        )
    _closed_content_identity(
        value,
        label="benchmark scientific workflow binding",
    )
    portable = value["portable_scientific_spec"]
    configuration = value["scientific_configuration_identity"]
    phase_code = value["phase_producer_code_identities"]
    workflow_code = value["workflow_producer_code_identity"]
    scientific = value["workflow_scientific_identity"]
    if (
        not isinstance(portable, Mapping)
        or value["portable_scientific_spec_sha256"]
        != identity_sha256(portable)
        or _SHA256.fullmatch(
            str(value["scientific_spec_source_sha256"])
        )
        is None
        or not isinstance(configuration, Mapping)
        or not isinstance(phase_code, Mapping)
        or not phase_code
        or any(
            _SHA256.fullmatch(str(identity)) is None
            for identity in phase_code.values()
        )
        or _SHA256.fullmatch(str(workflow_code)) is None
        or not isinstance(scientific, Mapping)
    ):
        raise ValueError(
            "benchmark scientific workflow binding identities are invalid"
        )
    configuration_body = {
        key: copy.deepcopy(child)
        for key, child in configuration.items()
        if key != "scientific_configuration_sha256"
    }
    expected_workflow_code = identity_sha256(
        {
            "schema_version": (
                "workflow_phase_producer_code_aggregate_v1"
            ),
            "phase_producer_code_identities": dict(phase_code),
        }
    )
    scientific_body = {
        key: copy.deepcopy(child)
        for key, child in scientific.items()
        if key != "scientific_sha256"
    }
    path_neutral_members = {
        "portable_scientific_spec": copy.deepcopy(dict(portable)),
        "scientific_configuration_identity": copy.deepcopy(
            dict(configuration)
        ),
        "phase_producer_code_identities": copy.deepcopy(
            dict(phase_code)
        ),
        "workflow_scientific_identity": copy.deepcopy(dict(scientific)),
    }
    if (
        configuration.get("scientific_configuration_sha256")
        != identity_sha256(configuration_body)
        or str(workflow_code) != expected_workflow_code
        or scientific.get("scientific_configuration_sha256")
        != configuration.get("scientific_configuration_sha256")
        or scientific.get("workflow_producer_code_identity")
        != workflow_code
        or scientific.get("phase_producer_code_identities")
        != phase_code
        or scientific.get("scientific_sha256")
        != identity_sha256(scientific_body)
        or scientific.get("scientific_sha256")
        != expected_workflow_scientific_sha256
        or _redact_absolute_locators(path_neutral_members)
        != path_neutral_members
    ):
        raise ValueError(
            "benchmark scientific workflow identity binding changed"
        )
    return copy.deepcopy(dict(value))


def _validate_request(
    value: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
) -> None:
    if (
        set(value) != _BENCHMARK_REQUEST_FIELDS
        or value.get("schema_version") != ROLE_NEUTRAL_BENCHMARK_REQUEST_SCHEMA
    ):
        raise ValueError("benchmark request does not match its closed schema")
    _closed_content_identity(value, label="benchmark request")
    if (
        value.get("config") != result.get("config")
        or value.get("config_sha256") != result.get("config_sha256")
        or value.get("workload_binding") != result.get("workload_binding")
        or value.get("execution_schedule") != result.get("execution_schedule")
    ):
        raise ValueError("benchmark request differs from its result")
    compression_source = value.get("compression_source")
    registered_source = result.get("preflight_compression_benchmark", {}).get(
        "source"
    )
    if (
        not isinstance(compression_source, Mapping)
        or not isinstance(registered_source, Mapping)
        or compression_source.get("artifact_content_sha256")
        != registered_source.get("artifact_content_sha256")
        or compression_source.get(
            "path_neutral_scientific_content_sha256"
        )
        != registered_source.get(
            "path_neutral_scientific_content_sha256"
        )
    ):
        raise ValueError(
            "benchmark request changed its compression source identity"
        )


def _normalize_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    logical_root: str,
) -> dict[str, Any]:
    observation = copy.deepcopy(dict(checkpoint["observation"]))
    observation.pop("artifact_path", None)
    observation["logical_artifact_root"] = logical_root
    detail = checkpoint["detail"]
    stable_detail_fields = (
        "candidate_name",
        "scope_label",
        "observation_kind",
        "repetition_index",
        "execution_sequence_index",
        "candidate_position_within_rotation",
        "candidate_rotation_offset",
        "scientific_artifact_sha256",
        "complete_scientific_artifacts_exactly_equal",
        "configured_fits_per_observation",
        "telemetry_accepted",
    )
    detail_summary = {
        key: copy.deepcopy(detail[key])
        for key in stable_detail_fields
        if key in detail
    }
    scientific_targets = [
        {
            "logical_relative_root": row["relative_root"],
            "scientific_artifact_sha256": row[
                "scientific_artifact_sha256"
            ],
        }
        for row in checkpoint["complete_artifacts"]
    ]
    body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_LOGICAL_CHECKPOINT_SCHEMA,
        "schedule_entry": copy.deepcopy(checkpoint["schedule_entry"]),
        "observation": _redact_absolute_locators(observation),
        "detail_summary": _redact_absolute_locators(detail_summary),
        "scientific_artifacts": scientific_targets,
        "historical_request_content_identity_retained": False,
        "physical_tree_and_manifest_identities_retained": False,
        "historical_scratch_locator_authoritative": False,
        "raw_fit_replica_retained_in_publication": False,
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _address_interrupted_logical_proof(
    *,
    schedule_entry: Mapping[str, Any],
    attempt_index: int,
) -> dict[str, Any]:
    body = {
        "schema_version": (
            "portable_role_neutral_benchmark_interrupted_proof_v1"
        ),
        "schedule_entry": copy.deepcopy(dict(schedule_entry)),
        "attempt_index": int(attempt_index),
        "fresh_source_tree_validated_before_omission": True,
        "historical_scratch_locator_authoritative": False,
        "physical_tree_identity_retained": False,
        "raw_interrupted_tree_retained_in_publication": False,
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _validate_execution_manifest(
    value: Mapping[str, Any],
    *,
    expected_manifest_content_sha256: str,
    expected_scientific_artifact_sha256: str,
    expected_plan_sha256: str,
) -> None:
    supplied = _closed_content_identity(
        value,
        label="canonical role-neutral execution manifest",
    )
    scientific = value.get("scientific_identity")
    if (
        supplied != expected_manifest_content_sha256
        or value.get("schema_version") != ROLE_NEUTRAL_STAGE1_EXECUTION_SCHEMA
        or value.get("status") != "complete"
        or value.get("plan_scientific_content_sha256")
        != expected_plan_sha256
        or not isinstance(scientific, Mapping)
        or scientific.get("content_sha256")
        != expected_scientific_artifact_sha256
        or value.get("every_physical_owner_executed_once") is not True
        or value.get(
            "every_component_executed_and_authenticated_once_per_owner"
        )
        is not True
        or value.get(
            "coordination_gate_published_after_complete_execution"
        )
        is not True
    ):
        raise ValueError(
            "canonical role-neutral execution manifest is invalid"
        )


@dataclass(frozen=True)
class PublishedBenchmarkPayload:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        relative = _safe_relative_path(
            self.relative_path,
            label="published benchmark payload path",
        )
        object.__setattr__(self, "relative_path", relative.as_posix())
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("published benchmark payload size is invalid")
        _require_sha256(
            self.sha256,
            label="published benchmark payload identity",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class RoleNeutralBenchmarkPublicationManifest:
    benchmark_bindings: Mapping[str, Any]
    source_record_policy: Mapping[str, Any]
    checkpoint_coverage: Mapping[str, Any]
    compression_publication: Mapping[str, Any]
    canonical_scientific_artifact: Mapping[str, Any]
    path_neutral_result_evidence: Mapping[str, Any]
    payload_inventory: tuple[PublishedBenchmarkPayload, ...]
    payload_content_root_sha256: str
    path_neutral_content_root_sha256: str
    content_sha256: str
    schema_version: str = ROLE_NEUTRAL_BENCHMARK_PUBLICATION_SCHEMA
    status: str = "complete"
    terminal_marker_written_last: bool = True

    def __post_init__(self) -> None:
        if (
            self.schema_version != ROLE_NEUTRAL_BENCHMARK_PUBLICATION_SCHEMA
            or self.status != "complete"
            or self.terminal_marker_written_last is not True
        ):
            raise ValueError("benchmark publication manifest envelope is invalid")
        for name in (
            "benchmark_bindings",
            "source_record_policy",
            "checkpoint_coverage",
            "compression_publication",
            "canonical_scientific_artifact",
            "path_neutral_result_evidence",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"benchmark publication {name} must be an object")
            object.__setattr__(self, name, copy.deepcopy(dict(value)))
        inventory = tuple(self.payload_inventory)
        if (
            not inventory
            or any(
                not isinstance(row, PublishedBenchmarkPayload)
                for row in inventory
            )
            or [row.relative_path for row in inventory]
            != sorted(row.relative_path for row in inventory)
            or len({row.relative_path for row in inventory}) != len(inventory)
        ):
            raise ValueError("benchmark publication payload inventory is invalid")
        object.__setattr__(self, "payload_inventory", inventory)
        expected_payload_root = identity_sha256(
            [row.as_dict() for row in inventory]
        )
        if self.payload_content_root_sha256 != expected_payload_root:
            raise ValueError("benchmark publication payload root is invalid")
        _require_sha256(
            self.path_neutral_content_root_sha256,
            label="benchmark publication path-neutral root",
        )
        expected_content = identity_sha256(self._body())
        if self.content_sha256 != expected_content:
            raise ValueError("benchmark publication manifest identity is invalid")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "benchmark_bindings": copy.deepcopy(dict(self.benchmark_bindings)),
            "source_record_policy": copy.deepcopy(
                dict(self.source_record_policy)
            ),
            "checkpoint_coverage": copy.deepcopy(
                dict(self.checkpoint_coverage)
            ),
            "compression_publication": copy.deepcopy(
                dict(self.compression_publication)
            ),
            "canonical_scientific_artifact": copy.deepcopy(
                dict(self.canonical_scientific_artifact)
            ),
            "path_neutral_result_evidence": copy.deepcopy(
                dict(self.path_neutral_result_evidence)
            ),
            "payload_inventory": [
                row.as_dict() for row in self.payload_inventory
            ],
            "payload_content_root_sha256": self.payload_content_root_sha256,
            "path_neutral_content_root_sha256": (
                self.path_neutral_content_root_sha256
            ),
            "terminal_marker_written_last": self.terminal_marker_written_last,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "RoleNeutralBenchmarkPublicationManifest":
        required = {
            "schema_version",
            "status",
            "benchmark_bindings",
            "source_record_policy",
            "checkpoint_coverage",
            "compression_publication",
            "canonical_scientific_artifact",
            "path_neutral_result_evidence",
            "payload_inventory",
            "payload_content_root_sha256",
            "path_neutral_content_root_sha256",
            "terminal_marker_written_last",
            "content_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                "benchmark publication manifest does not match its closed schema"
            )
        raw_inventory = value["payload_inventory"]
        if not isinstance(raw_inventory, list):
            raise TypeError("benchmark publication payload inventory must be a list")
        return cls(
            schema_version=str(value["schema_version"]),
            status=str(value["status"]),
            benchmark_bindings=value["benchmark_bindings"],
            source_record_policy=value["source_record_policy"],
            checkpoint_coverage=value["checkpoint_coverage"],
            compression_publication=value["compression_publication"],
            canonical_scientific_artifact=value[
                "canonical_scientific_artifact"
            ],
            path_neutral_result_evidence=value[
                "path_neutral_result_evidence"
            ],
            payload_inventory=tuple(
                PublishedBenchmarkPayload(**dict(row))
                for row in raw_inventory
                if isinstance(row, Mapping)
            ),
            payload_content_root_sha256=str(
                value["payload_content_root_sha256"]
            ),
            path_neutral_content_root_sha256=str(
                value["path_neutral_content_root_sha256"]
            ),
            terminal_marker_written_last=value[
                "terminal_marker_written_last"
            ],
            content_sha256=str(value["content_sha256"]),
        )


@dataclass(frozen=True)
class RoleNeutralBenchmarkSelectionEvidence:
    """Freshly authenticated durable evidence used for deployment selection."""

    publication_root: Path
    publication_manifest_path: Path
    publication_manifest_file_sha256: str
    publication_manifest: RoleNeutralBenchmarkPublicationManifest
    normalized_benchmark_result: Mapping[str, Any]
    workload_binding: Mapping[str, Any]
    scientific_workflow_binding: Mapping[str, Any]
    source_binding: RoleNeutralBenchmarkSourceBinding
    benchmark_result_file_sha256: str
    benchmark_result_content_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.publication_root.is_absolute()
            or not self.publication_manifest_path.is_absolute()
            or self.publication_manifest_path.parent
            != self.publication_root
            or self.publication_manifest_path.name
            != ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST
        ):
            raise ValueError(
                "benchmark selection evidence publication locator is invalid"
            )
        for value, label in (
            (
                self.publication_manifest_file_sha256,
                "benchmark publication manifest file",
            ),
            (
                self.benchmark_result_file_sha256,
                "published benchmark result file",
            ),
            (
                self.benchmark_result_content_sha256,
                "published benchmark result content",
            ),
        ):
            _require_sha256(value, label=label)
        if not isinstance(
            self.publication_manifest,
            RoleNeutralBenchmarkPublicationManifest,
        ):
            raise TypeError(
                "benchmark selection evidence requires a typed publication"
            )
        if not isinstance(self.source_binding, RoleNeutralBenchmarkSourceBinding):
            raise TypeError(
                "benchmark selection evidence requires a typed source binding"
            )
        object.__setattr__(
            self,
            "normalized_benchmark_result",
            copy.deepcopy(dict(self.normalized_benchmark_result)),
        )
        object.__setattr__(
            self,
            "workload_binding",
            copy.deepcopy(dict(self.workload_binding)),
        )
        object.__setattr__(
            self,
            "scientific_workflow_binding",
            copy.deepcopy(dict(self.scientific_workflow_binding)),
        )


@dataclass(frozen=True)
class _ValidatedSource:
    result: dict[str, Any]
    request: dict[str, Any]
    config: RoleNeutralBenchmarkConfig
    result_file_sha256: str
    result_file_size: int
    request_file_sha256: str
    request_file_size: int
    checkpoint_records: tuple[dict[str, Any], ...]
    interrupted_records: tuple[dict[str, Any], ...]
    logical_proofs: tuple[dict[str, Any], ...]
    canonical_reference: dict[str, Any]
    canonical_manifest_path: Path
    canonical_manifest_file_sha256: str
    canonical_manifest_file_size: int
    canonical_manifest: dict[str, Any]


def _validate_source_tree(source_root: Path) -> _ValidatedSource:
    expected_top_level = {
        "benchmark_request.json",
        "benchmark_result.json",
        "warmups",
        "runs",
        "executor_sessions",
        "checkpoints",
        "interrupted_observations",
        "preflight_compression_benchmark",
    }
    actual_top_level = {path.name for path in source_root.iterdir()}
    if actual_top_level != expected_top_level:
        raise ValueError("completed benchmark scratch root has extra/missing data")
    for name in (
        "warmups",
        "runs",
        "executor_sessions",
        "checkpoints",
        "interrupted_observations",
        "preflight_compression_benchmark",
    ):
        _canonical_private_directory(
            source_root / name,
            label=f"benchmark scratch {name!r}",
        )
    if any((source_root / "executor_sessions").iterdir()):
        raise ValueError("completed benchmark retains an executor session")

    result, result_file_sha256, result_file_size, config = (
        _validate_source_result(source_root / "benchmark_result.json")
    )
    request, request_file_sha256, request_file_size = _read_private_json(
        source_root / "benchmark_request.json",
        label="benchmark request",
    )
    _validate_request(request, result=result)
    workload_binding = _validate_workload_binding(result["workload_binding"])
    plans_by_scope = {
        str(row["scope_label"]): row
        for row in workload_binding["representative_scope_plans"]
    }
    schedule = result["execution_schedule"]
    entries = schedule.get("entries") if isinstance(schedule, Mapping) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("benchmark execution schedule is incomplete")
    if [entry.get("sequence_index") for entry in entries] != list(
        range(len(entries))
    ):
        raise ValueError("benchmark execution schedule is not ordered")
    if {str(entry.get("scope_label")) for entry in entries} != set(
        plans_by_scope
    ):
        raise ValueError("benchmark schedule differs from workload scope plans")

    checkpoint_root = source_root / "checkpoints"
    expected_checkpoint_names = {
        f"observation_{index:06d}.json" for index in range(len(entries))
    }
    if {path.name for path in checkpoint_root.iterdir()} != (
        expected_checkpoint_names
    ):
        raise ValueError("benchmark checkpoint coverage is incomplete")

    checkpoint_records: list[dict[str, Any]] = []
    logical_proofs: list[dict[str, Any]] = []
    target_evidence: list[dict[str, Any]] = []
    warmup_observations: list[dict[str, Any]] = []
    warmup_details: list[dict[str, Any]] = []
    measured_observations: list[dict[str, Any]] = []
    measured_details: list[dict[str, Any]] = []
    for entry in entries:
        sequence_index = int(entry["sequence_index"])
        checkpoint_path = (
            checkpoint_root / f"observation_{sequence_index:06d}.json"
        )
        checkpoint, file_sha256, file_size = _read_private_json(
            checkpoint_path,
            label=f"benchmark checkpoint {sequence_index}",
        )
        if (
            set(checkpoint) != _CHECKPOINT_FIELDS
            or checkpoint.get("schema_version")
            != ROLE_NEUTRAL_BENCHMARK_OBSERVATION_CHECKPOINT_SCHEMA
            or checkpoint.get("request_sha256")
            != request["content_sha256"]
            or checkpoint.get("schedule_entry") != entry
        ):
            raise ValueError("benchmark checkpoint is invalid or unrelated")
        _closed_content_identity(
            checkpoint,
            label=f"benchmark checkpoint {sequence_index}",
        )
        logical_root = _logical_observation_root(entry)
        observation_root = (source_root / logical_root).resolve(strict=True)
        if (
            observation_root.is_symlink()
            or not observation_root.is_dir()
            or observation_root != source_root / logical_root
        ):
            raise ValueError("benchmark observation root is unsafe")
        observation = checkpoint.get("observation")
        detail = checkpoint.get("detail")
        if (
            not isinstance(observation, Mapping)
            or not isinstance(detail, Mapping)
            or observation.get("artifact_path") != str(observation_root)
            or observation.get("candidate_name")
            != entry.get("candidate_name")
            or observation.get("scope_label") != entry.get("scope_label")
            or observation.get("repetition_index")
            != entry.get("observation_index")
            or detail.get("execution_sequence_index") != sequence_index
            or detail.get("candidate_name") != entry.get("candidate_name")
            or detail.get("scope_label") != entry.get("scope_label")
            or detail.get("observation_kind")
            != entry.get("observation_kind")
        ):
            raise ValueError("benchmark checkpoint observation changed schedule")
        inventory, tree_sha256, total_bytes = _inventory(
            observation_root,
            require_read_only=False,
        )
        tree_proof = checkpoint.get("observation_tree")
        if (
            not isinstance(tree_proof, Mapping)
            or set(tree_proof)
            != {"tree_sha256", "total_file_bytes", "file_count"}
            or tree_proof.get("tree_sha256") != tree_sha256
            or tree_proof.get("total_file_bytes") != total_bytes
            or tree_proof.get("file_count") != len(inventory)
        ):
            raise ValueError("benchmark checkpoint observation tree changed")
        targets = checkpoint.get("complete_artifacts")
        configured_scope = next(
            scope
            for scope in config.representative_scopes
            if scope.label == entry["scope_label"]
        )
        if (
            not isinstance(targets, list)
            or len(targets) != configured_scope.fits_per_observation
            or observation.get("completed_scope_fits") != len(targets)
            or observation.get("complete_artifacts_exactly_equal") is not True
            or observation.get("scientific_artifact_sha256") is None
        ):
            raise ValueError("benchmark checkpoint artifact coverage is incomplete")
        observed_target_roots: set[str] = set()
        expected_target_roots = {
            (
                PurePosixPath(logical_root) / f"fit_{index:03d}"
            ).as_posix()
            for index in range(configured_scope.fits_per_observation)
        }
        for target in targets:
            if not isinstance(target, Mapping) or set(target) != {
                "relative_root",
                "manifest_content_sha256",
                "scientific_artifact_sha256",
            }:
                raise ValueError("benchmark checkpoint artifact row is invalid")
            relative_root = _safe_relative_path(
                target["relative_root"],
                label="benchmark complete artifact root",
            )
            expected_prefix = PurePosixPath(logical_root)
            if (
                tuple(relative_root.parts[: len(expected_prefix.parts)])
                != expected_prefix.parts
                or relative_root.as_posix() in observed_target_roots
            ):
                raise ValueError("benchmark complete artifact root is unrelated")
            observed_target_roots.add(relative_root.as_posix())
            target_inventory = _subtree_inventory(
                inventory,
                relative_root=PurePosixPath(
                    *relative_root.parts[len(expected_prefix.parts) :]
                ),
            )
            if not target_inventory:
                raise ValueError("benchmark complete artifact tree is empty")
            manifest_relative = (
                relative_root / ROLE_NEUTRAL_EXECUTION_MANIFEST
            )
            manifest_path = (
                source_root / Path(*manifest_relative.parts)
            ).resolve(strict=True)
            manifest, manifest_file_sha256, manifest_file_size = (
                _read_private_json(
                    manifest_path,
                    label="role-neutral execution manifest",
                )
            )
            plan_sha256 = plans_by_scope[str(entry["scope_label"])][
                "plan_scientific_content_sha256"
            ]
            _validate_execution_manifest(
                manifest,
                expected_manifest_content_sha256=str(
                    target["manifest_content_sha256"]
                ),
                expected_scientific_artifact_sha256=str(
                    target["scientific_artifact_sha256"]
                ),
                expected_plan_sha256=str(plan_sha256),
            )
            if (
                target["scientific_artifact_sha256"]
                != observation["scientific_artifact_sha256"]
            ):
                raise ValueError(
                    "benchmark complete artifact changed scientific identity"
                )
            target_evidence.append(
                {
                    "sequence_index": sequence_index,
                    "observation_kind": entry["observation_kind"],
                    "scope_label": entry["scope_label"],
                    "relative_root": relative_root.as_posix(),
                    "tree_sha256": identity_sha256(target_inventory),
                    "total_file_bytes": sum(
                        int(row["size_bytes"]) for row in target_inventory
                    ),
                    "file_count": len(target_inventory),
                    "tree_inventory": target_inventory,
                    "manifest_content_sha256": target[
                        "manifest_content_sha256"
                    ],
                    "manifest_file_sha256": manifest_file_sha256,
                    "manifest_file_size": manifest_file_size,
                    "scientific_artifact_sha256": target[
                        "scientific_artifact_sha256"
                    ],
                    "plan_scientific_content_sha256": plan_sha256,
                    "manifest_path": manifest_path,
                    "manifest": manifest,
                }
            )
        if observed_target_roots != expected_target_roots:
            raise ValueError("benchmark complete artifact roots are incomplete")
        normalized = _normalize_checkpoint(
            checkpoint,
            logical_root=logical_root,
        )
        logical_proofs.append(normalized)
        checkpoint_records.append(
            {
                "sequence_index": sequence_index,
                "source_path": checkpoint_path,
                "file_sha256": file_sha256,
                "file_size": file_size,
                "content_sha256": checkpoint["content_sha256"],
            }
        )
        if entry["observation_kind"] == "warmup":
            warmup_observations.append(copy.deepcopy(dict(observation)))
            warmup_details.append(copy.deepcopy(dict(detail)))
        else:
            measured_observations.append(copy.deepcopy(dict(observation)))
            measured_details.append(copy.deepcopy(dict(detail)))

    expected_observation_roots = {
        (source_root / _logical_observation_root(entry)).resolve(strict=True)
        for entry in entries
    }
    for base_name in ("warmups", "runs"):
        base = source_root / base_name
        for path in base.rglob("*"):
            state = os.lstat(path)
            if stat.S_ISLNK(state.st_mode):
                raise ValueError("benchmark observation trees contain a symlink")
            if stat.S_ISREG(state.st_mode):
                if not any(
                    root == path or root in path.parents
                    for root in expected_observation_roots
                ):
                    raise ValueError(
                        "benchmark observation trees contain unrelated data"
                    )
                continue
            if not stat.S_ISDIR(state.st_mode) or not any(
                path == root
                or path in root.parents
                or root in path.parents
                for root in expected_observation_roots
            ):
                raise ValueError(
                    "benchmark observation trees contain unrelated data"
                )
            if any(
                root in path.parents
                for root in expected_observation_roots
            ) and not any(child.is_file() for child in path.rglob("*")):
                raise ValueError(
                    "benchmark observation trees contain an empty extra directory"
                )

    if (
        warmup_observations != result["warmup_observations"]
        or warmup_details != result["warmup_telemetry"]
        or measured_observations != result["benchmark_observations"]
        or measured_details != result["observation_telemetry"]
    ):
        raise ValueError("benchmark result differs from its sealed checkpoints")

    terminal = result.get("terminal_audit")
    terminal_rows = terminal.get("artifacts") if isinstance(terminal, Mapping) else None
    expected_terminal = [
        {
            "root": str(source_root / row["relative_root"]),
            "tree_sha256": row["tree_sha256"],
            "total_file_bytes": row["total_file_bytes"],
            "file_count": row["file_count"],
            "scientific_artifact_sha256": row[
                "scientific_artifact_sha256"
            ],
        }
        for row in target_evidence
    ]
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("exactly_one_completed_terminal_audit") is not True
        or terminal.get("audited_complete_artifact_count")
        != len(target_evidence)
        or terminal_rows != expected_terminal
    ):
        raise ValueError("benchmark terminal audit differs from checkpoint bytes")

    interrupted_root = source_root / "interrupted_observations"
    interrupted_children = tuple(interrupted_root.iterdir())
    interrupted_directories = {
        path.name: path
        for path in interrupted_children
        if path.is_dir() and not path.is_symlink()
    }
    interrupted_json = {
        path.stem: path
        for path in interrupted_children
        if path.is_file() and not path.is_symlink() and path.suffix == ".json"
    }
    if (
        len(interrupted_directories) + len(interrupted_json)
        != len(interrupted_children)
        or set(interrupted_directories) != set(interrupted_json)
    ):
        raise ValueError("interrupted benchmark attempts are malformed")
    interrupted_records: list[dict[str, Any]] = []
    schedule_by_index = {
        int(entry["sequence_index"]): entry for entry in entries
    }
    for name, attempt_root in sorted(interrupted_directories.items()):
        match = _INTERRUPTED_NAME.fullmatch(name + ".json")
        if match is None:
            raise ValueError("interrupted benchmark attempt name is invalid")
        sequence_index = int(match.group(1))
        record_path = interrupted_json[name]
        record, file_sha256, file_size = _read_private_json(
            record_path,
            label="interrupted benchmark attempt record",
        )
        _closed_content_identity(
            record,
            label="interrupted benchmark attempt record",
        )
        attempt_inventory, attempt_tree, attempt_bytes = _inventory(
            attempt_root,
            require_read_only=False,
        )
        if (
            set(record) != _INTERRUPTED_FIELDS
            or record.get("schema_version")
            != ROLE_NEUTRAL_BENCHMARK_INTERRUPTED_OBSERVATION_SCHEMA
            or record.get("request_sha256") != request["content_sha256"]
            or record.get("schedule_entry")
            != schedule_by_index.get(sequence_index)
            or record.get("preserved_relative_root")
            != attempt_root.relative_to(source_root).as_posix()
            or record.get("tree_sha256") != attempt_tree
            or record.get("total_file_bytes") != attempt_bytes
            or record.get("file_count") != len(attempt_inventory)
        ):
            raise ValueError("interrupted benchmark attempt changed")
        interrupted_records.append(
            {
                "source_path": record_path,
                "file_sha256": file_sha256,
                "file_size": file_size,
                "content_sha256": record["content_sha256"],
                "logical_proof": _address_interrupted_logical_proof(
                    schedule_entry=record["schedule_entry"],
                    attempt_index=int(match.group(2)),
                ),
            }
        )

    measured_targets = [
        row for row in target_evidence if row["observation_kind"] == "measured"
    ]
    if not measured_targets:
        raise ValueError("accepted benchmark lacks a measured scientific artifact")
    canonical = min(
        measured_targets,
        key=lambda row: (int(row["sequence_index"]), str(row["relative_root"])),
    )
    canonical_manifest_path = canonical.pop("manifest_path")
    canonical_manifest = canonical.pop("manifest")
    path_neutral_scientific_binding_body = {
        "scope_label": canonical["scope_label"],
        "scientific_artifact_sha256": canonical[
            "scientific_artifact_sha256"
        ],
        "plan_scientific_content_sha256": canonical[
            "plan_scientific_content_sha256"
        ],
    }
    canonical_body = {
        "schema_version": (
            ROLE_NEUTRAL_BENCHMARK_CANONICAL_ARTIFACT_REFERENCE_SCHEMA
        ),
        "selection_policy": (
            "earliest_measured_checkpoint_then_relative_fit_root_v1"
        ),
        "sequence_index": canonical["sequence_index"],
        "scope_label": canonical["scope_label"],
        "logical_relative_root": canonical["relative_root"],
        "tree_sha256": canonical["tree_sha256"],
        "total_file_bytes": canonical["total_file_bytes"],
        "file_count": canonical["file_count"],
        "tree_inventory": canonical["tree_inventory"],
        "manifest_content_sha256": canonical["manifest_content_sha256"],
        "manifest_file_sha256": canonical["manifest_file_sha256"],
        "manifest_file_size": canonical["manifest_file_size"],
        "scientific_artifact_sha256": canonical[
            "scientific_artifact_sha256"
        ],
        "plan_scientific_content_sha256": canonical[
            "plan_scientific_content_sha256"
        ],
        "exact_execution_manifest_retained": True,
        "complete_raw_artifact_tree_retained": False,
        "future_scientific_replay_possible_from_publication": False,
        "fresh_source_tree_validated_before_omission": True,
        "path_neutral_scientific_binding": {
            **path_neutral_scientific_binding_body,
            "content_sha256": identity_sha256(
                path_neutral_scientific_binding_body
            ),
        },
    }
    canonical_reference = {
        **canonical_body,
        "content_sha256": identity_sha256(canonical_body),
    }
    return _ValidatedSource(
        result=copy.deepcopy(result),
        request=copy.deepcopy(request),
        config=config,
        result_file_sha256=result_file_sha256,
        result_file_size=result_file_size,
        request_file_sha256=request_file_sha256,
        request_file_size=request_file_size,
        checkpoint_records=tuple(checkpoint_records),
        interrupted_records=tuple(interrupted_records),
        logical_proofs=tuple(logical_proofs),
        canonical_reference=canonical_reference,
        canonical_manifest_path=canonical_manifest_path,
        canonical_manifest_file_sha256=canonical[
            "manifest_file_sha256"
        ],
        canonical_manifest_file_size=canonical["manifest_file_size"],
        canonical_manifest=canonical_manifest,
    )


def _path_neutral_result_evidence(
    source: _ValidatedSource,
    *,
    compression_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_result = {
        "schema_version": source.result["schema_version"],
        "status": source.result["status"],
        "config": copy.deepcopy(source.result["config"]),
        "config_sha256": source.result["config_sha256"],
        "scientific_workload_binding": _path_neutral_workload_binding(
            source.result["workload_binding"]
        ),
        "execution_schedule": copy.deepcopy(
            source.result["execution_schedule"]
        ),
        "candidate_results": _redact_absolute_locators(
            source.result["candidate_results"]
        ),
        "benchmark_matrix_coverage": _redact_absolute_locators(
            source.result["benchmark_matrix_coverage"]
        ),
        "selected_candidate": source.result["selected_candidate"],
        "selection_policy": source.result["selection_policy"],
        "scientific_result_identity_sha256": source.result[
            "scientific_result_identity_sha256"
        ],
        "accepted": source.result["accepted"],
        "warmup_observations_excluded_from_selection": source.result[
            "warmup_observations_excluded_from_selection"
        ],
        "ordinary_observations_exclude_terminal_audit": source.result[
            "ordinary_observations_exclude_terminal_audit"
        ],
        "preflight_compression_benchmark": copy.deepcopy(
            dict(compression_evidence)
        ),
        "historical_result_content_identity_retained": False,
        "physical_observation_and_terminal_audit_records_retained": False,
    }
    canonical_binding = source.canonical_reference[
        "path_neutral_scientific_binding"
    ]
    body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_PATH_NEUTRAL_RESULT_SCHEMA,
        "benchmark_result_schema_version": source.result["schema_version"],
        "normalized_benchmark_result": normalized_result,
        "logical_checkpoint_proof_content_sha256": [
            proof["content_sha256"] for proof in source.logical_proofs
        ],
        "canonical_path_neutral_scientific_binding_content_sha256": (
            canonical_binding["content_sha256"]
        ),
        "historical_absolute_locators_redacted": True,
        "historical_scratch_locators_authoritative": False,
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _remove_fresh_publication(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    for child in sorted(
        path.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            os.chmod(child, 0o700 if child.is_dir() else 0o600)
        except OSError:
            pass
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    shutil.rmtree(path, ignore_errors=True)


def publish_role_neutral_performance_benchmark(
    *,
    scratch_root: Path | str,
    durable_root: Path | str,
    workload_deployment_path: Path | str,
) -> RoleNeutralBenchmarkPublicationManifest:
    """Publish one completed scratch benchmark into a fresh durable root."""

    source_root = _canonical_private_directory(
        Path(scratch_root),
        label="completed benchmark scratch root",
    )
    destination = Path(durable_root)
    if not destination.is_absolute():
        raise ValueError("benchmark publication root must be absolute")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("benchmark publication root must be fresh")
    parent = destination.parent.resolve(strict=True)
    if parent != destination.parent or not parent.is_dir():
        raise ValueError("benchmark publication parent must be canonical")

    source = _validate_source_tree(source_root)
    scientific_workflow_binding = _scientific_workflow_binding(
        source=source,
        workload_deployment_path=Path(workload_deployment_path),
    )
    producer_code_sha256 = _publication_producer_code_sha256()
    destination.mkdir(mode=0o700)
    try:
        source_records = destination / "source_records"
        checkpoint_records_root = source_records / "checkpoints"
        interrupted_records_root = source_records / "interrupted_observations"
        logical_root = destination / "logical_evidence"
        logical_checkpoint_root = logical_root / "checkpoints"
        canonical_root = destination / "canonical_scientific_artifact"
        for directory in (
            source_records,
            checkpoint_records_root,
            interrupted_records_root,
            logical_root,
            logical_checkpoint_root,
            canonical_root,
        ):
            directory.mkdir(mode=0o700, exist_ok=True)

        _copy_once(
            source_root / "benchmark_result.json",
            source_records / "benchmark_result.json",
            label="completed benchmark result",
        )
        _copy_once(
            source_root / "benchmark_request.json",
            source_records / "benchmark_request.json",
            label="completed benchmark request",
        )
        copied_checkpoint_bindings: list[dict[str, Any]] = []
        for record in source.checkpoint_records:
            sequence_index = int(record["sequence_index"])
            target = (
                checkpoint_records_root
                / f"observation_{sequence_index:06d}.json"
            )
            copied = _copy_once(
                Path(record["source_path"]),
                target,
                label=f"benchmark checkpoint {sequence_index}",
            )
            copied_checkpoint_bindings.append(
                {
                    "sequence_index": sequence_index,
                    "relative_path": target.relative_to(destination).as_posix(),
                    "file_sha256": copied["sha256"],
                    "file_size": copied["size_bytes"],
                    "content_sha256": record["content_sha256"],
                }
            )
        copied_interrupted_bindings: list[dict[str, Any]] = []
        for index, record in enumerate(source.interrupted_records):
            target = interrupted_records_root / f"attempt_{index:06d}.json"
            copied = _copy_once(
                Path(record["source_path"]),
                target,
                label=f"interrupted benchmark attempt {index}",
            )
            copied_interrupted_bindings.append(
                {
                    "relative_path": target.relative_to(destination).as_posix(),
                    "file_sha256": copied["sha256"],
                    "file_size": copied["size_bytes"],
                    "content_sha256": record["content_sha256"],
                }
            )

        for proof in source.logical_proofs:
            sequence_index = int(proof["schedule_entry"]["sequence_index"])
            _write_new_json(
                logical_checkpoint_root
                / f"observation_{sequence_index:06d}.json",
                proof,
            )
        logical_interrupted_proofs: list[dict[str, Any]] = []
        for index, record in enumerate(source.interrupted_records):
            logical_attempt = copy.deepcopy(record["logical_proof"])
            _write_new_json(
                logical_root / f"interrupted_attempt_{index:06d}.json",
                logical_attempt,
            )
            logical_interrupted_proofs.append(logical_attempt)

        config_record = {
            "schema_version": source.result["config"]["schema_version"],
            "config": copy.deepcopy(source.result["config"]),
            "config_sha256": source.result["config_sha256"],
        }
        config_record["content_sha256"] = identity_sha256(config_record)
        _write_new_json(logical_root / "benchmark_config.json", config_record)
        _write_new_json(
            logical_root / "workload_binding.json",
            source.result["workload_binding"],
        )
        source_binding = source.result["workload_binding"]["source"]
        source_binding_record = {
            "schema_version": ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA,
            "source_binding": copy.deepcopy(source_binding),
        }
        source_binding_record["content_sha256"] = identity_sha256(
            source_binding_record
        )
        _write_new_json(
            logical_root / "source_binding.json",
            source_binding_record,
        )
        _write_new_json(
            logical_root / "execution_schedule.json",
            source.result["execution_schedule"],
        )
        _write_new_json(
            logical_root / "scientific_workflow_binding.json",
            scientific_workflow_binding,
        )

        canonical_copy = _copy_once(
            source.canonical_manifest_path,
            canonical_root / ROLE_NEUTRAL_EXECUTION_MANIFEST,
            label="canonical role-neutral execution manifest",
        )
        if (
            canonical_copy["sha256"]
            != source.canonical_manifest_file_sha256
            or canonical_copy["size_bytes"]
            != source.canonical_manifest_file_size
        ):
            raise RuntimeError(
                "canonical execution manifest changed while being published"
            )
        _write_new_json(
            canonical_root / "artifact_reference.json",
            source.canonical_reference,
        )

        durable_compression = (
            publish_compact_preflight_compression_benchmark_result(
                source.result["preflight_compression_benchmark"],
                output_root=destination / "preflight_compression_benchmark",
            )
        )
        compression_evidence = _compression_path_neutral_evidence(
            durable_compression
        )
        _write_new_json(
            logical_root / "compression_evidence.json",
            compression_evidence,
        )
        path_neutral_result = _path_neutral_result_evidence(
            source,
            compression_evidence=compression_evidence,
        )
        _write_new_json(
            logical_root / "path_neutral_benchmark_result.json",
            path_neutral_result,
        )

        for path in destination.rglob("*"):
            if path.is_file() and not path.is_symlink():
                os.chmod(path, 0o444)
        for directory in sorted(
            (
                path
                for path in destination.rglob("*")
                if path.is_dir() and not path.is_symlink()
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            os.chmod(directory, 0o555)
            _fsync_directory(directory)
        inventory_rows, payload_root, _payload_bytes = _inventory(
            destination,
            require_read_only=True,
        )
        payload_inventory = tuple(
            PublishedBenchmarkPayload(**row) for row in inventory_rows
        )
        checkpoint_proof_root = identity_sha256(
            [
                {
                    "sequence_index": proof["schedule_entry"][
                        "sequence_index"
                    ],
                    "content_sha256": proof["content_sha256"],
                }
                for proof in source.logical_proofs
            ]
        )
        interrupted_proof_root = identity_sha256(
            [
                {
                    "publication_index": index,
                    "content_sha256": proof["content_sha256"],
                }
                for index, proof in enumerate(logical_interrupted_proofs)
            ]
        )
        path_neutral_body = {
            "publication_producer_code_sha256": producer_code_sha256,
            "config_sha256": source.result["config_sha256"],
            "scientific_workload_binding_content_sha256": (
                _path_neutral_workload_binding(
                    source.result["workload_binding"]
                )["content_sha256"]
            ),
            "execution_schedule_content_sha256": source.result[
                "execution_schedule"
            ]["content_sha256"],
            "scientific_result_identity_sha256": source.result[
                "scientific_result_identity_sha256"
            ],
            "path_neutral_result_evidence_content_sha256": (
                path_neutral_result["content_sha256"]
            ),
            "logical_checkpoint_proof_root_sha256": checkpoint_proof_root,
            "logical_interrupted_proof_root_sha256": interrupted_proof_root,
            "compression_path_neutral_evidence_content_sha256": (
                compression_evidence["content_sha256"]
            ),
            "canonical_path_neutral_scientific_binding_content_sha256": (
                source.canonical_reference[
                    "path_neutral_scientific_binding"
                ]["content_sha256"]
            ),
            "scientific_workflow_binding_content_sha256": (
                scientific_workflow_binding["content_sha256"]
            ),
        }
        path_neutral_root = identity_sha256(path_neutral_body)
        benchmark_bindings = {
            "publication_producer_code_sha256": producer_code_sha256,
            "benchmark_result": {
                "relative_path": "source_records/benchmark_result.json",
                "file_sha256": source.result_file_sha256,
                "file_size": source.result_file_size,
                "content_sha256": source.result["content_sha256"],
                "registered_field_names": sorted(source.result),
            },
            "benchmark_request": {
                "relative_path": "source_records/benchmark_request.json",
                "file_sha256": source.request_file_sha256,
                "file_size": source.request_file_size,
                "content_sha256": source.request["content_sha256"],
            },
            "config_sha256": source.result["config_sha256"],
            "workload_binding_content_sha256": source.result[
                "workload_binding"
            ]["content_sha256"],
            "execution_schedule_content_sha256": source.result[
                "execution_schedule"
            ]["content_sha256"],
            "scientific_result_identity_sha256": source.result[
                "scientific_result_identity_sha256"
            ],
            "selected_candidate": source.result["selected_candidate"],
            "workflow_request_sha256": source_binding[
                "workflow_request_sha256"
            ],
            "workflow_scientific_sha256": source_binding[
                "workflow_scientific_sha256"
            ],
            "workload_deployment_sha256": source_binding[
                "workload_deployment_sha256"
            ],
            "stage1_preflight_phase_content_sha256": source_binding[
                "stage1_preflight_phase_content_sha256"
            ],
            "prepared_stage1_context_content_root_sha256": source_binding[
                "prepared_stage1_context_content_root_sha256"
            ],
            "scientific_workflow_binding_content_sha256": (
                scientific_workflow_binding["content_sha256"]
            ),
        }
        source_record_policy = {
            "byte_exact_source_records_retained": True,
            "source_records_contain_historical_scratch_locators": True,
            "historical_scratch_locators_authoritative": False,
            "durable_path_neutral_evidence_is_authoritative": True,
            "complete_fit_replica_trees_published": False,
            "interrupted_attempt_trees_published": False,
            "future_complete_scientific_replay_from_publication": False,
            "publication_claim_limited_to_freshly_validated_terminal_evidence": True,
        }
        checkpoint_coverage = {
            "expected_checkpoint_count": len(source.logical_proofs),
            "published_checkpoint_count": len(
                copied_checkpoint_bindings
            ),
            "published_logical_proof_count": len(source.logical_proofs),
            "checkpoint_bindings": copied_checkpoint_bindings,
            "logical_checkpoint_proof_root_sha256": checkpoint_proof_root,
            "logical_interrupted_proof_root_sha256": interrupted_proof_root,
            "interrupted_attempt_record_count": len(
                copied_interrupted_bindings
            ),
            "interrupted_attempt_bindings": copied_interrupted_bindings,
        }
        compression_publication = {
            "relative_root": "preflight_compression_benchmark",
            "result_content_sha256": durable_compression["content_sha256"],
            "selected_parquet_compression": durable_compression[
                "selected_parquet_compression"
            ],
            "path_neutral_evidence_content_sha256": compression_evidence[
                "content_sha256"
            ],
            "all_published_compression_replica_bytes_reopened": True,
        }
        canonical_scientific_artifact = {
            "relative_manifest_path": (
                "canonical_scientific_artifact/"
                + ROLE_NEUTRAL_EXECUTION_MANIFEST
            ),
            "relative_reference_path": (
                "canonical_scientific_artifact/artifact_reference.json"
            ),
            "reference_content_sha256": source.canonical_reference[
                "content_sha256"
            ],
            "path_neutral_scientific_binding_content_sha256": (
                source.canonical_reference[
                    "path_neutral_scientific_binding"
                ]["content_sha256"]
            ),
            "manifest_content_sha256": source.canonical_reference[
                "manifest_content_sha256"
            ],
            "scientific_artifact_sha256": source.canonical_reference[
                "scientific_artifact_sha256"
            ],
            "raw_artifact_tree_published": False,
            "scientific_replay_claimed": False,
        }
        path_neutral_result_evidence = {
            "relative_path": (
                "logical_evidence/path_neutral_benchmark_result.json"
            ),
            "content_sha256": path_neutral_result["content_sha256"],
            "path_neutral_content_root_inputs": path_neutral_body,
        }
        manifest_body = {
            "schema_version": ROLE_NEUTRAL_BENCHMARK_PUBLICATION_SCHEMA,
            "status": "complete",
            "benchmark_bindings": benchmark_bindings,
            "source_record_policy": source_record_policy,
            "checkpoint_coverage": checkpoint_coverage,
            "compression_publication": compression_publication,
            "canonical_scientific_artifact": canonical_scientific_artifact,
            "path_neutral_result_evidence": path_neutral_result_evidence,
            "payload_inventory": [
                row.as_dict() for row in payload_inventory
            ],
            "payload_content_root_sha256": payload_root,
            "path_neutral_content_root_sha256": path_neutral_root,
            "terminal_marker_written_last": True,
        }
        manifest = RoleNeutralBenchmarkPublicationManifest(
            benchmark_bindings=benchmark_bindings,
            source_record_policy=source_record_policy,
            checkpoint_coverage=checkpoint_coverage,
            compression_publication=compression_publication,
            canonical_scientific_artifact=canonical_scientific_artifact,
            path_neutral_result_evidence=path_neutral_result_evidence,
            payload_inventory=payload_inventory,
            payload_content_root_sha256=payload_root,
            path_neutral_content_root_sha256=path_neutral_root,
            content_sha256=identity_sha256(manifest_body),
        )
        # This immutable marker is intentionally the final file created.
        _write_new_json(
            destination / ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST,
            manifest.as_dict(),
        )
        os.chmod(destination, 0o555)
        _fsync_directory(destination)
        _fsync_directory(parent)
    except BaseException:
        _remove_fresh_publication(destination)
        raise
    return validate_role_neutral_performance_benchmark_publication(
        destination
    )


def validate_role_neutral_performance_benchmark_publication(
    root: Path | str,
) -> RoleNeutralBenchmarkPublicationManifest:
    """Freshly reopen every durable byte and validate the compact publication."""

    publication_root = _canonical_private_directory(
        Path(root),
        label="durable benchmark publication root",
    )
    expected_top_level = {
        ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST,
        "source_records",
        "logical_evidence",
        "canonical_scientific_artifact",
        "preflight_compression_benchmark",
    }
    if {path.name for path in publication_root.iterdir()} != expected_top_level:
        raise ValueError("benchmark publication has extra/missing top-level data")
    manifest_value, _manifest_file_sha256, _manifest_file_size = (
        _read_private_json(
            publication_root / ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST,
            label="benchmark publication terminal manifest",
        )
    )
    manifest = RoleNeutralBenchmarkPublicationManifest.from_mapping(
        manifest_value
    )
    benchmark_bindings = manifest.benchmark_bindings
    if set(benchmark_bindings) != _BENCHMARK_BINDING_FIELDS:
        raise ValueError(
            "benchmark publication durable authority binding is not closed"
        )
    if benchmark_bindings.get(
        "publication_producer_code_sha256"
    ) != _publication_producer_code_sha256():
        raise ValueError("benchmark publication producer code changed")
    inventory, payload_root, _payload_bytes = _inventory(
        publication_root,
        exclude_relative_paths=(
            ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST,
        ),
        require_read_only=True,
    )
    if (
        inventory
        != [row.as_dict() for row in manifest.payload_inventory]
        or payload_root != manifest.payload_content_root_sha256
    ):
        raise ValueError("benchmark publication payload bytes changed")

    result_binding = benchmark_bindings.get("benchmark_result")
    request_binding = benchmark_bindings.get("benchmark_request")
    if not isinstance(result_binding, Mapping) or not isinstance(
        request_binding,
        Mapping,
    ):
        raise ValueError("benchmark publication source bindings are invalid")
    result_path = publication_root / str(result_binding["relative_path"])
    request_path = publication_root / str(request_binding["relative_path"])
    result, result_file_sha256, result_size = _read_private_json(
        result_path,
        label="published historical benchmark result",
    )
    request, request_file_sha256, request_size = _read_private_json(
        request_path,
        label="published historical benchmark request",
    )
    _validate_result_record_without_historical_reopen(result)
    _validate_request(request, result=result)
    if (
        result_file_sha256 != result_binding.get("file_sha256")
        or result_size != result_binding.get("file_size")
        or result.get("content_sha256")
        != result_binding.get("content_sha256")
        or sorted(result) != result_binding.get("registered_field_names")
        or request_file_sha256 != request_binding.get("file_sha256")
        or request_size != request_binding.get("file_size")
        or request.get("content_sha256")
        != request_binding.get("content_sha256")
    ):
        raise ValueError("published benchmark source record changed")

    logical_evidence_root = publication_root / "logical_evidence"
    config_record = _read_private_json(
        logical_evidence_root / "benchmark_config.json",
        label="published benchmark config",
    )[0]
    workload_record = _read_private_json(
        logical_evidence_root / "workload_binding.json",
        label="published benchmark workload binding",
    )[0]
    source_binding_record = _read_private_json(
        logical_evidence_root / "source_binding.json",
        label="published benchmark source binding",
    )[0]
    schedule_record = _read_private_json(
        logical_evidence_root / "execution_schedule.json",
        label="published benchmark execution schedule",
    )[0]
    scientific_workflow_binding = _read_private_json(
        logical_evidence_root / "scientific_workflow_binding.json",
        label="published benchmark scientific workflow binding",
    )[0]
    if (
        set(config_record)
        != {"schema_version", "config", "config_sha256", "content_sha256"}
        or config_record.get("schema_version")
        != result["config"]["schema_version"]
        or config_record.get("config") != result["config"]
        or config_record.get("config_sha256") != result["config_sha256"]
    ):
        raise ValueError("published benchmark config changed")
    _closed_content_identity(
        config_record,
        label="published benchmark config",
    )
    if (
        workload_record != result["workload_binding"]
        or schedule_record != result["execution_schedule"]
    ):
        raise ValueError("published benchmark logical binding changed")
    _validate_workload_binding(workload_record)
    _closed_content_identity(
        schedule_record,
        label="published benchmark execution schedule",
    )
    if (
        set(source_binding_record)
        != {"schema_version", "source_binding", "content_sha256"}
        or source_binding_record.get("schema_version")
        != ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA
        or source_binding_record.get("source_binding")
        != result["workload_binding"]["source"]
    ):
        raise ValueError("published benchmark source binding changed")
    _closed_content_identity(
        source_binding_record,
        label="published benchmark source binding",
    )
    typed_source_binding = RoleNeutralBenchmarkSourceBinding(
        **dict(source_binding_record["source_binding"])
    )
    validated_scientific_workflow_binding = (
        _validate_scientific_workflow_binding(
            scientific_workflow_binding,
            expected_workflow_scientific_sha256=(
                typed_source_binding.workflow_scientific_sha256
            ),
        )
    )
    if (
        benchmark_bindings.get("config_sha256")
        != result["config_sha256"]
        or benchmark_bindings.get("workload_binding_content_sha256")
        != result["workload_binding"]["content_sha256"]
        or benchmark_bindings.get("execution_schedule_content_sha256")
        != result["execution_schedule"]["content_sha256"]
        or benchmark_bindings.get("scientific_result_identity_sha256")
        != result["scientific_result_identity_sha256"]
        or benchmark_bindings.get("selected_candidate")
        != result["selected_candidate"]
        or any(
            benchmark_bindings.get(name) != getattr(typed_source_binding, name)
            for name in (
                "workflow_request_sha256",
                "workflow_scientific_sha256",
                "workload_deployment_sha256",
                "stage1_preflight_phase_content_sha256",
                "prepared_stage1_context_content_root_sha256",
            )
        )
        or benchmark_bindings.get(
            "scientific_workflow_binding_content_sha256"
        )
        != validated_scientific_workflow_binding["content_sha256"]
    ):
        raise ValueError(
            "benchmark publication durable authority binding changed"
        )

    durable_compression = (
        validate_compact_preflight_compression_benchmark_result(
            _read_private_json(
                publication_root
                / "preflight_compression_benchmark"
                / "compression_benchmark_result.json",
                label="published compression benchmark result",
            )[0],
            reopen_artifacts=True,
        )
    )
    compression_evidence = _compression_path_neutral_evidence(
        durable_compression
    )
    published_compression_evidence = _read_private_json(
        logical_evidence_root / "compression_evidence.json",
        label="published path-neutral compression evidence",
    )[0]
    _closed_content_identity(
        published_compression_evidence,
        label="published path-neutral compression evidence",
    )
    if published_compression_evidence != compression_evidence:
        raise ValueError("published compression evidence changed")
    registered_compression = manifest.compression_publication
    if (
        durable_compression.get("content_sha256")
        != registered_compression.get("result_content_sha256")
        or compression_evidence.get("content_sha256")
        != registered_compression.get(
            "path_neutral_evidence_content_sha256"
        )
        or durable_compression.get("selected_parquet_compression")
        != registered_compression.get("selected_parquet_compression")
    ):
        raise ValueError("published compression benchmark changed")

    checkpoint_bindings = manifest.checkpoint_coverage.get(
        "checkpoint_bindings"
    )
    if not isinstance(checkpoint_bindings, list):
        raise ValueError("published checkpoint bindings are invalid")
    logical_proof_hashes: list[dict[str, Any]] = []
    for expected_sequence, binding in enumerate(checkpoint_bindings):
        if (
            not isinstance(binding, Mapping)
            or binding.get("sequence_index") != expected_sequence
        ):
            raise ValueError("published checkpoint order is invalid")
        checkpoint, file_sha256, file_size = _read_private_json(
            publication_root / str(binding["relative_path"]),
            label=f"published benchmark checkpoint {expected_sequence}",
        )
        if (
            set(checkpoint) != _CHECKPOINT_FIELDS
            or checkpoint.get("schema_version")
            != ROLE_NEUTRAL_BENCHMARK_OBSERVATION_CHECKPOINT_SCHEMA
            or checkpoint.get("schedule_entry")
            != result["execution_schedule"]["entries"][expected_sequence]
            or file_sha256 != binding.get("file_sha256")
            or file_size != binding.get("file_size")
            or checkpoint.get("content_sha256")
            != binding.get("content_sha256")
            or checkpoint.get("request_sha256")
            != request.get("content_sha256")
        ):
            raise ValueError("published historical checkpoint changed")
        _closed_content_identity(
            checkpoint,
            label=f"published historical checkpoint {expected_sequence}",
        )
        proof_path = (
            publication_root
            / "logical_evidence"
            / "checkpoints"
            / f"observation_{expected_sequence:06d}.json"
        )
        proof = _read_private_json(
            proof_path,
            label=f"published logical checkpoint {expected_sequence}",
        )[0]
        if (
            proof.get("schema_version")
            != ROLE_NEUTRAL_BENCHMARK_LOGICAL_CHECKPOINT_SCHEMA
            or proof.get("schedule_entry")
            != checkpoint.get("schedule_entry")
            or proof.get("historical_scratch_locator_authoritative")
            is not False
            or proof.get("raw_fit_replica_retained_in_publication") is not False
        ):
            raise ValueError("published logical checkpoint proof is invalid")
        _closed_content_identity(
            proof,
            label=f"published logical checkpoint {expected_sequence}",
        )
        expected_proof = _normalize_checkpoint(
            checkpoint,
            logical_root=_logical_observation_root(
                checkpoint["schedule_entry"]
            ),
        )
        if proof != expected_proof:
            raise ValueError("published logical checkpoint proof changed")
        logical_proof_hashes.append(
            {
                "sequence_index": expected_sequence,
                "content_sha256": proof["content_sha256"],
            }
        )
    checkpoint_root = identity_sha256(logical_proof_hashes)
    if (
        len(checkpoint_bindings)
        != manifest.checkpoint_coverage.get("expected_checkpoint_count")
        or len(checkpoint_bindings)
        != manifest.checkpoint_coverage.get("published_checkpoint_count")
        or len(checkpoint_bindings)
        != manifest.checkpoint_coverage.get(
            "published_logical_proof_count"
        )
        or checkpoint_root
        != manifest.checkpoint_coverage.get(
            "logical_checkpoint_proof_root_sha256"
        )
    ):
        raise ValueError("published checkpoint coverage is incomplete")

    interrupted_bindings = manifest.checkpoint_coverage.get(
        "interrupted_attempt_bindings"
    )
    if not isinstance(interrupted_bindings, list):
        raise ValueError("published interrupted-attempt bindings are invalid")
    logical_interrupted_hashes: list[dict[str, Any]] = []
    for index, binding in enumerate(interrupted_bindings):
        if not isinstance(binding, Mapping):
            raise ValueError("published interrupted-attempt binding is invalid")
        record, file_sha256, file_size = _read_private_json(
            publication_root / str(binding["relative_path"]),
            label=f"published interrupted-attempt record {index}",
        )
        _closed_content_identity(
            record,
            label=f"published interrupted-attempt record {index}",
        )
        if (
            set(record) != _INTERRUPTED_FIELDS
            or record.get("schema_version")
            != ROLE_NEUTRAL_BENCHMARK_INTERRUPTED_OBSERVATION_SCHEMA
            or file_sha256 != binding.get("file_sha256")
            or file_size != binding.get("file_size")
            or record.get("content_sha256")
            != binding.get("content_sha256")
            or record.get("request_sha256") != request["content_sha256"]
        ):
            raise ValueError("published interrupted-attempt record changed")
        preserved_name = Path(
            str(record.get("preserved_relative_root", ""))
        ).name
        attempt_match = _INTERRUPTED_NAME.fullmatch(
            preserved_name + ".json"
        )
        if attempt_match is None:
            raise ValueError("published interrupted-attempt name is invalid")
        proof = _read_private_json(
            logical_evidence_root
            / f"interrupted_attempt_{index:06d}.json",
            label=f"published logical interrupted-attempt proof {index}",
        )[0]
        _closed_content_identity(
            proof,
            label=f"published logical interrupted-attempt proof {index}",
        )
        expected_proof = _address_interrupted_logical_proof(
            schedule_entry=record["schedule_entry"],
            attempt_index=int(attempt_match.group(2)),
        )
        if (
            proof != expected_proof
            or proof.get("fresh_source_tree_validated_before_omission")
            is not True
            or proof.get("raw_interrupted_tree_retained_in_publication")
            is not False
        ):
            raise ValueError("published interrupted-attempt proof changed")
        logical_interrupted_hashes.append(
            {
                "publication_index": index,
                "content_sha256": proof["content_sha256"],
            }
        )
    interrupted_root = identity_sha256(logical_interrupted_hashes)
    if (
        len(interrupted_bindings)
        != manifest.checkpoint_coverage.get(
            "interrupted_attempt_record_count"
        )
        or interrupted_root
        != manifest.checkpoint_coverage.get(
            "logical_interrupted_proof_root_sha256"
        )
    ):
        raise ValueError("published interrupted-attempt coverage is incomplete")

    expected_source_record_top_level = {
        "benchmark_result.json",
        "benchmark_request.json",
        "checkpoints",
        "interrupted_observations",
    }
    source_record_root = publication_root / "source_records"
    if {path.name for path in source_record_root.iterdir()} != (
        expected_source_record_top_level
    ):
        raise ValueError("published source-record tree contains extra data")
    if {
        path.name
        for path in (source_record_root / "checkpoints").iterdir()
    } != {
        Path(str(binding["relative_path"])).name
        for binding in checkpoint_bindings
    }:
        raise ValueError("published checkpoint record tree contains extra data")
    if {
        path.name
        for path in (
            source_record_root / "interrupted_observations"
        ).iterdir()
    } != {
        Path(str(binding["relative_path"])).name
        for binding in interrupted_bindings
    }:
        raise ValueError(
            "published interrupted record tree contains extra data"
        )
    expected_logical_names = {
        "benchmark_config.json",
        "workload_binding.json",
        "source_binding.json",
        "execution_schedule.json",
        "scientific_workflow_binding.json",
        "compression_evidence.json",
        "path_neutral_benchmark_result.json",
        "checkpoints",
        *{
            f"interrupted_attempt_{index:06d}.json"
            for index in range(len(interrupted_bindings))
        },
    }
    if {path.name for path in logical_evidence_root.iterdir()} != (
        expected_logical_names
    ):
        raise ValueError("published logical-evidence tree contains extra data")
    if {
        path.name
        for path in (logical_evidence_root / "checkpoints").iterdir()
    } != {
        f"observation_{index:06d}.json"
        for index in range(len(checkpoint_bindings))
    }:
        raise ValueError("published logical checkpoint tree contains extra data")

    canonical_root = publication_root / "canonical_scientific_artifact"
    if {path.name for path in canonical_root.iterdir()} != {
        ROLE_NEUTRAL_EXECUTION_MANIFEST,
        "artifact_reference.json",
    }:
        raise ValueError("canonical scientific artifact tree contains extra data")
    canonical = manifest.canonical_scientific_artifact
    reference, _reference_file_sha256, _reference_file_size = _read_private_json(
        publication_root / str(canonical["relative_reference_path"]),
        label="canonical scientific artifact reference",
    )
    (
        execution_manifest,
        execution_manifest_file_sha256,
        execution_manifest_file_size,
    ) = _read_private_json(
        publication_root / str(canonical["relative_manifest_path"]),
        label="canonical scientific execution manifest",
    )
    _closed_content_identity(
        reference,
        label="canonical scientific artifact reference",
    )
    if (
        reference.get("schema_version")
        != ROLE_NEUTRAL_BENCHMARK_CANONICAL_ARTIFACT_REFERENCE_SCHEMA
        or reference.get("content_sha256")
        != canonical.get("reference_content_sha256")
        or reference.get("complete_raw_artifact_tree_retained") is not False
        or reference.get(
            "future_scientific_replay_possible_from_publication"
        )
        is not False
        or execution_manifest.get("content_sha256")
        != canonical.get("manifest_content_sha256")
        or execution_manifest_file_sha256
        != reference.get("manifest_file_sha256")
        or execution_manifest_file_size
        != reference.get("manifest_file_size")
        or execution_manifest.get("scientific_identity", {}).get(
            "content_sha256"
        )
        != canonical.get("scientific_artifact_sha256")
        or not isinstance(
            reference.get("path_neutral_scientific_binding"),
            Mapping,
        )
        or reference["path_neutral_scientific_binding"].get(
            "content_sha256"
        )
        != canonical.get(
            "path_neutral_scientific_binding_content_sha256"
        )
    ):
        raise ValueError("canonical scientific artifact evidence changed")
    tree_inventory = reference.get("tree_inventory")
    binding = reference.get("path_neutral_scientific_binding")
    if (
        not isinstance(tree_inventory, list)
        or not tree_inventory
        or reference.get("tree_sha256") != identity_sha256(tree_inventory)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"relative_path", "size_bytes", "sha256"}
            or isinstance(row["size_bytes"], bool)
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] < 0
            or _SHA256.fullmatch(str(row["sha256"])) is None
            for row in tree_inventory
        )
        or reference.get("total_file_bytes")
        != sum(int(row["size_bytes"]) for row in tree_inventory)
        or reference.get("file_count") != len(tree_inventory)
        or not isinstance(binding, Mapping)
        or binding.get("scope_label") != reference.get("scope_label")
        or binding.get("scientific_artifact_sha256")
        != reference.get("scientific_artifact_sha256")
        or binding.get("plan_scientific_content_sha256")
        != reference.get("plan_scientific_content_sha256")
        or binding.get("content_sha256")
        != identity_sha256(
            {
                "scope_label": binding.get("scope_label"),
                "scientific_artifact_sha256": binding.get(
                    "scientific_artifact_sha256"
                ),
                "plan_scientific_content_sha256": binding.get(
                    "plan_scientific_content_sha256"
                ),
            }
        )
        or not any(
            row.get("relative_path") == ROLE_NEUTRAL_EXECUTION_MANIFEST
            and row.get("sha256") == execution_manifest_file_sha256
            and row.get("size_bytes") == execution_manifest_file_size
            for row in tree_inventory
            if isinstance(row, Mapping)
        )
    ):
        raise ValueError("canonical scientific artifact tree proof changed")
    _validate_execution_manifest(
        execution_manifest,
        expected_manifest_content_sha256=str(
            reference["manifest_content_sha256"]
        ),
        expected_scientific_artifact_sha256=str(
            reference["scientific_artifact_sha256"]
        ),
        expected_plan_sha256=str(
            reference["plan_scientific_content_sha256"]
        ),
    )

    path_neutral_result = _read_private_json(
        publication_root
        / str(manifest.path_neutral_result_evidence["relative_path"]),
        label="path-neutral benchmark result evidence",
    )[0]
    _closed_content_identity(
        path_neutral_result,
        label="path-neutral benchmark result evidence",
    )
    if path_neutral_result.get("content_sha256") != (
        manifest.path_neutral_result_evidence.get("content_sha256")
    ):
        raise ValueError("path-neutral benchmark result registration changed")
    expected_path_neutral_result_body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_PATH_NEUTRAL_RESULT_SCHEMA,
        "benchmark_result_schema_version": result["schema_version"],
        "normalized_benchmark_result": {
            "schema_version": result["schema_version"],
            "status": result["status"],
            "config": copy.deepcopy(result["config"]),
            "config_sha256": result["config_sha256"],
            "scientific_workload_binding": _path_neutral_workload_binding(
                result["workload_binding"]
            ),
            "execution_schedule": copy.deepcopy(
                result["execution_schedule"]
            ),
            "candidate_results": _redact_absolute_locators(
                result["candidate_results"]
            ),
            "benchmark_matrix_coverage": _redact_absolute_locators(
                result["benchmark_matrix_coverage"]
            ),
            "selected_candidate": result["selected_candidate"],
            "selection_policy": result["selection_policy"],
            "scientific_result_identity_sha256": result[
                "scientific_result_identity_sha256"
            ],
            "accepted": result["accepted"],
            "warmup_observations_excluded_from_selection": result[
                "warmup_observations_excluded_from_selection"
            ],
            "ordinary_observations_exclude_terminal_audit": result[
                "ordinary_observations_exclude_terminal_audit"
            ],
            "preflight_compression_benchmark": compression_evidence,
            "historical_result_content_identity_retained": False,
            "physical_observation_and_terminal_audit_records_retained": False,
        },
        "logical_checkpoint_proof_content_sha256": [
            row["content_sha256"] for row in logical_proof_hashes
        ],
        "canonical_path_neutral_scientific_binding_content_sha256": (
            reference["path_neutral_scientific_binding"][
                "content_sha256"
            ]
        ),
        "historical_absolute_locators_redacted": True,
        "historical_scratch_locators_authoritative": False,
    }
    expected_path_neutral_result = {
        **expected_path_neutral_result_body,
        "content_sha256": identity_sha256(
            expected_path_neutral_result_body
        ),
    }
    if path_neutral_result != expected_path_neutral_result:
        raise ValueError("path-neutral benchmark result evidence changed")

    path_neutral_inputs = manifest.path_neutral_result_evidence.get(
        "path_neutral_content_root_inputs"
    )
    expected_path_neutral_inputs = {
        "publication_producer_code_sha256": (
            _publication_producer_code_sha256()
        ),
        "config_sha256": result["config_sha256"],
        "scientific_workload_binding_content_sha256": (
            _path_neutral_workload_binding(
                result["workload_binding"]
            )["content_sha256"]
        ),
        "execution_schedule_content_sha256": result["execution_schedule"][
            "content_sha256"
        ],
        "scientific_result_identity_sha256": result[
            "scientific_result_identity_sha256"
        ],
        "path_neutral_result_evidence_content_sha256": path_neutral_result[
            "content_sha256"
        ],
        "logical_checkpoint_proof_root_sha256": checkpoint_root,
        "logical_interrupted_proof_root_sha256": interrupted_root,
        "compression_path_neutral_evidence_content_sha256": (
            compression_evidence["content_sha256"]
        ),
        "canonical_path_neutral_scientific_binding_content_sha256": (
            reference["path_neutral_scientific_binding"][
                "content_sha256"
            ]
        ),
        "scientific_workflow_binding_content_sha256": (
            validated_scientific_workflow_binding["content_sha256"]
        ),
    }
    if (
        path_neutral_inputs != expected_path_neutral_inputs
        or identity_sha256(expected_path_neutral_inputs)
        != manifest.path_neutral_content_root_sha256
    ):
        raise ValueError("benchmark publication path-neutral root changed")
    policy = manifest.source_record_policy
    if (
        policy.get("historical_scratch_locators_authoritative") is not False
        or policy.get("durable_path_neutral_evidence_is_authoritative") is not True
        or policy.get("complete_fit_replica_trees_published") is not False
        or policy.get("future_complete_scientific_replay_from_publication")
        is not False
    ):
        raise ValueError("benchmark publication overclaims retained evidence")
    return manifest


def load_role_neutral_benchmark_selection_evidence(
    root: Path | str,
) -> RoleNeutralBenchmarkSelectionEvidence:
    """Freshly validate and reopen the path-neutral selection authority."""

    publication_root = _canonical_private_directory(
        Path(root),
        label="durable benchmark publication root",
    )
    manifest = validate_role_neutral_performance_benchmark_publication(
        publication_root
    )
    manifest_path = (
        publication_root / ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST
    )
    manifest_value, manifest_file_sha256, _manifest_size = (
        _read_private_json(
            manifest_path,
            label="benchmark publication terminal manifest",
        )
    )
    if manifest_value != manifest.as_dict():
        raise RuntimeError(
            "benchmark publication manifest changed after fresh validation"
        )

    path_neutral_result = _read_private_json(
        publication_root
        / str(manifest.path_neutral_result_evidence["relative_path"]),
        label="path-neutral benchmark selection result",
    )[0]
    if (
        path_neutral_result.get("content_sha256")
        != manifest.path_neutral_result_evidence.get("content_sha256")
    ):
        raise ValueError(
            "path-neutral benchmark selection result changed"
        )
    normalized = path_neutral_result.get("normalized_benchmark_result")
    if (
        not isinstance(normalized, Mapping)
        or set(normalized) != _PATH_NEUTRAL_NORMALIZED_RESULT_FIELDS
        or normalized.get("historical_result_content_identity_retained")
        is not False
        or normalized.get(
            "physical_observation_and_terminal_audit_records_retained"
        )
        is not False
    ):
        raise ValueError(
            "path-neutral benchmark selection result is not closed"
        )

    workload_binding = _read_private_json(
        publication_root / "logical_evidence" / "workload_binding.json",
        label="published benchmark selection workload binding",
    )[0]
    validated_workload = _validate_workload_binding(workload_binding)
    source_binding = RoleNeutralBenchmarkSourceBinding(
        **dict(validated_workload["source"])
    )
    scientific_workflow_binding = _read_private_json(
        publication_root
        / "logical_evidence"
        / "scientific_workflow_binding.json",
        label="published benchmark selection scientific workflow binding",
    )[0]
    scientific_workflow_binding = _validate_scientific_workflow_binding(
        scientific_workflow_binding,
        expected_workflow_scientific_sha256=(
            source_binding.workflow_scientific_sha256
        ),
    )
    bindings = manifest.benchmark_bindings
    result_binding = bindings.get("benchmark_result")
    if (
        not isinstance(result_binding, Mapping)
        or result_binding.get("file_sha256")
        != _require_sha256(
            result_binding.get("file_sha256"),
            label="published benchmark result file",
        )
        or result_binding.get("content_sha256")
        != _require_sha256(
            result_binding.get("content_sha256"),
            label="published benchmark result content",
        )
        or bindings.get("workload_binding_content_sha256")
        != validated_workload["content_sha256"]
        or bindings.get("workload_deployment_sha256")
        != source_binding.workload_deployment_sha256
        or bindings.get(
            "scientific_workflow_binding_content_sha256"
        )
        != scientific_workflow_binding["content_sha256"]
    ):
        raise ValueError(
            "benchmark publication selection authority changed"
        )
    return RoleNeutralBenchmarkSelectionEvidence(
        publication_root=publication_root,
        publication_manifest_path=manifest_path,
        publication_manifest_file_sha256=manifest_file_sha256,
        publication_manifest=manifest,
        normalized_benchmark_result=normalized,
        workload_binding=validated_workload,
        scientific_workflow_binding=scientific_workflow_binding,
        source_binding=source_binding,
        benchmark_result_file_sha256=str(
            result_binding["file_sha256"]
        ),
        benchmark_result_content_sha256=str(
            result_binding["content_sha256"]
        ),
    )


__all__ = [
    "ROLE_NEUTRAL_BENCHMARK_CANONICAL_ARTIFACT_REFERENCE_SCHEMA",
    "ROLE_NEUTRAL_BENCHMARK_LOGICAL_CHECKPOINT_SCHEMA",
    "ROLE_NEUTRAL_BENCHMARK_PATH_NEUTRAL_RESULT_SCHEMA",
    "ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST",
    "ROLE_NEUTRAL_BENCHMARK_PUBLICATION_SCHEMA",
    "ROLE_NEUTRAL_BENCHMARK_SCIENTIFIC_WORKFLOW_BINDING_SCHEMA",
    "PublishedBenchmarkPayload",
    "RoleNeutralBenchmarkPublicationManifest",
    "RoleNeutralBenchmarkSelectionEvidence",
    "load_role_neutral_benchmark_selection_evidence",
    "publish_role_neutral_performance_benchmark",
    "validate_role_neutral_performance_benchmark_publication",
]
