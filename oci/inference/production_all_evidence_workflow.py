"""Resumable public orchestration for the all-evidence causal workflow."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .production_authenticated_tree_cache import (
    AUTHENTICATED_DIRECTORY_TREE_POLICY,
    authenticate_directory_tree,
)
from .production_stage1_bundle import (
    ProductionStage1BundleBuilder,
    Stage1BundleBuildOptions,
)
from .production_text_preparation import (
    TextPreparationOptions,
    prepare_modeling_cohort,
    stable_file_sha256,
)

WORKFLOW_SCHEMA = "production_all_evidence_workflow_v3"
PHASES = (
    "input_preparation",
    "embedding_cache",
    "stage1_preflight",
    "stage1_modeling",
    "handoff_validation",
    "stage2_canary",
    "stage2_inference",
    "oracle_evaluation",
    "terminal_validation",
)
STAGE1_ONLY_PHASES = PHASES[:5] + ("terminal_validation",)
EMBEDDING_CACHE_PHASE_SCHEMA = "production_embedding_cache_phase_result_v1"
STAGE1_PREFLIGHT_PHASE_SCHEMA = "production_stage1_preflight_phase_result_v2"
WORKFLOW_PROGRESS_SCHEMA = "production_all_evidence_workflow_progress_v1"
WORKFLOW_PHASE_MANIFEST_SCHEMA = "production_workflow_phase_manifest_v2"
WORKFLOW_TERMINAL_VALIDATION_SCHEMA = "production_all_evidence_fresh_terminal_validation_v1"
SOURCE_SNAPSHOT_EXECUTION_ENV = "OCI_PRODUCTION_SOURCE_SNAPSHOT_SHA256"


class WorkflowPhaseHook(Protocol):
    """Injected implementation for one expensive production phase.

    Hooks receive paths and immutable scalar configuration only.  They must
    return a mapping with ``terminal_files`` so the workflow can independently
    hash and seal every published result.
    """

    def __call__(
        self,
        attempt_dir: Path,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ProductionAllEvidenceWorkflowHooks:
    """Optional cache/preflight/scheduler integrations.

    The public command uses the built-in implementations unless a hook is
    supplied by an embedding-cache relocator or parallel Stage 1 scheduler.
    Hooks are code-identity-bound into the immutable workflow request.
    """

    embedding_cache: WorkflowPhaseHook | None = None
    stage1_preflight: WorkflowPhaseHook | None = None
    stage1_modeling: WorkflowPhaseHook | None = None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace one JSON control file in its existing parent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{label} contains duplicate JSON key: {key}")
            output[key] = value
        return output

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one real regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _attempt_tree_artifacts(attempt_dir: Path) -> list[dict[str, Any]]:
    """Return the exact closed regular-file inventory for one phase attempt."""

    if attempt_dir.is_symlink() or not attempt_dir.is_dir():
        raise ValueError(f"phase attempt must be one real directory: {attempt_dir}")
    root = attempt_dir.resolve(strict=True)
    artifacts: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        state = os.lstat(candidate)
        if stat.S_ISLNK(state.st_mode):
            raise ValueError(f"phase attempt contains a symlink: {candidate}")
        if stat.S_ISDIR(state.st_mode):
            continue
        if not stat.S_ISREG(state.st_mode):
            raise ValueError(f"phase attempt contains a special file: {candidate}")
        resolved = candidate.resolve(strict=True)
        digest, size = stable_file_sha256(resolved)
        artifacts.append(
            {
                "relative_path": resolved.relative_to(root).as_posix(),
                "path": str(resolved),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return artifacts


def _validate_phase_manifest_from_paths(
    *,
    work_root: Path,
    phase: str,
    request_sha256: str,
) -> dict[str, Any]:
    """Validate one completed phase without a live workflow runner."""

    manifest_path = work_root / "phases" / phase / "complete_manifest.json"
    value = _read_json_object(manifest_path, label=f"{phase} phase manifest")
    if set(value) != {
        "schema_version",
        "phase",
        "status",
        "request_sha256",
        "attempt_dir",
        "result",
        "artifacts",
        "content_sha256",
    }:
        raise ValueError(f"completed phase manifest is not closed: {phase}")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("schema_version") != WORKFLOW_PHASE_MANIFEST_SCHEMA
        or value.get("phase") != phase
        or value.get("request_sha256") != request_sha256
        or value.get("content_sha256") != _sha(body)
        or value.get("status") != "complete"
        or not isinstance(value.get("result"), Mapping)
        or not isinstance(value.get("artifacts"), list)
    ):
        raise ValueError(f"completed phase manifest failed validation: {phase}")
    attempt = Path(str(value.get("attempt_dir", "")))
    expected_phase_root = (work_root / "phases" / phase).resolve(strict=True)
    if (
        not attempt.is_absolute()
        or attempt.is_symlink()
        or not attempt.is_dir()
        or attempt.resolve(strict=True).parent != expected_phase_root
        or not attempt.name.startswith("attempt_")
    ):
        raise ValueError(f"completed phase attempt path is invalid: {phase}")
    observed = _attempt_tree_artifacts(attempt)
    if value["artifacts"] != observed:
        raise ValueError(f"completed phase attempt tree changed: {phase}")
    terminal_files = value["result"].get("terminal_files", [])
    if (
        not isinstance(terminal_files, list)
        or any(not isinstance(path, str) for path in terminal_files)
        or len(terminal_files) != len(set(terminal_files))
    ):
        raise ValueError(f"completed phase terminal-file registry is invalid: {phase}")
    registered = {row["path"] for row in observed}
    for raw in terminal_files:
        terminal = Path(raw)
        if not terminal.is_absolute() or str(terminal.resolve(strict=True)) not in registered:
            raise ValueError(
                f"completed phase terminal file escaped its sealed attempt: {terminal}"
            )
    return value


def validate_completed_workflow_prefix(
    *,
    work_root: Path | str,
    expected_request_sha256: str,
    expected_phases: Sequence[str],
) -> dict[str, Any]:
    """Fresh path-only validation for every phase completed before terminal seal."""

    root = Path(work_root).resolve(strict=True)
    request_path = root / "immutable_run_request.json"
    request = _read_json_object(request_path, label="immutable workflow request")
    request_body = {key: value for key, value in request.items() if key != "request_sha256"}
    phases = tuple(str(value) for value in expected_phases)
    if (
        request.get("request_sha256") != expected_request_sha256
        or request.get("request_sha256") != _sha(request_body)
        or tuple(request.get("phase_sequence") or ()) != (*phases, "terminal_validation")
    ):
        raise ValueError("immutable workflow request failed fresh validation")
    _revalidate_request_bound_external_inputs(request)
    phase_records = [
        _validate_phase_manifest_from_paths(
            work_root=root,
            phase=phase,
            request_sha256=expected_request_sha256,
        )
        for phase in phases
    ]
    source_snapshot = request.get("source_snapshot")
    if source_snapshot is not None:
        if not isinstance(source_snapshot, Mapping):
            raise ValueError("source snapshot identity is invalid")
        from .production_source_snapshot import validate_production_source_snapshot

        validated_snapshot = validate_production_source_snapshot(
            Path(str(source_snapshot.get("root", "")))
        ).as_dict()
        if validated_snapshot != dict(source_snapshot):
            raise ValueError("source snapshot changed before terminal validation")
    handoff = next(
        (row for row in phase_records if row.get("phase") == "handoff_validation"),
        None,
    )
    handoff_result = None if handoff is None else handoff["result"].get("fresh_process_validation")
    handoff_validated = bool(
        isinstance(handoff_result, Mapping)
        and handoff_result.get("schema_version") == "production_stage1_fresh_handoff_validation_v1"
        and handoff_result.get("status") == "accepted"
        and handoff_result.get("remote_clients_constructed") is False
        and handoff_result.get("remote_calls_made") is False
    )
    body = {
        "schema_version": WORKFLOW_TERMINAL_VALIDATION_SCHEMA,
        "status": "accepted",
        "request_path": str(request_path.resolve(strict=True)),
        "request_sha256": expected_request_sha256,
        "validated_phases": list(phases),
        "validated_phase_manifest_sha256": {
            row["phase"]: stable_file_sha256(
                root / "phases" / row["phase"] / "complete_manifest.json"
            )[0]
            for row in phase_records
        },
        "validated_artifact_count": sum(len(row["artifacts"]) for row in phase_records),
        "stage1_handoff_validated_in_fresh_process": handoff_validated,
        "source_snapshot": source_snapshot,
        "request_bound_external_inputs_revalidated": True,
        "live_runner_objects_received": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _stable_path_identity(
    path: Path,
    *,
    reuse_process_authenticated_tree: bool = False,
) -> Mapping[str, Any]:
    """Bind one file or directory tree without trusting names alone.

    Imported embedding caches never load the embedding model after its first
    full provenance authentication. For that one lifecycle, callers may reuse
    a PID-scoped content identity while every logical check still compares the
    complete filesystem inventory. Fresh cache builds and live HTR models
    always retain full byte-tree reauthentication.
    """

    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError(f"identity-bound path cannot be a symlink: {supplied}")
    resolved = supplied.resolve(strict=True)
    if resolved.is_file():
        digest, size = stable_file_sha256(resolved)
        return {
            "kind": "file",
            "path": str(resolved),
            "sha256": digest,
            "size_bytes": size,
        }
    if not resolved.is_dir():
        raise ValueError(f"identity-bound path is not a file or directory: {resolved}")
    if reuse_process_authenticated_tree:
        return authenticate_directory_tree(resolved).workflow_path_identity()
    inventory: list[dict[str, Any]] = []
    for candidate in sorted(resolved.rglob("*")):
        if candidate.is_symlink():
            raise ValueError(f"identity-bound tree cannot contain symlinks: {candidate}")
        if not candidate.is_file():
            continue
        digest, size = stable_file_sha256(candidate)
        inventory.append(
            {
                "relative_path": candidate.relative_to(resolved).as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    if not inventory:
        raise ValueError(f"identity-bound directory has no files: {resolved}")
    return {
        "kind": "directory",
        "path": str(resolved),
        "file_count": len(inventory),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in inventory),
        "tree_sha256": _sha(inventory),
        "files": inventory,
    }


def _revalidate_request_bound_external_inputs(
    request: Mapping[str, Any],
) -> None:
    """Reopen every external input whose bytes were bound into the run request."""

    def require_file_hash(
        *,
        path_field: str,
        sha_field: str,
        label: str,
    ) -> None:
        raw_path = request.get(path_field)
        expected = request.get(sha_field)
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ValueError(f"immutable workflow request lacks {label} identity")
        observed, _size = stable_file_sha256(Path(raw_path).resolve(strict=True))
        if observed != expected:
            raise RuntimeError(f"{label} changed after workflow initialization")

    require_file_hash(
        path_field="dataset_path",
        sha_field="source_sha256",
        label="source dataset",
    )
    require_file_hash(
        path_field="stage1_profile_path",
        sha_field="stage1_profile_sha256",
        label="Stage 1 profile",
    )
    require_file_hash(
        path_field="query_profile_path",
        sha_field="query_profile_sha256",
        label="neural-query profile",
    )

    cache_inputs = request.get("embedding_cache_import_inputs")
    imported_embedding_cache = cache_inputs is not None
    expected_model_policy = (
        AUTHENTICATED_DIRECTORY_TREE_POLICY
        if imported_embedding_cache
        else "full_byte_tree_reauthentication_v1"
    )
    if request.get("embedding_model_revalidation_policy") != expected_model_policy:
        raise ValueError(
            "immutable workflow request has an invalid embedding-model " "revalidation policy"
        )

    for field, label in (
        ("embedding_model_tree", "embedding model tree"),
        ("htr_model_tree", "HTR model tree"),
    ):
        expected = request.get(field)
        if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
            raise ValueError(f"immutable workflow request lacks {label} identity")
        if _stable_path_identity(
            Path(str(expected["path"])),
            reuse_process_authenticated_tree=(
                field == "embedding_model_tree" and imported_embedding_cache
            ),
        ) != dict(expected):
            raise RuntimeError(f"{label} changed after workflow initialization")

    if cache_inputs is not None:
        if not isinstance(cache_inputs, Mapping) or set(cache_inputs) != {
            "cache",
            "prepared_cohort",
            "preparation_manifest",
        }:
            raise ValueError("immutable workflow request has an invalid cache-import identity")
        for name, expected in cache_inputs.items():
            if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
                raise ValueError(f"immutable workflow request lacks cache-import {name} identity")
            if _stable_path_identity(
                Path(str(expected["path"])),
                reuse_process_authenticated_tree=(name == "cache"),
            ) != dict(expected):
                raise RuntimeError(f"cache-import {name} changed after workflow initialization")

    implementation_files = request.get("implementation_files")
    if not isinstance(implementation_files, Mapping) or not implementation_files:
        raise ValueError("immutable workflow request lacks implementation identities")
    for raw_path, expected_sha in implementation_files.items():
        observed_sha, _size = stable_file_sha256(Path(str(raw_path)).resolve(strict=True))
        if observed_sha != expected_sha:
            raise RuntimeError("workflow implementation changed after workflow initialization")

    for collection_name in ("integration_hooks", "phase_overrides"):
        collection = request.get(collection_name)
        if not isinstance(collection, Mapping):
            raise ValueError(f"immutable workflow request lacks {collection_name} identities")
        for identity in collection.values():
            if identity is None:
                continue
            if not isinstance(identity, Mapping):
                raise ValueError(
                    f"immutable workflow request has invalid {collection_name} identity"
                )
            source_file = identity.get("source_file")
            if source_file is None:
                continue
            if not isinstance(source_file, Mapping) or not isinstance(source_file.get("path"), str):
                raise ValueError(f"immutable workflow request has invalid {collection_name} source")
            observed_sha, observed_size = stable_file_sha256(
                Path(str(source_file["path"])).resolve(strict=True)
            )
            if observed_sha != source_file.get("sha256") or observed_size != int(
                source_file.get("size_bytes", -1)
            ):
                raise RuntimeError(f"{collection_name} implementation changed after initialization")

    source_snapshot = request.get("source_snapshot")
    if source_snapshot is not None:
        if not isinstance(source_snapshot, Mapping):
            raise ValueError("immutable workflow request has invalid source snapshot")
        from .production_source_snapshot import validate_production_source_snapshot

        observed_snapshot = validate_production_source_snapshot(
            Path(str(source_snapshot.get("root", "")))
        ).as_dict()
        if observed_snapshot != dict(source_snapshot):
            raise RuntimeError("source snapshot changed after workflow initialization")


def validate_stage1_canary_descriptor_preparation(
    work_root: Path | str,
) -> Mapping[str, Any]:
    """Fresh path-only validation of the pre-fit canary preparation boundary."""

    supplied = Path(work_root)
    if (
        not supplied.is_absolute()
        or supplied.is_symlink()
        or not supplied.is_dir()
        or supplied.resolve(strict=True) != supplied
    ):
        raise ValueError("canary preparation work root is invalid")
    root = supplied
    request = _read_json_object(
        root / "immutable_run_request.json",
        label="immutable workflow request",
    )
    request_body = {key: value for key, value in request.items() if key != "request_sha256"}
    request_sha = request.get("request_sha256")
    if (
        request_sha != _sha(request_body)
        or request.get("stage1_only") is not True
        or request.get("phase_sequence") != list(STAGE1_ONLY_PHASES)
        or not isinstance(request.get("source_snapshot"), Mapping)
    ):
        raise ValueError("canary preparation workflow request is invalid")
    _revalidate_request_bound_external_inputs(request)
    prefix = ("input_preparation", "embedding_cache", "stage1_preflight")
    phase_records = {
        phase: _validate_phase_manifest_from_paths(
            work_root=root,
            phase=phase,
            request_sha256=str(request_sha),
        )
        for phase in prefix
    }
    path = root / "recovery" / "canary_descriptor_preparation_manifest.json"
    manifest = _read_json_object(
        path,
        label="canary descriptor preparation manifest",
    )
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    expected_fields = {
        "schema_version",
        "status",
        "workflow_request_sha256",
        "stage1_request_sha256",
        "source_snapshot",
        "completed_workflow_prefix",
        "cluster_preflight_manifest",
        "stage1_preflight_phase_manifest",
        "descriptor_set_manifest",
        "descriptor_set_content_sha256",
        "descriptor_count",
        "selected_scope_id",
        "selected_scope_kind",
        "selected_logical_gpu_id",
        "selected_descriptor_manifest",
        "supervised_stage1_fits_started",
        "tfidf_component_started",
        "neural_query_component_started",
        "remote_clients_constructed",
        "remote_calls_made",
        "content_sha256",
    }
    if (
        set(manifest) != expected_fields
        or manifest.get("schema_version") != "production_stage1_canary_descriptor_preparation_v1"
        or manifest.get("status") != "complete"
        or manifest.get("content_sha256") != _sha(body)
        or manifest.get("workflow_request_sha256") != request_sha
        or manifest.get("source_snapshot") != request.get("source_snapshot")
        or manifest.get("completed_workflow_prefix") != list(prefix)
        or manifest.get("selected_scope_kind") != "full_outer"
        or manifest.get("selected_logical_gpu_id") != 0
        or manifest.get("supervised_stage1_fits_started") is not False
        or manifest.get("tfidf_component_started") is not False
        or manifest.get("neural_query_component_started") is not False
        or manifest.get("remote_clients_constructed") is not False
        or manifest.get("remote_calls_made") is not False
    ):
        raise ValueError("canary descriptor preparation manifest is invalid")

    def validate_registration(
        value: Any,
        *,
        label: str,
    ) -> Path:
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"{label} registration is invalid")
        registered = Path(str(value["path"]))
        if (
            not registered.is_absolute()
            or registered.is_symlink()
            or not registered.is_file()
            or registered.resolve(strict=True) != registered
        ):
            raise ValueError(f"{label} path is invalid")
        digest, size = stable_file_sha256(registered)
        if digest != value.get("sha256") or size != int(value.get("size_bytes", -1)):
            raise ValueError(f"{label} changed")
        return registered

    preflight_artifact = validate_registration(
        manifest["cluster_preflight_manifest"],
        label="cluster preflight manifest",
    )
    preflight_phase = validate_registration(
        manifest["stage1_preflight_phase_manifest"],
        label="Stage 1 preflight phase manifest",
    )
    descriptor_set_manifest = validate_registration(
        manifest["descriptor_set_manifest"],
        label="descriptor-set manifest",
    )
    selected_manifest = validate_registration(
        manifest["selected_descriptor_manifest"],
        label="selected descriptor manifest",
    )
    registered_preflight_files = {
        Path(row["path"]).resolve(strict=True)
        for row in phase_records["stage1_preflight"]["artifacts"]
    }
    if (
        preflight_artifact not in registered_preflight_files
        or preflight_phase != root / "phases" / "stage1_preflight" / "complete_manifest.json"
    ):
        raise ValueError("canary preparation preflight registration changed")

    from .production_stage1_legacy_scope_adapter import (
        validate_legacy_stage1_scope_descriptor_set,
    )

    descriptor_set = validate_legacy_stage1_scope_descriptor_set(
        descriptor_root=descriptor_set_manifest.parent,
        expected_stage1_request_sha256=str(manifest["stage1_request_sha256"]),
    )
    expected_count = int(request["outer_folds"]) * (
        1 + 3 + int(request["review_rounds"]) + int(request["review_rounds"])
    )
    selected_scope_id = str(manifest["selected_scope_id"])
    selected = descriptor_set.descriptors.get(selected_scope_id)
    if (
        len(descriptor_set.descriptors) != expected_count
        or manifest.get("descriptor_count") != expected_count
        or descriptor_set.manifest.get("content_sha256")
        != manifest.get("descriptor_set_content_sha256")
        or selected is None
        or selected.manifest_path != selected_manifest
        or selected.scope.scope_kind != "full_outer"
        or int(selected.assignment.gpu_id) != 0
    ):
        raise ValueError("canary descriptor set or selected scope changed")
    if (root / "phases" / "stage1_modeling").exists() or (
        root / "recovery" / "tfidf_component_recovery"
    ).exists():
        raise ValueError("supervised Stage 1 work began before the canary")
    return copy.deepcopy(manifest)


def _hook_identity(hook: WorkflowPhaseHook | None) -> Mapping[str, Any] | None:
    if hook is None:
        return None
    target = hook if inspect.isfunction(hook) else hook.__call__
    source = inspect.getsourcefile(target)
    identity: dict[str, Any] = {
        "module": str(getattr(target, "__module__", type(hook).__module__)),
        "qualname": str(getattr(target, "__qualname__", type(hook).__qualname__)),
    }
    if source is not None and Path(source).is_file():
        digest, size = stable_file_sha256(Path(source).resolve())
        identity["source_file"] = {
            "path": str(Path(source).resolve()),
            "sha256": digest,
            "size_bytes": size,
        }
    return identity


def _resolved_stage1_gpu_ids(options: "ProductionAllEvidenceWorkflowOptions") -> tuple[int, ...]:
    plural = tuple(int(value) for value in options.stage1_gpu_ids)
    singular = None if options.gpu_id is None else int(options.gpu_id)
    if plural and singular is not None and plural != (singular,):
        raise ValueError("--gpu-id conflicts with ordered --stage1-gpu-id values")
    resolved = plural or (() if singular is None else (singular,))
    if not resolved and str(options.stage1_device).startswith("cuda:"):
        try:
            resolved = (int(str(options.stage1_device).split(":", 1)[1]),)
        except ValueError as exc:
            raise ValueError("stage1_device must name one explicit CUDA index") from exc
    if any(value < 0 for value in resolved) or len(set(resolved)) != len(resolved):
        raise ValueError("Stage 1 GPU IDs must be nonnegative, unique, and ordered")
    return resolved


def _resolved_query_devices(options: "ProductionAllEvidenceWorkflowOptions") -> tuple[str, ...]:
    plural = tuple(str(value) for value in options.query_devices)
    singular = None if options.query_device is None else str(options.query_device)
    if plural and singular is not None and plural != (singular,):
        raise ValueError("query_device conflicts with ordered query_devices")
    resolved = plural or (() if singular is None else (singular,))
    return resolved or (str(options.stage1_device),)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ProductionAllEvidenceWorkflowOptions:
    dataset_path: Path
    work_root: Path
    stage1_profile_path: Path
    query_profile_path: Path
    unit_id_column: str
    text_column: str
    treatment_column: str
    outcome_column: str
    outcome_type: str
    clinical_question: str
    embedding_model_name: str
    embedding_local_model_path: Path
    htr_local_model_path: Path
    endpoint: str | None = None
    model_name: str | None = None
    outer_folds: int = 5
    review_rounds: int = 2
    interaction_inner_folds: int = 3
    tfidf_nested_calibration_folds: int = 3
    stage1_device: str = "cuda:1"
    query_device: str | None = None
    query_devices: tuple[str, ...] = ()
    review_device: str = "cuda:1"
    gpu_id: int | None = None
    stage1_gpu_ids: tuple[int, ...] = ()
    stage1_scope_workers_per_gpu: int = 1
    stage1_preflight_workers: int = 8
    stage1_seed_policy: str = "canonical_scope_sha256_v1"
    num_workers: int = 1
    tfidf_workers: int = 8
    tfidf_parallel_backend: str = "processes"
    seed: int = 42
    empty_text_policy: str = "marker"
    repeated_character_policy: str = "marker"
    repeated_character_threshold: int = 1000
    evaluate_oracle_posthoc: bool = False
    oracle_dataset_path: Path | None = None
    oracle_unit_id_column: str | None = None
    oracle_ite_column: str | None = None
    embedding_cache_import: Path | None = None
    embedding_cache_import_source_prepared_path: Path | None = None
    embedding_cache_import_source_preparation_manifest_path: Path | None = None
    source_snapshot_root: Path | None = None
    stage1_only: bool = False
    resume: bool = False


class ProductionAllEvidenceWorkflow:
    """Fail-closed phase runner; completed phases are content-addressed."""

    def __init__(
        self,
        options: ProductionAllEvidenceWorkflowOptions,
        *,
        phase_overrides: Mapping[str, Callable[[Path], Mapping[str, Any]]] | None = None,
        hooks: ProductionAllEvidenceWorkflowHooks | None = None,
    ) -> None:
        self.options = options
        self.phase_overrides = dict(phase_overrides or {})
        self.hooks = hooks or ProductionAllEvidenceWorkflowHooks()
        self.request: dict[str, Any] = {}
        self._validate_options()

    @property
    def stage1_gpu_ids(self) -> tuple[int, ...]:
        return _resolved_stage1_gpu_ids(self.options)

    @property
    def query_devices(self) -> tuple[str, ...]:
        return _resolved_query_devices(self.options)

    def _phase_sequence(self) -> tuple[str, ...]:
        return STAGE1_ONLY_PHASES if self.options.stage1_only else PHASES

    def _resolved_cache_import_sources(self) -> tuple[Path, Path] | None:
        o = self.options
        if o.embedding_cache_import is None:
            return None
        if (
            o.embedding_cache_import_source_prepared_path is not None
            and o.embedding_cache_import_source_preparation_manifest_path is not None
        ):
            return (
                o.embedding_cache_import_source_prepared_path.resolve(strict=True),
                o.embedding_cache_import_source_preparation_manifest_path.resolve(strict=True),
            )
        metadata_path = o.embedding_cache_import / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        provenance = metadata.get("production_provenance")
        dataset = provenance.get("dataset") if isinstance(provenance, Mapping) else None
        source = dataset.get("path") if isinstance(dataset, Mapping) else None
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "embedding-cache metadata does not identify its prepared cohort; "
                "supply the two explicit source-preparation options"
            )
        prepared = Path(source).resolve(strict=True)
        manifest = (prepared.parent / "preparation_manifest.json").resolve(strict=True)
        return prepared, manifest

    def _validate_options(self) -> None:
        o = self.options
        stage1_gpu_ids = _resolved_stage1_gpu_ids(o)
        query_devices = _resolved_query_devices(o)
        valid_device = lambda value: value == "cpu" or (
            value.startswith("cuda:") and value.split(":", 1)[1].isdigit()
        )
        if not valid_device(str(o.stage1_device)):
            raise ValueError("stage1_device must be cpu or one explicit cuda:N device")
        if (
            not query_devices
            or any(not valid_device(value) for value in query_devices)
            or len(query_devices) != len(set(query_devices))
        ):
            raise ValueError("query devices must contain unique explicit cpu/cuda:N values")
        if not valid_device(str(o.review_device)):
            raise ValueError("review_device must be cpu or one explicit cuda:N device")
        cuda_devices = {
            int(value.split(":", 1)[1])
            for value in (str(o.stage1_device), *query_devices)
            if value.startswith("cuda:")
        }
        if not o.stage1_only and str(o.review_device).startswith("cuda:"):
            cuda_devices.add(int(str(o.review_device).split(":", 1)[1]))
        if not cuda_devices.issubset(set(stage1_gpu_ids)):
            raise ValueError(
                "every Stage 1/query CUDA device must be included in the exclusive "
                "Stage 1 GPU IDs"
            )
        if any(
            value < 1
            for value in (
                o.stage1_scope_workers_per_gpu,
                o.stage1_preflight_workers,
                o.num_workers,
                o.tfidf_workers,
            )
        ):
            raise ValueError("Stage 1 and TF-IDF worker counts must be positive")
        if o.stage1_scope_workers_per_gpu != 1:
            raise ValueError("production Stage 1 requires exactly one scope worker per GPU")
        if o.stage1_seed_policy != "canonical_scope_sha256_v1":
            raise ValueError("stage1_seed_policy must be canonical_scope_sha256_v1")
        if (
            o.outer_folds < 2
            or o.review_rounds < 1
            or o.interaction_inner_folds < 2
            or o.tfidf_nested_calibration_folds < 2
            or o.repeated_character_threshold < 1
            or o.seed < 0
        ):
            raise ValueError("fold, review, threshold, and seed settings are invalid")
        if o.outcome_type != "binary":
            raise ValueError("this production workflow currently requires binary outcomes")
        if o.empty_text_policy != "marker" or o.repeated_character_policy != "marker":
            raise ValueError("production text preparation requires both neutral marker policies")
        if o.tfidf_parallel_backend not in {"threads", "processes"}:
            raise ValueError("unsupported TF-IDF parallel backend")
        if o.stage1_only:
            if o.evaluate_oracle_posthoc:
                raise ValueError("Stage-1-only mode cannot request oracle evaluation")
        elif not (
            isinstance(o.endpoint, str)
            and o.endpoint.strip()
            and isinstance(o.model_name, str)
            and o.model_name.strip()
        ):
            raise ValueError("the full workflow requires one endpoint and exact model name")
        auxiliary_import_values = (
            o.embedding_cache_import_source_prepared_path,
            o.embedding_cache_import_source_preparation_manifest_path,
        )
        if o.embedding_cache_import is None and any(
            value is not None for value in auxiliary_import_values
        ):
            raise ValueError(
                "embedding-cache source-preparation options require " "--embedding-cache-import"
            )
        if sum(value is not None for value in auxiliary_import_values) == 1:
            raise ValueError(
                "embedding-cache import source preparation requires both its cohort " "and manifest"
            )
        if o.evaluate_oracle_posthoc and not all(
            (o.oracle_dataset_path, o.oracle_unit_id_column, o.oracle_ite_column)
        ):
            raise ValueError("post-hoc oracle evaluation requires its dataset, ID, and ITE column")
        if o.source_snapshot_root is not None:
            from .production_source_snapshot import validate_production_source_snapshot

            snapshot = validate_production_source_snapshot(o.source_snapshot_root)
            loaded_root = Path(__file__).resolve().parents[2]
            marker = os.environ.get(SOURCE_SNAPSHOT_EXECUTION_ENV)
            if loaded_root != snapshot.root or marker != snapshot.content_sha256:
                raise ValueError(
                    "source_snapshot_root requires execution from that authenticated "
                    "snapshot; use the public CLI so it can re-exec safely"
                )

    def _request_body(self) -> dict[str, Any]:
        values = json.loads(json.dumps(asdict(self.options), default=str))
        values.pop("resume")
        values["schema_version"] = WORKFLOW_SCHEMA
        values["transport_retries"] = 0
        values["schema_repairs"] = 1
        values["extraction_context_strategy"] = "complete_paged_v1"
        values["final_estimator"] = "strict_outer_honest_final_context_fit_causal_forest_v2"
        values["phase_sequence"] = list(self._phase_sequence())
        values["resolved_stage1_gpu_ids"] = list(self.stage1_gpu_ids)
        values["resolved_query_devices"] = list(self.query_devices)
        values["stage1_resource_contract"] = {
            "scope_workers_per_gpu": self.options.stage1_scope_workers_per_gpu,
            "preflight_workers": self.options.stage1_preflight_workers,
            "tfidf_workers": self.options.tfidf_workers,
            "tfidf_parallel_backend": self.options.tfidf_parallel_backend,
            "seed": self.options.seed,
            "scope_seed_policy": self.options.stage1_seed_policy,
            "exclusive_gpu_preflight_required": bool(self.stage1_gpu_ids),
        }
        values["source_sha256"] = stable_file_sha256(self.options.dataset_path)[0]
        values["stage1_profile_sha256"] = stable_file_sha256(self.options.stage1_profile_path)[0]
        values["query_profile_sha256"] = stable_file_sha256(self.options.query_profile_path)[0]
        imported_embedding_cache = self.options.embedding_cache_import is not None
        values["embedding_model_revalidation_policy"] = (
            AUTHENTICATED_DIRECTORY_TREE_POLICY
            if imported_embedding_cache
            else "full_byte_tree_reauthentication_v1"
        )
        values["embedding_model_tree"] = _stable_path_identity(
            self.options.embedding_local_model_path,
            reuse_process_authenticated_tree=imported_embedding_cache,
        )
        values["htr_model_tree"] = _stable_path_identity(self.options.htr_local_model_path)
        if imported_embedding_cache:
            source_prepared, source_manifest = self._resolved_cache_import_sources()
            values["embedding_cache_import_inputs"] = {
                "cache": _stable_path_identity(
                    self.options.embedding_cache_import,
                    reuse_process_authenticated_tree=True,
                ),
                "prepared_cohort": _stable_path_identity(source_prepared),
                "preparation_manifest": _stable_path_identity(source_manifest),
            }
        if self.options.source_snapshot_root is not None:
            from .production_source_snapshot import validate_production_source_snapshot

            values["source_snapshot"] = validate_production_source_snapshot(
                self.options.source_snapshot_root
            ).as_dict()
        values["integration_hooks"] = {
            "embedding_cache": _hook_identity(self.hooks.embedding_cache),
            "stage1_preflight": _hook_identity(self.hooks.stage1_preflight),
            "stage1_modeling": _hook_identity(self.hooks.stage1_modeling),
        }
        values["phase_overrides"] = {
            phase: _hook_identity(self.phase_overrides.get(phase))
            for phase in self._phase_sequence()
        }
        implementation_files = (
            Path(__file__).resolve(),
            Path(__file__).with_name("production_text_preparation.py").resolve(),
            Path(__file__).with_name("production_oracle_evaluation.py").resolve(),
            Path(__file__).with_name("production_authenticated_tree_cache.py").resolve(),
            Path(__file__).with_name("production_embedding_cache_relocation.py").resolve(),
            Path(__file__).parents[1] / "extraction" / "complete_paged.py",
            Path(__file__).with_name("production_source_snapshot.py").resolve(),
            Path(__file__).with_name("production_stage1_cluster_preflight_artifact.py").resolve(),
            Path(__file__).parents[2] / "scripts" / "run_production_all_evidence_workflow.py",
        )
        values["implementation_files"] = {
            str(path.resolve()): stable_file_sha256(path.resolve())[0]
            for path in implementation_files
        }
        values["stage1_recovery_contract"] = {
            "scope_attempt_root": str(
                (self.options.work_root / "recovery" / "stage1_scope_attempts").resolve()
            ),
            "scope_progress_path": str(
                (self.options.work_root / "recovery" / "stage1_scope_progress.json").resolve()
            ),
            "scope_reuse_policy": "individually_sealed_matching_scope_attempts_only_v1",
        }
        return values

    def _initialize(self) -> None:
        root = self.options.work_root
        body = self._request_body()
        request = {**body, "request_sha256": _sha(body)}
        request_path = root / "immutable_run_request.json"
        if root.exists():
            if not self.options.resume or not request_path.is_file():
                raise ValueError("work root must be fresh unless --resume validates its request")
            existing = _read_json_object(request_path, label="immutable workflow request")
            if existing != request:
                raise ValueError("--resume request differs from the immutable run request")
        else:
            root.parent.mkdir(parents=True, exist_ok=True)
            initialization_attempt = Path(
                tempfile.mkdtemp(
                    prefix=f".{root.name}.initialization_attempt_",
                    dir=root.parent,
                )
            )
            staged_request = initialization_attempt / request_path.name
            # Preserve an interrupted initialization attempt for audit, while
            # keeping the requested work root absent and therefore reusable.
            # The root becomes visible only after the immutable request has
            # been durably written and reopened byte-for-byte.
            _atomic_write_json(staged_request, request)
            if (
                _read_json_object(
                    staged_request,
                    label="staged immutable workflow request",
                )
                != request
            ):
                raise RuntimeError("staged immutable workflow request changed")
            attempt_fd = os.open(
                initialization_attempt,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(attempt_fd)
            finally:
                os.close(attempt_fd)
            if root.exists() or root.is_symlink():
                raise ValueError("work root was populated during initialization")
            os.rename(initialization_attempt, root)
            parent_fd = os.open(
                root.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        self.request = request
        self._write_progress(status="initialized", completed=(), current_phase=None)

    def _write_progress(
        self,
        *,
        status: str,
        completed: Sequence[str],
        current_phase: str | None,
        error: str | None = None,
    ) -> None:
        sequence = self._phase_sequence()
        body = {
            "schema_version": WORKFLOW_PROGRESS_SCHEMA,
            "request_sha256": self.request.get("request_sha256"),
            "status": status,
            "phase_sequence": list(sequence),
            "planned_phase_count": len(sequence),
            "completed_phases": list(completed),
            "completed_phase_count": len(completed),
            "current_phase": current_phase,
            "remaining_phase_count": len(sequence) - len(completed),
            "stage1_gpu_ids": list(self.stage1_gpu_ids),
            "stage1_scope_workers_per_gpu": self.options.stage1_scope_workers_per_gpu,
            "stage1_preflight_workers": self.options.stage1_preflight_workers,
            "tfidf_workers": self.options.tfidf_workers,
            "updated_at": _utc_now(),
            "error": error,
        }
        target = self.options.work_root / "workflow_progress.json"
        _atomic_write_json(target, body)

    def _phase_manifest(self, phase: str) -> Path:
        return self.options.work_root / "phases" / phase / "complete_manifest.json"

    def _validated_complete(self, phase: str) -> Mapping[str, Any] | None:
        path = self._phase_manifest(phase)
        if not path.is_file():
            return None
        return _validate_phase_manifest_from_paths(
            work_root=self.options.work_root.resolve(strict=True),
            phase=phase,
            request_sha256=self.request["request_sha256"],
        )

    def _attempt_dir(self, phase: str) -> Path:
        phase_root = self.options.work_root / "phases" / phase
        phase_root.mkdir(parents=True, exist_ok=True)
        attempt = phase_root / f"attempt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        attempt.mkdir()
        return attempt

    def _complete(
        self,
        phase: str,
        result: Mapping[str, Any],
        *,
        attempt_dir: Path,
    ) -> Mapping[str, Any]:
        if not isinstance(result, Mapping):
            raise TypeError(f"phase {phase} did not return one result mapping")
        result_copy = dict(result)
        terminal_files = result_copy.get("terminal_files", [])
        if (
            not isinstance(terminal_files, list)
            or any(not isinstance(path, str) for path in terminal_files)
            or len(terminal_files) != len(set(terminal_files))
        ):
            raise ValueError(f"phase {phase} returned an invalid terminal_files list")
        artifacts = _attempt_tree_artifacts(attempt_dir)
        registered = {row["path"] for row in artifacts}
        for raw in terminal_files:
            terminal = Path(raw)
            if not terminal.is_absolute() or str(terminal.resolve(strict=True)) not in registered:
                raise ValueError(f"phase {phase} terminal file escaped its attempt: {terminal}")
        body = {
            "schema_version": WORKFLOW_PHASE_MANIFEST_SCHEMA,
            "phase": phase,
            "status": "complete",
            "request_sha256": self.request["request_sha256"],
            "attempt_dir": str(attempt_dir.resolve(strict=True)),
            "result": result_copy,
            "artifacts": artifacts,
        }
        manifest = {**body, "content_sha256": _sha(body)}
        target = self._phase_manifest(phase)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"completed phase manifest already exists: {phase}")
        _atomic_write_json(target, manifest)
        return manifest

    def _gpu_preflight(self) -> Mapping[str, Any]:
        requested = self.stage1_gpu_ids
        if not requested:
            return {
                "status": "accepted",
                "requested_gpu_ids": [],
                "exclusive_gpu_check_required": False,
                "checked_at": _utc_now(),
            }
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        resources: dict[int, dict[str, Any]] = {}
        for line in gpu.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if (
                len(fields) != 5
                or not fields[0].isdigit()
                or not fields[2].isdigit()
                or not fields[3].isdigit()
                or not fields[4].isdigit()
            ):
                continue
            resources[int(fields[0])] = {
                "uuid": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "utilization_percent": int(fields[4]),
            }
        mapping = {gpu_id: str(resource["uuid"]) for gpu_id, resource in resources.items()}
        missing = [gpu_id for gpu_id in requested if gpu_id not in mapping]
        if missing:
            raise RuntimeError(f"requested Stage 1 GPU IDs are unavailable: {missing}")
        active: dict[str, list[dict[str, Any]]] = {}
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 3:
                continue
            uuid, pid_text, used_memory = fields[:3]
            if not pid_text.isdigit() or int(pid_text) == os.getpid():
                continue
            active.setdefault(uuid, []).append(
                {"pid": int(pid_text), "used_memory_mib": used_memory}
            )
        occupied = {
            gpu_id: active[mapping[gpu_id]] for gpu_id in requested if active.get(mapping[gpu_id])
        }
        unexpected_idle_state: dict[int, Mapping[str, Any]] = {}
        for gpu_id in requested:
            resource = resources[gpu_id]
            total = int(resource["memory_total_mib"])
            # Driver/display contexts account for a small idle allocation on
            # these devices. Anything beyond two percent (at least 512 MiB) is
            # treated as an unreported occupant and fails closed.
            idle_memory_limit = max(512, int(math.ceil(total * 0.02)))
            used = int(resource["memory_used_mib"])
            utilization = int(resource["utilization_percent"])
            if used > idle_memory_limit or total - used < 6 * 1024 or utilization > 1:
                unexpected_idle_state[gpu_id] = {
                    **resource,
                    "idle_memory_limit_mib": idle_memory_limit,
                    "minimum_headroom_mib": 6 * 1024,
                }
        if occupied or unexpected_idle_state:
            raise RuntimeError(
                "Stage 1 GPUs are not exclusively available: "
                + _canonical(
                    {
                        "compute_processes": occupied,
                        "unexpected_idle_state": unexpected_idle_state,
                    }
                )
            )
        return {
            "status": "accepted",
            "requested_gpu_ids": list(requested),
            "gpu_uuids": {str(gpu_id): mapping[gpu_id] for gpu_id in requested},
            "gpu_resources": {str(gpu_id): resources[gpu_id] for gpu_id in requested},
            "minimum_headroom_mib": 6 * 1024,
            "exclusive_gpu_check_required": True,
            "checked_at": _utc_now(),
        }

    def _effective_stage1_profile(
        self,
        attempt: Path,
        *,
        dataset_path: Path,
        embedding_cache_dir: Path,
    ) -> Path:
        raw = json.loads(self.options.stage1_profile_path.read_text(encoding="utf-8"))
        config = raw.get("config", raw)
        config["dataset_path"] = str(dataset_path.resolve(strict=True))
        config["text_column"] = self.options.text_column
        config["treatment_column"] = self.options.treatment_column
        config["outcome_column"] = self.options.outcome_column
        config["outcome_type"] = self.options.outcome_type
        config["cv_folds"] = self.options.outer_folds
        config["architecture"]["htr_sentence_model"] = str(
            self.options.htr_local_model_path.resolve()
        )
        inner_partition_count = 3 + self.options.review_rounds
        for section_name in (
            "multi_model_forest",
            "multi_model_agentic_forest",
        ):
            section = config["architecture"].get(section_name)
            if not isinstance(section, dict):
                raise ValueError(f"Stage 1 profile lacks architecture.{section_name}")
            section["candidate_consistency_inner_folds"] = inner_partition_count
            section["tfidf_nested_calibration_folds"] = self.options.tfidf_nested_calibration_folds
        explicit_forest = config["architecture"].get("explicit_feature_forest")
        if not isinstance(explicit_forest, dict):
            raise ValueError("Stage 1 profile lacks architecture.explicit_feature_forest")
        explicit_forest["interaction_inner_folds"] = self.options.interaction_inner_folds

        def bind_embedding_sections(value: Any) -> None:
            if not isinstance(value, dict):
                return
            embedding = value.get("embedding_contrast")
            if isinstance(embedding, dict):
                embedding.update(
                    {
                        "model_name": self.options.embedding_model_name,
                        "cache_dir": str(embedding_cache_dir.resolve(strict=True)),
                        "device": self.options.stage1_device,
                        "chunk_size_words": 256,
                        "chunk_overlap_words": 64,
                        "max_chunks": 128,
                        "max_seq_length": 1024,
                        "batch_size": 1,
                        "normalize_embeddings": True,
                        "cluster_contrast_n_clusters": 10,
                        "cluster_contrast_kmeans_n_init": 20,
                        "cluster_contrast_min_cluster_size": 24,
                        "cluster_contrast_min_group_size": 8,
                        "cluster_contrast_min_cell_size": 4,
                        "cluster_contrast_max_components": 5,
                    }
                )
            for child in value.values():
                bind_embedding_sections(child)

        bind_embedding_sections(config["architecture"])
        forest = config["architecture"]["causal_forest"]
        forest.update(
            {
                "n_estimators": 200,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "honest": True,
                "inference": True,
            }
        )
        path = attempt / "effective_stage1_profile.json"
        path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @staticmethod
    def _embedding_chunk_configuration() -> Mapping[str, Any]:
        return {
            "chunk_size_words": 256,
            "chunk_overlap_words": 64,
            "max_chunks": 128,
            "chunk_selection": "last",
            "normalize_embeddings": True,
            "max_seq_length": 1024,
        }

    def _input_preparation_paths(self) -> tuple[Path, Path]:
        preparation = self._validated_complete("input_preparation")
        if preparation is None:
            raise RuntimeError("input preparation is not complete")
        output = Path(preparation["result"]["output"]["path"]).resolve(strict=True)
        manifest = next(
            Path(row["path"]).resolve(strict=True)
            for row in preparation["artifacts"]
            if Path(row["path"]).name == "preparation_manifest.json"
        )
        return output, manifest

    def _embedding_cache_paths(self) -> tuple[Path, Path]:
        phase = self._validated_complete("embedding_cache")
        if phase is None:
            raise RuntimeError("embedding-cache phase is not complete")
        result = phase["result"]
        if result.get("schema_version") != EMBEDDING_CACHE_PHASE_SCHEMA:
            raise RuntimeError("embedding-cache phase has an unsupported result schema")
        cache = Path(result["cache_path"]).resolve(strict=True)
        prepared = Path(result["prepared_cohort_path"]).resolve(strict=True)
        phase_root = (self.options.work_root / "phases" / "embedding_cache").resolve()
        if phase_root not in cache.parents or phase_root not in prepared.parents:
            raise RuntimeError("embedding-cache outputs escaped their fresh attempt directory")
        registered = {Path(row["path"]).resolve(strict=True) for row in phase["artifacts"]}
        actual_cache_files = {
            path.resolve(strict=True) for path in cache.rglob("*") if path.is_file()
        }
        if not actual_cache_files or not actual_cache_files.issubset(registered):
            raise RuntimeError("embedding-cache files are not fully terminally registered")
        if prepared not in registered:
            raise RuntimeError("cache-bound prepared cohort is not terminally registered")
        return cache, prepared

    def _embedding_cache_relocation_options(
        self,
        *,
        cache: Path,
        prepared: Path,
    ) -> Any | None:
        """Reconstruct the exact relocation proof input consumed by Stage 1."""

        if self.options.embedding_cache_import is None:
            return None
        from .production_embedding_cache_relocation import (
            ProductionEmbeddingCacheRelocationOptions,
        )

        phase = self._validated_complete("embedding_cache")
        if phase is None:
            raise RuntimeError("embedding-cache relocation phase is not complete")
        result = phase["result"]
        identity = result.get("cache_identity")
        if result.get("mode") != "authenticated_relocation" or not isinstance(identity, Mapping):
            raise RuntimeError("cache import did not produce an authenticated relocation")
        target = Path(str(identity.get("root", ""))).resolve(strict=True)
        if cache.parent != target or prepared.parent.parent != target:
            raise RuntimeError("relocated cache result paths differ from its sealed root")
        fresh_prepared, fresh_manifest = self._input_preparation_paths()
        source_prepared, source_manifest = self._resolved_cache_import_sources()
        return ProductionEmbeddingCacheRelocationOptions(
            source_cache_dir=self.options.embedding_cache_import,
            source_prepared_cohort_path=source_prepared,
            source_preparation_manifest_path=source_manifest,
            fresh_prepared_cohort_path=fresh_prepared,
            fresh_preparation_manifest_path=fresh_manifest,
            local_model_path=self.options.embedding_local_model_path,
            target_dir=target,
            unit_id_column=self.options.unit_id_column,
            text_column=self.options.text_column,
            treatment_column=self.options.treatment_column,
            outcome_column=self.options.outcome_column,
            sentence_model_name=self.options.embedding_model_name,
            chunk_configuration=self._embedding_chunk_configuration(),
        )

    def _stage1_preflight_paths(self) -> tuple[Path, Path]:
        phase = self._validated_complete("stage1_preflight")
        if phase is None:
            raise RuntimeError("Stage 1 scientific preflight is not complete")
        result = phase["result"]
        if result.get("schema_version") != STAGE1_PREFLIGHT_PHASE_SCHEMA:
            raise RuntimeError("Stage 1 preflight phase has an unsupported schema")
        profile = Path(str(result.get("effective_profile_path", ""))).resolve(strict=True)
        manifest = Path(str(result.get("cluster_preflight_manifest_path", ""))).resolve(strict=True)
        registered = {Path(row["path"]).resolve(strict=True) for row in phase["artifacts"]}
        if (
            profile not in registered
            or manifest not in registered
            or profile.name != "effective_stage1_profile.json"
            or manifest.name != "cluster_preflight_manifest.json"
        ):
            raise RuntimeError("Stage 1 preflight profile/artifact is not terminally sealed")
        return profile, manifest

    def _stage1_build_options(
        self,
        *,
        dataset: Path,
        profile: Path,
        cache: Path,
        output: Path,
        dry_run: bool,
        cluster_preflight_manifest_path: Path | None = None,
    ) -> Stage1BundleBuildOptions:
        values: dict[str, Any] = {
            "dataset_path": dataset,
            "config_path": profile,
            "embedding_cache_dir": cache,
            "embedding_local_model_path": None,
            "embedding_cache_output_dir": None,
            "output_dir": output,
            "unit_id_column": self.options.unit_id_column,
            "seed": self.options.seed,
            "device": self.options.stage1_device,
            "gpu_ids": self.stage1_gpu_ids,
            "num_workers": self.options.num_workers,
            "tfidf_workers": self.options.tfidf_workers,
            "tfidf_parallel_backend": self.options.tfidf_parallel_backend,
            "query_devices": self.query_devices,
            "query_nuisance_folds": self.options.interaction_inner_folds,
            "query_config_path": self.options.query_profile_path,
            "resume": False,
            "dry_run": dry_run,
        }
        if "embedding_cache_relocation" in Stage1BundleBuildOptions.__dataclass_fields__:
            values["embedding_cache_relocation"] = self._embedding_cache_relocation_options(
                cache=cache,
                prepared=dataset,
            )
        # Parallel scheduler fields are passed automatically as soon as the
        # Stage1BundleBuildOptions API exposes them.  This keeps the workflow
        # interface independently testable while that implementation lands.
        available = Stage1BundleBuildOptions.__dataclass_fields__
        if len(self.stage1_gpu_ids) > 1 and "scope_workers_per_gpu" not in available:
            raise RuntimeError(
                "multiple Stage 1 GPUs require the canonical scope scheduler; "
                "this builder does not expose it"
            )
        if (
            cluster_preflight_manifest_path is not None
            and "cluster_preflight_manifest_path" not in available
        ):
            raise RuntimeError(
                "supervised Stage 1 requires the independently sealed scientific "
                "preflight consumer API"
            )
        optional_bindings = {
            "scope_workers_per_gpu": self.options.stage1_scope_workers_per_gpu,
            "preflight_workers": self.options.stage1_preflight_workers,
            "scope_seed_policy": self.options.stage1_seed_policy,
            "cluster_preflight_manifest_path": cluster_preflight_manifest_path,
            "stage1_scope_attempt_root": (
                self.options.work_root / "recovery" / "stage1_scope_attempts"
            ).resolve(),
            "stage1_scope_progress_path": (
                self.options.work_root / "recovery" / "stage1_scope_progress.json"
            ).resolve(),
        }
        values.update({key: value for key, value in optional_bindings.items() if key in available})
        return Stage1BundleBuildOptions(**values)

    def _release_embedding_cuda_memory(self, *, cache_was_built: bool) -> None:
        """Drop cache-builder references before any Stage 1 model is fitted."""

        gc.collect()
        if not cache_was_built or not str(self.options.stage1_device).startswith("cuda:"):
            return
        try:
            import torch
        except ImportError:
            return
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return
        embedding_gpu_id = int(str(self.options.stage1_device).split(":", 1)[1])
        with torch.cuda.device(embedding_gpu_id):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def _run_embedding_cache_phase(self, attempt: Path) -> Mapping[str, Any]:
        o = self.options
        resource = self._gpu_preflight()
        fresh_prepared, fresh_preparation_manifest = self._input_preparation_paths()
        if o.embedding_cache_import is not None:
            from .production_embedding_cache_relocation import (
                ProductionEmbeddingCacheRelocationOptions,
                relocate_authenticated_production_embedding_cache,
                validate_relocated_production_embedding_cache,
            )

            source_prepared, source_preparation_manifest = self._resolved_cache_import_sources()
            relocation_options = ProductionEmbeddingCacheRelocationOptions(
                source_cache_dir=o.embedding_cache_import,
                source_prepared_cohort_path=source_prepared,
                source_preparation_manifest_path=source_preparation_manifest,
                fresh_prepared_cohort_path=fresh_prepared,
                fresh_preparation_manifest_path=fresh_preparation_manifest,
                local_model_path=o.embedding_local_model_path,
                target_dir=(attempt / "relocated_cache").resolve(),
                unit_id_column=o.unit_id_column,
                text_column=o.text_column,
                treatment_column=o.treatment_column,
                outcome_column=o.outcome_column,
                sentence_model_name=o.embedding_model_name,
                chunk_configuration=self._embedding_chunk_configuration(),
            )
            built = relocate_authenticated_production_embedding_cache(relocation_options)
            validated = validate_relocated_production_embedding_cache(relocation_options)
            if built.identity() != validated.identity():
                raise RuntimeError("relocated embedding cache changed during fresh validation")
            identity = validated.identity()
            cache_path = validated.cache_dir
            prepared_path = validated.prepared_cohort_path
            terminal_files = [
                str(path)
                for path in (
                    *sorted(validated.cache_dir.iterdir()),
                    validated.prepared_cohort_path,
                    validated.attestation_path,
                    validated.terminal_manifest_path,
                )
                if path.is_file()
            ]
            mode = "authenticated_relocation"
        else:
            from .production_embedding_cache_builder import (
                build_production_embedding_cache,
                validate_published_production_embedding_cache,
            )

            cache_path = (attempt / "embedding_cache").resolve()
            prepared_copy = attempt / "prepared"
            prepared_copy.mkdir()
            prepared_path = prepared_copy / "modeling_cohort.parquet"
            # Build and validate against the exact artifact Stage 1 will read;
            # the production cache provenance intentionally binds its path.
            import shutil

            shutil.copyfile(fresh_prepared, prepared_path)
            built = build_production_embedding_cache(
                dataset_path=prepared_path,
                text_column=o.text_column,
                local_model_path=o.embedding_local_model_path,
                sentence_model_name=o.embedding_model_name,
                chunk_configuration=self._embedding_chunk_configuration(),
                target_dir=cache_path,
                device=o.stage1_device,
                batch_size=1,
            )
            identity = validate_published_production_embedding_cache(
                cache_dir=built.cache_path,
                dataset_path=prepared_path,
                text_column=o.text_column,
                sentence_model_name=o.embedding_model_name,
                chunk_configuration=self._embedding_chunk_configuration(),
                expected_local_model_path=o.embedding_local_model_path,
            )
            if built.identity() != identity:
                raise RuntimeError("fresh embedding cache changed during read-only validation")
            terminal_files = [
                *(str(path) for path in sorted(cache_path.iterdir()) if path.is_file()),
                str(prepared_path),
            ]
            mode = "fresh_build"
        self._release_embedding_cuda_memory(
            cache_was_built=(mode == "fresh_build"),
        )
        return {
            "schema_version": EMBEDDING_CACHE_PHASE_SCHEMA,
            "mode": mode,
            "cache_path": str(Path(cache_path).resolve(strict=True)),
            "prepared_cohort_path": str(Path(prepared_path).resolve(strict=True)),
            "cache_identity": identity,
            "resource_preflight": resource,
            "embedding_model_materialized_in_workflow_process": mode == "fresh_build",
            "cuda_memory_release_requested": mode == "fresh_build",
            "terminal_files": terminal_files,
        }

    def _phase_hook_context(self, phase: str, attempt: Path) -> Mapping[str, Any]:
        prepared, preparation_manifest = self._input_preparation_paths()
        import_sources = self._resolved_cache_import_sources()
        cache: Path | None = None
        cache_prepared: Path | None = None
        cache_phase_identity: Mapping[str, Any] | None = None
        preflight_profile: Path | None = None
        cluster_preflight_manifest: Path | None = None
        if phase in {"stage1_preflight", "stage1_modeling"}:
            cache, cache_prepared = self._embedding_cache_paths()
            cache_phase = self._validated_complete("embedding_cache")
            assert cache_phase is not None
            raw_identity = cache_phase["result"].get("cache_identity")
            if not isinstance(raw_identity, Mapping):
                raise RuntimeError("embedding-cache phase did not expose a cache identity")
            cache_phase_identity = dict(raw_identity)
        if phase == "stage1_modeling":
            preflight_profile, cluster_preflight_manifest = self._stage1_preflight_paths()
        return {
            "schema_version": "production_workflow_phase_hook_context_v1",
            "phase": phase,
            "attempt_dir": str(attempt.resolve()),
            "request_sha256": self.request["request_sha256"],
            "prepared_cohort_path": str(prepared),
            "preparation_manifest_path": str(preparation_manifest),
            "embedding_cache_target_dir": str((attempt / "embedding_cache").resolve()),
            "embedding_cache_import": (
                None
                if self.options.embedding_cache_import is None
                else str(self.options.embedding_cache_import.resolve(strict=True))
            ),
            "embedding_cache_import_source_prepared_path": (
                None if import_sources is None else str(import_sources[0])
            ),
            "embedding_cache_import_source_preparation_manifest_path": (
                None if import_sources is None else str(import_sources[1])
            ),
            "embedding_chunk_configuration": dict(self._embedding_chunk_configuration()),
            "embedding_cache_path": None if cache is None else str(cache),
            "cache_bound_prepared_cohort_path": (
                None if cache_prepared is None else str(cache_prepared)
            ),
            "embedding_cache_phase_identity": cache_phase_identity,
            "effective_stage1_profile_path": (
                None if preflight_profile is None else str(preflight_profile)
            ),
            "cluster_preflight_manifest_path": (
                None if cluster_preflight_manifest is None else str(cluster_preflight_manifest)
            ),
            "stage1_profile_path": str(self.options.stage1_profile_path.resolve(strict=True)),
            "query_profile_path": str(self.options.query_profile_path.resolve(strict=True)),
            "stage1_gpu_ids": list(self.stage1_gpu_ids),
            "query_devices": list(self.query_devices),
            "stage1_scope_workers_per_gpu": self.options.stage1_scope_workers_per_gpu,
            "stage1_preflight_workers": self.options.stage1_preflight_workers,
            "stage1_seed_policy": self.options.stage1_seed_policy,
            "stage1_scope_attempt_root": str(
                (self.options.work_root / "recovery" / "stage1_scope_attempts").resolve()
            ),
            "stage1_scope_progress_path": str(
                (self.options.work_root / "recovery" / "stage1_scope_progress.json").resolve()
            ),
            "tfidf_workers": self.options.tfidf_workers,
            "tfidf_parallel_backend": self.options.tfidf_parallel_backend,
            "seed": self.options.seed,
            "resource_preflight": self._gpu_preflight(),
        }

    def _validate_handoff_in_fresh_process(
        self,
        *,
        bundle_manifest: Path,
        report_path: Path,
    ) -> Mapping[str, Any]:
        script = r"""
import hashlib
import json
import sys
from pathlib import Path
import oci.inference.production_stage1_hierarchy_handoff as handoff_module
from oci.inference.production_stage1_hierarchy_handoff import load_production_stage1_hierarchy_handoff

manifest = Path(sys.argv[1]).resolve(strict=True)
report = Path(sys.argv[2])
review_rounds = int(sys.argv[3])
interaction_folds = int(sys.argv[4])
tfidf_folds = int(sys.argv[5])
handoff = load_production_stage1_hierarchy_handoff(
    manifest,
    review_rounds=review_rounds,
    interaction_inner_folds=interaction_folds,
    tfidf_nested_calibration_folds=tfidf_folds,
)
body = {
    "schema_version": "production_stage1_fresh_handoff_validation_v1",
    "status": "accepted",
    "bundle_manifest_path": str(manifest),
    "review_rounds": review_rounds,
    "interaction_inner_folds": interaction_folds,
    "tfidf_nested_calibration_folds": tfidf_folds,
    "handoff": handoff.as_dict(),
    "remote_clients_constructed": False,
    "remote_calls_made": False,
    "loader_module_path": str(Path(handoff_module.__file__).resolve(strict=True)),
}
canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
payload = {**body, "content_sha256": hashlib.sha256(canonical.encode()).hexdigest()}
report.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
"""
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        interpreter = [sys.executable]
        if self.request.get("source_snapshot") is not None:
            interpreter.append("-P")
        subprocess.run(
            [
                *interpreter,
                "-c",
                script,
                str(bundle_manifest.resolve(strict=True)),
                str(report_path.resolve()),
                str(self.options.review_rounds),
                str(self.options.interaction_inner_folds),
                str(self.options.tfidf_nested_calibration_folds),
            ],
            check=True,
            env=environment,
        )
        value = json.loads(report_path.read_text(encoding="utf-8"))
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("schema_version") != "production_stage1_fresh_handoff_validation_v1"
            or value.get("status") != "accepted"
            or value.get("bundle_manifest_path") != str(bundle_manifest.resolve(strict=True))
            or value.get("content_sha256") != _sha(body)
            or value.get("remote_clients_constructed") is not False
            or value.get("remote_calls_made") is not False
        ):
            raise RuntimeError("fresh Stage 1 handoff validation report is invalid")
        source_snapshot = self.request.get("source_snapshot")
        if source_snapshot is not None:
            loaded = Path(str(value.get("loader_module_path", ""))).resolve(strict=True)
            snapshot_root = Path(str(source_snapshot["root"])).resolve(strict=True)
            try:
                loaded.relative_to(snapshot_root)
            except ValueError as exc:
                raise RuntimeError(
                    "fresh handoff loader did not execute from source snapshot"
                ) from exc
        return value

    def _validate_terminal_in_fresh_process(
        self,
        *,
        report_path: Path,
    ) -> Mapping[str, Any]:
        """Reopen the immutable request and every prior phase from paths only."""

        script = r"""
import hashlib
import json
import sys
from pathlib import Path
import oci.inference.production_all_evidence_workflow as workflow_module
from oci.inference.production_all_evidence_workflow import validate_completed_workflow_prefix

root = Path(sys.argv[1]).resolve(strict=True)
request_sha256 = sys.argv[2]
phases = json.loads(sys.argv[3])
stage1_only = json.loads(sys.argv[4])
report = Path(sys.argv[5])
validation = validate_completed_workflow_prefix(
    work_root=root,
    expected_request_sha256=request_sha256,
    expected_phases=phases,
)
body = {
    "schema_version": "production_all_evidence_fresh_terminal_validation_report_v1",
    "execution_completed": True,
    "run_validation_status": "accepted",
    "global_release_certified": False,
    "stage1_only": stage1_only,
    "validated_phase_sequence": [*phases, "terminal_validation"],
    "stage1_handoff_validated_in_fresh_process": validation[
        "stage1_handoff_validated_in_fresh_process"
    ],
    "read_only_prefix_validation": validation,
    "live_runner_objects_received": False,
    "validator_module_path": str(Path(workflow_module.__file__).resolve(strict=True)),
}
canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
payload = {**body, "content_sha256": hashlib.sha256(canonical.encode()).hexdigest()}
report.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
"""
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        prior_phases = list(self._phase_sequence()[:-1])
        interpreter = [sys.executable]
        if self.request.get("source_snapshot") is not None:
            interpreter.append("-P")
        subprocess.run(
            [
                *interpreter,
                "-c",
                script,
                str(self.options.work_root.resolve(strict=True)),
                self.request["request_sha256"],
                json.dumps(prior_phases),
                json.dumps(self.options.stage1_only),
                str(report_path.resolve()),
            ],
            check=True,
            env=environment,
        )
        value = _read_json_object(report_path, label="fresh terminal validation report")
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("schema_version")
            != "production_all_evidence_fresh_terminal_validation_report_v1"
            or value.get("run_validation_status") != "accepted"
            or value.get("validated_phase_sequence") != list(self._phase_sequence())
            or value.get("content_sha256") != _sha(body)
            or value.get("live_runner_objects_received") is not False
        ):
            raise RuntimeError("fresh terminal validation report is invalid")
        source_snapshot = self.request.get("source_snapshot")
        if source_snapshot is not None:
            loaded = Path(str(value.get("validator_module_path", ""))).resolve(strict=True)
            snapshot_root = Path(str(source_snapshot["root"])).resolve(strict=True)
            try:
                loaded.relative_to(snapshot_root)
            except ValueError as exc:
                raise RuntimeError(
                    "fresh terminal validator did not execute from source snapshot"
                ) from exc
        if (
            "handoff_validation" not in self.phase_overrides
            and value.get("stage1_handoff_validated_in_fresh_process") is not True
        ):
            raise RuntimeError("default Stage 1 handoff was not freshly validated")
        return value

    def _run_default(self, phase: str, attempt: Path) -> Mapping[str, Any]:
        o = self.options
        if phase == "input_preparation":
            prepared = attempt / "prepared"
            result = prepare_modeling_cohort(
                TextPreparationOptions(
                    o.dataset_path,
                    prepared,
                    o.unit_id_column,
                    o.text_column,
                    o.treatment_column,
                    o.outcome_column,
                    o.outcome_type,
                    o.repeated_character_threshold,
                    o.empty_text_policy,
                    o.repeated_character_policy,
                )
            )
            return {
                **result,
                "terminal_files": [
                    result["output"]["path"],
                    str(prepared / "preparation_manifest.json"),
                ],
            }
        if phase == "embedding_cache":
            return self._run_embedding_cache_phase(attempt)
        if phase == "stage1_preflight":
            resource = self._gpu_preflight()
            cache, prepared = self._embedding_cache_paths()
            profile = self._effective_stage1_profile(
                attempt,
                dataset_path=prepared,
                embedding_cache_dir=cache,
            )
            preflight_builder = ProductionStage1BundleBuilder(
                self._stage1_build_options(
                    dataset=prepared,
                    profile=profile,
                    cache=cache,
                    output=attempt / "preflight_no_model_output",
                    dry_run=True,
                )
            )
            prepared_build = preflight_builder.prepare()
            scope_input_identity = prepared_build.cluster_preflight_scope_input_set_identity
            if not isinstance(scope_input_identity, Mapping):
                raise RuntimeError(
                    "Stage 1 preflight omitted its recoverable private " "scope-input identity"
                )
            from .production_stage1_cluster_preflight_artifact import (
                seal_production_stage1_cluster_preflight_artifact,
            )

            artifact = seal_production_stage1_cluster_preflight_artifact(
                output_dir=(attempt / "cluster_preflight").resolve(),
                audit=prepared_build.embedding_cluster_feasibility_audit,
                stage1_request=prepared_build.request,
                config=prepared_build.config,
                registry=prepared_build.registry,
                registry_content_sha256=prepared_build.registry_content_sha256,
                embedding_cache_identity=prepared_build.embedding_cache_identity,
            )
            report = attempt / "stage1_preflight_report.json"
            inner_partition_count = 3 + o.review_rounds
            planned_scope_count = o.outer_folds * (1 + inner_partition_count + o.review_rounds)
            artifact_identity = artifact.identity()
            if artifact_identity["scope_count"] != planned_scope_count:
                raise RuntimeError("scientific Stage 1 preflight did not cover every planned scope")
            payload = {
                "schema_version": STAGE1_PREFLIGHT_PHASE_SCHEMA,
                "resource_preflight": resource,
                "cache_phase_reopened_and_rehashed": True,
                "effective_profile_path": str(profile.resolve(strict=True)),
                "cluster_preflight_manifest_path": str(artifact.manifest_path),
                "cluster_preflight_identity": artifact_identity,
                "cluster_preflight_scope_inputs_identity": copy.deepcopy(
                    dict(scope_input_identity)
                ),
                "planned_scope_counts": {
                    "full_outer": o.outer_folds,
                    "exact_inner": o.outer_folds * inner_partition_count,
                    "cumulative_review": o.outer_folds * o.review_rounds,
                    "total": planned_scope_count,
                },
                "scientific_cluster_preflight": "accepted_and_independently_sealed_v1",
                "scientific_preflight_recomputed_during_supervised_modeling": False,
                "supervised_fit_may_begin_before_scientific_preflight_acceptance": False,
                "stage1_gpu_ids": list(self.stage1_gpu_ids),
                "scope_workers_per_gpu": o.stage1_scope_workers_per_gpu,
                "preflight_workers": o.stage1_preflight_workers,
                "seed_policy": o.stage1_seed_policy,
            }
            report.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            return {
                **payload,
                "terminal_files": [
                    str(profile),
                    str(artifact.audit_path),
                    str(artifact.stage1_request_path),
                    str(artifact.manifest_path),
                    str(report),
                ],
            }
        if phase == "stage1_modeling":
            resource = self._gpu_preflight()
            cache, prepared = self._embedding_cache_paths()
            profile, cluster_preflight_manifest = self._stage1_preflight_paths()
            bundle = attempt / "stage1_bundle"
            result = ProductionStage1BundleBuilder(
                self._stage1_build_options(
                    dataset=prepared,
                    profile=profile,
                    cache=cache,
                    output=bundle,
                    dry_run=False,
                    cluster_preflight_manifest_path=cluster_preflight_manifest,
                )
            ).build()
            manifest = bundle / "bundle_manifest.json"
            return {
                **result,
                "resource_preflight": resource,
                "stage1_gpu_ids": list(self.stage1_gpu_ids),
                "effective_profile_reused_from_preflight": str(profile),
                "cluster_preflight_manifest_consumed": str(cluster_preflight_manifest),
                "scientific_cluster_preflight_recomputed": False,
                "terminal_files": [str(manifest)],
            }
        if phase == "handoff_validation":
            stage1 = self._validated_complete("stage1_modeling")
            manifest = next(
                Path(row["path"])
                for row in stage1["artifacts"]
                if Path(row["path"]).name == "bundle_manifest.json"
            )
            report = attempt / "fresh_handoff_validation.json"
            validation = self._validate_handoff_in_fresh_process(
                bundle_manifest=manifest,
                report_path=report,
            )
            return {
                "fresh_process_validation": validation,
                "source_snapshot": self.request.get("source_snapshot"),
                "terminal_files": [str(report)],
            }
        if phase == "stage2_canary":
            from scripts.canary_production_stage1_hierarchy import run_canary

            options = self._stage2_options(attempt, prefix="canary")
            result = run_canary(options)
            return {**result, "terminal_files": [result["report_path"]]}
        if phase == "stage2_inference":
            from .production_stage1_hierarchy_one_shot import (
                run_production_stage1_hierarchy_one_shot,
            )

            options = self._stage2_options(attempt, prefix="full")
            result = run_production_stage1_hierarchy_one_shot(options)
            prediction = options.output_dir / "frozen_predictions.parquet"
            manifest = options.output_dir / "immutable_run_manifest.json"
            return {**result, "terminal_files": [str(prediction), str(manifest)]}
        if phase == "oracle_evaluation":
            if not o.evaluate_oracle_posthoc:
                return {"skipped_by_configuration": True, "terminal_files": []}
            from .production_oracle_evaluation import evaluate_frozen_predictions_posthoc

            inference = self._validated_complete("stage2_inference")
            files = [Path(row["path"]) for row in inference["artifacts"]]
            prediction = next(path for path in files if path.name == "frozen_predictions.parquet")
            manifest = next(path for path in files if path.name == "immutable_run_manifest.json")
            stage1 = self._validated_complete("stage1_modeling")
            bundle_manifest = next(
                Path(row["path"])
                for row in stage1["artifacts"]
                if Path(row["path"]).name == "bundle_manifest.json"
            )
            row_map = bundle_manifest.parent / "row_registry.parquet"
            result = evaluate_frozen_predictions_posthoc(
                predictions_path=prediction,
                prediction_manifest_path=manifest,
                unit_id_map_path=row_map,
                oracle_dataset_path=o.oracle_dataset_path,
                output_dir=attempt / "evaluation",
                unit_id_column=o.unit_id_column,
                oracle_unit_id_column=o.oracle_unit_id_column,
                oracle_ite_column=o.oracle_ite_column,
            )
            return {
                **result,
                "terminal_files": [
                    result["joined_path"],
                    str(attempt / "evaluation/evaluation_metrics.json"),
                ],
            }
        if phase == "terminal_validation":
            report = attempt / "validation.json"
            validation = self._validate_terminal_in_fresh_process(report_path=report)
            return {**validation, "terminal_files": [str(report)]}
        raise AssertionError(phase)

    def _stage2_options(self, attempt: Path, *, prefix: str) -> Any:
        from .production_stage1_hierarchy_one_shot import (
            ProductionStage1HierarchyOneShotOptions,
        )

        o = self.options
        if not isinstance(o.endpoint, str) or not isinstance(o.model_name, str):
            raise RuntimeError("Stage 2 options were requested without endpoint/model identity")
        stage1 = self._validated_complete("stage1_modeling")
        bundle_manifest = next(
            Path(row["path"])
            for row in stage1["artifacts"]
            if Path(row["path"]).name == "bundle_manifest.json"
        )
        return ProductionStage1HierarchyOneShotOptions(
            bundle_manifest_path=bundle_manifest,
            output_dir=attempt / f"{prefix}_output",
            preparation_dir=attempt / f"{prefix}_preparation",
            attestation_dir=attempt / f"{prefix}_attestation",
            endpoint=o.endpoint,
            model_name=o.model_name,
            review_rounds=o.review_rounds,
            interaction_inner_folds=o.interaction_inner_folds,
            tfidf_nested_calibration_folds=o.tfidf_nested_calibration_folds,
            review_stage1_device=o.review_device,
            review_neural_query_devices=(o.review_device,),
            max_candidates=20,
            seed=o.seed,
            proposal_schema_repair_attempts=1,
            request_max_retries=0,
            extraction_batch_size=128,
            extraction_context_strategy="complete_paged_v1",
            extraction_max_text_length=14_000,
        )

    def _execute_phase_sequence(
        self,
        sequence: Sequence[str],
    ) -> dict[str, Mapping[str, Any]]:
        """Execute or authenticate an ordered workflow prefix."""

        completed: dict[str, Any] = {}
        hook_by_phase: Mapping[str, WorkflowPhaseHook | None] = {
            "embedding_cache": self.hooks.embedding_cache,
            "stage1_preflight": self.hooks.stage1_preflight,
            "stage1_modeling": self.hooks.stage1_modeling,
        }
        for phase in sequence:
            if phase not in self._phase_sequence():
                raise ValueError(f"phase is outside this workflow request: {phase}")
            _revalidate_request_bound_external_inputs(self.request)
            existing = self._validated_complete(phase) if self.options.resume else None
            if existing is not None:
                completed[phase] = existing
                self._write_progress(
                    status="running",
                    completed=tuple(completed),
                    current_phase=None,
                )
                continue
            attempt = self._attempt_dir(phase)
            self._write_progress(
                status="running",
                completed=tuple(completed),
                current_phase=phase,
            )
            try:
                if phase in self.phase_overrides:
                    result = self.phase_overrides[phase](attempt)
                elif hook_by_phase.get(phase) is not None:
                    hook = hook_by_phase[phase]
                    assert hook is not None
                    result = hook(attempt, self._phase_hook_context(phase, attempt))
                else:
                    result = self._run_default(phase, attempt)
                _revalidate_request_bound_external_inputs(self.request)
                completed[phase] = self._complete(
                    phase,
                    result,
                    attempt_dir=attempt,
                )
            except BaseException as exc:
                self._write_progress(
                    status="failed",
                    completed=tuple(completed),
                    current_phase=phase,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            self._write_progress(
                status="running",
                completed=tuple(completed),
                current_phase=None,
            )
        return completed

    @staticmethod
    def _registered_file_identity(path: Path) -> Mapping[str, Any]:
        resolved = path.resolve(strict=True)
        digest, size = stable_file_sha256(resolved)
        return {
            "path": str(resolved),
            "sha256": digest,
            "size_bytes": size,
        }

    def _validate_canary_preparation_in_fresh_process(
        self,
    ) -> Mapping[str, Any]:
        script = r"""
import json
import os
from pathlib import Path
import sys
import oci.inference.production_all_evidence_workflow as workflow_module
from oci.inference.production_all_evidence_workflow import validate_stage1_canary_descriptor_preparation

result = validate_stage1_canary_descriptor_preparation(Path(sys.argv[1]))
print(json.dumps({
    "result": result,
    "validator_module_path": str(Path(workflow_module.__file__).resolve(strict=True)),
    "source_snapshot_marker": os.environ.get(
        workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV
    ),
    "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    "python_path": os.environ.get("PYTHONPATH"),
    "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
}, sort_keys=True, allow_nan=False))
"""
        source_snapshot = self.request.get("source_snapshot")
        if not isinstance(source_snapshot, Mapping):
            raise RuntimeError("canary preparation fresh validation requires a source snapshot")
        snapshot_root = Path(str(source_snapshot.get("root", ""))).resolve(strict=True)
        snapshot_sha = str(source_snapshot.get("content_sha256") or "")
        expected_hash_seed = str(int(self.options.seed))
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONPATH"] = str(snapshot_root)
        environment[SOURCE_SNAPSHOT_EXECUTION_ENV] = snapshot_sha
        environment["PYTHONHASHSEED"] = expected_hash_seed
        completed = subprocess.run(
            [
                sys.executable,
                "-P",
                "-c",
                script,
                str(self.options.work_root.resolve(strict=True)),
            ],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
        payload = json.loads(completed.stdout)
        if (
            not isinstance(payload, Mapping)
            or not isinstance(payload.get("result"), Mapping)
            or not isinstance(payload.get("validator_module_path"), str)
            or payload.get("source_snapshot_marker") != snapshot_sha
            or payload.get("python_hash_seed") != expected_hash_seed
            or payload.get("python_path") != str(snapshot_root)
            or payload.get("python_no_user_site") != "1"
        ):
            raise RuntimeError("fresh canary preparation validator returned invalid output")
        loaded = Path(payload["validator_module_path"]).resolve(strict=True)
        try:
            loaded.relative_to(snapshot_root)
        except ValueError as exc:
            raise RuntimeError(
                "canary preparation validator did not execute from source snapshot"
            ) from exc
        return copy.deepcopy(dict(payload["result"]))

    def prepare_stage1_canary_descriptors_only(self) -> Mapping[str, Any]:
        """Seal the exact final-run prefix and descriptors without fitting."""

        if not self.options.stage1_only or len(self.stage1_gpu_ids) != 2:
            raise ValueError(
                "canary descriptor preparation requires Stage-1-only mode and "
                "exactly two Stage 1 GPUs"
            )
        if self.options.source_snapshot_root is None:
            raise ValueError(
                "canary descriptor preparation requires one authenticated " "source snapshot"
            )
        self._initialize()
        _revalidate_request_bound_external_inputs(self.request)
        prefix = ("input_preparation", "embedding_cache", "stage1_preflight")
        completed = self._execute_phase_sequence(prefix)
        self._write_progress(
            status="preparing_canary_descriptors",
            completed=tuple(completed),
            current_phase="canary_descriptor_preparation",
        )
        try:
            cache, prepared_path = self._embedding_cache_paths()
            profile, cluster_preflight_manifest = self._stage1_preflight_paths()
            prepared = ProductionStage1BundleBuilder(
                self._stage1_build_options(
                    dataset=prepared_path,
                    profile=profile,
                    cache=cache,
                    output=(
                        self.options.work_root / "recovery" / "canary_descriptor_no_model_output"
                    ).resolve(),
                    dry_run=False,
                    cluster_preflight_manifest_path=cluster_preflight_manifest,
                )
            ).prepare()
            from .production_stage1_legacy_scope_adapter import (
                LEGACY_STAGE1_SCOPE_DESCRIPTOR_SET_MANIFEST,
                publish_legacy_stage1_scope_descriptor,
                validate_legacy_stage1_scope_descriptor_set,
            )

            descriptor_set = publish_legacy_stage1_scope_descriptor(
                prepared=prepared,
                descriptor_root=prepared.scope_descriptor_root,
            )
            descriptor_set = validate_legacy_stage1_scope_descriptor_set(
                descriptor_root=descriptor_set.root,
                expected_stage1_request_sha256=prepared.request_sha256,
                prepared=prepared,
            )
            selected = next(
                (
                    descriptor
                    for descriptor in descriptor_set.descriptors.values()
                    if descriptor.scope.scope_kind == "full_outer"
                    and int(descriptor.assignment.gpu_id) == 0
                ),
                None,
            )
            if selected is None:
                raise RuntimeError(
                    "the canonical descriptor set has no full-outer logical cuda:0 scope"
                )
            descriptor_set_manifest = (
                descriptor_set.root / LEGACY_STAGE1_SCOPE_DESCRIPTOR_SET_MANIFEST
            )
            preflight_phase_manifest = self._phase_manifest("stage1_preflight")
            body = {
                "schema_version": ("production_stage1_canary_descriptor_preparation_v1"),
                "status": "complete",
                "workflow_request_sha256": self.request["request_sha256"],
                "stage1_request_sha256": prepared.request_sha256,
                "source_snapshot": copy.deepcopy(self.request.get("source_snapshot")),
                "completed_workflow_prefix": list(prefix),
                "cluster_preflight_manifest": self._registered_file_identity(
                    cluster_preflight_manifest
                ),
                "stage1_preflight_phase_manifest": (
                    self._registered_file_identity(preflight_phase_manifest)
                ),
                "descriptor_set_manifest": self._registered_file_identity(descriptor_set_manifest),
                "descriptor_set_content_sha256": descriptor_set.manifest["content_sha256"],
                "descriptor_count": len(descriptor_set.descriptors),
                "selected_scope_id": selected.scope_id,
                "selected_scope_kind": selected.scope.scope_kind,
                "selected_logical_gpu_id": int(selected.assignment.gpu_id),
                "selected_descriptor_manifest": self._registered_file_identity(
                    selected.manifest_path
                ),
                "supervised_stage1_fits_started": False,
                "tfidf_component_started": False,
                "neural_query_component_started": False,
                "remote_clients_constructed": False,
                "remote_calls_made": False,
            }
            manifest = {**body, "content_sha256": _sha(body)}
            target = (
                self.options.work_root / "recovery" / "canary_descriptor_preparation_manifest.json"
            )
            if target.exists() or target.is_symlink():
                observed = _read_json_object(
                    target,
                    label="canary descriptor preparation manifest",
                )
                if observed != manifest:
                    raise RuntimeError("existing canary descriptor preparation manifest changed")
            else:
                _atomic_write_json(target, manifest)
            reopened = _read_json_object(
                target,
                label="canary descriptor preparation manifest",
            )
            if reopened != manifest:
                raise RuntimeError("canary descriptor preparation manifest failed fresh validation")
            reopened = self._validate_canary_preparation_in_fresh_process()
            if reopened != manifest:
                raise RuntimeError("fresh process changed the canary preparation result")
            _revalidate_request_bound_external_inputs(self.request)
        except BaseException as exc:
            self._write_progress(
                status="failed",
                completed=tuple(completed),
                current_phase="canary_descriptor_preparation",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._write_progress(
            status="canary_descriptors_ready",
            completed=tuple(completed),
            current_phase=None,
        )
        return reopened

    def run(self) -> Mapping[str, Any]:
        self._initialize()
        _revalidate_request_bound_external_inputs(self.request)
        completed = self._execute_phase_sequence(self._phase_sequence())
        self._write_progress(
            status="complete",
            completed=tuple(completed),
            current_phase=None,
        )
        return completed["terminal_validation"]["result"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "dataset",
        "work-root",
        "stage1-profile",
        "query-profile",
        "embedding-local-model-path",
        "htr-local-model-path",
    ):
        parser.add_argument("--" + flag, required=True, type=Path)
    for flag in (
        "unit-id-column",
        "text-column",
        "treatment-column",
        "outcome-column",
        "outcome-type",
        "clinical-question",
        "embedding-model-name",
    ):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--review-rounds", type=int, default=2)
    parser.add_argument("--interaction-inner-folds", type=int, default=3)
    parser.add_argument("--tfidf-nested-calibration-folds", type=int, default=3)
    parser.add_argument("--stage1-device", default="cuda:1")
    parser.add_argument(
        "--query-device",
        action="append",
        default=[],
        help="Ordered Stage 1 neural-query device; repeat to use multiple devices.",
    )
    parser.add_argument("--review-device", default="cuda:1")
    parser.add_argument(
        "--stage1-gpu-id",
        type=int,
        action="append",
        default=[],
        help="Ordered exclusive Stage 1 GPU; repeat once per GPU.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        help="Backward-compatible singular alias for one --stage1-gpu-id.",
    )
    parser.add_argument("--stage1-scope-workers-per-gpu", type=int, default=1)
    parser.add_argument("--stage1-preflight-workers", type=int, default=8)
    parser.add_argument(
        "--stage1-seed-policy",
        default="canonical_scope_sha256_v1",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--tfidf-workers", type=int, default=8)
    parser.add_argument(
        "--tfidf-parallel-backend", choices=("threads", "processes"), default="processes"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--empty-text-policy", default="marker")
    parser.add_argument("--repeated-character-policy", default="marker")
    parser.add_argument("--repeated-character-threshold", type=int, default=1000)
    parser.add_argument(
        "--embedding-cache-import",
        type=Path,
        help="Authenticated source cache to relocate into the fresh work root.",
    )
    parser.add_argument(
        "--embedding-cache-import-source-prepared",
        dest="embedding_cache_import_source_prepared_path",
        type=Path,
    )
    parser.add_argument(
        "--embedding-cache-import-source-preparation-manifest",
        dest="embedding_cache_import_source_preparation_manifest_path",
        type=Path,
    )
    parser.add_argument("--source-snapshot-root", type=Path)
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help=(
            "Stop after a fresh-process Stage 1 handoff validation; endpoint/model "
            "are not required and no Stage 2 client is imported or constructed."
        ),
    )
    parser.add_argument(
        "--prepare-stage1-canary-descriptors-only",
        action="store_true",
        help=(
            "Operational pre-launch mode: seal input/cache/preflight and the "
            "exact private descriptor set, then exit before any Stage 1 fit. "
            "This flag is excluded from the immutable scientific request."
        ),
    )
    parser.add_argument("--evaluate-oracle-posthoc", action="store_true")
    parser.add_argument("--oracle-dataset", type=Path)
    parser.add_argument("--oracle-unit-id-column")
    parser.add_argument("--oracle-ite-column")
    parser.add_argument("--resume", action="store_true")
    return parser


def options_from_args(args: argparse.Namespace) -> ProductionAllEvidenceWorkflowOptions:
    values = vars(args).copy()
    values.pop("prepare_stage1_canary_descriptors_only", None)
    values["dataset_path"] = values.pop("dataset")
    values["stage1_profile_path"] = values.pop("stage1_profile")
    values["query_profile_path"] = values.pop("query_profile")
    values["oracle_dataset_path"] = values.pop("oracle_dataset")
    values["model_name"] = values.pop("model")
    values["stage1_gpu_ids"] = tuple(values.pop("stage1_gpu_id"))
    values["query_devices"] = tuple(values.pop("query_device"))
    values["query_device"] = None
    options = ProductionAllEvidenceWorkflowOptions(**values)
    # Validate CLI-only conflicts before any paths are hashed or work is begun.
    ProductionAllEvidenceWorkflow(options)
    return options


def _reexec_from_source_snapshot(
    *,
    parsed_args: argparse.Namespace,
    raw_argv: Sequence[str],
) -> None:
    """Replace the current process so all subsequent imports use the snapshot."""

    snapshot_root = getattr(parsed_args, "source_snapshot_root", None)
    if snapshot_root is None:
        return
    from .production_source_snapshot import validate_production_source_snapshot

    snapshot = validate_production_source_snapshot(snapshot_root)
    loaded_root = Path(__file__).resolve().parents[2]
    marker = os.environ.get(SOURCE_SNAPSHOT_EXECUTION_ENV)
    requested_hash_seed = int(getattr(parsed_args, "seed"))
    if requested_hash_seed < 0:
        raise ValueError("source-snapshot execution requires a nonnegative seed")
    expected_hash_seed = str(requested_hash_seed)
    if marker is not None:
        if (
            marker != snapshot.content_sha256
            or loaded_root != snapshot.root
            or os.environ.get("PYTHONHASHSEED") != expected_hash_seed
        ):
            raise RuntimeError(
                "source-snapshot execution marker, loaded source tree, or "
                "PYTHONHASHSEED does not match the requested run"
            )
        return
    entrypoint = snapshot.root / "scripts" / "run_production_all_evidence_workflow.py"
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise FileNotFoundError("source snapshot lacks the production workflow entry point")
    environment = os.environ.copy()
    environment[SOURCE_SNAPSHOT_EXECUTION_ENV] = snapshot.content_sha256
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(snapshot.root)
    # Python reads this setting only while starting the interpreter.  Binding
    # it to the configured global seed before exec makes parent-side hashing
    # deterministic as well as the per-scope child seeds enforced later by
    # the Stage 1 scheduler.
    environment["PYTHONHASHSEED"] = expected_hash_seed
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-P",
            "-u",
            str(entrypoint),
            *[str(value) for value in raw_argv],
        ],
        environment,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    try:
        _reexec_from_source_snapshot(parsed_args=args, raw_argv=raw_argv)
        options = options_from_args(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    workflow = ProductionAllEvidenceWorkflow(options)
    if bool(args.prepare_stage1_canary_descriptors_only):
        result = workflow.prepare_stage1_canary_descriptors_only()
    else:
        result = workflow.run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "EMBEDDING_CACHE_PHASE_SCHEMA",
    "PHASES",
    "STAGE1_PREFLIGHT_PHASE_SCHEMA",
    "STAGE1_ONLY_PHASES",
    "ProductionAllEvidenceWorkflow",
    "ProductionAllEvidenceWorkflowHooks",
    "ProductionAllEvidenceWorkflowOptions",
    "WorkflowPhaseHook",
    "build_parser",
    "main",
    "options_from_args",
    "validate_stage1_canary_descriptor_preparation",
    "validate_completed_workflow_prefix",
]
