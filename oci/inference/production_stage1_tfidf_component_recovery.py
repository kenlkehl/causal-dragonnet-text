"""Stable spawned recovery for the complete production TF-IDF component."""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import signal
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd

from .production_stage1_scope_scheduler import (
    _attempt_inventory,
    _directory_inode_binding,
    _durably_sync_attempt_parent_chain,
    _durably_sync_attempt_tree,
    _load_strict_json_file,
    _sha256_file,
    _sha256_json,
    _start_spawned_process_with_scope_hash_seed,
    _establish_worker_process_group,
    _terminate_process_and_descendants,
    _validate_directory_inode_binding,
    _write_immutable_json,
)

TFIDF_COMPONENT_DESCRIPTOR_SCHEMA = "production_tfidf_component_descriptor_v2"
TFIDF_COMPONENT_ATTEMPT_REQUEST_SCHEMA = (
    "production_tfidf_component_attempt_request_v1"
)
TFIDF_COMPONENT_ATTEMPT_MANIFEST_SCHEMA = (
    "production_tfidf_component_attempt_manifest_v1"
)
TFIDF_COMPONENT_PROGRESS_SCHEMA = "production_tfidf_component_progress_v1"
_UTC_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _file_registration(path: Path, *, root: Path) -> Mapping[str, Any]:
    resolved = path.resolve(strict=True)
    relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    return {
        "relative_path": relative,
        "size": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _current_code_identity() -> Mapping[str, str]:
    code_files = (
        Path(__file__).resolve(strict=True),
        Path(__file__).with_name("production_stage1_tfidf_parallel.py").resolve(
            strict=True
        ),
        Path(__file__).with_name("tfidf_topic_stage1.py").resolve(strict=True),
        Path(__file__).with_name("production_stage1_bundle.py").resolve(
            strict=True
        ),
    )
    return {path.name: _sha256_file(path) for path in code_files}


@dataclass(frozen=True)
class AuthenticatedTfidfComponentDescriptor:
    root: Path
    manifest: Mapping[str, Any]

    @property
    def manifest_path(self) -> Path:
        return self.root / "descriptor_manifest.json"

    @property
    def content_sha256(self) -> str:
        return str(self.manifest["content_sha256"])


def load_tfidf_component_descriptor(
    manifest_path: Path | str,
    *,
    expected_request_sha256: str,
) -> AuthenticatedTfidfComponentDescriptor:
    path = Path(manifest_path).resolve(strict=True)
    value = _load_strict_json_file(path, label="TF-IDF component descriptor")
    if not isinstance(value, Mapping):
        raise ValueError("TF-IDF component descriptor is not an object")
    body = dict(value)
    declared = body.pop("content_sha256", None)
    required = {
        "schema_version",
        "scientific_request_sha256",
        "row_count",
        "modeling_columns",
        "registry_content_sha256",
        "tfidf_workers",
        "seed",
        "descriptor_root_binding",
        "files",
        "code_identity",
        "content_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version") != TFIDF_COMPONENT_DESCRIPTOR_SCHEMA
        or value.get("scientific_request_sha256") != expected_request_sha256
        or _sha256_json(body) != declared
        or not isinstance(value.get("files"), Mapping)
        or value.get("code_identity") != _current_code_identity()
    ):
        raise ValueError("TF-IDF component descriptor has an invalid binding")
    root = path.parent.resolve(strict=True)
    _validate_directory_inode_binding(
        value.get("descriptor_root_binding"),
        path=root,
        label="TF-IDF descriptor publication",
    )
    expected_file_names = {
        "modeling_data": "modeling_data.parquet",
        "effective_config": "effective_config.json",
        "registry": "registry.json",
        "split_registry": "split_registry.json",
    }
    if set(value["files"]) != set(expected_file_names):
        raise ValueError("TF-IDF descriptor file registry is not closed")
    for name, registration in value["files"].items():
        if (
            not isinstance(registration, Mapping)
            or set(registration) != {"relative_path", "size", "sha256"}
            or registration.get("relative_path") != expected_file_names[name]
        ):
            raise ValueError("TF-IDF descriptor file registration is invalid")
        target = (root / str(registration.get("relative_path") or "")).resolve(
            strict=True
        )
        target.relative_to(root)
        if (
            int(target.stat().st_size) != int(registration.get("size", -1))
            or _sha256_file(target) != registration.get("sha256")
        ):
            raise ValueError(f"TF-IDF descriptor file changed: {name}")
    inventory = _attempt_inventory(root)
    if {row["relative_path"] for row in inventory} != {
        *expected_file_names.values(),
        "descriptor_manifest.json",
    }:
        raise ValueError("TF-IDF descriptor publication tree is not closed")
    return AuthenticatedTfidfComponentDescriptor(root=root, manifest=dict(value))


def publish_tfidf_component_descriptor(
    *,
    descriptor_root: Path | str,
    scientific_request_sha256: str,
    modeling_data: pd.DataFrame,
    effective_config: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_content_sha256: str,
    split_registry_path: Path | str,
    tfidf_workers: int,
    seed: int,
) -> AuthenticatedTfidfComponentDescriptor:
    container = Path(descriptor_root).resolve()
    container.mkdir(parents=True, exist_ok=True)
    completed = []
    for publication in sorted(container.glob("publication_*")):
        if not publication.is_dir() or publication.is_symlink():
            raise ValueError("TF-IDF descriptor container has an unsafe entry")
        manifest = publication / "descriptor_manifest.json"
        if manifest.is_file():
            completed.append(
                load_tfidf_component_descriptor(
                    manifest,
                    expected_request_sha256=scientific_request_sha256,
                )
            )
    if len(completed) > 1:
        raise RuntimeError("multiple completed TF-IDF descriptor publications exist")
    if completed:
        return completed[0]
    root = container / (
        "publication_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "_"
        + uuid.uuid4().hex
    )
    root.mkdir()
    manifest_path = root / "descriptor_manifest.json"
    frame = modeling_data.reset_index(drop=True).copy()
    if len(frame.columns) != 3:
        raise ValueError("TF-IDF descriptor requires exactly text/treatment/outcome")
    modeling_path = root / "modeling_data.parquet"
    frame.to_parquet(modeling_path, index=False)
    config_path = root / "effective_config.json"
    _write_immutable_json(config_path, dict(effective_config))
    registry_path = root / "registry.json"
    _write_immutable_json(registry_path, dict(registry))
    split_path = root / "split_registry.json"
    shutil.copyfile(Path(split_registry_path).resolve(strict=True), split_path)
    code_identity = _current_code_identity()
    files = {
        "modeling_data": _file_registration(modeling_path, root=root),
        "effective_config": _file_registration(config_path, root=root),
        "registry": _file_registration(registry_path, root=root),
        "split_registry": _file_registration(split_path, root=root),
    }
    body = {
        "schema_version": TFIDF_COMPONENT_DESCRIPTOR_SCHEMA,
        "scientific_request_sha256": str(scientific_request_sha256),
        "row_count": len(frame),
        "modeling_columns": list(map(str, frame.columns)),
        "registry_content_sha256": str(registry_content_sha256),
        "tfidf_workers": int(tfidf_workers),
        "seed": int(seed),
        "descriptor_root_binding": _directory_inode_binding(
            root,
            label="TF-IDF descriptor publication",
        ),
        "files": files,
        "code_identity": code_identity,
    }
    _durably_sync_attempt_tree(root)
    _durably_sync_attempt_parent_chain(root)
    _write_immutable_json(
        manifest_path,
        {**body, "content_sha256": _sha256_json(body)},
    )
    return load_tfidf_component_descriptor(
        manifest_path,
        expected_request_sha256=scientific_request_sha256,
    )


@dataclass(frozen=True)
class TfidfComponentAttemptRequest:
    attempt_dir: str
    scientific_request_sha256: str
    descriptor_manifest_path: str
    descriptor_content_sha256: str
    attempt_request_sha256: str
    seed: int


@dataclass(frozen=True)
class TfidfComponentAttemptHandle:
    request: TfidfComponentAttemptRequest
    process: mp.Process


@dataclass(frozen=True)
class ValidatedTfidfComponentAttempt:
    attempt_dir: Path
    component_root: Path
    manifest: Mapping[str, Any]


def _seal_tfidf_attempt(
    request: TfidfComponentAttemptRequest,
    *,
    result: Mapping[str, Any],
) -> Mapping[str, Any]:
    attempt = Path(request.attempt_dir)
    result_body = {
        "schema_version": "production_tfidf_component_worker_result_v1",
        "scientific_request_sha256": request.scientific_request_sha256,
        "descriptor_content_sha256": request.descriptor_content_sha256,
        "result": dict(result),
    }
    _write_immutable_json(
        attempt / "worker_result.json",
        {**result_body, "content_sha256": _sha256_json(result_body)},
    )
    _durably_sync_attempt_tree(attempt)
    _durably_sync_attempt_parent_chain(attempt)
    inventory = _attempt_inventory(attempt)
    body = {
        "schema_version": TFIDF_COMPONENT_ATTEMPT_MANIFEST_SCHEMA,
        "status": "completed",
        "scientific_request_sha256": request.scientific_request_sha256,
        "descriptor_content_sha256": request.descriptor_content_sha256,
        "attempt_request_sha256": request.attempt_request_sha256,
        "files": inventory,
    }
    manifest = {**body, "content_sha256": _sha256_json(body)}
    _write_immutable_json(attempt / "attempt_manifest.json", manifest)
    return manifest


def _tfidf_component_worker(
    request: TfidfComponentAttemptRequest,
) -> None:
    attempt = Path(request.attempt_dir)
    terminal = False
    try:
        _establish_worker_process_group(
            attempt / "process_group_ready.json"
        )
        from .production_stage1_scope_scheduler import (
            _enforce_stage1_torch_determinism,
            seed_stage1_scope_rngs,
        )

        _enforce_stage1_torch_determinism()
        seed_stage1_scope_rngs(request.seed, gpu_id=None)
        descriptor = load_tfidf_component_descriptor(
            request.descriptor_manifest_path,
            expected_request_sha256=request.scientific_request_sha256,
        )
        if descriptor.content_sha256 != request.descriptor_content_sha256:
            raise ValueError("TF-IDF descriptor identity changed")
        from .production_stage1_bundle import (
            ProductionStage1BundleBuilder,
            _load_component_manifest,
            _seal_component,
            load_applied_stage1_config,
        )

        files = descriptor.manifest["files"]
        config = load_applied_stage1_config(
            descriptor.root / files["effective_config"]["relative_path"]
        )
        modeling = pd.read_parquet(
            descriptor.root / files["modeling_data"]["relative_path"],
            columns=list(descriptor.manifest["modeling_columns"]),
        )
        registry = _load_strict_json_file(
            descriptor.root / files["registry"]["relative_path"],
            label="TF-IDF descriptor registry",
        )
        workspace = attempt / "payload"
        workspace.mkdir()
        shutil.copyfile(
            descriptor.root / files["split_registry"]["relative_path"],
            workspace / "split_registry.json",
        )
        config.architecture.multi_model_forest.split_registry_path = str(
            (workspace / "split_registry.json").resolve(strict=True)
        )
        prepared = SimpleNamespace(
            config=config,
            modeling_data=modeling,
            request_sha256=request.scientific_request_sha256,
            registry=registry,
            registry_content_sha256=descriptor.manifest[
                "registry_content_sha256"
            ],
            options=SimpleNamespace(
                tfidf_workers=int(descriptor.manifest["tfidf_workers"]),
                tfidf_parallel_backend="processes",
            ),
        )
        component = workspace / "tfidf"
        builder = object.__new__(ProductionStage1BundleBuilder)
        builder._run_tfidf_component(component, workspace, prepared)
        sealed = _seal_component(
            component,
            request_sha256=request.scientific_request_sha256,
            component="tfidf",
        )
        if (
            _load_component_manifest(
                component,
                request_sha256=request.scientific_request_sha256,
                component="tfidf",
            )
            != sealed
        ):
            raise RuntimeError("TF-IDF component seal changed after publication")
        _seal_tfidf_attempt(
            request,
            result={
                "component_relative_path": "payload/tfidf",
                "component_manifest_sha256": sealed["content_sha256"],
            },
        )
        terminal = True
    except BaseException as exc:
        if not terminal and not (attempt / "attempt_manifest.json").exists():
            failure = {
                "schema_version": "production_tfidf_component_failure_v1",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_immutable_json(
                attempt / "failure.json",
                {**failure, "content_sha256": _sha256_json(failure)},
            )
        raise


class TfidfComponentAttemptManager:
    """Start, authenticate, resume, and terminate one stable component attempt."""

    def __init__(
        self,
        *,
        attempt_root: Path | str,
        progress_path: Path | str,
        descriptor: AuthenticatedTfidfComponentDescriptor,
        scientific_request_sha256: str,
        seed: int,
    ) -> None:
        self.attempt_root = Path(attempt_root).resolve()
        self.attempt_root.mkdir(parents=True, exist_ok=True)
        self.progress_path = Path(progress_path).resolve()
        self.descriptor = descriptor
        self.scientific_request_sha256 = str(scientific_request_sha256)
        self.seed = int(seed)
        self._root_binding = _directory_inode_binding(
            self.attempt_root,
            label="TF-IDF component attempt root",
        )

    def _write_progress(
        self,
        status: str,
        *,
        attempt_dir: str | None,
        failure: Mapping[str, Any] | None = None,
    ) -> None:
        body = {
            "schema_version": TFIDF_COMPONENT_PROGRESS_SCHEMA,
            "scientific_request_sha256": self.scientific_request_sha256,
            "descriptor_content_sha256": self.descriptor.content_sha256,
            "attempt_root": str(self.attempt_root),
            "attempt_root_binding": self._root_binding,
            "progress_path": str(self.progress_path),
            "status": status,
            "attempt_dir": attempt_dir,
            "failure": None if failure is None else dict(failure),
            "updated_at": _utc_now(),
        }
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        from .production_stage1_scope_scheduler import _atomic_write_json

        _atomic_write_json(
            self.progress_path,
            {**body, "content_sha256": _sha256_json(body)},
        )

    def _validate(self, attempt: Path) -> ValidatedTfidfComponentAttempt:
        attempt = attempt.resolve(strict=True)
        if (
            attempt.parent != self.attempt_root
            or not attempt.name.startswith("attempt_")
        ):
            raise ValueError("TF-IDF attempt escaped its stable recovery root")
        _validate_directory_inode_binding(
            self._root_binding,
            path=self.attempt_root,
            label="TF-IDF component attempt root",
        )
        request = _load_strict_json_file(
            attempt / "attempt_request.json",
            label="TF-IDF component attempt request",
        )
        if not isinstance(request, Mapping):
            raise ValueError("TF-IDF component attempt request is not an object")
        request_body = dict(request)
        request_sha = request_body.pop("attempt_request_sha256", None)
        request_fields = {
            "schema_version",
            "scientific_request_sha256",
            "descriptor_manifest_path",
            "descriptor_content_sha256",
            "attempt_root_binding",
            "attempt_directory_binding",
            "seed",
            "created_at",
            "attempt_request_sha256",
        }
        if (
            set(request) != request_fields
            or request.get("schema_version")
            != TFIDF_COMPONENT_ATTEMPT_REQUEST_SCHEMA
            or _sha256_json(request_body) != request_sha
            or request.get("scientific_request_sha256")
            != self.scientific_request_sha256
            or request.get("descriptor_manifest_path")
            != str(self.descriptor.manifest_path)
            or request.get("descriptor_content_sha256")
            != self.descriptor.content_sha256
            or request.get("attempt_root_binding") != self._root_binding
            or int(request.get("seed", -1)) != self.seed
            or _UTC_TIMESTAMP.fullmatch(
                str(request.get("created_at") or "")
            )
            is None
        ):
            raise ValueError("TF-IDF component attempt request changed")
        _validate_directory_inode_binding(
            request.get("attempt_directory_binding"),
            path=attempt,
            label="TF-IDF component attempt",
        )
        descriptor = load_tfidf_component_descriptor(
            request["descriptor_manifest_path"],
            expected_request_sha256=self.scientific_request_sha256,
        )
        if descriptor.content_sha256 != self.descriptor.content_sha256:
            raise ValueError("TF-IDF attempt descriptor was substituted")
        manifest = _load_strict_json_file(
            attempt / "attempt_manifest.json",
            label="TF-IDF component terminal manifest",
        )
        if not isinstance(manifest, Mapping):
            raise ValueError("TF-IDF attempt manifest is not an object")
        body = dict(manifest)
        declared = body.pop("content_sha256", None)
        manifest_fields = {
            "schema_version",
            "status",
            "scientific_request_sha256",
            "descriptor_content_sha256",
            "attempt_request_sha256",
            "files",
            "content_sha256",
        }
        if (
            set(manifest) != manifest_fields
            or
            manifest.get("schema_version")
            != TFIDF_COMPONENT_ATTEMPT_MANIFEST_SCHEMA
            or manifest.get("status") != "completed"
            or manifest.get("scientific_request_sha256")
            != self.scientific_request_sha256
            or manifest.get("descriptor_content_sha256")
            != self.descriptor.content_sha256
            or manifest.get("attempt_request_sha256") != request_sha
            or _sha256_json(body) != declared
            or manifest.get("files") != _attempt_inventory(attempt)
        ):
            raise ValueError("TF-IDF attempt terminal binding is invalid")
        worker_result = _load_strict_json_file(
            attempt / "worker_result.json",
            label="TF-IDF component worker result",
        )
        if not isinstance(worker_result, Mapping):
            raise ValueError("TF-IDF component worker result is not an object")
        worker_body = dict(worker_result)
        worker_sha = worker_body.pop("content_sha256", None)
        worker_fields = {
            "schema_version",
            "scientific_request_sha256",
            "descriptor_content_sha256",
            "result",
            "content_sha256",
        }
        result = worker_result.get("result")
        if (
            set(worker_result) != worker_fields
            or worker_result.get("schema_version")
            != "production_tfidf_component_worker_result_v1"
            or _sha256_json(worker_body) != worker_sha
            or worker_result.get("scientific_request_sha256")
            != self.scientific_request_sha256
            or worker_result.get("descriptor_content_sha256")
            != self.descriptor.content_sha256
            or not isinstance(result, Mapping)
            or set(result)
            != {
                "component_relative_path",
                "component_manifest_sha256",
            }
            or result.get("component_relative_path") != "payload/tfidf"
        ):
            raise ValueError("TF-IDF component worker result changed")
        component = attempt / "payload" / "tfidf"
        from .production_stage1_bundle import _load_component_manifest

        component_manifest = _load_component_manifest(
            component,
            request_sha256=self.scientific_request_sha256,
            component="tfidf",
        )
        if component_manifest is None or result.get(
            "component_manifest_sha256"
        ) != component_manifest.get("content_sha256"):
            raise ValueError("TF-IDF component attempt lacks its component seal")
        _validate_directory_inode_binding(
            request.get("attempt_directory_binding"),
            path=attempt,
            label="TF-IDF component attempt after validation",
        )
        return ValidatedTfidfComponentAttempt(
            attempt_dir=attempt,
            component_root=component,
            manifest=dict(manifest),
        )

    def reusable(self) -> ValidatedTfidfComponentAttempt | None:
        completed = []
        for attempt in sorted(self.attempt_root.glob("attempt_*")):
            if not attempt.is_dir() or attempt.is_symlink():
                raise ValueError("TF-IDF attempt tree contains an unsafe entry")
            if not (attempt / "attempt_manifest.json").is_file():
                continue
            completed.append(self._validate(attempt.resolve(strict=True)))
        if len(completed) > 1:
            raise RuntimeError("multiple completed TF-IDF attempts exist")
        return None if not completed else completed[0]

    def start(self) -> TfidfComponentAttemptHandle | ValidatedTfidfComponentAttempt:
        reusable = self.reusable()
        if reusable is not None:
            self._write_progress(
                "completed",
                attempt_dir=str(reusable.attempt_dir),
            )
            return reusable
        attempt = self.attempt_root / (
            "attempt_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + "_"
            + uuid.uuid4().hex
        )
        attempt.mkdir()
        body = {
            "schema_version": TFIDF_COMPONENT_ATTEMPT_REQUEST_SCHEMA,
            "scientific_request_sha256": self.scientific_request_sha256,
            "descriptor_manifest_path": str(self.descriptor.manifest_path),
            "descriptor_content_sha256": self.descriptor.content_sha256,
            "attempt_root_binding": self._root_binding,
            "attempt_directory_binding": _directory_inode_binding(
                attempt,
                label="TF-IDF component attempt",
            ),
            "seed": self.seed,
            "created_at": _utc_now(),
        }
        payload = {**body, "attempt_request_sha256": _sha256_json(body)}
        _write_immutable_json(attempt / "attempt_request.json", payload)
        request = TfidfComponentAttemptRequest(
            attempt_dir=str(attempt),
            scientific_request_sha256=self.scientific_request_sha256,
            descriptor_manifest_path=str(self.descriptor.manifest_path),
            descriptor_content_sha256=self.descriptor.content_sha256,
            attempt_request_sha256=payload["attempt_request_sha256"],
            seed=self.seed,
        )
        process = mp.get_context("spawn").Process(
            target=_tfidf_component_worker,
            args=(request,),
            name="production-stage1-tfidf-component",
        )
        _start_spawned_process_with_scope_hash_seed(
            process,
            scope_seed=self.seed,
        )
        self._write_progress("running", attempt_dir=str(attempt))
        return TfidfComponentAttemptHandle(request=request, process=process)

    def wait(
        self,
        handle: TfidfComponentAttemptHandle,
    ) -> ValidatedTfidfComponentAttempt:
        handle.process.join()
        attempt = Path(handle.request.attempt_dir)
        try:
            validated = self._validate(attempt)
        except Exception as exc:
            failure = {
                "exception_type": (
                    "WorkerProcessError"
                    if handle.process.exitcode != 0
                    else "WorkerProtocolError"
                ),
                "message": f"{type(exc).__name__}: {exc}",
            }
            self._write_progress(
                "failed",
                attempt_dir=str(attempt),
                failure=failure,
            )
            raise RuntimeError(
                "TF-IDF component attempt failed authentication: "
                + failure["message"]
            ) from exc
        self._write_progress("completed", attempt_dir=str(attempt))
        return validated

    def terminate(
        self,
        handle: TfidfComponentAttemptHandle,
        *,
        reason: str,
    ) -> None:
        process = handle.process
        _terminate_process_and_descendants(
            process,
            process_group_marker_path=(
                Path(handle.request.attempt_dir)
                / "process_group_ready.json"
            ),
        )
        self._write_progress(
            "failed",
            attempt_dir=handle.request.attempt_dir,
            failure={
                "exception_type": "PeerComponentFailure",
                "message": str(reason),
            },
        )

    def materialize(
        self,
        attempt: ValidatedTfidfComponentAttempt,
        *,
        target: Path | str,
    ) -> Path:
        authenticated = self._validate(attempt.attempt_dir)
        destination = Path(target)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("TF-IDF materialization target must be fresh")
        shutil.copytree(authenticated.component_root, destination)
        from .production_stage1_bundle import _load_component_manifest

        if (
            _load_component_manifest(
                destination,
                request_sha256=self.scientific_request_sha256,
                component="tfidf",
            )
            is None
        ):
            raise RuntimeError("materialized TF-IDF component failed validation")
        return destination


__all__ = [
    "AuthenticatedTfidfComponentDescriptor",
    "TfidfComponentAttemptHandle",
    "TfidfComponentAttemptManager",
    "ValidatedTfidfComponentAttempt",
    "load_tfidf_component_descriptor",
    "publish_tfidf_component_descriptor",
]
