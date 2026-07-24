"""Physically isolated inputs for one clustered-embedding preflight scope.

The clustered preflight is label-dependent.  A process evaluating one scope
therefore receives only that scope's fit text/labels and a cache view whose
non-fit rows contain no chunks or embeddings.  This module publishes and
authenticates those capabilities without exposing the prepared cohort or the
global embedding-cache path in a worker payload.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import (
    AppliedInferenceConfig,
    EmbeddingContrastDiscoveryConfig,
    ExperimentConfig,
)
from .embedding_native_proof_capture import LOGICAL_FROZEN_EMBEDDING_CACHE_URI
from .production_stage1_config_wire import (
    production_stage1_effective_config_payload,
)
from .production_stage1_legacy_scope_adapter import (
    _RestrictedLogicalIdentityEmbeddingCache,
    _closed_tree_inventory,
    _file_registration,
    _read_exact_parquet,
    _read_json,
    _validate_registration,
    _write_json,
    _write_parquet,
    _write_private_embedding_cache,
)

PREFLIGHT_SCOPE_INPUT_SCHEMA = "production_stage1_preflight_scope_input_v2"
PREFLIGHT_SCOPE_INPUT_SET_SCHEMA = "production_stage1_preflight_scope_input_set_v2"
PREFLIGHT_ONE_SCOPE_AUTHORITY_SCHEMA = "production_stage1_preflight_one_scope_authority_v1"
PREFLIGHT_SCOPE_INPUT_MANIFEST = "preflight_scope_input_manifest.json"
PREFLIGHT_SCOPE_INPUT_SET_MANIFEST = "preflight_scope_input_set_manifest.json"

_CONFIG_FILE = "effective_config.json"
_SCOPE_AUTHORITY_FILE = "one_scope_authority.json"
_MODELING_FILE = "fit_only_modeling.parquet"
_HEX = frozenset("0123456789abcdef")


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


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _scope_value(scope: Mapping[str, Any], key: str) -> Any:
    if key not in scope:
        raise ValueError(f"preflight scope lacks {key}")
    return scope[key]


def _private_config_payload(
    *,
    config: AppliedInferenceConfig,
    forbidden_paths: Sequence[Path],
) -> dict[str, Any]:
    # Runtime receives the physical inputs as separately authenticated
    # capabilities.  Keeping neutral URIs here makes the scientific
    # configuration independent of an attempt/recovery location.
    modeling_path = "production://private-preflight/fit-only-modeling-v1"
    cache_path = LOGICAL_FROZEN_EMBEDDING_CACHE_URI

    def rewrite(value: Any, *, key: str | None = None) -> Any:
        if key == "dataset_path":
            return modeling_path
        if key == "cache_dir":
            return cache_path
        if key == "external_corpus_cache_dirs":
            return []
        if isinstance(value, Mapping):
            return {
                str(child_key): rewrite(child_value, key=str(child_key))
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [rewrite(child) for child in value]
        return copy.deepcopy(value)

    payload = rewrite(production_stage1_effective_config_payload(config))
    serialized = _canonical_json(payload)
    forbidden = tuple(str(path.resolve(strict=False)) for path in forbidden_paths)
    if any(value in serialized for value in forbidden):
        raise ValueError("preflight scope configuration exposes a prepared cohort or global cache")
    return payload


@dataclass(frozen=True)
class AuthenticatedPreflightScopeInput:
    root: Path
    manifest: Mapping[str, Any]
    modeling_data: pd.DataFrame
    config: AppliedInferenceConfig
    scope_authority: Mapping[str, Any]
    scope: Mapping[str, Any]
    embedding_cache: _RestrictedLogicalIdentityEmbeddingCache

    @property
    def manifest_path(self) -> Path:
        return self.root / PREFLIGHT_SCOPE_INPUT_MANIFEST

    @property
    def scope_id(self) -> str:
        return str(self.scope["scope_id"])

    def worker_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "production_stage1_preflight_worker_payload_v1",
            "scope_id": self.scope_id,
            "manifest_path": str(self.manifest_path),
            "manifest_content_sha256": str(self.manifest["content_sha256"]),
        }


@dataclass(frozen=True)
class AuthenticatedPreflightScopeInputSet:
    root: Path
    manifest: Mapping[str, Any]
    scopes: Mapping[str, AuthenticatedPreflightScopeInput]

    def worker_payloads(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(scope.worker_payload() for scope in self.scopes.values())

    def identity(self) -> dict[str, Any]:
        manifest_registration = _file_registration(
            self.root / PREFLIGHT_SCOPE_INPUT_SET_MANIFEST,
            self.root,
        )
        attempt_root = self.root.parent / f".{self.root.name}.scope_attempts"
        attempts = (
            sorted(entry.name for entry in os.scandir(attempt_root))
            if attempt_root.is_dir() and not attempt_root.is_symlink()
            else []
        )
        body = {
            "schema_version": "production_stage1_preflight_scope_input_set_identity_v1",
            "root": str(self.root),
            "manifest_path": str(self.root / PREFLIGHT_SCOPE_INPUT_SET_MANIFEST),
            "manifest": manifest_registration,
            "manifest_content_sha256": str(self.manifest["content_sha256"]),
            "scope_order": list(self.scopes),
            "scope_manifest_content_sha256": {
                scope_id: str(scope.manifest["content_sha256"])
                for scope_id, scope in self.scopes.items()
            },
            "attempt_root": str(attempt_root),
            "preserved_incomplete_attempts": attempts,
            "scope_inputs_outside_terminal_scientific_artifact": True,
        }
        return {**body, "content_sha256": _sha256_json(body)}


def _write_scope(
    *,
    root: Path,
    modeling_data: pd.DataFrame,
    config: AppliedInferenceConfig,
    embedding_cache: Any,
    embedding_cache_identity: Mapping[str, Any],
    registry_content_sha256: str,
    scope: Mapping[str, Any],
    forbidden_paths: Sequence[Path],
) -> None:
    root.mkdir(parents=True, exist_ok=False)
    fit_rows = tuple(map(int, _scope_value(scope, "fit_row_ids")))
    row_count = len(modeling_data)
    if not fit_rows or min(fit_rows) < 0 or max(fit_rows) >= row_count:
        raise ValueError("preflight scope fit rows are invalid")
    fit_texts = tuple(
        str(value)
        for value in modeling_data.iloc[list(fit_rows)][config.text_column].tolist()
    )
    parent_binding = embedding_cache.bind_spent(fit_rows, fit_texts)
    token_bounded = getattr(parent_binding, "token_bounded_row_ids", None)
    if not isinstance(token_bounded, tuple) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in token_bounded
    ):
        raise ValueError(
            f"embedding cache binding is malformed in {scope['scope_id']}"
        )
    if token_bounded:
        raise ValueError(
            "embedding cache binding used token-bounded text reconciliation in "
            f"{scope['scope_id']}"
        )
    private = pd.DataFrame(
        {
            config.text_column: np.full(row_count, "", dtype=object),
            config.treatment_column: np.full(row_count, np.nan, dtype=float),
            config.outcome_column: np.full(row_count, np.nan, dtype=float),
        }
    )
    private.loc[
        list(fit_rows),
        [
            config.text_column,
            config.treatment_column,
            config.outcome_column,
        ],
    ] = modeling_data.iloc[list(fit_rows)][
        [
            config.text_column,
            config.treatment_column,
            config.outcome_column,
        ]
    ].to_numpy(
        copy=True
    )
    _write_parquet(root / _MODELING_FILE, private)
    _write_json(
        root / _CONFIG_FILE,
        _private_config_payload(
            config=config,
            forbidden_paths=forbidden_paths,
        ),
    )
    authority_body = {
        "schema_version": PREFLIGHT_ONE_SCOPE_AUTHORITY_SCHEMA,
        "registry_content_sha256": registry_content_sha256,
        "dataset_row_count": row_count,
        "scope": copy.deepcopy(dict(scope)),
        "scope_binding_sha256": _sha256_json(
            {
                "registry_content_sha256": registry_content_sha256,
                "scope": scope,
            }
        ),
        "authorized_scope_count": 1,
        "other_scope_definitions_supplied": False,
        "other_scope_row_identities_supplied": False,
    }
    _write_json(
        root / _SCOPE_AUTHORITY_FILE,
        {
            **authority_body,
            "content_sha256": _sha256_json(authority_body),
        },
    )
    prepared = SimpleNamespace(
        embedding_cache=embedding_cache,
        embedding_cache_identity=copy.deepcopy(dict(embedding_cache_identity)),
    )
    scope_object = SimpleNamespace(
        scope_id=str(scope["scope_id"]),
        fit_row_ids=fit_rows,
    )
    private_cache = _write_private_embedding_cache(
        root=root,
        prepared=prepared,
        scope=scope_object,
    )
    files = {
        "effective_config": _file_registration(root / _CONFIG_FILE, root),
        "one_scope_authority": _file_registration(
            root / _SCOPE_AUTHORITY_FILE,
            root,
        ),
        "fit_only_modeling": _file_registration(root / _MODELING_FILE, root),
    }
    body = {
        "schema_version": PREFLIGHT_SCOPE_INPUT_SCHEMA,
        "scope": copy.deepcopy(dict(scope)),
        "scope_binding_sha256": _sha256_json(
            {
                "registry_content_sha256": registry_content_sha256,
                "scope": scope,
            }
        ),
        "registry_content_sha256": registry_content_sha256,
        "row_count": row_count,
        "columns": [
            config.text_column,
            config.treatment_column,
            config.outcome_column,
        ],
        "files": files,
        "embedding_cache": private_cache,
        "nonfit_text_supplied": False,
        "nonfit_labels_supplied": False,
        "global_cache_path_supplied": False,
        "source_dataset_path_supplied": False,
    }
    _write_json(
        root / PREFLIGHT_SCOPE_INPUT_MANIFEST,
        {**body, "content_sha256": _sha256_json(body)},
    )


def publish_preflight_scope_inputs(
    *,
    output_root: Path | str,
    modeling_data: pd.DataFrame,
    config: AppliedInferenceConfig,
    embedding_cache: Any,
    embedding_cache_identity: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_content_sha256: str,
    scopes: Sequence[Mapping[str, Any]],
    source_dataset_path: Path,
    global_embedding_cache_path: Path,
) -> AuthenticatedPreflightScopeInputSet:
    """Recoverably publish one fit-only capability per canonical scope."""

    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("preflight scope-input root must be absolute")
    canonical_scopes = tuple(json.loads(_canonical_json(dict(scope))) for scope in scopes)
    scope_ids = [str(scope.get("scope_id") or "") for scope in canonical_scopes]
    if (
        not scope_ids
        or any(not value for value in scope_ids)
        or len(scope_ids) != len(set(scope_ids))
    ):
        raise ValueError("preflight scope IDs must be unique and nonempty")
    if _sha256_json(registry) != str(registry_content_sha256):
        raise ValueError("preflight parent registry differs from its content identity")
    terminal_manifest = root / PREFLIGHT_SCOPE_INPUT_SET_MANIFEST
    if terminal_manifest.is_file():
        return validate_preflight_scope_input_set(
            root=root,
            expected_scopes=canonical_scopes,
            expected_registry_content_sha256=registry_content_sha256,
            parent_modeling_data=modeling_data,
            parent_config=config,
            parent_embedding_cache=embedding_cache,
            parent_embedding_cache_identity=embedding_cache_identity,
            forbidden_paths=(source_dataset_path, global_embedding_cache_path),
        )
    if root.is_symlink():
        raise ValueError("preflight scope-input root cannot be a symlink")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=True)
    if root.resolve(strict=True) != root:
        raise ValueError("preflight scope-input root is not canonical")
    allowed_entries = set(scope_ids)
    observed_entries = {entry.name for entry in os.scandir(root)}
    if not observed_entries.issubset(allowed_entries):
        raise ValueError("incomplete preflight scope-input root contains unknown entries")
    attempt_root = root.parent / f".{root.name}.scope_attempts"
    if attempt_root.is_symlink():
        raise ValueError("preflight scope-input attempt root cannot be a symlink")
    attempt_root.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for scope in canonical_scopes:
        scope_id = str(scope["scope_id"])
        scope_root = root / scope_id
        if scope_root.exists():
            completed = validate_preflight_scope_input(
                manifest_path=scope_root / PREFLIGHT_SCOPE_INPUT_MANIFEST,
                expected_scope_id=scope_id,
                expected_registry_content_sha256=registry_content_sha256,
                parent_modeling_data=modeling_data,
                parent_config=config,
                parent_embedding_cache=embedding_cache,
                parent_embedding_cache_identity=embedding_cache_identity,
                forbidden_paths=(
                    source_dataset_path,
                    global_embedding_cache_path,
                ),
            )
            if completed.scope != scope:
                raise ValueError("completed preflight scope input belongs to another scope")
        else:
            attempt = Path(
                tempfile.mkdtemp(
                    prefix=f"{scope_id}.attempt-",
                    dir=attempt_root,
                )
            )
            temporary = attempt / "scope_input"
            _write_scope(
                root=temporary,
                modeling_data=modeling_data,
                config=config,
                embedding_cache=embedding_cache,
                embedding_cache_identity=embedding_cache_identity,
                registry_content_sha256=registry_content_sha256,
                scope=scope,
                forbidden_paths=(source_dataset_path, global_embedding_cache_path),
            )
            completed = validate_preflight_scope_input(
                manifest_path=temporary / PREFLIGHT_SCOPE_INPUT_MANIFEST,
                expected_scope_id=scope_id,
                expected_registry_content_sha256=registry_content_sha256,
                parent_modeling_data=modeling_data,
                parent_config=config,
                parent_embedding_cache=embedding_cache,
                parent_embedding_cache_identity=embedding_cache_identity,
                forbidden_paths=(
                    source_dataset_path,
                    global_embedding_cache_path,
                ),
            )
            if completed.scope != scope:
                raise ValueError("new preflight scope input belongs to another scope")
            os.replace(temporary, scope_root)
            attempt.rmdir()
            descriptor = os.open(
                root,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        rows.append(
            {
                "scope_id": scope_id,
                "manifest": _file_registration(
                    scope_root / PREFLIGHT_SCOPE_INPUT_MANIFEST,
                    root,
                ),
            }
        )
    body = {
        "schema_version": PREFLIGHT_SCOPE_INPUT_SET_SCHEMA,
        "registry_content_sha256": registry_content_sha256,
        "scope_order": scope_ids,
        "scope_count": len(scope_ids),
        "scopes": rows,
        "one_scope_per_worker_payload": True,
    }
    _write_json(
        terminal_manifest,
        {**body, "content_sha256": _sha256_json(body)},
    )
    descriptor = os.open(
        root.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return validate_preflight_scope_input_set(
        root=root,
        expected_scopes=canonical_scopes,
        expected_registry_content_sha256=registry_content_sha256,
        parent_modeling_data=modeling_data,
        parent_config=config,
        parent_embedding_cache=embedding_cache,
        parent_embedding_cache_identity=embedding_cache_identity,
        forbidden_paths=(source_dataset_path, global_embedding_cache_path),
    )


def validate_preflight_scope_input(
    *,
    manifest_path: Path | str,
    expected_scope_id: str,
    expected_manifest_content_sha256: str | None = None,
    expected_registry_content_sha256: str | None = None,
    parent_modeling_data: pd.DataFrame | None = None,
    parent_config: AppliedInferenceConfig | None = None,
    parent_embedding_cache: Any | None = None,
    parent_embedding_cache_identity: Mapping[str, Any] | None = None,
    forbidden_paths: Sequence[Path] = (),
) -> AuthenticatedPreflightScopeInput:
    path = Path(manifest_path).absolute()
    root = path.parent
    if (
        path.name != PREFLIGHT_SCOPE_INPUT_MANIFEST
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
    ):
        raise ValueError("preflight scope-input manifest path is invalid")
    manifest = _read_json(path, label="preflight scope-input manifest")
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    required = {
        "schema_version",
        "scope",
        "scope_binding_sha256",
        "registry_content_sha256",
        "row_count",
        "columns",
        "files",
        "embedding_cache",
        "nonfit_text_supplied",
        "nonfit_labels_supplied",
        "global_cache_path_supplied",
        "source_dataset_path_supplied",
        "content_sha256",
    }
    scope = manifest.get("scope")
    if (
        set(manifest) != required
        or manifest.get("schema_version") != PREFLIGHT_SCOPE_INPUT_SCHEMA
        or not isinstance(scope, Mapping)
        or scope.get("scope_id") != expected_scope_id
        or manifest.get("content_sha256") != _sha256_json(body)
        or (
            expected_manifest_content_sha256 is not None
            and manifest.get("content_sha256") != expected_manifest_content_sha256
        )
        or manifest.get("nonfit_text_supplied") is not False
        or manifest.get("nonfit_labels_supplied") is not False
        or manifest.get("global_cache_path_supplied") is not False
        or manifest.get("source_dataset_path_supplied") is not False
    ):
        raise ValueError("preflight scope-input manifest is invalid")
    _require_sha256(
        manifest.get("content_sha256"),
        label="preflight scope-input content_sha256",
    )
    registry_sha = _require_sha256(
        manifest.get("registry_content_sha256"),
        label="preflight scope-input registry SHA",
    )
    if (
        expected_registry_content_sha256 is not None
        and registry_sha != expected_registry_content_sha256
    ):
        raise ValueError("preflight scope-input registry changed")
    if manifest.get("scope_binding_sha256") != _sha256_json(
        {"registry_content_sha256": registry_sha, "scope": scope}
    ):
        raise ValueError("preflight scope-input binding changed")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "effective_config",
        "one_scope_authority",
        "fit_only_modeling",
    }:
        raise ValueError("preflight scope-input files are incomplete")
    paths = {
        key: _validate_registration(root, registration, label=key)
        for key, registration in files.items()
    }
    columns = manifest.get("columns")
    if (
        not isinstance(columns, list)
        or len(columns) != 3
        or any(not isinstance(value, str) or not value for value in columns)
        or len(set(columns)) != 3
    ):
        raise ValueError("preflight scope-input columns are invalid")
    config_payload = _read_json(paths["effective_config"], label="preflight config")
    config = ExperimentConfig.from_dict({"applied_inference": config_payload}).applied_inference
    raw_embedding = (
        (config_payload.get("architecture") or {}).get("multi_model_forest") or {}
    ).get("embedding_contrast")
    if not isinstance(raw_embedding, Mapping):
        raise ValueError("preflight scope-input config lacks its embedding configuration")
    # The production wrapper already validated this effective configuration.
    # Restore its exact embedding block after the legacy config constructor's
    # compatibility normalization, which can otherwise disable it.
    restored_embedding = EmbeddingContrastDiscoveryConfig(**raw_embedding)
    config.architecture.multi_model_forest.embedding_contrast = restored_embedding
    config.architecture.multi_model_agentic_forest.embedding_contrast = copy.deepcopy(
        restored_embedding
    )
    if columns != [
        config.text_column,
        config.treatment_column,
        config.outcome_column,
    ]:
        raise ValueError("preflight scope-input config columns changed")
    authority = _read_json(
        paths["one_scope_authority"],
        label="preflight one-scope authority",
    )
    authority_body = {
        key: copy.deepcopy(value) for key, value in authority.items() if key != "content_sha256"
    }
    authority_fields = {
        "schema_version",
        "registry_content_sha256",
        "dataset_row_count",
        "scope",
        "scope_binding_sha256",
        "authorized_scope_count",
        "other_scope_definitions_supplied",
        "other_scope_row_identities_supplied",
        "content_sha256",
    }
    if (
        set(authority) != authority_fields
        or authority.get("schema_version") != PREFLIGHT_ONE_SCOPE_AUTHORITY_SCHEMA
        or authority.get("registry_content_sha256") != registry_sha
        or authority.get("scope") != scope
        or authority.get("scope_binding_sha256") != manifest.get("scope_binding_sha256")
        or authority.get("authorized_scope_count") != 1
        or authority.get("other_scope_definitions_supplied") is not False
        or authority.get("other_scope_row_identities_supplied") is not False
        or authority.get("content_sha256") != _sha256_json(authority_body)
    ):
        raise ValueError("preflight one-scope authority changed")
    modeling = _read_exact_parquet(
        paths["fit_only_modeling"],
        expected_columns=columns,
        label="preflight fit-only modeling data",
    )
    row_count = int(manifest["row_count"])
    if authority.get("dataset_row_count") != row_count:
        raise ValueError("preflight one-scope authority row count changed")
    fit_rows = tuple(map(int, scope.get("fit_row_ids") or ()))
    if len(modeling) != row_count or not fit_rows:
        raise ValueError("preflight scope-input row coverage changed")
    nonfit = sorted(set(range(row_count)) - set(fit_rows))
    if (
        not bool(
            modeling.iloc[list(fit_rows)][config.text_column]
            .map(lambda value: isinstance(value, str) and bool(value))
            .all()
        )
        or modeling.iloc[list(fit_rows)][[config.treatment_column, config.outcome_column]]
        .isna()
        .any()
        .any()
        or modeling.iloc[nonfit][config.text_column].map(bool).any()
        or modeling.iloc[nonfit][[config.treatment_column, config.outcome_column]]
        .notna()
        .any()
        .any()
    ):
        raise ValueError("preflight scope-input contains nonfit data or missing fit data")
    cache_registration = manifest.get("embedding_cache")
    if not isinstance(cache_registration, Mapping):
        raise ValueError("preflight scope-input lacks a private cache")
    cache_files = cache_registration.get("files")
    if not isinstance(cache_files, Mapping):
        raise ValueError("preflight scope-input cache files are malformed")
    for filename, registration in cache_files.items():
        cache_file = _validate_registration(
            root,
            registration,
            label=f"preflight private cache {filename}",
        )
        if (
            registration["relative_path"]
            != (Path(str(cache_registration["relative_path"])) / str(filename)).as_posix()
        ):
            raise ValueError("preflight private cache layout changed")
    cache = _RestrictedLogicalIdentityEmbeddingCache(
        cache_dir=root / str(cache_registration["relative_path"]),
        logical_identity=cache_registration["logical_identity"],
        allowed_row_ids=fit_rows,
    )
    if (
        cache.physical_identity() != cache_registration["physical_identity"]
        or cache.identity() != cache_registration["logical_identity"]
        or cache.row_count != row_count
    ):
        raise ValueError("preflight private cache changed")
    for row_id in nonfit:
        if int(cache._offsets[row_id]) != int(
            cache._offsets[row_id + 1]
        ) or cache._cache._cached_chunks(row_id):
            raise ValueError("preflight private cache retained a nonfit row")
    expected_files = {
        PREFLIGHT_SCOPE_INPUT_MANIFEST,
        *(str(value["relative_path"]) for value in files.values()),
        *(str(value["relative_path"]) for value in cache_registration["files"].values()),
    }
    observed_files, observed_directories = _closed_tree_inventory(
        root,
        label="preflight scope input",
    )
    expected_directories = {
        Path(value).parent.as_posix()
        for value in expected_files
        if Path(value).parent.as_posix() != "."
    }
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("preflight scope input contains unregistered entries")
    if forbidden_paths:
        serialized = b"".join(
            (root / relative).read_bytes()
            for relative in sorted(observed_files)
            if not relative.endswith((".npy", ".parquet"))
        )
        for forbidden in forbidden_paths:
            if str(forbidden.resolve(strict=False)).encode("utf-8") in serialized:
                raise ValueError("preflight scope input exposes a forbidden path")
    if parent_modeling_data is not None:
        if parent_config is None:
            raise ValueError("parent config is required with parent modeling data")
        expected = parent_modeling_data.iloc[list(fit_rows)][columns]
        actual = modeling.iloc[list(fit_rows)][columns]
        if actual.to_dict("records") != expected.to_dict("records"):
            raise ValueError("preflight scope input differs from parent fit rows")
    if parent_embedding_cache_identity is not None:
        if cache_registration["logical_identity"] != dict(parent_embedding_cache_identity):
            raise ValueError("preflight private cache logical identity changed")
    if parent_embedding_cache is not None:
        for row_id in fit_rows:
            if cache._cache._cached_chunks(row_id) != parent_embedding_cache._cached_chunks(row_id):
                raise ValueError("preflight private cache text differs from parent")
            private_start = int(cache._offsets[row_id])
            private_stop = int(cache._offsets[row_id + 1])
            parent_start = int(parent_embedding_cache._offsets[row_id])
            parent_stop = int(parent_embedding_cache._offsets[row_id + 1])
            if not np.array_equal(
                cache._embeddings[private_start:private_stop],
                parent_embedding_cache._embeddings[parent_start:parent_stop],
            ):
                raise ValueError("preflight private cache embeddings differ from parent")
    return AuthenticatedPreflightScopeInput(
        root=root,
        manifest=copy.deepcopy(manifest),
        modeling_data=modeling,
        config=config,
        scope_authority=authority,
        scope=copy.deepcopy(dict(scope)),
        embedding_cache=cache,
    )


def validate_preflight_scope_input_set(
    *,
    root: Path | str,
    expected_scopes: Sequence[Mapping[str, Any]],
    expected_registry_content_sha256: str,
    parent_modeling_data: pd.DataFrame | None = None,
    parent_config: AppliedInferenceConfig | None = None,
    parent_embedding_cache: Any | None = None,
    parent_embedding_cache_identity: Mapping[str, Any] | None = None,
    forbidden_paths: Sequence[Path] = (),
) -> AuthenticatedPreflightScopeInputSet:
    set_root = Path(root).absolute()
    if set_root.is_symlink() or not set_root.is_dir() or set_root.resolve(strict=True) != set_root:
        raise ValueError("preflight scope-input set root is invalid")
    manifest = _read_json(
        set_root / PREFLIGHT_SCOPE_INPUT_SET_MANIFEST,
        label="preflight scope-input set manifest",
    )
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    required = {
        "schema_version",
        "registry_content_sha256",
        "scope_order",
        "scope_count",
        "scopes",
        "one_scope_per_worker_payload",
        "content_sha256",
    }
    expected = tuple(json.loads(_canonical_json(dict(scope))) for scope in expected_scopes)
    expected_order = [str(scope["scope_id"]) for scope in expected]
    rows = manifest.get("scopes")
    if (
        set(manifest) != required
        or manifest.get("schema_version") != PREFLIGHT_SCOPE_INPUT_SET_SCHEMA
        or manifest.get("registry_content_sha256") != expected_registry_content_sha256
        or manifest.get("scope_order") != expected_order
        or manifest.get("scope_count") != len(expected)
        or manifest.get("one_scope_per_worker_payload") is not True
        or manifest.get("content_sha256") != _sha256_json(body)
        or not isinstance(rows, list)
        or len(rows) != len(expected)
    ):
        raise ValueError("preflight scope-input set manifest is invalid")
    authenticated: dict[str, AuthenticatedPreflightScopeInput] = {}
    for scope, row in zip(expected, rows):
        scope_id = str(scope["scope_id"])
        if (
            not isinstance(row, Mapping)
            or set(row) != {"scope_id", "manifest"}
            or row.get("scope_id") != scope_id
        ):
            raise ValueError("preflight scope-input set row changed")
        child_manifest = _validate_registration(
            set_root,
            row["manifest"],
            label=f"{scope_id} preflight manifest",
        )
        child = validate_preflight_scope_input(
            manifest_path=child_manifest,
            expected_scope_id=scope_id,
            expected_registry_content_sha256=expected_registry_content_sha256,
            parent_modeling_data=parent_modeling_data,
            parent_config=parent_config,
            parent_embedding_cache=parent_embedding_cache,
            parent_embedding_cache_identity=parent_embedding_cache_identity,
            forbidden_paths=forbidden_paths,
        )
        if child.scope != scope:
            raise ValueError("preflight scope-input set scope changed")
        authenticated[scope_id] = child
    expected_files = {PREFLIGHT_SCOPE_INPUT_SET_MANIFEST}
    expected_directories: set[str] = set()
    for scope_id, child in authenticated.items():
        child_files, child_directories = _closed_tree_inventory(
            child.root,
            label=f"{scope_id} preflight scope input",
        )
        expected_directories.add(scope_id)
        expected_files.update(f"{scope_id}/{relative}" for relative in child_files)
        expected_directories.update(f"{scope_id}/{relative}" for relative in child_directories)
    observed_files, observed_directories = _closed_tree_inventory(
        set_root,
        label="preflight scope-input set",
    )
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("preflight scope-input set contains unregistered entries")
    return AuthenticatedPreflightScopeInputSet(
        root=set_root,
        manifest=copy.deepcopy(manifest),
        scopes=authenticated,
    )


__all__ = [
    "AuthenticatedPreflightScopeInput",
    "AuthenticatedPreflightScopeInputSet",
    "PREFLIGHT_ONE_SCOPE_AUTHORITY_SCHEMA",
    "PREFLIGHT_SCOPE_INPUT_MANIFEST",
    "PREFLIGHT_SCOPE_INPUT_SET_MANIFEST",
    "publish_preflight_scope_inputs",
    "validate_preflight_scope_input",
    "validate_preflight_scope_input_set",
]
