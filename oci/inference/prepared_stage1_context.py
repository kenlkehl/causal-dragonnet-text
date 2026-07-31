"""Immutable, relocatable authority for a reusable prepared Stage 1 context.

The expensive Stage 1 preparation step authenticates the cohort, embedding
cache, clustered-preflight state, model tree, split registry, and scientific
configuration.  Physical-scope workers must not repeat that work for every
owner.  This module seals one prepared context as two deliberately separate
payloads:

* ``scientific_identity.json`` is a path-neutral compatibility projection;
* ``execution_locators.json`` is the exact, machine-local capability needed
  to reopen the authenticated inputs.

The manifest contains only relative payload names and content hashes, so the
context directory itself may be relocated byte-for-byte.  A spawned worker
freshly authenticates the complete artifact and reconstructs the prepared
context once.  Reconstruction must reproduce both the exact request hash and
the path-neutral scientific identity before any owner is executed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping


PREPARED_STAGE1_CONTEXT_ARTIFACT_SCHEMA = (
    "portable_prepared_stage1_context_artifact_v2"
)
PREPARED_STAGE1_CONTEXT_MANIFEST_SCHEMA = (
    "portable_prepared_stage1_context_manifest_v2"
)
PREPARED_STAGE1_CONTEXT_SCIENTIFIC_SCHEMA = (
    "portable_prepared_stage1_context_scientific_identity_v2"
)
PREPARED_STAGE1_CONTEXT_LOCATOR_SCHEMA = (
    "portable_prepared_stage1_context_execution_locators_v2"
)
PREPARED_STAGE1_CONTEXT_MANIFEST_NAME = "prepared_stage1_context_manifest.json"
PREPARED_STAGE1_CONTEXT_SCIENTIFIC_NAME = "scientific_identity.json"
PREPARED_STAGE1_CONTEXT_LOCATOR_NAME = "execution_locators.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_NAMES = (
    PREPARED_STAGE1_CONTEXT_SCIENTIFIC_NAME,
    PREPARED_STAGE1_CONTEXT_LOCATOR_NAME,
)
_READ_ONLY_FILE_MODE = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
_READ_ONLY_DIRECTORY_MODE = (
    stat.S_IRUSR
    | stat.S_IXUSR
    | stat.S_IRGRP
    | stat.S_IXGRP
    | stat.S_IROTH
    | stat.S_IXOTH
)
_PATH_OPTION_FIELDS = frozenset(
    {
        "dataset_path",
        "config_path",
        "embedding_cache_dir",
        "output_dir",
        "embedding_local_model_path",
        "embedding_cache_output_dir",
        "embedding_cache_relocation_prepublication_root",
        "embedding_cache_validation_dataset_path",
        "query_config_path",
        "cluster_preflight_manifest_path",
        "cluster_preflight_state_bundle_manifest_path",
        "reusable_preflight_import_manifest_path",
        "reusable_preflight_import_state_bundle_manifest_path",
        "reusable_preflight_store_root",
        "stage1_scope_descriptor_root",
        "stage1_scope_attempt_root",
        "stage1_scope_progress_path",
    }
)
_TUPLE_OPTION_FIELDS = frozenset({"gpu_ids", "query_devices"})
_RELOCATION_PATH_FIELDS = frozenset(
    {
        "source_cache_dir",
        "source_prepared_cohort_path",
        "source_preparation_manifest_path",
        "fresh_prepared_cohort_path",
        "fresh_preparation_manifest_path",
        "local_model_path",
        "target_dir",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TypeError(f"{label} must be closed finite JSON") from exc


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _option_mapping(prepared: Any) -> dict[str, Any]:
    """Serialize every build-option field without invoking a default."""

    from .production_embedding_cache_relocation import (
        ProductionEmbeddingCacheRelocationOptions,
    )
    from .production_stage1_bundle import Stage1BundleBuildOptions

    options = prepared.options
    if not isinstance(options, Stage1BundleBuildOptions):
        raise TypeError(
            "process authority requires typed Stage1BundleBuildOptions"
        )
    expected = {value.name for value in fields(Stage1BundleBuildOptions)}
    result: dict[str, Any] = {}
    for value in fields(Stage1BundleBuildOptions):
        name = value.name
        raw = getattr(options, name)
        if name in _PATH_OPTION_FIELDS:
            result[name] = None if raw is None else str(Path(raw))
        elif name in _TUPLE_OPTION_FIELDS:
            result[name] = list(raw)
        elif name == "embedding_cache_relocation":
            if raw is None:
                result[name] = None
            else:
                if not isinstance(
                    raw,
                    ProductionEmbeddingCacheRelocationOptions,
                ):
                    raise TypeError(
                        "process authority cache relocation must be typed"
                    )
                relocation: dict[str, Any] = {}
                for child in fields(
                    ProductionEmbeddingCacheRelocationOptions
                ):
                    child_value = getattr(raw, child.name)
                    relocation[child.name] = (
                        str(Path(child_value))
                        if child.name in _RELOCATION_PATH_FIELDS
                        else _json_copy(
                            child_value,
                            label=f"relocation.{child.name}",
                        )
                    )
                result[name] = relocation
        elif name == "semantic_witness_scientific_config":
            if raw is None:
                result[name] = None
            elif isinstance(raw, Mapping):
                result[name] = _json_copy(raw, label=name)
            else:
                as_dict = getattr(raw, "as_dict", None)
                if not callable(as_dict):
                    raise TypeError(
                        "semantic witness scientific config lacks as_dict()"
                    )
                result[name] = _json_copy(as_dict(), label=name)
        elif name == "physical_fit_identity":
            from .production_stage1_scope_scheduler import (
                Stage1PhysicalFitIdentity,
            )

            if not isinstance(raw, Stage1PhysicalFitIdentity):
                raise TypeError(
                    "physical_fit_identity must be one closed typed identity"
                )
            result[name] = _json_copy(raw.as_dict(), label=name)
        else:
            result[name] = _json_copy(raw, label=name)
    if set(result) != expected:
        raise RuntimeError("process authority omitted a Stage 1 build option")
    return result


def _options_from_mapping(value: Mapping[str, Any]) -> Any:
    """Reconstruct typed build options from an exact closed field mapping."""

    from .production_embedding_cache_relocation import (
        ProductionEmbeddingCacheRelocationOptions,
    )
    from .production_stage1_bundle import Stage1BundleBuildOptions

    expected = {child.name for child in fields(Stage1BundleBuildOptions)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(
            "process authority must contain every Stage1BundleBuildOptions "
            "field exactly"
        )
    kwargs: dict[str, Any] = {}
    for child in fields(Stage1BundleBuildOptions):
        name = child.name
        raw = copy.deepcopy(value[name])
        if name in _PATH_OPTION_FIELDS:
            kwargs[name] = None if raw is None else Path(str(raw))
        elif name in _TUPLE_OPTION_FIELDS:
            if not isinstance(raw, list):
                raise TypeError(
                    f"process authority option {name} must be a list"
                )
            kwargs[name] = tuple(raw)
        elif name == "embedding_cache_relocation":
            if raw is None:
                kwargs[name] = None
            else:
                relocation_fields = {
                    value.name
                    for value in fields(
                        ProductionEmbeddingCacheRelocationOptions
                    )
                }
                if (
                    not isinstance(raw, Mapping)
                    or set(raw) != relocation_fields
                ):
                    raise ValueError(
                        "process authority relocation fields are incomplete"
                    )
                relocation = {
                    key: (
                        Path(str(raw[key]))
                        if key in _RELOCATION_PATH_FIELDS
                        else copy.deepcopy(raw[key])
                    )
                    for key in relocation_fields
                }
                kwargs[name] = ProductionEmbeddingCacheRelocationOptions(
                    **relocation
                )
        else:
            kwargs[name] = raw
    # Every field is supplied. No constructor default participates.
    return Stage1BundleBuildOptions(**kwargs)


def _private_regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        value = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(value.st_mode)
        or not stat.S_ISREG(value.st_mode)
        or int(value.st_nlink) != 1
        or stat.S_IMODE(value.st_mode) != _READ_ONLY_FILE_MODE
    ):
        raise ValueError(
            f"{label} must be one private read-only regular file"
        )
    return value


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


def _input_stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_stable_bytes(path: Path, *, label: str) -> bytes:
    before = _private_regular_file(path, label=label)
    payload = path.read_bytes()
    after = _private_regular_file(path, label=label)
    if _stat_identity(before) != _stat_identity(after) or len(payload) != int(
        before.st_size
    ):
        raise RuntimeError(f"{label} changed while it was being read")
    return payload


def _parse_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("prepared-context write ended unexpectedly")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _payload_inventory(
    *,
    scientific_bytes: bytes,
    locator_bytes: bytes,
) -> list[dict[str, Any]]:
    by_name = {
        PREPARED_STAGE1_CONTEXT_SCIENTIFIC_NAME: scientific_bytes,
        PREPARED_STAGE1_CONTEXT_LOCATOR_NAME: locator_bytes,
    }
    return [
        {
            "relative_path": name,
            "size_bytes": len(by_name[name]),
            "sha256": _sha256_bytes(by_name[name]),
        }
        for name in _PAYLOAD_NAMES
    ]


def _validate_inventory(
    root: Path,
    rows: Any,
) -> dict[str, bytes]:
    if (
        not isinstance(rows, list)
        or len(rows) != len(_PAYLOAD_NAMES)
        or [row.get("relative_path") for row in rows if isinstance(row, Mapping)]
        != list(_PAYLOAD_NAMES)
    ):
        raise ValueError("prepared-context payload inventory is not exact and ordered")
    observed_names = {
        child.name
        for child in root.iterdir()
        if child.name != PREPARED_STAGE1_CONTEXT_MANIFEST_NAME
    }
    if observed_names != set(_PAYLOAD_NAMES):
        raise ValueError("prepared-context directory has missing or extra payloads")
    result: dict[str, bytes] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("prepared-context payload inventory row is malformed")
        name = str(row["relative_path"])
        size = int(row["size_bytes"])
        digest = _require_sha256(
            row["sha256"],
            label=f"prepared-context payload {name}",
        )
        if size < 1 or name not in _PAYLOAD_NAMES:
            raise ValueError("prepared-context payload inventory row is invalid")
        payload = _read_stable_bytes(
            root / name,
            label=f"prepared-context payload {name}",
        )
        if len(payload) != size or _sha256_bytes(payload) != digest:
            raise ValueError(f"prepared-context payload changed: {name}")
        result[name] = payload
    return result


def _scientific_payload(
    request: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    registry_content_sha256: str,
    architecture_profiles: Mapping[str, Mapping[str, Any]],
    runtime_compatibility_class: str,
) -> dict[str, Any]:
    from .production_stage1_cluster_preflight_artifact import (
        stage1_request_scientific_compatibility_projection,
    )

    projection = stage1_request_scientific_compatibility_projection(request)
    profiles = _json_copy(
        architecture_profiles,
        label="prepared-context scientific architecture profiles",
    )
    runtime = str(runtime_compatibility_class).strip()
    if not runtime:
        raise ValueError(
            "prepared-context scientific runtime compatibility class "
            "must be nonempty"
        )
    body = {
        "schema_version": PREPARED_STAGE1_CONTEXT_SCIENTIFIC_SCHEMA,
        "stage1_request_scientific_projection": projection,
        "stage1_request_scientific_compatibility_sha256": _require_sha256(
            projection.get("content_sha256"),
            label="prepared Stage 1 scientific compatibility",
        ),
        "split_registry": _json_copy(
            registry,
            label="prepared Stage 1 split registry",
        ),
        "split_registry_content_sha256": _require_sha256(
            registry_content_sha256,
            label="prepared Stage 1 split registry",
        ),
        "architecture_profiles": profiles,
        "architecture_profiles_content_sha256": _sha256_json(
            profiles
        ),
        "runtime_compatibility_class": runtime,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _locator_payload(
    *,
    stage1_build_options: Mapping[str, Any],
    architecture_profiles: Mapping[str, Mapping[str, Any]],
    runtime_compatibility_class: str,
    scientific_compatibility_sha256: str,
    exact_stage1_request: Mapping[str, Any],
) -> dict[str, Any]:
    options = _json_copy(
        stage1_build_options,
        label="prepared-context Stage 1 build options",
    )
    _options_from_mapping(options)
    profiles = _json_copy(
        architecture_profiles,
        label="prepared-context architecture profiles",
    )
    runtime = str(runtime_compatibility_class).strip()
    if not runtime:
        raise ValueError(
            "prepared-context runtime compatibility class must be nonempty"
        )
    request = _json_copy(
        exact_stage1_request,
        label="prepared-context exact Stage 1 request",
    )
    request_sha = _require_sha256(
        request.get("request_sha256"),
        label="prepared-context exact Stage 1 request",
    )
    request_body = {
        key: copy.deepcopy(value)
        for key, value in request.items()
        if key != "request_sha256"
    }
    if request_sha != _sha256_json(request_body):
        raise ValueError(
            "prepared-context exact Stage 1 request hash changed"
        )
    request_dataset = request.get("dataset")
    request_source = request.get("source_config")
    request_cache = request.get("embedding_cache")
    if all(
        isinstance(value, Mapping)
        for value in (request_dataset, request_source, request_cache)
    ):
        expected_locator_paths = (
            str(Path(str(options["dataset_path"])).resolve()),
            str(Path(str(options["config_path"])).resolve()),
            str(Path(str(options["embedding_cache_dir"])).resolve()),
        )
        observed_locator_paths = (
            str(Path(str(request_dataset.get("path", ""))).resolve()),
            str(Path(str(request_source.get("path", ""))).resolve()),
            str(Path(str(request_cache.get("path", ""))).resolve()),
        )
        if observed_locator_paths != expected_locator_paths:
            raise ValueError(
                "prepared-context exact request and execution locators differ"
            )
    from .production_stage1_cluster_preflight_artifact import (
        stage1_request_scientific_compatibility_projection,
    )

    request_projection = (
        stage1_request_scientific_compatibility_projection(request)
    )
    bound_scientific = _require_sha256(
        scientific_compatibility_sha256,
        label="prepared-context locator scientific binding",
    )
    if request_projection.get("content_sha256") != bound_scientific:
        raise ValueError(
            "prepared-context exact request differs from scientific binding"
        )
    body = {
        "schema_version": PREPARED_STAGE1_CONTEXT_LOCATOR_SCHEMA,
        "stage1_build_options": options,
        "architecture_profiles": profiles,
        "runtime_compatibility_class": runtime,
        "bound_scientific_compatibility_sha256": bound_scientific,
        "exact_stage1_request": request,
    }
    return {**body, "content_sha256": _sha256_json(body)}


@dataclass(frozen=True)
class PreparedStage1ContextArtifact:
    """Freshly authenticated prepared-context capability."""

    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    scientific_identity: Mapping[str, Any]
    execution_locators: Mapping[str, Any]

    @property
    def scientific_compatibility_sha256(self) -> str:
        return str(
            self.scientific_identity[
                "stage1_request_scientific_compatibility_sha256"
            ]
        )

    @property
    def content_root_sha256(self) -> str:
        return str(self.manifest["content_root_sha256"])

    def reconstruct(
        self,
        *,
        slot_cpu_budget: int | None = None,
        ordinary_full_byte_cache_fallback: bool = False,
        absent_htr_model_path_rebinding: Path | str | None = None,
    ) -> tuple[Any, Any]:
        """Rehydrate the sealed state without rerunning monolithic prepare().

        The two optional recovery controls are deliberately narrow and
        operational.  ``ordinary_full_byte_cache_fallback`` bypasses only an
        operator-trusted stat-continuity shortcut and instead reauthenticates
        every registered cache byte through the ordinary reader.
        ``absent_htr_model_path_rebinding`` may replace a legacy local HTR
        locator only when that recorded locator no longer exists.  The
        replacement is accepted only after the existing path-neutral
        effective-configuration projection and sealed model-tree digest both
        reproduce exactly.
        """

        import dataclasses

        import pandas as pd

        from .neural_query_agentic_forest import (
            NeuralQueryAgenticForestConfig,
        )
        from .production_role_neutral_producer_factories import (
            PreparedBuildRoleNeutralProducerFactoriesBuilder,
        )
        from .production_stage1_bundle import (
            ProductionStage1BundleBuilder,
            STAGE1_REUSABLE_ASSEMBLED_PREFLIGHT_PRODUCER_IDENTITY,
            SpentOnlyFrozenChunkEmbeddingCache,
            _PreparedBuild,
            _directory_tree_sha256,
            _read_stable_sha256,
            _validate_cache_configuration,
            _validate_effective_config,
            load_applied_stage1_config,
        )
        from .production_stage1_config_wire import (
            production_stage1_effective_config_payload,
        )
        from .production_stage1_scope_scheduler import (
            build_canonical_stage1_scope_plan,
        )
        from .operator_trusted_embedding_cache_reader import (
            OperatorTrustedSpentOnlyFrozenChunkEmbeddingCache,
            cache_build_identity_from_operator_trusted_proof,
            validate_operator_trusted_cache_read_proof,
        )
        from .review_spent_evidence_provider import (
            SemanticWitnessScientificConfig,
        )
        from .role_neutral_embedding_group_execution import (
            load_canonical_clustered_preflight_state_bundle,
        )

        locators = self.execution_locators
        exact_request = copy.deepcopy(
            dict(locators["exact_stage1_request"])
        )
        options = _options_from_mapping(locators["stage1_build_options"])
        if not isinstance(ordinary_full_byte_cache_fallback, bool):
            raise TypeError(
                "ordinary_full_byte_cache_fallback must be boolean"
            )
        if ordinary_full_byte_cache_fallback:
            options = dataclasses.replace(
                options,
                embedding_cache_operator_trusted_read_proof=None,
            )
        if slot_cpu_budget is not None:
            budget = int(slot_cpu_budget)
            if isinstance(slot_cpu_budget, bool) or budget < 1:
                raise ValueError(
                    "prepared-context slot CPU budget must be positive"
                )
            options = dataclasses.replace(
                options,
                num_workers=budget,
                tfidf_workers=budget,
            )
        projection = self.scientific_identity[
            "stage1_request_scientific_projection"
        ]
        dataset_identity = projection["dataset"]
        source_config_identity = projection["source_config"]
        exact_cache_identity = exact_request["embedding_cache"]["identity"]
        htr_identity = projection["htr_model"]

        dataset_path = Path(options.dataset_path).resolve(strict=True)
        config_path = Path(options.config_path).resolve(strict=True)
        cache_path = Path(options.embedding_cache_dir).resolve(strict=True)
        dataset_sha, dataset_stat = _read_stable_sha256(dataset_path)
        config_sha, config_stat = _read_stable_sha256(config_path)
        if (
            dataset_sha != dataset_identity["sha256"]
            or config_sha != source_config_identity["sha256"]
        ):
            raise ValueError(
                "prepared-context locator bytes differ from scientific identity"
            )

        source_config = load_applied_stage1_config(
            config_path,
            require_explicit_scientific_fields=True,
        )
        if absent_htr_model_path_rebinding is not None:
            supplied_htr_path = Path(
                absent_htr_model_path_rebinding
            )
            if not supplied_htr_path.is_absolute():
                raise ValueError(
                    "HTR model-path rebinding must be absolute"
                )
            if supplied_htr_path.is_symlink():
                raise ValueError(
                    "HTR model-path rebinding must name a real directory"
                )
            supplied_htr_path = supplied_htr_path.resolve(strict=True)
            if not supplied_htr_path.is_dir():
                raise ValueError(
                    "HTR model-path rebinding must name a directory"
                )
            recorded_htr_path = Path(
                str(source_config.architecture.htr_sentence_model)
            ).expanduser()
            if not recorded_htr_path.is_absolute():
                recorded_htr_path = config_path.parent / recorded_htr_path
            if recorded_htr_path.exists() or recorded_htr_path.is_symlink():
                if (
                    recorded_htr_path.resolve(strict=True)
                    != supplied_htr_path
                ):
                    raise ValueError(
                        "HTR model-path rebinding is permitted only when "
                        "the sealed legacy locator is absent"
                    )
            else:
                source_config.architecture.htr_sentence_model = str(
                    supplied_htr_path
                )
        config, htr_model_path = _validate_effective_config(
            source_config,
            dataset_path=dataset_path,
            embedding_cache_dir=cache_path,
            config_dir=config_path.parent,
            seed=int(options.seed),
        )
        if _input_stat_identity(config_path) != config_stat:
            raise RuntimeError(
                "prepared-context source config changed while it was opened"
            )
        # Parsing the profile is narrow and deterministic. It may not silently
        # change any non-locator effective configuration.
        from .production_stage1_cluster_preflight_artifact import (
            stage1_effective_config_scientific_compatibility_projection,
        )

        effective_payload = (
            stage1_effective_config_scientific_compatibility_projection(
                production_stage1_effective_config_payload(config)
            )
        )
        expected_effective = projection["effective_stage1_config"]
        if effective_payload != expected_effective:
            raise ValueError(
                "prepared-context effective scientific config changed"
            )

        projected_columns = tuple(dataset_identity["columns_read"])
        if (
            len(projected_columns) != 4
            or options.unit_id_column not in projected_columns
            or config.text_column not in projected_columns
            or config.treatment_column not in projected_columns
            or config.outcome_column not in projected_columns
        ):
            raise ValueError(
                "prepared-context dataset projection columns changed"
            )
        data = pd.read_parquet(
            dataset_path,
            columns=list(projected_columns),
        ).reset_index(drop=True)
        if (
            _input_stat_identity(dataset_path) != dataset_stat
            or len(data) != int(dataset_identity["row_count"])
        ):
            raise RuntimeError(
                "prepared-context cohort changed while it was being opened"
            )
        unit_rows = [
            {"type": type(value).__name__, "value": repr(value)}
            for value in data[options.unit_id_column].tolist()
        ]
        if _sha256_json(unit_rows) != dataset_identity[
            "ordered_unit_id_sha256"
        ]:
            raise ValueError(
                "prepared-context ordered unit identity changed"
            )
        modeling_data = data[
            [
                config.text_column,
                config.treatment_column,
                config.outcome_column,
            ]
        ].copy()

        trusted_cache_read_proof = (
            options.embedding_cache_operator_trusted_read_proof
        )
        if trusted_cache_read_proof is None:
            embedding_cache = SpentOnlyFrozenChunkEmbeddingCache(cache_path)
        else:
            validated_trusted_proof = (
                validate_operator_trusted_cache_read_proof(
                    trusted_cache_read_proof,
                    cache_dir=cache_path,
                )
            )
            if (
                validated_trusted_proof[
                    "legacy_terminal_migration_identity"
                ]
                != options.embedding_cache_legacy_migration_identity
            ):
                raise ValueError(
                    "prepared-context operator-trusted cache proof differs "
                    "from its legacy migration identity"
                )
            embedding_cache = (
                OperatorTrustedSpentOnlyFrozenChunkEmbeddingCache(
                    cache_path,
                    proof=validated_trusted_proof,
                )
            )
        # The ordinary reader authenticates every registered byte.  The
        # explicit operator-trusted reader instead inherits the sealed digests
        # after exact stat-continuity checks.  Both expose the same scientific
        # provider identity through this nonserializable handle.
        cache_identity = copy.deepcopy(
            embedding_cache.authenticated_snapshot_identity()
        )
        if cache_identity != exact_cache_identity:
            raise ValueError(
                "prepared-context embedding cache identity changed"
            )
        _validate_cache_configuration(
            embedding_cache,
            config,
            cache_configuration=(
                None
                if options.embedding_cache_configuration is None
                else options.embedding_cache_configuration
            ),
            legacy_terminal_migration_identity=(
                options.embedding_cache_legacy_migration_identity
            ),
        )
        if trusted_cache_read_proof is not None:
            trusted_build_identity = (
                cache_build_identity_from_operator_trusted_proof(
                    trusted_cache_read_proof,
                    cache_dir=cache_path,
                )
            )
            if trusted_build_identity != exact_request[
                "embedding_cache"
            ]["production_cache_build_identity"]:
                raise ValueError(
                    "prepared-context operator-trusted cache build identity "
                    "differs from its exact request"
                )
        htr_sha256 = _directory_tree_sha256(htr_model_path)
        if htr_sha256 != htr_identity["tree_sha256"]:
            raise ValueError("prepared-context HTR model tree changed")

        registry = copy.deepcopy(
            self.scientific_identity["split_registry"]
        )
        registry_sha256 = self.scientific_identity[
            "split_registry_content_sha256"
        ]
        if (
            registry_sha256
            != projection["split_registry_content_sha256"]
        ):
            raise ValueError(
                "prepared-context split registry binding changed"
            )
        initial_partitions = int(options.initial_training_partitions)
        review_rounds = (
            int(
                config.architecture.multi_model_forest
                .candidate_consistency_inner_folds
            )
            - initial_partitions
        )
        plan = build_canonical_stage1_scope_plan(
            registry=registry,
            registry_content_sha256=registry_sha256,
            global_seed=int(options.seed),
            physical_fit_identity=options.physical_fit_identity,
            gpu_ids=(),
            review_rounds=review_rounds,
            initial_training_partitions=initial_partitions,
            scope_workers_per_gpu=1,
            expected_outer_fold_count=int(config.cv_folds),
            expected_inner_fold_count=int(
                config.architecture.multi_model_forest
                .candidate_consistency_inner_folds
            ),
        )
        if (
            plan.scientific_content_sha256
            != projection["stage1_scope_plan"][
                "scientific_content_sha256"
            ]
        ):
            raise ValueError(
                "prepared-context scientific scope plan changed"
            )

        query_effective = projection["query_config"]["effective"]
        query_config = NeuralQueryAgenticForestConfig(
            **copy.deepcopy(dict(query_effective))
        )
        query_config.validate()
        if options.query_config_path is None:
            raise ValueError(
                "prepared-context query-config locator is missing"
            )
        loaded_query, query_identity = (
            ProductionStage1BundleBuilder._load_query_config(
                options.query_config_path
            )
        )
        if dataclasses.asdict(loaded_query) != dataclasses.asdict(
            query_config
        ):
            raise ValueError(
                "prepared-context query config locator changed"
            )
        semantic_payload = projection[
            "semantic_witness_scientific_config"
        ]
        semantic = (
            None
            if semantic_payload is None
            else SemanticWitnessScientificConfig.from_mapping(
                semantic_payload
            )
        )
        if options.cluster_preflight_manifest_path is None:
            raise ValueError(
                "prepared-context clustered-preflight locator is missing"
            )
        reusable_preflight_selected = False
        if options.portable_cluster_preflight_v2:
            from .production_stage1_reusable_preflight import (
                is_reusable_preflight_reference,
                load_reusable_preflight_reference,
            )

            if is_reusable_preflight_reference(
                options.cluster_preflight_manifest_path
            ):
                reusable_preflight_selected = True
                preflight = load_reusable_preflight_reference(
                    manifest_path=(
                        options.cluster_preflight_manifest_path
                    ),
                    expected_stage1_request=exact_request,
                    plan=plan,
                    producer_identity=(
                        STAGE1_REUSABLE_ASSEMBLED_PREFLIGHT_PRODUCER_IDENTITY
                    ),
                )
            else:
                from .production_stage1_cluster_preflight_artifact_v2 import (
                    load_portable_production_stage1_cluster_preflight_artifact,
                )

                preflight = (
                    load_portable_production_stage1_cluster_preflight_artifact(
                        manifest_path=options.cluster_preflight_manifest_path,
                        config=config,
                        registry=registry,
                        registry_content_sha256=registry_sha256,
                        embedding_cache_identity=cache_identity,
                    )
                )
        else:
            from .production_stage1_cluster_preflight_artifact import (
                load_production_stage1_cluster_preflight_artifact,
            )

            preflight = load_production_stage1_cluster_preflight_artifact(
                manifest_path=options.cluster_preflight_manifest_path,
                config=config,
                registry=registry,
                registry_content_sha256=registry_sha256,
                embedding_cache_identity=cache_identity,
            )
        if options.cluster_preflight_state_bundle_manifest_path is None:
            raise ValueError(
                "prepared-context clustered-state locator is missing"
            )
        if reusable_preflight_selected:
            from .production_stage1_reusable_preflight import (
                load_reusable_state_bundle_reference,
            )

            state_bundle = load_reusable_state_bundle_reference(
                manifest_path=(
                    options.cluster_preflight_state_bundle_manifest_path
                ),
                preflight=preflight,
                plan=plan,
            )
        else:
            state_bundle = load_canonical_clustered_preflight_state_bundle(
                manifest_path=(
                    options.cluster_preflight_state_bundle_manifest_path
                ),
                preflight=preflight,
                plan=plan,
            )
        physical_ids = {scope.scope_id for scope in plan.physical_scopes}
        if set(state_bundle.states) != physical_ids:
            raise ValueError(
                "prepared-context clustered state omitted a physical owner"
            )
        cluster_identity = preflight.identity()
        cluster_audit = copy.deepcopy(dict(preflight.audit))
        prepared = _PreparedBuild(
            options=options,
            output_path=Path(options.output_dir).resolve(),
            data=data,
            modeling_data=modeling_data,
            config=config,
            htr_model_path=htr_model_path,
            htr_model_sha256=htr_sha256,
            htr_input_nontruncation_audit=copy.deepcopy(
                exact_request["htr_input_nontruncation_audit"]
            ),
            embedding_cluster_feasibility_audit=cluster_audit,
            cluster_preflight_canonical_scope_states=None,
            cluster_preflight_scope_input_set_identity=None,
            cluster_preflight_manifest_path=Path(
                options.cluster_preflight_manifest_path
            ).resolve(strict=True),
            cluster_preflight_artifact_identity=cluster_identity,
            cluster_preflight_artifact_handle=preflight,
            cluster_preflight_state_bundle=state_bundle,
            embedding_cache_path=cache_path,
            embedding_cache=embedding_cache,
            embedding_cache_identity=cache_identity,
            embedding_cache_input_identity=copy.deepcopy(
                exact_request["embedding_cache"][
                    "production_cache_build_identity"
                ]
            ),
            embedding_cache_relocation=None,
            registry=registry,
            registry_content_sha256=registry_sha256,
            stage1_scope_plan=plan,
            scope_descriptor_root=Path(
                options.stage1_scope_descriptor_root
                or (
                    Path(options.output_dir)
                    / "stage1_scope_recovery"
                    / "descriptor"
                )
            ).resolve(),
            scope_attempt_root=Path(
                options.stage1_scope_attempt_root
                or (
                    Path(options.output_dir)
                    / "stage1_scope_recovery"
                    / "attempts"
                )
            ).resolve(),
            scope_progress_path=Path(
                options.stage1_scope_progress_path
                or (
                    Path(options.output_dir)
                    / "stage1_scope_recovery"
                    / "progress.json"
                )
            ).resolve(),
            exact_inner_contract_status=copy.deepcopy(
                projection["exact_inner_contract"]
            ),
            query_config=query_config,
            query_config_identity=query_identity,
            semantic_witness_scientific_config=semantic,
            input_file_identities={
                "dataset": {
                    "path": str(dataset_path),
                    "sha256": dataset_sha,
                    "stat_identity": list(dataset_stat),
                },
                "source_config": {
                    "path": str(config_path),
                    "sha256": config_sha,
                    "stat_identity": list(config_stat),
                },
            },
            behavior_identity=copy.deepcopy(
                exact_request["behavior_identity"]
            ),
            hierarchical_discovery_contract_identity=copy.deepcopy(
                exact_request[
                    "hierarchical_discovery_contract_identity"
                ]
            ),
            reusable_preflight_telemetry={
                "schema_version": (
                    "production_stage1_prepared_context_reopen_telemetry_v1"
                ),
                "prepared_context_reconstruction": True,
                "cluster_owner_states_loaded": 0,
            },
            request=exact_request,
            request_sha256=str(exact_request["request_sha256"]),
        )
        builder = PreparedBuildRoleNeutralProducerFactoriesBuilder(
            architecture_profiles=locators["architecture_profiles"],
            runtime_compatibility_class=locators[
                "runtime_compatibility_class"
            ],
        )
        factories = builder(prepared)
        return prepared, factories


def load_prepared_stage1_context(
    manifest_path: Path | str,
) -> PreparedStage1ContextArtifact:
    """Freshly reopen and authenticate every byte of one context artifact."""

    supplied = Path(manifest_path)
    if not supplied.is_absolute():
        raise ValueError("prepared-context manifest path must be absolute")
    manifest_file = supplied
    if supplied.name != PREPARED_STAGE1_CONTEXT_MANIFEST_NAME:
        raise ValueError("prepared-context manifest has a noncanonical name")
    root = supplied.parent
    try:
        root_stat = os.lstat(root)
    except FileNotFoundError as exc:
        raise FileNotFoundError("prepared-context root is missing") from exc
    if (
        stat.S_ISLNK(root_stat.st_mode)
        or not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_IMODE(root_stat.st_mode) != _READ_ONLY_DIRECTORY_MODE
    ):
        raise ValueError(
            "prepared-context root must be one canonical read-only "
            "non-symlink directory"
        )
    manifest_bytes = _read_stable_bytes(
        manifest_file,
        label="prepared-context manifest",
    )
    manifest = _parse_json(manifest_bytes, label="prepared-context manifest")
    required = {
        "schema_version",
        "artifact_schema_version",
        "payloads",
        "scientific_identity_sha256",
        "execution_locator_sha256",
        "content_root_sha256",
        "payload_inventory_sha256",
        "content_sha256",
    }
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    if (
        set(manifest) != required
        or manifest.get("schema_version")
        != PREPARED_STAGE1_CONTEXT_MANIFEST_SCHEMA
        or manifest.get("artifact_schema_version")
        != PREPARED_STAGE1_CONTEXT_ARTIFACT_SCHEMA
        or manifest.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("prepared-context manifest is invalid")
    payloads = _validate_inventory(root, manifest["payloads"])
    inventory_root = _sha256_json(manifest["payloads"])
    scientific_inventory = [
        row
        for row in manifest["payloads"]
        if row["relative_path"]
        == PREPARED_STAGE1_CONTEXT_SCIENTIFIC_NAME
    ]
    if (
        manifest.get("payload_inventory_sha256") != inventory_root
        or len(scientific_inventory) != 1
        or manifest.get("content_root_sha256")
        != _sha256_json(scientific_inventory)
    ):
        raise ValueError("prepared-context content roots changed")
    scientific = _parse_json(
        payloads[PREPARED_STAGE1_CONTEXT_SCIENTIFIC_NAME],
        label="prepared-context scientific identity",
    )
    locator = _parse_json(
        payloads[PREPARED_STAGE1_CONTEXT_LOCATOR_NAME],
        label="prepared-context execution locators",
    )
    scientific_body = {
        key: copy.deepcopy(value)
        for key, value in scientific.items()
        if key != "content_sha256"
    }
    locator_body = {
        key: copy.deepcopy(value)
        for key, value in locator.items()
        if key != "content_sha256"
    }
    if (
        set(scientific)
        != {
            "schema_version",
            "stage1_request_scientific_projection",
            "stage1_request_scientific_compatibility_sha256",
            "split_registry",
            "split_registry_content_sha256",
            "architecture_profiles",
            "architecture_profiles_content_sha256",
            "runtime_compatibility_class",
            "content_sha256",
        }
        or scientific.get("schema_version")
        != PREPARED_STAGE1_CONTEXT_SCIENTIFIC_SCHEMA
        or scientific.get("content_sha256") != _sha256_json(scientific_body)
        or scientific.get(
            "stage1_request_scientific_compatibility_sha256"
        )
        != (
            scientific.get("stage1_request_scientific_projection") or {}
        ).get("content_sha256")
        or manifest.get("scientific_identity_sha256")
        != scientific.get("content_sha256")
        or scientific.get("split_registry_content_sha256")
        != (
            scientific.get("stage1_request_scientific_projection") or {}
        ).get("split_registry_content_sha256")
        or scientific.get("architecture_profiles_content_sha256")
        != _sha256_json(scientific.get("architecture_profiles"))
        or not isinstance(
            scientific.get("runtime_compatibility_class"),
            str,
        )
        or not scientific["runtime_compatibility_class"].strip()
    ):
        raise ValueError("prepared-context scientific identity is invalid")
    if (
        set(locator)
        != {
            "schema_version",
            "stage1_build_options",
            "architecture_profiles",
            "runtime_compatibility_class",
            "bound_scientific_compatibility_sha256",
            "exact_stage1_request",
            "content_sha256",
        }
        or locator.get("schema_version")
        != PREPARED_STAGE1_CONTEXT_LOCATOR_SCHEMA
        or locator.get("content_sha256") != _sha256_json(locator_body)
        or manifest.get("execution_locator_sha256")
        != locator.get("content_sha256")
        or locator.get("bound_scientific_compatibility_sha256")
        != scientific.get(
            "stage1_request_scientific_compatibility_sha256"
        )
        or locator.get("architecture_profiles")
        != scientific.get("architecture_profiles")
        or locator.get("runtime_compatibility_class")
        != scientific.get("runtime_compatibility_class")
    ):
        raise ValueError("prepared-context execution locators are invalid")
    # Validate the closed locator schema without touching its target inputs.
    if (
        _locator_payload(
            stage1_build_options=locator["stage1_build_options"],
            architecture_profiles=locator["architecture_profiles"],
            runtime_compatibility_class=locator[
                "runtime_compatibility_class"
            ],
            scientific_compatibility_sha256=locator[
                "bound_scientific_compatibility_sha256"
            ],
            exact_stage1_request=locator["exact_stage1_request"],
        )
        != locator
    ):
        raise ValueError("prepared-context execution locator changed")
    return PreparedStage1ContextArtifact(
        root=root.resolve(strict=True),
        manifest_path=manifest_file.resolve(strict=True),
        manifest=copy.deepcopy(manifest),
        scientific_identity=copy.deepcopy(scientific),
        execution_locators=copy.deepcopy(locator),
    )


def _publish_prepared_stage1_context_payloads(
    *,
    root: Path,
    scientific: Mapping[str, Any],
    locator: Mapping[str, Any],
) -> PreparedStage1ContextArtifact:
    target = root
    if not target.is_absolute():
        raise ValueError("prepared-context root must be absolute")
    target = target.resolve()
    manifest_path = target / PREPARED_STAGE1_CONTEXT_MANIFEST_NAME
    if manifest_path.is_file() and not manifest_path.is_symlink():
        reopened = load_prepared_stage1_context(manifest_path)
        if (
            reopened.scientific_identity != scientific
            or reopened.execution_locators != locator
        ):
            raise FileExistsError(
                "existing prepared-context artifact has different content"
            )
        return reopened
    if target.exists() or target.is_symlink():
        raise FileExistsError("prepared-context root must be fresh")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-",
            dir=str(target.parent),
        )
    )
    published = False
    try:
        scientific_bytes = _json_bytes(scientific)
        locator_bytes = _json_bytes(locator)
        inventory = _payload_inventory(
            scientific_bytes=scientific_bytes,
            locator_bytes=locator_bytes,
        )
        scientific_inventory = [
            row
            for row in inventory
            if row["relative_path"]
            == PREPARED_STAGE1_CONTEXT_SCIENTIFIC_NAME
        ]
        manifest_body = {
            "schema_version": PREPARED_STAGE1_CONTEXT_MANIFEST_SCHEMA,
            "artifact_schema_version": PREPARED_STAGE1_CONTEXT_ARTIFACT_SCHEMA,
            "payloads": inventory,
            "scientific_identity_sha256": scientific["content_sha256"],
            "execution_locator_sha256": locator["content_sha256"],
            "content_root_sha256": _sha256_json(scientific_inventory),
            "payload_inventory_sha256": _sha256_json(inventory),
        }
        manifest = {
            **manifest_body,
            "content_sha256": _sha256_json(manifest_body),
        }
        _write_exclusive(
            staging / PREPARED_STAGE1_CONTEXT_SCIENTIFIC_NAME,
            scientific_bytes,
        )
        _write_exclusive(
            staging / PREPARED_STAGE1_CONTEXT_LOCATOR_NAME,
            locator_bytes,
        )
        _write_exclusive(
            staging / PREPARED_STAGE1_CONTEXT_MANIFEST_NAME,
            _json_bytes(manifest),
        )
        for child in staging.iterdir():
            os.chmod(child, _READ_ONLY_FILE_MODE)
        descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(staging, _READ_ONLY_DIRECTORY_MODE)
        os.rename(staging, target)
        published = True
        parent_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if not published and staging.exists():
            for child in staging.iterdir():
                try:
                    os.chmod(child, stat.S_IRUSR | stat.S_IWUSR)
                    child.unlink()
                except OSError:
                    pass
            try:
                os.chmod(
                    staging,
                    stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR,
                )
                staging.rmdir()
            except OSError:
                pass
    return load_prepared_stage1_context(manifest_path)


def rebind_prepared_stage1_context_locators(
    *,
    source_manifest_path: Path | str,
    output_root: Path | str,
    stage1_build_options: Mapping[str, Any],
    exact_stage1_request: Mapping[str, Any],
) -> PreparedStage1ContextArtifact:
    """Publish a new locator attestation for identical scientific content.

    Rebinding never mutates the source artifact and never treats a manual
    digest as authority. The caller supplies a complete typed-options mapping
    and an exact, self-hashed request; the public scientific projection must
    equal the source artifact before the new immutable attestation is written.
    """

    source = load_prepared_stage1_context(source_manifest_path)
    old = source.execution_locators
    rebound_locator = _locator_payload(
        stage1_build_options=stage1_build_options,
        architecture_profiles=old["architecture_profiles"],
        runtime_compatibility_class=old[
            "runtime_compatibility_class"
        ],
        scientific_compatibility_sha256=(
            source.scientific_compatibility_sha256
        ),
        exact_stage1_request=exact_stage1_request,
    )
    return _publish_prepared_stage1_context_payloads(
        root=Path(output_root),
        scientific=source.scientific_identity,
        locator=rebound_locator,
    )


def serialize_stage1_build_options(
    options: Any,
) -> dict[str, Any]:
    """Return the closed locator mapping for one typed build-options object."""

    from .production_stage1_bundle import Stage1BundleBuildOptions

    if not isinstance(options, Stage1BundleBuildOptions):
        raise TypeError(
            "prepared-context locator serialization requires typed "
            "Stage1BundleBuildOptions"
        )

    class _OptionsHolder:
        def __init__(self, value: Stage1BundleBuildOptions) -> None:
            self.options = value

    return _option_mapping(_OptionsHolder(options))


def seal_prepared_stage1_context(
    *,
    root: Path | str,
    prepared: Any,
    producer_factories_builder: Any,
) -> PreparedStage1ContextArtifact:
    """Publish or exactly reopen one immutable prepared-context artifact."""

    from .production_stage1_bundle import _PreparedBuild

    if not isinstance(prepared, _PreparedBuild):
        raise TypeError("prepared-context sealing requires one typed prepared build")
    from .production_role_neutral_producer_factories import (
        PreparedBuildRoleNeutralProducerFactoriesBuilder,
    )

    if not isinstance(
        producer_factories_builder,
        PreparedBuildRoleNeutralProducerFactoriesBuilder,
    ):
        raise TypeError(
            "prepared-context sealing requires the exact all-ten factory builder"
        )
    scientific = _scientific_payload(
        prepared.request,
        registry=prepared.registry,
        registry_content_sha256=prepared.registry_content_sha256,
        architecture_profiles=(
            producer_factories_builder.architecture_profiles
        ),
        runtime_compatibility_class=(
            producer_factories_builder.runtime_compatibility_class
        ),
    )
    locator = _locator_payload(
        stage1_build_options=serialize_stage1_build_options(
            prepared.options
        ),
        architecture_profiles=(
            producer_factories_builder.architecture_profiles
        ),
        runtime_compatibility_class=(
            producer_factories_builder.runtime_compatibility_class
        ),
        scientific_compatibility_sha256=scientific[
            "stage1_request_scientific_compatibility_sha256"
        ],
        exact_stage1_request=prepared.request,
    )
    return _publish_prepared_stage1_context_payloads(
        root=Path(root),
        scientific=scientific,
        locator=locator,
    )


def seal_prepared_stage1_context_from_authenticated_parts(
    *,
    root: Path | str,
    stage1_build_options: Mapping[str, Any],
    architecture_profiles: Mapping[str, Mapping[str, Any]],
    runtime_compatibility_class: str,
    exact_stage1_request: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_content_sha256: str,
) -> PreparedStage1ContextArtifact:
    """Seal a current context around already authenticated preflight state.

    This is the no-refit/no-retokenization path used after a reusable
    preflight hit.  It deliberately accepts only the same closed primitives
    used by the ordinary seal path; it cannot invent or weaken a scientific
    projection.
    """

    scientific = _scientific_payload(
        exact_stage1_request,
        registry=registry,
        registry_content_sha256=registry_content_sha256,
        architecture_profiles=architecture_profiles,
        runtime_compatibility_class=runtime_compatibility_class,
    )
    locator = _locator_payload(
        stage1_build_options=stage1_build_options,
        architecture_profiles=architecture_profiles,
        runtime_compatibility_class=runtime_compatibility_class,
        scientific_compatibility_sha256=scientific[
            "stage1_request_scientific_compatibility_sha256"
        ],
        exact_stage1_request=exact_stage1_request,
    )
    return _publish_prepared_stage1_context_payloads(
        root=Path(root),
        scientific=scientific,
        locator=locator,
    )


__all__ = [
    "PREPARED_STAGE1_CONTEXT_ARTIFACT_SCHEMA",
    "PREPARED_STAGE1_CONTEXT_LOCATOR_NAME",
    "PREPARED_STAGE1_CONTEXT_LOCATOR_SCHEMA",
    "PREPARED_STAGE1_CONTEXT_MANIFEST_NAME",
    "PREPARED_STAGE1_CONTEXT_MANIFEST_SCHEMA",
    "PREPARED_STAGE1_CONTEXT_SCIENTIFIC_NAME",
    "PREPARED_STAGE1_CONTEXT_SCIENTIFIC_SCHEMA",
    "PreparedStage1ContextArtifact",
    "load_prepared_stage1_context",
    "rebind_prepared_stage1_context_locators",
    "seal_prepared_stage1_context",
    "seal_prepared_stage1_context_from_authenticated_parts",
    "serialize_stage1_build_options",
]
