"""Measured codec selection for compact clustered-preflight artifacts.

The benchmark takes one already-authenticated portable preflight artifact and
publishes fresh, complete replicas for every codec configured by deployment.
It measures wall time, CPU time, process I/O, and explicit logical byte
activity.  A codec is selectable only when every replica reopens successfully
and has the source artifact's exact path-neutral scientific content.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import statistics
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from .performance_telemetry import _proc_io
from .portable_workflow_spec import identity_sha256
from .production_stage1_cluster_preflight_artifact_v2 import (
    PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME,
    SUPPORTED_PORTABLE_CLUSTER_PREFLIGHT_PARQUET_COMPRESSIONS,
    PortableProductionStage1ClusterPreflightArtifact,
    load_path_only_portable_production_stage1_cluster_preflight_artifact,
    transcode_portable_production_stage1_cluster_preflight_artifact,
)

COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_CONFIG_SCHEMA = (
    "compact_preflight_compression_benchmark_config_v1"
)
COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_RESULT_SCHEMA = (
    "compact_preflight_compression_benchmark_result_v1"
)
COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_OBSERVATION_SCHEMA = (
    "compact_preflight_compression_benchmark_observation_v1"
)
COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_SCHEDULE_SCHEMA = (
    "compact_preflight_compression_benchmark_schedule_v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BYTE_COUNTER_FIELDS = (
    "read",
    "written",
    "copied",
    "hashed",
    "compressed",
    "decompressed",
    "json_encoded",
    "json_decoded",
    "fsynced",
)
_STORAGE_FIELDS = {
    "parquet_compression",
    "registered_payload_bytes",
    "manifest_bytes",
    "tree_file_bytes",
    "parquet_file_bytes",
    "json_file_bytes",
    "parquet_compressed_column_bytes",
    "parquet_uncompressed_column_bytes",
}


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _strict_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(
                f"compression benchmark JSON repeats key {key!r}"
            )
        output[key] = value
    return output


@dataclass(frozen=True)
class CompactPreflightCompressionBenchmarkConfig:
    """Deployment-owned codec candidates and observation counts."""

    codecs: tuple[str, ...]
    warmup_repetitions_per_codec: int
    measured_repetitions_per_codec: int
    schema_version: str = (
        COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_CONFIG_SCHEMA
    )

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_CONFIG_SCHEMA
        ):
            raise ValueError(
                "unsupported compact-preflight compression benchmark config"
            )
        codecs = tuple(str(value) for value in self.codecs)
        supported = (
            SUPPORTED_PORTABLE_CLUSTER_PREFLIGHT_PARQUET_COMPRESSIONS
        )
        if (
            not codecs
            or len(codecs) != len(set(codecs))
            or set(codecs) != set(supported)
        ):
            raise ValueError(
                "compression benchmark must configure every supported compact-"
                "preflight codec exactly once"
            )
        object.__setattr__(self, "codecs", codecs)
        object.__setattr__(
            self,
            "warmup_repetitions_per_codec",
            _nonnegative_integer(
                self.warmup_repetitions_per_codec,
                label="warmup_repetitions_per_codec",
            ),
        )
        object.__setattr__(
            self,
            "measured_repetitions_per_codec",
            _positive_integer(
                self.measured_repetitions_per_codec,
                label="measured_repetitions_per_codec",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "codecs": list(self.codecs),
            "warmup_repetitions_per_codec": (
                self.warmup_repetitions_per_codec
            ),
            "measured_repetitions_per_codec": (
                self.measured_repetitions_per_codec
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "CompactPreflightCompressionBenchmarkConfig":
        required = {field.name for field in fields(cls)}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                "compression benchmark config must configure every field "
                f"exactly; required={sorted(required)}"
            )
        raw_codecs = value.get("codecs")
        if not isinstance(raw_codecs, list):
            raise TypeError("compression benchmark codecs must be a list")
        return cls(
            schema_version=str(value["schema_version"]),
            codecs=tuple(str(item) for item in raw_codecs),
            warmup_repetitions_per_codec=value[
                "warmup_repetitions_per_codec"
            ],
            measured_repetitions_per_codec=value[
                "measured_repetitions_per_codec"
            ],
        )


def _parquet_column_storage(path: Path) -> tuple[int, int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "compact-preflight compression benchmark requires pyarrow"
        ) from exc
    metadata = pq.read_metadata(path)
    compressed = 0
    uncompressed = 0
    for row_group in range(metadata.num_row_groups):
        group = metadata.row_group(row_group)
        for column in range(group.num_columns):
            child = group.column(column)
            compressed += int(child.total_compressed_size)
            uncompressed += int(child.total_uncompressed_size)
    return compressed, uncompressed


def _artifact_storage(
    artifact: PortableProductionStage1ClusterPreflightArtifact,
) -> dict[str, int | str]:
    manifest_path = artifact.manifest_path
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "compression benchmark source manifest is invalid"
        ) from exc
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    physical = (
        manifest.get("physical_storage")
        if isinstance(manifest, Mapping)
        else None
    )
    if not isinstance(files, list) or not isinstance(physical, Mapping):
        raise ValueError(
            "compression benchmark source lacks physical storage inventory"
        )
    registered_bytes = 0
    parquet_file_bytes = 0
    json_file_bytes = int(manifest_path.stat().st_size)
    parquet_compressed_column_bytes = 0
    parquet_uncompressed_column_bytes = 0
    for row in files:
        if not isinstance(row, Mapping):
            raise ValueError(
                "compression benchmark source inventory is malformed"
            )
        relative = row.get("relative_path")
        size = row.get("size_bytes")
        if (
            not isinstance(relative, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
        ):
            raise ValueError(
                "compression benchmark source inventory is malformed"
            )
        path = artifact.root / relative
        if int(path.stat().st_size) != size:
            raise ValueError(
                "compression benchmark artifact size changed"
            )
        registered_bytes += size
        if relative.endswith(".parquet"):
            parquet_file_bytes += size
            compressed, uncompressed = _parquet_column_storage(path)
            parquet_compressed_column_bytes += compressed
            parquet_uncompressed_column_bytes += uncompressed
        else:
            json_file_bytes += size
    codec = physical.get("parquet_compression")
    if codec not in (
        SUPPORTED_PORTABLE_CLUSTER_PREFLIGHT_PARQUET_COMPRESSIONS
    ):
        raise ValueError(
            "compression benchmark artifact has an unsupported codec"
        )
    return {
        "parquet_compression": str(codec),
        "registered_payload_bytes": registered_bytes,
        "manifest_bytes": int(manifest_path.stat().st_size),
        "tree_file_bytes": (
            registered_bytes + int(manifest_path.stat().st_size)
        ),
        "parquet_file_bytes": parquet_file_bytes,
        "json_file_bytes": json_file_bytes,
        "parquet_compressed_column_bytes": (
            parquet_compressed_column_bytes
        ),
        "parquet_uncompressed_column_bytes": (
            parquet_uncompressed_column_bytes
        ),
    }


def _validate_storage(value: Any) -> dict[str, int | str]:
    if not isinstance(value, Mapping) or set(value) != _STORAGE_FIELDS:
        raise ValueError(
            "compression benchmark storage telemetry is not closed"
        )
    codec = value.get("parquet_compression")
    if codec not in (
        SUPPORTED_PORTABLE_CLUSTER_PREFLIGHT_PARQUET_COMPRESSIONS
    ):
        raise ValueError(
            "compression benchmark storage codec is unsupported"
        )
    output: dict[str, int | str] = {
        "parquet_compression": str(codec),
    }
    for key in sorted(_STORAGE_FIELDS - {"parquet_compression"}):
        child = value.get(key)
        if (
            isinstance(child, bool)
            or not isinstance(child, int)
            or child < 0
        ):
            raise ValueError(
                "compression benchmark storage byte count is invalid"
            )
        output[key] = int(child)
    if (
        int(output["registered_payload_bytes"])
        + int(output["manifest_bytes"])
        != int(output["tree_file_bytes"])
        or int(output["parquet_file_bytes"])
        + int(output["json_file_bytes"])
        != int(output["tree_file_bytes"])
        or int(output["parquet_file_bytes"]) < 1
        or int(output["json_file_bytes"]) < 1
        or int(output["parquet_compressed_column_bytes"]) < 1
        or int(output["parquet_uncompressed_column_bytes"]) < 1
    ):
        raise ValueError(
            "compression benchmark storage totals are inconsistent"
        )
    return output


def _logical_byte_counters(
    *,
    source_storage: Mapping[str, int | str],
    output_storage: Mapping[str, int | str],
) -> dict[str, int]:
    source_parquet = int(source_storage["parquet_file_bytes"])
    output_parquet = int(output_storage["parquet_file_bytes"])
    output_payload = int(output_storage["registered_payload_bytes"])
    output_tree = int(output_storage["tree_file_bytes"])
    output_json = int(output_storage["json_file_bytes"])
    output_uncompressed = int(
        output_storage["parquet_uncompressed_column_bytes"]
    )
    source_uncompressed = int(
        source_storage["parquet_uncompressed_column_bytes"]
    )
    source_codec = str(source_storage["parquet_compression"])
    output_codec = str(output_storage["parquet_compression"])
    # Bulk-byte passes made by the transcoder and its mandatory fresh
    # validation. Metadata/footer reads remain visible in process I/O.
    return {
        "read": (
            source_parquet
            + output_payload
            + output_tree
            + output_parquet
        ),
        "written": output_tree,
        "copied": 0,
        "hashed": output_payload + output_tree,
        "compressed": (
            output_uncompressed if output_codec != "none" else 0
        ),
        "decompressed": (
            (source_uncompressed if source_codec != "none" else 0)
            + (output_uncompressed if output_codec != "none" else 0)
        ),
        "json_encoded": output_json,
        "json_decoded": output_json,
        "fsynced": output_tree,
    }


def _schedule(
    config: CompactPreflightCompressionBenchmarkConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sequence = 0
    for kind, repetitions in (
        ("warmup", config.warmup_repetitions_per_codec),
        ("measured", config.measured_repetitions_per_codec),
    ):
        for repetition in range(repetitions):
            offset = repetition % len(config.codecs)
            ordered = config.codecs[offset:] + config.codecs[:offset]
            for position, codec in enumerate(ordered):
                rows.append(
                    {
                        "sequence_index": sequence,
                        "observation_kind": kind,
                        "repetition_index": repetition,
                        "rotation_offset": offset,
                        "codec_position": position,
                        "parquet_compression": codec,
                    }
                )
                sequence += 1
    body = {
        "schema_version": (
            COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_SCHEDULE_SCHEMA
        ),
        "ordering_policy": (
            "configured_codec_rotation_warmups_excluded_v1"
        ),
        "entries": rows,
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _observation(
    *,
    source: PortableProductionStage1ClusterPreflightArtifact,
    source_storage: Mapping[str, int | str],
    codec: str,
    observation_kind: str,
    repetition_index: int,
    sequence_index: int,
    destination: Path,
) -> dict[str, Any]:
    before_io = _proc_io()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    replica = (
        transcode_portable_production_stage1_cluster_preflight_artifact(
            source=source,
            output_dir=destination,
            parquet_compression=codec,
        )
    )
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    after_io = _proc_io()
    output_storage = _artifact_storage(replica)
    identity = replica.identity()
    source_identity = source.identity()
    counters = _logical_byte_counters(
        source_storage=source_storage,
        output_storage=output_storage,
    )
    body = {
        "schema_version": (
            COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_OBSERVATION_SCHEMA
        ),
        "sequence_index": sequence_index,
        "observation_kind": observation_kind,
        "repetition_index": repetition_index,
        "parquet_compression": codec,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "process_read_bytes": (
            None
            if before_io is None or after_io is None
            else max(0, int(after_io[0]) - int(before_io[0]))
        ),
        "process_written_bytes": (
            None
            if before_io is None or after_io is None
            else max(0, int(after_io[1]) - int(before_io[1]))
        ),
        "logical_byte_counters": counters,
        "byte_accounting_basis": (
            "known_bulk_transcode_hash_reopen_and_semantic_parse_passes_v1"
        ),
        "output_storage": dict(output_storage),
        "artifact_manifest_path": str(replica.manifest_path),
        "artifact_content_sha256": identity["content_sha256"],
        "payload_inventory_content_sha256": identity[
            "payload_inventory_content_sha256"
        ],
        "path_neutral_scientific_content_sha256": identity[
            "path_neutral_scientific_content_sha256"
        ],
        "scientifically_equal_to_source": (
            identity["path_neutral_scientific_content_sha256"]
            == source_identity[
                "path_neutral_scientific_content_sha256"
            ]
        ),
        "status": "complete",
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _summaries(
    *,
    config: CompactPreflightCompressionBenchmarkConfig,
    source_scientific_sha256: str,
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    results: list[dict[str, Any]] = []
    for codec in config.codecs:
        rows = [
            row
            for row in observations
            if row.get("observation_kind") == "measured"
            and row.get("parquet_compression") == codec
        ]
        roots = {
            str(row.get("path_neutral_scientific_content_sha256"))
            for row in rows
        }
        inventories = {
            str(row.get("payload_inventory_content_sha256"))
            for row in rows
        }
        output_sizes = {
            int(row["output_storage"]["tree_file_bytes"])
            for row in rows
            if isinstance(row.get("output_storage"), Mapping)
            and isinstance(
                row["output_storage"].get("tree_file_bytes"),
                int,
            )
        }
        accepted = (
            len(rows) == config.measured_repetitions_per_codec
            and roots == {source_scientific_sha256}
            and len(inventories) == 1
            and len(output_sizes) == 1
            and all(
                row.get("status") == "complete"
                and row.get("scientifically_equal_to_source") is True
                for row in rows
            )
        )
        result = {
            "parquet_compression": codec,
            "measured_observation_count": len(rows),
            "median_wall_seconds": (
                statistics.median(float(row["wall_seconds"]) for row in rows)
                if rows
                else None
            ),
            "median_cpu_seconds": (
                statistics.median(float(row["cpu_seconds"]) for row in rows)
                if rows
                else None
            ),
            "median_process_read_bytes": (
                statistics.median(
                    int(row["process_read_bytes"])
                    for row in rows
                    if row.get("process_read_bytes") is not None
                )
                if rows
                and all(row.get("process_read_bytes") is not None for row in rows)
                else None
            ),
            "median_process_written_bytes": (
                statistics.median(
                    int(row["process_written_bytes"])
                    for row in rows
                    if row.get("process_written_bytes") is not None
                )
                if rows
                and all(row.get("process_written_bytes") is not None for row in rows)
                else None
            ),
            "output_tree_file_bytes": (
                next(iter(output_sizes)) if len(output_sizes) == 1 else None
            ),
            "deterministic_payload_inventory": len(inventories) == 1,
            "path_neutral_scientific_equality": (
                roots == {source_scientific_sha256}
            ),
            "accepted": accepted,
        }
        results.append(result)
    selectable = [
        (index, row)
        for index, row in enumerate(results)
        if row["accepted"] is True
    ]
    selected = (
        None
        if len(selectable) != len(config.codecs)
        else min(
            selectable,
            key=lambda item: (
                float(item[1]["median_wall_seconds"]),
                int(item[1]["output_tree_file_bytes"]),
                item[0],
            ),
        )[1]["parquet_compression"]
    )
    return results, selected


def run_compact_preflight_compression_benchmark(
    *,
    config: CompactPreflightCompressionBenchmarkConfig,
    source: PortableProductionStage1ClusterPreflightArtifact,
    output_root: Path | str,
) -> dict[str, Any]:
    """Measure every configured codec and select the fastest equal replica."""

    if not isinstance(
        config,
        CompactPreflightCompressionBenchmarkConfig,
    ):
        raise TypeError("compression benchmark requires a typed config")
    if not isinstance(
        source,
        PortableProductionStage1ClusterPreflightArtifact,
    ):
        raise TypeError("compression benchmark requires a typed source")
    destination = Path(output_root)
    if not destination.is_absolute():
        raise ValueError("compression benchmark output root must be absolute")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "compression benchmark output root must be fresh"
        )
    parent = destination.parent.resolve(strict=True)
    if parent != destination.parent or not parent.is_dir():
        raise ValueError(
            "compression benchmark output parent must be canonical"
        )
    destination.mkdir(exist_ok=False)
    (destination / "warmups").mkdir(exist_ok=False)
    (destination / "runs").mkdir(exist_ok=False)

    source_identity = source.identity()
    source_scientific = source_identity[
        "path_neutral_scientific_content_sha256"
    ]
    if _SHA256.fullmatch(str(source_scientific)) is None:
        raise ValueError(
            "compression benchmark source lacks scientific identity"
        )
    source_storage = _artifact_storage(source)
    schedule = _schedule(config)
    observations: list[dict[str, Any]] = []
    for row in schedule["entries"]:
        kind = str(row["observation_kind"])
        codec = str(row["parquet_compression"])
        repetition = int(row["repetition_index"])
        root = (
            destination
            / ("warmups" if kind == "warmup" else "runs")
            / f"{repetition:04d}-{int(row['codec_position']):04d}-{codec}"
        ).resolve()
        observations.append(
            _observation(
                source=source,
                source_storage=source_storage,
                codec=codec,
                observation_kind=kind,
                repetition_index=repetition,
                sequence_index=int(row["sequence_index"]),
                destination=root,
            )
        )
    codec_results, selected = _summaries(
        config=config,
        source_scientific_sha256=source_scientific,
        observations=observations,
    )
    warmups = [
        row for row in observations if row["observation_kind"] == "warmup"
    ]
    measured = [
        row for row in observations if row["observation_kind"] == "measured"
    ]
    warmups_equal = all(
        row["scientifically_equal_to_source"] is True for row in warmups
    )
    body = {
        "schema_version": (
            COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_RESULT_SCHEMA
        ),
        "status": "complete",
        "config": config.as_dict(),
        "config_sha256": identity_sha256(config.as_dict()),
        "source": {
            "manifest_path": str(source.manifest_path),
            "artifact_content_sha256": source_identity["content_sha256"],
            "path_neutral_scientific_content_sha256": source_scientific,
            "logical_scope_count": source_identity["scope_count"],
            "physical_fit_count": source_identity["physical_fit_count"],
            "physical_storage": copy.deepcopy(
                source_identity["physical_storage"]
            ),
            "storage": source_storage,
        },
        "execution_schedule": schedule,
        "warmup_observations": warmups,
        "warmup_observations_excluded_from_selection": True,
        "measured_observations": measured,
        "codec_results": codec_results,
        "selected_parquet_compression": selected,
        "selection_policy": (
            "lowest_median_wall_then_output_bytes_then_config_order_v1"
        ),
        "all_warmups_scientifically_equal": warmups_equal,
        "cpu_only_serial_codec_benchmark": True,
        "accepted": (
            selected is not None
            and warmups_equal
            and all(row["accepted"] is True for row in codec_results)
        ),
    }
    result = {**body, "content_sha256": identity_sha256(body)}
    result_path = destination / "compression_benchmark_result.json"
    payload = (
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        result_path,
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
                    "compression benchmark result write made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return copy.deepcopy(result)


def _copy_authenticated_replica_tree(
    *,
    source_manifest: Path,
    destination_root: Path,
) -> PortableProductionStage1ClusterPreflightArtifact:
    """Copy one authenticated replica once, hashing every byte while writing."""

    source = (
        load_path_only_portable_production_stage1_cluster_preflight_artifact(
            manifest_path=source_manifest,
        )
    )
    source_root = source.root
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    destination_root.mkdir(mode=0o700)
    directories: list[tuple[Path, int]] = []
    for path in sorted(
        source_root.rglob("*"),
        key=lambda value: (
            len(value.relative_to(source_root).parts),
            value.as_posix(),
        ),
    ):
        relative = path.relative_to(source_root)
        source_state = os.lstat(path)
        destination = destination_root / relative
        if stat.S_ISLNK(source_state.st_mode):
            raise ValueError(
                "compression benchmark publication cannot copy links"
            )
        if stat.S_ISDIR(source_state.st_mode):
            destination.mkdir(mode=0o700)
            directories.append(
                (destination, stat.S_IMODE(source_state.st_mode))
            )
            continue
        if (
            not stat.S_ISREG(source_state.st_mode)
            or int(source_state.st_nlink) != 1
        ):
            raise ValueError(
                "compression benchmark publication requires private files"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_descriptor = os.open(
            path,
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
            0o600,
        )
        digest = hashlib.sha256()
        copied = 0
        try:
            while True:
                block = os.read(source_descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                copied += len(block)
                view = memoryview(block)
                while view:
                    written = os.write(
                        destination_descriptor,
                        view,
                    )
                    if written < 1:
                        raise OSError(
                            "compression benchmark publication made no progress"
                        )
                    view = view[written:]
            os.fsync(destination_descriptor)
        finally:
            os.close(source_descriptor)
            os.close(destination_descriptor)
        after = os.lstat(path)
        if (
            copied != int(source_state.st_size)
            or _SHA256.fullmatch(digest.hexdigest()) is None
            or (
                int(source_state.st_dev),
                int(source_state.st_ino),
                int(source_state.st_mode),
                int(source_state.st_nlink),
                int(source_state.st_size),
                int(source_state.st_mtime_ns),
                int(source_state.st_ctime_ns),
            )
            != (
                int(after.st_dev),
                int(after.st_ino),
                int(after.st_mode),
                int(after.st_nlink),
                int(after.st_size),
                int(after.st_mtime_ns),
                int(after.st_ctime_ns),
            )
        ):
            raise RuntimeError(
                "compression benchmark replica changed while being published"
            )
        os.chmod(destination, stat.S_IMODE(source_state.st_mode))
    for directory, mode in sorted(
        directories,
        key=lambda value: len(value[0].parts),
        reverse=True,
    ):
        os.chmod(directory, mode)
        descriptor = os.open(
            directory,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    os.chmod(
        destination_root,
        stat.S_IMODE(os.lstat(source_root).st_mode),
    )
    descriptor = os.open(
        destination_root,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    copied_artifact = (
        load_path_only_portable_production_stage1_cluster_preflight_artifact(
            manifest_path=(
                destination_root
                / PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME
            ),
        )
    )
    copied_identity = copied_artifact.identity()
    source_identity = source.identity()
    if (
        copied_identity[
            "path_neutral_scientific_content_sha256"
        ]
        != source_identity[
            "path_neutral_scientific_content_sha256"
        ]
        or copied_identity["payload_inventory_content_sha256"]
        != source_identity["payload_inventory_content_sha256"]
    ):
        raise RuntimeError(
            "compression benchmark publication changed replica content"
        )
    return copied_artifact


def publish_compact_preflight_compression_benchmark_result(
    value: Mapping[str, Any],
    *,
    output_root: Path | str,
) -> dict[str, Any]:
    """Publish terminal codec evidence from local scratch to durable storage."""

    validated = validate_compact_preflight_compression_benchmark_result(
        value,
        reopen_artifacts=True,
    )
    destination = Path(output_root)
    if not destination.is_absolute():
        raise ValueError(
            "compression benchmark publication root must be absolute"
        )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "compression benchmark publication root must be fresh"
        )
    parent = destination.parent.resolve(strict=True)
    if parent != destination.parent or not parent.is_dir():
        raise ValueError(
            "compression benchmark publication parent must be canonical"
        )
    destination.mkdir(mode=0o700)
    temporary = destination
    try:
        (temporary / "warmups").mkdir()
        (temporary / "runs").mkdir()
        relocated_rows: dict[str, list[dict[str, Any]]] = {
            "warmup_observations": [],
            "measured_observations": [],
        }
        for collection in relocated_rows:
            kind = (
                "warmups"
                if collection == "warmup_observations"
                else "runs"
            )
            for row in validated[collection]:
                sequence = int(row["sequence_index"])
                codec = str(row["parquet_compression"])
                relative_root = (
                    Path(kind) / f"{sequence:04d}-{codec}"
                )
                copied = _copy_authenticated_replica_tree(
                    source_manifest=Path(
                        str(row["artifact_manifest_path"])
                    ),
                    destination_root=temporary / relative_root,
                )
                relocated_body = {
                    key: copy.deepcopy(child)
                    for key, child in row.items()
                    if key != "content_sha256"
                }
                relocated_body["artifact_manifest_path"] = str(
                    destination
                    / relative_root
                    / PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME
                )
                relocated_body["artifact_content_sha256"] = (
                    copied.identity()["content_sha256"]
                )
                if (
                    copied.identity()[
                        "path_neutral_scientific_content_sha256"
                    ]
                    != relocated_body[
                        "path_neutral_scientific_content_sha256"
                    ]
                ):
                    raise RuntimeError(
                        "published compression replica identity changed"
                    )
                relocated_rows[collection].append(
                    {
                        **relocated_body,
                        "content_sha256": identity_sha256(
                            relocated_body
                        ),
                    }
                )
        result_body = {
            key: copy.deepcopy(child)
            for key, child in validated.items()
            if key != "content_sha256"
        }
        result_body.update(relocated_rows)
        result = {
            **result_body,
            "content_sha256": identity_sha256(result_body),
        }
        result_path = temporary / "compression_benchmark_result.json"
        payload = (
            json.dumps(
                result,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            result_path,
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
                        "compression benchmark result publication made no progress"
                    )
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(temporary / "warmups", 0o555)
        os.chmod(temporary / "runs", 0o555)
        os.chmod(temporary, 0o555)
        descriptor = os.open(
            temporary,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        if temporary.exists():
            for path in sorted(
                temporary.rglob("*"),
                key=lambda item: len(item.parts),
                reverse=True,
            ):
                try:
                    os.chmod(
                        path,
                        0o700 if path.is_dir() else 0o600,
                    )
                except OSError:
                    pass
            try:
                os.chmod(temporary, 0o700)
            except OSError:
                pass
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_compact_preflight_compression_benchmark_result(
        result,
        reopen_artifacts=True,
    )


def validate_compact_preflight_compression_benchmark_result(
    value: Mapping[str, Any],
    *,
    reopen_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate one closed result and optionally reopen every replica."""

    required = {
        "schema_version",
        "status",
        "config",
        "config_sha256",
        "source",
        "execution_schedule",
        "warmup_observations",
        "warmup_observations_excluded_from_selection",
        "measured_observations",
        "codec_results",
        "selected_parquet_compression",
        "selection_policy",
        "all_warmups_scientifically_equal",
        "cpu_only_serial_codec_benchmark",
        "accepted",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(
            "compression benchmark result does not match its closed schema"
        )
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    if (
        value.get("schema_version")
        != COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_RESULT_SCHEMA
        or value.get("status") != "complete"
        or value.get("content_sha256") != identity_sha256(body)
        or value.get("warmup_observations_excluded_from_selection")
        is not True
        or value.get("cpu_only_serial_codec_benchmark") is not True
        or value.get("selection_policy")
        != "lowest_median_wall_then_output_bytes_then_config_order_v1"
    ):
        raise ValueError(
            "compression benchmark result envelope is invalid"
        )
    raw_config = value.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("compression benchmark result lacks its config")
    config = CompactPreflightCompressionBenchmarkConfig.from_mapping(
        raw_config
    )
    if value.get("config_sha256") != identity_sha256(config.as_dict()):
        raise ValueError(
            "compression benchmark config identity is invalid"
        )
    expected_schedule = _schedule(config)
    if value.get("execution_schedule") != expected_schedule:
        raise ValueError(
            "compression benchmark schedule is incomplete or changed"
        )
    source = value.get("source")
    warmups = value.get("warmup_observations")
    measured = value.get("measured_observations")
    source_fields = {
        "manifest_path",
        "artifact_content_sha256",
        "path_neutral_scientific_content_sha256",
        "logical_scope_count",
        "physical_fit_count",
        "physical_storage",
        "storage",
    }
    if (
        not isinstance(source, Mapping)
        or set(source) != source_fields
        or not Path(str(source.get("manifest_path", ""))).is_absolute()
        or Path(str(source.get("manifest_path", ""))).name
        != PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME
        or _SHA256.fullmatch(
            str(source.get("artifact_content_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(source.get("path_neutral_scientific_content_sha256", ""))
        )
        is None
        or isinstance(source.get("logical_scope_count"), bool)
        or not isinstance(source.get("logical_scope_count"), int)
        or int(source["logical_scope_count"]) < 1
        or isinstance(source.get("physical_fit_count"), bool)
        or not isinstance(source.get("physical_fit_count"), int)
        or int(source["physical_fit_count"]) < 1
        or not isinstance(source.get("physical_storage"), Mapping)
        or source["physical_storage"].get("parquet_compression")
        not in SUPPORTED_PORTABLE_CLUSTER_PREFLIGHT_PARQUET_COMPRESSIONS
        or not isinstance(warmups, list)
        or not isinstance(measured, list)
    ):
        raise ValueError(
            "compression benchmark source or observations are invalid"
        )
    source_storage = _validate_storage(source["storage"])
    observations = [*warmups, *measured]
    expected_entries = expected_schedule["entries"]
    if len(observations) != len(expected_entries):
        raise ValueError(
            "compression benchmark observation count changed"
        )
    by_sequence: dict[int, Mapping[str, Any]] = {}
    for row in observations:
        if not isinstance(row, Mapping):
            raise ValueError(
                "compression benchmark observation is not an object"
            )
        row_body = {
            key: copy.deepcopy(child)
            for key, child in row.items()
            if key != "content_sha256"
        }
        sequence = row.get("sequence_index")
        counters = row.get("logical_byte_counters")
        output_storage = _validate_storage(row.get("output_storage"))
        expected_counters = _logical_byte_counters(
            source_storage=source_storage,
            output_storage=output_storage,
        )
        manifest_path = Path(str(row.get("artifact_manifest_path", "")))
        process_read = row.get("process_read_bytes")
        process_written = row.get("process_written_bytes")
        if (
            set(row)
            != {
                "schema_version",
                "sequence_index",
                "observation_kind",
                "repetition_index",
                "parquet_compression",
                "wall_seconds",
                "cpu_seconds",
                "process_read_bytes",
                "process_written_bytes",
                "logical_byte_counters",
                "byte_accounting_basis",
                "output_storage",
                "artifact_manifest_path",
                "artifact_content_sha256",
                "payload_inventory_content_sha256",
                "path_neutral_scientific_content_sha256",
                "scientifically_equal_to_source",
                "status",
                "content_sha256",
            }
            or row.get("schema_version")
            != COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_OBSERVATION_SCHEMA
            or row.get("content_sha256") != identity_sha256(row_body)
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence in by_sequence
            or not isinstance(counters, Mapping)
            or set(counters) != set(_BYTE_COUNTER_FIELDS)
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                for item in counters.values()
            )
            or dict(counters) != expected_counters
            or not manifest_path.is_absolute()
            or manifest_path.name
            != PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME
            or _SHA256.fullmatch(
                str(row.get("artifact_content_sha256", ""))
            )
            is None
            or _SHA256.fullmatch(
                str(row.get("payload_inventory_content_sha256", ""))
            )
            is None
            or row.get("byte_accounting_basis")
            != "known_bulk_transcode_hash_reopen_and_semantic_parse_passes_v1"
            or output_storage["parquet_compression"]
            != row.get("parquet_compression")
            or (
                process_read is not None
                and (
                    isinstance(process_read, bool)
                    or not isinstance(process_read, int)
                    or process_read < 0
                )
            )
            or (
                process_written is not None
                and (
                    isinstance(process_written, bool)
                    or not isinstance(process_written, int)
                    or process_written < 0
                )
            )
            or not math.isfinite(float(row.get("wall_seconds", -1)))
            or float(row.get("wall_seconds", -1)) <= 0
            or not math.isfinite(float(row.get("cpu_seconds", -1)))
            or float(row.get("cpu_seconds", -1)) < 0
            or row.get("status") != "complete"
            or row.get("scientifically_equal_to_source") is not True
            or row.get("path_neutral_scientific_content_sha256")
            != source["path_neutral_scientific_content_sha256"]
        ):
            raise ValueError(
                "compression benchmark observation is invalid"
            )
        by_sequence[sequence] = row
    for expected in expected_entries:
        row = by_sequence.get(int(expected["sequence_index"]))
        if (
            row is None
            or row.get("observation_kind")
            != expected["observation_kind"]
            or row.get("repetition_index")
            != expected["repetition_index"]
            or row.get("parquet_compression")
            != expected["parquet_compression"]
        ):
            raise ValueError(
                "compression benchmark observations changed schedule"
            )
    expected_codec_results, expected_selected = _summaries(
        config=config,
        source_scientific_sha256=str(
            source["path_neutral_scientific_content_sha256"]
        ),
        observations=observations,
    )
    warmups_equal = all(
        row.get("scientifically_equal_to_source") is True
        for row in warmups
    )
    if (
        value.get("codec_results") != expected_codec_results
        or value.get("selected_parquet_compression") != expected_selected
        or value.get("all_warmups_scientifically_equal") is not warmups_equal
        or value.get("accepted")
        is not (
            expected_selected is not None
            and warmups_equal
            and all(
                row["accepted"] is True
                for row in expected_codec_results
            )
        )
    ):
        raise ValueError(
            "compression benchmark selection or equality gate is invalid"
        )
    if reopen_artifacts:
        for row in observations:
            manifest = Path(str(row["artifact_manifest_path"]))
            replica = (
                load_path_only_portable_production_stage1_cluster_preflight_artifact(
                    manifest_path=manifest,
                )
            )
            identity = replica.identity()
            for owner in identity["physical_scope_order"]:
                replica.owner_fit_identity(str(owner))
            if (
                identity["content_sha256"]
                != row["artifact_content_sha256"]
                or identity["payload_inventory_content_sha256"]
                != row["payload_inventory_content_sha256"]
                or identity["path_neutral_scientific_content_sha256"]
                != source["path_neutral_scientific_content_sha256"]
                or identity["physical_storage"]["parquet_compression"]
                != row["parquet_compression"]
            ):
                raise ValueError(
                    "compression benchmark replica changed on fresh reopen"
                )
    return copy.deepcopy(dict(value))


__all__ = [
    "COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_CONFIG_SCHEMA",
    "COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_OBSERVATION_SCHEMA",
    "COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_RESULT_SCHEMA",
    "COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_SCHEDULE_SCHEMA",
    "CompactPreflightCompressionBenchmarkConfig",
    "publish_compact_preflight_compression_benchmark_result",
    "run_compact_preflight_compression_benchmark",
    "validate_compact_preflight_compression_benchmark_result",
]
