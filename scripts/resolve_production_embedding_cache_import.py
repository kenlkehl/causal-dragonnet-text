#!/usr/bin/env python3
"""Resolve one prior run's authenticated embedding-cache relocation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Iterable, Mapping

from oci.inference.portable_artifacts import (
    materialize_portable_phase,
    validate_portable_artifact,
)


def _canonical_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    state = os.lstat(path)
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise ValueError(f"{label} must be one real directory")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{label} must be canonical and symlink-free")
    return resolved


def _canonical_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    state = os.lstat(path)
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
        raise ValueError(f"{label} must be one real file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{label} must be canonical and symlink-free")
    return resolved


def _digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def resolve_import_inputs(run_root: Path) -> tuple[Path, Path, Path]:
    prior = _canonical_directory(run_root, label="prior durable run root")
    checkpoint = (
        prior / "portable_checkpoints" / "embedding_cache"
    )
    artifact = validate_portable_artifact(checkpoint)
    if artifact.manifest.get("artifact_kind") != "embedding_cache":
        raise ValueError("prior checkpoint is not an embedding-cache artifact")
    phase = materialize_portable_phase(
        artifact,
        expected_phase="embedding_cache",
    )
    result = phase.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("embedding-cache checkpoint has no phase result")
    cache = _canonical_directory(
        Path(str(result.get("cache_path", ""))),
        label="source embedding cache",
    )
    cache_bound_prepared = _canonical_file(
        Path(str(result.get("prepared_cohort_path", ""))),
        label="cache-bound prepared cohort",
    )
    metadata_path = _canonical_file(
        cache / "metadata.json",
        label="source embedding-cache metadata",
    )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("source embedding-cache metadata is unreadable") from exc
    provenance = metadata.get("production_provenance")
    dataset = provenance.get("dataset") if isinstance(provenance, Mapping) else None
    raw_historical = dataset.get("path") if isinstance(dataset, Mapping) else None
    if not isinstance(raw_historical, str) or not raw_historical.strip():
        raise ValueError("source cache does not identify its historical cohort")
    historical_prepared = _canonical_file(
        Path(raw_historical),
        label="historical cache-bound prepared cohort",
    )
    historical_manifest = _canonical_file(
        historical_prepared.parent / "preparation_manifest.json",
        label="historical cache-bound preparation manifest",
    )
    if _digest(cache_bound_prepared) != _digest(historical_prepared):
        raise ValueError(
            "portable cache-bound cohort differs from its historical source"
        )
    return cache, historical_prepared, historical_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve_import_inputs(args.run_root)
    for path in paths:
        encoded = str(path)
        if "\n" in encoded or "\r" in encoded:
            raise ValueError("embedding-cache import path contains a line break")
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
