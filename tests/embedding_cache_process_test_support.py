"""Importable spawn targets for embedding-cache process-boundary tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def successful_target(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "received": dict(parameters),
        "pid": int(os.getpid()),
        "native_thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def failing_target(_parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    raise RuntimeError("intentional spawned embedding-cache failure")


def fake_production_target(
    parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    target = Path(str(parameters["target_dir"]))
    target.mkdir()
    payload = target / "payload.bin"
    payload.write_bytes(b"complete-cache")
    identity = {
        "cache_path": str(target.resolve(strict=True)),
        "cache_files": {
            payload.name: {
                "sha256": "a" * 64,
                "size_bytes": payload.stat().st_size,
            }
        },
    }
    return {
        "schema_version": "production_embedding_cache_spawn_build_v2",
        "cache_path": str(target.resolve(strict=True)),
        "build_identity": identity,
        "model_materialized_in_worker_process": True,
        "model_materialized_in_parent_process": False,
    }
