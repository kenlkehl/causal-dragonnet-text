from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import oci.inference.production_embedding_cache_builder as builder_module
import oci.inference.production_embedding_cache_process as process_module
from oci.inference.production_all_evidence_workflow import (
    ProductionAllEvidenceWorkflow,
)
from oci.inference.production_embedding_cache_process import (
    _run_spawned_target,
)


_SUPPORT = "tests.embedding_cache_process_test_support"


def test_spawn_boundary_returns_only_after_worker_exit_and_binds_cpu_budget() -> None:
    parent_pid = os.getpid()
    result = _run_spawned_target(
        worker_target=f"{_SUPPORT}:successful_target",
        worker_parameters={"sentinel": ["closed", 7]},
        cpu_budget=3,
    )

    assert result["status"] == "completed"
    assert result["result"]["received"] == {"sentinel": ["closed", 7]}
    assert result["result"]["pid"] != parent_pid
    assert set(result["result"]["native_thread_environment"].values()) == {
        "3"
    }
    assert result["telemetry"]["worker_pid"] == result["result"]["pid"]
    assert result["telemetry"]["wall_seconds"] >= 0
    assert result["telemetry"]["cpu_seconds"] >= 0


def test_spawn_boundary_propagates_worker_failure_after_process_cleanup() -> None:
    with pytest.raises(
        RuntimeError,
        match="intentional spawned embedding-cache failure",
    ):
        _run_spawned_target(
            worker_target=f"{_SUPPORT}:failing_target",
            worker_parameters={"sentinel": "failure"},
            cpu_budget=1,
        )


@pytest.mark.parametrize("cpu_budget", [True, 0, -1, 1.5])
def test_spawn_boundary_rejects_invalid_cpu_budget(cpu_budget) -> None:
    with pytest.raises(ValueError, match="CPU budget"):
        _run_spawned_target(
            worker_target=f"{_SUPPORT}:successful_target",
            worker_parameters={},
            cpu_budget=cpu_budget,
        )


def test_spawn_boundary_rejects_non_json_worker_parameters() -> None:
    with pytest.raises(TypeError, match="closed finite JSON"):
        _run_spawned_target(
            worker_target=f"{_SUPPORT}:successful_target",
            worker_parameters={"not_json": object()},
            cpu_budget=1,
        )


def test_public_spawn_builder_freshly_validates_in_parent_and_stat_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (tmp_path / "cache").resolve()
    monkeypatch.setattr(
        process_module,
        "PRODUCTION_EMBEDDING_CACHE_WORKER_TARGET",
        f"{_SUPPORT}:fake_production_target",
    )

    def validate(**kwargs):
        assert kwargs["cache_dir"] == target
        payload = target / "payload.bin"
        return {
            "cache_path": str(target),
            "cache_files": {
                payload.name: {
                    "sha256": "a" * 64,
                    "size_bytes": payload.stat().st_size,
                }
            },
        }

    monkeypatch.setattr(
        builder_module,
        "validate_published_production_embedding_cache",
        validate,
    )
    dataset = tmp_path / "dataset.parquet"
    model = tmp_path / "model"
    dataset.write_bytes(b"fixture")
    model.mkdir()
    result = (
        process_module.build_production_embedding_cache_in_spawned_worker(
            dataset_path=dataset,
            text_column="text",
            local_model_path=model,
            sentence_model_name="fixture/model",
            chunk_configuration={"fixture": True},
            target_dir=target,
            device="cpu",
            batch_size=2,
            cpu_budget=2,
        )
    )

    assert result.identity()["cache_path"] == str(target)
    assert result.execution_attestation["worker_exit_confirmed"] is True
    assert (
        result.execution_attestation[
            "model_materialized_in_parent_process"
        ]
        is False
    )
    (target / "payload.bin").write_bytes(b"changed-cache!")
    with pytest.raises(
        RuntimeError,
        match="changed after parent authentication",
    ):
        result.identity()


def test_workflow_fresh_cache_phase_uses_spawned_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "prepared.parquet"
    source.write_bytes(b"prepared-cohort")
    preparation_manifest = tmp_path / "preparation_manifest.json"
    preparation_manifest.write_text("{}\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    calls = []

    class Result:
        execution_attestation = {
            "schema_version": "production_embedding_cache_spawn_execution_v1",
            "worker_exit_confirmed": True,
        }

        @staticmethod
        def identity():
            return {"authenticated": "parent"}

    def build(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["target_dir"])
        target.mkdir()
        (target / "metadata.json").write_text("{}\n", encoding="utf-8")
        return Result()

    monkeypatch.setattr(
        process_module,
        "build_production_embedding_cache_in_spawned_worker",
        build,
    )
    workflow = object.__new__(ProductionAllEvidenceWorkflow)
    workflow.options = SimpleNamespace(
        embedding_cache_import=None,
        text_column="configured_text",
        embedding_local_model_path=model,
        embedding_model_name="configured/model",
        stage1_device="cpu",
        embedding_batch_size=7,
        cpu_budget=3,
    )
    workflow._gpu_preflight = lambda: {"status": "accepted"}
    workflow._input_preparation_paths = lambda: (
        source,
        preparation_manifest,
    )
    workflow._embedding_chunk_configuration = lambda: {
        "configured": "lossless"
    }
    attempt = tmp_path / "attempt"
    attempt.mkdir()

    result = workflow._run_embedding_cache_phase(attempt)

    assert len(calls) == 1
    assert calls[0]["cpu_budget"] == 3
    assert calls[0]["batch_size"] == 7
    assert result["cache_identity"] == {"authenticated": "parent"}
    assert result["embedding_model_materialized_in_workflow_process"] is False
    assert result["embedding_model_materialized_in_short_lived_worker"] is True
    assert result["cuda_memory_release_by_worker_exit"] is True
    assert result["embedding_cache_worker_execution"][
        "worker_exit_confirmed"
    ] is True
