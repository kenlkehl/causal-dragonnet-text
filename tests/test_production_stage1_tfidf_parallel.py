from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import oci.inference.production_stage1_tfidf_parallel as parallel_module
from oci.config import AppliedInferenceConfig
from oci.inference.all_evidence_discovery_interfaces import (
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPICS,
)
from oci.inference.production_stage1_tfidf_parallel import (
    CumulativeTfidfScopeTask,
    TFIDF_CUMULATIVE_RESULT_SCHEMA,
    project_tfidf_worker_config,
    run_cumulative_tfidf_scope_tasks,
)
from oci.inference.stage1_cumulative_spent_evidence import (
    CumulativeSpentStage1FamilyRequest,
    cumulative_spent_data_projection_sha256,
)
from oci.inference.stage1_cumulative_spent_native_adapters import (
    CumulativeSpentReplayCanary,
)
from oci.inference.stage1_exact_inner_evidence import Stage1FitRow


FAMILIES = (TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS)


def _request(*, family: str, outer_fold: int, epoch: int):
    base = (outer_fold - 1) * 100 + epoch * 20
    spent = tuple(
        Stage1FitRow(
            row_id=base + index,
            text=f"scope {outer_fold} epoch {epoch} spent note {index}",
            treatment=float(index % 2),
            outcome=float((index // 2) % 2),
        )
        for index in range(12)
    )
    sealed = tuple(base + 12 + index for index in range(4))
    projection = cumulative_spent_data_projection_sha256(
        outer_fold=outer_fold,
        context_epoch=epoch,
        spent_rows=spent,
        sealed_row_ids=sealed,
    )
    return CumulativeSpentStage1FamilyRequest(
        family=family,
        request_sha256="a" * 64,
        schedule_sha256="b" * 64,
        scope_id=f"outer_{outer_fold:03d}_hierarchy_epoch_{epoch:03d}",
        outer_fold=outer_fold,
        context_epoch=epoch,
        provider_inner_fold=epoch + 1,
        split_scope_fingerprint=parallel_module._sha256_json(
            {
                "outer_fold": outer_fold,
                "epoch": epoch,
                "spent": [row.row_id for row in spent],
                "sealed": list(sealed),
            }
        ),
        data_projection_sha256=projection,
        spent_rows=spent,
        sealed_row_ids=sealed,
    )


def _tasks(root: Path) -> tuple[CumulativeTfidfScopeTask, ...]:
    root.mkdir()
    artifacts = root / "artifacts"
    records = root / "records"
    proofs = root / "proofs"
    artifacts.mkdir()
    records.mkdir()
    proofs.mkdir()
    config = project_tfidf_worker_config(AppliedInferenceConfig())
    tasks = []
    canonical_index = 0
    for outer_fold in range(1, 6):
        for epoch in range(2):
            requests = {
                family: _request(
                    family=family,
                    outer_fold=outer_fold,
                    epoch=epoch,
                )
                for family in FAMILIES
            }
            canary = CumulativeSpentReplayCanary.from_request(
                requests[TFIDF_TOPICS]
            )
            scope_id = requests[TFIDF_TOPICS].scope_id
            tasks.append(
                CumulativeTfidfScopeTask(
                    canonical_index=canonical_index,
                    scope_id=scope_id,
                    family_order=FAMILIES,
                    requests=requests,
                    replay_canary=canary,
                    config=config,
                    component_root=root,
                    artifact_dir=artifacts / scope_id,
                    execution_record_dir=records / scope_id,
                    proof_dir=proofs / scope_id,
                )
            )
            canonical_index += 1
    return tuple(tasks)


def _fake_scope_executor(task: CumulativeTfidfScopeTask):
    task.artifact_dir.mkdir()
    task.execution_record_dir.mkdir()
    task.proof_dir.mkdir()
    payload = {
        "scope_id": task.scope_id,
        "canonical_index": task.canonical_index,
        "task_input_identity_sha256": task.input_identity["content_sha256"],
        "spent_row_ids": list(
            task.requests[task.family_order[0]].spent_row_ids
        ),
    }
    for directory in (
        task.artifact_dir,
        task.execution_record_dir,
        task.proof_dir,
    ):
        (directory / "deterministic.json").write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
    registration = {
        "scope_id": task.scope_id,
        "canonical_index": task.canonical_index,
        "data_projection_sha256": task.requests[
            task.family_order[0]
        ].data_projection_sha256,
        "worker_pid": os.getpid(),
    }
    # PID identity is reduced to a Boolean so serial/parallel result bytes stay
    # directly comparable while still proving loky used another process.
    body = {
        "schema_version": TFIDF_CUMULATIVE_RESULT_SCHEMA,
        "canonical_index": task.canonical_index,
        "scope_id": task.scope_id,
        "task_input_identity_sha256": task.input_identity["content_sha256"],
        "registration": registration,
    }
    return {**body, "content_sha256": parallel_module._sha256_json(body)}


def _without_process_marker(results):
    normalized = []
    for result in results:
        body = dict(result)
        registration = dict(body["registration"])
        registration.pop("worker_pid")
        body["registration"] = registration
        body.pop("content_sha256")
        normalized.append(body)
    return normalized


def test_ten_cumulative_scopes_are_loky_parallel_and_equal_to_serial(
    tmp_path: Path,
):
    serial_tasks = _tasks(tmp_path / "serial")
    parallel_tasks = _tasks(tmp_path / "parallel")

    serial = run_cumulative_tfidf_scope_tasks(
        tasks=serial_tasks,
        workers=1,
        executor=_fake_scope_executor,
    )
    parallel = run_cumulative_tfidf_scope_tasks(
        tasks=parallel_tasks,
        workers=4,
        executor=_fake_scope_executor,
    )

    assert len(serial) == len(parallel) == 10
    assert [row["scope_id"] for row in parallel] == [
        task.scope_id for task in parallel_tasks
    ]
    assert all(
        row["registration"]["worker_pid"] != os.getpid()
        for row in parallel
    )
    assert all(
        row["registration"]["worker_pid"] == os.getpid() for row in serial
    )
    assert _without_process_marker(serial) == _without_process_marker(parallel)
    for serial_task, parallel_task in zip(serial_tasks, parallel_tasks):
        for serial_dir, parallel_dir in (
            (serial_task.artifact_dir, parallel_task.artifact_dir),
            (
                serial_task.execution_record_dir,
                parallel_task.execution_record_dir,
            ),
            (serial_task.proof_dir, parallel_task.proof_dir),
        ):
            assert (serial_dir / "deterministic.json").read_bytes() == (
                parallel_dir / "deterministic.json"
            ).read_bytes()


def test_task_plan_and_results_reject_missing_duplicate_reorder_and_alias(
    tmp_path: Path,
):
    tasks = _tasks(tmp_path / "valid")
    with pytest.raises(ValueError, match="duplicated, missing, or reordered"):
        run_cumulative_tfidf_scope_tasks(
            tasks=tuple(reversed(tasks)),
            workers=1,
            executor=_fake_scope_executor,
        )

    alias_root = tmp_path / "alias"
    alias_tasks = list(_tasks(alias_root))
    aliased = alias_tasks[1]
    object.__setattr__(
        aliased,
        "artifact_dir",
        alias_tasks[0].artifact_dir,
    )
    with pytest.raises(ValueError, match="alias an output directory"):
        run_cumulative_tfidf_scope_tasks(
            tasks=alias_tasks,
            workers=1,
            executor=_fake_scope_executor,
        )

    result_root = tmp_path / "results"
    result_tasks = _tasks(result_root)
    results = [
        _fake_scope_executor(task)
        for task in result_tasks
    ]
    with pytest.raises(RuntimeError, match="missing results"):
        parallel_module._validate_results(result_tasks, results[:-1])
    duplicate = list(results)
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="duplicate or invalid"):
        parallel_module._validate_results(result_tasks, duplicate)


def test_task_identity_contains_only_spent_rows_and_sealed_ids(tmp_path: Path):
    task = _tasks(tmp_path / "projection")[0]
    identity = task.input_identity
    request = task.requests[TFIDF_TOPICS]
    nn_config = task.config.architecture.multi_model_forest

    assert identity["spent_row_ids"] == list(request.spent_row_ids)
    assert identity["sealed_row_ids"] == list(request.sealed_row_ids)
    assert identity["cohort_frame_supplied"] is False
    assert identity["source_cohort_locator_supplied"] is False
    serialized = json.dumps(identity)
    assert "dataset_path" not in serialized
    assert "modeling_data" not in serialized
    assert task.config.dataset_path == parallel_module.TFIDF_SPENT_ONLY_DATASET_MARKER
    assert nn_config.prespecified_features_json is None
    assert nn_config.embedding_contrast.cache_dir is None
    assert nn_config.embedding_contrast.external_corpus_cache_dirs == []
