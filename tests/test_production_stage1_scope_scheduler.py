from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import oci.inference.production_stage1_scope_scheduler as scope_scheduler
from oci.inference.production_stage1_scope_scheduler import (
    SpawnedStage1ScopeOrchestrator,
    Stage1PhysicalFitIdentity,
    Stage1ScopePlan,
    Stage1ScopeAttemptStore,
    Stage1ScopeExecutionRequest,
    Stage1ScopeProgressLedger,
    _enforce_stage1_torch_determinism,
    _observe_stage1_torch_determinism,
    build_canonical_stage1_scope_plan,
    derive_stage1_group_seed,
    seed_stage1_scope_rngs,
    stage1_torch_determinism_policy,
    validate_stage1_scope_plan,
)


_REGISTRY_SHA = "a" * 64
_PHYSICAL_FIT_IDENTITY = Stage1PhysicalFitIdentity(
    architecture_identity="1" * 64,
    target="test_all_ten_stage1_context_fit_v1",
    scientific_configuration_identity="2" * 64,
    producer_identity="3" * 64,
    runtime_compatibility_class="test-python-posix-v1",
)


class _SubprocessProcessAdapter:
    def __init__(self, process: subprocess.Popen):
        self._process = process
        self.pid = process.pid

    def is_alive(self):
        return self._process.poll() is None

    def join(self, timeout=None):
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self):
        self._process.terminate()

    def kill(self):
        self._process.kill()


def _registry(*, outer_count: int = 5, inner_count: int = 5) -> dict:
    row_count = outer_count * 20
    all_rows = tuple(range(row_count))
    outers = []
    heldout_size = row_count // outer_count
    for outer_fold in range(1, outer_count + 1):
        start = (outer_fold - 1) * heldout_size
        heldout = tuple(range(start, start + heldout_size))
        fit = tuple(row for row in all_rows if row not in set(heldout))
        partitions = tuple(
            fit[index::inner_count] for index in range(inner_count)
        )
        inner_rows = []
        for inner_fold, inner_heldout in enumerate(partitions, start=1):
            inner_fit = tuple(
                row for row in fit if row not in set(inner_heldout)
            )
            inner_rows.append(
                {
                    "inner_fold": inner_fold,
                    "fit_row_ids": list(inner_fit),
                    "heldout_row_ids": list(inner_heldout),
                }
            )
        outers.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": list(fit),
                "heldout_row_ids": list(heldout),
                "inner_folds": inner_rows,
            }
        )
    return {
        "dataset_row_count": row_count,
        "outer_folds": outers,
    }


def _plan(*, gpu_ids: tuple[int, ...] = (0, 1)):
    registry = _registry()
    return build_canonical_stage1_scope_plan(
        registry=registry,
        registry_content_sha256=_REGISTRY_SHA,
        global_seed=42,
        physical_fit_identity=_PHYSICAL_FIT_IDENTITY,
        gpu_ids=gpu_ids,
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=5,
        expected_inner_fold_count=5,
    )


def test_plan_supports_configured_nonbenchmark_initial_partition_count():
    registry = _registry()
    plan = build_canonical_stage1_scope_plan(
        registry=registry,
        registry_content_sha256=_REGISTRY_SHA,
        global_seed=42,
        physical_fit_identity=_PHYSICAL_FIT_IDENTITY,
        gpu_ids=(),
        review_rounds=3,
        initial_training_partitions=2,
        expected_outer_fold_count=5,
        expected_inner_fold_count=5,
    )
    assert plan.initial_training_partitions == 2
    first = plan.scope("outer_001_hierarchy_epoch_000")
    spent_rows = {
        row_id
        for inner in registry["outer_folds"][0]["inner_folds"][:2]
        for row_id in inner["heldout_row_ids"]
    }
    assert first.fit_row_ids == tuple(
        row_id
        for row_id in registry["outer_folds"][0]["fit_row_ids"]
        if row_id in spent_rows
    )
    assert plan.as_dict()["initial_training_partitions"] == 2


def test_scientific_scope_plan_identity_excludes_gpu_ids_and_assignments():
    cpu = _plan(gpu_ids=())
    heterogeneous_gpu = _plan(gpu_ids=(7, 2, 11))

    assert cpu.content_sha256 != heterogeneous_gpu.content_sha256
    assert (
        cpu.scientific_content_sha256
        == heterogeneous_gpu.scientific_content_sha256
    )
    assert cpu.as_dict()["scientific_content_sha256"] == (
        heterogeneous_gpu.as_dict()["scientific_content_sha256"]
    )
    assert cpu.scopes == heterogeneous_gpu.scopes
    assert cpu.assignments != heterogeneous_gpu.assignments


def test_role_neutral_plan_accepts_positive_operational_concurrency_but_legacy_spawn_does_not(
    tmp_path: Path,
):
    registry = _registry()
    concurrent = build_canonical_stage1_scope_plan(
        registry=registry,
        registry_content_sha256=_REGISTRY_SHA,
        global_seed=42,
        physical_fit_identity=_PHYSICAL_FIT_IDENTITY,
        gpu_ids=(0, 1),
        review_rounds=2,
        initial_training_partitions=3,
        scope_workers_per_gpu=2,
        expected_outer_fold_count=5,
        expected_inner_fold_count=5,
    )
    serial = _plan(gpu_ids=(0, 1))

    assert concurrent.scope_workers_per_gpu == 2
    assert concurrent.content_sha256 != serial.content_sha256
    assert concurrent.scientific_content_sha256 == serial.scientific_content_sha256
    with pytest.raises(ValueError, match="one active scope per GPU"):
        SpawnedStage1ScopeOrchestrator(
            plan=concurrent,
            attempt_root=tmp_path / "attempts",
            progress_path=tmp_path / "progress.json",
            worker_target=f"{__name__}:_spawn_test_worker",
        )


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_scope_plan_rejects_nonpositive_or_noninteger_concurrency(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        build_canonical_stage1_scope_plan(
            registry=_registry(),
            registry_content_sha256=_REGISTRY_SHA,
            global_seed=42,
            physical_fit_identity=_PHYSICAL_FIT_IDENTITY,
            gpu_ids=(0,),
            review_rounds=2,
            initial_training_partitions=3,
            scope_workers_per_gpu=invalid,
            expected_outer_fold_count=5,
            expected_inner_fold_count=5,
        )


def _subset_scope_plan(count: int) -> Stage1ScopePlan:
    base = _plan(gpu_ids=())
    scopes = tuple(base.scopes[: int(count)])
    running_load = 0
    assignments = []
    for execution_rank, scope in enumerate(scopes):
        running_load += scope.fit_row_count
        assignments.append(
            replace(
                base.assignment(scope.scope_id),
                execution_rank=execution_rank,
                assigned_gpu_load_after=running_load,
            )
        )
    body = scope_scheduler._stage1_scope_plan_body(
        registry_content_sha256=base.registry_content_sha256,
        global_seed=base.global_seed,
        review_rounds=base.review_rounds,
        initial_training_partitions=base.initial_training_partitions,
        physical_fit_identity=base.physical_fit_identity,
        gpu_ids=(),
        scope_workers_per_gpu=1,
        scopes=scopes,
        assignments=tuple(assignments),
    )
    return Stage1ScopePlan(
        registry_content_sha256=base.registry_content_sha256,
        global_seed=base.global_seed,
        review_rounds=base.review_rounds,
        initial_training_partitions=base.initial_training_partitions,
        physical_fit_identity=base.physical_fit_identity,
        gpu_ids=(),
        scope_workers_per_gpu=1,
        scopes=scopes,
        assignments=tuple(assignments),
        content_sha256=scope_scheduler._sha256_json(body),
    )


def _single_scope_plan() -> Stage1ScopePlan:
    return _subset_scope_plan(1)


def _torch_unavailable_determinism_observation() -> dict:
    return {
        **stage1_torch_determinism_policy(),
        "torch_available": False,
        "policy_active": True,
    }


def _spawn_test_worker(request: Stage1ScopeExecutionRequest) -> dict:
    output = request.payload_dir / "proof.json"
    output.write_text(
        json.dumps(
            {
                "scope_id": request.scope_id,
                "scope_seed": request.scope_seed,
                "heldout_labels_supplied": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"scope_id": request.scope_id, "proof_file": output.name}


def _hashseed_report_worker(request: Stage1ScopeExecutionRequest) -> dict:
    output = request.payload_dir / "hashseed.json"
    observed = os.environ.get("PYTHONHASHSEED")
    output.write_text(
        json.dumps(
            {
                "scope_id": request.scope_id,
                "scope_seed": request.scope_seed,
                "pythonhashseed": observed,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"pythonhashseed": observed}


def _sealed_attempt(tmp_path: Path):
    plan = _plan(gpu_ids=())
    store = Stage1ScopeAttemptStore(tmp_path / "attempts", plan)
    target = f"{__name__}:_spawn_test_worker"
    parameters = {"request_sha256": "d" * 64}
    request = store.begin(
        scope_id="outer_001_full",
        worker_target=target,
        worker_parameters=parameters,
    )
    request.payload_dir.mkdir()
    (request.payload_dir / "proof.json").write_text("proof", encoding="utf-8")
    store.seal(request, worker_result={"ok": True})
    return store, target, parameters, request


def test_fixed_profile_plan_has_5_full_25_inner_10_cumulative_and_balanced_gpus():
    plan = _plan()

    assert len(plan.scopes) == 40
    assert Counter(scope.scope_kind for scope in plan.scopes) == {
        "full_outer": 5,
        "exact_inner": 25,
        "cumulative_spent": 10,
    }
    assert [scope.scope_id for scope in plan.scopes[:7]] == [
        "outer_001_full",
        "outer_001_inner_001",
        "outer_001_inner_002",
        "outer_001_inner_003",
        "outer_001_inner_004",
        "outer_001_inner_005",
        "outer_002_full",
    ]
    assignment_counts = {
        gpu_id: Counter(
            plan.scope(assignment.scope_id).scope_kind
            for assignment in plan.assignments
            if assignment.gpu_id == gpu_id
        )
        for gpu_id in (0, 1)
    }
    assert assignment_counts == {
        0: Counter(
            {"full_outer": 3, "exact_inner": 12, "cumulative_spent": 5}
        ),
        1: Counter(
            {"full_outer": 2, "exact_inner": 13, "cumulative_spent": 5}
        ),
    }
    loads = {
        gpu_id: sum(
            assignment.fit_row_count
            for assignment in plan.assignments
            if assignment.gpu_id == gpu_id
        )
        for gpu_id in (0, 1)
    }
    assert loads == {0: 1280, 1: 1280}
    assert (
        plan.as_dict()["torch_determinism_policy"]
        == stage1_torch_determinism_policy()
    )
    assert (
        plan.as_dict()["torch_determinism_policy"]["failure_policy"]
        == "abort_scope_on_unsupported_nondeterministic_operation"
    )


def test_plan_is_schedule_independent_closed_and_contains_no_labels():
    registry = _registry()
    first = _plan()
    second = _plan()

    assert first.as_dict() == second.as_dict()
    assert len(first.scopes) == 40
    assert len(first.physical_scopes) == 35
    assert sum(
        len(members) == 2
        for _owner, members in first.physical_scope_groups
    ) == 5
    assert all(
        first.physical_owner(
            f"outer_{outer_fold:03d}_hierarchy_epoch_001"
        ).scope_id
        == f"outer_{outer_fold:03d}_inner_005"
        for outer_fold in range(1, 6)
    )
    assert len({scope.scope_seed for scope in first.scopes}) == 35
    assert (
        derive_stage1_group_seed(42, first.scopes[0].fit_row_ids)
        == first.scopes[0].scope_seed
    )
    assert "treatment" not in json.dumps(first.as_dict()).casefold()
    assert "outcome" not in json.dumps(first.as_dict()).casefold()
    validated = validate_stage1_scope_plan(
        first.as_dict(),
        registry=registry,
        registry_content_sha256=_REGISTRY_SHA,
        global_seed=42,
        physical_fit_identity=_PHYSICAL_FIT_IDENTITY,
        gpu_ids=(0, 1),
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=5,
        expected_inner_fold_count=5,
    )
    assert validated.as_dict() == first.as_dict()

    tampered = json.loads(json.dumps(first.as_dict()))
    tampered["scopes"][0]["fit_row_ids"].reverse()
    with pytest.raises(ValueError, match="changed or was substituted"):
        validate_stage1_scope_plan(
            tampered,
            registry=registry,
            registry_content_sha256=_REGISTRY_SHA,
            global_seed=42,
            physical_fit_identity=_PHYSICAL_FIT_IDENTITY,
            gpu_ids=(0, 1),
            review_rounds=2,
            initial_training_partitions=3,
            expected_outer_fold_count=5,
            expected_inner_fold_count=5,
        )


def test_immutable_plan_memoizes_exact_scientific_identity_and_fit_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(gpu_ids=(0, 1))
    expected_groups = scope_scheduler._physical_fit_groups(
        scopes=plan.scopes,
        physical_fit_identity=plan.physical_fit_identity,
    )
    expected_scientific_identity = scope_scheduler._sha256_json(
        scope_scheduler._stage1_scope_scientific_plan_body(
            registry_content_sha256=plan.registry_content_sha256,
            global_seed=plan.global_seed,
            review_rounds=plan.review_rounds,
            initial_training_partitions=(
                plan.initial_training_partitions
            ),
            physical_fit_identity=plan.physical_fit_identity,
            scopes=plan.scopes,
        )
    )
    original = scope_scheduler._physical_fit_groups
    calls = 0

    def counted_physical_fit_groups(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(
        scope_scheduler,
        "_physical_fit_groups",
        counted_physical_fit_groups,
    )

    assert plan.scientific_content_sha256 == expected_scientific_identity
    assert plan.scientific_content_sha256 == expected_scientific_identity
    assert plan.physical_fit_groups == expected_groups
    assert plan.physical_fit_groups == expected_groups
    assert calls == 2

    first_record = plan.as_dict()
    second_record = plan.as_dict()
    assert first_record == second_record
    assert first_record["scientific_content_sha256"] == (
        expected_scientific_identity
    )
    assert calls == 2


def test_physical_fit_key_binds_every_scientific_axis_and_excludes_resources():
    cpu = _plan(gpu_ids=())
    gpu = _plan(gpu_ids=(9, 4))
    owner_id = cpu.physical_scopes[0].scope_id
    baseline = cpu.physical_fit_key(owner_id)

    assert baseline == gpu.physical_fit_key(owner_id)
    assert baseline.as_dict() == (
        cpu.as_dict()["physical_fit_groups"][0][
            "physical_fit_key_record"
        ]
    )
    assert cpu.as_dict()["physical_fit_groups"][0][
        "physical_fit_key"
    ] == baseline.key
    assert set(baseline.as_dict()) == {
        "schema_version",
        "architecture_identity",
        "target",
        "fit_row_order_identity",
        "scientific_configuration_identity",
        "canonical_group_seed",
        "producer_identity",
        "runtime_compatibility_class",
        "content_sha256",
    }

    mutations = (
        replace(
            _PHYSICAL_FIT_IDENTITY,
            architecture_identity="4" * 64,
        ),
        replace(
            _PHYSICAL_FIT_IDENTITY,
            target="test_all_ten_stage1_context_fit_v2",
        ),
        replace(
            _PHYSICAL_FIT_IDENTITY,
            scientific_configuration_identity="5" * 64,
        ),
        replace(
            _PHYSICAL_FIT_IDENTITY,
            producer_identity="6" * 64,
        ),
        replace(
            _PHYSICAL_FIT_IDENTITY,
            runtime_compatibility_class="test-python-posix-v2",
        ),
    )
    for changed_identity in mutations:
        changed = build_canonical_stage1_scope_plan(
            registry=_registry(),
            registry_content_sha256=_REGISTRY_SHA,
            global_seed=42,
            physical_fit_identity=changed_identity,
            gpu_ids=(),
            review_rounds=2,
            initial_training_partitions=3,
            expected_outer_fold_count=5,
            expected_inner_fold_count=5,
        )
        assert changed.physical_fit_key(owner_id).key != baseline.key
        assert (
            changed.scientific_content_sha256
            != cpu.scientific_content_sha256
        )
        assert len(changed.physical_scopes) == 35

    owner = cpu.scope(owner_id)
    reordered = replace(
        owner,
        fit_row_ids=tuple(reversed(owner.fit_row_ids)),
    )
    changed_seed = replace(owner, scope_seed=owner.scope_seed + 1)
    assert (
        cpu.physical_fit_identity.key_for_scope(reordered).key
        != baseline.key
    )
    assert (
        cpu.physical_fit_identity.key_for_scope(changed_seed).key
        != baseline.key
    )


def test_physical_group_rejects_seed_drift_and_uses_earliest_content_owner():
    plan = _plan(gpu_ids=())
    alias_id = "outer_001_hierarchy_epoch_001"
    alias = plan.scope(alias_id)
    changed_scopes = tuple(
        replace(scope, scope_seed=scope.scope_seed + 1)
        if scope.scope_id == alias_id
        else scope
        for scope in plan.scopes
    )
    changed = replace(plan, scopes=changed_scopes)
    with pytest.raises(ValueError, match="canonical group seed"):
        _ = changed.physical_fit_groups

    reordered = replace(plan, scopes=tuple(reversed(plan.scopes)))
    assert (
        reordered.physical_owner(alias_id).scope_id
        == "outer_001_inner_005"
    )


def test_plan_rejects_reordered_or_incomplete_partitions():
    registry = _registry()
    registry["outer_folds"][0]["inner_folds"][0]["inner_fold"] = 2
    with pytest.raises(ValueError, match="missing or reordered"):
        build_canonical_stage1_scope_plan(
            registry=registry,
            registry_content_sha256=_REGISTRY_SHA,
            global_seed=42,
            physical_fit_identity=_PHYSICAL_FIT_IDENTITY,
            review_rounds=2,
            initial_training_partitions=3,
        )

    registry = _registry()
    registry["outer_folds"][0]["heldout_row_ids"][0] = 99
    with pytest.raises(ValueError, match="overlap|does not partition"):
        build_canonical_stage1_scope_plan(
            registry=registry,
            registry_content_sha256=_REGISTRY_SHA,
            global_seed=42,
            physical_fit_identity=_PHYSICAL_FIT_IDENTITY,
            review_rounds=2,
            initial_training_partitions=3,
        )


def test_attempt_resume_accepts_only_terminal_matching_manifest(tmp_path: Path):
    plan = _plan(gpu_ids=())
    store = Stage1ScopeAttemptStore(tmp_path / "attempts", plan)
    target = f"{__name__}:_spawn_test_worker"
    parameters = {"request_sha256": "b" * 64}
    request = store.begin(
        scope_id="outer_001_full",
        worker_target=target,
        worker_parameters=parameters,
    )
    request.payload_dir.mkdir()
    (request.payload_dir / "proof.json").write_text("proof", encoding="utf-8")
    sealed = store.seal(request, worker_result={"ok": True})
    assert sealed["torch_determinism_policy"] == stage1_torch_determinism_policy()
    assert sealed["torch_determinism_observed"]["policy_active"] is True

    assert (
        store.reusable(
            scope_id="outer_001_full",
            worker_target=target,
            worker_parameters=parameters,
        )
        == sealed
    )
    interrupted = store.begin(
        scope_id="outer_001_full",
        worker_target=target,
        worker_parameters=parameters,
    )
    interrupted.payload_dir.mkdir()
    (interrupted.payload_dir / "partial").write_text("partial", encoding="utf-8")
    assert (
        store.reusable(
            scope_id="outer_001_full",
            worker_target=target,
            worker_parameters=parameters,
        )
        == sealed
    )

    (Path(request.attempt_dir) / "payload" / "proof.json").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="inventory changed"):
        store.reusable(
            scope_id="outer_001_full",
            worker_target=target,
            worker_parameters=parameters,
        )


def test_torch_determinism_policy_is_strict_and_policy_weakening_is_detected():
    observed = _enforce_stage1_torch_determinism()
    assert observed["policy_active"] is True

    import torch

    torch.backends.cudnn.benchmark = True
    assert _observe_stage1_torch_determinism()["policy_active"] is False
    assert _enforce_stage1_torch_determinism()["policy_active"] is True


def test_scope_rng_seeding_touches_only_the_selected_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def forbidden_all_devices(_seed):
        pytest.fail("scope seeding touched every visible CUDA device")

    fake_torch = SimpleNamespace(
        default_generator=SimpleNamespace(
            manual_seed=lambda seed: calls.append(("cpu", seed))
        ),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            set_device=lambda gpu_id: calls.append(("set_device", gpu_id)),
            manual_seed=lambda seed: calls.append(("cuda_current", seed)),
            manual_seed_all=forbidden_all_devices,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    seed_stage1_scope_rngs(123, gpu_id=1)

    assert calls == [
        ("cpu", 123),
        ("set_device", 1),
        ("cuda_current", 123),
    ]


def test_attempt_request_rejects_heldout_labels(tmp_path: Path):
    store = Stage1ScopeAttemptStore(tmp_path / "attempts", _plan(gpu_ids=()))
    with pytest.raises(ValueError, match="forbidden"):
        store.begin(
            scope_id="outer_001_full",
            worker_target=f"{__name__}:_spawn_test_worker",
            worker_parameters={"heldout_outcome": [0, 1]},
        )


def test_attempt_validator_rejects_duplicate_json_keys(tmp_path: Path):
    store, target, parameters, request = _sealed_attempt(tmp_path)
    manifest_path = Path(request.attempt_dir) / "attempt_manifest.json"
    text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        text.replace(
            '"schema_version":',
            '"schema_version": "duplicate",\n  "schema_version":',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        store.validate_completed(
            request.attempt_dir,
            scope_id=request.scope_id,
            worker_target=target,
            worker_parameters=parameters,
        )


def test_attempt_validator_rejects_hardlinks_and_special_files(tmp_path: Path):
    hardlink_root = tmp_path / "hardlink"
    store, target, parameters, request = _sealed_attempt(hardlink_root)
    result_path = Path(request.attempt_dir) / "worker_result.json"
    (Path(request.attempt_dir) / "worker_result.alias").hardlink_to(result_path)
    with pytest.raises(ValueError, match="hard link"):
        store.validate_completed(
            request.attempt_dir,
            scope_id=request.scope_id,
            worker_target=target,
            worker_parameters=parameters,
        )

    special_root = tmp_path / "special"
    store, target, parameters, request = _sealed_attempt(special_root)
    result_path = Path(request.attempt_dir) / "worker_result.json"
    result_path.unlink()
    os.mkfifo(result_path)
    with pytest.raises(ValueError, match="special files"):
        store.validate_completed(
            request.attempt_dir,
            scope_id=request.scope_id,
            worker_target=target,
            worker_parameters=parameters,
        )


def test_attempt_validator_rejects_noncanonical_root_escape(tmp_path: Path):
    store, target, parameters, request = _sealed_attempt(tmp_path)
    with pytest.raises(ValueError, match="outside its canonical"):
        store.validate_completed(
            tmp_path,
            scope_id=request.scope_id,
            worker_target=target,
            worker_parameters=parameters,
        )


def test_attempt_validator_rejects_atomic_same_path_directory_substitution(
    tmp_path: Path,
):
    store, target, parameters, request = _sealed_attempt(tmp_path)
    attempt = Path(request.attempt_dir)
    preserved = tmp_path / "preserved-original-attempt"
    attempt.rename(preserved)
    shutil.copytree(preserved, attempt)

    with pytest.raises(
        ValueError,
        match="atomically substituted|invalid binding",
    ):
        store.validate_completed(
            attempt,
            scope_id=request.scope_id,
            worker_target=target,
            worker_parameters=parameters,
        )


def test_attempt_store_and_progress_are_bound_to_stable_execution_paths(
    tmp_path: Path,
):
    store, target, parameters, request = _sealed_attempt(tmp_path / "source")
    copied_root = tmp_path / "copied" / "attempts"
    copied_attempt = (
        copied_root / request.scope_id / Path(request.attempt_dir).name
    )
    copied_attempt.parent.mkdir(parents=True)
    shutil.copytree(request.attempt_dir, copied_attempt)
    copied_store = Stage1ScopeAttemptStore(copied_root, store.plan)
    with pytest.raises(ValueError, match="invalid binding"):
        copied_store.validate_completed(
            copied_attempt,
            scope_id=request.scope_id,
            worker_target=target,
            worker_parameters=parameters,
        )

    progress_path = tmp_path / "progress-bound.json"
    SpawnedStage1ScopeOrchestrator(
        plan=store.plan,
        attempt_root=tmp_path / "orchestrated-attempts",
        progress_path=progress_path,
        worker_target=target,
        worker_parameters={"request_sha256": "1" * 64},
    )
    with pytest.raises(ValueError, match="invalid binding"):
        SpawnedStage1ScopeOrchestrator(
            plan=store.plan,
            attempt_root=tmp_path / "orchestrated-attempts",
            progress_path=progress_path,
            worker_target=target,
            worker_parameters={"request_sha256": "2" * 64},
        )


def test_per_scope_worker_parameters_are_closed_and_child_request_is_private(
    tmp_path: Path,
):
    plan = _plan(gpu_ids=())
    target = f"{__name__}:_spawn_test_worker"
    parameters = {
        scope.scope_id: {
            "descriptor_manifest_path": (
                f"/private/descriptors/{scope.scope_id}/manifest.json"
            ),
            "stage1_request_sha256": "e" * 64,
        }
        for scope in plan.physical_scopes
    }
    orchestrator = SpawnedStage1ScopeOrchestrator(
        plan=plan,
        attempt_root=tmp_path / "attempts",
        progress_path=tmp_path / "progress.json",
        worker_target=target,
        worker_parameters_by_scope=parameters,
    )
    selected_scope = plan.scopes[0].scope_id
    selected = orchestrator.worker_parameters_for_scope(selected_scope)
    request = orchestrator.store.begin(
        scope_id=selected_scope,
        worker_target=target,
        worker_parameters=selected,
    )
    persisted = json.loads(
        (Path(request.attempt_dir) / "attempt_request.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["worker_parameters"] == parameters[selected_scope]
    assert parameters[plan.scopes[1].scope_id]["descriptor_manifest_path"] not in (
        json.dumps(persisted)
    )
    assert "/private/descriptors/" not in (
        tmp_path / "progress.json"
    ).read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="mutually exclusive"):
        SpawnedStage1ScopeOrchestrator(
            plan=plan,
            attempt_root=tmp_path / "both-attempts",
            progress_path=tmp_path / "both-progress.json",
            worker_target=target,
            worker_parameters={},
            worker_parameters_by_scope=parameters,
        )
    incomplete = dict(parameters)
    incomplete.pop(selected_scope)
    with pytest.raises(ValueError, match="cover exactly"):
        SpawnedStage1ScopeOrchestrator(
            plan=plan,
            attempt_root=tmp_path / "missing-attempts",
            progress_path=tmp_path / "missing-progress.json",
            worker_target=target,
            worker_parameters_by_scope=incomplete,
        )
    extra = dict(parameters)
    extra["outer_999_full"] = {"stage1_request_sha256": "f" * 64}
    with pytest.raises(ValueError, match="cover exactly"):
        SpawnedStage1ScopeOrchestrator(
            plan=plan,
            attempt_root=tmp_path / "extra-attempts",
            progress_path=tmp_path / "extra-progress.json",
            worker_target=target,
            worker_parameters_by_scope=extra,
        )

    # Mapping insertion order is operationally irrelevant: normalization is
    # always against canonical scope order and therefore has one binding.
    reversed_parameters = dict(reversed(tuple(parameters.items())))
    SpawnedStage1ScopeOrchestrator(
        plan=plan,
        attempt_root=tmp_path / "attempts",
        progress_path=tmp_path / "progress.json",
        worker_target=target,
        worker_parameters_by_scope=reversed_parameters,
    )
    substituted = {
        scope_id: dict(values) for scope_id, values in parameters.items()
    }
    substituted[selected_scope]["stage1_request_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="invalid binding"):
        SpawnedStage1ScopeOrchestrator(
            plan=plan,
            attempt_root=tmp_path / "attempts",
            progress_path=tmp_path / "progress.json",
            worker_target=target,
            worker_parameters_by_scope=substituted,
        )


def test_progress_ledger_is_plan_bound_and_completed_is_terminal(tmp_path: Path):
    plan = _plan(gpu_ids=())
    ledger = Stage1ScopeProgressLedger(tmp_path / "progress.json", plan)
    ledger.update("outer_001_full", "running", pid=123)
    ledger.update("outer_001_full", "sealing")
    ledger.update("outer_001_full", "completed", output_bytes=5)

    snapshot = ledger.snapshot()
    assert snapshot["counts"]["completed"] == 1
    assert snapshot["counts"]["pending"] == 39
    with pytest.raises(RuntimeError, match="cannot change"):
        ledger.update("outer_001_full", "running")

    raw = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    raw["plan_content_sha256"] = "f" * 64
    (tmp_path / "progress.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid binding"):
        Stage1ScopeProgressLedger(tmp_path / "progress.json", plan)


def test_resume_reconciles_a_sealed_attempt_left_running_in_progress(
    tmp_path: Path,
):
    plan = _plan(gpu_ids=())
    target = f"{__name__}:_spawn_test_worker"
    parameters = {"request_sha256": "7" * 64}
    orchestrator = SpawnedStage1ScopeOrchestrator(
        plan=plan,
        attempt_root=tmp_path / "attempts",
        progress_path=tmp_path / "progress.json",
        worker_target=target,
        worker_parameters=parameters,
    )
    request = orchestrator.store.begin(
        scope_id="outer_001_full",
        worker_target=target,
        worker_parameters=parameters,
    )
    request.payload_dir.mkdir()
    (request.payload_dir / "proof.json").write_text(
        "durable", encoding="utf-8"
    )
    orchestrator.ledger.update(
        request.scope_id,
        "running",
        attempt_dir=request.attempt_dir,
        pid=12345,
    )
    orchestrator.store.seal(request, worker_result={"ok": True})

    authenticated = orchestrator.store.reusable_attempt(
        scope_id=request.scope_id,
        worker_target=target,
        worker_parameters=parameters,
    )
    assert authenticated is not None
    orchestrator.ledger.reconcile_authenticated_completion(authenticated)
    row = next(
        row
        for row in orchestrator.ledger.snapshot()["scopes"]
        if row["scope_id"] == request.scope_id
    )
    assert row["status"] == "completed"
    assert row["attempt_dir"] == request.attempt_dir
    assert row["pid"] is None
    assert row["failure"] is None
    assert row["output_bytes"] > 0


def test_keyboard_interrupt_terminates_and_joins_every_active_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _plan(gpu_ids=(0, 1))
    target = f"{__name__}:_spawn_test_worker"
    processes = []

    class InterruptingQueue:
        closed = False
        joined = False

        def get(self, *, timeout):
            del timeout
            raise KeyboardInterrupt

        def close(self):
            self.closed = True

        def join_thread(self):
            self.joined = True

    messages = InterruptingQueue()

    class FakeProcess:
        def __init__(self, *, target, args, name):
            del target, args
            self.name = name
            self.pid = 20000 + len(processes)
            self.exitcode = None
            self.alive = False
            self.terminated = False
            self.joined = False
            self.killed = False
            processes.append(self)

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False
            self.exitcode = -15

        def join(self, timeout=None):
            del timeout
            self.joined = True

        def kill(self):
            self.killed = True
            self.alive = False
            self.exitcode = -9

    class FakeContext:
        def Queue(self):
            return messages

        def Process(self, *, target, args, name):
            return FakeProcess(target=target, args=args, name=name)

    monkeypatch.setattr(
        scope_scheduler.mp,
        "get_context",
        lambda start_method: (
            FakeContext()
            if start_method == "spawn"
            else pytest.fail("non-spawn context requested")
        ),
    )
    orchestrator = SpawnedStage1ScopeOrchestrator(
        plan=plan,
        attempt_root=tmp_path / "attempts",
        progress_path=tmp_path / "progress.json",
        worker_target=target,
        worker_parameters={"request_sha256": "8" * 64},
    )
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run()

    assert len(processes) == 2
    assert all(process.terminated for process in processes)
    assert all(process.joined for process in processes)
    assert not any(process.is_alive() for process in processes)
    assert messages.closed is True
    assert messages.joined is True
    failed_rows = [
        row
        for row in orchestrator.ledger.snapshot()["scopes"]
        if row["status"] == "failed"
    ]
    assert len(failed_rows) == 2
    assert all(
        row["failure"]["exception_type"] == "ParentInterruption"
        for row in failed_rows
    )


def test_worker_never_mutates_attempt_after_terminal_manifest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _single_scope_plan()
    store = Stage1ScopeAttemptStore(tmp_path / "attempts", plan)
    target = f"{__name__}:_spawn_test_worker"
    parameters = {"request_sha256": "6" * 64}
    request = store.begin(
        scope_id=plan.scopes[0].scope_id,
        worker_target=target,
        worker_parameters=parameters,
    )
    observed = _torch_unavailable_determinism_observation()
    fake_torch = SimpleNamespace(
        set_num_threads=lambda _count: None,
        set_num_interop_threads=lambda _count: None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        scope_scheduler,
        "_enforce_stage1_torch_determinism",
        lambda: observed,
    )
    monkeypatch.setattr(
        scope_scheduler,
        "_observe_stage1_torch_determinism",
        lambda: observed,
    )
    monkeypatch.setattr(
        scope_scheduler,
        "seed_stage1_scope_rngs",
        lambda _seed, *, gpu_id=None: None,
    )
    monkeypatch.setattr(
        scope_scheduler,
        "_resolve_worker_target",
        lambda _target: (lambda _request: {"ok": True}),
    )
    monkeypatch.setattr(
        scope_scheduler,
        "_establish_worker_process_group",
        lambda _marker_path=None: None,
    )

    class CompletionTransportFailure:
        def put(self, message):
            if message.get("event") == "completed":
                raise RuntimeError("completion queue unavailable")

    with pytest.raises(RuntimeError, match="completion queue unavailable"):
        scope_scheduler._spawned_scope_worker(
            request,
            CompletionTransportFailure(),
        )

    attempt_dir = Path(request.attempt_dir)
    assert (attempt_dir / "attempt_manifest.json").is_file()
    assert not (attempt_dir / "failure.json").exists()
    store.validate_completed(
        attempt_dir,
        scope_id=request.scope_id,
        worker_target=target,
        worker_parameters=parameters,
    )


@pytest.mark.parametrize("publish_terminal", (True, False))
def test_clean_exit_without_queue_completion_is_bounded_and_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publish_terminal: bool,
):
    plan = _single_scope_plan()
    observed = _torch_unavailable_determinism_observation()
    processes = []

    class EmptyQueue:
        def get(self, *, timeout):
            del timeout
            raise scope_scheduler.Empty

        def close(self):
            return None

        def join_thread(self):
            return None

    messages = EmptyQueue()

    class CleanExitProcess:
        def __init__(self, *, target, args, name):
            del target
            self.request = args[0]
            self.name = name
            self.pid = 31001
            self.exitcode = None
            self.alive = False
            processes.append(self)

        def start(self):
            if publish_terminal:
                self.request.payload_dir.mkdir()
                scope_scheduler._seal_scope_attempt(
                    self.request,
                    worker_result={"ok": True},
                    torch_determinism_observed=observed,
                )
            self.exitcode = 0

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            del timeout

        def terminate(self):
            self.alive = False

        def kill(self):
            self.alive = False

    class FakeContext:
        def Queue(self):
            return messages

        def Process(self, *, target, args, name):
            return CleanExitProcess(target=target, args=args, name=name)

    monkeypatch.setattr(
        scope_scheduler.mp,
        "get_context",
        lambda start_method: (
            FakeContext()
            if start_method == "spawn"
            else pytest.fail("non-spawn context requested")
        ),
    )
    orchestrator = SpawnedStage1ScopeOrchestrator(
        plan=plan,
        attempt_root=tmp_path / "attempts",
        progress_path=tmp_path / "progress.json",
        worker_target=f"{__name__}:_spawn_test_worker",
        worker_parameters={"request_sha256": "5" * 64},
        poll_interval_seconds=0.001,
        post_exit_message_grace_seconds=0.01,
    )
    if publish_terminal:
        results = orchestrator.run()
        assert len(results) == 1
        row = orchestrator.ledger.snapshot()["scopes"][0]
        assert row["status"] == "completed"
        assert row["elapsed_seconds"] is None
        assert row["peak_gpu_allocated_bytes"] is None
        assert row["throughput_fit_rows_per_second"] is None
    else:
        with pytest.raises(
            RuntimeError,
            match="terminal attempt failed authentication",
        ):
            orchestrator.run()
        assert orchestrator.ledger.snapshot()["scopes"][0]["status"] == "failed"
    assert len(processes) == 1


def test_operational_paths_are_bound_but_do_not_change_scientific_plan_hash(
    tmp_path: Path,
):
    plan_a = _plan(gpu_ids=(0, 1))
    plan_b = _plan(gpu_ids=(0, 1))
    store_a = Stage1ScopeAttemptStore(tmp_path / "recovery-a", plan_a)
    store_b = Stage1ScopeAttemptStore(tmp_path / "recovery-b", plan_b)

    assert plan_a.content_sha256 == plan_b.content_sha256
    assert store_a.identity()["plan_content_sha256"] == plan_a.content_sha256
    assert store_b.identity()["plan_content_sha256"] == plan_b.content_sha256
    assert store_a.identity()["content_sha256"] != store_b.identity()["content_sha256"]


def test_spawn_children_receive_scope_hash_seed_and_parent_env_is_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _subset_scope_plan(2)
    monkeypatch.setenv("PYTHONHASHSEED", "314159")
    orchestrator = SpawnedStage1ScopeOrchestrator(
        plan=plan,
        attempt_root=tmp_path / "attempts",
        progress_path=tmp_path / "progress.json",
        worker_target=f"{__name__}:_hashseed_report_worker",
        worker_parameters={"request_sha256": "4" * 64},
        poll_interval_seconds=0.01,
    )

    attempts = orchestrator.run()

    assert os.environ["PYTHONHASHSEED"] == "314159"
    assert len(attempts) == 2
    for attempt, scope in zip(attempts, plan.scopes):
        report = json.loads(
            (
                attempt.attempt_dir / "payload" / "hashseed.json"
            ).read_text(encoding="utf-8")
        )
        assert report == {
            "scope_id": scope.scope_id,
            "scope_seed": scope.scope_seed,
            "pythonhashseed": str(scope.scope_seed),
        }


def test_spawn_orchestrator_seals_scopes_and_resume_does_not_add_attempts(
    tmp_path: Path,
):
    # Two outer folds with four inner partitions give twelve small CPU scopes.
    registry = _registry(outer_count=2, inner_count=4)
    plan = build_canonical_stage1_scope_plan(
        registry=registry,
        registry_content_sha256=_REGISTRY_SHA,
        global_seed=42,
        physical_fit_identity=_PHYSICAL_FIT_IDENTITY,
        gpu_ids=(),
        review_rounds=1,
        initial_training_partitions=3,
        expected_outer_fold_count=2,
        expected_inner_fold_count=4,
    )
    target = f"{__name__}:_spawn_test_worker"
    orchestrator = SpawnedStage1ScopeOrchestrator(
        plan=plan,
        attempt_root=tmp_path / "attempts",
        progress_path=tmp_path / "progress.json",
        worker_target=target,
        worker_parameters={"request_sha256": "c" * 64},
        poll_interval_seconds=0.01,
    )
    manifests = orchestrator.run()

    assert len(plan.scopes) == 12
    assert len(plan.physical_scopes) == 10
    assert len(manifests) == 10
    assert orchestrator.ledger.snapshot()["counts"]["completed"] == 12
    logical_bindings = orchestrator.store.validate_logical_bindings()
    assert logical_bindings.manifest["logical_scope_count"] == 12
    assert logical_bindings.manifest["physical_fit_count"] == 10
    assert logical_bindings.manifest["deduplicated_fit_count"] == 2
    before = sorted((tmp_path / "attempts").glob("*/*"))
    replayed = orchestrator.run()
    after = sorted((tmp_path / "attempts").glob("*/*"))
    assert replayed == manifests
    assert after == before

    binding_path = (
        tmp_path / "attempts" / scope_scheduler.STAGE1_LOGICAL_SCOPE_BINDING_FILENAME
    )
    tampered = json.loads(binding_path.read_text(encoding="utf-8"))
    reused = next(
        row for row in tampered["logical_bindings"] if row["reuses_physical_fit"]
    )
    reused["physical_owner_scope_id"] = plan.physical_scopes[0].scope_id
    reused_body = {
        key: value for key, value in reused.items() if key != "content_sha256"
    }
    reused["content_sha256"] = scope_scheduler._sha256_json(reused_body)
    top_body = {
        key: value
        for key, value in tampered.items()
        if key != "content_sha256"
    }
    tampered["content_sha256"] = scope_scheduler._sha256_json(top_body)
    binding_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        orchestrator.store.validate_logical_bindings()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_peer_abort_terminates_worker_and_its_spawned_descendant(
    tmp_path: Path,
):
    descendant_pid_path = tmp_path / "descendant.pid"
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib,subprocess,sys,time;"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(300)']);"
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid));"
                "time.sleep(300)"
            ),
            str(descendant_pid_path),
        ],
        start_new_session=True,
    )
    process = _SubprocessProcessAdapter(leader)
    try:
        deadline = time.monotonic() + 10.0
        while (
            not descendant_pid_path.is_file()
        ) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert descendant_pid_path.is_file()
        descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))

        scope_scheduler._terminate_process_and_descendants(
            process,
            timeout_seconds=5.0,
        )
    finally:
        if process.is_alive():
            scope_scheduler._terminate_process_and_descendants(
                process,
                timeout_seconds=5.0,
            )

    assert not process.is_alive()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        stat_path = Path(f"/proc/{descendant_pid}/stat")
        if not stat_path.exists():
            break
        # A killed descendant can briefly remain as a zombie while its new
        # parent reaps it; it is no longer executing or holding resources.
        state = stat_path.read_text(encoding="utf-8").split()[2]
        if state == "Z":
            break
        time.sleep(0.02)
    else:
        pytest.fail("spawned descendant survived worker-group termination")
