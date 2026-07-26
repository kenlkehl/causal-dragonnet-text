from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import oci.inference.production_stage1_role_neutral_execution as execution_module
from oci.inference.neural_query_operational_controls import (
    ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA,
    RoleNeutralNeuralQueryOperationalControls,
)
from oci.inference.production_stage1_role_neutral_execution import (
    ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_POLICY_SCHEMA,
    NeuralQueryExecutionTopology,
    RoleNeutralFirstOwnerValidationPolicy,
    RoleNeutralOperationalComponentReport,
    RoleNeutralStage1ExecutionPolicy,
    execute_and_publish_role_neutral_stage1,
)
from oci.inference.role_neutral_htr_group_execution import (
    RoleNeutralHTROperationalControls,
)
from tests.test_production_stage1_role_neutral_execution import (
    _ProducerRecorder,
    _RecordingExecutor,
    _plan,
    _resource_plan,
    _sha,
)


class _CallRecordingExecutor(_RecordingExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, ...]] = []

    def execute(self, *, tasks, worker, max_workers, cpu_budget):
        self.calls.append(
            tuple(task.physical_owner.scope_id for task in tasks)
        )
        return super().execute(
            tasks=tasks,
            worker=worker,
            max_workers=max_workers,
            cpu_budget=cpu_budget,
        )


class _GateProducerRecorder(_ProducerRecorder):
    """Supply the typed reports required at the production boundary."""

    def factory(self, expected_component: str):
        base_factory = super().factory(expected_component)

        def bind(invocation):
            bound = base_factory(invocation)
            if expected_component not in {
                "matched_pair",
                "neural_query",
            }:
                return bound

            def execute():
                bound.execute()
                body = {
                    "schema_version": (
                        "test_first_owner_operational_attestation_v1"
                    ),
                    "component": expected_component,
                    "physical_owner_scope_id": (
                        invocation.physical_owner.scope_id
                    ),
                }
                return RoleNeutralOperationalComponentReport(
                    component=expected_component,
                    attestation={
                        **body,
                        "content_sha256": _sha(body),
                    },
                )

            return replace(bound, execute=execute)

        return bind


def _policy(
    *,
    minimum_headroom_bytes: int,
    maximum_allocation_fraction: float = 0.8,
) -> RoleNeutralStage1ExecutionPolicy:
    devices = ("cuda:3", "cuda:8")
    return RoleNeutralStage1ExecutionPolicy(
        resource_plan=_resource_plan(
            devices=devices,
            cpu_budget=8,
        ),
        max_parallel_owners=1,
        neural_query_execution_topologies={
            "cuda:3": NeuralQueryExecutionTopology(
                devices=("cuda:3", "cuda:8"),
            ),
            "cuda:8": NeuralQueryExecutionTopology(
                devices=("cuda:8", "cuda:3"),
            ),
        },
        htr_operational_controls=RoleNeutralHTROperationalControls(
            training_batch_size=4,
            sentence_encoder_batch_size=8,
            data_loader_workers=0,
            fold_parallelism=2,
            fold_parallel_backend="processes",
            fold_slots_per_device=1,
            reuse_tokenizer_and_chunk_plans=True,
            chunk_plan_cache_max_entries=100,
            tokenized_chunk_cache_max_entries=1000,
        ),
        neural_query_operational_controls=(
            RoleNeutralNeuralQueryOperationalControls(
                inner_fold_parallelism=2,
                fold_parallel_backend="processes",
                fold_slots_per_device=1,
                bank_parallelism=2,
                worker_cpu_threads=1,
                schema_version=(
                    ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA
                ),
            )
        ),
        first_owner_validation=(
            RoleNeutralFirstOwnerValidationPolicy(
                devices=devices,
                gpu_max_allocation_fraction=(
                    maximum_allocation_fraction
                ),
                gpu_minimum_headroom_bytes=minimum_headroom_bytes,
                gpu_sample_interval_seconds=0.01,
                required_tfidf_parallel_backend="processes",
                schema_version=(
                    ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_POLICY_SCHEMA
                ),
            )
        ),
    )


def _gpu_sample(*, used_bytes: int, total_bytes: int):
    return tuple(
        {
            "device": device,
            "uuid": f"uuid-{device}",
            "utilization_percent": 75.0,
            "memory_used_bytes": used_bytes,
            "memory_total_bytes": total_bytes,
        }
        for device in ("cuda:3", "cuda:8")
    )


def _install_component_report_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        execution_module,
        "_validate_first_owner_component_reports",
        lambda **_kwargs: {
            "every_parallel_component_report_self_authenticated": True,
            "configured_parallel_work_did_not_serialize": True,
        },
    )


def _gate_path(root: Path) -> Path:
    return root / "first_owner_validation_gate.json"


def test_first_owner_sampler_uses_acquisition_completion_and_host_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic_values = iter((10.0, 10.25, 20.0, 20.5))
    monkeypatch.setattr(
        execution_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: next(monotonic_values),
        ),
    )
    used_values = iter((400, 800))
    monkeypatch.setattr(
        execution_module.telemetry_module,
        "sample_nvidia_gpus",
        lambda _devices: _gpu_sample(
            used_bytes=next(used_values),
            total_bytes=1000,
        ),
    )
    sampler = execution_module._FirstOwnerGpuSampler(
        devices=("cuda:3", "cuda:8"),
        interval_seconds=0.01,
    )

    sampler._sample()
    sampler._sample()

    samples = sampler.samples
    first = samples[0]
    assert (
        first["sample_acquisition_started_monotonic_seconds"]
        == 10.0
    )
    assert (
        first["sample_acquisition_finished_monotonic_seconds"]
        == 10.25
    )
    assert first["sample_monotonic_seconds"] == 10.25
    policy = _policy(minimum_headroom_bytes=100)
    observation = execution_module._first_owner_memory_observation(
        samples=samples,
        sampling_errors=(),
        policy=policy.first_owner_validation,
    )
    for row in observation["devices"].values():
        assert row["host_peak_memory_used_bytes"] == 800
        assert row["memory_acceptance_peak_bytes"] == 800
        assert row["peak_allocation_fraction"] == 0.8
        assert row["minimum_headroom_bytes"] == 200
        assert (
            row["memory_acceptance_peak_source"]
            == "host_nvml_absolute_peak"
        )
    assert (
        observation[
            "memory_acceptance_checks_absolute_peak_fraction_and_headroom"
        ]
        is True
    )

    missing_bracket = [dict(row) for row in samples]
    missing_bracket[0].pop(
        "sample_acquisition_finished_monotonic_seconds"
    )
    with pytest.raises(RuntimeError, match="acquisition bracket"):
        execution_module._first_owner_memory_observation(
            samples=missing_bracket,
            sampling_errors=(),
            policy=policy.first_owner_validation,
        )


def _seal(body: dict) -> dict:
    return {**body, "content_sha256": _sha(body)}


def _task_phase(
    phase: str,
    *,
    resource_plan: dict,
) -> dict:
    intervals = [
        {
            "task": f"{phase}-a",
            "canonical_task_index": 0,
            "device": "cuda:3",
            "process_id": 101,
            "thread_id": 1,
            "started_monotonic_ns": 10,
            "finished_monotonic_ns": 30,
            "gpu_peak_allocated_bytes": 100,
            "torch_determinism_observed": {},
        },
        {
            "task": f"{phase}-b",
            "canonical_task_index": 1,
            "device": "cuda:8",
            "process_id": 102,
            "thread_id": 1,
            "started_monotonic_ns": 20,
            "finished_monotonic_ns": 40,
            "gpu_peak_allocated_bytes": 100,
            "torch_determinism_observed": {},
        },
    ]
    return _seal(
        {
            "schema_version": (
                "production_neural_query_task_phase_execution_attestation_v1"
            ),
            "phase": phase,
            "configured_parallelism": 2,
            "actual_task_count": 2,
            "maximum_concurrent_leases": 2,
            "task_intervals": intervals,
            "per_device": {
                device: {
                    "task_count": 1,
                    "maximum_concurrent_leases": 1,
                    "maximum_child_peak_allocated_bytes": 100,
                }
                for device in ("cuda:3", "cuda:8")
            },
            "configured_total_parallelism_respected": True,
            "configured_per_device_slots_respected": True,
            "waiting_tasks_hold_no_lease": True,
            "canonical_result_order_restored": True,
            "process_isolated": True,
            "worker_cpu_threads": 1,
            "resource_locators_in_scientific_payload": False,
        }
    )


def _complete_component_reports() -> dict:
    devices = ["cuda:3", "cuda:8"]
    htr_plan = {
        "devices": devices,
        "fold_parallelism": 2,
        "fold_parallel_backend": "processes",
        "fold_slots_per_device": 1,
    }
    htr_intervals = [
        {
            "stage": "nuisance",
            "objective": "propensity",
            "fold": 1,
            "device": "cuda:3",
            "process_id": 1,
            "thread_id": 1,
            "started_monotonic_ns": 10,
            "finished_monotonic_ns": 30,
            "gpu_peak_allocated_bytes": 1,
            "torch_determinism_observed": {},
        },
        {
            "stage": "nuisance",
            "objective": "outcome",
            "fold": 2,
            "device": "cuda:8",
            "process_id": 2,
            "thread_id": 1,
            "started_monotonic_ns": 20,
            "finished_monotonic_ns": 40,
            "gpu_peak_allocated_bytes": 1,
            "torch_determinism_observed": {},
        },
        {
            "stage": "effect",
            "objective": "effect",
            "fold": 1,
            "device": "cuda:3",
            "process_id": 1,
            "thread_id": 1,
            "started_monotonic_ns": 50,
            "finished_monotonic_ns": 70,
            "gpu_peak_allocated_bytes": 1,
            "torch_determinism_observed": {},
        },
        {
            "stage": "effect",
            "objective": "effect",
            "fold": 2,
            "device": "cuda:8",
            "process_id": 2,
            "thread_id": 1,
            "started_monotonic_ns": 60,
            "finished_monotonic_ns": 80,
            "gpu_peak_allocated_bytes": 1,
            "torch_determinism_observed": {},
        },
    ]
    htr = _seal(
        {
            "schema_version": (
                "production_role_neutral_htr_operational_attestation_v2"
            ),
            "fold_resource_plan": htr_plan,
            "fold_execution": {
                "resource_plan": htr_plan,
                "fold_intervals": htr_intervals,
                "nuisance_barrier_enforced": True,
                "effect_submitted_only_after_nuisance_oof_and_residuals": True,
                "every_selected_device_used_by_each_stage": True,
                "maximum_concurrent_fold_leases": 2,
                "process_isolated_rng": True,
            },
        }
    )
    matched_intervals = [
        {
            "fold": 1,
            "device": "cuda:3",
            "process_id": 3,
            "thread_id": 1,
            "started_monotonic_ns": 10,
            "finished_monotonic_ns": 30,
            "gpu_peak_allocated_bytes": 1,
            "torch_determinism_observed": {},
        },
        {
            "fold": 2,
            "device": "cuda:8",
            "process_id": 4,
            "thread_id": 1,
            "started_monotonic_ns": 20,
            "finished_monotonic_ns": 40,
            "gpu_peak_allocated_bytes": 1,
            "torch_determinism_observed": {},
        },
    ]
    matched = _seal(
        {
            "schema_version": (
                "production_role_neutral_matched_pair_operational_attestation_v1"
            ),
            "fold_resource_plan": htr_plan,
            "fold_execution": {
                "resource_plan": htr_plan,
                "fold_intervals": matched_intervals,
                "every_selected_device_used": True,
                "maximum_concurrent_fold_leases": 2,
                "process_isolated_rng_and_torch_determinism": True,
            },
        }
    )
    tfidf = _seal(
        {
            "schema_version": (
                "tfidf_joint_nuisance_fold_execution_attestation_v1"
            ),
            "configured_backend": "processes",
            "effective_workers": 2,
            "actual_peak_concurrent_fold_workers": 2,
            "subfold_parallelism": 1,
            "subfold_joblib_pools_created": False,
            "full_data_base_fits_after_fold_barrier": True,
            "final_stack_fits_after_fold_barrier": True,
            "fold_overlap_observed": True,
            "worker_pids": [5, 6],
        }
    )
    neural_plan = {
        "devices": devices,
        "inner_fold_parallelism": 2,
        "fold_parallel_backend": "processes",
        "fold_slots_per_device": 1,
        "bank_parallelism": 2,
        "worker_cpu_threads": 1,
        "owner_cpu_budget": 8,
    }
    inner = _task_phase("inner_folds", resource_plan=neural_plan)
    final = _task_phase(
        "consensus_and_final_refit_banks",
        resource_plan=neural_plan,
    )
    discovery = _seal(
        {
            "schema_version": (
                "production_neural_query_discovery_execution_attestation_v1"
            ),
            "resource_plan": neural_plan,
            "inner_fold_phase": inner,
            "inner_fold_barrier_monotonic_ns": 45,
            "inner_fold_barrier_enforced": True,
            "all_inner_results_verified_before_final_task_construction": True,
            "final_bank_phase": final,
        }
    )
    safe = _task_phase(
        "safe_evidence_banks",
        resource_plan=neural_plan,
    )
    heldout = _task_phase(
        "heldout_moment_banks",
        resource_plan=neural_plan,
    )
    phase_names = [
        "inner_folds_then_consensus_final_refits",
        "safe_evidence_banks",
        "heldout_moment_banks",
    ]
    neural = _seal(
        {
            "schema_version": (
                "production_role_neutral_neural_query_operational_attestation_v1"
            ),
            "resource_plan": neural_plan,
            "phase_order": phase_names,
            "phase_count": 3,
            "phases": [
                {
                    "phase_index": index,
                    "phase": name,
                    "attestation": attestation,
                }
                for index, (name, attestation) in enumerate(
                    zip(
                        phase_names,
                        (discovery, safe, heldout),
                        strict=True,
                    )
                )
            ],
            "all_phase_attestations_self_authenticated": True,
            "canonical_execution_order_preserved": True,
        }
    )
    return {
        "htr": htr,
        "matched_pair": matched,
        "tfidf": tfidf,
        "neural_query": neural,
    }


def test_component_report_gate_closes_every_parallel_phase() -> None:
    policy = _policy(minimum_headroom_bytes=100)
    reports = _complete_component_reports()
    result = SimpleNamespace(
        execution_telemetry={
            "schema_version": (
                "production_role_neutral_component_operational_reports_v2"
            ),
            "component_reports": reports,
        }
    )

    summary = execution_module._validate_first_owner_component_reports(
        result=result,
        policy=policy,
        gate=policy.first_owner_validation,
    )

    assert summary[
        "configured_parallel_work_did_not_serialize"
    ] is True
    assert summary["htr"]["nuisance_maximum_concurrent_leases"] == 2
    assert summary["matched_pair"]["maximum_concurrent_leases"] == 2
    assert summary["tfidf"]["maximum_concurrent_leases"] == 2
    assert set(summary["neural_query"]["phases"]) == {
        "inner_folds",
        "consensus_and_final_refit_banks",
        "safe_evidence_banks",
        "heldout_moment_banks",
    }

    changed_reports = _complete_component_reports()
    changed_tfidf = dict(changed_reports["tfidf"])
    changed_tfidf.pop("content_sha256")
    changed_tfidf["actual_peak_concurrent_fold_workers"] = 1
    changed_tfidf["fold_overlap_observed"] = False
    changed_reports["tfidf"] = _seal(changed_tfidf)
    result.execution_telemetry["component_reports"] = changed_reports
    with pytest.raises(ValueError, match="TF-IDF folds serialized"):
        execution_module._validate_first_owner_component_reports(
            result=result,
            policy=policy,
            gate=policy.first_owner_validation,
        )


def test_first_owner_gate_pass_preserves_owner_and_then_dispatches_rest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_component_report_stub(monkeypatch)
    monkeypatch.setattr(
        execution_module.telemetry_module,
        "sample_nvidia_gpus",
        lambda _devices: _gpu_sample(
            used_bytes=800,
            total_bytes=1000,
        ),
    )
    plan = _plan(gpu_ids=(3, 8))
    recorder = _GateProducerRecorder()
    executor = _CallRecordingExecutor()
    root = (tmp_path / "pass" / "execution").resolve()
    root.parent.mkdir()

    manifest = execute_and_publish_role_neutral_stage1(
        root=root,
        plan=plan,
        producer_factories=recorder.factories(),
        policy=_policy(minimum_headroom_bytes=100),
        executor=executor,
    )

    owner_order = tuple(plan.physical_execution_order)
    assert executor.calls == [
        (owner_order[0],),
        owner_order[1:],
    ]
    first_events = [
        row for row in recorder.events if row[0] == owner_order[0]
    ]
    assert len(first_events) == 18
    assert manifest["physical_fit_count"] == len(plan.physical_scopes)
    diagnostic = json.loads(
        _gate_path(root).read_text(encoding="utf-8")
    )
    execution_attestation = json.loads(
        (root / "execution_attestation.json").read_text(
            encoding="utf-8"
        )
    )
    registration = execution_attestation[
        "first_owner_validation"
    ]
    assert registration["relative_path"] == (
        "first_owner_validation_gate.json"
    )
    assert registration["content_sha256"] == diagnostic[
        "content_sha256"
    ]
    assert diagnostic["status"] == "passed"
    assert diagnostic["physical_owner_scope_id"] == owner_order[0]
    assert diagnostic["owner_two_submitted_before_gate"] is False
    assert (
        diagnostic["selected_owner_adopted_as_production_result"]
        is True
    )
    receipt_reauthentication = diagnostic[
        "fresh_component_receipt_reauthentication"
    ]
    assert (
        receipt_reauthentication[
            "every_component_root_reopened_and_tree_rehashed"
        ]
        is True
    )
    assert len(receipt_reauthentication["components"]) == 6
    assert diagnostic["complete_text_and_chunk_coverage_basis"] == (
        "fresh_parent_component_tree_terminal_and_receipt_validation_v1"
    )
    assert diagnostic["replica_b_executed"] is False
    assert (
        diagnostic["gpu_memory_observation"][
            "maximum_allocation_fraction_respected"
        ]
        is True
    )
    body = {
        key: value
        for key, value in diagnostic.items()
        if key != "content_sha256"
    }
    assert diagnostic["content_sha256"] == _sha(body)


@pytest.mark.parametrize(
    ("used_bytes", "minimum_headroom_bytes", "failure_kind"),
    (
        (850, 100, "allocation_fraction"),
        (750, 300, "headroom"),
    ),
)
def test_first_owner_memory_failure_never_invokes_owner_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    used_bytes: int,
    minimum_headroom_bytes: int,
    failure_kind: str,
) -> None:
    _install_component_report_stub(monkeypatch)
    monkeypatch.setattr(
        execution_module.telemetry_module,
        "sample_nvidia_gpus",
        lambda _devices: _gpu_sample(
            used_bytes=used_bytes,
            total_bytes=1000,
        ),
    )
    plan = _plan(gpu_ids=(3, 8))
    recorder = _GateProducerRecorder()
    executor = _CallRecordingExecutor()
    root = (
        tmp_path / failure_kind / "execution"
    ).resolve()
    root.parent.mkdir()

    with pytest.raises(
        RuntimeError,
        match="failed its hard validation gate before owner two",
    ):
        execute_and_publish_role_neutral_stage1(
            root=root,
            plan=plan,
            producer_factories=recorder.factories(),
            policy=_policy(
                minimum_headroom_bytes=minimum_headroom_bytes,
            ),
            executor=executor,
        )

    first_owner = plan.physical_execution_order[0]
    second_owner = plan.physical_execution_order[1]
    assert executor.calls == [(first_owner,)]
    assert not any(row[0] == second_owner for row in recorder.events)
    diagnostic = json.loads(
        _gate_path(root).read_text(encoding="utf-8")
    )
    assert diagnostic["status"] == "failed"
    assert diagnostic["owner_two_submitted_before_gate"] is False
    assert (
        diagnostic["selected_owner_adopted_as_production_result"]
        is False
    )
    assert [row["stage"] for row in diagnostic["failures"]] == [
        "gpu_memory"
    ]
    observation = diagnostic["gpu_memory_observation"]
    assert observation["accepted"] is False
    if failure_kind == "allocation_fraction":
        assert (
            observation[
                "maximum_allocation_fraction_respected"
            ]
            is False
        )
    else:
        assert observation["minimum_headroom_respected"] is False


def test_unconfigured_executor_still_receives_complete_owner_sequence(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=(3, 8))
    executor = _CallRecordingExecutor()
    root = (tmp_path / "unconfigured" / "execution").resolve()
    root.parent.mkdir()

    execute_and_publish_role_neutral_stage1(
        root=root,
        plan=plan,
        producer_factories=_ProducerRecorder().factories(),
        policy=RoleNeutralStage1ExecutionPolicy(
            resource_plan=_resource_plan(
                devices=("cuda:3", "cuda:8"),
                cpu_budget=8,
            ),
            max_parallel_owners=1,
        ),
        executor=executor,
    )

    assert executor.calls == [tuple(plan.physical_execution_order)]
    assert not _gate_path(root).exists()
