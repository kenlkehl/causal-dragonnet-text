from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

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
        self.calls.append(tuple(task.physical_owner.scope_id for task in tasks))
        return super().execute(
            tasks=tasks,
            worker=worker,
            max_workers=max_workers,
            cpu_budget=cpu_budget,
        )


class _GateProducerRecorder(_ProducerRecorder):
    def factory(self, expected_component: str):
        base_factory = super().factory(expected_component)

        def bind(invocation):
            bound = base_factory(invocation)
            if expected_component not in {"matched_pair", "neural_query"}:
                return bound

            def execute():
                bound.execute()
                body = {
                    "schema_version": "test_first_owner_operational_attestation_v1",
                    "component": expected_component,
                    "physical_owner_scope_id": invocation.physical_owner.scope_id,
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
        resource_plan=_resource_plan(devices=devices, cpu_budget=8),
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
        neural_query_operational_controls=RoleNeutralNeuralQueryOperationalControls(
            inner_fold_parallelism=2,
            fold_parallel_backend="processes",
            fold_slots_per_device=1,
            bank_parallelism=2,
            worker_cpu_threads=1,
            schema_version=ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA,
        ),
        first_owner_validation=RoleNeutralFirstOwnerValidationPolicy(
            devices=devices,
            gpu_max_allocation_fraction=maximum_allocation_fraction,
            gpu_minimum_headroom_bytes=minimum_headroom_bytes,
            gpu_sample_interval_seconds=0.01,
            required_tfidf_parallel_backend="processes",
            schema_version=ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_POLICY_SCHEMA,
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


def test_resume_requires_reports_only_from_fresh_parallel_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resumed = ("htr", "matched_pair", "tfidf")
    telemetry = {
        "schema_version": (
            "production_role_neutral_component_operational_reports_v2"
        ),
        "component_reports": {
            "neural_query": {"component": "neural_query"},
        },
        "resumed_components": list(resumed),
        "component_execution_intervals": [
            {
                "component": component,
                "status": "resumed",
                "interval_semantics": (
                    execution_module
                    ._ROLE_NEUTRAL_COMPONENT_RESUME_INTERVAL_SEMANTICS
                ),
            }
            for component in resumed
        ],
    }
    for name in ("htr", "matched", "tfidf"):
        monkeypatch.setattr(
            execution_module,
            f"_validate_first_owner_{name}_report",
            lambda *_args, **_kwargs: pytest.fail(
                "resumed component overlap report was replayed"
            ),
        )
    monkeypatch.setattr(
        execution_module,
        "_validate_first_owner_neural_report",
        lambda value, *, gate: {
            "validated_fresh_component": value["component"],
        },
    )
    policy = _policy(minimum_headroom_bytes=100)

    summary = execution_module._validate_first_owner_component_reports(
        execution_telemetry=telemetry,
        policy=policy,
        gate=policy.first_owner_validation,
    )

    assert summary["resumed_parallel_components"] == list(resumed)
    assert summary["neural_query"] == {
        "validated_fresh_component": "neural_query",
    }
    for component in resumed:
        assert summary[component] == {
            "operational_overlap_status": "not_replayed_on_resume",
        }

    telemetry["component_reports"] = {}
    with pytest.raises(
        ValueError,
        match="did not attest every fresh parallel producer",
    ):
        execution_module._validate_first_owner_component_reports(
            execution_telemetry=telemetry,
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
        lambda _devices: _gpu_sample(used_bytes=800, total_bytes=1000),
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
    assert executor.calls == [(owner_order[0],), owner_order[1:]]
    assert len([row for row in recorder.events if row[0] == owner_order[0]]) == 18
    assert manifest["physical_fit_count"] == len(plan.physical_scopes)
    diagnostic = json.loads(_gate_path(root).read_text(encoding="utf-8"))
    execution_attestation = json.loads(
        (root / "execution_attestation.json").read_text(encoding="utf-8")
    )
    registration = execution_attestation["first_owner_validation"]
    assert registration["relative_path"] == "first_owner_validation_gate.json"
    assert registration["content_sha256"] == diagnostic["content_sha256"]
    assert diagnostic["status"] == "passed"
    assert diagnostic["physical_owner_scope_id"] == owner_order[0]
    assert diagnostic["owner_two_submitted_before_gate"] is False
    assert diagnostic["selected_owner_adopted_as_production_result"] is True
    assert diagnostic["replica_b_executed"] is False


def test_resume_replays_missing_gate_before_owner_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_component_report_stub(monkeypatch)
    used_bytes = 950
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
    root = (tmp_path / "resume" / "execution").resolve()
    root.parent.mkdir()

    with pytest.raises(
        RuntimeError,
        match="failed its hard validation gate before owner two",
    ):
        execute_and_publish_role_neutral_stage1(
            root=root,
            plan=plan,
            producer_factories=recorder.factories(),
            policy=_policy(minimum_headroom_bytes=100),
            executor=_CallRecordingExecutor(),
        )

    used_bytes = 700
    resumed_executor = _CallRecordingExecutor()
    manifest = execute_and_publish_role_neutral_stage1(
        root=root,
        plan=plan,
        producer_factories=recorder.factories(),
        policy=_policy(minimum_headroom_bytes=100),
        executor=resumed_executor,
        resume=True,
    )

    owner_order = tuple(plan.physical_execution_order)
    assert resumed_executor.calls == [
        (owner_order[0],),
        owner_order[1:],
    ]
    assert manifest["physical_fit_count"] == len(plan.physical_scopes)
    diagnostic = json.loads(_gate_path(root).read_text(encoding="utf-8"))
    assert diagnostic["status"] == "passed"
    assert diagnostic["owner_two_submitted_before_gate"] is False
    assert diagnostic["selected_owner_adopted_as_production_result"] is True


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
    root = (tmp_path / failure_kind / "execution").resolve()
    root.parent.mkdir()

    with pytest.raises(
        RuntimeError,
        match="failed its hard validation gate before owner two",
    ):
        execute_and_publish_role_neutral_stage1(
            root=root,
            plan=plan,
            producer_factories=recorder.factories(),
            policy=_policy(minimum_headroom_bytes=minimum_headroom_bytes),
            executor=executor,
        )

    first_owner = plan.physical_execution_order[0]
    second_owner = plan.physical_execution_order[1]
    assert executor.calls == [(first_owner,)]
    assert not any(row[0] == second_owner for row in recorder.events)
    diagnostic = json.loads(_gate_path(root).read_text(encoding="utf-8"))
    assert diagnostic["status"] == "failed"
    assert diagnostic["owner_two_submitted_before_gate"] is False
    assert diagnostic["selected_owner_adopted_as_production_result"] is False
    assert [row["stage"] for row in diagnostic["failures"]] == ["gpu_memory"]
