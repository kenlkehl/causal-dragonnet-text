from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from oci.inference.performance_telemetry import (
    BenchmarkRunObservation,
    ImmutableInputObservation,
    PerformanceAcceptancePolicy,
    RepresentativeScope,
    TelemetryLedger,
    assess_benchmark_acceptance,
)
from oci.inference.portable_resource_scheduler import (
    BenchmarkCandidate,
    select_fastest_safe_candidate,
)
from oci.inference.portable_workflow_spec import (
    ResourcePerformanceSafetyPolicy,
)
from tests.resource_safety_test_support import resource_safety_policy

OUTER_SCOPE = "configured-full-fit"
INNER_SCOPE = "configured-inner-fit"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _policy(
    *,
    overhead: float = 0.22,
    reads: float = 1.6,
    peak: float = 0.77,
    headroom: int = 321,
    repetitions: int = 2,
    speedup: float = 1.3,
) -> PerformanceAcceptancePolicy:
    return PerformanceAcceptancePolicy(
        representative_scopes=(
            RepresentativeScope(label=OUTER_SCOPE, fit_row_count=17),
            RepresentativeScope(label=INNER_SCOPE, fit_row_count=11),
        ),
        resource_performance_safety=resource_safety_policy(
            gpu_max_allocation_fraction=peak,
            gpu_minimum_headroom_bytes=headroom,
            minimum_multi_device_throughput_ratio=speedup,
            maximum_coordination_proof_overhead_ratio=overhead,
            maximum_ordinary_read_amplification=reads,
            minimum_benchmark_repetitions_per_scope=repetitions,
            read_counter_source="logical_read_bytes",
            fail_on_external_gpu_occupants=True,
        ),
        scientific_reference_candidate="single",
        multi_device_baselines=(("dual", "single"),),
    )


def _record(
    name: str,
    *,
    activity_kind: str,
    wall: float,
    read: int,
    scope_label: str | None = None,
    status: str = "completed",
) -> dict[str, object]:
    return {
        "name": name,
        "activity_kind": activity_kind,
        "scope_label": scope_label,
        "wall_seconds": wall,
        "cpu_seconds": wall / 2,
        "process_read_bytes": read,
        "process_written_bytes": 0,
        "byte_counters": {
            "read": read,
            "written": 0,
            "copied": 0,
            "hashed": 0,
            "compressed": 0,
            "decompressed": 0,
            "json_encoded": 0,
            "fsynced": 0,
        },
        "gpu_samples": [],
        "gpu_peak_memory_bytes": {},
        "status": status,
    }


def _telemetry(
    *,
    coordination_multiplier: float = 1.0,
    ordinary_extra_read: int = 20,
    terminal_count: int = 1,
    devices: tuple[str, ...] = ("cuda:0",),
) -> dict[str, object]:
    records = [
        _record(
            "outer.model",
            activity_kind="model_fit",
            scope_label=OUTER_SCOPE,
            wall=10.0,
            read=15,
        ),
        _record(
            "outer.proof",
            activity_kind="coordination_proof",
            scope_label=OUTER_SCOPE,
            wall=coordination_multiplier,
            read=5,
        ),
        _record(
            "inner.model",
            activity_kind="model_fit",
            scope_label=INNER_SCOPE,
            wall=8.0,
            read=15,
        ),
        _record(
            "inner.proof",
            activity_kind="coordination_proof",
            scope_label=INNER_SCOPE,
            wall=0.8 * coordination_multiplier,
            read=5,
        ),
        _record(
            "ordinary.preparation",
            activity_kind="ordinary",
            wall=0.5,
            read=ordinary_extra_read,
        ),
    ]
    records.extend(
        _record(
            f"terminal.audit.{index}",
            activity_kind="terminal_audit",
            wall=0.4,
            read=900,
        )
        for index in range(terminal_count)
    )
    return {
        "schema_version": "portable_workflow_subphase_telemetry_v1",
        "devices": list(devices),
        "byte_counters": {},
        "subphases": records,
    }


def _run(
    candidate: str,
    scope: str,
    repetition: int,
    *,
    devices: tuple[str, ...],
    wall: float,
    artifact_path: str,
    peak: float = 0.55,
    headroom: int = 500,
) -> BenchmarkRunObservation:
    return BenchmarkRunObservation(
        candidate_name=candidate,
        scope_label=scope,
        repetition_index=repetition,
        device_ids=devices,
        concurrency_per_device=1,
        completed_scope_fits=1,
        model_fit_wall_seconds=wall,
        peak_allocation_fraction=peak,
        minimum_observed_headroom_bytes=headroom,
        oom_observed=False,
        scientific_artifact_sha256=_digest(f"artifact:{scope}"),
        artifact_path=artifact_path,
    )


def _runs(
    *,
    single_device: str = "cuda:0",
    dual_devices: tuple[str, str] = ("cuda:0", "cuda:1"),
    path_prefix: str = "/scratch/first",
    dual_wall_multiplier: float = 1.0,
) -> tuple[BenchmarkRunObservation, ...]:
    output = []
    for repetition in range(2):
        output.extend(
            (
                _run(
                    "single",
                    OUTER_SCOPE,
                    repetition,
                    devices=(single_device,),
                    wall=10.0,
                    artifact_path=f"{path_prefix}/single/outer/{repetition}",
                ),
                _run(
                    "single",
                    INNER_SCOPE,
                    repetition,
                    devices=(single_device,),
                    wall=8.0,
                    artifact_path=f"{path_prefix}/single/inner/{repetition}",
                ),
                _run(
                    "dual",
                    OUTER_SCOPE,
                    repetition,
                    devices=dual_devices,
                    wall=6.0 * dual_wall_multiplier,
                    artifact_path=f"{path_prefix}/dual/outer/{repetition}",
                ),
                _run(
                    "dual",
                    INNER_SCOPE,
                    repetition,
                    devices=dual_devices,
                    wall=4.8 * dual_wall_multiplier,
                    artifact_path=f"{path_prefix}/dual/inner/{repetition}",
                ),
            )
        )
    return tuple(output)


def _inputs() -> tuple[ImmutableInputObservation, ...]:
    return (
        ImmutableInputObservation(content_sha256=_digest("input-a"), size_bytes=20),
        ImmutableInputObservation(content_sha256=_digest("input-b"), size_bytes=30),
    )


def _assess(
    *,
    telemetry: dict[str, object] | None = None,
    policy: PerformanceAcceptancePolicy | None = None,
    runs: tuple[BenchmarkRunObservation, ...] | None = None,
) -> dict[str, object]:
    return dict(
        assess_benchmark_acceptance(
            telemetry or _telemetry(),
            policy=policy or _policy(),
            immutable_inputs=_inputs(),
            benchmark_runs=runs or _runs(),
        )
    )


def test_configured_acceptance_uses_opaque_scope_labels_and_separate_audit() -> None:
    result = _assess()
    assert result["accepted"] is True
    assert result["selected_candidate"] == "dual"
    telemetry = result["telemetry_acceptance"]
    assert telemetry["ordinary_read_bytes"] == 60
    assert telemetry["terminal_audit_read_bytes"] == 900
    assert telemetry["ordinary_read_amplification"] == pytest.approx(1.2)
    assert telemetry["coordination_overhead_ratio"] == pytest.approx(0.1)
    assert telemetry["exactly_one_completed_terminal_audit"] is True
    assert {row["scope_label"] for row in telemetry["representative_scope_telemetry"]} == {
        OUTER_SCOPE,
        INNER_SCOPE,
    }
    dual = next(row for row in result["candidate_results"] if row["candidate_name"] == "dual")
    assert dual["accepted"] is True
    assert all(row["accepted"] for row in dual["multi_device_speedup_results"])


def test_telemetry_targets_fail_closed_independently() -> None:
    overhead = _assess(telemetry=_telemetry(coordination_multiplier=3.0))
    assert overhead["accepted"] is False
    assert overhead["telemetry_acceptance"]["coordination_target_accepted"] is False

    read_amplified = _assess(telemetry=_telemetry(ordinary_extra_read=100))
    assert read_amplified["accepted"] is False
    assert read_amplified["telemetry_acceptance"]["read_target_accepted"] is False
    assert read_amplified["telemetry_acceptance"]["terminal_audit_read_bytes"] == 900

    no_audit = _assess(telemetry=_telemetry(terminal_count=0))
    duplicate_audit = _assess(telemetry=_telemetry(terminal_count=2))
    assert no_audit["accepted"] is False
    assert duplicate_audit["accepted"] is False
    assert no_audit["telemetry_acceptance"]["exactly_one_completed_terminal_audit"] is False
    assert duplicate_audit["telemetry_acceptance"]["terminal_audit_record_count"] == 2


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    (
        ("oom", "zero_oom_accepted"),
        ("nondeterministic", "deterministic_artifact_identity"),
        ("peak", "peak_allocation_and_headroom_accepted"),
        ("headroom", "peak_allocation_and_headroom_accepted"),
        ("single_repetition", "minimum_repetitions_accepted"),
    ),
)
def test_repeated_safety_and_determinism_evidence_is_required(
    mutation: str,
    expected_reason: str,
) -> None:
    runs = list(_runs())
    reference_index = next(
        index
        for index, value in enumerate(runs)
        if value.candidate_name == "single"
        and value.scope_label == OUTER_SCOPE
        and value.repetition_index == 1
    )
    if mutation == "oom":
        runs[reference_index] = replace(
            runs[reference_index],
            completed_scope_fits=0,
            oom_observed=True,
            scientific_artifact_sha256=None,
        )
    elif mutation == "nondeterministic":
        runs[reference_index] = replace(
            runs[reference_index],
            scientific_artifact_sha256=_digest("different-result"),
        )
    elif mutation == "peak":
        runs[reference_index] = replace(
            runs[reference_index],
            peak_allocation_fraction=_policy().maximum_peak_allocation_fraction,
        )
    elif mutation == "headroom":
        runs[reference_index] = replace(
            runs[reference_index],
            minimum_observed_headroom_bytes=(_policy().minimum_headroom_bytes - 1),
        )
    else:
        runs = [
            value
            for value in runs
            if not (
                value.candidate_name == "single"
                and value.scope_label == OUTER_SCOPE
                and value.repetition_index == 1
            )
        ]
    result = _assess(runs=tuple(runs))
    assert result["accepted"] is False
    reference = next(
        row for row in result["candidate_results"] if row["candidate_name"] == "single"
    )
    outer = next(row for row in reference["scope_results"] if row["scope_label"] == OUTER_SCOPE)
    assert outer[expected_reason] is False


def test_multi_device_claim_requires_configured_per_scope_speedup() -> None:
    result = _assess(runs=_runs(dual_wall_multiplier=1.55))
    assert result["accepted"] is True
    assert result["selected_candidate"] == "single"
    dual = next(row for row in result["candidate_results"] if row["candidate_name"] == "dual")
    assert dual["accepted"] is False
    assert dual["multi_device_throughput_claim_accepted"] is False
    assert any(not row["accepted"] for row in dual["multi_device_speedup_results"])


def test_scientific_identity_excludes_paths_devices_and_performance() -> None:
    first = _assess()
    relocated = _assess(
        telemetry=_telemetry(devices=("cuda:7", "cuda:9")),
        runs=_runs(
            single_device="cuda:7",
            dual_devices=("cuda:7", "cuda:9"),
            path_prefix="/different/host/scratch",
        ),
    )
    assert first["scientific_result_identity_sha256"] is not None
    assert (
        first["scientific_result_identity_sha256"] == relocated["scientific_result_identity_sha256"]
    )
    assert (
        first["candidate_results"][0]["device_assignments_observed"]
        != relocated["candidate_results"][0]["device_assignments_observed"]
    )
    assert (
        first["candidate_results"][0]["artifact_paths_observed"]
        != relocated["candidate_results"][0]["artifact_paths_observed"]
    )


def test_telemetry_ledger_records_explicit_roles_and_scope_labels() -> None:
    ledger = TelemetryLedger()
    with ledger.subphase(
        "configured.fit",
        activity_kind="model_fit",
        scope_label=OUTER_SCOPE,
    ):
        ledger.count_bytes(read=3)
    with ledger.subphase(
        "configured.audit",
        activity_kind="terminal_audit",
    ):
        ledger.count_bytes(read=7)
    records = ledger.as_dict()["subphases"]
    assert records[0]["activity_kind"] == "model_fit"
    assert records[0]["scope_label"] == OUTER_SCOPE
    assert records[1]["activity_kind"] == "terminal_audit"
    assert records[1]["scope_label"] is None


def test_legacy_scheduler_selector_requires_explicit_configured_gates() -> None:
    low_concurrency = BenchmarkCandidate(
        name="low",
        device_count=1,
        concurrency_per_device=1,
        throughput_scopes_per_second=3.0,
        single_device_baseline_throughput=3.0,
        peak_allocation_fraction=0.4,
        minimum_headroom_bytes=500,
        repeated_runs=3,
        oom_count=0,
        deterministic=True,
        scientifically_equal=True,
    )
    high_concurrency = replace(
        low_concurrency,
        name="high",
        concurrency_per_device=2,
    )
    selected = select_fastest_safe_candidate(
        (high_concurrency, low_concurrency),
        resource_performance_safety=resource_safety_policy(
            gpu_max_allocation_fraction=0.7,
            gpu_minimum_headroom_bytes=400,
            minimum_multi_device_throughput_ratio=1.2,
            maximum_coordination_proof_overhead_ratio=0.2,
            maximum_ordinary_read_amplification=1.8,
            minimum_benchmark_repetitions_per_scope=3,
            read_counter_source="logical_read_bytes",
            fail_on_external_gpu_occupants=True,
        ),
    )
    assert selected.name == "low"
