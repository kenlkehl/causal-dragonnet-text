from __future__ import annotations

from dataclasses import replace

import pytest

from oci.inference.role_neutral_lane_overlap_analysis import (
    CompletedFitIntervalObservation,
    FitLaneInterval,
    analyze_completed_fit_lane_overlap,
    analyze_completed_fit_observations_lane_overlap,
)


def _interval(
    interval_id: str,
    owner: str,
    lane: str,
    start: int,
    finish: int,
) -> FitLaneInterval:
    return FitLaneInterval(
        interval_id=interval_id,
        owner_execution_id=owner,
        lane_kind=lane,
        subphase_name=f"{lane}.work",
        resource_id=f"{lane}-resource",
        clock_domain_id="observation-clock",
        started_monotonic_ns=start,
        finished_monotonic_ns=finish,
        status="completed",
        timestamps_measured_directly=True,
    )


def _observation() -> CompletedFitIntervalObservation:
    return CompletedFitIntervalObservation.seal(
        observation_id="candidate/scope/repetition",
        owner_execution_ids=("owner-a", "owner-b"),
        clock_domain_id="observation-clock",
        observation_started_monotonic_ns=0,
        observation_finished_monotonic_ns=100,
        intervals=(
            _interval("cpu-a", "owner-a", "cpu", 0, 40),
            _interval("gpu-a-1", "owner-a", "gpu", 20, 50),
            _interval("cpu-b", "owner-b", "cpu", 50, 90),
            _interval("gpu-a-2", "owner-a", "gpu", 60, 80),
            _interval("gpu-b", "owner-b", "gpu", 70, 100),
        ),
    )


def test_overlap_analysis_separates_within_cross_and_simultaneous_time() -> None:
    observation = _observation()
    result = analyze_completed_fit_lane_overlap(
        observation.as_dict(),
        expected_observation_id=observation.observation_id,
        expected_owner_execution_ids=(
            "owner-a",
            "owner-b",
        ),
    )

    assert result.observation_window_duration_ns == 100
    assert result.any_lane_active_duration_ns == 100
    assert result.cpu_active_duration_ns == 80
    assert result.gpu_active_duration_ns == 70
    assert result.cpu_gpu_overlap_duration_ns == 50
    assert result.within_owner_overlap_duration_ns == 40
    assert result.cross_owner_overlap_duration_ns == 20
    assert result.within_owner_only_overlap_duration_ns == 30
    assert result.cross_owner_only_overlap_duration_ns == 10
    assert result.simultaneous_within_and_cross_overlap_duration_ns == 10
    assert result.per_owner_within_overlap_duration_ns == {
        "owner-a": 20,
        "owner-b": 20,
    }
    assert result.cpu_gpu_overlap_fraction_of_observation_window == 0.5
    assert result.cpu_gpu_overlap_fraction_of_any_lane_active == 0.5
    assert result.cpu_gpu_overlap_fraction_of_cpu_active == 0.625
    assert result.cpu_gpu_overlap_fraction_of_gpu_active == pytest.approx(
        5 / 7
    )
    assert result.within_owner_fraction_of_cpu_gpu_overlap == 0.8
    assert result.cross_owner_fraction_of_cpu_gpu_overlap == 0.4
    assert result.overlap_categories_are_nonadditive is True
    assert result.descriptive_overlap_only is True
    assert result.causal_speedup_claimed is False
    assert result.throughput_speedup_estimated is False
    assert result.as_dict()["content_sha256"] == result.content_sha256


def test_same_lane_overlap_is_unioned_instead_of_double_counted() -> None:
    observation = CompletedFitIntervalObservation.seal(
        observation_id="same-lane-union",
        owner_execution_ids=("owner",),
        clock_domain_id="clock",
        observation_started_monotonic_ns=0,
        observation_finished_monotonic_ns=30,
        intervals=(
            replace(
                _interval("cpu-1", "owner", "cpu", 0, 20),
                clock_domain_id="clock",
            ),
            replace(
                _interval("cpu-2", "owner", "cpu", 10, 30),
                clock_domain_id="clock",
            ),
            replace(
                _interval("gpu", "owner", "gpu", 15, 25),
                clock_domain_id="clock",
            ),
        ),
    )
    result = analyze_completed_fit_lane_overlap(
        observation,
        expected_observation_id="same-lane-union",
        expected_owner_execution_ids=("owner",),
    )
    assert result.cpu_active_duration_ns == 30
    assert result.gpu_active_duration_ns == 10
    assert result.cpu_gpu_overlap_duration_ns == 10
    assert result.within_owner_overlap_duration_ns == 10


def test_nonoverlapping_completed_lanes_report_zero_without_speedup_claim() -> None:
    observation = CompletedFitIntervalObservation.seal(
        observation_id="no-overlap",
        owner_execution_ids=("owner",),
        clock_domain_id="clock",
        observation_started_monotonic_ns=0,
        observation_finished_monotonic_ns=30,
        intervals=(
            replace(
                _interval("cpu", "owner", "cpu", 0, 10),
                clock_domain_id="clock",
            ),
            replace(
                _interval("gpu", "owner", "gpu", 20, 30),
                clock_domain_id="clock",
            ),
        ),
    )
    result = analyze_completed_fit_lane_overlap(
        observation,
        expected_observation_id="no-overlap",
        expected_owner_execution_ids=("owner",),
    )
    assert result.cpu_gpu_overlap_duration_ns == 0
    assert result.cpu_gpu_overlap_fraction_of_any_lane_active == 0.0
    assert result.within_owner_fraction_of_cpu_gpu_overlap == 0.0
    assert result.cross_owner_fraction_of_cpu_gpu_overlap == 0.0
    assert result.causal_speedup_claimed is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("status", "failed", "must be completed"),
        (
            "timestamps_measured_directly",
            False,
            "directly measured",
        ),
        ("finished_monotonic_ns", 0, "positive duration"),
    ),
)
def test_interval_records_fail_closed_on_incomplete_or_invalid_timing(
    field: str,
    value: object,
    message: str,
) -> None:
    row = _interval("interval", "owner", "cpu", 0, 1).as_dict()
    row[field] = value
    with pytest.raises(ValueError, match=message):
        FitLaneInterval.from_mapping(row)


def test_closed_schemas_and_content_identity_reject_extra_or_tampered_data() -> None:
    observation = _observation()
    extra = observation.as_dict()
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="fields must be closed"):
        CompletedFitIntervalObservation.from_mapping(extra)

    extra_interval = observation.as_dict()
    extra_interval["intervals"][0]["unexpected"] = True
    with pytest.raises(ValueError, match="fields must be closed"):
        CompletedFitIntervalObservation.from_mapping(extra_interval)

    tampered = observation.as_dict()
    tampered["intervals"][0]["finished_monotonic_ns"] = 39
    with pytest.raises(ValueError, match="content identity changed"):
        CompletedFitIntervalObservation.from_mapping(tampered)


def test_observation_rejects_missing_lane_unknown_owner_and_clock_substitution() -> None:
    cpu = _interval("cpu", "owner-a", "cpu", 0, 10)
    with pytest.raises(ValueError, match="CPU and GPU"):
        CompletedFitIntervalObservation.seal(
            observation_id="missing-gpu",
            owner_execution_ids=("owner-a",),
            clock_domain_id="observation-clock",
            observation_started_monotonic_ns=0,
            observation_finished_monotonic_ns=10,
            intervals=(cpu,),
        )

    gpu_unknown = _interval("gpu", "owner-b", "gpu", 0, 10)
    with pytest.raises(ValueError, match="cover declared owner"):
        CompletedFitIntervalObservation.seal(
            observation_id="unknown-owner",
            owner_execution_ids=("owner-a",),
            clock_domain_id="observation-clock",
            observation_started_monotonic_ns=0,
            observation_finished_monotonic_ns=10,
            intervals=(cpu, gpu_unknown),
        )

    gpu_other_clock = replace(
        _interval("gpu", "owner-a", "gpu", 0, 10),
        clock_domain_id="another-clock",
    )
    with pytest.raises(ValueError, match="clock domain"):
        CompletedFitIntervalObservation.seal(
            observation_id="clock-substitution",
            owner_execution_ids=("owner-a",),
            clock_domain_id="observation-clock",
            observation_started_monotonic_ns=0,
            observation_finished_monotonic_ns=10,
            intervals=(cpu, gpu_other_clock),
        )


def test_analysis_binds_exact_observation_and_ordered_owner_ids() -> None:
    observation = _observation()
    with pytest.raises(ValueError, match="differs from the expected"):
        analyze_completed_fit_lane_overlap(
            observation,
            expected_observation_id="another-observation",
            expected_owner_execution_ids=(
                "owner-a",
                "owner-b",
            ),
        )
    with pytest.raises(ValueError, match="differs from the expected"):
        analyze_completed_fit_lane_overlap(
            observation,
            expected_observation_id=observation.observation_id,
            expected_owner_execution_ids=(
                "owner-b",
                "owner-a",
            ),
        )


def test_multiple_observations_are_analyzed_independently_and_exactly_once() -> None:
    first = _observation()
    second = CompletedFitIntervalObservation.seal(
        observation_id="second",
        owner_execution_ids=("owner-c",),
        clock_domain_id="second-clock",
        observation_started_monotonic_ns=1_000,
        observation_finished_monotonic_ns=1_020,
        intervals=(
            replace(
                _interval("cpu", "owner-c", "cpu", 1_000, 1_010),
                clock_domain_id="second-clock",
            ),
            replace(
                _interval("gpu", "owner-c", "gpu", 1_005, 1_020),
                clock_domain_id="second-clock",
            ),
        ),
    )
    results = analyze_completed_fit_observations_lane_overlap(
        (first.as_dict(), second),
        expected_observation_owner_execution_bindings={
            first.observation_id: ("owner-a", "owner-b"),
            second.observation_id: ("owner-c",),
        },
    )
    assert [value.observation_id for value in results] == [
        first.observation_id,
        second.observation_id,
    ]
    assert results[1].cpu_gpu_overlap_duration_ns == 5

    with pytest.raises(ValueError, match="duplicated"):
        analyze_completed_fit_observations_lane_overlap(
            (first, first),
            expected_observation_owner_execution_bindings={
                first.observation_id: ("owner-a", "owner-b"),
            },
        )
