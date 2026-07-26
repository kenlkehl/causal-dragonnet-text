"""Explicit operational-capacity fixtures for portable workflow tests."""

from oci.inference.portable_workflow_spec import (
    ResourcePerformanceSafetyPolicy,
)


def resource_safety_policy(
    *,
    gpu_max_allocation_fraction: float,
    gpu_minimum_headroom_bytes: int,
    minimum_multi_device_throughput_ratio: float,
    maximum_coordination_proof_overhead_ratio: float,
    maximum_ordinary_read_amplification: float,
    minimum_benchmark_repetitions_per_scope: int,
    read_counter_source: str,
    fail_on_external_gpu_occupants: bool,
) -> ResourcePerformanceSafetyPolicy:
    """Build a policy whose abort-only capacities are explicit test data."""

    return ResourcePerformanceSafetyPolicy(
        gpu_max_allocation_fraction=gpu_max_allocation_fraction,
        gpu_minimum_headroom_bytes=gpu_minimum_headroom_bytes,
        minimum_multi_device_throughput_ratio=(
            minimum_multi_device_throughput_ratio
        ),
        maximum_coordination_proof_overhead_ratio=(
            maximum_coordination_proof_overhead_ratio
        ),
        maximum_ordinary_read_amplification=(
            maximum_ordinary_read_amplification
        ),
        minimum_benchmark_repetitions_per_scope=(
            minimum_benchmark_repetitions_per_scope
        ),
        read_counter_source=read_counter_source,
        fail_on_external_gpu_occupants=fail_on_external_gpu_occupants,
        hierarchical_job_cache_max_entry_bytes=32_000_000,
        first_untouched_gate_max_initial_spent_rows=1_000_000,
        first_untouched_gate_max_first_gate_rows=1_000_000,
        first_untouched_gate_max_total_text_utf8_bytes=1_073_741_824,
        first_untouched_gate_max_catalog_atoms=100_000,
        first_untouched_gate_max_source_manifest_bytes=16_777_216,
        first_untouched_gate_max_direct_numerical_signals=16_384,
        first_untouched_gate_max_single_matrix_file_bytes=1_073_741_824,
        first_untouched_gate_max_total_matrix_file_bytes=4_294_967_296,
    )
