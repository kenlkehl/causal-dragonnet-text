"""Shared closed identities for Stage 1 scheduler tests."""

from pathlib import Path

from oci.inference.portable_workflow_spec import Stage1ExecutionProfile
from oci.inference.production_stage1_scope_scheduler import (
    Stage1PhysicalFitIdentity,
)
from oci.inference.stage1_execution_topology_policy import (
    ONE_CONTEXT_PER_SELECTED_DEVICE,
    Stage1ExecutionTopologyPolicy,
)
from oci.inference.stage1_htr_operational_controls import (
    RoleNeutralHTROperationalControls,
)


PHYSICAL_FIT_IDENTITY = Stage1PhysicalFitIdentity(
    architecture_identity="1" * 64,
    target="test_all_ten_stage1_context_fit_v1",
    scientific_configuration_identity="2" * 64,
    producer_identity="3" * 64,
    runtime_compatibility_class="test-python-posix-v1",
)


def stage1_execution_profile(
    *,
    resource_kind: str,
    device_count: int,
    scope_workers_per_device: int,
    executor_mode: str = "persistent_slots",
    persistent_slot_startup_timeout_seconds: float = 30.0,
    topology_mode: str = ONE_CONTEXT_PER_SELECTED_DEVICE,
    training_batch_size: int = 4,
    sentence_encoder_batch_size: int = 8,
    data_loader_workers: int = 0,
    reuse_tokenizer_and_chunk_plans: bool = False,
    chunk_plan_cache_max_entries: int = 0,
    tokenized_chunk_cache_max_entries: int = 0,
    selection_method: str = "operator_configured",
    benchmark_evidence_kind: str = "none",
    selected_candidate: str | None = None,
    benchmark_result_sha256: str | None = None,
    benchmark_result_locator: Path | None = None,
    benchmark_workload_deployment_sha256: str | None = None,
    benchmark_workload_deployment_locator: Path | None = None,
    benchmark_publication_sha256: str | None = None,
    benchmark_publication_locator: Path | None = None,
) -> Stage1ExecutionProfile:
    topology = Stage1ExecutionTopologyPolicy(mode=topology_mode)
    max_parallel_owners = topology.effective_parallel_owners_for_shape(
        resource_kind=resource_kind,
        device_count=device_count,
        workers_per_device=scope_workers_per_device,
    )
    return Stage1ExecutionProfile(
        resource_kind=resource_kind,
        device_count=device_count,
        scope_workers_per_device=scope_workers_per_device,
        max_parallel_owners=max_parallel_owners,
        executor_mode=executor_mode,
        persistent_slot_startup_timeout_seconds=(
            persistent_slot_startup_timeout_seconds
        ),
        neural_query_topology=topology,
        htr_operational_controls=RoleNeutralHTROperationalControls(
            training_batch_size=training_batch_size,
            sentence_encoder_batch_size=sentence_encoder_batch_size,
            data_loader_workers=data_loader_workers,
            reuse_tokenizer_and_chunk_plans=(
                reuse_tokenizer_and_chunk_plans
            ),
            chunk_plan_cache_max_entries=(
                chunk_plan_cache_max_entries
            ),
            tokenized_chunk_cache_max_entries=(
                tokenized_chunk_cache_max_entries
            ),
        ),
        selection_method=selection_method,
        benchmark_evidence_kind=benchmark_evidence_kind,
        selected_candidate=selected_candidate,
        benchmark_result_sha256=benchmark_result_sha256,
        benchmark_result_locator=benchmark_result_locator,
        benchmark_workload_deployment_sha256=(
            benchmark_workload_deployment_sha256
        ),
        benchmark_workload_deployment_locator=(
            benchmark_workload_deployment_locator
        ),
        benchmark_publication_sha256=benchmark_publication_sha256,
        benchmark_publication_locator=benchmark_publication_locator,
    )


__all__ = [
    "PHYSICAL_FIT_IDENTITY",
    "stage1_execution_profile",
]
