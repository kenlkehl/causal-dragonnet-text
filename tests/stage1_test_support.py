"""Shared closed identities for Stage 1 scheduler tests."""

from pathlib import Path

from oci.inference.portable_workflow_spec import Stage1ExecutionProfile
from oci.inference.production_stage1_scope_scheduler import (
    Stage1PhysicalFitIdentity,
)
from oci.inference.stage1_execution_topology_policy import (
    ONE_CONTEXT_PER_SELECTED_DEVICE,
    ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES,
    Stage1ExecutionTopologyPolicy,
)
from oci.inference.stage1_htr_operational_controls import (
    RoleNeutralHTROperationalControls,
)
from oci.inference.neural_query_operational_controls import (
    ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA,
    RoleNeutralNeuralQueryOperationalControls,
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
    max_parallel_owners: int | None = None,
    executor_mode: str = "persistent_slots",
    persistent_slot_startup_timeout_seconds: float = 30.0,
    topology_mode: str = ONE_CONTEXT_PER_SELECTED_DEVICE,
    training_batch_size: int = 4,
    sentence_encoder_batch_size: int = 8,
    data_loader_workers: int = 0,
    reuse_tokenizer_and_chunk_plans: bool = True,
    chunk_plan_cache_max_entries: int = 100,
    tokenized_chunk_cache_max_entries: int = 1000,
    neural_query_inner_fold_parallelism: int = 1,
    neural_query_fold_parallel_backend: str = "threads",
    neural_query_fold_slots_per_device: int = 1,
    neural_query_bank_parallelism: int = 1,
    neural_query_worker_cpu_threads: int = 1,
    tfidf_parallel_backend: str = "processes",
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
    topology_capacity = topology.effective_parallel_owners_for_shape(
        resource_kind=resource_kind,
        device_count=device_count,
        workers_per_device=scope_workers_per_device,
    )
    if max_parallel_owners is None:
        max_parallel_owners = topology_capacity
    lease_device_count = (
        device_count
        if topology_mode
        == ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES
        else 1
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
            fold_parallelism=lease_device_count,
            fold_parallel_backend=(
                "threads"
                if lease_device_count == 1
                else "processes"
            ),
            fold_slots_per_device=1,
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
        neural_query_operational_controls=(
            RoleNeutralNeuralQueryOperationalControls(
                inner_fold_parallelism=(
                    neural_query_inner_fold_parallelism
                ),
                fold_parallel_backend=(
                    neural_query_fold_parallel_backend
                ),
                fold_slots_per_device=(
                    neural_query_fold_slots_per_device
                ),
                bank_parallelism=neural_query_bank_parallelism,
                worker_cpu_threads=neural_query_worker_cpu_threads,
                schema_version=(
                    ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA
                ),
            )
        ),
        tfidf_parallel_backend=tfidf_parallel_backend,
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
