from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import pytest

import oci.inference.role_neutral_benchmark_workload_provider as provider
from oci.inference.compact_preflight_compression_benchmark import (
    CompactPreflightCompressionBenchmarkConfig,
)
from oci.inference.performance_telemetry import ImmutableInputObservation
from oci.inference.portable_workflow_spec import (
    ResourcePerformanceSafetyPolicy,
    identity_sha256,
)
from oci.inference.production_all_evidence_workflow import (
    WORKFLOW_PHASE_MANIFEST_SCHEMA,
    WORKFLOW_PROGRESS_SCHEMA,
    WORKFLOW_SCHEMA,
    _attempt_tree_artifacts,
    _sha,
)
from oci.inference.production_stage1_scope_scheduler import (
    Stage1ScopeAssignment,
    Stage1ScopePlan,
    Stage1ScopeSpec,
    _sha256_json,
    _stage1_scope_plan_body,
)
from oci.inference.role_neutral_performance_benchmark import (
    RoleNeutralBenchmarkCandidate,
    RoleNeutralBenchmarkConfig,
    RoleNeutralBenchmarkScope,
)


def _scope(
    *,
    canonical_index: int,
    scope_id: str,
    scope_kind: str,
    fit_rows: tuple[int, ...],
    heldout_rows: tuple[int, ...],
    scope_seed: int,
) -> Stage1ScopeSpec:
    return Stage1ScopeSpec(
        canonical_index=canonical_index,
        scope_id=scope_id,
        scope_kind=scope_kind,
        outer_fold=1,
        inner_fold=(1 if scope_kind == "exact_inner" else None),
        context_epoch=(1 if scope_kind == "cumulative_spent" else None),
        provider_inner_fold=(1 if scope_kind == "cumulative_spent" else None),
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        global_seed=19,
        scope_seed=scope_seed,
    )


def _arbitrary_plan() -> Stage1ScopePlan:
    full = _scope(
        canonical_index=0,
        scope_id="outer_001_full",
        scope_kind="full_outer",
        fit_rows=tuple(range(7)),
        heldout_rows=(7, 8),
        scope_seed=101,
    )
    exact = _scope(
        canonical_index=1,
        scope_id="outer_001_inner_001",
        scope_kind="exact_inner",
        fit_rows=tuple(range(5)),
        heldout_rows=(5, 6),
        scope_seed=202,
    )
    equivalent_review = _scope(
        canonical_index=2,
        scope_id="outer_001_hierarchy_epoch_001",
        scope_kind="cumulative_spent",
        fit_rows=exact.fit_row_ids,
        heldout_rows=(7, 8),
        scope_seed=exact.scope_seed,
    )
    scopes = (full, exact, equivalent_review)
    assignments = tuple(
        Stage1ScopeAssignment(
            scope_id=value.scope_id,
            gpu_id=None,
            execution_rank=index,
            fit_row_count=value.fit_row_count,
            assigned_gpu_load_after=sum(item.fit_row_count for item in scopes[: index + 1]),
        )
        for index, value in enumerate(scopes)
    )
    body = _stage1_scope_plan_body(
        registry_content_sha256=hashlib.sha256(b"registry").hexdigest(),
        global_seed=19,
        review_rounds=1,
        initial_training_partitions=1,
        gpu_ids=(),
        scope_workers_per_gpu=1,
        scopes=scopes,
        assignments=assignments,
    )
    return Stage1ScopePlan(
        registry_content_sha256=body["registry_content_sha256"],
        global_seed=19,
        review_rounds=1,
        initial_training_partitions=1,
        gpu_ids=(),
        scope_workers_per_gpu=1,
        scopes=scopes,
        assignments=assignments,
        content_sha256=_sha256_json(body),
    )


def _benchmark_config() -> RoleNeutralBenchmarkConfig:
    safety = ResourcePerformanceSafetyPolicy(
        gpu_max_allocation_fraction=0.9,
        gpu_minimum_headroom_bytes=1,
        minimum_multi_device_throughput_ratio=1.0,
        maximum_coordination_proof_overhead_ratio=1.0,
        maximum_ordinary_read_amplification=2.0,
        minimum_benchmark_repetitions_per_scope=2,
        read_counter_source="process_read_bytes",
        fail_on_external_gpu_occupants=True,
    )
    return RoleNeutralBenchmarkConfig(
        representative_scopes=(
            RoleNeutralBenchmarkScope(
                label="arbitrary-outer",
                fit_row_count=7,
                fits_per_observation=2,
            ),
            RoleNeutralBenchmarkScope(
                label="arbitrary-inner",
                fit_row_count=5,
                fits_per_observation=2,
            ),
        ),
        candidates=(
            RoleNeutralBenchmarkCandidate(
                name="cpu-fresh",
                accelerator_count=0,
                concurrency_per_device=1,
                host_cpu_budget=2,
                executor_mode="fresh_per_fit",
            ),
            RoleNeutralBenchmarkCandidate(
                name="cpu-persistent",
                accelerator_count=0,
                concurrency_per_device=1,
                host_cpu_budget=2,
                executor_mode="persistent_slots",
            ),
        ),
        scientific_reference_candidate="cpu-fresh",
        multi_device_baselines=(),
        resource_performance_safety=safety,
        preflight_compression_benchmark=(
            CompactPreflightCompressionBenchmarkConfig(
                codecs=("none", "zstd"),
                warmup_repetitions_per_codec=0,
                measured_repetitions_per_codec=1,
            )
        ),
        gpu_sample_interval_seconds=0.01,
        warmup_observations_per_candidate_scope=1,
    )


def _deployment_payload(
    *,
    workflow_root: Path,
    context_root: Path,
    config: RoleNeutralBenchmarkConfig,
    request_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": (provider.ROLE_NEUTRAL_BENCHMARK_WORKLOAD_DEPLOYMENT_SCHEMA),
        "workflow_root": str(workflow_root),
        "expected_workflow_request_sha256": request_sha256,
        "prepared_context_root": str(context_root),
        "expected_benchmark_config_sha256": identity_sha256(config.as_dict()),
        "representative_scope_selectors": [
            {
                "scope_label": "arbitrary-outer",
                "logical_scope_kind": "full_outer",
                "ordinal": 0,
            },
            {
                "scope_label": "arbitrary-inner",
                "logical_scope_kind": "exact_inner",
                "ordinal": 0,
            },
        ],
    }


def _write_paused_fixture(root: Path) -> tuple[str, dict[str, object]]:
    root.mkdir()
    phase_sequence = [
        "input_preparation",
        "embedding_cache",
        "stage1_preflight",
        "stage1_modeling",
        "terminal_validation",
    ]
    request_body = {
        "schema_version": WORKFLOW_SCHEMA,
        "phase_sequence": phase_sequence,
        "resolved_stage1_gpu_ids": [],
        "stage1_execution_device_count": 1,
        "stage1_execution_profile": {
            "schema_version": "portable_stage1_execution_profile_v3",
            "resource_kind": "cpu",
            "device_count": 1,
            "scope_workers_per_device": 1,
            "executor_mode": "persistent_slots",
            "selection_method": "operator_configured",
            "selected_candidate": None,
            "benchmark_result_sha256": None,
            "benchmark_result_locator": None,
            "benchmark_workload_deployment_sha256": None,
            "benchmark_workload_deployment_locator": None,
        },
        "stage1_scope_workers_per_gpu": 1,
        "stage1_preflight_workers": 3,
        "tfidf_workers": 2,
    }
    request_sha256 = _sha(request_body)
    request = {
        **request_body,
        "request_sha256": request_sha256,
    }
    (root / "immutable_run_request.json").write_text(
        json.dumps(request),
        encoding="utf-8",
    )
    for phase in provider._PAUSED_PREFIX:
        attempt = root / "phases" / phase / "attempt_fixture"
        attempt.mkdir(parents=True)
        (attempt / "payload.bin").write_bytes(f"arbitrary-{phase}".encode("utf-8"))
        body = {
            "schema_version": WORKFLOW_PHASE_MANIFEST_SCHEMA,
            "phase": phase,
            "status": "complete",
            "request_sha256": request_sha256,
            "attempt_dir": str(attempt.resolve()),
            "result": {"terminal_files": []},
            "artifacts": _attempt_tree_artifacts(attempt),
        }
        (attempt.parent / "complete_manifest.json").write_text(
            json.dumps({**body, "content_sha256": _sha(body)}),
            encoding="utf-8",
        )
    progress = {
        "schema_version": WORKFLOW_PROGRESS_SCHEMA,
        "request_sha256": request_sha256,
        "status": "paused",
        "phase_sequence": phase_sequence,
        "planned_phase_count": len(phase_sequence),
        "completed_phases": list(provider._PAUSED_PREFIX),
        "completed_phase_count": len(provider._PAUSED_PREFIX),
        "current_phase": None,
        "remaining_phase_count": (len(phase_sequence) - len(provider._PAUSED_PREFIX)),
        "stage1_gpu_ids": [],
        "stage1_execution_device_count": 1,
        "stage1_execution_profile": request["stage1_execution_profile"],
        "stage1_scope_workers_per_gpu": 1,
        "stage1_preflight_workers": 3,
        "tfidf_workers": 2,
        "updated_at": "fixture-time",
        "error": None,
    }
    (root / "workflow_progress.json").write_text(
        json.dumps(progress),
        encoding="utf-8",
    )
    return request_sha256, request


def test_one_owner_plans_use_configured_arbitrary_sizes_and_content() -> None:
    source = _arbitrary_plan()
    outer = provider._one_physical_owner_plan(
        source=source,
        selector=provider.RoleNeutralBenchmarkScopeSelector(
            scope_label="outer",
            logical_scope_kind="full_outer",
            ordinal=0,
        ),
        fit_row_count=7,
    )
    inner = provider._one_physical_owner_plan(
        source=source,
        selector=provider.RoleNeutralBenchmarkScopeSelector(
            scope_label="inner",
            logical_scope_kind="exact_inner",
            ordinal=0,
        ),
        fit_row_count=5,
    )

    assert outer.physical_scopes[0].fit_row_ids == tuple(range(7))
    assert inner.physical_scopes[0].fit_row_ids == tuple(range(5))
    assert len(inner.physical_scopes) == 1
    assert {value.scope_kind for value in inner.scopes} == {
        "exact_inner",
        "cumulative_spent",
    }


def test_embedding_capacity_is_reconstructed_only_from_scientific_request() -> None:
    encoder = {
        "prompt_policy": "disabled",
        "prompt_name": None,
        "output_value": "sentence_embedding",
        "precision": "float32",
        "convert_to_numpy": True,
        "convert_to_tensor": False,
        "truncate_dim": None,
        "pooling_output_policy": "single_process_sentence_embedding_v1",
        "model_dtype": "float32",
        "stored_array_dtype": "float32",
        "zero_vector_policy": "reject",
    }
    configured = {
        "embedding_chunk_size_words": 37,
        "embedding_chunk_overlap_words": 6,
        "embedding_max_chunks": 113,
        "embedding_chunk_selection": "last",
        "embedding_max_seq_length": None,
        "embedding_normalize": True,
        "embedding_encoder": encoder,
    }
    request = {
        **configured,
        "portable_scientific_spec": {
            "text_windows": dict(configured),
        },
    }

    reconstructed = provider._embedding_chunk_configuration(request)

    assert reconstructed["chunk_size_words"] == 37
    assert reconstructed["chunk_overlap_words"] == 6
    assert reconstructed["max_chunks"] == 113
    assert reconstructed["chunk_selection"] == "last"
    assert reconstructed["truncate_dim"] is None


def test_paused_workflow_authentication_fails_on_registered_byte_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workflow = tmp_path / "arbitrary workflow"
    request_sha256, _request = _write_paused_fixture(workflow)
    scratch = tmp_path / "arbitrary scratch"
    scratch.mkdir()
    config = _benchmark_config()
    deployment = provider.RoleNeutralBenchmarkWorkloadDeployment.from_mapping(
        _deployment_payload(
            workflow_root=workflow,
            context_root=scratch / "prepared",
            config=config,
            request_sha256=request_sha256,
        )
    )
    monkeypatch.setattr(
        provider,
        "_revalidate_request_bound_external_inputs",
        lambda _request: None,
    )

    authenticated = provider._authenticate_paused_stage1_preflight(deployment)
    assert set(authenticated.phases) == set(provider._PAUSED_PREFIX)
    first_output = tmp_path / "deployment-one.json"
    second_output = tmp_path / "deployment-two.json"
    selectors = deployment.representative_scope_selectors
    first = provider.write_authenticated_role_neutral_benchmark_workload_deployment(
        workflow_root=workflow,
        benchmark_config=config,
        prepared_context_root=scratch / "prepared",
        representative_scope_selectors=selectors,
        output_path=first_output,
    )
    provider.write_authenticated_role_neutral_benchmark_workload_deployment(
        workflow_root=workflow,
        benchmark_config=config,
        prepared_context_root=scratch / "prepared",
        representative_scope_selectors=selectors,
        output_path=second_output,
    )
    assert first == deployment
    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_output.stat().st_mode & 0o777 == 0o444

    payload = workflow / "phases" / "embedding_cache" / "attempt_fixture" / "payload.bin"
    payload.write_bytes(b"substituted")
    with pytest.raises(ValueError, match="attempt tree changed"):
        provider._authenticate_paused_stage1_preflight(deployment)


def test_paused_workflow_rejects_any_later_partial_phase_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workflow = tmp_path / "workflow"
    request_sha256, _request = _write_paused_fixture(workflow)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    deployment = provider.RoleNeutralBenchmarkWorkloadDeployment.from_mapping(
        _deployment_payload(
            workflow_root=workflow,
            context_root=scratch / "prepared",
            config=_benchmark_config(),
            request_sha256=request_sha256,
        )
    )
    monkeypatch.setattr(
        provider,
        "_revalidate_request_bound_external_inputs",
        lambda _request: None,
    )
    partial = workflow / "phases" / "stage1_modeling" / "attempt_partial"
    partial.mkdir(parents=True)
    (partial / "unsealed.bin").write_bytes(b"partial")

    with pytest.raises(ValueError, match="advanced beyond stage1_preflight"):
        provider._authenticate_paused_stage1_preflight(deployment)


def test_provider_binds_arbitrary_paths_rows_and_typed_prepared_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _benchmark_config()
    workflow = tmp_path / "workflow elsewhere"
    workflow.mkdir()
    scratch = tmp_path / "scratch elsewhere"
    scratch.mkdir()
    request_sha256 = hashlib.sha256(b"workflow-request").hexdigest()
    deployment_path = tmp_path / "workload.json"
    deployment_path.write_text(
        json.dumps(
            _deployment_payload(
                workflow_root=workflow,
                context_root=scratch / "prepared",
                config=config,
                request_sha256=request_sha256,
            )
        ),
        encoding="utf-8",
    )
    plan = _arbitrary_plan()
    authenticated = provider.AuthenticatedPausedStage1Preflight(
        root=workflow,
        request={
            "portable_scientific_spec": {
                "architecture_profiles": {"fixture": {"closed": True}},
            },
            "runtime_compatibility_class": "fixture-runtime",
            "scientific_identity": {
                "scientific_sha256": "1" * 64,
            },
        },
        phases={
            "stage1_preflight": {
                "content_sha256": "2" * 64,
            },
        },
    )
    monkeypatch.setattr(
        provider,
        "_authenticate_paused_stage1_preflight",
        lambda _deployment: authenticated,
    )
    monkeypatch.setattr(
        provider,
        "_stage1_build_options",
        lambda **_kwargs: object(),
    )

    class FakeStage1Builder:
        def __init__(self, _options):
            pass

        def prepare(self):
            return SimpleNamespace(
                stage1_scope_plan=plan,
                cluster_preflight_artifact_handle="preflight-handle",
            )

    class FakeFactoryBuilder:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _prepared):
            return object()

    monkeypatch.setattr(
        provider,
        "ProductionStage1BundleBuilder",
        FakeStage1Builder,
    )
    monkeypatch.setattr(
        provider,
        "PreparedBuildRoleNeutralProducerFactoriesBuilder",
        FakeFactoryBuilder,
    )

    @dataclass(frozen=True)
    class FakeBoundProcessExecutor:
        max_workers_per_resource: int

    class FakeProcessExecutor:
        def __init__(self, *, max_workers_per_resource: int):
            self.max_workers_per_resource = max_workers_per_resource

        def bind_prepared(self, **_kwargs):
            return FakeBoundProcessExecutor(
                max_workers_per_resource=self.max_workers_per_resource,
            )

    @dataclass(frozen=True)
    class FakeBoundPersistentExecutor:
        max_workers_per_resource: int
        worker_parameters: dict[str, str]

    class FakePersistentExecutor:
        def __init__(self, *, max_workers_per_resource: int):
            self.max_workers_per_resource = max_workers_per_resource

        def bind_prepared(self, **_kwargs):
            return FakeBoundPersistentExecutor(
                max_workers_per_resource=self.max_workers_per_resource,
                worker_parameters={
                    "prepared_context_manifest_path": str(
                        (tmp_path / "prepared-context-manifest.json").resolve()
                    )
                },
            )

    monkeypatch.setattr(
        provider,
        "ProcessIsolatedRoleNeutralPhysicalOwnerExecutor",
        FakeProcessExecutor,
    )
    monkeypatch.setattr(
        provider,
        "PersistentSpawnRoleNeutralPhysicalOwnerExecutor",
        FakePersistentExecutor,
    )
    monkeypatch.setattr(
        provider,
        "load_prepared_stage1_context",
        lambda _path: SimpleNamespace(content_root_sha256="5" * 64),
    )
    monkeypatch.setattr(
        provider,
        "_immutable_inputs",
        lambda _authenticated: (
            ImmutableInputObservation(
                content_sha256=hashlib.sha256(b"input").hexdigest(),
                size_bytes=17,
            ),
        ),
    )

    workloads = provider.build_authenticated_role_neutral_benchmark_workloads(
        config,
        deployment_path,
    )
    assert set(workloads) == {"arbitrary-outer", "arbitrary-inner"}
    assert workloads["arbitrary-outer"].fit_row_count == 7
    assert workloads["arbitrary-inner"].fit_row_count == 5
    assert workloads["arbitrary-inner"].plan.physical_scopes[0].fit_row_ids == tuple(range(5))
    selected_executor = workloads[
        "arbitrary-inner"
    ].physical_owner_executor_builder("fresh_per_fit", 2)
    assert isinstance(selected_executor, FakeBoundProcessExecutor)
    assert selected_executor.max_workers_per_resource == 2
    selected_persistent = workloads[
        "arbitrary-inner"
    ].physical_owner_executor_builder("persistent_slots", 2)
    assert isinstance(selected_persistent, FakeBoundPersistentExecutor)
    assert selected_persistent.max_workers_per_resource == 2
    assert (
        workloads["arbitrary-inner"]
        .source_binding
        .prepared_stage1_context_content_root_sha256
        == "5" * 64
    )
    assert (
        workloads[
            "arbitrary-inner"
        ].preflight_compression_source_builder()
        == "preflight-handle"
    )


def test_workload_deployment_and_benchmark_identity_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _benchmark_config()
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    payload = _deployment_payload(
        workflow_root=workflow,
        context_root=scratch / "prepared",
        config=config,
        request_sha256=hashlib.sha256(b"request").hexdigest(),
    )
    payload["expected_benchmark_config_sha256"] = hashlib.sha256(b"another-config").hexdigest()
    deployment_path = tmp_path / "workload.json"
    deployment_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        provider,
        "_authenticate_paused_stage1_preflight",
        lambda _deployment: pytest.fail(
            "identity mismatch must fail before workflow bytes are opened"
        ),
    )
    with pytest.raises(ValueError, match="benchmark config differs"):
        provider.build_authenticated_role_neutral_benchmark_workloads(
            config,
            deployment_path,
        )

    relative = dict(payload)
    relative["workflow_root"] = "relative/workflow"
    with pytest.raises(ValueError, match="workflow_root must be an absolute"):
        provider.RoleNeutralBenchmarkWorkloadDeployment.from_mapping(relative)


def test_provider_source_has_no_checked_cohort_sizes_or_locator_defaults() -> None:
    source = Path(provider.__file__).read_text(encoding="utf-8")
    assert "cuda:0" not in source
    assert "cuda:1" not in source
    assert "fit_row_count=7" not in source
    assert "fit_row_count=5" not in source
    assert "patient_id" not in source
    assert "clinical_text" not in source
    assert "portable_cluster_preflight_v2=True" in source

    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "example_configs"
            / "portable_role_neutral_benchmark_workload.deployment.schema.json"
        ).read_text(encoding="utf-8")
    )
    runtime_fields = {
        field.name for field in fields(provider.RoleNeutralBenchmarkWorkloadDeployment)
    }
    assert set(schema["required"]) == runtime_fields
    assert set(schema["properties"]) == runtime_fields
    assert schema["additionalProperties"] is False
