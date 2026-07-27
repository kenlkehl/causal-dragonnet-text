from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from oci.inference.portable_resource_scheduler import (
    GPUResource,
    ResourceInventory,
)
from oci.inference.portable_workflow_spec import (
    DeploymentProfile,
    ScientificWorkflowSpec,
)
from oci.inference.production_all_evidence_workflow import (
    PHASES,
    _default_portable_role_neutral_hooks,
    build_parser,
    options_from_args,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_SPEC = (
    REPOSITORY_ROOT
    / "example_configs"
    / "portable_all_evidence_scientific_nsclc.json"
)
DEPLOYMENT_PROFILE = (
    REPOSITORY_ROOT
    / "example_configs"
    / "portable_all_evidence_deployment_nsclc.acceptance.json"
)
R14_HIGH_POWERED_PROFILE = (
    REPOSITORY_ROOT
    / "example_configs"
    / "portable_all_evidence_deployment_nsclc.r14-high-powered.json"
)
V5_PREPARATION = (
    REPOSITORY_ROOT
    / "artifacts"
    / "production_all_evidence_one_conf_one_mod_1000_v5_parallel_stage1"
    / "phases"
    / "input_preparation"
    / "complete_manifest.json"
)
V5_EMBEDDING_CACHE = (
    REPOSITORY_ROOT
    / "artifacts"
    / "production_all_evidence_one_conf_one_mod_1000_v5_parallel_stage1"
    / "phases"
    / "embedding_cache"
    / "complete_manifest.json"
)
V4_CLUSTER_PREFLIGHT = (
    REPOSITORY_ROOT
    / "artifacts"
    / "production_all_evidence_one_conf_one_mod_1000_v4_parallel_stage1"
    / "phases"
    / "stage1_preflight"
    / "attempt_20260723T195805360899Z"
    / "cluster_preflight"
    / "cluster_preflight_manifest.json"
)


def test_acceptance_deployment_is_closed_full_workflow_configuration() -> None:
    scientific = ScientificWorkflowSpec.from_json(SCIENTIFIC_SPEC)
    deployment = DeploymentProfile.from_json(DEPLOYMENT_PROFILE)

    assert asdict(scientific.columns) == {
        "unit_id": "patient_id",
        "text": "clinical_text",
        "treatment": "treatment_indicator",
        "outcome": "outcome_indicator",
    }
    assert deployment.endpoint == "http://camus:8010/v1"
    assert (
        deployment.endpoint_model
        == "RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
    )
    assert deployment.stage2_tokenizer_locator == Path(
        "../artifacts/local_models/"
        "gemma4_26b_a4b_it_fp8_dynamic_tokenizer_materialized"
    )
    tokenizer_root = (
        DEPLOYMENT_PROFILE.parent / deployment.stage2_tokenizer_locator
    ).resolve(strict=True)
    assert {path.name for path in tokenizer_root.iterdir()} == {
        "chat_template.jinja",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert not any(path.is_symlink() for path in tokenizer_root.rglob("*"))
    assert deployment.devices == ("auto",)
    assert deployment.stage1_execution.resource_kind == "accelerator"
    assert deployment.stage1_execution.device_count == 2
    assert deployment.stage1_execution.scope_workers_per_device == 1
    assert deployment.stage1_execution.max_parallel_owners == 2
    assert (
        deployment.stage1_execution.neural_query_topology.mode
        == "one_context_per_selected_device"
    )
    assert (
        deployment.stage1_execution.htr_operational_controls.as_dict()
        == {
            "schema_version": (
                "production_role_neutral_htr_operational_controls_v2"
            ),
            "training_batch_size": 8,
            "sentence_encoder_batch_size": 16,
            "data_loader_workers": 0,
            "fold_parallelism": 5,
            "fold_parallel_backend": "processes",
            "fold_slots_per_device": 3,
            "reuse_tokenizer_and_chunk_plans": True,
            "chunk_plan_cache_max_entries": 1000,
            "tokenized_chunk_cache_max_entries": 150000,
        }
    )
    assert (
        deployment.stage1_execution.neural_query_operational_controls.as_dict()
        == {
            "schema_version": (
                "production_role_neutral_neural_query_operational_controls_v1"
            ),
            "inner_fold_parallelism": 3,
            "fold_parallel_backend": "processes",
            "fold_slots_per_device": 3,
            "bank_parallelism": 3,
            "worker_cpu_threads": 1,
        }
    )
    assert (
        deployment.stage1_execution.tfidf_parallel_backend
        == "processes"
    )
    assert deployment.cpu_budget == 8
    assert deployment.forest_operational.requested_host_cpu_budget == 8
    assert deployment.oracle_source == deployment.dataset_path
    assert deployment.oracle_unit_id_column == "patient_id"
    assert deployment.oracle_ite_column == "true_ite_prob"
    with pytest.raises(ValueError, match="one outer owner slot"):
        replace(
            deployment.stage1_execution,
            scope_workers_per_device=2,
            max_parallel_owners=4,
        )
    reuse_disabled = replace(
        deployment.stage1_execution.htr_operational_controls,
        reuse_tokenizer_and_chunk_plans=False,
        chunk_plan_cache_max_entries=0,
        tokenized_chunk_cache_max_entries=0,
    )
    with pytest.raises(ValueError, match="reusable complete"):
        replace(
            deployment.stage1_execution,
            htr_operational_controls=reuse_disabled,
        )

    # Pause/resume and checkpoint selection are operational RunControl values.
    # They must not be smuggled into either immutable typed input file.
    deployment_payload = asdict(deployment)
    scientific_payload = scientific.identity_payload()
    for field in (
        "resume",
        "stop_after",
        "adopt_checkpoints",
        "trust_prior_adoption_attestations",
        "log_level",
        "validation_depth",
    ):
        assert field not in deployment_payload
        assert field not in scientific_payload


def test_acceptance_cli_compiles_offline_stage1_prefix_without_server_access(
    monkeypatch,
) -> None:
    # Exercise the exact ``auto`` resource path without touching host GPUs.
    fake_inventory = ResourceInventory(
        cpu_count=32,
        gpus=(
            GPUResource(
                device="cuda:0",
                uuid="TEST-GPU-0",
                total_memory_bytes=48 * 1024**3,
                free_memory_bytes=47 * 1024**3,
                utilization_percent=0.0,
            ),
            GPUResource(
                device="cuda:1",
                uuid="TEST-GPU-1",
                total_memory_bytes=48 * 1024**3,
                free_memory_bytes=47 * 1024**3,
                utilization_percent=0.0,
            ),
        ),
    )
    monkeypatch.setattr(
        "oci.inference.portable_resource_scheduler.discover_resources",
        lambda: fake_inventory,
    )
    arguments = build_parser().parse_args(
        [
            "--scientific-spec",
            str(SCIENTIFIC_SPEC),
            "--deployment-profile",
            str(DEPLOYMENT_PROFILE),
            "--adopt-checkpoint",
            str(V5_PREPARATION),
            "--adopt-checkpoint",
            str(V5_EMBEDDING_CACHE),
            "--adopt-checkpoint",
            str(V4_CLUSTER_PREFLIGHT),
            "--stop-after",
            "handoff_validation",
            "--validation-depth",
            "fresh_terminal_audit",
        ]
    )

    options = options_from_args(arguments)
    hooks = _default_portable_role_neutral_hooks(options)

    assert options.stage1_only is False
    assert options.endpoint == "http://camus:8010/v1"
    assert options.model_name == "RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
    assert options.run_control.stop_after == "handoff_validation"
    assert options.run_control.validation_depth == "fresh_terminal_audit"
    assert options.stage1_execution_device_count == 2
    assert options.stage1_scope_workers_per_gpu == 1
    assert options.tfidf_parallel_backend == "processes"
    assert options.run_control.adopt_checkpoints == (
        V5_PREPARATION,
        V5_EMBEDDING_CACHE,
        V4_CLUSTER_PREFLIGHT,
    )
    stop_index = PHASES.index(options.run_control.stop_after)
    assert PHASES[: stop_index + 1] == (
        "input_preparation",
        "embedding_cache",
        "stage1_preflight",
        "stage1_modeling",
        "handoff_validation",
    )
    assert hooks.role_neutral_stage1 is not None


def test_r14_high_powered_operational_controls_are_closed_and_bounded(
    monkeypatch,
) -> None:
    scientific = ScientificWorkflowSpec.from_json(SCIENTIFIC_SPEC)
    deployment = DeploymentProfile.from_json(R14_HIGH_POWERED_PROFILE)
    execution = deployment.stage1_execution
    controls = execution.neural_query_operational_controls

    assert deployment.devices == ("cuda:0", "cuda:1")
    assert deployment.cpu_budget == 64
    assert deployment.storage_backend == "local_posix"
    assert execution.max_parallel_owners == 1
    assert (
        execution.neural_query_topology.mode
        == "one_context_spanning_all_selected_devices"
    )
    assert controls.as_dict() == {
        "schema_version": (
            "production_role_neutral_neural_query_operational_controls_v1"
        ),
        "inner_fold_parallelism": 4,
        "fold_parallel_backend": "processes",
        "fold_slots_per_device": 2,
        "bank_parallelism": 3,
        "worker_cpu_threads": 1,
    }
    assert execution.tfidf_parallel_backend == "processes"
    scientific_identity = scientific.identity_payload()
    assert "neural_query_operational_controls" not in scientific_identity
    assert "tfidf_parallel_backend" not in scientific_identity
    assert "devices" not in scientific_identity

    monkeypatch.setattr(
        "oci.inference.portable_resource_scheduler.discover_resources",
        lambda: ResourceInventory(
            cpu_count=64,
            gpus=(
                GPUResource(
                    device="cuda:0",
                    uuid="TEST-H100-0",
                    total_memory_bytes=96 * 1024**3,
                    free_memory_bytes=92 * 1024**3,
                    utilization_percent=0.0,
                ),
                GPUResource(
                    device="cuda:1",
                    uuid="TEST-H100-1",
                    total_memory_bytes=96 * 1024**3,
                    free_memory_bytes=92 * 1024**3,
                    utilization_percent=0.0,
                ),
            ),
        ),
    )
    compiled = options_from_args(
        build_parser().parse_args(
            [
                "--scientific-spec",
                str(SCIENTIFIC_SPEC),
                "--deployment-profile",
                str(R14_HIGH_POWERED_PROFILE),
                "--stop-after",
                "handoff_validation",
                "--validation-depth",
                "fresh_terminal_audit",
            ]
        )
    )
    assert compiled.query_devices == ("cuda:0", "cuda:1")
    assert compiled.cpu_budget == compiled.tfidf_workers == 64
    assert compiled.tfidf_parallel_backend == "processes"
    assert (
        compiled.stage1_execution_profile
        .neural_query_operational_controls
        == controls
    )

    payload = json.loads(R14_HIGH_POWERED_PROFILE.read_text(encoding="utf-8"))
    payload["stage1_execution"].pop("neural_query_operational_controls")
    with pytest.raises(ValueError, match="configure every field exactly"):
        DeploymentProfile.from_mapping(payload)

    oversubscribed_slots = replace(
        controls,
        inner_fold_parallelism=7,
    )
    with pytest.raises(ValueError, match="device-slot capacity"):
        replace(
            execution,
            neural_query_operational_controls=oversubscribed_slots,
        )

    three_cpu_execution = replace(
        execution,
        htr_operational_controls=replace(
            execution.htr_operational_controls,
            fold_parallelism=4,
        ),
    )
    with pytest.raises(ValueError, match="global host CPU budget"):
        replace(
            deployment,
            cpu_budget=3,
            forest_operational=replace(
                deployment.forest_operational,
                requested_host_cpu_budget=3,
            ),
            stage1_execution=three_cpu_execution,
        )
