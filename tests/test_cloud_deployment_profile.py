from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/build_cloud_deployment_profile.py").resolve()
    spec = importlib.util.spec_from_file_location("cloud_profile_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> Namespace:
    root = Path.cwd().resolve()
    return Namespace(
        base=root
        / "example_configs/portable_all_evidence_deployment_nsclc.stage1-only.example.json",
        target=tmp_path / "deployment.json",
        dataset=root
        / "synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet",
        durable_root=tmp_path / "durable",
        scratch_root=tmp_path / "scratch",
        embedding_model=tmp_path / "embedding-model",
        htr_model=tmp_path / "htr-model",
        stage2_tokenizer=tmp_path / "stage2-tokenizer",
        stage1_profile=root
        / "example_configs/production_all_evidence_stage1_full.json",
        query_profile=root
        / "example_configs/production_all_evidence_neural_query_full.json",
        cpu_budget=64,
        preflight_memory_budget=64 * 1024**3,
        preflight_owner_peak=8 * 1024**3,
        preflight_lanes=8,
        embedding_batch_size=8,
        max_workers_per_device=4,
        estimated_device_memory_per_owner=8 * 1024**3,
        device_memory_reserve=6 * 1024**3,
        estimated_host_memory_per_owner=8 * 1024**3,
        host_memory_budget_fraction=0.75,
        minimum_cpu_threads_per_owner=1,
        endpoint="http://127.0.0.1:8002/v1",
        endpoint_model=(
            "nvidia/Gemma-4-31B-IT-NVFP4@"
            "4135a98a9b728a548947683219633b25682223ac"
        ),
    )


def test_cloud_profile_compiles_eight_disjoint_lanes_deterministically(
    tmp_path: Path,
) -> None:
    module = _module()
    args = _args(tmp_path)

    first = module.build_profile(args)
    first_bytes = args.target.read_bytes()
    second = module.build_profile(args)

    assert first == second
    assert args.target.read_bytes() == first_bytes
    assert first.devices == tuple(f"cuda:{index}" for index in range(8))
    assert first.embedding_batch_size == 8
    assert first.endpoint == "http://127.0.0.1:8002/v1"
    assert first.endpoint_model == (
        "nvidia/Gemma-4-31B-IT-NVFP4@"
        "4135a98a9b728a548947683219633b25682223ac"
    )
    assert first.stage1_execution.max_parallel_owners == 32
    assert first.stage1_execution.scope_workers_per_device == 4
    assert (
        first.stage1_execution.owner_capacity_policy.mode
        == "resource_autodetect"
    )
    assert first.stage1_execution.htr_operational_controls.fold_parallelism == 1
    assert first.stage1_execution.htr_operational_controls.fold_slots_per_device == 1
    assert (
        first.stage1_execution.neural_query_topology.mode
        == "one_context_per_selected_device"
    )

    changed = Namespace(**{**vars(args), "cpu_budget": 63})
    with pytest.raises(ValueError, match="existing current deployment profile differs"):
        module.build_profile(changed)


def test_cloud_profile_reuses_matching_legacy_profile_for_resume(
    tmp_path: Path,
) -> None:
    module = _module()
    args = _args(tmp_path)
    module.build_profile(args)
    legacy = json.loads(args.target.read_text(encoding="utf-8"))
    legacy["schema_version"] = (
        "portable_all_evidence_deployment_profile_v9"
    )
    legacy_stage1 = legacy["stage1_execution"]
    legacy_stage1["schema_version"] = (
        "portable_stage1_execution_profile_v8"
    )
    legacy_stage1.pop("owner_capacity_policy")
    legacy_bytes = (
        json.dumps(legacy, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    args.target.write_bytes(legacy_bytes)

    resumed = module.build_profile(args)

    assert args.target.read_bytes() == legacy_bytes
    assert resumed.stage1_execution.owner_capacity_policy.mode == "fixed"


def test_cloud_profile_rejects_changed_inputs_against_legacy_profile(
    tmp_path: Path,
) -> None:
    module = _module()
    args = _args(tmp_path)
    module.build_profile(args)
    legacy = json.loads(args.target.read_text(encoding="utf-8"))
    legacy["schema_version"] = (
        "portable_all_evidence_deployment_profile_v9"
    )
    legacy_stage1 = legacy["stage1_execution"]
    legacy_stage1["schema_version"] = (
        "portable_stage1_execution_profile_v8"
    )
    legacy_stage1.pop("owner_capacity_policy")
    args.target.write_text(
        json.dumps(legacy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    changed = Namespace(**{**vars(args), "embedding_batch_size": 7})
    with pytest.raises(
        ValueError,
        match="existing current deployment profile differs",
    ):
        module.build_profile(changed)
