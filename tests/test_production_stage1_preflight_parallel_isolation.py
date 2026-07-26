from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from oci.config import AppliedInferenceConfig
import oci.inference.production_stage1_bundle as bundle
import oci.inference.production_stage1_preflight_scope_inputs as scope_inputs
from tests.cluster_local_embedding_test_support import (
    cluster_local_embedding_config,
)
from tests.semantic_witness_test_support import semantic_witness_config


def test_parallel_preflight_submits_35_row_views_over_one_shared_cache(
    tmp_path: Path,
    monkeypatch,
):
    config = AppliedInferenceConfig()
    config.seed = 42
    config.architecture.multi_model_forest.embedding_contrast.cluster_local_scientific = (
        cluster_local_embedding_config()
    )
    config.dataset_path = "/global/prepared-label-cohort.parquet"
    modeling = pd.DataFrame(
        {
            config.text_column: [f"note {index}" for index in range(35)],
            config.treatment_column: [float(index % 2) for index in range(35)],
            config.outcome_column: [
                float((index // 2) % 2) for index in range(35)
            ],
        }
    )
    physical_scopes = tuple(
        {
            "scope_id": f"scope_{index:03d}",
            "scope_kind": "full_outer",
            "outer_fold": (index % 5) + 1,
            "inner_fold": None,
            "context_epoch": None,
            "provider_inner_fold": None,
            "fit_row_ids": (index,),
            "scope_seed": bundle.derive_stage1_group_seed(42, (index,)),
            "heldout_row_ids": (100 + index,),
        }
        for index in range(35)
    )
    aliases = tuple(
        {
            **physical_scopes[30 + index],
            "scope_id": f"alias_{index:03d}",
            "scope_kind": "cumulative_spent",
            "context_epoch": 1,
        }
        for index in range(5)
    )
    scopes = (*physical_scopes, *aliases)
    shared_reference_path = (
        tmp_path / "scope_inputs" / "shared_embedding_cache_reference.json"
    )
    shared_reference_sha256 = "a" * 64
    payloads = tuple(
        {
            "schema_version": "production_stage1_preflight_worker_payload_v2",
            "scope_id": scope["scope_id"],
            "manifest_path": str(
                tmp_path / "scope_inputs" / scope["scope_id"] / "manifest.json"
            ),
            "manifest_content_sha256": f"{index:064x}",
            "shared_cache_reference_path": str(shared_reference_path),
            "shared_cache_reference_content_sha256": shared_reference_sha256,
        }
        for index, scope in enumerate(physical_scopes)
    )
    published = SimpleNamespace(worker_payloads=lambda: payloads)
    publisher_calls = []

    def fake_publish(**kwargs):
        publisher_calls.append(kwargs)
        return published

    monkeypatch.setattr(
        scope_inputs,
        "publish_preflight_scope_inputs",
        fake_publish,
    )
    monkeypatch.setattr(
        bundle,
        "_embedding_cluster_feasibility_scopes",
        lambda _registry, *, initial_training_partitions, global_seed: scopes,
    )
    captured = []
    parallel_options = []

    class FakeParallel:
        def __init__(self, **kwargs):
            parallel_options.append(kwargs)

        def __call__(self, tasks):
            output = []
            for function, args, kwargs in tasks:
                assert function is bundle._embedding_cluster_preflight_loky_scope
                assert not kwargs
                payload = args[0]
                captured.append(payload)
                output.append({"scope_id": payload["scope_id"]})
            return output

    monkeypatch.setattr(bundle, "Parallel", FakeParallel)
    result = bundle.build_embedding_cluster_feasibility_audit(
        modeling_data=modeling,
        config=config,
        embedding_cache=SimpleNamespace(cache_dir=Path("/global/cache")),
        embedding_cache_identity={"content_sha256": "f" * 64},
        registry={"dataset_row_count": 35},
        registry_content_sha256="e" * 64,
        initial_training_partitions=3,
        semantic_witness_scientific_config=semantic_witness_config(),
        preflight_workers=8,
        preflight_scope_input_root=(tmp_path / "scope_inputs").resolve(),
        _return_scope_audits=True,
    )
    assert len(publisher_calls) == 1
    assert publisher_calls[0]["scopes"] == physical_scopes
    assert len(captured) == 35
    assert [row["scope_id"] for row in captured] == [
        scope["scope_id"] for scope in physical_scopes
    ]
    assert len({row["manifest_path"] for row in captured}) == 35
    assert {
        row["shared_cache_reference_path"] for row in captured
    } == {str(shared_reference_path)}
    assert {
        row["shared_cache_reference_content_sha256"] for row in captured
    } == {shared_reference_sha256}
    assert {
        row["schema_version"] for row in captured
    } == {"production_stage1_preflight_worker_payload_v2"}
    assert {
        row["initial_training_partitions"] for row in captured
    } == {3}
    assert parallel_options == [{"n_jobs": 8, "batch_size": 1, "pre_dispatch": "all"}]
    assert [row["scope_id"] for row in result["_scope_audits"]] == [
        scope["scope_id"] for scope in physical_scopes
    ]
    serialized = json.dumps(captured, sort_keys=True)
    assert "/global/prepared-label-cohort.parquet" not in serialized
    assert "/global/cache" not in serialized
    assert '"cache_dir"' not in serialized
    assert '"global_embedding_cache_path"' not in serialized
