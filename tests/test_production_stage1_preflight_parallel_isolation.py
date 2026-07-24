from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from oci.config import AppliedInferenceConfig
import oci.inference.production_stage1_bundle as bundle
import oci.inference.production_stage1_preflight_scope_inputs as scope_inputs


def test_parallel_preflight_submits_exactly_one_private_payload_per_scope(
    tmp_path: Path,
    monkeypatch,
):
    config = AppliedInferenceConfig()
    config.dataset_path = "/global/prepared-label-cohort.parquet"
    modeling = pd.DataFrame(
        {
            config.text_column: ["note"],
            config.treatment_column: [0.0],
            config.outcome_column: [1.0],
        }
    )
    scopes = tuple(
        {
            "scope_id": f"scope_{index:03d}",
            "scope_kind": "full_outer",
            "outer_fold": index + 1,
            "inner_fold": None,
            "context_epoch": None,
            "provider_inner_fold": None,
            "fit_row_ids": (0,),
            "heldout_row_ids": (index + 1,),
        }
        for index in range(40)
    )
    payloads = tuple(
        {
            "schema_version": "production_stage1_preflight_worker_payload_v1",
            "scope_id": scope["scope_id"],
            "manifest_path": str(tmp_path / "private" / scope["scope_id"] / "manifest.json"),
            "manifest_content_sha256": f"{index:064x}",
        }
        for index, scope in enumerate(scopes)
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
        lambda _registry: scopes,
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
        registry={"dataset_row_count": 1},
        registry_content_sha256="e" * 64,
        preflight_workers=8,
        preflight_scope_input_root=(tmp_path / "private").resolve(),
        _return_scope_audits=True,
    )
    assert len(publisher_calls) == 1
    assert publisher_calls[0]["scopes"] == scopes
    assert len(captured) == 40
    assert [row["scope_id"] for row in captured] == [scope["scope_id"] for scope in scopes]
    assert len({row["manifest_path"] for row in captured}) == 40
    assert parallel_options == [{"n_jobs": 8, "batch_size": 1, "pre_dispatch": "all"}]
    assert [row["scope_id"] for row in result["_scope_audits"]] == [
        scope["scope_id"] for scope in scopes
    ]
    serialized = json.dumps(captured, sort_keys=True)
    assert "/global/prepared-label-cohort.parquet" not in serialized
    assert "/global/cache" not in serialized
