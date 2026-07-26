from __future__ import annotations

import copy
import json
import stat
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import oci.inference.production_stage1_bundle as bundle_module
import oci.inference.production_stage1_cluster_preflight_artifact as artifact_module
from oci.inference.neural_query_agentic_forest import NeuralQueryAgenticForestConfig
from oci.inference.production_stage1_bundle import (
    ProductionStage1BundleBuilder,
    Stage1BundleBuildOptions,
)
from oci.inference.production_stage1_cluster_preflight_artifact import (
    seal_production_stage1_cluster_preflight_artifact,
)
from tests.test_production_stage1_bundle import _valid_config
from tests.stage1_test_support import PHYSICAL_FIT_IDENTITY


def _addressed(body):
    return {
        **body,
        "content_sha256": bundle_module._sha256_json(body),
    }


def _make_artifact_writable(root: Path) -> None:
    root.chmod(stat.S_IRWXU)
    for path in root.iterdir():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_modeling_prepare_consumes_sealed_preflight_without_recomputation(
    tmp_path: Path,
    monkeypatch,
):
    config, _model_dir = _valid_config(tmp_path)
    dataset_path = tmp_path / "cohort.parquet"
    dataset_path.write_bytes(b"authenticated parquet container placeholder")
    config_path = tmp_path / "stage1.json"
    config_path.write_text(
        json.dumps({"applied_inference": asdict(config)}),
        encoding="utf-8",
    )
    query_config_path = tmp_path / "query.json"
    query_config_path.write_text(
        json.dumps(asdict(NeuralQueryAgenticForestConfig()), sort_keys=True),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "embedding_cache"
    cache_dir.mkdir()
    rows = []
    for repetition in range(12):
        for treatment, outcome in ((0, 0), (0, 1), (1, 0), (1, 1)):
            rows.append(
                {
                    "person_key": f"person-{len(rows)}",
                    "clinical_text": f"safe baseline text {repetition}",
                    "treatment_indicator": treatment,
                    "outcome_indicator": outcome,
                }
            )
    projected = pd.DataFrame(rows)

    def fake_read_parquet(_path, *, columns):
        return projected.loc[:, columns].copy()

    class FakeCache:
        row_count = len(projected)
        metadata = {
            "sentence_model_name": config.architecture.multi_model_forest.embedding_contrast.model_name,
            "chunk_size_words": config.architecture.multi_model_forest.embedding_contrast.chunk_size_words,
            "chunk_overlap_words": config.architecture.multi_model_forest.embedding_contrast.chunk_overlap_words,
            "max_chunks": config.architecture.multi_model_forest.embedding_contrast.max_chunks,
            "normalize_embeddings": config.architecture.multi_model_forest.embedding_contrast.normalize_embeddings,
            "chunk_selection": "last",
            "max_seq_length": config.architecture.multi_model_forest.embedding_contrast.max_seq_length,
            "production_provenance": {
                "chunk_configuration": {
                    "chunk_size_words": config.architecture.multi_model_forest.embedding_contrast.chunk_size_words,
                    "chunk_overlap_words": config.architecture.multi_model_forest.embedding_contrast.chunk_overlap_words,
                    "max_chunks": config.architecture.multi_model_forest.embedding_contrast.max_chunks,
                    "chunk_selection": "last",
                    "normalize_embeddings": config.architecture.multi_model_forest.embedding_contrast.normalize_embeddings,
                    "max_seq_length": config.architecture.multi_model_forest.embedding_contrast.max_seq_length,
                }
            },
        }

        def __init__(self, path):
            assert Path(path) == cache_dir

        def bind_spent(self, row_ids, texts):
            assert tuple(row_ids) == tuple(range(len(projected)))
            assert len(texts) == len(projected)
            return SimpleNamespace(token_bounded_row_ids=())

        def identity(self):
            return {
                "cache_sha256": "d" * 64,
                "row_count": self.row_count,
            }

    monkeypatch.setattr(bundle_module.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(
        bundle_module,
        "SpentOnlyFrozenChunkEmbeddingCache",
        FakeCache,
    )
    monkeypatch.setattr(
        bundle_module,
        "validate_published_production_embedding_cache",
        lambda **_kwargs: {
            "schema_version": "production_arbitrary_cohort_embedding_cache_result_v2",
            "provider_identity": FakeCache(cache_dir).identity(),
        },
    )
    scope = {
        "scope_id": "outer_001_full",
        "scope_kind": "full_outer",
        "outer_fold": 1,
        "inner_fold": None,
        "context_epoch": None,
        "provider_inner_fold": None,
        "fit_row_count": 32,
        "fit_row_order_fingerprint": "a" * 64,
        "heldout_row_count": 16,
        "heldout_row_order_fingerprint": "b" * 64,
        "cluster_fit_identity": {"scope_id": "outer_001_full"},
    }
    audit_body = {
        "schema_version": "fixture_cluster_audit",
        "scope_count": 1,
        "scope_order": [scope["scope_id"]],
        "scopes": [scope],
    }
    cluster_audit = _addressed(audit_body)
    calls = 0

    def one_preflight(**_kwargs):
        nonlocal calls
        calls += 1
        return copy.deepcopy(cluster_audit)

    monkeypatch.setattr(
        bundle_module,
        "build_embedding_cluster_feasibility_audit",
        one_preflight,
    )
    common = Stage1BundleBuildOptions(
        dataset_path=dataset_path,
        config_path=config_path,
        embedding_cache_dir=cache_dir,
        output_dir=tmp_path / "output",
        unit_id_column="person_key",
        initial_training_partitions=3,
        physical_fit_identity=PHYSICAL_FIT_IDENTITY,
        query_config_path=query_config_path,
        dry_run=True,
    )
    preflight = ProductionStage1BundleBuilder(common).prepare()
    assert calls == 1
    monkeypatch.setattr(
        artifact_module,
        "_validate_scientific_audit",
        lambda audit, **_kwargs: copy.deepcopy(dict(audit)),
    )
    artifact = seal_production_stage1_cluster_preflight_artifact(
        output_dir=(tmp_path / "sealed_preflight").resolve(),
        audit=cluster_audit,
        stage1_request=preflight.request,
        config=preflight.config,
        registry=preflight.registry,
        registry_content_sha256=preflight.registry_content_sha256,
        embedding_cache_identity=preflight.embedding_cache_identity,
    )

    def forbid_recompute(**_kwargs):
        raise AssertionError("supervised modeling recomputed clustered preflight")

    monkeypatch.setattr(
        bundle_module,
        "build_embedding_cluster_feasibility_audit",
        forbid_recompute,
    )
    try:
        modeling = ProductionStage1BundleBuilder(
            replace(
                common,
                dry_run=False,
                cluster_preflight_manifest_path=artifact.manifest_path,
            )
        ).prepare()
        assert modeling.request == preflight.request
        assert modeling.request_sha256 == preflight.request_sha256
        assert modeling.embedding_cluster_feasibility_audit == cluster_audit
        assert modeling.cluster_preflight_artifact_identity == artifact.identity()
    finally:
        _make_artifact_writable(artifact.root)
