from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.config import AppliedInferenceConfig
import oci.inference.production_stage1_preflight_scope_inputs as scope_inputs_module
from oci.inference.production_stage1_bundle import load_applied_stage1_config
from oci.inference.production_stage1_preflight_scope_inputs import (
    publish_preflight_scope_inputs,
    validate_preflight_scope_input,
)
from oci.inference.review_spent_evidence_provider import (
    SpentOnlyFrozenChunkEmbeddingCache,
    _FrozenCacheEmbeddingEvidenceGenerator,
)
from tests.semantic_witness_test_support import semantic_witness_config
from tests.cluster_local_embedding_test_support import (
    cluster_local_embedding_config,
)


_SEMANTIC_WITNESS_CONFIG = semantic_witness_config()


def _cache(
    root: Path,
    *,
    texts: list[str],
    matrices: np.ndarray,
    config: AppliedInferenceConfig,
) -> SpentOnlyFrozenChunkEmbeddingCache:
    root.mkdir(parents=True)
    np.save(root / "chunk_embeddings.npy", matrices.astype(np.float32))
    np.save(root / "offsets.npy", np.arange(len(texts) + 1, dtype=np.int64))
    (root / "chunk_texts.jsonl").write_text(
        "".join(json.dumps({"chunks": [text]}, ensure_ascii=False) + "\n" for text in texts),
        encoding="utf-8",
    )
    embedding = config.architecture.multi_model_forest.embedding_contrast
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "num_samples": len(texts),
                "hidden_size": matrices.shape[1],
                "sentence_model_name": embedding.model_name,
                "chunk_size_words": embedding.chunk_size_words,
                "chunk_overlap_words": embedding.chunk_overlap_words,
                "max_chunks": embedding.max_chunks,
                "chunk_selection": embedding.chunk_selection,
                "normalize_embeddings": embedding.normalize_embeddings,
                "max_seq_length": embedding.max_seq_length,
            }
        ),
        encoding="utf-8",
    )
    return SpentOnlyFrozenChunkEmbeddingCache(root)


def _fixture(tmp_path: Path):
    config = AppliedInferenceConfig()
    cluster_config = cluster_local_embedding_config()
    config.architecture.multi_model_forest.embedding_contrast.cluster_local_scientific = (
        cluster_config
    )
    config.architecture.multi_model_agentic_forest.embedding_contrast.cluster_local_scientific = (
        copy.deepcopy(cluster_config)
    )
    source = tmp_path / "prepared.parquet"
    pd.DataFrame({"placeholder": [1]}).to_parquet(source, index=False)
    config.dataset_path = str(source)
    texts = [f"note-{index}" for index in range(8)]
    modeling = pd.DataFrame(
        {
            config.text_column: texts,
            config.treatment_column: np.asarray([0, 1] * 4, dtype=float),
            config.outcome_column: np.asarray([0, 0, 1, 1] * 2, dtype=float),
        }
    )
    matrix = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    cache = _cache(
        tmp_path / "global_cache",
        texts=texts,
        matrices=matrix,
        config=config,
    )
    scopes = (
        {
            "scope_id": "outer_001_full",
            "scope_kind": "full_outer",
            "outer_fold": 1,
            "inner_fold": None,
            "context_epoch": None,
            "provider_inner_fold": None,
            "fit_row_ids": (0, 1, 2, 3),
            "heldout_row_ids": (4, 5, 6, 7),
        },
        {
            "scope_id": "outer_001_inner_001",
            "scope_kind": "exact_inner",
            "outer_fold": 1,
            "inner_fold": 1,
            "context_epoch": None,
            "provider_inner_fold": None,
            "fit_row_ids": (0, 1, 4, 5),
            "heldout_row_ids": (2, 3),
        },
    )
    registry = {"dataset_row_count": 8, "outer_folds": []}
    registry_sha = hashlib.sha256(
        json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return config, source, modeling, cache, matrix, scopes, registry, registry_sha


def test_scope_inputs_expose_only_fit_rows_and_refuse_nonfit_cache(
    tmp_path: Path,
):
    (
        config,
        source,
        modeling,
        cache,
        matrix,
        scopes,
        registry,
        registry_sha,
    ) = _fixture(tmp_path)
    output_root = (tmp_path / "scope_inputs").resolve()
    published = publish_preflight_scope_inputs(
        output_root=output_root,
        modeling_data=modeling,
        config=config,
        embedding_cache=cache,
        embedding_cache_identity=cache.identity(),
        registry=registry,
        registry_content_sha256=registry_sha,
        scopes=scopes,
        source_dataset_path=source,
        global_embedding_cache_path=cache.cache_dir,
        semantic_witness_scientific_config=_SEMANTIC_WITNESS_CONFIG,
    )
    assert [row["scope_id"] for row in published.worker_payloads()] == [
        scope["scope_id"] for scope in scopes
    ]
    operational_identity = published.identity()
    assert operational_identity["scope_inputs_outside_terminal_scientific_artifact"]
    assert operational_identity["scope_order"] == [scope["scope_id"] for scope in scopes]
    assert operational_identity["per_scope_embedding_arrays_copied"] is False
    assert operational_identity["per_scope_chunk_texts_copied"] is False
    shared_references = list(output_root.rglob("shared_embedding_cache_reference.json"))
    assert shared_references == [
        output_root / "shared_embedding_cache_reference.json"
    ]
    assert not [
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and (
            path.suffix == ".npy"
            or path.name == "chunk_texts.jsonl"
        )
    ]
    assert not [
        path
        for path in output_root.rglob("*")
        if path.is_dir() and path.name == "private_embedding_cache"
    ]
    for payload in published.worker_payloads():
        assert (
            payload["schema_version"]
            == "production_stage1_preflight_worker_payload_v2"
        )
        assert Path(payload["shared_cache_reference_path"]) == shared_references[0]
        assert (
            payload["shared_cache_reference_content_sha256"]
            == published.shared_cache_reference["content_sha256"]
        )
        serialized = json.dumps(payload, sort_keys=True)
        assert str(source) not in serialized
        assert str(cache.cache_dir) not in serialized
        assert all(
            other["scope_id"] not in serialized
            for other in scopes
            if other["scope_id"] != payload["scope_id"]
        )
        private = validate_preflight_scope_input(
            manifest_path=payload["manifest_path"],
            expected_scope_id=payload["scope_id"],
            expected_manifest_content_sha256=payload["manifest_content_sha256"],
        )
        assert not (private.root / "split_registry.json").exists()
        assert private.scope_authority["authorized_scope_count"] == 1
        assert private.scope_authority["other_scope_definitions_supplied"] is False
        assert private.scope_authority["other_scope_row_identities_supplied"] is False
        observed_scope_definitions = []

        def collect_scope_definitions(value):
            if isinstance(value, dict):
                if {
                    "scope_id",
                    "fit_row_ids",
                    "heldout_row_ids",
                }.issubset(value):
                    observed_scope_definitions.append(
                        {
                            "scope_id": value["scope_id"],
                            "fit_row_ids": tuple(value["fit_row_ids"]),
                            "heldout_row_ids": tuple(value["heldout_row_ids"]),
                        }
                    )
                for child in value.values():
                    collect_scope_definitions(child)
            elif isinstance(value, list):
                for child in value:
                    collect_scope_definitions(child)

        for path in private.root.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl"}:
                if path.suffix == ".json":
                    collect_scope_definitions(json.loads(path.read_text(encoding="utf-8")))
                else:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        collect_scope_definitions(json.loads(line))
        expected_definition = {
            "scope_id": private.scope["scope_id"],
            "fit_row_ids": tuple(private.scope["fit_row_ids"]),
            "heldout_row_ids": tuple(private.scope["heldout_row_ids"]),
        }
        assert observed_scope_definitions
        assert all(row == expected_definition for row in observed_scope_definitions)
        fit = set(map(int, private.scope["fit_row_ids"]))
        nonfit = sorted(set(range(len(modeling))) - fit)
        fit_order = list(map(int, private.scope["fit_row_ids"]))
        assert (
            private.modeling_data.iloc[fit_order][
                [
                    config.text_column,
                    config.treatment_column,
                    config.outcome_column,
                ]
            ].to_dict("records")
            == modeling.iloc[fit_order][
                [
                    config.text_column,
                    config.treatment_column,
                    config.outcome_column,
                ]
            ].to_dict("records")
        )
        assert (private.modeling_data.iloc[nonfit][config.text_column] == "").all()
        assert (
            private.modeling_data.iloc[nonfit][[config.treatment_column, config.outcome_column]]
            .isna()
            .all()
            .all()
        )
        bound = private.embedding_cache.bind_spent(
            fit_order,
            modeling.iloc[fit_order][config.text_column].tolist(),
        )
        for row_id in fit_order:
            np.testing.assert_array_equal(
                bound.chunk_matrix(row_id),
                matrix[row_id : row_id + 1],
            )
            assert bound.chunk_texts([row_id]) == ((modeling.iloc[row_id][config.text_column],),)
        with pytest.raises(ValueError, match="non-fit"):
            private.embedding_cache.bind_spent(
                [nonfit[0]],
                [modeling.iloc[nonfit[0]][config.text_column]],
            )


def test_full_production_config_alias_publishes_and_validates_one_scope(
    tmp_path: Path,
):
    profile = (
        Path(__file__).resolve().parents[1]
        / "example_configs"
        / "production_all_evidence_stage1_full.json"
    )
    config = load_applied_stage1_config(profile)
    source = tmp_path / "prepared.parquet"
    pd.DataFrame({"placeholder": [1]}).to_parquet(source, index=False)
    config.dataset_path = str(source)
    texts = [f"production-note-{index}" for index in range(8)]
    modeling = pd.DataFrame(
        {
            config.text_column: texts,
            config.treatment_column: np.asarray([0, 1] * 4, dtype=float),
            config.outcome_column: np.asarray([0, 0, 1, 1] * 2, dtype=float),
        }
    )
    cache = _cache(
        tmp_path / "global_cache",
        texts=texts,
        matrices=np.arange(8 * 4, dtype=np.float32).reshape(8, 4),
        config=config,
    )
    scope = {
        "scope_id": "outer_001_full",
        "scope_kind": "full_outer",
        "outer_fold": 1,
        "inner_fold": None,
        "context_epoch": None,
        "provider_inner_fold": None,
        "fit_row_ids": (0, 1, 2, 3),
        "heldout_row_ids": (4, 5, 6, 7),
    }
    registry = {"dataset_row_count": 8, "outer_folds": []}
    registry_sha = hashlib.sha256(
        json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    # Production validation aliases the integrated subclass into the legacy
    # slot because shared embedding code reads that slot.
    integrated = config.architecture.multi_model_forest
    config.architecture.multi_model_agentic_forest = integrated
    assert integrated.bow_fold_parallelism == "1"
    published = publish_preflight_scope_inputs(
        output_root=(tmp_path / "private").resolve(),
        modeling_data=modeling,
        config=config,
        embedding_cache=cache,
        embedding_cache_identity=cache.identity(),
        registry=registry,
        registry_content_sha256=registry_sha,
        scopes=(scope,),
        source_dataset_path=source,
        global_embedding_cache_path=cache.cache_dir,
        semantic_witness_scientific_config=_SEMANTIC_WITNESS_CONFIG,
    )
    child = published.scopes[scope["scope_id"]]
    payload = json.loads(
        (child.root / "effective_config.json").read_text(encoding="utf-8")
    )
    legacy_payload = payload["architecture"]["multi_model_agentic_forest"]
    integrated_payload = payload["architecture"]["multi_model_forest"]
    assert legacy_payload["fold_parallelism"] == "auto"
    assert "bow_fold_parallelism" not in legacy_payload
    assert "tfidf_topic" not in legacy_payload
    assert integrated_payload["bow_fold_parallelism"] == "1"

    validated = validate_preflight_scope_input(
        manifest_path=child.manifest_path,
        expected_scope_id=scope["scope_id"],
        expected_manifest_content_sha256=child.manifest["content_sha256"],
    )
    assert validated.config.architecture.multi_model_forest.bow_fold_parallelism == "1"
    assert validated.config.architecture.multi_model_forest.htr_fold_parallelism == "1"
    assert (
        validated.config.architecture.multi_model_agentic_forest.fold_parallelism
        == "auto"
    )
    restored = validated.config.architecture.multi_model_forest.embedding_contrast
    expected = config.architecture.multi_model_forest.embedding_contrast
    assert (
        restored.cluster_contrast_n_clusters,
        restored.cluster_contrast_kmeans_n_init,
        restored.cluster_contrast_min_cluster_size,
        restored.cluster_contrast_min_group_size,
        restored.cluster_contrast_min_cell_size,
        restored.cluster_contrast_max_components,
    ) == (
        expected.cluster_contrast_n_clusters,
        expected.cluster_contrast_kmeans_n_init,
        expected.cluster_contrast_min_cluster_size,
        expected.cluster_contrast_min_group_size,
        expected.cluster_contrast_min_cell_size,
        expected.cluster_contrast_max_components,
    )
    fit_rows = list(map(int, validated.scope["fit_row_ids"]))
    generator = _FrozenCacheEmbeddingEvidenceGenerator(
        config=validated.config,
        embedding_provider=validated.embedding_cache.bind_spent(
            fit_rows,
            validated.modeling_data.iloc[fit_rows][
                validated.config.text_column
            ].tolist(),
        ),
        dataset_row_count=len(validated.modeling_data),
        output_dir=tmp_path / "unused_generator_output",
    )
    assert (
        generator.embedding_config.cluster_contrast_n_clusters
        == expected.cluster_contrast_n_clusters
    )


def test_nonfit_modeling_mutation_is_invisible_but_cache_substitution_fails(
    tmp_path: Path,
):
    (
        config,
        source,
        modeling,
        cache,
        matrix,
        scopes,
        registry,
        registry_sha,
    ) = _fixture(tmp_path)
    selected = (scopes[0],)
    cache_identity = cache.identity()
    first = publish_preflight_scope_inputs(
        output_root=(tmp_path / "first").resolve(),
        modeling_data=modeling,
        config=config,
        embedding_cache=cache,
        embedding_cache_identity=cache_identity,
        registry=registry,
        registry_content_sha256=registry_sha,
        scopes=selected,
        source_dataset_path=source,
        global_embedding_cache_path=cache.cache_dir,
        semantic_witness_scientific_config=_SEMANTIC_WITNESS_CONFIG,
    )
    changed = modeling.copy()
    nonfit = list(selected[0]["heldout_row_ids"])
    changed.loc[nonfit, config.text_column] = [f"secret-{row}" for row in nonfit]
    changed.loc[nonfit, config.treatment_column] = 1.0 - changed.loc[
        nonfit, config.treatment_column
    ].to_numpy(dtype=float)
    changed.loc[nonfit, config.outcome_column] = 1.0 - changed.loc[
        nonfit, config.outcome_column
    ].to_numpy(dtype=float)
    second = publish_preflight_scope_inputs(
        output_root=(tmp_path / "second").resolve(),
        modeling_data=changed,
        config=config,
        embedding_cache=cache,
        embedding_cache_identity=cache_identity,
        registry=registry,
        registry_content_sha256=registry_sha,
        scopes=selected,
        source_dataset_path=source,
        global_embedding_cache_path=cache.cache_dir,
        semantic_witness_scientific_config=_SEMANTIC_WITNESS_CONFIG,
    )
    first_scope = first.scopes[selected[0]["scope_id"]]
    second_scope = second.scopes[selected[0]["scope_id"]]
    assert first_scope.manifest["content_sha256"] == second_scope.manifest["content_sha256"]
    assert (
        first_scope.manifest["embedding_cache_view"]
        == second_scope.manifest["embedding_cache_view"]
    )

    changed_matrix = matrix.copy()
    changed_matrix[nonfit] += 10000.0
    changed_texts = modeling[config.text_column].tolist()
    for row in nonfit:
        changed_texts[row] = f"cache-secret-{row}"
    changed_cache = _cache(
        tmp_path / "changed_global_cache",
        texts=changed_texts,
        matrices=changed_matrix,
        config=config,
    )
    with pytest.raises(ValueError, match="authenticated logical identity"):
        publish_preflight_scope_inputs(
            output_root=(tmp_path / "substituted").resolve(),
            modeling_data=changed,
            config=config,
            embedding_cache=changed_cache,
            embedding_cache_identity=cache_identity,
            registry=registry,
            registry_content_sha256=registry_sha,
            scopes=selected,
            source_dataset_path=source,
            global_embedding_cache_path=changed_cache.cache_dir,
            semantic_witness_scientific_config=_SEMANTIC_WITNESS_CONFIG,
        )


def test_tampered_shared_cache_identity_fails_closed(tmp_path: Path):
    (
        config,
        source,
        modeling,
        cache,
        _matrix,
        scopes,
        registry,
        registry_sha,
    ) = _fixture(tmp_path)
    published = publish_preflight_scope_inputs(
        output_root=(tmp_path / "scope_inputs").resolve(),
        modeling_data=modeling,
        config=config,
        embedding_cache=cache,
        embedding_cache_identity=cache.identity(),
        registry=registry,
        registry_content_sha256=registry_sha,
        scopes=(scopes[0],),
        source_dataset_path=source,
        global_embedding_cache_path=cache.cache_dir,
        semantic_witness_scientific_config=_SEMANTIC_WITNESS_CONFIG,
    )
    reference_path = published.shared_cache_reference_path
    tampered = json.loads(reference_path.read_text(encoding="utf-8"))
    tampered["logical_identity"]["row_count"] += 1
    reference_path.write_text(json.dumps(tampered), encoding="utf-8")
    child = published.scopes[scopes[0]["scope_id"]]
    with pytest.raises(ValueError, match="shared preflight cache reference is invalid"):
        validate_preflight_scope_input(
            manifest_path=child.manifest_path,
            expected_scope_id=child.scope_id,
            expected_manifest_content_sha256=child.manifest["content_sha256"],
            shared_cache_reference_path=reference_path,
            expected_shared_cache_reference_content_sha256=(
                published.shared_cache_reference["content_sha256"]
            ),
        )


def test_interrupted_publication_reuses_completed_scope_views_byte_for_byte(
    tmp_path: Path,
    monkeypatch,
):
    (
        config,
        source,
        modeling,
        cache,
        _matrix,
        scopes,
        registry,
        registry_sha,
    ) = _fixture(tmp_path)
    root = (tmp_path / "recoverable").resolve()
    original = scope_inputs_module._write_scope
    calls = 0

    def interrupt_third(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("fixture interruption")
        return original(**kwargs)

    monkeypatch.setattr(
        scope_inputs_module,
        "_write_scope",
        interrupt_third,
    )
    with pytest.raises(KeyboardInterrupt, match="fixture"):
        publish_preflight_scope_inputs(
            output_root=root,
            modeling_data=modeling,
            config=config,
            embedding_cache=cache,
            embedding_cache_identity=cache.identity(),
            registry=registry,
            registry_content_sha256=registry_sha,
            scopes=scopes,
            source_dataset_path=source,
            global_embedding_cache_path=cache.cache_dir,
            semantic_witness_scientific_config=_SEMANTIC_WITNESS_CONFIG,
        )
    first_manifest = root / scopes[0]["scope_id"] / "preflight_scope_input_manifest.json"
    before = (
        first_manifest.read_bytes(),
        first_manifest.stat().st_ino,
        first_manifest.stat().st_mtime_ns,
    )
    assert any(
        (tmp_path / ".recoverable.scope_attempts").glob(f"{scopes[1]['scope_id']}.attempt-*")
    )

    monkeypatch.setattr(scope_inputs_module, "_write_scope", original)
    completed = publish_preflight_scope_inputs(
        output_root=root,
        modeling_data=modeling,
        config=config,
        embedding_cache=cache,
        embedding_cache_identity=cache.identity(),
        registry=registry,
        registry_content_sha256=registry_sha,
        scopes=scopes,
        source_dataset_path=source,
        global_embedding_cache_path=cache.cache_dir,
        semantic_witness_scientific_config=_SEMANTIC_WITNESS_CONFIG,
    )
    after = (
        first_manifest.read_bytes(),
        first_manifest.stat().st_ino,
        first_manifest.stat().st_mtime_ns,
    )
    assert before == after
    assert list(completed.scopes) == [scope["scope_id"] for scope in scopes]
