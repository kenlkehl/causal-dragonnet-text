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
    validate_preflight_scope_input_set,
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
    dataset_locator: Path | None = None,
    model_locator: Path | None = None,
) -> SpentOnlyFrozenChunkEmbeddingCache:
    root.mkdir(parents=True)
    np.save(root / "chunk_embeddings.npy", matrices.astype(np.float32))
    np.save(root / "offsets.npy", np.arange(len(texts) + 1, dtype=np.int64))
    (root / "chunk_texts.jsonl").write_text(
        "".join(json.dumps({"chunks": [text]}, ensure_ascii=False) + "\n" for text in texts),
        encoding="utf-8",
    )
    embedding = config.architecture.multi_model_forest.embedding_contrast
    chunk_counts = [1] * len(texts)
    chunk_configuration = {
        "chunk_size_words": embedding.chunk_size_words,
        "chunk_overlap_words": embedding.chunk_overlap_words,
        "max_chunks": embedding.max_chunks,
        "chunk_selection": embedding.chunk_selection,
        "normalize_embeddings": embedding.normalize_embeddings,
        "max_seq_length": embedding.max_seq_length,
    }
    provenance = {
        "schema_version": "production_embedding_cache_provenance_v3",
        "builder_version": "fixture-builder-v1",
        "dataset": {
            "path": str(
                (
                    dataset_locator
                    if dataset_locator is not None
                    else root.parent / "prepared.parquet"
                ).resolve()
            ),
            "sha256": "a" * 64,
            "size_bytes": 1,
            "text_column": config.text_column,
            "row_count": len(texts),
            "ordered_text_sha256": "b" * 64,
        },
        "sentence_model_name": embedding.model_name,
        "local_model": {
            "path": str(
                (
                    model_locator
                    if model_locator is not None
                    else root.parent / "global_embedding_model"
                ).resolve()
            ),
            "tree_sha256": "c" * 64,
            "file_count": 1,
            "directory_count": 1,
            "total_file_bytes": 1,
        },
        "chunk_configuration": chunk_configuration,
        "encoder_execution": {
            "device": "cuda:fixture",
            "batch_size": 17,
            "local_files_only": True,
        },
    }
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "num_samples": len(texts),
                "hidden_size": matrices.shape[1],
                "sentence_model_name": embedding.model_name,
                "total_chunks": len(texts),
                "chunk_counts": chunk_counts,
                **chunk_configuration,
                "uncapped_total_chunks": len(texts),
                "uncapped_chunk_counts_sha256": hashlib.sha256(
                    json.dumps(
                        chunk_counts,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "chunk_cap_nonbinding": True,
                "semantic_truncation_allowed": False,
                "tokenizer_truncation_allowed": False,
                "production_provenance": provenance,
                "production_provenance_sha256": hashlib.sha256(
                    json.dumps(
                        provenance,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
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
    model_locator = tmp_path / "global_embedding_model"
    model_locator.mkdir()
    (model_locator / "weights.bin").write_bytes(b"private model")
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
        dataset_locator=source,
        model_locator=model_locator,
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


def _shared_modeling_reference_path(scope_input) -> Path:
    return scope_input.root.parent / "shared_text_reference.json"


def _shared_modeling_path(scope_input) -> Path:
    reference_path = _shared_modeling_reference_path(scope_input)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    return reference_path.parent / reference["text_block"]["relative_path"]


def _reseal_shared_modeling_after_change(scope_input) -> None:
    reference_path = _shared_modeling_reference_path(scope_input)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    modeling_path = _shared_modeling_path(scope_input)
    payload = modeling_path.read_bytes()
    reference["text_block"] = {
        "relative_path": modeling_path.relative_to(reference_path.parent).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    reference_body = {
        key: value
        for key, value in reference.items()
        if key != "content_sha256"
    }
    reference["content_sha256"] = hashlib.sha256(
        json.dumps(
            reference_body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    reference_path.write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _reseal_scope_file(scope_input, key: str, path: Path) -> None:
    manifest = json.loads(scope_input.manifest_path.read_text(encoding="utf-8"))
    payload = path.read_bytes()
    manifest["files"][key] = {
        "relative_path": path.relative_to(scope_input.root).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    body = {
        field: value
        for field, value in manifest.items()
        if field != "content_sha256"
    }
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    scope_input.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _contains_instance(
    value,
    classes: tuple[type, ...],
    *,
    _seen: set[int] | None = None,
) -> bool:
    if isinstance(value, classes):
        return True
    if isinstance(
        value,
        (
            str,
            bytes,
            int,
            float,
            bool,
            type(None),
            Path,
            np.ndarray,
            pd.DataFrame,
            pd.Series,
        ),
    ):
        return False
    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, dict):
        return any(
            _contains_instance(key, classes, _seen=seen)
            or _contains_instance(child, classes, _seen=seen)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(
            _contains_instance(child, classes, _seen=seen)
            for child in value
        )
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return _contains_instance(attributes, classes, _seen=seen)
    return False


def _contains_text(
    value,
    target: str,
    *,
    _seen: set[int] | None = None,
) -> bool:
    if isinstance(value, (str, Path)):
        return target in str(value)
    if isinstance(
        value,
        (
            bytes,
            int,
            float,
            bool,
            type(None),
            np.ndarray,
            pd.DataFrame,
            pd.Series,
        ),
    ):
        return False
    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, dict):
        return any(
            _contains_text(key, target, _seen=seen)
            or _contains_text(child, target, _seen=seen)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(
            _contains_text(child, target, _seen=seen)
            for child in value
        )
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return _contains_text(attributes, target, _seen=seen)
    return False


def _contains_float(value, target: float) -> bool:
    if isinstance(value, (float, np.floating)):
        return float(value) == target
    if isinstance(value, dict):
        return any(
            _contains_float(key, target)
            or _contains_float(child, target)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_float(child, target) for child in value)
    return False


def test_scope_inputs_expose_only_fit_rows_and_refuse_nonfit_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    assert operational_identity["per_scope_full_cohort_modeling_copied"] is False
    assert operational_identity["worker_global_cache_locator_supplied"] is False
    assert operational_identity["worker_embedding_source"] == (
        "authenticated_fit_only_content_addressed_row_blocks_v1"
    )
    shared_modeling_path = output_root / "shared_text.parquet"
    label_projection_paths = {
        scope_input.root / "fit_label_projection.parquet"
        for scope_input in published.scopes.values()
    }
    assert set(output_root.rglob("*.parquet")) == {
        shared_modeling_path,
        *label_projection_paths,
    }
    shared_modeling = pd.read_parquet(shared_modeling_path)
    assert list(shared_modeling.columns) == [
        "__production_global_row_id",
        config.text_column,
    ]
    assert shared_modeling["__production_global_row_id"].tolist() == list(
        range(len(modeling))
    )
    assert (
        shared_modeling[config.text_column].tolist()
        == modeling[config.text_column].tolist()
    )
    assert config.treatment_column not in shared_modeling
    assert config.outcome_column not in shared_modeling
    assert operational_identity["shared_text_rows"] == len(modeling)
    assert (
        operational_identity["shared_text_bytes"]
        == shared_modeling_path.stat().st_size
    )
    shared_references = list(output_root.rglob("shared_embedding_cache_reference.json"))
    assert shared_references == [
        output_root / "shared_embedding_cache_reference.json"
    ]
    row_block_root = output_root / "shared_embedding_rows"
    expected_embedding_rows = sorted(
        {
            int(row_id)
            for scope in scopes
            for row_id in scope["fit_row_ids"]
        }
    )
    row_embedding_blocks = sorted(row_block_root.glob("*.npy"))
    row_chunk_blocks = sorted(row_block_root.glob("*.chunks.json"))
    assert len(row_embedding_blocks) == len(expected_embedding_rows)
    assert len(row_chunk_blocks) == len(expected_embedding_rows)
    assert operational_identity["shared_embedding_row_block_count"] == len(
        expected_embedding_rows
    )
    assert operational_identity["shared_embedding_row_store_bytes"] == sum(
        path.stat().st_size
        for path in (*row_embedding_blocks, *row_chunk_blocks)
    )
    assert not [
        path
        for scope_input in published.scopes.values()
        for path in scope_input.root.rglob("*")
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
    for scope_input in published.scopes.values():
        assert not hasattr(scope_input, "modeling_data")
        assert not hasattr(scope_input, "embedding_cache")
    full_cache_loader_calls = []

    def refuse_full_cache_loader(_reference):
        full_cache_loader_calls.append(True)
        raise AssertionError("worker attempted to open the global cache")

    monkeypatch.setattr(
        scope_inputs_module,
        "_load_shared_cache",
        refuse_full_cache_loader,
    )
    cache_handle_state_before = set(
        scope_inputs_module._SHARED_CACHE_HANDLES
    )
    for payload in published.worker_payloads():
        assert (
            payload["schema_version"]
            == "production_stage1_preflight_worker_payload_v4"
        )
        assert set(payload) == {
            "schema_version",
            "scope_id",
            "manifest_path",
            "manifest_content_sha256",
        }
        serialized = json.dumps(payload, sort_keys=True)
        assert str(source) not in serialized
        assert str(cache.cache_dir) not in serialized
        assert "shared_text" not in serialized
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
        private_values = vars(private)
        assert "shared_cache_reference_path" not in private_values
        assert "shared_cache_reference" not in private_values
        assert not _contains_instance(
            private_values,
            (SpentOnlyFrozenChunkEmbeddingCache,),
        )
        for forbidden_locator in (
            source.resolve(),
            cache.cache_dir,
            (tmp_path / "global_embedding_model").resolve(),
        ):
            assert not _contains_text(
                private_values,
                str(forbidden_locator),
            )
        assert not (private.root / "split_registry.json").exists()
        assert private.scope_authority["authorized_scope_count"] == 1
        assert private.scope_authority["other_scope_definitions_supplied"] is False
        assert private.scope_authority["other_scope_row_identities_supplied"] is False
        fit_order = list(map(int, private.scope["fit_row_ids"]))
        assert private.manifest["shared_modeling_view"] == {
            "schema_version": "production_stage1_preflight_text_and_fit_labels_view_v1",
            "fit_modeling_content_sha256": (
                scope_inputs_module._fit_modeling_content_sha256(
                    modeling_data=modeling,
                    fit_rows=fit_order,
                    columns=[
                        config.text_column,
                        config.treatment_column,
                        config.outcome_column,
                    ],
                )
            ),
            "dataset_row_count": len(modeling),
            "allowed_row_ids": fit_order,
            "allowed_row_order_sha256": hashlib.sha256(
                json.dumps(
                    fit_order,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "allowed_row_count": len(fit_order),
            "peer_row_access_allowed": False,
            "per_scope_text_payload_count": 0,
            "fit_label_projection_count": 1,
            "fit_label_projection_schema": (
                "production_stage1_preflight_fit_label_projection_v1"
            ),
            "nonfit_labels_stored": False,
            "nonfit_rows_returned_by_worker_api": False,
        }
        label_projection_path = private.root / "fit_label_projection.parquet"
        assert list(private.root.rglob("*.parquet")) == [
            label_projection_path
        ]
        label_projection = pd.read_parquet(label_projection_path)
        assert list(label_projection.columns) == [
            "__production_global_row_id",
            config.treatment_column,
            config.outcome_column,
        ]
        assert label_projection["__production_global_row_id"].tolist() == fit_order
        assert (
            label_projection[
                [config.treatment_column, config.outcome_column]
            ].to_dict("records")
            == modeling.iloc[fit_order][
                [config.treatment_column, config.outcome_column]
            ].to_dict("records")
        )
        assert not hasattr(private, "shared_modeling_reference_path")
        assert operational_identity["per_scope_fit_index_rows"][
            private.scope_id
        ] == len(fit_order)
        assert operational_identity["per_scope_label_projection_bytes"][
            private.scope_id
        ] == label_projection_path.stat().st_size
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
        assert private.embedding_cache._shared_cache_retained is False
        scoped_metadata = private.embedding_cache.metadata
        assert set(scoped_metadata) == {
            "schema_version",
            "source_metadata_sha256",
            "num_samples",
            "hidden_size",
            "total_chunks",
            "chunk_counts",
            "chunk_size_words",
            "chunk_overlap_words",
            "max_chunks",
            "chunk_selection",
            "normalize_embeddings",
            "max_seq_length",
            "uncapped_total_chunks",
            "uncapped_chunk_counts_sha256",
            "chunk_cap_nonbinding",
            "semantic_truncation_allowed",
            "tokenizer_truncation_allowed",
            "allowed_row_count",
            "nonfit_row_count",
            "nonfit_chunk_count",
            "nonfit_chunk_counts_zeroed",
            "production_provenance_included",
            "operational_execution_metadata_included",
        }
        assert scoped_metadata["schema_version"] == (
            "production_stage1_preflight_scoped_embedding_cache_metadata_v1"
        )
        assert scoped_metadata["production_provenance_included"] is False
        assert (
            scoped_metadata["operational_execution_metadata_included"]
            is False
        )
        assert scoped_metadata["nonfit_chunk_count"] == 0
        assert scoped_metadata["nonfit_chunk_counts_zeroed"] is True
        assert all(
            scoped_metadata["chunk_counts"][row_id] == 0
            for row_id in nonfit
        )
        assert not _contains_instance(
            vars(private.embedding_cache),
            (SpentOnlyFrozenChunkEmbeddingCache,),
        )
        assert set(private.embedding_cache._cached_by_row) == fit
        assert not np.intersect1d(
            private.embedding_cache._embeddings.reshape(-1),
            matrix[nonfit].reshape(-1),
        ).size
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
    assert full_cache_loader_calls == []
    assert (
        scope_inputs_module._SHARED_CACHE_HANDLES
        == cache_handle_state_before
    )
    assert not _contains_instance(
        scope_inputs_module._SHARED_MODELING_HANDLES,
        (pd.DataFrame, np.ndarray, SpentOnlyFrozenChunkEmbeddingCache),
    )
    assert not _contains_instance(
        scope_inputs_module._SHARED_CACHE_HANDLES,
        (pd.DataFrame, np.ndarray, SpentOnlyFrozenChunkEmbeddingCache),
    )
    assert not _contains_text(
        scope_inputs_module._SHARED_CACHE_HANDLES,
        str(cache.cache_dir),
    )


@pytest.mark.parametrize("label_dtype", [np.int64, np.bool_])
def test_fit_label_scalar_types_survive_capability_publication_and_reopen(
    tmp_path: Path,
    label_dtype,
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
    modeling[config.treatment_column] = modeling[
        config.treatment_column
    ].astype(label_dtype)
    modeling[config.outcome_column] = modeling[
        config.outcome_column
    ].astype(label_dtype)

    published = publish_preflight_scope_inputs(
        output_root=(tmp_path / "typed_scope_inputs").resolve(),
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

    for payload in published.worker_payloads():
        reopened = validate_preflight_scope_input(
            manifest_path=payload["manifest_path"],
            expected_scope_id=payload["scope_id"],
            expected_manifest_content_sha256=payload[
                "manifest_content_sha256"
            ],
        )
        fit_rows = list(map(int, reopened.scope["fit_row_ids"]))
        columns = [
            config.text_column,
            config.treatment_column,
            config.outcome_column,
        ]
        assert (
            reopened.modeling_data.iloc[fit_rows][columns].to_dict("records")
            == modeling.iloc[fit_rows][columns].to_dict("records")
        )
        assert reopened.manifest["shared_modeling_view"][
            "fit_modeling_content_sha256"
        ] == scope_inputs_module._fit_modeling_content_sha256(
            modeling_data=modeling,
            fit_rows=fit_rows,
            columns=columns,
        )


def test_shared_text_is_written_once_and_scopes_store_only_fit_labels(
    tmp_path: Path,
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

    def publish(name: str, selected_scopes):
        return publish_preflight_scope_inputs(
            output_root=(tmp_path / name).resolve(),
            modeling_data=modeling,
            config=config,
            embedding_cache=cache,
            embedding_cache_identity=cache.identity(),
            registry=registry,
            registry_content_sha256=registry_sha,
            scopes=selected_scopes,
            source_dataset_path=source,
            global_embedding_cache_path=cache.cache_dir,
            semantic_witness_scientific_config=_SEMANTIC_WITNESS_CONFIG,
        )

    one_scope = publish("one_scope", (scopes[0],))
    two_scopes = publish("two_scopes", scopes)
    one_path = one_scope.root / "shared_text.parquet"
    two_path = two_scopes.root / "shared_text.parquet"
    assert set(one_scope.root.rglob("*.parquet")) == {
        one_path,
        *(
            scope_input.root / "fit_label_projection.parquet"
            for scope_input in one_scope.scopes.values()
        ),
    }
    assert set(two_scopes.root.rglob("*.parquet")) == {
        two_path,
        *(
            scope_input.root / "fit_label_projection.parquet"
            for scope_input in two_scopes.scopes.values()
        ),
    }
    assert one_path.read_bytes() == two_path.read_bytes()
    assert (
        one_scope.shared_modeling_reference["content_sha256"]
        == two_scopes.shared_modeling_reference["content_sha256"]
    )
    assert one_scope.identity()["shared_text_bytes"] == one_path.stat().st_size
    assert two_scopes.identity()["shared_text_bytes"] == two_path.stat().st_size
    assert all(
        list(scope_input.root.rglob("*.parquet"))
        == [scope_input.root / "fit_label_projection.parquet"]
        for scope_input in two_scopes.scopes.values()
    )
    assert two_scopes.identity()["per_scope_label_projection_bytes"] == {
        scope_input.scope_id: (
            scope_input.root / "fit_label_projection.parquet"
        ).stat().st_size
        for scope_input in two_scopes.scopes.values()
    }


def test_shared_modeling_tamper_and_duplicate_rows_fail_closed(
    tmp_path: Path,
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

    def publish(name: str):
        result = publish_preflight_scope_inputs(
            output_root=(tmp_path / name).resolve(),
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
        return result.scopes[scopes[0]["scope_id"]]

    tampered = publish("tampered")
    table = pd.read_parquet(_shared_modeling_path(tampered))
    table.loc[0, config.text_column] = "tampered-fit-text"
    table.to_parquet(_shared_modeling_path(tampered), index=False)
    with pytest.raises(ValueError, match="text block .*changed"):
        validate_preflight_scope_input(
            manifest_path=tampered.manifest_path,
            expected_scope_id=tampered.scope_id,
        )
    _reseal_shared_modeling_after_change(tampered)
    with pytest.raises(ValueError, match="text or fit-label content changed"):
        validate_preflight_scope_input(
            manifest_path=tampered.manifest_path,
            expected_scope_id=tampered.scope_id,
            parent_modeling_data=modeling,
            parent_config=config,
        )

    label_tampered = publish("label_tampered")
    label_path = label_tampered.root / "fit_label_projection.parquet"
    labels = pd.read_parquet(label_path)
    labels.loc[0, config.treatment_column] = (
        1.0 - float(labels.loc[0, config.treatment_column])
    )
    labels.to_parquet(label_path, index=False)
    with pytest.raises(ValueError, match="fit_label_projection file changed"):
        validate_preflight_scope_input(
            manifest_path=label_tampered.manifest_path,
            expected_scope_id=label_tampered.scope_id,
        )
    _reseal_scope_file(
        label_tampered,
        "fit_label_projection",
        label_path,
    )
    with pytest.raises(ValueError, match="text or fit-label content changed"):
        validate_preflight_scope_input(
            manifest_path=label_tampered.manifest_path,
            expected_scope_id=label_tampered.scope_id,
        )

    label_duplicated = publish("label_duplicated")
    label_duplicate_path = (
        label_duplicated.root / "fit_label_projection.parquet"
    )
    duplicate_labels = pd.read_parquet(label_duplicate_path)
    duplicate_labels.loc[1, "__production_global_row_id"] = (
        duplicate_labels.loc[0, "__production_global_row_id"]
    )
    duplicate_labels.to_parquet(label_duplicate_path, index=False)
    _reseal_scope_file(
        label_duplicated,
        "fit_label_projection",
        label_duplicate_path,
    )
    with pytest.raises(
        ValueError,
        match="fit-label projection has duplicate global row IDs",
    ):
        validate_preflight_scope_input(
            manifest_path=label_duplicated.manifest_path,
            expected_scope_id=label_duplicated.scope_id,
        )

    duplicated = publish("duplicated")
    table = pd.read_parquet(_shared_modeling_path(duplicated))
    table.loc[1, "__production_global_row_id"] = table.loc[
        0, "__production_global_row_id"
    ]
    table.to_parquet(_shared_modeling_path(duplicated), index=False)
    _reseal_shared_modeling_after_change(duplicated)
    with pytest.raises(ValueError, match="duplicate global row IDs"):
        validate_preflight_scope_input(
            manifest_path=duplicated.manifest_path,
            expected_scope_id=duplicated.scope_id,
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
    expected_bow_parallelism = integrated.bow_fold_parallelism
    expected_htr_parallelism = integrated.htr_fold_parallelism
    assert isinstance(expected_bow_parallelism, str)
    assert isinstance(expected_htr_parallelism, str)
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
    assert (
        integrated_payload["bow_fold_parallelism"]
        == expected_bow_parallelism
    )

    validated = validate_preflight_scope_input(
        manifest_path=child.manifest_path,
        expected_scope_id=scope["scope_id"],
        expected_manifest_content_sha256=child.manifest["content_sha256"],
    )
    assert (
        validated.config.architecture.multi_model_forest.bow_fold_parallelism
        == expected_bow_parallelism
    )
    assert (
        validated.config.architecture.multi_model_forest.htr_fold_parallelism
        == expected_htr_parallelism
    )
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
    nonfit = list(selected[0]["heldout_row_ids"])
    labels_only = modeling.copy()
    heldout_treatment_sentinel = 123_456.125
    heldout_outcome_sentinel = 223_456.25
    labels_only.loc[nonfit, config.treatment_column] = np.asarray(
        [heldout_treatment_sentinel] * len(nonfit)
    )
    labels_only.loc[nonfit, config.outcome_column] = np.asarray(
        [heldout_outcome_sentinel] * len(nonfit)
    )
    labels_only_result = publish_preflight_scope_inputs(
        output_root=(tmp_path / "labels_only").resolve(),
        modeling_data=labels_only,
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
    labels_only_scope = labels_only_result.scopes[selected[0]["scope_id"]]
    assert (
        first_scope.manifest["content_sha256"]
        == labels_only_scope.manifest["content_sha256"]
    )
    assert (
        first.shared_modeling_reference["content_sha256"]
        == labels_only_result.shared_modeling_reference["content_sha256"]
    )
    labels_only_projection = pd.read_parquet(
        labels_only_scope.root / "fit_label_projection.parquet"
    )
    assert labels_only_projection["__production_global_row_id"].tolist() == list(
        selected[0]["fit_row_ids"]
    )
    assert labels_only_projection[config.treatment_column].max() <= 1.0
    assert labels_only_projection[config.outcome_column].max() <= 1.0
    for cache_state in (
        scope_inputs_module._SHARED_MODELING_HANDLES,
        scope_inputs_module._SHARED_CACHE_HANDLES,
    ):
        assert not _contains_float(
            cache_state,
            heldout_treatment_sentinel,
        )
        assert not _contains_float(
            cache_state,
            heldout_outcome_sentinel,
        )

    changed = modeling.copy()
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
    assert (
        first_scope.manifest["shared_modeling_view"]
        == second_scope.manifest["shared_modeling_view"]
    )
    assert (
        first.shared_modeling_reference["content_sha256"]
        != second.shared_modeling_reference["content_sha256"]
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
    child = published.scopes[scopes[0]["scope_id"]]
    with pytest.raises(
        ValueError,
        match="must not receive a global cache reference path",
    ):
        validate_preflight_scope_input(
            manifest_path=child.manifest_path,
            expected_scope_id=child.scope_id,
            expected_manifest_content_sha256=child.manifest[
                "content_sha256"
            ],
            shared_cache_reference_path=reference_path,
        )
    tampered = json.loads(reference_path.read_text(encoding="utf-8"))
    tampered["logical_identity"]["row_count"] += 1
    reference_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="cache reference.*changed",
    ):
        validate_preflight_scope_input_set(
            root=published.root,
            expected_scopes=(scopes[0],),
            expected_registry_content_sha256=registry_sha,
            parent_modeling_data=modeling,
            parent_config=config,
            parent_embedding_cache=cache,
            parent_embedding_cache_identity=cache.identity(),
            expected_shared_cache_reference=(
                published.shared_cache_reference
            ),
            expected_shared_modeling_reference=(
                published.shared_modeling_reference
            ),
            expected_semantic_witness_scientific_config=(
                _SEMANTIC_WITNESS_CONFIG
            ),
            forbidden_paths=(source, cache.cache_dir),
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
    shared_modeling_path = root / "shared_text.parquet"
    shared_before = (
        shared_modeling_path.read_bytes(),
        shared_modeling_path.stat().st_ino,
        shared_modeling_path.stat().st_mtime_ns,
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
    assert shared_before == (
        shared_modeling_path.read_bytes(),
        shared_modeling_path.stat().st_ino,
        shared_modeling_path.stat().st_mtime_ns,
    )
    assert list(completed.scopes) == [scope["scope_id"] for scope in scopes]
