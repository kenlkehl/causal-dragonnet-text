from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.config import AppliedInferenceConfig
from oci.inference.all_evidence_discovery_interfaces import (
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    TFIDF_SEMANTIC_RETRIEVAL,
)
from oci.inference.all_evidence_fusion import (
    LEGACY_ALL_SOURCE,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
)
from oci.inference.embedding_native_proof_capture import (
    EMBEDDING_CLUSTER_SUPPORT_CONTRACT_SCHEMA,
    SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA,
    NativeEmbeddingProofCaptureSink,
    build_semantic_retrieval_training_only_policy,
    semantic_retrieval_projection_bundle,
    validate_embedding_cluster_support_state,
    validate_embedding_native_capture,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    build_role_neutral_evidence_catalog,
)
from oci.inference.production_stage1_bundle import (
    PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
    STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
    _catalog_ready_legacy_digest,
    _canonical_embedding_scope_lineage,
    _component_file_registration,
    _native_scope_text_projections,
    _register_embedding_native_family_proofs,
    _sha256_json,
    _validate_embedding_native_family_proof_index,
    _write_immutable_json,
)
from oci.inference.review_spent_evidence_provider import (
    SpentOnlyFrozenChunkEmbeddingCache,
    _FrozenCacheEmbeddingEvidenceGenerator,
    _embedding_concepts_only,
)


def _write_cache(path: Path, texts: tuple[str, ...], embeddings: np.ndarray) -> None:
    path.mkdir(parents=True)
    if embeddings.shape[0] != len(texts):
        raise AssertionError("test cache needs one embedding per text")
    np.save(path / "chunk_embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    np.save(path / "offsets.npy", np.arange(len(texts) + 1, dtype=np.int64))
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "num_samples": len(texts),
                "hidden_size": int(embeddings.shape[1]),
                "chunk_size_words": 64,
                "chunk_overlap_words": 0,
                "max_chunks": 1,
                "chunk_selection": "last",
            }
        ),
        encoding="utf-8",
    )
    with (path / "chunk_texts.jsonl").open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"chunks": [text]}) + "\n")


def test_mixed_unicode_text_keeps_raw_embedding_and_normalized_legacy_projections(
    tmp_path: Path,
):
    texts = (
        "Baseline Cafe\u0301 – DOSE ≥ 2",
        "Follow‑Up—STABLE ≤ 3",
        "Heldout MIXED—Case",
        "Second heldout – unchanged for embedding",
    )
    fit_rows = (1, 0)
    heldout_rows = (3, 2)
    raw_fit, normalized_fit = _native_scope_text_projections(texts[row_id] for row_id in fit_rows)
    assert raw_fit == (texts[1], texts[0])
    assert normalized_fit == (
        "follow-up-stable <= 3",
        "baseline café - dose >= 2",
    )

    cache_dir = tmp_path / "mixed_unicode_cache"
    _write_cache(cache_dir, texts, np.eye(len(texts), dtype=np.float32))
    cache = SpentOnlyFrozenChunkEmbeddingCache(cache_dir)
    provider = cache.bind_spent(fit_rows, raw_fit)
    assert provider.row_ids == fit_rows
    assert provider.chunk_texts(fit_rows) == ((texts[1],), (texts[0],))
    with pytest.raises(ValueError, match="spent text does not match"):
        cache.bind_spent(fit_rows, normalized_fit)

    modeling_data = pd.DataFrame(
        {
            "clinical_text": texts,
            "treatment_indicator": [0.0, 1.0, 0.0, 1.0],
            "outcome_indicator": [0.0, 1.0, 1.0, 0.0],
        }
    )
    canonical = _canonical_embedding_scope_lineage(
        modeling_data=modeling_data,
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        embedding_config={"residualize_columns": []},
    )
    assert canonical["fit_texts"] == raw_fit

    indexed_data = modeling_data.copy()
    indexed_data["_oci_row_id"] = np.arange(len(indexed_data), dtype=int)
    indexed_data.loc[fit_rows[0], "_oci_row_id"] = 99
    with pytest.raises(ValueError, match="differ from modeling-data positions"):
        _canonical_embedding_scope_lineage(
            modeling_data=indexed_data,
            outer_fold=1,
            inner_fold=1,
            fit_row_ids=fit_rows,
            heldout_row_ids=heldout_rows,
            text_column="clinical_text",
            treatment_column="treatment_indicator",
            outcome_column="outcome_indicator",
            embedding_config={"residualize_columns": []},
        )


def _case(
    tmp_path: Path,
    *,
    generator_pseudo_offset: float = 0.0,
    generator_t_resid_offset: float = 0.0,
):
    row_count = 48
    heldout_rows = (3, 9, 15, 21, 27, 33, 39, 45)
    fit_rows = tuple(index for index in range(row_count) if index not in heldout_rows)
    full_treatment = np.asarray([(index // 4) % 2 for index in range(row_count)], dtype=float)
    full_outcome = np.asarray([(index // 8) % 2 for index in range(row_count)], dtype=float)
    treatment = full_treatment[list(fit_rows)]
    outcome = full_outcome[list(fit_rows)]
    pseudo = (2.0 * treatment - 1.0) * (2.0 * outcome - 1.0) + np.linspace(
        -0.2,
        0.2,
        len(fit_rows),
    )
    t_resid = treatment - 0.43
    texts = []
    embeddings = []
    for index in range(row_count):
        cluster = index % 4
        block = index // 4
        treated = block % 2
        positive = (block // 2) % 2
        texts.append(
            " ".join(
                (
                    f"clusterword{cluster}",
                    "active therapy response" if treated else "supportive care baseline",
                    "improved durable outcome" if positive else "frail declining symptoms",
                    f"patienttoken_{index}_end",
                )
            )
        )
        vector = np.zeros(8, dtype=np.float32)
        vector[cluster] = 2.5
        vector[4] = 1.2 if treated else -1.2
        vector[5] = 1.0 if positive else -1.0
        vector[6] = 0.8 if treated == positive else -0.8
        vector[7] = (index % 7) / 20.0
        embeddings.append(vector)
    texts = tuple(texts)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, texts, embeddings)
    cache = SpentOnlyFrozenChunkEmbeddingCache(cache_dir)
    provider = cache.bind_spent(fit_rows, tuple(texts[index] for index in fit_rows))

    config = AppliedInferenceConfig()
    config.outcome_type = "binary"
    embedding_config = config.architecture.multi_model_forest.embedding_contrast
    embedding_config.enabled = True
    embedding_config.disable_reason = None
    embedding_config.chunk_size_words = 64
    embedding_config.chunk_overlap_words = 0
    embedding_config.max_chunks = 1
    embedding_config.chunk_selection = "last"
    embedding_config.top_k_chunks_per_tail = 20
    embedding_config.max_chunks_per_patient = 1
    embedding_config.include_bow_phrases_as_concepts = False
    embedding_config.concept_phrases = []
    embedding_config.external_corpus_cache_dirs = []
    embedding_config.cluster_contrast_n_clusters = 4
    embedding_config.cluster_contrast_min_cluster_size = 4
    embedding_config.cluster_contrast_min_group_size = 2
    embedding_config.cluster_contrast_min_cell_size = 1
    embedding_config.cluster_contrast_max_components = 3
    embedding_config.cluster_contrast_top_loadings = 4
    embedding_config.cluster_contrast_kmeans_n_init = 3
    dataset = pd.DataFrame(
        {
            "_oci_row_id": np.arange(row_count, dtype=int),
            config.text_column: texts,
            config.treatment_column: full_treatment,
            config.outcome_column: full_outcome,
        }
    )
    generator = _FrozenCacheEmbeddingEvidenceGenerator(
        config=config,
        embedding_provider=provider,
        dataset_row_count=row_count,
        output_dir=tmp_path / "generator",
    )
    generator.prepare(dataset)
    sink = NativeEmbeddingProofCaptureSink(
        artifact_dir=tmp_path / "capture",
        scope_id="outer_01_inner_01",
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        fit_texts=tuple(texts[index] for index in fit_rows),
        expected_fit_treatment=treatment,
        expected_fit_outcome=outcome,
        text_column=config.text_column,
        outcome_type=config.outcome_type,
        embedding_provider=provider,
        embedding_config=generator.embedding_config,
        tfidf_nested_calibration_folds=5,
        seed=917,
    )
    generator._native_embedding_proof_observer = sink
    sink.record_registered_fit_outputs(
        fit_row_ids=fit_rows,
        treatment=treatment,
        outcome=outcome,
        pseudo_target=[pseudo],
        t_resid=[t_resid],
        pseudo_target_names=["ensemble_nuisance"],
    )
    evidence = generator.build_evidence(
        discovery_df=dataset.iloc[list(fit_rows)].copy(),
        y=outcome,
        t=treatment,
        pseudo_target=[pseudo + float(generator_pseudo_offset)],
        t_resid=[t_resid + float(generator_t_resid_offset)],
        pseudo_target_names=["ensemble_nuisance"],
        importance={"ignored_by_frozen_generator": True},
    )
    metadata = sink.finalize()
    return {
        "artifact_dir": tmp_path / "capture",
        "provider": provider,
        "fit_rows": fit_rows,
        "heldout_rows": heldout_rows,
        "fit_texts": tuple(texts[index] for index in fit_rows),
        "heldout_texts": tuple(texts[index] for index in heldout_rows),
        "evidence": evidence,
        "metadata": metadata,
        "cache": cache,
        "dataset": dataset,
        "text_column": config.text_column,
        "outcome_type": config.outcome_type,
        "treatment_column": config.treatment_column,
        "outcome_column": config.outcome_column,
    }


def test_embedding_capture_replays_generator_kmeans_svd_and_semantic_projection(tmp_path: Path):
    case = _case(tmp_path)
    replay = validate_embedding_native_capture(
        case["artifact_dir"],
        embedding_provider=case["provider"],
        fit_texts=case["fit_texts"],
        expected_scope_id="outer_01_inner_01",
        expected_fit_row_ids=case["fit_rows"],
        expected_heldout_row_ids=case["heldout_rows"],
    )

    assert replay == case["metadata"]
    assert replay["cluster_kmeans"] is not None
    assert replay["cluster_svds"]
    assert {row["family_key"] for row in replay["cluster_svds"]} == {
        "treatment",
        "residualized_interaction",
    }
    support = replay["cluster_support_contract"]
    assert "kmeans_inertia" not in support
    assert replay["cluster_kmeans"]["inertia"] >= 0.0
    assert support["kmeans_parameters"] == replay["cluster_kmeans"]["parameters"]
    assert support["kmeans_parameters"] == {
        "n_clusters": 4,
        "random_state": 42,
        "batch_size": 128,
        "n_init": 3,
        "max_iter": 300,
    }
    assert support["schema_version"] == EMBEDDING_CLUSTER_SUPPORT_CONTRACT_SCHEMA
    assert {row["family_key"] for row in support["svd_families"]} == {
        "treatment",
        "residualized_interaction",
    }
    assert all(row["local_contrast_count"] >= 2 for row in support["svd_families"])
    assert all(row["numerical_rank"] >= 2 for row in support["svd_families"])
    assert all(row["second_singular_value"] > 0.0 for row in support["svd_families"])
    assert all(
        row["second_singular_value"] > row["numerical_rank_tolerance_float32"]
        for row in support["svd_families"]
    )
    assert replay["tfidf_training_scope_policy"]["schema_version"] == (
        SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA
    )
    assert replay["heldout_text_accessed"] is False
    raw = json.loads((case["artifact_dir"] / "raw_embedding_evidence.json").read_text())
    assert raw == case["evidence"]


def test_cluster_support_contract_rejects_missing_single_cluster_and_rank_one_state():
    kmeans = {
        "fit_row_ids": [0, 1, 2, 3],
        "parameters": {
            "n_clusters": 2,
            "random_state": 42,
            "batch_size": 128,
            "n_init": 3,
            "max_iter": 300,
        },
        "usable_mask": np.asarray([True, True, True, True]),
        "cluster_labels": np.asarray([0, 0, 1, 1]),
        "cluster_centers": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        "cluster_counts": np.asarray([2, 2]),
        "n_iter": 2,
        "inertia": 0.5,
    }

    def state(family: str, matrix: np.ndarray) -> dict[str, object]:
        _left, values, components = np.linalg.svd(matrix, full_matrices=False)
        return {
            "family_key": family,
            "item_cluster_ids": [0, 1],
            "weighted_matrix": matrix,
            "singular_values": values,
            "components": components,
        }

    treatment = state("treatment", np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    interaction = state(
        "residualized_interaction",
        np.asarray([[1.0, 1.0], [1.0, -1.0]]),
    )
    expected_kmeans_configuration = {
        "cluster_contrast_n_clusters": 2,
        "cluster_contrast_random_state": 42,
        "cluster_contrast_kmeans_n_init": 3,
    }
    valid = validate_embedding_cluster_support_state(
        kmeans_state=kmeans,
        svd_states=[treatment, interaction],
        expected_cluster_count=2,
        expected_kmeans_configuration=expected_kmeans_configuration,
    )
    assert valid["required_svd_families"] == ["treatment", "residualized_interaction"]
    assert "kmeans_inertia" not in valid
    assert all(
        row["second_singular_value"] > row["numerical_rank_tolerance_float32"]
        for row in valid["svd_families"]
    )
    changed_inertia = dict(kmeans)
    changed_inertia["inertia"] = 999.0
    assert (
        validate_embedding_cluster_support_state(
            kmeans_state=changed_inertia,
            svd_states=[treatment, interaction],
            expected_cluster_count=2,
            expected_kmeans_configuration=expected_kmeans_configuration,
        )
        == valid
    )

    for field, changed_value in (
        ("random_state", 41),
        ("batch_size", 129),
        ("n_init", 4),
        ("max_iter", 301),
    ):
        changed_parameters = dict(kmeans)
        changed_parameters["parameters"] = dict(kmeans["parameters"])
        changed_parameters["parameters"][field] = changed_value
        with pytest.raises(ValueError, match="KMeans numerical state is invalid"):
            validate_embedding_cluster_support_state(
                kmeans_state=changed_parameters,
                svd_states=[treatment, interaction],
                expected_cluster_count=2,
                expected_kmeans_configuration=expected_kmeans_configuration,
            )

    with pytest.raises(ValueError, match="KMeans numerical state is invalid"):
        validate_embedding_cluster_support_state(
            kmeans_state=kmeans,
            svd_states=[treatment, interaction],
            expected_cluster_count=3,
        )

    with pytest.raises(ValueError, match="requires treatment and residualized_interaction"):
        validate_embedding_cluster_support_state(
            kmeans_state=kmeans,
            svd_states=[treatment],
        )

    one_cluster = dict(interaction)
    one_cluster.update(
        {
            "item_cluster_ids": [0],
            "weighted_matrix": np.asarray([[1.0, 1.0]]),
            "singular_values": np.asarray([np.sqrt(2.0)]),
            "components": np.asarray([[2**-0.5, 2**-0.5]]),
        }
    )
    with pytest.raises(ValueError, match="SVD state is invalid"):
        validate_embedding_cluster_support_state(
            kmeans_state=kmeans,
            svd_states=[treatment, one_cluster],
        )

    rank_one = state(
        "residualized_interaction",
        np.asarray([[1.0, 0.0], [2.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="rank-two support"):
        validate_embedding_cluster_support_state(
            kmeans_state=kmeans,
            svd_states=[treatment, rank_one],
        )

    near_collinear = state(
        "residualized_interaction",
        np.asarray([[1.0, 0.0], [1.0, 1e-8]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="rank-two support"):
        validate_embedding_cluster_support_state(
            kmeans_state=kmeans,
            svd_states=[treatment, near_collinear],
        )


def test_embedding_capture_artifacts_contain_no_heldout_text(tmp_path: Path):
    case = _case(tmp_path)
    artifact_bytes = b"".join(path.read_bytes() for path in sorted(case["artifact_dir"].iterdir()))
    for text in case["heldout_texts"]:
        assert text.encode("utf-8") not in artifact_bytes


@pytest.mark.parametrize(
    ("offsets", "message"),
    (
        ({"generator_pseudo_offset": 0.125}, "pseudo-target differs"),
        ({"generator_t_resid_offset": 0.125}, "treatment residual differs"),
    ),
)
def test_embedding_capture_rejects_registered_generator_output_mismatch(
    tmp_path: Path,
    offsets: dict[str, float],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        _case(tmp_path, **offsets)


def test_embedding_capture_rejects_evidence_tamper_and_wrong_index(tmp_path: Path):
    case = _case(tmp_path)
    raw_path = case["artifact_dir"] / "raw_embedding_evidence.json"
    raw = json.loads(raw_path.read_text())
    raw["contrasts"][0]["name"] = "tampered"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence file binding changed"):
        validate_embedding_native_capture(
            case["artifact_dir"],
            embedding_provider=case["provider"],
            fit_texts=case["fit_texts"],
        )

    second = _case(tmp_path / "second")
    with pytest.raises(ValueError, match="fit row order"):
        validate_embedding_native_capture(
            second["artifact_dir"],
            embedding_provider=second["provider"],
            fit_texts=second["fit_texts"],
            expected_fit_row_ids=tuple(reversed(second["fit_rows"])),
        )


@pytest.mark.parametrize("tamper_kind", ("extra_field", "duplicate_key"))
def test_embedding_capture_metadata_is_a_strict_json_envelope(
    tmp_path: Path,
    tamper_kind: str,
):
    case = _case(tmp_path)
    metadata_path = case["artifact_dir"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if tamper_kind == "extra_field":
        metadata["unregistered_field"] = "forbidden"
        body = {key: value for key, value in metadata.items() if key != "content_sha256"}
        metadata["content_sha256"] = _sha256_json(body)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        message = "invalid closed envelope"
    else:
        encoded = json.dumps(metadata, separators=(",", ":"))
        metadata_path.write_text(
            '{"scope_id":"duplicate",' + encoded[1:],
            encoding="utf-8",
        )
        message = "duplicate embedding JSON object key"
    with pytest.raises(ValueError, match=message):
        validate_embedding_native_capture(
            case["artifact_dir"],
            embedding_provider=case["provider"],
            fit_texts=case["fit_texts"],
            expected_scope_id="outer_01_inner_01",
            expected_fit_row_ids=case["fit_rows"],
            expected_heldout_row_ids=case["heldout_rows"],
        )


def test_semantic_policy_is_label_free_nonselecting_and_full_projection_is_lossless():
    fit_rows = tuple(range(8))
    policy = build_semantic_retrieval_training_only_policy(
        fit_row_ids=fit_rows,
        outer_fold=1,
        inner_fold=2,
        configured_fold_count=4,
        seed=19,
    )
    model_row = int(policy["model_fit_row_ids"][0])
    calibration_row = int(policy["calibration_row_ids"][0])
    raw = {
        "enabled": True,
        "contrasts": [
            {
                "name": "treatment",
                "contrast_family": "marginal",
                "direction_source": "fit_rows_only",
                "positive_aligned_chunks": [
                    {"row_id": model_row, "text": "modelpartitionuniqueterm active response"},
                    {
                        "row_id": calibration_row,
                        "text": "calibrationpartitionuniqueterm active response",
                    },
                ],
                "negative_aligned_chunks": [
                    {"row_id": model_row, "text": "model baseline decline"},
                    {"row_id": calibration_row, "text": "calibration baseline decline"},
                ],
            }
        ],
    }
    bundle = semantic_retrieval_projection_bundle(raw, policy=policy)
    full_terms = {
        row["concept"]
        for contrast in bundle["full"]["contrasts"]
        for row in contrast["concept_probe_scores"]
    }

    assert "selected_fold" not in policy
    assert policy["selection_kind"] == "none_deterministic_exhaustive"
    assert policy["nested_calibration_labels_accessed"] is False
    assert policy["partition_canaries_select_or_drop_terms"] is False
    assert policy["projection_vocabulary_max_features"] is None
    assert policy["projection_output_limit"] is None
    assert "modelpartitionuniqueterm" in full_terms
    assert "calibrationpartitionuniqueterm" in full_terms


def test_embedding_capture_rejects_provider_with_heldout_rows_bound(tmp_path: Path):
    row_count = 6
    texts = tuple(f"sufficient clinical words for row {index}" for index in range(row_count))
    _write_cache(tmp_path / "cache", texts, np.eye(row_count, dtype=np.float32))
    cache = SpentOnlyFrozenChunkEmbeddingCache(tmp_path / "cache")
    provider = cache.bind_spent(tuple(range(row_count)), texts)
    config = AppliedInferenceConfig()
    config.architecture.multi_model_forest.embedding_contrast.include_bow_phrases_as_concepts = (
        False
    )
    config.architecture.multi_model_forest.embedding_contrast.concept_phrases = []
    with pytest.raises(ValueError, match="fit rows only"):
        NativeEmbeddingProofCaptureSink(
            artifact_dir=tmp_path / "capture",
            scope_id="scope",
            outer_fold=1,
            inner_fold=1,
            fit_row_ids=(0, 1, 2, 3),
            heldout_row_ids=(4, 5),
            fit_texts=texts[:4],
            expected_fit_treatment=(0.0, 1.0, 0.0, 1.0),
            expected_fit_outcome=(0.0, 0.0, 1.0, 1.0),
            text_column=config.text_column,
            outcome_type=config.outcome_type,
            embedding_provider=provider,
            embedding_config=config.architecture.multi_model_forest.embedding_contrast,
            tfidf_nested_calibration_folds=2,
            seed=1,
        )


def _production_registration_context(case):
    digest = _catalog_ready_legacy_digest(
        importance={},
        embedding_evidence=_embedding_concepts_only(
            case["evidence"],
            contrastive_term_limit=None,
        ),
        htr_evidence={},
    )
    provenance = FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=case["fit_rows"],
        heldout_row_ids=case["heldout_rows"],
        scope="inner_train",
        inner_fold=1,
        artifact_id="embedding-native-production-registration-test",
    )
    catalog = build_role_neutral_evidence_catalog(
        (
            FoldEvidenceInput(
                LEGACY_ALL_SOURCE,
                {
                    "outer_fold": 1,
                    "inner_fold": 1,
                    "scope": "inner_train",
                    "n_rows": len(case["fit_rows"]),
                    "context": {"evidence_digest": digest},
                },
                provenance,
            ),
        ),
        require_all_source_kinds=False,
        require_all_architecture_families=False,
        require_upstream_completeness=False,
    )
    configuration = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
        "scope_id": "outer_01_inner_01",
        "text_column": case["text_column"],
        "treatment_column": case["treatment_column"],
        "outcome_column": case["outcome_column"],
        "outcome_type": case["outcome_type"],
        "embedding_config": case["metadata"]["embedding_config"],
        "capture_schema_version": case["metadata"]["schema_version"],
        "semantic_policy_schema_version": SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA,
        "tfidf_nested_calibration_folds": 5,
        "heldout_label_policy": "id_only_no_transform",
        "seed": 917,
        "split_registry_content_sha256": "b" * 64,
    }
    return catalog, configuration


@pytest.mark.parametrize(
    ("column_key", "lineage_name"),
    (
        ("treatment_column", "canonical_fit.treatment"),
        ("outcome_column", "canonical_fit.outcome"),
    ),
)
def test_production_embedding_registration_rejects_canonical_label_mismatch(
    tmp_path: Path,
    column_key: str,
    lineage_name: str,
):
    case = _case(tmp_path)
    catalog, configuration = _production_registration_context(case)
    modeling_data = case["dataset"].copy()
    column = case[column_key]
    modeling_data.loc[case["fit_rows"][0], column] += 0.375
    canonical = _canonical_embedding_scope_lineage(
        modeling_data=modeling_data,
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=case["fit_rows"],
        heldout_row_ids=case["heldout_rows"],
        text_column=case["text_column"],
        treatment_column=case["treatment_column"],
        outcome_column=case["outcome_column"],
        embedding_config=case["metadata"]["embedding_config"],
    )
    with pytest.raises(RuntimeError, match=lineage_name):
        _register_embedding_native_family_proofs(
            component_root=tmp_path,
            proof_directory=Path("rejected_proofs"),
            scope_id="outer_01_inner_01",
            catalog=catalog,
            capture_artifact_path=case["artifact_dir"],
            outer_fold=1,
            inner_fold=1,
            fit_row_ids=case["fit_rows"],
            heldout_row_ids=case["heldout_rows"],
            modeling_data=modeling_data,
            text_column=case["text_column"],
            treatment_column=case["treatment_column"],
            outcome_column=case["outcome_column"],
            embedding_provider=case["provider"],
            split_scope_fingerprint=canonical["split_scope_fingerprint"],
            data_projection_sha256=canonical["data_projection_sha256"],
            configuration=configuration,
        )


def test_production_embedding_registration_rejects_open_configuration(tmp_path: Path):
    case = _case(tmp_path)
    catalog, configuration = _production_registration_context(case)
    configuration["unregistered_field"] = "forbidden"
    with pytest.raises(ValueError, match="closed envelope"):
        _register_embedding_native_family_proofs(
            component_root=tmp_path,
            proof_directory=Path("rejected_proofs"),
            scope_id="outer_01_inner_01",
            catalog=catalog,
            capture_artifact_path=case["artifact_dir"],
            outer_fold=1,
            inner_fold=1,
            fit_row_ids=case["fit_rows"],
            heldout_row_ids=case["heldout_rows"],
            modeling_data=case["dataset"],
            text_column=case["text_column"],
            treatment_column=case["treatment_column"],
            outcome_column=case["outcome_column"],
            embedding_provider=case["provider"],
            split_scope_fingerprint="a" * 64,
            data_projection_sha256="c" * 64,
            configuration=configuration,
        )


def test_production_embedding_registration_replays_all_three_families_and_index(
    tmp_path: Path,
):
    case = _case(tmp_path)
    modeling_data = case["dataset"].drop(columns=["_oci_row_id"])
    safe_embedding = _embedding_concepts_only(
        case["evidence"],
        contrastive_term_limit=None,
    )
    digest = _catalog_ready_legacy_digest(
        importance={},
        embedding_evidence=safe_embedding,
        htr_evidence={},
    )
    provenance = FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=case["fit_rows"],
        heldout_row_ids=case["heldout_rows"],
        scope="inner_train",
        inner_fold=1,
        artifact_id="embedding-native-production-registration-test",
    )
    catalog = build_role_neutral_evidence_catalog(
        (
            FoldEvidenceInput(
                LEGACY_ALL_SOURCE,
                {
                    "outer_fold": 1,
                    "inner_fold": 1,
                    "scope": "inner_train",
                    "n_rows": len(case["fit_rows"]),
                    "context": {"evidence_digest": digest},
                },
                provenance,
            ),
        ),
        require_all_source_kinds=False,
        require_all_architecture_families=False,
        require_upstream_completeness=False,
    )
    assert all(
        catalog.family_atoms(family)
        for family in (
            EMBEDDING_WHOLE_COHORT,
            EMBEDDING_CLUSTERED,
            TFIDF_SEMANTIC_RETRIEVAL,
        )
    )
    configuration = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
        "scope_id": "outer_01_inner_01",
        "text_column": case["text_column"],
        "treatment_column": case["treatment_column"],
        "outcome_column": case["outcome_column"],
        "outcome_type": case["outcome_type"],
        "embedding_config": case["metadata"]["embedding_config"],
        "capture_schema_version": case["metadata"]["schema_version"],
        "semantic_policy_schema_version": (SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA),
        "tfidf_nested_calibration_folds": 5,
        "heldout_label_policy": "id_only_no_transform",
        "seed": 917,
        "split_registry_content_sha256": "b" * 64,
    }
    canonical = _canonical_embedding_scope_lineage(
        modeling_data=modeling_data,
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=case["fit_rows"],
        heldout_row_ids=case["heldout_rows"],
        text_column=case["text_column"],
        treatment_column=case["treatment_column"],
        outcome_column=case["outcome_column"],
        embedding_config=case["metadata"]["embedding_config"],
    )
    registration = _register_embedding_native_family_proofs(
        component_root=tmp_path,
        proof_directory=Path("native_embedding_family_proofs") / "outer_01_inner_01",
        scope_id="outer_01_inner_01",
        catalog=catalog,
        capture_artifact_path=case["artifact_dir"],
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=case["fit_rows"],
        heldout_row_ids=case["heldout_rows"],
        modeling_data=modeling_data,
        text_column=case["text_column"],
        treatment_column=case["treatment_column"],
        outcome_column=case["outcome_column"],
        embedding_provider=case["provider"],
        split_scope_fingerprint=canonical["split_scope_fingerprint"],
        data_projection_sha256=canonical["data_projection_sha256"],
        configuration=configuration,
    )
    assert registration["registered_families"] == list(
        PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS
    )
    assert len(registration["family_proofs"]) == 3
    semantic_metadata_path = (
        tmp_path / registration["family_proofs"][2]["native_fit_metadata"]["relative_path"]
    )
    semantic_metadata = json.loads(semantic_metadata_path.read_text(encoding="utf-8"))
    assert semantic_metadata["registered_heldout_columns_read"] == ["_oci_row_id"]
    assert semantic_metadata["registered_heldout_text_accessed"] is False
    assert semantic_metadata["tfidf_training_scope_policy"]["selection_kind"] == (
        "none_deterministic_exhaustive"
    )

    index_body = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
        "split_registry_content_sha256": "b" * 64,
        "registered_families": list(PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS),
        "exact_inner_scope_count": 1,
        "executable_checkpoint_files_retained": False,
        "scopes": [
            {
                "scope_id": "outer_01_inner_01",
                "outer_fold": 1,
                "inner_fold": 1,
                "registered_families": list(PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS),
                "content_sha256": registration["content_sha256"],
                "registration": registration["registration"],
            }
        ],
    }
    index_path = tmp_path / "embedding_native_family_proof_index.json"
    _write_immutable_json(
        index_path,
        {**index_body, "content_sha256": _sha256_json(index_body)},
    )
    index_registration = _component_file_registration(
        index_path,
        component_root=tmp_path,
    )
    expected_inner_scopes = {
        "outer_01_inner_01": {
            "outer_fold": 1,
            "inner_fold": 1,
            "fit_row_ids": list(case["fit_rows"]),
            "heldout_row_ids": list(case["heldout_rows"]),
        }
    }
    validated = _validate_embedding_native_family_proof_index(
        component_root=tmp_path,
        index_registration=index_registration,
        expected_inner_scopes=expected_inner_scopes,
        split_registry_content_sha256="b" * 64,
        modeling_data=modeling_data,
        text_column=case["text_column"],
        treatment_column=case["treatment_column"],
        outcome_column=case["outcome_column"],
        embedding_cache=case["cache"],
    )
    assert validated["exact_inner_scope_count"] == 1

    original_index_bytes = index_path.read_bytes()
    duplicate_encoded = json.dumps(validated, separators=(",", ":"))
    index_path.write_text(
        '{"schema_version":"duplicate",' + duplicate_encoded[1:],
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _validate_embedding_native_family_proof_index(
            component_root=tmp_path,
            index_registration=_component_file_registration(
                index_path,
                component_root=tmp_path,
            ),
            expected_inner_scopes=expected_inner_scopes,
            split_registry_content_sha256="b" * 64,
            modeling_data=modeling_data,
            text_column=case["text_column"],
            treatment_column=case["treatment_column"],
            outcome_column=case["outcome_column"],
            embedding_cache=case["cache"],
        )
    index_path.write_bytes(original_index_bytes)

    opened_index = dict(validated)
    opened_index["unregistered_field"] = "forbidden"
    opened_body = {key: value for key, value in opened_index.items() if key != "content_sha256"}
    opened_index["content_sha256"] = _sha256_json(opened_body)
    index_path.write_text(json.dumps(opened_index), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid closed envelope"):
        _validate_embedding_native_family_proof_index(
            component_root=tmp_path,
            index_registration=_component_file_registration(
                index_path,
                component_root=tmp_path,
            ),
            expected_inner_scopes=expected_inner_scopes,
            split_registry_content_sha256="b" * 64,
            modeling_data=modeling_data,
            text_column=case["text_column"],
            treatment_column=case["treatment_column"],
            outcome_column=case["outcome_column"],
            embedding_cache=case["cache"],
        )
    index_path.write_bytes(original_index_bytes)

    raw_path = case["artifact_dir"] / "raw_embedding_evidence.json"
    tampered = json.loads(raw_path.read_text(encoding="utf-8"))
    tampered["contrasts"][0]["name"] = "post_registration_tamper"
    raw_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="registered native component artifact changed"):
        _validate_embedding_native_family_proof_index(
            component_root=tmp_path,
            index_registration=index_registration,
            expected_inner_scopes=expected_inner_scopes,
            split_registry_content_sha256="b" * 64,
            modeling_data=modeling_data,
            text_column=case["text_column"],
            treatment_column=case["treatment_column"],
            outcome_column=case["outcome_column"],
            embedding_cache=case["cache"],
        )
