from __future__ import annotations

import hashlib
import json
from pathlib import Path

from oci.inference.production_stage1_bundle import (
    ProductionStage1BundleBuilder,
    load_applied_stage1_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE1_PROFILE = (
    REPOSITORY_ROOT / "example_configs" / "production_all_evidence_stage1_full.json"
)
QUERY_PROFILE = (
    REPOSITORY_ROOT
    / "example_configs"
    / "production_all_evidence_neural_query_full.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checked_in_full_profiles_match_the_fixed_stage1_scientific_design() -> None:
    assert _sha256(STAGE1_PROFILE) == (
        "1af35bb0a107c28a79a76fa74319de105d2ee4352c12345d8bdbe97869b9cfc0"
    )
    assert _sha256(QUERY_PROFILE) == (
        "2d465f6c2eae71d4c9f4d18716f0919aee954b0afde9ef4414a27c5ad4771997"
    )

    config = load_applied_stage1_config(STAGE1_PROFILE)
    query, query_identity = ProductionStage1BundleBuilder._load_query_config(
        QUERY_PROFILE
    )
    architecture = config.architecture
    multi_model = architecture.multi_model_forest
    embedding = multi_model.embedding_contrast
    htr_native = architecture.agentic_attention_variable_forest

    assert config.cv_folds == 5
    assert config.training.epochs == 50
    assert htr_native.nuisance_epochs == 50
    assert htr_native.effect_epochs == 50
    assert multi_model.nuisance_folds == 5
    assert multi_model.effect_folds == 5
    assert multi_model.candidate_consistency_inner_folds == 5
    assert multi_model.tfidf_nested_calibration_folds == 3
    assert architecture.explicit_feature_forest.interaction_inner_folds == 3

    assert embedding.model_name == "Qwen/Qwen3-Embedding-8B"
    assert (
        embedding.chunk_size_words,
        embedding.chunk_overlap_words,
        embedding.max_chunks,
        embedding.max_seq_length,
        embedding.batch_size,
    ) == (256, 64, 128, 1024, 1)
    assert embedding.normalize_embeddings is True
    assert (
        embedding.cluster_contrast_n_clusters,
        embedding.cluster_contrast_kmeans_n_init,
        embedding.cluster_contrast_min_cluster_size,
        embedding.cluster_contrast_min_group_size,
        embedding.cluster_contrast_min_cell_size,
        embedding.cluster_contrast_max_components,
    ) == (10, 20, 24, 8, 4, 5)
    assert embedding.include_cell_contrasts is True
    assert embedding.include_orthogonal_r_score_contrasts is True
    assert embedding.include_confounder_vector_contrast is True
    assert embedding.include_residualized_interaction_contrast is True
    assert embedding.include_cluster_contrast_vectors is True

    assert (
        architecture.htr_chunk_size_words,
        architecture.htr_chunk_overlap_words,
        architecture.htr_max_chunks,
        architecture.htr_max_chunk_length,
        architecture.htr_sentence_encoder_batch_size,
    ) == (96, 24, 512, 512, 16)
    assert architecture.htr_freeze_sentence_encoder is False

    for forest in (
        architecture.causal_forest,
        architecture.explicit_feature_forest,
    ):
        assert forest.n_estimators == 200
        assert forest.min_samples_leaf == 10
        assert forest.max_features == "sqrt"
        assert forest.honest is True
        assert forest.inference is True

    assert query_identity["provided"] is True
    assert query.query_epochs == 120
    assert query.final_refit_epochs == 80
    assert query.query_inner_folds == 5
    assert query.max_review_rounds == 2
    assert query.max_canonical_features == 20
    assert query.rag_max_chunks_per_patient == 128


def test_checked_in_profiles_are_secret_free_and_contain_only_inert_endpoints() -> None:
    raw = json.loads(STAGE1_PROFILE.read_text(encoding="utf-8"))
    applied = raw["config"]
    architecture = applied["architecture"]
    agent = architecture["agentic_feature_search"]
    explicit = applied["explicit_features"]

    assert agent["agent_api_key"] == "EMPTY"
    assert explicit["vllm_api_key"] == "EMPTY"
    assert agent["agent_server_url"] == "http://unused.invalid/v1"
    assert explicit["vllm_server_url"] == "http://unused.invalid/v1"
    assert agent["agent_model_name"] == "unused-stage1-model"
    assert explicit["vllm_model_name"] == "unused-stage1-model"

    serialized = json.dumps(raw, sort_keys=True)
    assert "camus" not in serialized.lower()
    assert "RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic" not in serialized
    assert "sk-" not in serialized
