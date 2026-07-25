from __future__ import annotations

from copy import deepcopy

import pytest

from oci.inference.scientific_profile_identity import (
    scientific_profile_projection,
)


def _stage1_profile() -> dict:
    return {
        "config": {
            "dataset_path": "/cohorts/first.parquet",
            "cv_folds": 7,
            "architecture": {
                "htr_sentence_model": "/models/first",
                "htr_chunk_size_words": 73,
                "htr_chunk_overlap_words": 11,
                "htr_max_chunks": 10_003,
                "multi_model_forest": {
                    "fold_parallelism": "1",
                    "htr_jobs_per_gpu": 1,
                    "embedding_contrast": {
                        "cache_dir": "/cache/first",
                        "device": "cuda:0",
                        "chunk_size_words": 89,
                        "chunk_overlap_words": 13,
                        "max_chunks": 10_019,
                        "maximum_semantic_terms": None,
                    },
                },
            },
            "training": {"dataloader_workers": 2, "learning_rate": 0.003},
        }
    }


def test_stage1_projection_ignores_only_execution_and_locator_changes() -> None:
    first = _stage1_profile()
    second = deepcopy(first)
    second["config"]["dataset_path"] = "/relocated/cohort.parquet"
    second["config"]["architecture"]["htr_sentence_model"] = "/relocated/model"
    embedding = second["config"]["architecture"]["multi_model_forest"][
        "embedding_contrast"
    ]
    embedding["cache_dir"] = "/relocated/cache"
    embedding["device"] = "cuda:9"
    second["config"]["architecture"]["multi_model_forest"][
        "fold_parallelism"
    ] = "auto"
    second["config"]["architecture"]["multi_model_forest"][
        "htr_jobs_per_gpu"
    ] = 4
    second["config"]["training"]["dataloader_workers"] = 19

    left = scientific_profile_projection(first, profile_kind="stage1")
    right = scientific_profile_projection(second, profile_kind="stage1")

    assert left["content_sha256"] == right["content_sha256"]
    rendered = left["scientific_profile"]
    assert "dataset_path" not in rendered["config"]
    assert (
        "device"
        not in rendered["config"]["architecture"]["multi_model_forest"][
            "embedding_contrast"
        ]
    )


def test_stage1_projection_retains_text_capacity_and_training_settings() -> None:
    baseline = scientific_profile_projection(
        _stage1_profile(),
        profile_kind="stage1",
    )["content_sha256"]
    for path, replacement in (
        (
            ("config", "architecture", "htr_chunk_size_words"),
            74,
        ),
        (
            (
                "config",
                "architecture",
                "multi_model_forest",
                "embedding_contrast",
                "max_chunks",
            ),
            10_020,
        ),
        (("config", "training", "learning_rate"), 0.004),
        (("config", "cv_folds"), 8),
    ):
        changed = _stage1_profile()
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        assert (
            scientific_profile_projection(
                changed,
                profile_kind="stage1",
            )["content_sha256"]
            != baseline
        )


def test_query_projection_retains_every_configured_hyperparameter() -> None:
    first = {"query_epochs": 17, "rag_max_chunks_per_patient": None}
    second = {"query_epochs": 18, "rag_max_chunks_per_patient": None}

    assert (
        scientific_profile_projection(
            first,
            profile_kind="neural_query",
        )["content_sha256"]
        != scientific_profile_projection(
            second,
            profile_kind="neural_query",
        )["content_sha256"]
    )


def test_stage1_projection_rejects_unregistered_external_corpus_locators() -> None:
    profile = _stage1_profile()
    profile["config"]["architecture"]["multi_model_forest"][
        "embedding_contrast"
    ]["external_corpus_cache_dirs"] = ["/unregistered/corpus"]

    with pytest.raises(ValueError, match="external corpus locators"):
        scientific_profile_projection(profile, profile_kind="stage1")
