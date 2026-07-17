import json

import numpy as np
import pandas as pd

from oci.config import (
    AppliedInferenceConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
)
from oci.inference.tfidf_topic_discovery import stable_hash
from oci.inference.tfidf_topic_stage1 import (
    _outer_split_plan,
    make_joint_treatment_outcome_splits,
    tfidf_topic_dataset_fingerprints,
    tfidf_topic_stage1_cache_is_valid,
    tfidf_topic_stage1_config_hash,
    tfidf_topic_stage1_identity,
)


def _dataset() -> pd.DataFrame:
    rows = []
    for treatment in (0, 1):
        for outcome in (0, 1):
            for replicate in range(9):
                rows.append(
                    {
                        "clinical_text": (
                            f"baseline record arm {treatment} response {outcome} "
                            f"patient {replicate}"
                        ),
                        "treatment_indicator": treatment,
                        "outcome_indicator": outcome,
                        "true_ite_prob": float(replicate) / 100.0,
                    }
                )
    return pd.DataFrame(rows)


def _config(seed: int = 17) -> AppliedInferenceConfig:
    config = AppliedInferenceConfig(
        dataset_path="in_memory",
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=3,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                candidate_consistency_inner_folds=3,
                nuisance_folds=3,
            ),
        ),
    )
    config.seed = int(seed)
    return config


def test_joint_treatment_outcome_split_is_balanced_and_seeded():
    data = _dataset()
    splits, metadata = make_joint_treatment_outcome_splits(
        data,
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        n_splits=3,
        seed=17,
    )
    repeated, repeated_metadata = make_joint_treatment_outcome_splits(
        data,
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        n_splits=3,
        seed=17,
    )
    changed_seed, _ = make_joint_treatment_outcome_splits(
        data,
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        n_splits=3,
        seed=18,
    )

    assert metadata == repeated_metadata
    assert metadata["method"] == "stratified_joint_treatment_outcome"
    assert metadata["fallback_reason"] is None
    assert all(np.array_equal(left[1], right[1]) for left, right in zip(splits, repeated))
    assert any(not np.array_equal(left[1], right[1]) for left, right in zip(splits, changed_seed))
    for _, heldout in splits:
        cell_counts = (
            data.iloc[heldout].groupby(["treatment_indicator", "outcome_indicator"]).size()
        )
        assert cell_counts.to_dict() == {(0, 0): 3, (0, 1): 3, (1, 0): 3, (1, 1): 3}


def test_joint_split_falls_back_only_when_a_joint_cell_is_too_small():
    data = _dataset().iloc[:10].copy()
    data.loc[data.index[-1], ["treatment_indicator", "outcome_indicator"]] = [1, 1]
    splits, metadata = make_joint_treatment_outcome_splits(
        data,
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        n_splits=3,
        seed=23,
    )

    assert len(splits) == 3
    assert metadata["method"] == "kfold_fallback"
    assert metadata["minimum_joint_stratum_count"] < 3
    assert metadata["fallback_reason"] == ("minimum_joint_stratum_count_below_requested_folds")


def test_outer_plan_uses_config_seed_instead_of_a_fixed_constant():
    data = _dataset()
    first, first_metadata = _outer_split_plan(data, _config(seed=101))
    second, second_metadata = _outer_split_plan(data, _config(seed=102))

    assert first_metadata["seed"] == 101
    assert second_metadata["seed"] == 102
    assert any(not np.array_equal(left[1], right[1]) for left, right in zip(first, second))


def test_stage1_hash_tracks_model_content_row_order_seed_and_split_semantics():
    data = _dataset()
    config = _config(seed=31)
    identity = tfidf_topic_dataset_fingerprints(data, config)
    baseline_hash = tfidf_topic_stage1_config_hash(config, data)

    oracle_only_change = data.copy()
    oracle_only_change["true_ite_prob"] += 100.0
    assert tfidf_topic_stage1_config_hash(config, oracle_only_change) == baseline_hash

    content_change = data.copy()
    content_change.loc[0, "clinical_text"] += " changed"
    assert tfidf_topic_stage1_config_hash(config, content_change) != baseline_hash

    reordered = data.iloc[::-1].reset_index(drop=True)
    reordered_identity = tfidf_topic_dataset_fingerprints(reordered, config)
    assert reordered_identity["content_fingerprint"] == identity["content_fingerprint"]
    assert reordered_identity["ordered_row_fingerprint"] != identity["ordered_row_fingerprint"]
    assert tfidf_topic_stage1_config_hash(config, reordered) != baseline_hash

    changed_seed = _config(seed=32)
    assert tfidf_topic_stage1_config_hash(changed_seed, data) != baseline_hash

    changed_inner_folds = _config(seed=31)
    changed_inner_folds.architecture.multi_model_forest.candidate_consistency_inner_folds = 2
    assert tfidf_topic_stage1_config_hash(changed_inner_folds, data) != baseline_hash


def test_stage1_manifest_cache_validation_fails_closed(tmp_path):
    data = _dataset()
    config = _config(seed=41)
    identity = tfidf_topic_stage1_identity(config, data)
    output_path = tmp_path / "stage1.parquet"
    handoff_path = tmp_path / "handoff" / "discovery_contexts.jsonl"
    manifest_path = handoff_path.parent / "manifest.json"
    output_path.write_bytes(b"placeholder")
    handoff_path.parent.mkdir(parents=True)
    handoff_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "stage1_config_hash": stable_hash(identity),
                "dataset_content_fingerprint": identity["dataset"]["content_fingerprint"],
                "dataset_ordered_row_fingerprint": identity["dataset"]["ordered_row_fingerprint"],
                "split_semantics_hash": identity["split_semantics_hash"],
            }
        ),
        encoding="utf-8",
    )

    assert tfidf_topic_stage1_cache_is_valid(
        dataset=data,
        config=config,
        output_path=output_path,
        handoff_path=handoff_path,
    )
    changed = data.copy()
    changed.loc[0, "outcome_indicator"] = 1 - changed.loc[0, "outcome_indicator"]
    assert not tfidf_topic_stage1_cache_is_valid(
        dataset=changed,
        config=config,
        output_path=output_path,
        handoff_path=handoff_path,
    )
    assert not tfidf_topic_stage1_cache_is_valid(
        dataset=data,
        config=_config(seed=42),
        output_path=output_path,
        handoff_path=handoff_path,
    )
