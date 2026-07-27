import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.config import (
    AppliedInferenceConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
    TfidfTopicDiscoveryConfig,
)
from oci.inference.tfidf_topic_discovery import (
    row_set_fingerprint,
)
from oci.inference.tfidf_topic_split_registry import (
    TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
    SplitRegistryError,
    load_tfidf_topic_split_registry,
    validate_handoff_rows_against_split_registry,
)
from oci.inference.tfidf_topic_stage1 import (
    _inner_split_plan,
    _outer_split_plan,
    tfidf_topic_stage1_config_hash,
)


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clinical_text": [f"baseline record {index}" for index in range(12)],
            "treatment_indicator": [0, 1] * 6,
            "outcome_indicator": [0, 0, 1, 1] * 3,
            "true_ite_prob": np.linspace(-0.2, 0.2, 12),
        }
    )


def _registry_payload() -> dict:
    heldout_groups = ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11])
    all_rows = list(range(12))
    outer_folds = []
    for outer_fold, heldout in enumerate(heldout_groups, start=1):
        fit = [row_id for row_id in all_rows if row_id not in heldout]
        midpoint = len(fit) // 2
        outer_folds.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": fit,
                "heldout_row_ids": list(heldout),
                "inner_folds": [
                    {
                        "inner_fold": 1,
                        "fit_row_ids": fit[midpoint:],
                        "heldout_row_ids": fit[:midpoint],
                    },
                    {
                        "inner_fold": 2,
                        "fit_row_ids": fit[:midpoint],
                        "heldout_row_ids": fit[midpoint:],
                    },
                ],
            }
        )
    return {
        "schema_version": TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
        "dataset_row_count": 12,
        "outer_folds": outer_folds,
    }


def _write_registry(path: Path, payload: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or _registry_payload()), encoding="utf-8")
    return path


def _config(registry_path: Path) -> AppliedInferenceConfig:
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
                split_registry_path=str(registry_path),
                candidate_consistency_inner_folds=2,
                nuisance_folds=2,
                tfidf_topic=TfidfTopicDiscoveryConfig(
                    orphan_ngram_enabled=False,
                ),
            ),
        ),
    )
    config.seed = 37
    return config


def test_registry_controls_outer_inner_plans_and_stage1_identity(tmp_path):
    first_path = _write_registry(tmp_path / "first" / "splits.json")
    second_path = _write_registry(tmp_path / "second" / "same-splits.json")
    data = _dataset()
    first_config = _config(first_path)
    second_config = _config(second_path)

    outer, metadata = _outer_split_plan(data, first_config)
    assert metadata["method"] == "explicit_split_registry"
    assert outer[0][0].tolist() == list(range(4, 12))
    assert outer[0][1].tolist() == list(range(4))
    outer_train = data.iloc[outer[0][0]].copy()
    outer_train["_oci_row_id"] = outer[0][0]
    inner, inner_metadata = _inner_split_plan(
        outer_train,
        first_config,
        outer_fold=1,
        validated_registry=load_tfidf_topic_split_registry(
            first_path,
            dataset_row_count=len(data),
            outer_fold_count=3,
            inner_fold_count=2,
        ),
    )
    assert inner_metadata["method"] == "explicit_split_registry"
    assert outer_train.iloc[inner[0][0]]["_oci_row_id"].tolist() == [8, 9, 10, 11]
    assert outer_train.iloc[inner[0][1]]["_oci_row_id"].tolist() == [4, 5, 6, 7]

    # Registry location is provenance only; its validated content is identity.
    assert tfidf_topic_stage1_config_hash(first_config, data) == tfidf_topic_stage1_config_hash(
        second_config, data
    )
    changed = _registry_payload()
    changed["outer_folds"][0]["fit_row_ids"] = list(
        reversed(changed["outer_folds"][0]["fit_row_ids"])
    )
    changed_path = _write_registry(tmp_path / "changed.json", changed)
    assert tfidf_topic_stage1_config_hash(first_config, data) != tfidf_topic_stage1_config_hash(
        _config(changed_path), data
    )


@pytest.mark.parametrize("failure", ["outer_overlap", "out_of_bounds", "inner_coverage"])
def test_registry_rejects_invalid_partitions(tmp_path, failure):
    payload = _registry_payload()
    if failure == "outer_overlap":
        payload["outer_folds"][0]["fit_row_ids"].append(0)
    elif failure == "out_of_bounds":
        payload["outer_folds"][0]["heldout_row_ids"][0] = 12
    else:
        payload["outer_folds"][0]["inner_folds"][1]["heldout_row_ids"] = [8, 9, 10]
        payload["outer_folds"][0]["inner_folds"][1]["fit_row_ids"] = [4, 5, 6, 7, 11]
    path = _write_registry(tmp_path / f"{failure}.json", payload)
    with pytest.raises(SplitRegistryError):
        load_tfidf_topic_split_registry(
            path,
            dataset_row_count=12,
            outer_fold_count=3,
            inner_fold_count=2,
        )

def test_registry_rejects_an_artifact_split_order_mismatch(tmp_path):
    registry_path = _write_registry(tmp_path / "registry.json")
    registry = load_tfidf_topic_split_registry(
        registry_path,
        dataset_row_count=12,
        outer_fold_count=3,
        inner_fold_count=2,
    )
    # A row-order change preserves set fingerprints, so exact registry matching
    # must still reject it before any migrated handoff is written.
    fold = registry["outer_folds"][0]
    assert row_set_fingerprint(fold["fit_row_ids"]) == row_set_fingerprint(
        list(reversed(fold["fit_row_ids"]))
    )
    rows = []
    for outer in registry["outer_folds"]:
        for split, scope in [
            *[(inner, "candidate_selection_inner_fit") for inner in outer["inner_folds"]],
            (outer, "full_outer_train"),
        ]:
            inner_fold = split.get("inner_fold")
            fit_ids = list(split["fit_row_ids"])
            heldout_ids = list(split["heldout_row_ids"])
            rows.append(
                {
                    "fold_key": (
                        outer["outer_fold"]
                        if inner_fold is None
                        else 1000 * outer["outer_fold"] + inner_fold
                    ),
                    "outer_fold": outer["outer_fold"],
                    "inner_fold": inner_fold,
                    "scope": scope,
                    "fit_row_ids": fit_ids,
                    "heldout_row_ids": heldout_ids,
                    "fit_row_fingerprint": row_set_fingerprint(fit_ids),
                    "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
                    "discovery": {
                        "fit_row_ids": fit_ids,
                        "heldout_row_ids": heldout_ids,
                        "fit_row_fingerprint": row_set_fingerprint(fit_ids),
                        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
                    },
                }
            )
    full = next(
        row for row in rows if row["outer_fold"] == 1 and row["scope"] == "full_outer_train"
    )
    full["fit_row_ids"] = list(reversed(full["fit_row_ids"]))
    full["discovery"]["fit_row_ids"] = list(full["fit_row_ids"])
    with pytest.raises(SplitRegistryError, match="row ids/order"):
        validate_handoff_rows_against_split_registry(rows, registry)
