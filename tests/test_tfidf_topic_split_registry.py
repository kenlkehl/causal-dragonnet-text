import json
from dataclasses import asdict
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
from oci.inference.tfidf_topic_agentic_forest import (
    validate_tfidf_topic_stage2_handoff,
)
from oci.inference.tfidf_topic_discovery import (
    HANDOFF_SCHEMA_VERSION,
    row_set_fingerprint,
    stable_hash,
)
from oci.inference.tfidf_topic_handoff_reseal import (
    PRE_ORPHAN_TOPIC_SCORE_TEST_SCHEMA_VERSION,
    _expected_stack_hashes,
    derive_tfidf_topic_split_registry_from_handoff,
    legacy_tfidf_topic_stage1_config_hash,
    pre_orphan_tfidf_topic_stage1_config_hash,
    reseal_tfidf_topic_handoff,
)
from oci.inference.tfidf_topic_score_selection import (
    TOPIC_SCORE_TEST_SCHEMA_VERSION,
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


def test_pre_orphan_legacy_hash_omits_only_orphan_controls(tmp_path):
    config = _config(_write_registry(tmp_path / "registry.json"))
    nn_config = config.architecture.multi_model_forest
    topic = {
        key: value
        for key, value in asdict(nn_config.tfidf_topic).items()
        if not key.startswith("orphan_ngram_")
    }
    expected = stable_hash(
        {
            "schema": HANDOFF_SCHEMA_VERSION,
            "topic_score_test_schema": PRE_ORPHAN_TOPIC_SCORE_TEST_SCHEMA_VERSION,
            "views": [asdict(view) for view in nn_config.bow_views],
            "nuisance_folds": nn_config.nuisance_folds,
            "topic": topic,
            "text_column": config.text_column,
            "treatment_column": config.treatment_column,
            "outcome_column": config.outcome_column,
            "outcome_type": config.outcome_type,
        }
    )
    assert pre_orphan_tfidf_topic_stage1_config_hash(config) == expected
    assert pre_orphan_tfidf_topic_stage1_config_hash(config) != (
        legacy_tfidf_topic_stage1_config_hash(config)
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


def _score_test_payload(*, fit_n: int, heldout_n: int) -> dict:
    return {
        "schema_version": TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "status": "completed",
        "uses_heldout_treatment_and_outcome": True,
        "fits_patient_level_cate_model": False,
        "constructs_divided_pseudo_target": False,
        "fit_n": int(fit_n),
        "heldout_n": int(heldout_n),
        "banks": {
            bank: {
                "topic_tests": [],
                "selected_topic_ids": [],
                "ngram_selection_count": 0,
                "bootstrap_calibration": {
                    "complete_topic_family": True,
                    "complete_term_group_family": True,
                    "complete_ngram_family": True,
                },
            }
            for bank in ("treatment", "outcome", "effect")
        },
    }


def _source_context_row(
    *,
    base_dir: Path,
    config: AppliedInferenceConfig,
    outer_fold: int,
    inner_fold: int | None,
    fit_ids: list[int],
    heldout_ids: list[int],
    source_hash: str,
) -> dict:
    scope = "full_outer_train" if inner_fold is None else "candidate_selection_inner_fit"
    fold_key = outer_fold if inner_fold is None else 1000 * outer_fold + inner_fold
    context_dir = base_dir / f"context-{fold_key}"
    context_dir.mkdir(parents=True)
    fitted_path = context_dir / "fitted_context.joblib"
    fitted_path.write_bytes(b"sealed fitted context placeholder")
    ngram_paths = {}
    for bank in ("treatment", "outcome", "effect"):
        path = context_dir / f"{bank}_scores.parquet"
        path.write_bytes(b"sealed score placeholder")
        ngram_paths[bank] = str(path)
    fit_topic_path = context_dir / "fit_topics.npz"
    heldout_topic_path = context_dir / "heldout_topics.npz"
    np.savez_compressed(fit_topic_path)
    np.savez_compressed(heldout_topic_path)

    nuisance_rows = []
    for row_id in fit_ids:
        nuisance_rows.append(
            {
                "_oci_row_id": row_id,
                "prediction_scope": "fit_oof",
                "fit_row_ids": [value for value in fit_ids if value != row_id],
                "treatment_stacked": 0.5,
                "outcome_stacked": 0.5,
            }
        )
    for row_id in heldout_ids:
        nuisance_rows.append(
            {
                "_oci_row_id": row_id,
                "prediction_scope": "external_heldout",
                "fit_row_ids": list(fit_ids),
                "treatment_stacked": 0.5,
                "outcome_stacked": 0.5,
            }
        )
    nuisance_path = context_dir / "nuisance.parquet"
    pd.DataFrame(nuisance_rows).to_parquet(nuisance_path, index=False)
    if inner_fold is None:
        score_path = None
        compact_score = {
            "status": "not_run",
            "uses_heldout_treatment_and_outcome": False,
        }
    else:
        score_path = context_dir / "topic_scores.json"
        score_path.write_text(
            json.dumps(
                _score_test_payload(
                    fit_n=len(fit_ids),
                    heldout_n=len(heldout_ids),
                )
            ),
            encoding="utf-8",
        )
        compact_score = {
            "status": "completed",
            "uses_heldout_treatment_and_outcome": True,
        }
    stack_hashes = _expected_stack_hashes(config)
    discovery = {
        "scope_id": f"scope-{fold_key}",
        "fit_row_ids": list(fit_ids),
        "heldout_row_ids": list(heldout_ids),
        "fit_row_fingerprint": row_set_fingerprint(fit_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
        "config_hash": stable_hash(asdict(config.architecture.multi_model_forest.tfidf_topic)),
        "common_vocabulary_size": 0,
        "common_vocabulary": [],
        "nuisance": {
            target: {"stack_config_hash": stack_hashes[target]}
            for target in ("treatment", "outcome")
        },
        "topic_banks": {bank: {"topics": []} for bank in ("treatment", "outcome", "effect")},
        "topic_score_tests": compact_score,
        "artifacts": {
            "fitted_context": str(fitted_path),
            "fit_topic_values": str(fit_topic_path),
            "heldout_topic_values": str(heldout_topic_path),
            "nuisance_predictions": str(nuisance_path),
            "ngram_scores": ngram_paths,
            "topic_score_tests": None if score_path is None else str(score_path),
        },
    }
    return {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "stage1_config_hash": source_hash,
        "fold_key": fold_key,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "scope": scope,
        "fit_row_ids": list(fit_ids),
        "heldout_row_ids": list(heldout_ids),
        "fit_row_fingerprint": row_set_fingerprint(fit_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
        "discovery": discovery,
    }


def test_reseal_is_non_mutating_and_stage2_accepts_registry_seal(tmp_path):
    registry_path = tmp_path / "registry.json"
    config = _config(registry_path)
    data = _dataset()
    source_dir = tmp_path / "legacy"
    source_dir.mkdir()
    source_path = source_dir / "handoff.jsonl"
    source_hash = pre_orphan_tfidf_topic_stage1_config_hash(config)
    rows = []
    for outer in _registry_payload()["outer_folds"]:
        for inner in outer["inner_folds"]:
            rows.append(
                _source_context_row(
                    base_dir=source_dir,
                    config=config,
                    outer_fold=outer["outer_fold"],
                    inner_fold=inner["inner_fold"],
                    fit_ids=inner["fit_row_ids"],
                    heldout_ids=inner["heldout_row_ids"],
                    source_hash=source_hash,
                )
            )
        rows.append(
            _source_context_row(
                base_dir=source_dir,
                config=config,
                outer_fold=outer["outer_fold"],
                inner_fold=None,
                fit_ids=outer["fit_row_ids"],
                heldout_ids=outer["heldout_row_ids"],
                source_hash=source_hash,
            )
        )
    for row in rows:
        if row["scope"] != "candidate_selection_inner_fit":
            continue
        score_path = Path(row["discovery"]["artifacts"]["topic_score_tests"])
        score = json.loads(score_path.read_text(encoding="utf-8"))
        score["schema_version"] = PRE_ORPHAN_TOPIC_SCORE_TEST_SCHEMA_VERSION
        score_path.write_text(json.dumps(score), encoding="utf-8")
        row["discovery"]["topic_score_tests"][
            "schema_version"
        ] = PRE_ORPHAN_TOPIC_SCORE_TEST_SCHEMA_VERSION
    source_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    source_manifest = source_dir / "manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": HANDOFF_SCHEMA_VERSION,
                "stage1_config_hash": source_hash,
            }
        ),
        encoding="utf-8",
    )
    derived = derive_tfidf_topic_split_registry_from_handoff(
        source_handoff_path=source_path,
        output_registry_path=registry_path,
        dataset_row_count=len(data),
        outer_fold_count=config.cv_folds,
        inner_fold_count=config.architecture.multi_model_forest.candidate_consistency_inner_folds,
    )
    assert derived["dataset_row_count"] == len(data)
    assert derived["outer_folds"] == _registry_payload()["outer_folds"]
    raw_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert raw_registry["provenance"]["folds_regenerated_from_seed"] is False
    assert raw_registry["provenance"]["source_stage1_config_hash"] == source_hash
    before_handoff = source_path.read_bytes()
    before_manifest = source_manifest.read_bytes()
    before_scores = {
        Path(row["discovery"]["artifacts"]["topic_score_tests"]): Path(
            row["discovery"]["artifacts"]["topic_score_tests"]
        ).read_bytes()
        for row in rows
        if row["scope"] == "candidate_selection_inner_fit"
    }

    output_path = tmp_path / "resealed" / "handoff.jsonl"
    manifest = reseal_tfidf_topic_handoff(
        source_handoff_path=source_path,
        output_handoff_path=output_path,
        dataset=data,
        config=config,
    )

    assert source_path.read_bytes() == before_handoff
    assert source_manifest.read_bytes() == before_manifest
    assert all(path.read_bytes() == content for path, content in before_scores.items())
    assert manifest["migration"]["source_artifacts_mutated"] is False
    score_migrations = manifest["migration"]["score_test_schema_migrations"]
    assert len(score_migrations) == 6
    assert all(not item["statistics_recomputed"] for item in score_migrations)
    assert all(not item["labels_read"] for item in score_migrations)
    assert manifest["stage1_config_hash"] == tfidf_topic_stage1_config_hash(config, data)
    output_rows = [
        json.loads(line) for line in output_path.read_text().splitlines() if line.strip()
    ]
    assert {row["stage1_config_hash"] for row in output_rows} == {manifest["stage1_config_hash"]}
    assert all(
        Path(row["discovery"]["artifacts"]["nuisance_predictions"]).is_absolute()
        for row in output_rows
    )
    inner_rows = [
        row for row in output_rows if row["scope"] == "candidate_selection_inner_fit"
    ]
    assert all(
        Path(row["discovery"]["artifacts"]["topic_score_tests"]).parent.name
        == "migrated_score_tests"
        for row in inner_rows
    )
    migrated_scores = [
        json.loads(
            Path(row["discovery"]["artifacts"]["topic_score_tests"]).read_text(
                encoding="utf-8"
            )
        )
        for row in inner_rows
    ]
    assert all(
        score["schema_version"] == TOPIC_SCORE_TEST_SCHEMA_VERSION
        and score["effect_orphan_ngram_branch"]["status"] == "disabled"
        and not score["schema_migration"]["statistics_recomputed"]
        and not score["schema_migration"]["labels_read"]
        for score in migrated_scores
    )
    preflight = validate_tfidf_topic_stage2_handoff(
        dataset=data,
        config=config,
        handoff_path=output_path,
    )
    assert preflight["status"] == "passed"
    assert preflight["outer_test_rows_predicted_once"] is True


def test_reseal_rejects_a_registry_artifact_split_mismatch(tmp_path):
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
