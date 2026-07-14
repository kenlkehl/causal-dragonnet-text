import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.inference.tfidf_topic_discovery import row_set_fingerprint
from oci.inference.tfidf_topic_score_forest import (
    TopicScoreForestConfig,
    prepare_topic_score_matrices,
    run_tfidf_topic_score_forest,
)


def _topic_metadata(width=2):
    return {
        bank: {
            "actual_topic_count": width,
            "topics": [
                {
                    "topic_id": f"{bank}_topic_{index + 1:03d}",
                    "terms": [{"term": f"{bank}_term_{index}", "loading": 1.0}],
                }
                for index in range(width)
            ],
        }
        for bank in ("treatment", "outcome", "effect")
    }


def test_prepare_topic_scores_uses_training_only_and_assigns_roles():
    fit = {
        "treatment": np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]]),
        "outcome": np.array([[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]]),
        "effect": np.array([[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]]),
    }
    heldout = {bank: values[:1] + 10_000.0 for bank, values in fit.items()}
    x_fit, x_test, w_fit, w_test, transforms = prepare_topic_score_matrices(
        fit_scores=fit,
        heldout_scores=heldout,
        discovery={"topic_banks": _topic_metadata()},
        config=TopicScoreForestConfig(n_estimators=4),
    )
    assert x_fit.shape == (3, 2)
    assert x_test.shape == (1, 2)
    assert w_fit.shape == (3, 4)
    assert w_test.shape == (1, 4)
    np.testing.assert_allclose(transforms["effect"].means, [4.0, 6.0])
    np.testing.assert_allclose(x_fit.mean(axis=0), 0.0, atol=1e-7)
    assert transforms["treatment"].topic_ids[0].startswith("treatment_topic_")


class _FakeForest:
    fit_shapes = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, x, treatment, outcome, W=None):
        self.__class__.fit_shapes.append((x.shape, W.shape, len(treatment), len(outcome)))
        return self

    def predict(self, x, return_ci=True):
        tau = 0.15 * x[:, 0] - 0.05 * x[:, 1]
        result = {"tau_pred": tau}
        if return_ci:
            result.update(tau_lower=tau - 0.1, tau_upper=tau + 0.1)
        return result


def _write_fake_handoff(tmp_path: Path, dataset: pd.DataFrame) -> Path:
    contexts = []
    all_ids = np.arange(len(dataset), dtype=int)
    heldout_sets = [all_ids[::2], all_ids[1::2]]
    for fold, heldout_ids in enumerate(heldout_sets, start=1):
        fit_ids = np.asarray([value for value in all_ids if value not in set(heldout_ids)])
        context_dir = tmp_path / f"context_{fold}"
        context_dir.mkdir()

        def scores(row_ids):
            base = np.asarray(row_ids, dtype=float)
            return {
                "treatment": np.column_stack([1.0 + base, 2.0 + base**2]),
                "outcome": np.column_stack([2.0 + base, 3.0 + base**2]),
                "effect": np.column_stack([1.0 + base, 1.0 + (base % 3)]),
            }

        np.savez_compressed(context_dir / "fit.npz", **scores(fit_ids))
        np.savez_compressed(context_dir / "heldout.npz", **scores(heldout_ids))
        nuisance_rows = []
        for scope, row_ids in (("fit_oof", fit_ids), ("external_heldout", heldout_ids)):
            for row_id in row_ids:
                nuisance_rows.append(
                    {
                        "_oci_row_id": int(row_id),
                        "prediction_scope": scope,
                        "treatment_stacked": 0.35 + 0.04 * (int(row_id) % 4),
                        "outcome_stacked": 0.30 + 0.05 * (int(row_id) % 5),
                    }
                )
        pd.DataFrame(nuisance_rows).to_parquet(context_dir / "nuisance.parquet", index=False)
        fit_fingerprint = row_set_fingerprint(fit_ids)
        heldout_fingerprint = row_set_fingerprint(heldout_ids)
        discovery = {
            "scope_id": f"outer_{fold:03d}_full_train",
            "fit_row_fingerprint": fit_fingerprint,
            "heldout_row_fingerprint": heldout_fingerprint,
            "topic_banks": _topic_metadata(),
            "artifacts": {
                "fit_topic_values": str(context_dir / "fit.npz"),
                "heldout_topic_values": str(context_dir / "heldout.npz"),
                "nuisance_predictions": str(context_dir / "nuisance.parquet"),
            },
        }
        contexts.append(
            {
                "schema_version": "multi_model_forest_handoff_v2",
                "stage1_config_hash": "fake-stage1",
                "fold_key": fold,
                "outer_fold": fold,
                "inner_fold": None,
                "scope": "full_outer_train",
                "fit_row_ids": fit_ids.tolist(),
                "heldout_row_ids": heldout_ids.tolist(),
                "fit_row_fingerprint": fit_fingerprint,
                "heldout_row_fingerprint": heldout_fingerprint,
                "discovery": discovery,
            }
        )
    handoff_path = tmp_path / "handoff.jsonl"
    handoff_path.write_text("".join(json.dumps(row) + "\n" for row in contexts))
    return handoff_path


def test_topic_score_forest_freezes_predictions_before_oracle_join(tmp_path):
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{index}" for index in range(12)],
            "clinical_text": [f"secret text {index}" for index in range(12)],
            "treatment_indicator": np.tile([0, 1], 6),
            "outcome_indicator": np.tile([0, 1, 1], 4),
            "true_ite_prob": np.linspace(-0.3, 0.3, 12),
            "true_unrelated_oracle": np.arange(12),
        }
    )
    handoff_path = _write_fake_handoff(tmp_path, dataset)
    config = TopicScoreForestConfig(
        n_estimators=4,
        inference=True,
        persist_fold_models=False,
    )
    _FakeForest.fit_shapes = []
    first_dir = tmp_path / "first"
    first = run_tfidf_topic_score_forest(
        dataset=dataset,
        handoff_path=handoff_path,
        output_dir=first_dir,
        config=config,
        forest_factory=_FakeForest,
    )
    frozen = pd.read_parquet(first_dir / "topic_score_predictions.parquet")
    assert not any(column.startswith("true_") for column in frozen.columns)
    assert len(frozen) == len(dataset)
    assert frozen["prediction_fitting_set_excludes_row_labels"].all()
    assert len(_FakeForest.fit_shapes) == 2
    assert all(
        x_shape[1] == 2 and w_shape[1] == 4 for x_shape, w_shape, _, _ in _FakeForest.fit_shapes
    )
    posthoc = json.loads((first_dir / "posthoc_oracle_metrics.json").read_text())
    assert posthoc["evaluation_is_post_hoc"] is True
    assert posthoc["frozen_prediction_sha256"] == first["prediction_sha256_before_oracle_join"]
    assert posthoc["overall"]["pearson_correlation"] is not None

    mutated = dataset.copy()
    mutated["true_ite_prob"] *= -7.0
    second_dir = tmp_path / "second"
    run_tfidf_topic_score_forest(
        dataset=mutated,
        handoff_path=handoff_path,
        output_dir=second_dir,
        config=config,
        forest_factory=_FakeForest,
    )
    second_frozen = pd.read_parquet(second_dir / "topic_score_predictions.parquet")
    pd.testing.assert_frame_equal(frozen, second_frozen)


def test_topic_score_forest_rejects_legacy_handoff(tmp_path):
    handoff = tmp_path / "legacy.jsonl"
    handoff.write_text(json.dumps({"schema_version": "multi_model_forest_handoff_v1"}) + "\n")
    dataset = pd.DataFrame({"treatment_indicator": [0, 1], "outcome_indicator": [0, 1]})
    with pytest.raises(ValueError, match="requires a v2"):
        run_tfidf_topic_score_forest(
            dataset=dataset,
            handoff_path=handoff,
            output_dir=tmp_path / "output",
            config=TopicScoreForestConfig(n_estimators=4),
            forest_factory=_FakeForest,
        )
