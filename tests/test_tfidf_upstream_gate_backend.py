from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from oci.config import AppliedInferenceConfig
from oci.inference.all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
import oci.inference.tfidf_upstream_gate_backend as module
from oci.inference.tfidf_safe_artifacts import write_named_array_bank


def test_tfidf_backend_transforms_label_free_gate_into_role_aware_banks(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    config = AppliedInferenceConfig()
    forest = config.architecture.multi_model_forest
    forest.bow_views = forest.bow_views[:1]
    forest.tfidf_topic.topic_count = 2
    snapshot = SimpleNamespace(
        source_path=config_path.resolve(),
        sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        applied_config=lambda: copy.deepcopy(config),
        verify_source=lambda: None,
    )
    monkeypatch.setattr(
        module,
        "_historical_stage1_config_snapshot",
        lambda _path, _snapshot=None: snapshot,
    )

    def fake_fit(**kwargs):
        assert list(kwargs["heldout_df"].columns) == ["_oci_row_id", "clinical_text"]
        output = Path(kwargs["artifact_dir"])
        output.mkdir(parents=True, exist_ok=True)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(
            ["high torque bearing vibration", "low torque stable output"]
        )
        fitted_root = output / "fitted_context"
        fitted_root.mkdir()
        fitted_path = fitted_root / "index.json"
        fitted_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            module,
            "load_fitted_topic_context",
            lambda _path: SimpleNamespace(common_vectorizer=vectorizer),
        )
        scores_path = output / "effect_ngram_scores.parquet"
        pd.DataFrame(
            {
                "feature": ["bearing vibration", "stable output"],
                "eligible": [True, True],
                "combined_importance": [4.0, 1.0],
                "support_control": [5, 5],
                "support_treated": [5, 5],
            }
        ).to_parquet(scores_path, index=False)
        topic_path = write_named_array_bank(
            {
                "treatment": np.asarray([[0.1], [0.2]]),
                "outcome": np.asarray([[0.3], [0.4]]),
                "effect": np.asarray([[0.5], [0.6]]),
            },
            output / "heldout_topic_values",
            row_count=2,
        )
        nuisance_path = output / "nuisance.parquet"
        pd.DataFrame(
            {
                "_oci_row_id": [8, 9],
                "prediction_scope": ["external_heldout", "external_heldout"],
                "treatment_stacked": [0.25, 0.75],
                "outcome_stacked": [0.4, 0.6],
            }
        ).to_parquet(nuisance_path, index=False)
        return {
            "topic_banks": {
                "effect": {
                    "topics": [
                        {"terms": [{"term": "stable output"}]},
                    ]
                }
            },
            "artifacts": {
                "fitted_context": str(fitted_path),
                "ngram_scores": {"effect": str(scores_path)},
                "heldout_topic_values": str(topic_path),
                "nuisance_predictions": str(nuisance_path),
            },
        }

    monkeypatch.setattr(module, "fit_tfidf_topic_context", fake_fit)
    backend = module.TfidfTopicOrphanContextBackend(
        stage1_config_path=config_path,
        max_orphan_features=2,
    )
    prediction = backend.fit_predict(
        outer_fold=1,
        context_row_ids=(1, 2, 3, 4),
        context_texts=("a", "b", "c", "d"),
        context_treatment=np.asarray([0.0, 1.0, 0.0, 1.0]),
        context_outcome=np.asarray([0.0, 0.0, 1.0, 1.0]),
        gate_row_ids=(8, 9),
        gate_texts=("bearing vibration", "stable output"),
        work_dir=tmp_path / "work",
    )

    assert prediction.calibrated_source_names == ()
    assert prediction.feature_values.shape == (2, 6)
    assert prediction.feature_roles[:2] == (
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
    )
    assert prediction.feature_roles[2:5] == (
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    )
    assert prediction.feature_roles[-1] == UNCALIBRATED_EFFECT_MODIFIER_ROLE
    assert prediction.feature_kinds[-1] == "tfidf_orphan_ngrams"


def test_orphan_capacity_is_nullable_complete_and_finite_binding_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(
        ["brain metastases durable response", "brain response"]
    )
    fitted_path = tmp_path / "fitted.json"
    fitted_path.write_text("{}", encoding="utf-8")
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        {
            "feature": ["brain metastases", "durable response"],
            "eligible": [True, True],
            "combined_importance": [4.0, 3.0],
            "support_control": [5, 5],
            "support_treated": [5, 5],
        }
    ).to_parquet(scores_path, index=False)
    metadata = {
        "topic_banks": {"effect": {"topics": []}},
        "artifacts": {
            "fitted_context": str(fitted_path),
            "ngram_scores": {"effect": str(scores_path)},
        },
    }
    monkeypatch.setattr(
        module,
        "load_fitted_topic_context",
        lambda _path: SimpleNamespace(common_vectorizer=vectorizer),
    )
    backend = object.__new__(module.TfidfTopicOrphanContextBackend)
    backend.minimum_orphan_arm_support = 2
    backend.max_orphan_features = None

    names, values = backend._orphan_values(
        metadata=metadata,
        gate_texts=("brain metastases durable response",),
    )
    assert len(names) == 2
    assert values.shape == (1, 2)

    backend.max_orphan_features = 1
    with pytest.raises(
        module.TfidfOrphanFeatureCapacityOverflowError,
        match="refusing silent orphan-feature omission",
    ):
        backend._orphan_values(
            metadata=metadata,
            gate_texts=("brain metastases durable response",),
        )
