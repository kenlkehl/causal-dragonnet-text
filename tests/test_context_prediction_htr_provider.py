from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from oci.config import AppliedInferenceConfig
import oci.inference.context_prediction_htr_provider as module
from oci.inference.context_prediction_htr_provider import (
    HistoricalStage1ContextPredictionHTRProvider,
    context_prediction_fit_profile,
    context_prediction_htr_provider_identity,
)
from oci.inference.multi_model_forest_stage1 import MultiModelForestStage1HTRProvider

_PRESERVED_SOURCE_HASHES = {
    "multi_model_forest_stage1.py": (
        "ea74e85b31e33fe61cf19474477bdc7c6118143bd11c8534eb1e8bbe0b759f12"
    ),
    "multi_model_pair_uplift.py": (
        "0d72709578f58d4318da2dcbc57ab2bdaf76d154d3aca3b9d25d55cd6f949bed"
    ),
    "review_spent_evidence_provider.py": (
        "681cb3cbb26302e6acd4c42f1d8c023ce37e644b48a9766946d2493daa4e3d5c"
    ),
}


class _TinyExtractor(torch.nn.Module):
    output_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, texts):
        return self.weight * torch.ones((len(texts), self.output_dim))

    @staticmethod
    def fit_tokenizer(_texts) -> None:
        return None


class _FakeRunner:
    def __init__(self, config: AppliedInferenceConfig) -> None:
        self.config = config
        self.avf_config = config.architecture.agentic_attention_variable_forest
        self.device = torch.device("cpu")
        self.cleanup_calls = 0
        self.encoder_state_audits = 0
        self.optimizer_coverage_audits = 0

    @staticmethod
    def _fold_n_jobs(_folds: int) -> int:
        return 1

    @staticmethod
    def _create_extractor() -> _TinyExtractor:
        return _TinyExtractor()

    @staticmethod
    def _predict_effect_model(_model, frame: pd.DataFrame) -> np.ndarray:
        return 0.1 + 0.001 * frame["_oci_row_id"].to_numpy(dtype=float)

    def _cleanup_model(self, _model) -> None:
        self.cleanup_calls += 1

    def _assert_htr_sentence_encoder_training_state(self, _extractor) -> None:
        self.encoder_state_audits += 1

    def _assert_htr_sentence_encoder_optimizer_coverage(
        self,
        _extractor,
        _optimizer,
    ) -> None:
        self.optimizer_coverage_audits += 1

    @staticmethod
    def _effect_epochs() -> int:
        return 1

    @staticmethod
    def _make_text_loader(*_args, **_kwargs):
        return []


def _config() -> AppliedInferenceConfig:
    config = AppliedInferenceConfig()
    forest = config.architecture.multi_model_forest
    forest.htr_evidence_enabled = True
    forest.matched_pair_uplift_enabled = True
    forest.matched_pair_htr_enabled = True
    config.architecture.htr_freeze_sentence_encoder = False
    config.architecture.htr_require_live_unfrozen_encoder_attestation = True
    config.architecture.agentic_attention_variable_forest.nuisance_folds = 5
    config.architecture.agentic_attention_variable_forest.effect_folds = 5
    config.architecture.agentic_attention_variable_forest.fold_parallelism = "1"
    config.architecture.multi_model_forest.htr_fold_parallelism = "1"
    return config


def _frames(config: AppliedInferenceConfig):
    train = pd.DataFrame(
        {
            "_oci_row_id": np.arange(10, dtype=int),
            config.text_column: [f"context text {index}" for index in range(10)],
            config.treatment_column: np.asarray([0, 1] * 5, dtype=float),
            config.outcome_column: np.asarray([0, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float),
        }
    )
    test = pd.DataFrame(
        {
            "_oci_row_id": np.asarray([100, 101], dtype=int),
            config.text_column: ["prediction alpha", "prediction beta"],
        }
    )
    return train, test


def _provider(tmp_path: Path):
    config = _config()
    provider = HistoricalStage1ContextPredictionHTRProvider(
        config=config,
        output_dir=tmp_path,
        device="cpu",
        num_workers=1,
    )
    fake_runner = _FakeRunner(config)
    provider._runner = fake_runner
    return provider, fake_runner


def test_context_prediction_profile_reduces_five_fold_htr_from_20_to_8() -> None:
    profile = context_prediction_fit_profile(
        n_context_rows=100,
        nuisance_folds=5,
        effect_folds=5,
    )
    assert profile["legacy_inner_ensemble_model_attempts"] == 20
    assert profile["context_prediction_model_attempts"] == 8
    assert profile["model_attempt_reduction"] == 12

    identity = context_prediction_htr_provider_identity(_config(), device="cpu")
    assert identity["configured_legacy_model_attempts"] == 20
    assert identity["configured_context_prediction_model_attempts"] == 8
    assert identity["spent_discovery_path_changed"] is False
    assert identity["prediction_frame_labels_accepted"] is False
    assert identity["nuisance_fold_execution_policy"] == "required_serial"
    assert identity["multi_model_htr_fold_parallelism"] == "1"
    assert identity["effective_htr_runner_fold_parallelism"] == "1"
    assert identity["degenerate_pair_policy"] == "deterministic_zero_delta_without_model"
    assert identity["seed_root"] == module._SEED_ROOT
    assert identity["seed_component_offsets"] == dict(module._SEED_OFFSETS)

    nonserial = _config()
    nonserial.architecture.agentic_attention_variable_forest.fold_parallelism = "auto"
    with pytest.raises(ValueError, match="derived serial fold policy"):
        context_prediction_htr_provider_identity(nonserial, device="cpu")

    pair_disabled = _config()
    pair_disabled.architecture.multi_model_forest.matched_pair_htr_enabled = False
    with pytest.raises(ValueError, match="matched-pair fitting must be enabled"):
        context_prediction_htr_provider_identity(pair_disabled, device="cpu")


def test_seed_policy_constants_are_identity_bound_and_offsets_are_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _runner = _provider(tmp_path)
    with pytest.raises(TypeError):
        module._SEED_OFFSETS["nuisance"] = 999
    monkeypatch.setattr(module, "_SEED_ROOT", module._SEED_ROOT + 1)
    with pytest.raises(RuntimeError, match="configuration changed"):
        provider.identity()


def test_context_provider_call_graph_is_5_plus_1_plus_2_and_seals_train_placeholders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _runner = _provider(tmp_path)
    config = provider.config
    _runner.avf_config.r_stage_min_propensity = 0.35
    _runner.avf_config.r_stage_max_propensity = 0.65
    train, test = _frames(config)

    def fake_nuisance(
        _self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        outer_fold: int,
    ):
        train_predictions = pd.DataFrame(
            {
                "_oci_row_id": train_df["_oci_row_id"],
                "e_hat": np.linspace(0.3, 0.7, len(train_df)),
                "m_hat": np.linspace(0.4, 0.6, len(train_df)),
            }
        )
        test_predictions = pd.DataFrame(
            {
                "_oci_row_id": test_df["_oci_row_id"],
                "e_hat": np.asarray([0.45, 0.55]),
                "m_hat": np.asarray([0.48, 0.52]),
            }
        )
        return {
            "train": {"predictions": train_predictions, "attention": []},
            "test_predictions": test_predictions,
            "inner_model_rows": [
                {"inner_fold": fold, "objective": "nuisance"} for fold in range(1, 6)
            ],
        }

    monkeypatch.setattr(
        MultiModelForestStage1HTRProvider,
        "fit_nuisance_inner_ensemble_predict",
        fake_nuisance,
    )
    monkeypatch.setattr(
        module,
        "_train_complete_context_pair_model",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "_predict_htr_pair_delta",
        lambda *, pairs, **_kwargs: np.zeros(len(pairs), dtype=float),
    )
    monkeypatch.setattr(
        module,
        "_train_complete_context_effect_model",
        lambda **_kwargs: None,
    )

    nuisance = provider.fit_nuisance_inner_ensemble_predict(train, test, outer_fold=2)
    assert len(nuisance["inner_model_rows"]) == 5
    e_train = nuisance["train"]["predictions"]["e_hat"].to_numpy(dtype=float)
    m_train = nuisance["train"]["predictions"]["m_hat"].to_numpy(dtype=float)
    e_test = nuisance["test_predictions"]["e_hat"].to_numpy(dtype=float)
    m_test = nuisance["test_predictions"]["m_hat"].to_numpy(dtype=float)
    pair = provider.fit_pair_uplift_inner_ensemble_predict(
        train_df=train,
        test_df=test,
        texts_train=train[config.text_column].tolist(),
        texts_test=test[config.text_column].tolist(),
        y_train=train[config.outcome_column].to_numpy(dtype=float),
        t_train=train[config.treatment_column].to_numpy(dtype=float),
        e_train=e_train,
        m_train=m_train,
        e_test=e_test,
        m_test=m_test,
        outer_fold=2,
        propensity_caliper=0.05,
        outcome_caliper=0.05,
        max_controls_per_candidate=3,
        nearest_fallback_controls=1,
        max_attention_pairs=0,
    )
    nuisance_frame = pd.DataFrame(
        {"_oci_row_id": train["_oci_row_id"], "e_hat": e_train, "m_hat": m_train}
    )
    test_nuisance = pd.DataFrame(
        {"_oci_row_id": test["_oci_row_id"], "e_hat": e_test, "m_hat": m_test}
    )
    pseudo = provider.fit_effect_variant_inner_ensemble_predict(
        train,
        test,
        nuisance_frame,
        2,
        effect_objective="pseudo_outcome_mse",
        test_nuisance_predictions=test_nuisance,
    )
    squared = provider.fit_effect_variant_inner_ensemble_predict(
        train,
        test,
        nuisance_frame,
        2,
        effect_objective="squared_r_loss",
        test_nuisance_predictions=test_nuisance,
    )

    np.testing.assert_array_equal(pair.train_delta_logit, np.zeros(len(train)))
    np.testing.assert_array_equal(pair.train_pred_prob, np.zeros(len(train)))
    np.testing.assert_array_equal(
        pseudo["train"]["predictions"]["tau_hat_r_stage"],
        np.zeros(len(train)),
    )
    expected_eligible = (e_train >= 0.35) & (e_train <= 0.65)
    np.testing.assert_array_equal(
        pseudo["train"]["predictions"]["r_stage_train_eligible"],
        expected_eligible,
    )
    assert pseudo["inner_model_rows"][0]["train_rows"] == int(expected_eligible.sum())
    assert np.all(np.isfinite(pair.test_delta_logit))
    assert np.all(np.isfinite(pseudo["test_predictions"]["tau_hat_r_stage"]))
    assert np.all(np.isfinite(squared["test_predictions"]["tau_hat_r_stage"]))

    x_names = (
        "htr__matched_pair_uplift_delta_logit",
        "htr__matched_pair_treated_outcome_prob",
        "htr__effect_pseudo_target_pred",
        "htr__effect_weighted_r_tau_pred",
        "genuine_other_source",
    )
    x_train = np.column_stack(
        [
            pair.train_delta_logit,
            pair.train_pred_prob,
            pseudo["train"]["predictions"]["tau_hat_r_stage"],
            squared["train"]["predictions"]["tau_hat_r_stage"],
            np.arange(len(train), dtype=float),
        ]
    )
    x_test = np.column_stack(
        [
            pair.test_delta_logit,
            pair.test_pred_prob,
            pseudo["test_predictions"]["tau_hat_r_stage"],
            squared["test_predictions"]["tau_hat_r_stage"],
            np.asarray([np.nan, 8.0]),
        ]
    )
    bundle = SimpleNamespace(
        x_train=x_train,
        x_test=x_test,
        w_train=np.empty((len(train), 0)),
        w_test=np.empty((len(test), 0)),
        x_names=x_names,
        w_names=(),
        feature_rows=tuple(
            {"feature_name": name, "provenance": "inner_oof_train_outer_train_fit_test"}
            for name in x_names
        ),
    )
    sealed = provider.seal_prediction_only_bundle(bundle)
    audit = sealed.audit
    assert audit["legacy_inner_ensemble_model_attempts"] == 20
    assert audit["context_prediction_model_attempts"] == 8
    assert audit["observed_model_attempts"] == 8
    assert audit["train_matrices_retained"] is False
    assert audit["placeholder_columns_used_for_test_imputation"] is False
    assert not hasattr(sealed, "x_train")
    np.testing.assert_allclose(sealed.x_test[:, :4], x_test[:, :4])
    assert sealed.x_test[0, 4] == pytest.approx(4.5)
    for row in sealed.feature_rows[:4]:
        assert row["provenance"] == "complete_allowed_context_fit_label_free_prediction"
        assert row["train_values_exposed_or_consumed"] is False


def test_context_provider_rejects_any_prediction_label_or_uncommitted_effect_objective(
    tmp_path: Path,
) -> None:
    provider, _runner = _provider(tmp_path)
    config = provider.config
    train, test = _frames(config)
    labeled = test.assign(**{config.outcome_column: [0.0, 1.0]})
    with pytest.raises(ValueError, match="label-free prediction frame"):
        provider.fit_pair_uplift_inner_ensemble_predict(
            train_df=train,
            test_df=labeled,
            texts_train=train[config.text_column].tolist(),
            texts_test=labeled[config.text_column].tolist(),
            y_train=train[config.outcome_column].to_numpy(dtype=float),
            t_train=train[config.treatment_column].to_numpy(dtype=float),
            e_train=np.full(len(train), 0.5),
            m_train=np.full(len(train), 0.5),
            e_test=np.full(len(labeled), 0.5),
            m_test=np.full(len(labeled), 0.5),
            outer_fold=1,
            propensity_caliper=0.05,
            outcome_caliper=0.05,
            max_controls_per_candidate=1,
            nearest_fallback_controls=1,
            max_attention_pairs=0,
        )
    with pytest.raises(ValueError, match="exact normalized frame texts"):
        provider.fit_pair_uplift_inner_ensemble_predict(
            train_df=train,
            test_df=test,
            texts_train=train[config.text_column].tolist(),
            texts_test=["oracle encoded replacement", test[config.text_column].iloc[1]],
            y_train=train[config.outcome_column].to_numpy(dtype=float),
            t_train=train[config.treatment_column].to_numpy(dtype=float),
            e_train=np.full(len(train), 0.5),
            m_train=np.full(len(train), 0.5),
            e_test=np.full(len(test), 0.5),
            m_test=np.full(len(test), 0.5),
            outer_fold=1,
            propensity_caliper=0.05,
            outcome_caliper=0.05,
            max_controls_per_candidate=1,
            nearest_fallback_controls=1,
            max_attention_pairs=0,
        )
    with pytest.raises(ValueError, match="not precommitted"):
        provider.fit_effect_variant_inner_ensemble_predict(
            train,
            test,
            pd.DataFrame(
                {
                    "_oci_row_id": train["_oci_row_id"],
                    "e_hat": np.full(len(train), 0.5),
                    "m_hat": np.full(len(train), 0.5),
                }
            ),
            1,
            effect_objective="logistic_r_loss",
        )


def test_complete_context_pair_and_effect_helpers_audit_live_encoder_and_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    runner = _FakeRunner(config)
    pairs = pd.DataFrame(
        {
            "control_text": ["control a", "control b"],
            "treated_text": ["treated a", "treated b"],
            "label": [0, 1],
            "base_logit": [0.0, 0.0],
        }
    )
    pair_model = module._train_complete_context_pair_model(
        runner=runner,
        pairs=pairs,
        outer_fold=1,
    )
    assert pair_model is not None
    assert runner.encoder_state_audits == 2
    assert runner.optimizer_coverage_audits == 1

    monkeypatch.setattr(module, "_make_linear_lr_scheduler", lambda *_args, **_kwargs: None)
    effect_model = module._EffectNet(
        extractor=_TinyExtractor(),
        hidden_dim=4,
    )
    frame = pd.DataFrame(
        {
            "_oci_row_id": [0, 1],
            config.text_column: ["alpha", "beta"],
        }
    )
    values = np.asarray([0.0, 1.0])
    module._train_complete_context_effect_model(
        runner=runner,
        model=effect_model,
        train_df=frame,
        positions=np.asarray([0, 1], dtype=int),
        y=values,
        t=values,
        e_clipped=np.asarray([0.4, 0.6]),
        m_clipped=np.asarray([0.3, 0.7]),
        y_resid=np.asarray([-0.3, 0.3]),
        t_resid=np.asarray([-0.4, 0.4]),
    )
    assert runner.encoder_state_audits == 4
    assert runner.optimizer_coverage_audits == 2


def test_nonfinite_pair_or_effect_predictions_fail_before_placeholder_cleaning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_provider, _pair_runner = _provider(tmp_path / "pair")
    config = pair_provider.config
    train, test = _frames(config)
    monkeypatch.setattr(
        module,
        "_train_complete_context_pair_model",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "_predict_htr_pair_delta",
        lambda *, pairs, **_kwargs: np.full(len(pairs), np.nan),
    )
    with pytest.raises(ValueError, match="did not cover every prediction row"):
        pair_provider.fit_pair_uplift_inner_ensemble_predict(
            train_df=train,
            test_df=test,
            texts_train=train[config.text_column].tolist(),
            texts_test=test[config.text_column].tolist(),
            y_train=train[config.outcome_column].to_numpy(dtype=float),
            t_train=train[config.treatment_column].to_numpy(dtype=float),
            e_train=np.full(len(train), 0.5),
            m_train=np.full(len(train), 0.5),
            e_test=np.full(len(test), 0.5),
            m_test=np.full(len(test), 0.5),
            outer_fold=1,
            propensity_caliper=0.05,
            outcome_caliper=0.05,
            max_controls_per_candidate=1,
            nearest_fallback_controls=1,
            max_attention_pairs=0,
        )

    effect_provider, effect_runner = _provider(tmp_path / "effect")
    monkeypatch.setattr(
        module,
        "_train_complete_context_effect_model",
        lambda **_kwargs: None,
    )
    effect_runner._predict_effect_model = lambda _model, frame: np.full(len(frame), np.nan)
    nuisance = pd.DataFrame(
        {
            "_oci_row_id": train["_oci_row_id"],
            "e_hat": np.full(len(train), 0.5),
            "m_hat": np.full(len(train), 0.5),
        }
    )
    with pytest.raises(ValueError, match="prediction must be one finite vector"):
        effect_provider.fit_effect_variant_inner_ensemble_predict(
            train,
            test,
            nuisance,
            1,
            effect_objective="squared_r_loss",
        )


def test_degenerate_complete_context_pair_targets_use_honest_zero_delta_fallback(
    tmp_path: Path,
) -> None:
    provider, runner = _provider(tmp_path)
    config = provider.config
    train, test = _frames(config)
    outcomes = np.zeros(len(train), dtype=float)
    result = provider.fit_pair_uplift_inner_ensemble_predict(
        train_df=train,
        test_df=test,
        texts_train=train[config.text_column].tolist(),
        texts_test=test[config.text_column].tolist(),
        y_train=outcomes,
        t_train=train[config.treatment_column].to_numpy(dtype=float),
        e_train=np.full(len(train), 0.5),
        m_train=np.full(len(train), 0.5),
        e_test=np.full(len(test), 0.5),
        m_test=np.full(len(test), 0.5),
        outer_fold=3,
        propensity_caliper=0.05,
        outcome_caliper=0.05,
        max_controls_per_candidate=1,
        nearest_fallback_controls=1,
        max_attention_pairs=0,
    )
    np.testing.assert_array_equal(result.test_delta_logit, np.zeros(len(test)))
    np.testing.assert_allclose(result.test_pred_prob, np.full(len(test), 0.5))
    assert result.evidence_rows[0]["htr_model_fit_attempts"] == 1
    assert result.evidence_rows[0]["htr_models_fit"] == 0
    assert runner.encoder_state_audits == 0
    assert runner.optimizer_coverage_audits == 0


def test_context_optimization_preserves_spent_discovery_source_bytes() -> None:
    inference_dir = Path(module.__file__).resolve().parent
    for filename, expected in _PRESERVED_SOURCE_HASHES.items():
        observed = hashlib.sha256((inference_dir / filename).read_bytes()).hexdigest()
        assert observed == expected, filename
