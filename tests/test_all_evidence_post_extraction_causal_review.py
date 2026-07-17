from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest

import oci.inference.all_evidence_post_extraction_review as review_module

from oci.inference.all_evidence_post_extraction_review import (
    CausalReviewConfig,
    GateFeatureBankView,
    GateSourceSignalView,
    ObservableCausalRows,
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    build_causal_review_diagnostics,
    evaluate_untouched_gate_acceptance,
)
from oci.inference.fold_honest_r_stack import FitRowProvenance


def _continuous(name: str, role: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "continuous",
        "roles": [role],
        "description": f"Baseline {name} measured before treatment initiation.",
    }


def _frame(c: np.ndarray, z: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "explicit_feat_baseline_risk": c,
            "explicit_feat_baseline_risk_missing": np.zeros(len(c), dtype=bool),
            "explicit_feat_baseline_marker": z,
            "explicit_feat_baseline_marker_missing": np.zeros(len(c), dtype=bool),
            "explicit_feat_marker_duplicate": z,
            "explicit_feat_marker_duplicate_missing": np.zeros(len(c), dtype=bool),
        }
    )


def _observed_rows(seed: int = 13) -> tuple[ObservableCausalRows, ObservableCausalRows, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_context = 90
    n_gate = 36
    c = rng.normal(size=n_context + n_gate)
    z = rng.normal(size=n_context + n_gate)
    propensity = 1.0 / (1.0 + np.exp(-0.9 * c))
    treatment = rng.binomial(1, propensity).astype(float)
    effect = 0.35 + 0.9 * z
    outcome = 0.8 * c + treatment * effect + rng.normal(scale=0.35, size=len(c))
    context_ids = tuple(range(10_000, 10_000 + n_context))
    gate_ids = tuple(range(20_000, 20_000 + n_gate))
    context = ObservableCausalRows(
        row_ids=context_ids,
        extracted=_frame(c[:n_context], z[:n_context]),
        treatment=treatment[:n_context],
        outcome=outcome[:n_context],
        inner_fold_ids=tuple(position % 3 for position in range(n_context)),
    )
    gate = ObservableCausalRows(
        row_ids=gate_ids,
        extracted=_frame(c[n_context:], z[n_context:]),
        treatment=treatment[n_context:],
        outcome=outcome[n_context:],
    )
    return context, gate, effect[n_context:]


def _all_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value] + [
            nested for item in value.values() for nested in _all_keys(item)
        ]
    if isinstance(value, list):
        return [nested for item in value for nested in _all_keys(item)]
    return []


def _vector_with_exact_correlation(reference: np.ndarray, correlation: float) -> np.ndarray:
    centered = np.asarray(reference, dtype=float) - float(np.mean(reference))
    reference_unit = centered / np.linalg.norm(centered)
    orthogonal = np.square(reference_unit)
    orthogonal -= float(np.mean(orthogonal))
    orthogonal -= float(np.dot(orthogonal, reference_unit)) * reference_unit
    orthogonal /= np.linalg.norm(orthogonal)
    return float(correlation) * reference_unit + np.sqrt(1.0 - float(correlation) ** 2) * orthogonal


def _context_oof_lineage(context: ObservableCausalRows) -> tuple[FitRowProvenance, ...]:
    assert context.inner_fold_ids is not None
    return tuple(
        FitRowProvenance(
            fit_row_ids=frozenset(
                candidate
                for candidate, candidate_fold in zip(context.row_ids, context.inner_fold_ids)
                if candidate_fold != fold_id
            )
        )
        for fold_id in context.inner_fold_ids
    )


def _conditional_context_kwargs(
    context: ObservableCausalRows,
    values: np.ndarray,
) -> dict[str, Any]:
    matrix = np.asarray(values, dtype=float)
    assert matrix.ndim == 2 and len(matrix) == len(context.row_ids)
    lineage = _context_oof_lineage(context)
    return {
        "context_row_ids": context.row_ids,
        "context_inner_fold_ids": context.inner_fold_ids,
        "context_values": matrix,
        "context_fit_row_provenance": tuple(lineage for _ in range(matrix.shape[1])),
    }


def test_nested_diagnostics_cover_nuisance_r_loss_stability_and_ablations():
    context, _gate, _effect = _observed_rows()
    specs = [
        _continuous("baseline_risk", "confounder"),
        _continuous("baseline_marker", "effect_modifier"),
    ]
    result = build_causal_review_diagnostics(context, specs, diagnostic_start=40)

    assert result["fixed_inner_fold_count"] == 3
    assert result["metrics"]["nuisance"]["treatment"]["log_loss"] is not None
    assert result["metrics"]["nuisance"]["treatment"]["brier"] is not None
    assert result["metrics"]["nuisance"]["outcome"]["kind"] == "continuous"
    assert result["metrics"]["nuisance"]["outcome"]["rmse"] is not None
    effect = result["metrics"]["effect"]
    assert effect["weighted_r_loss"] >= 0.0
    assert effect["zero_effect_r_loss"] >= 0.0
    assert np.isfinite(effect["r_loss_ratio"])
    assert len(result["inner_fold_stability"]["per_fold"]) == 3
    assert result["inner_fold_stability"]["summary"]["r_loss_ratio"]["std"] >= 0.0
    assert {row["contract_name"] for row in result["contract_ablations"]} == {
        "baseline_risk",
        "baseline_marker",
    }
    marker_ablation = next(
        row for row in result["contract_ablations"] if row["contract_name"] == "baseline_marker"
    )
    assert marker_ablation["weighted_r_loss_delta_when_removed"] > 0.0
    assert marker_ablation["encoded_columns_removed"] == 2
    assert not any("oracle" in key.lower() or "true" in key.lower() for key in _all_keys(result))


def test_gate_outcome_perturbation_cannot_change_preproposal_diagnostics():
    context, gate, _effect = _observed_rows()
    specs = [
        _continuous("baseline_risk", "confounder"),
        _continuous("baseline_marker", "effect_modifier"),
    ]
    before = build_causal_review_diagnostics(context, specs)
    _perturbed_gate = ObservableCausalRows(
        row_ids=gate.row_ids,
        extracted=gate.extracted,
        treatment=gate.treatment,
        outcome=np.asarray(gate.outcome) + np.linspace(-100.0, 100.0, len(gate.row_ids)),
    )
    after = build_causal_review_diagnostics(context, specs)
    assert before == after
    assert before["diagnostic_sha256"] == after["diagnostic_sha256"]


def test_binary_outcome_nuisance_metrics_are_reported_without_gate_data():
    context, _gate, _effect = _observed_rows()
    binary_context = ObservableCausalRows(
        row_ids=context.row_ids,
        extracted=context.extracted,
        treatment=context.treatment,
        outcome=(np.asarray(context.outcome) > np.median(context.outcome)).astype(float),
        inner_fold_ids=context.inner_fold_ids,
    )
    result = build_causal_review_diagnostics(
        binary_context,
        [
            _continuous("baseline_risk", "confounder"),
            _continuous("baseline_marker", "effect_modifier"),
        ],
    )
    outcome = result["metrics"]["nuisance"]["outcome"]
    assert outcome["kind"] == "binary"
    assert outcome["auroc"] is not None
    assert outcome["brier"] is not None
    assert outcome["log_loss"] is not None
    assert outcome["rmse"] is None


def test_gate_source_provenance_must_exclude_the_entire_gate():
    context, gate, effect = _observed_rows()
    safe = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
    view = GateSourceSignalView(
        row_ids=gate.row_ids,
        source_names=("neural_query_effect",),
        source_kinds=("nested_neural_query_weighted_r",),
        values=effect[:, None],
        fit_row_provenance=(safe,),
    )
    assert view.aligned_values(tuple(reversed(gate.row_ids))).shape == (len(gate.row_ids), 1)

    unsafe = FitRowProvenance(fit_row_ids=frozenset({context.row_ids[0], gate.row_ids[-1]}))
    with pytest.raises(ValueError, match="entire|lineage includes gate rows"):
        GateSourceSignalView(
            row_ids=gate.row_ids,
            source_names=("neural_query_effect",),
            source_kinds=("nested_neural_query_weighted_r",),
            values=effect[:, None],
            fit_row_provenance=(unsafe,),
        )


def test_gate_conditions_on_calibrated_sources_and_reports_preservation():
    context, gate, effect = _observed_rows()
    specs = [
        _continuous("baseline_risk", "confounder"),
        _continuous("baseline_marker", "effect_modifier"),
    ]
    lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
    context_source_values = np.column_stack(
        [
            context.extracted["explicit_feat_baseline_marker"].to_numpy(),
            context.extracted["explicit_feat_baseline_marker"].to_numpy() + 0.03,
        ]
    )
    source = GateSourceSignalView(
        row_ids=gate.row_ids,
        source_names=("bow_effect", "neural_query_effect"),
        source_kinds=("bow_weighted_r", "nested_neural_query_weighted_r"),
        values=np.column_stack([effect, effect + 0.03]),
        fit_row_provenance=(lineage, lineage),
        **_conditional_context_kwargs(context, context_source_values),
    )
    changed_source = GateSourceSignalView(
        row_ids=gate.row_ids,
        source_names=source.source_names,
        source_kinds=source.source_kinds,
        values=np.column_stack([-50.0 * effect, np.linspace(-20.0, 20.0, len(effect))]),
        fit_row_provenance=(lineage, lineage),
        **_conditional_context_kwargs(context, context_source_values),
    )
    first = evaluate_untouched_gate_acceptance(
        context,
        gate,
        specs,
        specs,
        source_view=source,
    )
    second = evaluate_untouched_gate_acceptance(
        context,
        gate,
        specs,
        specs,
        source_view=changed_source,
    )
    assert first.current["metrics"] != second.current["metrics"]
    source_metrics = first.current["source_signal_evaluation"]
    assert source_metrics["source_preservation_score"] is not None
    assert len(source_metrics["sources"]) == 2
    assert all(row["contextual_weighted_r_loss"] >= 0.0 for row in source_metrics["sources"])
    assert all(
        row["contextual_weighted_r_loss_delta_vs_zero_effect"]
        == pytest.approx(row["contextual_weighted_r_loss"] - row["zero_effect_weighted_r_loss"])
        for row in source_metrics["sources"]
    )
    assert "source_preservation" in first.guards
    assert "source_direction_preservation" in first.guards
    assert "source_context_r_loss" in first.guards
    assert source_metrics["calibrated_sources_used_as_effect_regression_covariates"] is True
    assert (
        first.guards["conditional_upstream_design"][
            "calibrated_sources_routed_to_effect_regression"
        ]
        is True
    )
    calibrated_ablations = first.current["upstream_predictive_family_ablations"]
    assert len(calibrated_ablations) == 2
    assert all(
        row["input_kind"] == "calibrated_effect_source"
        and row["ablation_refit_performed"]
        and row["weighted_r_loss_delta_when_removed"] is not None
        for row in calibrated_ablations
    )

    with pytest.raises(ValueError, match="GateFeatureBankView"):
        GateSourceSignalView(
            row_ids=gate.row_ids,
            source_names=("raw_neural_query_effect",),
            source_kinds=("neural_query_moments",),
            values=effect[:, None],
            fit_row_provenance=(lineage,),
        )


def test_calibrated_source_sign_reversal_fails_direction_guard(monkeypatch):
    context, gate, _effect = _observed_rows()
    specs = [
        _continuous("baseline_risk", "confounder"),
        _continuous("baseline_marker", "effect_modifier"),
    ]
    config = CausalReviewConfig()
    current = review_module._fit_context_predict_gate(
        context,
        gate,
        specs,
        config=config,
    )
    metrics, e_hat, m_hat, tau_hat = current
    lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
    source = GateSourceSignalView(
        row_ids=gate.row_ids,
        source_names=("calibrated_effect",),
        source_kinds=("nested_calibrated_effect",),
        values=np.asarray(tau_hat)[:, None],
        fit_row_provenance=(lineage,),
        **_conditional_context_kwargs(
            context,
            context.extracted[["explicit_feat_baseline_marker"]].to_numpy(),
        ),
    )
    calls = iter(
        [
            (metrics, e_hat, m_hat, tau_hat),
            (metrics, e_hat, m_hat, -np.asarray(tau_hat)),
        ]
    )
    monkeypatch.setattr(
        review_module,
        "_fit_context_predict_gate",
        lambda *args, **kwargs: next(calls),
    )
    monkeypatch.setattr(
        review_module,
        "_predictive_upstream_family_ablations",
        lambda *args, **kwargs: ([], []),
    )

    decision = evaluate_untouched_gate_acceptance(
        context,
        gate,
        specs,
        specs,
        source_view=source,
        config=config,
    )
    assert decision.accepted is False
    assert "source_direction_guard_failed" in decision.reasons
    guard = decision.guards["source_direction_preservation"]
    assert guard["passed"] is False
    source_guard = next(iter(guard["by_source"].values()))
    assert source_guard["current_signed_correlation"] == pytest.approx(1.0)
    assert source_guard["candidate_signed_correlation"] == pytest.approx(-1.0)
    assert source_guard["same_direction"] is False


@pytest.mark.parametrize(
    (
        "candidate_correlation",
        "aggregate_preservation_passed",
        "direction_preservation_passed",
    ),
    [
        (-0.10, False, False),
        (0.80, True, False),
        (-0.76, True, True),
        (-0.90, True, True),
    ],
    ids=(
        "negative_magnitude_collapse",
        "sign_reversal",
        "similar_negative_signal",
        "stronger_negative_signal",
    ),
)
def test_calibrated_negative_source_preservation_uses_absolute_magnitude_and_sign(
    monkeypatch,
    candidate_correlation,
    aggregate_preservation_passed,
    direction_preservation_passed,
):
    context, gate, _effect = _observed_rows()
    specs = [
        _continuous("baseline_risk", "confounder"),
        _continuous("baseline_marker", "effect_modifier"),
    ]
    config = CausalReviewConfig(source_preservation_tolerance=0.05)
    metrics, e_hat, m_hat, _tau_hat = review_module._fit_context_predict_gate(
        context,
        gate,
        specs,
        config=config,
    )
    source_values = np.linspace(-1.0, 1.0, len(gate.row_ids))
    current_tau = _vector_with_exact_correlation(source_values, -0.80)
    candidate_tau = _vector_with_exact_correlation(
        source_values,
        candidate_correlation,
    )
    lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
    source = GateSourceSignalView(
        row_ids=gate.row_ids,
        source_names=("calibrated_effect",),
        source_kinds=("nested_calibrated_effect",),
        values=source_values[:, None],
        fit_row_provenance=(lineage,),
        **_conditional_context_kwargs(
            context,
            context.extracted[["explicit_feat_baseline_marker"]].to_numpy(),
        ),
    )
    calls = iter(
        [
            (metrics, e_hat, m_hat, current_tau),
            (metrics, e_hat, m_hat, candidate_tau),
        ]
    )
    monkeypatch.setattr(
        review_module,
        "_fit_context_predict_gate",
        lambda *args, **kwargs: next(calls),
    )
    monkeypatch.setattr(
        review_module,
        "_predictive_upstream_family_ablations",
        lambda *args, **kwargs: ([], []),
    )

    decision = evaluate_untouched_gate_acceptance(
        context,
        gate,
        specs,
        specs,
        source_view=source,
        config=config,
    )

    aggregate_guard = decision.guards["source_preservation"]
    assert aggregate_guard["passed"] is aggregate_preservation_passed
    assert aggregate_guard["correlation_measure"] == "mean_absolute_source_correlation"
    assert aggregate_guard["current_score"] == pytest.approx(0.80)
    assert aggregate_guard["candidate_score"] == pytest.approx(abs(candidate_correlation))
    assert aggregate_guard["minimum_candidate_score"] == pytest.approx(0.75)
    assert decision.current["source_signal_evaluation"][
        "source_preservation_score"
    ] == pytest.approx(0.80)
    assert decision.current["source_signal_evaluation"][
        "mean_signed_source_correlation"
    ] == pytest.approx(-0.80)
    assert ("source_preservation_guard_failed" in decision.reasons) is (
        not aggregate_preservation_passed
    )

    direction_guard = decision.guards["source_direction_preservation"]
    assert direction_guard["passed"] is direction_preservation_passed
    source_guard = next(iter(direction_guard["by_source"].values()))
    assert source_guard["current_signed_correlation"] == pytest.approx(-0.80)
    assert source_guard["candidate_signed_correlation"] == pytest.approx(candidate_correlation)
    assert source_guard["current_absolute_correlation"] == pytest.approx(0.80)
    assert source_guard["candidate_absolute_correlation"] == pytest.approx(
        abs(candidate_correlation)
    )
    assert source_guard["minimum_candidate_absolute_correlation"] == pytest.approx(0.75)
    assert source_guard["magnitude_preserved"] is (abs(candidate_correlation) >= 0.75)
    assert source_guard["same_direction"] is (candidate_correlation < 0.0)
    assert ("source_direction_guard_failed" in decision.reasons) is (
        not direction_preservation_passed
    )


def test_raw_neural_query_banks_use_only_role_matched_preservation():
    context, gate, _effect = _observed_rows()
    specs = [
        _continuous("baseline_risk", "confounder"),
        _continuous("baseline_marker", "effect_modifier"),
    ]
    lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
    values = np.column_stack(
        [
            gate.extracted["explicit_feat_baseline_risk"].to_numpy(),
            gate.extracted["explicit_feat_baseline_risk"].to_numpy() + 0.1,
            gate.extracted["explicit_feat_baseline_marker"].to_numpy(),
        ]
    )
    bank = GateFeatureBankView(
        row_ids=gate.row_ids,
        feature_names=(
            "neural_query_treatment_0001",
            "neural_query_outcome_0001",
            "neural_query_effect_0001",
        ),
        source_kinds=(
            "neural_query_treatment_moments",
            "neural_query_outcome_moments",
            "neural_query_effect_moments",
        ),
        consumer_roles=(
            PROPENSITY_NUISANCE_FEATURE_ROLE,
            OUTCOME_NUISANCE_FEATURE_ROLE,
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        ),
        values=values,
        fit_row_provenance=(lineage, lineage, lineage),
        **_conditional_context_kwargs(
            context,
            np.column_stack(
                [
                    context.extracted["explicit_feat_baseline_risk"].to_numpy(),
                    context.extracted["explicit_feat_baseline_risk"].to_numpy() + 0.1,
                    context.extracted["explicit_feat_baseline_marker"].to_numpy(),
                ]
            ),
        ),
    )
    decision = evaluate_untouched_gate_acceptance(
        context,
        gate,
        specs,
        specs,
        feature_bank_view=bank,
    )
    evaluation = decision.current["feature_bank_evaluation"]
    assert len(evaluation["features"]) == 3
    assert evaluation["raw_feature_values_used_as_treatment_effects"] is False
    assert evaluation["raw_feature_values_used_as_model_inputs"] is True
    assert set(evaluation["preservation_score_by_consumer_role"]) == {
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    }
    family_rows = evaluation["preservation_by_source_kind_and_consumer_role"]
    assert {(row["source_kind"], row["consumer_role"]) for row in family_rows} == {
        (
            "neural_query_treatment_moments",
            PROPENSITY_NUISANCE_FEATURE_ROLE,
        ),
        (
            "neural_query_outcome_moments",
            OUTCOME_NUISANCE_FEATURE_ROLE,
        ),
        (
            "neural_query_effect_moments",
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        ),
    }
    assert all(row["feature_count"] == 1 for row in family_rows)
    family_guard = decision.guards["feature_bank_preservation"]
    assert family_guard["family_identities_match"] is True
    assert {
        (row["source_kind"], row["consumer_role"])
        for row in family_guard["by_source_kind_and_consumer_role"]
    } == {
        (
            "neural_query_treatment_moments",
            PROPENSITY_NUISANCE_FEATURE_ROLE,
        ),
        (
            "neural_query_outcome_moments",
            OUTCOME_NUISANCE_FEATURE_ROLE,
        ),
        (
            "neural_query_effect_moments",
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        ),
    }
    assert not any("contextual_r_loss" in key for key in _all_keys(evaluation))
    assert decision.guards["feature_bank_preservation"]["passed"]
    predictive = decision.current["upstream_predictive_family_ablations"]
    assert len(predictive) == 3
    assert all(row["ablation_refit_performed"] for row in predictive)
    assert all(not row["raw_feature_value_used_directly_as_tau"] for row in predictive)
    assert (
        decision.guards["upstream_predictive_family_ablations"][
            "correlation_deletion_used_as_predictive_ablation"
        ]
        is False
    )

    with pytest.raises(ValueError, match="consumer_roles"):
        GateFeatureBankView(
            row_ids=gate.row_ids,
            feature_names=("neural_query_effect_0001",),
            source_kinds=("neural_query_moments",),
            consumer_roles=("calibrated_treatment_effect",),
            values=values[:, :1],
            fit_row_provenance=(lineage,),
        )


def test_raw_feature_family_guard_prevents_within_role_compensation(monkeypatch):
    context, gate, _effect = _observed_rows()
    specs = [
        _continuous("baseline_risk", "confounder"),
        _continuous("baseline_marker", "effect_modifier"),
    ]
    config = CausalReviewConfig(feature_bank_preservation_tolerance=0.05)
    metrics, _e_hat, m_hat, tau_hat = review_module._fit_context_predict_gate(
        context,
        gate,
        specs,
        config=config,
    )
    phase = np.arange(len(gate.row_ids), dtype=float)
    family_a = np.sin(2.0 * np.pi * phase / len(phase))
    family_b = np.cos(2.0 * np.pi * phase / len(phase))
    current_e = 0.5 + 0.25 * family_a
    candidate_e = 0.5 + 0.25 * family_b
    lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
    bank = GateFeatureBankView(
        row_ids=gate.row_ids,
        feature_names=("family_a_feature", "family_b_feature"),
        source_kinds=("family_a", "family_b"),
        consumer_roles=(
            PROPENSITY_NUISANCE_FEATURE_ROLE,
            PROPENSITY_NUISANCE_FEATURE_ROLE,
        ),
        values=np.column_stack([family_a, family_b]),
        fit_row_provenance=(lineage, lineage),
        **_conditional_context_kwargs(
            context,
            np.column_stack(
                [
                    context.extracted["explicit_feat_baseline_risk"].to_numpy(),
                    context.extracted["explicit_feat_baseline_marker"].to_numpy(),
                ]
            ),
        ),
    )
    calls = iter(
        [
            (metrics, current_e, m_hat, tau_hat),
            (metrics, candidate_e, m_hat, tau_hat),
        ]
    )
    monkeypatch.setattr(
        review_module,
        "_fit_context_predict_gate",
        lambda *args, **kwargs: next(calls),
    )
    monkeypatch.setattr(
        review_module,
        "_predictive_upstream_family_ablations",
        lambda *args, **kwargs: ([], []),
    )

    decision = evaluate_untouched_gate_acceptance(
        context,
        gate,
        specs,
        specs,
        feature_bank_view=bank,
        config=config,
    )

    guard = decision.guards["feature_bank_preservation"]
    assert guard["by_consumer_role"][PROPENSITY_NUISANCE_FEATURE_ROLE]["passed"]
    assert guard["family_identities_match"] is True
    by_kind = {row["source_kind"]: row for row in guard["by_source_kind_and_consumer_role"]}
    assert by_kind["family_a"]["current_preservation_score"] == pytest.approx(1.0)
    assert by_kind["family_a"]["candidate_preservation_score"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert by_kind["family_a"]["passed"] is False
    assert by_kind["family_b"]["passed"] is True
    assert guard["passed"] is False
    assert decision.accepted is False
    assert "feature_bank_family_preservation_guard_failed" in decision.reasons


def test_raw_feature_family_guard_requires_exact_family_identity_sets(monkeypatch):
    context, gate, _effect = _observed_rows()
    specs = [
        _continuous("baseline_risk", "confounder"),
        _continuous("baseline_marker", "effect_modifier"),
    ]
    config = CausalReviewConfig()
    fitted = review_module._fit_context_predict_gate(
        context,
        gate,
        specs,
        config=config,
    )
    lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
    bank = GateFeatureBankView(
        row_ids=gate.row_ids,
        feature_names=("upstream_feature",),
        source_kinds=("family_a",),
        consumer_roles=(PROPENSITY_NUISANCE_FEATURE_ROLE,),
        values=np.arange(len(gate.row_ids), dtype=float)[:, None],
        fit_row_provenance=(lineage,),
        **_conditional_context_kwargs(
            context,
            context.extracted[["explicit_feat_baseline_risk"]].to_numpy(),
        ),
    )
    monkeypatch.setattr(
        review_module,
        "_fit_context_predict_gate",
        lambda *args, **kwargs: fitted,
    )
    monkeypatch.setattr(
        review_module,
        "_predictive_upstream_family_ablations",
        lambda *args, **kwargs: ([], []),
    )

    def evaluation(kind):
        return {
            "features": [],
            "preservation_score_by_consumer_role": {
                PROPENSITY_NUISANCE_FEATURE_ROLE: 0.8,
                OUTCOME_NUISANCE_FEATURE_ROLE: None,
                UNCALIBRATED_EFFECT_MODIFIER_ROLE: None,
            },
            "preservation_by_source_kind_and_consumer_role": [
                {
                    "source_kind": kind,
                    "consumer_role": PROPENSITY_NUISANCE_FEATURE_ROLE,
                    "feature_count": 1,
                    "mean_absolute_role_matched_prediction_correlation": 0.8,
                }
            ],
        }

    evaluations = iter([evaluation("family_a"), evaluation("family_b")])
    monkeypatch.setattr(
        review_module,
        "_gate_feature_bank_metrics",
        lambda *args, **kwargs: next(evaluations),
    )

    decision = evaluate_untouched_gate_acceptance(
        context,
        gate,
        specs,
        specs,
        feature_bank_view=bank,
        config=config,
    )

    guard = decision.guards["feature_bank_preservation"]
    assert guard["family_identities_match"] is False
    assert guard["passed"] is False
    assert decision.accepted is False
    assert "feature_bank_family_preservation_guard_failed" in decision.reasons


def test_raw_feature_bank_provenance_excludes_every_gate_row():
    context, gate, _effect = _observed_rows()
    unsafe = FitRowProvenance(fit_row_ids=frozenset({context.row_ids[0], gate.row_ids[1]}))
    with pytest.raises(ValueError, match="lineage includes gate rows"):
        GateFeatureBankView(
            row_ids=gate.row_ids,
            feature_names=("neural_query_treatment_0001",),
            source_kinds=("neural_query_moments",),
            consumer_roles=(PROPENSITY_NUISANCE_FEATURE_ROLE,),
            values=np.zeros((len(gate.row_ids), 1)),
            fit_row_provenance=(unsafe,),
        )


def test_gate_only_legacy_feature_view_fails_closed_for_adaptive_acceptance():
    context, gate, _effect = _observed_rows()
    legacy = GateFeatureBankView(
        row_ids=gate.row_ids,
        feature_names=("legacy_gate_only",),
        source_kinds=("neural_query_moments",),
        consumer_roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,),
        values=np.zeros((len(gate.row_ids), 1)),
        fit_row_provenance=(FitRowProvenance(fit_row_ids=frozenset(context.row_ids)),),
    )
    with pytest.raises(ValueError, match="lacks required cross-fitted context-side"):
        evaluate_untouched_gate_acceptance(
            context,
            gate,
            [
                _continuous("baseline_risk", "confounder"),
                _continuous("baseline_marker", "effect_modifier"),
            ],
            [
                _continuous("baseline_risk", "confounder"),
                _continuous("baseline_marker", "effect_modifier"),
            ],
            feature_bank_view=legacy,
        )


def test_context_upstream_lineage_must_equal_exact_fold_complement():
    context, gate, _effect = _observed_rows()
    gate_lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
    assert context.inner_fold_ids is not None
    incomplete = tuple(
        FitRowProvenance(
            fit_row_ids=frozenset(
                [
                    next(
                        candidate
                        for candidate, candidate_fold in zip(
                            context.row_ids, context.inner_fold_ids
                        )
                        if candidate_fold != fold_id
                    )
                ]
            )
        )
        for fold_id in context.inner_fold_ids
    )
    bank = GateFeatureBankView(
        row_ids=gate.row_ids,
        feature_names=("upstream_propensity",),
        source_kinds=("bow_nuisance",),
        consumer_roles=(PROPENSITY_NUISANCE_FEATURE_ROLE,),
        values=np.zeros((len(gate.row_ids), 1)),
        fit_row_provenance=(gate_lineage,),
        context_row_ids=context.row_ids,
        context_inner_fold_ids=context.inner_fold_ids,
        context_values=np.zeros((len(context.row_ids), 1)),
        context_fit_row_provenance=(incomplete,),
    )
    with pytest.raises(ValueError, match="exact complementary inner fold"):
        evaluate_untouched_gate_acceptance(
            context,
            gate,
            [
                _continuous("baseline_risk", "confounder"),
                _continuous("baseline_marker", "effect_modifier"),
            ],
            [
                _continuous("baseline_risk", "confounder"),
                _continuous("baseline_marker", "effect_modifier"),
            ],
            feature_bank_view=bank,
        )


def test_conditional_upstream_design_routes_each_role_without_raw_tau_misuse(monkeypatch):
    context, gate, _effect = _observed_rows()
    specs = [
        _continuous("baseline_risk", "confounder"),
        _continuous("baseline_marker", "effect_modifier"),
    ]
    gate_lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
    source = GateSourceSignalView(
        row_ids=gate.row_ids,
        source_names=("calibrated_tau",),
        source_kinds=("bow_weighted_r",),
        values=np.linspace(-2.0, 3.0, len(gate.row_ids))[:, None],
        fit_row_provenance=(gate_lineage,),
        **_conditional_context_kwargs(
            context,
            np.linspace(-1.0, 1.0, len(context.row_ids))[:, None],
        ),
    )
    raw_context = np.column_stack(
        [
            np.linspace(0.0, 1.0, len(context.row_ids)),
            np.linspace(10.0, 30.0, len(context.row_ids)),
            np.linspace(-5.0, 5.0, len(context.row_ids)),
        ]
    )
    bank = GateFeatureBankView(
        row_ids=gate.row_ids,
        feature_names=("raw_treatment", "raw_outcome", "raw_effect"),
        source_kinds=("bow", "htr", "neural_query_effect_moments"),
        consumer_roles=(
            PROPENSITY_NUISANCE_FEATURE_ROLE,
            OUTCOME_NUISANCE_FEATURE_ROLE,
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        ),
        values=np.column_stack(
            [
                np.linspace(0.0, 1.0, len(gate.row_ids)),
                np.linspace(10.0, 30.0, len(gate.row_ids)),
                np.linspace(-5.0, 5.0, len(gate.row_ids)),
            ]
        ),
        fit_row_provenance=(gate_lineage, gate_lineage, gate_lineage),
        **_conditional_context_kwargs(context, raw_context),
    )
    design = review_module._build_conditional_upstream_design(
        context,
        gate,
        source_view=source,
        feature_bank_view=bank,
    )
    binary_widths: list[int] = []
    continuous_widths: list[int] = []
    binary_predict_columns: list[np.ndarray] = []
    continuous_predict_columns: list[np.ndarray] = []
    effect_upstream_widths: list[tuple[int, int]] = []

    def binary(x_fit, y_fit, x_predict, *, alpha):
        binary_widths.append(x_fit.shape[1])
        binary_predict_columns.append(np.asarray(x_predict[:, -1]).copy())
        return np.full(len(x_predict), float(np.mean(y_fit)))

    def continuous(x_fit, y_fit, x_predict, *, alpha):
        continuous_widths.append(x_fit.shape[1])
        continuous_predict_columns.append(np.asarray(x_predict[:, -1]).copy())
        return np.full(len(x_predict), float(np.mean(y_fit)))

    def effect(*args, fit_effect_upstream=None, predict_effect_upstream=None, **kwargs):
        effect_upstream_widths.append(
            (fit_effect_upstream.shape[1], predict_effect_upstream.shape[1])
        )
        return np.zeros(len(args[1]), dtype=float)

    monkeypatch.setattr(review_module, "_fit_predict_binary", binary)
    monkeypatch.setattr(review_module, "_fit_predict_continuous", continuous)
    monkeypatch.setattr(review_module, "_fit_predict_effect", effect)
    review_module._fit_context_predict_gate(
        context,
        gate,
        specs,
        config=CausalReviewConfig(),
        upstream_design=design,
    )

    # Two explicit confounder columns plus exactly one matched nuisance bank.
    assert set(binary_widths) == {3}
    assert set(continuous_widths) == {3}
    np.testing.assert_allclose(
        binary_predict_columns[-1],
        design.values(PROPENSITY_NUISANCE_FEATURE_ROLE, scope="gate")[:, 0],
    )
    np.testing.assert_allclose(
        continuous_predict_columns[-1],
        design.values(OUTCOME_NUISANCE_FEATURE_ROLE, scope="gate")[:, 0],
    )
    # The effect regression receives calibrated tau + raw effect, but neither
    # raw treatment nor raw outcome features.
    assert effect_upstream_widths == [(2, 2)]


@pytest.mark.parametrize(
    "raw_kind",
    [
        "neural_query_treatment_moments",
        "neural_query_outcome_moments",
        "neural_query_effect_moments",
        "matched_pair_uplift",
        "whole_embedding_contrast",
        "cluster_embedding_contrast",
        "tfidf_topic_contrast",
        "tfidf_orphan_ngrams",
    ],
)
def test_uncalibrated_feature_bases_cannot_enter_tau_source_view(raw_kind):
    context, gate, _effect = _observed_rows()
    lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
    with pytest.raises(ValueError, match="GateFeatureBankView"):
        GateSourceSignalView(
            row_ids=gate.row_ids,
            source_names=("raw_effect_basis",),
            source_kinds=(raw_kind,),
            values=np.zeros((len(gate.row_ids), 1)),
            fit_row_provenance=(lineage,),
        )

    view = GateFeatureBankView(
        row_ids=gate.row_ids,
        feature_names=(f"{raw_kind}_feature",),
        source_kinds=(raw_kind,),
        consumer_roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,),
        values=np.zeros((len(gate.row_ids), 1)),
        fit_row_provenance=(lineage,),
    )
    assert view.source_kinds == (raw_kind,)


def test_gate_acceptance_applies_configurable_complexity_tradeoff():
    context, gate, _effect = _observed_rows()
    current = [
        _continuous("baseline_risk", "confounder"),
        _continuous("baseline_marker", "effect_modifier"),
        _continuous("marker_duplicate", "effect_modifier"),
    ]
    candidate = current[:2]
    unpenalized = evaluate_untouched_gate_acceptance(
        context,
        gate,
        current,
        candidate,
        config=CausalReviewConfig(
            contract_complexity_penalty=0.0,
            encoded_column_complexity_penalty=0.0,
            nuisance_relative_tolerance=1.0,
        ),
    )
    penalized = evaluate_untouched_gate_acceptance(
        context,
        gate,
        current,
        candidate,
        config=CausalReviewConfig(
            contract_complexity_penalty=0.10,
            encoded_column_complexity_penalty=0.02,
            nuisance_relative_tolerance=1.0,
        ),
    )
    assert penalized.candidate["complexity"]["contract_count"] == 2
    assert penalized.current["complexity"]["contract_count"] == 3
    assert (
        penalized.candidate["penalized_relative_r_loss_score"]
        < penalized.current["penalized_relative_r_loss_score"]
    )
    assert penalized.accepted
    # The penalty changes the declared objective, whether or not the duplicate
    # happens to improve raw R-loss under this finite sample.
    assert (
        penalized.candidate["penalized_relative_r_loss_score"]
        - penalized.current["penalized_relative_r_loss_score"]
        < unpenalized.candidate["penalized_relative_r_loss_score"]
        - unpenalized.current["penalized_relative_r_loss_score"]
    )


def test_gate_acceptance_uses_one_shared_r_loss_denominator(monkeypatch):
    context, gate, _effect = _observed_rows()
    specs = [
        _continuous("baseline_risk", "confounder"),
        _continuous("baseline_marker", "effect_modifier"),
    ]
    config = CausalReviewConfig(
        contract_complexity_penalty=0.0,
        encoded_column_complexity_penalty=0.0,
        nuisance_relative_tolerance=1.0,
    )
    metrics, e_hat, m_hat, tau_hat = review_module._fit_context_predict_gate(
        context,
        gate,
        specs,
        config=config,
    )
    current_metrics = deepcopy(metrics)
    candidate_metrics = deepcopy(metrics)
    current_metrics["effect"].update(
        weighted_r_loss=0.50,
        zero_effect_r_loss=1.00,
        r_loss_ratio=0.50,
    )
    # This candidate looks better only if it is divided by its own inflated
    # zero-effect denominator.  On the shared current scale, 0.60 is worse.
    candidate_metrics["effect"].update(
        weighted_r_loss=0.60,
        zero_effect_r_loss=2.00,
        r_loss_ratio=0.30,
    )
    calls = iter(
        [
            (current_metrics, e_hat, m_hat, tau_hat),
            (candidate_metrics, e_hat, m_hat, tau_hat),
        ]
    )
    monkeypatch.setattr(
        review_module,
        "_fit_context_predict_gate",
        lambda *args, **kwargs: next(calls),
    )

    decision = evaluate_untouched_gate_acceptance(
        context,
        gate,
        specs,
        specs,
        config=config,
    )

    assert decision.accepted is False
    assert "penalized_relative_r_loss_not_improved" in decision.reasons
    guard = decision.guards["penalized_relative_r_loss"]
    assert guard["reference_zero_effect_r_loss"] == pytest.approx(1.0)
    assert guard["candidate_specific_denominator_used"] is False
    assert guard["observed_score_improvement"] == pytest.approx(-0.10)


def test_gate_acceptance_rejects_an_exact_penalized_tie(monkeypatch):
    context, gate, _effect = _observed_rows()
    specs = [
        _continuous("baseline_risk", "confounder"),
        _continuous("baseline_marker", "effect_modifier"),
    ]
    config = CausalReviewConfig(
        contract_complexity_penalty=0.0,
        encoded_column_complexity_penalty=0.0,
        nuisance_relative_tolerance=1.0,
    )
    fitted = review_module._fit_context_predict_gate(
        context,
        gate,
        specs,
        config=config,
    )
    calls = iter([fitted, fitted])
    monkeypatch.setattr(
        review_module,
        "_fit_context_predict_gate",
        lambda *args, **kwargs: next(calls),
    )

    decision = evaluate_untouched_gate_acceptance(
        context,
        gate,
        specs,
        specs,
        config=config,
    )

    assert decision.accepted is False
    guard = decision.guards["penalized_relative_r_loss"]
    assert guard["strictly_positive_improvement_required"] is True
    assert guard["observed_score_improvement"] == pytest.approx(0.0)


def test_gate_acceptance_rejects_a_numerical_near_tie(monkeypatch):
    context, gate, _effect = _observed_rows()
    specs = [
        _continuous("baseline_risk", "confounder"),
        _continuous("baseline_marker", "effect_modifier"),
    ]
    config = CausalReviewConfig(
        contract_complexity_penalty=0.0,
        encoded_column_complexity_penalty=0.0,
        nuisance_relative_tolerance=1.0,
    )
    metrics, e_hat, m_hat, tau_hat = review_module._fit_context_predict_gate(
        context,
        gate,
        specs,
        config=config,
    )
    candidate_metrics = deepcopy(metrics)
    reference = float(metrics["effect"]["zero_effect_r_loss"])
    candidate_metrics["effect"]["weighted_r_loss"] = (
        float(metrics["effect"]["weighted_r_loss"]) - 5e-13 * reference
    )
    candidate_metrics["effect"]["r_loss_ratio"] = (
        candidate_metrics["effect"]["weighted_r_loss"]
        / candidate_metrics["effect"]["zero_effect_r_loss"]
    )
    calls = iter(
        [
            (metrics, e_hat, m_hat, tau_hat),
            (candidate_metrics, e_hat, m_hat, tau_hat),
        ]
    )
    monkeypatch.setattr(
        review_module,
        "_fit_context_predict_gate",
        lambda *args, **kwargs: next(calls),
    )

    decision = evaluate_untouched_gate_acceptance(
        context,
        gate,
        specs,
        specs,
        config=config,
    )

    assert decision.accepted is False
    assert 0.0 < decision.guards["penalized_relative_r_loss"]["observed_score_improvement"] < 1e-12


def test_same_name_semantic_revision_uses_separate_extraction_frames():
    candidate_context, candidate_gate, _effect = _observed_rows()
    current_context_frame = candidate_context.extracted.copy()
    current_gate_frame = candidate_gate.extracted.copy()
    current_context_frame["explicit_feat_baseline_marker"] = 0.0
    current_gate_frame["explicit_feat_baseline_marker"] = 0.0
    current_context = ObservableCausalRows(
        row_ids=candidate_context.row_ids,
        extracted=current_context_frame,
        treatment=candidate_context.treatment,
        outcome=candidate_context.outcome,
        inner_fold_ids=candidate_context.inner_fold_ids,
    )
    current_gate = ObservableCausalRows(
        row_ids=candidate_gate.row_ids,
        extracted=current_gate_frame,
        treatment=candidate_gate.treatment,
        outcome=candidate_gate.outcome,
    )
    current_spec = _continuous("baseline_marker", "effect_modifier")
    revised_spec = {
        **current_spec,
        "description": (
            "Revised baseline marker measurement using the documented value before "
            "treatment initiation."
        ),
    }
    decision = evaluate_untouched_gate_acceptance(
        current_context,
        current_gate,
        [current_spec],
        [revised_spec],
        candidate_context=candidate_context,
        candidate_gate=candidate_gate,
        config=CausalReviewConfig(nuisance_relative_tolerance=1.0),
    )
    assert decision.current["metrics"]["effect"]["tau_std"] == pytest.approx(0.0)
    assert decision.candidate["metrics"]["effect"]["tau_std"] > 0.0
    assert (
        decision.candidate["metrics"]["effect"]["weighted_r_loss"]
        < decision.current["metrics"]["effect"]["weighted_r_loss"]
    )
    assert decision.accepted

    changed_outcome = ObservableCausalRows(
        row_ids=candidate_gate.row_ids,
        extracted=candidate_gate.extracted,
        treatment=candidate_gate.treatment,
        outcome=np.asarray(candidate_gate.outcome) + 1.0,
    )
    with pytest.raises(ValueError, match="outcome must match"):
        evaluate_untouched_gate_acceptance(
            current_context,
            current_gate,
            [current_spec],
            [revised_spec],
            candidate_context=candidate_context,
            candidate_gate=changed_outcome,
        )


def test_gate_rejects_row_overlap_before_model_fitting():
    context, _gate, _effect = _observed_rows()
    specs = [_continuous("baseline_risk", "confounder")]
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_untouched_gate_acceptance(context, context, specs, specs)
