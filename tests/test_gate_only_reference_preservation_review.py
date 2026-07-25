from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import oci.inference.all_evidence_post_extraction_review as review_module
from oci.inference.all_evidence_post_extraction_review import (
    CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY,
    GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
    GateFeatureBankView,
    GateSourceSignalView,
    ObservableCausalRows,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    evaluate_untouched_gate_acceptance,
)
from oci.inference.fold_honest_r_stack import FitRowProvenance
from oci.inference.all_evidence_fusion_runner import (
    AllEvidenceFusionRunner,
    AllEvidenceFusionRunnerConfig,
)
from oci.inference.direct_upstream_numerical_reference_bank import (
    CALIBRATED_SOURCE_BANK,
    RAW_FEATURE_BANK,
    MaterializedRoleNeutralNumericalMatrix,
    RoleNeutralGateOnlyNumericalView,
)


def _specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_risk",
            "type": "continuous",
            "roles": ["confounder"],
            "description": "Baseline risk recorded before treatment.",
        },
        {
            "name": "baseline_marker",
            "type": "continuous",
            "roles": ["effect_modifier"],
            "description": "Baseline marker recorded before treatment.",
        },
    ]


def _rows() -> tuple[ObservableCausalRows, ObservableCausalRows]:
    rng = np.random.default_rng(812)
    context_count = 60
    gate_count = 24
    risk = rng.normal(size=context_count + gate_count)
    marker = rng.normal(size=context_count + gate_count)
    treatment = rng.binomial(
        1,
        1.0 / (1.0 + np.exp(-0.6 * risk)),
    ).astype(float)
    outcome = (
        0.7 * risk
        + treatment * (0.2 + 0.8 * marker)
        + rng.normal(scale=0.4, size=len(risk))
    )

    def frame(start: int, stop: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "explicit_feat_baseline_risk": risk[start:stop],
                "explicit_feat_baseline_risk_missing": False,
                "explicit_feat_baseline_marker": marker[start:stop],
                "explicit_feat_baseline_marker_missing": False,
            }
        )

    context = ObservableCausalRows(
        row_ids=tuple(f"spent_{index:03d}" for index in range(context_count)),
        extracted=frame(0, context_count),
        treatment=treatment[:context_count],
        outcome=outcome[:context_count],
        inner_fold_ids=tuple(index % 3 for index in range(context_count)),
    )
    gate = ObservableCausalRows(
        row_ids=tuple(f"gate_{index:03d}" for index in range(gate_count)),
        extracted=frame(context_count, context_count + gate_count),
        treatment=treatment[context_count:],
        outcome=outcome[context_count:],
    )
    return context, gate


def _gate_source(
    context: ObservableCausalRows,
    gate: ObservableCausalRows,
    *,
    fit_rows: frozenset[Any] | None = None,
    with_context_half: bool = False,
) -> GateSourceSignalView:
    lineage = FitRowProvenance(
        fit_row_ids=fit_rows
        if fit_rows is not None
        else frozenset(context.row_ids)
    )
    kwargs: dict[str, Any] = {}
    if with_context_half:
        assert context.inner_fold_ids is not None
        context_lineage = tuple(
            FitRowProvenance(
                fit_row_ids=frozenset(
                    row_id
                    for row_id, candidate_fold in zip(
                        context.row_ids,
                        context.inner_fold_ids,
                    )
                    if candidate_fold != fold_id
                )
            )
            for fold_id in context.inner_fold_ids
        )
        kwargs = {
            "context_row_ids": context.row_ids,
            "context_inner_fold_ids": context.inner_fold_ids,
            "context_values": np.linspace(-1.0, 1.0, len(context.row_ids))[:, None],
            "context_fit_row_provenance": (context_lineage,),
        }
    return GateSourceSignalView(
        row_ids=gate.row_ids,
        source_names=("cumulative_bow_effect",),
        source_kinds=("bow_weighted_r",),
        values=np.linspace(-0.8, 0.9, len(gate.row_ids))[:, None],
        fit_row_provenance=(lineage,),
        **kwargs,
    )


def _gate_feature(
    context: ObservableCausalRows,
    gate: ObservableCausalRows,
) -> GateFeatureBankView:
    lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
    return GateFeatureBankView(
        row_ids=gate.row_ids,
        feature_names=("cumulative_embedding_modifier",),
        source_kinds=("embedding_whole_cohort",),
        consumer_roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,),
        values=np.linspace(0.2, 1.2, len(gate.row_ids))[:, None],
        fit_row_provenance=(lineage,),
    )


def test_gate_only_references_never_enter_fit_or_conditional_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, gate = _rows()
    source = _gate_source(context, gate)
    feature = _gate_feature(context, gate)
    original_fit = review_module._fit_context_predict_gate
    fit_designs: list[Any] = []

    def checked_fit(*args: Any, **kwargs: Any):
        fit_designs.append(kwargs.get("upstream_design"))
        return original_fit(*args, **kwargs)

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("gate-only review attempted conditional access or a predictive refit")

    monkeypatch.setattr(review_module, "_fit_context_predict_gate", checked_fit)
    monkeypatch.setattr(
        review_module.GateSourceSignalView,
        "aligned_conditional_values",
        forbidden,
    )
    monkeypatch.setattr(
        review_module.GateFeatureBankView,
        "aligned_conditional_values",
        forbidden,
    )
    monkeypatch.setattr(
        review_module.GateSourceSignalView,
        "bind_fold",
        forbidden,
        raising=False,
    )
    monkeypatch.setattr(
        review_module.GateFeatureBankView,
        "bind_fold",
        forbidden,
        raising=False,
    )
    monkeypatch.setattr(
        review_module,
        "_predictive_upstream_family_ablations",
        forbidden,
    )

    decision = evaluate_untouched_gate_acceptance(
        context,
        gate,
        _specs(),
        _specs(),
        source_view=source,
        feature_bank_view=feature,
        upstream_review_policy=GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
    )

    assert fit_designs == [None, None]
    design_guard = decision.guards["conditional_upstream_design"]
    assert design_guard["upstream_review_policy"] == (
        GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY
    )
    assert design_guard["upstream_gate_values_used_as_training_or_prediction_covariates"] is False
    assert design_guard["gate_views_used_only_as_post_fit_reference_diagnostics"] is True
    assert decision.current["source_signal_evaluation"]["gate_reference_only"] is True
    assert (
        decision.current["source_signal_evaluation"][
            "calibrated_sources_used_as_effect_regression_covariates"
        ]
        is False
    )
    assert decision.current["feature_bank_evaluation"]["gate_reference_only"] is True
    assert (
        decision.current["feature_bank_evaluation"]["raw_feature_values_used_as_model_inputs"]
        is False
    )
    assert decision.current["source_signal_evaluation"]["sources"]
    assert decision.current["feature_bank_evaluation"]["features"]
    assert "source_gate_reference_r_loss" in decision.guards
    ablation = decision.guards["upstream_predictive_family_ablations"]
    assert ablation["status"] == "unavailable_by_design"
    assert ablation["passed"] is None
    assert ablation["predictive_refit_performed"] is False
    assert ablation["by_family"] == []
    assert decision.current["upstream_predictive_family_ablations"] == []
    assert decision.current["upstream_predictive_family_ablation_status"] == (
        "unavailable_by_design"
    )


def test_gate_only_rejects_any_context_side_upstream_bank() -> None:
    context, gate = _rows()
    source = _gate_source(context, gate, with_context_half=True)
    with pytest.raises(ValueError, match="must not supply context-side"):
        evaluate_untouched_gate_acceptance(
            context,
            gate,
            _specs(),
            _specs(),
            source_view=source,
            upstream_review_policy=GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
        )


@pytest.mark.parametrize(
    "fit_rows",
    [
        lambda context, gate: frozenset(context.row_ids[:-1]),
        lambda context, gate: frozenset((*context.row_ids, "future_partition_row")),
    ],
    ids=("missing_spent_row", "includes_future_row"),
)
def test_gate_only_requires_exact_complete_spent_lineage(fit_rows) -> None:
    context, gate = _rows()
    source = _gate_source(context, gate, fit_rows=fit_rows(context, gate))
    with pytest.raises(ValueError, match="exactly all spent rows"):
        evaluate_untouched_gate_acceptance(
            context,
            gate,
            _specs(),
            _specs(),
            source_view=source,
            upstream_review_policy=GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
        )


def test_gate_only_rejects_gate_lineage_and_wrong_gate_identity() -> None:
    context, gate = _rows()
    with pytest.raises(ValueError, match="lineage includes gate rows"):
        _gate_source(
            context,
            gate,
            fit_rows=frozenset((*context.row_ids, gate.row_ids[0])),
        )

    other_gate = ObservableCausalRows(
        row_ids=tuple(f"other_{index:03d}" for index in range(len(gate.row_ids))),
        extracted=gate.extracted,
        treatment=gate.treatment,
        outcome=gate.outcome,
    )
    source = _gate_source(context, other_gate)
    with pytest.raises(ValueError, match="exactly equal the untouched gate"):
        evaluate_untouched_gate_acceptance(
            context,
            gate,
            _specs(),
            _specs(),
            source_view=source,
            upstream_review_policy=GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
        )


def test_legacy_default_and_explicit_policy_remain_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, gate = _rows()
    source = _gate_source(context, gate, with_context_half=True)
    original = review_module.GateSourceSignalView.aligned_conditional_values
    calls: list[tuple[Any, ...]] = []

    def tracked(self, **kwargs: Any):
        calls.append(tuple(kwargs["exact_gate_row_ids"]))
        return original(self, **kwargs)

    monkeypatch.setattr(
        review_module.GateSourceSignalView,
        "aligned_conditional_values",
        tracked,
    )
    implicit = evaluate_untouched_gate_acceptance(
        context,
        gate,
        _specs(),
        _specs(),
        source_view=source,
    )
    explicit = evaluate_untouched_gate_acceptance(
        context,
        gate,
        _specs(),
        _specs(),
        source_view=source,
        upstream_review_policy=CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY,
    )

    assert len(calls) == 2
    assert implicit.decision_sha256 == explicit.decision_sha256
    assert implicit.current == explicit.current
    assert implicit.guards == explicit.guards
    assert (
        implicit.guards["conditional_upstream_design"]["upstream_review_policy"]
        == CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY
    )


def test_unknown_review_policy_fails_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, gate = _rows()

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("fit must not start for an unknown review policy")

    monkeypatch.setattr(review_module, "_fit_context_predict_gate", forbidden)
    with pytest.raises(ValueError, match="upstream_review_policy"):
        evaluate_untouched_gate_acceptance(
            context,
            gate,
            _specs(),
            _specs(),
            upstream_review_policy="implicit_or_unknown",
        )


def test_runner_gate_only_adapter_uses_prefit_view_without_bind_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, gate = _rows()
    gate_ids = tuple(map(int, range(500, 500 + len(gate.row_ids))))
    int_context = ObservableCausalRows(
        row_ids=tuple(range(len(context.row_ids))),
        extracted=context.extracted,
        treatment=context.treatment,
        outcome=context.outcome,
        inner_fold_ids=context.inner_fold_ids,
    )
    source_matrix = MaterializedRoleNeutralNumericalMatrix(
        row_ids=gate_ids,
        coordinate_ids=("source_coord",),
        names=("source_name",),
        source_families=("bow_r_loss",),
        source_kinds=("bow_weighted_r",),
        consumer_roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,),
        observable_axes=(("heterogeneity",),),
        bank_kinds=(CALIBRATED_SOURCE_BANK,),
        values=np.linspace(-1.0, 1.0, len(gate_ids))[:, None],
    )
    feature_matrix = MaterializedRoleNeutralNumericalMatrix(
        row_ids=gate_ids,
        coordinate_ids=("feature_coord",),
        names=("feature_name",),
        source_families=("embedding_whole_cohort",),
        source_kinds=("embedding_whole_cohort",),
        consumer_roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,),
        observable_axes=(("heterogeneity",),),
        bank_kinds=(RAW_FEATURE_BANK,),
        values=np.linspace(0.0, 2.0, len(gate_ids))[:, None],
    )
    opened = object.__new__(RoleNeutralGateOnlyNumericalView)
    opened.spent_row_ids = int_context.row_ids
    opened.gate_row_ids = gate_ids
    opened.context_oof_available = False
    opened.fit_or_refit_performed = False
    opened_calls: list[tuple[str, ...]] = []

    def identity(_self):
        return {
            "content_sha256": "a" * 64,
            "gate_fit_row_provenance": list(int_context.row_ids),
            "context_oof_available": False,
            "conditional_context_gate_view_claimed": False,
            "fit_or_refit_performed": False,
            "registered_gate_labels_accessed": False,
        }

    def materialize(_self, *, bank_kinds, **_kwargs):
        opened_calls.append(tuple(bank_kinds))
        return (
            source_matrix
            if tuple(bank_kinds) == (CALIBRATED_SOURCE_BANK,)
            else feature_matrix
        )

    monkeypatch.setattr(RoleNeutralGateOnlyNumericalView, "identity", identity)
    monkeypatch.setattr(RoleNeutralGateOnlyNumericalView, "materialize", materialize)

    class Provider:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def identity(self):
            return {"provider": "prefit_gate_only_test_v1"}

        def get_gate_only_view(self, **kwargs):
            self.calls.append(dict(kwargs))
            return opened

    provider = Provider()
    runner = object.__new__(AllEvidenceFusionRunner)
    runner.config = AllEvidenceFusionRunnerConfig(
        upstream_review_policy=GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
    )
    runner.gate_only_reference_review = True
    runner.review_gate_source_provider = provider
    runner.review_gate_feature_bank_provider = provider
    # Use the runner's canonical wrapper instead of duplicating its private
    # provider-identity normalization in the test.
    import oci.inference.all_evidence_fusion_runner as runner_module

    runner.review_gate_source_provider_identity = runner_module._review_provider_identity(
        provider,
        label="review_gate_source_provider",
    )
    runner.review_gate_feature_bank_provider_identity = runner_module._review_provider_identity(
        provider,
        label="review_gate_feature_bank_provider",
    )

    source, features, audit = runner._gate_only_reference_views(
        outer_fold=2,
        context_epoch=1,
        gate_row_ids=gate_ids,
        context=int_context,
    )

    assert provider.calls == [
        {
            "outer_fold": 2,
            "context_epoch": 1,
            "exact_spent_row_ids": int_context.row_ids,
            "exact_gate_row_ids": gate_ids,
        }
    ]
    assert opened_calls == [(CALIBRATED_SOURCE_BANK,), (RAW_FEATURE_BANK,)]
    assert source.context_row_ids == ()
    assert features.context_row_ids == ()
    assert all(
        lineage.recursive_fit_row_ids() == frozenset(int_context.row_ids)
        for view in (source, features)
        for column in view.fit_row_provenance
        for lineage in column
    )
    assert audit["conditional_context_values_accessed"] is False
    assert audit["bind_fold_called"] is False
    assert audit["fit_or_refit_performed"] is False


def test_runner_config_legacy_default_and_explicit_gate_only_policy() -> None:
    assert (
        AllEvidenceFusionRunnerConfig().upstream_review_policy
        == CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY
    )
    configured = AllEvidenceFusionRunnerConfig(
        upstream_review_policy=GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
    )
    assert configured.upstream_review_policy == (
        GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY
    )
    with pytest.raises(ValueError, match="registered review policy"):
        AllEvidenceFusionRunnerConfig(upstream_review_policy="implicit")
