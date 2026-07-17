from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

from oci.inference.all_evidence_post_extraction_review import (
    ObservableCausalRows,
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from oci.inference.fold_honest_r_stack import FitRowProvenance
from oci.inference.nested_fold_signal_producer import (
    FoldPredictionRows,
    FoldTrainingRows,
    NuisanceFoldPrediction,
    SignalFoldPrediction,
)
from oci.inference.neural_query_signal_fusion_adapter import (
    NeuralQueryFeatureBank,
    NeuralQueryFeatureBanks,
)
from oci.inference.post_extraction_review_providers import (
    NeuralQueryGateFeatureBankProvider,
    NestedGateSourceSignalProvider,
    neural_query_gate_feature_view,
)


class _CountingNuisanceBackend:
    def __init__(self, identity_suffix: str = "v1") -> None:
        self.calls = 0
        self.identity_suffix = identity_suffix

    def identity(self) -> Mapping[str, Any]:
        return {"backend": "test_nested_nuisance", "version": self.identity_suffix}

    def fit_predict(
        self,
        fit_rows: FoldTrainingRows,
        prediction_rows: FoldPredictionRows,
        *,
        random_state: int,
    ) -> NuisanceFoldPrediction:
        del random_state
        self.calls += 1
        lineage = FitRowProvenance(fit_row_ids=frozenset(fit_rows.row_ids))
        return NuisanceFoldPrediction(
            propensity=np.full(len(prediction_rows.row_ids), np.mean(fit_rows.treatment)),
            outcome_prediction=np.full(len(prediction_rows.row_ids), np.mean(fit_rows.outcome)),
            propensity_provenance=lineage,
            outcome_provenance=lineage,
        )


class _CountingEffectBackend:
    signal_name = "test_nested_effect"
    source_kind = "bow_r_loss"

    def __init__(self, identity_suffix: str = "v1") -> None:
        self.calls = 0
        self.identity_suffix = identity_suffix
        self.observed_inner_inner_folds: list[int] = []

    def identity(self) -> Mapping[str, Any]:
        return {"backend": "test_nested_effect", "version": self.identity_suffix}

    def fit_predict(
        self,
        fit_rows: FoldTrainingRows,
        prediction_rows: FoldPredictionRows,
        *,
        nuisance_backend: _CountingNuisanceBackend,
        inner_inner_folds: int,
        random_state: int,
    ) -> SignalFoldPrediction:
        del random_state
        self.calls += 1
        self.observed_inner_inner_folds.append(inner_inner_folds)
        split = max(1, len(fit_rows.row_ids) // 2)
        nuisance_fit = fit_rows.subset(np.arange(split))
        nuisance_prediction = FoldPredictionRows.from_training_subset(
            fit_rows, np.arange(split, len(fit_rows.row_ids))
        )
        nuisance = nuisance_backend.fit_predict(
            nuisance_fit,
            nuisance_prediction,
            random_state=99,
        )
        values = np.asarray(
            [len(text) / 100.0 + np.mean(fit_rows.treatment) for text in prediction_rows.texts],
            dtype=float,
        )
        lineage = FitRowProvenance(
            fit_row_ids=frozenset(fit_rows.row_ids),
            upstream=(nuisance.propensity_provenance, nuisance.outcome_provenance),
        )
        return SignalFoldPrediction(values=values, provenance=lineage)


def _review_rows() -> tuple[ObservableCausalRows, ObservableCausalRows, list[str], list[str]]:
    context_ids = tuple(range(10, 18))
    gate_ids = (30, 31)
    context = ObservableCausalRows(
        row_ids=context_ids,
        extracted=pd.DataFrame(index=range(len(context_ids))),
        treatment=np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=float),
        outcome=np.asarray([0.1, 1.2, 0.3, 1.4, 0.5, 1.6, 0.7, 1.8]),
        inner_fold_ids=(1, 1, 2, 2, 3, 3, 4, 4),
    )
    gate = ObservableCausalRows(
        row_ids=gate_ids,
        extracted=pd.DataFrame(index=range(len(gate_ids))),
        treatment=np.asarray([0.0, 1.0]),
        outcome=np.asarray([0.25, 1.75]),
    )
    context_texts = [f"Exact Context Note {row_id} " for row_id in context_ids]
    gate_texts = [f"Exact Gate Note {row_id} " for row_id in gate_ids]
    return context, gate, context_texts, gate_texts


def _provider(tmp_path, *, effect_suffix: str = "v1"):
    nuisance = _CountingNuisanceBackend()
    effect = _CountingEffectBackend(effect_suffix)
    provider = NestedGateSourceSignalProvider(
        tmp_path,
        nuisance_backend=nuisance,
        effect_backends=[effect],
        inner_inner_folds=3,
        random_state=17,
    )
    return provider, nuisance, effect


def test_nested_provider_fits_context_only_then_serves_narrow_bound_view(tmp_path):
    context, gate, context_texts, gate_texts = _review_rows()
    provider, nuisance, effect = _provider(tmp_path)
    bound = provider.bind_fold(
        outer_fold=2,
        context=context,
        context_texts=context_texts,
        gate_texts=gate_texts,
        exact_gate_row_ids=gate.row_ids,
    )
    view = bound.get_gate_source_view(outer_fold=2, exact_gate_row_ids=gate.row_ids)

    assert effect.calls == 1
    assert nuisance.calls == 1
    assert effect.observed_inner_inner_folds == [3]
    assert view.row_ids == gate.row_ids
    assert view.source_names == ("review_effect_source_0001",)
    assert view.source_kinds == ("nested_calibrated_effect_0001",)
    assert all(
        lineage.recursive_fit_row_ids().isdisjoint(gate.row_ids)
        for lineage in view.fit_row_provenance[0]
    )
    assert provider.get_gate_source_view(outer_fold=2, exact_gate_row_ids=gate.row_ids) is view
    assert provider.identity()["adaptive_acceptance_conditional_context_supported"] is False
    assert provider.identity()["intended_use"] == "legacy_gate_preservation_only"
    with pytest.raises(ValueError, match="different fold or gate"):
        bound.get_gate_source_view(outer_fold=3, exact_gate_row_ids=gate.row_ids)


def test_nested_provider_cache_binds_exact_inputs_and_skips_refit(tmp_path):
    context, gate, context_texts, gate_texts = _review_rows()
    provider, _nuisance, effect = _provider(tmp_path)
    first = provider.prepare_gate_source_view(
        outer_fold=1,
        context=context,
        context_texts=context_texts,
        gate_texts=gate_texts,
        exact_gate_row_ids=gate.row_ids,
    )
    assert effect.calls == 1
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text())
    binding = payload["binding"]
    assert {
        "context_row_ids_sha256",
        "context_text_sha256",
        "context_treatment_sha256",
        "context_outcome_sha256",
        "gate_row_ids_sha256",
        "gate_text_sha256",
    } <= set(binding)
    assert not any(
        token in key
        for key in binding
        for token in ("gate_treatment", "gate_outcome", "gate_label")
    )
    assert binding["gate_bind_api"] == "exact_row_ids_and_text_only_v1"

    fresh, _fresh_nuisance, fresh_effect = _provider(tmp_path)
    cached = fresh.prepare_gate_source_view(
        outer_fold=1,
        context=context,
        context_texts=context_texts,
        gate_texts=gate_texts,
        exact_gate_row_ids=gate.row_ids,
    )
    assert fresh_effect.calls == 0
    np.testing.assert_allclose(first.values, cached.values)

    fresh.prepare_gate_source_view(
        outer_fold=1,
        context=context,
        context_texts=context_texts,
        gate_texts=[*gate_texts[:-1], f"{gate_texts[-1]} changed"],
        exact_gate_row_ids=gate.row_ids,
    )
    assert fresh_effect.calls == 1
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_nested_provider_cache_and_backend_identity_tampering_fail_closed(tmp_path):
    context, gate, context_texts, gate_texts = _review_rows()
    provider, _nuisance, _effect = _provider(tmp_path)
    provider.prepare_gate_source_view(
        outer_fold=1,
        context=context,
        context_texts=context_texts,
        gate_texts=gate_texts,
        exact_gate_row_ids=gate.row_ids,
    )
    cache_path = next(tmp_path.glob("*.json"))
    payload = json.loads(cache_path.read_text())
    payload["values"][0][0] += 99.0
    cache_path.write_text(json.dumps(payload))
    fresh, _fresh_nuisance, fresh_effect = _provider(tmp_path)
    with pytest.raises(ValueError, match="content SHA-256 mismatch"):
        fresh.prepare_gate_source_view(
            outer_fold=1,
            context=context,
            context_texts=context_texts,
            gate_texts=gate_texts,
            exact_gate_row_ids=gate.row_ids,
        )
    assert fresh_effect.calls == 0

    class _ForbiddenIdentityEffect(_CountingEffectBackend):
        def identity(self):
            return {"backend": "test", "oracle_score": 0.9}

    with pytest.raises(ValueError, match="forbidden benchmark identity field"):
        NestedGateSourceSignalProvider(
            tmp_path / "forbidden",
            nuisance_backend=_CountingNuisanceBackend(),
            effect_backends=[_ForbiddenIdentityEffect()],
        )

    mutable, _mutable_nuisance, mutable_effect = _provider(tmp_path / "mutable")
    mutable_effect.identity_suffix = "changed-after-construction"
    with pytest.raises(ValueError, match="identity changed"):
        mutable.prepare_gate_source_view(
            outer_fold=1,
            context=context,
            context_texts=context_texts,
            gate_texts=gate_texts,
            exact_gate_row_ids=gate.row_ids,
        )


def test_nested_provider_binary_mode_rejects_continuous_outcomes(tmp_path):
    context, gate, context_texts, gate_texts = _review_rows()
    provider = NestedGateSourceSignalProvider(
        tmp_path,
        nuisance_backend=_CountingNuisanceBackend(),
        effect_backends=[_CountingEffectBackend()],
        outcome_type="binary",
    )
    with pytest.raises(ValueError, match="training.outcome must be binary"):
        provider.prepare_gate_source_view(
            outer_fold=1,
            context=context,
            context_texts=context_texts,
            gate_texts=gate_texts,
            exact_gate_row_ids=gate.row_ids,
        )


def _query_bank(
    bank: str,
    role: str,
    *,
    offset: float,
) -> NeuralQueryFeatureBank:
    train_ids = (0, 1, 2, 3, 4, 5)
    heldout_ids = (6, 7)
    folds = (1, 1, 2, 2, 3, 3)
    rows_by_fold = {
        fold: {row_id for row_id, row_fold in zip(train_ids, folds) if row_fold == fold}
        for fold in (1, 2, 3)
    }
    lineage = tuple(
        FitRowProvenance(fit_row_ids=frozenset(set(train_ids) - rows_by_fold[fold]))
        for fold in folds
    )
    outer_lineage = tuple(
        FitRowProvenance(fit_row_ids=frozenset(train_ids)) for _row_id in heldout_ids
    )
    return NeuralQueryFeatureBank(
        bank=bank,
        consumer_role=role,
        feature_names=(f"neural_query_{bank}_signed_mean",),
        outer_train_row_ids=train_ids,
        outer_heldout_row_ids=heldout_ids,
        outer_train_inner_oof=(np.arange(len(train_ids), dtype=float) + offset)[:, None],
        outer_heldout_final_refit=(np.arange(len(heldout_ids), dtype=float) + offset + 10.0)[
            :, None
        ],
        inner_fold_ids=folds,
        inner_fit_row_provenance=lineage,
        outer_fit_row_provenance=outer_lineage,
    )


def _query_banks() -> NeuralQueryFeatureBanks:
    treatment = _query_bank("treatment", PROPENSITY_NUISANCE_FEATURE_ROLE, offset=0.0)
    outcome = _query_bank("outcome", OUTCOME_NUISANCE_FEATURE_ROLE, offset=10.0)
    effect = _query_bank("effect", UNCALIBRATED_EFFECT_MODIFIER_ROLE, offset=20.0)
    return NeuralQueryFeatureBanks(
        outer_fold=1,
        split_fingerprint="a" * 64,
        manifest_sha256="b" * 64,
        signal_parquet_sha256="c" * 64,
        outer_train_row_ids=treatment.outer_train_row_ids,
        outer_heldout_row_ids=treatment.outer_heldout_row_ids,
        treatment=treatment,
        outcome=outcome,
        effect=effect,
    )


def test_neural_query_adapter_requires_one_complete_inner_fold():
    banks = _query_banks()
    view = neural_query_gate_feature_view(banks, exact_gate_row_ids=(1, 0))
    assert view.row_ids == (1, 0)
    assert view.feature_names == (
        "neural_query_treatment_signed_mean",
        "neural_query_outcome_signed_mean",
        "neural_query_effect_signed_mean",
    )
    assert view.consumer_roles == (
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    )
    np.testing.assert_allclose(view.values[:, 0], [1.0, 0.0])
    assert all(
        lineage.recursive_fit_row_ids().isdisjoint(view.row_ids)
        for feature_lineages in view.fit_row_provenance
        for lineage in feature_lineages
    )

    with pytest.raises(ValueError, match="spans multiple inner folds"):
        neural_query_gate_feature_view(banks, exact_gate_row_ids=(0, 2))
    with pytest.raises(ValueError, match="one complete inner fold"):
        neural_query_gate_feature_view(banks, exact_gate_row_ids=(0,))


def test_neural_query_adapter_rechecks_tampered_provenance_and_values():
    banks = _query_banks()
    original = banks.effect.inner_fit_row_provenance
    tampered = list(original)
    tampered[0] = FitRowProvenance(fit_row_ids=frozenset({1, 2, 3}))
    object.__setattr__(banks.effect, "inner_fit_row_provenance", tuple(tampered))
    with pytest.raises(ValueError, match="provenance touches an untouched gate row"):
        neural_query_gate_feature_view(banks, exact_gate_row_ids=(0, 1))

    object.__setattr__(banks.effect, "inner_fit_row_provenance", original)
    values = np.asarray(banks.effect.outer_train_inner_oof).copy()
    values[0, 0] = np.nan
    object.__setattr__(banks.effect, "outer_train_inner_oof", values)
    with pytest.raises(ValueError, match="non-finite"):
        neural_query_gate_feature_view(banks, exact_gate_row_ids=(0, 1))


def test_neural_query_adapter_rejects_unauthenticated_mapping():
    with pytest.raises(TypeError, match="authenticated NeuralQueryFeatureBanks"):
        neural_query_gate_feature_view({}, exact_gate_row_ids=(0, 1))


def test_neural_query_provider_exposes_authenticated_partitions_and_gate_view():
    banks = _query_banks()
    provider = NeuralQueryGateFeatureBankProvider({1: banks})
    assignments = provider.get_review_partition_assignments(
        outer_fold=1,
        exact_outer_train_row_ids=banks.outer_train_row_ids,
    )
    assert assignments == {1: (0, 1), 2: (2, 3), 3: (4, 5)}
    view = provider.get_gate_feature_bank_view(
        outer_fold=1,
        exact_gate_row_ids=assignments[2],
    )
    assert view.row_ids == (2, 3)
    identity = provider.identity()
    assert identity["raw_activations_are_calibrated_treatment_effects"] is False
    assert identity["adaptive_acceptance_conditional_context_supported"] is False
    assert identity["intended_use"] == "legacy_gate_preservation_only"
    assert json.dumps(list(banks.outer_train_row_ids)) not in json.dumps(identity)

    with pytest.raises(ValueError, match="identity/order"):
        provider.get_review_partition_assignments(
            outer_fold=1,
            exact_outer_train_row_ids=tuple(reversed(banks.outer_train_row_ids)),
        )
    with pytest.raises(ValueError, match="no authenticated"):
        provider.get_gate_feature_bank_view(outer_fold=2, exact_gate_row_ids=(0, 1))


def test_neural_query_provider_rechecks_shared_fold_assignments_after_tampering():
    banks = _query_banks()
    provider = NeuralQueryGateFeatureBankProvider({1: banks})
    object.__setattr__(banks.outcome, "inner_fold_ids", (1, 2, 2, 2, 3, 3))
    with pytest.raises(ValueError, match="share exact inner-fold assignments"):
        provider.get_review_partition_assignments(
            outer_fold=1,
            exact_outer_train_row_ids=banks.outer_train_row_ids,
        )
