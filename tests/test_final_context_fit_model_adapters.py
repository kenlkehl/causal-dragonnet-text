import inspect
from pathlib import Path

import numpy as np
import pytest

import oci.models.causal_forest_head as causal_forest_head_module
from oci.inference.all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from oci.inference.context_fit_upstream_gate_provider import ContextFitUpstreamPrediction
from oci.inference.final_context_fit_causal_forest_adapter import (
    FixedCausalForestHeadBackend,
    NestedFinalForestFeaturesRequired,
    SealedFinalForestExplicitBlock,
    StrictOuterHonestFinalCausalForestAdapter,
    prepare_final_causal_forest_design,
)
from oci.inference.final_context_fit_r_stack_adapter import (
    EXACT_OUTCOME_PREDICTION,
    EXACT_PROPENSITY_PREDICTION,
    SealedExactNuisanceBankExtension,
    StrictFinalContextFitRStackAdapter,
)
from oci.inference.final_context_fit_upstream_bank import FinalContextFitUpstreamProducer
from oci.inference.fold_honest_r_stack import FitRowProvenance


def _useful_signal(row_ids):
    rows = np.asarray(row_ids, dtype=float)
    return 0.24 * np.sin(rows * 0.071)


def _noise_signal(row_ids):
    rows = np.asarray(row_ids, dtype=float)
    return 0.20 * np.cos(rows * 0.173 + 0.4)


class _DeterministicAllEvidenceBackend:
    def identity(self):
        return {"backend": "deterministic_all_evidence_for_adapter_v1", "revision": 1}

    def fit_predict(
        self,
        *,
        outer_fold,
        context_row_ids,
        context_texts,
        context_treatment,
        context_outcome,
        gate_row_ids,
        gate_texts,
        work_dir,
    ):
        del (
            outer_fold,
            context_row_ids,
            context_texts,
            context_treatment,
            context_outcome,
            gate_texts,
            work_dir,
        )
        useful = _useful_signal(gate_row_ids)
        noise = _noise_signal(gate_row_ids)
        raw_values = np.column_stack(
            (
                np.sin(np.asarray(gate_row_ids, dtype=float) * 0.031),
                1000.0 * useful,
                -1000.0 * noise,
            )
        )
        return ContextFitUpstreamPrediction(
            gate_row_ids=gate_row_ids,
            calibrated_source_names=("bow_direct_tau", "htr_direct_tau"),
            calibrated_source_kinds=(
                "nested_calibrated_bow_weighted_r",
                "nested_calibrated_htr_weighted_r",
            ),
            calibrated_source_values=np.column_stack((useful, noise)),
            feature_names=("pair_basis", "bow_treatment_basis", "htr_outcome_basis"),
            feature_kinds=("matched_pair_uplift", "bow_nuisance", "htr_nuisance"),
            feature_roles=(
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                OUTCOME_NUISANCE_FEATURE_ROLE,
            ),
            feature_values=raw_values,
        )


class _FirstEffectColumnForestBackend:
    def __init__(self):
        self.calls = []

    def identity(self):
        return {
            "backend": "test_first_effect_column_causal_forest_v1",
            "honest": True,
            "tune_model": False,
        }

    def fit_predict(
        self,
        *,
        effect_train,
        control_train,
        treatment,
        outcome,
        effect_heldout,
        control_heldout,
    ):
        self.calls.append(
            {
                "effect_train": np.asarray(effect_train).copy(),
                "control_train": np.asarray(control_train).copy(),
                "treatment": np.asarray(treatment).copy(),
                "outcome": np.asarray(outcome).copy(),
                "effect_heldout": np.asarray(effect_heldout).copy(),
                "control_heldout": np.asarray(control_heldout).copy(),
            }
        )
        return np.asarray(effect_heldout, dtype=float)[:, 0]

    def fit_audit(self):
        return {
            "configured_parameters": {"test_backend": "first_effect_column"},
            "tuning_configured": False,
            "tuning_attempted": False,
            "tuning_succeeded": None,
            "tuning_failure_fell_back_to_configured_parameters": False,
            "tuning_params": None,
            "effective_parameters": {"test_backend": "first_effect_column"},
            "outer_train_labels_only": True,
            "outer_heldout_labels_accepted": False,
        }


@pytest.fixture(scope="module")
def final_package(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("final_context_fit_model_adapters")
    rng = np.random.default_rng(20260716)
    n_train = 800
    n_heldout = 60
    train_rows = tuple(range(10_000, 10_000 + n_train))
    heldout_rows = tuple(range(50_000, 50_000 + n_heldout))
    folds = tuple((index % 5) + 1 for index in range(n_train))
    propensity = np.full(n_train, 0.5, dtype=float)
    treatment = rng.binomial(1, propensity).astype(float)
    tau = 0.07 + 0.82 * _useful_signal(train_rows)
    probability = np.clip(0.5 + (treatment - propensity) * tau, 0.02, 0.98)
    outcome = rng.binomial(1, probability).astype(float)
    inputs = {
        "outer_fold": 2,
        "outer_train_row_ids": train_rows,
        "outer_train_texts": tuple(f"train row {row}" for row in train_rows),
        "outer_train_treatment": treatment,
        "outer_train_outcome": outcome,
        "outer_heldout_row_ids": heldout_rows,
        "outer_heldout_texts": tuple(f"heldout row {row}" for row in heldout_rows),
        "meta_inner_fold_ids": folds,
    }
    package = FinalContextFitUpstreamProducer(
        tmp_path / "final_upstream",
        backend=_DeterministicAllEvidenceBackend(),
    ).produce(**inputs)
    return package, inputs


def _exact_nuisance(package):
    source = package.calibrated_sources
    names = (
        "bow_propensity_prediction",
        "htr_propensity_prediction",
        "bow_outcome_prediction",
        "htr_outcome_prediction",
    )
    semantics = (
        EXACT_PROPENSITY_PREDICTION,
        EXACT_PROPENSITY_PREDICTION,
        EXACT_OUTCOME_PREDICTION,
        EXACT_OUTCOME_PREDICTION,
    )
    train_values = np.full((len(source.train_row_ids), 4), 0.5, dtype=float)
    heldout_values = np.full((len(source.heldout_row_ids), 4), 0.5, dtype=float)
    train_lineage = tuple(
        tuple(row[0] for _ in names) for row in source.train_oof_fit_row_provenance
    )
    heldout_lineage = tuple(
        tuple(row[0] for _ in names) for row in source.outer_heldout_fit_row_provenance
    )
    return SealedExactNuisanceBankExtension.seal_for_package(
        package,
        prediction_names=names,
        prediction_kinds=(
            "bow_nuisance",
            "htr_nuisance",
            "bow_nuisance",
            "htr_nuisance",
        ),
        prediction_semantics=semantics,
        train_oof_values=train_values,
        outer_heldout_values=heldout_values,
        train_oof_fit_row_provenance=train_lineage,
        outer_heldout_fit_row_provenance=heldout_lineage,
    )


def _explicit_block(package):
    source = package.calibrated_sources
    train_rows = np.asarray(source.train_row_ids, dtype=float)
    heldout_rows = np.asarray(source.heldout_row_ids, dtype=float)
    train_values = np.sin(train_rows[:, None] * 0.013)
    heldout_values = np.sin(heldout_rows[:, None] * 0.013)
    return SealedFinalForestExplicitBlock.seal_for_package(
        package,
        effect_names=("dual_role_variable",),
        control_names=("dual_role_variable",),
        effect_train_values=train_values,
        effect_heldout_values=heldout_values,
        control_train_values=train_values,
        control_heldout_values=heldout_values,
    )


def test_r_stack_uses_only_calibrated_tau_and_preserves_useful_direct_signal(final_package):
    package, inputs = final_package
    nuisance = _exact_nuisance(package)
    stack = StrictFinalContextFitRStackAdapter(ridge_alpha=1.0, nonnegative=True).fit(
        package,
        outer_train_row_ids=inputs["outer_train_row_ids"],
        treatment=inputs["outer_train_treatment"],
        outcome=inputs["outer_train_outcome"],
        exact_nuisance=nuisance,
    )
    prediction = stack.predict_effect(package, exact_nuisance=nuisance)
    expected_signal = _useful_signal(inputs["outer_heldout_row_ids"])

    assert np.std(prediction) > 0.02
    assert np.corrcoef(prediction, expected_signal)[0, 1] > 0.85
    assert stack._stack.weights_[0] > stack._stack.weights_[1]
    audit = stack.audit_record()
    assert audit["raw_feature_bank_used_as_tau"] is False
    assert audit["raw_feature_bank_used_as_nuisance_predictions"] is False
    assert audit["calibrated_source_names"] == ["bow_direct_tau", "htr_direct_tau"]
    package.verify_authenticated_content()
    nuisance.verify_authenticated_content()


def test_r_stack_rejects_row_reordering_and_raw_feature_substitution(final_package):
    package, inputs = final_package
    nuisance = _exact_nuisance(package)
    with pytest.raises(ValueError, match="row identity or order"):
        StrictFinalContextFitRStackAdapter().fit(
            package,
            outer_train_row_ids=tuple(reversed(inputs["outer_train_row_ids"])),
            treatment=inputs["outer_train_treatment"][::-1],
            outcome=inputs["outer_train_outcome"][::-1],
            exact_nuisance=nuisance,
        )
    with pytest.raises(TypeError, match="sealed exact-nuisance"):
        StrictFinalContextFitRStackAdapter().fit(
            package,
            outer_train_row_ids=inputs["outer_train_row_ids"],
            treatment=inputs["outer_train_treatment"],
            outcome=inputs["outer_train_outcome"],
            exact_nuisance=package.raw_features,
        )


def test_exact_nuisance_extension_detects_tampering_and_bad_lineage(final_package):
    package, _inputs = final_package
    nuisance = _exact_nuisance(package)
    nuisance.train_oof_values.setflags(write=True)
    nuisance.train_oof_values[0, 0] += 0.01
    with pytest.raises(ValueError, match="content was modified"):
        nuisance.verify_authenticated_content()

    source = package.calibrated_sources
    names = ("propensity_prediction", "outcome_prediction")
    honest = tuple(tuple(row[0] for _ in names) for row in source.train_oof_fit_row_provenance)
    bad_first = FitRowProvenance(fit_row_ids=frozenset(source.train_row_ids))
    bad = ((bad_first, bad_first), *honest[1:])
    heldout = tuple(tuple(row[0] for _ in names) for row in source.outer_heldout_fit_row_provenance)
    with pytest.raises(ValueError, match="exact complementary fit rows"):
        SealedExactNuisanceBankExtension.seal_for_package(
            package,
            prediction_names=names,
            prediction_kinds=("bow_nuisance", "bow_nuisance"),
            prediction_semantics=(
                EXACT_PROPENSITY_PREDICTION,
                EXACT_OUTCOME_PREDICTION,
            ),
            train_oof_values=np.full((len(source.train_row_ids), 2), 0.5),
            outer_heldout_values=np.full((len(source.heldout_row_ids), 2), 0.5),
            train_oof_fit_row_provenance=bad,
            outer_heldout_fit_row_provenance=heldout,
        )


def test_forest_design_routes_tau_modifier_controls_and_exact_nuisance_separately(
    final_package,
):
    package, _inputs = final_package
    nuisance = _exact_nuisance(package)
    explicit = _explicit_block(package)
    design = prepare_final_causal_forest_design(
        package,
        exact_nuisance=nuisance,
        explicit_features=explicit,
    )

    source = package.calibrated_sources
    np.testing.assert_array_equal(design.effect_train_values[:, :2], source.train_oof_values)
    assert design.effect_train_values.shape[1] == 2 + 1 + 1
    assert design.control_train_values.shape[1] == 2 + 4 + 1
    assert explicit.effect_names == explicit.control_names == ("dual_role_variable",)
    np.testing.assert_array_equal(
        design.effect_train_values[:, -1], design.control_train_values[:, -1]
    )
    assert set(design.effect_names).isdisjoint(design.control_names)
    routing = design.routing_audit
    assert routing["effect_columns"] == {
        "calibrated_tau_count": 2,
        "raw_modifier_count": 1,
        "explicit_modifier_count": 1,
    }
    assert routing["control_columns"] == {
        "raw_nuisance_basis_count": 2,
        "exact_nuisance_prediction_count": 4,
        "explicit_control_count": 1,
    }
    assert routing["raw_modifier_features_relabelled_as_calibrated_tau"] is False
    assert routing["raw_nuisance_bases_relabelled_as_exact_predictions"] is False
    assert routing["safe_for_meta_inner_forest_oof_generation"] is False


def test_outer_forest_uses_direct_tau_and_refuses_to_fake_oof_signal(final_package):
    package, inputs = final_package
    nuisance = _exact_nuisance(package)
    explicit = _explicit_block(package)
    backend = _FirstEffectColumnForestBackend()
    adapter = StrictOuterHonestFinalCausalForestAdapter(backend=backend)
    result = adapter.fit_predict(
        package,
        outer_train_row_ids=inputs["outer_train_row_ids"],
        treatment=inputs["outer_train_treatment"],
        outcome=inputs["outer_train_outcome"],
        exact_nuisance=nuisance,
        explicit_features=explicit,
    )

    np.testing.assert_allclose(
        result.values,
        package.calibrated_sources.outer_heldout_values[:, 0],
    )
    assert backend.calls[0]["effect_train"].shape[1] == 4
    assert backend.calls[0]["control_train"].shape[1] == 7
    for row_id, lineage in zip(result.heldout_row_ids, result.fit_row_provenance):
        assert lineage.recursive_fit_row_ids() == frozenset(inputs["outer_train_row_ids"])
        assert row_id not in lineage.recursive_fit_row_ids()
    result.verify_authenticated_content()
    audit = adapter.audit_record()
    assert audit["single_final_outer_heldout_fit"] is True
    assert audit["meta_inner_forest_oof_emitted"] is False
    assert audit["forest_tuning_from_assembled_oof_bank"] is False
    assert audit["forest_tuning_succeeded"] is None
    assert audit["forest_effective_parameters"] == {"test_backend": "first_effect_column"}
    with pytest.raises(NestedFinalForestFeaturesRequired, match="complement-only"):
        adapter.emit_meta_inner_oof_tau(package)


def test_model_adapter_apis_have_no_heldout_label_or_posthoc_target_channel():
    methods = (
        StrictFinalContextFitRStackAdapter.fit,
        StrictFinalContextFitRStackAdapter.predict_effect,
        StrictOuterHonestFinalCausalForestAdapter.fit_predict,
        _FirstEffectColumnForestBackend.fit_predict,
    )
    forbidden_parameters = {
        "heldout_treatment",
        "heldout_outcome",
        "outer_heldout_treatment",
        "outer_heldout_outcome",
        "true_ite",
        "oracle_ite",
    }
    for method in methods:
        assert forbidden_parameters.isdisjoint(inspect.signature(method).parameters)

    module_root = Path(__file__).parents[1] / "oci" / "inference"
    for filename in (
        "final_context_fit_r_stack_adapter.py",
        "final_context_fit_causal_forest_adapter.py",
    ):
        source = (module_root / filename).read_text(encoding="utf-8")
        assert "freeze_sentence_encoder" not in source


def test_repository_forest_backend_exposes_all_scientific_defaults():
    backend = FixedCausalForestHeadBackend()
    identity = dict(backend.identity())
    runtime = identity.pop("repository_runtime")
    assert identity == {
        "backend": "repository_strict_causal_forest_path_v3",
        "configuration_mode": "legacy_compatibility_shim_v1",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "honest": True,
        "inference": True,
        "subforest_size": 4,
        "tune_model": True,
        "nuisance_n_estimators": 100,
        "nuisance_max_depth": None,
        "nuisance_min_samples_leaf": 10,
        "nuisance_treatment_max_features": "sqrt",
        "nuisance_outcome_max_features": 1.0,
        "random_state": 42,
        "exact_nuisance_used_as_fixed_internal_predictions": False,
        "tuning_labels": "outer_train_only",
        "outer_heldout_labels_accepted": False,
    }
    assert runtime["causal_forest_head_module_sha256"]
    assert runtime["strict_runtime_module_sha256"]
    assert runtime["causal_forest_head_create_strict_model_code_sha256"]
    assert runtime["causal_forest_head_fit_code_sha256"]
    assert runtime["causal_forest_head_fit_audit_code_sha256"]
    assert runtime["econml_distribution_version"] != "not_installed"
    assert runtime["sklearn_distribution_version"] != "not_installed"
    assert isinstance(runtime["econml_import_available"], bool)
    assert "n_jobs" not in backend.identity()
    assert FixedCausalForestHeadBackend(n_jobs=1).identity() == (
        FixedCausalForestHeadBackend(n_jobs=7).identity()
    )
    with pytest.raises(ValueError, match="honest tree splitting"):
        FixedCausalForestHeadBackend(honest=False)


def test_repository_forest_backend_attests_dynamic_head_implementation(monkeypatch):
    backend = FixedCausalForestHeadBackend()
    adapter = StrictOuterHonestFinalCausalForestAdapter(backend=backend)

    def changed_fit(self, X, T, Y, W=None, **_kwargs):
        del X, T, Y, W
        return self

    monkeypatch.setattr(causal_forest_head_module.CausalForestHead, "fit", changed_fit)
    with pytest.raises(ValueError, match="identity changed|runtime code changed"):
        adapter._assert_backend_stable()
