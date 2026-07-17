from dataclasses import replace

import numpy as np
import pytest

from oci.inference.fold_honest_r_stack import FitRowProvenance
from oci.inference.fold_honest_signal_fusion import (
    HTR_NEURAL,
    FoldHonestNumericalSignalFusion,
)
from oci.inference.nested_fold_signal_producer import (
    BoWNuisanceBackend,
    FoldPredictionRows,
    FoldTrainingRows,
    NestedBoWSignalConfig,
    NestedFoldSignalOrchestrator,
    SignalFoldPrediction,
    make_nested_bow_r_orchestrator,
)


def _nested_data(seed=31, n_train=96, n_heldout=16):
    rng = np.random.default_rng(seed)
    train_ids = np.arange(1_000, 1_000 + n_train)
    heldout_ids = np.arange(5_000, 5_000 + n_heldout)
    confounder = rng.integers(0, 2, size=n_train)
    modifier = rng.integers(0, 2, size=n_train)
    train_texts = [
        " ".join(
            [
                "confounder_high" if confounder[index] else "confounder_low",
                "modifier_high" if modifier[index] else "modifier_low",
                f"noise_{index % 7}",
            ]
        )
        for index in range(n_train)
    ]
    propensity = 0.2 + 0.55 * confounder
    treatment = rng.binomial(1, propensity).astype(float)
    treatment_effect = 0.15 + 0.9 * modifier
    outcome_probability = 0.08 + 0.14 * confounder + treatment * (
        0.08 + 0.5 * modifier
    )
    outcome = rng.binomial(1, outcome_probability).astype(float)
    heldout_confounder = np.arange(n_heldout) % 2
    heldout_modifier = (np.arange(n_heldout) // 2) % 2
    heldout_texts = [
        " ".join(
            [
                "confounder_high" if heldout_confounder[index] else "confounder_low",
                "modifier_high" if heldout_modifier[index] else "modifier_low",
                f"noise_{index % 7}",
            ]
        )
        for index in range(n_heldout)
    ]
    folds = np.arange(n_train) % 4 + 1
    return {
        "train": FoldTrainingRows(
            row_ids=tuple(train_ids.tolist()),
            texts=tuple(train_texts),
            treatment=treatment,
            outcome=outcome,
        ),
        "heldout": FoldPredictionRows(
            row_ids=tuple(heldout_ids.tolist()),
            texts=tuple(heldout_texts),
        ),
        "folds": folds,
        "heldout_modifier": heldout_modifier,
    }


def _config(**overrides):
    values = {
        "outcome_type": "binary",
        "inner_inner_folds": 3,
        "ngram_range_min": 1,
        "ngram_range_max": 1,
        "min_df": 2,
        "max_features": 200,
        "nuisance_ridge_alpha": 0.2,
        "effect_ridge_alpha": 0.2,
        "random_state": 73,
    }
    values.update(overrides)
    return NestedBoWSignalConfig(**values)


def test_nested_bow_r_producer_excludes_each_meta_fold_recursively(tmp_path):
    data = _nested_data()
    orchestrator = make_nested_bow_r_orchestrator(_config())
    input_path = tmp_path / "label_free_outer_split.json"
    input_path.write_text('{"split":"outer"}\n', encoding="utf-8")
    input_paths = {"label_free_outer_split": input_path}
    audit = orchestrator.producer_audit(input_artifact_paths=input_paths)
    written = orchestrator.produce_and_write(
        tmp_path / "nested_bow_signal.json",
        outer_fold=3,
        split_fingerprint="a" * 64,
        outer_train=data["train"],
        outer_heldout=data["heldout"],
        inner_fold_ids=data["folds"],
        input_artifact_paths=input_paths,
        producer_audit=audit,
    )
    package = written.package
    signal = package.signals[0]
    assert signal.source_kind == "bow_r_loss"
    assert package.producer_audit.outer_heldout_labels_consumed is False
    assert package.producer_audit.posthoc_targets_consumed is False
    assert "fold_model_inputs" in package.producer_audit.input_artifact_sha256s
    input_material = next(
        record
        for record in package._material_records
        if record.category == "input" and record.name == "fold_model_inputs"
    )
    assert input_material.path.read_bytes()

    train_ids = np.asarray(package.outer_train_row_ids)
    folds = np.asarray(data["folds"])
    for position, lineage in enumerate(signal.inner_oof.fit_row_provenance):
        expected_fit = set(train_ids[folds != folds[position]])
        assert set(lineage.recursive_fit_row_ids()) == expected_fit
    for vector in (package.nuisance.propensity, package.nuisance.outcome_prediction):
        for position, lineage in enumerate(vector.fit_row_provenance):
            expected_fit = set(train_ids[folds != folds[position]])
            assert set(lineage.recursive_fit_row_ids()) == expected_fit
    assert all(
        set(lineage.recursive_fit_row_ids()) == set(train_ids)
        for lineage in signal.outer_heldout.fit_row_provenance
    )

    fusion = FoldHonestNumericalSignalFusion(ridge_alphas=(0.1,)).fit(
        package,
        row_ids=package.outer_train_row_ids,
        treatment=data["train"].treatment,
        outcome=data["train"].outcome,
    )
    predictions = fusion.predict(package)
    assert np.isfinite(predictions).all()
    assert float(np.std(predictions)) > 0.0


class _LeakySignalBackend:
    signal_name = "leaky_htr_signal"
    source_kind = HTR_NEURAL

    def identity(self):
        return {"backend": "deliberately_leaky_test_backend"}

    def fit_predict(
        self,
        fit_rows,
        prediction_rows,
        *,
        nuisance_backend,
        inner_inner_folds,
        random_state,
    ):
        del nuisance_backend, inner_inner_folds, random_state
        return SignalFoldPrediction(
            values=np.zeros(len(prediction_rows.row_ids), dtype=float),
            provenance=FitRowProvenance(
                fit_row_ids=frozenset([*fit_rows.row_ids, prediction_rows.row_ids[0]])
            ),
        )


def test_generic_orchestrator_rejects_backend_recursive_fold_leakage(tmp_path):
    config = _config(min_df=1)
    with pytest.raises(TypeError, match="unauthenticated generic signal backends"):
        NestedFoldSignalOrchestrator(
            nuisance_backend=BoWNuisanceBackend(config),
            signal_backends=[_LeakySignalBackend()],
            inner_inner_folds=2,
            random_state=11,
        )


def test_nested_producer_binds_code_and_config_hashes(tmp_path):
    data = _nested_data(n_train=32, n_heldout=8)
    orchestrator = make_nested_bow_r_orchestrator(_config(min_df=1))
    input_path = tmp_path / "label_free_outer_split.json"
    input_path.write_text('{"split":"outer"}\n', encoding="utf-8")
    input_paths = {"label_free_outer_split": input_path}
    audit = orchestrator.producer_audit(input_artifact_paths=input_paths)
    assert orchestrator.identity()["adaptive_untouched_gate_diagnostic_views"] is False
    stale = replace(audit, producer_config_sha256="0" * 64)

    with pytest.raises(ValueError, match="self-attested or stale"):
        orchestrator.produce_and_write(
            tmp_path / "stale.json",
            outer_fold=2,
            split_fingerprint="c" * 64,
            outer_train=data["train"],
            outer_heldout=data["heldout"],
            inner_fold_ids=data["folds"],
            input_artifact_paths=input_paths,
            producer_audit=stale,
        )


def test_nested_producer_rejects_runtime_backend_override(tmp_path):
    data = _nested_data(n_train=32, n_heldout=8)
    orchestrator = make_nested_bow_r_orchestrator(_config(min_df=1))
    orchestrator.nuisance_backend.fit_predict = lambda *args, **kwargs: None

    with pytest.raises(TypeError, match="unauthenticated instance overrides"):
        orchestrator.produce_and_write(
            tmp_path / "override.json",
            outer_fold=2,
            split_fingerprint="c" * 64,
            outer_train=data["train"],
            outer_heldout=data["heldout"],
            inner_fold_ids=data["folds"],
            input_artifact_paths={"input": tmp_path / "missing"},
        )


def test_nested_bow_producer_has_safe_constant_fallback_for_empty_vocabulary(tmp_path):
    data = _nested_data(n_train=32, n_heldout=8)
    blank_train = replace(data["train"], texts=("",) * len(data["train"].row_ids))
    blank_heldout = replace(data["heldout"], texts=("",) * len(data["heldout"].row_ids))
    orchestrator = make_nested_bow_r_orchestrator(_config(min_df=1))
    input_path = tmp_path / "label_free_outer_split.json"
    input_path.write_text('{"split":"outer"}\n', encoding="utf-8")
    input_paths = {"label_free_outer_split": input_path}
    audit = orchestrator.producer_audit(input_artifact_paths=input_paths)
    written = orchestrator.produce_and_write(
        tmp_path / "blank.json",
        outer_fold=4,
        split_fingerprint="2" * 64,
        outer_train=blank_train,
        outer_heldout=blank_heldout,
        inner_fold_ids=data["folds"],
        input_artifact_paths=input_paths,
        producer_audit=audit,
    )

    assert np.isfinite(written.package.signals[0].inner_oof.tau_predictions).all()
    assert np.isfinite(written.package.signals[0].outer_heldout.tau_predictions).all()
