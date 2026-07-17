import numpy as np
import pytest

from oci.inference.fold_honest_r_stack import (
    FitRowProvenance,
    FoldHonestRStack,
    INNER_OOF_SCOPE,
    OUTER_HELDOUT_SCOPE,
    SignalBundle,
)


def _inner_oof_bundle(row_ids, fold_ids, predictions, source_family):
    row_ids = np.asarray(row_ids)
    fold_ids = np.asarray(fold_ids)
    lineage_by_fold = {
        fold_id: FitRowProvenance(
            fit_row_ids=frozenset(row_ids[fold_ids != fold_id].tolist()),
        )
        for fold_id in np.unique(fold_ids)
    }
    provenance = tuple(lineage_by_fold[fold_id] for fold_id in fold_ids)
    return SignalBundle(
        row_ids=tuple(row_ids.tolist()),
        source_family=source_family,
        tau_predictions=np.asarray(predictions, dtype=float),
        prediction_scope=INNER_OOF_SCOPE,
        fit_row_provenance=provenance,
    )


def _outer_bundle(row_ids, predictions, source_family, train_row_ids):
    provenance = tuple(
        FitRowProvenance(fit_row_ids=frozenset(train_row_ids)) for _ in row_ids
    )
    return SignalBundle(
        row_ids=tuple(row_ids),
        source_family=source_family,
        tau_predictions=np.asarray(predictions, dtype=float),
        prediction_scope=OUTER_HELDOUT_SCOPE,
        fit_row_provenance=provenance,
    )


def _r_stack_data(seed=7, n_rows=4000, second_weight=0.25):
    rng = np.random.default_rng(seed)
    row_ids = np.arange(10_000, 10_000 + n_rows)
    fold_ids = np.arange(n_rows) % 4
    first_signal = rng.uniform(-0.5, 0.5, size=n_rows)
    second_signal = rng.uniform(-0.5, 0.5, size=n_rows)
    propensity = np.full(n_rows, 0.5)
    treatment = rng.binomial(1, propensity).astype(float)
    treatment_residual = treatment - propensity
    outcome_prediction = np.full(n_rows, 0.5)
    tau = 0.15 + 0.7 * first_signal + second_weight * second_signal
    outcome_probability = outcome_prediction + treatment_residual * tau
    outcome = rng.binomial(1, outcome_probability).astype(float)
    return {
        "row_ids": row_ids,
        "fold_ids": fold_ids,
        "first_signal": first_signal,
        "second_signal": second_signal,
        "propensity": propensity,
        "treatment": treatment,
        "outcome_prediction": outcome_prediction,
        "outcome": outcome,
    }


def _fit_stack(data, *, nonnegative, ridge_alphas=(1.0,)):
    order = np.arange(len(data["row_ids"]))[::-1]
    first = _inner_oof_bundle(
        data["row_ids"],
        data["fold_ids"],
        data["first_signal"],
        "bow_weighted_r",
    )
    second = _inner_oof_bundle(
        data["row_ids"][order],
        data["fold_ids"][order],
        data["second_signal"][order],
        "htr_effect",
    )
    return FoldHonestRStack(
        ridge_alphas=ridge_alphas,
        nonnegative=nonnegative,
    ).fit(
        row_ids=data["row_ids"],
        treatment=data["treatment"],
        outcome=data["outcome"],
        propensity=data["propensity"],
        outcome_prediction=data["outcome_prediction"],
        inner_fold_ids=data["fold_ids"],
        signals=[first, second],
    )


def test_signal_bundle_rejects_recursive_prediction_fit_overlap():
    nuisance_lineage = FitRowProvenance(fit_row_ids=frozenset({101, 102}))
    effect_lineage = FitRowProvenance(
        fit_row_ids=frozenset({102, 103}),
        upstream=(nuisance_lineage,),
    )

    with pytest.raises(ValueError, match="recursive fit-row provenance"):
        SignalBundle(
            row_ids=(101,),
            source_family="leaky_effect_model",
            tau_predictions=np.asarray([0.2]),
            prediction_scope=INNER_OOF_SCOPE,
            fit_row_provenance=(effect_lineage,),
        )


def test_regularized_r_stack_recovers_weights_and_aligns_outer_sources_and_rows():
    data = _r_stack_data()
    stack = _fit_stack(data, nonnegative=True)

    assert stack.source_families_ == ("bow_weighted_r", "htr_effect")
    assert stack.selected_alpha_ == 1.0
    assert stack.regularization_strategy_ == "precommitted_single_alpha"
    assert stack.selected_cv_r_loss_ is None
    assert stack.cv_results_ == []
    assert stack.constant_effect_ == pytest.approx(0.15, abs=0.08)
    np.testing.assert_allclose(stack.weights_, [0.7, 0.25], atol=0.12)
    assert stack.source_weights_ == pytest.approx(
        {"bow_weighted_r": 0.7, "htr_effect": 0.25}, abs=0.12
    )
    assert np.all(stack.weights_ >= 0.0)

    outer_ids = np.asarray([201, 202, 203, 204, 205])
    first_outer = np.asarray([-0.5, 0.1, 0.4, 1.2, -1.0])
    second_outer = np.asarray([0.3, -0.2, 0.8, -0.7, 0.5])
    first_order = np.asarray([4, 2, 0, 3, 1])
    second_order = np.asarray([1, 3, 4, 0, 2])
    first_bundle = _outer_bundle(
        outer_ids[first_order],
        first_outer[first_order],
        "bow_weighted_r",
        data["row_ids"],
    )
    second_bundle = _outer_bundle(
        outer_ids[second_order],
        second_outer[second_order],
        "htr_effect",
        data["row_ids"],
    )
    requested_order = np.asarray([203, 201, 205, 202, 204])
    source_positions = {row_id: index for index, row_id in enumerate(outer_ids)}
    expected = np.asarray(
        [
            stack.constant_effect_
            + stack.weights_[0] * first_outer[source_positions[row_id]]
            + stack.weights_[1] * second_outer[source_positions[row_id]]
            for row_id in requested_order
        ]
    )

    predictions = stack.predict(
        row_ids=requested_order,
        signals=[second_bundle, first_bundle],
    )
    np.testing.assert_allclose(predictions, expected)

    output = stack.predict_bundle(
        row_ids=requested_order,
        signals=[second_bundle, first_bundle],
    )
    assert output.row_ids == tuple(requested_order.tolist())
    assert output.prediction_scope == OUTER_HELDOUT_SCOPE
    np.testing.assert_allclose(output.tau_predictions, expected)
    for row_id, provenance in zip(output.row_ids, output.fit_row_provenance):
        recursive_rows = provenance.recursive_fit_row_ids()
        assert set(data["row_ids"]) <= set(recursive_rows)
        assert row_id not in recursive_rows


def test_signed_and_nonnegative_r_stack_weight_constraints():
    data = _r_stack_data(seed=19, second_weight=-0.6)
    signed = _fit_stack(data, nonnegative=False, ridge_alphas=(0.0,))
    constrained = _fit_stack(data, nonnegative=True, ridge_alphas=(0.0,))

    assert signed.weights_[1] == pytest.approx(-0.6, abs=0.15)
    assert constrained.weights_[1] >= -1e-10


def test_r_stack_rejects_recursive_provenance_from_another_row_in_heldout_fold():
    data = _r_stack_data(n_rows=40)
    row_ids = data["row_ids"]
    fold_ids = data["fold_ids"]
    provenance = []
    for position, (row_id, fold_id) in enumerate(zip(row_ids, fold_ids)):
        honest_fit_rows = set(row_ids[fold_ids != fold_id].tolist())
        if position == 0:
            same_fold_other = next(
                int(candidate)
                for candidate, candidate_fold in zip(row_ids, fold_ids)
                if candidate != row_id and candidate_fold == fold_id
            )
            honest_fit_rows.add(same_fold_other)
        provenance.append(FitRowProvenance(fit_row_ids=frozenset(honest_fit_rows)))
    signal = SignalBundle(
        row_ids=tuple(row_ids.tolist()),
        source_family="cross_fold_leak",
        tau_predictions=data["first_signal"],
        prediction_scope=INNER_OOF_SCOPE,
        fit_row_provenance=tuple(provenance),
    )

    with pytest.raises(ValueError, match="supplied inner heldout fold"):
        FoldHonestRStack(ridge_alphas=(0.0,)).fit(
            row_ids=row_ids,
            treatment=data["treatment"],
            outcome=data["outcome"],
            propensity=data["propensity"],
            outcome_prediction=data["outcome_prediction"],
            inner_fold_ids=fold_ids,
            signals=[signal],
        )


def test_r_stack_requires_inner_oof_scope_and_multiple_tuning_folds():
    data = _r_stack_data(n_rows=40)
    outer_scoped_signal = _outer_bundle(
        data["row_ids"],
        data["first_signal"],
        "wrong_scope",
        train_row_ids=(),
    )
    stack = FoldHonestRStack(ridge_alphas=(0.0,))

    with pytest.raises(ValueError, match="expected 'inner_oof'"):
        stack.fit(
            row_ids=data["row_ids"],
            treatment=data["treatment"],
            outcome=data["outcome"],
            propensity=data["propensity"],
            outcome_prediction=data["outcome_prediction"],
            inner_fold_ids=data["fold_ids"],
            signals=[outer_scoped_signal],
        )

    honest_signal = _inner_oof_bundle(
        data["row_ids"],
        data["fold_ids"],
        data["first_signal"],
        "honest_signal",
    )
    with pytest.raises(ValueError, match="at least two folds"):
        stack.fit(
            row_ids=data["row_ids"],
            treatment=data["treatment"],
            outcome=data["outcome"],
            propensity=data["propensity"],
            outcome_prediction=data["outcome_prediction"],
            inner_fold_ids=np.zeros(len(data["row_ids"]), dtype=int),
            signals=[honest_signal],
        )


def test_fitted_r_stack_rejects_prediction_on_meta_fit_rows():
    data = _r_stack_data(n_rows=60)
    stack = _fit_stack(data, nonnegative=False, ridge_alphas=(0.0,))
    training_id = int(data["row_ids"][0])
    first = _outer_bundle(
        [training_id],
        [data["first_signal"][0]],
        "bow_weighted_r",
        train_row_ids=(),
    )
    second = _outer_bundle(
        [training_id],
        [data["second_signal"][0]],
        "htr_effect",
        train_row_ids=(),
    )

    with pytest.raises(ValueError, match="overlap R-stack fit rows"):
        stack.predict(row_ids=[training_id], signals=[first, second])


def test_r_stack_rejects_adaptive_alpha_grids():
    with pytest.raises(ValueError, match="adaptive ridge alpha grids are forbidden"):
        FoldHonestRStack(ridge_alphas=(0.0, 1.0))


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("treatment", 0.5, "treatment must be binary"),
        ("outcome", 0.5, "outcome must be binary"),
        ("propensity", 0.0, "strictly inside"),
        ("propensity", 1.0, "strictly inside"),
        ("outcome_prediction", -0.1, r"inside \[0, 1\]"),
        ("outcome_prediction", 1.1, r"inside \[0, 1\]"),
    ],
)
def test_r_stack_rejects_invalid_binary_domains(field, bad_value, message):
    data = _r_stack_data(n_rows=40)
    corrupted = dict(data)
    corrupted[field] = np.asarray(data[field], dtype=float).copy()
    corrupted[field][0] = bad_value
    with pytest.raises(ValueError, match=message):
        _fit_stack(corrupted, nonnegative=False, ridge_alphas=(1.0,))


def test_r_stack_standardizes_sources_using_training_statistics_only():
    data = _r_stack_data(n_rows=1000)
    scaled = dict(data)
    scaled["second_signal"] = 10_000.0 * data["second_signal"] + 123.0
    original = _fit_stack(data, nonnegative=False, ridge_alphas=(1.0,))
    transformed = _fit_stack(scaled, nonnegative=False, ridge_alphas=(1.0,))

    outer_ids = [50_001, 50_002]
    first = np.asarray([-0.2, 0.3])
    second = np.asarray([0.1, -0.4])
    original_prediction = original.predict(
        row_ids=outer_ids,
        signals=[
            _outer_bundle(outer_ids, first, "bow_weighted_r", data["row_ids"]),
            _outer_bundle(outer_ids, second, "htr_effect", data["row_ids"]),
        ],
    )
    transformed_prediction = transformed.predict(
        row_ids=outer_ids,
        signals=[
            _outer_bundle(outer_ids, first, "bow_weighted_r", data["row_ids"]),
            _outer_bundle(
                outer_ids,
                10_000.0 * second + 123.0,
                "htr_effect",
                data["row_ids"],
            ),
        ],
    )
    np.testing.assert_allclose(original_prediction, transformed_prediction, atol=1e-10)
