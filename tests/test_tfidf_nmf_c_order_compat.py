from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import NMF

from oci.inference.sklearn_nmf_compat import (
    fit_transform_nmf_with_c_order_initial_w,
)


def _matrix():
    source = sparse.random(
        24,
        1_000,
        density=0.05,
        format="csr",
        dtype=np.float64,
        random_state=42,
    )
    selected = np.sort(
        np.random.default_rng(42).choice(
            source.shape[1],
            size=800,
            replace=False,
        )
    )
    weights = np.linspace(0.2, 2.0, len(selected))
    return source[:, selected].multiply(weights).tocsr()


def _model() -> NMF:
    return NMF(
        n_components=10,
        init="nndsvdar",
        solver="cd",
        beta_loss="frobenius",
        max_iter=20,
        tol=1e-4,
        random_state=3,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        verbose=0,
        shuffle=False,
    )


def test_c_order_compatibility_is_identical_on_working_runtime() -> None:
    matrix = _matrix()
    reference = _model()
    try:
        expected = reference.fit_transform(matrix)
    except ValueError as exc:
        if "C-contiguous" not in str(exc):
            raise
        pytest.skip("installed sklearn/NumPy combination requires the compatibility path")

    subject = _model()
    observed = fit_transform_nmf_with_c_order_initial_w(subject, matrix)

    assert type(subject) is NMF
    assert subject.init == "nndsvdar"
    assert subject.get_params(deep=False) == reference.get_params(deep=False)
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(subject.components_, reference.components_)
    np.testing.assert_array_equal(
        subject.transform(matrix),
        reference.transform(matrix),
    )


def test_c_order_compatibility_normalizes_fortran_initial_w(
    monkeypatch,
) -> None:
    import oci.inference.sklearn_nmf_compat as subject_module

    matrix = _matrix()
    original = subject_module._initialize_nmf
    observed_layouts: list[bool] = []

    def fortran_initial_w(*args, **kwargs):
        initial_w, initial_h = original(*args, **kwargs)
        initial_w = np.asfortranarray(initial_w)
        assert initial_w.flags.f_contiguous
        return initial_w, initial_h

    original_fit_transform = NMF.fit_transform

    def record_layout(self, matrix, W=None, H=None):
        assert W is not None
        observed_layouts.append(bool(W.flags.c_contiguous))
        return original_fit_transform(self, matrix, W=W, H=H)

    monkeypatch.setattr(
        subject_module,
        "_initialize_nmf",
        fortran_initial_w,
    )
    monkeypatch.setattr(NMF, "fit_transform", record_layout)

    model = _model()
    values = fit_transform_nmf_with_c_order_initial_w(model, matrix)

    assert observed_layouts == [True]
    assert values.shape == (matrix.shape[0], model.n_components)
    assert model.init == "nndsvdar"
