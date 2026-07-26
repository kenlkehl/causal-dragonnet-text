import numpy as np
import pytest

from oci.inference.neural_numerical_replay import (
    neural_float_arrays_within_tolerance,
)


def test_declared_neural_tolerance_preserves_exact_structure_and_finite_mask():
    expected = np.asarray([1.0, 2.0, np.nan], dtype=np.float64)
    within = np.asarray([1.0 + 5e-5, 2.0, np.nan], dtype=np.float64)
    outside = np.asarray([1.0 + 2e-3, 2.0, np.nan], dtype=np.float64)
    settings = {
        "policy": "allclose_and_exact_discrete_state_v1",
        "relative_tolerance": 1e-4,
        "absolute_tolerance": 1e-5,
    }

    assert neural_float_arrays_within_tolerance(within, expected, **settings)
    assert not neural_float_arrays_within_tolerance(outside, expected, **settings)
    assert not neural_float_arrays_within_tolerance(
        within.astype(np.float32),
        expected,
        **settings,
    )
    assert not neural_float_arrays_within_tolerance(
        np.asarray([1.0 + 5e-5, 2.0, 0.0], dtype=np.float64),
        expected,
        **settings,
    )
    with pytest.raises(ValueError, match="unsupported"):
        neural_float_arrays_within_tolerance(
            within,
            expected,
            **{**settings, "policy": "raw_byte_identity_v1"},
        )
