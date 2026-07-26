"""Declared comparison policy for replayed neural numerical outputs.

Artifact bytes, schemas, identities, and discrete state remain exact.  This
module only governs the comparison between a freshly recomputed floating
neural output and the already-authenticated floating array stored in that
artifact.
"""

from __future__ import annotations

from typing import Any

import numpy as np


NEURAL_REPLAY_COMPARISON_POLICY = "allclose_and_exact_discrete_state_v1"


def validate_neural_replay_settings(
    *,
    policy: Any,
    relative_tolerance: Any,
    absolute_tolerance: Any,
) -> tuple[str, float, float]:
    """Validate one explicit, default-free neural replay policy."""

    normalized_policy = str(policy)
    if normalized_policy != NEURAL_REPLAY_COMPARISON_POLICY:
        raise ValueError("neural replay comparison policy is unsupported")
    tolerances: list[float] = []
    for name, value in (
        ("relative", relative_tolerance),
        ("absolute", absolute_tolerance),
    ):
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError(f"neural replay {name} tolerance must be numeric")
        normalized = float(value)
        if not np.isfinite(normalized) or normalized < 0.0:
            raise ValueError(
                f"neural replay {name} tolerance must be finite and non-negative"
            )
        tolerances.append(normalized)
    return normalized_policy, tolerances[0], tolerances[1]


def neural_float_arrays_within_tolerance(
    observed: Any,
    expected: Any,
    *,
    policy: Any,
    relative_tolerance: Any,
    absolute_tolerance: Any,
) -> bool:
    """Compare only floating values while keeping structural state exact."""

    _policy, rtol, atol = validate_neural_replay_settings(
        policy=policy,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    left = np.asarray(observed)
    right = np.asarray(expected)
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.dtype.kind != "f"
        or not np.array_equal(np.isfinite(left), np.isfinite(right))
    ):
        return False
    return bool(
        np.allclose(
            left,
            right,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    )


__all__ = [
    "NEURAL_REPLAY_COMPARISON_POLICY",
    "neural_float_arrays_within_tolerance",
    "validate_neural_replay_settings",
]
