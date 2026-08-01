"""Narrow scikit-learn NMF layout compatibility for production TF-IDF.

scikit-learn 1.6 coordinate-descent NMF assumes that its initialized W array
is C-contiguous.  NumPy 2.4 can return the same NNDSVD values in Fortran order
for sufficiently rectangular sparse matrices, causing the Cython update to
reject the array before the first iteration.  This module preserves the
configured NMF algorithm and initialization values while making that implicit
memory-layout requirement explicit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition._nmf import _initialize_nmf


def fit_transform_nmf_with_c_order_initial_w(
    model: NMF,
    matrix: Any,
) -> np.ndarray:
    """Fit exact ``NMF`` after normalizing only CD initial-W memory order.

    The configured initializer, random state, objective, solver, and all
    numerical values are unchanged.  ``model.init`` is restored before this
    function returns so safe-artifact metadata continues to attest the actual
    scientific initializer rather than the temporary public ``custom`` route
    used to pass its values back into scikit-learn.
    """

    if type(model) is not NMF:
        raise TypeError("production TF-IDF requires exact sklearn NMF")
    if model.solver != "cd":
        return np.asarray(model.fit_transform(matrix))
    if model.init == "custom":
        raise ValueError(
            "production TF-IDF cannot normalize an unspecified custom NMF "
            "initialization"
        )
    if isinstance(model.n_components, bool) or not isinstance(
        model.n_components,
        (int, np.integer),
    ):
        raise TypeError(
            "production TF-IDF requires an explicit integer NMF component count"
        )

    configured_init = model.init
    initial_w, initial_h = _initialize_nmf(
        matrix,
        int(model.n_components),
        init=configured_init,
        random_state=model.random_state,
    )
    initial_w = np.ascontiguousarray(initial_w)
    model.init = "custom"
    try:
        transformed = model.fit_transform(
            matrix,
            W=initial_w,
            H=initial_h,
        )
    finally:
        model.init = configured_init
    return np.asarray(transformed)
