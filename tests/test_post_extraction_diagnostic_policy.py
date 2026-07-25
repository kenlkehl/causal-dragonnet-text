from __future__ import annotations

import pandas as pd

from oci.inference.all_evidence_post_extraction_review import (
    build_extraction_quality_diagnostics,
    build_redundancy_diagnostics,
)
from oci.inference.post_extraction_scientific_policy import (
    ExtractionQualityPolicy,
    ExtractionRedundancyPolicy,
)


def _continuous(name: str) -> dict[str, object]:
    return {
        "name": name,
        "type": "continuous",
        "roles": ["confounder"],
        "description": f"Documented {name} measurement.",
    }


def _quality_policy(**overrides) -> ExtractionQualityPolicy:
    values = {
        "minimum_coverage": 0.05,
        "maximum_unknown_category_rate": 0.05,
        "continuous_outlier_minimum_rows": 8,
        "continuous_outlier_iqr_multiplier": 6.0,
        "continuous_outlier_warning_rate": 0.10,
        "fold_coverage_range_warning": 0.35,
        "fold_continuous_scale_epsilon": 1e-8,
    }
    values.update(overrides)
    return ExtractionQualityPolicy(**values)


def _redundancy_policy(**overrides) -> ExtractionRedundancyPolicy:
    values = {
        "association_threshold": 0.80,
        "missingness_jaccard_threshold": 0.90,
        "minimum_pairwise_complete_rows": 3,
    }
    values.update(overrides)
    return ExtractionRedundancyPolicy(**values)


def test_configured_quality_coverage_threshold_changes_failure() -> None:
    frame = pd.DataFrame(
        {
            "explicit_feat_marker": [1.0, 2.0, None, None],
            "explicit_feat_marker_missing": [False, False, True, True],
        }
    )
    strict = build_extraction_quality_diagnostics(
        frame,
        [_continuous("marker")],
        fold_ids=[1, 1, 2, 2],
        policy=_quality_policy(minimum_coverage=0.75),
    )["features"][0]
    permissive = build_extraction_quality_diagnostics(
        frame,
        [_continuous("marker")],
        fold_ids=[1, 1, 2, 2],
        policy=_quality_policy(minimum_coverage=0.25),
    )["features"][0]

    assert "coverage_below_minimum" in strict["hard_failures"]
    assert "coverage_below_minimum" not in permissive["hard_failures"]


def test_configured_redundancy_threshold_and_row_floor_are_binding() -> None:
    frame = pd.DataFrame(
        {
            "explicit_feat_left": [0.0, 1.0, 2.0, 3.0],
            "explicit_feat_left_missing": [False] * 4,
            "explicit_feat_right": [0.0, 1.0, 2.0, 2.0],
            "explicit_feat_right_missing": [False] * 4,
        }
    )
    specs = [_continuous("left"), _continuous("right")]

    included = build_redundancy_diagnostics(
        frame,
        specs,
        policy=_redundancy_policy(association_threshold=0.80),
    )
    high_threshold = build_redundancy_diagnostics(
        frame,
        specs,
        policy=_redundancy_policy(association_threshold=0.999),
    )
    insufficient_rows = build_redundancy_diagnostics(
        frame,
        specs,
        policy=_redundancy_policy(
            association_threshold=0.80,
            minimum_pairwise_complete_rows=5,
        ),
    )

    assert len(included) == 1
    assert included[0]["redundancy_reasons"] == ["high_value_association"]
    assert high_threshold == []
    assert insufficient_rows == []
