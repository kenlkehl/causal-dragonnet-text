"""Closed scientific settings for post-extraction review.

The portable workflow must never inherit diagnostic thresholds, lexical
windows, estimator solvers, or fallback behavior from library/source defaults.
This module deliberately provides no benchmark defaults.  A deployment must
construct the complete policy from its scientific specification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


POST_EXTRACTION_SCIENTIFIC_POLICY_VERSION = (
    "post_extraction_scientific_policy_v1"
)


def _closed_mapping(
    value: Mapping[str, Any],
    *,
    expected: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(
            f"{label} must be explicitly and exactly configured; "
            f"missing={missing}, extra={extra}"
        )
    return dict(value)


def _finite_float(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _integer(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _probability(value: Any, *, label: str) -> float:
    normalized = _finite_float(value, label=label)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return normalized


@dataclass(frozen=True)
class ExtractionQualityPolicy:
    minimum_coverage: float
    maximum_unknown_category_rate: float
    continuous_outlier_minimum_rows: int
    continuous_outlier_iqr_multiplier: float
    continuous_outlier_warning_rate: float
    fold_coverage_range_warning: float
    fold_continuous_scale_epsilon: float

    def __post_init__(self) -> None:
        for name in (
            "minimum_coverage",
            "maximum_unknown_category_rate",
            "continuous_outlier_warning_rate",
            "fold_coverage_range_warning",
        ):
            object.__setattr__(
                self,
                name,
                _probability(
                    getattr(self, name),
                    label=f"extraction_quality.{name}",
                ),
            )
        object.__setattr__(
            self,
            "continuous_outlier_minimum_rows",
            _integer(
                self.continuous_outlier_minimum_rows,
                label="extraction_quality.continuous_outlier_minimum_rows",
                minimum=1,
            ),
        )
        for name in (
            "continuous_outlier_iqr_multiplier",
            "fold_continuous_scale_epsilon",
        ):
            normalized = _finite_float(
                getattr(self, name),
                label=f"extraction_quality.{name}",
            )
            if normalized <= 0.0:
                raise ValueError(f"extraction_quality.{name} must be positive")
            object.__setattr__(self, name, normalized)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExtractionQualityPolicy":
        return cls(
            **_closed_mapping(
                value,
                expected=set(cls.__dataclass_fields__),
                label="extraction_quality",
            )
        )


@dataclass(frozen=True)
class ExtractionRedundancyPolicy:
    association_threshold: float
    missingness_jaccard_threshold: float
    minimum_pairwise_complete_rows: int

    def __post_init__(self) -> None:
        for name in ("association_threshold", "missingness_jaccard_threshold"):
            object.__setattr__(
                self,
                name,
                _probability(
                    getattr(self, name),
                    label=f"extraction_redundancy.{name}",
                ),
            )
        object.__setattr__(
            self,
            "minimum_pairwise_complete_rows",
            _integer(
                self.minimum_pairwise_complete_rows,
                label="extraction_redundancy.minimum_pairwise_complete_rows",
                minimum=2,
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ExtractionRedundancyPolicy":
        return cls(
            **_closed_mapping(
                value,
                expected=set(cls.__dataclass_fields__),
                label="extraction_redundancy",
            )
        )


@dataclass(frozen=True)
class ExtractionGroundingPolicy:
    anchor_group_selection: str
    maximum_group_span_chars: int
    anchor_value_window_chars: int
    category_assertion_prefix_chars: int
    unit_window_min_chars: int
    unit_window_max_chars: int
    unit_window_divisor: int
    minimum_evaluable_rows: int
    maximum_alternative_category_only_rate: float
    unsupported_value_warning_rate: float
    minimum_unit_support_rate: float

    def __post_init__(self) -> None:
        if self.anchor_group_selection != "all_source_attested_unbounded":
            raise ValueError(
                "extraction_grounding.anchor_group_selection must preserve all "
                "source-attested anchor groups"
            )
        for name, minimum in (
            ("maximum_group_span_chars", 1),
            ("anchor_value_window_chars", 1),
            ("category_assertion_prefix_chars", 1),
            ("unit_window_min_chars", 1),
            ("unit_window_max_chars", 1),
            ("unit_window_divisor", 1),
            ("minimum_evaluable_rows", 1),
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    label=f"extraction_grounding.{name}",
                    minimum=minimum,
                ),
            )
        if self.unit_window_min_chars > self.unit_window_max_chars:
            raise ValueError(
                "extraction_grounding.unit_window_min_chars cannot exceed "
                "unit_window_max_chars"
            )
        for name in (
            "maximum_alternative_category_only_rate",
            "unsupported_value_warning_rate",
            "minimum_unit_support_rate",
        ):
            object.__setattr__(
                self,
                name,
                _probability(
                    getattr(self, name),
                    label=f"extraction_grounding.{name}",
                ),
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ExtractionGroundingPolicy":
        return cls(
            **_closed_mapping(
                value,
                expected=set(cls.__dataclass_fields__),
                label="extraction_grounding",
            )
        )


@dataclass(frozen=True)
class ReviewEstimatorPolicy:
    standardization_scale_epsilon: float
    logistic_alpha_floor: float
    logistic_solver: str
    logistic_max_iter: int
    logistic_random_seed: int
    logistic_fit_intercept: bool
    logistic_class_weight: str | None
    binary_no_features_fallback: str
    binary_single_class_fallback: str
    binary_fit_failure_policy: str
    continuous_minimum_fit_rows: int
    continuous_degenerate_fallback: str
    effect_minimum_usable_rows: int
    effect_no_usable_fallback: str
    effect_degenerate_fallback: str
    ridge_solver: str
    ridge_fit_intercept: bool
    ridge_tolerance: float
    ridge_max_iter: int | None
    ridge_positive: bool
    ridge_random_seed: int | None

    def __post_init__(self) -> None:
        for name in ("standardization_scale_epsilon", "logistic_alpha_floor"):
            normalized = _finite_float(
                getattr(self, name),
                label=f"review_estimator.{name}",
            )
            if normalized <= 0.0:
                raise ValueError(f"review_estimator.{name} must be positive")
            object.__setattr__(self, name, normalized)
        if self.logistic_solver not in {
            "lbfgs",
            "liblinear",
            "newton-cg",
            "newton-cholesky",
            "sag",
            "saga",
        }:
            raise ValueError("review_estimator.logistic_solver is unsupported")
        object.__setattr__(
            self,
            "logistic_max_iter",
            _integer(
                self.logistic_max_iter,
                label="review_estimator.logistic_max_iter",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "logistic_random_seed",
            _integer(
                self.logistic_random_seed,
                label="review_estimator.logistic_random_seed",
                minimum=0,
            ),
        )
        if not isinstance(self.logistic_fit_intercept, bool):
            raise ValueError(
                "review_estimator.logistic_fit_intercept must be boolean"
            )
        if self.logistic_class_weight not in {None, "balanced"}:
            raise ValueError(
                "review_estimator.logistic_class_weight must be null or balanced"
            )
        for name, allowed in (
            ("binary_no_features_fallback", {"prevalence"}),
            ("binary_single_class_fallback", {"prevalence"}),
            ("binary_fit_failure_policy", {"abort", "prevalence"}),
            ("continuous_degenerate_fallback", {"mean"}),
            ("effect_no_usable_fallback", {"zero"}),
            ("effect_degenerate_fallback", {"weighted_mean"}),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"review_estimator.{name} must be one of {sorted(allowed)}"
                )
        for name in (
            "continuous_minimum_fit_rows",
            "effect_minimum_usable_rows",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    label=f"review_estimator.{name}",
                    minimum=1,
                ),
            )
        if self.ridge_solver not in {
            "auto",
            "svd",
            "cholesky",
            "lsqr",
            "sparse_cg",
            "sag",
            "saga",
            "lbfgs",
        }:
            raise ValueError("review_estimator.ridge_solver is unsupported")
        if not isinstance(self.ridge_fit_intercept, bool):
            raise ValueError(
                "review_estimator.ridge_fit_intercept must be boolean"
            )
        tolerance = _finite_float(
            self.ridge_tolerance,
            label="review_estimator.ridge_tolerance",
        )
        if tolerance <= 0.0:
            raise ValueError("review_estimator.ridge_tolerance must be positive")
        object.__setattr__(self, "ridge_tolerance", tolerance)
        if self.ridge_max_iter is not None:
            object.__setattr__(
                self,
                "ridge_max_iter",
                _integer(
                    self.ridge_max_iter,
                    label="review_estimator.ridge_max_iter",
                    minimum=1,
                ),
            )
        if not isinstance(self.ridge_positive, bool):
            raise ValueError("review_estimator.ridge_positive must be boolean")
        if self.ridge_random_seed is not None:
            object.__setattr__(
                self,
                "ridge_random_seed",
                _integer(
                    self.ridge_random_seed,
                    label="review_estimator.ridge_random_seed",
                    minimum=0,
                ),
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ReviewEstimatorPolicy":
        return cls(
            **_closed_mapping(
                value,
                expected=set(cls.__dataclass_fields__),
                label="review_estimator",
            )
        )


@dataclass(frozen=True)
class PostExtractionScientificPolicy:
    extraction_quality: ExtractionQualityPolicy
    extraction_redundancy: ExtractionRedundancyPolicy
    extraction_grounding: ExtractionGroundingPolicy
    review_estimator: ReviewEstimatorPolicy
    schema_version: str

    def __post_init__(self) -> None:
        if self.schema_version != POST_EXTRACTION_SCIENTIFIC_POLICY_VERSION:
            raise ValueError(
                "unsupported post-extraction scientific policy version"
            )
        expected_types = {
            "extraction_quality": ExtractionQualityPolicy,
            "extraction_redundancy": ExtractionRedundancyPolicy,
            "extraction_grounding": ExtractionGroundingPolicy,
            "review_estimator": ReviewEstimatorPolicy,
        }
        for name, expected_type in expected_types.items():
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__}")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "PostExtractionScientificPolicy":
        normalized = _closed_mapping(
            value,
            expected=set(cls.__dataclass_fields__),
            label="post_extraction_scientific_policy",
        )
        normalized["extraction_quality"] = ExtractionQualityPolicy.from_mapping(
            normalized["extraction_quality"]
        )
        normalized["extraction_redundancy"] = (
            ExtractionRedundancyPolicy.from_mapping(
                normalized["extraction_redundancy"]
            )
        )
        normalized["extraction_grounding"] = (
            ExtractionGroundingPolicy.from_mapping(
                normalized["extraction_grounding"]
            )
        )
        normalized["review_estimator"] = ReviewEstimatorPolicy.from_mapping(
            normalized["review_estimator"]
        )
        return cls(**normalized)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ExtractionGroundingPolicy",
    "ExtractionQualityPolicy",
    "ExtractionRedundancyPolicy",
    "POST_EXTRACTION_SCIENTIFIC_POLICY_VERSION",
    "PostExtractionScientificPolicy",
    "ReviewEstimatorPolicy",
]
