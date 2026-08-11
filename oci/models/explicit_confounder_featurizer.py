"""Compatibility aliases for the role-aware explicit-feature featurizer."""

from .explicit_feature_featurizer import (
    ExplicitFeatureFeaturizer,
    get_raw_explicit_features,
)

ExplicitConfounderFeaturizer = ExplicitFeatureFeaturizer
get_raw_confounder_features = get_raw_explicit_features

__all__ = [
    "ExplicitConfounderFeaturizer",
    "get_raw_confounder_features",
]
