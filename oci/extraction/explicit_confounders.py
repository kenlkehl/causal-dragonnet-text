"""Compatibility aliases for the role-aware explicit-feature extractor.

Explicit confounders are now ordinary :class:`ExplicitFeatureSpec` instances
whose roles include ``"confounder"``.  Keep the historical import names without
maintaining a second extraction implementation.
"""

from .explicit_features import (
    ExplicitFeatureValue,
    VLLMFeatureExtractor,
    build_extraction_prompt,
    extract_explicit_features,
    parse_extraction_response,
)

ExplicitConfounderValue = ExplicitFeatureValue
VLLMConfounderExtractor = VLLMFeatureExtractor
extract_explicit_confounders = extract_explicit_features

__all__ = [
    "ExplicitConfounderValue",
    "VLLMConfounderExtractor",
    "build_extraction_prompt",
    "extract_explicit_confounders",
    "parse_extraction_response",
]
