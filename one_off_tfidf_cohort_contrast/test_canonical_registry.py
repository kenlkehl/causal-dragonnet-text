"""Focused tests for feature harmonization and bounded extraction planning."""

import argparse

import pytest

from one_off_tfidf_cohort_contrast.build_canonical_feature_registry import (
    normalize_harmonization_response,
    validate_harmonization_response,
)
from one_off_tfidf_cohort_contrast.extract_canonical_features import (
    HARD_MAX_VARIABLES_PER_REQUEST,
    extraction_groups,
    validate_args,
    validate_values,
)


def _feature(index: int, domain: str = "laboratory_vitals") -> dict:
    return {
        "canonical_id": f"{domain}__feature_{index}",
        "canonical_name": f"feature_{index}",
        "clinical_domain": domain,
        "parent_object": "panel_a" if index % 2 else "panel_b",
        "data_type": "continuous",
        "description": f"Feature {index}",
        "action": "extract",
    }


def test_extraction_groups_pack_without_exceeding_hard_cap() -> None:
    groups = extraction_groups([_feature(index) for index in range(23)], 10)
    assert [group["variable_count"] for group in groups] == [10, 10, 3]
    assert max(group["variable_count"] for group in groups) == 10
    assert all(group["clinical_domain"] == "laboratory_vitals" for group in groups)


def test_configuration_rejects_more_than_ten_variables() -> None:
    args = argparse.Namespace(
        variables_per_request=HARD_MAX_VARIABLES_PER_REQUEST + 1,
        concurrency=1,
        max_tokens=100,
        max_text_length=100,
        max_attempts=1,
        checkpoint_every=1,
        limit_rows=None,
        timeout=1.0,
    )
    with pytest.raises(ValueError, match="larger requests are forbidden"):
        validate_args(args)


def test_literal_drop_mapping_gets_defined_provenance_record() -> None:
    candidates = [
        {
            "candidate_id": "C001",
            "name": "Template fragment",
            "all_exact_variants": ["Template fragment"],
        }
    ]
    parsed = {
        "canonical_features": [],
        "candidate_mappings": [
            {
                "candidate_id": "C001",
                "local_id": "drop",
                "relation": "drop",
                "rationale": "Not a clinical variable",
            }
        ],
    }
    normalized = normalize_harmonization_response(parsed, candidates)
    features, mappings = validate_harmonization_response(normalized, ["C001"])
    assert len(features) == 1
    assert features[0]["action"] == "drop"
    assert mappings[0]["local_id"] == features[0]["local_id"]


def test_binary_presence_accepts_severity_as_present() -> None:
    feature = {
        "canonical_id": "symptoms__appetite_loss",
        "data_type": "binary",
        "categories": ["absent", "present", "unknown"],
    }
    values, missing, issues = validate_values(
        {"symptoms__appetite_loss": "mild"}, [feature]
    )
    assert values["symptoms__appetite_loss"] == "present"
    assert not missing["symptoms__appetite_loss"]
    assert not issues
