import json

import pandas as pd
import pytest

from oci.inference.all_evidence_fusion import SOURCE_TEXT_TEMPORAL_POLICY
from oci.inference.extraction_grounding_diagnostics import (
    build_extraction_grounding_diagnostics,
)
from oci.inference.post_extraction_scientific_policy import (
    ExtractionGroundingPolicy,
)


def _grounding_policy(**overrides):
    values = {
        "anchor_group_selection": "all_source_attested_unbounded",
        "maximum_group_span_chars": 96,
        "anchor_value_window_chars": 96,
        "category_assertion_prefix_chars": 64,
        "unit_window_min_chars": 12,
        "unit_window_max_chars": 32,
        "unit_window_divisor": 3,
        "minimum_evaluable_rows": 3,
        "maximum_alternative_category_only_rate": 0.50,
        "unsupported_value_warning_rate": 0.25,
        "minimum_unit_support_rate": 0.50,
    }
    values.update(overrides)
    return ExtractionGroundingPolicy(**values)


def _continuous_frame(values, *, name="orbital_flux"):
    return pd.DataFrame(
        {
            f"explicit_feat_{name}": values,
            f"explicit_feat_{name}_missing": [False] * len(values),
        }
    )


def _orbital_flux_spec():
    return {
        "name": "orbital_flux",
        "type": "continuous",
        "roles": ["confounder"],
        "description": "Orbital flux in quanta/turn.",
    }


def _categorical_frame(name, values, *, missing=None):
    missing = [False] * len(values) if missing is None else missing
    return pd.DataFrame(
        {
            f"explicit_feat_{name}": values,
            f"explicit_feat_{name}_missing": missing,
        }
    )


def _signal_mode_spec(*, aliases=False):
    spec = {
        "name": "signal_mode",
        "type": "categorical",
        "categories": ["amber", "violet"],
        "roles": ["effect_modifier"],
        "description": "Signal mode selected by the device.",
    }
    if aliases:
        spec["value_aliases"] = {"amber": ["ochre"]}
    return spec


def test_later_wording_is_grounded_without_temporal_eligibility_check():
    texts = [
        "Initial record: orbital flux 41 quanta/turn. <new_note> The intervention began. "
        "At a later observation after intervention, orbital flux 43 quanta/turn was recorded."
    ] * 4
    row = build_extraction_grounding_diagnostics(
        _continuous_frame([43.0] * 4), texts, [_orbital_flux_spec()]
    )[0]

    assert row["passed"] is True
    assert row["hard_failures"] == []
    assert row["value_grounding"]["supported_row_count"] == 4
    assert row["value_grounding"]["supported_rate"] == 1.0
    assert "temporal_correctness" not in row
    assert not any("time" in warning for warning in row["warnings"])
    policy = row["source_text_temporal_policy"]
    assert policy["policy"] == SOURCE_TEXT_TEMPORAL_POLICY
    assert policy["temporal_boundary_enforced"] is False
    assert policy["post_treatment_semantic_filtering_enabled"] is False


def test_semantic_timepoint_wording_does_not_change_grounding_eligibility():
    texts = [
        "Initially, orbital flux was 41 quanta/turn.",
        "At a later observation, the record restates orbital flux as 41 quanta/turn.",
        "After intervention, orbital flux 41 quanta/turn was documented.",
    ]
    row = build_extraction_grounding_diagnostics(
        _continuous_frame([41.0] * 3), texts, [_orbital_flux_spec()]
    )[0]

    assert row["passed"] is True
    assert row["value_grounding"]["supported_row_count"] == 3
    assert row["unit_alignment"]["expected_unit"] == "quanta/turn"


def test_categorical_alias_grounding_is_contract_derived_and_aggregate_only():
    frame = _categorical_frame("signal_mode", ["amber"] * 3)
    row = build_extraction_grounding_diagnostics(
        frame,
        ["After an intervention, signal mode: ochre."] * 3,
        [_signal_mode_spec(aliases=True)],
    )[0]

    assert row["passed"] is True
    assert row["value_grounding"]["alias_supported_row_count"] == 3
    serialized = json.dumps(row)
    assert "After an intervention, signal mode: ochre" not in serialized
    assert row["raw_note_text_exposed"] is False


def test_missing_categorical_alias_opportunity_is_revision_guidance_only():
    frame = _categorical_frame("signal_mode", [None] * 3, missing=[True] * 3)
    frame["_oci_row_id"] = [91001, 91002, 91003]
    row = build_extraction_grounding_diagnostics(
        frame,
        ["Sensitive phrase: signal mode was ochre."] * 3,
        [_signal_mode_spec(aliases=True)],
    )[0]

    alignment = row["categorical_ontology_alignment"]
    assert row["passed"] is True
    assert alignment["missing_single_declared_category_supported_row_count"] == 3
    assert row["revision_guidance"] == ["review_missingness_logic_and_declared_alias_coverage"]
    serialized = json.dumps(row)
    assert "Sensitive phrase" not in serialized
    assert "91001" not in serialized


@pytest.mark.parametrize(
    "text",
    [
        "Signal mode was violet.",
        "After an intervention, signal mode was violet.",
        "At a later observation after intervention, signal mode was violet.",
    ],
)
def test_repeated_alternative_category_is_ontology_failure_independent_of_timing(text):
    row = build_extraction_grounding_diagnostics(
        _categorical_frame("signal_mode", ["amber"] * 3),
        [text] * 3,
        [_signal_mode_spec()],
    )[0]

    alignment = row["categorical_ontology_alignment"]
    assert row["hard_failures"] == ["alternative_category_only_value_support"]
    assert alignment["locally_grounded_alternative_category_only_supported_row_count"] == 3
    assert alignment["locally_grounded_evaluable_row_count"] == 3


@pytest.mark.parametrize(
    "text",
    [
        "Signal mode was not violet.",
        "Calibration was pending whether signal mode was violet.",
    ],
)
def test_negated_or_hypothetical_category_is_not_asserted_alternative(text):
    row = build_extraction_grounding_diagnostics(
        _categorical_frame("signal_mode", ["amber"] * 3),
        [text] * 3,
        [_signal_mode_spec()],
    )[0]

    alignment = row["categorical_ontology_alignment"]
    assert row["passed"] is True
    assert alignment["alternative_category_only_supported_row_count"] == 0
    assert alignment["locally_grounded_evaluable_row_count"] == 0


def test_conflicting_categories_near_anchor_are_revision_warning_not_failure():
    frame = _categorical_frame("signal_mode", [None] * 3, missing=[True] * 3)
    row = build_extraction_grounding_diagnostics(
        frame,
        ["Signal mode was reported as both amber and violet."] * 3,
        [_signal_mode_spec()],
    )[0]

    alignment = row["categorical_ontology_alignment"]
    assert row["passed"] is True
    assert alignment["conflicting_multiple_categories_supported_row_count"] == 3
    assert row["revision_guidance"] == ["review_category_mutual_exclusivity_and_aliases"]


def test_grounding_validates_exact_text_alignment():
    with pytest.raises(ValueError, match="one exact string"):
        build_extraction_grounding_diagnostics(
            _continuous_frame([41.0, 42.0]), ["orbital flux 41"], [_orbital_flux_spec()]
        )


def test_numeric_value_must_share_a_tight_clause_with_conjunctive_anchor():
    row = build_extraction_grounding_diagnostics(
        _continuous_frame([43.0] * 3),
        ["Orbital flux was unavailable. Batch 43 was entered in error."] * 3,
        [_orbital_flux_spec()],
    )[0]

    assert row["anchor_detected_row_count"] == 3
    assert row["value_grounding"]["unsupported_row_count"] == 3


def test_contract_declared_unit_is_bound_to_matched_value_window():
    text = (
        "Orbital flux 43 crates was recorded, followed by extensive unrelated narrative "
        "that only much later names quanta/turn."
    )
    row = build_extraction_grounding_diagnostics(
        _continuous_frame([43.0] * 3), [text] * 3, [_orbital_flux_spec()]
    )[0]

    assert row["passed"] is True
    assert row["unit_alignment"]["evaluable_row_count"] == 3
    assert row["unit_alignment"]["supported_rate"] == 0.0
    assert "expected_unit_not_consistently_supported" in row["warnings"]


def test_source_unit_is_not_guessed_without_explicit_contract_unit_syntax():
    spec = {
        **_orbital_flux_spec(),
        "description": "The orbital flux reading.",
    }
    row = build_extraction_grounding_diagnostics(
        _continuous_frame([43.0] * 3),
        ["Orbital flux was 43 widgets."] * 3,
        [spec],
    )[0]

    assert row["value_grounding"]["supported_row_count"] == 3
    assert row["unit_alignment"] == {
        "expected_unit": None,
        "evaluable_row_count": 0,
        "supported_rate": None,
        "unit_bound_to_matched_value_window": True,
    }


def test_explicit_three_letter_acronym_is_eligible_when_source_attested():
    spec = {
        "name": "zog_level",
        "type": "continuous",
        "roles": ["confounder"],
        "description": "ZOG level in quanta/turn.",
    }
    row = build_extraction_grounding_diagnostics(
        _continuous_frame([7.0] * 3, name="zog_level"),
        ["The zog level was 7 quanta/turn."] * 3,
        [spec],
    )[0]

    assert row["contract_anchor_group_count"] >= 2
    assert row["anchor_detected_row_count"] == 3
    assert row["value_grounding"]["supported_row_count"] == 3


def test_ordinary_short_token_cannot_partially_anchor_a_multitoken_name():
    spec = {
        "name": "zog_cycle_rate",
        "type": "continuous",
        "roles": ["confounder"],
        "description": "The zog cycle rate in quanta/turn.",
    }
    row = build_extraction_grounding_diagnostics(
        _continuous_frame([7.0] * 3, name="zog_cycle_rate"),
        ["A zog reading was 7 quanta/turn."] * 3,
        [spec],
    )[0]

    assert row["contract_anchor_group_count"] == 0
    assert row["anchor_detected_row_count"] == 0
    assert row["value_grounding"]["unsupported_row_count"] == 3


def test_description_acronym_can_anchor_without_name_vocabulary_in_source():
    spec = {
        "name": "resonance_index",
        "type": "continuous",
        "roles": ["effect_modifier"],
        "description": "ZQ resonance measurement in quanta/turn.",
    }
    row = build_extraction_grounding_diagnostics(
        _continuous_frame([7.0] * 3, name="resonance_index"),
        ["ZQ was 7 quanta/turn."] * 3,
        [spec],
    )[0]

    assert row["contract_anchor_group_count"] == 1
    assert row["contract_anchor_token_count"] == 1
    assert row["value_grounding"]["supported_row_count"] == 3


def test_digit_identifier_separator_tolerance_is_generic_and_source_attested():
    spec = {
        "name": "qx7_state",
        "type": "categorical",
        "categories": ["amber", "violet"],
        "roles": ["effect_modifier"],
        "description": "QX7 state selected by the device.",
    }
    row = build_extraction_grounding_diagnostics(
        _categorical_frame("qx7_state", ["amber"] * 3),
        ["QX-7 state was amber."] * 3,
        [spec],
    )[0]

    assert row["anchor_detected_row_count"] == 3
    assert row["value_grounding"]["supported_row_count"] == 3


def test_unattested_short_contract_token_cannot_create_an_anchor():
    spec = {
        "name": "zog_level",
        "type": "continuous",
        "roles": ["confounder"],
        "description": "ZOG level in quanta/turn.",
    }
    row = build_extraction_grounding_diagnostics(
        _continuous_frame([7.0] * 3, name="zog_level"),
        ["An unrelated batch was 7 quanta/turn."] * 3,
        [spec],
    )[0]

    assert row["contract_anchor_group_count"] == 0
    assert row["contract_anchor_token_count"] == 0
    assert row["anchor_detected_row_count"] == 0
    assert row["value_grounding"]["unsupported_row_count"] == 3
    assert "contract_anchor_not_discriminative" in row["warnings"]


def test_all_source_attested_anchor_groups_are_retained_without_top_k_cap():
    spec = {
        "name": "qx1_qx2_qx3_qx4_qx5",
        "type": "continuous",
        "roles": ["confounder"],
        "description": "QX1 QX2 QX3 QX4 QX5 measurement in quanta/turn.",
    }
    text = "QX1 QX2 QX3 QX4 QX5 measurement was 7 quanta/turn."
    row = build_extraction_grounding_diagnostics(
        _continuous_frame([7.0] * 3, name=spec["name"]),
        [text] * 3,
        [spec],
        policy=_grounding_policy(),
    )[0]

    # One conjunctive name signature plus every independently attested
    # digit-bearing identifier; no fixed four-anchor slice is permitted.
    assert row["contract_anchor_group_count"] == 6
    assert row["value_grounding"]["supported_row_count"] == 3


def test_configured_grounding_warning_threshold_changes_diagnostic():
    frame = _continuous_frame([7.0] * 4)
    texts = [
        "Orbital flux was not recorded.",
        "Orbital flux was 7 quanta/turn.",
        "Orbital flux was not recorded.",
        "Orbital flux was 7 quanta/turn.",
    ]
    strict = build_extraction_grounding_diagnostics(
        frame,
        texts,
        [_orbital_flux_spec()],
        policy=_grounding_policy(unsupported_value_warning_rate=0.25),
    )[0]
    permissive = build_extraction_grounding_diagnostics(
        frame,
        texts,
        [_orbital_flux_spec()],
        policy=_grounding_policy(unsupported_value_warning_rate=0.75),
    )[0]

    assert "many_values_not_lexically_grounded_to_contract" in strict["warnings"]
    assert "many_values_not_lexically_grounded_to_contract" not in permissive["warnings"]
