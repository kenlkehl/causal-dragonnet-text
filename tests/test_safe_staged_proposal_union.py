from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace

import pytest

from oci.inference.all_evidence_fusion import ALL_SOURCE_FAMILIES
from oci.inference.safe_staged_proposal_union import (
    SAFE_STAGED_PROPOSAL_UNION_INPUT_SCHEMA_VERSION,
    SAFE_STAGED_PROPOSAL_UNION_OUTPUT_SCHEMA_VERSION,
    SAFE_STAGED_PROPOSAL_UNION_POLICY_VERSION,
    SAFE_STAGED_PROPOSAL_SOURCE_FAMILIES,
    assert_safe_staged_proposal_union_identity,
    safe_staged_proposal_union,
    safe_staged_proposal_union_identity,
)


def test_isolated_source_family_vocabulary_matches_authoritative_fusion_boundary() -> None:
    assert SAFE_STAGED_PROPOSAL_SOURCE_FAMILIES == ALL_SOURCE_FAMILIES


def _spec(
    name: str,
    roles=("confounder",),
    *,
    kind: str = "continuous",
    description: str | None = None,
    categories=None,
    aliases=None,
):
    output = {
        "name": name,
        "type": kind,
        "roles": list(roles),
        "description": description or f"Baseline observable definition for {name}",
    }
    if categories is not None:
        output["categories"] = list(categories)
    if aliases is not None:
        output["value_aliases"] = copy.deepcopy(aliases)
    return output


def _candidate(
    candidate_id: str,
    spec,
    *,
    evidence=("evidence_0001",),
    families=("bow_nuisance",),
    occurrences=1,
):
    return {
        "candidate_id": candidate_id,
        "extraction_spec": copy.deepcopy(spec),
        "supporting_evidence_ids": list(evidence),
        "supporting_source_families": list(families),
        "validated_occurrence_count": occurrences,
    }


def test_exact_duplicates_consolidate_all_support_and_occurrences() -> None:
    spec = _spec("baseline_measure")
    first = _candidate(
        "candidate_0002",
        spec,
        evidence=("evidence_0002",),
        families=("htr_neural",),
        occurrences=3,
    )
    duplicate = _candidate(
        "candidate_0001",
        spec,
        evidence=("evidence_0001",),
        families=("bow_nuisance",),
    )

    result = safe_staged_proposal_union((first, duplicate))

    assert result.representative_candidate_ids == ("candidate_0002",)
    assert result.exact_duplicate_candidate_ids == ("candidate_0001",)
    assert result.compatible_role_merge_candidate_ids == ()
    assert result.omitted_conflict_candidate_ids == ()
    retained = result.candidates[0]
    assert retained.validated_occurrence_count == 4
    assert retained.supporting_evidence_ids == ("evidence_0001", "evidence_0002")
    assert retained.supporting_source_families == ("bow_nuisance", "htr_neural")
    result.verify(candidates=(first, duplicate))


def test_role_only_variants_merge_but_no_other_extraction_field_changes() -> None:
    confounder = _candidate(
        "candidate_0010",
        _spec("shared_measure", ("confounder",)),
        evidence=("evidence_0010",),
        families=("bow_nuisance",),
        occurrences=3,
    )
    modifier = _candidate(
        "candidate_0011",
        _spec("shared_measure", ("effect_modifier",)),
        evidence=("evidence_0011",),
        families=("neural_query_moments",),
        occurrences=2,
    )

    result = safe_staged_proposal_union((confounder, modifier))

    retained = result.candidates[0]
    assert retained.candidate_id == "candidate_0010"
    assert retained.extraction_spec == {
        "name": "shared_measure",
        "type": "continuous",
        "roles": ("confounder", "effect_modifier"),
        "description": "Baseline observable definition for shared_measure",
    }
    assert retained.supporting_evidence_ids == ("evidence_0010", "evidence_0011")
    assert retained.supporting_source_families == (
        "bow_nuisance",
        "neural_query_moments",
    )
    assert result.compatible_role_merge_candidate_ids == ("candidate_0011",)
    assert result.conflicts == ()


def test_continuous_and_categorical_same_name_are_incompatible() -> None:
    continuous = _candidate(
        "candidate_0020",
        _spec("typed_measure", ("confounder",)),
        evidence=("evidence_continuous",),
        families=("bow_nuisance",),
        occurrences=2,
    )
    categorical = _candidate(
        "candidate_0021",
        _spec(
            "typed_measure",
            ("effect_modifier",),
            kind="categorical",
            categories=("absent", "present"),
        ),
        evidence=("evidence_categorical",),
        families=("neural_query_moments",),
    )

    result = safe_staged_proposal_union((continuous, categorical))

    retained = result.candidates[0]
    assert retained.candidate_id == "candidate_0020"
    assert retained.extraction_spec["type"] == "continuous"
    assert retained.extraction_spec["roles"] == ("confounder",)
    assert retained.supporting_evidence_ids == ("evidence_continuous",)
    assert retained.supporting_source_families == ("bow_nuisance",)
    assert retained.validated_occurrence_count == 2
    assert result.omitted_conflict_candidate_ids == ("candidate_0021",)
    assert result.conflicts[0].differing_non_role_fields == ("type", "categories")


def test_category_mismatch_uses_generic_strength_without_cross_import() -> None:
    weaker = _candidate(
        "candidate_0030",
        _spec(
            "category_measure",
            kind="categorical",
            categories=("low", "high"),
        ),
        evidence=("evidence_0030",),
        families=("bow_nuisance",),
    )
    broader = _candidate(
        "candidate_0031",
        _spec(
            "category_measure",
            ("effect_modifier",),
            kind="categorical",
            categories=("absent", "present"),
        ),
        evidence=("evidence_0031",),
        families=("htr_neural", "neural_query_moments"),
    )

    result = safe_staged_proposal_union((weaker, broader))

    retained = result.candidates[0]
    assert retained.candidate_id == "candidate_0031"
    assert retained.extraction_spec["categories"] == ("absent", "present")
    assert retained.extraction_spec["roles"] == ("effect_modifier",)
    assert retained.supporting_evidence_ids == ("evidence_0031",)
    assert result.omitted_conflict_candidate_ids == ("candidate_0030",)
    assert result.conflicts[0].differing_non_role_fields == ("categories",)


def test_temporal_description_mismatch_is_never_role_merged() -> None:
    baseline = _candidate(
        "candidate_0040",
        _spec(
            "timed_measure",
            description="Observed before assignment at the baseline encounter",
        ),
        evidence=("evidence_baseline",),
    )
    historical = _candidate(
        "candidate_0041",
        _spec(
            "timed_measure",
            ("effect_modifier",),
            description="Most recent historical observation before assignment",
        ),
        evidence=("evidence_historical",),
    )

    result = safe_staged_proposal_union((baseline, historical))

    retained = result.candidates[0]
    assert retained.candidate_id == "candidate_0040"
    assert retained.extraction_spec["roles"] == ("confounder",)
    assert retained.supporting_evidence_ids == ("evidence_baseline",)
    assert result.conflicts[0].differing_non_role_fields == ("description",)


def test_value_alias_mismatch_is_a_conflict() -> None:
    categories = ("absent", "present")
    left = _candidate(
        "candidate_0050",
        _spec(
            "alias_measure",
            kind="categorical",
            categories=categories,
            aliases={"absent": ["negative"], "present": ["positive"]},
        ),
        evidence=("evidence_0050",),
    )
    right = _candidate(
        "candidate_0051",
        _spec(
            "alias_measure",
            ("effect_modifier",),
            kind="categorical",
            categories=categories,
            aliases={"absent": ["none"], "present": ["positive"]},
        ),
        evidence=("evidence_0051", "evidence_0052"),
    )

    result = safe_staged_proposal_union((left, right))

    assert result.representative_candidate_ids == ("candidate_0051",)
    assert result.omitted_conflict_candidate_ids == ("candidate_0050",)
    assert result.conflicts[0].differing_non_role_fields == ("value_aliases",)
    retained = result.candidates[0]
    assert retained.supporting_evidence_ids == ("evidence_0051", "evidence_0052")
    assert retained.extraction_spec["roles"] == ("effect_modifier",)


def test_strength_order_is_occurrence_then_family_then_evidence_then_opaque_id() -> None:
    rows = [
        _candidate(
            "candidate_occurrence",
            _spec("rank_occurrence", description="Variant one"),
            occurrences=2,
        ),
        _candidate(
            "candidate_broad_but_rare",
            _spec("rank_occurrence", description="Variant two"),
            evidence=("evidence_a", "evidence_b", "evidence_c"),
            families=("bow_nuisance", "htr_neural", "neural_query_moments"),
        ),
        _candidate(
            "candidate_family_narrow",
            _spec("rank_family", description="Variant one"),
            evidence=("evidence_d", "evidence_e", "evidence_f"),
            families=("bow_nuisance",),
        ),
        _candidate(
            "candidate_family_wide",
            _spec("rank_family", description="Variant two"),
            evidence=("evidence_g",),
            families=("bow_nuisance", "htr_neural"),
        ),
        _candidate(
            "candidate_evidence_one",
            _spec("rank_evidence", description="Variant one"),
            evidence=("evidence_h",),
        ),
        _candidate(
            "candidate_evidence_two",
            _spec("rank_evidence", description="Variant two"),
            evidence=("evidence_i", "evidence_j"),
        ),
        _candidate(
            "candidate_tie_b",
            _spec("rank_tie", description="Variant one"),
        ),
        _candidate(
            "candidate_tie_a",
            _spec("rank_tie", description="Variant two"),
        ),
    ]

    result = safe_staged_proposal_union(rows)

    assert set(result.representative_candidate_ids) == {
        "candidate_occurrence",
        "candidate_family_wide",
        "candidate_evidence_two",
        "candidate_tie_a",
    }


def test_full_candidate_id_partition_and_conflict_metadata_are_exact() -> None:
    base = _spec("partition_measure", ("confounder",))
    representative = _candidate(
        "candidate_0100",
        base,
        evidence=("evidence_0100",),
        occurrences=4,
    )
    exact = _candidate(
        "candidate_0101",
        base,
        evidence=("evidence_0101",),
    )
    role = _candidate(
        "candidate_0102",
        _spec("partition_measure", ("effect_modifier",)),
        evidence=("evidence_0102",),
    )
    conflict = _candidate(
        "candidate_0103",
        _spec("partition_measure", description="A conflicting baseline definition"),
        evidence=("evidence_0103",),
    )
    conflict_duplicate = _candidate(
        "candidate_0104",
        _spec("partition_measure", description="A conflicting baseline definition"),
        evidence=("evidence_0104",),
    )
    independent = _candidate("candidate_0200", _spec("independent_measure"))
    rows = (
        representative,
        exact,
        role,
        conflict,
        conflict_duplicate,
        independent,
    )

    result = safe_staged_proposal_union(rows)

    assert result.representative_candidate_ids == (
        "candidate_0100",
        "candidate_0200",
    )
    assert result.exact_duplicate_candidate_ids == ("candidate_0101",)
    assert result.compatible_role_merge_candidate_ids == ("candidate_0102",)
    assert result.omitted_conflict_candidate_ids == (
        "candidate_0103",
        "candidate_0104",
    )
    partitions = (
        set(result.representative_candidate_ids),
        set(result.exact_duplicate_candidate_ids),
        set(result.compatible_role_merge_candidate_ids),
        set(result.omitted_conflict_candidate_ids),
    )
    assert set.union(*partitions) == {row["candidate_id"] for row in rows}
    assert sum(len(part) for part in partitions) == len(set.union(*partitions))
    assert result.conflicts[0].omitted_candidate_ids == (
        "candidate_0103",
        "candidate_0104",
    )
    assert result.conflicts[0].retained_candidate_id == "candidate_0100"
    assert len(result.conflicts[0].conflict_id) == len("conflict_") + 64
    result.verify(candidates=rows)


def test_canonical_hashes_ignore_mapping_and_set_like_input_order() -> None:
    categories = ("absent", "present")
    original = _candidate(
        "candidate_0300",
        _spec(
            "canonical_measure",
            ("confounder", "effect_modifier"),
            kind="categorical",
            categories=categories,
            aliases={"absent": ["negative"], "present": ["positive"]},
        ),
        evidence=("evidence_b", "evidence_a"),
        families=("neural_query_moments", "bow_nuisance"),
    )
    reordered = {
        "validated_occurrence_count": 1,
        "supporting_source_families": ["bow_nuisance", "neural_query_moments"],
        "supporting_evidence_ids": ["evidence_a", "evidence_b"],
        "extraction_spec": {
            "description": "Baseline observable definition for canonical_measure",
            "roles": ["effect_modifier", "confounder"],
            "value_aliases": {"present": ["positive"], "absent": ["negative"]},
            "categories": ["absent", "present"],
            "type": "categorical",
            "name": "canonical_measure",
        },
        "candidate_id": "candidate_0300",
    }

    first = safe_staged_proposal_union((original,))
    second = safe_staged_proposal_union((reordered,))

    assert first.input_sha256 == second.input_sha256
    assert first.output_sha256 == second.output_sha256
    assert first.canonical_output_json == second.canonical_output_json
    assert len(first.input_sha256) == 64
    assert len(first.output_sha256) == 64


def test_output_is_immutable_detached_and_tampering_is_detected() -> None:
    row = _candidate("candidate_0400", _spec("immutable_measure"))
    result = safe_staged_proposal_union((row,))

    with pytest.raises(FrozenInstanceError):
        result.input_sha256 = "0" * 64
    with pytest.raises(TypeError):
        result.candidates[0].extraction_spec["description"] = "changed"
    detached = result.as_dict()
    detached["candidates"][0]["extraction_spec"]["description"] = "changed"
    assert result.candidates[0].extraction_spec["description"] != "changed"
    result.verify(candidates=(row,))

    modified_envelope = replace(
        result,
        _canonical_output_json=result.canonical_output_json + " ",
    )
    with pytest.raises(ValueError, match="canonical output envelope"):
        modified_envelope.verify()
    modified_hash = replace(result, output_sha256="0" * 64)
    with pytest.raises(ValueError, match="output hash mismatch"):
        modified_hash.verify()
    modified_accounting = replace(result, omitted_conflict_candidate_ids=("candidate_0400",))
    with pytest.raises(ValueError, match="full disjoint partition"):
        modified_accounting.verify()


def test_identity_and_schema_versions_are_explicit_and_attestable() -> None:
    identity = safe_staged_proposal_union_identity()
    assert identity.policy_version == SAFE_STAGED_PROPOSAL_UNION_POLICY_VERSION
    assert identity.input_schema_version == SAFE_STAGED_PROPOSAL_UNION_INPUT_SCHEMA_VERSION
    assert identity.output_schema_version == SAFE_STAGED_PROPOSAL_UNION_OUTPUT_SCHEMA_VERSION
    assert len(identity.implementation_sha256) == 64
    assert_safe_staged_proposal_union_identity(identity.implementation_sha256)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        assert_safe_staged_proposal_union_identity("0" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        assert_safe_staged_proposal_union_identity("NOT-A-HASH")


@pytest.mark.parametrize(
    "mutation, error",
    [
        (lambda row: row.update({"raw_document_text": "not permitted"}), "closed schema"),
        (lambda row: row.update({"outcome_label": 1}), "closed schema"),
        (
            lambda row: row["extraction_spec"].update({"oracle_value": "not permitted"}),
            "unsupported fields",
        ),
        (
            lambda row: row["extraction_spec"].update({"categories": ["a", "b"]}),
            "continuous spec",
        ),
        (
            lambda row: row.update({"supporting_source_families": ["unknown_family"]}),
            "unknown families",
        ),
        (lambda row: row.update({"validated_occurrence_count": True}), "positive integer"),
    ],
)
def test_closed_schema_rejects_raw_text_labels_oracle_and_malformed_values(
    mutation, error
) -> None:
    row = _candidate("candidate_0500", _spec("schema_measure"))
    mutation(row)
    with pytest.raises(ValueError, match=error):
        safe_staged_proposal_union((row,))


def test_partial_alias_map_from_a_valid_live_style_contract_is_preserved() -> None:
    live_style = _candidate(
        "candidate_0600",
        _spec(
            "machine_operating_grade",
            ("confounder", "effect_modifier"),
            kind="categorical",
            description="Machine operating grade on a four-level synthetic scale.",
            categories=("0", "1", "2", "3"),
            aliases={
                "0": ["grade 0"],
                "2": ["grade 2", "level 2"],
                "3": ["grade 3", "operating grade 3"],
            },
        ),
        evidence=("evidence_0002", "evidence_0016"),
        families=("bow_nuisance", "matched_pair_uplift", "tfidf_topics"),
    )

    result = safe_staged_proposal_union((live_style,))

    assert result.candidates[0].extraction_spec["value_aliases"] == {
        "0": ("grade 0",),
        "2": ("grade 2", "level 2"),
        "3": ("grade 3", "operating grade 3"),
    }
    assert "1" not in result.candidates[0].extraction_spec["value_aliases"]
    result.verify(candidates=(live_style,))


@pytest.mark.parametrize(
    ("categories", "aliases", "message"),
    [
        (("tier_three", "Tier Three"), None, "distinct after case/spacing"),
        (("absent", "present"), {"absent": []}, "non-empty string list"),
        (("absent", "present"), {"absent": ["Present"]}, "normalized collision"),
        (
            ("absent", "present"),
            {"absent": ["negative"], "present": [" NEGATIVE "]},
            "normalized collision",
        ),
        (("absent", "present"), {"unknown": ["other"]}, "subset of declared"),
    ],
)
def test_category_and_alias_normalization_collisions_are_rejected(
    categories, aliases, message
) -> None:
    spec = _spec(
        "alias_validation_measure",
        kind="categorical",
        categories=categories,
        aliases=aliases,
    )
    row = _candidate("candidate_0601", spec)
    with pytest.raises(ValueError, match=message):
        safe_staged_proposal_union((row,))


def test_duplicate_candidate_ids_are_rejected() -> None:
    row = _candidate("candidate_0601", _spec("duplicate_id_measure"))
    duplicate = _candidate("candidate_0601", _spec("another_measure"))
    with pytest.raises(ValueError, match="duplicates"):
        safe_staged_proposal_union((row, duplicate))
