from __future__ import annotations

import copy
import hashlib
import json

import pytest

from oci.inference.all_evidence_fusion import (
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    HTR_NEURAL,
    NEURAL_QUERY_MOMENTS,
    TFIDF_TOPICS,
)
from oci.inference.minimal_staged_selection_postprocessor import (
    postprocess_minimal_staged_selection,
)


def _spec(name: str, roles, *, kind: str = "continuous", categories=None):
    value = {
        "name": name,
        "description": f"Observable baseline {name}",
        "type": kind,
        "roles": list(roles),
    }
    if categories is not None:
        value["categories"] = list(categories)
    return value


def _candidate(
    candidate_id: str,
    spec,
    families,
    *,
    evidence=("evidence_0001",),
    occurrences=1,
):
    return {
        "candidate_id": candidate_id,
        "extraction_spec": copy.deepcopy(spec),
        "supporting_evidence_ids": list(evidence),
        "supporting_source_families": list(families),
        "validated_occurrence_count": occurrences,
    }


def _remote(candidate, *, cited_families=None):
    proposal = copy.deepcopy(candidate["extraction_spec"])
    proposal.update(
        {
            "supporting_evidence_ids": list(candidate["supporting_evidence_ids"][:1]),
            "supporting_source_families": list(
                cited_families or candidate["supporting_source_families"][:1]
            ),
            "rationale": "remote reasoning selection",
        }
    )
    return proposal


def _run(remote_candidates, pool, *, request_families=None, cap=20, **kwargs):
    if request_families is None:
        request_families = tuple(
            dict.fromkeys(
                family for candidate in pool for family in candidate["supporting_source_families"]
            )
        )
    return postprocess_minimal_staged_selection(
        remote_response={"proposals": [_remote(candidate) for candidate in remote_candidates]},
        remote_selected_candidate_ids=tuple(
            candidate["candidate_id"] for candidate in remote_candidates
        ),
        candidate_pool=tuple(pool),
        original_request_source_families=request_families,
        max_candidates=cap,
        **kwargs,
    )


def test_cap_is_not_a_fill_target_and_weak_redundant_candidate_is_omitted() -> None:
    selected = _candidate(
        "candidate_0001",
        _spec("sensor_temperature", ("confounder", "effect_modifier")),
        (BOW_NUISANCE, BOW_R_LOSS, TFIDF_TOPICS),
        evidence=("evidence_0001", "evidence_0002"),
        occurrences=3,
    )
    neural = _candidate(
        "candidate_0002",
        _spec(
            "material_phase",
            ("effect_modifier",),
            kind="categorical",
            categories=("crystalline", "amorphous"),
        ),
        (NEURAL_QUERY_MOMENTS,),
        evidence=("evidence_0003",),
    )
    weak = _candidate(
        "candidate_0003",
        _spec("weak_numeric", ("confounder",)),
        (TFIDF_TOPICS,),
        evidence=("evidence_0004",),
    )
    result = _run((selected,), (selected, neural, weak))

    assert [row["name"] for row in result.response["proposals"]] == [
        "sensor_temperature",
        "material_phase",
    ]
    assert result.mandatory_coverage_candidate_ids == ("candidate_0002",)
    assert result.high_confidence_reserve_candidate_ids == ()
    assert result.omitted_candidate_ids == ("candidate_0003",)
    assert result.candidate_pool_coverage_complete
    assert not result.cap_limited


def test_no_lexical_alias_merge_or_role_mutation_is_performed() -> None:
    sensor_temperature = _candidate(
        "candidate_0001",
        _spec("sensor_temperature", ("confounder",)),
        (BOW_NUISANCE,),
    )
    temperature = _candidate(
        "candidate_0002",
        _spec("temperature", ("effect_modifier",)),
        (BOW_R_LOSS,),
    )
    result = _run(
        (sensor_temperature, temperature),
        (sensor_temperature, temperature),
    )

    assert [row["name"] for row in result.response["proposals"]] == [
        "sensor_temperature",
        "temperature",
    ]
    assert result.response["proposals"][0]["roles"] == ["confounder"]
    assert result.response["proposals"][1]["roles"] == ["effect_modifier"]
    assert result.remote_selected_candidate_ids == (
        "candidate_0001",
        "candidate_0002",
    )


def test_high_confidence_reserve_protects_recall_without_adding_weak_pool_tail() -> None:
    selected = _candidate(
        "candidate_0001",
        _spec("selected_index", ("confounder", "effect_modifier")),
        (BOW_NUISANCE, NEURAL_QUERY_MOMENTS),
        evidence=("evidence_0001", "evidence_0002"),
    )
    recurrent = _candidate(
        "candidate_0002",
        _spec("recurrent_gauge", ("effect_modifier",)),
        (BOW_NUISANCE,),
        evidence=("evidence_0003",),
        occurrences=2,
    )
    independent = _candidate(
        "candidate_0003",
        _spec("independent_gauge", ("confounder",)),
        (BOW_R_LOSS, TFIDF_TOPICS),
        evidence=("evidence_0004", "evidence_0005"),
    )
    weak = _candidate(
        "candidate_0004",
        _spec("weak_gauge", ("confounder",)),
        (BOW_NUISANCE,),
        evidence=("evidence_0006",),
    )
    result = _run((selected,), (selected, recurrent, independent, weak))

    # Independent is needed for family coverage; recurrent is then retained as
    # the strong reserve. The weak tail is not used to fill the cap.
    assert result.mandatory_coverage_candidate_ids == ("candidate_0003",)
    assert result.high_confidence_reserve_candidate_ids == ("candidate_0002",)
    assert result.omitted_candidate_ids == ("candidate_0004",)


def test_minimum_coverage_prefers_one_broad_candidate_but_reserve_remains_separate() -> None:
    selected = _candidate(
        "candidate_0001",
        _spec("baseline", ("confounder",)),
        (BOW_NUISANCE,),
    )
    broad = _candidate(
        "candidate_0002",
        _spec("broad_modifier", ("effect_modifier",)),
        (EMBEDDING_CLUSTERED, NEURAL_QUERY_MOMENTS),
        evidence=("evidence_0002",),
    )
    cluster_only = _candidate(
        "candidate_0003",
        _spec("cluster_modifier", ("effect_modifier",)),
        (EMBEDDING_CLUSTERED,),
        evidence=("evidence_0003",),
    )
    neural_only = _candidate(
        "candidate_0004",
        _spec("neural_modifier", ("effect_modifier",)),
        (NEURAL_QUERY_MOMENTS,),
        evidence=("evidence_0004",),
    )
    result = _run((selected,), (selected, broad, cluster_only, neural_only))

    assert result.mandatory_coverage_candidate_ids == ("candidate_0002",)
    assert result.high_confidence_reserve_candidate_ids == ()
    assert set(result.omitted_candidate_ids) == {"candidate_0003", "candidate_0004"}


def test_original_request_family_without_candidate_is_never_hidden_by_pool_coverage() -> None:
    selected = _candidate(
        "candidate_0001",
        _spec("baseline", ("confounder", "effect_modifier")),
        (BOW_NUISANCE, NEURAL_QUERY_MOMENTS),
        evidence=("evidence_0001", "evidence_0002"),
    )
    result = _run(
        (selected,),
        (selected,),
        request_families=(BOW_NUISANCE, HTR_NEURAL, NEURAL_QUERY_MOMENTS),
    )

    assert result.candidate_pool_coverage_complete
    assert not result.original_request_candidate_coverage_complete
    assert result.original_request_families_without_candidate == (HTR_NEURAL,)
    assert not result.cap_limited


def test_cap_limited_partial_coverage_and_reserve_are_audited() -> None:
    selected = _candidate(
        "candidate_0001",
        _spec("baseline", ("confounder",)),
        (BOW_NUISANCE,),
    )
    neural = _candidate(
        "candidate_0002",
        _spec("neural", ("effect_modifier",)),
        (NEURAL_QUERY_MOMENTS,),
        occurrences=2,
    )
    cluster = _candidate(
        "candidate_0003",
        _spec("cluster", ("effect_modifier",)),
        (EMBEDDING_CLUSTERED,),
        occurrences=2,
    )
    result = _run((selected,), (selected, neural, cluster), cap=2)

    assert len(result.response["proposals"]) == 2
    assert not result.candidate_pool_coverage_complete
    assert not result.high_confidence_reserve_complete
    assert result.cap_limited
    assert len(result.omitted_candidate_ids) == 1


def test_full_validated_support_is_preserved_without_changing_contract() -> None:
    selected = _candidate(
        "candidate_0001",
        _spec("gauge", ("effect_modifier",)),
        (BOW_R_LOSS, TFIDF_TOPICS),
        evidence=("evidence_0001", "evidence_0002"),
    )
    result = _run((selected,), (selected,))
    proposal = result.response["proposals"][0]

    assert proposal["supporting_evidence_ids"] == ["evidence_0001", "evidence_0002"]
    assert proposal["supporting_source_families"] == [BOW_R_LOSS, TFIDF_TOPICS]
    assert proposal["name"] == "gauge"
    assert proposal["roles"] == ["effect_modifier"]


def test_hashes_closed_audit_and_returned_values_are_detached() -> None:
    selected = _candidate(
        "candidate_0001",
        _spec("gauge", ("confounder", "effect_modifier")),
        (BOW_NUISANCE,),
    )
    pool = (selected,)
    result = _run((selected,), pool)
    audit = result.audit()

    assert (
        result.output_sha256
        == hashlib.sha256(
            json.dumps(
                result.response,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert len(result.input_sha256) == len(result.output_sha256) == 64
    assert len(result.postprocessor_code_sha256) == 64
    assert audit["final_count"] == 1
    selected["extraction_spec"]["name"] = "mutated_after_call"
    assert result.response["proposals"][0]["name"] == "gauge"
    returned = result.response
    returned["proposals"][0]["name"] = "mutated_return_value"
    assert result.response["proposals"][0]["name"] == "gauge"
    assert audit["candidate_pool_source_family_counts"] == {BOW_NUISANCE: 1}


def test_minimum_cover_tie_uses_strength_then_opaque_id_without_type_error() -> None:
    selected = _candidate(
        "candidate_0001",
        _spec("baseline", ("confounder",)),
        (BOW_NUISANCE,),
    )
    weak = _candidate(
        "candidate_0002",
        _spec("weak_neural", ("effect_modifier",)),
        (NEURAL_QUERY_MOMENTS,),
    )
    strong = _candidate(
        "candidate_0003",
        _spec("strong_neural", ("effect_modifier",)),
        (NEURAL_QUERY_MOMENTS,),
        evidence=("evidence_0003", "evidence_0004"),
        occurrences=3,
    )
    result = _run((selected,), (selected, weak, strong))

    assert result.mandatory_coverage_candidate_ids == ("candidate_0003",)


def test_closed_schema_canonical_ids_alignment_and_cap_are_enforced() -> None:
    selected = _candidate(
        "candidate_0001",
        _spec("gauge", ("confounder",)),
        (BOW_NUISANCE,),
    )
    malformed = {**selected, "unexpected": True}
    with pytest.raises(ValueError, match="closed schema"):
        _run((selected,), (malformed,))

    bad_id = {**selected, "candidate_id": "not_canonical"}
    with pytest.raises(ValueError, match="not canonical"):
        _run((bad_id,), (bad_id,))

    changed = _remote(selected)
    changed["name"] = "different_gauge"
    with pytest.raises(ValueError, match="changed its selected extraction contract"):
        postprocess_minimal_staged_selection(
            remote_response={"proposals": [changed]},
            remote_selected_candidate_ids=("candidate_0001",),
            candidate_pool=(selected,),
            original_request_source_families=(BOW_NUISANCE,),
            max_candidates=2,
        )

    with pytest.raises(ValueError, match=r"\[1, 64\]"):
        _run((selected,), (selected,), cap=65)
