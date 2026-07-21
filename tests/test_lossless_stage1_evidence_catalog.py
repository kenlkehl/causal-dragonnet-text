from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    BOW_NUISANCE,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    TFIDF_TOPICS,
)
from oci.inference.authenticated_semantic_retrieval_compatibility import (
    restore_current_spent_projection_semantic_retrieval_view,
)
from oci.inference.all_evidence_fusion import (
    LEGACY_ALL_SOURCE,
    NEURAL_QUERY_SOURCE,
    SPARSE_QUERY_SOURCE,
    TFIDF_TOPIC_SOURCE,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK,
    SEMANTIC_RETRIEVAL_DERIVATION,
    audit_complete_architecture_delivery,
    assemble_cumulative_spent_role_neutral_catalog,
    build_complete_architecture_chunks,
    build_role_neutral_evidence_catalog,
)
from oci.inference.stage1_exact_inner_family_adapters import family_payload_from_catalog


def _provenance() -> FoldEvidenceProvenance:
    return FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=(0, 1, 2, 3),
        heldout_row_ids=(4, 5),
        artifact_id="closed-test-evidence",
    )


def _bow_group(source: str, view: str, evidence_type: str, meaning: str, term: str) -> dict:
    model = "linear" if view.startswith(("linear", "pair_uplift__linear")) else "extratrees"
    return {
        "source": source,
        "view_name": view,
        "bow_model": model,
        "evidence_type": evidence_type,
        "meaning": meaning,
        "rows": [
            {"feature": term, "score": 0.4},
            {"feature": f"{term} secondary", "score": -0.2},
        ],
    }


def _embedding(name: str, contrast_family: str, concept: str) -> dict:
    return {
        "name": name,
        "contrast_family": contrast_family,
        "direction_source": "mean_difference",
        "role_hint": "legacy label must not be rendered",
        "concept_derivation": SEMANTIC_RETRIEVAL_DERIVATION,
        "raw_retrieved_excerpts_retained": False,
        "concept_probe_scores": [
            {"concept": concept, "score": 0.7},
            {"concept": f"{concept} status", "score": -0.3},
        ],
    }


def _legacy_payload() -> dict:
    nuisance = _bow_group(
        "linear_1_2.treatment_positive",
        "linear_1_2",
        "treatment_positive",
        "Terms associated with treatment assignment.",
        "baseline age",
    )
    r_loss = _bow_group(
        "linear_1_2.pseudo_target_positive",
        "linear_1_2",
        "pseudo_target_positive",
        "Terms associated with an R-stage pseudo-target.",
        "performance status",
    )
    pair = _bow_group(
        "matched_pair_uplift.pair_uplift__linear_1_2.uplift_pair_features",
        "pair_uplift__linear_1_2",
        "uplift_pair_features",
        "Matched-pair treated versus control outcome terms.",
        "histology type",
    )
    return {
        "outer_fold": 1,
        "scope": "full_outer_train",
        "context": {
            "evidence_digest": {
                "confounders": {
                    "bow_blurbs": [nuisance],
                    "embedding_chunks": [_embedding("treatment", "marginal", "age")],
                    "htr_blurbs": [
                        {
                            "stage": "nuisance",
                            "meaning": "Attention for treatment and outcome nuisance models.",
                            "metrics": {},
                            "rows": [
                                {"token": "creatinine clearance", "attention_score": 0.8},
                                {"phrase": "prior platinum", "attention_score": 0.7},
                                {"concept": "patient sex", "attention_score": 0.6},
                                {"attended_token_summary": "baseline age", "attention_score": 0.5},
                            ],
                        }
                    ],
                },
                "effect_modifiers": {
                    "bow_blurbs": [r_loss, pair],
                    "embedding_chunks": [
                        _embedding(
                            "cluster_residualized_interaction_pc1",
                            "cluster_local_residualized_interaction_contrast_basis",
                            "brain metastases",
                        )
                    ],
                    "htr_blurbs": [
                        {
                            "stage": "effect",
                            "meaning": "Attention for residual-effect heterogeneity.",
                            "metrics": {},
                            "rows": [{"feature": "baseline nlr", "attention_score": 0.9}],
                        },
                        {
                            "stage": "pair_uplift",
                            "meaning": "Matched-pair uplift attention.",
                            "metrics": {},
                            "rows": [{"token": "egfr mutation", "attention_score": 0.95}],
                        },
                    ],
                },
            }
        },
    }


def _tfidf_payload() -> dict:
    banks = {}
    for bank, term in (
        ("treatment", "smoking history"),
        ("outcome", "performance status"),
        ("effect", "hemoglobin level"),
    ):
        banks[bank] = {
            "topics": [
                {
                    "topic_id": f"{bank}_topic_001",
                    "bank": bank,
                    "terms": [
                        {
                            "term": term,
                            "loading": 0.6,
                            "screen_rank": 1,
                            "signed_score": -0.4,
                        },
                        {
                            "term": f"{term} secondary",
                            "loading": 0.2,
                            "screen_rank": 2,
                            "signed_score": 0.1,
                        },
                    ],
                }
            ]
        }
    orphan_cluster = {
        "cluster_id": "effect_orphan_outer_001_001",
        "terms": [
            {
                "term": "baseline nlr",
                "combined_importance": 2.2,
                "fit_rank": 1,
                "fit_signed_score": 2.1,
                "lexical_similarity_to_seed": 1.0,
                "signed_score": 2.1,
                "support_control": 20,
                "support_treated": 21,
            }
        ],
    }
    return {
        "outer_fold": 1,
        "scope": "full_outer_train",
        "discovery": {
            "topic_banks": banks,
            "effect_orphan_ngram_branch": {
                "status": "completed",
                "selected_cluster_ids": ["effect_orphan_outer_001_001"],
                "selected_clusters": [orphan_cluster],
                "selection_count": 1,
            },
        },
    }


def _query_payload() -> dict:
    rows = []
    for bank, term in (
        ("treatment", "baseline age"),
        ("outcome", "performance status"),
        ("effect", "egfr mutation"),
    ):
        rows.append(
            {
                "query_id": f"{bank}_query_001",
                "bank": bank,
                "mechanical_role": "effect_modifier" if bank == "effect" else "confounder",
                "statistical_gate_applied": False,
                "member_count": 4,
                "member_subfolds": [1, 2],
                "fit_standardized_score": 3.2,
                "top_chunks": [],
                "top_contrastive_ngrams": [
                    {"term": term, "tfidf_contrast": 0.4},
                    {"term": f"{term} secondary", "tfidf_contrast": -0.2},
                ],
            }
        )
    return {
        "outer_fold": 1,
        "scope": "outer_train",
        "query_evidence": rows,
    }


def _inputs(*, reverse: bool = False) -> list[FoldEvidenceInput]:
    legacy = _legacy_payload()
    tfidf = _tfidf_payload()
    query = _query_payload()
    if reverse:
        for section in legacy["context"]["evidence_digest"].values():
            for key in ("bow_blurbs", "embedding_chunks", "htr_blurbs"):
                section[key].reverse()
        for bank in tfidf["discovery"]["topic_banks"].values():
            bank["topics"].reverse()
            for topic in bank["topics"]:
                topic["terms"].reverse()
        query["query_evidence"].reverse()
        for row in query["query_evidence"]:
            row["top_contrastive_ngrams"].reverse()
    rows = [
        FoldEvidenceInput(LEGACY_ALL_SOURCE, legacy, _provenance()),
        FoldEvidenceInput(TFIDF_TOPIC_SOURCE, tfidf, _provenance()),
        FoldEvidenceInput(NEURAL_QUERY_SOURCE, query, _provenance()),
    ]
    return list(reversed(rows)) if reverse else rows


def _current_spent_provider_identity() -> dict:
    provider_code_sha256 = "de11740a862c13d59d340e1dba26fb1202820dec4a0055c49819b7e01eccc1f1"
    return {
        "provider": "context_fit_review_spent_evidence_provider_v3",
        "provider_code_sha256": provider_code_sha256,
        "backends": [
            {
                "backend": "historical_stage1_spent_discovery_v5",
                "code_sha256": provider_code_sha256,
                "concept_projection": (
                    "short_bow_terms_htr_tokens_or_per_row_chunk_attention_contrast_"
                    "embedding_tail_ngrams_v2"
                ),
                "raw_attention_or_embedding_excerpts_retained": False,
            }
        ],
    }


def _inputs_missing_semantic_projection_attestations() -> list[FoldEvidenceInput]:
    inputs = _inputs()
    digest = inputs[0].payload["context"]["evidence_digest"]
    for section in ("confounders", "effect_modifiers"):
        for contrast in digest[section]["embedding_chunks"]:
            contrast.pop("concept_derivation")
            contrast.pop("raw_retrieved_excerpts_retained")
    return inputs


def _runtime_request_kwargs() -> dict:
    return {
        "outer_fold": 1,
        "review_round": 0,
        "exact_spent_row_ids": (0, 1, 2, 3),
        "exact_sealed_row_ids": (4, 5),
        "spent_texts": ("a", "b", "c", "d"),
        "spent_treatment": (0.0, 1.0, 0.0, 1.0),
        "spent_outcome": (1.0, 0.0, 1.0, 0.0),
    }


def test_current_spent_projection_compatibility_rejects_identity_mapping_relabeling():
    with pytest.raises(TypeError, match="exact production raw provider"):
        restore_current_spent_projection_semantic_retrieval_view(
            _inputs_missing_semantic_projection_attestations(),
            spent_evidence_provider=_current_spent_provider_identity(),
            **_runtime_request_kwargs(),
        )


def test_current_spent_projection_compatibility_rejects_overlay_identity_mapping():
    delegate = _current_spent_provider_identity()
    overlay_identity = {
        "provider": "authenticated_review_spent_cache_overlay_identity_v1",
        "delegate_provider_identity": delegate,
    }
    with pytest.raises(TypeError, match="exact production raw provider"):
        restore_current_spent_projection_semantic_retrieval_view(
            _inputs_missing_semantic_projection_attestations(),
            spent_evidence_provider=overlay_identity,
            **_runtime_request_kwargs(),
        )


def test_current_spent_projection_compatibility_rejects_protocol_spoof():
    class SpoofProvider:
        def identity(self):
            return _current_spent_provider_identity()

    with pytest.raises(TypeError, match="exact production raw provider"):
        restore_current_spent_projection_semantic_retrieval_view(
            _inputs_missing_semantic_projection_attestations(),
            spent_evidence_provider=SpoofProvider(),
            **_runtime_request_kwargs(),
        )


def test_catalog_covers_every_active_architecture_without_role_first_fields():
    catalog = build_role_neutral_evidence_catalog(_inputs())
    assert set(catalog.audit["atom_count_by_family"]) == set(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    assert all(
        catalog.audit["atom_count_by_family"][family] > 0
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    assert catalog.audit["inactive_sparse_query_present"] is False
    assert catalog.audit["role_fields_emitted"] is False
    assert catalog.audit["global_top_k_applied"] is False

    all_member_ids = [member_id for atom in catalog.atoms for member_id in atom.member_ids]
    assert len(all_member_ids) == len(set(all_member_ids))
    assert all(atom.member_ids for atom in catalog.atoms)
    assert all(
        set(atom.member_ids)
        == {value for key, value in _walk_key_values(atom.content) if key == "member_id"}
        for atom in catalog.atoms
    )


def _cumulative_family_payloads(catalog):
    return {
        family: family_payload_from_catalog(catalog, family=family)[0]
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }


def _cumulative_family_artifact_hashes():
    return {
        family: hashlib.sha256(f"{index}:{family}".encode()).hexdigest()
        for index, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    }


def _assemble_cumulative_catalog(payloads, *, artifact_hashes=None):
    return assemble_cumulative_spent_role_neutral_catalog(
        family_payload_by_family=payloads,
        family_artifact_sha256_by_family=(
            _cumulative_family_artifact_hashes() if artifact_hashes is None else artifact_hashes
        ),
        scope_binding_sha256="a" * 64,
        scope_id="outer_001_hierarchy_epoch_000",
        outer_fold=1,
        provider_inner_fold=1,
        split_fingerprint=_provenance().split_fingerprint,
    )


def test_cumulative_ten_payload_assembler_roundtrips_every_member_and_pages_losslessly():
    source = build_role_neutral_evidence_catalog(_inputs())
    payloads = _cumulative_family_payloads(source)
    assembled = _assemble_cumulative_catalog(payloads)

    assert assembled.audit["family_payload_roundtrip_verified"] is True
    assert assembled.audit["atom_count_by_family"] == source.audit["atom_count_by_family"]
    assert (
        assembled.audit["semantic_member_count_by_family"]
        == source.audit["semantic_member_count_by_family"]
    )
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        projected, projected_count = family_payload_from_catalog(assembled, family=family)
        assert projected == payloads[family]
        assert projected_count == len(payloads[family]["architecture_evidence"])

    plan = build_complete_architecture_chunks(assembled)
    assert plan.audit["all_catalog_atoms_delivered_exactly_once"] is True
    assert plan.audit["all_catalog_semantic_member_ids_delivered_exactly_once"] is True
    assert plan.audit["mixed_architecture_chunks_present"] is False
    assert plan.audit["atoms_truncated"] is False


def test_cumulative_ten_payload_assembler_is_mapping_order_independent():
    source = build_role_neutral_evidence_catalog(_inputs())
    payloads = _cumulative_family_payloads(source)
    hashes = _cumulative_family_artifact_hashes()
    first = _assemble_cumulative_catalog(payloads, artifact_hashes=hashes)
    second = _assemble_cumulative_catalog(
        dict(reversed(tuple(payloads.items()))),
        artifact_hashes=dict(reversed(tuple(hashes.items()))),
    )
    assert second.as_dict() == first.as_dict()


def test_cumulative_ten_payload_assembler_rejects_missing_family_alias_and_open_payload():
    source = build_role_neutral_evidence_catalog(_inputs())
    payloads = _cumulative_family_payloads(source)

    missing = deepcopy(payloads)
    missing.pop(ACTIVE_STAGE1_CONCEPT_FAMILIES[-1])
    with pytest.raises(ValueError, match="exactly ten family payloads"):
        _assemble_cumulative_catalog(missing)

    aliased_hashes = _cumulative_family_artifact_hashes()
    aliased_hashes[ACTIVE_STAGE1_CONCEPT_FAMILIES[-1]] = aliased_hashes[
        ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    ]
    with pytest.raises(ValueError, match="distinct identities"):
        _assemble_cumulative_catalog(payloads, artifact_hashes=aliased_hashes)

    opened = deepcopy(payloads)
    opened[BOW_NUISANCE]["unexpected"] = True
    with pytest.raises(ValueError, match="not a closed schema"):
        _assemble_cumulative_catalog(opened)


@pytest.mark.parametrize("tamper", ("schema", "source_kind", "member_id", "order"))
def test_cumulative_ten_payload_assembler_rejects_semantic_or_envelope_tamper(tamper):
    source = build_role_neutral_evidence_catalog(_inputs())
    payloads = _cumulative_family_payloads(source)
    family = BOW_NUISANCE
    changed = deepcopy(payloads)
    if tamper == "schema":
        changed[family]["schema_version"] = "wrong"
    elif tamper == "source_kind":
        changed[family]["architecture_evidence"][0]["source_kind"] = TFIDF_TOPIC_SOURCE
    elif tamper == "member_id":
        changed[family]["architecture_evidence"][0]["content"]["terms"][0][
            "member_id"
        ] = "member_injected"
    else:
        multi = HTR_NEURAL
        changed[multi]["architecture_evidence"].reverse()
    with pytest.raises(ValueError):
        _assemble_cumulative_catalog(changed)


def test_cumulative_ten_payload_assembler_rejects_a_missing_semantic_member_batch():
    inputs = _inputs()
    nuisance_rows = inputs[0].payload["context"]["evidence_digest"]["confounders"]["bow_blurbs"][0][
        "rows"
    ]
    nuisance_rows.extend(
        [
            {"feature": "baseline albumin", "score": 0.3},
            {"feature": "baseline bilirubin", "score": -0.1},
        ]
    )
    source = build_role_neutral_evidence_catalog(inputs)
    payloads = _cumulative_family_payloads(source)
    nuisance = payloads[BOW_NUISANCE]["architecture_evidence"]
    assert len(nuisance) == 2
    nuisance.pop()
    with pytest.raises(ValueError, match="omitted or duplicated a batch"):
        _assemble_cumulative_catalog(payloads)


def test_empty_adapter_records_cannot_satisfy_strict_architecture_completeness():
    inputs = _inputs()
    tfidf = next(row for row in inputs if row.source_kind == TFIDF_TOPIC_SOURCE)
    for bank in tfidf.payload["discovery"]["topic_banks"].values():
        for topic in bank["topics"]:
            topic["terms"] = []

    relaxed = build_role_neutral_evidence_catalog(
        inputs,
        require_all_architecture_families=False,
    )
    assert not relaxed.family_atoms(TFIDF_TOPICS)
    assert all(atom.member_ids for atom in relaxed.atoms)
    assert relaxed.audit["semantic_member_count_by_family"][TFIDF_TOPICS] == 0

    with pytest.raises(ValueError, match="tfidf_topics"):
        build_role_neutral_evidence_catalog(inputs)


def _walk_key_values(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk_key_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_key_values(child)


def test_htr_token_phrase_concept_and_summary_survive_but_pair_is_delivered_once():
    catalog = build_role_neutral_evidence_catalog(_inputs())
    htr_surfaces = {
        atom.content["phrase_evidence"]["phrase"] for atom in catalog.family_atoms(HTR_NEURAL)
    }
    assert {
        "creatinine clearance",
        "prior platinum",
        "patient sex",
        "baseline age",
        "baseline nlr",
    } <= htr_surfaces
    assert "egfr mutation" not in htr_surfaces
    pair_htr = [
        atom
        for atom in catalog.family_atoms(MATCHED_PAIR_UPLIFT)
        if atom.atom_kind == "matched_pair_htr_phrase"
    ]
    assert [atom.content["phrase_evidence"]["phrase"] for atom in pair_htr] == ["egfr mutation"]


def test_neural_query_aggregate_moments_are_separate_and_never_enter_discovery_chunks():
    catalog = build_role_neutral_evidence_catalog(_inputs())
    assert len(catalog.non_grounding_numerical_summaries) == 3
    assert all(
        summary.source_family == NEURAL_QUERY_MOMENTS
        for summary in catalog.non_grounding_numerical_summaries
    )
    plan = build_complete_architecture_chunks(catalog, max_atoms_per_chunk=2)
    serialized = str([chunk.as_dict() for chunk in plan.chunks])
    assert "fit_standardized_score" not in serialized
    assert "mechanical_role" not in serialized
    assert "member_subfolds" not in serialized
    assert "3.2" not in serialized
    assert plan.audit["non_grounding_numerical_summaries_delivered"] is False


def test_architecture_chunks_are_single_family_complete_and_exactly_audited():
    catalog = build_role_neutral_evidence_catalog(_inputs())
    plan = build_complete_architecture_chunks(
        catalog,
        max_atoms_per_chunk=1,
        max_bytes_per_chunk=20_000,
        max_semantic_member_ids_per_chunk=2,
    )
    assert all(
        {row["source_family"] for row in chunk.evidence} == {chunk.source_family}
        for chunk in plan.chunks
    )
    audit = audit_complete_architecture_delivery(catalog, plan)
    assert audit["all_catalog_atoms_delivered_exactly_once"] is True
    assert audit["all_catalog_semantic_member_ids_delivered_exactly_once"] is True
    assert audit["catalog_semantic_member_id_count"] == sum(
        len(atom.member_ids) for atom in catalog.atoms
    )
    assert (
        audit["observed_semantic_member_id_delivery_count"]
        == audit["catalog_semantic_member_id_count"]
    )
    assert audit["observed_max_semantic_member_ids_per_chunk"] <= 2
    assert audit["mixed_architecture_chunks_present"] is False
    assert audit["atoms_truncated"] is False
    assert audit["arbitrary_structural_fragments_emitted"] is False


def test_catalog_and_chunk_hashes_are_order_independent():
    first = build_role_neutral_evidence_catalog(_inputs())
    second = build_role_neutral_evidence_catalog(_inputs(reverse=True))
    assert first.catalog_sha256 == second.catalog_sha256
    first_plan = build_complete_architecture_chunks(first, max_atoms_per_chunk=2)
    second_plan = build_complete_architecture_chunks(second, max_atoms_per_chunk=2)
    assert first_plan.plan_sha256 == second_plan.plan_sha256


def test_semantic_member_bound_repacks_deterministically_without_member_loss():
    catalog = build_role_neutral_evidence_catalog(_inputs())
    first = build_complete_architecture_chunks(
        catalog,
        max_atoms_per_chunk=24,
        max_bytes_per_chunk=48_000,
        max_semantic_member_ids_per_chunk=3,
    )
    second = build_complete_architecture_chunks(
        catalog,
        max_atoms_per_chunk=24,
        max_bytes_per_chunk=48_000,
        max_semantic_member_ids_per_chunk=3,
    )

    assert first.as_dict() == second.as_dict()
    delivered_member_ids = [
        member_id
        for chunk in first.chunks
        for item in chunk.evidence
        for member_id in item["member_ids"]
    ]
    catalog_member_ids = [member_id for atom in catalog.atoms for member_id in atom.member_ids]
    assert sorted(delivered_member_ids) == sorted(catalog_member_ids)
    assert len(delivered_member_ids) == len(set(delivered_member_ids))
    assert all(
        sum(len(item["member_ids"]) for item in chunk.evidence) <= 3 for chunk in first.chunks
    )
    assert first.audit["all_catalog_atoms_delivered_exactly_once"] is True
    assert first.audit["all_catalog_semantic_member_ids_delivered_exactly_once"] is True


def test_semantic_member_bound_is_part_of_plan_identity_even_when_layout_is_unchanged():
    catalog = build_role_neutral_evidence_catalog(_inputs())
    first = build_complete_architecture_chunks(
        catalog,
        max_atoms_per_chunk=1,
        max_semantic_member_ids_per_chunk=63,
    )
    second = build_complete_architecture_chunks(
        catalog,
        max_atoms_per_chunk=1,
        max_semantic_member_ids_per_chunk=64,
    )

    assert [chunk.chunk_id for chunk in first.chunks] == [chunk.chunk_id for chunk in second.chunks]
    assert first.plan_sha256 != second.plan_sha256
    assert first.as_dict()["max_semantic_member_ids_per_chunk"] == 63
    assert second.as_dict()["max_semantic_member_ids_per_chunk"] == 64


def test_default_semantic_member_bound_is_conservative_and_oversized_atom_fails_closed():
    catalog = build_role_neutral_evidence_catalog(_inputs())
    plan = build_complete_architecture_chunks(catalog)
    assert DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK == 3
    assert plan.max_semantic_member_ids_per_chunk == 3

    with pytest.raises(ValueError, match="semantic atom exceeds max_semantic_member_ids"):
        build_complete_architecture_chunks(
            catalog,
            max_semantic_member_ids_per_chunk=1,
        )


def test_sparse_query_unknown_fields_and_oversize_atoms_fail_closed():
    sparse = FoldEvidenceInput(
        SPARSE_QUERY_SOURCE,
        {"query_evidence": []},
        _provenance(),
    )
    with pytest.raises(ValueError, match="inactive sparse-query fallback"):
        build_role_neutral_evidence_catalog(
            [sparse],
            require_all_source_kinds=False,
        )

    bad = _inputs()
    bad[0].payload["context"]["evidence_digest"]["confounders"]["bow_blurbs"][0][
        "unhandled_branch"
    ] = []
    with pytest.raises(ValueError, match="unhandled fields"):
        build_role_neutral_evidence_catalog(bad)

    catalog = build_role_neutral_evidence_catalog(_inputs())
    with pytest.raises(ValueError, match="semantic atom exceeds"):
        build_complete_architecture_chunks(catalog, max_bytes_per_chunk=100)


def test_duplicate_semantic_records_retain_distinct_evidence_and_member_instances():
    inputs = _inputs()
    section = inputs[0].payload["context"]["evidence_digest"]["confounders"]
    section["bow_blurbs"].append(deepcopy(section["bow_blurbs"][0]))
    catalog = build_role_neutral_evidence_catalog(inputs)
    matching = [
        atom
        for atom in catalog.family_atoms(BOW_NUISANCE)
        if atom.content["group"]["source"] == "linear_1_2.treatment_positive"
    ]
    assert len(matching) == 2
    assert len({atom.evidence_id for atom in matching}) == 2
    assert set(matching[0].member_ids).isdisjoint(matching[1].member_ids)
