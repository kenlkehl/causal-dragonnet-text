from __future__ import annotations

import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
    canonical_json,
)
from oci.inference.htr_attention_evidence_schema import (
    ROLE_NEUTRAL_HTR_CHUNK_EVIDENCE_SCHEMA,
    ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA,
    ROLE_NEUTRAL_HTR_READABLE_SPAN_SCHEMA,
    ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_BATCH_SCHEMA,
    ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_PACKAGE_SCHEMA,
)
from oci.inference.htr_native_proof_capture import _array_sha256
from oci.inference.htr_stage2_complete_semantic_aggregation import (
    build_htr_semantic_aggregation_scope,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
    NATIVE_ROLE_NEUTRAL_UNIT_SCHEMA_VERSION,
    SEMANTIC_RETRIEVAL_DERIVATION,
    assemble_cumulative_spent_role_neutral_catalog,
    audit_complete_architecture_delivery,
    build_complete_architecture_chunks,
)


def _sha(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _payload(family: str, evidence: list[dict]) -> dict:
    return {
        "schema_version": NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
        "family": family,
        "architecture_evidence": evidence,
    }


def _embedding_item(
    *,
    family: str,
    name: str,
    contrast_family: str,
    axes: list[str],
    preflight: bool = False,
) -> dict:
    semantic = family == TFIDF_SEMANTIC_RETRIEVAL
    content = {
        "architecture_view": (
            SEMANTIC_RETRIEVAL_DERIVATION if semantic else "embedding_contrast"
        ),
        **({"source_passages_removed": True} if semantic else {}),
        "contrast": {
            "name": name,
            "contrast_family": contrast_family,
            "direction_source": "fit_target:configured_target:configured_split",
        },
        "concept_witnesses": [
            {"concept": "complete semantic witness", "score": 0.75},
            {"concept": "second complete witness", "score": -0.25},
        ],
        "member_batch_index": 1,
        "member_batch_count": 1,
        "full_member_count": 2,
        "source_chunk_count": 17,
        "all_source_chunks_accounted_once": True,
        "all_configured_semantic_terms_accounted_once": True,
    }
    row = {
        "atom_kind": (
            "tfidf_semantic_retrieval_contrast"
            if semantic
            else "embedding_contrast"
        ),
        "source_kind": "legacy_all_source",
        "observable_axes": axes,
        "content": content,
    }
    if preflight:
        row["canonical_preflight_scope_reused"] = True
        row["canonical_preflight_atom_index"] = 0
    return row


def _matched_proof(
    *,
    source_seal: str,
    subproducer: str,
    atoms: list[dict],
) -> dict:
    payload = {
        "subproducer": subproducer,
        "evidence_kind": (
            "complete_fold_vocabulary_coefficients_v1"
            if subproducer == "bow"
            else "complete_validation_pair_witnesses_v1"
        ),
        "top_k_applied": False,
        "text_truncation_applied": False,
        "atoms": atoms,
    }
    return {
        "source_family_seal_content_sha256": source_seal,
        "subproducer": subproducer,
        "evidence_payload_sha256": _sha(payload),
        "evidence_payload": payload,
    }


def _native_payloads() -> dict[str, dict]:
    long_chunk = " ".join(f"configured_chunk_token_{index:04d}" for index in range(700))
    htr_row = {
        "witness_kind": "complete_htr_chunk_attention",
        "schema_version": ROLE_NEUTRAL_HTR_CHUNK_EVIDENCE_SCHEMA,
        "stage": "nuisance",
        "objective": "joint_treatment_outcome_nuisance",
        "fold": 1,
        "fit_note_position": 0,
        "fit_row_id": 10,
        "chunk_index": 0,
        "chunk_text": long_chunk,
        "chunk_sha256": hashlib.sha256(long_chunk.encode("utf-8")).hexdigest(),
        "attention": 1.0,
        "readable_token_spans": [
            {
                "schema_version": ROLE_NEUTRAL_HTR_READABLE_SPAN_SCHEMA,
                "selection_rank": 1,
                "text": "configured_chunk_token_0000",
                "char_start": 0,
                "char_end": len("configured_chunk_token_0000"),
                "focus_token_position": 1,
                "focus_token_id": 101,
                "focus_decoded_token_text": "configured",
                "token_attention": 0.25,
                "chunk_attention": 1.0,
                "hierarchical_attention_score": 0.25,
                "special_tokens_excluded_from_readable_projection": True,
                "raw_special_token_mass_retained_in_sidecar": True,
                "overlap_handling": (
                    "retain_each_chunk_local_occurrence_no_note_level_"
                    "deduplication_v1"
                ),
            }
        ],
        "readable_span_policy": {
            "schema_version": (
                "deterministic_chunk_local_token_span_projection_v1"
            ),
            "maximum_spans_per_chunk": 4,
            "ranking": (
                "hierarchical_attention_desc_then_token_attention_desc_"
                "then_token_position_asc_v1"
            ),
            "special_tokens_excluded": True,
            "padding_excluded": True,
            "note_level_deduplication_applied": False,
            "overlapping_chunk_occurrences_retained": True,
            "complete_raw_inventory_retained": True,
        },
        "token_inventory_content_sha256": "e" * 64,
    }
    effect_row = deepcopy(htr_row)
    effect_row["stage"] = "effect_modifier"
    effect_row["objective"] = "pseudo_outcome_mse"
    token_fold_batches = [
        {
            "schema_version": ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_BATCH_SCHEMA,
            "stage": stage,
            "objective": objective,
            "fold": 1,
            "sentence_pooling": "token_attention",
            "effective_sentence_pooling": "token_attention",
            "fold_honesty": {
                "evidence_rows": "fold_validation_only",
                "fit_and_validation_rows_disjoint": True,
                "generated_after_fit": True,
                "validation_rows_used_for_model_fit": False,
            },
            "top_k_applied_to_raw_inventory": False,
            "all_overlapping_chunk_occurrences_retained": True,
            "token_occurrence_count": 3,
            "chunk_count": 1,
            "note_count": 1,
            "special_token_occurrence_count": 2,
            "padding_occurrence_count": 0,
            "special_token_attention_mass": 0.2,
        }
        for stage, objective in (
            ("nuisance", "joint_treatment_outcome_nuisance"),
            ("effect_modifier", "pseudo_outcome_mse"),
        )
    ]
    token_package_body = {
        "schema_version": ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_PACKAGE_SCHEMA,
        "sentence_pooling": "token_attention",
        "effective_sentence_pooling": "token_attention",
        "fold_batches": token_fold_batches,
        "fold_batch_count": len(token_fold_batches),
        "token_occurrence_count": 6,
        "chunk_interpretation_count": 2,
        "note_interpretation_count": 2,
        "special_token_occurrence_count": 4,
        "special_token_attention_mass": 0.4,
        "padding_occurrence_count": 0,
        "all_raw_token_occurrences_authenticated": True,
        "all_chunk_occurrences_authenticated": True,
        "top_k_applied_to_raw_inventory": False,
        "readable_spans_are_deterministic_projections_only": True,
        "hierarchical_attention_is_ranking_not_causal_attribution": True,
        "fold_honest_validation_only_evidence": True,
        "exact_oof_note_coverage": True,
    }
    htr_payload = {
        "schema_version": ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA,
        "family": HTR_NEURAL,
        "architecture_evidence": [htr_row, effect_row],
        "token_attention_evidence": {
            **token_package_body,
            "content_sha256": _sha(token_package_body),
        },
    }
    matched_seal = "f" * 64
    matched_bow = _matched_proof(
        source_seal=matched_seal,
        subproducer="bow",
        atoms=[
            {
                "fold": 1,
                "view_name": "word_1_2",
                "feature_index": index,
                "term": term,
                "control_delta_logit_coefficient": -0.2 + index,
                "treated_delta_logit_coefficient": 0.3 + index,
            }
            for index, term in enumerate(("histology", "smoking history"))
        ],
    )
    matched_htr = _matched_proof(
        source_seal=matched_seal,
        subproducer="htr",
        atoms=[
            {
                "fold": 1,
                "pair_index": index,
                "candidate_row_id": 10 + index,
                "control_row_id": 20 + index,
                "propensity_abs_diff": 0.01 + index / 100,
                "outcome_abs_diff": 0.02 + index / 100,
                "delta_logit": -0.4 + index,
            }
            for index in range(2)
        ],
    )
    return {
        BOW_NUISANCE: _payload(
            BOW_NUISANCE,
            [
                {
                    "objective": objective,
                    "view_name": "word_1_2",
                    "fold": 1,
                    "witness_kind": "fitted_tfidf_term",
                    "feature_index": index,
                    "term": term,
                    "idf": 1.25 + index,
                }
                for index, (objective, term) in enumerate(
                    (
                        ("treatment_nuisance", "baseline age"),
                        ("outcome_nuisance", "performance status"),
                    )
                )
            ],
        ),
        BOW_R_LOSS: _payload(
            BOW_R_LOSS,
            [
                {
                    "objective": "effect_weighted_r",
                    "view_name": "word_1_2",
                    "fold": 1,
                    "witness_kind": "fitted_tfidf_term",
                    "feature_index": 0,
                    "term": "pd l1 expression",
                    "idf": 2.5,
                }
            ],
        ),
        HTR_NEURAL: htr_payload,
        MATCHED_PAIR_UPLIFT: _payload(
            MATCHED_PAIR_UPLIFT,
            [matched_bow, matched_htr],
        ),
        EMBEDDING_WHOLE_COHORT: _payload(
            EMBEDDING_WHOLE_COHORT,
            [
                _embedding_item(
                    family=EMBEDDING_WHOLE_COHORT,
                    name="treatment",
                    contrast_family="marginal",
                    axes=["treatment"],
                )
            ],
        ),
        EMBEDDING_CLUSTERED: _payload(
            EMBEDDING_CLUSTERED,
            [
                _embedding_item(
                    family=EMBEDDING_CLUSTERED,
                    name="cluster_treatment_pc1",
                    contrast_family="cluster_local_treatment_contrast_basis",
                    axes=["treatment"],
                    preflight=True,
                )
            ],
        ),
        TFIDF_SEMANTIC_RETRIEVAL: _payload(
            TFIDF_SEMANTIC_RETRIEVAL,
            [
                _embedding_item(
                    family=TFIDF_SEMANTIC_RETRIEVAL,
                    name="outcome",
                    contrast_family="marginal",
                    axes=["outcome"],
                )
            ],
        ),
        TFIDF_TOPICS: _payload(
            TFIDF_TOPICS,
            [
                {
                    "bank": "effect",
                    "topic_id": "effect_topic_001",
                    "topic_position": 0,
                    "term_position": 0,
                    "witness_kind": "fitted_consensus_nmf_topic_term",
                    "term": "immune checkpoint expression",
                    "loading": 0.61,
                }
            ],
        ),
        TFIDF_ORPHAN_NGRAMS: _payload(
            TFIDF_ORPHAN_NGRAMS,
            [
                {
                    "witness_kind": "fit_side_residual_tfidf_ngram",
                    "fit_rank": 1,
                    "represented_in_effect_topic": False,
                    "feature": "tumor proportion score",
                    "signed_score": 2.2,
                    "combined_importance": 2.4,
                }
            ],
        ),
        NEURAL_QUERY_MOMENTS: _payload(
            NEURAL_QUERY_MOMENTS,
            [
                {
                    "query_id": "effect_query_001",
                    "bank": "effect",
                    "mechanical_role": "effect_modifier",
                    "statistical_gate_applied": False,
                    "member_count": 8,
                    "fit_standardized_score": 3.1,
                    "top_chunks": [],
                    "top_contrastive_ngrams": [
                        {"term": "pd l1 high", "tfidf_contrast": 0.8},
                        {"term": "tumor proportion", "tfidf_contrast": 0.4},
                    ],
                }
            ],
        ),
    }


def _assemble(payloads: dict[str, dict]):
    payloads = deepcopy(payloads)
    if (
        payloads[HTR_NEURAL].get("schema_version")
        == ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA
    ):
        source_payload = payloads[HTR_NEURAL]
        with tempfile.TemporaryDirectory(
            prefix="htr-stage2-catalog-adapter-"
        ) as temporary:
            array_root = Path(temporary).resolve() / "raw_arrays"
            array_root.mkdir()
            package = source_payload["token_attention_evidence"]
            for batch in package["fold_batches"]:
                stage = str(batch["stage"])
                prefix = f"{stage}_0001"
                decoded = ("[CLS]", "configured", "[SEP]")
                utf8 = "".join(decoded).encode("utf-8")
                offsets = np.asarray(
                    [
                        0,
                        len("[CLS]"),
                        len("[CLS]configured"),
                        len("[CLS]configured[SEP]"),
                    ],
                    dtype=np.int64,
                )
                values = {
                    "fit_note_position": np.asarray(
                        [0, 0, 0], dtype=np.int64
                    ),
                    "fit_row_id": np.asarray([10, 10, 10], dtype=np.int64),
                    "chunk_index": np.asarray([0, 0, 0], dtype=np.int32),
                    "token_position": np.asarray([0, 1, 2], dtype=np.int32),
                    "token_id": np.asarray([101, 2001, 102], dtype=np.int32),
                    "decoded_token_text_utf8": np.frombuffer(
                        utf8, dtype=np.uint8
                    ).copy(),
                    "decoded_token_text_byte_offsets": offsets,
                    "char_start": np.asarray([0, 0, 0], dtype=np.int32),
                    "char_end": np.asarray(
                        [0, len("configured"), 0], dtype=np.int32
                    ),
                    "is_special_token": np.asarray(
                        [1, 0, 1], dtype=np.uint8
                    ),
                    "is_padding": np.asarray([0, 0, 0], dtype=np.uint8),
                    "token_attention": np.asarray(
                        [0.1, 0.8, 0.1], dtype=np.float64
                    ),
                    "chunk_attention": np.asarray(
                        [1.0, 1.0, 1.0], dtype=np.float64
                    ),
                    "hierarchical_attention_score": np.asarray(
                        [0.1, 0.8, 0.1], dtype=np.float64
                    ),
                }
                columns: dict[str, dict] = {}
                for name, value in values.items():
                    array_name = f"{prefix}_{name}"
                    np.save(
                        array_root / f"{array_name}.npy",
                        value,
                        allow_pickle=False,
                    )
                    columns[name] = {
                        "array": array_name,
                        "content_sha256": _array_sha256(value),
                        "dtype": value.dtype.str,
                        "shape": list(value.shape),
                    }
                batch.update(
                    {
                        "raw_occurrence_order": (
                            "fit_note_position_then_chunk_index_then_"
                            "token_position_v1"
                        ),
                        "decoded_token_text_encoding": (
                            "concatenated_utf8_with_offsets_v1"
                        ),
                        "tokenizer_identity": {
                            "model_name": "prajjwal1/bert-tiny",
                            "vocabulary_sha256": "d" * 64,
                        },
                        "fit_note_positions": [0],
                        "fit_row_ids": [10],
                        "columns": columns,
                    }
                )
                body = {
                    key: value
                    for key, value in batch.items()
                    if key != "content_sha256"
                }
                batch["content_sha256"] = _sha(body)
            package_body = {
                key: value
                for key, value in package.items()
                if key != "content_sha256"
            }
            package["content_sha256"] = _sha(package_body)
            result = build_htr_semantic_aggregation_scope(
                root=(Path(temporary) / "aggregate").resolve(),
                source_payload=source_payload,
                source_array_store_root=array_root,
                source_fit_seal_content_sha256="c" * 64,
                source_payload_content_sha256=_sha(source_payload),
                source_fit_seal_locator=(
                    "components/outer_001_inner_001/htr/"
                    "fit_only_family_seal.json"
                ),
                logical_scope_id="outer_001_hierarchy_epoch_000",
                physical_owner_scope_id="outer_001_inner_001",
                outer_fold=1,
                context_epoch=0,
                scope_binding_sha256="a" * 64,
            )
            payloads[HTR_NEURAL] = dict(result.payload)
    hashes = {
        family: hashlib.sha256(f"artifact:{index}:{family}".encode()).hexdigest()
        for index, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    }
    return assemble_cumulative_spent_role_neutral_catalog(
        family_payload_by_family=payloads,
        family_artifact_sha256_by_family=hashes,
        scope_binding_sha256="a" * 64,
        scope_id="outer_001_hierarchy_epoch_000",
        outer_fold=1,
        provider_inner_fold=1,
        split_fingerprint="b" * 64,
        semantic_member_batch_size=2,
    )


def _native_units(value) -> list[dict]:
    output: list[dict] = []
    if isinstance(value, dict):
        if value.get("schema_version") == NATIVE_ROLE_NEUTRAL_UNIT_SCHEMA_VERSION:
            output.append(value)
        else:
            for child in value.values():
                output.extend(_native_units(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(_native_units(child))
    return output


def test_native_role_neutral_payloads_cover_all_ten_families_without_truncation():
    payloads = _native_payloads()
    catalog = _assemble(payloads)

    assert set(catalog.audit["atom_count_by_family"]) == set(
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    assert all(catalog.family_atoms(family) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES)
    assert (
        catalog.audit["native_payload_adapter_selection_or_truncation_applied"]
        is False
    )
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        adapter = catalog.audit["native_payload_adapter_by_family"][family]
        assert adapter["adapter_applied"] is (family != HTR_NEURAL)
        assert adapter["source_record_count"] == len(
            (
                catalog.family_atoms(family)
                if family == HTR_NEURAL
                else payloads[family]["architecture_evidence"]
            )
        )
        assert adapter["selection_or_truncation_applied"] is False
        if family == HTR_NEURAL:
            continue
        units = [
            unit
            for atom in catalog.family_atoms(family)
            for unit in _native_units(atom.content)
        ]
        assert units
        assert {unit["source_record_index"] for unit in units} == set(
            range(len(payloads[family]["architecture_evidence"]))
        )
        for unit in units:
            source = payloads[family]["architecture_evidence"][
                unit["source_record_index"]
            ]
            assert unit["source_record_sha256"] == _sha(source)
            assert canonical_json(json.loads(unit["native_record_json"])) == unit[
                "native_record_json"
            ]

    htr_content = [
        atom.content for atom in catalog.family_atoms(HTR_NEURAL)
    ]
    assert htr_content
    assert all(
        content["aggregate_batch"]["complete_semantic_aggregate_delivery"]
        is True
        for content in htr_content
    )
    assert "complete_htr_chunk_attention" not in canonical_json(htr_content)

    plan = build_complete_architecture_chunks(
        catalog,
        max_atoms_per_chunk=2,
        max_bytes_per_chunk=200_000,
        max_semantic_member_ids_per_chunk=2,
    )
    delivery = audit_complete_architecture_delivery(catalog, plan)
    assert delivery["all_catalog_atoms_delivered_exactly_once"] is True
    assert delivery["all_catalog_semantic_member_ids_delivered_exactly_once"] is True
    assert delivery["atoms_truncated"] is False


def test_native_matched_pair_nested_atoms_are_all_preserved_as_units():
    payloads = _native_payloads()
    catalog = _assemble(payloads)
    units = [
        unit
        for atom in catalog.family_atoms(MATCHED_PAIR_UPLIFT)
        for unit in _native_units(atom.content)
    ]
    expected_atoms = sum(
        len(record["evidence_payload"]["atoms"])
        for record in payloads[MATCHED_PAIR_UPLIFT]["architecture_evidence"]
    )
    assert len(units) == expected_atoms
    assert {
        json.loads(unit["native_proof_context_json"])["subproducer"]
        for unit in units
    } == {"bow", "htr"}


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("unknown_bow_witness", "BoW evidence schema changed"),
        ("mixed_schema", "mixes catalog and native schemas"),
        ("matched_hash", "does not authenticate"),
        ("embedding_coverage", "source-chunk coverage is incomplete"),
        ("neural_excerpt", "safe query semantics"),
    ),
)
def test_native_role_neutral_payload_adapter_fails_closed(tamper: str, message: str):
    payloads = _native_payloads()
    if tamper == "unknown_bow_witness":
        payloads[BOW_NUISANCE]["architecture_evidence"][0][
            "witness_kind"
        ] = "ranked_top_terms"
    elif tamper == "mixed_schema":
        payloads[BOW_NUISANCE]["architecture_evidence"].append(
            {
                "atom_kind": "bow_term_group",
                "source_kind": "legacy_all_source",
                "observable_axes": ["treatment"],
                "content": {},
            }
        )
    elif tamper == "matched_hash":
        payloads[MATCHED_PAIR_UPLIFT]["architecture_evidence"][0][
            "evidence_payload_sha256"
        ] = "0" * 64
    elif tamper == "embedding_coverage":
        payloads[EMBEDDING_WHOLE_COHORT]["architecture_evidence"][0]["content"][
            "all_source_chunks_accounted_once"
        ] = False
    else:
        payloads[NEURAL_QUERY_MOMENTS]["architecture_evidence"][0]["top_chunks"] = [
            {"text": "forbidden excerpt"}
        ]

    with pytest.raises(ValueError, match=message):
        _assemble(payloads)
