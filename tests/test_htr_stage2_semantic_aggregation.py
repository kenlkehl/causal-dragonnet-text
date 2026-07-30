from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    HTR_NEURAL,
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
    HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA,
    build_htr_semantic_aggregation_scope,
    normalize_htr_complete_readable_token,
    validate_htr_semantic_aggregation_scope,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    _normalize_cumulative_family_payload,
    build_complete_architecture_chunks,
)
from tests.test_native_role_neutral_payload_catalog_adapter import (
    _assemble,
    _native_payloads,
)


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _source_payload(array_root: Path) -> dict:
    array_root.mkdir()
    rows: list[dict] = []
    batches: list[dict] = []
    note_rows = (
        (1, 10, "PD-L1 high response marker"),
        (2, 20, "pd-l1 HIGH response marker"),
    )
    for stage, objective in (
        ("nuisance", "joint_treatment_outcome_nuisance"),
        ("effect_modifier", "pseudo_outcome_mse"),
    ):
        for fold, row_id, text in note_rows:
            columns: dict[str, list] = {
                "fit_note_position": [],
                "fit_row_id": [],
                "chunk_index": [],
                "token_position": [],
                "token_id": [],
                "char_start": [],
                "char_end": [],
                "is_special_token": [],
                "is_padding": [],
                "token_attention": [],
                "chunk_attention": [],
                "hierarchical_attention_score": [],
            }
            decoded: list[str] = []
            for chunk_index in (0, 1):
                chunk_text = f"{text} overlap chunk {chunk_index}"
                high_start = chunk_text.casefold().index("high")
                response_start = chunk_text.casefold().index("response")
                marker_start = chunk_text.casefold().index("marker")
                hyphen_start = chunk_text.index("-")
                token_rows = (
                    (101, "[CLS]", 0, 0, True, 0.05),
                    (
                        2031,
                        "high",
                        high_start,
                        high_start + 4,
                        False,
                        0.25,
                    ),
                    (
                        118,
                        "-",
                        hyphen_start,
                        hyphen_start + 1,
                        False,
                        0.05,
                    ),
                    (
                        3433,
                        "response",
                        response_start,
                        response_start + 8,
                        False,
                        0.25,
                    ),
                    (
                        3341,
                        "marker",
                        marker_start,
                        marker_start + 6,
                        False,
                        0.35,
                    ),
                    (102, "[SEP]", 0, 0, True, 0.05),
                )
                for (
                    token_position,
                    (
                        token_id,
                        decoded_text,
                        char_start,
                        char_end,
                        special,
                        token_weight,
                    ),
                ) in enumerate(token_rows):
                    columns["fit_note_position"].append(row_id)
                    columns["fit_row_id"].append(row_id)
                    columns["chunk_index"].append(chunk_index)
                    columns["token_position"].append(token_position)
                    columns["token_id"].append(token_id)
                    columns["char_start"].append(char_start)
                    columns["char_end"].append(char_end)
                    columns["is_special_token"].append(int(special))
                    columns["is_padding"].append(0)
                    columns["token_attention"].append(token_weight)
                    columns["chunk_attention"].append(0.5)
                    columns["hierarchical_attention_score"].append(
                        token_weight * 0.5
                    )
                    decoded.append(decoded_text)
                spans = [
                    {
                        "schema_version": (
                            ROLE_NEUTRAL_HTR_READABLE_SPAN_SCHEMA
                        ),
                        "selection_rank": 1,
                        "text": text,
                        "char_start": 0,
                        "char_end": len(text),
                        "focus_token_position": 1,
                        "focus_token_id": 2031,
                        "focus_decoded_token_text": (
                            "HIGH" if fold == 2 else "high"
                        ),
                        "token_attention": 0.25,
                        "chunk_attention": 0.5,
                        "hierarchical_attention_score": 0.125,
                        "special_tokens_excluded_from_readable_projection": (
                            True
                        ),
                        "raw_special_token_mass_retained_in_sidecar": True,
                        "overlap_handling": (
                            "retain_each_chunk_local_occurrence_no_note_level_"
                            "deduplication_v1"
                        ),
                    },
                    {
                        "schema_version": (
                            ROLE_NEUTRAL_HTR_READABLE_SPAN_SCHEMA
                        ),
                        "selection_rank": 2,
                        "text": f"response marker overlap {chunk_index}",
                        "char_start": 6,
                        "char_end": len(chunk_text),
                        "focus_token_position": 4,
                        "focus_token_id": 3341,
                        "focus_decoded_token_text": "marker",
                        "token_attention": 0.35,
                        "chunk_attention": 0.5,
                        "hierarchical_attention_score": 0.175,
                        "special_tokens_excluded_from_readable_projection": (
                            True
                        ),
                        "raw_special_token_mass_retained_in_sidecar": True,
                        "overlap_handling": (
                            "retain_each_chunk_local_occurrence_no_note_level_"
                            "deduplication_v1"
                        ),
                    },
                ]
                rows.append(
                    {
                        "witness_kind": "complete_htr_chunk_attention",
                        "schema_version": (
                            ROLE_NEUTRAL_HTR_CHUNK_EVIDENCE_SCHEMA
                        ),
                        "stage": stage,
                        "objective": objective,
                        "fold": fold,
                        "fit_note_position": row_id,
                        "fit_row_id": row_id,
                        "chunk_index": chunk_index,
                        "chunk_text": chunk_text,
                        "chunk_sha256": hashlib.sha256(
                            chunk_text.encode("utf-8")
                        ).hexdigest(),
                        "attention": 0.5,
                        "readable_token_spans": spans,
                        "readable_span_policy": {
                            "schema_version": (
                                "deterministic_chunk_local_token_span_"
                                "projection_v1"
                            ),
                            "maximum_spans_per_chunk": 4,
                            "ranking": (
                                "hierarchical_attention_desc_then_token_"
                                "attention_desc_then_token_position_asc_v1"
                            ),
                            "special_tokens_excluded": True,
                            "padding_excluded": True,
                            "note_level_deduplication_applied": False,
                            "overlapping_chunk_occurrences_retained": True,
                            "complete_raw_inventory_retained": True,
                        },
                        "token_inventory_content_sha256": "a" * 64,
                    }
                )
            batches.append(
                {}
            )
            arrays = {
                "fit_note_position": np.asarray(
                    columns["fit_note_position"], dtype=np.int64
                ),
                "fit_row_id": np.asarray(
                    columns["fit_row_id"], dtype=np.int64
                ),
                "chunk_index": np.asarray(
                    columns["chunk_index"], dtype=np.int32
                ),
                "token_position": np.asarray(
                    columns["token_position"], dtype=np.int32
                ),
                "token_id": np.asarray(columns["token_id"], dtype=np.int32),
                "char_start": np.asarray(
                    columns["char_start"], dtype=np.int32
                ),
                "char_end": np.asarray(columns["char_end"], dtype=np.int32),
                "is_special_token": np.asarray(
                    columns["is_special_token"], dtype=np.uint8
                ),
                "is_padding": np.asarray(
                    columns["is_padding"], dtype=np.uint8
                ),
                "token_attention": np.asarray(
                    columns["token_attention"], dtype=np.float64
                ),
                "chunk_attention": np.asarray(
                    columns["chunk_attention"], dtype=np.float64
                ),
                "hierarchical_attention_score": np.asarray(
                    columns["hierarchical_attention_score"],
                    dtype=np.float64,
                ),
            }
            utf8 = bytearray()
            offsets = [0]
            for value in decoded:
                utf8.extend(value.encode("utf-8"))
                offsets.append(len(utf8))
            arrays["decoded_token_text_utf8"] = np.frombuffer(
                bytes(utf8), dtype=np.uint8
            ).copy()
            arrays["decoded_token_text_byte_offsets"] = np.asarray(
                offsets, dtype=np.int64
            )
            registration: dict[str, dict] = {}
            prefix = f"{stage}_{fold}"
            for name, value in arrays.items():
                array_name = f"{prefix}_{name}"
                np.save(
                    array_root / f"{array_name}.npy",
                    value,
                    allow_pickle=False,
                )
                registration[name] = {
                    "array": array_name,
                    "content_sha256": _array_sha256(value),
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                }
            batch_body = {
                    "schema_version": (
                        ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_BATCH_SCHEMA
                    ),
                    "stage": stage,
                    "objective": objective,
                    "fold": fold,
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
                    "raw_occurrence_order": (
                        "fit_note_position_then_chunk_index_then_"
                        "token_position_v1"
                    ),
                    "decoded_token_text_encoding": (
                        "concatenated_utf8_with_offsets_v1"
                    ),
                    "tokenizer_identity": {
                        "model_name": "prajjwal1/bert-tiny",
                        "tokenizer_class": "BertTokenizerFast",
                        "vocabulary_sha256": "d" * 64,
                    },
                    "fit_note_positions": [row_id],
                    "fit_row_ids": [row_id],
                    "columns": registration,
                    "token_occurrence_count": 12,
                    "chunk_count": 2,
                    "note_count": 1,
                    "special_token_occurrence_count": 4,
                    "padding_occurrence_count": 0,
                    "special_token_attention_mass": 0.2,
                }
            batches[-1] = {
                **batch_body,
                "content_sha256": _sha(batch_body),
            }
    package = {
        "schema_version": ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_PACKAGE_SCHEMA,
        "sentence_pooling": "token_attention",
        "effective_sentence_pooling": "token_attention",
        "fold_batches": batches,
        "fold_batch_count": len(batches),
        "token_occurrence_count": 48,
        "chunk_interpretation_count": 8,
        "note_interpretation_count": 4,
        "special_token_occurrence_count": 16,
        "special_token_attention_mass": 0.8,
        "padding_occurrence_count": 0,
        "all_raw_token_occurrences_authenticated": True,
        "all_chunk_occurrences_authenticated": True,
        "top_k_applied_to_raw_inventory": False,
        "readable_spans_are_deterministic_projections_only": True,
        "hierarchical_attention_is_ranking_not_causal_attribution": True,
        "fold_honest_validation_only_evidence": True,
        "exact_oof_note_coverage": True,
    }
    package["content_sha256"] = _sha(package)
    return {
        "schema_version": ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA,
        "family": HTR_NEURAL,
        "architecture_evidence": rows,
        "token_attention_evidence": package,
    }


def _build(root: Path):
    array_root = (root.parent / f"{root.name}_raw_arrays").resolve()
    payload = _source_payload(array_root)
    return (
        payload,
        array_root,
        build_htr_semantic_aggregation_scope(
            root=root,
            source_payload=payload,
            source_array_store_root=array_root,
            source_fit_seal_content_sha256="b" * 64,
            source_payload_content_sha256=_sha(payload),
            source_fit_seal_locator=(
                "components/outer_001_inner_001/htr/"
                "fit_only_family_seal.json"
            ),
            logical_scope_id="outer_001_hierarchy_epoch_000",
            physical_owner_scope_id="outer_001_inner_001",
            outer_fold=1,
            context_epoch=0,
            scope_binding_sha256="c" * 64,
            max_model_facing_batch_bytes=20_000,
            max_model_facing_token_upper_bound=20_000,
        ),
    )


def test_complete_reverse_index_overlap_and_special_accounting(
    tmp_path: Path,
) -> None:
    payload, array_root, result = _build((tmp_path / "aggregate").resolve())
    reopened = validate_htr_semantic_aggregation_scope(
        root=result.scope_manifest_path.parent,
        source_payload=payload,
        source_array_store_root=array_root,
        expected_source_fit_seal_content_sha256="b" * 64,
        expected_source_payload_content_sha256=_sha(payload),
        expected_scope_binding_sha256="c" * 64,
    )
    summary = reopened.scope_manifest["summary"]
    assert summary["eligible_readable_token_occurrence_count"] == 24
    assert summary["aggregated_readable_token_occurrence_count"] == 24
    assert summary["special_token_accounting_bucket"] == {
        "occurrence_count": 16,
        "attention_mass": 0.8,
        "excluded_from_readable_phrases": True,
        "retained_in_raw_authenticated_package": True,
    }
    assert summary["non_readable_accounting_bucket"][
        "occurrence_count"
    ] == 8
    assert summary["eligible_readable_token_occurrence_count"] > sum(
        len(row["readable_token_spans"])
        for row in payload["architecture_evidence"]
    )
    cross_registration = reopened.scope_manifest[
        "cross_fold_aggregates"
    ]
    cross_payload = json.loads(
        (
            reopened.scope_manifest_path.parent
            / cross_registration["relative_path"]
        ).read_text(encoding="utf-8")
    )
    aggregates = cross_payload["cross_fold_aggregates"]
    nuisance_high = next(
        aggregate
        for aggregate in aggregates
        if aggregate["stage"] == "nuisance"
        and aggregate["normalized_focus_text"] == "high"
    )
    assert nuisance_high["occurrence_count"] == 4
    assert nuisance_high["unique_note_count"] == 2
    assert nuisance_high["unique_chunk_count"] == 4
    assert nuisance_high["overlap_accounting"][
        "unique_supporting_note_count"
    ] == 2
    assert nuisance_high["attention_summaries"][
        "hierarchical_attention_score"
    ]["note_level_max"]["note_count"] == 2


def test_aggregation_and_batching_are_deterministic_and_catalog_consumable(
    tmp_path: Path,
) -> None:
    payload, _first_arrays, first = _build((tmp_path / "first").resolve())
    _payload, _second_arrays, second = _build((tmp_path / "second").resolve())
    assert first.payload == second.payload
    assert first.scope_manifest == second.scope_manifest
    assert normalize_htr_complete_readable_token("  HIGH  ") == (
        "high",
        "whole_or_initial_token",
    )

    payloads = _native_payloads()
    payloads[HTR_NEURAL] = dict(first.payload)
    catalog = _assemble(payloads)
    plan = build_complete_architecture_chunks(
        catalog,
        max_atoms_per_chunk=2,
        max_bytes_per_chunk=100_000,
        max_semantic_member_ids_per_chunk=3,
    )
    htr_chunks = [
        chunk for chunk in plan.chunks if chunk.source_family == HTR_NEURAL
    ]
    assert len(htr_chunks) == first.scope_manifest["summary"][
        "model_facing_batch_count"
    ]
    assert all(len(chunk.evidence) == 1 for chunk in htr_chunks)
    rendered = canonical_json([chunk.as_dict() for chunk in htr_chunks])
    assert "complete_htr_chunk_attention" not in rendered
    assert "raw_token_arrays" not in rendered
    assert plan.audit[
        "htr_semantic_batches_are_one_per_interpretation_request"
    ] is True

    normalized, audit = _normalize_cumulative_family_payload(
        first.payload,
        family=HTR_NEURAL,
        semantic_member_batch_size=2,
    )
    assert normalized["architecture_evidence"]
    assert audit["complete_token_attention_evidence"][
        "selection_or_truncation_applied_to_semantic_aggregates"
    ] is False


@pytest.mark.parametrize(
    "mutation",
    ("altered", "missing", "duplicated"),
)
def test_altered_missing_or_duplicated_reverse_index_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload, array_root, result = _build((tmp_path / mutation).resolve())
    manifest = result.scope_manifest
    reverse_registration = manifest["reverse_index_manifest"]
    reverse = json.loads(
        (
            result.scope_manifest_path.parent
            / reverse_registration["relative_path"]
        ).read_text(encoding="utf-8")
    )
    array_registration = reverse["arrays"][
        "raw_occurrence_index_by_cross_aggregate"
    ]
    path = (
        result.scope_manifest_path.parent
        / array_registration["relative_path"]
    )
    if mutation == "missing":
        path.rename(path.with_suffix(".missing"))
    else:
        values = np.load(path, allow_pickle=False)
        changed = np.array(values, copy=True)
        changed[0] = (
            changed[1] if mutation == "duplicated" else changed[0] + 1
        )
        np.save(path, changed, allow_pickle=False)
    with pytest.raises(ValueError):
        validate_htr_semantic_aggregation_scope(
            root=result.scope_manifest_path.parent,
            source_payload=payload,
            source_array_store_root=array_root,
            expected_source_fit_seal_content_sha256="b" * 64,
            expected_source_payload_content_sha256=_sha(payload),
            expected_scope_binding_sha256="c" * 64,
        )


def test_raw_chunk_payload_cannot_bypass_semantic_aggregation(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="semantic aggregate payload"):
        _normalize_cumulative_family_payload(
            _source_payload((tmp_path / "raw_bypass_arrays").resolve()),
            family=HTR_NEURAL,
            semantic_member_batch_size=2,
        )


def test_altered_authenticated_raw_token_sidecar_is_rejected(
    tmp_path: Path,
) -> None:
    payload, array_root, result = _build(
        (tmp_path / "altered-source").resolve()
    )
    first_batch = payload["token_attention_evidence"]["fold_batches"][0]
    registration = first_batch["columns"]["token_attention"]
    path = array_root / f"{registration['array']}.npy"
    values = np.load(path, allow_pickle=False)
    changed = np.array(values, copy=True)
    changed[1] += 0.01
    np.save(path, changed, allow_pickle=False)
    with pytest.raises(ValueError, match="does not authenticate"):
        validate_htr_semantic_aggregation_scope(
            root=result.scope_manifest_path.parent,
            source_payload=payload,
            source_array_store_root=array_root,
            expected_source_fit_seal_content_sha256="b" * 64,
            expected_source_payload_content_sha256=_sha(payload),
            expected_scope_binding_sha256="c" * 64,
        )


def _reseal_registered_json(
    *,
    scope_manifest_path: Path,
    registration_key: str,
    value: dict,
) -> None:
    scope = json.loads(scope_manifest_path.read_text(encoding="utf-8"))
    registration = scope[registration_key]
    payload = canonical_json(value).encode("utf-8")
    (
        scope_manifest_path.parent / registration["relative_path"]
    ).write_bytes(payload)
    scope[registration_key] = {
        **registration,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "content_sha256": _sha(value),
    }
    scope_body = {
        key: child
        for key, child in scope.items()
        if key != "content_sha256"
    }
    scope["content_sha256"] = _sha(scope_body)
    scope_manifest_path.write_bytes(
        canonical_json(scope).encode("utf-8")
    )


@pytest.mark.parametrize(
    "mutation",
    ("altered_full_aggregate", "missing_model_aggregate", "duplicated_model_aggregate"),
)
def test_resealed_aggregate_tampering_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload, array_root, result = _build((tmp_path / mutation).resolve())
    scope_path = result.scope_manifest_path
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    registration_key = (
        "cross_fold_aggregates"
        if mutation == "altered_full_aggregate"
        else "model_facing_payload"
    )
    registration = scope[registration_key]
    registered_path = (
        scope_path.parent / registration["relative_path"]
    )
    value = json.loads(registered_path.read_text(encoding="utf-8"))
    if mutation == "altered_full_aggregate":
        aggregate = value["cross_fold_aggregates"][0]
        aggregate["occurrence_count"] += 1
        aggregate_body = {
            key: child
            for key, child in aggregate.items()
            if key != "content_sha256"
        }
        aggregate["content_sha256"] = _sha(aggregate_body)
    else:
        batch = value["architecture_evidence"][0]["content"][
            "aggregate_batch"
        ]
        if mutation == "missing_model_aggregate":
            batch["aggregates"].pop()
        else:
            batch["aggregates"].append(
                deepcopy(batch["aggregates"][0])
            )
        batch["aggregate_count"] = len(batch["aggregates"])
        batch_body = {
            key: child
            for key, child in batch.items()
            if key != "content_sha256"
        }
        batch["content_sha256"] = _sha(batch_body)
    value_body = {
        key: child
        for key, child in value.items()
        if key != "content_sha256"
    }
    value["content_sha256"] = _sha(value_body)
    _reseal_registered_json(
        scope_manifest_path=scope_path,
        registration_key=registration_key,
        value=value,
    )
    with pytest.raises(ValueError):
        validate_htr_semantic_aggregation_scope(
            root=scope_path.parent,
            source_payload=payload,
            source_array_store_root=array_root,
            expected_source_fit_seal_content_sha256="b" * 64,
            expected_source_payload_content_sha256=_sha(payload),
            expected_scope_binding_sha256="c" * 64,
        )
