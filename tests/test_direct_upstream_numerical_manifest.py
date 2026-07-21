from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ArchitectureDossier,
    BOW_R_LOSS,
    HTR_NEURAL,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)
from oci.inference.all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from oci.inference.direct_upstream_numerical_manifest import (
    CONDITIONAL_PRESENCE_ALIGNMENT,
    EXACT_NAMED_RAW_ALIGNMENT,
    EXACT_PRECOMMITTED_ALIGNMENT,
    PERMUTATION_SUMMARY_ALIGNMENT,
    PREAGGREGATED_PERMUTATION_SUMMARY_ALIGNMENT,
    SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON,
    build_direct_upstream_numerical_manifest,
    content_sha256,
    load_authenticated_numerical_bank_snapshot,
    selector_facing_numerical_summary,
    validate_architecture_dossier_numerical_binding,
    write_direct_upstream_numerical_manifest,
)
from oci.inference.production_coordinate_preserving_upstream_schema import (
    build_production_coordinate_preserving_schema,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group(prefix: str) -> tuple[str, ...]:
    return (
        f"closed__{prefix}__signed_mean",
        f"closed__{prefix}__absolute_max",
        f"closed__{prefix}__signed_order_001",
    )


def _write_gate_cache(tmp_path: Path) -> Path:
    source_names = (
        "stage1_calibrated__bow__linear_1_2__effect_weighted_r_tau_pred",
        "stage1_calibrated__htr__effect_weighted_r_tau_pred",
    )
    source_kinds = (
        "nested_calibrated_bow_weighted_r",
        "nested_calibrated_htr_weighted_r",
    )
    groups = (
        ("bow_nuisance", PROPENSITY_NUISANCE_FEATURE_ROLE, _group("family_001")),
        ("bow_r_loss", UNCALIBRATED_EFFECT_MODIFIER_ROLE, _group("family_002")),
        ("htr_nuisance", OUTCOME_NUISANCE_FEATURE_ROLE, _group("family_003")),
        ("htr_neural", UNCALIBRATED_EFFECT_MODIFIER_ROLE, _group("family_004")),
        (
            "matched_pair_uplift",
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            _group("family_005"),
        ),
        (
            "embedding_whole_cohort",
            PROPENSITY_NUISANCE_FEATURE_ROLE,
            _group("family_006"),
        ),
        (
            "embedding_clustered",
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            _group("family_007"),
        ),
        ("tfidf_topics", OUTCOME_NUISANCE_FEATURE_ROLE, _group("family_008")),
        (
            "tfidf_topic_contrast",
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            _group("family_009"),
        ),
        (
            "tfidf_orphan_ngrams",
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            _group("family_010"),
        ),
        (
            "neural_query_effect_moments",
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            (
                "neural_query_effect_signed_mean",
                "neural_query_effect_absolute_max",
                "neural_query_effect_signed_order_01",
            ),
        ),
    )
    feature_names = tuple(name for _kind, _role, names in groups for name in names)
    feature_kinds = tuple(kind for kind, _role, names in groups for _name in names)
    feature_roles = tuple(role for _kind, role, names in groups for _name in names)

    context_rows = [0, 1, 2, 3]
    gate_rows = [4, 5]
    configured_sources = [
        {
            "child_name": name,
            "source_kind": kind,
            "output_name": name,
            "exact_name_and_kind_required": True,
        }
        for name, kind in zip(source_names, source_kinds)
    ]
    configured_families = []
    for kind, role, names in groups:
        if kind.startswith("neural_query_"):
            configured_families.append(
                {
                    "source_kind": kind,
                    "consumer_role": role,
                    "signed_order_width": 1,
                    "required": True,
                    "reduction": "exact_preaggregated_passthrough",
                    "exact_passthrough_feature_names": list(names),
                }
            )
        else:
            configured_families.append(
                {
                    "source_kind": kind,
                    "consumer_role": role,
                    "signed_order_width": 1,
                    "required": True,
                    "summaries": [
                        "signed_mean",
                        "absolute_max",
                        "signed_descending_order",
                    ],
                }
            )
    binding = {
        "provider_identity": {
            "provider": "closed_test_provider_v1",
            "backend": {
                "backend": "stable_context_fit_upstream_backend_v2",
                "config": {
                    "namespace": "closed",
                    "calibrated_sources": configured_sources,
                    "raw_families": configured_families,
                    "reject_unconfigured_calibrated_sources": True,
                    "reject_unconfigured_raw_families": True,
                },
            },
        },
        "outer_fold": 1,
        "context_row_ids_sha256": "1" * 64,
        "context_text_sha256": "2" * 64,
        "context_treatment_sha256": "3" * 64,
        "context_outcome_sha256": "4" * 64,
        "context_inner_fold_assignment_sha256": "5" * 64,
        "gate_row_ids_sha256": "6" * 64,
        "gate_text_sha256": "7" * 64,
        "context_row_count": len(context_rows),
        "gate_row_count": len(gate_rows),
        "gate_labels_in_binding": False,
        "gate_labels_exposed_to_backend": False,
        "context_values_cross_fitted_by_exact_inner_fold": True,
    }
    cache_key = content_sha256(binding)
    root = tmp_path / cache_key
    root.mkdir(parents=True)
    arrays = {
        "calibrated_sources.npy": np.arange(4, dtype=float).reshape(2, 2),
        "calibrated_sources_context_oof.npy": np.arange(8, dtype=float).reshape(4, 2),
        "features.npy": np.arange(2 * len(feature_names), dtype=float).reshape(
            2, len(feature_names)
        ),
        "features_context_oof.npy": np.arange(4 * len(feature_names), dtype=float).reshape(
            4, len(feature_names)
        ),
    }
    # One authenticated fixed-width rank slot is structural all-zero padding.
    arrays["features.npy"][:, -1] = 0.0
    arrays["features_context_oof.npy"][:, -1] = 0.0
    for filename, values in arrays.items():
        with (root / filename).open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
    content = {
        "schema_version": "context_fit_upstream_gate_cache_v6",
        "cache_key": cache_key,
        "binding": binding,
        "context_row_ids": context_rows,
        "context_inner_fold_ids": [1, 1, 2, 2],
        "gate_row_ids": gate_rows,
        "source_names": list(source_names),
        "source_kinds": list(source_kinds),
        "source_values_file": "calibrated_sources.npy",
        "source_values_sha256": _sha256_file(root / "calibrated_sources.npy"),
        "source_context_values_file": "calibrated_sources_context_oof.npy",
        "source_context_values_sha256": _sha256_file(root / "calibrated_sources_context_oof.npy"),
        "feature_names": list(feature_names),
        "feature_kinds": list(feature_kinds),
        "feature_roles": list(feature_roles),
        "feature_values_file": "features.npy",
        "feature_values_sha256": _sha256_file(root / "features.npy"),
        "feature_context_values_file": "features_context_oof.npy",
        "feature_context_values_sha256": _sha256_file(root / "features_context_oof.npy"),
    }
    payload = {**content, "content_sha256": content_sha256(content)}
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_v3_gate_cache(tmp_path: Path) -> tuple[Path, object]:
    views = (
        "linear_unigram_c0p5",
        "linear_1_2",
        "linear_1_3",
        "linear_2_4_min_df3",
        "extratrees_1_3",
        "random_forest_1_2",
    )
    config = build_production_coordinate_preserving_schema(
        namespace="closed_v3",
        bow_view_names=views,
        source_config_sha256="f" * 64,
        cluster_max_components=1,
        tfidf_topic_count=2,
        max_orphan_features=2,
        neural_query_counts={"treatment": 1, "outcome": 1, "effect": 1},
    )
    source_names = tuple(str(item.output_name) for item in config.calibrated_sources)
    source_kinds = tuple(item.source_kind for item in config.calibrated_sources)
    schema = config.raw_output_schema()
    feature_names = tuple(item[0] for item in schema)
    feature_kinds = tuple(item[1] for item in schema)
    feature_roles = tuple(item[2] for item in schema)
    context_rows = [0, 1, 2, 3]
    gate_rows = [4, 5]
    backend_identity = {
        "backend": "coordinate_preserving_context_fit_upstream_backend_v3",
        "child": {"backend": "closed_test_all_architecture_child_v1"},
        "config": config.identity(),
        "gate_labels_exposed_to_child": False,
        "raw_features_relabelled_as_calibrated_sources": False,
        "named_raw_coordinate_alignment": "exact_child_name_kind_and_role",
        "volatile_raw_reduction": "permutation_invariant_after_named_claims",
        "child_column_consumption": "exactly_once",
        "fixed_output_order": True,
        "same_rectangular_schema_safe_for_gate_and_final_consumers": True,
    }
    binding = {
        "provider_identity": {
            "provider": "closed_test_provider_v1",
            "backend": backend_identity,
        },
        "outer_fold": 1,
        "context_row_ids_sha256": "1" * 64,
        "context_text_sha256": "2" * 64,
        "context_treatment_sha256": "3" * 64,
        "context_outcome_sha256": "4" * 64,
        "context_inner_fold_assignment_sha256": "5" * 64,
        "gate_row_ids_sha256": "6" * 64,
        "gate_text_sha256": "7" * 64,
        "context_row_count": len(context_rows),
        "gate_row_count": len(gate_rows),
        "gate_labels_in_binding": False,
        "gate_labels_exposed_to_backend": False,
        "context_values_cross_fitted_by_exact_inner_fold": True,
    }
    cache_key = content_sha256(binding)
    root = tmp_path / cache_key
    root.mkdir(parents=True)
    arrays = {
        "calibrated_sources.npy": np.arange(
            len(gate_rows) * len(source_names), dtype=float
        ).reshape(len(gate_rows), len(source_names)),
        "calibrated_sources_context_oof.npy": np.arange(
            len(context_rows) * len(source_names), dtype=float
        ).reshape(len(context_rows), len(source_names)),
        "features.npy": np.arange(len(gate_rows) * len(feature_names), dtype=float).reshape(
            len(gate_rows), len(feature_names)
        ),
        "features_context_oof.npy": np.arange(
            len(context_rows) * len(feature_names), dtype=float
        ).reshape(len(context_rows), len(feature_names)),
    }
    for filename, values in arrays.items():
        with (root / filename).open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
    content = {
        "schema_version": "context_fit_upstream_gate_cache_v6",
        "cache_key": cache_key,
        "binding": binding,
        "context_row_ids": context_rows,
        "context_inner_fold_ids": [1, 1, 2, 2],
        "gate_row_ids": gate_rows,
        "source_names": list(source_names),
        "source_kinds": list(source_kinds),
        "source_values_file": "calibrated_sources.npy",
        "source_values_sha256": _sha256_file(root / "calibrated_sources.npy"),
        "source_context_values_file": "calibrated_sources_context_oof.npy",
        "source_context_values_sha256": _sha256_file(root / "calibrated_sources_context_oof.npy"),
        "feature_names": list(feature_names),
        "feature_kinds": list(feature_kinds),
        "feature_roles": list(feature_roles),
        "feature_values_file": "features.npy",
        "feature_values_sha256": _sha256_file(root / "features.npy"),
        "feature_context_values_file": "features_context_oof.npy",
        "feature_context_values_sha256": _sha256_file(root / "features_context_oof.npy"),
    }
    payload = {**content, "content_sha256": content_sha256(content)}
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return manifest, config


def _semantic_ids() -> dict[str, tuple[str, ...]]:
    return {
        family: (f"evidence.{index:02d}",)
        for index, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES, start=1)
    }


def _manifest(tmp_path: Path):
    snapshot = load_authenticated_numerical_bank_snapshot(_write_gate_cache(tmp_path))
    return build_direct_upstream_numerical_manifest(
        snapshot,
        semantic_catalog_sha256="a" * 64,
        semantic_atom_ids_by_family=_semantic_ids(),
        numerical_zero_reasons={TFIDF_SEMANTIC_RETRIEVAL: SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON},
    )


def test_manifest_has_complete_architecture_coverage_and_honest_alignment_modes(tmp_path):
    manifest = _manifest(tmp_path)

    assert tuple(row.source_family for row in manifest.family_coverage) == (
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    assert manifest.signal_count == 35
    assert manifest.as_dict()["observed_varying_signal_count"] == 34
    assert len(manifest.family(BOW_R_LOSS).coordinate_ids) == 4
    assert len(manifest.family(HTR_NEURAL).coordinate_ids) == 7
    assert len(manifest.family(TFIDF_TOPICS).coordinate_ids) == 6
    semantic = manifest.family(TFIDF_SEMANTIC_RETRIEVAL)
    assert semantic.coordinate_ids == ()
    assert semantic.numerical_zero_reason == SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON

    calibrated = [row for row in manifest.coordinates if row.matrix_block == "calibrated_sources"]
    assert {row.alignment_mode for row in calibrated} == {EXACT_PRECOMMITTED_ALIGNMENT}
    assert all(row.source_coordinate_identity_preserved for row in calibrated)
    query = [row for row in manifest.coordinates if row.source_kind.startswith("neural_query_")]
    assert {row.alignment_mode for row in query} == {PREAGGREGATED_PERMUTATION_SUMMARY_ALIGNMENT}
    reduced = [
        row
        for row in manifest.coordinates
        if row.matrix_block == "raw_features" and row not in query
    ]
    assert {row.alignment_mode for row in reduced} == {PERMUTATION_SUMMARY_ALIGNMENT}
    assert all(not row.source_coordinate_identity_preserved for row in reduced + query)
    assert all(row.concept_grounding_allowed is False for row in manifest.coordinates)


def test_v3_manifest_distinguishes_exact_named_presence_and_volatile_coordinates(tmp_path):
    source_manifest, config = _write_v3_gate_cache(tmp_path)
    snapshot = load_authenticated_numerical_bank_snapshot(source_manifest)
    manifest = build_direct_upstream_numerical_manifest(
        snapshot,
        semantic_catalog_sha256="e" * 64,
        semantic_atom_ids_by_family=_semantic_ids(),
        numerical_zero_reasons={TFIDF_SEMANTIC_RETRIEVAL: SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON},
    )

    raw = [row for row in manifest.coordinates if row.matrix_block == "raw_features"]
    exact = [row for row in raw if row.alignment_mode == EXACT_NAMED_RAW_ALIGNMENT]
    presence = [row for row in raw if row.alignment_mode == CONDITIONAL_PRESENCE_ALIGNMENT]
    volatile = [row for row in raw if row.alignment_mode == PERMUTATION_SUMMARY_ALIGNMENT]
    assert len(exact) == len(config.named_raw_coordinates)
    assert len(presence) == sum(not item.required for item in config.named_raw_coordinates)
    assert len(volatile) == sum(
        item.signed_order_width + 2 for item in config.volatile_raw_families
    )
    assert all(row.statistic_kind == "exact_named_coordinate" for row in exact)
    assert all(row.source_coordinate_identity_preserved for row in exact)
    assert all(not row.source_coordinate_identity_preserved for row in presence + volatile)
    assert {row.statistic_kind for row in presence} == {"presence"}
    assert {row.alignment_mode for row in raw} == {
        EXACT_NAMED_RAW_ALIGNMENT,
        CONDITIONAL_PRESENCE_ALIGNMENT,
        PERMUTATION_SUMMARY_ALIGNMENT,
    }
    query = [row for row in raw if row.source_kind.startswith("neural_query_")]
    assert {row.alignment_mode for row in query} == {EXACT_NAMED_RAW_ALIGNMENT}
    assert any(row.producer_subarchitecture == "bow_nuisance_view:linear_1_2" for row in exact)
    assert tuple(row.source_family for row in manifest.family_coverage) == (
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )


def test_v3_manifest_rejects_output_metadata_not_bound_by_structured_config(tmp_path):
    source_manifest, _config = _write_v3_gate_cache(tmp_path)
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    payload["feature_names"][0] += "_changed"
    content = {key: value for key, value in payload.items() if key != "content_sha256"}
    payload["content_sha256"] = content_sha256(content)
    source_manifest.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differs from the structured v3 config"):
        load_authenticated_numerical_bank_snapshot(source_manifest)


def test_selector_view_omits_coordinate_names_atom_ids_and_row_values(tmp_path):
    manifest = _manifest(tmp_path)
    view = selector_facing_numerical_summary(
        manifest,
        strength_by_family={BOW_R_LOSS: {"available": True, "mean_abs_strength": 0.2}},
    )
    wire = json.dumps(view, sort_keys=True)

    assert view["manifest_sha256"] == manifest.content_sha256
    assert view["signal_count"] == manifest.signal_count
    assert view["observed_varying_signal_count"] == 34
    assert "family_001__signed_mean" not in wire
    assert "evidence.01" not in wire
    assert "matrix" not in wire
    assert view["concept_grounding_allowed"] is False
    with pytest.raises(ValueError, match="cannot link"):
        selector_facing_numerical_summary(
            manifest,
            strength_by_family={BOW_R_LOSS: {"feature_name_strength": 0.2}},
        )


def test_manifest_sidecar_is_immutable_and_reauthenticated(tmp_path):
    manifest = _manifest(tmp_path / "source")
    destination = tmp_path / "sidecar" / "direct_upstream_numerical_manifest.json"
    persisted = write_direct_upstream_numerical_manifest(manifest, destination)
    persisted.verify()
    replay = write_direct_upstream_numerical_manifest(manifest, destination)
    assert replay.file_sha256 == persisted.file_sha256

    destination.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes changed"):
        persisted.verify()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_direct_upstream_numerical_manifest(manifest, destination)


def test_source_matrix_tampering_and_missing_zero_reason_fail_closed(tmp_path):
    source_manifest = _write_gate_cache(tmp_path / "tamper")
    (source_manifest.parent / "features.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="authentication"):
        load_authenticated_numerical_bank_snapshot(source_manifest)

    good_snapshot = load_authenticated_numerical_bank_snapshot(
        _write_gate_cache(tmp_path / "missing_zero")
    )
    with pytest.raises(ValueError, match="zero reason"):
        build_direct_upstream_numerical_manifest(
            good_snapshot,
            semantic_catalog_sha256="b" * 64,
            semantic_atom_ids_by_family=_semantic_ids(),
        )


def test_semantic_atom_bindings_are_complete_and_not_coordinate_paired(tmp_path):
    snapshot = load_authenticated_numerical_bank_snapshot(_write_gate_cache(tmp_path))
    incomplete = _semantic_ids()
    incomplete.pop(TFIDF_SEMANTIC_RETRIEVAL)
    with pytest.raises(ValueError, match="cover every active"):
        build_direct_upstream_numerical_manifest(
            snapshot,
            semantic_catalog_sha256="c" * 64,
            semantic_atom_ids_by_family=incomplete,
            numerical_zero_reasons={
                TFIDF_SEMANTIC_RETRIEVAL: SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON
            },
        )

    manifest = build_direct_upstream_numerical_manifest(
        snapshot,
        semantic_catalog_sha256="c" * 64,
        semantic_atom_ids_by_family=_semantic_ids(),
        numerical_zero_reasons={TFIDF_SEMANTIC_RETRIEVAL: SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON},
    )
    assert manifest.as_dict()["coordinate_to_semantic_atom_linkage"] is False
    assert all("semantic_atom_ids" not in row.as_dict() for row in manifest.coordinates)


def test_architecture_dossier_must_bind_exact_family_slice(tmp_path):
    manifest = _manifest(tmp_path)
    coverage = manifest.family(BOW_R_LOSS)
    dossier = ArchitectureDossier(
        source_family=BOW_R_LOSS,
        catalog_sha256=manifest.semantic_catalog_sha256,
        catalog_evidence_ids=coverage.semantic_atom_ids,
        coverage_disposition_ids=coverage.semantic_atom_ids,
        coverage_audit_sha256="d" * 64,
        architecture_candidates=(),
        direct_numerical_manifest_sha256=manifest.content_sha256,
        direct_numerical_signal_count=len(coverage.coordinate_ids),
    )
    validate_architecture_dossier_numerical_binding(dossier, manifest)

    with pytest.raises(ValueError, match="signal count"):
        validate_architecture_dossier_numerical_binding(
            replace(
                dossier,
                direct_numerical_signal_count=dossier.direct_numerical_signal_count - 1,
            ),
            manifest,
        )
