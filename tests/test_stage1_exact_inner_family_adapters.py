from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    BOW_NUISANCE,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)
from oci.inference.all_evidence_fusion import FoldEvidenceProvenance
from oci.inference.embedding_native_proof_capture import (
    EMBEDDING_NATIVE_CAPTURE_SCHEMA,
    SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA,
    build_semantic_retrieval_training_only_policy,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
    SEMANTIC_RETRIEVAL_DERIVATION,
    RoleNeutralEvidenceCatalog,
    Stage1EvidenceAtom,
)
from oci.inference.stage1_exact_inner_evidence import (
    EXACT_SCOPE_CACHE_REPLAY,
    CanonicalStage1SplitRegistry,
    ExactInnerStage1FamilyRequest,
    Stage1FitRow,
    Stage1HeldoutRow,
    exact_inner_data_projection_sha256,
    produce_exact_inner_stage1_evidence_bundle,
    row_order_fingerprint,
)
from oci.inference.stage1_exact_inner_family_adapters import (
    FAMILY_NATIVE_APIS,
    NativeFamilyFitProof,
    bind_native_family_fit_proof,
    family_payload_from_catalog,
    family_producers_for_native_scope,
    native_artifact_sha256,
    native_family_code_identity,
    native_family_configuration_sha256,
    native_family_execution_record,
    native_full_outer_payload_registry_from_catalog,
    native_scope_from_catalog,
)
from oci.inference.tfidf_topic_discovery import (
    TOPIC_SCORE_TEST_SCHEMA_VERSION,
    row_set_fingerprint,
)
from oci.inference.tfidf_topic_stage1 import TFIDF_NESTED_CALIBRATION_SCHEMA_VERSION


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(value), encoding="utf-8")


def _scope_inputs():
    data = pd.DataFrame(
        {
            "_oci_row_id": list(range(12)),
            "clinical_text": [f"baseline marker {index}" for index in range(12)],
            "treatment_indicator": [index % 2 for index in range(12)],
            "outcome_indicator": [(index // 2) % 2 for index in range(12)],
        }
    )
    registry = CanonicalStage1SplitRegistry.build(
        dataset_row_ids=tuple(range(12)),
        outer_heldout_row_ids={
            1: (0, 3, 6, 9),
            2: (1, 4, 7, 10),
            3: (2, 5, 8, 11),
        },
        inner_fold_count=2,
    )
    split = registry.inner_split(1, 1)
    by_id = data.set_index("_oci_row_id")
    fit_rows = tuple(
        Stage1FitRow(
            row_id=row_id,
            text=str(by_id.loc[row_id, "clinical_text"]),
            treatment=float(by_id.loc[row_id, "treatment_indicator"]),
            outcome=float(by_id.loc[row_id, "outcome_indicator"]),
        )
        for row_id in split.fit_row_ids
    )
    heldout_rows = tuple(
        Stage1HeldoutRow(
            row_id=row_id,
            text=str(by_id.loc[row_id, "clinical_text"]),
        )
        for row_id in split.heldout_row_ids
    )
    return data, registry, split, fit_rows, heldout_rows


def _catalog(
    *,
    outer_fold: int,
    scope: str,
    inner_fold: int | None,
    fit_row_ids: tuple[int, ...],
    heldout_row_ids: tuple[int, ...],
    marker: str,
) -> RoleNeutralEvidenceCatalog:
    split_fingerprint = FoldEvidenceProvenance(
        outer_fold=outer_fold,
        train_row_ids=fit_row_ids,
        heldout_row_ids=heldout_row_ids,
        scope=scope,
        inner_fold=inner_fold,
        artifact_id=f"test-{marker}",
    ).split_fingerprint
    atoms = []
    for index, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES):
        member_ids = (f"member_{marker}_{index}",)
        content: dict[str, Any] = {
            "clinical_marker": f"{marker} {family} performance status",
        }
        atom_kind = "native_architecture_evidence"
        if family == TFIDF_SEMANTIC_RETRIEVAL:
            atom_kind = "tfidf_semantic_retrieval_contrast"
            content.update(
                {
                    "architecture_view": SEMANTIC_RETRIEVAL_DERIVATION,
                    "source_passages_removed": True,
                }
            )
        origin = {"native_family": family, "marker": marker}
        origin_json = _canonical(origin)
        content_json = _canonical(content)
        origin_sha256 = _sha(origin)
        content_sha256 = _sha(content)
        identity = {
            "atom_kind": atom_kind,
            "source_kind": "native_stage1_test_artifact",
            "source_family": family,
            "observable_axes": ("treatment",),
            "member_ids": member_ids,
            "split_fingerprint": split_fingerprint,
            "origin_sha256": origin_sha256,
            "content_sha256": content_sha256,
        }
        atoms.append(
            Stage1EvidenceAtom(
                evidence_id=f"evidence_{_sha(identity)}",
                atom_kind=atom_kind,
                source_kind="native_stage1_test_artifact",
                source_family=family,
                observable_axes=("treatment",),
                member_ids=member_ids,
                split_fingerprint=split_fingerprint,
                origin_sha256=origin_sha256,
                content_sha256=content_sha256,
                _origin_json=origin_json,
                _content_json=content_json,
            )
        )
    identity = {
        "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
        "outer_fold": outer_fold,
        "scope": scope,
        "inner_fold": inner_fold,
        "split_fingerprint": split_fingerprint,
        "atoms": [atom.as_dict() for atom in atoms],
        "non_grounding_numerical_summaries": [],
    }
    return RoleNeutralEvidenceCatalog(
        outer_fold=outer_fold,
        scope=scope,
        inner_fold=inner_fold,
        split_fingerprint=split_fingerprint,
        atoms=tuple(atoms),
        non_grounding_numerical_summaries=(),
        catalog_sha256=_sha(identity),
        _audit_json="{}",
    )


def _safe_tfidf_metadata(
    path: Path,
    *,
    fit_row_ids: tuple[int, ...],
    heldout_row_ids: tuple[int, ...],
    model_artifact_path: Path,
) -> Path:
    model_rows = fit_row_ids[::2]
    calibration_rows = fit_row_ids[1::2]
    if not model_rows or not calibration_rows:
        raise AssertionError("test scope is too small for nested calibration")
    frozen_sha256 = _sha({"fit": fit_row_ids, "selection": "frozen"})
    score = {
        "schema_version": TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "status": "completed",
        "score_selection_label_policy": "nested_fit_calibration",
        "uses_heldout_treatment_and_outcome": False,
        "uses_registered_heldout_treatment_and_outcome": False,
        "uses_nested_fit_calibration_treatment_and_outcome": True,
        "selection_frozen_sha256": frozen_sha256,
        "nested_calibration_schema_version": TFIDF_NESTED_CALIBRATION_SCHEMA_VERSION,
        "nested_model_fit_row_fingerprint": row_set_fingerprint(model_rows),
        "nested_calibration_row_fingerprint": row_set_fingerprint(calibration_rows),
        "effect_orphan_ngram_branch": {
            "uses_heldout_treatment_and_outcome": False,
            "uses_registered_heldout_treatment_and_outcome": False,
            "uses_nested_fit_calibration_treatment_and_outcome": True,
            "cluster_construction_uses_heldout_rows_or_labels": False,
            "topic_term_exclusion_is_fit_side": True,
        },
    }
    score_path = path.parent / "topic_score_tests.json"
    _write_json(score_path, score)
    metadata = {
        "fit_row_ids": list(fit_row_ids),
        "heldout_row_ids": list(heldout_row_ids),
        "fit_row_fingerprint": row_set_fingerprint(fit_row_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_row_ids),
        "score_selection_label_policy": "nested_fit_calibration",
        "registered_heldout_labels_accessed": False,
        "registered_heldout_columns_read": ["_oci_row_id", "clinical_text"],
        "selection_frozen_sha256": frozen_sha256,
        "selection_nesting": {
            "schema_version": TFIDF_NESTED_CALIBRATION_SCHEMA_VERSION,
            "policy": "nested_fit_calibration",
            "fold_parameter": "tfidf_nested_calibration_folds",
            "configured_fold_count": 3,
            "fold_count": 2,
            "selected_fold": 1,
            "canonical_hierarchy_partition_count_used": False,
            "interaction_inner_folds_used": False,
            "model_fit_row_ids": list(model_rows),
            "calibration_row_ids": list(calibration_rows),
            "model_fit_row_fingerprint": row_set_fingerprint(model_rows),
            "calibration_row_fingerprint": row_set_fingerprint(calibration_rows),
            "registered_heldout_labels_accessed": False,
            "nested_calibration_labels_accessed": True,
            "selection_frozen_before_registered_heldout_transform": True,
        },
        "topic_score_tests": score,
        "artifacts": {
            "fitted_context": str(model_artifact_path),
            "topic_score_tests": str(score_path),
        },
    }
    _write_json(path, metadata)
    return score_path


def _safe_semantic_metadata(
    path: Path,
    *,
    fit_row_ids: tuple[int, ...],
    heldout_row_ids: tuple[int, ...],
) -> None:
    policy = build_semantic_retrieval_training_only_policy(
        fit_row_ids=fit_row_ids,
        outer_fold=1,
        inner_fold=1,
        configured_fold_count=3,
        seed=19,
    )
    _write_json(
        path,
        {
            "capture_schema_version": EMBEDDING_NATIVE_CAPTURE_SCHEMA,
            "outer_fold": 1,
            "inner_fold": 1,
            "seed": 19,
            "fit_row_ids": list(fit_row_ids),
            "heldout_row_ids": list(heldout_row_ids),
            "fit_row_order_fingerprint": row_order_fingerprint(fit_row_ids),
            "heldout_row_order_fingerprint": row_order_fingerprint(heldout_row_ids),
            "registered_heldout_columns_read": ["_oci_row_id"],
            "registered_heldout_labels_accessed": False,
            "registered_heldout_text_accessed": False,
            "registered_heldout_transform_performed": False,
            "tfidf_training_scope_policy": policy,
        },
    )


def _family_proof(
    root: Path,
    *,
    family: str,
    split,
    data_projection_sha256: str,
    evidence_payload: dict[str, Any],
) -> tuple[NativeFamilyFitProof, dict[str, Any], dict[str, Path]]:
    family_dir = root / family
    family_dir.mkdir(parents=True, exist_ok=True)
    model_path = family_dir / "model.bin"
    source_path = family_dir / "native_source.json"
    metadata_path = family_dir / "fit_metadata.json"
    execution_path = family_dir / "execution.json"
    model_path.write_bytes(f"native-model::{family}".encode("utf-8"))
    _write_json(source_path, {"family": family, "native_output": True})
    if family == TFIDF_SEMANTIC_RETRIEVAL:
        _safe_semantic_metadata(
            metadata_path,
            fit_row_ids=split.fit_row_ids,
            heldout_row_ids=split.heldout_row_ids,
        )
    elif family in {TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS}:
        source_path = _safe_tfidf_metadata(
            metadata_path,
            fit_row_ids=split.fit_row_ids,
            heldout_row_ids=split.heldout_row_ids,
            model_artifact_path=model_path,
        )
    else:
        _write_json(metadata_path, {"family": family, "status": "completed"})
    configuration = {
        "family": family,
        "text_column": "clinical_text",
        "tfidf_nested_calibration_folds": 3,
        "test_hyperparameter": 1,
    }
    if family == TFIDF_SEMANTIC_RETRIEVAL:
        configuration.update(
            {
                "capture_schema_version": EMBEDDING_NATIVE_CAPTURE_SCHEMA,
                "semantic_policy_schema_version": (SEMANTIC_RETRIEVAL_TRAINING_ONLY_SCHEMA),
                "heldout_label_policy": "id_only_no_transform",
                "seed": 19,
            }
        )
    record = native_family_execution_record(
        family=family,
        fit_semantics=EXACT_SCOPE_CACHE_REPLAY,
        outer_fold=1,
        inner_fold=1,
        split_scope_fingerprint=split.scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        fit_row_ids=split.fit_row_ids,
        heldout_row_ids=split.heldout_row_ids,
        evidence_payload=evidence_payload,
        configuration=configuration,
        native_fit_metadata_path=metadata_path,
        model_artifact_path=model_path,
        source_artifact_path=source_path,
        model_artifact_semantics="sealed native model outputs and concept evidence",
    )
    _write_json(execution_path, record)
    proof = bind_native_family_fit_proof(
        family=family,
        fit_semantics=EXACT_SCOPE_CACHE_REPLAY,
        outer_fold=1,
        inner_fold=1,
        split_scope_fingerprint=split.scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        fit_row_ids=split.fit_row_ids,
        heldout_row_ids=split.heldout_row_ids,
        evidence_payload=evidence_payload,
        configuration=configuration,
        native_fit_metadata_path=metadata_path,
        native_execution_record_path=execution_path,
        model_artifact_path=model_path,
        source_artifact_path=source_path,
        model_artifact_semantics="sealed native model outputs and concept evidence",
    )
    return (
        proof,
        configuration,
        {
            "model": model_path,
            "source": source_path,
            "metadata": metadata_path,
            "execution": execution_path,
        },
    )


def _native_scope(tmp_path: Path):
    data, registry, split, fit_rows, heldout_rows = _scope_inputs()
    outer = registry.outer_splits[0]
    outer_catalog = _catalog(
        outer_fold=1,
        scope="outer_train",
        inner_fold=None,
        fit_row_ids=outer.train_row_ids,
        heldout_row_ids=outer.heldout_row_ids,
        marker="outer",
    )
    outer_catalog_path = tmp_path / "outer_catalog.json"
    _write_json(outer_catalog_path, outer_catalog.as_dict())
    outer_registry = native_full_outer_payload_registry_from_catalog(
        catalog=outer_catalog,
        outer_fold=1,
        fit_row_ids=outer.train_row_ids,
        heldout_row_ids=outer.heldout_row_ids,
        catalog_artifact_path=outer_catalog_path,
    )

    inner_catalog = _catalog(
        outer_fold=1,
        scope="inner_train",
        inner_fold=1,
        fit_row_ids=split.fit_row_ids,
        heldout_row_ids=split.heldout_row_ids,
        marker="inner",
    )
    inner_catalog_path = tmp_path / "inner_catalog.json"
    _write_json(inner_catalog_path, inner_catalog.as_dict())
    payloads = {
        family: family_payload_from_catalog(inner_catalog, family=family)[0]
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    data_projection_sha256 = exact_inner_data_projection_sha256(
        fit_rows=fit_rows,
        heldout_rows=heldout_rows,
    )
    proofs: dict[str, NativeFamilyFitProof] = {}
    configurations: dict[str, dict[str, Any]] = {}
    paths: dict[str, dict[str, Path]] = {}
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        proof, configuration, family_paths = _family_proof(
            tmp_path / "families",
            family=family,
            split=split,
            data_projection_sha256=data_projection_sha256,
            evidence_payload=payloads[family],
        )
        proofs[family] = proof
        configurations[family] = configuration
        paths[family] = family_paths
    scope = native_scope_from_catalog(
        catalog=inner_catalog,
        catalog_artifact_path=inner_catalog_path,
        full_outer_registry=outer_registry,
        outer_fold=1,
        inner_fold=1,
        split_scope_fingerprint=split.scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        fit_row_ids=split.fit_row_ids,
        heldout_row_ids=split.heldout_row_ids,
        fit_proof_by_family=proofs,
    )
    return {
        "data": data,
        "registry": registry,
        "split": split,
        "fit_rows": fit_rows,
        "heldout_rows": heldout_rows,
        "scope": scope,
        "inner_catalog": inner_catalog,
        "inner_catalog_path": inner_catalog_path,
        "outer_catalog_path": outer_catalog_path,
        "outer_registry": outer_registry,
        "payloads": payloads,
        "proofs": proofs,
        "configurations": configurations,
        "paths": paths,
    }


def test_all_ten_native_adapters_satisfy_exact_inner_contract(tmp_path: Path):
    built = _native_scope(tmp_path)
    scope = built["scope"]
    expected_configurations = {
        family: native_family_configuration_sha256(
            family,
            built["configurations"][family],
        )
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    producers = family_producers_for_native_scope(
        scope,
        expected_configuration_sha256_by_family=expected_configurations,
    )
    bundle = produce_exact_inner_stage1_evidence_bundle(
        dataset=built["data"],
        registry=built["registry"],
        outer_fold=1,
        inner_fold=1,
        producers=producers,
        full_outer_payload_sha256_by_family=scope.full_outer_payload_sha256_by_family,
    )

    assert bundle["architecture_order"] == list(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    assert len(bundle["family_artifacts"]) == 10
    for artifact in bundle["family_artifacts"]:
        family = artifact["family"]
        proof = scope.fit_proof_by_family[family]
        assert artifact["fit_semantics"] == EXACT_SCOPE_CACHE_REPLAY
        assert artifact["fit_audit"]["heldout_labels_accessed"] is False
        assert artifact["fit_audit"]["native_fit_metadata_sha256"]
        assert artifact["fit_audit"]["native_catalog_artifact_sha256"] == (
            scope.catalog_artifact_sha256
        )
        assert artifact["fit_audit"]["native_full_outer_payload_registry_sha256"] == (
            scope.full_outer_registry.content_sha256
        )
        assert artifact["fit_audit"]["producer_code_sha256"] == proof.producer_code_sha256
        assert artifact["producer_identity"]["configuration_sha256"] == (proof.configuration_sha256)
        assert artifact["producer_identity"]["native_fit_apis"] == list(FAMILY_NATIVE_APIS[family])
        assert artifact["producer_identity"]["code_sha256"] == (
            native_family_code_identity(family)["content_sha256"]
        )


def test_scope_is_catalog_only_immutable_and_rejects_wrong_order(tmp_path: Path):
    built = _native_scope(tmp_path)
    scope = built["scope"]
    detached = scope.payload(BOW_NUISANCE)
    detached["architecture_evidence"][0]["content"]["clinical_marker"] = "mutated"
    assert (
        "inner"
        in scope.payload(BOW_NUISANCE)["architecture_evidence"][0]["content"]["clinical_marker"]
    )

    with pytest.raises(TypeError, match="must be built from a catalog artifact"):
        replace(scope, _construction_authority=object())

    request = ExactInnerStage1FamilyRequest(
        family=BOW_NUISANCE,
        outer_fold=1,
        inner_fold=1,
        split_registry_sha256="a" * 64,
        split_scope_fingerprint=built["split"].scope_fingerprint,
        data_projection_sha256=scope.data_projection_sha256,
        fit_rows=tuple(reversed(built["fit_rows"])),
        heldout_rows=built["heldout_rows"],
    )
    with pytest.raises(ValueError, match="different exact-inner scope"):
        scope.validate_request(request)


def test_scope_fails_closed_on_missing_leaky_or_mutated_proof(tmp_path: Path):
    built = _native_scope(tmp_path)
    proofs = dict(built["proofs"])
    proofs.pop(BOW_NUISANCE)
    with pytest.raises(ValueError, match="exactly all ten"):
        native_scope_from_catalog(
            catalog=built["inner_catalog"],
            catalog_artifact_path=built["inner_catalog_path"],
            full_outer_registry=built["outer_registry"],
            outer_fold=1,
            inner_fold=1,
            split_scope_fingerprint=built["split"].scope_fingerprint,
            data_projection_sha256=built["scope"].data_projection_sha256,
            fit_row_ids=built["split"].fit_row_ids,
            heldout_row_ids=built["split"].heldout_row_ids,
            fit_proof_by_family=proofs,
        )

    with pytest.raises(ValueError, match="heldout_labels_accessed=false"):
        replace(
            built["proofs"][BOW_NUISANCE],
            heldout_labels_accessed=True,
        )
    with pytest.raises(ValueError, match="execution identity is not scope-bound"):
        replace(
            built["proofs"][BOW_NUISANCE],
            evidence_payload_sha256="f" * 64,
        )
    with pytest.raises(TypeError, match="must be built from verified artifacts"):
        replace(
            built["proofs"][BOW_NUISANCE],
            _construction_authority=object(),
        )


@pytest.mark.parametrize("tampered", ["model", "source", "metadata", "execution"])
def test_binder_rejects_tampered_native_artifacts(tmp_path: Path, tampered: str):
    built = _native_scope(tmp_path / "base")
    family = BOW_NUISANCE
    paths = built["paths"][family]
    if tampered == "model":
        paths[tampered].write_bytes(b"changed model")
    else:
        paths[tampered].write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="execution record differs|native artifact"):
        bind_native_family_fit_proof(
            family=family,
            fit_semantics=EXACT_SCOPE_CACHE_REPLAY,
            outer_fold=1,
            inner_fold=1,
            split_scope_fingerprint=built["split"].scope_fingerprint,
            data_projection_sha256=built["scope"].data_projection_sha256,
            fit_row_ids=built["split"].fit_row_ids,
            heldout_row_ids=built["split"].heldout_row_ids,
            evidence_payload=built["payloads"][family],
            configuration=built["configurations"][family],
            native_fit_metadata_path=paths["metadata"],
            native_execution_record_path=paths["execution"],
            model_artifact_path=paths["model"],
            source_artifact_path=paths["source"],
            model_artifact_semantics="sealed native model outputs and concept evidence",
        )


def test_producer_rejects_unregistered_configuration_or_code_hash(tmp_path: Path):
    built = _native_scope(tmp_path)
    expected = {
        family: built["proofs"][family].configuration_sha256
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    expected[BOW_NUISANCE] = _sha({"different": "configuration"})
    with pytest.raises(RuntimeError, match="configuration differs from deployment"):
        family_producers_for_native_scope(
            built["scope"],
            expected_configuration_sha256_by_family=expected,
        )
    expected_code = {
        family: built["proofs"][family].producer_code_sha256
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    expected_code[BOW_NUISANCE] = _sha({"different": "code"})
    with pytest.raises(RuntimeError, match="adapter code identity changed"):
        family_producers_for_native_scope(
            built["scope"],
            expected_code_sha256_by_family=expected_code,
        )


def test_producer_rechecks_artifact_bytes_after_proof_binding(tmp_path: Path):
    built = _native_scope(tmp_path)
    built["paths"][BOW_NUISANCE]["model"].write_bytes(b"post-bind mutation")
    with pytest.raises(RuntimeError, match="artifact changed after binding"):
        family_producers_for_native_scope(built["scope"])


@pytest.mark.parametrize("catalog_kind", ["inner", "outer"])
def test_producer_rechecks_catalog_bytes_after_scope_binding(
    tmp_path: Path,
    catalog_kind: str,
):
    built = _native_scope(tmp_path)
    built[f"{catalog_kind}_catalog_path"].write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="catalog artifact changed after binding"):
        family_producers_for_native_scope(built["scope"])


def test_full_outer_clone_registry_is_catalog_derived_and_rejects_clone(tmp_path: Path):
    built = _native_scope(tmp_path / "base")
    full_outer_hash = built["outer_registry"].payload_sha256_by_family[BOW_NUISANCE]
    assert len(full_outer_hash) == 64

    outer = built["registry"].outer_splits[0]
    copied_catalog = _catalog(
        outer_fold=1,
        scope="inner_train",
        inner_fold=1,
        fit_row_ids=built["split"].fit_row_ids,
        heldout_row_ids=built["split"].heldout_row_ids,
        marker="outer",
    )
    copied_path = tmp_path / "copied_inner_catalog.json"
    _write_json(copied_path, copied_catalog.as_dict())
    copied_payloads = {
        family: family_payload_from_catalog(copied_catalog, family=family)[0]
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    copied_proofs = {}
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        proof, _configuration, _paths = _family_proof(
            tmp_path / "copied_families",
            family=family,
            split=built["split"],
            data_projection_sha256=built["scope"].data_projection_sha256,
            evidence_payload=copied_payloads[family],
        )
        copied_proofs[family] = proof
    with pytest.raises(ValueError, match="identical to the authenticated full-outer"):
        native_scope_from_catalog(
            catalog=copied_catalog,
            catalog_artifact_path=copied_path,
            full_outer_registry=built["outer_registry"],
            outer_fold=outer.outer_fold,
            inner_fold=1,
            split_scope_fingerprint=built["split"].scope_fingerprint,
            data_projection_sha256=built["scope"].data_projection_sha256,
            fit_row_ids=built["split"].fit_row_ids,
            heldout_row_ids=built["split"].heldout_row_ids,
            fit_proof_by_family=copied_proofs,
        )


def test_semantic_projection_and_tfidf_metadata_fail_closed(tmp_path: Path):
    built = _native_scope(tmp_path)
    payload = built["payloads"][TFIDF_SEMANTIC_RETRIEVAL]
    assert payload["architecture_evidence"][0]["content"]["architecture_view"] == (
        SEMANTIC_RETRIEVAL_DERIVATION
    )
    assert payload["architecture_evidence"][0]["content"]["source_passages_removed"] is True

    family = TFIDF_TOPICS
    paths = built["paths"][family]
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata["registered_heldout_labels_accessed"] = True
    _write_json(paths["metadata"], metadata)
    with pytest.raises(ValueError, match="nested heldout-label isolation"):
        native_family_execution_record(
            family=family,
            fit_semantics=EXACT_SCOPE_CACHE_REPLAY,
            outer_fold=1,
            inner_fold=1,
            split_scope_fingerprint=built["split"].scope_fingerprint,
            data_projection_sha256=built["scope"].data_projection_sha256,
            fit_row_ids=built["split"].fit_row_ids,
            heldout_row_ids=built["split"].heldout_row_ids,
            evidence_payload=built["payloads"][family],
            configuration=built["configurations"][family],
            native_fit_metadata_path=paths["metadata"],
            model_artifact_path=paths["model"],
            source_artifact_path=paths["source"],
            model_artifact_semantics="sealed native model outputs and concept evidence",
        )


def test_family_payload_clone_canary_ignores_split_specific_member_ids():
    first = _catalog(
        outer_fold=1,
        scope="inner_train",
        inner_fold=1,
        fit_row_ids=(1, 2),
        heldout_row_ids=(3, 4),
        marker="same",
    )
    second = _catalog(
        outer_fold=1,
        scope="inner_train",
        inner_fold=2,
        fit_row_ids=(3, 4),
        heldout_row_ids=(1, 2),
        marker="same",
    )
    first_payload, first_count = family_payload_from_catalog(first, family=BOW_NUISANCE)
    second_payload, second_count = family_payload_from_catalog(second, family=BOW_NUISANCE)
    assert first_count == second_count == 1
    assert first_payload == second_payload
    assert "member_id" not in json.dumps(first_payload)


def test_native_artifact_hashes_real_file_bytes_and_closed_tree(tmp_path: Path):
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"native model bytes")
    assert native_artifact_sha256(model_file) == hashlib.sha256(b"native model bytes").hexdigest()

    artifact_tree = tmp_path / "artifact_tree"
    artifact_tree.mkdir()
    (artifact_tree / "model.bin").write_bytes(b"model")
    (artifact_tree / "evidence.json").write_text('{"fit":true}', encoding="utf-8")
    first = native_artifact_sha256(artifact_tree)
    (artifact_tree / "evidence.json").write_text('{"fit":false}', encoding="utf-8")
    assert native_artifact_sha256(artifact_tree) != first

    alias = tmp_path / "model_alias"
    alias.symlink_to(model_file)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        native_artifact_sha256(alias)
