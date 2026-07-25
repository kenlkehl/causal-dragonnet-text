from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import oci.inference.direct_upstream_numerical_reference_bank as reference_module
import oci.inference.production_role_neutral_stage2_handoff as handoff_module
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
)
from oci.inference.all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from oci.inference.production_neural_query_binary_layout import (
    write_npy_array_set,
)
from oci.inference.production_stage1_scope_scheduler import (
    Stage1ScopePlan,
    build_canonical_stage1_scope_plan,
)


def _sha_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _closed(value: dict) -> dict:
    body = dict(value)
    return {**body, "content_sha256": _sha_json(body)}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_registration(path: Path, *, root: Path, content_sha256: str) -> dict:
    payload = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "content_sha256": content_sha256,
    }


def _array_registration(
    root: Path,
    relative_path: str,
    values: np.ndarray,
    *,
    columns: tuple[str, ...] | None = None,
) -> dict:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(values)
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    payload = path.read_bytes()
    registration = {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "content_sha256": _sha_json(
            {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "values": array.tolist(),
            }
        ),
    }
    if columns is not None:
        registration["columns"] = list(columns)
    return registration


def _view_registration(root: Path, relative_path: str, body: dict) -> dict:
    view = _closed(body)
    path = root / relative_path
    _write_json(path, view)
    return {
        "logical_scope_id": body["logical_scope_id"],
        "family": body["family"],
        **_file_registration(
            path,
            root=root,
            content_sha256=view["content_sha256"],
        ),
    }


def _registry() -> dict:
    return {
        "dataset_row_count": 8,
        "outer_folds": [
            {
                "outer_fold": 1,
                "fit_row_ids": [4, 5, 6, 7],
                "heldout_row_ids": [0, 1, 2, 3],
                "inner_folds": [
                    {
                        "inner_fold": 1,
                        "fit_row_ids": [5, 7],
                        "heldout_row_ids": [4, 6],
                    },
                    {
                        "inner_fold": 2,
                        "fit_row_ids": [4, 6],
                        "heldout_row_ids": [5, 7],
                    },
                ],
            },
            {
                "outer_fold": 2,
                "fit_row_ids": [0, 1, 2, 3],
                "heldout_row_ids": [4, 5, 6, 7],
                "inner_folds": [
                    {
                        "inner_fold": 1,
                        "fit_row_ids": [1, 3],
                        "heldout_row_ids": [0, 2],
                    },
                    {
                        "inner_fold": 2,
                        "fit_row_ids": [0, 2],
                        "heldout_row_ids": [1, 3],
                    },
                ],
            },
        ],
    }


def _plan() -> Stage1ScopePlan:
    return build_canonical_stage1_scope_plan(
        registry=_registry(),
        registry_content_sha256=_sha_json(_registry()),
        global_seed=42,
        gpu_ids=(),
        review_rounds=1,
        initial_training_partitions=1,
        expected_outer_fold_count=2,
        expected_inner_fold_count=2,
    )


def _primary_body(scope, family: str, **extra) -> dict:
    return {
        "schema_version": f"test_{family}_logical_view_v1",
        "logical_scope_id": scope.scope_id,
        "logical_scope_sha256": scope.as_dict()["scope_sha256"],
        "logical_purpose": scope.scope_kind,
        "physical_owner_scope_id": scope.scope_id,
        "family": family,
        "logical_heldout_row_ids": list(scope.heldout_row_ids),
        "logical_transform_performed": True,
        "registered_heldout_labels_accessed": False,
        **extra,
    }


def _write_simple_dense_component(
    *,
    component_root: Path,
    scope,
    families_and_columns: tuple[tuple[str, tuple[str, ...]], ...],
    base: float,
) -> None:
    logical_views = []
    for family_index, (family, columns) in enumerate(families_and_columns):
        values = (
            np.arange(len(scope.heldout_row_ids) * len(columns), dtype=np.float64)
            .reshape(len(scope.heldout_row_ids), len(columns))
            + base
            + family_index
        )
        artifact = _array_registration(
            component_root,
            f"logical_views/{scope.scope_id}.{family}.predictions.npy",
            values,
            columns=columns,
        )
        logical_views.append(
            _view_registration(
                component_root,
                f"logical_views/{scope.scope_id}.{family}.json",
                _primary_body(
                    scope,
                    family,
                    prediction_artifact=artifact,
                ),
            )
        )
    terminal = _closed(
        {
            "schema_version": f"test_{component_root.name}_terminal_v1",
            "status": "complete",
            "logical_views": logical_views,
            "registered_heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
        }
    )
    _write_json(component_root / "execution_manifest.json", terminal)


def _write_matched_component(
    component_root: Path,
    scope,
    *,
    base: float,
) -> None:
    artifacts = {}
    for offset, subproducer in enumerate(("bow", "htr")):
        columns = tuple(
            f"{subproducer}::{name}"
            for name in ("delta", "probability", "n_controls")
        )
        values = (
            np.arange(len(scope.heldout_row_ids) * 3, dtype=np.float64)
            .reshape(len(scope.heldout_row_ids), 3)
            + base
            + offset
        )
        artifacts[subproducer] = _array_registration(
            component_root,
            f"logical_views/{scope.scope_id}.{subproducer}.predictions.npy",
            values,
            columns=columns,
        )
    registration = _view_registration(
        component_root,
        f"logical_views/{scope.scope_id}.json",
        _primary_body(
            scope,
            MATCHED_PAIR_UPLIFT,
            prediction_artifacts=artifacts,
        ),
    )
    _write_json(
        component_root / "execution_manifest.json",
        _closed(
            {
                "schema_version": "test_matched_terminal_v1",
                "status": "complete",
                "logical_views": [registration],
                "registered_heldout_labels_accessed": False,
                "oracle_fields_accessed": False,
            }
        ),
    )


def _write_embedding_component(
    component_root: Path,
    scope,
    *,
    base: float,
    vocabulary_width: int,
) -> None:
    vocabulary = tuple(f"term_{index:03d}" for index in range(vocabulary_width))
    vocabulary_value = _closed(
        {
            "schema_version": (
                "production_role_neutral_embedding_semantic_vocabulary_v1"
            ),
            "terms": list(vocabulary),
            "term_count": len(vocabulary),
            "feature_indices": list(range(len(vocabulary))),
            "all_configured_terms_accounted_once": True,
            "semantic_term_truncation_applied": False,
        }
    )
    vocabulary_path = component_root / "fit_state" / "semantic_vocabulary.json"
    _write_json(vocabulary_path, vocabulary_value)
    vocabulary_registration = _file_registration(
        vocabulary_path,
        root=component_root,
        content_sha256=vocabulary_value["content_sha256"],
    )
    contrasts = [
        {
            "name": "treatment",
            "contrast_family": "marginal",
            "target_source": "fit_treatment",
        },
        {
            "name": "outcome",
            "contrast_family": "marginal",
            "target_source": "fit_outcome",
        },
        {
            "name": "r_pseudo_target",
            "contrast_family": "r_pseudo_target",
            "target_source": "fit_r_pseudo_from_authenticated_bow_nuisance",
        },
    ]
    _write_json(
        component_root / "fit_state" / "metadata.json",
        _closed(
            {
                "schema_version": "test_embedding_fit_state_v1",
                "scientific_configuration": {"contrasts": contrasts},
                "semantic_vocabulary": vocabulary_registration,
            }
        ),
    )
    row_count = len(scope.heldout_row_ids)
    whole = _array_registration(
        component_root,
        "exact_transforms/arrays/heldout_whole_patient_scores.npy",
        np.arange(row_count * 3, dtype=np.float64).reshape(row_count, 3)
        + base,
    )
    distances = _array_registration(
        component_root,
        "exact_transforms/arrays/heldout_cluster_distances.npy",
        np.arange(row_count * 2, dtype=np.float64).reshape(row_count, 2)
        + base,
    )
    svd = _array_registration(
        component_root,
        "exact_transforms/arrays/heldout_cluster_svd_0_projections.npy",
        np.arange(row_count * 2, dtype=np.float64).reshape(row_count, 2)
        - base,
    )
    dense_lexical = (
        np.arange(row_count * vocabulary_width, dtype=np.float64)
        .reshape(row_count, vocabulary_width)
        + base
        + 1.0
    )
    csr = sparse.csr_matrix(dense_lexical)
    csr_registrations = [
        _array_registration(
            component_root,
            f"exact_transforms/arrays/heldout_lexical_csr_{name}.npy",
            np.asarray(values),
        )
        for name, values in (
            ("data", csr.data),
            ("indices", csr.indices.astype(np.int64)),
            ("indptr", csr.indptr.astype(np.int64)),
        )
    ]
    exact_metadata = _closed(
        {
            "schema_version": "production_role_neutral_embedding_exact_transform_v1",
            "heldout_row_ids": list(scope.heldout_row_ids),
            "transform_metadata": {
                "whole_contrast_names": [row["name"] for row in contrasts],
                "cluster_svd_projections": [
                    {
                        "family_key": "cluster_000",
                        "array_key": (
                            "heldout_cluster_svd_0_projections"
                        ),
                        "component_count": 2,
                    }
                ],
                "lexical_csr_shape": [row_count, vocabulary_width],
            },
            "registered_heldout_labels_accessed": False,
        }
    )
    _write_json(
        component_root / "exact_transforms" / "metadata.json",
        exact_metadata,
    )
    views = [
        _view_registration(
            component_root,
            f"logical_views/{scope.scope_id}.{EMBEDDING_WHOLE_COHORT}.json",
            _primary_body(
                scope,
                EMBEDDING_WHOLE_COHORT,
                exact_transform_content_sha256=exact_metadata["content_sha256"],
                prediction_artifacts=[whole],
            ),
        ),
        _view_registration(
            component_root,
            f"logical_views/{scope.scope_id}.{EMBEDDING_CLUSTERED}.json",
            _primary_body(
                scope,
                EMBEDDING_CLUSTERED,
                exact_transform_content_sha256=exact_metadata["content_sha256"],
                prediction_artifacts=[distances, svd],
            ),
        ),
        _view_registration(
            component_root,
            (
                f"logical_views/{scope.scope_id}."
                f"{TFIDF_SEMANTIC_RETRIEVAL}.json"
            ),
            _primary_body(
                scope,
                TFIDF_SEMANTIC_RETRIEVAL,
                exact_transform_content_sha256=exact_metadata["content_sha256"],
                prediction_artifacts=csr_registrations,
            ),
        ),
    ]
    _write_json(
        component_root / "execution_manifest.json",
        _closed(
            {
                "schema_version": "test_embedding_terminal_v1",
                "status": "complete",
                "logical_views": views,
                "registered_heldout_labels_accessed": False,
                "oracle_fields_accessed": False,
            }
        ),
    )


def _write_neural_query_component(
    component_root: Path,
    scope,
    *,
    base: float,
) -> None:
    names = (
        "neural_query_treatment_signed_mean",
        "neural_query_outcome_signed_mean",
        "neural_query_effect_signed_mean",
    )
    kinds = (
        "neural_query_treatment_moments",
        "neural_query_outcome_moments",
        "neural_query_effect_moments",
    )
    roles = (
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    )
    prediction_root = component_root / "logical_views" / "primary_predictions"
    descriptor = write_npy_array_set(
        prediction_root,
        {
            "gate_row_ids": np.asarray(scope.heldout_row_ids, dtype=np.int64),
            "feature_values": (
                np.arange(len(scope.heldout_row_ids) * 3, dtype=np.float64)
                .reshape(len(scope.heldout_row_ids), 3)
                + base
            ),
        },
        ordered_names=("gate_row_ids", "feature_values"),
    )
    artifact = {
        "relative_path": "logical_views/primary_predictions",
        "array_order": ["gate_row_ids", "feature_values"],
        "array_inventory": descriptor["array_inventory"],
        "index_sha256": descriptor["index_sha256"],
        "arrays_content_sha256": descriptor["content_sha256"],
        "feature_names": list(names),
        "feature_kinds": list(kinds),
        "feature_roles": list(roles),
        "feature_count": len(names),
        "row_count": len(scope.heldout_row_ids),
        "heldout_labels_present": False,
    }
    registration = _view_registration(
        component_root,
        "logical_views/000_primary.json",
        _primary_body(
            scope,
            NEURAL_QUERY_MOMENTS,
            prediction_artifact=artifact,
        ),
    )
    _write_json(
        component_root / "execution_manifest.json",
        _closed(
            {
                "schema_version": "test_neural_query_terminal_v1",
                "status": "complete",
                "logical_views": [registration],
                "registered_heldout_labels_accessed": False,
                "oracle_fields_accessed": False,
            }
        ),
    )


def _write_execution(
    tmp_path: Path,
    *,
    missing_family: str | None = None,
    reordered_bow_rows: bool = False,
) -> tuple[Path, Stage1ScopePlan, dict]:
    plan = _plan()
    execution_root = (tmp_path / "execution").resolve()
    execution_root.mkdir()
    for owner_index, scope in enumerate(plan.physical_scopes):
        owner_root = execution_root / "components" / scope.scope_id
        base = float(10 * (owner_index + 1))
        _write_simple_dense_component(
            component_root=owner_root / "bow",
            scope=scope,
            families_and_columns=(
                (
                    BOW_NUISANCE,
                    (
                        "linear::treatment_nuisance",
                        "linear::outcome_nuisance",
                    ),
                ),
                (
                    BOW_R_LOSS,
                    (
                        "linear::effect_pseudo_target",
                        "linear::effect_weighted_r",
                    ),
                ),
            ),
            base=base,
        )
        _write_simple_dense_component(
            component_root=owner_root / "htr",
            scope=scope,
            families_and_columns=(
                (
                    HTR_NEURAL,
                    (
                        "htr_nuisance::e_hat",
                        "htr_nuisance::m_hat",
                        "htr_effect::pseudo_outcome_mse",
                    ),
                ),
            ),
            base=base + 1,
        )
        _write_matched_component(
            owner_root / "matched_pair",
            scope,
            base=base + 2,
        )
        _write_embedding_component(
            owner_root / "embeddings",
            scope,
            base=base + 3,
            vocabulary_width=2 + owner_index % 2,
        )
        _write_simple_dense_component(
            component_root=owner_root / "tfidf",
            scope=scope,
            families_and_columns=(
                (
                    TFIDF_TOPICS,
                    (
                        "treatment::topic_000",
                        "outcome::topic_000",
                        "effect::topic_000",
                    ),
                ),
                (
                    TFIDF_ORPHAN_NGRAMS,
                    tuple(
                        f"residual_tfidf::term_{index}"
                        for index in range(2 + owner_index % 2)
                    ),
                ),
            ),
            base=base + 4,
        )
        _write_neural_query_component(
            owner_root / "neural_query",
            scope,
            base=base + 5,
        )

    if missing_family is not None:
        owner = plan.physical_scopes[0]
        component = (
            "embeddings"
            if missing_family
            in {
                EMBEDDING_WHOLE_COHORT,
                EMBEDDING_CLUSTERED,
                TFIDF_SEMANTIC_RETRIEVAL,
            }
            else "bow"
        )
        terminal_path = (
            execution_root
            / "components"
            / owner.scope_id
            / component
            / "execution_manifest.json"
        )
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal["logical_views"] = [
            row
            for row in terminal["logical_views"]
            if row.get("family") != missing_family
        ]
        body = {
            key: value
            for key, value in terminal.items()
            if key != "content_sha256"
        }
        _write_json(terminal_path, _closed(body))

    if reordered_bow_rows:
        owner = plan.physical_scopes[0]
        component_root = execution_root / "components" / owner.scope_id / "bow"
        terminal_path = component_root / "execution_manifest.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        registration = next(
            row
            for row in terminal["logical_views"]
            if row["family"] == BOW_NUISANCE
        )
        view_path = component_root / registration["relative_path"]
        view = json.loads(view_path.read_text(encoding="utf-8"))
        view["logical_heldout_row_ids"] = list(
            reversed(view["logical_heldout_row_ids"])
        )
        body = {
            key: value for key, value in view.items() if key != "content_sha256"
        }
        view = _closed(body)
        _write_json(view_path, view)
        updated = _file_registration(
            view_path,
            root=component_root,
            content_sha256=view["content_sha256"],
        )
        registration.update(updated)
        terminal_body = {
            key: value
            for key, value in terminal.items()
            if key != "content_sha256"
        }
        _write_json(terminal_path, _closed(terminal_body))

    execution_manifest = _closed(
        {
            "schema_version": "test_role_neutral_execution_v1",
            "status": "complete",
            "plan_scientific_content_sha256": plan.scientific_content_sha256,
        }
    )
    _write_json(
        execution_root / "execution_manifest.json",
        execution_manifest,
    )
    return execution_root, plan, execution_manifest


def _trust_test_execution(monkeypatch, manifest: dict) -> None:
    monkeypatch.setattr(
        reference_module,
        "validate_role_neutral_stage1_execution",
        lambda *, root, plan: dict(manifest),
    )


def _bind_test_bank(
    bank,
    monkeypatch,
    *,
    owner_proof_overrides: dict[str, str] | None = None,
) -> dict:
    overrides = dict(owner_proof_overrides or {})
    owner_proofs = [
        {
            "physical_owner_scope_id": owner.scope_id,
            "projection_proof_content_sha256": overrides.get(
                owner.scope_id,
                _sha_json({"owner": owner.scope_id}),
            ),
        }
        for owner in bank.plan.physical_scopes
    ]
    body = {
        "schema_version": (
            "authenticated_role_neutral_prepared_cohort_projection_binding_v1"
        ),
        "plan_scientific_content_sha256": (
            bank.plan.scientific_content_sha256
        ),
        "prepared_request_sha256": _sha_json({"prepared_request": "test"}),
        "source_execution_content_sha256": bank.manifest[
            "source_execution_content_sha256"
        ],
        "provider_identity_sha256": _sha_json({"provider": "test"}),
        "prepared_cohort_artifact_sha256": _sha_json(
            {"prepared_cohort": "test"}
        ),
        "row_map_sha256": _sha_json({"row_map": "test"}),
        "row_count": len(
            {
                row_id
                for scope in bank.plan.scopes
                for row_id in (*scope.fit_row_ids, *scope.heldout_row_ids)
            }
        ),
        "unit_id_column": "configured_unit",
        "text_column": "configured_text",
        "treatment_column": "configured_treatment",
        "outcome_column": "configured_outcome",
        "physical_owner_projection_proofs": owner_proofs,
        "all_physical_fit_projections_verified": True,
        "raw_text_persisted": False,
        "raw_treatment_persisted": False,
        "raw_outcome_persisted": False,
        "text_truncation_applied": False,
    }
    payload = {**body, "content_sha256": _sha_json(body)}
    token = object()

    def _validate(
        value,
        *,
        expected_plan_scientific_content_sha256,
        expected_source_execution_content_sha256,
    ):
        assert value is token
        assert (
            expected_plan_scientific_content_sha256
            == bank.plan.scientific_content_sha256
        )
        assert (
            expected_source_execution_content_sha256
            == bank.manifest["source_execution_content_sha256"]
        )
        return dict(payload)

    monkeypatch.setattr(
        reference_module,
        "validate_authenticated_prepared_projection_binding",
        _validate,
    )
    assert bank.bind_prepared_projection(token) is bank
    fold_bindings = []
    for full in (
        scope for scope in bank.plan.scopes if scope.scope_kind == "full_outer"
    ):
        meta_by_row = {
            row_id: int(inner.inner_fold)
            for inner in bank.plan.scopes
            if inner.scope_kind == "exact_inner"
            and inner.outer_fold == full.outer_fold
            for row_id in inner.heldout_row_ids
        }
        fold_bindings.append(
            {
                "outer_fold": full.outer_fold,
                "outer_train_row_ids": list(full.fit_row_ids),
                "outer_heldout_row_ids": list(full.heldout_row_ids),
                "meta_inner_fold_ids": [
                    meta_by_row[row_id] for row_id in full.fit_row_ids
                ],
                "outer_train_row_count": len(full.fit_row_ids),
                "outer_heldout_row_count": len(full.heldout_row_ids),
            }
        )
    runtime_body = {
        "schema_version": "authenticated_role_neutral_stage2_runtime_binding_v1",
        "plan_scientific_content_sha256": bank.plan.scientific_content_sha256,
        "prepared_request_sha256": payload["prepared_request_sha256"],
        "source_execution_content_sha256": payload[
            "source_execution_content_sha256"
        ],
        "provider_identity_sha256": payload["provider_identity_sha256"],
        "runner_dataset_artifact_sha256": payload[
            "prepared_cohort_artifact_sha256"
        ],
        "prepared_projection_binding_content_sha256": payload["content_sha256"],
        "row_map_sha256": payload["row_map_sha256"],
        "fold_bindings": fold_bindings,
        "runner_dataset_matches_prepared_projection": True,
        "fold_row_order_and_meta_assignments_precommitted": True,
        "per_fold_text_treatment_outcome_rehash_required": False,
        "outer_heldout_labels_authorized": False,
    }
    runtime_payload = {
        **runtime_body,
        "content_sha256": _sha_json(runtime_body),
    }

    class _TestRuntimeToken:
        def authorize_final_fold_shapes(self, **kwargs):
            matches = [
                row
                for row in fold_bindings
                if row["outer_fold"] == kwargs["outer_fold"]
            ]
            assert len(matches) == 1
            row = matches[0]
            if (
                list(kwargs["exact_outer_train_row_ids"])
                != row["outer_train_row_ids"]
                or list(kwargs["exact_outer_heldout_row_ids"])
                != row["outer_heldout_row_ids"]
                or list(kwargs["exact_meta_inner_fold_ids"])
                != row["meta_inner_fold_ids"]
                or (
                    kwargs["outer_train_text_count"],
                    kwargs["outer_train_treatment_count"],
                    kwargs["outer_train_outcome_count"],
                    kwargs["outer_heldout_text_count"],
                )
                != (
                    row["outer_train_row_count"],
                    row["outer_train_row_count"],
                    row["outer_train_row_count"],
                    row["outer_heldout_row_count"],
                )
            ):
                raise ValueError(
                    "runner fold assignments differ: rows, meta assignments, "
                    "or observable shapes changed"
                )
            return {"per_fold_text_treatment_outcome_rehashed": False}

    runtime_token = _TestRuntimeToken()

    def _validate_runtime(
        value,
        *,
        expected_plan_scientific_content_sha256,
        expected_source_execution_content_sha256,
    ):
        assert value is runtime_token
        assert expected_plan_scientific_content_sha256 == (
            bank.plan.scientific_content_sha256
        )
        assert expected_source_execution_content_sha256 == (
            bank.manifest["source_execution_content_sha256"]
        )
        return dict(runtime_payload)

    monkeypatch.setattr(
        reference_module,
        "validate_authenticated_role_neutral_stage2_runtime_binding",
        _validate_runtime,
    )
    assert bank.bind_runtime_authorization(runtime_token) is bank
    return payload


def test_reference_bank_is_all_ten_oof_full_outer_and_no_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution_root, plan, execution_manifest = _write_execution(tmp_path)
    _trust_test_execution(monkeypatch, execution_manifest)
    source_stats = {
        path.relative_to(execution_root).as_posix(): (
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in execution_root.rglob("*.npy")
    }

    bank = reference_module.publish_role_neutral_direct_numerical_reference_bank(
        root=(tmp_path / "reference_bank").resolve(),
        execution_root=execution_root,
        plan=plan,
        execution_manifest=execution_manifest,
    )

    assert {
        path.name for path in bank.manifest_path.parent.iterdir()
    } == {
        reference_module.DIRECT_NUMERICAL_REFERENCE_MANIFEST,
        reference_module.DIRECT_NUMERICAL_REFERENCE_LOCATOR,
    }
    assert not tuple(bank.manifest_path.parent.rglob("*.npy"))
    assert bank.manifest["source_numerical_payloads_copied"] is False
    assert bank.manifest["combined_npy_payloads_persisted"] is False
    assert {
        row["source_family"] for row in bank.manifest["family_coverage"]
    } == set(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    assert all(
        row["coordinate_ids"] for row in bank.manifest["family_coverage"]
    )
    whole_ids = {
        row["coordinate_id"]
        for row in bank.manifest["coordinates"]
        if row["source_family"] == EMBEDDING_WHOLE_COHORT
    }
    cluster_ids = {
        row["coordinate_id"]
        for row in bank.manifest["coordinates"]
        if row["source_family"] == EMBEDDING_CLUSTERED
    }
    assert whole_ids
    assert cluster_ids
    assert whole_ids.isdisjoint(cluster_ids)

    full = next(
        scope
        for scope in plan.scopes
        if scope.scope_kind == "full_outer" and scope.outer_fold == 1
    )
    canonical_train = tuple(full.fit_row_ids)
    canonical_heldout = tuple(full.heldout_row_ids)
    with pytest.raises(RuntimeError, match="bind_runtime_authorization"):
        bank.get_meta_inner_fold_ids(
            outer_fold=1,
            exact_outer_train_row_ids=canonical_train,
        )
    _bind_test_bank(bank, monkeypatch)
    audit_before = bank.payload_cache_audit()
    fold_ids = bank.get_meta_inner_fold_ids(
        outer_fold=1,
        exact_outer_train_row_ids=canonical_train,
    )
    assert len(fold_ids) == len(canonical_train)
    assert set(fold_ids) == {1, 2}
    view = bank.fold_view(
        outer_fold=1,
        exact_outer_train_row_ids=canonical_train,
        exact_outer_heldout_row_ids=canonical_heldout,
    )
    train = view.materialize(scope=reference_module.OUTER_TRAIN_OOF_SCOPE)
    heldout = view.materialize(scope=reference_module.OUTER_HELDOUT_SCOPE)
    assert train.row_ids == canonical_train
    assert heldout.row_ids == canonical_heldout
    assert train.coordinate_ids == heldout.coordinate_ids
    assert set(train.source_families) == set(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    assert np.isfinite(train.values).all()
    assert np.isfinite(heldout.values).all()
    assert train.values.flags.writeable is False
    forest = view.forest_blocks()
    assert forest.effect_train_values.shape[0] == len(canonical_train)
    assert forest.control_train_values.shape[0] == len(canonical_train)
    assert forest.effect_train_values.shape[1] > 0
    assert forest.control_train_values.shape[1] > 0
    audit_after = bank.payload_cache_audit()
    assert audit_after == audit_before
    assert audit_after["unique_payload_file_count"] > 0
    assert (
        audit_after["byte_authenticated_payload_file_count"]
        + audit_after["externally_authenticated_payload_file_count"]
        == audit_after["unique_payload_file_count"]
    )
    assert (
        audit_after["ordinary_materialization_payload_file_open_count"]
        == 0
    )

    after_stats = {
        path.relative_to(execution_root).as_posix(): (
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in execution_root.rglob("*.npy")
    }
    assert after_stats == source_stats


def test_gate_only_view_uses_cumulative_primary_and_rejects_context_oof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution_root, plan, execution_manifest = _write_execution(tmp_path)
    _trust_test_execution(monkeypatch, execution_manifest)
    bank = reference_module.publish_role_neutral_direct_numerical_reference_bank(
        root=(tmp_path / "reference_bank").resolve(),
        execution_root=execution_root,
        plan=plan,
        execution_manifest=execution_manifest,
    )
    _bind_test_bank(bank, monkeypatch)
    cumulative = next(
        scope
        for scope in plan.scopes
        if scope.scope_kind == "cumulative_spent"
        and scope.outer_fold == 1
        and scope.context_epoch == 0
    )
    next_gate = next(
        scope
        for scope in plan.scopes
        if scope.scope_kind == "exact_inner"
        and scope.outer_fold == 1
        and scope.inner_fold == 2
    )
    gate = bank.get_gate_only_view(
        outer_fold=1,
        context_epoch=0,
        exact_spent_row_ids=cumulative.fit_row_ids,
        exact_gate_row_ids=next_gate.heldout_row_ids,
    )
    assert gate.context_oof_available is False
    assert gate.identity()["gate_fit_row_provenance"] == list(
        cumulative.fit_row_ids
    )
    values = gate.materialize()
    assert values.row_ids == next_gate.heldout_row_ids
    assert set(values.source_families) == set(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    with pytest.raises(RuntimeError, match="no spent-context inner-OOF"):
        gate.aligned_conditional_values()
    with pytest.raises(RuntimeError, match="were not produced"):
        gate.context_oof_values()


def test_tampered_source_byte_fails_authenticated_handle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution_root, plan, execution_manifest = _write_execution(tmp_path)
    _trust_test_execution(monkeypatch, execution_manifest)
    bank = reference_module.publish_role_neutral_direct_numerical_reference_bank(
        root=(tmp_path / "reference_bank").resolve(),
        execution_root=execution_root,
        plan=plan,
        execution_manifest=execution_manifest,
    )
    target = next(execution_root.rglob("*bow_nuisance.predictions.npy"))
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="source graph changed"):
        bank.verify_authenticated_content()


def test_reordered_registered_rows_fail_before_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution_root, plan, execution_manifest = _write_execution(
        tmp_path,
        reordered_bow_rows=True,
    )
    _trust_test_execution(monkeypatch, execution_manifest)
    with pytest.raises(ValueError, match="heldout row order changed"):
        reference_module.publish_role_neutral_direct_numerical_reference_bank(
            root=(tmp_path / "reference_bank").resolve(),
            execution_root=execution_root,
            plan=plan,
            execution_manifest=execution_manifest,
        )


def test_missing_native_family_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution_root, plan, execution_manifest = _write_execution(
        tmp_path,
        missing_family=EMBEDDING_CLUSTERED,
    )
    _trust_test_execution(monkeypatch, execution_manifest)
    with pytest.raises(ValueError, match="has no unique logical view"):
        reference_module.publish_role_neutral_direct_numerical_reference_bank(
            root=(tmp_path / "reference_bank").resolve(),
            execution_root=execution_root,
            plan=plan,
            execution_manifest=execution_manifest,
        )


def test_wrong_oof_assignments_and_noncanonical_gate_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution_root, plan, execution_manifest = _write_execution(tmp_path)
    _trust_test_execution(monkeypatch, execution_manifest)
    bank = reference_module.publish_role_neutral_direct_numerical_reference_bank(
        root=(tmp_path / "reference_bank").resolve(),
        execution_root=execution_root,
        plan=plan,
        execution_manifest=execution_manifest,
    )
    full = next(
        scope
        for scope in plan.scopes
        if scope.scope_kind == "full_outer" and scope.outer_fold == 1
    )
    owner = plan.physical_owner(full.scope_id)
    sealed = handoff_module.build_role_neutral_stage2_fit_projection_proof(
        plan_scientific_content_sha256=plan.scientific_content_sha256,
        physical_owner_scope_id=owner.scope_id,
        fit_row_ids=full.fit_row_ids,
        fit_texts=("unused",) * len(full.fit_row_ids),
        fit_treatment=np.zeros(len(full.fit_row_ids)),
        fit_outcome=np.zeros(len(full.fit_row_ids)),
    )
    _bind_test_bank(
        bank,
        monkeypatch,
        owner_proof_overrides={
            owner.scope_id: sealed["content_sha256"],
        },
    )
    expected = bank.get_meta_inner_fold_ids(
        outer_fold=1,
        exact_outer_train_row_ids=full.fit_row_ids,
    )
    with pytest.raises(ValueError, match="assignments differ"):
        bank.produce(
            outer_fold=1,
            outer_train_row_ids=full.fit_row_ids,
            outer_train_texts=("unused",) * len(full.fit_row_ids),
            outer_train_treatment=np.zeros(len(full.fit_row_ids)),
            outer_train_outcome=np.zeros(len(full.fit_row_ids)),
            outer_heldout_row_ids=full.heldout_row_ids,
            outer_heldout_texts=("unused",) * len(full.heldout_row_ids),
            meta_inner_fold_ids=tuple(reversed(expected)),
        )
    projection_proof_calls = 0

    def forbidden_projection_rehash(**_kwargs):
        nonlocal projection_proof_calls
        projection_proof_calls += 1
        raise AssertionError("ordinary fold production must not rehash text/T/Y")

    monkeypatch.setattr(
        handoff_module,
        "build_role_neutral_stage2_fit_projection_proof",
        forbidden_projection_rehash,
    )
    produced = bank.produce(
        outer_fold=1,
        outer_train_row_ids=full.fit_row_ids,
        outer_train_texts=("already-bound",) * len(full.fit_row_ids),
        outer_train_treatment=np.zeros(len(full.fit_row_ids)),
        outer_train_outcome=np.zeros(len(full.fit_row_ids)),
        outer_heldout_row_ids=full.heldout_row_ids,
        outer_heldout_texts=("already-bound",) * len(full.heldout_row_ids),
        meta_inner_fold_ids=expected,
    )
    assert produced.outer_fold == full.outer_fold
    assert projection_proof_calls == 0
    cumulative = next(
        scope
        for scope in plan.scopes
        if scope.scope_kind == "cumulative_spent" and scope.outer_fold == 1
    )
    with pytest.raises(ValueError, match="precommitted next partition"):
        bank.get_gate_only_view(
            outer_fold=1,
            context_epoch=0,
            exact_spent_row_ids=cumulative.fit_row_ids,
            exact_gate_row_ids=(cumulative.heldout_row_ids[0],),
        )


def test_materialized_owned_float64_buffer_is_frozen_without_second_copy() -> None:
    values = np.arange(4, dtype=np.float64).reshape(2, 2).copy(order="C")
    materialized = reference_module.MaterializedRoleNeutralNumericalMatrix(
        row_ids=(0, 1),
        coordinate_ids=("c0", "c1"),
        names=("n0", "n1"),
        source_families=(BOW_NUISANCE, BOW_NUISANCE),
        source_kinds=("k0", "k1"),
        consumer_roles=(
            PROPENSITY_NUISANCE_FEATURE_ROLE,
            OUTCOME_NUISANCE_FEATURE_ROLE,
        ),
        observable_axes=(("treatment",), ("outcome",)),
        bank_kinds=(
            reference_module.RAW_FEATURE_BANK,
            reference_module.RAW_FEATURE_BANK,
        ),
        values=values,
    )
    assert materialized.values is values
    assert materialized.values.flags.writeable is False
