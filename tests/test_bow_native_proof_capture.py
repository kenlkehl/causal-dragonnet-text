import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge

from oci.config import (
    AppliedInferenceConfig,
    BoWViewConfig,
    EmbeddingContrastDiscoveryConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
)
from oci.inference.bow_native_proof_capture import (
    NativeBoWProofCaptureSink,
    _ArrayStore,
    _capture_learner,
    _capture_vectorizer,
    _predict_learner,
    _replay_fit_transform,
    _restore_vectorizer,
    _sha256_json,
    validate_bow_native_capture,
)
from oci.inference.all_evidence_fusion import (
    FoldEvidenceInput,
    FoldEvidenceProvenance,
    LEGACY_ALL_SOURCE,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    build_role_neutral_evidence_catalog,
)
from oci.inference.multi_model_agentic_forest import _normalize_texts
from oci.inference.multi_model_agentic_forest import _make_bow_vectorizer
from oci.inference.multi_model_forest_stage1 import MultiModelForestStage1Runner
from oci.inference.production_stage1_bundle import (
    PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
    STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
    STAGE1_RAW_EVIDENCE_SIDECAR_SCHEMA,
    _catalog_ready_legacy_digest,
    _component_file_registration,
    _register_bow_native_family_proofs,
    _sha256_json as _bundle_sha256_json,
    _validate_bow_native_family_proof_index,
    _write_immutable_json,
)
from oci.inference.tfidf_topic_discovery import row_set_fingerprint


def _dataset(*, heldout_labels_flipped: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "clinical_text": [
                "alpha brain stable baseline",
                "beta liver frail baseline",
                "alpha lung active response",
                "beta bone quiet progression",
                "alpha brain active benefit",
                "beta liver quiet risk",
                "alpha lung stable benefit",
                "beta bone frail progression",
                "alpha brain active response",
                "beta liver stable risk",
                "heldout alpha narrative",
                "heldout beta narrative",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
            "outcome_indicator": [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1],
        }
    )
    if heldout_labels_flipped:
        frame.loc[10:, ["treatment_indicator", "outcome_indicator"]] = np.asarray([[1, 1], [0, 0]])
    return frame


def _config() -> AppliedInferenceConfig:
    return AppliedInferenceConfig(
        cv_folds=2,
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                feature_discovery_methods=["bow"],
                bow_views=[
                    BoWViewConfig(
                        name="linear_1_2",
                        max_features=1000,
                        min_df=1,
                        max_df=1.0,
                        ngram_range_min=1,
                        ngram_range_max=2,
                    )
                ],
                nuisance_folds=2,
                effect_folds=2,
                fold_parallelism="1",
                matched_pair_uplift_enabled=False,
                matched_pair_bow_enabled=False,
                matched_pair_htr_enabled=False,
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=False,
                    disable_reason="capture unit test",
                ),
            ),
        ),
    )


def _capture(tmp_path: Path, *, flipped: bool = False, name: str = "capture"):
    dataset = _dataset(heldout_labels_flipped=flipped)
    config = _config()
    fit_rows = tuple(range(10))
    heldout_rows = (10, 11)
    fit_texts = tuple(_normalize_texts(dataset.iloc[list(fit_rows)]["clinical_text"]))
    heldout_texts = tuple(_normalize_texts(dataset.iloc[list(heldout_rows)]["clinical_text"]))
    sink = NativeBoWProofCaptureSink(
        artifact_dir=tmp_path / name,
        scope_id="outer_001_inner_001",
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        fit_texts=fit_texts,
        heldout_texts=heldout_texts,
        text_column="clinical_text",
        outcome_type="binary",
        e_clip=config.architecture.multi_model_forest.e_clip,
        nuisance_folds=2,
        effect_folds=2,
        view_configs=[vars(view) for view in config.architecture.multi_model_forest.bow_views],
    )
    runner = MultiModelForestStage1Runner(
        dataset=dataset,
        config=config,
        output_path=tmp_path / f"runner_{name.replace('/', '_')}.parquet",
        num_workers=1,
        bow_native_capture_sink=sink,
    )
    train_df = runner.dataset.iloc[list(fit_rows)].reset_index(drop=True)
    heldout_df = runner.dataset.iloc[list(heldout_rows)][
        ["_oci_row_id", "clinical_text"]
    ].reset_index(drop=True)
    bundle = runner._build_feature_bundle(
        train_df=train_df,
        test_df=heldout_df,
        outer_fold=1,
    )
    metadata = sink.finalize()
    return metadata, bundle, fit_texts, heldout_texts


def test_native_bow_capture_replays_every_fold_and_full_fit(tmp_path: Path):
    metadata, _bundle, fit_texts, heldout_texts = _capture(tmp_path)
    validated = validate_bow_native_capture(
        tmp_path / "capture",
        expected_scope_id="outer_001_inner_001",
        expected_fit_row_ids=range(10),
        expected_heldout_row_ids=(10, 11),
        fit_texts=fit_texts,
        heldout_texts=heldout_texts,
        expected_fit_treatment=_dataset().iloc[:10]["treatment_indicator"],
        expected_fit_outcome=_dataset().iloc[:10]["outcome_indicator"],
    )
    assert validated == metadata
    assert {row["family"] for row in validated["folds"]} == {
        "bow_nuisance",
        "bow_r_loss",
    }
    assert {row["objective"] for row in validated["folds"]} == {
        "treatment_nuisance",
        "outcome_nuisance",
        "effect_pseudo_target",
        "effect_weighted_r",
    }
    drifted_treatment = _dataset().iloc[:10]["treatment_indicator"].to_numpy(dtype=float)
    drifted_treatment[0] = 1.0 - drifted_treatment[0]
    with pytest.raises(ValueError, match="treatment differs from canonical"):
        validate_bow_native_capture(
            tmp_path / "capture",
            expected_scope_id="outer_001_inner_001",
            expected_fit_row_ids=range(10),
            expected_heldout_row_ids=(10, 11),
            fit_texts=fit_texts,
            heldout_texts=heldout_texts,
            expected_fit_treatment=drifted_treatment,
            expected_fit_outcome=_dataset().iloc[:10]["outcome_indicator"],
        )
    drifted_outcome = _dataset().iloc[:10]["outcome_indicator"].to_numpy(dtype=float)
    drifted_outcome[0] = 1.0 - drifted_outcome[0]
    with pytest.raises(ValueError, match="outcome differs from canonical"):
        validate_bow_native_capture(
            tmp_path / "capture",
            expected_scope_id="outer_001_inner_001",
            expected_fit_row_ids=range(10),
            expected_heldout_row_ids=(10, 11),
            fit_texts=fit_texts,
            heldout_texts=heldout_texts,
            expected_fit_treatment=_dataset().iloc[:10]["treatment_indicator"],
            expected_fit_outcome=drifted_outcome,
        )
    assert not list((tmp_path / "capture").rglob("*.joblib"))
    assert not list((tmp_path / "capture").rglob("*.pkl"))


@pytest.mark.parametrize("mutation", ["missing_fold", "wrong_objective"])
def test_native_bow_capture_rejects_missing_fold_or_objective(
    tmp_path: Path,
    mutation: str,
):
    _metadata, _bundle, fit_texts, heldout_texts = _capture(tmp_path)
    path = tmp_path / "capture" / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing_fold":
        payload["folds"].pop()
    else:
        payload["folds"][-1]["objective"] = "wrong_objective"
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    payload["content_sha256"] = _sha256_json(body)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing|objective"):
        validate_bow_native_capture(
            tmp_path / "capture",
            fit_texts=fit_texts,
            heldout_texts=heldout_texts,
        )


def test_native_bow_capture_rejects_numerical_tamper(tmp_path: Path):
    _metadata, _bundle, fit_texts, heldout_texts = _capture(tmp_path)
    arrays_path = tmp_path / "capture" / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
    key = next(key for key in arrays if key.endswith("_coef"))
    arrays[key].reshape(-1)[0] += 0.5
    np.savez_compressed(arrays_path, **arrays)
    with pytest.raises(ValueError, match="closed envelope"):
        validate_bow_native_capture(
            tmp_path / "capture",
            fit_texts=fit_texts,
            heldout_texts=heldout_texts,
        )


def test_native_bow_capture_is_invariant_to_external_heldout_labels(tmp_path: Path):
    first, first_bundle, _fit_texts, _heldout_texts = _capture(
        tmp_path,
        name="first",
    )
    second, second_bundle, _fit_texts, _heldout_texts = _capture(
        tmp_path,
        flipped=True,
        name="second",
    )
    np.testing.assert_allclose(first_bundle.w_test, second_bundle.w_test)
    np.testing.assert_allclose(first_bundle.x_test, second_bundle.x_test)
    assert first["array_inventory"] == second["array_inventory"]
    assert first["folds"] == second["folds"]
    assert first["full_fit_models"] == second["full_fit_models"]
    assert first["heldout_labels_accessed"] is False
    assert second["heldout_labels_accessed"] is False


def test_paired_bow_proofs_bind_and_revalidate_actual_capture(tmp_path: Path):
    component = tmp_path / "component"
    component.mkdir()
    metadata, bundle, fit_texts, heldout_texts = _capture(
        component,
        name="native_bow_models/outer_001_inner_001",
    )
    fit_rows = tuple(range(10))
    heldout_rows = (10, 11)
    digest = _catalog_ready_legacy_digest(
        importance=bundle.handoff_evidence["importance"],
        embedding_evidence={},
        htr_evidence={},
    )
    provenance = FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        scope="inner_train",
        inner_fold=1,
        artifact_id="bow-native-proof-test",
    )
    catalog = build_role_neutral_evidence_catalog(
        (
            FoldEvidenceInput(
                LEGACY_ALL_SOURCE,
                {
                    "outer_fold": 1,
                    "inner_fold": 1,
                    "scope": "inner_train",
                    "n_rows": len(fit_rows),
                    "context": {"evidence_digest": digest},
                },
                provenance,
            ),
        ),
        require_all_source_kinds=False,
        require_all_architecture_families=False,
        require_upstream_completeness=False,
    )
    source_body = {
        "schema_version": STAGE1_RAW_EVIDENCE_SIDECAR_SCHEMA,
        "scope_id": "outer_001_inner_001",
        "outer_fold": 1,
        "inner_fold": 1,
        "fit_row_fingerprint": row_set_fingerprint(fit_rows),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_rows),
        "split_registry_content_sha256": "a" * 64,
        "prompt_grounding_allowed": False,
        "raw_drillback_requires_authenticated_id": True,
        "model_evidence": {},
        "matched_pair_subproducer_proofs": {},
    }
    source_path = component / "raw_evidence_sidecar.json"
    _write_immutable_json(
        source_path,
        {**source_body, "content_sha256": _bundle_sha256_json(source_body)},
    )
    configuration = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
        "scope_id": "outer_001_inner_001",
        "text_column": "clinical_text",
        "outcome_type": "binary",
        "e_clip": 0.01,
        "nuisance_folds": 2,
        "effect_folds": 2,
        "bow_views": metadata["view_configs"],
        "capture_schema_version": metadata["schema_version"],
        "heldout_label_policy": "id_and_text_only",
        "r_loss_nuisance_source": "ensemble_mean_nuisance",
        "split_registry_content_sha256": "a" * 64,
    }
    registration = _register_bow_native_family_proofs(
        component_root=component,
        proof_directory=Path("native_family_proofs/outer_001_inner_001"),
        scope_id="outer_001_inner_001",
        catalog=catalog,
        capture_artifact_path=(component / "native_bow_models" / "outer_001_inner_001"),
        source_artifact_path=source_path,
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        fit_texts=fit_texts,
        heldout_texts=heldout_texts,
        fit_treatment=_dataset().iloc[:10]["treatment_indicator"],
        fit_outcome=_dataset().iloc[:10]["outcome_indicator"],
        split_scope_fingerprint="b" * 64,
        data_projection_sha256="c" * 64,
        configuration=configuration,
    )
    assert registration["registered_families"] == list(
        PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS
    )
    assert [row["family"] for row in registration["family_proofs"]] == list(
        PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS
    )
    index_body = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
        "split_registry_content_sha256": "a" * 64,
        "registered_families": list(PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS),
        "exact_inner_scope_count": 1,
        "executable_checkpoint_files_retained": False,
        "scopes": [
            {
                "scope_id": "outer_001_inner_001",
                "outer_fold": 1,
                "inner_fold": 1,
                "registered_families": list(PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS),
                "content_sha256": registration["content_sha256"],
                "registration": registration["registration"],
            }
        ],
    }
    index_path = component / "bow_native_family_proof_index.json"
    _write_immutable_json(
        index_path,
        {**index_body, "content_sha256": _bundle_sha256_json(index_body)},
    )
    validated = _validate_bow_native_family_proof_index(
        component_root=component,
        index_registration=_component_file_registration(
            index_path,
            component_root=component,
        ),
        expected_inner_scopes={
            "outer_001_inner_001": {
                "outer_fold": 1,
                "inner_fold": 1,
                "fit_row_ids": list(fit_rows),
                "heldout_row_ids": list(heldout_rows),
            }
        },
        split_registry_content_sha256="a" * 64,
        modeling_data=_dataset(),
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
    )
    assert validated["exact_inner_scope_count"] == 1
    index_registration = _component_file_registration(
        index_path,
        component_root=component,
    )
    first_family = registration["family_proofs"][0]
    tamper_targets = [
        component / first_family["model_artifact"]["relative_path"] / "arrays.npz",
        component / first_family["source_artifact"]["relative_path"],
        component / first_family["native_fit_metadata"]["relative_path"],
        component / first_family["native_execution_record"]["relative_path"],
        component / registration["registration"]["relative_path"],
        index_path,
    ]
    for target in tamper_targets:
        original = target.read_bytes()
        target.write_bytes(original + b"tamper")
        with pytest.raises((RuntimeError, ValueError)):
            _validate_bow_native_family_proof_index(
                component_root=component,
                index_registration=index_registration,
                expected_inner_scopes={
                    "outer_001_inner_001": {
                        "outer_fold": 1,
                        "inner_fold": 1,
                        "fit_row_ids": list(fit_rows),
                        "heldout_row_ids": list(heldout_rows),
                    }
                },
                split_registry_content_sha256="a" * 64,
                modeling_data=_dataset(),
                text_column="clinical_text",
                treatment_column="treatment_indicator",
                outcome_column="outcome_indicator",
            )
        target.write_bytes(original)


@pytest.mark.parametrize(
    ("model", "classification"),
    [
        (ExtraTreesClassifier(n_estimators=4, random_state=7), True),
        (RandomForestRegressor(n_estimators=4, random_state=9), False),
    ],
)
def test_safe_tree_state_replays_without_pickle(model, classification: bool):
    texts = [
        "alpha brain stable",
        "beta liver frail",
        "alpha lung response",
        "beta bone progression",
        "alpha brain benefit",
        "beta liver risk",
    ]
    params = {
        "ngram_range_min": 1,
        "ngram_range_max": 2,
        "min_df": 1,
        "max_df": 1.0,
        "sublinear_tf": True,
        "max_features": 100,
    }
    vectorizer = _make_bow_vectorizer(params)
    x = vectorizer.fit_transform(texts)
    target = (
        np.asarray([1, 0, 1, 0, 1, 0], dtype=int)
        if classification
        else np.asarray([0.8, -0.2, 0.5, -0.7, 0.9, -0.4], dtype=float)
    )
    model.fit(x, target)
    expected = model.predict_proba(x)[:, 1] if classification else model.predict(x)
    store = _ArrayStore()
    vectorizer_state = _capture_vectorizer(
        vectorizer,
        store,
        "vectorizer",
        vectorizer_params=params,
    )
    learner_state = _capture_learner(
        model,
        store,
        "learner",
        classification=classification,
    )
    restored = _restore_vectorizer(vectorizer_state, store.arrays)
    observed = _predict_learner(
        learner_state,
        store.arrays,
        restored.transform(texts).tocsr(),
    )
    np.testing.assert_allclose(observed, expected, rtol=2e-7, atol=2e-8)


def test_safe_ridge_replay_preserves_native_float32_arithmetic():
    texts = [
        f"common group{index % 7} marker{index % 11} detail{index % 5} row{index}"
        for index in range(75)
    ]
    params = {
        "ngram_range_min": 1,
        "ngram_range_max": 2,
        "min_df": 1,
        "max_df": 1.0,
        "sublinear_tf": True,
        "max_features": 1000,
    }
    vectorizer = _make_bow_vectorizer(params)
    matrix = vectorizer.fit_transform(texts)
    target = np.random.default_rng(917).normal(size=len(texts))
    model = Ridge(alpha=0.5).fit(matrix, target)
    validation_matrix = vectorizer.transform(texts).tocsr()
    expected = model.predict(validation_matrix)
    assert matrix.dtype == np.float32
    assert model.coef_.dtype == np.float32

    store = _ArrayStore()
    vectorizer_state = _capture_vectorizer(
        vectorizer,
        store,
        "vectorizer",
        vectorizer_params=params,
    )
    learner_state = _capture_learner(
        model,
        store,
        "learner",
        classification=False,
    )
    # Capture deliberately stores numerical learner state losslessly in
    # float64; replay must nevertheless reproduce sklearn's float32 dot.
    stored_coef = store.arrays[str(learner_state["coef"])]
    assert stored_coef.dtype == np.float64
    restored_matrix = _restore_vectorizer(vectorizer_state, store.arrays).transform(texts)
    observed = _predict_learner(learner_state, store.arrays, restored_matrix.tocsr())
    np.testing.assert_array_equal(observed, expected.astype(np.float64))

    replayed_fit_matrix = _replay_fit_transform(
        vectorizer_state,
        store.arrays,
        texts,
    )
    fit_observed = _predict_learner(learner_state, store.arrays, replayed_fit_matrix)
    np.testing.assert_array_equal(fit_observed, model.predict(matrix).astype(np.float64))

    float64_replay = (
        np.asarray(restored_matrix @ stored_coef.reshape(-1), dtype=np.float64).reshape(-1)
        + float(store.arrays[str(learner_state["intercept"])].reshape(-1)[0])
    )
    assert float(np.max(np.abs(float64_replay - observed))) > 2e-8
