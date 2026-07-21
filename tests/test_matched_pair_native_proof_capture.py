from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from oci.config import (
    AgenticAttentionVariableForestConfig,
    AppliedInferenceConfig,
    BoWViewConfig,
    EmbeddingContrastDiscoveryConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
    TrainingConfig,
)
from oci.inference.matched_pair_native_proof_capture import (
    NativeMatchedPairProofCaptureSink,
    _sha256_json,
    validate_matched_pair_native_capture,
)
from oci.inference.htr_native_proof_capture import directory_tree_sha256
from oci.inference.all_evidence_discovery_interfaces import MATCHED_PAIR_UPLIFT
from oci.inference.all_evidence_fusion import (
    LEGACY_ALL_SOURCE,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    build_role_neutral_evidence_catalog,
)
from oci.inference.multi_model_forest_stage1 import MultiModelForestStage1Runner
from oci.inference.multi_model_pair_uplift import (
    OffsetLogitBoWPairModel,
    fit_bow_pair_uplift_train_test,
    fit_htr_pair_uplift_train_test,
)
from oci.inference.production_stage1_bundle import (
    PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    STAGE1_MATCHED_PAIR_NATIVE_FIT_METADATA_SCHEMA,
    STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
    STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
    STAGE1_RAW_EVIDENCE_SIDECAR_SCHEMA,
    _catalog_ready_legacy_digest,
    _component_file_registration,
    _matched_pair_subproducer_proofs,
    _register_matched_pair_native_family_proof,
    _sha256_json as _bundle_sha256_json,
    _validate_matched_pair_native_family_proof_index,
    _write_immutable_json,
)
from oci.inference.tfidf_topic_discovery import row_set_fingerprint


def _dataset(*, heldout_labels_flipped: bool = False) -> pd.DataFrame:
    treatment = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    outcome = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1]
    frame = pd.DataFrame(
        {
            "clinical_text": [
                f"patient {index} "
                f"{'smoker brain' if index % 2 else 'never liver'} "
                f"{'response stable' if index % 3 else 'progression frail'}"
                for index in range(14)
            ],
            "treatment_indicator": treatment,
            "outcome_indicator": outcome,
        }
    )
    if heldout_labels_flipped:
        frame.loc[12:, ["treatment_indicator", "outcome_indicator"]] = np.asarray([[1, 1], [0, 0]])
    return frame


def _config(
    tmp_path: Path,
    *,
    htr_sentence_model: str = "hash",
    htr_effect_folds: int = 2,
) -> AppliedInferenceConfig:
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "unused.parquet"),
        cv_folds=2,
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            feature_extractor_type="hierarchical_transformer",
            htr_sentence_model=htr_sentence_model,
            htr_freeze_sentence_encoder=False,
            htr_chunk_size_words=5,
            htr_chunk_overlap_words=1,
            htr_max_chunks=4,
            htr_num_layers=1,
            htr_num_heads=2,
            htr_transformer_dim=16,
            htr_projection_dim=8,
            htr_hash_embedding_dim=16,
            htr_dropout=0.0,
            causal_head_hidden_outcome_dim=8,
            multi_model_forest=MultiModelForestConfig(
                feature_discovery_methods=["bow", "htr"],
                bow_views=[
                    BoWViewConfig(
                        name="linear_1_2",
                        max_features=1000,
                        min_df=1,
                        max_df=1.0,
                        ngram_range_min=1,
                        ngram_range_max=2,
                        bow_model="linear",
                    )
                ],
                nuisance_folds=2,
                effect_folds=2,
                fold_parallelism="1",
                matched_pair_uplift_enabled=True,
                matched_pair_bow_enabled=True,
                matched_pair_htr_enabled=True,
                matched_pair_propensity_caliper=1.0,
                matched_pair_outcome_caliper=1.0,
                matched_pair_max_controls_per_candidate=2,
                matched_pair_nearest_fallback_controls=1,
                matched_pair_htr_attention_pairs_per_fold=0,
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=False,
                    disable_reason="native matched-pair proof unit test",
                ),
            ),
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=htr_effect_folds,
                nuisance_epochs=1,
                effect_epochs=1,
                nuisance_calibration="none",
                fold_parallelism="1",
                attention_top_k_chunks=1,
            ),
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=4,
            effect_batch_size=4,
            learning_rate=1e-3,
            gradient_clip_norm=1.0,
        ),
    )
    config.seed = 23
    return config


def _run_capture(
    tmp_path: Path,
    *,
    name: str = "capture",
    heldout_labels_flipped: bool = False,
    htr_effect_folds: int = 2,
):
    torch.manual_seed(4321)
    dataset = _dataset(heldout_labels_flipped=heldout_labels_flipped)
    config = _config(tmp_path, htr_effect_folds=htr_effect_folds)
    view_configs = [vars(view) for view in config.architecture.multi_model_forest.bow_views]
    sink = NativeMatchedPairProofCaptureSink(
        artifact_dir=tmp_path / name,
        scope_id="outer_001_inner_001",
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=range(12),
        heldout_row_ids=(12, 13),
        fit_texts=dataset.iloc[:12]["clinical_text"].tolist(),
        heldout_texts=dataset.iloc[12:]["clinical_text"].tolist(),
        text_column="clinical_text",
        effect_folds=2,
        view_configs=view_configs,
        propensity_caliper=1.0,
        outcome_caliper=1.0,
        max_controls_per_candidate=2,
        nearest_fallback_controls=1,
        htr_model_tree_sha256=None,
        htr_prediction_batch_size=4,
        seed=23,
    )
    runner = MultiModelForestStage1Runner(
        dataset=dataset,
        config=config,
        output_path=tmp_path / f"runner_{name}.parquet",
        device="cpu",
        num_workers=0,
        matched_pair_native_capture_sink=sink,
    )
    train_df = runner.dataset.iloc[:12].reset_index(drop=True)
    heldout_df = runner.dataset.iloc[12:][["_oci_row_id", "clinical_text"]].reset_index(drop=True)
    provider = runner._htr_provider()
    native_runner = provider._ensure_runner(train_df)
    native_runner._attention_evidence = lambda *args, **kwargs: []
    bundle = runner._build_feature_bundle(
        train_df=train_df,
        test_df=heldout_df,
        outer_fold=1,
    )
    metadata = sink.finalize()
    return metadata, bundle, train_df, heldout_df


def _validate(path: Path, train_df: pd.DataFrame, heldout_df: pd.DataFrame):
    return validate_matched_pair_native_capture(
        path,
        expected_scope_id="outer_001_inner_001",
        expected_fit_row_ids=train_df["_oci_row_id"].tolist(),
        expected_heldout_row_ids=heldout_df["_oci_row_id"].tolist(),
        fit_texts=train_df["clinical_text"].tolist(),
        heldout_texts=heldout_df["clinical_text"].tolist(),
        expected_fit_treatment=train_df["treatment_indicator"].tolist(),
        expected_fit_outcome=train_df["outcome_indicator"].tolist(),
        device="cpu",
    )


def _tiny_local_bert(path: Path) -> Path:
    from transformers import BertConfig, BertModel, BertTokenizer

    path.mkdir(parents=True, exist_ok=False)
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "patient",
        "smoker",
        "brain",
        "never",
        "liver",
        "response",
        "stable",
        "progression",
        "frail",
        *[str(index) for index in range(20)],
    ]
    vocab_path = path / "vocab.txt"
    vocab_path.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=True)
    tokenizer.save_pretrained(path)
    torch.manual_seed(991)
    model = BertModel(
        BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=128,
        )
    )
    model.save_pretrained(path, safe_serialization=True)
    return path


def test_native_matched_pair_capture_replays_both_subproducers(tmp_path: Path):
    metadata, _bundle, train_df, heldout_df = _run_capture(tmp_path)
    validated = _validate(tmp_path / "capture", train_df, heldout_df)
    treatment_drift = train_df.copy()
    treatment_drift.loc[0, "treatment_indicator"] = (
        1.0 - float(treatment_drift.loc[0, "treatment_indicator"])
    )
    with pytest.raises(ValueError, match="treatment differs from canonical"):
        _validate(tmp_path / "capture", treatment_drift, heldout_df)
    outcome_drift = train_df.copy()
    outcome_drift.loc[0, "outcome_indicator"] = (
        1.0 - float(outcome_drift.loc[0, "outcome_indicator"])
    )
    with pytest.raises(ValueError, match="outcome differs from canonical"):
        _validate(tmp_path / "capture", outcome_drift, heldout_df)

    assert validated == metadata
    assert validated["subproducer_coverage"] == ["bow", "htr"]
    assert len(validated["bow_fold_states"]) == 2
    assert len(validated["htr_fold_states"]) == 2
    assert validated["heldout_labels_accessed"] is False
    assert {path.suffix for path in (tmp_path / "capture").iterdir()} == {
        ".json",
        ".npz",
    }


def test_native_matched_pair_capture_uses_pair_fold_contract_when_htr_folds_differ(
    tmp_path: Path,
):
    metadata, _bundle, train_df, heldout_df = _run_capture(
        tmp_path,
        htr_effect_folds=3,
    )
    validated = _validate(tmp_path / "capture", train_df, heldout_df)

    assert validated == metadata
    assert validated["effect_folds"] == 2
    assert [row["fold"] for row in validated["bow_fold_states"]] == [1, 2]
    assert [row["fold"] for row in validated["htr_fold_states"]] == [1, 2]


def test_native_matched_pair_capture_replays_local_transformer_pair_state(
    tmp_path: Path,
):
    model_path = _tiny_local_bert(tmp_path / "tiny_bert")
    model_tree_sha256 = directory_tree_sha256(model_path)
    dataset = _dataset()
    config = _config(tmp_path, htr_sentence_model=str(model_path))
    runner = MultiModelForestStage1Runner(
        dataset=dataset,
        config=config,
        output_path=tmp_path / "local_runner.parquet",
        device="cpu",
        num_workers=0,
    )
    train_df = runner.dataset.iloc[:12].reset_index(drop=True)
    heldout_df = runner.dataset.iloc[12:][["_oci_row_id", "clinical_text"]].reset_index(drop=True)
    fit_texts = train_df["clinical_text"].astype(str).tolist()
    heldout_texts = heldout_df["clinical_text"].astype(str).tolist()
    treatment = train_df["treatment_indicator"].to_numpy(dtype=float)
    outcome = train_df["outcome_indicator"].to_numpy(dtype=float)
    e_fit = np.linspace(0.35, 0.65, len(train_df))
    m_fit = np.linspace(0.30, 0.70, len(train_df))
    e_heldout = np.asarray([0.45, 0.55])
    m_heldout = np.asarray([0.40, 0.60])
    view_config = vars(config.architecture.multi_model_forest.bow_views[0])
    sink = NativeMatchedPairProofCaptureSink(
        artifact_dir=tmp_path / "local_capture",
        scope_id="outer_001_inner_001",
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=train_df["_oci_row_id"].tolist(),
        heldout_row_ids=heldout_df["_oci_row_id"].tolist(),
        fit_texts=fit_texts,
        heldout_texts=heldout_texts,
        text_column="clinical_text",
        effect_folds=2,
        view_configs=(view_config,),
        propensity_caliper=1.0,
        outcome_caliper=1.0,
        max_controls_per_candidate=2,
        nearest_fallback_controls=1,
        htr_model_tree_sha256=model_tree_sha256,
        htr_prediction_batch_size=4,
        seed=23,
    )
    sink.record_scope_inputs(
        treatment=treatment,
        outcome=outcome,
        e_fit=e_fit,
        m_fit=m_fit,
        e_heldout=e_heldout,
        m_heldout=m_heldout,
    )
    vectorizer_params = {
        key: view_config[key]
        for key in (
            "ngram_range_min",
            "ngram_range_max",
            "min_df",
            "max_df",
            "sublinear_tf",
            "max_features",
        )
    }
    bow_result = fit_bow_pair_uplift_train_test(
        train_df=train_df,
        test_df=heldout_df,
        texts_train=fit_texts,
        texts_test=heldout_texts,
        y_train=outcome,
        t_train=treatment,
        e_train=e_fit,
        m_train=m_fit,
        e_test=e_heldout,
        m_test=m_heldout,
        vectorizer_params=vectorizer_params,
        model_params={
            "bow_model": "linear",
            "logistic_c": 1.0,
            "logistic_max_iter": 1000,
            "ridge_alpha": 10.0,
        },
        outer_fold=1,
        view_name="linear_1_2",
        view_index=0,
        effect_folds=2,
        propensity_caliper=1.0,
        outcome_caliper=1.0,
        max_controls_per_candidate=2,
        nearest_fallback_controls=1,
        l2_alpha=1.0,
        max_iter=100,
        top_n=5,
        native_capture_sink=sink,
    )
    provider = runner._htr_provider()
    htr_runner = provider._ensure_runner(train_df)
    htr_result = fit_htr_pair_uplift_train_test(
        runner=htr_runner,
        train_df=train_df,
        test_df=heldout_df,
        texts_train=fit_texts,
        texts_test=heldout_texts,
        y_train=outcome,
        t_train=treatment,
        e_train=e_fit,
        m_train=m_fit,
        e_test=e_heldout,
        m_test=m_heldout,
        outer_fold=1,
        effect_folds=2,
        propensity_caliper=1.0,
        outcome_caliper=1.0,
        max_controls_per_candidate=2,
        nearest_fallback_controls=1,
        max_attention_pairs=0,
        native_capture_sink=sink,
    )
    for prefix, result in (("bow_view_0000", bow_result), ("htr", htr_result)):
        for value_name, fit_value, heldout_value, role in (
            (
                "delta",
                result.train_delta_logit,
                result.test_delta_logit,
                "uplift_delta_logit",
            ),
            (
                "probability",
                result.train_pred_prob,
                result.test_pred_prob,
                "treated_outcome_probability",
            ),
            (
                "n_controls",
                result.train_n_controls,
                result.test_n_controls,
                "matched_control_count",
            ),
        ):
            sink.record_scope_output(
                f"{prefix}_{value_name}_fit",
                fit_value,
                role=f"fit_{role}",
            )
            sink.record_scope_output(
                f"{prefix}_{value_name}_heldout",
                heldout_value,
                role=f"heldout_{role}",
            )
    metadata = sink.finalize()
    validated = validate_matched_pair_native_capture(
        tmp_path / "local_capture",
        expected_scope_id="outer_001_inner_001",
        expected_fit_row_ids=train_df["_oci_row_id"].tolist(),
        expected_heldout_row_ids=heldout_df["_oci_row_id"].tolist(),
        fit_texts=fit_texts,
        heldout_texts=heldout_texts,
        htr_model_path=model_path,
        expected_htr_model_tree_sha256=model_tree_sha256,
        device="cpu",
    )
    assert validated == metadata
    assert validated["htr_extractor_identity"]["hash_backend"] is False
    assert validated["htr_model_tree_sha256"] == model_tree_sha256


def test_native_matched_pair_capture_rejects_output_only_model_fallbacks(
    tmp_path: Path,
):
    sink = NativeMatchedPairProofCaptureSink(
        artifact_dir=tmp_path / "fallback",
        scope_id="outer_001_inner_001",
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=(0, 1, 2, 3),
        heldout_row_ids=(4,),
        fit_texts=("a", "b", "c", "d"),
        heldout_texts=("e",),
        text_column="clinical_text",
        effect_folds=2,
        view_configs=(
            {
                "name": "linear_1_2",
                "max_features": 100,
                "min_df": 1,
                "max_df": 1.0,
                "ngram_range_min": 1,
                "ngram_range_max": 2,
                "sublinear_tf": True,
            },
        ),
        propensity_caliper=1.0,
        outcome_caliper=1.0,
        max_controls_per_candidate=1,
        nearest_fallback_controls=1,
        htr_model_tree_sha256=None,
        htr_prediction_batch_size=2,
        seed=1,
    )
    empty = pd.DataFrame()
    constant = OffsetLogitBoWPairModel(
        vectorizer_params={
            "max_features": 100,
            "min_df": 1,
            "max_df": 1.0,
            "ngram_range_min": 1,
            "ngram_range_max": 2,
            "sublinear_tf": True,
        },
        l2_alpha=1.0,
        max_iter=10,
        random_state=1,
    ).fit(empty)
    common = {
        "fold": 1,
        "fit_pos": (0, 1),
        "validation_pos": (2, 3),
        "fit_pairs": empty,
        "validation_pairs": empty,
        "heldout_pairs": empty,
        "validation_pair_delta": np.zeros(0),
        "validation_delta": np.full(2, np.nan),
        "validation_probability": np.full(2, np.nan),
        "validation_n_controls": np.zeros(2),
        "heldout_pair_delta": np.zeros(0),
        "heldout_delta": np.full(1, np.nan),
        "heldout_probability": np.full(1, np.nan),
        "heldout_n_controls": np.zeros(1),
    }
    with pytest.raises(RuntimeError, match="genuinely fitted offset-logit"):
        sink.record_bow_pair_fold(
            view_name="linear_1_2",
            view_index=0,
            model=constant,
            **common,
        )
    with pytest.raises(RuntimeError, match="genuinely fitted HTR pair network"):
        sink.record_htr_pair_fold(model=None, **common)


@pytest.mark.parametrize(
    "mutation",
    ["missing_bow_fold", "missing_htr_fold", "wrong_objective", "wrong_split_seed"],
)
def test_native_matched_pair_capture_rejects_missing_or_changed_fold_proof(
    tmp_path: Path,
    mutation: str,
):
    _metadata, _bundle, train_df, heldout_df = _run_capture(tmp_path)
    path = tmp_path / "capture" / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing_bow_fold":
        payload["bow_fold_states"].pop()
    elif mutation == "missing_htr_fold":
        payload["htr_fold_states"].pop()
    elif mutation == "wrong_objective":
        payload["bow_fold_states"][0]["objective"] = "output_only_pair_claim"
    else:
        payload["htr_fold_states"][0]["split_seed"] += 1
    body = {name: value for name, value in payload.items() if name != "content_sha256"}
    payload["content_sha256"] = _sha256_json(body)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fold coverage|row/objective identity"):
        _validate(tmp_path / "capture", train_df, heldout_df)


def test_native_matched_pair_capture_rejects_numerical_state_tamper(tmp_path: Path):
    _metadata, _bundle, train_df, heldout_df = _run_capture(tmp_path)
    arrays_path = tmp_path / "capture" / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
    state_key = next(key for key in arrays if key.startswith("htr_") and "_state_" in key)
    state = arrays[state_key]
    if np.issubdtype(state.dtype, np.floating):
        state.reshape(-1)[0] += 0.25
    else:
        state.reshape(-1)[0] ^= 1
    np.savez_compressed(arrays_path, **arrays)

    with pytest.raises(ValueError, match="invalid envelope|changed"):
        _validate(tmp_path / "capture", train_df, heldout_df)


def test_native_matched_pair_capture_is_invariant_to_external_heldout_labels(
    tmp_path: Path,
):
    first, first_bundle, _train_df, _heldout_df = _run_capture(
        tmp_path,
        name="first",
    )
    second, second_bundle, _train_df, _heldout_df = _run_capture(
        tmp_path,
        name="second",
        heldout_labels_flipped=True,
    )

    np.testing.assert_allclose(first_bundle.w_test, second_bundle.w_test)
    np.testing.assert_allclose(first_bundle.x_test, second_bundle.x_test, equal_nan=True)
    assert first["array_inventory"] == second["array_inventory"]
    assert first["bow_fold_states"] == second["bow_fold_states"]
    assert first["bow_full_fit_states"] == second["bow_full_fit_states"]
    assert first["htr_fold_states"] == second["htr_fold_states"]
    assert first["scope_outputs"] == second["scope_outputs"]
    assert first["heldout_labels_accessed"] is False
    assert second["heldout_labels_accessed"] is False


def test_matched_pair_native_registration_revalidates_both_actual_subproducers(
    tmp_path: Path,
):
    component = tmp_path / "component"
    component.mkdir()
    metadata, bundle, train_df, heldout_df = _run_capture(
        component,
        name="native_matched_pair_models/outer_001_inner_001",
    )
    fit_rows = tuple(train_df["_oci_row_id"].astype(int))
    heldout_rows = tuple(heldout_df["_oci_row_id"].astype(int))
    matched_proofs = _matched_pair_subproducer_proofs(
        bundle=bundle,
        expected_bow_views=("linear_1_2",),
        scope_id="outer_001_inner_001",
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
    )
    digest = _catalog_ready_legacy_digest(
        importance=bundle.handoff_evidence["importance"],
        embedding_evidence={},
        htr_evidence=bundle.handoff_evidence["htr_evidence"],
    )
    provenance = FoldEvidenceProvenance(
        outer_fold=1,
        inner_fold=1,
        train_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        scope="inner_train",
        artifact_id="matched-pair-native-proof-test",
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
    assert catalog.family_atoms(MATCHED_PAIR_UPLIFT)
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
        "matched_pair_subproducer_proofs": matched_proofs,
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
        "effect_folds": 2,
        "bow_views": metadata["view_configs"],
        "matching_configuration": metadata["matching_configuration"],
        "required_subproducers": ["bow", "htr"],
        "capture_schema_version": metadata["schema_version"],
        "htr_model_tree_sha256": None,
        "heldout_label_policy": "id_and_text_only",
        "split_registry_content_sha256": "a" * 64,
    }
    registration = _register_matched_pair_native_family_proof(
        component_root=component,
        proof_directory=Path("native_matched_pair_family_proofs/outer_001_inner_001"),
        scope_id="outer_001_inner_001",
        catalog=catalog,
        capture_artifact_path=(component / "native_matched_pair_models/outer_001_inner_001"),
        source_artifact_path=source_path,
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        fit_texts=train_df["clinical_text"].tolist(),
        heldout_texts=heldout_df["clinical_text"].tolist(),
        fit_treatment=train_df["treatment_indicator"].tolist(),
        fit_outcome=train_df["outcome_indicator"].tolist(),
        split_scope_fingerprint="b" * 64,
        data_projection_sha256="c" * 64,
        configuration=configuration,
        htr_model_path=None,
        htr_model_sha256=None,
        device="cpu",
    )
    assert registration["registered_families"] == list(
        PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS
    )
    family_row = registration["family_proofs"][0]
    fit_metadata = json.loads(
        (component / family_row["native_fit_metadata"]["relative_path"]).read_text(encoding="utf-8")
    )
    assert fit_metadata["schema_version"] == STAGE1_MATCHED_PAIR_NATIVE_FIT_METADATA_SCHEMA
    assert fit_metadata["required_subproducers"] == ["bow", "htr"]
    index_body = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
        "split_registry_content_sha256": "a" * 64,
        "registered_families": list(PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS),
        "exact_inner_scope_count": 1,
        "executable_checkpoint_files_retained": False,
        "scopes": [
            {
                "scope_id": "outer_001_inner_001",
                "outer_fold": 1,
                "inner_fold": 1,
                "registered_families": list(
                    PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS
                ),
                "content_sha256": registration["content_sha256"],
                "registration": registration["registration"],
            }
        ],
    }
    index_path = component / "matched_pair_native_family_proof_index.json"
    _write_immutable_json(
        index_path,
        {**index_body, "content_sha256": _bundle_sha256_json(index_body)},
    )
    modeling_data = _dataset()
    validated = _validate_matched_pair_native_family_proof_index(
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
        modeling_data=modeling_data,
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        htr_model_path=None,
        htr_model_sha256=None,
        device="cpu",
    )
    assert validated["exact_inner_scope_count"] == 1

    fit_metadata_path = component / family_row["native_fit_metadata"]["relative_path"]
    tampered = json.loads(fit_metadata_path.read_text(encoding="utf-8"))
    tampered["required_subproducers"] = ["bow"]
    fit_metadata_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError), match="changed|invalid"):
        _validate_matched_pair_native_family_proof_index(
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
            modeling_data=modeling_data,
            text_column="clinical_text",
            treatment_column="treatment_indicator",
            outcome_column="outcome_indicator",
            htr_model_path=None,
            htr_model_sha256=None,
            device="cpu",
        )
