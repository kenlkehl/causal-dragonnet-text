from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from oci.config import (
    AgenticAttentionVariableForestConfig,
    AppliedInferenceConfig,
    EmbeddingContrastDiscoveryConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
    TrainingConfig,
)
from oci.inference.htr_native_proof_capture import (
    NativeHTRProofCaptureSink,
    directory_tree_sha256,
    validate_htr_native_capture,
)
from oci.inference.all_evidence_discovery_interfaces import HTR_NEURAL
from oci.inference.all_evidence_fusion import (
    LEGACY_ALL_SOURCE,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    build_role_neutral_evidence_catalog,
)
from oci.inference.multi_model_agentic_forest import _normalize_texts
from oci.inference.multi_model_forest_stage1 import MultiModelForestStage1Runner
from oci.inference.production_stage1_bundle import (
    PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
    STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
    STAGE1_RAW_EVIDENCE_SIDECAR_SCHEMA,
    _component_file_registration,
    _register_htr_native_family_proof,
    _sha256_json,
    _validate_htr_native_family_proof_index,
    _write_immutable_json,
)
from oci.inference.tfidf_topic_discovery import row_set_fingerprint


def _dataset(*, heldout_outcomes: tuple[int, int] = (0, 1)) -> pd.DataFrame:
    texts = [
        f"Ｐatient {index} ≥50% — "
        f"{'Older Smoker' if index % 2 else 'Younger Never Smoker'} "
        f"{'brain disease' if index % 3 else 'liver disease'}"
        for index in range(12)
    ]
    return pd.DataFrame(
        {
            "clinical_text": texts,
            "treatment_indicator": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "outcome_indicator": [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, *heldout_outcomes],
        }
    )


def _config(
    tmp_path: Path,
    *,
    htr_sentence_model: str = "hash",
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
                feature_discovery_methods=["htr"],
                nuisance_folds=2,
                effect_folds=2,
                fold_parallelism="1",
                matched_pair_uplift_enabled=False,
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=False,
                    disable_reason="native HTR proof unit test",
                ),
            ),
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
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
    config.seed = 17
    return config


def _run_capture(
    tmp_path: Path,
    *,
    heldout_outcomes: tuple[int, int] = (0, 1),
    artifact_name: str = "capture",
    htr_model_path: Path | None = None,
):
    torch.manual_seed(1234)
    data = _dataset(heldout_outcomes=heldout_outcomes)
    config = _config(
        tmp_path,
        htr_sentence_model=("hash" if htr_model_path is None else str(htr_model_path)),
    )
    runner = MultiModelForestStage1Runner(
        dataset=data,
        config=config,
        output_path=tmp_path / f"{artifact_name}.parquet",
        device=torch.device("cpu"),
        num_workers=1,
    )
    train_df = runner.dataset.iloc[:10].reset_index(drop=True)
    heldout_df = runner.dataset.iloc[10:][["_oci_row_id", "clinical_text"]].reset_index(
        drop=True
    )
    sink = NativeHTRProofCaptureSink(
        artifact_dir=tmp_path / artifact_name,
        scope_id="outer_001_inner_001",
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=train_df["_oci_row_id"].tolist(),
        heldout_row_ids=heldout_df["_oci_row_id"].tolist(),
        fit_texts=train_df["clinical_text"].tolist(),
        heldout_texts=heldout_df["clinical_text"].tolist(),
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        e_clip=float(config.architecture.multi_model_forest.e_clip),
        nuisance_folds=2,
        effect_folds=2,
        model_tree_sha256=(
            None if htr_model_path is None else directory_tree_sha256(htr_model_path)
        ),
        prediction_batch_size=4,
        seed=17,
    )
    runner.htr_native_capture_sink = sink
    provider = runner._htr_provider()
    native_runner = provider._ensure_runner(train_df)
    native_runner._attention_evidence = lambda *args, **kwargs: []
    runner._build_feature_bundle(train_df=train_df, test_df=heldout_df, outer_fold=1)
    metadata = sink.finalize()
    return metadata, train_df, heldout_df


def _validate(path: Path, train_df: pd.DataFrame, heldout_df: pd.DataFrame):
    return validate_htr_native_capture(
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
        "older",
        "younger",
        "smoker",
        "never",
        "brain",
        "liver",
        "disease",
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


def _htr_catalog(fit_row_ids, heldout_row_ids):
    provenance = FoldEvidenceProvenance(
        outer_fold=1,
        inner_fold=1,
        train_row_ids=tuple(map(int, fit_row_ids)),
        heldout_row_ids=tuple(map(int, heldout_row_ids)),
        scope="inner_train",
        artifact_id="htr-native-proof-test",
    )
    payload = {
        "outer_fold": 1,
        "inner_fold": 1,
        "scope": "inner_train",
        "n_rows": len(fit_row_ids),
        "context": {
            "evidence_digest": {
                "confounders": {
                    "bow_blurbs": [],
                    "embedding_chunks": [],
                    "htr_blurbs": [
                        {
                            "stage": "nuisance",
                            "meaning": "Nested HTR nuisance attention.",
                            "metrics": {},
                            "rows": [
                                {
                                    "phrase": "older smoker",
                                    "attention_score": 0.8,
                                }
                            ],
                        }
                    ],
                },
                "effect_modifiers": {
                    "bow_blurbs": [],
                    "embedding_chunks": [],
                    "htr_blurbs": [
                        {
                            "stage": "effect",
                            "meaning": "Nested HTR effect attention.",
                            "metrics": {},
                            "rows": [
                                {
                                    "phrase": "brain disease",
                                    "attention_score": 0.7,
                                }
                            ],
                        }
                    ],
                },
            }
        },
    }
    return build_role_neutral_evidence_catalog(
        (FoldEvidenceInput(LEGACY_ALL_SOURCE, payload, provenance),),
        require_all_source_kinds=False,
        require_all_architecture_families=False,
        require_upstream_completeness=False,
    )


def test_real_htr_runner_capture_replays_every_nested_fold(tmp_path: Path):
    metadata, train_df, heldout_df = _run_capture(tmp_path)
    validated = _validate(tmp_path / "capture", train_df, heldout_df)
    normalized_train = train_df.copy()
    normalized_heldout = heldout_df.copy()
    normalized_train["clinical_text"] = _normalize_texts(
        normalized_train["clinical_text"]
    )
    normalized_heldout["clinical_text"] = _normalize_texts(
        normalized_heldout["clinical_text"]
    )
    assert normalized_train["clinical_text"].tolist() != train_df[
        "clinical_text"
    ].tolist()
    with pytest.raises(ValueError, match="exact row/text scope"):
        _validate(tmp_path / "capture", normalized_train, normalized_heldout)
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

    assert validated["content_sha256"] == metadata["content_sha256"]
    assert len(validated["nuisance_fold_states"]) == 2
    assert {
        row["effect_objective"] for row in validated["effect_fold_states"]
    } == {"pseudo_outcome_mse", "squared_r_loss"}
    assert validated["heldout_labels_accessed"] is False
    assert sorted(path.name for path in (tmp_path / "capture").iterdir()) == [
        "arrays.npz",
        "metadata.json",
    ]


def test_htr_capture_sink_rejects_normalized_binding_for_raw_runner_text(
    tmp_path: Path,
) -> None:
    data = _dataset()
    config = _config(tmp_path)
    runner = MultiModelForestStage1Runner(
        dataset=data,
        config=config,
        output_path=tmp_path / "unused.parquet",
        device=torch.device("cpu"),
        num_workers=1,
    )
    train_df = runner.dataset.iloc[:10].reset_index(drop=True)
    heldout_df = runner.dataset.iloc[10:][["_oci_row_id", "clinical_text"]].reset_index(
        drop=True
    )
    sink = NativeHTRProofCaptureSink(
        artifact_dir=tmp_path / "wrong_projection",
        scope_id="outer_001_inner_001",
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=train_df["_oci_row_id"].tolist(),
        heldout_row_ids=heldout_df["_oci_row_id"].tolist(),
        fit_texts=_normalize_texts(train_df["clinical_text"]),
        heldout_texts=_normalize_texts(heldout_df["clinical_text"]),
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        e_clip=float(config.architecture.multi_model_forest.e_clip),
        nuisance_folds=2,
        effect_folds=2,
        model_tree_sha256=None,
        prediction_batch_size=4,
        seed=17,
    )
    with pytest.raises(ValueError, match="HTR fit text projection changed"):
        sink._check_rows(
            train_df,
            sink.fit_row_ids,
            sink.fit_texts,
            name="fit",
        )


def test_htr_capture_rejects_missing_fold_and_objective(tmp_path: Path):
    _, train_df, heldout_df = _run_capture(tmp_path)
    metadata_path = tmp_path / "capture" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["effect_fold_states"] = [
        row
        for row in metadata["effect_fold_states"]
        if not (
            row["effect_objective"] == "squared_r_loss" and int(row["fold"]) == 2
        )
    ]
    body = {key: value for key, value in metadata.items() if key != "content_sha256"}
    import hashlib

    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    metadata["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="fold coverage"):
        _validate(tmp_path / "capture", train_df, heldout_df)


def test_htr_capture_rejects_numerical_tamper(tmp_path: Path):
    _, train_df, heldout_df = _run_capture(tmp_path)
    arrays_path = tmp_path / "capture" / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    key = next(name for name in arrays if name.endswith("validation_e_hat"))
    arrays[key][0] += 0.1
    np.savez_compressed(arrays_path, **arrays)

    with pytest.raises((ValueError, RuntimeError)):
        _validate(tmp_path / "capture", train_df, heldout_df)


def test_htr_capture_is_invariant_to_unavailable_heldout_labels(tmp_path: Path):
    first, first_train, first_heldout = _run_capture(
        tmp_path,
        heldout_outcomes=(0, 1),
        artifact_name="first",
    )
    second, second_train, second_heldout = _run_capture(
        tmp_path,
        heldout_outcomes=(1, 0),
        artifact_name="second",
    )

    assert first["array_inventory"] == second["array_inventory"]
    assert first["nuisance_fold_states"] == second["nuisance_fold_states"]
    assert first["effect_fold_states"] == second["effect_fold_states"]
    assert first["scope_outputs"] == second["scope_outputs"]
    _validate(tmp_path / "first", first_train, first_heldout)
    _validate(tmp_path / "second", second_train, second_heldout)


def test_htr_native_family_registration_replays_and_rejects_every_artifact_tamper(
    tmp_path: Path,
):
    htr_model_path = _tiny_local_bert(tmp_path / "htr_model")
    model_tree_sha256 = directory_tree_sha256(htr_model_path)
    component_root = tmp_path / "component"
    component_root.mkdir()
    capture_relative = Path("native_htr_models") / "outer_001_inner_001"
    _, train_df, heldout_df = _run_capture(
        component_root,
        artifact_name=capture_relative.as_posix(),
        htr_model_path=htr_model_path,
    )
    fit_row_ids = train_df["_oci_row_id"].tolist()
    heldout_row_ids = heldout_df["_oci_row_id"].tolist()
    source_path = component_root / "raw_evidence_sidecars" / "scope.json"
    source_body = {
        "schema_version": STAGE1_RAW_EVIDENCE_SIDECAR_SCHEMA,
        "scope_id": "outer_001_inner_001",
        "outer_fold": 1,
        "inner_fold": 1,
        "fit_row_fingerprint": row_set_fingerprint(fit_row_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_row_ids),
        "prompt_grounding_allowed": False,
        "heldout_labels_supplied": False,
    }
    _write_immutable_json(
        source_path,
        {**source_body, "content_sha256": _sha256_json(source_body)},
    )
    configuration = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
        "scope_id": "outer_001_inner_001",
        "text_column": "clinical_text",
        "treatment_column": "treatment_indicator",
        "outcome_column": "outcome_indicator",
        "outcome_type": "binary",
        "e_clip": 0.01,
        "nuisance_folds": 2,
        "effect_folds": 2,
        "effect_objectives": ["pseudo_outcome_mse", "squared_r_loss"],
        "nuisance_calibration": "none",
        "capture_schema_version": "production_htr_native_capture_v1",
        "htr_model_tree_sha256": model_tree_sha256,
        "heldout_label_policy": "id_and_text_only",
        "split_registry_content_sha256": "c" * 64,
    }
    catalog = _htr_catalog(fit_row_ids, heldout_row_ids)
    assert catalog.family_atoms(HTR_NEURAL)
    registration = _register_htr_native_family_proof(
        component_root=component_root,
        proof_directory=Path("native_htr_family_proofs") / "outer_001_inner_001",
        scope_id="outer_001_inner_001",
        catalog=catalog,
        capture_artifact_path=component_root / capture_relative,
        source_artifact_path=source_path,
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=fit_row_ids,
        heldout_row_ids=heldout_row_ids,
        fit_texts=train_df["clinical_text"].tolist(),
        heldout_texts=heldout_df["clinical_text"].tolist(),
        fit_treatment=train_df["treatment_indicator"].tolist(),
        fit_outcome=train_df["outcome_indicator"].tolist(),
        split_scope_fingerprint="a" * 64,
        data_projection_sha256="b" * 64,
        configuration=configuration,
        htr_model_path=htr_model_path,
        htr_model_sha256=model_tree_sha256,
        device="cpu",
    )
    assert registration["registered_families"] == list(
        PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS
    )
    index_body = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
        "split_registry_content_sha256": "c" * 64,
        "registered_families": list(PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS),
        "exact_inner_scope_count": 1,
        "executable_checkpoint_files_retained": False,
        "scopes": [
            {
                "scope_id": "outer_001_inner_001",
                "outer_fold": 1,
                "inner_fold": 1,
                "registered_families": list(
                    PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS
                ),
                "content_sha256": registration["content_sha256"],
                "registration": registration["registration"],
            }
        ],
    }
    index_path = component_root / "htr_native_family_proof_index.json"
    _write_immutable_json(
        index_path,
        {**index_body, "content_sha256": _sha256_json(index_body)},
    )
    index_registration = _component_file_registration(
        index_path,
        component_root=component_root,
    )
    expected_scopes = {
        "outer_001_inner_001": {
            "scope_id": "outer_001_inner_001",
            "outer_fold": 1,
            "inner_fold": 1,
            "fit_row_ids": fit_row_ids,
            "heldout_row_ids": heldout_row_ids,
        }
    }
    modeling_data = _dataset().copy()
    validated = _validate_htr_native_family_proof_index(
        component_root=component_root,
        index_registration=index_registration,
        expected_inner_scopes=expected_scopes,
        split_registry_content_sha256="c" * 64,
        modeling_data=modeling_data,
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        htr_model_path=htr_model_path,
        htr_model_sha256=model_tree_sha256,
        device="cpu",
    )
    assert validated["exact_inner_scope_count"] == 1

    family_row = registration["family_proofs"][0]
    targets = {
        "model": family_row["model_artifact"]["relative_path"] + "/arrays.npz",
        "source": family_row["source_artifact"]["relative_path"],
        "metadata": family_row["native_fit_metadata"]["relative_path"],
        "execution": family_row["native_execution_record"]["relative_path"],
        "registration": registration["registration"]["relative_path"],
        "index": index_registration["relative_path"],
    }
    for target_name, relative_path in targets.items():
        tampered_root = tmp_path / f"tampered_{target_name}"
        shutil.copytree(component_root, tampered_root)
        target = tampered_root / relative_path
        tampered_index_registration = dict(index_registration)
        if target_name == "index":
            payload = json.loads(target.read_text(encoding="utf-8"))
            payload["exact_inner_scope_count"] = 2
            target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            tampered_index_registration = _component_file_registration(
                target,
                component_root=tampered_root,
            )
        else:
            target.write_bytes(target.read_bytes() + b"\n")
        with pytest.raises((ValueError, RuntimeError, json.JSONDecodeError)):
            _validate_htr_native_family_proof_index(
                component_root=tampered_root,
                index_registration=tampered_index_registration,
                expected_inner_scopes=expected_scopes,
                split_registry_content_sha256="c" * 64,
                modeling_data=modeling_data,
                text_column="clinical_text",
                treatment_column="treatment_indicator",
                outcome_column="outcome_indicator",
                htr_model_path=htr_model_path,
                htr_model_sha256=model_tree_sha256,
                device="cpu",
            )
