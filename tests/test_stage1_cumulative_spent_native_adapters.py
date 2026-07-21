from __future__ import annotations

import json
from pathlib import Path

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
from oci.inference.all_evidence_discovery_interfaces import (
    BOW_NUISANCE,
    BOW_R_LOSS,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
)
from oci.inference.all_evidence_fusion import (
    LEGACY_ALL_SOURCE,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
)
from oci.inference.bow_native_proof_capture import NativeBoWProofCaptureSink
from oci.inference.htr_native_proof_capture import NativeHTRProofCaptureSink
from oci.inference.matched_pair_native_proof_capture import (
    NativeMatchedPairProofCaptureSink,
)
from oci.inference.multi_model_agentic_forest import _normalize_texts
from oci.inference.multi_model_forest_stage1 import MultiModelForestStage1Runner
from oci.inference.lossless_stage1_evidence_catalog import (
    build_role_neutral_evidence_catalog,
)
from oci.inference.production_stage1_bundle import (
    PRODUCTION_CUMULATIVE_LEGACY_NATIVE_FAMILY_ADAPTERS,
    _catalog_ready_legacy_digest,
    _component_file_registration,
    _cumulative_legacy_configuration_by_family,
    _register_legacy_cumulative_spent_native_scope,
    _validate_legacy_cumulative_spent_native_index,
    _write_legacy_cumulative_spent_native_index,
)
from oci.inference.review_spent_evidence_provider import _htr_concepts_only
from oci.inference.stage1_cumulative_spent_evidence import (
    CUMULATIVE_SPENT_REFIT,
    CumulativeSpentStage1FamilyRequest,
    cumulative_spent_data_projection_sha256,
)
from oci.inference.stage1_cumulative_spent_native_adapters import (
    CUMULATIVE_SPENT_REPLAY_CANARY_SEMANTICS,
    CumulativeSpentReplayCanary,
    bind_cumulative_spent_native_family_producer,
    cumulative_spent_native_execution_record,
    cumulative_spent_native_family_identity,
)
from oci.inference.stage1_exact_inner_evidence import Stage1FitRow


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
                    disable_reason="cumulative native adapter unit test",
                ),
            ),
        ),
    )


def _request() -> CumulativeSpentStage1FamilyRequest:
    texts = (
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
    )
    treatment = (1, 0, 1, 0, 1, 0, 1, 0, 1, 0)
    outcome = (1, 0, 0, 1, 1, 0, 1, 0, 1, 0)
    spent = tuple(
        Stage1FitRow(
            row_id=index,
            text=text,
            treatment=treatment[index],
            outcome=outcome[index],
        )
        for index, text in enumerate(texts)
    )
    sealed = (10, 11)
    projection = cumulative_spent_data_projection_sha256(
        outer_fold=1,
        context_epoch=0,
        spent_rows=spent,
        sealed_row_ids=sealed,
    )
    provenance = FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=tuple(row.row_id for row in spent),
        heldout_row_ids=sealed,
        scope="inner_train",
        inner_fold=1,
        artifact_id="cumulative-native-all-legacy-test",
    )
    return CumulativeSpentStage1FamilyRequest(
        family=BOW_NUISANCE,
        request_sha256="a" * 64,
        schedule_sha256="b" * 64,
        scope_id="outer_001_hierarchy_epoch_000",
        outer_fold=1,
        context_epoch=0,
        provider_inner_fold=1,
        split_scope_fingerprint="c" * 64,
        data_projection_sha256=projection,
        spent_rows=spent,
        sealed_row_ids=sealed,
    )


def _all_legacy_request(family: str) -> CumulativeSpentStage1FamilyRequest:
    texts = tuple(
        f"Patient {index} ≥50% – "
        f"{'smoker brain' if index % 2 else 'never liver'} "
        f"{'response stable' if index % 3 else 'progression frail'}"
        for index in range(12)
    )
    treatment = tuple(index % 2 for index in range(12))
    outcome = (0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1)
    spent = tuple(
        Stage1FitRow(
            row_id=index,
            text=text,
            treatment=treatment[index],
            outcome=outcome[index],
        )
        for index, text in enumerate(texts)
    )
    sealed = (12, 13)
    projection = cumulative_spent_data_projection_sha256(
        outer_fold=1,
        context_epoch=0,
        spent_rows=spent,
        sealed_row_ids=sealed,
    )
    provenance = FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=tuple(row.row_id for row in spent),
        heldout_row_ids=sealed,
        scope="inner_train",
        inner_fold=1,
        artifact_id="cumulative-native-all-legacy-test",
    )
    return CumulativeSpentStage1FamilyRequest(
        family=family,
        request_sha256="d" * 64,
        schedule_sha256="e" * 64,
        scope_id="outer_001_hierarchy_epoch_000",
        outer_fold=1,
        context_epoch=0,
        provider_inner_fold=1,
        split_scope_fingerprint=provenance.split_fingerprint,
        data_projection_sha256=projection,
        spent_rows=spent,
        sealed_row_ids=sealed,
    )


def _all_legacy_config(tmp_path: Path) -> AppliedInferenceConfig:
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
            htr_sentence_model="hash",
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
                    disable_reason="cumulative legacy capture test",
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
    config.seed = 23
    return config


def _capture(tmp_path: Path):
    request = _request()
    canary = CumulativeSpentReplayCanary.from_request(request)
    config = _config()
    frame = pd.DataFrame(
        {
            "clinical_text": [row.text for row in request.spent_rows],
            "treatment_indicator": [row.treatment for row in request.spent_rows],
            "outcome_indicator": [row.outcome for row in request.spent_rows],
        }
    )
    capture_path = tmp_path / "native_bow_capture"
    sink = NativeBoWProofCaptureSink(
        artifact_dir=capture_path,
        scope_id=request.scope_id,
        outer_fold=request.outer_fold,
        inner_fold=request.provider_inner_fold,
        fit_row_ids=request.spent_row_ids,
        heldout_row_ids=(canary.alias_row_id,),
        fit_texts=tuple(row.text for row in request.spent_rows),
        heldout_texts=(canary.text,),
        text_column="clinical_text",
        outcome_type="binary",
        e_clip=config.architecture.multi_model_forest.e_clip,
        nuisance_folds=2,
        effect_folds=2,
        view_configs=[vars(view) for view in config.architecture.multi_model_forest.bow_views],
    )
    runner = MultiModelForestStage1Runner(
        dataset=frame,
        config=config,
        output_path=tmp_path / "unused.parquet",
        num_workers=1,
        bow_native_capture_sink=sink,
    )
    train_df = runner.dataset.copy()
    transform_df = canary.transform_frame(text_column="clinical_text")
    bundle = runner._build_feature_bundle(
        train_df=train_df,
        test_df=transform_df,
        outer_fold=1,
    )
    sink.finalize()
    return request, canary, capture_path, bundle


def test_replay_canary_contains_only_a_spent_text_alias() -> None:
    request = _request()
    canary = CumulativeSpentReplayCanary.from_request(request)
    frame = canary.transform_frame(text_column="clinical_text")

    assert list(frame.columns) == ["_oci_row_id", "clinical_text"]
    assert frame.iloc[0]["clinical_text"] == request.spent_rows[0].text
    assert canary.alias_row_id not in set(request.spent_row_ids) | set(request.sealed_row_ids)
    assert canary.binding["semantics"] == CUMULATIVE_SPENT_REPLAY_CANARY_SEMANTICS
    assert canary.binding["alias_is_cohort_row"] is False
    assert canary.binding["source_labels_copied_to_transform"] is False
    assert canary.binding["contributes_to_concept_evidence"] is False
    serialized = json.dumps(canary.binding, sort_keys=True)
    assert request.spent_rows[0].text not in serialized


def test_genuine_bow_capture_binds_as_cumulative_spent_producer_and_revalidates(
    tmp_path: Path,
) -> None:
    request, canary, capture_path, bundle = _capture(tmp_path)
    phrase_rows = bundle.handoff_evidence["importance"]["views"][0]["phrase_features"]
    payload = {
        "schema_version": "native_stage1_family_concept_evidence_v1",
        "family": BOW_NUISANCE,
        "architecture_evidence": [
            {
                "atom_kind": "bow_phrase_importance",
                "source_kind": "legacy_all",
                "observable_axes": ["treatment", "outcome"],
                "content": {"terms": phrase_rows},
            }
        ],
    }
    source_path = tmp_path / "component_source.json"
    source_path.write_text(json.dumps({"payload": payload}, sort_keys=True), encoding="utf-8")
    identity = cumulative_spent_native_family_identity(
        family=BOW_NUISANCE,
        configuration={
            "capture_schema": "production_bow_native_capture_v1",
            "transform_policy": CUMULATIVE_SPENT_REPLAY_CANARY_SEMANTICS,
            "nuisance_folds": 2,
            "effect_folds": 2,
        },
    )
    record = cumulative_spent_native_execution_record(
        request=request,
        producer_identity=identity,
        evidence_payload=payload,
        evidence_item_count=1,
        replay_canary=canary,
        capture_artifact_path=capture_path,
        source_artifact_path=source_path,
    )
    execution_path = tmp_path / "component_execution.json"
    execution_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    producer = bind_cumulative_spent_native_family_producer(
        request=request,
        producer_identity=identity,
        evidence_payload=payload,
        evidence_item_count=1,
        replay_canary=canary,
        capture_artifact_path=capture_path,
        source_artifact_path=source_path,
        execution_record_path=execution_path,
    )

    draft = producer.produce_cumulative_spent(request)
    assert draft.fit_semantics == CUMULATIVE_SPENT_REFIT
    assert draft.input_binding_sha256 == request.binding_sha256
    assert draft.fit_audit["sealed_text_accessed"] is False
    assert draft.fit_audit["sealed_labels_accessed"] is False
    assert record["replay_canary"]["alias_is_cohort_row"] is False
    assert record["replay_canary_contributes_to_concept_evidence"] is False

    source_path.write_text(json.dumps({"payload": {"changed": True}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="execution record changed"):
        producer.produce_cumulative_spent(request)


def test_cumulative_native_producer_rejects_another_family_request(tmp_path: Path) -> None:
    request, canary, capture_path, _bundle = _capture(tmp_path)
    payload = {"concepts": [{"term": "alpha brain"}]}
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    identity = cumulative_spent_native_family_identity(
        family=BOW_NUISANCE,
        configuration={"test": True},
    )
    record = cumulative_spent_native_execution_record(
        request=request,
        producer_identity=identity,
        evidence_payload=payload,
        evidence_item_count=1,
        replay_canary=canary,
        capture_artifact_path=capture_path,
        source_artifact_path=source_path,
    )
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    producer = bind_cumulative_spent_native_family_producer(
        request=request,
        producer_identity=identity,
        evidence_payload=payload,
        evidence_item_count=1,
        replay_canary=canary,
        capture_artifact_path=capture_path,
        source_artifact_path=source_path,
        execution_record_path=execution_path,
    )
    changed = CumulativeSpentStage1FamilyRequest(
        family=BOW_R_LOSS,
        request_sha256=request.request_sha256,
        schedule_sha256=request.schedule_sha256,
        scope_id=request.scope_id,
        outer_fold=request.outer_fold,
        context_epoch=request.context_epoch,
        provider_inner_fold=request.provider_inner_fold,
        split_scope_fingerprint=request.split_scope_fingerprint,
        data_projection_sha256=request.data_projection_sha256,
        spent_rows=request.spent_rows,
        sealed_row_ids=request.sealed_row_ids,
    )
    with pytest.raises(ValueError, match="another request"):
        producer.produce_cumulative_spent(changed)


def test_htr_and_matched_captures_replay_on_spent_alias_without_sealed_text(
    tmp_path: Path,
) -> None:
    torch.manual_seed(4321)
    htr_request = _all_legacy_request(HTR_NEURAL)
    matched_request = _all_legacy_request(MATCHED_PAIR_UPLIFT)
    canary = CumulativeSpentReplayCanary.from_request(htr_request)
    canary.assert_matches(matched_request)
    config = _all_legacy_config(tmp_path)
    frame = pd.DataFrame(
        {
            "clinical_text": [row.text for row in htr_request.spent_rows],
            "treatment_indicator": [row.treatment for row in htr_request.spent_rows],
            "outcome_indicator": [row.outcome for row in htr_request.spent_rows],
        }
    )
    raw_fit_texts = tuple(row.text for row in htr_request.spent_rows)
    raw_canary_texts = (canary.text,)
    normalized_fit_texts = tuple(_normalize_texts(raw_fit_texts))
    normalized_canary_texts = tuple(_normalize_texts(raw_canary_texts))
    assert raw_fit_texts != normalized_fit_texts
    assert raw_canary_texts != normalized_canary_texts
    bow_path = tmp_path / "native_bow_capture"
    pair_config = config.architecture.multi_model_forest
    bow_sink = NativeBoWProofCaptureSink(
        artifact_dir=bow_path,
        scope_id=htr_request.scope_id,
        outer_fold=htr_request.outer_fold,
        inner_fold=htr_request.provider_inner_fold,
        fit_row_ids=htr_request.spent_row_ids,
        heldout_row_ids=(canary.alias_row_id,),
        fit_texts=normalized_fit_texts,
        heldout_texts=normalized_canary_texts,
        text_column="clinical_text",
        outcome_type="binary",
        e_clip=float(pair_config.e_clip),
        nuisance_folds=2,
        effect_folds=2,
        view_configs=[vars(view) for view in pair_config.bow_views],
    )
    htr_path = tmp_path / "native_htr_capture"
    htr_config = config.architecture.agentic_attention_variable_forest
    htr_sink = NativeHTRProofCaptureSink(
        artifact_dir=htr_path,
        scope_id=htr_request.scope_id,
        outer_fold=htr_request.outer_fold,
        inner_fold=htr_request.provider_inner_fold,
        fit_row_ids=htr_request.spent_row_ids,
        heldout_row_ids=(canary.alias_row_id,),
        fit_texts=raw_fit_texts,
        heldout_texts=raw_canary_texts,
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        e_clip=float(htr_config.e_clip),
        nuisance_folds=2,
        effect_folds=2,
        model_tree_sha256=None,
        prediction_batch_size=4,
        seed=23,
    )
    pair_path = tmp_path / "native_matched_capture"
    matched_sink = NativeMatchedPairProofCaptureSink(
        artifact_dir=pair_path,
        scope_id=matched_request.scope_id,
        outer_fold=matched_request.outer_fold,
        inner_fold=matched_request.provider_inner_fold,
        fit_row_ids=matched_request.spent_row_ids,
        heldout_row_ids=(canary.alias_row_id,),
        fit_texts=normalized_fit_texts,
        heldout_texts=normalized_canary_texts,
        text_column="clinical_text",
        effect_folds=2,
        view_configs=[vars(view) for view in pair_config.bow_views],
        propensity_caliper=1.0,
        outcome_caliper=1.0,
        max_controls_per_candidate=2,
        nearest_fallback_controls=1,
        htr_model_tree_sha256=None,
        htr_prediction_batch_size=4,
        seed=23,
    )
    runner = MultiModelForestStage1Runner(
        dataset=frame,
        config=config,
        output_path=tmp_path / "legacy_runner.parquet",
        device="cpu",
        num_workers=0,
        bow_native_capture_sink=bow_sink,
        htr_native_capture_sink=htr_sink,
        matched_pair_native_capture_sink=matched_sink,
    )
    train_df = runner.dataset.copy()
    provider = runner._htr_provider()
    native_runner = provider._ensure_runner(train_df)
    native_runner._attention_evidence = lambda *args, **kwargs: []
    bundle = runner._build_feature_bundle(
        train_df=train_df,
        test_df=canary.transform_frame(text_column="clinical_text"),
        outer_fold=1,
    )
    bow_sink.finalize()
    htr_sink.finalize()
    matched_sink.finalize()

    for request, capture_path, expected_kind in (
        (htr_request, htr_path, "htr"),
        (matched_request, pair_path, "matched_pair"),
    ):
        payload = {"concepts": [{"term": f"{request.family} spent marker"}]}
        source_path = tmp_path / f"{request.family}_source.json"
        source_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        identity = cumulative_spent_native_family_identity(
            family=request.family,
            configuration={
                "transform_policy": CUMULATIVE_SPENT_REPLAY_CANARY_SEMANTICS,
                "nuisance_folds": 2,
                "effect_folds": 2,
            },
        )
        record = cumulative_spent_native_execution_record(
            request=request,
            producer_identity=identity,
            evidence_payload=payload,
            evidence_item_count=1,
            replay_canary=canary,
            capture_artifact_path=capture_path,
            source_artifact_path=source_path,
            device="cpu",
        )
        assert record["capture_kind"] == expected_kind
        assert record["sealed_text_accessed"] is False
        assert record["sealed_labels_accessed"] is False
        assert record["replay_canary"]["source_labels_copied_to_transform"] is False

    fitted = bundle.handoff_evidence
    assert fitted is not None
    digest = _catalog_ready_legacy_digest(
        importance=fitted.get("importance") or {},
        embedding_evidence={},
        htr_evidence={
            "nuisance": {
                "attention": [
                    {"phrase": "baseline brain marker", "attention_score": 0.7}
                ]
            },
            "effect": {
                "attention": [
                    {"phrase": "response modifier marker", "attention_score": 0.6}
                ]
            },
            "pair_uplift": {
                "attention": [
                    {"phrase": "paired uplift marker", "attention_score": 0.5}
                ]
            },
        },
    )
    provenance = FoldEvidenceProvenance(
        outer_fold=htr_request.outer_fold,
        train_row_ids=htr_request.spent_row_ids,
        heldout_row_ids=htr_request.sealed_row_ids,
        scope="inner_train",
        inner_fold=htr_request.provider_inner_fold,
        artifact_id="production-cumulative-legacy-registration-test",
    )
    catalog = build_role_neutral_evidence_catalog(
        (
            FoldEvidenceInput(
                LEGACY_ALL_SOURCE,
                {
                    "outer_fold": htr_request.outer_fold,
                    "inner_fold": htr_request.provider_inner_fold,
                    "scope": "inner_train",
                    "n_rows": len(htr_request.spent_row_ids),
                    "context": {"evidence_digest": digest},
                },
                provenance,
            ),
        ),
        require_all_source_kinds=False,
        require_all_architecture_families=False,
        require_upstream_completeness=False,
    )
    configurations = _cumulative_legacy_configuration_by_family(
        config=config,
        scope_id=htr_request.scope_id,
        split_registry_content_sha256="1" * 64,
        htr_model_tree_sha256="2" * 64,
        seed=23,
    )
    registration = _register_legacy_cumulative_spent_native_scope(
        component_root=tmp_path,
        proof_directory=Path("cumulative_proofs") / htr_request.scope_id,
        request=htr_request,
        catalog=catalog,
        replay_canary=canary,
        capture_artifact_by_family={
            BOW_NUISANCE: bow_path,
            BOW_R_LOSS: bow_path,
            HTR_NEURAL: htr_path,
            MATCHED_PAIR_UPLIFT: pair_path,
        },
        configuration_by_family=configurations,
        device="cpu",
    )
    assert registration["registered_families"] == list(
        PRODUCTION_CUMULATIVE_LEGACY_NATIVE_FAMILY_ADAPTERS
    )
    assert all(row["evidence_item_count"] > 0 for row in registration["family_proofs"])
    index_registration = _write_legacy_cumulative_spent_native_index(
        component_root=tmp_path,
        index_path=Path("cumulative_legacy_index.json"),
        request_sha256=htr_request.request_sha256,
        schedule_sha256=htr_request.schedule_sha256,
        split_registry_content_sha256="1" * 64,
        scope_registrations=[registration],
    )
    validated, producers_by_scope = _validate_legacy_cumulative_spent_native_index(
        component_root=tmp_path,
        index_registration=index_registration,
        expected_requests={htr_request.scope_id: htr_request},
        expected_configuration_by_scope={htr_request.scope_id: configurations},
        request_sha256=htr_request.request_sha256,
        schedule_sha256=htr_request.schedule_sha256,
        split_registry_content_sha256="1" * 64,
        device="cpu",
    )
    assert validated["cumulative_scope_count"] == 1
    assert validated["sealed_text_available_to_producers"] is False
    assert validated["sealed_labels_available_to_producers"] is False
    assert set(producers_by_scope[htr_request.scope_id]) == set(
        PRODUCTION_CUMULATIVE_LEGACY_NATIVE_FAMILY_ADAPTERS
    )

    execution_registration = registration["family_proofs"][0]["execution_record"]
    execution_path = tmp_path / execution_registration["relative_path"]
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["sealed_text_accessed"] = True
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    with pytest.raises(RuntimeError, match="registered native component artifact changed"):
        _validate_legacy_cumulative_spent_native_index(
            component_root=tmp_path,
            index_registration=index_registration,
            expected_requests={htr_request.scope_id: htr_request},
            expected_configuration_by_scope={htr_request.scope_id: configurations},
            request_sha256=htr_request.request_sha256,
            schedule_sha256=htr_request.schedule_sha256,
            split_registry_content_sha256="1" * 64,
            device="cpu",
        )
