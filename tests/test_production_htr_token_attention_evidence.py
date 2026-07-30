from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from oci.inference.all_evidence_discovery_interfaces import HTR_NEURAL
from oci.inference.htr_attention_evidence_schema import (
    ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA,
    ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_PACKAGE_SCHEMA,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    _normalize_cumulative_family_payload,
)
from oci.inference.htr_stage2_complete_semantic_aggregation import (
    build_htr_semantic_aggregation_scope,
    validate_htr_semantic_aggregation_scope,
)
from oci.inference.role_neutral_htr_group_execution import (
    RoleNeutralHTRConfig,
    RoleNeutralHTRPhysicalGroupRequest,
    _SafeArrayStore,
    _complete_attention_evidence,
    _coverage_plan,
    _coverage_plan_values,
    _model_tree_sha256,
    _token_attention_package,
    _validate_complete_attention_evidence,
    execute_role_neutral_htr_physical_group,
    validate_role_neutral_htr_group_execution,
)
from oci.inference.production_stage1_scope_scheduler import (
    Stage1PhysicalFitIdentity,
    Stage1ScopeAssignment,
    Stage1ScopePlan,
    Stage1ScopeSpec,
    _stage1_scope_plan_body,
    derive_stage1_group_seed,
)
from oci.models.gated_attention_pooling import GatedAttentionPooling
from oci.models.hierarchical_transformer_extractor import (
    HierarchicalTransformerExtractor,
)


_REPOSITORY = Path(__file__).resolve().parents[1]
_PROFILE_ROOT = (
    _REPOSITORY
    / "artifacts/runtime_profiles/"
    "portable_all_evidence_r15_token_attention_complete_evidence_v1"
)
_MODEL_ROOT = (
    _REPOSITORY
    / "artifacts/local_models/bert_tiny_6f75de8b60a9_materialized"
)


def _sha256_json(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _minimal_scope_plan() -> Stage1ScopePlan:
    global_seed = 42
    fit_rows = (0, 1, 2, 3)
    scope = Stage1ScopeSpec(
        canonical_index=0,
        scope_id="outer_001_full",
        scope_kind="full_outer",
        outer_fold=1,
        inner_fold=None,
        context_epoch=None,
        provider_inner_fold=None,
        fit_row_ids=fit_rows,
        heldout_row_ids=(4,),
        global_seed=global_seed,
        scope_seed=derive_stage1_group_seed(global_seed, fit_rows),
    )
    assignment = Stage1ScopeAssignment(
        scope_id=scope.scope_id,
        gpu_id=None,
        execution_rank=0,
        fit_row_count=len(fit_rows),
        assigned_gpu_load_after=len(fit_rows),
    )
    identity = Stage1PhysicalFitIdentity(
        architecture_identity="a" * 64,
        target="focused_token_attention_fixture",
        scientific_configuration_identity="b" * 64,
        producer_identity="c" * 64,
        runtime_compatibility_class="focused-cpu-fixture",
    )
    body = _stage1_scope_plan_body(
        registry_content_sha256="d" * 64,
        global_seed=global_seed,
        review_rounds=1,
        initial_training_partitions=1,
        physical_fit_identity=identity,
        gpu_ids=(),
        scope_workers_per_gpu=1,
        scopes=(scope,),
        assignments=(assignment,),
    )
    plan = Stage1ScopePlan(
        registry_content_sha256="d" * 64,
        global_seed=global_seed,
        review_rounds=1,
        initial_training_partitions=1,
        physical_fit_identity=identity,
        gpu_ids=(),
        scope_workers_per_gpu=1,
        scopes=(scope,),
        assignments=(assignment,),
        content_sha256=_sha256_json(body),
    )
    plan.as_dict()
    return plan


def _scientific_htr_configuration() -> RoleNeutralHTRConfig:
    payload = json.loads(
        (
            _PROFILE_ROOT / "portable_all_evidence_scientific_nsclc.json"
        ).read_text(encoding="utf-8")
    )["architecture_profiles"]["hierarchical_transformer"][
        "producer_configuration"
    ]
    payload["model_tree_sha256"] = _model_tree_sha256(_MODEL_ROOT)
    return RoleNeutralHTRConfig.from_mapping(payload)


@pytest.fixture(scope="module")
def token_attention_extractor() -> HierarchicalTransformerExtractor:
    if not _MODEL_ROOT.is_dir():
        pytest.skip("materialized bert-tiny model is unavailable")
    torch.manual_seed(7302026)
    extractor = HierarchicalTransformerExtractor(
        sentence_encoder_model=str(_MODEL_ROOT),
        freeze_sentence_encoder=False,
        chunk_size_words=4,
        chunk_overlap_words=2,
        max_chunks=16,
        max_chunk_length=32,
        num_transformer_layers=1,
        num_attention_heads=1,
        transformer_dim=16,
        transformer_dropout=0.0,
        projection_dim=8,
        sentence_encoder_batch_size=4,
        sentence_encoder_backend="transformers",
        sentence_pooling="token_attention",
        normalize_sentence_embeddings=True,
        trainable_sentence_encoder_layers=0,
        transformer_feedforward_dim=32,
        transformer_attention_dropout=0.0,
        transformer_residual_dropout=0.0,
        transformer_feedforward_dropout=0.0,
        output_projection_depth=1,
        output_projection_hidden_dim=16,
        output_projection_dropout=0.0,
        environment_override_policy="forbid",
        device=torch.device("cpu"),
    )
    extractor.fit_tokenizer(
        ["alpha beta gamma delta epsilon zeta"]
    )
    return extractor


def test_production_profile_requires_learned_token_attention() -> None:
    config = _scientific_htr_configuration()
    scientific = json.loads(
        (
            _PROFILE_ROOT / "portable_all_evidence_scientific_nsclc.json"
        ).read_text(encoding="utf-8")
    )
    stage1 = json.loads(
        (
            _PROFILE_ROOT / "production_all_evidence_stage1_full.json"
        ).read_text(encoding="utf-8")
    )
    matched_extractor = scientific["architecture_profiles"][
        "matched_patient_uplift"
    ]["producer_configuration"]["htr_extractor"]

    assert config.sentence_pooling == "token_attention"
    assert config.freeze_sentence_encoder is False
    assert matched_extractor["sentence_pooling"] == "token_attention"
    assert (
        stage1["config"]["architecture"]["htr_sentence_pooling"]
        == "token_attention"
    )

    with pytest.raises(
        ValueError,
        match="sentence_pooling=token_attention",
    ):
        replace(config, sentence_pooling="auto").validated()
    with pytest.raises(
        ValueError,
        match="sentence_pooling=token_attention",
    ):
        replace(config, sentence_pooling="cls").validated()


def test_gated_token_pooling_masks_padding_and_normalizes() -> None:
    torch.manual_seed(11)
    pooler = GatedAttentionPooling(hidden_dim=5, attention_dim=7)
    hidden = torch.randn(2, 5, 5)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]],
        dtype=torch.int64,
    )

    _pooled, weights = pooler(hidden, attention_mask=mask)

    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(2),
        rtol=0.0,
        atol=1e-7,
    )
    assert torch.count_nonzero(weights[mask == 0]).item() == 0
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)


def test_complete_inventory_retains_specials_ids_offsets_and_overlap(
    token_attention_extractor: HierarchicalTransformerExtractor,
) -> None:
    text = (
        "alpha beta gamma delta epsilon zeta eta theta "
        "gamma delta"
    )
    first = token_attention_extractor.complete_attention_inventory(
        [text],
        role="effect_modifier",
    )
    second = token_attention_extractor.complete_attention_inventory(
        [text],
        role="effect_modifier",
    )

    assert first == second
    assert first["sentence_pooling"] == "token_attention"
    assert first["effective_sentence_pooling"] == "token_attention"
    chunks = first["notes"][0]["chunks"]
    assert len(chunks) > 1
    assert np.isclose(
        sum(row["chunk_attention"] for row in chunks),
        1.0,
        rtol=0.0,
        atol=1e-5,
    )

    tokenizer = token_attention_extractor._tokenizer
    special_mass = 0.0
    gamma_occurrences = 0
    for chunk in chunks:
        tokens = chunk["tokens"]
        assert np.isclose(
            sum(row["token_attention"] for row in tokens),
            1.0,
            rtol=0.0,
            atol=1e-5,
        )
        ids = [row["token_id"] for row in tokens]
        assert tokenizer.convert_ids_to_tokens(ids) == [
            row["decoded_token_text"] for row in tokens
        ]
        encoded = tokenizer(
            chunk["chunk_text"],
            add_special_tokens=True,
            truncation=True,
            max_length=32,
            return_offsets_mapping=True,
        )
        assert ids == encoded["input_ids"]
        assert [
            (row["char_start"], row["char_end"]) for row in tokens
        ] == [tuple(value) for value in encoded["offset_mapping"]]
        for row in tokens:
            if row["is_special_token"]:
                special_mass += row["token_attention"]
                assert (row["char_start"], row["char_end"]) == (0, 0)
            else:
                surface = chunk["chunk_text"][
                    row["char_start"] : row["char_end"]
                ]
                assert surface
                gamma_occurrences += int(surface.lower() == "gamma")
            assert row["is_padding"] is False
    assert special_mass > 0.0
    assert gamma_occurrences >= 2


def test_real_htr_component_seals_replays_and_attests_complete_attention(
    tmp_path: Path,
) -> None:
    if not _MODEL_ROOT.is_dir():
        pytest.skip("materialized bert-tiny model is unavailable")
    plan = _minimal_scope_plan()
    request = RoleNeutralHTRPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id="outer_001_full",
    )
    config = replace(
        _scientific_htr_configuration(),
        chunk_size_words=4,
        chunk_overlap_words=1,
        max_chunks=8,
        max_chunk_length=32,
        num_transformer_layers=1,
        num_attention_heads=1,
        transformer_dim=16,
        transformer_dropout=0.0,
        projection_dim=8,
        hash_embedding_dim=8,
        sentence_encoder_batch_size=2,
        sentence_encoder_backend="transformers",
        normalize_sentence_embeddings=True,
        trainable_sentence_encoder_layers=0,
        role_attention=False,
        w_attention_heads=1,
        x_attention_heads=1,
        transformer_feedforward_dim=32,
        transformer_attention_dropout=0.0,
        transformer_residual_dropout=0.0,
        transformer_feedforward_dropout=0.0,
        output_projection_depth=1,
        output_projection_hidden_dim=16,
        output_projection_dropout=0.0,
        hidden_dim=8,
        nuisance_head_depth=1,
        nuisance_head_dropout=0.0,
        effect_head_depth=1,
        effect_head_dropout=0.0,
        nuisance_folds=2,
        effect_folds=2,
        nuisance_epochs=1,
        effect_epochs=1,
        batch_size=2,
        prediction_batch_size=2,
        nuisance_calibration="none",
        nuisance_label_smoothing=0.0,
        effect_objectives=("pseudo_outcome_mse",),
        r_stage_min_propensity=0.0,
        r_stage_max_propensity=1.0,
    ).validated()
    fit_texts = (
        "alpha beta gamma delta epsilon",
        "therapy marker response alpha beta",
        "outcome gamma delta modifier",
        "epsilon therapy response marker",
    )
    treatment = np.asarray([0.0, 1.0, 0.0, 1.0])
    outcome = np.asarray([0.0, 0.0, 1.0, 1.0])
    root = (tmp_path / "real-token-attention-htr").resolve()

    terminal = execute_role_neutral_htr_physical_group(
        request=request,
        output_root=root,
        fit_texts=fit_texts,
        fit_treatment=treatment,
        fit_outcome=outcome,
        config=config,
        runtime_compatibility_class="focused-cpu-fixture",
        exact_heldout_text_loader=lambda row_ids: (
            "heldout alpha gamma response",
        )
        if row_ids == (4,)
        else (_ for _ in ()).throw(
            AssertionError("unexpected held-out rows")
        ),
        htr_model_path=_MODEL_ROOT,
        device="cpu",
    )
    reopened = validate_role_neutral_htr_group_execution(
        root=root,
        request=request,
        htr_model_path=_MODEL_ROOT,
        device="cpu",
    )

    assert reopened == terminal
    assert terminal["pooling_attestation"]["sentence_pooling"] == (
        "token_attention"
    )
    assert terminal["pooling_attestation"][
        "effective_sentence_pooling"
    ] == "token_attention"
    evidence = terminal["attention_evidence_attestation"]
    assert evidence["token_occurrence_count"] > 0
    assert evidence["chunk_interpretation_count"] > 0
    assert evidence["special_token_occurrence_count"] > 0
    assert evidence["all_token_and_chunk_attention_normalized"] is True
    assert evidence["exact_fold_heldout_coverage"] is True
    assert (
        terminal["nuisance_oof_performance"][
            "every_fit_patient_predicted_exactly_once_while_held_out"
        ]
        is True
    )


def _complete_fold_payload(
    extractor: HierarchicalTransformerExtractor,
) -> tuple[
    RoleNeutralHTRConfig,
    dict[str, object],
    dict[str, np.ndarray],
    object,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    texts = (
        "alpha beta gamma delta epsilon zeta",
        "therapy response marker alpha beta gamma",
        "outcome modifier delta epsilon theta",
        "gamma delta overlap witness response",
    )
    row_ids = (101, 203, 307, 409)
    config = replace(
        _scientific_htr_configuration(),
        chunk_size_words=4,
        chunk_overlap_words=2,
        max_chunks=16,
        max_chunk_length=32,
        nuisance_folds=2,
        effect_folds=2,
        effect_objectives=("pseudo_outcome_mse",),
        prediction_batch_size=2,
    ).validated()
    coverage = _coverage_plan(
        texts=texts,
        config=config,
        phase="focused_token_attention_test",
    )
    nuisance_partitions = ((0, 2), (1, 3))
    effect_partitions = ((1, 2), (0, 3))
    store = _SafeArrayStore()
    atoms: list[dict[str, object]] = []
    batches: list[dict[str, object]] = []
    nuisance_records: list[dict[str, object]] = []
    effect_records: list[dict[str, object]] = []

    for fold, validation in enumerate(nuisance_partitions, start=1):
        fit = tuple(index for index in range(4) if index not in validation)
        evidence = _complete_attention_evidence(
            extractor,
            texts=[texts[index] for index in validation],
            coverage=coverage,
            row_positions=validation,
            row_ids=row_ids,
            fold=fold,
            stage="nuisance",
            objective="joint_treatment_outcome_nuisance",
            batch_size=2,
            array_store=store,
            array_prefix=f"test_nuisance_{fold:04d}",
        )
        atoms.extend(copy.deepcopy(list(evidence.architecture_evidence)))
        batches.append(copy.deepcopy(dict(evidence.token_attention_evidence)))
        nuisance_records.append(
            {
                "fold": fold,
                "fit_positions": list(fit),
                "validation_positions": list(validation),
                "fit_row_ids": [row_ids[index] for index in fit],
                "validation_row_ids": [
                    row_ids[index] for index in validation
                ],
            }
        )

    for fold, validation in enumerate(effect_partitions, start=1):
        fit = tuple(index for index in range(4) if index not in validation)
        evidence = _complete_attention_evidence(
            extractor,
            texts=[texts[index] for index in validation],
            coverage=coverage,
            row_positions=validation,
            row_ids=row_ids,
            fold=fold,
            stage="effect_modifier",
            objective="pseudo_outcome_mse",
            batch_size=2,
            array_store=store,
            array_prefix=f"test_effect_{fold:04d}",
        )
        atoms.extend(copy.deepcopy(list(evidence.architecture_evidence)))
        batches.append(copy.deepcopy(dict(evidence.token_attention_evidence)))
        effect_records.append(
            {
                "effect_objective": "pseudo_outcome_mse",
                "fold": fold,
                "fit_positions": list(fit),
                "eligible_fit_positions": list(fit),
                "validation_positions": list(validation),
                "fit_row_ids": [row_ids[index] for index in fit],
                "eligible_fit_row_ids": [
                    row_ids[index] for index in fit
                ],
                "validation_row_ids": [
                    row_ids[index] for index in validation
                ],
            }
        )

    package = _token_attention_package(batches, config=config)
    payload: dict[str, object] = {
        "schema_version": ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA,
        "family": HTR_NEURAL,
        "architecture_evidence": atoms,
        "token_attention_evidence": package,
    }
    return (
        config,
        payload,
        store.arrays,
        coverage,
        nuisance_records,
        effect_records,
    )


def test_complete_fold_honest_token_evidence_validates_and_catalogs(
    token_attention_extractor: HierarchicalTransformerExtractor,
    tmp_path: Path,
) -> None:
    (
        config,
        payload,
        arrays,
        coverage,
        nuisance_records,
        effect_records,
    ) = _complete_fold_payload(token_attention_extractor)

    for row in [*nuisance_records, *effect_records]:
        assert set(row["fit_positions"]).isdisjoint(
            row["validation_positions"]
        )
    _validate_complete_attention_evidence(
        payload=payload,
        coverage=_coverage_plan_values(coverage),
        nuisance_records=nuisance_records,
        effect_records=effect_records,
        config=config,
        arrays=arrays,
    )

    package = payload["token_attention_evidence"]
    assert package["all_raw_token_occurrences_authenticated"] is True
    assert package["top_k_applied_to_raw_inventory"] is False
    assert package["special_token_occurrence_count"] > 0
    assert package["padding_occurrence_count"] == 0
    expected_note_interpretations = 4 * (
        1 + len(config.effect_objectives)
    )
    assert package["note_interpretation_count"] == (
        expected_note_interpretations
    )

    source_payload_sha256 = _sha256_json(payload)
    array_root = (tmp_path / "semantic-raw-arrays").resolve()
    array_root.mkdir()
    for name, value in arrays.items():
        np.save(
            array_root / f"{name}.npy",
            value,
            allow_pickle=False,
        )
    aggregate = build_htr_semantic_aggregation_scope(
        root=(tmp_path / "semantic-aggregate").resolve(),
        source_payload=payload,
        source_array_store_root=array_root,
        source_fit_seal_content_sha256="a" * 64,
        source_payload_content_sha256=source_payload_sha256,
        source_fit_seal_locator=(
            "components/outer_001_full/htr/fit_only_family_seal.json"
        ),
        logical_scope_id="outer_001_hierarchy_epoch_000",
        physical_owner_scope_id="outer_001_full",
        outer_fold=1,
        context_epoch=0,
        scope_binding_sha256="b" * 64,
    )
    reopened = validate_htr_semantic_aggregation_scope(
        root=aggregate.scope_manifest_path.parent,
        source_payload=payload,
        source_array_store_root=array_root,
        expected_source_fit_seal_content_sha256="a" * 64,
        expected_source_payload_content_sha256=source_payload_sha256,
        expected_scope_binding_sha256="b" * 64,
    )
    normalized, audit = _normalize_cumulative_family_payload(
        reopened.payload,
        family=HTR_NEURAL,
        semantic_member_batch_size=16,
    )
    assert normalized["family"] == HTR_NEURAL
    assert normalized["architecture_evidence"]
    token_audit = audit["complete_token_attention_evidence"]
    assert token_audit["token_attention_package_content_sha256"] == (
        package["content_sha256"]
    )
    assert token_audit["token_occurrence_count"] == (
        package["token_occurrence_count"]
    )
    assert (
        token_audit[
            "raw_sidecars_authenticated_in_zero_copy_source_graph"
        ]
        is True
    )
    assert all(
        atom["content"]["aggregate_batch"][
            "complete_semantic_aggregate_delivery"
        ]
        is True
        for atom in normalized["architecture_evidence"]
    )


@pytest.mark.parametrize(
    "mutation",
    ("token_weight", "chunk_atom", "token_package", "missing_token"),
)
def test_altered_or_incomplete_token_evidence_is_rejected(
    token_attention_extractor: HierarchicalTransformerExtractor,
    mutation: str,
) -> None:
    (
        config,
        payload,
        arrays,
        coverage,
        nuisance_records,
        effect_records,
    ) = _complete_fold_payload(token_attention_extractor)
    changed_payload = copy.deepcopy(payload)
    changed_arrays = {
        key: np.array(value, copy=True) for key, value in arrays.items()
    }

    if mutation == "token_weight":
        descriptor = changed_payload["token_attention_evidence"][
            "fold_batches"
        ][0]["columns"]["token_attention"]
        changed_arrays[descriptor["array"]][0] += 0.125
    elif mutation == "chunk_atom":
        changed_payload["architecture_evidence"][0]["attention"] += 0.125
    elif mutation == "token_package":
        changed_payload["token_attention_evidence"][
            "token_occurrence_count"
        ] += 1
    else:
        descriptor = changed_payload["token_attention_evidence"][
            "fold_batches"
        ][0]["columns"]["token_id"]
        changed_arrays[descriptor["array"]] = changed_arrays[
            descriptor["array"]
        ][:-1]

    with pytest.raises(ValueError):
        _validate_complete_attention_evidence(
            payload=changed_payload,
            coverage=_coverage_plan_values(coverage),
            nuisance_records=nuisance_records,
            effect_records=effect_records,
            config=config,
            arrays=changed_arrays,
        )
