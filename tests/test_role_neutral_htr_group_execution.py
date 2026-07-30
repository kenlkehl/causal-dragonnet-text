from __future__ import annotations

import copy
import json
import os
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import oci.inference.role_neutral_htr_group_execution as group_module
from oci.inference.production_stage1_scope_scheduler import (
    build_canonical_stage1_scope_plan,
    stage1_torch_determinism_policy,
)
from tests.stage1_test_support import PHYSICAL_FIT_IDENTITY
from oci.inference.role_neutral_htr_group_execution import (
    RoleNeutralHTRConfig,
    RoleNeutralHTROperationalControls,
    RoleNeutralHTRPhysicalGroupRequest,
    _execute_htr_fold_tasks,
    _model_tree_sha256,
    _resolve_model_marker,
    execute_role_neutral_htr_physical_group,
    replay_role_neutral_htr_exact_transform,
    validate_role_neutral_htr_group_execution,
)
from oci.inference.stage1_upstream_gate_backend import (
    _directory_tree_sha256,
)
from oci.inference.role_neutral_all_ten_binding import (
    authenticate_role_neutral_htr_component,
)


def _registry() -> dict:
    row_count = 30
    all_rows = tuple(range(row_count))
    outer_rows = []
    for outer_fold in range(1, 3):
        start = (outer_fold - 1) * (row_count // 2)
        heldout = tuple(range(start, start + row_count // 2))
        fit = tuple(row for row in all_rows if row not in set(heldout))
        partitions = tuple(fit[index::5] for index in range(5))
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": list(fit),
                "heldout_row_ids": list(heldout),
                "inner_folds": [
                    {
                        "inner_fold": inner_fold,
                        "fit_row_ids": [
                            row for row in fit if row not in set(inner_heldout)
                        ],
                        "heldout_row_ids": list(inner_heldout),
                    }
                    for inner_fold, inner_heldout in enumerate(
                        partitions,
                        start=1,
                    )
                ],
            }
        )
    return {"dataset_row_count": row_count, "outer_folds": outer_rows}


def _plan(*, gpu_ids=(), scope_workers_per_gpu: int = 1):
    return build_canonical_stage1_scope_plan(
        registry=_registry(),
        registry_content_sha256="a" * 64,
        global_seed=42,
        physical_fit_identity=PHYSICAL_FIT_IDENTITY,
        gpu_ids=gpu_ids,
        scope_workers_per_gpu=scope_workers_per_gpu,
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=2,
        expected_inner_fold_count=5,
    )


def _request(plan=None) -> RoleNeutralHTRPhysicalGroupRequest:
    plan = _plan() if plan is None else plan
    owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if len(members) > 1
    )
    assert owner.scope_kind == "exact_inner"
    assert members[1].scope_kind == "cumulative_spent"
    return RoleNeutralHTRPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
    )


def _singleton_request(
    owner_kind: str,
) -> RoleNeutralHTRPhysicalGroupRequest:
    plan = _plan()
    owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if owner.scope_kind == owner_kind and len(members) == 1
    )
    assert members == (owner,)
    return RoleNeutralHTRPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
    )


def _config() -> RoleNeutralHTRConfig:
    # Every capacity/training value is explicit. There are no source defaults
    # for scientific HTR settings in the role-neutral executor.
    return RoleNeutralHTRConfig(
        sentence_encoder_model_kind="hash",
        model_tree_sha256=None,
        freeze_sentence_encoder=True,
        chunk_size_words=4,
        chunk_overlap_words=1,
        max_chunks=16,
        max_chunk_length=32,
        num_transformer_layers=1,
        num_attention_heads=1,
        transformer_dim=8,
        transformer_dropout=0.0,
        projection_dim=8,
        hash_embedding_dim=8,
        sentence_encoder_batch_size=8,
        sentence_encoder_backend="auto",
        sentence_pooling="auto",
        normalize_sentence_embeddings=True,
        trainable_sentence_encoder_layers=0,
        role_attention=False,
        w_attention_heads=1,
        x_attention_heads=1,
        transformer_feedforward_dim=13,
        transformer_activation="silu",
        transformer_norm_style="pre_norm",
        transformer_layer_norm_eps=1e-5,
        transformer_layer_norm_elementwise_affine=True,
        transformer_layer_norm_bias=True,
        transformer_attention_dropout=0.0,
        transformer_residual_dropout=0.0,
        transformer_feedforward_dropout=0.0,
        transformer_attention_bias=True,
        transformer_feedforward_bias=True,
        output_projection_depth=2,
        output_projection_hidden_dim=7,
        output_projection_activation="tanh",
        output_projection_dropout=0.0,
        output_projection_hidden_layer_norm=True,
        output_projection_final_layer_norm=True,
        output_projection_bias=True,
        pool_token_init_std=0.02,
        positional_encoding_base=10_000.0,
        environment_override_policy="forbid",
        require_live_unfrozen_encoder_attestation=False,
        hidden_dim=4,
        nuisance_head_depth=2,
        nuisance_head_activation="gelu_tanh",
        nuisance_head_dropout=0.05,
        nuisance_head_layer_norm=True,
        nuisance_head_bias=True,
        effect_head_depth=3,
        effect_head_activation="silu",
        effect_head_dropout=0.05,
        effect_head_layer_norm=True,
        effect_head_bias=True,
        nuisance_folds=2,
        effect_folds=2,
        nuisance_epochs=1,
        effect_epochs=1,
        batch_size=4,
        prediction_batch_size=4,
        optimizer_name="adamw",
        learning_rate=0.01,
        weight_decay=0.0,
        adamw_beta1=0.85,
        adamw_beta2=0.97,
        adamw_eps=1e-7,
        adamw_amsgrad=True,
        adamw_maximize=False,
        adamw_foreach=False,
        adamw_capturable=False,
        adamw_differentiable=False,
        adamw_fused=False,
        optimizer_zero_grad_set_to_none=True,
        alpha_propensity=1.0,
        nuisance_label_smoothing=0.0,
        nuisance_calibration="none",
        e_clip=0.02,
        r_stage_min_propensity=0.0,
        r_stage_max_propensity=1.0,
        gradient_clip_norm=1.0,
        gradient_clip_norm_type=2.0,
        gradient_clip_error_if_nonfinite=True,
        gradient_clip_foreach=False,
        effect_objectives=("squared_r_loss",),
        outcome_type="binary",
        replay_comparison_policy="allclose_and_exact_discrete_state_v1",
        replay_relative_tolerance=1e-4,
        replay_absolute_tolerance=1e-5,
    ).validated()


def test_runtime_model_tree_identity_matches_prepared_context_identity(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_bytes(b'{"model_type":"bert"}\n')
    (model_root / "weights.bin").write_bytes(b"\x00\x01\x02")
    prepared_identity = _directory_tree_sha256(model_root)
    config = replace(
        _config(),
        sentence_encoder_model_kind="authenticated_local_tree",
        model_tree_sha256=prepared_identity,
    ).validated()

    assert _model_tree_sha256(model_root) == prepared_identity
    assert _resolve_model_marker(config, model_root) == str(model_root.resolve())


def _inputs(request: RoleNeutralHTRPhysicalGroupRequest):
    fit_texts = [
        (
            f"patient row {row_id} biomarker_{position % 3} "
            f"therapy_{position % 2} stable"
        )
        for position, row_id in enumerate(request.physical_owner.fit_row_ids)
    ]
    # This sentinel is beyond 14,000 characters but remains in the complete
    # two-word HTR input. It catches any hidden character slicing.
    fit_texts[0] = ("x" * 14_500) + " sentinel_after_former_page_boundary"
    treatment = np.asarray(
        [position % 2 for position in range(len(fit_texts))],
        dtype=float,
    )
    outcome = np.asarray(
        [(position // 2) % 2 for position in range(len(fit_texts))],
        dtype=float,
    )
    heldout_texts = tuple(
        f"heldout row {row_id} complete exact transform"
        for row_id in request.physical_owner.heldout_row_ids
    )
    return tuple(fit_texts), treatment, outcome, heldout_texts


def _load_fit_artifact(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads(
        (root / "fit_state" / "metadata.json").read_text(encoding="utf-8")
    )
    arrays = {}
    for key, registration in metadata["array_inventory"].items():
        with (
            root / "fit_state" / registration["relative_path"]
        ).open("rb") as handle:
            arrays[key] = np.load(handle, allow_pickle=False)
    return metadata, arrays


def _assert_serial_parallel_science_equal(
    *,
    serial_root: Path,
    parallel_root: Path,
    request: RoleNeutralHTRPhysicalGroupRequest,
    config: RoleNeutralHTRConfig,
) -> None:
    serial, serial_arrays = _load_fit_artifact(serial_root)
    parallel, parallel_arrays = _load_fit_artifact(parallel_root)

    # Byte digests may differ for tolerance-valid neural floats. Everything
    # that defines or indexes the science must remain exact.
    def discrete_metadata(value: dict) -> dict:
        projected = copy.deepcopy(value)
        projected.pop("content_sha256")
        projected.pop("array_inventory")
        projected.pop("evidence_payload")
        return projected

    assert discrete_metadata(parallel) == discrete_metadata(serial)
    assert set(parallel_arrays) == set(serial_arrays)
    for key in sorted(serial_arrays):
        expected = serial_arrays[key]
        observed = parallel_arrays[key]
        serial_registration = serial["array_inventory"][key]
        parallel_registration = parallel["array_inventory"][key]
        for field in ("relative_path", "dtype", "shape", "size_bytes"):
            assert parallel_registration[field] == serial_registration[field]
        assert observed.dtype == expected.dtype
        assert observed.shape == expected.shape
        if np.issubdtype(expected.dtype, np.floating):
            assert np.isfinite(expected).all()
            assert np.isfinite(observed).all()
            np.testing.assert_allclose(
                observed,
                expected,
                rtol=config.replay_relative_tolerance,
                atol=config.replay_absolute_tolerance,
            )
        else:
            np.testing.assert_array_equal(observed, expected)

    serial_evidence = serial["evidence_payload"]
    parallel_evidence = parallel["evidence_payload"]
    assert {
        key: value
        for key, value in parallel_evidence.items()
        if key != "architecture_evidence"
    } == {
        key: value
        for key, value in serial_evidence.items()
        if key != "architecture_evidence"
    }
    serial_atoms = serial_evidence["architecture_evidence"]
    parallel_atoms = parallel_evidence["architecture_evidence"]
    assert len(parallel_atoms) == len(serial_atoms)
    for observed, expected in zip(
        parallel_atoms,
        serial_atoms,
        strict=True,
    ):
        assert {
            key: value for key, value in observed.items() if key != "attention"
        } == {
            key: value for key, value in expected.items() if key != "attention"
        }
        np.testing.assert_allclose(
            observed["attention"],
            expected["attention"],
            rtol=config.replay_relative_tolerance,
            atol=config.replay_absolute_tolerance,
        )

    exact_relative = (
        Path("logical_views")
        / f"{request.physical_owner.scope_id}.json"
    )
    serial_view = json.loads(
        (serial_root / exact_relative).read_text(encoding="utf-8")
    )
    parallel_view = json.loads(
        (parallel_root / exact_relative).read_text(encoding="utf-8")
    )
    for field in (
        "schema_version",
        "group_request_content_sha256",
        "logical_scope_id",
        "logical_scope_sha256",
        "logical_purpose",
        "physical_owner_scope_id",
        "family",
        "view_input_policy",
        "logical_heldout_row_ids",
        "logical_heldout_text_sha256",
        "coverage_proof",
        "registered_heldout_text_accessed",
        "registered_heldout_labels_accessed",
        "model_state_reloaded_for_primary_transform",
        "sealed_state_replay_checked",
    ):
        assert parallel_view[field] == serial_view[field]
    assert set(parallel_view["coverage_artifacts"]) == set(
        serial_view["coverage_artifacts"]
    )
    for key, serial_registration in serial_view[
        "coverage_artifacts"
    ].items():
        parallel_registration = parallel_view["coverage_artifacts"][key]
        assert parallel_registration == serial_registration
        with (
            serial_root / serial_registration["relative_path"]
        ).open("rb") as handle:
            serial_coverage = np.load(handle, allow_pickle=False)
        with (
            parallel_root / parallel_registration["relative_path"]
        ).open("rb") as handle:
            parallel_coverage = np.load(handle, allow_pickle=False)
        np.testing.assert_array_equal(parallel_coverage, serial_coverage)
    serial_prediction = serial_view["prediction_artifact"]
    parallel_prediction = parallel_view["prediction_artifact"]
    for field in ("dtype", "shape", "columns"):
        assert parallel_prediction[field] == serial_prediction[field]
    with (
        serial_root / serial_prediction["relative_path"]
    ).open("rb") as handle:
        serial_values = np.load(handle, allow_pickle=False)
    with (
        parallel_root / parallel_prediction["relative_path"]
    ).open("rb") as handle:
        parallel_values = np.load(handle, allow_pickle=False)
    np.testing.assert_allclose(
        parallel_values,
        serial_values,
        rtol=config.replay_relative_tolerance,
        atol=config.replay_absolute_tolerance,
    )


def test_request_scientific_identity_is_independent_of_device_assignments():
    cpu = _plan(gpu_ids=())
    gpu_zero = _plan(gpu_ids=(0,))
    heterogeneous = _plan(gpu_ids=(3, 9))

    assert len({cpu.content_sha256, gpu_zero.content_sha256, heterogeneous.content_sha256}) == 3
    assert {
        cpu.scientific_content_sha256,
        gpu_zero.scientific_content_sha256,
        heterogeneous.scientific_content_sha256,
    } == {cpu.scientific_content_sha256}

    requests = [_request(plan) for plan in (cpu, gpu_zero, heterogeneous)]
    assert {request.content_sha256 for request in requests} == {
        requests[0].content_sha256
    }
    assert {request.plan_scientific_content_sha256 for request in requests} == {
        cpu.scientific_content_sha256
    }
    assert all(
        "gpu" not in json.dumps(request.as_dict()).lower()
        for request in requests
    )


def test_complete_htr_configuration_roundtrips_without_source_defaults():
    config = _config()
    assert RoleNeutralHTRConfig.from_mapping(config.as_dict()) == config


def test_effect_fold_task_builder_preserves_legacy_production_semantics():
    from sklearn.model_selection import KFold

    config = replace(
        _config(),
        effect_folds=3,
        effect_objectives=("pseudo_outcome_mse", "squared_r_loss"),
        e_clip=0.1,
        r_stage_min_propensity=0.2,
        r_stage_max_propensity=0.8,
    ).validated()
    owner_scope_seed = 918_273
    treatment = np.asarray(
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        dtype=np.float64,
    )
    outcome = np.asarray(
        [0, 1, 1, 0, 0, 1, 1, 1, 0],
        dtype=np.float64,
    )
    nuisance_oof_e = np.asarray(
        [0.01, 0.2, 0.35, 0.5, 0.65, 0.8, 0.99, 0.4, 0.6],
        dtype=np.float64,
    )
    nuisance_oof_m = np.asarray(
        [0.1, 0.7, 0.4, 0.3, 0.2, 0.8, 0.6, 0.5, 0.25],
        dtype=np.float64,
    )
    controls = object()
    text_authority = object()

    plan = group_module._build_effect_fold_tasks(
        owner_scope_seed=owner_scope_seed,
        text_count=len(treatment),
        treatment=treatment,
        outcome=outcome,
        nuisance_oof_e=nuisance_oof_e,
        nuisance_oof_m=nuisance_oof_m,
        config=config,
        model_marker="authenticated-model-marker",
        operational_controls=controls,
        text_authority=text_authority,
    )

    # This is the production transformation that preceded the helper
    # extraction. Keep the oracle explicit so calibration and full execution
    # cannot silently diverge on clipping, residuals, or eligibility.
    expected_clipped_e = np.clip(
        nuisance_oof_e,
        config.e_clip,
        1.0 - config.e_clip,
    )
    expected_y_residual = outcome - nuisance_oof_m
    expected_t_residual = treatment - expected_clipped_e
    expected_pseudo_outcome = (
        expected_y_residual / expected_t_residual
    )
    expected_eligible = (
        (nuisance_oof_e >= config.r_stage_min_propensity)
        & (nuisance_oof_e <= config.r_stage_max_propensity)
        & np.isfinite(expected_pseudo_outcome)
    )

    np.testing.assert_array_equal(plan.clipped_e, expected_clipped_e)
    np.testing.assert_array_equal(plan.y_residual, expected_y_residual)
    np.testing.assert_array_equal(plan.t_residual, expected_t_residual)
    np.testing.assert_array_equal(
        plan.pseudo_outcome,
        expected_pseudo_outcome,
    )
    np.testing.assert_array_equal(plan.eligible, expected_eligible)
    assert plan.clipped_e.dtype == np.float64
    assert plan.y_residual.dtype == np.float64
    assert plan.t_residual.dtype == np.float64
    assert plan.pseudo_outcome.dtype == np.float64
    assert plan.eligible.dtype == np.bool_
    assert all(
        value.flags.c_contiguous
        for value in (
            plan.clipped_e,
            plan.y_residual,
            plan.t_residual,
            plan.pseudo_outcome,
            plan.eligible,
        )
    )

    expected_tasks = []
    all_positions = np.arange(len(treatment))
    for objective in config.effect_objectives:
        split_seed = group_module._derived_seed(
            owner_scope_seed,
            purpose="split",
            objective=objective,
            fold=0,
        )
        for fold, (fit_positions, validation_positions) in enumerate(
            KFold(
                n_splits=config.effect_folds,
                shuffle=True,
                random_state=split_seed,
            ).split(all_positions),
            start=1,
        ):
            fit_positions = np.asarray(fit_positions, dtype=np.int64)
            validation_positions = np.asarray(
                validation_positions,
                dtype=np.int64,
            )
            expected_tasks.append(
                (
                    objective,
                    fold,
                    split_seed,
                    group_module._derived_seed(
                        owner_scope_seed,
                        purpose="fit",
                        objective=objective,
                        fold=fold,
                    ),
                    fit_positions,
                    fit_positions[expected_eligible[fit_positions]],
                    validation_positions,
                )
            )

    assert len(plan.tasks) == (
        config.effect_folds * len(config.effect_objectives)
    )
    for task, expected in zip(plan.tasks, expected_tasks, strict=True):
        (
            objective,
            fold,
            split_seed,
            model_seed,
            fit_positions,
            eligible_fit_positions,
            validation_positions,
        ) = expected
        assert task.objective == objective
        assert task.fold == fold
        assert task.split_seed == split_seed
        assert task.model_seed == model_seed
        np.testing.assert_array_equal(task.fit_positions, fit_positions)
        np.testing.assert_array_equal(
            task.eligible_fit_positions,
            eligible_fit_positions,
        )
        np.testing.assert_array_equal(
            task.validation_positions,
            validation_positions,
        )
        np.testing.assert_array_equal(
            task.y_residual,
            expected_y_residual,
        )
        np.testing.assert_array_equal(
            task.t_residual,
            expected_t_residual,
        )
        np.testing.assert_array_equal(
            task.pseudo_outcome,
            expected_pseudo_outcome,
        )
        assert task.config is config
        assert task.model_marker == "authenticated-model-marker"
        assert task.operational_controls is controls
        assert task.text_authority is text_authority


def test_htr_operational_controls_fail_closed_for_scientific_or_idle_workers():
    config = _config()
    with pytest.raises(ValueError, match="optimizer training_batch_size"):
        RoleNeutralHTROperationalControls(
            training_batch_size=config.batch_size + 1,
            sentence_encoder_batch_size=16,
            data_loader_workers=0,
            fold_parallelism=1,
            fold_parallel_backend="threads",
            fold_slots_per_device=1,
            reuse_tokenizer_and_chunk_plans=False,
            chunk_plan_cache_max_entries=0,
            tokenized_chunk_cache_max_entries=0,
        ).validate_for(config)
    with pytest.raises(ValueError, match="require reusable plans"):
        RoleNeutralHTROperationalControls(
            training_batch_size=config.batch_size,
            sentence_encoder_batch_size=16,
            data_loader_workers=2,
            fold_parallelism=1,
            fold_parallel_backend="threads",
            fold_slots_per_device=1,
            reuse_tokenizer_and_chunk_plans=False,
            chunk_plan_cache_max_entries=0,
            tokenized_chunk_cache_max_entries=0,
        )
    missing = config.as_dict()
    del missing["max_chunks"]
    with pytest.raises(ValueError, match="missing=.*max_chunks"):
        RoleNeutralHTRConfig.from_mapping(missing)


def test_parallel_fold_scheduler_overlaps_leases_and_enforces_nuisance_barrier():
    controls = RoleNeutralHTROperationalControls(
        training_batch_size=_config().batch_size,
        sentence_encoder_batch_size=16,
        data_loader_workers=0,
        fold_parallelism=4,
        fold_parallel_backend="threads",
        fold_slots_per_device=2,
        reuse_tokenizer_and_chunk_plans=False,
        chunk_plan_cache_max_entries=0,
        tokenized_chunk_cache_max_entries=0,
    )
    resource_plan = controls.bind_fold_resources(
        devices=("cuda:3", "cuda:9"),
        owner_cpu_budget=4,
    )
    events: list[dict] = []
    active_lock = threading.Lock()
    active_total = 0
    active_by_device = {"cuda:3": 0, "cuda:9": 0}
    maximum_total = 0
    maximum_by_device = {"cuda:3": 0, "cuda:9": 0}
    first_wave_barriers = {
        "nuisance": threading.Barrier(4),
        "effect": threading.Barrier(4),
    }

    def fake_worker(task: dict, device: str) -> tuple[str, int, str]:
        nonlocal active_total, maximum_total
        stage = str(task["stage"])
        fold = int(task["fold"])
        with active_lock:
            active_total += 1
            active_by_device[device] += 1
            maximum_total = max(maximum_total, active_total)
            maximum_by_device[device] = max(
                maximum_by_device[device],
                active_by_device[device],
            )
        try:
            if fold <= 4:
                first_wave_barriers[stage].wait(timeout=2.0)
            time.sleep(0.03)
            return stage, fold, device
        finally:
            with active_lock:
                active_total -= 1
                active_by_device[device] -= 1

    nuisance_tasks = tuple(
        {
            "stage": "nuisance",
            "objective": "joint_treatment_outcome_nuisance",
            "fold": fold,
        }
        for fold in range(1, 6)
    )
    effect_tasks = tuple(
        {
            "stage": "effect",
            "objective": "squared_r_loss",
            "fold": fold,
        }
        for fold in range(1, 6)
    )
    nuisance_values = _execute_htr_fold_tasks(
        nuisance_tasks,
        resource_plan=resource_plan,
        worker=fake_worker,
        stage="nuisance",
        event_sink=lambda value: events.append(dict(value)),
    )
    barrier_monotonic_ns = time.monotonic_ns()
    effect_values = _execute_htr_fold_tasks(
        effect_tasks,
        resource_plan=resource_plan,
        worker=fake_worker,
        stage="effect",
        event_sink=lambda value: events.append(dict(value)),
    )

    assert [value[1] for value in nuisance_values] == list(range(1, 6))
    assert [value[1] for value in effect_values] == list(range(1, 6))
    assert maximum_total == resource_plan.fold_parallelism
    assert maximum_by_device == {
        device: resource_plan.fold_slots_per_device
        for device in resource_plan.devices
    }
    assert set(value[2] for value in nuisance_values) == set(
        resource_plan.devices
    )
    assert set(value[2] for value in effect_values) == set(
        resource_plan.devices
    )

    def intervals(stage: str) -> dict[int, tuple[int, int, str]]:
        by_fold: dict[int, dict[str, dict]] = {}
        for event in events:
            if event["stage"] == stage:
                by_fold.setdefault(int(event["fold"]), {})[
                    str(event["event"])
                ] = event
        return {
            fold: (
                values["fold_started"]["monotonic_ns"],
                values["fold_finished"]["monotonic_ns"],
                values["fold_started"]["device"],
            )
            for fold, values in by_fold.items()
        }

    nuisance_intervals = intervals("nuisance")
    effect_intervals = intervals("effect")
    assert len(nuisance_intervals) == len(effect_intervals) == 5
    assert max(value[1] for value in nuisance_intervals.values()) < (
        barrier_monotonic_ns
    )
    assert barrier_monotonic_ns < min(
        value[0] for value in effect_intervals.values()
    )
    for stage_intervals in (nuisance_intervals, effect_intervals):
        first, third = stage_intervals[1], stage_intervals[3]
        assert first[2] == third[2] == "cuda:3"
        assert max(first[0], third[0]) < min(first[1], third[1])
        second, fourth = stage_intervals[2], stage_intervals[4]
        assert second[2] == fourth[2] == "cuda:9"
        assert max(second[0], fourth[0]) < min(second[1], fourth[1])


def test_process_fold_invocation_enforces_and_reobserves_torch_determinism(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    observed = {
        **stage1_torch_determinism_policy(),
        "torch_available": False,
        "policy_active": True,
    }

    def enforce() -> dict:
        calls.append("enforce")
        return dict(observed)

    def observe() -> dict:
        calls.append("observe")
        return dict(observed)

    monkeypatch.setattr(
        group_module,
        "_enforce_stage1_torch_determinism",
        enforce,
    )
    monkeypatch.setattr(
        group_module,
        "_observe_stage1_torch_determinism",
        observe,
    )
    monkeypatch.setattr(group_module.torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(
        group_module.torch,
        "get_num_interop_threads",
        lambda: 1,
    )
    completed = group_module._invoke_htr_fold_worker(
        lambda task, device: (task, device),
        {"fold": 1},
        "cpu",
        worker_cpu_threads=1,
        process_isolated=True,
    )

    assert calls == ["enforce", "observe"]
    assert completed.torch_determinism_observed == observed


def test_serial_and_parallel_htr_preserve_science_and_complete_text(
    tmp_path: Path,
):
    request = _request()
    config = _config()
    fit_texts, treatment, outcome, heldout_texts = _inputs(request)
    serial_root = (tmp_path / "serial").resolve()
    parallel_root = (tmp_path / "parallel").resolve()

    serial_terminal = execute_role_neutral_htr_physical_group(
        request=request,
        output_root=serial_root,
        fit_texts=fit_texts,
        fit_treatment=treatment,
        fit_outcome=outcome,
        config=config,
        runtime_compatibility_class="torch-cpu-float32-v1",
        exact_heldout_text_loader=lambda rows: (
            heldout_texts
            if rows == request.physical_owner.heldout_row_ids
            else (_ for _ in ()).throw(AssertionError("unexpected held-out rows"))
        ),
        device="cpu",
    )

    controls = RoleNeutralHTROperationalControls(
        training_batch_size=config.batch_size,
        sentence_encoder_batch_size=config.sentence_encoder_batch_size,
        data_loader_workers=0,
        fold_parallelism=2,
        fold_parallel_backend="processes",
        fold_slots_per_device=2,
        reuse_tokenizer_and_chunk_plans=True,
        chunk_plan_cache_max_entries=len(fit_texts),
        tokenized_chunk_cache_max_entries=len(fit_texts) * config.max_chunks,
    )
    resource_plan = controls.bind_fold_resources(
        devices=("cpu",),
        owner_cpu_budget=2,
    )
    attestations: list[dict] = []
    fold_events: list[dict] = []
    parallel_terminal = execute_role_neutral_htr_physical_group(
        request=request,
        output_root=parallel_root,
        fit_texts=fit_texts,
        fit_treatment=treatment,
        fit_outcome=outcome,
        config=config,
        runtime_compatibility_class="torch-cpu-float32-v1",
        exact_heldout_text_loader=lambda rows: (
            heldout_texts
            if rows == request.physical_owner.heldout_row_ids
            else (_ for _ in ()).throw(AssertionError("unexpected held-out rows"))
        ),
        device="cpu",
        operational_controls=controls,
        fold_resource_plan=resource_plan,
        operational_attestation_sink=lambda value: attestations.append(
            dict(value)
        ),
        fold_event_sink=lambda value: fold_events.append(dict(value)),
    )

    assert serial_terminal["text_truncation_applied"] is False
    assert parallel_terminal["text_truncation_applied"] is False
    assert len(attestations) == 1
    attestation = attestations[0]
    assert attestation["controls"] == controls.as_dict()
    assert attestation["fold_resource_plan"] == resource_plan.as_dict()
    assert (
        attestation["complete_owner_tokenizer_chunk_plan_built_once"]
        is True
    )
    assert (
        attestation["process_reusable_plan"][
            "complete_owner_tokenizer_chunk_plan_built_once"
        ]
        is True
    )
    assert (
        attestation["process_reusable_plan"]["fold_workers_retokenized"]
        is False
    )
    assert (
        attestation["process_reusable_plan"]["fold_workers_rechunked"]
        is False
    )
    assert attestation["process_reusable_plan"][
        "semantic_truncation_applied"
    ] is False
    assert (
        attestation[
            "temporary_process_plan_removed_before_artifact_publication"
        ]
        is True
    )
    assert (
        attestation[
            "raw_text_persisted_in_temporary_process_plan_after_folds"
        ]
        is False
    )
    assert (
        attestation["shared_mutable_array_store_used_by_fold_workers"]
        is False
    )
    assert attestation["fit_reusable_plan"]["note_count"] == len(fit_texts)
    assert (
        attestation["fit_reusable_plan"]["semantic_truncation_applied"]
        is False
    )
    assert (
        attestation["fold_execution"]["nuisance_barrier_enforced"]
        is True
    )
    barrier = next(
        event
        for event in fold_events
        if event["event"] == "nuisance_barrier_completed"
    )
    nuisance_finishes = [
        event["monotonic_ns"]
        for event in fold_events
        if event["stage"] == "nuisance"
        and event["event"] == "fold_finished"
    ]
    effect_starts = [
        event["monotonic_ns"]
        for event in fold_events
        if event["stage"] == "effect"
        and event["event"] == "fold_started"
    ]
    assert max(nuisance_finishes) <= barrier["monotonic_ns"]
    assert barrier["monotonic_ns"] <= min(effect_starts)

    _assert_serial_parallel_science_equal(
        serial_root=serial_root,
        parallel_root=parallel_root,
        request=request,
        config=config,
    )
    serial_metadata, _serial_arrays = _load_fit_artifact(serial_root)
    parallel_metadata, _parallel_arrays = _load_fit_artifact(parallel_root)
    for metadata in (serial_metadata, parallel_metadata):
        assert metadata["fit_coverage"]["max_chunks_nonbinding"] is True
        assert (
            metadata["fit_coverage"]["semantic_truncation_applied"]
            is False
        )
        assert any(
            "sentinel_after_former_page_boundary" in atom["chunk_text"]
            for atom in metadata["evidence_payload"]["architecture_evidence"]
        )
    assert len(fit_texts[0]) > 14_000


def test_typed_htr_rejects_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCI_HTR_ENCODER_BATCH_SIZE", "1")
    with pytest.raises(RuntimeError, match="forbids environment overrides"):
        from oci.inference.role_neutral_htr_group_execution import _new_extractor

        _new_extractor(
            config=_config(),
            model_marker="hash",
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("owner_kind", ("full_outer", "cumulative_spent"))
def test_singleton_owner_gets_primary_heldout_numerical_transform(
    tmp_path: Path,
    owner_kind: str,
):
    request = _singleton_request(owner_kind)
    fit_texts, treatment, outcome, heldout_texts = _inputs(request)
    root = (tmp_path / owner_kind).resolve()
    calls: list[tuple[int, ...]] = []

    def loader(row_ids: tuple[int, ...]):
        calls.append(row_ids)
        return heldout_texts

    terminal = execute_role_neutral_htr_physical_group(
        request=request,
        output_root=root,
        fit_texts=fit_texts,
        fit_treatment=treatment,
        fit_outcome=outcome,
        config=_config(),
        runtime_compatibility_class="torch-cpu-float32-v1",
        exact_heldout_text_loader=loader,
        device="cpu",
    )

    assert calls == [request.physical_owner.heldout_row_ids]
    assert terminal["registered_heldout_labels_accessed"] is False
    assert all(
        event["event"] != "cumulative_fit_only_view_published"
        for event in terminal["event_order"]
    )
    view = json.loads(
        (
            root
            / "logical_views"
            / f"{request.physical_owner.scope_id}.json"
        ).read_text(encoding="utf-8")
    )
    assert view["logical_purpose"] == owner_kind
    assert view["logical_transform_performed"] is True
    assert view["registered_heldout_labels_accessed"] is False
    with (root / view["prediction_artifact"]["relative_path"]).open(
        "rb"
    ) as handle:
        values = np.load(handle, allow_pickle=False)
    assert values.shape[0] == len(request.physical_owner.heldout_row_ids)
    assert np.isfinite(values).all()


@pytest.fixture(scope="module")
def completed_htr_group(tmp_path_factory):
    request = _request()
    config = _config()
    fit_texts, treatment, outcome, heldout_texts = _inputs(request)
    root = (tmp_path_factory.mktemp("role_neutral_htr") / "artifact").resolve()
    calls: list[tuple[int, ...]] = []
    cumulative = request.logical_members[1]

    def heldout_loader(row_ids: tuple[int, ...]):
        # The loader is the first possible entry point for exact held-out text.
        assert (root / "fit_only_family_seal.json").is_file()
        cumulative_path = root / "logical_views" / f"{cumulative.scope_id}.json"
        assert cumulative_path.is_file()
        cumulative_view = json.loads(cumulative_path.read_text(encoding="utf-8"))
        assert cumulative_view["registered_heldout_text_accessed"] is False
        assert cumulative_view["prediction_artifact"] is None
        calls.append(row_ids)
        return heldout_texts

    terminal = execute_role_neutral_htr_physical_group(
        request=request,
        output_root=root,
        fit_texts=fit_texts,
        fit_treatment=treatment,
        fit_outcome=outcome,
        config=config,
        runtime_compatibility_class="torch-cpu-float32-v1",
        exact_heldout_text_loader=heldout_loader,
        device="cpu",
    )
    assert calls == [request.physical_owner.heldout_row_ids]
    return {
        "root": root,
        "request": request,
        "config": config,
        "fit_texts": fit_texts,
        "treatment": treatment,
        "outcome": outcome,
        "heldout_texts": heldout_texts,
        "terminal": terminal,
    }


def test_htr_npy_authentication_uses_one_bounded_in_memory_read(
    tmp_path: Path,
    monkeypatch,
):
    target = tmp_path / "state.npy"
    expected = np.arange(24, dtype=np.float64).reshape(4, 6)
    with target.open("xb") as stream:
        np.save(stream, expected, allow_pickle=False)

    original_load = np.load
    observed: list[tuple[object, dict]] = []

    def guarded_load(source, *args, **kwargs):
        observed.append((source, dict(kwargs)))
        assert hasattr(source, "read")
        assert "mmap_mode" not in kwargs
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(group_module.np, "load", guarded_load)
    digest, size, loaded = group_module._read_npy_file_once(
        target,
        label="test HTR array",
        invalid_message="test HTR array is invalid",
    )

    assert observed and len(observed) == 1
    assert observed[0][1] == {"allow_pickle": False}
    assert digest == group_module._sha256_file(target)[0]
    assert size == target.stat().st_size
    assert isinstance(loaded, np.ndarray)
    assert not isinstance(loaded, np.memmap)
    np.testing.assert_array_equal(loaded, expected)


def test_operational_encoder_batch_workers_and_complete_plan_reuse_are_exact(
    completed_htr_group,
    tmp_path: Path,
):
    request = completed_htr_group["request"]
    config = completed_htr_group["config"]
    fit_texts = completed_htr_group["fit_texts"]
    heldout_texts = completed_htr_group["heldout_texts"]
    controls = RoleNeutralHTROperationalControls(
        training_batch_size=config.batch_size,
        sentence_encoder_batch_size=16,
        data_loader_workers=2,
        fold_parallelism=1,
        fold_parallel_backend="threads",
        fold_slots_per_device=1,
        reuse_tokenizer_and_chunk_plans=True,
        chunk_plan_cache_max_entries=len(fit_texts),
        tokenized_chunk_cache_max_entries=(
            len(fit_texts) * config.max_chunks
        ),
    )
    attestations: list[dict] = []
    root = (tmp_path / "operational-htr").resolve()

    terminal = execute_role_neutral_htr_physical_group(
        request=request,
        output_root=root,
        fit_texts=fit_texts,
        fit_treatment=completed_htr_group["treatment"],
        fit_outcome=completed_htr_group["outcome"],
        config=config,
        runtime_compatibility_class="torch-cpu-float32-v1",
        exact_heldout_text_loader=lambda row_ids: (
            heldout_texts
            if row_ids == request.physical_owner.heldout_row_ids
            else (_ for _ in ()).throw(AssertionError("unexpected held-out rows"))
        ),
        device="cpu",
        operational_controls=controls,
        fold_resource_plan=controls.bind_fold_resources(
            devices=("cpu",),
            owner_cpu_budget=1,
        ),
        operational_attestation_sink=lambda value: attestations.append(
            dict(value)
        ),
    )

    assert terminal["content_sha256"] == completed_htr_group["terminal"][
        "content_sha256"
    ]
    assert len(attestations) == 1
    attestation = attestations[0]
    assert attestation["controls"] == controls.as_dict()
    assert attestation["training_batch_override_applied"] is False
    assert attestation["cache_capacities_nonbinding"] is True
    assert attestation["positive_data_loader_workers_exercised"] is True
    assert (
        attestation[
            "operational_predictions_within_declared_tolerance_of_scientific_replay"
        ]
        is True
    )
    assert attestation["replay_relative_tolerance"] == 1e-4
    assert attestation["replay_absolute_tolerance"] == 1e-5
    for plan_name in (
        "fit_reusable_plan",
        "exact_heldout_reusable_plan",
    ):
        plan = attestation[plan_name]
        assert plan["positive_data_loader_workers_exercised"] is True
        assert plan["parallel_plan_task_count"] == plan["note_count"]
        assert 1 <= plan["parallel_plan_thread_count"] <= 2
        assert (
            plan["unique_note_count"]
            <= controls.chunk_plan_cache_max_entries
        )
        assert (
            plan["unique_chunk_count"]
            <= controls.tokenized_chunk_cache_max_entries
        )
        assert plan["semantic_truncation_applied"] is False
        assert plan["raw_text_persisted"] is False
    assert "sentinel_after_former_page_boundary" not in json.dumps(
        attestation,
        sort_keys=True,
    )


def test_fit_seals_before_loader_and_fresh_json_npy_replay(completed_htr_group):
    root = completed_htr_group["root"]
    request = completed_htr_group["request"]
    heldout_texts = completed_htr_group["heldout_texts"]
    terminal = completed_htr_group["terminal"]

    assert (
        validate_role_neutral_htr_group_execution(
            root=root,
            request=request,
            device="cpu",
        )
        == terminal
    )
    event_names = [event["event"] for event in terminal["event_order"]]
    assert event_names[:2] == ["fit_completed", "fit_family_artifact_sealed"]
    assert event_names.index("cumulative_fit_only_view_published") < (
        event_names.index("exact_heldout_text_opened")
    )
    assert terminal["model_state_reloaded_for_primary_transform"] is True
    assert terminal["registered_heldout_labels_accessed"] is False
    assert terminal["text_truncation_applied"] is False

    metadata = json.loads(
        (root / "fit_state" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["fit_coverage"]["max_chunks_nonbinding"] is True
    assert metadata["fit_coverage"]["semantic_truncation_applied"] is False
    assert any(
        "sentinel_after_former_page_boundary" in atom["chunk_text"]
        for atom in metadata["evidence_payload"]["architecture_evidence"]
    )
    assert metadata["plan_scientific_content_sha256"] == (
        request.plan_scientific_content_sha256
    )
    receipt = authenticate_role_neutral_htr_component(
        root=root,
        plan=_plan(),
        physical_owner_scope_id=request.physical_owner.scope_id,
        device="cpu",
    )
    assert tuple(receipt.family_fit_seals) == ("htr_neural",)
    assert receipt.text_truncation_applied is False

    replay = replay_role_neutral_htr_exact_transform(
        root=root,
        request=request,
        exact_heldout_texts=heldout_texts,
        device="cpu",
    )
    exact_view = json.loads(
        (
            root
            / "logical_views"
            / f"{request.physical_owner.scope_id}.json"
        ).read_text(encoding="utf-8")
    )
    with (
        root / exact_view["prediction_artifact"]["relative_path"]
    ).open("rb") as handle:
        registered = np.load(handle, allow_pickle=False)
    assert replay["state_source"] == "authenticated_json_and_per_array_npy_only"
    assert replay["allow_pickle"] is False
    assert replay["heldout_labels_accessed"] is False
    np.testing.assert_allclose(
        replay["predictions"],
        registered,
        rtol=completed_htr_group["config"].replay_relative_tolerance,
        atol=completed_htr_group["config"].replay_absolute_tolerance,
    )

    assert not tuple(root.rglob("*.npz"))
    assert not tuple(root.rglob("*.pkl"))
    assert not tuple(root.rglob("*.pickle"))
    assert not tuple(root.rglob("*.joblib"))
    assert all(path.suffix in {".json", ".npy"} for path in root.rglob("*") if path.is_file())


def test_fresh_htr_replay_authenticates_bytes_then_uses_declared_tolerance(
    completed_htr_group,
    monkeypatch,
):
    original = group_module._predict_from_state
    configured = completed_htr_group["config"]
    drift = {"value": configured.replay_absolute_tolerance / 2.0}

    def replay_with_declared_drift(**kwargs):
        columns, values = original(**kwargs)
        return columns, values + drift["value"]

    monkeypatch.setattr(
        group_module,
        "_predict_from_state",
        replay_with_declared_drift,
    )
    replay = replay_role_neutral_htr_exact_transform(
        root=completed_htr_group["root"],
        request=completed_htr_group["request"],
        exact_heldout_texts=completed_htr_group["heldout_texts"],
        device="cpu",
    )
    assert replay["predictions"].shape[0] == len(
        completed_htr_group["request"].physical_owner.heldout_row_ids
    )

    drift["value"] = (
        configured.replay_absolute_tolerance
        + configured.replay_relative_tolerance
        + 1.0
    )
    with pytest.raises(RuntimeError, match="declared tolerance"):
        replay_role_neutral_htr_exact_transform(
            root=completed_htr_group["root"],
            request=completed_htr_group["request"],
            exact_heldout_texts=completed_htr_group["heldout_texts"],
            device="cpu",
        )


def test_binding_chunk_capacity_fails_before_fit_or_loader(tmp_path: Path):
    request = _request()
    fit_texts, treatment, outcome, _heldout_texts = _inputs(request)
    config = replace(_config(), max_chunks=1)
    calls = 0
    root = (tmp_path / "binding_capacity").resolve()

    def loader(_rows):
        nonlocal calls
        calls += 1
        raise AssertionError("held-out loader must not run")

    with pytest.raises(ValueError, match="would truncate"):
        execute_role_neutral_htr_physical_group(
            request=request,
            output_root=root,
            fit_texts=fit_texts,
            fit_treatment=treatment,
            fit_outcome=outcome,
            config=config,
            runtime_compatibility_class="torch-cpu-float32-v1",
            exact_heldout_text_loader=loader,
            device="cpu",
        )
    assert calls == 0
    assert not root.exists()


def _copied_group(completed_htr_group, tmp_path: Path) -> tuple[Path, object]:
    copied = (tmp_path / "copied").resolve()
    shutil.copytree(completed_htr_group["root"], copied)
    return copied, completed_htr_group["request"]


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        (np.float32, (1,)),
        (np.int16, (2, 3)),
    ],
)
def test_fresh_validation_rejects_tampered_array_dtype_or_shape(
    completed_htr_group,
    tmp_path: Path,
    dtype,
    shape,
):
    root, request = _copied_group(completed_htr_group, tmp_path)
    target = next((root / "fit_state" / "arrays").glob("*.npy"))
    with target.open("wb") as handle:
        np.save(handle, np.zeros(shape, dtype=dtype), allow_pickle=False)

    with pytest.raises((ValueError, RuntimeError), match="array|fit"):
        validate_role_neutral_htr_group_execution(
            root=root,
            request=request,
            device="cpu",
        )


@pytest.mark.parametrize("link_kind", ["symbolic", "hard"])
def test_fresh_validation_rejects_linked_artifacts(
    completed_htr_group,
    tmp_path: Path,
    link_kind: str,
):
    root, request = _copied_group(completed_htr_group, tmp_path)
    arrays = sorted((root / "fit_state" / "arrays").glob("*.npy"))
    target, peer = arrays[:2]
    target.unlink()
    if link_kind == "symbolic":
        target.symlink_to(peer.name)
    else:
        os.link(peer, target)

    with pytest.raises((ValueError, RuntimeError), match="link|artifact|regular"):
        validate_role_neutral_htr_group_execution(
            root=root,
            request=request,
            device="cpu",
        )


def test_replay_rejects_reordered_or_substituted_exact_text(
    completed_htr_group,
):
    request = completed_htr_group["request"]
    heldout = completed_htr_group["heldout_texts"]

    with pytest.raises(ValueError, match="differ"):
        replay_role_neutral_htr_exact_transform(
            root=completed_htr_group["root"],
            request=request,
            exact_heldout_texts=tuple(reversed(heldout)),
            device="cpu",
        )
