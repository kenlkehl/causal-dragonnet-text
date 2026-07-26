from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from oci.config import BoWViewConfig
from oci.inference.all_evidence_discovery_interfaces import MATCHED_PAIR_UPLIFT
from oci.inference.production_stage1_scope_scheduler import (
    build_canonical_stage1_scope_plan,
)
from tests.stage1_test_support import PHYSICAL_FIT_IDENTITY
from oci.inference.role_neutral_bow_group_execution import (
    RoleNeutralBoWPhysicalGroupRequest,
    execute_role_neutral_bow_physical_group,
    load_authenticated_role_neutral_bow_nuisance_bank,
)
from oci.inference.role_neutral_matched_pair_group_execution import (
    RoleNeutralMatchedPairConfig,
    RoleNeutralMatchedPairExactInput,
    RoleNeutralMatchedPairPhysicalGroupRequest,
    _assert_text_capacity,
    execute_role_neutral_matched_pair_from_bow_nuisance_bank,
    execute_role_neutral_matched_pair_physical_group,
    replay_role_neutral_matched_pair_exact_transform,
    validate_role_neutral_matched_pair_group_execution,
)
from oci.inference.stage1_htr_operational_controls import (
    RoleNeutralHTRFoldResourcePlan,
)
from oci.inference.role_neutral_all_ten_binding import (
    authenticate_role_neutral_matched_pair_component,
)
from oci.models.hierarchical_transformer_extractor import (
    HierarchicalTransformerExtractor,
)


def _registry() -> dict:
    row_count = 80
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
                    for inner_fold, inner_heldout in enumerate(partitions, start=1)
                ],
            }
        )
    return {"dataset_row_count": row_count, "outer_folds": outer_rows}


def _plan(gpu_ids=()):
    return build_canonical_stage1_scope_plan(
        registry=_registry(),
        registry_content_sha256="a" * 64,
        global_seed=42,
        physical_fit_identity=PHYSICAL_FIT_IDENTITY,
        gpu_ids=gpu_ids,
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=2,
        expected_inner_fold_count=5,
    )


def _request(plan=None) -> RoleNeutralMatchedPairPhysicalGroupRequest:
    plan = _plan() if plan is None else plan
    owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if len(members) > 1
    )
    assert owner.scope_kind == "exact_inner"
    assert members[1].scope_kind == "cumulative_spent"
    return RoleNeutralMatchedPairPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
        htr_model_identity_sha256="b" * 64,
        nuisance_artifact_identity_sha256="c" * 64,
        runtime_compatibility_class="torch-cpu-float32-v1",
    )


def _singleton_request(
    owner_kind: str,
) -> RoleNeutralMatchedPairPhysicalGroupRequest:
    plan = _plan()
    owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if owner.scope_kind == owner_kind and len(members) == 1
    )
    assert members == (owner,)
    return RoleNeutralMatchedPairPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
        htr_model_identity_sha256="b" * 64,
        nuisance_artifact_identity_sha256="c" * 64,
        runtime_compatibility_class="torch-cpu-float32-v1",
    )


def _extractor_configuration(
    *,
    chunk_size_words: int = 16,
    max_chunks: int = 8,
) -> dict:
    return {
        "sentence_encoder_model": "hash",
        "freeze_sentence_encoder": True,
        "chunk_size_words": chunk_size_words,
        "chunk_overlap_words": 0,
        "max_chunks": max_chunks,
        "max_chunk_length": 32,
        "num_transformer_layers": 1,
        "num_attention_heads": 1,
        "transformer_dim": 8,
        "transformer_dropout": 0.0,
        "projection_dim": 8,
        "hash_embedding_dim": 8,
        "sentence_encoder_batch_size": 8,
        "sentence_encoder_backend": "auto",
        "sentence_pooling": "auto",
        "normalize_sentence_embeddings": True,
        "trainable_sentence_encoder_layers": 0,
        "role_attention": False,
        "w_attention_heads": 1,
        "x_attention_heads": 1,
        "transformer_feedforward_dim": 13,
        "transformer_activation": "silu",
        "transformer_norm_style": "pre_norm",
        "transformer_layer_norm_eps": 1e-5,
        "transformer_layer_norm_elementwise_affine": True,
        "transformer_layer_norm_bias": True,
        "transformer_attention_dropout": 0.0,
        "transformer_residual_dropout": 0.0,
        "transformer_feedforward_dropout": 0.0,
        "transformer_attention_bias": True,
        "transformer_feedforward_bias": True,
        "output_projection_depth": 2,
        "output_projection_hidden_dim": 7,
        "output_projection_activation": "tanh",
        "output_projection_dropout": 0.0,
        "output_projection_hidden_layer_norm": True,
        "output_projection_final_layer_norm": True,
        "output_projection_bias": True,
        "pool_token_init_std": 0.02,
        "positional_encoding_base": 10_000.0,
        "environment_override_policy": "forbid",
    }


def test_overlap_capacity_uses_complete_first_chunk_plus_strides() -> None:
    configuration = _extractor_configuration(chunk_size_words=4, max_chunks=2)
    configuration["chunk_overlap_words"] = 2

    _assert_text_capacity(
        ("one two three four five six",),
        extractor_config=configuration,
        stage="test",
    )
    with pytest.raises(ValueError, match=r"requires 7 HTR words.*capacity is 6"):
        _assert_text_capacity(
            ("one two three four five six seven",),
            extractor_config=configuration,
            stage="test",
        )


def _config(**overrides) -> RoleNeutralMatchedPairConfig:
    values = {
        "effect_folds": 2,
        "propensity_caliper": 1.0,
        "outcome_caliper": 1.0,
        "max_controls_per_candidate": 2,
        "nearest_fallback_controls": 1,
        "bow_l2_alpha": 0.1,
        "bow_max_iter": 50,
        "bow_optimizer_method": "L-BFGS-B",
        "bow_optimizer_ftol": 3e-9,
        "bow_optimizer_gtol": 2e-6,
        "bow_optimizer_maxls": 17,
        "bow_optimizer_maxcor": 7,
        "bow_optimizer_maxfun": 12_345,
        "bow_optimizer_tol": None,
        "bow_optimizer_initialization": "zeros",
        "bow_require_optimizer_success": False,
        "htr_epochs": 1,
        "htr_batch_size": 8,
        "htr_learning_rate": 0.002,
        "htr_weight_decay": 0.0,
        "htr_optimizer_name": "adamw",
        "htr_adamw_beta1": 0.85,
        "htr_adamw_beta2": 0.97,
        "htr_adamw_eps": 1e-7,
        "htr_adamw_amsgrad": True,
        "htr_adamw_maximize": False,
        "htr_adamw_foreach": False,
        "htr_adamw_capturable": False,
        "htr_adamw_differentiable": False,
        "htr_adamw_fused": False,
        "htr_optimizer_zero_grad_set_to_none": True,
        "htr_gradient_clip_norm": 1.0,
        "htr_gradient_clip_norm_type": 2.0,
        "htr_gradient_clip_error_if_nonfinite": True,
        "htr_gradient_clip_foreach": False,
        "htr_hidden_dim": 8,
        "htr_dropout": 0.0,
        "htr_head_depth": 3,
        "htr_head_activation": "gelu_tanh",
        "htr_head_layer_norm": True,
        "htr_head_bias": True,
        "htr_extractor": _extractor_configuration(),
        "replay_comparison_policy": "allclose_and_exact_discrete_state_v1",
        "replay_relative_tolerance": 1e-4,
        "replay_absolute_tolerance": 1e-5,
    }
    values.update(overrides)
    return RoleNeutralMatchedPairConfig(**values)


def _view() -> BoWViewConfig:
    return BoWViewConfig(
        name="configured_unigram",
        max_features=5000,
        min_df=1,
        max_df=1.0,
        ngram_range_min=1,
        ngram_range_max=1,
        sublinear_tf=True,
        bow_model="linear",
        logistic_c=0.75,
        logistic_max_iter=500,
        ridge_alpha=2.0,
    )


def _inputs(request: RoleNeutralMatchedPairPhysicalGroupRequest):
    fit_rows = request.physical_owner.fit_row_ids
    texts = [
        (
            f"patient_{row_id} biomarker_{position % 4} "
            f"response_{(position // 2) % 2} complete_note"
        )
        for position, row_id in enumerate(fit_rows)
    ]
    # More than 14,000 characters but only two HTR words. This simultaneously
    # proves that no character boundary clips BoW input and keeps configured
    # HTR word capacity non-binding.
    texts[1] = ("z" * 14500) + " sentinelafterfourteenthousand"
    treatment = np.asarray([position % 2 for position in range(len(texts))], dtype=float)
    outcome = np.asarray(
        [
            ((position // 2) % 2 if treatment[position] == 1 else position % 2)
            for position in range(len(texts))
        ],
        dtype=float,
    )
    propensity = np.linspace(0.35, 0.65, len(texts), dtype=float)
    outcome_nuisance = np.linspace(0.30, 0.70, len(texts), dtype=float)
    heldout_rows = request.physical_owner.heldout_row_ids
    exact = RoleNeutralMatchedPairExactInput(
        row_ids=heldout_rows,
        texts=tuple(
            f"heldout_{row_id} full_exact_note biomarker_{position % 3}"
            for position, row_id in enumerate(heldout_rows)
        ),
        propensity_probability=tuple(
            np.linspace(0.40, 0.60, len(heldout_rows), dtype=float)
        ),
        outcome_nuisance_probability=tuple(
            np.linspace(0.38, 0.62, len(heldout_rows), dtype=float)
        ),
    )
    return (
        tuple(texts),
        treatment,
        outcome,
        propensity,
        outcome_nuisance,
        exact,
    )


def _factory(config: RoleNeutralMatchedPairConfig):
    constructor = config.as_dict()["htr_extractor"]

    def create(device: torch.device):
        return HierarchicalTransformerExtractor(
            **constructor,
            device=device,
        )

    return create


def _execute(
    *,
    root: Path,
    request: RoleNeutralMatchedPairPhysicalGroupRequest,
    loader,
    config: RoleNeutralMatchedPairConfig | None = None,
    fold_resource_plan: RoleNeutralHTRFoldResourcePlan | None = None,
    operational_attestation_sink=None,
    fold_event_sink=None,
):
    config = _config() if config is None else config
    texts, treatment, outcome, propensity, outcome_nuisance, _exact = _inputs(request)
    return execute_role_neutral_matched_pair_physical_group(
        request=request,
        output_root=root,
        fit_texts=texts,
        fit_treatment=treatment,
        fit_outcome=outcome,
        fit_propensity_probability=propensity,
        fit_outcome_nuisance_probability=outcome_nuisance,
        view_configs=(_view(),),
        config=config,
        htr_extractor_factory=_factory(config),
        exact_heldout_input_loader=loader,
        device="cpu",
        fold_resource_plan=fold_resource_plan,
        operational_attestation_sink=operational_attestation_sink,
        fold_event_sink=fold_event_sink,
    )


@pytest.fixture(scope="module")
def completed_execution(tmp_path_factory):
    work = tmp_path_factory.mktemp("role_neutral_matched_pair")
    request = _request()
    texts, _t, _y, _e, _m, exact = _inputs(request)
    root = (work / "execution").resolve()
    calls: list[tuple[int, ...]] = []
    cumulative = request.logical_members[1]

    def loader(row_ids: tuple[int, ...]):
        # Both native subproducers and the reference-only cumulative view must
        # be durable before the first exact text can enter worker memory.
        seal_path = root / "fit_only_family_seal.json"
        cumulative_path = root / "logical_views" / f"{cumulative.scope_id}.json"
        assert seal_path.is_file()
        assert cumulative_path.is_file()
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        assert seal["subproducer_coverage"] == ["bow", "htr"]
        assert [row["subproducer"] for row in seal["subproducer_proofs"]] == [
            "bow",
            "htr",
        ]
        view = json.loads(cumulative_path.read_text(encoding="utf-8"))
        assert view["logical_transform_performed"] is False
        assert view["prediction_artifacts"] is None
        assert view["registered_heldout_text_accessed"] is False
        calls.append(row_ids)
        return exact

    terminal = _execute(root=root, request=request, loader=loader)
    assert calls == [request.physical_owner.heldout_row_ids]
    return {
        "work": work,
        "root": root,
        "request": request,
        "fit_texts": texts,
        "exact": exact,
        "terminal": terminal,
    }


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("owner_kind", ("full_outer", "cumulative_spent"))
def test_singleton_owner_gets_both_primary_heldout_numerical_transforms(
    tmp_path: Path,
    owner_kind: str,
):
    request = _singleton_request(owner_kind)
    *_fit_inputs, exact = _inputs(request)
    root = (tmp_path / owner_kind).resolve()
    calls: list[tuple[int, ...]] = []

    def loader(row_ids: tuple[int, ...]):
        calls.append(row_ids)
        return exact

    terminal = _execute(root=root, request=request, loader=loader)

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
    assert view["exact_input_identity"][
        "heldout_treatment_field_present"
    ] is False
    assert view["exact_input_identity"][
        "heldout_outcome_field_present"
    ] is False
    assert set(view["prediction_artifacts"]) == {"bow", "htr"}
    for registration in view["prediction_artifacts"].values():
        with (root / registration["relative_path"]).open("rb") as handle:
            values = np.load(handle, allow_pickle=False)
        assert values.shape[0] == len(
            request.physical_owner.heldout_row_ids
        )


def _rewrite_terminal(root: Path, mutate) -> None:
    path = root / "execution_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    body = {key: child for key, child in value.items() if key != "content_sha256"}
    value["content_sha256"] = _canonical_sha256(body)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _copy_execution(completed_execution, name: str) -> Path:
    target = (completed_execution["work"] / name).resolve()
    shutil.copytree(completed_execution["root"], target)
    return target


def test_request_is_invariant_to_device_ids_counts_and_assignment_order():
    cpu_plan = _plan(())
    one_gpu_plan = _plan((7,))
    multi_gpu_plan = _plan((9, 3, 12))
    assert len({plan.content_sha256 for plan in (cpu_plan, one_gpu_plan, multi_gpu_plan)}) == 3
    assert len(
        {
            plan.scientific_content_sha256
            for plan in (cpu_plan, one_gpu_plan, multi_gpu_plan)
        }
    ) == 1

    requests = [_request(plan) for plan in (cpu_plan, one_gpu_plan, multi_gpu_plan)]
    assert len({request.content_sha256 for request in requests}) == 1
    assert requests[0].as_dict() == requests[1].as_dict() == requests[2].as_dict()
    assert "plan_content_sha256" not in requests[0].as_dict()
    assert requests[0].as_dict()["execution_device_metadata_in_scientific_identity"] is False


def test_config_has_no_defaults_and_exact_input_has_no_label_fields():
    for field in dataclasses.fields(RoleNeutralMatchedPairConfig):
        assert field.default is dataclasses.MISSING
        assert field.default_factory is dataclasses.MISSING
    exact_fields = {field.name for field in dataclasses.fields(RoleNeutralMatchedPairExactInput)}
    assert exact_fields == {
        "row_ids",
        "texts",
        "propensity_probability",
        "outcome_nuisance_probability",
    }
    parameters = inspect.signature(
        execute_role_neutral_matched_pair_physical_group
    ).parameters
    assert "heldout_treatment" not in parameters
    assert "heldout_outcome" not in parameters


def test_matched_configuration_roundtrips_without_optimizer_defaults() -> None:
    config = _config()
    assert RoleNeutralMatchedPairConfig.from_mapping(config.as_dict()) == config
    missing = config.as_dict()
    del missing["bow_optimizer_maxls"]
    with pytest.raises(ValueError, match="missing=.*bow_optimizer_maxls"):
        RoleNeutralMatchedPairConfig.from_mapping(missing)


def test_matched_typed_htr_rejects_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCI_HTR_ENCODER_BATCH_SIZE", "1")
    with pytest.raises(RuntimeError, match="forbids environment overrides"):
        _factory(_config())(torch.device("cpu"))


def test_serial_and_process_parallel_matched_pair_preserve_science_and_attest_leases(
    tmp_path: Path,
) -> None:
    request = _request()
    config = _config()
    *_fit_inputs, exact = _inputs(request)
    serial_root = (tmp_path / "matched_serial").resolve()
    parallel_root = (tmp_path / "matched_parallel").resolve()
    serial_attestations: list[dict] = []
    parallel_attestations: list[dict] = []

    serial_plan = RoleNeutralHTRFoldResourcePlan(
        devices=("cpu",),
        fold_parallelism=1,
        fold_slots_per_device=1,
        owner_cpu_budget=2,
        fold_parallel_backend="threads",
    )
    parallel_plan = RoleNeutralHTRFoldResourcePlan(
        devices=("cpu",),
        fold_parallelism=2,
        fold_slots_per_device=2,
        owner_cpu_budget=2,
        fold_parallel_backend="processes",
    )

    def serial_loader(row_ids: tuple[int, ...]):
        assert serial_attestations
        assert row_ids == request.physical_owner.heldout_row_ids
        return exact

    def parallel_loader(row_ids: tuple[int, ...]):
        assert parallel_attestations
        assert row_ids == request.physical_owner.heldout_row_ids
        return exact

    serial_terminal = _execute(
        root=serial_root,
        request=request,
        loader=serial_loader,
        config=config,
        fold_resource_plan=serial_plan,
        operational_attestation_sink=serial_attestations.append,
    )
    parallel_terminal = _execute(
        root=parallel_root,
        request=request,
        loader=parallel_loader,
        config=config,
        fold_resource_plan=parallel_plan,
        operational_attestation_sink=parallel_attestations.append,
    )

    assert len(serial_attestations) == len(parallel_attestations) == 1
    for attestation in (serial_attestations[0], parallel_attestations[0]):
        attestation_body = {
            key: value
            for key, value in attestation.items()
            if key != "content_sha256"
        }
        assert attestation["content_sha256"] == _canonical_sha256(
            attestation_body
        )
    serial_execution = serial_attestations[0]["fold_execution"]
    parallel_execution = parallel_attestations[0]["fold_execution"]
    assert serial_execution["maximum_concurrent_fold_leases"] == 1
    assert parallel_execution["maximum_concurrent_fold_leases"] == 2
    assert [row["fold"] for row in parallel_execution["fold_intervals"]] == [
        1,
        2,
    ]
    assert len(
        {
            row["process_id"]
            for row in parallel_execution["fold_intervals"]
        }
    ) == 2
    assert parallel_attestations[0]["fold_resource_plan"] == (
        parallel_plan.as_dict()
    )
    assert parallel_attestations[0][
        "shared_mutable_array_store_used_by_fold_workers"
    ] is False
    assert parallel_attestations[0][
        "live_models_returned_across_fold_boundary"
    ] is False

    def load_fit(root: Path):
        metadata = json.loads(
            (root / "fit_state" / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        arrays = {}
        for key, row in metadata["array_inventory"].items():
            with (root / "fit_state" / row["relative_path"]).open(
                "rb"
            ) as handle:
                arrays[key] = np.load(handle, allow_pickle=False)
        return metadata, arrays

    serial_metadata, serial_arrays = load_fit(serial_root)
    parallel_metadata, parallel_arrays = load_fit(parallel_root)

    def discrete_metadata(metadata: dict) -> dict:
        result = json.loads(json.dumps(metadata))
        result.pop("content_sha256")
        result.pop("subproducer_evidence_identity_sha256")
        for row in result["array_inventory"].values():
            row.pop("content_sha256")
            row.pop("file_sha256")
            row.pop("size_bytes")
        return result

    assert discrete_metadata(serial_metadata) == discrete_metadata(
        parallel_metadata
    )
    assert set(serial_arrays) == set(parallel_arrays)
    htr_array_keys = {
        key for key in serial_arrays if "_htr_" in key
    } | {
        serial_metadata["fit_numerical_bank"][name]
        for name in serial_metadata["fit_numerical_bank"]
        if name.startswith("htr::")
    }
    for key in sorted(serial_arrays):
        if key in htr_array_keys:
            np.testing.assert_allclose(
                parallel_arrays[key],
                serial_arrays[key],
                rtol=config.replay_relative_tolerance,
                atol=config.replay_absolute_tolerance,
                equal_nan=True,
            )
        else:
            np.testing.assert_array_equal(
                parallel_arrays[key],
                serial_arrays[key],
            )

    serial_seal = json.loads(
        (serial_root / "fit_only_family_seal.json").read_text(
            encoding="utf-8"
        )
    )
    parallel_seal = json.loads(
        (parallel_root / "fit_only_family_seal.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        serial_seal["subproducer_proofs"][0]["evidence_payload"]
        == parallel_seal["subproducer_proofs"][0]["evidence_payload"]
    )
    serial_htr_atoms = serial_seal["subproducer_proofs"][1][
        "evidence_payload"
    ]["atoms"]
    parallel_htr_atoms = parallel_seal["subproducer_proofs"][1][
        "evidence_payload"
    ]["atoms"]
    assert len(serial_htr_atoms) == len(parallel_htr_atoms)
    for serial_atom, parallel_atom in zip(
        serial_htr_atoms,
        parallel_htr_atoms,
        strict=True,
    ):
        serial_discrete = dict(serial_atom)
        parallel_discrete = dict(parallel_atom)
        serial_delta = serial_discrete.pop("delta_logit")
        parallel_delta = parallel_discrete.pop("delta_logit")
        assert serial_discrete == parallel_discrete
        assert parallel_delta == pytest.approx(
            serial_delta,
            rel=config.replay_relative_tolerance,
            abs=config.replay_absolute_tolerance,
        )

    for subproducer in ("bow", "htr"):
        serial_view = json.loads(
            (
                serial_root
                / "logical_views"
                / f"{request.physical_owner.scope_id}.json"
            ).read_text(encoding="utf-8")
        )
        parallel_view = json.loads(
            (
                parallel_root
                / "logical_views"
                / f"{request.physical_owner.scope_id}.json"
            ).read_text(encoding="utf-8")
        )
        serial_registration = serial_view["prediction_artifacts"][
            subproducer
        ]
        parallel_registration = parallel_view["prediction_artifacts"][
            subproducer
        ]
        with (serial_root / serial_registration["relative_path"]).open(
            "rb"
        ) as handle:
            serial_prediction = np.load(handle, allow_pickle=False)
        with (parallel_root / parallel_registration["relative_path"]).open(
            "rb"
        ) as handle:
            parallel_prediction = np.load(handle, allow_pickle=False)
        if subproducer == "bow":
            np.testing.assert_array_equal(
                parallel_prediction,
                serial_prediction,
            )
        else:
            np.testing.assert_allclose(
                parallel_prediction,
                serial_prediction,
                rtol=config.replay_relative_tolerance,
                atol=config.replay_absolute_tolerance,
                equal_nan=True,
            )

    for terminal in (serial_terminal, parallel_terminal):
        assert "fold_resource_plan" not in terminal
        assert "fold_execution" not in terminal
        assert terminal["live_model_objects_reused_for_exact_transform"] is False
        assert terminal["model_state_reloaded_for_primary_transform"] is True


def test_matched_pair_consumes_authenticated_bow_bank_without_heldout_labels(
    tmp_path: Path,
):
    plan = _plan()
    preliminary_request = _request(plan)
    texts, treatment, outcome, _propensity, _outcome_nuisance, exact = (
        _inputs(preliminary_request)
    )
    bow_request = RoleNeutralBoWPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=(
            preliminary_request.physical_owner.scope_id
        ),
    )
    bow_root = (tmp_path / "bow_nuisance").resolve()
    execute_role_neutral_bow_physical_group(
        request=bow_request,
        output_root=bow_root,
        fit_texts=texts,
        fit_treatment=treatment,
        fit_outcome=outcome,
        view_configs=(_view(),),
        nuisance_folds=2,
        effect_folds=2,
        e_clip=0.02,
        bow_fold_parallelism=1,
        bow_parallel_backend="threads",
        owner_cpu_budget=1,
        exact_heldout_text_loader=lambda row_ids: (
            exact.texts
            if row_ids
            == preliminary_request.physical_owner.heldout_row_ids
            else ()
        ),
    )
    bank = load_authenticated_role_neutral_bow_nuisance_bank(
        root=bow_root,
        request=bow_request,
    )
    request = RoleNeutralMatchedPairPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=(
            preliminary_request.physical_owner.scope_id
        ),
        htr_model_identity_sha256="b" * 64,
        nuisance_artifact_identity_sha256=bank.content_sha256,
        runtime_compatibility_class="torch-cpu-float32-v1",
    )
    config = _config()
    heldout_calls: list[tuple[int, ...]] = []

    def heldout_text_loader(row_ids: tuple[int, ...]):
        heldout_calls.append(row_ids)
        return exact.texts

    terminal = execute_role_neutral_matched_pair_from_bow_nuisance_bank(
        request=request,
        output_root=(tmp_path / "matched_pair").resolve(),
        fit_texts=texts,
        fit_treatment=treatment,
        fit_outcome=outcome,
        nuisance_bank=bank,
        view_configs=(_view(),),
        config=config,
        htr_extractor_factory=_factory(config),
        exact_heldout_text_loader=heldout_text_loader,
        device="cpu",
    )

    assert heldout_calls == [request.physical_owner.heldout_row_ids]
    assert terminal["registered_heldout_labels_accessed"] is False
    assert terminal["group_request"][
        "nuisance_artifact_identity_sha256"
    ] == bank.content_sha256
    owner_view = json.loads(
        (
            tmp_path
            / "matched_pair"
            / "logical_views"
            / f"{request.physical_owner.scope_id}.json"
        ).read_text(encoding="utf-8")
    )
    assert owner_view["exact_input_identity"][
        "heldout_treatment_field_present"
    ] is False
    assert owner_view["exact_input_identity"][
        "heldout_outcome_field_present"
    ] is False


def test_both_subproducers_seal_before_text_and_fresh_replay_is_exact(
    completed_execution,
):
    root = completed_execution["root"]
    request = completed_execution["request"]
    terminal = completed_execution["terminal"]
    assert (
        validate_role_neutral_matched_pair_group_execution(
            root=root,
            request=request,
        )
        == terminal
    )
    events = terminal["event_order"]
    names = [row["event"] for row in events]
    assert names[:4] == [
        "fit_completed",
        "matched_pair_subproducer_sealed",
        "matched_pair_subproducer_sealed",
        "fit_family_artifact_sealed",
    ]
    opened = names.index("exact_heldout_text_opened")
    assert all(
        index < opened
        for index, name in enumerate(names)
        if name == "cumulative_fit_only_view_published"
    )
    assert [events[1]["subproducer"], events[2]["subproducer"]] == ["bow", "htr"]
    assert terminal["family"] == MATCHED_PAIR_UPLIFT
    assert terminal["text_truncation_applied"] is False
    assert terminal["top_k_evidence_applied"] is False
    assert terminal["pickle_joblib_npz_loaded_or_written"] is False

    seal = json.loads((root / "fit_only_family_seal.json").read_text(encoding="utf-8"))
    bow_evidence = seal["subproducer_proofs"][0]["evidence_payload"]
    assert any(
        atom.get("term") == "sentinelafterfourteenthousand"
        for atom in bow_evidence["atoms"]
    )
    assert all(
        proof["evidence_payload"]["top_k_applied"] is False
        and proof["evidence_payload"]["text_truncation_applied"] is False
        and proof["evidence_payload"]["atoms"]
        for proof in seal["subproducer_proofs"]
    )
    suffixes = {
        path.suffix
        for path in root.rglob("*")
        if path.is_file()
    }
    assert suffixes <= {".json", ".npy"}
    receipt = authenticate_role_neutral_matched_pair_component(
        root=root,
        plan=_plan(),
        physical_owner_scope_id=request.physical_owner.scope_id,
        htr_model_identity_sha256=request.htr_model_identity_sha256,
        nuisance_artifact_identity_sha256=(
            request.nuisance_artifact_identity_sha256
        ),
        runtime_compatibility_class=request.runtime_compatibility_class,
    )
    assert tuple(receipt.family_fit_seals) == (MATCHED_PAIR_UPLIFT,)
    assert receipt.lossy_evidence_selection_applied is False

    replayed = replay_role_neutral_matched_pair_exact_transform(
        root=root,
        request=request,
        fit_texts=completed_execution["fit_texts"],
        exact_input=completed_execution["exact"],
    )
    assert set(replayed) == {"bow", "htr"}
    assert replayed["bow"]["values"].shape[0] == len(
        request.physical_owner.heldout_row_ids
    )
    assert replayed["htr"]["values"].shape[1] == 3


def test_binding_htr_capacity_aborts_instead_of_truncating(tmp_path: Path):
    request = _request()
    config = _config(
        htr_extractor=_extractor_configuration(
            chunk_size_words=1,
            max_chunks=1,
        )
    )
    called = False

    def loader(_row_ids):
        nonlocal called
        called = True
        return _inputs(request)[-1]

    with pytest.raises(ValueError, match="truncation is forbidden"):
        _execute(
            root=(tmp_path / "capacity_failure").resolve(),
            request=request,
            loader=loader,
            config=config,
        )
    assert called is False


def test_missing_extra_reordered_tampered_and_linked_artifacts_fail_closed(
    completed_execution,
):
    request = completed_execution["request"]

    missing = _copy_execution(completed_execution, "missing")
    (missing / "logical_views" / f"{request.physical_owner.scope_id}.bow.predictions.npy").unlink()
    with pytest.raises(ValueError):
        validate_role_neutral_matched_pair_group_execution(root=missing, request=request)

    extra = _copy_execution(completed_execution, "extra")
    (extra / "unregistered.npy").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory"):
        validate_role_neutral_matched_pair_group_execution(root=extra, request=request)

    reordered = _copy_execution(completed_execution, "reordered")

    def reverse_views(value):
        value["logical_views"].reverse()

    _rewrite_terminal(reordered, reverse_views)
    with pytest.raises(ValueError, match="registration order"):
        validate_role_neutral_matched_pair_group_execution(root=reordered, request=request)

    tampered = _copy_execution(completed_execution, "tampered")
    prediction = (
        tampered
        / "logical_views"
        / f"{request.physical_owner.scope_id}.bow.predictions.npy"
    )
    with prediction.open("rb") as handle:
        values = np.load(handle, allow_pickle=False)
    values = np.array(values, copy=True)
    values[0, 0] += 1.0
    with prediction.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    with pytest.raises(ValueError, match="prediction file changed"):
        validate_role_neutral_matched_pair_group_execution(root=tampered, request=request)

    symlinked = _copy_execution(completed_execution, "symlinked")
    prediction = (
        symlinked
        / "logical_views"
        / f"{request.physical_owner.scope_id}.htr.predictions.npy"
    )
    outside = completed_execution["work"] / "outside_prediction.npy"
    shutil.copy2(prediction, outside)
    prediction.unlink()
    prediction.symlink_to(outside)
    with pytest.raises(ValueError):
        validate_role_neutral_matched_pair_group_execution(root=symlinked, request=request)

    hardlinked = _copy_execution(completed_execution, "hardlinked")
    prediction = (
        hardlinked
        / "logical_views"
        / f"{request.physical_owner.scope_id}.htr.predictions.npy"
    )
    outside_link = completed_execution["work"] / "outside_hardlink.npy"
    os.link(prediction, outside_link)
    try:
        with pytest.raises(ValueError, match="linked"):
            validate_role_neutral_matched_pair_group_execution(
                root=hardlinked,
                request=request,
            )
    finally:
        outside_link.unlink()


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("dtype", np.zeros((1, 1), dtype=np.int16)),
        ("shape", np.zeros((2, 7), dtype=np.float64)),
    ],
)
def test_prediction_dtype_and_shape_substitution_fails_closed(
    completed_execution,
    name,
    replacement,
):
    request = completed_execution["request"]
    root = _copy_execution(completed_execution, f"bad_{name}")
    prediction = (
        root
        / "logical_views"
        / f"{request.physical_owner.scope_id}.bow.predictions.npy"
    )
    with prediction.open("wb") as handle:
        np.save(handle, replacement, allow_pickle=False)
    with pytest.raises(ValueError):
        validate_role_neutral_matched_pair_group_execution(root=root, request=request)


def test_semantically_reordered_event_ledger_fails_even_when_resigned(
    completed_execution,
):
    request = completed_execution["request"]
    root = _copy_execution(completed_execution, "bad_event_order")

    def reorder(value):
        events = value["event_order"]
        opened = next(
            index
            for index, row in enumerate(events)
            if row["event"] == "exact_heldout_text_opened"
        )
        cumulative = next(
            index
            for index, row in enumerate(events)
            if row["event"] == "cumulative_fit_only_view_published"
        )
        events[opened], events[cumulative] = events[cumulative], events[opened]
        for sequence, row in enumerate(events, start=1):
            row["sequence"] = sequence
        # Resign the self-contained terminal envelope so semantic event-order
        # validation, not merely its digest check, rejects the substitution.

    _rewrite_terminal(root, reorder)
    with pytest.raises(ValueError, match="published after text access|text-access state"):
        validate_role_neutral_matched_pair_group_execution(root=root, request=request)


def test_reordered_exact_input_is_rejected_before_replay(completed_execution):
    request = completed_execution["request"]
    exact = completed_execution["exact"]
    reordered = RoleNeutralMatchedPairExactInput(
        row_ids=tuple(reversed(exact.row_ids)),
        texts=tuple(reversed(exact.texts)),
        propensity_probability=tuple(reversed(exact.propensity_probability)),
        outcome_nuisance_probability=tuple(
            reversed(exact.outcome_nuisance_probability)
        ),
    )
    with pytest.raises(ValueError, match="authorized row/text order"):
        replay_role_neutral_matched_pair_exact_transform(
            root=completed_execution["root"],
            request=request,
            fit_texts=completed_execution["fit_texts"],
            exact_input=reordered,
        )
