from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from oci.inference.production_stage1_scope_scheduler import (
    build_canonical_stage1_scope_plan,
)
from oci.inference.role_neutral_htr_group_execution import (
    RoleNeutralHTRConfig,
    RoleNeutralHTRPhysicalGroupRequest,
    execute_role_neutral_htr_physical_group,
    replay_role_neutral_htr_exact_transform,
    validate_role_neutral_htr_group_execution,
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
    ).validated()


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
    missing = config.as_dict()
    del missing["max_chunks"]
    with pytest.raises(ValueError, match="missing=.*max_chunks"):
        RoleNeutralHTRConfig.from_mapping(missing)


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
    np.testing.assert_allclose(replay["predictions"], registered, rtol=3e-5, atol=3e-6)

    assert not tuple(root.rglob("*.npz"))
    assert not tuple(root.rglob("*.pkl"))
    assert not tuple(root.rglob("*.pickle"))
    assert not tuple(root.rglob("*.joblib"))
    assert all(path.suffix in {".json", ".npy"} for path in root.rglob("*") if path.is_file())


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
