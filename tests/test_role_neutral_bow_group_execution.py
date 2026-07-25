from __future__ import annotations

import dataclasses
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from oci.config import BoWViewConfig
from oci.inference.all_evidence_discovery_interfaces import (
    BOW_NUISANCE,
    BOW_R_LOSS,
)
from oci.inference.production_role_neutral_stage2_handoff import (
    ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_PROOF_SCHEMA,
    ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD,
    validate_role_neutral_stage2_fit_projection_proof,
)
from oci.inference.role_neutral_bow_group_execution import (
    AuthenticatedRoleNeutralBoWNuisanceBank,
    RoleNeutralBoWPhysicalGroupRequest,
    execute_role_neutral_bow_physical_group,
    load_authenticated_role_neutral_bow_nuisance_bank,
    replay_role_neutral_bow_exact_transform,
    validate_role_neutral_bow_group_execution,
)
from oci.inference.production_stage1_legacy_scope_fragments import (
    build_role_neutral_fit_only_family_seal,
)
from oci.inference.production_stage1_scope_scheduler import (
    build_canonical_stage1_scope_plan,
)
from oci.inference.role_neutral_all_ten_binding import (
    authenticate_role_neutral_bow_component,
    validate_authenticated_role_neutral_component_receipt,
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
                        "fit_row_ids": [row for row in fit if row not in set(inner_heldout)],
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


def _plan(*, gpu_ids: tuple[int, ...] = ()):
    return build_canonical_stage1_scope_plan(
        registry=_registry(),
        registry_content_sha256="a" * 64,
        global_seed=42,
        gpu_ids=gpu_ids,
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=2,
        expected_inner_fold_count=5,
    )


def _request() -> RoleNeutralBoWPhysicalGroupRequest:
    plan = _plan()
    owner, members = next(
        (owner, members) for owner, members in plan.physical_scope_groups if len(members) > 1
    )
    assert owner.scope_kind == "exact_inner"
    assert members[1].scope_kind == "cumulative_spent"
    return RoleNeutralBoWPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
    )


def _singleton_request(
    owner_kind: str,
) -> RoleNeutralBoWPhysicalGroupRequest:
    plan = _plan()
    owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if owner.scope_kind == owner_kind and len(members) == 1
    )
    assert members == (owner,)
    return RoleNeutralBoWPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
    )


def test_group_request_scientific_identity_is_device_independent():
    cpu = _plan()
    gpu = _plan(gpu_ids=(7, 2))
    owner = next(owner for owner, members in cpu.physical_scope_groups if len(members) > 1)
    cpu_request = RoleNeutralBoWPhysicalGroupRequest.from_plan(
        plan=cpu,
        physical_owner_scope_id=owner.scope_id,
    )
    gpu_request = RoleNeutralBoWPhysicalGroupRequest.from_plan(
        plan=gpu,
        physical_owner_scope_id=owner.scope_id,
    )

    assert cpu.content_sha256 != gpu.content_sha256
    assert cpu_request.as_dict() == gpu_request.as_dict()


def _inputs(request: RoleNeutralBoWPhysicalGroupRequest):
    fit_texts = [
        f"patient row {row_id} biomarker_{position % 3} " f"therapy_{position % 2}"
        for position, row_id in enumerate(request.physical_owner.fit_row_ids)
    ]
    # A suffix beyond the benchmark's former 14k-character boundary proves
    # this executor passes complete configured text to the vectorizer.
    fit_texts[0] = ("paddingword " * 1400) + " sentinelafterfourteenthousand"
    treatment = np.asarray(
        [position % 2 for position in range(len(fit_texts))],
        dtype=float,
    )
    outcome = 1.0 - treatment
    heldout_texts = tuple(
        f"heldout row {row_id} exact_transform_token"
        for row_id in request.physical_owner.heldout_row_ids
    )
    view = BoWViewConfig(
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
    return tuple(fit_texts), treatment, outcome, heldout_texts, (view,)


def _execute(
    *,
    root: Path,
    request: RoleNeutralBoWPhysicalGroupRequest,
    loader,
):
    fit_texts, treatment, outcome, _heldout_texts, views = _inputs(request)
    return execute_role_neutral_bow_physical_group(
        request=request,
        output_root=root,
        fit_texts=fit_texts,
        fit_treatment=treatment,
        fit_outcome=outcome,
        view_configs=views,
        nuisance_folds=2,
        effect_folds=2,
        e_clip=0.02,
        exact_heldout_text_loader=loader,
    )


def _load_array(root: Path, metadata: dict, reference: str) -> np.ndarray:
    registration = metadata["array_inventory"][reference]
    with (root / "fit_state" / registration["relative_path"]).open("rb") as handle:
        return np.load(handle, allow_pickle=False)


def _content_sha256(value: dict) -> str:
    body = {key: child for key, child in value.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_group_request_contains_only_owner_and_complete_equivalence_group():
    request = _request()
    payload = request.as_dict()

    assert payload["logical_scope_count"] == 2
    assert payload["physical_owner"]["scope_kind"] == "exact_inner"
    assert payload["logical_members"][1]["scope_kind"] == "cumulative_spent"
    assert {tuple(sorted(member["fit_row_ids"])) for member in payload["logical_members"]} == {
        tuple(sorted(payload["fit_row_ids"]))
    }
    assert payload["heldout_labels_supplied"] is False
    assert payload["peer_group_definitions_supplied"] is False


@pytest.mark.parametrize("owner_kind", ("full_outer", "cumulative_spent"))
def test_singleton_owner_gets_primary_heldout_numerical_transforms(
    tmp_path: Path,
    owner_kind: str,
):
    request = _singleton_request(owner_kind)
    _fit_texts, _treatment, _outcome, heldout_texts, _views = _inputs(request)
    root = (tmp_path / owner_kind).resolve()
    calls: list[tuple[int, ...]] = []

    def loader(row_ids: tuple[int, ...]):
        calls.append(row_ids)
        return heldout_texts

    terminal = _execute(root=root, request=request, loader=loader)

    assert calls == [request.physical_owner.heldout_row_ids]
    assert terminal["registered_heldout_labels_accessed"] is False
    assert all(
        event["event"] != "cumulative_fit_only_view_published" for event in terminal["event_order"]
    )
    for family, suffix in (
        (BOW_NUISANCE, ""),
        (BOW_R_LOSS, f".{BOW_R_LOSS}"),
    ):
        view = json.loads(
            (root / "logical_views" / f"{request.physical_owner.scope_id}{suffix}.json").read_text(
                encoding="utf-8"
            )
        )
        assert view["family"] == family
        assert view["logical_purpose"] == owner_kind
        assert view["logical_transform_performed"] is True
        assert view["registered_heldout_labels_accessed"] is False
        with (root / view["prediction_artifact"]["relative_path"]).open("rb") as handle:
            values = np.load(handle, allow_pickle=False)
        assert values.shape[0] == len(request.physical_owner.heldout_row_ids)
        assert np.isfinite(values).all()


def test_authenticated_nuisance_bank_is_label_free_and_location_neutral(
    tmp_path: Path,
):
    request = _request()
    _fit_texts, _treatment, _outcome, heldout_texts, _views = _inputs(request)
    root = (tmp_path / "nuisance_source").resolve()
    _execute(
        root=root,
        request=request,
        loader=lambda row_ids: (
            heldout_texts if row_ids == request.physical_owner.heldout_row_ids else ()
        ),
    )

    bank = load_authenticated_role_neutral_bow_nuisance_bank(
        root=root,
        request=request,
    )

    assert type(bank) is AuthenticatedRoleNeutralBoWNuisanceBank
    assert bank.fit_row_ids == request.physical_owner.fit_row_ids
    assert bank.heldout_row_ids == request.physical_owner.heldout_row_ids
    assert len(bank.fit_propensity_probability) == len(bank.fit_row_ids)
    assert len(bank.fit_outcome_nuisance_probability) == len(bank.fit_row_ids)
    assert len(bank.heldout_propensity_probability) == len(bank.heldout_row_ids)
    assert len(bank.heldout_outcome_nuisance_probability) == len(bank.heldout_row_ids)
    assert all(
        0.0 <= value <= 1.0
        for values in (
            bank.fit_propensity_probability,
            bank.fit_outcome_nuisance_probability,
            bank.heldout_propensity_probability,
            bank.heldout_outcome_nuisance_probability,
        )
        for value in values
    )
    assert {field.name for field in dataclasses.fields(type(bank))}.isdisjoint(
        {
            "fit_treatment",
            "fit_outcome",
            "heldout_treatment",
            "heldout_outcome",
        }
    )
    assert bank.as_dict()["heldout_treatment_field_present"] is False
    assert bank.as_dict()["heldout_outcome_field_present"] is False

    relocated = (tmp_path / "relocated_nuisance_source").resolve()
    root.rename(relocated)
    reopened = load_authenticated_role_neutral_bow_nuisance_bank(
        root=relocated,
        request=request,
    )
    assert reopened == bank


def test_fit_seals_before_loader_and_replays_without_live_models(
    tmp_path: Path,
):
    request = _request()
    fit_texts, treatment, outcome, heldout_texts, views = _inputs(request)
    root = (tmp_path / "role_neutral_bow").resolve()
    loader_calls: list[tuple[int, ...]] = []
    cumulative = request.logical_members[1]

    def heldout_loader(row_ids: tuple[int, ...]):
        # This callback is the first point at which exact held-out text can
        # enter worker memory. Both family seals and both cumulative
        # reference-only views must already be durable.
        for filename in (
            "fit_only_family_seal.json",
            "fit_only_bow_r_loss_family_seal.json",
        ):
            assert (root / filename).is_file()
        for suffix in ("", f".{BOW_R_LOSS}"):
            cumulative_path = root / "logical_views" / f"{cumulative.scope_id}{suffix}.json"
            assert cumulative_path.is_file()
            cumulative_view = json.loads(cumulative_path.read_text(encoding="utf-8"))
            assert cumulative_view["registered_heldout_text_accessed"] is False
            assert cumulative_view["prediction_artifact"] is None
        loader_calls.append(row_ids)
        return heldout_texts

    terminal = _execute(
        root=root,
        request=request,
        loader=heldout_loader,
    )
    assert loader_calls == [request.physical_owner.heldout_row_ids]
    assert (
        validate_role_neutral_bow_group_execution(
            root=root,
            request=request,
        )
        == terminal
    )

    event_names = [row["event"] for row in terminal["event_order"]]
    assert event_names[:3] == [
        "fit_completed",
        "fit_family_artifact_sealed",
        "fit_family_artifact_sealed",
    ]
    exact_open_index = event_names.index("exact_heldout_text_opened")
    assert all(
        event_names.index("cumulative_fit_only_view_published", index) < exact_open_index
        for index, value in enumerate(event_names)
        if value == "cumulative_fit_only_view_published"
    )
    assert terminal["families"] == [BOW_NUISANCE, BOW_R_LOSS]
    assert terminal["live_model_objects_reused_for_exact_transform"] is True
    assert terminal["model_state_reloaded_for_primary_transform"] is False
    assert terminal["text_truncation_applied"] is False
    projection = terminal[ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD]
    metadata = json.loads((root / "fit_state" / "metadata.json").read_text(encoding="utf-8"))
    assert projection["schema_version"] == (ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_PROOF_SCHEMA)
    assert projection["fit_row_ids"] == list(request.physical_owner.fit_row_ids)
    assert projection["raw_text_persisted"] is False
    assert projection["raw_treatment_persisted"] is False
    assert projection["raw_outcome_persisted"] is False
    assert projection["text_truncation_applied"] is False
    assert projection == validate_role_neutral_stage2_fit_projection_proof(
        projection,
        expected_plan_scientific_content_sha256=(request.plan_scientific_content_sha256),
        expected_physical_owner_scope_id=(request.physical_owner.scope_id),
        expected_fit_row_ids=request.physical_owner.fit_row_ids,
        expected_fit_text_sha256=metadata["fit_text_sha256"],
        expected_fit_treatment_sha256=metadata["fit_treatment_sha256"],
        expected_fit_outcome_sha256=metadata["fit_outcome_sha256"],
    )

    seal_paths = {
        BOW_NUISANCE: root / "fit_only_family_seal.json",
        BOW_R_LOSS: root / "fit_only_bow_r_loss_family_seal.json",
    }
    for family, seal_path in seal_paths.items():
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        terms = {row.get("term") for row in seal["evidence_payload"]["architecture_evidence"]}
        assert "sentinelafterfourteenthousand" in terms
        assert seal["family"] == family
        assert seal["registered_heldout_text_accessed"] is False
        assert seal == build_role_neutral_fit_only_family_seal(
            plan=_plan(),
            physical_owner_scope_id=request.physical_owner.scope_id,
            family=family,
            evidence_payload=seal["evidence_payload"],
            producer_identity_sha256=seal["producer_identity_sha256"],
            configuration_identity_sha256=seal["configuration_identity_sha256"],
            fit_state_artifact_sha256=seal["fit_state_artifact_sha256"],
        )
    receipt = authenticate_role_neutral_bow_component(
        root=root,
        plan=_plan(),
        physical_owner_scope_id=request.physical_owner.scope_id,
    )
    assert set(receipt.family_fit_seals) == {BOW_NUISANCE, BOW_R_LOSS}
    assert receipt.text_truncation_applied is False
    assert (
        validate_authenticated_role_neutral_component_receipt(
            root=root,
            plan=_plan(),
            physical_owner_scope_id=request.physical_owner.scope_id,
            receipt=receipt,
            expected_component="bow",
        )
        is receipt
    )

    # The execution call has returned, so no live model handle is supplied.
    # Replay freshly reopens only the JSON + individual non-object NPY files.
    replayed = replay_role_neutral_bow_exact_transform(
        root=root,
        request=request,
        exact_heldout_texts=heldout_texts,
    )
    assert replayed["live_model_objects_available"] is False
    assert replayed["pickle_or_joblib_loaded"] is False
    assert replayed["state_source"] == "authenticated_json_and_npy_only"
    live_by_family = {}
    for family in (BOW_NUISANCE, BOW_R_LOSS):
        suffix = "" if family == BOW_NUISANCE else f".{family}"
        exact_view = json.loads(
            (root / "logical_views" / f"{request.physical_owner.scope_id}{suffix}.json").read_text(
                encoding="utf-8"
            )
        )
        prediction_path = root / exact_view["prediction_artifact"]["relative_path"]
        with prediction_path.open("rb") as handle:
            live_predictions = np.load(handle, allow_pickle=False)
        live_by_family[family] = live_predictions
        replay_family = replayed["family_predictions"][family]
        assert replay_family["columns"] == exact_view["prediction_artifact"]["columns"]
        np.testing.assert_allclose(
            replay_family["predictions"],
            live_predictions,
            rtol=1e-10,
            atol=1e-10,
        )
    assert replayed["columns"] == replayed["family_predictions"][BOW_NUISANCE]["columns"]
    np.testing.assert_array_equal(
        replayed["predictions"],
        replayed["family_predictions"][BOW_NUISANCE]["predictions"],
    )
    assert replayed["family_predictions"][BOW_R_LOSS]["columns"] == [
        "configured_unigram::effect_pseudo_target",
        "configured_unigram::effect_weighted_r",
    ]
    assert np.isfinite(live_by_family[BOW_R_LOSS]).all()
    assert not np.allclose(
        live_by_family[BOW_R_LOSS][:, 0],
        live_by_family[BOW_R_LOSS][:, 1],
    )
    assert not tuple(root.rglob("*.pkl"))
    assert not tuple(root.rglob("*.pickle"))
    assert not tuple(root.rglob("*.joblib"))


def test_loader_failure_occurs_only_after_fit_and_cumulative_publication(
    tmp_path: Path,
):
    request = _request()
    fit_texts, treatment, outcome, _heldout_texts, views = _inputs(request)
    root = (tmp_path / "interrupted_after_seal").resolve()
    cumulative = request.logical_members[1]
    calls = 0

    def unavailable_loader(row_ids: tuple[int, ...]):
        nonlocal calls
        calls += 1
        assert row_ids == request.physical_owner.heldout_row_ids
        assert (root / "fit_only_family_seal.json").is_file()
        assert (root / "fit_only_bow_r_loss_family_seal.json").is_file()
        assert (root / "logical_views" / f"{cumulative.scope_id}.json").is_file()
        assert (root / "logical_views" / f"{cumulative.scope_id}.{BOW_R_LOSS}.json").is_file()
        raise RuntimeError("held-out text transport unavailable")

    with pytest.raises(RuntimeError, match="transport unavailable"):
        _execute(
            root=root,
            request=request,
            loader=unavailable_loader,
        )

    assert calls == 1
    assert (root / "fit_only_family_seal.json").is_file()
    assert (root / "fit_only_bow_r_loss_family_seal.json").is_file()
    assert (root / "logical_views" / f"{cumulative.scope_id}.json").is_file()
    assert (root / "logical_views" / f"{cumulative.scope_id}.{BOW_R_LOSS}.json").is_file()
    assert not (root / "execution_manifest.json").exists()


def test_fit_only_oof_and_r_loss_formulas_replay_from_persisted_arrays(
    tmp_path: Path,
):
    request = _request()
    fit_texts, treatment, outcome, heldout_texts, _views = _inputs(request)
    root = (tmp_path / "fit_formula_proof").resolve()
    _execute(
        root=root,
        request=request,
        loader=lambda _rows: heldout_texts,
    )
    metadata = json.loads((root / "fit_state" / "metadata.json").read_text(encoding="utf-8"))
    derived = {
        name: _load_array(root, metadata, reference)
        for name, reference in metadata["derived_fit_quantities"].items()
    }
    oof = {
        name: _load_array(root, metadata, reference)
        for name, reference in metadata["oof_predictions"].items()
    }
    clip = metadata["configuration"]["e_clip"]
    expected_e_hat = oof["configured_unigram::treatment_nuisance"]
    expected_m_hat = oof["configured_unigram::outcome_nuisance"]
    expected_clipped_e = np.clip(expected_e_hat, clip, 1.0 - clip)
    expected_t_residual = treatment - expected_clipped_e
    expected_y_residual = outcome - expected_m_hat
    expected_pseudo = expected_y_residual / expected_t_residual
    expected_weight = np.square(expected_t_residual)

    np.testing.assert_array_equal(derived["fit_treatment"], treatment)
    np.testing.assert_array_equal(derived["fit_outcome"], outcome)
    np.testing.assert_allclose(derived["ensemble_e_hat"], expected_e_hat)
    np.testing.assert_allclose(derived["ensemble_m_hat"], expected_m_hat)
    np.testing.assert_allclose(derived["clipped_e_hat"], expected_clipped_e)
    np.testing.assert_allclose(derived["t_residual"], expected_t_residual)
    np.testing.assert_allclose(derived["y_residual"], expected_y_residual)
    np.testing.assert_allclose(derived["pseudo_target"], expected_pseudo)
    np.testing.assert_allclose(derived["r_weight"], expected_weight)

    owner_fit_rows = set(request.physical_owner.fit_row_ids)
    owner_heldout_rows = set(request.physical_owner.heldout_row_ids)
    effect_records = [
        record for record in metadata["fold_records"] if record["family"] == BOW_R_LOSS
    ]
    assert {record["objective"] for record in effect_records} == {
        "effect_pseudo_target",
        "effect_weighted_r",
    }
    assert effect_records
    for record in effect_records:
        assert set(record["fit_row_ids"]).issubset(owner_fit_rows)
        assert set(record["validation_row_ids"]).issubset(owner_fit_rows)
        assert not set(record["fit_row_ids"]).intersection(owner_heldout_rows)
        positions = [
            request.physical_owner.fit_row_ids.index(row_id) for row_id in record["fit_row_ids"]
        ]
        fit_target = _load_array(root, metadata, record["fit_target"])
        np.testing.assert_allclose(fit_target, expected_pseudo[positions])
        if record["objective"] == "effect_weighted_r":
            fit_weight = _load_array(
                root,
                metadata,
                record["fit_sample_weight"],
            )
            np.testing.assert_allclose(fit_weight, expected_weight[positions])
        else:
            assert record["fit_sample_weight"] is None

    assert len(fit_texts[0]) > 14_000
    assert metadata["configuration"]["text_truncation_applied"] is False


def test_fresh_validation_rejects_tampered_r_loss_fit_array(tmp_path: Path):
    request = _request()
    fit_texts, treatment, outcome, heldout_texts, views = _inputs(request)
    root = (tmp_path / "tampered").resolve()
    _execute(
        root=root,
        request=request,
        loader=lambda _rows: heldout_texts,
    )
    metadata = json.loads((root / "fit_state" / "metadata.json").read_text(encoding="utf-8"))
    target_reference = metadata["derived_fit_quantities"]["r_weight"]
    target = root / "fit_state" / metadata["array_inventory"][target_reference]["relative_path"]
    with target.open("wb") as handle:
        np.save(handle, np.asarray([999.0]), allow_pickle=False)

    with pytest.raises((ValueError, RuntimeError), match="array|fit"):
        validate_role_neutral_bow_group_execution(
            root=root,
            request=request,
        )


def test_fresh_validation_rejects_tampered_r_loss_seal_event(
    tmp_path: Path,
):
    request = _request()
    _fit_texts, _treatment, _outcome, heldout_texts, _views = _inputs(request)
    root = (tmp_path / "tampered_r_loss_seal").resolve()
    _execute(
        root=root,
        request=request,
        loader=lambda _rows: heldout_texts,
    )
    path = root / "fit_only_bow_r_loss_family_seal.json"
    seal = json.loads(path.read_text(encoding="utf-8"))
    seal["event_order"][1]["registered_heldout_text_accessed"] = True
    seal["content_sha256"] = _content_sha256(seal)
    path.write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bow_r_loss.*seal"):
        validate_role_neutral_bow_group_execution(
            root=root,
            request=request,
        )


def test_fresh_validation_rejects_terminal_event_reordering(
    tmp_path: Path,
):
    request = _request()
    _fit_texts, _treatment, _outcome, heldout_texts, _views = _inputs(request)
    root = (tmp_path / "reordered_terminal_events").resolve()
    _execute(
        root=root,
        request=request,
        loader=lambda _rows: heldout_texts,
    )
    path = root / "execution_manifest.json"
    terminal = json.loads(path.read_text(encoding="utf-8"))
    events = terminal["event_order"]
    cumulative_index = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "cumulative_fit_only_view_published"
    )
    text_open_index = next(
        index for index, event in enumerate(events) if event["event"] == "exact_heldout_text_opened"
    )
    events[cumulative_index], events[text_open_index] = (
        events[text_open_index],
        events[cumulative_index],
    )
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
    terminal["content_sha256"] = _content_sha256(terminal)
    path.write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="text access|cumulative"):
        validate_role_neutral_bow_group_execution(
            root=root,
            request=request,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("fit_text_sha256", "1" * 64),
        ("fit_treatment_sha256", "2" * 64),
        ("fit_outcome_sha256", "3" * 64),
    ),
)
def test_fresh_validation_rejects_tampered_stage2_projection_proof(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    request = _request()
    _fit_texts, _treatment, _outcome, heldout_texts, _views = _inputs(request)
    root = (tmp_path / f"tampered_projection_{field}").resolve()
    _execute(
        root=root,
        request=request,
        loader=lambda _rows: heldout_texts,
    )
    path = root / "execution_manifest.json"
    terminal = json.loads(path.read_text(encoding="utf-8"))
    projection = terminal[ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD]
    projection[field] = replacement
    projection["fit_data_projection_sha256"] = _content_sha256(
        {
            "fit_row_ids": projection["fit_row_ids"],
            "fit_text_sha256": projection["fit_text_sha256"],
            "fit_treatment_sha256": projection["fit_treatment_sha256"],
            "fit_outcome_sha256": projection["fit_outcome_sha256"],
        }
    )
    projection["content_sha256"] = _content_sha256(projection)
    terminal["content_sha256"] = _content_sha256(terminal)
    path.write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=f"sealed {field}",
    ):
        validate_role_neutral_bow_group_execution(
            root=root,
            request=request,
        )
