from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from oci.config import (
    AppliedInferenceConfig,
    BoWViewConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
    TfidfTopicDiscoveryConfig,
)
from oci.inference.all_evidence_discovery_interfaces import (
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPICS,
)
from oci.inference.production_stage1_legacy_scope_fragments import (
    build_role_neutral_fit_only_family_seal,
)
from oci.inference.production_stage1_scope_scheduler import (
    build_canonical_stage1_scope_plan,
)
from oci.inference.role_neutral_tfidf_group_execution import (
    RoleNeutralTfidfPhysicalGroupRequest,
    _configuration,
    _scientific_fit_plan,
    execute_role_neutral_tfidf_physical_group,
    replay_role_neutral_tfidf_exact_transform,
    validate_role_neutral_tfidf_group_execution,
)
from oci.inference.role_neutral_all_ten_binding import (
    authenticate_role_neutral_tfidf_component,
)
from oci.inference.tfidf_safe_artifacts import load_fitted_topic_context


def _registry() -> dict:
    row_count = 60
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


def _request(
    *,
    plan=None,
) -> RoleNeutralTfidfPhysicalGroupRequest:
    selected_plan = _plan() if plan is None else plan
    owner, members = next(
        (owner, members)
        for owner, members in selected_plan.physical_scope_groups
        if len(members) > 1
    )
    assert owner.scope_kind == "exact_inner"
    assert members[1].scope_kind == "cumulative_spent"
    return RoleNeutralTfidfPhysicalGroupRequest.from_plan(
        plan=selected_plan,
        physical_owner_scope_id=owner.scope_id,
    )


def _singleton_request(
    owner_kind: str,
) -> RoleNeutralTfidfPhysicalGroupRequest:
    plan = _plan()
    owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if owner.scope_kind == owner_kind and len(members) == 1
    )
    assert members == (owner,)
    return RoleNeutralTfidfPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
    )


def _config(*, terms_per_topic: int = 7) -> AppliedInferenceConfig:
    topic = TfidfTopicDiscoveryConfig(
        max_features=4096,
        min_df=1,
        max_df=1.0,
        top_fraction=1.0,
        topic_count=2,
        topic_seeds=[3],
        terms_per_topic=terms_per_topic,
        nmf_max_iter=50,
        stability_repeats=0,
        minimum_arm_document_support=1,
        minimum_nuisance_source_agreement=0.0,
        minimum_subsample_selection_fraction=0.0,
        minimum_tail_sign_agreement=0.0,
        score_test_bootstrap_repeats=0,
        score_test_bootstrap_chunk_size=8,
        score_test_min_topics_per_bank=1,
        score_test_max_topics_per_bank=2,
        orphan_ngram_min_abs_fit_score=0.0,
        orphan_ngram_min_selected_clusters=0,
        orphan_ngram_max_selected_clusters=2,
        score_selection_label_policy="nested_fit_calibration",
    )
    forest = MultiModelForestConfig(
        candidate_consistency_inner_folds=5,
        tfidf_nested_calibration_folds=3,
        nuisance_folds=2,
        bow_views=[
            BoWViewConfig(
                name="linear_1_3",
                max_features=4096,
                min_df=1,
                max_df=1.0,
                ngram_range_min=1,
                ngram_range_max=3,
                bow_model="linear",
            )
        ],
        tfidf_topic=topic,
    )
    config = AppliedInferenceConfig(
        dataset_path="in_memory",
        outcome_type="binary",
        text_column="configured_note",
        treatment_column="configured_treatment",
        outcome_column="configured_outcome",
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=forest,
        ),
    )
    config.seed = 42
    return config


def _inputs(request: RoleNeutralTfidfPhysicalGroupRequest):
    texts = []
    treatment = []
    outcome = []
    baseline_terms = " ".join(f"baseline_marker_{index}" for index in range(12))
    for position, row_id in enumerate(request.physical_owner.fit_row_ids):
        arm = position % 2
        result = (position // 2) % 2
        treatment.append(arm)
        outcome.append(result)
        texts.append(
            f"{baseline_terms} patient_{row_id} arm_pattern_{arm} "
            f"outcome_pattern_{result} modifier_pattern_{position % 3} "
            f"cycle_pattern_{position % 4}"
        )
    # The suffix is well beyond the benchmark-specific historical boundary.
    # Its presence in the sealed vocabulary proves complete strings were used.
    texts = [
        ("paddingword " * 1400)
        + "sentinelafterfourteenthousand "
        + text
        for text in texts
    ]
    heldout = tuple(
        f"heldout patient {row_id} baseline_marker_1 exact_transform_token"
        for row_id in request.physical_owner.heldout_row_ids
    )
    return (
        tuple(texts),
        np.asarray(treatment, dtype=float),
        np.asarray(outcome, dtype=float),
        heldout,
    )


def _execute(*, root: Path, request, loader):
    texts, treatment, outcome, _heldout = _inputs(request)
    return execute_role_neutral_tfidf_physical_group(
        request=request,
        output_root=root,
        fit_texts=texts,
        fit_treatment=treatment,
        fit_outcome=outcome,
        config=_config(),
        exact_heldout_text_loader=loader,
    )


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


def test_request_scientific_identity_is_device_independent():
    cpu = _plan()
    gpu = _plan(gpu_ids=(7, 2))
    cpu_request = _request(plan=cpu)
    gpu_request = _request(plan=gpu)

    assert cpu.content_sha256 != gpu.content_sha256
    assert cpu_request.as_dict() == gpu_request.as_dict()


@pytest.mark.parametrize("owner_kind", ("full_outer", "cumulative_spent"))
def test_singleton_owner_gets_both_primary_heldout_numerical_transforms(
    tmp_path: Path,
    owner_kind: str,
):
    request = _singleton_request(owner_kind)
    _texts, _treatment, _outcome, heldout = _inputs(request)
    root = (tmp_path / owner_kind).resolve()
    calls: list[tuple[int, ...]] = []

    def loader(row_ids: tuple[int, ...]):
        calls.append(row_ids)
        return heldout

    terminal = _execute(root=root, request=request, loader=loader)

    assert calls == [request.physical_owner.heldout_row_ids]
    assert terminal["registered_heldout_labels_accessed"] is False
    assert all(
        event["event"] != "cumulative_fit_only_view_published"
        for event in terminal["event_order"]
    )
    for family in (TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS):
        view = json.loads(
            (
                root
                / "logical_views"
                / f"{request.physical_owner.scope_id}.{family}.json"
            ).read_text(encoding="utf-8")
        )
        assert view["logical_purpose"] == owner_kind
        assert view["logical_transform_performed"] is True
        assert view["registered_heldout_labels_accessed"] is False
        with (root / view["prediction_artifact"]["relative_path"]).open(
            "rb"
        ) as handle:
            values = np.load(handle, allow_pickle=False)
        assert values.shape[0] == len(
            request.physical_owner.heldout_row_ids
        )
        assert np.isfinite(values).all()


def test_two_family_fit_seals_before_loader_and_replays_exactly(
    tmp_path: Path,
):
    request = _request()
    texts, _treatment, _outcome, heldout, = _inputs(request)
    root = (tmp_path / "role_neutral_tfidf").resolve()
    cumulative = request.logical_members[1]
    calls: list[tuple[int, ...]] = []

    def loader(row_ids: tuple[int, ...]):
        for filename in (
            "fit_only_tfidf_topics_family_seal.json",
            "fit_only_residual_tfidf_ngrams_family_seal.json",
        ):
            assert (root / filename).is_file()
        for family in (TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS):
            view_path = (
                root
                / "logical_views"
                / f"{cumulative.scope_id}.{family}.json"
            )
            view = json.loads(view_path.read_text(encoding="utf-8"))
            assert view["prediction_artifact"] is None
            assert view["registered_heldout_text_accessed"] is False
            assert "raw_text" not in view
            assert "clinical_text" not in view
        calls.append(row_ids)
        return heldout

    terminal = _execute(root=root, request=request, loader=loader)

    assert calls == [request.physical_owner.heldout_row_ids]
    assert (
        validate_role_neutral_tfidf_group_execution(
            root=root,
            request=request,
        )
        == terminal
    )
    assert terminal["families"] == [TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS]
    assert terminal["profile_by_family"][TFIDF_ORPHAN_NGRAMS] == (
        "residual_tfidf_ngrams"
    )
    assert terminal["physical_fit_count"] == 1
    assert terminal["all_ten_family_adapter_enabled"] is False
    events = [row["event"] for row in terminal["event_order"]]
    opened = events.index("exact_heldout_text_opened")
    assert events[:3] == [
        "fit_completed",
        "fit_family_artifact_sealed",
        "fit_family_artifact_sealed",
    ]
    assert all(
        index < opened
        for index, event in enumerate(events)
        if event == "cumulative_fit_only_view_published"
    )

    metadata = json.loads(
        (root / "fit_state" / "metadata.json").read_text(encoding="utf-8")
    )
    fitted = load_fitted_topic_context(
        root / "fit_state" / metadata["fitted_context"]["relative_path"]
    )
    assert "sentinelafterfourteenthousand" in fitted.common_vectorizer.vocabulary_
    assert len(texts[0]) > 14_000
    assert metadata["configuration"]["tfidf_topic"]["terms_per_topic"] == 7
    assert metadata["configuration"]["text_truncation_applied"] is False
    assert metadata["configuration"][
        "implicit_feature_or_topic_caps_added_by_executor"
    ] is False
    published_suffixes = {path.suffix for path in root.rglob("*") if path.is_file()}
    assert published_suffixes <= {".json", ".npy"}

    for family, filename in (
        (TFIDF_TOPICS, "fit_only_tfidf_topics_family_seal.json"),
        (
            TFIDF_ORPHAN_NGRAMS,
            "fit_only_residual_tfidf_ngrams_family_seal.json",
        ),
    ):
        seal = json.loads((root / filename).read_text(encoding="utf-8"))
        assert seal == build_role_neutral_fit_only_family_seal(
            plan=request.authority_plan,
            physical_owner_scope_id=request.physical_owner.scope_id,
            family=family,
            evidence_payload=seal["evidence_payload"],
            producer_identity_sha256=seal["producer_identity_sha256"],
            configuration_identity_sha256=seal[
                "configuration_identity_sha256"
            ],
            fit_state_artifact_sha256=seal["fit_state_artifact_sha256"],
        )
    receipt = authenticate_role_neutral_tfidf_component(
        root=root,
        plan=request.authority_plan,
        physical_owner_scope_id=request.physical_owner.scope_id,
    )
    assert set(receipt.family_fit_seals) == {
        TFIDF_TOPICS,
        TFIDF_ORPHAN_NGRAMS,
    }
    assert receipt.text_truncation_applied is False

    replay = replay_role_neutral_tfidf_exact_transform(
        root=root,
        request=request,
        exact_heldout_texts=heldout,
    )
    assert replay["state_source"] == "authenticated_json_and_npy_only"
    assert replay["pickle_or_joblib_loaded"] is False
    for family in (TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS):
        view = json.loads(
            (
                root
                / "logical_views"
                / f"{request.physical_owner.scope_id}.{family}.json"
            ).read_text(encoding="utf-8")
        )
        with (root / view["prediction_artifact"]["relative_path"]).open(
            "rb"
        ) as handle:
            live = np.load(handle, allow_pickle=False)
        np.testing.assert_allclose(
            replay["family_predictions"][family]["predictions"],
            live,
            rtol=1e-12,
            atol=1e-12,
        )


def test_loader_failure_leaves_sealed_reference_only_state(
    tmp_path: Path,
):
    request = _request()
    root = (tmp_path / "unavailable").resolve()
    cumulative = request.logical_members[1]

    def loader(_row_ids):
        for family in (TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS):
            assert (
                root
                / "logical_views"
                / f"{cumulative.scope_id}.{family}.json"
            ).is_file()
        raise RuntimeError("held-out text transport unavailable")

    with pytest.raises(RuntimeError, match="transport unavailable"):
        _execute(root=root, request=request, loader=loader)

    assert not (root / "execution_manifest.json").exists()
    assert (root / "fit_only_tfidf_topics_family_seal.json").is_file()
    assert (
        root / "fit_only_residual_tfidf_ngrams_family_seal.json"
    ).is_file()


def test_fresh_validation_rejects_tampered_and_extra_payloads(
    tmp_path: Path,
):
    request = _request()
    _texts, _treatment, _outcome, heldout = _inputs(request)
    root = (tmp_path / "tampered").resolve()
    _execute(root=root, request=request, loader=lambda _rows: heldout)
    metadata = json.loads(
        (root / "fit_state" / "metadata.json").read_text(encoding="utf-8")
    )
    target = (
        root
        / "fit_state"
        / metadata["fit_family_arrays"][TFIDF_TOPICS]["relative_path"]
    )
    with target.open("wb") as handle:
        np.save(handle, np.asarray([[999.0]]), allow_pickle=False)

    with pytest.raises((ValueError, RuntimeError), match="array|fit"):
        validate_role_neutral_tfidf_group_execution(
            root=root,
            request=request,
        )

    extra_root = (tmp_path / "extra").resolve()
    _execute(root=extra_root, request=request, loader=lambda _rows: heldout)
    (extra_root / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="closed"):
        validate_role_neutral_tfidf_group_execution(
            root=extra_root,
            request=request,
        )


def test_terminal_event_reordering_fails_closed(tmp_path: Path):
    request = _request()
    _texts, _treatment, _outcome, heldout = _inputs(request)
    root = (tmp_path / "reordered").resolve()
    _execute(root=root, request=request, loader=lambda _rows: heldout)
    path = root / "execution_manifest.json"
    terminal = json.loads(path.read_text(encoding="utf-8"))
    events = terminal["event_order"]
    cumulative = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "cumulative_fit_only_view_published"
    )
    opened = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "exact_heldout_text_opened"
    )
    events[cumulative], events[opened] = events[opened], events[cumulative]
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
    terminal["content_sha256"] = _content_sha256(terminal)
    path.write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="terminal manifest"):
        validate_role_neutral_tfidf_group_execution(
            root=root,
            request=request,
        )


def test_explicit_config_change_changes_scientific_fit_plan():
    request = _request()
    first = _config(terms_per_topic=7)
    second = deepcopy(first)
    second.architecture.multi_model_forest.tfidf_topic.terms_per_topic = 9
    first_config = _configuration(first, request=request)
    second_config = _configuration(second, request=request)

    assert first_config["tfidf_topic"]["terms_per_topic"] == 7
    assert second_config["tfidf_topic"]["terms_per_topic"] == 9
    assert (
        _scientific_fit_plan(
            request=request,
            configuration=first_config,
        )["content_sha256"]
        != _scientific_fit_plan(
            request=request,
            configuration=second_config,
        )["content_sha256"]
    )


def test_executor_has_no_heldout_label_parameter():
    parameters = set(
        __import__("inspect").signature(
            execute_role_neutral_tfidf_physical_group
        ).parameters
    )
    assert "heldout_treatment" not in parameters
    assert "heldout_outcome" not in parameters
    assert "heldout_labels" not in parameters
