from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import oci.inference.production_role_neutral_producer_factories as factories_module
from oci.inference.portable_workflow_spec import ScientificWorkflowSpec
from oci.inference.production_role_neutral_producer_factories import (
    LEXICAL_SEMANTIC_RETRIEVAL,
    PreparedBuildRoleNeutralProducerFactoriesBuilder,
    RoleNeutralScientificContractError,
    _GroupInputs,
    _embedding_targets,
    _scientific_bindings,
    missing_role_neutral_architecture_profile_fields,
)
from oci.inference.production_stage1_bundle import (
    ProductionStage1BundleBuilder,
    _PreparedBuild,
    load_applied_stage1_config,
)
from oci.inference.production_stage1_role_neutral_execution import (
    RoleNeutralComponentInvocation,
)
from oci.inference.production_stage1_scope_scheduler import (
    build_canonical_stage1_scope_plan,
)
from oci.inference.role_neutral_bow_group_execution import (
    AuthenticatedRoleNeutralBoWNuisanceBank,
)
from oci.inference.role_neutral_embedding_group_execution import (
    EmbeddingContrastSpec,
    _contrast_geometry,
)
from oci.inference.review_spent_evidence_provider import (
    semantic_witness_config_from_portable_scientific_spec,
)


ROOT = Path(__file__).resolve().parents[1]
PORTABLE = (
    ROOT / "example_configs/portable_all_evidence_scientific_nsclc.json"
)
STAGE1 = ROOT / "example_configs/production_all_evidence_stage1_full.json"
QUERY = (
    ROOT / "example_configs/production_all_evidence_neural_query_full.json"
)


def _portable_raw() -> dict:
    return json.loads(PORTABLE.read_text(encoding="utf-8"))


def _registry() -> dict:
    row_count = 30
    rows = tuple(range(row_count))
    outer_folds = []
    for outer_fold in range(1, 3):
        start = (outer_fold - 1) * 15
        heldout = tuple(range(start, start + 15))
        fit = tuple(row for row in rows if row not in set(heldout))
        partitions = tuple(fit[index::5] for index in range(5))
        outer_folds.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": list(fit),
                "heldout_row_ids": list(heldout),
                "inner_folds": [
                    {
                        "inner_fold": index,
                        "fit_row_ids": [
                            row
                            for row in fit
                            if row not in set(partition)
                        ],
                        "heldout_row_ids": list(partition),
                    }
                    for index, partition in enumerate(
                        partitions,
                        start=1,
                    )
                ],
            }
        )
    return {
        "dataset_row_count": row_count,
        "outer_folds": outer_folds,
    }


def _plan():
    return build_canonical_stage1_scope_plan(
        registry=_registry(),
        registry_content_sha256="a" * 64,
        global_seed=42,
        gpu_ids=(),
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=2,
        expected_inner_fold_count=5,
    )


def _unchecked_bank(owner) -> AuthenticatedRoleNeutralBoWNuisanceBank:
    bank = object.__new__(AuthenticatedRoleNeutralBoWNuisanceBank)
    values = {
        "plan_scientific_content_sha256": "a" * 64,
        "physical_owner_scope_id": owner.scope_id,
        "fit_row_ids": owner.fit_row_ids,
        "heldout_row_ids": owner.heldout_row_ids,
        "fit_propensity_probability": tuple(
            0.2 + 0.05 * (index % 4)
            for index in range(owner.fit_row_count)
        ),
        "fit_outcome_nuisance_probability": tuple(
            0.3 + 0.05 * (index % 3)
            for index in range(owner.fit_row_count)
        ),
        # The factory never reads these label-free held-out predictions when
        # constructing fit-side embedding targets.
        "heldout_propensity_probability": ("sealed-heldout",),
        "heldout_outcome_nuisance_probability": ("sealed-heldout",),
        "source_terminal_content_sha256": "b" * 64,
        "fit_state_artifact_sha256": "c" * 64,
        "content_sha256": "d" * 64,
    }
    for name, value in values.items():
        object.__setattr__(bank, name, value)
    return bank


def test_real_portable_profile_is_complete_and_capacity_is_scientific() -> None:
    raw = _portable_raw()
    profiles = raw["architecture_profiles"]

    assert missing_role_neutral_architecture_profile_fields(profiles) == ()
    assert raw["stage2_prompt_protocol"][
        "final_upstream_meta_inner_folds"
    ] == (
        raw["folds"]["initial_training_partitions"]
        + raw["folds"]["review_rounds"]
    )
    baseline = ScientificWorkflowSpec.from_mapping(raw).scientific_sha256

    changed = copy.deepcopy(raw)
    changed["architecture_profiles"]["whole_cohort_embeddings"][
        "producer_configuration"
    ]["maximum_source_chunks_per_row"] = 1000
    assert (
        ScientificWorkflowSpec.from_mapping(changed).scientific_sha256
        != baseline
    )

    missing = copy.deepcopy(raw)
    del missing["architecture_profiles"]["whole_cohort_embeddings"][
        "producer_configuration"
    ]["maximum_semantic_terms"]
    assert (
        "architecture_profiles.whole_cohort_embeddings."
        "producer_configuration.maximum_semantic_terms"
        in missing_role_neutral_architecture_profile_fields(
            missing["architecture_profiles"]
        )
    )


def test_real_portable_profile_matches_authenticated_source_profiles() -> None:
    raw = _portable_raw()
    config = load_applied_stage1_config(
        STAGE1,
        require_explicit_scientific_fields=True,
    )
    query, _ = ProductionStage1BundleBuilder._load_query_config(QUERY)
    prepared = SimpleNamespace(
        config=config,
        query_config=query,
        options=SimpleNamespace(query_nuisance_folds=3),
        htr_model_sha256="a" * 64,
        semantic_witness_scientific_config=(
            semantic_witness_config_from_portable_scientific_spec(raw)
        ),
    )

    bindings = _scientific_bindings(
        prepared=prepared,
        profiles=raw["architecture_profiles"],
    )

    assert bindings.htr.require_live_unfrozen_encoder_attestation is True
    assert bindings.embedding.maximum_source_chunks_per_row is None
    assert bindings.embedding.maximum_retrieval_chunks_per_side is None
    assert bindings.embedding.maximum_semantic_terms is None
    assert len(bindings.embedding.contrasts) == 9
    assert (
        bindings.semantic_witness.identity_sha256
        == prepared.semantic_witness_scientific_config.identity_sha256
    )


def test_lexical_profile_must_equal_prepared_portable_configuration() -> None:
    raw = _portable_raw()
    config = load_applied_stage1_config(
        STAGE1,
        require_explicit_scientific_fields=True,
    )
    query, _ = ProductionStage1BundleBuilder._load_query_config(QUERY)
    prepared = SimpleNamespace(
        config=config,
        query_config=query,
        options=SimpleNamespace(query_nuisance_folds=3),
        htr_model_sha256="a" * 64,
        semantic_witness_scientific_config=(
            semantic_witness_config_from_portable_scientific_spec(raw)
        ),
    )
    changed = copy.deepcopy(raw["architecture_profiles"])
    changed[LEXICAL_SEMANTIC_RETRIEVAL][
        "producer_configuration"
    ]["retrieval_min_positive_documents"] += 1

    with pytest.raises(
        RoleNeutralScientificContractError,
        match="differs from the authenticated portable Stage 1 request",
    ):
        _scientific_bindings(prepared=prepared, profiles=changed)


def test_live_unfrozen_attestation_cannot_fall_back_to_a_default(
    tmp_path: Path,
) -> None:
    raw = json.loads(STAGE1.read_text(encoding="utf-8"))
    del raw["config"]["architecture"][
        "htr_require_live_unfrozen_encoder_attestation"
    ]
    path = tmp_path / "missing-live-attestation.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            "architecture is missing: "
            "htr_require_live_unfrozen_encoder_attestation"
        ),
    ):
        load_applied_stage1_config(
            path,
            require_explicit_scientific_fields=True,
        )


@pytest.mark.parametrize(
    ("section", "field_name"),
    (
        ("architecture", "htr_transformer_activation"),
        ("training", "adamw_beta1"),
        (
            "multi_model_forest",
            "matched_pair_bow_optimizer_maxls",
        ),
    ),
)
def test_htr_matched_scientific_controls_cannot_inherit_defaults(
    tmp_path: Path,
    section: str,
    field_name: str,
) -> None:
    raw = json.loads(STAGE1.read_text(encoding="utf-8"))
    if section == "architecture":
        target = raw["config"]["architecture"]
    elif section == "training":
        target = raw["config"]["training"]
    else:
        target = raw["config"]["architecture"]["multi_model_forest"]
    del target[field_name]
    path = tmp_path / f"missing-{field_name}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=field_name):
        load_applied_stage1_config(
            path,
            require_explicit_scientific_fields=True,
        )


def test_authenticated_bow_targets_use_exact_fit_side_formulas() -> None:
    owner, _members = next(iter(_plan().physical_scope_groups))
    bank = _unchecked_bank(owner)
    treatment = tuple(
        float(index % 2) for index in range(owner.fit_row_count)
    )
    outcome = tuple(
        float((index // 2) % 2)
        for index in range(owner.fit_row_count)
    )
    inputs = _GroupInputs(
        fit_texts=tuple("fit" for _ in treatment),
        fit_treatment=treatment,
        fit_outcome=outcome,
        heldout_texts=("heldout-text-only",),
    )
    sources = {
        "cell": "fit_treatment_outcome_cell_code",
        "r": "fit_r_pseudo_from_authenticated_bow_nuisance",
        "orthogonal": (
            "fit_orthogonal_r_score_from_authenticated_bow_nuisance"
        ),
        "weight": (
            "fit_treatment_residual_squared_from_authenticated_bow_nuisance"
        ),
    }

    targets = _embedding_targets(
        inputs=inputs,
        target_sources=sources,
        nuisance_bank=bank,
        fit_row_ids=owner.fit_row_ids,
    )

    treatment_array = np.asarray(treatment)
    outcome_array = np.asarray(outcome)
    propensity = np.asarray(bank.fit_propensity_probability)
    outcome_nuisance = np.asarray(
        bank.fit_outcome_nuisance_probability
    )
    t_residual = treatment_array - propensity
    y_residual = outcome_array - outcome_nuisance
    np.testing.assert_array_equal(
        targets["cell"],
        2.0 * treatment_array + outcome_array,
    )
    np.testing.assert_allclose(targets["r"], y_residual / t_residual)
    np.testing.assert_allclose(
        targets["orthogonal"],
        y_residual * t_residual,
    )
    np.testing.assert_allclose(targets["weight"], t_residual**2)
    assert set(_GroupInputs.__dataclass_fields__) == {
        "fit_texts",
        "fit_treatment",
        "fit_outcome",
        "heldout_texts",
    }


def test_residualized_cell_direction_matches_legacy_formula() -> None:
    cells = np.repeat(np.arange(4, dtype=float), 2)
    embeddings = np.asarray(
        [
            [0.1, 0.2, 0.8],
            [0.2, 0.4, 0.7],
            [0.7, 0.3, 0.2],
            [0.8, 0.2, 0.1],
            [0.3, 0.9, 0.4],
            [0.4, 0.8, 0.5],
            [0.9, 0.7, 0.9],
            [0.8, 0.6, 0.8],
        ],
        dtype=float,
    )
    contrast = EmbeddingContrastSpec(
        name="residualized",
        contrast_family=(
            "residualized_treatment_outcome_cell_interaction"
        ),
        target_name="cell",
        sample_weight_target_name=None,
        split_rule=(
            "cell_difference_in_differences_residualized_from_marginals"
        ),
    )
    config = SimpleNamespace(
        contrasts=(contrast,),
        direction_norm_epsilon=1e-10,
        lstsq_rcond=None,
        minimum_contrast_side_rows=2,
        pseudo_target_quantile=0.2,
        vector_norm_order="l2",
    )

    _groups, _coefficients, directions = _contrast_geometry(
        target_matrix=cells[:, None],
        patient_embeddings=embeddings,
        config=config,
    )

    mean = {
        cell: embeddings[cells == cell].mean(axis=0)
        for cell in range(4)
    }
    raw = mean[3] - mean[2] - mean[1] + mean[0]
    treatment_direction = embeddings[cells >= 2].mean(axis=0) - embeddings[
        cells < 2
    ].mean(axis=0)
    outcome_direction = embeddings[(cells % 2) == 1].mean(
        axis=0
    ) - embeddings[(cells % 2) == 0].mean(axis=0)
    basis = np.stack(
        [
            treatment_direction / np.linalg.norm(treatment_direction),
            outcome_direction / np.linalg.norm(outcome_direction),
        ],
        axis=1,
    )
    projection, *_ = np.linalg.lstsq(basis, raw, rcond=None)
    legacy = raw - basis @ projection
    legacy /= np.linalg.norm(legacy)
    np.testing.assert_allclose(directions[0], legacy, atol=1e-12)


def test_builder_binds_matched_and_embedding_to_prior_bow_without_heldout_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    owner, members = next(iter(plan.physical_scope_groups))
    bank = _unchecked_bank(owner)
    frame = pd.DataFrame(
        {
            "text": [f"complete note {row}" for row in range(30)],
            "treatment": pd.Series(["SEALED"] * 30, dtype=object),
            "outcome": pd.Series(["SEALED"] * 30, dtype=object),
        }
    )
    for index, row in enumerate(owner.fit_row_ids):
        frame.loc[row, "treatment"] = float(index % 2)
        frame.loc[row, "outcome"] = float((index // 2) % 2)

    class Cache:
        def __init__(self):
            self.calls = []

        def bind_spent(self, row_ids, texts):
            self.calls.append((tuple(row_ids), tuple(texts)))
            return SimpleNamespace(row_ids=tuple(row_ids))

    class StateBundle:
        states = {
            scope.scope_id: object()
            for scope in plan.physical_scopes
        }

        def manifest_path_for_owner(self, scope_id):
            return tmp_path / f"{scope_id}.cluster-state.json"

    prepared = object.__new__(_PreparedBuild)
    prepared.cluster_preflight_manifest_path = (
        tmp_path / "preflight.json"
    )
    prepared.cluster_preflight_state_bundle = StateBundle()
    prepared.config = SimpleNamespace(
        text_column="text",
        treatment_column="treatment",
        outcome_column="outcome",
    )
    prepared.registry = {}
    prepared.registry_content_sha256 = "a" * 64
    prepared.embedding_cache_identity = {}
    prepared.stage1_scope_plan = plan
    prepared.modeling_data = frame
    prepared.htr_model_path = tmp_path / "model"
    prepared.htr_model_sha256 = "e" * 64
    prepared.embedding_cache = Cache()

    bindings = SimpleNamespace(
        bow_views=(),
        matched_pair=object(),
        embedding=object(),
        embedding_target_sources={
            "cell": "fit_treatment_outcome_cell_code",
            "r": "fit_r_pseudo_from_authenticated_bow_nuisance",
            "orthogonal": (
                "fit_orthogonal_r_score_from_authenticated_bow_nuisance"
            ),
            "weight": (
                "fit_treatment_residual_squared_from_authenticated_bow_nuisance"
            ),
        },
    )
    bank_loads = []
    matched_calls = []
    embedding_calls = []

    monkeypatch.setattr(
        factories_module,
        "_scientific_bindings",
        lambda **_kwargs: bindings,
    )
    monkeypatch.setattr(
        factories_module,
        "load_production_stage1_cluster_preflight_artifact",
        lambda **_kwargs: object(),
    )

    def load_bank(*, root, request):
        bank_loads.append((Path(root), request))
        return bank

    monkeypatch.setattr(
        factories_module,
        "load_authenticated_role_neutral_bow_nuisance_bank",
        load_bank,
    )
    monkeypatch.setattr(
        factories_module,
        "_htr_extractor_factory",
        lambda **_kwargs: object(),
    )

    def execute_matched(**kwargs):
        matched_calls.append(kwargs)
        assert kwargs["exact_heldout_text_loader"](
            owner.heldout_row_ids
        ) == tuple(frame.loc[list(owner.heldout_row_ids), "text"])
        return {"matched": True}

    monkeypatch.setattr(
        factories_module,
        "execute_role_neutral_matched_pair_from_bow_nuisance_bank",
        execute_matched,
    )
    monkeypatch.setattr(
        factories_module,
        "authenticate_role_neutral_matched_pair_component",
        lambda **_kwargs: "matched-receipt",
    )

    class FakeBatch:
        def __init__(self, *, row_ids, texts, embedding_provider):
            self.row_ids = row_ids
            self.texts = texts
            self.embedding_provider = embedding_provider

    monkeypatch.setattr(
        factories_module,
        "ExactHeldoutEmbeddingBatch",
        FakeBatch,
    )

    def execute_embedding(**kwargs):
        embedding_calls.append(kwargs)
        batch = kwargs["exact_heldout_loader"](
            owner.heldout_row_ids
        )
        assert batch.texts == tuple(
            frame.loc[list(owner.heldout_row_ids), "text"]
        )
        return {"embedding": True}

    monkeypatch.setattr(
        factories_module,
        "execute_role_neutral_embedding_physical_group",
        execute_embedding,
    )
    monkeypatch.setattr(
        factories_module,
        "authenticate_role_neutral_embedding_component",
        lambda **_kwargs: "embedding-receipt",
    )

    builder = PreparedBuildRoleNeutralProducerFactoriesBuilder(
        architecture_profiles=_portable_raw()["architecture_profiles"],
        runtime_compatibility_class="test-runtime",
    )
    factories = builder(prepared)
    assert tuple(factories.as_mapping()) == (
        "bow",
        "htr",
        "matched_pair",
        "embeddings",
        "tfidf",
        "neural_query",
    )
    parent = tmp_path / "components"
    matched_invocation = RoleNeutralComponentInvocation(
        plan=plan,
        physical_owner=owner,
        logical_members=members,
        component="matched_pair",
        output_root=parent / "matched_pair",
        resource="cpu",
    )
    embedding_invocation = RoleNeutralComponentInvocation(
        plan=plan,
        physical_owner=owner,
        logical_members=members,
        component="embeddings",
        output_root=parent / "embeddings",
        resource="cpu",
    )

    matched = factories.matched_pair(matched_invocation)
    assert matched.execute() == {"matched": True}
    assert matched.authenticate() == "matched-receipt"
    embedding = factories.embeddings(embedding_invocation)
    assert embedding.execute() == {"embedding": True}
    assert embedding.authenticate() == "embedding-receipt"

    assert [row[0] for row in bank_loads] == [
        parent / "bow",
        parent / "bow",
    ]
    assert matched_calls[0]["nuisance_bank"] is bank
    assert embedding_calls[0]["fit_targets"]["r"].shape == (
        owner.fit_row_count,
    )
    forbidden = {
        "heldout_treatment",
        "heldout_outcome",
        "heldout_labels",
    }
    assert forbidden.isdisjoint(matched_calls[0])
    assert forbidden.isdisjoint(embedding_calls[0])
