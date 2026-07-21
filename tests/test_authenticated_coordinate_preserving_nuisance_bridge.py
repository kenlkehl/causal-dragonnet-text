import hashlib

import numpy as np
import pytest

from oci.inference.all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from oci.inference.authenticated_coordinate_preserving_nuisance_bridge import (
    coordinate_preserving_nuisance_contract_sha256,
    coordinate_preserving_nuisance_schema,
    derive_exact_nuisance_from_coordinate_preserved_stage1,
    precommit_runtime_producer_identity_sha256,
)
from oci.inference.context_fit_upstream_gate_provider import ContextFitUpstreamPrediction
from oci.inference.coordinate_preserving_context_fit_upstream_backend import (
    CoordinatePreservingContextFitUpstreamBackend,
    CoordinatePreservingUpstreamSchemaConfig,
    PrecommittedExactCalibratedSource,
    PrecommittedNamedRawCoordinate,
    PrecommittedVolatileRawFeatureFamily,
)
from oci.inference.final_context_fit_causal_forest_adapter import (
    SealedFinalForestExplicitBlock,
    prepare_final_causal_forest_design,
)
from oci.inference.final_context_fit_r_stack_adapter import (
    EXACT_OUTCOME_PREDICTION,
    EXACT_PROPENSITY_PREDICTION,
)
from oci.inference.final_context_fit_upstream_bank import FinalContextFitUpstreamProducer

_VIEWS = (
    "linear_unigram_c0p5",
    "linear_1_2",
    "linear_1_3",
    "linear_2_4_min_df3",
    "extratrees_1_3",
    "random_forest_1_2",
)


class _ExactCoordinateBackend:
    def __init__(self, *, corruption="none"):
        self.corruption = corruption

    def identity(self):
        return {
            "backend": "test_exact_coordinate_stage1_v1",
            "corruption": self.corruption,
        }

    def fit_predict(self, **kwargs):
        row_ids = kwargs["gate_row_ids"]
        rows = np.asarray(row_ids, dtype=float)
        schema = [dict(item) for item in coordinate_preserving_nuisance_schema(_VIEWS)]
        values = []
        for index, item in enumerate(schema):
            role = item["consumer_role"]
            base = 0.12 if role == PROPENSITY_NUISANCE_FEATURE_ROLE else 0.42
            values.append(base + 0.015 * index + (rows % 7.0) * 0.001)

        if self.corruption == "renamed":
            schema[0]["feature_name"] += "_renamed"
        elif self.corruption == "wrong_role":
            schema[0]["consumer_role"] = OUTCOME_NUISANCE_FEATURE_ROLE
        elif self.corruption == "extra_bow":
            schema.insert(
                6,
                {
                    "feature_name": "stage1_raw__bow__extra__treatment_pred__as_propensity",
                    "feature_kind": "bow_nuisance",
                    "consumer_role": PROPENSITY_NUISANCE_FEATURE_ROLE,
                },
            )
            values.insert(6, np.full(len(rows), 0.22, dtype=float))
        elif self.corruption == "invalid_member_probability":
            # The six-member mean remains in (0, 1), so this specifically tests
            # member-level validation rather than the sealed output check.
            values[0] = np.full(len(rows), 1.2, dtype=float)

        schema.append(
            {
                "feature_name": "unrelated_effect_modifier",
                "feature_kind": "embedding_clustered",
                "consumer_role": UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            }
        )
        values.append(np.sin(rows * 0.01))
        return ContextFitUpstreamPrediction(
            gate_row_ids=row_ids,
            calibrated_source_names=("one_calibrated_tau",),
            calibrated_source_kinds=("nested_calibrated_bow_weighted_r",),
            calibrated_source_values=np.zeros((len(rows), 1), dtype=float),
            feature_names=tuple(item["feature_name"] for item in schema),
            feature_kinds=tuple(item["feature_kind"] for item in schema),
            feature_roles=tuple(item["consumer_role"] for item in schema),
            feature_values=np.column_stack(values),
        )


def _configured_child_schema(corruption):
    schema = [dict(item) for item in coordinate_preserving_nuisance_schema(_VIEWS)]
    if corruption == "renamed":
        schema[0]["feature_name"] += "_renamed"
    elif corruption == "wrong_role":
        schema[0]["consumer_role"] = OUTCOME_NUISANCE_FEATURE_ROLE
    elif corruption == "extra_bow":
        schema.insert(
            6,
            {
                "feature_name": "stage1_raw__bow__extra__treatment_pred__as_propensity",
                "feature_kind": "bow_nuisance",
                "consumer_role": PROPENSITY_NUISANCE_FEATURE_ROLE,
            },
        )
    return schema


def _build(tmp_path, *, corruption="none"):
    child_schema = _configured_child_schema(corruption)
    stable = CoordinatePreservingContextFitUpstreamBackend(
        _ExactCoordinateBackend(corruption=corruption),
        config=CoordinatePreservingUpstreamSchemaConfig(
            namespace="all_evidence_upstream",
            calibrated_sources=(
                PrecommittedExactCalibratedSource(
                    child_name="one_calibrated_tau",
                    source_kind="nested_calibrated_bow_weighted_r",
                ),
            ),
            named_raw_coordinates=tuple(
                PrecommittedNamedRawCoordinate(
                    child_name=item["feature_name"],
                    source_kind=item["feature_kind"],
                    consumer_role=item["consumer_role"],
                )
                for item in child_schema
            ),
            volatile_raw_families=(
                PrecommittedVolatileRawFeatureFamily(
                    source_kind="embedding_clustered",
                    consumer_role=UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                    signed_order_width=1,
                ),
            ),
        ),
    )
    producer = FinalContextFitUpstreamProducer(
        tmp_path / f"cache_{corruption}",
        backend=stable,
    )
    producer_sha = precommit_runtime_producer_identity_sha256(producer)
    contract_sha = coordinate_preserving_nuisance_contract_sha256(_VIEWS)
    train_rows = (101, 102, 103, 104, 105, 106)
    heldout_rows = (901, 902)
    package = producer.produce(
        outer_fold=3,
        outer_train_row_ids=train_rows,
        outer_train_texts=tuple(f"train {row}" for row in train_rows),
        outer_train_treatment=np.asarray([0, 1, 0, 1, 0, 1], dtype=float),
        outer_train_outcome=np.asarray([0, 0, 1, 1, 0, 1], dtype=float),
        outer_heldout_row_ids=heldout_rows,
        outer_heldout_texts=tuple(f"heldout {row}" for row in heldout_rows),
        meta_inner_fold_ids=(1, 2, 3, 1, 2, 3),
    )
    return package, producer, producer_sha, contract_sha


def _derive(package, producer, producer_sha, contract_sha):
    return derive_exact_nuisance_from_coordinate_preserved_stage1(
        package,
        runtime_producer=producer,
        bow_view_names=_VIEWS,
        precommitted_producer_identity_sha256=producer_sha,
        precommitted_coordinate_contract_sha256=contract_sha,
    )


def test_v3_derives_six_bow_means_and_htr_singletons_with_lineage(tmp_path):
    package, producer, producer_sha, contract_sha = _build(tmp_path)
    result = _derive(package, producer, producer_sha, contract_sha)
    raw = package.raw_features

    def ordered_mean(start, stop):
        expected = np.zeros(len(raw.train_row_ids), dtype=np.float64)
        for index in range(start, stop):
            expected = np.add(expected, raw.train_oof_values[:, index])
        return np.multiply(expected, np.float64(1.0) / np.float64(6))

    np.testing.assert_array_equal(
        result.nuisance.train_oof_values[:, 0],
        ordered_mean(0, 6),
    )
    np.testing.assert_array_equal(
        result.nuisance.train_oof_values[:, 1], raw.train_oof_values[:, 6]
    )
    np.testing.assert_array_equal(
        result.nuisance.train_oof_values[:, 2],
        ordered_mean(7, 13),
    )
    np.testing.assert_array_equal(
        result.nuisance.train_oof_values[:, 3], raw.train_oof_values[:, 13]
    )
    assert result.nuisance.prediction_semantics == (
        EXACT_PROPENSITY_PREDICTION,
        EXACT_PROPENSITY_PREDICTION,
        EXACT_OUTCOME_PREDICTION,
        EXACT_OUTCOME_PREDICTION,
    )
    assert [len(item["source_coordinates"]) for item in result.output_records] == [6, 1, 6, 1]
    assert [
        item["source_coordinates"][0]["raw_column_index"] for item in result.output_records
    ] == [0, 6, 7, 13]
    assert (
        result.output_records[0]["arithmetic"]["weight_float64_hex"]
        == float(np.float64(1.0) / np.float64(6)).hex()
    )

    # The mean retains all six parents rather than borrowing one member's
    # lineage; its recursive fit rows remain exactly complement-only.
    first_mean_lineage = result.nuisance.train_oof_fit_row_provenance[0][0]
    assert len(first_mean_lineage.upstream) == 6
    assert first_mean_lineage.recursive_fit_row_ids() == frozenset({102, 103, 105, 106})
    audit = result.audit_record()
    assert audit["source_lineages_bound"] is True
    assert audit["semantic_inference_from_feature_names"] is False
    assert audit["package_only_derivation_supported"] is False
    result.verify_authenticated_content(package, runtime_producer=producer)


def test_v3_extension_enters_existing_final_forest_adapter_without_relabelling(tmp_path):
    package, producer, producer_sha, contract_sha = _build(tmp_path)
    result = _derive(package, producer, producer_sha, contract_sha)
    source = package.calibrated_sources
    explicit = SealedFinalForestExplicitBlock.seal_for_package(
        package,
        effect_names=(),
        control_names=(),
        effect_train_values=np.empty((len(source.train_row_ids), 0), dtype=float),
        effect_heldout_values=np.empty((len(source.heldout_row_ids), 0), dtype=float),
        control_train_values=np.empty((len(source.train_row_ids), 0), dtype=float),
        control_heldout_values=np.empty((len(source.heldout_row_ids), 0), dtype=float),
    )
    design = prepare_final_causal_forest_design(
        package,
        exact_nuisance=result.nuisance,
        explicit_features=explicit,
    )
    raw_control_count = sum(
        role in {PROPENSITY_NUISANCE_FEATURE_ROLE, OUTCOME_NUISANCE_FEATURE_ROLE}
        for role in package.raw_features.consumer_roles
    )
    np.testing.assert_array_equal(
        design.control_train_values[:, raw_control_count : raw_control_count + 4],
        result.nuisance.train_oof_values,
    )
    assert design.routing_audit["control_columns"]["exact_nuisance_prediction_count"] == 4
    assert design.routing_audit["exact_nuisance_routed_as_fixed_causal_forest_nuisance"] is False
    assert design.routing_audit["exact_nuisance_routed_as_control_covariates"] is True


@pytest.mark.parametrize("corruption", ["renamed", "wrong_role", "extra_bow"])
def test_v3_fails_closed_on_inexact_complete_coordinate_groups(tmp_path, corruption):
    package, producer, producer_sha, contract_sha = _build(tmp_path, corruption=corruption)
    with pytest.raises(ValueError, match="exact fourteen BoW/HTR nuisance coordinates"):
        _derive(package, producer, producer_sha, contract_sha)


def test_v3_rejects_invalid_member_even_when_mean_would_be_valid(tmp_path):
    package, producer, producer_sha, contract_sha = _build(
        tmp_path, corruption="invalid_member_probability"
    )
    with pytest.raises(ValueError, match="source propensity coordinates"):
        _derive(package, producer, producer_sha, contract_sha)


def test_v3_requires_both_precommitments_and_matching_live_producer(tmp_path):
    package, producer, producer_sha, contract_sha = _build(tmp_path)
    bad_sha = hashlib.sha256(b"not the producer").hexdigest()
    with pytest.raises(ValueError, match="producer identities must match"):
        _derive(package, producer, bad_sha, contract_sha)

    with pytest.raises(ValueError, match="coordinate contract"):
        _derive(package, producer, producer_sha, hashlib.sha256(b"bad contract").hexdigest())

    other = FinalContextFitUpstreamProducer(
        tmp_path / "other_cache", backend=_ExactCoordinateBackend(corruption="renamed")
    )
    with pytest.raises(ValueError, match="producer identities must match"):
        derive_exact_nuisance_from_coordinate_preserved_stage1(
            package,
            runtime_producer=other,
            bow_view_names=_VIEWS,
            precommitted_producer_identity_sha256=producer_sha,
            precommitted_coordinate_contract_sha256=contract_sha,
        )


def test_contract_schema_is_exactly_six_plus_one_per_target():
    schema = coordinate_preserving_nuisance_schema(_VIEWS)
    assert len(schema) == 14
    assert [item["feature_kind"] for item in schema].count("bow_nuisance") == 12
    assert [item["feature_kind"] for item in schema].count("htr_nuisance") == 2
    with pytest.raises(ValueError, match="exactly six"):
        coordinate_preserving_nuisance_schema(_VIEWS[:5])
