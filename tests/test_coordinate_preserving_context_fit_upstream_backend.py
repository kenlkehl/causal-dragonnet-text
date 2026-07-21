from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from oci.inference.all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from oci.inference.context_fit_upstream_gate_provider import ContextFitUpstreamPrediction
from oci.inference.coordinate_preserving_context_fit_upstream_backend import (
    COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID,
    CoordinatePreservingContextFitUpstreamBackend,
    CoordinatePreservingUpstreamSchemaConfig,
    PrecommittedExactCalibratedSource,
    PrecommittedNamedRawCoordinate,
    PrecommittedVolatileRawFeatureFamily,
)
from oci.inference.final_context_fit_upstream_bank import FinalContextFitUpstreamProducer


class _HybridChildBackend:
    def __init__(
        self,
        *,
        optional_coordinate: bool = True,
        optional_family: bool = True,
        omit_required_coordinate: bool = False,
        omit_required_family: bool = False,
        extra_raw: bool = False,
        omit_source: bool = False,
        extra_source: bool = False,
        wrong_optional_metadata: bool = False,
        extra_volatile_member: bool = False,
        bad_volatile_name: bool = False,
    ) -> None:
        self.optional_coordinate = optional_coordinate
        self.optional_family = optional_family
        self.omit_required_coordinate = omit_required_coordinate
        self.omit_required_family = omit_required_family
        self.extra_raw = extra_raw
        self.omit_source = omit_source
        self.extra_source = extra_source
        self.wrong_optional_metadata = wrong_optional_metadata
        self.extra_volatile_member = extra_volatile_member
        self.bad_volatile_name = bad_volatile_name
        self.calls: list[dict[str, object]] = []

    def identity(self):
        return {
            "backend": "hybrid_coordinate_test_backend_v1",
            "optional_coordinate": self.optional_coordinate,
            "optional_family": self.optional_family,
            "omit_required_coordinate": self.omit_required_coordinate,
            "omit_required_family": self.omit_required_family,
            "extra_raw": self.extra_raw,
            "omit_source": self.omit_source,
            "extra_source": self.extra_source,
            "wrong_optional_metadata": self.wrong_optional_metadata,
            "extra_volatile_member": self.extra_volatile_member,
            "bad_volatile_name": self.bad_volatile_name,
        }

    def fit_predict(self, **kwargs):
        self.calls.append(dict(kwargs))
        call_number = len(self.calls)
        rows = tuple(kwargs["gate_row_ids"])
        row_values = np.asarray(rows, dtype=float)

        source_records = []
        if not self.omit_source:
            source_records.extend(
                [
                    ("source_a", "calibrated_a", row_values / 100.0),
                    ("source_b", "calibrated_b", row_values / 200.0),
                ]
            )
        if self.extra_source:
            source_records.append(("source_extra", "calibrated_extra", row_values / 300.0))
        if call_number % 2 == 0:
            source_records.reverse()

        feature_records = []
        if not self.omit_required_coordinate:
            feature_records.append(
                (
                    "stable_named_signal",
                    "mixed_raw_family",
                    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                    row_values * 10.0,
                )
            )
        if not self.omit_required_family:
            volatile = [
                (
                    (
                        f"unexpected_name_{call_number}"
                        if self.bad_volatile_name
                        else f"fit_local_alpha_{call_number}"
                    ),
                    "mixed_raw_family",
                    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                    row_values,
                ),
                (
                    f"fit_local_beta_{call_number}",
                    "mixed_raw_family",
                    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                    -row_values / 2.0,
                ),
            ]
            if self.extra_volatile_member:
                volatile.append(
                    (
                        f"fit_local_gamma_{call_number}",
                        "mixed_raw_family",
                        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                        row_values / 3.0,
                    )
                )
            if call_number % 2 == 0:
                volatile.reverse()
            feature_records.extend(volatile)
        if self.optional_coordinate:
            feature_records.append(
                (
                    "optional_named_signal",
                    (
                        "mixed_raw_family"
                        if self.wrong_optional_metadata
                        else "optional_named_family"
                    ),
                    (
                        UNCALIBRATED_EFFECT_MODIFIER_ROLE
                        if self.wrong_optional_metadata
                        else OUTCOME_NUISANCE_FEATURE_ROLE
                    ),
                    np.zeros(len(rows), dtype=float),
                )
            )
        if self.optional_family:
            feature_records.append(
                (
                    f"fit_local_optional_{call_number}",
                    "optional_volatile_family",
                    PROPENSITY_NUISANCE_FEATURE_ROLE,
                    np.zeros(len(rows), dtype=float),
                )
            )
        if self.extra_raw:
            feature_records.append(
                (
                    "unconfigured_raw_column",
                    "unconfigured_family",
                    OUTCOME_NUISANCE_FEATURE_ROLE,
                    row_values,
                )
            )
        if call_number % 2 == 0:
            feature_records.reverse()

        return ContextFitUpstreamPrediction(
            gate_row_ids=rows,
            calibrated_source_names=tuple(item[0] for item in source_records),
            calibrated_source_kinds=tuple(item[1] for item in source_records),
            calibrated_source_values=(
                np.column_stack([item[2] for item in source_records])
                if source_records
                else np.empty((len(rows), 0), dtype=float)
            ),
            feature_names=tuple(item[0] for item in feature_records),
            feature_kinds=tuple(item[1] for item in feature_records),
            feature_roles=tuple(item[2] for item in feature_records),
            feature_values=(
                np.column_stack([item[3] for item in feature_records])
                if feature_records
                else np.empty((len(rows), 0), dtype=float)
            ),
        )


def _config() -> CoordinatePreservingUpstreamSchemaConfig:
    return CoordinatePreservingUpstreamSchemaConfig(
        namespace="hybrid",
        calibrated_sources=(
            PrecommittedExactCalibratedSource(
                child_name="source_b",
                source_kind="calibrated_b",
                output_name="aligned_source_b",
            ),
            PrecommittedExactCalibratedSource(
                child_name="source_a",
                source_kind="calibrated_a",
                output_name="aligned_source_a",
            ),
        ),
        named_raw_coordinates=(
            PrecommittedNamedRawCoordinate(
                child_name="stable_named_signal",
                source_kind="mixed_raw_family",
                consumer_role=UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                output_name="preserved_named_signal",
            ),
            PrecommittedNamedRawCoordinate(
                child_name="optional_named_signal",
                source_kind="optional_named_family",
                consumer_role=OUTCOME_NUISANCE_FEATURE_ROLE,
                output_name="preserved_optional_signal",
                required=False,
            ),
        ),
        volatile_raw_families=(
            PrecommittedVolatileRawFeatureFamily(
                source_kind="mixed_raw_family",
                consumer_role=UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                signed_order_width=2,
                child_name_pattern=r"fit_local_(?:alpha|beta|gamma)_[0-9]+",
            ),
            PrecommittedVolatileRawFeatureFamily(
                source_kind="optional_volatile_family",
                consumer_role=PROPENSITY_NUISANCE_FEATURE_ROLE,
                signed_order_width=1,
                required=False,
            ),
        ),
        source_config_sha256="a" * 64,
    )


def _call(backend, *, gate_rows=(8, 9)):
    return backend.fit_predict(
        outer_fold=2,
        context_row_ids=(1, 2, 3, 4),
        context_texts=("a", "b", "c", "d"),
        context_treatment=np.asarray([0.0, 1.0, 0.0, 1.0]),
        context_outcome=np.asarray([0.1, 0.8, 0.2, 0.9]),
        gate_row_ids=tuple(gate_rows),
        gate_texts=tuple(f"gate {row}" for row in gate_rows),
        work_dir=Path("unused-coordinate-test-work"),
    )


def test_named_coordinates_are_preserved_and_volatile_residuals_are_summarized_once():
    child = _HybridChildBackend()
    backend = CoordinatePreservingContextFitUpstreamBackend(child, config=_config())

    first = _call(backend)
    second = _call(backend)

    assert (
        first.calibrated_source_names
        == second.calibrated_source_names
        == (
            "aligned_source_b",
            "aligned_source_a",
        )
    )
    np.testing.assert_allclose(
        first.calibrated_source_values,
        np.column_stack([np.asarray([8.0, 9.0]) / 200, np.asarray([8.0, 9.0]) / 100]),
    )
    assert (
        first.feature_names
        == second.feature_names
        == tuple(item[0] for item in _config().raw_output_schema())
    )
    np.testing.assert_allclose(first.feature_values, second.feature_values)

    # The stable coordinate remains exact even when the child column order
    # changes.  It is claimed before aggregation and therefore is not also in
    # the volatile mean/order statistics for the same kind and role.
    np.testing.assert_allclose(first.feature_values[:, 0], [80.0, 90.0])
    np.testing.assert_allclose(first.feature_values[:, 3], [2.0, 2.25])
    np.testing.assert_allclose(first.feature_values[:, 4], [8.0, 9.0])
    np.testing.assert_allclose(first.feature_values[:, 5], [8.0, 9.0])
    np.testing.assert_allclose(first.feature_values[:, 6], [-4.0, -4.5])
    assert first.feature_roles[0] == UNCALIBRATED_EFFECT_MODIFIER_ROLE
    assert first.feature_roles[1:3] == (OUTCOME_NUISANCE_FEATURE_ROLE,) * 2


def test_optional_absence_has_fixed_schema_and_is_distinct_from_observed_zero():
    present = _call(
        CoordinatePreservingContextFitUpstreamBackend(
            _HybridChildBackend(optional_coordinate=True, optional_family=True),
            config=_config(),
        )
    )
    absent = _call(
        CoordinatePreservingContextFitUpstreamBackend(
            _HybridChildBackend(optional_coordinate=False, optional_family=False),
            config=_config(),
        )
    )

    assert present.feature_names == absent.feature_names
    np.testing.assert_array_equal(present.feature_values[:, 1], np.ones(2))
    np.testing.assert_array_equal(absent.feature_values[:, 1], np.zeros(2))
    np.testing.assert_array_equal(present.feature_values[:, 2], np.zeros(2))
    np.testing.assert_array_equal(absent.feature_values[:, 2], np.zeros(2))
    np.testing.assert_array_equal(present.feature_values[:, 7], np.ones(2))
    np.testing.assert_array_equal(absent.feature_values[:, 7], np.zeros(2))
    np.testing.assert_array_equal(present.feature_values[:, 8:], np.zeros((2, 3)))
    np.testing.assert_array_equal(absent.feature_values[:, 8:], np.zeros((2, 3)))


@pytest.mark.parametrize(
    ("child_kwargs", "message"),
    [
        ({"omit_required_coordinate": True}, "missing required named raw coordinate"),
        ({"omit_required_family": True}, "missing required volatile raw feature family"),
        ({"extra_raw": True}, "unconfigured raw feature columns"),
        ({"omit_source": True}, "missing exact calibrated sources"),
        ({"extra_source": True}, "unconfigured calibrated sources"),
        ({"wrong_optional_metadata": True}, "named raw coordinate metadata changed"),
        ({"extra_volatile_member": True}, "exceeds its precommitted member capacity"),
        ({"bad_volatile_name": True}, "outside its precommitted membership pattern"),
    ],
)
def test_required_and_unconfigured_child_columns_fail_closed(child_kwargs, message):
    backend = CoordinatePreservingContextFitUpstreamBackend(
        _HybridChildBackend(**child_kwargs),
        config=_config(),
    )
    with pytest.raises(RuntimeError, match=message):
        _call(backend)


def test_identity_is_bound_and_gate_labels_are_structurally_unavailable():
    child = _HybridChildBackend()
    config = _config()
    backend = CoordinatePreservingContextFitUpstreamBackend(child, config=config)
    prediction = _call(backend)

    parameters = inspect.signature(
        CoordinatePreservingContextFitUpstreamBackend.fit_predict
    ).parameters
    assert "gate_treatment" not in parameters
    assert "gate_outcome" not in parameters
    received = child.calls[0]
    assert "gate_treatment" not in received
    assert "gate_outcome" not in received
    assert received["context_treatment"].flags.writeable is False
    assert received["context_outcome"].flags.writeable is False
    assert prediction.feature_values.shape[1] == len(config.raw_output_schema())

    identity = backend.identity()
    assert identity["backend"] == COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID
    assert identity["child"] == child.identity()
    assert identity["config"] == config.identity()
    assert identity["child_column_consumption"] == "exactly_once"
    assert identity["gate_labels_exposed_to_child"] is False
    child.extra_raw = True
    with pytest.raises(ValueError, match="child backend identity changed"):
        backend.identity()


def _final_inputs():
    return {
        "outer_fold": 3,
        "outer_train_row_ids": (10, 11, 12, 13, 14, 15),
        "outer_train_texts": tuple(f"train {row}" for row in range(10, 16)),
        "outer_train_treatment": np.asarray([0, 1, 0, 1, 0, 1], dtype=float),
        "outer_train_outcome": np.asarray([0.1, 0.8, 0.2, 0.7, 0.3, 0.9]),
        "outer_heldout_row_ids": (20, 21),
        "outer_heldout_texts": ("held 20", "held 21"),
        "meta_inner_fold_ids": (1, 1, 2, 2, 3, 3),
    }


def test_wrapper_runtime_is_recursively_authenticated_by_final_producer(tmp_path):
    child = _HybridChildBackend()
    backend = CoordinatePreservingContextFitUpstreamBackend(child, config=_config())
    producer = FinalContextFitUpstreamProducer(tmp_path / "final", backend=backend)
    producer_identity = producer.identity()

    runtime = producer_identity["backend_runtime_attestation"]
    assert runtime["class_qualname"] == "CoordinatePreservingContextFitUpstreamBackend"
    assert runtime["members"][0]["class_qualname"] == "_HybridChildBackend"

    package = producer.produce(**_final_inputs())
    assert package.calibrated_sources.source_names == (
        "aligned_source_b",
        "aligned_source_a",
    )
    assert package.raw_features.feature_names == tuple(
        item[0] for item in _config().raw_output_schema()
    )
    assert package.raw_features.train_oof_values.shape == (6, 11)
    assert package.raw_features.outer_heldout_values.shape == (2, 11)
    package.verify_authenticated_content()


def test_config_rejects_ambiguous_or_unsupported_precommitments():
    with pytest.raises(ValueError, match="unsupported"):
        PrecommittedNamedRawCoordinate(
            child_name="safe_name",
            source_kind="safe_kind",
            consumer_role="calibrated_effect",
        )
    with pytest.raises(ValueError, match="globally unique"):
        CoordinatePreservingUpstreamSchemaConfig(
            namespace="collision",
            named_raw_coordinates=(
                PrecommittedNamedRawCoordinate(
                    child_name="optional",
                    source_kind="kind_a",
                    consumer_role=OUTCOME_NUISANCE_FEATURE_ROLE,
                    required=False,
                ),
                PrecommittedNamedRawCoordinate(
                    child_name="required",
                    source_kind="kind_b",
                    consumer_role=OUTCOME_NUISANCE_FEATURE_ROLE,
                    output_name="collision__named_coordinate_001__presence",
                ),
            ),
        )
