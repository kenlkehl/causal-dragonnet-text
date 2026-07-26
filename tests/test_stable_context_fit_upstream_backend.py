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
from oci.inference.context_fit_upstream_gate_provider import (
    CompositeContextFitUpstreamBackend,
    ContextFitUpstreamPrediction,
)
from oci.inference.final_context_fit_upstream_bank import FinalContextFitUpstreamProducer
from oci.inference.stable_context_fit_upstream_backend import (
    CrossFitStableUpstreamBackend,
    CrossFitStableUpstreamSchemaConfig,
    PrecommittedCalibratedSource,
    PrecommittedRawFeatureFamily,
)


class _VaryingDiscoveryBackend:
    def __init__(
        self,
        *,
        feature_kind: str = "tfidf_topic_contrast",
        feature_role: str = UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        include_source: bool = True,
        add_third_every: int | None = None,
    ) -> None:
        self.feature_kind = feature_kind
        self.feature_role = feature_role
        self.include_source = include_source
        self.add_third_every = add_third_every
        self.calls: list[dict[str, object]] = []

    def identity(self):
        return {
            "backend": "varying_discovery_test_v1",
            "feature_kind": self.feature_kind,
            "feature_role": self.feature_role,
            "include_source": self.include_source,
            "add_third_every": self.add_third_every,
        }

    def fit_predict(self, **kwargs):
        self.calls.append(dict(kwargs))
        rows = tuple(kwargs["gate_row_ids"])
        row_values = np.asarray(rows, dtype=float)
        base = np.column_stack([row_values / 10.0, -row_values / 20.0])
        call_number = len(self.calls)
        if call_number % 2:
            names = ["discovered_topic_alpha", "discovered_topic_beta"]
            values = base
        else:
            # Both identities and column order change while the raw vector is
            # otherwise identical.
            names = ["new_local_id_902", "new_local_id_117"]
            values = base[:, ::-1]
        if self.add_third_every and call_number % self.add_third_every == 0:
            names.append("fit_local_extra_component")
            values = np.column_stack([values, row_values / 30.0])
        if self.include_source:
            source_names = ("stable_bow_r",)
            source_kinds = ("nested_calibrated_bow_r",)
            source_values = row_values[:, None] / 100.0
        else:
            source_names = ()
            source_kinds = ()
            source_values = np.empty((len(rows), 0), dtype=float)
        return ContextFitUpstreamPrediction(
            gate_row_ids=rows,
            calibrated_source_names=source_names,
            calibrated_source_kinds=source_kinds,
            calibrated_source_values=source_values,
            feature_names=tuple(names),
            feature_kinds=tuple(self.feature_kind for _ in names),
            feature_roles=tuple(self.feature_role for _ in names),
            feature_values=values,
        )


def _config(
    *,
    namespace: str = "topics",
    kind: str = "tfidf_topic_contrast",
    role: str = UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    width: int = 3,
    required: bool = True,
    include_source: bool = True,
) -> CrossFitStableUpstreamSchemaConfig:
    return CrossFitStableUpstreamSchemaConfig(
        namespace=namespace,
        calibrated_sources=(
            (
                PrecommittedCalibratedSource(
                    child_name="stable_bow_r",
                    source_kind="nested_calibrated_bow_r",
                ),
            )
            if include_source
            else ()
        ),
        raw_families=(
            PrecommittedRawFeatureFamily(
                source_kind=kind,
                consumer_role=role,
                signed_order_width=width,
                required=required,
            ),
        ),
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
        work_dir=Path("unused-test-work"),
    )


def test_permuted_and_renamed_discovery_columns_have_identical_schema_and_values():
    child = _VaryingDiscoveryBackend(add_third_every=3)
    backend = CrossFitStableUpstreamBackend(child, config=_config())

    first = _call(backend)
    second = _call(backend)
    third = _call(backend)

    assert (
        first.feature_names
        == second.feature_names
        == third.feature_names
        == (
            "topics__family_001__signed_mean",
            "topics__family_001__absolute_max",
            "topics__family_001__signed_order_001",
            "topics__family_001__signed_order_002",
            "topics__family_001__signed_order_003",
        )
    )
    assert not any(
        discovered in name
        for name in first.feature_names
        for discovered in ("alpha", "beta", "new_local", "fit_local")
    )
    assert first.feature_kinds == ("tfidf_topic_contrast",) * 5
    assert first.feature_roles == (UNCALIBRATED_EFFECT_MODIFIER_ROLE,) * 5
    np.testing.assert_allclose(first.feature_values, second.feature_values)
    # A changed discovered count changes summaries, but never the rectangular
    # precommitted schema.
    assert third.feature_values.shape == first.feature_values.shape == (2, 5)
    assert not np.array_equal(third.feature_values, first.feature_values)
    assert first.calibrated_source_names == ("stable_bow_r",)


def test_finite_signed_order_capacity_fails_before_child_column_omission():
    backend = CrossFitStableUpstreamBackend(
        _VaryingDiscoveryBackend(add_third_every=1),
        config=_config(width=2),
    )

    with pytest.raises(RuntimeError, match="refusing silent child-column omission"):
        _call(backend)


def test_signed_order_width_has_no_hidden_legacy_upper_bound():
    family = PrecommittedRawFeatureFamily(
        source_kind="safe_family",
        consumer_role=UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        signed_order_width=257,
    )

    assert family.signed_order_width == 257


class _SourceOnlyOrZeroRawBackend:
    def __init__(self, *, expose_zero_raw: bool) -> None:
        self.expose_zero_raw = expose_zero_raw

    def identity(self):
        return {
            "backend": "source_only_or_zero_raw_v1",
            "expose_zero_raw": self.expose_zero_raw,
        }

    def fit_predict(self, **kwargs):
        rows = tuple(kwargs["gate_row_ids"])
        if self.expose_zero_raw:
            names = ("fit_specific_zero",)
            kinds = ("tfidf_orphan_ngrams",)
            roles = (UNCALIBRATED_EFFECT_MODIFIER_ROLE,)
            values = np.zeros((len(rows), 1), dtype=float)
        else:
            names, kinds, roles = (), (), ()
            values = np.empty((len(rows), 0), dtype=float)
        return ContextFitUpstreamPrediction(
            gate_row_ids=rows,
            calibrated_source_names=("stable_bow_r",),
            calibrated_source_kinds=("nested_calibrated_bow_r",),
            calibrated_source_values=np.zeros((len(rows), 1), dtype=float),
            feature_names=names,
            feature_kinds=kinds,
            feature_roles=roles,
            feature_values=values,
        )


def test_optional_absence_is_zero_padded_with_presence_and_required_absence_fails():
    optional_config = _config(
        kind="tfidf_orphan_ngrams",
        width=2,
        required=False,
    )
    absent = _call(
        CrossFitStableUpstreamBackend(
            _SourceOnlyOrZeroRawBackend(expose_zero_raw=False),
            config=optional_config,
        )
    )
    observed_zero = _call(
        CrossFitStableUpstreamBackend(
            _SourceOnlyOrZeroRawBackend(expose_zero_raw=True),
            config=optional_config,
        )
    )
    np.testing.assert_array_equal(absent.feature_values, np.zeros((2, 5)))
    np.testing.assert_array_equal(observed_zero.feature_values[:, 0], np.ones(2))
    np.testing.assert_array_equal(observed_zero.feature_values[:, 1:], np.zeros((2, 4)))
    assert absent.feature_names == observed_zero.feature_names

    with pytest.raises(RuntimeError, match="missing required raw feature family"):
        _call(
            CrossFitStableUpstreamBackend(
                _SourceOnlyOrZeroRawBackend(expose_zero_raw=False),
                config=_config(kind="tfidf_orphan_ngrams", required=True),
            )
        )


def test_wrapper_is_gate_label_free_and_never_relabels_raw_features_as_tau():
    child = _VaryingDiscoveryBackend()
    backend = CrossFitStableUpstreamBackend(child, config=_config())
    prediction = _call(backend)

    parameters = inspect.signature(CrossFitStableUpstreamBackend.fit_predict).parameters
    assert "gate_treatment" not in parameters
    assert "gate_outcome" not in parameters
    received = child.calls[0]
    assert "gate_treatment" not in received
    assert "gate_outcome" not in received
    assert received["context_treatment"].flags.writeable is False
    assert received["context_outcome"].flags.writeable is False
    assert prediction.calibrated_source_names == ("stable_bow_r",)
    assert prediction.calibrated_source_values.shape == (2, 1)
    assert prediction.feature_values.shape == (2, 5)
    assert set(prediction.feature_roles) == {UNCALIBRATED_EFFECT_MODIFIER_ROLE}
    identity = backend.identity()
    assert identity["gate_labels_exposed_to_child"] is False
    assert identity["raw_features_relabelled_as_calibrated_sources"] is False


def test_wrapper_identity_binds_precommitment_and_child_identity():
    child = _VaryingDiscoveryBackend()
    config = _config(width=2)
    backend = CrossFitStableUpstreamBackend(child, config=config)

    identity = backend.identity()
    assert identity["config"] == config.identity()
    assert identity["child"] == child.identity()
    child.feature_kind = "silently_changed_family"
    with pytest.raises(ValueError, match="child backend identity changed"):
        backend.identity()


def test_unconfigured_calibrated_sources_and_raw_families_fail_closed():
    child = _VaryingDiscoveryBackend()
    missing_source_config = CrossFitStableUpstreamSchemaConfig(
        namespace="strict",
        calibrated_sources=(),
        raw_families=_config(include_source=False).raw_families,
    )
    with pytest.raises(RuntimeError, match="unconfigured calibrated sources"):
        _call(CrossFitStableUpstreamBackend(child, config=missing_source_config))

    wrong_family_config = _config(kind="tfidf_orphan_ngrams")
    with pytest.raises(RuntimeError, match="unconfigured raw feature families"):
        _call(
            CrossFitStableUpstreamBackend(
                _VaryingDiscoveryBackend(),
                config=wrong_family_config,
            )
        )


@pytest.mark.parametrize("variant", ["missing", "extra", "reordered"])
def test_exact_preaggregated_passthrough_rejects_malformed_exact_names(variant):
    expected_names = (
        "neural_query_treatment_signed_mean",
        "neural_query_treatment_absolute_max",
        "neural_query_treatment_signed_order_01",
        "neural_query_treatment_signed_order_02",
    )

    class MalformedPreaggregatedBackend:
        def identity(self):
            return {
                "backend": "malformed_preaggregated_neural_test_v1",
                "variant": variant,
            }

        def fit_predict(self, **kwargs):
            rows = tuple(kwargs["gate_row_ids"])
            if variant == "missing":
                names = expected_names[:-1]
            elif variant == "extra":
                names = (*expected_names, "neural_query_treatment_signed_order_03")
            else:
                names = (*expected_names[:2], expected_names[3], expected_names[2])
            return ContextFitUpstreamPrediction(
                gate_row_ids=rows,
                calibrated_source_values=np.empty((len(rows), 0), dtype=float),
                feature_names=names,
                feature_kinds=("neural_query_treatment_moments",) * len(names),
                feature_roles=(PROPENSITY_NUISANCE_FEATURE_ROLE,) * len(names),
                feature_values=np.zeros((len(rows), len(names)), dtype=float),
            )

    config = CrossFitStableUpstreamSchemaConfig(
        namespace="exact_neural",
        raw_families=(
            PrecommittedRawFeatureFamily(
                source_kind="neural_query_treatment_moments",
                consumer_role=PROPENSITY_NUISANCE_FEATURE_ROLE,
                signed_order_width=2,
                exact_passthrough_feature_names=expected_names,
            ),
        ),
    )
    backend = CrossFitStableUpstreamBackend(
        MalformedPreaggregatedBackend(),
        config=config,
    )

    with pytest.raises(RuntimeError, match="exact passthrough schema"):
        _call(backend)


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


def test_stable_wrappers_compose_and_feed_final_producer(tmp_path):
    effect = CrossFitStableUpstreamBackend(
        _VaryingDiscoveryBackend(add_third_every=2),
        config=_config(namespace="effect", width=3),
    )
    propensity = CrossFitStableUpstreamBackend(
        _VaryingDiscoveryBackend(
            feature_kind="neural_query_treatment_moments",
            feature_role=PROPENSITY_NUISANCE_FEATURE_ROLE,
            include_source=False,
            add_third_every=3,
        ),
        config=_config(
            namespace="query_treatment",
            kind="neural_query_treatment_moments",
            role=PROPENSITY_NUISANCE_FEATURE_ROLE,
            width=3,
            include_source=False,
        ),
    )
    backend = CompositeContextFitUpstreamBackend((effect, propensity))
    package = FinalContextFitUpstreamProducer(
        tmp_path / "final",
        backend=backend,
    ).produce(**_final_inputs())

    assert package.calibrated_sources.source_names == ("stable_bow_r",)
    assert len(package.raw_features.feature_names) == 10
    assert package.raw_features.feature_names[:5] == tuple(
        item[0] for item in effect.config.raw_output_schema()
    )
    assert package.raw_features.feature_names[5:] == tuple(
        item[0] for item in propensity.config.raw_output_schema()
    )
    assert package.raw_features.consumer_roles[:5] == (UNCALIBRATED_EFFECT_MODIFIER_ROLE,) * 5
    assert package.raw_features.consumer_roles[5:] == (PROPENSITY_NUISANCE_FEATURE_ROLE,) * 5
    package.verify_authenticated_content()


def test_config_rejects_unsupported_roles_and_forbidden_metadata():
    with pytest.raises(ValueError, match="unsupported"):
        PrecommittedRawFeatureFamily(
            source_kind="safe",
            consumer_role="calibrated_tau",
            signed_order_width=2,
        )
    with pytest.raises(ValueError, match="forbidden"):
        PrecommittedCalibratedSource(
            child_name="synthetic_oracle_effect",
            source_kind="nested_r",
        )
    # All three allowed consumer roles remain usable.
    for role in (
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    ):
        assert (
            PrecommittedRawFeatureFamily(
                source_kind="safe_family",
                consumer_role=role,
                signed_order_width=1,
            ).consumer_role
            == role
        )
