import inspect
import json
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
from oci.inference.final_context_fit_upstream_bank import (
    AuthenticatedCalibratedTauBank,
    AuthenticatedRoleAwareFeatureBank,
    FinalContextFitUpstreamProducer,
)


class _RecordingBackend:
    def __init__(self):
        self.calls = []

    def identity(self):
        return {"backend": "recording_context_fit_v1", "revision": 1}

    def fit_predict(
        self,
        *,
        outer_fold,
        context_row_ids,
        context_texts,
        context_treatment,
        context_outcome,
        gate_row_ids,
        gate_texts,
        work_dir,
    ):
        self.calls.append(
            {
                "outer_fold": outer_fold,
                "context_row_ids": context_row_ids,
                "context_texts": context_texts,
                "context_treatment": np.asarray(context_treatment).copy(),
                "context_outcome": np.asarray(context_outcome).copy(),
                "treatment_writeable": context_treatment.flags.writeable,
                "outcome_writeable": context_outcome.flags.writeable,
                "gate_row_ids": gate_row_ids,
                "gate_texts": gate_texts,
                "work_dir": Path(work_dir),
            }
        )
        gate = np.asarray(gate_row_ids, dtype=float)
        source_values = np.column_stack(
            [
                gate / 1000.0 + float(np.mean(context_treatment)),
                gate / 2000.0 + float(np.mean(context_outcome)),
            ]
        )
        feature_values = np.column_stack(
            [
                gate / 100.0,
                np.full(len(gate), len(context_row_ids), dtype=float),
                np.full(len(gate), float(np.sum(context_outcome)), dtype=float),
            ]
        )
        return ContextFitUpstreamPrediction(
            gate_row_ids=gate_row_ids,
            calibrated_source_names=("bow_weighted_r", "htr_weighted_r"),
            calibrated_source_kinds=("bow_r_loss", "htr_neural"),
            calibrated_source_values=source_values,
            feature_names=("pair_uplift", "bow_propensity", "query_effect_001"),
            feature_kinds=(
                "matched_pair_uplift",
                "bow_nuisance",
                "neural_query_effect_moments",
            ),
            feature_roles=(
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            ),
            feature_values=feature_values,
        )


def _inputs():
    return {
        "outer_fold": 4,
        "outer_train_row_ids": (70, 11, 42, 99, 5, 18),
        "outer_train_texts": (
            " Text 70 ",
            "TEXT 11",
            "text 42",
            "text 99",
            "text 5",
            "text 18",
        ),
        "outer_train_treatment": np.asarray([0, 1, 1, 0, 0, 1], dtype=float),
        "outer_train_outcome": np.asarray([0.2, 0.8, 0.7, 0.1, 0.3, 0.9], dtype=float),
        "outer_heldout_row_ids": (501, 400),
        "outer_heldout_texts": ("held 501", "held 400"),
        "meta_inner_fold_ids": (2, 1, 2, 3, 1, 3),
    }


def test_final_producer_builds_complete_oof_and_label_free_heldout_banks(tmp_path):
    backend = _RecordingBackend()
    producer = FinalContextFitUpstreamProducer(tmp_path / "cache", backend=backend)
    package = producer.produce(**_inputs())

    assert isinstance(package.calibrated_sources, AuthenticatedCalibratedTauBank)
    assert isinstance(package.raw_features, AuthenticatedRoleAwareFeatureBank)
    assert package.calibrated_sources.train_row_ids == _inputs()["outer_train_row_ids"]
    assert package.raw_features.heldout_row_ids == _inputs()["outer_heldout_row_ids"]
    assert package.raw_features.consumer_roles == (
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    )
    package.verify_authenticated_content()

    # Folds are traversed in their precommitted first-seen order, and no call
    # receives labels for its prediction rows.
    assert [call["gate_row_ids"] for call in backend.calls] == [
        (70, 42),
        (11, 5),
        (99, 18),
        (501, 400),
    ]
    train_ids = _inputs()["outer_train_row_ids"]
    train_texts = _inputs()["outer_train_texts"]
    treatment = _inputs()["outer_train_treatment"]
    outcome = _inputs()["outer_train_outcome"]
    folds = _inputs()["meta_inner_fold_ids"]
    for call_index, fold_id in enumerate((2, 1, 3)):
        expected_positions = [index for index, value in enumerate(folds) if value != fold_id]
        call = backend.calls[call_index]
        assert call["context_row_ids"] == tuple(train_ids[index] for index in expected_positions)
        assert call["context_texts"] == tuple(train_texts[index] for index in expected_positions)
        np.testing.assert_array_equal(call["context_treatment"], treatment[expected_positions])
        np.testing.assert_array_equal(call["context_outcome"], outcome[expected_positions])
        assert call["treatment_writeable"] is False
        assert call["outcome_writeable"] is False
    assert backend.calls[-1]["context_row_ids"] == train_ids
    assert backend.calls[-1]["context_texts"] == train_texts

    # Every OOF cell carries exactly its complementary fit set; every final
    # heldout cell carries exactly the complete outer train.
    source = package.calibrated_sources
    features = package.raw_features
    for position, fold_id in enumerate(folds):
        expected = {
            row_id for row_id, candidate_fold in zip(train_ids, folds) if candidate_fold != fold_id
        }
        for bank in (source, features):
            assert all(
                set(lineage.recursive_fit_row_ids()) == expected
                for lineage in bank.train_oof_fit_row_provenance[position]
            )
    for bank in (source, features):
        assert all(
            set(lineage.recursive_fit_row_ids()) == set(train_ids)
            for row in bank.outer_heldout_fit_row_provenance
            for lineage in row
        )

    # Loading the same content address re-authenticates bytes and performs no
    # additional fit.
    calls_before = len(backend.calls)
    loaded = producer.produce(**_inputs())
    assert len(backend.calls) == calls_before
    assert loaded.cache_key == package.cache_key
    np.testing.assert_array_equal(
        loaded.calibrated_sources.train_oof_values, source.train_oof_values
    )


class _StageLikeBackend:
    def identity(self):
        return {"backend": "stage_like_v1"}

    def fit_predict(self, **kwargs):
        rows = kwargs["gate_row_ids"]
        return ContextFitUpstreamPrediction(
            gate_row_ids=rows,
            calibrated_source_names=("stage_calibrated_r",),
            calibrated_source_kinds=("nested_calibrated_bow_weighted_r",),
            calibrated_source_values=np.asarray(rows, dtype=float)[:, None] / 100.0,
            feature_names=("stage_embedding_contrast",),
            feature_kinds=("embedding_whole_cohort",),
            feature_roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,),
            feature_values=np.asarray(rows, dtype=float)[:, None] / 10.0,
        )


class _TfidfLikeBackend:
    def identity(self):
        return {"backend": "tfidf_like_v1"}

    def fit_predict(self, **kwargs):
        rows = kwargs["gate_row_ids"]
        return ContextFitUpstreamPrediction(
            gate_row_ids=rows,
            calibrated_source_values=np.empty((len(rows), 0), dtype=float),
            feature_names=("tfidf_outcome_topic_001",),
            feature_kinds=("tfidf_topics",),
            feature_roles=(OUTCOME_NUISANCE_FEATURE_ROLE,),
            feature_values=np.asarray(rows, dtype=float)[:, None] / 20.0,
        )


class _QueryLikeBackend:
    def identity(self):
        return {"backend": "query_like_v1"}

    def fit_predict(self, **kwargs):
        rows = kwargs["gate_row_ids"]
        return ContextFitUpstreamPrediction(
            gate_row_ids=rows,
            calibrated_source_values=np.empty((len(rows), 0), dtype=float),
            feature_names=("neural_query_effect_context_001",),
            feature_kinds=("neural_query_effect_moments",),
            feature_roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,),
            feature_values=np.asarray(rows, dtype=float)[:, None] / 30.0,
        )


def test_final_producer_composes_stage_tfidf_and_neural_query_backends(tmp_path):
    backend = CompositeContextFitUpstreamBackend(
        [_StageLikeBackend(), _TfidfLikeBackend(), _QueryLikeBackend()]
    )
    package = FinalContextFitUpstreamProducer(tmp_path / "composite", backend=backend).produce(
        **_inputs()
    )

    assert package.calibrated_sources.source_names == ("stage_calibrated_r",)
    assert package.raw_features.feature_names == (
        "stage_embedding_contrast",
        "tfidf_outcome_topic_001",
        "neural_query_effect_context_001",
    )
    assert package.raw_features.consumer_roles == (
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    )


class _WrongRowOrderBackend(_RecordingBackend):
    def identity(self):
        return {"backend": "wrong_row_order_v1"}

    def fit_predict(self, **kwargs):
        prediction = super().fit_predict(**kwargs)
        order = np.arange(len(prediction.gate_row_ids))[::-1]
        return ContextFitUpstreamPrediction(
            gate_row_ids=tuple(reversed(prediction.gate_row_ids)),
            calibrated_source_names=prediction.calibrated_source_names,
            calibrated_source_kinds=prediction.calibrated_source_kinds,
            calibrated_source_values=prediction.calibrated_source_values[order],
            feature_names=prediction.feature_names,
            feature_kinds=prediction.feature_kinds,
            feature_roles=prediction.feature_roles,
            feature_values=prediction.feature_values[order],
        )


class _UnstableSchemaBackend(_RecordingBackend):
    def identity(self):
        return {"backend": "unstable_schema_v1"}

    def fit_predict(self, **kwargs):
        prediction = super().fit_predict(**kwargs)
        if len(self.calls) == 1:
            return prediction
        return ContextFitUpstreamPrediction(
            gate_row_ids=prediction.gate_row_ids,
            calibrated_source_names=("bow_weighted_r", "renamed_htr_weighted_r"),
            calibrated_source_kinds=prediction.calibrated_source_kinds,
            calibrated_source_values=prediction.calibrated_source_values,
            feature_names=prediction.feature_names,
            feature_kinds=prediction.feature_kinds,
            feature_roles=prediction.feature_roles,
            feature_values=prediction.feature_values,
        )


class _IdentityDriftBackend(_RecordingBackend):
    def __init__(self):
        super().__init__()
        self.revision = 1

    def identity(self):
        return {"backend": "identity_drift_v1", "revision": self.revision}

    def fit_predict(self, **kwargs):
        prediction = super().fit_predict(**kwargs)
        self.revision += 1
        return prediction


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (_WrongRowOrderBackend(), "row identity or order"),
        (_UnstableSchemaBackend(), "changed across meta-inner fits"),
        (_IdentityDriftBackend(), "identity changed"),
    ],
)
def test_final_producer_rejects_malicious_backend_outputs(tmp_path, backend, message):
    producer = FinalContextFitUpstreamProducer(tmp_path / type(backend).__name__, backend=backend)
    with pytest.raises((TypeError, ValueError), match=message):
        producer.produce(**_inputs())


class _ForbiddenIdentityBackend(_RecordingBackend):
    def identity(self):
        return {"backend": "benchmark_oracle_reader"}


def test_final_producer_rejects_forbidden_identity_and_instance_override(tmp_path):
    with pytest.raises(ValueError, match="forbidden"):
        FinalContextFitUpstreamProducer(tmp_path / "forbidden", backend=_ForbiddenIdentityBackend())

    backend = _RecordingBackend()
    backend.fit_predict = lambda **_kwargs: None
    with pytest.raises(TypeError, match="per-instance method overrides"):
        FinalContextFitUpstreamProducer(tmp_path / "override", backend=backend)


def test_public_production_api_structurally_cannot_receive_heldout_labels():
    parameters = inspect.signature(FinalContextFitUpstreamProducer.produce).parameters
    assert "outer_heldout_treatment" not in parameters
    assert "outer_heldout_outcome" not in parameters
    assert set(parameters) == {
        "self",
        "outer_fold",
        "outer_train_row_ids",
        "outer_train_texts",
        "outer_train_treatment",
        "outer_train_outcome",
        "outer_heldout_row_ids",
        "outer_heldout_texts",
        "meta_inner_fold_ids",
    }


def test_final_cache_rejects_matrix_tampering(tmp_path):
    backend = _RecordingBackend()
    producer = FinalContextFitUpstreamProducer(tmp_path / "cache", backend=backend)
    package = producer.produce(**_inputs())
    matrix_path = package.manifest_path.parent / "calibrated_source_train_oof.npy"
    values = np.load(matrix_path, allow_pickle=False)
    values[0, 0] += 1.0
    with matrix_path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)

    with pytest.raises(ValueError, match="SHA-256 authentication"):
        package.verify_authenticated_content()
    with pytest.raises(ValueError, match="failed authentication"):
        producer.produce(**_inputs())


def test_final_cache_rejects_manifest_tampering(tmp_path):
    backend = _RecordingBackend()
    producer = FinalContextFitUpstreamProducer(tmp_path / "cache", backend=backend)
    package = producer.produce(**_inputs())
    payload = json.loads(package.manifest_path.read_text(encoding="utf-8"))
    payload["calibrated_sources"]["names"][0] = "silently_replaced_source"
    package.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest bytes were modified"):
        package.verify_authenticated_content()
    with pytest.raises(ValueError, match="content SHA-256 mismatch"):
        producer.produce(**_inputs())


def test_authenticated_bank_detects_in_memory_value_tampering(tmp_path):
    package = FinalContextFitUpstreamProducer(
        tmp_path / "cache", backend=_RecordingBackend()
    ).produce(**_inputs())
    package.raw_features.train_oof_values.setflags(write=True)
    package.raw_features.train_oof_values[0, 0] += 0.5
    with pytest.raises(ValueError, match="in-memory authenticated content was modified"):
        package.raw_features.verify_authenticated_content()
