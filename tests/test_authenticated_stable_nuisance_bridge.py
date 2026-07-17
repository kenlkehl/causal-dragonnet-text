from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

import oci.inference.authenticated_stable_nuisance_bridge as bridge
from oci.config import AppliedInferenceConfig
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
from oci.inference.neural_query_context_backend import NEURAL_QUERY_CONTEXT_BACKEND_ID
from oci.inference.stable_context_fit_upstream_backend import (
    CrossFitStableUpstreamBackend,
    CrossFitStableUpstreamSchemaConfig,
    PrecommittedCalibratedSource,
    PrecommittedRawFeatureFamily,
)
from oci.inference.stage1_upstream_gate_backend import STAGE1_CONTEXT_BACKEND_ID
from oci.inference.shared_tfidf_context_fit_service import (
    InMemorySharedTfidfContextFitService,
    SharedTfidfContextBackend,
)
from oci.inference.tfidf_upstream_gate_backend import TFIDF_CONTEXT_BACKEND_ID

_CONFIG_SHA = "a" * 64


def _row_probability(rows, *, base):
    values = np.asarray(rows, dtype=int)
    return base + (values % 5) * 0.01


class _FakeHistoricalStage1ConfigSnapshot:
    def __init__(self, *, freeze_encoder):
        self.sha256 = _CONFIG_SHA
        self._freeze_encoder = freeze_encoder

    def verify_source(self):
        return None

    def applied_config(self):
        config = AppliedInferenceConfig()
        config.architecture.feature_extractor_type = "hierarchical_transformer"
        config.architecture.htr_freeze_sentence_encoder = self._freeze_encoder
        return config


class _FakeHistoricalStage1ContextBackend:
    def __init__(self, *, freeze_encoder=False, snapshot_freeze_encoder=None):
        if snapshot_freeze_encoder is None:
            snapshot_freeze_encoder = freeze_encoder
        self._stage1_config_snapshot = _FakeHistoricalStage1ConfigSnapshot(
            freeze_encoder=snapshot_freeze_encoder
        )
        self.config = AppliedInferenceConfig()
        self.config.architecture.feature_extractor_type = "hierarchical_transformer"
        self.config.architecture.htr_freeze_sentence_encoder = freeze_encoder
        self.config.architecture.htr_require_live_unfrozen_encoder_attestation = True
        self.config.architecture.agentic_attention_variable_forest.fold_parallelism = "1"
        self.config.architecture.multi_model_forest.htr_fold_parallelism = "1"
        self.config.architecture.multi_model_forest.htr_evidence_enabled = True
        self.config.architecture.multi_model_forest.matched_pair_uplift_enabled = True
        self.config.architecture.multi_model_forest.matched_pair_htr_enabled = True
        self.device = "cpu"
        self._effective_config_sha256 = self.effective_config_sha256()

    def effective_config_sha256(self):
        return bridge.stage1_runtime_module._effective_applied_config_sha256(self.config)

    def htr_runtime_source_attestation(self):
        return bridge.stage1_runtime_module._htr_runtime_source_attestation()

    def identity(self):
        identity_config = AppliedInferenceConfig()
        identity_config.architecture.feature_extractor_type = "hierarchical_transformer"
        identity_config.architecture.htr_freeze_sentence_encoder = False
        identity_config.architecture.htr_require_live_unfrozen_encoder_attestation = True
        identity_config.architecture.agentic_attention_variable_forest.fold_parallelism = "1"
        identity_config.architecture.multi_model_forest.htr_fold_parallelism = "1"
        identity_config.architecture.multi_model_forest.htr_evidence_enabled = True
        identity_config.architecture.multi_model_forest.matched_pair_uplift_enabled = True
        identity_config.architecture.multi_model_forest.matched_pair_htr_enabled = True
        context_htr_identity = bridge.context_prediction_htr_provider_identity(
            identity_config,
            device=self.device,
        )
        return {
            "backend": STAGE1_CONTEXT_BACKEND_ID,
            "stage1_config_sha256": _CONFIG_SHA,
            "effective_config_schema_version": bridge.EFFECTIVE_STAGE1_CONFIG_ID,
            "effective_config_sha256": self._effective_config_sha256,
            "stage1_code_sha256": hashlib.sha256(
                Path(bridge.stage1_model_module.__file__).read_bytes()
            ).hexdigest(),
            "pair_code_sha256": hashlib.sha256(
                Path(bridge.pair_runtime_module.__file__).read_bytes()
            ).hexdigest(),
            "htr_model_tree_sha256": "b" * 64,
            "htr_runtime_source_attestation": self.htr_runtime_source_attestation(),
            "context_prediction_htr_provider": context_htr_identity,
            "context_prediction_htr_provider_required": True,
            "context_prediction_htr_provider_id": bridge.CONTEXT_PREDICTION_HTR_PROVIDER_ID,
            "context_train_pair_or_effect_predictions_consumed": False,
            "spent_discovery_path_changed": False,
            "required_families": [
                "bow_nuisance",
                "htr_nuisance",
                "bow_weighted_r",
                "htr_weighted_r",
            ],
            "revision": 1,
        }

    def fit_predict(self, **kwargs):
        rows = kwargs["gate_row_ids"]
        bow_e = _row_probability(rows, base=0.20)
        htr_e = _row_probability(rows, base=0.35)
        bow_m = _row_probability(rows, base=0.45)
        htr_m = _row_probability(rows, base=0.60)
        feature_names = (
            "bow_e_view_1",
            "bow_e_view_2",
            "bow_e_view_3",
            "htr_e",
            "bow_m_view_1",
            "bow_m_view_2",
            "bow_m_view_3",
            "htr_m",
        )
        feature_values = np.column_stack(
            (
                bow_e,
                bow_e + 0.03,
                bow_e + 0.06,
                htr_e,
                bow_m,
                bow_m + 0.03,
                bow_m + 0.06,
                htr_m,
            )
        )
        return ContextFitUpstreamPrediction(
            gate_row_ids=rows,
            calibrated_source_names=("stage1_tau",),
            calibrated_source_kinds=("nested_calibrated_bow_weighted_r",),
            calibrated_source_values=(bow_m - bow_e)[:, None],
            feature_names=feature_names,
            feature_kinds=(
                "bow_nuisance",
                "bow_nuisance",
                "bow_nuisance",
                "htr_nuisance",
                "bow_nuisance",
                "bow_nuisance",
                "bow_nuisance",
                "htr_nuisance",
            ),
            feature_roles=(
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                OUTCOME_NUISANCE_FEATURE_ROLE,
                OUTCOME_NUISANCE_FEATURE_ROLE,
                OUTCOME_NUISANCE_FEATURE_ROLE,
                OUTCOME_NUISANCE_FEATURE_ROLE,
            ),
            feature_values=feature_values,
        )


class _FakeTfidfContextBackend:
    max_orphan_features = 8
    minimum_orphan_arm_support = 2

    def identity(self):
        return {
            "backend": TFIDF_CONTEXT_BACKEND_ID,
            "revision": 1,
            "max_orphan_features": self.max_orphan_features,
            "minimum_orphan_arm_support": self.minimum_orphan_arm_support,
        }

    def fit_predict(self, **kwargs):
        rows = kwargs["gate_row_ids"]
        return ContextFitUpstreamPrediction(
            gate_row_ids=rows,
            calibrated_source_names=(),
            calibrated_source_kinds=(),
            calibrated_source_values=np.empty((len(rows), 0)),
            feature_names=("tfidf_mixed_family_column",),
            feature_kinds=("tfidf_topics",),
            feature_roles=(PROPENSITY_NUISANCE_FEATURE_ROLE,),
            feature_values=_row_probability(rows, base=0.1)[:, None],
        )


class _InexactTfidfDelegate(_FakeTfidfContextBackend):
    pass


class _FakeNeuralQueryContextBackend:
    def identity(self):
        return {"backend": NEURAL_QUERY_CONTEXT_BACKEND_ID, "revision": 1}

    def fit_predict(self, **kwargs):
        rows = kwargs["gate_row_ids"]
        return ContextFitUpstreamPrediction(
            gate_row_ids=rows,
            calibrated_source_names=(),
            calibrated_source_kinds=(),
            calibrated_source_values=np.empty((len(rows), 0)),
            feature_names=("query_effect",),
            feature_kinds=("neural_query_effect_moments",),
            feature_roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,),
            feature_values=np.sin(np.asarray(rows, dtype=float))[:, None],
        )


class _WrongStageBackend(_FakeHistoricalStage1ContextBackend):
    pass


def _patch_runtime_types(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "HistoricalStage1ContextBackend",
        _FakeHistoricalStage1ContextBackend,
    )
    monkeypatch.setattr(
        bridge,
        "HistoricalStage1ConfigSnapshot",
        _FakeHistoricalStage1ConfigSnapshot,
    )
    monkeypatch.setattr(bridge, "TfidfTopicOrphanContextBackend", _FakeTfidfContextBackend)
    monkeypatch.setattr(bridge, "NeuralQueryContextBackend", _FakeNeuralQueryContextBackend)
    monkeypatch.setattr(
        bridge,
        "_AUTHENTICATED_STAGE1_IDENTITY",
        _FakeHistoricalStage1ContextBackend.identity,
    )
    monkeypatch.setattr(
        bridge,
        "_AUTHENTICATED_STAGE1_FIT_PREDICT",
        _FakeHistoricalStage1ContextBackend.fit_predict,
    )
    monkeypatch.setattr(
        bridge,
        "_AUTHENTICATED_STAGE1_EFFECTIVE_CONFIG",
        _FakeHistoricalStage1ContextBackend.effective_config_sha256,
    )
    monkeypatch.setattr(
        bridge,
        "_AUTHENTICATED_STAGE1_HTR_RUNTIME_SOURCES",
        _FakeHistoricalStage1ContextBackend.htr_runtime_source_attestation,
    )
    monkeypatch.setattr(
        bridge,
        "_AUTHENTICATED_CONFIG_SNAPSHOT_VERIFY_SOURCE",
        _FakeHistoricalStage1ConfigSnapshot.verify_source,
    )
    monkeypatch.setattr(
        bridge,
        "_AUTHENTICATED_CONFIG_SNAPSHOT_APPLIED_CONFIG",
        _FakeHistoricalStage1ConfigSnapshot.applied_config,
    )
    monkeypatch.setattr(
        bridge,
        "_AUTHENTICATED_TFIDF_IDENTITY",
        _FakeTfidfContextBackend.identity,
    )
    monkeypatch.setattr(
        bridge,
        "_AUTHENTICATED_TFIDF_FIT_PREDICT",
        _FakeTfidfContextBackend.fit_predict,
    )
    monkeypatch.setattr(
        bridge,
        "_AUTHENTICATED_QUERY_IDENTITY",
        _FakeNeuralQueryContextBackend.identity,
    )
    monkeypatch.setattr(
        bridge,
        "_AUTHENTICATED_QUERY_FIT_PREDICT",
        _FakeNeuralQueryContextBackend.fit_predict,
    )


def _schema_config(*, bad_reduction=False, ambiguous=False):
    bow_prop = PrecommittedRawFeatureFamily(
        source_kind="bow_nuisance",
        consumer_role=PROPENSITY_NUISANCE_FEATURE_ROLE,
        signed_order_width=1,
        required=True,
    )
    if bad_reduction:
        bow_prop = PrecommittedRawFeatureFamily(
            source_kind="bow_nuisance",
            consumer_role=PROPENSITY_NUISANCE_FEATURE_ROLE,
            signed_order_width=1,
            required=True,
            exact_passthrough_feature_names=(
                "bow_e_view_1",
                "bow_e_view_2",
                "bow_e_view_3",
            ),
        )
    families = (
        bow_prop,
        PrecommittedRawFeatureFamily(
            source_kind="htr_nuisance",
            consumer_role=PROPENSITY_NUISANCE_FEATURE_ROLE,
            signed_order_width=1,
        ),
        PrecommittedRawFeatureFamily(
            source_kind="bow_nuisance",
            consumer_role=OUTCOME_NUISANCE_FEATURE_ROLE,
            signed_order_width=1,
        ),
        PrecommittedRawFeatureFamily(
            source_kind="htr_nuisance",
            consumer_role=OUTCOME_NUISANCE_FEATURE_ROLE,
            signed_order_width=1,
        ),
        PrecommittedRawFeatureFamily(
            source_kind="tfidf_topics",
            consumer_role=PROPENSITY_NUISANCE_FEATURE_ROLE,
            signed_order_width=1,
        ),
        PrecommittedRawFeatureFamily(
            source_kind="neural_query_effect_moments",
            consumer_role=UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            signed_order_width=1,
        ),
    )
    config = CrossFitStableUpstreamSchemaConfig(
        namespace="all_evidence_upstream",
        calibrated_sources=(
            PrecommittedCalibratedSource(
                child_name="stage1_tau",
                source_kind="nested_calibrated_bow_weighted_r",
            ),
        ),
        raw_families=families,
        reject_unconfigured_calibrated_sources=True,
        reject_unconfigured_raw_families=True,
        source_config_sha256=_CONFIG_SHA,
    )
    if ambiguous:
        object.__setattr__(config, "raw_families", (*families, families[0]))
    return config


def _build_package(
    tmp_path,
    monkeypatch,
    *,
    freeze_encoder=False,
    snapshot_freeze_encoder=None,
    bad_reduction=False,
    ambiguous=False,
    shared_tfidf_wrapper=False,
    tfidf_delegate_type=_FakeTfidfContextBackend,
):
    _patch_runtime_types(monkeypatch)
    stage = _FakeHistoricalStage1ContextBackend(
        freeze_encoder=freeze_encoder,
        snapshot_freeze_encoder=snapshot_freeze_encoder,
    )
    tfidf_delegate = tfidf_delegate_type()
    tfidf_runtime = tfidf_delegate
    if shared_tfidf_wrapper:
        tfidf_runtime = SharedTfidfContextBackend(
            backend=tfidf_delegate,
            service=InMemorySharedTfidfContextFitService(
                source_backend_identity=tfidf_delegate.identity()
            ),
        )
    composite = CompositeContextFitUpstreamBackend(
        (stage, tfidf_runtime, _FakeNeuralQueryContextBackend())
    )
    stable = CrossFitStableUpstreamBackend(
        composite,
        config=_schema_config(bad_reduction=bad_reduction, ambiguous=ambiguous),
    )
    producer = FinalContextFitUpstreamProducer(tmp_path / "cache", backend=stable)
    train_rows = tuple(range(100, 112))
    heldout_rows = (900, 901, 902)
    treatment = np.asarray([0, 1] * 6, dtype=float)
    outcome = np.asarray([0, 0, 1, 1] * 3, dtype=float)
    package = producer.produce(
        outer_fold=2,
        outer_train_row_ids=train_rows,
        outer_train_texts=tuple(f"train {row}" for row in train_rows),
        outer_train_treatment=treatment,
        outer_train_outcome=outcome,
        outer_heldout_row_ids=heldout_rows,
        outer_heldout_texts=tuple(f"heldout {row}" for row in heldout_rows),
        meta_inner_fold_ids=tuple((index % 3) + 1 for index in range(len(train_rows))),
    )
    return package, producer


def test_bridge_derives_only_four_exact_stage1_nuisance_means(tmp_path, monkeypatch):
    package, producer = _build_package(tmp_path, monkeypatch)
    result = bridge.derive_exact_nuisance_from_runtime_stable_stage1(
        package,
        runtime_producer=producer,
    )
    raw = package.raw_features
    selected = [record["raw_column_index"] for record in result.selected_columns]
    np.testing.assert_array_equal(
        result.nuisance.train_oof_values,
        raw.train_oof_values[:, selected],
    )
    expected_heldout = np.column_stack(
        (
            _row_probability((900, 901, 902), base=0.20) + 0.03,
            _row_probability((900, 901, 902), base=0.35),
            _row_probability((900, 901, 902), base=0.45) + 0.03,
            _row_probability((900, 901, 902), base=0.60),
        )
    )
    np.testing.assert_allclose(result.nuisance.outer_heldout_values, expected_heldout)
    assert result.nuisance.prediction_semantics == (
        bridge.EXACT_PROPENSITY_PREDICTION,
        bridge.EXACT_PROPENSITY_PREDICTION,
        bridge.EXACT_OUTCOME_PREDICTION,
        bridge.EXACT_OUTCOME_PREDICTION,
    )
    assert {record["source_kind"] for record in result.selected_columns} == {
        "bow_nuisance",
        "htr_nuisance",
    }
    assert all("tfidf" not in record["source_kind"] for record in result.selected_columns)
    audit = result.audit_record()
    assert audit["semantic_inference_from_feature_names"] is False
    assert audit["tfidf_columns_eligible"] is False
    assert audit["package_only_derivation_supported"] is False
    assert audit["stage1_config_snapshot_sha256"] == _CONFIG_SHA
    assert audit["htr_sentence_encoder_unfrozen_from_snapshot"] is True
    result.verify_authenticated_content(package, runtime_producer=producer)


def test_bridge_has_no_package_only_or_names_only_entry_point(tmp_path, monkeypatch):
    package, producer = _build_package(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(package)
    with pytest.raises(TypeError, match="exact FinalContextFitUpstreamProducer"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=None,
        )
    with pytest.raises(TypeError):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
            feature_names=package.raw_features.feature_names,
        )


def test_bridge_accepts_only_fully_attested_shared_tfidf_wrapper(tmp_path, monkeypatch):
    package, producer = _build_package(
        tmp_path,
        monkeypatch,
        shared_tfidf_wrapper=True,
    )
    result = bridge.derive_exact_nuisance_from_runtime_stable_stage1(
        package,
        runtime_producer=producer,
    )

    audit = result.audit_record()
    assert audit["tfidf_shared_wrapper_active"] is True
    assert audit["tfidf_runtime_backend_identity_sha256"] != (
        audit["tfidf_delegate_backend_identity_sha256"]
    )
    assert {
        "shared_tfidf_identity_code_sha256",
        "shared_tfidf_fit_predict_code_sha256",
        "shared_tfidf_assert_stable_code_sha256",
        "shared_tfidf_service_identity_code_sha256",
        "shared_tfidf_service_assert_source_code_sha256",
        "shared_tfidf_service_transform_code_sha256",
    }.issubset(audit["runtime_code_attestation"])
    result.verify_authenticated_content(package, runtime_producer=producer)


def test_bridge_rejects_tampered_shared_tfidf_method_and_inexact_delegate(
    tmp_path,
    monkeypatch,
):
    package, producer = _build_package(
        tmp_path / "method",
        monkeypatch,
        shared_tfidf_wrapper=True,
    )

    def changed_fit_predict(self, **kwargs):
        return self.backend.fit_predict(**kwargs)

    monkeypatch.setattr(SharedTfidfContextBackend, "fit_predict", changed_fit_predict)
    with pytest.raises(
        (TypeError, RuntimeError),
        match=(
            "upstream backend runtime implementation changed|authenticated runtime method "
            "changed|runtime implementation changed"
        ),
    ):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
        )

    monkeypatch.undo()
    inexact_package, inexact_producer = _build_package(
        tmp_path / "delegate",
        monkeypatch,
        shared_tfidf_wrapper=True,
        tfidf_delegate_type=_InexactTfidfDelegate,
    )
    with pytest.raises(TypeError, match="delegate has the wrong exact backend type"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            inexact_package,
            runtime_producer=inexact_producer,
        )


@pytest.mark.parametrize("bad_reduction,ambiguous", [(True, False), (False, True)])
def test_bridge_rejects_wrong_reduction_and_ambiguous_family(
    tmp_path,
    monkeypatch,
    bad_reduction,
    ambiguous,
):
    package, producer = _build_package(
        tmp_path,
        monkeypatch,
        bad_reduction=bad_reduction,
        ambiguous=ambiguous,
    )
    message = "required signed-mean" if bad_reduction else "ambiguous duplicate"
    with pytest.raises(ValueError, match=message):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
        )


def test_bridge_rejects_wrong_runtime_type_digest_and_frozen_htr(tmp_path, monkeypatch):
    package, producer = _build_package(tmp_path / "valid", monkeypatch)
    monkeypatch.setattr(bridge, "HistoricalStage1ContextBackend", _WrongStageBackend)
    with pytest.raises(TypeError, match="exactly Stage-1"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
        )

    monkeypatch.setattr(
        bridge,
        "HistoricalStage1ContextBackend",
        _FakeHistoricalStage1ContextBackend,
    )
    object.__setattr__(package, "producer_identity_sha256", "0" * 64)
    with pytest.raises(ValueError, match="producer identity changed"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
        )

    frozen_package, frozen_producer = _build_package(
        tmp_path / "frozen",
        monkeypatch,
        freeze_encoder=True,
    )
    with pytest.raises(ValueError, match="unfrozen HTR encoder"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            frozen_package,
            runtime_producer=frozen_producer,
        )

    mismatch_package, mismatch_producer = _build_package(
        tmp_path / "snapshot_mismatch",
        monkeypatch,
        freeze_encoder=False,
        snapshot_freeze_encoder=True,
    )
    with pytest.raises(ValueError, match="immutable Stage-1 config snapshot"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            mismatch_package,
            runtime_producer=mismatch_producer,
        )


def test_bridge_rejects_runtime_method_change(tmp_path, monkeypatch):
    package, producer = _build_package(tmp_path, monkeypatch)

    def replaced_fit_predict(self, **kwargs):
        return _FakeHistoricalStage1ContextBackend.fit_predict(self, **kwargs)

    monkeypatch.setattr(_FakeHistoricalStage1ContextBackend, "fit_predict", replaced_fit_predict)
    with pytest.raises(TypeError, match="runtime (?:method|implementation) changed"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
        )


def test_bridge_rejects_stage1_runner_symbol_change(tmp_path, monkeypatch):
    package, producer = _build_package(tmp_path, monkeypatch)
    monkeypatch.setattr(
        bridge.stage1_runtime_module,
        "MultiModelForestStage1Runner",
        object,
    )
    with pytest.raises(TypeError, match="Stage-1/HTR runtime symbol path changed"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
        )


@pytest.mark.parametrize(
    "alias_name,replacement",
    (
        ("HistoricalStage1ContextPredictionHTRProvider", object),
        ("context_prediction_htr_provider_identity", lambda *_args, **_kwargs: {}),
        ("CONTEXT_PREDICTION_HTR_PROVIDER_ID", "mutated-provider-id"),
    ),
)
def test_bridge_rejects_stage1_context_provider_alias_mutations(
    tmp_path,
    monkeypatch,
    alias_name,
    replacement,
):
    package, producer = _build_package(tmp_path, monkeypatch)
    monkeypatch.setattr(bridge.stage1_runtime_module, alias_name, replacement)
    with pytest.raises(TypeError, match="Stage-1/HTR runtime symbol path changed"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
        )


def test_bridge_rejects_mutated_complete_effective_stage1_config(tmp_path, monkeypatch):
    package, producer = _build_package(tmp_path, monkeypatch)
    stage = producer.backend.backend.backends[0]
    stage.config.training.learning_rate *= 2.0

    with pytest.raises(ValueError, match="complete effective config"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
        )


@pytest.mark.parametrize(
    "owner,method_name",
    (
        (bridge.MultiModelHTREvidenceProvider, "__init__"),
        (bridge.MultiModelHTREvidenceProvider, "_ensure_runner"),
        (bridge.MultiModelForestStage1HTRProvider, "__init__"),
        (bridge.MultiModelForestStage1HTRProvider, "fit_nuisance_inner_ensemble_predict"),
        (bridge.MultiModelForestStage1HTRProvider, "_temporary_effect_objective"),
        (bridge.ContextPredictionOnlyFeatureBundle, "__init__"),
        (bridge.ContextPredictionOnlyFeatureBundle, "__post_init__"),
        (bridge.pair_runtime_module.HTRPairUpliftNet, "__init__"),
        (bridge.pair_runtime_module.HTRPairUpliftNet, "forward"),
        (bridge.pair_runtime_module.PairUpliftFitResult, "__init__"),
        (bridge.attention_runtime_module._EffectNet, "__init__"),
        (bridge.attention_runtime_module._EffectNet, "forward"),
        (bridge.attention_runtime_module._NuisanceNet, "__init__"),
        (bridge.attention_runtime_module._NuisanceNet, "forward"),
        (bridge.BinaryProbabilityCalibrator, "__init__"),
        (bridge.BinaryProbabilityCalibrator, "fit"),
        (bridge.BinaryProbabilityCalibrator, "transform"),
        (bridge.KFold, "__init__"),
        (bridge.KFold, "split"),
        (bridge.attention_runtime_module._FoldTextDataset, "__init__"),
        (bridge.attention_runtime_module._FoldTextDataset, "__len__"),
        (bridge.attention_runtime_module._FoldTextDataset, "__getitem__"),
        (bridge.attention_runtime_module._FoldTextBatchCollator, "__init__"),
        (bridge.attention_runtime_module._FoldTextBatchCollator, "__call__"),
        (
            bridge.HistoricalStage1ContextPredictionHTRProvider,
            "__init__",
        ),
        (
            bridge.HistoricalStage1ContextPredictionHTRProvider,
            "fit_nuisance_inner_ensemble_predict",
        ),
        (
            bridge.HistoricalStage1ContextPredictionHTRProvider,
            "fit_pair_uplift_inner_ensemble_predict",
        ),
        (
            bridge.HistoricalStage1ContextPredictionHTRProvider,
            "fit_effect_variant_inner_ensemble_predict",
        ),
        (
            bridge.HistoricalStage1ContextPredictionHTRProvider,
            "assert_complete_context_prediction_call",
        ),
        (
            bridge.HistoricalStage1ContextPredictionHTRProvider,
            "assert_bundle_placeholder_safety",
        ),
        (
            bridge.HistoricalStage1ContextPredictionHTRProvider,
            "seal_prediction_only_bundle",
        ),
        (bridge.AgenticAttentionVariableForestRunner, "__init__"),
        (bridge.AgenticAttentionVariableForestRunner, "_create_extractor"),
        (bridge.AgenticAttentionVariableForestRunner, "_train_nuisance_model"),
        (bridge.AgenticAttentionVariableForestRunner, "_train_effect_model"),
        (bridge.AgenticAttentionVariableForestRunner, "_predict_nuisance_model"),
        (bridge.AgenticAttentionVariableForestRunner, "_predict_effect_model"),
        (bridge.AgenticAttentionVariableForestRunner, "_make_text_loader"),
        (bridge.AgenticAttentionVariableForestRunner, "_effect_epochs"),
        (bridge.AgenticAttentionVariableForestRunner, "_clip_and_step"),
        (bridge.AgenticAttentionVariableForestRunner, "_cleanup_model"),
        (bridge.AgenticAttentionVariableForestRunner, "_fold_n_jobs"),
        (
            bridge.AgenticAttentionVariableForestRunner,
            "_assert_htr_sentence_encoder_training_state",
        ),
        (
            bridge.AgenticAttentionVariableForestRunner,
            "_assert_htr_sentence_encoder_optimizer_coverage",
        ),
        (bridge.HierarchicalTransformerExtractor, "__init__"),
        (bridge.HierarchicalTransformerExtractor, "_ensure_transformers_initialized"),
        (bridge.HierarchicalTransformerExtractor, "_configure_sentence_encoder_training"),
        (bridge.HierarchicalTransformerExtractor, "fit_tokenizer"),
        (bridge.HierarchicalTransformerExtractor, "forward"),
        (bridge.HierarchicalTransformerExtractor, "make_batch_preprocessor"),
        (bridge.HierarchicalTransformerExtractor, "sentence_encoder_training_audit"),
        (bridge.htr_extractor_module.HierarchicalTransformerBatchPreprocessor, "__init__"),
        (bridge.htr_extractor_module.HierarchicalTransformerBatchPreprocessor, "__call__"),
    ),
)
def test_bridge_rejects_htr_runtime_method_mutations(
    tmp_path,
    monkeypatch,
    owner,
    method_name,
):
    package, producer = _build_package(tmp_path, monkeypatch)

    def changed_method(*_args, **_kwargs):
        return None

    monkeypatch.setattr(owner, method_name, changed_method)
    with pytest.raises(TypeError, match="authenticated runtime method changed"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
        )


@pytest.mark.parametrize(
    "function_name",
    (
        "context_prediction_htr_provider_identity",
        "context_prediction_htr_policy_constants",
        "context_prediction_fit_profile",
        "context_prediction_seed",
        "_train_complete_context_pair_model",
        "_train_complete_context_effect_model",
        "_isolated_seed",
        "_assert_label_free_test_frame",
        "_finite_vector",
        "_bounded_fold_count",
        "_canonical_sha256",
        "_normalize_texts",
        "build_training_pairs",
        "build_candidate_pairs",
        "aggregate_pair_predictions",
        "_predict_htr_pair_delta",
        "_iter_batches",
        "_effect_objective_name",
        "_make_linear_lr_scheduler",
        "_torch_pseudo_outcome_mse_loss_vector",
        "_r_pseudo_outcome",
        "clip_probability",
    ),
)
def test_bridge_rejects_context_htr_helper_mutations(
    tmp_path,
    monkeypatch,
    function_name,
):
    package, producer = _build_package(tmp_path, monkeypatch)

    def changed_function(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        bridge.context_htr_runtime_module,
        function_name,
        changed_function,
    )
    identity_bound = {
        "context_prediction_htr_policy_constants",
        "_canonical_sha256",
    }
    expected_error = ValueError if function_name in identity_bound else TypeError
    expected_match = (
        "composite upstream member identity changed"
        if function_name in identity_bound
        else "Stage-1/HTR runtime symbol path changed"
    )
    with pytest.raises(expected_error, match=expected_match):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
        )


@pytest.mark.parametrize(
    "module_object,function_name",
    (
        (bridge.pair_runtime_module, "build_training_pairs"),
        (bridge.pair_runtime_module, "build_candidate_pairs"),
        (bridge.pair_runtime_module, "aggregate_pair_predictions"),
        (bridge.pair_runtime_module, "_predict_htr_pair_delta"),
        (bridge.pair_runtime_module, "_iter_batches"),
        (bridge.pair_runtime_module, "probability_logit"),
        (bridge.pair_runtime_module, "hopcroft_karp"),
        (bridge.pair_runtime_module, "_empty_pair_frame"),
        (bridge.pair_runtime_module, "expit"),
        (bridge.pair_runtime_module, "logit"),
        (bridge.htr_provider_module, "_normalize_texts"),
        (bridge.htr_provider_module, "_normalize_text"),
        (bridge.calibration_runtime_module, "clip_probability"),
        (bridge.calibration_runtime_module, "_fit_temperature"),
        (bridge.calibration_runtime_module, "_apply_temperature"),
        (bridge.calibration_runtime_module, "_logit"),
        (bridge.calibration_runtime_module, "_sigmoid"),
        (bridge.attention_runtime_module, "_effect_objective_name"),
        (bridge.attention_runtime_module, "_make_linear_lr_scheduler"),
        (
            bridge.attention_runtime_module,
            "_torch_pseudo_outcome_mse_loss_vector",
        ),
        (bridge.attention_runtime_module, "_r_pseudo_outcome"),
        (bridge.attention_runtime_module, "clip_probability"),
    ),
)
def test_bridge_rejects_context_htr_canonical_dependency_mutations(
    tmp_path,
    monkeypatch,
    module_object,
    function_name,
):
    package, producer = _build_package(tmp_path, monkeypatch)

    def changed_function(*_args, **_kwargs):
        return None

    monkeypatch.setattr(module_object, function_name, changed_function)
    with pytest.raises(TypeError, match="Stage-1/HTR runtime symbol path changed"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
        )


@pytest.mark.parametrize(
    "module_object,alias_name",
    (
        (bridge.stage1_model_module, "_NuisanceNet"),
        (bridge.stage1_model_module, "BinaryProbabilityCalibrator"),
        (bridge.stage1_model_module, "_run_crossfit_fold_tasks"),
        (bridge.stage1_model_module, "_bounded_fold_count"),
        (bridge.stage1_model_module, "KFold"),
        (bridge.htr_provider_module, "_bounded_fold_count"),
        (bridge.sklearn_model_selection_module, "KFold"),
        (bridge.sklearn_split_runtime_module, "KFold"),
        (bridge.attention_runtime_module, "_FoldTextDataset"),
        (bridge.attention_runtime_module, "_FoldTextBatchCollator"),
        (bridge.attention_runtime_module, "DataLoader"),
        (bridge.torch_data_runtime_module, "DataLoader"),
        (bridge.htr_extractor_module, "HierarchicalTransformerBatchPreprocessor"),
    ),
)
def test_bridge_rejects_nuisance_stage1_alias_mutations(
    tmp_path,
    monkeypatch,
    module_object,
    alias_name,
):
    package, producer = _build_package(tmp_path, monkeypatch)
    monkeypatch.setattr(module_object, alias_name, object())
    with pytest.raises(TypeError, match="Stage-1/HTR runtime symbol path changed"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
        )


def test_bridge_rejects_nuisance_calibration_policy_mutation(tmp_path, monkeypatch):
    package, producer = _build_package(tmp_path, monkeypatch)
    monkeypatch.setattr(
        bridge.calibration_runtime_module,
        "_DISABLED",
        set(bridge.calibration_runtime_module._DISABLED) | {"mutated"},
    )
    with pytest.raises(ValueError, match="calibration disabled-method policy changed"):
        bridge.derive_exact_nuisance_from_runtime_stable_stage1(
            package,
            runtime_producer=producer,
        )


def test_live_htr_trainability_and_optimizer_audits_use_parameter_state():
    extractor = bridge.HierarchicalTransformerExtractor(
        sentence_encoder_model="hash",
        freeze_sentence_encoder=False,
        transformer_dim=8,
        projection_dim=4,
        hash_embedding_dim=8,
        num_transformer_layers=1,
        num_attention_heads=1,
    )
    # Install a tiny already-initialized stand-in for a real local transformer;
    # this exercises live parameter inspection without loading or fitting one.
    extractor._hash_backend = False
    extractor._sentence_encoder_backend = "transformers"
    extractor._sentence_encoder_model = "local-test-transformer"
    extractor._sentence_encoder = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Linear(4, 3),
    )
    extractor._encoder_initialized = True
    extractor._freeze = False
    for parameter in extractor._sentence_encoder.parameters():
        parameter.requires_grad = True

    runner = object.__new__(bridge.AgenticAttentionVariableForestRunner)
    runner.config = AppliedInferenceConfig()
    runner.config.architecture.feature_extractor_type = "hierarchical_transformer"
    runner.config.architecture.htr_freeze_sentence_encoder = False
    runner.config.architecture.htr_require_live_unfrozen_encoder_attestation = True

    audit = runner._assert_htr_sentence_encoder_training_state(extractor)
    assert audit["all_sentence_encoder_parameters_trainable"] is True
    optimizer = torch.optim.AdamW(extractor._sentence_encoder.parameters(), lr=1e-3)
    runner._assert_htr_sentence_encoder_optimizer_coverage(extractor, optimizer)

    next(extractor._sentence_encoder.parameters()).requires_grad = False
    with pytest.raises(RuntimeError, match="not reflected"):
        runner._assert_htr_sentence_encoder_training_state(extractor)

    for parameter in extractor._sentence_encoder.parameters():
        parameter.requires_grad = True
    unrelated = torch.nn.Parameter(torch.ones(()))
    missing_optimizer = torch.optim.AdamW((unrelated,), lr=1e-3)
    with pytest.raises(RuntimeError, match="optimizer omits"):
        runner._assert_htr_sentence_encoder_optimizer_coverage(
            extractor,
            missing_optimizer,
        )


def test_production_htr_trainability_audit_rejects_hash_fallback():
    extractor = bridge.HierarchicalTransformerExtractor(
        sentence_encoder_model="hash",
        freeze_sentence_encoder=False,
        transformer_dim=8,
        projection_dim=4,
        hash_embedding_dim=8,
        num_transformer_layers=1,
        num_attention_heads=1,
    )
    runner = object.__new__(bridge.AgenticAttentionVariableForestRunner)
    runner.config = AppliedInferenceConfig()
    runner.config.architecture.feature_extractor_type = "hierarchical_transformer"
    runner.config.architecture.htr_freeze_sentence_encoder = False
    runner.config.architecture.htr_require_live_unfrozen_encoder_attestation = True

    with pytest.raises(RuntimeError, match="hash fallback is forbidden"):
        runner._assert_htr_sentence_encoder_training_state(extractor)
