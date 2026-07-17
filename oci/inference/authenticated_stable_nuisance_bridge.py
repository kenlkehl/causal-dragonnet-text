"""Runtime-attested derivation of exact Stage-1 nuisance ensembles.

The current final upstream manifest labels BoW/HTR nuisance outputs as raw
role-aware features.  Under the exact production backend graph, however, each
configured ``signed_mean`` cell for these four families is a deterministic
equal mean of genuine Stage-1 nuisance probabilities:

* BoW treatment predictions;
* HTR treatment predictions;
* BoW outcome predictions; and
* HTR outcome predictions.

Names and role tags alone do not prove that semantic fact.  This bridge only
derives a :class:`SealedExactNuisanceBankExtension` when it can re-authenticate
the complete package and the live ``FinalContextFitUpstreamProducer``, prove
the exact stable/composite/member runtime types and method implementations,
including the exact in-process shared TF-IDF wrapper/service when active,
prove the stable reduction configuration, and bind the selected columns back
to the authenticated raw bank and its exact fit-row lineage.

TF-IDF families are never eligible: their stable family groups mix nuisance
predictions with topic activations.  There is intentionally no package-only or
feature-name-only entry point.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import marshal
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn.model_selection as sklearn_model_selection_module
import sklearn.model_selection._split as sklearn_split_runtime_module
from sklearn.model_selection import KFold
import torch.utils.data as torch_data_runtime_module

from ..models import extractor_factory as extractor_factory_module
from ..models import hierarchical_transformer_extractor as htr_extractor_module
from ..models.hierarchical_transformer_extractor import (
    HTR_SENTENCE_ENCODER_TRAINING_AUDIT_SCHEMA,
    HierarchicalTransformerExtractor,
)
from ..utils import calibration as calibration_runtime_module
from ..utils.calibration import BinaryProbabilityCalibrator
from . import agentic_attention_variable_forest as attention_runtime_module
from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
)
from .context_fit_upstream_gate_provider import CompositeContextFitUpstreamBackend
from . import context_prediction_htr_provider as context_htr_runtime_module
from .context_prediction_htr_provider import (
    CONTEXT_PREDICTION_HTR_PROVIDER_ID,
    ContextPredictionOnlyFeatureBundle,
    HistoricalStage1ContextPredictionHTRProvider,
    context_prediction_fit_profile,
    context_prediction_htr_policy_constants,
    context_prediction_htr_provider_identity,
    context_prediction_seed,
)
from .final_context_fit_r_stack_adapter import (
    EXACT_OUTCOME_PREDICTION,
    EXACT_PROPENSITY_PREDICTION,
    SealedExactNuisanceBankExtension,
)
from .final_context_fit_upstream_bank import (
    AuthenticatedFinalContextFitUpstreamBank,
    FinalContextFitUpstreamProducer,
)
from .neural_query_context_backend import (
    NEURAL_QUERY_CONTEXT_BACKEND_ID,
    NeuralQueryContextBackend,
)
from .stable_context_fit_upstream_backend import (
    STABLE_CONTEXT_FIT_UPSTREAM_BACKEND_ID,
    CrossFitStableUpstreamBackend,
    CrossFitStableUpstreamSchemaConfig,
    PrecommittedRawFeatureFamily,
)
from . import multi_model_forest_stage1 as stage1_model_module
from . import multi_model_pair_uplift as pair_runtime_module
from . import multi_model_agentic_forest as htr_provider_module
from . import stage1_upstream_gate_backend as stage1_runtime_module
from .agentic_attention_variable_forest import AgenticAttentionVariableForestRunner
from .multi_model_agentic_forest import MultiModelHTREvidenceProvider
from .multi_model_forest_stage1 import MultiModelForestStage1HTRProvider
from .stage1_upstream_gate_backend import (
    EFFECTIVE_STAGE1_CONFIG_ID,
    HTR_RUNTIME_SOURCE_ATTESTATION_ID,
    STAGE1_CONTEXT_BACKEND_ID,
    HistoricalStage1ConfigSnapshot,
    HistoricalStage1ContextBackend,
)
from .shared_tfidf_context_fit_service import (
    SHARED_TFIDF_CONTEXT_BACKEND_ID,
    InMemorySharedTfidfContextFitService,
    SharedTfidfContextBackend,
)
from .tfidf_upstream_gate_backend import (
    TFIDF_CONTEXT_BACKEND_ID,
    TfidfTopicOrphanContextBackend,
)

AUTHENTICATED_STABLE_NUISANCE_BRIDGE_ID = "authenticated_stable_stage1_nuisance_bridge_v4"
AUTHENTICATED_STABLE_NUISANCE_DERIVATION_SCHEMA = (
    "authenticated_stable_stage1_nuisance_derivation_v4"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_NAMESPACE = "all_evidence_upstream"
_REQUIRED_TARGETS = (
    (
        "bow_nuisance",
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        "bow_equal_mean_propensity_prediction",
        "bow_nuisance_equal_mean",
        EXACT_PROPENSITY_PREDICTION,
    ),
    (
        "htr_nuisance",
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        "htr_propensity_prediction",
        "htr_nuisance_singleton_mean",
        EXACT_PROPENSITY_PREDICTION,
    ),
    (
        "bow_nuisance",
        OUTCOME_NUISANCE_FEATURE_ROLE,
        "bow_equal_mean_outcome_prediction",
        "bow_nuisance_equal_mean",
        EXACT_OUTCOME_PREDICTION,
    ),
    (
        "htr_nuisance",
        OUTCOME_NUISANCE_FEATURE_ROLE,
        "htr_outcome_prediction",
        "htr_nuisance_singleton_mean",
        EXACT_OUTCOME_PREDICTION,
    ),
)
_TARGET_KEYS = frozenset((kind, role) for kind, role, *_rest in _REQUIRED_TARGETS)

# Capture the implementations at module import.  A runtime monkeypatch of one
# of these semantic boundaries must fail even when an object retains the right
# class name and identity strings.
_AUTHENTICATED_PRODUCER_IDENTITY = FinalContextFitUpstreamProducer.identity
_AUTHENTICATED_STABLE_IDENTITY = CrossFitStableUpstreamBackend.identity
_AUTHENTICATED_STABLE_FIT_PREDICT = CrossFitStableUpstreamBackend.fit_predict
_AUTHENTICATED_COMPOSITE_IDENTITY = CompositeContextFitUpstreamBackend.identity
_AUTHENTICATED_COMPOSITE_FIT_PREDICT = CompositeContextFitUpstreamBackend.fit_predict
_AUTHENTICATED_STAGE1_IDENTITY = HistoricalStage1ContextBackend.identity
_AUTHENTICATED_STAGE1_FIT_PREDICT = HistoricalStage1ContextBackend.fit_predict
_AUTHENTICATED_STAGE1_EFFECTIVE_CONFIG = HistoricalStage1ContextBackend.effective_config_sha256
_AUTHENTICATED_STAGE1_HTR_RUNTIME_SOURCES = (
    HistoricalStage1ContextBackend.htr_runtime_source_attestation
)
_AUTHENTICATED_CONFIG_SNAPSHOT_VERIFY_SOURCE = HistoricalStage1ConfigSnapshot.verify_source
_AUTHENTICATED_CONFIG_SNAPSHOT_APPLIED_CONFIG = HistoricalStage1ConfigSnapshot.applied_config
_AUTHENTICATED_STAGE1_RUNNER = stage1_runtime_module.MultiModelForestStage1Runner
_AUTHENTICATED_STAGE1_BUILD_FEATURE_BUNDLE = (
    stage1_model_module.MultiModelForestStage1Runner._build_feature_bundle
)
_AUTHENTICATED_STAGE1_HTR_PROVIDER = stage1_model_module.MultiModelForestStage1Runner._htr_provider
_AUTHENTICATED_HTR_PROVIDER_ENSURE_RUNNER = MultiModelHTREvidenceProvider._ensure_runner
_AUTHENTICATED_HTR_PROVIDER_INIT = MultiModelHTREvidenceProvider.__init__
_AUTHENTICATED_STAGE1_HTR_PROVIDER_INIT = MultiModelForestStage1HTRProvider.__init__
_AUTHENTICATED_HTR_NUISANCE_INNER = (
    MultiModelForestStage1HTRProvider.fit_nuisance_inner_ensemble_predict
)
_AUTHENTICATED_HTR_NUISANCE_FULL = MultiModelForestStage1HTRProvider.fit_nuisance_full_predict
_AUTHENTICATED_HTR_TEMPORARY_EFFECT_OBJECTIVE = (
    MultiModelForestStage1HTRProvider._temporary_effect_objective
)
_AUTHENTICATED_CONTEXT_HTR_PROVIDER_CLASS = HistoricalStage1ContextPredictionHTRProvider
_AUTHENTICATED_CONTEXT_HTR_BUNDLE_CLASS = ContextPredictionOnlyFeatureBundle
_AUTHENTICATED_CONTEXT_HTR_BUNDLE_INIT = ContextPredictionOnlyFeatureBundle.__init__
_AUTHENTICATED_CONTEXT_HTR_BUNDLE_POST_INIT = ContextPredictionOnlyFeatureBundle.__post_init__
_AUTHENTICATED_CONTEXT_HTR_INIT = HistoricalStage1ContextPredictionHTRProvider.__init__
_AUTHENTICATED_CONTEXT_HTR_IDENTITY = HistoricalStage1ContextPredictionHTRProvider.identity
_AUTHENTICATED_CONTEXT_HTR_NUISANCE = (
    HistoricalStage1ContextPredictionHTRProvider.fit_nuisance_inner_ensemble_predict
)
_AUTHENTICATED_CONTEXT_HTR_PAIR = (
    HistoricalStage1ContextPredictionHTRProvider.fit_pair_uplift_inner_ensemble_predict
)
_AUTHENTICATED_CONTEXT_HTR_EFFECT = (
    HistoricalStage1ContextPredictionHTRProvider.fit_effect_variant_inner_ensemble_predict
)
_AUTHENTICATED_CONTEXT_HTR_ASSERT_COMPLETE = (
    HistoricalStage1ContextPredictionHTRProvider.assert_complete_context_prediction_call
)
_AUTHENTICATED_CONTEXT_HTR_ASSERT_BUNDLE = (
    HistoricalStage1ContextPredictionHTRProvider.assert_bundle_placeholder_safety
)
_AUTHENTICATED_CONTEXT_HTR_SEAL_BUNDLE = (
    HistoricalStage1ContextPredictionHTRProvider.seal_prediction_only_bundle
)
_AUTHENTICATED_CONTEXT_HTR_PROVIDER_IDENTITY_FUNCTION = context_prediction_htr_provider_identity
_AUTHENTICATED_CONTEXT_HTR_POLICY_CONSTANTS_FUNCTION = context_prediction_htr_policy_constants
_AUTHENTICATED_CONTEXT_HTR_POLICY_CONSTANTS = MappingProxyType(
    copy.deepcopy(dict(context_prediction_htr_policy_constants()))
)
_AUTHENTICATED_CONTEXT_HTR_POLICY_CONSTANTS_SHA256 = hashlib.sha256(
    json.dumps(
        dict(_AUTHENTICATED_CONTEXT_HTR_POLICY_CONSTANTS),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
_AUTHENTICATED_CONTEXT_HTR_FIT_PROFILE = context_prediction_fit_profile
_AUTHENTICATED_CONTEXT_HTR_SEED = context_prediction_seed
_AUTHENTICATED_CONTEXT_HTR_PAIR_TRAIN = (
    context_htr_runtime_module._train_complete_context_pair_model
)
_AUTHENTICATED_CONTEXT_HTR_EFFECT_TRAIN = (
    context_htr_runtime_module._train_complete_context_effect_model
)
_AUTHENTICATED_CONTEXT_HTR_ISOLATED_SEED = context_htr_runtime_module._isolated_seed
_AUTHENTICATED_CONTEXT_HTR_LABEL_FREE_ASSERTION = (
    context_htr_runtime_module._assert_label_free_test_frame
)
_AUTHENTICATED_CONTEXT_HTR_FINITE_VECTOR = context_htr_runtime_module._finite_vector
_AUTHENTICATED_CONTEXT_HTR_BOUNDED_FOLDS = context_htr_runtime_module._bounded_fold_count
_AUTHENTICATED_CONTEXT_HTR_CANONICAL_SHA256 = context_htr_runtime_module._canonical_sha256
_AUTHENTICATED_EFFECT_NET_CLASS = attention_runtime_module._EffectNet
_AUTHENTICATED_EFFECT_NET_INIT = attention_runtime_module._EffectNet.__init__
_AUTHENTICATED_EFFECT_NET_FORWARD = attention_runtime_module._EffectNet.forward
_AUTHENTICATED_PAIR_NET_CLASS = pair_runtime_module.HTRPairUpliftNet
_AUTHENTICATED_PAIR_NET_INIT = pair_runtime_module.HTRPairUpliftNet.__init__
_AUTHENTICATED_PAIR_NET_FORWARD = pair_runtime_module.HTRPairUpliftNet.forward
_AUTHENTICATED_PAIR_RESULT_CLASS = pair_runtime_module.PairUpliftFitResult
_AUTHENTICATED_PAIR_RESULT_INIT = pair_runtime_module.PairUpliftFitResult.__init__
_AUTHENTICATED_PAIR_BUILD_TRAINING = pair_runtime_module.build_training_pairs
_AUTHENTICATED_PAIR_BUILD_CANDIDATE = pair_runtime_module.build_candidate_pairs
_AUTHENTICATED_PAIR_AGGREGATE = pair_runtime_module.aggregate_pair_predictions
_AUTHENTICATED_PAIR_PREDICT_DELTA = pair_runtime_module._predict_htr_pair_delta
_AUTHENTICATED_PAIR_ITER_BATCHES = pair_runtime_module._iter_batches
_AUTHENTICATED_PAIR_PROBABILITY_LOGIT = pair_runtime_module.probability_logit
_AUTHENTICATED_PAIR_HOPCROFT_KARP = pair_runtime_module.hopcroft_karp
_AUTHENTICATED_PAIR_EMPTY_FRAME = pair_runtime_module._empty_pair_frame
_AUTHENTICATED_PAIR_EXPIT = pair_runtime_module.expit
_AUTHENTICATED_PAIR_LOGIT = pair_runtime_module.logit
_AUTHENTICATED_EFFECT_OBJECTIVE_NAME = attention_runtime_module._effect_objective_name
_AUTHENTICATED_EFFECT_MAKE_SCHEDULER = attention_runtime_module._make_linear_lr_scheduler
_AUTHENTICATED_EFFECT_PSEUDO_LOSS = attention_runtime_module._torch_pseudo_outcome_mse_loss_vector
_AUTHENTICATED_R_PSEUDO_OUTCOME = attention_runtime_module._r_pseudo_outcome
_AUTHENTICATED_CLIP_PROBABILITY = attention_runtime_module.clip_probability
_AUTHENTICATED_NORMALIZE_TEXTS = htr_provider_module._normalize_texts
_AUTHENTICATED_NORMALIZE_TEXT = htr_provider_module._normalize_text
_AUTHENTICATED_NUISANCE_BOUNDED_FOLD_COUNT = htr_provider_module._bounded_fold_count
_AUTHENTICATED_KFOLD_CLASS = KFold
_AUTHENTICATED_KFOLD_INIT = KFold.__init__
_AUTHENTICATED_KFOLD_SPLIT = KFold.split
_AUTHENTICATED_FOLD_TEXT_DATASET_CLASS = attention_runtime_module._FoldTextDataset
_AUTHENTICATED_FOLD_TEXT_DATASET_INIT = attention_runtime_module._FoldTextDataset.__init__
_AUTHENTICATED_FOLD_TEXT_DATASET_LEN = attention_runtime_module._FoldTextDataset.__len__
_AUTHENTICATED_FOLD_TEXT_DATASET_GETITEM = attention_runtime_module._FoldTextDataset.__getitem__
_AUTHENTICATED_FOLD_TEXT_COLLATOR_CLASS = attention_runtime_module._FoldTextBatchCollator
_AUTHENTICATED_FOLD_TEXT_COLLATOR_INIT = attention_runtime_module._FoldTextBatchCollator.__init__
_AUTHENTICATED_FOLD_TEXT_COLLATOR_CALL = attention_runtime_module._FoldTextBatchCollator.__call__
_AUTHENTICATED_TORCH_DATALOADER_CLASS = torch_data_runtime_module.DataLoader
_AUTHENTICATED_ATTENTION_CREATE_EXTRACTOR = AgenticAttentionVariableForestRunner._create_extractor
_AUTHENTICATED_ATTENTION_RUNNER_INIT = AgenticAttentionVariableForestRunner.__init__
_AUTHENTICATED_NUISANCE_NET_CLASS = attention_runtime_module._NuisanceNet
_AUTHENTICATED_NUISANCE_NET_INIT = attention_runtime_module._NuisanceNet.__init__
_AUTHENTICATED_NUISANCE_NET_FORWARD = attention_runtime_module._NuisanceNet.forward
_AUTHENTICATED_CROSSFIT_FOLD_RUNNER = attention_runtime_module._run_crossfit_fold_tasks
_AUTHENTICATED_CALIBRATOR_CLASS = BinaryProbabilityCalibrator
_AUTHENTICATED_CALIBRATOR_INIT = BinaryProbabilityCalibrator.__init__
_AUTHENTICATED_CALIBRATOR_FIT = inspect.getattr_static(BinaryProbabilityCalibrator, "fit")
_AUTHENTICATED_CALIBRATOR_TRANSFORM = BinaryProbabilityCalibrator.transform
_AUTHENTICATED_CALIBRATION_CLIP = calibration_runtime_module.clip_probability
_AUTHENTICATED_CALIBRATION_FIT_TEMPERATURE = calibration_runtime_module._fit_temperature
_AUTHENTICATED_CALIBRATION_APPLY_TEMPERATURE = calibration_runtime_module._apply_temperature
_AUTHENTICATED_CALIBRATION_LOGIT = calibration_runtime_module._logit
_AUTHENTICATED_CALIBRATION_SIGMOID = calibration_runtime_module._sigmoid
_AUTHENTICATED_CALIBRATION_ISOTONIC = calibration_runtime_module.IsotonicRegression
_AUTHENTICATED_CALIBRATION_DISABLED = frozenset(calibration_runtime_module._DISABLED)
_AUTHENTICATED_ATTENTION_TRAIN_NUISANCE = AgenticAttentionVariableForestRunner._train_nuisance_model
_AUTHENTICATED_ATTENTION_TRAIN_EFFECT = AgenticAttentionVariableForestRunner._train_effect_model
_AUTHENTICATED_ATTENTION_PREDICT_NUISANCE = (
    AgenticAttentionVariableForestRunner._predict_nuisance_model
)
_AUTHENTICATED_ATTENTION_PREDICT_EFFECT = AgenticAttentionVariableForestRunner._predict_effect_model
_AUTHENTICATED_ATTENTION_MAKE_TEXT_LOADER = AgenticAttentionVariableForestRunner._make_text_loader
_AUTHENTICATED_ATTENTION_EFFECT_EPOCHS = AgenticAttentionVariableForestRunner._effect_epochs
_AUTHENTICATED_ATTENTION_CLIP_AND_STEP = AgenticAttentionVariableForestRunner._clip_and_step
_AUTHENTICATED_ATTENTION_CLEANUP_MODEL = AgenticAttentionVariableForestRunner._cleanup_model
_AUTHENTICATED_ATTENTION_FOLD_N_JOBS = AgenticAttentionVariableForestRunner._fold_n_jobs
_AUTHENTICATED_ATTENTION_ASSERT_ENCODER_STATE = (
    AgenticAttentionVariableForestRunner._assert_htr_sentence_encoder_training_state
)
_AUTHENTICATED_ATTENTION_ASSERT_OPTIMIZER_COVERAGE = (
    AgenticAttentionVariableForestRunner._assert_htr_sentence_encoder_optimizer_coverage
)
_AUTHENTICATED_EXTRACTOR_FACTORY = extractor_factory_module.create_feature_extractor
_AUTHENTICATED_HTR_EXTRACTOR_CLASS = HierarchicalTransformerExtractor
_AUTHENTICATED_HTR_EXTRACTOR_INIT = HierarchicalTransformerExtractor.__init__
_AUTHENTICATED_HTR_ENSURE_TRANSFORMERS = (
    HierarchicalTransformerExtractor._ensure_transformers_initialized
)
_AUTHENTICATED_HTR_CONFIGURE_TRAINING = (
    HierarchicalTransformerExtractor._configure_sentence_encoder_training
)
_AUTHENTICATED_HTR_FIT_TOKENIZER = HierarchicalTransformerExtractor.fit_tokenizer
_AUTHENTICATED_HTR_EXTRACTOR_FORWARD = HierarchicalTransformerExtractor.forward
_AUTHENTICATED_HTR_MAKE_BATCH_PREPROCESSOR = (
    HierarchicalTransformerExtractor.make_batch_preprocessor
)
_AUTHENTICATED_HTR_TRAINING_AUDIT = HierarchicalTransformerExtractor.sentence_encoder_training_audit
_AUTHENTICATED_HTR_BATCH_PREPROCESSOR_CLASS = (
    htr_extractor_module.HierarchicalTransformerBatchPreprocessor
)
_AUTHENTICATED_HTR_BATCH_PREPROCESSOR_INIT = (
    htr_extractor_module.HierarchicalTransformerBatchPreprocessor.__init__
)
_AUTHENTICATED_HTR_BATCH_PREPROCESSOR_CALL = (
    htr_extractor_module.HierarchicalTransformerBatchPreprocessor.__call__
)
_AUTHENTICATED_TFIDF_IDENTITY = TfidfTopicOrphanContextBackend.identity
_AUTHENTICATED_TFIDF_FIT_PREDICT = TfidfTopicOrphanContextBackend.fit_predict
_AUTHENTICATED_SHARED_TFIDF_IDENTITY = SharedTfidfContextBackend.identity
_AUTHENTICATED_SHARED_TFIDF_FIT_PREDICT = SharedTfidfContextBackend.fit_predict
_AUTHENTICATED_SHARED_TFIDF_ASSERT_STABLE = SharedTfidfContextBackend._assert_stable
_AUTHENTICATED_SHARED_TFIDF_SERVICE_IDENTITY = InMemorySharedTfidfContextFitService.identity
_AUTHENTICATED_SHARED_TFIDF_SERVICE_ASSERT_SOURCE = (
    InMemorySharedTfidfContextFitService.assert_source_identity
)
_AUTHENTICATED_SHARED_TFIDF_SERVICE_TRANSFORM = (
    InMemorySharedTfidfContextFitService.transform_active_exact
)
_AUTHENTICATED_QUERY_IDENTITY = NeuralQueryContextBackend.identity
_AUTHENTICATED_QUERY_FIT_PREDICT = NeuralQueryContextBackend.fit_predict


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _valid_sha256(value: Any, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized != str(value) or _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return normalized


def _method_code_sha256(owner: type[Any], name: str) -> str:
    value = inspect.getattr_static(owner, name)
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    code = getattr(value, "__code__", None)
    if code is None:
        raise TypeError(f"{owner.__qualname__}.{name} must be implemented in Python")
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def _function_code_sha256(value: Any, *, name: str) -> str:
    code = getattr(value, "__code__", None)
    if code is None:
        raise TypeError(f"{name} must be implemented in Python")
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def _assert_exact_method(owner: type[Any], name: str, expected: Any) -> None:
    if inspect.getattr_static(owner, name) is not expected:
        raise TypeError(f"authenticated runtime method changed: {owner.__qualname__}.{name}")


def _family_width(family: PrecommittedRawFeatureFamily) -> int:
    if family.exact_passthrough_feature_names:
        return len(family.exact_passthrough_feature_names)
    return family.signed_order_width + (2 if family.required else 3)


@dataclass(frozen=True)
class _SelectedStableColumn:
    source_kind: str
    consumer_role: str
    family_ordinal: int
    raw_column_index: int
    output_name: str
    output_kind: str
    semantic: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "source_kind": self.source_kind,
            "consumer_role": self.consumer_role,
            "family_ordinal": self.family_ordinal,
            "raw_column_index": self.raw_column_index,
            "output_name": self.output_name,
            "output_kind": self.output_kind,
            "semantic": self.semantic,
            "reduction": "required_family_signed_mean",
        }


@dataclass(frozen=True)
class _RuntimeProof:
    runtime_producer_identity_sha256: str
    stable_backend_identity_sha256: str
    stable_schema_identity_sha256: str
    stage1_backend_identity_sha256: str
    tfidf_runtime_backend_identity_sha256: str
    tfidf_delegate_backend_identity_sha256: str
    tfidf_shared_wrapper_active: bool
    stage1_config_snapshot_sha256: str
    stage1_effective_config_sha256: str
    htr_runtime_source_attestation_sha256: str
    htr_model_tree_sha256: str
    htr_sentence_encoder_unfrozen_from_snapshot: bool
    htr_sentence_encoder_unfrozen_runtime_attested: bool
    runtime_code_attestation: Mapping[str, str]
    selected_columns: tuple[_SelectedStableColumn, ...]


def _prove_runtime_and_select_columns(
    package: AuthenticatedFinalContextFitUpstreamBank,
    runtime_producer: FinalContextFitUpstreamProducer,
) -> _RuntimeProof:
    if type(package) is not AuthenticatedFinalContextFitUpstreamBank:
        raise TypeError("package must be the exact authenticated final upstream type")
    if type(runtime_producer) is not FinalContextFitUpstreamProducer:
        raise TypeError("runtime_producer must be the exact FinalContextFitUpstreamProducer")
    package.verify_authenticated_content()
    _assert_exact_method(
        FinalContextFitUpstreamProducer, "identity", _AUTHENTICATED_PRODUCER_IDENTITY
    )
    runtime_identity = runtime_producer.identity()
    runtime_identity_sha = _sha256_json(runtime_identity)
    if runtime_identity_sha != package.producer_identity_sha256:
        raise ValueError("runtime producer identity does not match the authenticated package")

    stable = runtime_producer.backend
    if type(stable) is not CrossFitStableUpstreamBackend:
        raise TypeError("runtime producer is not backed by CrossFitStableUpstreamBackend")
    _assert_exact_method(CrossFitStableUpstreamBackend, "identity", _AUTHENTICATED_STABLE_IDENTITY)
    _assert_exact_method(
        CrossFitStableUpstreamBackend, "fit_predict", _AUTHENTICATED_STABLE_FIT_PREDICT
    )
    stable_identity = stable.identity()
    if stable_identity.get("backend") != STABLE_CONTEXT_FIT_UPSTREAM_BACKEND_ID:
        raise ValueError("stable backend identity has the wrong implementation ID")
    config = stable.config
    if type(config) is not CrossFitStableUpstreamSchemaConfig:
        raise TypeError("stable backend config has the wrong closed type")
    if (
        config.namespace != _EXPECTED_NAMESPACE
        or not config.reject_unconfigured_calibrated_sources
        or not config.reject_unconfigured_raw_families
    ):
        raise ValueError("stable schema is not the strict all-evidence production schema")

    composite = stable.backend
    if type(composite) is not CompositeContextFitUpstreamBackend:
        raise TypeError("stable backend child is not the exact composite backend")
    _assert_exact_method(
        CompositeContextFitUpstreamBackend,
        "identity",
        _AUTHENTICATED_COMPOSITE_IDENTITY,
    )
    _assert_exact_method(
        CompositeContextFitUpstreamBackend,
        "fit_predict",
        _AUTHENTICATED_COMPOSITE_FIT_PREDICT,
    )
    members = tuple(composite.backends)
    if len(members) != 3:
        raise TypeError(
            "composite runtime must contain exactly Stage-1, TF-IDF, and neural-query "
            "backends in the authenticated production order"
        )
    stage1, tfidf, query = members
    if (
        type(stage1) is not HistoricalStage1ContextBackend
        or type(query) is not NeuralQueryContextBackend
    ):
        raise TypeError(
            "composite runtime must contain exactly Stage-1 and neural-query backends "
            "in the authenticated production order"
        )
    for owner, identity_method, fit_method, expected_identity, expected_fit in (
        (
            HistoricalStage1ContextBackend,
            "identity",
            "fit_predict",
            _AUTHENTICATED_STAGE1_IDENTITY,
            _AUTHENTICATED_STAGE1_FIT_PREDICT,
        ),
        (
            NeuralQueryContextBackend,
            "identity",
            "fit_predict",
            _AUTHENTICATED_QUERY_IDENTITY,
            _AUTHENTICATED_QUERY_FIT_PREDICT,
        ),
    ):
        _assert_exact_method(owner, identity_method, expected_identity)
        _assert_exact_method(owner, fit_method, expected_fit)
    _assert_exact_method(
        HistoricalStage1ContextBackend,
        "effective_config_sha256",
        _AUTHENTICATED_STAGE1_EFFECTIVE_CONFIG,
    )
    _assert_exact_method(
        HistoricalStage1ContextBackend,
        "htr_runtime_source_attestation",
        _AUTHENTICATED_STAGE1_HTR_RUNTIME_SOURCES,
    )

    tfidf_shared_wrapper_active = type(tfidf) is SharedTfidfContextBackend
    if tfidf_shared_wrapper_active:
        _assert_exact_method(
            SharedTfidfContextBackend,
            "identity",
            _AUTHENTICATED_SHARED_TFIDF_IDENTITY,
        )
        _assert_exact_method(
            SharedTfidfContextBackend,
            "fit_predict",
            _AUTHENTICATED_SHARED_TFIDF_FIT_PREDICT,
        )
        _assert_exact_method(
            SharedTfidfContextBackend,
            "_assert_stable",
            _AUTHENTICATED_SHARED_TFIDF_ASSERT_STABLE,
        )
        service = tfidf.service
        if type(service) is not InMemorySharedTfidfContextFitService:
            raise TypeError("shared TF-IDF runtime has the wrong exact service type")
        _assert_exact_method(
            InMemorySharedTfidfContextFitService,
            "identity",
            _AUTHENTICATED_SHARED_TFIDF_SERVICE_IDENTITY,
        )
        _assert_exact_method(
            InMemorySharedTfidfContextFitService,
            "assert_source_identity",
            _AUTHENTICATED_SHARED_TFIDF_SERVICE_ASSERT_SOURCE,
        )
        _assert_exact_method(
            InMemorySharedTfidfContextFitService,
            "transform_active_exact",
            _AUTHENTICATED_SHARED_TFIDF_SERVICE_TRANSFORM,
        )
        tfidf_delegate = tfidf.backend
    elif type(tfidf) is TfidfTopicOrphanContextBackend:
        service = None
        tfidf_delegate = tfidf
    else:
        raise TypeError(
            "TF-IDF composite member must be the exact context backend or the exact "
            "authenticated shared wrapper"
        )
    if type(tfidf_delegate) is not TfidfTopicOrphanContextBackend:
        raise TypeError("shared TF-IDF runtime delegate has the wrong exact backend type")
    _assert_exact_method(
        TfidfTopicOrphanContextBackend,
        "identity",
        _AUTHENTICATED_TFIDF_IDENTITY,
    )
    _assert_exact_method(
        TfidfTopicOrphanContextBackend,
        "fit_predict",
        _AUTHENTICATED_TFIDF_FIT_PREDICT,
    )

    stage1_identity = stage1.identity()
    tfidf_runtime_identity = tfidf.identity()
    tfidf_identity = tfidf_delegate.identity()
    query_identity = query.identity()
    if stage1_identity.get("backend") != STAGE1_CONTEXT_BACKEND_ID:
        raise ValueError("Stage-1 member has the wrong backend identity")
    if tfidf_identity.get("backend") != TFIDF_CONTEXT_BACKEND_ID:
        raise ValueError("TF-IDF delegate has the wrong backend identity")
    if tfidf_shared_wrapper_active:
        assert service is not None
        if tfidf_runtime_identity.get("backend") != SHARED_TFIDF_CONTEXT_BACKEND_ID:
            raise ValueError("shared TF-IDF wrapper has the wrong backend identity")
        if tfidf_runtime_identity.get("delegate") != tfidf_identity:
            raise ValueError("shared TF-IDF wrapper identity changed its exact delegate")
        if tfidf_runtime_identity.get("service") != service.identity():
            raise ValueError("shared TF-IDF wrapper identity changed its exact service")
    elif tfidf_runtime_identity != tfidf_identity:
        raise RuntimeError("direct TF-IDF runtime and delegate identities differ")
    if query_identity.get("backend") != NEURAL_QUERY_CONTEXT_BACKEND_ID:
        raise ValueError("neural-query member has the wrong backend identity")
    required_stage1_families = set(stage1_identity.get("required_families") or ())
    if not {"bow_nuisance", "htr_nuisance"}.issubset(required_stage1_families):
        raise ValueError("Stage-1 runtime does not require both nuisance model families")
    stage1_config_sha = _valid_sha256(
        stage1_identity.get("stage1_config_sha256"), name="stage1_config_sha256"
    )
    if stage1_identity.get("effective_config_schema_version") != EFFECTIVE_STAGE1_CONFIG_ID:
        raise ValueError("Stage-1 runtime has the wrong effective-config schema")
    effective_config_sha = _valid_sha256(
        stage1_identity.get("effective_config_sha256"),
        name="effective_config_sha256",
    )
    if stage1.effective_config_sha256() != effective_config_sha:
        raise ValueError("Stage-1 identity does not match its complete effective config")
    htr_runtime_sources = stage1_identity.get("htr_runtime_source_attestation")
    if not isinstance(htr_runtime_sources, Mapping):
        raise ValueError("Stage-1 identity has no HTR runtime source attestation")
    if htr_runtime_sources.get("schema_version") != HTR_RUNTIME_SOURCE_ATTESTATION_ID:
        raise ValueError("Stage-1 identity has the wrong HTR runtime source schema")
    observed_htr_runtime_sources = stage1.htr_runtime_source_attestation()
    if dict(htr_runtime_sources) != dict(observed_htr_runtime_sources):
        raise ValueError("Stage-1 identity does not match the live HTR runtime sources")
    htr_runtime_source_sha = _sha256_json(dict(htr_runtime_sources))
    htr_model_tree_sha = _valid_sha256(
        stage1_identity.get("htr_model_tree_sha256"), name="htr_model_tree_sha256"
    )
    if stage1_identity.get("context_prediction_htr_provider_required") is not True:
        raise ValueError("Stage-1 runtime does not require its context-prediction HTR provider")
    if stage1_identity.get("context_prediction_htr_provider_id") != (
        CONTEXT_PREDICTION_HTR_PROVIDER_ID
    ):
        raise ValueError("Stage-1 runtime has the wrong context-prediction HTR provider ID")
    expected_context_htr_identity = context_prediction_htr_provider_identity(
        stage1.config,
        device=stage1.device,
    )
    if stage1_identity.get("context_prediction_htr_provider") != expected_context_htr_identity:
        raise ValueError("Stage-1 context-prediction HTR provider identity is inexact")
    if stage1_identity.get("context_train_pair_or_effect_predictions_consumed") is not False:
        raise ValueError("Stage-1 runtime permits context train placeholders to be consumed")
    if stage1_identity.get("spent_discovery_path_changed") is not False:
        raise ValueError("Stage-1 runtime changed the historical spent-discovery path")
    if (
        stage1_runtime_module.MultiModelForestStage1Runner is not _AUTHENTICATED_STAGE1_RUNNER
        or stage1_runtime_module.HistoricalStage1ContextPredictionHTRProvider
        is not _AUTHENTICATED_CONTEXT_HTR_PROVIDER_CLASS
        or stage1_runtime_module.context_prediction_htr_provider_identity
        is not _AUTHENTICATED_CONTEXT_HTR_PROVIDER_IDENTITY_FUNCTION
        or stage1_runtime_module.CONTEXT_PREDICTION_HTR_PROVIDER_ID
        != CONTEXT_PREDICTION_HTR_PROVIDER_ID
        or stage1_model_module.MultiModelForestStage1Runner is not _AUTHENTICATED_STAGE1_RUNNER
        or stage1_model_module.MultiModelForestStage1Runner._build_feature_bundle
        is not _AUTHENTICATED_STAGE1_BUILD_FEATURE_BUNDLE
        or stage1_model_module.MultiModelForestStage1Runner._htr_provider
        is not _AUTHENTICATED_STAGE1_HTR_PROVIDER
        or stage1_model_module.MultiModelForestStage1HTRProvider
        is not MultiModelForestStage1HTRProvider
        or stage1_model_module.MultiModelHTREvidenceProvider is not MultiModelHTREvidenceProvider
        or attention_runtime_module._NuisanceNet is not _AUTHENTICATED_NUISANCE_NET_CLASS
        or stage1_model_module._NuisanceNet is not _AUTHENTICATED_NUISANCE_NET_CLASS
        or stage1_model_module.BinaryProbabilityCalibrator is not _AUTHENTICATED_CALIBRATOR_CLASS
        or calibration_runtime_module.BinaryProbabilityCalibrator
        is not _AUTHENTICATED_CALIBRATOR_CLASS
        or attention_runtime_module._run_crossfit_fold_tasks
        is not _AUTHENTICATED_CROSSFIT_FOLD_RUNNER
        or stage1_model_module._run_crossfit_fold_tasks is not _AUTHENTICATED_CROSSFIT_FOLD_RUNNER
        or context_htr_runtime_module.HistoricalStage1ContextPredictionHTRProvider
        is not _AUTHENTICATED_CONTEXT_HTR_PROVIDER_CLASS
        or context_htr_runtime_module.ContextPredictionOnlyFeatureBundle
        is not _AUTHENTICATED_CONTEXT_HTR_BUNDLE_CLASS
        or context_htr_runtime_module.MultiModelForestStage1HTRProvider
        is not MultiModelForestStage1HTRProvider
        or context_htr_runtime_module.context_prediction_htr_provider_identity
        is not _AUTHENTICATED_CONTEXT_HTR_PROVIDER_IDENTITY_FUNCTION
        or context_htr_runtime_module.context_prediction_htr_policy_constants
        is not _AUTHENTICATED_CONTEXT_HTR_POLICY_CONSTANTS_FUNCTION
        or context_htr_runtime_module.context_prediction_fit_profile
        is not _AUTHENTICATED_CONTEXT_HTR_FIT_PROFILE
        or context_htr_runtime_module.context_prediction_seed is not _AUTHENTICATED_CONTEXT_HTR_SEED
        or pair_runtime_module.HTRPairUpliftNet is not _AUTHENTICATED_PAIR_NET_CLASS
        or context_htr_runtime_module.HTRPairUpliftNet is not _AUTHENTICATED_PAIR_NET_CLASS
        or pair_runtime_module.PairUpliftFitResult is not _AUTHENTICATED_PAIR_RESULT_CLASS
        or context_htr_runtime_module.PairUpliftFitResult is not _AUTHENTICATED_PAIR_RESULT_CLASS
        or pair_runtime_module.build_training_pairs is not _AUTHENTICATED_PAIR_BUILD_TRAINING
        or context_htr_runtime_module.build_training_pairs is not _AUTHENTICATED_PAIR_BUILD_TRAINING
        or pair_runtime_module.build_candidate_pairs is not _AUTHENTICATED_PAIR_BUILD_CANDIDATE
        or context_htr_runtime_module.build_candidate_pairs
        is not _AUTHENTICATED_PAIR_BUILD_CANDIDATE
        or pair_runtime_module.aggregate_pair_predictions is not _AUTHENTICATED_PAIR_AGGREGATE
        or context_htr_runtime_module.aggregate_pair_predictions
        is not _AUTHENTICATED_PAIR_AGGREGATE
        or pair_runtime_module._predict_htr_pair_delta is not _AUTHENTICATED_PAIR_PREDICT_DELTA
        or context_htr_runtime_module._predict_htr_pair_delta
        is not _AUTHENTICATED_PAIR_PREDICT_DELTA
        or context_htr_runtime_module._train_complete_context_pair_model
        is not _AUTHENTICATED_CONTEXT_HTR_PAIR_TRAIN
        or context_htr_runtime_module._train_complete_context_effect_model
        is not _AUTHENTICATED_CONTEXT_HTR_EFFECT_TRAIN
        or context_htr_runtime_module._isolated_seed is not _AUTHENTICATED_CONTEXT_HTR_ISOLATED_SEED
        or context_htr_runtime_module._assert_label_free_test_frame
        is not _AUTHENTICATED_CONTEXT_HTR_LABEL_FREE_ASSERTION
        or context_htr_runtime_module._finite_vector is not _AUTHENTICATED_CONTEXT_HTR_FINITE_VECTOR
        or context_htr_runtime_module._bounded_fold_count
        is not _AUTHENTICATED_CONTEXT_HTR_BOUNDED_FOLDS
        or context_htr_runtime_module._canonical_sha256
        is not _AUTHENTICATED_CONTEXT_HTR_CANONICAL_SHA256
        or attention_runtime_module._EffectNet is not _AUTHENTICATED_EFFECT_NET_CLASS
        or context_htr_runtime_module._EffectNet is not _AUTHENTICATED_EFFECT_NET_CLASS
        or attention_runtime_module._make_linear_lr_scheduler
        is not _AUTHENTICATED_EFFECT_MAKE_SCHEDULER
        or context_htr_runtime_module._make_linear_lr_scheduler
        is not _AUTHENTICATED_EFFECT_MAKE_SCHEDULER
        or attention_runtime_module._torch_pseudo_outcome_mse_loss_vector
        is not _AUTHENTICATED_EFFECT_PSEUDO_LOSS
        or context_htr_runtime_module._torch_pseudo_outcome_mse_loss_vector
        is not _AUTHENTICATED_EFFECT_PSEUDO_LOSS
        or attention_runtime_module._effect_objective_name
        is not _AUTHENTICATED_EFFECT_OBJECTIVE_NAME
        or context_htr_runtime_module._effect_objective_name
        is not _AUTHENTICATED_EFFECT_OBJECTIVE_NAME
        or attention_runtime_module._r_pseudo_outcome is not _AUTHENTICATED_R_PSEUDO_OUTCOME
        or context_htr_runtime_module._r_pseudo_outcome is not _AUTHENTICATED_R_PSEUDO_OUTCOME
        or attention_runtime_module.clip_probability is not _AUTHENTICATED_CLIP_PROBABILITY
        or context_htr_runtime_module.clip_probability is not _AUTHENTICATED_CLIP_PROBABILITY
        or pair_runtime_module._iter_batches is not _AUTHENTICATED_PAIR_ITER_BATCHES
        or context_htr_runtime_module._iter_batches is not _AUTHENTICATED_PAIR_ITER_BATCHES
        or pair_runtime_module.probability_logit is not _AUTHENTICATED_PAIR_PROBABILITY_LOGIT
        or pair_runtime_module.hopcroft_karp is not _AUTHENTICATED_PAIR_HOPCROFT_KARP
        or pair_runtime_module._empty_pair_frame is not _AUTHENTICATED_PAIR_EMPTY_FRAME
        or pair_runtime_module.expit is not _AUTHENTICATED_PAIR_EXPIT
        or pair_runtime_module.logit is not _AUTHENTICATED_PAIR_LOGIT
        or htr_provider_module._normalize_texts is not _AUTHENTICATED_NORMALIZE_TEXTS
        or context_htr_runtime_module._normalize_texts is not _AUTHENTICATED_NORMALIZE_TEXTS
        or htr_provider_module._normalize_text is not _AUTHENTICATED_NORMALIZE_TEXT
        or htr_provider_module._bounded_fold_count is not _AUTHENTICATED_NUISANCE_BOUNDED_FOLD_COUNT
        or stage1_model_module._bounded_fold_count is not _AUTHENTICATED_NUISANCE_BOUNDED_FOLD_COUNT
        or stage1_model_module.KFold is not _AUTHENTICATED_KFOLD_CLASS
        or sklearn_model_selection_module.KFold is not _AUTHENTICATED_KFOLD_CLASS
        or sklearn_split_runtime_module.KFold is not _AUTHENTICATED_KFOLD_CLASS
        or attention_runtime_module._FoldTextDataset is not _AUTHENTICATED_FOLD_TEXT_DATASET_CLASS
        or attention_runtime_module._FoldTextBatchCollator
        is not _AUTHENTICATED_FOLD_TEXT_COLLATOR_CLASS
        or attention_runtime_module.DataLoader is not _AUTHENTICATED_TORCH_DATALOADER_CLASS
        or torch_data_runtime_module.DataLoader is not _AUTHENTICATED_TORCH_DATALOADER_CLASS
        or calibration_runtime_module.clip_probability is not _AUTHENTICATED_CALIBRATION_CLIP
        or calibration_runtime_module._fit_temperature
        is not _AUTHENTICATED_CALIBRATION_FIT_TEMPERATURE
        or calibration_runtime_module._apply_temperature
        is not _AUTHENTICATED_CALIBRATION_APPLY_TEMPERATURE
        or calibration_runtime_module._logit is not _AUTHENTICATED_CALIBRATION_LOGIT
        or calibration_runtime_module._sigmoid is not _AUTHENTICATED_CALIBRATION_SIGMOID
        or calibration_runtime_module.IsotonicRegression is not _AUTHENTICATED_CALIBRATION_ISOTONIC
        or stage1_model_module.AgenticAttentionVariableForestRunner
        is not AgenticAttentionVariableForestRunner
        or htr_provider_module.AgenticAttentionVariableForestRunner
        is not AgenticAttentionVariableForestRunner
        or attention_runtime_module.create_feature_extractor is not _AUTHENTICATED_EXTRACTOR_FACTORY
        or extractor_factory_module.create_feature_extractor is not _AUTHENTICATED_EXTRACTOR_FACTORY
        or htr_extractor_module.HierarchicalTransformerExtractor
        is not _AUTHENTICATED_HTR_EXTRACTOR_CLASS
        or htr_extractor_module.HierarchicalTransformerBatchPreprocessor
        is not _AUTHENTICATED_HTR_BATCH_PREPROCESSOR_CLASS
    ):
        raise TypeError("authenticated Stage-1/HTR runtime symbol path changed")
    if (
        _sha256_json(dict(context_prediction_htr_policy_constants()))
        != _AUTHENTICATED_CONTEXT_HTR_POLICY_CONSTANTS_SHA256
    ):
        raise ValueError("context-prediction HTR runtime policy constants changed")
    if frozenset(calibration_runtime_module._DISABLED) != _AUTHENTICATED_CALIBRATION_DISABLED:
        raise ValueError("nuisance calibration disabled-method policy changed")
    for owner, method_name, expected in (
        (
            MultiModelHTREvidenceProvider,
            "__init__",
            _AUTHENTICATED_HTR_PROVIDER_INIT,
        ),
        (
            MultiModelHTREvidenceProvider,
            "_ensure_runner",
            _AUTHENTICATED_HTR_PROVIDER_ENSURE_RUNNER,
        ),
        (
            MultiModelForestStage1HTRProvider,
            "__init__",
            _AUTHENTICATED_STAGE1_HTR_PROVIDER_INIT,
        ),
        (
            MultiModelForestStage1HTRProvider,
            "fit_nuisance_inner_ensemble_predict",
            _AUTHENTICATED_HTR_NUISANCE_INNER,
        ),
        (
            MultiModelForestStage1HTRProvider,
            "fit_nuisance_full_predict",
            _AUTHENTICATED_HTR_NUISANCE_FULL,
        ),
        (
            MultiModelForestStage1HTRProvider,
            "_temporary_effect_objective",
            _AUTHENTICATED_HTR_TEMPORARY_EFFECT_OBJECTIVE,
        ),
        (
            ContextPredictionOnlyFeatureBundle,
            "__init__",
            _AUTHENTICATED_CONTEXT_HTR_BUNDLE_INIT,
        ),
        (
            ContextPredictionOnlyFeatureBundle,
            "__post_init__",
            _AUTHENTICATED_CONTEXT_HTR_BUNDLE_POST_INIT,
        ),
        (
            _AUTHENTICATED_PAIR_NET_CLASS,
            "__init__",
            _AUTHENTICATED_PAIR_NET_INIT,
        ),
        (
            _AUTHENTICATED_PAIR_NET_CLASS,
            "forward",
            _AUTHENTICATED_PAIR_NET_FORWARD,
        ),
        (
            _AUTHENTICATED_PAIR_RESULT_CLASS,
            "__init__",
            _AUTHENTICATED_PAIR_RESULT_INIT,
        ),
        (
            _AUTHENTICATED_EFFECT_NET_CLASS,
            "__init__",
            _AUTHENTICATED_EFFECT_NET_INIT,
        ),
        (
            _AUTHENTICATED_EFFECT_NET_CLASS,
            "forward",
            _AUTHENTICATED_EFFECT_NET_FORWARD,
        ),
        (
            _AUTHENTICATED_NUISANCE_NET_CLASS,
            "__init__",
            _AUTHENTICATED_NUISANCE_NET_INIT,
        ),
        (
            _AUTHENTICATED_NUISANCE_NET_CLASS,
            "forward",
            _AUTHENTICATED_NUISANCE_NET_FORWARD,
        ),
        (
            _AUTHENTICATED_CALIBRATOR_CLASS,
            "__init__",
            _AUTHENTICATED_CALIBRATOR_INIT,
        ),
        (
            _AUTHENTICATED_CALIBRATOR_CLASS,
            "fit",
            _AUTHENTICATED_CALIBRATOR_FIT,
        ),
        (
            _AUTHENTICATED_CALIBRATOR_CLASS,
            "transform",
            _AUTHENTICATED_CALIBRATOR_TRANSFORM,
        ),
        (
            _AUTHENTICATED_KFOLD_CLASS,
            "__init__",
            _AUTHENTICATED_KFOLD_INIT,
        ),
        (
            _AUTHENTICATED_KFOLD_CLASS,
            "split",
            _AUTHENTICATED_KFOLD_SPLIT,
        ),
        (
            _AUTHENTICATED_FOLD_TEXT_DATASET_CLASS,
            "__init__",
            _AUTHENTICATED_FOLD_TEXT_DATASET_INIT,
        ),
        (
            _AUTHENTICATED_FOLD_TEXT_DATASET_CLASS,
            "__len__",
            _AUTHENTICATED_FOLD_TEXT_DATASET_LEN,
        ),
        (
            _AUTHENTICATED_FOLD_TEXT_DATASET_CLASS,
            "__getitem__",
            _AUTHENTICATED_FOLD_TEXT_DATASET_GETITEM,
        ),
        (
            _AUTHENTICATED_FOLD_TEXT_COLLATOR_CLASS,
            "__init__",
            _AUTHENTICATED_FOLD_TEXT_COLLATOR_INIT,
        ),
        (
            _AUTHENTICATED_FOLD_TEXT_COLLATOR_CLASS,
            "__call__",
            _AUTHENTICATED_FOLD_TEXT_COLLATOR_CALL,
        ),
        (
            HistoricalStage1ContextPredictionHTRProvider,
            "__init__",
            _AUTHENTICATED_CONTEXT_HTR_INIT,
        ),
        (
            HistoricalStage1ContextPredictionHTRProvider,
            "identity",
            _AUTHENTICATED_CONTEXT_HTR_IDENTITY,
        ),
        (
            HistoricalStage1ContextPredictionHTRProvider,
            "fit_nuisance_inner_ensemble_predict",
            _AUTHENTICATED_CONTEXT_HTR_NUISANCE,
        ),
        (
            HistoricalStage1ContextPredictionHTRProvider,
            "fit_pair_uplift_inner_ensemble_predict",
            _AUTHENTICATED_CONTEXT_HTR_PAIR,
        ),
        (
            HistoricalStage1ContextPredictionHTRProvider,
            "fit_effect_variant_inner_ensemble_predict",
            _AUTHENTICATED_CONTEXT_HTR_EFFECT,
        ),
        (
            HistoricalStage1ContextPredictionHTRProvider,
            "assert_complete_context_prediction_call",
            _AUTHENTICATED_CONTEXT_HTR_ASSERT_COMPLETE,
        ),
        (
            HistoricalStage1ContextPredictionHTRProvider,
            "assert_bundle_placeholder_safety",
            _AUTHENTICATED_CONTEXT_HTR_ASSERT_BUNDLE,
        ),
        (
            HistoricalStage1ContextPredictionHTRProvider,
            "seal_prediction_only_bundle",
            _AUTHENTICATED_CONTEXT_HTR_SEAL_BUNDLE,
        ),
        (
            AgenticAttentionVariableForestRunner,
            "__init__",
            _AUTHENTICATED_ATTENTION_RUNNER_INIT,
        ),
        (
            AgenticAttentionVariableForestRunner,
            "_create_extractor",
            _AUTHENTICATED_ATTENTION_CREATE_EXTRACTOR,
        ),
        (
            AgenticAttentionVariableForestRunner,
            "_train_nuisance_model",
            _AUTHENTICATED_ATTENTION_TRAIN_NUISANCE,
        ),
        (
            AgenticAttentionVariableForestRunner,
            "_train_effect_model",
            _AUTHENTICATED_ATTENTION_TRAIN_EFFECT,
        ),
        (
            AgenticAttentionVariableForestRunner,
            "_predict_nuisance_model",
            _AUTHENTICATED_ATTENTION_PREDICT_NUISANCE,
        ),
        (
            AgenticAttentionVariableForestRunner,
            "_predict_effect_model",
            _AUTHENTICATED_ATTENTION_PREDICT_EFFECT,
        ),
        (
            AgenticAttentionVariableForestRunner,
            "_make_text_loader",
            _AUTHENTICATED_ATTENTION_MAKE_TEXT_LOADER,
        ),
        (
            AgenticAttentionVariableForestRunner,
            "_effect_epochs",
            _AUTHENTICATED_ATTENTION_EFFECT_EPOCHS,
        ),
        (
            AgenticAttentionVariableForestRunner,
            "_clip_and_step",
            _AUTHENTICATED_ATTENTION_CLIP_AND_STEP,
        ),
        (
            AgenticAttentionVariableForestRunner,
            "_cleanup_model",
            _AUTHENTICATED_ATTENTION_CLEANUP_MODEL,
        ),
        (
            AgenticAttentionVariableForestRunner,
            "_fold_n_jobs",
            _AUTHENTICATED_ATTENTION_FOLD_N_JOBS,
        ),
        (
            AgenticAttentionVariableForestRunner,
            "_assert_htr_sentence_encoder_training_state",
            _AUTHENTICATED_ATTENTION_ASSERT_ENCODER_STATE,
        ),
        (
            AgenticAttentionVariableForestRunner,
            "_assert_htr_sentence_encoder_optimizer_coverage",
            _AUTHENTICATED_ATTENTION_ASSERT_OPTIMIZER_COVERAGE,
        ),
        (
            HierarchicalTransformerExtractor,
            "__init__",
            _AUTHENTICATED_HTR_EXTRACTOR_INIT,
        ),
        (
            HierarchicalTransformerExtractor,
            "_ensure_transformers_initialized",
            _AUTHENTICATED_HTR_ENSURE_TRANSFORMERS,
        ),
        (
            HierarchicalTransformerExtractor,
            "_configure_sentence_encoder_training",
            _AUTHENTICATED_HTR_CONFIGURE_TRAINING,
        ),
        (
            HierarchicalTransformerExtractor,
            "fit_tokenizer",
            _AUTHENTICATED_HTR_FIT_TOKENIZER,
        ),
        (
            HierarchicalTransformerExtractor,
            "forward",
            _AUTHENTICATED_HTR_EXTRACTOR_FORWARD,
        ),
        (
            HierarchicalTransformerExtractor,
            "make_batch_preprocessor",
            _AUTHENTICATED_HTR_MAKE_BATCH_PREPROCESSOR,
        ),
        (
            HierarchicalTransformerExtractor,
            "sentence_encoder_training_audit",
            _AUTHENTICATED_HTR_TRAINING_AUDIT,
        ),
        (
            _AUTHENTICATED_HTR_BATCH_PREPROCESSOR_CLASS,
            "__init__",
            _AUTHENTICATED_HTR_BATCH_PREPROCESSOR_INIT,
        ),
        (
            _AUTHENTICATED_HTR_BATCH_PREPROCESSOR_CLASS,
            "__call__",
            _AUTHENTICATED_HTR_BATCH_PREPROCESSOR_CALL,
        ),
    ):
        _assert_exact_method(owner, method_name, expected)

    expected_runtime_source_files = {
        "multi_model_forest_stage1_sha256": stage1_model_module.__file__,
        "multi_model_agentic_forest_sha256": htr_provider_module.__file__,
        "agentic_attention_variable_forest_sha256": attention_runtime_module.__file__,
        "context_prediction_htr_provider_sha256": context_htr_runtime_module.__file__,
        "multi_model_pair_uplift_sha256": pair_runtime_module.__file__,
        "extractor_factory_sha256": extractor_factory_module.__file__,
        "hierarchical_transformer_extractor_sha256": htr_extractor_module.__file__,
        "binary_probability_calibration_sha256": calibration_runtime_module.__file__,
    }
    if set(htr_runtime_sources) != {"schema_version", *expected_runtime_source_files}:
        raise ValueError("Stage-1 HTR runtime source attestation is not the closed schema")
    for name, module_file in expected_runtime_source_files.items():
        observed_sha = _valid_sha256(htr_runtime_sources.get(name), name=name)
        if observed_sha != _sha256_file(Path(module_file).resolve()):
            raise ValueError(f"Stage-1 HTR runtime source changed: {name}")
    current_stage1_code_sha = _sha256_file(Path(stage1_model_module.__file__).resolve())
    if stage1_identity.get("stage1_code_sha256") != current_stage1_code_sha:
        raise ValueError("Stage-1 identity does not match the live model-builder source")
    current_pair_code_sha = _sha256_file(Path(pair_runtime_module.__file__).resolve())
    if stage1_identity.get("pair_code_sha256") != current_pair_code_sha:
        raise ValueError("Stage-1 identity does not match the live pair-model source")
    if config.source_config_sha256 != stage1_config_sha:
        raise ValueError("stable schema and Stage-1 runtime use different config snapshots")
    snapshot = getattr(stage1, "_stage1_config_snapshot", None)
    if type(snapshot) is not HistoricalStage1ConfigSnapshot:
        raise TypeError("Stage-1 runtime has no exact immutable config snapshot")
    _assert_exact_method(
        HistoricalStage1ConfigSnapshot,
        "verify_source",
        _AUTHENTICATED_CONFIG_SNAPSHOT_VERIFY_SOURCE,
    )
    _assert_exact_method(
        HistoricalStage1ConfigSnapshot,
        "applied_config",
        _AUTHENTICATED_CONFIG_SNAPSHOT_APPLIED_CONFIG,
    )
    snapshot.verify_source()
    snapshot_sha = _valid_sha256(snapshot.sha256, name="snapshot.sha256")
    if snapshot_sha != stage1_config_sha:
        raise ValueError("Stage-1 identity does not match its immutable config snapshot")
    snapshot_config = snapshot.applied_config()
    snapshot_freeze_encoder = getattr(
        getattr(snapshot_config, "architecture", None),
        "htr_freeze_sentence_encoder",
        None,
    )
    current_freeze_encoder = getattr(
        getattr(stage1.config, "architecture", None),
        "htr_freeze_sentence_encoder",
        None,
    )
    if snapshot_freeze_encoder is not False:
        raise ValueError("immutable Stage-1 config snapshot must specify the unfrozen HTR encoder")
    if current_freeze_encoder is not snapshot_freeze_encoder:
        raise ValueError("mutable Stage-1 config differs from its immutable HTR policy")
    configured_extractor_type = (
        str(getattr(getattr(stage1.config, "architecture", None), "feature_extractor_type", ""))
        .strip()
        .lower()
    )
    if configured_extractor_type not in {
        "hierarchical_transformer",
        "htr",
        "frozen_llm_pooler",
    }:
        raise ValueError(
            "authenticated HTR nuisance runtime must use the exact hierarchical "
            "transformer construction path"
        )
    if (
        getattr(
            getattr(stage1.config, "architecture", None),
            "htr_require_live_unfrozen_encoder_attestation",
            None,
        )
        is not True
    ):
        raise ValueError(
            "authenticated HTR nuisance runtime must require a live fully "
            "trainable transformer encoder"
        )
    if HTR_SENTENCE_ENCODER_TRAINING_AUDIT_SCHEMA != "htr_sentence_encoder_training_state_v1":
        raise RuntimeError("HTR sentence-encoder training audit contract changed")

    families = tuple(config.raw_families)
    if not all(type(family) is PrecommittedRawFeatureFamily for family in families):
        raise TypeError("stable raw families contain a non-canonical config type")
    target_positions: dict[tuple[str, str], tuple[int, int]] = {}
    offset = 0
    for ordinal, family in enumerate(families, start=1):
        key = family.key
        if key in _TARGET_KEYS:
            if key in target_positions:
                raise ValueError("stable schema contains an ambiguous duplicate nuisance family")
            if not family.required or family.exact_passthrough_feature_names:
                raise ValueError(
                    "eligible Stage-1 nuisance families must use the required signed-mean "
                    "stable reduction"
                )
            identity = family.identity()
            if identity.get("summaries") != [
                "signed_mean",
                "absolute_max",
                "signed_descending_order",
            ]:
                raise ValueError("eligible nuisance family changed its stable reduction")
            target_positions[key] = (ordinal, offset)
        offset += _family_width(family)
    if set(target_positions) != set(_TARGET_KEYS):
        missing = sorted(set(_TARGET_KEYS) - set(target_positions))
        raise ValueError(f"stable schema is missing exact Stage-1 nuisance families: {missing}")

    expected_raw_schema = config.raw_output_schema()
    raw = package.raw_features
    observed_raw_schema = tuple(zip(raw.feature_names, raw.feature_kinds, raw.consumer_roles))
    if observed_raw_schema != expected_raw_schema:
        raise ValueError("authenticated raw bank does not match the runtime stable schema")
    if len(expected_raw_schema) != offset:
        raise RuntimeError("stable raw schema width accounting changed")

    selected: list[_SelectedStableColumn] = []
    for source_kind, role, output_name, output_kind, semantic in _REQUIRED_TARGETS:
        ordinal, column = target_positions[(source_kind, role)]
        schema_name, schema_kind, schema_role = expected_raw_schema[column]
        if schema_kind != source_kind or schema_role != role:
            raise RuntimeError("stable signed-mean column routing is internally inconsistent")
        # The exact column position comes from the authenticated config and the
        # stable reducer's fixed output order.  ``schema_name`` is recorded for
        # audit only and never parsed to infer semantics.
        selected.append(
            _SelectedStableColumn(
                source_kind=source_kind,
                consumer_role=role,
                family_ordinal=ordinal,
                raw_column_index=column,
                output_name=output_name,
                output_kind=output_kind,
                semantic=semantic,
            )
        )

    runtime_code_values = {
        "bridge_module_sha256": _module_sha256(),
        "producer_identity_code_sha256": _method_code_sha256(
            FinalContextFitUpstreamProducer, "identity"
        ),
        "stable_fit_predict_code_sha256": _method_code_sha256(
            CrossFitStableUpstreamBackend, "fit_predict"
        ),
        "composite_fit_predict_code_sha256": _method_code_sha256(
            CompositeContextFitUpstreamBackend, "fit_predict"
        ),
        "stage1_fit_predict_code_sha256": _method_code_sha256(
            HistoricalStage1ContextBackend, "fit_predict"
        ),
        "stage1_effective_config_code_sha256": _method_code_sha256(
            HistoricalStage1ContextBackend, "effective_config_sha256"
        ),
        "stage1_htr_runtime_sources_code_sha256": _method_code_sha256(
            HistoricalStage1ContextBackend, "htr_runtime_source_attestation"
        ),
        "stage1_build_feature_bundle_code_sha256": _method_code_sha256(
            stage1_model_module.MultiModelForestStage1Runner,
            "_build_feature_bundle",
        ),
        "stage1_htr_provider_code_sha256": _method_code_sha256(
            stage1_model_module.MultiModelForestStage1Runner,
            "_htr_provider",
        ),
        "htr_provider_ensure_runner_code_sha256": _method_code_sha256(
            MultiModelHTREvidenceProvider,
            "_ensure_runner",
        ),
        "htr_provider_init_code_sha256": _method_code_sha256(
            MultiModelHTREvidenceProvider,
            "__init__",
        ),
        "stage1_htr_provider_init_code_sha256": _method_code_sha256(
            MultiModelForestStage1HTRProvider,
            "__init__",
        ),
        "htr_nuisance_inner_code_sha256": _method_code_sha256(
            MultiModelForestStage1HTRProvider,
            "fit_nuisance_inner_ensemble_predict",
        ),
        "htr_nuisance_full_code_sha256": _method_code_sha256(
            MultiModelForestStage1HTRProvider,
            "fit_nuisance_full_predict",
        ),
        "htr_temporary_effect_objective_code_sha256": _method_code_sha256(
            MultiModelForestStage1HTRProvider,
            "_temporary_effect_objective",
        ),
        "context_htr_bundle_post_init_code_sha256": _method_code_sha256(
            ContextPredictionOnlyFeatureBundle,
            "__post_init__",
        ),
        "context_htr_bundle_init_code_sha256": _method_code_sha256(
            ContextPredictionOnlyFeatureBundle,
            "__init__",
        ),
        "context_htr_pair_net_init_code_sha256": _method_code_sha256(
            _AUTHENTICATED_PAIR_NET_CLASS,
            "__init__",
        ),
        "context_htr_pair_net_forward_code_sha256": _method_code_sha256(
            _AUTHENTICATED_PAIR_NET_CLASS,
            "forward",
        ),
        "context_htr_pair_result_init_code_sha256": _method_code_sha256(
            _AUTHENTICATED_PAIR_RESULT_CLASS,
            "__init__",
        ),
        "context_htr_build_training_pairs_code_sha256": _function_code_sha256(
            _AUTHENTICATED_PAIR_BUILD_TRAINING,
            name="build_training_pairs",
        ),
        "context_htr_build_candidate_pairs_code_sha256": _function_code_sha256(
            _AUTHENTICATED_PAIR_BUILD_CANDIDATE,
            name="build_candidate_pairs",
        ),
        "context_htr_aggregate_pair_predictions_code_sha256": _function_code_sha256(
            _AUTHENTICATED_PAIR_AGGREGATE,
            name="aggregate_pair_predictions",
        ),
        "context_htr_predict_pair_delta_code_sha256": _function_code_sha256(
            _AUTHENTICATED_PAIR_PREDICT_DELTA,
            name="_predict_htr_pair_delta",
        ),
        "context_htr_pair_iter_batches_code_sha256": _function_code_sha256(
            _AUTHENTICATED_PAIR_ITER_BATCHES,
            name="_iter_batches",
        ),
        "context_htr_pair_probability_logit_code_sha256": _function_code_sha256(
            _AUTHENTICATED_PAIR_PROBABILITY_LOGIT,
            name="probability_logit",
        ),
        "context_htr_pair_hopcroft_karp_code_sha256": _function_code_sha256(
            _AUTHENTICATED_PAIR_HOPCROFT_KARP,
            name="hopcroft_karp",
        ),
        "context_htr_pair_empty_frame_code_sha256": _function_code_sha256(
            _AUTHENTICATED_PAIR_EMPTY_FRAME,
            name="_empty_pair_frame",
        ),
        "context_htr_effect_net_init_code_sha256": _method_code_sha256(
            _AUTHENTICATED_EFFECT_NET_CLASS,
            "__init__",
        ),
        "context_htr_effect_net_forward_code_sha256": _method_code_sha256(
            _AUTHENTICATED_EFFECT_NET_CLASS,
            "forward",
        ),
        "context_htr_nuisance_net_init_code_sha256": _method_code_sha256(
            _AUTHENTICATED_NUISANCE_NET_CLASS,
            "__init__",
        ),
        "context_htr_nuisance_net_forward_code_sha256": _method_code_sha256(
            _AUTHENTICATED_NUISANCE_NET_CLASS,
            "forward",
        ),
        "context_htr_calibrator_init_code_sha256": _method_code_sha256(
            _AUTHENTICATED_CALIBRATOR_CLASS,
            "__init__",
        ),
        "context_htr_calibrator_fit_code_sha256": _method_code_sha256(
            _AUTHENTICATED_CALIBRATOR_CLASS,
            "fit",
        ),
        "context_htr_calibrator_transform_code_sha256": _method_code_sha256(
            _AUTHENTICATED_CALIBRATOR_CLASS,
            "transform",
        ),
        "context_htr_calibration_clip_code_sha256": _function_code_sha256(
            _AUTHENTICATED_CALIBRATION_CLIP,
            name="calibration.clip_probability",
        ),
        "context_htr_calibration_fit_temperature_code_sha256": _function_code_sha256(
            _AUTHENTICATED_CALIBRATION_FIT_TEMPERATURE,
            name="_fit_temperature",
        ),
        "context_htr_calibration_apply_temperature_code_sha256": _function_code_sha256(
            _AUTHENTICATED_CALIBRATION_APPLY_TEMPERATURE,
            name="_apply_temperature",
        ),
        "context_htr_calibration_logit_code_sha256": _function_code_sha256(
            _AUTHENTICATED_CALIBRATION_LOGIT,
            name="calibration._logit",
        ),
        "context_htr_calibration_sigmoid_code_sha256": _function_code_sha256(
            _AUTHENTICATED_CALIBRATION_SIGMOID,
            name="calibration._sigmoid",
        ),
        "context_htr_calibration_module_sha256": _sha256_file(
            Path(calibration_runtime_module.__file__).resolve()
        ),
        "context_htr_crossfit_fold_runner_code_sha256": _function_code_sha256(
            _AUTHENTICATED_CROSSFIT_FOLD_RUNNER,
            name="_run_crossfit_fold_tasks",
        ),
        "context_htr_nuisance_bounded_fold_count_code_sha256": _function_code_sha256(
            _AUTHENTICATED_NUISANCE_BOUNDED_FOLD_COUNT,
            name="multi_model_agentic_forest._bounded_fold_count",
        ),
        "context_htr_nuisance_kfold_init_code_sha256": _method_code_sha256(
            _AUTHENTICATED_KFOLD_CLASS,
            "__init__",
        ),
        "context_htr_nuisance_kfold_split_code_sha256": _method_code_sha256(
            _AUTHENTICATED_KFOLD_CLASS,
            "split",
        ),
        "context_htr_fold_text_dataset_init_code_sha256": _method_code_sha256(
            _AUTHENTICATED_FOLD_TEXT_DATASET_CLASS,
            "__init__",
        ),
        "context_htr_fold_text_dataset_len_code_sha256": _method_code_sha256(
            _AUTHENTICATED_FOLD_TEXT_DATASET_CLASS,
            "__len__",
        ),
        "context_htr_fold_text_dataset_getitem_code_sha256": _method_code_sha256(
            _AUTHENTICATED_FOLD_TEXT_DATASET_CLASS,
            "__getitem__",
        ),
        "context_htr_fold_text_collator_init_code_sha256": _method_code_sha256(
            _AUTHENTICATED_FOLD_TEXT_COLLATOR_CLASS,
            "__init__",
        ),
        "context_htr_fold_text_collator_call_code_sha256": _method_code_sha256(
            _AUTHENTICATED_FOLD_TEXT_COLLATOR_CLASS,
            "__call__",
        ),
        "context_htr_effect_objective_code_sha256": _function_code_sha256(
            _AUTHENTICATED_EFFECT_OBJECTIVE_NAME,
            name="_effect_objective_name",
        ),
        "context_htr_effect_scheduler_code_sha256": _function_code_sha256(
            _AUTHENTICATED_EFFECT_MAKE_SCHEDULER,
            name="_make_linear_lr_scheduler",
        ),
        "context_htr_effect_pseudo_loss_code_sha256": _function_code_sha256(
            _AUTHENTICATED_EFFECT_PSEUDO_LOSS,
            name="_torch_pseudo_outcome_mse_loss_vector",
        ),
        "context_htr_r_pseudo_outcome_code_sha256": _function_code_sha256(
            _AUTHENTICATED_R_PSEUDO_OUTCOME,
            name="_r_pseudo_outcome",
        ),
        "context_htr_clip_probability_code_sha256": _function_code_sha256(
            _AUTHENTICATED_CLIP_PROBABILITY,
            name="clip_probability",
        ),
        "context_htr_provider_identity_code_sha256": _method_code_sha256(
            HistoricalStage1ContextPredictionHTRProvider,
            "identity",
        ),
        "context_htr_provider_init_code_sha256": _method_code_sha256(
            HistoricalStage1ContextPredictionHTRProvider,
            "__init__",
        ),
        "context_htr_nuisance_code_sha256": _method_code_sha256(
            HistoricalStage1ContextPredictionHTRProvider,
            "fit_nuisance_inner_ensemble_predict",
        ),
        "context_htr_pair_code_sha256": _method_code_sha256(
            HistoricalStage1ContextPredictionHTRProvider,
            "fit_pair_uplift_inner_ensemble_predict",
        ),
        "context_htr_effect_code_sha256": _method_code_sha256(
            HistoricalStage1ContextPredictionHTRProvider,
            "fit_effect_variant_inner_ensemble_predict",
        ),
        "context_htr_assert_complete_code_sha256": _method_code_sha256(
            HistoricalStage1ContextPredictionHTRProvider,
            "assert_complete_context_prediction_call",
        ),
        "context_htr_assert_bundle_code_sha256": _method_code_sha256(
            HistoricalStage1ContextPredictionHTRProvider,
            "assert_bundle_placeholder_safety",
        ),
        "context_htr_seal_bundle_code_sha256": _method_code_sha256(
            HistoricalStage1ContextPredictionHTRProvider,
            "seal_prediction_only_bundle",
        ),
        "context_htr_static_identity_code_sha256": _function_code_sha256(
            context_prediction_htr_provider_identity,
            name="context_prediction_htr_provider_identity",
        ),
        "context_htr_policy_constants_code_sha256": _function_code_sha256(
            context_prediction_htr_policy_constants,
            name="context_prediction_htr_policy_constants",
        ),
        "context_htr_policy_constants_sha256": (_AUTHENTICATED_CONTEXT_HTR_POLICY_CONSTANTS_SHA256),
        "context_htr_fit_profile_code_sha256": _function_code_sha256(
            context_prediction_fit_profile,
            name="context_prediction_fit_profile",
        ),
        "context_htr_seed_code_sha256": _function_code_sha256(
            context_prediction_seed,
            name="context_prediction_seed",
        ),
        "context_htr_pair_train_code_sha256": _function_code_sha256(
            context_htr_runtime_module._train_complete_context_pair_model,
            name="_train_complete_context_pair_model",
        ),
        "context_htr_effect_train_code_sha256": _function_code_sha256(
            context_htr_runtime_module._train_complete_context_effect_model,
            name="_train_complete_context_effect_model",
        ),
        "context_htr_isolated_seed_code_sha256": _function_code_sha256(
            context_htr_runtime_module._isolated_seed,
            name="_isolated_seed",
        ),
        "context_htr_label_free_assertion_code_sha256": _function_code_sha256(
            context_htr_runtime_module._assert_label_free_test_frame,
            name="_assert_label_free_test_frame",
        ),
        "context_htr_finite_vector_code_sha256": _function_code_sha256(
            context_htr_runtime_module._finite_vector,
            name="_finite_vector",
        ),
        "context_htr_bounded_folds_code_sha256": _function_code_sha256(
            context_htr_runtime_module._bounded_fold_count,
            name="_bounded_fold_count",
        ),
        "context_htr_canonical_sha256_code_sha256": _function_code_sha256(
            _AUTHENTICATED_CONTEXT_HTR_CANONICAL_SHA256,
            name="_canonical_sha256",
        ),
        "context_htr_normalize_texts_code_sha256": _function_code_sha256(
            _AUTHENTICATED_NORMALIZE_TEXTS,
            name="_normalize_texts",
        ),
        "context_htr_normalize_text_code_sha256": _function_code_sha256(
            _AUTHENTICATED_NORMALIZE_TEXT,
            name="_normalize_text",
        ),
        "attention_runner_init_code_sha256": _method_code_sha256(
            AgenticAttentionVariableForestRunner,
            "__init__",
        ),
        "attention_create_extractor_code_sha256": _method_code_sha256(
            AgenticAttentionVariableForestRunner,
            "_create_extractor",
        ),
        "attention_train_nuisance_code_sha256": _method_code_sha256(
            AgenticAttentionVariableForestRunner,
            "_train_nuisance_model",
        ),
        "attention_train_effect_code_sha256": _method_code_sha256(
            AgenticAttentionVariableForestRunner,
            "_train_effect_model",
        ),
        "attention_predict_nuisance_code_sha256": _method_code_sha256(
            AgenticAttentionVariableForestRunner,
            "_predict_nuisance_model",
        ),
        "attention_predict_effect_code_sha256": _method_code_sha256(
            AgenticAttentionVariableForestRunner,
            "_predict_effect_model",
        ),
        "attention_make_text_loader_code_sha256": _method_code_sha256(
            AgenticAttentionVariableForestRunner,
            "_make_text_loader",
        ),
        "attention_effect_epochs_code_sha256": _method_code_sha256(
            AgenticAttentionVariableForestRunner,
            "_effect_epochs",
        ),
        "attention_clip_and_step_code_sha256": _method_code_sha256(
            AgenticAttentionVariableForestRunner,
            "_clip_and_step",
        ),
        "attention_cleanup_model_code_sha256": _method_code_sha256(
            AgenticAttentionVariableForestRunner,
            "_cleanup_model",
        ),
        "attention_fold_n_jobs_code_sha256": _method_code_sha256(
            AgenticAttentionVariableForestRunner,
            "_fold_n_jobs",
        ),
        "attention_assert_encoder_state_code_sha256": _method_code_sha256(
            AgenticAttentionVariableForestRunner,
            "_assert_htr_sentence_encoder_training_state",
        ),
        "attention_assert_optimizer_coverage_code_sha256": _method_code_sha256(
            AgenticAttentionVariableForestRunner,
            "_assert_htr_sentence_encoder_optimizer_coverage",
        ),
        "extractor_factory_code_sha256": _function_code_sha256(
            extractor_factory_module.create_feature_extractor,
            name="create_feature_extractor",
        ),
        "htr_extractor_init_code_sha256": _method_code_sha256(
            HierarchicalTransformerExtractor,
            "__init__",
        ),
        "htr_ensure_transformers_code_sha256": _method_code_sha256(
            HierarchicalTransformerExtractor,
            "_ensure_transformers_initialized",
        ),
        "htr_configure_training_code_sha256": _method_code_sha256(
            HierarchicalTransformerExtractor,
            "_configure_sentence_encoder_training",
        ),
        "htr_fit_tokenizer_code_sha256": _method_code_sha256(
            HierarchicalTransformerExtractor,
            "fit_tokenizer",
        ),
        "htr_extractor_forward_code_sha256": _method_code_sha256(
            HierarchicalTransformerExtractor,
            "forward",
        ),
        "htr_make_batch_preprocessor_code_sha256": _method_code_sha256(
            HierarchicalTransformerExtractor,
            "make_batch_preprocessor",
        ),
        "htr_batch_preprocessor_init_code_sha256": _method_code_sha256(
            _AUTHENTICATED_HTR_BATCH_PREPROCESSOR_CLASS,
            "__init__",
        ),
        "htr_batch_preprocessor_call_code_sha256": _method_code_sha256(
            _AUTHENTICATED_HTR_BATCH_PREPROCESSOR_CLASS,
            "__call__",
        ),
        "htr_training_audit_code_sha256": _method_code_sha256(
            HierarchicalTransformerExtractor,
            "sentence_encoder_training_audit",
        ),
        "stage1_model_module_sha256": current_stage1_code_sha,
        "pair_model_module_sha256": current_pair_code_sha,
        "context_htr_provider_module_sha256": _sha256_file(
            Path(context_htr_runtime_module.__file__).resolve()
        ),
        "htr_runtime_source_attestation_sha256": htr_runtime_source_sha,
        "config_snapshot_verify_source_code_sha256": _method_code_sha256(
            HistoricalStage1ConfigSnapshot, "verify_source"
        ),
        "config_snapshot_applied_config_code_sha256": _method_code_sha256(
            HistoricalStage1ConfigSnapshot, "applied_config"
        ),
        "tfidf_delegate_identity_code_sha256": _method_code_sha256(
            TfidfTopicOrphanContextBackend, "identity"
        ),
        "tfidf_delegate_fit_predict_code_sha256": _method_code_sha256(
            TfidfTopicOrphanContextBackend, "fit_predict"
        ),
        "query_fit_predict_code_sha256": _method_code_sha256(
            NeuralQueryContextBackend, "fit_predict"
        ),
    }
    if tfidf_shared_wrapper_active:
        runtime_code_values.update(
            {
                "shared_tfidf_identity_code_sha256": _method_code_sha256(
                    SharedTfidfContextBackend, "identity"
                ),
                "shared_tfidf_fit_predict_code_sha256": _method_code_sha256(
                    SharedTfidfContextBackend, "fit_predict"
                ),
                "shared_tfidf_assert_stable_code_sha256": _method_code_sha256(
                    SharedTfidfContextBackend, "_assert_stable"
                ),
                "shared_tfidf_service_identity_code_sha256": _method_code_sha256(
                    InMemorySharedTfidfContextFitService, "identity"
                ),
                "shared_tfidf_service_assert_source_code_sha256": _method_code_sha256(
                    InMemorySharedTfidfContextFitService, "assert_source_identity"
                ),
                "shared_tfidf_service_transform_code_sha256": _method_code_sha256(
                    InMemorySharedTfidfContextFitService, "transform_active_exact"
                ),
            }
        )
    runtime_code = MappingProxyType(runtime_code_values)
    package.verify_authenticated_content()
    if _sha256_json(runtime_producer.identity()) != runtime_identity_sha:
        raise ValueError("runtime producer identity changed during nuisance derivation")
    return _RuntimeProof(
        runtime_producer_identity_sha256=runtime_identity_sha,
        stable_backend_identity_sha256=_sha256_json(stable_identity),
        stable_schema_identity_sha256=_sha256_json(config.identity()),
        stage1_backend_identity_sha256=_sha256_json(stage1_identity),
        tfidf_runtime_backend_identity_sha256=_sha256_json(tfidf_runtime_identity),
        tfidf_delegate_backend_identity_sha256=_sha256_json(tfidf_identity),
        tfidf_shared_wrapper_active=tfidf_shared_wrapper_active,
        stage1_config_snapshot_sha256=snapshot_sha,
        stage1_effective_config_sha256=effective_config_sha,
        htr_runtime_source_attestation_sha256=htr_runtime_source_sha,
        htr_model_tree_sha256=htr_model_tree_sha,
        htr_sentence_encoder_unfrozen_from_snapshot=True,
        htr_sentence_encoder_unfrozen_runtime_attested=True,
        runtime_code_attestation=runtime_code,
        selected_columns=tuple(selected),
    )


def _derivation_digest(
    *,
    package_cache_key: str,
    package_manifest_sha256: str,
    raw_bank_content_sha256: str,
    nuisance_content_sha256: str,
    proof: _RuntimeProof,
) -> str:
    return _derivation_digest_from_records(
        package_cache_key=package_cache_key,
        package_manifest_sha256=package_manifest_sha256,
        raw_bank_content_sha256=raw_bank_content_sha256,
        nuisance_content_sha256=nuisance_content_sha256,
        runtime_producer_identity_sha256=proof.runtime_producer_identity_sha256,
        stable_backend_identity_sha256=proof.stable_backend_identity_sha256,
        stable_schema_identity_sha256=proof.stable_schema_identity_sha256,
        stage1_backend_identity_sha256=proof.stage1_backend_identity_sha256,
        tfidf_runtime_backend_identity_sha256=(proof.tfidf_runtime_backend_identity_sha256),
        tfidf_delegate_backend_identity_sha256=(proof.tfidf_delegate_backend_identity_sha256),
        tfidf_shared_wrapper_active=proof.tfidf_shared_wrapper_active,
        stage1_config_snapshot_sha256=proof.stage1_config_snapshot_sha256,
        stage1_effective_config_sha256=proof.stage1_effective_config_sha256,
        htr_runtime_source_attestation_sha256=(proof.htr_runtime_source_attestation_sha256),
        htr_model_tree_sha256=proof.htr_model_tree_sha256,
        htr_sentence_encoder_unfrozen_from_snapshot=(
            proof.htr_sentence_encoder_unfrozen_from_snapshot
        ),
        htr_sentence_encoder_unfrozen_runtime_attested=(
            proof.htr_sentence_encoder_unfrozen_runtime_attested
        ),
        runtime_code_attestation=proof.runtime_code_attestation,
        selected_columns=tuple(item.payload() for item in proof.selected_columns),
    )


def _derivation_digest_from_records(
    *,
    package_cache_key: str,
    package_manifest_sha256: str,
    raw_bank_content_sha256: str,
    nuisance_content_sha256: str,
    runtime_producer_identity_sha256: str,
    stable_backend_identity_sha256: str,
    stable_schema_identity_sha256: str,
    stage1_backend_identity_sha256: str,
    tfidf_runtime_backend_identity_sha256: str,
    tfidf_delegate_backend_identity_sha256: str,
    tfidf_shared_wrapper_active: bool,
    stage1_config_snapshot_sha256: str,
    stage1_effective_config_sha256: str,
    htr_runtime_source_attestation_sha256: str,
    htr_model_tree_sha256: str,
    htr_sentence_encoder_unfrozen_from_snapshot: bool,
    htr_sentence_encoder_unfrozen_runtime_attested: bool,
    runtime_code_attestation: Mapping[str, str],
    selected_columns: Sequence[Mapping[str, Any]],
) -> str:
    return _sha256_json(
        {
            "schema_version": AUTHENTICATED_STABLE_NUISANCE_DERIVATION_SCHEMA,
            "bridge": AUTHENTICATED_STABLE_NUISANCE_BRIDGE_ID,
            "package_cache_key": package_cache_key,
            "package_manifest_sha256": package_manifest_sha256,
            "raw_bank_content_sha256": raw_bank_content_sha256,
            "nuisance_content_sha256": nuisance_content_sha256,
            "runtime_producer_identity_sha256": runtime_producer_identity_sha256,
            "stable_backend_identity_sha256": stable_backend_identity_sha256,
            "stable_schema_identity_sha256": stable_schema_identity_sha256,
            "stage1_backend_identity_sha256": stage1_backend_identity_sha256,
            "tfidf_runtime_backend_identity_sha256": (tfidf_runtime_backend_identity_sha256),
            "tfidf_delegate_backend_identity_sha256": (tfidf_delegate_backend_identity_sha256),
            "tfidf_shared_wrapper_active": tfidf_shared_wrapper_active,
            "stage1_config_snapshot_sha256": stage1_config_snapshot_sha256,
            "stage1_effective_config_sha256": stage1_effective_config_sha256,
            "htr_runtime_source_attestation_sha256": (htr_runtime_source_attestation_sha256),
            "htr_model_tree_sha256": htr_model_tree_sha256,
            "htr_sentence_encoder_unfrozen_from_snapshot": (
                htr_sentence_encoder_unfrozen_from_snapshot
            ),
            "htr_sentence_encoder_unfrozen_runtime_attested": (
                htr_sentence_encoder_unfrozen_runtime_attested
            ),
            "runtime_code_attestation": dict(runtime_code_attestation),
            "selected_columns": [dict(item) for item in selected_columns],
        }
    )


@dataclass(frozen=True)
class AuthenticatedStableNuisanceDerivation:
    """One runtime-attested bridge result and its estimator-facing extension."""

    package_cache_key: str
    package_manifest_sha256: str
    raw_bank_content_sha256: str
    runtime_producer_identity_sha256: str
    stable_backend_identity_sha256: str
    stable_schema_identity_sha256: str
    stage1_backend_identity_sha256: str
    tfidf_runtime_backend_identity_sha256: str
    tfidf_delegate_backend_identity_sha256: str
    tfidf_shared_wrapper_active: bool
    stage1_config_snapshot_sha256: str
    stage1_effective_config_sha256: str
    htr_runtime_source_attestation_sha256: str
    htr_model_tree_sha256: str
    htr_sentence_encoder_unfrozen_from_snapshot: bool
    htr_sentence_encoder_unfrozen_runtime_attested: bool
    runtime_code_attestation: Mapping[str, str]
    selected_columns: tuple[Mapping[str, Any], ...]
    nuisance: SealedExactNuisanceBankExtension = field(repr=False)
    content_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "package_cache_key",
            "package_manifest_sha256",
            "raw_bank_content_sha256",
            "runtime_producer_identity_sha256",
            "stable_backend_identity_sha256",
            "stable_schema_identity_sha256",
            "stage1_backend_identity_sha256",
            "tfidf_runtime_backend_identity_sha256",
            "tfidf_delegate_backend_identity_sha256",
            "stage1_config_snapshot_sha256",
            "stage1_effective_config_sha256",
            "htr_runtime_source_attestation_sha256",
            "htr_model_tree_sha256",
            "content_sha256",
        ):
            object.__setattr__(self, name, _valid_sha256(getattr(self, name), name=name))
        if type(self.nuisance) is not SealedExactNuisanceBankExtension:
            raise TypeError("nuisance must use the exact sealed nuisance-extension type")
        if not isinstance(self.tfidf_shared_wrapper_active, bool):
            raise TypeError("tfidf_shared_wrapper_active must be boolean")
        if self.htr_sentence_encoder_unfrozen_from_snapshot is not True:
            raise ValueError("immutable HTR policy attestation must be true")
        if self.htr_sentence_encoder_unfrozen_runtime_attested is not True:
            raise ValueError("production HTR runtime attestation must be true")
        code = {
            str(key): _valid_sha256(value, name=str(key))
            for key, value in self.runtime_code_attestation.items()
        }
        columns = tuple(MappingProxyType(dict(item)) for item in self.selected_columns)
        if len(columns) != len(_REQUIRED_TARGETS):
            raise ValueError("selected_columns must contain the four exact nuisance reductions")
        object.__setattr__(self, "runtime_code_attestation", MappingProxyType(code))
        object.__setattr__(self, "selected_columns", columns)
        expected_digest = _derivation_digest_from_records(
            package_cache_key=self.package_cache_key,
            package_manifest_sha256=self.package_manifest_sha256,
            raw_bank_content_sha256=self.raw_bank_content_sha256,
            nuisance_content_sha256=self.nuisance.content_sha256,
            runtime_producer_identity_sha256=self.runtime_producer_identity_sha256,
            stable_backend_identity_sha256=self.stable_backend_identity_sha256,
            stable_schema_identity_sha256=self.stable_schema_identity_sha256,
            stage1_backend_identity_sha256=self.stage1_backend_identity_sha256,
            tfidf_runtime_backend_identity_sha256=(self.tfidf_runtime_backend_identity_sha256),
            tfidf_delegate_backend_identity_sha256=(self.tfidf_delegate_backend_identity_sha256),
            tfidf_shared_wrapper_active=self.tfidf_shared_wrapper_active,
            stage1_config_snapshot_sha256=self.stage1_config_snapshot_sha256,
            stage1_effective_config_sha256=self.stage1_effective_config_sha256,
            htr_runtime_source_attestation_sha256=(self.htr_runtime_source_attestation_sha256),
            htr_model_tree_sha256=self.htr_model_tree_sha256,
            htr_sentence_encoder_unfrozen_from_snapshot=(
                self.htr_sentence_encoder_unfrozen_from_snapshot
            ),
            htr_sentence_encoder_unfrozen_runtime_attested=(
                self.htr_sentence_encoder_unfrozen_runtime_attested
            ),
            runtime_code_attestation=code,
            selected_columns=columns,
        )
        if self.content_sha256 != expected_digest:
            raise ValueError("authenticated stable nuisance derivation digest mismatch")

    def audit_record(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": AUTHENTICATED_STABLE_NUISANCE_DERIVATION_SCHEMA,
                "bridge": AUTHENTICATED_STABLE_NUISANCE_BRIDGE_ID,
                "package_cache_key": self.package_cache_key,
                "package_manifest_sha256": self.package_manifest_sha256,
                "raw_bank_content_sha256": self.raw_bank_content_sha256,
                "nuisance_content_sha256": self.nuisance.content_sha256,
                "runtime_producer_identity_sha256": self.runtime_producer_identity_sha256,
                "stable_backend_identity_sha256": self.stable_backend_identity_sha256,
                "stable_schema_identity_sha256": self.stable_schema_identity_sha256,
                "stage1_backend_identity_sha256": self.stage1_backend_identity_sha256,
                "tfidf_runtime_backend_identity_sha256": (
                    self.tfidf_runtime_backend_identity_sha256
                ),
                "tfidf_delegate_backend_identity_sha256": (
                    self.tfidf_delegate_backend_identity_sha256
                ),
                "tfidf_shared_wrapper_active": self.tfidf_shared_wrapper_active,
                "stage1_config_snapshot_sha256": self.stage1_config_snapshot_sha256,
                "stage1_effective_config_sha256": self.stage1_effective_config_sha256,
                "htr_runtime_source_attestation_sha256": (
                    self.htr_runtime_source_attestation_sha256
                ),
                "htr_model_tree_sha256": self.htr_model_tree_sha256,
                "htr_sentence_encoder_unfrozen_from_snapshot": (
                    self.htr_sentence_encoder_unfrozen_from_snapshot
                ),
                "htr_sentence_encoder_unfrozen_runtime_attested": (
                    self.htr_sentence_encoder_unfrozen_runtime_attested
                ),
                "runtime_code_attestation": dict(self.runtime_code_attestation),
                "selected_columns": [dict(item) for item in self.selected_columns],
                "semantic_inference_from_feature_names": False,
                "tfidf_columns_eligible": False,
                "htr_sentence_encoder_required_unfrozen": True,
                "htr_live_parameter_trainability_checked_before_optimizer": True,
                "htr_optimizer_parameter_coverage_checked": True,
                "context_prediction_htr_provider_runtime_attested": True,
                "context_prediction_htr_pair_and_effect_use_complete_context": True,
                "context_prediction_htr_train_placeholders_consumed": False,
                "context_prediction_frame_labels_accepted": False,
                "spent_discovery_stage1_sources_preserved": True,
                "package_only_derivation_supported": False,
                "first_class_parent_manifest_record": False,
            }
        )

    def verify_authenticated_content(
        self,
        package: AuthenticatedFinalContextFitUpstreamBank,
        *,
        runtime_producer: FinalContextFitUpstreamProducer,
    ) -> None:
        proof = _prove_runtime_and_select_columns(package, runtime_producer)
        raw = package.raw_features
        columns = [item.raw_column_index for item in proof.selected_columns]
        if not np.array_equal(
            self.nuisance.train_oof_values, raw.train_oof_values[:, columns]
        ) or not np.array_equal(
            self.nuisance.outer_heldout_values,
            raw.outer_heldout_values[:, columns],
        ):
            raise ValueError("derived nuisance values differ from the authenticated raw bank")
        self.nuisance.validate_parent(package)
        expected_digest = _derivation_digest(
            package_cache_key=package.cache_key,
            package_manifest_sha256=package.manifest_sha256,
            raw_bank_content_sha256=raw.content_sha256,
            nuisance_content_sha256=self.nuisance.content_sha256,
            proof=proof,
        )
        if (
            self.package_cache_key != package.cache_key
            or self.package_manifest_sha256 != package.manifest_sha256
            or self.raw_bank_content_sha256 != raw.content_sha256
            or self.runtime_producer_identity_sha256 != proof.runtime_producer_identity_sha256
            or self.stable_backend_identity_sha256 != proof.stable_backend_identity_sha256
            or self.stable_schema_identity_sha256 != proof.stable_schema_identity_sha256
            or self.stage1_backend_identity_sha256 != proof.stage1_backend_identity_sha256
            or self.tfidf_runtime_backend_identity_sha256
            != proof.tfidf_runtime_backend_identity_sha256
            or self.tfidf_delegate_backend_identity_sha256
            != proof.tfidf_delegate_backend_identity_sha256
            or self.tfidf_shared_wrapper_active != proof.tfidf_shared_wrapper_active
            or self.stage1_config_snapshot_sha256 != proof.stage1_config_snapshot_sha256
            or self.stage1_effective_config_sha256 != proof.stage1_effective_config_sha256
            or self.htr_runtime_source_attestation_sha256
            != proof.htr_runtime_source_attestation_sha256
            or self.htr_model_tree_sha256 != proof.htr_model_tree_sha256
            or self.htr_sentence_encoder_unfrozen_from_snapshot
            != proof.htr_sentence_encoder_unfrozen_from_snapshot
            or self.htr_sentence_encoder_unfrozen_runtime_attested
            != proof.htr_sentence_encoder_unfrozen_runtime_attested
            or dict(self.runtime_code_attestation) != dict(proof.runtime_code_attestation)
            or tuple(dict(item) for item in self.selected_columns)
            != tuple(dict(item.payload()) for item in proof.selected_columns)
            or self.content_sha256 != expected_digest
        ):
            raise ValueError("authenticated stable nuisance derivation changed")


def derive_exact_nuisance_from_runtime_stable_stage1(
    package: AuthenticatedFinalContextFitUpstreamBank,
    *,
    runtime_producer: FinalContextFitUpstreamProducer,
) -> AuthenticatedStableNuisanceDerivation:
    """Derive four exact nuisance columns after complete runtime attestation."""

    proof = _prove_runtime_and_select_columns(package, runtime_producer)
    raw = package.raw_features
    indices = [item.raw_column_index for item in proof.selected_columns]
    train_values = np.array(raw.train_oof_values[:, indices], dtype=float, copy=True)
    heldout_values = np.array(raw.outer_heldout_values[:, indices], dtype=float, copy=True)
    train_provenance = tuple(
        tuple(row[index] for index in indices) for row in raw.train_oof_fit_row_provenance
    )
    heldout_provenance = tuple(
        tuple(row[index] for index in indices) for row in raw.outer_heldout_fit_row_provenance
    )
    nuisance = SealedExactNuisanceBankExtension.seal_for_package(
        package,
        prediction_names=tuple(item.output_name for item in proof.selected_columns),
        prediction_kinds=tuple(item.output_kind for item in proof.selected_columns),
        prediction_semantics=tuple(item.semantic for item in proof.selected_columns),
        train_oof_values=train_values,
        outer_heldout_values=heldout_values,
        train_oof_fit_row_provenance=train_provenance,
        outer_heldout_fit_row_provenance=heldout_provenance,
    )
    digest = _derivation_digest(
        package_cache_key=package.cache_key,
        package_manifest_sha256=package.manifest_sha256,
        raw_bank_content_sha256=raw.content_sha256,
        nuisance_content_sha256=nuisance.content_sha256,
        proof=proof,
    )
    result = AuthenticatedStableNuisanceDerivation(
        package_cache_key=package.cache_key,
        package_manifest_sha256=package.manifest_sha256,
        raw_bank_content_sha256=raw.content_sha256,
        runtime_producer_identity_sha256=proof.runtime_producer_identity_sha256,
        stable_backend_identity_sha256=proof.stable_backend_identity_sha256,
        stable_schema_identity_sha256=proof.stable_schema_identity_sha256,
        stage1_backend_identity_sha256=proof.stage1_backend_identity_sha256,
        tfidf_runtime_backend_identity_sha256=(proof.tfidf_runtime_backend_identity_sha256),
        tfidf_delegate_backend_identity_sha256=(proof.tfidf_delegate_backend_identity_sha256),
        tfidf_shared_wrapper_active=proof.tfidf_shared_wrapper_active,
        stage1_config_snapshot_sha256=proof.stage1_config_snapshot_sha256,
        stage1_effective_config_sha256=proof.stage1_effective_config_sha256,
        htr_runtime_source_attestation_sha256=(proof.htr_runtime_source_attestation_sha256),
        htr_model_tree_sha256=proof.htr_model_tree_sha256,
        htr_sentence_encoder_unfrozen_from_snapshot=(
            proof.htr_sentence_encoder_unfrozen_from_snapshot
        ),
        htr_sentence_encoder_unfrozen_runtime_attested=(
            proof.htr_sentence_encoder_unfrozen_runtime_attested
        ),
        runtime_code_attestation=proof.runtime_code_attestation,
        selected_columns=tuple(item.payload() for item in proof.selected_columns),
        nuisance=nuisance,
        content_sha256=digest,
    )
    result.verify_authenticated_content(package, runtime_producer=runtime_producer)
    return result


__all__ = [
    "AUTHENTICATED_STABLE_NUISANCE_BRIDGE_ID",
    "AUTHENTICATED_STABLE_NUISANCE_DERIVATION_SCHEMA",
    "AuthenticatedStableNuisanceDerivation",
    "derive_exact_nuisance_from_runtime_stable_stage1",
]
