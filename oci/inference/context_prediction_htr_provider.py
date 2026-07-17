"""Prediction-only HTR provider for disjoint context -> heldout calls.

The historical Stage-1 builder needs train-side arrays to keep a common matrix
schema, even when its caller consumes only predictions for a disjoint, label-
free frame.  Its default HTR provider therefore spends five effect-model fits
on train OOF values for each effect source and five more on matched-pair train
OOF values.  Those values are useful in the ordinary Stage-1/discovery path,
but they are discarded by :class:`HistoricalStage1ContextBackend`.

This provider is injected only by that context backend.  It delegates nuisance
cross-fitting unchanged, preserving the OOF residuals and per-fold probability
calibrators.  It replaces the pair and two effect inner ensembles with one fit
each on the complete allowed context, then predicts the disjoint label-free
rows.  Finite zero train placeholders preserve the historical bundle schema;
the context backend verifies that their corresponding test columns are finite
before discarding the complete train matrix.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..config import AppliedInferenceConfig
from .agentic_attention_variable_forest import (
    _EffectNet,
    _effect_objective_name,
    _make_linear_lr_scheduler,
    _r_pseudo_outcome,
    _torch_pseudo_outcome_mse_loss_vector,
    clip_probability,
)
from .multi_model_forest_stage1 import MultiModelForestStage1HTRProvider
from .multi_model_agentic_forest import _normalize_texts
from .multi_model_pair_uplift import (
    HTRPairUpliftNet,
    PairUpliftFitResult,
    _iter_batches,
    _predict_htr_pair_delta,
    aggregate_pair_predictions,
    build_candidate_pairs,
    build_training_pairs,
)

CONTEXT_PREDICTION_HTR_PROVIDER_ID = "historical_stage1_context_prediction_htr_v1"
CONTEXT_PREDICTION_HTR_SEED_POLICY_ID = "outer_fold_component_isolated_rng_v1"
CONTEXT_PREDICTION_HTR_PLACEHOLDER_POLICY_ID = "finite_zero_train_placeholder_v1"

_SEED_ROOT = 610_000
_SEED_OFFSETS = MappingProxyType(
    {
        "nuisance": 1_000,
        "matched_pair_uplift": 2_000,
        "pseudo_outcome_mse": 3_000,
        "squared_r_loss": 4_000,
    }
)
_PAIR_FEATURE_NAMES = (
    "htr__matched_pair_uplift_delta_logit",
    "htr__matched_pair_treated_outcome_prob",
)
_EFFECT_FEATURE_NAMES = (
    "htr__effect_pseudo_target_pred",
    "htr__effect_weighted_r_tau_pred",
)
_PLACEHOLDER_FEATURE_NAMES = (*_PAIR_FEATURE_NAMES, *_EFFECT_FEATURE_NAMES)
_PLACEHOLDER_VALUE = 0.0
_FORBIDDEN_PREDICTION_COLUMNS = frozenset(
    {
        "true_ite",
        "true_ite_prob",
        "ite",
        "oracle_ite",
    }
)


@dataclass(frozen=True)
class ContextPredictionOnlyFeatureBundle:
    """Sealed test-side view with no train values or diagnostic artifacts."""

    x_test: np.ndarray
    w_test: np.ndarray
    x_names: tuple[str, ...]
    w_names: tuple[str, ...]
    feature_rows: tuple[Mapping[str, Any], ...]
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        x_test = np.asarray(self.x_test, dtype=np.float32)
        w_test = np.asarray(self.w_test, dtype=np.float32)
        if x_test.ndim != 2 or w_test.ndim != 2 or x_test.shape[0] != w_test.shape[0]:
            raise ValueError("sealed context-prediction matrices have invalid shapes")
        if x_test.shape[1] != len(self.x_names) or w_test.shape[1] != len(self.w_names):
            raise ValueError("sealed context-prediction names do not match matrix widths")
        if not np.all(np.isfinite(x_test)) or not np.all(np.isfinite(w_test)):
            raise ValueError("sealed context-prediction matrices must be finite")
        x_test = np.array(x_test, copy=True)
        w_test = np.array(w_test, copy=True)
        x_test.setflags(write=False)
        w_test.setflags(write=False)
        object.__setattr__(self, "x_test", x_test)
        object.__setattr__(self, "w_test", w_test)
        object.__setattr__(self, "x_names", tuple(map(str, self.x_names)))
        object.__setattr__(self, "w_names", tuple(map(str, self.w_names)))
        object.__setattr__(
            self,
            "feature_rows",
            tuple(MappingProxyType(copy.deepcopy(dict(row))) for row in self.feature_rows),
        )
        object.__setattr__(
            self,
            "audit",
            MappingProxyType(copy.deepcopy(dict(self.audit))),
        )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def context_prediction_htr_policy_constants() -> Mapping[str, Any]:
    """Return every mutable module-level value that can change semantics."""

    return {
        "provider_id": CONTEXT_PREDICTION_HTR_PROVIDER_ID,
        "seed_policy_id": CONTEXT_PREDICTION_HTR_SEED_POLICY_ID,
        "placeholder_policy_id": CONTEXT_PREDICTION_HTR_PLACEHOLDER_POLICY_ID,
        "seed_root": int(_SEED_ROOT),
        "seed_component_offsets": {
            str(key): int(value) for key, value in sorted(_SEED_OFFSETS.items())
        },
        "pair_feature_names": list(_PAIR_FEATURE_NAMES),
        "effect_feature_names": list(_EFFECT_FEATURE_NAMES),
        "placeholder_feature_names": list(_PLACEHOLDER_FEATURE_NAMES),
        "placeholder_value": float(_PLACEHOLDER_VALUE),
        "forbidden_prediction_columns": sorted(_FORBIDDEN_PREDICTION_COLUMNS),
    }


def _bounded_fold_count(requested: int, n_rows: int) -> int:
    if int(n_rows) < 2:
        raise ValueError("HTR context prediction requires at least two context rows")
    return max(2, min(int(requested), int(n_rows)))


def context_prediction_seed(*, outer_fold: int, component: str) -> int:
    normalized = str(component).strip()
    if normalized not in _SEED_OFFSETS:
        raise ValueError(f"unknown context-prediction HTR component: {component!r}")
    if int(outer_fold) < 0:
        raise ValueError("outer_fold must be non-negative")
    return int(_SEED_ROOT + 10_000 * int(outer_fold) + _SEED_OFFSETS[normalized])


@contextmanager
def _isolated_seed(seed: int, device: torch.device):
    """Restore caller RNG state after one reproducibly seeded HTR component."""

    seed = int(seed)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    normalized_device = torch.device(device)
    cuda_devices: list[int] = []
    if normalized_device.type == "cuda" and torch.cuda.is_available():
        cuda_devices = [
            int(
                normalized_device.index
                if normalized_device.index is not None
                else torch.cuda.current_device()
            )
        ]
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def context_prediction_fit_profile(
    *,
    n_context_rows: int,
    nuisance_folds: int,
    effect_folds: int,
) -> Mapping[str, Any]:
    nuisance_count = _bounded_fold_count(nuisance_folds, n_context_rows)
    legacy_effect_count = _bounded_fold_count(effect_folds, n_context_rows)
    legacy_total = nuisance_count + 3 * legacy_effect_count
    optimized_total = nuisance_count + 3
    return {
        "schema_version": CONTEXT_PREDICTION_HTR_PROVIDER_ID,
        "n_context_rows": int(n_context_rows),
        "cross_fitted_nuisance_model_attempts": int(nuisance_count),
        "complete_context_pair_model_attempts": 1,
        "complete_context_effect_model_attempts": 2,
        "legacy_inner_ensemble_model_attempts": int(legacy_total),
        "context_prediction_model_attempts": int(optimized_total),
        "model_attempt_reduction": int(legacy_total - optimized_total),
    }


def context_prediction_htr_provider_identity(
    config: AppliedInferenceConfig,
    *,
    device: str | torch.device,
) -> Mapping[str, Any]:
    if type(config) is not AppliedInferenceConfig:
        raise TypeError("context-prediction HTR config must be AppliedInferenceConfig")
    avf = config.architecture.agentic_attention_variable_forest
    forest = config.architecture.multi_model_forest
    nuisance_folds = int(avf.nuisance_folds)
    effect_folds = int(avf.effect_folds)
    if nuisance_folds < 2 or effect_folds < 2:
        raise ValueError("HTR nuisance/effect fold counts must be at least two")
    if config.architecture.htr_freeze_sentence_encoder is not False:
        raise ValueError("context-prediction HTR requires an unfrozen HTR encoder")
    if config.architecture.htr_require_live_unfrozen_encoder_attestation is not True:
        raise ValueError("context-prediction HTR requires live encoder attestation")
    forest_fold_policy = str(config.architecture.multi_model_forest.htr_fold_parallelism).strip()
    avf_fold_policy = str(avf.fold_parallelism).strip()
    if forest_fold_policy != "1" or avf_fold_policy != "1":
        raise ValueError("context-prediction HTR requires the derived serial fold policy")
    if str(config.outcome_type).strip().lower() != "binary":
        raise ValueError("context-prediction HTR matched-pair policy requires binary outcomes")
    if not bool(forest.htr_evidence_enabled):
        raise ValueError("context-prediction HTR evidence must be enabled")
    if not bool(forest.matched_pair_uplift_enabled) or not bool(forest.matched_pair_htr_enabled):
        raise ValueError("context-prediction HTR matched-pair fitting must be enabled")
    payload = {
        "provider": CONTEXT_PREDICTION_HTR_PROVIDER_ID,
        "seed_policy": CONTEXT_PREDICTION_HTR_SEED_POLICY_ID,
        "seed_root": int(_SEED_ROOT),
        "seed_component_offsets": {
            str(key): int(value) for key, value in sorted(_SEED_OFFSETS.items())
        },
        "train_placeholder_policy": CONTEXT_PREDICTION_HTR_PLACEHOLDER_POLICY_ID,
        "train_placeholder_value": _PLACEHOLDER_VALUE,
        "placeholder_feature_names": list(_PLACEHOLDER_FEATURE_NAMES),
        "requested_nuisance_folds": nuisance_folds,
        "requested_effect_folds": effect_folds,
        "multi_model_htr_fold_parallelism": forest_fold_policy,
        "effective_htr_runner_fold_parallelism": avf_fold_policy,
        "nuisance_fold_execution_policy": "required_serial",
        "htr_evidence_enabled": True,
        "matched_pair_uplift_enabled": True,
        "matched_pair_htr_enabled": True,
        "configured_legacy_model_attempts": nuisance_folds + 3 * effect_folds,
        "configured_context_prediction_model_attempts": nuisance_folds + 3,
        "nuisance_path": "unchanged_inner_crossfit_with_existing_per_fold_calibrators",
        "pair_path": "one_complete_allowed_context_fit_label_free_prediction",
        "degenerate_pair_policy": "deterministic_zero_delta_without_model",
        "effect_paths": [
            "one_complete_allowed_context_pseudo_outcome_mse_fit",
            "one_complete_allowed_context_squared_r_loss_fit",
        ],
        "effect_targets_use_context_oof_nuisance_predictions": True,
        "prediction_frame_labels_accepted": False,
        "context_train_pair_or_effect_predictions_consumed": False,
        "test_predictions_must_be_finite_before_train_placeholder_cleaning": True,
        "sentence_encoder_unfrozen": True,
        "live_encoder_state_checked_for_every_new_model": True,
        "optimizer_encoder_coverage": (
            "explicit_live_runtime_check_for_pair_and_both_effect_models"
        ),
        "spent_discovery_path_changed": False,
        "device": str(torch.device(device)),
        # Keep the exact policy closed by digest without placing prohibited
        # benchmark/oracle column spellings in a public backend identity.  The
        # runtime bridge separately authenticates the complete constant map.
        "runtime_policy_constants_sha256": _canonical_sha256(
            context_prediction_htr_policy_constants()
        ),
    }
    return {**payload, "identity_sha256": _canonical_sha256(payload)}


def _assert_label_free_test_frame(
    *,
    runner: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    forbidden = set(_FORBIDDEN_PREDICTION_COLUMNS)
    forbidden.add(str(runner.config.treatment_column))
    forbidden.add(str(runner.config.outcome_column))
    exposed = sorted(forbidden & set(map(str, test_df.columns)))
    if exposed:
        raise ValueError(
            "context-prediction HTR requires a label-free prediction frame; " f"found {exposed}"
        )
    expected_columns = {"_oci_row_id", str(runner.config.text_column)}
    observed_columns = set(map(str, test_df.columns))
    if observed_columns != expected_columns:
        raise ValueError(
            "context-prediction HTR prediction frame must contain exactly row ID and text"
        )
    train_ids = tuple(int(value) for value in train_df["_oci_row_id"])
    test_ids = tuple(int(value) for value in test_df["_oci_row_id"])
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise ValueError("context and prediction row IDs must each be unique")
    if set(train_ids) & set(test_ids):
        raise ValueError("context and prediction rows must be disjoint")


def _finite_vector(values: Any, *, length: int, name: str) -> np.ndarray:
    output = np.asarray(values, dtype=float)
    if output.shape != (int(length),) or not np.all(np.isfinite(output)):
        raise ValueError(f"{name} must be one finite vector of length {length}")
    return output


def _train_complete_context_pair_model(
    *,
    runner: Any,
    pairs: pd.DataFrame,
    outer_fold: int,
) -> Optional[HTRPairUpliftNet]:
    if pairs.empty or len(np.unique(pairs["label"].to_numpy(dtype=int))) < 2:
        return None
    extractor = runner._create_extractor()
    hidden_dim = int(getattr(runner.config.architecture, "causal_head_hidden_outcome_dim", 64))
    model = HTRPairUpliftNet(extractor=extractor, hidden_dim=hidden_dim).to(runner.device)
    model.extractor.fit_tokenizer(
        pairs["control_text"].astype(str).tolist() + pairs["treated_text"].astype(str).tolist()
    )
    runner._assert_htr_sentence_encoder_training_state(model.extractor)
    training = runner.config.training
    batch_size = training.effect_batch_size or training.batch_size
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=training.learning_rate,
        weight_decay=getattr(training, "weight_decay", 0.01),
    )
    runner._assert_htr_sentence_encoder_optimizer_coverage(model.extractor, optimizer)
    labels = torch.as_tensor(
        pairs["label"].to_numpy(dtype=np.float32),
        device=runner.device,
    )
    base_logits = torch.as_tensor(
        pairs["base_logit"].to_numpy(dtype=np.float32),
        device=runner.device,
    )
    control_texts = pairs["control_text"].astype(str).tolist()
    treated_texts = pairs["treated_text"].astype(str).tolist()
    for epoch in range(1, runner._effect_epochs() + 1):
        model.train()
        for positions in _iter_batches(
            len(pairs),
            int(batch_size),
            shuffle=True,
            seed=55_000 + 100 * int(epoch),
        ):
            optimizer.zero_grad(set_to_none=True)
            delta = model(
                [control_texts[int(position)] for position in positions],
                [treated_texts[int(position)] for position in positions],
            )
            loss = F.binary_cross_entropy_with_logits(
                base_logits[positions] + delta,
                labels[positions],
            )
            loss.backward()
            optimizer.step()
    runner._assert_htr_sentence_encoder_training_state(model.extractor)
    return model


def _train_complete_context_effect_model(
    *,
    runner: Any,
    model: _EffectNet,
    train_df: pd.DataFrame,
    positions: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    e_clipped: np.ndarray,
    m_clipped: np.ndarray,
    y_resid: np.ndarray,
    t_resid: np.ndarray,
) -> None:
    """Train the two precommitted effect objectives with live encoder audits."""

    training = runner.config.training
    model.extractor.fit_tokenizer(
        train_df.iloc[positions][runner.config.text_column].astype(str).tolist()
    )
    runner._assert_htr_sentence_encoder_training_state(model.extractor)
    loader = runner._make_text_loader(
        model,
        train_df,
        positions,
        fields={
            "outcome": np.asarray(y, dtype=np.float32),
            "treatment": np.asarray(t, dtype=np.float32),
            "e_hat": np.asarray(e_clipped, dtype=np.float32),
            "m_hat": np.asarray(m_clipped, dtype=np.float32),
            "y_residual": np.asarray(y_resid, dtype=np.float32),
            "t_residual": np.asarray(t_resid, dtype=np.float32),
        },
        shuffle=True,
        total_folds=1,
        batch_size=getattr(training, "effect_batch_size", None),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=training.learning_rate,
        weight_decay=getattr(training, "weight_decay", 0.01),
    )
    runner._assert_htr_sentence_encoder_optimizer_coverage(model.extractor, optimizer)
    scheduler = _make_linear_lr_scheduler(
        optimizer,
        training,
        max(1, len(loader)),
        epochs_override=runner._effect_epochs(),
    )
    objective = _effect_objective_name(runner.avf_config)
    if objective not in {"pseudo_outcome_mse", "squared_r_loss"}:
        raise ValueError("complete-context effect trainer received an uncommitted objective")
    for _epoch in range(1, runner._effect_epochs() + 1):
        model.train()
        for batch in loader:
            batch_y_resid = batch["y_residual"].to(runner.device, non_blocking=True)
            batch_t_resid = batch["t_residual"].to(runner.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            effect = model(batch["model_input"])
            if objective == "pseudo_outcome_mse":
                loss_vector, valid = _torch_pseudo_outcome_mse_loss_vector(
                    effect,
                    batch_y_resid,
                    batch_t_resid,
                )
                loss = loss_vector[valid].mean() if torch.any(valid) else loss_vector.mean()
            else:
                loss = torch.mean(torch.square(batch_y_resid - effect * batch_t_resid))
            loss.backward()
            runner._clip_and_step(model, optimizer, scheduler)
    runner._assert_htr_sentence_encoder_training_state(model.extractor)


class HistoricalStage1ContextPredictionHTRProvider(MultiModelForestStage1HTRProvider):
    """HTR provider used only for disjoint context-prediction feature banks."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._static_identity = context_prediction_htr_provider_identity(
            self.config,
            device=self.device,
        )
        self._nuisance_calls = 0
        self._pair_calls = 0
        self._effect_calls: Dict[str, int] = {}
        self._nuisance_model_attempts = 0
        self._pair_model_attempts = 0
        self._pair_models_fit = 0
        self._effect_model_attempts = 0

    def identity(self) -> Mapping[str, Any]:
        current = context_prediction_htr_provider_identity(
            self.config,
            device=self.device,
        )
        if current != self._static_identity:
            raise RuntimeError("context-prediction HTR configuration changed")
        return copy.deepcopy(self._static_identity)

    def fit_nuisance_inner_ensemble_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        if self._nuisance_calls:
            raise RuntimeError("context-prediction nuisance path may run only once")
        runner = self._ensure_runner(train_df)
        _assert_label_free_test_frame(runner=runner, train_df=train_df, test_df=test_df)
        folds = _bounded_fold_count(runner.avf_config.nuisance_folds, len(train_df))
        if runner._fold_n_jobs(folds) != 1:
            raise RuntimeError("deterministic context HTR nuisance fitting must be serial")
        seed = context_prediction_seed(outer_fold=outer_fold, component="nuisance")
        with _isolated_seed(seed, runner.device):
            result = super().fit_nuisance_inner_ensemble_predict(
                train_df,
                test_df,
                outer_fold,
            )
        evidence = list(result.get("inner_model_rows") or ())
        if len(evidence) != folds or any(row.get("objective") != "nuisance" for row in evidence):
            raise RuntimeError("unchanged nuisance provider returned an inexact fold schema")
        self._nuisance_calls = 1
        self._nuisance_model_attempts = folds
        return result

    def fit_pair_uplift_inner_ensemble_predict(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        texts_train: Sequence[str],
        texts_test: Sequence[str],
        y_train: np.ndarray,
        t_train: np.ndarray,
        e_train: np.ndarray,
        m_train: np.ndarray,
        e_test: np.ndarray,
        m_test: np.ndarray,
        outer_fold: int,
        propensity_caliper: float,
        outcome_caliper: float,
        max_controls_per_candidate: int,
        nearest_fallback_controls: int,
        max_attention_pairs: int,
    ) -> PairUpliftFitResult:
        if self._pair_calls:
            raise RuntimeError("context-prediction pair path may run only once")
        runner = self._ensure_runner(train_df)
        _assert_label_free_test_frame(runner=runner, train_df=train_df, test_df=test_df)
        expected_train_texts = tuple(
            _normalize_texts(train_df[runner.config.text_column].fillna(""))
        )
        expected_test_texts = tuple(_normalize_texts(test_df[runner.config.text_column].fillna("")))
        if tuple(texts_train) != expected_train_texts or tuple(texts_test) != expected_test_texts:
            raise ValueError("pair text vectors are not the exact normalized frame texts")
        y = _finite_vector(y_train, length=len(train_df), name="context outcome")
        t = _finite_vector(t_train, length=len(train_df), name="context treatment")
        e = _finite_vector(e_train, length=len(train_df), name="context propensity")
        m = _finite_vector(m_train, length=len(train_df), name="context outcome nuisance")
        e_pred = _finite_vector(e_test, length=len(test_df), name="prediction propensity")
        m_pred = _finite_vector(
            m_test,
            length=len(test_df),
            name="prediction outcome nuisance",
        )
        control_positions = np.flatnonzero(t.astype(int) == 0)
        if len(control_positions) < 1:
            raise ValueError("context-prediction pair uplift requires a context control row")
        training_pairs = build_training_pairs(
            train_df,
            texts=texts_train,
            treatment=t,
            outcome=y,
            propensity=e,
            outcome_prob=m,
            propensity_caliper=propensity_caliper,
            outcome_caliper=outcome_caliper,
        )
        seed = context_prediction_seed(
            outer_fold=outer_fold,
            component="matched_pair_uplift",
        )
        model: Optional[HTRPairUpliftNet] = None
        self._pair_model_attempts = 1
        try:
            with _isolated_seed(seed, runner.device):
                model = _train_complete_context_pair_model(
                    runner=runner,
                    pairs=training_pairs,
                    outer_fold=outer_fold,
                )
                self._pair_models_fit = int(model is not None)
                control_df = train_df.iloc[control_positions].reset_index(drop=True)
                prediction_pairs = build_candidate_pairs(
                    test_df,
                    control_df,
                    candidate_texts=texts_test,
                    control_texts=[texts_train[int(pos)] for pos in control_positions],
                    candidate_propensity=e_pred,
                    candidate_outcome_prob=m_pred,
                    control_propensity=e[control_positions],
                    control_outcome_prob=m[control_positions],
                    propensity_caliper=propensity_caliper,
                    outcome_caliper=outcome_caliper,
                    max_controls_per_candidate=max_controls_per_candidate,
                    nearest_fallback_controls=nearest_fallback_controls,
                )
                pair_delta = _predict_htr_pair_delta(
                    runner=runner,
                    model=model,
                    pairs=prediction_pairs,
                )
                test_delta, test_prob, test_n_controls = aggregate_pair_predictions(
                    prediction_pairs,
                    pair_delta,
                    len(test_df),
                )
                if not all(
                    np.all(np.isfinite(values))
                    for values in (test_delta, test_prob, test_n_controls)
                ):
                    raise ValueError("complete-context pair fit did not cover every prediction row")
                # Context-only feature-bank consumers do not use attention.
                # Avoid the extra encoder passes while retaining numerical pair semantics.
                attention_rows: list[dict[str, Any]] = []
        finally:
            if model is not None:
                runner._cleanup_model(model)

        # The Stage-1 context backend consumes only aggregate numerical test
        # columns.  Do not manufacture a train-OOF or per-pair diagnostic frame
        # for this prediction-only path.
        prediction_frame = pd.DataFrame()
        self._pair_calls = 1
        placeholder = np.full(len(train_df), _PLACEHOLDER_VALUE, dtype=float)
        return PairUpliftFitResult(
            train_delta_logit=placeholder.copy(),
            test_delta_logit=np.asarray(test_delta, dtype=float),
            train_pred_prob=placeholder.copy(),
            test_pred_prob=np.asarray(test_prob, dtype=float),
            train_n_controls=placeholder.copy(),
            test_n_controls=np.asarray(test_n_controls, dtype=float),
            feature_importance={
                "source_family": "htr_pair_uplift",
                "context_prediction_only": True,
                "attention_rows": 0,
            },
            evidence_rows=[
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": 0,
                    "source_family": "htr_pair_uplift",
                    "objective": "matched_pair_uplift_delta_logit",
                    "target_name": "treated_observed_outcome",
                    "train_rows": int(len(train_df)),
                    "heldout_rows": 0,
                    "outer_test_rows": int(len(test_df)),
                    "matched_pair_train_rows": int(len(training_pairs)),
                    "outer_test_candidate_pair_rows": int(len(prediction_pairs)),
                    "prediction_provenance": ("complete_allowed_context_fit_label_free_prediction"),
                    "context_prediction_only": True,
                    "train_values_are_finite_unconsumed_placeholders": True,
                    "deterministic_seed": int(seed),
                    "htr_model_fit_attempts": 1,
                    "htr_models_fit": int(self._pair_models_fit),
                }
            ],
            attention_rows=attention_rows,
            prediction_frame=prediction_frame,
            metrics={
                "n_train_matched_pairs": int(len(training_pairs)),
                "treated_oof": {"n_eval": 0, "auroc": None},
                "context_prediction_only": True,
            },
        )

    def fit_effect_variant_inner_ensemble_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
        *,
        effect_objective: str,
        test_nuisance_predictions: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        objective = str(effect_objective).strip()
        if objective not in {"pseudo_outcome_mse", "squared_r_loss"}:
            raise ValueError("context-prediction effect objective is not precommitted")
        if self._effect_calls.get(objective, 0):
            raise RuntimeError(f"context-prediction {objective} path may run only once")
        runner = self._ensure_runner(train_df)
        _assert_label_free_test_frame(runner=runner, train_df=train_df, test_df=test_df)
        if test_nuisance_predictions is not None:
            observed_ids = tuple(int(value) for value in test_nuisance_predictions["_oci_row_id"])
            expected_ids = tuple(int(value) for value in test_df["_oci_row_id"])
            if observed_ids != expected_ids:
                raise ValueError("test nuisance rows do not match label-free prediction rows")

        merged = train_df[["_oci_row_id"]].merge(
            nuisance_predictions.copy(),
            on="_oci_row_id",
            how="left",
            sort=False,
        )
        e = _finite_vector(merged["e_hat"], length=len(train_df), name="context e_hat")
        m = _finite_vector(merged["m_hat"], length=len(train_df), name="context m_hat")
        y = _finite_vector(
            train_df[runner.config.outcome_column],
            length=len(train_df),
            name="context outcome",
        )
        t = _finite_vector(
            train_df[runner.config.treatment_column],
            length=len(train_df),
            name="context treatment",
        )
        e_clipped = np.clip(e, runner.avf_config.e_clip, 1.0 - runner.avf_config.e_clip)
        m_clipped = clip_probability(m)
        y_resid = y - m
        t_resid = t - e_clipped
        pseudo = _r_pseudo_outcome(y_resid, t_resid)
        lower = float(getattr(runner.avf_config, "r_stage_min_propensity", 0.0))
        upper = float(getattr(runner.avf_config, "r_stage_max_propensity", 1.0))
        eligible = np.isfinite(e) & (e >= lower) & (e <= upper)
        if objective == "pseudo_outcome_mse":
            eligible &= np.isfinite(pseudo)
        positions = np.flatnonzero(eligible)
        if len(positions) < 1:
            raise ValueError("no eligible context rows remain for HTR effect fitting")

        seed = context_prediction_seed(outer_fold=outer_fold, component=objective)
        model = None
        self._effect_model_attempts += 1
        try:
            with self._temporary_effect_objective(objective):
                if _effect_objective_name(runner.avf_config) != objective:
                    raise RuntimeError("HTR effect objective changed before fitting")
                with _isolated_seed(seed, runner.device):
                    model = _EffectNet(
                        extractor=runner._create_extractor(),
                        hidden_dim=getattr(
                            runner.config.architecture,
                            "causal_head_hidden_outcome_dim",
                            64,
                        ),
                    ).to(runner.device)
                    _train_complete_context_effect_model(
                        runner=runner,
                        model=model,
                        train_df=train_df,
                        positions=positions,
                        y=y,
                        t=t,
                        e_clipped=e_clipped,
                        m_clipped=m_clipped,
                        y_resid=y_resid,
                        t_resid=t_resid,
                    )
                    test_tau = _finite_vector(
                        runner._predict_effect_model(model, test_df),
                        length=len(test_df),
                        name=f"{objective} prediction",
                    )
        finally:
            if model is not None:
                runner._cleanup_model(model)

        self._effect_calls[objective] = 1
        placeholder = np.full(len(train_df), _PLACEHOLDER_VALUE, dtype=float)
        train_predictions = pd.DataFrame(
            {
                "_oci_row_id": train_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "tau_hat_r_stage": placeholder,
                "tau_logit_modifier": placeholder,
                "r_pseudo_outcome": placeholder,
                "r_loss": placeholder,
                "effect_loss": placeholder,
                "effect_loss_at_zero_tau": placeholder,
                "effect_fold": 0,
                "r_stage_train_eligible": eligible,
                "effect_objective": objective,
                "target_source": "finite_unconsumed_context_placeholder",
            }
        )
        test_predictions = pd.DataFrame(
            {
                "_oci_row_id": test_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "tau_hat_r_stage": test_tau,
                "model_family": "htr",
                "view_name": f"htr_effect_{objective}",
                "target_source": "complete_allowed_context_fit_label_free_prediction",
                "effect_objective": objective,
            }
        )
        return {
            "train": {"predictions": train_predictions, "attention": []},
            "test_predictions": test_predictions,
            "inner_model_rows": [
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": 0,
                    "source_family": "htr",
                    "objective": f"effect_{objective}",
                    "target_name": objective,
                    "effect_objective": objective,
                    "train_rows": int(len(positions)),
                    "context_rows": int(len(train_df)),
                    "heldout_rows": 0,
                    "outer_test_rows": int(len(test_df)),
                    "prediction_provenance": ("complete_allowed_context_fit_label_free_prediction"),
                    "nuisance_prediction_provenance": "context_inner_oof",
                    "context_prediction_only": True,
                    "train_values_are_finite_unconsumed_placeholders": True,
                    "deterministic_seed": int(seed),
                    "htr_model_fit_attempts": 1,
                    "htr_models_fit": 1,
                }
            ],
        }

    def assert_complete_context_prediction_call(self, *, n_context_rows: int) -> Mapping[str, Any]:
        runner = self._runner
        if runner is None:
            raise RuntimeError("context-prediction HTR provider was never invoked")
        expected_effects = {"pseudo_outcome_mse": 1, "squared_r_loss": 1}
        if (
            self._nuisance_calls != 1
            or self._pair_calls != 1
            or self._effect_calls != expected_effects
        ):
            raise RuntimeError("context-prediction HTR component call graph is incomplete")
        profile = dict(
            context_prediction_fit_profile(
                n_context_rows=n_context_rows,
                nuisance_folds=runner.avf_config.nuisance_folds,
                effect_folds=runner.avf_config.effect_folds,
            )
        )
        observed_attempts = (
            self._nuisance_model_attempts + self._pair_model_attempts + self._effect_model_attempts
        )
        if observed_attempts != int(profile["context_prediction_model_attempts"]):
            raise RuntimeError("context-prediction HTR model-attempt count changed")
        observed_models_fit = int(
            self._nuisance_model_attempts + self._pair_models_fit + self._effect_model_attempts
        )
        if observed_models_fit not in {observed_attempts - 1, observed_attempts}:
            raise RuntimeError("context-prediction HTR fitted-model count is impossible")
        profile.update(
            {
                "observed_model_attempts": int(observed_attempts),
                "observed_models_fit": observed_models_fit,
                "degenerate_pair_zero_fallback_used": self._pair_models_fit == 0,
                "component_calls": {
                    "nuisance": self._nuisance_calls,
                    "matched_pair_uplift": self._pair_calls,
                    **copy.deepcopy(self._effect_calls),
                },
                "call_graph_complete": True,
            }
        )
        return profile

    def assert_bundle_placeholder_safety(self, bundle: Any) -> Mapping[str, Any]:
        profile = self.assert_complete_context_prediction_call(
            n_context_rows=int(np.asarray(bundle.x_train).shape[0])
        )
        names = tuple(str(value) for value in bundle.x_names)
        x_train = np.asarray(bundle.x_train, dtype=float)
        x_test = np.asarray(bundle.x_test, dtype=float)
        placeholder_indices = []
        for name in _PLACEHOLDER_FEATURE_NAMES:
            if names.count(name) != 1:
                raise RuntimeError(f"context HTR bundle does not contain exactly one {name}")
            index = names.index(name)
            placeholder_indices.append(index)
            if not np.all(x_train[:, index] == _PLACEHOLDER_VALUE):
                raise RuntimeError(f"context HTR train placeholder changed for {name}")
            if not np.all(np.isfinite(x_test[:, index])):
                raise RuntimeError(f"context HTR prediction values are non-finite for {name}")
        return {
            **dict(profile),
            "placeholder_feature_indices": placeholder_indices,
            "placeholder_train_values_verified": True,
            "corresponding_test_values_finite_before_cleaning": True,
        }

    def seal_prediction_only_bundle(self, bundle: Any) -> ContextPredictionOnlyFeatureBundle:
        """Remove every train/OOF diagnostic before the backend consumes values."""

        audit = dict(self.assert_bundle_placeholder_safety(bundle))
        x_train = np.asarray(bundle.x_train, dtype=float)
        x_test = np.asarray(bundle.x_test, dtype=float).copy()
        w_train = np.asarray(bundle.w_train, dtype=float)
        w_test = np.asarray(bundle.w_test, dtype=float).copy()
        placeholder_indices = set(audit["placeholder_feature_indices"])

        # Placeholder train values must never supply an imputation statistic.
        # Their test cells were proven finite above and are copied unchanged.
        for index in range(x_test.shape[1]):
            if index in placeholder_indices:
                continue
            finite_train = np.isfinite(x_train[:, index])
            mean = float(np.mean(x_train[finite_train, index])) if np.any(finite_train) else 0.0
            x_test[:, index] = np.where(np.isfinite(x_test[:, index]), x_test[:, index], mean)
        for index in range(w_test.shape[1]):
            finite_train = np.isfinite(w_train[:, index])
            mean = float(np.mean(w_train[finite_train, index])) if np.any(finite_train) else 0.0
            w_test[:, index] = np.where(np.isfinite(w_test[:, index]), w_test[:, index], mean)

        rows = []
        placeholder_names = set(_PLACEHOLDER_FEATURE_NAMES)
        for raw in bundle.feature_rows:
            row = copy.deepcopy(dict(raw))
            if str(row.get("feature_name")) in placeholder_names:
                row["provenance"] = "complete_allowed_context_fit_label_free_prediction"
                row["train_values_exposed_or_consumed"] = False
            rows.append(row)
        audit.update(
            {
                "prediction_only_bundle_sealed": True,
                "train_matrices_retained": False,
                "prediction_frames_retained": False,
                "attention_or_train_metrics_retained": False,
                "placeholder_columns_used_for_test_imputation": False,
            }
        )
        return ContextPredictionOnlyFeatureBundle(
            x_test=x_test,
            w_test=w_test,
            x_names=tuple(bundle.x_names),
            w_names=tuple(bundle.w_names),
            feature_rows=tuple(rows),
            audit=audit,
        )


__all__ = [
    "CONTEXT_PREDICTION_HTR_PLACEHOLDER_POLICY_ID",
    "CONTEXT_PREDICTION_HTR_PROVIDER_ID",
    "CONTEXT_PREDICTION_HTR_SEED_POLICY_ID",
    "ContextPredictionOnlyFeatureBundle",
    "HistoricalStage1ContextPredictionHTRProvider",
    "context_prediction_fit_profile",
    "context_prediction_htr_provider_identity",
    "context_prediction_htr_policy_constants",
    "context_prediction_seed",
]
