# oci/models/causal_forest_head.py
"""Causal Forest head for ITE estimation from neural network features."""

import logging
from typing import Optional, Dict, Any
import numpy as np

from .strict_causal_forest_runtime import (
    StrictCausalForestRuntimeConfig,
    assert_supported_constructor_signatures,
    audit_strict_fitted_estimator,
    audit_strict_unfitted_estimator,
)

try:
    from econml.dml import CausalForestDML
    from econml.grf import CausalForest as EconMLCausalForest
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    CausalForestDML = None
    EconMLCausalForest = None
    RandomForestRegressor = None
    RandomForestClassifier = None
    StratifiedKFold = None

logger = logging.getLogger(__name__)


def _validate_max_features(value: Any, *, name: str, allow_none: bool) -> None:
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{name} cannot be None")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} cannot be boolean")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{name} cannot be empty")
        return
    if isinstance(value, (int, np.integer)):
        if int(value) < 1:
            raise ValueError(f"integer {name} must be positive")
        return
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or not 0 < numeric <= 1:
            raise ValueError(f"fractional {name} must be finite and in (0, 1]")
        return
    raise TypeError(f"{name} must be a supported scalar")


def tune_causal_forest_model(
    model: Any,
    Y: np.ndarray,
    T: np.ndarray,
    X: Optional[np.ndarray],
    W: Optional[np.ndarray] = None,
    params: Any = "auto",
) -> bool:
    """Tune an EconML CausalForestDML model, falling back to configured parameters."""
    try:
        logger.info("Tuning CausalForestDML hyperparameters with params=%r", params)
        model.tune(Y=Y, T=T, X=X, W=W, params=params)
        logger.info("CausalForestDML hyperparameter tuning complete")
        return True
    except Exception as exc:
        logger.warning(
            "CausalForestDML hyperparameter tuning failed; fitting with configured "
            "hyperparameters. Error: %s",
            exc,
        )
        return False


class CausalForestHead:
    """
    Causal Forest head for ITE estimation from neural features.

    Unlike end-to-end neural causal heads, this uses
    econml's CausalForestDML to estimate treatment effects.

    The causal forest provides:
    - Doubly-robust estimation (robust to misspecification of either propensity or outcome)
    - Honest trees for unbiased effect estimates
    - Built-in confidence intervals
    - Direct estimation of τ(X) = E[Y(1) - Y(0) | X]

    References:
        Athey, Tibshirani, Wager (2019). Generalized Random Forests. Annals of Statistics.
        Chernozhukov et al. (2018). Double/Debiased Machine Learning. Econometrica.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        honest: bool = True,
        inference: bool = True,
        random_state: int = 42,
        tune_model: bool = True,
        subforest_size: int = 4,
        nuisance_n_estimators: Optional[int] = None,
        nuisance_max_depth: Optional[int] = None,
        nuisance_min_samples_leaf: Optional[int] = None,
        nuisance_treatment_max_features: Any = "sqrt",
        nuisance_outcome_max_features: Any = 1.0,
        n_jobs: int = -1,
        runtime_config: Optional[StrictCausalForestRuntimeConfig] = None,
    ):
        """
        Initialize Causal Forest head.

        Args:
            n_estimators: Number of trees in the forest (must be divisible by 4)
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_leaf: Minimum samples per leaf
            max_features: Feature subset strategy for splitting
            honest: Use honest estimation (sample splitting within trees)
            inference: Enable inference for confidence intervals
            random_state: Random seed for reproducibility
            tune_model: Run EconML's automatic tuning step before fitting. This
                remains enabled by default for backward compatibility; callers
                with a fixed, pre-evaluated configuration can disable it.
            subforest_size: Number of trees in each inference subforest.
            nuisance_n_estimators: Number of trees in each nuisance forest.
                ``None`` preserves the legacy derived value.
            nuisance_max_depth: Maximum depth of both nuisance forests.
            nuisance_min_samples_leaf: Minimum nuisance-forest leaf size.
                ``None`` preserves the legacy coupling to ``min_samples_leaf``.
            nuisance_treatment_max_features: Feature sampling policy for the
                treatment nuisance classifier.
            nuisance_outcome_max_features: Feature sampling policy for the
                outcome nuisance regressor.
            n_jobs: Operational CPU parallelism for all three forests.  This
                setting does not change the portable scientific identity.

        Note: Nuisance functions (propensity, outcome) are estimated using sklearn
        random forests on the neural network's learned features.
        """
        if not ECONML_AVAILABLE:
            raise ImportError(
                "econml is required for CausalForestHead. " "Install with: pip install econml"
            )

        if runtime_config is not None:
            if not isinstance(runtime_config, StrictCausalForestRuntimeConfig):
                raise TypeError("runtime_config must be StrictCausalForestRuntimeConfig")
            scientific = runtime_config.causal_forest
            self.runtime_config = runtime_config
            self.runtime_mode = "portable_strict_runtime_config_v1"
            self.n_estimators = int(scientific.n_estimators)
            self.max_depth = scientific.max_depth
            self.min_samples_leaf = scientific.min_samples_leaf
            self.max_features = scientific.max_features
            self.honest = bool(scientific.honest)
            self.inference = bool(scientific.inference)
            self.random_state = int(scientific.random_seed)
            self.tune_model = bool(scientific.tune_model)
            self.subforest_size = int(scientific.subforest_size)
            self.nuisance_n_estimators = int(scientific.treatment_model.n_estimators)
            self.nuisance_max_depth = scientific.treatment_model.max_depth
            self.nuisance_min_samples_leaf = scientific.treatment_model.min_samples_leaf
            self.nuisance_treatment_max_features = scientific.treatment_model.max_features
            self.nuisance_outcome_max_features = scientific.outcome_model.max_features
            self.n_jobs = 1
            self.requested_host_cpu_budget = int(
                runtime_config.operational.requested_host_cpu_budget
            )
            self.model = None
            self._fitted = False
            self.tuning_attempted_ = False
            self.tuning_succeeded_ = None
            self.effective_forest_parameters_ = None
            self.effective_nuisance_parameters_ = None
            self.strict_unfitted_estimator_audit_ = None
            self.strict_fitted_estimator_audit_ = None
            self.crossfit_split_audit_ = None
            return

        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int):
            raise TypeError("n_estimators must be an integer")
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if isinstance(subforest_size, bool) or not isinstance(subforest_size, int):
            raise TypeError("subforest_size must be an integer")
        if subforest_size < 1:
            raise ValueError("subforest_size must be positive")
        if inference and n_estimators % subforest_size:
            raise ValueError(
                "n_estimators must be divisible by subforest_size when inference is enabled"
            )
        if max_depth is not None and (
            isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1
        ):
            raise ValueError("max_depth must be None or a positive integer")
        if (
            isinstance(min_samples_leaf, bool)
            or not isinstance(min_samples_leaf, int)
            or min_samples_leaf < 1
        ):
            raise ValueError("min_samples_leaf must be a positive integer")
        for name, value in (
            ("honest", honest),
            ("inference", inference),
            ("tune_model", tune_model),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
        if isinstance(random_state, bool) or not isinstance(random_state, int):
            raise TypeError("random_state must be an integer")
        if nuisance_n_estimators is None:
            nuisance_n_estimators = max(50, n_estimators // 2)
        if (
            isinstance(nuisance_n_estimators, bool)
            or not isinstance(nuisance_n_estimators, int)
            or nuisance_n_estimators < 1
        ):
            raise ValueError("nuisance_n_estimators must be a positive integer")
        if nuisance_max_depth is not None and (
            isinstance(nuisance_max_depth, bool)
            or not isinstance(nuisance_max_depth, int)
            or nuisance_max_depth < 1
        ):
            raise ValueError("nuisance_max_depth must be None or a positive integer")
        if nuisance_min_samples_leaf is None:
            nuisance_min_samples_leaf = min_samples_leaf
        if (
            isinstance(nuisance_min_samples_leaf, bool)
            or not isinstance(nuisance_min_samples_leaf, int)
            or nuisance_min_samples_leaf < 1
        ):
            raise ValueError("nuisance_min_samples_leaf must be a positive integer")
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
            raise ValueError("n_jobs must be a nonzero integer")
        _validate_max_features(
            max_features,
            name="max_features",
            allow_none=False,
        )
        _validate_max_features(
            nuisance_treatment_max_features,
            name="nuisance_treatment_max_features",
            allow_none=True,
        )
        _validate_max_features(
            nuisance_outcome_max_features,
            name="nuisance_outcome_max_features",
            allow_none=True,
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.honest = honest
        self.inference = inference
        self.random_state = random_state
        self.tune_model = bool(tune_model)
        self.subforest_size = subforest_size
        self.nuisance_n_estimators = nuisance_n_estimators
        self.nuisance_max_depth = nuisance_max_depth
        self.nuisance_min_samples_leaf = nuisance_min_samples_leaf
        self.nuisance_treatment_max_features = nuisance_treatment_max_features
        self.nuisance_outcome_max_features = nuisance_outcome_max_features
        self.n_jobs = n_jobs
        self.requested_host_cpu_budget = None
        self.runtime_config = None
        self.runtime_mode = "legacy_compatibility_shim_v1"

        # The CausalForestDML model (created during fit)
        self.model = None
        self._fitted = False
        self.tuning_attempted_ = False
        self.tuning_succeeded_ = None
        self.effective_forest_parameters_ = None
        self.effective_nuisance_parameters_ = None
        self.strict_unfitted_estimator_audit_ = None
        self.strict_fitted_estimator_audit_ = None
        self.crossfit_split_audit_ = None

    def _configured_forest_parameters(self) -> Dict[str, Any]:
        return {
            "n_estimators": int(self.n_estimators),
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "honest": bool(self.honest),
            "inference": bool(self.inference),
            "subforest_size": int(self.subforest_size),
            "random_state": int(self.random_state),
        }

    def _configured_nuisance_parameters(self) -> Dict[str, Any]:
        return {
            "n_estimators": int(self.nuisance_n_estimators),
            "max_depth": self.nuisance_max_depth,
            "min_samples_leaf": int(self.nuisance_min_samples_leaf),
            "treatment_max_features": self.nuisance_treatment_max_features,
            "outcome_max_features": self.nuisance_outcome_max_features,
            "random_state": int(self.random_state),
        }

    def _operational_parameters(self) -> Dict[str, Any]:
        return {"n_jobs": int(self.n_jobs)}

    @staticmethod
    def _estimator_parameter(estimator: Any, name: str) -> Any:
        get_params = getattr(estimator, "get_params", None)
        if callable(get_params):
            parameters = get_params(deep=False)
            if isinstance(parameters, dict) and name in parameters:
                value = parameters[name]
                return value.item() if isinstance(value, np.generic) else value
        parameters = getattr(estimator, "kwargs", None)
        if isinstance(parameters, dict) and name in parameters:
            value = parameters[name]
            return value.item() if isinstance(value, np.generic) else value
        if hasattr(estimator, name):
            value = getattr(estimator, name)
            return value.item() if isinstance(value, np.generic) else value
        raise TypeError(f"nuisance estimator does not expose configured parameter {name}")

    def _effective_nuisance_parameters(
        self,
        *,
        model_t: Any,
        model_y: Any,
    ) -> Dict[str, Any]:
        result = {
            "n_estimators": self._estimator_parameter(model_t, "n_estimators"),
            "max_depth": self._estimator_parameter(model_t, "max_depth"),
            "min_samples_leaf": self._estimator_parameter(model_t, "min_samples_leaf"),
            "treatment_max_features": self._estimator_parameter(model_t, "max_features"),
            "outcome_max_features": self._estimator_parameter(model_y, "max_features"),
            "random_state": self._estimator_parameter(model_t, "random_state"),
        }
        for shared_name in (
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "random_state",
        ):
            outcome_value = self._estimator_parameter(model_y, shared_name)
            if outcome_value != result[shared_name]:
                raise RuntimeError(
                    "treatment and outcome nuisance forests disagree on " f"{shared_name}"
                )
        if result != self._configured_nuisance_parameters():
            raise RuntimeError(
                "effective nuisance-forest parameters differ from the configured "
                "scientific settings"
            )
        for estimator, label in ((model_t, "treatment"), (model_y, "outcome")):
            if self._estimator_parameter(estimator, "n_jobs") != int(self.n_jobs):
                raise RuntimeError(
                    f"effective {label} nuisance n_jobs differs from the " "operational setting"
                )
        return result

    def _effective_forest_parameters(self) -> Dict[str, Any]:
        """Return the post-tuning forest parameters that affect final fitting."""

        if self.model is None:
            raise RuntimeError("causal forest model has not been created")
        effective: Dict[str, Any] = {}
        for name in self._configured_forest_parameters():
            if not hasattr(self.model, name):
                raise RuntimeError(
                    "causal-forest runtime does not expose configured "
                    f"attribute {name}; refusing to substitute the requested value"
                )
            value = getattr(self.model, name)
            if isinstance(value, np.generic):
                value = value.item()
            if value is not None and not isinstance(value, (bool, int, float, str)):
                raise TypeError(f"effective causal-forest parameter {name} is not scalar")
            effective[name] = value
        return effective

    def _create_model(self):
        """Create an unfitted CausalForestDML with configured nuisance models."""
        if self.runtime_config is not None:
            return self._create_strict_model()

        model_t = RandomForestClassifier(
            n_estimators=self.nuisance_n_estimators,
            max_depth=self.nuisance_max_depth,
            min_samples_leaf=self.nuisance_min_samples_leaf,
            max_features=self.nuisance_treatment_max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        logger.info("Using random forest for propensity estimation (on neural features)")

        model_y = RandomForestRegressor(
            n_estimators=self.nuisance_n_estimators,
            max_depth=self.nuisance_max_depth,
            min_samples_leaf=self.nuisance_min_samples_leaf,
            max_features=self.nuisance_outcome_max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        logger.info("Using random forest for outcome estimation (on neural features)")

        model = CausalForestDML(
            model_t=model_t,
            model_y=model_y,
            discrete_treatment=True,  # Binary treatment indicator
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            honest=self.honest,
            inference=self.inference,
            subforest_size=self.subforest_size,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.effective_nuisance_parameters_ = self._effective_nuisance_parameters(
            model_t=model_t,
            model_y=model_y,
        )
        if self._estimator_parameter(model, "n_jobs") != int(self.n_jobs):
            raise RuntimeError(
                "effective causal-forest n_jobs differs from the operational setting"
            )
        return model

    def _create_strict_model(self):
        """Create and authenticate the closed portable estimator graph."""

        if self.runtime_config is None:
            raise RuntimeError("strict model creation requires runtime_config")
        assert_supported_constructor_signatures(
            causal_forest_class=CausalForestDML,
            treatment_forest_class=RandomForestClassifier,
            outcome_forest_class=RandomForestRegressor,
            stratified_crossfit_class=StratifiedKFold,
        )
        model_t = RandomForestClassifier(**self.runtime_config.treatment_constructor_kwargs())
        model_y = RandomForestRegressor(**self.runtime_config.outcome_constructor_kwargs())
        crossfit = StratifiedKFold(**self.runtime_config.crossfit_constructor_kwargs())
        model = CausalForestDML(
            **self.runtime_config.causal_forest_constructor_kwargs(
                model_t=model_t,
                model_y=model_y,
                cv=crossfit,
            )
        )
        self.strict_unfitted_estimator_audit_ = audit_strict_unfitted_estimator(
            model=model,
            config=self.runtime_config,
            causal_forest_class=CausalForestDML,
            treatment_forest_class=RandomForestClassifier,
            outcome_forest_class=RandomForestRegressor,
            stratified_crossfit_class=StratifiedKFold,
        )
        return model

    def fit(
        self,
        X: Optional[np.ndarray],
        T: np.ndarray,
        Y: np.ndarray,
        W: Optional[np.ndarray] = None,
        propensity: Optional[np.ndarray] = None,
        outcome_pred: Optional[np.ndarray] = None,
    ) -> "CausalForestHead":
        """
        Fit causal forest on extracted features.

        Args:
            X: Effect-modifier feature matrix, shape (n_samples, n_features), or None
            W: Optional control/confounder feature matrix passed to EconML W
            T: Binary treatment indicator, shape (n_samples,)
            Y: Binary outcome indicator, shape (n_samples,)
            propensity: Optional propensity scores from neural network P(T=1|X)
            outcome_pred: Optional outcome predictions from neural network E[Y|X]

        Returns:
            self
        """
        n_samples = X.shape[0] if X is not None else len(Y)
        x_dim = X.shape[1] if X is not None else 0
        w_msg = f", W={W.shape[1]} controls" if W is not None else ""
        logger.info(
            f"Fitting CausalForestDML on {n_samples} samples with X={x_dim} features{w_msg}"
        )

        # Ensure arrays are the right shape
        T = np.asarray(T).flatten()
        Y = np.asarray(Y).flatten()
        if self.runtime_config is not None:
            strict_x = np.asarray(X, dtype=float)
            strict_w = None if W is None else np.asarray(W, dtype=float)
            self.runtime_config.validate_fit_inputs(
                effect=strict_x,
                controls=strict_w,
                treatment=T,
                outcome=Y,
            )
            self.crossfit_split_audit_ = self.runtime_config.split_audit(T)

        self.model = self._create_model()
        self.tuning_attempted_ = bool(self.tune_model)
        self.tuning_succeeded_ = None
        if self.tune_model:
            self.tuning_succeeded_ = tune_causal_forest_model(
                self.model,
                Y=Y,
                T=T,
                X=X,
                W=W,
            )
            if not self.tuning_succeeded_:
                logger.info("Rebuilding CausalForestDML after failed tuning attempt")
                self.model = self._create_model()
        else:
            logger.info("Skipping CausalForestDML tuning; using fixed configuration")

        # Fit the model.  The portable path spells out every accepted
        # fit-time channel so no labels, weights, groups, or cached values can
        # arrive through an implicit call-site convention.
        if self.runtime_config is not None:
            self.model.fit(
                Y=Y,
                T=T,
                X=np.asarray(X, dtype=float),
                W=None if W is None else np.asarray(W, dtype=float),
                sample_weight=None,
                groups=None,
                cache_values=False,
                inference="auto",
            )
            self.strict_fitted_estimator_audit_ = audit_strict_fitted_estimator(
                model=self.model,
                config=self.runtime_config,
                causal_forest_class=CausalForestDML,
                treatment_forest_class=RandomForestClassifier,
                outcome_forest_class=RandomForestRegressor,
                stratified_crossfit_class=StratifiedKFold,
                grf_class=EconMLCausalForest,
            )
            self.effective_nuisance_parameters_ = {
                "treatment_model": (self.runtime_config.treatment_constructor_kwargs()),
                "outcome_model": (self.runtime_config.outcome_constructor_kwargs()),
            }
        else:
            # CausalForestDML expects T as 1D and Y as 1D.
            self.model.fit(Y=Y, T=T, X=X, W=W)
        self.effective_forest_parameters_ = self._effective_forest_parameters()
        if (
            not self.tune_model
            and self.effective_forest_parameters_ != self._configured_forest_parameters()
        ):
            raise RuntimeError(
                "effective causal-forest parameters differ from the fixed "
                "scientific configuration"
            )
        self._fitted = True

        logger.info("CausalForestDML fitting complete")
        return self

    def fit_audit(self) -> Dict[str, Any]:
        """Return actual tuning status and effective post-tuning parameters."""

        if self.runtime_config is not None:
            if (
                not self._fitted
                or self.strict_unfitted_estimator_audit_ is None
                or self.strict_fitted_estimator_audit_ is None
                or self.crossfit_split_audit_ is None
            ):
                raise RuntimeError(
                    "strict CausalForestHead must be fit before requesting " "its audit"
                )
            return {
                "configuration_mode": self.runtime_mode,
                "runtime_schema_version": (self.runtime_config.schema_version),
                "scientific_identity": (self.runtime_config.scientific_identity()),
                "scientific_identity_sha256": (self.runtime_config.scientific_identity_sha256()),
                "operational_attestation": (self.runtime_config.operational_attestation()),
                "tuning_configured": False,
                "tuning_attempted": False,
                "tuning_succeeded": None,
                "tuning_failure_fell_back_to_configured_parameters": False,
                "tuning_params": None,
                "crossfit_split_audit": self.crossfit_split_audit_,
                "unfitted_estimator_audit": (self.strict_unfitted_estimator_audit_),
                "fitted_estimator_audit": (self.strict_fitted_estimator_audit_),
                "fit_call_contract": {
                    "sample_weight": None,
                    "groups": None,
                    "cache_values": False,
                    "inference": "auto",
                    "fit_call_count": 1,
                },
                "prediction_contrast": {"T0": 0, "T1": 1},
                "effective_parameters": dict(self.effective_forest_parameters_),
                "effective_nuisance_parameters": dict(self.effective_nuisance_parameters_),
            }
        if (
            not self._fitted
            or self.effective_forest_parameters_ is None
            or self.effective_nuisance_parameters_ is None
        ):
            raise RuntimeError("CausalForestHead must be fit before requesting its audit")
        return {
            "configuration_mode": self.runtime_mode,
            "configured_parameters": self._configured_forest_parameters(),
            "configured_nuisance_parameters": (self._configured_nuisance_parameters()),
            "operational_parameters": self._operational_parameters(),
            "tuning_configured": bool(self.tune_model),
            "tuning_attempted": bool(self.tuning_attempted_),
            "tuning_succeeded": self.tuning_succeeded_,
            "tuning_failure_fell_back_to_configured_parameters": bool(
                self.tuning_attempted_ and self.tuning_succeeded_ is False
            ),
            "tuning_params": "auto" if self.tune_model else None,
            "effective_parameters": dict(self.effective_forest_parameters_),
            "effective_nuisance_parameters": dict(self.effective_nuisance_parameters_),
        }

    def predict(
        self, X: Optional[np.ndarray], return_ci: bool = True, alpha: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Predict ITE with optional confidence intervals.

        Args:
            X: Effect-modifier feature matrix, shape (n_samples, n_features), or None
            return_ci: Whether to return confidence intervals
            alpha: Significance level for confidence intervals (default 0.05 = 95% CI)

        Returns:
            Dictionary with predictions:
                - tau_pred: Point estimates of τ(X), shape (n_samples,)
                - tau_lower: Lower CI bound (if return_ci and inference enabled)
                - tau_upper: Upper CI bound (if return_ci and inference enabled)
        """
        if not self._fitted:
            raise RuntimeError("CausalForestHead must be fitted before predicting")

        # Point estimates
        tau_pred = self.model.effect(X, T0=0, T1=1).flatten()

        result = {"tau_pred": tau_pred}

        # Confidence intervals (if available)
        if return_ci and self.inference:
            try:
                # Get inference object
                inference_result = self.model.effect_inference(X)
                ci = inference_result.conf_int(alpha=alpha)
                result["tau_lower"] = ci[0].flatten()
                result["tau_upper"] = ci[1].flatten()
                result["tau_std"] = (
                    inference_result.std_point.flatten()
                    if hasattr(inference_result, "std_point")
                    else None
                )
            except Exception as e:
                logger.warning(f"Could not compute confidence intervals: {e}")

        return result

    def effect_summary(self, X: Optional[np.ndarray], alpha: float = 0.05) -> Dict[str, Any]:
        """
        Get summary statistics of treatment effects.

        Args:
            X: Feature matrix
            alpha: Significance level for CIs

        Returns:
            Dictionary with summary statistics
        """
        preds = self.predict(X, return_ci=True, alpha=alpha)
        tau = preds["tau_pred"]

        summary = {
            "ate": np.mean(tau),
            "ate_std": np.std(tau),
            "tau_min": np.min(tau),
            "tau_max": np.max(tau),
            "tau_median": np.median(tau),
            "n_samples": len(tau),
            "n_positive_effect": np.sum(tau > 0),
            "n_negative_effect": np.sum(tau < 0),
        }

        if "tau_lower" in preds and preds["tau_lower"] is not None:
            # Proportion of significant effects (CI doesn't include 0)
            significant = (preds["tau_lower"] > 0) | (preds["tau_upper"] < 0)
            summary["n_significant"] = np.sum(significant)
            summary["pct_significant"] = np.mean(significant) * 100

        return summary

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for checkpointing."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "honest": self.honest,
            "inference": self.inference,
            "subforest_size": self.subforest_size,
            "random_state": self.random_state,
            "tune_model": self.tune_model,
            "nuisance_n_estimators": self.nuisance_n_estimators,
            "nuisance_max_depth": self.nuisance_max_depth,
            "nuisance_min_samples_leaf": self.nuisance_min_samples_leaf,
            "nuisance_treatment_max_features": (self.nuisance_treatment_max_features),
            "nuisance_outcome_max_features": self.nuisance_outcome_max_features,
            "n_jobs": self.n_jobs,
            "requested_host_cpu_budget": self.requested_host_cpu_budget,
            "runtime_mode": self.runtime_mode,
            "runtime_config": (
                None if self.runtime_config is None else self.runtime_config.as_dict()
            ),
            "fitted": self._fitted,
        }


class _FixedPropensityModel:
    """
    Dummy model that returns pre-computed propensity scores.

    Used when we want to use neural network's propensity predictions
    as the nuisance function in CausalForestDML.
    """

    def __init__(self, propensity_scores: np.ndarray):
        self.propensity_scores = propensity_scores.flatten()
        self._idx = 0

    def fit(self, X, y, **kwargs):
        return self

    def predict_proba(self, X):
        """Return propensity as probability."""
        n = X.shape[0]
        # Return as 2-column array [P(T=0), P(T=1)]
        p = self.propensity_scores[self._idx : self._idx + n]
        self._idx += n
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Return binary predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class _FixedOutcomeModel:
    """
    Dummy model that returns pre-computed outcome predictions.

    Used when we want to use neural network's outcome predictions
    as the nuisance function in CausalForestDML.
    """

    def __init__(self, outcome_preds: np.ndarray):
        self.outcome_preds = outcome_preds.flatten()
        self._idx = 0

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X):
        """Return outcome predictions."""
        n = X.shape[0]
        preds = self.outcome_preds[self._idx : self._idx + n]
        self._idx += n
        return preds
