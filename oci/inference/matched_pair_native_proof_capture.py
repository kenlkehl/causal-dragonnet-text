"""Non-executable capture and replay for native matched-pair uplift fits.

The matched-pair Stage 1 family has two genuine subproducers: a TF-IDF
offset-logit model for every configured BoW view and an HTR pair network.  Both
consume the same honest nuisance vectors and deterministic matching routine.
This module captures both sides in one closed JSON/NPZ artifact and refuses to
validate incomplete subproducer coverage.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.model_selection import KFold

from ..models.hierarchical_transformer_extractor import HierarchicalTransformerExtractor
from .bow_native_proof_capture import (
    _capture_learner,
    _capture_vectorizer,
    _predict_learner,
    _restore_vectorizer,
)
from .htr_native_proof_capture import (
    _extractor_descriptor,
    _restore_tensor,
    _tensor_storage,
    directory_tree_sha256,
)
from .multi_model_pair_uplift import (
    HTRPairUpliftNet,
    OffsetLogitBoWPairModel,
    RidgeDeltaBoWPairModel,
    aggregate_pair_predictions,
    build_candidate_pairs,
    build_training_pairs,
)

MATCHED_PAIR_NATIVE_CAPTURE_SCHEMA = "production_matched_pair_native_capture_v1"
MATCHED_PAIR_NATIVE_CAPTURE_ARRAY_SCHEMA = "production_matched_pair_native_capture_array_v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_KEY = re.compile(r"^[a-z0-9_]+$")
_FORBIDDEN_SUFFIXES = (
    ".joblib",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".safetensors",
)
_PAIR_COLUMNS = (
    "candidate_pos",
    "treated_pos",
    "control_pos",
    "candidate_row_id",
    "treated_row_id",
    "control_row_id",
    "treated_text",
    "control_text",
    "label",
    "base_prob",
    "base_logit",
    "propensity_abs_diff",
    "outcome_abs_diff",
    "score_abs_diff_sum",
    "used_nearest_fallback",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "schema_version": MATCHED_PAIR_NATIVE_CAPTURE_ARRAY_SCHEMA,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        ).encode("utf-8")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(row_ids: Sequence[int], texts: Sequence[str]) -> str:
    rows = tuple(map(int, row_ids))
    values = tuple(map(str, texts))
    if len(rows) != len(values):
        raise ValueError("matched-pair text binding has inconsistent lengths")
    digest = hashlib.sha256()
    digest.update(b"production-matched-pair-text-binding-v1\0")
    for row_id, text in zip(rows, values):
        encoded = text.encode("utf-8")
        digest.update(int(row_id).to_bytes(8, "little", signed=False))
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _row_fingerprint(row_ids: Sequence[int]) -> str:
    rows = tuple(map(int, row_ids))
    if not rows or len(rows) != len(set(rows)) or any(row < 0 for row in rows):
        raise ValueError("matched-pair row IDs must be unique non-negative integers")
    return _sha256_json({"ordered_row_ids": list(rows)})


def _pair_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    return str(value)


def _pair_fingerprint(frame: pd.DataFrame) -> str:
    rows = []
    for _, raw in frame.iterrows():
        rows.append(
            {
                column: _pair_value(raw[column]) if column in frame.columns else None
                for column in _PAIR_COLUMNS
            }
        )
    return _sha256_json({"columns": list(_PAIR_COLUMNS), "rows": rows})


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to replace immutable matched-pair artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to replace immutable matched-pair artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _ArrayStore:
    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.inventory: dict[str, dict[str, Any]] = {}

    def add(self, key: str, value: Any) -> str:
        key = str(key)
        if _SAFE_KEY.fullmatch(key) is None or key in self.arrays:
            raise ValueError(f"invalid or duplicate matched-pair array key: {key}")
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise ValueError("matched-pair arrays cannot use object dtype")
        self.arrays[key] = array
        self.inventory[key] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "content_sha256": _array_sha256(array),
        }
        return key


def _finite_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (int(length),) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite vector of length {length}")
    return array


def _capture_offset_model(
    model: OffsetLogitBoWPairModel,
    store: _ArrayStore,
    prefix: str,
) -> Mapping[str, Any]:
    if type(model) is not OffsetLogitBoWPairModel:
        raise TypeError("matched-pair proof requires OffsetLogitBoWPairModel")
    if model.constant_delta_ is not None:
        raise RuntimeError(
            "matched-pair proof requires a genuinely fitted offset-logit model, "
            "not its constant fallback"
        )
    body: dict[str, Any] = {
        "kind": "offset_logit_bow_pair",
        "class_name": "OffsetLogitBoWPairModel",
        "vectorizer_params": json.loads(_canonical_json(model.vectorizer_params)),
        "l2_alpha": float(model.l2_alpha),
        "max_iter": int(model.max_iter),
        "optimizer": {
            "method": str(model.optimizer_method),
            "ftol": float(model.optimizer_ftol),
            "gtol": float(model.optimizer_gtol),
            "maxls": int(model.optimizer_maxls),
            "maxcor": int(model.optimizer_maxcor),
            "maxfun": int(model.optimizer_maxfun),
            "tol": (
                None
                if model.optimizer_tol is None
                else float(model.optimizer_tol)
            ),
            "initialization": str(model.optimizer_initialization),
            "require_success": bool(model.require_optimizer_success),
            "jacobian": "analytic",
        },
        "constant_delta": (None if model.constant_delta_ is None else float(model.constant_delta_)),
    }
    if model.constant_delta_ is None:
        if model.vectorizer is None or model.coef_ is None:
            raise ValueError("fitted offset-logit pair model lacks native state")
        body["vectorizer"] = _capture_vectorizer(
            model.vectorizer,
            store,
            f"{prefix}_vectorizer",
            vectorizer_params=model.vectorizer_params,
        )
        body["coefficient"] = store.add(
            f"{prefix}_coefficient", np.asarray(model.coef_, dtype=np.float64)
        )
        body["intercept"] = float(model.intercept_)
    else:
        body["vectorizer"] = None
        body["coefficient"] = None
        body["intercept"] = 0.0
    return {**body, "state_sha256": _sha256_json(body)}


def _predict_offset_model(
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    pairs: pd.DataFrame,
) -> np.ndarray:
    body = {key: value for key, value in descriptor.items() if key != "state_sha256"}
    if descriptor.get("state_sha256") != _sha256_json(body):
        raise ValueError("captured offset-logit pair state changed")
    if pairs.empty:
        return np.zeros(0, dtype=float)
    constant = descriptor.get("constant_delta")
    if constant is not None:
        return np.full(len(pairs), float(constant), dtype=float)
    vectorizer = _restore_vectorizer(descriptor["vectorizer"], arrays)
    control = vectorizer.transform(pairs["control_text"].astype(str).tolist())
    treated = vectorizer.transform(pairs["treated_text"].astype(str).tolist())
    matrix = sparse.hstack([control, treated], format="csr")
    coefficient = np.asarray(arrays[str(descriptor["coefficient"])], dtype=float)
    if coefficient.shape != (matrix.shape[1],):
        raise ValueError("captured offset-logit coefficient shape changed")
    return np.asarray(float(descriptor["intercept"]) + matrix.dot(coefficient), dtype=float)


def _capture_htr_pair_model(
    model: HTRPairUpliftNet | None,
    store: _ArrayStore,
    prefix: str,
    *,
    training_configuration: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if model is None:
        body = {"kind": "constant_zero_htr_pair", "class_name": None}
        return {**body, "state_sha256": _sha256_json(body)}
    if type(model) is not HTRPairUpliftNet:
        raise TypeError("matched-pair proof requires HTRPairUpliftNet")
    extractor = _extractor_descriptor(model.extractor)
    state_rows = []
    for index, (name, tensor) in enumerate(model.state_dict().items()):
        storage, dtype_name = _tensor_storage(tensor)
        state_rows.append(
            {
                "state_key": str(name),
                "array": store.add(f"{prefix}_state_{index:05d}", storage),
                "torch_dtype": dtype_name,
                "shape": list(tensor.shape),
            }
        )
    head_configuration = model.head_configuration()
    if set(head_configuration) != {
        "hidden_dim",
        "depth",
        "activation",
        "dropout",
        "layer_norm",
        "bias",
    }:
        raise RuntimeError("matched-pair HTR head constructor is incomplete")
    body = {
        "kind": "htr_pair_network",
        "class_name": "HTRPairUpliftNet",
        "head_configuration": head_configuration,
        "training_configuration": (
            None
            if training_configuration is None
            else json.loads(_canonical_json(dict(training_configuration)))
        ),
        "extractor": extractor,
        "state_tensors": state_rows,
    }
    return {**body, "state_sha256": _sha256_json(body)}


def _build_htr_pair_model(
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    initialization_texts: Sequence[str],
    htr_model_path: Path | str | None,
    device: torch.device,
) -> HTRPairUpliftNet | None:
    body = {key: value for key, value in descriptor.items() if key != "state_sha256"}
    if descriptor.get("state_sha256") != _sha256_json(body):
        raise ValueError("captured HTR pair state changed")
    if descriptor.get("kind") == "constant_zero_htr_pair":
        return None
    if descriptor.get("kind") != "htr_pair_network":
        raise ValueError("captured HTR pair model kind is unsupported")
    extractor_row = descriptor.get("extractor")
    constructor = dict((extractor_row or {}).get("constructor") or {})
    marker = constructor.get("sentence_encoder_model")
    if marker == "authenticated_local_tree":
        if htr_model_path is None:
            raise ValueError("HTR pair replay requires an authenticated local model tree")
        constructor["sentence_encoder_model"] = str(Path(htr_model_path).resolve())
    elif marker != "hash":
        raise ValueError("captured HTR pair sentence-model marker is unsupported")
    constructor["device"] = device
    extractor = HierarchicalTransformerExtractor(**constructor)
    extractor.fit_tokenizer(list(map(str, initialization_texts)))
    if _extractor_descriptor(extractor) != extractor_row:
        raise RuntimeError("reconstructed HTR pair extractor identity changed")
    head_configuration = descriptor.get("head_configuration")
    if not isinstance(head_configuration, Mapping) or set(head_configuration) != {
        "hidden_dim",
        "depth",
        "activation",
        "dropout",
        "layer_norm",
        "bias",
    }:
        raise ValueError(
            "captured typed HTR pair model lacks its complete head constructor"
        )
    model = HTRPairUpliftNet(
        extractor=extractor,
        hidden_dim=int(head_configuration["hidden_dim"]),
        dropout=float(head_configuration["dropout"]),
        head_depth=int(head_configuration["depth"]),
        head_activation=str(head_configuration["activation"]),
        head_layer_norm=head_configuration["layer_norm"],
        head_bias=head_configuration["bias"],
    ).to(device)
    state: dict[str, torch.Tensor] = {}
    for row in descriptor.get("state_tensors") or ():
        name = str(row.get("state_key") or "")
        array_key = str(row.get("array") or "")
        if not name or name in state or array_key not in arrays:
            raise ValueError("captured HTR pair tensor descriptor is invalid")
        tensor = _restore_tensor(arrays[array_key], str(row.get("torch_dtype") or ""))
        if list(tensor.shape) != row.get("shape"):
            raise ValueError("captured HTR pair tensor shape changed")
        state[name] = tensor
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("captured HTR pair state does not fit its native model") from exc
    model.eval()
    return model


def _predict_htr_pair(
    model: HTRPairUpliftNet | None,
    pairs: pd.DataFrame,
    *,
    batch_size: int,
) -> np.ndarray:
    if pairs.empty:
        return np.zeros(0, dtype=np.float64)
    if model is None:
        return np.zeros(len(pairs), dtype=np.float64)
    control = pairs["control_text"].astype(str).tolist()
    treated = pairs["treated_text"].astype(str).tolist()
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), max(1, int(batch_size))):
            end = start + max(1, int(batch_size))
            outputs.append(
                model(control[start:end], treated[start:end])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
    return (
        np.concatenate(outputs)
        if outputs
        else np.zeros(0, dtype=np.float64)
    )


class NativeMatchedPairProofCaptureSink:
    """Capture both native matched-pair subproducers for one exact scope."""

    def __init__(
        self,
        *,
        artifact_dir: Path | str,
        scope_id: str,
        outer_fold: int,
        inner_fold: int,
        fit_row_ids: Sequence[int],
        heldout_row_ids: Sequence[int],
        fit_texts: Sequence[str],
        heldout_texts: Sequence[str],
        text_column: str,
        effect_folds: int,
        view_configs: Sequence[Mapping[str, Any]],
        propensity_caliper: float,
        outcome_caliper: float,
        max_controls_per_candidate: int,
        nearest_fallback_controls: int,
        htr_model_tree_sha256: str | None,
        htr_prediction_batch_size: int,
        seed: int,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        if self.artifact_dir.exists():
            raise RuntimeError(
                f"matched-pair capture directory already exists: {self.artifact_dir}"
            )
        self.scope_id = str(scope_id)
        self.outer_fold = int(outer_fold)
        self.inner_fold = int(inner_fold)
        if self.outer_fold < 1 or self.inner_fold < 1:
            raise ValueError("matched-pair proof requires an exact-inner scope")
        self.fit_row_ids = tuple(map(int, fit_row_ids))
        self.heldout_row_ids = tuple(map(int, heldout_row_ids))
        self.fit_texts = tuple(map(str, fit_texts))
        self.heldout_texts = tuple(map(str, heldout_texts))
        _row_fingerprint(self.fit_row_ids)
        _row_fingerprint(self.heldout_row_ids)
        if set(self.fit_row_ids) & set(self.heldout_row_ids):
            raise ValueError("matched-pair fit and heldout rows overlap")
        if len(self.fit_texts) != len(self.fit_row_ids) or len(self.heldout_texts) != len(
            self.heldout_row_ids
        ):
            raise ValueError("matched-pair text and row counts differ")
        self.text_column = str(text_column)
        self.effect_folds = int(effect_folds)
        if not 2 <= self.effect_folds <= len(self.fit_row_ids):
            raise ValueError("matched-pair proof effect-fold count is not feasible")
        self.view_configs = tuple(json.loads(_canonical_json(row)) for row in view_configs)
        view_names = tuple(str(row.get("name") or "") for row in self.view_configs)
        if (
            not view_names
            or any(not name for name in view_names)
            or len(view_names) != len(set(view_names))
        ):
            raise ValueError("matched-pair proof requires unique named BoW views")
        self.propensity_caliper = float(propensity_caliper)
        self.outcome_caliper = float(outcome_caliper)
        self.max_controls_per_candidate = int(max_controls_per_candidate)
        self.nearest_fallback_controls = int(nearest_fallback_controls)
        if (
            not np.isfinite(self.propensity_caliper)
            or not np.isfinite(self.outcome_caliper)
            or self.propensity_caliper < 0.0
            or self.outcome_caliper < 0.0
            or self.max_controls_per_candidate < 1
            or self.nearest_fallback_controls < 0
        ):
            raise ValueError("matched-pair proof matching configuration is invalid")
        self.htr_model_tree_sha256 = (
            None if htr_model_tree_sha256 is None else str(htr_model_tree_sha256)
        )
        if (
            self.htr_model_tree_sha256 is not None
            and _SHA256.fullmatch(self.htr_model_tree_sha256) is None
        ):
            raise ValueError("matched-pair HTR tree digest is invalid")
        self.htr_prediction_batch_size = max(1, int(htr_prediction_batch_size))
        self.seed = int(seed)
        self._store = _ArrayStore()
        self._scope_inputs: dict[str, str] = {}
        self._bow_folds: list[dict[str, Any]] = []
        self._bow_full: list[dict[str, Any]] = []
        self._htr_folds: list[dict[str, Any]] = []
        self._scope_outputs: dict[str, dict[str, Any]] = {}
        self._htr_extractor_identity: Mapping[str, Any] | None = None
        self._lock = threading.Lock()
        self._finalized = False

    def record_scope_inputs(
        self,
        *,
        treatment: Any,
        outcome: Any,
        e_fit: Any,
        m_fit: Any,
        e_heldout: Any,
        m_heldout: Any,
    ) -> None:
        if self._scope_inputs:
            raise ValueError("matched-pair scope inputs were already captured")
        values = {
            "treatment": _finite_vector(
                treatment, name="matched-pair treatment", length=len(self.fit_row_ids)
            ),
            "outcome": _finite_vector(
                outcome, name="matched-pair outcome", length=len(self.fit_row_ids)
            ),
            "e_fit": _finite_vector(
                e_fit, name="matched-pair fit propensity", length=len(self.fit_row_ids)
            ),
            "m_fit": _finite_vector(
                m_fit, name="matched-pair fit outcome", length=len(self.fit_row_ids)
            ),
            "e_heldout": _finite_vector(
                e_heldout,
                name="matched-pair heldout propensity",
                length=len(self.heldout_row_ids),
            ),
            "m_heldout": _finite_vector(
                m_heldout,
                name="matched-pair heldout outcome",
                length=len(self.heldout_row_ids),
            ),
        }
        if not set(np.unique(values["treatment"])).issubset({0.0, 1.0}) or set(
            np.unique(values["treatment"])
        ) != {0.0, 1.0}:
            raise ValueError("matched-pair treatment must contain both binary arms")
        if not set(np.unique(values["outcome"])).issubset({0.0, 1.0}):
            raise ValueError("matched-pair outcome must be binary")
        self._scope_inputs = {
            name: self._store.add(f"scope_input_{name}", value) for name, value in values.items()
        }

    def _fold_identity(
        self,
        *,
        fit_pos: Sequence[int],
        validation_pos: Sequence[int],
    ) -> Mapping[str, Any]:
        fit_pos = np.asarray(fit_pos, dtype=int)
        validation_pos = np.asarray(validation_pos, dtype=int)
        if (
            fit_pos.ndim != 1
            or validation_pos.ndim != 1
            or set(fit_pos.tolist()) & set(validation_pos.tolist())
            or sorted(np.concatenate([fit_pos, validation_pos]).tolist())
            != list(range(len(self.fit_row_ids)))
        ):
            raise ValueError("matched-pair fold is not a fit-scope partition")
        fit_ids = [self.fit_row_ids[int(pos)] for pos in fit_pos]
        validation_ids = [self.fit_row_ids[int(pos)] for pos in validation_pos]
        return {
            "fit_positions": fit_pos.tolist(),
            "validation_positions": validation_pos.tolist(),
            "fit_row_ids": fit_ids,
            "validation_row_ids": validation_ids,
            "fit_row_fingerprint": _row_fingerprint(fit_ids),
            "validation_row_fingerprint": _row_fingerprint(validation_ids),
        }

    def _pair_outputs(
        self,
        *,
        prefix: str,
        validation_pairs: pd.DataFrame,
        validation_pair_delta: Any,
        validation_delta: Any,
        validation_probability: Any,
        validation_n_controls: Any,
        heldout_pairs: pd.DataFrame,
        heldout_pair_delta: Any,
        heldout_delta: Any,
        heldout_probability: Any,
        heldout_n_controls: Any,
        validation_count: int,
    ) -> Mapping[str, Any]:
        return {
            "validation_pair_fingerprint": _pair_fingerprint(validation_pairs),
            "heldout_pair_fingerprint": _pair_fingerprint(heldout_pairs),
            "validation_pair_delta": self._store.add(
                f"{prefix}_validation_pair_delta",
                _finite_vector(
                    validation_pair_delta,
                    name="validation pair delta",
                    length=len(validation_pairs),
                ),
            ),
            "heldout_pair_delta": self._store.add(
                f"{prefix}_heldout_pair_delta",
                _finite_vector(
                    heldout_pair_delta,
                    name="heldout pair delta",
                    length=len(heldout_pairs),
                ),
            ),
            "validation_delta": self._store.add(
                f"{prefix}_validation_delta", np.asarray(validation_delta, dtype=float)
            ),
            "validation_probability": self._store.add(
                f"{prefix}_validation_probability",
                np.asarray(validation_probability, dtype=float),
            ),
            "validation_n_controls": self._store.add(
                f"{prefix}_validation_n_controls",
                np.asarray(validation_n_controls, dtype=float),
            ),
            "heldout_delta": self._store.add(
                f"{prefix}_heldout_delta", np.asarray(heldout_delta, dtype=float)
            ),
            "heldout_probability": self._store.add(
                f"{prefix}_heldout_probability",
                np.asarray(heldout_probability, dtype=float),
            ),
            "heldout_n_controls": self._store.add(
                f"{prefix}_heldout_n_controls",
                np.asarray(heldout_n_controls, dtype=float),
            ),
            "validation_candidate_count": int(validation_count),
            "heldout_candidate_count": len(self.heldout_row_ids),
        }

    def record_bow_pair_fold(
        self,
        *,
        view_name: str,
        view_index: int,
        fold: int,
        fit_pos: Sequence[int],
        validation_pos: Sequence[int],
        fit_pairs: pd.DataFrame,
        validation_pairs: pd.DataFrame,
        heldout_pairs: pd.DataFrame,
        model: OffsetLogitBoWPairModel,
        validation_pair_delta: Any,
        validation_delta: Any,
        validation_probability: Any,
        validation_n_controls: Any,
        heldout_pair_delta: Any,
        heldout_delta: Any,
        heldout_probability: Any,
        heldout_n_controls: Any,
    ) -> None:
        if self._finalized:
            raise RuntimeError("matched-pair capture was already finalized")
        fold = int(fold)
        view_index = int(view_index)
        prefix = f"bow_{view_index:04d}_{fold:04d}"
        with self._lock:
            if any(
                int(row["view_index"]) == view_index and int(row["fold"]) == fold
                for row in self._bow_folds
            ):
                raise ValueError(f"duplicate BoW matched-pair fold: {view_index}/{fold}")
            row = {
                "subproducer": "bow",
                "objective": "matched_pair_uplift_delta_logit",
                "view_name": str(view_name),
                "view_index": view_index,
                "fold": fold,
                "split_seed": 91_000 + 100 * self.outer_fold + 1_000 * view_index,
                **self._fold_identity(
                    fit_pos=fit_pos,
                    validation_pos=validation_pos,
                ),
                "fit_pair_fingerprint": _pair_fingerprint(fit_pairs),
                "model": _capture_offset_model(model, self._store, prefix),
                **self._pair_outputs(
                    prefix=prefix,
                    validation_pairs=validation_pairs,
                    validation_pair_delta=validation_pair_delta,
                    validation_delta=validation_delta,
                    validation_probability=validation_probability,
                    validation_n_controls=validation_n_controls,
                    heldout_pairs=heldout_pairs,
                    heldout_pair_delta=heldout_pair_delta,
                    heldout_delta=heldout_delta,
                    heldout_probability=heldout_probability,
                    heldout_n_controls=heldout_n_controls,
                    validation_count=len(validation_pos),
                ),
                "heldout_labels_accessed": False,
            }
            self._bow_folds.append(row)

    def record_bow_pair_full(
        self,
        *,
        view_name: str,
        view_index: int,
        full_pairs: pd.DataFrame,
        offset_model: OffsetLogitBoWPairModel,
        ridge_model: RidgeDeltaBoWPairModel,
    ) -> None:
        view_index = int(view_index)
        prefix = f"bow_{view_index:04d}_full"
        if ridge_model.vectorizer is None or ridge_model.model is None:
            raise ValueError("matched-pair proof requires the fitted full ridge diagnostic")
        with self._lock:
            if any(int(row["view_index"]) == view_index for row in self._bow_full):
                raise ValueError(f"duplicate full BoW matched-pair model: {view_index}")
            ridge_vectorizer = _capture_vectorizer(
                ridge_model.vectorizer,
                self._store,
                f"{prefix}_ridge_vectorizer",
                vectorizer_params=ridge_model.vectorizer_params,
            )
            ridge_learner = _capture_learner(
                ridge_model.model,
                self._store,
                f"{prefix}_ridge_learner",
                classification=False,
            )
            offset_prediction = offset_model.predict_delta_logit(full_pairs)
            ridge_prediction = ridge_model.predict_delta_prob(full_pairs)
            self._bow_full.append(
                {
                    "subproducer": "bow",
                    "objective": "matched_pair_full_importance_models",
                    "view_name": str(view_name),
                    "view_index": view_index,
                    "fit_row_ids": list(self.fit_row_ids),
                    "fit_row_fingerprint": _row_fingerprint(self.fit_row_ids),
                    "full_pair_fingerprint": _pair_fingerprint(full_pairs),
                    "offset_model": _capture_offset_model(
                        offset_model,
                        self._store,
                        f"{prefix}_offset",
                    ),
                    "ridge_vectorizer": ridge_vectorizer,
                    "ridge_learner": ridge_learner,
                    "offset_prediction": self._store.add(
                        f"{prefix}_offset_prediction", offset_prediction
                    ),
                    "ridge_prediction": self._store.add(
                        f"{prefix}_ridge_prediction", ridge_prediction
                    ),
                    "heldout_labels_accessed": False,
                }
            )

    def record_htr_pair_fold(
        self,
        *,
        fold: int,
        fit_pos: Sequence[int],
        validation_pos: Sequence[int],
        fit_pairs: pd.DataFrame,
        validation_pairs: pd.DataFrame,
        heldout_pairs: pd.DataFrame,
        model: HTRPairUpliftNet | None,
        validation_pair_delta: Any,
        validation_delta: Any,
        validation_probability: Any,
        validation_n_controls: Any,
        heldout_pair_delta: Any,
        heldout_delta: Any,
        heldout_probability: Any,
        heldout_n_controls: Any,
    ) -> None:
        fold = int(fold)
        prefix = f"htr_{fold:04d}"
        if model is None:
            raise RuntimeError(
                "matched-pair proof requires a genuinely fitted HTR pair network, "
                "not its zero fallback"
            )
        with self._lock:
            if any(int(row["fold"]) == fold for row in self._htr_folds):
                raise ValueError(f"duplicate HTR matched-pair fold: {fold}")
            model_state = _capture_htr_pair_model(model, self._store, prefix)
            if model is not None:
                extractor = model_state["extractor"]
                if extractor.get("hash_backend") is False and self.htr_model_tree_sha256 is None:
                    raise ValueError("HTR pair model lacks its authenticated model-tree binding")
                if self._htr_extractor_identity is None:
                    self._htr_extractor_identity = json.loads(_canonical_json(extractor))
                elif self._htr_extractor_identity != extractor:
                    raise RuntimeError("HTR pair extractor configuration changed between folds")
            self._htr_folds.append(
                {
                    "subproducer": "htr",
                    "objective": "matched_pair_uplift_delta_logit",
                    "fold": fold,
                    "split_seed": 92_000 + self.outer_fold,
                    **self._fold_identity(
                        fit_pos=fit_pos,
                        validation_pos=validation_pos,
                    ),
                    "fit_pair_fingerprint": _pair_fingerprint(fit_pairs),
                    "model": model_state,
                    **self._pair_outputs(
                        prefix=prefix,
                        validation_pairs=validation_pairs,
                        validation_pair_delta=validation_pair_delta,
                        validation_delta=validation_delta,
                        validation_probability=validation_probability,
                        validation_n_controls=validation_n_controls,
                        heldout_pairs=heldout_pairs,
                        heldout_pair_delta=heldout_pair_delta,
                        heldout_delta=heldout_delta,
                        heldout_probability=heldout_probability,
                        heldout_n_controls=heldout_n_controls,
                        validation_count=len(validation_pos),
                    ),
                    "heldout_labels_accessed": False,
                }
            )

    def record_scope_output(self, name: str, value: Any, *, role: str) -> None:
        name = str(name)
        if _SAFE_KEY.fullmatch(name) is None or name in self._scope_outputs:
            raise ValueError(f"invalid or duplicate matched-pair scope output: {name}")
        self._scope_outputs[name] = {
            "role": str(role),
            "array": self._store.add(f"scope_output_{name}", np.asarray(value)),
        }

    def finalize(self) -> Mapping[str, Any]:
        if self._finalized:
            raise RuntimeError("matched-pair capture was already finalized")
        if set(self._scope_inputs) != {
            "treatment",
            "outcome",
            "e_fit",
            "m_fit",
            "e_heldout",
            "m_heldout",
        }:
            raise RuntimeError("matched-pair capture lacks shared nuisance inputs")
        bow_folds = sorted(
            self._bow_folds,
            key=lambda row: (int(row["view_index"]), int(row["fold"])),
        )
        bow_full = sorted(self._bow_full, key=lambda row: int(row["view_index"]))
        htr_folds = sorted(self._htr_folds, key=lambda row: int(row["fold"]))
        for view_index, view in enumerate(self.view_configs):
            rows = [row for row in bow_folds if int(row["view_index"]) == view_index]
            if [int(row["fold"]) for row in rows] != list(range(1, self.effect_folds + 1)) or any(
                row["view_name"] != str(view["name"]) for row in rows
            ):
                raise RuntimeError(
                    f"matched-pair capture lacks BoW fold coverage for view {view_index}"
                )
        if [int(row["view_index"]) for row in bow_full] != list(range(len(self.view_configs))):
            raise RuntimeError("matched-pair capture lacks full BoW importance models")
        if [int(row["fold"]) for row in htr_folds] != list(range(1, self.effect_folds + 1)):
            raise RuntimeError("matched-pair capture lacks HTR fold coverage")
        if self._htr_extractor_identity is None:
            raise RuntimeError("matched-pair capture lacks a genuine HTR extractor fit")
        required_outputs = {
            *(
                f"bow_view_{index:04d}_{value}_{split}"
                for index in range(len(self.view_configs))
                for value in ("delta", "probability", "n_controls")
                for split in ("fit", "heldout")
            ),
            *(
                f"htr_{value}_{split}"
                for value in ("delta", "probability", "n_controls")
                for split in ("fit", "heldout")
            ),
        }
        if set(self._scope_outputs) != required_outputs:
            raise RuntimeError("matched-pair capture final-output coverage changed")
        body = {
            "schema_version": MATCHED_PAIR_NATIVE_CAPTURE_SCHEMA,
            "scope_id": self.scope_id,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "fit_row_ids": list(self.fit_row_ids),
            "heldout_row_ids": list(self.heldout_row_ids),
            "fit_row_fingerprint": _row_fingerprint(self.fit_row_ids),
            "heldout_row_fingerprint": _row_fingerprint(self.heldout_row_ids),
            "fit_text_sha256": _text_sha256(self.fit_row_ids, self.fit_texts),
            "heldout_text_sha256": _text_sha256(self.heldout_row_ids, self.heldout_texts),
            "text_column": self.text_column,
            "effect_folds": self.effect_folds,
            "view_configs": self.view_configs,
            "matching_configuration": {
                "propensity_caliper": self.propensity_caliper,
                "outcome_caliper": self.outcome_caliper,
                "max_controls_per_candidate": self.max_controls_per_candidate,
                "nearest_fallback_controls": self.nearest_fallback_controls,
            },
            "htr_model_tree_sha256": self.htr_model_tree_sha256,
            "htr_prediction_batch_size": self.htr_prediction_batch_size,
            "htr_extractor_identity": self._htr_extractor_identity,
            "seed": self.seed,
            "scope_inputs": self._scope_inputs,
            "bow_fold_states": bow_folds,
            "bow_full_fit_states": bow_full,
            "htr_fold_states": htr_folds,
            "scope_outputs": self._scope_outputs,
            "array_inventory": self._store.inventory,
            "subproducer_coverage": ["bow", "htr"],
            "heldout_columns_read": ["_oci_row_id", self.text_column],
            "heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
            "secrets_accessed": False,
            "executable_checkpoint_retained": False,
            "pickle_or_joblib_loaded": False,
        }
        self.artifact_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_npz(self.artifact_dir / "arrays.npz", self._store.arrays)
        closed = {
            **body,
            "arrays_file": "arrays.npz",
            "arrays_file_sha256": _sha256_file(self.artifact_dir / "arrays.npz"),
        }
        metadata = {**closed, "content_sha256": _sha256_json(closed)}
        _atomic_write_bytes(
            self.artifact_dir / "metadata.json",
            (_canonical_json(metadata) + "\n").encode("utf-8"),
        )
        self._finalized = True
        return json.loads(_canonical_json(metadata))


def _load_capture(path: Path | str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("matched-pair native capture must be one regular directory")
    if any(item.is_symlink() for item in root.rglob("*")):
        raise ValueError("matched-pair native capture cannot contain symlinks")
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if {item.relative_to(root).as_posix() for item in files} != {
        "arrays.npz",
        "metadata.json",
    } or any(item.suffix.lower() in _FORBIDDEN_SUFFIXES for item in files):
        raise ValueError("matched-pair native capture has an unsafe file inventory")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("matched-pair native metadata is invalid JSON") from exc
    body = {key: value for key, value in metadata.items() if key != "content_sha256"}
    if (
        metadata.get("schema_version") != MATCHED_PAIR_NATIVE_CAPTURE_SCHEMA
        or metadata.get("content_sha256") != _sha256_json(body)
        or metadata.get("arrays_file") != "arrays.npz"
        or metadata.get("arrays_file_sha256") != _sha256_file(root / "arrays.npz")
        or metadata.get("subproducer_coverage") != ["bow", "htr"]
        or metadata.get("heldout_columns_read") != ["_oci_row_id", metadata.get("text_column")]
        or metadata.get("heldout_labels_accessed") is not False
        or metadata.get("oracle_fields_accessed") is not False
        or metadata.get("secrets_accessed") is not False
        or metadata.get("executable_checkpoint_retained") is not False
        or metadata.get("pickle_or_joblib_loaded") is not False
    ):
        raise ValueError("matched-pair native metadata has an invalid envelope")
    try:
        loaded = np.load(root / "arrays.npz", allow_pickle=False)
        arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
        loaded.close()
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError("matched-pair native numerical artifact is invalid") from exc
    inventory = metadata.get("array_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != set(arrays):
        raise ValueError("matched-pair numerical inventory is not closed")
    for key, array in arrays.items():
        row = inventory.get(key)
        if (
            not isinstance(row, Mapping)
            or row.get("dtype") != array.dtype.str
            or row.get("shape") != list(array.shape)
            or row.get("content_sha256") != _array_sha256(array)
            or array.dtype.hasobject
        ):
            raise ValueError(f"matched-pair numerical array changed: {key}")
    return metadata, arrays


def _assert_close(observed: Any, expected: Any, *, name: str) -> None:
    left = np.asarray(observed)
    right = np.asarray(expected)
    if left.shape != right.shape or not np.allclose(
        left,
        right,
        rtol=3e-5,
        atol=3e-6,
        equal_nan=True,
    ):
        raise RuntimeError(f"matched-pair native replay changed {name}")


def _scope_frames(
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame({"_oci_row_id": np.asarray(fit_row_ids, dtype=np.int64)}),
        pd.DataFrame({"_oci_row_id": np.asarray(heldout_row_ids, dtype=np.int64)}),
    )


def validate_matched_pair_native_capture(
    path: Path | str,
    *,
    expected_scope_id: str,
    expected_fit_row_ids: Sequence[int],
    expected_heldout_row_ids: Sequence[int],
    fit_texts: Sequence[str],
    heldout_texts: Sequence[str],
    expected_fit_treatment: Sequence[float] | None = None,
    expected_fit_outcome: Sequence[float] | None = None,
    htr_model_path: Path | str | None = None,
    expected_htr_model_tree_sha256: str | None = None,
    device: torch.device | str = "cpu",
) -> Mapping[str, Any]:
    metadata, arrays = _load_capture(path)
    fit_rows = tuple(map(int, expected_fit_row_ids))
    heldout_rows = tuple(map(int, expected_heldout_row_ids))
    fit_texts = tuple(map(str, fit_texts))
    heldout_texts = tuple(map(str, heldout_texts))
    if (
        metadata.get("scope_id") != str(expected_scope_id)
        or tuple(map(int, metadata.get("fit_row_ids") or ())) != fit_rows
        or tuple(map(int, metadata.get("heldout_row_ids") or ())) != heldout_rows
        or metadata.get("fit_row_fingerprint") != _row_fingerprint(fit_rows)
        or metadata.get("heldout_row_fingerprint") != _row_fingerprint(heldout_rows)
        or metadata.get("fit_text_sha256") != _text_sha256(fit_rows, fit_texts)
        or metadata.get("heldout_text_sha256") != _text_sha256(heldout_rows, heldout_texts)
    ):
        raise ValueError("matched-pair native capture changed its row/text scope")
    declared_tree = metadata.get("htr_model_tree_sha256")
    htr_identity = metadata.get("htr_extractor_identity")
    if not isinstance(htr_identity, Mapping) or not isinstance(
        htr_identity.get("hash_backend"), bool
    ):
        raise ValueError("matched-pair capture lacks a genuine HTR extractor identity")
    if htr_identity.get("hash_backend") is False:
        expected_tree = str(expected_htr_model_tree_sha256 or "")
        if (
            _SHA256.fullmatch(expected_tree) is None
            or declared_tree != expected_tree
            or htr_model_path is None
            or directory_tree_sha256(htr_model_path) != expected_tree
        ):
            raise RuntimeError("matched-pair HTR model-tree binding changed")
    elif declared_tree is not None or expected_htr_model_tree_sha256 is not None:
        raise ValueError("hash/constant HTR pair capture cannot claim a model tree")
    scope_inputs = metadata.get("scope_inputs")
    if not isinstance(scope_inputs, Mapping) or set(scope_inputs) != {
        "treatment",
        "outcome",
        "e_fit",
        "m_fit",
        "e_heldout",
        "m_heldout",
    }:
        raise ValueError("matched-pair native capture lacks nuisance inputs")
    t = _finite_vector(
        arrays[str(scope_inputs["treatment"])],
        name="captured treatment",
        length=len(fit_rows),
    )
    y = _finite_vector(
        arrays[str(scope_inputs["outcome"])],
        name="captured outcome",
        length=len(fit_rows),
    )
    if (expected_fit_treatment is None) != (expected_fit_outcome is None):
        raise ValueError(
            "matched-pair canonical fit treatment/outcome must be supplied together"
        )
    if expected_fit_treatment is not None:
        canonical_treatment = _finite_vector(
            expected_fit_treatment,
            name="canonical fit treatment",
            length=len(fit_rows),
        )
        canonical_outcome = _finite_vector(
            expected_fit_outcome,
            name="canonical fit outcome",
            length=len(fit_rows),
        )
        if not np.array_equal(t, canonical_treatment):
            raise ValueError(
                "matched-pair native capture treatment differs from canonical fit labels"
            )
        if not np.array_equal(y, canonical_outcome):
            raise ValueError(
                "matched-pair native capture outcome differs from canonical fit labels"
            )
    e_fit = _finite_vector(
        arrays[str(scope_inputs["e_fit"])],
        name="captured fit propensity",
        length=len(fit_rows),
    )
    m_fit = _finite_vector(
        arrays[str(scope_inputs["m_fit"])],
        name="captured fit outcome",
        length=len(fit_rows),
    )
    e_heldout = _finite_vector(
        arrays[str(scope_inputs["e_heldout"])],
        name="captured heldout propensity",
        length=len(heldout_rows),
    )
    m_heldout = _finite_vector(
        arrays[str(scope_inputs["m_heldout"])],
        name="captured heldout outcome",
        length=len(heldout_rows),
    )
    if (
        set(np.unique(t)) != {0.0, 1.0}
        or not set(np.unique(y)).issubset({0.0, 1.0})
        or any(
            np.any((values < 0.0) | (values > 1.0))
            for values in (e_fit, m_fit, e_heldout, m_heldout)
        )
    ):
        raise ValueError("matched-pair captured labels/nuisance values are invalid")
    fit_df, heldout_df = _scope_frames(fit_rows, heldout_rows)
    matching = metadata.get("matching_configuration") or {}
    common_training = {
        "propensity_caliper": float(matching["propensity_caliper"]),
        "outcome_caliper": float(matching["outcome_caliper"]),
    }
    common_candidate = {
        **common_training,
        "max_controls_per_candidate": int(matching["max_controls_per_candidate"]),
        "nearest_fallback_controls": int(matching["nearest_fallback_controls"]),
    }
    if (
        not np.isfinite(common_training["propensity_caliper"])
        or not np.isfinite(common_training["outcome_caliper"])
        or common_training["propensity_caliper"] < 0.0
        or common_training["outcome_caliper"] < 0.0
        or common_candidate["max_controls_per_candidate"] < 1
        or common_candidate["nearest_fallback_controls"] < 0
    ):
        raise ValueError("matched-pair matching configuration changed")
    effect_folds = int(metadata["effect_folds"])
    if not 2 <= effect_folds <= len(fit_rows):
        raise ValueError("matched-pair effect-fold count changed")
    device = torch.device(device)

    def rebuild_pairs(fit_pos: np.ndarray, validation_pos: np.ndarray):
        fit_subset = fit_df.iloc[fit_pos].reset_index(drop=True)
        validation_subset = fit_df.iloc[validation_pos].reset_index(drop=True)
        training_pairs = build_training_pairs(
            fit_subset,
            texts=[fit_texts[int(pos)] for pos in fit_pos],
            treatment=t[fit_pos],
            outcome=y[fit_pos],
            propensity=e_fit[fit_pos],
            outcome_prob=m_fit[fit_pos],
            **common_training,
        )
        control_pos = fit_pos[t[fit_pos].astype(int) == 0]
        control_df = fit_df.iloc[control_pos].reset_index(drop=True)
        control_texts = [fit_texts[int(pos)] for pos in control_pos]
        validation_pairs = build_candidate_pairs(
            validation_subset,
            control_df,
            candidate_texts=[fit_texts[int(pos)] for pos in validation_pos],
            control_texts=control_texts,
            candidate_propensity=e_fit[validation_pos],
            candidate_outcome_prob=m_fit[validation_pos],
            control_propensity=e_fit[control_pos],
            control_outcome_prob=m_fit[control_pos],
            **common_candidate,
        )
        heldout_pairs = build_candidate_pairs(
            heldout_df,
            control_df,
            candidate_texts=heldout_texts,
            control_texts=control_texts,
            candidate_propensity=e_heldout,
            candidate_outcome_prob=m_heldout,
            control_propensity=e_fit[control_pos],
            control_outcome_prob=m_fit[control_pos],
            **common_candidate,
        )
        return training_pairs, validation_pairs, heldout_pairs

    def assert_fold_identity(
        row: Mapping[str, Any],
        *,
        fit_pos: np.ndarray,
        validation_pos: np.ndarray,
        subproducer: str,
        split_seed: int,
    ) -> None:
        expected_fit_rows = [fit_rows[int(pos)] for pos in fit_pos]
        expected_validation_rows = [fit_rows[int(pos)] for pos in validation_pos]
        if (
            row.get("subproducer") != subproducer
            or row.get("objective") != "matched_pair_uplift_delta_logit"
            or int(row.get("split_seed", -1)) != int(split_seed)
            or row.get("fit_row_ids") != expected_fit_rows
            or row.get("validation_row_ids") != expected_validation_rows
            or row.get("fit_row_fingerprint") != _row_fingerprint(expected_fit_rows)
            or row.get("validation_row_fingerprint") != _row_fingerprint(expected_validation_rows)
            or int(row.get("validation_candidate_count", -1)) != len(validation_pos)
            or int(row.get("heldout_candidate_count", -1)) != len(heldout_rows)
            or row.get("heldout_labels_accessed") is not False
        ):
            raise ValueError(f"matched-pair {subproducer} fold row/objective identity changed")

    output_vectors: dict[str, np.ndarray] = {}
    view_configs = metadata.get("view_configs") or ()
    if (
        not isinstance(view_configs, list)
        or not view_configs
        or any(not isinstance(view, Mapping) for view in view_configs)
        or any(not str(view.get("name") or "") for view in view_configs)
        or len({str(view["name"]) for view in view_configs}) != len(view_configs)
    ):
        raise ValueError("matched-pair BoW view configuration changed")
    bow_rows = metadata.get("bow_fold_states") or ()
    for view_index, view in enumerate(view_configs):
        rows = [row for row in bow_rows if int(row.get("view_index", -1)) == view_index]
        if [int(row.get("fold", 0)) for row in rows] != list(range(1, effect_folds + 1)) or any(
            row.get("view_name") != view.get("name") for row in rows
        ):
            raise ValueError("matched-pair BoW fold coverage changed")
        splits = list(
            KFold(
                n_splits=effect_folds,
                shuffle=True,
                random_state=91_000 + 100 * int(metadata["outer_fold"]) + 1_000 * view_index,
            ).split(fit_df)
        )
        oof_delta = np.full(len(fit_rows), np.nan)
        oof_probability = np.full(len(fit_rows), np.nan)
        oof_n = np.zeros(len(fit_rows))
        heldout_delta_rows = []
        heldout_probability_rows = []
        heldout_n_rows = []
        for row, (expected_fit, expected_validation) in zip(rows, splits):
            fit_pos = np.asarray(row["fit_positions"], dtype=int)
            validation_pos = np.asarray(row["validation_positions"], dtype=int)
            if not np.array_equal(fit_pos, expected_fit) or not np.array_equal(
                validation_pos, expected_validation
            ):
                raise ValueError("matched-pair BoW fold split changed")
            split_seed = 91_000 + 100 * int(metadata["outer_fold"]) + 1_000 * view_index
            assert_fold_identity(
                row,
                fit_pos=fit_pos,
                validation_pos=validation_pos,
                subproducer="bow",
                split_seed=split_seed,
            )
            expected_vectorizer_params = view.get("vectorizer_scientific")
            if not isinstance(expected_vectorizer_params, Mapping):
                raise ValueError(
                    "matched-pair BoW view lacks exact vectorizer science"
                )
            if row.get("model", {}).get("vectorizer_params") != expected_vectorizer_params:
                raise ValueError("matched-pair BoW view/model configuration changed")
            fit_pairs, validation_pairs, heldout_pairs = rebuild_pairs(fit_pos, validation_pos)
            if (
                row.get("fit_pair_fingerprint") != _pair_fingerprint(fit_pairs)
                or row.get("validation_pair_fingerprint") != _pair_fingerprint(validation_pairs)
                or row.get("heldout_pair_fingerprint") != _pair_fingerprint(heldout_pairs)
                or row.get("heldout_labels_accessed") is not False
            ):
                raise ValueError("matched-pair BoW pair construction changed")
            validation_pair_delta = _predict_offset_model(row["model"], arrays, validation_pairs)
            heldout_pair_delta = _predict_offset_model(row["model"], arrays, heldout_pairs)
            validation_values = aggregate_pair_predictions(
                validation_pairs, validation_pair_delta, len(validation_pos)
            )
            heldout_values = aggregate_pair_predictions(
                heldout_pairs, heldout_pair_delta, len(heldout_rows)
            )
            for observed, key, name in (
                (validation_pair_delta, row["validation_pair_delta"], "validation pair delta"),
                (heldout_pair_delta, row["heldout_pair_delta"], "heldout pair delta"),
                (validation_values[0], row["validation_delta"], "validation delta"),
                (validation_values[1], row["validation_probability"], "validation probability"),
                (validation_values[2], row["validation_n_controls"], "validation controls"),
                (heldout_values[0], row["heldout_delta"], "heldout delta"),
                (heldout_values[1], row["heldout_probability"], "heldout probability"),
                (heldout_values[2], row["heldout_n_controls"], "heldout controls"),
            ):
                _assert_close(observed, arrays[str(key)], name=f"BoW {name}")
            oof_delta[validation_pos] = validation_values[0]
            oof_probability[validation_pos] = validation_values[1]
            oof_n[validation_pos] = validation_values[2]
            heldout_delta_rows.append(heldout_values[0])
            heldout_probability_rows.append(heldout_values[1])
            heldout_n_rows.append(heldout_values[2])
        for value_name, fit_value, heldout_value in (
            ("delta", oof_delta, np.nanmean(np.vstack(heldout_delta_rows), axis=0)),
            (
                "probability",
                oof_probability,
                np.nanmean(np.vstack(heldout_probability_rows), axis=0),
            ),
            ("n_controls", oof_n, np.nanmean(np.vstack(heldout_n_rows), axis=0)),
        ):
            output_vectors[f"bow_view_{view_index:04d}_{value_name}_fit"] = fit_value
            output_vectors[f"bow_view_{view_index:04d}_{value_name}_heldout"] = heldout_value

    full_pairs = build_training_pairs(
        fit_df,
        texts=fit_texts,
        treatment=t,
        outcome=y,
        propensity=e_fit,
        outcome_prob=m_fit,
        **common_training,
    )
    full_rows = metadata.get("bow_full_fit_states") or ()
    if [int(row.get("view_index", -1)) for row in full_rows] != list(range(len(view_configs))):
        raise ValueError("matched-pair full BoW model coverage changed")
    for row in full_rows:
        view_index = int(row["view_index"])
        view = view_configs[view_index]
        expected_vectorizer_params = view.get("vectorizer_scientific")
        if not isinstance(expected_vectorizer_params, Mapping):
            raise ValueError(
                "matched-pair BoW view lacks exact vectorizer science"
            )
        if (
            row.get("subproducer") != "bow"
            or row.get("objective") != "matched_pair_full_importance_models"
            or row.get("view_name") != view.get("name")
            or row.get("fit_row_ids") != list(fit_rows)
            or row.get("fit_row_fingerprint") != _row_fingerprint(fit_rows)
            or row.get("full_pair_fingerprint") != _pair_fingerprint(full_pairs)
            or row.get("offset_model", {}).get("vectorizer_params") != expected_vectorizer_params
            or row.get("ridge_vectorizer", {}).get("params") != expected_vectorizer_params
            or row.get("heldout_labels_accessed") is not False
        ):
            raise ValueError("matched-pair full training-pair construction changed")
        offset_prediction = _predict_offset_model(row["offset_model"], arrays, full_pairs)
        ridge_vectorizer = _restore_vectorizer(row["ridge_vectorizer"], arrays)
        ridge_matrix = sparse.hstack(
            [
                ridge_vectorizer.transform(full_pairs["control_text"].astype(str).tolist()),
                ridge_vectorizer.transform(full_pairs["treated_text"].astype(str).tolist()),
            ],
            format="csr",
        )
        ridge_prediction = _predict_learner(row["ridge_learner"], arrays, ridge_matrix)
        _assert_close(
            offset_prediction,
            arrays[str(row["offset_prediction"])],
            name="full offset prediction",
        )
        _assert_close(
            ridge_prediction,
            arrays[str(row["ridge_prediction"])],
            name="full ridge prediction",
        )

    htr_rows = metadata.get("htr_fold_states") or ()
    if [int(row.get("fold", 0)) for row in htr_rows] != list(range(1, effect_folds + 1)):
        raise ValueError("matched-pair HTR fold coverage changed")
    splits = list(
        KFold(
            n_splits=effect_folds,
            shuffle=True,
            random_state=92_000 + int(metadata["outer_fold"]),
        ).split(fit_df)
    )
    htr_oof_delta = np.full(len(fit_rows), np.nan)
    htr_oof_probability = np.full(len(fit_rows), np.nan)
    htr_oof_n = np.zeros(len(fit_rows))
    htr_heldout_delta = []
    htr_heldout_probability = []
    htr_heldout_n = []
    for row, (expected_fit, expected_validation) in zip(htr_rows, splits):
        fit_pos = np.asarray(row["fit_positions"], dtype=int)
        validation_pos = np.asarray(row["validation_positions"], dtype=int)
        if not np.array_equal(fit_pos, expected_fit) or not np.array_equal(
            validation_pos, expected_validation
        ):
            raise ValueError("matched-pair HTR fold split changed")
        assert_fold_identity(
            row,
            fit_pos=fit_pos,
            validation_pos=validation_pos,
            subproducer="htr",
            split_seed=92_000 + int(metadata["outer_fold"]),
        )
        if (
            row.get("model", {}).get("kind") != "htr_pair_network"
            or row.get("model", {}).get("extractor") != htr_identity
        ):
            raise ValueError("matched-pair HTR native model identity changed")
        fit_pairs, validation_pairs, heldout_pairs = rebuild_pairs(fit_pos, validation_pos)
        if (
            row.get("fit_pair_fingerprint") != _pair_fingerprint(fit_pairs)
            or row.get("validation_pair_fingerprint") != _pair_fingerprint(validation_pairs)
            or row.get("heldout_pair_fingerprint") != _pair_fingerprint(heldout_pairs)
        ):
            raise ValueError("matched-pair HTR pair construction changed")
        init_texts = (
            fit_pairs["control_text"].astype(str).tolist()
            + fit_pairs["treated_text"].astype(str).tolist()
        )
        model = _build_htr_pair_model(
            row["model"],
            arrays,
            initialization_texts=init_texts,
            htr_model_path=htr_model_path,
            device=device,
        )
        validation_pair_delta = _predict_htr_pair(
            model,
            validation_pairs,
            batch_size=int(metadata["htr_prediction_batch_size"]),
        )
        heldout_pair_delta = _predict_htr_pair(
            model,
            heldout_pairs,
            batch_size=int(metadata["htr_prediction_batch_size"]),
        )
        validation_values = aggregate_pair_predictions(
            validation_pairs, validation_pair_delta, len(validation_pos)
        )
        heldout_values = aggregate_pair_predictions(
            heldout_pairs, heldout_pair_delta, len(heldout_rows)
        )
        for observed, key, name in (
            (validation_pair_delta, row["validation_pair_delta"], "validation pair delta"),
            (heldout_pair_delta, row["heldout_pair_delta"], "heldout pair delta"),
            (validation_values[0], row["validation_delta"], "validation delta"),
            (validation_values[1], row["validation_probability"], "validation probability"),
            (validation_values[2], row["validation_n_controls"], "validation controls"),
            (heldout_values[0], row["heldout_delta"], "heldout delta"),
            (heldout_values[1], row["heldout_probability"], "heldout probability"),
            (heldout_values[2], row["heldout_n_controls"], "heldout controls"),
        ):
            _assert_close(observed, arrays[str(key)], name=f"HTR {name}")
        htr_oof_delta[validation_pos] = validation_values[0]
        htr_oof_probability[validation_pos] = validation_values[1]
        htr_oof_n[validation_pos] = validation_values[2]
        htr_heldout_delta.append(heldout_values[0])
        htr_heldout_probability.append(heldout_values[1])
        htr_heldout_n.append(heldout_values[2])
    for value_name, fit_value, heldout_value in (
        ("delta", htr_oof_delta, np.nanmean(np.vstack(htr_heldout_delta), axis=0)),
        (
            "probability",
            htr_oof_probability,
            np.nanmean(np.vstack(htr_heldout_probability), axis=0),
        ),
        ("n_controls", htr_oof_n, np.nanmean(np.vstack(htr_heldout_n), axis=0)),
    ):
        output_vectors[f"htr_{value_name}_fit"] = fit_value
        output_vectors[f"htr_{value_name}_heldout"] = heldout_value
    scope_outputs = metadata.get("scope_outputs")
    if not isinstance(scope_outputs, Mapping) or set(scope_outputs) != set(output_vectors):
        raise ValueError("matched-pair final-output coverage changed")
    for name, expected in output_vectors.items():
        split = "heldout" if name.endswith("_heldout") else "fit"
        if "_n_controls_" in name:
            value_role = "matched_control_count"
        elif "_probability_" in name:
            value_role = "treated_outcome_probability"
        else:
            value_role = "uplift_delta_logit"
        if scope_outputs[name].get("role") != f"{split}_{value_role}":
            raise ValueError(f"matched-pair scope output role changed: {name}")
        _assert_close(
            arrays[str(scope_outputs[name]["array"])],
            expected,
            name=f"scope output {name}",
        )
    return json.loads(_canonical_json(metadata))


__all__ = [
    "MATCHED_PAIR_NATIVE_CAPTURE_SCHEMA",
    "NativeMatchedPairProofCaptureSink",
    "validate_matched_pair_native_capture",
]
