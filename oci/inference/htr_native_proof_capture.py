"""Safe native-state capture and replay for Stage 1 HTR neural fits.

The production Stage 1 wrapper uses this module to retain the *actual* fitted
HTR nuisance and effect models before the ordinary runner releases them.  A
capture consists only of closed JSON metadata and plain NumPy arrays; it never
writes or loads a Python, pickle, joblib, or torch checkpoint.

Validation reconstructs the exact registered neural architecture from code,
loads every tensor from the non-executable NPZ payload, and replays fit,
validation, and held-out text transforms.  The nested nuisance calibrators and
the effect-objective formulas are replayed as well, so stored output vectors by
themselves cannot satisfy this proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.model_selection import KFold

from ..models.hierarchical_transformer_extractor import HierarchicalTransformerExtractor
from ..utils.calibration import BinaryProbabilityCalibrator, clip_probability
from .agentic_attention_variable_forest import (
    _EffectNet,
    _NuisanceNet,
    _binary_log_loss_from_logits,
    _logistic_r_logits,
    _logistic_r_tau_from_delta,
    _r_pseudo_outcome,
)

HTR_NATIVE_CAPTURE_SCHEMA = "production_htr_native_capture_v1"
HTR_NATIVE_CAPTURE_ARRAY_SCHEMA = "production_htr_native_capture_array_v1"

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
_EFFECT_OBJECTIVES = ("pseudo_outcome_mse", "squared_r_loss")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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
                "schema_version": HTR_NATIVE_CAPTURE_ARRAY_SCHEMA,
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


def directory_tree_sha256(path: Path | str) -> str:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HTR model directory does not exist: {root}")
    rows = [
        {
            "relative_path": candidate.relative_to(root).as_posix(),
            "size": int(candidate.stat().st_size),
            "sha256": _sha256_file(candidate),
        }
        for candidate in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    if not rows:
        raise ValueError(f"HTR model directory contains no files: {root}")
    return _sha256_json(rows)


def _text_sha256(row_ids: Sequence[int], texts: Sequence[str]) -> str:
    rows = tuple(map(int, row_ids))
    values = tuple(str(text) for text in texts)
    if len(rows) != len(values):
        raise ValueError("HTR text binding requires one text per row ID")
    digest = hashlib.sha256()
    digest.update(b"production-htr-text-binding-v1\0")
    for row_id, text in zip(rows, values):
        encoded = text.encode("utf-8")
        digest.update(int(row_id).to_bytes(8, byteorder="little", signed=False))
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _row_fingerprint(row_ids: Sequence[int]) -> str:
    rows = tuple(map(int, row_ids))
    if not rows or len(rows) != len(set(rows)) or any(row < 0 for row in rows):
        raise ValueError("HTR row IDs must be unique non-negative integers")
    return _sha256_json({"ordered_row_ids": list(rows)})


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"refusing to replace immutable HTR artifact: {path}")
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
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"refusing to replace immutable HTR artifact: {path}")
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
            raise ValueError(f"invalid or duplicate HTR capture array key: {key}")
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise ValueError("HTR capture arrays cannot use object dtype")
        self.arrays[key] = array
        self.inventory[key] = {
            "dtype": array.dtype.str,
            "shape": [int(item) for item in array.shape],
            "content_sha256": _array_sha256(array),
        }
        return key


def _tensor_storage(tensor: torch.Tensor) -> tuple[np.ndarray, str]:
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise TypeError("HTR proof supports only dense strided torch tensors")
    detached = tensor.detach().cpu().contiguous()
    if detached.dtype == torch.bfloat16:
        return detached.view(torch.uint16).numpy().copy(), "torch.bfloat16"
    supported = {
        torch.float64,
        torch.float32,
        torch.float16,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    }
    if detached.dtype not in supported:
        raise TypeError(f"unsupported HTR tensor dtype: {detached.dtype}")
    return detached.numpy().copy(), str(detached.dtype)


def _restore_tensor(array: np.ndarray, dtype_name: str) -> torch.Tensor:
    storage = np.array(array, copy=True)
    if dtype_name == "torch.bfloat16":
        if storage.dtype != np.dtype("uint16"):
            raise ValueError("bfloat16 HTR state has the wrong storage dtype")
        return torch.from_numpy(storage).view(torch.bfloat16)
    expected = {
        "torch.float64": torch.float64,
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.int16": torch.int16,
        "torch.int8": torch.int8,
        "torch.uint8": torch.uint8,
        "torch.bool": torch.bool,
    }.get(dtype_name)
    if expected is None:
        raise ValueError(f"unsupported captured HTR tensor dtype: {dtype_name}")
    tensor = torch.from_numpy(storage)
    if tensor.dtype != expected:
        raise ValueError("captured HTR tensor storage dtype changed")
    return tensor


def _extractor_descriptor(extractor: Any) -> dict[str, Any]:
    if type(extractor) is not HierarchicalTransformerExtractor:
        raise TypeError(
            "native HTR proof requires the exact HierarchicalTransformerExtractor"
        )
    if not bool(extractor._encoder_initialized):
        raise RuntimeError("HTR extractor state was captured before encoder initialization")
    hash_backend = bool(extractor._hash_backend)
    return {
        "class_name": "HierarchicalTransformerExtractor",
        "constructor": {
            "sentence_encoder_model": "hash" if hash_backend else "authenticated_local_tree",
            "freeze_sentence_encoder": bool(extractor._freeze),
            "chunk_size_words": int(extractor._chunk_size_words),
            "chunk_overlap_words": int(extractor._chunk_overlap_words),
            "max_chunks": int(extractor._max_chunks),
            "max_chunk_length": int(extractor._max_chunk_length),
            "num_transformer_layers": int(extractor._num_layers),
            "num_attention_heads": int(extractor._num_heads),
            "transformer_dim": int(extractor._transformer_dim),
            "transformer_dropout": float(extractor._dropout),
            "projection_dim": int(extractor._projection_dim),
            "hash_embedding_dim": int(extractor._hash_embedding_dim),
            "sentence_encoder_batch_size": int(extractor._sentence_encoder_batch_size),
            "sentence_encoder_backend": str(extractor._sentence_encoder_backend),
            "sentence_pooling": str(extractor._sentence_pooling),
            "normalize_sentence_embeddings": bool(
                extractor._normalize_sentence_embeddings
            ),
            "trainable_sentence_encoder_layers": int(
                extractor._trainable_sentence_encoder_layers
            ),
            "role_attention": bool(extractor._role_attention),
            "w_attention_heads": int(extractor._w_attention_heads),
            "x_attention_heads": int(extractor._x_attention_heads),
            "transformer_feedforward_dim": int(
                extractor._transformer_feedforward_dim
            ),
            "transformer_activation": str(
                extractor._transformer_activation
            ),
            "transformer_norm_style": str(
                extractor._transformer_norm_style
            ),
            "transformer_layer_norm_eps": float(
                extractor._transformer_layer_norm_eps
            ),
            "transformer_layer_norm_elementwise_affine": bool(
                extractor._transformer_layer_norm_elementwise_affine
            ),
            "transformer_layer_norm_bias": bool(
                extractor._transformer_layer_norm_bias
            ),
            "transformer_attention_dropout": float(
                extractor._transformer_attention_dropout
            ),
            "transformer_residual_dropout": float(
                extractor._transformer_residual_dropout
            ),
            "transformer_feedforward_dropout": float(
                extractor._transformer_feedforward_dropout
            ),
            "transformer_attention_bias": bool(
                extractor._transformer_attention_bias
            ),
            "transformer_feedforward_bias": bool(
                extractor._transformer_feedforward_bias
            ),
            "output_projection_depth": int(
                extractor._output_projection_depth
            ),
            "output_projection_hidden_dim": int(
                extractor._output_projection_hidden_dim
            ),
            "output_projection_activation": str(
                extractor._output_projection_activation
            ),
            "output_projection_dropout": float(
                extractor._output_projection_dropout
            ),
            "output_projection_hidden_layer_norm": bool(
                extractor._output_projection_hidden_layer_norm
            ),
            "output_projection_final_layer_norm": bool(
                extractor._output_projection_final_layer_norm
            ),
            "output_projection_bias": bool(
                extractor._output_projection_bias
            ),
            "pool_token_init_std": float(extractor._pool_token_init_std),
            "positional_encoding_base": float(
                extractor._positional_encoding_base
            ),
            "environment_override_policy": str(
                extractor._environment_override_policy
            ),
        },
        "hash_backend": hash_backend,
        "effective_sentence_encoder_backend": str(
            extractor._effective_sentence_encoder_backend()
        ),
        "effective_sentence_pooling": str(extractor._effective_sentence_pooling()),
        "sentence_dimension": int(extractor._sentence_dim),
        "output_dimension": int(extractor.output_dim),
    }


def _capture_model_state(
    model: torch.nn.Module,
    store: _ArrayStore,
    prefix: str,
    *,
    kind: str,
    outcome_type: str,
    training_configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if kind not in {"nuisance", "effect"}:
        raise ValueError("unsupported HTR proof model kind")
    if (kind == "nuisance" and type(model) is not _NuisanceNet) or (
        kind == "effect" and type(model) is not _EffectNet
    ):
        raise TypeError("HTR proof received a model with the wrong native class")
    extractor = _extractor_descriptor(model.extractor)
    state_rows = []
    for index, (name, tensor) in enumerate(model.state_dict().items()):
        storage, dtype_name = _tensor_storage(tensor)
        key = store.add(f"{prefix}_state_{index:05d}", storage)
        state_rows.append(
            {
                "state_key": str(name),
                "array": key,
                "torch_dtype": dtype_name,
                "shape": [int(item) for item in tensor.shape],
            }
        )
    if not state_rows:
        raise ValueError("fitted HTR model has no tensor state")
    head_configuration = model.head_configuration()
    required_head_fields = {
        "hidden_dim",
        "depth",
        "activation",
        "dropout",
        "layer_norm",
        "bias",
    }
    if set(head_configuration) != required_head_fields:
        raise RuntimeError("HTR head did not expose its closed constructor")
    descriptor = {
        "kind": kind,
        "class_name": type(model).__name__,
        "head_configuration": head_configuration,
        "training_configuration": (
            None
            if training_configuration is None
            else json.loads(_canonical_json(dict(training_configuration)))
        ),
        "outcome_type": str(outcome_type),
        "extractor": extractor,
        "state_tensors": state_rows,
    }
    return {**descriptor, "state_sha256": _sha256_json(descriptor)}


def _capture_calibrator(
    calibrator: BinaryProbabilityCalibrator,
    store: _ArrayStore,
    prefix: str,
) -> dict[str, Any]:
    if type(calibrator) is not BinaryProbabilityCalibrator:
        raise TypeError("HTR proof accepts only BinaryProbabilityCalibrator")
    row: dict[str, Any] = {
        "class_name": "BinaryProbabilityCalibrator",
        "method": str(calibrator.method),
        "temperature": float(calibrator.temperature),
        "isotonic": calibrator.isotonic is not None,
    }
    if calibrator.isotonic is not None:
        x = np.asarray(calibrator.isotonic.X_thresholds_, dtype=np.float64)
        y = np.asarray(calibrator.isotonic.y_thresholds_, dtype=np.float64)
        if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
            raise ValueError("fitted HTR isotonic calibrator has invalid state")
        row["x_thresholds"] = store.add(f"{prefix}_isotonic_x", x)
        row["y_thresholds"] = store.add(f"{prefix}_isotonic_y", y)
    return row


def _apply_calibrator(
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    probability: Any,
) -> np.ndarray:
    method = str(descriptor.get("method") or "")
    if method not in {"none", "temperature", "isotonic", "temperature_isotonic"}:
        raise ValueError("captured HTR calibrator has an unsupported method")
    p = clip_probability(probability)
    if "temperature" in method:
        temperature = max(float(descriptor.get("temperature", 1.0)), 1e-3)
        logit = np.log(p / (1.0 - p))
        p = clip_probability(1.0 / (1.0 + np.exp(-np.clip(logit / temperature, -50, 50))))
    if descriptor.get("isotonic") is True:
        x = np.asarray(arrays[str(descriptor["x_thresholds"])], dtype=float)
        y = np.asarray(arrays[str(descriptor["y_thresholds"])], dtype=float)
        p = np.interp(p, x, y, left=float(y[0]), right=float(y[-1]))
    elif descriptor.get("isotonic") is not False:
        raise ValueError("captured HTR calibrator isotonic flag is invalid")
    return clip_probability(p)


def _finite_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (int(length),) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite vector of length {length}")
    return array


class NativeHTRProofCaptureSink:
    """Thread-safe sink for genuine nested HTR nuisance/effect fits."""

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
        treatment_column: str,
        outcome_column: str,
        outcome_type: str,
        e_clip: float,
        nuisance_folds: int,
        effect_folds: int,
        model_tree_sha256: str | None,
        prediction_batch_size: int,
        seed: int,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        if self.artifact_dir.exists():
            raise RuntimeError(f"HTR capture directory already exists: {self.artifact_dir}")
        self.scope_id = str(scope_id)
        self.outer_fold = int(outer_fold)
        self.inner_fold = int(inner_fold)
        if self.outer_fold < 1 or self.inner_fold < 1:
            raise ValueError("HTR proof capture requires an exact-inner scope")
        self.fit_row_ids = tuple(map(int, fit_row_ids))
        self.heldout_row_ids = tuple(map(int, heldout_row_ids))
        self.fit_texts = tuple(str(value) for value in fit_texts)
        self.heldout_texts = tuple(str(value) for value in heldout_texts)
        _row_fingerprint(self.fit_row_ids)
        _row_fingerprint(self.heldout_row_ids)
        if set(self.fit_row_ids) & set(self.heldout_row_ids):
            raise ValueError("HTR fit and heldout rows overlap")
        if len(self.fit_texts) != len(self.fit_row_ids) or len(self.heldout_texts) != len(
            self.heldout_row_ids
        ):
            raise ValueError("HTR text and row counts differ")
        self.text_column = str(text_column)
        self.treatment_column = str(treatment_column)
        self.outcome_column = str(outcome_column)
        if not self.treatment_column or not self.outcome_column:
            raise ValueError("HTR proof requires treatment and outcome column names")
        self.outcome_type = str(outcome_type)
        if self.outcome_type not in {"binary", "continuous"}:
            raise ValueError("HTR proof outcome type is unsupported")
        self.e_clip = float(e_clip)
        if not 0 < self.e_clip < 0.5:
            raise ValueError("HTR proof e_clip must be in (0, 0.5)")
        self.nuisance_folds = int(nuisance_folds)
        self.effect_folds = int(effect_folds)
        self.prediction_batch_size = max(1, int(prediction_batch_size))
        self.seed = int(seed)
        self.model_tree_sha256 = (
            None if model_tree_sha256 is None else str(model_tree_sha256)
        )
        if self.model_tree_sha256 is not None and _SHA256.fullmatch(
            self.model_tree_sha256
        ) is None:
            raise ValueError("HTR model tree digest is invalid")
        self._store = _ArrayStore()
        self._nuisance_folds: list[dict[str, Any]] = []
        self._effect_folds: list[dict[str, Any]] = []
        self._scope_outputs: dict[str, dict[str, Any]] = {}
        self._extractor_identity: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._finalized = False

    def _check_rows(
        self,
        df: Any,
        expected_rows: Sequence[int],
        expected_texts: Sequence[str],
        *,
        name: str,
    ) -> None:
        observed = tuple(map(int, df["_oci_row_id"].to_numpy().tolist()))
        if observed != tuple(map(int, expected_rows)):
            raise ValueError(f"HTR {name} row order changed")
        observed_texts = tuple(str(value) for value in df[self.text_column].tolist())
        if observed_texts != tuple(map(str, expected_texts)):
            raise ValueError(f"HTR {name} text projection changed")

    def _bind_extractor(self, descriptor: Mapping[str, Any]) -> None:
        identity = dict(descriptor)
        if identity.get("hash_backend") is False and self.model_tree_sha256 is None:
            raise ValueError("non-hash HTR capture requires an authenticated model tree")
        if self._extractor_identity is None:
            self._extractor_identity = json.loads(_canonical_json(identity))
        elif self._extractor_identity != identity:
            raise RuntimeError("HTR extractor configuration changed between folds")

    def record_nuisance_fold(
        self,
        *,
        model: torch.nn.Module,
        train_df: Any,
        test_df: Any,
        fit_pos: Sequence[int],
        validation_pos: Sequence[int],
        fold: int,
        fit_e_raw: Any,
        fit_m_raw: Any,
        validation_e_raw: Any,
        validation_m_raw: Any,
        validation_e_hat: Any,
        validation_m_hat: Any,
        heldout_e_raw: Any,
        heldout_m_raw: Any,
        heldout_e_hat: Any,
        heldout_m_hat: Any,
        propensity_calibrator: BinaryProbabilityCalibrator,
        outcome_calibrator: BinaryProbabilityCalibrator | None,
    ) -> None:
        if self._finalized:
            raise RuntimeError("HTR capture was already finalized")
        self._check_rows(train_df, self.fit_row_ids, self.fit_texts, name="fit")
        self._check_rows(
            test_df,
            self.heldout_row_ids,
            self.heldout_texts,
            name="heldout",
        )
        fit_pos = np.asarray(fit_pos, dtype=int)
        validation_pos = np.asarray(validation_pos, dtype=int)
        if (
            fit_pos.ndim != 1
            or validation_pos.ndim != 1
            or set(fit_pos.tolist()) & set(validation_pos.tolist())
            or sorted(np.concatenate([fit_pos, validation_pos]).tolist())
            != list(range(len(self.fit_row_ids)))
        ):
            raise ValueError("HTR nuisance fold is not a partition of the fit scope")
        fold = int(fold)
        prefix = f"nuisance_{fold:04d}"
        with self._lock:
            if any(int(row["fold"]) == fold for row in self._nuisance_folds):
                raise ValueError(f"duplicate HTR nuisance fold: {fold}")
            state = _capture_model_state(
                model,
                self._store,
                prefix,
                kind="nuisance",
                outcome_type=self.outcome_type,
            )
            self._bind_extractor(state["extractor"])
            n_fit = len(fit_pos)
            n_validation = len(validation_pos)
            n_heldout = len(self.heldout_row_ids)
            fit_t = train_df.iloc[fit_pos]
            validation = train_df.iloc[validation_pos]
            treatment_column = self.treatment_column
            outcome_column = self.outcome_column
            row = {
                "fold": fold,
                "objective": "joint_treatment_outcome_nuisance",
                "split_seed": 10_000 + self.outer_fold,
                "fit_positions": fit_pos.tolist(),
                "validation_positions": validation_pos.tolist(),
                "fit_row_ids": [self.fit_row_ids[int(pos)] for pos in fit_pos],
                "validation_row_ids": [
                    self.fit_row_ids[int(pos)] for pos in validation_pos
                ],
                "fit_row_fingerprint": _row_fingerprint(
                    [self.fit_row_ids[int(pos)] for pos in fit_pos]
                ),
                "validation_row_fingerprint": _row_fingerprint(
                    [self.fit_row_ids[int(pos)] for pos in validation_pos]
                ),
                "treatment_column": treatment_column,
                "outcome_column": outcome_column,
                "model": state,
                "propensity_calibrator": _capture_calibrator(
                    propensity_calibrator,
                    self._store,
                    f"{prefix}_propensity",
                ),
                "outcome_calibrator": (
                    None
                    if outcome_calibrator is None
                    else _capture_calibrator(
                        outcome_calibrator,
                        self._store,
                        f"{prefix}_outcome",
                    )
                ),
                "fit_treatment": self._store.add(
                    f"{prefix}_fit_treatment",
                    _finite_vector(
                        fit_t[treatment_column], name="fit treatment", length=n_fit
                    ),
                ),
                "fit_outcome": self._store.add(
                    f"{prefix}_fit_outcome",
                    _finite_vector(fit_t[outcome_column], name="fit outcome", length=n_fit),
                ),
                "validation_treatment": self._store.add(
                    f"{prefix}_validation_treatment",
                    _finite_vector(
                        validation[treatment_column],
                        name="validation treatment",
                        length=n_validation,
                    ),
                ),
                "validation_outcome": self._store.add(
                    f"{prefix}_validation_outcome",
                    _finite_vector(
                        validation[outcome_column],
                        name="validation outcome",
                        length=n_validation,
                    ),
                ),
                "fit_e_raw": self._store.add(
                    f"{prefix}_fit_e_raw",
                    _finite_vector(fit_e_raw, name="fit e raw", length=n_fit),
                ),
                "fit_m_raw": self._store.add(
                    f"{prefix}_fit_m_raw",
                    _finite_vector(fit_m_raw, name="fit m raw", length=n_fit),
                ),
                "validation_e_raw": self._store.add(
                    f"{prefix}_validation_e_raw",
                    _finite_vector(
                        validation_e_raw, name="validation e raw", length=n_validation
                    ),
                ),
                "validation_m_raw": self._store.add(
                    f"{prefix}_validation_m_raw",
                    _finite_vector(
                        validation_m_raw, name="validation m raw", length=n_validation
                    ),
                ),
                "validation_e_hat": self._store.add(
                    f"{prefix}_validation_e_hat",
                    _finite_vector(
                        validation_e_hat, name="validation e hat", length=n_validation
                    ),
                ),
                "validation_m_hat": self._store.add(
                    f"{prefix}_validation_m_hat",
                    _finite_vector(
                        validation_m_hat, name="validation m hat", length=n_validation
                    ),
                ),
                "heldout_e_raw": self._store.add(
                    f"{prefix}_heldout_e_raw",
                    _finite_vector(heldout_e_raw, name="heldout e raw", length=n_heldout),
                ),
                "heldout_m_raw": self._store.add(
                    f"{prefix}_heldout_m_raw",
                    _finite_vector(heldout_m_raw, name="heldout m raw", length=n_heldout),
                ),
                "heldout_e_hat": self._store.add(
                    f"{prefix}_heldout_e_hat",
                    _finite_vector(heldout_e_hat, name="heldout e hat", length=n_heldout),
                ),
                "heldout_m_hat": self._store.add(
                    f"{prefix}_heldout_m_hat",
                    _finite_vector(heldout_m_hat, name="heldout m hat", length=n_heldout),
                ),
                "heldout_labels_accessed": False,
            }
            self._nuisance_folds.append(row)

    def record_effect_fold(
        self,
        *,
        model: torch.nn.Module,
        train_df: Any,
        test_df: Any,
        fit_pos: Sequence[int],
        eligible_fit_pos: Sequence[int],
        validation_pos: Sequence[int],
        fold: int,
        effect_objective: str,
        treatment: Any,
        outcome: Any,
        e_hat: Any,
        m_hat: Any,
        validation_raw_effect: Any,
        validation_tau: Any,
        validation_r_loss: Any,
        validation_effect_loss: Any,
        heldout_raw_effect: Any,
        heldout_tau: Any,
        r_stage_min_propensity: float,
        r_stage_max_propensity: float,
    ) -> None:
        if self._finalized:
            raise RuntimeError("HTR capture was already finalized")
        objective = str(effect_objective)
        if objective not in _EFFECT_OBJECTIVES:
            raise ValueError("unsupported production HTR effect objective")
        self._check_rows(train_df, self.fit_row_ids, self.fit_texts, name="fit")
        self._check_rows(
            test_df,
            self.heldout_row_ids,
            self.heldout_texts,
            name="heldout",
        )
        n_scope = len(self.fit_row_ids)
        fit_pos = np.asarray(fit_pos, dtype=int)
        eligible_fit_pos = np.asarray(eligible_fit_pos, dtype=int)
        validation_pos = np.asarray(validation_pos, dtype=int)
        if (
            fit_pos.ndim != 1
            or eligible_fit_pos.ndim != 1
            or validation_pos.ndim != 1
            or not set(eligible_fit_pos.tolist()).issubset(set(fit_pos.tolist()))
            or set(fit_pos.tolist()) & set(validation_pos.tolist())
            or sorted(np.concatenate([fit_pos, validation_pos]).tolist())
            != list(range(n_scope))
        ):
            raise ValueError("HTR effect fold has invalid fit/eligible/validation rows")
        t = _finite_vector(treatment, name="effect treatment", length=n_scope)
        y = _finite_vector(outcome, name="effect outcome", length=n_scope)
        e = _finite_vector(e_hat, name="effect e_hat", length=n_scope)
        m = _finite_vector(m_hat, name="effect m_hat", length=n_scope)
        e_clipped = np.clip(e, self.e_clip, 1.0 - self.e_clip)
        y_residual = y - m
        t_residual = t - e_clipped
        pseudo = _r_pseudo_outcome(y_residual, t_residual)
        eligible = (
            np.isfinite(e)
            & (e >= float(r_stage_min_propensity))
            & (e <= float(r_stage_max_propensity))
        )
        if objective == "pseudo_outcome_mse":
            eligible &= np.isfinite(pseudo)
        expected_eligible = fit_pos[eligible[fit_pos]]
        if not np.array_equal(expected_eligible, eligible_fit_pos):
            raise ValueError("HTR effect eligible rows differ from the native objective")
        fold = int(fold)
        prefix = f"effect_{objective}_{fold:04d}"
        with self._lock:
            if any(
                row["effect_objective"] == objective and int(row["fold"]) == fold
                for row in self._effect_folds
            ):
                raise ValueError(f"duplicate HTR effect fold: {objective}/{fold}")
            state = _capture_model_state(
                model,
                self._store,
                prefix,
                kind="effect",
                outcome_type=self.outcome_type,
            )
            self._bind_extractor(state["extractor"])
            n_validation = len(validation_pos)
            n_heldout = len(self.heldout_row_ids)
            row = {
                "fold": fold,
                "objective": f"effect_{objective}",
                "effect_objective": objective,
                "split_seed": 20_000 + self.outer_fold,
                "fit_positions": fit_pos.tolist(),
                "eligible_fit_positions": eligible_fit_pos.tolist(),
                "validation_positions": validation_pos.tolist(),
                "fit_row_ids": [self.fit_row_ids[int(pos)] for pos in fit_pos],
                "eligible_fit_row_ids": [
                    self.fit_row_ids[int(pos)] for pos in eligible_fit_pos
                ],
                "validation_row_ids": [
                    self.fit_row_ids[int(pos)] for pos in validation_pos
                ],
                "fit_row_fingerprint": _row_fingerprint(
                    [self.fit_row_ids[int(pos)] for pos in fit_pos]
                ),
                "eligible_fit_row_fingerprint": _row_fingerprint(
                    [self.fit_row_ids[int(pos)] for pos in eligible_fit_pos]
                ),
                "validation_row_fingerprint": _row_fingerprint(
                    [self.fit_row_ids[int(pos)] for pos in validation_pos]
                ),
                "model": state,
                "r_stage_min_propensity": float(r_stage_min_propensity),
                "r_stage_max_propensity": float(r_stage_max_propensity),
                "treatment": self._store.add(f"{prefix}_treatment", t),
                "outcome": self._store.add(f"{prefix}_outcome", y),
                "e_hat": self._store.add(f"{prefix}_e_hat", e),
                "m_hat": self._store.add(f"{prefix}_m_hat", m),
                "e_clipped": self._store.add(f"{prefix}_e_clipped", e_clipped),
                "y_residual": self._store.add(f"{prefix}_y_residual", y_residual),
                "t_residual": self._store.add(f"{prefix}_t_residual", t_residual),
                "r_pseudo_outcome": self._store.add(
                    f"{prefix}_r_pseudo_outcome", pseudo
                ),
                "train_eligible": self._store.add(
                    f"{prefix}_train_eligible", eligible.astype(np.uint8)
                ),
                "validation_raw_effect": self._store.add(
                    f"{prefix}_validation_raw_effect",
                    _finite_vector(
                        validation_raw_effect,
                        name="validation raw effect",
                        length=n_validation,
                    ),
                ),
                "validation_tau": self._store.add(
                    f"{prefix}_validation_tau",
                    _finite_vector(
                        validation_tau, name="validation tau", length=n_validation
                    ),
                ),
                "validation_r_loss": self._store.add(
                    f"{prefix}_validation_r_loss",
                    _finite_vector(
                        validation_r_loss,
                        name="validation r loss",
                        length=n_validation,
                    ),
                ),
                "validation_effect_loss": self._store.add(
                    f"{prefix}_validation_effect_loss",
                    _finite_vector(
                        validation_effect_loss,
                        name="validation effect loss",
                        length=n_validation,
                    ),
                ),
                "heldout_raw_effect": self._store.add(
                    f"{prefix}_heldout_raw_effect",
                    _finite_vector(
                        heldout_raw_effect,
                        name="heldout raw effect",
                        length=n_heldout,
                    ),
                ),
                "heldout_tau": self._store.add(
                    f"{prefix}_heldout_tau",
                    _finite_vector(heldout_tau, name="heldout tau", length=n_heldout),
                ),
                "heldout_labels_accessed": False,
            }
            self._effect_folds.append(row)

    def record_scope_output(self, name: str, values: Any, *, role: str) -> None:
        if self._finalized:
            raise RuntimeError("HTR capture was already finalized")
        name = str(name)
        if _SAFE_KEY.fullmatch(name) is None:
            raise ValueError(f"invalid HTR scope output name: {name}")
        with self._lock:
            if name in self._scope_outputs:
                raise ValueError(f"duplicate HTR scope output: {name}")
            key = self._store.add(f"scope_{name}", np.asarray(values))
            self._scope_outputs[name] = {"role": str(role), "array": key}

    def finalize(self) -> Mapping[str, Any]:
        if self._finalized:
            raise RuntimeError("HTR capture was already finalized")
        nuisance = sorted(self._nuisance_folds, key=lambda row: int(row["fold"]))
        effects = sorted(
            self._effect_folds,
            key=lambda row: (str(row["effect_objective"]), int(row["fold"])),
        )
        expected_nuisance = list(range(1, self.nuisance_folds + 1))
        if [int(row["fold"]) for row in nuisance] != expected_nuisance:
            raise RuntimeError("HTR native capture lacks complete nuisance-fold coverage")
        for objective in _EFFECT_OBJECTIVES:
            observed = [
                int(row["fold"])
                for row in effects
                if row["effect_objective"] == objective
            ]
            if observed != list(range(1, self.effect_folds + 1)):
                raise RuntimeError(
                    f"HTR native capture lacks complete {objective} fold coverage"
                )
        required_outputs = {
            "htr_e_fit",
            "htr_m_fit",
            "htr_e_heldout",
            "htr_m_heldout",
            "effect_pseudo_outcome_mse_fit",
            "effect_pseudo_outcome_mse_heldout",
            "effect_squared_r_loss_fit",
            "effect_squared_r_loss_heldout",
        }
        if set(self._scope_outputs) != required_outputs:
            missing = sorted(required_outputs - set(self._scope_outputs))
            extra = sorted(set(self._scope_outputs) - required_outputs)
            raise RuntimeError(
                f"HTR native capture scope outputs changed; missing={missing}, extra={extra}"
            )
        if self._extractor_identity is None:
            raise RuntimeError("HTR native capture has no fitted extractor identity")
        body = {
            "schema_version": HTR_NATIVE_CAPTURE_SCHEMA,
            "scope_id": self.scope_id,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "fit_row_ids": list(self.fit_row_ids),
            "heldout_row_ids": list(self.heldout_row_ids),
            "fit_row_fingerprint": _row_fingerprint(self.fit_row_ids),
            "heldout_row_fingerprint": _row_fingerprint(self.heldout_row_ids),
            "fit_text_sha256": _text_sha256(self.fit_row_ids, self.fit_texts),
            "heldout_text_sha256": _text_sha256(
                self.heldout_row_ids, self.heldout_texts
            ),
            "text_column": self.text_column,
            "treatment_column": self.treatment_column,
            "outcome_column": self.outcome_column,
            "outcome_type": self.outcome_type,
            "e_clip": self.e_clip,
            "nuisance_folds": self.nuisance_folds,
            "effect_folds": self.effect_folds,
            "prediction_batch_size": self.prediction_batch_size,
            "seed": self.seed,
            "model_tree_sha256": self.model_tree_sha256,
            "extractor_identity": self._extractor_identity,
            "nuisance_fold_states": nuisance,
            "effect_fold_states": effects,
            "scope_outputs": self._scope_outputs,
            "array_inventory": self._store.inventory,
            "heldout_columns_read": ["_oci_row_id", self.text_column],
            "heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
            "secrets_accessed": False,
            "executable_checkpoint_retained": False,
            "pickle_or_joblib_loaded": False,
        }
        metadata = {**body, "content_sha256": _sha256_json(body)}
        self.artifact_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_npz(self.artifact_dir / "arrays.npz", self._store.arrays)
        closed_body = {
            **metadata,
            "arrays_file": "arrays.npz",
            "arrays_file_sha256": _sha256_file(self.artifact_dir / "arrays.npz"),
        }
        metadata = {
            **closed_body,
            "content_sha256": _sha256_json(
                {key: value for key, value in closed_body.items() if key != "content_sha256"}
            ),
        }
        _atomic_write_bytes(
            self.artifact_dir / "metadata.json",
            (_canonical_json(metadata) + "\n").encode("utf-8"),
        )
        self._finalized = True
        return json.loads(_canonical_json(metadata))


def _load_capture(path: Path | str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("HTR native capture must be one regular directory")
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if any(item.is_symlink() for item in root.rglob("*")):
        raise ValueError("HTR native capture cannot contain symlinks")
    if {item.relative_to(root).as_posix() for item in files} != {
        "arrays.npz",
        "metadata.json",
    }:
        raise ValueError("HTR native capture has an unexpected file inventory")
    if any(item.suffix.lower() in _FORBIDDEN_SUFFIXES for item in files):
        raise ValueError("HTR native capture contains an executable checkpoint")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("HTR native metadata is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("HTR native metadata must be one JSON object")
    body = {key: value for key, value in metadata.items() if key != "content_sha256"}
    if (
        metadata.get("schema_version") != HTR_NATIVE_CAPTURE_SCHEMA
        or metadata.get("content_sha256") != _sha256_json(body)
        or metadata.get("arrays_file") != "arrays.npz"
        or metadata.get("arrays_file_sha256") != _sha256_file(root / "arrays.npz")
        or metadata.get("heldout_labels_accessed") is not False
        or metadata.get("executable_checkpoint_retained") is not False
        or metadata.get("pickle_or_joblib_loaded") is not False
    ):
        raise ValueError("HTR native metadata has an invalid closed envelope")
    try:
        loaded = np.load(root / "arrays.npz", allow_pickle=False)
        arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
        loaded.close()
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError("HTR native numerical artifact is invalid") from exc
    inventory = metadata.get("array_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != set(arrays):
        raise ValueError("HTR native numerical inventory is not closed")
    for key, array in arrays.items():
        row = inventory.get(key)
        if (
            _SAFE_KEY.fullmatch(str(key)) is None
            or not isinstance(row, Mapping)
            or row.get("dtype") != array.dtype.str
            or row.get("shape") != list(array.shape)
            or row.get("content_sha256") != _array_sha256(array)
            or array.dtype.hasobject
        ):
            raise ValueError(f"HTR native numerical array changed: {key}")
    return metadata, arrays


def _build_model(
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    initialization_texts: Sequence[str],
    htr_model_path: Path | str | None,
    device: torch.device,
) -> torch.nn.Module:
    if descriptor.get("state_sha256") != _sha256_json(
        {key: value for key, value in descriptor.items() if key != "state_sha256"}
    ):
        raise ValueError("captured HTR state descriptor changed")
    extractor_row = descriptor.get("extractor")
    if not isinstance(extractor_row, Mapping):
        raise ValueError("captured HTR model lacks an extractor descriptor")
    constructor = dict(extractor_row.get("constructor") or {})
    marker = constructor.get("sentence_encoder_model")
    if marker == "authenticated_local_tree":
        if htr_model_path is None:
            raise ValueError("HTR replay requires the authenticated local model tree")
        constructor["sentence_encoder_model"] = str(Path(htr_model_path).resolve())
    elif marker != "hash":
        raise ValueError("captured HTR sentence-model marker is unsupported")
    constructor["device"] = device
    extractor = HierarchicalTransformerExtractor(**constructor)
    extractor.fit_tokenizer(list(map(str, initialization_texts)))
    observed_descriptor = _extractor_descriptor(extractor)
    expected_descriptor = json.loads(_canonical_json(extractor_row))
    if observed_descriptor != expected_descriptor:
        raise RuntimeError("reconstructed HTR extractor identity changed")
    kind = str(descriptor.get("kind") or "")
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
            "captured typed HTR model lacks its complete head constructor"
        )
    head_kwargs = {
        "hidden_dim": int(head_configuration["hidden_dim"]),
        "head_depth": int(head_configuration["depth"]),
        "head_activation": str(head_configuration["activation"]),
        "head_dropout": float(head_configuration["dropout"]),
        "head_layer_norm": head_configuration["layer_norm"],
        "head_bias": head_configuration["bias"],
    }
    if kind == "nuisance":
        model = _NuisanceNet(
            extractor=extractor,
            outcome_type=str(descriptor.get("outcome_type") or ""),
            **head_kwargs,
        )
    elif kind == "effect":
        model = _EffectNet(extractor=extractor, **head_kwargs)
    else:
        raise ValueError("captured HTR model kind is unsupported")
    model = model.to(device)
    state_rows = descriptor.get("state_tensors")
    if not isinstance(state_rows, list) or not state_rows:
        raise ValueError("captured HTR model has no tensor state")
    state: dict[str, torch.Tensor] = {}
    for row in state_rows:
        if not isinstance(row, Mapping):
            raise ValueError("captured HTR tensor descriptor is malformed")
        name = str(row.get("state_key") or "")
        key = str(row.get("array") or "")
        if not name or name in state or key not in arrays:
            raise ValueError("captured HTR tensor descriptor is incomplete")
        tensor = _restore_tensor(arrays[key], str(row.get("torch_dtype") or ""))
        if list(tensor.shape) != row.get("shape"):
            raise ValueError("captured HTR tensor shape changed")
        state[name] = tensor
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("captured HTR tensor state does not fit its native model") from exc
    model.eval()
    return model


def _predict_model(
    model: torch.nn.Module,
    texts: Sequence[str],
    *,
    kind: str,
    outcome_type: str,
    batch_size: int,
) -> tuple[np.ndarray, ...]:
    outputs: list[list[np.ndarray]] = [[], []] if kind == "nuisance" else [[]]
    values = tuple(map(str, texts))
    with torch.no_grad():
        for start in range(0, len(values), max(1, int(batch_size))):
            batch = values[start : start + max(1, int(batch_size))]
            if kind == "nuisance":
                t_logit, y_raw = model(list(batch))
                outputs[0].append(torch.sigmoid(t_logit).detach().cpu().numpy())
                outputs[1].append(
                    (
                        y_raw
                        if outcome_type == "continuous"
                        else torch.sigmoid(y_raw)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                outputs[0].append(model(list(batch)).detach().cpu().numpy())
    return tuple(
        np.concatenate(parts) if parts else np.zeros(0, dtype=float) for parts in outputs
    )


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
        raise RuntimeError(f"HTR native replay changed {name}")


def validate_htr_native_capture(
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
    expected_model_tree_sha256: str | None = None,
    device: torch.device | str = "cpu",
) -> Mapping[str, Any]:
    """Validate and exactly replay one non-executable HTR native capture."""

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
        or metadata.get("heldout_text_sha256") != _text_sha256(
            heldout_rows, heldout_texts
        )
    ):
        raise ValueError("HTR native capture changed its exact row/text scope")
    extractor_identity = metadata.get("extractor_identity")
    if not isinstance(extractor_identity, Mapping):
        raise ValueError("HTR native capture has no extractor identity")
    hash_backend = extractor_identity.get("hash_backend") is True
    declared_tree = metadata.get("model_tree_sha256")
    if hash_backend:
        if declared_tree is not None:
            raise ValueError("hash HTR capture cannot claim a private model tree")
    else:
        expected_tree = str(expected_model_tree_sha256 or "")
        if _SHA256.fullmatch(expected_tree) is None or declared_tree != expected_tree:
            raise ValueError("HTR native capture has the wrong model-tree binding")
        if htr_model_path is None or directory_tree_sha256(htr_model_path) != expected_tree:
            raise RuntimeError("authenticated HTR model tree changed before replay")
    nuisance_rows = metadata.get("nuisance_fold_states")
    effect_rows = metadata.get("effect_fold_states")
    if not isinstance(nuisance_rows, list) or not isinstance(effect_rows, list):
        raise ValueError("HTR native capture lacks fold state")
    nuisance_folds = int(metadata.get("nuisance_folds", 0))
    effect_folds = int(metadata.get("effect_folds", 0))
    if [int(row.get("fold", 0)) for row in nuisance_rows] != list(
        range(1, nuisance_folds + 1)
    ):
        raise ValueError("HTR native capture lacks exact nuisance-fold coverage")
    device = torch.device(device)
    outcome_type = str(metadata.get("outcome_type") or "")
    batch_size = int(metadata.get("prediction_batch_size", 0))
    nuisance_oof_e = np.full(len(fit_rows), np.nan, dtype=float)
    nuisance_oof_m = np.full(len(fit_rows), np.nan, dtype=float)
    nuisance_heldout_e: list[np.ndarray] = []
    nuisance_heldout_m: list[np.ndarray] = []
    expected_splits = list(
        KFold(
            n_splits=nuisance_folds,
            shuffle=True,
            random_state=10_000 + int(metadata["outer_fold"]),
        ).split(np.arange(len(fit_rows)))
    )
    if (expected_fit_treatment is None) != (expected_fit_outcome is None):
        raise ValueError(
            "HTR canonical fit treatment/outcome must be supplied together"
        )
    canonical_treatment = (
        None
        if expected_fit_treatment is None
        else _finite_vector(
            expected_fit_treatment,
            name="canonical fit treatment",
            length=len(fit_rows),
        )
    )
    canonical_outcome = (
        None
        if expected_fit_outcome is None
        else _finite_vector(
            expected_fit_outcome,
            name="canonical fit outcome",
            length=len(fit_rows),
        )
    )
    for row, (expected_fit, expected_validation) in zip(
        nuisance_rows, expected_splits
    ):
        fit_pos = np.asarray(row.get("fit_positions"), dtype=int)
        validation_pos = np.asarray(row.get("validation_positions"), dtype=int)
        if not np.array_equal(fit_pos, expected_fit) or not np.array_equal(
            validation_pos, expected_validation
        ):
            raise ValueError("HTR nuisance fold split changed")
        if (
            row.get("objective") != "joint_treatment_outcome_nuisance"
            or row.get("heldout_labels_accessed") is not False
            or row.get("fit_row_ids") != [fit_rows[int(pos)] for pos in fit_pos]
            or row.get("validation_row_ids")
            != [fit_rows[int(pos)] for pos in validation_pos]
        ):
            raise ValueError("HTR nuisance fold identity changed")
        if canonical_treatment is not None:
            captured_fit_treatment = np.asarray(
                arrays[str(row["fit_treatment"])], dtype=float
            )
            captured_fit_outcome = np.asarray(arrays[str(row["fit_outcome"])], dtype=float)
            captured_validation_treatment = np.asarray(
                arrays[str(row["validation_treatment"])], dtype=float
            )
            captured_validation_outcome = np.asarray(
                arrays[str(row["validation_outcome"])], dtype=float
            )
            if not np.array_equal(
                captured_fit_treatment, canonical_treatment[fit_pos]
            ) or not np.array_equal(
                captured_validation_treatment, canonical_treatment[validation_pos]
            ):
                raise ValueError(
                    "HTR nuisance capture treatment differs from canonical fit labels"
                )
            if not np.array_equal(
                captured_fit_outcome, canonical_outcome[fit_pos]
            ) or not np.array_equal(
                captured_validation_outcome, canonical_outcome[validation_pos]
            ):
                raise ValueError(
                    "HTR nuisance capture outcome differs from canonical fit labels"
                )
        model = _build_model(
            row["model"],
            arrays,
            initialization_texts=[fit_texts[int(pos)] for pos in fit_pos],
            htr_model_path=htr_model_path,
            device=device,
        )
        try:
            fit_raw = _predict_model(
                model,
                [fit_texts[int(pos)] for pos in fit_pos],
                kind="nuisance",
                outcome_type=outcome_type,
                batch_size=batch_size,
            )
            validation_raw = _predict_model(
                model,
                [fit_texts[int(pos)] for pos in validation_pos],
                kind="nuisance",
                outcome_type=outcome_type,
                batch_size=batch_size,
            )
            heldout_raw = _predict_model(
                model,
                heldout_texts,
                kind="nuisance",
                outcome_type=outcome_type,
                batch_size=batch_size,
            )
        finally:
            del model
        for observed, key, label in (
            (fit_raw[0], row["fit_e_raw"], "fit propensity"),
            (fit_raw[1], row["fit_m_raw"], "fit outcome"),
            (validation_raw[0], row["validation_e_raw"], "validation propensity"),
            (validation_raw[1], row["validation_m_raw"], "validation outcome"),
            (heldout_raw[0], row["heldout_e_raw"], "heldout propensity"),
            (heldout_raw[1], row["heldout_m_raw"], "heldout outcome"),
        ):
            _assert_close(observed, arrays[str(key)], name=label)
        validation_e = _apply_calibrator(
            row["propensity_calibrator"], arrays, validation_raw[0]
        )
        heldout_e = _apply_calibrator(
            row["propensity_calibrator"], arrays, heldout_raw[0]
        )
        if outcome_type == "continuous":
            if row.get("outcome_calibrator") is not None:
                raise ValueError("continuous HTR nuisance unexpectedly has outcome calibration")
            validation_m = validation_raw[1]
            heldout_m = heldout_raw[1]
        else:
            if not isinstance(row.get("outcome_calibrator"), Mapping):
                raise ValueError("binary HTR nuisance lacks outcome calibration")
            validation_m = _apply_calibrator(
                row["outcome_calibrator"], arrays, validation_raw[1]
            )
            heldout_m = _apply_calibrator(
                row["outcome_calibrator"], arrays, heldout_raw[1]
            )
        _assert_close(validation_e, arrays[str(row["validation_e_hat"])], name="e_hat")
        _assert_close(validation_m, arrays[str(row["validation_m_hat"])], name="m_hat")
        _assert_close(heldout_e, arrays[str(row["heldout_e_hat"])], name="heldout e_hat")
        _assert_close(heldout_m, arrays[str(row["heldout_m_hat"])], name="heldout m_hat")
        nuisance_oof_e[validation_pos] = validation_e
        nuisance_oof_m[validation_pos] = validation_m
        nuisance_heldout_e.append(heldout_e)
        nuisance_heldout_m.append(heldout_m)

    effect_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for objective in _EFFECT_OBJECTIVES:
        rows = [row for row in effect_rows if row.get("effect_objective") == objective]
        if [int(row.get("fold", 0)) for row in rows] != list(
            range(1, effect_folds + 1)
        ):
            raise ValueError(f"HTR native capture lacks {objective} fold coverage")
        expected_splits = list(
            KFold(
                n_splits=effect_folds,
                shuffle=True,
                random_state=20_000 + int(metadata["outer_fold"]),
            ).split(np.arange(len(fit_rows)))
        )
        oof = np.full(len(fit_rows), np.nan, dtype=float)
        heldout_predictions: list[np.ndarray] = []
        reference_inputs = None
        for row, (expected_fit, expected_validation) in zip(rows, expected_splits):
            fit_pos = np.asarray(row.get("fit_positions"), dtype=int)
            eligible_fit_pos = np.asarray(row.get("eligible_fit_positions"), dtype=int)
            validation_pos = np.asarray(row.get("validation_positions"), dtype=int)
            if (
                row.get("objective") != f"effect_{objective}"
                or row.get("heldout_labels_accessed") is not False
                or not np.array_equal(fit_pos, expected_fit)
                or not np.array_equal(validation_pos, expected_validation)
            ):
                raise ValueError(f"HTR {objective} fold identity changed")
            t = np.asarray(arrays[str(row["treatment"])], dtype=float)
            y = np.asarray(arrays[str(row["outcome"])], dtype=float)
            if canonical_treatment is not None and not np.array_equal(
                t, canonical_treatment
            ):
                raise ValueError(
                    "HTR effect capture treatment differs from canonical fit labels"
                )
            if canonical_outcome is not None and not np.array_equal(y, canonical_outcome):
                raise ValueError(
                    "HTR effect capture outcome differs from canonical fit labels"
                )
            e = np.asarray(arrays[str(row["e_hat"])], dtype=float)
            m = np.asarray(arrays[str(row["m_hat"])], dtype=float)
            e_clipped = np.clip(e, float(metadata["e_clip"]), 1.0 - float(metadata["e_clip"]))
            y_residual = y - m
            t_residual = t - e_clipped
            pseudo = _r_pseudo_outcome(y_residual, t_residual)
            eligible = (
                np.isfinite(e)
                & (e >= float(row["r_stage_min_propensity"]))
                & (e <= float(row["r_stage_max_propensity"]))
            )
            if objective == "pseudo_outcome_mse":
                eligible &= np.isfinite(pseudo)
            if not np.array_equal(eligible_fit_pos, fit_pos[eligible[fit_pos]]):
                raise ValueError(f"HTR {objective} eligible rows changed")
            for observed, key, name in (
                (e_clipped, row["e_clipped"], "e_clipped"),
                (y_residual, row["y_residual"], "y_residual"),
                (t_residual, row["t_residual"], "t_residual"),
                (pseudo, row["r_pseudo_outcome"], "r_pseudo_outcome"),
                (eligible.astype(np.uint8), row["train_eligible"], "train_eligible"),
            ):
                _assert_close(observed, arrays[str(key)], name=f"{objective} {name}")
            inputs_hash = _sha256_json(
                {
                    "t": _array_sha256(t),
                    "y": _array_sha256(y),
                    "e": _array_sha256(e),
                    "m": _array_sha256(m),
                }
            )
            if reference_inputs is None:
                reference_inputs = inputs_hash
            elif reference_inputs != inputs_hash:
                raise ValueError(f"HTR {objective} objective inputs changed between folds")
            model = _build_model(
                row["model"],
                arrays,
                initialization_texts=[fit_texts[int(pos)] for pos in eligible_fit_pos],
                htr_model_path=htr_model_path,
                device=device,
            )
            try:
                [validation_raw] = _predict_model(
                    model,
                    [fit_texts[int(pos)] for pos in validation_pos],
                    kind="effect",
                    outcome_type=outcome_type,
                    batch_size=batch_size,
                )
                [heldout_raw] = _predict_model(
                    model,
                    heldout_texts,
                    kind="effect",
                    outcome_type=outcome_type,
                    batch_size=batch_size,
                )
            finally:
                del model
            _assert_close(
                validation_raw,
                arrays[str(row["validation_raw_effect"])],
                name=f"{objective} validation raw effect",
            )
            _assert_close(
                heldout_raw,
                arrays[str(row["heldout_raw_effect"])],
                name=f"{objective} heldout raw effect",
            )
            if objective == "logistic_r_loss":
                validation_tau = _logistic_r_tau_from_delta(
                    validation_raw,
                    e_clipped[validation_pos],
                    clip_probability(m[validation_pos]),
                    e_clip=float(metadata["e_clip"]),
                )
                validation_effect_loss = _binary_log_loss_from_logits(
                    _logistic_r_logits(
                        validation_raw,
                        t[validation_pos],
                        e_clipped[validation_pos],
                        clip_probability(m[validation_pos]),
                        e_clip=float(metadata["e_clip"]),
                    ),
                    y[validation_pos],
                )
            else:
                validation_tau = validation_raw
                validation_effect_loss = (
                    (validation_tau - pseudo[validation_pos]) ** 2
                    if objective == "pseudo_outcome_mse"
                    else (
                        y_residual[validation_pos]
                        - validation_tau * t_residual[validation_pos]
                    )
                    ** 2
                )
            validation_r_loss = (
                y_residual[validation_pos]
                - validation_tau * t_residual[validation_pos]
            ) ** 2
            for observed, key, name in (
                (validation_tau, row["validation_tau"], "validation tau"),
                (validation_r_loss, row["validation_r_loss"], "validation r loss"),
                (
                    validation_effect_loss,
                    row["validation_effect_loss"],
                    "validation effect loss",
                ),
                (heldout_raw, row["heldout_tau"], "heldout tau"),
            ):
                _assert_close(observed, arrays[str(key)], name=f"{objective} {name}")
            oof[validation_pos] = validation_tau
            heldout_predictions.append(heldout_raw)
        effect_outputs[objective] = (
            oof,
            np.nanmean(np.vstack(heldout_predictions), axis=0),
        )

    scope_outputs = metadata.get("scope_outputs")
    if not isinstance(scope_outputs, Mapping):
        raise ValueError("HTR native capture lacks final scope outputs")

    def scope(name: str) -> np.ndarray:
        row = scope_outputs.get(name)
        if not isinstance(row, Mapping) or str(row.get("array") or "") not in arrays:
            raise ValueError(f"HTR native capture lacks scope output: {name}")
        return arrays[str(row["array"])]

    expected_scope = {
        "htr_e_fit": nuisance_oof_e,
        "htr_m_fit": nuisance_oof_m,
        "htr_e_heldout": np.nanmean(np.vstack(nuisance_heldout_e), axis=0),
        "htr_m_heldout": np.nanmean(np.vstack(nuisance_heldout_m), axis=0),
        "effect_pseudo_outcome_mse_fit": effect_outputs["pseudo_outcome_mse"][0],
        "effect_pseudo_outcome_mse_heldout": effect_outputs["pseudo_outcome_mse"][1],
        "effect_squared_r_loss_fit": effect_outputs["squared_r_loss"][0],
        "effect_squared_r_loss_heldout": effect_outputs["squared_r_loss"][1],
    }
    if set(scope_outputs) != set(expected_scope):
        raise ValueError("HTR native final-output coverage changed")
    for name, expected in expected_scope.items():
        _assert_close(scope(name), expected, name=f"scope output {name}")
    return json.loads(_canonical_json(metadata))


__all__ = [
    "HTR_NATIVE_CAPTURE_SCHEMA",
    "NativeHTRProofCaptureSink",
    "directory_tree_sha256",
    "validate_htr_native_capture",
]
