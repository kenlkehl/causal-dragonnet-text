"""Two-phase role-neutral matched-patient uplift execution.

This module is an isolated physical-fit/logical-view implementation for the
matched-patient Stage 1 family.  It deliberately is not wired into the legacy
all-ten-family worker yet.  One physical owner is fitted from canonical fit
rows, both native subproducers (BoW and HTR) are authenticated, and a fit-only
family seal plus any cumulative-review references are published before an
exact-inner text loader can run.

The exact loader returns a closed object with row IDs, complete text, and the
two already-authenticated nuisance probability banks needed for matching.  It
has no treatment or outcome field.  Text is never sliced in this module.  HTR
chunk capacity is configured explicitly and binding capacity aborts before a
model or transform can silently omit words.

Numerical state is stored as one mmap-safe ``.npy`` file per array.  Manifests
and ordered indexes are canonical JSON.  Pickle, joblib, Torch checkpoints,
and NPZ are neither written nor loaded.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import expit
from sklearn.model_selection import KFold

from ..config import BoWViewConfig
from ..models.hierarchical_transformer_extractor import (
    HierarchicalTransformerExtractor,
)
from .all_evidence_discovery_interfaces import MATCHED_PAIR_UPLIFT
from .htr_native_proof_capture import (
    _extractor_descriptor,
    directory_tree_sha256,
)
from .matched_pair_native_proof_capture import (
    _ArrayStore,
    _array_sha256,
    _build_htr_pair_model,
    _capture_htr_pair_model,
    _capture_offset_model,
    _pair_fingerprint,
    _predict_htr_pair,
    _predict_offset_model,
)
from .multi_model_forest_stage1 import _bow_view_to_dict, _vectorizer_params
from .multi_model_pair_uplift import (
    HTRPairUpliftNet,
    OffsetLogitBoWPairModel,
    aggregate_pair_predictions,
    build_candidate_pairs,
    build_training_pairs,
)
from .neural_numerical_replay import (
    neural_float_arrays_within_tolerance,
    validate_neural_replay_settings,
)
from .production_stage1_scope_scheduler import (
    Stage1ScopePlan,
    Stage1ScopeSpec,
    _enforce_stage1_torch_determinism,
)
from .role_neutral_bow_group_execution import (
    AuthenticatedRoleNeutralBoWNuisanceBank,
)
from .role_neutral_htr_group_execution import _execute_htr_fold_tasks
from .stage1_htr_operational_controls import (
    RoleNeutralHTRFoldResourcePlan,
)


ROLE_NEUTRAL_MATCHED_PAIR_GROUP_REQUEST_SCHEMA = (
    "production_role_neutral_matched_pair_physical_group_request_v2"
)
ROLE_NEUTRAL_MATCHED_PAIR_FIT_STATE_SCHEMA = (
    "production_role_neutral_matched_pair_fit_state_v2"
)
ROLE_NEUTRAL_MATCHED_PAIR_FIT_SEAL_SCHEMA = (
    "production_role_neutral_matched_pair_fit_only_family_seal_v2"
)
ROLE_NEUTRAL_MATCHED_PAIR_LOGICAL_VIEW_SCHEMA = (
    "production_role_neutral_matched_pair_logical_view_v2"
)
ROLE_NEUTRAL_MATCHED_PAIR_GROUP_EXECUTION_SCHEMA = (
    "production_role_neutral_matched_pair_group_execution_v2"
)
ROLE_NEUTRAL_MATCHED_PAIR_EXACT_INPUT_SCHEMA = (
    "production_role_neutral_matched_pair_exact_transform_input_v1"
)
ROLE_NEUTRAL_MATCHED_PAIR_CONFIG_SCHEMA = (
    "production_role_neutral_matched_pair_config_v4"
)
ROLE_NEUTRAL_MATCHED_PAIR_OPERATIONAL_ATTESTATION_SCHEMA = (
    "production_role_neutral_matched_pair_operational_attestation_v1"
)

_SUBPRODUCERS = ("bow", "htr")
_FIT_STATE_DIRECTORY = "fit_state"
_FIT_METADATA = "metadata.json"
_FIT_SEAL = "fit_only_family_seal.json"
_LOGICAL_DIRECTORY = "logical_views"
_TERMINAL = "execution_manifest.json"
_HEX = frozenset("0123456789abcdef")
_SAFE_RUNTIME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_WORD = re.compile(r"\S+")
_SUPPORTED_ACTIVATIONS = frozenset(
    {"gelu_exact", "gelu_tanh", "relu", "silu", "tanh"}
)
_EXTRACTOR_KEYS = frozenset(
    {
        "sentence_encoder_model",
        "freeze_sentence_encoder",
        "chunk_size_words",
        "chunk_overlap_words",
        "max_chunks",
        "max_chunk_length",
        "num_transformer_layers",
        "num_attention_heads",
        "transformer_dim",
        "transformer_dropout",
        "projection_dim",
        "hash_embedding_dim",
        "sentence_encoder_batch_size",
        "sentence_encoder_backend",
        "sentence_pooling",
        "normalize_sentence_embeddings",
        "trainable_sentence_encoder_layers",
        "role_attention",
        "w_attention_heads",
        "x_attention_heads",
        "transformer_feedforward_dim",
        "transformer_activation",
        "transformer_norm_style",
        "transformer_layer_norm_eps",
        "transformer_layer_norm_elementwise_affine",
        "transformer_layer_norm_bias",
        "transformer_attention_dropout",
        "transformer_residual_dropout",
        "transformer_feedforward_dropout",
        "transformer_attention_bias",
        "transformer_feedforward_bias",
        "output_projection_depth",
        "output_projection_hidden_dim",
        "output_projection_activation",
        "output_projection_dropout",
        "output_projection_hidden_layer_norm",
        "output_projection_final_layer_norm",
        "output_projection_bias",
        "pool_token_init_std",
        "positional_encoding_base",
        "environment_override_policy",
    }
)


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


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _row_order_fingerprint(row_ids: Sequence[int]) -> str:
    rows = tuple(map(int, row_ids))
    if not rows or len(rows) != len(set(rows)) or any(row < 0 for row in rows):
        raise ValueError("matched-pair row IDs must be unique non-negative integers")
    return _sha256_json(list(rows))


def _text_sha256(row_ids: Sequence[int], texts: Sequence[str]) -> str:
    rows = tuple(map(int, row_ids))
    values = tuple(texts)
    if len(rows) != len(values) or any(not isinstance(text, str) for text in values):
        raise ValueError("matched-pair text binding must align strings to row IDs")
    digest = hashlib.sha256()
    digest.update(b"production-role-neutral-matched-pair-text-v1\0")
    for row_id, text in zip(rows, values, strict=True):
        encoded = text.encode("utf-8")
        digest.update(int(row_id).to_bytes(8, "little", signed=False))
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _float_hex_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return _sha256_json([float(value).hex() for value in array])


def _stable_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _sha256_file(path: Path) -> tuple[str, int]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"artifact is not one regular file: {target}")
    before = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise ValueError(f"artifact file is linked or nonregular: {target}")
    digest = hashlib.sha256()
    size = 0
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    after = target.stat(follow_symlinks=False)
    if _stable_stat(before) != _stable_stat(after) or size != int(after.st_size):
        raise RuntimeError(f"artifact changed while hashing: {target}")
    return digest.hexdigest(), size


def _tree_sha256(root: Path) -> str:
    tree = Path(root)
    if tree.is_symlink() or not tree.is_dir():
        raise ValueError("matched-pair fit tree must be one real directory")
    inventory: list[dict[str, Any]] = []
    for child in sorted(
        tree.rglob("*"),
        key=lambda item: item.relative_to(tree).as_posix(),
    ):
        relative = child.relative_to(tree).as_posix()
        if child.is_symlink():
            raise ValueError("matched-pair fit tree cannot contain symlinks")
        if child.is_dir():
            inventory.append({"path": relative, "kind": "directory"})
        else:
            digest, size = _sha256_file(child)
            inventory.append(
                {
                    "path": relative,
                    "kind": "file",
                    "sha256": digest,
                    "size_bytes": size,
                }
            )
    if not inventory:
        raise ValueError("matched-pair fit tree is empty")
    return _sha256_json(
        {
            "schema_version": "production_role_neutral_matched_pair_tree_v1",
            "inventory": inventory,
        }
    )


def _write_new_bytes(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace immutable artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_bytes(
        path,
        (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_new_npy(path: Path, value: np.ndarray) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace immutable array: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("matched-pair arrays cannot use object dtype")
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        suffix=".npy",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _sha256_file(path)
    try:
        value = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _finite_probability_vector(
    values: Sequence[Any],
    *,
    label: str,
    length: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.shape != (int(length),)
        or not np.isfinite(array).all()
        or np.any((array < 0.0) | (array > 1.0))
    ):
        raise ValueError(f"{label} must be one finite probability vector")
    return array


def _binary_vector(
    values: Sequence[Any],
    *,
    label: str,
    length: int,
    require_both: bool,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (int(length),) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be one finite fit-row vector")
    unique = set(np.unique(array))
    if not unique.issubset({0.0, 1.0}) or (require_both and unique != {0.0, 1.0}):
        raise ValueError(f"{label} must be binary")
    return array


def _derived_seed(group_seed: int, *, purpose: str, fold: int, view: str) -> int:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": "production_role_neutral_matched_pair_seed_v1",
                "canonical_group_seed": int(group_seed),
                "purpose": str(purpose),
                "fold": int(fold),
                "view": str(view),
            }
        ).encode("utf-8")
    ).digest()
    result = int.from_bytes(digest[:8], "big") % (2**31 - 1)
    return result or 1


def _closed_extractor_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _EXTRACTOR_KEYS:
        raise ValueError(
            "matched-pair HTR extractor configuration must provide the complete "
            "closed constructor contract"
        )
    result = json.loads(_canonical_json(dict(value)))
    if result["sentence_encoder_model"] not in {"hash", "authenticated_local_tree"}:
        raise ValueError("HTR sentence encoder must be hash or an authenticated local tree")
    boolean_keys = {
        "freeze_sentence_encoder",
        "normalize_sentence_embeddings",
        "role_attention",
        "transformer_layer_norm_elementwise_affine",
        "transformer_layer_norm_bias",
        "transformer_attention_bias",
        "transformer_feedforward_bias",
        "output_projection_hidden_layer_norm",
        "output_projection_final_layer_norm",
        "output_projection_bias",
    }
    if any(type(result[key]) is not bool for key in boolean_keys):
        raise TypeError("HTR extractor Boolean configuration is not closed")
    positive_integer_keys = {
        "chunk_size_words",
        "max_chunks",
        "max_chunk_length",
        "num_transformer_layers",
        "num_attention_heads",
        "transformer_dim",
        "projection_dim",
        "hash_embedding_dim",
        "sentence_encoder_batch_size",
        "w_attention_heads",
        "x_attention_heads",
        "transformer_feedforward_dim",
        "output_projection_hidden_dim",
    }
    if any(
        isinstance(result[key], bool)
        or not isinstance(result[key], int)
        or int(result[key]) < 1
        for key in positive_integer_keys
    ):
        raise ValueError("HTR extractor positive integer configuration is invalid")
    if (
        isinstance(result["chunk_overlap_words"], bool)
        or not isinstance(result["chunk_overlap_words"], int)
        or not 0 <= int(result["chunk_overlap_words"]) < int(result["chunk_size_words"])
    ):
        raise ValueError("HTR chunk overlap configuration is invalid")
    if (
        isinstance(result["trainable_sentence_encoder_layers"], bool)
        or not isinstance(result["trainable_sentence_encoder_layers"], int)
        or int(result["trainable_sentence_encoder_layers"]) < 0
    ):
        raise ValueError("HTR trainable encoder-layer count is invalid")
    if (
        isinstance(result["output_projection_depth"], bool)
        or not isinstance(result["output_projection_depth"], int)
        or int(result["output_projection_depth"]) < 0
    ):
        raise ValueError("HTR output projection depth is invalid")
    for key in (
        "transformer_dropout",
        "transformer_attention_dropout",
        "transformer_residual_dropout",
        "transformer_feedforward_dropout",
        "output_projection_dropout",
    ):
        dropout = float(result[key])
        if not np.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError(f"HTR {key} is invalid")
    for key in ("transformer_activation", "output_projection_activation"):
        if result[key] not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(f"HTR {key} is unsupported")
    if result["transformer_norm_style"] not in {"pre_norm", "post_norm"}:
        raise ValueError("HTR transformer norm style is unsupported")
    if (
        not np.isfinite(float(result["transformer_layer_norm_eps"]))
        or float(result["transformer_layer_norm_eps"]) <= 0.0
        or not np.isfinite(float(result["pool_token_init_std"]))
        or float(result["pool_token_init_std"]) < 0.0
        or not np.isfinite(float(result["positional_encoding_base"]))
        or float(result["positional_encoding_base"]) <= 1.0
    ):
        raise ValueError("HTR norm/initialization geometry is invalid")
    if result["environment_override_policy"] != "forbid":
        raise ValueError(
            "typed matched-pair HTR requires environment_override_policy=forbid"
        )
    if not str(result["sentence_encoder_backend"]).strip() or not str(
        result["sentence_pooling"]
    ).strip():
        raise ValueError("HTR encoder backend/pooling configuration is empty")
    if (
        result["sentence_encoder_model"] == "authenticated_local_tree"
        and result["sentence_pooling"] != "token_attention"
    ):
        raise ValueError(
            "authenticated matched-pair HTR requires learned "
            "token_attention pooling"
        )
    return result


@dataclass(frozen=True)
class RoleNeutralMatchedPairConfig:
    """All matched-pair scientific settings used by this physical fit.

    There are intentionally no field defaults.  Deployment-specific text
    capacity and training settings must be supplied by a typed scientific
    specification.
    """

    effect_folds: int
    propensity_caliper: float
    outcome_caliper: float
    max_controls_per_candidate: int
    nearest_fallback_controls: int
    bow_l2_alpha: float
    bow_max_iter: int
    bow_optimizer_method: str
    bow_optimizer_ftol: float
    bow_optimizer_gtol: float
    bow_optimizer_maxls: int
    bow_optimizer_maxcor: int
    bow_optimizer_maxfun: int
    bow_optimizer_tol: float | None
    bow_optimizer_initialization: str
    bow_require_optimizer_success: bool
    htr_epochs: int
    htr_batch_size: int
    htr_learning_rate: float
    htr_weight_decay: float
    htr_optimizer_name: str
    htr_adamw_beta1: float
    htr_adamw_beta2: float
    htr_adamw_eps: float
    htr_adamw_amsgrad: bool
    htr_adamw_maximize: bool
    htr_adamw_foreach: bool
    htr_adamw_capturable: bool
    htr_adamw_differentiable: bool
    htr_adamw_fused: bool
    htr_optimizer_zero_grad_set_to_none: bool
    htr_gradient_clip_norm: float
    htr_gradient_clip_norm_type: float
    htr_gradient_clip_error_if_nonfinite: bool
    htr_gradient_clip_foreach: bool
    htr_hidden_dim: int
    htr_dropout: float
    htr_head_depth: int
    htr_head_activation: str
    htr_head_layer_norm: bool
    htr_head_bias: bool
    htr_extractor: Mapping[str, Any]
    replay_comparison_policy: str
    replay_relative_tolerance: float
    replay_absolute_tolerance: float

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "RoleNeutralMatchedPairConfig":
        if not isinstance(value, Mapping):
            raise TypeError("matched-pair configuration must be a mapping")
        payload = dict(value)
        expected = set(cls.__dataclass_fields__)
        wire_fields = {
            "schema_version",
            "text_truncation_policy",
            "matched_pair_subproducers",
        }
        if set(payload) != expected | wire_fields:
            raise ValueError(
                "matched-pair wire configuration keys differ; "
                f"missing={sorted((expected | wire_fields) - set(payload))}, "
                f"extra={sorted(set(payload) - (expected | wire_fields))}"
            )
        if (
            payload.pop("schema_version")
            != ROLE_NEUTRAL_MATCHED_PAIR_CONFIG_SCHEMA
            or payload.pop("text_truncation_policy")
            != "forbidden_capacity_must_not_bind_v1"
            or payload.pop("matched_pair_subproducers") != list(_SUBPRODUCERS)
        ):
            raise ValueError("matched-pair wire configuration policy changed")
        result = cls(**payload)
        result.as_dict()
        return result

    def as_dict(self) -> dict[str, Any]:
        integer_values = {
            "effect_folds": self.effect_folds,
            "max_controls_per_candidate": self.max_controls_per_candidate,
            "nearest_fallback_controls": self.nearest_fallback_controls,
            "bow_max_iter": self.bow_max_iter,
            "bow_optimizer_maxls": self.bow_optimizer_maxls,
            "bow_optimizer_maxcor": self.bow_optimizer_maxcor,
            "bow_optimizer_maxfun": self.bow_optimizer_maxfun,
            "htr_epochs": self.htr_epochs,
            "htr_batch_size": self.htr_batch_size,
            "htr_hidden_dim": self.htr_hidden_dim,
            "htr_head_depth": self.htr_head_depth,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in integer_values.values()
        ):
            raise TypeError("matched-pair integer configuration fields must be integers")
        if (
            int(self.effect_folds) < 2
            or int(self.max_controls_per_candidate) < 1
            or int(self.nearest_fallback_controls) < 0
            or int(self.bow_max_iter) < 1
            or int(self.bow_optimizer_maxls) < 1
            or int(self.bow_optimizer_maxcor) < 1
            or int(self.bow_optimizer_maxfun) < 1
            or int(self.htr_epochs) < 1
            or int(self.htr_batch_size) < 1
            or int(self.htr_hidden_dim) < 1
            or int(self.htr_head_depth) < 1
        ):
            raise ValueError("matched-pair integer configuration is infeasible")
        finite = (
            float(self.propensity_caliper),
            float(self.outcome_caliper),
            float(self.bow_l2_alpha),
            float(self.bow_optimizer_ftol),
            float(self.bow_optimizer_gtol),
            (
                0.0
                if self.bow_optimizer_tol is None
                else float(self.bow_optimizer_tol)
            ),
            float(self.htr_learning_rate),
            float(self.htr_weight_decay),
            float(self.htr_dropout),
            float(self.htr_adamw_beta1),
            float(self.htr_adamw_beta2),
            float(self.htr_adamw_eps),
            float(self.htr_gradient_clip_norm),
            float(self.htr_gradient_clip_norm_type),
            float(self.replay_relative_tolerance),
            float(self.replay_absolute_tolerance),
        )
        if not np.isfinite(finite).all():
            raise ValueError("matched-pair floating configuration must be finite")
        if (
            self.propensity_caliper < 0.0
            or self.outcome_caliper < 0.0
            or self.bow_l2_alpha < 0.0
            or self.bow_optimizer_ftol < 0.0
            or self.bow_optimizer_gtol < 0.0
            or (
                self.bow_optimizer_tol is not None
                and self.bow_optimizer_tol < 0.0
            )
            or self.htr_learning_rate <= 0.0
            or self.htr_weight_decay < 0.0
            or not 0.0 <= self.htr_dropout < 1.0
            or not 0.0 <= self.htr_adamw_beta1 < 1.0
            or not 0.0 <= self.htr_adamw_beta2 < 1.0
            or self.htr_adamw_eps <= 0.0
            or self.htr_gradient_clip_norm < 0.0
            or self.htr_gradient_clip_norm_type <= 0.0
        ):
            raise ValueError("matched-pair floating configuration is invalid")
        validate_neural_replay_settings(
            policy=self.replay_comparison_policy,
            relative_tolerance=self.replay_relative_tolerance,
            absolute_tolerance=self.replay_absolute_tolerance,
        )
        if self.bow_optimizer_method != "L-BFGS-B":
            raise ValueError("matched-pair BoW optimizer must be L-BFGS-B")
        if self.bow_optimizer_initialization != "zeros":
            raise ValueError("matched-pair BoW optimizer initialization must be zeros")
        if self.htr_optimizer_name != "adamw":
            raise ValueError("matched-pair HTR optimizer must be adamw")
        if self.htr_head_activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError("matched-pair HTR head activation is unsupported")
        boolean_values = {
            "bow_require_optimizer_success": self.bow_require_optimizer_success,
            "htr_adamw_amsgrad": self.htr_adamw_amsgrad,
            "htr_adamw_maximize": self.htr_adamw_maximize,
            "htr_adamw_foreach": self.htr_adamw_foreach,
            "htr_adamw_capturable": self.htr_adamw_capturable,
            "htr_adamw_differentiable": self.htr_adamw_differentiable,
            "htr_adamw_fused": self.htr_adamw_fused,
            "htr_optimizer_zero_grad_set_to_none": (
                self.htr_optimizer_zero_grad_set_to_none
            ),
            "htr_gradient_clip_error_if_nonfinite": (
                self.htr_gradient_clip_error_if_nonfinite
            ),
            "htr_gradient_clip_foreach": self.htr_gradient_clip_foreach,
            "htr_head_layer_norm": self.htr_head_layer_norm,
            "htr_head_bias": self.htr_head_bias,
        }
        if any(type(value) is not bool for value in boolean_values.values()):
            raise TypeError("matched-pair Boolean configuration must be exact")
        return {
            "schema_version": ROLE_NEUTRAL_MATCHED_PAIR_CONFIG_SCHEMA,
            "effect_folds": int(self.effect_folds),
            "propensity_caliper": float(self.propensity_caliper),
            "outcome_caliper": float(self.outcome_caliper),
            "max_controls_per_candidate": int(self.max_controls_per_candidate),
            "nearest_fallback_controls": int(self.nearest_fallback_controls),
            "bow_l2_alpha": float(self.bow_l2_alpha),
            "bow_max_iter": int(self.bow_max_iter),
            "bow_optimizer_method": self.bow_optimizer_method,
            "bow_optimizer_ftol": float(self.bow_optimizer_ftol),
            "bow_optimizer_gtol": float(self.bow_optimizer_gtol),
            "bow_optimizer_maxls": int(self.bow_optimizer_maxls),
            "bow_optimizer_maxcor": int(self.bow_optimizer_maxcor),
            "bow_optimizer_maxfun": int(self.bow_optimizer_maxfun),
            "bow_optimizer_tol": (
                None
                if self.bow_optimizer_tol is None
                else float(self.bow_optimizer_tol)
            ),
            "bow_optimizer_initialization": (
                self.bow_optimizer_initialization
            ),
            "bow_require_optimizer_success": (
                self.bow_require_optimizer_success
            ),
            "htr_epochs": int(self.htr_epochs),
            "htr_batch_size": int(self.htr_batch_size),
            "htr_learning_rate": float(self.htr_learning_rate),
            "htr_weight_decay": float(self.htr_weight_decay),
            "htr_optimizer_name": self.htr_optimizer_name,
            "htr_adamw_beta1": float(self.htr_adamw_beta1),
            "htr_adamw_beta2": float(self.htr_adamw_beta2),
            "htr_adamw_eps": float(self.htr_adamw_eps),
            "htr_adamw_amsgrad": self.htr_adamw_amsgrad,
            "htr_adamw_maximize": self.htr_adamw_maximize,
            "htr_adamw_foreach": self.htr_adamw_foreach,
            "htr_adamw_capturable": self.htr_adamw_capturable,
            "htr_adamw_differentiable": self.htr_adamw_differentiable,
            "htr_adamw_fused": self.htr_adamw_fused,
            "htr_optimizer_zero_grad_set_to_none": (
                self.htr_optimizer_zero_grad_set_to_none
            ),
            "htr_gradient_clip_norm": float(self.htr_gradient_clip_norm),
            "htr_gradient_clip_norm_type": float(
                self.htr_gradient_clip_norm_type
            ),
            "htr_gradient_clip_error_if_nonfinite": (
                self.htr_gradient_clip_error_if_nonfinite
            ),
            "htr_gradient_clip_foreach": self.htr_gradient_clip_foreach,
            "htr_hidden_dim": int(self.htr_hidden_dim),
            "htr_dropout": float(self.htr_dropout),
            "htr_head_depth": int(self.htr_head_depth),
            "htr_head_activation": self.htr_head_activation,
            "htr_head_layer_norm": self.htr_head_layer_norm,
            "htr_head_bias": self.htr_head_bias,
            "htr_extractor": _closed_extractor_config(self.htr_extractor),
            "replay_comparison_policy": self.replay_comparison_policy,
            "replay_relative_tolerance": float(
                self.replay_relative_tolerance
            ),
            "replay_absolute_tolerance": float(
                self.replay_absolute_tolerance
            ),
            "text_truncation_policy": "forbidden_capacity_must_not_bind_v1",
            "matched_pair_subproducers": list(_SUBPRODUCERS),
        }


def _replayed_predictions_match(
    observed: Any,
    expected: Any,
    *,
    subproducer: str,
    config: Mapping[str, Any],
) -> bool:
    left = np.asarray(observed)
    right = np.asarray(expected)
    if subproducer == "bow":
        return bool(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(left, right, equal_nan=True)
        )
    if subproducer != "htr":
        raise ValueError("matched-pair replay named an unknown subproducer")
    return neural_float_arrays_within_tolerance(
        left,
        right,
        policy=config.get("replay_comparison_policy"),
        relative_tolerance=config.get("replay_relative_tolerance"),
        absolute_tolerance=config.get("replay_absolute_tolerance"),
    )


def _producer_identity() -> str:
    module_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dependency_functions = (
        build_training_pairs,
        build_candidate_pairs,
        aggregate_pair_predictions,
        _capture_offset_model,
        _predict_offset_model,
        _capture_htr_pair_model,
        _build_htr_pair_model,
        _predict_htr_pair,
    )
    return _sha256_json(
        {
            "schema_version": (
                "production_role_neutral_matched_pair_producer_identity_v2"
            ),
            "module_file_sha256": module_digest,
            "dependency_sources": [
                inspect.getsource(function) for function in dependency_functions
            ],
        }
    )


@dataclass(frozen=True)
class RoleNeutralMatchedPairPhysicalGroupRequest:
    """Device-neutral scientific authority for one matched-pair physical fit."""

    scientific_plan_content_sha256: str
    physical_owner: Stage1ScopeSpec
    logical_members: tuple[Stage1ScopeSpec, ...]
    htr_model_identity_sha256: str
    nuisance_artifact_identity_sha256: str
    runtime_compatibility_class: str
    producer_identity_sha256: str
    content_sha256: str

    @classmethod
    def from_plan(
        cls,
        *,
        plan: Stage1ScopePlan,
        physical_owner_scope_id: str,
        htr_model_identity_sha256: str,
        nuisance_artifact_identity_sha256: str,
        runtime_compatibility_class: str,
    ) -> "RoleNeutralMatchedPairPhysicalGroupRequest":
        if not isinstance(plan, Stage1ScopePlan):
            raise TypeError("matched-pair request requires a Stage1ScopePlan")
        owner = plan.scope(str(physical_owner_scope_id))
        if plan.physical_owner(owner.scope_id).scope_id != owner.scope_id:
            raise ValueError("matched-pair request must name a physical owner")
        groups = [
            members
            for candidate, members in plan.physical_scope_groups
            if candidate.scope_id == owner.scope_id
        ]
        if len(groups) != 1:
            raise RuntimeError("matched-pair owner has no unique logical group")
        members = groups[0]
        if (
            not members
            or members[0].scope_id != owner.scope_id
            or any(
                tuple(member.fit_row_ids) != tuple(owner.fit_row_ids)
                or member.scope_seed != owner.scope_seed
                for member in members
            )
        ):
            raise ValueError("matched-pair logical group changed fit identity or seed")
        aliases = members[1:]
        if aliases and (
            owner.scope_kind != "exact_inner"
            or any(member.scope_kind != "cumulative_spent" for member in aliases)
        ):
            raise ValueError("matched-pair reuse supports exact/cumulative groups only")
        model_identity = _require_sha256(
            htr_model_identity_sha256,
            label="HTR model identity",
        )
        nuisance_identity = _require_sha256(
            nuisance_artifact_identity_sha256,
            label="nuisance artifact identity",
        )
        runtime = str(runtime_compatibility_class)
        if _SAFE_RUNTIME.fullmatch(runtime) is None:
            raise ValueError("runtime compatibility class is invalid")
        producer = _producer_identity()
        body = _group_request_body(
            scientific_plan_content_sha256=plan.scientific_content_sha256,
            owner=owner,
            members=members,
            htr_model_identity_sha256=model_identity,
            nuisance_artifact_identity_sha256=nuisance_identity,
            runtime_compatibility_class=runtime,
            producer_identity_sha256=producer,
        )
        return cls(
            scientific_plan_content_sha256=plan.scientific_content_sha256,
            physical_owner=owner,
            logical_members=members,
            htr_model_identity_sha256=model_identity,
            nuisance_artifact_identity_sha256=nuisance_identity,
            runtime_compatibility_class=runtime,
            producer_identity_sha256=producer,
            content_sha256=_sha256_json(body),
        )

    def as_dict(self) -> dict[str, Any]:
        _require_sha256(
            self.scientific_plan_content_sha256,
            label="scientific Stage 1 plan identity",
        )
        if self.producer_identity_sha256 != _producer_identity():
            raise RuntimeError("matched-pair producer code changed after request creation")
        body = _group_request_body(
            scientific_plan_content_sha256=self.scientific_plan_content_sha256,
            owner=self.physical_owner,
            members=self.logical_members,
            htr_model_identity_sha256=self.htr_model_identity_sha256,
            nuisance_artifact_identity_sha256=self.nuisance_artifact_identity_sha256,
            runtime_compatibility_class=self.runtime_compatibility_class,
            producer_identity_sha256=self.producer_identity_sha256,
        )
        if _sha256_json(body) != self.content_sha256:
            raise RuntimeError("matched-pair group request changed")
        return {**body, "content_sha256": self.content_sha256}


def _group_request_body(
    *,
    scientific_plan_content_sha256: str,
    owner: Stage1ScopeSpec,
    members: Sequence[Stage1ScopeSpec],
    htr_model_identity_sha256: str,
    nuisance_artifact_identity_sha256: str,
    runtime_compatibility_class: str,
    producer_identity_sha256: str,
) -> dict[str, Any]:
    if (
        not members
        or members[0].scope_id != owner.scope_id
        or len({member.scope_id for member in members}) != len(members)
        or any(
            tuple(member.fit_row_ids) != tuple(owner.fit_row_ids)
            or member.scope_seed != owner.scope_seed
            for member in members
        )
    ):
        raise ValueError("matched-pair group authority is invalid")
    return {
        "schema_version": ROLE_NEUTRAL_MATCHED_PAIR_GROUP_REQUEST_SCHEMA,
        "scientific_plan_content_sha256": _require_sha256(
            scientific_plan_content_sha256,
            label="scientific plan identity",
        ),
        "physical_owner": owner.as_dict(),
        "logical_members": [member.as_dict() for member in members],
        "logical_scope_count": len(members),
        "fit_row_ids": list(owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(owner.fit_row_ids),
        "canonical_group_seed": int(owner.scope_seed),
        "htr_model_identity_sha256": _require_sha256(
            htr_model_identity_sha256,
            label="HTR model identity",
        ),
        "nuisance_artifact_identity_sha256": _require_sha256(
            nuisance_artifact_identity_sha256,
            label="nuisance artifact identity",
        ),
        "runtime_compatibility_class": str(runtime_compatibility_class),
        "producer_identity_sha256": _require_sha256(
            producer_identity_sha256,
            label="producer identity",
        ),
        "heldout_labels_supplied": False,
        "peer_group_definitions_supplied": False,
        "execution_device_metadata_in_scientific_identity": False,
    }


@dataclass(frozen=True)
class RoleNeutralMatchedPairExactInput:
    """Closed exact-transform input; treatment/outcome cannot be supplied."""

    row_ids: tuple[int, ...]
    texts: tuple[str, ...]
    propensity_probability: tuple[float, ...]
    outcome_nuisance_probability: tuple[float, ...]

    def validated(self, expected_row_ids: Sequence[int]) -> dict[str, Any]:
        expected = tuple(map(int, expected_row_ids))
        rows = tuple(map(int, self.row_ids))
        texts = tuple(self.texts)
        if rows != expected or len(rows) != len(texts) or any(
            not isinstance(text, str) for text in texts
        ):
            raise ValueError("exact matched-pair input changed authorized row/text order")
        propensity = _finite_probability_vector(
            self.propensity_probability,
            label="exact propensity nuisance",
            length=len(rows),
        )
        outcome_nuisance = _finite_probability_vector(
            self.outcome_nuisance_probability,
            label="exact outcome nuisance",
            length=len(rows),
        )
        return {
            "schema_version": ROLE_NEUTRAL_MATCHED_PAIR_EXACT_INPUT_SCHEMA,
            "row_ids": list(rows),
            "row_order_fingerprint": _row_order_fingerprint(rows),
            "text_sha256": _text_sha256(rows, texts),
            "propensity_probability_sha256": _float_hex_sha256(propensity),
            "outcome_nuisance_probability_sha256": _float_hex_sha256(
                outcome_nuisance
            ),
            "heldout_treatment_field_present": False,
            "heldout_outcome_field_present": False,
        }


@dataclass(frozen=True)
class _MatchedPairEffectFoldTask:
    """Spawn-safe authority for one canonical matched-pair effect fold."""

    objective: str
    fold: int
    split_seed: int
    htr_seed: int
    owner_scope_seed: int
    owner_fit_row_ids: tuple[int, ...]
    fit_texts: tuple[str, ...]
    treatment: np.ndarray
    outcome: np.ndarray
    propensity_probability: np.ndarray
    outcome_nuisance_probability: np.ndarray
    fit_positions: np.ndarray
    validation_positions: np.ndarray
    view_configs: tuple[BoWViewConfig, ...]
    config: Mapping[str, Any]
    htr_model_path: str | None


@dataclass(frozen=True)
class _MatchedPairEffectFoldResult:
    """Isolated CPU state returned across the bounded fold barrier."""

    objective: str
    fold: int
    split_seed: int
    htr_seed: int
    fit_positions: np.ndarray
    validation_positions: np.ndarray
    control_positions: np.ndarray
    fold_record: Mapping[str, Any]
    bow_oof: Mapping[str, Mapping[str, np.ndarray]]
    htr_oof: Mapping[str, np.ndarray]
    arrays: Mapping[str, np.ndarray]
    gpu_peak_allocated_bytes: int | None


def _assert_text_capacity(
    texts: Sequence[str],
    *,
    extractor_config: Mapping[str, Any],
    stage: str,
) -> None:
    chunk_size = int(extractor_config["chunk_size_words"])
    overlap = int(extractor_config["chunk_overlap_words"])
    max_chunks = int(extractor_config["max_chunks"])
    if chunk_size < 1 or max_chunks < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"{stage} HTR chunk geometry is invalid")
    maximum_words = chunk_size + (max_chunks - 1) * (chunk_size - overlap)
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"{stage} text {index} is not a string")
        word_count = len(_WORD.findall(text))
        if word_count > maximum_words:
            raise ValueError(
                f"{stage} text {index} requires {word_count} HTR words but configured "
                f"lossless capacity is {maximum_words}; truncation is forbidden"
            )


def _validate_htr_model_locator(
    *,
    request: RoleNeutralMatchedPairPhysicalGroupRequest,
    config: Mapping[str, Any],
    htr_model_path: Path | str | None,
) -> None:
    marker = config["htr_extractor"]["sentence_encoder_model"]
    if marker == "hash":
        if htr_model_path is not None:
            raise ValueError("hash HTR execution cannot claim a local model locator")
        return
    if marker != "authenticated_local_tree" or htr_model_path is None:
        raise ValueError("local HTR execution requires its authenticated model tree")
    if directory_tree_sha256(htr_model_path) != request.htr_model_identity_sha256:
        raise ValueError("HTR model tree differs from the request's content identity")


def _make_frame(row_ids: Sequence[int]) -> pd.DataFrame:
    return pd.DataFrame({"_oci_row_id": np.asarray(row_ids, dtype=np.int64)})


def _matching_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "propensity_caliper": float(config["propensity_caliper"]),
        "outcome_caliper": float(config["outcome_caliper"]),
    }


def _matching_candidate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_matching_training_config(config),
        "max_controls_per_candidate": int(config["max_controls_per_candidate"]),
        "nearest_fallback_controls": int(config["nearest_fallback_controls"]),
    }


def _capture_pair_table(
    store: _ArrayStore,
    *,
    prefix: str,
    pairs: pd.DataFrame,
) -> dict[str, Any]:
    columns = {
        "candidate_pos": np.int64,
        "control_pos": np.int64,
        "candidate_row_id": np.int64,
        "control_row_id": np.int64,
        "label": np.float64,
        "base_prob": np.float64,
        "base_logit": np.float64,
        "propensity_abs_diff": np.float64,
        "outcome_abs_diff": np.float64,
        "score_abs_diff_sum": np.float64,
    }
    references: dict[str, str] = {}
    for column, dtype in columns.items():
        if column not in pairs:
            raise RuntimeError(f"matched-pair table lacks {column}")
        references[column] = store.add(
            f"{prefix}_{column}",
            pairs[column].to_numpy(dtype=dtype),
        )
    return {
        "row_count": len(pairs),
        "pair_fingerprint": _pair_fingerprint(pairs),
        "columns": references,
    }


def _train_htr(
    *,
    pairs: pd.DataFrame,
    config: Mapping[str, Any],
    seed: int,
    extractor_factory: Callable[[torch.device], HierarchicalTransformerExtractor],
    device: torch.device,
) -> HTRPairUpliftNet:
    if pairs.empty or len(np.unique(pairs["label"].to_numpy(dtype=int))) < 2:
        raise RuntimeError(
            "role-neutral matched-pair HTR requires nonempty pairs with both outcomes"
        )
    torch.default_generator.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed(int(seed))
    extractor = extractor_factory(device)
    if type(extractor) is not HierarchicalTransformerExtractor:
        raise TypeError("HTR extractor factory returned a non-native extractor")
    initialization_texts = (
        pairs["control_text"].astype(str).tolist()
        + pairs["treated_text"].astype(str).tolist()
    )
    _assert_text_capacity(
        initialization_texts,
        extractor_config=config["htr_extractor"],
        stage="HTR fit pair",
    )
    extractor.fit_tokenizer(initialization_texts)
    descriptor = _extractor_descriptor(extractor)
    if (
        descriptor.get("constructor") != config["htr_extractor"]
        or (
            config["htr_extractor"]["sentence_encoder_model"]
            == "authenticated_local_tree"
            and descriptor.get("effective_sentence_pooling")
            != "token_attention"
        )
    ):
        raise RuntimeError("HTR extractor differs from the configured architecture")
    model = HTRPairUpliftNet(
        extractor=extractor,
        hidden_dim=int(config["htr_hidden_dim"]),
        dropout=float(config["htr_dropout"]),
        head_depth=int(config["htr_head_depth"]),
        head_activation=str(config["htr_head_activation"]),
        head_layer_norm=config["htr_head_layer_norm"],
        head_bias=config["htr_head_bias"],
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("HTR matched-pair model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["htr_learning_rate"]),
        betas=(
            float(config["htr_adamw_beta1"]),
            float(config["htr_adamw_beta2"]),
        ),
        eps=float(config["htr_adamw_eps"]),
        weight_decay=float(config["htr_weight_decay"]),
        amsgrad=config["htr_adamw_amsgrad"],
        maximize=config["htr_adamw_maximize"],
        foreach=config["htr_adamw_foreach"],
        capturable=config["htr_adamw_capturable"],
        differentiable=config["htr_adamw_differentiable"],
        fused=config["htr_adamw_fused"],
    )
    labels = torch.as_tensor(
        pairs["label"].to_numpy(dtype=np.float32),
        device=device,
    )
    base = torch.as_tensor(
        pairs["base_logit"].to_numpy(dtype=np.float32),
        device=device,
    )
    control = pairs["control_text"].astype(str).tolist()
    treated = pairs["treated_text"].astype(str).tolist()
    batch_size = int(config["htr_batch_size"])
    for epoch in range(1, int(config["htr_epochs"]) + 1):
        order = np.arange(len(pairs), dtype=np.int64)
        rng = np.random.default_rng(
            _derived_seed(seed, purpose="htr_epoch_order", fold=epoch, view="htr")
        )
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            optimizer.zero_grad(
                set_to_none=config["htr_optimizer_zero_grad_set_to_none"]
            )
            delta = model(
                [control[int(position)] for position in positions],
                [treated[int(position)] for position in positions],
            )
            loss = F.binary_cross_entropy_with_logits(
                base[positions] + delta,
                labels[positions],
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("HTR matched-pair training produced non-finite loss")
            loss.backward()
            if float(config["htr_gradient_clip_norm"]) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["htr_gradient_clip_norm"]),
                    norm_type=float(config["htr_gradient_clip_norm_type"]),
                    error_if_nonfinite=config[
                        "htr_gradient_clip_error_if_nonfinite"
                    ],
                    foreach=config["htr_gradient_clip_foreach"],
                )
            optimizer.step()
    model.eval()
    return model


def _htr_training_configuration(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "optimizer_name": config["htr_optimizer_name"],
        "learning_rate": config["htr_learning_rate"],
        "weight_decay": config["htr_weight_decay"],
        "adamw_beta1": config["htr_adamw_beta1"],
        "adamw_beta2": config["htr_adamw_beta2"],
        "adamw_eps": config["htr_adamw_eps"],
        "adamw_amsgrad": config["htr_adamw_amsgrad"],
        "adamw_maximize": config["htr_adamw_maximize"],
        "adamw_foreach": config["htr_adamw_foreach"],
        "adamw_capturable": config["htr_adamw_capturable"],
        "adamw_differentiable": config["htr_adamw_differentiable"],
        "adamw_fused": config["htr_adamw_fused"],
        "optimizer_zero_grad_set_to_none": config[
            "htr_optimizer_zero_grad_set_to_none"
        ],
        "gradient_clip_norm": config["htr_gradient_clip_norm"],
        "gradient_clip_norm_type": config["htr_gradient_clip_norm_type"],
        "gradient_clip_error_if_nonfinite": config[
            "htr_gradient_clip_error_if_nonfinite"
        ],
        "gradient_clip_foreach": config["htr_gradient_clip_foreach"],
        "batch_size": config["htr_batch_size"],
        "epochs": config["htr_epochs"],
    }


def _predict_live_htr(
    model: HTRPairUpliftNet,
    pairs: pd.DataFrame,
    *,
    batch_size: int,
) -> np.ndarray:
    if pairs.empty:
        return np.zeros(0, dtype=np.float64)
    control = pairs["control_text"].astype(str).tolist()
    treated = pairs["treated_text"].astype(str).tolist()
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), int(batch_size)):
            end = start + int(batch_size)
            outputs.append(
                model(control[start:end], treated[start:end])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
    result = np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float64)
    if result.shape != (len(pairs),) or not np.isfinite(result).all():
        raise RuntimeError("HTR matched-pair transform emitted invalid deltas")
    return result


def _mean_with_nan(values: Sequence[np.ndarray], *, length: int) -> np.ndarray:
    if not values:
        raise RuntimeError("matched-pair ensemble has no fold predictions")
    matrix = np.vstack([np.asarray(value, dtype=np.float64) for value in values])
    if matrix.shape != (len(values), int(length)):
        raise RuntimeError("matched-pair fold prediction shapes changed")
    finite = np.isfinite(matrix)
    count = np.sum(finite, axis=0)
    total = np.sum(np.where(finite, matrix, 0.0), axis=0)
    result = np.full(length, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _subproducer_evidence(
    *,
    fold_records: Sequence[Mapping[str, Any]],
    store: _ArrayStore,
) -> dict[str, dict[str, Any]]:
    bow_atoms: list[dict[str, Any]] = []
    htr_atoms: list[dict[str, Any]] = []
    for fold_row in fold_records:
        fold = int(fold_row["fold"])
        for state in fold_row["bow_states"]:
            descriptor = state["model"]
            vectorizer = descriptor.get("vectorizer")
            coefficient_key = descriptor.get("coefficient")
            names = list((vectorizer or {}).get("feature_names") or ())
            if not names or not isinstance(coefficient_key, str):
                raise RuntimeError("matched-pair BoW state has no complete vocabulary")
            coefficient = np.asarray(store.arrays[coefficient_key], dtype=np.float64)
            if coefficient.shape != (2 * len(names),):
                raise RuntimeError("matched-pair BoW coefficient/vocabulary shape changed")
            for index, term in enumerate(names):
                bow_atoms.append(
                    {
                        "fold": fold,
                        "view_name": state["view_name"],
                        "feature_index": index,
                        "term": str(term),
                        "control_delta_logit_coefficient": float(coefficient[index]),
                        "treated_delta_logit_coefficient": float(
                            coefficient[len(names) + index]
                        ),
                    }
                )
        validation = fold_row["validation_pair_table"]
        columns = validation["columns"]
        candidate = np.asarray(store.arrays[columns["candidate_row_id"]], dtype=np.int64)
        control = np.asarray(store.arrays[columns["control_row_id"]], dtype=np.int64)
        propensity_diff = np.asarray(
            store.arrays[columns["propensity_abs_diff"]],
            dtype=np.float64,
        )
        outcome_diff = np.asarray(
            store.arrays[columns["outcome_abs_diff"]],
            dtype=np.float64,
        )
        delta = np.asarray(
            store.arrays[fold_row["htr_validation_pair_delta"]],
            dtype=np.float64,
        )
        if not (
            candidate.shape
            == control.shape
            == propensity_diff.shape
            == outcome_diff.shape
            == delta.shape
        ):
            raise RuntimeError("matched-pair HTR witness arrays changed shape")
        for index in range(len(candidate)):
            htr_atoms.append(
                {
                    "fold": fold,
                    "pair_index": index,
                    "candidate_row_id": int(candidate[index]),
                    "control_row_id": int(control[index]),
                    "propensity_abs_diff": float(propensity_diff[index]),
                    "outcome_abs_diff": float(outcome_diff[index]),
                    "delta_logit": float(delta[index]),
                }
            )
    if not bow_atoms or not htr_atoms:
        raise RuntimeError("matched-pair fit lacks one mandated subproducer evidence path")
    return {
        "bow": {
            "subproducer": "bow",
            "evidence_kind": "complete_fold_vocabulary_coefficients_v1",
            "top_k_applied": False,
            "text_truncation_applied": False,
            "atoms": bow_atoms,
        },
        "htr": {
            "subproducer": "htr",
            "evidence_kind": "complete_validation_pair_witnesses_v1",
            "top_k_applied": False,
            "text_truncation_applied": False,
            "atoms": htr_atoms,
        },
    }


def _new_fold_htr_extractor(
    *,
    config: Mapping[str, Any],
    htr_model_path: str | None,
    device: torch.device,
) -> HierarchicalTransformerExtractor:
    constructor = copy.deepcopy(dict(config["htr_extractor"]))
    marker = constructor["sentence_encoder_model"]
    if marker == "authenticated_local_tree":
        if htr_model_path is None:
            raise ValueError(
                "matched-pair fold lacks its authenticated HTR model locator"
            )
        constructor["sentence_encoder_model"] = htr_model_path
    elif marker != "hash" or htr_model_path is not None:
        raise ValueError("matched-pair fold HTR model locator changed")
    return HierarchicalTransformerExtractor(**constructor, device=device)


def _run_matched_pair_effect_fold(
    task: _MatchedPairEffectFoldTask,
    device_name: str,
) -> _MatchedPairEffectFoldResult:
    """Own, fit, replay, and close one matched-pair fold in one worker."""

    if not isinstance(task, _MatchedPairEffectFoldTask):
        raise TypeError("matched-pair worker received another fold task type")
    if task.objective != MATCHED_PAIR_UPLIFT:
        raise ValueError("matched-pair worker objective changed")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("matched-pair fold lease must be CPU or CUDA")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    frame = _make_frame(task.owner_fit_row_ids)
    fit_pos = np.asarray(task.fit_positions, dtype=np.int64)
    validation_pos = np.asarray(task.validation_positions, dtype=np.int64)
    fit_frame = frame.iloc[fit_pos].reset_index(drop=True)
    validation_frame = frame.iloc[validation_pos].reset_index(drop=True)
    fit_pairs = build_training_pairs(
        fit_frame,
        texts=[task.fit_texts[int(position)] for position in fit_pos],
        treatment=task.treatment[fit_pos],
        outcome=task.outcome[fit_pos],
        propensity=task.propensity_probability[fit_pos],
        outcome_prob=task.outcome_nuisance_probability[fit_pos],
        **_matching_training_config(task.config),
    )
    if fit_pairs.empty or set(fit_pairs["label"].astype(int)) != {0, 1}:
        raise RuntimeError(
            f"matched-pair fold {task.fold} cannot genuinely fit both "
            "subproducers"
        )
    control_pos = fit_pos[task.treatment[fit_pos].astype(int) == 0]
    if not len(control_pos):
        raise RuntimeError(
            f"matched-pair fold {task.fold} has no fit controls"
        )
    control_frame = frame.iloc[control_pos].reset_index(drop=True)
    validation_pairs = build_candidate_pairs(
        validation_frame,
        control_frame,
        candidate_texts=[
            task.fit_texts[int(position)] for position in validation_pos
        ],
        control_texts=[
            task.fit_texts[int(position)] for position in control_pos
        ],
        candidate_propensity=task.propensity_probability[validation_pos],
        candidate_outcome_prob=(
            task.outcome_nuisance_probability[validation_pos]
        ),
        control_propensity=task.propensity_probability[control_pos],
        control_outcome_prob=(
            task.outcome_nuisance_probability[control_pos]
        ),
        **_matching_candidate_config(task.config),
    )
    if validation_pairs.empty:
        raise RuntimeError(
            f"matched-pair fold {task.fold} has no validation witnesses"
        )

    store = _ArrayStore()
    prefix = f"fold_{task.fold:04d}"
    fit_pair_table = _capture_pair_table(
        store,
        prefix=f"{prefix}_fit_pairs",
        pairs=fit_pairs,
    )
    validation_pair_table = _capture_pair_table(
        store,
        prefix=f"{prefix}_validation_pairs",
        pairs=validation_pairs,
    )
    bow_rows: list[dict[str, Any]] = []
    bow_oof: dict[str, dict[str, np.ndarray]] = {}
    htr_model: HTRPairUpliftNet | None = None
    replay_model: HTRPairUpliftNet | None = None
    try:
        for view_index, view in enumerate(task.view_configs):
            seed = _derived_seed(
                task.owner_scope_seed,
                purpose="bow_pair_model",
                fold=task.fold,
                view=view.name,
            )
            model = OffsetLogitBoWPairModel(
                vectorizer_params=_vectorizer_params(view),
                l2_alpha=float(task.config["bow_l2_alpha"]),
                max_iter=int(task.config["bow_max_iter"]),
                random_state=seed,
                optimizer_method=str(task.config["bow_optimizer_method"]),
                optimizer_ftol=float(task.config["bow_optimizer_ftol"]),
                optimizer_gtol=float(task.config["bow_optimizer_gtol"]),
                optimizer_maxls=int(task.config["bow_optimizer_maxls"]),
                optimizer_maxcor=int(task.config["bow_optimizer_maxcor"]),
                optimizer_maxfun=int(task.config["bow_optimizer_maxfun"]),
                optimizer_tol=(
                    None
                    if task.config["bow_optimizer_tol"] is None
                    else float(task.config["bow_optimizer_tol"])
                ),
                optimizer_initialization=str(
                    task.config["bow_optimizer_initialization"]
                ),
                require_optimizer_success=task.config[
                    "bow_require_optimizer_success"
                ],
            ).fit(fit_pairs)
            descriptor = _capture_offset_model(
                model,
                store,
                f"{prefix}_bow_{view_index:04d}",
            )
            pair_delta = np.asarray(
                model.predict_delta_logit(validation_pairs),
                dtype=np.float64,
            )
            replay_delta = _predict_offset_model(
                descriptor,
                store.arrays,
                validation_pairs,
            )
            if not _replayed_predictions_match(
                pair_delta,
                replay_delta,
                subproducer="bow",
                config=task.config,
            ):
                raise RuntimeError(
                    "live/sealed BoW matched-pair validation differs"
                )
            delta, probability, n_controls = aggregate_pair_predictions(
                validation_pairs,
                pair_delta,
                len(validation_frame),
            )
            bow_oof[view.name] = {
                "delta": np.ascontiguousarray(delta),
                "probability": np.ascontiguousarray(probability),
                "n_controls": np.ascontiguousarray(n_controls),
            }
            bow_rows.append(
                {
                    "view_name": view.name,
                    "view_index": view_index,
                    "view_config": _bow_view_to_dict(view),
                    "seed": seed,
                    "model": descriptor,
                    "validation_pair_delta": store.add(
                        f"{prefix}_bow_{view_index:04d}_validation_pair_delta",
                        pair_delta,
                    ),
                    "validation_delta": store.add(
                        f"{prefix}_bow_{view_index:04d}_validation_delta",
                        delta,
                    ),
                    "validation_probability": store.add(
                        f"{prefix}_bow_{view_index:04d}_validation_probability",
                        probability,
                    ),
                    "validation_n_controls": store.add(
                        f"{prefix}_bow_{view_index:04d}_validation_n_controls",
                        n_controls,
                    ),
                }
            )

        def extractor_factory(
            worker_device: torch.device,
        ) -> HierarchicalTransformerExtractor:
            return _new_fold_htr_extractor(
                config=task.config,
                htr_model_path=task.htr_model_path,
                device=worker_device,
            )

        htr_model = _train_htr(
            pairs=fit_pairs,
            config=task.config,
            seed=task.htr_seed,
            extractor_factory=extractor_factory,
            device=device,
        )
        htr_state = _capture_htr_pair_model(
            htr_model,
            store,
            f"{prefix}_htr",
            training_configuration=_htr_training_configuration(task.config),
        )
        htr_pair_delta = _predict_live_htr(
            htr_model,
            validation_pairs,
            batch_size=int(task.config["htr_batch_size"]),
        )
        replay_model = _build_htr_pair_model(
            htr_state,
            store.arrays,
            initialization_texts=task.fit_texts,
            htr_model_path=task.htr_model_path,
            device=device,
        )
        replay_htr = _predict_htr_pair(
            replay_model,
            validation_pairs,
            batch_size=int(task.config["htr_batch_size"]),
        )
        if not _replayed_predictions_match(
            htr_pair_delta,
            replay_htr,
            subproducer="htr",
            config=task.config,
        ):
            raise RuntimeError(
                "live/sealed HTR matched-pair validation differs beyond its "
                "declared tolerance"
            )
        htr_delta, htr_probability, htr_n_controls = (
            aggregate_pair_predictions(
                validation_pairs,
                htr_pair_delta,
                len(validation_frame),
            )
        )
        htr_oof = {
            "delta": np.ascontiguousarray(htr_delta),
            "probability": np.ascontiguousarray(htr_probability),
            "n_controls": np.ascontiguousarray(htr_n_controls),
        }
        fold_record = {
            "fold": task.fold,
            "split_seed": task.split_seed,
            "fit_positions": fit_pos.tolist(),
            "validation_positions": validation_pos.tolist(),
            "fit_row_ids": [
                int(task.owner_fit_row_ids[int(position)])
                for position in fit_pos
            ],
            "validation_row_ids": [
                int(task.owner_fit_row_ids[int(position)])
                for position in validation_pos
            ],
            "control_positions": store.add(
                f"{prefix}_control_positions",
                control_pos,
            ),
            "fit_pair_table": fit_pair_table,
            "validation_pair_table": validation_pair_table,
            "bow_states": bow_rows,
            "htr_seed": task.htr_seed,
            "htr_model": htr_state,
            "htr_validation_pair_delta": store.add(
                f"{prefix}_htr_validation_pair_delta",
                htr_pair_delta,
            ),
            "htr_validation_delta": store.add(
                f"{prefix}_htr_validation_delta",
                htr_delta,
            ),
            "htr_validation_probability": store.add(
                f"{prefix}_htr_validation_probability",
                htr_probability,
            ),
            "htr_validation_n_controls": store.add(
                f"{prefix}_htr_validation_n_controls",
                htr_n_controls,
            ),
            "fit_labels_accessed": True,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
            "text_truncation_applied": False,
        }
        peak = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
        return _MatchedPairEffectFoldResult(
            objective=task.objective,
            fold=task.fold,
            split_seed=task.split_seed,
            htr_seed=task.htr_seed,
            fit_positions=np.ascontiguousarray(fit_pos),
            validation_positions=np.ascontiguousarray(validation_pos),
            control_positions=np.ascontiguousarray(control_pos),
            fold_record=fold_record,
            bow_oof=bow_oof,
            htr_oof=htr_oof,
            arrays={
                key: np.ascontiguousarray(value)
                for key, value in store.arrays.items()
            },
            gpu_peak_allocated_bytes=peak,
        )
    finally:
        del replay_model
        del htr_model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _maximum_fold_overlap(
    intervals: Sequence[Mapping[str, Any]],
) -> int:
    boundaries = [
        (int(row["started_monotonic_ns"]), 1)
        for row in intervals
    ] + [
        (int(row["finished_monotonic_ns"]), -1)
        for row in intervals
    ]
    active = 0
    maximum = 0
    for _timestamp, delta in sorted(
        boundaries,
        key=lambda value: (value[0], value[1]),
    ):
        active += delta
        if active < 0:
            raise RuntimeError(
                "matched-pair fold telemetry released an idle lease"
            )
        maximum = max(maximum, active)
    if active != 0:
        raise RuntimeError("matched-pair fold telemetry left a lease active")
    return maximum


def _matched_pair_fold_execution_summary(
    *,
    events: Sequence[Mapping[str, Any]],
    resource_plan: RoleNeutralHTRFoldResourcePlan,
    effect_folds: int,
) -> dict[str, Any]:
    starts = [
        dict(row) for row in events if row.get("event") == "fold_started"
    ]
    finishes = [
        dict(row) for row in events if row.get("event") == "fold_finished"
    ]

    def key(row: Mapping[str, Any]) -> tuple[str, int]:
        return str(row.get("objective")), int(row.get("fold", 0))

    starts_by_key = {key(row): row for row in starts}
    finishes_by_key = {key(row): row for row in finishes}
    expected = {
        (MATCHED_PAIR_UPLIFT, fold)
        for fold in range(1, int(effect_folds) + 1)
    }
    if (
        len(starts) != len(starts_by_key)
        or len(finishes) != len(finishes_by_key)
        or set(starts_by_key) != expected
        or set(finishes_by_key) != expected
    ):
        raise RuntimeError(
            "matched-pair fold telemetry changed canonical coverage"
        )
    intervals: list[dict[str, Any]] = []
    for fold_key in sorted(expected, key=lambda value: value[1]):
        start = starts_by_key[fold_key]
        finish = finishes_by_key[fold_key]
        if (
            start.get("stage") != "effect"
            or finish.get("stage") != "effect"
            or start.get("device") != finish.get("device")
            or start.get("process_id") != finish.get("process_id")
            or int(finish["monotonic_ns"]) <= int(start["monotonic_ns"])
        ):
            raise RuntimeError(
                "matched-pair fold telemetry changed one lease interval"
            )
        determinism = finish.get("torch_determinism_observed")
        if (
            resource_plan.fold_parallel_backend == "processes"
            and not isinstance(determinism, Mapping)
        ):
            raise RuntimeError(
                "matched-pair process fold lacks Torch determinism telemetry"
            )
        if (
            resource_plan.fold_parallel_backend == "threads"
            and determinism is not None
        ):
            raise RuntimeError(
                "matched-pair thread fold claimed child determinism telemetry"
            )
        intervals.append(
            {
                "fold": fold_key[1],
                "device": str(start["device"]),
                "process_id": int(start["process_id"]),
                "thread_id": int(start["thread_id"]),
                "started_monotonic_ns": int(start["monotonic_ns"]),
                "finished_monotonic_ns": int(finish["monotonic_ns"]),
                "gpu_peak_allocated_bytes": finish.get(
                    "gpu_peak_allocated_bytes"
                ),
                "torch_determinism_observed": (
                    None
                    if determinism is None
                    else copy.deepcopy(dict(determinism))
                ),
            }
        )
    if {row["device"] for row in intervals} != set(resource_plan.devices):
        raise RuntimeError(
            "matched-pair folds did not exercise every selected device"
        )
    per_device: dict[str, dict[str, Any]] = {}
    for device in resource_plan.devices:
        rows = [row for row in intervals if row["device"] == device]
        maximum = _maximum_fold_overlap(rows)
        if maximum > resource_plan.fold_slots_per_device:
            raise RuntimeError(
                "matched-pair folds exceeded configured per-device slots"
            )
        peaks = [
            int(row["gpu_peak_allocated_bytes"])
            for row in rows
            if row["gpu_peak_allocated_bytes"] is not None
        ]
        per_device[device] = {
            "task_count": len(rows),
            "maximum_concurrent_leases": maximum,
            "maximum_child_peak_allocated_bytes": (
                max(peaks) if peaks else None
            ),
        }
    overall = _maximum_fold_overlap(intervals)
    if overall > resource_plan.fold_parallelism:
        raise RuntimeError(
            "matched-pair folds exceeded configured total concurrency"
        )
    if (
        resource_plan.fold_parallelism > 1
        and len(intervals) > 1
        and overall < 2
    ):
        raise RuntimeError(
            "configured matched-pair effect folds did not overlap"
        )
    if (
        resource_plan.fold_parallel_backend == "processes"
        and resource_plan.fold_parallelism > 1
        and len({row["process_id"] for row in intervals}) < 2
    ):
        raise RuntimeError(
            "parallel matched-pair folds were not process isolated"
        )
    return {
        "resource_plan": resource_plan.as_dict(),
        "fold_intervals": intervals,
        "per_device": per_device,
        "maximum_concurrent_fold_leases": overall,
        "configured_total_fold_concurrency_respected": True,
        "configured_per_device_slots_respected": True,
        "every_selected_device_used": True,
        "nested_native_worker_threads": (
            resource_plan.worker_cpu_threads
        ),
        "process_isolated_rng_and_torch_determinism": (
            resource_plan.fold_parallel_backend == "processes"
        ),
        "resource_locators_in_scientific_identity": False,
    }


def _fit_models(
    *,
    request: RoleNeutralMatchedPairPhysicalGroupRequest,
    fit_texts: tuple[str, ...],
    treatment: np.ndarray,
    outcome: np.ndarray,
    e_fit: np.ndarray,
    m_fit: np.ndarray,
    view_configs: tuple[BoWViewConfig, ...],
    config: Mapping[str, Any],
    device: torch.device,
    htr_model_path: Path | str | None,
    store: _ArrayStore,
    resource_plan: RoleNeutralHTRFoldResourcePlan,
    external_event_sink: (
        Callable[[Mapping[str, Any]], None] | None
    ),
) -> tuple[
    list[dict[str, Any]],
    dict[str, np.ndarray],
    tuple[Mapping[str, Any], ...],
    Mapping[str, Any],
]:
    """Build canonical tasks, execute them under leases, and merge by fold."""

    owner = request.physical_owner
    frame = _make_frame(owner.fit_row_ids)
    split_seed = _derived_seed(
        owner.scope_seed,
        purpose="effect_cross_fit_split",
        fold=0,
        view="matched_pair",
    )
    splits = tuple(
        KFold(
            n_splits=int(config["effect_folds"]),
            shuffle=True,
            random_state=split_seed,
        ).split(frame)
    )
    resolved_model_path = (
        None
        if htr_model_path is None
        else str(Path(htr_model_path).resolve())
    )
    tasks = tuple(
        _MatchedPairEffectFoldTask(
            objective=MATCHED_PAIR_UPLIFT,
            fold=fold,
            split_seed=split_seed,
            htr_seed=_derived_seed(
                owner.scope_seed,
                purpose="htr_pair_model",
                fold=fold,
                view="htr",
            ),
            owner_scope_seed=int(owner.scope_seed),
            owner_fit_row_ids=tuple(map(int, owner.fit_row_ids)),
            fit_texts=fit_texts,
            treatment=np.ascontiguousarray(treatment),
            outcome=np.ascontiguousarray(outcome),
            propensity_probability=np.ascontiguousarray(e_fit),
            outcome_nuisance_probability=np.ascontiguousarray(m_fit),
            fit_positions=np.asarray(raw_fit, dtype=np.int64),
            validation_positions=np.asarray(
                raw_validation,
                dtype=np.int64,
            ),
            view_configs=view_configs,
            config=copy.deepcopy(dict(config)),
            htr_model_path=resolved_model_path,
        )
        for fold, (raw_fit, raw_validation) in enumerate(splits, start=1)
    )
    fold_events: list[Mapping[str, Any]] = []

    def emit(value: Mapping[str, Any]) -> None:
        closed = json.loads(_canonical_json(dict(value)))
        fold_events.append(closed)
        if external_event_sink is not None:
            external_event_sink(copy.deepcopy(closed))

    raw_results = _execute_htr_fold_tasks(
        tasks,
        resource_plan=resource_plan,
        worker=_run_matched_pair_effect_fold,
        stage="effect",
        event_sink=emit,
    )
    by_fold: dict[int, _MatchedPairEffectFoldResult] = {}
    for raw_result in raw_results:
        if not isinstance(raw_result, _MatchedPairEffectFoldResult):
            raise TypeError(
                "matched-pair effect fold returned another result type"
            )
        if raw_result.fold in by_fold:
            raise RuntimeError(
                "matched-pair effect worker duplicated a canonical fold"
            )
        by_fold[raw_result.fold] = raw_result
    expected_folds = list(range(1, len(tasks) + 1))
    if sorted(by_fold) != expected_folds:
        raise RuntimeError(
            "matched-pair effect workers omitted or substituted a fold"
        )

    records: list[dict[str, Any]] = []
    bow_oof: dict[str, dict[str, np.ndarray]] = {
        view.name: {
            "delta": np.full(len(frame), np.nan, dtype=np.float64),
            "probability": np.full(len(frame), np.nan, dtype=np.float64),
            "n_controls": np.zeros(len(frame), dtype=np.float64),
        }
        for view in view_configs
    }
    htr_oof = {
        "delta": np.full(len(frame), np.nan, dtype=np.float64),
        "probability": np.full(len(frame), np.nan, dtype=np.float64),
        "n_controls": np.zeros(len(frame), dtype=np.float64),
    }
    for task in tasks:
        result = by_fold[task.fold]
        expected_control_positions = task.fit_positions[
            treatment[task.fit_positions].astype(int) == 0
        ]
        control_reference = str(
            result.fold_record.get("control_positions") or ""
        )
        if (
            result.objective != task.objective
            or result.fold != task.fold
            or result.split_seed != task.split_seed
            or result.htr_seed != task.htr_seed
            or not np.array_equal(
                result.fit_positions,
                task.fit_positions,
            )
            or not np.array_equal(
                result.validation_positions,
                task.validation_positions,
            )
            or not np.array_equal(
                result.control_positions,
                expected_control_positions,
            )
            or result.fold_record.get("fold") != task.fold
            or result.fold_record.get("split_seed") != task.split_seed
            or result.fold_record.get("htr_seed") != task.htr_seed
            or set(result.bow_oof)
            != {view.name for view in view_configs}
            or set(result.htr_oof)
            != {"delta", "probability", "n_controls"}
            or control_reference not in result.arrays
            or not np.array_equal(
                np.asarray(
                    result.arrays[control_reference],
                    dtype=np.int64,
                ),
                expected_control_positions,
            )
        ):
            raise RuntimeError(
                "matched-pair effect result changed rows, seeds, or identity"
            )
        expected_shape = task.validation_positions.shape
        if any(
            set(values) != {"delta", "probability", "n_controls"}
            or any(
                np.asarray(array).shape != expected_shape
                for array in values.values()
            )
            for values in result.bow_oof.values()
        ) or any(
            np.asarray(array).shape != expected_shape
            for array in result.htr_oof.values()
        ):
            raise RuntimeError(
                "matched-pair effect result changed validation shape"
            )
        if not result.arrays:
            raise RuntimeError(
                "matched-pair effect fold returned no private proof arrays"
            )
        for key in sorted(result.arrays):
            store.add(key, result.arrays[key])
        for view in view_configs:
            for value_name, values in result.bow_oof[view.name].items():
                bow_oof[view.name][value_name][
                    result.validation_positions
                ] = values
        for value_name, values in result.htr_oof.items():
            htr_oof[value_name][result.validation_positions] = values
        records.append(copy.deepcopy(dict(result.fold_record)))

    numerical_bank: dict[str, np.ndarray] = {}
    for view in view_configs:
        for value_name, values in bow_oof[view.name].items():
            numerical_bank[f"bow::{view.name}::{value_name}"] = values
    for value_name, values in htr_oof.items():
        numerical_bank[f"htr::{value_name}"] = values
    if any(
        values.shape != (len(frame),)
        for values in numerical_bank.values()
    ):
        raise RuntimeError("matched-pair fit numerical bank changed shape")
    execution_summary = _matched_pair_fold_execution_summary(
        events=fold_events,
        resource_plan=resource_plan,
        effect_folds=int(config["effect_folds"]),
    )
    return records, numerical_bank, tuple(fold_events), execution_summary


def _write_fit_state(
    *,
    root: Path,
    request: RoleNeutralMatchedPairPhysicalGroupRequest,
    fit_texts: tuple[str, ...],
    treatment: np.ndarray,
    outcome: np.ndarray,
    e_fit: np.ndarray,
    m_fit: np.ndarray,
    view_configs: tuple[BoWViewConfig, ...],
    config: Mapping[str, Any],
    fold_records: Sequence[Mapping[str, Any]],
    numerical_bank: Mapping[str, np.ndarray],
    subproducer_evidence: Mapping[str, Mapping[str, Any]],
    store: _ArrayStore,
) -> tuple[dict[str, Any], str, str]:
    fit_root = root / _FIT_STATE_DIRECTORY
    arrays_root = fit_root / "arrays"
    arrays_root.mkdir(parents=True, exist_ok=False)
    scope_inputs = {
        "treatment": store.add("scope_input_treatment", treatment),
        "outcome": store.add("scope_input_outcome", outcome),
        "e_fit": store.add("scope_input_e_fit", e_fit),
        "m_fit": store.add("scope_input_m_fit", m_fit),
    }
    bank_references = {
        name: store.add(f"fit_bank_{index:04d}", values)
        for index, (name, values) in enumerate(sorted(numerical_bank.items()))
    }
    inventory: dict[str, dict[str, Any]] = {}
    for key in sorted(store.arrays):
        path = arrays_root / f"{key}.npy"
        _write_new_npy(path, store.arrays[key])
        digest, size = _sha256_file(path)
        array_row = store.inventory[key]
        inventory[key] = {
            "relative_path": path.relative_to(fit_root).as_posix(),
            "dtype": array_row["dtype"],
            "shape": list(array_row["shape"]),
            "content_sha256": array_row["content_sha256"],
            "file_sha256": digest,
            "size_bytes": size,
        }
    scientific_configuration = {
        "matched_pair": copy.deepcopy(dict(config)),
        "bow_views": [_bow_view_to_dict(view) for view in view_configs],
        "outcome_type": "binary",
        "matching_input_scale": "probability",
        "scaling_state_policy": "no_separate_scaler_probability_inputs_v1",
        "text_truncation_applied": False,
        "top_k_evidence_applied": False,
    }
    configuration_identity = _sha256_json(scientific_configuration)
    evidence_identities = {
        subproducer: _sha256_json(subproducer_evidence[subproducer])
        for subproducer in _SUBPRODUCERS
    }
    body = {
        "schema_version": ROLE_NEUTRAL_MATCHED_PAIR_FIT_STATE_SCHEMA,
        "group_request_content_sha256": request.content_sha256,
        "scientific_plan_content_sha256": request.scientific_plan_content_sha256,
        "physical_owner_scope_id": request.physical_owner.scope_id,
        "physical_owner_scope_sha256": request.physical_owner.as_dict()["scope_sha256"],
        "fit_row_ids": list(request.physical_owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(
            request.physical_owner.fit_row_ids
        ),
        "canonical_group_seed": int(request.physical_owner.scope_seed),
        "fit_text_sha256": _text_sha256(
            request.physical_owner.fit_row_ids,
            fit_texts,
        ),
        "fit_treatment_sha256": _float_hex_sha256(treatment),
        "fit_outcome_sha256": _float_hex_sha256(outcome),
        "fit_propensity_nuisance_sha256": _float_hex_sha256(e_fit),
        "fit_outcome_nuisance_sha256": _float_hex_sha256(m_fit),
        "htr_model_identity_sha256": request.htr_model_identity_sha256,
        "nuisance_artifact_identity_sha256": (
            request.nuisance_artifact_identity_sha256
        ),
        "runtime_compatibility_class": request.runtime_compatibility_class,
        "producer_identity_sha256": request.producer_identity_sha256,
        "scientific_configuration": scientific_configuration,
        "configuration_identity_sha256": configuration_identity,
        "scope_inputs": scope_inputs,
        "fold_records": copy.deepcopy(list(fold_records)),
        "fit_numerical_bank": bank_references,
        "subproducer_evidence_identity_sha256": evidence_identities,
        "array_inventory": inventory,
        "array_layout": "one_npy_per_array_mmap_safe_v1",
        "subproducer_coverage": list(_SUBPRODUCERS),
        "model_objects_retained_in_worker_memory": False,
        "registered_heldout_text_accessed": False,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
        "top_k_evidence_applied": False,
        "pickle_joblib_npz_loaded_or_written": False,
    }
    metadata = {**body, "content_sha256": _sha256_json(body)}
    _write_new_json(fit_root / _FIT_METADATA, metadata)
    return metadata, _tree_sha256(fit_root), configuration_identity


def _fit_seal(
    *,
    request: RoleNeutralMatchedPairPhysicalGroupRequest,
    fit_state_sha256: str,
    configuration_identity_sha256: str,
    subproducer_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    proofs = []
    for subproducer in _SUBPRODUCERS:
        evidence = copy.deepcopy(dict(subproducer_evidence[subproducer]))
        proofs.append(
            {
                "subproducer": subproducer,
                "evidence_payload": evidence,
                "evidence_payload_sha256": _sha256_json(evidence),
                "fit_state_artifact_sha256": fit_state_sha256,
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
            }
        )
    events = [
        {
            "sequence": 1,
            "event": "fit_completed",
            "fit_state_artifact_sha256": fit_state_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        },
        *[
            {
                "sequence": index + 2,
                "event": "matched_pair_subproducer_sealed",
                "subproducer": subproducer,
                "fit_state_artifact_sha256": fit_state_sha256,
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
            }
            for index, subproducer in enumerate(_SUBPRODUCERS)
        ],
        {
            "sequence": len(_SUBPRODUCERS) + 2,
            "event": "fit_family_artifact_sealed",
            "family": MATCHED_PAIR_UPLIFT,
            "fit_state_artifact_sha256": fit_state_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        },
    ]
    body = {
        "schema_version": ROLE_NEUTRAL_MATCHED_PAIR_FIT_SEAL_SCHEMA,
        "scientific_plan_content_sha256": request.scientific_plan_content_sha256,
        "group_request_content_sha256": request.content_sha256,
        "physical_owner_scope_id": request.physical_owner.scope_id,
        "physical_owner_scope_sha256": request.physical_owner.as_dict()["scope_sha256"],
        "family": MATCHED_PAIR_UPLIFT,
        "fit_row_ids": list(request.physical_owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(
            request.physical_owner.fit_row_ids
        ),
        "canonical_group_seed": int(request.physical_owner.scope_seed),
        "htr_model_identity_sha256": request.htr_model_identity_sha256,
        "nuisance_artifact_identity_sha256": (
            request.nuisance_artifact_identity_sha256
        ),
        "runtime_compatibility_class": request.runtime_compatibility_class,
        "producer_identity_sha256": request.producer_identity_sha256,
        "configuration_identity_sha256": configuration_identity_sha256,
        "fit_state_artifact_sha256": fit_state_sha256,
        "subproducer_coverage": list(_SUBPRODUCERS),
        "subproducer_proofs": proofs,
        "event_order": events,
        "logical_view_transform_started": False,
        "registered_heldout_text_accessed": False,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
        "top_k_evidence_applied": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _load_fit_state(
    *,
    root: Path,
    request: RoleNeutralMatchedPairPhysicalGroupRequest,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fit_root = root / _FIT_STATE_DIRECTORY
    metadata = _read_json(
        fit_root / _FIT_METADATA,
        label="role-neutral matched-pair fit metadata",
    )
    body = {key: copy.deepcopy(value) for key, value in metadata.items() if key != "content_sha256"}
    if (
        metadata.get("schema_version") != ROLE_NEUTRAL_MATCHED_PAIR_FIT_STATE_SCHEMA
        or metadata.get("content_sha256") != _sha256_json(body)
        or metadata.get("group_request_content_sha256") != request.content_sha256
        or metadata.get("scientific_plan_content_sha256")
        != request.scientific_plan_content_sha256
        or metadata.get("physical_owner_scope_id")
        != request.physical_owner.scope_id
        or metadata.get("physical_owner_scope_sha256")
        != request.physical_owner.as_dict()["scope_sha256"]
        or metadata.get("fit_row_ids") != list(request.physical_owner.fit_row_ids)
        or metadata.get("fit_row_order_fingerprint")
        != _row_order_fingerprint(request.physical_owner.fit_row_ids)
        or metadata.get("canonical_group_seed")
        != int(request.physical_owner.scope_seed)
        or metadata.get("htr_model_identity_sha256")
        != request.htr_model_identity_sha256
        or metadata.get("nuisance_artifact_identity_sha256")
        != request.nuisance_artifact_identity_sha256
        or metadata.get("runtime_compatibility_class")
        != request.runtime_compatibility_class
        or metadata.get("producer_identity_sha256")
        != request.producer_identity_sha256
        or metadata.get("subproducer_coverage") != list(_SUBPRODUCERS)
        or metadata.get("registered_heldout_text_accessed") is not False
        or metadata.get("registered_heldout_labels_accessed") is not False
        or metadata.get("oracle_fields_accessed") is not False
        or metadata.get("text_truncation_applied") is not False
        or metadata.get("top_k_evidence_applied") is not False
        or metadata.get("pickle_joblib_npz_loaded_or_written") is not False
        or metadata.get("model_objects_retained_in_worker_memory") is not False
        or metadata.get("array_layout") != "one_npy_per_array_mmap_safe_v1"
    ):
        raise ValueError("role-neutral matched-pair fit-state envelope changed")
    configuration = metadata.get("scientific_configuration")
    if (
        not isinstance(configuration, Mapping)
        or metadata.get("configuration_identity_sha256")
        != _sha256_json(configuration)
    ):
        raise ValueError("matched-pair scientific configuration changed")
    inventory = metadata.get("array_inventory")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("matched-pair fit state has no closed numerical inventory")
    expected_files = {
        _FIT_METADATA,
        *{
            str(row.get("relative_path") or "")
            for row in inventory.values()
            if isinstance(row, Mapping)
        },
    }
    observed_files: set[str] = set()
    for path in fit_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("matched-pair fit state contains a symlink")
        if path.is_file():
            observed_files.add(path.relative_to(fit_root).as_posix())
    if observed_files != expected_files or any(
        not path or not path.startswith("arrays/") or not path.endswith(".npy")
        for path in expected_files - {_FIT_METADATA}
    ):
        raise ValueError("matched-pair fit-state file inventory is not closed")
    arrays: dict[str, np.ndarray] = {}
    for key in sorted(inventory):
        row = inventory[key]
        if not isinstance(row, Mapping):
            raise ValueError("matched-pair numerical registration is malformed")
        path = fit_root / str(row.get("relative_path") or "")
        digest, size = _sha256_file(path)
        if (
            row.get("file_sha256") != digest
            or int(row.get("size_bytes", -1)) != size
        ):
            raise ValueError(f"matched-pair numerical file changed: {key}")
        try:
            with path.open("rb") as handle:
                array = np.load(handle, allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError(f"matched-pair array is invalid: {key}") from exc
        if (
            array.dtype.hasobject
            or row.get("dtype") != array.dtype.str
            or row.get("shape") != list(array.shape)
            or row.get("content_sha256") != _array_sha256(array)
        ):
            raise ValueError(f"matched-pair array dtype/shape/content changed: {key}")
        arrays[str(key)] = np.array(array, copy=True)
    return metadata, arrays


def _validate_fit_side(
    *,
    root: Path,
    request: RoleNeutralMatchedPairPhysicalGroupRequest,
    expected_fit_texts: Sequence[str] | None = None,
    expected_treatment: np.ndarray | None = None,
    expected_outcome: np.ndarray | None = None,
    expected_e_fit: np.ndarray | None = None,
    expected_m_fit: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    metadata, arrays = _load_fit_state(root=root, request=request)
    scope_inputs = metadata.get("scope_inputs")
    if not isinstance(scope_inputs, Mapping) or set(scope_inputs) != {
        "treatment",
        "outcome",
        "e_fit",
        "m_fit",
    }:
        raise ValueError("matched-pair fit state lacks canonical scope inputs")
    inputs = {
        name: np.asarray(arrays[str(reference)], dtype=np.float64)
        for name, reference in scope_inputs.items()
    }
    length = len(request.physical_owner.fit_row_ids)
    treatment = _binary_vector(
        inputs["treatment"],
        label="stored fit treatment",
        length=length,
        require_both=True,
    )
    outcome = _binary_vector(
        inputs["outcome"],
        label="stored fit outcome",
        length=length,
        require_both=False,
    )
    e_fit = _finite_probability_vector(
        inputs["e_fit"],
        label="stored fit propensity",
        length=length,
    )
    m_fit = _finite_probability_vector(
        inputs["m_fit"],
        label="stored fit outcome nuisance",
        length=length,
    )
    expected_values = (
        expected_fit_texts,
        expected_treatment,
        expected_outcome,
        expected_e_fit,
        expected_m_fit,
    )
    if any(value is None for value in expected_values) and not all(
        value is None for value in expected_values
    ):
        raise ValueError("matched-pair expected fit inputs must be supplied together")
    if expected_fit_texts is not None:
        texts = tuple(expected_fit_texts)
        if (
            metadata.get("fit_text_sha256")
            != _text_sha256(request.physical_owner.fit_row_ids, texts)
            or not np.array_equal(treatment, np.asarray(expected_treatment))
            or not np.array_equal(outcome, np.asarray(expected_outcome))
            or not np.array_equal(e_fit, np.asarray(expected_e_fit))
            or not np.array_equal(m_fit, np.asarray(expected_m_fit))
        ):
            raise ValueError("matched-pair canonical fit inputs changed")
    if (
        metadata.get("fit_treatment_sha256") != _float_hex_sha256(treatment)
        or metadata.get("fit_outcome_sha256") != _float_hex_sha256(outcome)
        or metadata.get("fit_propensity_nuisance_sha256") != _float_hex_sha256(e_fit)
        or metadata.get("fit_outcome_nuisance_sha256") != _float_hex_sha256(m_fit)
    ):
        raise ValueError("matched-pair fit input identity changed")
    records = metadata.get("fold_records")
    raw_config = metadata["scientific_configuration"]["matched_pair"]
    typed_config = RoleNeutralMatchedPairConfig.from_mapping(raw_config)
    config = typed_config.as_dict()
    if config != raw_config:
        raise ValueError("matched-pair typed scientific configuration changed")
    views = metadata["scientific_configuration"]["bow_views"]
    if (
        not isinstance(records, list)
        or len(records) != int(config["effect_folds"])
        or [int(row.get("fold", -1)) for row in records]
        != list(range(1, int(config["effect_folds"]) + 1))
        or any(len(row.get("bow_states") or ()) != len(views) for row in records)
        or any(row.get("registered_heldout_text_accessed") is not False for row in records)
        or any(row.get("registered_heldout_labels_accessed") is not False for row in records)
    ):
        raise ValueError("matched-pair fold coverage changed")
    all_positions = set(range(length))
    seen_validation: set[int] = set()
    for row in records:
        fit_positions = tuple(map(int, row.get("fit_positions") or ()))
        validation_positions = tuple(map(int, row.get("validation_positions") or ()))
        if (
            set(fit_positions) & set(validation_positions)
            or set(fit_positions) | set(validation_positions) != all_positions
            or seen_validation & set(validation_positions)
            or row.get("fit_row_ids")
            != [request.physical_owner.fit_row_ids[position] for position in fit_positions]
            or row.get("validation_row_ids")
            != [
                request.physical_owner.fit_row_ids[position]
                for position in validation_positions
            ]
        ):
            raise ValueError("matched-pair fold partition/order changed")
        seen_validation.update(validation_positions)
        control_reference = str(row.get("control_positions") or "")
        control_positions = np.asarray(arrays[control_reference], dtype=np.int64)
        if (
            control_positions.ndim != 1
            or not set(control_positions.tolist()).issubset(set(fit_positions))
            or np.any(treatment[control_positions] != 0.0)
        ):
            raise ValueError("matched-pair fold control pool changed")
        for pair_name in ("fit_pair_table", "validation_pair_table"):
            pair_table = row.get(pair_name)
            if (
                not isinstance(pair_table, Mapping)
                or set(pair_table.get("columns") or ()) != {
                    "candidate_pos",
                    "control_pos",
                    "candidate_row_id",
                    "control_row_id",
                    "label",
                    "base_prob",
                    "base_logit",
                    "propensity_abs_diff",
                    "outcome_abs_diff",
                    "score_abs_diff_sum",
                }
            ):
                raise ValueError("matched-pair table registration changed")
            column_arrays = [
                arrays[str(reference)]
                for reference in pair_table["columns"].values()
            ]
            if any(
                np.asarray(value).shape != (int(pair_table["row_count"]),)
                for value in column_arrays
            ):
                raise ValueError("matched-pair table array shape changed")
        if [state.get("view_name") for state in row["bow_states"]] != [
            view["name"] for view in views
        ]:
            raise ValueError("matched-pair BoW view order changed")
        for state in row["bow_states"]:
            descriptor = state.get("model")
            body = {
                key: value
                for key, value in descriptor.items()
                if key != "state_sha256"
            }
            if descriptor.get("state_sha256") != _sha256_json(body):
                raise ValueError("matched-pair BoW model descriptor changed")
            expected_optimizer = {
                "method": config["bow_optimizer_method"],
                "ftol": config["bow_optimizer_ftol"],
                "gtol": config["bow_optimizer_gtol"],
                "maxls": config["bow_optimizer_maxls"],
                "maxcor": config["bow_optimizer_maxcor"],
                "maxfun": config["bow_optimizer_maxfun"],
                "tol": config["bow_optimizer_tol"],
                "initialization": config["bow_optimizer_initialization"],
                "require_success": config["bow_require_optimizer_success"],
                "jacobian": "analytic",
            }
            if descriptor.get("optimizer") != expected_optimizer:
                raise ValueError(
                    "matched-pair BoW optimizer descriptor changed"
                )
        htr = row.get("htr_model")
        htr_body = {
            key: value for key, value in htr.items() if key != "state_sha256"
        }
        if (
            htr.get("state_sha256") != _sha256_json(htr_body)
            or htr.get("kind") != "htr_pair_network"
            or (htr.get("extractor") or {}).get("constructor")
            != config["htr_extractor"]
            or htr.get("head_configuration")
            != {
                "hidden_dim": config["htr_hidden_dim"],
                "depth": config["htr_head_depth"],
                "activation": config["htr_head_activation"],
                "dropout": config["htr_dropout"],
                "layer_norm": config["htr_head_layer_norm"],
                "bias": config["htr_head_bias"],
            }
            or htr.get("training_configuration")
            != _htr_training_configuration(config)
        ):
            raise ValueError("matched-pair HTR model descriptor changed")
    if seen_validation != all_positions:
        raise ValueError("matched-pair OOF validation coverage changed")
    fit_bank = metadata.get("fit_numerical_bank")
    expected_bank_names = {
        *(
            f"bow::{view['name']}::{value}"
            for view in views
            for value in ("delta", "probability", "n_controls")
        ),
        *(f"htr::{value}" for value in ("delta", "probability", "n_controls")),
    }
    if not isinstance(fit_bank, Mapping) or set(fit_bank) != expected_bank_names:
        raise ValueError("matched-pair fit numerical bank coverage changed")
    if any(np.asarray(arrays[str(reference)]).shape != (length,) for reference in fit_bank.values()):
        raise ValueError("matched-pair fit numerical bank shape changed")
    seal = _read_json(root / _FIT_SEAL, label="matched-pair fit-only family seal")
    seal_body = {key: copy.deepcopy(value) for key, value in seal.items() if key != "content_sha256"}
    if (
        seal.get("schema_version") != ROLE_NEUTRAL_MATCHED_PAIR_FIT_SEAL_SCHEMA
        or seal.get("content_sha256") != _sha256_json(seal_body)
        or seal.get("scientific_plan_content_sha256")
        != request.scientific_plan_content_sha256
        or seal.get("group_request_content_sha256") != request.content_sha256
        or seal.get("family") != MATCHED_PAIR_UPLIFT
        or seal.get("fit_state_artifact_sha256") != _tree_sha256(root / _FIT_STATE_DIRECTORY)
        or seal.get("subproducer_coverage") != list(_SUBPRODUCERS)
        or [row.get("subproducer") for row in seal.get("subproducer_proofs") or ()]
        != list(_SUBPRODUCERS)
        or seal.get("registered_heldout_text_accessed") is not False
        or seal.get("registered_heldout_labels_accessed") is not False
        or seal.get("text_truncation_applied") is not False
        or seal.get("top_k_evidence_applied") is not False
    ):
        raise ValueError("matched-pair fit-only seal changed")
    for proof in seal["subproducer_proofs"]:
        evidence = proof.get("evidence_payload")
        subproducer = proof["subproducer"]
        if (
            not isinstance(evidence, Mapping)
            or not evidence.get("atoms")
            or evidence.get("top_k_applied") is not False
            or evidence.get("text_truncation_applied") is not False
            or proof.get("evidence_payload_sha256") != _sha256_json(evidence)
            or metadata["subproducer_evidence_identity_sha256"][subproducer]
            != proof["evidence_payload_sha256"]
        ):
            raise ValueError("matched-pair subproducer evidence changed")
    expected_seal_events = [
        "fit_completed",
        "matched_pair_subproducer_sealed",
        "matched_pair_subproducer_sealed",
        "fit_family_artifact_sealed",
    ]
    if (
        [row.get("event") for row in seal.get("event_order") or ()]
        != expected_seal_events
        or [row.get("sequence") for row in seal["event_order"]]
        != list(range(1, len(expected_seal_events) + 1))
        or [
            row.get("subproducer")
            for row in seal["event_order"]
            if row.get("event") == "matched_pair_subproducer_sealed"
        ]
        != list(_SUBPRODUCERS)
    ):
        raise ValueError("matched-pair fit-only seal event order changed")
    return metadata, arrays, seal


def _candidate_pairs_for_exact(
    *,
    request: RoleNeutralMatchedPairPhysicalGroupRequest,
    fit_texts: tuple[str, ...],
    exact_input: RoleNeutralMatchedPairExactInput,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    fold_row: Mapping[str, Any],
) -> pd.DataFrame:
    exact_input.validated(request.physical_owner.heldout_row_ids)
    fit_inputs = metadata["scope_inputs"]
    e_fit = np.asarray(arrays[str(fit_inputs["e_fit"])], dtype=np.float64)
    m_fit = np.asarray(arrays[str(fit_inputs["m_fit"])], dtype=np.float64)
    control_pos = np.asarray(
        arrays[str(fold_row["control_positions"])],
        dtype=np.int64,
    )
    frame = _make_frame(request.physical_owner.fit_row_ids)
    exact_frame = _make_frame(exact_input.row_ids)
    config = metadata["scientific_configuration"]["matched_pair"]
    return build_candidate_pairs(
        exact_frame,
        frame.iloc[control_pos].reset_index(drop=True),
        candidate_texts=exact_input.texts,
        control_texts=[fit_texts[int(position)] for position in control_pos],
        candidate_propensity=np.asarray(
            exact_input.propensity_probability,
            dtype=np.float64,
        ),
        candidate_outcome_prob=np.asarray(
            exact_input.outcome_nuisance_probability,
            dtype=np.float64,
        ),
        control_propensity=e_fit[control_pos],
        control_outcome_prob=m_fit[control_pos],
        **_matching_candidate_config(config),
    )


def _sealed_exact_predictions(
    *,
    request: RoleNeutralMatchedPairPhysicalGroupRequest,
    fit_texts: tuple[str, ...],
    exact_input: RoleNeutralMatchedPairExactInput,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    htr_model_path: Path | str | None,
    device: torch.device,
) -> dict[str, tuple[list[str], np.ndarray]]:
    config = metadata["scientific_configuration"]["matched_pair"]
    views = metadata["scientific_configuration"]["bow_views"]
    count = len(exact_input.row_ids)
    bow_values: dict[str, dict[str, list[np.ndarray]]] = {
        view["name"]: {
            "delta": [],
            "probability": [],
            "n_controls": [],
        }
        for view in views
    }
    htr_values = {"delta": [], "probability": [], "n_controls": []}
    for fold_row in metadata["fold_records"]:
        pairs = _candidate_pairs_for_exact(
            request=request,
            fit_texts=fit_texts,
            exact_input=exact_input,
            metadata=metadata,
            arrays=arrays,
            fold_row=fold_row,
        )
        for bow_state in fold_row["bow_states"]:
            pair_delta = _predict_offset_model(
                bow_state["model"],
                arrays,
                pairs,
            )
            delta, probability, n_controls = aggregate_pair_predictions(
                pairs,
                pair_delta,
                count,
            )
            values = bow_values[bow_state["view_name"]]
            values["delta"].append(delta)
            values["probability"].append(probability)
            values["n_controls"].append(n_controls)
        model = _build_htr_pair_model(
            fold_row["htr_model"],
            arrays,
            initialization_texts=fit_texts,
            htr_model_path=htr_model_path,
            device=device,
        )
        try:
            pair_delta = _predict_htr_pair(
                model,
                pairs,
                batch_size=int(config["htr_batch_size"]),
            )
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        delta, probability, n_controls = aggregate_pair_predictions(
            pairs,
            pair_delta,
            count,
        )
        htr_values["delta"].append(delta)
        htr_values["probability"].append(probability)
        htr_values["n_controls"].append(n_controls)
    bow_columns: list[str] = []
    bow_arrays: list[np.ndarray] = []
    for view in views:
        name = view["name"]
        for value_name in ("delta", "probability", "n_controls"):
            bow_columns.append(f"{name}::{value_name}")
            bow_arrays.append(
                _mean_with_nan(bow_values[name][value_name], length=count)
            )
    htr_columns = [f"htr::{name}" for name in ("delta", "probability", "n_controls")]
    htr_arrays = [
        _mean_with_nan(htr_values[name], length=count)
        for name in ("delta", "probability", "n_controls")
    ]
    return {
        "bow": (bow_columns, np.column_stack(bow_arrays).astype(np.float64)),
        "htr": (htr_columns, np.column_stack(htr_arrays).astype(np.float64)),
    }


def execute_role_neutral_matched_pair_physical_group(
    *,
    request: RoleNeutralMatchedPairPhysicalGroupRequest,
    output_root: Path | str,
    fit_texts: Sequence[str],
    fit_treatment: Sequence[Any],
    fit_outcome: Sequence[Any],
    fit_propensity_probability: Sequence[Any],
    fit_outcome_nuisance_probability: Sequence[Any],
    view_configs: Sequence[BoWViewConfig],
    config: RoleNeutralMatchedPairConfig,
    htr_extractor_factory: Callable[
        [torch.device],
        HierarchicalTransformerExtractor,
    ],
    exact_heldout_input_loader: Callable[
        [tuple[int, ...]],
        RoleNeutralMatchedPairExactInput,
    ],
    device: torch.device | str,
    htr_model_path: Path | str | None = None,
    fold_resource_plan: RoleNeutralHTRFoldResourcePlan | None = None,
    operational_attestation_sink: (
        Callable[[Mapping[str, Any]], None] | None
    ) = None,
    fold_event_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> Mapping[str, Any]:
    """Fit/seal both subproducers before opening exact held-out text."""

    if not isinstance(request, RoleNeutralMatchedPairPhysicalGroupRequest):
        raise TypeError("matched-pair execution requires its typed request")
    request.as_dict()
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("matched-pair output root must be absolute")
    if root.exists() or root.is_symlink():
        raise FileExistsError("matched-pair output root must be fresh")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    owner = request.physical_owner
    texts = tuple(fit_texts)
    if len(texts) != len(owner.fit_row_ids) or any(
        not isinstance(text, str) for text in texts
    ):
        raise ValueError("matched-pair fit texts do not align to physical fit rows")
    configuration = config.as_dict()
    _validate_htr_model_locator(
        request=request,
        config=configuration,
        htr_model_path=htr_model_path,
    )
    if int(configuration["effect_folds"]) > len(texts):
        raise ValueError("configured matched-pair effect folds exceed fit rows")
    _assert_text_capacity(
        texts,
        extractor_config=configuration["htr_extractor"],
        stage="fit",
    )
    treatment = _binary_vector(
        fit_treatment,
        label="fit treatment",
        length=len(texts),
        require_both=True,
    )
    outcome = _binary_vector(
        fit_outcome,
        label="fit outcome",
        length=len(texts),
        require_both=False,
    )
    e_fit = _finite_probability_vector(
        fit_propensity_probability,
        label="fit propensity nuisance",
        length=len(texts),
    )
    m_fit = _finite_probability_vector(
        fit_outcome_nuisance_probability,
        label="fit outcome nuisance",
        length=len(texts),
    )
    views = tuple(view_configs)
    if (
        not views
        or any(type(view) is not BoWViewConfig for view in views)
        or len({view.name for view in views}) != len(views)
        or any(not view.name for view in views)
    ):
        raise ValueError("matched-pair execution requires unique typed BoW views")
    if not callable(htr_extractor_factory) or not callable(exact_heldout_input_loader):
        raise TypeError("matched-pair execution requires callable factory/loaders")
    # The legacy factory remains a caller-compatibility argument. Fold tasks
    # deliberately carry only the authenticated typed constructor and model
    # tree locator, because deployment factories are commonly closures and
    # therefore cannot be part of a spawn-pickleable task.
    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("matched-pair execution device must be CPU or CUDA")
    # The serial executor runs in this process, so establish the same strict
    # policy before its one worker can initialize a model or CUDA. Spawned
    # workers independently re-establish and attest this policy inside the
    # shared HTR fold executor.
    _enforce_stage1_torch_determinism()
    if fold_event_sink is not None and not callable(fold_event_sink):
        raise TypeError("matched-pair fold event sink must be callable")
    if fold_resource_plan is None:
        if operational_attestation_sink is not None:
            raise ValueError(
                "matched-pair operational attestation requires an explicit "
                "HTR fold resource plan"
            )
        effective_resource_plan = RoleNeutralHTRFoldResourcePlan(
            devices=(str(resolved_device),),
            fold_parallelism=1,
            fold_slots_per_device=1,
            owner_cpu_budget=1,
            fold_parallel_backend="threads",
        )
    else:
        if not isinstance(
            fold_resource_plan,
            RoleNeutralHTRFoldResourcePlan,
        ):
            raise TypeError(
                "matched-pair fold resources require the typed HTR plan"
            )
        if not callable(operational_attestation_sink):
            raise TypeError(
                "matched-pair fold resources require an attestation sink"
            )
        effective_resource_plan = fold_resource_plan
        if str(resolved_device) not in effective_resource_plan.devices:
            raise ValueError(
                "matched-pair fold resource plan omits the primary device"
            )
        if (
            len(effective_resource_plan.devices)
            > int(configuration["effect_folds"])
        ):
            raise ValueError(
                "matched-pair selected devices exceed available effect folds"
            )
        if (
            effective_resource_plan.fold_parallelism > 1
            and effective_resource_plan.fold_parallel_backend
            != "processes"
        ):
            raise ValueError(
                "overlapping matched-pair folds require process-isolated RNG"
            )
    store = _ArrayStore()
    (
        fold_records,
        numerical_bank,
        fold_execution_events,
        fold_execution_summary,
    ) = _fit_models(
        request=request,
        fit_texts=texts,
        treatment=treatment,
        outcome=outcome,
        e_fit=e_fit,
        m_fit=m_fit,
        view_configs=views,
        config=configuration,
        device=resolved_device,
        htr_model_path=htr_model_path,
        store=store,
        resource_plan=effective_resource_plan,
        external_event_sink=fold_event_sink,
    )
    subproducer_evidence = _subproducer_evidence(
        fold_records=fold_records,
        store=store,
    )
    _metadata, fit_state_sha256, configuration_identity = _write_fit_state(
        root=root,
        request=request,
        fit_texts=texts,
        treatment=treatment,
        outcome=outcome,
        e_fit=e_fit,
        m_fit=m_fit,
        view_configs=views,
        config=configuration,
        fold_records=fold_records,
        numerical_bank=numerical_bank,
        subproducer_evidence=subproducer_evidence,
        store=store,
    )
    seal = _fit_seal(
        request=request,
        fit_state_sha256=fit_state_sha256,
        configuration_identity_sha256=configuration_identity,
        subproducer_evidence=subproducer_evidence,
    )
    _write_new_json(root / _FIT_SEAL, seal)
    seal_sha256, seal_size = _sha256_file(root / _FIT_SEAL)
    _validate_fit_side(
        root=root,
        request=request,
        expected_fit_texts=texts,
        expected_treatment=treatment,
        expected_outcome=outcome,
        expected_e_fit=e_fit,
        expected_m_fit=m_fit,
    )
    if operational_attestation_sink is not None:
        operational_body = {
            "schema_version": (
                ROLE_NEUTRAL_MATCHED_PAIR_OPERATIONAL_ATTESTATION_SCHEMA
            ),
            "fold_resource_plan": effective_resource_plan.as_dict(),
            "fold_execution": copy.deepcopy(
                dict(fold_execution_summary)
            ),
            "fold_execution_events": [
                copy.deepcopy(dict(value))
                for value in fold_execution_events
            ],
            "canonical_fold_result_merge_order": (
                "matched_pair_effect_fold_numeric_order_v1"
            ),
            "shared_mutable_array_store_used_by_fold_workers": False,
            "live_models_returned_across_fold_boundary": False,
            "worker_private_pair_tables_and_model_state": True,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
            "resource_locators_in_scientific_identity": False,
            "resource_telemetry_persisted_in_scientific_artifact": False,
        }
        operational_attestation_sink(
            {
                **operational_body,
                "content_sha256": _sha256_json(operational_body),
            }
        )
    logical_root = root / _LOGICAL_DIRECTORY
    logical_root.mkdir(parents=True, exist_ok=False)
    events: list[dict[str, Any]] = [
        {
            "sequence": 1,
            "event": "fit_completed",
            "fit_state_artifact_sha256": fit_state_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        },
        *[
            {
                "sequence": index + 2,
                "event": "matched_pair_subproducer_sealed",
                "subproducer": subproducer,
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
            }
            for index, subproducer in enumerate(_SUBPRODUCERS)
        ],
        {
            "sequence": len(_SUBPRODUCERS) + 2,
            "event": "fit_family_artifact_sealed",
            "family": MATCHED_PAIR_UPLIFT,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        },
    ]
    registrations: list[dict[str, Any]] = []
    for member in request.logical_members[1:]:
        if member.scope_kind != "cumulative_spent":
            raise RuntimeError("matched-pair logical alias changed purpose")
        body = {
            "schema_version": ROLE_NEUTRAL_MATCHED_PAIR_LOGICAL_VIEW_SCHEMA,
            "scientific_plan_content_sha256": (
                request.scientific_plan_content_sha256
            ),
            "group_request_content_sha256": request.content_sha256,
            "logical_scope_id": member.scope_id,
            "logical_scope_sha256": member.as_dict()["scope_sha256"],
            "logical_purpose": member.scope_kind,
            "physical_owner_scope_id": owner.scope_id,
            "family": MATCHED_PAIR_UPLIFT,
            "subproducer_coverage": list(_SUBPRODUCERS),
            "fit_only_family_seal_sha256": seal_sha256,
            "fit_only_family_seal_content_sha256": seal["content_sha256"],
            "view_input_policy": "sealed_row_ids_only_no_sealed_text_or_labels_v1",
            "logical_heldout_row_ids": list(member.heldout_row_ids),
            "logical_transform_performed": False,
            "prediction_artifacts": None,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
            "reuses_live_physical_fit": True,
        }
        view = {**body, "content_sha256": _sha256_json(body)}
        path = logical_root / f"{member.scope_id}.json"
        _write_new_json(path, view)
        digest, size = _sha256_file(path)
        registrations.append(
            {
                "logical_scope_id": member.scope_id,
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "size_bytes": size,
                "content_sha256": view["content_sha256"],
            }
        )
        events.append(
            {
                "sequence": len(events) + 1,
                "event": "cumulative_fit_only_view_published",
                "logical_scope_id": member.scope_id,
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
            }
        )
    loaded = exact_heldout_input_loader(tuple(owner.heldout_row_ids))
    if type(loaded) is not RoleNeutralMatchedPairExactInput:
        raise TypeError("exact matched-pair loader returned an open or untyped payload")
    exact_input = loaded
    exact_identity = exact_input.validated(owner.heldout_row_ids)
    _assert_text_capacity(
        exact_input.texts,
        extractor_config=configuration["htr_extractor"],
        stage="exact held-out",
    )
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "exact_heldout_text_opened",
            "logical_scope_id": owner.scope_id,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
        }
    )
    metadata, arrays, _validated_seal = _validate_fit_side(
        root=root,
        request=request,
    )
    sealed_predictions = _sealed_exact_predictions(
        request=request,
        fit_texts=texts,
        exact_input=exact_input,
        metadata=metadata,
        arrays=arrays,
        htr_model_path=htr_model_path,
        device=resolved_device,
    )
    prediction_registrations: dict[str, dict[str, Any]] = {}
    for subproducer in _SUBPRODUCERS:
        replay_columns, replay_matrix = sealed_predictions[subproducer]
        prediction_path = logical_root / f"{owner.scope_id}.{subproducer}.predictions.npy"
        _write_new_npy(prediction_path, replay_matrix)
        digest, size = _sha256_file(prediction_path)
        prediction_registrations[subproducer] = {
            "relative_path": prediction_path.relative_to(root).as_posix(),
            "sha256": digest,
            "size_bytes": size,
            "dtype": replay_matrix.dtype.str,
            "shape": list(replay_matrix.shape),
            "content_sha256": _array_sha256(replay_matrix),
            "columns": replay_columns,
        }
        events.append(
            {
                "sequence": len(events) + 1,
                "event": "exact_heldout_transform_completed",
                "logical_scope_id": owner.scope_id,
                "subproducer": subproducer,
                "registered_heldout_text_accessed": True,
                "registered_heldout_labels_accessed": False,
            }
        )
    exact_body = {
        "schema_version": ROLE_NEUTRAL_MATCHED_PAIR_LOGICAL_VIEW_SCHEMA,
        "scientific_plan_content_sha256": request.scientific_plan_content_sha256,
        "group_request_content_sha256": request.content_sha256,
        "logical_scope_id": owner.scope_id,
        "logical_scope_sha256": owner.as_dict()["scope_sha256"],
        "logical_purpose": owner.scope_kind,
        "physical_owner_scope_id": owner.scope_id,
        "family": MATCHED_PAIR_UPLIFT,
        "subproducer_coverage": list(_SUBPRODUCERS),
        "fit_only_family_seal_sha256": seal_sha256,
        "fit_only_family_seal_content_sha256": seal["content_sha256"],
        "view_input_policy": (
            "authorized_row_text_and_authenticated_nuisance_no_labels_v1"
        ),
        "logical_heldout_row_ids": list(owner.heldout_row_ids),
        "exact_input_identity": exact_identity,
        "logical_transform_performed": True,
        "prediction_artifacts": prediction_registrations,
        "registered_heldout_text_accessed": True,
        "registered_heldout_labels_accessed": False,
        "reuses_live_physical_fit": True,
        "model_state_reloaded_for_primary_transform": True,
        "sealed_state_replay_checked": True,
    }
    exact_view = {**exact_body, "content_sha256": _sha256_json(exact_body)}
    exact_path = logical_root / f"{owner.scope_id}.json"
    _write_new_json(exact_path, exact_view)
    exact_digest, exact_size = _sha256_file(exact_path)
    registrations.append(
        {
            "logical_scope_id": owner.scope_id,
            "relative_path": exact_path.relative_to(root).as_posix(),
            "sha256": exact_digest,
            "size_bytes": exact_size,
            "content_sha256": exact_view["content_sha256"],
        }
    )
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "exact_logical_view_published",
            "logical_scope_id": owner.scope_id,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
        }
    )
    registrations.sort(
        key=lambda row: next(
            index
            for index, member in enumerate(request.logical_members)
            if member.scope_id == row["logical_scope_id"]
        )
    )
    terminal_body = {
        "schema_version": ROLE_NEUTRAL_MATCHED_PAIR_GROUP_EXECUTION_SCHEMA,
        "status": "complete",
        "group_request": request.as_dict(),
        "family": MATCHED_PAIR_UPLIFT,
        "subproducer_coverage": list(_SUBPRODUCERS),
        "fit_state_artifact_sha256": fit_state_sha256,
        "fit_only_family_seal": {
            "relative_path": _FIT_SEAL,
            "sha256": seal_sha256,
            "size_bytes": seal_size,
            "content_sha256": seal["content_sha256"],
        },
        "logical_views": registrations,
        "event_order": events,
        "fit_completed_before_registered_heldout_text_access": True,
        "both_subproducers_sealed_before_registered_heldout_text_access": True,
        "cumulative_views_published_without_sealed_text": True,
        "live_model_objects_reused_for_exact_transform": False,
        "model_state_reloaded_for_primary_transform": True,
        "sealed_state_replay_checked": True,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
        "top_k_evidence_applied": False,
        "pickle_joblib_npz_loaded_or_written": False,
    }
    terminal = {**terminal_body, "content_sha256": _sha256_json(terminal_body)}
    _write_new_json(root / _TERMINAL, terminal)
    return validate_role_neutral_matched_pair_group_execution(
        root=root,
        request=request,
    )


def execute_role_neutral_matched_pair_from_bow_nuisance_bank(
    *,
    request: RoleNeutralMatchedPairPhysicalGroupRequest,
    output_root: Path | str,
    fit_texts: Sequence[str],
    fit_treatment: Sequence[Any],
    fit_outcome: Sequence[Any],
    nuisance_bank: AuthenticatedRoleNeutralBoWNuisanceBank,
    view_configs: Sequence[BoWViewConfig],
    config: RoleNeutralMatchedPairConfig,
    htr_extractor_factory: Callable[
        [torch.device],
        HierarchicalTransformerExtractor,
    ],
    exact_heldout_text_loader: Callable[
        [tuple[int, ...]],
        Sequence[str],
    ],
    device: torch.device | str,
    htr_model_path: Path | str | None = None,
    fold_resource_plan: RoleNeutralHTRFoldResourcePlan | None = None,
    operational_attestation_sink: (
        Callable[[Mapping[str, Any]], None] | None
    ) = None,
    fold_event_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> Mapping[str, Any]:
    """Execute matched-pair uplift from the authenticated prior BoW component.

    The wrapper accepts only held-out text.  It constructs the closed exact
    input from the nuisance bank, so neither a factory nor a deployment hook
    can accidentally pass held-out treatment/outcome labels into this
    producer.
    """

    if type(nuisance_bank) is not AuthenticatedRoleNeutralBoWNuisanceBank:
        raise TypeError(
            "matched-pair execution requires the authenticated BoW nuisance bank"
        )
    nuisance_bank.as_dict()
    owner = request.physical_owner
    if (
        request.scientific_plan_content_sha256
        != nuisance_bank.plan_scientific_content_sha256
        or request.nuisance_artifact_identity_sha256
        != nuisance_bank.content_sha256
        or owner.scope_id != nuisance_bank.physical_owner_scope_id
        or tuple(owner.fit_row_ids) != nuisance_bank.fit_row_ids
        or tuple(owner.heldout_row_ids) != nuisance_bank.heldout_row_ids
    ):
        raise ValueError(
            "matched-pair request and authenticated BoW nuisance bank differ"
        )
    if not callable(exact_heldout_text_loader):
        raise TypeError("matched-pair held-out text loader must be callable")

    def exact_input_loader(
        row_ids: tuple[int, ...],
    ) -> RoleNeutralMatchedPairExactInput:
        expected = tuple(owner.heldout_row_ids)
        if tuple(map(int, row_ids)) != expected:
            raise ValueError(
                "matched-pair producer requested another held-out row order"
            )
        texts = tuple(exact_heldout_text_loader(expected))
        if (
            len(texts) != len(expected)
            or any(not isinstance(text, str) for text in texts)
        ):
            raise ValueError(
                "matched-pair held-out text loader changed its row alignment"
            )
        return RoleNeutralMatchedPairExactInput(
            row_ids=expected,
            texts=texts,
            propensity_probability=(
                nuisance_bank.heldout_propensity_probability
            ),
            outcome_nuisance_probability=(
                nuisance_bank.heldout_outcome_nuisance_probability
            ),
        )

    return execute_role_neutral_matched_pair_physical_group(
        request=request,
        output_root=output_root,
        fit_texts=fit_texts,
        fit_treatment=fit_treatment,
        fit_outcome=fit_outcome,
        fit_propensity_probability=(
            nuisance_bank.fit_propensity_probability
        ),
        fit_outcome_nuisance_probability=(
            nuisance_bank.fit_outcome_nuisance_probability
        ),
        view_configs=view_configs,
        config=config,
        htr_extractor_factory=htr_extractor_factory,
        exact_heldout_input_loader=exact_input_loader,
        device=device,
        htr_model_path=htr_model_path,
        fold_resource_plan=fold_resource_plan,
        operational_attestation_sink=operational_attestation_sink,
        fold_event_sink=fold_event_sink,
    )


def _load_prediction(
    *,
    root: Path,
    registration: Mapping[str, Any],
    expected_rows: int,
) -> np.ndarray:
    path = root / str(registration.get("relative_path") or "")
    digest, size = _sha256_file(path)
    if (
        registration.get("sha256") != digest
        or int(registration.get("size_bytes", -1)) != size
    ):
        raise ValueError("matched-pair prediction file changed")
    try:
        with path.open("rb") as handle:
            array = np.load(handle, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError("matched-pair prediction array is invalid") from exc
    if (
        array.dtype.hasobject
        or registration.get("dtype") != array.dtype.str
        or registration.get("shape") != list(array.shape)
        or registration.get("content_sha256") != _array_sha256(array)
        or array.ndim != 2
        or array.shape[0] != int(expected_rows)
        or array.shape[1] != len(registration.get("columns") or ())
    ):
        raise ValueError("matched-pair prediction dtype/shape/content changed")
    return np.array(array, copy=True)


def validate_role_neutral_matched_pair_group_execution(
    *,
    root: Path | str,
    request: RoleNeutralMatchedPairPhysicalGroupRequest,
) -> Mapping[str, Any]:
    """Fresh path-only structural authentication of a completed group."""

    if not isinstance(request, RoleNeutralMatchedPairPhysicalGroupRequest):
        raise TypeError("matched-pair validation requires its typed request")
    request.as_dict()
    artifact_root = Path(root)
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ValueError("matched-pair execution root must be one real directory")
    terminal = _read_json(
        artifact_root / _TERMINAL,
        label="matched-pair execution manifest",
    )
    body = {
        key: copy.deepcopy(value)
        for key, value in terminal.items()
        if key != "content_sha256"
    }
    if (
        terminal.get("schema_version")
        != ROLE_NEUTRAL_MATCHED_PAIR_GROUP_EXECUTION_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("content_sha256") != _sha256_json(body)
        or terminal.get("group_request") != request.as_dict()
        or terminal.get("family") != MATCHED_PAIR_UPLIFT
        or terminal.get("subproducer_coverage") != list(_SUBPRODUCERS)
        or terminal.get("fit_state_artifact_sha256")
        != _tree_sha256(artifact_root / _FIT_STATE_DIRECTORY)
        or terminal.get("fit_completed_before_registered_heldout_text_access")
        is not True
        or terminal.get(
            "both_subproducers_sealed_before_registered_heldout_text_access"
        )
        is not True
        or terminal.get("cumulative_views_published_without_sealed_text") is not True
        or terminal.get("live_model_objects_reused_for_exact_transform") is not False
        or terminal.get("model_state_reloaded_for_primary_transform") is not True
        or terminal.get("sealed_state_replay_checked") is not True
        or terminal.get("registered_heldout_labels_accessed") is not False
        or terminal.get("oracle_fields_accessed") is not False
        or terminal.get("text_truncation_applied") is not False
        or terminal.get("top_k_evidence_applied") is not False
        or terminal.get("pickle_joblib_npz_loaded_or_written") is not False
    ):
        raise ValueError("matched-pair terminal execution envelope changed")
    metadata, _arrays, seal = _validate_fit_side(
        root=artifact_root,
        request=request,
    )
    seal_registration = terminal.get("fit_only_family_seal")
    if not isinstance(seal_registration, Mapping):
        raise ValueError("matched-pair terminal lacks its fit seal")
    seal_path = artifact_root / str(seal_registration.get("relative_path") or "")
    seal_digest, seal_size = _sha256_file(seal_path)
    if (
        seal_registration.get("relative_path") != _FIT_SEAL
        or seal_registration.get("sha256") != seal_digest
        or int(seal_registration.get("size_bytes", -1)) != seal_size
        or seal_registration.get("content_sha256") != seal["content_sha256"]
    ):
        raise ValueError("matched-pair fit-seal registration changed")
    logical_rows = terminal.get("logical_views")
    if (
        not isinstance(logical_rows, list)
        or len(logical_rows) != len(request.logical_members)
        or [row.get("logical_scope_id") for row in logical_rows]
        != [member.scope_id for member in request.logical_members]
    ):
        raise ValueError("matched-pair logical-view registration order changed")
    registered_files = {
        _TERMINAL,
        _FIT_SEAL,
        f"{_FIT_STATE_DIRECTORY}/{_FIT_METADATA}",
        *{
            f"{_FIT_STATE_DIRECTORY}/{row['relative_path']}"
            for row in metadata["array_inventory"].values()
        },
    }
    exact_seen = False
    for member, registration in zip(
        request.logical_members,
        logical_rows,
        strict=True,
    ):
        path = artifact_root / str(registration.get("relative_path") or "")
        digest, size = _sha256_file(path)
        view = _read_json(path, label=f"matched-pair logical view {member.scope_id}")
        view_body = {
            key: copy.deepcopy(value)
            for key, value in view.items()
            if key != "content_sha256"
        }
        if (
            registration.get("relative_path")
            != f"{_LOGICAL_DIRECTORY}/{member.scope_id}.json"
            or registration.get("sha256") != digest
            or int(registration.get("size_bytes", -1)) != size
            or registration.get("content_sha256") != view.get("content_sha256")
            or view.get("content_sha256") != _sha256_json(view_body)
            or view.get("schema_version")
            != ROLE_NEUTRAL_MATCHED_PAIR_LOGICAL_VIEW_SCHEMA
            or view.get("scientific_plan_content_sha256")
            != request.scientific_plan_content_sha256
            or view.get("group_request_content_sha256") != request.content_sha256
            or view.get("logical_scope_id") != member.scope_id
            or view.get("logical_scope_sha256")
            != member.as_dict()["scope_sha256"]
            or view.get("logical_purpose") != member.scope_kind
            or view.get("physical_owner_scope_id")
            != request.physical_owner.scope_id
            or view.get("family") != MATCHED_PAIR_UPLIFT
            or view.get("subproducer_coverage") != list(_SUBPRODUCERS)
            or view.get("fit_only_family_seal_sha256") != seal_digest
            or view.get("fit_only_family_seal_content_sha256")
            != seal["content_sha256"]
            or view.get("registered_heldout_labels_accessed") is not False
        ):
            raise ValueError("matched-pair logical view changed")
        registered_files.add(str(registration["relative_path"]))
        if member.scope_id == request.physical_owner.scope_id:
            exact_seen = True
            exact_identity = view.get("exact_input_identity")
            predictions = view.get("prediction_artifacts")
            if (
                view.get("logical_transform_performed") is not True
                or view.get("registered_heldout_text_accessed") is not True
                or view.get("view_input_policy")
                != "authorized_row_text_and_authenticated_nuisance_no_labels_v1"
                or view.get("model_state_reloaded_for_primary_transform")
                is not True
                or view.get("sealed_state_replay_checked") is not True
                or not isinstance(exact_identity, Mapping)
                or exact_identity.get("row_ids") != list(member.heldout_row_ids)
                or exact_identity.get("heldout_treatment_field_present") is not False
                or exact_identity.get("heldout_outcome_field_present") is not False
                or not isinstance(predictions, Mapping)
                or list(predictions) != list(_SUBPRODUCERS)
            ):
                raise ValueError("matched-pair exact logical view changed")
            for subproducer in _SUBPRODUCERS:
                prediction = predictions[subproducer]
                _load_prediction(
                    root=artifact_root,
                    registration=prediction,
                    expected_rows=len(member.heldout_row_ids),
                )
                registered_files.add(str(prediction["relative_path"]))
        elif (
            member.scope_kind != "cumulative_spent"
            or view.get("logical_transform_performed") is not False
            or view.get("prediction_artifacts") is not None
            or view.get("registered_heldout_text_accessed") is not False
            or view.get("view_input_policy")
            != "sealed_row_ids_only_no_sealed_text_or_labels_v1"
        ):
            raise ValueError("matched-pair cumulative reference view changed")
    if not exact_seen:
        raise ValueError("matched-pair execution lacks its exact logical view")
    events = terminal.get("event_order")
    if (
        not isinstance(events, list)
        or [row.get("sequence") for row in events]
        != list(range(1, len(events) + 1))
        or any(row.get("registered_heldout_labels_accessed") is not False for row in events)
    ):
        raise ValueError("matched-pair terminal event sequence changed")
    names = [row.get("event") for row in events]
    prefix = [
        "fit_completed",
        "matched_pair_subproducer_sealed",
        "matched_pair_subproducer_sealed",
        "fit_family_artifact_sealed",
    ]
    if names[: len(prefix)] != prefix:
        raise ValueError("matched-pair fit/seal event order changed")
    if [
        row.get("subproducer")
        for row in events
        if row.get("event") == "matched_pair_subproducer_sealed"
    ] != list(_SUBPRODUCERS):
        raise ValueError("matched-pair subproducer seal order changed")
    open_positions = [
        index for index, name in enumerate(names) if name == "exact_heldout_text_opened"
    ]
    if len(open_positions) != 1:
        raise ValueError("matched-pair exact text open event changed")
    opened = open_positions[0]
    if any(
        index > opened
        for index, name in enumerate(names)
        if name == "cumulative_fit_only_view_published"
    ):
        raise ValueError("matched-pair cumulative view was published after text access")
    if names[opened + 1 : opened + 3] != [
        "exact_heldout_transform_completed",
        "exact_heldout_transform_completed",
    ] or [
        row.get("subproducer")
        for row in events[opened + 1 : opened + 3]
    ] != list(_SUBPRODUCERS):
        raise ValueError("matched-pair exact subproducer transform order changed")
    if names[-1] != "exact_logical_view_published":
        raise ValueError("matched-pair exact logical publication event changed")
    for index, row in enumerate(events):
        expected_access = index >= opened
        if row.get("registered_heldout_text_accessed") is not expected_access:
            raise ValueError("matched-pair event text-access state changed")
    observed_files: set[str] = set()
    for path in artifact_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("matched-pair execution contains a symlink")
        if path.is_file():
            _sha256_file(path)
            observed_files.add(path.relative_to(artifact_root).as_posix())
    if observed_files != registered_files:
        raise ValueError("matched-pair terminal file inventory is not closed")
    return terminal


def replay_role_neutral_matched_pair_exact_transform(
    *,
    root: Path | str,
    request: RoleNeutralMatchedPairPhysicalGroupRequest,
    fit_texts: Sequence[str],
    exact_input: RoleNeutralMatchedPairExactInput,
    htr_model_path: Path | str | None = None,
    device: torch.device | str = "cpu",
) -> Mapping[str, Any]:
    """Reopen safe state and replay both exact matched-pair subproducers."""

    artifact_root = Path(root)
    terminal = validate_role_neutral_matched_pair_group_execution(
        root=artifact_root,
        request=request,
    )
    metadata, arrays, _seal = _validate_fit_side(
        root=artifact_root,
        request=request,
    )
    texts = tuple(fit_texts)
    if metadata.get("fit_text_sha256") != _text_sha256(
        request.physical_owner.fit_row_ids,
        texts,
    ):
        raise ValueError("matched-pair replay fit texts changed")
    if type(exact_input) is not RoleNeutralMatchedPairExactInput:
        raise TypeError("matched-pair replay requires its closed exact input")
    exact_identity = exact_input.validated(request.physical_owner.heldout_row_ids)
    _assert_text_capacity(
        exact_input.texts,
        extractor_config=metadata["scientific_configuration"]["matched_pair"][
            "htr_extractor"
        ],
        stage="exact replay",
    )
    _validate_htr_model_locator(
        request=request,
        config=metadata["scientific_configuration"]["matched_pair"],
        htr_model_path=htr_model_path,
    )
    owner_view_registration = next(
        row
        for row in terminal["logical_views"]
        if row["logical_scope_id"] == request.physical_owner.scope_id
    )
    owner_view = _read_json(
        artifact_root / owner_view_registration["relative_path"],
        label="matched-pair exact logical view",
    )
    if owner_view.get("exact_input_identity") != exact_identity:
        raise ValueError("matched-pair replay exact input changed")
    predictions = _sealed_exact_predictions(
        request=request,
        fit_texts=texts,
        exact_input=exact_input,
        metadata=metadata,
        arrays=arrays,
        htr_model_path=htr_model_path,
        device=torch.device(device),
    )
    result: dict[str, Any] = {}
    for subproducer in _SUBPRODUCERS:
        columns, observed = predictions[subproducer]
        registration = owner_view["prediction_artifacts"][subproducer]
        expected = _load_prediction(
            root=artifact_root,
            registration=registration,
            expected_rows=len(exact_input.row_ids),
        )
        if columns != registration["columns"] or not _replayed_predictions_match(
            observed,
            expected,
            subproducer=subproducer,
            config=metadata["scientific_configuration"]["matched_pair"],
        ):
            raise RuntimeError(
                f"fresh matched-pair {subproducer} replay changed predictions"
            )
        result[subproducer] = {
            "columns": columns,
            "values": observed,
        }
    return result
