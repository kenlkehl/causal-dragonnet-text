"""Two-phase role-neutral HTR execution for one physical Stage 1 group.

This module deliberately remains separate from the legacy all-ten-family
worker.  It establishes a native boundary for the hierarchical-transformer
family:

* only the canonical physical owner's ordered fit rows, complete texts, and
  fit labels are accepted by the fitting phase;
* every fitted native HTR model is reduced to closed JSON descriptors plus one
  non-object ``.npy`` file per tensor/array;
* the complete fit state and fit-only family seal are freshly authenticated
  before a registered held-out-text loader can be called;
* cumulative-review logical scopes receive immutable fit-only references and
  never receive their sealed text; and
* the exact-inner logical scope is transformed by reconstructing the sealed
  models with row IDs and complete text only.  Its treatment/outcome labels
  are not an argument anywhere in the view phase.

All capacity and training values are required in :class:`RoleNeutralHTRConfig`.
The executor validates that ``max_chunks`` and ``max_chunk_length`` are
nonbinding for every supplied note.  It fails closed instead of invoking the
historical tail-retention behavior of ``split_text_into_word_chunks``.

The all-ten-family production adapter must remain fail-closed until every
family has an equivalent fit/view boundary.
"""

from __future__ import annotations

import copy
import concurrent.futures
import hashlib
import io
import inspect
import json
import multiprocessing as mp
import os
import re
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from ..models.hierarchical_transformer_extractor import (
    HierarchicalTransformerExtractor,
    split_text_into_word_chunks,
)
from ..utils.calibration import BinaryProbabilityCalibrator
from .agentic_attention_variable_forest import _EffectNet, _NuisanceNet
from .all_evidence_discovery_interfaces import HTR_NEURAL
from .htr_native_proof_capture import (
    _apply_calibrator,
    _array_sha256,
    _build_model,
    _capture_calibrator,
    _capture_model_state,
    _extractor_descriptor,
    _predict_model,
)
from .lossless_stage1_evidence_catalog import (
    NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
)
from .neural_numerical_replay import (
    neural_float_arrays_within_tolerance,
    validate_neural_replay_settings,
)
from .production_stage1_legacy_scope_fragments import (
    LEGACY_STAGE1_FIT_ONLY_FAMILY_SEAL_SCHEMA,
)
from .production_stage1_scope_scheduler import (
    Stage1ScopePlan,
    Stage1ScopeSpec,
    _enforce_stage1_torch_determinism,
    _observe_stage1_torch_determinism,
    _validate_torch_determinism_observation,
)
from .stage1_htr_operational_controls import (
    ROLE_NEUTRAL_HTR_OPERATIONAL_CONTROLS_SCHEMA,
    RoleNeutralHTRFoldResourcePlan,
    RoleNeutralHTROperationalControls,
)

ROLE_NEUTRAL_HTR_GROUP_REQUEST_SCHEMA = (
    "production_role_neutral_htr_physical_group_request_v1"
)
ROLE_NEUTRAL_HTR_CONFIG_SCHEMA = "production_role_neutral_htr_config_v3"
ROLE_NEUTRAL_HTR_FIT_STATE_SCHEMA = "production_role_neutral_htr_fit_state_v1"
ROLE_NEUTRAL_HTR_LOGICAL_VIEW_SCHEMA = "production_role_neutral_htr_logical_view_v1"
ROLE_NEUTRAL_HTR_GROUP_EXECUTION_SCHEMA = (
    "production_role_neutral_htr_group_execution_v1"
)
ROLE_NEUTRAL_HTR_COVERAGE_SCHEMA = "production_role_neutral_htr_word_coverage_v1"
ROLE_NEUTRAL_HTR_REUSABLE_PLAN_SCHEMA = (
    "production_role_neutral_htr_reusable_text_plan_v1"
)
ROLE_NEUTRAL_HTR_PROCESS_PLAN_SCHEMA = (
    "production_role_neutral_htr_process_text_plan_v1"
)
ROLE_NEUTRAL_HTR_OPERATIONAL_ATTESTATION_SCHEMA = (
    "production_role_neutral_htr_operational_attestation_v2"
)

_FIT_STATE_DIRECTORY = "fit_state"
_FIT_STATE_METADATA = "metadata.json"
_FIT_SEAL_FILE = "fit_only_family_seal.json"
_LOGICAL_VIEW_DIRECTORY = "logical_views"
_TERMINAL_FILE = "execution_manifest.json"
_HEX = frozenset("0123456789abcdef")
_SAFE_ARRAY_KEY = re.compile(r"^[a-z0-9_]+$")
_SUPPORTED_EFFECT_OBJECTIVES = frozenset(
    {"pseudo_outcome_mse", "squared_r_loss"}
)
_SUPPORTED_CALIBRATION = frozenset(
    {"none", "temperature", "isotonic", "temperature_isotonic"}
)
_SUPPORTED_ACTIVATIONS = frozenset(
    {"gelu_exact", "gelu_tanh", "relu", "silu", "tanh"}
)
_FORBIDDEN_ARTIFACT_SUFFIXES = frozenset(
    {
        ".joblib",
        ".npz",
        ".pkl",
        ".pickle",
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
        ".safetensors",
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


def _duplicate_rejecting_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_key, value in pairs:
        key = str(raw_key)
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = _read_regular_file(path, label=label)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _read_regular_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int | None = None,
) -> bytes:
    target = Path(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(target, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ValueError(f"{label} must be one singly-linked regular file")
        if maximum_bytes is not None and int(before.st_size) > int(maximum_bytes):
            raise ValueError(f"{label} exceeds its configured read bound")
        payload = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            payload.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        tuple(getattr(before, field) for field in identity)
        != tuple(getattr(after, field) for field in identity)
        or len(payload) != int(after.st_size)
    ):
        raise RuntimeError(f"{label} changed while reading")
    named = os.stat(target, follow_symlinks=False)
    if (
        not stat.S_ISREG(named.st_mode)
        or int(named.st_nlink) != 1
        or (int(named.st_dev), int(named.st_ino))
        != (int(after.st_dev), int(after.st_ino))
    ):
        raise RuntimeError(f"{label} path was substituted while reading")
    return bytes(payload)


def _sha256_file(path: Path, *, label: str = "artifact") -> tuple[str, int]:
    payload = _read_regular_file(path, label=label)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _read_npy_file_once(
    path: Path,
    *,
    label: str,
    invalid_message: str,
) -> tuple[str, int, np.ndarray]:
    """Authenticate and parse one ``.npy`` file without a filesystem mmap.

    Production HTR state contains hundreds of immutable arrays.  Keeping one
    read-only ``numpy.memmap`` per array alive during validation consumes a
    correspondingly large number of filesystem handles and is not portable to
    every shared mount.  Read the already-authenticated bytes once and parse
    them from memory instead.  This also removes the path-level race between
    hashing an artifact and reopening it for NumPy parsing.
    """

    payload = _read_regular_file(path, label=label)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        with io.BytesIO(payload) as stream:
            loaded = np.load(stream, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(invalid_message) from exc
    if not isinstance(loaded, np.ndarray) or loaded.dtype.hasobject:
        close = getattr(loaded, "close", None)
        if callable(close):
            close()
        raise ValueError(invalid_message)
    return digest, len(payload), loaded


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
        directory = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
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


def _write_new_npy(path: Path, value: Any) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace immutable array: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("role-neutral HTR arrays cannot use object dtype")
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
        directory = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _inventory_tree(root: Path) -> tuple[set[str], set[str]]:
    tree = Path(root)
    if tree.is_symlink() or not tree.is_dir():
        raise ValueError("role-neutral HTR artifact must be one real directory")
    before = os.stat(tree, follow_symlinks=False)
    files: set[str] = set()
    directories: set[str] = set()
    for candidate in tree.rglob("*"):
        relative = candidate.relative_to(tree).as_posix()
        observed = os.stat(candidate, follow_symlinks=False)
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError("role-neutral HTR artifact cannot contain symbolic links")
        if stat.S_ISDIR(observed.st_mode):
            directories.add(relative)
        elif stat.S_ISREG(observed.st_mode):
            if int(observed.st_nlink) != 1:
                raise ValueError("role-neutral HTR artifact cannot contain hard links")
            if candidate.suffix.lower() in _FORBIDDEN_ARTIFACT_SUFFIXES:
                raise ValueError("role-neutral HTR artifact contains a forbidden checkpoint")
            files.add(relative)
        else:
            raise ValueError("role-neutral HTR artifact contains a special file")
    after = os.stat(tree, follow_symlinks=False)
    identity = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if tuple(getattr(before, field) for field in identity) != tuple(
        getattr(after, field) for field in identity
    ):
        raise RuntimeError("role-neutral HTR tree changed during inventory")
    return files, directories


def _tree_sha256(root: Path) -> str:
    files, directories = _inventory_tree(root)
    rows: list[dict[str, Any]] = [
        {"path": relative, "kind": "directory"}
        for relative in sorted(directories)
    ]
    for relative in sorted(files):
        digest, size = _sha256_file(
            Path(root) / relative,
            label=f"HTR tree file {relative}",
        )
        rows.append(
            {
                "path": relative,
                "kind": "file",
                "sha256": digest,
                "size_bytes": size,
            }
        )
    if not rows:
        raise ValueError("role-neutral HTR tree is empty")
    return _sha256_json(
        {
            "schema_version": "production_role_neutral_htr_tree_v1",
            "inventory": rows,
        }
    )


def _row_order_fingerprint(row_ids: Sequence[int]) -> str:
    rows = tuple(int(row_id) for row_id in row_ids)
    if not rows or len(rows) != len(set(rows)) or any(row < 0 for row in rows):
        raise ValueError("HTR row IDs must be unique non-negative integers")
    # Match Stage1ScopeSpec exactly so the architecture-specific fit seal can
    # be consumed by the shared all-ten role-neutral binding contract.
    return _sha256_json(list(rows))


def _text_sha256(row_ids: Sequence[int], texts: Sequence[str]) -> str:
    rows = tuple(int(row_id) for row_id in row_ids)
    values = tuple(texts)
    if len(rows) != len(values) or any(not isinstance(text, str) for text in values):
        raise ValueError("HTR text binding requires one string per ordered row ID")
    digest = hashlib.sha256()
    digest.update(b"production-role-neutral-htr-text-v1\0")
    for row_id, text in zip(rows, values, strict=True):
        encoded = text.encode("utf-8")
        digest.update(int(row_id).to_bytes(8, "little", signed=False))
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _float_hex_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return _sha256_json([float(value).hex() for value in array])


def _binary_vector(values: Sequence[Any], *, label: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (int(length),) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be one finite vector aligned to fit rows")
    if not set(np.unique(array)).issubset({0.0, 1.0}):
        raise ValueError(f"{label} must be binary")
    return array


def _derived_seed(
    group_seed: int,
    *,
    purpose: str,
    objective: str,
    fold: int,
) -> int:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": "production_role_neutral_htr_seed_v1",
                "canonical_group_seed": int(group_seed),
                "purpose": str(purpose),
                "objective": str(objective),
                "fold": int(fold),
            }
        ).encode("utf-8")
    ).digest()
    result = int.from_bytes(digest[:8], "big") % (2**31 - 1)
    return result or 1


@dataclass(frozen=True)
class RoleNeutralHTRConfig:
    """Complete HTR scientific configuration; every field is explicit."""

    sentence_encoder_model_kind: str
    model_tree_sha256: str | None
    freeze_sentence_encoder: bool
    chunk_size_words: int
    chunk_overlap_words: int
    max_chunks: int
    max_chunk_length: int
    num_transformer_layers: int
    num_attention_heads: int
    transformer_dim: int
    transformer_dropout: float
    projection_dim: int
    hash_embedding_dim: int
    sentence_encoder_batch_size: int
    sentence_encoder_backend: str
    sentence_pooling: str
    normalize_sentence_embeddings: bool
    trainable_sentence_encoder_layers: int
    role_attention: bool
    w_attention_heads: int
    x_attention_heads: int
    transformer_feedforward_dim: int
    transformer_activation: str
    transformer_norm_style: str
    transformer_layer_norm_eps: float
    transformer_layer_norm_elementwise_affine: bool
    transformer_layer_norm_bias: bool
    transformer_attention_dropout: float
    transformer_residual_dropout: float
    transformer_feedforward_dropout: float
    transformer_attention_bias: bool
    transformer_feedforward_bias: bool
    output_projection_depth: int
    output_projection_hidden_dim: int
    output_projection_activation: str
    output_projection_dropout: float
    output_projection_hidden_layer_norm: bool
    output_projection_final_layer_norm: bool
    output_projection_bias: bool
    pool_token_init_std: float
    positional_encoding_base: float
    environment_override_policy: str
    require_live_unfrozen_encoder_attestation: bool
    hidden_dim: int
    nuisance_head_depth: int
    nuisance_head_activation: str
    nuisance_head_dropout: float
    nuisance_head_layer_norm: bool
    nuisance_head_bias: bool
    effect_head_depth: int
    effect_head_activation: str
    effect_head_dropout: float
    effect_head_layer_norm: bool
    effect_head_bias: bool
    nuisance_folds: int
    effect_folds: int
    nuisance_epochs: int
    effect_epochs: int
    batch_size: int
    prediction_batch_size: int
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    adamw_beta1: float
    adamw_beta2: float
    adamw_eps: float
    adamw_amsgrad: bool
    adamw_maximize: bool
    adamw_foreach: bool
    adamw_capturable: bool
    adamw_differentiable: bool
    adamw_fused: bool
    optimizer_zero_grad_set_to_none: bool
    alpha_propensity: float
    nuisance_label_smoothing: float
    nuisance_calibration: str
    e_clip: float
    r_stage_min_propensity: float
    r_stage_max_propensity: float
    gradient_clip_norm: float
    gradient_clip_norm_type: float
    gradient_clip_error_if_nonfinite: bool
    gradient_clip_foreach: bool
    effect_objectives: tuple[str, ...]
    outcome_type: str
    replay_comparison_policy: str
    replay_relative_tolerance: float
    replay_absolute_tolerance: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoleNeutralHTRConfig":
        if not isinstance(value, Mapping):
            raise TypeError("role-neutral HTR configuration must be a mapping")
        payload = dict(value)
        expected = set(cls.__dataclass_fields__)
        if "schema_version" in payload or "text_truncation_applied" in payload:
            wire_expected = expected | {
                "schema_version",
                "text_truncation_applied",
            }
            if set(payload) != wire_expected:
                raise ValueError(
                    "role-neutral HTR wire configuration keys differ; "
                    f"missing={sorted(wire_expected - set(payload))}, "
                    f"extra={sorted(set(payload) - wire_expected)}"
                )
            if (
                payload.pop("schema_version") != ROLE_NEUTRAL_HTR_CONFIG_SCHEMA
                or payload.pop("text_truncation_applied") is not False
            ):
                raise ValueError("role-neutral HTR wire configuration changed")
        observed = set(payload)
        if observed != expected:
            raise ValueError(
                "role-neutral HTR configuration keys differ; "
                f"missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )
        return cls(
            sentence_encoder_model_kind=str(payload["sentence_encoder_model_kind"]),
            model_tree_sha256=(
                None
                if payload["model_tree_sha256"] is None
                else str(payload["model_tree_sha256"])
            ),
            freeze_sentence_encoder=payload["freeze_sentence_encoder"],
            chunk_size_words=int(payload["chunk_size_words"]),
            chunk_overlap_words=int(payload["chunk_overlap_words"]),
            max_chunks=int(payload["max_chunks"]),
            max_chunk_length=int(payload["max_chunk_length"]),
            num_transformer_layers=int(payload["num_transformer_layers"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            transformer_dim=int(payload["transformer_dim"]),
            transformer_dropout=float(payload["transformer_dropout"]),
            projection_dim=int(payload["projection_dim"]),
            hash_embedding_dim=int(payload["hash_embedding_dim"]),
            sentence_encoder_batch_size=int(payload["sentence_encoder_batch_size"]),
            sentence_encoder_backend=str(payload["sentence_encoder_backend"]),
            sentence_pooling=str(payload["sentence_pooling"]),
            normalize_sentence_embeddings=payload["normalize_sentence_embeddings"],
            trainable_sentence_encoder_layers=int(
                payload["trainable_sentence_encoder_layers"]
            ),
            role_attention=payload["role_attention"],
            w_attention_heads=int(payload["w_attention_heads"]),
            x_attention_heads=int(payload["x_attention_heads"]),
            transformer_feedforward_dim=int(
                payload["transformer_feedforward_dim"]
            ),
            transformer_activation=str(payload["transformer_activation"]),
            transformer_norm_style=str(payload["transformer_norm_style"]),
            transformer_layer_norm_eps=float(
                payload["transformer_layer_norm_eps"]
            ),
            transformer_layer_norm_elementwise_affine=payload[
                "transformer_layer_norm_elementwise_affine"
            ],
            transformer_layer_norm_bias=payload[
                "transformer_layer_norm_bias"
            ],
            transformer_attention_dropout=float(
                payload["transformer_attention_dropout"]
            ),
            transformer_residual_dropout=float(
                payload["transformer_residual_dropout"]
            ),
            transformer_feedforward_dropout=float(
                payload["transformer_feedforward_dropout"]
            ),
            transformer_attention_bias=payload[
                "transformer_attention_bias"
            ],
            transformer_feedforward_bias=payload[
                "transformer_feedforward_bias"
            ],
            output_projection_depth=int(payload["output_projection_depth"]),
            output_projection_hidden_dim=int(
                payload["output_projection_hidden_dim"]
            ),
            output_projection_activation=str(
                payload["output_projection_activation"]
            ),
            output_projection_dropout=float(
                payload["output_projection_dropout"]
            ),
            output_projection_hidden_layer_norm=payload[
                "output_projection_hidden_layer_norm"
            ],
            output_projection_final_layer_norm=payload[
                "output_projection_final_layer_norm"
            ],
            output_projection_bias=payload["output_projection_bias"],
            pool_token_init_std=float(payload["pool_token_init_std"]),
            positional_encoding_base=float(
                payload["positional_encoding_base"]
            ),
            environment_override_policy=str(
                payload["environment_override_policy"]
            ),
            require_live_unfrozen_encoder_attestation=payload[
                "require_live_unfrozen_encoder_attestation"
            ],
            hidden_dim=int(payload["hidden_dim"]),
            nuisance_head_depth=int(payload["nuisance_head_depth"]),
            nuisance_head_activation=str(
                payload["nuisance_head_activation"]
            ),
            nuisance_head_dropout=float(payload["nuisance_head_dropout"]),
            nuisance_head_layer_norm=payload["nuisance_head_layer_norm"],
            nuisance_head_bias=payload["nuisance_head_bias"],
            effect_head_depth=int(payload["effect_head_depth"]),
            effect_head_activation=str(payload["effect_head_activation"]),
            effect_head_dropout=float(payload["effect_head_dropout"]),
            effect_head_layer_norm=payload["effect_head_layer_norm"],
            effect_head_bias=payload["effect_head_bias"],
            nuisance_folds=int(payload["nuisance_folds"]),
            effect_folds=int(payload["effect_folds"]),
            nuisance_epochs=int(payload["nuisance_epochs"]),
            effect_epochs=int(payload["effect_epochs"]),
            batch_size=int(payload["batch_size"]),
            prediction_batch_size=int(payload["prediction_batch_size"]),
            optimizer_name=str(payload["optimizer_name"]),
            learning_rate=float(payload["learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            adamw_beta1=float(payload["adamw_beta1"]),
            adamw_beta2=float(payload["adamw_beta2"]),
            adamw_eps=float(payload["adamw_eps"]),
            adamw_amsgrad=payload["adamw_amsgrad"],
            adamw_maximize=payload["adamw_maximize"],
            adamw_foreach=payload["adamw_foreach"],
            adamw_capturable=payload["adamw_capturable"],
            adamw_differentiable=payload["adamw_differentiable"],
            adamw_fused=payload["adamw_fused"],
            optimizer_zero_grad_set_to_none=payload[
                "optimizer_zero_grad_set_to_none"
            ],
            alpha_propensity=float(payload["alpha_propensity"]),
            nuisance_label_smoothing=float(payload["nuisance_label_smoothing"]),
            nuisance_calibration=str(payload["nuisance_calibration"]),
            e_clip=float(payload["e_clip"]),
            r_stage_min_propensity=float(payload["r_stage_min_propensity"]),
            r_stage_max_propensity=float(payload["r_stage_max_propensity"]),
            gradient_clip_norm=float(payload["gradient_clip_norm"]),
            gradient_clip_norm_type=float(
                payload["gradient_clip_norm_type"]
            ),
            gradient_clip_error_if_nonfinite=payload[
                "gradient_clip_error_if_nonfinite"
            ],
            gradient_clip_foreach=payload["gradient_clip_foreach"],
            effect_objectives=tuple(
                str(item) for item in payload["effect_objectives"]
            ),
            outcome_type=str(payload["outcome_type"]),
            replay_comparison_policy=str(
                payload["replay_comparison_policy"]
            ),
            replay_relative_tolerance=payload["replay_relative_tolerance"],
            replay_absolute_tolerance=payload["replay_absolute_tolerance"],
        ).validated()

    def validated(self) -> "RoleNeutralHTRConfig":
        if self.sentence_encoder_model_kind not in {
            "hash",
            "authenticated_local_tree",
        }:
            raise ValueError("HTR sentence encoder kind is unsupported")
        if self.sentence_encoder_model_kind == "hash":
            if self.model_tree_sha256 is not None:
                raise ValueError("hash HTR configuration cannot bind a model tree")
            if self.require_live_unfrozen_encoder_attestation:
                raise ValueError("production live-encoder attestation rejects hash HTR")
        else:
            _require_sha256(
                self.model_tree_sha256,
                label="HTR model tree identity",
            )
        for name, value in (
            ("chunk_size_words", self.chunk_size_words),
            ("max_chunks", self.max_chunks),
            ("max_chunk_length", self.max_chunk_length),
            ("num_transformer_layers", self.num_transformer_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("transformer_dim", self.transformer_dim),
            ("projection_dim", self.projection_dim),
            ("hash_embedding_dim", self.hash_embedding_dim),
            ("sentence_encoder_batch_size", self.sentence_encoder_batch_size),
            ("w_attention_heads", self.w_attention_heads),
            ("x_attention_heads", self.x_attention_heads),
            (
                "transformer_feedforward_dim",
                self.transformer_feedforward_dim,
            ),
            (
                "output_projection_hidden_dim",
                self.output_projection_hidden_dim,
            ),
            ("hidden_dim", self.hidden_dim),
            ("nuisance_head_depth", self.nuisance_head_depth),
            ("effect_head_depth", self.effect_head_depth),
            ("nuisance_folds", self.nuisance_folds),
            ("effect_folds", self.effect_folds),
            ("nuisance_epochs", self.nuisance_epochs),
            ("effect_epochs", self.effect_epochs),
            ("batch_size", self.batch_size),
            ("prediction_batch_size", self.prediction_batch_size),
        ):
            if int(value) < 1:
                raise ValueError(f"configured {name} must be positive")
        if self.nuisance_folds < 2 or self.effect_folds < 2:
            raise ValueError("HTR nuisance/effect folds must each be at least two")
        if (
            isinstance(self.output_projection_depth, bool)
            or self.output_projection_depth < 0
        ):
            raise ValueError("HTR output projection depth cannot be negative")
        if (
            self.chunk_overlap_words < 0
            or self.chunk_overlap_words >= self.chunk_size_words
        ):
            raise ValueError("HTR chunk overlap must be in [0, chunk size)")
        if self.trainable_sentence_encoder_layers < 0:
            raise ValueError("HTR trainable sentence encoder layers cannot be negative")
        if self.num_attention_heads > self.transformer_dim or (
            self.transformer_dim % self.num_attention_heads
        ):
            raise ValueError("HTR transformer dimension must divide evenly across heads")
        for name, value in (
            ("transformer_dropout", self.transformer_dropout),
            (
                "transformer_attention_dropout",
                self.transformer_attention_dropout,
            ),
            (
                "transformer_residual_dropout",
                self.transformer_residual_dropout,
            ),
            (
                "transformer_feedforward_dropout",
                self.transformer_feedforward_dropout,
            ),
            (
                "output_projection_dropout",
                self.output_projection_dropout,
            ),
            ("nuisance_head_dropout", self.nuisance_head_dropout),
            ("effect_head_dropout", self.effect_head_dropout),
        ):
            if not np.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"HTR {name} must be in [0, 1)")
        if self.transformer_activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError("HTR transformer activation is unsupported")
        if self.output_projection_activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError("HTR output projection activation is unsupported")
        if self.nuisance_head_activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError("HTR nuisance-head activation is unsupported")
        if self.effect_head_activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError("HTR effect-head activation is unsupported")
        if self.transformer_norm_style not in {"pre_norm", "post_norm"}:
            raise ValueError("HTR transformer norm style is unsupported")
        if (
            not np.isfinite(self.transformer_layer_norm_eps)
            or self.transformer_layer_norm_eps <= 0.0
        ):
            raise ValueError("HTR layer-norm epsilon must be positive")
        if (
            not np.isfinite(self.pool_token_init_std)
            or self.pool_token_init_std < 0.0
        ):
            raise ValueError("HTR pool-token initialization std is invalid")
        if (
            not np.isfinite(self.positional_encoding_base)
            or self.positional_encoding_base <= 1.0
        ):
            raise ValueError("HTR positional-encoding base must exceed one")
        if self.environment_override_policy != "forbid":
            raise ValueError(
                "typed role-neutral HTR requires environment_override_policy=forbid"
            )
        if self.sentence_encoder_backend not in {
            "auto",
            "sentence_transformers",
            "transformers",
        }:
            raise ValueError("HTR sentence encoder backend is unsupported")
        if self.sentence_pooling not in {
            "auto",
            "cls",
            "last",
            "mean",
            "token_attention",
        }:
            raise ValueError("HTR sentence pooling is unsupported")
        if not isinstance(self.freeze_sentence_encoder, bool):
            raise TypeError("HTR freeze_sentence_encoder must be boolean")
        for name, value in (
            ("normalize_sentence_embeddings", self.normalize_sentence_embeddings),
            ("role_attention", self.role_attention),
            (
                "require_live_unfrozen_encoder_attestation",
                self.require_live_unfrozen_encoder_attestation,
            ),
            (
                "transformer_layer_norm_elementwise_affine",
                self.transformer_layer_norm_elementwise_affine,
            ),
            (
                "transformer_layer_norm_bias",
                self.transformer_layer_norm_bias,
            ),
            ("transformer_attention_bias", self.transformer_attention_bias),
            ("transformer_feedforward_bias", self.transformer_feedforward_bias),
            (
                "output_projection_hidden_layer_norm",
                self.output_projection_hidden_layer_norm,
            ),
            (
                "output_projection_final_layer_norm",
                self.output_projection_final_layer_norm,
            ),
            ("output_projection_bias", self.output_projection_bias),
            ("nuisance_head_layer_norm", self.nuisance_head_layer_norm),
            ("nuisance_head_bias", self.nuisance_head_bias),
            ("effect_head_layer_norm", self.effect_head_layer_norm),
            ("effect_head_bias", self.effect_head_bias),
            ("adamw_amsgrad", self.adamw_amsgrad),
            ("adamw_maximize", self.adamw_maximize),
            ("adamw_foreach", self.adamw_foreach),
            ("adamw_capturable", self.adamw_capturable),
            ("adamw_differentiable", self.adamw_differentiable),
            ("adamw_fused", self.adamw_fused),
            (
                "optimizer_zero_grad_set_to_none",
                self.optimizer_zero_grad_set_to_none,
            ),
            (
                "gradient_clip_error_if_nonfinite",
                self.gradient_clip_error_if_nonfinite,
            ),
            ("gradient_clip_foreach", self.gradient_clip_foreach),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"HTR {name} must be boolean")
        if self.optimizer_name != "adamw":
            raise ValueError("role-neutral HTR v2 requires optimizer_name=adamw")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("alpha_propensity", self.alpha_propensity),
            ("gradient_clip_norm", self.gradient_clip_norm),
            ("gradient_clip_norm_type", self.gradient_clip_norm_type),
            ("adamw_eps", self.adamw_eps),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"configured {name} must be finite and non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("configured HTR learning rate must be positive")
        if not 0.0 <= self.adamw_beta1 < 1.0 or not 0.0 <= self.adamw_beta2 < 1.0:
            raise ValueError("configured HTR AdamW betas must be in [0, 1)")
        if self.adamw_eps <= 0.0:
            raise ValueError("configured HTR AdamW epsilon must be positive")
        if self.gradient_clip_norm_type <= 0.0:
            raise ValueError("configured HTR gradient norm type must be positive")
        if not 0.0 <= self.nuisance_label_smoothing < 1.0:
            raise ValueError("HTR nuisance label smoothing must be in [0, 1)")
        if self.nuisance_calibration not in _SUPPORTED_CALIBRATION:
            raise ValueError("HTR nuisance calibration is unsupported")
        if not 0.0 < self.e_clip < 0.5:
            raise ValueError("configured HTR e_clip must be in (0, 0.5)")
        if not (
            0.0 <= self.r_stage_min_propensity
            < self.r_stage_max_propensity
            <= 1.0
        ):
            raise ValueError("HTR R-stage propensity bounds are invalid")
        if (
            not self.effect_objectives
            or len(set(self.effect_objectives)) != len(self.effect_objectives)
            or not set(self.effect_objectives).issubset(_SUPPORTED_EFFECT_OBJECTIVES)
        ):
            raise ValueError("HTR effect objectives are empty, duplicated, or unsupported")
        if self.outcome_type != "binary":
            raise ValueError("role-neutral HTR v1 supports binary outcomes only")
        validate_neural_replay_settings(
            policy=self.replay_comparison_policy,
            relative_tolerance=self.replay_relative_tolerance,
            absolute_tolerance=self.replay_absolute_tolerance,
        )
        return self

    def as_dict(self) -> dict[str, Any]:
        self.validated()
        return {
            "schema_version": ROLE_NEUTRAL_HTR_CONFIG_SCHEMA,
            "sentence_encoder_model_kind": self.sentence_encoder_model_kind,
            "model_tree_sha256": self.model_tree_sha256,
            "freeze_sentence_encoder": self.freeze_sentence_encoder,
            "chunk_size_words": self.chunk_size_words,
            "chunk_overlap_words": self.chunk_overlap_words,
            "max_chunks": self.max_chunks,
            "max_chunk_length": self.max_chunk_length,
            "num_transformer_layers": self.num_transformer_layers,
            "num_attention_heads": self.num_attention_heads,
            "transformer_dim": self.transformer_dim,
            "transformer_dropout": self.transformer_dropout,
            "projection_dim": self.projection_dim,
            "hash_embedding_dim": self.hash_embedding_dim,
            "sentence_encoder_batch_size": self.sentence_encoder_batch_size,
            "sentence_encoder_backend": self.sentence_encoder_backend,
            "sentence_pooling": self.sentence_pooling,
            "normalize_sentence_embeddings": self.normalize_sentence_embeddings,
            "trainable_sentence_encoder_layers": (
                self.trainable_sentence_encoder_layers
            ),
            "role_attention": self.role_attention,
            "w_attention_heads": self.w_attention_heads,
            "x_attention_heads": self.x_attention_heads,
            "transformer_feedforward_dim": self.transformer_feedforward_dim,
            "transformer_activation": self.transformer_activation,
            "transformer_norm_style": self.transformer_norm_style,
            "transformer_layer_norm_eps": self.transformer_layer_norm_eps,
            "transformer_layer_norm_elementwise_affine": (
                self.transformer_layer_norm_elementwise_affine
            ),
            "transformer_layer_norm_bias": self.transformer_layer_norm_bias,
            "transformer_attention_dropout": (
                self.transformer_attention_dropout
            ),
            "transformer_residual_dropout": (
                self.transformer_residual_dropout
            ),
            "transformer_feedforward_dropout": (
                self.transformer_feedforward_dropout
            ),
            "transformer_attention_bias": self.transformer_attention_bias,
            "transformer_feedforward_bias": self.transformer_feedforward_bias,
            "output_projection_depth": self.output_projection_depth,
            "output_projection_hidden_dim": (
                self.output_projection_hidden_dim
            ),
            "output_projection_activation": (
                self.output_projection_activation
            ),
            "output_projection_dropout": self.output_projection_dropout,
            "output_projection_hidden_layer_norm": (
                self.output_projection_hidden_layer_norm
            ),
            "output_projection_final_layer_norm": (
                self.output_projection_final_layer_norm
            ),
            "output_projection_bias": self.output_projection_bias,
            "pool_token_init_std": self.pool_token_init_std,
            "positional_encoding_base": self.positional_encoding_base,
            "environment_override_policy": self.environment_override_policy,
            "require_live_unfrozen_encoder_attestation": (
                self.require_live_unfrozen_encoder_attestation
            ),
            "hidden_dim": self.hidden_dim,
            "nuisance_head_depth": self.nuisance_head_depth,
            "nuisance_head_activation": self.nuisance_head_activation,
            "nuisance_head_dropout": self.nuisance_head_dropout,
            "nuisance_head_layer_norm": self.nuisance_head_layer_norm,
            "nuisance_head_bias": self.nuisance_head_bias,
            "effect_head_depth": self.effect_head_depth,
            "effect_head_activation": self.effect_head_activation,
            "effect_head_dropout": self.effect_head_dropout,
            "effect_head_layer_norm": self.effect_head_layer_norm,
            "effect_head_bias": self.effect_head_bias,
            "nuisance_folds": self.nuisance_folds,
            "effect_folds": self.effect_folds,
            "nuisance_epochs": self.nuisance_epochs,
            "effect_epochs": self.effect_epochs,
            "batch_size": self.batch_size,
            "prediction_batch_size": self.prediction_batch_size,
            "optimizer_name": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "adamw_beta1": self.adamw_beta1,
            "adamw_beta2": self.adamw_beta2,
            "adamw_eps": self.adamw_eps,
            "adamw_amsgrad": self.adamw_amsgrad,
            "adamw_maximize": self.adamw_maximize,
            "adamw_foreach": self.adamw_foreach,
            "adamw_capturable": self.adamw_capturable,
            "adamw_differentiable": self.adamw_differentiable,
            "adamw_fused": self.adamw_fused,
            "optimizer_zero_grad_set_to_none": (
                self.optimizer_zero_grad_set_to_none
            ),
            "alpha_propensity": self.alpha_propensity,
            "nuisance_label_smoothing": self.nuisance_label_smoothing,
            "nuisance_calibration": self.nuisance_calibration,
            "e_clip": self.e_clip,
            "r_stage_min_propensity": self.r_stage_min_propensity,
            "r_stage_max_propensity": self.r_stage_max_propensity,
            "gradient_clip_norm": self.gradient_clip_norm,
            "gradient_clip_norm_type": self.gradient_clip_norm_type,
            "gradient_clip_error_if_nonfinite": (
                self.gradient_clip_error_if_nonfinite
            ),
            "gradient_clip_foreach": self.gradient_clip_foreach,
            "effect_objectives": list(self.effect_objectives),
            "outcome_type": self.outcome_type,
            "replay_comparison_policy": self.replay_comparison_policy,
            "replay_relative_tolerance": self.replay_relative_tolerance,
            "replay_absolute_tolerance": self.replay_absolute_tolerance,
            "text_truncation_applied": False,
        }


@dataclass(frozen=True)
class RoleNeutralHTRPhysicalGroupRequest:
    """Closed authority passed to one physical HTR worker."""

    plan_scientific_content_sha256: str
    physical_owner: Stage1ScopeSpec
    logical_members: tuple[Stage1ScopeSpec, ...]
    content_sha256: str

    @classmethod
    def from_plan(
        cls,
        *,
        plan: Stage1ScopePlan,
        physical_owner_scope_id: str,
    ) -> "RoleNeutralHTRPhysicalGroupRequest":
        if not isinstance(plan, Stage1ScopePlan):
            raise TypeError("role-neutral HTR request requires a Stage1ScopePlan")
        owner = plan.scope(str(physical_owner_scope_id))
        if plan.physical_owner(owner.scope_id).scope_id != owner.scope_id:
            raise ValueError("role-neutral HTR request must name a physical owner")
        matches = [
            members
            for candidate, members in plan.physical_scope_groups
            if candidate.scope_id == owner.scope_id
        ]
        if len(matches) != 1:
            raise RuntimeError("HTR physical owner has no unique logical group")
        members = matches[0]
        if (
            not members
            or members[0].scope_id != owner.scope_id
            or any(
                tuple(member.fit_row_ids)
                != tuple(owner.fit_row_ids)
                or int(member.scope_seed) != int(owner.scope_seed)
                for member in members
            )
        ):
            raise ValueError(
                "role-neutral HTR reuse requires identical ordered fit rows and seed"
            )
        if len(members) > 1 and (
            owner.scope_kind != "exact_inner"
            or any(
                member.scope_kind != "cumulative_spent"
                for member in members[1:]
            )
        ):
            raise ValueError("HTR reuse supports exact-inner/cumulative groups only")
        body = _group_request_body(
            plan_scientific_content_sha256=plan.scientific_content_sha256,
            owner=owner,
            members=members,
        )
        return cls(
            plan_scientific_content_sha256=plan.scientific_content_sha256,
            physical_owner=owner,
            logical_members=members,
            content_sha256=_sha256_json(body),
        )

    def as_dict(self) -> dict[str, Any]:
        _require_sha256(
            self.plan_scientific_content_sha256,
            label="role-neutral HTR plan identity",
        )
        if (
            not self.logical_members
            or self.logical_members[0].scope_id != self.physical_owner.scope_id
            or len({member.scope_id for member in self.logical_members})
            != len(self.logical_members)
            or any(
                tuple(member.fit_row_ids)
                != tuple(self.physical_owner.fit_row_ids)
                or int(member.scope_seed)
                != int(self.physical_owner.scope_seed)
                for member in self.logical_members
            )
        ):
            raise ValueError("role-neutral HTR logical-group authority changed")
        body = _group_request_body(
            plan_scientific_content_sha256=self.plan_scientific_content_sha256,
            owner=self.physical_owner,
            members=self.logical_members,
        )
        if _sha256_json(body) != self.content_sha256:
            raise RuntimeError("role-neutral HTR group request changed")
        return {**body, "content_sha256": self.content_sha256}


def _group_request_body(
    *,
    plan_scientific_content_sha256: str,
    owner: Stage1ScopeSpec,
    members: Sequence[Stage1ScopeSpec],
) -> dict[str, Any]:
    return {
        "schema_version": ROLE_NEUTRAL_HTR_GROUP_REQUEST_SCHEMA,
        "plan_scientific_content_sha256": plan_scientific_content_sha256,
        "physical_owner": owner.as_dict(),
        "logical_members": [member.as_dict() for member in members],
        "logical_scope_count": len(members),
        "fit_row_ids": list(owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(owner.fit_row_ids),
        "canonical_group_seed": int(owner.scope_seed),
        "heldout_labels_supplied": False,
        "peer_group_definitions_supplied": False,
    }


class _SafeArrayStore:
    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.inventory: dict[str, dict[str, Any]] = {}

    def add(self, key: str, value: Any) -> str:
        name = str(key)
        if _SAFE_ARRAY_KEY.fullmatch(name) is None or name in self.arrays:
            raise ValueError(f"invalid or duplicate HTR array key: {name}")
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise ValueError("HTR arrays cannot use object dtype")
        self.arrays[name] = array
        self.inventory[name] = {
            "dtype": array.dtype.str,
            "shape": [int(item) for item in array.shape],
            "content_sha256": _array_sha256(array),
        }
        return name


@dataclass(frozen=True)
class _CoveragePlan:
    summary: Mapping[str, Any]
    note_word_counts: np.ndarray
    note_chunk_counts: np.ndarray
    chunk_note_positions: np.ndarray
    chunk_word_starts: np.ndarray
    chunk_word_ends: np.ndarray
    chunk_sha256_bytes: np.ndarray
    chunks_by_note: tuple[tuple[str, ...], ...]


def _coverage_plan(
    *,
    texts: Sequence[str],
    config: RoleNeutralHTRConfig,
    phase: str,
) -> _CoveragePlan:
    values = tuple(texts)
    if not values or any(not isinstance(text, str) for text in values):
        raise ValueError(f"{phase} HTR coverage requires nonempty string-aligned notes")
    stride = config.chunk_size_words - config.chunk_overlap_words
    note_word_counts: list[int] = []
    note_chunk_counts: list[int] = []
    chunk_note_positions: list[int] = []
    chunk_word_starts: list[int] = []
    chunk_word_ends: list[int] = []
    chunk_digests: list[bytes] = []
    chunks_by_note: list[tuple[str, ...]] = []
    for note_position, text in enumerate(values):
        words = [match.group(0) for match in re.finditer(r"\S+", text)]
        word_count = len(words)
        required = max(1, (word_count + stride - 1) // stride)
        if required > config.max_chunks:
            raise ValueError(
                "configured HTR max_chunks would truncate a note; "
                f"phase={phase} note_position={note_position} "
                f"required={required} configured={config.max_chunks}"
            )
        expected_chunks = tuple(
            split_text_into_word_chunks(
                text,
                config.chunk_size_words,
                config.chunk_overlap_words,
                config.max_chunks,
            )
        )
        spans: list[tuple[int, int]] = []
        if word_count == 0:
            spans.append((0, 0))
        else:
            start = 0
            while start < word_count:
                end = min(word_count, start + config.chunk_size_words)
                spans.append((start, end))
                start += stride
        planned_chunks = tuple(
            "" if end == start else " ".join(words[start:end])
            for start, end in spans
        )
        if (
            len(spans) != required
            or planned_chunks != expected_chunks
            or len(expected_chunks) > config.max_chunks
        ):
            raise RuntimeError("HTR full-note chunk planner differs from live extractor")
        covered = np.zeros(word_count, dtype=np.uint8)
        for chunk_index, ((start, end), chunk) in enumerate(
            zip(spans, expected_chunks, strict=True)
        ):
            if end > start:
                covered[start:end] = 1
            chunk_note_positions.append(note_position)
            chunk_word_starts.append(start)
            chunk_word_ends.append(end)
            chunk_digests.append(hashlib.sha256(chunk.encode("utf-8")).digest())
            if chunk_index >= config.max_chunks:
                raise RuntimeError("HTR planner emitted a binding chunk cap")
        if word_count and not bool(np.all(covered == 1)):
            raise RuntimeError("HTR chunk plan omitted one or more note words")
        note_word_counts.append(word_count)
        note_chunk_counts.append(len(expected_chunks))
        chunks_by_note.append(expected_chunks)
    return _CoveragePlan(
        summary={
            "schema_version": ROLE_NEUTRAL_HTR_COVERAGE_SCHEMA,
            "phase": str(phase),
            "coverage_unit": "all_non_whitespace_words_v1",
            "note_count": len(values),
            "total_word_count": int(sum(note_word_counts)),
            "total_chunk_count": int(sum(note_chunk_counts)),
            "chunk_size_words": config.chunk_size_words,
            "chunk_overlap_words": config.chunk_overlap_words,
            "configured_max_chunks": config.max_chunks,
            "configured_max_chunk_length": config.max_chunk_length,
            "max_chunks_nonbinding": True,
            "semantic_truncation_applied": False,
        },
        note_word_counts=np.asarray(note_word_counts, dtype=np.int64),
        note_chunk_counts=np.asarray(note_chunk_counts, dtype=np.int64),
        chunk_note_positions=np.asarray(chunk_note_positions, dtype=np.int64),
        chunk_word_starts=np.asarray(chunk_word_starts, dtype=np.int64),
        chunk_word_ends=np.asarray(chunk_word_ends, dtype=np.int64),
        chunk_sha256_bytes=np.frombuffer(
            b"".join(chunk_digests),
            dtype=np.uint8,
        ).reshape(len(chunk_digests), 32),
        chunks_by_note=tuple(chunks_by_note),
    )


def _coverage_arrays(
    *,
    store: _SafeArrayStore,
    plan: _CoveragePlan,
    prefix: str,
) -> dict[str, Any]:
    return {
        **copy.deepcopy(dict(plan.summary)),
        "note_word_counts": store.add(
            f"{prefix}_note_word_counts",
            plan.note_word_counts,
        ),
        "note_chunk_counts": store.add(
            f"{prefix}_note_chunk_counts",
            plan.note_chunk_counts,
        ),
        "chunk_note_positions": store.add(
            f"{prefix}_chunk_note_positions",
            plan.chunk_note_positions,
        ),
        "chunk_word_starts": store.add(
            f"{prefix}_chunk_word_starts",
            plan.chunk_word_starts,
        ),
        "chunk_word_ends": store.add(
            f"{prefix}_chunk_word_ends",
            plan.chunk_word_ends,
        ),
        "chunk_sha256_bytes": store.add(
            f"{prefix}_chunk_sha256_bytes",
            plan.chunk_sha256_bytes,
        ),
    }


_COVERAGE_REFERENCE_FIELDS = (
    "note_word_counts",
    "note_chunk_counts",
    "chunk_note_positions",
    "chunk_word_starts",
    "chunk_word_ends",
    "chunk_sha256_bytes",
)


def _coverage_numeric_values(
    *,
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    config: RoleNeutralHTRConfig,
    expected_phase: str,
) -> dict[str, np.ndarray]:
    expected_keys = {
        "schema_version",
        "phase",
        "coverage_unit",
        "note_count",
        "total_word_count",
        "total_chunk_count",
        "chunk_size_words",
        "chunk_overlap_words",
        "configured_max_chunks",
        "configured_max_chunk_length",
        "max_chunks_nonbinding",
        "semantic_truncation_applied",
        *_COVERAGE_REFERENCE_FIELDS,
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_keys
        or record.get("schema_version") != ROLE_NEUTRAL_HTR_COVERAGE_SCHEMA
        or record.get("phase") != expected_phase
        or record.get("coverage_unit") != "all_non_whitespace_words_v1"
        or int(record.get("chunk_size_words", 0)) != config.chunk_size_words
        or int(record.get("chunk_overlap_words", -1))
        != config.chunk_overlap_words
        or int(record.get("configured_max_chunks", 0)) != config.max_chunks
        or int(record.get("configured_max_chunk_length", 0))
        != config.max_chunk_length
        or record.get("max_chunks_nonbinding") is not True
        or record.get("semantic_truncation_applied") is not False
    ):
        raise ValueError(f"{expected_phase} HTR coverage envelope changed")
    values: dict[str, np.ndarray] = {}
    for field in _COVERAGE_REFERENCE_FIELDS:
        key = str(record.get(field) or "")
        if key not in arrays:
            raise ValueError(f"{expected_phase} HTR coverage array is missing: {field}")
        values[field] = np.asarray(arrays[key])
    note_words = values["note_word_counts"]
    note_chunks = values["note_chunk_counts"]
    chunk_notes = values["chunk_note_positions"]
    starts = values["chunk_word_starts"]
    ends = values["chunk_word_ends"]
    digests = values["chunk_sha256_bytes"]
    note_count = int(record.get("note_count", -1))
    total_chunks = int(record.get("total_chunk_count", -1))
    if (
        note_count < 1
        or note_words.dtype.kind not in {"i", "u"}
        or note_chunks.dtype.kind not in {"i", "u"}
        or note_words.shape != (note_count,)
        or note_chunks.shape != (note_count,)
        or np.any(note_words < 0)
        or np.any(note_chunks < 1)
        or np.any(note_chunks > config.max_chunks)
        or int(np.sum(note_words)) != int(record.get("total_word_count", -1))
        or int(np.sum(note_chunks)) != total_chunks
        or chunk_notes.shape != (total_chunks,)
        or starts.shape != (total_chunks,)
        or ends.shape != (total_chunks,)
        or digests.shape != (total_chunks, 32)
        or digests.dtype != np.dtype("uint8")
        or not all(
            array.dtype.kind in {"i", "u"}
            for array in (chunk_notes, starts, ends)
        )
    ):
        raise ValueError(f"{expected_phase} HTR coverage array shapes changed")
    expected_note_positions = np.repeat(
        np.arange(note_count, dtype=np.int64),
        note_chunks.astype(np.int64),
    )
    if not np.array_equal(chunk_notes.astype(np.int64), expected_note_positions):
        raise ValueError(f"{expected_phase} HTR coverage note order changed")
    stride = config.chunk_size_words - config.chunk_overlap_words
    offset = 0
    for note_position in range(note_count):
        word_count = int(note_words[note_position])
        chunk_count = int(note_chunks[note_position])
        observed_starts = starts[offset : offset + chunk_count].astype(np.int64)
        observed_ends = ends[offset : offset + chunk_count].astype(np.int64)
        if word_count == 0:
            expected_spans = [(0, 0)]
        else:
            expected_spans = []
            start = 0
            while start < word_count:
                expected_spans.append(
                    (start, min(word_count, start + config.chunk_size_words))
                )
                start += stride
        if (
            len(expected_spans) != chunk_count
            or not np.array_equal(
                observed_starts,
                np.asarray([span[0] for span in expected_spans]),
            )
            or not np.array_equal(
                observed_ends,
                np.asarray([span[1] for span in expected_spans]),
            )
        ):
            raise ValueError(f"{expected_phase} HTR word-span coverage changed")
        offset += chunk_count
    return values


def _coverage_plan_values(plan: _CoveragePlan) -> dict[str, np.ndarray]:
    return {
        "note_word_counts": plan.note_word_counts,
        "note_chunk_counts": plan.note_chunk_counts,
        "chunk_note_positions": plan.chunk_note_positions,
        "chunk_word_starts": plan.chunk_word_starts,
        "chunk_word_ends": plan.chunk_word_ends,
        "chunk_sha256_bytes": plan.chunk_sha256_bytes,
    }


def _assert_coverage_matches_plan(
    observed: Mapping[str, np.ndarray],
    expected: _CoveragePlan,
    *,
    label: str,
) -> None:
    expected_values = _coverage_plan_values(expected)
    if any(
        not np.array_equal(np.asarray(observed[field]), expected_values[field])
        for field in _COVERAGE_REFERENCE_FIELDS
    ):
        raise ValueError(f"{label} differs from complete source-text coverage")


@dataclass(frozen=True)
class _ReusableTextPlan:
    phase: str
    ordered_row_ids: tuple[int, ...]
    text_sha256: str
    configuration_sha256: str
    text_rows: tuple[tuple[str, tuple[str, ...]], ...]
    token_rows: tuple[
        tuple[str, tuple[int, ...], tuple[int, ...]],
        ...,
    ]
    unique_note_count: int
    unique_chunk_count: int
    parallel_plan_task_count: int
    parallel_plan_thread_count: int
    positive_data_loader_workers_exercised: bool
    content_sha256: str

    def attestation(self) -> dict[str, Any]:
        body = {
            "schema_version": ROLE_NEUTRAL_HTR_REUSABLE_PLAN_SCHEMA,
            "phase": self.phase,
            "ordered_row_ids_sha256": _sha256_json(
                {"ordered_row_ids": list(self.ordered_row_ids)}
            ),
            "text_sha256": self.text_sha256,
            "configuration_sha256": self.configuration_sha256,
            "note_count": len(self.ordered_row_ids),
            "unique_note_count": self.unique_note_count,
            "unique_chunk_count": self.unique_chunk_count,
            "tokenized_unique_chunk_count": len(self.token_rows),
            "parallel_plan_task_count": self.parallel_plan_task_count,
            "parallel_plan_thread_count": self.parallel_plan_thread_count,
            "positive_data_loader_workers_exercised": (
                self.positive_data_loader_workers_exercised
            ),
            "raw_text_persisted": False,
            "semantic_truncation_applied": False,
        }
        if _sha256_json(body) != self.content_sha256:
            raise RuntimeError("in-process HTR reusable plan was mutated")
        return {**body, "content_sha256": self.content_sha256}


def _reusable_plan_configuration_sha256(
    *,
    config: RoleNeutralHTRConfig,
) -> str:
    return _sha256_json(
        {
            "schema_version": "production_role_neutral_htr_plan_config_v1",
            "sentence_encoder_model_kind": config.sentence_encoder_model_kind,
            "model_tree_sha256": config.model_tree_sha256,
            "sentence_encoder_backend": config.sentence_encoder_backend,
            "sentence_pooling": config.sentence_pooling,
            "chunk_size_words": config.chunk_size_words,
            "chunk_overlap_words": config.chunk_overlap_words,
            "max_chunks": config.max_chunks,
            "max_chunk_length": config.max_chunk_length,
        }
    )


def _build_reusable_text_plan(
    *,
    extractor: HierarchicalTransformerExtractor,
    texts: Sequence[str],
    row_ids: Sequence[int],
    coverage: _CoveragePlan,
    config: RoleNeutralHTRConfig,
    controls: RoleNeutralHTROperationalControls,
    phase: str,
) -> _ReusableTextPlan:
    controls.validate_for(config)
    if not controls.reuse_tokenizer_and_chunk_plans:
        raise ValueError("disabled HTR reuse cannot build a reusable plan")
    values = tuple(texts)
    rows = tuple(int(value) for value in row_ids)
    if (
        len(values) != len(rows)
        or len(values) != len(coverage.chunks_by_note)
        or any(not isinstance(value, str) for value in values)
    ):
        raise ValueError("HTR reusable plan rows, texts, and coverage differ")
    unique_notes = len(set(values))
    if unique_notes > controls.chunk_plan_cache_max_entries:
        raise ValueError(
            "configured HTR chunk-plan cache capacity would omit a note; "
            f"required={unique_notes} "
            f"configured={controls.chunk_plan_cache_max_entries}"
        )

    def prepare_row(
        row: tuple[int, str, tuple[str, ...]],
    ) -> tuple[int, str, tuple[str, ...], int]:
        position, text, expected_chunks = row
        observed_chunks = tuple(
            split_text_into_word_chunks(
                text,
                config.chunk_size_words,
                config.chunk_overlap_words,
                config.max_chunks,
            )
        )
        if observed_chunks != expected_chunks:
            raise RuntimeError(
                "parallel HTR data-loader plan changed complete chunk coverage"
            )
        return position, text, observed_chunks, threading.get_ident()

    work = tuple(
        (position, values[position], coverage.chunks_by_note[position])
        for position in range(len(values))
    )
    if controls.data_loader_workers:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=controls.data_loader_workers,
            thread_name_prefix="oci-htr-plan",
        ) as pool:
            prepared_rows = tuple(pool.map(prepare_row, work))
        parallel_plan_task_count = len(prepared_rows)
        parallel_plan_thread_count = len(
            {row[3] for row in prepared_rows}
        )
        positive_data_loader_workers_exercised = (
            parallel_plan_task_count == len(work)
            and parallel_plan_thread_count >= 1
        )
        if not positive_data_loader_workers_exercised:
            raise RuntimeError(
                "positive HTR data-loader workers did not execute every "
                "complete-text plan task"
            )
    else:
        prepared_rows = tuple(map(prepare_row, work))
        parallel_plan_task_count = 0
        parallel_plan_thread_count = 0
        positive_data_loader_workers_exercised = False
    if tuple(
        position for position, _text, _chunks, _thread_id in prepared_rows
    ) != tuple(
        range(len(values))
    ):
        raise RuntimeError("HTR data-loader workers reordered the text plan")

    text_cache = {
        text: chunks
        for _position, text, chunks, _thread_id in prepared_rows
    }
    if (
        len(text_cache) != unique_notes
        or any(
            text_cache[text] != chunks
            for _position, text, chunks, _thread_id in prepared_rows
        )
    ):
        raise RuntimeError("HTR reusable chunk plan has conflicting duplicate notes")
    unique_chunks = tuple(
        dict.fromkeys(
            chunk
            for _position, _text, chunks, _thread_id in prepared_rows
            for chunk in chunks
        )
    )
    if len(unique_chunks) > controls.tokenized_chunk_cache_max_entries:
        raise ValueError(
            "configured HTR tokenized-chunk cache capacity would omit a chunk; "
            f"required={len(unique_chunks)} "
            f"configured={controls.tokenized_chunk_cache_max_entries}"
        )

    # The preprocessor is CPU-only.  Its cache capacities are replaced with
    # the explicit deployment bounds before it sees any text.
    preprocessor = extractor.make_batch_preprocessor()
    preprocessor._chunk_cache_max_entries = (
        controls.chunk_plan_cache_max_entries
    )
    preprocessor._tokenization_cache_max_entries = (
        controls.tokenized_chunk_cache_max_entries
    )
    preprocessor._chunk_cache = {
        text: list(chunks) for text, chunks in text_cache.items()
    }
    for start in range(0, len(values), controls.training_batch_size):
        prepared = preprocessor(
            values[start : start + controls.training_batch_size]
        )
        observed = prepared.get("chunks")
        expected = [
            list(coverage.chunks_by_note[position])
            for position in range(
                start,
                min(len(values), start + controls.training_batch_size),
            )
        ]
        if observed != expected:
            raise RuntimeError("HTR reusable tokenizer plan changed chunk order")
    tokenize_for_transformers = bool(
        getattr(preprocessor, "_tokenize_for_transformers", False)
    )
    token_cache = dict(preprocessor._tokenization_cache)
    if tokenize_for_transformers and set(token_cache) != set(unique_chunks):
        raise RuntimeError(
            "HTR tokenizer plan omitted a unique complete-text chunk"
        )
    if not tokenize_for_transformers and token_cache:
        raise RuntimeError("non-transformer HTR plan unexpectedly tokenized chunks")
    token_rows = tuple(
        (
            chunk,
            tuple(token_cache[chunk][0]),
            tuple(token_cache[chunk][1]),
        )
        for chunk in sorted(token_cache)
    )
    body = {
        "schema_version": ROLE_NEUTRAL_HTR_REUSABLE_PLAN_SCHEMA,
        "phase": str(phase),
        "ordered_row_ids_sha256": _sha256_json(
            {"ordered_row_ids": list(rows)}
        ),
        "text_sha256": _text_sha256(rows, values),
        "configuration_sha256": _reusable_plan_configuration_sha256(
            config=config
        ),
        "note_count": len(rows),
        "unique_note_count": unique_notes,
        "unique_chunk_count": len(unique_chunks),
        "tokenized_unique_chunk_count": len(token_rows),
        "parallel_plan_task_count": parallel_plan_task_count,
        "parallel_plan_thread_count": parallel_plan_thread_count,
        "positive_data_loader_workers_exercised": (
            positive_data_loader_workers_exercised
        ),
        "raw_text_persisted": False,
        "semantic_truncation_applied": False,
    }
    return _ReusableTextPlan(
        phase=str(phase),
        ordered_row_ids=rows,
        text_sha256=body["text_sha256"],
        configuration_sha256=body["configuration_sha256"],
        text_rows=tuple(
            (text, chunks) for text, chunks in sorted(text_cache.items())
        ),
        token_rows=token_rows,
        unique_note_count=unique_notes,
        unique_chunk_count=len(unique_chunks),
        parallel_plan_task_count=parallel_plan_task_count,
        parallel_plan_thread_count=parallel_plan_thread_count,
        positive_data_loader_workers_exercised=(
            positive_data_loader_workers_exercised
        ),
        content_sha256=_sha256_json(body),
    )


def _install_reusable_text_plan(
    *,
    extractor: HierarchicalTransformerExtractor,
    plan: _ReusableTextPlan,
    texts: Sequence[str],
    row_ids: Sequence[int],
    config: RoleNeutralHTRConfig,
    controls: RoleNeutralHTROperationalControls,
) -> None:
    controls.validate_for(config)
    attestation = plan.attestation()
    values = tuple(texts)
    rows = tuple(int(value) for value in row_ids)
    if (
        rows != plan.ordered_row_ids
        or _text_sha256(rows, values) != plan.text_sha256
        or plan.configuration_sha256
        != _reusable_plan_configuration_sha256(config=config)
        or attestation["unique_note_count"]
        > controls.chunk_plan_cache_max_entries
        or attestation["unique_chunk_count"]
        > controls.tokenized_chunk_cache_max_entries
    ):
        raise ValueError("HTR reusable plan differs from its authorized rows")
    text_cache = {text: chunks for text, chunks in plan.text_rows}
    if set(values) != set(text_cache):
        raise ValueError("HTR reusable plan omits or substitutes a note")
    extractor._chunk_cache_max_entries = (
        controls.chunk_plan_cache_max_entries
    )
    extractor._tokenization_cache_max_entries = (
        controls.tokenized_chunk_cache_max_entries
    )
    extractor._chunk_cache = {
        text: list(chunks) for text, chunks in text_cache.items()
    }
    extractor._tokenization_cache = {
        chunk: (input_ids, attention_mask)
        for chunk, input_ids, attention_mask in plan.token_rows
    }
    observed_chunks = extractor._chunks_for_texts(values)
    if tuple(tuple(row) for row in observed_chunks) != tuple(
        text_cache[text] for text in values
    ):
        raise RuntimeError("installed HTR chunk plan changed note coverage")


@dataclass(frozen=True)
class _MaterializedReusableTextPlan:
    """Authenticated locator for one owner-scoped temporary Arrow plan."""

    root: str
    manifest_sha256: str
    manifest_size_bytes: int
    content_sha256: str

    def attestation(self) -> dict[str, Any]:
        return {
            "schema_version": ROLE_NEUTRAL_HTR_PROCESS_PLAN_SCHEMA,
            "manifest_sha256": _require_sha256(
                self.manifest_sha256,
                label="HTR process-plan manifest",
            ),
            "manifest_size_bytes": int(self.manifest_size_bytes),
            "content_sha256": _require_sha256(
                self.content_sha256,
                label="HTR process-plan content",
            ),
            "complete_owner_tokenizer_chunk_plan_built_once": True,
            "fold_workers_retokenized": False,
            "fold_workers_rechunked": False,
            "parent_authenticated_full_bytes_while_publishing": True,
            "child_duplicate_full_hash_pass": False,
            "child_reopen_guarded_by_authenticated_stat_inventory": True,
            "locator_included": False,
            "raw_text_temporary_only": True,
            "semantic_truncation_applied": False,
        }


def _write_new_arrow_table(path: Path, table: Any) -> None:
    """Publish one Arrow IPC file durably without replacing an existing path."""

    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError(
            "process-isolated HTR fold execution requires pyarrow"
        ) from exc
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace immutable Arrow plan: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(table, pa.Table):
        raise TypeError("HTR temporary Arrow payload must be a pyarrow Table")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        suffix=".arrow",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with pa.OSFile(str(temporary), "wb") as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
        file_descriptor = os.open(
            temporary,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        os.replace(temporary, target)
        directory = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _regular_file_stat_identity(path: Path, *, label: str) -> dict[str, int]:
    observed = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(observed.st_mode) or int(observed.st_nlink) != 1:
        raise ValueError(f"{label} must be one singly-linked regular file")
    return {
        "st_dev": int(observed.st_dev),
        "st_ino": int(observed.st_ino),
        "st_mode": int(observed.st_mode),
        "st_nlink": int(observed.st_nlink),
        "st_size": int(observed.st_size),
        "st_mtime_ns": int(observed.st_mtime_ns),
        "st_ctime_ns": int(observed.st_ctime_ns),
    }


def _materialize_reusable_text_plan(
    *,
    root: Path,
    plan: _ReusableTextPlan,
    coverage: _CoveragePlan,
    texts: Sequence[str],
    row_ids: Sequence[int],
) -> _MaterializedReusableTextPlan:
    """Write one complete owner text/token plan for process workers to mmap."""

    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError(
            "process-isolated HTR fold execution requires pyarrow"
        ) from exc
    target = Path(root)
    if not target.is_absolute():
        raise ValueError("HTR process-plan root must be absolute")
    if target.exists() or target.is_symlink():
        raise FileExistsError("HTR process-plan root must be fresh")
    values = tuple(texts)
    rows = tuple(int(value) for value in row_ids)
    if (
        rows != plan.ordered_row_ids
        or len(values) != len(rows)
        or _text_sha256(rows, values) != plan.text_sha256
        or tuple(coverage.chunks_by_note)
        != tuple(
            dict(plan.text_rows)[text]
            for text in values
        )
    ):
        raise ValueError("HTR process plan differs from the complete owner plan")
    target.mkdir(parents=True, exist_ok=False)
    starts_by_note: list[list[int]] = []
    ends_by_note: list[list[int]] = []
    digests_by_note: list[list[bytes]] = []
    offset = 0
    for chunk_count in coverage.note_chunk_counts.astype(np.int64):
        count = int(chunk_count)
        starts_by_note.append(
            coverage.chunk_word_starts[offset : offset + count]
            .astype(np.int64)
            .tolist()
        )
        ends_by_note.append(
            coverage.chunk_word_ends[offset : offset + count]
            .astype(np.int64)
            .tolist()
        )
        digests_by_note.append(
            [
                bytes(row.tolist())
                for row in coverage.chunk_sha256_bytes[
                    offset : offset + count
                ]
            ]
        )
        offset += count
    if offset != len(coverage.chunk_note_positions):
        raise RuntimeError("HTR process plan lost a complete chunk span")
    notes_schema = pa.schema(
        [
            ("position", pa.int64()),
            ("row_id", pa.int64()),
            ("text", pa.large_string()),
            ("chunks", pa.large_list(pa.large_string())),
            ("note_word_count", pa.int64()),
            ("chunk_word_starts", pa.large_list(pa.int64())),
            ("chunk_word_ends", pa.large_list(pa.int64())),
            ("chunk_sha256", pa.large_list(pa.binary(32))),
        ]
    )
    notes = pa.Table.from_arrays(
        [
            pa.array(range(len(values)), type=pa.int64()),
            pa.array(rows, type=pa.int64()),
            pa.array(values, type=pa.large_string()),
            pa.array(coverage.chunks_by_note, type=pa.large_list(pa.large_string())),
            pa.array(
                coverage.note_word_counts.astype(np.int64),
                type=pa.int64(),
            ),
            pa.array(starts_by_note, type=pa.large_list(pa.int64())),
            pa.array(ends_by_note, type=pa.large_list(pa.int64())),
            pa.array(digests_by_note, type=pa.large_list(pa.binary(32))),
        ],
        schema=notes_schema,
    )
    tokens_schema = pa.schema(
        [
            ("chunk", pa.large_string()),
            ("input_ids", pa.large_list(pa.int64())),
            ("attention_mask", pa.large_list(pa.int64())),
        ]
    )
    tokens = pa.Table.from_arrays(
        [
            pa.array(
                [row[0] for row in plan.token_rows],
                type=pa.large_string(),
            ),
            pa.array(
                [row[1] for row in plan.token_rows],
                type=pa.large_list(pa.int64()),
            ),
            pa.array(
                [row[2] for row in plan.token_rows],
                type=pa.large_list(pa.int64()),
            ),
        ],
        schema=tokens_schema,
    )
    notes_path = target / "notes.arrow"
    tokens_path = target / "tokens.arrow"
    _write_new_arrow_table(notes_path, notes)
    _write_new_arrow_table(tokens_path, tokens)
    files: dict[str, Any] = {}
    for name, path in (("notes", notes_path), ("tokens", tokens_path)):
        digest, size = _sha256_file(path, label=f"HTR process plan {name}")
        files[name] = {
            "relative_path": path.relative_to(target).as_posix(),
            "sha256": digest,
            "size_bytes": size,
            "authenticated_stat_identity": _regular_file_stat_identity(
                path,
                label=f"HTR process plan {name}",
            ),
        }
    body = {
        "schema_version": ROLE_NEUTRAL_HTR_PROCESS_PLAN_SCHEMA,
        "reusable_plan": plan.attestation(),
        "coverage_summary": copy.deepcopy(dict(coverage.summary)),
        "files": files,
        "file_order": ["notes", "tokens"],
        "arrow_memory_map_safe": True,
        "complete_owner_plan_built_once": True,
        "fold_workers_retokenize": False,
        "fold_workers_rechunk": False,
        "parent_authenticated_full_bytes_while_publishing": True,
        "child_duplicate_full_hash_pass": False,
        "child_reopen_guarded_by_authenticated_stat_inventory": True,
        "raw_text_temporary_only": True,
        "semantic_truncation_applied": False,
    }
    manifest = {**body, "content_sha256": _sha256_json(body)}
    manifest_path = target / "manifest.json"
    _write_new_json(manifest_path, manifest)
    manifest_sha256, manifest_size = _sha256_file(
        manifest_path,
        label="HTR process-plan manifest",
    )
    return _MaterializedReusableTextPlan(
        root=str(target),
        manifest_sha256=manifest_sha256,
        manifest_size_bytes=manifest_size,
        content_sha256=manifest["content_sha256"],
    )


def _load_materialized_reusable_text_plan(
    descriptor: _MaterializedReusableTextPlan,
) -> tuple[tuple[str, ...], tuple[int, ...], _CoveragePlan, _ReusableTextPlan]:
    """Freshly authenticate and mmap one process worker's complete plan."""

    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError(
            "process-isolated HTR fold execution requires pyarrow"
        ) from exc
    if not isinstance(descriptor, _MaterializedReusableTextPlan):
        raise TypeError("HTR process worker requires a typed plan descriptor")
    root = Path(descriptor.root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("HTR process-plan locator is not one real directory")
    manifest_path = root / "manifest.json"
    observed_sha256, observed_size = _sha256_file(
        manifest_path,
        label="HTR process-plan manifest",
    )
    if (
        observed_sha256 != descriptor.manifest_sha256
        or observed_size != descriptor.manifest_size_bytes
    ):
        raise RuntimeError("HTR process-plan manifest changed")
    manifest = _read_json(manifest_path, label="HTR process-plan manifest")
    content_sha256 = manifest.pop("content_sha256", None)
    if (
        content_sha256 != descriptor.content_sha256
        or _sha256_json(manifest) != content_sha256
        or manifest.get("schema_version") != ROLE_NEUTRAL_HTR_PROCESS_PLAN_SCHEMA
        or manifest.get("file_order") != ["notes", "tokens"]
        or manifest.get("arrow_memory_map_safe") is not True
        or manifest.get("complete_owner_plan_built_once") is not True
        or manifest.get("fold_workers_retokenize") is not False
        or manifest.get("fold_workers_rechunk") is not False
        or manifest.get("parent_authenticated_full_bytes_while_publishing")
        is not True
        or manifest.get("child_duplicate_full_hash_pass") is not False
        or manifest.get(
            "child_reopen_guarded_by_authenticated_stat_inventory"
        )
        is not True
        or manifest.get("raw_text_temporary_only") is not True
        or manifest.get("semantic_truncation_applied") is not False
    ):
        raise RuntimeError("HTR process-plan manifest content changed")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {"notes", "tokens"}:
        raise ValueError("HTR process-plan inventory changed")
    tables: dict[str, Any] = {}
    expected_files = {"manifest.json"}
    for name in ("notes", "tokens"):
        record = files[name]
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path",
            "sha256",
            "size_bytes",
            "authenticated_stat_identity",
        }:
            raise ValueError("HTR process-plan file record changed")
        relative = str(record["relative_path"])
        if relative != f"{name}.arrow":
            raise ValueError("HTR process-plan file locator changed")
        expected_files.add(relative)
        path = root / relative
        expected_stat = record["authenticated_stat_identity"]
        if (
            not isinstance(expected_stat, Mapping)
            or set(expected_stat)
            != {
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            }
            or int(expected_stat["st_size"]) != int(record["size_bytes"])
            or _regular_file_stat_identity(
                path,
                label=f"HTR process plan {name}",
            )
            != dict(expected_stat)
        ):
            raise RuntimeError(f"HTR process-plan {name} locator changed")
        with pa.memory_map(str(path), "r") as source:
            tables[name] = ipc.open_file(source).read_all()
        if _regular_file_stat_identity(
            path,
            label=f"HTR process plan {name}",
        ) != dict(expected_stat):
            raise RuntimeError(f"HTR process-plan {name} changed while mapped")
    files_on_disk, directories = _inventory_tree(root)
    if files_on_disk != expected_files or directories:
        raise ValueError("HTR process-plan tree contains unregistered entries")
    notes = tables["notes"]
    tokens = tables["tokens"]
    expected_note_columns = (
        "position",
        "row_id",
        "text",
        "chunks",
        "note_word_count",
        "chunk_word_starts",
        "chunk_word_ends",
        "chunk_sha256",
    )
    expected_token_columns = ("chunk", "input_ids", "attention_mask")
    if (
        tuple(notes.column_names) != expected_note_columns
        or tuple(tokens.column_names) != expected_token_columns
    ):
        raise ValueError("HTR process-plan Arrow schema changed")
    positions = tuple(int(value) for value in notes["position"].to_pylist())
    if positions != tuple(range(len(positions))):
        raise ValueError("HTR process-plan note order changed")
    row_ids = tuple(int(value) for value in notes["row_id"].to_pylist())
    texts = tuple(str(value) for value in notes["text"].to_pylist())
    chunks_by_note = tuple(
        tuple(str(chunk) for chunk in row)
        for row in notes["chunks"].to_pylist()
    )
    note_word_counts = np.asarray(
        notes["note_word_count"].to_pylist(),
        dtype=np.int64,
    )
    starts_by_note = notes["chunk_word_starts"].to_pylist()
    ends_by_note = notes["chunk_word_ends"].to_pylist()
    digests_by_note = notes["chunk_sha256"].to_pylist()
    note_chunk_counts = np.asarray(
        [len(row) for row in chunks_by_note],
        dtype=np.int64,
    )
    if any(
        len({len(chunks), len(starts), len(ends), len(digests)}) != 1
        for chunks, starts, ends, digests in zip(
            chunks_by_note,
            starts_by_note,
            ends_by_note,
            digests_by_note,
            strict=True,
        )
    ):
        raise ValueError("HTR process-plan chunk spans changed")
    chunk_note_positions = np.repeat(
        np.arange(len(texts), dtype=np.int64),
        note_chunk_counts,
    )
    chunk_word_starts = np.asarray(
        [value for row in starts_by_note for value in row],
        dtype=np.int64,
    )
    chunk_word_ends = np.asarray(
        [value for row in ends_by_note for value in row],
        dtype=np.int64,
    )
    flat_digests = [
        bytes(value)
        for row in digests_by_note
        for value in row
    ]
    chunk_sha256_bytes = np.frombuffer(
        b"".join(flat_digests),
        dtype=np.uint8,
    ).reshape(len(flat_digests), 32)
    coverage = _CoveragePlan(
        summary=copy.deepcopy(dict(manifest["coverage_summary"])),
        note_word_counts=note_word_counts,
        note_chunk_counts=note_chunk_counts,
        chunk_note_positions=chunk_note_positions,
        chunk_word_starts=chunk_word_starts,
        chunk_word_ends=chunk_word_ends,
        chunk_sha256_bytes=chunk_sha256_bytes,
        chunks_by_note=chunks_by_note,
    )
    reusable_attestation = manifest.get("reusable_plan")
    if not isinstance(reusable_attestation, Mapping):
        raise ValueError("HTR process plan lacks its reusable-plan attestation")
    text_cache: dict[str, tuple[str, ...]] = {}
    for text, chunks in zip(texts, chunks_by_note, strict=True):
        previous = text_cache.setdefault(text, chunks)
        if previous != chunks:
            raise ValueError("HTR process plan has conflicting duplicate notes")
    token_rows = tuple(
        (
            str(chunk),
            tuple(int(value) for value in input_ids),
            tuple(int(value) for value in attention_mask),
        )
        for chunk, input_ids, attention_mask in zip(
            tokens["chunk"].to_pylist(),
            tokens["input_ids"].to_pylist(),
            tokens["attention_mask"].to_pylist(),
            strict=True,
        )
    )
    plan = _ReusableTextPlan(
        phase=str(reusable_attestation["phase"]),
        ordered_row_ids=row_ids,
        text_sha256=str(reusable_attestation["text_sha256"]),
        configuration_sha256=str(
            reusable_attestation["configuration_sha256"]
        ),
        text_rows=tuple(sorted(text_cache.items())),
        token_rows=token_rows,
        unique_note_count=int(reusable_attestation["unique_note_count"]),
        unique_chunk_count=int(reusable_attestation["unique_chunk_count"]),
        parallel_plan_task_count=int(
            reusable_attestation["parallel_plan_task_count"]
        ),
        parallel_plan_thread_count=int(
            reusable_attestation["parallel_plan_thread_count"]
        ),
        positive_data_loader_workers_exercised=bool(
            reusable_attestation[
                "positive_data_loader_workers_exercised"
            ]
        ),
        content_sha256=str(reusable_attestation["content_sha256"]),
    )
    if (
        plan.attestation() != dict(reusable_attestation)
        or plan.ordered_row_ids != row_ids
        or _text_sha256(row_ids, texts) != plan.text_sha256
        or int(np.sum(note_chunk_counts))
        != int(coverage.summary["total_chunk_count"])
        or len(
            {
                chunk
                for chunks in chunks_by_note
                for chunk in chunks
            }
        )
        != plan.unique_chunk_count
        or tuple(row[0] for row in token_rows)
        != tuple(sorted(row[0] for row in token_rows))
        or any(
            hashlib.sha256(chunk.encode("utf-8")).digest() != digest
            for chunks, digests in zip(
                chunks_by_note,
                digests_by_note,
                strict=True,
            )
            for chunk, digest in zip(chunks, digests, strict=True)
        )
    ):
        raise RuntimeError("HTR process plan differs from its authenticated content")
    return texts, row_ids, coverage, plan


def _set_operational_encoder_batch_size(
    extractor: HierarchicalTransformerExtractor,
    *,
    config: RoleNeutralHTRConfig,
    controls: RoleNeutralHTROperationalControls,
) -> None:
    controls.validate_for(config)
    if int(extractor._sentence_encoder_batch_size) != int(
        config.sentence_encoder_batch_size
    ):
        raise RuntimeError(
            "HTR extractor did not begin with its scientific encoder batch"
        )
    extractor._sentence_encoder_batch_size = int(
        controls.sentence_encoder_batch_size
    )


def _restore_scientific_encoder_batch_size(
    extractor: HierarchicalTransformerExtractor,
    *,
    config: RoleNeutralHTRConfig,
    controls: RoleNeutralHTROperationalControls,
) -> None:
    observed = int(extractor._sentence_encoder_batch_size)
    scientific = int(config.sentence_encoder_batch_size)
    operational = int(controls.sentence_encoder_batch_size)
    if observed == scientific:
        return
    if observed != operational:
        raise RuntimeError("HTR operational encoder batch changed during execution")
    extractor._sentence_encoder_batch_size = scientific


def _model_tree_sha256(path: Path) -> str:
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("HTR model tree must be one real directory")
    files: list[Path] = []
    for candidate in root.rglob("*"):
        observed = os.stat(candidate, follow_symlinks=False)
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError("HTR model tree cannot contain symbolic links")
        if stat.S_ISREG(observed.st_mode):
            if int(observed.st_nlink) != 1:
                raise ValueError("HTR model tree cannot contain hard-linked files")
            files.append(candidate)
        elif not stat.S_ISDIR(observed.st_mode):
            raise ValueError("HTR model tree contains a special file")
    if not files:
        raise ValueError("HTR model tree contains no regular files")
    rows = []
    for candidate in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root).as_posix()
        digest, size = _sha256_file(
            candidate,
            label=f"HTR model tree file {relative}",
        )
        rows.append(
            {
                "relative_path": relative,
                "size": size,
                "sha256": digest,
            }
        )
    return _sha256_json(rows)


def _resolve_model_marker(
    config: RoleNeutralHTRConfig,
    htr_model_path: Path | str | None,
) -> str:
    if config.sentence_encoder_model_kind == "hash":
        if htr_model_path is not None:
            raise ValueError("hash HTR cannot receive a model-tree locator")
        return "hash"
    if htr_model_path is None:
        raise ValueError("authenticated HTR requires a local model-tree locator")
    path = Path(htr_model_path)
    if _model_tree_sha256(path) != config.model_tree_sha256:
        raise RuntimeError("authenticated HTR model tree differs from configuration")
    return str(path.resolve())


def _new_extractor(
    *,
    config: RoleNeutralHTRConfig,
    model_marker: str,
    device: torch.device,
) -> HierarchicalTransformerExtractor:
    return HierarchicalTransformerExtractor(
        sentence_encoder_model=model_marker,
        freeze_sentence_encoder=config.freeze_sentence_encoder,
        chunk_size_words=config.chunk_size_words,
        chunk_overlap_words=config.chunk_overlap_words,
        max_chunks=config.max_chunks,
        max_chunk_length=config.max_chunk_length,
        num_transformer_layers=config.num_transformer_layers,
        num_attention_heads=config.num_attention_heads,
        transformer_dim=config.transformer_dim,
        transformer_dropout=config.transformer_dropout,
        projection_dim=config.projection_dim,
        hash_embedding_dim=config.hash_embedding_dim,
        sentence_encoder_batch_size=config.sentence_encoder_batch_size,
        sentence_encoder_backend=config.sentence_encoder_backend,
        sentence_pooling=config.sentence_pooling,
        normalize_sentence_embeddings=config.normalize_sentence_embeddings,
        trainable_sentence_encoder_layers=(
            config.trainable_sentence_encoder_layers
        ),
        role_attention=config.role_attention,
        w_attention_heads=config.w_attention_heads,
        x_attention_heads=config.x_attention_heads,
        transformer_feedforward_dim=config.transformer_feedforward_dim,
        transformer_activation=config.transformer_activation,
        transformer_norm_style=config.transformer_norm_style,
        transformer_layer_norm_eps=config.transformer_layer_norm_eps,
        transformer_layer_norm_elementwise_affine=(
            config.transformer_layer_norm_elementwise_affine
        ),
        transformer_layer_norm_bias=config.transformer_layer_norm_bias,
        transformer_attention_dropout=config.transformer_attention_dropout,
        transformer_residual_dropout=config.transformer_residual_dropout,
        transformer_feedforward_dropout=config.transformer_feedforward_dropout,
        transformer_attention_bias=config.transformer_attention_bias,
        transformer_feedforward_bias=config.transformer_feedforward_bias,
        output_projection_depth=config.output_projection_depth,
        output_projection_hidden_dim=config.output_projection_hidden_dim,
        output_projection_activation=config.output_projection_activation,
        output_projection_dropout=config.output_projection_dropout,
        output_projection_hidden_layer_norm=(
            config.output_projection_hidden_layer_norm
        ),
        output_projection_final_layer_norm=(
            config.output_projection_final_layer_norm
        ),
        output_projection_bias=config.output_projection_bias,
        pool_token_init_std=config.pool_token_init_std,
        positional_encoding_base=config.positional_encoding_base,
        environment_override_policy=config.environment_override_policy,
        device=device,
    )


def _expected_extractor_constructor(
    config: RoleNeutralHTRConfig,
) -> dict[str, Any]:
    return {
        "sentence_encoder_model": (
            "hash"
            if config.sentence_encoder_model_kind == "hash"
            else "authenticated_local_tree"
        ),
        "freeze_sentence_encoder": config.freeze_sentence_encoder,
        "chunk_size_words": config.chunk_size_words,
        "chunk_overlap_words": config.chunk_overlap_words,
        "max_chunks": config.max_chunks,
        "max_chunk_length": config.max_chunk_length,
        "num_transformer_layers": config.num_transformer_layers,
        "num_attention_heads": config.num_attention_heads,
        "transformer_dim": config.transformer_dim,
        "transformer_dropout": config.transformer_dropout,
        "projection_dim": config.projection_dim,
        "hash_embedding_dim": config.hash_embedding_dim,
        "sentence_encoder_batch_size": config.sentence_encoder_batch_size,
        "sentence_encoder_backend": config.sentence_encoder_backend,
        "sentence_pooling": config.sentence_pooling,
        "normalize_sentence_embeddings": config.normalize_sentence_embeddings,
        "trainable_sentence_encoder_layers": (
            config.trainable_sentence_encoder_layers
        ),
        "role_attention": config.role_attention,
        "w_attention_heads": config.w_attention_heads,
        "x_attention_heads": config.x_attention_heads,
        "transformer_feedforward_dim": config.transformer_feedforward_dim,
        "transformer_activation": config.transformer_activation,
        "transformer_norm_style": config.transformer_norm_style,
        "transformer_layer_norm_eps": config.transformer_layer_norm_eps,
        "transformer_layer_norm_elementwise_affine": (
            config.transformer_layer_norm_elementwise_affine
        ),
        "transformer_layer_norm_bias": config.transformer_layer_norm_bias,
        "transformer_attention_dropout": (
            config.transformer_attention_dropout
        ),
        "transformer_residual_dropout": (
            config.transformer_residual_dropout
        ),
        "transformer_feedforward_dropout": (
            config.transformer_feedforward_dropout
        ),
        "transformer_attention_bias": config.transformer_attention_bias,
        "transformer_feedforward_bias": config.transformer_feedforward_bias,
        "output_projection_depth": config.output_projection_depth,
        "output_projection_hidden_dim": config.output_projection_hidden_dim,
        "output_projection_activation": config.output_projection_activation,
        "output_projection_dropout": config.output_projection_dropout,
        "output_projection_hidden_layer_norm": (
            config.output_projection_hidden_layer_norm
        ),
        "output_projection_final_layer_norm": (
            config.output_projection_final_layer_norm
        ),
        "output_projection_bias": config.output_projection_bias,
        "pool_token_init_std": config.pool_token_init_std,
        "positional_encoding_base": config.positional_encoding_base,
        "environment_override_policy": config.environment_override_policy,
    }


def _attest_extractor(
    extractor: HierarchicalTransformerExtractor,
    *,
    config: RoleNeutralHTRConfig,
) -> dict[str, Any]:
    descriptor = _extractor_descriptor(extractor)
    audit = extractor.sentence_encoder_training_audit()
    if descriptor.get("constructor") != _expected_extractor_constructor(config):
        raise RuntimeError(
            "HTR extractor constructor differs from its typed scientific configuration"
        )
    if (
        config.sentence_encoder_model_kind != "hash"
        and descriptor.get("effective_sentence_encoder_backend") != "transformers"
    ):
        raise RuntimeError(
            "lossless HTR execution currently requires the transformers "
            "sentence-encoder backend so truncation=False token counts can be "
            "authenticated before fitting"
        )
    if config.require_live_unfrozen_encoder_attestation:
        if (
            config.freeze_sentence_encoder is not False
            or audit.get("effective_backend") != "transformers"
            or audit.get("encoder_initialized") is not True
            or audit.get("sentence_encoder_present") is not True
            or int(audit.get("sentence_encoder_parameter_tensors", 0)) <= 0
            or audit.get("all_sentence_encoder_parameters_trainable") is not True
        ):
            raise RuntimeError(
                "live unfrozen HTR sentence encoder does not satisfy its "
                "configured production attestation"
            )
    return {
        "extractor_descriptor": descriptor,
        "sentence_encoder_training_audit": audit,
    }


def _preflight_token_lengths(
    extractor: HierarchicalTransformerExtractor,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> None:
    values = tuple(texts)
    for start in range(0, len(values), int(batch_size)):
        # The preprocessor calls the tokenizer with truncation=False and
        # raises if any configured max_chunk_length would bind.
        prepared = extractor.prepare_batch(
            values[start : start + int(batch_size)]
        )
        chunks = prepared.get("chunks")
        if not isinstance(chunks, list):
            raise RuntimeError("HTR preprocessor omitted its complete chunk plan")


def _set_model_seed(seed: int, device: torch.device) -> None:
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _batch_positions(
    positions: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> Iterable[np.ndarray]:
    generator = np.random.default_rng(int(seed) + int(epoch))
    shuffled = np.asarray(positions, dtype=np.int64).copy()
    generator.shuffle(shuffled)
    for start in range(0, len(shuffled), int(batch_size)):
        yield shuffled[start : start + int(batch_size)]


def _clip_and_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    gradient_clip_norm: float,
    gradient_clip_norm_type: float,
    gradient_clip_error_if_nonfinite: bool,
    gradient_clip_foreach: bool,
) -> None:
    if gradient_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip_norm,
            norm_type=gradient_clip_norm_type,
            error_if_nonfinite=gradient_clip_error_if_nonfinite,
            foreach=gradient_clip_foreach,
        )
    optimizer.step()


def _adamw_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    config: RoleNeutralHTRConfig,
) -> torch.optim.AdamW:
    if config.optimizer_name != "adamw":
        raise ValueError("unsupported role-neutral HTR optimizer")
    return torch.optim.AdamW(
        list(parameters),
        lr=config.learning_rate,
        betas=(config.adamw_beta1, config.adamw_beta2),
        eps=config.adamw_eps,
        weight_decay=config.weight_decay,
        amsgrad=config.adamw_amsgrad,
        maximize=config.adamw_maximize,
        foreach=config.adamw_foreach,
        capturable=config.adamw_capturable,
        differentiable=config.adamw_differentiable,
        fused=config.adamw_fused,
    )


def _training_configuration(
    config: RoleNeutralHTRConfig,
    *,
    kind: str,
) -> dict[str, Any]:
    if kind not in {"nuisance", "effect"}:
        raise ValueError("unsupported HTR training-configuration kind")
    return {
        "optimizer_name": config.optimizer_name,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "adamw_beta1": config.adamw_beta1,
        "adamw_beta2": config.adamw_beta2,
        "adamw_eps": config.adamw_eps,
        "adamw_amsgrad": config.adamw_amsgrad,
        "adamw_maximize": config.adamw_maximize,
        "adamw_foreach": config.adamw_foreach,
        "adamw_capturable": config.adamw_capturable,
        "adamw_differentiable": config.adamw_differentiable,
        "adamw_fused": config.adamw_fused,
        "optimizer_zero_grad_set_to_none": (
            config.optimizer_zero_grad_set_to_none
        ),
        "gradient_clip_norm": config.gradient_clip_norm,
        "gradient_clip_norm_type": config.gradient_clip_norm_type,
        "gradient_clip_error_if_nonfinite": (
            config.gradient_clip_error_if_nonfinite
        ),
        "gradient_clip_foreach": config.gradient_clip_foreach,
        "batch_size": config.batch_size,
        "epochs": (
            config.nuisance_epochs
            if kind == "nuisance"
            else config.effect_epochs
        ),
    }


def _validate_typed_model_configuration(
    descriptor: Mapping[str, Any],
    *,
    config: RoleNeutralHTRConfig,
    kind: str,
) -> None:
    if kind == "nuisance":
        expected_head = {
            "hidden_dim": config.hidden_dim,
            "depth": config.nuisance_head_depth,
            "activation": config.nuisance_head_activation,
            "dropout": config.nuisance_head_dropout,
            "layer_norm": config.nuisance_head_layer_norm,
            "bias": config.nuisance_head_bias,
        }
    elif kind == "effect":
        expected_head = {
            "hidden_dim": config.hidden_dim,
            "depth": config.effect_head_depth,
            "activation": config.effect_head_activation,
            "dropout": config.effect_head_dropout,
            "layer_norm": config.effect_head_layer_norm,
            "bias": config.effect_head_bias,
        }
    else:
        raise ValueError("unsupported HTR model descriptor kind")
    if (
        not isinstance(descriptor, Mapping)
        or descriptor.get("kind") != kind
        or descriptor.get("head_configuration") != expected_head
        or descriptor.get("training_configuration")
        != _training_configuration(config, kind=kind)
        or (descriptor.get("extractor") or {}).get("constructor")
        != _expected_extractor_constructor(config)
    ):
        raise ValueError(
            f"HTR {kind} model differs from its typed scientific configuration"
        )


def _train_nuisance(
    model: _NuisanceNet,
    *,
    texts: tuple[str, ...],
    treatment: np.ndarray,
    outcome: np.ndarray,
    positions: np.ndarray,
    config: RoleNeutralHTRConfig,
    seed: int,
    device: torch.device,
) -> None:
    optimizer = _adamw_optimizer(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        config=config,
    )
    for epoch in range(config.nuisance_epochs):
        model.train()
        for batch_pos in _batch_positions(
            positions,
            batch_size=config.batch_size,
            seed=seed,
            epoch=epoch,
        ):
            batch_texts = [texts[int(position)] for position in batch_pos]
            t = torch.as_tensor(
                treatment[batch_pos],
                dtype=torch.float32,
                device=device,
            )
            y = torch.as_tensor(
                outcome[batch_pos],
                dtype=torch.float32,
                device=device,
            )
            if config.nuisance_label_smoothing > 0.0:
                smoothing = config.nuisance_label_smoothing
                t = t * (1.0 - smoothing) + 0.5 * smoothing
                y = y * (1.0 - smoothing) + 0.5 * smoothing
            optimizer.zero_grad(
                set_to_none=config.optimizer_zero_grad_set_to_none
            )
            t_logit, y_logit = model(batch_texts)
            propensity_loss = F.binary_cross_entropy_with_logits(t_logit, t)
            outcome_loss = F.binary_cross_entropy_with_logits(y_logit, y)
            loss = outcome_loss + config.alpha_propensity * propensity_loss
            if not torch.isfinite(loss):
                raise RuntimeError("HTR nuisance optimization emitted non-finite loss")
            loss.backward()
            _clip_and_step(
                model,
                optimizer,
                gradient_clip_norm=config.gradient_clip_norm,
                gradient_clip_norm_type=config.gradient_clip_norm_type,
                gradient_clip_error_if_nonfinite=(
                    config.gradient_clip_error_if_nonfinite
                ),
                gradient_clip_foreach=config.gradient_clip_foreach,
            )


def _train_effect(
    model: _EffectNet,
    *,
    texts: tuple[str, ...],
    positions: np.ndarray,
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    pseudo_outcome: np.ndarray,
    objective: str,
    config: RoleNeutralHTRConfig,
    seed: int,
    device: torch.device,
) -> None:
    optimizer = _adamw_optimizer(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        config=config,
    )
    for epoch in range(config.effect_epochs):
        model.train()
        for batch_pos in _batch_positions(
            positions,
            batch_size=config.batch_size,
            seed=seed,
            epoch=epoch,
        ):
            batch_texts = [texts[int(position)] for position in batch_pos]
            optimizer.zero_grad(
                set_to_none=config.optimizer_zero_grad_set_to_none
            )
            tau = model(batch_texts)
            if objective == "pseudo_outcome_mse":
                target = torch.as_tensor(
                    pseudo_outcome[batch_pos],
                    dtype=torch.float32,
                    device=device,
                )
                loss = F.mse_loss(tau, target)
            elif objective == "squared_r_loss":
                y_resid = torch.as_tensor(
                    y_residual[batch_pos],
                    dtype=torch.float32,
                    device=device,
                )
                t_resid = torch.as_tensor(
                    t_residual[batch_pos],
                    dtype=torch.float32,
                    device=device,
                )
                loss = torch.mean(torch.square(y_resid - tau * t_resid))
            else:
                raise ValueError("unsupported HTR effect objective")
            if not torch.isfinite(loss):
                raise RuntimeError("HTR effect optimization emitted non-finite loss")
            loss.backward()
            _clip_and_step(
                model,
                optimizer,
                gradient_clip_norm=config.gradient_clip_norm,
                gradient_clip_norm_type=config.gradient_clip_norm_type,
                gradient_clip_error_if_nonfinite=(
                    config.gradient_clip_error_if_nonfinite
                ),
                gradient_clip_foreach=config.gradient_clip_foreach,
            )


def _complete_attention_evidence(
    extractor: HierarchicalTransformerExtractor,
    *,
    texts: Sequence[str],
    coverage: _CoveragePlan,
    row_positions: Sequence[int],
    fold: int,
    stage: str,
    objective: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    values = tuple(str(text) for text in texts)
    positions = tuple(int(position) for position in row_positions)
    if len(values) != len(positions):
        raise ValueError("HTR attention rows and texts differ")
    evidence: list[dict[str, Any]] = []
    for start in range(0, len(values), int(batch_size)):
        batch_texts = list(values[start : start + int(batch_size)])
        batch_positions = positions[start : start + int(batch_size)]
        interpretations = extractor.interpret_attention(
            batch_texts,
            top_k=coverage.summary["configured_max_chunks"],
            role=stage,
        )
        if len(interpretations) != len(batch_texts):
            raise RuntimeError("HTR attention interpretation omitted a note")
        for local_index, (fit_position, interpretation) in enumerate(
            zip(batch_positions, interpretations, strict=True)
        ):
            expected_chunks = coverage.chunks_by_note[fit_position]
            observed_chunks = tuple(interpretation.get("chunks") or ())
            if observed_chunks != expected_chunks:
                raise RuntimeError("HTR attention changed the complete chunk plan")
            top = interpretation.get("top_chunks")
            if not isinstance(top, list) or len(top) != len(expected_chunks):
                raise RuntimeError("HTR evidence omitted one or more configured chunks")
            by_index = {
                int(item["chunk_index"]): item
                for item in top
                if isinstance(item, Mapping)
            }
            if set(by_index) != set(range(len(expected_chunks))):
                raise RuntimeError("HTR attention chunk coverage is not exact")
            for chunk_index, chunk_text in enumerate(expected_chunks):
                item = by_index[chunk_index]
                if item.get("chunk") != chunk_text:
                    raise RuntimeError("HTR evidence chunk text changed")
                evidence.append(
                    {
                        "witness_kind": "complete_htr_chunk_attention",
                        "stage": stage,
                        "objective": objective,
                        "fold": int(fold),
                        "fit_note_position": int(fit_position),
                        "chunk_index": int(chunk_index),
                        "chunk_text": chunk_text,
                        "chunk_sha256": hashlib.sha256(
                            chunk_text.encode("utf-8")
                        ).hexdigest(),
                        "attention": float(item["attention"]),
                    }
                )
    return evidence


def _validate_complete_attention_evidence(
    *,
    payload: Mapping[str, Any],
    coverage: Mapping[str, np.ndarray],
    nuisance_records: Sequence[Mapping[str, Any]],
    effect_records: Sequence[Mapping[str, Any]],
    config: RoleNeutralHTRConfig,
) -> None:
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {"schema_version", "family", "architecture_evidence"}
        or payload.get("schema_version")
        != NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION
        or payload.get("family") != HTR_NEURAL
        or not isinstance(payload.get("architecture_evidence"), list)
    ):
        raise ValueError("HTR native evidence envelope changed")
    note_chunks = np.asarray(coverage["note_chunk_counts"], dtype=np.int64)
    chunk_notes = np.asarray(coverage["chunk_note_positions"], dtype=np.int64)
    digests = np.asarray(coverage["chunk_sha256_bytes"], dtype=np.uint8)
    digest_by_note_chunk: dict[tuple[int, int], str] = {}
    local_indexes = np.zeros(len(note_chunks), dtype=np.int64)
    for flat_index, note_position in enumerate(chunk_notes):
        note = int(note_position)
        chunk_index = int(local_indexes[note])
        local_indexes[note] += 1
        digest_by_note_chunk[(note, chunk_index)] = bytes(
            digests[flat_index].tolist()
        ).hex()
    expected: set[tuple[str, str, int, int, int]] = set()
    for record in nuisance_records:
        fold = int(record["fold"])
        for note_position in record["validation_positions"]:
            for chunk_index in range(note_chunks[int(note_position)]):
                expected.add(
                    (
                        "nuisance",
                        "joint_treatment_outcome_nuisance",
                        fold,
                        int(note_position),
                        int(chunk_index),
                    )
                )
    for record in effect_records:
        objective = str(record["effect_objective"])
        fold = int(record["fold"])
        for note_position in record["validation_positions"]:
            for chunk_index in range(note_chunks[int(note_position)]):
                expected.add(
                    (
                        "effect_modifier",
                        objective,
                        fold,
                        int(note_position),
                        int(chunk_index),
                    )
                )
    observed: set[tuple[str, str, int, int, int]] = set()
    expected_atom_keys = {
        "witness_kind",
        "stage",
        "objective",
        "fold",
        "fit_note_position",
        "chunk_index",
        "chunk_text",
        "chunk_sha256",
        "attention",
    }
    for atom in payload["architecture_evidence"]:
        if not isinstance(atom, Mapping) or set(atom) != expected_atom_keys:
            raise ValueError("HTR attention evidence atom schema changed")
        key = (
            str(atom.get("stage")),
            str(atom.get("objective")),
            int(atom.get("fold", 0)),
            int(atom.get("fit_note_position", -1)),
            int(atom.get("chunk_index", -1)),
        )
        chunk_text = atom.get("chunk_text")
        attention = atom.get("attention")
        expected_digest = digest_by_note_chunk.get((key[3], key[4]))
        if (
            atom.get("witness_kind") != "complete_htr_chunk_attention"
            or key in observed
            or key not in expected
            or not isinstance(chunk_text, str)
            or expected_digest is None
            or atom.get("chunk_sha256")
            != hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            or atom.get("chunk_sha256") != expected_digest
            or isinstance(attention, bool)
            or not isinstance(attention, (int, float))
            or not np.isfinite(float(attention))
        ):
            raise ValueError("HTR attention evidence changed or lost coverage")
        observed.add(key)
    if observed != expected:
        raise ValueError("HTR attention evidence is not complete across fit chunks")
    expected_total = int(np.sum(note_chunks)) * (
        1 + len(config.effect_objectives)
    )
    if len(observed) != expected_total:
        raise ValueError("HTR attention evidence count changed")


@dataclass(frozen=True)
class _FoldTextAuthority:
    """One immutable complete-text authority shared by every owner fold."""

    texts: tuple[str, ...] | None
    row_ids: tuple[int, ...] | None
    coverage: _CoveragePlan | None
    reusable_plan: _ReusableTextPlan | None
    materialized_plan: _MaterializedReusableTextPlan | None

    @classmethod
    def in_memory(
        cls,
        *,
        texts: Sequence[str],
        row_ids: Sequence[int],
        coverage: _CoveragePlan,
        reusable_plan: _ReusableTextPlan | None,
    ) -> "_FoldTextAuthority":
        return cls(
            texts=tuple(texts),
            row_ids=tuple(int(value) for value in row_ids),
            coverage=coverage,
            reusable_plan=reusable_plan,
            materialized_plan=None,
        )

    @classmethod
    def materialized(
        cls,
        descriptor: _MaterializedReusableTextPlan,
    ) -> "_FoldTextAuthority":
        return cls(
            texts=None,
            row_ids=None,
            coverage=None,
            reusable_plan=None,
            materialized_plan=descriptor,
        )


_PROCESS_TEXT_PLAN_CACHE: dict[
    str,
    tuple[tuple[str, ...], tuple[int, ...], _CoveragePlan, _ReusableTextPlan],
] = {}


def _resolve_fold_text_authority(
    authority: _FoldTextAuthority,
) -> tuple[tuple[str, ...], tuple[int, ...], _CoveragePlan, _ReusableTextPlan | None]:
    if not isinstance(authority, _FoldTextAuthority):
        raise TypeError("HTR fold worker requires one typed text authority")
    if authority.materialized_plan is not None:
        if any(
            value is not None
            for value in (
                authority.texts,
                authority.row_ids,
                authority.coverage,
                authority.reusable_plan,
            )
        ):
            raise ValueError("materialized HTR text authority leaked in-memory rows")
        key = authority.materialized_plan.content_sha256
        resolved = _PROCESS_TEXT_PLAN_CACHE.get(key)
        if resolved is None:
            resolved = _load_materialized_reusable_text_plan(
                authority.materialized_plan
            )
            _PROCESS_TEXT_PLAN_CACHE[key] = resolved
        return resolved
    if (
        authority.texts is None
        or authority.row_ids is None
        or authority.coverage is None
    ):
        raise ValueError("in-memory HTR text authority is incomplete")
    if len(authority.texts) != len(authority.row_ids):
        raise ValueError("HTR text authority row order changed")
    if authority.reusable_plan is not None and (
        authority.reusable_plan.ordered_row_ids != authority.row_ids
        or authority.reusable_plan.text_sha256
        != _text_sha256(authority.row_ids, authority.texts)
    ):
        raise ValueError("HTR reusable plan differs from its in-memory authority")
    return (
        authority.texts,
        authority.row_ids,
        authority.coverage,
        authority.reusable_plan,
    )


@dataclass(frozen=True)
class _NuisanceFoldTask:
    fold: int
    split_seed: int
    model_seed: int
    fit_positions: np.ndarray
    validation_positions: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    config: RoleNeutralHTRConfig
    model_marker: str
    operational_controls: RoleNeutralHTROperationalControls | None
    text_authority: _FoldTextAuthority
    preflight_complete_text: bool


@dataclass(frozen=True)
class _EffectFoldTask:
    objective: str
    fold: int
    split_seed: int
    model_seed: int
    fit_positions: np.ndarray
    eligible_fit_positions: np.ndarray
    validation_positions: np.ndarray
    y_residual: np.ndarray
    t_residual: np.ndarray
    pseudo_outcome: np.ndarray
    config: RoleNeutralHTRConfig
    model_marker: str
    operational_controls: RoleNeutralHTROperationalControls | None
    text_authority: _FoldTextAuthority


@dataclass(frozen=True)
class _NuisanceFoldResult:
    fold: int
    split_seed: int
    model_seed: int
    fit_positions: np.ndarray
    validation_positions: np.ndarray
    model: Mapping[str, Any]
    propensity_calibrator: Mapping[str, Any]
    outcome_calibrator: Mapping[str, Any]
    validation_e_hat: np.ndarray
    validation_m_hat: np.ndarray
    architecture_evidence: tuple[Mapping[str, Any], ...]
    extractor_attestation: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    gpu_peak_allocated_bytes: int | None


@dataclass(frozen=True)
class _EffectFoldResult:
    objective: str
    fold: int
    split_seed: int
    model_seed: int
    fit_positions: np.ndarray
    eligible_fit_positions: np.ndarray
    validation_positions: np.ndarray
    model: Mapping[str, Any]
    validation_tau: np.ndarray
    architecture_evidence: tuple[Mapping[str, Any], ...]
    extractor_attestation: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    gpu_peak_allocated_bytes: int | None


@dataclass(frozen=True)
class _CompletedFoldWork:
    value: Any
    device: str
    started_monotonic_ns: int
    finished_monotonic_ns: int
    process_id: int
    thread_id: int
    gpu_peak_allocated_bytes: int | None
    torch_determinism_observed: Mapping[str, Any] | None


def _prepare_fold_extractor(
    *,
    config: RoleNeutralHTRConfig,
    model_marker: str,
    device: torch.device,
    texts: tuple[str, ...],
    row_ids: tuple[int, ...],
    fit_positions: np.ndarray,
    coverage: _CoveragePlan,
    reusable_plan: _ReusableTextPlan | None,
    operational_controls: RoleNeutralHTROperationalControls | None,
    preflight_complete_text: bool,
) -> tuple[HierarchicalTransformerExtractor, Mapping[str, Any]]:
    extractor = _new_extractor(
        config=config,
        model_marker=model_marker,
        device=device,
    )
    if reusable_plan is not None:
        if operational_controls is None:
            raise ValueError("HTR reusable fold plan lacks operational controls")
        extractor.fit_tokenizer([])
        _install_reusable_text_plan(
            extractor=extractor,
            plan=reusable_plan,
            texts=texts,
            row_ids=row_ids,
            config=config,
            controls=operational_controls,
        )
    else:
        extractor.fit_tokenizer(
            [texts[int(position)] for position in fit_positions]
        )
        if preflight_complete_text:
            _preflight_token_lengths(
                extractor,
                texts,
                batch_size=config.prediction_batch_size,
            )
    attestation = _attest_extractor(extractor, config=config)
    if operational_controls is not None:
        _set_operational_encoder_batch_size(
            extractor,
            config=config,
            controls=operational_controls,
        )
    return extractor, attestation


def _begin_fold_gpu_telemetry(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)


def _finish_fold_gpu_telemetry(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def _run_nuisance_fold(
    task: _NuisanceFoldTask,
    device_name: str,
) -> _NuisanceFoldResult:
    if not isinstance(task, _NuisanceFoldTask):
        raise TypeError("HTR nuisance worker received another task type")
    device = torch.device(device_name)
    _begin_fold_gpu_telemetry(device)
    texts, row_ids, coverage, reusable_plan = _resolve_fold_text_authority(
        task.text_authority
    )
    fit_positions = np.asarray(task.fit_positions, dtype=np.int64)
    validation_positions = np.asarray(
        task.validation_positions,
        dtype=np.int64,
    )
    _set_model_seed(task.model_seed, device)
    extractor, attestation = _prepare_fold_extractor(
        config=task.config,
        model_marker=task.model_marker,
        device=device,
        texts=texts,
        row_ids=row_ids,
        fit_positions=fit_positions,
        coverage=coverage,
        reusable_plan=reusable_plan,
        operational_controls=task.operational_controls,
        preflight_complete_text=task.preflight_complete_text,
    )
    model = _NuisanceNet(
        extractor=extractor,
        hidden_dim=task.config.hidden_dim,
        outcome_type=task.config.outcome_type,
        head_depth=task.config.nuisance_head_depth,
        head_activation=task.config.nuisance_head_activation,
        head_dropout=task.config.nuisance_head_dropout,
        head_layer_norm=task.config.nuisance_head_layer_norm,
        head_bias=task.config.nuisance_head_bias,
    ).to(device)
    try:
        _train_nuisance(
            model,
            texts=texts,
            treatment=task.treatment,
            outcome=task.outcome,
            positions=fit_positions,
            config=task.config,
            seed=task.model_seed,
            device=device,
        )
        fit_raw_e, fit_raw_m = _predict_model(
            model,
            [texts[int(position)] for position in fit_positions],
            kind="nuisance",
            outcome_type=task.config.outcome_type,
            batch_size=task.config.prediction_batch_size,
        )
        validation_raw_e, validation_raw_m = _predict_model(
            model,
            [texts[int(position)] for position in validation_positions],
            kind="nuisance",
            outcome_type=task.config.outcome_type,
            batch_size=task.config.prediction_batch_size,
        )
        propensity_calibrator = BinaryProbabilityCalibrator.fit(
            fit_raw_e,
            task.treatment[fit_positions],
            method=task.config.nuisance_calibration,
        )
        outcome_calibrator = BinaryProbabilityCalibrator.fit(
            fit_raw_m,
            task.outcome[fit_positions],
            method=task.config.nuisance_calibration,
        )
        validation_e = np.asarray(
            propensity_calibrator.transform(validation_raw_e),
            dtype=np.float64,
        )
        validation_m = np.asarray(
            outcome_calibrator.transform(validation_raw_m),
            dtype=np.float64,
        )
        evidence = tuple(
            _complete_attention_evidence(
                model.extractor,
                texts=[
                    texts[int(position)]
                    for position in validation_positions
                ],
                coverage=coverage,
                row_positions=validation_positions,
                fold=task.fold,
                stage="nuisance",
                objective="joint_treatment_outcome_nuisance",
                batch_size=task.config.prediction_batch_size,
            )
        )
        if task.operational_controls is not None:
            _restore_scientific_encoder_batch_size(
                model.extractor,
                config=task.config,
                controls=task.operational_controls,
            )
        local_store = _SafeArrayStore()
        prefix = f"nuisance_{task.fold:04d}"
        model_descriptor = _capture_model_state(
            model,
            local_store,
            prefix,
            kind="nuisance",
            outcome_type=task.config.outcome_type,
            training_configuration=_training_configuration(
                task.config,
                kind="nuisance",
            ),
        )
        propensity_descriptor = _capture_calibrator(
            propensity_calibrator,
            local_store,
            f"{prefix}_propensity",
        )
        outcome_descriptor = _capture_calibrator(
            outcome_calibrator,
            local_store,
            f"{prefix}_outcome",
        )
        peak = _finish_fold_gpu_telemetry(device)
        return _NuisanceFoldResult(
            fold=task.fold,
            split_seed=task.split_seed,
            model_seed=task.model_seed,
            fit_positions=fit_positions,
            validation_positions=validation_positions,
            model=model_descriptor,
            propensity_calibrator=propensity_descriptor,
            outcome_calibrator=outcome_descriptor,
            validation_e_hat=validation_e,
            validation_m_hat=validation_m,
            architecture_evidence=evidence,
            extractor_attestation=attestation,
            arrays={
                key: np.ascontiguousarray(value)
                for key, value in local_store.arrays.items()
            },
            gpu_peak_allocated_bytes=peak,
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _run_effect_fold(
    task: _EffectFoldTask,
    device_name: str,
) -> _EffectFoldResult:
    if not isinstance(task, _EffectFoldTask):
        raise TypeError("HTR effect worker received another task type")
    device = torch.device(device_name)
    _begin_fold_gpu_telemetry(device)
    texts, row_ids, coverage, reusable_plan = _resolve_fold_text_authority(
        task.text_authority
    )
    fit_positions = np.asarray(task.fit_positions, dtype=np.int64)
    eligible_fit_positions = np.asarray(
        task.eligible_fit_positions,
        dtype=np.int64,
    )
    validation_positions = np.asarray(
        task.validation_positions,
        dtype=np.int64,
    )
    _set_model_seed(task.model_seed, device)
    extractor, attestation = _prepare_fold_extractor(
        config=task.config,
        model_marker=task.model_marker,
        device=device,
        texts=texts,
        row_ids=row_ids,
        fit_positions=eligible_fit_positions,
        coverage=coverage,
        reusable_plan=reusable_plan,
        operational_controls=task.operational_controls,
        preflight_complete_text=False,
    )
    model = _EffectNet(
        extractor=extractor,
        hidden_dim=task.config.hidden_dim,
        head_depth=task.config.effect_head_depth,
        head_activation=task.config.effect_head_activation,
        head_dropout=task.config.effect_head_dropout,
        head_layer_norm=task.config.effect_head_layer_norm,
        head_bias=task.config.effect_head_bias,
    ).to(device)
    try:
        _train_effect(
            model,
            texts=texts,
            positions=eligible_fit_positions,
            y_residual=task.y_residual,
            t_residual=task.t_residual,
            pseudo_outcome=task.pseudo_outcome,
            objective=task.objective,
            config=task.config,
            seed=task.model_seed,
            device=device,
        )
        [validation_tau_raw] = _predict_model(
            model,
            [texts[int(position)] for position in validation_positions],
            kind="effect",
            outcome_type=task.config.outcome_type,
            batch_size=task.config.prediction_batch_size,
        )
        validation_tau = np.asarray(validation_tau_raw, dtype=np.float64)
        evidence = tuple(
            _complete_attention_evidence(
                model.extractor,
                texts=[
                    texts[int(position)]
                    for position in validation_positions
                ],
                coverage=coverage,
                row_positions=validation_positions,
                fold=task.fold,
                stage="effect_modifier",
                objective=task.objective,
                batch_size=task.config.prediction_batch_size,
            )
        )
        if task.operational_controls is not None:
            _restore_scientific_encoder_batch_size(
                model.extractor,
                config=task.config,
                controls=task.operational_controls,
            )
        local_store = _SafeArrayStore()
        prefix = f"effect_{task.objective}_{task.fold:04d}"
        model_descriptor = _capture_model_state(
            model,
            local_store,
            prefix,
            kind="effect",
            outcome_type=task.config.outcome_type,
            training_configuration=_training_configuration(
                task.config,
                kind="effect",
            ),
        )
        peak = _finish_fold_gpu_telemetry(device)
        return _EffectFoldResult(
            objective=task.objective,
            fold=task.fold,
            split_seed=task.split_seed,
            model_seed=task.model_seed,
            fit_positions=fit_positions,
            eligible_fit_positions=eligible_fit_positions,
            validation_positions=validation_positions,
            model=model_descriptor,
            validation_tau=validation_tau,
            architecture_evidence=evidence,
            extractor_attestation=attestation,
            arrays={
                key: np.ascontiguousarray(value)
                for key, value in local_store.arrays.items()
            },
            gpu_peak_allocated_bytes=peak,
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _invoke_htr_fold_worker(
    worker: Callable[[Any, str], Any],
    task: Any,
    device: str,
    *,
    worker_cpu_threads: int,
    process_isolated: bool,
) -> _CompletedFoldWork:
    # The lease is active as soon as the child begins the submitted task.
    # Include process-local determinism and native-thread setup in its
    # interval; excluding that preamble can falsely report serial execution
    # even while multiple fold workers are occupied concurrently.
    started = time.monotonic_ns()
    determinism_before: Mapping[str, Any] | None = None
    if process_isolated:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Spawned fold processes do not inherit Torch's process-global
        # determinism switches. Re-establish the one Stage 1 policy before
        # any worker CUDA/model initialization and fail if training weakens it.
        determinism_before = _validate_torch_determinism_observation(
            _enforce_stage1_torch_determinism()
        )
        torch.set_num_threads(worker_cpu_threads)
        if (
            torch.get_num_threads() != worker_cpu_threads
            or torch.get_num_interop_threads() != worker_cpu_threads
        ):
            raise RuntimeError(
                "spawned HTR fold worker did not preserve its one-thread "
                "Torch CPU lease"
            )
    with threadpool_limits(limits=worker_cpu_threads):
        value = worker(task, device)
    finished = time.monotonic_ns()
    if finished <= started:
        raise RuntimeError("HTR fold interval clock did not advance")
    peak = getattr(value, "gpu_peak_allocated_bytes", None)
    if peak is not None and (
        isinstance(peak, bool) or not isinstance(peak, int) or peak < 0
    ):
        raise ValueError("HTR fold worker returned invalid GPU peak telemetry")
    determinism_after: Mapping[str, Any] | None = None
    if process_isolated:
        determinism_after = _validate_torch_determinism_observation(
            _observe_stage1_torch_determinism()
        )
        if dict(determinism_after) != dict(determinism_before or {}):
            raise RuntimeError(
                "HTR fold worker weakened strict Stage 1 Torch determinism"
            )
    return _CompletedFoldWork(
        value=value,
        device=str(device),
        started_monotonic_ns=started,
        finished_monotonic_ns=finished,
        process_id=os.getpid(),
        thread_id=threading.get_ident(),
        gpu_peak_allocated_bytes=peak,
        torch_determinism_observed=determinism_after,
    )


def _warm_htr_process_slot(worker_cpu_threads: int) -> tuple[int, int, int]:
    """Configure and prove one spawned slot without initializing CUDA."""

    threads = int(worker_cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    return (
        os.getpid(),
        int(torch.get_num_threads()),
        int(torch.get_num_interop_threads()),
    )


class _HTRFoldExecutor:
    """One stable single-worker executor per configured fold lease."""

    def __init__(self, resource_plan: RoleNeutralHTRFoldResourcePlan) -> None:
        if not isinstance(resource_plan, RoleNeutralHTRFoldResourcePlan):
            raise TypeError("HTR fold executor requires a typed resource plan")
        self.resource_plan = resource_plan
        self._executors: tuple[concurrent.futures.Executor, ...] = ()

    def __enter__(self) -> "_HTRFoldExecutor":
        if self._executors:
            raise RuntimeError("HTR fold executor cannot be entered twice")
        if self.resource_plan.fold_parallel_backend == "processes":
            context = mp.get_context("spawn")
            self._executors = tuple(
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=context,
                )
                for _slot in range(self.resource_plan.fold_parallelism)
            )
            # ProcessPoolExecutor starts workers lazily.  If the first tiny
            # fold is submitted while later slots are still importing this
            # module, genuine parallel capacity can look serial in the lease
            # attestation.  Ready every stable slot before any fold receives
            # a lease; the warm-up itself does not initialize CUDA or carry
            # scientific inputs.
            try:
                warm_futures = tuple(
                    executor.submit(
                        _warm_htr_process_slot,
                        self.resource_plan.worker_cpu_threads,
                    )
                    for executor in self._executors
                )
                worker_reports = tuple(
                    future.result() for future in warm_futures
                )
                worker_pids = tuple(
                    int(report[0]) for report in worker_reports
                )
                if len(set(worker_pids)) != len(worker_pids):
                    raise RuntimeError(
                        "HTR process fold slots did not receive isolated "
                        "workers"
                    )
                if any(
                    report[1:] != (
                        self.resource_plan.worker_cpu_threads,
                        self.resource_plan.worker_cpu_threads,
                    )
                    for report in worker_reports
                ):
                    raise RuntimeError(
                        "HTR process fold slot failed to bind its Torch CPU "
                        "thread lease"
                    )
            except BaseException:
                executors = self._executors
                self._executors = ()
                for executor in executors:
                    executor.shutdown(wait=True, cancel_futures=True)
                raise
        else:
            self._executors = tuple(
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=f"oci-htr-fold-{slot:02d}",
                )
                for slot in range(self.resource_plan.fold_parallelism)
            )
        return self

    def submit(
        self,
        *,
        slot: int,
        worker: Callable[[Any, str], Any],
        task: Any,
        device: str,
    ) -> concurrent.futures.Future[_CompletedFoldWork]:
        if not self._executors:
            raise RuntimeError("HTR fold executor is not active")
        index = int(slot)
        if index < 0 or index >= len(self._executors):
            raise ValueError("HTR fold executor received an invalid lease slot")
        return self._executors[index].submit(
            _invoke_htr_fold_worker,
            worker,
            task,
            device,
            worker_cpu_threads=self.resource_plan.worker_cpu_threads,
            process_isolated=(
                self.resource_plan.fold_parallel_backend == "processes"
            ),
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        executors = self._executors
        self._executors = ()
        for executor in executors:
            executor.shutdown(
                wait=True,
                cancel_futures=exc_type is not None,
            )


def _fold_task_identity(task: Any) -> tuple[str, int]:
    if isinstance(task, _NuisanceFoldTask):
        return "joint_treatment_outcome_nuisance", int(task.fold)
    if isinstance(task, _EffectFoldTask):
        return str(task.objective), int(task.fold)
    if isinstance(task, Mapping):
        return str(task.get("objective") or "test"), int(task.get("fold", 0))
    return str(getattr(task, "objective", "test")), int(
        getattr(task, "fold", 0)
    )


def _execute_htr_fold_tasks(
    tasks: Sequence[Any],
    *,
    resource_plan: RoleNeutralHTRFoldResourcePlan,
    worker: Callable[[Any, str], Any],
    stage: str,
    event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    executor: _HTRFoldExecutor | None = None,
) -> tuple[Any, ...]:
    """Execute a canonical task sequence under deterministic device leases."""

    rows = tuple(tasks)
    if not rows:
        return ()
    if not isinstance(resource_plan, RoleNeutralHTRFoldResourcePlan):
        raise TypeError("HTR fold tasks require a typed resource plan")
    if not callable(worker):
        raise TypeError("HTR fold worker must be callable")
    stage_name = str(stage)
    if stage_name not in {"nuisance", "effect"}:
        raise ValueError("HTR fold stage is unsupported")
    if event_sink is not None and not callable(event_sink):
        raise TypeError("HTR fold event sink must be callable")
    owned_executor = executor is None
    active = _HTRFoldExecutor(resource_plan) if owned_executor else executor
    if not isinstance(active, _HTRFoldExecutor):
        raise TypeError("HTR fold tasks require a typed executor")

    def run() -> tuple[Any, ...]:
        completed_by_index: dict[int, _CompletedFoldWork] = {}
        active_futures: dict[
            concurrent.futures.Future[_CompletedFoldWork],
            tuple[int, int],
        ] = {}
        next_index = 0

        def submit_to_slot(slot: int) -> None:
            nonlocal next_index
            if next_index >= len(rows):
                return
            task_index = next_index
            next_index += 1
            device = resource_plan.fold_devices[slot]
            future = active.submit(
                slot=slot,
                worker=worker,
                task=rows[task_index],
                device=device,
            )
            active_futures[future] = (task_index, slot)

        for slot in range(
            min(resource_plan.fold_parallelism, len(rows))
        ):
            submit_to_slot(slot)
        while active_futures:
            done, _pending = concurrent.futures.wait(
                tuple(active_futures),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            released_slots: list[int] = []
            for future in sorted(
                done,
                key=lambda item: active_futures[item][0],
            ):
                task_index, slot = active_futures.pop(future)
                completed_by_index[task_index] = future.result()
                released_slots.append(slot)
            for slot in sorted(released_slots):
                submit_to_slot(slot)
        if set(completed_by_index) != set(range(len(rows))):
            raise RuntimeError("HTR fold executor omitted a canonical task")
        completed = tuple(
            completed_by_index[index] for index in range(len(rows))
        )
        for task, record in zip(rows, completed, strict=True):
            objective, fold = _fold_task_identity(task)
            common = {
                "stage": stage_name,
                "objective": objective,
                "fold": fold,
                "device": record.device,
                "process_id": record.process_id,
                "thread_id": record.thread_id,
                "gpu_peak_allocated_bytes": record.gpu_peak_allocated_bytes,
                "torch_determinism_observed": (
                    None
                    if record.torch_determinism_observed is None
                    else copy.deepcopy(
                        dict(record.torch_determinism_observed)
                    )
                ),
                "resource_locator_in_scientific_identity": False,
            }
            if event_sink is not None:
                event_sink(
                    {
                        **common,
                        "event": "fold_started",
                        "monotonic_ns": record.started_monotonic_ns,
                    }
                )
                event_sink(
                    {
                        **common,
                        "event": "fold_finished",
                        "monotonic_ns": record.finished_monotonic_ns,
                    }
                )
        return tuple(record.value for record in completed)

    if owned_executor:
        with active:
            return run()
    return run()


def _merge_isolated_fold_arrays(
    *,
    target: _SafeArrayStore,
    arrays: Mapping[str, np.ndarray],
) -> None:
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("HTR fold returned no isolated proof arrays")
    for key in sorted(arrays):
        target.add(str(key), arrays[key])


@dataclass(frozen=True)
class _OwnerFoldFitResult:
    fit_reusable_plan: _ReusableTextPlan | None
    process_plan_attestation: Mapping[str, Any] | None
    nuisance_oof_e: np.ndarray
    nuisance_oof_m: np.ndarray
    clipped_e: np.ndarray
    y_residual: np.ndarray
    t_residual: np.ndarray
    pseudo_outcome: np.ndarray
    eligible: np.ndarray
    effect_oof: Mapping[str, np.ndarray]
    nuisance_records: tuple[Mapping[str, Any], ...]
    effect_records: tuple[Mapping[str, Any], ...]
    architecture_evidence: tuple[Mapping[str, Any], ...]
    fold_execution_events: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _EffectFoldTaskPlan:
    """Canonical R-stage quantities and tasks built after the nuisance barrier."""

    tasks: tuple[_EffectFoldTask, ...]
    clipped_e: np.ndarray
    y_residual: np.ndarray
    t_residual: np.ndarray
    pseudo_outcome: np.ndarray
    eligible: np.ndarray


def _build_effect_fold_tasks(
    *,
    owner_scope_seed: int,
    text_count: int,
    treatment: np.ndarray,
    outcome: np.ndarray,
    nuisance_oof_e: np.ndarray,
    nuisance_oof_m: np.ndarray,
    config: RoleNeutralHTRConfig,
    model_marker: str,
    operational_controls: RoleNeutralHTROperationalControls | None,
    text_authority: _FoldTextAuthority,
) -> _EffectFoldTaskPlan:
    """Build the exact production effect tasks from one complete nuisance OOF.

    Keeping this barrier transformation in one production helper lets bounded
    kernel calibration exercise the same splits, seeds, eligibility rule, and
    optimizer inputs without duplicating the scientific R-stage builder.
    """

    row_count = int(text_count)
    treatment_values = np.asarray(treatment, dtype=np.float64)
    outcome_values = np.asarray(outcome, dtype=np.float64)
    nuisance_e_values = np.asarray(nuisance_oof_e, dtype=np.float64)
    nuisance_m_values = np.asarray(nuisance_oof_m, dtype=np.float64)
    if (
        row_count < 1
        or any(
            value.shape != (row_count,) or not np.isfinite(value).all()
            for value in (
                treatment_values,
                outcome_values,
                nuisance_e_values,
                nuisance_m_values,
            )
        )
    ):
        raise ValueError(
            "HTR effect-task builder requires complete finite nuisance OOF rows"
        )
    clipped_e = np.clip(
        nuisance_e_values,
        config.e_clip,
        1.0 - config.e_clip,
    )
    y_residual = outcome_values - nuisance_m_values
    t_residual = treatment_values - clipped_e
    pseudo_outcome = y_residual / t_residual
    eligible = (
        (nuisance_e_values >= config.r_stage_min_propensity)
        & (nuisance_e_values <= config.r_stage_max_propensity)
        & np.isfinite(pseudo_outcome)
    )
    if not np.any(eligible):
        raise ValueError("configured HTR R-stage bounds retain no fit rows")

    tasks: list[_EffectFoldTask] = []
    for objective in config.effect_objectives:
        split_seed = _derived_seed(
            int(owner_scope_seed),
            purpose="split",
            objective=objective,
            fold=0,
        )
        splits = tuple(
            KFold(
                n_splits=config.effect_folds,
                shuffle=True,
                random_state=split_seed,
            ).split(np.arange(row_count))
        )
        for fold, (fit_pos_raw, validation_pos_raw) in enumerate(
            splits,
            start=1,
        ):
            fit_pos = np.asarray(fit_pos_raw, dtype=np.int64)
            validation_pos = np.asarray(
                validation_pos_raw,
                dtype=np.int64,
            )
            eligible_fit_pos = fit_pos[eligible[fit_pos]]
            if not len(eligible_fit_pos):
                raise ValueError(
                    f"configured HTR {objective} fold {fold} "
                    "has no eligible fit rows"
                )
            tasks.append(
                _EffectFoldTask(
                    objective=objective,
                    fold=fold,
                    split_seed=split_seed,
                    model_seed=_derived_seed(
                        int(owner_scope_seed),
                        purpose="fit",
                        objective=objective,
                        fold=fold,
                    ),
                    fit_positions=fit_pos,
                    eligible_fit_positions=eligible_fit_pos,
                    validation_positions=validation_pos,
                    y_residual=y_residual,
                    t_residual=t_residual,
                    pseudo_outcome=pseudo_outcome,
                    config=config,
                    model_marker=model_marker,
                    operational_controls=operational_controls,
                    text_authority=text_authority,
                )
            )
    return _EffectFoldTaskPlan(
        tasks=tuple(tasks),
        clipped_e=np.ascontiguousarray(clipped_e),
        y_residual=np.ascontiguousarray(y_residual),
        t_residual=np.ascontiguousarray(t_residual),
        pseudo_outcome=np.ascontiguousarray(pseudo_outcome),
        eligible=np.ascontiguousarray(eligible),
    )


def _fit_owner_htr_folds(
    *,
    owner: Stage1ScopeSpec,
    texts: tuple[str, ...],
    treatment: np.ndarray,
    outcome: np.ndarray,
    coverage: _CoveragePlan,
    config: RoleNeutralHTRConfig,
    model_marker: str,
    store: _SafeArrayStore,
    operational_controls: RoleNeutralHTROperationalControls | None,
    resource_plan: RoleNeutralHTRFoldResourcePlan,
    scratch_parent: Path,
    external_event_sink: Callable[[Mapping[str, Any]], None] | None,
) -> _OwnerFoldFitResult:
    """Fit both cross-fit stages and merge isolated results canonically."""

    fold_events: list[Mapping[str, Any]] = []

    def emit(value: Mapping[str, Any]) -> None:
        closed = json.loads(_canonical_json(dict(value)))
        fold_events.append(closed)
        if external_event_sink is not None:
            external_event_sink(copy.deepcopy(closed))

    fit_reusable_plan: _ReusableTextPlan | None = None
    process_plan_attestation: Mapping[str, Any] | None = None
    temporary_plan: tempfile.TemporaryDirectory[str] | None = None
    temporary_plan_path: Path | None = None
    try:
        if (
            operational_controls is not None
            and operational_controls.reuse_tokenizer_and_chunk_plans
        ):
            # Build the complete tokenizer/chunk plan once on CPU before any
            # fold process can initialize CUDA. All fold workers consume only
            # this authenticated plan; none rechunk or retokenize owner text.
            plan_extractor = _new_extractor(
                config=config,
                model_marker=model_marker,
                device=torch.device("cpu"),
            )
            try:
                plan_extractor.fit_tokenizer([])
                fit_reusable_plan = _build_reusable_text_plan(
                    extractor=plan_extractor,
                    texts=texts,
                    row_ids=owner.fit_row_ids,
                    coverage=coverage,
                    config=config,
                    controls=operational_controls,
                    phase="fit",
                )
            finally:
                del plan_extractor
            if resource_plan.fold_parallel_backend == "processes":
                temporary_plan = tempfile.TemporaryDirectory(
                    dir=scratch_parent,
                    prefix=f".htr-{owner.scope_id}-fold-plan-",
                )
                temporary_plan_path = Path(temporary_plan.name)
                descriptor = _materialize_reusable_text_plan(
                    root=temporary_plan_path / "plan",
                    plan=fit_reusable_plan,
                    coverage=coverage,
                    texts=texts,
                    row_ids=owner.fit_row_ids,
                )
                process_plan_attestation = descriptor.attestation()
                text_authority = _FoldTextAuthority.materialized(descriptor)
            else:
                text_authority = _FoldTextAuthority.in_memory(
                    texts=texts,
                    row_ids=owner.fit_row_ids,
                    coverage=coverage,
                    reusable_plan=fit_reusable_plan,
                )
        else:
            text_authority = _FoldTextAuthority.in_memory(
                texts=texts,
                row_ids=owner.fit_row_ids,
                coverage=coverage,
                reusable_plan=None,
            )

        nuisance_split_seed = _derived_seed(
            owner.scope_seed,
            purpose="split",
            objective="nuisance",
            fold=0,
        )
        nuisance_splits = tuple(
            KFold(
                n_splits=config.nuisance_folds,
                shuffle=True,
                random_state=nuisance_split_seed,
            ).split(np.arange(len(texts)))
        )
        nuisance_tasks = tuple(
            _NuisanceFoldTask(
                fold=fold,
                split_seed=nuisance_split_seed,
                model_seed=_derived_seed(
                    owner.scope_seed,
                    purpose="fit",
                    objective="nuisance",
                    fold=fold,
                ),
                fit_positions=np.asarray(fit_pos, dtype=np.int64),
                validation_positions=np.asarray(
                    validation_pos,
                    dtype=np.int64,
                ),
                treatment=treatment,
                outcome=outcome,
                config=config,
                model_marker=model_marker,
                operational_controls=operational_controls,
                text_authority=text_authority,
                preflight_complete_text=(
                    fit_reusable_plan is None and fold == 1
                ),
            )
            for fold, (fit_pos, validation_pos) in enumerate(
                nuisance_splits,
                start=1,
            )
        )
        nuisance_oof_e = np.full(len(texts), np.nan, dtype=np.float64)
        nuisance_oof_m = np.full(len(texts), np.nan, dtype=np.float64)
        nuisance_records: list[Mapping[str, Any]] = []
        effect_records: list[Mapping[str, Any]] = []
        evidence: list[Mapping[str, Any]] = []
        effect_oof: dict[str, np.ndarray] = {}

        with _HTRFoldExecutor(resource_plan) as fold_executor:
            nuisance_results = _execute_htr_fold_tasks(
                nuisance_tasks,
                resource_plan=resource_plan,
                worker=_run_nuisance_fold,
                stage="nuisance",
                event_sink=emit,
                executor=fold_executor,
            )
            for task, raw_result in zip(
                nuisance_tasks,
                nuisance_results,
                strict=True,
            ):
                if not isinstance(raw_result, _NuisanceFoldResult):
                    raise TypeError("HTR nuisance fold returned another result type")
                result = raw_result
                if (
                    result.fold != task.fold
                    or result.split_seed != task.split_seed
                    or result.model_seed != task.model_seed
                    or not np.array_equal(
                        result.fit_positions,
                        task.fit_positions,
                    )
                    or not np.array_equal(
                        result.validation_positions,
                        task.validation_positions,
                    )
                    or result.validation_e_hat.shape
                    != task.validation_positions.shape
                    or result.validation_m_hat.shape
                    != task.validation_positions.shape
                    or not np.isfinite(result.validation_e_hat).all()
                    or not np.isfinite(result.validation_m_hat).all()
                ):
                    raise RuntimeError(
                        "HTR nuisance fold result changed rows, seeds, or shape"
                    )
                _validate_typed_model_configuration(
                    result.model,
                    config=config,
                    kind="nuisance",
                )
                _merge_isolated_fold_arrays(
                    target=store,
                    arrays=result.arrays,
                )
                nuisance_oof_e[result.validation_positions] = (
                    result.validation_e_hat
                )
                nuisance_oof_m[result.validation_positions] = (
                    result.validation_m_hat
                )
                evidence.extend(result.architecture_evidence)
                prefix = f"nuisance_{result.fold:04d}"
                nuisance_records.append(
                    {
                        "fold": result.fold,
                        "split_seed": result.split_seed,
                        "model_seed": result.model_seed,
                        "fit_positions": result.fit_positions.tolist(),
                        "validation_positions": (
                            result.validation_positions.tolist()
                        ),
                        "fit_row_ids": [
                            owner.fit_row_ids[int(position)]
                            for position in result.fit_positions
                        ],
                        "validation_row_ids": [
                            owner.fit_row_ids[int(position)]
                            for position in result.validation_positions
                        ],
                        "model": copy.deepcopy(dict(result.model)),
                        "propensity_calibrator": copy.deepcopy(
                            dict(result.propensity_calibrator)
                        ),
                        "outcome_calibrator": copy.deepcopy(
                            dict(result.outcome_calibrator)
                        ),
                        "validation_e_hat": store.add(
                            f"{prefix}_validation_e_hat",
                            result.validation_e_hat,
                        ),
                        "validation_m_hat": store.add(
                            f"{prefix}_validation_m_hat",
                            result.validation_m_hat,
                        ),
                        "extractor_attestation": copy.deepcopy(
                            dict(result.extractor_attestation)
                        ),
                        "registered_heldout_text_accessed": False,
                        "registered_heldout_labels_accessed": False,
                    }
                )
            if not np.isfinite(nuisance_oof_e).all() or not np.isfinite(
                nuisance_oof_m
            ).all():
                raise RuntimeError("HTR nuisance cross-fit omitted a fit row")

            # This is the strict stage barrier. No effect task is constructed
            # or submitted until every nuisance result is merged into OOF
            # order and every derived R-stage quantity is complete.
            effect_plan = _build_effect_fold_tasks(
                owner_scope_seed=int(owner.scope_seed),
                text_count=len(texts),
                treatment=treatment,
                outcome=outcome,
                nuisance_oof_e=nuisance_oof_e,
                nuisance_oof_m=nuisance_oof_m,
                config=config,
                model_marker=model_marker,
                operational_controls=operational_controls,
                text_authority=text_authority,
            )
            clipped_e = effect_plan.clipped_e
            y_residual = effect_plan.y_residual
            t_residual = effect_plan.t_residual
            pseudo_outcome = effect_plan.pseudo_outcome
            eligible = effect_plan.eligible
            emit(
                {
                    "event": "nuisance_barrier_completed",
                    "stage": "nuisance_to_effect",
                    "objective": "all",
                    "fold": 0,
                    "device": None,
                    "process_id": os.getpid(),
                    "thread_id": threading.get_ident(),
                    "gpu_peak_allocated_bytes": None,
                    "monotonic_ns": time.monotonic_ns(),
                    "all_nuisance_oof_assembled": True,
                    "residual_quantities_computed": True,
                    "effect_task_submitted": False,
                    "resource_locator_in_scientific_identity": False,
                }
            )

            effect_tasks = effect_plan.tasks
            effect_results = _execute_htr_fold_tasks(
                effect_tasks,
                resource_plan=resource_plan,
                worker=_run_effect_fold,
                stage="effect",
                event_sink=emit,
                executor=fold_executor,
            )
            by_objective_oof = {
                objective: np.full(
                    len(texts),
                    np.nan,
                    dtype=np.float64,
                )
                for objective in config.effect_objectives
            }
            for task, raw_result in zip(
                effect_tasks,
                effect_results,
                strict=True,
            ):
                if not isinstance(raw_result, _EffectFoldResult):
                    raise TypeError("HTR effect fold returned another result type")
                result = raw_result
                if (
                    result.objective != task.objective
                    or result.fold != task.fold
                    or result.split_seed != task.split_seed
                    or result.model_seed != task.model_seed
                    or not np.array_equal(
                        result.fit_positions,
                        task.fit_positions,
                    )
                    or not np.array_equal(
                        result.eligible_fit_positions,
                        task.eligible_fit_positions,
                    )
                    or not np.array_equal(
                        result.validation_positions,
                        task.validation_positions,
                    )
                    or result.validation_tau.shape
                    != task.validation_positions.shape
                    or not np.isfinite(result.validation_tau).all()
                ):
                    raise RuntimeError(
                        "HTR effect fold result changed rows, seeds, or shape"
                    )
                _validate_typed_model_configuration(
                    result.model,
                    config=config,
                    kind="effect",
                )
                _merge_isolated_fold_arrays(
                    target=store,
                    arrays=result.arrays,
                )
                by_objective_oof[result.objective][
                    result.validation_positions
                ] = result.validation_tau
                evidence.extend(result.architecture_evidence)
                prefix = f"effect_{result.objective}_{result.fold:04d}"
                effect_records.append(
                    {
                        "effect_objective": result.objective,
                        "fold": result.fold,
                        "split_seed": result.split_seed,
                        "model_seed": result.model_seed,
                        "fit_positions": result.fit_positions.tolist(),
                        "eligible_fit_positions": (
                            result.eligible_fit_positions.tolist()
                        ),
                        "validation_positions": (
                            result.validation_positions.tolist()
                        ),
                        "fit_row_ids": [
                            owner.fit_row_ids[int(position)]
                            for position in result.fit_positions
                        ],
                        "eligible_fit_row_ids": [
                            owner.fit_row_ids[int(position)]
                            for position in result.eligible_fit_positions
                        ],
                        "validation_row_ids": [
                            owner.fit_row_ids[int(position)]
                            for position in result.validation_positions
                        ],
                        "model": copy.deepcopy(dict(result.model)),
                        "validation_tau": store.add(
                            f"{prefix}_validation_tau",
                            result.validation_tau,
                        ),
                        "extractor_attestation": copy.deepcopy(
                            dict(result.extractor_attestation)
                        ),
                        "registered_heldout_text_accessed": False,
                        "registered_heldout_labels_accessed": False,
                    }
                )
            for objective in config.effect_objectives:
                oof = by_objective_oof[objective]
                if not np.isfinite(oof).all():
                    raise RuntimeError(
                        f"HTR {objective} cross-fit omitted a fit row"
                    )
                effect_oof[objective] = oof
    finally:
        if temporary_plan is not None:
            temporary_plan.cleanup()
            if temporary_plan_path is not None and (
                temporary_plan_path.exists()
                or temporary_plan_path.is_symlink()
            ):
                raise RuntimeError(
                    "HTR temporary raw-text process plan survived fold fitting"
                )

    return _OwnerFoldFitResult(
        fit_reusable_plan=fit_reusable_plan,
        process_plan_attestation=process_plan_attestation,
        nuisance_oof_e=nuisance_oof_e,
        nuisance_oof_m=nuisance_oof_m,
        clipped_e=clipped_e,
        y_residual=y_residual,
        t_residual=t_residual,
        pseudo_outcome=pseudo_outcome,
        eligible=eligible,
        effect_oof=effect_oof,
        nuisance_records=tuple(nuisance_records),
        effect_records=tuple(effect_records),
        architecture_evidence=tuple(evidence),
        fold_execution_events=tuple(fold_events),
    )


def _fold_execution_summary(
    *,
    events: Sequence[Mapping[str, Any]],
    resource_plan: RoleNeutralHTRFoldResourcePlan,
    config: RoleNeutralHTRConfig,
) -> dict[str, Any]:
    """Validate the stage barrier and actual lease use from worker clocks."""

    rows = tuple(dict(value) for value in events)
    barriers = [
        row for row in rows if row.get("event") == "nuisance_barrier_completed"
    ]
    starts = [row for row in rows if row.get("event") == "fold_started"]
    finishes = [row for row in rows if row.get("event") == "fold_finished"]
    if len(barriers) != 1 or len(starts) != len(finishes):
        raise RuntimeError("HTR fold telemetry lacks a unique stage barrier")

    def key(row: Mapping[str, Any]) -> tuple[str, str, int]:
        return (
            str(row.get("stage")),
            str(row.get("objective")),
            int(row.get("fold", 0)),
        )

    starts_by_key = {key(row): row for row in starts}
    finishes_by_key = {key(row): row for row in finishes}
    expected_count = config.nuisance_folds + (
        config.effect_folds * len(config.effect_objectives)
    )
    if (
        len(starts_by_key) != expected_count
        or len(finishes_by_key) != expected_count
        or set(starts_by_key) != set(finishes_by_key)
    ):
        raise RuntimeError("HTR fold telemetry changed canonical fold coverage")
    intervals: list[dict[str, Any]] = []
    for fold_key in sorted(starts_by_key):
        start = starts_by_key[fold_key]
        finish = finishes_by_key[fold_key]
        if (
            start.get("device") != finish.get("device")
            or start.get("process_id") != finish.get("process_id")
            or int(finish["monotonic_ns"]) <= int(start["monotonic_ns"])
        ):
            raise RuntimeError("HTR fold telemetry changed one lease interval")
        intervals.append(
            {
                "stage": fold_key[0],
                "objective": fold_key[1],
                "fold": fold_key[2],
                "device": str(start["device"]),
                "process_id": int(start["process_id"]),
                "thread_id": int(start["thread_id"]),
                "started_monotonic_ns": int(start["monotonic_ns"]),
                "finished_monotonic_ns": int(finish["monotonic_ns"]),
                "gpu_peak_allocated_bytes": finish.get(
                    "gpu_peak_allocated_bytes"
                ),
                "torch_determinism_observed": finish.get(
                    "torch_determinism_observed"
                ),
            }
        )
    if resource_plan.fold_parallel_backend == "processes":
        for row in intervals:
            _validate_torch_determinism_observation(
                row["torch_determinism_observed"]
            )
    elif any(
        row["torch_determinism_observed"] is not None
        for row in intervals
    ):
        raise RuntimeError(
            "in-process HTR fold telemetry claimed child determinism enforcement"
        )
    nuisance_intervals = [
        row for row in intervals if row["stage"] == "nuisance"
    ]
    effect_intervals = [
        row for row in intervals if row["stage"] == "effect"
    ]
    barrier_ns = int(barriers[0]["monotonic_ns"])
    if (
        len(nuisance_intervals) != config.nuisance_folds
        or len(effect_intervals)
        != config.effect_folds * len(config.effect_objectives)
        or max(row["finished_monotonic_ns"] for row in nuisance_intervals)
        >= barrier_ns
        or min(row["started_monotonic_ns"] for row in effect_intervals)
        <= barrier_ns
    ):
        raise RuntimeError("HTR effect folds crossed the nuisance barrier")

    selected_devices = tuple(resource_plan.devices)
    nuisance_devices = {
        str(row["device"]) for row in nuisance_intervals
    }
    effect_devices = {str(row["device"]) for row in effect_intervals}
    if (
        nuisance_devices != set(selected_devices)
        or effect_devices != set(selected_devices)
    ):
        raise RuntimeError("HTR folds did not exercise every selected device")

    def maximum_overlap(values: Sequence[Mapping[str, Any]]) -> int:
        boundaries = [
            (int(row["started_monotonic_ns"]), 1)
            for row in values
        ] + [
            (int(row["finished_monotonic_ns"]), -1)
            for row in values
        ]
        active = 0
        maximum = 0
        # A finishing lease is released before another starts at the same
        # monotonic tick.
        for _timestamp, delta in sorted(
            boundaries,
            key=lambda value: (value[0], value[1]),
        ):
            active += delta
            if active < 0:
                raise RuntimeError("HTR lease telemetry released an idle slot")
            maximum = max(maximum, active)
        if active != 0:
            raise RuntimeError("HTR lease telemetry left one fold active")
        return maximum

    per_device: dict[str, dict[str, Any]] = {}
    for device in selected_devices:
        device_rows = [
            row for row in intervals if row["device"] == device
        ]
        peaks = [
            int(row["gpu_peak_allocated_bytes"])
            for row in device_rows
            if row["gpu_peak_allocated_bytes"] is not None
        ]
        maximum = maximum_overlap(device_rows)
        if maximum > resource_plan.fold_slots_per_device:
            raise RuntimeError("HTR exceeded configured per-device fold slots")
        per_device[device] = {
            "task_count": len(device_rows),
            "maximum_concurrent_leases": maximum,
            "maximum_child_peak_allocated_bytes": (
                max(peaks) if peaks else None
            ),
            "conservative_sum_of_overlapping_child_peaks_bytes": (
                max(peaks) * maximum if peaks else None
            ),
        }
    overall_maximum = maximum_overlap(intervals)
    if overall_maximum > resource_plan.fold_parallelism:
        raise RuntimeError("HTR exceeded configured total fold concurrency")
    if (
        resource_plan.fold_parallelism > 1
        and min(config.nuisance_folds, len(nuisance_intervals)) > 1
        and maximum_overlap(nuisance_intervals) < 2
    ):
        raise RuntimeError("configured HTR nuisance folds did not overlap")
    if (
        resource_plan.fold_parallelism > 1
        and len(effect_intervals) > 1
        and maximum_overlap(effect_intervals) < 2
    ):
        raise RuntimeError("configured HTR effect folds did not overlap")
    if (
        resource_plan.fold_parallel_backend == "processes"
        and resource_plan.fold_parallelism > 1
        and len({row["process_id"] for row in intervals}) < 2
    ):
        raise RuntimeError("parallel HTR folds were not process isolated")
    return {
        "resource_plan": resource_plan.as_dict(),
        "fold_intervals": intervals,
        "nuisance_barrier_monotonic_ns": barrier_ns,
        "nuisance_barrier_enforced": True,
        "effect_submitted_only_after_nuisance_oof_and_residuals": True,
        "per_device": per_device,
        "maximum_concurrent_fold_leases": overall_maximum,
        "configured_total_fold_concurrency_respected": True,
        "configured_per_device_slots_respected": True,
        "every_selected_device_used_by_each_stage": True,
        "nested_native_worker_threads": resource_plan.worker_cpu_threads,
        "process_isolated_rng": (
            resource_plan.fold_parallel_backend == "processes"
        ),
        "process_isolated_torch_determinism_enforced_and_observed": (
            resource_plan.fold_parallel_backend == "processes"
        ),
        "multi_gpu_acceleration_claimed": False,
        "throughput_speedup_claimed": False,
        "memory_acceptance_thresholds_require_real_gpu_smoke": True,
        "resource_locators_in_scientific_identity": False,
    }


def _producer_identity() -> str:
    sources = [
        inspect.getsource(RoleNeutralHTRConfig),
        inspect.getsource(RoleNeutralHTROperationalControls),
        inspect.getsource(_coverage_plan),
        inspect.getsource(_build_reusable_text_plan),
        inspect.getsource(_install_reusable_text_plan),
        inspect.getsource(_materialize_reusable_text_plan),
        inspect.getsource(_load_materialized_reusable_text_plan),
        inspect.getsource(_train_nuisance),
        inspect.getsource(_train_effect),
        inspect.getsource(_run_nuisance_fold),
        inspect.getsource(_run_effect_fold),
        inspect.getsource(_invoke_htr_fold_worker),
        inspect.getsource(_execute_htr_fold_tasks),
        inspect.getsource(_fit_owner_htr_folds),
        inspect.getsource(_complete_attention_evidence),
        inspect.getsource(execute_role_neutral_htr_physical_group),
        inspect.getsource(HierarchicalTransformerExtractor),
        inspect.getsource(_NuisanceNet),
        inspect.getsource(_EffectNet),
        inspect.getsource(_capture_model_state),
        inspect.getsource(_build_model),
        inspect.getsource(_predict_model),
    ]
    return _sha256_json(
        {
            "schema_version": "production_role_neutral_htr_producer_identity_v1",
            "transitive_sources": sources,
        }
    )


def _fit_seal(
    *,
    request: RoleNeutralHTRPhysicalGroupRequest,
    evidence_payload: Mapping[str, Any],
    producer_identity_sha256: str,
    configuration_identity_sha256: str,
    fit_state_artifact_sha256: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(evidence_payload))
    if (
        set(payload)
        != {"schema_version", "family", "architecture_evidence"}
        or payload.get("schema_version")
        != NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION
        or payload.get("family") != HTR_NEURAL
        or not isinstance(payload.get("architecture_evidence"), list)
        or not payload["architecture_evidence"]
    ):
        raise ValueError("HTR fit-only seal requires nonempty native evidence")
    payload_sha256 = _sha256_json(payload)
    owner = request.physical_owner
    events = [
        {
            "sequence": 1,
            "event": "fit_completed",
            "fit_state_artifact_sha256": fit_state_artifact_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
        },
        {
            "sequence": 2,
            "event": "fit_family_artifact_sealed",
            "fit_state_artifact_sha256": fit_state_artifact_sha256,
            "evidence_payload_sha256": payload_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
        },
    ]
    body = {
        "schema_version": LEGACY_STAGE1_FIT_ONLY_FAMILY_SEAL_SCHEMA,
        "plan_scientific_content_sha256": (
            request.plan_scientific_content_sha256
        ),
        "physical_owner_scope_id": owner.scope_id,
        "physical_owner_scope_sha256": owner.as_dict()["scope_sha256"],
        "family": HTR_NEURAL,
        "fit_row_ids": list(owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(owner.fit_row_ids),
        "canonical_group_seed": int(owner.scope_seed),
        "producer_identity_sha256": _require_sha256(
            producer_identity_sha256,
            label="HTR producer identity",
        ),
        "configuration_identity_sha256": _require_sha256(
            configuration_identity_sha256,
            label="HTR configuration identity",
        ),
        "fit_state_artifact_sha256": _require_sha256(
            fit_state_artifact_sha256,
            label="HTR fit-state identity",
        ),
        "evidence_payload_sha256": payload_sha256,
        "evidence_payload": payload,
        "event_order": events,
        "logical_view_transform_started": False,
        "registered_heldout_text_accessed": False,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _write_array_store(
    *,
    fit_root: Path,
    store: _SafeArrayStore,
) -> dict[str, dict[str, Any]]:
    arrays_root = fit_root / "arrays"
    arrays_root.mkdir(parents=True, exist_ok=False)
    inventory: dict[str, dict[str, Any]] = {}
    for key in sorted(store.arrays):
        path = arrays_root / f"{key}.npy"
        _write_new_npy(path, store.arrays[key])
        file_sha256, size = _sha256_file(path, label=f"HTR array {key}")
        registered = store.inventory[key]
        inventory[key] = {
            "relative_path": path.relative_to(fit_root).as_posix(),
            "dtype": registered["dtype"],
            "shape": registered["shape"],
            "content_sha256": registered["content_sha256"],
            "file_sha256": file_sha256,
            "size_bytes": size,
        }
    if not inventory:
        raise RuntimeError("HTR fit produced no persisted numerical state")
    return inventory


def _load_array_store(
    fit_root: Path,
    inventory: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("HTR array inventory is missing")
    arrays: dict[str, np.ndarray] = {}
    expected_paths: set[str] = set()
    for raw_key in sorted(inventory):
        key = str(raw_key)
        row = inventory[key]
        if _SAFE_ARRAY_KEY.fullmatch(key) is None or not isinstance(row, Mapping):
            raise ValueError("HTR array inventory contains an invalid entry")
        expected_relative = f"arrays/{key}.npy"
        if row.get("relative_path") != expected_relative:
            raise ValueError("HTR array path is not canonical")
        expected_paths.add(expected_relative)
        path = fit_root / expected_relative
        digest, size, loaded = _read_npy_file_once(
            path,
            label=f"HTR array {key}",
            invalid_message=f"HTR array is not safe NumPy data: {key}",
        )
        if (
            digest != row.get("file_sha256")
            or size != int(row.get("size_bytes", -1))
        ):
            raise ValueError(f"HTR array file changed: {key}")
        if (
            loaded.dtype.str != row.get("dtype")
            or list(loaded.shape) != row.get("shape")
            or _array_sha256(loaded) != row.get("content_sha256")
        ):
            raise ValueError(f"HTR array dtype, shape, or content changed: {key}")
        arrays[key] = loaded
    observed = {
        path.relative_to(fit_root).as_posix()
        for path in (fit_root / "arrays").iterdir()
        if path.is_file()
    }
    if observed != expected_paths:
        raise ValueError("HTR array directory has missing or extra files")
    return arrays


def _array_references(value: Any, available: frozenset[str]) -> list[str]:
    references: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "array_inventory":
                continue
            references.extend(_array_references(child, available))
    elif isinstance(value, list):
        for child in value:
            references.extend(_array_references(child, available))
    elif isinstance(value, str) and value in available:
        references.append(value)
    return references


def _predict_from_state(
    *,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    texts: Sequence[str],
    htr_model_path: Path | str | None,
    device: torch.device,
    operational_controls: RoleNeutralHTROperationalControls | None = None,
    reusable_plan: _ReusableTextPlan | None = None,
    row_ids: Sequence[int] | None = None,
) -> tuple[list[str], np.ndarray]:
    values = tuple(str(text) for text in texts)
    config = RoleNeutralHTRConfig.from_mapping(
        {
            key: value
            for key, value in dict(metadata["configuration"]).items()
            if key
            not in {
                "schema_version",
                "text_truncation_applied",
            }
        }
    )
    if operational_controls is not None:
        operational_controls.validate_for(config)
        if (
            operational_controls.reuse_tokenizer_and_chunk_plans
            != (reusable_plan is not None)
            or row_ids is None
        ):
            raise ValueError(
                "HTR prediction operational controls and reusable plan differ"
            )
    elif reusable_plan is not None or row_ids is not None:
        raise ValueError("HTR prediction plan requires operational controls")

    def prepare_model(model: torch.nn.Module) -> None:
        if operational_controls is None:
            return
        if reusable_plan is not None:
            _install_reusable_text_plan(
                extractor=model.extractor,
                plan=reusable_plan,
                texts=values,
                row_ids=tuple(int(value) for value in row_ids or ()),
                config=config,
                controls=operational_controls,
            )
        _set_operational_encoder_batch_size(
            model.extractor,
            config=config,
            controls=operational_controls,
        )

    def restore_model(model: torch.nn.Module) -> None:
        if operational_controls is None:
            return
        _restore_scientific_encoder_batch_size(
            model.extractor,
            config=config,
            controls=operational_controls,
        )

    nuisance_e: list[np.ndarray] = []
    nuisance_m: list[np.ndarray] = []
    for record in metadata["nuisance_fold_states"]:
        model = _build_model(
            record["model"],
            arrays,
            initialization_texts=[""],
            htr_model_path=htr_model_path,
            device=device,
        )
        try:
            prepare_model(model)
            raw_e, raw_m = _predict_model(
                model,
                values,
                kind="nuisance",
                outcome_type=config.outcome_type,
                batch_size=config.prediction_batch_size,
            )
        finally:
            restore_model(model)
            del model
        nuisance_e.append(
            _apply_calibrator(
                record["propensity_calibrator"],
                arrays,
                raw_e,
            )
        )
        nuisance_m.append(
            _apply_calibrator(
                record["outcome_calibrator"],
                arrays,
                raw_m,
            )
        )
    if not nuisance_e or not nuisance_m:
        raise RuntimeError("sealed HTR state lacks nuisance models")
    columns = ["htr_nuisance::e_hat", "htr_nuisance::m_hat"]
    predictions = [
        np.mean(np.vstack(nuisance_e), axis=0),
        np.mean(np.vstack(nuisance_m), axis=0),
    ]
    effect_rows = metadata["effect_fold_states"]
    for objective in config.effect_objectives:
        objective_predictions: list[np.ndarray] = []
        for record in effect_rows:
            if record.get("effect_objective") != objective:
                continue
            model = _build_model(
                record["model"],
                arrays,
                initialization_texts=[""],
                htr_model_path=htr_model_path,
                device=device,
            )
            try:
                prepare_model(model)
                [raw_tau] = _predict_model(
                    model,
                    values,
                    kind="effect",
                    outcome_type=config.outcome_type,
                    batch_size=config.prediction_batch_size,
                )
            finally:
                restore_model(model)
                del model
            objective_predictions.append(raw_tau)
        if len(objective_predictions) != config.effect_folds:
            raise RuntimeError(f"sealed HTR state lacks {objective} fold coverage")
        columns.append(f"htr_effect::{objective}")
        predictions.append(np.mean(np.vstack(objective_predictions), axis=0))
    matrix = np.column_stack(predictions).astype(np.float64, copy=False)
    if matrix.shape != (len(values), len(columns)) or not np.isfinite(matrix).all():
        raise RuntimeError("sealed HTR state emitted invalid exact predictions")
    return columns, matrix


def _validate_fit_side(
    *,
    root: Path,
    request: RoleNeutralHTRPhysicalGroupRequest,
    htr_model_path: Path | str | None,
    expected_fit_texts: Sequence[str] | None = None,
    expected_fit_treatment: Sequence[Any] | None = None,
    expected_fit_outcome: Sequence[Any] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    fit_root = root / _FIT_STATE_DIRECTORY
    metadata = _read_json(
        fit_root / _FIT_STATE_METADATA,
        label="HTR fit-state metadata",
    )
    body = {key: value for key, value in metadata.items() if key != "content_sha256"}
    if (
        metadata.get("schema_version") != ROLE_NEUTRAL_HTR_FIT_STATE_SCHEMA
        or metadata.get("content_sha256") != _sha256_json(body)
        or metadata.get("group_request_content_sha256") != request.content_sha256
        or metadata.get("plan_scientific_content_sha256")
        != request.plan_scientific_content_sha256
        or metadata.get("physical_owner_scope_id")
        != request.physical_owner.scope_id
        or metadata.get("physical_owner_scope_sha256")
        != request.physical_owner.as_dict()["scope_sha256"]
        or metadata.get("fit_row_ids")
        != list(request.physical_owner.fit_row_ids)
        or metadata.get("fit_row_order_fingerprint")
        != _row_order_fingerprint(request.physical_owner.fit_row_ids)
        or int(metadata.get("canonical_group_seed", -1))
        != int(request.physical_owner.scope_seed)
        or metadata.get("producer_identity_sha256") != _producer_identity()
        or metadata.get("registered_heldout_text_accessed") is not False
        or metadata.get("registered_heldout_labels_accessed") is not False
        or metadata.get("text_truncation_applied") is not False
        or metadata.get("array_layout") != "one_npy_per_array_mmap_safe_v1"
    ):
        raise ValueError("HTR fit-state envelope changed")
    config_mapping = metadata.get("configuration")
    if not isinstance(config_mapping, Mapping):
        raise ValueError("HTR fit-state lacks configuration")
    config = RoleNeutralHTRConfig.from_mapping(
        {
            key: value
            for key, value in config_mapping.items()
            if key not in {"schema_version", "text_truncation_applied"}
        }
    )
    runtime_class = metadata.get("runtime_compatibility_class")
    if not isinstance(runtime_class, str) or not runtime_class.strip():
        raise ValueError("HTR fit-state runtime compatibility class is invalid")
    expected_configuration_identity = _sha256_json(
        {
            "configuration": config.as_dict(),
            "runtime_compatibility_class": runtime_class,
        }
    )
    if metadata.get("configuration_identity_sha256") != expected_configuration_identity:
        raise ValueError("HTR fit-state configuration identity changed")
    _resolve_model_marker(config, htr_model_path)
    arrays = _load_array_store(fit_root, metadata.get("array_inventory"))
    references = _array_references(metadata, frozenset(arrays))
    if set(references) != set(arrays) or len(references) != len(set(references)):
        raise ValueError("HTR arrays are unused, multiply referenced, or unregistered")
    fit_coverage_values = _coverage_numeric_values(
        record=metadata.get("fit_coverage"),
        arrays=arrays,
        config=config,
        expected_phase="fit",
    )
    if expected_fit_texts is not None:
        texts = tuple(expected_fit_texts)
        if (
            len(texts) != len(request.physical_owner.fit_row_ids)
            or metadata.get("fit_text_sha256")
            != _text_sha256(request.physical_owner.fit_row_ids, texts)
        ):
            raise ValueError("HTR fit texts differ from the canonical owner")
        _assert_coverage_matches_plan(
            fit_coverage_values,
            _coverage_plan(texts=texts, config=config, phase="fit"),
            label="HTR fit coverage",
        )
    if (expected_fit_treatment is None) != (expected_fit_outcome is None):
        raise ValueError("canonical HTR fit labels must be supplied together")
    if expected_fit_treatment is not None:
        treatment = _binary_vector(
            expected_fit_treatment,
            label="expected HTR fit treatment",
            length=len(request.physical_owner.fit_row_ids),
        )
        outcome = _binary_vector(
            expected_fit_outcome,
            label="expected HTR fit outcome",
            length=len(request.physical_owner.fit_row_ids),
        )
        if (
            metadata.get("fit_treatment_sha256") != _float_hex_sha256(treatment)
            or metadata.get("fit_outcome_sha256") != _float_hex_sha256(outcome)
            or not np.array_equal(
                np.asarray(arrays[str(metadata["fit_treatment"])]),
                treatment,
            )
            or not np.array_equal(
                np.asarray(arrays[str(metadata["fit_outcome"])]),
                outcome,
            )
        ):
            raise ValueError("HTR fit labels differ from the canonical owner")
    expected_nuisance = list(range(1, config.nuisance_folds + 1))
    nuisance_rows = metadata.get("nuisance_fold_states")
    if (
        not isinstance(nuisance_rows, list)
        or [int(row.get("fold", 0)) for row in nuisance_rows] != expected_nuisance
    ):
        raise ValueError("HTR nuisance fold coverage changed")
    row_count = len(request.physical_owner.fit_row_ids)
    nuisance_split_seed = _derived_seed(
        request.physical_owner.scope_seed,
        purpose="split",
        objective="nuisance",
        fold=0,
    )
    nuisance_splits = list(
        KFold(
            n_splits=config.nuisance_folds,
            shuffle=True,
            random_state=nuisance_split_seed,
        ).split(np.arange(row_count))
    )
    replay_oof_e = np.full(row_count, np.nan, dtype=np.float64)
    replay_oof_m = np.full(row_count, np.nan, dtype=np.float64)
    for row, (expected_fit, expected_validation) in zip(
        nuisance_rows,
        nuisance_splits,
        strict=True,
    ):
        _validate_typed_model_configuration(
            row.get("model"),
            config=config,
            kind="nuisance",
        )
        fold = int(row["fold"])
        fit_pos = np.asarray(row.get("fit_positions"), dtype=np.int64)
        validation_pos = np.asarray(
            row.get("validation_positions"),
            dtype=np.int64,
        )
        validation_e = np.asarray(arrays[str(row.get("validation_e_hat"))])
        validation_m = np.asarray(arrays[str(row.get("validation_m_hat"))])
        expected_model_seed = _derived_seed(
            request.physical_owner.scope_seed,
            purpose="fit",
            objective="nuisance",
            fold=fold,
        )
        if (
            int(row.get("split_seed", -1)) != nuisance_split_seed
            or int(row.get("model_seed", -1)) != expected_model_seed
            or not np.array_equal(fit_pos, expected_fit)
            or not np.array_equal(validation_pos, expected_validation)
            or row.get("fit_row_ids")
            != [
                request.physical_owner.fit_row_ids[int(position)]
                for position in expected_fit
            ]
            or row.get("validation_row_ids")
            != [
                request.physical_owner.fit_row_ids[int(position)]
                for position in expected_validation
            ]
            or validation_e.shape != (len(expected_validation),)
            or validation_m.shape != (len(expected_validation),)
            or not np.isfinite(validation_e).all()
            or not np.isfinite(validation_m).all()
            or row.get("registered_heldout_text_accessed") is not False
            or row.get("registered_heldout_labels_accessed") is not False
        ):
            raise ValueError("HTR nuisance fold split/state changed")
        replay_oof_e[validation_pos] = validation_e
        replay_oof_m[validation_pos] = validation_m
    effect_rows = metadata.get("effect_fold_states")
    if not isinstance(effect_rows, list):
        raise ValueError("HTR effect fold state is missing")
    for objective in config.effect_objectives:
        observed = [
            int(row.get("fold", 0))
            for row in effect_rows
            if row.get("effect_objective") == objective
        ]
        if observed != list(range(1, config.effect_folds + 1)):
            raise ValueError(f"HTR {objective} fold coverage changed")
    expected_effect_count = config.effect_folds * len(config.effect_objectives)
    if len(effect_rows) != expected_effect_count:
        raise ValueError("HTR effect state contains another objective")
    derived = metadata.get("derived_fit_quantities")
    if not isinstance(derived, Mapping):
        raise ValueError("HTR derived fit quantities are missing")
    e_hat = np.asarray(arrays[str(derived.get("nuisance_oof_e"))])
    m_hat = np.asarray(arrays[str(derived.get("nuisance_oof_m"))])
    clipped_e = np.asarray(arrays[str(derived.get("clipped_e_hat"))])
    y_residual = np.asarray(arrays[str(derived.get("y_residual"))])
    t_residual = np.asarray(arrays[str(derived.get("t_residual"))])
    pseudo = np.asarray(arrays[str(derived.get("pseudo_outcome"))])
    eligible = np.asarray(
        arrays[str(derived.get("r_stage_eligible"))],
        dtype=np.uint8,
    )
    fit_treatment = np.asarray(arrays[str(metadata.get("fit_treatment"))])
    fit_outcome = np.asarray(arrays[str(metadata.get("fit_outcome"))])
    expected_clipped = np.clip(e_hat, config.e_clip, 1.0 - config.e_clip)
    expected_y_residual = fit_outcome - m_hat
    expected_t_residual = fit_treatment - expected_clipped
    expected_pseudo = expected_y_residual / expected_t_residual
    expected_eligible = (
        (e_hat >= config.r_stage_min_propensity)
        & (e_hat <= config.r_stage_max_propensity)
        & np.isfinite(expected_pseudo)
    )
    if (
        not np.array_equal(e_hat, replay_oof_e)
        or not np.array_equal(m_hat, replay_oof_m)
        or not np.array_equal(clipped_e, expected_clipped)
        or not np.array_equal(y_residual, expected_y_residual)
        or not np.array_equal(t_residual, expected_t_residual)
        or not np.array_equal(pseudo, expected_pseudo)
        or not np.array_equal(eligible, expected_eligible.astype(np.uint8))
    ):
        raise ValueError("HTR derived nuisance/R-stage quantities changed")
    for objective in config.effect_objectives:
        split_seed = _derived_seed(
            request.physical_owner.scope_seed,
            purpose="split",
            objective=objective,
            fold=0,
        )
        expected_splits = list(
            KFold(
                n_splits=config.effect_folds,
                shuffle=True,
                random_state=split_seed,
            ).split(np.arange(row_count))
        )
        rows = [
            row
            for row in effect_rows
            if row.get("effect_objective") == objective
        ]
        replay_oof = np.full(row_count, np.nan, dtype=np.float64)
        for row, (expected_fit, expected_validation) in zip(
            rows,
            expected_splits,
            strict=True,
        ):
            _validate_typed_model_configuration(
                row.get("model"),
                config=config,
                kind="effect",
            )
            fold = int(row["fold"])
            fit_pos = np.asarray(row.get("fit_positions"), dtype=np.int64)
            eligible_fit_pos = np.asarray(
                row.get("eligible_fit_positions"),
                dtype=np.int64,
            )
            validation_pos = np.asarray(
                row.get("validation_positions"),
                dtype=np.int64,
            )
            validation_tau = np.asarray(
                arrays[str(row.get("validation_tau"))]
            )
            expected_eligible_fit = expected_fit[
                expected_eligible[expected_fit]
            ]
            expected_model_seed = _derived_seed(
                request.physical_owner.scope_seed,
                purpose="fit",
                objective=objective,
                fold=fold,
            )
            if (
                int(row.get("split_seed", -1)) != split_seed
                or int(row.get("model_seed", -1)) != expected_model_seed
                or not np.array_equal(fit_pos, expected_fit)
                or not np.array_equal(validation_pos, expected_validation)
                or not np.array_equal(
                    eligible_fit_pos,
                    expected_eligible_fit,
                )
                or row.get("fit_row_ids")
                != [
                    request.physical_owner.fit_row_ids[int(position)]
                    for position in expected_fit
                ]
                or row.get("eligible_fit_row_ids")
                != [
                    request.physical_owner.fit_row_ids[int(position)]
                    for position in expected_eligible_fit
                ]
                or row.get("validation_row_ids")
                != [
                    request.physical_owner.fit_row_ids[int(position)]
                    for position in expected_validation
                ]
                or validation_tau.shape != (len(expected_validation),)
                or not np.isfinite(validation_tau).all()
                or row.get("registered_heldout_text_accessed") is not False
                or row.get("registered_heldout_labels_accessed") is not False
            ):
                raise ValueError(f"HTR {objective} fold split/state changed")
            replay_oof[validation_pos] = validation_tau
        if not np.array_equal(
            replay_oof,
            np.asarray(arrays[str(derived.get(f"effect_oof_{objective}"))]),
        ):
            raise ValueError(f"HTR {objective} OOF state changed")
    _validate_complete_attention_evidence(
        payload=metadata.get("evidence_payload"),
        coverage=fit_coverage_values,
        nuisance_records=nuisance_rows,
        effect_records=effect_rows,
        config=config,
    )
    # Reconstruct every native model before permitting registered held-out text.
    replay_device = torch.device(device)
    for record in [*nuisance_rows, *effect_rows]:
        model = _build_model(
            record["model"],
            arrays,
            initialization_texts=[""],
            htr_model_path=htr_model_path,
            device=replay_device,
        )
        del model
    fit_state_sha256 = _tree_sha256(fit_root)
    seal = _read_json(root / _FIT_SEAL_FILE, label="HTR fit-only family seal")
    expected_seal = _fit_seal(
        request=request,
        evidence_payload=metadata["evidence_payload"],
        producer_identity_sha256=metadata["producer_identity_sha256"],
        configuration_identity_sha256=metadata[
            "configuration_identity_sha256"
        ],
        fit_state_artifact_sha256=fit_state_sha256,
    )
    if seal != expected_seal:
        raise ValueError("HTR fit-only family seal changed")
    return metadata, arrays, seal


def execute_role_neutral_htr_physical_group(
    *,
    request: RoleNeutralHTRPhysicalGroupRequest,
    output_root: Path | str,
    fit_texts: Sequence[str],
    fit_treatment: Sequence[Any],
    fit_outcome: Sequence[Any],
    config: RoleNeutralHTRConfig,
    runtime_compatibility_class: str,
    exact_heldout_text_loader: Callable[[tuple[int, ...]], Sequence[str]],
    htr_model_path: Path | str | None = None,
    device: torch.device | str = "cpu",
    operational_controls: RoleNeutralHTROperationalControls | None = None,
    fold_resource_plan: RoleNeutralHTRFoldResourcePlan | None = None,
    operational_attestation_sink: (
        Callable[[Mapping[str, Any]], None] | None
    ) = None,
    fold_event_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> Mapping[str, Any]:
    """Fit, seal, publish aliases, then transform authorized exact text."""

    if not isinstance(request, RoleNeutralHTRPhysicalGroupRequest):
        raise TypeError("role-neutral HTR execution requires its typed request")
    request.as_dict()
    if not isinstance(config, RoleNeutralHTRConfig):
        raise TypeError("role-neutral HTR execution requires its typed config")
    config.validated()
    runtime_class = str(runtime_compatibility_class).strip()
    if not runtime_class:
        raise ValueError("HTR runtime compatibility class cannot be empty")
    if not callable(exact_heldout_text_loader):
        raise TypeError("exact held-out HTR text loader must be callable")
    if operational_controls is None:
        if (
            operational_attestation_sink is not None
            or fold_resource_plan is not None
        ):
            raise ValueError(
                "HTR operational resources and attestation require typed controls"
            )
    else:
        if not isinstance(
            operational_controls,
            RoleNeutralHTROperationalControls,
        ):
            raise TypeError("HTR operational controls must use the typed contract")
        operational_controls.validate_for(config)
        if not callable(operational_attestation_sink):
            raise TypeError(
                "typed HTR operational controls require an attestation sink"
            )
        if not isinstance(fold_resource_plan, RoleNeutralHTRFoldResourcePlan):
            raise TypeError(
                "typed HTR operational controls require a fold resource plan"
            )
        if (
            fold_resource_plan.fold_parallelism
            != operational_controls.fold_parallelism
            or fold_resource_plan.fold_slots_per_device
            != operational_controls.fold_slots_per_device
            or fold_resource_plan.fold_parallel_backend
            != operational_controls.fold_parallel_backend
        ):
            raise ValueError(
                "HTR fold resource plan differs from operational controls"
            )
        if (
            fold_resource_plan.fold_parallelism > 1
            and fold_resource_plan.fold_parallel_backend != "processes"
        ):
            raise ValueError(
                "overlapping HTR training folds require process-isolated RNG"
            )
        if (
            any(
                value != "cpu"
                for value in fold_resource_plan.devices
            )
            and fold_resource_plan.fold_parallel_backend != "processes"
        ):
            raise ValueError(
                "CUDA HTR fold execution requires process isolation"
            )
        if (
            fold_resource_plan.fold_parallel_backend == "processes"
            or fold_resource_plan.fold_parallelism > 1
        ) and not operational_controls.reuse_tokenizer_and_chunk_plans:
            raise ValueError(
                "parallel/process HTR folds require one complete reusable "
                "owner tokenizer/chunk plan"
            )
    if fold_event_sink is not None and not callable(fold_event_sink):
        raise TypeError("HTR fold event sink must be callable")
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("role-neutral HTR output root must be absolute")
    if root.exists() or root.is_symlink():
        raise FileExistsError("role-neutral HTR output root must be fresh")
    owner = request.physical_owner
    texts = tuple(fit_texts)
    if (
        len(texts) != len(owner.fit_row_ids)
        or any(not isinstance(text, str) for text in texts)
    ):
        raise ValueError("HTR fit texts must align exactly to owner fit rows")
    if len(texts) < max(config.nuisance_folds, config.effect_folds):
        raise ValueError("HTR fit rows are insufficient for configured folds")
    treatment = _binary_vector(
        fit_treatment,
        label="HTR fit treatment",
        length=len(texts),
    )
    outcome = _binary_vector(
        fit_outcome,
        label="HTR fit outcome",
        length=len(texts),
    )
    fit_coverage = _coverage_plan(texts=texts, config=config, phase="fit")
    fit_reusable_plan: _ReusableTextPlan | None = None
    heldout_reusable_plan: _ReusableTextPlan | None = None
    model_marker = _resolve_model_marker(config, htr_model_path)
    execution_device = torch.device(device)
    if operational_controls is None:
        effective_fold_resource_plan = RoleNeutralHTRFoldResourcePlan(
            devices=(str(execution_device),),
            fold_parallelism=1,
            fold_slots_per_device=1,
            owner_cpu_budget=1,
            fold_parallel_backend="threads",
        )
    else:
        assert fold_resource_plan is not None
        effective_fold_resource_plan = fold_resource_plan
        if str(execution_device) not in effective_fold_resource_plan.devices:
            raise ValueError(
                "HTR fold resource plan omits the primary execution device"
            )
        if (
            len(effective_fold_resource_plan.devices)
            > config.nuisance_folds
            or len(effective_fold_resource_plan.devices)
            > config.effect_folds * len(config.effect_objectives)
        ):
            raise ValueError(
                "HTR selected devices exceed folds available in one stage"
            )
    producer_identity = _producer_identity()
    configuration_identity = _sha256_json(
        {
            "configuration": config.as_dict(),
            "runtime_compatibility_class": runtime_class,
        }
    )

    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    store = _SafeArrayStore()
    fit_treatment_key = store.add("fit_treatment", treatment)
    fit_outcome_key = store.add("fit_outcome", outcome)
    coverage_record = _coverage_arrays(
        store=store,
        plan=fit_coverage,
        prefix="fit_coverage",
    )
    fold_fit = _fit_owner_htr_folds(
        owner=owner,
        texts=texts,
        treatment=treatment,
        outcome=outcome,
        coverage=fit_coverage,
        config=config,
        model_marker=model_marker,
        store=store,
        operational_controls=operational_controls,
        resource_plan=effective_fold_resource_plan,
        scratch_parent=root.parent,
        external_event_sink=fold_event_sink,
    )
    fit_reusable_plan = fold_fit.fit_reusable_plan
    process_plan_attestation = fold_fit.process_plan_attestation
    fold_execution_events = fold_fit.fold_execution_events
    nuisance_oof_e = np.asarray(
        fold_fit.nuisance_oof_e,
        dtype=np.float64,
    )
    nuisance_oof_m = np.asarray(
        fold_fit.nuisance_oof_m,
        dtype=np.float64,
    )
    nuisance_records: list[dict[str, Any]] = [
        copy.deepcopy(dict(value)) for value in fold_fit.nuisance_records
    ]
    architecture_evidence: list[dict[str, Any]] = [
        copy.deepcopy(dict(value))
        for value in fold_fit.architecture_evidence
    ]
    if not np.isfinite(nuisance_oof_e).all() or not np.isfinite(
        nuisance_oof_m
    ).all():
        raise RuntimeError("HTR nuisance cross-fit omitted a fit row")
    clipped_e = np.clip(nuisance_oof_e, config.e_clip, 1.0 - config.e_clip)
    y_residual = outcome - nuisance_oof_m
    t_residual = treatment - clipped_e
    pseudo_outcome = y_residual / t_residual
    eligible = (
        (nuisance_oof_e >= config.r_stage_min_propensity)
        & (nuisance_oof_e <= config.r_stage_max_propensity)
        & np.isfinite(pseudo_outcome)
    )
    if not np.any(eligible):
        raise ValueError("configured HTR R-stage bounds retain no fit rows")
    if any(
        not np.array_equal(observed, expected)
        for observed, expected in (
            (clipped_e, fold_fit.clipped_e),
            (y_residual, fold_fit.y_residual),
            (t_residual, fold_fit.t_residual),
            (pseudo_outcome, fold_fit.pseudo_outcome),
            (eligible, fold_fit.eligible),
        )
    ):
        raise RuntimeError("HTR nuisance barrier quantities changed after merge")
    derived = {
        "nuisance_oof_e": store.add("nuisance_oof_e", nuisance_oof_e),
        "nuisance_oof_m": store.add("nuisance_oof_m", nuisance_oof_m),
        "clipped_e_hat": store.add("clipped_e_hat", clipped_e),
        "y_residual": store.add("y_residual", y_residual),
        "t_residual": store.add("t_residual", t_residual),
        "pseudo_outcome": store.add("pseudo_outcome", pseudo_outcome),
        "r_stage_eligible": store.add(
            "r_stage_eligible",
            eligible.astype(np.uint8),
        ),
    }
    effect_records: list[dict[str, Any]] = [
        copy.deepcopy(dict(value)) for value in fold_fit.effect_records
    ]
    for objective in config.effect_objectives:
        derived[f"effect_oof_{objective}"] = store.add(
            f"effect_oof_{objective}",
            fold_fit.effect_oof[objective],
        )

    evidence_payload = {
        "schema_version": NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
        "family": HTR_NEURAL,
        "architecture_evidence": architecture_evidence,
    }
    fit_root = root / _FIT_STATE_DIRECTORY
    fit_root.mkdir(parents=True, exist_ok=False)
    array_inventory = _write_array_store(fit_root=fit_root, store=store)
    metadata_body = {
        "schema_version": ROLE_NEUTRAL_HTR_FIT_STATE_SCHEMA,
        "group_request_content_sha256": request.content_sha256,
        "plan_scientific_content_sha256": (
            request.plan_scientific_content_sha256
        ),
        "physical_owner_scope_id": owner.scope_id,
        "physical_owner_scope_sha256": owner.as_dict()["scope_sha256"],
        "fit_row_ids": list(owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(owner.fit_row_ids),
        "canonical_group_seed": int(owner.scope_seed),
        "fit_text_sha256": _text_sha256(owner.fit_row_ids, texts),
        "fit_treatment_sha256": _float_hex_sha256(treatment),
        "fit_outcome_sha256": _float_hex_sha256(outcome),
        "fit_treatment": fit_treatment_key,
        "fit_outcome": fit_outcome_key,
        "configuration": config.as_dict(),
        "configuration_identity_sha256": configuration_identity,
        "producer_identity_sha256": producer_identity,
        "runtime_compatibility_class": runtime_class,
        "fit_coverage": coverage_record,
        "nuisance_fold_states": nuisance_records,
        "effect_fold_states": effect_records,
        "derived_fit_quantities": derived,
        "evidence_payload": evidence_payload,
        "array_inventory": array_inventory,
        "array_layout": "one_npy_per_array_mmap_safe_v1",
        "registered_heldout_text_accessed": False,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "pickle_or_joblib_loaded": False,
        "text_truncation_applied": False,
    }
    metadata = {
        **metadata_body,
        "content_sha256": _sha256_json(metadata_body),
    }
    _write_new_json(fit_root / _FIT_STATE_METADATA, metadata)
    fit_state_sha256 = _tree_sha256(fit_root)
    seal = _fit_seal(
        request=request,
        evidence_payload=evidence_payload,
        producer_identity_sha256=producer_identity,
        configuration_identity_sha256=configuration_identity,
        fit_state_artifact_sha256=fit_state_sha256,
    )
    _write_new_json(root / _FIT_SEAL_FILE, seal)

    # Freshly reopen every fit-side byte and reconstruct every native model.
    # Only a successful result permits the registered held-out loader below.
    _validate_fit_side(
        root=root,
        request=request,
        htr_model_path=htr_model_path,
        expected_fit_texts=texts,
        expected_fit_treatment=treatment,
        expected_fit_outcome=outcome,
        device=execution_device,
    )

    logical_root = root / _LOGICAL_VIEW_DIRECTORY
    logical_root.mkdir(parents=True, exist_ok=False)
    seal_sha256, seal_size = _sha256_file(
        root / _FIT_SEAL_FILE,
        label="HTR fit-only family seal",
    )
    events: list[dict[str, Any]] = [
        {
            "sequence": 1,
            "event": "fit_completed",
            "family": HTR_NEURAL,
            "fit_state_artifact_sha256": fit_state_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        },
        {
            "sequence": 2,
            "event": "fit_family_artifact_sealed",
            "family": HTR_NEURAL,
            "fit_only_family_seal_sha256": seal_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        },
    ]
    logical_registrations: list[dict[str, Any]] = []
    for member in request.logical_members[1:]:
        body = {
            "schema_version": ROLE_NEUTRAL_HTR_LOGICAL_VIEW_SCHEMA,
            "group_request_content_sha256": request.content_sha256,
            "logical_scope_id": member.scope_id,
            "logical_scope_sha256": member.as_dict()["scope_sha256"],
            "logical_purpose": member.scope_kind,
            "physical_owner_scope_id": owner.scope_id,
            "family": HTR_NEURAL,
            "fit_only_family_seal_sha256": seal_sha256,
            "fit_only_family_seal_content_sha256": seal["content_sha256"],
            "view_input_policy": "sealed_row_ids_only_no_sealed_text_or_labels_v1",
            "logical_heldout_row_ids": list(member.heldout_row_ids),
            "logical_transform_performed": False,
            "prediction_artifact": None,
            "coverage_artifacts": None,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
            "reuses_physical_fit_state_by_immutable_reference": True,
        }
        view = {**body, "content_sha256": _sha256_json(body)}
        path = logical_root / f"{member.scope_id}.json"
        _write_new_json(path, view)
        digest, size = _sha256_file(path, label="cumulative HTR view")
        logical_registrations.append(
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
                "family": HTR_NEURAL,
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
            }
        )

    loaded = exact_heldout_text_loader(tuple(owner.heldout_row_ids))
    heldout_texts = tuple(loaded)
    if (
        len(heldout_texts) != len(owner.heldout_row_ids)
        or any(not isinstance(text, str) for text in heldout_texts)
    ):
        raise ValueError("exact HTR text loader returned another row/text shape")
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "exact_heldout_text_opened",
            "logical_scope_id": owner.scope_id,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
        }
    )
    heldout_coverage = _coverage_plan(
        texts=heldout_texts,
        config=config,
        phase="exact_heldout",
    )
    # The first sealed model's preprocessor proves max_chunk_length remains
    # nonbinding for the newly authorized notes before any prediction.
    first_model = _build_model(
        metadata["nuisance_fold_states"][0]["model"],
        store.arrays,
        initialization_texts=[""],
        htr_model_path=htr_model_path,
        device=execution_device,
    )
    try:
        if (
            operational_controls is not None
            and operational_controls.reuse_tokenizer_and_chunk_plans
        ):
            heldout_reusable_plan = _build_reusable_text_plan(
                extractor=first_model.extractor,
                texts=heldout_texts,
                row_ids=owner.heldout_row_ids,
                coverage=heldout_coverage,
                config=config,
                controls=operational_controls,
                phase="exact_heldout",
            )
            _install_reusable_text_plan(
                extractor=first_model.extractor,
                plan=heldout_reusable_plan,
                texts=heldout_texts,
                row_ids=owner.heldout_row_ids,
                config=config,
                controls=operational_controls,
            )
        else:
            _preflight_token_lengths(
                first_model.extractor,
                heldout_texts,
                batch_size=config.prediction_batch_size,
            )
    finally:
        del first_model
    columns, prediction_matrix = _predict_from_state(
        metadata=metadata,
        arrays=store.arrays,
        texts=heldout_texts,
        htr_model_path=htr_model_path,
        device=execution_device,
        operational_controls=operational_controls,
        reusable_plan=heldout_reusable_plan,
        row_ids=(
            owner.heldout_row_ids
            if operational_controls is not None
            else None
        ),
    )
    operational_replay_equal = True
    if operational_controls is not None:
        replay_columns, replay_matrix = _predict_from_state(
            metadata=metadata,
            arrays=store.arrays,
            texts=heldout_texts,
            htr_model_path=htr_model_path,
            device=execution_device,
        )
        operational_replay_equal = (
            replay_columns == columns
            and neural_float_arrays_within_tolerance(
                replay_matrix,
                prediction_matrix,
                policy=config.replay_comparison_policy,
                relative_tolerance=config.replay_relative_tolerance,
                absolute_tolerance=config.replay_absolute_tolerance,
            )
        )
        if not operational_replay_equal:
            raise RuntimeError(
                "HTR operational encoder batching differs from the "
                "scientific replay beyond its declared tolerance"
            )
    prediction_path = logical_root / f"{owner.scope_id}.predictions.npy"
    _write_new_npy(prediction_path, prediction_matrix)
    prediction_sha256, prediction_size = _sha256_file(
        prediction_path,
        label="exact HTR predictions",
    )
    coverage_store = _SafeArrayStore()
    coverage_record = _coverage_arrays(
        store=coverage_store,
        plan=heldout_coverage,
        prefix="heldout_coverage",
    )
    coverage_registrations: dict[str, Any] = {}
    for key in sorted(coverage_store.arrays):
        path = logical_root / f"{owner.scope_id}.{key}.npy"
        _write_new_npy(path, coverage_store.arrays[key])
        digest, size = _sha256_file(path, label=f"exact HTR coverage {key}")
        registered = coverage_store.inventory[key]
        coverage_registrations[key] = {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "size_bytes": size,
            "dtype": registered["dtype"],
            "shape": registered["shape"],
            "content_sha256": registered["content_sha256"],
        }
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "exact_heldout_transform_completed",
            "logical_scope_id": owner.scope_id,
            "family": HTR_NEURAL,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
        }
    )
    exact_body = {
        "schema_version": ROLE_NEUTRAL_HTR_LOGICAL_VIEW_SCHEMA,
        "group_request_content_sha256": request.content_sha256,
        "logical_scope_id": owner.scope_id,
        "logical_scope_sha256": owner.as_dict()["scope_sha256"],
        "logical_purpose": owner.scope_kind,
        "physical_owner_scope_id": owner.scope_id,
        "family": HTR_NEURAL,
        "fit_only_family_seal_sha256": seal_sha256,
        "fit_only_family_seal_content_sha256": seal["content_sha256"],
        "view_input_policy": "heldout_row_id_and_complete_text_no_labels_v1",
        "logical_heldout_row_ids": list(owner.heldout_row_ids),
        "logical_heldout_text_sha256": _text_sha256(
            owner.heldout_row_ids,
            heldout_texts,
        ),
        "logical_transform_performed": True,
        "prediction_artifact": {
            "relative_path": prediction_path.relative_to(root).as_posix(),
            "sha256": prediction_sha256,
            "size_bytes": prediction_size,
            "dtype": prediction_matrix.dtype.str,
            "shape": list(prediction_matrix.shape),
            "columns": columns,
            "content_sha256": _array_sha256(prediction_matrix),
        },
        "coverage_proof": coverage_record,
        "coverage_artifacts": coverage_registrations,
        "registered_heldout_text_accessed": True,
        "registered_heldout_labels_accessed": False,
        "reuses_physical_fit_state_by_immutable_reference": True,
        "model_state_reloaded_for_primary_transform": True,
        "sealed_state_replay_checked": True,
    }
    exact_view = {
        **exact_body,
        "content_sha256": _sha256_json(exact_body),
    }
    exact_path = logical_root / f"{owner.scope_id}.json"
    _write_new_json(exact_path, exact_view)
    exact_sha256, exact_size = _sha256_file(
        exact_path,
        label="exact HTR logical view",
    )
    logical_registrations.append(
        {
            "logical_scope_id": owner.scope_id,
            "relative_path": exact_path.relative_to(root).as_posix(),
            "sha256": exact_sha256,
            "size_bytes": exact_size,
            "content_sha256": exact_view["content_sha256"],
        }
    )
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "exact_logical_view_published",
            "logical_scope_id": owner.scope_id,
            "family": HTR_NEURAL,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
        }
    )
    logical_registrations.sort(
        key=lambda registration: next(
            index
            for index, member in enumerate(request.logical_members)
            if member.scope_id == registration["logical_scope_id"]
        )
    )
    terminal_body = {
        "schema_version": ROLE_NEUTRAL_HTR_GROUP_EXECUTION_SCHEMA,
        "status": "complete",
        "group_request": request.as_dict(),
        "family": HTR_NEURAL,
        "fit_state_artifact_sha256": fit_state_sha256,
        "fit_only_family_seal": {
            "relative_path": _FIT_SEAL_FILE,
            "sha256": seal_sha256,
            "size_bytes": seal_size,
            "content_sha256": seal["content_sha256"],
        },
        "logical_views": logical_registrations,
        "event_order": events,
        "fit_completed_before_registered_heldout_text_access": True,
        "fit_sealed_before_registered_heldout_text_access": True,
        "cumulative_views_published_without_sealed_text": True,
        "model_state_reloaded_for_primary_transform": True,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "pickle_or_joblib_loaded": False,
        "text_truncation_applied": False,
    }
    terminal = {
        **terminal_body,
        "content_sha256": _sha256_json(terminal_body),
    }
    _write_new_json(root / _TERMINAL_FILE, terminal)
    validated_terminal = validate_role_neutral_htr_group_execution(
        root=root,
        request=request,
        htr_model_path=htr_model_path,
        device=execution_device,
    )
    if operational_controls is not None:
        if fit_reusable_plan is not None:
            fit_plan_attestation = fit_reusable_plan.attestation()
        else:
            fit_plan_attestation = None
        if heldout_reusable_plan is not None:
            heldout_plan_attestation = heldout_reusable_plan.attestation()
        else:
            heldout_plan_attestation = None
        reusable_plan_attestations = tuple(
            value
            for value in (
                fit_plan_attestation,
                heldout_plan_attestation,
            )
            if value is not None
        )
        capacities_nonbinding = (
            not operational_controls.reuse_tokenizer_and_chunk_plans
            or (
                len(reusable_plan_attestations) == 2
                and all(
                    int(value["unique_note_count"])
                    <= operational_controls.chunk_plan_cache_max_entries
                    and int(value["unique_chunk_count"])
                    <= operational_controls.tokenized_chunk_cache_max_entries
                    for value in reusable_plan_attestations
                )
            )
        )
        positive_workers_exercised = (
            operational_controls.data_loader_workers == 0
            or (
                len(reusable_plan_attestations) == 2
                and all(
                    value.get(
                        "positive_data_loader_workers_exercised"
                    )
                    is True
                    and int(value.get("parallel_plan_task_count", 0)) > 0
                    and int(value.get("parallel_plan_thread_count", 0)) > 0
                    for value in reusable_plan_attestations
                )
            )
        )
        if not capacities_nonbinding:
            raise RuntimeError(
                "HTR reusable-plan cache capacities bound complete evidence"
            )
        if not positive_workers_exercised:
            raise RuntimeError(
                "configured positive HTR data-loader workers were not exercised"
            )
        fold_summary = _fold_execution_summary(
            events=fold_execution_events,
            resource_plan=effective_fold_resource_plan,
            config=config,
        )
        if (
            effective_fold_resource_plan.fold_parallel_backend == "processes"
        ) != (process_plan_attestation is not None):
            raise RuntimeError(
                "HTR process folds and their temporary plan attestation differ"
            )
        operational_body = {
            "schema_version": ROLE_NEUTRAL_HTR_OPERATIONAL_ATTESTATION_SCHEMA,
            "controls": operational_controls.as_dict(),
            "fold_resource_plan": effective_fold_resource_plan.as_dict(),
            "fold_execution": fold_summary,
            "fold_execution_events": [
                copy.deepcopy(dict(value))
                for value in fold_execution_events
            ],
            "process_reusable_plan": (
                None
                if process_plan_attestation is None
                else copy.deepcopy(dict(process_plan_attestation))
            ),
            "temporary_process_plan_removed_before_artifact_publication": True,
            "raw_text_persisted_in_temporary_process_plan_after_folds": False,
            "complete_owner_tokenizer_chunk_plan_built_once": (
                fit_plan_attestation is not None
            ),
            "canonical_fold_result_merge_order": (
                "nuisance_fold_then_effect_objective_fold_v1"
            ),
            "shared_mutable_array_store_used_by_fold_workers": False,
            "scientific_training_batch_size": config.batch_size,
            "training_batch_override_applied": False,
            "scientific_sentence_encoder_batch_size": (
                config.sentence_encoder_batch_size
            ),
            "effective_sentence_encoder_batch_size": (
                operational_controls.sentence_encoder_batch_size
            ),
            "fit_reusable_plan": fit_plan_attestation,
            "exact_heldout_reusable_plan": heldout_plan_attestation,
            "cache_capacities_nonbinding": capacities_nonbinding,
            "positive_data_loader_workers_exercised": (
                positive_workers_exercised
            ),
            "replay_comparison_policy": config.replay_comparison_policy,
            "replay_relative_tolerance": config.replay_relative_tolerance,
            "replay_absolute_tolerance": config.replay_absolute_tolerance,
            "operational_predictions_within_declared_tolerance_of_scientific_replay": (
                operational_replay_equal
            ),
            "complete_artifact_equality_decided_by_benchmark": True,
            "raw_text_persisted_in_operational_attestation": False,
            "semantic_truncation_applied": False,
        }
        operational_attestation = {
            **operational_body,
            "content_sha256": _sha256_json(operational_body),
        }
        assert operational_attestation_sink is not None
        operational_attestation_sink(
            copy.deepcopy(operational_attestation)
        )
    return validated_terminal


def replay_role_neutral_htr_exact_transform(
    *,
    root: Path | str,
    request: RoleNeutralHTRPhysicalGroupRequest,
    exact_heldout_texts: Sequence[str],
    htr_model_path: Path | str | None = None,
    device: torch.device | str = "cpu",
) -> Mapping[str, Any]:
    """Freshly replay the exact logical transform from JSON/NPY state only."""

    artifact_root = Path(root)
    metadata, arrays, _seal = _validate_fit_side(
        root=artifact_root,
        request=request,
        htr_model_path=htr_model_path,
        device=device,
    )
    exact_view = _read_json(
        artifact_root
        / _LOGICAL_VIEW_DIRECTORY
        / f"{request.physical_owner.scope_id}.json",
        label="exact HTR logical view",
    )
    texts = tuple(exact_heldout_texts)
    if (
        len(texts) != len(request.physical_owner.heldout_row_ids)
        or any(not isinstance(text, str) for text in texts)
        or exact_view.get("logical_heldout_text_sha256")
        != _text_sha256(request.physical_owner.heldout_row_ids, texts)
    ):
        raise ValueError("replay HTR texts differ from the exact logical rows")
    config = RoleNeutralHTRConfig.from_mapping(
        {
            key: value
            for key, value in metadata["configuration"].items()
            if key not in {"schema_version", "text_truncation_applied"}
        }
    )
    expected_coverage = _coverage_plan(
        texts=texts,
        config=config,
        phase="exact_heldout",
    )
    coverage_artifacts = exact_view.get("coverage_artifacts")
    if not isinstance(coverage_artifacts, Mapping):
        raise ValueError("replay HTR view lacks coverage artifacts")
    coverage_arrays: dict[str, np.ndarray] = {}
    for key, registration in coverage_artifacts.items():
        if not isinstance(registration, Mapping):
            raise ValueError("replay HTR coverage registration is invalid")
        path = artifact_root / str(registration.get("relative_path") or "")
        digest, size, array = _read_npy_file_once(
            path,
            label=f"replay HTR coverage {key}",
            invalid_message=(
                f"replay HTR coverage array is not safe NumPy data: {key}"
            ),
        )
        if (
            digest != registration.get("sha256")
            or size != int(registration.get("size_bytes", -1))
            or array.dtype.str != registration.get("dtype")
            or list(array.shape) != registration.get("shape")
            or _array_sha256(array) != registration.get("content_sha256")
        ):
            raise ValueError("replay HTR coverage array changed")
        coverage_arrays[str(key)] = array
    observed_coverage = _coverage_numeric_values(
        record=exact_view.get("coverage_proof"),
        arrays=coverage_arrays,
        config=config,
        expected_phase="exact_heldout",
    )
    _assert_coverage_matches_plan(
        observed_coverage,
        expected_coverage,
        label="HTR exact replay coverage",
    )
    columns, predictions = _predict_from_state(
        metadata=metadata,
        arrays=arrays,
        texts=texts,
        htr_model_path=htr_model_path,
        device=torch.device(device),
    )
    registered = exact_view.get("prediction_artifact")
    expected_prediction_relative = (
        f"{_LOGICAL_VIEW_DIRECTORY}/"
        f"{request.physical_owner.scope_id}.predictions.npy"
    )
    if (
        not isinstance(registered, Mapping)
        or registered.get("relative_path") != expected_prediction_relative
    ):
        raise RuntimeError("registered HTR replay output is missing or noncanonical")
    prediction_path = artifact_root / expected_prediction_relative
    prediction_digest, prediction_size, sealed_predictions = _read_npy_file_once(
        prediction_path,
        label="registered exact HTR predictions",
        invalid_message=(
            "registered exact HTR predictions are not safe NumPy data"
        ),
    )
    if (
        prediction_digest != registered.get("sha256")
        or prediction_size != int(registered.get("size_bytes", -1))
        or registered.get("columns") != columns
        or registered.get("dtype") != sealed_predictions.dtype.str
        or registered.get("shape") != list(sealed_predictions.shape)
        or registered.get("content_sha256")
        != _array_sha256(sealed_predictions)
    ):
        raise ValueError("registered exact HTR prediction bytes changed")
    if not neural_float_arrays_within_tolerance(
        predictions,
        sealed_predictions,
        policy=config.replay_comparison_policy,
        relative_tolerance=config.replay_relative_tolerance,
        absolute_tolerance=config.replay_absolute_tolerance,
    ):
        raise RuntimeError(
            "fresh HTR replay differs from registered output beyond its "
            "declared tolerance"
        )
    return {
        "columns": columns,
        "predictions": predictions,
        "state_source": "authenticated_json_and_per_array_npy_only",
        "allow_pickle": False,
        "pickle_or_joblib_loaded": False,
        "heldout_labels_accessed": False,
        "text_truncation_applied": False,
    }


def validate_role_neutral_htr_group_execution(
    *,
    root: Path | str,
    request: RoleNeutralHTRPhysicalGroupRequest,
    htr_model_path: Path | str | None = None,
    device: torch.device | str = "cpu",
) -> Mapping[str, Any]:
    """Fresh path-only validation of one completed HTR physical group."""

    if not isinstance(request, RoleNeutralHTRPhysicalGroupRequest):
        raise TypeError("HTR validation requires its typed group request")
    request.as_dict()
    artifact_root = Path(root)
    files, directories = _inventory_tree(artifact_root)
    metadata, arrays, seal = _validate_fit_side(
        root=artifact_root,
        request=request,
        htr_model_path=htr_model_path,
        device=device,
    )
    config = RoleNeutralHTRConfig.from_mapping(metadata["configuration"])
    terminal = _read_json(
        artifact_root / _TERMINAL_FILE,
        label="HTR execution manifest",
    )
    terminal_body = {
        key: value for key, value in terminal.items() if key != "content_sha256"
    }
    if (
        terminal.get("schema_version") != ROLE_NEUTRAL_HTR_GROUP_EXECUTION_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("content_sha256") != _sha256_json(terminal_body)
        or terminal.get("group_request") != request.as_dict()
        or terminal.get("family") != HTR_NEURAL
        or terminal.get("fit_state_artifact_sha256")
        != _tree_sha256(artifact_root / _FIT_STATE_DIRECTORY)
        or terminal.get("fit_completed_before_registered_heldout_text_access")
        is not True
        or terminal.get("fit_sealed_before_registered_heldout_text_access")
        is not True
        or terminal.get("cumulative_views_published_without_sealed_text")
        is not True
        or terminal.get("model_state_reloaded_for_primary_transform") is not True
        or terminal.get("registered_heldout_labels_accessed") is not False
        or terminal.get("pickle_or_joblib_loaded") is not False
        or terminal.get("text_truncation_applied") is not False
    ):
        raise ValueError("HTR terminal execution envelope changed")
    seal_registration = terminal.get("fit_only_family_seal")
    seal_sha256, seal_size = _sha256_file(
        artifact_root / _FIT_SEAL_FILE,
        label="HTR fit-only family seal",
    )
    if (
        not isinstance(seal_registration, Mapping)
        or seal_registration
        != {
            "relative_path": _FIT_SEAL_FILE,
            "sha256": seal_sha256,
            "size_bytes": seal_size,
            "content_sha256": seal["content_sha256"],
        }
    ):
        raise ValueError("HTR fit-only seal registration changed")
    events = terminal.get("event_order")
    if not isinstance(events, list) or [
        int(event.get("sequence", 0)) for event in events
    ] != list(range(1, len(events) + 1)):
        raise ValueError("HTR event order is incomplete")
    first_text_access = next(
        (
            index
            for index, event in enumerate(events)
            if event.get("registered_heldout_text_accessed") is True
        ),
        None,
    )
    expected_cumulative = len(request.logical_members) - 1
    if (
        first_text_access != 2 + expected_cumulative
        or events[first_text_access].get("event") != "exact_heldout_text_opened"
        or any(
            event.get("registered_heldout_text_accessed") is not False
            or event.get("registered_heldout_labels_accessed") is not False
            for event in events[:first_text_access]
        )
    ):
        raise ValueError("HTR held-out text access preceded fit/cumulative seals")
    registrations = terminal.get("logical_views")
    if (
        not isinstance(registrations, list)
        or len(registrations) != len(request.logical_members)
        or len(
            {
                str(registration.get("logical_scope_id"))
                for registration in registrations
            }
        )
        != len(registrations)
    ):
        raise ValueError("HTR logical-view registration coverage changed")
    expected_files = {
        f"{_FIT_STATE_DIRECTORY}/{_FIT_STATE_METADATA}",
        _FIT_SEAL_FILE,
        _TERMINAL_FILE,
    }
    expected_files.update(
        f"{_FIT_STATE_DIRECTORY}/{row['relative_path']}"
        for row in metadata["array_inventory"].values()
    )
    for member in request.logical_members:
        path = (
            artifact_root
            / _LOGICAL_VIEW_DIRECTORY
            / f"{member.scope_id}.json"
        )
        view = _read_json(path, label=f"HTR logical view {member.scope_id}")
        body = {
            key: value for key, value in view.items() if key != "content_sha256"
        }
        if (
            view.get("schema_version") != ROLE_NEUTRAL_HTR_LOGICAL_VIEW_SCHEMA
            or view.get("content_sha256") != _sha256_json(body)
            or view.get("group_request_content_sha256") != request.content_sha256
            or view.get("logical_scope_id") != member.scope_id
            or view.get("logical_scope_sha256")
            != member.as_dict()["scope_sha256"]
            or view.get("logical_purpose") != member.scope_kind
            or view.get("physical_owner_scope_id")
            != request.physical_owner.scope_id
            or view.get("family") != HTR_NEURAL
            or view.get("fit_only_family_seal_sha256") != seal_sha256
            or view.get("fit_only_family_seal_content_sha256")
            != seal["content_sha256"]
            or view.get("logical_heldout_row_ids")
            != list(member.heldout_row_ids)
            or view.get("registered_heldout_labels_accessed") is not False
        ):
            raise ValueError(f"HTR logical view changed: {member.scope_id}")
        relative = path.relative_to(artifact_root).as_posix()
        expected_files.add(relative)
        registration = next(
            row
            for row in registrations
            if row["logical_scope_id"] == member.scope_id
        )
        digest, size = _sha256_file(path, label=f"HTR view {member.scope_id}")
        if registration != {
            "logical_scope_id": member.scope_id,
            "relative_path": relative,
            "sha256": digest,
            "size_bytes": size,
            "content_sha256": view["content_sha256"],
        }:
            raise ValueError("HTR logical-view registration changed")
        if member.scope_id != request.physical_owner.scope_id:
            if (
                view.get("view_input_policy")
                != "sealed_row_ids_only_no_sealed_text_or_labels_v1"
                or view.get("logical_transform_performed") is not False
                or view.get("prediction_artifact") is not None
                or view.get("coverage_artifacts") is not None
                or view.get("registered_heldout_text_accessed") is not False
                or "logical_heldout_text_sha256" in view
            ):
                raise ValueError("cumulative HTR view accessed sealed text")
            continue
        prediction = view.get("prediction_artifact")
        if not isinstance(prediction, Mapping):
            raise ValueError("exact HTR view lacks predictions")
        prediction_relative = str(prediction.get("relative_path") or "")
        prediction_path = artifact_root / prediction_relative
        prediction_digest, prediction_size, prediction_array = (
            _read_npy_file_once(
                prediction_path,
                label="exact HTR predictions",
                invalid_message=(
                    "exact HTR predictions are not safe NumPy data"
                ),
            )
        )
        if (
            view.get("view_input_policy")
            != "heldout_row_id_and_complete_text_no_labels_v1"
            or view.get("logical_transform_performed") is not True
            or view.get("registered_heldout_text_accessed") is not True
            or view.get("model_state_reloaded_for_primary_transform") is not True
            or prediction_digest != prediction.get("sha256")
            or prediction_size != int(prediction.get("size_bytes", -1))
            or prediction_array.dtype.str != prediction.get("dtype")
            or list(prediction_array.shape) != prediction.get("shape")
            or _array_sha256(prediction_array) != prediction.get("content_sha256")
        ):
            raise ValueError("exact HTR prediction artifact changed")
        expected_files.add(prediction_relative)
        coverage_artifacts = view.get("coverage_artifacts")
        coverage_proof = view.get("coverage_proof")
        if (
            not isinstance(coverage_artifacts, Mapping)
            or not isinstance(coverage_proof, Mapping)
            or coverage_proof.get("schema_version")
            != ROLE_NEUTRAL_HTR_COVERAGE_SCHEMA
            or coverage_proof.get("max_chunks_nonbinding") is not True
            or coverage_proof.get("semantic_truncation_applied") is not False
        ):
            raise ValueError("exact HTR coverage proof changed")
        coverage_references = {
            str(value)
            for key, value in coverage_proof.items()
            if key
            in {
                "note_word_counts",
                "note_chunk_counts",
                "chunk_note_positions",
                "chunk_word_starts",
                "chunk_word_ends",
                "chunk_sha256_bytes",
            }
        }
        if coverage_references != set(coverage_artifacts):
            raise ValueError("exact HTR coverage arrays are missing or extra")
        loaded_coverage: dict[str, np.ndarray] = {}
        for key, registered in coverage_artifacts.items():
            if not isinstance(registered, Mapping):
                raise ValueError("exact HTR coverage registration is invalid")
            relative_path = str(registered.get("relative_path") or "")
            array_path = artifact_root / relative_path
            digest, size, array = _read_npy_file_once(
                array_path,
                label=f"exact HTR coverage {key}",
                invalid_message=(
                    f"exact HTR coverage array is not safe NumPy data: {key}"
                ),
            )
            if (
                digest != registered.get("sha256")
                or size != int(registered.get("size_bytes", -1))
                or array.dtype.str != registered.get("dtype")
                or list(array.shape) != registered.get("shape")
                or _array_sha256(array) != registered.get("content_sha256")
            ):
                raise ValueError(f"exact HTR coverage array changed: {key}")
            loaded_coverage[str(key)] = array
            expected_files.add(relative_path)
        _coverage_numeric_values(
            record=coverage_proof,
            arrays=loaded_coverage,
            config=config,
            expected_phase="exact_heldout",
        )
    expected_directories = {
        _FIT_STATE_DIRECTORY,
        f"{_FIT_STATE_DIRECTORY}/arrays",
        _LOGICAL_VIEW_DIRECTORY,
    }
    if files != expected_files or directories != expected_directories:
        raise ValueError(
            "HTR artifact inventory changed; "
            f"missing={sorted(expected_files - files)}, "
            f"extra={sorted(files - expected_files)}"
        )
    # Ensure fit-state numerical arrays were authenticated by this trust boundary.
    if not arrays:
        raise RuntimeError("HTR validation did not authenticate numerical state")
    return json.loads(_canonical_json(terminal))


__all__ = [
    "ROLE_NEUTRAL_HTR_CONFIG_SCHEMA",
    "ROLE_NEUTRAL_HTR_FIT_STATE_SCHEMA",
    "ROLE_NEUTRAL_HTR_GROUP_EXECUTION_SCHEMA",
    "ROLE_NEUTRAL_HTR_GROUP_REQUEST_SCHEMA",
    "ROLE_NEUTRAL_HTR_LOGICAL_VIEW_SCHEMA",
    "ROLE_NEUTRAL_HTR_OPERATIONAL_CONTROLS_SCHEMA",
    "RoleNeutralHTRConfig",
    "RoleNeutralHTROperationalControls",
    "RoleNeutralHTRPhysicalGroupRequest",
    "execute_role_neutral_htr_physical_group",
    "replay_role_neutral_htr_exact_transform",
    "validate_role_neutral_htr_group_execution",
]
