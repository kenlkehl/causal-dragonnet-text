"""Agentic, fold-local evidence and role selection for plain Stage 2.

This module deliberately separates four concerns:

* deterministic evidence computed only from an outer fold's inner partitions;
* unsupervised, mixed-type variable clustering;
* bounded structured-latent construction and evaluation; and
* audited agent decisions for confounder and effect-modifier roles.

The agent never receives the outer-heldout rows.  Statistical tools may inspect
row-level values (including configured identifiers) inside the spent inner
partitions, while treatment and outcome labels remain available only through
typed role-evaluation operations.  This keeps latent construction unsupervised
without imposing a privacy redaction policy on trusted endpoints.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import logging
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from joblib.externals.loky import ProcessPoolExecutor
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error

from .stage2_statistical_selection import (
    _binary_nested_p_value,
    _continuous_nested_p_value,
    _design_for_features,
    _encode_feature,
    _feature_key,
    _feature_strategy,
    _modifier_test_chunk,
    _rank_safe_columns,
)

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "stage2_agentic_role_selection_v1"
EVIDENCE_SCHEMA_VERSION = "stage2_inner_fold_evidence_v2_loky_chunks"
LATENT_SCHEMA_VERSION = "stage2_structured_latent_v1"
TOOL_PROTOCOL_VERSION = "stage2_agentic_selection_tools_v1"
TEMPORAL_SCOPE = "pre_index_treatment"
PAIRWISE_CONTEXT_SCHEMA_VERSION = "stage2_pairwise_encoded_context_v1"
PAIRWISE_CHUNK_SCHEMA_VERSION = "stage2_pairwise_chunk_checkpoint_v2_content_fingerprint"
DEFAULT_PAIRWISE_CHUNK_SIZE = 512


class RequestJSON(Protocol):
    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        validate: Callable[[Mapping[str, Any]], dict[str, Any]],
        *,
        request_kind: str = "interpretation",
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Stage2AgenticSelectionConfig:
    """Deterministic policy for the agentic Stage 2 selector."""

    minimum_pairwise_complete_rows: int = 10
    categorical_rare_level_min_count: int = 5
    missingness_weight: float = 0.15
    cluster_similarity_threshold: float = 0.60
    cluster_consensus_fraction: float = 0.60
    cluster_max_size: int = 12
    max_latents_per_cluster: int = 2
    latent_min_coverage: float = 0.05
    cluster_tool_call_limit: int = 8
    adjudicator_tool_call_limit: int = 10
    row_query_page_size: int = 200

    def validate(self) -> None:
        for name in (
            "minimum_pairwise_complete_rows",
            "categorical_rare_level_min_count",
            "cluster_max_size",
            "max_latents_per_cluster",
            "cluster_tool_call_limit",
            "adjudicator_tool_call_limit",
            "row_query_page_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"stage2.agentic_selection.{name} must be a positive integer")
        if self.cluster_max_size < 2:
            raise ValueError("stage2.agentic_selection.cluster_max_size must be at least 2")
        if self.max_latents_per_cluster > self.cluster_max_size:
            raise ValueError(
                "stage2.agentic_selection.max_latents_per_cluster cannot exceed cluster_max_size"
            )
        for name in (
            "missingness_weight",
            "cluster_similarity_threshold",
            "cluster_consensus_fraction",
            "latent_min_coverage",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"stage2.agentic_selection.{name} must be in [0, 1]")
        if self.cluster_similarity_threshold <= 0.0:
            raise ValueError(
                "stage2.agentic_selection.cluster_similarity_threshold must be positive"
            )
        if self.cluster_consensus_fraction <= 0.0:
            raise ValueError(
                "stage2.agentic_selection.cluster_consensus_fraction must be positive"
            )

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def agentic_selection_config_from_mapping(
    value: Mapping[str, Any] | None,
) -> Stage2AgenticSelectionConfig:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("stage2.agentic_selection must be an object")
    raw = dict(value or {})
    known = set(Stage2AgenticSelectionConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            "stage2.agentic_selection contains unsupported fields: " f"{unknown}"
        )
    config = Stage2AgenticSelectionConfig(**raw)
    config.validate()
    return config


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _atomic_write_text(path: Path, value: str) -> None:
    """Atomically publish one checkpoint produced by a thread or loky worker."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, default=_json_default) + "\n")


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + "\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint ordered values without serializing a wide matrix to JSON."""

    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "columns": list(map(str, frame.columns)),
                "dtypes": [str(dtype) for dtype in frame.dtypes],
            }
        ).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _feature_by_id(
    definitions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    names: set[str] = set()
    for feature in definitions:
        key = _feature_key(feature)
        if key in result:
            raise ValueError(f"duplicate Stage 2 feature key {key!r}")
        name = str(feature["name"])
        if name in names:
            raise ValueError(f"duplicate Stage 2 feature name {name!r}")
        names.add(name)
        result[key] = dict(feature)
    return result


def _benjamini_hochberg(values: Sequence[float | None]) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    valid = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    if not valid:
        return result
    ordered = sorted(valid, key=lambda item: item[1])
    count = len(ordered)
    running = 1.0
    for reverse_position, (index, p_value) in enumerate(reversed(ordered), start=1):
        rank = count - reverse_position + 1
        running = min(running, p_value * count / rank)
        result[index] = float(np.clip(running, 0.0, 1.0))
    return result


def _missing_mask(series: pd.Series) -> np.ndarray:
    return series.isna().to_numpy(dtype=bool)


def _categorical_values(
    series: pd.Series,
    *,
    rare_minimum: int,
) -> tuple[pd.Series, dict[str, str]]:
    text = series.astype(str)
    counts = text.value_counts(dropna=False)
    rare = {str(value) for value, count in counts.items() if int(count) < rare_minimum}
    mapping = {value: "__OTHER__" for value in sorted(rare)}
    collapsed = text.map(lambda value: mapping.get(str(value), str(value)))
    return collapsed, mapping


def _bias_corrected_cramers_v(table: np.ndarray) -> float | None:
    table = np.asarray(table, dtype=float)
    n = float(table.sum())
    if n <= 1.0 or table.ndim != 2 or min(table.shape) < 2:
        return None
    try:
        chi2 = float(stats.chi2_contingency(table, correction=False)[0])
    except ValueError:
        return None
    rows, columns = table.shape
    phi2 = chi2 / n
    correction = ((columns - 1) * (rows - 1)) / max(1.0, n - 1.0)
    corrected_phi2 = max(0.0, phi2 - correction)
    corrected_rows = rows - ((rows - 1) ** 2) / max(1.0, n - 1.0)
    corrected_columns = columns - ((columns - 1) ** 2) / max(1.0, n - 1.0)
    denominator = min(corrected_rows - 1.0, corrected_columns - 1.0)
    if denominator <= 0.0:
        return None
    return float(np.clip(math.sqrt(corrected_phi2 / denominator), 0.0, 1.0))


def _correlation_ratio(continuous: np.ndarray, categorical: np.ndarray) -> float | None:
    continuous = np.asarray(continuous, dtype=float)
    categorical = np.asarray(categorical, dtype=str)
    if len(continuous) < 3 or np.std(continuous) <= 0.0:
        return None
    grand = float(np.mean(continuous))
    denominator = float(np.sum(np.square(continuous - grand)))
    if denominator <= 0.0:
        return None
    numerator = 0.0
    for level in np.unique(categorical):
        group = continuous[categorical == level]
        if len(group):
            numerator += len(group) * float(np.mean(group) - grand) ** 2
    return float(np.clip(math.sqrt(max(0.0, numerator / denominator)), 0.0, 1.0))


def _contingency_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"left_level": str(left), "right_level": str(right), "count": int(count)}
        for left in table.index
        for right in table.columns
        if (count := table.loc[left, right]) != 0
    ]


def _pairwise_evidence(
    frame: pd.DataFrame,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    policy: Stage2AgenticSelectionConfig,
) -> dict[str, Any]:
    left_name, right_name = str(left["name"]), str(right["name"])
    left_values = frame[left_name] if left_name in frame else pd.Series([None] * len(frame))
    right_values = frame[right_name] if right_name in frame else pd.Series([None] * len(frame))
    left_missing = _missing_mask(left_values)
    right_missing = _missing_mask(right_values)
    complete = ~(left_missing | right_missing)
    complete_count = int(complete.sum())

    missing_table = pd.crosstab(
        pd.Series(left_missing, name="left_missing"),
        pd.Series(right_missing, name="right_missing"),
        dropna=False,
    ).reindex(index=[False, True], columns=[False, True], fill_value=0)
    missing_phi = None
    if np.std(left_missing.astype(float)) > 0 and np.std(right_missing.astype(float)) > 0:
        missing_phi = _finite(
            np.corrcoef(left_missing.astype(float), right_missing.astype(float))[0, 1]
        )
    missing_union = int(np.sum(left_missing | right_missing))
    missing_jaccard = (
        None
        if missing_union == 0
        else float(np.sum(left_missing & right_missing) / missing_union)
    )

    row: dict[str, Any] = {
        "left_feature_id": _feature_key(left),
        "left_name": left_name,
        "right_feature_id": _feature_key(right),
        "right_name": right_name,
        "n_rows": int(len(frame)),
        "n_pairwise_complete": complete_count,
        "missingness": {
            "table": _contingency_records(missing_table),
            "phi": missing_phi,
            "absolute_phi": None if missing_phi is None else abs(missing_phi),
            "agreement": float(np.mean(left_missing == right_missing)),
            "jaccard": missing_jaccard,
        },
        "evaluable": complete_count >= policy.minimum_pairwise_complete_rows,
        "association_kind": None,
        "association": None,
        "p_value": None,
        "details": {},
    }
    if not row["evaluable"]:
        row["details"] = {"reason": "insufficient_pairwise_complete_rows"}
        return row

    left_strategy = _feature_strategy(left)
    right_strategy = _feature_strategy(right)
    left_continuous = left_strategy in {"continuous", "continuous_with_categorical_fallback"}
    right_continuous = right_strategy in {"continuous", "continuous_with_categorical_fallback"}

    if left_continuous and right_continuous:
        x = pd.to_numeric(left_values[complete], errors="coerce")
        y = pd.to_numeric(right_values[complete], errors="coerce")
        valid = x.notna() & y.notna()
        if int(valid.sum()) < policy.minimum_pairwise_complete_rows:
            row["evaluable"] = False
            row["details"] = {"reason": "insufficient_numeric_pairwise_rows"}
            return row
        valid_x = x[valid]
        valid_y = y[valid]
        if np.unique(valid_x).size < 2 or np.unique(valid_y).size < 2:
            rho = None
            p_value = None
        else:
            result = stats.spearmanr(valid_x, valid_y)
            rho = _finite(result.statistic)
            p_value = _finite(result.pvalue)
        row.update(
            {
                "association_kind": "absolute_spearman",
                "association": None if rho is None else abs(rho),
                "signed_association": rho,
                "p_value": p_value,
                "details": {"numeric_pairwise_rows": int(valid.sum())},
            }
        )
        return row

    if not left_continuous and not right_continuous:
        raw_left = left_values[complete].astype(str)
        raw_right = right_values[complete].astype(str)
        raw_table = pd.crosstab(raw_left, raw_right, dropna=False)
        collapsed_left, left_mapping = _categorical_values(
            raw_left,
            rare_minimum=policy.categorical_rare_level_min_count,
        )
        collapsed_right, right_mapping = _categorical_values(
            raw_right,
            rare_minimum=policy.categorical_rare_level_min_count,
        )
        inferential = pd.crosstab(collapsed_left, collapsed_right, dropna=False)
        details: dict[str, Any] = {
            "raw_table": _contingency_records(raw_table),
            "inferential_table": _contingency_records(inferential),
            "raw_shape": list(map(int, raw_table.shape)),
            "inferential_shape": list(map(int, inferential.shape)),
            "left_rare_level_mapping": left_mapping,
            "right_rare_level_mapping": right_mapping,
        }
        if min(inferential.shape) < 2:
            row["evaluable"] = False
            details["reason"] = "contingency_has_one_level"
            row["details"] = details
            return row
        try:
            chi2, p_value, degrees, expected = stats.chi2_contingency(
                inferential.to_numpy(), correction=False
            )
        except ValueError as exc:
            row["evaluable"] = False
            details["reason"] = f"chi_square_error: {exc}"
            row["details"] = details
            return row
        details.update(
            {
                "chi_square": float(chi2),
                "degrees_of_freedom": int(degrees),
                "minimum_expected_count": float(np.min(expected)),
                "expected_cells_below_five": int(np.sum(expected < 5.0)),
            }
        )
        association = _bias_corrected_cramers_v(inferential.to_numpy())
        signed_phi = None
        if inferential.shape == (2, 2):
            denominator = math.sqrt(
                float(inferential.sum(axis=1).prod() * inferential.sum(axis=0).prod())
            )
            if denominator > 0:
                values = inferential.to_numpy(dtype=float)
                signed_phi = float((values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0]) / denominator)
        row.update(
            {
                "association_kind": "bias_corrected_cramers_v",
                "association": association,
                "signed_association": signed_phi,
                "p_value": float(p_value),
                "details": details,
            }
        )
        return row

    continuous_values = left_values if left_continuous else right_values
    categorical_values = right_values if left_continuous else left_values
    numeric = pd.to_numeric(continuous_values[complete], errors="coerce")
    categorical = categorical_values[complete].astype(str)
    valid = numeric.notna()
    numeric = numeric[valid]
    categorical = categorical[valid]
    details = {"numeric_pairwise_rows": int(valid.sum())}
    if int(valid.sum()) < policy.minimum_pairwise_complete_rows or categorical.nunique() < 2:
        row["evaluable"] = False
        details["reason"] = "insufficient_mixed_pair_variation"
        row["details"] = details
        return row
    groups = [
        numeric[categorical == level].to_numpy(dtype=float)
        for level in sorted(categorical.unique())
        if int(np.sum(categorical == level)) > 0
    ]
    if np.unique(numeric).size < 2:
        p_value = None
        statistic = None
    else:
        try:
            test = stats.kruskal(*groups)
            p_value = _finite(test.pvalue)
            statistic = _finite(test.statistic)
        except ValueError:
            p_value = None
            statistic = None
    details.update(
        {
            "kruskal_wallis": statistic,
            "levels": [str(value) for value in sorted(categorical.unique())],
            "level_counts": {
                str(level): int(np.sum(categorical == level))
                for level in sorted(categorical.unique())
            },
        }
    )
    row.update(
        {
            "association_kind": "correlation_ratio",
            "association": _correlation_ratio(
                numeric.to_numpy(dtype=float), categorical.to_numpy(dtype=str)
            ),
            "p_value": p_value,
            "details": details,
        }
    )
    return row


@dataclass(frozen=True)
class _EncodedPairwiseContext:
    """Read-only, mmap-backed values shared by loky pair-chunk workers."""

    fingerprint: str
    features: tuple[dict[str, Any], ...]
    text_levels: tuple[tuple[str, ...], ...]
    missing: np.ndarray
    numeric: np.ndarray
    text_codes: np.ndarray


_PAIRWISE_CONTEXT_CACHE: dict[str, _EncodedPairwiseContext] = {}


def _atomic_write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def _encode_pairwise_context(
    *,
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    output_dir: Path,
    input_fingerprint: str,
) -> dict[str, Any]:
    """Encode mixed columns once and publish safe mmap-compatible worker inputs."""

    rows = len(frame)
    feature_count = len(definitions)
    missing = np.zeros((feature_count, rows), dtype=bool)
    numeric = np.full((feature_count, rows), np.nan, dtype=np.float64)
    text_codes = np.full((feature_count, rows), -1, dtype=np.int32)
    text_levels: list[list[str]] = []
    feature_metadata: list[dict[str, Any]] = []
    for index, definition in enumerate(definitions):
        name = str(definition["name"])
        series = (
            frame[name].reset_index(drop=True)
            if name in frame
            else pd.Series([None] * rows, dtype=object)
        )
        feature_missing = _missing_mask(series)
        missing[index] = feature_missing
        numeric[index] = pd.to_numeric(series, errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        observed_positions = np.flatnonzero(~feature_missing)
        observed_text = series.iloc[observed_positions].astype(str).tolist()
        levels = sorted(set(observed_text))
        level_positions = {value: position for position, value in enumerate(levels)}
        if observed_positions.size:
            text_codes[index, observed_positions] = np.asarray(
                [level_positions[value] for value in observed_text],
                dtype=np.int32,
            )
        text_levels.append(levels)
        feature_metadata.append(
            {
                "feature_id": _feature_key(definition),
                "name": name,
                "strategy": _feature_strategy(definition),
            }
        )

    context_fingerprint = _fingerprint(
        {
            "schema_version": PAIRWISE_CONTEXT_SCHEMA_VERSION,
            "input_fingerprint": input_fingerprint,
            "features": feature_metadata,
            "text_levels": text_levels,
            "rows": rows,
        }
    )
    metadata = {
        "schema_version": PAIRWISE_CONTEXT_SCHEMA_VERSION,
        "input_fingerprint": input_fingerprint,
        "context_fingerprint": context_fingerprint,
        "rows": rows,
        "features": feature_metadata,
        "text_levels": text_levels,
        "array_shapes": {
            "missing": list(missing.shape),
            "numeric": list(numeric.shape),
            "text_codes": list(text_codes.shape),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_npy(output_dir / "missing.npy", missing)
    _atomic_write_npy(output_dir / "numeric.npy", numeric)
    _atomic_write_npy(output_dir / "text_codes.npy", text_codes)
    # Publish metadata last. A worker never accepts a partially written context.
    _atomic_write_json(output_dir / "context.json", metadata)
    return metadata


def _load_pairwise_context(
    context_dir: str | Path,
    *,
    expected_fingerprint: str,
) -> _EncodedPairwiseContext:
    directory = Path(context_dir)
    cache_key = f"{directory.resolve()}::{expected_fingerprint}"
    cached = _PAIRWISE_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    metadata = json.loads((directory / "context.json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != PAIRWISE_CONTEXT_SCHEMA_VERSION
        or metadata.get("context_fingerprint") != expected_fingerprint
    ):
        raise RuntimeError(f"incompatible Stage 2 pairwise context under {directory}")
    context = _EncodedPairwiseContext(
        fingerprint=expected_fingerprint,
        features=tuple(dict(value) for value in metadata["features"]),
        text_levels=tuple(tuple(map(str, values)) for values in metadata["text_levels"]),
        missing=np.load(directory / "missing.npy", mmap_mode="r", allow_pickle=False),
        numeric=np.load(directory / "numeric.npy", mmap_mode="r", allow_pickle=False),
        text_codes=np.load(directory / "text_codes.npy", mmap_mode="r", allow_pickle=False),
    )
    expected_shape = (len(context.features), int(metadata["rows"]))
    if any(
        tuple(values.shape) != expected_shape
        for values in (context.missing, context.numeric, context.text_codes)
    ):
        raise RuntimeError(f"invalid Stage 2 pairwise context arrays under {directory}")
    _PAIRWISE_CONTEXT_CACHE[cache_key] = context
    return context


def _encoded_text_values(
    context: _EncodedPairwiseContext,
    feature_index: int,
    row_mask: np.ndarray,
) -> pd.Series:
    codes = np.asarray(context.text_codes[feature_index, row_mask], dtype=np.int32)
    if np.any(codes < 0):
        raise RuntimeError("encoded pairwise text requested for missing rows")
    levels = np.asarray(context.text_levels[feature_index], dtype=object)
    return pd.Series(levels[codes], dtype=object)


def _encoded_pairwise_evidence(
    context: _EncodedPairwiseContext,
    left_index: int,
    right_index: int,
    *,
    policy: Stage2AgenticSelectionConfig,
) -> dict[str, Any]:
    """Evaluate one pair from pre-encoded values with legacy-identical semantics."""

    left = context.features[left_index]
    right = context.features[right_index]
    left_missing = np.asarray(context.missing[left_index], dtype=bool)
    right_missing = np.asarray(context.missing[right_index], dtype=bool)
    complete = ~(left_missing | right_missing)
    complete_count = int(complete.sum())
    missing_table = pd.crosstab(
        pd.Series(left_missing, name="left_missing"),
        pd.Series(right_missing, name="right_missing"),
        dropna=False,
    ).reindex(index=[False, True], columns=[False, True], fill_value=0)
    missing_phi = None
    if np.std(left_missing.astype(float)) > 0 and np.std(right_missing.astype(float)) > 0:
        missing_phi = _finite(
            np.corrcoef(left_missing.astype(float), right_missing.astype(float))[0, 1]
        )
    missing_union = int(np.sum(left_missing | right_missing))
    missing_jaccard = (
        None
        if missing_union == 0
        else float(np.sum(left_missing & right_missing) / missing_union)
    )
    row: dict[str, Any] = {
        "left_feature_id": str(left["feature_id"]),
        "left_name": str(left["name"]),
        "right_feature_id": str(right["feature_id"]),
        "right_name": str(right["name"]),
        "n_rows": int(context.missing.shape[1]),
        "n_pairwise_complete": complete_count,
        "missingness": {
            "table": _contingency_records(missing_table),
            "phi": missing_phi,
            "absolute_phi": None if missing_phi is None else abs(missing_phi),
            "agreement": float(np.mean(left_missing == right_missing)),
            "jaccard": missing_jaccard,
        },
        "evaluable": complete_count >= policy.minimum_pairwise_complete_rows,
        "association_kind": None,
        "association": None,
        "p_value": None,
        "details": {},
    }
    if not row["evaluable"]:
        row["details"] = {"reason": "insufficient_pairwise_complete_rows"}
        return row

    left_continuous = str(left["strategy"]) in {
        "continuous",
        "continuous_with_categorical_fallback",
    }
    right_continuous = str(right["strategy"]) in {
        "continuous",
        "continuous_with_categorical_fallback",
    }
    if left_continuous and right_continuous:
        x = np.asarray(context.numeric[left_index, complete], dtype=float)
        y = np.asarray(context.numeric[right_index, complete], dtype=float)
        valid = pd.notna(x) & pd.notna(y)
        if int(valid.sum()) < policy.minimum_pairwise_complete_rows:
            row["evaluable"] = False
            row["details"] = {"reason": "insufficient_numeric_pairwise_rows"}
            return row
        valid_x = x[valid]
        valid_y = y[valid]
        if np.unique(valid_x).size < 2 or np.unique(valid_y).size < 2:
            rho = None
            p_value = None
        else:
            result = stats.spearmanr(valid_x, valid_y)
            rho = _finite(result.statistic)
            p_value = _finite(result.pvalue)
        row.update(
            {
                "association_kind": "absolute_spearman",
                "association": None if rho is None else abs(rho),
                "signed_association": rho,
                "p_value": p_value,
                "details": {"numeric_pairwise_rows": int(valid.sum())},
            }
        )
        return row

    if not left_continuous and not right_continuous:
        raw_left = _encoded_text_values(context, left_index, complete)
        raw_right = _encoded_text_values(context, right_index, complete)
        raw_table = pd.crosstab(raw_left, raw_right, dropna=False)
        collapsed_left, left_mapping = _categorical_values(
            raw_left,
            rare_minimum=policy.categorical_rare_level_min_count,
        )
        collapsed_right, right_mapping = _categorical_values(
            raw_right,
            rare_minimum=policy.categorical_rare_level_min_count,
        )
        inferential = pd.crosstab(collapsed_left, collapsed_right, dropna=False)
        details: dict[str, Any] = {
            "raw_table": _contingency_records(raw_table),
            "inferential_table": _contingency_records(inferential),
            "raw_shape": list(map(int, raw_table.shape)),
            "inferential_shape": list(map(int, inferential.shape)),
            "left_rare_level_mapping": left_mapping,
            "right_rare_level_mapping": right_mapping,
        }
        if min(inferential.shape) < 2:
            row["evaluable"] = False
            details["reason"] = "contingency_has_one_level"
            row["details"] = details
            return row
        try:
            chi2, p_value, degrees, expected = stats.chi2_contingency(
                inferential.to_numpy(), correction=False
            )
        except ValueError as exc:
            row["evaluable"] = False
            details["reason"] = f"chi_square_error: {exc}"
            row["details"] = details
            return row
        details.update(
            {
                "chi_square": float(chi2),
                "degrees_of_freedom": int(degrees),
                "minimum_expected_count": float(np.min(expected)),
                "expected_cells_below_five": int(np.sum(expected < 5.0)),
            }
        )
        association = _bias_corrected_cramers_v(inferential.to_numpy())
        signed_phi = None
        if inferential.shape == (2, 2):
            denominator = math.sqrt(
                float(inferential.sum(axis=1).prod() * inferential.sum(axis=0).prod())
            )
            if denominator > 0:
                values = inferential.to_numpy(dtype=float)
                signed_phi = float(
                    (
                        values[0, 0] * values[1, 1]
                        - values[0, 1] * values[1, 0]
                    )
                    / denominator
                )
        row.update(
            {
                "association_kind": "bias_corrected_cramers_v",
                "association": association,
                "signed_association": signed_phi,
                "p_value": float(p_value),
                "details": details,
            }
        )
        return row

    continuous_index = left_index if left_continuous else right_index
    categorical_index = right_index if left_continuous else left_index
    numeric = np.asarray(context.numeric[continuous_index, complete], dtype=float)
    categorical = _encoded_text_values(context, categorical_index, complete)
    valid = pd.notna(numeric)
    numeric = numeric[valid]
    categorical = categorical[valid].reset_index(drop=True)
    details = {"numeric_pairwise_rows": int(valid.sum())}
    if int(valid.sum()) < policy.minimum_pairwise_complete_rows or categorical.nunique() < 2:
        row["evaluable"] = False
        details["reason"] = "insufficient_mixed_pair_variation"
        row["details"] = details
        return row
    groups = [
        numeric[(categorical == level).to_numpy()]
        for level in sorted(categorical.unique())
        if int(np.sum(categorical == level)) > 0
    ]
    if np.unique(numeric).size < 2:
        p_value = None
        statistic = None
    else:
        try:
            test = stats.kruskal(*groups)
            p_value = _finite(test.pvalue)
            statistic = _finite(test.statistic)
        except ValueError:
            p_value = None
            statistic = None
    details.update(
        {
            "kruskal_wallis": statistic,
            "levels": [str(value) for value in sorted(categorical.unique())],
            "level_counts": {
                str(level): int(np.sum(categorical == level))
                for level in sorted(categorical.unique())
            },
        }
    )
    categorical_values = categorical.to_numpy(dtype=str)
    row.update(
        {
            "association_kind": "correlation_ratio",
            "association": _correlation_ratio(numeric, categorical_values),
            "p_value": p_value,
            "details": details,
        }
    )
    return row


def _pair_specifications(
    definitions: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int, int]]:
    specifications: list[tuple[int, int, int]] = []
    pair_index = 0
    for left_index in range(len(definitions)):
        for right_index in range(left_index + 1, len(definitions)):
            pair_index += 1
            specifications.append((pair_index, left_index, right_index))
    return specifications


def _pairwise_chunk_fingerprint(
    *,
    input_fingerprint: str,
    chunk_index: int,
    pair_specs: Sequence[tuple[int, int, int]],
) -> str:
    return _fingerprint(
        {
            "schema_version": PAIRWISE_CHUNK_SCHEMA_VERSION,
            "input_fingerprint": input_fingerprint,
            "chunk_index": int(chunk_index),
            "pair_specs": [list(map(int, value)) for value in pair_specs],
        }
    )


def _read_pairwise_chunk(
    path: Path,
    *,
    expected_fingerprint: str,
    expected_rows: int,
) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        values = _read_jsonl(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not values:
        return None
    metadata = values[0].get("_checkpoint")
    rows = values[1:]
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != PAIRWISE_CHUNK_SCHEMA_VERSION
        or metadata.get("chunk_fingerprint") != expected_fingerprint
        or metadata.get("rows_fingerprint") != _fingerprint(rows)
        or int(metadata.get("rows", -1)) != expected_rows
        or len(rows) != expected_rows
    ):
        return None
    return rows


def _compute_pairwise_chunk_checkpoint(
    *,
    context_dir: str,
    context_fingerprint: str,
    input_fingerprint: str,
    inner_fold: int,
    chunk_index: int,
    pair_specs: Sequence[tuple[int, int, int]],
    output_path: str,
    policy: Stage2AgenticSelectionConfig,
) -> dict[str, Any]:
    """Loky-safe worker: evaluate and atomically checkpoint one pair chunk."""

    context = _load_pairwise_context(
        context_dir,
        expected_fingerprint=context_fingerprint,
    )
    rows: list[dict[str, Any]] = []
    for pair_index, left_index, right_index in pair_specs:
        row = _encoded_pairwise_evidence(
            context,
            int(left_index),
            int(right_index),
            policy=policy,
        )
        row["evidence_id"] = f"s2e_inner_{int(inner_fold):03d}_pair_{int(pair_index):06d}"
        row["pair_index"] = int(pair_index)
        rows.append(row)
    chunk_fingerprint = _pairwise_chunk_fingerprint(
        input_fingerprint=input_fingerprint,
        chunk_index=chunk_index,
        pair_specs=pair_specs,
    )
    checkpoint = {
        "_checkpoint": {
            "schema_version": PAIRWISE_CHUNK_SCHEMA_VERSION,
            "chunk_fingerprint": chunk_fingerprint,
            "rows_fingerprint": _fingerprint(rows),
            "chunk_index": int(chunk_index),
            "rows": len(rows),
            "first_pair_index": int(pair_specs[0][0]) if pair_specs else None,
            "last_pair_index": int(pair_specs[-1][0]) if pair_specs else None,
        }
    }
    _atomic_write_jsonl(Path(output_path), [checkpoint, *rows])
    return {
        "chunk_index": int(chunk_index),
        "rows": len(rows),
        "output_path": str(output_path),
    }


def _build_pairwise_evidence(
    *,
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    inner_fold: int,
    output_dir: Path,
    input_fingerprint: str,
    policy: Stage2AgenticSelectionConfig,
    workers: int,
    chunk_size: int,
    executor: ProcessPoolExecutor | None,
    pool_scope: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(workers, bool) or int(workers) < 1:
        raise ValueError("Stage 2 evidence workers must be a positive integer")
    if isinstance(chunk_size, bool) or int(chunk_size) < 1:
        raise ValueError("Stage 2 pairwise chunk size must be a positive integer")
    pair_specs = _pair_specifications(definitions)
    chunks = [
        pair_specs[start : start + int(chunk_size)]
        for start in range(0, len(pair_specs), int(chunk_size))
    ]
    work_dir = output_dir / "pairwise_work"
    context_dir = work_dir / "encoded_context"
    context_metadata = _encode_pairwise_context(
        frame=frame,
        definitions=definitions,
        output_dir=context_dir,
        input_fingerprint=input_fingerprint,
    )
    chunk_dir = work_dir / "chunks"
    expected: list[tuple[int, list[tuple[int, int, int]], Path, str]] = []
    reused_chunks = 0
    pending: list[tuple[int, list[tuple[int, int, int]], Path, str]] = []
    for chunk_index, specifications in enumerate(chunks, start=1):
        path = chunk_dir / f"chunk_{chunk_index:06d}.jsonl"
        fingerprint = _pairwise_chunk_fingerprint(
            input_fingerprint=input_fingerprint,
            chunk_index=chunk_index,
            pair_specs=specifications,
        )
        item = (chunk_index, specifications, path, fingerprint)
        expected.append(item)
        if (
            _read_pairwise_chunk(
                path,
                expected_fingerprint=fingerprint,
                expected_rows=len(specifications),
            )
            is None
        ):
            pending.append(item)
        else:
            reused_chunks += 1
    worker_upper_bound = min(max(1, int(workers)), len(pending)) if pending else 0
    effective_workers = (
        None if pool_scope == "shared_stage2_run" else worker_upper_bound
    )
    _atomic_write_json(
        work_dir / "manifest.json",
        {
            "schema_version": PAIRWISE_CHUNK_SCHEMA_VERSION,
            "input_fingerprint": input_fingerprint,
            "context_fingerprint": context_metadata["context_fingerprint"],
            "inner_fold": int(inner_fold),
            "pairs": len(pair_specs),
            "chunks": len(chunks),
            "chunk_size": int(chunk_size),
            "requested_workers": int(workers),
            "effective_workers": effective_workers,
            "worker_upper_bound": worker_upper_bound,
            "pool_scope": pool_scope,
            "reused_chunks_at_start": reused_chunks,
        },
    )
    LOGGER.info(
        "Stage 2 pairwise evidence inner_fold=%s pairs=%s chunks=%s "
        "pending_chunks=%s reused_chunks=%s workers=%s backend=%s",
        inner_fold,
        len(pair_specs),
        len(chunks),
        len(pending),
        reused_chunks,
        worker_upper_bound,
        "loky" if executor is not None else "sequential",
    )

    task_arguments = [
        {
            "context_dir": str(context_dir),
            "context_fingerprint": str(context_metadata["context_fingerprint"]),
            "input_fingerprint": input_fingerprint,
            "inner_fold": int(inner_fold),
            "chunk_index": chunk_index,
            "pair_specs": specifications,
            "output_path": str(path),
            "policy": policy,
        }
        for chunk_index, specifications, path, _fingerprint_value in pending
    ]
    if executor is None:
        for arguments in task_arguments:
            _compute_pairwise_chunk_checkpoint(**arguments)
    else:
        futures = [
            executor.submit(_compute_pairwise_chunk_checkpoint, **arguments)
            for arguments in task_arguments
        ]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    rows: list[dict[str, Any]] = []
    for _chunk_index, specifications, path, fingerprint in expected:
        chunk_rows = _read_pairwise_chunk(
            path,
            expected_fingerprint=fingerprint,
            expected_rows=len(specifications),
        )
        if chunk_rows is None:
            raise RuntimeError(f"missing completed Stage 2 pairwise chunk: {path}")
        rows.extend(chunk_rows)
    rows.sort(key=lambda value: int(value["pair_index"]))
    if [int(row["pair_index"]) for row in rows] != list(range(1, len(pair_specs) + 1)):
        raise RuntimeError("Stage 2 pairwise chunk assembly lost or duplicated pair indices")
    for row in rows:
        row.pop("pair_index", None)
    return rows, {
        "backend": "loky" if executor is not None else "sequential",
        "unit": "pair_chunk",
        "pool_scope": pool_scope,
        "requested_workers": int(workers),
        "effective_workers": effective_workers,
        "worker_upper_bound": worker_upper_bound,
        "chunk_size": int(chunk_size),
        "chunks": len(chunks),
        "reused_chunks": reused_chunks,
        "pairs": len(pair_specs),
    }


def _cluster_from_similarity(
    feature_ids: Sequence[str],
    similarity: np.ndarray,
    *,
    threshold: float,
    max_size: int,
) -> list[list[str]]:
    ids = list(map(str, feature_ids))
    count = len(ids)
    if count == 0:
        return []
    if count == 1:
        return [ids]
    matrix = np.asarray(similarity, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)
    matrix = np.clip((matrix + matrix.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    distance = 1.0 - matrix
    np.fill_diagonal(distance, 0.0)
    linkage = hierarchy.linkage(squareform(distance, checks=False), method="average")
    labels = hierarchy.fcluster(linkage, t=1.0 - float(threshold), criterion="distance")
    order = list(map(int, hierarchy.leaves_list(linkage)))
    groups: dict[int, list[int]] = {}
    for index in order:
        groups.setdefault(int(labels[index]), []).append(index)
    clusters: list[list[str]] = []
    for label in sorted(groups, key=lambda item: min(groups[item])):
        members = groups[label]
        for start in range(0, len(members), int(max_size)):
            clusters.append([ids[index] for index in members[start : start + int(max_size)]])
    return clusters


def _fold_clusters(
    definitions: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    policy: Stage2AgenticSelectionConfig,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    feature_ids = [_feature_key(feature) for feature in definitions]
    positions = {feature_id: index for index, feature_id in enumerate(feature_ids)}
    similarity = np.eye(len(feature_ids), dtype=float)
    for row in pair_rows:
        left = positions[str(row["left_feature_id"])]
        right = positions[str(row["right_feature_id"])]
        value = _finite(row.get("association")) or 0.0
        missing = _finite((row.get("missingness") or {}).get("absolute_phi")) or 0.0
        combined = (
            (1.0 - float(policy.missingness_weight)) * value
            + float(policy.missingness_weight) * missing
        )
        similarity[left, right] = similarity[right, left] = float(np.clip(combined, 0.0, 1.0))
    member_sets = _cluster_from_similarity(
        feature_ids,
        similarity,
        threshold=policy.cluster_similarity_threshold,
        max_size=policy.cluster_max_size,
    )
    clusters = [
        {
            "cluster_id": f"inner_cluster_{index:03d}",
            "member_feature_ids": members,
            "member_names": [
                str(definitions[positions[feature_id]]["name"]) for feature_id in members
            ],
            "n_members": len(members),
        }
        for index, members in enumerate(member_sets, start=1)
    ]
    return clusters, similarity


def _consensus_clusters(
    definitions: Sequence[Mapping[str, Any]],
    fold_clusters: Sequence[Sequence[Mapping[str, Any]]],
    *,
    policy: Stage2AgenticSelectionConfig,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    feature_ids = [_feature_key(feature) for feature in definitions]
    positions = {feature_id: index for index, feature_id in enumerate(feature_ids)}
    coassociation = np.zeros((len(feature_ids), len(feature_ids)), dtype=float)
    for clusters in fold_clusters:
        for cluster in clusters:
            members = [positions[str(value)] for value in cluster["member_feature_ids"]]
            for left in members:
                for right in members:
                    coassociation[left, right] += 1.0
    if fold_clusters:
        coassociation /= len(fold_clusters)
    np.fill_diagonal(coassociation, 1.0)
    member_sets = _cluster_from_similarity(
        feature_ids,
        coassociation,
        threshold=policy.cluster_consensus_fraction,
        max_size=policy.cluster_max_size,
    )
    clusters: list[dict[str, Any]] = []
    for index, members in enumerate(member_sets, start=1):
        pair_recurrence = [
            float(coassociation[positions[left], positions[right]])
            for left_position, left in enumerate(members)
            for right in members[left_position + 1 :]
        ]
        clusters.append(
            {
                "cluster_id": f"consensus_cluster_{index:03d}",
                "member_feature_ids": members,
                "member_names": [
                    str(definitions[positions[feature_id]]["name"])
                    for feature_id in members
                ],
                "n_members": len(members),
                "mean_pair_coclustering": (
                    float(np.mean(pair_recurrence)) if pair_recurrence else 1.0
                ),
                "minimum_pair_coclustering": (
                    float(np.min(pair_recurrence)) if pair_recurrence else 1.0
                ),
            }
        )
    return clusters, coassociation


def _coefficient_summary(
    target: np.ndarray,
    base: np.ndarray,
    additions: np.ndarray,
    names: Sequence[str],
    *,
    binary: bool,
) -> list[dict[str, Any]]:
    full, kept = _rank_safe_columns(base, additions)
    if not kept:
        return []
    try:
        import statsmodels.api as sm

        if binary:
            fit = sm.GLM(target, full, family=sm.families.Binomial()).fit(disp=0)
        else:
            fit = sm.OLS(target, full).fit()
    except Exception as exc:
        return [{"status": "not_evaluable", "reason": f"{type(exc).__name__}: {exc}"}]
    offset = base.shape[1]
    rows: list[dict[str, Any]] = []
    for local_position, source_index in enumerate(kept):
        parameter_index = offset + local_position
        coefficient = _finite(fit.params[parameter_index])
        standard_error = _finite(fit.bse[parameter_index])
        p_value = _finite(fit.pvalues[parameter_index])
        interval = fit.conf_int(alpha=0.05)[parameter_index]
        row = {
            "status": "ok",
            "column": str(names[source_index]),
            "coefficient": coefficient,
            "standard_error": standard_error,
            "p_value": p_value,
            "confidence_interval_95": [_finite(interval[0]), _finite(interval[1])],
        }
        if binary and coefficient is not None:
            row["odds_ratio"] = float(math.exp(np.clip(coefficient, -50.0, 50.0)))
        rows.append(row)
    return rows


def _confounder_univariable_rows(
    frame: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    definitions: Sequence[Mapping[str, Any]],
    *,
    binary_outcome: bool,
) -> list[dict[str, Any]]:
    intercept = np.ones((len(frame), 1), dtype=float)
    treatment_base, _ = _rank_safe_columns(intercept, treatment.reshape(-1, 1))
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(definitions, start=1):
        design = _encode_feature(frame, feature)
        treatment_p, treatment_test = _binary_nested_p_value(
            treatment,
            intercept,
            design.main,
        )
        if binary_outcome:
            outcome_p, outcome_test = _binary_nested_p_value(
                outcome,
                intercept,
                design.main,
            )
            adjusted_p, adjusted_test = _binary_nested_p_value(
                outcome,
                treatment_base,
                design.main,
            )
        else:
            outcome_p, outcome_test = _continuous_nested_p_value(
                outcome,
                intercept,
                design.main,
            )
            adjusted_p, adjusted_test = _continuous_nested_p_value(
                outcome,
                treatment_base,
                design.main,
            )
        rows.append(
            {
                "evidence_id": f"confounder_univariable_{index:04d}",
                "feature_id": _feature_key(feature),
                "name": str(feature["name"]),
                "treatment_p_value": treatment_p,
                "outcome_p_value": outcome_p,
                "outcome_adjusted_for_treatment_p_value": adjusted_p,
                "treatment_test": treatment_test,
                "outcome_test": outcome_test,
                "outcome_adjusted_for_treatment_test": adjusted_test,
                "treatment_coefficients": _coefficient_summary(
                    treatment,
                    intercept,
                    design.main,
                    design.main_names,
                    binary=True,
                ),
                "outcome_coefficients": _coefficient_summary(
                    outcome,
                    intercept,
                    design.main,
                    design.main_names,
                    binary=binary_outcome,
                ),
                "outcome_adjusted_for_treatment_coefficients": _coefficient_summary(
                    outcome,
                    treatment_base,
                    design.main,
                    design.main_names,
                    binary=binary_outcome,
                ),
            }
        )
    for key in (
        "treatment_p_value",
        "outcome_p_value",
        "outcome_adjusted_for_treatment_p_value",
    ):
        adjusted = _benjamini_hochberg([row.get(key) for row in rows])
        q_key = key.replace("p_value", "q_value")
        for row, q_value in zip(rows, adjusted):
            row[q_key] = q_value
    return rows


def _feature_value_summaries(
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(definitions, start=1):
        name = str(feature["name"])
        series = frame[name] if name in frame else pd.Series([None] * len(frame))
        observed = series.dropna()
        row: dict[str, Any] = {
            "evidence_id": f"feature_summary_{index:04d}",
            "feature_id": _feature_key(feature),
            "name": name,
            "rows": int(len(series)),
            "observed_rows": int(series.notna().sum()),
            "coverage": float(series.notna().mean()) if len(series) else 0.0,
            "unique_observed": int(observed.nunique(dropna=True)),
            "top_values": {
                str(key): int(value)
                for key, value in observed.astype(str).value_counts().head(12).items()
            },
        }
        if _feature_strategy(feature) in {
            "continuous",
            "continuous_with_categorical_fallback",
        }:
            numeric = pd.to_numeric(observed, errors="coerce").dropna()
            if len(numeric):
                row["numeric_summary"] = {
                    "count": int(len(numeric)),
                    "mean": float(numeric.mean()),
                    "standard_deviation": float(numeric.std(ddof=0)),
                    "minimum": float(numeric.min()),
                    "quantiles": {
                        str(q): float(numeric.quantile(q))
                        for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
                    },
                    "maximum": float(numeric.max()),
                }
        rows.append(row)
    return rows


def _stage2_evidence_fingerprints(
    *,
    dataset: pd.DataFrame,
    extracted_fit: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    inner_splits: Sequence[Mapping[str, Any]],
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    policy: Stage2AgenticSelectionConfig,
) -> tuple[str, str]:
    """Return scientific base/root fingerprints, excluding execution tuning."""

    outer_fit_row_ids = extracted_fit["_oci_row_id"].to_numpy(dtype=int)
    base_fingerprint = _fingerprint(
        {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "definitions": list(definitions),
            "policy": policy.public_dict(),
            "temporal_scope": TEMPORAL_SCOPE,
            "treatment_column": str(treatment_column),
            "outcome_column": str(outcome_column),
            "outcome_type": str(outcome_type),
            "extracted_fit_fingerprint": _frame_fingerprint(extracted_fit),
            "target_fingerprint": _frame_fingerprint(
                dataset.iloc[outer_fit_row_ids][
                    [treatment_column, outcome_column]
                ].reset_index(drop=True)
            ),
        }
    )
    root_fingerprint = _fingerprint(
        {
            "base_fingerprint": base_fingerprint,
            "inner_splits": list(inner_splits),
        }
    )
    return base_fingerprint, root_fingerprint


def _inner_evidence_fingerprint(
    *,
    base_fingerprint: str,
    inner_fold: int,
    train_ids: Sequence[int],
    heldout_ids: Sequence[int],
) -> str:
    return _fingerprint(
        {
            "base_fingerprint": base_fingerprint,
            "inner_fold": int(inner_fold),
            "fit_row_ids": list(map(int, train_ids)),
            "heldout_row_ids": list(map(int, heldout_ids)),
        }
    )


def _load_completed_inner_evidence(
    *,
    fold_dir: Path,
    expected_fingerprint: str,
) -> dict[str, Any] | None:
    """Load an atomically completed inner fold, or decline a stale/partial cache."""

    complete_path = fold_dir / "complete.json"
    if not complete_path.is_file():
        return None
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            complete.get("status") != "complete"
            or complete.get("schema_version") != EVIDENCE_SCHEMA_VERSION
            or complete.get("input_fingerprint") != expected_fingerprint
        ):
            return None
        summaries = _read_jsonl(fold_dir / "feature_summaries.jsonl")
        confounder_rows = _read_jsonl(fold_dir / "confounder_univariable.jsonl")
        pairs = _read_jsonl(fold_dir / "pairwise_associations.jsonl")
        cluster_payload = json.loads(
            (fold_dir / "clusters.json").read_text(encoding="utf-8")
        )
        clusters = cluster_payload["clusters"]
        if not isinstance(clusters, list):
            return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    try:
        fold_report: dict[str, Any] = {
            "inner_fold": int(complete["inner_fold"]),
            "fit_row_ids": list(map(int, complete["fit_row_ids"])),
            "heldout_row_ids": list(map(int, complete["heldout_row_ids"])),
            "training_rows": int(complete["training_rows"]),
            "heldout_rows": int(complete["heldout_rows"]),
            "feature_summaries": summaries,
            "confounder_univariable": confounder_rows,
            "pairwise_associations": pairs,
            "clusters": clusters,
            "pairwise_parallelization": dict(
                complete.get("pairwise_parallelization") or {}
            ),
            "evidence_cache_reused": True,
        }
    except (KeyError, TypeError, ValueError):
        return None
    modifier_path = fold_dir / "effect_modifier_univariable.jsonl"
    if modifier_path.is_file():
        try:
            fold_report["effect_modifier_univariable"] = _read_jsonl(modifier_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return fold_report


def _cleanup_completed_pairwise_work(fold_dir: Path) -> None:
    """Drop bulky mmap/chunk intermediates after final evidence is durable."""

    work_dir = fold_dir / "pairwise_work"
    for name in ("encoded_context", "chunks"):
        try:
            shutil.rmtree(work_dir / name)
        except FileNotFoundError:
            pass
        except OSError as exc:
            LOGGER.warning("could not clean Stage 2 pairwise %s: %s", name, exc)
    manifest_path = work_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cleaned_after_complete"] = True
            _atomic_write_json(manifest_path, manifest)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass


def build_stage2_evidence(
    *,
    dataset: pd.DataFrame,
    extracted_fit: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    inner_splits: Sequence[Mapping[str, Any]],
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    output_dir: Path,
    policy: Stage2AgenticSelectionConfig,
    workers: int = 1,
    pairwise_chunk_size: int = DEFAULT_PAIRWISE_CHUNK_SIZE,
    pairwise_executor: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    """Build and persist label-aware and role-blind inner-fold evidence."""

    policy.validate()
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("Stage 2 evidence workers must be a positive integer")
    if (
        isinstance(pairwise_chunk_size, bool)
        or not isinstance(pairwise_chunk_size, int)
        or pairwise_chunk_size < 1
    ):
        raise ValueError("Stage 2 pairwise chunk size must be a positive integer")
    definitions = [dict(feature) for feature in definitions]
    _feature_by_id(definitions)
    if not inner_splits:
        raise ValueError("Stage 2 agentic selection requires inner folds")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_fingerprint, root_fingerprint = _stage2_evidence_fingerprints(
        dataset=dataset,
        extracted_fit=extracted_fit,
        definitions=definitions,
        inner_splits=inner_splits,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        outcome_type=outcome_type,
        policy=policy,
    )
    extracted_by_id = extracted_fit.set_index("_oci_row_id", drop=False)
    binary_outcome = str(outcome_type) == "binary"
    fold_reports: list[dict[str, Any]] = []
    fold_cluster_sets: list[list[dict[str, Any]]] = []
    local_executor: ProcessPoolExecutor | None = None
    executor = pairwise_executor
    if executor is None and workers > 1:
        local_executor = ProcessPoolExecutor(
            max_workers=workers,
            timeout=300,
            env={
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
        )
        executor = local_executor
    pool_scope = (
        "shared_stage2_run"
        if pairwise_executor is not None
        else "outer_fold_local"
        if local_executor is not None
        else "sequential"
    )
    try:
        for position, split in enumerate(inner_splits, start=1):
            inner_fold = int(split.get("inner_fold", position))
            train_ids = [int(value) for value in split.get("fit_row_ids") or []]
            heldout_ids = [int(value) for value in split.get("heldout_row_ids") or []]
            if not train_ids:
                raise ValueError(f"inner fold {inner_fold} has no fit rows")
            fold_dir = output_dir / f"inner_{inner_fold:03d}"
            inner_fingerprint = _inner_evidence_fingerprint(
                base_fingerprint=base_fingerprint,
                inner_fold=inner_fold,
                train_ids=train_ids,
                heldout_ids=heldout_ids,
            )
            cached = _load_completed_inner_evidence(
                fold_dir=fold_dir,
                expected_fingerprint=inner_fingerprint,
            )
            if cached is not None:
                LOGGER.info("reusing completed Stage 2 evidence inner_fold=%s", inner_fold)
                fold_reports.append(cached)
                fold_cluster_sets.append(cached["clusters"])
                continue

            frame = extracted_by_id.loc[train_ids].reset_index(drop=True)
            treatment = dataset.iloc[train_ids][treatment_column].to_numpy(dtype=float)
            outcome = dataset.iloc[train_ids][outcome_column].to_numpy(dtype=float)
            summaries = _feature_value_summaries(frame, definitions)
            confounder_rows = _confounder_univariable_rows(
                frame,
                treatment,
                outcome,
                definitions,
                binary_outcome=binary_outcome,
            )
            for row in summaries:
                row["evidence_id"] = f"s2e_inner_{inner_fold:03d}_{row['evidence_id']}"
            for row in confounder_rows:
                row["evidence_id"] = f"s2e_inner_{inner_fold:03d}_{row['evidence_id']}"
            pairs, parallelization = _build_pairwise_evidence(
                frame=frame,
                definitions=definitions,
                inner_fold=inner_fold,
                output_dir=fold_dir,
                input_fingerprint=inner_fingerprint,
                policy=policy,
                workers=workers,
                chunk_size=pairwise_chunk_size,
                executor=executor,
                pool_scope=pool_scope,
            )
            pair_q_values = _benjamini_hochberg([row.get("p_value") for row in pairs])
            for row, q_value in zip(pairs, pair_q_values):
                row["q_value"] = q_value
            clusters, similarity = _fold_clusters(definitions, pairs, policy=policy)
            for cluster_index, cluster in enumerate(clusters, start=1):
                cluster["cluster_id"] = (
                    f"inner_{inner_fold:03d}_cluster_{cluster_index:03d}"
                )
                cluster["evidence_id"] = f"s2e_{cluster['cluster_id']}"
            fold_cluster_sets.append(clusters)
            _atomic_write_jsonl(fold_dir / "feature_summaries.jsonl", summaries)
            _atomic_write_jsonl(
                fold_dir / "confounder_univariable.jsonl", confounder_rows
            )
            _atomic_write_jsonl(fold_dir / "pairwise_associations.jsonl", pairs)
            _atomic_write_json(fold_dir / "clusters.json", {"clusters": clusters})
            pd.DataFrame(
                similarity,
                index=[_feature_key(feature) for feature in definitions],
                columns=[_feature_key(feature) for feature in definitions],
            ).to_parquet(fold_dir / "similarity_matrix.parquet")
            fold_report = {
                "inner_fold": inner_fold,
                "fit_row_ids": train_ids,
                "heldout_row_ids": heldout_ids,
                "training_rows": len(train_ids),
                "heldout_rows": len(heldout_ids),
                "feature_summaries": summaries,
                "confounder_univariable": confounder_rows,
                "pairwise_associations": pairs,
                "clusters": clusters,
                "pairwise_parallelization": parallelization,
                "evidence_cache_reused": False,
            }
            _atomic_write_json(
                fold_dir / "complete.json",
                {
                    "status": "complete",
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "input_fingerprint": inner_fingerprint,
                    "inner_fold": inner_fold,
                    "fit_row_ids": train_ids,
                    "heldout_row_ids": heldout_ids,
                    "training_rows": len(train_ids),
                    "heldout_rows": len(heldout_ids),
                    "pairwise_parallelization": parallelization,
                },
            )
            fold_reports.append(fold_report)
            _cleanup_completed_pairwise_work(fold_dir)
    except BaseException:
        if local_executor is not None:
            local_executor.shutdown(wait=True, kill_workers=True)
        raise
    else:
        if local_executor is not None:
            local_executor.shutdown(wait=True)

    consensus, coassociation = _consensus_clusters(
        definitions,
        fold_cluster_sets,
        policy=policy,
    )
    for cluster in consensus:
        cluster["evidence_id"] = f"s2e_{cluster['cluster_id']}"
        cluster["inner_fold_memberships"] = {
            str(fold_reports[index]["inner_fold"]): [
                item["cluster_id"]
                for item in fold_cluster_sets[index]
                if set(item["member_feature_ids"]).intersection(
                    cluster["member_feature_ids"]
                )
            ]
            for index in range(len(fold_reports))
        }
    pd.DataFrame(
        coassociation,
        index=[_feature_key(feature) for feature in definitions],
        columns=[_feature_key(feature) for feature in definitions],
    ).to_parquet(output_dir / "consensus_coassociation.parquet")
    _write_json(output_dir / "consensus_clusters.json", {"clusters": consensus})
    summary = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "temporal_scope": TEMPORAL_SCOPE,
        "temporal_policy": (
            "All supplied data are pre-index-treatment by construction. Historical "
            "treatments are valid; no semantic timepoint filtering is applied."
        ),
        "policy": policy.public_dict(),
        "inner_folds": len(fold_reports),
        "features": len(definitions),
        "pairwise_associations_per_fold": (
            len(fold_reports[0]["pairwise_associations"]) if fold_reports else 0
        ),
        "consensus_clusters": len(consensus),
        "univariable_selection_threshold": None,
        "p_values_are_evidence_only": True,
        "parallelization": {
            "backend": "loky" if executor is not None else "sequential",
            "unit": "pair_chunk",
            "requested_workers": workers,
            "pool_scope": pool_scope,
            "pairwise_chunk_size": pairwise_chunk_size,
            "cached_inner_folds": sum(
                bool(fold.get("evidence_cache_reused")) for fold in fold_reports
            ),
            "inner_folds": [
                {
                    "inner_fold": fold["inner_fold"],
                    **dict(fold.get("pairwise_parallelization") or {}),
                }
                for fold in fold_reports
            ],
        },
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    result = {
        **summary,
        "definitions": definitions,
        "folds": fold_reports,
        "consensus_clusters_detail": consensus,
    }
    _atomic_write_json(
        output_dir / "complete.json",
        {
            "status": "complete",
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "input_fingerprint": root_fingerprint,
        },
    )
    return result


def _safe_latent_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug[:48] or "structured"


def _validate_condition(
    value: Mapping[str, Any],
    *,
    allowed_feature_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("a categorical-rule condition must be an object")
    feature_id = str(value.get("feature_id") or "")
    if feature_id not in allowed_feature_ids:
        raise ValueError(f"rule condition references unavailable feature {feature_id!r}")
    operator = str(value.get("operator") or "")
    if operator not in {"eq", "in", "gt", "ge", "lt", "le", "present", "missing"}:
        raise ValueError(f"unsupported categorical-rule condition operator {operator!r}")
    result: dict[str, Any] = {"feature_id": feature_id, "operator": operator}
    if operator == "in":
        values = value.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError("an 'in' condition requires a nonempty values list")
        result["values"] = [_validated_rule_scalar(item) for item in values]
    elif operator not in {"present", "missing"}:
        if "value" not in value:
            raise ValueError(f"a {operator!r} condition requires value")
        result["value"] = _validated_rule_scalar(value["value"])
    return result


def _validated_rule_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("categorical-rule values must be finite JSON scalars")


def _validate_rule_expression(
    value: Mapping[str, Any],
    *,
    allowed_feature_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("categorical_rule.expression must be an object")
    operation = str(value.get("op") or "")
    if operation in {"count_present", "sum", "mean", "minimum", "maximum", "coalesce"}:
        raw_features = value.get("feature_ids")
        if not isinstance(raw_features, list) or not raw_features:
            raise ValueError(f"rule operation {operation!r} requires feature_ids")
        feature_ids = list(dict.fromkeys(map(str, raw_features)))
        unknown = sorted(set(feature_ids) - allowed_feature_ids)
        if unknown:
            raise ValueError(f"rule references unavailable features: {unknown}")
        return {"op": operation, "feature_ids": feature_ids}
    if operation in {"any", "all", "count_true"}:
        raw_conditions = value.get("conditions")
        if not isinstance(raw_conditions, list) or not raw_conditions:
            raise ValueError(f"rule operation {operation!r} requires conditions")
        return {
            "op": operation,
            "conditions": [
                _validate_condition(item, allowed_feature_ids=allowed_feature_ids)
                for item in raw_conditions
            ],
        }
    if operation == "case":
        raw_cases = value.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("rule operation 'case' requires cases")
        cases: list[dict[str, Any]] = []
        for item in raw_cases:
            if not isinstance(item, Mapping) or "then" not in item:
                raise ValueError("each case requires when and then")
            cases.append(
                {
                    "when": _validate_condition(
                        item.get("when") or {},
                        allowed_feature_ids=allowed_feature_ids,
                    ),
                    "then": _validated_rule_scalar(item["then"]),
                }
            )
        if "else" not in value:
            raise ValueError("rule operation 'case' requires else")
        return {
            "op": operation,
            "cases": cases,
            "else": _validated_rule_scalar(value["else"]),
        }
    raise ValueError(f"unsupported categorical-rule operation {operation!r}")


def validate_latent_spec(
    value: Mapping[str, Any],
    *,
    cluster: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    """Validate the deliberately small, non-executable latent language."""

    if not isinstance(value, Mapping):
        raise ValueError("latent spec must be an object")
    kind = str(value.get("kind") or "")
    if kind not in {"mixed_component", "categorical_rule"}:
        raise ValueError("latent kind must be mixed_component or categorical_rule")
    cluster_members = set(map(str, cluster.get("member_feature_ids") or []))
    raw_sources = value.get("source_feature_ids")
    if not isinstance(raw_sources, list) or len(raw_sources) < 2:
        raise ValueError("a latent requires at least two source_feature_ids")
    sources = list(dict.fromkeys(map(str, raw_sources)))
    if len(sources) < 2 or not set(sources) <= cluster_members:
        raise ValueError("latent sources must be distinct members of the current cluster")
    label = str(value.get("label") or value.get("name") or "structured latent").strip()
    rationale = str(value.get("rationale") or "").strip()
    if not rationale:
        raise ValueError("latent rationale must explain the consolidation rule")
    result: dict[str, Any] = {
        "schema_version": LATENT_SCHEMA_VERSION,
        "kind": kind,
        "role_pass": role,
        "cluster_id": str(cluster["cluster_id"]),
        "source_feature_ids": sources,
        "label": label[:160],
        "rationale": rationale[:2000],
    }
    if kind == "mixed_component":
        result["output_type"] = "continuous"
    else:
        output_type = str(value.get("output_type") or "categorical")
        if output_type not in {"binary", "categorical", "ordinal", "continuous"}:
            raise ValueError("categorical_rule.output_type is unsupported")
        result["output_type"] = output_type
        result["expression"] = _validate_rule_expression(
            value.get("expression") or {},
            allowed_feature_ids=set(sources),
        )
        operation = str(result["expression"]["op"])
        if operation in {"any", "all"} and output_type != "binary":
            raise ValueError("any/all categorical rules require output_type='binary'")
        if operation in {
            "count_present",
            "count_true",
            "sum",
            "mean",
            "minimum",
            "maximum",
        } and output_type not in {"continuous", "ordinal"}:
            raise ValueError(
                f"categorical-rule operation {operation!r} requires a numeric output type"
            )
    latent_hash = _fingerprint(result)[:16]
    cluster_slug = _safe_latent_slug(str(cluster["cluster_id"]))
    result["latent_id"] = f"s2latent_{role}_{latent_hash}"
    result["name"] = f"s2_latent_{role}_{cluster_slug}_{latent_hash[:8]}"
    return result


def _condition_mask(
    frame: pd.DataFrame,
    condition: Mapping[str, Any],
    definitions_by_id: Mapping[str, Mapping[str, Any]],
) -> pd.Series:
    name = str(definitions_by_id[str(condition["feature_id"])]["name"])
    series = frame[name] if name in frame else pd.Series([None] * len(frame))
    operator = str(condition["operator"])
    if operator == "present":
        return series.notna()
    if operator == "missing":
        return series.isna()
    if operator == "in":
        allowed = {str(item) for item in condition["values"]}
        return series.notna() & series.astype(str).isin(allowed)
    target = condition.get("value")
    if operator == "eq":
        return series.notna() & (series.astype(str) == str(target))
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        threshold = float(target)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"numeric rule condition has nonnumeric value {target!r}") from exc
    return {
        "gt": numeric > threshold,
        "ge": numeric >= threshold,
        "lt": numeric < threshold,
        "le": numeric <= threshold,
    }[operator].fillna(False)


def _apply_rule(
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
    definitions_by_id: Mapping[str, Mapping[str, Any]],
) -> pd.Series:
    expression = spec["expression"]
    operation = str(expression["op"])
    if operation in {"count_present", "sum", "mean", "minimum", "maximum", "coalesce"}:
        columns = [
            frame[str(definitions_by_id[feature_id]["name"])]
            if str(definitions_by_id[feature_id]["name"]) in frame
            else pd.Series([None] * len(frame))
            for feature_id in expression["feature_ids"]
        ]
        values = pd.concat(columns, axis=1)
        if operation == "count_present":
            return values.notna().sum(axis=1).astype(float)
        if operation == "coalesce":
            if str(spec.get("output_type") or "") == "continuous":
                # A continuous ontology can still contain an occasional malformed
                # extractor value (for example, a complete blood-pressure pair in a
                # systolic-only column). Treat values that cannot be parsed as missing
                # and continue to the next equivalent source instead of allowing one
                # bad alias value to invalidate the canonical measurement.
                numeric = values.apply(pd.to_numeric, errors="coerce")
                return numeric.bfill(axis=1).iloc[:, 0]
            return values.bfill(axis=1).iloc[:, 0]
        numeric = values.apply(pd.to_numeric, errors="coerce")
        if operation == "sum":
            return numeric.sum(axis=1, min_count=1)
        if operation == "mean":
            return numeric.mean(axis=1, skipna=True)
        if operation == "minimum":
            return numeric.min(axis=1, skipna=True)
        return numeric.max(axis=1, skipna=True)
    if operation in {"any", "all", "count_true"}:
        masks = [
            _condition_mask(frame, condition, definitions_by_id)
            for condition in expression["conditions"]
        ]
        values = pd.concat(masks, axis=1)
        if operation == "any":
            return values.any(axis=1).astype(int)
        if operation == "all":
            return values.all(axis=1).astype(int)
        return values.sum(axis=1).astype(float)
    result = pd.Series([expression["else"]] * len(frame), index=frame.index, dtype=object)
    # First matching case wins.
    assigned = pd.Series(False, index=frame.index)
    for case in expression["cases"]:
        mask = _condition_mask(frame, case["when"], definitions_by_id) & ~assigned
        result.loc[mask] = case["then"]
        assigned |= mask
    return result


def _fit_mixed_component(
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
    definitions_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    columns: list[np.ndarray] = []
    encodings: list[dict[str, Any]] = []
    for feature_id in spec["source_feature_ids"]:
        definition = definitions_by_id[str(feature_id)]
        name = str(definition["name"])
        series = frame[name] if name in frame else pd.Series([None] * len(frame))
        if _feature_strategy(definition) in {
            "continuous",
            "continuous_with_categorical_fallback",
        }:
            numeric = pd.to_numeric(series, errors="coerce")
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            scale = float(numeric.std(ddof=0)) if numeric.notna().any() else 1.0
            if not math.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            columns.extend(
                [
                    ((numeric.fillna(median) - median) / scale).to_numpy(dtype=float),
                    series.isna().to_numpy(dtype=float),
                ]
            )
            fallback = series.notna() & numeric.isna()
            fallback_levels = sorted(series.loc[fallback].astype(str).unique())
            normalized = series.astype(str)
            for level in fallback_levels:
                columns.append((fallback & (normalized == level)).to_numpy(dtype=float))
            encodings.append(
                {
                    "feature_id": str(feature_id),
                    "kind": "continuous",
                    "median": median,
                    "scale": scale,
                    "fallback_levels": fallback_levels,
                }
            )
        else:
            observed = series.dropna().astype(str)
            levels = sorted(observed.unique())
            normalized = series.astype(str)
            for level in levels:
                columns.append((series.notna() & (normalized == level)).to_numpy(dtype=float))
            columns.append(series.isna().to_numpy(dtype=float))
            encodings.append(
                {
                    "feature_id": str(feature_id),
                    "kind": "categorical",
                    "levels": levels,
                }
            )
    matrix = np.column_stack(columns).astype(float) if columns else np.empty((len(frame), 0))
    if matrix.shape[1] < 1:
        raise ValueError("mixed component has no estimable columns")
    centers = matrix.mean(axis=0)
    centered = matrix - centers
    if not np.any(np.std(centered, axis=0) > 1e-12):
        raise ValueError("mixed component sources are constant in the fitting partition")
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    loading = vh[0].astype(float)
    anchor = int(np.argmax(np.abs(loading)))
    if loading[anchor] < 0:
        loading *= -1.0
    total_variance = float(np.sum(np.square(singular_values)))
    explained = (
        float(np.square(singular_values[0]) / total_variance)
        if total_variance > 0
        else 0.0
    )
    return {
        "schema_version": LATENT_SCHEMA_VERSION,
        "kind": "mixed_component",
        "spec": dict(spec),
        "encodings": encodings,
        "centers": centers.tolist(),
        "loading": loading.tolist(),
        "explained_variance_fraction": explained,
    }


def _mixed_component_matrix(
    frame: pd.DataFrame,
    state: Mapping[str, Any],
    definitions_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    columns: list[np.ndarray] = []
    source_missing: list[np.ndarray] = []
    for encoding in state["encodings"]:
        definition = definitions_by_id[str(encoding["feature_id"])]
        name = str(definition["name"])
        series = frame[name] if name in frame else pd.Series([None] * len(frame))
        source_missing.append(series.isna().to_numpy(dtype=bool))
        if encoding["kind"] == "continuous":
            numeric = pd.to_numeric(series, errors="coerce")
            median = float(encoding["median"])
            scale = float(encoding["scale"])
            columns.extend(
                [
                    ((numeric.fillna(median) - median) / scale).to_numpy(dtype=float),
                    series.isna().to_numpy(dtype=float),
                ]
            )
            fallback = series.notna() & numeric.isna()
            normalized = series.astype(str)
            for level in encoding.get("fallback_levels") or []:
                columns.append(
                    (fallback & (normalized == str(level))).to_numpy(dtype=float)
                )
        else:
            normalized = series.astype(str)
            for level in encoding["levels"]:
                columns.append(
                    (series.notna() & (normalized == str(level))).to_numpy(dtype=float)
                )
            columns.append(series.isna().to_numpy(dtype=float))
    matrix = np.column_stack(columns).astype(float)
    all_missing = np.logical_and.reduce(source_missing) if source_missing else np.ones(len(frame), bool)
    return matrix, all_missing


def _apply_latent_state(
    frame: pd.DataFrame,
    state: Mapping[str, Any],
    definitions_by_id: Mapping[str, Mapping[str, Any]],
) -> pd.Series:
    spec = state["spec"]
    if state["kind"] == "categorical_rule":
        return _apply_rule(frame, spec, definitions_by_id)
    matrix, all_missing = _mixed_component_matrix(frame, state, definitions_by_id)
    scores = (matrix - np.asarray(state["centers"], dtype=float)) @ np.asarray(
        state["loading"], dtype=float
    )
    scores = scores.astype(float)
    scores[all_missing] = np.nan
    return pd.Series(scores, index=frame.index)


def fit_latent_state(
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
    definitions_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if spec["kind"] == "mixed_component":
        return _fit_mixed_component(frame, spec, definitions_by_id)
    # Declarative rules have no learned parameters, but retain an explicit
    # fold-fitted state so their provenance matches learned components.
    return {
        "schema_version": LATENT_SCHEMA_VERSION,
        "kind": "categorical_rule",
        "spec": dict(spec),
    }


def latent_definition(
    spec: Mapping[str, Any],
    definitions_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_names = [
        str(definitions_by_id[str(feature_id)]["name"])
        for feature_id in spec["source_feature_ids"]
    ]
    value_type = str(spec["output_type"])
    definition: dict[str, Any] = {
        "feature_id": str(spec["latent_id"]),
        "name": str(spec["name"]),
        "display_name": str(spec["label"]),
        "description": str(spec["rationale"]),
        "question": "Derived deterministically from structured Stage 2 measurements.",
        "value_type": value_type,
        "categories_or_unit": [],
        "roles": [],
        "configured_explicit_feature": False,
        "derived_structured_latent": True,
        "latent_schema_version": LATENT_SCHEMA_VERSION,
        "latent_spec": dict(spec),
        "source_feature_ids": list(spec["source_feature_ids"]),
        "source_feature_names": source_names,
        "measurement_dependency_feature_ids": list(spec["source_feature_ids"]),
        "measurement_dependency_names": source_names,
        "temporal_scope": TEMPORAL_SCOPE,
    }
    if value_type == "continuous":
        definition["modeling_strategy"] = "continuous"
        definition["categories_or_unit"] = ["unitless derived score"]
    return definition


def _fit_predict_binary_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
) -> np.ndarray:
    train_y = np.asarray(train_y, dtype=float)
    if len(np.unique(train_y)) < 2:
        return np.full(len(valid_x), float(np.mean(train_y)), dtype=float)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    model.fit(train_x, train_y.astype(int))
    return np.clip(model.predict_proba(valid_x)[:, 1], 1e-6, 1.0 - 1e-6)


def _fit_predict_outcome_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    *,
    binary: bool,
) -> np.ndarray:
    if binary:
        return _fit_predict_binary_ridge(train_x, train_y, valid_x)
    model = Ridge(alpha=1.0)
    model.fit(train_x, train_y)
    return model.predict(valid_x)


def _encode_train_valid(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    for_interaction: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    train_columns: list[np.ndarray] = []
    valid_columns: list[np.ndarray] = []
    for definition in definitions:
        name = str(definition["name"])
        train_series = train[name] if name in train else pd.Series([None] * len(train))
        valid_series = valid[name] if name in valid else pd.Series([None] * len(valid))
        if _feature_strategy(definition) in {
            "continuous",
            "continuous_with_categorical_fallback",
        }:
            train_numeric = pd.to_numeric(train_series, errors="coerce")
            valid_numeric = pd.to_numeric(valid_series, errors="coerce")
            median = float(train_numeric.median()) if train_numeric.notna().any() else 0.0
            scale = float(train_numeric.std(ddof=0)) if train_numeric.notna().any() else 1.0
            if not math.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            train_columns.append(
                ((train_numeric.fillna(median) - median) / scale).to_numpy(dtype=float)
            )
            valid_columns.append(
                ((valid_numeric.fillna(median) - median) / scale).to_numpy(dtype=float)
            )
            train_fallback = train_series.notna() & train_numeric.isna()
            valid_fallback = valid_series.notna() & valid_numeric.isna()
            fallback_levels = sorted(train_series.loc[train_fallback].astype(str).unique())
            train_text, valid_text = train_series.astype(str), valid_series.astype(str)
            for level in fallback_levels:
                train_columns.append(
                    (train_fallback & (train_text == level)).to_numpy(dtype=float)
                )
                valid_columns.append(
                    (valid_fallback & (valid_text == level)).to_numpy(dtype=float)
                )
            if not for_interaction:
                train_columns.append(train_series.isna().to_numpy(dtype=float))
                valid_columns.append(valid_series.isna().to_numpy(dtype=float))
        else:
            levels = sorted(train_series.dropna().astype(str).unique())
            train_text, valid_text = train_series.astype(str), valid_series.astype(str)
            modeled_levels = levels[1:] if for_interaction and levels else levels
            for level in modeled_levels:
                train_columns.append(
                    (train_series.notna() & (train_text == level)).to_numpy(dtype=float)
                )
                valid_columns.append(
                    (valid_series.notna() & (valid_text == level)).to_numpy(dtype=float)
                )
            if not for_interaction:
                train_columns.append(train_series.isna().to_numpy(dtype=float))
                valid_columns.append(valid_series.isna().to_numpy(dtype=float))
    return (
        np.column_stack(train_columns).astype(float)
        if train_columns
        else np.empty((len(train), 0), dtype=float),
        np.column_stack(valid_columns).astype(float)
        if valid_columns
        else np.empty((len(valid), 0), dtype=float),
    )


class Stage2SelectionToolbox:
    """Typed, audited statistical operations available to selection agents."""

    def __init__(
        self,
        *,
        dataset: pd.DataFrame,
        extracted_fit: pd.DataFrame,
        definitions: Sequence[Mapping[str, Any]],
        inner_splits: Sequence[Mapping[str, Any]],
        evidence: Mapping[str, Any],
        treatment_column: str,
        outcome_column: str,
        outcome_type: str,
        unit_id_column: str,
        policy: Stage2AgenticSelectionConfig,
    ) -> None:
        self.dataset = dataset
        self.extracted_fit = extracted_fit.copy()
        self.extracted_by_id = self.extracted_fit.set_index("_oci_row_id", drop=False)
        self.original_definitions = [dict(item) for item in definitions]
        self.original_by_id = _feature_by_id(self.original_definitions)
        self.inner_splits = [dict(item) for item in inner_splits]
        self.evidence = dict(evidence)
        self.treatment_column = treatment_column
        self.outcome_column = outcome_column
        self.outcome_type = str(outcome_type)
        self.unit_id_column = unit_id_column
        self.policy = policy
        self.latent_specs: dict[str, dict[str, Any]] = {}
        self.latent_definitions: dict[str, dict[str, Any]] = {}
        self.latent_fold_states: dict[str, dict[int, dict[str, Any]]] = {}
        self.latent_structural_reports: dict[str, dict[str, Any]] = {}
        self.final_confounder_ids: list[str] = []
        self.audit: list[dict[str, Any]] = []
        self.role_evaluation_reports: list[dict[str, Any]] = []

    def _all_definitions(self) -> dict[str, dict[str, Any]]:
        return {**self.original_by_id, **self.latent_definitions}

    def _definition(self, candidate_id: str) -> dict[str, Any]:
        definitions = self._all_definitions()
        if str(candidate_id) not in definitions:
            raise ValueError(f"unknown Stage 2 candidate {candidate_id!r}")
        return definitions[str(candidate_id)]

    def _base_fold_frames(
        self, split: Mapping[str, Any]
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[int]]:
        train_ids = [int(value) for value in split.get("fit_row_ids") or []]
        valid_ids = [int(value) for value in split.get("heldout_row_ids") or []]
        train = self.extracted_by_id.loc[train_ids].reset_index(drop=True).copy()
        valid = self.extracted_by_id.loc[valid_ids].reset_index(drop=True).copy()
        return train, valid, train_ids, valid_ids

    def _augmented_fold_frames(
        self, split: Mapping[str, Any]
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[int]]:
        train, valid, train_ids, valid_ids = self._base_fold_frames(split)
        fold_number = int(split.get("inner_fold", 0))
        for latent_id, fold_states in self.latent_fold_states.items():
            if fold_number not in fold_states:
                continue
            definition = self.latent_definitions[latent_id]
            state = fold_states[fold_number]
            train[str(definition["name"])] = _apply_latent_state(
                train, state, self.original_by_id
            )
            valid[str(definition["name"])] = _apply_latent_state(
                valid, state, self.original_by_id
            )
        return train, valid, train_ids, valid_ids

    def get_cluster_evidence(
        self,
        cluster_id: str,
        *,
        record_audit: bool = True,
    ) -> dict[str, Any]:
        cluster = next(
            (
                item
                for item in self.evidence.get("consensus_clusters_detail") or []
                if str(item.get("cluster_id")) == str(cluster_id)
            ),
            None,
        )
        if cluster is None:
            raise ValueError(f"unknown consensus cluster {cluster_id!r}")
        members = set(map(str, cluster["member_feature_ids"]))
        folds: list[dict[str, Any]] = []
        for fold in self.evidence.get("folds") or []:
            folds.append(
                {
                    "inner_fold": int(fold["inner_fold"]),
                    "feature_summaries": [
                        row
                        for row in fold.get("feature_summaries") or []
                        if str(row["feature_id"]) in members
                    ],
                    "confounder_univariable": [
                        row
                        for row in fold.get("confounder_univariable") or []
                        if str(row["feature_id"]) in members
                    ],
                    "effect_modifier_univariable": [
                        row
                        for row in fold.get("effect_modifier_univariable") or []
                        if str(row["feature_id"]) in members
                    ],
                    "pairwise_associations": [
                        {
                            key: copy.deepcopy(value)
                            for key, value in row.items()
                            if key != "details"
                        }
                        | {
                            "details": {
                                key: copy.deepcopy(value)
                                for key, value in (row.get("details") or {}).items()
                                if key
                                not in {
                                    "raw_table",
                                    "inferential_table",
                                    "left_rare_level_mapping",
                                    "right_rare_level_mapping",
                                    "level_counts",
                                }
                            }
                        }
                        for row in fold.get("pairwise_associations") or []
                        if str(row["left_feature_id"]) in members
                        and str(row["right_feature_id"]) in members
                    ],
                }
            )
        result = {"cluster": cluster, "folds": folds}
        if record_audit:
            self.audit.append(
                {"tool": "get_cluster_evidence", "cluster_id": str(cluster_id)}
            )
        return result

    def inspect_pair(
        self,
        *,
        inner_fold: int,
        left_feature_id: str,
        right_feature_id: str,
        table: str = "raw",
        page: int = 1,
    ) -> dict[str, Any]:
        if table not in {"raw", "inferential"}:
            raise ValueError("inspect_pair.table must be raw or inferential")
        fold = next(
            (
                item
                for item in self.evidence.get("folds") or []
                if int(item["inner_fold"]) == int(inner_fold)
            ),
            None,
        )
        if fold is None:
            raise ValueError(f"unknown inner fold {inner_fold}")
        requested = {str(left_feature_id), str(right_feature_id)}
        row = next(
            (
                item
                for item in fold.get("pairwise_associations") or []
                if {
                    str(item["left_feature_id"]),
                    str(item["right_feature_id"]),
                }
                == requested
            ),
            None,
        )
        if row is None:
            raise ValueError("the requested feature pair is unavailable")
        details = dict(row.get("details") or {})
        table_key = f"{table}_table"
        records = list(details.get(table_key) or [])
        page_number = max(1, int(page))
        start = (page_number - 1) * self.policy.row_query_page_size
        end = start + self.policy.row_query_page_size
        result = {
            **{
                key: copy.deepcopy(value)
                for key, value in row.items()
                if key != "details"
            },
            "details": {
                key: copy.deepcopy(value)
                for key, value in details.items()
                if key not in {"raw_table", "inferential_table"}
            },
            "requested_table": table,
            "table_page": page_number,
            "table_page_size": self.policy.row_query_page_size,
            "table_total_records": len(records),
            "table_records": records[start:end],
        }
        self.audit.append(
            {
                "tool": "inspect_pair",
                "inner_fold": int(inner_fold),
                "left_feature_id": str(left_feature_id),
                "right_feature_id": str(right_feature_id),
                "table": table,
                "page": page_number,
            }
        )
        return result

    def query_rows(
        self,
        *,
        feature_ids: Sequence[str],
        inner_fold: int,
        partition: str,
        page: int = 1,
    ) -> dict[str, Any]:
        if partition not in {"fit", "heldout"}:
            raise ValueError("query_rows.partition must be fit or heldout")
        split = next(
            (
                item
                for item in self.inner_splits
                if int(item.get("inner_fold", 0)) == int(inner_fold)
            ),
            None,
        )
        if split is None:
            raise ValueError(f"unknown inner fold {inner_fold}")
        ids = list(dict.fromkeys(map(str, feature_ids)))
        definitions = [self._definition(candidate_id) for candidate_id in ids]
        # Row inspection is for source-value structure. Derived candidates are
        # deliberately excluded because evaluate_latent returns their results.
        if any(item.get("derived_structured_latent") for item in definitions):
            raise ValueError("query_rows accepts original measured features only")
        row_ids = [
            int(value)
            for value in split[
                "fit_row_ids" if partition == "fit" else "heldout_row_ids"
            ]
        ]
        start = (max(1, int(page)) - 1) * self.policy.row_query_page_size
        selected_ids = row_ids[start : start + self.policy.row_query_page_size]
        columns = [str(item["name"]) for item in definitions]
        values = self.extracted_by_id.loc[selected_ids][["_oci_row_id", *columns]].copy()
        if self.unit_id_column in self.dataset.columns:
            patient_ids = self.dataset.iloc[selected_ids][self.unit_id_column].tolist()
            values.insert(1, self.unit_id_column, patient_ids)
        result = {
            "inner_fold": int(inner_fold),
            "partition": partition,
            "page": max(1, int(page)),
            "page_size": self.policy.row_query_page_size,
            "total_rows": len(row_ids),
            "rows": values.astype(object).where(values.notna(), None).to_dict(
                orient="records"
            ),
            "labels_included": False,
        }
        self.audit.append(
            {
                "tool": "query_rows",
                "inner_fold": int(inner_fold),
                "partition": partition,
                "feature_ids": ids,
                "page": max(1, int(page)),
            }
        )
        return result

    def evaluate_latent(
        self,
        *,
        raw_spec: Mapping[str, Any],
        cluster: Mapping[str, Any],
        role: str,
    ) -> dict[str, Any]:
        spec = validate_latent_spec(raw_spec, cluster=cluster, role=role)
        latent_id = str(spec["latent_id"])
        definition = latent_definition(spec, self.original_by_id)
        fold_states: dict[int, dict[str, Any]] = {}
        fold_rows: list[dict[str, Any]] = []
        for position, split in enumerate(self.inner_splits, start=1):
            fold_number = int(split.get("inner_fold", position))
            train, valid, _train_ids, _valid_ids = self._base_fold_frames(split)
            state = fit_latent_state(train, spec, self.original_by_id)
            train_values = _apply_latent_state(train, state, self.original_by_id)
            valid_values = _apply_latent_state(valid, state, self.original_by_id)
            fold_states[fold_number] = state
            temporary = valid.copy()
            temporary[str(definition["name"])] = valid_values
            associations: list[dict[str, Any]] = []
            for source_id in spec["source_feature_ids"]:
                row = _pairwise_evidence(
                    temporary,
                    definition,
                    self.original_by_id[str(source_id)],
                    policy=self.policy,
                )
                associations.append(
                    {
                        "source_feature_id": str(source_id),
                        "association_kind": row.get("association_kind"),
                        "association": row.get("association"),
                        "signed_association": row.get("signed_association"),
                        "p_value": row.get("p_value"),
                        "n_pairwise_complete": row.get("n_pairwise_complete"),
                    }
                )
            fold_rows.append(
                {
                    "inner_fold": fold_number,
                    "fit_coverage": float(train_values.notna().mean()) if len(train_values) else 0.0,
                    "heldout_coverage": float(valid_values.notna().mean()) if len(valid_values) else 0.0,
                    "fit_unique": int(train_values.nunique(dropna=True)),
                    "heldout_unique": int(valid_values.nunique(dropna=True)),
                    "fit_standard_deviation": _finite(
                        pd.to_numeric(train_values, errors="coerce").std(ddof=0)
                    ),
                    "heldout_standard_deviation": _finite(
                        pd.to_numeric(valid_values, errors="coerce").std(ddof=0)
                    ),
                    "explained_variance_fraction": state.get(
                        "explained_variance_fraction"
                    ),
                    "heldout_source_associations": associations,
                }
            )
        mean_coverage = float(np.mean([row["heldout_coverage"] for row in fold_rows]))
        variable_folds = sum(int(row["heldout_unique"] >= 2) for row in fold_rows)
        accepted = bool(
            mean_coverage >= self.policy.latent_min_coverage and variable_folds >= 1
        )
        report = {
            "schema_version": LATENT_SCHEMA_VERSION,
            "latent_id": latent_id,
            "definition": definition,
            "spec": spec,
            "accepted_for_role_consideration": accepted,
            "mean_inner_heldout_coverage": mean_coverage,
            "variable_inner_heldout_folds": variable_folds,
            "folds": fold_rows,
            "construction_used_treatment_or_outcome": False,
        }
        if accepted:
            self.latent_specs[latent_id] = spec
            self.latent_definitions[latent_id] = definition
            self.latent_fold_states[latent_id] = fold_states
            self.latent_structural_reports[latent_id] = report
        self.audit.append(
            {
                "tool": "evaluate_latent",
                "role": role,
                "cluster_id": str(cluster["cluster_id"]),
                "latent_id": latent_id,
                "accepted": accepted,
                "spec_fingerprint": _fingerprint(spec),
            }
        )
        return report

    def _confounder_role_evidence(self, candidate_ids: Sequence[str]) -> dict[str, Any]:
        definitions = [self._definition(value) for value in candidate_ids]
        folds: list[dict[str, Any]] = []
        for position, split in enumerate(self.inner_splits, start=1):
            train, _valid, train_ids, _valid_ids = self._augmented_fold_frames(split)
            treatment = self.dataset.iloc[train_ids][self.treatment_column].to_numpy(dtype=float)
            outcome = self.dataset.iloc[train_ids][self.outcome_column].to_numpy(dtype=float)
            rows = _confounder_univariable_rows(
                train,
                treatment,
                outcome,
                definitions,
                binary_outcome=self.outcome_type == "binary",
            )
            folds.append(
                {
                    "inner_fold": int(split.get("inner_fold", position)),
                    "tests": rows,
                }
            )
        return {
            "role": "confounder",
            "candidate_ids": list(map(str, candidate_ids)),
            "folds": folds,
            "predictive_set_evaluation": self.evaluate_role_set(
                role="confounder", candidate_ids=candidate_ids, record_audit=False
            ),
        }

    def _loss(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        if self.outcome_type == "binary":
            return float(log_loss(observed, np.clip(predicted, 1e-6, 1 - 1e-6), labels=[0, 1]))
        return float(mean_squared_error(observed, predicted))

    def evaluate_role_set(
        self,
        *,
        role: str,
        candidate_ids: Sequence[str],
        record_audit: bool = True,
    ) -> dict[str, Any]:
        ids = list(dict.fromkeys(map(str, candidate_ids)))
        candidate_definitions = [self._definition(value) for value in ids]
        if role == "confounder":
            baseline = [
                item
                for item in self.original_definitions
                if item.get("configured_explicit_feature") is True
                and "confounder" in set(map(str, item.get("roles") or []))
                and _feature_key(item) not in ids
            ]
            full = [*baseline, *candidate_definitions]
            fold_rows: list[dict[str, Any]] = []
            for position, split in enumerate(self.inner_splits, start=1):
                train, valid, train_ids, valid_ids = self._augmented_fold_frames(split)
                t_train = self.dataset.iloc[train_ids][self.treatment_column].to_numpy(dtype=float)
                y_train = self.dataset.iloc[train_ids][self.outcome_column].to_numpy(dtype=float)
                t_valid = self.dataset.iloc[valid_ids][self.treatment_column].to_numpy(dtype=float)
                y_valid = self.dataset.iloc[valid_ids][self.outcome_column].to_numpy(dtype=float)
                base_train, base_valid = _encode_train_valid(train, valid, baseline)
                full_train, full_valid = _encode_train_valid(train, valid, full)
                if base_train.shape[1]:
                    base_t = _fit_predict_binary_ridge(base_train, t_train, base_valid)
                else:
                    base_t = np.full(len(valid), float(np.mean(t_train)))
                full_t = (
                    _fit_predict_binary_ridge(full_train, t_train, full_valid)
                    if full_train.shape[1]
                    else base_t
                )
                base_y_train = np.column_stack([t_train, base_train])
                base_y_valid = np.column_stack([t_valid, base_valid])
                full_y_train = np.column_stack([t_train, full_train])
                full_y_valid = np.column_stack([t_valid, full_valid])
                base_y = _fit_predict_outcome_ridge(
                    base_y_train,
                    y_train,
                    base_y_valid,
                    binary=self.outcome_type == "binary",
                )
                full_y = _fit_predict_outcome_ridge(
                    full_y_train,
                    y_train,
                    full_y_valid,
                    binary=self.outcome_type == "binary",
                )
                base_t_loss = float(log_loss(t_valid, np.clip(base_t, 1e-6, 1 - 1e-6), labels=[0, 1]))
                full_t_loss = float(log_loss(t_valid, np.clip(full_t, 1e-6, 1 - 1e-6), labels=[0, 1]))
                base_y_loss, full_y_loss = self._loss(y_valid, base_y), self._loss(y_valid, full_y)
                fold_rows.append(
                    {
                        "inner_fold": int(split.get("inner_fold", position)),
                        "treatment_loss_improvement": base_t_loss - full_t_loss,
                        "outcome_loss_improvement": base_y_loss - full_y_loss,
                        "baseline_treatment_loss": base_t_loss,
                        "candidate_treatment_loss": full_t_loss,
                        "baseline_outcome_loss": base_y_loss,
                        "candidate_outcome_loss": full_y_loss,
                    }
                )
            result = {
                "role": role,
                "candidate_ids": ids,
                "baseline_candidate_ids": [_feature_key(item) for item in baseline],
                "folds": fold_rows,
                "mean_treatment_loss_improvement": float(
                    np.mean([row["treatment_loss_improvement"] for row in fold_rows])
                ),
                "mean_outcome_loss_improvement": float(
                    np.mean([row["outcome_loss_improvement"] for row in fold_rows])
                ),
                "folds_improving_both": sum(
                    int(
                        row["treatment_loss_improvement"] > 0
                        and row["outcome_loss_improvement"] > 0
                    )
                    for row in fold_rows
                ),
            }
        elif role == "effect_modifier":
            confounders = [self._definition(value) for value in self.final_confounder_ids]
            fold_rows = []
            interaction_rows: list[dict[str, Any]] = []
            for position, split in enumerate(self.inner_splits, start=1):
                train, valid, train_ids, valid_ids = self._augmented_fold_frames(split)
                t_train = self.dataset.iloc[train_ids][self.treatment_column].to_numpy(dtype=float)
                y_train = self.dataset.iloc[train_ids][self.outcome_column].to_numpy(dtype=float)
                t_valid = self.dataset.iloc[valid_ids][self.treatment_column].to_numpy(dtype=float)
                y_valid = self.dataset.iloc[valid_ids][self.outcome_column].to_numpy(dtype=float)
                conf_train, conf_valid = _encode_train_valid(train, valid, confounders)
                if conf_train.shape[1]:
                    combined_conf = np.vstack([conf_train, conf_valid])
                    all_e = _fit_predict_binary_ridge(conf_train, t_train, combined_conf)
                else:
                    all_e = np.full(len(train) + len(valid), float(np.mean(t_train)))
                e_train, e_valid = all_e[: len(train)], all_e[len(train) :]
                y_train_design = np.column_stack([t_train, conf_train])
                y_both_design = np.vstack(
                    [y_train_design, np.column_stack([t_valid, conf_valid])]
                )
                all_m = _fit_predict_outcome_ridge(
                    y_train_design,
                    y_train,
                    y_both_design,
                    binary=self.outcome_type == "binary",
                )
                m_train, m_valid = all_m[: len(train)], all_m[len(train) :]
                modifier_train, modifier_valid = _encode_train_valid(
                    train,
                    valid,
                    candidate_definitions,
                    for_interaction=True,
                )
                rt_train, rt_valid = t_train - e_train, t_valid - e_valid
                ry_train, ry_valid = y_train - m_train, y_valid - m_valid
                null_train = rt_train.reshape(-1, 1)
                null_valid = rt_valid.reshape(-1, 1)
                full_train = rt_train.reshape(-1, 1) * np.column_stack(
                    [np.ones(len(train)), modifier_train]
                )
                full_valid = rt_valid.reshape(-1, 1) * np.column_stack(
                    [np.ones(len(valid)), modifier_valid]
                )
                null_model, full_model = Ridge(alpha=1.0), Ridge(alpha=1.0)
                null_model.fit(null_train, ry_train)
                full_model.fit(full_train, ry_train)
                null_loss = float(mean_squared_error(ry_valid, null_model.predict(null_valid)))
                full_loss = float(mean_squared_error(ry_valid, full_model.predict(full_valid)))
                conf_design = _design_for_features(train, confounders)
                reduced, _ = _rank_safe_columns(conf_design, t_train.reshape(-1, 1))
                tests = _modifier_test_chunk(
                    train,
                    t_train,
                    y_train,
                    reduced,
                    candidate_definitions,
                    binary_outcome=self.outcome_type == "binary",
                    p_value_threshold=1.0,
                )
                for row in tests:
                    row.pop("vote", None)
                interaction_rows.append(
                    {
                        "inner_fold": int(split.get("inner_fold", position)),
                        "tests": tests,
                    }
                )
                fold_rows.append(
                    {
                        "inner_fold": int(split.get("inner_fold", position)),
                        "heldout_r_loss_improvement": null_loss - full_loss,
                        "baseline_r_loss": null_loss,
                        "candidate_r_loss": full_loss,
                    }
                )
            result = {
                "role": role,
                "candidate_ids": ids,
                "nuisance_confounder_ids": list(self.final_confounder_ids),
                "folds": fold_rows,
                "interaction_tests": interaction_rows,
                "mean_heldout_r_loss_improvement": float(
                    np.mean([row["heldout_r_loss_improvement"] for row in fold_rows])
                ),
                "folds_with_positive_r_loss_improvement": sum(
                    int(row["heldout_r_loss_improvement"] > 0) for row in fold_rows
                ),
            }
        else:
            raise ValueError("role must be confounder or effect_modifier")
        if record_audit:
            self.audit.append(
                {"tool": "evaluate_role_set", "role": role, "candidate_ids": ids}
            )
            self.role_evaluation_reports.append(
                {"tool": "evaluate_role_set", "role": role, "candidate_ids": ids, "result": result}
            )
        return result

    def evaluate_role(self, *, role: str, candidate_ids: Sequence[str]) -> dict[str, Any]:
        ids = list(dict.fromkeys(map(str, candidate_ids)))
        if not ids:
            raise ValueError("evaluate_role requires candidate_ids")
        if role == "confounder":
            result = self._confounder_role_evidence(ids)
        elif role == "effect_modifier":
            result = self.evaluate_role_set(role=role, candidate_ids=ids, record_audit=False)
        else:
            raise ValueError("role must be confounder or effect_modifier")
        self.audit.append({"tool": "evaluate_role", "role": role, "candidate_ids": ids})
        self.role_evaluation_reports.append(
            {"tool": "evaluate_role", "role": role, "candidate_ids": ids, "result": result}
        )
        return result


def _stage1_evidence_for_features(
    definitions: Sequence[Mapping[str, Any]],
    packets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    packet_by_id = {
        str(packet.get("packet_id")): dict(packet)
        for packet in packets
        if packet.get("packet_id") is not None
    }
    result: dict[str, Any] = {}
    for definition in definitions:
        feature_id = _feature_key(definition)
        packet_ids = list(
            dict.fromkeys(
                map(
                    str,
                    definition.get("supporting_packet_ids")
                    or definition.get("packet_ids")
                    or [],
                )
            )
        )
        packet_summaries: list[dict[str, Any]] = []
        for packet_id in packet_ids[:8]:
            packet = packet_by_id.get(packet_id)
            if packet is None:
                continue
            content = packet.get("content")
            rendered = _canonical_json(content)
            packet_summaries.append(
                {
                    "packet_id": packet_id,
                    "source": packet.get("source"),
                    "architecture": packet.get("architecture"),
                    "scope": packet.get("scope"),
                    "json_path": packet.get("json_path"),
                    "observable_axes": packet.get("observable_axes"),
                    "content_excerpt": rendered[:1600],
                    "content_truncated": len(rendered) > 1600,
                }
            )
        result[feature_id] = {
            "supporting_packet_ids": packet_ids,
            "packets": packet_summaries,
            "packets_omitted_from_prompt": max(0, len(packet_ids) - len(packet_summaries)),
        }
    return result


def _agent_turn_validator(
    *,
    allowed_tools: set[str],
    final_validator: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def validate(value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("agent response must be one JSON object")
        action = str(value.get("action") or "")
        if action == "tool":
            tool = str(value.get("tool") or "")
            if tool not in allowed_tools:
                raise ValueError(f"unsupported tool {tool!r}; allowed: {sorted(allowed_tools)}")
            arguments = value.get("arguments")
            if not isinstance(arguments, Mapping):
                raise ValueError("tool action requires an arguments object")
            return {
                "action": "tool",
                "tool": tool,
                "arguments": dict(arguments),
                "reasoning": str(value.get("reasoning") or "")[:4000],
            }
        if action == "final":
            return final_validator(value)
        raise ValueError("agent action must be tool or final")

    return validate


def _run_json_agent(
    *,
    request_json: RequestJSON,
    system_prompt: str,
    task_payload: Mapping[str, Any],
    allowed_tools: set[str],
    dispatch: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
    final_validator: Callable[[Mapping[str, Any]], dict[str, Any]],
    tool_call_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _canonical_json(task_payload)},
    ]
    transcript: list[dict[str, Any]] = []
    validator = _agent_turn_validator(
        allowed_tools=allowed_tools,
        final_validator=final_validator,
    )
    for _turn in range(tool_call_limit + 1):
        response = request_json(messages, validator, request_kind="interpretation")
        transcript.append(copy.deepcopy(response))
        messages.append({"role": "assistant", "content": _canonical_json(response)})
        if response["action"] == "final":
            return response, transcript
        result = dict(dispatch(str(response["tool"]), response["arguments"]))
        messages.append(
            {
                "role": "user",
                "content": _canonical_json(
                    {
                        "type": "tool_result",
                        "tool": response["tool"],
                        "result": result,
                    }
                ),
            }
        )
    raise RuntimeError(
        f"Stage 2 selection agent exceeded its {tool_call_limit}-tool-call limit"
    )


_ROLE_AGENT_RULES = """
You are a statistical variable-selection agent operating inside one outer
training fold. All supplied measurements are pre-index-treatment by a hard
pipeline invariant; historical treatments are valid covariates. Do not invent
or apply semantic timepoint filters.

P-values and q-values are evidence, never gates. Favor sensitivity, but weigh
evidence both for and against each candidate and explicitly assess consistency
across inner folds. Clinical plausibility, diagnosis knowledge, and semantic
stories are not evidence for either causal role. A confounder recommendation
must rest on empirical treatment and outcome evidence. An effect-modifier
recommendation must rest on empirical interaction behavior and inner-heldout
R-loss under the supplied nuisance confounders. Stage-1 packets may ground what
was measured, but may not substitute for those empirical role criteria.

Latent construction itself is unsupervised: you may inspect structured feature
values, definitions, missingness, pairwise relationships, and clusters, but the
construction rule may not use treatment or outcome. Only the typed role tools
may use those labels after a latent has been structurally evaluated. Never ask
for or emit Python, SQL, shell, or executable expressions. The endpoint is a
trusted local/BAA analysis endpoint; row tools may return configured patient
identifiers, so use them only to distinguish records and do not reproduce them
in the final rationale.
""".strip()


def _cluster_system_prompt(role: str) -> str:
    return (
        _ROLE_AGENT_RULES
        + "\n\n"
        + f"You are the cluster analyst for the {role} pass. Examine every member, "
        "decide whether measured variables can be consolidated into zero, one, or "
        "two structured latents, evaluate every proposed latent with tools, then "
        "recommend promotion or rejection for every cluster member and every latent "
        "you retain for consideration. Original investigator-locked measurements "
        "must be recommended for their locked role, though a redundant latent may "
        "still be evaluated. Return JSON actions only.\n\n"
        "Available tools:\n"
        "- get_cluster_evidence({}): all inner-fold summaries, univariable evidence, "
        "compact contingency/correlation evidence, and cluster recurrence.\n"
        "- inspect_pair({inner_fold,left_feature_id,right_feature_id,table,page}): "
        "paginated raw or inferential contingency detail for one pair.\n"
        "- query_rows({feature_ids, inner_fold, partition, page}): paginated original "
        "structured values and identifiers, never treatment/outcome.\n"
        "- evaluate_latent({spec}): cross-fit an unsupervised latent. spec.kind is "
        "mixed_component or categorical_rule. A categorical_rule uses only the "
        "documented declarative operations count_present, sum, mean, minimum, "
        "maximum, coalesce, any, all, count_true, or case.\n"
        "- evaluate_role({candidate_ids}): per-candidate role evidence.\n"
        "- evaluate_role_set({candidate_ids}): joint heldout role evidence.\n"
        "A tool action is {action:'tool',tool:'...',arguments:{...},reasoning:'...'}. "
        "Finish with action:'final'."
    )


def _global_system_prompt(role: str) -> str:
    nuisance = (
        "The final confounder set is already fixed and is the nuisance adjustment "
        "set for modifier testing. It does not restrict which original variables "
        "remain eligible as modifiers."
        if role == "effect_modifier"
        else "This pass determines the final nuisance confounder set."
    )
    return (
        _ROLE_AGENT_RULES
        + "\n\n"
        + f"You are the global outer-fold adjudicator for the {role} role. {nuisance} "
        "Reconcile cluster reports with full fold evidence and make one audited "
        "decision for every eligible original feature and evaluated latent. Use "
        "tools for unresolved comparisons or joint models. A selected latent and "
        "one of its source features are mutually exclusive for this same role by "
        "default; if both are essential, record an explicit pairwise exception with "
        "an empirical rationale. Locked investigator features must retain their "
        "configured role. Return JSON actions only."
    )


def _cluster_final_validator(
    *,
    role: str,
    cluster: Mapping[str, Any],
    toolbox: Stage2SelectionToolbox,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def validate(value: Mapping[str, Any]) -> dict[str, Any]:
        if str(value.get("role") or "") != role:
            raise ValueError(f"cluster final role must be {role!r}")
        if str(value.get("cluster_id") or "") != str(cluster["cluster_id"]):
            raise ValueError("cluster final references the wrong cluster")
        latent_ids = list(dict.fromkeys(map(str, value.get("latent_ids") or [])))
        if len(latent_ids) > toolbox.policy.max_latents_per_cluster:
            raise ValueError("cluster final retains too many latent candidates")
        for latent_id in latent_ids:
            report = toolbox.latent_structural_reports.get(latent_id)
            if report is None or not report["accepted_for_role_consideration"]:
                raise ValueError(f"latent {latent_id!r} was not successfully evaluated")
            spec = toolbox.latent_specs[latent_id]
            if str(spec["cluster_id"]) != str(cluster["cluster_id"]):
                raise ValueError(f"latent {latent_id!r} belongs to another cluster")
        eligible = set(map(str, cluster["member_feature_ids"])) | set(latent_ids)
        recommendations = value.get("recommendations")
        if not isinstance(recommendations, list):
            raise ValueError("cluster final requires recommendations")
        by_id: dict[str, dict[str, Any]] = {}
        for raw in recommendations:
            if not isinstance(raw, Mapping):
                raise ValueError("each cluster recommendation must be an object")
            candidate_id = str(raw.get("candidate_id") or "")
            if candidate_id in by_id or candidate_id not in eligible:
                raise ValueError(f"invalid or duplicate cluster candidate {candidate_id!r}")
            if not isinstance(raw.get("promote"), bool):
                raise ValueError("cluster recommendation promote must be boolean")
            rationale = str(raw.get("rationale") or "").strip()
            if not rationale:
                raise ValueError("cluster recommendation requires a rationale")
            by_id[candidate_id] = {
                "candidate_id": candidate_id,
                "promote": bool(raw["promote"]),
                "evidence_for": list(map(str, raw.get("evidence_for") or [])),
                "evidence_against": list(map(str, raw.get("evidence_against") or [])),
                "inner_fold_consistency": str(raw.get("inner_fold_consistency") or "")[:2000],
                "rationale": rationale[:3000],
            }
        if set(by_id) != eligible:
            raise ValueError(
                "cluster recommendations must cover every original member and retained latent; "
                f"missing={sorted(eligible - set(by_id))}, extra={sorted(set(by_id) - eligible)}"
            )
        for candidate_id in cluster["member_feature_ids"]:
            definition = toolbox.original_by_id[str(candidate_id)]
            if definition.get("configured_explicit_feature") is True:
                configured = role in set(map(str, definition.get("roles") or []))
                if by_id[str(candidate_id)]["promote"] != configured:
                    raise ValueError(
                        "investigator-locked features must preserve exactly their "
                        "configured roles"
                    )
        role_tool_audits = [
            item
            for item in toolbox.audit
            if item.get("tool") in {"evaluate_role", "evaluate_role_set"}
            and item.get("role") == role
        ]
        for latent_id in latent_ids:
            if by_id[latent_id]["promote"] and not any(
                latent_id in set(map(str, item.get("candidate_ids") or []))
                for item in role_tool_audits
            ):
                raise ValueError(
                    f"promoted latent {latent_id!r} requires typed empirical role evaluation"
                )
        return {
            "action": "final",
            "role": role,
            "cluster_id": str(cluster["cluster_id"]),
            "assessment": str(value.get("assessment") or "")[:5000],
            "latent_ids": latent_ids,
            "recommendations": [by_id[candidate_id] for candidate_id in sorted(by_id)],
        }

    return validate


def _run_cluster_agent(
    *,
    role: str,
    cluster: Mapping[str, Any],
    toolbox: Stage2SelectionToolbox,
    stage1_by_feature: Mapping[str, Any],
    request_json: RequestJSON,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cluster_members = set(map(str, cluster["member_feature_ids"]))
    evaluated_here: set[str] = set()

    def existing_cluster_latent_ids() -> set[str]:
        return {
            latent_id
            for latent_id, spec in toolbox.latent_specs.items()
            if str(spec["cluster_id"]) == str(cluster["cluster_id"])
        }

    def allowed_candidate_ids() -> set[str]:
        return cluster_members | existing_cluster_latent_ids()

    def dispatch(tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if tool == "get_cluster_evidence":
            return toolbox.get_cluster_evidence(str(cluster["cluster_id"]))
        if tool == "query_rows":
            feature_ids = list(map(str, arguments.get("feature_ids") or []))
            if not set(feature_ids) <= cluster_members:
                raise ValueError("cluster agent may query only original members of its cluster")
            return toolbox.query_rows(
                feature_ids=feature_ids,
                inner_fold=int(arguments.get("inner_fold", 0)),
                partition=str(arguments.get("partition") or "fit"),
                page=int(arguments.get("page", 1)),
            )
        if tool == "inspect_pair":
            left = str(arguments.get("left_feature_id") or "")
            right = str(arguments.get("right_feature_id") or "")
            if {left, right} - cluster_members:
                raise ValueError("cluster agent may inspect only pairs in its cluster")
            return toolbox.inspect_pair(
                inner_fold=int(arguments.get("inner_fold", 0)),
                left_feature_id=left,
                right_feature_id=right,
                table=str(arguments.get("table") or "raw"),
                page=int(arguments.get("page", 1)),
            )
        if tool == "evaluate_latent":
            raw_spec = arguments.get("spec")
            if not isinstance(raw_spec, Mapping):
                raise ValueError("evaluate_latent requires spec")
            preview = validate_latent_spec(raw_spec, cluster=cluster, role=role)
            latent_id = str(preview["latent_id"])
            current_latents = existing_cluster_latent_ids() | evaluated_here
            if (
                latent_id not in current_latents
                and len(current_latents) >= toolbox.policy.max_latents_per_cluster
            ):
                raise ValueError("maximum total latent proposals reached for this cluster")
            evaluated_here.add(latent_id)
            return toolbox.evaluate_latent(
                raw_spec=raw_spec,
                cluster=cluster,
                role=role,
            )
        if tool in {"evaluate_role", "evaluate_role_set"}:
            ids = list(map(str, arguments.get("candidate_ids") or []))
            if not ids or not set(ids) <= allowed_candidate_ids():
                raise ValueError("role tool candidates must belong to the current cluster")
            if tool == "evaluate_role":
                return toolbox.evaluate_role(role=role, candidate_ids=ids)
            return toolbox.evaluate_role_set(role=role, candidate_ids=ids)
        raise ValueError(f"unsupported cluster tool {tool!r}")

    definitions = [toolbox.original_by_id[value] for value in cluster["member_feature_ids"]]
    existing_latents = [
        report
        for latent_id, report in toolbox.latent_structural_reports.items()
        if str(toolbox.latent_specs[latent_id]["cluster_id"]) == str(cluster["cluster_id"])
    ]
    payload = {
        "task": "analyze_cluster",
        "role": role,
        "temporal_scope": TEMPORAL_SCOPE,
        "cluster": cluster,
        "definitions": definitions,
        "stage1_evidence": {
            feature_id: stage1_by_feature.get(feature_id, {})
            for feature_id in cluster["member_feature_ids"]
        },
        "inner_fold_evidence": toolbox.get_cluster_evidence(
            str(cluster["cluster_id"]), record_audit=False
        ),
        "existing_evaluated_latents": existing_latents,
        "policy": {
            "maximum_total_latents": toolbox.policy.max_latents_per_cluster,
            "remaining_new_latent_slots": max(
                0,
                toolbox.policy.max_latents_per_cluster
                - len(existing_cluster_latent_ids()),
            ),
            "p_values_are_evidence_not_gates": True,
            "favor_sensitivity": True,
            "role_criterion": (
                "empirical treatment and outcome evidence"
                if role == "confounder"
                else "empirical interaction evidence and inner-heldout R-loss"
            ),
        },
        "required_final_shape": {
            "action": "final",
            "role": role,
            "cluster_id": str(cluster["cluster_id"]),
            "assessment": "string",
            "latent_ids": ["only IDs returned by successful evaluate_latent calls"],
            "recommendations": [
                {
                    "candidate_id": "every cluster member and retained latent",
                    "promote": "boolean",
                    "evidence_for": ["evidence IDs or concise statistical facts"],
                    "evidence_against": ["evidence IDs or concise statistical facts"],
                    "inner_fold_consistency": "string",
                    "rationale": "string",
                }
            ],
        },
    }
    return _run_json_agent(
        request_json=request_json,
        system_prompt=_cluster_system_prompt(role),
        task_payload=payload,
        allowed_tools={
            "get_cluster_evidence",
            "inspect_pair",
            "query_rows",
            "evaluate_latent",
            "evaluate_role",
            "evaluate_role_set",
        },
        dispatch=dispatch,
        final_validator=_cluster_final_validator(role=role, cluster=cluster, toolbox=toolbox),
        tool_call_limit=toolbox.policy.cluster_tool_call_limit,
    )


def _global_final_validator(
    *,
    role: str,
    eligible_ids: Sequence[str],
    toolbox: Stage2SelectionToolbox,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    eligible = set(map(str, eligible_ids))

    def validate(value: Mapping[str, Any]) -> dict[str, Any]:
        if str(value.get("role") or "") != role:
            raise ValueError(f"global final role must be {role!r}")
        raw_decisions = value.get("decisions")
        if not isinstance(raw_decisions, list):
            raise ValueError("global final requires decisions")
        decisions: dict[str, dict[str, Any]] = {}
        for raw in raw_decisions:
            if not isinstance(raw, Mapping):
                raise ValueError("each global decision must be an object")
            candidate_id = str(raw.get("candidate_id") or "")
            if candidate_id not in eligible or candidate_id in decisions:
                raise ValueError(f"invalid or duplicate global candidate {candidate_id!r}")
            if not isinstance(raw.get("promote"), bool):
                raise ValueError("global decision promote must be boolean")
            rationale = str(raw.get("rationale") or "").strip()
            consistency = str(raw.get("inner_fold_consistency") or "").strip()
            if not rationale or not consistency:
                raise ValueError("global decisions require rationale and inner_fold_consistency")
            decisions[candidate_id] = {
                "candidate_id": candidate_id,
                "promote": bool(raw["promote"]),
                "evidence_for": list(map(str, raw.get("evidence_for") or [])),
                "evidence_against": list(map(str, raw.get("evidence_against") or [])),
                "inner_fold_consistency": consistency[:3000],
                "rationale": rationale[:4000],
            }
        if set(decisions) != eligible:
            raise ValueError(
                "global decisions must cover every eligible candidate; "
                f"missing={sorted(eligible - set(decisions))}, extra={sorted(set(decisions) - eligible)}"
            )
        selected = set(map(str, value.get("selected_candidate_ids") or []))
        expected = {candidate_id for candidate_id, row in decisions.items() if row["promote"]}
        if selected != expected:
            raise ValueError("selected_candidate_ids must exactly match promote=true decisions")
        for candidate_id in eligible:
            definition = toolbox._definition(candidate_id)
            if definition.get("configured_explicit_feature") is True:
                configured = role in set(map(str, definition.get("roles") or []))
                if (candidate_id in selected) != configured:
                    raise ValueError(
                        "investigator-locked features must preserve exactly their "
                        "configured roles"
                    )
        role_tool_audits = [
            item
            for item in toolbox.audit
            if item.get("tool") in {"evaluate_role", "evaluate_role_set"}
            and item.get("role") == role
        ]
        for latent_id in selected.intersection(toolbox.latent_specs):
            if not any(
                latent_id in set(map(str, item.get("candidate_ids") or []))
                for item in role_tool_audits
            ):
                raise ValueError(
                    f"selected latent {latent_id!r} requires typed empirical role evaluation"
                )

        raw_exceptions = value.get("latent_source_exceptions") or []
        if not isinstance(raw_exceptions, list):
            raise ValueError("latent_source_exceptions must be a list")
        exceptions: dict[tuple[str, str], dict[str, str]] = {}
        for raw in raw_exceptions:
            if not isinstance(raw, Mapping):
                raise ValueError("each latent/source exception must be an object")
            latent_id = str(raw.get("latent_id") or "")
            source_id = str(raw.get("source_feature_id") or "")
            rationale = str(raw.get("empirical_rationale") or "").strip()
            if latent_id not in toolbox.latent_specs or source_id not in eligible or not rationale:
                raise ValueError("invalid latent/source exception")
            if source_id not in set(map(str, toolbox.latent_specs[latent_id]["source_feature_ids"])):
                raise ValueError("exception source is not a component of the latent")
            exceptions[(latent_id, source_id)] = {
                "latent_id": latent_id,
                "source_feature_id": source_id,
                "empirical_rationale": rationale[:3000],
            }
        required_exceptions = {
            (latent_id, source_id)
            for latent_id in selected
            if latent_id in toolbox.latent_specs
            for source_id in toolbox.latent_specs[latent_id]["source_feature_ids"]
            if str(source_id) in selected
        }
        if not required_exceptions <= set(exceptions):
            raise ValueError(
                "selecting a latent with a source for the same role requires an explicit "
                f"empirical exception for {sorted(required_exceptions - set(exceptions))}"
            )
        return {
            "action": "final",
            "role": role,
            "summary": str(value.get("summary") or "")[:6000],
            "decisions": [decisions[candidate_id] for candidate_id in sorted(decisions)],
            "selected_candidate_ids": sorted(selected),
            "latent_source_exceptions": [exceptions[key] for key in sorted(exceptions)],
        }

    return validate


def _candidate_evidence_summary(
    candidate_id: str,
    *,
    toolbox: Stage2SelectionToolbox,
    stage1_by_feature: Mapping[str, Any],
) -> dict[str, Any]:
    if candidate_id in toolbox.latent_structural_reports:
        structural: Mapping[str, Any] | None = toolbox.latent_structural_reports[candidate_id]
    else:
        structural = None
    folds: list[dict[str, Any]] = []
    for fold in toolbox.evidence.get("folds") or []:
        folds.append(
            {
                "inner_fold": fold["inner_fold"],
                "feature_summary": next(
                    (
                        row
                        for row in fold.get("feature_summaries") or []
                        if str(row["feature_id"]) == candidate_id
                    ),
                    None,
                ),
                "confounder_univariable": next(
                    (
                        row
                        for row in fold.get("confounder_univariable") or []
                        if str(row["feature_id"]) == candidate_id
                    ),
                    None,
                ),
                "effect_modifier_univariable": next(
                    (
                        row
                        for row in fold.get("effect_modifier_univariable") or []
                        if str(row["feature_id"]) == candidate_id
                    ),
                    None,
                ),
            }
        )
    return {
        "candidate_id": candidate_id,
        "definition": toolbox._definition(candidate_id),
        "fold_evidence": folds,
        "latent_structural_evidence": structural,
        "stage1_evidence": stage1_by_feature.get(candidate_id),
    }


def _latent_prompt_summary(report: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "latent_id": report.get("latent_id"),
        "spec": report.get("spec"),
        "accepted_for_role_consideration": report.get(
            "accepted_for_role_consideration"
        ),
        "mean_inner_heldout_coverage": report.get("mean_inner_heldout_coverage"),
        "variable_inner_heldout_folds": report.get("variable_inner_heldout_folds"),
        "folds": [
            {
                "inner_fold": row.get("inner_fold"),
                "heldout_coverage": row.get("heldout_coverage"),
                "heldout_unique": row.get("heldout_unique"),
                "heldout_standard_deviation": row.get("heldout_standard_deviation"),
                "explained_variance_fraction": row.get(
                    "explained_variance_fraction"
                ),
            }
            for row in report.get("folds") or []
        ],
    }


def _role_evidence_preview(
    candidate_id: str,
    *,
    role: str,
    toolbox: Stage2SelectionToolbox,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in toolbox.evidence.get("folds") or []:
        if role == "confounder":
            row = next(
                (
                    item
                    for item in fold.get("confounder_univariable") or []
                    if str(item["feature_id"]) == candidate_id
                ),
                None,
            )
            if row is not None:
                rows.append(
                    {
                        "inner_fold": fold["inner_fold"],
                        "evidence_id": row.get("evidence_id"),
                        "treatment_p_value": row.get("treatment_p_value"),
                        "treatment_q_value": row.get("treatment_q_value"),
                        "outcome_p_value": row.get("outcome_p_value"),
                        "outcome_q_value": row.get("outcome_q_value"),
                        "outcome_adjusted_for_treatment_p_value": row.get(
                            "outcome_adjusted_for_treatment_p_value"
                        ),
                        "outcome_adjusted_for_treatment_q_value": row.get(
                            "outcome_adjusted_for_treatment_q_value"
                        ),
                    }
                )
        else:
            row = next(
                (
                    item
                    for item in fold.get("effect_modifier_univariable") or []
                    if str(item["feature_id"]) == candidate_id
                ),
                None,
            )
            if row is not None:
                rows.append(
                    {
                        "inner_fold": fold["inner_fold"],
                        "evidence_id": row.get("evidence_id"),
                        "interaction_p_value": row.get("interaction_p_value"),
                        "interaction_q_value": row.get("interaction_q_value"),
                        "interaction_test": row.get("interaction_test"),
                    }
                )
    return rows


def _run_global_agent(
    *,
    role: str,
    eligible_ids: Sequence[str],
    cluster_reports: Sequence[Mapping[str, Any]],
    toolbox: Stage2SelectionToolbox,
    stage1_by_feature: Mapping[str, Any],
    request_json: RequestJSON,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible = set(map(str, eligible_ids))

    def dispatch(tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if tool == "get_candidate_evidence":
            candidate_id = str(arguments.get("candidate_id") or "")
            if candidate_id not in eligible:
                raise ValueError("candidate is not eligible in this role pass")
            toolbox.audit.append(
                {"tool": "get_candidate_evidence", "role": role, "candidate_id": candidate_id}
            )
            return _candidate_evidence_summary(
                candidate_id,
                toolbox=toolbox,
                stage1_by_feature=stage1_by_feature,
            )
        if tool == "get_cluster_evidence":
            return toolbox.get_cluster_evidence(str(arguments.get("cluster_id") or ""))
        if tool == "inspect_pair":
            left = str(arguments.get("left_feature_id") or "")
            right = str(arguments.get("right_feature_id") or "")
            if {left, right} - eligible or {left, right} - set(toolbox.original_by_id):
                raise ValueError(
                    "adjudicator pair features must be eligible original measurements"
                )
            return toolbox.inspect_pair(
                inner_fold=int(arguments.get("inner_fold", 0)),
                left_feature_id=left,
                right_feature_id=right,
                table=str(arguments.get("table") or "raw"),
                page=int(arguments.get("page", 1)),
            )
        if tool in {"evaluate_role", "evaluate_role_set"}:
            ids = list(map(str, arguments.get("candidate_ids") or []))
            if not ids or not set(ids) <= eligible:
                raise ValueError("role tool candidates must be eligible")
            if tool == "evaluate_role":
                return toolbox.evaluate_role(role=role, candidate_ids=ids)
            return toolbox.evaluate_role_set(role=role, candidate_ids=ids)
        raise ValueError(f"unsupported adjudicator tool {tool!r}")

    locked_ids = [
        candidate_id
        for candidate_id in eligible_ids
        if toolbox._definition(candidate_id).get("configured_explicit_feature") is True
        and role in set(map(str, toolbox._definition(candidate_id).get("roles") or []))
    ]
    payload = {
        "task": "outer_fold_role_adjudication",
        "role": role,
        "temporal_scope": TEMPORAL_SCOPE,
        "eligible_candidates": [
            {
                "candidate_id": candidate_id,
                "definition": toolbox._definition(candidate_id),
                "stage1_evidence": stage1_by_feature.get(candidate_id),
                "role_evidence_preview": _role_evidence_preview(
                    candidate_id,
                    role=role,
                    toolbox=toolbox,
                ),
                "typed_role_evaluations": [
                    item
                    for item in toolbox.role_evaluation_reports
                    if item["role"] == role
                    and candidate_id in set(map(str, item["candidate_ids"]))
                ],
                "latent_structural_summary": _latent_prompt_summary(
                    toolbox.latent_structural_reports.get(candidate_id)
                ),
            }
            for candidate_id in eligible_ids
        ],
        "cluster_reports": list(cluster_reports),
        "locked_candidate_ids": locked_ids,
        "nuisance_confounder_ids": (
            list(toolbox.final_confounder_ids) if role == "effect_modifier" else []
        ),
        "policy": {
            "p_values_are_evidence_not_gates": True,
            "favor_sensitivity": True,
            "require_inner_fold_consistency_assessment": True,
            "latent_source_exclusive_same_role_by_default": True,
            "agent_failure_fails_fold": True,
        },
        "tools": {
            "get_candidate_evidence": {"candidate_id": "one eligible ID"},
            "get_cluster_evidence": {"cluster_id": "one consensus cluster ID"},
            "inspect_pair": {
                "inner_fold": "fold number",
                "left_feature_id": "eligible original ID",
                "right_feature_id": "eligible original ID",
                "table": "raw or inferential",
                "page": "positive integer",
            },
            "evaluate_role": {"candidate_ids": ["one or more eligible IDs"]},
            "evaluate_role_set": {"candidate_ids": ["a joint eligible set"]},
        },
        "required_final_shape": {
            "action": "final",
            "role": role,
            "summary": "string",
            "decisions": [
                {
                    "candidate_id": "every eligible ID exactly once",
                    "promote": "boolean",
                    "evidence_for": ["evidence IDs or statistical facts"],
                    "evidence_against": ["evidence IDs or statistical facts"],
                    "inner_fold_consistency": "string",
                    "rationale": "string",
                }
            ],
            "selected_candidate_ids": ["exactly promote=true IDs"],
            "latent_source_exceptions": [
                {
                    "latent_id": "selected latent",
                    "source_feature_id": "selected source",
                    "empirical_rationale": "required if both share this role",
                }
            ],
        },
    }
    return _run_json_agent(
        request_json=request_json,
        system_prompt=_global_system_prompt(role),
        task_payload=payload,
        allowed_tools={
            "get_candidate_evidence",
            "get_cluster_evidence",
            "inspect_pair",
            "evaluate_role",
            "evaluate_role_set",
        },
        dispatch=dispatch,
        final_validator=_global_final_validator(
            role=role,
            eligible_ids=eligible_ids,
            toolbox=toolbox,
        ),
        tool_call_limit=toolbox.policy.adjudicator_tool_call_limit,
    )


def build_modifier_evidence(
    *,
    toolbox: Stage2SelectionToolbox,
    candidate_ids: Sequence[str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Persist all fold-local univariable interaction tests after W is fixed."""

    definitions = [toolbox._definition(value) for value in candidate_ids]
    reports: list[dict[str, Any]] = []
    for position, split in enumerate(toolbox.inner_splits, start=1):
        train, _valid, train_ids, _valid_ids = toolbox._augmented_fold_frames(split)
        treatment = toolbox.dataset.iloc[train_ids][toolbox.treatment_column].to_numpy(dtype=float)
        outcome = toolbox.dataset.iloc[train_ids][toolbox.outcome_column].to_numpy(dtype=float)
        confounders = [toolbox._definition(value) for value in toolbox.final_confounder_ids]
        confounder_design = _design_for_features(train, confounders)
        reduced, _ = _rank_safe_columns(confounder_design, treatment.reshape(-1, 1))
        rows = _modifier_test_chunk(
            train,
            treatment,
            outcome,
            reduced,
            definitions,
            binary_outcome=toolbox.outcome_type == "binary",
            p_value_threshold=1.0,
        )
        q_values = _benjamini_hochberg([row.get("interaction_p_value") for row in rows])
        inner_fold = int(split.get("inner_fold", position))
        for index, (row, q_value) in enumerate(zip(rows, q_values), start=1):
            row.pop("vote", None)
            row["interaction_q_value"] = q_value
            row["evidence_id"] = f"s2e_inner_{inner_fold:03d}_modifier_{index:04d}"
        evidence_fold = next(
            item
            for item in toolbox.evidence["folds"]
            if int(item["inner_fold"]) == inner_fold
        )
        evidence_fold["effect_modifier_univariable"] = rows
        fold_dir = output_dir / f"inner_{inner_fold:03d}"
        _write_jsonl(fold_dir / "effect_modifier_univariable.jsonl", rows)
        reports.append(
            {"inner_fold": inner_fold, "training_rows": len(train_ids), "tests": rows}
        )
    _write_json(
        output_dir / "modifier_summary.json",
        {
            "nuisance_confounder_ids": list(toolbox.final_confounder_ids),
            "candidate_ids": list(map(str, candidate_ids)),
            "folds": len(reports),
            "p_values_are_evidence_only": True,
        },
    )
    return reports


def measurement_definitions_for_selected(
    selected: Sequence[Mapping[str, Any]],
    original_definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    original_by_id = _feature_by_id(original_definitions)
    required: set[str] = set()
    for definition in selected:
        if definition.get("derived_structured_latent"):
            required.update(map(str, definition.get("source_feature_ids") or []))
        else:
            required.add(_feature_key(definition))
    unknown = sorted(required - set(original_by_id))
    if unknown:
        raise ValueError(f"selected latents have unavailable measurement dependencies: {unknown}")
    return [
        dict(definition)
        for definition in original_definitions
        if _feature_key(definition) in required
    ]


def fit_selected_latent_states(
    *,
    fit_frame: pd.DataFrame,
    selected: Sequence[Mapping[str, Any]],
    original_definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    original_by_id = _feature_by_id(original_definitions)
    states: list[dict[str, Any]] = []
    for definition in selected:
        if not definition.get("derived_structured_latent"):
            continue
        spec = definition.get("latent_spec")
        if not isinstance(spec, Mapping):
            raise ValueError(f"latent {definition.get('feature_id')!r} has no latent_spec")
        state = fit_latent_state(fit_frame, spec, original_by_id)
        states.append(
            {
                "latent_id": _feature_key(definition),
                "name": str(definition["name"]),
                "state": state,
            }
        )
    return states


def materialize_selected_latents(
    *,
    frame: pd.DataFrame,
    latent_states: Sequence[Mapping[str, Any]],
    original_definitions: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    result = frame.copy()
    original_by_id = _feature_by_id(original_definitions)
    for item in latent_states:
        result[str(item["name"])] = _apply_latent_state(
            result,
            item["state"],
            original_by_id,
        )
    return result


def select_stage2_features_agentically(
    *,
    dataset: pd.DataFrame,
    extracted_fit: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    inner_splits: Sequence[Mapping[str, Any]],
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    unit_id_column: str,
    stage1_packets: Sequence[Mapping[str, Any]],
    output_dir: Path,
    request_json: RequestJSON,
    policy: Stage2AgenticSelectionConfig,
    evidence_workers: int = 1,
    pairwise_chunk_size: int = DEFAULT_PAIRWISE_CHUNK_SIZE,
    evidence_executor: ProcessPoolExecutor | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Run cluster consolidation and two role-specific outer-fold adjudications."""

    policy.validate()
    originals = [dict(item) for item in definitions]
    if not originals:
        report = {
            "schema_version": SCHEMA_VERSION,
            "temporal_scope": TEMPORAL_SCOPE,
            "status": "complete_no_candidates",
            "features": [],
            "measurement_dependencies": [],
            "latent_states": [],
        }
        _write_json(output_dir / "agentic_selection.json", report)
        return [], report, [], []
    evidence_dir = output_dir / "stage2_evidence"
    evidence = build_stage2_evidence(
        dataset=dataset,
        extracted_fit=extracted_fit,
        definitions=originals,
        inner_splits=inner_splits,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        outcome_type=outcome_type,
        output_dir=evidence_dir,
        policy=policy,
        workers=evidence_workers,
        pairwise_chunk_size=pairwise_chunk_size,
        pairwise_executor=evidence_executor,
    )
    toolbox = Stage2SelectionToolbox(
        dataset=dataset,
        extracted_fit=extracted_fit,
        definitions=originals,
        inner_splits=inner_splits,
        evidence=evidence,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        outcome_type=outcome_type,
        unit_id_column=unit_id_column,
        policy=policy,
    )
    stage1_by_feature = _stage1_evidence_for_features(originals, stage1_packets)

    confounder_cluster_reports: list[dict[str, Any]] = []
    confounder_transcripts: list[dict[str, Any]] = []
    for cluster in evidence["consensus_clusters_detail"]:
        report, transcript = _run_cluster_agent(
            role="confounder",
            cluster=cluster,
            toolbox=toolbox,
            stage1_by_feature=stage1_by_feature,
            request_json=request_json,
        )
        confounder_cluster_reports.append(report)
        confounder_transcripts.append(
            {"cluster_id": cluster["cluster_id"], "turns": transcript}
        )
        _write_json(
            output_dir / "confounder_pass" / "clusters" / f"{cluster['cluster_id']}.json",
            {"report": report, "transcript": transcript},
        )
    confounder_eligible = [
        *[_feature_key(item) for item in originals],
        *sorted(toolbox.latent_definitions),
    ]
    confounder_adjudication, confounder_global_transcript = _run_global_agent(
        role="confounder",
        eligible_ids=confounder_eligible,
        cluster_reports=confounder_cluster_reports,
        toolbox=toolbox,
        stage1_by_feature=stage1_by_feature,
        request_json=request_json,
    )
    toolbox.final_confounder_ids = list(confounder_adjudication["selected_candidate_ids"])
    _write_json(
        output_dir / "confounder_pass" / "adjudication.json",
        {"decision": confounder_adjudication, "transcript": confounder_global_transcript},
    )

    modifier_baseline_candidates = [
        *[_feature_key(item) for item in originals],
        *sorted(toolbox.latent_definitions),
    ]
    modifier_univariable = build_modifier_evidence(
        toolbox=toolbox,
        candidate_ids=modifier_baseline_candidates,
        output_dir=evidence_dir,
    )
    modifier_cluster_reports: list[dict[str, Any]] = []
    modifier_transcripts: list[dict[str, Any]] = []
    for cluster in evidence["consensus_clusters_detail"]:
        report, transcript = _run_cluster_agent(
            role="effect_modifier",
            cluster=cluster,
            toolbox=toolbox,
            stage1_by_feature=stage1_by_feature,
            request_json=request_json,
        )
        modifier_cluster_reports.append(report)
        modifier_transcripts.append(
            {"cluster_id": cluster["cluster_id"], "turns": transcript}
        )
        _write_json(
            output_dir / "effect_modifier_pass" / "clusters" / f"{cluster['cluster_id']}.json",
            {"report": report, "transcript": transcript},
        )
    modifier_eligible = [
        *[_feature_key(item) for item in originals],
        *sorted(toolbox.latent_definitions),
    ]
    modifier_adjudication, modifier_global_transcript = _run_global_agent(
        role="effect_modifier",
        eligible_ids=modifier_eligible,
        cluster_reports=modifier_cluster_reports,
        toolbox=toolbox,
        stage1_by_feature=stage1_by_feature,
        request_json=request_json,
    )
    _write_json(
        output_dir / "effect_modifier_pass" / "adjudication.json",
        {"decision": modifier_adjudication, "transcript": modifier_global_transcript},
    )

    confounder_ids = set(confounder_adjudication["selected_candidate_ids"])
    modifier_ids = set(modifier_adjudication["selected_candidate_ids"])
    all_by_id = {**toolbox.original_by_id, **toolbox.latent_definitions}
    ordered_ids = [
        *[_feature_key(item) for item in originals],
        *sorted(toolbox.latent_definitions),
    ]
    selected: list[dict[str, Any]] = []
    for candidate_id in ordered_ids:
        roles: list[str] = []
        if candidate_id in confounder_ids:
            roles.append("confounder")
        if candidate_id in modifier_ids:
            roles.append("effect_modifier")
        if not roles:
            continue
        definition = copy.deepcopy(all_by_id[candidate_id])
        definition["roles"] = roles
        definition["selection_source"] = (
            "investigator_locked"
            if definition.get("configured_explicit_feature") is True
            else "agentic_outer_fold_adjudication"
        )
        selected.append(definition)
    dependencies = measurement_definitions_for_selected(selected, originals)
    latent_states = fit_selected_latent_states(
        fit_frame=extracted_fit,
        selected=selected,
        original_definitions=originals,
    )
    _write_json(
        output_dir / "latent_registry.json",
        {
            "schema_version": LATENT_SCHEMA_VERSION,
            "latents": list(toolbox.latent_structural_reports.values()),
            "selected_latent_states": latent_states,
        },
    )
    decisions: list[dict[str, Any]] = []
    conf_decisions = {
        str(row["candidate_id"]): row for row in confounder_adjudication["decisions"]
    }
    mod_decisions = {
        str(row["candidate_id"]): row for row in modifier_adjudication["decisions"]
    }
    for candidate_id in ordered_ids:
        decisions.append(
            {
                "candidate_id": candidate_id,
                "name": str(all_by_id[candidate_id]["name"]),
                "derived_structured_latent": bool(
                    all_by_id[candidate_id].get("derived_structured_latent")
                ),
                "confounder": conf_decisions.get(candidate_id),
                "effect_modifier": mod_decisions.get(candidate_id),
                "retained": candidate_id in confounder_ids or candidate_id in modifier_ids,
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "tool_protocol_version": TOOL_PROTOCOL_VERSION,
        "temporal_scope": TEMPORAL_SCOPE,
        "temporal_scope_is_hard_input_invariant": True,
        "semantic_temporal_filtering": False,
        "historical_treatments_are_eligible": True,
        "agent_failure_policy": "fail_outer_fold_without_statistical_fallback",
        "agent_endpoint_data_contract": "trusted_local_or_baa_row_level_access",
        "row_tools_may_include_configured_patient_identifiers": True,
        "p_values_are_evidence_only": True,
        "policy": policy.public_dict(),
        "confounder_pass": {
            "cluster_reports": confounder_cluster_reports,
            "adjudication": confounder_adjudication,
        },
        "effect_modifier_pass": {
            "nuisance_confounder_ids": list(toolbox.final_confounder_ids),
            "univariable_folds": modifier_univariable,
            "cluster_reports": modifier_cluster_reports,
            "adjudication": modifier_adjudication,
        },
        "decisions": decisions,
        "retained_feature_ids": [_feature_key(item) for item in selected],
        "measurement_dependency_feature_ids": [_feature_key(item) for item in dependencies],
        "tool_audit": toolbox.audit,
        "typed_role_evaluations": toolbox.role_evaluation_reports,
    }
    _write_json(output_dir / "agentic_selection.json", report)
    return selected, report, dependencies, latent_states


__all__ = [
    "DEFAULT_PAIRWISE_CHUNK_SIZE",
    "EVIDENCE_SCHEMA_VERSION",
    "LATENT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TEMPORAL_SCOPE",
    "Stage2AgenticSelectionConfig",
    "agentic_selection_config_from_mapping",
    "build_stage2_evidence",
    "fit_selected_latent_states",
    "materialize_selected_latents",
    "measurement_definitions_for_selected",
    "select_stage2_features_agentically",
    "validate_latent_spec",
]
