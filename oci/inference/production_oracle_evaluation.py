"""Strict post-freeze oracle evaluation joined by configured unit ID."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .production_text_preparation import stable_file_sha256


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_manifest(path: Path) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"prediction manifest contains duplicate key {key!r}")
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(
                    f"prediction manifest contains non-finite value {token}"
                )
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prediction manifest is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("prediction manifest must contain one object")
    if set(value) == {"schema_version", "content_sha256", "body"}:
        body = value["body"]
        if (
            not isinstance(body, Mapping)
            or value["content_sha256"] != _canonical_sha256(body)
        ):
            raise ValueError("prediction manifest wrapper failed content validation")
        return body
    # Compatibility with an older unwrapped manifest remains fail-closed: it
    # must carry and satisfy its own content hash.
    content_sha = value.get("content_sha256")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if content_sha != _canonical_sha256(body):
        raise ValueError("unwrapped prediction manifest failed content validation")
    return body


def _metrics(frame: pd.DataFrame, truth: str, estimate: str) -> dict[str, Any]:
    y = frame[truth].to_numpy(dtype=float)
    p = frame[estimate].to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("oracle truth and estimates must be finite")
    truth_var = float(np.var(y))
    estimate_var = float(np.var(p))
    pearson = float(pearsonr(y, p).statistic) if truth_var > 0 and estimate_var > 0 else None
    spearman = float(spearmanr(y, p).statistic) if truth_var > 0 and estimate_var > 0 else None
    error = p - y
    return {
        "row_count": len(frame),
        "pearson_correlation_primary": pearson,
        "spearman_correlation_secondary": spearman,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(np.square(error)))),
        "mean_signed_error": float(np.mean(error)),
        "truth_variance": truth_var,
        "estimate_variance": estimate_var,
    }


def evaluate_frozen_predictions_posthoc(
    *, predictions_path: Path, prediction_manifest_path: Path, unit_id_map_path: Path,
    oracle_dataset_path: Path, output_dir: Path, unit_id_column: str,
    oracle_unit_id_column: str, oracle_ite_column: str,
    estimate_column: str = "pred_ite_prob", fold_column: str = "outer_fold",
) -> Mapping[str, Any]:
    """Validate frozen bytes and row map before projecting the oracle columns."""
    events: list[dict[str, Any]] = []
    prediction_sha, prediction_size = stable_file_sha256(predictions_path)
    events.append(
        {
            "sequence": 1,
            "event": "frozen_prediction_bytes_authenticated",
            "sha256": prediction_sha,
        }
    )
    manifest_sha, manifest_size = stable_file_sha256(prediction_manifest_path)
    events.append(
        {
            "sequence": 2,
            "event": "prediction_manifest_bytes_authenticated",
            "sha256": manifest_sha,
        }
    )
    row_map_sha, row_map_size = stable_file_sha256(unit_id_map_path)
    events.append(
        {
            "sequence": 3,
            "event": "stage1_row_map_bytes_authenticated",
            "sha256": row_map_sha,
        }
    )
    manifest_body = _strict_manifest(Path(prediction_manifest_path))
    declared = manifest_body.get("prediction_sha256") or manifest_body.get(
        "frozen_prediction_sha256"
    )
    declared_path = manifest_body.get("prediction_path")
    if (
        declared != prediction_sha
        or (
            declared_path is not None
            and Path(str(declared_path)).resolve(strict=True)
            != Path(predictions_path).resolve(strict=True)
        )
    ):
        raise ValueError("prediction bytes do not match the immutable run manifest")
    predictions = pd.read_parquet(predictions_path)
    if any(str(column).lower().startswith(("true_", "oracle_")) for column in predictions):
        raise ValueError("frozen predictions contain an oracle column")
    required = {estimate_column, fold_column, "_oci_row_id"}
    if (
        not required.issubset(predictions.columns)
        or predictions["_oci_row_id"].duplicated().any()
        or (
            manifest_body.get("prediction_row_count") is not None
            and int(manifest_body["prediction_row_count"]) != len(predictions)
        )
    ):
        raise ValueError("frozen predictions have an invalid key/metric schema")
    row_map = pd.read_parquet(unit_id_map_path)
    map_id_column = unit_id_column if unit_id_column in row_map else "unit_id"
    if map_id_column not in row_map or row_map[map_id_column].duplicated().any():
        raise ValueError("Stage-1 unit-ID row map is invalid")
    if "_oci_row_id" not in row_map or row_map["_oci_row_id"].duplicated().any():
        raise ValueError("Stage-1 positional row map is invalid")
    predictions = predictions.merge(
        row_map[["_oci_row_id", map_id_column]].rename(columns={map_id_column: unit_id_column}),
        on="_oci_row_id", how="left", validate="one_to_one",
    )
    if predictions[unit_id_column].isna().any():
        raise ValueError("prediction row IDs differ from the authenticated Stage-1 row map")
    events.append(
        {
            "sequence": 4,
            "event": "prediction_manifest_schema_and_row_map_validated",
            "oracle_opened": False,
        }
    )
    # This is intentionally the first oracle read in this function.
    oracle = pd.read_parquet(
        oracle_dataset_path, columns=[oracle_unit_id_column, oracle_ite_column]
    )
    events.append(
        {
            "sequence": 5,
            "event": "oracle_source_opened",
            "all_freeze_validations_preceded_oracle_open": True,
        }
    )
    if oracle[oracle_unit_id_column].isna().any() or oracle[oracle_unit_id_column].duplicated().any():
        raise ValueError("oracle IDs must be complete and unique")
    oracle = oracle.rename(columns={oracle_unit_id_column: unit_id_column})
    joined = predictions.merge(oracle, on=unit_id_column, how="left", validate="one_to_one")
    if len(joined) != len(predictions) or joined[oracle_ite_column].isna().any():
        raise ValueError("oracle join is not complete one-to-one")
    root = Path(output_dir)
    if root.exists():
        raise ValueError("oracle evaluation directory must be fresh")
    root.mkdir(parents=True)
    joined_path = root / "predictions_with_oracle.parquet"
    joined.to_parquet(joined_path, index=False)
    result = {
        "schema_version": "posthoc_oracle_ite_evaluation_v1",
        "prediction_sha256_before_oracle_read": prediction_sha,
        "prediction_size_bytes": prediction_size,
        "prediction_manifest_sha256_before_oracle_read": manifest_sha,
        "prediction_manifest_size_bytes": manifest_size,
        "stage1_row_map_sha256_before_oracle_read": row_map_sha,
        "stage1_row_map_size_bytes": row_map_size,
        "event_order": events,
        "oracle_open_sequence": 5,
        "freeze_validation_completed_sequence": 4,
        "oracle_access_only_after_prediction_manifest_and_row_map_validation": True,
        "overall": _metrics(joined, oracle_ite_column, estimate_column),
        "per_fold": [
            {"outer_fold": int(fold), **_metrics(group, oracle_ite_column, estimate_column)}
            for fold, group in joined.groupby(fold_column, sort=True)
        ],
        "oracle_join_performed_posthoc": True,
        "joined_path": str(joined_path.resolve()),
    }
    (root / "evaluation_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return result


__all__ = ["evaluate_frozen_predictions_posthoc"]
