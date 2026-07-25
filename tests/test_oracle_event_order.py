from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import oci.inference.production_oracle_evaluation as evaluator


def _sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    predictions = tmp_path / "frozen_predictions.parquet"
    pd.DataFrame(
        {
            "_oci_row_id": list(range(10)),
            "outer_fold": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "pred_ite_prob": [value / 100 for value in range(10)],
        }
    ).to_parquet(predictions, index=False)
    prediction_sha, _ = evaluator.stable_file_sha256(predictions)
    body = {
        "prediction_path": str(predictions.resolve()),
        "prediction_sha256": prediction_sha,
        "prediction_row_count": 10,
    }
    manifest = tmp_path / "immutable_run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "all_evidence_fusion_predictions_v5",
                "content_sha256": _sha(body),
                "body": body,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    row_map = tmp_path / "row_registry.parquet"
    pd.DataFrame(
        {
            "_oci_row_id": list(range(10)),
            "person": [f"p{value}" for value in range(10)],
        }
    ).to_parquet(row_map, index=False)
    oracle = tmp_path / "separate_oracle.parquet"
    pd.DataFrame(
        {
            "oracle_person": [f"p{value}" for value in range(10)],
            "true_effect": [value / 90 for value in range(10)],
        }
    ).to_parquet(oracle, index=False)
    return predictions, manifest, row_map, oracle


def test_oracle_is_first_opened_after_freeze_manifest_and_row_map_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions, manifest, row_map, oracle = _inputs(tmp_path)
    observed_reads: list[Path] = []
    original_read = pd.read_parquet

    def recording_read(path, *args, **kwargs):
        observed_reads.append(Path(path).resolve())
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(evaluator.pd, "read_parquet", recording_read)
    result = evaluator.evaluate_frozen_predictions_posthoc(
        predictions_path=predictions,
        prediction_manifest_path=manifest,
        unit_id_map_path=row_map,
        oracle_dataset_path=oracle,
        output_dir=tmp_path / "evaluation",
        unit_id_column="person",
        oracle_unit_id_column="oracle_person",
        oracle_ite_column="true_effect",
    )
    assert observed_reads[:3] == [
        predictions.resolve(),
        row_map.resolve(),
        oracle.resolve(),
    ]
    assert [row["sequence"] for row in result["event_order"]] == [1, 2, 3, 4, 5]
    assert result[
        "oracle_access_only_after_prediction_manifest_and_row_map_validation"
    ] is True
    assert len(result["per_fold"]) == 5


def test_invalid_prediction_manifest_aborts_before_oracle_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions, manifest, row_map, oracle = _inputs(tmp_path)
    wrapper = json.loads(manifest.read_text(encoding="utf-8"))
    wrapper["body"]["prediction_row_count"] = 999
    # Preserve the old digest so wrapper authentication fails.
    manifest.write_text(json.dumps(wrapper), encoding="utf-8")
    oracle_opened = False
    original_read = pd.read_parquet

    def guarded_read(path, *args, **kwargs):
        nonlocal oracle_opened
        if Path(path).resolve() == oracle.resolve():
            oracle_opened = True
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(evaluator.pd, "read_parquet", guarded_read)
    with pytest.raises(ValueError, match="manifest wrapper"):
        evaluator.evaluate_frozen_predictions_posthoc(
            predictions_path=predictions,
            prediction_manifest_path=manifest,
            unit_id_map_path=row_map,
            oracle_dataset_path=oracle,
            output_dir=tmp_path / "evaluation",
            unit_id_column="person",
            oracle_unit_id_column="oracle_person",
            oracle_ite_column="true_effect",
        )
    assert oracle_opened is False
