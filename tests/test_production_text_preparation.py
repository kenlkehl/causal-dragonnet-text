import json

import pandas as pd
import pytest

from oci.inference.production_text_preparation import (
    NEUTRAL_RUN_MARKER,
    TextPreparationOptions,
    prepare_modeling_cohort,
    verify_tokenizer_character_coverage,
)


def test_preparation_is_configurable_lossless_and_audited(tmp_path):
    source = pd.DataFrame({
        "subject": ["x", "y", "z"], "note": ["ok", "  ", "a" + "—" * 5 + "b"],
        "tx": [0, 1, 0], "event": [0, 1, 1], "true_age": [999, 999, 999],
    })
    path = tmp_path / "source.parquet"
    source.to_parquet(path, index=False)
    manifest = prepare_modeling_cohort(TextPreparationOptions(
        path, tmp_path / "prepared", "subject", "note", "tx", "event",
        repeated_character_threshold=5,
    ))
    output = pd.read_parquet(manifest["output"]["path"])
    assert list(output) == ["subject", "note", "tx", "event"]
    assert output[["subject", "tx", "event"]].equals(source[["subject", "tx", "event"]])
    assert output.loc[2, "note"] == "a" + NEUTRAL_RUN_MARKER + "b"
    assert manifest["affected_unit_ids"] == ["y", "z"]
    assert "true_age" not in json.dumps(manifest)


def test_tokenizer_coverage_aborts_on_omission():
    with pytest.raises(ValueError, match="omitted"):
        verify_tokenizer_character_coverage(["abcdef"], {"bad": lambda text: [(0, 3)]})


def test_numpy_integer_unit_ids_are_manifest_serializable(tmp_path):
    source = pd.DataFrame({"id": [101, 102], "text": ["a", "b"], "tx": [0, 1], "y": [1, 0]})
    path = tmp_path / "source.parquet"
    source.to_parquet(path, index=False)
    result = prepare_modeling_cohort(TextPreparationOptions(
        path, tmp_path / "prepared", "id", "text", "tx", "y"
    ))
    assert result["rows"][0]["unit_id"] == 101
