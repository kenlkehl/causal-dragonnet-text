from __future__ import annotations

import copy
from pathlib import Path

import pytest

from oci.inference.compact_preflight_compression_benchmark import (
    CompactPreflightCompressionBenchmarkConfig,
    run_compact_preflight_compression_benchmark,
    validate_compact_preflight_compression_benchmark_result,
)
from oci.inference.portable_workflow_spec import identity_sha256
from tests.test_production_stage1_cluster_preflight_artifact_v2 import (
    _seal,
    portable_validators,
)


def _config() -> CompactPreflightCompressionBenchmarkConfig:
    return CompactPreflightCompressionBenchmarkConfig(
        codecs=("none", "zstd"),
        warmup_repetitions_per_codec=0,
        measured_repetitions_per_codec=1,
    )


def test_codec_config_requires_every_supported_choice_exactly_once() -> None:
    with pytest.raises(ValueError, match="every supported"):
        CompactPreflightCompressionBenchmarkConfig(
            codecs=("zstd",),
            warmup_repetitions_per_codec=0,
            measured_repetitions_per_codec=1,
        )
    with pytest.raises(ValueError, match="every supported"):
        CompactPreflightCompressionBenchmarkConfig(
            codecs=("none", "zstd", "zstd"),
            warmup_repetitions_per_codec=0,
            measured_repetitions_per_codec=1,
        )
    with pytest.raises(ValueError, match="measured_repetitions"):
        CompactPreflightCompressionBenchmarkConfig(
            codecs=("none", "zstd"),
            warmup_repetitions_per_codec=0,
            measured_repetitions_per_codec=0,
        )


def test_benchmark_measures_both_codecs_and_requires_scientific_equality(
    tmp_path: Path,
    portable_validators,
) -> None:
    _audit, _request, source = _seal(tmp_path)
    result = run_compact_preflight_compression_benchmark(
        config=_config(),
        source=source,
        output_root=(tmp_path / "compression_benchmark").resolve(),
    )

    assert result["accepted"] is True
    assert result["selected_parquet_compression"] in {"none", "zstd"}
    assert {
        row["parquet_compression"] for row in result["codec_results"]
    } == {"none", "zstd"}
    assert all(
        row["path_neutral_scientific_equality"] is True
        and row["deterministic_payload_inventory"] is True
        and row["median_wall_seconds"] > 0
        and row["median_cpu_seconds"] >= 0
        for row in result["codec_results"]
    )
    observations = result["measured_observations"]
    assert len(observations) == 2
    assert all(
        set(row["logical_byte_counters"])
        == {
            "read",
            "written",
            "copied",
            "hashed",
            "compressed",
            "decompressed",
            "json_encoded",
            "json_decoded",
            "fsynced",
        }
        for row in observations
    )
    by_codec = {
        row["parquet_compression"]: row for row in observations
    }
    assert by_codec["none"]["logical_byte_counters"]["compressed"] == 0
    assert by_codec["zstd"]["logical_byte_counters"]["compressed"] > 0
    assert validate_compact_preflight_compression_benchmark_result(
        result,
    ) == result

    changed = copy.deepcopy(result)
    changed["measured_observations"][0][
        "path_neutral_scientific_content_sha256"
    ] = "0" * 64
    changed["measured_observations"][0]["scientifically_equal_to_source"] = False
    row = changed["measured_observations"][0]
    row_body = {
        key: value for key, value in row.items() if key != "content_sha256"
    }
    row["content_sha256"] = identity_sha256(row_body)
    body = {
        key: value for key, value in changed.items() if key != "content_sha256"
    }
    changed["content_sha256"] = identity_sha256(body)
    with pytest.raises(ValueError, match="observation is invalid"):
        validate_compact_preflight_compression_benchmark_result(
            changed,
            reopen_artifacts=False,
        )
