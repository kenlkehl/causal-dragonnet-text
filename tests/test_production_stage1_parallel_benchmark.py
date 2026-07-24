from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import oci.inference.production_stage1_parallel_benchmark as benchmark
from oci.inference.production_stage1_parallel_benchmark import (
    BENCHMARK_TERMINAL_NAME,
    Stage1ParallelBenchmarkOptions,
    load_fixture_plan,
    run_stage1_parallel_benchmark,
    validate_stage1_parallel_benchmark,
)


def _small_fixture(path: Path) -> Path:
    raw = dict(
        benchmark.make_fixture_plan(
            job_count=8,
            workload_scale=1,
            seed=42,
        )
    )
    body = {key: value for key, value in raw.items() if key != "content_sha256"}
    for row in body["cluster_preflight_jobs"]:
        row["sample_count"] = 32
        row["feature_count"] = 6
        row["cluster_count"] = 3
    for row in body["tfidf_jobs"]:
        row["document_count"] = 32
        row["vocabulary_size"] = 32
        row["topic_count"] = 3
    path.write_text(
        json.dumps(
            {**body, "content_sha256": benchmark._sha256_json(body)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path.resolve(strict=True)


def test_fixture_benchmark_is_exact_bounded_and_terminal(
    tmp_path: Path,
) -> None:
    fixture = _small_fixture(tmp_path / "fixture.json")
    output = (tmp_path / "output").resolve()

    summary = run_stage1_parallel_benchmark(
        Stage1ParallelBenchmarkOptions(
            output_root=output,
            fixture_plan_path=fixture,
        )
    )

    assert summary["status"] == "accepted"
    assert summary["family_order"] == [
        "cluster_preflight_fixture",
        "tfidf_fixture",
    ]
    for family in summary["family_order"]:
        result = json.loads((output / f"{family}.json").read_text(encoding="utf-8"))
        assert result["required_worker_counts"] == [1, 4, 8]
        assert [row["requested_workers"] for row in result["runs"]] == [1, 4, 8]
        assert len({row["scientific_identity_sha256"] for row in result["runs"]}) == 1
        assert all(
            cap["all_observed_pools_limited_to_one"]
            and all(pool["num_threads"] == 1 for pool in cap["effective_pools"])
            for run in result["runs"]
            for cap in run["native_thread_caps"]
        )
    request = json.loads((output / "benchmark_request.json").read_text(encoding="utf-8"))
    assert request["worker_counts"] == [1, 4, 8]
    assert request["oracle_input_supplied"] is False
    assert request["cohort_dataset_path_supplied"] is False
    assert request["fixture_input_identity"]["file_sha256"] == (benchmark._sha256_file(fixture))
    terminal = output / BENCHMARK_TERMINAL_NAME
    assert terminal.stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file() and path != terminal
    )
    assert validate_stage1_parallel_benchmark(output) == summary


def test_terminal_validation_rejects_changed_and_extra_files(
    tmp_path: Path,
) -> None:
    fixture = _small_fixture(tmp_path / "fixture.json")
    output = (tmp_path / "output").resolve()
    run_stage1_parallel_benchmark(
        Stage1ParallelBenchmarkOptions(
            output_root=output,
            fixture_plan_path=fixture,
        )
    )
    result = output / "tfidf_fixture.json"
    original = result.read_bytes()
    os.chmod(result, 0o644)
    result.write_text(
        result.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact changed"):
        validate_stage1_parallel_benchmark(output)
    result.write_bytes(original)
    (output / "extra.txt").write_text("not registered", encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered"):
        validate_stage1_parallel_benchmark(output)


def test_fixture_plan_rejects_forbidden_fields_and_nonproduction_counts(
    tmp_path: Path,
) -> None:
    fixture = _small_fixture(tmp_path / "fixture.json")
    value = json.loads(fixture.read_text(encoding="utf-8"))
    body = {key: child for key, child in value.items() if key != "content_sha256"}
    body["true_ite_prob"] = "forbidden"
    fixture.write_text(
        json.dumps({**body, "content_sha256": benchmark._sha256_json(body)}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden benchmark input"):
        load_fixture_plan(fixture)
    with pytest.raises(ValueError, match="fixed at 1, 4, and 8"):
        Stage1ParallelBenchmarkOptions(
            output_root=(tmp_path / "output").resolve(),
            fixture_plan_path=fixture,
            worker_counts=(1, 2),
        )


def test_authenticated_preflight_set_loader_preserves_canonical_selection(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "preflight").resolve()
    root.mkdir()
    scope_ids = ["outer_001_full", "outer_002_full", "outer_003_full"]
    rows = []
    for index, scope_id in enumerate(scope_ids):
        scope_root = root / scope_id
        scope_root.mkdir()
        child_body = {
            "schema_version": "production_stage1_preflight_scope_input_v2",
            "scope": {"scope_id": scope_id},
            "sentinel": index,
        }
        child = {
            **child_body,
            "content_sha256": benchmark._sha256_json(child_body),
        }
        path = scope_root / "preflight_scope_input_manifest.json"
        path.write_text(
            json.dumps(child, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "scope_id": scope_id,
                "manifest": {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": benchmark._sha256_file(path),
                },
            }
        )
    body = {
        "schema_version": "production_stage1_preflight_scope_input_set_v2",
        "registry_content_sha256": "a" * 64,
        "scope_order": scope_ids,
        "scope_count": len(scope_ids),
        "scopes": rows,
        "one_scope_per_worker_payload": True,
    }
    set_manifest = root / "preflight_scope_input_set_manifest.json"
    set_manifest.write_text(
        json.dumps(
            {**body, "content_sha256": benchmark._sha256_json(body)},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    payloads, identity = benchmark._load_preflight_payloads(
        set_manifest,
        selected_scope_ids=("outer_003_full", "outer_001_full"),
    )

    assert [row["scope_id"] for row in payloads] == [
        "outer_001_full",
        "outer_003_full",
    ]
    assert identity["selected_scope_order"] == [
        "outer_001_full",
        "outer_003_full",
    ]
    assert identity["set_manifest_file_sha256"] == benchmark._sha256_file(set_manifest)
    child = root / "outer_003_full" / "preflight_scope_input_manifest.json"
    child.write_text(child.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        benchmark._load_preflight_payloads(set_manifest)


def test_fixture_plan_cli_writes_a_closed_plan(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "fixture.json").resolve()
    assert (
        benchmark.main(
            [
                "--write-fixture-plan",
                str(path),
                "--fixture-job-count",
                "8",
                "--fixture-workload-scale",
                "1",
                "--seed",
                "71",
            ]
        )
        == 0
    )
    plan = load_fixture_plan(path)
    assert plan["seed"] == 71
    assert len(plan["cluster_preflight_jobs"]) == 8
    assert len(plan["tfidf_jobs"]) == 8
