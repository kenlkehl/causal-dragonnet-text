import json
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from joblib import Parallel, delayed, parallel_config

import oci.inference.tfidf_topic_stage1 as tfidf_topic_stage1_module
from oci.config import AppliedInferenceConfig, ModelArchitectureConfig, MultiModelForestConfig
from oci.inference.tfidf_topic_discovery import row_set_fingerprint, stable_hash
from oci.inference.tfidf_topic_stage1 import (
    _build_tfidf_worker_context_spec,
    _fit_tfidf_topic_stage1_spec,
    _resolve_tfidf_topic_stage1_parallel_backend,
    _tfidf_context_scope_seed,
)
from scripts.run_tfidf_topic_stage1_from_primary_splits import (
    _apply_fork_guard_environment,
    _fork_backend_requested,
    build_config,
    parse_args,
)


def _config(backend: str = "multiprocessing") -> AppliedInferenceConfig:
    return AppliedInferenceConfig(
        seed=42,
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                feature_discovery_methods=["bow", "tfidf_topic_contrast"],
                outer_parallel_backend=backend,
            ),
        ),
    )


def test_parallel_backend_config_preserves_loky_and_canonicalizes_fork():
    assert _config("threads").architecture.multi_model_forest.outer_parallel_backend == "threads"
    assert _config("processes").architecture.multi_model_forest.outer_parallel_backend == (
        "processes"
    )
    assert _config("loky").architecture.multi_model_forest.outer_parallel_backend == "processes"
    assert _config("fork").architecture.multi_model_forest.outer_parallel_backend == (
        "multiprocessing"
    )


def test_stage1_cli_accepts_fork_alias_and_builds_canonical_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tfidf_topic_stage1_from_primary_splits.py",
            "--dataset",
            str(tmp_path / "dataset.parquet"),
            "--primary-predictions",
            str(tmp_path / "primary.parquet"),
            "--output-dir",
            str(tmp_path / "output"),
            "--parallel-backend",
            "fork",
        ],
    )
    args = parse_args()
    config = build_config(args, tmp_path / "split_registry.json")

    assert args.parallel_backend == "fork"
    assert config.architecture.multi_model_forest.outer_parallel_backend == "multiprocessing"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--parallel-backend", "fork"], True),
        (["--parallel-backend=multiprocessing"], True),
        (["--parallel-backend", "threads"], False),
        ([], False),
    ],
)
def test_stage1_cli_preimport_fork_guard_detection(argv, expected):
    assert _fork_backend_requested(argv) is expected


def test_stage1_cli_fork_guard_preserves_user_environment():
    environment = {"OMP_NUM_THREADS": "7"}
    effective = _apply_fork_guard_environment(environment)

    assert effective["OMP_NUM_THREADS"] == "7"
    assert effective["OPENBLAS_NUM_THREADS"] == "1"
    assert effective["ARROW_DEFAULT_MEMORY_POOL"] == "system"
    assert effective["MALLOC_CONF"] == "background_thread:false"


def test_parallel_backend_mapping_keeps_existing_semantics(monkeypatch):
    assert _resolve_tfidf_topic_stage1_parallel_backend("threads") == (
        "threads",
        "threading",
    )
    assert _resolve_tfidf_topic_stage1_parallel_backend("processes") == (
        "processes",
        "loky",
    )
    assert _resolve_tfidf_topic_stage1_parallel_backend("loky") == (
        "processes",
        "loky",
    )
    monkeypatch.setattr("oci.inference.tfidf_topic_stage1.sys.platform", "linux")
    monkeypatch.setattr(
        "oci.inference.tfidf_topic_stage1.mp.get_start_method",
        lambda allow_none=True: "fork",
    )
    assert _resolve_tfidf_topic_stage1_parallel_backend("fork") == (
        "multiprocessing",
        "multiprocessing",
    )


def test_nested_production_worker_spec_serializes_no_heldout_labels():
    config = _config("processes")
    config.architecture.multi_model_forest.tfidf_topic.score_selection_label_policy = (
        "nested_fit_calibration"
    )
    fit = pd.DataFrame(
        {
            "_oci_row_id": [0, 1],
            "clinical_text": ["fit alpha", "fit beta"],
            "treatment_indicator": [0, 1],
            "outcome_indicator": [0, 1],
        }
    )
    heldout = pd.DataFrame(
        {
            "_oci_row_id": [2, 3],
            "clinical_text": ["heldout gamma", "heldout delta"],
            "treatment_indicator": [0, 1],
            "outcome_indicator": [1, 0],
        }
    )
    mutated = heldout.copy()
    mutated["treatment_indicator"] = [1, 0]
    mutated["outcome_indicator"] = [0, 1]

    first = _build_tfidf_worker_context_spec(
        outer_fold=1,
        inner_fold=1,
        scope="candidate_selection_inner_fit",
        fold_key=1001,
        fit_df=fit,
        heldout_df=heldout,
        scope_id="outer_001_inner_001",
        config=config,
    )
    second = _build_tfidf_worker_context_spec(
        outer_fold=1,
        inner_fold=1,
        scope="candidate_selection_inner_fit",
        fold_key=1001,
        fit_df=fit,
        heldout_df=mutated,
        scope_id="outer_001_inner_001",
        config=config,
    )

    assert list(first["heldout_df"].columns) == [
        "_oci_row_id",
        "clinical_text",
    ]
    assert first["registered_heldout_labels_serialized"] is False
    pd.testing.assert_frame_equal(first["heldout_df"], second["heldout_df"])
    serialized = json.dumps(
        first["heldout_df"].to_dict(orient="list"),
        sort_keys=True,
    )
    assert "treatment_indicator" not in serialized
    assert "outcome_indicator" not in serialized


def test_multiprocessing_backend_fails_closed_without_linux_fork(monkeypatch):
    monkeypatch.setattr("oci.inference.tfidf_topic_stage1.sys.platform", "linux")
    monkeypatch.setattr(
        "oci.inference.tfidf_topic_stage1.mp.get_start_method",
        lambda allow_none=True: "spawn",
    )
    with pytest.raises(ValueError, match="requires Linux.*'fork'"):
        _resolve_tfidf_topic_stage1_parallel_backend("multiprocessing")


def _cached_spec(
    base_dir: Path,
    *,
    scope_id: str,
    config: AppliedInferenceConfig,
    stage1_hash: str,
    dataset_identity: dict,
    split_semantics_hash: str,
) -> dict:
    context_dir = base_dir / scope_id
    context_dir.mkdir(parents=True)
    fit_df = pd.DataFrame(
        {
            "_oci_row_id": [0, 1],
            "clinical_text": ["baseline alpha", "baseline beta"],
            "treatment_indicator": [0, 1],
            "outcome_indicator": [0, 1],
        }
    )
    heldout_df = pd.DataFrame(
        {
            "_oci_row_id": [2],
            "clinical_text": ["baseline gamma"],
            "treatment_indicator": [0],
            "outcome_indicator": [1],
        }
    )
    artifact_paths = {}
    for artifact_name in (
        "fitted_context",
        "fit_topic_values",
        "heldout_topic_values",
        "nuisance_predictions",
    ):
        artifact_path = context_dir / f"{artifact_name}.placeholder"
        artifact_path.write_text("sealed\n", encoding="utf-8")
        artifact_paths[artifact_name] = str(artifact_path)
    ngram_score_paths = {}
    for bank in ("treatment", "outcome", "effect"):
        artifact_path = context_dir / f"{bank}_ngram_scores.placeholder"
        artifact_path.write_text("sealed\n", encoding="utf-8")
        ngram_score_paths[bank] = str(artifact_path)
    artifact_paths["ngram_scores"] = ngram_score_paths
    artifact_paths["topic_score_tests"] = None
    metadata = {
        "fit_row_fingerprint": row_set_fingerprint(fit_df["_oci_row_id"]),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_df["_oci_row_id"]),
        "config_hash": stable_hash(asdict(config.architecture.multi_model_forest.tfidf_topic)),
        "stage1_config_hash": stage1_hash,
        "dataset_content_fingerprint": dataset_identity["content_fingerprint"],
        "dataset_ordered_row_fingerprint": dataset_identity["ordered_row_fingerprint"],
        "split_semantics_hash": split_semantics_hash,
        "heldout_score_tests_enabled": False,
        "fit_row_ids": [0, 1],
        "heldout_row_ids": [2],
        "artifact_inventory": {"fixture": "closed"},
        "artifacts": artifact_paths,
    }
    (context_dir / "context_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return {
        "scope_id": scope_id,
        "scope": "full_outer_train",
        "fit_df": fit_df,
        "heldout_df": heldout_df,
        "worker_scope_seed": _tfidf_context_scope_seed(
            global_seed=int(getattr(config, "seed", 42)),
            scope_id=scope_id,
        ),
    }


@pytest.mark.skipif(
    not __import__("sys").platform.startswith("linux")
    or mp.get_context().get_start_method() != "fork",
    reason="joblib multiprocessing Stage1 backend is intentionally Linux-fork-only",
)
def test_pickle_safe_context_worker_runs_in_joblib_multiprocessing(
    monkeypatch,
    tmp_path,
):
    config = _config("multiprocessing")
    config_hash = stable_hash(
        asdict(config.architecture.multi_model_forest.tfidf_topic)
    )
    monkeypatch.setattr(
        tfidf_topic_stage1_module,
        "tfidf_context_artifact_inventory",
        lambda _artifacts: {"fixture": "closed"},
    )
    monkeypatch.setattr(
        tfidf_topic_stage1_module,
        "load_fitted_topic_context",
        lambda _path: SimpleNamespace(config_hash=config_hash),
    )
    monkeypatch.setattr(
        tfidf_topic_stage1_module,
        "load_named_array_bank",
        lambda _path, *, expected_row_count: {},
    )
    monkeypatch.setattr(
        tfidf_topic_stage1_module.pd,
        "read_parquet",
        lambda _path: pd.DataFrame(
            {
                "_oci_row_id": [0, 1, 2],
                "prediction_scope": [
                    "fit_oof",
                    "fit_oof",
                    "external_heldout",
                ],
            }
        ),
    )
    stage1_hash = "stage1-hash"
    split_semantics_hash = "split-hash"
    dataset_identity = {
        "content_fingerprint": "content-hash",
        "ordered_row_fingerprint": "ordered-hash",
    }
    contexts_dir = tmp_path / "contexts"
    specs = [
        _cached_spec(
            contexts_dir,
            scope_id=f"outer_{fold:03d}_full_train",
            config=config,
            stage1_hash=stage1_hash,
            dataset_identity=dataset_identity,
            split_semantics_hash=split_semantics_hash,
        )
        for fold in (1, 2)
    ]

    configured_backend, joblib_backend = _resolve_tfidf_topic_stage1_parallel_backend(
        config.architecture.multi_model_forest.outer_parallel_backend
    )
    assert configured_backend == "multiprocessing"
    with parallel_config(backend=joblib_backend, n_jobs=2):
        completed = Parallel(batch_size=1, pre_dispatch="all")(
            delayed(_fit_tfidf_topic_stage1_spec)(
                spec,
                contexts_dir=contexts_dir,
                config=config,
                stage1_hash=stage1_hash,
                dataset_identity=dataset_identity,
                split_semantics_hash=split_semantics_hash,
                split_schema_version="test-split-v1",
                limit_native_threads=True,
            )
            for spec in specs
        )

    assert [spec["scope_id"] for spec, _metadata in completed] == [
        "outer_001_full_train",
        "outer_002_full_train",
    ]
