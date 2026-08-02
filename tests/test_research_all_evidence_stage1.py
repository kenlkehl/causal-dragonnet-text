from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import oci.inference.research_all_evidence_stage1 as stage1_workflow

from oci.inference.research_all_evidence_stage1 import (
    COMPONENT_ORDER,
    ResearchAllEvidenceStage1,
    Stage1RunContext,
    _stage1_context_specs,
    build_parser,
    compile_config,
    iter_stage1_handoff,
)


def _inputs(tmp_path: Path, *, components=COMPONENT_ORDER):
    dataset = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "patient_id": ["a", "b", "c", "d", "e", "f"],
            "clinical_text": ["one", "two", "three", "four", "five", "six"],
            "treatment_indicator": [0, 1, 0, 1, 0, 1],
            "outcome_indicator": [0, 0, 1, 1, 0, 1],
        }
    ).to_parquet(dataset, index=False)
    stage1_template = tmp_path / "stage1.json"
    stage1_template.write_text(
        json.dumps(
            {
                "config": {
                    "architecture": {
                        "model_type": "multi_model_forest",
                        "multi_model_forest": {
                            "feature_discovery_methods": [
                                "bow",
                                "tfidf_topic_contrast",
                            ]
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    query_template = tmp_path / "queries.json"
    query_template.write_text(
        json.dumps({"query_inner_folds": 2}),
        encoding="utf-8",
    )
    raw = {
        "dataset": str(dataset),
        "output_dir": str(tmp_path / "output"),
        "stage1_template": str(stage1_template),
        "neural_query_template": str(query_template),
        "science": {
            "outer_folds": 2,
            "inner_folds": 2,
            "seed": 7,
            "stage1": {"training": {"epochs": 3}},
        },
        "run": {"components": list(components), "workers": 2},
        "models": {
            "htr": "test-htr-model",
            "embeddings": "test-embedding-model",
        },
    }
    return raw, compile_config(raw, config_dir=tmp_path)


def test_compile_config_keeps_run_and_science_in_one_file(tmp_path):
    _raw, config = _inputs(tmp_path, components=("text_models",))

    assert config.dataset == tmp_path / "cohort.parquet"
    assert config.output_dir == tmp_path / "output"
    assert config.outer_folds == 2
    assert config.inner_folds == 2
    assert config.seed == 7
    assert config.workers == 2
    assert config.components == ("text_models",)
    assert config.stage1_overrides["training"]["epochs"] == 3


def test_resolved_context_syncs_legacy_embedding_config_alias(tmp_path):
    raw, _config = _inputs(tmp_path, components=())
    raw["science"]["stage1"]["architecture"] = {
        "multi_model_forest": {
            "embedding_contrast": {"max_chunks": 128},
        }
    }
    config = compile_config(raw, config_dir=tmp_path)

    context = ResearchAllEvidenceStage1(config)._resolved_context()
    architecture = context.applied_config.architecture

    assert architecture.model_type == "multi_model_forest"
    assert architecture.multi_model_forest.embedding_contrast.max_chunks == 128
    assert architecture.multi_model_agentic_forest.embedding_contrast.max_chunks == 128
    assert architecture.multi_model_agentic_forest.embedding_contrast.cache_dir == str(
        config.output_dir / "components" / "embedding_cache" / "cache"
    )
    assert (
        architecture.multi_model_agentic_forest.embedding_contrast.model_name
        == "test-embedding-model"
    )


def test_completed_components_are_skipped_without_revalidation(tmp_path):
    _raw, config = _inputs(tmp_path)
    calls: list[str] = []

    def runner(name):
        def run(_context, component_dir):
            calls.append(name)
            artifact = component_dir / "result.txt"
            artifact.write_text(name, encoding="utf-8")
            return {"artifacts": [str(artifact)]}

        return run

    runners = {name: runner(name) for name in COMPONENT_ORDER}
    workflow = ResearchAllEvidenceStage1(config, component_runners=runners)
    first = workflow.run()

    assert first["status"] == "complete"
    assert calls == list(COMPONENT_ORDER)
    for name in COMPONENT_ORDER:
        assert (workflow._component_dir(name) / "complete.json").is_file()

    calls.clear()
    second = workflow.run()
    assert second["status"] == "complete"
    assert calls == []
    assert all(record["status"] == "skipped" for record in second["components"].values())

    (workflow._component_dir("tfidf") / "complete.json").unlink()
    third = workflow.run()
    assert third["status"] == "complete"
    assert calls == ["tfidf"]


def test_partial_component_is_reentered_after_interruption(tmp_path):
    _raw, config = _inputs(
        tmp_path,
        components=("embedding_cache", "text_models"),
    )
    calls: list[str] = []

    def embedding(_context, component_dir):
        calls.append("embedding_cache")
        (component_dir / "cache.npy").write_bytes(b"cache")

    def fail_text(_context, component_dir):
        calls.append("text_models_failed")
        (component_dir / "partial.txt").write_text("keep me", encoding="utf-8")
        raise RuntimeError("interrupted model")

    workflow = ResearchAllEvidenceStage1(
        config,
        component_runners={
            "embedding_cache": embedding,
            "text_models": fail_text,
        },
    )
    with pytest.raises(RuntimeError, match="interrupted model"):
        workflow.run()

    progress = json.loads(workflow.progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert (workflow._component_dir("embedding_cache") / "complete.json").is_file()
    assert not (workflow._component_dir("text_models") / "complete.json").exists()

    def finish_text(_context, component_dir):
        calls.append("text_models_resumed")
        assert (component_dir / "partial.txt").read_text(encoding="utf-8") == "keep me"
        (component_dir / "result.txt").write_text("done", encoding="utf-8")

    resumed = ResearchAllEvidenceStage1(
        config,
        component_runners={
            "embedding_cache": embedding,
            "text_models": finish_text,
        },
    ).run()
    assert resumed["status"] == "complete"
    assert calls == ["embedding_cache", "text_models_failed", "text_models_resumed"]


def test_stage2_handoff_reader_uses_the_obvious_output_path(tmp_path):
    _raw, config = _inputs(tmp_path, components=())
    handoff = config.output_dir / "handoff" / "evidence.jsonl"
    handoff.parent.mkdir(parents=True)
    handoff.write_text(
        json.dumps({"source": "tfidf", "outer_fold": 1, "evidence": {"x": 1}}) + "\n",
        encoding="utf-8",
    )

    assert list(iter_stage1_handoff(config.output_dir)) == [
        {"source": "tfidf", "outer_fold": 1, "evidence": {"x": 1}}
    ]


def test_other_evidence_families_reuse_visible_tfidf_splits(tmp_path):
    _raw, config = _inputs(tmp_path, components=("tfidf", "text_models"))
    split_path = config.output_dir / "components" / "tfidf" / "split_provenance.jsonl"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(
        json.dumps(
            {
                "outer_fold": 1,
                "fit_row_ids": [5, 3, 1, 0],
                "heldout_row_ids": [2, 4],
                "inner_splits": [
                    {
                        "inner_fold": 1,
                        "fit_row_ids": [5, 1],
                        "heldout_row_ids": [3, 0],
                    },
                    {
                        "inner_fold": 2,
                        "fit_row_ids": [3, 0],
                        "heldout_row_ids": [5, 1],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = pd.read_parquet(config.dataset)
    context = Stage1RunContext(
        config=config,
        dataset=dataset,
        applied_config=None,
        neural_query_config=None,
    )

    specs = _stage1_context_specs(context)

    assert [spec["scope_id"] for spec in specs] == [
        "outer_001_full",
        "outer_001_inner_001",
        "outer_001_inner_002",
    ]
    assert specs[0]["train_idx"].tolist() == [5, 3, 1, 0]
    assert specs[1]["train_idx"].tolist() == [5, 1]


def test_text_model_contexts_are_independently_resumable(tmp_path, monkeypatch):
    _raw, config = _inputs(tmp_path, components=("text_models",))
    config = replace(config, workers=1)
    context = Stage1RunContext(
        config=config,
        dataset=pd.read_parquet(config.dataset),
        applied_config=None,
        neural_query_config=None,
    )
    calls: list[str] = []

    def fit_one(*, spec, context_dir, **_unused):
        calls.append(str(spec["scope_id"]))
        context_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "outer_fold": spec["outer_fold"],
            "inner_fold": spec["inner_fold"],
            "scope": spec["scope"],
        }
        (context_dir / "evidence.json").write_text(json.dumps(row), encoding="utf-8")
        (context_dir / "complete.json").write_text("{}", encoding="utf-8")
        return row

    monkeypatch.setattr(stage1_workflow, "_run_one_text_model_context", fit_one)
    component_dir = context.component_dir("text_models")
    first = stage1_workflow._text_models_component(context, component_dir)
    expected_contexts = 2 * (1 + 2)

    assert first["contexts"] == expected_contexts
    assert len(calls) == expected_contexts

    calls.clear()
    second = stage1_workflow._text_models_component(context, component_dir)
    assert second["contexts"] == expected_contexts
    assert calls == []

    marker = component_dir / "outer_001_inner_001" / "complete.json"
    marker.unlink()
    stage1_workflow._text_models_component(context, component_dir)
    assert calls == ["outer_001_inner_001"]


def test_phase_flags_are_mutually_exclusive():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--stage1-only", "--stage2-only"])


def test_stage2_only_runs_from_saved_handoff_and_resumes(tmp_path):
    raw, _config = _inputs(tmp_path, components=())
    raw["run"]["mode"] = "stage2"
    raw["stage2"] = {
        "command": [
            sys.executable,
            "-c",
            (
                "import os; from pathlib import Path; "
                "Path(os.environ['OCI_STAGE2_OUTPUT'], 'result.txt').write_text('done')"
            ),
        ]
    }
    config = compile_config(raw, config_dir=tmp_path)
    handoff = config.output_dir / "handoff" / "evidence.jsonl"
    handoff.parent.mkdir(parents=True)
    handoff.write_text('{"source":"tfidf"}\n', encoding="utf-8")
    (handoff.parent / "complete.json").write_text("{}", encoding="utf-8")
    workflow = ResearchAllEvidenceStage1(config)

    first = workflow.run()
    assert first["mode"] == "stage2"
    assert (config.output_dir / "stage2" / "result.txt").read_text() == "done"
    assert (config.output_dir / "stage2" / "complete.json").is_file()

    second = workflow.run()
    assert second["components"]["stage2"]["status"] == "skipped"


def test_stage2_command_makes_full_run_the_default(tmp_path):
    raw, _config = _inputs(tmp_path)
    raw["stage2"] = {"command": ["stage2-program", "--handoff", "{handoff}"]}

    config = compile_config(raw, config_dir=tmp_path)

    assert config.mode == "full"
    assert config.components == (*COMPONENT_ORDER, "stage2")
