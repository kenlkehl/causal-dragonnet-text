from dataclasses import replace
from pathlib import Path

import pytest

from oci.inference.production_all_evidence_workflow import (
    PHASES,
    ProductionAllEvidenceWorkflow,
    ProductionAllEvidenceWorkflowOptions,
    build_parser,
    options_from_args,
)


def _options(tmp_path: Path, *, endpoint="https://different.example/v1", model="different/model"):
    files = []
    for name in ("dataset.parquet", "stage1.json", "query.json", "embed", "htr"):
        path = tmp_path / name
        path.write_text(name)
        files.append(path)
    return ProductionAllEvidenceWorkflowOptions(
        dataset_path=files[0], work_root=tmp_path / "run",
        stage1_profile_path=files[1], query_profile_path=files[2],
        unit_id_column="id", text_column="text", treatment_column="a", outcome_column="y",
        outcome_type="binary", clinical_question="question",
        embedding_model_name="logical/embed", embedding_local_model_path=files[3],
        htr_local_model_path=files[4], endpoint=endpoint, model_name=model,
        stage1_device="cpu", query_device="cpu", review_device="cpu", gpu_id=0,
    )


def test_endpoint_model_and_phase_resume_are_configuration_bound(tmp_path):
    options = _options(tmp_path)
    calls = []
    overrides = {
        phase: (lambda _attempt, value=phase: calls.append(value) or {"terminal_files": []})
        for phase in PHASES
    }
    ProductionAllEvidenceWorkflow(options, phase_overrides=overrides).run()
    assert calls == list(PHASES)
    calls.clear()
    ProductionAllEvidenceWorkflow(replace(options, resume=True), phase_overrides=overrides).run()
    assert calls == []
    with pytest.raises(ValueError, match="differs"):
        ProductionAllEvidenceWorkflow(
            replace(options, resume=True, model_name="substituted/model"), phase_overrides=overrides
        ).run()


def test_cli_has_no_embedded_endpoint_or_model(tmp_path):
    o = _options(tmp_path)
    args = build_parser().parse_args([
        "--dataset", str(o.dataset_path), "--work-root", str(o.work_root),
        "--stage1-profile", str(o.stage1_profile_path), "--query-profile", str(o.query_profile_path),
        "--unit-id-column", "id", "--text-column", "note", "--treatment-column", "tx",
        "--outcome-column", "y", "--outcome-type", "binary", "--clinical-question", "q",
        "--embedding-model-name", "embed", "--embedding-local-model-path", str(o.embedding_local_model_path),
        "--htr-local-model-path", str(o.htr_local_model_path), "--endpoint", "https://fake.example/v1",
        "--model", "fake/model",
    ])
    parsed = options_from_args(args)
    assert parsed.endpoint == "https://fake.example/v1"
    assert parsed.model_name == "fake/model"

