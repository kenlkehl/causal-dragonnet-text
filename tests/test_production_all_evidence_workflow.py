from dataclasses import replace
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

import oci.inference.production_authenticated_tree_cache as tree_module
import oci.inference.production_all_evidence_workflow as workflow_module
from oci.inference.production_all_evidence_workflow import (
    EMBEDDING_CACHE_PHASE_SCHEMA,
    PHASES,
    STAGE1_PREFLIGHT_PHASE_SCHEMA,
    STAGE1_ONLY_PHASES,
    ProductionAllEvidenceWorkflow,
    ProductionAllEvidenceWorkflowHooks,
    ProductionAllEvidenceWorkflowOptions,
    build_parser,
    options_from_args,
)


def _options(tmp_path: Path, *, endpoint="https://different.example/v1", model="different/model"):
    files = []
    for name in ("dataset.parquet", "stage1.json", "query.json"):
        path = tmp_path / name
        path.write_text(name)
        files.append(path)
    embed = (tmp_path / "embed").resolve()
    embed.mkdir(exist_ok=True)
    (embed / "model.safetensors").write_bytes(b"safe embedding model")
    htr = (tmp_path / "htr").resolve()
    htr.mkdir(exist_ok=True)
    # The HTR path intentionally keeps the legacy full-byte validation path.
    (htr / "model.safetensors").write_bytes(b"safe htr model")
    return ProductionAllEvidenceWorkflowOptions(
        dataset_path=files[0],
        work_root=tmp_path / "run",
        stage1_profile_path=files[1],
        query_profile_path=files[2],
        unit_id_column="id",
        text_column="text",
        treatment_column="a",
        outcome_column="y",
        outcome_type="binary",
        clinical_question="question",
        embedding_model_name="logical/embed",
        embedding_local_model_path=embed,
        htr_local_model_path=htr,
        endpoint=endpoint,
        model_name=model,
        stage1_device="cpu",
        query_device="cpu",
        review_device="cpu",
        gpu_id=0,
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
    args = build_parser().parse_args(
        [
            "--dataset",
            str(o.dataset_path),
            "--work-root",
            str(o.work_root),
            "--stage1-profile",
            str(o.stage1_profile_path),
            "--query-profile",
            str(o.query_profile_path),
            "--unit-id-column",
            "id",
            "--text-column",
            "note",
            "--treatment-column",
            "tx",
            "--outcome-column",
            "y",
            "--outcome-type",
            "binary",
            "--clinical-question",
            "q",
            "--embedding-model-name",
            "embed",
            "--embedding-local-model-path",
            str(o.embedding_local_model_path),
            "--htr-local-model-path",
            str(o.htr_local_model_path),
            "--endpoint",
            "https://fake.example/v1",
            "--model",
            "fake/model",
        ]
    )
    parsed = options_from_args(args)
    assert parsed.endpoint == "https://fake.example/v1"
    assert parsed.model_name == "fake/model"


def test_stage1_only_needs_no_endpoint_and_stops_after_fresh_handoff_boundary(
    tmp_path,
    monkeypatch,
):
    options = replace(
        _options(tmp_path),
        endpoint=None,
        model_name=None,
        stage1_only=True,
    )
    calls = []
    overrides = {
        phase: (lambda _attempt, value=phase: calls.append(value) or {"terminal_files": []})
        for phase in STAGE1_ONLY_PHASES[:-1]
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Stage 2 construction is forbidden in Stage-1-only mode")

    monkeypatch.setattr(ProductionAllEvidenceWorkflow, "_stage2_options", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    result = ProductionAllEvidenceWorkflow(options, phase_overrides=overrides).run()

    assert calls == list(STAGE1_ONLY_PHASES[:-1])
    assert result["stage1_only"] is True
    # The override did not perform the real loader subprocess, so the terminal
    # validator must not claim that it did.
    assert result["stage1_handoff_validated_in_fresh_process"] is False
    assert result["validated_phase_sequence"] == list(STAGE1_ONLY_PHASES)
    request = json.loads(
        (options.work_root / "immutable_run_request.json").read_text(encoding="utf-8")
    )
    assert request["endpoint"] is None
    assert request["model_name"] is None
    assert request["phase_sequence"] == list(STAGE1_ONLY_PHASES)
    assert "oci.inference.production_stage1_hierarchy_one_shot" not in sys.modules
    assert "scripts.canary_production_stage1_hierarchy" not in sys.modules
    assert "openai" not in sys.modules


def test_canary_descriptor_preparation_is_an_operational_prefix_only(
    tmp_path,
    monkeypatch,
):
    loaded_root = Path(workflow_module.__file__).resolve().parents[2]
    snapshot_sha = "c" * 64
    snapshot_identity = {
        "root": str(loaded_root),
        "manifest_path": str(loaded_root / "source_snapshot_manifest.json"),
        "content_sha256": snapshot_sha,
        "file_count": 1,
    }
    fake_snapshot = SimpleNamespace(
        root=loaded_root,
        content_sha256=snapshot_sha,
        as_dict=lambda: dict(snapshot_identity),
    )
    monkeypatch.setattr(
        "oci.inference.production_source_snapshot.validate_production_source_snapshot",
        lambda _path: fake_snapshot,
    )
    monkeypatch.setenv(
        workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV,
        snapshot_sha,
    )
    options = replace(
        _options(tmp_path),
        endpoint=None,
        model_name=None,
        stage1_only=True,
        gpu_id=None,
        stage1_gpu_ids=(0, 1),
        source_snapshot_root=loaded_root,
    )
    calls = []
    prefix = ("input_preparation", "embedding_cache", "stage1_preflight")
    overrides = {
        phase: (lambda _attempt, value=phase: calls.append(value) or {"terminal_files": []})
        for phase in prefix
    }
    cache = (tmp_path / "prepared_cache").resolve()
    cache.mkdir()
    prepared_path = (tmp_path / "prepared.parquet").resolve()
    prepared_path.write_bytes(b"prepared")
    profile = (tmp_path / "effective_profile.json").resolve()
    profile.write_text("{}", encoding="utf-8")
    preflight = (tmp_path / "cluster_preflight_manifest.json").resolve()
    preflight.write_text("{}", encoding="utf-8")
    descriptor_root = (options.work_root / "recovery" / "descriptor").resolve()
    prepared = SimpleNamespace(
        request_sha256="a" * 64,
        scope_descriptor_root=descriptor_root,
    )

    class _FakeBuilder:
        def __init__(self, _options):
            pass

        def prepare(self):
            return prepared

    selected_root = descriptor_root / "outer_001_full"
    selected_manifest = selected_root / "descriptor_manifest.json"

    def publish(*, prepared, descriptor_root):
        del prepared
        descriptor_root.mkdir(parents=True)
        selected_root.mkdir()
        selected_manifest.write_text("selected", encoding="utf-8")
        set_manifest = descriptor_root / "descriptor_set_manifest.json"
        set_manifest.write_text("set", encoding="utf-8")
        selected = SimpleNamespace(
            scope_id="outer_001_full",
            scope=SimpleNamespace(scope_kind="full_outer"),
            assignment=SimpleNamespace(gpu_id=0),
            manifest_path=selected_manifest,
        )
        return SimpleNamespace(
            root=descriptor_root,
            manifest={"content_sha256": "b" * 64},
            descriptors={"outer_001_full": selected},
        )

    monkeypatch.setattr(
        workflow_module,
        "ProductionStage1BundleBuilder",
        _FakeBuilder,
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_legacy_scope_adapter."
        "publish_legacy_stage1_scope_descriptor",
        publish,
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_legacy_scope_adapter."
        "validate_legacy_stage1_scope_descriptor_set",
        lambda **kwargs: (
            publish(
                prepared=prepared,
                descriptor_root=Path(kwargs["descriptor_root"]),
            )
            if not selected_manifest.exists()
            else SimpleNamespace(
                root=descriptor_root,
                manifest={"content_sha256": "b" * 64},
                descriptors={
                    "outer_001_full": SimpleNamespace(
                        scope_id="outer_001_full",
                        scope=SimpleNamespace(scope_kind="full_outer"),
                        assignment=SimpleNamespace(gpu_id=0),
                        manifest_path=selected_manifest,
                    )
                },
            )
        ),
    )
    workflow = ProductionAllEvidenceWorkflow(
        options,
        phase_overrides=overrides,
    )
    monkeypatch.setattr(
        workflow,
        "_embedding_cache_paths",
        lambda: (cache, prepared_path),
    )
    monkeypatch.setattr(
        workflow,
        "_stage1_preflight_paths",
        lambda: (profile, preflight),
    )
    monkeypatch.setattr(
        workflow,
        "_stage1_build_options",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        workflow,
        "_validate_canary_preparation_in_fresh_process",
        lambda: json.loads(
            (
                options.work_root / "recovery" / "canary_descriptor_preparation_manifest.json"
            ).read_text(encoding="utf-8")
        ),
    )

    result = workflow.prepare_stage1_canary_descriptors_only()

    assert calls == list(prefix)
    assert result["status"] == "complete"
    assert result["selected_scope_id"] == "outer_001_full"
    assert result["selected_logical_gpu_id"] == 0
    assert result["supervised_stage1_fits_started"] is False
    assert not (options.work_root / "phases" / "stage1_modeling").exists()
    request_before = (options.work_root / "immutable_run_request.json").read_bytes()
    resumed = ProductionAllEvidenceWorkflow(
        replace(options, resume=True),
        phase_overrides=overrides,
    )
    resumed._initialize()
    assert (options.work_root / "immutable_run_request.json").read_bytes() == request_before


def test_canary_descriptor_preparation_requires_source_snapshot(tmp_path):
    options = replace(
        _options(tmp_path),
        endpoint=None,
        model_name=None,
        stage1_only=True,
        gpu_id=None,
        stage1_gpu_ids=(0, 1),
    )
    workflow = ProductionAllEvidenceWorkflow(options)

    with pytest.raises(ValueError, match="requires one authenticated source snapshot"):
        workflow.prepare_stage1_canary_descriptors_only()

    assert not options.work_root.exists()


def test_full_workflow_still_requires_endpoint_and_model(tmp_path):
    with pytest.raises(ValueError, match="requires one endpoint"):
        ProductionAllEvidenceWorkflow(replace(_options(tmp_path), endpoint=None, model_name=None))


def test_cli_binds_ordered_plural_gpus_and_worker_contract(tmp_path):
    o = _options(tmp_path)
    args = build_parser().parse_args(
        [
            "--dataset",
            str(o.dataset_path),
            "--work-root",
            str(o.work_root),
            "--stage1-profile",
            str(o.stage1_profile_path),
            "--query-profile",
            str(o.query_profile_path),
            "--unit-id-column",
            "id",
            "--text-column",
            "note",
            "--treatment-column",
            "tx",
            "--outcome-column",
            "y",
            "--outcome-type",
            "binary",
            "--clinical-question",
            "q",
            "--embedding-model-name",
            "embed",
            "--embedding-local-model-path",
            str(o.embedding_local_model_path),
            "--htr-local-model-path",
            str(o.htr_local_model_path),
            "--stage1-only",
            "--stage1-gpu-id",
            "0",
            "--stage1-gpu-id",
            "1",
            "--query-device",
            "cuda:0",
            "--query-device",
            "cuda:1",
            "--stage1-scope-workers-per-gpu",
            "1",
            "--stage1-preflight-workers",
            "8",
        ]
    )
    parsed = options_from_args(args)
    workflow = ProductionAllEvidenceWorkflow(parsed)
    assert workflow.stage1_gpu_ids == (0, 1)
    assert workflow.query_devices == ("cuda:0", "cuda:1")
    assert parsed.stage1_scope_workers_per_gpu == 1
    assert parsed.stage1_preflight_workers == 8


def test_singular_gpu_alias_is_accepted_but_conflicts_are_rejected(tmp_path):
    options = _options(tmp_path)
    assert ProductionAllEvidenceWorkflow(options).stage1_gpu_ids == (0,)
    with pytest.raises(ValueError, match="conflicts"):
        ProductionAllEvidenceWorkflow(replace(options, gpu_id=1, stage1_gpu_ids=(0, 1)))


def test_gpu_preflight_checks_every_requested_gpu(monkeypatch, tmp_path):
    workflow = ProductionAllEvidenceWorkflow(
        replace(
            _options(tmp_path),
            gpu_id=None,
            stage1_gpu_ids=(0, 1),
            stage1_device="cuda:1",
        )
    )

    def fake_run(command, **_kwargs):
        if any(value.startswith("--query-gpu=") for value in command):
            return SimpleNamespace(stdout="0, GPU-a, 49140, 15, 0\n1, GPU-b, 49140, 188, 0\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(
        "oci.inference.production_all_evidence_workflow.subprocess.run",
        fake_run,
    )
    report = workflow._gpu_preflight()
    assert report["requested_gpu_ids"] == [0, 1]
    assert report["gpu_uuids"] == {"0": "GPU-a", "1": "GPU-b"}


def test_gpu_preflight_rejects_occupancy_on_either_gpu(monkeypatch, tmp_path):
    workflow = ProductionAllEvidenceWorkflow(
        replace(
            _options(tmp_path),
            gpu_id=None,
            stage1_gpu_ids=(0, 1),
            stage1_device="cuda:1",
        )
    )

    def fake_run(command, **_kwargs):
        if any(value.startswith("--query-gpu=") for value in command):
            return SimpleNamespace(stdout="0, GPU-a, 49140, 15, 0\n1, GPU-b, 49140, 188, 0\n")
        return SimpleNamespace(stdout="GPU-b, 999999, 1024\n")

    monkeypatch.setattr(
        "oci.inference.production_all_evidence_workflow.subprocess.run",
        fake_run,
    )
    with pytest.raises(RuntimeError, match="not exclusively available"):
        workflow._gpu_preflight()


def test_gpu_preflight_rejects_large_unreported_memory_occupant(monkeypatch, tmp_path):
    workflow = ProductionAllEvidenceWorkflow(
        replace(
            _options(tmp_path),
            gpu_id=None,
            stage1_gpu_ids=(0, 1),
            stage1_device="cuda:1",
        )
    )

    def fake_run(command, **_kwargs):
        if any(value.startswith("--query-gpu=") for value in command):
            return SimpleNamespace(stdout="0, GPU-a, 49140, 15, 0\n1, GPU-b, 49140, 12000, 0\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(
        "oci.inference.production_all_evidence_workflow.subprocess.run",
        fake_run,
    )
    with pytest.raises(RuntimeError, match="unexpected_idle_state"):
        workflow._gpu_preflight()


def test_cuda_devices_must_be_covered_by_exclusive_gpu_ids(tmp_path):
    with pytest.raises(ValueError, match="included in the exclusive"):
        ProductionAllEvidenceWorkflow(
            replace(
                _options(tmp_path),
                gpu_id=None,
                stage1_gpu_ids=(0,),
                stage1_device="cuda:1",
                query_device="cuda:0",
            )
        )
    with pytest.raises(ValueError, match="included in the exclusive"):
        ProductionAllEvidenceWorkflow(
            replace(
                _options(tmp_path),
                gpu_id=None,
                stage1_gpu_ids=(0,),
                stage1_device="cuda:0",
                query_device=None,
                query_devices=("cuda:1",),
            )
        )


def test_cache_import_rejects_partial_explicit_source_preparation(tmp_path):
    with pytest.raises(ValueError, match="requires both"):
        ProductionAllEvidenceWorkflow(
            replace(
                _options(tmp_path),
                embedding_cache_import=tmp_path / "cache",
                embedding_cache_import_source_prepared_path=tmp_path / "prepared.parquet",
            )
        )


def test_cache_import_can_discover_its_authenticated_source_preparation(tmp_path):
    options = _options(tmp_path)
    source = tmp_path / "source_prepared"
    source.mkdir()
    prepared = source / "modeling_cohort.parquet"
    manifest = source / "preparation_manifest.json"
    prepared.write_bytes(b"prepared")
    manifest.write_text("{}", encoding="utf-8")
    cache = tmp_path / "source_cache"
    cache.mkdir()
    (cache / "metadata.json").write_text(
        json.dumps({"production_provenance": {"dataset": {"path": str(prepared.resolve())}}}),
        encoding="utf-8",
    )
    workflow = ProductionAllEvidenceWorkflow(replace(options, embedding_cache_import=cache))
    assert workflow._resolved_cache_import_sources() == (
        prepared.resolve(),
        manifest.resolve(),
    )


def test_parallel_cache_preflight_and_modeling_hooks_receive_immutable_context(
    tmp_path,
):
    options = replace(
        _options(tmp_path),
        endpoint=None,
        model_name=None,
        stage1_only=True,
        gpu_id=None,
        stage1_gpu_ids=(),
        stage1_device="cpu",
        query_device="cpu",
        stage1_scope_workers_per_gpu=1,
        stage1_preflight_workers=8,
    )
    observed = {}

    def prepare(attempt):
        prepared = attempt / "prepared"
        prepared.mkdir()
        cohort = prepared / "modeling_cohort.parquet"
        manifest = prepared / "preparation_manifest.json"
        cohort.write_bytes(b"cohort")
        manifest.write_text("{}", encoding="utf-8")
        return {
            "output": {"path": str(cohort)},
            "terminal_files": [str(cohort), str(manifest)],
        }

    def cache(attempt, context):
        observed["cache"] = context
        cache_dir = attempt / "embedding_cache"
        prepared_dir = attempt / "prepared"
        cache_dir.mkdir()
        prepared_dir.mkdir()
        cache_file = cache_dir / "metadata.json"
        cache_file.write_text("{}", encoding="utf-8")
        cohort = prepared_dir / "modeling_cohort.parquet"
        cohort.write_bytes(b"cohort")
        return {
            "schema_version": EMBEDDING_CACHE_PHASE_SCHEMA,
            "cache_path": str(cache_dir),
            "prepared_cohort_path": str(cohort),
            "cache_identity": {"test_identity": True},
            "terminal_files": [str(cache_file), str(cohort)],
        }

    def preflight(attempt, context):
        observed["preflight"] = context
        profile = attempt / "effective_stage1_profile.json"
        artifact = attempt / "cluster_preflight" / "cluster_preflight_manifest.json"
        artifact.parent.mkdir()
        profile.write_text("{}", encoding="utf-8")
        artifact.write_text("{}", encoding="utf-8")
        return {
            "schema_version": STAGE1_PREFLIGHT_PHASE_SCHEMA,
            "effective_profile_path": str(profile),
            "cluster_preflight_manifest_path": str(artifact),
            "terminal_files": [str(profile), str(artifact)],
        }

    def modeling(attempt, context):
        observed["modeling"] = context
        manifest = attempt / "bundle_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return {"terminal_files": [str(manifest)]}

    workflow = ProductionAllEvidenceWorkflow(
        options,
        phase_overrides={
            "input_preparation": prepare,
            "handoff_validation": lambda _attempt: {"terminal_files": []},
        },
        hooks=ProductionAllEvidenceWorkflowHooks(
            embedding_cache=cache,
            stage1_preflight=preflight,
            stage1_modeling=modeling,
        ),
    )
    result = workflow.run()
    assert result["stage1_only"] is True
    for phase in ("cache", "preflight", "modeling"):
        assert observed[phase]["request_sha256"]
        assert observed[phase]["stage1_scope_workers_per_gpu"] == 1
        assert observed[phase]["stage1_preflight_workers"] == 8
        assert observed[phase]["resource_preflight"]["requested_gpu_ids"] == []
        assert observed[phase]["stage1_scope_attempt_root"].endswith(
            "/recovery/stage1_scope_attempts"
        )
        assert observed[phase]["stage1_scope_progress_path"].endswith(
            "/recovery/stage1_scope_progress.json"
        )
    assert observed["modeling"]["embedding_cache_path"].endswith("/embedding_cache")


def test_relocated_cache_attestation_is_propagated_to_stage1_builder(
    tmp_path,
    monkeypatch,
):
    source_cache = tmp_path / "source_cache"
    source_cache.mkdir()
    source_prepared = tmp_path / "source_prepared.parquet"
    source_manifest = tmp_path / "source_preparation_manifest.json"
    source_prepared.write_bytes(b"source")
    source_manifest.write_text("{}", encoding="utf-8")
    options = replace(
        _options(tmp_path),
        embedding_cache_import=source_cache,
        embedding_cache_import_source_prepared_path=source_prepared,
        embedding_cache_import_source_preparation_manifest_path=source_manifest,
    )
    workflow = ProductionAllEvidenceWorkflow(options)
    sentinel = object()
    monkeypatch.setattr(
        workflow,
        "_embedding_cache_relocation_options",
        lambda **_kwargs: sentinel,
    )
    profile = tmp_path / "effective.json"
    profile.write_text("{}", encoding="utf-8")
    cache = tmp_path / "relocated" / "embedding_cache"
    cache.mkdir(parents=True)
    prepared = tmp_path / "relocated" / "prepared" / "modeling_cohort.parquet"
    prepared.parent.mkdir()
    prepared.write_bytes(b"prepared")
    built = workflow._stage1_build_options(
        dataset=prepared,
        profile=profile,
        cache=cache,
        output=tmp_path / "bundle",
        dry_run=False,
    )
    assert built.embedding_cache_relocation is sentinel
    assert (
        built.stage1_scope_attempt_root
        == (options.work_root / "recovery/stage1_scope_attempts").resolve()
    )
    assert (
        built.stage1_scope_progress_path
        == (options.work_root / "recovery/stage1_scope_progress.json").resolve()
    )


def test_effective_profile_binds_review_tfidf_and_interaction_cli_settings(
    tmp_path,
):
    options = replace(
        _options(tmp_path),
        review_rounds=4,
        tfidf_nested_calibration_folds=6,
        interaction_inner_folds=7,
    )
    profile = {
        "config": {
            "architecture": {
                "htr_sentence_model": "old",
                "multi_model_forest": {
                    "candidate_consistency_inner_folds": 2,
                    "tfidf_nested_calibration_folds": 2,
                    "embedding_contrast": {},
                },
                "multi_model_agentic_forest": {
                    "candidate_consistency_inner_folds": 2,
                    "tfidf_nested_calibration_folds": 2,
                    "embedding_contrast": {},
                },
                "explicit_feature_forest": {"interaction_inner_folds": 2},
                "causal_forest": {},
            }
        }
    }
    options.stage1_profile_path.write_text(json.dumps(profile), encoding="utf-8")
    workflow = ProductionAllEvidenceWorkflow(options)
    attempt = tmp_path / "effective"
    attempt.mkdir()
    cache = tmp_path / "cache_for_profile"
    cache.mkdir()
    path = workflow._effective_stage1_profile(
        attempt,
        dataset_path=options.dataset_path,
        embedding_cache_dir=cache,
    )
    architecture = json.loads(path.read_text(encoding="utf-8"))["config"]["architecture"]
    for section_name in ("multi_model_forest", "multi_model_agentic_forest"):
        assert architecture[section_name]["candidate_consistency_inner_folds"] == 7
        assert architecture[section_name]["tfidf_nested_calibration_folds"] == 6
    assert architecture["explicit_feature_forest"]["interaction_inner_folds"] == 7


@pytest.mark.parametrize("mutation", ["change", "extra"])
def test_resume_rejects_any_change_to_a_sealed_attempt_tree(tmp_path, mutation):
    options = _options(tmp_path)
    payload_path = None

    def input_phase(attempt):
        nonlocal payload_path
        payload_path = attempt / "unlisted" / "nested.bin"
        payload_path.parent.mkdir()
        payload_path.write_bytes(b"sealed")
        return {"terminal_files": []}

    overrides = {phase: (lambda _attempt: {"terminal_files": []}) for phase in PHASES}
    overrides["input_preparation"] = input_phase
    ProductionAllEvidenceWorkflow(options, phase_overrides=overrides).run()
    assert payload_path is not None
    manifest = json.loads(
        (options.work_root / "phases/input_preparation/complete_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert any(row["relative_path"] == "unlisted/nested.bin" for row in manifest["artifacts"])
    if mutation == "change":
        payload_path.write_bytes(b"changed")
    else:
        (payload_path.parent / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="attempt tree changed"):
        ProductionAllEvidenceWorkflow(
            replace(options, resume=True),
            phase_overrides=overrides,
        ).run()


def test_source_snapshot_option_reexecs_from_authenticated_tree(tmp_path, monkeypatch):
    snapshot_root = tmp_path / "snapshot"
    entrypoint = snapshot_root / "scripts/run_production_all_evidence_workflow.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# fixture", encoding="utf-8")
    snapshot = SimpleNamespace(
        root=snapshot_root.resolve(),
        content_sha256="a" * 64,
    )
    monkeypatch.setattr(
        "oci.inference.production_source_snapshot.validate_production_source_snapshot",
        lambda _path: snapshot,
    )
    monkeypatch.delenv(workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV, raising=False)
    observed = {}

    class ReexecObserved(Exception):
        pass

    def fake_execve(executable, arguments, environment):
        observed.update(
            executable=executable,
            arguments=arguments,
            environment=environment,
        )
        raise ReexecObserved

    monkeypatch.setattr(workflow_module.os, "execve", fake_execve)
    parsed = SimpleNamespace(source_snapshot_root=snapshot_root, seed=42)
    with pytest.raises(ReexecObserved):
        workflow_module._reexec_from_source_snapshot(
            parsed_args=parsed,
            raw_argv=("--source-snapshot-root", str(snapshot_root)),
        )
    assert observed["arguments"][:3] == [
        workflow_module.sys.executable,
        "-P",
        "-u",
    ]
    assert observed["arguments"][3] == str(entrypoint)
    assert observed["environment"]["PYTHONPATH"] == str(snapshot_root.resolve())
    assert (
        observed["environment"][workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV]
        == snapshot.content_sha256
    )
    assert observed["environment"]["PYTHONHASHSEED"] == "42"


def test_source_snapshot_reexec_rejects_changed_parent_python_hash_seed(tmp_path, monkeypatch):
    snapshot_root = tmp_path / "snapshot"
    entrypoint = snapshot_root / "scripts/run_production_all_evidence_workflow.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# fixture", encoding="utf-8")
    snapshot = SimpleNamespace(
        root=snapshot_root.resolve(),
        content_sha256="b" * 64,
    )
    monkeypatch.setattr(
        "oci.inference.production_source_snapshot.validate_production_source_snapshot",
        lambda _path: snapshot,
    )
    monkeypatch.setattr(
        workflow_module,
        "__file__",
        str(snapshot_root / "oci/inference/production_all_evidence_workflow.py"),
    )
    monkeypatch.setenv(
        workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV,
        snapshot.content_sha256,
    )
    monkeypatch.setenv("PYTHONHASHSEED", "41")

    with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
        workflow_module._reexec_from_source_snapshot(
            parsed_args=SimpleNamespace(source_snapshot_root=snapshot_root, seed=42),
            raw_argv=("--source-snapshot-root", str(snapshot_root), "--seed", "42"),
        )


def test_fresh_canary_validator_sets_and_verifies_snapshot_environment(
    tmp_path,
    monkeypatch,
):
    options = replace(
        _options(tmp_path),
        endpoint=None,
        model_name=None,
        stage1_only=True,
        seed=42,
    )
    snapshot_root = (tmp_path / "snapshot").resolve()
    module_path = snapshot_root / "oci" / "inference" / "production_all_evidence_workflow.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# snapshot fixture\n", encoding="utf-8")
    snapshot_sha = "d" * 64
    workflow = ProductionAllEvidenceWorkflow(options)
    workflow.request = {
        "source_snapshot": {
            "root": str(snapshot_root),
            "content_sha256": snapshot_sha,
        }
    }
    options.work_root.mkdir()
    expected_result = {"status": "complete"}
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        observed["environment"] = dict(kwargs["env"])
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "result": expected_result,
                    "validator_module_path": str(module_path),
                    "source_snapshot_marker": kwargs["env"][
                        workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV
                    ],
                    "python_hash_seed": kwargs["env"]["PYTHONHASHSEED"],
                    "python_path": kwargs["env"]["PYTHONPATH"],
                    "python_no_user_site": kwargs["env"]["PYTHONNOUSERSITE"],
                }
            )
        )

    monkeypatch.setattr(workflow_module.subprocess, "run", fake_run)

    assert workflow._validate_canary_preparation_in_fresh_process() == expected_result
    assert observed["command"][1] == "-P"
    assert observed["environment"]["PYTHONHASHSEED"] == "42"
    assert observed["environment"]["PYTHONPATH"] == str(snapshot_root)
    assert observed["environment"]["PYTHONNOUSERSITE"] == "1"
    assert observed["environment"][workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV] == snapshot_sha


def test_interrupted_initial_request_publication_preserves_attempt_and_fresh_root(
    tmp_path,
    monkeypatch,
):
    options = _options(tmp_path)
    workflow = ProductionAllEvidenceWorkflow(options)
    original_atomic_write = workflow_module._atomic_write_json

    def interrupt_request(path, value):
        if path.name == "immutable_run_request.json":
            raise KeyboardInterrupt("fixture interruption")
        return original_atomic_write(path, value)

    monkeypatch.setattr(workflow_module, "_atomic_write_json", interrupt_request)
    with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
        workflow._initialize()

    assert not options.work_root.exists()
    attempts = tuple(
        options.work_root.parent.glob(f".{options.work_root.name}.initialization_attempt_*")
    )
    assert len(attempts) == 1
    assert attempts[0].is_dir()

    monkeypatch.setattr(workflow_module, "_atomic_write_json", original_atomic_write)
    ProductionAllEvidenceWorkflow(options)._initialize()
    assert (options.work_root / "immutable_run_request.json").is_file()
    assert attempts[0].is_dir()


@pytest.mark.parametrize(
    ("field", "expected_message"),
    (
        ("query_profile_path", "neural-query profile changed"),
        ("embedding_local_model_path", "embedding model tree changed"),
    ),
)
def test_phase_boundary_rejects_request_bound_external_input_changes(
    tmp_path,
    field,
    expected_message,
):
    options = _options(tmp_path)
    target = Path(getattr(options, field))
    if target.is_dir():
        target = target / "model.safetensors"

    def mutate_bound_input(_attempt):
        target.write_text("changed after immutable request", encoding="utf-8")
        return {"terminal_files": []}

    overrides = {phase: (lambda _attempt: {"terminal_files": []}) for phase in PHASES}
    overrides["input_preparation"] = mutate_bound_input
    workflow = ProductionAllEvidenceWorkflow(
        options,
        phase_overrides=overrides,
    )

    with pytest.raises(RuntimeError, match=expected_message):
        workflow.run()
    assert not (
        options.work_root / "phases" / "input_preparation" / "complete_manifest.json"
    ).exists()


def test_imported_cache_workflow_hashes_embedding_tree_once_per_process(
    tmp_path,
    monkeypatch,
):
    options = _options(tmp_path)
    source_cache = (tmp_path / "source-cache").resolve()
    source_cache.mkdir()
    (source_cache / "metadata.json").write_text("{}\n", encoding="utf-8")
    source_prepared = (tmp_path / "source-prepared.parquet").resolve()
    source_prepared.write_bytes(b"prepared")
    source_manifest = (tmp_path / "source-preparation.json").resolve()
    source_manifest.write_text("{}\n", encoding="utf-8")
    options = replace(
        options,
        embedding_cache_import=source_cache,
        embedding_cache_import_source_prepared_path=source_prepared,
        embedding_cache_import_source_preparation_manifest_path=source_manifest,
    )
    tree_module.clear_authenticated_directory_tree_cache()
    calls: dict[Path, int] = {}
    original = tree_module._stable_file_authentication

    def counted(root: Path, relative_path: str):
        calls[root] = calls.get(root, 0) + 1
        return original(root, relative_path)

    monkeypatch.setattr(tree_module, "_stable_file_authentication", counted)
    overrides = {phase: (lambda _attempt: {"terminal_files": []}) for phase in PHASES}
    ProductionAllEvidenceWorkflow(options, phase_overrides=overrides).run()

    assert calls[options.embedding_local_model_path] == 1
    assert calls[source_cache] == 1
    request = json.loads(
        (options.work_root / "immutable_run_request.json").read_text(encoding="utf-8")
    )
    assert (
        request["embedding_model_revalidation_policy"]
        == tree_module.AUTHENTICATED_DIRECTORY_TREE_POLICY
    )


def test_fresh_cache_build_keeps_full_model_tree_validation_path(
    tmp_path,
    monkeypatch,
):
    options = _options(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("fresh embedding-cache builds cannot use process-local tree reuse")

    monkeypatch.setattr(workflow_module, "authenticate_directory_tree", forbidden)
    overrides = {phase: (lambda _attempt: {"terminal_files": []}) for phase in PHASES}
    ProductionAllEvidenceWorkflow(options, phase_overrides=overrides).run()
    request = json.loads(
        (options.work_root / "immutable_run_request.json").read_text(encoding="utf-8")
    )
    assert request["embedding_model_revalidation_policy"] == "full_byte_tree_reauthentication_v1"
