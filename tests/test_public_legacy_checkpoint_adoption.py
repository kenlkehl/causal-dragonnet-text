from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

import oci.inference.legacy_checkpoint_migration as legacy_migration_module
import oci.inference.production_all_evidence_workflow as workflow_module
import oci.inference.production_embedding_cache_builder as cache_builder_module
from oci.inference.physical_fit_deduplication import (
    derive_logical_context_plan,
)
from oci.inference.portable_artifacts import relocate_portable_artifact
from oci.inference.portable_workflow_spec import (
    Stage1ExecutionProfile,
    identity_sha256,
)
from oci.inference.production_all_evidence_workflow import (
    ProductionAllEvidenceWorkflow,
    build_parser,
)
from oci.inference.production_embedding_cache_builder import (
    build_production_embedding_cache,
)
from oci.inference.production_embedding_cache_relocation import (
    ProductionEmbeddingCacheRelocationOptions,
    relocate_authenticated_production_embedding_cache,
)
from oci.inference.production_text_preparation import (
    TextPreparationOptions,
    prepare_modeling_cohort,
    stable_file_sha256,
)
from oci.models.strict_causal_forest_runtime import (
    STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
    StrictCausalForestRuntimeConfig,
)
from tests.test_portable_workflow_contracts import (
    _forest_operational,
    _scientific_spec,
)
from tests.test_production_all_evidence_workflow import (
    _options,
    _with_run_control,
)
from tests.stage1_test_support import stage1_execution_profile


class _Tokenizer:
    def __call__(
        self,
        inputs,
        *,
        add_special_tokens,
        truncation,
        padding,
        return_length,
    ):
        assert add_special_tokens is True
        assert truncation is False
        assert padding is False
        assert return_length is True
        return {"length": [len(value.split()) + 2 for value in inputs]}


class _Encoder:
    def __init__(self, *, max_seq_length: int) -> None:
        self.max_seq_length = int(max_seq_length)
        self.tokenizer = _Tokenizer()
        self.default_prompt_name = None
        self.prompts: dict[str, str] = {}

    def encode(
        self,
        chunks,
        *,
        prompt_name,
        prompt,
        batch_size,
        output_value,
        precision,
        convert_to_numpy,
        convert_to_tensor,
        normalize_embeddings,
        truncate_dim,
        show_progress_bar,
        pool,
        chunk_size,
    ):
        assert prompt_name is None
        assert prompt == ""
        assert 1 <= batch_size
        assert output_value == "sentence_embedding"
        assert precision == "float32"
        assert convert_to_numpy is True
        assert convert_to_tensor is False
        assert truncate_dim is None
        assert show_progress_bar is False
        assert pool is None
        assert chunk_size is None
        values = []
        for chunk in chunks:
            digest = hashlib.sha256(chunk.encode("utf-8")).digest()
            vector = np.asarray(
                [float(value + 1) for value in digest[:5]],
                dtype=np.float32,
            )
            if normalize_embeddings:
                vector /= np.linalg.norm(vector)
            values.append(vector)
        return np.asarray(values, dtype=np.float32)


def _typed_options(tmp_path: Path):
    spec = _scientific_spec()
    base = _options(tmp_path)
    # The legacy validator requires a nonempty directory count in the
    # authenticated model-tree provenance. Keep the fixture genuinely small
    # while satisfying the same closed tree schema as the production model.
    model_metadata = base.embedding_local_model_path / "metadata"
    model_metadata.mkdir()
    (model_metadata / "config.json").write_text(
        '{"fixture":"legacy-public-adoption"}\n',
        encoding="utf-8",
    )
    return replace(
        base,
        unit_id_column=spec.columns.unit_id,
        text_column=spec.columns.text,
        treatment_column=spec.columns.treatment,
        outcome_column=spec.columns.outcome,
        clinical_question=spec.clinical_question,
        outer_folds=spec.folds.outer_folds,
        review_rounds=spec.folds.review_rounds,
        initial_training_partitions=spec.folds.initial_training_partitions,
        interaction_inner_folds=spec.folds.interaction_inner_folds,
        tfidf_nested_calibration_folds=(spec.folds.tfidf_nested_calibration_folds),
        seed=spec.seed,
        empty_text_policy=spec.preprocessing.empty_text_policy,
        repeated_character_policy=(spec.preprocessing.repeated_character_policy),
        repeated_character_threshold=(spec.preprocessing.repeated_character_threshold),
        source_text_temporally_valid_by_design=(
            spec.preprocessing.source_text_temporally_valid_by_design
        ),
        complete_page_core_chars=spec.text_windows.complete_page_core_chars,
        complete_page_context_chars=(spec.text_windows.complete_page_context_chars),
        complete_page_max_chars=spec.text_windows.complete_page_max_chars,
        complete_reconciliation_fan_in=(spec.text_windows.reconciliation_fan_in),
        embedding_chunk_size_words=(spec.text_windows.embedding_chunk_size_words),
        embedding_chunk_overlap_words=(spec.text_windows.embedding_chunk_overlap_words),
        embedding_max_chunks=spec.text_windows.embedding_max_chunks,
        embedding_chunk_selection=(spec.text_windows.embedding_chunk_selection),
        embedding_max_seq_length=(spec.text_windows.embedding_max_seq_length),
        embedding_normalize=spec.text_windows.embedding_normalize,
        embedding_encoder=spec.text_windows.embedding_encoder,
        stage2_prompt_protocol=spec.stage2_prompt_protocol,
        post_extraction_causal_review=spec.post_extraction_causal_review,
        max_candidate_variables=spec.max_candidate_variables,
        forest_runtime_config=StrictCausalForestRuntimeConfig(
            schema_version=STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
            causal_forest=spec.causal_estimator,
            operational=_forest_operational(base.cpu_budget),
        ),
        forest_n_estimators=None,
        forest_max_depth=None,
        forest_min_samples_leaf=None,
        forest_max_features=None,
        forest_honest=None,
        forest_inference=None,
        forest_subforest_size=None,
        forest_tune_model=None,
        forest_nuisance_n_estimators=None,
        forest_nuisance_max_depth=None,
        forest_nuisance_min_samples_leaf=None,
        forest_nuisance_treatment_max_features=None,
        forest_nuisance_outcome_max_features=None,
        forest_random_seed=None,
        portable_scientific_spec=spec.identity_payload(),
        stage1_execution_profile=stage1_execution_profile(
            resource_kind="cpu",
            device_count=base.stage1_execution_device_count,
            scope_workers_per_device=base.stage1_scope_workers_per_gpu,
        ),
    )


def _write_source(options, path: Path) -> Path:
    pd.DataFrame(
        {
            options.unit_id_column: ["p3", "p1", "p4", "p2"],
            options.text_column: [
                "alpha beta gamma delta epsilon",
                "",
                "one two three four five six",
                "punctuation !!!!! remains complete",
            ],
            options.treatment_column: [0, 1, 0, 1],
            options.outcome_column: [1, 0, 0, 1],
        }
    ).to_parquet(path, index=False)
    return path


def _write_terminal_phase(
    *,
    phase: str,
    attempt: Path,
    result: Mapping[str, Any],
) -> Path:
    registrations = []
    for path in sorted(
        (value for value in attempt.rglob("*") if value.is_file()),
        key=lambda value: value.relative_to(attempt).as_posix(),
    ):
        digest, size = stable_file_sha256(path)
        registrations.append(
            {
                "path": str(path.resolve()),
                "relative_path": path.relative_to(attempt).as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    body = {
        "schema_version": "production_workflow_phase_manifest_v2",
        "status": "complete",
        "phase": phase,
        "request_sha256": identity_sha256({"legacy_request": phase}),
        "attempt_dir": str(attempt.resolve()),
        "result": dict(result),
        "artifacts": registrations,
    }
    manifest = attempt.parent / "complete_manifest.json"
    manifest.write_text(
        json.dumps(
            {**body, "content_sha256": identity_sha256(body)},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def _build_legacy_preparation(options, root: Path) -> tuple[Path, Path, Path]:
    attempt = root / "attempt"
    attempt.mkdir(parents=True)
    result = prepare_modeling_cohort(
        TextPreparationOptions(
            dataset_path=options.dataset_path,
            output_dir=attempt / "prepared",
            unit_id_column=options.unit_id_column,
            text_column=options.text_column,
            treatment_column=options.treatment_column,
            outcome_column=options.outcome_column,
            repeated_character_threshold=(options.repeated_character_threshold),
            empty_text_policy=options.empty_text_policy,
            repeated_character_policy=options.repeated_character_policy,
        )
    )
    cohort = attempt / "prepared" / "modeling_cohort.parquet"
    preparation_manifest = attempt / "prepared" / "preparation_manifest.json"
    terminal = _write_terminal_phase(
        phase="input_preparation",
        attempt=attempt,
        result={
            **result,
            "terminal_files": [
                str(cohort.resolve()),
                str(preparation_manifest.resolve()),
            ],
        },
    )
    return terminal, cohort, preparation_manifest


def _chunk_configuration(options) -> dict[str, object]:
    return {
        "chunk_size_words": int(options.embedding_chunk_size_words),
        "chunk_overlap_words": int(options.embedding_chunk_overlap_words),
        "max_chunks": int(options.embedding_max_chunks),
        "chunk_selection": str(options.embedding_chunk_selection),
        "normalize_embeddings": bool(options.embedding_normalize),
        "max_seq_length": options.embedding_max_seq_length,
        **options.embedding_encoder.as_configuration(
            normalize_embeddings=bool(options.embedding_normalize)
        ),
    }


def _build_legacy_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    options = _typed_options(tmp_path)
    source = _write_source(options, tmp_path / "configured-source.parquet")
    options = replace(options, dataset_path=source)
    prepared_terminal, prepared, preparation_manifest = _build_legacy_preparation(
        options,
        tmp_path / "legacy-input-preparation",
    )

    monkeypatch.setattr(
        cache_builder_module,
        "_load_local_sentence_encoder",
        lambda **_kwargs: _Encoder(max_seq_length=int(options.embedding_max_seq_length)),
    )
    source_cache = tmp_path / "legacy-source-cache"
    build_production_embedding_cache(
        dataset_path=prepared,
        text_column=options.text_column,
        local_model_path=options.embedding_local_model_path,
        sentence_model_name=options.embedding_model_name,
        chunk_configuration=_chunk_configuration(options),
        target_dir=source_cache,
        device="cpu",
        batch_size=2,
    )
    fresh_root = tmp_path / "fresh-prepared-for-relocation"
    prepare_modeling_cohort(
        TextPreparationOptions(
            dataset_path=source,
            output_dir=fresh_root,
            unit_id_column=options.unit_id_column,
            text_column=options.text_column,
            treatment_column=options.treatment_column,
            outcome_column=options.outcome_column,
            repeated_character_threshold=(options.repeated_character_threshold),
            empty_text_policy=options.empty_text_policy,
            repeated_character_policy=options.repeated_character_policy,
        )
    )
    cache_attempt = tmp_path / "legacy-embedding-cache" / "attempt"
    cache_attempt.mkdir(parents=True)
    relocation = relocate_authenticated_production_embedding_cache(
        ProductionEmbeddingCacheRelocationOptions(
            source_cache_dir=source_cache,
            source_prepared_cohort_path=prepared,
            source_preparation_manifest_path=preparation_manifest,
            fresh_prepared_cohort_path=(fresh_root / "modeling_cohort.parquet"),
            fresh_preparation_manifest_path=(fresh_root / "preparation_manifest.json"),
            local_model_path=options.embedding_local_model_path,
            target_dir=cache_attempt / "relocated_cache",
            unit_id_column=options.unit_id_column,
            text_column=options.text_column,
            treatment_column=options.treatment_column,
            outcome_column=options.outcome_column,
            sentence_model_name=options.embedding_model_name,
            chunk_configuration=_chunk_configuration(options),
        )
    )
    terminal_files = [
        *(str(path.resolve()) for path in sorted(relocation.cache_dir.iterdir()) if path.is_file()),
        str(relocation.prepared_cohort_path.resolve()),
        str(relocation.attestation_path.resolve()),
        str(relocation.terminal_manifest_path.resolve()),
    ]
    cache_terminal = _write_terminal_phase(
        phase="embedding_cache",
        attempt=cache_attempt,
        result={
            "schema_version": (workflow_module.EMBEDDING_CACHE_PHASE_SCHEMA),
            "mode": "authenticated_relocation",
            "cache_path": str(relocation.cache_dir.resolve()),
            "prepared_cohort_path": str(relocation.prepared_cohort_path.resolve()),
            "cache_identity": relocation.identity(),
            "resource_preflight": {"fixture": "cpu"},
            "embedding_model_materialized_in_workflow_process": False,
            "cuda_memory_release_requested": False,
            "terminal_files": terminal_files,
        },
    )
    return options, prepared_terminal, cache_terminal


def _payload_override(phase: str, calls: list[str]):
    def run(attempt: Path) -> Mapping[str, Any]:
        calls.append(phase)
        payload = attempt / f"{phase}.bin"
        payload.write_bytes(phase.encode("utf-8"))
        return {"terminal_files": [str(payload.resolve())]}

    return run


def _find_json_objects(root: Path, *, key: str, value: Any):
    matches = []
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get(key) == value:
            matches.append((path, payload))
    return matches


def _write_minimal_legacy_preflight_candidate(root: Path) -> Path:
    root.mkdir()
    audit = root / "cluster_feasibility_audit.json"
    request = root / "stage1_preflight_request.json"
    audit.write_text('{"complete":true}\n', encoding="utf-8")
    request.write_text('{"complete":true}\n', encoding="utf-8")

    def registration(path: Path) -> dict[str, Any]:
        digest, size = stable_file_sha256(path)
        return {
            "relative_path": path.name,
            "sha256": digest,
            "size_bytes": size,
        }

    scope = {
        "canonical_index": 0,
        "scope_id": "fixture_scope",
        "scope_kind": "full_outer",
        "outer_fold": 1,
        "inner_fold": None,
        "context_epoch": None,
        "provider_inner_fold": None,
        "fit_row_count": 1,
        "fit_row_order_fingerprint": identity_sha256(["fixture-fit"]),
        "heldout_row_count": 1,
        "heldout_row_order_fingerprint": identity_sha256(["fixture-heldout"]),
        "scope_record_sha256": identity_sha256({"scope": "fixture"}),
        "cluster_fit_identity_sha256": identity_sha256({"cluster": "fixture"}),
    }
    body = {
        "schema_version": "production_stage1_cluster_preflight_manifest_v1",
        "status": "complete",
        "artifact_version": "legacy-public-fixture-v1",
        "artifact_code_sha256": identity_sha256({"producer": "fixture"}),
        "root": str(root.resolve()),
        "files": {
            "audit": registration(audit),
            "stage1_request": registration(request),
        },
        "bindings": {},
        "scope_records": [scope],
    }
    manifest = root / "cluster_preflight_manifest.json"
    manifest.write_text(
        json.dumps(
            {**body, "content_sha256": identity_sha256(body)},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def _representative_legacy_contexts():
    partitions = {
        fold: tuple(
            tuple(
                fold * 1_000 + partition * 10 + row
                for row in range(3)
            )
            for partition in range(1, 6)
        )
        for fold in range(1, 6)
    }
    heldout = {
        fold: tuple(fold * 1_000 + 900 + row for row in range(3))
        for fold in range(1, 6)
    }
    return derive_logical_context_plan(
        outer_training_partitions=partitions,
        outer_heldout_rows=heldout,
        architecture_identity=identity_sha256({"architecture": "representative"}),
        target="cluster_preflight",
        scientific_configuration_identity=identity_sha256({"configuration": "representative"}),
        global_seed=42,
        producer_identity=identity_sha256({"producer": "representative"}),
        runtime_compatibility_class="representative-runtime-v1",
        review_rounds=2,
    )


def _legacy_row_fingerprint(values) -> str:
    normalized = []
    for value in values:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            normalized.append(str(value))
    return identity_sha256({"ordered_row_ids": normalized})


def _write_representative_legacy_preflight_candidate(
    root: Path,
    contexts,
    *,
    historical_cumulative_order_drift: bool = False,
) -> Path:
    root.mkdir()
    audit = root / "cluster_feasibility_audit.json"
    request = root / "stage1_preflight_request.json"
    audit.write_text('{"complete":true}\n', encoding="utf-8")
    request.write_text('{"complete":true}\n', encoding="utf-8")

    def registration(path: Path) -> dict[str, Any]:
        digest, size = stable_file_sha256(path)
        return {
            "relative_path": path.name,
            "sha256": digest,
            "size_bytes": size,
        }

    scope_records = []
    for index, context in enumerate(contexts):
        fit_values = list(context.fit_row_ids)
        heldout_values = list(context.heldout_row_ids)
        is_cumulative = context.purpose.startswith("cumulative_review_epoch_")
        if historical_cumulative_order_drift and is_cumulative:
            fit_values.reverse()
            heldout_values.reverse()
        scope_records.append(
            {
                "canonical_index": index,
                "scope_id": context.scope_id,
                "scope_kind": context.purpose,
                "outer_fold": context.outer_fold,
                "inner_fold": None,
                "context_epoch": None,
                "provider_inner_fold": None,
                "fit_row_count": len(context.fit_row_ids),
                "fit_row_order_fingerprint": _legacy_row_fingerprint(fit_values),
                "heldout_row_count": len(context.heldout_row_ids),
                "heldout_row_order_fingerprint": _legacy_row_fingerprint(heldout_values),
                "scope_record_sha256": identity_sha256({"scope": context.scope_id}),
                "cluster_fit_identity_sha256": identity_sha256({"cluster": context.scope_id}),
            }
        )
    body = {
        "schema_version": "production_stage1_cluster_preflight_manifest_v1",
        "status": "complete",
        "artifact_version": "legacy-public-representative-v1",
        "artifact_code_sha256": identity_sha256({"producer": "legacy-public-representative"}),
        "root": str(root.resolve()),
        "files": {
            "audit": registration(audit),
            "stage1_request": registration(request),
        },
        "bindings": {},
        "scope_records": scope_records,
    }
    manifest = root / "cluster_preflight_manifest.json"
    manifest.write_text(
        json.dumps(
            {**body, "content_sha256": identity_sha256(body)},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def test_public_parser_keeps_legacy_phase_and_preflight_paths_separate(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prep" / "complete_manifest.json"
    cache = tmp_path / "cache" / "complete_manifest.json"
    preflight = tmp_path / "cluster_preflight_manifest.json"
    parsed = build_parser().parse_args(
        [
            "--adopt-checkpoint",
            str(prepared),
            "--adopt-checkpoint",
            str(cache),
            "--legacy-preflight-candidate",
            str(preflight),
        ]
    )
    assert parsed.adopt_checkpoint == [prepared, cache]
    assert parsed.legacy_preflight_candidate == preflight


def test_public_legacy_prepared_and_cache_migrate_as_one_adopted_dag_and_relocate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, prepared_manifest, cache_manifest = _build_legacy_prefix(
        tmp_path,
        monkeypatch,
    )
    calls: list[str] = []
    first_options = _with_run_control(
        options,
        stop_after="embedding_cache",
        # CLI order is operational. The authenticated dependency graph decides
        # that preparation precedes cache.
        adopt_checkpoints=(cache_manifest, prepared_manifest),
    )
    first = ProductionAllEvidenceWorkflow(
        first_options,
        phase_overrides={
            phase: _payload_override(phase, calls)
            for phase in ("input_preparation", "embedding_cache")
        },
    )
    result = first.run()
    assert result["completed_phases"] == [
        "input_preparation",
        "embedding_cache",
    ]
    assert calls == []
    migrated = {
        artifact.manifest["artifact_kind"]: artifact
        for artifact in first._adopted_artifact_handles.values()
    }
    assert set(migrated) == {"prepared_cohort", "embedding_cache"}
    assert migrated["embedding_cache"].manifest["upstream_artifact_ids"] == [
        migrated["prepared_cohort"].artifact_id
    ]
    assert all(
        "legacy_terminal_migration_identity"
        in workflow_module.materialize_portable_phase(
            artifact,
            expected_phase=phase,
        )["result"]
        for phase, artifact in (
            ("input_preparation", migrated["prepared_cohort"]),
            ("embedding_cache", migrated["embedding_cache"]),
        )
    )

    relocated_prepared = relocate_portable_artifact(
        source=migrated["prepared_cohort"].root,
        target_root=tmp_path / "arbitrary-a" / "prepared",
    )
    relocated_cache = relocate_portable_artifact(
        source=migrated["embedding_cache"].root,
        target_root=tmp_path / "arbitrary-b" / "cache",
    )
    assert relocated_prepared.artifact_id == migrated["prepared_cohort"].artifact_id
    assert relocated_cache.artifact_id == migrated["embedding_cache"].artifact_id

    second_calls: list[str] = []
    second_options_base = replace(
        options,
        work_root=tmp_path / "relocated-consumer",
    )
    second_options = _with_run_control(
        second_options_base,
        stop_after="embedding_cache",
        adopt_checkpoints=(
            relocated_prepared.root,
            relocated_cache.root,
        ),
    )
    second = ProductionAllEvidenceWorkflow(
        second_options,
        phase_overrides={
            phase: _payload_override(phase, second_calls)
            for phase in ("input_preparation", "embedding_cache")
        },
    )
    second.run()
    assert second_calls == []
    assert {value.artifact_id for value in second._adopted_artifact_handles.values()} == {
        relocated_prepared.artifact_id,
        relocated_cache.artifact_id,
    }


def test_public_legacy_adoption_rejects_tamper_and_nonterminal_paths_before_run_root(
    tmp_path: Path,
) -> None:
    options = _typed_options(tmp_path)
    source = _write_source(options, tmp_path / "configured-source.parquet")
    options = replace(options, dataset_path=source)
    terminal, cohort, _preparation_manifest = _build_legacy_preparation(
        options,
        tmp_path / "legacy-input-preparation",
    )
    original = cohort.read_bytes()
    cohort.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    tampered = _with_run_control(
        options,
        stop_after="input_preparation",
        adopt_checkpoints=(terminal,),
    )
    with pytest.raises(
        ValueError,
        match="changed|registration|prepared cohort|authenticated file",
    ):
        ProductionAllEvidenceWorkflow(tampered).run()
    assert not tampered.work_root.exists()

    incomplete_options_base = replace(
        options,
        work_root=tmp_path / "incomplete-consumer",
    )
    incomplete_options = _with_run_control(
        incomplete_options_base,
        stop_after="input_preparation",
        # A phase directory or attempt is not an exact terminal-manifest path.
        adopt_checkpoints=(terminal.parent,),
    )
    with pytest.raises(
        (ValueError, FileNotFoundError),
        match="exact|complete_manifest|unsupported checkpoint|manifest",
    ):
        ProductionAllEvidenceWorkflow(incomplete_options).run()
    assert not incomplete_options.work_root.exists()


def test_adopt_checkpoint_routes_legacy_preflight_as_audit_only_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    options = _typed_options(tmp_path)
    candidate = _write_minimal_legacy_preflight_candidate(tmp_path / "legacy-cluster-preflight")
    calls: list[str] = []
    configured = _with_run_control(
        options,
        stop_after="input_preparation",
        adopt_checkpoints=(candidate,),
    )
    workflow = ProductionAllEvidenceWorkflow(
        configured,
        phase_overrides={
            "input_preparation": _payload_override(
                "input_preparation",
                calls,
            )
        },
    )
    workflow.run()
    assert calls == ["input_preparation"]
    assert (
        workflow.request["legacy_preflight_candidate_identity"]["selection_source"]
        == "adopt_checkpoint"
    )
    assert workflow.request["requested_checkpoint_adoptions"] == []
    assert workflow.request["checkpoint_adoption_locators"] == []
    assert not (configured.work_root / "checkpoint_adoptions").exists()

    conflicting_base = replace(
        options,
        work_root=tmp_path / "conflicting-consumer",
        legacy_preflight_candidate=candidate,
    )
    conflicting = _with_run_control(
        conflicting_base,
        stop_after="input_preparation",
        adopt_checkpoints=(candidate,),
    )
    with pytest.raises(ValueError, match="both --adopt-checkpoint"):
        ProductionAllEvidenceWorkflow(conflicting).run()
    assert not conflicting.work_root.exists()

    with pytest.raises(ValueError, match="cannot be duplicated"):
        ProductionAllEvidenceWorkflow(
            _with_run_control(
                replace(
                    options,
                    work_root=tmp_path / "duplicate-consumer",
                ),
                adopt_checkpoints=(candidate, candidate),
            )
        )


def test_legacy_preflight_override_cannot_bypass_full_byte_migration(
    tmp_path: Path,
) -> None:
    options = _typed_options(tmp_path)
    candidate = _write_minimal_legacy_preflight_candidate(tmp_path / "legacy-cluster-preflight")
    with pytest.raises(
        ValueError,
        match="hooks and phase overrides cannot bypass",
    ):
        ProductionAllEvidenceWorkflow(
            _with_run_control(
                options,
                adopt_checkpoints=(candidate,),
            ),
            phase_overrides={
                "stage1_preflight": _payload_override(
                    "stage1_preflight",
                    [],
                )
            },
        )


def test_representative_v4_recompute_decision_is_closed_and_terminal(
    tmp_path: Path,
) -> None:
    contexts = _representative_legacy_contexts()
    assert len(contexts) == 40
    candidate = _write_representative_legacy_preflight_candidate(
        tmp_path / "representative-legacy-preflight",
        contexts,
        historical_cumulative_order_drift=True,
    )
    migration = legacy_migration_module.plan_legacy_v4_preflight_migration(
        manifest_path=candidate,
        logical_contexts=contexts,
        authenticate_registered_payload_bytes=True,
    )
    validated = legacy_migration_module.validate_legacy_preflight_manifest(
        candidate,
        authenticate_registered_payload_bytes=False,
    )
    manifest_sha256, manifest_size = stable_file_sha256(candidate)
    source_identity = {
        "selection_source": "adopt_checkpoint",
        "manifest_path": str(candidate.resolve()),
        "manifest_sha256": manifest_sha256,
        "manifest_size_bytes": manifest_size,
        "manifest_content_sha256": validated["manifest"]["content_sha256"],
        "registered_payloads": {
            name: {
                "path": row["path"],
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
            }
            for name, row in validated["payloads"].items()
        },
        "registered_payload_bytes_authenticated_during_request": False,
        "direct_reuse_allowed": False,
    }
    attempt = tmp_path / "preflight-attempt"
    attempt.mkdir()
    path, decision = workflow_module._persist_legacy_preflight_recompute_decision(
        attempt=attempt,
        consumer_request_sha256=identity_sha256({"consumer": "representative"}),
        source_candidate_identity=source_identity,
        migration=migration,
        expected_logical_scope_count=40,
        expected_physical_fit_count=35,
    )
    assert set(decision) == {
        "schema_version",
        "consumer_request_sha256",
        "source_candidate_identity",
        "migration_decision",
        "adoption_disposition",
        "current_preflight_recomputed",
        "legacy_fitted_output_reused",
        "terminal_registration_required",
        "source_tree_mutated",
        "content_sha256",
    }
    assert decision["schema_version"] == (workflow_module.WORKFLOW_LEGACY_PREFLIGHT_DECISION_SCHEMA)
    assert decision["adoption_disposition"] == ("audit_only_not_checkpoint_adoption")
    assert decision["legacy_fitted_output_reused"] is False
    assert decision["terminal_registration_required"] is True
    assert migration["logical_scope_count"] == 40
    assert migration["physical_fit_count"] == 35
    assert migration["recompute_physical_fit_count"] == 35
    assert migration["deduplicated_group_count"] == 5
    assert len(migration["accounting"]["superseded_duplicate_outputs"]) == 5
    assert (
        "legacy_cumulative_row_order_not_reusable_for_current_request"
        in migration["recompute_reason_codes"]
    )
    assert migration["dependency_proof"]["requested_fit_row_orders_match_legacy_records"] is False
    assert (
        sum(
            row["legacy_order_disposition"] == "exact_request_match"
            for row in migration["accounting"]["logical_bindings"]
        )
        == 30
    )
    assert (
        sum(
            row["legacy_order_disposition"] == "cumulative_historical_order_not_reusable"
            for row in migration["accounting"]["logical_bindings"]
        )
        == 10
    )
    assert (
        sum(
            row["legacy_owner_order_reusable_for_current_fit"]
            for row in migration["accounting"]["physical_records"]
        )
        == 30
    )
    assert all(
        row["same_fit_row_content_proven"] is False
        and row["current_equivalence_proven"] is True
        and row["legacy_order_reusable_for_current_fit"] is False
        for row in migration["accounting"]["superseded_duplicate_outputs"]
    )
    registered = {row["path"] for row in workflow_module._attempt_tree_artifacts(attempt)}
    assert str(path) in registered


def test_legacy_v4_row_fingerprint_reconstructs_authentic_wrapped_shape() -> None:
    values = (10, 2, 30)
    assert legacy_migration_module._legacy_row_fingerprint(values) == identity_sha256(
        {"ordered_row_ids": [10, 2, 30]}
    )
    assert legacy_migration_module._legacy_row_fingerprint(values) != identity_sha256([10, 2, 30])
    with pytest.raises(TypeError, match="exact integer"):
        legacy_migration_module._legacy_row_fingerprint(("10", 2, 30))
    with pytest.raises(TypeError, match="exact integer"):
        legacy_migration_module._legacy_row_fingerprint((True, 2, 30))


def test_representative_v4_rejects_noncumulative_order_drift(
    tmp_path: Path,
) -> None:
    contexts = _representative_legacy_contexts()
    candidate = _write_representative_legacy_preflight_candidate(
        tmp_path / "noncumulative-order-drift",
        contexts,
    )
    manifest = json.loads(candidate.read_text(encoding="utf-8"))
    first_context = contexts[0]
    manifest["scope_records"][0]["fit_row_order_fingerprint"] = _legacy_row_fingerprint(
        reversed(first_context.fit_row_ids)
    )
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    candidate.write_text(
        json.dumps(
            {**body, "content_sha256": identity_sha256(body)},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="legacy non-cumulative scope row order differs from request",
    ):
        legacy_migration_module.plan_legacy_v4_preflight_migration(
            manifest_path=candidate,
            logical_contexts=contexts,
            authenticate_registered_payload_bytes=True,
        )


def test_legacy_preflight_candidate_validation_rejects_wrong_locator_kind(
    tmp_path: Path,
) -> None:
    options = _typed_options(tmp_path)
    wrong = tmp_path / "not-a-cluster-manifest.json"
    wrong.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="cluster_preflight_manifest",
    ):
        ProductionAllEvidenceWorkflow(
            replace(
                options,
                legacy_preflight_candidate=wrong,
            )
        )
