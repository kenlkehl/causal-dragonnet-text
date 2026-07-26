from __future__ import annotations

import functools
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from oci.inference.physical_fit_deduplication import (
    PhysicalFitKey,
    ordered_row_identity,
)
from oci.inference.portable_artifacts import (
    ArtifactCompatibility,
    COMPLETE_PAYLOAD_TREE,
    REGISTERED_PAYLOAD_PATHS_ONLY,
    relocate_portable_artifact,
    publish_portable_reference_artifact,
    validate_portable_artifact,
)
from oci.inference.production_all_evidence_workflow import (
    GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS,
    PORTABLE_ROLE_NEUTRAL_STAGE1_PHASE_SCHEMA,
    WORKFLOW_GRANULAR_CHECKPOINT_INDEX_SCHEMA,
    WORKFLOW_GRANULAR_CHECKPOINT_LOCATOR_SCHEMA,
    WORKFLOW_EXPECTED_GRANULAR_PLAN_SCHEMA,
    ProductionAllEvidenceWorkflow,
    _ScientificIdentityMemo,
    _attempt_tree_artifacts,
    _bind_workflow_scientific_identity,
    _derive_expected_granular_checkpoint_plan,
    _granular_checkpoint_coverage,
    _granular_checkpoint_index_paths,
    _granular_primary_metadata_from_index,
    _hook_identity,
    _local_import_paths,
    _phase_payload_stat_inventory,
    _phase_transitive_producer_code_records,
    _portable_stage1_terminal_file_inventory,
    _portable_stage2_terminal_file_inventory,
    _sha,
    _transitive_local_source_inventory,
    _repository_local_callable_import_closure,
    _read_json_object,
    _validate_granular_handles_against_plan,
    _validate_granular_checkpoint_index_from_paths,
    _validate_primary_granular_binding_digests,
    _validated_stage1_granular_physical_fit_key,
    _write_immutable_json,
)
from oci.inference.production_stage1_scope_scheduler import (
    Stage1PhysicalFitIdentity,
)
from tests.hook_identity_test_support import (
    CallableStateHook,
    UnclosedCallableStateHook,
    closure_state_hook_factory,
    default_state_hook_factory,
    keyword_default_state_hook_factory,
    partial_state_hook,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _compatibility(phase: str) -> ArtifactCompatibility:
    return ArtifactCompatibility(
        dataset_identity=_digest("dataset"),
        split_identity=_digest("splits"),
        row_order_identity=_digest("rows"),
        model_identities={"model": _digest("model")},
        prompt_identities={"prompt": _digest("prompt")},
        configuration_identity=_digest("configuration"),
        seed_identity=_digest("seed"),
        producer_code_identity=_digest(f"producer:{phase}"),
        runtime_compatibility_class="test-runtime",
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _phase_manifest(
    workflow: ProductionAllEvidenceWorkflow,
    *,
    phase: str,
    attempt: Path,
    result: dict[str, object],
) -> dict[str, object]:
    artifacts = _attempt_tree_artifacts(attempt)
    workflow._phase_payload_stat_inventories[phase] = (
        _phase_payload_stat_inventory(attempt, artifacts)
    )
    control = workflow.options.work_root / "phases" / phase / "complete_manifest.json"
    _write_immutable_json(control, {"phase": phase})
    return {
        "attempt_dir": str(attempt.resolve(strict=True)),
        "result": result,
        "artifacts": artifacts,
    }


def _stub_workflow(tmp_path: Path) -> ProductionAllEvidenceWorkflow:
    workflow = object.__new__(ProductionAllEvidenceWorkflow)
    workflow.options = SimpleNamespace(work_root=tmp_path / "work")
    workflow.options.work_root.mkdir()
    compatibilities = {
        phase: _compatibility(phase).as_dict()
        for phase in (
            "input_preparation",
            "embedding_cache",
            "stage1_preflight",
            "stage1_modeling",
            "stage2_canary",
            "stage2_inference",
            "oracle_evaluation",
        )
    }
    physical_identity = Stage1PhysicalFitIdentity(
        architecture_identity=_digest("architectures"),
        target="all_ten_stage1_context_fit_v1",
        scientific_configuration_identity=_digest("configuration"),
        producer_identity=_digest("producer:stage1_modeling"),
        runtime_compatibility_class="test-runtime",
    )
    workflow.request = {
        "portable_typed_workflow": True,
        "expected_checkpoint_compatibilities_by_phase": compatibilities,
        "stage1_physical_fit_identity": physical_identity.as_dict(),
    }
    plan_body = {
        "schema_version": WORKFLOW_EXPECTED_GRANULAR_PLAN_SCHEMA,
        "outer_fold_count": 1,
        "initial_training_partitions": 1,
        "review_rounds": 1,
        "inner_partition_count": 2,
        "outer_fold_ids": [1],
        "stage1_physical_owner_scope_ids": ["outer_001_full"],
        "stage1_logical_scope_ids": ["outer_001_full"],
        "stage1_logical_to_physical_owner": {
            "outer_001_full": "outer_001_full"
        },
        "stage1_physical_fit_count": 1,
        "stage1_logical_scope_count": 1,
        "stage1_artifact_kind_counts": {
            "logical_scope_bindings": 1,
            "neural_query_component": 1,
            "physical_scope_fit": 1,
            "row_map": 1,
            "tfidf_component": 1,
        },
        "stage2_fold_ids": [1],
        "stage2_review_fold_ids": [1],
        "stage2_artifact_kind_counts": {
            "stage2_extraction_component": 1,
            "stage2_fold": 1,
            "stage2_response_component": 1,
            "stage2_review_component": 1,
        },
    }
    workflow.request["expected_granular_checkpoint_plan"] = {
        **plan_body,
        "content_sha256": _sha(plan_body),
    }
    workflow._phase_payload_stat_inventories = {}
    workflow._published_granular_checkpoint_handles = {}
    workflow._published_granular_checkpoint_indexes = {}
    workflow._adopted_artifact_handles = {}
    workflow._published_checkpoint_handles = {}
    return workflow


def _single_scope_plan(
    workflow: ProductionAllEvidenceWorkflow,
    *,
    owner: str,
    fit_rows: tuple[int, ...],
    seed: int,
) -> SimpleNamespace:
    identity = Stage1PhysicalFitIdentity.from_mapping(
        workflow.request["stage1_physical_fit_identity"]
    )
    key = PhysicalFitKey(
        architecture_identity=identity.architecture_identity,
        target=identity.target,
        fit_row_order_identity=ordered_row_identity(fit_rows),
        scientific_configuration_identity=(
            identity.scientific_configuration_identity
        ),
        canonical_group_seed=seed,
        producer_identity=identity.producer_identity,
        runtime_compatibility_class=(
            identity.runtime_compatibility_class
        ),
    )
    scope = SimpleNamespace(scope_id=owner)
    return SimpleNamespace(
        physical_fit_identity=identity,
        physical_scopes=(scope,),
        scopes=(scope,),
        physical_owner=lambda scope_id: (
            scope
            if scope_id == owner
            else (_ for _ in ()).throw(ValueError(scope_id))
        ),
        physical_fit_key=lambda scope_id: (
            key
            if scope_id == owner
            else (_ for _ in ()).throw(ValueError(scope_id))
        ),
    )


def _standalone_artifact(
    tmp_path: Path,
    *,
    name: str,
    kind: str,
    phase: str,
) -> object:
    payload = tmp_path / f"{name}_payload"
    payload.mkdir()
    (payload / "payload.bin").write_bytes(name.encode("utf-8"))
    return publish_portable_reference_artifact(
        control_root=tmp_path / f"{name}_control",
        payload_root=payload,
        artifact_kind=kind,
        artifact_schema=f"test_{kind}_v1",
        compatibility=_compatibility(phase),
        upstream_artifact_ids=(),
        payload_paths=("payload.bin",),
        artifact_metadata={
            "schema_version": "test_node_v1",
            "producer_phase": phase,
            "node_ordinal": 0,
            "node_key": name,
        },
    )


class _Stage2OnlyProducerMutation(ProductionAllEvidenceWorkflow):
    def _stage2_options(
        self,
        attempt: Path,
        *,
        prefix: str,
    ) -> object:
        return ("stage2-only-producer-mutation", attempt, prefix)


class _Stage2GranularPublisherMutation(
    ProductionAllEvidenceWorkflow
):
    def _publish_stage2_inference_granular_checkpoints(
        self,
        *,
        phase_manifest: object,
    ) -> object:
        return ("mutated-stage2-granular-publisher", phase_manifest)


def _mutated_granular_validator(*args: object, **kwargs: object) -> object:
    return ("mutated-granular-validator", args, kwargs)


def _mutated_request_compatibility_helper(
    *,
    scientific_configuration_body: object,
    phase_code_records: object,
) -> object:
    return (
        "mutated-request-compatibility",
        scientific_configuration_body,
        phase_code_records,
    )


def test_realistic_granular_phase_mappers_publish_all_declared_kinds_without_copy(
    tmp_path: Path,
) -> None:
    workflow = _stub_workflow(tmp_path)
    work_root = workflow.options.work_root

    preflight_attempt = (
        work_root / "phases" / "stage1_preflight" / "attempt_preflight"
    )
    context_root = preflight_attempt / "prepared_stage1_context"
    context_root.mkdir(parents=True)
    context_manifest = context_root / "prepared_stage1_context_manifest.json"
    _write_json(context_manifest, {"content_sha256": _digest("context")})
    (context_root / "prepared_rows.parquet").write_bytes(b"prepared-context")
    preflight_result = {
        "prepared_stage1_context_manifest_path": str(context_manifest.resolve()),
        "prepared_stage1_context_scientific_content_root_sha256": _digest(
            "context-root"
        ),
    }
    preflight_phase = _phase_manifest(
        workflow,
        phase="stage1_preflight",
        attempt=preflight_attempt,
        result=preflight_result,
    )
    clustered = _standalone_artifact(
        tmp_path,
        name="clustered",
        kind="clustered_preflight",
        phase="stage1_preflight",
    )
    preflight_index = workflow._publish_prepared_stage1_context_checkpoint(
        phase_manifest=preflight_phase,
        clustered_preflight=clustered,
    )
    assert preflight_index["coverage"]["artifact_kind_counts"] == {
        "prepared_stage1_context": 1
    }

    modeling_attempt = (
        work_root / "phases" / "stage1_modeling" / "attempt_modeling"
    )
    owner = "outer_001_full"
    logical = "outer_001_full"
    for component in ("tfidf", "neural_query", "bow"):
        component_root = (
            modeling_attempt
            / "role_neutral_stage1_execution"
            / "components"
            / owner
            / component
        )
        component_root.mkdir(parents=True)
        _write_json(
            component_root / "execution_manifest.json",
            {"component": component},
        )
        (component_root / "evidence.npy").write_bytes(
            f"{component}-evidence".encode("utf-8")
        )
    binding_root = (
        modeling_attempt
        / "role_neutral_stage1_execution"
        / "coordination_gate"
        / "scientific_bindings"
    )
    # Production scope descriptors persist positional row IDs as JSON
    # integers.  Keeping that exact scalar type catches string-coercion bugs in
    # the portable PhysicalFitKey bridge.
    fit_rows = (1, 2, 3)
    workflow._authenticated_current_stage1_scope_plan = lambda: (
        _single_scope_plan(
            workflow,
            owner=owner,
            fit_rows=fit_rows,
            seed=42,
        )
    )
    physical_relative = f"physical_fit_payloads/{owner}.json"
    logical_relative = f"logical_views/{logical}.json"
    _write_json(
        binding_root / physical_relative,
        {
            "physical_owner_scope_id": owner,
            "fit_row_ids": list(fit_rows),
            "fit_row_order_fingerprint": ordered_row_identity(fit_rows),
            "canonical_group_seed": 42,
        },
    )
    _write_json(
        binding_root / logical_relative,
        {
            "logical_scope_id": logical,
            "physical_owner_scope_id": owner,
            "logical_purpose": "full_outer",
        },
    )
    _write_json(
        binding_root / "role_neutral_binding_set.json",
        {
            "physical_payloads": [
                {
                    "physical_owner_scope_id": owner,
                    "relative_path": physical_relative,
                }
            ],
            "logical_views": [
                {
                    "logical_scope_id": logical,
                    "relative_path": logical_relative,
                }
            ],
        },
    )
    bundle = modeling_attempt / "stage1_bundle"
    bundle.mkdir(parents=True)
    bundle_manifest = bundle / "bundle_manifest.json"
    _write_json(bundle_manifest, {"bundle_sha256": _digest("bundle")})
    (bundle / "row_registry.parquet").write_bytes(b"row-map")
    numerical_bank = (
        modeling_attempt / "direct_upstream_numerical_reference_bank"
    )
    numerical_bank.mkdir()
    _write_json(numerical_bank / "bank_manifest.json", {"rows": 3})
    handoff_binding = (
        modeling_attempt / "role_neutral_handoff_binding.json"
    )
    _write_json(handoff_binding, {"bound": True})
    stage1_terminal_inventory = (
        _portable_stage1_terminal_file_inventory(
            execution_root=(
                modeling_attempt
                / "role_neutral_stage1_execution"
            ),
            bundle_root=bundle,
            numerical_bank_root=numerical_bank,
            binding_path=handoff_binding,
        )
    )
    assert set(stage1_terminal_inventory) == {
        str(path.resolve())
        for path in modeling_attempt.rglob("*")
        if path.is_file()
    }
    modeling_result = {
        "schema_version": PORTABLE_ROLE_NEUTRAL_STAGE1_PHASE_SCHEMA,
        "role_neutral_execution_root": str(
            (
                modeling_attempt / "role_neutral_stage1_execution"
            ).resolve()
        ),
        "bundle_manifest_path": str(bundle_manifest.resolve()),
        "physical_fit_count": 1,
        "logical_scope_count": 1,
    }
    modeling_phase = _phase_manifest(
        workflow,
        phase="stage1_modeling",
        attempt=modeling_attempt,
        result=modeling_result,
    )
    modeling_index = (
        workflow._publish_stage1_modeling_granular_checkpoints(
            phase_manifest=modeling_phase,
        )
    )
    assert modeling_index["coverage"]["artifact_kind_counts"] == {
        "logical_scope_bindings": 1,
        "neural_query_component": 1,
        "physical_scope_fit": 1,
        "row_map": 1,
        "tfidf_component": 1,
    }
    physical_node = next(
        node
        for node in modeling_index["nodes"]
        if node["artifact_kind"] == "physical_scope_fit"
    )
    assert physical_node["artifact_metadata"]["physical_fit_key_record"][
        "content_sha256"
    ] == physical_node["artifact_metadata"]["physical_fit_key"]

    stage1_base = _standalone_artifact(
        tmp_path,
        name="stage1_base",
        kind="stage1_handoff",
        phase="stage1_modeling",
    )
    canary_base = _standalone_artifact(
        tmp_path,
        name="canary_base",
        kind="stage2_canary",
        phase="stage2_canary",
    )
    workflow._checkpoint_artifact_for_phase = lambda phase, required: {
        "stage1_modeling": stage1_base,
        "stage2_canary": canary_base,
    }[phase]

    inference_attempt = (
        work_root / "phases" / "stage2_inference" / "attempt_inference"
    )
    response_root = inference_attempt / "full_preparation"
    response_root.mkdir(parents=True)
    batch = response_root / "authenticated_hierarchical_batch_result.json"
    _write_json(batch, {"body": {"ordered_fold_results": [1]}})
    _write_json(response_root / "response_page_000.json", {"status": "ok"})
    ledger_root = inference_attempt / "extraction_ledger"
    ledger_root.mkdir(parents=True)
    ledger_paths = []
    for name in (
        "complete_paged_ledger.json",
        "page_table.parquet",
        "reconciliation_table.parquet",
    ):
        path = ledger_root / name
        path.write_bytes(name.encode("utf-8"))
        ledger_paths.append(str(path.resolve()))
    fold_root = inference_attempt / "full" / "outer_fold_001"
    review_root = fold_root / "post_extraction_review"
    review_root.mkdir(parents=True)
    _write_json(review_root / "round_000.json", {"accepted": True})
    fold_manifest = fold_root / "immutable_fold_manifest.json"
    _write_json(fold_manifest, {"body": {"outer_fold": 1}})
    fold_prediction = fold_root / "frozen_predictions.parquet"
    fold_prediction.write_bytes(b"fold-prediction")
    output_root = inference_attempt / "full"
    runner_input = output_root / "immutable_input_manifest.json"
    combined_prediction = output_root / "frozen_predictions.parquet"
    run_manifest = output_root / "immutable_run_manifest.json"
    attestation = (
        inference_attempt / "attestation" / "stage2_result.json"
    )
    _write_json(runner_input, {"body": {"input": True}})
    combined_prediction.write_bytes(b"combined-prediction")
    _write_json(run_manifest, {"body": {"frozen": True}})
    _write_json(attestation, {"status": "complete"})
    inference_result = {
        "mode": "reference_only_role_neutral_stage2",
        "runner_input_manifest_path": str(runner_input.resolve()),
        "hierarchical_batch_result_path": str(batch.resolve()),
        "complete_paged_ledger_artifact_paths": ledger_paths,
        "fold_manifest_paths": [str(fold_manifest.resolve())],
        "fold_prediction_paths": [str(fold_prediction.resolve())],
    }
    terminal_inventory = _portable_stage2_terminal_file_inventory(
        result=inference_result,
        prediction_path=combined_prediction,
        run_manifest_path=run_manifest,
        attestation_path=attestation,
    )
    assert set(terminal_inventory) == {
        str(path.resolve())
        for path in (
            runner_input,
            batch,
            response_root / "response_page_000.json",
            *(Path(path) for path in ledger_paths),
            review_root / "round_000.json",
            fold_manifest,
            fold_prediction,
            combined_prediction,
            run_manifest,
            attestation,
        )
    }
    inference_phase = _phase_manifest(
        workflow,
        phase="stage2_inference",
        attempt=inference_attempt,
        result=inference_result,
    )
    inference_index = (
        workflow._publish_stage2_inference_granular_checkpoints(
            phase_manifest=inference_phase,
        )
    )
    assert inference_index["coverage"]["artifact_kind_counts"] == {
        "stage2_extraction_component": 1,
        "stage2_fold": 1,
        "stage2_response_component": 1,
        "stage2_review_component": 1,
    }

    all_observed_kinds = {
        node["artifact_kind"]
        for index in (
            preflight_index,
            modeling_index,
            inference_index,
        )
        for node in index["nodes"]
    }
    assert all_observed_kinds == set(
        GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS
    )
    for phase, expected in (
        ("stage1_preflight", preflight_index),
        ("stage1_modeling", modeling_index),
        ("stage2_inference", inference_index),
    ):
        validated, handles = (
            _validate_granular_checkpoint_index_from_paths(
                work_root=work_root,
                phase=phase,
                compatibility=_compatibility(phase),
                payload_authentication_cache={},
            )
        )
        assert validated == expected
        assert len(handles) == expected["node_count"]
        for artifact in handles:
            assert {
                path.name for path in artifact.root.iterdir()
            } == {
                "artifact_manifest.json",
                "artifact_locator.json",
            }
            assert artifact.payload_root != artifact.root
            for payload in artifact.payloads:
                assert (
                    os.lstat(
                        artifact.payload_root / payload.relative_path
                    ).st_nlink
                    == 1
                )

    primary_metadata = _granular_primary_metadata_from_index(
        phase="stage1_modeling",
        index=modeling_index,
    )
    expected_terminal_ids = [
        node["artifact_id"]
        for node in modeling_index["nodes"]
        if node["artifact_kind"]
        in {"logical_scope_bindings", "row_map"}
    ]
    assert (
        primary_metadata["granular_terminal_artifact_ids"]
        == expected_terminal_ids
    )


@pytest.mark.parametrize("scenario", ("missing_review", "missing_fold"))
def test_stage2_mapper_rejects_request_plan_coverage_gaps(
    tmp_path: Path,
    scenario: str,
) -> None:
    workflow = _stub_workflow(tmp_path)
    if scenario == "missing_fold":
        plan = dict(
            workflow.request["expected_granular_checkpoint_plan"]
        )
        plan.update(
            {
                "outer_fold_count": 2,
                "outer_fold_ids": [1, 2],
                "stage2_fold_ids": [1, 2],
                "stage2_review_fold_ids": [1, 2],
                "stage2_artifact_kind_counts": {
                    "stage2_extraction_component": 1,
                    "stage2_fold": 2,
                    "stage2_response_component": 1,
                    "stage2_review_component": 2,
                },
            }
        )
        plan_body = {
            key: value
            for key, value in plan.items()
            if key != "content_sha256"
        }
        workflow.request["expected_granular_checkpoint_plan"] = {
            **plan_body,
            "content_sha256": _sha(plan_body),
        }
    stage1 = _standalone_artifact(
        tmp_path,
        name="stage1_for_gap",
        kind="stage1_handoff",
        phase="stage1_modeling",
    )
    canary = _standalone_artifact(
        tmp_path,
        name="canary_for_gap",
        kind="stage2_canary",
        phase="stage2_canary",
    )
    workflow._checkpoint_artifact_for_phase = (
        lambda phase, required: {
            "stage1_modeling": stage1,
            "stage2_canary": canary,
        }[phase]
    )
    attempt = (
        workflow.options.work_root
        / "phases"
        / "stage2_inference"
        / "attempt_gap"
    )
    response_root = attempt / "full_preparation"
    response_root.mkdir(parents=True)
    batch = response_root / "batch.json"
    _write_json(batch, {"status": "complete"})
    ledger = attempt / "extraction_ledger.json"
    _write_json(ledger, {"status": "complete"})
    fold_root = attempt / "fold_001"
    fold_root.mkdir()
    fold_manifest = fold_root / "fold_manifest.json"
    _write_json(fold_manifest, {"body": {"outer_fold": 1}})
    prediction = fold_root / "prediction.parquet"
    prediction.write_bytes(b"prediction")
    if scenario != "missing_review":
        review = fold_root / "post_extraction_review"
        review.mkdir()
        _write_json(review / "review.json", {"status": "complete"})
    result = {
        "mode": "reference_only_role_neutral_stage2",
        "hierarchical_batch_result_path": str(batch.resolve()),
        "complete_paged_ledger_artifact_paths": [
            str(ledger.resolve())
        ],
        "fold_manifest_paths": [str(fold_manifest.resolve())],
        "fold_prediction_paths": [str(prediction.resolve())],
    }
    phase_manifest = _phase_manifest(
        workflow,
        phase="stage2_inference",
        attempt=attempt,
        result=result,
    )
    with pytest.raises(RuntimeError, match="fold|review"):
        workflow._publish_stage2_inference_granular_checkpoints(
            phase_manifest=phase_manifest,
        )


def _two_node_index(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], ArtifactCompatibility]:
    work_root = tmp_path / "work"
    phase = "stage2_inference"
    attempt = work_root / "phases" / phase / "attempt_test"
    attempt.mkdir(parents=True)
    first_payload = attempt / "response.json"
    second_payload = attempt / "fold.parquet"
    first_payload.write_bytes(b"response")
    second_payload.write_bytes(b"fold")
    _write_immutable_json(
        work_root / "phases" / phase / "complete_manifest.json",
        {"phase": phase},
    )
    compatibility = _compatibility(phase)
    root, index_path, locator_path = _granular_checkpoint_index_paths(
        work_root=work_root,
        phase=phase,
    )
    (root / "nodes").mkdir(parents=True)
    first = publish_portable_reference_artifact(
        control_root=root / "nodes" / "00000-response",
        payload_root=attempt,
        artifact_kind="stage2_response_component",
        artifact_schema=GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS[
            "stage2_response_component"
        ],
        compatibility=compatibility,
        upstream_artifact_ids=(),
        payload_paths=("response.json",),
        artifact_metadata={
            "schema_version": "production_workflow_granular_checkpoint_node_v1",
            "producer_phase": phase,
            "node_ordinal": 0,
            "node_key": "response",
        },
        payload_inventory_policy=REGISTERED_PAYLOAD_PATHS_ONLY,
    )
    second = publish_portable_reference_artifact(
        control_root=root / "nodes" / "00001-fold",
        payload_root=attempt,
        artifact_kind="stage2_fold",
        artifact_schema=GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS[
            "stage2_fold"
        ],
        compatibility=compatibility,
        upstream_artifact_ids=(first.artifact_id,),
        payload_paths=("fold.parquet",),
        artifact_metadata={
            "schema_version": "production_workflow_granular_checkpoint_node_v1",
            "producer_phase": phase,
            "node_ordinal": 1,
            "node_key": "fold",
        },
        payload_inventory_policy=REGISTERED_PAYLOAD_PATHS_ONLY,
    )
    nodes = [
        {
            "node_ordinal": ordinal,
            "node_key": artifact.artifact_metadata["node_key"],
            "artifact_id": artifact.artifact_id,
            "artifact_kind": artifact.manifest["artifact_kind"],
            "artifact_schema": artifact.manifest["artifact_schema"],
            "upstream_artifact_ids": list(
                artifact.manifest["upstream_artifact_ids"]
            ),
            "artifact_metadata": dict(artifact.artifact_metadata),
        }
        for ordinal, artifact in enumerate((first, second))
    ]
    body = {
        "schema_version": WORKFLOW_GRANULAR_CHECKPOINT_INDEX_SCHEMA,
        "phase": phase,
        "node_count": len(nodes),
        "nodes": nodes,
        "coverage": _granular_checkpoint_coverage(nodes),
        "relative_filesystem_layout_included": False,
    }
    index = {**body, "content_sha256": _sha(body)}
    _write_immutable_json(index_path, index)
    phase_manifest = (
        work_root / "phases" / phase / "complete_manifest.json"
    ).resolve(strict=True)
    phase_bytes = phase_manifest.read_bytes()
    locator_body = {
        "schema_version": WORKFLOW_GRANULAR_CHECKPOINT_LOCATOR_SCHEMA,
        "phase": phase,
        "index_content_sha256": index["content_sha256"],
        "index_path": str(index_path.resolve(strict=True)),
        "phase_manifest_path": str(phase_manifest),
        "phase_manifest_sha256": hashlib.sha256(phase_bytes).hexdigest(),
        "phase_manifest_size_bytes": len(phase_bytes),
        "node_controls": [
            {
                "node_ordinal": ordinal,
                "artifact_id": artifact.artifact_id,
                "control_root": str(artifact.root),
            }
            for ordinal, artifact in enumerate((first, second))
        ],
    }
    _write_immutable_json(
        locator_path,
        {**locator_body, "content_sha256": _sha(locator_body)},
    )
    return work_root, index, compatibility


@pytest.mark.parametrize(
    "mutation",
    ("payload", "missing", "extra", "reorder", "symlink", "hardlink"),
)
def test_granular_index_fails_closed_for_tree_substitution(
    tmp_path: Path,
    mutation: str,
) -> None:
    work_root, _index, compatibility = _two_node_index(tmp_path)
    phase = "stage2_inference"
    root, index_path, _locator_path = _granular_checkpoint_index_paths(
        work_root=work_root,
        phase=phase,
    )
    controls = sorted((root / "nodes").iterdir())
    if mutation == "payload":
        (
            work_root
            / "phases"
            / phase
            / "attempt_test"
            / "response.json"
        ).write_bytes(b"tampered")
    elif mutation == "missing":
        shutil.rmtree(controls[0])
    elif mutation == "extra":
        (root / "nodes" / "unregistered").mkdir()
    elif mutation == "reorder":
        value = json.loads(index_path.read_text(encoding="utf-8"))
        value["nodes"] = list(reversed(value["nodes"]))
        os.chmod(index_path, 0o644)
        _write_json(index_path, value)
    elif mutation == "symlink":
        target = tmp_path / "substitute"
        shutil.copytree(controls[0], target)
        shutil.rmtree(controls[0])
        controls[0].symlink_to(target, target_is_directory=True)
    elif mutation == "hardlink":
        os.link(index_path, tmp_path / "index_alias.json")
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(
        (ValueError, RuntimeError, FileNotFoundError)
    ):
        _validate_granular_checkpoint_index_from_paths(
            work_root=work_root,
            phase=phase,
            compatibility=compatibility,
            payload_authentication_cache={},
        )


@pytest.mark.parametrize(
    "field",
    (
        "granular_index_content_sha256",
        "granular_coverage_content_sha256",
    ),
)
def test_adoption_recomputes_primary_granular_digest_claims(
    tmp_path: Path,
    field: str,
) -> None:
    work_root, index, compatibility = _two_node_index(tmp_path)
    _validated, handles = _validate_granular_checkpoint_index_from_paths(
        work_root=work_root,
        phase="stage2_inference",
        compatibility=compatibility,
    )
    metadata = dict(
        _granular_primary_metadata_from_index(
            phase="stage2_inference",
            index=index,
        )
    )
    _validate_primary_granular_binding_digests(
        phase="stage2_inference",
        primary_metadata=metadata,
        artifacts=handles,
    )
    metadata[field] = _digest(f"tampered:{field}")
    with pytest.raises(ValueError, match="digest binding"):
        _validate_primary_granular_binding_digests(
            phase="stage2_inference",
            primary_metadata=metadata,
            artifacts=handles,
        )


def test_request_plan_rejects_self_consistent_missing_logical_and_fold_nodes(
) -> None:
    plan = _derive_expected_granular_checkpoint_plan(
        outer_folds=2,
        initial_training_partitions=1,
        review_rounds=1,
    )

    def artifact(
        kind: str,
        ordinal: int,
        **metadata: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            artifact_id=_digest(f"{kind}:{ordinal}:{metadata}"),
            manifest={"artifact_kind": kind},
            artifact_metadata=metadata,
        )

    physical = list(plan["stage1_physical_owner_scope_ids"])
    logical = list(plan["stage1_logical_scope_ids"])
    logical_to_owner = dict(
        plan["stage1_logical_to_physical_owner"]
    )
    stage1 = []
    ordinal = 0
    for kind in (
        "tfidf_component",
        "neural_query_component",
        "physical_scope_fit",
    ):
        for owner in physical:
            stage1.append(
                artifact(
                    kind,
                    ordinal,
                    physical_owner_scope_id=owner,
                )
            )
            ordinal += 1
    for logical_id in logical[:-1]:
        stage1.append(
            artifact(
                "logical_scope_bindings",
                ordinal,
                logical_scope_id=logical_id,
                physical_owner_scope_id=logical_to_owner[logical_id],
            )
        )
        ordinal += 1
    stage1.append(artifact("row_map", ordinal))
    with pytest.raises(ValueError, match="request plan"):
        _validate_granular_handles_against_plan(
            phase="stage1_modeling",
            artifacts=stage1,
            expected_plan=plan,
        )

    stage2 = [
        artifact("stage2_response_component", 0),
        artifact("stage2_extraction_component", 1),
        artifact("stage2_review_component", 2, outer_fold=1),
        artifact("stage2_review_component", 3, outer_fold=2),
        artifact("stage2_fold", 4, outer_fold=1),
    ]
    with pytest.raises(ValueError, match="request plan"):
        _validate_granular_handles_against_plan(
            phase="stage2_inference",
            artifacts=stage2,
            expected_plan=plan,
        )


def test_representative_request_plan_derives_35_physical_from_40_logical(
) -> None:
    plan = _derive_expected_granular_checkpoint_plan(
        outer_folds=5,
        initial_training_partitions=3,
        review_rounds=2,
    )
    assert plan["stage1_physical_fit_count"] == 35
    assert plan["stage1_logical_scope_count"] == 40
    assert (
        len(plan["stage1_logical_to_physical_owner"])
        == 40
    )
    for outer_fold in range(1, 6):
        assert plan["stage1_logical_to_physical_owner"][
            f"outer_{outer_fold:03d}_hierarchy_epoch_001"
        ] == f"outer_{outer_fold:03d}_inner_005"


def test_stage1_granular_key_requires_exact_request_identity_and_full_record(
) -> None:
    identity = Stage1PhysicalFitIdentity(
        architecture_identity=_digest("architecture"),
        target="target",
        scientific_configuration_identity=_digest("configuration"),
        producer_identity=_digest("producer"),
        runtime_compatibility_class="runtime",
    )
    key = PhysicalFitKey(
        architecture_identity=identity.architecture_identity,
        target=identity.target,
        fit_row_order_identity=_digest("row-order"),
        scientific_configuration_identity=(
            identity.scientific_configuration_identity
        ),
        canonical_group_seed=42,
        producer_identity=identity.producer_identity,
        runtime_compatibility_class=(
            identity.runtime_compatibility_class
        ),
    )
    metadata = {
        "physical_owner_scope_id": "outer_001_full",
        "physical_fit_key": key.key,
        "physical_fit_key_record": key.as_dict(),
    }
    owner, record = _validated_stage1_granular_physical_fit_key(
        metadata=metadata,
        expected_identity=identity,
        expected_key_record=key.as_dict(),
    )
    assert owner == "outer_001_full"
    assert record == key.as_dict()
    changed = dict(record)
    changed["canonical_group_seed"] = 43
    with pytest.raises(ValueError, match="full physical-fit key"):
        _validated_stage1_granular_physical_fit_key(
            metadata={
                **metadata,
                "physical_fit_key_record": changed,
            },
            expected_identity=identity,
            expected_key_record=key.as_dict(),
        )
    self_consistent_wrong_key = PhysicalFitKey(
        architecture_identity=identity.architecture_identity,
        target=identity.target,
        fit_row_order_identity=_digest("other-row-order"),
        scientific_configuration_identity=(
            identity.scientific_configuration_identity
        ),
        canonical_group_seed=43,
        producer_identity=identity.producer_identity,
        runtime_compatibility_class=(
            identity.runtime_compatibility_class
        ),
    )
    with pytest.raises(ValueError, match="full physical-fit key"):
        _validated_stage1_granular_physical_fit_key(
            metadata={
                **metadata,
                "physical_fit_key": self_consistent_wrong_key.key,
                "physical_fit_key_record": (
                    self_consistent_wrong_key.as_dict()
                ),
            },
            expected_identity=identity,
            expected_key_record=key.as_dict(),
        )
    other_identity = Stage1PhysicalFitIdentity(
        architecture_identity=_digest("other-architecture"),
        target=identity.target,
        scientific_configuration_identity=(
            identity.scientific_configuration_identity
        ),
        producer_identity=identity.producer_identity,
        runtime_compatibility_class=(
            identity.runtime_compatibility_class
        ),
    )
    with pytest.raises(ValueError, match="full physical-fit key"):
        _validated_stage1_granular_physical_fit_key(
            metadata=metadata,
            expected_identity=other_identity,
            expected_key_record=key.as_dict(),
        )


def test_exact_granular_edges_reject_self_consistent_rewiring(
    tmp_path: Path,
) -> None:
    workflow = _stub_workflow(tmp_path)
    plan = workflow.request["expected_granular_checkpoint_plan"]
    owner = "outer_001_full"
    scope_plan = _single_scope_plan(
        workflow,
        owner=owner,
        fit_rows=(1, 2, 3),
        seed=42,
    )
    physical_key = scope_plan.physical_fit_key(owner)

    def node(
        name: str,
        kind: str,
        upstream: tuple[str, ...],
        **metadata: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            artifact_id=_digest(
                json.dumps(
                    {
                        "name": name,
                        "kind": kind,
                        "upstream": upstream,
                    },
                    sort_keys=True,
                )
            ),
            manifest={
                "artifact_kind": kind,
                "upstream_artifact_ids": list(upstream),
            },
            artifact_metadata=metadata,
        )

    prepared_context_id = _digest("prepared-context")
    key_metadata = {
        "physical_owner_scope_id": owner,
        "physical_fit_key": physical_key.key,
        "physical_fit_key_record": physical_key.as_dict(),
    }
    tfidf = node(
        "tfidf",
        "tfidf_component",
        (prepared_context_id,),
        **key_metadata,
    )
    neural = node(
        "neural",
        "neural_query_component",
        (prepared_context_id,),
        **key_metadata,
    )
    physical = node(
        "physical",
        "physical_scope_fit",
        (prepared_context_id, tfidf.artifact_id, neural.artifact_id),
        **key_metadata,
    )
    logical = node(
        "logical",
        "logical_scope_bindings",
        (physical.artifact_id,),
        logical_scope_id=owner,
        **key_metadata,
    )
    row_map = node(
        "row-map",
        "row_map",
        (logical.artifact_id,),
    )
    stage1 = (tfidf, neural, physical, logical, row_map)
    _validate_granular_handles_against_plan(
        phase="stage1_modeling",
        artifacts=stage1,
        expected_plan=plan,
        expected_stage1_scope_plan=scope_plan,
        expected_external_upstream_artifact_ids=(
            prepared_context_id,
        ),
    )

    rewired_physical = node(
        "physical-rewired",
        "physical_scope_fit",
        (prepared_context_id, neural.artifact_id, tfidf.artifact_id),
        **key_metadata,
    )
    rewired_logical = node(
        "logical-rewired",
        "logical_scope_bindings",
        (rewired_physical.artifact_id,),
        logical_scope_id=owner,
        **key_metadata,
    )
    rewired_row_map = node(
        "row-map-rewired",
        "row_map",
        (rewired_logical.artifact_id,),
    )
    with pytest.raises(ValueError, match="upstream edge"):
        _validate_granular_handles_against_plan(
            phase="stage1_modeling",
            artifacts=(
                tfidf,
                neural,
                rewired_physical,
                rewired_logical,
                rewired_row_map,
            ),
            expected_plan=plan,
            expected_stage1_scope_plan=scope_plan,
            expected_external_upstream_artifact_ids=(
                prepared_context_id,
            ),
        )

    stage1_id = _digest("stage1-handoff")
    canary_id = _digest("stage2-canary")
    response = node(
        "response",
        "stage2_response_component",
        (stage1_id, canary_id),
    )
    extraction = node(
        "extraction",
        "stage2_extraction_component",
        (response.artifact_id,),
    )
    review = node(
        "review",
        "stage2_review_component",
        (extraction.artifact_id,),
        outer_fold=1,
    )
    fold = node(
        "fold",
        "stage2_fold",
        (
            response.artifact_id,
            extraction.artifact_id,
            review.artifact_id,
        ),
        outer_fold=1,
    )
    _validate_granular_handles_against_plan(
        phase="stage2_inference",
        artifacts=(response, extraction, review, fold),
        expected_plan=plan,
        expected_external_upstream_artifact_ids=(
            stage1_id,
            canary_id,
        ),
    )

    rewired_extraction = node(
        "extraction-rewired",
        "stage2_extraction_component",
        (stage1_id,),
    )
    rewired_review = node(
        "review-rewired",
        "stage2_review_component",
        (rewired_extraction.artifact_id,),
        outer_fold=1,
    )
    rewired_fold = node(
        "fold-rewired",
        "stage2_fold",
        (
            response.artifact_id,
            rewired_extraction.artifact_id,
            rewired_review.artifact_id,
        ),
        outer_fold=1,
    )
    with pytest.raises(ValueError, match="upstream edge"):
        _validate_granular_handles_against_plan(
            phase="stage2_inference",
            artifacts=(
                response,
                rewired_extraction,
                rewired_review,
                rewired_fold,
            ),
            expected_plan=plan,
            expected_external_upstream_artifact_ids=(
                stage1_id,
                canary_id,
            ),
        )


def test_ordered_row_identity_preserves_json_scalar_types_and_order() -> None:
    assert ordered_row_identity((1, 2, 3)) != ordered_row_identity(
        ("1", "2", "3")
    )
    assert ordered_row_identity((1, 2, 3)) != ordered_row_identity(
        (3, 2, 1)
    )
    with pytest.raises(TypeError, match="string or integer"):
        ordered_row_identity((True,))


def test_portable_content_root_binds_logical_roles_and_relocation_preserves_it(
    tmp_path: Path,
) -> None:
    compatibility = _compatibility("stage1_modeling")
    handles = []
    for name, relative in (
        ("first", "nested/a.bin"),
        ("second", "other/name.bin"),
    ):
        payload = tmp_path / f"{name}_payload"
        path = payload / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(b"identical-scientific-bytes")
        handles.append(
            publish_portable_reference_artifact(
                control_root=tmp_path / f"{name}_control",
                payload_root=payload,
                artifact_kind="tfidf_component",
                artifact_schema=GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS[
                    "tfidf_component"
                ],
                compatibility=compatibility,
                upstream_artifact_ids=(),
                payload_paths=(relative,),
                artifact_metadata={"node_key": "same-node"},
                payload_inventory_policy=COMPLETE_PAYLOAD_TREE,
            )
        )
    assert handles[0].artifact_id != handles[1].artifact_id
    relocated = relocate_portable_artifact(
        source=handles[0].root,
        target_root=tmp_path / "relocated",
    )
    assert relocated.artifact_id == handles[0].artifact_id
    assert relocated.payload_root != handles[0].payload_root


def test_portable_content_root_binds_payload_role_assignment_and_order(
    tmp_path: Path,
) -> None:
    compatibility = _compatibility("stage1_modeling")

    def publish(
        name: str,
        *,
        order: tuple[str, ...],
        values: dict[str, bytes],
    ) -> object:
        payload = tmp_path / f"{name}_payload"
        payload.mkdir()
        for relative, value in values.items():
            (payload / relative).write_bytes(value)
        return publish_portable_reference_artifact(
            control_root=tmp_path / f"{name}_control",
            payload_root=payload,
            artifact_kind="tfidf_component",
            artifact_schema=GRANULAR_CHECKPOINT_ARTIFACT_SCHEMAS[
                "tfidf_component"
            ],
            compatibility=compatibility,
            upstream_artifact_ids=(),
            payload_paths=order,
            payload_inventory_policy=COMPLETE_PAYLOAD_TREE,
        )

    baseline = publish(
        "baseline",
        order=("design.npy", "evidence.npy"),
        values={"design.npy": b"design", "evidence.npy": b"evidence"},
    )
    relocated_layout = publish(
        "same_roles_elsewhere",
        order=("design.npy", "evidence.npy"),
        values={"design.npy": b"design", "evidence.npy": b"evidence"},
    )
    role_swap = publish(
        "role_swap",
        order=("design.npy", "evidence.npy"),
        values={"design.npy": b"evidence", "evidence.npy": b"design"},
    )
    reordered = publish(
        "reordered",
        order=("evidence.npy", "design.npy"),
        values={"design.npy": b"design", "evidence.npy": b"evidence"},
    )
    assert baseline.artifact_id == relocated_layout.artifact_id
    assert baseline.artifact_id != role_swap.artifact_id
    assert baseline.artifact_id != reordered.artifact_id


def test_operational_phase_result_does_not_change_scientific_content_root(
    tmp_path: Path,
) -> None:
    compatibility = _compatibility("embedding_cache")
    artifacts = []
    for name, operational in (
        (
            "gpu_zero",
            {
                "gpu_ids": [0],
                "gpu_uuids": ["GPU-machine-a"],
                "worker_pid": 111,
            },
        ),
        (
            "gpu_seven",
            {
                "gpu_ids": [7],
                "gpu_uuids": ["GPU-machine-b"],
                "worker_pid": 999,
            },
        ),
    ):
        payload = tmp_path / f"{name}_payload"
        payload.mkdir()
        terminal = payload / "cache.npy"
        terminal.write_bytes(b"same-scientific-cache")
        artifacts.append(
            publish_portable_reference_artifact(
                control_root=tmp_path / f"{name}_control",
                payload_root=payload,
                artifact_kind="embedding_cache",
                artifact_schema="test_embedding_cache_v1",
                compatibility=compatibility,
                upstream_artifact_ids=(),
                payload_paths=("cache.npy",),
                workflow_phase="embedding_cache",
                workflow_phase_result={
                    "terminal_files": [str(terminal.resolve())],
                    "resource_preflight": operational,
                    "execution_uuid": name,
                },
            )
        )
    assert artifacts[0].artifact_id == artifacts[1].artifact_id
    assert (
        artifacts[0].phase_binding["content_sha256"]
        != artifacts[1].phase_binding["content_sha256"]
    )


def test_transitive_source_inventory_resolves_relative_import_alias_modules(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    package = repository / "oci" / "example"
    package.mkdir(parents=True)
    (repository / "oci" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "root.py").write_text(
        "from . import sibling\n"
        "from oci.example import nested\n",
        encoding="utf-8",
    )
    (package / "sibling.py").write_text(
        "from . import leaf\n",
        encoding="utf-8",
    )
    (package / "nested.py").write_text("", encoding="utf-8")
    (package / "leaf.py").write_text("", encoding="utf-8")

    direct = _local_import_paths(
        package / "root.py",
        repository_root=repository,
    )
    assert {
        path.relative_to(repository).as_posix() for path in direct
    } == {
        "oci/example/__init__.py",
        "oci/example/nested.py",
        "oci/example/sibling.py",
    }
    inventory = _transitive_local_source_inventory(
        repository_root=repository,
        roots=("oci/example/root.py",),
    )
    assert {
        row["relative_path"] for row in inventory
    } == {
        "oci/example/__init__.py",
        "oci/example/leaf.py",
        "oci/example/nested.py",
        "oci/example/root.py",
        "oci/example/sibling.py",
    }


def test_transitive_source_inventory_memoizes_shared_imports_and_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    package = repository / "oci" / "memo"
    package.mkdir(parents=True)
    (repository / "oci" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in ("first", "second"):
        (package / f"{name}.py").write_text(
            "from . import shared\n",
            encoding="utf-8",
        )
    shared = package / "shared.py"
    shared.write_text("VALUE = 1\n", encoding="utf-8")

    from oci.inference import production_all_evidence_workflow as module

    real_hash = module.stable_file_sha256
    real_imports = module._local_import_paths
    hash_counts: dict[Path, int] = {}
    import_counts: dict[Path, int] = {}

    def counted_hash(path: Path) -> tuple[str, int]:
        resolved = Path(path).resolve(strict=True)
        hash_counts[resolved] = hash_counts.get(resolved, 0) + 1
        return real_hash(resolved)

    def counted_imports(
        path: Path,
        *,
        repository_root: Path,
    ) -> tuple[Path, ...]:
        resolved = Path(path).resolve(strict=True)
        import_counts[resolved] = import_counts.get(resolved, 0) + 1
        return real_imports(
            resolved,
            repository_root=repository_root,
        )

    monkeypatch.setattr(module, "stable_file_sha256", counted_hash)
    monkeypatch.setattr(module, "_local_import_paths", counted_imports)
    import_cache: dict[Path, tuple[Path, ...]] = {}
    file_cache: dict[Path, object] = {}
    for root in ("oci/memo/first.py", "oci/memo/second.py"):
        _transitive_local_source_inventory(
            repository_root=repository,
            roots=(root,),
            import_cache=import_cache,
            file_identity_cache=file_cache,
        )
    assert hash_counts[shared.resolve()] == 1
    assert import_counts[shared.resolve()] == 1


def test_stat_guarded_identity_memo_is_exact_fast_and_invalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oci.inference import production_all_evidence_workflow as module

    repository = tmp_path / "repository"
    package = repository / "identity_chain"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    source_paths: list[Path] = []
    for index in range(12):
        source = package / f"module_{index}.py"
        next_import = (
            ""
            if index == 11
            else f"from . import module_{index + 1}\n"
        )
        source.write_text(
            f"{next_import}VALUE = {index}\n",
            encoding="utf-8",
        )
        source_paths.append(source.resolve())

    real_hash = module.stable_file_sha256
    real_parser = module._parse_local_import_module_names
    hash_counts: dict[Path, int] = {}
    parse_counts: dict[Path, int] = {}

    def delayed_hash(path: Path) -> tuple[str, int]:
        resolved = Path(path).resolve(strict=True)
        hash_counts[resolved] = hash_counts.get(resolved, 0) + 1
        time.sleep(0.01)
        return real_hash(resolved)

    def delayed_parser(
        path: Path,
        *,
        repository_root: Path,
    ) -> tuple[str, ...]:
        resolved = Path(path).resolve(strict=True)
        parse_counts[resolved] = parse_counts.get(resolved, 0) + 1
        time.sleep(0.01)
        return real_parser(
            resolved,
            repository_root=repository_root,
        )

    monkeypatch.setattr(module, "stable_file_sha256", delayed_hash)
    monkeypatch.setattr(
        module,
        "_parse_local_import_module_names",
        delayed_parser,
    )
    memo = _ScientificIdentityMemo()
    arguments = {
        "repository_root": repository,
        "roots": ("identity_chain/module_0.py",),
        "identity_memo": memo,
    }
    started = time.perf_counter()
    first = _transitive_local_source_inventory(**arguments)
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warm = _transitive_local_source_inventory(**arguments)
    warm_seconds = time.perf_counter() - started

    assert warm == first
    assert warm_seconds < cold_seconds * 0.5
    assert cold_seconds - warm_seconds >= 0.1
    assert all(hash_counts[path] == 1 for path in source_paths)
    assert all(parse_counts[path] == 1 for path in source_paths)

    monkeypatch.setattr(module, "stable_file_sha256", real_hash)
    uncached = _transitive_local_source_inventory(
        repository_root=repository,
        roots=("identity_chain/module_0.py",),
    )
    assert uncached == first
    monkeypatch.setattr(module, "stable_file_sha256", delayed_hash)

    changed_source = source_paths[6]
    changed_source.write_text(
        "from . import module_7\nVALUE = 99\n",
        encoding="utf-8",
    )
    changed = _transitive_local_source_inventory(**arguments)
    assert changed != first
    assert hash_counts[changed_source] == 2
    assert parse_counts[changed_source] == 2

    original_mode = changed_source.stat().st_mode & 0o777
    changed_mode = 0o600 if original_mode != 0o600 else 0o640
    os.chmod(changed_source, changed_mode)
    stat_only_changed = _transitive_local_source_inventory(**arguments)
    assert stat_only_changed == changed
    assert hash_counts[changed_source] == 3
    assert parse_counts[changed_source] == 3

    alternate_calls: set[Path] = set()

    def alternate_hash(path: Path) -> tuple[str, int]:
        resolved = Path(path).resolve(strict=True)
        alternate_calls.add(resolved)
        digest, size = real_hash(resolved)
        if resolved == changed_source:
            digest = _digest("monkeypatched-scientific-hasher")
        return digest, size

    monkeypatch.setattr(module, "stable_file_sha256", alternate_hash)
    alternate = _transitive_local_source_inventory(**arguments)
    assert alternate != stat_only_changed
    assert changed_source in alternate_calls


def test_negative_import_resolution_invalidates_on_directory_change(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    package = repository / "negative_resolution"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    root = package / "root.py"
    root.write_text(
        "from . import optional_dependency\n",
        encoding="utf-8",
    )
    memo = _ScientificIdentityMemo()
    arguments = {
        "repository_root": repository,
        "roots": ("negative_resolution/root.py",),
        "identity_memo": memo,
    }
    before = _transitive_local_source_inventory(**arguments)
    assert "negative_resolution/optional_dependency.py" not in {
        row["relative_path"] for row in before
    }

    (package / "optional_dependency.py").write_text(
        "VALUE = 'now-present'\n",
        encoding="utf-8",
    )
    after = _transitive_local_source_inventory(**arguments)
    assert "negative_resolution/optional_dependency.py" in {
        row["relative_path"] for row in after
    }
    assert after != before


@pytest.mark.parametrize(
    "name",
    (
        "immutable_run_request.json",
        "complete_manifest.json",
        "stage1_modeling.json",
    ),
)
def test_immutable_control_json_reader_rejects_hardlinks(
    tmp_path: Path,
    name: str,
) -> None:
    control = tmp_path / name
    _write_json(control, {"status": "complete"})
    os.link(control, tmp_path / f"{name}.alias")
    with pytest.raises(ValueError, match="private|hard link"):
        _read_json_object(control, label=name)


@pytest.mark.parametrize(
    "payload",
    (
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": 1, "value": 2}',
    ),
)
def test_immutable_control_json_reader_rejects_nonfinite_and_duplicate_values(
    tmp_path: Path,
    payload: str,
) -> None:
    control = tmp_path / "immutable_run_request.json"
    control.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        _read_json_object(control, label="immutable request")


def test_phase_producer_identities_isolate_stage2_and_bind_shared_runtime_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oci.inference import production_all_evidence_workflow as module

    hooks = {
        "embedding_cache": None,
        "stage1_preflight": None,
        "stage1_modeling": None,
        "role_neutral_stage1": None,
    }
    overrides = {
        phase: None
        for phase in module.PORTABLE_CHECKPOINT_PHASE_SPECS
    }
    baseline = _phase_transitive_producer_code_records(
        workflow_type=ProductionAllEvidenceWorkflow,
        integration_hooks=hooks,
        phase_overrides=overrides,
    )
    stage2_changed = _phase_transitive_producer_code_records(
        workflow_type=_Stage2OnlyProducerMutation,
        integration_hooks=hooks,
        phase_overrides=overrides,
    )
    unaffected = {
        "input_preparation",
        "embedding_cache",
        "stage1_preflight",
        "stage1_modeling",
        "oracle_evaluation",
    }
    assert all(
        baseline[phase]["content_sha256"]
        == stage2_changed[phase]["content_sha256"]
        for phase in unaffected
    )
    assert all(
        baseline[phase]["content_sha256"]
        != stage2_changed[phase]["content_sha256"]
        for phase in ("stage2_canary", "stage2_inference")
    )
    configuration = {
        "schema_version": "test_scientific_configuration_v1",
        "question": "test",
    }
    baseline_binding = _bind_workflow_scientific_identity(
        scientific_configuration_body=configuration,
        phase_code_records=baseline,
    )
    stage2_binding = _bind_workflow_scientific_identity(
        scientific_configuration_body=configuration,
        phase_code_records=stage2_changed,
    )
    assert (
        baseline_binding["scientific_identity"]["scientific_sha256"]
        != stage2_binding["scientific_identity"]["scientific_sha256"]
    )
    base_compatibility = {
        "dataset_identity": _digest("dataset"),
        "split_identity": _digest("splits"),
        "row_order_identity": _digest("rows"),
        "model_identities": {"model": _digest("model")},
        "prompt_identities": {"prompt": _digest("prompt")},
        "configuration_identity": baseline_binding[
            "scientific_configuration_identity"
        ]["scientific_configuration_sha256"],
        "seed_identity": _digest("seed"),
        "runtime_compatibility_class": "test-runtime",
    }
    baseline_compatibilities = {
        phase: ArtifactCompatibility(
            **base_compatibility,
            producer_code_identity=baseline_binding[
                "phase_producer_code_identities"
            ][phase],
        )
        for phase in baseline
    }
    changed_compatibilities = {
        phase: ArtifactCompatibility(
            **base_compatibility,
            producer_code_identity=stage2_binding[
                "phase_producer_code_identities"
            ][phase],
        )
        for phase in stage2_changed
    }
    assert (
        baseline_compatibilities["input_preparation"].key
        == changed_compatibilities["input_preparation"].key
    )
    assert (
        baseline_compatibilities["embedding_cache"].key
        == changed_compatibilities["embedding_cache"].key
    )
    assert (
        baseline_compatibilities["stage2_inference"].key
        != changed_compatibilities["stage2_inference"].key
    )

    real_hash = module.stable_file_sha256

    def changed_stage2_module_hash(path: Path) -> tuple[str, int]:
        digest, size = real_hash(path)
        if Path(path).name == "all_evidence_post_extraction_review.py":
            return _digest("changed-stage2-review-module"), size
        return digest, size

    monkeypatch.setattr(
        module,
        "stable_file_sha256",
        changed_stage2_module_hash,
    )
    imported_stage2_changed = _phase_transitive_producer_code_records(
        workflow_type=ProductionAllEvidenceWorkflow,
        integration_hooks=hooks,
        phase_overrides=overrides,
    )
    # The review module is a real imported dependency of Stage 1 producer
    # helpers and of the remote canary, so those compatibility identities must
    # change too. Preparation/cache/oracle do not import it and stay reusable.
    assert all(
        imported_stage2_changed[phase]["content_sha256"]
        != baseline[phase]["content_sha256"]
        for phase in (
            "stage1_preflight",
            "stage1_modeling",
            "stage2_canary",
            "stage2_inference",
        )
    )
    assert all(
        imported_stage2_changed[phase]["content_sha256"]
        == baseline[phase]["content_sha256"]
        for phase in (
            "input_preparation",
            "embedding_cache",
            "oracle_evaluation",
        )
    )

    def changed_lock_hash(path: Path) -> tuple[str, int]:
        digest, size = real_hash(path)
        if Path(path).name == "uv.lock":
            return _digest("changed-uv-lock"), size
        return digest, size

    monkeypatch.setattr(
        module,
        "stable_file_sha256",
        changed_lock_hash,
    )
    lock_changed = _phase_transitive_producer_code_records(
        workflow_type=ProductionAllEvidenceWorkflow,
        integration_hooks=hooks,
        phase_overrides=overrides,
    )
    assert all(
        baseline[phase]["content_sha256"]
        != lock_changed[phase]["content_sha256"]
        for phase in baseline
    )

    def changed_shared_hash(path: Path) -> tuple[str, int]:
        digest, size = real_hash(path)
        if Path(path).name == "portable_artifacts.py":
            return _digest("changed-portable-artifacts"), size
        return digest, size

    monkeypatch.setattr(
        module,
        "stable_file_sha256",
        changed_shared_hash,
    )
    shared_changed = _phase_transitive_producer_code_records(
        workflow_type=ProductionAllEvidenceWorkflow,
        integration_hooks=hooks,
        phase_overrides=overrides,
    )
    assert all(
        baseline[phase]["content_sha256"]
        != shared_changed[phase]["content_sha256"]
        for phase in baseline
    )

    monkeypatch.setattr(module, "stable_file_sha256", real_hash)
    original_phase_specs = module.PORTABLE_CHECKPOINT_PHASE_SPECS
    changed_specs = {
        phase: dict(spec)
        for phase, spec in module.PORTABLE_CHECKPOINT_PHASE_SPECS.items()
    }
    changed_specs["oracle_evaluation"] = {
        **changed_specs["oracle_evaluation"],
        "artifact_schema": "mutated_oracle_schema_for_test",
    }
    monkeypatch.setattr(
        module,
        "PORTABLE_CHECKPOINT_PHASE_SPECS",
        changed_specs,
    )
    constant_changed = _phase_transitive_producer_code_records(
        workflow_type=ProductionAllEvidenceWorkflow,
        integration_hooks=hooks,
        phase_overrides=overrides,
    )
    assert (
        baseline["oracle_evaluation"]["content_sha256"]
        != constant_changed["oracle_evaluation"]["content_sha256"]
    )
    assert (
        baseline["input_preparation"]["content_sha256"]
        == constant_changed["input_preparation"]["content_sha256"]
    )
    monkeypatch.setattr(
        module,
        "PORTABLE_CHECKPOINT_PHASE_SPECS",
        original_phase_specs,
    )

    publisher_changed = _phase_transitive_producer_code_records(
        workflow_type=_Stage2GranularPublisherMutation,
        integration_hooks=hooks,
        phase_overrides=overrides,
    )
    assert (
        publisher_changed["stage2_inference"]["content_sha256"]
        != baseline["stage2_inference"]["content_sha256"]
    )
    assert all(
        publisher_changed[phase]["content_sha256"]
        == baseline[phase]["content_sha256"]
        for phase in (
            "input_preparation",
            "embedding_cache",
            "stage1_preflight",
            "stage1_modeling",
            "stage2_canary",
            "oracle_evaluation",
        )
    )

    monkeypatch.setattr(
        module,
        "_validate_granular_checkpoint_index_from_paths",
        _mutated_granular_validator,
    )
    validator_changed = _phase_transitive_producer_code_records(
        workflow_type=ProductionAllEvidenceWorkflow,
        integration_hooks=hooks,
        phase_overrides=overrides,
    )
    for phase in (
        "stage1_preflight",
        "stage1_modeling",
        "stage2_inference",
    ):
        assert (
            validator_changed[phase]["content_sha256"]
            != baseline[phase]["content_sha256"]
        )
    for phase in (
        "input_preparation",
        "embedding_cache",
        "stage2_canary",
        "oracle_evaluation",
    ):
        assert (
            validator_changed[phase]["content_sha256"]
            == baseline[phase]["content_sha256"]
        )
    monkeypatch.setattr(
        module,
        "_validate_granular_checkpoint_index_from_paths",
        _validate_granular_checkpoint_index_from_paths,
    )
    monkeypatch.setattr(
        module,
        "_bind_workflow_scientific_identity",
        _mutated_request_compatibility_helper,
    )
    request_compatibility_changed = (
        _phase_transitive_producer_code_records(
            workflow_type=ProductionAllEvidenceWorkflow,
            integration_hooks=hooks,
            phase_overrides=overrides,
        )
    )
    assert all(
        request_compatibility_changed[phase]["content_sha256"]
        != baseline[phase]["content_sha256"]
        for phase in baseline
    )


@pytest.mark.parametrize(
    "first,second",
    (
        (
            default_state_hook_factory("first"),
            default_state_hook_factory("second"),
        ),
        (
            keyword_default_state_hook_factory("first"),
            keyword_default_state_hook_factory("second"),
        ),
        (
            closure_state_hook_factory("first"),
            closure_state_hook_factory("second"),
        ),
        (
            functools.partial(
                partial_state_hook,
                selected="first",
            ),
            functools.partial(
                partial_state_hook,
                selected="second",
            ),
        ),
        (
            CallableStateHook("first"),
            CallableStateHook("second"),
        ),
    ),
)
def test_injected_hook_identity_binds_all_closed_callable_state(
    first,
    second,
) -> None:
    memo = _ScientificIdentityMemo()
    first_identity = _hook_identity(first, identity_memo=memo)
    second_identity = _hook_identity(second, identity_memo=memo)
    assert first_identity is not None
    assert second_identity is not None
    assert first_identity["content_sha256"] != second_identity[
        "content_sha256"
    ]


def test_phase_identity_recomputes_for_closed_override_state() -> None:
    from oci.inference import production_all_evidence_workflow as module

    memo = _ScientificIdentityMemo()
    first_override = _hook_identity(
        closure_state_hook_factory("first"),
        identity_memo=memo,
    )
    second_override = _hook_identity(
        closure_state_hook_factory("second"),
        identity_memo=memo,
    )
    overrides = {
        phase: None
        for phase in module.PORTABLE_CHECKPOINT_PHASE_SPECS
    }
    first_records = _phase_transitive_producer_code_records(
        workflow_type=ProductionAllEvidenceWorkflow,
        integration_hooks={
            "embedding_cache": None,
            "stage1_preflight": None,
            "stage1_modeling": None,
            "role_neutral_stage1": None,
        },
        phase_overrides={
            **overrides,
            "stage2_canary": first_override,
        },
        identity_memo=memo,
    )
    second_records = _phase_transitive_producer_code_records(
        workflow_type=ProductionAllEvidenceWorkflow,
        integration_hooks={
            "embedding_cache": None,
            "stage1_preflight": None,
            "stage1_modeling": None,
            "role_neutral_stage1": None,
        },
        phase_overrides={
            **overrides,
            "stage2_canary": second_override,
        },
        identity_memo=memo,
    )
    assert (
        first_records["stage2_canary"]["content_sha256"]
        != second_records["stage2_canary"]["content_sha256"]
    )
    assert all(
        first_records[phase]["content_sha256"]
        == second_records[phase]["content_sha256"]
        for phase in first_records
        if phase != "stage2_canary"
    )


def test_injected_hook_identity_rejects_unclosed_callable_state() -> None:
    with pytest.raises(
        TypeError,
        match="unclosed state|explicit closed scientific identity",
    ):
        _hook_identity(UnclosedCallableStateHook())


def test_injected_hook_identity_rejects_external_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = tmp_path / "external_hook_package"
    external.mkdir()
    (external / "__init__.py").write_text("", encoding="utf-8")
    (external / "hook.py").write_text(
        "def run_hook(_attempt):\n"
        "    return {'terminal_files': []}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    imported = importlib.import_module("external_hook_package.hook")
    try:
        with pytest.raises(
            ValueError,
            match="inside the authenticated repository|external callable",
        ):
            _hook_identity(imported.run_hook)
    finally:
        sys.modules.pop("external_hook_package.hook", None)
        sys.modules.pop("external_hook_package", None)


def test_injected_hook_identity_binds_repository_local_import_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    package = repository / "hook_package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    dependency = package / "dependency.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    (package / "hook.py").write_text(
        "from . import dependency\n"
        "def run_hook(*args, **kwargs):\n"
        "    return dependency.VALUE\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(repository))
    imported = importlib.import_module("hook_package.hook")
    try:
        baseline_closure = (
            _repository_local_callable_import_closure(
                imported.run_hook,
                repository_root=repository,
            )
        )
        dependency.write_text("VALUE = 2\n", encoding="utf-8")
        changed_closure = (
            _repository_local_callable_import_closure(
                imported.run_hook,
                repository_root=repository,
            )
        )
    finally:
        sys.modules.pop("hook_package.hook", None)
        sys.modules.pop("hook_package.dependency", None)
        sys.modules.pop("hook_package", None)
    assert baseline_closure != changed_closure

    from oci.inference import production_all_evidence_workflow as module

    overrides = {
        phase: None
        for phase in module.PORTABLE_CHECKPOINT_PHASE_SPECS
    }
    base_hooks = {
        "embedding_cache": {
            "module": "hook_package.hook",
            "qualname": "run_hook",
            "repository_local_import_closure": list(
                baseline_closure
            ),
        },
        "stage1_preflight": None,
        "stage1_modeling": None,
        "role_neutral_stage1": None,
    }
    changed_hooks = {
        **base_hooks,
        "embedding_cache": {
            **base_hooks["embedding_cache"],
            "repository_local_import_closure": list(
                changed_closure
            ),
        },
    }
    baseline = _phase_transitive_producer_code_records(
        workflow_type=ProductionAllEvidenceWorkflow,
        integration_hooks=base_hooks,
        phase_overrides=overrides,
    )
    changed = _phase_transitive_producer_code_records(
        workflow_type=ProductionAllEvidenceWorkflow,
        integration_hooks=changed_hooks,
        phase_overrides=overrides,
    )
    assert (
        baseline["embedding_cache"]["content_sha256"]
        != changed["embedding_cache"]["content_sha256"]
    )
    assert all(
        baseline[phase]["content_sha256"]
        == changed[phase]["content_sha256"]
        for phase in baseline
        if phase != "embedding_cache"
    )
