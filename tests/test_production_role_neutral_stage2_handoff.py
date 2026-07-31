from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.resource_safety_test_support import resource_safety_policy

import oci.inference.production_role_neutral_stage2_handoff as handoff_module
from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    BOW_NUISANCE,
    BOW_R_LOSS,
    HTR_NEURAL,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    build_role_neutral_evidence_catalog,
)
from oci.inference.htr_native_proof_capture import _array_sha256
from oci.inference.portable_workflow_spec import EVIDENCE_FAMILIES
from oci.inference.portable_workflow_spec import (
    ResourcePerformanceSafetyPolicy,
)
from oci.inference.portable_resource_scheduler import (
    ResourceInventory,
    ResourcePlan,
)
from oci.inference.production_role_neutral_stage2_handoff import (
    AuthenticatedPreparedCohortProjectionBinding,
    AuthenticatedRoleNeutralStage2RuntimeBinding,
    AuthenticatedRoleNeutralStage2Provider,
    FailClosedRoleNeutralStage2HandoffPublisher,
    ReferenceOnlyRoleNeutralStage1HandoffPublisher,
    ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
    ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_DIRECTORY,
    ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_MANIFEST,
    ROLE_NEUTRAL_STAGE1_REFERENCE_LOCATOR,
    ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST,
    ROLE_NEUTRAL_STAGE1_REFERENCE_PLAN,
    ROLE_NEUTRAL_STAGE1_REFERENCE_REGISTRY,
    ROLE_NEUTRAL_STAGE1_REFERENCE_ROW_MAP,
    ROLE_NEUTRAL_STAGE2_COMPONENT_EXPORT_INDEX_SCHEMA,
    ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_PROOF_SCHEMA,
    ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD,
    RoleNeutralStage2LoaderContractUnavailable,
    RoleNeutralStage2ProjectionProofUnavailable,
    build_role_neutral_stage2_fit_projection_proof,
    load_reference_only_role_neutral_stage1_handoff,
    validate_authenticated_prepared_projection_binding,
    validate_authenticated_role_neutral_stage2_runtime_binding,
    validate_role_neutral_stage2_bridge,
)
from oci.inference.production_stage1_legacy_scope_fragments import (
    ROLE_NEUTRAL_FIT_ONLY_FAMILY_PRIOR_AUTH_REFERENCE_SCHEMA,
    build_role_neutral_fit_only_family_seal,
)
from oci.inference.production_stage1_role_neutral_execution import (
    BoundRoleNeutralComponentProducer,
    ROLE_NEUTRAL_COMPONENT_DIRECTORY,
    RoleNeutralStage1ExecutionPolicy,
    execute_and_publish_role_neutral_stage1,
)
from oci.inference.production_stage1_scope_scheduler import (
    build_canonical_stage1_scope_plan,
)
from tests.stage1_test_support import PHYSICAL_FIT_IDENTITY
from oci.inference.role_neutral_all_ten_binding import (
    AuthenticatedRoleNeutralComponentReceipt,
    EXPECTED_COMPONENT_FAMILIES,
)
from tests.test_lossless_stage1_evidence_catalog import (
    _cumulative_family_payloads,
    _inputs,
)
from tests.test_native_role_neutral_payload_catalog_adapter import (
    _native_payloads,
)
from tests.test_production_stage1_role_neutral_execution import (
    _ProducerRecorder,
    _RecordingExecutor,
    _plan,
    _registry,
    _sha,
)


def _cpu_resource_plan() -> ResourcePlan:
    return ResourcePlan(
        devices=("cpu",),
        cpu_budget=4,
        inventory=ResourceInventory(cpu_count=32, gpus=()),
        policy=("cpu",),
        resource_performance_safety=resource_safety_policy(
            gpu_max_allocation_fraction=0.85,
            gpu_minimum_headroom_bytes=6 * 1024**3,
            minimum_multi_device_throughput_ratio=1.5,
            maximum_coordination_proof_overhead_ratio=0.3,
            maximum_ordinary_read_amplification=2.0,
            minimum_benchmark_repetitions_per_scope=2,
            read_counter_source="logical_read_bytes",
            fail_on_external_gpu_occupants=True,
        ),
    )


def _execution(tmp_path: Path):
    plan = _plan(gpu_ids=())
    root = (tmp_path / "role_neutral_execution").resolve()
    manifest = execute_and_publish_role_neutral_stage1(
        root=root,
        plan=plan,
        producer_factories=_ProducerRecorder().factories(),
        policy=RoleNeutralStage1ExecutionPolicy(
            resource_plan=_cpu_resource_plan(),
            max_parallel_owners=2,
        ),
        executor=_RecordingExecutor(),
    )
    return plan, root, manifest


def _fit_text(row_id: int) -> str:
    return f"complete prepared clinical note for row {row_id}"


def _fit_treatment(row_id: int) -> float:
    return float(row_id % 2)


def _fit_outcome(row_id: int) -> float:
    return float((row_id // 2) % 2)


class _ProviderReadyProducerRecorder(_ProducerRecorder):
    def __init__(self, family_payloads):
        super().__init__()
        self.family_payloads = copy.deepcopy(family_payloads)
        self.htr_arrays: dict[str, np.ndarray] = {}
        package = self.family_payloads[HTR_NEURAL][
            "token_attention_evidence"
        ]
        for batch in package["fold_batches"]:
            stage = str(batch["stage"])
            prefix = f"provider_{stage}_0001"
            decoded = "[CLS]configured[SEP]".encode("utf-8")
            values = {
                "fit_note_position": np.asarray([0, 0, 0], dtype=np.int64),
                "fit_row_id": np.asarray([10, 10, 10], dtype=np.int64),
                "chunk_index": np.asarray([0, 0, 0], dtype=np.int32),
                "token_position": np.asarray([0, 1, 2], dtype=np.int32),
                "token_id": np.asarray([101, 2001, 102], dtype=np.int32),
                "decoded_token_text_utf8": np.frombuffer(
                    decoded, dtype=np.uint8
                ).copy(),
                "decoded_token_text_byte_offsets": np.asarray(
                    [0, 5, 15, 20], dtype=np.int64
                ),
                "char_start": np.asarray([0, 0, 0], dtype=np.int32),
                "char_end": np.asarray([0, 10, 0], dtype=np.int32),
                "is_special_token": np.asarray(
                    [1, 0, 1], dtype=np.uint8
                ),
                "is_padding": np.asarray([0, 0, 0], dtype=np.uint8),
                "token_attention": np.asarray(
                    [0.1, 0.8, 0.1], dtype=np.float64
                ),
                "chunk_attention": np.asarray(
                    [1.0, 1.0, 1.0], dtype=np.float64
                ),
                "hierarchical_attention_score": np.asarray(
                    [0.1, 0.8, 0.1], dtype=np.float64
                ),
            }
            columns: dict[str, dict] = {}
            for name, value in values.items():
                array_name = f"{prefix}_{name}"
                self.htr_arrays[array_name] = value
                columns[name] = {
                    "array": array_name,
                    "content_sha256": _array_sha256(value),
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                }
            batch.update(
                {
                    "raw_occurrence_order": (
                        "fit_note_position_then_chunk_index_then_"
                        "token_position_v1"
                    ),
                    "decoded_token_text_encoding": (
                        "concatenated_utf8_with_offsets_v1"
                    ),
                    "tokenizer_identity": {
                        "model_name": "prajjwal1/bert-tiny",
                        "vocabulary_sha256": "d" * 64,
                    },
                    "fit_note_positions": [0],
                    "fit_row_ids": [10],
                    "columns": columns,
                }
            )
            batch_body = {
                key: value
                for key, value in batch.items()
                if key != "content_sha256"
            }
            batch["content_sha256"] = _sha(batch_body)
        package_body = {
            key: value
            for key, value in package.items()
            if key != "content_sha256"
        }
        package["content_sha256"] = _sha(package_body)

    def factory(self, expected_component: str):
        base_factory = super().factory(expected_component)

        def bind(invocation):
            base = base_factory(invocation)

            def family_seals():
                owner_id = invocation.physical_owner.scope_id
                return {
                    family: build_role_neutral_fit_only_family_seal(
                        plan=invocation.plan,
                        physical_owner_scope_id=owner_id,
                        family=family,
                        evidence_payload=self.family_payloads[family],
                        producer_identity_sha256=_sha(
                            {
                                "component": expected_component,
                                "family": family,
                                "producer": "provider-ready-test",
                            }
                        ),
                        configuration_identity_sha256=_sha(
                            {
                                "component": expected_component,
                                "family": family,
                                "configuration": "provider-ready-test",
                            }
                        ),
                        fit_state_artifact_sha256=_sha(
                            {
                                "owner": owner_id,
                                "family": family,
                                "fit_state": "provider-ready-test",
                            }
                        ),
                    )
                    for family in EXPECTED_COMPONENT_FAMILIES[
                        expected_component
                    ]
                }

            def execute():
                base.execute()
                if expected_component == "htr":
                    htr_seal = family_seals()[HTR_NEURAL]
                    array_root = (
                        invocation.output_root / "fit_state" / "arrays"
                    )
                    array_root.mkdir(parents=True)
                    for name, value in self.htr_arrays.items():
                        np.save(
                            array_root / f"{name}.npy",
                            value,
                            allow_pickle=False,
                        )
                    (
                        invocation.output_root
                        / "fit_only_family_seal.json"
                    ).write_text(
                        json.dumps(
                            htr_seal,
                            indent=2,
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                if expected_component != "bow":
                    return
                path = invocation.output_root / "execution_manifest.json"
                terminal = json.loads(path.read_text(encoding="utf-8"))
                terminal_body = {
                    key: value for key, value in terminal.items() if key != "content_sha256"
                }
                rows = invocation.physical_owner.fit_row_ids
                terminal_body[ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD] = (
                    build_role_neutral_stage2_fit_projection_proof(
                        plan_scientific_content_sha256=(invocation.plan.scientific_content_sha256),
                        physical_owner_scope_id=(invocation.physical_owner.scope_id),
                        fit_row_ids=rows,
                        fit_texts=tuple(_fit_text(row_id) for row_id in rows),
                        fit_treatment=tuple(_fit_treatment(row_id) for row_id in rows),
                        fit_outcome=tuple(_fit_outcome(row_id) for row_id in rows),
                    )
                )
                changed = {
                    **terminal_body,
                    "content_sha256": _sha(terminal_body),
                }
                path.write_text(
                    json.dumps(
                        changed,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            def authenticate():
                base_receipt = base.authenticate()
                owner_id = invocation.physical_owner.scope_id
                seals = family_seals()
                return AuthenticatedRoleNeutralComponentReceipt.create(
                    plan=invocation.plan,
                    physical_owner_scope_id=owner_id,
                    component=expected_component,
                    family_fit_seals=seals,
                    family_logical_view_content_sha256=(
                        base_receipt.family_logical_view_content_sha256
                    ),
                    source_terminal_content_sha256=(base_receipt.source_terminal_content_sha256),
                    source_tree_sha256=(base_receipt.source_tree_sha256),
                )

            return BoundRoleNeutralComponentProducer(
                execute=execute,
                authenticate=authenticate,
            )

        return bind


def _provider_ready_execution(
    tmp_path: Path,
    *,
    family_payloads=None,
):
    if family_payloads is None:
        family_payloads = _native_payloads()
    assert set(family_payloads) == set(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    registry = _registry()
    plan = build_canonical_stage1_scope_plan(
        registry=registry,
        registry_content_sha256=_sha(registry),
        global_seed=42,
        physical_fit_identity=PHYSICAL_FIT_IDENTITY,
        gpu_ids=(),
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=len(registry["outer_folds"]),
        expected_inner_fold_count=len(registry["outer_folds"][0]["inner_folds"]),
    )
    root = (tmp_path / "provider_ready_execution").resolve()
    manifest = execute_and_publish_role_neutral_stage1(
        root=root,
        plan=plan,
        producer_factories=_ProviderReadyProducerRecorder(family_payloads).factories(),
        policy=RoleNeutralStage1ExecutionPolicy(
            resource_plan=_cpu_resource_plan(),
            max_parallel_owners=2,
        ),
        executor=_RecordingExecutor(),
    )
    bridge = validate_role_neutral_stage2_bridge(
        execution_root=root,
        plan=plan,
        execution_manifest=manifest,
    )
    aggregation_store = handoff_module._build_htr_semantic_aggregation_store(
        root=(tmp_path / "htr_semantic_aggregation_store").resolve(),
        execution_root=root,
        execution_content_sha256=manifest["content_sha256"],
        plan=plan,
        bridge=bridge,
    )
    provider = AuthenticatedRoleNeutralStage2Provider(
        execution_root=root,
        plan=plan,
        execution_manifest=manifest,
        semantic_member_batch_size=3,
        htr_aggregation_store_root=aggregation_store.root,
    )
    return plan, root, manifest, provider


def test_compact_fit_seal_reference_reopens_complete_payload_on_demand(
    tmp_path,
):
    plan = _plan(gpu_ids=())
    owner = plan.physical_scopes[0]
    component_root = (tmp_path / "bow").resolve()
    component_root.mkdir()
    payloads = _native_payloads()
    seals = {}
    registrations = {}
    for index, family in enumerate((BOW_NUISANCE, BOW_R_LOSS)):
        seal = build_role_neutral_fit_only_family_seal(
            plan=plan,
            physical_owner_scope_id=owner.scope_id,
            family=family,
            evidence_payload=payloads[family],
            producer_identity_sha256=_sha(
                {"family": family, "identity": "producer"}
            ),
            configuration_identity_sha256=_sha(
                {"family": family, "identity": "configuration"}
            ),
            fit_state_artifact_sha256=_sha(
                {"family": family, "identity": "fit-state"}
            ),
        )
        relative = f"fit_only_{index}.json"
        encoded = (
            json.dumps(
                seal,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        (component_root / relative).write_bytes(encoded)
        seals[family] = seal
        registrations[family] = {
            "relative_path": relative,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
            "content_sha256": seal["content_sha256"],
        }
    scope_ids = tuple(
        scope.scope_id
        for scope in plan.scopes
        if plan.physical_owner(scope.scope_id).scope_id == owner.scope_id
    )
    receipt = AuthenticatedRoleNeutralComponentReceipt.create(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
        component="bow",
        family_fit_seals=seals,
        family_fit_seal_registrations=registrations,
        family_logical_view_content_sha256={
            family: {
                scope_id: _sha(
                    {
                        "family": family,
                        "logical_scope_id": scope_id,
                    }
                )
                for scope_id in scope_ids
            }
            for family in (BOW_NUISANCE, BOW_R_LOSS)
        },
        source_terminal_content_sha256="a" * 64,
        source_tree_sha256="b" * 64,
    )
    reference = receipt.family_fit_seals[BOW_NUISANCE]
    assert "evidence_payload" not in reference
    reopened = handoff_module._resolve_fit_seal_reference(
        owner_scope_id=owner.scope_id,
        family=BOW_NUISANCE,
        seal_or_reference=reference,
        component_roots={
            (owner.scope_id, "bow"): component_root,
        },
    )
    assert reopened["evidence_payload"] == payloads[BOW_NUISANCE]
    assert reopened["content_sha256"] == reference["content_sha256"]

    source_path = (
        component_root
        / reference["source_seal_registration"]["relative_path"]
    )
    source_path.write_bytes(source_path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        handoff_module._resolve_fit_seal_reference(
            owner_scope_id=owner.scope_id,
            family=BOW_NUISANCE,
            seal_or_reference=reference,
            component_roots={
                (owner.scope_id, "bow"): component_root,
            },
        )


def test_prior_authenticated_seal_reference_reopens_complete_payload_on_demand(
    tmp_path,
):
    plan = _plan(gpu_ids=())
    owner = plan.physical_scopes[0]
    component_root = (tmp_path / "bow-prior-auth").resolve()
    component_root.mkdir()
    payload = _native_payloads()[BOW_NUISANCE]
    seal = build_role_neutral_fit_only_family_seal(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
        family=BOW_NUISANCE,
        evidence_payload=payload,
        producer_identity_sha256="a" * 64,
        configuration_identity_sha256="b" * 64,
        fit_state_artifact_sha256="c" * 64,
    )
    encoded = (
        json.dumps(
            seal,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    relative = "fit_only_prior.json"
    (component_root / relative).write_bytes(encoded)
    registration = {
        "relative_path": relative,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        "content_sha256": seal["content_sha256"],
    }
    body = {
        "schema_version": (
            ROLE_NEUTRAL_FIT_ONLY_FAMILY_PRIOR_AUTH_REFERENCE_SCHEMA
        ),
        "plan_scientific_content_sha256": (
            plan.scientific_content_sha256
        ),
        "physical_owner_scope_id": owner.scope_id,
        "family": BOW_NUISANCE,
        "content_sha256": seal["content_sha256"],
        "source_seal_registration": registration,
        "source_evidence_projection": "identity_evidence_payload_v1",
        "prior_component_import_attestation_content_sha256": "d" * 64,
        "source_terminal_content_sha256": "e" * 64,
        "source_tree_sha256": "f" * 64,
        "complete_evidence_payload_retained_by_reference": True,
        "evidence_payload_in_receipt": False,
        "hierarchical_raw_sidecars_retained": True,
    }
    reference = {
        **body,
        "reference_content_sha256": _sha(body),
    }
    reopened = handoff_module._resolve_fit_seal_reference(
        owner_scope_id=owner.scope_id,
        family=BOW_NUISANCE,
        seal_or_reference=reference,
        component_roots={
            (owner.scope_id, "bow"): component_root,
        },
    )
    assert reopened["evidence_payload"] == payload
    assert reopened["content_sha256"] == reference["content_sha256"]

    changed = copy.deepcopy(reference)
    changed["content_sha256"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="authenticated source identity",
    ):
        handoff_module._resolve_fit_seal_reference(
            owner_scope_id=owner.scope_id,
            family=BOW_NUISANCE,
            seal_or_reference=changed,
            component_roots={
                (owner.scope_id, "bow"): component_root,
            },
        )


@pytest.fixture(scope="module")
def authenticated_execution(tmp_path_factory):
    return _execution(tmp_path_factory.mktemp("role_neutral_stage2_bridge"))


@pytest.fixture(scope="module")
def provider_ready_execution(tmp_path_factory):
    return _provider_ready_execution(tmp_path_factory.mktemp("role_neutral_stage2_provider"))


@pytest.fixture(scope="module")
def reference_handoff(provider_ready_execution):
    plan, root, manifest, _provider = provider_ready_execution
    registry = _registry()
    prepared = SimpleNamespace(
        stage1_scope_plan=plan,
        registry=registry,
        registry_content_sha256=_sha(registry),
        request_sha256="b" * 64,
        data=pd.DataFrame(
            {
                "configured_patient_key": [
                    f"patient-{row_id}" for row_id in range(registry["dataset_row_count"])
                ]
            }
        ),
        options=SimpleNamespace(
            unit_id_column="configured_patient_key",
        ),
    )
    target = (root.parent / "reference_only_handoff").resolve()
    execution_before = {
        path.relative_to(root).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat(follow_symlinks=False).st_size,
        )
        for path in root.rglob("*")
        if path.is_file()
    }
    publication = ReferenceOnlyRoleNeutralStage1HandoffPublisher(
        semantic_member_batch_size=3,
    )(
        target_dir=target,
        prepared=prepared,
        role_neutral_execution_root=root,
        role_neutral_execution_manifest=manifest,
    )
    execution_after = {
        path.relative_to(root).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat(follow_symlinks=False).st_size,
        )
        for path in root.rglob("*")
        if path.is_file()
    }
    assert execution_after == execution_before
    return plan, root, manifest, target, publication


def test_benchmark_scope_plan_has_40_logical_and_35_physical_fits() -> None:
    plan = _plan(gpu_ids=())

    assert len(plan.scopes) == 40
    assert len(plan.physical_scopes) == 35
    assert len(plan.scopes) - len(plan.physical_scopes) == 5


def test_valid_execution_yields_path_neutral_typed_stage2_bridge(
    authenticated_execution,
) -> None:
    plan, root, manifest = authenticated_execution

    bridge = validate_role_neutral_stage2_bridge(
        execution_root=root,
        plan=plan,
        execution_manifest=manifest,
    )

    identity = bridge.scientific_identity()
    encoded_identity = json.dumps(identity, sort_keys=True)
    assert len(bridge.physical_fits) == len(plan.physical_scopes)
    assert len(bridge.logical_contexts) == len(plan.scopes)
    assert identity["deduplicated_fit_count"] == (len(plan.scopes) - len(plan.physical_scopes))
    assert identity["portable_family_order"] == list(EVIDENCE_FAMILIES)
    assert identity["whole_cohort_and_cluster_local_embeddings_independent"] is True
    assert identity["text_truncation_applied"] is False
    assert identity["lossy_evidence_selection_applied"] is False
    assert identity["evidence_payloads_copied"] is False
    assert identity["evidence_payloads_recomputed"] is False
    assert identity["legacy_bundle_build_invoked"] is False
    assert all(
        len(context.family_logical_view_content_sha256) == len(EVIDENCE_FAMILIES)
        for context in bridge.logical_contexts
    )
    assert all(
        len(context["family_logical_view_content_sha256"]) == len(EVIDENCE_FAMILIES)
        for context in identity["logical_contexts"]
    )
    first_family_views = {
        portable: digest
        for portable, _native, digest in (
            bridge.logical_contexts[0].family_logical_view_content_sha256
        )
    }
    assert "whole_cohort_embeddings" in first_family_views
    assert "cluster_local_embeddings" in first_family_views
    assert str(root) not in encoded_identity
    assert bridge.as_dict()["source_execution_attestation"]["root_locator"] == str(root)

    stale = dict(manifest)
    stale["content_sha256"] = "f" * 64
    with pytest.raises(
        ValueError,
        match="differs from fresh path-only validation",
    ):
        validate_role_neutral_stage2_bridge(
            execution_root=root,
            plan=plan,
            execution_manifest=stale,
        )


def test_bridge_reopens_component_bytes_and_rejects_tamper(
    authenticated_execution,
) -> None:
    plan, root, manifest = authenticated_execution
    owner = plan.physical_scopes[0]
    terminal = (
        root / ROLE_NEUTRAL_COMPONENT_DIRECTORY / owner.scope_id / "htr" / "execution_manifest.json"
    )
    original = terminal.read_bytes()
    try:
        terminal.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="component tree changed"):
            validate_role_neutral_stage2_bridge(
                execution_root=root,
                plan=plan,
                execution_manifest=manifest,
            )
    finally:
        terminal.write_bytes(original)


def test_publisher_fails_closed_before_materializing_legacy_bundle(
    tmp_path: Path,
    authenticated_execution,
) -> None:
    plan, root, manifest = authenticated_execution
    target = (tmp_path / "stage1_bundle").resolve()
    publisher = FailClosedRoleNeutralStage2HandoffPublisher()

    with pytest.raises(
        RoleNeutralStage2LoaderContractUnavailable,
        match="aborted without copying or recomputing evidence",
    ) as caught:
        publisher(
            target_dir=target,
            prepared=SimpleNamespace(stage1_scope_plan=plan),
            role_neutral_execution_root=root,
            role_neutral_execution_manifest=manifest,
        )

    assert not target.exists()
    assert caught.value.bridge.source_execution_content_sha256 == (manifest["content_sha256"])
    requirements = caught.value.requirements.as_dict()
    assert requirements["component_export_index_schema"] == (
        ROLE_NEUTRAL_STAGE2_COMPONENT_EXPORT_INDEX_SCHEMA
    )
    assert "legacy_bundle_build" in requirements["forbidden_compatibility_actions"]
    assert (
        "raw_evidence_copy"
        in requirements["forbidden_compatibility_actions"]
    )
    assert (
        "bind_prepared_request_dataset_split_model_prompt_and_seed_identity"
        in requirements["required_direct_loader_capabilities"]
    )


def test_current_execution_without_projection_export_fails_closed(
    tmp_path: Path,
    authenticated_execution,
) -> None:
    plan, root, manifest = authenticated_execution
    aggregate_placeholder = (tmp_path / "aggregate-placeholder").resolve()
    aggregate_placeholder.mkdir()

    with pytest.raises(
        RoleNeutralStage2ProjectionProofUnavailable,
        match="cannot be inferred",
    ) as caught:
        AuthenticatedRoleNeutralStage2Provider(
            execution_root=root,
            plan=plan,
            execution_manifest=manifest,
            semantic_member_batch_size=3,
            htr_aggregation_store_root=aggregate_placeholder,
        )

    addition = caught.value.required_schema_addition
    assert addition["producer_component"] == "bow"
    assert addition["terminal_field"] == (ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD)
    assert addition["field_schema"] == (ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_PROOF_SCHEMA)
    assert addition["raw_values_persisted"] is False
    assert addition["compatibility_default_allowed"] is False


def test_authenticated_provider_serves_lossless_catalog_without_raw_fallback(
    provider_ready_execution,
) -> None:
    plan, root, _manifest, provider = provider_ready_execution
    identity = provider.identity()
    assert str(root) not in json.dumps(identity, sort_keys=True)
    assert identity["all_ten_architectures_required"] is True
    assert identity["catalogs_assembled_losslessly_from_fit_only_seals"] is True
    assert identity["raw_spent_evidence_input_fallback_available"] is False

    assignments = provider.get_review_partition_assignments(
        outer_fold=1,
        exact_outer_train_row_ids=tuple(
            row_id
            for partition in range(1, 6)
            for row_id in next(
                scope.heldout_row_ids
                for scope in plan.scopes
                if scope.outer_fold == 1
                and scope.scope_kind == "exact_inner"
                and scope.inner_fold == partition
            )
        ),
    )
    assert tuple(assignments) == (1, 2, 3, 4, 5)

    scope = next(
        scope
        for scope in plan.scopes
        if scope.outer_fold == 1
        and scope.scope_kind == "cumulative_spent"
        and scope.context_epoch == 1
    )
    assert plan.physical_owner(scope.scope_id).scope_id != scope.scope_id
    catalog = provider.get_spent_evidence_catalog(
        outer_fold=scope.outer_fold,
        review_round=scope.context_epoch,
        exact_spent_row_ids=scope.fit_row_ids,
        exact_sealed_row_ids=scope.heldout_row_ids,
        spent_texts=tuple(_fit_text(row_id) for row_id in scope.fit_row_ids),
        spent_treatment=np.asarray(
            [_fit_treatment(row_id) for row_id in scope.fit_row_ids],
            dtype=np.float64,
        ),
        spent_outcome=np.asarray(
            [_fit_outcome(row_id) for row_id in scope.fit_row_ids],
            dtype=np.float64,
        ),
    )
    assert catalog.outer_fold == scope.outer_fold
    assert catalog.inner_fold == scope.provider_inner_fold
    assert all(catalog.family_atoms(family) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES)
    assert catalog.audit["family_payload_roundtrip_verified"] is True

    with pytest.raises(RuntimeError, match="raw-input"):
        provider.get_spent_evidence_inputs()


def test_authenticated_provider_accepts_real_shaped_native_all_ten_payloads(
    tmp_path: Path,
) -> None:
    payloads = _native_payloads()
    plan, _root, _manifest, provider = _provider_ready_execution(
        tmp_path,
        family_payloads=payloads,
    )
    scope = next(
        scope
        for scope in plan.scopes
        if scope.outer_fold == 1
        and scope.scope_kind == "cumulative_spent"
        and scope.context_epoch == 0
    )
    catalog = provider.get_spent_evidence_catalog(
        outer_fold=scope.outer_fold,
        review_round=scope.context_epoch,
        exact_spent_row_ids=scope.fit_row_ids,
        exact_sealed_row_ids=scope.heldout_row_ids,
        spent_texts=tuple(_fit_text(row_id) for row_id in scope.fit_row_ids),
        spent_treatment=np.asarray(
            [_fit_treatment(row_id) for row_id in scope.fit_row_ids],
            dtype=np.float64,
        ),
        spent_outcome=np.asarray(
            [_fit_outcome(row_id) for row_id in scope.fit_row_ids],
            dtype=np.float64,
        ),
    )

    assert all(catalog.family_atoms(family) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES)
    assert all(
        catalog.audit["native_payload_adapter_by_family"][family][
            "adapter_applied"
        ]
        is (family != HTR_NEURAL)
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    assert (
        catalog.audit["native_payload_adapter_selection_or_truncation_applied"]
        is False
    )
    token_audit = catalog.audit["native_payload_adapter_by_family"][
        "htr_neural"
    ]["complete_token_attention_evidence"]
    assert token_audit["token_occurrence_count"] == 6
    assert token_audit["special_token_occurrence_count"] == 4
    assert (
        token_audit[
            "raw_sidecars_authenticated_in_zero_copy_source_graph"
        ]
        is True
    )
    assert catalog.audit["family_payload_roundtrip_verified"] is True


def test_authenticated_provider_rejects_runtime_projection_drift(
    provider_ready_execution,
) -> None:
    plan, _root, _manifest, provider = provider_ready_execution
    scope = next(
        scope
        for scope in plan.scopes
        if scope.outer_fold == 1
        and scope.scope_kind == "cumulative_spent"
        and scope.context_epoch == 0
    )
    texts = [_fit_text(row_id) for row_id in scope.fit_row_ids]
    texts[0] += " changed"

    with pytest.raises(
        ValueError,
        match="differs from the sealed role-neutral producer proof",
    ):
        provider.get_spent_evidence_catalog(
            outer_fold=scope.outer_fold,
            review_round=scope.context_epoch,
            exact_spent_row_ids=scope.fit_row_ids,
            exact_sealed_row_ids=scope.heldout_row_ids,
            spent_texts=tuple(texts),
            spent_treatment=np.asarray(
                [_fit_treatment(row_id) for row_id in scope.fit_row_ids],
                dtype=np.float64,
            ),
            spent_outcome=np.asarray(
                [_fit_outcome(row_id) for row_id in scope.fit_row_ids],
                dtype=np.float64,
            ),
        )


def test_provider_authenticates_and_assembles_once_per_trust_boundary(
    monkeypatch,
    provider_ready_execution,
) -> None:
    plan, root, manifest, _existing_provider = provider_ready_execution
    calls = {"validate": 0, "assemble": 0}
    original_validate = handoff_module.validate_role_neutral_stage2_bridge
    original_assemble = handoff_module.assemble_cumulative_spent_role_neutral_catalog

    def counted_validate(**kwargs):
        calls["validate"] += 1
        return original_validate(**kwargs)

    def counted_assemble(**kwargs):
        calls["assemble"] += 1
        return original_assemble(**kwargs)

    monkeypatch.setattr(
        handoff_module,
        "validate_role_neutral_stage2_bridge",
        counted_validate,
    )
    monkeypatch.setattr(
        handoff_module,
        "assemble_cumulative_spent_role_neutral_catalog",
        counted_assemble,
    )
    provider = AuthenticatedRoleNeutralStage2Provider(
        execution_root=root,
        plan=plan,
        execution_manifest=manifest,
        semantic_member_batch_size=3,
        htr_aggregation_store_root=(
            root.parent / "htr_semantic_aggregation_store"
        ),
    )
    cumulative_scopes = tuple(
        scope for scope in plan.scopes if scope.scope_kind == "cumulative_spent"
    )
    expected_calls = {
        "validate": 1,
        "assemble": len(cumulative_scopes),
    }
    assert calls == expected_calls

    provider.identity()
    provider.identity()
    assignments = provider.get_review_partition_assignments(
        outer_fold=1,
        exact_outer_train_row_ids=tuple(
            row_id
            for scope in plan.scopes
            if scope.outer_fold == 1 and scope.scope_kind == "exact_inner"
            for row_id in scope.heldout_row_ids
        ),
    )
    assert assignments
    scope = cumulative_scopes[0]
    provider.get_spent_evidence_catalog(
        outer_fold=scope.outer_fold,
        review_round=scope.context_epoch,
        exact_spent_row_ids=scope.fit_row_ids,
        exact_sealed_row_ids=scope.heldout_row_ids,
        spent_texts=tuple(_fit_text(row_id) for row_id in scope.fit_row_ids),
        spent_treatment=np.asarray(
            [_fit_treatment(row_id) for row_id in scope.fit_row_ids],
            dtype=np.float64,
        ),
        spent_outcome=np.asarray(
            [_fit_outcome(row_id) for row_id in scope.fit_row_ids],
            dtype=np.float64,
        ),
    )
    assert calls == expected_calls


def test_reference_handoff_is_positive_path_neutral_and_zero_copy(
    reference_handoff,
) -> None:
    plan, execution_root, manifest, target, publication = reference_handoff
    assert publication.handoff_kind == (ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND)
    assert publication.stage2_provider is not None
    assert publication.source_role_neutral_execution_content_sha256 == (manifest["content_sha256"])
    assert publication.legacy_bundle_build_invoked is False
    published_files = set(
        path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()
    )
    assert {
        ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST,
        ROLE_NEUTRAL_STAGE1_REFERENCE_LOCATOR,
        ROLE_NEUTRAL_STAGE1_REFERENCE_REGISTRY,
        ROLE_NEUTRAL_STAGE1_REFERENCE_PLAN,
        ROLE_NEUTRAL_STAGE1_REFERENCE_ROW_MAP,
        (
            f"{ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_DIRECTORY}/"
            f"{ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_MANIFEST}"
        ),
    }.issubset(published_files)
    assert all(
        path in {
            ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST,
            ROLE_NEUTRAL_STAGE1_REFERENCE_LOCATOR,
            ROLE_NEUTRAL_STAGE1_REFERENCE_REGISTRY,
            ROLE_NEUTRAL_STAGE1_REFERENCE_PLAN,
            ROLE_NEUTRAL_STAGE1_REFERENCE_ROW_MAP,
        }
        or path.startswith(
            f"{ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_DIRECTORY}/"
        )
        for path in published_files
    )
    assert tuple(
        target.glob(
            f"{ROLE_NEUTRAL_STAGE1_HTR_AGGREGATION_DIRECTORY}/"
            "*/reverse_index/*.npy"
        )
    )
    assert not tuple(target.rglob("components"))
    scientific = json.loads(
        (target / ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST).read_text(encoding="utf-8")
    )
    assert str(target) not in json.dumps(scientific, sort_keys=True)
    assert str(execution_root) not in json.dumps(
        scientific,
        sort_keys=True,
    )
    assert scientific["physical_fit_count"] == len(plan.physical_scopes)
    assert scientific["logical_scope_count"] == len(plan.scopes)
    assert scientific["semantic_member_batch_size"] == 3
    assert (
        publication.stage2_provider.identity()[
            "semantic_member_batch_size"
        ]
        == 3
    )
    assert scientific["evidence_payloads_copied"] is False
    assert scientific["legacy_bundle_build_invoked"] is False
    assert scientific["independent_stage1_refit_invoked"] is False
    assert scientific["text_truncation_applied"] is False
    assert scientific["offline_handoff_validation_complete"] is True
    assert scientific["full_stage2_one_shot_runtime_complete"] is False
    locator = json.loads(
        (target / ROLE_NEUTRAL_STAGE1_REFERENCE_LOCATOR).read_text(encoding="utf-8")
    )
    assert locator["references_only"] is True
    assert locator["evidence_payloads_materialized_here"] is False
    assert locator["derived_htr_aggregate_payloads_materialized_here"] is True
    assert locator["raw_htr_token_arrays_materialized_here"] is False
    assert locator["role_neutral_execution"]["execution_tree_materialized_here"] is False


def test_reference_provider_exposes_plan_derived_outer_rows_and_binds_arbitrary_id_column(
    reference_handoff,
) -> None:
    plan, _execution_root, _manifest, _target, publication = reference_handoff
    provider = publication.stage2_provider
    assignments = provider.get_outer_fold_assignments()
    expected_scopes = tuple(
        scope for scope in plan.scopes if scope.scope_kind == "full_outer"
    )

    assert tuple(assignments) == tuple(scope.outer_fold for scope in expected_scopes)
    assert assignments == {
        scope.outer_fold: {
            "fit_row_ids": scope.fit_row_ids,
            "heldout_row_ids": scope.heldout_row_ids,
        }
        for scope in expected_scopes
    }
    row_count = len(expected_scopes[0].fit_row_ids) + len(
        expected_scopes[0].heldout_row_ids
    )
    prepared = pd.DataFrame(
        {
            "site_specific_record_locator": [
                f"patient-{row_id}" for row_id in range(row_count)
            ],
            "arbitrary_note_field": [f"note {row_id}" for row_id in range(row_count)],
        }
    )
    binding = provider.bind_prepared_row_map(
        prepared=prepared,
        unit_id_column="site_specific_record_locator",
    )
    assert binding["row_count"] == row_count
    assert binding["row_ids"] == tuple(range(row_count))
    assert binding["configured_unit_id_column"] == "site_specific_record_locator"
    assert binding["exact_unit_id_order_verified"] is True

    reordered = prepared.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="row order differ"):
        provider.bind_prepared_row_map(
            prepared=reordered,
            unit_id_column="site_specific_record_locator",
        )


def test_reference_provider_binds_every_physical_fit_to_complete_prepared_projection(
    reference_handoff,
) -> None:
    plan, _execution_root, manifest, _target, publication = reference_handoff
    provider = publication.stage2_provider
    outer = next(scope for scope in plan.scopes if scope.scope_kind == "full_outer")
    row_count = len(outer.fit_row_ids) + len(outer.heldout_row_ids)
    prepared = pd.DataFrame(
        {
            "encounter_locator": [
                f"patient-{row_id}" for row_id in range(row_count)
            ],
            "complete_narrative": [
                _fit_text(row_id) for row_id in range(row_count)
            ],
            "assigned_therapy": [
                _fit_treatment(row_id) for row_id in range(row_count)
            ],
            "binary_endpoint": [
                _fit_outcome(row_id) for row_id in range(row_count)
            ],
        }
    )
    binding = provider.bind_prepared_cohort_projection(
        prepared=prepared,
        prepared_cohort_artifact_sha256="d" * 64,
        unit_id_column="encounter_locator",
        text_column="complete_narrative",
        treatment_column="assigned_therapy",
        outcome_column="binary_endpoint",
    )
    assert type(binding) is AuthenticatedPreparedCohortProjectionBinding
    payload = validate_authenticated_prepared_projection_binding(
        binding,
        expected_plan_scientific_content_sha256=(
            plan.scientific_content_sha256
        ),
        expected_source_execution_content_sha256=manifest["content_sha256"],
    )
    assert payload["row_count"] == row_count
    assert len(payload["physical_owner_projection_proofs"]) == len(
        plan.physical_scopes
    )
    assert payload["all_physical_fit_projections_verified"] is True
    assert payload["text_truncation_applied"] is False

    changed = prepared.copy()
    changed.loc[0, "complete_narrative"] += " substituted"
    with pytest.raises(ValueError, match="differs from sealed physical fit"):
        provider.bind_prepared_cohort_projection(
            prepared=changed,
            prepared_cohort_artifact_sha256="e" * 64,
            unit_id_column="encounter_locator",
            text_column="complete_narrative",
            treatment_column="assigned_therapy",
            outcome_column="binary_endpoint",
        )


def test_reference_provider_runtime_token_precommits_plan_rows_without_rehashing(
    reference_handoff,
) -> None:
    plan, _execution_root, manifest, _target, publication = reference_handoff
    provider = publication.stage2_provider
    outer = next(scope for scope in plan.scopes if scope.scope_kind == "full_outer")
    row_count = len(outer.fit_row_ids) + len(outer.heldout_row_ids)
    prepared = pd.DataFrame(
        {
            "encounter_locator": [
                f"patient-{row_id}" for row_id in range(row_count)
            ],
            "complete_narrative": [
                _fit_text(row_id) for row_id in range(row_count)
            ],
            "assigned_therapy": [
                _fit_treatment(row_id) for row_id in range(row_count)
            ],
            "binary_endpoint": [
                _fit_outcome(row_id) for row_id in range(row_count)
            ],
        }
    )
    projection = provider.bind_prepared_cohort_projection(
        prepared=prepared,
        prepared_cohort_artifact_sha256="d" * 64,
        unit_id_column="encounter_locator",
        text_column="complete_narrative",
        treatment_column="assigned_therapy",
        outcome_column="binary_endpoint",
    )
    runtime = provider.issue_direct_runtime_binding(
        prepared_projection_binding=projection,
    )
    assert type(runtime) is AuthenticatedRoleNeutralStage2RuntimeBinding
    payload = validate_authenticated_role_neutral_stage2_runtime_binding(
        runtime,
        expected_plan_scientific_content_sha256=(
            plan.scientific_content_sha256
        ),
        expected_source_execution_content_sha256=manifest["content_sha256"],
    )
    assert payload["runner_dataset_artifact_sha256"] == "d" * 64
    assert payload["per_fold_text_treatment_outcome_rehash_required"] is False
    fold = payload["fold_bindings"][0]
    authorized = runtime.authorize_final_fold_shapes(
        outer_fold=fold["outer_fold"],
        exact_outer_train_row_ids=fold["outer_train_row_ids"],
        exact_outer_heldout_row_ids=fold["outer_heldout_row_ids"],
        exact_meta_inner_fold_ids=fold["meta_inner_fold_ids"],
        outer_train_text_count=fold["outer_train_row_count"],
        outer_train_treatment_count=fold["outer_train_row_count"],
        outer_train_outcome_count=fold["outer_train_row_count"],
        outer_heldout_text_count=fold["outer_heldout_row_count"],
        runner_dataset_artifact_sha256="d" * 64,
    )
    assert authorized["per_fold_text_treatment_outcome_rehashed"] is False
    with pytest.raises(ValueError, match="assignments differ"):
        runtime.authorize_final_fold_shapes(
            outer_fold=fold["outer_fold"],
            exact_outer_train_row_ids=tuple(
                reversed(fold["outer_train_row_ids"])
            ),
            exact_outer_heldout_row_ids=fold["outer_heldout_row_ids"],
            exact_meta_inner_fold_ids=fold["meta_inner_fold_ids"],
            outer_train_text_count=fold["outer_train_row_count"],
            outer_train_treatment_count=fold["outer_train_row_count"],
            outer_train_outcome_count=fold["outer_train_row_count"],
            outer_heldout_text_count=fold["outer_heldout_row_count"],
            runner_dataset_artifact_sha256="d" * 64,
        )


def test_reference_handoff_reopens_after_parent_relative_relocation(
    reference_handoff,
) -> None:
    _plan_value, _execution_root, manifest, target, _publication = reference_handoff
    relocated = target.with_name("relocated_reference_handoff")
    target.rename(relocated)
    try:
        reopened = load_reference_only_role_neutral_stage1_handoff(
            relocated / ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST,
        )
        assert reopened.source_role_neutral_execution_content_sha256 == (manifest["content_sha256"])
        assert reopened.stage2_provider is not None
    finally:
        relocated.rename(target)


@pytest.mark.parametrize(
    "relative_path",
    (
        ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST,
        ROLE_NEUTRAL_STAGE1_REFERENCE_REGISTRY,
        ROLE_NEUTRAL_STAGE1_REFERENCE_ROW_MAP,
    ),
)
def test_reference_handoff_rejects_tampered_registered_bytes(
    reference_handoff,
    relative_path: str,
) -> None:
    _plan_value, _execution_root, _manifest, target, _publication = reference_handoff
    path = target / relative_path
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(
            (ValueError, RuntimeError),
            match="registration|invalid|changed|identity",
        ):
            load_reference_only_role_neutral_stage1_handoff(
                target / ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST,
            )
    finally:
        path.write_bytes(original)


def test_reference_handoff_fresh_process_loader(
    reference_handoff,
    tmp_path: Path,
) -> None:
    _plan_value, _execution_root, manifest, target, _publication = reference_handoff
    output = tmp_path / "fresh_loader.json"
    script = """
import json
import sys
from pathlib import Path
from oci.inference.production_role_neutral_stage2_handoff import (
    load_reference_only_role_neutral_stage1_handoff,
)
publication = load_reference_only_role_neutral_stage1_handoff(Path(sys.argv[1]))
Path(sys.argv[2]).write_text(
    json.dumps(publication.as_dict(), sort_keys=True),
    encoding="utf-8",
)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(target / ROLE_NEUTRAL_STAGE1_REFERENCE_MANIFEST),
            str(output),
        ],
        check=True,
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["handoff_kind"] == (ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND)
    assert (
        value["stage1_inputs"]["source_role_neutral_execution_content_sha256"]
        == manifest["content_sha256"]
    )
    assert value["offline_handoff_validation_complete"] is True
    assert value["full_stage2_one_shot_runtime_complete"] is False
